//! MNIST small-CNN trainer guest for the `wasmer_gpu_v0` bridge.
//!
//! Architecture: conv 5x5 (8 filters, valid) -> ReLU -> max-pool 2x2 ->
//! fully connected 1152 -> 10 -> softmax. Six guest-owned WGSL kernels per
//! training step (conv forward, pool, fc forward, fc backward/routing, fc
//! update, conv update); three for evaluation.
//!
//! The guest owns the model: it initialises the parameters from the seed,
//! uploads them once into a GPU-resident state buffer, and drives every
//! minibatch step by writing the batch into that buffer and dispatching its
//! own six WGSL kernels (three of them for evaluation). The host only feeds
//! raw MNIST bytes into guest memory before `_start` and receives read-back
//! numbers for metrics and independent checks through `mnist_demo_v0`.
//!
//! State buffer layout is in `layout.wgsl`; read-back regions come first.
#![no_std]

use core::panic::PanicInfo;

#[link(wasm_import_module = "wasmer_gpu_v0")]
extern "C" {
    // Pointer-typed parameters (i32 on wasm32) let the optimizer see that the
    // host reads and writes through them.
    fn buffer_upload(ptr: *const f32, len: i32) -> i32;
    fn buffer_read(handle: i32, ptr: *mut f32, len: i32) -> i32;
    fn buffer_write(handle: i32, offset: i32, ptr: *const f32, len: i32) -> i32;
    fn buffer_release(handle: i32) -> i32;
    fn pipeline_create(ptr: *const u8, len: i32) -> i32;
    fn pipeline_dispatch(pipeline: i32, buffer: i32, x: i32, y: i32, z: i32) -> i32;
    fn pipeline_release(handle: i32) -> i32;
}

#[link(wasm_import_module = "mnist_demo_v0")]
extern "C" {
    /// Fills a [`Config`] at `out_ptr`; returns 0 on success.
    fn configure(out_ptr: *mut Config) -> i32;
    /// `tag`: 0 initial parameters, 1 after the first update, 2 final.
    fn report_weights(tag: i32, ptr: *const f32, count: i32);
    fn report_progress(step: i32, samples_seen: i32, mean_batch_loss: f32);
    fn report_eval_probs(first_index: i32, ptr: *const f32, count: i32);
    fn report_eval_done(step: i32);
}

#[link(wasm_import_module = "wasi_snapshot_preview1")]
extern "C" {
    fn proc_exit(code: i32) -> !;
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Config {
    pub train_ptr: u32,
    pub train_n: u32,
    pub train_labels_ptr: u32,
    pub test_ptr: u32,
    pub test_n: u32,
    pub test_labels_ptr: u32,
    pub steps: u32,
    pub batch: u32,
    pub eval_every: u32,
    pub learning_rate: f32,
    pub seed: u32,
    pub progress_every: u32,
}

const INPUTS: usize = 784;
const CLASSES: usize = 10;
const MAX_BATCH: usize = 64;
const FILTERS: usize = 8;
const K: usize = 5;
const CONV: usize = 24;
const POOL: usize = 12;
const FEAT: usize = FILTERS * POOL * POOL;
/// W2, b2, W1, b1 in buffer order.
const PARAMS: usize = CLASSES * FEAT + CLASSES + FILTERS * K * K + FILTERS;
const OFF_L: usize = 16;
const OFF_P: usize = OFF_L + MAX_BATCH;
const OFF_W2: usize = OFF_P + MAX_BATCH * CLASSES;
const OFF_B2: usize = OFF_W2 + CLASSES * FEAT;
const OFF_W1: usize = OFF_B2 + CLASSES;
const OFF_B1: usize = OFF_W1 + FILTERS * K * K;
const OFF_X: usize = OFF_B1 + FILTERS;
const OFF_Y: usize = OFF_X + MAX_BATCH * INPUTS;
const OFF_CONV: usize = OFF_Y + MAX_BATCH;
const OFF_POOLED: usize = OFF_CONV + MAX_BATCH * FILTERS * CONV * CONV;
const OFF_ARGMAX: usize = OFF_POOLED + MAX_BATCH * FEAT;
const OFF_DPOOLED: usize = OFF_ARGMAX + MAX_BATCH * FEAT;
const OFF_DCONV: usize = OFF_DPOOLED + MAX_BATCH * FEAT;
const TOTAL: usize = OFF_DCONV + MAX_BATCH * FILTERS * CONV * CONV;
const FRONT: usize = OFF_B1 + FILTERS;

const LAYOUT: &str = include_str!("layout.wgsl");
const CONV_FORWARD: &str = concat!(include_str!("layout.wgsl"), include_str!("conv_forward.wgsl"));
const POOL_FORWARD: &str = concat!(include_str!("layout.wgsl"), include_str!("pool_forward.wgsl"));
const FC_FORWARD: &str = concat!(include_str!("layout.wgsl"), include_str!("fc_forward.wgsl"));
const FC_BACKWARD: &str = concat!(include_str!("layout.wgsl"), include_str!("fc_backward.wgsl"));
const FC_UPDATE: &str = concat!(include_str!("layout.wgsl"), include_str!("fc_update.wgsl"));
const CONV_UPDATE: &str = concat!(include_str!("layout.wgsl"), include_str!("conv_update.wgsl"));

/// Whole state as staged in guest memory. Uploaded once; afterwards only the
/// header, X and y regions are rewritten per step, and the leading regions are
/// read back for metrics.
static mut STATE: [f32; TOTAL] = [0.0; TOTAL];
static mut READBACK: [f32; FRONT] = [0.0; FRONT];

// Exit codes: 10 + stage, so a bridge error is attributable in the host log.
const EXIT_CONFIG: i32 = 11;
const EXIT_UPLOAD: i32 = 12;
const EXIT_PIPELINE: i32 = 13;
const EXIT_WRITE: i32 = 14;
const EXIT_DISPATCH: i32 = 15;
const EXIT_READ: i32 = 16;
const EXIT_RELEASE: i32 = 17;

struct XorShift32(u32);

impl XorShift32 {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    /// Uniform in [-scale, scale].
    fn symmetric(&mut self, scale: f32) -> f32 {
        (self.next() as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
    }
}


unsafe fn fail(code: i32) -> ! {
    proc_exit(code)
}

unsafe fn check(status: i32, code: i32) {
    if status < 0 {
        fail(code);
    }
}

/// Copies `n` images starting at `first` from a u8 dataset into the X/y staging
/// region as f32 in [0, 1], and writes the batch size into the header.
unsafe fn stage_batch(state: &mut [f32; TOTAL], images: *const u8, labels: *const u8, first: usize, n: usize) {
    for s in 0..n {
        let src = images.add((first + s) * INPUTS);
        for i in 0..INPUTS {
            state[OFF_X + s * INPUTS + i] = *src.add(i) as f32 / 255.0;
        }
        state[OFF_Y + s] = *labels.add(first + s) as f32;
    }
    state[0] = n as f32;
}

unsafe fn push_batch(state: &[f32; TOTAL], handle: i32, n: usize) {
    let bytes = |elements: usize| (elements * 4) as i32;
    check(buffer_write(handle, 0, state.as_ptr(), bytes(16)), EXIT_WRITE);
    check(buffer_write(handle, bytes(OFF_X), state.as_ptr().add(OFF_X), bytes(n * INPUTS)), EXIT_WRITE);
    check(buffer_write(handle, bytes(OFF_Y), state.as_ptr().add(OFF_Y), bytes(n)), EXIT_WRITE);
}

unsafe fn read_front(handle: i32, readback: &mut [f32; FRONT], elements: usize) {
    check(buffer_read(handle, readback.as_mut_ptr(), (elements * 4) as i32), EXIT_READ);
}

/// conv+ReLU, pool, fc+softmax for the staged batch.
unsafe fn forward_pass(p: &[i32; 3], handle: i32, n: usize) {
    check(pipeline_dispatch(p[0], handle, (n * FILTERS * CONV * CONV).div_ceil(64) as i32, 1, 1), EXIT_DISPATCH);
    check(pipeline_dispatch(p[1], handle, (n * FEAT).div_ceil(64) as i32, 1, 1), EXIT_DISPATCH);
    check(pipeline_dispatch(p[2], handle, n as i32, 1, 1), EXIT_DISPATCH);
}

#[no_mangle]
pub extern "C" fn _start() {
    unsafe { run() }
}

unsafe fn run() -> ! {
    let mut cfg = Config {
        train_ptr: 0, train_n: 0, train_labels_ptr: 0, test_ptr: 0, test_n: 0, test_labels_ptr: 0,
        steps: 0, batch: 0, eval_every: 0, learning_rate: 0.0, seed: 0, progress_every: 0,
    };
    check(configure(&mut cfg), EXIT_CONFIG);
    let cfg = core::ptr::read_volatile(&cfg);
    let batch = (cfg.batch as usize).clamp(1, MAX_BATCH);
    if cfg.train_n == 0 || cfg.test_n == 0 || cfg.steps == 0 {
        fail(EXIT_CONFIG);
    }
    let state = &mut *core::ptr::addr_of_mut!(STATE);
    let readback = &mut *core::ptr::addr_of_mut!(READBACK);

    // Seeded initialisation: uniform weights scaled by fan-in, zero biases.
    let mut rng = XorShift32(cfg.seed | 1);
    for k in 0..CLASSES * FEAT {
        state[OFF_W2 + k] = rng.symmetric(0.05);
    }
    for k in 0..FILTERS * K * K {
        state[OFF_W1 + k] = rng.symmetric(0.2);
    }
    state[0] = batch as f32;
    state[1] = cfg.learning_rate;
    report_weights(0, state.as_ptr().add(OFF_W2), PARAMS as i32);

    let handle = buffer_upload(state.as_ptr(), (TOTAL * 4) as i32);
    check(handle, EXIT_UPLOAD);
    let _ = LAYOUT;
    let mut pipelines = [0i32; 6];
    for (slot, source) in pipelines.iter_mut().zip([CONV_FORWARD, POOL_FORWARD, FC_FORWARD, FC_BACKWARD, FC_UPDATE, CONV_UPDATE]) {
        *slot = pipeline_create(source.as_ptr(), source.len() as i32);
        check(*slot, EXIT_PIPELINE);
    }
    let [conv_forward, pool_forward, fc_forward, fc_backward, fc_update, conv_update] = pipelines;

    let train = cfg.train_ptr as *const u8;
    let train_labels = cfg.train_labels_ptr as *const u8;
    let test = cfg.test_ptr as *const u8;
    let test_labels = cfg.test_labels_ptr as *const u8;
    let train_n = cfg.train_n as usize;
    let test_n = cfg.test_n as usize;
    let mut samples_seen: usize = 0;

    for step in 0..cfg.steps as usize {
        let first = (step * batch) % train_n;
        let n = batch.min(train_n - first);
        stage_batch(state, train, train_labels, first, n);
        push_batch(state, handle, n);
        forward_pass(&[conv_forward, pool_forward, fc_forward], handle, n);
        check(pipeline_dispatch(fc_backward, handle, (n * FEAT).div_ceil(64) as i32, 1, 1), EXIT_DISPATCH);
        check(pipeline_dispatch(fc_update, handle, (CLASSES * FEAT + CLASSES).div_ceil(64) as i32, 1, 1), EXIT_DISPATCH);
        check(pipeline_dispatch(conv_update, handle, (FILTERS * K * K + FILTERS).div_ceil(8) as i32, 1, 1), EXIT_DISPATCH);
        samples_seen += n;

        let last = step + 1 == cfg.steps as usize;
        let progress_due = cfg.progress_every != 0 && (step + 1) % cfg.progress_every as usize == 0;
        if step == 0 || progress_due || last {
            // Loss was computed by the forward pass before this step's update.
            read_front(handle, readback, FRONT);
            let mut total = 0.0f32;
            for s in 0..n {
                total += readback[OFF_L + s];
            }
            report_progress(step as i32, samples_seen as i32, total / n as f32);
            if step == 0 {
                report_weights(1, readback.as_ptr().add(OFF_W2), PARAMS as i32);
            }
        }
        let eval_due = cfg.eval_every != 0 && (step + 1) % cfg.eval_every as usize == 0;
        if eval_due || last {
            let mut first_index = 0usize;
            while first_index < test_n {
                let n = batch.min(test_n - first_index);
                stage_batch(state, test, test_labels, first_index, n);
                push_batch(state, handle, n);
                forward_pass(&[conv_forward, pool_forward, fc_forward], handle, n);
                read_front(handle, readback, OFF_P + MAX_BATCH * CLASSES);
                report_eval_probs(first_index as i32, readback.as_ptr().add(OFF_P), (n * CLASSES) as i32);
                first_index += n;
            }
            report_eval_done(step as i32);
        }
    }

    read_front(handle, readback, FRONT);
    report_weights(2, readback.as_ptr().add(OFF_W2), PARAMS as i32);
    for pipeline in pipelines {
        check(pipeline_release(pipeline), EXIT_RELEASE);
    }
    check(buffer_release(handle), EXIT_RELEASE);
    proc_exit(0)
}

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    core::arch::wasm32::unreachable()
}
