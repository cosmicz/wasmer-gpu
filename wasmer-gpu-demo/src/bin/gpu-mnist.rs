//! `gpu-mnist`: trains a 784-input / 10-output softmax regression (multinomial
//! logistic regression) on a bounded MNIST subset with minibatch SGD, where
//! every forward pass and parameter update is a guest-owned WGSL dispatch
//! through the `wasmer_gpu_v0` bridge registered on a WASIX runtime hook.
//!
//! The host's jobs: load and validate the IDX files, feed raw bytes into guest
//! memory, receive read-back numbers, compute metrics, and check the first
//! update against an independent CPU reference. It never updates parameters.
//!
//! Output: one JSON object per line on stdout (`start`, `progress`, `eval`,
//! `check`, `done`, or `error`), also appended to a run log under `data/`.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
    time::Instant,
};
use wasmer::{imports, Function, FunctionEnv, FunctionEnvMut, Memory, Module, Pages, Store};
use wasmer_gpu_demo::bridge::{self, GpuBridge, SessionLimits};
use wasmer_gpu_demo::hook::GpuInstantiationHook;
use wasmer_wasix::{
    runtime::{task_manager::tokio::TokioTaskManager, PluggableRuntime},
    WasiEnvBuilder, WasiError,
};

const INPUTS: usize = 784;
const CLASSES: usize = 10;
const SOFTMAX_PARAMS: usize = INPUTS * CLASSES + CLASSES;
/// conv 5x5 x8 (200 + 8) and fc 1152->10 (11,520 + 10).
const CNN_PARAMS: usize = 10 * 1152 + 10 + 8 * 25 + 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ModelKind {
    Cnn,
    Softmax,
}

impl ModelKind {
    fn name(self) -> &'static str {
        match self {
            ModelKind::Cnn => "small CNN: conv 5x5 x8 + ReLU -> max-pool 2x2 -> fc 1152->10 softmax (11,738 parameters), minibatch SGD",
            ModelKind::Softmax => "softmax regression 784->10 (multinomial logistic regression), minibatch SGD",
        }
    }
    fn params(self) -> usize {
        match self {
            ModelKind::Cnn => CNN_PARAMS,
            ModelKind::Softmax => SOFTMAX_PARAMS,
        }
    }
    fn guest(self) -> &'static str {
        match self {
            ModelKind::Cnn => "guest/mnist-cnn/mnist_cnn_guest.wasm",
            ModelKind::Softmax => "guest/mnist/mnist_guest.wasm",
        }
    }
    fn default_learning_rate(self) -> f32 {
        match self {
            ModelKind::Cnn => 0.1,
            ModelKind::Softmax => 0.5,
        }
    }
    fn weights_layout(self) -> &'static str {
        match self {
            ModelKind::Cnn => "cnn: W2 10x1152 row-major, b2 10, W1 8x5x5, b1 8, f32",
            ModelKind::Softmax => "softmax: 10x784 weights row-major then 10 biases, f32",
        }
    }
    /// Documented per-run session budget: the CNN state buffer is about
    /// 3.5 MB (activations and gradients for a batch of 64), so its session
    /// gets an 8 MiB byte budget; everything else keeps the bridge defaults.
    fn session_limits(self) -> SessionLimits {
        match self {
            ModelKind::Cnn => SessionLimits {
                max_bytes: 8 * 1024 * 1024,
                max_pipelines: 8,
                ..SessionLimits::default()
            },
            ModelKind::Softmax => SessionLimits::default(),
        }
    }
}
const DATASET_SOURCE: &str =
    "https://github.com/cvdfoundation/mnist -> https://storage.googleapis.com/cvdf-datasets/mnist/";
/// Max abs difference allowed between the GPU's first update and the f64 CPU
/// reference applied to the same initial weights and minibatch.
const FIRST_UPDATE_TOLERANCE: f32 = 1e-4;
/// Held-out digits whose prediction confidences are reported at every
/// evaluation so a viewer can watch predictions change as training proceeds.
const GALLERY: usize = 24;

#[derive(Clone, Debug)]
struct Options {
    model: ModelKind,
    steps: u32,
    batch: u32,
    learning_rate: f32,
    eval_every: u32,
    progress_every: u32,
    train_samples: usize,
    test_samples: usize,
    seed: u32,
    data_dir: PathBuf,
    guest: PathBuf,
    log_dir: PathBuf,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GuestConfig {
    train_ptr: u32,
    train_n: u32,
    train_labels_ptr: u32,
    test_ptr: u32,
    test_n: u32,
    test_labels_ptr: u32,
    steps: u32,
    batch: u32,
    eval_every: u32,
    learning_rate: f32,
    seed: u32,
    progress_every: u32,
}

struct Dataset {
    train_images: Vec<u8>,
    train_labels: Vec<u8>,
    test_images: Vec<u8>,
    test_labels: Vec<u8>,
    digests: Vec<(String, String)>,
}

/// Shared between the host imports and the runner.
struct HostState {
    params: usize,
    memory: Option<Memory>,
    config: GuestConfig,
    test_labels: Vec<u8>,
    weights: [Option<Vec<f32>>; 3],
    eval_correct: usize,
    eval_total: usize,
    gallery: Vec<serde_json::Value>,
    last_loss: Option<f32>,
    last_accuracy: Option<f32>,
    first_loss: Option<f32>,
    started: Instant,
    log: Log,
}

struct Log {
    file: Option<fs::File>,
}

impl Log {
    fn emit<T: Serialize>(&mut self, value: &T) {
        let line = serde_json::to_string(value).expect("serializable event");
        println!("{line}");
        if let Some(file) = &mut self.file {
            let _ = writeln!(file, "{line}");
        }
    }
}

type Shared = Arc<Mutex<HostState>>;

fn main() {
    if let Err(error) = run() {
        let event = serde_json::json!({"event": "error", "message": format!("{error:#}")});
        println!("{event}");
        std::process::exit(1);
    }
}

fn parse_options() -> Result<Options> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut options = Options {
        model: ModelKind::Cnn,
        steps: 600,
        batch: 64,
        learning_rate: f32::NAN,
        eval_every: 100,
        progress_every: 10,
        train_samples: 12_000,
        test_samples: 2_000,
        seed: 42,
        data_dir: manifest.join("data/mnist"),
        guest: PathBuf::new(),
        log_dir: manifest.join("data/mnist/runs"),
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--model" => {
                options.model = match value()?.as_str() {
                    "cnn" => ModelKind::Cnn,
                    "softmax" => ModelKind::Softmax,
                    other => bail!("--model must be cnn or softmax, not {other}"),
                }
            }
            "--steps" => options.steps = value()?.parse()?,
            "--batch" => options.batch = value()?.parse()?,
            "--lr" => options.learning_rate = value()?.parse()?,
            "--eval-every" => options.eval_every = value()?.parse()?,
            "--progress-every" => options.progress_every = value()?.parse()?,
            "--train" => options.train_samples = value()?.parse()?,
            "--test" => options.test_samples = value()?.parse()?,
            "--seed" => options.seed = value()?.parse()?,
            "--data-dir" => options.data_dir = PathBuf::from(value()?),
            "--guest" => options.guest = PathBuf::from(value()?),
            "--log-dir" => options.log_dir = PathBuf::from(value()?),
            other => bail!("unknown flag {other}"),
        }
    }
    if options.learning_rate.is_nan() {
        options.learning_rate = options.model.default_learning_rate();
    }
    if options.guest.as_os_str().is_empty() {
        options.guest = manifest.join(options.model.guest());
    }
    if !(1..=64).contains(&options.batch) {
        bail!("--batch must be 1..=64 (guest staging size)");
    }
    if options.steps == 0 || options.steps > 100_000 {
        bail!("--steps must be 1..=100000");
    }
    if !options.learning_rate.is_finite() || options.learning_rate < 0.0 {
        bail!("--lr must be finite and non-negative");
    }
    Ok(options)
}

fn run() -> Result<()> {
    let options = parse_options()?;
    let dataset = load_dataset(&options)?;
    fs::create_dir_all(&options.log_dir).context("create run log dir")?;
    let log_path = options.log_dir.join(format!(
        "run-{}.jsonl",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    ));
    let log = Log {
        file: Some(fs::File::create(&log_path).context("create run log")?),
    };

    let setup_started = Instant::now();
    let gpu = GpuBridge::new().context("initialize wgpu bridge")?;
    let snapshot = gpu.snapshot();
    let setup_us = setup_started.elapsed().as_micros();

    let config = GuestConfig {
        train_n: dataset.train_labels.len() as u32,
        test_n: dataset.test_labels.len() as u32,
        steps: options.steps,
        batch: options.batch,
        eval_every: options.eval_every,
        learning_rate: options.learning_rate,
        seed: options.seed,
        progress_every: options.progress_every,
        ..GuestConfig::default()
    };
    let shared: Shared = Arc::new(Mutex::new(HostState {
        params: options.model.params(),
        memory: None,
        config,
        test_labels: dataset.test_labels.clone(),
        weights: [None, None, None],
        eval_correct: 0,
        eval_total: 0,
        gallery: Vec::new(),
        last_loss: None,
        last_accuracy: None,
        first_loss: None,
        started: Instant::now(),
        log,
    }));
    shared.lock().unwrap().log.emit(&serde_json::json!({
        "event": "start",
        "model": options.model.name(),
        "model_kind": match options.model { ModelKind::Cnn => "cnn", ModelKind::Softmax => "softmax" },
        "parameters": options.model.params(),
        "execution": "Wasmer WASIX guest (wasm32 core module via WasiEnvBuilder) -> wasmer_gpu_v0 InstantiationHook -> guest WGSL -> wgpu",
        "adapter": snapshot.adapter_name,
        "backend": snapshot.backend,
        "setup_us": setup_us,
        "seed": options.seed,
        "steps": options.steps,
        "batch": options.batch,
        "learning_rate": options.learning_rate,
        "eval_every": options.eval_every,
        "train_samples": dataset.train_labels.len(),
        "test_samples": dataset.test_labels.len(),
        "train_split": format!("first {} of the official 60,000 training images", dataset.train_labels.len()),
        "test_split": format!("first {} of the official 10,000 test images (never trained on)", dataset.test_labels.len()),
        "dataset_source": DATASET_SOURCE,
        "dataset_sha256": dataset.digests.iter().map(|(n, d)| serde_json::json!({"file": n, "sha256": d})).collect::<Vec<_>>(),
        "guest": options.guest.display().to_string(),
        "log_path": log_path.display().to_string(),
    }));

    // --- instantiate the guest through the WASIX runtime with both hooks ---
    let tokio_runtime = tokio_runtime()?;
    let _entered = tokio_runtime.enter();
    let mut runtime = PluggableRuntime::new(Arc::new(TokioTaskManager::new(
        tokio_runtime.handle().clone(),
    )));
    let mut store = Store::default();
    runtime.set_engine(store.engine().clone());
    // Documented session budget for this trainer: the single state buffer is
    // 235,240 bytes and two pipelines; defaults suffice and are kept.
    runtime.with_instantiation_hook(GpuInstantiationHook::new(
        gpu.clone(),
        options.model.session_limits(),
    ));
    let host_shared = shared.clone();
    runtime.with_additional_imports(move |_module, store| {
        let env = FunctionEnv::new(store, host_shared.clone());
        Ok(imports! {
            "mnist_demo_v0" => {
                "configure" => Function::new_typed_with_env(store, &env, configure),
                "report_weights" => Function::new_typed_with_env(store, &env, report_weights),
                "report_progress" => Function::new_typed_with_env(store, &env, report_progress),
                "report_eval_probs" => Function::new_typed_with_env(store, &env, report_eval_probs),
                "report_eval_done" => Function::new_typed_with_env(store, &env, report_eval_done),
            }
        })
    });
    let wasm = fs::read(&options.guest)
        .with_context(|| format!("read guest {}", options.guest.display()))?;
    let module = Module::new(&store, wasm).context("compile MNIST guest module")?;
    let (instance, wasi_env) = WasiEnvBuilder::new("gpu-mnist-guest")
        .runtime(Arc::new(runtime))
        .instantiate(module, &mut store)
        .context("instantiate MNIST guest through WASIX")?;
    let memory = instance
        .exports
        .get_memory("memory")
        .context("guest exports no memory")?
        .clone();

    // --- feed the dataset into guest memory beyond the guest's own pages ---
    let base = memory.view(&store).data_size();
    let needed = dataset.train_images.len()
        + dataset.train_labels.len()
        + dataset.test_images.len()
        + dataset.test_labels.len();
    let pages = needed.div_ceil(65_536) as u32 + 1;
    memory
        .grow(&mut store, Pages(pages))
        .context("grow guest memory for the dataset")?;
    let mut cursor = base;
    let mut place = |bytes: &[u8]| -> Result<u32> {
        bridge::write_bytes(&memory, &store, cursor, bytes)
            .map_err(|code| anyhow::anyhow!("write dataset into guest memory: {code}"))?;
        let at = cursor as u32;
        cursor += bytes.len() as u64;
        Ok(at)
    };
    let train_ptr = place(&dataset.train_images)?;
    let train_labels_ptr = place(&dataset.train_labels)?;
    let test_ptr = place(&dataset.test_images)?;
    let test_labels_ptr = place(&dataset.test_labels)?;
    {
        let mut state = shared.lock().unwrap();
        state.config.train_ptr = train_ptr;
        state.config.train_labels_ptr = train_labels_ptr;
        state.config.test_ptr = test_ptr;
        state.config.test_labels_ptr = test_labels_ptr;
        state.memory = Some(memory.clone());
        state.started = Instant::now();
    }

    // --- run: the guest drives every step ---
    let start = instance
        .exports
        .get_function("_start")
        .context("guest exports no _start")?;
    let exit = match start.call(&mut store, &[]) {
        Ok(_) => 0,
        Err(error) => match error.downcast_ref::<WasiError>() {
            Some(WasiError::Exit(code)) => i32::from(*code),
            _ => bail!("guest trapped: {error}"),
        },
    };
    wasi_env.on_exit(&mut store, None);
    let runtime_ms = shared.lock().unwrap().started.elapsed().as_millis();
    drop(store);
    let after = gpu.snapshot();

    if exit != 0 {
        bail!("guest exited with status {exit} (see guest/mnist/src/lib.rs EXIT_* for the failing stage)");
    }

    // --- independent CPU check of the first update and the parameter change ---
    let mut state = shared.lock().unwrap();
    let (w0, w1, w2) = match &state.weights {
        [Some(a), Some(b), Some(c)] => (a.clone(), b.clone(), c.clone()),
        _ => bail!("guest did not report initial, first-update and final parameters"),
    };
    let changed = max_abs_diff(&w0, &w2);
    let changed_ok = if options.learning_rate == 0.0 {
        changed == 0.0
    } else {
        changed > 0.0
    };
    let finite = w2.iter().all(|v| v.is_finite()) && state.last_loss.is_some_and(f32::is_finite);
    let check_passed = match options.model {
        ModelKind::Softmax => {
            // Softmax keeps the exact f64 parity check of the first update.
            let first_n = (options.batch as usize).min(dataset.train_labels.len());
            let reference = cpu_sgd_step(
                &w0,
                &dataset.train_images[..first_n * INPUTS],
                &dataset.train_labels[..first_n],
                options.learning_rate as f64,
            );
            let first_update_diff = max_abs_diff(&reference, &w1);
            let passed = first_update_diff <= FIRST_UPDATE_TOLERANCE && changed_ok && finite;
            state.log.emit(&serde_json::json!({
                "event": "check", "kind": "cpu_parity_first_update",
                "first_update_max_abs_diff": first_update_diff, "tolerance": FIRST_UPDATE_TOLERANCE,
                "parameters_compared": SOFTMAX_PARAMS, "final_vs_initial_max_abs_diff": changed,
                "learning_rate": options.learning_rate,
                "expectation": if options.learning_rate == 0.0 { "parameters unchanged" } else { "parameters changed" },
                "passed": passed,
            }));
            passed
        }
        ModelKind::Cnn => {
            // CNN: learning signals only (no device-parity harness by direction).
            let first_step_changed = max_abs_diff(&w0, &w1) > 0.0 || options.learning_rate == 0.0;
            let passed = changed_ok && finite && first_step_changed;
            state.log.emit(&serde_json::json!({
                "event": "check", "kind": "learning_signals",
                "parameters": CNN_PARAMS, "first_step_max_abs_change": max_abs_diff(&w0, &w1),
                "final_vs_initial_max_abs_diff": changed, "all_finite": finite,
                "learning_rate": options.learning_rate,
                "expectation": if options.learning_rate == 0.0 { "parameters unchanged" } else { "parameters changed, finite" },
                "passed": passed,
            }));
            passed
        }
    };
    let (first_loss, final_loss, final_accuracy) =
        (state.first_loss, state.last_loss, state.last_accuracy);
    state.log.emit(&serde_json::json!({
        "event": "done",
        "exit_status": exit,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "final_test_accuracy": final_accuracy,
        "runtime_ms": runtime_ms,
        "live_buffers_after": after.live_buffers,
        "live_pipelines_after": after.live_pipelines,
        "log_path": log_path.display().to_string(),
        "checks_passed": check_passed && after.live_buffers == 0 && after.live_pipelines == 0,
    }));
    if check_passed {
        // Final parameters, for browser-side inference on user-drawn digits.
        // Labelled as such by consumers: this is not the GPU path.
        state.log.emit(&serde_json::json!({
            "event": "weights",
            "layout": options.model.weights_layout(),
            "trained_by": "GPU dispatches above; consumers using these on the CPU must say so",
            "values": w2,
        }));
    }
    if !check_passed {
        bail!("independent CPU check failed");
    }
    Ok(())
}

// --- host imports for the guest -------------------------------------------

fn configure(env: FunctionEnvMut<Shared>, out_ptr: i32) -> i32 {
    let state = env.data().lock().unwrap();
    let Some(memory) = state.memory.clone() else {
        return -1;
    };
    match bridge::write_bytes(
        &memory,
        &env,
        out_ptr as u32 as u64,
        bytemuck::bytes_of(&state.config),
    ) {
        Ok(()) => 0,
        Err(code) => code,
    }
}

fn read_f32s(env: &FunctionEnvMut<Shared>, ptr: i32, count: i32) -> Option<Vec<f32>> {
    let memory = env.data().lock().unwrap().memory.clone()?;
    let count = usize::try_from(count).ok()?;
    let bytes = bridge::read_bytes(&memory, env, ptr as u32 as u64, count * 4).ok()?;
    Some(bytemuck::cast_slice(&bytes).to_vec())
}

fn report_weights(env: FunctionEnvMut<Shared>, tag: i32, ptr: i32, count: i32) {
    if count as usize != env.data().lock().unwrap().params {
        return;
    }
    let values = read_f32s(&env, ptr, count);
    if let (Some(values), Ok(index)) = (values, usize::try_from(tag)) {
        if index < 3 {
            env.data().lock().unwrap().weights[index] = Some(values);
        }
    }
}

fn report_progress(env: FunctionEnvMut<Shared>, step: i32, samples_seen: i32, loss: f32) {
    let mut state = env.data().lock().unwrap();
    if state.first_loss.is_none() {
        state.first_loss = Some(loss);
    }
    state.last_loss = Some(loss);
    let elapsed_ms = state.started.elapsed().as_millis();
    let lr = state.config.learning_rate;
    state.log.emit(&serde_json::json!({
        "event": "progress", "step": step, "samples_seen": samples_seen,
        "loss": loss, "loss_finite": loss.is_finite(), "learning_rate": lr, "elapsed_ms": elapsed_ms,
    }));
}

fn report_eval_probs(env: FunctionEnvMut<Shared>, first_index: i32, ptr: i32, count: i32) {
    let Some(probs) = read_f32s(&env, ptr, count) else {
        return;
    };
    let mut state = env.data().lock().unwrap();
    let first = first_index as usize;
    for (offset, row) in probs.chunks_exact(CLASSES).enumerate() {
        let Some(&label) = state.test_labels.get(first + offset) else {
            break;
        };
        let predicted = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(j, _)| j)
            .unwrap_or(usize::MAX);
        state.eval_total += 1;
        if predicted == label as usize {
            state.eval_correct += 1;
        }
        if first + offset < GALLERY {
            state.gallery.push(serde_json::json!({
                "index": first + offset, "label": label, "predicted": predicted,
                "probs": row.iter().map(|p| (p * 10_000.0).round() / 10_000.0).collect::<Vec<f32>>(),
            }));
        }
    }
}

fn report_eval_done(env: FunctionEnvMut<Shared>, step: i32) {
    let mut state = env.data().lock().unwrap();
    let accuracy = if state.eval_total == 0 {
        0.0
    } else {
        state.eval_correct as f32 / state.eval_total as f32
    };
    state.last_accuracy = Some(accuracy);
    let (correct, total) = (state.eval_correct, state.eval_total);
    let elapsed_ms = state.started.elapsed().as_millis();
    state.log.emit(&serde_json::json!({
        "event": "eval", "step": step, "test_accuracy": accuracy,
        "correct": correct, "total": total, "elapsed_ms": elapsed_ms,
    }));
    let gallery = std::mem::take(&mut state.gallery);
    state.log.emit(&serde_json::json!({
        "event": "predictions", "step": step, "source": "GPU forward pass on held-out test digits",
        "items": gallery,
    }));
    state.eval_correct = 0;
    state.eval_total = 0;
}

// --- independent CPU reference ---------------------------------------------

/// One minibatch SGD step in f64 from `w0` (10x784 weights then 10 biases).
fn cpu_sgd_step(w0: &[f32], images: &[u8], labels: &[u8], lr: f64) -> Vec<f32> {
    let n = labels.len();
    let mut grad = vec![0.0f64; SOFTMAX_PARAMS];
    for s in 0..n {
        let x: Vec<f64> = images[s * INPUTS..(s + 1) * INPUTS]
            .iter()
            .map(|&v| v as f64 / 255.0)
            .collect();
        let mut z = [0.0f64; CLASSES];
        for (j, zj) in z.iter_mut().enumerate() {
            let mut acc = w0[INPUTS * CLASSES + j] as f64;
            for i in 0..INPUTS {
                acc += w0[j * INPUTS + i] as f64 * x[i];
            }
            *zj = acc;
        }
        let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = z.iter().map(|v| (v - zmax).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..CLASSES {
            let p = exps[j] / sum;
            let t = if labels[s] as usize == j { 1.0 } else { 0.0 };
            let d = p - t;
            for i in 0..INPUTS {
                grad[j * INPUTS + i] += d * x[i];
            }
            grad[INPUTS * CLASSES + j] += d;
        }
    }
    w0.iter()
        .zip(grad.iter())
        .map(|(&w, &g)| (w as f64 - lr * g / n as f64) as f32)
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// --- dataset ----------------------------------------------------------------

fn load_dataset(options: &Options) -> Result<Dataset> {
    let (train_images, d1) =
        read_idx_images(&options.data_dir.join("train-images-idx3-ubyte"), 60_000)?;
    let (train_labels, d2) =
        read_idx_labels(&options.data_dir.join("train-labels-idx1-ubyte"), 60_000)?;
    let (test_images, d3) =
        read_idx_images(&options.data_dir.join("t10k-images-idx3-ubyte"), 10_000)?;
    let (test_labels, d4) =
        read_idx_labels(&options.data_dir.join("t10k-labels-idx1-ubyte"), 10_000)?;
    let train_n = options.train_samples.clamp(1, 60_000);
    let test_n = options.test_samples.clamp(1, 10_000);
    Ok(Dataset {
        train_images: train_images[..train_n * INPUTS].to_vec(),
        train_labels: train_labels[..train_n].to_vec(),
        test_images: test_images[..test_n * INPUTS].to_vec(),
        test_labels: test_labels[..test_n].to_vec(),
        digests: vec![
            ("train-images-idx3-ubyte".into(), d1),
            ("train-labels-idx1-ubyte".into(), d2),
            ("t10k-images-idx3-ubyte".into(), d3),
            ("t10k-labels-idx1-ubyte".into(), d4),
        ],
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    // Small dependency-free SHA-256 so digests are recorded without a new crate.
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut data = bytes.to_vec();
    let bit_len = (bytes.len() as u64) * 8;
    data.push(0x80);
    while data.len() % 64 != 56 {
        data.push(0);
    }
    data.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in data.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (slot, v) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *slot = slot.wrapping_add(v);
        }
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

fn read_idx_images(path: &Path, expected: usize) -> Result<(Vec<u8>, String)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = sha256_hex(&bytes);
    if bytes.len() < 16 {
        bail!("{} is truncated", path.display());
    }
    let word = |i: usize| {
        u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize
    };
    if word(0) != 2051 || word(4) != expected || word(8) != 28 || word(12) != 28 {
        bail!("{} has an unexpected IDX header", path.display());
    }
    if bytes.len() != 16 + expected * INPUTS {
        bail!("{} has an unexpected length", path.display());
    }
    Ok((bytes[16..].to_vec(), digest))
}

fn read_idx_labels(path: &Path, expected: usize) -> Result<(Vec<u8>, String)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = sha256_hex(&bytes);
    if bytes.len() < 8 {
        bail!("{} is truncated", path.display());
    }
    let word = |i: usize| {
        u32::from_be_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]) as usize
    };
    if word(0) != 2049 || word(4) != expected || bytes.len() != 8 + expected {
        bail!("{} has an unexpected IDX header or length", path.display());
    }
    let labels = bytes[8..].to_vec();
    if labels.iter().any(|&l| l > 9) {
        bail!("{} contains a label outside 0..9", path.display());
    }
    Ok((labels, digest))
}

fn tokio_runtime() -> Result<&'static tokio::runtime::Runtime> {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    if let Some(runtime) = RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("create tokio runtime for WASIX")?;
    Ok(RUNTIME.get_or_init(|| runtime))
}
