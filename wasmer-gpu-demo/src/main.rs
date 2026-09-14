//! `gpu-smoke`: runs guest WAT modules through the `wasmer-gpu` bridge on the
//! local hardware GPU and reports correctness against independent CPU
//! references plus the outcome of guest-driven negative controls.
//!
//! The runner knows which guest carries which kernel and how to compute the
//! CPU reference; the bridge in [`bridge`] does not. Kernel WGSL lives inside
//! the guest modules under `guest/`.

use anyhow::{bail, Context, Result};
use bytemuck::cast_slice;
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{Arc, OnceLock},
    time::Instant,
};
use wasmer::{
    imports, FunctionEnv, Instance, Memory, MemoryType, Module, Pages, Store, TypedFunction,
};
use wasmer_gpu_demo::bridge::{
    self, GpuBridge, GpuSession, HostEnv, SessionLimits, Timing, SUCCESS,
};
use wasmer_gpu_demo::hook::GpuInstantiationHook;
use wasmer_wasix::{
    runtime::{task_manager::tokio::TokioTaskManager, PluggableRuntime},
    WasiEnvBuilder, WasiError,
};

const OVERFLOW: u32 = 0xffff_ffff;

/// A guest that runs a kernel over synthetic input. The host writes `input`
/// at guest offset 0 before `_start`; the guest is expected to leave its
/// result immediately after the input. `rounds` is how many times the guest
/// dispatches the same kernel over the same buffer, so the CPU reference is
/// applied the same number of times.
struct KernelSpec {
    guest: &'static str,
    runtime: GuestRuntime,
    input: fn() -> Vec<u32>,
    rounds: usize,
    cpu_reference: fn(&[u32]) -> Vec<u32>,
}

/// A negative or lifecycle control. `_start` returns 0 when the guest observed
/// the documented error codes; any other status is the failing check number.
struct ControlSpec {
    guest: &'static str,
    runtime: GuestRuntime,
    with_input: bool,
    expect_trap: bool,
}

/// How a guest is instantiated. `Core` builds the import object by hand for a
/// core module with imported memory and an `i32`-returning `_start`. `Wasix`
/// runs a WASIX command guest (exported memory, void `_start`, exit through
/// `proc_exit`) whose bridge imports arrive via the runtime's
/// `InstantiationHook`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum GuestRuntime {
    Core,
    Wasix,
}

const KERNELS: &[KernelSpec] = &[
    KernelSpec {
        guest: "collatz_valid.wat",
        runtime: GuestRuntime::Core,
        input: || vec![1, 2, 3, 4],
        rounds: 1,
        cpu_reference: collatz_reference,
    },
    KernelSpec {
        guest: "lowbias32_warm.wat",
        runtime: GuestRuntime::Core,
        input: || vec![0, 1, 2, 3, 42, 1000, 65_535, 4_294_967_295],
        rounds: 8,
        cpu_reference: lowbias32_reference,
    },
    KernelSpec {
        guest: "wasix_collatz.wat",
        runtime: GuestRuntime::Wasix,
        input: || vec![1, 2, 3, 4],
        rounds: 1,
        cpu_reference: collatz_reference,
    },
];

const CONTROLS: &[ControlSpec] = &[
    ControlSpec {
        guest: "invalid_range.wat",
        runtime: GuestRuntime::Core,
        with_input: false,
        expect_trap: false,
    },
    ControlSpec {
        guest: "invalid_handle.wat",
        runtime: GuestRuntime::Core,
        with_input: false,
        expect_trap: false,
    },
    ControlSpec {
        guest: "invalid_shader.wat",
        runtime: GuestRuntime::Core,
        with_input: true,
        expect_trap: false,
    },
    ControlSpec {
        guest: "unaligned_read.wat",
        runtime: GuestRuntime::Core,
        with_input: true,
        expect_trap: false,
    },
    ControlSpec {
        guest: "quota.wat",
        runtime: GuestRuntime::Core,
        with_input: true,
        expect_trap: false,
    },
    ControlSpec {
        guest: "early_exit.wat",
        runtime: GuestRuntime::Core,
        with_input: true,
        expect_trap: false,
    },
    ControlSpec {
        guest: "leak.wat",
        runtime: GuestRuntime::Core,
        with_input: true,
        expect_trap: false,
    },
    ControlSpec {
        guest: "trap_after_create.wat",
        runtime: GuestRuntime::Core,
        with_input: true,
        expect_trap: true,
    },
    ControlSpec {
        guest: "wasix_leak.wat",
        runtime: GuestRuntime::Wasix,
        with_input: true,
        expect_trap: false,
    },
];

#[derive(Serialize)]
struct SmokeReport {
    subject: &'static str,
    wasmer_pin: &'static str,
    wasi_webgpu_reference_pin: &'static str,
    wasi_gfx_runtime_reference_pin: &'static str,
    build_host_triple: &'static str,
    adapter: String,
    backend: String,
    abi: &'static str,
    setup_us: u128,
    kernels: Vec<KernelReport>,
    controls: Vec<ControlReport>,
    all_kernels_match: bool,
    all_controls_pass: bool,
    resources_reclaimed: bool,
    limitations: Vec<&'static str>,
}

#[derive(Serialize)]
struct KernelReport {
    guest: &'static str,
    runtime: GuestRuntime,
    elements: usize,
    rounds: usize,
    status: i32,
    gpu_result: Vec<u32>,
    cpu_result: Vec<u32>,
    matches: bool,
    cpu_reference_ns: u128,
    upload_us: u128,
    pipeline_create_us: u128,
    dispatch_us: Vec<u128>,
    readback_us: u128,
    live_buffers_after: usize,
    live_pipelines_after: usize,
}

#[derive(Serialize)]
struct ControlReport {
    guest: &'static str,
    runtime: GuestRuntime,
    status: Option<i32>,
    trapped: bool,
    passed: bool,
    live_buffers_after: usize,
    live_pipelines_after: usize,
}

struct GuestRun {
    status: Option<i32>,
    trapped: bool,
    output: Option<Vec<u32>>,
    timing: Timing,
    live_buffers_after: usize,
    live_pipelines_after: usize,
}

fn main() -> Result<()> {
    let report_json = std::env::args().any(|arg| arg == "--json");

    let setup_start = Instant::now();
    let bridge = GpuBridge::new().context("initialize wgpu bridge")?;
    let setup_us = setup_start.elapsed().as_micros();

    let mut kernels = Vec::new();
    for spec in KERNELS {
        kernels.push(run_kernel(&bridge, spec)?);
    }
    let mut controls = Vec::new();
    for spec in CONTROLS {
        controls.push(run_control(&bridge, spec)?);
    }

    let snapshot = bridge.snapshot();
    let report = SmokeReport {
        subject: "Wasmer core and WASIX guests -> wasmer_gpu_v0 host imports -> wgpu hardware compute, guest-supplied WGSL",
        wasmer_pin: "github.com/wasmerio/wasmer@6a844cff9bd2eb391ab91b7e41fee94260f0ec46",
        wasi_webgpu_reference_pin:
            "sponsors/wasi-webgpu@6a776bada0b66d3dbf9da304a49ff2947ce4e1f8",
        wasi_gfx_runtime_reference_pin:
            "sponsors/wasi-gfx-runtime@772bc344d3d0e24ba2d3ee29fc0033fc6ccea81d",
        build_host_triple: env!("BUILD_HOST_TRIPLE"),
        adapter: snapshot.adapter_name,
        backend: snapshot.backend,
        abi: "wasmer_gpu_v0: buffer_upload, buffer_read, buffer_release, pipeline_create, pipeline_dispatch, pipeline_release",
        setup_us,
        all_kernels_match: kernels.iter().all(|k| k.matches),
        all_controls_pass: controls.iter().all(|c| c.passed),
        resources_reclaimed: snapshot.live_buffers == 0 && snapshot.live_pipelines == 0,
        kernels,
        controls,
        limitations: vec![
            "wasi-webgpu and wasi-gfx-runtime are reference semantics only; this runner does not use Wasmtime or the component model.",
            "WASIX coverage is one command guest instantiated through WasiEnvBuilder with the bridge registered as a PluggableRuntime InstantiationHook; no threads, side modules, filesystem or networking are exercised.",
            "The host exposes a bounded compute subset: one read_write storage binding, a fixed entry point, bounded dispatch dimensions, per-instance quotas. It is not full WebGPU, CUDA, graphics, zero-copy, GPU preemption, or distributed training.",
            "Dispatch timings are host wall-clock around submit and wait on tiny inputs; they are not a speedup claim.",
            "No cloud runner, public upload, deployment, model call, or third-party probing is used.",
        ],
    };

    if report_json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        print_human_report(&report);
    }

    if !report.all_kernels_match || !report.all_controls_pass || !report.resources_reclaimed {
        bail!("smoke check failed");
    }
    Ok(())
}

fn run_kernel(bridge: &GpuBridge, spec: &KernelSpec) -> Result<KernelReport> {
    let input = (spec.input)();
    let cpu_start = Instant::now();
    let mut cpu_result = input.clone();
    for _ in 0..spec.rounds {
        cpu_result = (spec.cpu_reference)(&cpu_result);
    }
    let cpu_reference_ns = cpu_start.elapsed().as_nanos();

    let run = execute_guest(bridge, spec.runtime, spec.guest, Some(&input), false)
        .with_context(|| format!("run kernel guest {}", spec.guest))?;
    let status = run.status.unwrap_or(i32::MIN);
    let gpu_result = run.output.unwrap_or_default();
    Ok(KernelReport {
        guest: spec.guest,
        runtime: spec.runtime,
        elements: input.len(),
        rounds: spec.rounds,
        status,
        matches: status == SUCCESS
            && gpu_result == cpu_result
            && run.timing.dispatches.len() == spec.rounds
            && run.live_buffers_after == 0
            && run.live_pipelines_after == 0,
        gpu_result,
        cpu_result,
        cpu_reference_ns,
        upload_us: run.timing.upload.as_micros(),
        pipeline_create_us: run.timing.pipeline_create.as_micros(),
        dispatch_us: run
            .timing
            .dispatches
            .iter()
            .map(|d| d.as_micros())
            .collect(),
        readback_us: run.timing.readback.as_micros(),
        live_buffers_after: run.live_buffers_after,
        live_pipelines_after: run.live_pipelines_after,
    })
}

fn run_control(bridge: &GpuBridge, spec: &ControlSpec) -> Result<ControlReport> {
    let input = spec.with_input.then(|| vec![1u32, 2, 3, 4]);
    let run = execute_guest(
        bridge,
        spec.runtime,
        spec.guest,
        input.as_deref(),
        spec.expect_trap,
    )
    .with_context(|| format!("run control guest {}", spec.guest))?;
    let outcome_ok = if spec.expect_trap {
        run.trapped
    } else {
        run.status == Some(SUCCESS)
    };
    Ok(ControlReport {
        guest: spec.guest,
        runtime: spec.runtime,
        status: run.status,
        trapped: run.trapped,
        passed: outcome_ok && run.live_buffers_after == 0 && run.live_pipelines_after == 0,
        live_buffers_after: run.live_buffers_after,
        live_pipelines_after: run.live_pipelines_after,
    })
}

/// Runs one guest with a fresh session and reads the output that follows the
/// input region. The session drops with the store, which is what reclaims
/// resources after a leak, early exit or trap.
fn execute_guest(
    bridge: &GpuBridge,
    runtime: GuestRuntime,
    guest_name: &str,
    input: Option<&[u32]>,
    capture_trap: bool,
) -> Result<GuestRun> {
    bridge.drain_timing();
    let wat = std::fs::read_to_string(guest_path(guest_name)).context("read guest WAT")?;
    let wasm = wat::parse_str(&wat).context("compile WAT guest")?;
    let mut store = Store::default();
    let module = Module::new(&store, wasm).context("compile Wasmer module")?;
    let (status, trapped, memory) = match runtime {
        GuestRuntime::Core => run_core_guest(bridge, &mut store, &module, input, capture_trap)?,
        GuestRuntime::Wasix => run_wasix_guest(bridge, &mut store, &module, input, capture_trap)?,
    };
    let output = match (input, status) {
        (Some(input), Some(SUCCESS)) => {
            let bytes = bridge::read_bytes(
                &memory,
                &store,
                std::mem::size_of_val(input) as u64,
                std::mem::size_of_val(input),
            )
            .map_err(|code| anyhow::anyhow!("read guest output: {code}"))?;
            Some(cast_slice(&bytes).to_vec())
        }
        _ => None,
    };
    let timing = bridge.drain_timing();
    drop(memory);
    drop(store);
    let snapshot = bridge.snapshot();
    Ok(GuestRun {
        status,
        trapped,
        output,
        timing,
        live_buffers_after: snapshot.live_buffers,
        live_pipelines_after: snapshot.live_pipelines,
    })
}

/// Core module: host-created imported memory, imports built by hand,
/// `_start () -> i32` where the return value is the status.
fn run_core_guest(
    bridge: &GpuBridge,
    store: &mut Store,
    module: &Module,
    input: Option<&[u32]>,
    capture_trap: bool,
) -> Result<(Option<i32>, bool, Memory)> {
    let memory = Memory::new(
        &mut *store,
        MemoryType::new(Pages(1), Some(Pages(1)), false),
    )
    .context("create guest linear memory")?;
    if let Some(input) = input {
        bridge::write_bytes(&memory, store, 0, cast_slice(input))
            .map_err(|code| anyhow::anyhow!("write guest input: {code}"))?;
    }
    let session = GpuSession::new(bridge.clone(), SessionLimits::default());
    let env = FunctionEnv::new(
        &mut *store,
        HostEnv {
            memory: Some(memory.clone()),
            session,
        },
    );
    let mut import_object = imports! { "env" => { "memory" => memory.clone() } };
    bridge::register_imports(store, &env, &mut import_object);
    let instance =
        Instance::new(&mut *store, module, &import_object).context("instantiate guest")?;
    let start: TypedFunction<(), i32> = instance
        .exports
        .get_typed_function(&*store, "_start")
        .context("load guest _start")?;
    let (status, trapped) = match start.call(&mut *store) {
        Ok(status) => (Some(status), false),
        Err(_) if capture_trap => (None, true),
        Err(error) => return Err(error).context("run guest _start"),
    };
    Ok((status, trapped, memory))
}

/// WASIX command guest: the runtime supplies WASI imports, and the
/// `GpuInstantiationHook` registered on it supplies `wasmer_gpu_v0`. The
/// guest exports its memory and reports its status through `proc_exit`.
fn run_wasix_guest(
    bridge: &GpuBridge,
    store: &mut Store,
    module: &Module,
    input: Option<&[u32]>,
    capture_trap: bool,
) -> Result<(Option<i32>, bool, Memory)> {
    // PluggableRuntime::new and its networking default require a live tokio
    // context, so the process-wide runtime is entered for the whole call and
    // outlives every store that borrows it.
    let tokio_runtime = tokio_runtime()?;
    let _entered = tokio_runtime.enter();
    let task_manager = Arc::new(TokioTaskManager::new(tokio_runtime.handle().clone()));
    let mut runtime = PluggableRuntime::new(task_manager);
    runtime.set_engine(store.engine().clone());
    runtime.with_instantiation_hook(GpuInstantiationHook::new(
        bridge.clone(),
        SessionLimits::default(),
    ));
    let (instance, wasi_env) = WasiEnvBuilder::new("wasmer-gpu-guest")
        .runtime(Arc::new(runtime))
        .instantiate(module.clone(), &mut *store)
        .context("instantiate WASIX guest")?;
    let memory = instance
        .exports
        .get_memory("memory")
        .context("WASIX guest exports no memory")?
        .clone();
    if let Some(input) = input {
        bridge::write_bytes(&memory, store, 0, cast_slice(input))
            .map_err(|code| anyhow::anyhow!("write guest input: {code}"))?;
    }
    let start = instance
        .exports
        .get_function("_start")
        .context("load WASIX guest _start")?;
    let outcome = start.call(&mut *store, &[]);
    let (status, trapped) = match outcome {
        Ok(_) => (Some(SUCCESS), false),
        Err(error) => match error.downcast_ref::<WasiError>() {
            Some(WasiError::Exit(code)) => (Some(i32::from(*code)), false),
            _ if capture_trap => (None, true),
            _ => return Err(error).context("run WASIX guest _start"),
        },
    };
    wasi_env.on_exit(&mut *store, None);
    Ok((status, trapped, memory))
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

fn guest_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("guest")
        .join(name)
}

// ---------------------------------------------------------------------------
// Independent CPU references (kept out of the bridge on purpose)
// ---------------------------------------------------------------------------

fn collatz_reference(numbers: &[u32]) -> Vec<u32> {
    numbers.iter().copied().map(collatz_iterations).collect()
}

fn collatz_iterations(n_base: u32) -> u32 {
    let mut n = n_base;
    let mut i = 0;
    while n > 1 {
        if n % 2 == 0 {
            n /= 2;
        } else {
            if n >= 0x5555_5555 {
                return OVERFLOW;
            }
            n = 3 * n + 1;
        }
        i += 1;
    }
    i
}

fn lowbias32_reference(numbers: &[u32]) -> Vec<u32> {
    numbers.iter().copied().map(lowbias32).collect()
}

/// Chris Wellons' lowbias32 integer hash; every step is defined modulo 2^32.
fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn print_human_report(report: &SmokeReport) {
    println!("wasmer-gpu compute smoke");
    println!("========================");
    println!("subject: {}", report.subject);
    println!("wasmer pin: {}", report.wasmer_pin);
    println!(
        "wasi-webgpu reference: {}",
        report.wasi_webgpu_reference_pin
    );
    println!(
        "wasi-gfx-runtime reference: {}",
        report.wasi_gfx_runtime_reference_pin
    );
    println!("build host: {}", report.build_host_triple);
    println!("adapter: {} ({})", report.adapter, report.backend);
    println!("abi: {}", report.abi);
    println!("setup: {}us", report.setup_us);
    println!("kernels:");
    for k in &report.kernels {
        println!(
            "  {} [{:?}] ({} u32 x {} round(s)): status={} match={}",
            k.guest, k.runtime, k.elements, k.rounds, k.status, k.matches
        );
        println!("    gpu: {:?}", k.gpu_result);
        println!("    cpu: {:?}", k.cpu_result);
        println!(
            "    upload={}us pipeline_create={}us dispatch={:?}us readback={}us cpu_ref={}ns",
            k.upload_us, k.pipeline_create_us, k.dispatch_us, k.readback_us, k.cpu_reference_ns
        );
    }
    println!("controls:");
    for c in &report.controls {
        println!(
            "  {} [{:?}]: status={:?} trapped={} live_buffers={} live_pipelines={} passed={}",
            c.guest,
            c.runtime,
            c.status,
            c.trapped,
            c.live_buffers_after,
            c.live_pipelines_after,
            c.passed
        );
    }
    println!("all kernels match: {}", report.all_kernels_match);
    println!("all controls pass: {}", report.all_controls_pass);
    println!("resources reclaimed: {}", report.resources_reclaimed);
    println!("limitations:");
    for limitation in &report.limitations {
        println!("  - {limitation}");
    }
}
