//! `gpu-image`: runs the image-filter WASIX guest through the `wasmer-gpu`
//! bridge on the local hardware GPU and reports the actual output pixels next
//! to an independent integer CPU reference of the same filter.
//!
//! The guest (`guest/image_filter.wat`) owns every WGSL kernel. This runner
//! owns the input image (a deterministic procedural scene or a bounded raw
//! RGBA8 file), the header the guest reads, the CPU references and the report.
//!
//! Controls: `--control wrong-shader` asks the guest for its identity kernel
//! while the CPU reference applies the requested filter; `--control
//! no-dispatch` tells the guest to touch no GPU handle at all. Both must be
//! reported as a mismatch, which is what makes `matches: true` meaningful.

use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::{path::PathBuf, sync::Arc, time::Instant};
use wasmer::{Module, Store};
use wasmer_gpu_demo::bridge::{self, GpuBridge, SessionLimits, Timing, SUCCESS};
use wasmer_gpu_demo::hook::GpuInstantiationHook;
use wasmer_wasix::{
    runtime::{task_manager::tokio::TokioTaskManager, PluggableRuntime},
    WasiEnvBuilder, WasiError,
};

/// Guest memory layout, mirrored from the WAT comment header.
const IMAGE_BASE: u64 = 16_384;
const HEADER_BYTES: usize = 16;
const GUEST: &str = "image_filter.wat";
/// The guest module is embedded at build time, so the executable's hash
/// identifies the exact WGSL it runs; nothing is read from the checkout.
const GUEST_WAT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/guest/image_filter.wat"
));
/// Largest side the guest memory and session quota are sized for.
pub const MAX_SIDE: u32 = 512;
const NO_DISPATCH_FILTER: u32 = 255;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Filter {
    Blur,
    Edges,
    Emboss,
    Identity,
}

impl Filter {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "blur" => Self::Blur,
            "edges" => Self::Edges,
            "emboss" => Self::Emboss,
            "identity" => Self::Identity,
            other => bail!("unknown filter {other:?}; expected blur, edges, emboss or identity"),
        })
    }

    /// Shader slot index inside the guest.
    fn id(self) -> u32 {
        match self {
            Self::Blur => 0,
            Self::Edges => 1,
            Self::Emboss => 2,
            Self::Identity => 3,
        }
    }

    fn default_param(self) -> u32 {
        match self {
            Self::Blur => 4,
            Self::Edges => 2,
            Self::Emboss => 1,
            Self::Identity => 1,
        }
    }

    /// Inclusive parameter range: blur radius (kernel `2r+1`, the i32
    /// accumulator overflows past 5), edge gain, emboss relief.
    fn max_param(self) -> u32 {
        match self {
            Self::Blur => 5,
            Self::Edges | Self::Emboss => 16,
            Self::Identity => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Control {
    None,
    WrongShader,
    NoDispatch,
}

impl Control {
    fn parse(name: &str) -> Result<Self> {
        Ok(match name {
            "none" => Self::None,
            "wrong-shader" => Self::WrongShader,
            "no-dispatch" => Self::NoDispatch,
            other => bail!("unknown control {other:?}; expected none, wrong-shader or no-dispatch"),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Source {
    Generated,
    File,
}

struct Options {
    filter: Filter,
    param: u32,
    control: Control,
    input: Option<PathBuf>,
    width: u32,
    height: u32,
    json: bool,
}

/// RGBA8 image, row-major. Filters act on RGB; every kernel and every CPU
/// reference carries the source pixel's alpha through unchanged.
#[derive(Clone)]
struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

#[derive(Serialize)]
struct ImageReport {
    subject: &'static str,
    guest: &'static str,
    guest_bytes: usize,
    guest_fnv1a64: String,
    runtime: &'static str,
    wasmer_pin: &'static str,
    build_host_triple: &'static str,
    adapter: String,
    backend: String,
    source: Source,
    width: u32,
    height: u32,
    pixels: usize,
    filter: Filter,
    param: u32,
    control: Control,
    exit_code: Option<i32>,
    setup_us: u128,
    upload_us: u128,
    pipeline_create_us: u128,
    dispatch_us: Vec<u128>,
    readback_us: u128,
    cpu_reference_us: u128,
    matches: bool,
    mismatched_pixels: usize,
    changed_pixels: usize,
    input_fnv1a64: String,
    output_fnv1a64: String,
    input_rgba_b64: String,
    output_rgba_b64: String,
    live_buffers_after: usize,
    live_pipelines_after: usize,
    resources_reclaimed: bool,
    limitations: Vec<&'static str>,
}

fn main() -> Result<()> {
    let options = parse_args()?;
    let (input, source) = match &options.input {
        Some(path) => (
            load_raw_rgba(path, options.width, options.height)?,
            Source::File,
        ),
        None => (
            generate_scene(options.width, options.height),
            Source::Generated,
        ),
    };

    let setup_start = Instant::now();
    let bridge = GpuBridge::new().context("initialize wgpu bridge")?;
    let setup_us = setup_start.elapsed().as_micros();

    let cpu_start = Instant::now();
    let expected = cpu_reference(&input, options.filter, options.param);
    let cpu_reference_us = cpu_start.elapsed().as_micros();

    let guest_filter = match options.control {
        Control::None => options.filter.id(),
        Control::WrongShader => Filter::Identity.id(),
        Control::NoDispatch => NO_DISPATCH_FILTER,
    };
    let run = run_guest(&bridge, &input, guest_filter, options.param)?;
    let snapshot = bridge.snapshot();

    let output = run.output;
    let mismatched_pixels = count_differing_pixels(&expected.rgba, &output.rgba);
    let changed_pixels = count_differing_pixels(&input.rgba, &output.rgba);
    let matches = run.exit_code == Some(SUCCESS)
        && mismatched_pixels == 0
        && run.live_buffers_after == 0
        && run.live_pipelines_after == 0;
    let report = ImageReport {
        subject: "WASIX guest -> wasmer_gpu_v0 -> wgpu hardware compute over an RGBA8 image, guest-supplied WGSL",
        guest: GUEST,
        guest_bytes: GUEST_WAT.len(),
        guest_fnv1a64: format!("{:016x}", fnv1a64(GUEST_WAT.as_bytes())),
        runtime: "wasix",
        wasmer_pin: "github.com/wasmerio/wasmer@6a844cff9bd2eb391ab91b7e41fee94260f0ec46",
        build_host_triple: env!("BUILD_HOST_TRIPLE"),
        adapter: snapshot.adapter_name,
        backend: snapshot.backend,
        source,
        width: input.width,
        height: input.height,
        pixels: (input.width * input.height) as usize,
        filter: options.filter,
        param: options.param,
        control: options.control,
        exit_code: run.exit_code,
        setup_us,
        upload_us: run.timing.upload.as_micros(),
        pipeline_create_us: run.timing.pipeline_create.as_micros(),
        dispatch_us: run.timing.dispatches.iter().map(|d| d.as_micros()).collect(),
        readback_us: run.timing.readback.as_micros(),
        cpu_reference_us,
        matches,
        mismatched_pixels,
        changed_pixels,
        input_fnv1a64: format!("{:016x}", fnv1a64(&input.rgba)),
        output_fnv1a64: format!("{:016x}", fnv1a64(&output.rgba)),
        input_rgba_b64: base64(&input.rgba),
        output_rgba_b64: base64(&output.rgba),
        live_buffers_after: run.live_buffers_after,
        live_pipelines_after: run.live_pipelines_after,
        resources_reclaimed: snapshot.live_buffers == 0 && snapshot.live_pipelines == 0,
        limitations: vec![
            "One read_write storage buffer carries header, input and output; kernels are integer-only so the CPU reference is exact, not approximate.",
            "Timings are host wall-clock around each bridge call on a small image; they are not a speedup claim.",
            "Input is a generated scene or a bounded local raw RGBA8 file (at most 512x512); no network input.",
            "Guest WGSL is trusted demo input from the owned guest module; nothing here preempts a nonterminating kernel.",
        ],
    };

    if options.json {
        println!("{}", serde_json::to_string(&report)?);
    } else {
        print_human_report(&report);
    }
    if !report.matches || !report.resources_reclaimed {
        bail!("image check failed (control={:?})", options.control);
    }
    Ok(())
}

fn parse_args() -> Result<Options> {
    let mut filter = Filter::Blur;
    let mut param = None;
    let mut control = Control::None;
    let mut input = None;
    let mut width = 512;
    let mut height = 384;
    let mut json = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| -> Result<String> {
            args.next().with_context(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "--filter" => filter = Filter::parse(&value("--filter")?)?,
            "--param" => param = Some(value("--param")?.parse::<u32>().context("--param")?),
            "--control" => control = Control::parse(&value("--control")?)?,
            "--input" => input = Some(PathBuf::from(value("--input")?)),
            "--width" => width = value("--width")?.parse::<u32>().context("--width")?,
            "--height" => height = value("--height")?.parse::<u32>().context("--height")?,
            "--json" => json = true,
            other => bail!("unknown argument {other:?}"),
        }
    }
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        bail!("width and height must be within 1..={MAX_SIDE}");
    }
    let param = param.unwrap_or(filter.default_param());
    if !(1..=filter.max_param()).contains(&param) {
        bail!(
            "--param for {filter:?} must be within 1..={}",
            filter.max_param()
        );
    }
    Ok(Options {
        filter,
        param,
        control,
        input,
        width,
        height,
        json,
    })
}

fn load_raw_rgba(path: &PathBuf, width: u32, height: u32) -> Result<Image> {
    let rgba = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let expected = (width * height * 4) as usize;
    if rgba.len() != expected {
        bail!(
            "raw RGBA8 input is {} bytes; {width}x{height} needs exactly {expected}",
            rgba.len()
        );
    }
    Ok(Image {
        width,
        height,
        rgba,
    })
}

// ---------------------------------------------------------------------------
// Guest execution
// ---------------------------------------------------------------------------

struct GuestRun {
    exit_code: Option<i32>,
    output: Image,
    timing: Timing,
    live_buffers_after: usize,
    live_pipelines_after: usize,
}

/// Runs the guest once with a fresh WASIX environment and session. The header
/// and input are written after instantiation; the output region is read after
/// `_start` returns through `proc_exit`.
fn run_guest(bridge: &GpuBridge, input: &Image, filter: u32, param: u32) -> Result<GuestRun> {
    bridge.drain_timing();
    let wasm = wat::parse_str(GUEST_WAT).context("compile embedded WAT guest")?;
    let mut store = Store::default();
    let module = Module::new(&store, wasm).context("compile Wasmer module")?;

    let tokio_runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("create tokio runtime for WASIX")?;
    let _entered = tokio_runtime.enter();
    let task_manager = Arc::new(TokioTaskManager::new(tokio_runtime.handle().clone()));
    let mut runtime = PluggableRuntime::new(task_manager);
    runtime.set_engine(store.engine().clone());
    runtime.with_instantiation_hook(GpuInstantiationHook::new(
        bridge.clone(),
        SessionLimits {
            max_buffers: 2,
            max_bytes: 4 * 1024 * 1024,
            max_pipelines: 2,
        },
    ));
    let (instance, wasi_env) = WasiEnvBuilder::new("wasmer-gpu-image")
        .runtime(Arc::new(runtime))
        .instantiate(module, &mut store)
        .context("instantiate WASIX guest")?;
    let memory = instance
        .exports
        .get_memory("memory")
        .context("guest exports no memory")?
        .clone();

    let pixel_bytes = input.rgba.len();
    let mut region = Vec::with_capacity(HEADER_BYTES + 2 * pixel_bytes);
    for word in [input.width, input.height, filter, param] {
        region.extend_from_slice(&word.to_le_bytes());
    }
    region.extend_from_slice(&input.rgba);
    region.resize(HEADER_BYTES + 2 * pixel_bytes, 0);
    bridge::write_bytes(&memory, &store, IMAGE_BASE, &region)
        .map_err(|code| anyhow::anyhow!("write guest image region: {code}"))?;

    let start = instance
        .exports
        .get_function("_start")
        .context("load guest _start")?;
    let exit_code = match start.call(&mut store, &[]) {
        Ok(_) => Some(SUCCESS),
        Err(error) => match error.downcast_ref::<WasiError>() {
            Some(WasiError::Exit(code)) => Some(i32::from(*code)),
            _ => return Err(error).context("run guest _start"),
        },
    };
    wasi_env.on_exit(&mut store, None);

    let rgba = bridge::read_bytes(
        &memory,
        &store,
        IMAGE_BASE + (HEADER_BYTES + pixel_bytes) as u64,
        pixel_bytes,
    )
    .map_err(|code| anyhow::anyhow!("read guest output: {code}"))?;
    let timing = bridge.drain_timing();
    drop(memory);
    drop(store);
    let snapshot = bridge.snapshot();
    Ok(GuestRun {
        exit_code,
        output: Image {
            width: input.width,
            height: input.height,
            rgba,
        },
        timing,
        live_buffers_after: snapshot.live_buffers,
        live_pipelines_after: snapshot.live_pipelines,
    })
}

// ---------------------------------------------------------------------------
// Independent integer CPU references (mirror the guest kernels bit for bit)
// ---------------------------------------------------------------------------

fn cpu_reference(input: &Image, filter: Filter, param: u32) -> Image {
    let w = input.width as i32;
    let h = input.height as i32;
    let px = |x: i32, y: i32| -> [i32; 3] {
        let cx = x.clamp(0, w - 1);
        let cy = y.clamp(0, h - 1);
        let i = ((cy * w + cx) * 4) as usize;
        [
            input.rgba[i] as i32,
            input.rgba[i + 1] as i32,
            input.rgba[i + 2] as i32,
        ]
    };
    let lum = |x: i32, y: i32| -> i32 {
        let c = px(x, y);
        (c[0] * 77 + c[1] * 151 + c[2] * 28) >> 8
    };
    let param = param as i32;
    let blur_radius = param.clamp(1, 5);
    let blur_norm = 1i32 << (4 * blur_radius);
    let blur_weights: Vec<i32> = (0..=2 * blur_radius)
        .map(|k| binomial(2 * blur_radius, k))
        .collect();
    let mut rgba = vec![0u8; input.rgba.len()];
    for y in 0..h {
        for x in 0..w {
            let c: [i32; 3] = match filter {
                Filter::Blur => {
                    let mut acc = [0i32; 3];
                    for (j, wj) in blur_weights.iter().enumerate() {
                        for (i, wi) in blur_weights.iter().enumerate() {
                            let weight = wi * wj;
                            let p = px(x + i as i32 - blur_radius, y + j as i32 - blur_radius);
                            for k in 0..3 {
                                acc[k] += p[k] * weight;
                            }
                        }
                    }
                    [
                        (acc[0] + blur_norm / 2) / blur_norm,
                        (acc[1] + blur_norm / 2) / blur_norm,
                        (acc[2] + blur_norm / 2) / blur_norm,
                    ]
                }
                Filter::Edges => {
                    let gx = -lum(x - 1, y - 1) - 2 * lum(x - 1, y) - lum(x - 1, y + 1)
                        + lum(x + 1, y - 1)
                        + 2 * lum(x + 1, y)
                        + lum(x + 1, y + 1);
                    let gy = -lum(x - 1, y - 1) - 2 * lum(x, y - 1) - lum(x + 1, y - 1)
                        + lum(x - 1, y + 1)
                        + 2 * lum(x, y + 1)
                        + lum(x + 1, y + 1);
                    let m = (((gx.abs() + gy.abs()) * param) / 8).min(255);
                    [m, m, m]
                }
                Filter::Emboss => {
                    let e = -2 * lum(x - 1, y - 1) - lum(x, y - 1) - lum(x - 1, y)
                        + lum(x + 1, y)
                        + lum(x, y + 1)
                        + 2 * lum(x + 1, y + 1);
                    let v = 128 + (e * param) / 2;
                    [v, v, v]
                }
                Filter::Identity => px(x, y),
            };
            let i = ((y * w + x) * 4) as usize;
            rgba[i] = c[0].clamp(0, 255) as u8;
            rgba[i + 1] = c[1].clamp(0, 255) as u8;
            rgba[i + 2] = c[2].clamp(0, 255) as u8;
            rgba[i + 3] = input.rgba[i + 3];
        }
    }
    Image {
        width: input.width,
        height: input.height,
        rgba,
    }
}

/// `C(n, k)` by the same exact integer recurrence the WGSL kernel uses.
fn binomial(n: i32, k: i32) -> i32 {
    (0..k).fold(1i32, |c, i| c * (n - i) / (i + 1))
}

fn count_differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(p, q)| p != q)
        .count()
}

// ---------------------------------------------------------------------------
// Procedural scene: a glacier valley with enough texture for every filter
// ---------------------------------------------------------------------------

/// Chris Wellons' lowbias32, the same hash the smoke kernels use.
fn lowbias32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    x
}

fn hash01(x: i32, y: i32, seed: u32) -> f32 {
    let h = lowbias32(
        (x as u32)
            .wrapping_mul(0x9e37_79b1)
            .wrapping_add((y as u32).wrapping_mul(0x85eb_ca6b))
            .wrapping_add(seed.wrapping_mul(0xc2b2_ae35)),
    );
    (h >> 8) as f32 / (1u32 << 24) as f32
}

fn value_noise(x: f32, y: f32, seed: u32) -> f32 {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let sx = fx * fx * (3.0 - 2.0 * fx);
    let sy = fy * fy * (3.0 - 2.0 * fy);
    let (ix, iy) = (x0 as i32, y0 as i32);
    let a = hash01(ix, iy, seed);
    let b = hash01(ix + 1, iy, seed);
    let c = hash01(ix, iy + 1, seed);
    let d = hash01(ix + 1, iy + 1, seed);
    let top = a + (b - a) * sx;
    let bottom = c + (d - c) * sx;
    top + (bottom - top) * sy
}

fn fbm(x: f32, y: f32, seed: u32, octaves: u32) -> f32 {
    let mut sum = 0.0;
    let mut amplitude = 0.5;
    let mut frequency = 1.0;
    for octave in 0..octaves {
        sum += amplitude * value_noise(x * frequency, y * frequency, seed + octave);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    sum
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Deterministic glacier valley: sky gradient with thin cirrus, a sun disc,
/// two slope-shaded ridges lit from the sun (snow on gentle slopes, striated
/// rock on steep ones), a row of spruce silhouettes on the shore, and a lake
/// with a streaked reflection and a sun path. Film grain everywhere.
fn generate_scene(width: u32, height: u32) -> Image {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    let wf = width as f32;
    let hf = height as f32;
    let aspect = wf / hf;
    let sun_u = 0.72;
    let sun_v = 0.17;
    let far_ridge = |u: f32| 0.50 + 0.09 * fbm(u * 3.0 + 40.0, 7.0, 21, 5);
    let near_ridge = |u: f32| 0.60 + 0.15 * fbm(u * 2.2 + 12.0, 3.0, 31, 6);
    let shore = |u: f32| 0.84 + 0.012 * fbm(u * 20.0, 9.0, 41, 2);
    // Slopes are taken over eight columns so ridge shading follows the large
    // shapes instead of the per-pixel noise.
    let du = 8.0 / wf;
    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / wf;
            let v = y as f32 / hf;
            let mut color = mix(
                [0.07, 0.20, 0.60],
                [0.84, 0.92, 0.99],
                (v / 0.55).powf(0.85),
            );
            let cirrus = fbm(u * 5.0 + 3.1, v * 30.0, 11, 4);
            let cirrus = ((cirrus - 0.55) * 6.0).clamp(0.0, 1.0) * (1.0 - v / 0.5).clamp(0.0, 1.0);
            color = mix(color, [0.98, 0.99, 1.0], cirrus * 0.8);

            let sun = ((u - sun_u) * aspect).hypot(v - sun_v);
            let disc = (1.0 - (sun - 0.05) / 0.005).clamp(0.0, 1.0);
            let halo = (1.0 - sun / 0.32).clamp(0.0, 1.0).powi(3);
            color = mix(color, [1.0, 0.94, 0.78], halo * 0.55);
            color = mix(color, [1.0, 0.99, 0.94], disc);

            let far = far_ridge(u);
            if v > far {
                let slope = (far_ridge(u + du) - far) / du;
                let lit = (0.5 - slope * 0.6).clamp(0.0, 1.0);
                let snow = fbm(u * 18.0, v * 18.0, 22, 4);
                let gully = fbm(u * 7.0, v * 3.0 + 4.0, 23, 3);
                let shaded = mix(
                    [0.38, 0.48, 0.74],
                    [0.90, 0.94, 0.99],
                    (snow * 1.3 + gully * 0.5 + lit * 0.4 - 0.55).clamp(0.0, 1.0),
                );
                let haze = (1.0 - (v - far) / 0.10).clamp(0.0, 1.0);
                color = mix(shaded, [0.74, 0.83, 0.96], haze * 0.45);
            }

            let near = near_ridge(u);
            if v > near {
                let depth = v - near;
                let lit = (0.55 - (near_ridge(u + du) - near) / du * 0.5).clamp(0.2, 1.0);
                let snowline = 0.05 + 0.06 * fbm(u * 9.0, 5.0, 32, 3);
                let strata =
                    ((v * 90.0 + fbm(u * 5.0, v * 5.0, 33, 3) * 12.0).sin() * 0.5 + 0.5) * 0.35;
                let grit = fbm(u * 60.0, v * 60.0, 34, 4);
                let patches = fbm(u * 14.0, v * 10.0, 35, 3);
                let rock = mix(
                    [0.09, 0.12, 0.20],
                    [0.36, 0.42, 0.54],
                    (strata + grit * 0.6) * (0.5 + lit * 0.6),
                );
                let snow = mix(
                    [0.94, 0.97, 1.0],
                    [0.70, 0.80, 0.95],
                    grit * 0.6 + (1.0 - lit) * 0.6,
                );
                let blend = ((depth - snowline) * 16.0 + (patches - 0.5) * 1.6).clamp(0.0, 1.0);
                color = mix(snow, rock, blend);
            }

            let shore_v = shore(u);
            let tree_band = shore_v - 0.11;
            if v > tree_band && v <= shore_v {
                let slot = (u * 36.0).floor() as i32;
                for t in slot - 1..=slot + 1 {
                    let centre = (t as f32 + 0.2 + 0.6 * hash01(t, 0, 51)) / 36.0;
                    let tree_height = 0.05 + 0.06 * hash01(t, 1, 52);
                    let top = shore_v - tree_height;
                    if v < top || hash01(t, 2, 53) < 0.25 {
                        continue;
                    }
                    let fraction = (v - top) / tree_height;
                    let tier = ((fraction * 7.0).fract() * 0.5 + 0.5) * 0.35 + 0.65;
                    let half_width =
                        (0.0015 + 0.024 * fraction) * tier * (0.8 + 0.4 * hash01(t, 3, 54));
                    if ((u - centre) * aspect).abs() < half_width {
                        let side = ((u - centre) * aspect / half_width).clamp(-1.0, 1.0);
                        color = mix(
                            [0.04, 0.08, 0.10],
                            [0.12, 0.22, 0.20],
                            (0.5 - side * 0.5) * 0.6,
                        );
                    }
                }
            }

            if v > shore_v {
                let mirror = 2.0 * shore_v - v;
                let streak = fbm(u * 3.0, v * 120.0, 42, 4);
                let ripple = fbm(u * 40.0, v * 200.0, 43, 3);
                let deep = [0.05, 0.13, 0.36];
                let lit = [0.36, 0.60, 0.90];
                color = mix(deep, lit, streak * 0.8 + ripple * 0.3 - 0.25);
                if mirror > tree_band && mirror < shore_v {
                    color = mix(color, [0.03, 0.06, 0.10], 0.35 * (1.0 - ripple * 0.5));
                } else if mirror > near {
                    color = mix(color, [0.50, 0.60, 0.80], 0.25 * ripple);
                }
                let sun_path = (1.0 - ((u - sun_u) / 0.05).abs()).clamp(0.0, 1.0) * ripple;
                color = mix(color, [0.98, 0.94, 0.80], sun_path * 0.75);
            }

            let grain = (hash01(x as i32, y as i32, 99) - 0.5) * 0.045;
            let to_byte = |c: f32| ((c + grain).clamp(0.0, 1.0) * 255.0).round() as u8;
            rgba.extend_from_slice(&[to_byte(color[0]), to_byte(color[1]), to_byte(color[2]), 255]);
        }
    }
    Image {
        width,
        height,
        rgba,
    }
}

// ---------------------------------------------------------------------------
// Encoding helpers (kept dependency-free so Cargo.lock stays as merged)
// ---------------------------------------------------------------------------

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| {
        (hash ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk
            .iter()
            .enumerate()
            .fold(0u32, |acc, (i, &b)| acc | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(TABLE[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn print_human_report(report: &ImageReport) {
    println!("wasmer-gpu image workbench");
    println!("==========================");
    println!("subject: {}", report.subject);
    println!(
        "guest: {} [{}] embedded {} bytes fnv1a64={}",
        report.guest, report.runtime, report.guest_bytes, report.guest_fnv1a64
    );
    println!("adapter: {} ({})", report.adapter, report.backend);
    println!(
        "image: {}x{} {:?}, filter={:?} param={} control={:?}",
        report.width, report.height, report.source, report.filter, report.param, report.control
    );
    println!("exit code: {:?}", report.exit_code);
    println!(
        "setup={}us upload={}us pipeline_create={}us dispatch={:?}us readback={}us cpu_ref={}us",
        report.setup_us,
        report.upload_us,
        report.pipeline_create_us,
        report.dispatch_us,
        report.readback_us,
        report.cpu_reference_us
    );
    println!(
        "matches CPU reference: {} (mismatched {}, changed {} of {} pixels)",
        report.matches, report.mismatched_pixels, report.changed_pixels, report.pixels
    );
    println!(
        "input fnv1a64={} output fnv1a64={}",
        report.input_fnv1a64, report.output_fnv1a64
    );
    println!("resources reclaimed: {}", report.resources_reclaimed);
    println!("limitations:");
    for limitation in &report.limitations {
        println!("  - {limitation}");
    }
}
