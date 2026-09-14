//! `wasmer-gpu` bridge: the `wasmer_gpu_v0` host imports for Wasmer core guests.
//!
//! Caller map (guest import -> host entry here -> wgpu backend call):
//!
//! | import                            | host entry          | backend                                   |
//! |-----------------------------------|---------------------|-------------------------------------------|
//! | `buffer_upload(ptr, len)`         | [`buffer_upload`]   | `create_buffer_init` (STORAGE, COPY_SRC)  |
//! | `buffer_read(buf, ptr, len)`      | [`buffer_read`]     | `copy_buffer_to_buffer` + `map_async`     |
//! | `buffer_write(buf, off, ptr, len)`| [`buffer_write`]    | `Queue::write_buffer` into an owned buffer|
//! | `buffer_release(buf)`             | [`buffer_release`]  | drop `wgpu::Buffer`                       |
//! | `pipeline_create(ptr, len)`       | [`pipeline_create`] | `create_shader_module` + compute pipeline |
//! | `pipeline_dispatch(pipe, buf, x, y, z)` | [`pipeline_dispatch`] | bind group + compute pass + submit  |
//! | `pipeline_release(pipe)`          | [`pipeline_release`]| drop `wgpu::ComputePipeline`              |
//!
//! Every import validates guest memory ranges and session-owned handles before
//! touching the device, and all device work runs inside wgpu error scopes so a
//! validation or allocation failure becomes a return code, never a host abort.
//! A [`GpuSession`] is created per guest instance; dropping it releases every
//! handle the instance still owns, including after a trap.
//!
//! Kernel source is never part of this module. The guest supplies WGSL text.
//! The enforced binding contract is one read-write storage buffer at
//! `@group(0) @binding(0)` and a `@compute` entry point named `main`; that is
//! what the fixed `BindGroupLayout` and pipeline creation validate. The host
//! does not inspect the WGSL element type: the demo kernels interpret the
//! buffer's raw bytes as `array<u32>`, but any layout the guest chooses is
//! its own business. Guest WGSL is trusted demo input; nothing here can
//! preempt a nonterminating kernel once submitted.

use anyhow::{bail, Context, Result};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use wasmer::{AsStoreMut, Function, FunctionEnv, FunctionEnvMut, Imports, Memory, Module};
use wgpu::util::DeviceExt;

/// Import namespace. Provisional; not a standardized ABI claim.
pub const NAMESPACE: &str = "wasmer_gpu_v0";
/// Compute entry point every guest kernel must export.
pub const ENTRY_POINT: &str = "main";

pub const SUCCESS: i32 = 0;
/// Range, size, alignment or dispatch-dimension violation.
pub const ERR_RANGE: i32 = -1;
/// Handle unknown to this instance, released, or of the wrong kind.
pub const ERR_HANDLE: i32 = -2;
/// Captured wgpu allocation or runtime error.
pub const ERR_GPU: i32 = -3;
/// Per-instance buffer, byte or pipeline quota exhausted.
pub const ERR_QUOTA: i32 = -4;
/// Shader text rejected: not UTF-8, failed WGSL validation, or incompatible
/// with the fixed binding contract.
pub const ERR_SHADER: i32 = -5;

pub const TRANSFER_ALIGNMENT: usize = wgpu::COPY_BUFFER_ALIGNMENT as usize;
pub const MAX_SHADER_BYTES: usize = 16 * 1024;
/// WebGPU `maxComputeWorkgroupsPerDimension`.
pub const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;
/// Host-imposed cap on `x * y * z` workgroups per dispatch.
pub const MAX_WORKGROUPS_PER_DISPATCH: u64 = 1 << 16;

#[derive(Clone, Copy, Debug)]
pub struct SessionLimits {
    pub max_buffers: usize,
    pub max_bytes: usize,
    pub max_pipelines: usize,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_buffers: 8,
            max_bytes: 1024 * 1024,
            max_pipelines: 4,
        }
    }
}

/// Per-instance function environment handed to every import.
///
/// `memory` is `None` until the guest's linear memory is known: immediately
/// for the core-module runner, and after instantiation for a WASIX guest
/// whose memory is exported rather than imported (see [`crate::hook`]).
#[derive(Clone)]
pub struct HostEnv {
    pub memory: Option<Memory>,
    pub session: GpuSession,
}

/// Registers the six `wasmer_gpu_v0` imports bound to `env` into `imports`,
/// leaving any entry that is already present untouched.
pub fn register_imports(
    store: &mut impl AsStoreMut,
    env: &FunctionEnv<HostEnv>,
    imports: &mut Imports,
) {
    let entries: [(&str, Function); 7] = [
        (
            "buffer_upload",
            Function::new_typed_with_env(store, env, buffer_upload),
        ),
        (
            "buffer_read",
            Function::new_typed_with_env(store, env, buffer_read),
        ),
        (
            "buffer_write",
            Function::new_typed_with_env(store, env, buffer_write),
        ),
        (
            "buffer_release",
            Function::new_typed_with_env(store, env, buffer_release),
        ),
        (
            "pipeline_create",
            Function::new_typed_with_env(store, env, pipeline_create),
        ),
        (
            "pipeline_dispatch",
            Function::new_typed_with_env(store, env, pipeline_dispatch),
        ),
        (
            "pipeline_release",
            Function::new_typed_with_env(store, env, pipeline_release),
        ),
    ];
    for (name, function) in entries {
        if imports.get_export(NAMESPACE, name).is_none() {
            imports.define(NAMESPACE, name, function);
        }
    }
}

/// True when `module` imports anything from the bridge namespace.
pub fn module_uses_bridge(module: &Module) -> bool {
    module.imports().any(|import| import.module() == NAMESPACE)
}

fn guest_memory(env: &FunctionEnvMut<HostEnv>) -> std::result::Result<Memory, i32> {
    env.data().memory.clone().ok_or(ERR_RANGE)
}

// ---------------------------------------------------------------------------
// Guest-facing imports
// ---------------------------------------------------------------------------

pub fn buffer_upload(env: FunctionEnvMut<HostEnv>, ptr: i32, len: i32) -> i32 {
    let (ptr, len) = match guest_transfer_range(ptr, len) {
        Ok(range) => range,
        Err(code) => return code,
    };
    let memory = match guest_memory(&env) {
        Ok(memory) => memory,
        Err(code) => return code,
    };
    let bytes = match read_bytes(&memory, &env, ptr, len) {
        Ok(bytes) => bytes,
        Err(code) => return code,
    };
    let session = &env.data().session;
    if !session.can_allocate_buffer(len) {
        return ERR_QUOTA;
    }
    match session.bridge().upload(&bytes) {
        Ok(handle) => {
            session.track_buffer(handle, len);
            handle
        }
        Err(code) => code,
    }
}

pub fn buffer_read(env: FunctionEnvMut<HostEnv>, handle: i32, ptr: i32, len: i32) -> i32 {
    let (ptr, len) = match guest_transfer_range(ptr, len) {
        Ok(range) => range,
        Err(code) => return code,
    };
    let session = &env.data().session;
    if !session.owns_buffer(handle) {
        return ERR_HANDLE;
    }
    let bytes = match session.bridge().read_buffer(handle, len) {
        Ok(bytes) => bytes,
        Err(code) => return code,
    };
    let memory = match guest_memory(&env) {
        Ok(memory) => memory,
        Err(code) => return code,
    };
    match write_bytes(&memory, &env, ptr, &bytes) {
        Ok(()) => SUCCESS,
        Err(code) => code,
    }
}

/// Overwrites `len` bytes at `offset` of a session-owned buffer with guest
/// memory, so a guest can refresh a GPU-resident buffer (a minibatch, a
/// parameter block) without reallocating. Same alignment and range rules as
/// upload; the write is awaited before returning.
pub fn buffer_write(
    env: FunctionEnvMut<HostEnv>,
    handle: i32,
    offset: i32,
    ptr: i32,
    len: i32,
) -> i32 {
    let (ptr, len) = match guest_transfer_range(ptr, len) {
        Ok(range) => range,
        Err(code) => return code,
    };
    let offset = offset as u32 as usize;
    if offset % TRANSFER_ALIGNMENT != 0 {
        return ERR_RANGE;
    }
    let session = &env.data().session;
    if !session.owns_buffer(handle) {
        return ERR_HANDLE;
    }
    let memory = match guest_memory(&env) {
        Ok(memory) => memory,
        Err(code) => return code,
    };
    let bytes = match read_bytes(&memory, &env, ptr, len) {
        Ok(bytes) => bytes,
        Err(code) => return code,
    };
    session.bridge().write_buffer(handle, offset, &bytes)
}

pub fn buffer_release(env: FunctionEnvMut<HostEnv>, handle: i32) -> i32 {
    let session = &env.data().session;
    if !session.forget_buffer(handle) {
        return ERR_HANDLE;
    }
    session.bridge().release_buffer(handle)
}

pub fn pipeline_create(env: FunctionEnvMut<HostEnv>, ptr: i32, len: i32) -> i32 {
    let len = len as u32 as usize;
    if len == 0 || len > MAX_SHADER_BYTES {
        return ERR_RANGE;
    }
    let memory = match guest_memory(&env) {
        Ok(memory) => memory,
        Err(code) => return code,
    };
    let bytes = match read_bytes(&memory, &env, ptr as u32 as u64, len) {
        Ok(bytes) => bytes,
        Err(code) => return code,
    };
    let Ok(source) = std::str::from_utf8(&bytes) else {
        return ERR_SHADER;
    };
    let session = &env.data().session;
    if !session.can_allocate_pipeline() {
        return ERR_QUOTA;
    }
    match session.bridge().create_pipeline(source) {
        Ok(handle) => {
            session.track_pipeline(handle);
            handle
        }
        Err(code) => code,
    }
}

pub fn pipeline_dispatch(
    env: FunctionEnvMut<HostEnv>,
    pipeline: i32,
    buffer: i32,
    x: i32,
    y: i32,
    z: i32,
) -> i32 {
    let dims = [x as u32, y as u32, z as u32];
    if dims
        .iter()
        .any(|&d| d == 0 || d > MAX_WORKGROUPS_PER_DIMENSION)
    {
        return ERR_RANGE;
    }
    let total = dims.iter().map(|&d| d as u64).product::<u64>();
    if total > MAX_WORKGROUPS_PER_DISPATCH {
        return ERR_RANGE;
    }
    let session = &env.data().session;
    if !session.owns_pipeline(pipeline) || !session.owns_buffer(buffer) {
        return ERR_HANDLE;
    }
    session.bridge().dispatch(pipeline, buffer, dims)
}

pub fn pipeline_release(env: FunctionEnvMut<HostEnv>, handle: i32) -> i32 {
    let session = &env.data().session;
    if !session.forget_pipeline(handle) {
        return ERR_HANDLE;
    }
    session.bridge().release_pipeline(handle)
}

// ---------------------------------------------------------------------------
// Per-instance session: ownership and quotas
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GpuSession {
    inner: Arc<Mutex<GpuSessionInner>>,
}

struct GpuSessionInner {
    bridge: GpuBridge,
    buffers: BTreeMap<i32, usize>,
    pipelines: BTreeSet<i32>,
    live_bytes: usize,
    limits: SessionLimits,
}

impl Drop for GpuSessionInner {
    fn drop(&mut self) {
        for handle in std::mem::take(&mut self.buffers).into_keys() {
            let _ = self.bridge.release_buffer(handle);
        }
        for handle in std::mem::take(&mut self.pipelines) {
            let _ = self.bridge.release_pipeline(handle);
        }
        self.live_bytes = 0;
    }
}

impl GpuSession {
    pub fn new(bridge: GpuBridge, limits: SessionLimits) -> Self {
        Self {
            inner: Arc::new(Mutex::new(GpuSessionInner {
                bridge,
                buffers: BTreeMap::new(),
                pipelines: BTreeSet::new(),
                live_bytes: 0,
                limits,
            })),
        }
    }

    fn bridge(&self) -> GpuBridge {
        self.lock().bridge.clone()
    }

    fn can_allocate_buffer(&self, len: usize) -> bool {
        let inner = self.lock();
        inner.buffers.len() < inner.limits.max_buffers
            && inner
                .live_bytes
                .checked_add(len)
                .is_some_and(|bytes| bytes <= inner.limits.max_bytes)
    }

    fn can_allocate_pipeline(&self) -> bool {
        let inner = self.lock();
        inner.pipelines.len() < inner.limits.max_pipelines
    }

    fn track_buffer(&self, handle: i32, len: usize) {
        let mut inner = self.lock();
        inner.live_bytes += len;
        inner.buffers.insert(handle, len);
    }

    fn track_pipeline(&self, handle: i32) {
        self.lock().pipelines.insert(handle);
    }

    fn owns_buffer(&self, handle: i32) -> bool {
        self.lock().buffers.contains_key(&handle)
    }

    fn owns_pipeline(&self, handle: i32) -> bool {
        self.lock().pipelines.contains(&handle)
    }

    fn forget_buffer(&self, handle: i32) -> bool {
        let mut inner = self.lock();
        match inner.buffers.remove(&handle) {
            Some(len) => {
                inner.live_bytes = inner.live_bytes.saturating_sub(len);
                true
            }
            None => false,
        }
    }

    fn forget_pipeline(&self, handle: i32) -> bool {
        self.lock().pipelines.remove(&handle)
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GpuSessionInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

// ---------------------------------------------------------------------------
// Device bridge: wgpu resources keyed by opaque handles
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct GpuBridge {
    inner: Arc<Mutex<GpuBridgeInner>>,
}

impl std::fmt::Debug for GpuBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("GpuBridge")
            .field("adapter", &inner.adapter_name)
            .field("backend", &inner.backend)
            .field("live_buffers", &inner.buffers.len())
            .field("live_pipelines", &inner.pipelines.len())
            .finish()
    }
}

struct GpuBridgeInner {
    adapter_name: String,
    backend: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    next_handle: i32,
    buffers: BTreeMap<i32, GpuBuffer>,
    pipelines: BTreeMap<i32, GpuPipeline>,
    timing: Timing,
}

struct GpuBuffer {
    buffer: wgpu::Buffer,
    len: usize,
}

struct GpuPipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

/// Accumulated device-side wall-clock costs since the last [`GpuBridge::drain_timing`].
/// Dispatch samples are recorded individually so cold and warm submissions can
/// be told apart instead of summed.
#[derive(Clone, Debug, Default)]
pub struct Timing {
    pub upload: Duration,
    pub pipeline_create: Duration,
    pub dispatches: Vec<Duration>,
    pub readback: Duration,
}

#[derive(Clone, Debug)]
pub struct BridgeSnapshot {
    pub adapter_name: String,
    pub backend: String,
    pub live_buffers: usize,
    pub live_pipelines: usize,
}

impl GpuBridge {
    pub fn new() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .context("no hardware GPU adapter available")?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            bail!("wgpu selected a CPU adapter; refusing to present this as hardware GPU");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("wasmer-gpu-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
            },
            None,
        ))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(GpuBridgeInner {
                adapter_name: info.name,
                backend: format!("{:?}", info.backend),
                device,
                queue,
                next_handle: 1,
                buffers: BTreeMap::new(),
                pipelines: BTreeMap::new(),
                timing: Timing::default(),
            })),
        })
    }

    pub fn snapshot(&self) -> BridgeSnapshot {
        let inner = self.lock();
        BridgeSnapshot {
            adapter_name: inner.adapter_name.clone(),
            backend: inner.backend.clone(),
            live_buffers: inner.buffers.len(),
            live_pipelines: inner.pipelines.len(),
        }
    }

    pub fn drain_timing(&self) -> Timing {
        std::mem::take(&mut self.lock().timing)
    }

    fn upload(&self, bytes: &[u8]) -> std::result::Result<i32, i32> {
        let start = Instant::now();
        let mut inner = self.lock();
        push_error_scopes(&inner.device);
        let buffer = inner
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("guest-upload"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        inner.device.poll(wgpu::Maintain::Wait);
        if let Some(code) = pop_error_scopes(&inner.device, ERR_GPU) {
            return Err(code);
        }
        let handle = inner.allocate_handle();
        inner.buffers.insert(
            handle,
            GpuBuffer {
                buffer,
                len: bytes.len(),
            },
        );
        inner.timing.upload += start.elapsed();
        Ok(handle)
    }

    fn create_pipeline(&self, source: &str) -> std::result::Result<i32, i32> {
        let start = Instant::now();
        let mut inner = self.lock();
        push_error_scopes(&inner.device);
        let module = inner
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("guest-shader"),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = inner
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("guest-layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let pipeline_layout =
            inner
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("guest-pipeline-layout"),
                    bind_group_layouts: &[&layout],
                    push_constant_ranges: &[],
                });
        let pipeline = inner
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("guest-pipeline"),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: ENTRY_POINT,
            });
        inner.device.poll(wgpu::Maintain::Wait);
        if let Some(code) = pop_error_scopes(&inner.device, ERR_SHADER) {
            return Err(code);
        }
        let handle = inner.allocate_handle();
        inner
            .pipelines
            .insert(handle, GpuPipeline { pipeline, layout });
        inner.timing.pipeline_create += start.elapsed();
        Ok(handle)
    }

    fn dispatch(&self, pipeline: i32, buffer: i32, dims: [u32; 3]) -> i32 {
        let start = Instant::now();
        let mut inner = self.lock();
        let (Some(gpu_pipeline), Some(gpu_buffer)) =
            (inner.pipelines.get(&pipeline), inner.buffers.get(&buffer))
        else {
            return ERR_HANDLE;
        };
        push_error_scopes(&inner.device);
        let bind_group = inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("guest-bind-group"),
            layout: &gpu_pipeline.layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: gpu_buffer.buffer.as_entire_binding(),
            }],
        });
        let mut encoder = inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("guest-dispatch"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("guest-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&gpu_pipeline.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(dims[0], dims[1], dims[2]);
        }
        inner.queue.submit(Some(encoder.finish()));
        inner.device.poll(wgpu::Maintain::Wait);
        if let Some(code) = pop_error_scopes(&inner.device, ERR_GPU) {
            return code;
        }
        inner.timing.dispatches.push(start.elapsed());
        SUCCESS
    }

    fn read_buffer(&self, handle: i32, len: usize) -> std::result::Result<Vec<u8>, i32> {
        let start = Instant::now();
        let mut inner = self.lock();
        let Some(gpu_buffer) = inner.buffers.get(&handle) else {
            return Err(ERR_HANDLE);
        };
        if !valid_transfer_len(len) || gpu_buffer.len < len {
            return Err(ERR_RANGE);
        }
        push_error_scopes(&inner.device);
        let staging = inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-staging"),
            size: len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = inner
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("readback-encoder"),
            });
        encoder.copy_buffer_to_buffer(&gpu_buffer.buffer, 0, &staging, 0, len as u64);
        inner.queue.submit(Some(encoder.finish()));
        let slice = staging.slice(..);
        let completed = Arc::new(AtomicBool::new(false));
        let ok = Arc::new(AtomicBool::new(false));
        let completed_callback = Arc::clone(&completed);
        let ok_callback = Arc::clone(&ok);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            ok_callback.store(result.is_ok(), Ordering::Release);
            completed_callback.store(true, Ordering::Release);
        });
        while !completed.load(Ordering::Acquire) {
            inner.device.poll(wgpu::Maintain::Wait);
        }
        if let Some(code) = pop_error_scopes(&inner.device, ERR_GPU) {
            return Err(code);
        }
        if !ok.load(Ordering::Acquire) {
            return Err(ERR_GPU);
        }
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        inner.timing.readback += start.elapsed();
        Ok(data)
    }

    fn write_buffer(&self, handle: i32, offset: usize, bytes: &[u8]) -> i32 {
        let start = Instant::now();
        let mut inner = self.lock();
        let Some(gpu_buffer) = inner.buffers.get(&handle) else {
            return ERR_HANDLE;
        };
        let Some(end) = offset.checked_add(bytes.len()) else {
            return ERR_RANGE;
        };
        if end > gpu_buffer.len {
            return ERR_RANGE;
        }
        push_error_scopes(&inner.device);
        inner
            .queue
            .write_buffer(&gpu_buffer.buffer, offset as u64, bytes);
        inner.queue.submit(std::iter::empty());
        inner.device.poll(wgpu::Maintain::Wait);
        if let Some(code) = pop_error_scopes(&inner.device, ERR_GPU) {
            return code;
        }
        inner.timing.upload += start.elapsed();
        SUCCESS
    }

    fn release_buffer(&self, handle: i32) -> i32 {
        if self.lock().buffers.remove(&handle).is_some() {
            SUCCESS
        } else {
            ERR_HANDLE
        }
    }

    fn release_pipeline(&self, handle: i32) -> i32 {
        if self.lock().pipelines.remove(&handle).is_some() {
            SUCCESS
        } else {
            ERR_HANDLE
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GpuBridgeInner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl GpuBridgeInner {
    /// One counter for both resource kinds, so a buffer handle can never alias
    /// a pipeline handle; kind confusion is caught by the per-kind maps.
    fn allocate_handle(&mut self) -> i32 {
        let handle = self.next_handle;
        self.next_handle += 1;
        handle
    }
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Guest pointers and lengths are wasm32 unsigned values carried in `i32`.
fn guest_transfer_range(ptr: i32, len: i32) -> std::result::Result<(u64, usize), i32> {
    let len = len as u32 as usize;
    if !valid_transfer_len(len) {
        return Err(ERR_RANGE);
    }
    Ok((ptr as u32 as u64, len))
}

fn valid_transfer_len(len: usize) -> bool {
    len > 0 && len % TRANSFER_ALIGNMENT == 0
}

fn push_error_scopes(device: &wgpu::Device) {
    device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    device.push_error_scope(wgpu::ErrorFilter::Validation);
}

/// Pops both scopes. Validation failures map to `validation_code`; allocation
/// failures always map to [`ERR_GPU`].
fn pop_error_scopes(device: &wgpu::Device, validation_code: i32) -> Option<i32> {
    let validation = pollster::block_on(device.pop_error_scope());
    let out_of_memory = pollster::block_on(device.pop_error_scope());
    // Captured errors become return codes for the guest; the message goes to
    // stderr so an operator can see why a shader or dispatch was refused.
    if let Some(error) = &out_of_memory {
        eprintln!("wasmer-gpu: captured wgpu out-of-memory error: {error}");
        return Some(ERR_GPU);
    }
    if let Some(error) = &validation {
        eprintln!("wasmer-gpu: captured wgpu validation error: {error}");
        return Some(validation_code);
    }
    None
}

pub fn read_bytes(
    memory: &Memory,
    store: &impl wasmer::AsStoreRef,
    offset: u64,
    len: usize,
) -> std::result::Result<Vec<u8>, i32> {
    let view = memory.view(store);
    let end = offset.checked_add(len as u64).ok_or(ERR_RANGE)?;
    if end > view.data_size() {
        return Err(ERR_RANGE);
    }
    let mut bytes = vec![0; len];
    view.read(offset, &mut bytes).map_err(|_| ERR_RANGE)?;
    Ok(bytes)
}

pub fn write_bytes(
    memory: &Memory,
    store: &impl wasmer::AsStoreRef,
    offset: u64,
    bytes: &[u8],
) -> std::result::Result<(), i32> {
    let view = memory.view(store);
    let end = offset.checked_add(bytes.len() as u64).ok_or(ERR_RANGE)?;
    if end > view.data_size() {
        return Err(ERR_RANGE);
    }
    view.write(offset, bytes).map_err(|_| ERR_RANGE)
}
