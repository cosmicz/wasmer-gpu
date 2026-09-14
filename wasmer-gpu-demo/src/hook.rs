//! WASIX integration: registers the bridge imports through
//! `wasmer_wasix::runtime::InstantiationHook`, the same seam `wasmer-napi`
//! uses, so a WASIX command guest gets `wasmer_gpu_v0` next to its WASI
//! imports without the runner touching the import object by hand.
//!
//! One [`GpuSession`] is created per instance in `prepare_imports`; the
//! guest's memory (imported or exported) is bound in `configure_new_instance`.
//! Dropping the store drops the `FunctionEnv`, which drops the session and
//! releases everything the instance still owns.

use crate::bridge::{self, GpuBridge, GpuSession, HostEnv, SessionLimits};
use anyhow::{Context, Result};
use wasmer::{FunctionEnv, Imports, Instance, Memory, Module, StoreMut};
use wasmer_wasix::runtime::{InstantiationHook, InstantiationState};

#[derive(Debug)]
pub struct GpuInstantiationHook {
    bridge: GpuBridge,
    limits: SessionLimits,
}

/// Routed from the import phase to the setup phase of the same instance.
struct GpuInstanceState {
    env: FunctionEnv<HostEnv>,
}

impl GpuInstantiationHook {
    pub fn new(bridge: GpuBridge, limits: SessionLimits) -> Self {
        Self { bridge, limits }
    }
}

impl InstantiationHook for GpuInstantiationHook {
    fn prepare_imports(
        &self,
        module: &Module,
        store: &mut StoreMut,
        imports: &mut Imports,
    ) -> Result<InstantiationState> {
        if !bridge::module_uses_bridge(module) {
            return Ok(InstantiationState::empty());
        }
        let session = GpuSession::new(self.bridge.clone(), self.limits);
        let env = FunctionEnv::new(
            store,
            HostEnv {
                memory: None,
                session,
            },
        );
        bridge::register_imports(store, &env, imports);
        Ok(InstantiationState::new(GpuInstanceState { env }))
    }

    fn configure_new_instance(
        &self,
        module: &Module,
        store: &mut StoreMut,
        instance: &Instance,
        imported_memory: Option<&Memory>,
        state: InstantiationState,
    ) -> Result<()> {
        if !bridge::module_uses_bridge(module) {
            return Ok(());
        }
        let state = state.take::<GpuInstanceState>().context(
            "missing wasmer-gpu instance state (imports were not prepared by this hook)",
        )?;
        let memory = match imported_memory {
            Some(memory) => memory.clone(),
            None => instance
                .exports
                .get_memory("memory")
                .context("wasmer-gpu guest has neither an imported nor an exported memory")?
                .clone(),
        };
        state.env.as_mut(store).memory = Some(memory);
        Ok(())
    }
}
