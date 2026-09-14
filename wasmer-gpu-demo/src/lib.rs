//! `wasmer-gpu-demo`: a bounded GPU compute bridge for Wasmer guests.
//!
//! - [`bridge`] holds the kernel-agnostic `wasmer_gpu_v0` host imports, the
//!   per-instance session (quotas, handle ownership, reclamation) and the
//!   wgpu device bridge.
//! - [`hook`] registers those imports through a WASIX `InstantiationHook`.
//!
//! Binaries in this crate (`gpu-smoke`, and any further demo runner) build on
//! these two modules; kernels and guests live outside them.

pub mod bridge;
pub mod hook;
