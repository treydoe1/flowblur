//! T Blur: Adobe Premiere Pro / After Effects effect shell around `tblur_core`.
//!
//! * `pixels`    – host pixel buffers <-> premultiplied RGBA f32 (pure, unit-tested)
//! * `neighbors` – choosing prev/next frames and frame distances (pure, unit-tested)
//! * `source_cache` – process-global per-clip record: source-rate verdict, current-checkout match (Premiere)
//! * `effect`    – the Adobe glue (entry point `EffectMain`, parameters, checkout, render)
//! * `diag`      – opt-in diagnostics log (/tmp/tblur-debug -> /tmp/tblur-debug.log)
//! * `metal_io`  – macOS Metal buffer synchronization, readback and writeback (Premiere GPU path)
//! * `gpu_filter` – hand-written panic-safe Premiere GPU filter (`xGPUFilterEntry`, adjustment layers)
//! * `update_check` – once-a-day check for a newer release on GitHub (notice only, never installs)
/// The effect's version, from `version` in Cargo.toml (the only place it is written). Shown in
/// the About box and in every diagnostics line as `version=<x.y.z>`. build.rs encodes the same
/// Cargo version into the PiPL `AE_Effect_Version`, which `define_effect!` reports to the host.
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod neighbors;
pub mod pixels;
mod diag;
mod instance_registry;
mod source_cache;
mod effect;
#[cfg(target_os = "macos")]
pub mod metal_io;
#[cfg(target_os = "macos")]
pub mod gpu_filter;
#[cfg(target_os = "macos")]
mod update_check;
