//! Hand-written, panic-safe Premiere Pro GPU filter (`xGPUFilterEntry`) for adjustment layers.
//!
//! Fetches neighbour frames via Premiere's GPU dependency graph (`GetFrameDependencies` -> Metal
//! `MTLBuffer`s) instead of slow CPU `PF_CHECKOUT_PARAM` calls, then runs the exact same
//! neighbour-selection rule (`effect::select_legacy_neighbors`) and CPU optical-flow engine
//! (`tblur_core::render_with_owned_luma`) as the CPU path so the rendered image is bit-identical.
//!
//! Safety architecture (intentionally avoids `premiere::define_gpu_filter!`):
//! * Every FFI entry point and callback wraps its entire body in `catch_unwind` + `autoreleasepool`.
//! * `CreateInstance` stores `Box::into_raw(Box::new(GpuFilterInstance))`; `GetFrameDependencies`
//!   and `Render` borrow `&GpuFilterInstance` immutably (safe under concurrent host callbacks);
//!   `DisposeInstance` nulls `ioPrivatePluginData` BEFORE `Box::from_raw`.
//! * Every host pointer, suite function pointer, `MTLBuffer` type, frame count, and buffer length
//!   is validated before allocating output frames or reading memory.
//! * `Render` never writes into any input frame (`inFrames[0..N]`), returns `suiteError_Fail` on
//!   any failure (so Premiere falls back to the CPU effect path rather than outputting an
//!   unblurred frame), returns `suiteError_NoError` only when `*outFrame` holds the rendered
//!   result, a valid no-blur copy (blur/sensitivity <= 0 or no usable neighbour), or the debug
//!   `passthrough` fault mode, disposes any `CreateGPUPPix` frame via RAII on error/panic unless
//!   in-flight GPU work still targets it (in which case it deliberately leaks and logs), and
//!   bounds total GPU waits per `Render` to one timeout.

use std::cell::Cell;
use std::collections::HashSet;
use std::ffi::{c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use tblur_core as core;
use objc2::rc::autoreleasepool;
use premiere::sys;

use crate::diag;
use crate::VERSION;
use crate::effect;
use crate::instance_registry;
use crate::metal_io::{MetalContext, MetalError, MetalFrameDesc, objc_class_name};
use crate::neighbors::{MAX_K, Outcome};
use crate::pixels::{self, Layout};

/// Default number of whole sequence-frame steps declared in each direction by
/// `GetFrameDependencies`. Matches `neighbors::MAX_K` (4) so the GPU path can skip up to 3
/// duplicate frames on each side, matching the CPU legacy path. Declares `2 * dep_steps` input
/// frames in interleaved order (`-1, +1, -2, +2, ...`); only frames inspected by
/// `select_legacy_neighbors` are read back from Metal.
///
/// May be overridden in debug mode (`CreateInstance` only) via `/tmp/tblur-gpu-steps`
/// containing a single digit `'1'..='4'` when `/tmp/tblur-debug` also exists.
pub const GPU_DEPENDENCY_STEPS: u32 = MAX_K;

/// Per-instance state stored in `PrGPUFilterInstance::ioPrivatePluginData`.
/// Immutable after `CreateInstance`, so concurrent callbacks borrowing `&GpuFilterInstance` are
/// data-race-free.
struct GpuFilterInstance {
    device_index: u32,
    timeline_id: sys::PrTimelineID,
    node_id: i32,
    dep_steps: u32,
    fault_mode: diag::GpuFaultMode,
    metal: MetalContext,
    #[cfg(test)]
    kill_switch_path: Option<std::path::PathBuf>,
}

impl GpuFilterInstance {
    fn is_gpu_disabled(&self) -> bool {
        #[cfg(test)]
        if let Some(path) = self.kill_switch_path.as_deref() {
            return diag::gpu_disabled_from(path);
        }
        diag::gpu_disabled()
    }
}

// ───────────────────────────── safe host pointer & suite helpers ─────────────────────────────

/// Extracts `SPBasicSuite*` from a Premiere `piSuitesPtr`, checking every link for null.
unsafe fn sp_basic_from_pi_suites(pi_suites: sys::piSuitesPtr) -> Option<*const sys::SPBasicSuite> {
    if pi_suites.is_null() {
        return None;
    }
    // SAFETY: `pi_suites` is non-null; `piSuites` is `#[repr(C, packed)]` so we use unaligned read.
    let util_funcs = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*pi_suites).utilFuncs)) };
    if util_funcs.is_null() {
        return None;
    }
    // SAFETY: `util_funcs` is non-null; `piSuites::utilFuncs` is a host-provided struct pointer.
    let get_sp =
        unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*util_funcs).getSPBasicSuite)) }?;
    // SAFETY: Calling host-provided `getSPBasicSuite` function pointer.
    let sp = unsafe { get_sp() };
    (!sp.is_null()).then_some(sp)
}

/// RAII wrapper around a Premiere suite acquired via `SPBasicSuite::AcquireSuite`.
/// Releases the suite on drop without panicking.
struct RawSuite<T> {
    sp: *const sys::SPBasicSuite,
    name: &'static [u8],
    version: i32,
    ptr: *const T,
}

impl<T> RawSuite<T> {
    unsafe fn acquire(
        sp: *const sys::SPBasicSuite,
        name: &'static [u8],
        version: u32,
    ) -> Option<Self> {
        if sp.is_null() {
            return None;
        }
        // SAFETY: `sp` is checked non-null.
        let acquire_fn = unsafe { (*sp).AcquireSuite }?;
        let mut raw: *const c_void = std::ptr::null();
        // SAFETY: `name` is a NUL-terminated suite identifier from `premiere::sys`.
        let err = unsafe { acquire_fn(name.as_ptr() as *const c_char, version as i32, &mut raw) };
        if err != 0 || raw.is_null() {
            if err == 0
                && let Some(rel) = unsafe { (*sp).ReleaseSuite }
            {
                // SAFETY: Release if AcquireSuite returned 0 with a null pointer defensively.
                unsafe {
                    let _ = rel(name.as_ptr() as *const c_char, version as i32);
                }
            }
            return None;
        }
        Some(Self {
            sp,
            name,
            version: version as i32,
            ptr: raw as *const T,
        })
    }

    /// Acquires the highest supported version in `min_ver..=max_ver`.
    unsafe fn acquire_any(
        sp: *const sys::SPBasicSuite,
        name: &'static [u8],
        min_ver: u32,
        max_ver: u32,
    ) -> Option<Self> {
        for v in (min_ver..=max_ver).rev() {
            if let Some(s) = unsafe { Self::acquire(sp, name, v) } {
                return Some(s);
            }
        }
        None
    }
}

impl<T> Drop for RawSuite<T> {
    fn drop(&mut self) {
        // SAFETY: `self.sp` is valid for the duration of the host callback and `self.name` was
        // acquired with `self.version`.
        unsafe {
            if let Some(release_fn) = (*self.sp).ReleaseSuite {
                let _ = release_fn(self.name.as_ptr() as *const c_char, self.version);
            }
        }
    }
}

/// RAII guard that releases an acquired video segment node ID via `ReleaseVideoNodeID`.
struct VideoNodeGuard {
    release_fn: unsafe extern "C" fn(sys::csSDK_int32) -> sys::prSuiteError,
    node_id: sys::csSDK_int32,
}

impl Drop for VideoNodeGuard {
    fn drop(&mut self) {
        // SAFETY: `self.node_id` was acquired from the same `PrSDKVideoSegmentSuite` instance
        // whose `ReleaseVideoNodeID` pointer is still valid while the suite guard is alive.
        unsafe {
            let _ = (self.release_fn)(self.node_id);
        }
    }
}

/// RAII guard that releases an acquired `VideoSegmentsID` via `ReleaseVideoSegmentsID`.
struct VideoSegmentsGuard {
    release_fn: unsafe extern "C" fn(sys::csSDK_int32) -> sys::prSuiteError,
    segs_id: sys::csSDK_int32,
}

impl Drop for VideoSegmentsGuard {
    fn drop(&mut self) {
        // SAFETY: `self.segs_id` was acquired from the same `PrSDKVideoSegmentSuite` instance
        // whose `ReleaseVideoSegmentsID` pointer is still valid while the suite guard is alive.
        unsafe {
            let _ = (self.release_fn)(self.segs_id);
        }
    }
}

enum GetPtrSizeFn {
    SuiteU32(unsafe extern "C" fn(sys::PrMemoryPtr) -> sys::csSDK_uint32),
    PlugI32(unsafe extern "C" fn(sys::PrMemoryPtr) -> i32),
}

/// RAII guard that disposes a `PrMemoryPtr` returned by `PrSDKVideoSegmentSuite::GetNodeProperty`
/// and exposes its allocation size via `GetPtrSize` (preferring `PrSDKMemoryManagerSuite` v3..=v4,
/// falling back to `piSuites->memFuncs`).
struct PrMemPtrGuard {
    ptr: sys::PrMemoryPtr,
    dispose_fn: Option<unsafe extern "C" fn(sys::PrMemoryPtr)>,
    get_size_fn: Option<GetPtrSizeFn>,
    _mem_suite: Option<RawSuite<sys::PrSDKMemoryManagerSuite>>,
}

impl PrMemPtrGuard {
    unsafe fn new(
        sp: *const sys::SPBasicSuite,
        pi_suites: sys::piSuitesPtr,
        ptr: sys::PrMemoryPtr,
    ) -> Self {
        let mem_suite = unsafe {
            RawSuite::<sys::PrSDKMemoryManagerSuite>::acquire_any(
                sp,
                sys::kPrSDKMemoryManagerSuite,
                sys::kPrSDKMemoryManagerSuiteVersion3,
                sys::kPrSDKMemoryManagerSuiteVersion,
            )
        };
        let mut dispose_fn = mem_suite.as_ref().and_then(|ms| unsafe {
            std::ptr::read_unaligned(std::ptr::addr_of!((*ms.ptr).PrDisposePtr))
        });
        let mut get_size_fn = mem_suite
            .as_ref()
            .and_then(|ms| unsafe {
                std::ptr::read_unaligned(std::ptr::addr_of!((*ms.ptr).GetPtrSize))
            })
            .map(GetPtrSizeFn::SuiteU32);
        if (dispose_fn.is_none() || get_size_fn.is_none()) && !pi_suites.is_null() {
            let mem_funcs =
                unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*pi_suites).memFuncs)) };
            if !mem_funcs.is_null() {
                if dispose_fn.is_none() {
                    dispose_fn = unsafe {
                        std::ptr::read_unaligned(std::ptr::addr_of!((*mem_funcs).disposePtr))
                    };
                }
                if get_size_fn.is_none() {
                    get_size_fn = unsafe {
                        std::ptr::read_unaligned(std::ptr::addr_of!((*mem_funcs).getPtrSize))
                    }
                    .map(GetPtrSizeFn::PlugI32);
                }
            }
        }
        Self {
            ptr,
            dispose_fn,
            get_size_fn,
            _mem_suite: mem_suite,
        }
    }

    /// Queries the host allocation size in bytes. Returns `None` if `GetPtrSize` is unavailable
    /// or returns <= 0.
    fn alloc_size(&self) -> Option<usize> {
        if self.ptr.is_null() {
            return None;
        }
        let n: usize = match self.get_size_fn.as_ref()? {
            // SAFETY: `self.ptr` is a live host `PrMemoryPtr` owned by this guard.
            GetPtrSizeFn::SuiteU32(f) => (unsafe { f(self.ptr) }) as usize,
            GetPtrSizeFn::PlugI32(f) => {
                let s = unsafe { f(self.ptr) };
                if s <= 0 {
                    return None;
                }
                s as usize
            }
        };
        (n > 0).then_some(n)
    }
}

impl Drop for PrMemPtrGuard {
    fn drop(&mut self) {
        if !self.ptr.is_null()
            && let Some(dispose_fn) = self.dispose_fn
        {
            // SAFETY: `self.ptr` was allocated by Premiere via `PrNewPtr` (`GetNodeProperty`),
            // and `dispose_fn` stays valid while `_mem_suite` / `pi_suites` is alive.
            unsafe {
                dispose_fn(self.ptr);
            }
            self.ptr = std::ptr::null_mut();
        }
    }
}

/// RAII guard for a GPU `PPixHand` allocated via `CreateGPUPPix`.
///
/// Disposes the handle on every exit path (including panic unwind) UNLESS:
/// * `take()` transfers ownership to the host via `*out_frame`, OR
/// * `leak_in_flight()` disarms disposal because an in-flight timed-out GPU command buffer still
///   targets this buffer (preventing GPU use-after-dispose).
struct OwnedGpuPPix {
    dispose_fn: Option<unsafe extern "C" fn(sys::PPixHand) -> sys::prSuiteError>,
    ppix: sys::PPixHand,
}

impl OwnedGpuPPix {
    unsafe fn new(ppix_suite: *const sys::PrSDKPPixSuite, ppix: sys::PPixHand) -> Self {
        // SAFETY: `ppix_suite` is a live acquired `PrSDKPPixSuite*` that outlives this guard.
        let dispose_fn =
            unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*ppix_suite).Dispose)) };
        Self { dispose_fn, ppix }
    }

    fn handle(&self) -> sys::PPixHand {
        self.ppix
    }

    /// Transfers ownership of the `PPixHand` to the host (`*out_frame`), disarming `Drop`.
    fn take(mut self) -> sys::PPixHand {
        let h = self.ppix;
        self.ppix = std::ptr::null_mut();
        h
    }

    /// Deliberately leaks the `PPixHand` without calling `Dispose`, because in-flight GPU work
    /// still references its backing `MTLBuffer` after a wait timeout.
    fn leak_in_flight(mut self) {
        self.ppix = std::ptr::null_mut();
    }
}

impl Drop for OwnedGpuPPix {
    fn drop(&mut self) {
        if !self.ppix.is_null()
            && let Some(dispose_fn) = self.dispose_fn
        {
            // SAFETY: `self.ppix` was allocated by `CreateGPUPPix`, not yet transferred or
            // disposed, and `ppix_suite` outlives this guard (LIFO drop order).
            unsafe {
                let _ = dispose_fn(self.ppix);
            }
            self.ppix = std::ptr::null_mut();
        }
    }
}

fn framework_name(fw: sys::PrGPUDeviceFramework) -> &'static str {
    match fw {
        sys::PrGPUDeviceFramework_PrGPUDeviceFramework_CUDA => "CUDA",
        sys::PrGPUDeviceFramework_PrGPUDeviceFramework_OpenCL => "OpenCL",
        sys::PrGPUDeviceFramework_PrGPUDeviceFramework_Metal => "Metal",
        sys::PrGPUDeviceFramework_PrGPUDeviceFramework_DirectX => "DirectX",
        _ => "Unknown",
    }
}

fn pixel_format_info(raw: sys::PrPixelFormat) -> (&'static str, Option<Layout>) {
    if raw == sys::PrPixelFormatGpu_GPU_BGRA_4444_32f {
        ("GPU_BGRA_4444_32f", Some(Layout::Bgra32f))
    } else if raw == sys::PrPixelFormatGpu_GPU_BGRA_4444_16f {
        ("GPU_BGRA_4444_16f", Some(Layout::Bgra16f))
    } else if raw == sys::PrPixelFormat_PrPixelFormat_BGRA_4444_32f {
        ("BGRA_4444_32f", None)
    } else if raw == sys::PrPixelFormat_PrPixelFormat_BGRA_4444_8u {
        ("BGRA_4444_8u", None)
    } else if raw == sys::PrPixelFormat_PrPixelFormat_Invalid {
        ("Invalid", None)
    } else {
        ("Other", None)
    }
}

// ───────────────────────────── GPU scope gate & graph diagnostics ─────────────────────────────

/// SDK node type strings from `PrSDKVideoSegmentSuite.h` (via `premiere-sys`).
const NODE_TYPE_ADJUSTMENT: &str = "RenderableNode_AdjustmentImpl";
const NODE_TYPE_ADJUSTMENT_EFFECT: &str = "RenderableNode_AdjustmentEffectImpl";
const NODE_TYPE_EFFECT: &str = "RenderableNodeEffectImpl";

/// Result of inspecting a GPU filter's `inNodeID`, its `EffectNode::RuntimeInstanceID` property,
/// its operator owner node, and the owner's primary input node via `PrSDKVideoSegmentSuite`.
#[derive(Clone, Debug)]
pub struct NodeClassification {
    pub is_adjustment: bool,
    pub matched_by: &'static str,
    pub node_type: String,
    pub node_flags: u32,
    pub node_hash: [u8; 37],
    pub runtime_instance_id_raw: Option<String>,
    pub runtime_instance_id: Option<i32>,
    pub runtime_id_err: Option<i32>,
    pub owner_id: Option<i32>,
    pub owner_type: Option<String>,
    pub owner_flags: Option<u32>,
    pub owner_err: Option<i32>,
    pub owner_input0_type: Option<String>,
    pub owner_input0_flags: Option<u32>,
}

/// Pure gate decision for accepting or declining a GPU filter instance in `CreateInstance`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GateDecision {
    pub accept: bool,
    pub reason: &'static str,
    pub forced: &'static str,
}

/// Premiere SDK `suiteError_IDNotValid` (`0x80000005` as `i32`), returned by
/// `AcquireOperatorOwnerNodeID` on Premiere 2027 adjustment-layer effect nodes.
pub const OWNER_ID_NOT_VALID: i32 = -2147483643;

/// Pure decision table for the GPU filter scope gate (after the kill switch and framework gates).
///
/// Priority:
/// 1. Debug-only force switch (`/tmp/tblur-gpu-force`, only when `/tmp/tblur-debug` exists):
///    - `GpuForceMode::All` -> accept everything (`forced="all"`).
///    - `GpuForceMode::Adjust` -> accept effect nodes whose operator-owner lookup failed with
///      `OWNER_ID_NOT_VALID` (`-2147483643`, no owner), matching Premiere 2027 adjustment-layer
///      effects (`forced="adjust"`).
/// 2. PF synthetic map authority (`instance_registry`) + adjustment-layer node signature:
///    - `runtime_instance_id == None` -> decline (`reason="no_runtime_instance_id"`).
///    - `pf_synthetic == Some(true)` AND effect node with failed operator-owner lookup
///      (`node_type == "RenderableNodeEffectImpl"`, `owner_id.is_none()`, `owner_err.is_some()`)
///      -> accept (`reason="pf_synthetic"`).
///    - `pf_synthetic == Some(true)` on a node that does NOT match the adjustment-layer signature
///      (e.g. stale/reused instance ID on a clip-owned effect) -> decline
///      (`reason="not_adjustment_node"`).
///    - `pf_synthetic == Some(false)` -> decline (`reason="pf_not_synthetic"`).
///    - `pf_synthetic == None` -> decline (`reason="pf_unknown_id"`), letting CPU render record
///      the instance so a subsequent `CreateInstance` can accept.
pub fn decide_gpu_gate(
    force_mode: diag::GpuForceMode,
    node_type: &str,
    owner_id: Option<i32>,
    owner_err: Option<i32>,
    runtime_instance_id: Option<i32>,
    pf_synthetic: Option<bool>,
) -> GateDecision {
    let has_adjustment_owner_fail =
        node_type == NODE_TYPE_EFFECT && owner_id.is_none() && owner_err.is_some();
    let has_exact_id_not_valid = node_type == NODE_TYPE_EFFECT
        && owner_id.is_none()
        && owner_err == Some(OWNER_ID_NOT_VALID);

    match force_mode {
        diag::GpuForceMode::All => {
            return GateDecision {
                accept: true,
                reason: "forced_all",
                forced: "all",
            };
        }
        diag::GpuForceMode::Adjust => {
            if has_exact_id_not_valid {
                return GateDecision {
                    accept: true,
                    reason: "forced_adjust",
                    forced: "adjust",
                };
            }
        }
        diag::GpuForceMode::None => {}
    }

    if runtime_instance_id.is_none() {
        return GateDecision {
            accept: false,
            reason: "no_runtime_instance_id",
            forced: "none",
        };
    }

    match pf_synthetic {
        Some(true) if has_adjustment_owner_fail => GateDecision {
            accept: true,
            reason: "pf_synthetic",
            forced: "none",
        },
        Some(true) => GateDecision {
            accept: false,
            reason: "not_adjustment_node",
            forced: "none",
        },
        Some(false) => GateDecision {
            accept: false,
            reason: "pf_not_synthetic",
            forced: "none",
        },
        None => GateDecision {
            accept: false,
            reason: "pf_unknown_id",
            forced: "none",
        },
    }
}

/// Parses Premiere's `EffectNode::RuntimeInstanceID` UTF-8 property string into an `i32` matching
/// `PFUtilitySuite::GetFilterInstanceID`.
pub fn parse_runtime_instance_id(raw: &str) -> Option<i32> {
    let s = raw.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(v) = s.parse::<i32>() {
        return Some(v);
    }
    if let Ok(v) = s.parse::<u32>() {
        return Some(v as i32);
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))
        && let Ok(v) = u32::from_str_radix(hex, 16)
    {
        return Some(v as i32);
    }
    None
}

/// Pure classification of whether a video-segment node, its operator owner, or the owner's input 0
/// carries SDK adjustment-layer markers (logged as diagnostic metadata alongside the PF map gate).
pub fn classify_adjustment_node(
    node_type: &str,
    node_flags: u32,
    owner_type: Option<&str>,
    owner_flags: Option<u32>,
    owner_input0_type: Option<&str>,
    owner_input0_flags: Option<u32>,
) -> (bool, &'static str) {
    let is_adj_flag = sys::PrNodeInfoFlag_kPrNodeInfoFlag_IsAdjustment;
    if node_type == NODE_TYPE_ADJUSTMENT_EFFECT {
        (true, "node_type:AdjustmentEffect")
    } else if node_type == NODE_TYPE_ADJUSTMENT {
        (true, "node_type:Adjustment")
    } else if (node_flags & is_adj_flag) != 0 {
        (true, "node_flag:IsAdjustment")
    } else if owner_type == Some(NODE_TYPE_ADJUSTMENT) {
        (true, "owner_type:Adjustment")
    } else if owner_type == Some(NODE_TYPE_ADJUSTMENT_EFFECT) {
        (true, "owner_type:AdjustmentEffect")
    } else if owner_flags.is_some_and(|f| (f & is_adj_flag) != 0) {
        (true, "owner_flag:IsAdjustment")
    } else if owner_input0_type == Some(NODE_TYPE_ADJUSTMENT) {
        (true, "owner_input0_type:Adjustment")
    } else if owner_input0_type == Some(NODE_TYPE_ADJUSTMENT_EFFECT) {
        (true, "owner_input0_type:AdjustmentEffect")
    } else if owner_input0_flags.is_some_and(|f| (f & is_adj_flag) != 0) {
        (true, "owner_input0_flag:IsAdjustment")
    } else {
        (false, "none")
    }
}

fn c_buf_to_string(buf: &[c_char]) -> String {
    let nul_pos = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let bytes: Vec<u8> = buf[..nul_pos].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

fn hash_bytes(id: &sys::prPluginID) -> [u8; 37] {
    let mut out = [0u8; 37];
    for (dst, &src) in out.iter_mut().zip(id.mGUID.iter()) {
        *dst = src as u8;
    }
    out
}

/// Maximum bytes scanned from a `GetNodeProperty` allocation even if the host returns a larger block.
const MAX_NODE_PROPERTY_BYTES: usize = 64 * 1024;

/// Reads one UTF-8 node property via `PrSDKVideoSegmentSuite::GetNodeProperty`, bounded strictly
/// by the memory manager's `GetPtrSize` allocation size (never scanning past the buffer on an
/// unterminated string) and always disposing the `PrMemoryPtr` via `PrMemPtrGuard`.
/// Declines (`Err`) if `GetPtrSize` is unavailable or returns 0.
unsafe fn read_node_property(
    vs_ptr: *const sys::PrSDKVideoSegmentSuite,
    sp: *const sys::SPBasicSuite,
    pi_suites: sys::piSuitesPtr,
    node_id: i32,
    key: &'static [u8],
) -> Result<String, i32> {
    let Some(get_prop) =
        (unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).GetNodeProperty)) })
    else {
        return Err(sys::suiteError_NotImplemented);
    };
    let mut raw: sys::PrMemoryPtr = std::ptr::null_mut();
    // SAFETY: `key` is a static NUL-terminated SDK constant and `raw` is a stack out-pointer.
    let err = unsafe { get_prop(node_id, key.as_ptr() as *const c_char, &mut raw) };
    if err != sys::suiteError_NoError {
        return Err(err);
    }
    if raw.is_null() {
        return Err(sys::suiteError_Fail);
    }
    // SAFETY: Ownership of `raw` was transferred to us by `GetNodeProperty`; dispose on drop.
    let guard = unsafe { PrMemPtrGuard::new(sp, pi_suites, raw) };
    let Some(alloc_size) = guard.alloc_size() else {
        return Err(sys::suiteError_Fail);
    };
    let scan_len = alloc_size.min(MAX_NODE_PROPERTY_BYTES);
    // SAFETY: `GetPtrSize` confirmed `raw` points to at least `alloc_size >= scan_len` bytes.
    let bytes = unsafe { std::slice::from_raw_parts(raw as *const u8, scan_len) };
    let nul_pos = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    Ok(String::from_utf8_lossy(&bytes[..nul_pos]).into_owned())
}

/// Queries `PrSDKVideoSegmentSuite` for `node_id` (including `EffectNode::RuntimeInstanceID`, its
/// owner via `AcquireOperatorOwnerNodeID`, and the owner's input 0 via `AcquireInputNodeID`).
unsafe fn inspect_node_is_adjustment(
    sp: *const sys::SPBasicSuite,
    pi_suites: sys::piSuitesPtr,
    node_id: i32,
) -> Result<NodeClassification, &'static str> {
    // `AcquireOperatorOwnerNodeID` was introduced in suite version 6 (CC 2014).
    let Some(vs) = (unsafe {
        RawSuite::<sys::PrSDKVideoSegmentSuite>::acquire_any(
            sp,
            sys::kPrSDKVideoSegmentSuite,
            sys::kPrSDKVideoSegmentSuiteVersion6,
            sys::kPrSDKVideoSegmentSuiteVersion,
        )
    }) else {
        return Err("no_video_segment_suite");
    };

    // SAFETY: `vs.ptr` is a valid acquired `PrSDKVideoSegmentSuite` v6..=v9 pointer.
    let (get_node_info, acq_owner, rel_node, get_input_count, acq_input) = unsafe {
        (
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).GetNodeInfo)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).AcquireOperatorOwnerNodeID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).ReleaseVideoNodeID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).GetNodeInputCount)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).AcquireInputNodeID)),
        )
    };
    let Some(get_node_info) = get_node_info else {
        return Err("no_get_node_info");
    };

    let mut type_buf = [0 as c_char; sys::kMaxNodeTypeStringSize as usize];
    let mut hash: sys::prPluginID = unsafe { std::mem::zeroed() };
    let mut flags: sys::csSDK_int32 = 0;
    // SAFETY: `type_buf` has `kMaxNodeTypeStringSize` (256) bytes as required by `GetNodeInfo`.
    let err = unsafe { get_node_info(node_id, type_buf.as_mut_ptr(), &mut hash, &mut flags) };
    if err != sys::suiteError_NoError {
        return Err("get_node_info_failed");
    }
    let node_type = c_buf_to_string(&type_buf);
    let node_flags = flags as u32;
    let node_hash = hash_bytes(&hash);

    let (runtime_instance_id_raw, runtime_instance_id, runtime_id_err) = match unsafe {
        read_node_property(
            vs.ptr,
            sp,
            pi_suites,
            node_id,
            sys::kVideoSegmentProperty_Effect_RuntimeInstanceID,
        )
    } {
        Ok(raw) => {
            let parsed = parse_runtime_instance_id(&raw);
            (Some(raw), parsed, None)
        }
        Err(e) => (None, None, Some(e)),
    };

    let mut owner_id: Option<i32> = None;
    let mut owner_type: Option<String> = None;
    let mut owner_flags: Option<u32> = None;
    let mut owner_err: Option<i32> = None;
    let mut owner_input0_type: Option<String> = None;
    let mut owner_input0_flags: Option<u32> = None;

    if let (Some(acq_owner), Some(rel_node)) = (acq_owner, rel_node) {
        let mut raw_owner: sys::csSDK_int32 = 0;
        // SAFETY: Valid `node_id` and stack out-pointer.
        let oerr = unsafe { acq_owner(node_id, &mut raw_owner) };
        if oerr == sys::suiteError_NoError {
            // Release owner node automatically when exiting this block (including on unwind).
            let _owner_guard = VideoNodeGuard {
                release_fn: rel_node,
                node_id: raw_owner,
            };
            owner_id = Some(raw_owner);
            let mut otype_buf = [0 as c_char; sys::kMaxNodeTypeStringSize as usize];
            let mut ohash: sys::prPluginID = unsafe { std::mem::zeroed() };
            let mut oflags: sys::csSDK_int32 = 0;
            // SAFETY: `raw_owner` is a live acquired node ID; `otype_buf` is 256 bytes.
            let info_err = unsafe {
                get_node_info(raw_owner, otype_buf.as_mut_ptr(), &mut ohash, &mut oflags)
            };
            if info_err == sys::suiteError_NoError {
                owner_type = Some(c_buf_to_string(&otype_buf));
                owner_flags = Some(oflags as u32);
            } else {
                owner_err = Some(info_err);
            }

            if let (Some(get_in_cnt), Some(acq_in)) = (get_input_count, acq_input) {
                let mut in_cnt: sys::csSDK_int32 = 0;
                // SAFETY: `raw_owner` is a live acquired node ID; stack out-pointer.
                if unsafe { get_in_cnt(raw_owner, &mut in_cnt) } == sys::suiteError_NoError
                    && in_cnt > 0
                {
                    let mut in_off: sys::PrTime = 0;
                    let mut raw_in0: sys::csSDK_int32 = 0;
                    // SAFETY: `raw_owner` is live and index `0 < in_cnt`.
                    if unsafe { acq_in(raw_owner, 0, &mut in_off, &mut raw_in0) }
                        == sys::suiteError_NoError
                    {
                        let _in0_guard = VideoNodeGuard {
                            release_fn: rel_node,
                            node_id: raw_in0,
                        };
                        let mut itype_buf = [0 as c_char; sys::kMaxNodeTypeStringSize as usize];
                        let mut ihash: sys::prPluginID = unsafe { std::mem::zeroed() };
                        let mut iflags: sys::csSDK_int32 = 0;
                        // SAFETY: `raw_in0` is live; `itype_buf` is 256 bytes.
                        if unsafe {
                            get_node_info(raw_in0, itype_buf.as_mut_ptr(), &mut ihash, &mut iflags)
                        } == sys::suiteError_NoError
                        {
                            owner_input0_type = Some(c_buf_to_string(&itype_buf));
                            owner_input0_flags = Some(iflags as u32);
                        }
                    }
                }
            }
        } else {
            owner_err = Some(oerr);
        }
    }

    let (is_adjustment, matched_by) = classify_adjustment_node(
        &node_type,
        node_flags,
        owner_type.as_deref(),
        owner_flags,
        owner_input0_type.as_deref(),
        owner_input0_flags,
    );

    Ok(NodeClassification {
        is_adjustment,
        matched_by,
        node_type,
        node_flags,
        node_hash,
        runtime_instance_id_raw,
        runtime_instance_id,
        runtime_id_err,
        owner_id,
        owner_type,
        owner_flags,
        owner_err,
        owner_input0_type,
        owner_input0_flags,
    })
}

/// Bounded set of runtime instance IDs (or fallback timeline/node keys) already dumped in this
/// process, so `CreateInstance` dumps the video segment graph at most once per instance.
static GRAPH_DUMP_SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
const MAX_GRAPH_DUMP_KEYS: usize = 512;
const GRAPH_MAX_SEGMENTS_SCAN: i32 = 24;
const GRAPH_MAX_NODES: usize = 24;
const GRAPH_MAX_DEPTH: usize = 8;
const GRAPH_MAX_CHILDREN_PER_NODE: i32 = 8;

/// Walks one video segment root node (its inputs and each node's operators), bounded to
/// `GRAPH_MAX_NODES` nodes and `GRAPH_MAX_DEPTH` depth, releasing every acquired node ID via
/// RAII `VideoNodeGuard`. Returns `(formatted_nodes, found_this_effect)`.
#[allow(clippy::too_many_arguments)]
unsafe fn walk_segment_tree(
    vs_ptr: *const sys::PrSDKVideoSegmentSuite,
    sp: *const sys::SPBasicSuite,
    pi_suites: sys::piSuitesPtr,
    root_node_id: i32,
    target_node_id: i32,
    target_hash: &[u8; 37],
    target_rtid: Option<&str>,
) -> (Vec<String>, bool) {
    let (get_info, get_in_cnt, acq_in, get_op_cnt, acq_op, rel_node) = unsafe {
        (
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).GetNodeInfo)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).GetNodeInputCount)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).AcquireInputNodeID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).GetNodeOperatorCount)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).AcquireOperatorNodeID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs_ptr).ReleaseVideoNodeID)),
        )
    };
    let Some(get_info) = get_info else {
        return (vec!["no_get_node_info".to_string()], false);
    };

    let target_hash_nonzero = target_hash.iter().any(|&b| b != 0);
    let mut entries: Vec<String> = Vec::new();
    let mut found_this = false;

    #[allow(clippy::too_many_arguments)]
    unsafe fn visit(
        vs_ptr: *const sys::PrSDKVideoSegmentSuite,
        sp: *const sys::SPBasicSuite,
        pi_suites: sys::piSuitesPtr,
        get_info: unsafe extern "C" fn(i32, *mut c_char, *mut sys::prPluginID, *mut i32) -> i32,
        get_in_cnt: Option<unsafe extern "C" fn(i32, *mut i32) -> i32>,
        acq_in: Option<unsafe extern "C" fn(i32, i32, *mut sys::PrTime, *mut i32) -> i32>,
        get_op_cnt: Option<unsafe extern "C" fn(i32, *mut i32) -> i32>,
        acq_op: Option<unsafe extern "C" fn(i32, i32, *mut i32) -> i32>,
        rel_node: Option<unsafe extern "C" fn(i32) -> i32>,
        cur_node_id: i32,
        edge_label: &str,
        depth: usize,
        target_node_id: i32,
        target_hash: &[u8; 37],
        target_hash_nonzero: bool,
        target_rtid: Option<&str>,
        entries: &mut Vec<String>,
        found_this: &mut bool,
    ) {
        if entries.len() >= GRAPH_MAX_NODES || depth > GRAPH_MAX_DEPTH {
            return;
        }
        let mut type_buf = [0 as c_char; sys::kMaxNodeTypeStringSize as usize];
        let mut hash: sys::prPluginID = unsafe { std::mem::zeroed() };
        let mut flags: sys::csSDK_int32 = 0;
        let info_err = unsafe { get_info(cur_node_id, type_buf.as_mut_ptr(), &mut hash, &mut flags) };
        let (ntype, nflags, nhash) = if info_err == sys::suiteError_NoError {
            (c_buf_to_string(&type_buf), flags as u32, hash_bytes(&hash))
        } else {
            (format!("err({info_err})"), 0u32, [0u8; 37])
        };

        let rtid_raw = unsafe {
            read_node_property(
                vs_ptr,
                sp,
                pi_suites,
                cur_node_id,
                sys::kVideoSegmentProperty_Effect_RuntimeInstanceID,
            )
            .ok()
        };
        let is_this = cur_node_id == target_node_id
            || (target_hash_nonzero && nhash == *target_hash)
            || (target_rtid.is_some()
                && rtid_raw.as_deref().map(str::trim) == target_rtid.map(str::trim));
        if is_this {
            *found_this = true;
        }

        let mut in_cnt = 0i32;
        if let Some(f) = get_in_cnt
            && unsafe { f(cur_node_id, &mut in_cnt) } != sys::suiteError_NoError
        {
            in_cnt = -1;
        }
        let mut op_cnt = 0i32;
        if let Some(f) = get_op_cnt
            && unsafe { f(cur_node_id, &mut op_cnt) } != sys::suiteError_NoError
        {
            op_cnt = -1;
        }

        let rtid_tok = rtid_raw
            .as_deref()
            .map(diag::sanitize_token)
            .unwrap_or_else(|| "n/a".to_string());
        let marker = if is_this { "*THIS*" } else { "" };
        entries.push(format!(
            "{marker}{edge_label}@d{depth}:{}(f=0x{nflags:x},rtid={rtid_tok},in={in_cnt},op={op_cnt})",
            diag::sanitize_token(&ntype),
        ));

        if depth >= GRAPH_MAX_DEPTH || entries.len() >= GRAPH_MAX_NODES {
            return;
        }
        let Some(rel_fn) = rel_node else {
            return;
        };

        // Operators on this node (effects live here on clip/composite/adjustment nodes).
        if op_cnt > 0
            && let Some(acq_op_fn) = acq_op
        {
            for idx in 0..op_cnt.min(GRAPH_MAX_CHILDREN_PER_NODE) {
                if entries.len() >= GRAPH_MAX_NODES {
                    break;
                }
                let mut child_id = 0i32;
                if unsafe { acq_op_fn(cur_node_id, idx, &mut child_id) } == sys::suiteError_NoError
                {
                    let _g = VideoNodeGuard {
                        release_fn: rel_fn,
                        node_id: child_id,
                    };
                    let lbl = format!("op{idx}");
                    unsafe {
                        visit(
                            vs_ptr,
                            sp,
                            pi_suites,
                            get_info,
                            get_in_cnt,
                            acq_in,
                            get_op_cnt,
                            acq_op,
                            rel_node,
                            child_id,
                            &lbl,
                            depth + 1,
                            target_node_id,
                            target_hash,
                            target_hash_nonzero,
                            target_rtid,
                            entries,
                            found_this,
                        );
                    }
                }
            }
        }

        // Inputs to this node (tracks / clips / media / nested compositors).
        if in_cnt > 0
            && let Some(acq_in_fn) = acq_in
        {
            for idx in 0..in_cnt.min(GRAPH_MAX_CHILDREN_PER_NODE) {
                if entries.len() >= GRAPH_MAX_NODES {
                    break;
                }
                let mut off: sys::PrTime = 0;
                let mut child_id = 0i32;
                if unsafe { acq_in_fn(cur_node_id, idx, &mut off, &mut child_id) }
                    == sys::suiteError_NoError
                {
                    let _g = VideoNodeGuard {
                        release_fn: rel_fn,
                        node_id: child_id,
                    };
                    let lbl = format!("in{idx}");
                    unsafe {
                        visit(
                            vs_ptr,
                            sp,
                            pi_suites,
                            get_info,
                            get_in_cnt,
                            acq_in,
                            get_op_cnt,
                            acq_op,
                            rel_node,
                            child_id,
                            &lbl,
                            depth + 1,
                            target_node_id,
                            target_hash,
                            target_hash_nonzero,
                            target_rtid,
                            entries,
                            found_this,
                        );
                    }
                }
            }
        }
    }

    unsafe {
        visit(
            vs_ptr,
            sp,
            pi_suites,
            get_info,
            get_in_cnt,
            acq_in,
            get_op_cnt,
            acq_op,
            rel_node,
            root_node_id,
            "root",
            0,
            target_node_id,
            target_hash,
            target_hash_nonzero,
            target_rtid,
            &mut entries,
            &mut found_this,
        );
    }
    (entries, found_this)
}

/// Builds a compact, bounded diagnostic dump of the video segment graph around `target_node_id`.
/// Acquires and releases every `VideoSegmentsID`, `VideoNodeID`, and `PrMemoryPtr` via RAII guards.
unsafe fn collect_segment_graph_dump(
    sp: *const sys::SPBasicSuite,
    pi_suites: sys::piSuitesPtr,
    timeline_id: sys::PrTimelineID,
    target_node_id: i32,
    node_cls: &NodeClassification,
) -> String {
    let rtid_tok = node_cls
        .runtime_instance_id_raw
        .as_deref()
        .map(diag::sanitize_token)
        .unwrap_or_else(|| "n/a".to_string());
    let Some(vs) = (unsafe {
        RawSuite::<sys::PrSDKVideoSegmentSuite>::acquire_any(
            sp,
            sys::kPrSDKVideoSegmentSuite,
            sys::kPrSDKVideoSegmentSuiteVersion6,
            sys::kPrSDKVideoSegmentSuiteVersion,
        )
    }) else {
        return format!(
            "event=gpu_graph_dump path=gpu timeline_id={timeline_id} node_id={target_node_id} \
             runtime_instance_id={rtid_tok} err=no_video_segment_suite"
        );
    };

    let (acq_segs, rel_segs, get_seg_cnt, get_seg_info, acq_node_id, acq_node_for_time, rel_node) = unsafe {
        (
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).AcquireVideoSegmentsID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).ReleaseVideoSegmentsID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).GetSegmentCount)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).GetSegmentInfo)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).AcquireNodeID)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).AcquireNodeForTime)),
            std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).ReleaseVideoNodeID)),
        )
    };
    let (Some(acq_segs), Some(rel_segs), Some(rel_node)) = (acq_segs, rel_segs, rel_node) else {
        return format!(
            "event=gpu_graph_dump path=gpu timeline_id={timeline_id} node_id={target_node_id} \
             runtime_instance_id={rtid_tok} err=missing_suite_fns"
        );
    };

    let mut segs_id = 0i32;
    let err = unsafe { acq_segs(timeline_id, &mut segs_id) };
    if err != sys::suiteError_NoError {
        return format!(
            "event=gpu_graph_dump path=gpu timeline_id={timeline_id} node_id={target_node_id} \
             runtime_instance_id={rtid_tok} err=acq_segs({err})"
        );
    }
    let _segs_guard = VideoSegmentsGuard {
        release_fn: rel_segs,
        segs_id,
    };

    let mut seg_count = 0i32;
    if let Some(f) = get_seg_cnt
        && unsafe { f(segs_id, &mut seg_count) } != sys::suiteError_NoError
    {
        seg_count = 0;
    }

    let mut best_walk: Option<(i32, i64, i64, Vec<String>, bool)> = None;

    if seg_count > 0
        && let (Some(seg_info_fn), Some(acq_node_fn)) = (get_seg_info, acq_node_id)
    {
        for s_idx in 0..seg_count.min(GRAPH_MAX_SEGMENTS_SCAN) {
            let mut start_t: sys::PrTime = 0;
            let mut end_t: sys::PrTime = 0;
            let mut seg_off: sys::PrTime = 0;
            let mut seg_hash: sys::prPluginID = unsafe { std::mem::zeroed() };
            if unsafe {
                seg_info_fn(segs_id, s_idx, &mut start_t, &mut end_t, &mut seg_off, &mut seg_hash)
            } != sys::suiteError_NoError
            {
                continue;
            }
            let mut root_id = 0i32;
            if unsafe { acq_node_fn(segs_id, &mut seg_hash, &mut root_id) }
                != sys::suiteError_NoError
            {
                continue;
            }
            let _root_guard = VideoNodeGuard {
                release_fn: rel_node,
                node_id: root_id,
            };
            let (nodes, found) = unsafe {
                walk_segment_tree(
                    vs.ptr,
                    sp,
                    pi_suites,
                    root_id,
                    target_node_id,
                    &node_cls.node_hash,
                    node_cls.runtime_instance_id_raw.as_deref(),
                )
            };
            if found {
                best_walk = Some((s_idx, start_t, end_t, nodes, true));
                break;
            }
            if best_walk.is_none() {
                best_walk = Some((s_idx, start_t, end_t, nodes, false));
            }
        }
    } else if let Some(acq_time_fn) = acq_node_for_time {
        let mut root_id = 0i32;
        let mut seg_off: sys::PrTime = 0;
        if unsafe { acq_time_fn(segs_id, 0, &mut root_id, &mut seg_off) }
            == sys::suiteError_NoError
        {
            let _root_guard = VideoNodeGuard {
                release_fn: rel_node,
                node_id: root_id,
            };
            let (nodes, found) = unsafe {
                walk_segment_tree(
                    vs.ptr,
                    sp,
                    pi_suites,
                    root_id,
                    target_node_id,
                    &node_cls.node_hash,
                    node_cls.runtime_instance_id_raw.as_deref(),
                )
            };
            best_walk = Some((0, 0, 0, nodes, found));
        }
    }

    match best_walk {
        Some((s_idx, start_t, end_t, nodes, found)) => format!(
            "event=gpu_graph_dump path=gpu timeline_id={timeline_id} node_id={target_node_id} \
             runtime_instance_id={rtid_tok} segs_total={seg_count} seg_idx={s_idx} \
             seg_range={start_t}..{end_t} found_this={found} nodes=[{}]",
            nodes.join(" ")
        ),
        None => format!(
            "event=gpu_graph_dump path=gpu timeline_id={timeline_id} node_id={target_node_id} \
             runtime_instance_id={rtid_tok} segs_total={seg_count} found_this=false nodes=[]"
        ),
    }
}

/// Dumps the video segment graph once per unique runtime instance ID when `/tmp/tblur-debug`
/// is active. Bounded, panic-safe, and non-blocking.
unsafe fn maybe_dump_segment_graph(
    sp: *const sys::SPBasicSuite,
    pi_suites: sys::piSuitesPtr,
    timeline_id: sys::PrTimelineID,
    target_node_id: i32,
    node_cls: &NodeClassification,
) {
    let key = match &node_cls.runtime_instance_id_raw {
        Some(raw) if !raw.trim().is_empty() => format!("rtid:{}", diag::sanitize_token(raw.trim())),
        _ => format!("tl:{timeline_id}:node:{target_node_id}"),
    };
    let should_dump = {
        let set = GRAPH_DUMP_SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let mut g = set.lock().unwrap_or_else(|e| e.into_inner());
        if g.contains(&key) || g.len() >= MAX_GRAPH_DUMP_KEYS {
            false
        } else {
            g.insert(key);
            true
        }
    };
    if !should_dump {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let line =
            unsafe { collect_segment_graph_dump(sp, pi_suites, timeline_id, target_node_id, node_cls) };
        diag::append(&line);
    }));
}

// ───────────────────────────── parameter reading & mapping ─────────────────────────────

/// GPU `GetParam` indices (0-based, excluding PF input layer at index 0).
/// Must stay in exact 1-to-1 correspondence with `FbParam` in `effect.rs`:
/// * PF 1 -> GPU 0: BlurAmount (FloatSlider, 0.0..=4.0, default 0.5)
/// * PF 2 -> GPU 1: MotionSensitivity (FloatSlider, 0.0..=100.0, default 70.0)
/// * PF 3 -> GPU 2: MaxBlur (FloatSlider, 0.0..=50.0, default 15.0)
/// * PF 4 -> GPU 3: Quality (Popup 0-based: 0=Draft, 1=Normal, 2=High, default 1)
/// * PF 5 -> GPU 4: LinearLight (CheckBox, default true)
/// * PF 6 -> GPU 5: View (Popup 0-based: 0=Output, 1=MotionVectors, default 0)
/// * PF 7 -> GPU 6: UseSourceRate (CheckBox, default true; unused on adjustment layers)
/// * PF 8 -> GPU 7: UpdateNotice (Button, no value; never read here). It is the last
///   parameter, so it cannot shift the indices above (pinned by a test in `effect.rs`).
pub const GPU_PARAM_BLUR_AMOUNT: i32 = 0;
pub const GPU_PARAM_MOTION_SENSITIVITY: i32 = 1;
pub const GPU_PARAM_MAX_BLUR: i32 = 2;
pub const GPU_PARAM_QUALITY: i32 = 3;
pub const GPU_PARAM_LINEAR_LIGHT: i32 = 4;
pub const GPU_PARAM_VIEW: i32 = 5;
pub const GPU_PARAM_USE_SOURCE_RATE: i32 = 6;

fn pr_param_as_f64(p: &sys::PrParam) -> Option<f64> {
    // SAFETY: Each match arm reads only the union member matching `p.mType`. Copying into a local
    // avoids creating an unaligned reference into the `#[repr(C, packed)]` union.
    unsafe {
        match p.mType {
            sys::PrParamType_kPrParamType_Float64 => Some(p.__bindgen_anon_1.mFloat64),
            sys::PrParamType_kPrParamType_Float32 => Some(p.__bindgen_anon_1.mFloat32 as f64),
            sys::PrParamType_kPrParamType_Int32 => Some(p.__bindgen_anon_1.mInt32 as f64),
            sys::PrParamType_kPrParamType_Int64 => Some(p.__bindgen_anon_1.mInt64 as f64),
            _ => None,
        }
    }
}

fn pr_param_as_i32(p: &sys::PrParam) -> Option<i32> {
    // SAFETY: Each match arm reads only the union member matching `p.mType`.
    unsafe {
        match p.mType {
            sys::PrParamType_kPrParamType_Int8 => Some(p.__bindgen_anon_1.mInt8 as i32),
            sys::PrParamType_kPrParamType_Int16 => Some(p.__bindgen_anon_1.mInt16 as i32),
            sys::PrParamType_kPrParamType_Int32 => Some(p.__bindgen_anon_1.mInt32),
            sys::PrParamType_kPrParamType_Int64 => i32::try_from(p.__bindgen_anon_1.mInt64).ok(),
            sys::PrParamType_kPrParamType_Float32 => Some(p.__bindgen_anon_1.mFloat32.round() as i32),
            sys::PrParamType_kPrParamType_Float64 => Some(p.__bindgen_anon_1.mFloat64.round() as i32),
            sys::PrParamType_kPrParamType_Bool => Some(p.__bindgen_anon_1.mBool as i32),
            _ => None,
        }
    }
}

fn pr_param_as_bool(p: &sys::PrParam) -> Option<bool> {
    // SAFETY: Each match arm reads only the union member matching `p.mType`.
    unsafe {
        match p.mType {
            sys::PrParamType_kPrParamType_Bool => Some(p.__bindgen_anon_1.mBool != 0),
            sys::PrParamType_kPrParamType_Int8 => Some(p.__bindgen_anon_1.mInt8 != 0),
            sys::PrParamType_kPrParamType_Int16 => Some(p.__bindgen_anon_1.mInt16 != 0),
            sys::PrParamType_kPrParamType_Int32 => Some(p.__bindgen_anon_1.mInt32 != 0),
            sys::PrParamType_kPrParamType_Int64 => Some(p.__bindgen_anon_1.mInt64 != 0),
            sys::PrParamType_kPrParamType_Float32 => Some(p.__bindgen_anon_1.mFloat32 != 0.0),
            sys::PrParamType_kPrParamType_Float64 => Some(p.__bindgen_anon_1.mFloat64 != 0.0),
            _ => None,
        }
    }
}

/// Pure mapping from raw GPU `PrParam` values (indices 0..=5) to `core::Params`.
/// Matches `effect::read_core_params` identically (accounting for 0-based GPU popups).
pub fn map_gpu_params(raw: &[sys::PrParam; 6]) -> Result<core::Params, &'static str> {
    let blur = pr_param_as_f64(&raw[GPU_PARAM_BLUR_AMOUNT as usize])
        .ok_or("param_blur_amount_type")? as f32;
    let sens = pr_param_as_f64(&raw[GPU_PARAM_MOTION_SENSITIVITY as usize])
        .ok_or("param_motion_sensitivity_type")? as f32;
    let max_blur =
        pr_param_as_f64(&raw[GPU_PARAM_MAX_BLUR as usize]).ok_or("param_max_blur_type")? as f32;
    let qual = pr_param_as_i32(&raw[GPU_PARAM_QUALITY as usize]).ok_or("param_quality_type")?;
    let linear =
        pr_param_as_bool(&raw[GPU_PARAM_LINEAR_LIGHT as usize]).ok_or("param_linear_light_type")?;
    let view = pr_param_as_i32(&raw[GPU_PARAM_VIEW as usize]).ok_or("param_view_type")?;

    Ok(core::Params {
        blur_amount: blur.clamp(0.0, 4.0),
        motion_sensitivity: sens.clamp(0.0, 100.0),
        max_blur_percent: max_blur.clamp(0.0, 50.0),
        quality: match qual {
            0 => core::Quality::Draft,
            2 => core::Quality::High,
            _ => core::Quality::Normal,
        },
        linear_light: linear,
        view: match view {
            1 => core::View::MotionVectors,
            _ => core::View::Output,
        },
    })
}

/// Reads T Blur's core parameters from Premiere via `PrSDKVideoSegmentSuite::GetParam`.
unsafe fn read_gpu_core_params(
    sp: *const sys::SPBasicSuite,
    node_id: i32,
    clip_time: sys::PrTime,
) -> Result<(core::Params, bool), String> {
    let Some(vs) = (unsafe {
        RawSuite::<sys::PrSDKVideoSegmentSuite>::acquire_any(
            sp,
            sys::kPrSDKVideoSegmentSuite,
            sys::kPrSDKVideoSegmentSuiteVersion6,
            sys::kPrSDKVideoSegmentSuiteVersion,
        )
    }) else {
        return Err("no_video_segment_suite".to_string());
    };
    // SAFETY: `vs.ptr` is a valid acquired `PrSDKVideoSegmentSuite*`.
    let Some(get_param) =
        (unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*vs.ptr).GetParam)) })
    else {
        return Err("no_get_param".to_string());
    };

    let fetch = |idx: i32| -> Result<sys::PrParam, i32> {
        // SAFETY: `PrParam` is a C POD struct zeroed before passing to `GetParam`.
        let mut p: sys::PrParam = unsafe { std::mem::zeroed() };
        // SAFETY: Valid `node_id`, parameter index, `clip_time`, and stack out-pointer.
        let err = unsafe { get_param(node_id, idx, clip_time, &mut p) };
        if err == sys::suiteError_NoError {
            Ok(p)
        } else {
            Err(err)
        }
    };

    let raw = [
        fetch(GPU_PARAM_BLUR_AMOUNT).map_err(|e| format!("get_param_0_err:{e}"))?,
        fetch(GPU_PARAM_MOTION_SENSITIVITY).map_err(|e| format!("get_param_1_err:{e}"))?,
        fetch(GPU_PARAM_MAX_BLUR).map_err(|e| format!("get_param_2_err:{e}"))?,
        fetch(GPU_PARAM_QUALITY).map_err(|e| format!("get_param_3_err:{e}"))?,
        fetch(GPU_PARAM_LINEAR_LIGHT).map_err(|e| format!("get_param_4_err:{e}"))?,
        fetch(GPU_PARAM_VIEW).map_err(|e| format!("get_param_5_err:{e}"))?,
    ];
    let core_params = map_gpu_params(&raw).map_err(|e| e.to_string())?;
    let use_source_rate = fetch(GPU_PARAM_USE_SOURCE_RATE)
        .ok()
        .and_then(|p| pr_param_as_bool(&p))
        .unwrap_or(false);
    Ok((core_params, use_source_rate))
}

// ───────────────────────────── dependency protocol ─────────────────────────────

/// One declared input-frame dependency in the exact order reported by `GetFrameDependencies`
/// and delivered in `inFrames[1..]` to `Render`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeclaredDep {
    /// Direction: `-1` for previous, `+1` for next.
    pub dir: i32,
    /// Step distance `1..=steps`.
    pub k: u32,
    /// Non-negative sequence time (`seq_time + dir * k * ticks_per_frame >= 0`).
    pub sequence_time: i64,
    /// Track ID (`0` = same track).
    pub track_id: i32,
}

/// One step of Premiere's `GetFrameDependencies` query protocol.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DependencyStep {
    pub sequence_time: i64,
    pub track_id: i32,
    pub next_query_index: i32,
}

/// Single source of truth for declared GPU dependencies, shared by `GetFrameDependencies` and
/// `Render` so the two can never disagree.
///
/// * Returns empty when blur is disabled (`blur_amount <= 0` or `motion_sensitivity <= 0`),
///   `ticks_per_frame <= 0`, or `steps == 0`.
/// * Iterates `k` in `1..=steps` in interleaved order (`-k`, then `+k`), omitting any candidate
///   whose sequence time would overflow or fall before sequence start (`< 0`). Omitted neighbours
///   are treated as `DepFrame::Missing` during selection, matching the CPU checkout path.
pub fn gpu_declared_dependencies(
    params: &core::Params,
    seq_time: i64,
    ticks_per_frame: i64,
    steps: u32,
) -> Vec<DeclaredDep> {
    if ticks_per_frame <= 0
        || steps == 0
        || !params.blur_amount.is_finite()
        || params.blur_amount <= 0.0
        || !params.motion_sensitivity.is_finite()
        || params.motion_sensitivity <= 0.0
    {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((2 * steps) as usize);
    for k in 1..=steps {
        let Some(delta) = (k as i64).checked_mul(ticks_per_frame) else {
            break;
        };
        if let Some(prev_t) = seq_time.checked_sub(delta)
            && prev_t >= 0
        {
            out.push(DeclaredDep {
                dir: -1,
                k,
                sequence_time: prev_t,
                track_id: 0,
            });
        }
        if let Some(next_t) = seq_time.checked_add(delta)
            && next_t >= 0
        {
            out.push(DeclaredDep {
                dir: 1,
                k,
                sequence_time: next_t,
                track_id: 0,
            });
        }
    }
    out
}

/// Number of input-frame dependencies declared for `(params, seq_time, ticks_per_frame, steps)`.
pub fn gpu_dependency_count_steps(
    params: &core::Params,
    seq_time: i64,
    ticks_per_frame: i64,
    steps: u32,
) -> u32 {
    gpu_declared_dependencies(params, seq_time, ticks_per_frame, steps).len() as u32
}

/// Default-step variant of `gpu_dependency_count_steps`.
pub fn gpu_dependency_count(params: &core::Params, seq_time: i64, ticks_per_frame: i64) -> u32 {
    gpu_dependency_count_steps(params, seq_time, ticks_per_frame, GPU_DEPENDENCY_STEPS)
}

/// Pure state machine for `GetFrameDependencies` backed directly by `gpu_declared_dependencies`.
/// Leaves `next_query_index == query_index` on the final dependency, and returns
/// `query_index + 1` on all earlier ones.
pub fn gpu_dependency_at_steps(
    params: &core::Params,
    seq_time: i64,
    ticks_per_frame: i64,
    query_index: i32,
    steps: u32,
) -> Option<DependencyStep> {
    if query_index < 0 {
        return None;
    }
    let deps = gpu_declared_dependencies(params, seq_time, ticks_per_frame, steps);
    let idx = query_index as usize;
    let dep = *deps.get(idx)?;
    let next_query_index = if idx + 1 < deps.len() {
        query_index + 1
    } else {
        query_index
    };
    Some(DependencyStep {
        sequence_time: dep.sequence_time,
        track_id: dep.track_id,
        next_query_index,
    })
}

/// Default-step variant of `gpu_dependency_at_steps`.
pub fn gpu_dependency_at(
    params: &core::Params,
    seq_time: i64,
    ticks_per_frame: i64,
    query_index: i32,
) -> Option<DependencyStep> {
    gpu_dependency_at_steps(params, seq_time, ticks_per_frame, query_index, GPU_DEPENDENCY_STEPS)
}

/// Maps `(dir, k)` (`dir < 0` for previous, `dir > 0` for next, `k >= 1`) to the 0-based index
/// inside `inFrames[1..]` (i.e. `inFrames` index minus 1), matching `gpu_dependency_at_steps`.
pub fn dep_slice_index(dir: i32, k: u32) -> Option<usize> {
    if k == 0 || dir == 0 {
        return None;
    }
    let base = ((k - 1) as usize).checked_mul(2)?;
    if dir < 0 {
        Some(base)
    } else {
        base.checked_add(1)
    }
}

// ───────────────────────────── GPU frame inspection ─────────────────────────────

#[derive(Clone, Copy)]
struct FrameInspect {
    ppix: sys::PPixHand,
    data_ptr: *mut c_void,
    layout: Option<Layout>,
    raw_fmt: sys::PrPixelFormat,
    fmt_name: &'static str,
    width: i32,
    height: i32,
    row_bytes: i32,
}

impl FrameInspect {
    fn as_metal_desc(&self) -> Option<MetalFrameDesc> {
        let layout = self.layout?;
        if self.ppix.is_null()
            || self.data_ptr.is_null()
            || self.width <= 0
            || self.height <= 0
            || self.row_bytes <= 0
        {
            return None;
        }
        Some(MetalFrameDesc {
            ptr: self.data_ptr,
            width: self.width as usize,
            height: self.height as usize,
            row_bytes: self.row_bytes,
            layout,
        })
    }

    fn as_dep_frame(&self) -> DepFrame {
        if self.ppix.is_null() || self.data_ptr.is_null() {
            return DepFrame::Missing;
        }
        let Some(layout) = self.layout else {
            return DepFrame::BadFormat;
        };
        if self.width <= 0 || self.height <= 0 || self.row_bytes <= 0 {
            return DepFrame::BadGeometry;
        }
        DepFrame::Candidate(MetalFrameDesc {
            ptr: self.data_ptr,
            width: self.width as usize,
            height: self.height as usize,
            row_bytes: self.row_bytes,
            layout,
        })
    }
}

unsafe fn inspect_gpu_ppix(
    gpu_suite: *const sys::PrSDKGPUDeviceSuite,
    ppix_suite: *const sys::PrSDKPPixSuite,
    ppix: sys::PPixHand,
) -> FrameInspect {
    let mut info = FrameInspect {
        ppix,
        data_ptr: std::ptr::null_mut(),
        layout: None,
        raw_fmt: sys::PrPixelFormat_PrPixelFormat_Invalid,
        fmt_name: "null",
        width: 0,
        height: 0,
        row_bytes: 0,
    };
    if ppix.is_null() {
        return info;
    }
    // SAFETY: `ppix_suite` and `gpu_suite` are acquired host suites checked non-null by caller.
    unsafe {
        if let Some(f) = std::ptr::read_unaligned(std::ptr::addr_of!((*ppix_suite).GetPixelFormat))
        {
            let mut raw: sys::PrPixelFormat = sys::PrPixelFormat_PrPixelFormat_Invalid;
            if f(ppix, &mut raw) == sys::suiteError_NoError {
                info.raw_fmt = raw;
                let (name, layout) = pixel_format_info(raw);
                info.fmt_name = name;
                info.layout = layout;
            }
        }
        if let Some(f) = std::ptr::read_unaligned(std::ptr::addr_of!((*ppix_suite).GetBounds)) {
            let mut r: sys::prRect = std::mem::zeroed();
            if f(ppix, &mut r) == sys::suiteError_NoError {
                info.width = (r.right as i32).saturating_sub(r.left as i32);
                info.height = (r.bottom as i32).saturating_sub(r.top as i32);
            }
        }
        if let Some(f) = std::ptr::read_unaligned(std::ptr::addr_of!((*ppix_suite).GetRowBytes)) {
            let mut rb: i32 = 0;
            if f(ppix, &mut rb) == sys::suiteError_NoError {
                info.row_bytes = rb;
            }
        }
        if let Some(f) = std::ptr::read_unaligned(std::ptr::addr_of!((*gpu_suite).GetGPUPPixData))
        {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            if f(ppix, &mut ptr) == sys::suiteError_NoError {
                info.data_ptr = ptr;
            }
        }
    }
    info
}

// ───────────────────────────── core Metal-backed render pipeline ─────────────────────────────

/// Lightweight per-dependency metadata before lazy readback.
#[derive(Clone, Copy, Debug)]
pub enum DepFrame {
    Missing,
    BadFormat,
    BadGeometry,
    Candidate(MetalFrameDesc),
}

/// Per-reason counts of neighbours rejected during `select_legacy_neighbors` (logged on every GPU
/// render line for direct comparison against CPU logs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UnusableCounts {
    pub missing: u32,
    pub bad_format: u32,
    pub not_buffer: u32,
    pub size_mismatch: u32,
    pub bad_geometry: u32,
    pub transparent: u32,
    pub read_err: u32,
}

impl UnusableCounts {
    pub fn format_log(&self) -> String {
        format!(
            "missing:{},fmt:{},not_buf:{},size:{},geom:{},alpha0:{},read:{}",
            self.missing,
            self.bad_format,
            self.not_buffer,
            self.size_mismatch,
            self.bad_geometry,
            self.transparent,
            self.read_err,
        )
    }
}

/// Metrics and neighbour-selection trace produced by `render_metal_frames`.
#[derive(Debug)]
pub struct GpuRenderReport {
    pub stats: core::RenderStats,
    pub deps_read: u32,
    pub prev_k: Option<u32>,
    pub next_k: Option<u32>,
    pub fd: (Option<f32>, Option<f32>),
    pub probes: Vec<(&'static str, u32, &'static str)>,
    pub unusable: UnusableCounts,
    pub dup_ref: &'static str,
    pub readback_ms: f64,
    pub write_ms: f64,
    pub cur_storage: &'static str,
    pub out_storage: &'static str,
}

/// Runs the T Blur adjustment-layer render pipeline directly over Metal buffers:
/// 1. Reads `cur` into premultiplied f32 RGBA + luma + `max_alpha` (subject to `deadline`).
/// 2. Runs `effect::select_legacy_neighbors` (exact same function as the CPU legacy path),
///    reading back ONLY the dependency buffers actually inspected by the search, rejecting
///    empty/fully-transparent edge frames with the same `cur_has_alpha && max_a <= 0.0` rule, and
///    aborting immediately if any GPU wait times out (Item 5).
/// 3. Runs `effect::call_core_owned` (`tblur_core::render_with_owned_luma`).
/// 4. Writes the blurred output into `out` (subject to the same single `deadline`).
///
/// # Safety
/// Every non-null `MetalFrameDesc::ptr` in `cur`, `deps`, and `out` must be a live Objective-C
/// object (`id`); if it is an `MTLBuffer`, it must belong to `metal`'s device, and `out.ptr` must
/// not alias `cur.ptr` or any `deps` entry.
pub unsafe fn render_metal_frames(
    metal: &MetalContext,
    core_params: &core::Params,
    cur: &MetalFrameDesc,
    deps: &[DepFrame],
    out: &MetalFrameDesc,
    max_k: u32,
    wait_budget: &Cell<Duration>,
) -> Result<GpuRenderReport, MetalError> {
    let _render_guard = core::par::enter_render();
    let queue_synced = Cell::new(false);
    let mut readback_ms = 0.0f64;

    // 1. Read current frame (RGBA + luma + max alpha in one pass, matching `effect::decode_with_luma`).
    let mut cur_rgba = Vec::new();
    let t_cur = Instant::now();
    let (cur_res, cur_storage) = unsafe {
        metal.with_readable_bytes(cur, &queue_synced, wait_budget, |bytes, desc| {
            pixels::read_rgba_and_luma(bytes, desc, &mut cur_rgba)
        })
    }?;
    readback_ms += t_cur.elapsed().as_secs_f64() * 1000.0;
    let (cur_luma, cur_max_alpha) =
        cur_res.map_err(|e| MetalError::Other(format!("decode_cur_err:{e:?}")))?;

    let (w, h) = (cur.width, cur.height);
    if (out.width, out.height) != (w, h) {
        return Err(MetalError::BadGeometry(format!(
            "out_size_mismatch:{}x{}!={}x{}",
            out.width, out.height, w, h
        )));
    }
    let cur_has_alpha = cur_max_alpha > 0.0;

    // 2. Select previous and next neighbours using the EXACT shared legacy selection rule.
    //    Any GPU timeout immediately short-circuits the remaining search (Item 5).
    let mut deps_read = 0u32;
    let mut unusable = UnusableCounts::default();
    let mut dep_timeout: Option<MetalError> = None;

    let sel = {
        let cur_l = core::Luma {
            width: w,
            height: h,
            data: &cur_luma,
        };
        let is_dup = |luma: &Vec<f32>| {
            core::is_duplicate_frame(
                cur_l,
                core::Luma {
                    width: w,
                    height: h,
                    data: luma,
                },
            )
        };
        effect::select_legacy_neighbors(
            max_k,
            |dir, k| {
                if dep_timeout.is_some() {
                    return Err(Outcome::Unusable);
                }
                let Some(idx) = dep_slice_index(dir, k) else {
                    unusable.bad_geometry += 1;
                    return Err(Outcome::Unusable);
                };
                let Some(&dep) = deps.get(idx) else {
                    unusable.missing += 1;
                    return Err(Outcome::Missing);
                };
                let cand = match dep {
                    DepFrame::Missing => {
                        unusable.missing += 1;
                        return Err(Outcome::Missing);
                    }
                    DepFrame::BadFormat => {
                        unusable.bad_format += 1;
                        return Err(Outcome::Unusable);
                    }
                    DepFrame::BadGeometry => {
                        unusable.bad_geometry += 1;
                        return Err(Outcome::Unusable);
                    }
                    DepFrame::Candidate(d) => d,
                };
                if (cand.width, cand.height) != (w, h) {
                    unusable.size_mismatch += 1;
                    return Err(Outcome::Unusable);
                }
                deps_read += 1;
                let t_dep = Instant::now();
                let read = unsafe {
                    metal.with_readable_bytes(&cand, &queue_synced, wait_budget, pixels::read_luma)
                };
                readback_ms += t_dep.elapsed().as_secs_f64() * 1000.0;
                let (luma_res, _) = match read {
                    Ok(v) => v,
                    Err(e @ MetalError::Timeout { .. }) => {
                        dep_timeout = Some(e);
                        return Err(Outcome::Unusable);
                    }
                    Err(MetalError::NotBuffer) => {
                        unusable.not_buffer += 1;
                        return Err(Outcome::Unusable);
                    }
                    Err(MetalError::BadGeometry(_)) => {
                        unusable.bad_geometry += 1;
                        return Err(Outcome::Unusable);
                    }
                    Err(MetalError::Other(_)) => {
                        unusable.read_err += 1;
                        return Err(Outcome::Unusable);
                    }
                };
                let (luma, max_a) = match luma_res {
                    Ok(v) => v,
                    Err(_) => {
                        unusable.read_err += 1;
                        return Err(Outcome::Unusable);
                    }
                };
                if cur_has_alpha && max_a <= 0.0 {
                    unusable.transparent += 1;
                    return Err(Outcome::Unusable);
                }
                Ok(luma)
            },
            &is_dup,
        )
    };

    // Abort immediately on timeout: do not compute blur or enqueue an output write.
    if let Some(err) = dep_timeout {
        return Err(err);
    }

    let probes: Vec<(&'static str, u32, &'static str)> = sel
        .probes
        .into_iter()
        .map(|(dir, k, outcome)| (if dir < 0 { "lprev" } else { "lnext" }, k, outcome.name()))
        .collect();
    let prev_k = sel.prev.as_ref().map(|p| p.0);
    let next_k = sel.next.as_ref().map(|n| n.0);
    let fd = sel.fd;

    // 3. Run the existing CPU optical-flow blur engine identically to `effect::render`.
    let mut out_rgba = vec![0.0f32; cur_rgba.len()];
    let stats = effect::call_core_owned(
        core::Frame {
            width: w,
            height: h,
            data: &cur_rgba,
        },
        cur_luma,
        sel.prev.map(|p| p.1).zip(fd.0),
        sel.next.map(|n| n.1).zip(fd.1),
        core_params,
        &mut out_rgba,
    );

    // 4. Write output RGBA into the destination Metal buffer (using the remaining GPU wait budget).
    let t_wr = Instant::now();
    let out_storage =
        unsafe { metal.write_rgba_to_buffer(&out_rgba, out, &queue_synced, wait_budget) }?;
    let write_ms = t_wr.elapsed().as_secs_f64() * 1000.0;

    Ok(GpuRenderReport {
        stats,
        deps_read,
        prev_k,
        next_k,
        fd,
        probes,
        unusable,
        dup_ref: "inFrames[0]",
        readback_ms,
        write_ms,
        cur_storage,
        out_storage,
    })
}

// ───────────────────────────── output allocation & safe pass-through ─────────────────────────────

/// Allocates a new GPU `PPixHand` matching `cur`'s format and dimensions, wrapped in an RAII
/// `OwnedGpuPPix` disposer so any early return or panic disposes it cleanly.
unsafe fn alloc_output_ppix(
    gpu_suite: *const sys::PrSDKGPUDeviceSuite,
    ppix_suite: *const sys::PrSDKPPixSuite,
    device_index: u32,
    rp: &sys::PrGPUFilterRenderParams,
    cur: &FrameInspect,
) -> Result<OwnedGpuPPix, String> {
    let mut par_num = rp.inRenderPARNum as i32;
    let mut par_den = rp.inRenderPARDen as i32;
    // SAFETY: `ppix_suite` is a live acquired suite; `cur.ppix` is validated non-null by caller.
    if let Some(f) =
        unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*ppix_suite).GetPixelAspectRatio)) }
    {
        let (mut n, mut d) = (0u32, 0u32);
        if unsafe { f(cur.ppix, &mut n, &mut d) } == sys::suiteError_NoError && d != 0 {
            par_num = n as i32;
            par_den = d as i32;
        }
    }
    if par_den == 0 {
        par_num = 1;
        par_den = 1;
    }
    // SAFETY: `gpu_suite` is a live acquired suite.
    let Some(create_fn) =
        (unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*gpu_suite).CreateGPUPPix)) })
    else {
        return Err("no_create_gpu_ppix".to_string());
    };
    let mut new_h: sys::PPixHand = std::ptr::null_mut();
    // SAFETY: Valid host device index, format, dimensions, PAR, field type, and stack out-pointer.
    let err = unsafe {
        create_fn(
            device_index,
            cur.raw_fmt,
            cur.width,
            cur.height,
            par_num,
            par_den,
            rp.inRenderFieldType,
            &mut new_h,
        )
    };
    if err != sys::suiteError_NoError || new_h.is_null() {
        return Err(format!("create_gpu_ppix_err:{err}"));
    }
    // SAFETY: `new_h` was just created by `CreateGPUPPix`; `ppix_suite` outlives the guard.
    Ok(unsafe { OwnedGpuPPix::new(ppix_suite, new_h) })
}

/// Outcome of attempting to deliver an unmodified copy of `inFrames[0]` on a render fallback.
struct PassthroughResult {
    code: sys::prSuiteError,
    leaked_in_flight_out: bool,
}

/// Enforces Rules 1 and 2 on every fallback path:
/// * NEVER writes into any input frame (`inFrames[0..N]`).
/// * Returns `suiteError_NoError` ONLY when `*out_frame` holds either `inFrames[0]` itself (when
///   it already aliased `inFrames[0]`) or a verified row-by-row copy of `inFrames[0]`.
/// * Otherwise returns `suiteError_Fail`.
/// * When `allow_gpu_copy` is false (e.g. a GPU wait already timed out), never enqueues another
///   GPU wait.
#[allow(clippy::too_many_arguments)]
unsafe fn finish_passthrough(
    inst: &GpuFilterInstance,
    gpu_suite: *const sys::PrSDKGPUDeviceSuite,
    ppix_suite: *const sys::PrSDKPPixSuite,
    rp: &sys::PrGPUFilterRenderParams,
    frame_slice: &[sys::PPixHand],
    frames_truncated: bool,
    cur: &FrameInspect,
    host_out_in: sys::PPixHand,
    out_frame: *mut sys::PPixHand,
    allow_gpu_copy: bool,
    wait_budget: &Cell<Duration>,
) -> PassthroughResult {
    // Case A: `*out_frame` already aliases `inFrames[0]` -> zero-cost pass-through!
    if !cur.ppix.is_null() && host_out_in == cur.ppix {
        return PassthroughResult {
            code: sys::suiteError_NoError,
            leaked_in_flight_out: false,
        };
    }

    // Without GPU copy permission (e.g. already timed out), we cannot populate a separate output.
    if !allow_gpu_copy {
        return PassthroughResult {
            code: sys::suiteError_Fail,
            leaked_in_flight_out: false,
        };
    }

    let Some(cur_desc) = cur.as_metal_desc() else {
        return PassthroughResult {
            code: sys::suiteError_Fail,
            leaked_in_flight_out: false,
        };
    };
    if unsafe { MetalContext::validate_buffer(&cur_desc) }.is_err() {
        return PassthroughResult {
            code: sys::suiteError_Fail,
            leaked_in_flight_out: false,
        };
    }

    // Case B: `*out_frame` is null, aliases a dependency input `inFrames[k]` (k >= 1), or
    // `in_frame_count` exceeded our inspected slice so aliasing cannot be ruled out.
    // NEVER write into `host_out_in`: allocate a fresh GPU PPix and copy `cur` into it.
    if host_out_in.is_null() || frames_truncated || frame_slice.contains(&host_out_in) {
        let Ok(fresh) = (unsafe {
            alloc_output_ppix(gpu_suite, ppix_suite, inst.device_index, rp, cur)
        }) else {
            return PassthroughResult {
                code: sys::suiteError_Fail,
                leaked_in_flight_out: false,
            };
        };
        let fresh_info = unsafe { inspect_gpu_ppix(gpu_suite, ppix_suite, fresh.handle()) };
        let Some(fresh_desc) = fresh_info.as_metal_desc() else {
            return PassthroughResult {
                code: sys::suiteError_Fail,
                leaked_in_flight_out: false,
            };
        };
        match unsafe { inst.metal.copy_frame_rows(&cur_desc, &fresh_desc, wait_budget) } {
            Ok(()) => {
                // SAFETY: `out_frame` is checked non-null by `cb_render`.
                unsafe {
                    *out_frame = fresh.take();
                }
                PassthroughResult {
                    code: sys::suiteError_NoError,
                    leaked_in_flight_out: false,
                }
            }
            Err(e) => {
                let leaked = e.targets_output();
                if leaked {
                    fresh.leak_in_flight();
                }
                PassthroughResult {
                    code: sys::suiteError_Fail,
                    leaked_in_flight_out: leaked,
                }
            }
        }
    } else {
        // Case C: `*out_frame` is a distinct non-null host frame. Copy `cur` into it (supporting
        // differing `row_bytes`). Return `NoError` ONLY if the copy succeeded.
        let out_info = unsafe { inspect_gpu_ppix(gpu_suite, ppix_suite, host_out_in) };
        let Some(out_desc) = out_info.as_metal_desc() else {
            return PassthroughResult {
                code: sys::suiteError_Fail,
                leaked_in_flight_out: false,
            };
        };
        match unsafe { inst.metal.copy_frame_rows(&cur_desc, &out_desc, wait_budget) } {
            Ok(()) => PassthroughResult {
                code: sys::suiteError_NoError,
                leaked_in_flight_out: false,
            },
            Err(_) => PassthroughResult {
                code: sys::suiteError_Fail,
                leaked_in_flight_out: false,
            },
        }
    }
}

// ───────────────────────────── Premiere GPU callbacks ─────────────────────────────

unsafe extern "C" fn cb_create_instance(
    io_instance_data: *mut sys::PrGPUFilterInstance,
) -> sys::prSuiteError {
    let log = diag::enabled();
    let res = catch_unwind(AssertUnwindSafe(|| {
        autoreleasepool(|_| unsafe {
            if io_instance_data.is_null() {
                if log {
                    diag::append(
                        &format!("event=create_instance path=gpu version={VERSION} result=declined reason=null_instance_data"),
                    );
                }
                return sys::suiteError_InvalidParms;
            }
            // Always start with a clean null private data pointer before any fallible work.
            std::ptr::write_unaligned(
                std::ptr::addr_of_mut!((*io_instance_data).ioPrivatePluginData),
                std::ptr::null_mut(),
            );

            let pi_suites =
                std::ptr::read_unaligned(std::ptr::addr_of!((*io_instance_data).piSuites));
            let device_index =
                std::ptr::read_unaligned(std::ptr::addr_of!((*io_instance_data).inDeviceIndex));
            let timeline_id =
                std::ptr::read_unaligned(std::ptr::addr_of!((*io_instance_data).inTimelineID));
            let node_id =
                std::ptr::read_unaligned(std::ptr::addr_of!((*io_instance_data).inNodeID));

            // 1. Kill switch: /tmp/tblur-no-gpu forces the CPU path without reinstalling.
            if diag::gpu_disabled() {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         result=declined reason=kill_switch"
                    ));
                }
                return sys::suiteError_Fail;
            }

            let Some(sp) = sp_basic_from_pi_suites(pi_suites) else {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         result=declined reason=no_sp_basic"
                    ));
                }
                return sys::suiteError_Fail;
            };

            // 2. Framework gate: Metal only.
            let Some(gpu_suite) = RawSuite::<sys::PrSDKGPUDeviceSuite>::acquire(
                sp,
                sys::kPrSDKGPUDeviceSuite,
                sys::kPrSDKGPUDeviceSuiteVersion,
            ) else {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         result=declined reason=no_gpu_device_suite"
                    ));
                }
                return sys::suiteError_Fail;
            };

            let Some(get_dev_info) =
                std::ptr::read_unaligned(std::ptr::addr_of!((*gpu_suite.ptr).GetDeviceInfo))
            else {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         result=declined reason=no_get_device_info"
                    ));
                }
                return sys::suiteError_Fail;
            };
            let mut dev_info: sys::PrGPUDeviceInfo = std::mem::zeroed();
            let err = get_dev_info(sys::kPrSDKGPUDeviceSuiteVersion, device_index, &mut dev_info);
            if err != sys::suiteError_NoError {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         result=declined reason=get_device_info_err({err})"
                    ));
                }
                return sys::suiteError_Fail;
            }

            let fw = dev_info.outDeviceFramework;
            let meets = dev_info.outMeetsMinimumRequirementsForAcceleration != 0;
            let dev_ptr = dev_info.outDeviceHandle;
            let q_ptr = dev_info.outCommandQueueHandle;
            let is_metal = fw == sys::PrGPUDeviceFramework_PrGPUDeviceFramework_Metal;

            if !is_metal {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         framework={}({fw}) meets_min_req={meets} \
                         result=declined reason=unsupported_framework({})",
                        framework_name(fw),
                        framework_name(fw),
                    ));
                }
                return sys::suiteError_Fail;
            }
            if !meets {
                if log {
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         framework=Metal({fw}) meets_min_req=false \
                         result=declined reason=below_min_req"
                    ));
                }
                return sys::suiteError_Fail;
            }

            // 3. Inspect effect node (`EffectNode::RuntimeInstanceID`, type, flags, owner) via
            //    `PrSDKVideoSegmentSuite`, then gate using the PF synthetic map (or debug-only
            //    force switch).
            let node_cls = match inspect_node_is_adjustment(sp, pi_suites, node_id) {
                Ok(cls) => cls,
                Err(why) => {
                    if log {
                        diag::append(&format!(
                            "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                             timeline_id={timeline_id} device_index={device_index} \
                             framework=Metal({fw}) meets_min_req=true \
                             result=declined reason={why}"
                        ));
                    }
                    return sys::suiteError_Fail;
                }
            };

            if log {
                maybe_dump_segment_graph(sp, pi_suites, timeline_id, node_id, &node_cls);
            }

            let (dep_steps, dep_steps_override) = diag::resolve_gpu_steps(log, GPU_DEPENDENCY_STEPS);
            let fault_mode = diag::resolve_gpu_fault(log);
            let force_mode = diag::resolve_gpu_force(log);
            let pf_synthetic = node_cls
                .runtime_instance_id
                .and_then(|id| instance_registry::global().lookup(id));
            let pf_map_str = match (node_cls.runtime_instance_id, pf_synthetic) {
                (None, _) => "no_id",
                (Some(_), Some(true)) => "synthetic",
                (Some(_), Some(false)) => "not_synthetic",
                (Some(_), None) => "unknown",
            };
            let gate = decide_gpu_gate(
                force_mode,
                &node_cls.node_type,
                node_cls.owner_id,
                node_cls.owner_err,
                node_cls.runtime_instance_id,
                pf_synthetic,
            );

            let log_with_node = |result: &str, reason: &str, dev_class: &str, q_class: &str| {
                if log {
                    let rtid_raw = node_cls
                        .runtime_instance_id_raw
                        .as_deref()
                        .map(diag::sanitize_token)
                        .unwrap_or_else(|| "n/a".to_string());
                    let owner_type = node_cls
                        .owner_type
                        .as_deref()
                        .map(diag::sanitize_token)
                        .unwrap_or_else(|| "n/a".to_string());
                    let owner_flags = node_cls
                        .owner_flags
                        .map(|f| format!("0x{f:x}"))
                        .unwrap_or_else(|| "n/a".to_string());
                    let owner_in0_type = node_cls
                        .owner_input0_type
                        .as_deref()
                        .map(diag::sanitize_token)
                        .unwrap_or_else(|| "n/a".to_string());
                    let owner_in0_flags = node_cls
                        .owner_input0_flags
                        .map(|f| format!("0x{f:x}"))
                        .unwrap_or_else(|| "n/a".to_string());
                    diag::append(&format!(
                        "event=create_instance path=gpu version={VERSION} node_id={node_id} \
                         timeline_id={timeline_id} device_index={device_index} \
                         framework=Metal({fw}) meets_min_req=true \
                         dep_steps={dep_steps} dep_steps_override={dep_steps_override} \
                         fault_mode={} forced={} \
                         runtime_instance_id={} runtime_instance_id_raw={rtid_raw} runtime_id_err={} \
                         pf_map={pf_map_str} \
                         node_type={} node_flags=0x{:x} owner_id={} owner_type={owner_type} \
                         owner_flags={owner_flags} owner_err={} \
                         owner_input0_type={owner_in0_type} owner_input0_flags={owner_in0_flags} \
                         matched_by={} device_class={} queue_class={} result={result} reason={reason}",
                        fault_mode.as_str(),
                        gate.forced,
                        diag::opt(node_cls.runtime_instance_id),
                        diag::opt(node_cls.runtime_id_err),
                        diag::sanitize_token(&node_cls.node_type),
                        node_cls.node_flags,
                        diag::opt(node_cls.owner_id),
                        diag::opt(node_cls.owner_err),
                        node_cls.matched_by,
                        diag::sanitize_token(dev_class),
                        diag::sanitize_token(q_class),
                    ));
                }
            };

            if !gate.accept {
                log_with_node("declined", gate.reason, "skipped", "skipped");
                return sys::suiteError_Fail;
            }

            let (dev_class, q_class) = if log {
                (objc_class_name(dev_ptr), objc_class_name(q_ptr))
            } else {
                (String::new(), String::new())
            };

            let metal = match MetalContext::from_raw_handles(dev_ptr, q_ptr) {
                Ok(m) => m,
                Err(e) => {
                    log_with_node(
                        "declined",
                        &format!("metal_init_err({})", diag::sanitize_token(&e)),
                        &dev_class,
                        &q_class,
                    );
                    return sys::suiteError_Fail;
                }
            };

            let boxed = Box::new(GpuFilterInstance {
                device_index,
                timeline_id,
                node_id,
                dep_steps,
                fault_mode,
                metal,
                #[cfg(test)]
                kill_switch_path: None,
            });
            std::ptr::write_unaligned(
                std::ptr::addr_of_mut!((*io_instance_data).ioPrivatePluginData),
                Box::into_raw(boxed) as *mut c_void,
            );
            std::ptr::write_unaligned(std::ptr::addr_of_mut!((*io_instance_data).outIsRealtime), 1);

            log_with_node("ok", gate.reason, &dev_class, &q_class);
            sys::suiteError_NoError
        })
    }));

    match res {
        Ok(code) => code,
        Err(_) => {
            if log {
                diag::append(
                    &format!("event=create_instance path=gpu version={VERSION} result=declined reason=panic"),
                );
            }
            sys::suiteError_Fail
        }
    }
}

unsafe extern "C" fn cb_dispose_instance(
    io_instance_data: *mut sys::PrGPUFilterInstance,
) -> sys::prSuiteError {
    let res = catch_unwind(AssertUnwindSafe(|| {
        autoreleasepool(|_| unsafe {
            if io_instance_data.is_null() {
                return sys::suiteError_NoError;
            }
            // Take and immediately null out `ioPrivatePluginData` BEFORE dropping, so even a panic
            // during drop can never leave a dangling pointer or cause a double-free.
            let raw = std::ptr::read_unaligned(std::ptr::addr_of!(
                (*io_instance_data).ioPrivatePluginData
            ));
            std::ptr::write_unaligned(
                std::ptr::addr_of_mut!((*io_instance_data).ioPrivatePluginData),
                std::ptr::null_mut(),
            );
            if !raw.is_null() {
                // SAFETY: `raw` was produced by `Box::into_raw` in `cb_create_instance` and has
                // not been freed before. Dropped inside `autoreleasepool`.
                drop(Box::from_raw(raw as *mut GpuFilterInstance));
            }
            sys::suiteError_NoError
        })
    }));
    res.unwrap_or(sys::suiteError_NoError)
}

unsafe extern "C" fn cb_get_frame_dependencies(
    in_instance_data: *mut sys::PrGPUFilterInstance,
    in_render_params: *const sys::PrGPUFilterRenderParams,
    io_query_index: *mut sys::csSDK_int32,
    out_frame_dependencies: *mut sys::PrGPUFilterFrameDependency,
) -> sys::prSuiteError {
    let res = catch_unwind(AssertUnwindSafe(|| {
        autoreleasepool(|_| unsafe {
            if in_instance_data.is_null()
                || in_render_params.is_null()
                || io_query_index.is_null()
                || out_frame_dependencies.is_null()
            {
                return sys::suiteError_NotImplemented;
            }

            let priv_ptr = std::ptr::read_unaligned(std::ptr::addr_of!(
                (*in_instance_data).ioPrivatePluginData
            ));
            if priv_ptr.is_null() {
                return sys::suiteError_NotImplemented;
            }
            // SAFETY: Borrowed immutably; lifetime lasts until `cb_dispose_instance`.
            let inst: &GpuFilterInstance = &*(priv_ptr as *const GpuFilterInstance);

            let pi_suites =
                std::ptr::read_unaligned(std::ptr::addr_of!((*in_instance_data).piSuites));
            let rp = std::ptr::read_unaligned(in_render_params);
            let query_in = *io_query_index;

            // If reading parameters fails here, default to enabled blur so dependencies are still
            // provided to Render (where a parameter read failure will return an error so Premiere
            // falls back to the CPU effect path).
            let core_params = sp_basic_from_pi_suites(pi_suites)
                .and_then(|sp| read_gpu_core_params(sp, inst.node_id, rp.inClipTime).ok())
                .map(|(p, _)| p)
                .unwrap_or_default();

            match gpu_dependency_at_steps(
                &core_params,
                rp.inSequenceTime,
                rp.inRenderTicksPerFrame,
                query_in,
                inst.dep_steps,
            ) {
                Some(dep) => {
                    let mut d: sys::PrGPUFilterFrameDependency = std::mem::zeroed();
                    d.outDependencyType =
                        sys::PrGPUFilterFrameDependencyType_PrGPUDependency_InputFrame;
                    d.outTrackID = dep.track_id;
                    d.outSequenceTime = dep.sequence_time;
                    std::ptr::write_unaligned(out_frame_dependencies, d);
                    *io_query_index = dep.next_query_index;
                    sys::suiteError_NoError
                }
                None => sys::suiteError_NotImplemented,
            }
        })
    }));

    res.unwrap_or(sys::suiteError_NotImplemented)
}

unsafe extern "C" fn cb_precompute(
    _in_instance_data: *mut sys::PrGPUFilterInstance,
    _in_render_params: *const sys::PrGPUFilterRenderParams,
    _in_index: sys::csSDK_int32,
    _in_frame: sys::PPixHand,
) -> sys::prSuiteError {
    sys::suiteError_NotImplemented
}

unsafe extern "C" fn cb_render(
    in_instance_data: *mut sys::PrGPUFilterInstance,
    in_render_params: *const sys::PrGPUFilterRenderParams,
    in_frames: *const sys::PPixHand,
    in_frame_count: sys::csSDK_size_t,
    out_frame: *mut sys::PPixHand,
) -> sys::prSuiteError {
    let log = diag::enabled();
    let t0 = log.then(Instant::now);

    let res = catch_unwind(AssertUnwindSafe(|| {
        autoreleasepool(|_| unsafe {
            if in_instance_data.is_null()
                || in_render_params.is_null()
                || in_frames.is_null()
                || out_frame.is_null()
                || in_frame_count == 0
            {
                if log {
                    let wall_ms = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                    diag::append(&format!(
                        "event=render path=gpu result=error(invalid_args) \
                         frame_count={in_frame_count} dup_ref=none unusable={} \
                         ret={} wall_ms={wall_ms:.2}",
                        UnusableCounts::default().format_log(),
                        sys::suiteError_Fail,
                    ));
                }
                return sys::suiteError_Fail;
            }

            let host_out_in = *out_frame;

            let priv_ptr = std::ptr::read_unaligned(std::ptr::addr_of!(
                (*in_instance_data).ioPrivatePluginData
            ));
            if priv_ptr.is_null() {
                if log {
                    let wall_ms = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                    diag::append(&format!(
                        "event=render path=gpu result=error(null_private_data) \
                         dup_ref=none unusable={} ret={} wall_ms={wall_ms:.2}",
                        UnusableCounts::default().format_log(),
                        sys::suiteError_Fail,
                    ));
                }
                return sys::suiteError_Fail;
            }
            // SAFETY: Borrowed immutably; instance stays alive until `cb_dispose_instance`.
            let inst: &GpuFilterInstance = &*(priv_ptr as *const GpuFilterInstance);
            let wait_budget = if inst.fault_mode == diag::GpuFaultMode::Timeout {
                Cell::new(Duration::ZERO)
            } else {
                Cell::new(inst.metal.render_wait_budget())
            };

            let pi_suites =
                std::ptr::read_unaligned(std::ptr::addr_of!((*in_instance_data).piSuites));
            let rp = std::ptr::read_unaligned(in_render_params);
            let seq_time = rp.inSequenceTime;
            let clip_time = rp.inClipTime;
            let tpf = rp.inRenderTicksPerFrame;
            let raw_frame_count = in_frame_count as usize;
            let deps_received = raw_frame_count.saturating_sub(1);

            // Cap slice defensively while always covering 1 + 2 * inst.dep_steps.
            let max_frames = (1 + 2 * inst.dep_steps as usize).max(32);
            let count = raw_frame_count.min(max_frames);
            let frames_truncated = raw_frame_count > count;
            // SAFETY: `in_frames` is checked non-null and host passes `in_frame_count` elements.
            let frame_slice = std::slice::from_raw_parts(in_frames, count);

            let log_outcome = |kind: &str,
                               reason: &str,
                               fmt: &str,
                               size: (i32, i32),
                               declared: usize,
                               dup_ref: &str,
                               unusable: &UnusableCounts,
                               leaked_out: bool,
                               ret: sys::prSuiteError| {
                if log {
                    let wall_ms = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                    diag::append(&format!(
                        "event=render path=gpu result={kind}({}) node_id={} timeline_id={} \
                         dep_steps={} fault_mode={} seq_time={seq_time} clip_time={clip_time} ticks_per_frame={tpf} \
                         size={}x{} fmt={fmt} deps_declared={declared} deps_received={deps_received} deps_read=0 \
                         prev_k=n/a next_k=n/a probes=[] fd_prev=n/a fd_next=n/a \
                         dup_ref={dup_ref} unusable={} leaked_in_flight_out={leaked_out} ret={ret} \
                         used_prev=n/a used_next=n/a max_velocity_px=n/a \
                         readback_ms=0.00 write_ms=0.00 flow_ms=n/a blur_ms=n/a wall_ms={wall_ms:.2}",
                        diag::sanitize_token(reason),
                        inst.node_id,
                        inst.timeline_id,
                        inst.dep_steps,
                        inst.fault_mode.as_str(),
                        size.0,
                        size.1,
                        unusable.format_log(),
                    ));
                }
            };

            // Immediate kill switch: `/tmp/tblur-no-gpu` hands frames on existing GPU instances
            // straight back to Premiere's CPU path without touching any frame.
            if inst.is_gpu_disabled() {
                log_outcome(
                    "error",
                    "kill_switch",
                    "n/a",
                    (0, 0),
                    deps_received,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            }

            // Debug-only fault mode `error`: return an error immediately without touching any frame.
            if inst.fault_mode == diag::GpuFaultMode::Error {
                log_outcome(
                    "error",
                    "fault_error",
                    "n/a",
                    (0, 0),
                    deps_received,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            }

            let Some(sp) = sp_basic_from_pi_suites(pi_suites) else {
                log_outcome(
                    "error",
                    "no_sp_basic",
                    "n/a",
                    (0, 0),
                    deps_received,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            };

            let Some(gpu_suite) = RawSuite::<sys::PrSDKGPUDeviceSuite>::acquire(
                sp,
                sys::kPrSDKGPUDeviceSuite,
                sys::kPrSDKGPUDeviceSuiteVersion,
            ) else {
                log_outcome(
                    "error",
                    "no_gpu_device_suite",
                    "n/a",
                    (0, 0),
                    deps_received,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            };
            let Some(ppix_suite) = RawSuite::<sys::PrSDKPPixSuite>::acquire(
                sp,
                sys::kPrSDKPPixSuite,
                sys::kPrSDKPPixSuiteVersion,
            ) else {
                log_outcome(
                    "error",
                    "no_ppix_suite",
                    "n/a",
                    (0, 0),
                    deps_received,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            };

            let cur = inspect_gpu_ppix(gpu_suite.ptr, ppix_suite.ptr, frame_slice[0]);
            let fmt_str = format!("{}/0x{:08x}", cur.fmt_name, cur.raw_fmt);
            let cur_size = (cur.width, cur.height);

            // Debug-only fault mode `passthrough`: output the current frame unchanged and return
            // success (kept intentionally so both error-fallback and pass-through stay testable).
            if inst.fault_mode == diag::GpuFaultMode::Passthrough {
                let pt = finish_passthrough(
                    inst,
                    gpu_suite.ptr,
                    ppix_suite.ptr,
                    &rp,
                    frame_slice,
                    frames_truncated,
                    &cur,
                    host_out_in,
                    out_frame,
                    true,
                    &wait_budget,
                );
                let kind = if pt.code == sys::suiteError_NoError {
                    "passthrough"
                } else {
                    "error"
                };
                log_outcome(
                    kind,
                    "fault_passthrough",
                    &fmt_str,
                    cur_size,
                    deps_received,
                    "none",
                    &UnusableCounts::default(),
                    pt.leaked_in_flight_out,
                    pt.code,
                );
                return pt.code;
            }

            let (core_params, _use_source_rate) =
                match read_gpu_core_params(sp, inst.node_id, clip_time) {
                    Ok(v) => v,
                    Err(e) => {
                        log_outcome(
                            "error",
                            &e,
                            &fmt_str,
                            cur_size,
                            deps_received,
                            "none",
                            &UnusableCounts::default(),
                            false,
                            sys::suiteError_Fail,
                        );
                        return sys::suiteError_Fail;
                    }
                };

            let blur_disabled = !core_params.blur_amount.is_finite()
                || core_params.blur_amount <= 0.0
                || !core_params.motion_sensitivity.is_finite()
                || core_params.motion_sensitivity <= 0.0;

            if !blur_disabled && (tpf <= 0 || inst.dep_steps == 0) {
                log_outcome(
                    "error",
                    "invalid_ticks_per_frame",
                    &fmt_str,
                    cur_size,
                    0,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            }

            let declared = gpu_declared_dependencies(&core_params, seq_time, tpf, inst.dep_steps);
            let deps_declared = declared.len();
            let expected_frames = 1 + deps_declared;

            // Item 10: if received frame count disagrees with declared count, do not guess which
            // neighbour is which — return an error so Premiere re-renders via CPU.
            if raw_frame_count != expected_frames {
                log_outcome(
                    "error",
                    &format!("frame_count_mismatch:{deps_received}!={deps_declared}"),
                    &fmt_str,
                    cur_size,
                    deps_declared,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            }

            // Items 7 & 9: validate `cur` geometry, `MTLBuffer` type, and backing buffer length
            // BEFORE ever calling `CreateGPUPPix`.
            let Some(cur_desc) = cur.as_metal_desc() else {
                log_outcome(
                    "error",
                    "invalid_or_unsupported_cur_frame",
                    &fmt_str,
                    cur_size,
                    deps_declared,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            };
            if let Err(e) = MetalContext::validate_buffer(&cur_desc) {
                log_outcome(
                    "error",
                    &format!("invalid_cur_buffer:{}", e.summary()),
                    &fmt_str,
                    cur_size,
                    deps_declared,
                    "none",
                    &UnusableCounts::default(),
                    false,
                    sys::suiteError_Fail,
                );
                return sys::suiteError_Fail;
            }

            // Correct no-blur copy: when blur amount or motion sensitivity is <= 0, the unchanged
            // current frame is the exact desired output.
            if blur_disabled {
                let pt = finish_passthrough(
                    inst,
                    gpu_suite.ptr,
                    ppix_suite.ptr,
                    &rp,
                    frame_slice,
                    frames_truncated,
                    &cur,
                    host_out_in,
                    out_frame,
                    true,
                    &wait_budget,
                );
                let (kind, reason) = if pt.code == sys::suiteError_NoError {
                    ("ok", "no_blur")
                } else {
                    ("error", "no_blur_copy_failed")
                };
                log_outcome(
                    kind,
                    reason,
                    &fmt_str,
                    cur_size,
                    deps_declared,
                    "none",
                    &UnusableCounts::default(),
                    pt.leaked_in_flight_out,
                    pt.code,
                );
                return pt.code;
            }

            // Never overwrite `inFrames[0]` or any dependency frame in `inFrames`. Allocate a
            // fresh output PPix guarded by `OwnedGpuPPix` (Item 8) whenever `*out_frame` is null
            // or aliases any input frame.
            let need_new_out = host_out_in.is_null() || frame_slice.contains(&host_out_in);
            let mut created_guard: Option<OwnedGpuPPix> = None;
            let (target_ppix, out_alloc) = if need_new_out {
                match alloc_output_ppix(gpu_suite.ptr, ppix_suite.ptr, inst.device_index, &rp, &cur)
                {
                    Ok(g) => {
                        let h = g.handle();
                        created_guard = Some(g);
                        (h, "created")
                    }
                    Err(e) => {
                        log_outcome(
                            "error",
                            &e,
                            &fmt_str,
                            cur_size,
                            deps_declared,
                            "none",
                            &UnusableCounts::default(),
                            false,
                            sys::suiteError_Fail,
                        );
                        return sys::suiteError_Fail;
                    }
                }
            } else {
                (host_out_in, "reused")
            };

            let out_info = inspect_gpu_ppix(gpu_suite.ptr, ppix_suite.ptr, target_ppix);
            let out_desc_res = out_info
                .as_metal_desc()
                .ok_or_else(|| "invalid_out_frame".to_string())
                .and_then(|d| {
                    if d.layout != cur_desc.layout
                        || d.width != cur_desc.width
                        || d.height != cur_desc.height
                    {
                        return Err("out_frame_mismatch".to_string());
                    }
                    MetalContext::validate_buffer(&d)
                        .map_err(|e| format!("invalid_out_buffer:{}", e.summary()))?;
                    Ok(d)
                });
            let out_desc = match out_desc_res {
                Ok(d) => d,
                Err(why) => {
                    drop(created_guard);
                    log_outcome(
                        "error",
                        &why,
                        &fmt_str,
                        cur_size,
                        deps_declared,
                        "none",
                        &UnusableCounts::default(),
                        false,
                        sys::suiteError_Fail,
                    );
                    return sys::suiteError_Fail;
                }
            };

            // Map received dependency handles into the canonical `2 * inst.dep_steps` interleaved
            // slots (NO GPU pixel readback yet). Any undeclared negative-time neighbour remains
            // `DepFrame::Missing`, matching the CPU path when a checkout cannot be obtained.
            let mut dep_frames = vec![DepFrame::Missing; 2 * inst.dep_steps as usize];
            for (decl, &h) in declared.iter().zip(&frame_slice[1..]) {
                if let Some(slot) = dep_slice_index(decl.dir, decl.k) {
                    dep_frames[slot] = inspect_gpu_ppix(gpu_suite.ptr, ppix_suite.ptr, h).as_dep_frame();
                }
            }

            match render_metal_frames(
                &inst.metal,
                &core_params,
                &cur_desc,
                &dep_frames,
                &out_desc,
                inst.dep_steps,
                &wait_budget,
            ) {
                Ok(rep) => {
                    if let Some(g) = created_guard.take() {
                        *out_frame = g.take();
                    }
                    if log {
                        let wall_ms = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                        let probes_s: Vec<String> = rep
                            .probes
                            .iter()
                            .map(|(kind, k, outcome)| format!("{kind}@k{k}:{outcome}"))
                            .collect();
                        diag::append(&format!(
                            "event=render path=gpu result=ok node_id={} timeline_id={} \
                             dep_steps={} fault_mode={} seq_time={seq_time} clip_time={clip_time} ticks_per_frame={tpf} \
                             size={}x{} fmt={fmt_str} storage={}/{} out_alloc={out_alloc} \
                             deps_declared={deps_declared} deps_received={deps_received} deps_read={} \
                             prev_k={} next_k={} probes=[{}] fd_prev={} fd_next={} \
                             dup_ref={} unusable={} leaked_in_flight_out=false ret=0 \
                             used_prev={} used_next={} max_velocity_px={:.2} \
                             readback_ms={:.2} write_ms={:.2} flow_ms={:.1} blur_ms={:.1} wall_ms={wall_ms:.2}",
                            inst.node_id,
                            inst.timeline_id,
                            inst.dep_steps,
                            inst.fault_mode.as_str(),
                            cur_desc.width,
                            cur_desc.height,
                            rep.cur_storage,
                            rep.out_storage,
                            rep.deps_read,
                            diag::opt(rep.prev_k),
                            diag::opt(rep.next_k),
                            probes_s.join(","),
                            diag::opt(rep.fd.0),
                            diag::opt(rep.fd.1),
                            rep.dup_ref,
                            rep.unusable.format_log(),
                            rep.stats.used_prev,
                            rep.stats.used_next,
                            rep.stats.max_velocity_px,
                            rep.readback_ms,
                            rep.write_ms,
                            rep.stats.flow_ms,
                            rep.stats.blur_ms,
                        ));
                    }
                    sys::suiteError_NoError
                }
                Err(err) => {
                    // Item 4: if a GPU timeout left an in-flight blit targeting `created_guard`,
                    // deliberately leak it (do NOT call `Dispose` while GPU work targets it).
                    let mut leaked_out = false;
                    if err.targets_output() {
                        if let Some(g) = created_guard.take() {
                            g.leak_in_flight();
                            leaked_out = true;
                        }
                    } else {
                        drop(created_guard);
                    }
                    log_outcome(
                        "error",
                        &err.summary(),
                        &fmt_str,
                        cur_size,
                        deps_declared,
                        "none",
                        &UnusableCounts::default(),
                        leaked_out,
                        sys::suiteError_Fail,
                    );
                    sys::suiteError_Fail
                }
            }
        })
    }));

    match res {
        Ok(code) => code,
        Err(_) => {
            if log {
                let wall_ms = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                diag::append(&format!(
                    "event=render path=gpu result=error(panic) dup_ref=none unusable={} \
                     ret={} wall_ms={wall_ms:.2}",
                    UnusableCounts::default().format_log(),
                    sys::suiteError_Fail,
                ));
            }
            sys::suiteError_Fail
        }
    }
}

// ───────────────────────────── exported entry point ─────────────────────────────

/// Exported Premiere Pro GPU filter entry point (`xGPUFilterEntry`).
///
/// Leaves `outFilterInfo->outMatchName` zeroed (NULL `PrSDKString`), which Adobe's SDK documents
/// as defaulting to this bundle's PiPL `AE_Effect_Match_Name` (`"TBLUR Motion Blur"`).
#[unsafe(no_mangle)]
#[allow(non_snake_case)]
pub unsafe extern "C" fn xGPUFilterEntry(
    host_interface_version: sys::csSDK_uint32,
    io_index: *mut sys::csSDK_int32,
    is_startup: sys::prBool,
    _pi_suites: sys::piSuitesPtr,
    out_filter: *mut sys::PrGPUFilter,
    out_filter_info: *mut sys::PrGPUFilterInfo,
) -> sys::prSuiteError {
    let log = diag::enabled();
    let res = catch_unwind(AssertUnwindSafe(|| {
        autoreleasepool(|_| unsafe {
            if is_startup != 0 {
                if io_index.is_null() || out_filter.is_null() || out_filter_info.is_null() {
                    if log {
                        diag::append(
                            &format!("event=gpu_entry path=gpu version={VERSION} startup=true result=err(null_arg)"),
                        );
                    }
                    return sys::suiteError_InvalidParms;
                }
                let index = *io_index;
                if index != 0 {
                    return sys::suiteError_InvalidParms;
                }

                let filter = sys::PrGPUFilter {
                    CreateInstance: Some(cb_create_instance),
                    DisposeInstance: Some(cb_dispose_instance),
                    GetFrameDependencies: Some(cb_get_frame_dependencies),
                    Precompute: Some(cb_precompute),
                    Render: Some(cb_render),
                };
                std::ptr::write_unaligned(out_filter, filter);

                let out_ver = host_interface_version.min(sys::PrSDKGPUFilterInterfaceVersion);
                let info = sys::PrGPUFilterInfo {
                    outInterfaceVersion: out_ver,
                    outMatchName: std::mem::zeroed(),
                };
                std::ptr::write_unaligned(out_filter_info, info);

                if log {
                    diag::append(&format!(
                        "event=gpu_entry path=gpu version={VERSION} startup=true \
                         host_ver={host_interface_version} out_ver={out_ver} index={index} \
                         dep_steps={GPU_DEPENDENCY_STEPS} result=ok"
                    ));
                }
            } else if log {
                diag::append(&format!("event=gpu_entry path=gpu version={VERSION} startup=false result=ok"));
            }
            sys::suiteError_NoError
        })
    }));

    res.unwrap_or(sys::suiteError_Fail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::ptr::NonNull;
    use std::sync::{Mutex, OnceLock};

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLCreateSystemDefaultDevice, MTLDevice, MTLResourceOptions, MTLSharedEvent,
    };

    // ── helpers to build real Metal buffers on this Mac's default device ──

    fn make_metal_context() -> (
        Retained<ProtocolObject<dyn MTLDevice>>,
        Retained<ProtocolObject<dyn MTLCommandQueue>>,
        MetalContext,
    ) {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device required on macOS");
        let queue = device.newCommandQueue().expect("command queue");
        let ctx = unsafe {
            MetalContext::from_raw_handles(
                Retained::as_ptr(&device) as *mut c_void,
                Retained::as_ptr(&queue) as *mut c_void,
            )
        }
        .expect("MetalContext");
        (device, queue, ctx)
    }

    fn make_shared_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let ptr = NonNull::new(bytes.as_ptr() as *mut c_void).expect("non-empty");
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("shared MTLBuffer")
        }
    }

    fn make_private_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        bytes: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        autoreleasepool(|_| {
            let staging = make_shared_buffer(device, bytes);
            let priv_buf = device
                .newBufferWithLength_options(bytes.len(), MTLResourceOptions::StorageModePrivate)
                .expect("private MTLBuffer");
            let cmd_buf = queue.commandBuffer().expect("cmd_buf");
            let blit = cmd_buf.blitCommandEncoder().expect("blit");
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &staging,
                    0,
                    &priv_buf,
                    0,
                    bytes.len(),
                );
            }
            blit.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
            priv_buf
        })
    }

    /// Generates a moving textured 32f straight-alpha BGRA buffer with row padding (`shift_x`
    /// shifts the pattern horizontally so optical flow sees real motion).
    fn synth_bgra32f_frame(w: usize, h: usize, rb: usize, shift_x: f32, alpha: f32) -> Vec<u8> {
        let mut buf = vec![0xCDu8; h * rb];
        for y in 0..h {
            for x in 0..w {
                let fx = (x as f32 - shift_x) / w as f32;
                let fy = y as f32 / h as f32;
                let r = (0.5 + 0.35 * (fx * 18.0).sin() * (fy * 12.0).cos()).clamp(0.0, 1.0);
                let g = (0.45 + 0.35 * (fx * 11.0 + fy * 7.0).cos()).clamp(0.0, 1.0);
                let b = (0.4 + 0.3 * (fx * 25.0 - fy * 9.0).sin()).clamp(0.0, 1.0);
                let a = if (x + y) % 13 == 0 { alpha * 0.75 } else { alpha };
                let off = y * rb + x * 16;
                for (slot, val) in [b, g, r, a].iter().enumerate() {
                    buf[off + slot * 4..off + slot * 4 + 4].copy_from_slice(&val.to_ne_bytes());
                }
            }
        }
        buf
    }

    fn quantize_to_16f(
        src32: &[u8],
        w: usize,
        h: usize,
        rb32: usize,
        rb16: usize,
    ) -> (Vec<u8>, Vec<u8>) {
        let mut buf16 = vec![0x5Au8; h * rb16];
        let mut rounded32 = vec![0xCDu8; h * rb32];
        for y in 0..h {
            for x in 0..w {
                let o32 = y * rb32 + x * 16;
                let o16 = y * rb16 + x * 8;
                for c in 0..4 {
                    let f = f32::from_ne_bytes(
                        src32[o32 + c * 4..o32 + c * 4 + 4].try_into().unwrap(),
                    );
                    let h16 = pixels::f32_to_f16(f);
                    buf16[o16 + c * 2..o16 + c * 2 + 2].copy_from_slice(&h16.to_ne_bytes());
                    let f_rounded = pixels::f16_to_f32(h16);
                    rounded32[o32 + c * 4..o32 + c * 4 + 4]
                        .copy_from_slice(&f_rounded.to_ne_bytes());
                }
            }
        }
        (buf16, rounded32)
    }

    fn run_cpu_bgra32f(
        cur_bytes: &[u8],
        prev_bytes: Option<(&[u8], f32)>,
        next_bytes: Option<(&[u8], f32)>,
        w: usize,
        h: usize,
        rb: usize,
        params: &core::Params,
    ) -> Vec<u8> {
        let desc = pixels::ImageDesc {
            width: w,
            height: h,
            rowbytes: rb as isize,
            layout: Layout::Bgra32f,
            alpha: pixels::Alpha::Straight,
        };
        let mut cur_rgba = Vec::new();
        let (cur_luma, _) = pixels::read_rgba_and_luma(cur_bytes, &desc, &mut cur_rgba).unwrap();
        let prev = prev_bytes.map(|(b, fd)| (pixels::read_luma(b, &desc).unwrap().0, fd));
        let next = next_bytes.map(|(b, fd)| (pixels::read_luma(b, &desc).unwrap().0, fd));
        let mut out_rgba = vec![0.0f32; cur_rgba.len()];
        effect::call_core_owned(
            core::Frame {
                width: w,
                height: h,
                data: &cur_rgba,
            },
            cur_luma,
            prev,
            next,
            params,
            &mut out_rgba,
        );
        let mut out_bytes = vec![0u8; h * rb];
        pixels::write_rgba(&out_rgba, &mut out_bytes, &desc).unwrap();
        out_bytes
    }

    #[test]
    fn gpu_32f_bit_identical_to_cpu_shared_and_private_with_padding() {
        let (device, queue, ctx) = make_metal_context();
        let (w, h) = (96usize, 64usize);
        let rb = w * 16 + 64;

        let prev_raw = synth_bgra32f_frame(w, h, rb, -6.0, 1.0);
        let cur_raw = synth_bgra32f_frame(w, h, rb, 0.0, 1.0);
        let next_raw = synth_bgra32f_frame(w, h, rb, 6.0, 1.0);

        let param_cases = [
            core::Params::default(),
            core::Params {
                blur_amount: 1.25,
                motion_sensitivity: 85.0,
                max_blur_percent: 20.0,
                quality: core::Quality::High,
                linear_light: false,
                view: core::View::Output,
            },
            core::Params {
                blur_amount: 0.75,
                motion_sensitivity: 70.0,
                max_blur_percent: 15.0,
                quality: core::Quality::Draft,
                linear_light: true,
                view: core::View::MotionVectors,
            },
        ];

        for use_private in [false, true] {
            let cur_buf = if use_private {
                make_private_buffer(&device, &queue, &cur_raw)
            } else {
                make_shared_buffer(&device, &cur_raw)
            };
            let prev_buf = if use_private {
                make_private_buffer(&device, &queue, &prev_raw)
            } else {
                make_shared_buffer(&device, &prev_raw)
            };
            let next_buf = if use_private {
                make_private_buffer(&device, &queue, &next_raw)
            } else {
                make_shared_buffer(&device, &next_raw)
            };
            let out_buf = device
                .newBufferWithLength_options(
                    h * rb,
                    if use_private {
                        MTLResourceOptions::StorageModePrivate
                    } else {
                        MTLResourceOptions::StorageModeShared
                    },
                )
                .expect("out MTLBuffer");

            let desc_of = |b: &Retained<ProtocolObject<dyn MTLBuffer>>| MetalFrameDesc {
                ptr: Retained::as_ptr(b) as *const c_void,
                width: w,
                height: h,
                row_bytes: rb as i32,
                layout: Layout::Bgra32f,
            };

            let deps = vec![
                DepFrame::Candidate(desc_of(&prev_buf)),
                DepFrame::Candidate(desc_of(&next_buf)),
                DepFrame::Candidate(desc_of(&cur_buf)),
                DepFrame::Candidate(desc_of(&cur_buf)),
                DepFrame::Candidate(desc_of(&cur_buf)),
                DepFrame::Candidate(desc_of(&cur_buf)),
                DepFrame::Candidate(desc_of(&cur_buf)),
                DepFrame::Candidate(desc_of(&cur_buf)),
            ];

            for params in &param_cases {
                let cpu_out = run_cpu_bgra32f(
                    &cur_raw,
                    Some((&prev_raw, 1.0)),
                    Some((&next_raw, 1.0)),
                    w,
                    h,
                    rb,
                    params,
                );

                let budget = Cell::new(crate::metal_io::METAL_RENDER_WAIT_BUDGET);
                let rep = unsafe {
                    render_metal_frames(
                        &ctx,
                        params,
                        &desc_of(&cur_buf),
                        &deps,
                        &desc_of(&out_buf),
                        GPU_DEPENDENCY_STEPS,
                        &budget,
                    )
                }
                .expect("GPU render_metal_frames");

                assert_eq!(rep.deps_read, 2, "only k=1 prev and k=1 next should be read");
                assert_eq!((rep.prev_k, rep.next_k), (Some(1), Some(1)));
                assert_eq!(rep.fd, (Some(1.0), Some(1.0)));
                assert_eq!(rep.dup_ref, "inFrames[0]");
                assert_eq!(rep.unusable, UnusableCounts::default());
                assert_eq!(rep.cur_storage, if use_private { "private" } else { "shared" });
                assert_eq!(rep.out_storage, if use_private { "private" } else { "shared" });
                assert!(rep.stats.used_prev && rep.stats.used_next);

                let synced = Cell::new(false);
                let (gpu_out, _) = unsafe {
                    ctx.with_readable_bytes(&desc_of(&out_buf), &synced, &budget, |b, _| b.to_vec())
                }
                .unwrap();

                for y in 0..h {
                    let row_off = y * rb;
                    let active = w * 16;
                    assert_eq!(
                        &gpu_out[row_off..row_off + active],
                        &cpu_out[row_off..row_off + active],
                        "32f bit mismatch at row {y} (private={use_private}, params={params:?})"
                    );
                }
            }
        }
    }

    #[test]
    fn gpu_16f_matches_cpu_fed_16f_rounded_values_bit_for_bit() {
        let (device, queue, ctx) = make_metal_context();
        let (w, h) = (96usize, 64usize);
        let rb32 = w * 16 + 64;
        let rb16 = w * 8 + 48;

        let prev32 = synth_bgra32f_frame(w, h, rb32, -5.0, 1.0);
        let cur32 = synth_bgra32f_frame(w, h, rb32, 0.0, 1.0);
        let next32 = synth_bgra32f_frame(w, h, rb32, 5.0, 1.0);

        let (prev16, prev32_q) = quantize_to_16f(&prev32, w, h, rb32, rb16);
        let (cur16, cur32_q) = quantize_to_16f(&cur32, w, h, rb32, rb16);
        let (next16, next32_q) = quantize_to_16f(&next32, w, h, rb32, rb16);

        let params = core::Params::default();
        let cpu_out32 = run_cpu_bgra32f(
            &cur32_q,
            Some((&prev32_q, 1.0)),
            Some((&next32_q, 1.0)),
            w,
            h,
            rb32,
            &params,
        );
        let (cpu_out16, _) = quantize_to_16f(&cpu_out32, w, h, rb32, rb16);

        for use_private in [false, true] {
            let cur_buf = if use_private {
                make_private_buffer(&device, &queue, &cur16)
            } else {
                make_shared_buffer(&device, &cur16)
            };
            let prev_buf = if use_private {
                make_private_buffer(&device, &queue, &prev16)
            } else {
                make_shared_buffer(&device, &prev16)
            };
            let next_buf = if use_private {
                make_private_buffer(&device, &queue, &next16)
            } else {
                make_shared_buffer(&device, &next16)
            };
            let out_buf = device
                .newBufferWithLength_options(
                    h * rb16,
                    if use_private {
                        MTLResourceOptions::StorageModePrivate
                    } else {
                        MTLResourceOptions::StorageModeShared
                    },
                )
                .expect("out 16f MTLBuffer");

            let desc_of = |b: &Retained<ProtocolObject<dyn MTLBuffer>>| MetalFrameDesc {
                ptr: Retained::as_ptr(b) as *const c_void,
                width: w,
                height: h,
                row_bytes: rb16 as i32,
                layout: Layout::Bgra16f,
            };
            let deps = vec![
                DepFrame::Candidate(desc_of(&prev_buf)),
                DepFrame::Candidate(desc_of(&next_buf)),
            ];
            let budget = Cell::new(crate::metal_io::METAL_RENDER_WAIT_BUDGET);
            let rep = unsafe {
                render_metal_frames(
                    &ctx,
                    &params,
                    &desc_of(&cur_buf),
                    &deps,
                    &desc_of(&out_buf),
                    GPU_DEPENDENCY_STEPS,
                    &budget,
                )
            }
            .expect("GPU 16f render");
            assert_eq!(rep.deps_read, 2);

            let synced = Cell::new(false);
            let (gpu_out16, _) = unsafe {
                ctx.with_readable_bytes(&desc_of(&out_buf), &synced, &budget, |b, _| b.to_vec())
            }
            .unwrap();

            for y in 0..h {
                let row_off = y * rb16;
                let active = w * 8;
                assert_eq!(
                    &gpu_out16[row_off..row_off + active],
                    &cpu_out16[row_off..row_off + active],
                    "16f bit mismatch at row {y} (private={use_private})"
                );
            }
        }
    }

    #[test]
    fn selection_equivalence_with_cpu_legacy_rule() {
        let (device, _queue, ctx) = make_metal_context();
        let (w, h) = (32usize, 16usize);
        let rb = w * 16 + 32;
        let desc = pixels::ImageDesc {
            width: w,
            height: h,
            rowbytes: rb as isize,
            layout: Layout::Bgra32f,
            alpha: pixels::Alpha::Straight,
        };

        let cur_raw = synth_bgra32f_frame(w, h, rb, 0.0, 1.0);
        let dup_raw = cur_raw.clone();
        let diff_a_raw = synth_bgra32f_frame(w, h, rb, -8.0, 1.0);
        let diff_b_raw = synth_bgra32f_frame(w, h, rb, 8.0, 1.0);
        let empty_raw = vec![0u8; h * rb];

        #[derive(Clone, Copy)]
        enum Slot {
            Missing,
            Empty,
            Dup,
            DiffA,
            DiffB,
        }

        let cur_buf = make_shared_buffer(&device, &cur_raw);
        let dup_buf = make_shared_buffer(&device, &dup_raw);
        let diff_a_buf = make_shared_buffer(&device, &diff_a_raw);
        let diff_b_buf = make_shared_buffer(&device, &diff_b_raw);
        let empty_buf = make_shared_buffer(&device, &empty_raw);
        let out_buf = device
            .newBufferWithLength_options(h * rb, MTLResourceOptions::StorageModeShared)
            .expect("out");

        let metal_desc = |b: &Retained<ProtocolObject<dyn MTLBuffer>>| MetalFrameDesc {
            ptr: Retained::as_ptr(b) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb as i32,
            layout: Layout::Bgra32f,
        };

        let bytes_of = |s: Slot| -> Option<&[u8]> {
            match s {
                Slot::Missing => None,
                Slot::Empty => Some(&empty_raw),
                Slot::Dup => Some(&dup_raw),
                Slot::DiffA => Some(&diff_a_raw),
                Slot::DiffB => Some(&diff_b_raw),
            }
        };
        let gpu_slot = |s: Slot| -> DepFrame {
            match s {
                Slot::Missing => DepFrame::Missing,
                Slot::Empty => DepFrame::Candidate(metal_desc(&empty_buf)),
                Slot::Dup => DepFrame::Candidate(metal_desc(&dup_buf)),
                Slot::DiffA => DepFrame::Candidate(metal_desc(&diff_a_buf)),
                Slot::DiffB => DepFrame::Candidate(metal_desc(&diff_b_buf)),
            }
        };

        let scenarios: [(&str, [Slot; 4], [Slot; 4], (Option<u32>, Option<u32>), (Option<f32>, Option<f32>), u32); 7] = [
            (
                "normal_k1",
                [Slot::DiffA, Slot::DiffA, Slot::DiffA, Slot::DiffA],
                [Slot::DiffB, Slot::DiffB, Slot::DiffB, Slot::DiffB],
                (Some(1), Some(1)),
                (Some(1.0), Some(1.0)),
                2,
            ),
            (
                "dup_at_k1_prev_and_k1_k2_next",
                [Slot::Dup, Slot::DiffA, Slot::DiffA, Slot::DiffA],
                [Slot::Dup, Slot::Dup, Slot::DiffB, Slot::DiffB],
                (Some(2), Some(3)),
                (Some(4.0), Some(4.0)),
                5,
            ),
            (
                "dup_at_k1_k2_k3_both_sides",
                [Slot::Dup, Slot::Dup, Slot::Dup, Slot::DiffA],
                [Slot::Dup, Slot::Dup, Slot::Dup, Slot::DiffB],
                (Some(4), Some(4)),
                (Some(7.0), Some(7.0)),
                8,
            ),
            (
                "missing_prev_side_layer_start",
                [Slot::Missing, Slot::Missing, Slot::Missing, Slot::Missing],
                [Slot::Dup, Slot::DiffB, Slot::DiffB, Slot::DiffB],
                (None, Some(2)),
                (None, Some(2.0)),
                2,
            ),
            (
                "empty_transparent_prev_side_layer_start",
                [Slot::Empty, Slot::Empty, Slot::Empty, Slot::Empty],
                [Slot::DiffB, Slot::DiffB, Slot::DiffB, Slot::DiffB],
                (None, Some(1)),
                (None, Some(1.0)),
                5,
            ),
            (
                "empty_k1_then_valid_k2_on_next",
                [Slot::DiffA, Slot::DiffA, Slot::DiffA, Slot::DiffA],
                [Slot::Empty, Slot::DiffB, Slot::DiffB, Slot::DiffB],
                (Some(1), Some(2)),
                (Some(2.0), Some(2.0)),
                3,
            ),
            (
                "all_duplicate",
                [Slot::Dup, Slot::Dup, Slot::Dup, Slot::Dup],
                [Slot::Dup, Slot::Dup, Slot::Dup, Slot::Dup],
                (None, None),
                (None, None),
                8,
            ),
        ];

        let mut cur_rgba = Vec::new();
        let (cur_luma, cur_max_a) =
            pixels::read_rgba_and_luma(&cur_raw, &desc, &mut cur_rgba).unwrap();
        let cur_has_alpha = cur_max_a > 0.0;
        assert!(cur_has_alpha);

        for (name, prev_slots, next_slots, exp_k, exp_fd, exp_reads) in scenarios {
            let cur_l = core::Luma {
                width: w,
                height: h,
                data: &cur_luma,
            };
            let is_dup = |l: &Vec<f32>| {
                core::is_duplicate_frame(
                    cur_l,
                    core::Luma {
                        width: w,
                        height: h,
                        data: l,
                    },
                )
            };
            let cpu_sel = effect::select_legacy_neighbors(
                MAX_K,
                |dir, k| {
                    let s = if dir < 0 {
                        prev_slots[(k - 1) as usize]
                    } else {
                        next_slots[(k - 1) as usize]
                    };
                    let bytes = bytes_of(s).ok_or(Outcome::Missing)?;
                    let (luma, max_a) =
                        pixels::read_luma(bytes, &desc).map_err(|_| Outcome::Unusable)?;
                    if cur_has_alpha && max_a <= 0.0 {
                        return Err(Outcome::Unusable);
                    }
                    Ok(luma)
                },
                &is_dup,
            );
            let cpu_probes: Vec<(&'static str, u32, &'static str)> = cpu_sel
                .probes
                .iter()
                .map(|&(dir, k, o)| (if dir < 0 { "lprev" } else { "lnext" }, k, o.name()))
                .collect();
            assert_eq!(
                (cpu_sel.prev.as_ref().map(|p| p.0), cpu_sel.next.as_ref().map(|n| n.0)),
                exp_k,
                "cpu {name}"
            );
            assert_eq!(cpu_sel.fd, exp_fd, "cpu fd {name}");

            let mut deps = Vec::with_capacity(8);
            for i in 0..4 {
                deps.push(gpu_slot(prev_slots[i]));
                deps.push(gpu_slot(next_slots[i]));
            }
            let budget = Cell::new(crate::metal_io::METAL_RENDER_WAIT_BUDGET);
            let rep = unsafe {
                render_metal_frames(
                    &ctx,
                    &core::Params::default(),
                    &metal_desc(&cur_buf),
                    &deps,
                    &metal_desc(&out_buf),
                    GPU_DEPENDENCY_STEPS,
                    &budget,
                )
            }
            .unwrap();

            assert_eq!((rep.prev_k, rep.next_k), exp_k, "gpu {name}");
            assert_eq!(rep.fd, cpu_sel.fd, "gpu vs cpu fd {name}");
            assert_eq!(rep.probes, cpu_probes, "gpu vs cpu probe trace {name}");
            assert_eq!(rep.deps_read, exp_reads, "gpu deps_read {name}");
        }
    }

    #[test]
    fn gpu_dependency_protocol_interleaved_and_terminates_cleanly() {
        let tpf = 10_594_584_000i64;
        let seq = 100 * tpf;
        let p = core::Params::default();
        assert_eq!(gpu_dependency_count(&p, seq, tpf), 2 * GPU_DEPENDENCY_STEPS);

        let mut q = 0i32;
        let mut collected = Vec::new();
        loop {
            let step = gpu_dependency_at(&p, seq, tpf, q).unwrap();
            collected.push((q, step));
            if step.next_query_index == q {
                break;
            }
            assert_eq!(step.next_query_index, q + 1);
            q = step.next_query_index;
            assert!(collected.len() <= 32);
        }
        assert_eq!(collected.len(), (2 * GPU_DEPENDENCY_STEPS) as usize);
        for k in 1..=GPU_DEPENDENCY_STEPS {
            let pi = dep_slice_index(-1, k).unwrap();
            let ni = dep_slice_index(1, k).unwrap();
            assert_eq!(collected[pi].1.sequence_time, seq - (k as i64) * tpf);
            assert_eq!(collected[ni].1.sequence_time, seq + (k as i64) * tpf);
            assert_eq!(collected[pi].1.track_id, 0);
            assert_eq!(collected[ni].1.track_id, 0);
        }
        assert_eq!(gpu_dependency_at(&p, seq, tpf, -1), None);
        assert_eq!(
            gpu_dependency_at(&p, seq, tpf, (2 * GPU_DEPENDENCY_STEPS) as i32),
            None
        );

        let zero_blur = core::Params {
            blur_amount: 0.0,
            ..core::Params::default()
        };
        assert_eq!(gpu_dependency_count(&zero_blur, seq, tpf), 0);
        assert_eq!(gpu_dependency_at(&zero_blur, seq, tpf, 0), None);
        assert_eq!(gpu_dependency_at(&p, seq, 0, 0), None);
    }

    #[test]
    fn gpu_param_mapping_matches_pf_semantics() {
        let mk_f64 = |v: f64| -> sys::PrParam {
            let mut p: sys::PrParam = unsafe { std::mem::zeroed() };
            p.mType = sys::PrParamType_kPrParamType_Float64;
            p.__bindgen_anon_1.mFloat64 = v;
            p
        };
        let mk_i32 = |v: i32| -> sys::PrParam {
            let mut p: sys::PrParam = unsafe { std::mem::zeroed() };
            p.mType = sys::PrParamType_kPrParamType_Int32;
            p.__bindgen_anon_1.mInt32 = v;
            p
        };
        let mk_bool = |v: bool| -> sys::PrParam {
            let mut p: sys::PrParam = unsafe { std::mem::zeroed() };
            p.mType = sys::PrParamType_kPrParamType_Bool;
            p.__bindgen_anon_1.mBool = u8::from(v);
            p
        };

        let p = map_gpu_params(&[
            mk_f64(0.5),
            mk_f64(70.0),
            mk_f64(15.0),
            mk_i32(1),
            mk_bool(true),
            mk_i32(0),
        ])
        .unwrap();
        assert_eq!(p.blur_amount, 0.5);
        assert_eq!(p.motion_sensitivity, 70.0);
        assert_eq!(p.max_blur_percent, 15.0);
        assert_eq!(p.quality, core::Quality::Normal);
        assert!(p.linear_light);
        assert_eq!(p.view, core::View::Output);

        let p2 = map_gpu_params(&[
            mk_f64(9.0),
            mk_f64(-5.0),
            mk_f64(80.0),
            mk_i32(0),
            mk_bool(false),
            mk_i32(1),
        ])
        .unwrap();
        assert_eq!(p2.blur_amount, 4.0);
        assert_eq!(p2.motion_sensitivity, 0.0);
        assert_eq!(p2.max_blur_percent, 50.0);
        assert_eq!(p2.quality, core::Quality::Draft);
        assert!(!p2.linear_light);
        assert_eq!(p2.view, core::View::MotionVectors);

        let p3 = map_gpu_params(&[
            mk_f64(1.0),
            mk_f64(50.0),
            mk_f64(10.0),
            mk_i32(2),
            mk_bool(true),
            mk_i32(0),
        ])
        .unwrap();
        assert_eq!(p3.quality, core::Quality::High);
    }

    #[test]
    fn adjustment_layer_node_classification() {
        let adj_flag = sys::PrNodeInfoFlag_kPrNodeInfoFlag_IsAdjustment;
        assert_eq!(
            classify_adjustment_node(
                "RenderableNode_AdjustmentEffectImpl",
                0,
                None,
                None,
                None,
                None
            ),
            (true, "node_type:AdjustmentEffect")
        );
        assert_eq!(
            classify_adjustment_node(
                "RenderableNodeEffectImpl",
                0,
                Some("RenderableNode_AdjustmentImpl"),
                Some(0),
                None,
                None,
            ),
            (true, "owner_type:Adjustment")
        );
        assert_eq!(
            classify_adjustment_node(
                "RenderableNodeEffectImpl",
                0,
                Some("RenderableNodeClipImpl"),
                Some(0),
                Some("RenderableNode_AdjustmentImpl"),
                Some(0),
            ),
            (true, "owner_input0_type:Adjustment")
        );
        assert_eq!(
            classify_adjustment_node(
                "RenderableNodeEffectImpl",
                adj_flag,
                Some("RenderableNodeClipImpl"),
                Some(0),
                Some("RenderableNodeMediaImpl"),
                Some(0),
            ),
            (true, "node_flag:IsAdjustment")
        );
        assert_eq!(
            classify_adjustment_node(
                "RenderableNodeEffectImpl",
                0,
                Some("RenderableNodeClipImpl"),
                Some(adj_flag),
                Some("RenderableNodeMediaImpl"),
                Some(0),
            ),
            (true, "owner_flag:IsAdjustment")
        );
        assert_eq!(
            classify_adjustment_node(
                "RenderableNodeEffectImpl",
                0,
                Some("RenderableNodeClipImpl"),
                Some(0),
                Some("RenderableNodeMediaImpl"),
                Some(adj_flag),
            ),
            (true, "owner_input0_flag:IsAdjustment")
        );
        assert_eq!(
            classify_adjustment_node(
                "RenderableNodeEffectImpl",
                0,
                Some("RenderableNodeClipImpl"),
                Some(0),
                Some("RenderableNodeMediaImpl"),
                Some(0),
            ),
            (false, "none")
        );
        assert_eq!(
            classify_adjustment_node("RenderableNodeEffectImpl", 0, None, None, None, None),
            (false, "none")
        );
    }

    #[test]
    fn debug_gpu_steps_override_parsing_uses_injectable_path() {
        // Item 12 regression: never touch /tmp/tblur-*; use an isolated temp file.
        let path = std::env::temp_dir().join(format!(
            "tblur-unit-gpu-steps-{}-{:?}.txt",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_file(&path);

        // Missing file -> default.
        assert_eq!(diag::resolve_gpu_steps_from(true, &path, 4), (4, false));

        // Valid single digit 1..=4, with optional whitespace/newline.
        for (txt, want) in [("1", 1), ("2\n", 2), (" 3 \r\n", 3), ("4", 4)] {
            std::fs::write(&path, txt).unwrap();
            assert_eq!(diag::resolve_gpu_steps_from(true, &path, 4), (want, true));
            // Without debug flag, file MUST be ignored.
            assert_eq!(diag::resolve_gpu_steps_from(false, &path, 4), (4, false));
        }

        // Invalid contents -> default.
        for bad in ["", "0", "5", "12", "-1", "abc", "2 frames"] {
            std::fs::write(&path, bad).unwrap();
            assert_eq!(diag::resolve_gpu_steps_from(true, &path, 4), (4, false), "bad={bad:?}");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn debug_flag_safe_reader_and_fault_parsing() {
        // Items 3 & 4: regular files only, no symlink following, no FIFO blocking, <= 16 bytes,
        // and fault mode parsing through an injectable path (never touches /tmp/tblur-*).
        let base = std::env::temp_dir().join(format!(
            "tblur-safe-flag-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();

        let steps_path = base.join("steps");
        let fault_path = base.join("fault");
        let link_steps = base.join("steps.link");
        let link_fault = base.join("fault.link");
        let fifo_path = base.join("pipe");
        let huge_path = base.join("huge");
        let dir_path = base.join("subdir");
        std::fs::create_dir_all(&dir_path).unwrap();

        // 1. Valid fault modes & debug-off gate.
        assert_eq!(diag::resolve_gpu_fault_from(true, &fault_path), diag::GpuFaultMode::None);
        for (txt, want) in [
            ("error", diag::GpuFaultMode::Error),
            ("error\n", diag::GpuFaultMode::Error),
            (" passthrough \r\n", diag::GpuFaultMode::Passthrough),
            ("timeout\n", diag::GpuFaultMode::Timeout),
        ] {
            std::fs::write(&fault_path, txt).unwrap();
            assert_eq!(diag::resolve_gpu_fault_from(true, &fault_path), want, "txt={txt:?}");
            assert_eq!(
                diag::resolve_gpu_fault_from(false, &fault_path),
                diag::GpuFaultMode::None,
                "debug=false must ignore fault file"
            );
        }
        for bad in ["", "none", "crash", "error now", "TIMEOUT", "passthrough_extra"] {
            std::fs::write(&fault_path, bad).unwrap();
            assert_eq!(
                diag::resolve_gpu_fault_from(true, &fault_path),
                diag::GpuFaultMode::None,
                "bad={bad:?}"
            );
        }

        // 2. Symlinks to valid regular files MUST be rejected.
        std::fs::write(&steps_path, "2\n").unwrap();
        std::fs::write(&fault_path, "error\n").unwrap();
        std::os::unix::fs::symlink(&steps_path, &link_steps).unwrap();
        std::os::unix::fs::symlink(&fault_path, &link_fault).unwrap();
        assert_eq!(diag::resolve_gpu_steps_from(true, &link_steps, 4), (4, false));
        assert_eq!(diag::resolve_gpu_fault_from(true, &link_fault), diag::GpuFaultMode::None);

        // 3. Directories MUST be rejected.
        assert_eq!(diag::resolve_gpu_steps_from(true, &dir_path, 4), (4, false));
        assert_eq!(diag::resolve_gpu_fault_from(true, &dir_path), diag::GpuFaultMode::None);

        // 4. Huge regular files (> 16 bytes, e.g. 1 MiB) MUST be rejected immediately without
        //    reading the whole file.
        let mut huge_data = b"2\n".to_vec();
        huge_data.resize(1024 * 1024, b' ');
        std::fs::write(&huge_path, &huge_data).unwrap();
        let t_huge = Instant::now();
        assert_eq!(diag::resolve_gpu_steps_from(true, &huge_path, 4), (4, false));
        assert_eq!(diag::resolve_gpu_fault_from(true, &huge_path), diag::GpuFaultMode::None);
        assert!(t_huge.elapsed() < Duration::from_millis(200));

        // 5. Named pipe (FIFO) with no writer MUST NOT block.
        let status = std::process::Command::new("mkfifo")
            .arg(&fifo_path)
            .status()
            .expect("mkfifo");
        assert!(status.success());
        let t_fifo = Instant::now();
        assert_eq!(diag::resolve_gpu_steps_from(true, &fifo_path, 4), (4, false));
        assert_eq!(diag::resolve_gpu_fault_from(true, &fifo_path), diag::GpuFaultMode::None);
        assert!(
            t_fifo.elapsed() < Duration::from_millis(200),
            "FIFO flag read took too long: {:?}",
            t_fifo.elapsed()
        );

        let _ = std::fs::remove_dir_all(&base);
    }

    // ───────────────────────────── self-contained mock Premiere host ─────────────────────────────
    // Exercises the real exported `xGPUFilterEntry` callbacks for Items 1, 2, 4, 5, 7, 8, 9, 10, 11.

    struct MockPPix {
        fmt: sys::PrPixelFormat,
        bounds: [i32; 4],
        row_bytes: i32,
        data_ptr: *mut c_void,
        _buf: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
        _non_buf_obj: Option<Retained<NSString>>,
        created_by_plugin: bool,
        disposed: bool,
    }
    unsafe impl Send for MockPPix {}

    struct MockHostState {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
        ppix: HashMap<usize, MockPPix>,
        next_handle: usize,
        fail_param: Option<i32>,
        blur_amount: f64,
        motion_sensitivity: f64,
        fail_create: bool,
        create_private: bool,
        create_size_delta: i32,
        created_count: usize,
        disposed_by_plugin: usize,
        big_create_count: usize,
        runtime_id_prop: Option<String>,
        unterminated_prop_bytes: Option<Vec<u8>>,
        fail_get_ptr_size: bool,
        owner_fail: bool,
        segs_acquired: usize,
        segs_released: usize,
        nodes_acquired: usize,
        nodes_released: usize,
        mem_allocs: usize,
        mem_frees: usize,
        mem_sizes: HashMap<usize, usize>,
    }
    unsafe impl Send for MockHostState {}

    // Serialize mock-host tests so they share one deterministic mock state cleanly.
    static MOCK_TEST_LOCK: Mutex<()> = Mutex::new(());
    static MOCK_HOST: OnceLock<Mutex<MockHostState>> = OnceLock::new();

    fn mock_state() -> &'static Mutex<MockHostState> {
        MOCK_HOST.get_or_init(|| {
            let device = MTLCreateSystemDefaultDevice().expect("Metal device");
            let queue = device.newCommandQueue().expect("command queue");
            Mutex::new(MockHostState {
                device,
                queue,
                ppix: HashMap::new(),
                next_handle: 0x1000,
                fail_param: None,
                blur_amount: 0.5,
                motion_sensitivity: 70.0,
                fail_create: false,
                create_private: false,
                create_size_delta: 0,
                created_count: 0,
                disposed_by_plugin: 0,
                big_create_count: 0,
                runtime_id_prop: Some("60210".to_string()),
                unterminated_prop_bytes: None,
                fail_get_ptr_size: false,
                owner_fail: true,
                segs_acquired: 0,
                segs_released: 0,
                nodes_acquired: 0,
                nodes_released: 0,
                mem_allocs: 0,
                mem_frees: 0,
                mem_sizes: HashMap::new(),
            })
        })
    }

    fn reset_mock_host() {
        let mut st = mock_state().lock().unwrap_or_else(|e| e.into_inner());
        st.ppix.clear();
        st.fail_param = None;
        st.blur_amount = 0.5;
        st.motion_sensitivity = 70.0;
        st.fail_create = false;
        st.create_private = false;
        st.create_size_delta = 0;
        st.created_count = 0;
        st.disposed_by_plugin = 0;
        st.big_create_count = 0;
        st.runtime_id_prop = Some("60210".to_string());
        st.unterminated_prop_bytes = None;
        st.fail_get_ptr_size = false;
        st.owner_fail = true;
        st.segs_acquired = 0;
        st.segs_released = 0;
        st.nodes_acquired = 0;
        st.nodes_released = 0;
        st.mem_allocs = 0;
        st.mem_frees = 0;
        st.mem_sizes.clear();
    }

    fn register_mock_ppix(
        fmt: sys::PrPixelFormat,
        bounds: [i32; 4],
        row_bytes: i32,
        buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> sys::PPixHand {
        let mut st = mock_state().lock().unwrap();
        let h = st.next_handle;
        st.next_handle += 16;
        let ptr = Retained::as_ptr(&buf) as *mut c_void;
        st.ppix.insert(
            h,
            MockPPix {
                fmt,
                bounds,
                row_bytes,
                data_ptr: ptr,
                _buf: Some(buf),
                _non_buf_obj: None,
                created_by_plugin: false,
                disposed: false,
            },
        );
        h as sys::PPixHand
    }

    fn register_mock_non_buffer_ppix(w: i32, h: i32, rb: i32) -> sys::PPixHand {
        let mut st = mock_state().lock().unwrap();
        let handle = st.next_handle;
        st.next_handle += 16;
        let ns = NSString::from_str("hostile_non_mtlbuffer");
        let ptr = Retained::as_ptr(&ns) as *mut c_void;
        st.ppix.insert(
            handle,
            MockPPix {
                fmt: sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                bounds: [0, 0, w, h],
                row_bytes: rb,
                data_ptr: ptr,
                _buf: None,
                _non_buf_obj: Some(ns),
                created_by_plugin: false,
                disposed: false,
            },
        );
        handle as sys::PPixHand
    }

    unsafe extern "C" fn mock_get_device_info(
        _ver: sys::csSDK_uint32,
        _idx: sys::csSDK_uint32,
        out: *mut sys::PrGPUDeviceInfo,
    ) -> sys::prSuiteError {
        let st = mock_state().lock().unwrap();
        unsafe {
            (*out).outDeviceFramework = sys::PrGPUDeviceFramework_PrGPUDeviceFramework_Metal;
            (*out).outMeetsMinimumRequirementsForAcceleration = 1;
            (*out).outDeviceHandle = Retained::as_ptr(&st.device) as *mut c_void;
            (*out).outCommandQueueHandle = Retained::as_ptr(&st.queue) as *mut c_void;
        }
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_create_gpu_ppix(
        _dev: sys::csSDK_uint32,
        fmt: sys::PrPixelFormat,
        w: i32,
        h: i32,
        _pn: i32,
        _pd: i32,
        _ft: sys::prFieldType,
        out_h: *mut sys::PPixHand,
    ) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        let bpp = if fmt == sys::PrPixelFormatGpu_GPU_BGRA_4444_16f { 8 } else { 16 };
        let req_bytes = (w.max(0) as u64) * (h.max(0) as u64) * (bpp as u64);
        if req_bytes > (2u64 << 30) {
            st.big_create_count += 1;
            return sys::suiteError_Fail;
        }
        if st.fail_create {
            return sys::suiteError_Fail;
        }
        let aw = (w + st.create_size_delta).max(1);
        let ah = h.max(1);
        let rb = aw * bpp;
        let opts = if st.create_private {
            MTLResourceOptions::StorageModePrivate
        } else {
            MTLResourceOptions::StorageModeShared
        };
        let Some(buf) = st.device.newBufferWithLength_options((ah * rb) as usize, opts) else {
            return sys::suiteError_Fail;
        };
        let handle = st.next_handle;
        st.next_handle += 16;
        let ptr = Retained::as_ptr(&buf) as *mut c_void;
        st.ppix.insert(
            handle,
            MockPPix {
                fmt,
                bounds: [0, 0, aw, ah],
                row_bytes: rb,
                data_ptr: ptr,
                _buf: Some(buf),
                _non_buf_obj: None,
                created_by_plugin: true,
                disposed: false,
            },
        );
        st.created_count += 1;
        unsafe {
            *out_h = handle as sys::PPixHand;
        }
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_get_gpu_ppix_data(
        h: sys::PPixHand,
        out: *mut *mut c_void,
    ) -> sys::prSuiteError {
        let st = mock_state().lock().unwrap();
        if let Some(p) = st.ppix.get(&(h as usize))
            && !p.disposed
        {
            unsafe { *out = p.data_ptr };
            return sys::suiteError_NoError;
        }
        sys::suiteError_Fail
    }

    unsafe extern "C" fn mock_ppix_dispose(h: sys::PPixHand) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        if let Some(p) = st.ppix.get_mut(&(h as usize))
            && p.created_by_plugin
            && !p.disposed
        {
            p.disposed = true;
            st.disposed_by_plugin += 1;
            return sys::suiteError_NoError;
        }
        sys::suiteError_Fail
    }

    unsafe extern "C" fn mock_ppix_get_bounds(
        h: sys::PPixHand,
        out: *mut sys::prRect,
    ) -> sys::prSuiteError {
        let st = mock_state().lock().unwrap();
        if let Some(p) = st.ppix.get(&(h as usize))
            && !p.disposed
        {
            unsafe {
                (*out).left = p.bounds[0] as i16;
                (*out).top = p.bounds[1] as i16;
                (*out).right = p.bounds[2] as i16;
                (*out).bottom = p.bounds[3] as i16;
            }
            return sys::suiteError_NoError;
        }
        sys::suiteError_Fail
    }

    unsafe extern "C" fn mock_ppix_get_row_bytes(
        h: sys::PPixHand,
        out: *mut i32,
    ) -> sys::prSuiteError {
        let st = mock_state().lock().unwrap();
        if let Some(p) = st.ppix.get(&(h as usize))
            && !p.disposed
        {
            unsafe { *out = p.row_bytes };
            return sys::suiteError_NoError;
        }
        sys::suiteError_Fail
    }

    unsafe extern "C" fn mock_ppix_get_par(
        _h: sys::PPixHand,
        n: *mut u32,
        d: *mut u32,
    ) -> sys::prSuiteError {
        unsafe {
            *n = 1;
            *d = 1;
        }
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_ppix_get_fmt(
        h: sys::PPixHand,
        out: *mut sys::PrPixelFormat,
    ) -> sys::prSuiteError {
        let st = mock_state().lock().unwrap();
        if let Some(p) = st.ppix.get(&(h as usize))
            && !p.disposed
        {
            unsafe { *out = p.fmt };
            return sys::suiteError_NoError;
        }
        sys::suiteError_Fail
    }

    unsafe extern "C" fn mock_vs_acq_segs(
        _tl: sys::PrTimelineID,
        out_id: *mut i32,
    ) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        st.segs_acquired += 1;
        unsafe { *out_id = 500 };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_rel_segs(_segs_id: i32) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        st.segs_released += 1;
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_get_seg_count(
        _segs_id: i32,
        out_cnt: *mut i32,
    ) -> sys::prSuiteError {
        unsafe { *out_cnt = 2 };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_get_seg_info(
        _segs_id: i32,
        idx: i32,
        out_start: *mut sys::PrTime,
        out_end: *mut sys::PrTime,
        out_off: *mut sys::PrTime,
        out_hash: *mut sys::prPluginID,
    ) -> sys::prSuiteError {
        unsafe {
            *out_start = (idx as i64) * 1000;
            *out_end = ((idx + 1) as i64) * 1000;
            *out_off = 0;
            *out_hash = std::mem::zeroed();
            (*out_hash).mGUID[0] = if idx == 0 { 1 } else { 2 };
        }
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_acq_node_id(
        _segs_id: i32,
        in_hash: *mut sys::prPluginID,
        out_node_id: *mut i32,
    ) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        st.nodes_acquired += 1;
        let tag = unsafe { (*in_hash).mGUID[0] };
        unsafe { *out_node_id = if tag == 1 { 300 } else { 400 } };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_get_node_info(
        node_id: i32,
        out_type: *mut c_char,
        out_hash: *mut sys::prPluginID,
        out_flags: *mut i32,
    ) -> sys::prSuiteError {
        let (name, flags, hash_tag): (&[u8], i32, i8) = match node_id {
            100 | 402 => (b"RenderableNodeEffectImpl\0", 0x4c, 42),
            200 | 301 => (b"RenderableNodeClipImpl\0", 0x54, 11),
            300 => (b"RenderableNodeMultitrackImpl\0", 0x0, 10),
            302 => (b"RenderableNodeMediaImpl\0", 0x55, 12),
            400 => (b"RenderableNodeMultitrackImpl\0", 0x0, 20),
            401 => (b"RenderableNodeClipImpl\0", 0x54, 21),
            _ => (b"RenderableNodeUnknownImpl\0", 0, 0),
        };
        unsafe {
            std::ptr::copy_nonoverlapping(name.as_ptr() as *const c_char, out_type, name.len());
            *out_flags = flags;
            if !out_hash.is_null() {
                *out_hash = std::mem::zeroed();
                (*out_hash).mGUID[0] = hash_tag;
            }
        }
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_acq_owner(
        _node_id: i32,
        out_owner: *mut i32,
    ) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        if st.owner_fail {
            return -2147483643;
        }
        st.nodes_acquired += 1;
        unsafe { *out_owner = 200 };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_rel_node(_node_id: i32) -> sys::prSuiteError {
        let mut st = mock_state().lock().unwrap();
        st.nodes_released += 1;
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_get_input_count(
        node_id: i32,
        out_cnt: *mut i32,
    ) -> sys::prSuiteError {
        let c = match node_id {
            200 | 300 | 301 | 400 => 1,
            _ => 0,
        };
        unsafe { *out_cnt = c };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_acq_input(
        node_id: i32,
        idx: i32,
        out_off: *mut sys::PrTime,
        out_child: *mut i32,
    ) -> sys::prSuiteError {
        let child = match (node_id, idx) {
            (200, 0) => 302,
            (300, 0) => 301,
            (301, 0) => 302,
            (400, 0) => 401,
            _ => return sys::suiteError_Fail,
        };
        let mut st = mock_state().lock().unwrap();
        st.nodes_acquired += 1;
        unsafe {
            *out_off = 0;
            *out_child = child;
        }
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_get_op_count(
        node_id: i32,
        out_cnt: *mut i32,
    ) -> sys::prSuiteError {
        let c = match node_id {
            400 => 1,
            _ => 0,
        };
        unsafe { *out_cnt = c };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_vs_acq_op(
        node_id: i32,
        idx: i32,
        out_child: *mut i32,
    ) -> sys::prSuiteError {
        if node_id == 400 && idx == 0 {
            let mut st = mock_state().lock().unwrap();
            st.nodes_acquired += 1;
            unsafe { *out_child = 402 };
            return sys::suiteError_NoError;
        }
        sys::suiteError_Fail
    }

    unsafe extern "C" fn mock_vs_get_node_property(
        node_id: i32,
        key: *const c_char,
        out_val: *mut sys::PrMemoryPtr,
    ) -> sys::prSuiteError {
        let k = unsafe { std::ffi::CStr::from_ptr(key).to_bytes_with_nul() };
        if k != sys::kVideoSegmentProperty_Effect_RuntimeInstanceID {
            return sys::suiteError_Fail;
        }
        let mut st = mock_state().lock().unwrap();
        if node_id != 100 && node_id != 402 {
            return sys::suiteError_Fail;
        }
        let raw_bytes: Vec<u8> = if let Some(ref b) = st.unterminated_prop_bytes {
            b.clone()
        } else if let Some(ref val) = st.runtime_id_prop {
            let mut v = val.as_bytes().to_vec();
            v.push(0);
            v
        } else {
            return sys::suiteError_Fail;
        };
        let n = raw_bytes.len().max(1);
        let layout = std::alloc::Layout::from_size_align(n, 1).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        assert!(!ptr.is_null());
        unsafe {
            std::ptr::copy_nonoverlapping(raw_bytes.as_ptr(), ptr, raw_bytes.len());
        }
        st.mem_allocs += 1;
        st.mem_sizes.insert(ptr as usize, n);
        unsafe { *out_val = ptr as sys::PrMemoryPtr };
        sys::suiteError_NoError
    }

    unsafe extern "C" fn mock_mem_get_ptr_size(p: sys::PrMemoryPtr) -> sys::csSDK_uint32 {
        if p.is_null() {
            return 0;
        }
        let st = mock_state().lock().unwrap();
        if st.fail_get_ptr_size {
            return 0;
        }
        st.mem_sizes.get(&(p as usize)).copied().unwrap_or(0) as sys::csSDK_uint32
    }

    unsafe extern "C" fn mock_plug_get_ptr_size(p: sys::PrMemoryPtr) -> i32 {
        unsafe { mock_mem_get_ptr_size(p) as i32 }
    }

    unsafe extern "C" fn mock_mem_dispose_ptr(p: sys::PrMemoryPtr) {
        if !p.is_null() {
            let mut st = mock_state().lock().unwrap();
            st.mem_frees += 1;
            if let Some(n) = st.mem_sizes.remove(&(p as usize)) {
                let layout = std::alloc::Layout::from_size_align(n, 1).unwrap();
                unsafe { std::alloc::dealloc(p as *mut u8, layout) };
            }
        }
    }

    unsafe extern "C" fn mock_vs_get_param(
        _node_id: i32,
        idx: i32,
        _time: sys::PrTime,
        out: *mut sys::PrParam,
    ) -> sys::prSuiteError {
        let st = mock_state().lock().unwrap();
        if st.fail_param == Some(idx) {
            return sys::suiteError_Fail;
        }
        unsafe {
            *out = std::mem::zeroed();
            match idx {
                0 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Float64;
                    (*out).__bindgen_anon_1.mFloat64 = st.blur_amount;
                }
                1 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Float64;
                    (*out).__bindgen_anon_1.mFloat64 = st.motion_sensitivity;
                }
                2 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Float64;
                    (*out).__bindgen_anon_1.mFloat64 = 15.0;
                }
                3 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Int32;
                    (*out).__bindgen_anon_1.mInt32 = 1;
                }
                4 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Bool;
                    (*out).__bindgen_anon_1.mBool = 1;
                }
                5 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Int32;
                    (*out).__bindgen_anon_1.mInt32 = 0;
                }
                6 => {
                    (*out).mType = sys::PrParamType_kPrParamType_Bool;
                    (*out).__bindgen_anon_1.mBool = 1;
                }
                _ => return sys::suiteError_InvalidParms,
            }
        }
        sys::suiteError_NoError
    }

    struct MockSuites {
        sp: Box<sys::SPBasicSuite>,
        _util: Box<sys::PlugUtilFuncs>,
        _mem_funcs: Box<sys::PlugMemoryFuncs>,
        pi: Box<sys::piSuites>,
        gpu: Box<sys::PrSDKGPUDeviceSuite>,
        ppix: Box<sys::PrSDKPPixSuite>,
        vs: Box<sys::PrSDKVideoSegmentSuite>,
        mem_suite: Box<sys::PrSDKMemoryManagerSuite>,
    }
    unsafe impl Send for MockSuites {}
    unsafe impl Sync for MockSuites {}

    static MOCK_SUITES: OnceLock<MockSuites> = OnceLock::new();

    unsafe extern "C" fn mock_get_sp_basic() -> *mut sys::SPBasicSuite {
        let s = MOCK_SUITES.get().unwrap();
        (&*s.sp as *const sys::SPBasicSuite) as *mut sys::SPBasicSuite
    }

    unsafe extern "C" fn mock_acquire_suite(
        name: *const c_char,
        _ver: i32,
        out: *mut *const c_void,
    ) -> sys::SPErr {
        let s = MOCK_SUITES.get().unwrap();
        let cname = unsafe { std::ffi::CStr::from_ptr(name).to_bytes_with_nul() };
        let ptr: *const c_void = if cname == sys::kPrSDKGPUDeviceSuite {
            &*s.gpu as *const _ as *const c_void
        } else if cname == sys::kPrSDKPPixSuite {
            &*s.ppix as *const _ as *const c_void
        } else if cname == sys::kPrSDKVideoSegmentSuite {
            &*s.vs as *const _ as *const c_void
        } else if cname == sys::kPrSDKMemoryManagerSuite {
            &*s.mem_suite as *const _ as *const c_void
        } else {
            return -1;
        };
        unsafe { *out = ptr };
        0
    }

    unsafe extern "C" fn mock_release_suite(_name: *const c_char, _ver: i32) -> sys::SPErr {
        0
    }

    fn mock_pi_suites() -> sys::piSuitesPtr {
        let s = MOCK_SUITES.get_or_init(|| {
            let mut sp: Box<sys::SPBasicSuite> = Box::new(unsafe { std::mem::zeroed() });
            sp.AcquireSuite = Some(mock_acquire_suite);
            sp.ReleaseSuite = Some(mock_release_suite);
            let mut util: Box<sys::PlugUtilFuncs> = Box::new(unsafe { std::mem::zeroed() });
            util.getSPBasicSuite = Some(mock_get_sp_basic);
            let mut mem_funcs: Box<sys::PlugMemoryFuncs> = Box::new(unsafe { std::mem::zeroed() });
            mem_funcs.getPtrSize = Some(mock_plug_get_ptr_size);
            mem_funcs.disposePtr = Some(mock_mem_dispose_ptr);
            let mut pi: Box<sys::piSuites> = Box::new(unsafe { std::mem::zeroed() });
            pi.utilFuncs = (&*util as *const sys::PlugUtilFuncs) as *mut sys::PlugUtilFuncs;
            pi.memFuncs = (&*mem_funcs as *const sys::PlugMemoryFuncs) as *mut sys::PlugMemoryFuncs;
            let mut gpu: Box<sys::PrSDKGPUDeviceSuite> = Box::new(unsafe { std::mem::zeroed() });
            gpu.GetDeviceInfo = Some(mock_get_device_info);
            gpu.CreateGPUPPix = Some(mock_create_gpu_ppix);
            gpu.GetGPUPPixData = Some(mock_get_gpu_ppix_data);
            let mut ppix: Box<sys::PrSDKPPixSuite> = Box::new(unsafe { std::mem::zeroed() });
            ppix.Dispose = Some(mock_ppix_dispose);
            ppix.GetBounds = Some(mock_ppix_get_bounds);
            ppix.GetRowBytes = Some(mock_ppix_get_row_bytes);
            ppix.GetPixelAspectRatio = Some(mock_ppix_get_par);
            ppix.GetPixelFormat = Some(mock_ppix_get_fmt);
            let mut vs: Box<sys::PrSDKVideoSegmentSuite> =
                Box::new(unsafe { std::mem::zeroed() });
            vs.AcquireVideoSegmentsID = Some(mock_vs_acq_segs);
            vs.ReleaseVideoSegmentsID = Some(mock_vs_rel_segs);
            vs.GetSegmentCount = Some(mock_vs_get_seg_count);
            vs.GetSegmentInfo = Some(mock_vs_get_seg_info);
            vs.AcquireNodeID = Some(mock_vs_acq_node_id);
            vs.GetNodeInfo = Some(mock_vs_get_node_info);
            vs.AcquireOperatorOwnerNodeID = Some(mock_vs_acq_owner);
            vs.ReleaseVideoNodeID = Some(mock_vs_rel_node);
            vs.GetNodeInputCount = Some(mock_vs_get_input_count);
            vs.AcquireInputNodeID = Some(mock_vs_acq_input);
            vs.GetNodeOperatorCount = Some(mock_vs_get_op_count);
            vs.AcquireOperatorNodeID = Some(mock_vs_acq_op);
            vs.GetNodeProperty = Some(mock_vs_get_node_property);
            vs.GetParam = Some(mock_vs_get_param);
            let mut mem_suite: Box<sys::PrSDKMemoryManagerSuite> =
                Box::new(unsafe { std::mem::zeroed() });
            mem_suite.GetPtrSize = Some(mock_mem_get_ptr_size);
            mem_suite.PrDisposePtr = Some(mock_mem_dispose_ptr);
            MockSuites {
                sp,
                _util: util,
                _mem_funcs: mem_funcs,
                pi,
                gpu,
                ppix,
                vs,
                mem_suite,
            }
        });
        (&*s.pi as *const sys::piSuites) as sys::piSuitesPtr
    }

    struct MockRig {
        filter: sys::PrGPUFilter,
        inst: Box<sys::PrGPUFilterInstance>,
    }

    impl MockRig {
        fn try_new_raw(node_id: i32) -> (i32, Self) {
            let mut filter: sys::PrGPUFilter = unsafe { std::mem::zeroed() };
            let mut info: sys::PrGPUFilterInfo = unsafe { std::mem::zeroed() };
            let mut idx = 0i32;
            let e = unsafe {
                xGPUFilterEntry(
                    sys::PrSDKGPUFilterInterfaceVersion,
                    &mut idx,
                    1,
                    mock_pi_suites(),
                    &mut filter,
                    &mut info,
                )
            };
            assert_eq!(e, 0);
            let mut inst: Box<sys::PrGPUFilterInstance> =
                Box::new(unsafe { std::mem::zeroed() });
            inst.piSuites = mock_pi_suites();
            inst.inDeviceIndex = 0;
            inst.inTimelineID = 1;
            inst.inNodeID = node_id;
            let e = unsafe { (filter.CreateInstance.unwrap())(&mut *inst) };
            (e, MockRig { filter, inst })
        }

        fn new() -> Self {
            // Seed the PF synthetic map for mock instance 60210, matching PF-recorded adjustment layers.
            instance_registry::global().record(60210, true, "sequence_setup");
            let (e, rig) = Self::try_new_raw(100);
            assert_eq!(e, 0);
            rig
        }

        fn set_wait_limits(&mut self, per_wait: Duration, render_budget: Duration) {
            unsafe {
                let raw = std::ptr::read_unaligned(std::ptr::addr_of!(
                    (*self.inst).ioPrivatePluginData
                ));
                let inner = &mut *(raw as *mut GpuFilterInstance);
                inner.metal.set_wait_limits(per_wait, render_budget);
            }
        }

        fn set_fault_mode(&mut self, mode: diag::GpuFaultMode) {
            unsafe {
                let raw = std::ptr::read_unaligned(std::ptr::addr_of!(
                    (*self.inst).ioPrivatePluginData
                ));
                let inner = &mut *(raw as *mut GpuFilterInstance);
                inner.fault_mode = mode;
            }
        }

        fn set_kill_switch_path(&mut self, path: Option<std::path::PathBuf>) {
            unsafe {
                let raw = std::ptr::read_unaligned(std::ptr::addr_of!(
                    (*self.inst).ioPrivatePluginData
                ));
                let inner = &mut *(raw as *mut GpuFilterInstance);
                inner.kill_switch_path = path;
            }
        }

        fn get_frame_dependencies(&mut self, seq_time: i64, tpf: i64) -> Vec<i64> {
            let mut rp: sys::PrGPUFilterRenderParams = unsafe { std::mem::zeroed() };
            rp.inSequenceTime = seq_time;
            rp.inClipTime = seq_time;
            rp.inRenderTicksPerFrame = tpf;
            let mut times = Vec::new();
            let mut q = 0i32;
            loop {
                let mut dep: sys::PrGPUFilterFrameDependency = unsafe { std::mem::zeroed() };
                let prev_q = q;
                let e = unsafe {
                    (self.filter.GetFrameDependencies.unwrap())(&mut *self.inst, &rp, &mut q, &mut dep)
                };
                if e == sys::suiteError_NotImplemented {
                    break;
                }
                assert_eq!(e, sys::suiteError_NoError);
                times.push(dep.outSequenceTime);
                if q == prev_q {
                    break;
                }
                assert!(times.len() <= 32);
            }
            times
        }

        fn render(&mut self, frames: &[sys::PPixHand], out: &mut sys::PPixHand) -> i32 {
            let tpf = 10_594_584_000;
            self.render_at_seq_time(100 * tpf, tpf, frames, out)
        }

        fn render_at_seq_time(
            &mut self,
            seq_time: i64,
            tpf: i64,
            frames: &[sys::PPixHand],
            out: &mut sys::PPixHand,
        ) -> i32 {
            let mut rp: sys::PrGPUFilterRenderParams = unsafe { std::mem::zeroed() };
            rp.inSequenceTime = seq_time;
            rp.inClipTime = seq_time;
            rp.inRenderTicksPerFrame = tpf;
            rp.inRenderWidth = 64;
            rp.inRenderHeight = 32;
            rp.inRenderPARNum = 1;
            rp.inRenderPARDen = 1;
            unsafe {
                (self.filter.Render.unwrap())(
                    &mut *self.inst,
                    &rp,
                    frames.as_ptr(),
                    frames.len() as u64,
                    out,
                )
            }
        }
    }

    impl Drop for MockRig {
        fn drop(&mut self) {
            unsafe {
                let _ = (self.filter.DisposeInstance.unwrap())(&mut *self.inst);
            }
        }
    }

    fn make_mock_frames(w: usize, h: usize, rb: usize, private: bool, all_dup: bool) -> (Vec<sys::PPixHand>, Vec<Vec<u8>>) {
        let (dev, queue) = {
            let st = mock_state().lock().unwrap();
            (st.device.clone(), st.queue.clone())
        };
        let mut hs = Vec::new();
        let mut raws = Vec::new();
        for i in 0..9 {
            let shift = if i == 0 || all_dup {
                0.0
            } else if i % 2 == 1 {
                -3.0 * ((i + 1) / 2) as f32
            } else {
                3.0 * ((i + 1) / 2) as f32
            };
            let raw = synth_bgra32f_frame(w, h, rb, shift, 1.0);
            let buf = if private {
                make_private_buffer(&dev, &queue, &raw)
            } else {
                make_shared_buffer(&dev, &raw)
            };
            let handle = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, w as i32, h as i32],
                rb as i32,
                buf,
            );
            hs.push(handle);
            raws.push(raw);
        }
        (hs, raws)
    }

    fn read_mock_ppix_bytes(h: sys::PPixHand) -> Vec<u8> {
        let (dev, queue, buf, len) = {
            let st = mock_state().lock().unwrap();
            let p = st.ppix.get(&(h as usize)).unwrap();
            let b = p._buf.as_ref().unwrap().clone();
            let l = b.length();
            (st.device.clone(), st.queue.clone(), b, l)
        };
        let ctx = unsafe {
            MetalContext::from_raw_handles(
                Retained::as_ptr(&dev) as *mut c_void,
                Retained::as_ptr(&queue) as *mut c_void,
            )
        }
        .unwrap();
        let d = MetalFrameDesc {
            ptr: Retained::as_ptr(&buf) as *const c_void,
            width: len / 16,
            height: 1,
            row_bytes: len as i32,
            layout: Layout::Bgra32f,
        };
        let budget = Cell::new(crate::metal_io::METAL_RENDER_WAIT_BUDGET);
        unsafe { ctx.with_readable_bytes(&d, &Cell::new(false), &budget, |b, _| b.to_vec()) }.unwrap().0
    }

    #[test]
    fn regression_items_1_2_7_8_9_10_via_mock_host() {
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (32usize, 16usize);
        let rb = w * 16 + 32;

        // ── Item 1: `*outFrame` aliases a dependency input (inFrames[3]) on render failure ──
        // Must NEVER overwrite any input frame and MUST return `suiteError_Fail` so Premiere
        // falls back to the CPU effect path (both on param read error and CreateGPUPPix error).
        for (fail_param, fail_create) in [(Some(0), false), (Some(0), true), (None, true)] {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, raws) = make_mock_frames(w, h, rb, false, false);
            {
                let mut st = mock_state().lock().unwrap();
                st.fail_param = fail_param;
                st.fail_create = fail_create;
            }
            let mut out = hs[3];
            let ret = rig.render(&hs, &mut out);
            assert_eq!(ret, sys::suiteError_Fail, "failure must hand off to CPU");
            assert_eq!(out, hs[3]);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);
            // Every input frame MUST remain byte-identical.
            for i in 0..9 {
                assert_eq!(read_mock_ppix_bytes(hs[i]), raws[i], "input {i} corrupted!");
            }
        }

        // ── Item 2: distinct `*outFrame` or null private data on failure -> MUST return Fail
        //    without mutating `*outFrame` or any input frame ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, raws) = make_mock_frames(w, h, rb, false, false);
            let out_rb = w * 16 + 96;
            let dev = mock_state().lock().unwrap().device.clone();
            let out_init = vec![0x99u8; h * out_rb];
            let out_buf = make_shared_buffer(&dev, &out_init);
            let out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, w as i32, h as i32],
                out_rb as i32,
                out_buf,
            );
            mock_state().lock().unwrap().fail_param = Some(0);
            let mut out = out_h;
            let ret = rig.render(&hs, &mut out);
            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(out, out_h);
            assert_eq!(read_mock_ppix_bytes(out_h), out_init);
            for i in 0..9 {
                assert_eq!(read_mock_ppix_bytes(hs[i]), raws[i]);
            }

            // Incompatible distinct output (different width) on render -> MUST return Fail.
            mock_state().lock().unwrap().fail_param = None;
            let bad_out_buf = make_shared_buffer(&dev, &vec![0x77u8; h * (w + 1) * 16]);
            let bad_out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, (w + 1) as i32, h as i32],
                ((w + 1) * 16) as i32,
                bad_out_buf,
            );
            let mut bad_out = bad_out_h;
            let ret2 = rig.render(&hs, &mut bad_out);
            assert_eq!(ret2, sys::suiteError_Fail, "incompatible distinct output must fail");

            // Render after DisposeInstance (both distinct output and `*outFrame == inFrames[0]`)
            // -> MUST return Fail.
            unsafe { (rig.filter.DisposeInstance.unwrap())(&mut *rig.inst) };
            let mut out3 = out_h;
            assert_eq!(rig.render(&hs, &mut out3), sys::suiteError_Fail);
            let mut out4 = hs[0];
            assert_eq!(rig.render(&hs, &mut out4), sys::suiteError_Fail);
        }

        // ── Item 7: bogus bounds / short cur buffer rejected BEFORE calling CreateGPUPPix ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, _) = make_mock_frames(w, h, rb, false, false);
            mock_state().lock().unwrap().ppix.get_mut(&(hs[0] as usize)).unwrap().bounds =
                [-32768, -32768, 32767, 32767];
            let mut out = hs[0];
            let ret = rig.render(&hs, &mut out);
            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(out, hs[0]);
            let st = mock_state().lock().unwrap();
            assert_eq!(st.created_count, 0, "CreateGPUPPix must not be called for bogus bounds");
            assert_eq!(st.big_create_count, 0, "no >2 GiB CreateGPUPPix request allowed");
        }

        // ── Item 8: created output PPix disposed on post-create mismatch, returns Fail ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, _) = make_mock_frames(w, h, rb, false, false);
            mock_state().lock().unwrap().create_size_delta = 1;
            let mut out = hs[0];
            let ret = rig.render(&hs, &mut out);
            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(out, hs[0]);
            let st = mock_state().lock().unwrap();
            assert_eq!(st.created_count, 1);
            assert_eq!(st.disposed_by_plugin, 1);
        }

        // ── Item 9: non-MTLBuffer pointer from GetGPUPPixData rejected cleanly ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (mut hs, _) = make_mock_frames(w, h, rb, false, false);
            // 9a. Current frame is an NSString (non-MTLBuffer) -> fails before CreateGPUPPix.
            let bad_cur = register_mock_non_buffer_ppix(w as i32, h as i32, rb as i32);
            hs[0] = bad_cur;
            let mut out = hs[0];
            let ret = rig.render(&hs, &mut out);
            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);

            // 9b. Neighbour k=1 prev is an NSString -> rejected as Unusable (not_buffer),
            //     render succeeds using k=2 prev!
            reset_mock_host();
            let mut rig2 = MockRig::new();
            let (mut hs2, _) = make_mock_frames(w, h, rb, false, false);
            hs2[1] = register_mock_non_buffer_ppix(w as i32, h as i32, rb as i32);
            let mut out2 = hs2[0];
            let ret2 = rig2.render(&hs2, &mut out2);
            assert_eq!(ret2, sys::suiteError_NoError);
            assert_ne!(out2, hs2[0]);
        }

        // ── Item 10: frame count mismatch (fewer OR more than declared) returns Fail ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, _) = make_mock_frames(w, h, rb, false, false);
            // Fewer frames (only cur, 1 != 9)
            let mut out = hs[0];
            let ret = rig.render(&hs[..1], &mut out);
            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(out, hs[0]);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);

            // More frames (11 != 9)
            let mut extra = hs.clone();
            extra.push(hs[1]);
            extra.push(hs[2]);
            let mut out2 = hs[0];
            let ret2 = rig.render(&extra, &mut out2);
            assert_eq!(ret2, sys::suiteError_Fail);
            assert_eq!(out2, hs[0]);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);
        }
    }

    #[test]
    fn regression_items_4_and_5_gpu_timeouts_bounded_and_no_dispose_in_flight() {
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (32usize, 16usize);
        let rb = w * 16;

        let stall_queue = || -> Retained<ProtocolObject<dyn MTLSharedEvent>> {
            autoreleasepool(|_| {
                let st = mock_state().lock().unwrap();
                let ev = st.device.newSharedEvent().expect("MTLSharedEvent");
                let cb = st.queue.commandBuffer().expect("cb");
                cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*ev), 1);
                cb.commit();
                ev
            })
        };
        let release_and_drain = |ev: Retained<ProtocolObject<dyn MTLSharedEvent>>| {
            ev.setSignaledValue(10);
            autoreleasepool(|_| {
                let st = mock_state().lock().unwrap();
                let cb = st.queue.commandBuffer().expect("cb");
                cb.commit();
                cb.waitUntilCompleted();
            });
        };

        // Verify named timeout constants (5 s per wait, 30 s per render).
        assert_eq!(crate::metal_io::METAL_PER_WAIT_TIMEOUT, Duration::from_secs(5));
        assert_eq!(crate::metal_io::METAL_RENDER_WAIT_BUDGET, Duration::from_secs(30));

        // ── Item 1a (busy healthy queue): 10 private waits paced at ~40ms each (~400ms total)
        //    exceeds a 160ms per-wait limit in total, but every individual wait is <160ms and
        //    total is <1500ms budget -> MUST succeed without false timeout! ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_wait_limits(Duration::from_millis(160), Duration::from_millis(1500));
            mock_state().lock().unwrap().create_private = true;
            // All-duplicate private deps + private cur + private out -> 10 private GPU waits.
            let (hs, _) = make_mock_frames(w, h, rb, true, true);
            let (dev, queue) = {
                let st = mock_state().lock().unwrap();
                (st.device.clone(), st.queue.clone())
            };
            // Pre-populate the queue with a short paced barrier chain (6 steps x 40ms = 240ms)
            // so total GPU wait exceeds the 160ms per-wait limit while each wait completes <160ms.
            let ev = dev.newSharedEvent().expect("ev");
            let ev_bg = ev.clone();
            let q_bg = queue.clone();
            let stop_pacer = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stop_bg = std::sync::Arc::clone(&stop_pacer);
            // Insert one initial barrier on the queue, then pace additional barriers every ~40ms.
            autoreleasepool(|_| {
                let cb = queue.commandBuffer().unwrap();
                cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*ev), 1);
                cb.commit();
            });
            let pacer = std::thread::spawn(move || {
                let mut val = 1u64;
                while !stop_bg.load(std::sync::atomic::Ordering::Relaxed) {
                    std::thread::sleep(Duration::from_millis(40));
                    ev_bg.setSignaledValue(val);
                    val += 1;
                    if !stop_bg.load(std::sync::atomic::Ordering::Relaxed) {
                        autoreleasepool(|_| {
                            let cb = q_bg.commandBuffer().unwrap();
                            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*ev_bg), val);
                            cb.commit();
                        });
                    }
                }
                ev_bg.setSignaledValue(val + 100);
            });

            let mut out = hs[0];
            let ret = rig.render(&hs, &mut out);
            stop_pacer.store(true, std::sync::atomic::Ordering::Relaxed);
            pacer.join().unwrap();
            release_and_drain(ev);

            assert_eq!(ret, sys::suiteError_NoError);
            assert_ne!(
                out, hs[0],
                "busy but healthy paced queue must complete render, not pass through"
            );
            assert_eq!(mock_state().lock().unwrap().disposed_by_plugin, 0);
        }

        // ── Item 1b (hung queue): 8 private duplicate dependencies on a stalled queue MUST abort
        //    after ONE per-wait timeout (~180ms), NOT 8x and NOT full 1080ms render budget! ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_wait_limits(Duration::from_millis(180), Duration::from_millis(1080));
            let (hs, _) = make_mock_frames(w, h, rb, true, true);
            let ev = stall_queue();

            let mut out = hs[0];
            let t0 = Instant::now();
            let ret = rig.render(&hs, &mut out);
            let elapsed = t0.elapsed();
            release_and_drain(ev);

            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(out, hs[0]);
            // Must finish in ~1 per-wait timeout (~180ms), well below 3x per-wait and render budget.
            assert!(
                elapsed < Duration::from_millis(550),
                "expected single per-wait timeout bound (<550ms), got {elapsed:?}"
            );
            // Output buffer was not targeted by an in-flight write (timeout happened during read),
            // so the created PPix MUST be cleanly disposed.
            let st = mock_state().lock().unwrap();
            assert_eq!(st.created_count, 1);
            assert_eq!(st.disposed_by_plugin, 1);
        }

        // ── Item 1c (total render budget cap): per-wait=300ms, total_budget=220ms -> capped by
        //    total_budget (~220ms) across multiple waits! ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_wait_limits(Duration::from_millis(300), Duration::from_millis(220));
            let (hs, _) = make_mock_frames(w, h, rb, true, true);
            let ev = stall_queue();

            let mut out = hs[0];
            let t0 = Instant::now();
            let ret = rig.render(&hs, &mut out);
            let elapsed = t0.elapsed();
            release_and_drain(ev);

            assert_eq!(ret, sys::suiteError_Fail);
            assert_eq!(out, hs[0]);
            assert!(
                elapsed < Duration::from_millis(550),
                "expected total render budget bound (<550ms), got {elapsed:?}"
            );
            let st = mock_state().lock().unwrap();
            assert_eq!(st.created_count, 1);
            assert_eq!(st.disposed_by_plugin, 1);
        }

        // ── Item 4 (preserved): private output write timeout MUST NOT dispose created_ppix while
        //    GPU blit still targets it! ──
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_wait_limits(Duration::from_millis(180), Duration::from_millis(1080));
            mock_state().lock().unwrap().create_private = true;
            let (hs, _) = make_mock_frames(w, h, rb, false, false);

            let (dev, queue) = {
                let st = mock_state().lock().unwrap();
                (st.device.clone(), st.queue.clone())
            };
            let ev = dev.newSharedEvent().expect("ev");
            let ev_bg = ev.clone();
            let q_bg = queue.clone();
            // First test direct `MetalContext::write_rgba_to_buffer` timeout flag:
            let mut ctx = unsafe {
                MetalContext::from_raw_handles(
                    Retained::as_ptr(&dev) as *mut c_void,
                    Retained::as_ptr(&queue) as *mut c_void,
                )
            }
            .unwrap();
            ctx.set_wait_limits(Duration::from_millis(120), Duration::from_millis(720));
            let priv_out = dev
                .newBufferWithLength_options(h * rb, MTLResourceOptions::StorageModePrivate)
                .unwrap();
            let out_desc = MetalFrameDesc {
                ptr: Retained::as_ptr(&priv_out) as *const c_void,
                width: w,
                height: h,
                row_bytes: rb as i32,
                layout: Layout::Bgra32f,
            };
            let cb = q_bg.commandBuffer().unwrap();
            cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*ev_bg), 1);
            cb.commit();
            let err = unsafe {
                ctx.write_rgba_to_buffer(
                    &vec![0.5f32; w * h * 4],
                    &out_desc,
                    &Cell::new(true),
                    &Cell::new(Duration::from_millis(720)),
                )
            }
            .unwrap_err();
            assert!(err.is_timeout() && err.targets_output(), "expected in-flight output timeout, got {err:?}");
            release_and_drain(ev);

            // End-to-end through `cb_render` with a larger frame (640x360) so CPU optical flow
            // takes >20ms while a background thread stalls the queue before `write_private_blit`.
            reset_mock_host();
            mock_state().lock().unwrap().create_private = true;
            let mut rig2 = MockRig::new();
            rig2.set_wait_limits(Duration::from_millis(200), Duration::from_millis(1200));
            let (bw, bh) = (640usize, 360usize);
            let brb = bw * 16;
            let (hs2, _) = make_mock_frames(bw, bh, brb, false, false);

            let ev2 = dev.newSharedEvent().expect("ev2");
            let ev2_bg = ev2.clone();
            let q2_bg = queue.clone();
            let bg = std::thread::spawn(move || {
                // Shared readback finishes in <2 ms; flow+blur takes ~25-250 ms -> stalling at
                // 8 ms reliably lands right before `write_private_blit`!
                std::thread::sleep(Duration::from_millis(8));
                autoreleasepool(|_| {
                    let cb = q2_bg.commandBuffer().unwrap();
                    cb.encodeWaitForEvent_value(ProtocolObject::from_ref(&*ev2_bg), 1);
                    cb.commit();
                });
            });

            let mut out2 = hs2[0];
            let ret2 = rig2.render(&hs2, &mut out2);
            bg.join().unwrap();
            let (created_now, disposed_now) = {
                let st = mock_state().lock().unwrap();
                (st.created_count, st.disposed_by_plugin)
            };
            // Release queue so any in-flight blit completes safely into the live buffer.
            release_and_drain(ev2);

            assert_eq!(ret2, sys::suiteError_Fail);
            assert_eq!(out2, hs2[0]);
            assert_eq!(created_now, 1);
            // If the timeout hit during `write_private_blit`, disposed_now MUST be 0 (not disposed
            // while GPU work targets it); verify explicitly that if the created private PPix was
            // ever disposed, GPU did NOT write non-zero data into it after unblock.
            let st = mock_state().lock().unwrap();
            let created_ppix = st.ppix.values().find(|p| p.created_by_plugin).unwrap();
            if created_ppix.disposed {
                let buf = created_ppix._buf.as_ref().unwrap().clone();
                drop(st);
                let after = unsafe {
                    ctx.with_readable_bytes(
                        &MetalFrameDesc {
                            ptr: Retained::as_ptr(&buf) as *const c_void,
                            width: bw,
                            height: bh,
                            row_bytes: brb as i32,
                            layout: Layout::Bgra32f,
                        },
                        &Cell::new(false),
                        &Cell::new(crate::metal_io::METAL_RENDER_WAIT_BUDGET),
                        |b, _| b.to_vec(),
                    )
                }
                .unwrap()
                .0;
                assert!(
                    after.iter().all(|&b| b == 0),
                    "GPU wrote into a frame that plugin disposed!"
                );
            } else {
                assert_eq!(disposed_now, 0);
            }
            let _ = hs;
        }
    }

    #[test]
    fn item_11_report_unusable_counts_and_dup_ref() {
        let (dev, queue, ctx) = make_metal_context();
        let (w, h) = (32usize, 16usize);
        let rb = w * 16;

        let cur_raw = synth_bgra32f_frame(w, h, rb, 0.0, 1.0);
        let cur_buf = make_shared_buffer(&dev, &cur_raw);
        let out_buf = make_shared_buffer(&dev, &vec![0u8; h * rb]);
        let cur_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&cur_buf) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb as i32,
            layout: Layout::Bgra32f,
        };
        let out_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&out_buf) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb as i32,
            layout: Layout::Bgra32f,
        };

        let ns_bad = NSString::from_str("not_a_buffer");
        let wrong_size_buf = make_shared_buffer(&dev, &vec![0u8; h * (w + 1) * 16]);
        let alpha0_buf = make_shared_buffer(&dev, &synth_bgra32f_frame(w, h, rb, 4.0, 0.0));
        let good_next_buf = make_shared_buffer(&dev, &synth_bgra32f_frame(w, h, rb, 5.0, 1.0));

        // Prev side (indices 0, 2, 4, 6 -> k=1..4): BadFormat, BadGeometry, not_buffer, size_mismatch
        // Next side (indices 1, 3, 5, 7 -> k=1..4): Missing, transparent (alpha=0), good next, unused
        let deps = vec![
            DepFrame::BadFormat,
            DepFrame::Missing,
            DepFrame::BadGeometry,
            DepFrame::Candidate(MetalFrameDesc {
                ptr: Retained::as_ptr(&alpha0_buf) as *const c_void,
                width: w,
                height: h,
                row_bytes: rb as i32,
                layout: Layout::Bgra32f,
            }),
            DepFrame::Candidate(MetalFrameDesc {
                ptr: Retained::as_ptr(&ns_bad) as *const c_void,
                width: w,
                height: h,
                row_bytes: rb as i32,
                layout: Layout::Bgra32f,
            }),
            DepFrame::Candidate(MetalFrameDesc {
                ptr: Retained::as_ptr(&good_next_buf) as *const c_void,
                width: w,
                height: h,
                row_bytes: rb as i32,
                layout: Layout::Bgra32f,
            }),
            DepFrame::Candidate(MetalFrameDesc {
                ptr: Retained::as_ptr(&wrong_size_buf) as *const c_void,
                width: w + 1,
                height: h,
                row_bytes: ((w + 1) * 16) as i32,
                layout: Layout::Bgra32f,
            }),
            DepFrame::Missing,
        ];

        let params = core::Params::default();
        let budget = Cell::new(crate::metal_io::METAL_RENDER_WAIT_BUDGET);
        let rep = unsafe {
            render_metal_frames(
                &ctx,
                &params,
                &cur_desc,
                &deps,
                &out_desc,
                GPU_DEPENDENCY_STEPS,
                &budget,
            )
        }
        .expect("render with unusable deps");

        assert_eq!(rep.dup_ref, "inFrames[0]");
        assert_eq!(rep.prev_k, None);
        assert_eq!(rep.next_k, Some(3));
        assert_eq!(rep.unusable.missing, 1);
        assert_eq!(rep.unusable.bad_format, 1);
        assert_eq!(rep.unusable.not_buffer, 1);
        assert_eq!(rep.unusable.size_mismatch, 1);
        assert_eq!(rep.unusable.bad_geometry, 1);
        assert_eq!(rep.unusable.transparent, 1);
        assert_eq!(rep.unusable.read_err, 0);
        assert_eq!(
            rep.unusable.format_log(),
            "missing:1,fmt:1,not_buf:1,size:1,geom:1,alpha0:1,read:0"
        );
        let _ = queue;
    }

    #[test]
    fn sequence_start_dependencies_omit_negative_times_and_match_cpu_selection() {
        // Item 2: at sequence time 0, 1, and 3 frames with 4 steps, declared dependency counts
        // are 4, 5, and 7 (never declaring negative sequence times), and GPU render matches CPU
        // `select_legacy_neighbors` with those undeclared neighbours treated as missing.
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let tpf = 10_594_584_000i64;
        let params = core::Params::default();
        let (w, h) = (32usize, 16usize);
        let rb = w * 16 + 32;
        let desc = pixels::ImageDesc {
            width: w,
            height: h,
            rowbytes: rb as isize,
            layout: Layout::Bgra32f,
            alpha: pixels::Alpha::Straight,
        };

        // Synth timeline of frames 0..8 where frames 0, 1, 2 are duplicates of cur when testing
        // duplicate-skip at sequence start, plus distinct frames at every other index.
        let cur_frame_idx_cases: [(i64, usize); 3] = [(0, 4), (1, 5), (3, 7)];

        for (seq_frame, expected_declared) in cur_frame_idx_cases {
            let seq_time = seq_frame * tpf;
            let declared = gpu_declared_dependencies(&params, seq_time, tpf, 4);
            assert_eq!(declared.len(), expected_declared, "seq_frame={seq_frame}");
            assert!(
                declared.iter().all(|d| d.sequence_time >= 0),
                "negative sequence time declared at seq_frame={seq_frame}: {declared:?}"
            );

            // Verify `GetFrameDependencies` FFI callback returns the EXACT same list.
            reset_mock_host();
            let mut rig = MockRig::new();
            let cb_times = rig.get_frame_dependencies(seq_time, tpf);
            let want_times: Vec<i64> = declared.iter().map(|d| d.sequence_time).collect();
            assert_eq!(cb_times, want_times, "cb_get_frame_dependencies mismatch at seq_frame={seq_frame}");

            // Test two content patterns:
            // (a) distinct frames at all non-negative sequence times;
            // (b) duplicate at k=1 prev & next (tests skipping to k=2 where prev@k=2 is missing at seq_frame=1!).
            for dup_k1 in [false, true] {
                reset_mock_host();
                let mut rig = MockRig::new();
                let dev = mock_state().lock().unwrap().device.clone();

                let cur_raw = synth_bgra32f_frame(w, h, rb, 0.0, 1.0);
                let mut cur_rgba = Vec::new();
                let (cur_luma, _) = pixels::read_rgba_and_luma(&cur_raw, &desc, &mut cur_rgba).unwrap();

                let make_dep_raw = |dir: i32, k: u32| -> Vec<u8> {
                    if dup_k1 && k == 1 {
                        cur_raw.clone()
                    } else {
                        let shift = (dir as f32) * (k as f32) * 3.5;
                        synth_bgra32f_frame(w, h, rb, shift, 1.0)
                    }
                };

                // Build inFrames = [cur, declared deps in exact GetFrameDependencies order].
                let cur_h = register_mock_ppix(
                    sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                    [0, 0, w as i32, h as i32],
                    rb as i32,
                    make_shared_buffer(&dev, &cur_raw),
                );
                let mut frames = vec![cur_h];
                let mut dep_raws_by_dk: HashMap<(i32, u32), Vec<u8>> = HashMap::new();
                for d in &declared {
                    let raw = make_dep_raw(d.dir, d.k);
                    let h_dep = register_mock_ppix(
                        sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                        [0, 0, w as i32, h as i32],
                        rb as i32,
                        make_shared_buffer(&dev, &raw),
                    );
                    frames.push(h_dep);
                    dep_raws_by_dk.insert((d.dir, d.k), raw);
                }

                // Compute CPU reference using `effect::select_legacy_neighbors` with undeclared
                // negative-time neighbours returning `Outcome::Missing`.
                let cur_l = core::Luma { width: w, height: h, data: &cur_luma };
                let is_dup = |l: &Vec<f32>| {
                    core::is_duplicate_frame(cur_l, core::Luma { width: w, height: h, data: l })
                };
                let cpu_sel = effect::select_legacy_neighbors(
                    4,
                    |dir, k| {
                        let Some(raw) = dep_raws_by_dk.get(&(dir, k)) else {
                            return Err(Outcome::Missing);
                        };
                        Ok(pixels::read_luma(raw, &desc).unwrap().0)
                    },
                    &is_dup,
                );
                let prev_ref = cpu_sel
                    .prev
                    .as_ref()
                    .and_then(|(k, _)| dep_raws_by_dk.get(&(-1, *k)).map(|r| (r.as_slice(), cpu_sel.fd.0.unwrap())));
                let next_ref = cpu_sel
                    .next
                    .as_ref()
                    .and_then(|(k, _)| dep_raws_by_dk.get(&(1, *k)).map(|r| (r.as_slice(), cpu_sel.fd.1.unwrap())));
                let cpu_out = run_cpu_bgra32f(&cur_raw, prev_ref, next_ref, w, h, rb, &params);

                // Run GPU `cb_render` at `seq_time` with the exact declared frame list.
                let mut out = cur_h;
                let ret = rig.render_at_seq_time(seq_time, tpf, &frames, &mut out);
                assert_eq!(ret, sys::suiteError_NoError, "seq_frame={seq_frame} dup_k1={dup_k1}");
                assert_ne!(out, cur_h, "must render (not pass through) at seq_frame={seq_frame}");
                let gpu_out = read_mock_ppix_bytes(out);
                for y in 0..h {
                    assert_eq!(
                        &gpu_out[y * w * 16..(y + 1) * w * 16],
                        &cpu_out[y * rb..y * rb + w * 16],
                        "GPU vs CPU mismatch at seq_frame={seq_frame} dup_k1={dup_k1} row={y}"
                    );
                }

                // Count check still rejects genuinely wrong frame counts (e.g. one fewer than declared).
                let mut out_bad = cur_h;
                let ret_bad =
                    rig.render_at_seq_time(seq_time, tpf, &frames[..frames.len() - 1], &mut out_bad);
                assert_eq!(ret_bad, sys::suiteError_Fail);
                assert_eq!(out_bad, cur_h, "count mismatch must fail and leave out untouched");
            }
        }
    }

    #[test]
    fn debug_gpu_fault_injection_modes_via_mock_host() {
        // Item 4: verify `error`, `passthrough`, and `timeout` fault modes end-to-end in cb_render.
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (32usize, 16usize);
        let rb = w * 16 + 32;

        // 1. `error`: returns Fail immediately without touching any frame or `*outFrame`.
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_fault_mode(diag::GpuFaultMode::Error);
            let (hs, raws) = make_mock_frames(w, h, rb, false, false);
            let dev = mock_state().lock().unwrap().device.clone();
            let out_init = vec![0x42u8; h * rb];
            let out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, w as i32, h as i32],
                rb as i32,
                make_shared_buffer(&dev, &out_init),
            );
            let mut out = out_h;
            let ret = rig.render(&hs, &mut out);
            assert_ne!(ret, sys::suiteError_NoError);
            assert_eq!(out, out_h);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);
            assert_eq!(read_mock_ppix_bytes(out_h), out_init);
            for i in 0..9 {
                assert_eq!(read_mock_ppix_bytes(hs[i]), raws[i]);
            }
        }

        // 2. `passthrough`: outputs current frame unchanged and returns NoError (never mutating inputs).
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_fault_mode(diag::GpuFaultMode::Passthrough);
            let (hs, raws) = make_mock_frames(w, h, rb, false, false);

            // 2a. `*outFrame == inFrames[0]` -> stays `inFrames[0]`, NoError, no alloc.
            let mut out_a = hs[0];
            assert_eq!(rig.render(&hs, &mut out_a), sys::suiteError_NoError);
            assert_eq!(out_a, hs[0]);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);

            // 2b. `*outFrame == inFrames[2]` (aliases dep) -> allocates fresh copy of cur, dep untouched.
            let mut out_b = hs[2];
            assert_eq!(rig.render(&hs, &mut out_b), sys::suiteError_NoError);
            assert_ne!(out_b, hs[2]);
            let got_b = read_mock_ppix_bytes(out_b);
            for y in 0..h {
                assert_eq!(&got_b[y * w * 16..(y + 1) * w * 16], &raws[0][y * rb..y * rb + w * 16]);
            }
            for i in 0..9 {
                assert_eq!(read_mock_ppix_bytes(hs[i]), raws[i]);
            }

            // 2c. Distinct `*outFrame` with differing `row_bytes` -> copies rows cleanly and returns NoError;
            //     incompatible width -> returns Fail.
            let dev = mock_state().lock().unwrap().device.clone();
            let out_rb = w * 16 + 96;
            let out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, w as i32, h as i32],
                out_rb as i32,
                make_shared_buffer(&dev, &vec![0x99u8; h * out_rb]),
            );
            let mut out_c = out_h;
            assert_eq!(rig.render(&hs, &mut out_c), sys::suiteError_NoError);
            assert_eq!(out_c, out_h);
            let got_c = read_mock_ppix_bytes(out_h);
            for y in 0..h {
                assert_eq!(&got_c[y * out_rb..y * out_rb + w * 16], &raws[0][y * rb..y * rb + w * 16]);
            }

            let bad_out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, (w + 1) as i32, h as i32],
                ((w + 1) * 16) as i32,
                make_shared_buffer(&dev, &vec![0x77u8; h * (w + 1) * 16]),
            );
            let mut bad_out = bad_out_h;
            assert_eq!(rig.render(&hs, &mut bad_out), sys::suiteError_Fail);
        }

        // 3. `timeout`: behaves as if the first GPU wait timed out (instant, no 5s stall)
        //    and returns Fail so Premiere falls back to the CPU path.
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            rig.set_fault_mode(diag::GpuFaultMode::Timeout);
            let (hs, _) = make_mock_frames(w, h, rb, false, false);

            // 3a. `*outFrame == inFrames[0]` -> created PPix disposed cleanly, returns Fail.
            let mut out_a = hs[0];
            let t0 = Instant::now();
            let ret_a = rig.render(&hs, &mut out_a);
            assert!(t0.elapsed() < Duration::from_millis(500));
            assert_eq!(ret_a, sys::suiteError_Fail);
            assert_eq!(out_a, hs[0]);
            {
                let st = mock_state().lock().unwrap();
                assert_eq!(st.created_count, 1);
                assert_eq!(st.disposed_by_plugin, 1);
            }

            // 3b. Distinct `*outFrame` -> cannot copy after timeout, returns Fail.
            let dev = mock_state().lock().unwrap().device.clone();
            let out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, w as i32, h as i32],
                rb as i32,
                make_shared_buffer(&dev, &vec![0x11u8; h * rb]),
            );
            let mut out_b = out_h;
            let ret_b = rig.render(&hs, &mut out_b);
            assert_ne!(ret_b, sys::suiteError_NoError);
        }
    }

    #[test]
    fn gpu_gate_decision_table_and_runtime_id_parsing() {
        use diag::GpuForceMode;

        // RuntimeInstanceID parsing.
        assert_eq!(parse_runtime_instance_id("60210"), Some(60210));
        assert_eq!(parse_runtime_instance_id("  60288\n"), Some(60288));
        assert_eq!(parse_runtime_instance_id("-1"), Some(-1));
        assert_eq!(parse_runtime_instance_id("0x0000eb32"), Some(0xeb32));
        assert_eq!(parse_runtime_instance_id(""), None);
        assert_eq!(parse_runtime_instance_id("not-an-id"), None);

        // Normal mode (Force::None): requires BOTH PF synthetic map AND adjustment-layer node
        // signature (effect node whose operator-owner lookup fails).
        assert_eq!(
            decide_gpu_gate(GpuForceMode::None, NODE_TYPE_EFFECT, None, Some(OWNER_ID_NOT_VALID), Some(60210), Some(true)),
            GateDecision { accept: true, reason: "pf_synthetic", forced: "none" }
        );
        // Stale synthetic entry on a clip-owned effect node MUST be declined!
        assert_eq!(
            decide_gpu_gate(GpuForceMode::None, NODE_TYPE_EFFECT, Some(200), None, Some(60210), Some(true)),
            GateDecision { accept: false, reason: "not_adjustment_node", forced: "none" }
        );
        // Stale synthetic entry on a non-effect node MUST be declined!
        assert_eq!(
            decide_gpu_gate(GpuForceMode::None, "RenderableNodeClipImpl", None, Some(OWNER_ID_NOT_VALID), Some(60210), Some(true)),
            GateDecision { accept: false, reason: "not_adjustment_node", forced: "none" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::None, NODE_TYPE_EFFECT, Some(200), None, Some(60210), Some(false)),
            GateDecision { accept: false, reason: "pf_not_synthetic", forced: "none" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::None, NODE_TYPE_EFFECT, None, Some(OWNER_ID_NOT_VALID), Some(60210), None),
            GateDecision { accept: false, reason: "pf_unknown_id", forced: "none" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::None, NODE_TYPE_EFFECT, None, Some(OWNER_ID_NOT_VALID), None, None),
            GateDecision { accept: false, reason: "no_runtime_instance_id", forced: "none" }
        );

        // Force::Adjust: narrowed to exact OWNER_ID_NOT_VALID (-2147483643), NOT arbitrary errors.
        assert_eq!(
            decide_gpu_gate(GpuForceMode::Adjust, NODE_TYPE_EFFECT, None, Some(OWNER_ID_NOT_VALID), Some(60210), None),
            GateDecision { accept: true, reason: "forced_adjust", forced: "adjust" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::Adjust, NODE_TYPE_EFFECT, None, Some(OWNER_ID_NOT_VALID), None, None),
            GateDecision { accept: true, reason: "forced_adjust", forced: "adjust" }
        );
        // Non-IDNotValid owner error must NOT trigger forced_adjust.
        assert_eq!(
            decide_gpu_gate(GpuForceMode::Adjust, NODE_TYPE_EFFECT, None, Some(-1), Some(60210), None),
            GateDecision { accept: false, reason: "pf_unknown_id", forced: "none" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::Adjust, NODE_TYPE_EFFECT, Some(200), None, Some(60210), Some(false)),
            GateDecision { accept: false, reason: "pf_not_synthetic", forced: "none" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::Adjust, NODE_TYPE_EFFECT, Some(200), None, Some(60210), Some(true)),
            GateDecision { accept: false, reason: "not_adjustment_node", forced: "none" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::Adjust, "RenderableNodeClipImpl", None, Some(OWNER_ID_NOT_VALID), Some(60210), None),
            GateDecision { accept: false, reason: "pf_unknown_id", forced: "none" }
        );

        // Force::All: accepts everything regardless of node type, owner, or PF map.
        assert_eq!(
            decide_gpu_gate(GpuForceMode::All, NODE_TYPE_EFFECT, Some(200), None, None, None),
            GateDecision { accept: true, reason: "forced_all", forced: "all" }
        );
        assert_eq!(
            decide_gpu_gate(GpuForceMode::All, NODE_TYPE_EFFECT, Some(200), None, Some(60210), Some(false)),
            GateDecision { accept: true, reason: "forced_all", forced: "all" }
        );

        // Item 6: host strings sanitized for logs / dump keys are truncated to 64 bytes.
        let long_host_str = "A".repeat(200);
        let tok = diag::sanitize_token(&long_host_str);
        assert_eq!(tok.len(), 64);
        assert!(tok.bytes().all(|b| b == b'A'));
    }

    #[test]
    fn debug_gpu_force_override_parsing_uses_safe_reader() {
        use diag::GpuForceMode;
        let base = std::env::temp_dir().join(format!(
            "tblur-force-flag-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();

        let force_path = base.join("force");
        let link_path = base.join("force.link");
        let fifo_path = base.join("force.pipe");
        let huge_path = base.join("force.huge");
        let dir_path = base.join("force.dir");
        std::fs::create_dir_all(&dir_path).unwrap();

        // Missing file -> None.
        assert_eq!(diag::resolve_gpu_force_from(true, &force_path), GpuForceMode::None);

        // Valid values & debug-off gate.
        for (txt, want) in [
            ("adjust", GpuForceMode::Adjust),
            ("adjust\n", GpuForceMode::Adjust),
            (" all \r\n", GpuForceMode::All),
        ] {
            std::fs::write(&force_path, txt).unwrap();
            assert_eq!(diag::resolve_gpu_force_from(true, &force_path), want, "txt={txt:?}");
            assert_eq!(
                diag::resolve_gpu_force_from(false, &force_path),
                GpuForceMode::None,
                "debug=false must ignore force file"
            );
        }
        for bad in ["", "none", "ADJUST", "ALL", "adjust all", "true", "1"] {
            std::fs::write(&force_path, bad).unwrap();
            assert_eq!(diag::resolve_gpu_force_from(true, &force_path), GpuForceMode::None, "bad={bad:?}");
        }

        // Symlink, directory, huge file, and FIFO all rejected without hanging.
        std::fs::write(&force_path, "adjust\n").unwrap();
        std::os::unix::fs::symlink(&force_path, &link_path).unwrap();
        assert_eq!(diag::resolve_gpu_force_from(true, &link_path), GpuForceMode::None);
        assert_eq!(diag::resolve_gpu_force_from(true, &dir_path), GpuForceMode::None);

        let mut huge = b"all\n".to_vec();
        huge.resize(1024 * 1024, b' ');
        std::fs::write(&huge_path, &huge).unwrap();
        assert_eq!(diag::resolve_gpu_force_from(true, &huge_path), GpuForceMode::None);

        let st = std::process::Command::new("mkfifo")
            .arg(&fifo_path)
            .status()
            .expect("mkfifo");
        assert!(st.success());
        let t0 = Instant::now();
        assert_eq!(diag::resolve_gpu_force_from(true, &fifo_path), GpuForceMode::None);
        assert!(t0.elapsed() < Duration::from_millis(200));

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn create_instance_pf_synthetic_map_and_graph_dump_release_all_handles() {
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        const TEST_RTID: i32 = 90555;
        instance_registry::global().remove(TEST_RTID);

        // 1. Unknown runtime_instance_id -> CreateInstance declines (CPU fallback), freeing property ptr.
        reset_mock_host();
        mock_state().lock().unwrap().runtime_id_prop = Some(TEST_RTID.to_string());
        let (e_unk, rig_unk) = MockRig::try_new_raw(100);
        assert_ne!(e_unk, sys::suiteError_NoError, "unknown id must decline");
        drop(rig_unk);
        {
            let st = mock_state().lock().unwrap();
            assert_eq!(st.mem_allocs, 1);
            assert_eq!(st.mem_frees, 1);
            assert_eq!(st.nodes_acquired, st.nodes_released);
        }

        // 2. PF records synthetic=false -> CreateInstance still declines.
        instance_registry::global().record(TEST_RTID, false, "sequence_setup");
        let (e_false, rig_false) = MockRig::try_new_raw(100);
        assert_ne!(e_false, sys::suiteError_NoError, "synthetic=false must decline");
        drop(rig_false);

        // 3. PF records synthetic=true on an adjustment-like node (owner_fail=true) -> accepts!
        instance_registry::global().record(TEST_RTID, true, "render");
        let (e_true, rig_true) = MockRig::try_new_raw(100);
        assert_eq!(e_true, sys::suiteError_NoError, "synthetic=true on adjustment node must accept");
        drop(rig_true);

        // 3b. Item 1 reproduction: stale synthetic=true entry + ordinary clip node (owner
        //     RenderableNodeClipImpl with RenderableNodeMediaImpl input 0) -> MUST decline!
        mock_state().lock().unwrap().owner_fail = false;
        let (e_stale_clip, rig_stale_clip) = MockRig::try_new_raw(100);
        assert_ne!(
            e_stale_clip,
            sys::suiteError_NoError,
            "stale synthetic entry on a clip-owned node must be declined"
        );
        drop(rig_stale_clip);
        mock_state().lock().unwrap().owner_fail = true;

        // 3c. Item 2: unterminated 8-byte GetNodeProperty buffer (no NUL byte) is read strictly
        //     within GetPtrSize without OOB read, and missing GetPtrSize declines cleanly.
        {
            reset_mock_host();
            mock_state().lock().unwrap().unterminated_prop_bytes = Some(b"90555   ".to_vec());
            let (e_unterm, rig_unterm) = MockRig::try_new_raw(100);
            assert_eq!(
                e_unterm,
                sys::suiteError_NoError,
                "unterminated 8-byte buffer with valid GetPtrSize should parse safely"
            );
            drop(rig_unterm);
            {
                let st = mock_state().lock().unwrap();
                assert_eq!(st.mem_allocs, 1);
                assert_eq!(st.mem_frees, 1);
            }

            // Without GetPtrSize, read_node_property MUST decline rather than scanning with CStr.
            reset_mock_host();
            {
                let mut st = mock_state().lock().unwrap();
                st.unterminated_prop_bytes = Some(b"90555   ".to_vec());
                st.fail_get_ptr_size = true;
            }
            let (e_nosize, rig_nosize) = MockRig::try_new_raw(100);
            assert_ne!(
                e_nosize,
                sys::suiteError_NoError,
                "missing GetPtrSize must decline rather than scan past buffer"
            );
            drop(rig_nosize);
            {
                let st = mock_state().lock().unwrap();
                assert_eq!(st.mem_allocs, 1);
                assert_eq!(st.mem_frees, 1);
                assert_eq!(st.nodes_acquired, st.nodes_released);
            }
        }

        // 4. Graph dump walks bounded segments, locates `*THIS*` on segment 1 operator 402,
        //    and releases every acquired VideoSegmentsID, VideoNodeID, and PrMemoryPtr.
        reset_mock_host();
        mock_state().lock().unwrap().runtime_id_prop = Some(TEST_RTID.to_string());
        let pi = mock_pi_suites();
        let sp = unsafe { sp_basic_from_pi_suites(pi) }.unwrap();
        let node_cls = unsafe { inspect_node_is_adjustment(sp, pi, 100) }.unwrap();
        let dump = unsafe { collect_segment_graph_dump(sp, pi, 1, 100, &node_cls) };
        assert!(dump.contains("found_this=true"), "dump={dump}");
        assert!(dump.contains("seg_idx=1"), "should locate segment 1 containing *THIS*: {dump}");
        assert!(
            dump.contains("*THIS*op0@d1:RenderableNodeEffectImpl(f=0x4c,rtid=90555"),
            "missing *THIS* marker with type, flags, and rtid: {dump}"
        );
        {
            let st = mock_state().lock().unwrap();
            assert!(st.segs_acquired > 0 && st.segs_acquired == st.segs_released);
            assert!(st.nodes_acquired > 0 && st.nodes_acquired == st.nodes_released);
            assert!(st.mem_allocs > 0 && st.mem_allocs == st.mem_frees);
        }

        assert!(instance_registry::global().remove(TEST_RTID));
        assert!(!instance_registry::global().remove(TEST_RTID));
    }

    #[test]
    fn debug_flag_ownership_and_writable_bits_rejected() {
        use std::os::unix::fs::PermissionsExt;

        // Pure owner/mode check (covers wrong-UID case that non-root tests cannot chown).
        assert!(diag::is_trusted_owner_and_mode(501, 0o100600, 501));
        assert!(diag::is_trusted_owner_and_mode(501, 0o100644, 501));
        assert!(diag::is_trusted_owner_and_mode(501, 0o100400, 501));
        assert!(!diag::is_trusted_owner_and_mode(502, 0o100600, 501));
        assert!(!diag::is_trusted_owner_and_mode(0, 0o100644, 501));
        assert!(!diag::is_trusted_owner_and_mode(501, 0o100664, 501));
        assert!(!diag::is_trusted_owner_and_mode(501, 0o100646, 501));
        assert!(!diag::is_trusted_owner_and_mode(501, 0o100666, 501));

        // Real filesystem checks across -debug, -gpu-steps, -gpu-fault, and -gpu-force.
        let base = std::env::temp_dir().join(format!(
            "tblur-perm-flag-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();

        let dbg_f = base.join("debug");
        let steps_f = base.join("steps");
        let fault_f = base.join("fault");
        let force_f = base.join("force");
        let dbg_link = base.join("debug.link");

        std::fs::write(&dbg_f, "").unwrap();
        std::fs::write(&steps_f, "2\n").unwrap();
        std::fs::write(&fault_f, "error\n").unwrap();
        std::fs::write(&force_f, "adjust\n").unwrap();
        std::os::unix::fs::symlink(&dbg_f, &dbg_link).unwrap();

        // Owner-only (0600) and owner-write/world-read (0644): accepted.
        for mode in [0o600, 0o644] {
            for p in [&dbg_f, &steps_f, &fault_f, &force_f] {
                std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap();
            }
            assert!(diag::flag_enabled_from(&dbg_f), "mode={mode:o}");
            assert!(!diag::flag_enabled_from(&dbg_link), "symlink must be rejected");
            assert_eq!(diag::resolve_gpu_steps_from(true, &steps_f, 4), (2, true));
            assert_eq!(diag::resolve_gpu_fault_from(true, &fault_f), diag::GpuFaultMode::Error);
            assert_eq!(diag::resolve_gpu_force_from(true, &force_f), diag::GpuForceMode::Adjust);
        }

        // Group-writable (0664), world-writable (0646), or both (0666): ALL rejected as absent.
        for bad_mode in [0o664, 0o646, 0o666] {
            for p in [&dbg_f, &steps_f, &fault_f, &force_f] {
                std::fs::set_permissions(p, std::fs::Permissions::from_mode(bad_mode)).unwrap();
            }
            assert!(!diag::flag_enabled_from(&dbg_f), "bad_mode={bad_mode:o}");
            assert_eq!(diag::resolve_gpu_steps_from(true, &steps_f, 4), (4, false));
            assert_eq!(diag::resolve_gpu_fault_from(true, &fault_f), diag::GpuFaultMode::None);
            assert_eq!(diag::resolve_gpu_force_from(true, &force_f), diag::GpuForceMode::None);
        }

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn render_kill_switch_hands_off_to_cpu_immediately_and_recovers() {
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let base = std::env::temp_dir().join(format!(
            "tblur-kill-switch-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&base).unwrap();
        let flag = base.join("no-gpu");

        reset_mock_host();
        let mut rig = MockRig::new();
        rig.set_kill_switch_path(Some(flag.clone()));
        let (w, h) = (32usize, 16usize);
        let rb = w * 16 + 32;
        let (hs, raws) = make_mock_frames(w, h, rb, false, false);

        // 1. Kill switch absent -> existing instance renders normally.
        let mut out1 = hs[0];
        assert_eq!(rig.render(&hs, &mut out1), sys::suiteError_NoError);
        assert_ne!(out1, hs[0]);
        let created_before = mock_state().lock().unwrap().created_count;
        assert_eq!(created_before, 1);

        // 2. Create kill switch file -> next Render on the SAME existing instance immediately
        //    returns `suiteError_Fail` without allocating or mutating any frame.
        std::fs::write(&flag, "").unwrap();
        let mut out2 = hs[0];
        assert_eq!(rig.render(&hs, &mut out2), sys::suiteError_Fail);
        assert_eq!(out2, hs[0]);
        assert_eq!(
            mock_state().lock().unwrap().created_count,
            created_before,
            "kill switch must bail before CreateGPUPPix"
        );
        for i in 0..9 {
            assert_eq!(read_mock_ppix_bytes(hs[i]), raws[i]);
        }

        // Also overrides debug fault mode `passthrough` on an existing instance.
        rig.set_fault_mode(diag::GpuFaultMode::Passthrough);
        let mut out3 = hs[0];
        assert_eq!(rig.render(&hs, &mut out3), sys::suiteError_Fail);
        rig.set_fault_mode(diag::GpuFaultMode::None);

        // 3. Remove kill switch file -> subsequent Render on the same instance succeeds again.
        std::fs::remove_file(&flag).unwrap();
        let mut out4 = hs[0];
        assert_eq!(rig.render(&hs, &mut out4), sys::suiteError_NoError);
        assert_ne!(out4, hs[0]);

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn render_no_blur_and_no_usable_neighbor_return_exact_copy() {
        let _guard = MOCK_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (32usize, 16usize);
        let rb = w * 16 + 32;

        // 1. `blur_amount == 0.0` (0 declared deps -> inFrames = [cur]) returns NoError with exact
        //    unmodified copy of cur across all three `*outFrame` shapes.
        {
            reset_mock_host();
            mock_state().lock().unwrap().blur_amount = 0.0;
            let mut rig = MockRig::new();
            let (hs, raws) = make_mock_frames(w, h, rb, false, false);
            let cur_only = &hs[..1];

            // 1a. `*outFrame == inFrames[0]` -> zero-alloc no-blur pass-through.
            let mut out_a = hs[0];
            assert_eq!(rig.render(cur_only, &mut out_a), sys::suiteError_NoError);
            assert_eq!(out_a, hs[0]);
            assert_eq!(mock_state().lock().unwrap().created_count, 0);

            // 1b. Distinct `*outFrame` with differing `row_bytes` -> exact row copy.
            let dev = mock_state().lock().unwrap().device.clone();
            let out_rb = w * 16 + 64;
            let out_h = register_mock_ppix(
                sys::PrPixelFormatGpu_GPU_BGRA_4444_32f,
                [0, 0, w as i32, h as i32],
                out_rb as i32,
                make_shared_buffer(&dev, &vec![0xAAu8; h * out_rb]),
            );
            let mut out_b = out_h;
            assert_eq!(rig.render(cur_only, &mut out_b), sys::suiteError_NoError);
            assert_eq!(out_b, out_h);
            let got_b = read_mock_ppix_bytes(out_h);
            for y in 0..h {
                assert_eq!(&got_b[y * out_rb..y * out_rb + w * 16], &raws[0][y * rb..y * rb + w * 16]);
            }

            // 1c. Null `*outFrame` -> allocates fresh GPU PPix containing exact cur pixels.
            let mut out_c: sys::PPixHand = std::ptr::null_mut();
            assert_eq!(rig.render(cur_only, &mut out_c), sys::suiteError_NoError);
            assert!(!out_c.is_null() && out_c != hs[0]);
            let got_c = read_mock_ppix_bytes(out_c);
            for y in 0..h {
                assert_eq!(&got_c[y * w * 16..(y + 1) * w * 16], &raws[0][y * rb..y * rb + w * 16]);
            }
        }

        // 2. `motion_sensitivity == 0.0` -> NoError exact copy.
        {
            reset_mock_host();
            mock_state().lock().unwrap().motion_sensitivity = 0.0;
            let mut rig = MockRig::new();
            let (hs, _) = make_mock_frames(w, h, rb, false, false);
            let mut out = hs[0];
            assert_eq!(rig.render(&hs[..1], &mut out), sys::suiteError_NoError);
            assert_eq!(out, hs[0]);
        }

        // 3. Blur enabled with all-duplicate neighbours (no usable neighbour on either side) ->
        //    `tblur_core` outputs exact unblurred copy matching CPU bit-for-bit and returns NoError.
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, raws) = make_mock_frames(w, h, rb, false, true);
            let cpu_ref =
                run_cpu_bgra32f(&raws[0], None, None, w, h, rb, &core::Params::default());
            let mut out = hs[0];
            assert_eq!(rig.render(&hs, &mut out), sys::suiteError_NoError);
            assert_ne!(out, hs[0]);
            let got = read_mock_ppix_bytes(out);
            for y in 0..h {
                assert_eq!(
                    &got[y * w * 16..(y + 1) * w * 16],
                    &cpu_ref[y * rb..y * rb + w * 16]
                );
            }
        }

        // 4. Blur enabled with non-positive `ticks_per_frame` -> returns Fail (hand off to CPU).
        {
            reset_mock_host();
            let mut rig = MockRig::new();
            let (hs, _) = make_mock_frames(w, h, rb, false, false);
            let mut out = hs[0];
            assert_eq!(rig.render_at_seq_time(0, 0, &hs[..1], &mut out), sys::suiteError_Fail);
            assert_eq!(out, hs[0]);
        }
    }
}
