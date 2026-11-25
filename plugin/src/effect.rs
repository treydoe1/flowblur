//! Adobe-facing glue: parameters, global setup, frame checkout and the render path.
//!
//! Everything in the render path degrades to copying the input to the output. No error is ever
//! returned to the host from a render (that would show an error dialog) and panics are caught
//! twice: around our render (-> pass-through frame) and, as a last resort, by the bindings'
//! catch_unwind around EffectMain (enabled in release via the `catch_panics` cfg in build.rs).

use after_effects as ae;
use ae::sys;
use flowblur_core as core;
use std::cell::Cell;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Concurrency counters touched ONLY when the debug flag file exists (zero cost otherwise).
static IN_FLIGHT_RENDERS: AtomicUsize = AtomicUsize::new(0);
static CO_IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

struct AtomicCountGuard(&'static AtomicUsize);

impl AtomicCountGuard {
    #[inline]
    fn enter(c: &'static AtomicUsize) -> (Self, usize) {
        let now = c.fetch_add(1, Ordering::Relaxed) + 1;
        (AtomicCountGuard(c), now)
    }
}

impl Drop for AtomicCountGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

use crate::diag;
use crate::VERSION;
use crate::instance_registry;
use crate::neighbors::{self, Advance, Class, ClipTiming, MAX_K, Outcome, SourceRate, SourceState, Verdict};
use crate::source_cache;
use crate::pixels::{self, Alpha, ImageDesc, Layout};

/// Effect parameters. The input layer is parameter index 0 and is not part of this enum.
///
/// DISK IDs: each parameter gets an explicit, frozen id (`disk_id`) instead of the bindings'
/// default (a hash of the variant name), so renaming a variant can never break saved projects.
/// Never reuse or renumber an id; append new parameters at the end with a new id.
#[derive(Eq, PartialEq, Hash, Clone, Copy, Debug)]
pub enum FbParam {
    BlurAmount,
    MotionSensitivity,
    MaxBlur,
    Quality,
    LinearLight,
    View,
    /// Premiere only: pick neighbours at source-frame spacing (see `neighbors`). Off = legacy.
    UseSourceRate,
}

impl FbParam {
    const fn disk_id(self) -> i32 {
        match self {
            FbParam::BlurAmount => 1,
            FbParam::MotionSensitivity => 2,
            FbParam::MaxBlur => 3,
            FbParam::Quality => 4,
            FbParam::LinearLight => 5,
            FbParam::View => 6,
            FbParam::UseSourceRate => 7,
        }
    }
}

#[derive(Default)]
pub struct Plugin {}

ae::define_effect!(Plugin, (), FbParam);

fn add_param<'a>(params: &mut ae::Parameters<FbParam>, p: FbParam, name: &str, def: impl Into<ae::Param<'a>>) -> Result<(), ae::Error> {
    params.add_customized(p, name, def, |pd| {
        pd.set_id(p.disk_id());
        -1
    })
}

impl AdobePluginGlobal for Plugin {
    fn params_setup(&self, params: &mut ae::Parameters<FbParam>, _in_data: ae::InData, _: ae::OutData) -> Result<(), ae::Error> {
        add_param(params, FbParam::BlurAmount, "Blur Amount", ae::FloatSliderDef::setup(|f| {
            f.set_valid_min(0.0);
            f.set_valid_max(4.0);
            f.set_slider_min(0.0);
            f.set_slider_max(2.0);
            f.set_default(0.5);
            f.set_value(0.5);
            f.set_precision(2);
        }))?;
        add_param(params, FbParam::MotionSensitivity, "Motion Sensitivity", ae::FloatSliderDef::setup(|f| {
            f.set_valid_min(0.0);
            f.set_valid_max(100.0);
            f.set_slider_min(0.0);
            f.set_slider_max(100.0);
            f.set_default(70.0);
            f.set_value(70.0);
            f.set_precision(1);
        }))?;
        add_param(params, FbParam::MaxBlur, "Max Blur (%)", ae::FloatSliderDef::setup(|f| {
            f.set_valid_min(0.0);
            f.set_valid_max(50.0);
            f.set_slider_min(0.0);
            f.set_slider_max(50.0);
            f.set_default(15.0);
            f.set_value(15.0);
            f.set_precision(1);
        }))?;
        add_param(params, FbParam::Quality, "Quality", ae::PopupDef::setup(|f| {
            f.set_options(&["Draft", "Normal", "High"]);
            f.set_default(2);
            f.set_value(2);
        }))?;
        add_param(params, FbParam::LinearLight, "Blend in Linear Light", ae::CheckBoxDef::setup(|f| {
            f.set_default(true);
            f.set_value(true);
        }))?;
        add_param(params, FbParam::View, "View", ae::PopupDef::setup(|f| {
            f.set_options(&["Output", "Motion Vectors"]);
            f.set_default(1);
            f.set_value(1);
        }))?;
        add_param(params, FbParam::UseSourceRate, "Use Source Frame Rate", ae::CheckBoxDef::setup(|f| {
            f.set_default(true);
            f.set_value(true);
        }))?;
        Ok(())
    }

    fn handle_command(&self, cmd: ae::Command, in_data: ae::InData, mut out_data: ae::OutData, params: &mut ae::Parameters<FbParam>) -> Result<(), ae::Error> {
        match cmd {
            ae::Command::About => {
                out_data.set_return_msg(&format!("FlowBlur {VERSION}\rOptical-flow motion blur."));
            }
            ae::Command::GlobalSetup => {
                // Out-flags are written by define_effect! from the same PIPL_OUTFLAGS* values that
                // are compiled into the PiPL, so the two always agree.
                let summary = if in_data.is_premiere() {
                    // Failures here are not fatal: Premiere then simply keeps sending 8-bit BGRA.
                    catch_unwind(AssertUnwindSafe(|| premiere_global_setup(&in_data))).unwrap_or_else(|_| "panic".to_string())
                } else {
                    String::new()
                };
                if diag::enabled() {
                    let host = if in_data.is_premiere() { "premiere" } else { "ae" };
                    diag::append(&format!("event=global_setup version={VERSION} host={host} {summary}"));
                }
            }
            ae::Command::SequenceSetup => {
                record_pf_instance_synthetic(&in_data, "sequence_setup");
            }
            ae::Command::SequenceResetup => {
                record_pf_instance_synthetic(&in_data, "sequence_resetup");
            }
            ae::Command::SequenceFlatten => {
                record_pf_instance_synthetic(&in_data, "sequence_flatten");
            }
            ae::Command::SequenceSetdown => {
                remove_pf_instance_synthetic(&in_data);
            }
            ae::Command::FrameSetup { .. } => {
                record_pf_instance_synthetic(&in_data, "frame_setup");
            }
            ae::Command::UpdateParamsUi => {
                record_pf_instance_synthetic(&in_data, "update_params_ui");
            }
            ae::Command::UserChangedParam { .. } => {
                record_pf_instance_synthetic(&in_data, "user_changed_param");
            }
            ae::Command::Render { in_layer, mut out_layer } => {
                let log = diag::enabled();
                let t0 = log.then(Instant::now);
                let mut dg = Diag { enabled: log, ..Diag::default() };
                let _in_flight = if log {
                    let (g, n) = AtomicCountGuard::enter(&IN_FLIGHT_RENDERS);
                    dg.in_flight = n;
                    Some(g)
                } else {
                    None
                };
                let rendered = catch_unwind(AssertUnwindSafe(|| {
                    let s = read_params(params).map_err(Fallback::from)?;
                    let req = source_request(&in_data, s.use_source_rate, &mut dg);
                    render(&in_data, &s.core, req, &in_layer, &mut out_layer, &mut dg)
                }));
                let result = match &rendered {
                    Ok(Ok(())) => "ok",
                    Ok(Err(_)) => "passthrough",
                    Err(_) => "passthrough(panic)",
                };
                if !matches!(rendered, Ok(Ok(()))) {
                    passthrough(&in_data, &in_layer, &mut out_layer);
                }
                if log {
                    if diag::seghash_enabled() {
                        dg.seg = Some(probe_video_segment_hash(&in_data, dg.host));
                    }
                    let wall_ms = t0.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1000.0);
                    let _ = catch_unwind(AssertUnwindSafe(|| diag::append(&dg.line(&in_data, result, wall_ms))));
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Premiere: register BGRA 32f and BGRA 8u (in order of preference, as Adobe's SDK_Noise does),
/// and opt in to checked-out frames matching the render format (ae: ppro/bigger-differences).
/// Returns a `key=value` summary of each call's result for the diagnostics log.
fn premiere_global_setup(in_data: &ae::InData) -> String {
    fn res<T>(r: &Result<T, ae::Error>) -> String {
        match r {
            Ok(_) => "ok".into(),
            Err(e) => format!("err({e:?})"),
        }
    }
    let mut s = String::new();
    let effect = in_data.effect_ref();
    if effect.is_null() {
        return "effect_ref=null".to_string();
    }
    match ae::pf::suites::PixelFormat::new() {
        Ok(suite) => {
            let clear = suite.clear_supported_pixel_formats(effect);
            s += &format!("clear_formats={}", res(&clear));
            if clear.is_ok() {
                s += &format!(" add_bgra32f={}", res(&suite.add_supported_pixel_format(effect, ae::pr::PixelFormat::Bgra4444_32f)));
                s += &format!(" add_bgra8u={}", res(&suite.add_supported_pixel_format(effect, ae::pr::PixelFormat::Bgra4444_8u)));
            }
        }
        Err(e) => s += &format!("pixel_format_suite=err({e:?})"),
    }
    match ae::pf::suites::Utility::new() {
        Ok(util) => s += &format!(" checked_out_match_render_format={}", res(&util.effect_wants_checked_out_frames_to_match_render_pixel_format(effect))),
        Err(e) => s += &format!(" utility_suite=err({e:?})"),
    }
    s
}

// ───────────────────────────── host clip timing (Premiere) ─────────────────────────────

/// Runs a host query so that neither an error nor a panic in the bindings can escape.
fn guarded<R>(f: impl FnOnce() -> Option<R>) -> Option<R> {
    catch_unwind(AssertUnwindSafe(f)).ok().flatten()
}

/// Host timing values plus extras that are only logged or only used as cache key.
#[derive(Clone, Copy, Debug, Default)]
struct HostTiming {
    /// Reported rates. `media_ticks` returned the SEQUENCE rate in Premiere Pro 2027, so none of
    /// these computes D; they are logged and part of the cache key (`speed` also feeds `a`
    /// when local_time_step is 0).
    timing: ClipTiming,
    /// `Utility::original_clip_frame_rate`. Since 1.3 the authority for D (`D_orig`) and for the
    /// gate; seen correct in Premiere Pro 2027 for real media (119.88 fps clip -> 2118916800 on
    /// 119.88 and 23.976 sequences), and equal to the sequence rate on adjustment layers.
    orig_ticks: Option<i64>,
    /// `Utility::is_track_item_effect_applied_to_synthetic` (None when the call failed; counts as
    /// not synthetic). UNVERIFIED in host which items report true (adjustment layers?).
    synthetic: Option<bool>,
    /// `Utility::filter_instance_id` (None when the call fails); cache key and log only.
    instance_id: Option<i32>,
    timeline_id: Option<i32>,
    /// `Utility::sequence_time`: documented in the bindings as "the timebase of the sequence",
    /// but the SDK parameter is named outSequenceTime, so it is logged only, never used.
    util_sequence_time: Option<i64>,
}

/// Queries Premiere's PF Utility suite (and the Sequence Info suite) for the clip's timing.
/// Every call is independent and may fail.
fn query_host_timing(in_data: &ae::InData) -> HostTiming {
    let mut h = HostTiming::default();
    let effect_ref = in_data.effect_ref();
    if effect_ref.is_null() {
        return h;
    }
    let Some(util) = guarded(|| ae::pf::suites::Utility::new().ok()) else { return h };
    h.timing.media_ticks = guarded(|| util.media_frame_rate(effect_ref).ok());
    h.orig_ticks = guarded(|| util.original_clip_frame_rate(effect_ref).ok());
    h.synthetic = guarded(|| util.is_track_item_effect_applied_to_synthetic(effect_ref).ok());
    h.instance_id = guarded(|| util.filter_instance_id(effect_ref).ok());
    h.timing.speed = guarded(|| util.clip_speed(effect_ref).ok());
    h.util_sequence_time = guarded(|| util.sequence_time(effect_ref).ok());
    h.timeline_id = guarded(|| util.containing_timeline_id(effect_ref).ok());
    if let Some(tl) = h.timeline_id {
        h.timing.seq_ticks = guarded(|| sequence_ticks_per_frame(in_data, tl));
    }
    h
}

/// Leading part of PrSDKSequenceInfoSuite (premiere-sys bindings: `#[repr(C, packed)]`, function
/// pointers in declaration order). GetFrameRate is the third entry since suite version 2
/// (CS5.5); later versions only append entries, so this prefix is valid for versions 2..=9.
#[repr(C, packed)]
struct SequenceInfoPrefix {
    _get_frame_rect: *const std::ffi::c_void,
    _get_pixel_aspect_ratio: *const std::ffi::c_void,
    get_frame_rate: Option<unsafe extern "C" fn(timeline: sys::PrTimelineID, out_ticks_per_frame: *mut sys::PrTime) -> i32>,
}

/// PrTime ticks per sequence frame via the "MediaCore Sequence Info Suite", acquired directly
/// through the PICA basic suite in PF_InData (the same SPBasicSuite through which the bindings
/// acquire the PF Utility suite in Premiere). The `premiere` crate's wrapper is not used: it
/// would need its own PICA registration and the premiere-sys crate, for one function call.
fn sequence_ticks_per_frame(in_data: &ae::InData, timeline: i32) -> Option<i64> {
    const NAME: &std::ffi::CStr = c"MediaCore Sequence Info Suite";
    let basic = in_data.pica_basic_suite_ptr();
    if basic.is_null() {
        return None;
    }
    let (acquire, release) = unsafe { ((*basic).AcquireSuite?, (*basic).ReleaseSuite?) };
    for version in (2..=9).rev() {
        let mut p: *const std::ffi::c_void = std::ptr::null();
        if unsafe { acquire(NAME.as_ptr(), version, &mut p) } != 0 {
            continue;
        }
        let mut ticks: sys::PrTime = 0;
        let mut err = -1;
        if !p.is_null() {
            let s = p as *const SequenceInfoPrefix;
            let f = unsafe { std::ptr::read_unaligned(std::ptr::addr_of!((*s).get_frame_rate)) };
            if let Some(f) = f {
                err = unsafe { f(timeline, &mut ticks) };
            }
        }
        unsafe { release(NAME.as_ptr(), version) };
        return (err == 0 && ticks > 0).then_some(ticks);
    }
    None
}

/// Premiere `prPluginID` (`mGUID: [c_char; 37]`).
#[repr(C)]
#[derive(Clone, Copy)]
struct PrPluginID {
    m_guid: [u8; 37],
}

/// Leading 22 function pointers of `PrSDKVideoSegmentSuite` (versions 4..=9, `#[repr(C, packed)]`,
/// exact declaration order from Premiere SDK `PrSDKVideoSegmentSuite.h` / `premiere-sys`).
#[repr(C, packed)]
struct VideoSegmentPrefix {
    acquire_video_segments_id: Option<unsafe extern "C" fn(sys::PrTimelineID, *mut i32) -> i32>,
    _acquire_with_previews: *const std::ffi::c_void,
    _acquire_with_opaque_previews: *const std::ffi::c_void,
    release_video_segments_id: Option<unsafe extern "C" fn(i32) -> i32>,
    get_hash: Option<unsafe extern "C" fn(i32, *mut PrPluginID) -> i32>,
    _get_segment_count: *const std::ffi::c_void,
    _get_segment_info: *const std::ffi::c_void,
    _acquire_node_id: *const std::ffi::c_void,
    release_video_node_id: Option<unsafe extern "C" fn(i32) -> i32>,
    get_node_info: Option<unsafe extern "C" fn(i32, *mut std::ffi::c_char, *mut PrPluginID, *mut i32) -> i32>,
    _mid: [*const std::ffi::c_void; 11],
    acquire_node_for_time: Option<unsafe extern "C" fn(i32, sys::PrTime, *mut i32, *mut sys::PrTime) -> i32>,
}

#[derive(Clone, Debug, Default)]
struct SegHashDiag {
    seg_hash: Option<String>,
    node_hash: Option<String>,
    seg_us: u64,
    err: Option<String>,
}

/// Formats a `PrPluginID` in hex without whitespace. Handles both 16-byte binary GUIDs (zero-padded
/// tail) and 36-char NUL-terminated ASCII UUID strings; falls back to full 37-byte hex otherwise.
fn format_plugin_id(id: &PrPluginID) -> String {
    let b = &id.m_guid;
    if b[16..].iter().all(|&x| x == 0) {
        let mut s = String::with_capacity(32);
        for &byte in &b[..16] {
            use std::fmt::Write;
            let _ = write!(&mut s, "{byte:02x}");
        }
        s
    } else if b[36] == 0 && b[..36].iter().all(|&x| x.is_ascii_hexdigit() || x == b'-') {
        b[..36].iter().filter(|&&x| x != b'-').map(|&x| (x as char).to_ascii_lowercase()).collect()
    } else {
        let mut s = String::with_capacity(74);
        for &byte in b {
            use std::fmt::Write;
            let _ = write!(&mut s, "{byte:02x}");
        }
        s
    }
}

/// Opt-in Premiere Video Segment suite probe behind `/tmp/flowblur-debug-seghash`.
/// Acquires the suite through PICA, queries segment hash and node hash at `util_sequence_time`,
/// and releases every handle and the suite on every exit path. Never panics.
fn probe_video_segment_hash(in_data: &ae::InData, host: Option<HostTiming>) -> SegHashDiag {
    let t0 = Instant::now();
    let mut out = SegHashDiag::default();
    let res = catch_unwind(AssertUnwindSafe(|| {
        if !in_data.is_premiere() {
            out.err = Some("not_premiere".into());
            return;
        }
        let Some(h) = host else {
            out.err = Some("no_host_timing".into());
            return;
        };
        let Some(timeline_id) = h.timeline_id else {
            out.err = Some("no_timeline_id".into());
            return;
        };
        let basic = in_data.pica_basic_suite_ptr();
        if basic.is_null() {
            out.err = Some("null_basic_suite".into());
            return;
        }
        let Some((acquire, release)) = (unsafe { (*basic).AcquireSuite.zip((*basic).ReleaseSuite) }) else {
            out.err = Some("null_basic_fns".into());
            return;
        };
        const NAME: &std::ffi::CStr = c"MediaCore Video Segment Suite";
        let mut ptr: *const std::ffi::c_void = std::ptr::null();
        let mut acquired_ver: i32 = 0;
        let mut last_err: i32 = -1;
        for version in (4..=9).rev() {
            let mut p: *const std::ffi::c_void = std::ptr::null();
            let err = unsafe { acquire(NAME.as_ptr(), version, &mut p) };
            if err == 0 && !p.is_null() {
                ptr = p;
                acquired_ver = version;
                break;
            }
            if err == 0 && p.is_null() {
                unsafe { release(NAME.as_ptr(), version) };
                last_err = -2;
            } else {
                last_err = err;
            }
        }
        if ptr.is_null() {
            out.err = Some(format!("acquire_suite:{last_err}"));
            return;
        }
        struct SuiteGuard {
            release: unsafe extern "C" fn(*const std::ffi::c_char, i32) -> i32,
            version: i32,
        }
        impl Drop for SuiteGuard {
            fn drop(&mut self) {
                unsafe {
                    let _ = (self.release)(NAME.as_ptr(), self.version);
                }
            }
        }
        let _suite_guard = SuiteGuard { release, version: acquired_ver };

        let s = ptr as *const VideoSegmentPrefix;
        let (acq_seg, rel_seg, get_hash, rel_node, get_node_info, acq_node) = unsafe {
            (
                std::ptr::read_unaligned(std::ptr::addr_of!((*s).acquire_video_segments_id)),
                std::ptr::read_unaligned(std::ptr::addr_of!((*s).release_video_segments_id)),
                std::ptr::read_unaligned(std::ptr::addr_of!((*s).get_hash)),
                std::ptr::read_unaligned(std::ptr::addr_of!((*s).release_video_node_id)),
                std::ptr::read_unaligned(std::ptr::addr_of!((*s).get_node_info)),
                std::ptr::read_unaligned(std::ptr::addr_of!((*s).acquire_node_for_time)),
            )
        };
        let (Some(acq_seg), Some(rel_seg), Some(get_hash), Some(rel_node), Some(get_node_info), Some(acq_node)) =
            (acq_seg, rel_seg, get_hash, rel_node, get_node_info, acq_node)
        else {
            out.err = Some("null_suite_fn".into());
            return;
        };

        let mut errs: Vec<String> = Vec::new();
        let mut seg_id: i32 = 0;
        let err = unsafe { acq_seg(timeline_id, &mut seg_id) };
        if err != 0 {
            out.err = Some(format!("acq_seg:{err}"));
            return;
        }
        struct SegGuard {
            rel: unsafe extern "C" fn(i32) -> i32,
            id: i32,
        }
        impl Drop for SegGuard {
            fn drop(&mut self) {
                unsafe {
                    let _ = (self.rel)(self.id);
                }
            }
        }
        let _seg_guard = SegGuard { rel: rel_seg, id: seg_id };

        let mut seg_hash = PrPluginID { m_guid: [0u8; 37] };
        let err = unsafe { get_hash(seg_id, &mut seg_hash) };
        if err == 0 {
            out.seg_hash = Some(format_plugin_id(&seg_hash));
        } else {
            errs.push(format!("get_hash:{err}"));
        }

        if let Some(seq_time) = h.util_sequence_time {
            let mut node_id: i32 = 0;
            let mut seg_off: sys::PrTime = 0;
            let err = unsafe { acq_node(seg_id, seq_time, &mut node_id, &mut seg_off) };
            if err != 0 {
                errs.push(format!("acq_node:{err}"));
            } else {
                struct NodeGuard {
                    rel: unsafe extern "C" fn(i32) -> i32,
                    id: i32,
                }
                impl Drop for NodeGuard {
                    fn drop(&mut self) {
                        unsafe {
                            let _ = (self.rel)(self.id);
                        }
                    }
                }
                let _node_guard = NodeGuard { rel: rel_node, id: node_id };

                let mut node_type = [0 as std::ffi::c_char; 256];
                let mut node_hash = PrPluginID { m_guid: [0u8; 37] };
                let mut flags: i32 = 0;
                let err = unsafe { get_node_info(node_id, node_type.as_mut_ptr(), &mut node_hash, &mut flags) };
                if err == 0 {
                    out.node_hash = Some(format_plugin_id(&node_hash));
                } else {
                    errs.push(format!("node_info:{err}"));
                }
            }
        } else {
            errs.push("no_seq_time".into());
        }

        if !errs.is_empty() {
            out.err = Some(errs.join(","));
        }
    }));
    if res.is_err() {
        out.err = Some("panic".into());
    }
    out.seg_us = t0.elapsed().as_micros() as u64;
    out
}

/// What the render may try. `host` is Premiere's clip timing (None in After Effects; also used
/// for the per-clip record when the source strategy is off); `allowed` is `Err(reason)` when the
/// source strategy is off before looking at the clip (host, parameter).
#[derive(Clone, Copy, Debug)]
struct SourceReq {
    host: Option<HostTiming>,
    allowed: Result<(), &'static str>,
}

/// Records `(instance_id, synthetic)` into the shared process-wide registry and logs when first
/// recorded (or if `synthetic` ever flips) while `/tmp/flowblur-debug` is active.
fn record_instance_synthetic_pair(instance_id: i32, synthetic: bool, cmd_name: &'static str) {
    let outcome = instance_registry::global().record(instance_id, synthetic, cmd_name);
    if (outcome.is_first || outcome.changed) && diag::enabled() {
        diag::append(&format!(
            "event=instance_synthetic_record path=pf cmd={cmd_name} instance_id={instance_id} \
             synthetic={synthetic} first_cmd={} changed={}",
            outcome.first_cmd, outcome.changed,
        ));
    }
}

/// Queries Premiere's PF Utility suite during setup/lifecycle commands to populate the shared
/// `instance_id -> synthetic` registry as early as the host exposes both values. Returns early
/// when `effect_ref` is null; any failing suite call or panic is silently ignored.
fn record_pf_instance_synthetic(in_data: &ae::InData, cmd_name: &'static str) {
    if !in_data.is_premiere() {
        return;
    }
    let effect_ref = in_data.effect_ref();
    if effect_ref.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(util) = guarded(|| ae::pf::suites::Utility::new().ok()) else {
            return;
        };
        let instance_id = guarded(|| util.filter_instance_id(effect_ref).ok());
        let synthetic = guarded(|| util.is_track_item_effect_applied_to_synthetic(effect_ref).ok());
        if let (Some(id), Some(syn)) = (instance_id, synthetic) {
            record_instance_synthetic_pair(id, syn, cmd_name);
        }
    }));
}

/// Removes the effect instance from the shared synthetic registry on `SequenceSetdown` so stale
/// entries cannot outlive their instances if Premiere later reuses an instance ID.
fn remove_pf_instance_synthetic(in_data: &ae::InData) {
    if !in_data.is_premiere() {
        return;
    }
    let effect_ref = in_data.effect_ref();
    if effect_ref.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let Some(util) = guarded(|| ae::pf::suites::Utility::new().ok()) else {
            return;
        };
        if let Some(id) = guarded(|| util.filter_instance_id(effect_ref).ok()) {
            let removed = instance_registry::global().remove(id);
            if removed && diag::enabled() {
                diag::append(&format!(
                    "event=instance_synthetic_remove path=pf cmd=sequence_setdown instance_id={id}"
                ));
            }
        }
    }));
}

fn source_request(in_data: &ae::InData, enabled: bool, dg: &mut Diag) -> SourceReq {
    if !in_data.is_premiere() {
        return SourceReq { host: None, allowed: Err("after effects host") };
    }
    // Queried even with the checkbox off: the instance id keys the current-checkout record.
    let h = query_host_timing(in_data);
    if let (Some(id), Some(syn)) = (h.instance_id, h.synthetic) {
        record_instance_synthetic_pair(id, syn, "render");
    }
    dg.host = Some(h);
    SourceReq { host: Some(h), allowed: if enabled { Ok(()) } else { Err("disabled by parameter") } }
}

/// Gate for the source strategy (`neighbors::source_gate`): host and parameter first, then the
/// clip's original rate and synthetic flag. Decided without any checkout.
fn source_gate(req: &SourceReq, step: i32, scale: u32, dg: &mut Diag) -> Result<(HostTiming, f64), &'static str> {
    let r = (|| {
        req.allowed?;
        let h = req.host.ok_or("no host timing")?;
        neighbors::source_gate(neighbors::ticks_to_units(h.orig_ticks, scale), step, h.synthetic).map(|d| (h, d))
    })();
    dg.gate = Some(r.map(|_| ()));
    r
}

// ───────────────────────────── diagnostics record ─────────────────────────────

#[derive(Clone, Copy, Debug)]
struct ProbeLog {
    /// "prev" / "next" (source ladder), "confirm" / "edge" (extra ladder probe), "verify"
    /// (scaled current time for the ladder), "lprev" / "lnext" (legacy).
    kind: &'static str,
    time: i64,
    m: u32,
    outcome: &'static str,
}

/// Per-render checkout & pixel-conversion timings collected only when debug logging is enabled.
#[derive(Default)]
struct CoMetrics {
    co_ms: Cell<f64>,
    co_max_ms: Cell<f64>,
    pix_ms: Cell<f64>,
    co_in_flight_max: Cell<usize>,
}

impl CoMetrics {
    #[inline]
    fn record_co(&self, ms: f64) {
        self.co_ms.set(self.co_ms.get() + ms);
        if ms > self.co_max_ms.get() {
            self.co_max_ms.set(ms);
        }
    }
    #[inline]
    fn record_pix(&self, ms: f64) {
        self.pix_ms.set(self.pix_ms.get() + ms);
    }
    #[inline]
    fn record_co_in_flight(&self, n: usize) {
        if n > self.co_in_flight_max.get() {
            self.co_in_flight_max.set(n);
        }
    }
}

/// Everything one render learned, for the diagnostics line. Strings are only built when
/// `enabled` (the log is on); the rest is cheap and always filled (tests inspect it).
#[derive(Default)]
struct Diag {
    enabled: bool,
    host: Option<HostTiming>,
    in_size: Option<(usize, usize)>,
    in_fmt: Option<String>,
    cur_fmt: Option<String>,
    /// Current-time checkout: "skipped" / "match" / "mismatch" / "failed".
    cur_outcome: Option<&'static str>,
    /// Why the current-time checkout failed ("missing" / "unusable").
    cur_fail: Option<&'static str>,
    /// D1/D2 from the host-reported rates (logged only).
    rate: Option<SourceRate>,
    /// `Utility::original_clip_frame_rate` in native units.
    d_orig: Option<f64>,
    /// Gate decision: Ok = source strategy attempted, Err(reason) = legacy without probing.
    gate: Option<Result<(), &'static str>>,
    /// Source state of the key at the start of the render and after it.
    state: Option<SourceState>,
    state_after: Option<SourceState>,
    /// Classification of this render's ladder.
    class: Option<Class>,
    dups: Option<(u32, u32)>,
    /// D used (native units; always D_orig).
    d: Option<f64>,
    adv: Option<Advance>,
    mode: &'static str,
    legacy_reason: Option<&'static str>,
    /// Legacy frame-distance rule applied (only for `NOT_ABOVE_SEQUENCE` legacy renders).
    legacy_fd: Option<neighbors::LegacyFd>,
    /// `in_data.local_time_step` of this render.
    local_time_step: i32,
    m: u32,
    probes: Vec<ProbeLog>,
    /// Host checkouts made by this render.
    checkouts: u32,
    fd: (Option<f32>, Option<f32>),
    stats: Option<core::RenderStats>,
    /// Debug-only performance & concurrency fields (populated only when `enabled`).
    co_ms: f64,
    co_max_ms: f64,
    pix_ms: f64,
    in_flight: usize,
    co_in_flight_max: usize,
    computing: Option<usize>,
    budget: Option<usize>,
    /// Populated only when `/tmp/flowblur-debug-seghash` also exists.
    seg: Option<SegHashDiag>,
}

impl Diag {
    fn line(&self, in_data: &ae::InData, result: &str, wall_ms: f64) -> String {
        use diag::{opt, optf};
        let host = if in_data.is_premiere() { "premiere" } else { "ae" };
        let h = self.host.unwrap_or_default();
        let probes: Vec<String> = self.probes.iter().map(|p| format!("{}@{}x{}:{}", p.kind, p.time, p.m, p.outcome)).collect();
        let st = self.stats;
        let state = |s: Option<SourceState>| s.map_or("n/a".into(), |s| format!("{}/{}", s.verdict.name(), s.strikes));
        let mut s = format!(
            "event=render host={host} result={result} current_time={} time_step={} time_scale={} local_time_step={} in_size={} in_fmt={} cur_fmt={} cur_checkout={} cur_fail={} checkouts={} media_ticks={} seq_ticks={} orig_ticks={} synthetic={} instance_id={} timeline_id={} util_sequence_time={} clip_speed={} d1={} d2={} d_orig={} gate={} gate_reason=\"{}\" cache={} strikes={} cache_after={} class={} dups={} d={} d_src=\"{}\" a={} a_local={} a_speed={} mode={} legacy_reason=\"{}\" legacy_fd={} m={} probes=[{}] fd_prev={} fd_next={} flow_ms={} blur_ms={} used_prev={} used_next={} max_velocity_px={} wall_ms={:.1} co_ms={:.1} co_max_ms={:.1} pix_ms={:.1} in_flight={} co_in_flight_max={} computing={} budget={}",
            in_data.current_time(),
            in_data.time_step(),
            in_data.time_scale(),
            in_data.local_time_step(),
            self.in_size.map_or("n/a".into(), |(w, h)| format!("{w}x{h}")),
            self.in_fmt.as_deref().unwrap_or("n/a"),
            self.cur_fmt.as_deref().unwrap_or("n/a"),
            self.cur_outcome.unwrap_or("n/a"),
            self.cur_fail.unwrap_or("n/a"),
            self.checkouts,
            opt(h.timing.media_ticks),
            opt(h.timing.seq_ticks),
            opt(h.orig_ticks),
            opt(h.synthetic),
            opt(h.instance_id),
            opt(h.timeline_id),
            opt(h.util_sequence_time),
            opt(h.timing.speed),
            optf(self.rate.and_then(|r| r.d1), 4),
            optf(self.rate.and_then(|r| r.d2), 4),
            optf(self.d_orig, 4),
            self.gate.map_or("n/a", |g| if g.is_ok() { "source" } else { "legacy" }),
            self.gate.and_then(|g| g.err()).unwrap_or(""),
            self.state.map_or("n/a", |s| s.verdict.name()),
            opt(self.state.map(|s| s.strikes)),
            state(self.state_after),
            self.class.map_or("n/a", |c| c.name()),
            self.dups.map_or("n/a".into(), |(p, n)| format!("{p}/{n}")),
            optf(self.d, 4),
            if self.d.is_some() { "orig" } else { "n/a" },
            optf(self.adv.map(|a| a.a), 4),
            optf(self.adv.and_then(|a| a.a_local), 4),
            optf(self.adv.map(|a| a.a_speed), 4),
            if self.mode.is_empty() { "n/a" } else { self.mode },
            self.legacy_reason.unwrap_or(""),
            self.legacy_fd.map_or("n/a", |r| r.name()),
            self.m,
            probes.join(","),
            opt(self.fd.0),
            opt(self.fd.1),
            optf(st.map(|s| s.flow_ms as f64), 1),
            optf(st.map(|s| s.blur_ms as f64), 1),
            opt(st.map(|s| s.used_prev)),
            opt(st.map(|s| s.used_next)),
            optf(st.map(|s| s.max_velocity_px as f64), 2),
            wall_ms,
            self.co_ms,
            self.co_max_ms,
            self.pix_ms,
            self.in_flight,
            self.co_in_flight_max,
            opt(self.computing),
            opt(self.budget),
        );
        if let Some(seg) = &self.seg {
            use std::fmt::Write;
            let _ = write!(
                &mut s,
                " seg_hash={} node_hash={} seg_us={} seg_err={}",
                seg.seg_hash.as_deref().unwrap_or("n/a"),
                seg.node_hash.as_deref().unwrap_or("n/a"),
                seg.seg_us,
                seg.err.as_deref().unwrap_or("none"),
            );
        }
        s
    }
}

// ───────────────────────────── raw world access ─────────────────────────────

/// Raw view of a PF_EffectWorld's geometry. We read the struct ourselves instead of using
/// `Layer::buffer()` / `Layer::bit_depth()`, which assert / unwrap and would panic on a null
/// buffer or unknown world flags.
#[derive(Clone, Copy)]
struct WorldGeom {
    data: *mut u8,
    width: usize,
    height: usize,
    rowbytes: isize,
    world_flags: u32,
}

fn geom_of(world: *const sys::PF_EffectWorld) -> Option<WorldGeom> {
    if world.is_null() {
        return None;
    }
    let w = unsafe { &*world };
    if w.data.is_null() || w.width <= 0 || w.height <= 0 || w.rowbytes == 0 {
        return None;
    }
    Some(WorldGeom {
        data: w.data as *mut u8,
        width: w.width as usize,
        height: w.height as usize,
        rowbytes: w.rowbytes as isize,
        world_flags: w.world_flags as u32,
    })
}

impl WorldGeom {
    /// Lowest-addressed byte of the buffer (row height-1 when rowbytes is negative).
    fn low(&self) -> *mut u8 {
        if self.rowbytes < 0 {
            unsafe { self.data.offset(self.rowbytes * (self.height as isize - 1)) }
        } else {
            self.data
        }
    }
    /// Bytes spanned, counting only `last_row` bytes for the final row: the host does not
    /// guarantee the padding after the last row exists, so we never touch it.
    fn len(&self, last_row: usize) -> usize {
        (self.height - 1) * self.rowbytes.unsigned_abs() + last_row
    }
    unsafe fn bytes<'a>(&self, layout: Layout) -> &'a [u8] {
        unsafe { std::slice::from_raw_parts(self.low(), self.len(self.width * layout.bytes_per_pixel())) }
    }
    unsafe fn bytes_mut<'a>(&self, layout: Layout) -> &'a mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.low(), self.len(self.width * layout.bytes_per_pixel())) }
    }
}

/// Decides the layout and alpha convention of a world. `None` = a format we do not handle.
///
/// Premiere (appl_id 'PrMr'): ask the PF Pixel Format suite for every world separately, since a
/// checked-out frame need not share the render format (8-bit ARGB unless the opt-in worked).
/// We registered only BGRA_4444_32f and BGRA_4444_8u; anything else is unsupported.
/// After Effects: no SmartFX, so only 8 or 16 bpc reach PF_Cmd_RENDER; PF_WorldFlag_DEEP marks
/// 16 bpc (ae: effect-basics/PF_EffectWorld). A float world would carry RESERVED1; we reject it.
fn classify(in_data: &ae::InData, layer: &ae::Layer, g: &WorldGeom) -> Option<(Layout, Alpha)> {
    if in_data.is_premiere() {
        // `pr_pixel_format` converts the raw value with a panicking `From`; contain that.
        let fmt = catch_unwind(AssertUnwindSafe(|| layer.pr_pixel_format())).ok()?.ok()?;
        match fmt {
            ae::pr::PixelFormat::Bgra4444_8u => Some((Layout::Bgra8, Alpha::Straight)),
            ae::pr::PixelFormat::Bgra4444_32f => Some((Layout::Bgra32f, Alpha::Straight)),
            // Checked-out frames when the "match render format" opt-in did not take: the docs
            // say "8-bit ARGB" (ae: ppro/bigger-differences). ASSUMED: reported as ARGB_4444_8u,
            // straight alpha like Premiere's BGRA. Only ever used for analysis luma, never output.
            // CHECK IN HOST: if reported as anything else, neighbours are treated as absent.
            ae::pr::PixelFormat::Argb4444_8u => Some((Layout::Argb8, Alpha::Straight)),
            _ => None,
        }
    } else {
        let deep = g.world_flags & sys::PF_WorldFlag_DEEP != 0;
        let float = g.world_flags & sys::PF_WorldFlag_RESERVED1 != 0;
        match (deep, float) {
            (false, false) => Some((Layout::Argb8, Alpha::Premultiplied)),
            (true, false) => Some((Layout::Argb16, Alpha::Premultiplied)),
            _ => None,
        }
    }
}

/// Decodes a world into packed premultiplied RGBA f32, luma plane, and max alpha in one pass.
fn decode_with_luma(in_data: &ae::InData, layer: &ae::Layer) -> Option<(ImageDesc, Vec<f32>, Vec<f32>, f32)> {
    let g = geom_of(layer.as_ptr())?;
    let (layout, alpha) = classify(in_data, layer, &g)?;
    let desc = ImageDesc { width: g.width, height: g.height, rowbytes: g.rowbytes, layout, alpha };
    let bytes = unsafe { g.bytes(layout) };
    let mut rgba = Vec::new();
    let (luma, max_a) = pixels::read_rgba_and_luma(bytes, &desc, &mut rgba).ok()?;
    Some((desc, rgba, luma, max_a))
}

// ───────────────────────────── checkout ─────────────────────────────

/// One PF_CHECKOUT_PARAM of the input layer (param index 0), checked back in on drop, on every
/// path, including when the checkout itself reported an error (Adobe's samples likewise call
/// PF_CHECKIN_PARAM unconditionally via ERR2). We call the interact callbacks directly rather than
/// `ae::ParamDef::checkout` (raw-index API) or `Parameters::checkout_at` (enum API, which cannot
/// name index 0) because the bindings' drop does `checkin_param(..).unwrap()`, i.e. it would
/// panic inside Drop if the host returned an error from check-in.
struct InputCheckout<'a> {
    in_data: *const sys::PF_InData,
    def: Box<sys::PF_ParamDef>,
    metrics: Option<&'a CoMetrics>,
}

impl<'a> InputCheckout<'a> {
    fn new(in_data: &ae::InData, time: i32, step: i32, scale: u32, metrics: Option<&'a CoMetrics>) -> Option<Self> {
        let raw = in_data.as_ptr();
        if raw.is_null() {
            return None;
        }
        let inter = unsafe { &(*raw).inter };
        let (checkout, _checkin) = (inter.checkout_param?, inter.checkin_param?);
        let mut co = InputCheckout { in_data: raw, def: Box::new(unsafe { std::mem::zeroed() }), metrics };
        let err = {
            let _pause = core::par::pause_render();
            if let Some(m) = metrics {
                let (_co_guard, n) = AtomicCountGuard::enter(&CO_IN_FLIGHT);
                m.record_co_in_flight(n);
                let t0 = Instant::now();
                let err = unsafe { checkout((*raw).effect_ref, 0, time, step, scale, &mut *co.def) };
                m.record_co(t0.elapsed().as_secs_f64() * 1000.0);
                err
            } else {
                unsafe { checkout((*raw).effect_ref, 0, time, step, scale, &mut *co.def) }
            }
        };
        if err != 0 {
            return None; // `co` drops here -> checked in.
        }
        Some(co)
    }

    fn layer(&self, in_data: &ae::InData) -> Option<ae::Layer> {
        let ld = unsafe { &self.def.u.ld } as *const sys::PF_LayerDef as *mut sys::PF_LayerDef;
        if unsafe { (*ld).data.is_null() } {
            return None; // no frame at this time (e.g. beyond the clip)
        }
        Some(ae::Layer::from_raw(ld, *in_data, None))
    }
}

impl Drop for InputCheckout<'_> {
    fn drop(&mut self) {
        unsafe {
            if let Some(checkin) = (*self.in_data).inter.checkin_param {
                let _pause = core::par::pause_render();
                if let Some(m) = self.metrics {
                    let t0 = Instant::now();
                    let _ = checkin((*self.in_data).effect_ref, &mut *self.def);
                    m.record_co(t0.elapsed().as_secs_f64() * 1000.0);
                } else {
                    let _ = checkin((*self.in_data).effect_ref, &mut *self.def);
                }
            }
        }
    }
}

/// Human-readable pixel format of a world for the diagnostics log: Premiere's PrPixelFormat
/// (name and raw value) or the AE world depth.
fn format_name(in_data: &ae::InData, layer: &ae::Layer, g: &WorldGeom) -> String {
    if in_data.is_premiere() {
        match catch_unwind(AssertUnwindSafe(|| layer.pr_pixel_format())) {
            Ok(Ok(f)) => {
                let raw: sys::PrPixelFormat = f.into();
                format!("{f:?}/0x{raw:08x}")
            }
            Ok(Err(e)) => format!("err({e:?})"),
            Err(_) => "unknown(unconvertible)".into(),
        }
    } else {
        let deep = g.world_flags & sys::PF_WorldFlag_DEEP != 0;
        let float = g.world_flags & sys::PF_WorldFlag_RESERVED1 != 0;
        match (deep, float) {
            (false, false) => "ae8".into(),
            (true, false) => "ae16".into(),
            _ => format!("ae_flags_0x{:x}", g.world_flags),
        }
    }
}

/// Checks out the input at `time` (in units of `scale`, with `step` as the frame duration) and
/// returns its luma if the frame exists, matches `want` in size, is in a format we decode, and is
/// not fully transparent while the current frame is not (some hosts return an empty world past
/// the media end). Checked back in before returning. `Err(Missing)`: no frame (host error or
/// null data); `Err(Unusable)`: a frame we cannot use. When `fmt` is given, the frame's pixel
/// format name is stored there.
#[allow(clippy::too_many_arguments)]
fn checkout_luma(
    in_data: &ae::InData,
    time: i32,
    step: i32,
    scale: u32,
    want: (usize, usize),
    cur_has_alpha: bool,
    fmt: Option<&mut Option<String>>,
    metrics: Option<&CoMetrics>,
) -> Result<Vec<f32>, Outcome> {
    let co = InputCheckout::new(in_data, time, step, scale, metrics).ok_or(Outcome::Missing)?;
    let layer = co.layer(in_data).ok_or(Outcome::Missing)?;
    let g = geom_of(layer.as_ptr()).ok_or(Outcome::Missing)?;
    if let Some(fmt) = fmt {
        *fmt = Some(format_name(in_data, &layer, &g));
    }
    // Reduced-resolution playback / field renders: a neighbour that does not match the current
    // frame's size is ignored rather than resampled.
    if (g.width, g.height) != want {
        return Err(Outcome::Unusable);
    }
    // Straight to luma: checked-out frames are only ever used for motion analysis.
    let (layout, alpha) = classify(in_data, &layer, &g).ok_or(Outcome::Unusable)?;
    let desc = ImageDesc { width: g.width, height: g.height, rowbytes: g.rowbytes, layout, alpha };
    let t_pix = metrics.map(|_| Instant::now());
    let (luma, max_a) = pixels::read_luma(unsafe { g.bytes(layout) }, &desc).map_err(|_| Outcome::Unusable)?;
    if let (Some(m), Some(t0)) = (metrics, t_pix) {
        m.record_pix(t0.elapsed().as_secs_f64() * 1000.0);
    }
    if cur_has_alpha && max_a <= 0.0 {
        return Err(Outcome::Unusable);
    }
    drop(layer);
    drop(co);
    Ok(luma)
}

// ───────────────────────────── render ─────────────────────────────

struct Settings {
    core: core::Params,
    use_source_rate: bool,
}

fn read_params(params: &ae::Parameters<FbParam>) -> Result<Settings, ae::Error> {
    // A failure to read the new checkbox (should not happen: hosts supply the default for
    // parameters missing from older projects) selects the legacy path, i.e. 1.0 behaviour.
    let use_source_rate = params.get(FbParam::UseSourceRate).and_then(|p| Ok(p.as_checkbox()?.value())).unwrap_or(false);
    Ok(Settings { core: read_core_params(params)?, use_source_rate })
}

fn read_core_params(params: &ae::Parameters<FbParam>) -> Result<core::Params, ae::Error> {
    let f = |p| -> Result<f32, ae::Error> { Ok(params.get(p)?.as_float_slider()?.value() as f32) };
    let mut out = core::Params::default();
    out.blur_amount = f(FbParam::BlurAmount)?.clamp(0.0, 4.0);
    out.motion_sensitivity = f(FbParam::MotionSensitivity)?.clamp(0.0, 100.0);
    out.max_blur_percent = f(FbParam::MaxBlur)?.clamp(0.0, 50.0);
    out.quality = match params.get(FbParam::Quality)?.as_popup()?.value() {
        1 => core::Quality::Draft,
        3 => core::Quality::High,
        _ => core::Quality::Normal,
    };
    out.linear_light = params.get(FbParam::LinearLight)?.as_checkbox()?.value();
    out.view = match params.get(FbParam::View)?.as_popup()?.value() {
        2 => core::View::MotionVectors,
        _ => core::View::Output,
    };
    Ok(out)
}

#[derive(Debug)]
struct Fallback;
impl From<ae::Error> for Fallback {
    fn from(_: ae::Error) -> Self {
        Fallback
    }
}

/// Input checkouts for one render: the frame size and alpha test every checkout needs, and the
/// number of host checkouts made (for the log and the tests).
struct Prober<'a> {
    in_data: &'a ae::InData,
    size: (usize, usize),
    cur_has_alpha: bool,
    count: Cell<u32>,
    metrics: Option<&'a CoMetrics>,
}

impl Prober<'_> {
    fn luma(&self, time: i32, step: i32, scale: u32, fmt: Option<&mut Option<String>>) -> Result<Vec<f32>, Outcome> {
        self.count.set(self.count.get() + 1);
        checkout_luma(self.in_data, time, step, scale, self.size, self.cur_has_alpha, fmt, self.metrics)
    }
}

/// The render path proper. Any `Err` makes the caller copy input to output.
///
/// TIME (ae: ppro/basic-host-differences): Premiere's time_scale/time_step differ from AE's
/// (e.g. 60000/1001 for 29.97) and only their ratio matters. The legacy path only adds multiples
/// of time_step to current_time and passes time_step/time_scale back unchanged; the source path
/// adds fractions of one source frame and, when a source frame is shorter than 8 native units,
/// multiplies time, step and scale by the same power of two (verified against the current frame
/// before use). Premiere renders out of order from several effect instances on different
/// threads; the only state kept across calls is the per-clip record in `source_cache`
/// (mutex-guarded, never locked across a host call), and only in Premiere.
///
/// NEIGHBOURS: the source-frame strategy is tried only when `source_gate` passes (Premiere,
/// "Use Source Frame Rate" on, original clip rate clearly above the sequence rate, not a
/// synthetic track item); otherwise, and whenever the clip's ladder says so, the legacy search
/// is used, which is exactly the 1.0 behaviour. Measured in Premiere Pro 2027: the effect's time
/// is MEDIA time and time_step is one sequence frame of media time regardless of clip speed.
///
/// CURRENT FRAME: the analysis luma of the current frame comes through the same checkout path as
/// the neighbours (in case checked-out frames are processed differently), until a checkout has
/// been shown to duplicate the effect's own input for this clip; from then on the input's luma
/// is used and the checkout (a full composite render on an adjustment layer) is skipped.
fn render(in_data: &ae::InData, core_params: &core::Params, req: SourceReq, in_layer: &ae::Layer, out_layer: &mut ae::Layer, dg: &mut Diag) -> Result<(), Fallback> {
    let _render_guard = core::par::enter_render();
    let step = in_data.time_step();
    let scale = in_data.time_scale();
    dg.local_time_step = in_data.local_time_step();
    if dg.enabled {
        if let Some(g) = geom_of(in_layer.as_ptr()) {
            dg.in_size = Some((g.width, g.height));
            dg.in_fmt = Some(format_name(in_data, in_layer, &g));
        }
    }
    if step == 0 {
        dg.legacy_reason = Some("zero time step");
        return Err(Fallback);
    }

    let t_dec = dg.enabled.then(Instant::now);
    let (desc, cur_rgba, input_luma, cur_max_alpha) = decode_with_luma(in_data, in_layer).ok_or(Fallback)?;
    if let Some(t0) = t_dec {
        dg.pix_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    let out_g = geom_of(out_layer.as_ptr()).ok_or(Fallback)?;
    let (out_layout, out_alpha) = classify(in_data, out_layer, &out_g).ok_or(Fallback)?;
    if (out_g.width, out_g.height) != (desc.width, desc.height) {
        return Err(Fallback);
    }
    let (w, h) = (desc.width, desc.height);
    let size = (w, h);
    let cur_frame = core::Frame { width: w, height: h, data: &cur_rgba };
    let co_metrics = CoMetrics::default();
    let p = Prober {
        in_data,
        size,
        cur_has_alpha: cur_max_alpha > 0.0,
        count: Cell::new(0),
        metrics: dg.enabled.then_some(&co_metrics),
    };
    let now = in_data.current_time();

    if let Some(host) = &req.host {
        dg.rate = Some(neighbors::source_rate(&host.timing, step, scale));
        dg.d_orig = neighbors::ticks_to_units(host.orig_ticks, scale);
    }
    // Per-clip record (Premiere). Skipping the current checkout needs a real instance id.
    let key = req.host.as_ref().map(|h| cache_key(h, step, scale, size));
    let entry = key.map(|k| source_cache::get(&k)).unwrap_or_default();
    let match_key = key.filter(|k| k.instance_id != source_cache::INSTANCE_UNKNOWN);
    let (cur_luma, have_cur) = current_luma(&p, now, step, scale, input_luma, match_key, entry.input_matches_checkout, dg);
    let source = {
        let cur_l = core::Luma { width: w, height: h, data: &cur_luma };
        let is_dup = |luma: &Vec<f32>| core::is_duplicate_frame(cur_l, core::Luma { width: w, height: h, data: luma });
        match source_gate(&req, step, scale, dg) {
            Ok((host, d_orig)) => try_source(&p, &host, entry.source, d_orig, now, step, scale, have_cur, &is_dup, dg),
            Err(reason) => Err(reason),
        }
    };
    let (prev, next, d_prev, d_next) = match source {
        Ok((prev, next, fd)) => {
            dg.mode = "source";
            let (dp, dn) = (prev.as_ref().map(|_| fd), next.as_ref().map(|_| fd));
            (prev, next, dp, dn)
        }
        Err(reason) => {
            dg.mode = "legacy";
            dg.legacy_reason = Some(reason);
            let (prev, next, dp, dn) = {
                let cur_l = core::Luma { width: w, height: h, data: &cur_luma };
                let is_dup = |luma: &Vec<f32>| core::is_duplicate_frame(cur_l, core::Luma { width: w, height: h, data: luma });
                legacy_neighbors(&p, now, step, scale, &is_dup, dg)
            };
            let (dp, dn) = legacy_rescale(&req, step, (dp, dn), dg);
            (prev, next, dp, dn)
        }
    };
    dg.checkouts = p.count.get();
    dg.fd = (d_prev, d_next);
    if dg.enabled {
        dg.co_ms = co_metrics.co_ms.get();
        dg.co_max_ms = co_metrics.co_max_ms.get();
        dg.pix_ms += co_metrics.pix_ms.get();
        dg.co_in_flight_max = co_metrics.co_in_flight_max.get();
        dg.computing = Some(core::par::active_renders());
        dg.budget = Some(core::par::budget());
    }

    let mut out_rgba = vec![0f32; cur_rgba.len()];
    dg.stats = Some(call_core_owned(
        cur_frame,
        cur_luma,
        prev.zip(d_prev),
        next.zip(d_next),
        core_params,
        &mut out_rgba,
    ));

    let out_desc = ImageDesc { width: out_g.width, height: out_g.height, rowbytes: out_g.rowbytes, layout: out_layout, alpha: out_alpha };
    let out_bytes = unsafe { out_g.bytes_mut(out_layout) };
    let t_wr = dg.enabled.then(Instant::now);
    let wr = pixels::write_rgba(&out_rgba, out_bytes, &out_desc).map_err(|_| Fallback);
    if let Some(t0) = t_wr {
        dg.pix_ms += t0.elapsed().as_secs_f64() * 1000.0;
    }
    wr
}

/// Frame distances of a legacy result for real media in Premiere: only when the gate said
/// `NOT_ABOVE_SEQUENCE` (D_orig known and plausible, item not synthetic), rescaled by
/// `neighbors::legacy_fd_rule`; the frames chosen never change. Everything else (After Effects,
/// checkbox off, synthetic items, no D_orig, rejected / anomalous clips) is returned unchanged.
fn legacy_rescale(req: &SourceReq, step: i32, fd: (Option<f32>, Option<f32>), dg: &mut Diag) -> (Option<f32>, Option<f32>) {
    let (Some(Err(neighbors::NOT_ABOVE_SEQUENCE)), Some(host), Some(d_orig)) = (dg.gate, req.host, dg.d_orig) else {
        return fd;
    };
    let adv = neighbors::source_advance(dg.local_time_step, step, d_orig, host.timing.speed);
    dg.adv = Some(adv);
    let rule = neighbors::legacy_fd_rule(d_orig, step, adv.a);
    dg.legacy_fd = Some(rule);
    (neighbors::legacy_fd(rule, adv.a, fd.0), neighbors::legacy_fd(rule, adv.a, fd.1))
}

/// Analysis luma of the current frame and whether it is trustworthy for duplicate tests
/// (false only when the checkout failed). Skips the checkout when `match_key`'s record says a
/// checkout duplicated the input before; otherwise checks out, compares with `input` and records
/// the result.
#[allow(clippy::too_many_arguments)]
fn current_luma(p: &Prober, now: i32, step: i32, scale: u32, input: Vec<f32>, match_key: Option<source_cache::Key>, known: Option<bool>, dg: &mut Diag) -> (Vec<f32>, bool) {
    if match_key.is_some() && known == Some(true) {
        dg.cur_outcome = Some("skipped");
        return (input, true);
    }
    let mut fmt = None;
    let r = p.luma(now, step, scale, dg.enabled.then_some(&mut fmt));
    dg.cur_fmt = fmt;
    match r {
        Ok(luma) => {
            let (w, h) = p.size;
            let matched = core::is_duplicate_frame(core::Luma { width: w, height: h, data: &input }, core::Luma { width: w, height: h, data: &luma });
            dg.cur_outcome = Some(if matched { "match" } else { "mismatch" });
            if let Some(k) = match_key {
                source_cache::record_input_match(k, matched);
            }
            (luma, true)
        }
        Err(o) => {
            dg.cur_outcome = Some("failed");
            dg.cur_fail = Some(o.name());
            (input, false)
        }
    }
}

type Found = (Option<Vec<f32>>, Option<Vec<f32>>, Option<f32>, Option<f32>);

/// Cache key for this clip in this timing context (see `source_cache::Key`).
fn cache_key(host: &HostTiming, step: i32, scale: u32, size: (usize, usize)) -> source_cache::Key {
    source_cache::Key {
        instance_id: host.instance_id.unwrap_or(source_cache::INSTANCE_UNKNOWN),
        time_step: step,
        time_scale: scale,
        orig_ticks: host.orig_ticks,
        media_ticks: host.timing.media_ticks,
        seq_ticks: host.timing.seq_ticks,
        width: size.0,
        height: size.1,
    }
}

/// Source-frame strategy (see `neighbors::evaluate`) for a gated-in clip: returns (prev, next,
/// frame distance) or the reason to use the legacy path instead. D is always D_orig. `state` is
/// the key's state read at the start of the render; the classification is applied to the stored
/// state afterwards.
#[allow(clippy::too_many_arguments)]
fn try_source(
    p: &Prober,
    host: &HostTiming,
    state: SourceState,
    d_orig: f64,
    now: i32,
    step: i32,
    scale: u32,
    have_cur: bool,
    is_dup: &dyn Fn(&Vec<f32>) -> bool,
    dg: &mut Diag,
) -> Result<(Option<Vec<f32>>, Option<Vec<f32>>, f32), &'static str> {
    dg.state = Some(state);
    if state.verdict == Verdict::Rejected {
        return Err("original clip frame rate rejected for this clip");
    }
    if !have_cur {
        return Err("current-time checkout failed");
    }
    dg.d = Some(d_orig);
    let adv = neighbors::source_advance(p.in_data.local_time_step(), step, d_orig, host.timing.speed);
    dg.adv = Some(adv);
    let rep = neighbors::choose_time_rep(now, step, scale, d_orig).ok_or("time resolution out of range")?;
    dg.m = rep.m;
    if rep.m > 1 {
        // Proves the host honours the scaled (time, scale) pair: the current time expressed in
        // the finer units must return the same frame. Never needed with Premiere's 1001-unit
        // steps (the gate keeps D_orig >= step / 64 > 8 units there).
        let v = p.luma(rep.now, rep.step, rep.scale, None);
        let outcome = match &v {
            Ok(l) if is_dup(l) => "duplicate",
            Ok(_) => "mismatch",
            Err(o) => o.name(),
        };
        dg.probes.push(ProbeLog { kind: "verify", time: rep.now as i64, m: rep.m, outcome });
        if outcome != "duplicate" {
            return Err("host did not honour scaled time");
        }
    }
    let (w, h) = p.size;
    let ev = neighbors::evaluate(
        rep.now,
        rep.d,
        state.verdict == Verdict::Verified,
        |t| p.luma(t, rep.step, rep.scale, None),
        is_dup,
        |a: &Vec<f32>, b: &Vec<f32>| core::is_duplicate_frame(core::Luma { width: w, height: h, data: a }, core::Luma { width: w, height: h, data: b }),
    );
    dg.probes.extend(ev.probes.iter().map(|r| ProbeLog { kind: r.role.name(), time: r.time, m: rep.m, outcome: r.outcome.name() }));
    dg.class = Some(ev.class);
    dg.dups = Some(ev.dups);
    let key = cache_key(host, step, scale, p.size);
    dg.state_after = Some(source_cache::apply(key, ev.class));
    if !ev.class.uses_source() {
        return Err(ev.class.legacy_reason());
    }
    Ok((ev.prev, ev.next, neighbors::source_frame_distance(adv.a)))
}

/// Result of running the legacy neighbour-selection rule (`neighbors::find_neighbor` +
/// `neighbors::frame_distances`) over a generic per-side prober.
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub(crate) struct LegacySelection {
    pub prev: Option<(u32, Vec<f32>)>,
    pub next: Option<(u32, Vec<f32>)>,
    pub fd: (Option<f32>, Option<f32>),
    pub probes: Vec<(i32, u32, Outcome)>,
}

/// Shared legacy neighbour-selection rule used by both the CPU path (`legacy_neighbors`) and the
/// macOS Premiere GPU adjustment-layer path (`gpu_filter`). Calls `neighbors::find_neighbor` for
/// `dir = -1` (previous) then `dir = +1` (next) up to `max_k` steps, and computes frame distances
/// via `neighbors::frame_distances`.
pub(crate) fn select_legacy_neighbors(
    max_k: u32,
    mut probe_side: impl FnMut(i32, u32) -> Result<Vec<f32>, Outcome>,
    is_dup: &dyn Fn(&Vec<f32>) -> bool,
) -> LegacySelection {
    let probes = std::cell::RefCell::new(Vec::new());
    let mut search = |dir: i32| {
        let last_k = Cell::new(0u32);
        neighbors::find_neighbor(
            max_k,
            |k| {
                last_k.set(k);
                match probe_side(dir, k) {
                    Ok(luma) => Some(luma),
                    Err(o) => {
                        probes.borrow_mut().push((dir, k, o));
                        None
                    }
                }
            },
            |luma: &Vec<f32>| {
                let d = is_dup(luma);
                probes.borrow_mut().push((dir, last_k.get(), if d { Outcome::Duplicate } else { Outcome::Used }));
                d
            },
        )
    };
    let prev = search(-1);
    let next = search(1);
    let fd = neighbors::frame_distances(prev.as_ref().map(|p| p.0), next.as_ref().map(|n| n.0));
    LegacySelection { prev, next, fd, probes: probes.into_inner() }
}

/// Legacy (1.0) search: k = 1..=MAX_K whole time_steps away, skipping held frames.
fn legacy_neighbors(p: &Prober, now: i32, step: i32, scale: u32, is_dup: &dyn Fn(&Vec<f32>) -> bool, dg: &mut Diag) -> Found {
    let sel = select_legacy_neighbors(
        MAX_K,
        |dir, k| {
            let t = neighbors::offset_time(now, step, k, dir);
            t.ok_or(Outcome::Unusable).and_then(|t| p.luma(t, step, scale, None))
        },
        is_dup,
    );
    dg.probes.extend(sel.probes.into_iter().map(|(dir, k, outcome)| {
        let kind = if dir < 0 { "lprev" } else { "lnext" };
        let time = neighbors::offset_time(now, step, k, dir)
            .map_or(now as i64 + dir as i64 * k as i64 * (step as i64).abs(), |t| t as i64);
        ProbeLog { kind, time, m: 1, outcome: outcome.name() }
    }));
    (sel.prev.map(|p| p.1), sel.next.map(|n| n.1), sel.fd.0, sel.fd.1)
}

#[cfg(core_has_render_with_luma)]
pub(crate) fn call_core_owned(
    cur: core::Frame,
    cur_luma: Vec<f32>,
    prev: Option<(Vec<f32>, f32)>,
    next: Option<(Vec<f32>, f32)>,
    p: &core::Params,
    out: &mut [f32],
) -> core::RenderStats {
    core::render_with_owned_luma(cur, cur_luma, prev, next, p, out)
}

/// Fallback when building against an older core without `render_with_luma`.
#[cfg(not(core_has_render_with_luma))]
pub(crate) fn call_core_owned(
    cur: core::Frame,
    _cur_luma: Vec<f32>,
    prev: Option<(Vec<f32>, f32)>,
    next: Option<(Vec<f32>, f32)>,
    p: &core::Params,
    out: &mut [f32],
) -> core::RenderStats {
    let (w, h) = (cur.width, cur.height);
    let mk = |n: &Option<(Vec<f32>, f32)>| {
        n.as_ref().map(|(luma, d)| core::Neighbor { luma: core::Luma { width: w, height: h, data: luma }, frame_distance: *d })
    };
    core::render(cur, mk(&prev), mk(&next), p, out)
}

/// Copies input to output without interpreting pixels (input and output share format and size,
/// so a byte copy of width*bpp per row is exact). If the format is unknown, the host's PF_COPY.
fn passthrough(in_data: &ae::InData, in_layer: &ae::Layer, out_layer: &mut ae::Layer) {
    let ok = catch_unwind(AssertUnwindSafe(|| {
        let (Some(i), Some(o)) = (geom_of(in_layer.as_ptr()), geom_of(out_layer.as_ptr())) else { return false };
        if i.height != o.height || i.width != o.width {
            return false;
        }
        let Some((layout, _)) = classify(in_data, in_layer, &i) else { return false };
        let row = i.width * layout.bytes_per_pixel();
        let (src, dst) = unsafe { (i.bytes(layout), o.bytes_mut(layout)) };
        pixels::copy_rows(src, i.rowbytes, dst, o.rowbytes, i.height, row).is_ok()
    }));
    if !matches!(ok, Ok(true)) {
        let _ = catch_unwind(AssertUnwindSafe(|| out_layer.copy_from(in_layer, None, None)));
    }
}

#[cfg(test)]
mod tests {
    //! Drives `render` / `passthrough` against a fake host (8-bit ARGB, padded rows): only
    //! PF_InData's time fields and the checkout/checkin callbacks are implemented. Two fake media
    //! models: a map from time to frame (legacy tests), and a "media" model where the frame index
    //! is floor(seconds * source fps) for any (time, scale) pair, which is what the source-frame
    //! strategy assumes of Premiere (and what Premiere Pro 2027 was seen to do). A media model
    //! whose "fps" equals the time scale returns a different frame for every time unit: input
    //! that changes continuously (an adjustment layer over animated tracks, a nested sequence).
    //!
    //! The per-clip cache is process-global and tests run in parallel: every fake clip gets its
    //! own effect instance id (from `NEXT_CLIP`), so tests never share cache entries.
    use super::*;
    use crate::neighbors::Verdict;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicI32, Ordering};

    const W: usize = 8;
    const H: usize = 4;
    const RB: usize = W * 4 + 12; // padded rows
    const STEP: i32 = 10;
    const BASE_SCALE: u32 = 100; // default native time_scale: timeline = 10 fps

    struct Media {
        /// Source frame rate as a fraction.
        fps_num: i64,
        fps_den: i64,
        frames: Vec<Vec<u8>>,
        /// false: the host ignores the scale argument and reads `time` in native units.
        honour_scale: bool,
        /// true: times outside the media return the edge frame (as Premiere does before a clip's
        /// start); false: no frame.
        repeat_edges: bool,
    }

    thread_local! {
        static FRAMES: RefCell<HashMap<i32, Vec<u8>>> = RefCell::new(HashMap::new());
        static MEDIA: RefCell<Option<Media>> = const { RefCell::new(None) };
        static LOG: RefCell<(Vec<i32>, usize, usize)> = const { RefCell::new((Vec::new(), 0, 0)) };
        static SCALES: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
        /// Native (time_step, time_scale) of the fake timeline.
        static NATIVE: Cell<(i32, u32)> = const { Cell::new((STEP, BASE_SCALE)) };
        /// local_time_step override; default: time_step * |speed| of the request (as observed).
        static LOCAL_STEP: Cell<Option<i32>> = const { Cell::new(None) };
        /// Effect instance id of the current fake clip (`Utility::filter_instance_id`).
        static CLIP: Cell<i32> = const { Cell::new(0) };
        /// Seed of a frame handed to the effect as its input instead of the media frame (models
        /// a checkout that differs from the input).
        static INPUT_SEED: Cell<Option<u8>> = const { Cell::new(None) };
    }
    static NEXT_CLIP: AtomicI32 = AtomicI32::new(1);

    fn new_clip() {
        CLIP.with(|c| c.set(NEXT_CLIP.fetch_add(1, Ordering::Relaxed)));
    }

    fn native() -> (i32, u32) {
        NATIVE.with(|n| n.get())
    }

    fn frame(seed: u8) -> Vec<u8> {
        let mut v = vec![0xABu8; RB * H];
        for y in 0..H {
            for x in 0..W {
                let o = y * RB + x * 4;
                v[o..o + 4].copy_from_slice(&[255, seed.wrapping_add((x * 20) as u8), (y * 50) as u8, seed]);
            }
        }
        v
    }

    fn media_index(m: &Media, t: i32, scale: u32) -> i64 {
        let sc = if m.honour_scale { scale } else { native().1 } as i64;
        (t as i64 * m.fps_num).div_euclid(sc * m.fps_den)
    }

    unsafe extern "C" fn fake_checkout(_: sys::PF_ProgPtr, index: sys::PF_ParamIndex, t: sys::A_long, step: sys::A_long, scale: sys::A_u_long, def: *mut sys::PF_ParamDef) -> sys::PF_Err {
        assert_eq!(index, 0);
        LOG.with(|l| {
            let mut l = l.borrow_mut();
            l.0.push(t);
            l.1 += 1;
        });
        SCALES.with(|s| s.borrow_mut().push(scale));
        let ld = unsafe { &mut (*def).u.ld };
        let set = |ld: &mut sys::PF_LayerDef, buf: &mut Vec<u8>| {
            ld.data = buf.as_mut_ptr() as _;
            ld.width = W as _;
            ld.height = H as _;
            ld.rowbytes = RB as _;
            ld.world_flags = 0;
        };
        let (nstep, nscale) = native();
        let media = MEDIA.with(|m| {
            let mut m = m.borrow_mut();
            let Some(m) = m.as_mut() else { return false };
            // Any representation must describe the same frame duration as the native one.
            assert_eq!(step as i64 * nscale as i64, nstep as i64 * scale as i64, "step {step} scale {scale}");
            let mut i = media_index(m, t, scale);
            if m.repeat_edges {
                i = i.clamp(0, m.frames.len() as i64 - 1);
            }
            if (0..m.frames.len() as i64).contains(&i) {
                set(ld, &mut m.frames[i as usize]);
            }
            true
        });
        if !media {
            assert_eq!(step, nstep);
            assert_eq!(scale, nscale);
            FRAMES.with(|f| {
                if let Some(buf) = f.borrow_mut().get_mut(&t) {
                    set(ld, buf);
                }
            });
        }
        0
    }
    unsafe extern "C" fn fake_checkin(_: sys::PF_ProgPtr, _: *mut sys::PF_ParamDef) -> sys::PF_Err {
        LOG.with(|l| l.borrow_mut().2 += 1);
        0
    }

    fn world(buf: &mut [u8]) -> sys::PF_LayerDef {
        let mut ld: sys::PF_LayerDef = unsafe { std::mem::zeroed() };
        ld.data = buf.as_mut_ptr() as _;
        ld.width = W as _;
        ld.height = H as _;
        ld.rowbytes = RB as _;
        ld
    }

    fn in_data(now: i32, step: i32, local: i32) -> Box<sys::PF_InData> {
        let mut d: Box<sys::PF_InData> = Box::new(unsafe { std::mem::zeroed() });
        d.appl_id = i32::from_be_bytes(*b"FXTC");
        d.current_time = now;
        d.time_step = step;
        d.local_time_step = local;
        d.time_scale = native().1;
        d.inter.checkout_param = Some(fake_checkout);
        d.inter.checkin_param = Some(fake_checkin);
        d
    }

    fn reset_logs() {
        LOG.with(|l| *l.borrow_mut() = (Vec::new(), 0, 0));
        SCALES.with(|s| s.borrow_mut().clear());
    }

    /// Source frames held for 2 timeline frames: timeline t = 10*i shows source frame i/2.
    fn setup_held_clip(last: i32) {
        MEDIA.with(|m| *m.borrow_mut() = None);
        NATIVE.with(|n| n.set((STEP, BASE_SCALE)));
        LOCAL_STEP.with(|l| l.set(None));
        INPUT_SEED.with(|s| s.set(None));
        new_clip();
        FRAMES.with(|f| {
            let mut f = f.borrow_mut();
            f.clear();
            for i in 0..=last {
                f.insert(i * STEP, frame(((i / 2) * 40) as u8));
            }
        });
        reset_logs();
    }

    fn media_frames(count: usize) -> Vec<Vec<u8>> {
        (0..count).map(|i| frame((i * 37 % 251) as u8)).collect()
    }

    /// Media of `count` source frames at fps_num/fps_den, starting at time 0, on a native
    /// timeline of (time_step, time_scale); a new clip identity. Frame i has seed 37*i mod 251, so
    /// frames closer than 251 never repeat.
    fn setup_media_on(step: i32, scale: u32, fps_num: i64, fps_den: i64, count: usize, honour_scale: bool) {
        NATIVE.with(|n| n.set((step, scale)));
        LOCAL_STEP.with(|l| l.set(None));
        INPUT_SEED.with(|s| s.set(None));
        new_clip();
        MEDIA.with(|m| *m.borrow_mut() = Some(Media { fps_num, fps_den, frames: media_frames(count), honour_scale, repeat_edges: false }));
        reset_logs();
    }

    /// As `setup_media_on`, on the default 10 fps timeline.
    fn setup_media(fps_num: i64, fps_den: i64, count: usize, honour_scale: bool) {
        setup_media_on(STEP, BASE_SCALE, fps_num, fps_den, count, honour_scale);
    }

    fn set_repeat_edges(on: bool) {
        MEDIA.with(|m| m.borrow_mut().as_mut().unwrap().repeat_edges = on);
    }

    /// Replaces the media behind the CURRENT clip identity (same cache key).
    fn swap_media(fps_num: i64, fps_den: i64, count: usize) {
        MEDIA.with(|m| *m.borrow_mut() = Some(Media { fps_num, fps_den, frames: media_frames(count), honour_scale: true, repeat_edges: false }));
        reset_logs();
    }

    /// Index of the source frame shown at a native time (the frame the host renders at `now`).
    fn shown(now: i32) -> i64 {
        MEDIA.with(|m| m.borrow().as_ref().map(|m| media_index(m, now, native().1))).unwrap()
    }

    /// Source frames of the ladder probes that were used as (prev, next).
    fn used_frames(dg: &Diag) -> (Option<i64>, Option<i64>) {
        let at = |kind: &str| {
            dg.probes.iter().find(|p| p.kind == kind && p.outcome == "used").map(|p| MEDIA.with(|m| media_index(m.borrow().as_ref().unwrap(), p.time as i32, native().1 * p.m)))
        };
        (at("prev"), at("next"))
    }

    struct Run {
        r: Result<(), Fallback>,
        input: Vec<u8>,
        output: Vec<u8>,
        times: Vec<i32>,
        scales: Vec<u32>,
        co: usize,
        ci: usize,
        dg: Diag,
    }

    /// One render at `now`. Logs are reset first, so `times` / `co` are this render's only.
    fn run(now: i32, step: i32, req: SourceReq) -> Run {
        reset_logs();
        let mut input = match INPUT_SEED.with(|s| s.get()) {
            Some(seed) => frame(seed),
            None => MEDIA
                .with(|m| {
                    m.borrow().as_ref().map(|m| {
                        let i = media_index(m, now, native().1);
                        m.frames[i.clamp(0, m.frames.len() as i64 - 1) as usize].clone()
                    })
                })
                .unwrap_or_else(|| FRAMES.with(|f| f.borrow().get(&now).cloned().unwrap())),
        };
        let mut output = vec![0x11u8; RB * H];
        let local = LOCAL_STEP.with(|l| l.get()).unwrap_or_else(|| match req.host {
            Some(h) => (step as f64 * neighbors::norm_speed(h.timing.speed)).round() as i32,
            None => step,
        });
        let d = in_data(now, step, local);
        let idata = ae::InData::from_raw(&*d);
        let mut in_ld = world(&mut input);
        let mut out_ld = world(&mut output);
        let in_layer = ae::Layer::from_raw(&mut in_ld, idata, None);
        let mut out_layer = ae::Layer::from_raw(&mut out_ld, idata, None);
        let mut dg = Diag { enabled: true, host: req.host, ..Diag::default() };
        let r = render(&idata, &core::Params::default(), req, &in_layer, &mut out_layer, &mut dg);
        if r.is_err() {
            passthrough(&idata, &in_layer, &mut out_layer);
        }
        // The diagnostics line must format without panicking.
        assert!(dg.line(&idata, "test", 0.0).contains("event=render"));
        drop((in_layer, out_layer));
        let (times, co, ci) = LOG.with(|l| l.borrow().clone());
        assert_eq!(co, ci, "every checkout must be checked in");
        if r.is_ok() {
            assert_eq!(dg.checkouts as usize, co, "checkout counter");
        }
        Run { r, input, output, times, scales: SCALES.with(|s| s.borrow().clone()), co, ci, dg }
    }

    const AE: SourceReq = SourceReq { host: None, allowed: Err("after effects host") };

    fn tps() -> i64 {
        neighbors::TICKS_PER_SECOND
    }

    /// A Premiere request for the current fake clip with the given reported timing.
    fn host_full(timing: ClipTiming, orig_ticks: Option<i64>, synthetic: Option<bool>) -> SourceReq {
        SourceReq { host: Some(HostTiming { timing, orig_ticks, synthetic, instance_id: Some(CLIP.with(|c| c.get())), ..HostTiming::default() }), allowed: Ok(()) }
    }

    fn host(timing: ClipTiming, orig_ticks: Option<i64>) -> SourceReq {
        host_full(timing, orig_ticks, None)
    }

    /// Premiere on the 10 fps timeline with media of `src_fps` (original rate reported correctly).
    fn timing(src_fps: i64, speed: f64) -> SourceReq {
        host(ClipTiming { media_ticks: Some(tps() / 10), seq_ticks: Some(tps() / 10), speed: Some(speed) }, Some(tps() / src_fps))
    }

    fn key_of(req: SourceReq) -> source_cache::Key {
        cache_key(&req.host.unwrap(), native().0, native().1, (W, H))
    }

    fn verdict(req: SourceReq) -> SourceState {
        source_cache::get(&key_of(req)).source
    }

    fn outcomes(dg: &Diag) -> Vec<(&'static str, i64, u32, &'static str)> {
        dg.probes.iter().map(|p| (p.kind, p.time, p.m, p.outcome)).collect()
    }

    fn approx(fd: (Option<f32>, Option<f32>), want: f32) -> bool {
        matches!(fd, (Some(a), Some(b)) if (a - want).abs() < 1e-4 && (b - want).abs() < 1e-4)
    }

    const VERIFIED: SourceState = SourceState { verdict: Verdict::Verified, strikes: 0 };

    // ── legacy / After Effects ──

    #[test]
    fn skips_held_frames_and_balances_checkouts() {
        setup_held_clip(10);
        // t=40 is source frame 2 (first of its hold). prev t=30 is source 1 (k=1);
        // next t=50 repeats source 2, so t=60 is used (k=2).
        let rr = run(40, STEP, AE);
        assert!(rr.r.is_ok());
        assert_eq!(rr.times, vec![40, 30, 50, 60]);
        // Row padding of the output is untouched.
        assert!(rr.output[W * 4..RB].iter().all(|&b| b == 0x11));
        assert_eq!((rr.dg.mode, rr.dg.legacy_reason), ("legacy", Some("after effects host")));
        assert_eq!((rr.dg.fd, rr.dg.legacy_fd), ((Some(2.0), Some(2.0)), None));
        assert_eq!(outcomes(&rr.dg), vec![("lprev", 30, 1, "used"), ("lnext", 50, 1, "duplicate"), ("lnext", 60, 1, "used")]);
        // After Effects: no per-clip record, the current checkout is always made.
        assert_eq!((rr.dg.state, rr.dg.cur_outcome), (None, Some("match")));
        assert_eq!(run(40, STEP, AE).times, vec![40, 30, 50, 60]);
    }

    #[test]
    fn clip_edges_and_missing_frames() {
        setup_held_clip(3);
        // At t=0 there is nothing before; t=-10..-40 are probed and checked in.
        let rr = run(0, STEP, AE);
        assert!(rr.r.is_ok());
        assert_eq!(rr.times, vec![0, -10, -20, -30, -40, 10, 20]);
        assert_eq!(rr.co, rr.ci);
    }

    #[test]
    fn zero_time_step_passes_input_through() {
        setup_held_clip(3);
        let rr = run(10, 0, AE);
        assert!(rr.r.is_err());
        assert_eq!(rr.co, rr.ci);
        for y in 0..H {
            assert_eq!(&rr.output[y * RB..y * RB + W * 4], &rr.input[y * RB..y * RB + W * 4]);
        }
    }

    // ── gate ──

    /// What the host reported on the adjustment layer in the 1.2 log: every rate = the sequence's.
    fn seq_ticks_23_976() -> i64 {
        tps() * 1001 / 24000
    }

    fn premiere_timing(speed: f64) -> ClipTiming {
        let seq = seq_ticks_23_976();
        ClipTiming { media_ticks: Some(seq), seq_ticks: Some(seq), speed: Some(speed) }
    }

    /// Input that changes with every time unit on a 23.976 sequence (1001 / 24000).
    fn setup_continuous_23_976() {
        setup_media_on(1001, 24000, 24000, 1, 20_000, true);
        LOCAL_STEP.with(|l| l.set(Some(1001)));
    }

    #[test]
    fn regression_adjustment_layer_is_gated_out() {
        assert_eq!(seq_ticks_23_976(), 10_594_584_000);
        // As in the 1.2 log: orig_ticks = sequence rate, input changing continuously. Both with
        // the synthetic flag unknown / false and true.
        for (synthetic, reason) in [(None, "source rate not above sequence rate"), (Some(false), "source rate not above sequence rate"), (Some(true), "synthetic track item")] {
            setup_continuous_23_976();
            let req = host_full(premiere_timing(1.0), Some(seq_ticks_23_976()), synthetic);
            for (i, now) in [10_010, 11_011, 12_012, 13_013].into_iter().enumerate() {
                let rr = run(now, 1001, req);
                assert!(rr.r.is_ok());
                assert_eq!((rr.dg.mode, rr.dg.legacy_reason, rr.dg.gate), ("legacy", Some(reason), Some(Err(reason))));
                // Exactly the 1.0 neighbours: +-1001, frame distance 1, no source probing.
                let neighbours: Vec<_> = outcomes(&rr.dg).into_iter().map(|p| (p.0, p.1, p.3)).collect();
                assert_eq!(neighbours, vec![("lprev", now as i64 - 1001, "used"), ("lnext", now as i64 + 1001, "used")]);
                assert_eq!(rr.dg.fd, (Some(1.0), Some(1.0)));
                // a = 1, D_orig = step: the legacy frame-distance rule leaves the 1.0 value.
                assert_eq!(rr.dg.legacy_fd, (reason == neighbors::NOT_ABOVE_SEQUENCE).then_some(neighbors::LegacyFd::H));
                assert_eq!((rr.dg.state, rr.dg.class), (None, None));
                // current + prev + next on the first render; the current checkout matched the
                // input, so later renders skip it.
                if i == 0 {
                    assert_eq!((rr.co, rr.dg.cur_outcome), (3, Some("match")));
                    assert_eq!(rr.times, vec![now, now - 1001, now + 1001]);
                } else {
                    assert_eq!((rr.co, rr.dg.cur_outcome), (2, Some("skipped")));
                }
            }
        }
    }

    #[test]
    fn synthetic_hfr_clip_is_gated_out() {
        setup_119_88_on_23_976();
        let req = host_full(premiere_timing(1.0), Some(orig_119_88()), Some(true));
        let rr = run(108_108, 1001, req);
        assert_eq!((rr.dg.mode, rr.dg.legacy_reason), ("legacy", Some("synthetic track item")));
        assert!(rr.dg.probes.iter().all(|p| p.kind.starts_with('l')));
        assert_eq!(rr.co, 3);
        assert_eq!(verdict(req), SourceState::default());
    }

    #[test]
    fn same_rate_and_missing_orig_are_gated_out() {
        // 10 fps media on the 10 fps timeline: legacy, exactly the 1.0 checkouts.
        setup_media(10, 1, 100, true);
        let req = timing(10, 1.0);
        let rr = run(73, STEP, req);
        assert_eq!((rr.dg.mode, rr.dg.legacy_reason), ("legacy", Some("source rate not above sequence rate")));
        assert_eq!(rr.times, vec![73, 63, 83]);
        assert_eq!(run(83, STEP, req).times, vec![73, 93]);
        // No original clip frame rate (or a nonsense one): legacy, whatever else is reported.
        for orig in [None, Some(0), Some(tps() * 10)] {
            setup_media(50, 1, 100, true);
            let req = host(ClipTiming { media_ticks: Some(tps() / 50), seq_ticks: Some(tps() / 10), speed: None }, orig);
            let rr = run(70, STEP, req);
            assert_eq!(rr.dg.legacy_reason, Some("no original clip frame rate"));
            assert_eq!(rr.co, 3);
        }
        // Checkbox off in Premiere: legacy, but the current checkout is still skipped once shown
        // to match.
        setup_media(50, 1, 100, true);
        let req = SourceReq { allowed: Err("disabled by parameter"), ..timing(50, 1.0) };
        assert_eq!(run(70, STEP, req).dg.legacy_reason, Some("disabled by parameter"));
        assert_eq!(run(80, STEP, req).times, vec![70, 90]);
    }

    // ── HFR media: 119.88 fps on a 23.976 sequence ──

    fn orig_119_88() -> i64 {
        // 2118916800, as logged in host for a 119.88 fps clip.
        tps() * 1001 / 120_000
    }

    fn setup_119_88_on_23_976() {
        setup_media_on(1001, 24000, 120_000, 1001, 1000, true);
        LOCAL_STEP.with(|l| l.set(Some(1001)));
    }

    #[test]
    fn hfr_media_verifies_on_first_render_and_stays_verified() {
        assert_eq!(orig_119_88(), 2_118_916_800);
        setup_119_88_on_23_976();
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        // 100% speed: 5 source frames per sequence frame, aligned phase (108108 = 540 * 200.2).
        for i in 0..6 {
            let now = 108_108 + i * 1001;
            let rr = run(now, 1001, req);
            let ctx = format!("now {now} probes {:?} reason {:?}", outcomes(&rr.dg), rr.dg.legacy_reason);
            assert_eq!((rr.dg.mode, rr.dg.class), ("source", Some(Class::Consistent)), "{ctx}");
            assert_eq!(rr.dg.state, Some(if i == 0 { SourceState::default() } else { VERIFIED }), "{ctx}");
            assert_eq!(rr.dg.state_after, Some(VERIFIED));
            assert!((rr.dg.d.unwrap() - 200.2).abs() < 1e-9);
            let cur = shown(now);
            assert_eq!(used_frames(&rr.dg), (Some(cur - 1), Some(cur + 1)), "{ctx}");
            assert!(approx(rr.dg.fd, 0.2), "{:?} {ctx}", rr.dg.fd);
            assert!(rr.scales.iter().all(|&s| s == 24000));
            // Aligned: prev -50 used; next +150 duplicate, +250 used. Plus the current checkout
            // on the first render only.
            assert_eq!(rr.dg.dups, Some((0, 1)), "{ctx}");
            assert_eq!(rr.co, if i == 0 { 4 } else { 3 }, "{ctx}");
        }
    }

    #[test]
    fn hfr_media_at_20_percent() {
        setup_119_88_on_23_976();
        LOCAL_STEP.with(|l| l.set(Some(200)));
        let req = host(premiere_timing(0.2), Some(orig_119_88()));
        for i in 0..4 {
            let now = 108_708 + i * 200;
            let rr = run(now, 1001, req);
            assert_eq!(rr.dg.mode, "source", "{:?}", rr.dg.legacy_reason);
            let cur = shown(now);
            assert_eq!(used_frames(&rr.dg), (Some(cur - 1), Some(cur + 1)));
            assert_eq!(rr.dg.fd, (Some(1.0), Some(1.0)));
            assert!((rr.dg.adv.unwrap().a - 200.0 / 200.2).abs() < 1e-9);
        }
        assert_eq!(verdict(req), VERIFIED);
    }

    /// Plays renders at the given times; returns (class, state after, checkouts) per render.
    fn play(req: SourceReq, times: impl Iterator<Item = i32>) -> Vec<(Option<Class>, SourceState, usize, &'static str)> {
        times
            .map(|now| {
                let rr = run(now, native().0, req);
                assert!(rr.r.is_ok());
                if rr.dg.class == Some(Class::Consistent) {
                    let cur = shown(now);
                    assert_eq!(used_frames(&rr.dg), (Some(cur - 1), Some(cur + 1)), "now {now}");
                }
                (rr.dg.class, verdict(req), rr.co, rr.dg.mode)
            })
            .collect()
    }

    #[test]
    fn quarter_phase_sequences_verify_and_never_strike() {
        // 119.88 fps at 25% (local step 250: phase cycles through ~0, 0.25, 0.5, 0.75).
        setup_119_88_on_23_976();
        LOCAL_STEP.with(|l| l.set(Some(250)));
        let req = host(premiere_timing(0.25), Some(orig_119_88()));
        let r = play(req, (0..16).map(|i| 108_108 + (i as f64 * 250.25).round() as i32));
        assert!(r.iter().all(|x| x.0 == Some(Class::Consistent) && x.1 == VERIFIED && x.3 == "source"), "{r:?}");
        assert!(r.iter().skip(1).all(|x| x.2 <= 6), "{r:?}");
        eprintln!("119.88@25% checkouts per render: {:?}", r.iter().map(|x| x.2).collect::<Vec<_>>());
        // 29.97 fps on 23.976 (D 800.8, phase += 1.25 per frame).
        setup_media_on(1001, 24000, 30_000, 1001, 1000, true);
        LOCAL_STEP.with(|l| l.set(Some(1001)));
        let req = host(premiere_timing(1.0), Some(tps() * 1001 / 30_000));
        let r = play(req, (0..16).map(|i| 100_100 + i * 1001));
        assert!(r.iter().all(|x| x.0 == Some(Class::Consistent) && x.1 == VERIFIED), "{r:?}");
        eprintln!("29.97 on 23.976 checkouts per render: {:?}", r.iter().map(|x| x.2).collect::<Vec<_>>());
        // 120.0 fps on 23.976 (D = 200 units: the phase drifts through every value).
        setup_media_on(1001, 24000, 120, 1, 2000, true);
        LOCAL_STEP.with(|l| l.set(Some(1001)));
        let req = host(premiere_timing(1.0), Some(tps() / 120));
        let r = play(req, (0..40).map(|i| 100_100 + i * 1001));
        assert!(r.iter().all(|x| x.0 == Some(Class::Consistent) && x.1 == VERIFIED), "{r:?}");
    }

    #[test]
    fn phase_quarter_probe_count() {
        // Verified key, now at phase exactly 0.25 of a 200-unit frame (120.0 fps media).
        setup_media_on(1001, 24000, 120, 1, 2000, true);
        let req = host(premiere_timing(1.0), Some(tps() / 120));
        assert_eq!(run(100_000, 1001, req).dg.class, Some(Class::Consistent));
        let rr = run(100_050, 1001, req);
        assert_eq!((rr.dg.class, rr.dg.dups), (Some(Class::Consistent), Some((1, 0))));
        // The fake host floors exactly: prev -50 lands on the current frame's first unit (dup),
        // -150 on the previous frame; next +150 on the next frame's first unit. (A host that
        // jitters at the boundary may give (0, 0) + confirmation or (1, 1); see neighbors tests.)
        assert_eq!(outcomes(&rr.dg).iter().map(|p| (p.0, p.3)).collect::<Vec<_>>(), vec![("prev", "duplicate"), ("prev", "used"), ("next", "used")]);
        assert_eq!(rr.co, 3);
    }

    // ── problems ──

    #[test]
    fn interpret_footage_like_media_is_rejected_after_three_renders() {
        // orig says 119.88 but the media's true frame is 5x longer (23.976 fps).
        for (fps_num, fps_den) in [(24_000, 1001), (60_000, 1001)] {
            setup_media_on(1001, 24000, fps_num, fps_den, 1000, true);
            LOCAL_STEP.with(|l| l.set(Some(1001)));
            let req = host(premiere_timing(1.0), Some(orig_119_88()));
            let r = play(req, (0..6).map(|i| 108_108 + i * 1001));
            eprintln!("{fps_num}/{fps_den} with 119.88 orig: {r:?}");
            assert!(r.iter().all(|x| x.1.verdict != Verdict::Verified && x.3 == "legacy"), "{r:?}");
            assert!(r[..3].iter().all(|x| x.0 == Some(Class::Anomaly)), "{r:?}");
            assert_eq!((r[1].1.verdict, r[2].1.verdict), (Verdict::Unknown, Verdict::Rejected));
            // From then on: no ladder, only the legacy probes (current checkout skipped).
            assert!(r[3..].iter().all(|x| x.0.is_none() && x.2 == 2), "{r:?}");
        }
    }

    #[test]
    fn continuous_input_passing_the_gate_is_rejected_on_first_render() {
        setup_continuous_23_976();
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        let first = run(10_010, 1001, req);
        assert_eq!((first.dg.class, first.dg.mode), (Some(Class::Continuous), "legacy"));
        assert_eq!(outcomes(&first.dg).iter().filter(|p| p.0 == "confirm").count(), 1);
        assert_eq!(verdict(req).verdict, Verdict::Rejected);
        // current + prev + next + confirm + 2 legacy.
        assert_eq!(first.co, 6);
        let later = run(11_011, 1001, req);
        assert_eq!((later.dg.class, later.dg.mode, later.co), (None, "legacy", 2));
        assert_eq!(later.dg.fd, (Some(1.0), Some(1.0)));
    }

    #[test]
    fn frozen_content_keeps_state_and_later_verifies() {
        setup_119_88_on_23_976();
        MEDIA.with(|m| {
            let mut m = m.borrow_mut();
            let m = m.as_mut().unwrap();
            let f0 = m.frames[0].clone();
            for f in m.frames.iter_mut().take(100) {
                *f = f0.clone();
            }
        });
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        // Unknown key: the ladder finds nothing, the legacy search finds nothing either (all
        // within 4 steps is the same frame): no neighbours, state stays Unknown.
        let rr = run(1001 * 8, 1001, req);
        assert_eq!((rr.dg.mode, rr.dg.class, rr.dg.fd), ("legacy", Some(Class::UnverifiedNoNeighbours), (None, None)));
        assert_eq!(rr.dg.legacy_reason, Some("no source-frame neighbours on an unverified clip"));
        assert_eq!(verdict(req), SourceState::default());
        // Motion resumes: verifies.
        let later = run(108_108, 1001, req);
        assert_eq!((later.dg.mode, later.dg.class), ("source", Some(Class::Consistent)));
        assert_eq!(verdict(req), VERIFIED);
        // Frozen again on the Verified key: no neighbours on the source path, no legacy probes.
        let again = run(1001 * 8, 1001, req);
        assert_eq!((again.dg.mode, again.dg.class, again.dg.fd), ("source", Some(Class::NoNeighbours), (None, None)));
        assert_eq!((again.co, verdict(req)), (5, VERIFIED));
    }

    /// Source frames of the legacy probes that were used as (prev, next).
    fn legacy_used(dg: &Diag) -> (Option<i64>, Option<i64>) {
        let at = |kind: &str| dg.probes.iter().find(|p| p.kind == kind && p.outcome == "used").map(|p| shown(p.time as i32));
        (at("lprev"), at("lnext"))
    }

    #[test]
    fn reinterpreted_footage_at_mid_phase_takes_legacy_with_real_neighbours() {
        // orig says 119.88 but the true frame is 5x longer (23.976 fps), and every render sits
        // at phase 0.5 of its true frame (100% speed advances exactly one true frame).
        setup_media_on(1001, 24000, 24_000, 1001, 1000, true);
        LOCAL_STEP.with(|l| l.set(Some(1001)));
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        for i in 0..6 {
            let now = 1001 * (100 + i) + 500;
            let rr = run(now, 1001, req);
            let ctx = format!("now {now} {:?}", outcomes(&rr.dg));
            assert_eq!((rr.dg.class, rr.dg.mode), (Some(Class::UnverifiedNoNeighbours), "legacy"), "{ctx}");
            let cur = shown(now);
            assert_eq!(legacy_used(&rr.dg), (Some(cur - 1), Some(cur + 1)), "{ctx}");
            assert_eq!(&rr.times[rr.times.len() - 2..], &[now - 1001, now + 1001]);
            assert_eq!(rr.dg.fd, (Some(1.0), Some(1.0)));
            // Never Rejected: ladder (5) + legacy (2) probes, plus the current checkout once.
            assert_eq!(verdict(req), SourceState::default());
            assert_eq!(rr.co, if i == 0 { 8 } else { 7 }, "{ctx}");
        }
    }

    #[test]
    fn legacy_frame_distance_follows_media_speed() {
        // 23.976 media at 200% on a 23.976 sequence: neighbours +-1001 (adjacent source frames),
        // but the media advances 2 source frames per sequence frame: fd 0.5.
        setup_media_on(1001, 24000, 24_000, 1001, 1000, true);
        LOCAL_STEP.with(|l| l.set(Some(2002)));
        let req = host(premiere_timing(2.0), Some(seq_ticks_23_976()));
        let rr = run(100_100, 1001, req);
        assert_eq!((rr.dg.mode, rr.dg.legacy_fd), ("legacy", Some(neighbors::LegacyFd::Divided)));
        assert_eq!(&rr.times[rr.times.len() - 2..], &[99_099, 101_101]);
        assert_eq!(rr.dg.fd, (Some(0.5), Some(0.5)));
        // Checkbox off (same clip): the 1.0 value, no rule.
        let off = SourceReq { allowed: Err("disabled by parameter"), ..req };
        let rr = run(100_100, 1001, off);
        assert_eq!((rr.dg.fd, rr.dg.legacy_fd), ((Some(1.0), Some(1.0)), None));

        // 23.976 media on a 59.94 sequence (step 1001 / 60000, D_orig = 2502.5): held runs of 2
        // and 3 sequence frames. Every frame of both runs: adjacent source frames, fd 1 (not
        // the alternating H = 2 / 3).
        for (local, want) in [(1001, 1.0), (4004, 1.0 / 1.6)] {
            setup_media_on(1001, 60000, 24_000, 1001, 1000, true);
            LOCAL_STEP.with(|l| l.set(Some(local)));
            let req = host(premiere_timing(local as f64 / 1001.0), Some(seq_ticks_23_976()));
            let mut runs = std::collections::BTreeSet::new();
            for k in 100..110 {
                let now = 1001 * k;
                let rr = run(now, 1001, req);
                let cur = shown(now);
                assert!((rr.dg.d_orig.unwrap() - 2502.5).abs() < 1e-9);
                assert_eq!((rr.dg.mode, rr.dg.legacy_fd), ("legacy", Some(neighbors::LegacyFd::SourceRelative)));
                assert_eq!(legacy_used(&rr.dg), (Some(cur - 1), Some(cur + 1)), "now {now} {:?}", outcomes(&rr.dg));
                assert!(approx(rr.dg.fd, want), "local {local} now {now} {:?}", rr.dg.fd);
                // The legacy H this replaced: 2 or 3 depending on the run.
                let k_prev = rr.dg.probes.iter().filter(|p| p.kind == "lprev").count();
                let k_next = rr.dg.probes.iter().filter(|p| p.kind == "lnext").count();
                runs.insert(k_prev + k_next - 1);
            }
            assert_eq!(runs.into_iter().collect::<Vec<_>>(), vec![2, 3], "both run lengths covered");
        }
    }

    #[test]
    fn clip_start_and_end_with_cold_and_verified_cache() {
        for repeat in [true, false] {
            // Start: now = 10 shows frame 0 (phase 0.05).
            setup_119_88_on_23_976();
            set_repeat_edges(repeat);
            let req = host(premiere_timing(1.0), Some(orig_119_88()));
            let rr = run(10, 1001, req);
            let ctx = format!("repeat {repeat} {:?}", outcomes(&rr.dg));
            assert_eq!((rr.dg.mode, rr.dg.class), ("source", Some(Class::EdgeConfirmed)), "{ctx}");
            assert_eq!(used_frames(&rr.dg), (None, Some(1)), "{ctx}");
            assert_eq!(rr.dg.fd, (None, Some(0.2)));
            assert_eq!(verdict(req), VERIFIED);
            eprintln!("cold clip start (repeat {repeat}): {} checkouts", rr.co);
            // Verified key: no edge test.
            let again = run(10, 1001, req);
            assert_eq!(again.dg.class, Some(Class::Edge));
            assert!(again.dg.probes.iter().all(|p| p.kind != "edge"));
            // End: the last frame (999).
            setup_119_88_on_23_976();
            set_repeat_edges(repeat);
            let req = host(premiere_timing(1.0), Some(orig_119_88()));
            let now = (999.0 * 200.2) as i32 + 100;
            assert_eq!(shown(now), 999);
            let rr = run(now, 1001, req);
            assert_eq!((rr.dg.mode, rr.dg.class), ("source", Some(Class::EdgeConfirmed)), "{:?}", outcomes(&rr.dg));
            assert_eq!(used_frames(&rr.dg), (Some(998), None));
            assert_eq!(run(now, 1001, req).dg.class, Some(Class::Edge));
        }
        // Cold clip start with a true frame 5x longer than D_orig: anomaly, legacy.
        setup_media_on(1001, 24000, 24_000, 1001, 1000, true);
        set_repeat_edges(true);
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        let rr = run(900, 1001, req);
        assert_eq!((rr.dg.class, rr.dg.mode), (Some(Class::Anomaly), "legacy"), "{:?}", outcomes(&rr.dg));
        assert_eq!(outcomes(&rr.dg).iter().find(|p| p.0 == "edge").map(|p| p.3), Some("duplicate"));
        assert_eq!(verdict(req).strikes, 1);
    }

    #[test]
    fn verified_key_strikes_and_recovers() {
        setup_119_88_on_23_976();
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        assert_eq!(run(108_108, 1001, req).dg.state_after, Some(VERIFIED));
        // One anomaly (the same key now serves 59.94 media at phase 0.5: 2 + 1 duplicates),
        // then a consistent frame: still Verified, strikes cleared.
        swap_media(60_000, 1001, 1000);
        let rr = run(108_308, 1001, req);
        assert_eq!((rr.dg.class, rr.dg.dups, rr.dg.mode), (Some(Class::Anomaly), Some((2, 1)), "legacy"));
        assert_eq!(verdict(req), SourceState { verdict: Verdict::Verified, strikes: 1 });
        swap_media(120_000, 1001, 1000);
        assert_eq!(run(108_108, 1001, req).dg.class, Some(Class::Consistent));
        assert_eq!(verdict(req), VERIFIED);
        // Three consecutive anomalies: Rejected, never probed again.
        swap_media(60_000, 1001, 1000);
        for i in 0..3 {
            assert_eq!(run(108_308 + i * 2002, 1001, req).dg.class, Some(Class::Anomaly));
        }
        assert_eq!(verdict(req).verdict, Verdict::Rejected);
        swap_media(120_000, 1001, 1000);
        let rr = run(108_108, 1001, req);
        assert_eq!((rr.dg.class, rr.dg.mode, rr.co), (None, "legacy", 2));
    }

    #[test]
    fn hfr_scaled_time_on_a_coarse_timeline() {
        // 50 fps media on the 10 fps timeline (a = 5): D_orig = 2 native units, so the ladder
        // scales time by m = 4 (D = 8) after a scaled-time verify. now = 70 shows frame 35.
        setup_media(50, 1, 100, true);
        let req = timing(50, 1.0);
        let rr = run(70, STEP, req);
        assert_eq!((rr.dg.mode, rr.dg.class, rr.dg.m), ("source", Some(Class::Consistent), 4), "{:?}", rr.dg.legacy_reason);
        // Native current, scaled verify, prev -2 (frame 34), next +6 (35, dup), +10 (36).
        assert_eq!(rr.times, vec![70, 280, 278, 286, 290]);
        assert_eq!(rr.scales, vec![100, 400, 400, 400, 400]);
        assert_eq!(
            outcomes(&rr.dg),
            vec![("verify", 280, 4, "duplicate"), ("prev", 278, 4, "used"), ("next", 286, 4, "duplicate"), ("next", 290, 4, "used")]
        );
        assert_eq!(rr.dg.fd, (Some(0.2), Some(0.2)));
        assert_eq!(run(70, STEP, req).times, vec![280, 278, 286, 290]);
        // Slowed to 20% (local_time_step 2): a = 1, frame distance 1; further (a = 0.5): still 1.
        for speed in [-0.2, 0.1] {
            setup_media(50, 1, 100, true);
            let rr = run(70, STEP, timing(50, speed));
            assert_eq!((rr.dg.mode, rr.dg.fd), ("source", (Some(1.0), Some(1.0))));
        }
    }

    #[test]
    fn host_ignoring_scaled_time_falls_back_to_legacy() {
        setup_media(50, 1, 100, false);
        let req = timing(50, 1.0);
        let rr = run(70, STEP, req);
        assert_eq!((rr.dg.mode, rr.dg.legacy_reason), ("legacy", Some("host did not honour scaled time")));
        assert_eq!(outcomes(&rr.dg)[0], ("verify", 280, 4, "missing"));
        // Legacy: native k = 1 on both sides (50 fps media, frames 5 apart).
        assert_eq!(rr.times, vec![70, 280, 60, 80]);
        assert_eq!(verdict(req), SourceState::default());
    }

    #[test]
    fn effect_instances_keep_separate_cache_entries() {
        // Two effect instances with identical reported timing (keys differ only by instance id):
        // A holds 119.88 media, B continuously changing input.
        setup_119_88_on_23_976();
        let req_a = host(premiere_timing(1.0), Some(orig_119_88()));
        assert_eq!(run(108_108, 1001, req_a).dg.class, Some(Class::Consistent));
        setup_continuous_23_976();
        let req_b = host(premiere_timing(1.0), Some(orig_119_88()));
        let (ka, kb) = (key_of(req_a), key_of(req_b));
        assert_ne!(ka.instance_id, kb.instance_id);
        assert_eq!(source_cache::Key { instance_id: ka.instance_id, ..kb }, ka);
        assert_eq!(run(10_010, 1001, req_b).dg.class, Some(Class::Continuous));
        assert_eq!((verdict(req_a), verdict(req_b).verdict), (VERIFIED, Verdict::Rejected));
        // A failed instance-id query keys on the sentinel; the rest of the key still applies.
        let unknown = cache_key(&HostTiming { instance_id: None, ..req_b.host.unwrap() }, 1001, 24000, (W, H));
        assert_eq!(unknown, source_cache::Key { instance_id: source_cache::INSTANCE_UNKNOWN, ..kb });
    }

    #[test]
    fn current_checkout_skipped_only_after_a_match() {
        // Matching input: checked out once, skipped afterwards.
        setup_119_88_on_23_976();
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        assert_eq!(run(108_108, 1001, req).dg.cur_outcome, Some("match"));
        let rr = run(109_109, 1001, req);
        assert_eq!(rr.dg.cur_outcome, Some("skipped"));
        assert!(!rr.times.contains(&109_109));
        // Input that differs from the checkout (e.g. the checkout misses an earlier effect):
        // never skipped, and the checked-out frame is what the ladder compares against.
        setup_119_88_on_23_976();
        INPUT_SEED.with(|s| s.set(Some(3)));
        let req = host(premiere_timing(1.0), Some(orig_119_88()));
        for now in [108_108, 109_109, 110_110] {
            let rr = run(now, 1001, req);
            assert_eq!((rr.dg.cur_outcome, rr.times[0]), (Some("mismatch"), now));
            assert_eq!(rr.dg.class, Some(Class::Consistent));
        }
        assert_eq!(source_cache::get(&key_of(req)).input_matches_checkout, Some(false));
        // Unknown instance id: never skipped (the record would be shared between clips).
        setup_119_88_on_23_976();
        let mut req = host(premiere_timing(1.0), Some(orig_119_88()));
        req.host.as_mut().unwrap().instance_id = None;
        for now in [108_108, 109_109] {
            assert_eq!(run(now, 1001, req).dg.cur_outcome, Some("match"));
        }
    }

    #[test]
    fn video_segment_prefix_layout_and_plugin_id_hex() {
        // Exact 22-pointer prefix matching PrSDKVideoSegmentSuite v4..=9.
        assert_eq!(std::mem::size_of::<VideoSegmentPrefix>(), 22 * std::mem::size_of::<*const std::ffi::c_void>());
        // 16-byte binary GUID with zero tail -> 32 lowercase hex chars.
        let mut bin = PrPluginID { m_guid: [0u8; 37] };
        for i in 0..16 {
            bin.m_guid[i] = i as u8;
        }
        assert_eq!(format_plugin_id(&bin), "000102030405060708090a0b0c0d0e0f");
        // 36-char ASCII UUID + NUL -> 32 lowercase hex chars (dashes stripped, no whitespace).
        let mut asc = PrPluginID { m_guid: [0u8; 37] };
        asc.m_guid[..36].copy_from_slice(b"550E8400-e29b-41d4-a716-446655440000");
        assert_eq!(format_plugin_id(&asc), "550e8400e29b41d4a716446655440000");
    }

    #[test]
    fn pause_render_suspends_active_count_and_unwinds_cleanly() {
        // Test thread-local pause nesting and unwind restoration without racing against
        // sibling tests that enter/exit renders concurrently on other threads.
        assert!(!core::par::is_render_paused());
        let _r = core::par::enter_render();
        assert!(core::par::active_renders() >= 1);
        {
            let _p1 = core::par::pause_render();
            assert!(core::par::is_render_paused());
            {
                let _p2 = core::par::pause_render();
                assert!(core::par::is_render_paused());
            }
            assert!(core::par::is_render_paused());
        }
        assert!(!core::par::is_render_paused());
        assert!(core::par::active_renders() >= 1);
        let res = catch_unwind(AssertUnwindSafe(|| {
            let _p = core::par::pause_render();
            assert!(core::par::is_render_paused());
            panic!("simulated checkout unwind");
        }));
        assert!(res.is_err());
        assert!(!core::par::is_render_paused());
        assert!(core::par::active_renders() >= 1);
    }
}
