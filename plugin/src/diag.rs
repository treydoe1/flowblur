//! Opt-in diagnostics log. When the file `/tmp/tblur-debug` exists (checked with one stat per
//! render), one `key=value` line per render and one per global setup are appended to
//! `/tmp/tblur-debug.log`. Logging never fails a render: IO errors are ignored, panics are
//! caught, and a mutex keeps lines from concurrent render threads whole.

use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;

const FLAG_PATH: &str = "/tmp/tblur-debug";
const SEGHASH_FLAG_PATH: &str = "/tmp/tblur-debug-seghash";
const NO_GPU_FLAG_PATH: &str = "/tmp/tblur-no-gpu";
#[cfg(target_os = "macos")]
const GPU_STEPS_PATH: &str = "/tmp/tblur-gpu-steps";
#[cfg(target_os = "macos")]
const GPU_FAULT_PATH: &str = "/tmp/tblur-gpu-fault";
#[cfg(target_os = "macos")]
const GPU_FORCE_PATH: &str = "/tmp/tblur-gpu-force";
#[cfg(target_os = "macos")]
const MAX_FLAG_FILE_BYTES: usize = 16;
const LOG_PATH: &str = "/tmp/tblur-debug.log";

/// Debug-only GPU gate force modes parsed from `/tmp/tblur-gpu-force` in `CreateInstance`
/// when `/tmp/tblur-debug` also exists.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuForceMode {
    None,
    /// Accept every effect node whose operator-owner lookup fails the way adjustment layers do.
    Adjust,
    /// Accept every GPU filter instance regardless of node/PF classification.
    All,
}

#[cfg(target_os = "macos")]
impl GpuForceMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Adjust => "adjust",
            Self::All => "all",
        }
    }
}

/// Debug-only GPU fault injection modes parsed from `/tmp/tblur-gpu-fault` in `CreateInstance`
/// when `/tmp/tblur-debug` also exists.
#[cfg(target_os = "macos")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFaultMode {
    None,
    Error,
    Passthrough,
    Timeout,
}

#[cfg(target_os = "macos")]
impl GpuFaultMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Error => "error",
            Self::Passthrough => "passthrough",
            Self::Timeout => "timeout",
        }
    }
}

static LOCK: Mutex<()> = Mutex::new(());

#[cfg(unix)]
const MAX_LOG_TOKEN_BYTES: usize = 64;

#[cfg(unix)]
pub(crate) fn current_euid() -> u32 {
    unsafe extern "C" {
        fn geteuid() -> u32;
    }
    // SAFETY: POSIX `geteuid(2)` takes no arguments and never fails.
    unsafe { geteuid() }
}

/// Pure check that a file's owner UID and POSIX mode bits are trusted: owned by `expected_uid`
/// and neither group-writable (`0o020`) nor world-writable (`0o002`).
#[cfg(unix)]
pub(crate) fn is_trusted_owner_and_mode(uid: u32, mode: u32, expected_uid: u32) -> bool {
    uid == expected_uid && (mode & 0o022) == 0
}

#[cfg(unix)]
fn is_trusted_regular_meta(meta: &std::fs::Metadata, expected_uid: u32) -> bool {
    use std::os::unix::fs::MetadataExt;
    meta.file_type().is_file() && is_trusted_owner_and_mode(meta.uid(), meta.mode(), expected_uid)
}

/// True when `path` is a regular file (never following symlinks) owned by the current user and
/// not group- or world-writable.
pub fn flag_enabled_from(path: &std::path::Path) -> bool {
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(unix)]
        {
            let Ok(meta) = std::fs::symlink_metadata(path) else {
                return false;
            };
            is_trusted_regular_meta(&meta, current_euid())
        }
        #[cfg(not(unix))]
        {
            std::fs::symlink_metadata(path)
                .map(|m| m.file_type().is_file())
                .unwrap_or(false)
        }
    }))
    .unwrap_or(false)
}

/// True when `/tmp/tblur-debug` is a trusted regular file owned by the current user and not
/// group- or world-writable.
pub fn enabled() -> bool {
    flag_enabled_from(std::path::Path::new(FLAG_PATH))
}

/// True when `/tmp/tblur-debug-seghash` is a trusted regular file owned by the current user
/// and not group- or world-writable.
pub fn seghash_enabled() -> bool {
    flag_enabled_from(std::path::Path::new(SEGHASH_FLAG_PATH))
}

/// True when the GPU kill switch file at `path` exists. Intentionally permissive because it only
/// disables the GPU path (falling back to CPU).
#[cfg(target_os = "macos")]
pub fn gpu_disabled_from(path: &std::path::Path) -> bool {
    catch_unwind(|| std::fs::metadata(path).is_ok()).unwrap_or(false)
}

/// True when the GPU kill switch file `/tmp/tblur-no-gpu` exists. Intentionally permissive
/// because it only disables the GPU path (falling back to CPU).
#[cfg(target_os = "macos")]
pub fn gpu_disabled() -> bool {
    gpu_disabled_from(std::path::Path::new(NO_GPU_FLAG_PATH))
}

/// Safely reads a small debug flag file: see `read_regular_file_capped`, with a cap of
/// `MAX_FLAG_FILE_BYTES` (16) bytes and UTF-8 content.
#[cfg(target_os = "macos")]
fn read_regular_flag_file(path: &std::path::Path) -> Option<String> {
    String::from_utf8(read_regular_file_capped(path, MAX_FLAG_FILE_BYTES)?).ok()
}

/// Safely reads a small file: requires a regular file (never follows symlinks, never blocks on a
/// FIFO/named pipe, rejects directories/devices/sockets), owned by the current user and not
/// group- or world-writable, no larger than `max_bytes`, and reads at most `max_bytes` bytes.
#[cfg(target_os = "macos")]
pub(crate) fn read_regular_file_capped(path: &std::path::Path, max_bytes: usize) -> Option<Vec<u8>> {
    use std::io::Read;
    use std::os::unix::fs::OpenOptionsExt;

    // Darwin `<sys/fcntl.h>` flags: never block on a FIFO and never follow symlinks.
    const O_NONBLOCK: i32 = 0x0004;
    const O_NOFOLLOW: i32 = 0x0100;

    let uid = current_euid();

    // 1. Pre-check with `lstat` before `open` so symlinks/FIFOs/untrusted files are rejected immediately.
    let lmeta = std::fs::symlink_metadata(path).ok()?;
    if !is_trusted_regular_meta(&lmeta, uid) || lmeta.len() > max_bytes as u64 {
        return None;
    }

    // 2. Open with `O_NOFOLLOW | O_NONBLOCK` and re-check with `fstat` to close any TOCTOU window.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(path)
        .ok()?;
    let fmeta = file.metadata().ok()?;
    if !is_trusted_regular_meta(&fmeta, uid) || fmeta.len() > max_bytes as u64 {
        return None;
    }

    // 3. Bounded read of at most `max_bytes` bytes.
    let mut buf = Vec::with_capacity(max_bytes);
    file.take(max_bytes as u64).read_to_end(&mut buf).ok()?;
    Some(buf)
}

/// Resolves the GPU dependency step count from `steps_path` when `debug_enabled` is true.
/// Accepts a regular file (no symlinks, <= 16 bytes) containing a single ASCII digit `'1'..='4'`
/// (optional surrounding whitespace); otherwise returns `(default_steps, false)`. Ignored whenever
/// `debug_enabled` is false.
#[cfg(target_os = "macos")]
pub fn resolve_gpu_steps_from(
    debug_enabled: bool,
    steps_path: &std::path::Path,
    default_steps: u32,
) -> (u32, bool) {
    if !debug_enabled {
        return (default_steps, false);
    }
    let parsed = catch_unwind(AssertUnwindSafe(|| {
        let content = read_regular_flag_file(steps_path)?;
        let trimmed = content.trim();
        if trimmed.len() == 1 {
            let b = trimmed.as_bytes()[0];
            if (b'1'..=b'4').contains(&b) {
                return Some((b - b'0') as u32);
            }
        }
        None
    }));
    match parsed {
        Ok(Some(v)) => (v, true),
        _ => (default_steps, false),
    }
}

/// Reads `/tmp/tblur-gpu-steps` when `debug_enabled` is true. Called in `CreateInstance` only.
#[cfg(target_os = "macos")]
pub fn resolve_gpu_steps(debug_enabled: bool, default_steps: u32) -> (u32, bool) {
    resolve_gpu_steps_from(debug_enabled, std::path::Path::new(GPU_STEPS_PATH), default_steps)
}

/// Resolves the debug-only GPU fault mode from `fault_path` when `debug_enabled` is true.
/// Accepts a regular file (no symlinks, <= 16 bytes) containing one word: `error`, `passthrough`,
/// or `timeout` (optional surrounding whitespace). Ignored whenever `debug_enabled` is false.
#[cfg(target_os = "macos")]
pub fn resolve_gpu_fault_from(debug_enabled: bool, fault_path: &std::path::Path) -> GpuFaultMode {
    if !debug_enabled {
        return GpuFaultMode::None;
    }
    let parsed = catch_unwind(AssertUnwindSafe(|| {
        let content = read_regular_flag_file(fault_path)?;
        match content.trim() {
            "error" => Some(GpuFaultMode::Error),
            "passthrough" => Some(GpuFaultMode::Passthrough),
            "timeout" => Some(GpuFaultMode::Timeout),
            _ => None,
        }
    }));
    match parsed {
        Ok(Some(mode)) => mode,
        _ => GpuFaultMode::None,
    }
}

/// Reads `/tmp/tblur-gpu-fault` when `debug_enabled` is true. Called in `CreateInstance` only.
#[cfg(target_os = "macos")]
pub fn resolve_gpu_fault(debug_enabled: bool) -> GpuFaultMode {
    resolve_gpu_fault_from(debug_enabled, std::path::Path::new(GPU_FAULT_PATH))
}

/// Resolves the debug-only GPU force mode from `force_path` when `debug_enabled` is true.
/// Accepts a regular file (no symlinks, <= 16 bytes) containing one word: `adjust` or `all`
/// (optional surrounding whitespace). Ignored whenever `debug_enabled` is false.
#[cfg(target_os = "macos")]
pub fn resolve_gpu_force_from(debug_enabled: bool, force_path: &std::path::Path) -> GpuForceMode {
    if !debug_enabled {
        return GpuForceMode::None;
    }
    let parsed = catch_unwind(AssertUnwindSafe(|| {
        let content = read_regular_flag_file(force_path)?;
        match content.trim() {
            "adjust" => Some(GpuForceMode::Adjust),
            "all" => Some(GpuForceMode::All),
            _ => None,
        }
    }));
    match parsed {
        Ok(Some(mode)) => mode,
        _ => GpuForceMode::None,
    }
}

/// Reads `/tmp/tblur-gpu-force` when `debug_enabled` is true. Called in `CreateInstance` only.
#[cfg(target_os = "macos")]
pub fn resolve_gpu_force(debug_enabled: bool) -> GpuForceMode {
    resolve_gpu_force_from(debug_enabled, std::path::Path::new(GPU_FORCE_PATH))
}

/// Sanitizes an arbitrary host string into a single whitespace-free ASCII log token truncated to
/// at most 64 bytes.
#[cfg(target_os = "macos")]
pub fn sanitize_token(s: &str) -> String {
    if s.is_empty() {
        return "empty".to_string();
    }
    s.chars()
        .take(MAX_LOG_TOKEN_BYTES)
        .map(|c| {
            if c.is_ascii_graphic() && c != '=' && c != '"' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Appends one line (timestamp and thread prefixed). Opens the file in append mode per line.
pub fn append(line: &str) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0);
        let tid = format!("{:?}", std::thread::current().id());
        let full = format!("ts={ts:.3} tid={} {line}\n", tid.trim_start_matches("ThreadId(").trim_end_matches(')'));
        let _guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(LOG_PATH) {
            let _ = f.write_all(full.as_bytes());
        }
    }));
}

/// Formats an optional value as `n/a` when absent.
pub fn opt<T: std::fmt::Display>(v: Option<T>) -> String {
    v.map_or_else(|| "n/a".to_string(), |v| v.to_string())
}

/// Formats an optional float with fixed precision, `n/a` when absent.
pub fn optf(v: Option<f64>, prec: usize) -> String {
    v.map_or_else(|| "n/a".to_string(), |v| format!("{v:.prec$}"))
}
