//! Update notice. At most once every 24 hours, when the host loads the plug-in, a background
//! thread asks GitHub which release is the latest and stores only its numeric version. The effect
//! then shows "x.y.z is available" in its About text and on its last parameter row. Nothing is
//! ever downloaded or installed.
//!
//! * The network request runs in the system `curl` as a child process, so no network or TLS
//!   code runs inside the host's process. The child is started with `posix_spawn` (absolute
//!   path, no shell, empty environment, signals reset, and no descriptor inherited from the host
//!   except the three it is given). It asks for `releases/latest`, which answers with a redirect
//!   to `releases/tag/<tag>`; only the redirect target is read and it is never followed.
//! * The reply is accepted only if it is exactly the expected tag URL with a
//!   `MAJOR.MINOR.PATCH` version; anything else means "no information". Only the parsed numbers
//!   are stored and shown.
//! * State lives in `~/Library/Application Support/TBlur/update-check` (attempt time in Unix
//!   seconds and the latest version seen). The attempt time is written before `curl` starts,
//!   which makes it unlikely (not impossible) that several host processes loading the plug-in
//!   together all check.
//! * The file `~/Library/Application Support/TBlur/no-update-check` turns all of this off: no
//!   child process, no request, no notice.
//! * Every failure is silent; with the diagnostics log enabled, one line per attempt is logged.

use std::ffi::{CString, c_char, c_int, c_short, c_uint};
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock, PoisonError};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime};

use crate::VERSION;
use crate::diag;

/// The page a click on the notice opens.
pub const RELEASES_URL: &str = "https://github.com/treydoe1/tblur/releases";
/// GitHub answers this with a redirect to the latest release's tag page.
const LATEST_URL: &str = "https://github.com/treydoe1/tblur/releases/latest";
/// The only redirect target accepted, followed by the tag.
const TAG_URL_PREFIX: &str = "https://github.com/treydoe1/tblur/releases/tag/";

const CURL: &str = "/usr/bin/curl";
const OPEN: &str = "/usr/bin/open";

const STATE_FILE: &str = "update-check";
const OFF_SWITCH_FILE: &str = "no-update-check";
const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;
/// The largest valid state file is 57 bytes.
const MAX_STATE_BYTES: usize = 128;
/// The largest valid redirect URL is 68 bytes.
const MAX_OUTPUT_BYTES: usize = 1024;
/// Hard wall-clock limit for the `curl` child; its own `--max-time` is 8 s.
const CURL_TIME_LIMIT: Duration = Duration::from_secs(12);
const OPEN_TIME_LIMIT: Duration = Duration::from_secs(10);
/// At most one release page opens per this interval, however often the host reports a click.
const OPEN_INTERVAL_MS: u64 = 3000;
const POLL_INTERVAL: Duration = Duration::from_millis(20);
/// Temporary and output files older than this are left over from an ended process.
const LEFTOVER_AGE: Duration = Duration::from_secs(60 * 60);
/// The leftover sweep looks at no more directory entries than this.
const MAX_SWEEP_ENTRIES: usize = 1000;
const MAX_VERSION_DIGITS: usize = 6;

/// A release version. Ordering is numeric, part by part.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

/// Parses exactly `MAJOR.MINOR.PATCH`, each part 1 to 6 ASCII digits, with nothing else.
fn parse_version(s: &[u8]) -> Option<Version> {
    let mut parts = [0u32; 3];
    let mut fields = s.split(|&b| b == b'.');
    for part in &mut parts {
        let field = fields.next()?;
        if field.is_empty() || field.len() > MAX_VERSION_DIGITS || !field.iter().all(u8::is_ascii_digit) {
            return None;
        }
        *part = field.iter().fold(0u32, |acc, &d| acc * 10 + u32::from(d - b'0'));
    }
    if fields.next().is_some() {
        return None;
    }
    Some(Version { major: parts[0], minor: parts[1], patch: parts[2] })
}

/// Parses `curl`'s output: exactly `TAG_URL_PREFIX`, an optional `v`, and a version.
fn parse_redirect(out: &[u8]) -> Option<Version> {
    let tag = out.strip_prefix(TAG_URL_PREFIX.as_bytes())?;
    parse_version(tag.strip_prefix(b"v").unwrap_or(tag))
}

fn running_version() -> Option<Version> {
    parse_version(VERSION.as_bytes())
}

// ───────────────────────────── stored state ─────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct State {
    /// Time of the last attempt, in Unix seconds.
    checked: u64,
    /// The latest version seen, if any attempt found one.
    latest: Option<Version>,
}

fn format_state(s: &State) -> String {
    let latest = s.latest.map(|v| v.to_string()).unwrap_or_default();
    format!("checked={}\nlatest={latest}\n", s.checked)
}

/// Accepts the form `format_state` writes: `checked=<1 to 20 digits>\nlatest=<version or
/// nothing>\n`. Leading zeros in the numbers are accepted.
fn parse_state(bytes: &[u8]) -> Option<State> {
    let rest = bytes.strip_prefix(b"checked=")?;
    let nl = rest.iter().position(|&b| b == b'\n')?;
    let (digits, rest) = (&rest[..nl], &rest[nl + 1..]);
    if digits.is_empty() || digits.len() > 20 || !digits.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let checked = std::str::from_utf8(digits).ok()?.parse::<u64>().ok()?;
    let latest = rest.strip_prefix(b"latest=")?.strip_suffix(b"\n")?;
    let latest = if latest.is_empty() { None } else { Some(parse_version(latest)?) };
    Some(State { checked, latest })
}

/// True when the state directory is a real directory (not a symlink) owned by the effective
/// user and not writable by group or others.
fn is_trusted_dir(dir: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    std::fs::symlink_metadata(dir).is_ok_and(|m| {
        m.file_type().is_dir() && diag::is_trusted_owner_and_mode(m.uid(), m.mode(), diag::current_euid())
    })
}

/// Creates the state directory (one level, mode 0700) if it is missing, then checks it.
fn ensure_dir(dir: &Path) -> bool {
    use std::os::unix::fs::DirBuilderExt;
    if let Err(e) = std::fs::DirBuilder::new().mode(0o700).create(dir)
        && e.kind() != std::io::ErrorKind::AlreadyExists
    {
        return false;
    }
    is_trusted_dir(dir)
}

/// Any entry at the off-switch path, of any type, turns the feature off. So does any error
/// other than "not found" while looking for it, since the switch might then exist unseen.
fn is_off(dir: &Path) -> bool {
    match std::fs::symlink_metadata(dir.join(OFF_SWITCH_FILE)) {
        Ok(_) => true,
        Err(e) => e.kind() != std::io::ErrorKind::NotFound,
    }
}

/// Reads the state with the flag-file safety rules; unreadable, unsafe or malformed is `None`.
fn read_state(dir: &Path) -> Option<State> {
    if !is_trusted_dir(dir) {
        return None;
    }
    parse_state(&diag::read_regular_file_capped(&dir.join(STATE_FILE), MAX_STATE_BYTES)?)
}

/// A fresh name for a temporary (`tmp`) or output (`out`) file in the state directory:
/// `.update-check.<pid>.<seq>.<ext>`.
fn temp_name(ext: &str) -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    format!(".{STATE_FILE}.{}.{}.{ext}", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed))
}

/// Removes temporary and output files that a host process which ended mid-check left behind:
/// names of the `temp_name` form, not directories, last modified more than `LEFTOVER_AGE`
/// before `now`. Fresh ones may belong to a check running in another process and are kept.
fn sweep_leftovers(dir: &Path, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let prefix = format!(".{STATE_FILE}.");
    for entry in entries.take(MAX_SWEEP_ENTRIES).flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(&prefix) || !(name.ends_with(".tmp") || name.ends_with(".out")) {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = std::fs::symlink_metadata(&path) else { continue };
        let old = meta.modified().ok().and_then(|m| now.duration_since(m).ok()).is_some_and(|age| age > LEFTOVER_AGE);
        if !meta.is_dir() && old {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Writes the state atomically: a temporary file created exclusively with mode 0600 in the same
/// directory, then renamed over the state file. The temporary file is removed on any failure.
fn write_state(dir: &Path, state: &State) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;

    let tmp = dir.join(temp_name("tmp"));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&tmp)?;
    // From here on the temporary file is ours, so it is removed if anything fails. There is no
    // fsync (a full drive flush on macOS): a torn or empty file after a power cut fails to
    // parse and only causes one more check.
    let result = file.write_all(format_state(state).as_bytes()).and_then(|()| {
        drop(file);
        std::fs::rename(&tmp, dir.join(STATE_FILE))
    });
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

// ───────────────────────────── the check ─────────────────────────────

/// Everything the check depends on, so tests can run it without the network, the real clock or
/// the real home directory.
struct Ctx {
    /// The state directory; `None` (home directory not resolved) turns the feature off.
    dir: Option<PathBuf>,
    now: Box<dyn Fn() -> u64 + Send>,
    fetch: Box<dyn Fn() -> Option<Version> + Send>,
    /// Receives a version the check found (and stored), so host threads see it without
    /// reading the state file again.
    publish: Box<dyn Fn(Version) + Send>,
    log: Box<dyn Fn(&str) + Send>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Outcome {
    NoHome,
    Off,
    /// The state directory could not be created or is not trusted.
    BadDir,
    /// The last attempt is less than 24 hours old.
    Recent,
    /// The attempt time could not be recorded, so no request was made.
    WriteFailed,
    /// A request was made; `Some` when it returned a version.
    Fetched(Option<Version>),
}

impl Outcome {
    fn log_line(self) -> String {
        let result = match self {
            Outcome::NoHome => "no_home".to_string(),
            Outcome::Off => "off".to_string(),
            Outcome::BadDir => "bad_dir".to_string(),
            Outcome::Recent => "recent".to_string(),
            Outcome::WriteFailed => "write_failed".to_string(),
            Outcome::Fetched(None) => "no_information".to_string(),
            Outcome::Fetched(Some(v)) => format!("latest={v}"),
        };
        format!("event=update_check version={VERSION} result={result}")
    }
}

fn check_once(ctx: &Ctx) -> Outcome {
    let Some(dir) = ctx.dir.as_deref() else {
        return Outcome::NoHome;
    };
    if is_off(dir) {
        return Outcome::Off;
    }
    if !ensure_dir(dir) {
        return Outcome::BadDir;
    }
    let previous = read_state(dir).unwrap_or_default();
    let now = (ctx.now)();
    // A stored time in the future counts as stale.
    if previous.checked <= now && now - previous.checked < CHECK_INTERVAL_SECS {
        return Outcome::Recent;
    }
    sweep_leftovers(dir, SystemTime::now());
    // Record the attempt before the request, so other host processes most likely skip theirs.
    if write_state(dir, &State { checked: now, latest: previous.latest }).is_err() {
        return Outcome::WriteFailed;
    }
    let fetched = (ctx.fetch)();
    if let Some(latest) = fetched {
        let _ = write_state(dir, &State { checked: now, latest: Some(latest) });
    }
    Outcome::Fetched(fetched)
}

/// The stored version when it is strictly newer than `running` and the feature is on.
fn available_update_in(dir: Option<&Path>, running: Option<Version>) -> Option<Version> {
    let dir = dir?;
    if is_off(dir) {
        return None;
    }
    let latest = read_state(dir)?.latest?;
    (latest > running?).then_some(latest)
}

/// The update notice as host threads see it, kept per process: filled from the state file by
/// the first caller, then read without touching the file system, and replaced by the check
/// thread when it finds a version.
struct NoticeCache(Mutex<Option<Option<Version>>>);

impl NoticeCache {
    const fn new() -> Self {
        NoticeCache(Mutex::new(None))
    }

    /// The cached notice; the first call fills it with `fill`.
    fn get(&self, fill: impl FnOnce() -> Option<Version>) -> Option<Version> {
        *self.0.lock().unwrap_or_else(PoisonError::into_inner).get_or_insert_with(fill)
    }

    /// Replaces the notice with what the state file now says: `latest` when it is strictly
    /// newer than `running`, otherwise none. The check only gets this far with the off switch
    /// clear.
    fn publish(&self, latest: Version, running: Option<Version>) {
        let notice = running.filter(|r| latest > *r).map(|_| latest);
        *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Some(notice);
    }
}

static NOTICE: NoticeCache = NoticeCache::new();

/// Runs `check_once` on a named background thread, at most once per `once`.
fn start_with(once: &Once, make_ctx: impl FnOnce() -> Ctx + Send + 'static) -> Option<std::thread::JoinHandle<()>> {
    let mut handle = None;
    once.call_once(|| {
        handle = std::thread::Builder::new()
            .name("tblur-update-check".to_string())
            .spawn(move || {
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    let ctx = make_ctx();
                    let outcome = check_once(&ctx);
                    if let Outcome::Fetched(Some(latest)) = outcome {
                        (ctx.publish)(latest);
                    }
                    (ctx.log)(&outcome.log_line());
                }));
            })
            .ok();
    });
    handle
}

// ───────────────────────────── the real environment ─────────────────────────────

/// The user's home directory: absolute, not `/`, a directory owned by the effective user.
/// Tests never resolve it, so they cannot touch the real home directory.
fn home_dir() -> Option<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    if cfg!(test) {
        return None;
    }
    let home = std::env::home_dir()?;
    if !home.is_absolute() || home == Path::new("/") {
        return None;
    }
    let meta = std::fs::metadata(&home).ok()?;
    (meta.is_dir() && meta.uid() == diag::current_euid()).then_some(home)
}

/// The state directory, resolved once per process by whichever thread asks first.
fn state_dir() -> Option<PathBuf> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| Some(home_dir()?.join("Library").join("Application Support").join("TBlur"))).clone()
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// A child to start: absolute program path, arguments after `argv[0]`, and the complete
/// environment (nothing is taken from the host's).
#[derive(Clone, Debug, PartialEq, Eq)]
struct Spawn {
    program: String,
    args: Vec<String>,
    env: Vec<String>,
}

fn curl_spawn() -> Spawn {
    let args = [
        "-q",
        "--proto", "=https",
        "--proto-redir", "=https",
        "--tlsv1.2",
        "--connect-timeout", "5",
        "--max-time", "8",
        "--max-redirs", "0",
        "-sS",
        "-o", "/dev/null",
        "-w", "%{redirect_url}",
        "-A", &format!("TBlur/{VERSION}"),
        LATEST_URL,
    ];
    Spawn { program: CURL.to_string(), args: args.iter().map(|a| a.to_string()).collect(), env: Vec::new() }
}

/// `open` gets only `HOME` (when resolved) and a fixed `PATH`.
fn open_spawn(home: Option<&Path>) -> Spawn {
    let mut env: Vec<String> = home.and_then(Path::to_str).map(|h| format!("HOME={h}")).into_iter().collect();
    env.push("PATH=/usr/bin:/bin".to_string());
    Spawn { program: OPEN.to_string(), args: vec![RELEASES_URL.to_string()], env }
}

/// Runs `curl` with its output file in the (already checked) state directory `dir`.
fn fetch_latest(dir: &Path) -> Option<Version> {
    parse_redirect(&run_spawned(&curl_spawn(), Some(dir), CURL_TIME_LIMIT, MAX_OUTPUT_BYTES, &mut RealChildControl).usable_output()?)
}

fn real_ctx() -> Ctx {
    let dir = state_dir();
    Ctx {
        dir: dir.clone(),
        now: Box::new(unix_now),
        fetch: {
            let dir = dir.clone();
            Box::new(move || fetch_latest(dir.as_deref()?))
        },
        publish: Box::new(|latest| NOTICE.publish(latest, running_version())),
        log: Box::new(|line| {
            if diag::enabled() {
                diag::append(line);
            }
        }),
    }
}

/// Starts the once-a-day check in the background and returns immediately. Runs at most once per
/// process; does nothing when the off switch is set or the last attempt is under 24 hours old.
pub fn start_background_check() {
    static ONCE: Once = Once::new();
    let _ = catch_unwind(|| {
        // The off switch is checked before any thread or child process exists.
        if state_dir().is_none_or(|d| is_off(&d)) {
            return;
        }
        start_with(&ONCE, real_ctx);
    });
}

/// The newer release found by a check, if any. The first call in a process reads the small
/// state file (and the off switch); later calls only read the per-process cache, which the
/// check thread updates.
pub fn available_update() -> Option<Version> {
    catch_unwind(|| NOTICE.get(|| available_update_in(state_dir().as_deref(), running_version()))).ok().flatten()
}

/// Records an open at `now_ms` (monotonic milliseconds) and returns true, unless the last
/// recorded open is less than `OPEN_INTERVAL_MS` earlier. `last` holds the last open time plus
/// one, so 0 means "never".
fn allow_open(last: &AtomicU64, now_ms: u64) -> bool {
    let mut prev = last.load(Ordering::Acquire);
    loop {
        if prev != 0 && now_ms < (prev - 1).saturating_add(OPEN_INTERVAL_MS) {
            return false;
        }
        match last.compare_exchange_weak(prev, now_ms.saturating_add(1), Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return true,
            Err(p) => prev = p,
        }
    }
}

/// Milliseconds on a monotonic clock that starts at the first call.
fn monotonic_ms() -> u64 {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    u64::try_from(START.get_or_init(Instant::now).elapsed().as_millis()).unwrap_or(u64::MAX)
}

/// Opens `RELEASES_URL` in the default browser. Called only from an explicit click, and at most
/// once every 3 seconds per process; calls inside that window do nothing. The `open` child is
/// waited for on a background thread.
pub fn open_releases_page() {
    static LAST_OPEN: AtomicU64 = AtomicU64::new(0);
    let _ = catch_unwind(|| {
        if !allow_open(&LAST_OPEN, monotonic_ms()) {
            return;
        }
        let _ = std::thread::Builder::new().name("tblur-open-releases".to_string()).spawn(|| {
            let _ = catch_unwind(|| {
                run_spawned(&open_spawn(home_dir().as_deref()), None, OPEN_TIME_LIMIT, 0, &mut RealChildControl)
            });
        });
    });
}

// ───────────────────────────── child processes ─────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RunEnd {
    SpawnFailed,
    /// The child exited; the exit code, or `None` if a signal ended it.
    Exited(Option<i32>),
    /// The time limit passed; the child was sent SIGKILL and waited for.
    TimedOut,
    /// The child had already been reaped by someone else (the host ignores SIGCHLD or calls
    /// `waitpid(-1)`), so its exit status is unknown.
    ReapedElsewhere,
    /// Waiting failed for another reason. No signal was sent.
    WaitFailed,
}

#[derive(Debug)]
struct RunResult {
    /// Read by the tests, which check that the child was reaped.
    #[cfg_attr(not(test), allow(dead_code))]
    pid: Option<libc::pid_t>,
    end: RunEnd,
    /// The child's standard output, or `None` when there was no output file, or it was over
    /// the cap or failed the safety checks.
    output: Option<Vec<u8>>,
}

impl RunResult {
    /// The output of a child that exited with status 0, or that was reaped by someone else
    /// before its status could be read (the parser validates the output on its own). A
    /// non-zero exit, a signal, a timeout, another wait error or an over-cap output is
    /// "no information".
    fn usable_output(self) -> Option<Vec<u8>> {
        match self.end {
            RunEnd::Exited(Some(0)) | RunEnd::ReapedElsewhere => self.output,
            _ => None,
        }
    }
}

/// One `waitpid` result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Waited {
    Running,
    /// The child ended and was reaped; the raw wait status.
    Ended(c_int),
    /// `waitpid` failed with this errno (never EINTR, which is retried).
    Error(c_int),
}

/// Waiting for and signalling a child, so tests can stand in for a host that reaps children
/// itself and can see which signals are sent.
trait ChildControl {
    fn wait(&mut self, pid: libc::pid_t, block: bool) -> Waited;
    fn kill(&mut self, pid: libc::pid_t);
}

struct RealChildControl;

impl ChildControl for RealChildControl {
    fn wait(&mut self, pid: libc::pid_t, block: bool) -> Waited {
        waitpid_once(pid, block)
    }
    fn kill(&mut self, pid: libc::pid_t) {
        if !is_child_pid(pid) {
            return;
        }
        // SAFETY: `kill(2)` takes plain integers and touches no memory of ours.
        let _ = unsafe { libc::kill(pid, libc::SIGKILL) };
    }
}

/// True for a pid that names one process. 0 and negative values would address a process group
/// (or every process the host may signal) in `kill` and any child in `waitpid`.
fn is_child_pid(pid: libc::pid_t) -> bool {
    pid > 0
}

/// `waitpid(pid, &status, block ? 0 : WNOHANG)`, retried on EINTR. A pid of 0 or less is
/// never passed on; it is reported as EINVAL, which counts as a failed wait (not ECHILD).
fn waitpid_once(pid: libc::pid_t, block: bool) -> Waited {
    if !is_child_pid(pid) {
        return Waited::Error(libc::EINVAL);
    }
    let options = if block { 0 } else { libc::WNOHANG };
    loop {
        let mut status: c_int = 0;
        // SAFETY: `status` is a valid, writable `int` for the duration of the call.
        let r = unsafe { libc::waitpid(pid, &mut status, options) };
        if r == pid {
            return Waited::Ended(status);
        }
        if r == 0 {
            return Waited::Running;
        }
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        if errno != libc::EINTR {
            return Waited::Error(errno);
        }
    }
}

/// The exit code of a raw wait status, or `None` when a signal ended the child.
fn exit_code(status: c_int) -> Option<i32> {
    libc::WIFEXITED(status).then(|| libc::WEXITSTATUS(status))
}

// Mach task exception ports. A child otherwise inherits the host's (such as its crash
// reporter's), which would then be told about a crash of `curl` or `open`. The child's ports
// are set to the null port for every exception type, so its exceptions go to the system's
// default handling instead. Values from the SDK's <mach/exception_types.h>,
// <mach/arm/thread_status.h> and <mach/port.h>; the declaration from <spawn.h>.
const EXC_MASK_ALL: c_uint = (1 << 1) // EXC_BAD_ACCESS
    | (1 << 2) // EXC_BAD_INSTRUCTION
    | (1 << 3) // EXC_ARITHMETIC
    | (1 << 4) // EXC_EMULATION
    | (1 << 5) // EXC_SOFTWARE
    | (1 << 6) // EXC_BREAKPOINT
    | (1 << 7) // EXC_SYSCALL
    | (1 << 8) // EXC_MACH_SYSCALL
    | (1 << 9) // EXC_RPC_ALERT
    | (1 << 11) // EXC_RESOURCE
    | (1 << 12); // EXC_GUARD (EXC_MASK_MACHINE is 0)
/// `EXC_MASK_ALL` leaves out the two types crash reporters usually register for.
const CHILD_EXC_MASK: c_uint = EXC_MASK_ALL | (1 << 10) /* EXC_CRASH */ | (1 << 13) /* EXC_CORPSE_NOTIFY */;
const MACH_PORT_NULL: libc::mach_port_t = 0;
const EXCEPTION_DEFAULT: c_int = 1;
#[cfg(target_arch = "aarch64")]
const THREAD_STATE_NONE: c_int = 5;
#[cfg(target_arch = "x86_64")]
const THREAD_STATE_NONE: c_int = 13;

unsafe extern "C" {
    /// `int posix_spawnattr_setexceptionports_np(posix_spawnattr_t *, exception_mask_t,
    /// mach_port_t, exception_behavior_t, thread_state_flavor_t)`; the libc crate lacks it.
    fn posix_spawnattr_setexceptionports_np(
        attr: *mut libc::posix_spawnattr_t,
        mask: c_uint,
        new_port: libc::mach_port_t,
        behavior: c_int,
        flavor: c_int,
    ) -> c_int;
}

/// A `posix_spawnattr_t`, destroyed on drop.
struct SpawnAttr(libc::posix_spawnattr_t);

impl SpawnAttr {
    /// Close-on-exec for every descriptor not set up by the file actions, every signal reset to
    /// its default action, an empty signal mask, and no task exception ports.
    fn new() -> Option<Self> {
        let mut raw: libc::posix_spawnattr_t = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; on success it owns an attribute object, which
        // `Drop` destroys exactly once.
        if unsafe { libc::posix_spawnattr_init(&mut raw) } != 0 {
            return None;
        }
        let mut attr = SpawnAttr(raw);
        let mut all: libc::sigset_t = 0;
        let mut none: libc::sigset_t = 0;
        // SAFETY: both point at local `sigset_t` values for the duration of the calls.
        unsafe {
            libc::sigfillset(&mut all);
            libc::sigemptyset(&mut none);
        }
        let flags = (libc::POSIX_SPAWN_CLOEXEC_DEFAULT | libc::POSIX_SPAWN_SETSIGDEF | libc::POSIX_SPAWN_SETSIGMASK) as c_short;
        // SAFETY: `attr.0` was initialised above; the set pointers are valid for each call and
        // the values are copied into the attribute object. The exception-port call passes plain
        // integers (a null port).
        let ok = unsafe {
            libc::posix_spawnattr_setflags(&mut attr.0, flags) == 0
                && libc::posix_spawnattr_setsigdefault(&mut attr.0, &all) == 0
                && libc::posix_spawnattr_setsigmask(&mut attr.0, &none) == 0
                && posix_spawnattr_setexceptionports_np(&mut attr.0, CHILD_EXC_MASK, MACH_PORT_NULL, EXCEPTION_DEFAULT, THREAD_STATE_NONE) == 0
        };
        ok.then_some(attr)
    }
}

impl Drop for SpawnAttr {
    fn drop(&mut self) {
        // SAFETY: initialised by `posix_spawnattr_init` and destroyed only here.
        unsafe { libc::posix_spawnattr_destroy(&mut self.0) };
    }
}

/// A `posix_spawn_file_actions_t`, destroyed on drop.
struct FileActions(libc::posix_spawn_file_actions_t);

impl FileActions {
    fn new() -> Option<Self> {
        let mut raw: libc::posix_spawn_file_actions_t = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out-pointer; on success it owns a file-actions object, which
        // `Drop` destroys exactly once.
        (unsafe { libc::posix_spawn_file_actions_init(&mut raw) } == 0).then_some(FileActions(raw))
    }

    /// Makes the child open `path` as descriptor `fd`.
    fn open(&mut self, fd: c_int, path: &CString, flags: c_int, mode: libc::mode_t) -> bool {
        // SAFETY: `self.0` is initialised and `path` is a NUL-terminated string that the caller
        // keeps alive until after `posix_spawn`.
        unsafe { libc::posix_spawn_file_actions_addopen(&mut self.0, fd, path.as_ptr(), flags, mode) == 0 }
    }
}

impl Drop for FileActions {
    fn drop(&mut self) {
        // SAFETY: initialised by `posix_spawn_file_actions_init` and destroyed only here.
        unsafe { libc::posix_spawn_file_actions_destroy(&mut self.0) };
    }
}

/// Starts `spawn` with `posix_spawn`: fd 0 from `/dev/null`, fd 2 to `/dev/null`, fd 1 to
/// `/dev/null` or, with `out`, a new file created by the child (exclusive, no-follow, mode
/// 0600) at `out`. Nothing else is inherited. Returns the child's pid.
fn spawn_child(spawn: &Spawn, out: Option<&Path>) -> Option<libc::pid_t> {
    use std::os::unix::ffi::OsStrExt;
    let program = CString::new(spawn.program.as_str()).ok()?;
    let argv: Vec<CString> = std::iter::once(Some(program.clone()))
        .chain(spawn.args.iter().map(|a| CString::new(a.as_str()).ok()))
        .collect::<Option<_>>()?;
    let envv: Vec<CString> = spawn.env.iter().map(|e| CString::new(e.as_str()).ok()).collect::<Option<_>>()?;
    let dev_null = CString::new("/dev/null").ok()?;
    let out_path = match out {
        Some(p) => CString::new(p.as_os_str().as_bytes()).ok()?,
        None => dev_null.clone(),
    };
    let out_flags = if out.is_some() {
        libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW
    } else {
        libc::O_WRONLY
    };

    let attr = SpawnAttr::new()?;
    let mut actions = FileActions::new()?;
    if !(actions.open(0, &dev_null, libc::O_RDONLY, 0)
        && actions.open(1, &out_path, out_flags, 0o600)
        && actions.open(2, &dev_null, libc::O_WRONLY, 0))
    {
        return None;
    }

    // NULL-terminated pointer arrays into `argv` and `envv`, which outlive the call below.
    let argv_ptrs: Vec<*mut c_char> = argv.iter().map(|a| a.as_ptr() as *mut c_char).chain(std::iter::once(std::ptr::null_mut())).collect();
    let envp_ptrs: Vec<*mut c_char> = envv.iter().map(|e| e.as_ptr() as *mut c_char).chain(std::iter::once(std::ptr::null_mut())).collect();
    let mut pid: libc::pid_t = 0;
    // SAFETY: every pointer refers to live storage owned by this function (`program`, `argv`,
    // `envv`, the two pointer arrays, `actions`, `attr`, `pid`); both arrays end in NULL; and
    // `posix_spawn` does not write through `argv` or `envp` despite their `char *` type.
    let err = unsafe { libc::posix_spawn(&mut pid, program.as_ptr(), &actions.0, &attr.0, argv_ptrs.as_ptr(), envp_ptrs.as_ptr()) };
    (err == 0 && is_child_pid(pid)).then_some(pid)
}

/// Runs `spawn` with `limit` as a hard wall-clock limit. With `out_dir`, the child's standard
/// output goes to a fresh file there, which is read (at most `cap` bytes, with the flag-file
/// safety rules) after the child has ended and then removed on every path. The child is
/// reaped here unless the host reaps it first; it is signalled only on timeout.
fn run_spawned(spawn: &Spawn, out_dir: Option<&Path>, limit: Duration, cap: usize, control: &mut dyn ChildControl) -> RunResult {
    let out = out_dir.map(|d| d.join(temp_name("out")));
    let result = run_with_output(spawn, out.as_deref(), limit, cap, control);
    if let Some(out) = &out {
        let _ = std::fs::remove_file(out);
    }
    result
}

fn run_with_output(spawn: &Spawn, out: Option<&Path>, limit: Duration, cap: usize, control: &mut dyn ChildControl) -> RunResult {
    let Some(pid) = spawn_child(spawn, out) else {
        return RunResult { pid: None, end: RunEnd::SpawnFailed, output: None };
    };
    let deadline = Instant::now() + limit;
    let end = loop {
        match control.wait(pid, false) {
            Waited::Ended(status) => break RunEnd::Exited(exit_code(status)),
            Waited::Running if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
            Waited::Running => {
                // A non-blocking wait has just found the child unreaped, so the pid still names
                // it at that moment. A host that reaps children could reap it before the signal
                // below and the pid could then be reused; without a process handle (macOS has
                // none) that window cannot be closed, only kept short.
                control.kill(pid);
                // Blocking wait; ECHILD (reaped elsewhere meanwhile) is fine.
                let _ = control.wait(pid, true);
                break RunEnd::TimedOut;
            }
            // After a wait error the pid may already be free and reused, so no signal is sent.
            Waited::Error(libc::ECHILD) => break RunEnd::ReapedElsewhere,
            // With a valid pid of our own child, a non-blocking `waitpid` fails only with ECHILD
            // (EINTR is retried), so this is essentially unreachable. If it happens, the child is
            // left alone: it is not signalled (see above) and `curl` ends by its own
            // `--max-time`; at worst it stays unreaped until the host exits.
            Waited::Error(_) => break RunEnd::WaitFailed,
        }
    };
    let output = match (end, out) {
        // The child may still be running, and its output is not used.
        (RunEnd::WaitFailed, _) | (_, None) => None,
        (_, Some(path)) => diag::read_regular_file_capped(path, cap),
    };
    RunResult { pid: Some(pid), end, output }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    const NOW: u64 = 1_800_000_000;

    fn v(major: u32, minor: u32, patch: u32) -> Version {
        Version { major, minor, patch }
    }

    /// A fresh empty directory under the system temporary directory, removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!("fbuc-test-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir(&p).unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
            TempDir(p)
        }
        /// The state directory inside it, not yet created.
        fn state(&self) -> PathBuf {
            self.0.join("TBlur")
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    struct Probe {
        fetches: Arc<AtomicUsize>,
        /// The state file as it was when the fetch step ran.
        seen_at_fetch: Arc<Mutex<Option<Option<State>>>>,
    }

    fn ctx_with(dir: Option<PathBuf>, now: u64, result: Option<Version>) -> (Ctx, Probe) {
        let fetches = Arc::new(AtomicUsize::new(0));
        let seen = Arc::new(Mutex::new(None));
        let (f, s, d) = (fetches.clone(), seen.clone(), dir.clone());
        let ctx = Ctx {
            dir,
            now: Box::new(move || now),
            fetch: Box::new(move || {
                f.fetch_add(1, Ordering::SeqCst);
                *s.lock().unwrap() = Some(d.as_deref().and_then(read_state));
                result
            }),
            publish: Box::new(|_| {}),
            log: Box::new(|_| {}),
        };
        (ctx, Probe { fetches, seen_at_fetch: seen })
    }

    fn put_state(dir: &Path, s: &State) {
        if !dir.exists() {
            std::fs::create_dir(dir).unwrap();
            std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        }
        write_state(dir, s).unwrap();
    }

    fn entries(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        v.sort();
        v
    }

    // ── parser ──

    #[test]
    fn parser_accepts_only_exact_tag_urls() {
        let p = TAG_URL_PREFIX;
        for (tag, want) in [
            ("v0.7.0", v(0, 7, 0)),
            ("0.7.0", v(0, 7, 0)),
            ("v10.20.30", v(10, 20, 30)),
            ("v999999.999999.999999", v(999_999, 999_999, 999_999)),
            ("v0.0.0", v(0, 0, 0)),
            ("v01.002.0003", v(1, 2, 3)),
        ] {
            assert_eq!(parse_redirect(format!("{p}{tag}").as_bytes()), Some(want), "{tag}");
        }

        let rejected: Vec<String> = [
            // other hosts, schemes and paths
            "https://github.com/someone/tblur/releases/tag/v1.0.0",
            "https://github.com/treydoe1/other/releases/tag/v1.0.0",
            "https://gitlab.com/treydoe1/tblur/releases/tag/v1.0.0",
            "https://github.com.example.com/treydoe1/tblur/releases/tag/v1.0.0",
            "https://GitHub.com/treydoe1/tblur/releases/tag/v1.0.0",
            "https://github.com:443/treydoe1/tblur/releases/tag/v1.0.0",
            "https://user@github.com/treydoe1/tblur/releases/tag/v1.0.0",
            "http://github.com/treydoe1/tblur/releases/tag/v1.0.0",
            "HTTPS://github.com/treydoe1/tblur/releases/tag/v1.0.0",
            "//github.com/treydoe1/tblur/releases/tag/v1.0.0",
            "/treydoe1/tblur/releases/tag/v1.0.0",
            "https://github.com/treydoe1/tblur/releases",
            "https://github.com/treydoe1/tblur/releases/latest",
            "https://github.com/treydoe1/tblur/releases/tag",
            "",
        ]
        .iter()
        .map(|s| s.to_string())
        .chain(
            [
                "",
                "v",
                // extra path, query, fragment, whitespace, control bytes
                "v1.0.0/",
                "v1.0.0/x",
                "x/v1.0.0",
                "v1.0.0?x=1",
                "v1.0.0#notes",
                "v1.0.0\n",
                "v1.0.0\r\n",
                "v1.0.0 ",
                " v1.0.0",
                "v 1.0.0",
                "v1.0.0\0",
                "\tv1.0.0",
                // userinfo-style and encoding tricks
                "v1.0.0@example.com",
                "v1.0.0@example.com/x",
                "%761.0.0",
                "v%31.0.0",
                "v1%2E0.0",
                "v1.0.0%00",
                // non-ASCII digits (Arabic-Indic, fullwidth, superscript)
                "v\u{661}.0.0",
                "v\u{ff11}.0.0",
                "v\u{b9}.0.0",
                // signs
                "v+1.0.0",
                "v-1.0.0",
                "v1.-0.0",
                "v1.0.+0",
                "-v1.0.0",
                // empty parts, wrong part counts, long parts
                "v1..0",
                "v.1.0",
                "v1.0.",
                "v..",
                "v1.0",
                "v1",
                "v1.0.0.0",
                "v1234567.0.0",
                "v0.1234567.0",
                "v0.0.1234567",
                // prefixes and suffixes
                "V1.0.0",
                "vv1.0.0",
                "release-1.0.0",
                "v1.0.0-beta",
                "v1.0.0-rc.1",
                "v1.0.0-alpha",
                "v1.0.0+build.5",
                "v1.0.0a",
                "v1.0.0.",
                "v1.0.0x",
                "v1,0,0",
                "v1_0_0",
                "v0x1.0.0",
            ]
            .iter()
            .map(|tag| format!("{TAG_URL_PREFIX}{tag}")),
        )
        .collect();
        for s in &rejected {
            assert_eq!(parse_redirect(s.as_bytes()), None, "{s:?}");
        }
        assert_eq!(parse_redirect(b"\xff\xfe"), None);
        let mut long = TAG_URL_PREFIX.as_bytes().to_vec();
        long.extend(std::iter::repeat_n(b'1', 2000));
        assert_eq!(parse_redirect(&long), None);
    }

    #[test]
    fn version_ordering_is_numeric() {
        assert!(v(0, 10, 0) > v(0, 9, 9));
        assert!(v(1, 0, 0) > v(0, 999_999, 999_999));
        assert!(v(0, 6, 1) > v(0, 6, 0));
        assert!(v(0, 6, 10) > v(0, 6, 9));
        assert!(v(10, 0, 0) > v(9, 99, 99));
        assert_eq!(v(0, 6, 0).cmp(&v(0, 6, 0)), std::cmp::Ordering::Equal);
        assert_eq!(v(0, 10, 0).to_string(), "0.10.0");
        assert_eq!(running_version().map(|v| v.to_string()).as_deref(), Some(VERSION));
    }

    #[test]
    fn available_update_requires_a_strictly_newer_version() {
        let t = TempDir::new();
        let dir = t.state();
        let running = Some(v(0, 6, 0));
        put_state(&dir, &State { checked: NOW, latest: Some(v(0, 6, 0)) });
        assert_eq!(available_update_in(Some(&dir), running), None);
        put_state(&dir, &State { checked: NOW, latest: Some(v(0, 5, 9)) });
        assert_eq!(available_update_in(Some(&dir), running), None);
        put_state(&dir, &State { checked: NOW, latest: Some(v(0, 10, 0)) });
        assert_eq!(available_update_in(Some(&dir), running), Some(v(0, 10, 0)));
        assert_eq!(available_update_in(Some(&dir), None), None);
        assert_eq!(available_update_in(None, running), None);
        put_state(&dir, &State { checked: NOW, latest: None });
        assert_eq!(available_update_in(Some(&dir), running), None);
    }

    // ── state file ──

    #[test]
    fn state_round_trips_and_rejects_anything_else() {
        for s in [State { checked: 0, latest: None }, State { checked: u64::MAX, latest: Some(v(999_999, 0, 12)) }] {
            assert_eq!(parse_state(format_state(&s).as_bytes()), Some(s));
        }
        for bad in [
            "",
            "garbage",
            "checked=1\n",
            "checked=1\nlatest=\n\n",
            "checked=1\nlatest=1.0.0",
            "checked=1\nlatest=v1.0.0\n",
            "checked=1\nlatest=1.0\n",
            "checked=\nlatest=\n",
            "checked=-1\nlatest=\n",
            "checked=18446744073709551616\nlatest=\n",
            "checked=1 \nlatest=\n",
            "latest=\nchecked=1\n",
            " checked=1\nlatest=\n",
        ] {
            assert_eq!(parse_state(bad.as_bytes()), None, "{bad:?}");
        }
    }

    #[test]
    fn unsafe_or_malformed_state_files_mean_no_state() {
        let t = TempDir::new();
        let dir = t.state();
        let good = State { checked: NOW, latest: Some(v(9, 0, 0)) };
        put_state(&dir, &good);
        let file = dir.join(STATE_FILE);
        assert_eq!(read_state(&dir), Some(good));
        assert_eq!(std::fs::metadata(&file).unwrap().permissions().mode() & 0o777, 0o600);

        // group- or world-writable
        for mode in [0o620, 0o602, 0o666] {
            std::fs::set_permissions(&file, std::fs::Permissions::from_mode(mode)).unwrap();
            assert_eq!(read_state(&dir), None, "{mode:o}");
            assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), None);
        }
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o600)).unwrap();

        // oversize, even with a valid start
        let mut big = format_state(&good).into_bytes();
        big.resize(MAX_STATE_BYTES + 1, b'\n');
        std::fs::write(&file, &big).unwrap();
        assert_eq!(read_state(&dir), None);

        // garbage
        std::fs::write(&file, b"\x00\xffnot a state file").unwrap();
        assert_eq!(read_state(&dir), None);

        // a symlink to a valid state file
        let real = t.0.join("real-state");
        std::fs::write(&real, format_state(&good)).unwrap();
        std::fs::set_permissions(&real, std::fs::Permissions::from_mode(0o600)).unwrap();
        std::fs::remove_file(&file).unwrap();
        symlink(&real, &file).unwrap();
        assert_eq!(read_state(&dir), None);
        assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), None);

        // a directory in place of the file
        std::fs::remove_file(&file).unwrap();
        std::fs::create_dir(&file).unwrap();
        assert_eq!(read_state(&dir), None);
    }

    #[test]
    fn missing_or_untrusted_directories() {
        // A missing state directory is created with mode 0700.
        let t = TempDir::new();
        let dir = t.state();
        let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::Fetched(None));
        assert_eq!(std::fs::symlink_metadata(&dir).unwrap().permissions().mode() & 0o777, 0o700);
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 1);

        // A missing parent is not created.
        let t = TempDir::new();
        let dir = t.0.join("missing").join("TBlur");
        let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::BadDir);
        assert!(!t.0.join("missing").exists());
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);

        // A state directory that is a symlink is refused for reading and writing.
        let t = TempDir::new();
        let real = t.0.join("real");
        put_state(&real, &State { checked: 0, latest: Some(v(9, 0, 0)) });
        let link = t.state();
        symlink(&real, &link).unwrap();
        let (ctx, probe) = ctx_with(Some(link.clone()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::BadDir);
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
        assert_eq!(read_state(&link), None);
        assert_eq!(available_update_in(Some(&link), Some(v(0, 6, 0))), None);
        assert_eq!(read_state(&real).unwrap().checked, 0);

        // A group-writable state directory is refused.
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 0, latest: Some(v(9, 0, 0)) });
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o770)).unwrap();
        let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::BadDir);
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
        assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), None);

        // A regular file where the directory should be: looking for the off switch fails with
        // "not a directory", which counts as off.
        let t = TempDir::new();
        std::fs::write(t.state(), b"").unwrap();
        let (ctx, probe) = ctx_with(Some(t.state()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::Off);
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);

        // No home directory: the feature is off.
        let (ctx, probe) = ctx_with(None, NOW, None);
        assert_eq!(check_once(&ctx), Outcome::NoHome);
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn atomic_write_leaves_no_temporary_files() {
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 1, latest: None });
        for i in 0..5 {
            write_state(&dir, &State { checked: i, latest: Some(v(0, 7, i as u32)) }).unwrap();
        }
        assert_eq!(entries(&dir), vec![STATE_FILE.to_string()]);
        assert_eq!(read_state(&dir), Some(State { checked: 4, latest: Some(v(0, 7, 4)) }));

        // The rename fails when a non-empty directory stands at the state path.
        std::fs::remove_file(dir.join(STATE_FILE)).unwrap();
        std::fs::create_dir(dir.join(STATE_FILE)).unwrap();
        std::fs::write(dir.join(STATE_FILE).join("keep"), b"").unwrap();
        assert!(write_state(&dir, &State { checked: 9, latest: None }).is_err());
        assert_eq!(entries(&dir), vec![STATE_FILE.to_string()]);

        // Creating the temporary file fails in a directory without write permission.
        std::fs::remove_dir_all(dir.join(STATE_FILE)).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        assert!(write_state(&dir, &State { checked: 9, latest: None }).is_err());
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(entries(&dir).is_empty());
    }

    // ── the 24-hour rule and the off switch ──

    #[test]
    fn checks_at_most_once_a_day() {
        let t = TempDir::new();
        let dir = t.state();
        for (last, fetch_expected) in [
            (NOW, false),
            (NOW - 1, false),
            (NOW - CHECK_INTERVAL_SECS + 1, false),
            (NOW - CHECK_INTERVAL_SECS, true),
            (NOW - 10 * CHECK_INTERVAL_SECS, true),
            (0, true),
            // a time in the future is stale
            (NOW + 1, true),
            (NOW + CHECK_INTERVAL_SECS * 365, true),
        ] {
            put_state(&dir, &State { checked: last, latest: Some(v(0, 7, 0)) });
            let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, Some(v(0, 8, 0)));
            let outcome = check_once(&ctx);
            let fetches = probe.fetches.load(Ordering::SeqCst);
            if fetch_expected {
                assert_eq!((outcome, fetches), (Outcome::Fetched(Some(v(0, 8, 0))), 1), "last={last}");
                assert_eq!(read_state(&dir), Some(State { checked: NOW, latest: Some(v(0, 8, 0)) }));
            } else {
                assert_eq!((outcome, fetches), (Outcome::Recent, 0), "last={last}");
                assert_eq!(read_state(&dir), Some(State { checked: last, latest: Some(v(0, 7, 0)) }));
            }
        }
    }

    #[test]
    fn attempt_time_is_written_before_the_fetch() {
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 5, latest: Some(v(0, 7, 0)) });
        let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::Fetched(None));
        assert_eq!(*probe.seen_at_fetch.lock().unwrap(), Some(Some(State { checked: NOW, latest: Some(v(0, 7, 0)) })));
        // No information keeps the version already known.
        assert_eq!(read_state(&dir), Some(State { checked: NOW, latest: Some(v(0, 7, 0)) }));

        // Without a state file the attempt is recorded too.
        let t = TempDir::new();
        let (ctx, probe) = ctx_with(Some(t.state()), NOW, Some(v(1, 0, 0)));
        assert_eq!(check_once(&ctx), Outcome::Fetched(Some(v(1, 0, 0))));
        assert_eq!(*probe.seen_at_fetch.lock().unwrap(), Some(Some(State { checked: NOW, latest: None })));
        assert_eq!(read_state(&t.state()), Some(State { checked: NOW, latest: Some(v(1, 0, 0)) }));
    }

    #[test]
    fn no_fetch_when_the_attempt_cannot_be_recorded() {
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 0, latest: None });
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o500)).unwrap();
        let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, Some(v(1, 0, 0)));
        assert_eq!(check_once(&ctx), Outcome::WriteFailed);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn off_switch_stops_the_fetch_and_the_notice() {
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 0, latest: Some(v(9, 0, 0)) });
        assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), Some(v(9, 0, 0)));

        // Any kind of entry counts: an empty file, a directory, a dangling symlink.
        for make in [
            (|p: &Path| std::fs::write(p, b"").unwrap()) as fn(&Path),
            |p| std::fs::create_dir(p).unwrap(),
            |p| symlink("/nonexistent-fbuc-target", p).unwrap(),
        ] {
            let switch = dir.join(OFF_SWITCH_FILE);
            make(&switch);
            let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, Some(v(10, 0, 0)));
            assert_eq!(check_once(&ctx), Outcome::Off);
            assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
            assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), None);

            let once = Once::new();
            let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, Some(v(10, 0, 0)));
            if let Some(h) = start_with(&once, move || ctx) {
                h.join().unwrap();
            }
            assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
            assert_eq!(read_state(&dir).unwrap().checked, 0);

            if switch.is_dir() && !switch.is_symlink() {
                std::fs::remove_dir(&switch).unwrap();
            } else {
                std::fs::remove_file(&switch).unwrap();
            }
        }
        assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), Some(v(9, 0, 0)));
    }

    #[test]
    fn off_switch_lookup_errors_count_as_off() {
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 0, latest: Some(v(9, 0, 0)) });
        // Without search permission the switch cannot be seen, so the feature is off.
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o600)).unwrap();
        assert!(is_off(&dir));
        let (ctx, probe) = ctx_with(Some(dir.clone()), NOW, Some(v(10, 0, 0)));
        assert_eq!(check_once(&ctx), Outcome::Off);
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 0);
        assert_eq!(available_update_in(Some(&dir), Some(v(0, 6, 0))), None);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        assert!(!is_off(&dir));
        // A missing directory or parent is "not found": not off.
        assert!(!is_off(&t.0.join("missing")));
        assert!(!is_off(&t.0.join("missing").join("deeper")));
    }

    #[test]
    fn notice_cache_reads_the_file_once_and_takes_published_versions() {
        let cache = NoticeCache::new();
        let fills = AtomicUsize::new(0);
        let fill = || {
            fills.fetch_add(1, Ordering::SeqCst);
            Some(v(0, 7, 0))
        };
        assert_eq!(cache.get(fill), Some(v(0, 7, 0)));
        assert_eq!(cache.get(|| panic!("filled once")), Some(v(0, 7, 0)));
        assert_eq!(fills.load(Ordering::SeqCst), 1);

        let running = Some(v(0, 6, 0));
        cache.publish(v(0, 8, 0), running);
        assert_eq!(cache.get(|| panic!("filled once")), Some(v(0, 8, 0)));
        // A version that is not newer clears the notice, as the state file now says.
        cache.publish(v(0, 6, 0), running);
        assert_eq!(cache.get(|| panic!("filled once")), None);
        cache.publish(v(0, 5, 0), running);
        assert_eq!(cache.get(|| panic!("filled once")), None);
        cache.publish(v(1, 0, 0), None);
        assert_eq!(cache.get(|| panic!("filled once")), None);

        // A publish before anyone asked also fills the cache.
        let cache = NoticeCache::new();
        cache.publish(v(0, 9, 0), running);
        assert_eq!(cache.get(|| panic!("already filled")), Some(v(0, 9, 0)));
        // A cache filled with "no notice" stays empty without reading again.
        let cache = NoticeCache::new();
        assert_eq!(cache.get(|| None), None);
        assert_eq!(cache.get(|| panic!("filled once")), None);
    }

    #[test]
    fn the_check_thread_publishes_what_it_found() {
        let t = TempDir::new();
        let cache = Arc::new(NoticeCache::new());
        let running = Some(v(0, 6, 0));
        // The cache was filled with "no notice" when the host first asked.
        assert_eq!(cache.get(|| None), None);
        let (mut ctx, probe) = ctx_with(Some(t.state()), NOW, Some(v(0, 7, 0)));
        let c = cache.clone();
        ctx.publish = Box::new(move |latest| c.publish(latest, running));
        let once = Once::new();
        start_with(&once, move || ctx).unwrap().join().unwrap();
        assert_eq!(probe.fetches.load(Ordering::SeqCst), 1);
        assert_eq!(cache.get(|| panic!("filled once")), Some(v(0, 7, 0)));
        assert_eq!(read_state(&t.state()).unwrap().latest, Some(v(0, 7, 0)));

        // No information publishes nothing.
        let t = TempDir::new();
        let cache = Arc::new(NoticeCache::new());
        let (mut ctx, _) = ctx_with(Some(t.state()), NOW, None);
        let c = cache.clone();
        ctx.publish = Box::new(move |latest| c.publish(latest, running));
        start_with(&Once::new(), move || ctx).unwrap().join().unwrap();
        assert_eq!(cache.get(|| Some(v(9, 9, 9))), Some(v(9, 9, 9)), "still unfilled");
    }

    #[test]
    fn state_directory_is_resolved_once_and_never_in_tests() {
        assert_eq!(state_dir(), None);
        assert_eq!(state_dir(), None);
        assert_eq!(available_update(), None);
    }

    #[test]
    fn two_starts_cause_one_fetch() {
        let t = TempDir::new();
        let once = Once::new();
        let (ctx1, probe1) = ctx_with(Some(t.state()), NOW, Some(v(0, 9, 0)));
        let (ctx2, probe2) = ctx_with(Some(t.state()), NOW + 10 * CHECK_INTERVAL_SECS, Some(v(0, 9, 0)));
        let first = start_with(&once, move || ctx1);
        let second = start_with(&once, move || ctx2);
        assert!(second.is_none());
        first.expect("first start spawns a thread").join().unwrap();
        assert_eq!(probe1.fetches.load(Ordering::SeqCst) + probe2.fetches.load(Ordering::SeqCst), 1);
    }

    // ── child processes ──

    fn sp(program: &str, args: &[&str], env: &[&str]) -> Spawn {
        Spawn {
            program: program.to_string(),
            args: args.iter().map(|a| a.to_string()).collect(),
            env: env.iter().map(|e| e.to_string()).collect(),
        }
    }

    fn sh(script: &str) -> Spawn {
        sp("/bin/sh", &["-c", script], &[])
    }

    fn run(spawn: &Spawn, dir: &Path, limit: Duration) -> RunResult {
        run_spawned(spawn, Some(dir), limit, MAX_OUTPUT_BYTES, &mut RealChildControl)
    }

    fn is_reaped(pid: libc::pid_t) -> bool {
        // SAFETY: signal 0 only checks whether the process exists; nothing is delivered.
        let r = unsafe { libc::kill(pid, 0) };
        r == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    }

    /// `/bin/sh` asking `lsof` for the shell's numeric descriptors. The trailing `; true` keeps
    /// the shell from exec-ing `lsof` in its place (which would list `lsof`'s own descriptors),
    /// and `ls /dev/fd` would list two descriptors of its own as well.
    const LIST_FDS: &str = "/usr/sbin/lsof -a -p $$ -d 0-65535 -Ff; true";

    fn fds_in(lsof_out: &[u8]) -> Vec<String> {
        String::from_utf8_lossy(lsof_out).lines().filter_map(|l| l.strip_prefix('f')).map(str::to_string).collect()
    }

    #[test]
    fn child_inherits_only_fds_0_1_2() {
        use std::os::fd::AsRawFd;
        let t = TempDir::new();
        // Descriptors without close-on-exec, as a host might hold them: the lowest free one
        // and one at 64 or above.
        let file = std::fs::File::open("/dev/null").unwrap();
        // SAFETY: F_DUPFD duplicates a valid descriptor; each copy has close-on-exec cleared.
        let leaked = [3, 64].map(|min| unsafe { libc::fcntl(file.as_raw_fd(), libc::F_DUPFD, min) });
        assert!(leaked.iter().all(|&fd| fd >= 3), "{leaked:?}");

        // Control: a child started by std inherits them, so the stand-in detects a leak.
        let std_out = std::process::Command::new("/bin/sh").args(["-c", LIST_FDS]).output().unwrap();
        let r = run(&sh(LIST_FDS), &t.0, Duration::from_secs(20));
        for fd in leaked {
            // SAFETY: each is a descriptor this test opened and closes once.
            unsafe { libc::close(fd) };
        }

        let std_seen = fds_in(&std_out.stdout);
        assert!(leaked.iter().all(|fd| std_seen.contains(&fd.to_string())), "control: {std_seen:?}, leaked {leaked:?}");
        assert_eq!(fds_in(&r.usable_output().unwrap()), ["0", "1", "2"]);
        assert!(entries(&t.0).is_empty());
    }

    #[test]
    fn child_environment_is_exactly_what_is_passed() {
        let t = TempDir::new();
        let env_of = |env: Vec<String>| {
            let s = Spawn { program: "/usr/bin/env".to_string(), args: Vec::new(), env };
            String::from_utf8(run(&s, &t.0, Duration::from_secs(10)).usable_output().unwrap()).unwrap()
        };
        // The test process has an environment of its own; none of it reaches the child.
        assert!(std::env::vars_os().next().is_some());
        assert_eq!(env_of(curl_spawn().env), "");
        assert_eq!(env_of(open_spawn(Some(Path::new("/home-for-test"))).env), "HOME=/home-for-test\nPATH=/usr/bin:/bin\n");
        assert_eq!(env_of(open_spawn(None).env), "PATH=/usr/bin:/bin\n");
        assert_eq!(env_of(vec!["A=1".into(), "B=two words".into()]), "A=1\nB=two words\n");
    }

    #[test]
    fn child_signal_dispositions_are_reset() {
        let t = TempDir::new();
        // The Rust runtime ignores SIGPIPE in this process (read here, not changed), and an
        // ignored disposition survives exec unless it is reset.
        let mut old: libc::sigaction = unsafe { std::mem::zeroed() };
        // SAFETY: a null new action only reads the current one into `old`.
        assert_eq!(unsafe { libc::sigaction(libc::SIGPIPE, std::ptr::null(), &mut old) }, 0);
        assert_eq!(old.sa_sigaction, libc::SIG_IGN, "precondition: SIGPIPE ignored in the test process");
        // With the default action restored, the shell dies of its own SIGPIPE.
        let r = run(&sh("kill -PIPE $$; echo survived"), &t.0, Duration::from_secs(10));
        assert_eq!(r.end, RunEnd::Exited(None));
        assert_eq!(r.output.as_deref(), Some(&b""[..]));
    }

    #[test]
    fn child_signal_mask_is_empty() {
        let t = TempDir::new();
        // Block SIGUSR2 on this test thread only; posix_spawn would pass the caller's mask on.
        let mut block: libc::sigset_t = 0;
        let mut old: libc::sigset_t = 0;
        // SAFETY: valid pointers to local sets; the change affects only this thread and is
        // undone below.
        unsafe {
            libc::sigemptyset(&mut block);
            libc::sigaddset(&mut block, libc::SIGUSR2);
            assert_eq!(libc::pthread_sigmask(libc::SIG_BLOCK, &block, &mut old), 0);
        }
        let r = run(&sh("/bin/ps -o sigmask= -p $$"), &t.0, Duration::from_secs(10));
        // SAFETY: restores this thread's previous mask.
        unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, &old, std::ptr::null_mut()) };
        let out = String::from_utf8(r.usable_output().unwrap()).unwrap();
        assert_eq!(out.trim(), "0", "{out:?}");
    }

    #[test]
    fn runner_kills_and_reaps_on_timeout() {
        let t = TempDir::new();
        let t0 = Instant::now();
        let r = run(&sp("/bin/sleep", &["30"], &[]), &t.0, Duration::from_millis(300));
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        assert_eq!(r.end, RunEnd::TimedOut);
        assert!(is_reaped(r.pid.unwrap()));
        assert_eq!(r.usable_output(), None);
        assert!(entries(&t.0).is_empty(), "output file removed after a timeout");
    }

    #[test]
    fn runner_enforces_the_output_cap() {
        let t = TempDir::new();
        let r = run(&sh("head -c 1024 /dev/zero"), &t.0, Duration::from_secs(10));
        assert_eq!(r.usable_output().map(|o| o.len()), Some(1024));
        let r = run(&sh("head -c 1025 /dev/zero"), &t.0, Duration::from_secs(10));
        assert_eq!(r.end, RunEnd::Exited(Some(0)));
        assert_eq!(r.usable_output(), None);
        let r = run(&sh("head -c 1000000 /dev/zero"), &t.0, Duration::from_secs(10));
        assert_eq!(r.usable_output(), None);
        assert!(entries(&t.0).is_empty());
    }

    #[test]
    fn runner_reports_only_successful_output_and_removes_the_file() {
        let t = TempDir::new();
        let r = run(&sp("/bin/echo", &["hello"], &[]), &t.0, Duration::from_secs(10));
        assert!(is_reaped(r.pid.unwrap()));
        assert_eq!(r.usable_output(), Some(b"hello\n".to_vec()));
        assert!(entries(&t.0).is_empty(), "removed after success");

        let r = run(&sp("/usr/bin/false", &[], &[]), &t.0, Duration::from_secs(10));
        assert_eq!(r.end, RunEnd::Exited(Some(1)));
        assert!(is_reaped(r.pid.unwrap()));

        let r = run(&sh(&format!("printf '{TAG_URL_PREFIX}v9.9.9'; exit 3")), &t.0, Duration::from_secs(10));
        assert_eq!(r.end, RunEnd::Exited(Some(3)));
        assert_eq!(r.usable_output(), None);
        assert!(entries(&t.0).is_empty(), "removed after failure");

        let r = run(&sh(&format!("printf '{TAG_URL_PREFIX}v9.9.9'; kill -9 $$")), &t.0, Duration::from_secs(10));
        assert_eq!(r.end, RunEnd::Exited(None));
        assert_eq!(r.usable_output(), None);

        // Spawn failures: missing program, NUL in an argument, missing output directory.
        for (spawn, dir) in [
            (sp("/nonexistent-fbuc-binary", &[], &[]), t.0.clone()),
            (sp("/bin/echo", &["a\0b"], &[]), t.0.clone()),
            (sp("/bin/echo", &["x"], &["A=\0"]), t.0.clone()),
            (sp("/bin/echo", &["x"], &[]), t.0.join("missing")),
        ] {
            let r = run(&spawn, &dir, Duration::from_secs(1));
            assert_eq!((r.end, r.pid), (RunEnd::SpawnFailed, None), "{spawn:?}");
        }
        assert!(entries(&t.0).is_empty(), "removed after spawn failures");

        // Without an output directory, stdout goes to /dev/null.
        let r = run_spawned(&sp("/bin/echo", &["x"], &[]), None, Duration::from_secs(10), 0, &mut RealChildControl);
        assert_eq!((r.end, r.output), (RunEnd::Exited(Some(0)), None));
    }

    #[test]
    fn output_file_is_created_by_the_child_exclusively_with_mode_0600() {
        use std::os::unix::fs::MetadataExt;
        let t = TempDir::new();
        let out = t.0.join(".update-check.test.out");
        // `run_with_output` leaves the file in place, so it can be inspected here.
        let r = run_with_output(&sp("/bin/echo", &["x"], &[]), Some(&out), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut RealChildControl);
        assert_eq!(r.usable_output().as_deref(), Some(&b"x\n"[..]));
        let meta = std::fs::symlink_metadata(&out).unwrap();
        assert!(meta.is_file());
        assert_eq!(meta.mode() & 0o777, 0o600);
        assert_eq!(meta.uid(), diag::current_euid());

        // An existing file or a symlink at that name is never written through.
        let r = run_with_output(&sp("/bin/echo", &["y"], &[]), Some(&out), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut RealChildControl);
        assert_eq!(r.end, RunEnd::SpawnFailed);
        assert_eq!(std::fs::read(&out).unwrap(), b"x\n");
        std::fs::remove_file(&out).unwrap();
        let target = t.0.join("target");
        symlink(&target, &out).unwrap();
        let r = run_with_output(&sp("/bin/echo", &["z"], &[]), Some(&out), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut RealChildControl);
        assert_eq!(r.end, RunEnd::SpawnFailed);
        assert!(!target.exists());
    }

    #[test]
    fn leftover_sweep_removes_only_old_temporary_and_output_files() {
        let t = TempDir::new();
        let dir = &t.0;
        let now = SystemTime::now();
        let old = now - LEFTOVER_AGE - Duration::from_secs(60);
        let fresh = now - LEFTOVER_AGE + Duration::from_secs(60);
        let make = |name: &str, mtime: SystemTime| {
            let f = std::fs::File::create(dir.join(name)).unwrap();
            f.set_modified(mtime).unwrap();
        };
        make(".update-check.1.2.tmp", old);
        make(".update-check.1.3.out", old);
        make(".update-check.1.4.tmp", fresh);
        make(".update-check.1.5.out", fresh);
        make("update-check", old);
        make("no-update-check", old);
        make(".update-check.1.6.txt", old);
        make(".other.1.7.tmp", old);
        std::fs::create_dir(dir.join(".update-check.9.9.out")).unwrap();
        sweep_leftovers(dir, now);
        assert_eq!(
            entries(dir),
            [".other.1.7.tmp", ".update-check.1.4.tmp", ".update-check.1.5.out", ".update-check.1.6.txt", ".update-check.9.9.out", "no-update-check", "update-check"]
        );
    }

    #[test]
    fn check_past_the_gate_sweeps_leftovers() {
        let t = TempDir::new();
        let dir = t.state();
        put_state(&dir, &State { checked: 0, latest: None });
        let f = std::fs::File::create(dir.join(".update-check.1.1.out")).unwrap();
        f.set_modified(SystemTime::now() - 2 * LEFTOVER_AGE).unwrap();
        let (ctx, _) = ctx_with(Some(dir.clone()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::Fetched(None));
        assert_eq!(entries(&dir), [STATE_FILE]);
    }

    /// Stands in for a host that reaps children itself (SIGCHLD ignored, or its own
    /// `waitpid`): before the runner's first wait it reaps the child with a blocking `waitpid`,
    /// so the runner's real non-blocking `waitpid` then fails with ECHILD. With `other_error`,
    /// it reports a different wait error instead. It counts kill requests.
    struct HostReaps {
        other_error: bool,
        kills: usize,
    }

    impl ChildControl for HostReaps {
        fn wait(&mut self, pid: libc::pid_t, block: bool) -> Waited {
            assert!(!block, "only the timeout path waits blocking");
            assert!(matches!(waitpid_once(pid, true), Waited::Ended(_)));
            if self.other_error {
                return Waited::Error(libc::EINVAL);
            }
            waitpid_once(pid, false)
        }
        fn kill(&mut self, _: libc::pid_t) {
            self.kills += 1;
        }
    }

    #[test]
    fn output_is_used_when_the_host_reaps_the_child() {
        let t = TempDir::new();
        let url = format!("{TAG_URL_PREFIX}v0.7.0");
        // The exit status is unknown once someone else has reaped the child; the output is used
        // and validated by the parser.
        for exit in [0, 3] {
            let mut host = HostReaps { other_error: false, kills: 0 };
            let r = run_spawned(&sh(&format!("printf '{url}'; exit {exit}")), Some(&t.0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
            assert_eq!(r.end, RunEnd::ReapedElsewhere);
            assert_eq!(host.kills, 0, "no signal after a wait error");
            assert!(is_reaped(r.pid.unwrap()));
            assert_eq!(r.usable_output().as_deref().and_then(parse_redirect), Some(v(0, 7, 0)));
        }

        // Unexpected or over-cap output is still no information.
        let mut host = HostReaps { other_error: false, kills: 0 };
        let r = run_spawned(&sh("printf 'https://example.com/'"), Some(&t.0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
        assert_eq!(r.end, RunEnd::ReapedElsewhere);
        assert_eq!(r.usable_output().as_deref().and_then(parse_redirect), None);
        let r = run_spawned(&sh("head -c 5000 /dev/zero"), Some(&t.0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
        assert_eq!((r.end, host.kills), (RunEnd::ReapedElsewhere, 0));
        assert_eq!(r.usable_output(), None);
        assert!(entries(&t.0).is_empty());
    }

    #[test]
    fn other_wait_errors_send_no_signal_and_give_no_information() {
        let t = TempDir::new();
        let mut host = HostReaps { other_error: true, kills: 0 };
        let r = run_spawned(&sh(&format!("printf '{TAG_URL_PREFIX}v0.7.0'")), Some(&t.0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
        assert_eq!(r.end, RunEnd::WaitFailed);
        assert_eq!(host.kills, 0, "no signal after a wait error");
        assert_eq!(r.usable_output(), None);
        assert!(entries(&t.0).is_empty());
    }

    #[test]
    fn only_the_timeout_path_sends_a_signal() {
        /// The real wait and kill, counting kill requests.
        struct Counting(usize);
        impl ChildControl for Counting {
            fn wait(&mut self, pid: libc::pid_t, block: bool) -> Waited {
                waitpid_once(pid, block)
            }
            fn kill(&mut self, pid: libc::pid_t) {
                self.0 += 1;
                RealChildControl.kill(pid);
            }
        }
        let t = TempDir::new();
        let mut c = Counting(0);
        let r = run_spawned(&sp("/bin/sleep", &["30"], &[]), Some(&t.0), Duration::from_millis(200), MAX_OUTPUT_BYTES, &mut c);
        assert_eq!((r.end, c.0), (RunEnd::TimedOut, 1));
        assert!(is_reaped(r.pid.unwrap()));

        let mut c = Counting(0);
        let r = run_spawned(&sp("/usr/bin/true", &[], &[]), Some(&t.0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut c);
        assert_eq!((r.end, c.0), (RunEnd::Exited(Some(0)), 0));
    }

    #[test]
    fn pids_of_zero_or_less_never_reach_waitpid_or_kill() {
        for pid in [0, -1, -2, libc::pid_t::MIN] {
            assert!(!is_child_pid(pid));
            // 0 or -1 would otherwise reap (or, blocking, wait for) any child of this process.
            assert_eq!(waitpid_once(pid, false), Waited::Error(libc::EINVAL));
            assert_eq!(waitpid_once(pid, true), Waited::Error(libc::EINVAL));
        }
        assert!(is_child_pid(1));
        // EINVAL is a failed wait, not "reaped elsewhere", so its output would never be used.
        struct BadPid;
        impl ChildControl for BadPid {
            fn wait(&mut self, _: libc::pid_t, block: bool) -> Waited {
                waitpid_once(0, block)
            }
            fn kill(&mut self, _: libc::pid_t) {
                panic!("no signal after a wait error");
            }
        }
        let t = TempDir::new();
        let r = run_spawned(&sp("/bin/echo", &["x"], &[]), Some(&t.0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut BadPid);
        assert_eq!((r.end, r.output), (RunEnd::WaitFailed, None));
        // The real child is still ours to reap.
        assert!(matches!(waitpid_once(r.pid.unwrap(), true), Waited::Ended(_)));
    }

    #[test]
    fn spawn_attributes_include_cleared_exception_ports() {
        // Every exception type from EXC_BAD_ACCESS (1) to EXC_CORPSE_NOTIFY (13).
        assert_eq!(CHILD_EXC_MASK, 0x3ffe);
        assert_eq!(CHILD_EXC_MASK & (1 << 10), 1 << 10, "EXC_CRASH");
        assert!(SpawnAttr::new().is_some(), "the host accepts the whole attribute set");
    }

    #[test]
    fn wait_status_decoding() {
        assert_eq!(exit_code(0), Some(0));
        assert_eq!(exit_code(3 << 8), Some(3));
        assert_eq!(exit_code(255 << 8), Some(255));
        assert_eq!(exit_code(libc::SIGKILL), None);
        assert_eq!(exit_code(libc::SIGTERM | 0o200), None);
    }

    #[test]
    fn release_page_opens_at_most_once_every_three_seconds() {
        let last = AtomicU64::new(0);
        assert!(allow_open(&last, 0));
        assert!(!allow_open(&last, 0));
        assert!(!allow_open(&last, 1));
        assert!(!allow_open(&last, OPEN_INTERVAL_MS - 1));
        assert!(allow_open(&last, OPEN_INTERVAL_MS));
        assert!(!allow_open(&last, 2 * OPEN_INTERVAL_MS - 1));
        assert!(allow_open(&last, 10 * OPEN_INTERVAL_MS));
        // A refused call does not extend the window.
        assert!(!allow_open(&last, 11 * OPEN_INTERVAL_MS - 1));
        assert!(allow_open(&last, 11 * OPEN_INTERVAL_MS));

        let last = AtomicU64::new(0);
        assert!(allow_open(&last, 1 << 40));
        assert!(!allow_open(&last, (1 << 40) + 5));

        // Concurrent clicks at the same moment open one page.
        let last = Arc::new(AtomicU64::new(0));
        let opened: usize = (0..8)
            .map(|_| {
                let last = last.clone();
                std::thread::spawn(move || allow_open(&last, 500))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| usize::from(h.join().unwrap()))
            .sum();
        assert_eq!(opened, 1);
    }

    #[test]
    fn curl_and_open_command_lines() {
        let c = curl_spawn();
        assert_eq!(c.program, CURL);
        let ua = format!("TBlur/{VERSION}");
        let want = [
            "-q", "--proto", "=https", "--proto-redir", "=https", "--tlsv1.2", "--connect-timeout", "5",
            "--max-time", "8", "--max-redirs", "0", "-sS", "-o", "/dev/null", "-w", "%{redirect_url}",
            "-A", &ua, LATEST_URL,
        ];
        assert_eq!(c.args, want);
        assert_eq!(c.args[0], "-q", "-q only works as the first argument");
        assert!(c.env.is_empty());

        // Checked as data only: running it would open a browser window.
        let o = open_spawn(Some(Path::new("/home-for-test")));
        assert_eq!(o, sp(OPEN, &[RELEASES_URL], &["HOME=/home-for-test", "PATH=/usr/bin:/bin"]));
        assert_eq!(open_spawn(None).env, ["PATH=/usr/bin:/bin"]);
    }

    #[test]
    fn tests_never_resolve_the_real_home() {
        assert_eq!(home_dir(), None);
        assert_eq!(state_dir(), None);
        assert_eq!(available_update(), None);
    }

    /// Runs the real `curl` against GitHub and prints what it parsed. Run by hand:
    /// `cargo test --release live_fetch -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_fetch() {
        let t = TempDir::new();
        let r = run(&curl_spawn(), &t.0, CURL_TIME_LIMIT);
        println!("end={:?} output_bytes={:?}", r.end, r.output.as_ref().map(Vec::len));
        match r.usable_output().as_deref().and_then(parse_redirect) {
            Some(v) => println!("latest release: {v}"),
            None => println!("no information (no release, no network, or an unexpected reply)"),
        }
        assert!(entries(&t.0).is_empty());
    }
}
