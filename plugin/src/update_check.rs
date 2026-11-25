//! Update notice. At most once every 24 hours, when the host loads the plug-in, a background
//! thread asks GitHub which release is the latest and stores only its numeric version. The effect
//! then shows "x.y.z is available" in its About text and on its last parameter row. Nothing is
//! ever downloaded or installed.
//!
//! * The network request runs in the system `curl` as a child process (absolute path, empty
//!   environment, no shell), so no network or TLS code runs inside the host's process. It asks
//!   for `releases/latest`, which answers with a redirect to `releases/tag/<tag>`; only the
//!   redirect target is read and it is never followed.
//! * The reply is accepted only if it is exactly the expected tag URL with a
//!   `MAJOR.MINOR.PATCH` version; anything else means "no information". Only the parsed numbers
//!   are stored and shown.
//! * State lives in `~/Library/Application Support/FlowBlur/update-check` (attempt time in Unix
//!   seconds and the latest version seen). The attempt time is written before `curl` starts, so
//!   several host processes loading the plug-in together do not all check.
//! * The file `~/Library/Application Support/FlowBlur/no-update-check` turns all of this off: no
//!   child process, no request, no notice.
//! * Every failure is silent; with the diagnostics log enabled, one line per attempt is logged.

use std::io::{Read, Write};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::VERSION;
use crate::diag;

/// The page a click on the notice opens.
pub const RELEASES_URL: &str = "https://github.com/treydoe1/flowblur/releases";
/// GitHub answers this with a redirect to the latest release's tag page.
const LATEST_URL: &str = "https://github.com/treydoe1/flowblur/releases/latest";
/// The only redirect target accepted, followed by the tag.
const TAG_URL_PREFIX: &str = "https://github.com/treydoe1/flowblur/releases/tag/";

const CURL: &str = "/usr/bin/curl";
const OPEN: &str = "/usr/bin/open";

const STATE_FILE: &str = "update-check";
const OFF_SWITCH_FILE: &str = "no-update-check";
const CHECK_INTERVAL_SECS: u64 = 24 * 60 * 60;
/// The largest valid state file is 57 bytes.
const MAX_STATE_BYTES: usize = 128;
/// The largest valid redirect URL is 74 bytes.
const MAX_OUTPUT_BYTES: usize = 1024;
/// Hard wall-clock limit for the `curl` child; its own `--max-time` is 8 s.
const CURL_TIME_LIMIT: Duration = Duration::from_secs(12);
const OPEN_TIME_LIMIT: Duration = Duration::from_secs(10);
/// At most one release page opens per this interval, however often the host reports a click.
const OPEN_INTERVAL_MS: u64 = 3000;
const POLL_INTERVAL: Duration = Duration::from_millis(20);
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

/// Accepts exactly the output of `format_state`.
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

/// Any entry at the off-switch path, of any type, turns the feature off.
fn is_off(dir: &Path) -> bool {
    std::fs::symlink_metadata(dir.join(OFF_SWITCH_FILE)).is_ok()
}

/// Reads the state with the flag-file safety rules; unreadable, unsafe or malformed is `None`.
fn read_state(dir: &Path) -> Option<State> {
    if !is_trusted_dir(dir) {
        return None;
    }
    parse_state(&diag::read_regular_file_capped(&dir.join(STATE_FILE), MAX_STATE_BYTES)?)
}

/// Writes the state atomically: a temporary file created exclusively with mode 0600 in the same
/// directory, then renamed over the state file. The temporary file is removed on any failure.
fn write_state(dir: &Path, state: &State) -> std::io::Result<()> {
    use std::os::unix::fs::OpenOptionsExt;
    // Darwin `<sys/fcntl.h>`: never follow a symlink at the temporary path.
    const O_NOFOLLOW: i32 = 0x0100;
    static SEQ: AtomicU64 = AtomicU64::new(0);

    let tmp = dir.join(format!(".{STATE_FILE}.{}.{}.tmp", std::process::id(), SEQ.fetch_add(1, Ordering::Relaxed)));
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(O_NOFOLLOW)
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
    // Record the attempt before the request, so other host processes skip theirs.
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

/// Runs `check_once` on a named background thread, at most once per `once`.
fn start_with(once: &Once, make_ctx: impl FnOnce() -> Ctx + Send + 'static) -> Option<std::thread::JoinHandle<()>> {
    let mut handle = None;
    once.call_once(|| {
        handle = std::thread::Builder::new()
            .name("flowblur-update-check".to_string())
            .spawn(move || {
                let _ = catch_unwind(AssertUnwindSafe(|| {
                    let ctx = make_ctx();
                    let outcome = check_once(&ctx);
                    (ctx.log)(&outcome.log_line());
                }));
            })
            .ok();
    });
    handle
}

// ───────────────────────────── the real environment ─────────────────────────────

/// The user's home directory: absolute, not `/`, a directory owned by the effective user.
fn home_dir() -> Option<PathBuf> {
    use std::os::unix::fs::MetadataExt;
    let home = std::env::home_dir()?;
    if !home.is_absolute() || home == Path::new("/") {
        return None;
    }
    let meta = std::fs::metadata(&home).ok()?;
    (meta.is_dir() && meta.uid() == diag::current_euid()).then_some(home)
}

fn state_dir() -> Option<PathBuf> {
    Some(home_dir()?.join("Library").join("Application Support").join("FlowBlur"))
}

fn unix_now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

fn curl_command() -> Command {
    let mut cmd = Command::new(CURL);
    cmd.env_clear().args([
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
        "-A", &format!("FlowBlur/{VERSION}"),
        LATEST_URL,
    ]);
    cmd
}

fn fetch_latest() -> Option<Version> {
    parse_redirect(&run_capped(curl_command(), CURL_TIME_LIMIT, MAX_OUTPUT_BYTES).usable_output()?)
}

fn real_ctx() -> Ctx {
    Ctx {
        dir: state_dir(),
        now: Box::new(unix_now),
        fetch: Box::new(fetch_latest),
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

/// The newer release found by an earlier check, if any. Reads only the small state file.
pub fn available_update() -> Option<Version> {
    catch_unwind(|| available_update_in(state_dir().as_deref(), running_version())).ok().flatten()
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
        let _ = std::thread::Builder::new().name("flowblur-open-releases".to_string()).spawn(|| {
            let _ = catch_unwind(|| {
                let mut cmd = Command::new(OPEN);
                cmd.arg(RELEASES_URL);
                run_capped(cmd, OPEN_TIME_LIMIT, 0)
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
    /// The time limit passed; the child was killed and reaped.
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
    pid: Option<u32>,
    end: RunEnd,
    /// Standard output, or `None` when it exceeded the cap or could not be read.
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

/// Waiting for and signalling a child, so tests can stand in for a host that reaps children
/// itself and can see which signals are sent.
trait ChildControl {
    fn try_wait(&mut self, child: &mut std::process::Child) -> std::io::Result<Option<std::process::ExitStatus>>;
    fn kill(&mut self, child: &mut std::process::Child);
}

struct RealChildControl;

impl ChildControl for RealChildControl {
    fn try_wait(&mut self, child: &mut std::process::Child) -> std::io::Result<Option<std::process::ExitStatus>> {
        child.try_wait()
    }
    fn kill(&mut self, child: &mut std::process::Child) {
        let _ = child.kill();
    }
}

/// Reads at most `cap` bytes; more than `cap` is `None`. Dropping the pipe afterwards makes a
/// child that keeps writing fail on its next write.
fn read_capped(out: impl Read, cap: usize) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    out.take(cap as u64 + 1).read_to_end(&mut buf).ok()?;
    (buf.len() <= cap).then_some(buf)
}

/// Runs `cmd` (no shell) with stdin and stderr null and stdout piped and read up to `cap` bytes,
/// killing it once `limit` has passed. The child is reaped here unless the host reaps it first.
fn run_capped(cmd: Command, limit: Duration, cap: usize) -> RunResult {
    run_capped_with(cmd, limit, cap, &mut RealChildControl)
}

fn run_capped_with(mut cmd: Command, limit: Duration, cap: usize, control: &mut dyn ChildControl) -> RunResult {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::null());
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(_) => return RunResult { pid: None, end: RunEnd::SpawnFailed, output: None },
    };
    let pid = Some(child.id());
    let deadline = Instant::now() + limit;

    // The pipe is read on its own thread so a child that never closes it cannot stall the
    // time limit. If that thread cannot start, the pipe is closed and the output is `None`.
    let (tx, rx) = std::sync::mpsc::channel();
    if let Some(out) = child.stdout.take() {
        let _ = std::thread::Builder::new().name("flowblur-update-read".to_string()).spawn(move || {
            let read = catch_unwind(AssertUnwindSafe(|| read_capped(out, cap))).ok().flatten();
            let _ = tx.send(read);
        });
    }

    // Darwin `<sys/errno.h>`: no such child, i.e. it has already been reaped.
    const ECHILD: i32 = 10;
    let end = loop {
        match control.try_wait(&mut child) {
            Ok(Some(status)) => break RunEnd::Exited(status.code()),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL_INTERVAL),
            Ok(None) => {
                // `try_wait` has just reported the child as running, so the pid is still ours.
                control.kill(&mut child);
                let _ = child.wait();
                break RunEnd::TimedOut;
            }
            // After a wait error the pid may already be free and reused, so no signal is sent.
            Err(e) if e.raw_os_error() == Some(ECHILD) => break RunEnd::ReapedElsewhere,
            Err(_) => break RunEnd::WaitFailed,
        }
    };
    if end == RunEnd::WaitFailed {
        // The child may still be running and holding the pipe, and its output is not used.
        return RunResult { pid, end, output: None };
    }
    // The child is gone, so the pipe reaches end-of-file; the timeout only guards against a
    // process the child left behind that still holds the pipe open.
    let output = rx.recv_timeout(Duration::from_secs(1)).ok().flatten();
    RunResult { pid, end, output }
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
            self.0.join("FlowBlur")
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
            "https://github.com/someone/flowblur/releases/tag/v1.0.0",
            "https://github.com/treydoe1/other/releases/tag/v1.0.0",
            "https://gitlab.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "https://github.com.example.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "https://GitHub.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "https://github.com:443/treydoe1/flowblur/releases/tag/v1.0.0",
            "https://user@github.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "http://github.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "HTTPS://github.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "//github.com/treydoe1/flowblur/releases/tag/v1.0.0",
            "/treydoe1/flowblur/releases/tag/v1.0.0",
            "https://github.com/treydoe1/flowblur/releases",
            "https://github.com/treydoe1/flowblur/releases/latest",
            "https://github.com/treydoe1/flowblur/releases/tag",
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
        let dir = t.0.join("missing").join("FlowBlur");
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

        // A regular file where the directory should be.
        let t = TempDir::new();
        std::fs::write(t.state(), b"").unwrap();
        let (ctx, _) = ctx_with(Some(t.state()), NOW, None);
        assert_eq!(check_once(&ctx), Outcome::BadDir);

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

    fn is_reaped(pid: u32) -> bool {
        unsafe extern "C" {
            fn kill(pid: i32, sig: i32) -> i32;
        }
        const ESRCH: i32 = 3;
        // SAFETY: signal 0 only checks whether the process exists; nothing is delivered.
        let r = unsafe { kill(pid as i32, 0) };
        r == -1 && std::io::Error::last_os_error().raw_os_error() == Some(ESRCH)
    }

    #[test]
    fn runner_kills_and_reaps_on_timeout() {
        let mut cmd = Command::new("/bin/sleep");
        cmd.arg("30");
        let t0 = Instant::now();
        let r = run_capped(cmd, Duration::from_millis(300), MAX_OUTPUT_BYTES);
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        assert_eq!(r.end, RunEnd::TimedOut);
        assert!(is_reaped(r.pid.unwrap()));
        assert_eq!(r.usable_output(), None);
    }

    #[test]
    fn runner_enforces_the_output_cap() {
        // `yes` writes forever; the reader stops after the cap and closes the pipe.
        let t0 = Instant::now();
        let r = run_capped(Command::new("/usr/bin/yes"), Duration::from_secs(10), MAX_OUTPUT_BYTES);
        assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
        assert_eq!(r.output, None);
        assert!(is_reaped(r.pid.unwrap()));

        // Exactly the cap is accepted, one byte more is not, even with exit status 0.
        let mut ok = Command::new("/bin/sh");
        ok.args(["-c", "head -c 1024 /dev/zero"]);
        assert_eq!(run_capped(ok, Duration::from_secs(10), MAX_OUTPUT_BYTES).usable_output().map(|o| o.len()), Some(1024));
        let mut over = Command::new("/bin/sh");
        over.args(["-c", "head -c 1025 /dev/zero"]);
        let r = run_capped(over, Duration::from_secs(10), MAX_OUTPUT_BYTES);
        assert_eq!(r.end, RunEnd::Exited(Some(0)));
        assert_eq!(r.usable_output(), None);
    }

    #[test]
    fn runner_reports_only_successful_output() {
        let mut echo = Command::new("/bin/echo");
        echo.arg("hello");
        assert_eq!(run_capped(echo, Duration::from_secs(10), MAX_OUTPUT_BYTES).usable_output(), Some(b"hello\n".to_vec()));

        let r = run_capped(Command::new("/usr/bin/false"), Duration::from_secs(10), MAX_OUTPUT_BYTES);
        assert_eq!(r.end, RunEnd::Exited(Some(1)));
        assert!(is_reaped(r.pid.unwrap()));

        let mut fail = Command::new("/bin/sh");
        fail.args(["-c", &format!("printf '{TAG_URL_PREFIX}v9.9.9'; exit 3")]);
        let r = run_capped(fail, Duration::from_secs(10), MAX_OUTPUT_BYTES);
        assert_eq!(r.end, RunEnd::Exited(Some(3)));
        assert_eq!(r.usable_output(), None);

        let mut killed = Command::new("/bin/sh");
        killed.args(["-c", &format!("printf '{TAG_URL_PREFIX}v9.9.9'; kill -9 $$")]);
        assert_eq!(run_capped(killed, Duration::from_secs(10), MAX_OUTPUT_BYTES).usable_output(), None);

        let r = run_capped(Command::new("/nonexistent-fbuc-binary"), Duration::from_secs(1), MAX_OUTPUT_BYTES);
        assert_eq!((r.end, r.pid), (RunEnd::SpawnFailed, None));
    }

    /// Stands in for a host that reaps children itself (SIGCHLD ignored, or its own
    /// `waitpid`): before the runner's first wait it reaps the child with a blocking `waitpid` on
    /// the child's pid, so the runner's real `Child::try_wait` then fails with ECHILD. With
    /// `other_error`, it reports a different wait error instead. It counts kill requests.
    struct HostReaps {
        other_error: bool,
        kills: usize,
    }

    impl ChildControl for HostReaps {
        fn try_wait(&mut self, child: &mut std::process::Child) -> std::io::Result<Option<std::process::ExitStatus>> {
            unsafe extern "C" {
                fn waitpid(pid: i32, status: *mut i32, options: i32) -> i32;
            }
            let pid = child.id() as i32;
            let mut status = 0;
            // SAFETY: `status` is a valid out-pointer; this reaps only our own child.
            assert_eq!(unsafe { waitpid(pid, &mut status, 0) }, pid);
            if self.other_error {
                const EINVAL: i32 = 22;
                return Err(std::io::Error::from_raw_os_error(EINVAL));
            }
            child.try_wait()
        }
        fn kill(&mut self, _: &mut std::process::Child) {
            self.kills += 1;
        }
    }

    fn printf_cmd(text: &str, exit: i32) -> Command {
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", &format!("printf '{text}'; exit {exit}")]);
        cmd
    }

    #[test]
    fn output_is_used_when_the_host_reaps_the_child() {
        let url = format!("{TAG_URL_PREFIX}v0.7.0");
        // The exit status is unknown once someone else has reaped the child; the output is used
        // and validated by the parser.
        for exit in [0, 3] {
            let mut host = HostReaps { other_error: false, kills: 0 };
            let r = run_capped_with(printf_cmd(&url, exit), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
            assert_eq!(r.end, RunEnd::ReapedElsewhere);
            assert_eq!(host.kills, 0, "no signal after a wait error");
            assert!(is_reaped(r.pid.unwrap()));
            assert_eq!(r.usable_output().as_deref().and_then(parse_redirect), Some(v(0, 7, 0)));
        }

        // Unexpected output is still no information.
        let mut host = HostReaps { other_error: false, kills: 0 };
        let r = run_capped_with(printf_cmd("https://example.com/", 0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
        assert_eq!(r.end, RunEnd::ReapedElsewhere);
        assert_eq!(r.usable_output().as_deref().and_then(parse_redirect), None);

        // An over-cap output is still no information.
        let mut host = HostReaps { other_error: false, kills: 0 };
        let r = run_capped_with(Command::new("/usr/bin/yes"), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
        assert_eq!((r.end, host.kills), (RunEnd::ReapedElsewhere, 0));
        assert_eq!(r.usable_output(), None);
    }

    #[test]
    fn other_wait_errors_send_no_signal_and_give_no_information() {
        let mut host = HostReaps { other_error: true, kills: 0 };
        let r = run_capped_with(printf_cmd(&format!("{TAG_URL_PREFIX}v0.7.0"), 0), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut host);
        assert_eq!(r.end, RunEnd::WaitFailed);
        assert_eq!(host.kills, 0, "no signal after a wait error");
        assert_eq!(r.usable_output(), None);
    }

    #[test]
    fn only_the_timeout_path_sends_a_signal() {
        /// The real wait and kill, counting kill requests.
        struct Counting(usize);
        impl ChildControl for Counting {
            fn try_wait(&mut self, child: &mut std::process::Child) -> std::io::Result<Option<std::process::ExitStatus>> {
                child.try_wait()
            }
            fn kill(&mut self, child: &mut std::process::Child) {
                self.0 += 1;
                let _ = child.kill();
            }
        }
        let mut sleep = Command::new("/bin/sleep");
        sleep.arg("30");
        let mut c = Counting(0);
        let r = run_capped_with(sleep, Duration::from_millis(200), MAX_OUTPUT_BYTES, &mut c);
        assert_eq!((r.end, c.0), (RunEnd::TimedOut, 1));
        assert!(is_reaped(r.pid.unwrap()));

        let mut c = Counting(0);
        let r = run_capped_with(Command::new("/usr/bin/true"), Duration::from_secs(10), MAX_OUTPUT_BYTES, &mut c);
        assert_eq!((r.end, c.0), (RunEnd::Exited(Some(0)), 0));
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
    fn curl_command_line() {
        let cmd = curl_command();
        assert_eq!(cmd.get_program(), CURL);
        let args: Vec<_> = cmd.get_args().map(|a| a.to_str().unwrap().to_string()).collect();
        let ua = format!("FlowBlur/{VERSION}");
        let want = [
            "-q", "--proto", "=https", "--proto-redir", "=https", "--tlsv1.2", "--connect-timeout", "5",
            "--max-time", "8", "--max-redirs", "0", "-sS", "-o", "/dev/null", "-w", "%{redirect_url}",
            "-A", &ua, LATEST_URL,
        ];
        assert_eq!(args, want);
        assert_eq!(args[0], "-q", "-q only works as the first argument");
        assert_eq!(cmd.get_envs().count(), 0);
    }

    /// Runs the real `curl` against GitHub and prints what it parsed. Run by hand:
    /// `cargo test --release live_fetch -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn live_fetch() {
        let r = run_capped(curl_command(), CURL_TIME_LIMIT, MAX_OUTPUT_BYTES);
        println!("end={:?} output_bytes={:?}", r.end, r.output.as_ref().map(Vec::len));
        match r.usable_output().as_deref().and_then(parse_redirect) {
            Some(v) => println!("latest release: {v}"),
            None => println!("no information (no release, no network, or an unexpected reply)"),
        }
    }
}
