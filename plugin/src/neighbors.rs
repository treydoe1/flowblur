//! Host-independent logic for choosing the previous/next frames and their frame distances.
//!
//! Two strategies:
//! * LEGACY (`find_neighbor`, `frame_distances`): step whole timeline frames (`time_step`) away
//!   from `current_time`, skipping held (duplicate) frames. Used in After Effects, when "Use
//!   Source Frame Rate" is off, and whenever the source strategy rejects its result.
//! * SOURCE (`source_rate`, `choose_time_rep`, `ladder`, `source_frame_distance`): Premiere hands
//!   the effect MEDIA time, and `time_step` is one sequence frame of media time whatever the clip
//!   speed (measured in Premiere Pro 2027). Stepping by `time_step` therefore skips source frames
//!   when the media frame rate is higher than the sequence's. Instead we probe at fractions of
//!   one source-frame duration `D` so that the first non-duplicate in each direction is exactly
//!   the adjacent source frame, and hand the core a fractional frame distance.
//!
//! Since 1.3, `D` is always the host's original clip frame rate `D_orig`
//! (`Utility::original_clip_frame_rate`, `ticks_to_units`), and the source strategy is only
//! attempted when `source_gate` passes: D_orig exists and is clearly shorter than one sequence
//! frame (the media has a higher frame rate than the sequence) and the track item is not
//! synthetic. Everything else (adjustment layers, nested sequences, same-rate footage) uses the
//! legacy path without any extra checkout. There is no measurement of D any more (1.2's
//! bisection cost ~22 checkouts, multi-second stalls when each checkout re-renders a composite,
//! and gave unstable D on continuously changing input): the ladder's own duplicate pattern
//! validates D_orig every render (`evaluate`, `Class`), and a small per-clip state
//! (`SourceState`: Unknown / Verified / Rejected plus strikes) decides between source and legacy
//! path. In Premiere Pro 2027 `Utility::media_frame_rate` returned the SEQUENCE rate;
//! `source_rate` (D1/D2 from the reported rates) is kept for the diagnostics log only.

/// Premiere's PrTime ticks per second.
pub const TICKS_PER_SECOND: i64 = 254_016_000_000;

/// How far (in timeline frames) we look in each direction for a frame that is not a repeat.
pub const MAX_K: u32 = 4;

/// Tries k = 1..=max_k. `probe(k)` returns the frame at offset k if it exists and is usable
/// (non-null, same size, decodable format); `is_dup` says whether it repeats the current frame.
/// Returns the first usable, non-duplicate frame and its k, or `None` (clip edge, long hold).
/// Missing / unusable offsets are skipped, not fatal, so one bad checkout does not stop the search.
pub fn find_neighbor<T>(max_k: u32, mut probe: impl FnMut(u32) -> Option<T>, mut is_dup: impl FnMut(&T) -> bool) -> Option<(u32, T)> {
    for k in 1..=max_k {
        if let Some(frame) = probe(k)
            && !is_dup(&frame) {
            return Some((k, frame));
        }
    }
    None
}

/// Frame distances handed to the core. If both directions were found, each source frame is
/// held for H = k_prev + k_next - 1 timeline frames and both neighbours get H; a lone neighbour
/// gets its own k.
pub fn frame_distances(k_prev: Option<u32>, k_next: Option<u32>) -> (Option<f32>, Option<f32>) {
    match (k_prev, k_next) {
        (Some(p), Some(n)) => {
            let h = (p + n - 1) as f32;
            (Some(h), Some(h))
        }
        (p, n) => (p.map(|k| k as f32), n.map(|k| k as f32)),
    }
}

/// Times to check out for offset k in direction `dir` (-1 or +1). Uses |time_step| so a
/// time-reversed layer (negative step) still looks at the previous/next displayed frame.
/// Returns `None` on zero step or i32 overflow.
pub fn offset_time(current_time: i32, time_step: i32, k: u32, dir: i32) -> Option<i32> {
    if time_step == 0 {
        return None;
    }
    let t = current_time as i64 + dir as i64 * k as i64 * (time_step as i64).abs();
    i32::try_from(t).ok()
}

// ───────────────────────────── source-frame strategy ─────────────────────────────

/// Clip timing reported by the host (Premiere's PF Utility suite / Sequence Info suite). Every
/// field is optional: each host call may fail independently.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ClipTiming {
    /// PrTime ticks per media (source) frame.
    pub media_ticks: Option<i64>,
    /// PrTime ticks per sequence frame.
    pub seq_ticks: Option<i64>,
    /// Clip speed multiplier as reported (sign = direction).
    pub speed: Option<f64>,
}

/// Source-frame rates outside 0.5 .. 2000 fps are treated as nonsense host values.
fn plausible_ticks(t: Option<i64>) -> Option<i64> {
    t.filter(|&t| (TICKS_PER_SECOND / 2000..=TICKS_PER_SECOND * 2).contains(&t))
}

/// |speed|, with a missing, non-finite or zero speed treated as 1.
pub fn norm_speed(s: Option<f64>) -> f64 {
    match s {
        Some(v) if v.is_finite() && v != 0.0 => v.abs(),
        _ => 1.0,
    }
}

/// Source frames per `time_step` must lie in this range for the source strategy (beyond it the
/// estimate of D is considered absurd and the legacy path is used).
pub const MIN_SOURCE_FRAMES_PER_STEP: f64 = 1.0 / 64.0;
pub const MAX_SOURCE_FRAMES_PER_STEP: f64 = 64.0;

/// Both estimates of D (one source frame in `in_data` time units) and the choice made.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SourceRate {
    /// `media_ticks / TICKS_PER_SECOND * time_scale` (assumes time_scale = units per media second).
    pub d1: Option<f64>,
    /// `|time_step| * media_ticks / seq_ticks` (assumes time_step = one sequence frame).
    pub d2: Option<f64>,
    /// `norm_speed(timing.speed)`.
    pub speed: f64,
    /// `Ok((D, a))` with a = source frames the media advances per sequence frame
    /// = |time_step| / D * speed; `Err(reason)` -> legacy path.
    pub choice: Result<(f64, f64), &'static str>,
}

/// Computes D1, D2 and picks D (D2 when available, else D1), rejecting absurd values.
pub fn source_rate(t: &ClipTiming, time_step: i32, time_scale: u32) -> SourceRate {
    let step = (time_step as f64).abs();
    let speed = norm_speed(t.speed);
    let media = plausible_ticks(t.media_ticks);
    let seq = plausible_ticks(t.seq_ticks);
    let d1 = media.filter(|_| time_scale > 0).map(|m| m as f64 / TICKS_PER_SECOND as f64 * time_scale as f64);
    let d2 = media.zip(seq).filter(|_| step > 0.0).map(|(m, s)| step * m as f64 / s as f64);
    let choice = (|| {
        if step == 0.0 {
            return Err("zero time step");
        }
        if media.is_none() {
            return Err("no media frame rate");
        }
        let d = d2.or(d1).ok_or("no time scale")?;
        if !(d.is_finite() && d > 0.0) {
            return Err("source frame duration not finite/positive");
        }
        let per_step = step / d;
        if !(MIN_SOURCE_FRAMES_PER_STEP..=MAX_SOURCE_FRAMES_PER_STEP).contains(&per_step) {
            return Err("source frames per step out of range");
        }
        Ok((d, per_step * speed))
    })();
    SourceRate { d1, d2, speed, choice }
}

/// Frame distance handed to the core for a source-strategy neighbour: 1 / max(a, 1).
/// a > 1 (e.g. 120 fps media on a 24 fps sequence, a = 5 -> 0.2): the adjacent source frame is
/// a fraction of an output frame away. a <= 1 (slowed-down footage, source frames held): the blur
/// stays relative to the source frame interval, as the legacy path does.
pub fn source_frame_distance(a: f64) -> f32 {
    let fd = 1.0 / a.max(1.0);
    (fd as f32).max(1.0 / 64.0)
}

/// Minimum length of one source frame, in the units passed to the host, for the ladder offsets
/// (multiples of D/4) to survive rounding to integer times plus about one unit of host jitter.
pub const MIN_UNITS_PER_SOURCE_FRAME: f64 = 8.0;
/// Largest power-of-two multiplier applied to (time, step, scale) to reach that resolution.
pub const MAX_TIME_MULT: u32 = 256;

/// Time representation used for checkouts in the source strategy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimeRep {
    /// Multiplier applied to what_time, time_step and time_scale (1 = native).
    pub m: u32,
    pub now: i32,
    pub step: i32,
    pub scale: u32,
    /// D in this representation's units (D * m).
    pub d: f64,
}

/// Native units when D >= 8, else the smallest power of two m <= 256 with D*m >= 8 (checked
/// arithmetic). `None` -> legacy path.
pub fn choose_time_rep(now: i32, step: i32, scale: u32, d: f64) -> Option<TimeRep> {
    if !(d.is_finite() && d > 0.0) {
        return None;
    }
    let mut m = 1u32;
    while d * (m as f64) < MIN_UNITS_PER_SOURCE_FRAME {
        m = m.checked_mul(2).filter(|&m| m <= MAX_TIME_MULT)?;
    }
    let mi = i32::try_from(m).ok()?;
    Some(TimeRep { m, now: now.checked_mul(mi)?, step: step.checked_mul(mi)?, scale: scale.checked_mul(m)?, d: d * m as f64 })
}

/// Probe offsets in units of D relative to current_time, in the order tried.
///
/// Why these values: let the current frame's index be f and its phase inside the frame
/// φ ∈ [0,1), so a probe at offset r lands on floor(f + φ + r + e) with e the rounding/jitter
/// error in units of D.
/// * previous: -0.25 -> f-1 if φ < 0.25 else f (dup); -0.75 (reached only when φ >= 0.25)
///   -> f-1 if φ < 0.75 else f (dup); -1.25 (reached only when φ >= 0.75) -> f-1.
/// * next: +0.75 -> f+1 if φ >= 0.25 else f (dup); +1.25 (reached only when φ < 0.25) -> f+1.
///
/// So the first non-duplicate is always exactly f±1, and in total at most 2 probes are
/// duplicates (φ < 0.25: 0+1; 0.25..0.75: 1+0; >= 0.75: 2+0). Each step above keeps a margin of
/// 0.25 D, so |e| < 0.25 D is tolerated (round-to-nearest of the probe time contributes up to
/// 0.5 unit; with D >= 8 units that leaves 1.5 units for host jitter). The same holds for a host
/// that maps time to frames by rounding to nearest instead of flooring (it only shifts φ).
/// Do not add rungs without re-deriving this.
pub const PREV_RUNGS: [f64; 3] = [-0.25, -0.75, -1.25];
pub const NEXT_RUNGS: [f64; 2] = [0.75, 1.25];

/// Result of one probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The host returned no frame (error or null data, e.g. beyond the clip).
    Missing,
    /// A frame came back but cannot be used (size, format, transparent, time overflow).
    Unusable,
    /// A frame identical to the reference (the current frame; for the edge test, the neighbour).
    Duplicate,
    /// Ladder: the frame chosen as neighbour. Confirmation / edge test: a frame that differs
    /// from the reference.
    Used,
}

impl Outcome {
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Missing => "missing",
            Outcome::Unusable => "unusable",
            Outcome::Duplicate => "duplicate",
            Outcome::Used => "used",
        }
    }
}

/// What a probe was for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Prev,
    Next,
    /// `CONFIRM_RUNG` probe after a (0, 0) duplicate pattern.
    Confirm,
    /// `EDGE_TEST_RUNG` probe beyond a lone neighbour.
    Edge,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Prev => "prev",
            Role::Next => "next",
            Role::Confirm => "confirm",
            Role::Edge => "edge",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeRecord {
    pub role: Role,
    /// Requested time in the representation's units (i64 so an overflowing time can be logged).
    pub time: i64,
    pub outcome: Outcome,
}

/// How one side of the ladder ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SideKind {
    /// A non-duplicate frame was found.
    Found,
    /// No non-duplicate; at least one probe returned a frame (a duplicate of the current one).
    AllDuplicate,
    /// No probe returned a usable frame (missing / unusable only).
    Nothing,
}

pub struct Side<T> {
    pub kind: SideKind,
    pub frame: Option<T>,
    /// Time of the probe that found `frame`.
    pub time: Option<i64>,
    /// Duplicate probes on this side.
    pub dups: u32,
}

pub struct Ladder<T> {
    pub prev: Side<T>,
    pub next: Side<T>,
    pub probes: Vec<ProbeRecord>,
}

fn probe_at<T>(t: i64, probe: &mut impl FnMut(i32) -> Result<T, Outcome>) -> Result<T, Outcome> {
    i32::try_from(t).map_err(|_| Outcome::Unusable).and_then(probe)
}

/// Runs the probe ladder around `now` with source-frame length `d` (both in the same units) and
/// reports each side as it is; judging the pattern is `evaluate`'s job. `probe(time)` checks out
/// a frame (`Err(Missing | Unusable)` if none); `is_dup` compares with the current frame. Each
/// side stops at its first non-duplicate.
pub fn ladder<T>(now: i32, d: f64, mut probe: impl FnMut(i32) -> Result<T, Outcome>, mut is_dup: impl FnMut(&T) -> bool) -> Ladder<T> {
    let mut probes = Vec::new();
    let mut run = |role: Role, rungs: &[f64], probes: &mut Vec<ProbeRecord>| -> Side<T> {
        let (mut dups, mut got_frame) = (0u32, false);
        for &r in rungs {
            let t = now as i64 + (r * d).round() as i64;
            match probe_at(t, &mut probe) {
                Ok(frame) => {
                    got_frame = true;
                    if is_dup(&frame) {
                        dups += 1;
                        probes.push(ProbeRecord { role, time: t, outcome: Outcome::Duplicate });
                    } else {
                        probes.push(ProbeRecord { role, time: t, outcome: Outcome::Used });
                        return Side { kind: SideKind::Found, frame: Some(frame), time: Some(t), dups };
                    }
                }
                Err(o) => probes.push(ProbeRecord { role, time: t, outcome: o }),
            }
        }
        Side { kind: if got_frame { SideKind::AllDuplicate } else { SideKind::Nothing }, frame: None, time: None, dups }
    };
    let prev = run(Role::Prev, &PREV_RUNGS, &mut probes);
    let next = run(Role::Next, &NEXT_RUNGS, &mut probes);
    Ladder { prev, next, probes }
}

// ───────────────────────────── validating D_orig ─────────────────────────────

/// Offset (units of D) of the confirmation probe after a (0, 0) duplicate pattern.
pub const CONFIRM_RUNG: f64 = 0.13;
/// Offset (units of D) of the edge-test probe beyond a lone neighbour, away from the current frame.
pub const EDGE_TEST_RUNG: f64 = 1.25;

/// What one render's ladder says about D = D_orig.
///
/// With source frames of length exactly D and phase φ of `now` in its frame f (see
/// `PREV_RUNGS`), the duplicate counts (prev, next) are (0, 1) for φ < 0.25, (1, 0) for
/// 0.25 <= φ < 0.75 and (2, 0) for φ >= 0.75. Exact-quarter phases are common (119.88 fps at
/// 25%, 29.97 fps on 23.976 cycle φ through 0, 0.25, 0.5, 0.75; 120.0 fps drifts through all
/// phases); there one rung lands on a frame boundary and may come back either way, which can
/// also give (1, 1) (φ ≈ 0.75 or 0) or (0, 0) (φ ≈ 0.25). The neighbours are still f±1.
/// (0, 0) is otherwise the signature of input that changes inside a source frame, hence the
/// confirmation probe at +0.13 D: still f at φ ≈ 0.25, a new frame for continuous input.
/// With a true frame k >= 2 times longer than D, every interior frame shows either 3
/// duplicates (both sides found) or an all-duplicate side, and the edge-test probe beyond a lone
/// neighbour stays inside the neighbour's (longer) frame, so it is a duplicate of it.
///
/// | ladder                              | Unknown                      | Verified        |
/// |-------------------------------------|------------------------------|-----------------|
/// | both found, 3 dups                  | Anomaly                      | Anomaly         |
/// | both found, 0 dups, confirm dup     | Consistent                   | Consistent      |
/// | both found, 0 dups, confirm differs | Continuous                   | Continuous      |
/// | both found, 0 dups, confirm missing | Inconclusive                 | Inconclusive    |
/// | both found, 1 or 2 dups             | Consistent                   | Consistent      |
/// | one found, edge test differs        | EdgeConfirmed                | Edge (no test)  |
/// | one found, edge test duplicate      | Anomaly                      | Edge (no test)  |
/// | one found, edge test missing        | Inconclusive                 | Edge (no test)  |
/// | neither found                       | UnverifiedNoNeighbours       | NoNeighbours    |
///
/// Neither side found on an Unknown key uses the legacy path: re-interpreted footage (true
/// frame >= 3x D_orig) at a mid phase reaches neither neighbour with these rungs, and the
/// legacy search then finds its real neighbours; genuinely frozen content gets none either way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// Both adjacent frames, pattern consistent with D: use both.
    Consistent,
    /// Unknown key, one neighbour, and the edge test showed a genuine clip edge: use it.
    EdgeConfirmed,
    /// Verified key, one neighbour (clip edge): use it, no edge test.
    Edge,
    /// Verified key, neither side found a frame (frozen content, long hold, single frame): no
    /// neighbours on the source path.
    NoNeighbours,
    /// Unknown key, neither side found a frame: legacy for this render, state unchanged.
    UnverifiedNoNeighbours,
    /// Input changes inside a source frame (frame blending, retiming, nested content): legacy.
    Continuous,
    /// The true frame is longer than D (e.g. Interpret Footage): legacy.
    Anomaly,
    /// The extra probe returned nothing: legacy for this render, state unchanged.
    Inconclusive,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Consistent => "consistent",
            Class::EdgeConfirmed => "edge_confirmed",
            Class::Edge => "edge",
            Class::NoNeighbours => "no_neighbours",
            Class::UnverifiedNoNeighbours => "unverified_no_neighbours",
            Class::Continuous => "continuous",
            Class::Anomaly => "anomaly",
            Class::Inconclusive => "inconclusive",
        }
    }
    /// False: this render uses the legacy path. (`NoNeighbours` stays on the source path with
    /// no neighbours, so the core leaves the frame unblurred.)
    pub fn uses_source(self) -> bool {
        matches!(self, Class::Consistent | Class::EdgeConfirmed | Class::Edge | Class::NoNeighbours)
    }
    /// Reason logged when `!uses_source()`.
    pub fn legacy_reason(self) -> &'static str {
        match self {
            Class::Continuous => "continuous input (changes inside a source frame)",
            Class::Anomaly => "source frame longer than original clip frame rate",
            Class::Inconclusive => "edge or confirmation probe returned no frame",
            Class::UnverifiedNoNeighbours => "no source-frame neighbours on an unverified clip",
            _ => "",
        }
    }
}

pub struct Evaluation<T> {
    pub class: Class,
    /// Neighbours to use (both `None` unless `class.uses_source()`).
    pub prev: Option<T>,
    pub next: Option<T>,
    /// (prev, next) duplicate counts of the ladder.
    pub dups: (u32, u32),
    pub probes: Vec<ProbeRecord>,
}

/// Ladder with D = D_orig (`d`, representation units) plus at most one extra probe, classified
/// as in the `Class` table. `verified`: the key's state is Verified. `is_dup` compares with the
/// current frame, `same` compares two checked-out frames (edge test). At most 6 probes.
pub fn evaluate<T>(
    now: i32,
    d: f64,
    verified: bool,
    mut probe: impl FnMut(i32) -> Result<T, Outcome>,
    mut is_dup: impl FnMut(&T) -> bool,
    mut same: impl FnMut(&T, &T) -> bool,
) -> Evaluation<T> {
    let Ladder { prev, next, mut probes } = ladder(now, d, &mut probe, &mut is_dup);
    let dups = (prev.dups, next.dups);
    let off = |r: f64| (r * d).round() as i64;
    let mut extra = |role: Role, t: i64, probes: &mut Vec<ProbeRecord>, reference: &mut dyn FnMut(&T) -> bool| -> Option<bool> {
        let r = probe_at(t, &mut probe);
        let (outcome, dup) = match &r {
            Ok(f) if reference(f) => (Outcome::Duplicate, Some(true)),
            Ok(_) => (Outcome::Used, Some(false)),
            Err(o) => (*o, None),
        };
        probes.push(ProbeRecord { role, time: t, outcome });
        dup
    };
    let class = match (&prev.frame, &next.frame) {
        (Some(_), Some(_)) => match dups.0 + dups.1 {
            3.. => Class::Anomaly,
            0 => match extra(Role::Confirm, now as i64 + off(CONFIRM_RUNG), &mut probes, &mut is_dup) {
                Some(true) => Class::Consistent,
                Some(false) => Class::Continuous,
                None => Class::Inconclusive,
            },
            _ => Class::Consistent,
        },
        (Some(_), None) | (None, Some(_)) if verified => Class::Edge,
        (Some(n), None) | (None, Some(n)) => {
            let (side, dir) = if prev.frame.is_some() { (&prev, -1) } else { (&next, 1) };
            let t = side.time.unwrap_or(now as i64) + dir * off(EDGE_TEST_RUNG);
            match extra(Role::Edge, t, &mut probes, &mut |f| same(f, n)) {
                Some(false) => Class::EdgeConfirmed,
                Some(true) => Class::Anomaly,
                None => Class::Inconclusive,
            }
        }
        (None, None) if verified => Class::NoNeighbours,
        (None, None) => Class::UnverifiedNoNeighbours,
    };
    let keep = class.uses_source();
    Evaluation { class, prev: prev.frame.filter(|_| keep), next: next.frame.filter(|_| keep), dups, probes }
}

/// Verdict on D_orig for one cache key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Verdict {
    #[default]
    Unknown,
    Verified,
    /// Permanent for the key within the process: legacy path, no ladder.
    Rejected,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Unknown => "unknown",
            Verdict::Verified => "verified",
            Verdict::Rejected => "rejected",
        }
    }
}

/// Strikes (anomalies, continuous frames on a Verified key) that make a key Rejected.
pub const MAX_STRIKES: u8 = 3;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceState {
    pub verdict: Verdict,
    pub strikes: u8,
}

impl SourceState {
    /// State after a render classified as `class`: consistent frames (and a confirmed edge)
    /// verify and clear strikes; continuous input rejects an Unknown key at once and strikes a
    /// Verified one; an anomaly strikes; `MAX_STRIKES` strikes reject; Rejected never changes.
    pub fn after(self, class: Class) -> SourceState {
        if self.verdict == Verdict::Rejected {
            return self;
        }
        let strike = |s: SourceState| {
            let strikes = s.strikes.saturating_add(1);
            SourceState { verdict: if strikes >= MAX_STRIKES { Verdict::Rejected } else { s.verdict }, strikes }
        };
        match class {
            Class::Consistent | Class::EdgeConfirmed => SourceState { verdict: Verdict::Verified, strikes: 0 },
            Class::Continuous if self.verdict == Verdict::Unknown => SourceState { verdict: Verdict::Rejected, ..self },
            Class::Continuous | Class::Anomaly => strike(self),
            Class::Edge | Class::NoNeighbours | Class::UnverifiedNoNeighbours | Class::Inconclusive => self,
        }
    }
}

// ───────────────────────────── gate ─────────────────────────────

/// Converts PrTime ticks per frame to `in_data` time units (`time_scale` units per second),
/// rejecting implausible tick values (outside 0.5 .. 2000 fps).
pub fn ticks_to_units(ticks: Option<i64>, time_scale: u32) -> Option<f64> {
    plausible_ticks(ticks).filter(|_| time_scale > 0).map(|t| t as f64 / TICKS_PER_SECOND as f64 * time_scale as f64)
}

/// The source strategy is attempted only when one source frame is shorter than this fraction
/// of a sequence frame (the media clearly has a higher frame rate than the sequence).
pub const SOURCE_GATE_RATIO: f64 = 0.9;

/// Gate reason for media whose original rate is not clearly above the sequence rate (the only
/// gate reason under which `legacy_fd_rule` applies).
pub const NOT_ABOVE_SEQUENCE: &str = "source rate not above sequence rate";

/// Gate for the source strategy (the caller has already checked host and parameter).
/// `d_orig`: the original clip frame rate in native units (`ticks_to_units`, so `None` when
/// unavailable or implausible); `synthetic`: `Utility::is_track_item_effect_applied_to_synthetic`
/// (`None` when the call failed, which counts as not synthetic). Returns D_orig or the reason
/// for the legacy path. Pure arithmetic: no checkout is needed to decide.
pub fn source_gate(d_orig: Option<f64>, time_step: i32, synthetic: Option<bool>) -> Result<f64, &'static str> {
    if synthetic == Some(true) {
        return Err("synthetic track item");
    }
    let d = d_orig.filter(|d| d.is_finite() && *d > 0.0).ok_or("no original clip frame rate")?;
    let step = (time_step as f64).abs();
    if step == 0.0 {
        return Err("zero time step");
    }
    if d >= SOURCE_GATE_RATIO * step {
        return Err(NOT_ABOVE_SEQUENCE);
    }
    if step / d > MAX_SOURCE_FRAMES_PER_STEP {
        return Err("source frames per step out of range");
    }
    Ok(d)
}

/// Frame-distance rule for the legacy path when the gate said `NOT_ABOVE_SEQUENCE` (so D_orig
/// is known and plausible and the item is not synthetic). In Premiere `time_step` is one
/// sequence frame of MEDIA time whatever the speed, so the legacy H (whole steps between the
/// adjacent source frames) ignores clip speed, and on held frames alternates between runs
/// (23.976 on 59.94: H = 2, 3, 2, ...), which makes the blur pump.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LegacyFd {
    /// Unchanged: the 1.0 value.
    H,
    /// Source frames longer than a sequence frame (held): `1 / max(a, 1)` for every neighbour.
    SourceRelative,
    /// Media advances more than one source frame per sequence frame (sped up): legacy value / a.
    Divided,
}

impl LegacyFd {
    pub fn name(self) -> &'static str {
        match self {
            LegacyFd::H => "h",
            LegacyFd::SourceRelative => "source_relative",
            LegacyFd::Divided => "divided",
        }
    }
}

/// D_orig above this multiple of |time_step| means frames are held on the timeline.
pub const HELD_RATIO: f64 = 1.1;
/// `a` above this means the media is sped up.
pub const SPED_UP_A: f64 = 1.05;

/// Which rule applies, for D_orig (native units), the time step and a (`source_advance(..).a`).
pub fn legacy_fd_rule(d_orig: f64, time_step: i32, a: f64) -> LegacyFd {
    if !(d_orig.is_finite() && d_orig > 0.0 && a.is_finite() && a > 0.0) {
        LegacyFd::H
    } else if d_orig > HELD_RATIO * (time_step as f64).abs() {
        LegacyFd::SourceRelative
    } else if a > SPED_UP_A {
        LegacyFd::Divided
    } else {
        LegacyFd::H
    }
}

/// Applies `rule` to one legacy frame distance (`None` = no neighbour stays `None`).
pub fn legacy_fd(rule: LegacyFd, a: f64, d: Option<f32>) -> Option<f32> {
    d.map(|d| match rule {
        LegacyFd::H => d,
        LegacyFd::SourceRelative => source_frame_distance(a),
        LegacyFd::Divided => ((d as f64 / a) as f32).max(1.0 / 64.0),
    })
}

/// Source frames the media advances per sequence frame, both ways.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Advance {
    /// |local_time_step| / D (None when local_time_step is 0).
    pub a_local: Option<f64>,
    /// |time_step| / D * |speed|.
    pub a_speed: f64,
    /// `a_local` when available, else `a_speed`.
    pub a: f64,
}

/// In Premiere Pro 2027 `local_time_step` is the media time advanced per sequence frame (1001 at
/// 100%, 500 at 50%, 200 at 20% on a 1001-unit step), so it already includes clip speed.
pub fn source_advance(local_time_step: i32, time_step: i32, d: f64, speed: Option<f64>) -> Advance {
    let a_speed = (time_step as f64).abs() / d * norm_speed(speed);
    let a_local = (local_time_step != 0).then(|| (local_time_step as f64).abs() / d);
    Advance { a_local, a_speed, a: a_local.unwrap_or(a_speed) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_first_non_duplicate() {
        // Source frame held for 3 timeline frames: offsets 1 and 2 repeat the current frame.
        let r = find_neighbor(4, |k| Some(k), |&k| k < 3);
        assert_eq!(r, Some((3, 3)));
    }

    #[test]
    fn skips_missing_and_gives_up_after_max_k() {
        assert_eq!(find_neighbor(4, |k| if k == 1 { None } else { Some(k) }, |_| false), Some((2, 2)));
        assert_eq!(find_neighbor(4, |_| None::<u32>, |_| false), None);
        assert_eq!(find_neighbor(4, |k| Some(k), |_| true), None);
        let mut probed = vec![];
        let _ = find_neighbor(4, |k| { probed.push(k); Some(k) }, |_| true);
        assert_eq!(probed, vec![1, 2, 3, 4]);
    }

    #[test]
    fn distances() {
        assert_eq!(frame_distances(Some(1), Some(1)), (Some(1.0), Some(1.0)));
        // Current frame is the 2nd timeline frame of a 3-frame hold: k_prev = 2, k_next = 2 -> H = 3.
        assert_eq!(frame_distances(Some(2), Some(2)), (Some(3.0), Some(3.0)));
        assert_eq!(frame_distances(Some(1), Some(3)), (Some(3.0), Some(3.0)));
        assert_eq!(frame_distances(None, Some(2)), (None, Some(2.0)));
        assert_eq!(frame_distances(Some(4), None), (Some(4.0), None));
        assert_eq!(frame_distances(None, None), (None, None));
    }

    #[test]
    fn times() {
        assert_eq!(offset_time(1001 * 10, 1001, 2, -1), Some(1001 * 8));
        assert_eq!(offset_time(100, -10, 1, 1), Some(110));
        assert_eq!(offset_time(-5, 10, 1, -1), Some(-15));
        assert_eq!(offset_time(0, 0, 1, 1), None);
        assert_eq!(offset_time(i32::MAX - 1, 10, 1, 1), None);
    }

    // ── simulated media ──

    /// Deterministic host jitter in {-1, 0, +1} units.
    fn jitter(t: i64) -> i64 {
        (((t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) % 3) as i64 - 1
    }

    /// What a simulated host returns outside the clip's frames.
    #[derive(Clone, Copy, Debug)]
    enum Beyond {
        Missing,
        /// The edge frame is repeated.
        Repeat,
    }

    /// Simulated media: frame = floor(t / (l_true * m) + phase) for a time t in a representation
    /// scaled by m; frames `first..=last` exist. `jit`: probe times perturbed by +-1 unit.
    #[derive(Clone, Copy, Debug)]
    struct Sim {
        l_true: f64,
        phase: f64,
        first: i64,
        last: i64,
        beyond: Beyond,
        jit: bool,
    }

    impl Sim {
        fn open(l_true: f64, phase: f64) -> Self {
            Sim { l_true, phase, first: i64::MIN, last: i64::MAX, beyond: Beyond::Missing, jit: false }
        }
        fn raw(&self, t: i64, m: u32) -> i64 {
            (t as f64 / (self.l_true * m as f64) + self.phase).floor() as i64
        }
        fn at(&self, t: i64, m: u32) -> Result<i64, Outcome> {
            let i = self.raw(t + if self.jit { jitter(t) } else { 0 }, m);
            if (self.first..=self.last).contains(&i) {
                return Ok(i);
            }
            match self.beyond {
                Beyond::Missing => Err(Outcome::Missing),
                Beyond::Repeat => Ok(i.clamp(self.first, self.last)),
            }
        }
    }

    /// The effect's chain: choose_time_rep -> evaluate with D = D_orig (native units). Returns the
    /// evaluation and the current frame.
    fn eval(sim: Sim, now: i32, step: i32, scale: u32, d_orig: f64, verified: bool) -> (Evaluation<i64>, i64) {
        let rep = choose_time_rep(now, step, scale, d_orig).unwrap();
        let cur = Sim { jit: false, ..sim }.at(rep.now as i64, rep.m).unwrap();
        (evaluate(rep.now, rep.d, verified, |t| sim.at(t as i64, rep.m), |&f| f == cur, |a, b| a == b), cur)
    }

    const RATIOS: [f64; 9] = [1.2, 1.25, 2.0, 2.5, 4.0, 5.0, 10.0, 1.0 / 0.89, 30.0];
    /// (time_step, time_scale) pairs from very fine to very coarse, so both the native path
    /// (D >= 8 units) and the scaled path (m up to 128) are exercised.
    const REPS: [(i32, u32); 9] = [(1001, 24000), (1001, 30000), (100, 2400), (10, 240), (7, 168), (3, 72), (1, 24), (1, 25), (1, 60)];

    #[test]
    fn true_length_is_consistent_at_every_phase() {
        let mut cases = 0;
        for &(step, scale) in &REPS {
            for &ratio in &RATIOS {
                let d = step as f64 / ratio;
                assert!(source_gate(Some(d), step, None).is_ok());
                for jit in [false, true] {
                    // Phases in steps of 1/400 including the exact quarters 0, 0.25, 0.5, 0.75.
                    for p in 0..400 {
                        let sim = Sim { jit, ..Sim::open(d, p as f64 / 400.0) };
                        for verified in [false, true] {
                            let (ev, cur) = eval(sim, 0, step, scale, d, verified);
                            let ctx = format!("step {step} scale {scale} ratio {ratio} jit {jit} p {p} dups {:?} probes {:?}", ev.dups, ev.probes);
                            assert_eq!(ev.class, Class::Consistent, "{ctx}");
                            assert_eq!((ev.prev, ev.next), (Some(cur - 1), Some(cur + 1)), "{ctx}");
                            assert!(ev.probes.len() <= 6, "{ctx}");
                            assert_eq!(SourceState::default().after(ev.class), SourceState { verdict: Verdict::Verified, strikes: 0 });
                            cases += 1;
                        }
                    }
                }
            }
        }
        assert!(cases > 100_000, "{cases}");
    }

    #[test]
    fn exact_quarter_phase_patterns() {
        let d = 100.0;
        // Aligned (φ = 0): prev 1 probe, next 2 -> 3 probes, (0, 1).
        let (ev, _) = eval(Sim::open(d, 0.0), 0, 1001, 24000, d, true);
        assert_eq!((ev.class, ev.dups, ev.probes.len()), (Class::Consistent, (0, 1), 3));
        // φ = 0.25 on a host that floors: -25 lands exactly on f's start (f, dup), -75 on f-1;
        // +75 exactly on f+1's start: (1, 0), 3 probes.
        let (ev, _) = eval(Sim::open(d, 0.25), 0, 1001, 24000, d, false);
        assert_eq!((ev.class, ev.dups, ev.probes.len()), (Class::Consistent, (1, 0), 3));
        // φ = 0.75: -25, -75 (exactly f's start) dups, -125 f-1; +75 f+1: (2, 0), 4 probes.
        let (ev, _) = eval(Sim::open(d, 0.75), 0, 1001, 24000, d, false);
        assert_eq!((ev.class, ev.dups, ev.probes.len()), (Class::Consistent, (2, 0), 4));
        // The same φ = 0.25 when the host's frame mapping jitters by one unit at the boundaries:
        // earlier-than-asked on both sides gives (0, 0), confirmed by the +13 probe (still f);
        // later-than-asked on both sides gives (1, 1). Both consistent, neighbours f±1.
        let sim = Sim::open(d, 0.25);
        for (shift_prev, shift_next, want, n) in [(-1, 1, (0, 0), 3), (1, -1, (1, 1), 4)] {
            let probe = |t: i32| sim.at(t as i64 + if t < 0 { shift_prev } else if t > 0 { shift_next } else { 0 }, 1);
            let ev = evaluate(0, d, false, probe, |&f| f == 0, |a, b| a == b);
            assert_eq!((ev.class, ev.dups, ev.prev, ev.next), (Class::Consistent, want, Some(-1), Some(1)), "{:?}", ev.probes);
            assert_eq!(ev.probes.len(), n);
        }
        let probe = |t: i32| sim.at(t as i64 + if t < 0 { -1 } else if t > 0 { 1 } else { 0 }, 1);
        let ev = evaluate(0, d, false, probe, |&f| f == 0, |a, b| a == b);
        assert_eq!(ev.probes.last().map(|p| (p.role, p.time, p.outcome)), Some((Role::Confirm, 13, Outcome::Duplicate)));
    }

    /// Plays `frames` renders of a clip (render i at native time `time(i)`) through evaluate and
    /// the state machine, returning the states after each render and the classes.
    fn play(sim: Sim, step: i32, scale: u32, d_orig: f64, frames: usize, time: impl Fn(usize) -> i32) -> Vec<(Class, SourceState, usize)> {
        let mut st = SourceState::default();
        let mut out = Vec::new();
        for i in 0..frames {
            if st.verdict == Verdict::Rejected {
                // Legacy path: no ladder at all.
                out.push((Class::Inconclusive, st, 0));
                continue;
            }
            let (ev, cur) = eval(sim, time(i), step, scale, d_orig, st.verdict == Verdict::Verified);
            if matches!(ev.class, Class::Consistent) {
                assert_eq!((ev.prev, ev.next), (Some(cur - 1), Some(cur + 1)));
            }
            st = st.after(ev.class);
            out.push((ev.class, st, ev.probes.len()));
        }
        out
    }

    #[test]
    fn real_world_phase_sequences_verify_and_never_strike() {
        // 119.88 fps at 25% on 23.976 (local step 250.25 media units per render, φ += 1.25),
        // 29.97 fps on 23.976 (D 800.8, φ += 1.25), 120.0 fps on 23.976 (D 200, φ drifts),
        // 59.94 at 40%, 119.88 at 80%, 59.94 at 200%.
        let cases: [(f64, f64); 6] = [(200.2, 250.25), (800.8, 1001.0), (200.0, 1001.0), (400.4, 400.4), (200.2, 800.8), (400.4, 2002.0)];
        for (d, adv) in cases {
            for start in [0.0, 0.1, 0.25, 0.5, 0.75] {
                for jit in [false, true] {
                    let sim = Sim { jit, ..Sim::open(d, start) };
                    let r = play(sim, 1001, 24000, d, 200, |i| 1_000_000 + (i as f64 * adv).round() as i32);
                    let ctx = format!("d {d} adv {adv} start {start} jit {jit}");
                    assert_eq!(r[0].1, SourceState { verdict: Verdict::Verified, strikes: 0 }, "{ctx}");
                    assert!(r.iter().all(|&(c, s, n)| c == Class::Consistent && s.strikes == 0 && n <= 6), "{ctx} {:?}", r.iter().find(|x| x.0 != Class::Consistent));
                }
            }
        }
    }

    #[test]
    fn longer_true_frames_never_verify() {
        // Interpret Footage: D_orig = 200.2 (119.88) but the true frame is k times longer.
        for k in [2.0, 3.0, 4.0, 5.0] {
            for p in 0..80 {
                let start = p as f64 / 80.0;
                // Playback at 100% (1001 units per render) and at 25%.
                for adv in [1001.0, 250.25] {
                    let sim = Sim::open(200.2 * k, start);
                    let r = play(sim, 1001, 24000, 200.2, 12, |i| 1_000_000 + (i as f64 * adv).round() as i32);
                    let ctx = format!("k {k} start {start} adv {adv} {:?}", r);
                    for &(c, s, _) in &r {
                        assert_ne!(s.verdict, Verdict::Verified, "{ctx}");
                        assert!(matches!(c, Class::Anomaly | Class::UnverifiedNoNeighbours | Class::Inconclusive), "{ctx}");
                    }
                    let anomalies = r.iter().filter(|x| x.0 == Class::Anomaly).count();
                    if k == 2.0 {
                        // k = 2: one side always reaches the adjacent true frame, so every render
                        // is an anomaly: Rejected after exactly 3 renders, no ladder afterwards.
                        assert_eq!(r[2].1.verdict, Verdict::Rejected, "{ctx}");
                        assert!(r[1].1.verdict == Verdict::Unknown && r[3..].iter().all(|x| x.2 == 0), "{ctx}");
                    } else if anomalies >= 3 {
                        assert_eq!(r.last().unwrap().1.verdict, Verdict::Rejected, "{ctx}");
                    }
                }
            }
        }
        // k >= 3: a phase in the middle of the true frame reaches neither neighbour with rungs of
        // at most 1.25 D_orig. On an Unknown key that is the legacy path (which then finds the
        // real neighbours), state unchanged; only a Verified key keeps "no neighbours".
        let (ev, _) = eval(Sim::open(1001.0, 0.5), 1_000_000, 1001, 24000, 200.2, false);
        assert_eq!((ev.class, ev.probes.len()), (Class::UnverifiedNoNeighbours, 5));
        assert!(!ev.class.uses_source());
        assert_eq!(SourceState::default().after(ev.class), SourceState::default());
        let (ev, _) = eval(Sim::open(1001.0, 0.5), 1_000_000, 1001, 24000, 200.2, true);
        assert_eq!(ev.class, Class::NoNeighbours);
        // Cold clip start (host repeats frame 0 before it) with a 5x frame: the next side finds
        // frame 1 only late in frame 0, and the edge test stays inside frame 1: anomaly.
        let sim = Sim { first: 0, beyond: Beyond::Repeat, ..Sim::open(1001.0, 0.0) };
        let (ev, cur) = eval(sim, 900, 1001, 24000, 200.2, false);
        assert_eq!(cur, 0);
        assert_eq!(ev.class, Class::Anomaly, "{:?}", ev.probes);
        assert_eq!(ev.probes.last().map(|p| (p.role, p.outcome)), Some((Role::Edge, Outcome::Duplicate)));
    }

    #[test]
    fn continuous_input_is_rejected_at_once() {
        // Every time returns a distinct frame (nested sequence, frame blending).
        let ev = evaluate(10_000, 200.2, false, |t| Ok(t as i64), |&f| f == 10_000, |a, b| a == b);
        assert_eq!((ev.class, ev.dups), (Class::Continuous, (0, 0)));
        assert_eq!(ev.probes.iter().map(|p| p.role).collect::<Vec<_>>(), vec![Role::Prev, Role::Next, Role::Confirm]);
        assert_eq!((ev.prev, ev.next), (None, None));
        assert_eq!(SourceState::default().after(ev.class).verdict, Verdict::Rejected);
        // On a Verified key it is a strike.
        let v = SourceState { verdict: Verdict::Verified, strikes: 0 };
        assert_eq!(v.after(Class::Continuous), SourceState { verdict: Verdict::Verified, strikes: 1 });
    }

    #[test]
    fn frozen_content_has_no_neighbours_and_keeps_state() {
        // Unknown key: legacy path for the render (which finds nothing either), state unchanged.
        let ev = evaluate(10_000, 200.2, false, |_| Ok(7i64), |&f| f == 7, |a, b| a == b);
        assert_eq!((ev.class, ev.probes.len()), (Class::UnverifiedNoNeighbours, 5));
        assert_eq!(SourceState::default().after(ev.class), SourceState::default());
        // Verified key: no neighbours on the source path, state unchanged.
        let ev = evaluate(10_000, 200.2, true, |_| Ok(7i64), |&f| f == 7, |a, b| a == b);
        assert!(ev.class == Class::NoNeighbours && ev.class.uses_source());
        // Then motion resumes: verifies normally.
        let (ev, _) = eval(Sim::open(200.2, 0.4), 1_000_000, 1001, 24000, 200.2, false);
        assert_eq!(SourceState::default().after(ev.class).verdict, Verdict::Verified);
    }

    #[test]
    fn clip_edges_with_cold_and_verified_cache() {
        let d = 200.2;
        for beyond in [Beyond::Missing, Beyond::Repeat] {
            for p in 0..40 {
                let phase = p as f64 / 40.0;
                // Clip start: the current frame is the first frame (index 540).
                let sim = Sim { first: 540, beyond, ..Sim::open(d, phase) };
                let now = ((540.0 - phase) * d).ceil() as i32 + 1;
                let (ev, cur) = eval(sim, now, 1001, 24000, d, false);
                assert_eq!(cur, 540);
                assert_eq!((ev.class, ev.prev, ev.next), (Class::EdgeConfirmed, None, Some(541)), "{beyond:?} start {phase} {:?}", ev.probes);
                assert_eq!(ev.probes.last().unwrap().role, Role::Edge);
                let (ev2, _) = eval(sim, now, 1001, 24000, d, true);
                assert_eq!((ev2.class, ev2.next), (Class::Edge, Some(541)));
                assert!(ev2.probes.iter().all(|p| p.role != Role::Edge));
                // Clip end: the current frame is the last frame.
                let sim = Sim { last: 540, beyond, ..Sim::open(d, phase) };
                let (ev, _) = eval(sim, now, 1001, 24000, d, false);
                assert_eq!((ev.class, ev.prev, ev.next), (Class::EdgeConfirmed, Some(539), None), "{beyond:?} end {phase} {:?}", ev.probes);
                let (ev2, _) = eval(sim, now, 1001, 24000, d, true);
                assert_eq!((ev2.class, ev2.prev), (Class::Edge, Some(539)));
            }
        }
        // A one-frame-beyond-the-neighbour clip edge (2-frame clip, host returns nothing beyond):
        // the edge test is inconclusive -> legacy for this render, state unchanged.
        let sim = Sim { first: 540, last: 541, ..Sim::open(d, 0.5) };
        let (ev, _) = eval(sim, (540.0 * d) as i32 + 1, 1001, 24000, d, false);
        assert_eq!(ev.class, Class::Inconclusive);
        assert_eq!(SourceState::default().after(ev.class), SourceState::default());
    }

    #[test]
    fn state_machine() {
        let u = SourceState::default();
        let v = SourceState { verdict: Verdict::Verified, strikes: 0 };
        // Three consecutive anomalies reject a Verified key; one anomaly then a consistent frame
        // keeps it Verified with no strikes.
        let s = v.after(Class::Anomaly).after(Class::Anomaly);
        assert_eq!(s, SourceState { verdict: Verdict::Verified, strikes: 2 });
        assert_eq!(s.after(Class::Anomaly).verdict, Verdict::Rejected);
        assert_eq!(v.after(Class::Anomaly).after(Class::Consistent), v);
        // Unknown: anomalies strike too, consistent / confirmed edge verify.
        assert_eq!(u.after(Class::Anomaly).after(Class::Anomaly).after(Class::Anomaly).verdict, Verdict::Rejected);
        assert_eq!(u.after(Class::EdgeConfirmed), v);
        // Neutral classes change nothing (strikes are kept, not cleared).
        let s1 = v.after(Class::Anomaly);
        for c in [Class::Edge, Class::NoNeighbours, Class::UnverifiedNoNeighbours, Class::Inconclusive] {
            assert_eq!(s1.after(c), s1);
            assert_eq!(u.after(c), u);
        }
        // Rejected is permanent.
        let r = SourceState { verdict: Verdict::Rejected, strikes: 0 };
        for c in [Class::Consistent, Class::EdgeConfirmed, Class::Anomaly, Class::Continuous] {
            assert_eq!(r.after(c), r);
        }
        // Legacy vs source per class.
        assert!(Class::NoNeighbours.uses_source() && Class::Edge.uses_source());
        assert!(!Class::Anomaly.uses_source() && !Class::Continuous.uses_source() && !Class::Inconclusive.uses_source());
        assert!(!Class::UnverifiedNoNeighbours.uses_source());
    }

    #[test]
    fn ladder_reports_sides() {
        let d = 100.0;
        let idx = |t: i32| (t as f64 / d).floor() as i64;
        // Phase 0.5 in a 100-unit frame: prev -25 is a dup, -75 is used; next +75 is used.
        let lad = ladder(1050, d, |t| Ok(idx(t)), |&f| f == 10);
        assert_eq!((lad.prev.frame, lad.next.frame, lad.prev.dups, lad.next.dups), (Some(9), Some(11), 1, 0));
        assert_eq!((lad.prev.time, lad.next.time), (Some(975), Some(1125)));
        let seen: Vec<(Role, i64, Outcome)> = lad.probes.iter().map(|p| (p.role, p.time, p.outcome)).collect();
        assert_eq!(seen, vec![(Role::Prev, 1025, Outcome::Duplicate), (Role::Prev, 975, Outcome::Used), (Role::Next, 1125, Outcome::Used)]);
        // Nothing before frame 0 vs. a repeated frame 0.
        let lad = ladder(10, d, |t| if t < 0 { Err(Outcome::Missing) } else { Ok(idx(t)) }, |&f| f == 0);
        assert_eq!((lad.prev.kind, lad.next.kind, lad.next.frame), (SideKind::Nothing, SideKind::Found, Some(1)));
        let lad = ladder(10, d, |t| Ok(idx(t).max(0)), |&f| f == 0);
        assert_eq!((lad.prev.kind, lad.prev.dups), (SideKind::AllDuplicate, 3));
        // Unusable probes are skipped, not fatal.
        let lad = ladder(1050, d, |t| if t == 1025 { Err(Outcome::Unusable) } else { Ok(idx(t)) }, |&f| f == 10);
        assert_eq!((lad.prev.frame, lad.next.frame), (Some(9), Some(11)));
        // Time overflow -> unusable, no panic.
        let lad = ladder(i32::MAX - 10, d, |t| Ok(idx(t)), |_| false);
        assert!(lad.probes.iter().any(|p| p.outcome == Outcome::Unusable));
        assert_eq!(lad.next.frame, None);
    }

    #[test]
    fn rate_selection() {
        let tps = TICKS_PER_SECOND;
        // 119.88 fps media on a 23.976 sequence (step 1001 / scale 24000): D = 200.2, a = 5.
        let t = ClipTiming { media_ticks: Some(tps * 1001 / 120_000), seq_ticks: Some(tps * 1001 / 24_000), speed: Some(1.0) };
        let r = source_rate(&t, 1001, 24000);
        let (d, a) = r.choice.unwrap();
        assert!((d - 200.2).abs() < 1e-9 && (a - 5.0).abs() < 1e-9);
        assert_eq!(r.d2, Some(d));
        // Slowed to 20%: a = 1 -> frame distance 1.
        let r = source_rate(&ClipTiming { speed: Some(-0.2), ..t }, 1001, 24000);
        assert!((r.choice.unwrap().1 - 1.0).abs() < 1e-9);
        assert_eq!(r.speed, 0.2);
        // No sequence rate: falls back to D1.
        let r = source_rate(&ClipTiming { seq_ticks: None, ..t }, 1001, 24000);
        assert_eq!(r.d2, None);
        assert!((r.choice.unwrap().0 - 200.2).abs() < 1e-9);
        // Implausible sequence ticks are ignored as well.
        let r = source_rate(&ClipTiming { seq_ticks: Some(0), ..t }, 1001, 24000);
        assert_eq!(r.d2, None);
        // Rejections.
        assert!(source_rate(&ClipTiming { media_ticks: None, ..t }, 1001, 24000).choice.is_err());
        assert!(source_rate(&ClipTiming { media_ticks: Some(-5), ..t }, 1001, 24000).choice.is_err());
        assert!(source_rate(&t, 0, 24000).choice.is_err());
        // 2000 fps media on a 1 fps-ish step: > 64 source frames per step.
        let fast = ClipTiming { media_ticks: Some(tps / 2000), seq_ticks: Some(tps), speed: None };
        assert!(source_rate(&fast, 24000, 24000).choice.is_err());
        // Bad speeds count as 1.
        for s in [None, Some(0.0), Some(f64::NAN), Some(f64::INFINITY)] {
            assert_eq!(norm_speed(s), 1.0);
        }
    }

    #[test]
    fn frame_distance_for_source_neighbours() {
        assert_eq!(source_frame_distance(5.0), 0.2);
        assert_eq!(source_frame_distance(1.0), 1.0);
        assert_eq!(source_frame_distance(0.5), 1.0);
        assert_eq!(source_frame_distance(2.5), 0.4);
        assert_eq!(source_frame_distance(1000.0), 1.0 / 64.0);
    }

    #[test]
    fn time_representation() {
        assert_eq!(choose_time_rep(100, 1001, 24000, 200.2).unwrap().m, 1);
        assert_eq!(choose_time_rep(100, 1001, 24000, 8.0).unwrap().m, 1);
        let r = choose_time_rep(7, 1, 24, 0.2).unwrap();
        assert_eq!((r.m, r.now, r.step, r.scale), (64, 448, 64, 1536));
        assert!((r.d - 12.8).abs() < 1e-9);
        assert_eq!(choose_time_rep(7, 1, 24, 8.0 / 256.0).unwrap().m, 256);
        assert_eq!(choose_time_rep(7, 1, 24, 8.0 / 257.0), None); // would need m > 256
        assert_eq!(choose_time_rep(i32::MAX / 2, 1, 24, 0.2), None); // overflow
        assert_eq!(choose_time_rep(1, 1, u32::MAX / 2, 0.2), None);
        assert_eq!(choose_time_rep(1, 1, 24, f64::NAN), None);
        assert_eq!(choose_time_rep(1, 1, 24, 0.0), None);
    }

    #[test]
    fn gate() {
        // HFR media on a 23.976 sequence passes; the sequence's own rate (adjustment layer,
        // nested sequence, same-rate footage) and anything slower do not.
        assert_eq!(source_gate(Some(200.2), 1001, None), Ok(200.2));
        assert_eq!(source_gate(Some(200.2), -1001, Some(false)), Ok(200.2));
        assert_eq!(source_gate(Some(1001.0), 1001, Some(false)), Err("source rate not above sequence rate"));
        assert_eq!(source_gate(Some(24000.0 / 25.0), 1001, None), Err("source rate not above sequence rate")); // 25 on 23.976: 0.959
        assert_eq!(source_gate(Some(900.0), 1001, None), Ok(900.0)); // 0.899
        assert_eq!(source_gate(Some(901.0), 1001, None), Err("source rate not above sequence rate"));
        assert_eq!(source_gate(Some(5005.0), 1001, None), Err("source rate not above sequence rate"));
        assert_eq!(source_gate(Some(200.2), 1001, Some(true)), Err("synthetic track item"));
        assert_eq!(source_gate(None, 1001, None), Err("no original clip frame rate"));
        assert_eq!(source_gate(Some(f64::NAN), 1001, None), Err("no original clip frame rate"));
        assert_eq!(source_gate(Some(200.2), 0, None), Err("zero time step"));
        assert_eq!(source_gate(Some(10.0), 1001, None), Err("source frames per step out of range"));
        // Ticks -> units.
        let tps = TICKS_PER_SECOND;
        assert!((ticks_to_units(Some(tps * 1001 / 120_000), 24000).unwrap() - 200.2).abs() < 1e-9);
        assert!((ticks_to_units(Some(2_118_916_800), 24000).unwrap() - 200.2).abs() < 1e-9); // as logged in host
        assert_eq!(ticks_to_units(Some(10_594_584_000), 24000), Some(1001.0)); // adjustment layer
        assert_eq!(ticks_to_units(None, 24000), None);
        assert_eq!(ticks_to_units(Some(0), 24000), None);
        assert_eq!(ticks_to_units(Some(tps / 24), 0), None);
    }

    #[test]
    fn legacy_frame_distance_rules() {
        // Same-rate 23.976 at 200%: D_orig = step, a = 2 -> H / 2.
        assert_eq!(legacy_fd_rule(1001.0, 1001, 2.0), LegacyFd::Divided);
        assert_eq!(legacy_fd(LegacyFd::Divided, 2.0, Some(1.0)), Some(0.5));
        // 23.976 on 59.94 (D_orig 2502.5 of a 1001 step): held -> 1 / max(a, 1).
        assert_eq!(legacy_fd_rule(2502.5, 1001, 0.4), LegacyFd::SourceRelative);
        assert_eq!(legacy_fd(LegacyFd::SourceRelative, 0.4, Some(3.0)), Some(1.0));
        assert_eq!(legacy_fd(LegacyFd::SourceRelative, 1.6, Some(2.0)), Some(0.625));
        // Adjustment layer (D_orig = step, a = 1) and slight speed-ups: unchanged.
        assert_eq!(legacy_fd_rule(1001.0, 1001, 1.0), LegacyFd::H);
        assert_eq!(legacy_fd_rule(1001.0, 1001, 1.04), LegacyFd::H);
        assert_eq!(legacy_fd_rule(1100.0, 1001, 1.0), LegacyFd::H); // 1.099 step: not held
        assert_eq!(legacy_fd_rule(1001.0, 1001, f64::NAN), LegacyFd::H);
        assert_eq!(legacy_fd(LegacyFd::Divided, 2.0, None), None);
    }

    #[test]
    fn advance_per_sequence_frame() {
        // 119.88 media on 23.976 at 100%: a = 5 -> fd 0.2.
        let a = source_advance(1001, 1001, 200.2, Some(1.0));
        assert!((a.a - 5.0).abs() < 1e-9 && a.a_local == Some(a.a) && (a.a_speed - 5.0).abs() < 1e-9);
        assert!((source_frame_distance(a.a) - 0.2).abs() < 1e-6);
        // Same at 20%: local_time_step 200 -> a ~ 1 -> fd 1 (speed form agrees).
        let a = source_advance(200, 1001, 200.2, Some(0.2));
        assert!((a.a - 0.999).abs() < 1e-3 && (a.a_speed - 1.0).abs() < 1e-9);
        assert_eq!(source_frame_distance(a.a), 1.0);
        // 119.88 media on a 119.88 sequence at 50%: a = 0.5 -> fd 1.
        let a = source_advance(500, 1001, 1001.0, Some(0.5));
        assert!((a.a - 500.0 / 1001.0).abs() < 1e-12);
        assert_eq!(source_frame_distance(a.a), 1.0);
        // local_time_step wins over a wrong speed; reversed clips use magnitudes.
        let a = source_advance(-1001, 1001, 200.2, Some(0.2));
        assert!((a.a - 5.0).abs() < 1e-9 && (a.a_speed - 1.0).abs() < 1e-9);
        // No local step: the speed form.
        let a = source_advance(0, 1001, 200.2, Some(-0.2));
        assert_eq!(a.a_local, None);
        assert!((a.a - 1.0).abs() < 1e-9);
    }
}
