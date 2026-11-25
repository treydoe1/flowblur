//! Process-global per-clip record (Premiere only), keyed by effect instance and timing context.
//!
//! Each record holds two independent facts:
//! * `source`: the verdict on the host's original clip frame rate `D_orig`
//!   (`neighbors::SourceState`: Unknown / Verified / Rejected plus a strike counter), advanced
//!   by each render's ladder classification (`apply`). Rejected is permanent for the key.
//! * `input_matches_checkout`: whether a checkout at the current time returned the effect's own
//!   input (`Some(true)`: the current-time checkout is skipped from then on; `Some(false)`: it
//!   differed once, so the checkout is always made). Recorded for every Premiere clip with a known
//!   instance id, including clips on the legacy path (adjustment layers, same-rate footage).
//!
//! Premiere renders several frames concurrently on different threads: the map is behind a mutex
//! held only for the map operation itself (never across a host call). State transitions are
//! applied to the stored state under the lock (not to the render's earlier snapshot), so
//! concurrent renders never lose a strike or undo a rejection; a mismatch beats a match. The
//! map is cleared when it would exceed `MAX_ENTRIES`, which bounds memory without eviction
//! bookkeeping (a cleared clip starts again from Unknown).

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

use crate::neighbors::{Class, SourceState};

/// Everything we can get reliably that identifies "this media in this timing context".
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    /// `Utility::filter_instance_id`: identifies this effect instance (so this clip on the
    /// timeline); `INSTANCE_UNKNOWN` when that call fails, so the rest of the key still applies.
    pub instance_id: i32,
    pub time_step: i32,
    pub time_scale: u32,
    /// `Utility::original_clip_frame_rate` (so the verdict is about this key's D_orig).
    pub orig_ticks: Option<i64>,
    /// `Utility::media_frame_rate`.
    pub media_ticks: Option<i64>,
    /// Sequence Info suite ticks per sequence frame.
    pub seq_ticks: Option<i64>,
    pub width: usize,
    pub height: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Entry {
    pub source: SourceState,
    pub input_matches_checkout: Option<bool>,
}

pub const MAX_ENTRIES: usize = 1024;

/// `Key::instance_id` when the host did not report one.
pub const INSTANCE_UNKNOWN: i32 = -1;

/// The bounded map itself (kept separate from the global so it can be tested in isolation).
#[derive(Default)]
struct Map(HashMap<Key, Entry>);

impl Map {
    fn get(&self, k: &Key) -> Entry {
        self.0.get(k).copied().unwrap_or_default()
    }
    fn update<R>(&mut self, k: Key, f: impl FnOnce(&mut Entry) -> R) -> R {
        if self.0.len() >= MAX_ENTRIES && !self.0.contains_key(&k) {
            self.0.clear();
        }
        f(self.0.entry(k).or_default())
    }
    fn apply(&mut self, k: Key, class: Class) -> SourceState {
        self.update(k, |e| {
            e.source = e.source.after(class);
            e.source
        })
    }
    /// A mismatch is final: once the checkout differed from the input it is always made.
    fn record_input_match(&mut self, k: Key, matched: bool) {
        self.update(k, |e| {
            if e.input_matches_checkout != Some(false) {
                e.input_matches_checkout = Some(matched);
            }
        })
    }
}

static CACHE: LazyLock<Mutex<Map>> = LazyLock::new(|| Mutex::new(Map::default()));

fn with<R>(f: impl FnOnce(&mut Map) -> R) -> R {
    // A poisoned lock only means another thread panicked mid-operation on a plain map; the map
    // itself is still valid.
    let mut g = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    f(&mut g)
}

/// The record for `k` (Unknown / no match information when there is none).
pub fn get(k: &Key) -> Entry {
    with(|m| m.get(k))
}

/// Advances the stored source state of `k` by one render's classification; returns the new state.
pub fn apply(k: Key, class: Class) -> SourceState {
    with(|m| m.apply(k, class))
}

pub fn record_input_match(k: Key, matched: bool) {
    with(|m| m.record_input_match(k, matched))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbors::Verdict;

    fn key(instance_id: i32) -> Key {
        Key { instance_id, time_step: 1001, time_scale: 24000, orig_ticks: None, media_ticks: None, seq_ticks: None, width: 8, height: 4 }
    }

    #[test]
    fn merge_rules_and_bound() {
        let mut m = Map::default();
        assert_eq!(m.get(&key(1)), Entry::default());
        assert_eq!(m.apply(key(1), Class::Consistent).verdict, Verdict::Verified);
        assert_eq!(m.get(&Key { time_scale: 30000, ..key(1) }), Entry::default());
        // The two facts are independent.
        m.record_input_match(key(1), true);
        assert_eq!(m.get(&key(1)), Entry { source: SourceState { verdict: Verdict::Verified, strikes: 0 }, input_matches_checkout: Some(true) });
        // Strikes accumulate on the stored state; three reject for good.
        m.apply(key(1), Class::Anomaly);
        m.apply(key(1), Class::Anomaly);
        assert_eq!(m.apply(key(1), Class::Anomaly).verdict, Verdict::Rejected);
        assert_eq!(m.apply(key(1), Class::Consistent).verdict, Verdict::Rejected);
        // A mismatch replaces a match and is final.
        m.record_input_match(key(1), false);
        m.record_input_match(key(1), true);
        assert_eq!(m.get(&key(1)).input_matches_checkout, Some(false));
        // Input-match records exist without any source verdict (legacy-path clips).
        m.record_input_match(key(2), true);
        assert_eq!(m.get(&key(2)), Entry { source: SourceState::default(), input_matches_checkout: Some(true) });

        let mut m = Map::default();
        for i in 0..MAX_ENTRIES as i32 {
            m.apply(key(100 + i), Class::Consistent);
        }
        assert_eq!(m.0.len(), MAX_ENTRIES);
        // Updating an existing key does not clear; a new key past the bound does.
        m.record_input_match(key(100), true);
        assert_eq!(m.0.len(), MAX_ENTRIES);
        m.apply(key(5), Class::Continuous);
        assert_eq!(m.0.len(), 1);
        assert_eq!(m.get(&key(5)).source.verdict, Verdict::Rejected);
    }

    #[test]
    fn global_round_trip() {
        // An id far from the small counter values the effect tests use.
        let k = key(0x7EAD_0001);
        apply(k, Class::Consistent);
        record_input_match(k, false);
        assert_eq!(get(&k), Entry { source: SourceState { verdict: Verdict::Verified, strikes: 0 }, input_matches_checkout: Some(false) });
    }

    #[test]
    fn instances_keep_separate_entries() {
        // Two effect instances with otherwise identical keys never share a verdict.
        let mut m = Map::default();
        m.apply(key(11), Class::Consistent);
        assert_eq!(m.get(&key(12)).source, SourceState::default());
        m.apply(key(12), Class::Continuous);
        assert_eq!((m.get(&key(11)).source.verdict, m.get(&key(12)).source.verdict), (Verdict::Verified, Verdict::Rejected));
        // An unknown instance is just another key value.
        assert_eq!(m.get(&key(INSTANCE_UNKNOWN)), Entry::default());
    }
}
