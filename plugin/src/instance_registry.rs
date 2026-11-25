//! Process-wide, bounded, mutex-guarded map from Premiere effect instance ID (`filter_instance_id`)
//! to the PF side's `is_track_item_effect_applied_to_synthetic` authority flag.
//!
//! The PF side records `(instance_id, synthetic)` as early as Premiere exposes both values
//! (`SequenceSetup`, `SequenceResetup`, `SequenceFlatten`, `FrameSetup`, `UpdateParamsUi`,
//! `UserChangedParam`, and `Render`). The GPU filter's `CreateInstance` reads
//! `EffectNode::RuntimeInstanceID` and accepts only when this map says `Some(true)`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Upper bound on tracked effect instances in the global registry.
pub const MAX_ENTRIES: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    synthetic: bool,
    first_cmd: &'static str,
    seq: u64,
}

/// Outcome of recording one `(instance_id, synthetic)` observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordOutcome {
    /// True when `instance_id` was not previously present in the map.
    pub is_first: bool,
    /// True when `instance_id` was already present and `synthetic` changed value.
    pub changed: bool,
    /// Name of the PF command that first recorded this `instance_id`.
    pub first_cmd: &'static str,
}

struct Inner {
    map: HashMap<i32, Entry>,
    next_seq: u64,
}

/// Bounded, thread-safe map from Premiere effect `instance_id` to `synthetic: bool`.
pub struct SyntheticMap {
    cap: usize,
    inner: Mutex<Inner>,
}

impl SyntheticMap {
    /// Creates a new map holding at most `cap.max(1)` entries (evicting the least-recently-updated
    /// entry when full).
    pub fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            inner: Mutex::new(Inner {
                map: HashMap::new(),
                next_seq: 1,
            }),
        }
    }

    /// Records `instance_id -> synthetic` from PF command `cmd`.
    /// Updates recency on every call; preserves `first_cmd` while the entry stays resident.
    pub fn record(&self, instance_id: i32, synthetic: bool, cmd: &'static str) -> RecordOutcome {
        let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let seq = g.next_seq;
        g.next_seq = g.next_seq.wrapping_add(1).max(1);

        if let Some(existing) = g.map.get_mut(&instance_id) {
            let changed = existing.synthetic != synthetic;
            existing.synthetic = synthetic;
            existing.seq = seq;
            return RecordOutcome {
                is_first: false,
                changed,
                first_cmd: existing.first_cmd,
            };
        }

        if g.map.len() >= self.cap
            && let Some((&oldest_key, _)) = g.map.iter().min_by_key(|(_, v)| v.seq)
        {
            g.map.remove(&oldest_key);
        }

        g.map.insert(
            instance_id,
            Entry {
                synthetic,
                first_cmd: cmd,
                seq,
            },
        );
        RecordOutcome {
            is_first: true,
            changed: false,
            first_cmd: cmd,
        }
    }

    /// Looks up whether `instance_id` is known to be synthetic (`Some(true)`), non-synthetic
    /// (`Some(false)`), or unknown (`None`).
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    pub fn lookup(&self, instance_id: i32) -> Option<bool> {
        let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        g.map.get(&instance_id).map(|e| e.synthetic)
    }

    /// Removes one entry (called on `SequenceSetdown` so entries do not outlive their instances).
    /// Returns `true` if an entry was present and removed.
    pub fn remove(&self, instance_id: i32) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .map
            .remove(&instance_id)
            .is_some()
    }

    /// Current number of resident entries (for tests).
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).map.len()
    }
}

static GLOBAL: OnceLock<SyntheticMap> = OnceLock::new();

/// Process-wide singleton shared by the PF and GPU sides of the plug-in.
pub fn global() -> &'static SyntheticMap {
    GLOBAL.get_or_init(|| SyntheticMap::new(MAX_ENTRIES))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn bounded_eviction_preserves_recent_and_tracks_first_cmd() {
        let m = SyntheticMap::new(3);
        assert_eq!(m.lookup(10), None);

        let r1 = m.record(10, true, "sequence_setup");
        assert_eq!(
            r1,
            RecordOutcome {
                is_first: true,
                changed: false,
                first_cmd: "sequence_setup",
            }
        );
        assert_eq!(m.lookup(10), Some(true));

        // Same value from a later command keeps `first_cmd`, not first, not changed.
        let r2 = m.record(10, true, "render");
        assert_eq!(
            r2,
            RecordOutcome {
                is_first: false,
                changed: false,
                first_cmd: "sequence_setup",
            }
        );

        // Fill capacity (keys 10, 20, 30).
        m.record(20, false, "sequence_resetup");
        m.record(30, true, "frame_setup");
        assert_eq!(m.len(), 3);

        // Touch 10 so 20 becomes the oldest entry.
        let r3 = m.record(10, false, "render");
        assert_eq!(
            r3,
            RecordOutcome {
                is_first: false,
                changed: true,
                first_cmd: "sequence_setup",
            }
        );
        assert_eq!(m.lookup(10), Some(false));

        // Inserting 40 must evict 20 (oldest), keeping len bounded at 3.
        m.record(40, true, "update_params_ui");
        assert_eq!(m.len(), 3);
        assert_eq!(m.lookup(20), None);
        assert_eq!(m.lookup(10), Some(false));
        assert_eq!(m.lookup(30), Some(true));
        assert_eq!(m.lookup(40), Some(true));
    }

    #[test]
    fn concurrent_writers_and_readers_stay_bounded_and_consistent() {
        let map = Arc::new(SyntheticMap::new(64));
        let mut handles = Vec::new();
        for t in 0..8i32 {
            let m = Arc::clone(&map);
            handles.push(std::thread::spawn(move || {
                for i in 0..2_000i32 {
                    let id = (t * 25 + (i % 40)) % 120;
                    let syn = id % 2 == 0;
                    m.record(id, syn, "render");
                    if let Some(v) = m.lookup(id) {
                        assert_eq!(v, syn, "inconsistent synthetic value for id={id}");
                    }
                    assert!(m.len() <= 64, "map exceeded capacity bound");
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert!(map.len() <= 64);
    }
}
