//! Adaptive, 100% safe work-sharing parallelism built on `rayon::in_place_scope`.
//!
//! Why this replaces raw `par_iter` / nested `rayon::join`:
//! When K host threads (e.g. K=16 in Premiere Pro) enter rayon via `par_iter`/`join`, every
//! foreign thread parks on a condvar while rayon's 30 workers bounce between ~90 fine-grained
//! nested barriers per frame, causing severe LIFO tail-latency inflation (flow p50 4.4x slower,
//! p90 14x slower) and burning ~43% of active CPU in `swtch_pri` spin-yields.
//!
//! With `rayon::in_place_scope` + helper-slot leasing:
//! * Calling (host) threads ALWAYS execute chunks directly and never park waiting for a queued
//!   job to start.
//! * `ACTIVE_RENDERS + ACTIVE_HELPERS` is capped at `max_threads()`, preventing oversubscription
//!   when K host threads render concurrently (e.g., at K=16 on 30 cores, at most 14 helpers are
//!   leased across all renders).
//! * Strictly 1-level (or bounded 2-level when `budget > 4`) parallelism: both branches of `join`
//!   and all spawned workers set `IN_WORKER = true` via an unwind-safe RAII guard, eliminating
//!   nested barrier storms.
//! * Zero `unsafe` code, zero custom OS threads, zero raw stack pointers, and full `catch_unwind`
//!   panic safety (rayon's scope joins helpers and propagates panics cleanly).

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

static ACTIVE_RENDERS: AtomicUsize = AtomicUsize::new(0);
static ACTIVE_HELPERS: AtomicUsize = AtomicUsize::new(0);

thread_local! {
    static IN_WORKER: Cell<bool> = const { Cell::new(false) };
    static RENDER_DEPTH: Cell<usize> = const { Cell::new(0) };
    static PAUSED_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// RAII guard restoring `IN_WORKER` on exit or unwind.
struct InWorkerGuard {
    prev: bool,
}

impl InWorkerGuard {
    #[inline]
    fn enter() -> Self {
        let prev = IN_WORKER.with(|f| f.replace(true));
        InWorkerGuard { prev }
    }
}

impl Drop for InWorkerGuard {
    fn drop(&mut self) {
        IN_WORKER.with(|f| f.set(self.prev));
    }
}

/// RAII guard tracking an in-flight render on the current thread (re-entrant and unwind-safe).
pub struct RenderGuard {
    counted: bool,
}

impl Drop for RenderGuard {
    fn drop(&mut self) {
        if self.counted {
            ACTIVE_RENDERS.fetch_sub(1, Ordering::Relaxed);
        }
        RENDER_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Marks the start of a render call on the current thread for adaptive concurrency budgeting.
pub fn enter_render() -> RenderGuard {
    let counted = RENDER_DEPTH.with(|d| {
        let prev = d.get();
        d.set(prev + 1);
        prev == 0
    });
    if counted {
        ACTIVE_RENDERS.fetch_add(1, Ordering::Relaxed);
    }
    RenderGuard { counted }
}

/// Temporarily suspends this thread's active-render count while blocked inside a host call
/// (e.g. Premiere frame checkout / check-in), restoring it on drop (re-entrant and unwind-safe).
pub struct PauseRenderGuard {
    paused: bool,
}

impl Drop for PauseRenderGuard {
    fn drop(&mut self) {
        if self.paused {
            ACTIVE_RENDERS.fetch_add(1, Ordering::Relaxed);
        }
        PAUSED_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Suspends the current thread's active-render count for the lifetime of the returned guard.
/// Has no effect if the thread has not entered a render or is already paused.
pub fn pause_render() -> PauseRenderGuard {
    let in_render = RENDER_DEPTH.with(|d| d.get() > 0);
    let first_pause = PAUSED_DEPTH.with(|d| {
        let prev = d.get();
        d.set(prev + 1);
        prev == 0
    });
    let paused = in_render && first_pause;
    if paused {
        ACTIVE_RENDERS.fetch_sub(1, Ordering::Relaxed);
    }
    PauseRenderGuard { paused }
}

/// Number of renders currently counted as actively computing.
#[inline]
pub fn active_renders() -> usize {
    ACTIVE_RENDERS.load(Ordering::Relaxed)
}

/// True when the current thread has suspended its active-render count via `pause_render`.
#[inline]
pub fn is_render_paused() -> bool {
    PAUSED_DEPTH.with(|d| d.get() > 0)
}

/// RAII lease on up to `want` rayon helper threads so `ACTIVE_RENDERS + ACTIVE_HELPERS <= max_threads()`.
struct HelperLease {
    leased: usize,
}

impl HelperLease {
    #[inline]
    fn try_acquire(want: usize) -> Self {
        if want == 0 {
            return HelperLease { leased: 0 };
        }
        let mt = max_threads();
        let active = ACTIVE_RENDERS.load(Ordering::Relaxed).max(1);
        let max_helpers_global = mt.saturating_sub(active);
        if max_helpers_global == 0 {
            return HelperLease { leased: 0 };
        }
        let mut cur = ACTIVE_HELPERS.load(Ordering::Relaxed);
        loop {
            let avail = max_helpers_global.saturating_sub(cur);
            let take = want.min(avail);
            if take == 0 {
                return HelperLease { leased: 0 };
            }
            match ACTIVE_HELPERS.compare_exchange_weak(
                cur,
                cur + take,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return HelperLease { leased: take },
                Err(actual) => cur = actual,
            }
        }
    }
}

impl Drop for HelperLease {
    fn drop(&mut self) {
        if self.leased > 0 {
            ACTIVE_HELPERS.fetch_sub(self.leased, Ordering::Release);
        }
    }
}

/// Effective logical thread limit (`FLOWBLUR_THREADS` / `RAYON_NUM_THREADS` / rayon pool size).
pub fn max_threads() -> usize {
    static MAX_T: OnceLock<usize> = OnceLock::new();
    *MAX_T.get_or_init(|| {
        for var in ["FLOWBLUR_THREADS", "RAYON_NUM_THREADS"] {
            if let Ok(s) = std::env::var(var)
                && let Ok(n) = s.trim().parse::<usize>()
                && n >= 1
            {
                return n;
            }
        }
        rayon::current_num_threads().max(1)
    })
}

/// Thread budget for the current render (including the calling thread).
#[inline]
pub fn budget() -> usize {
    if IN_WORKER.with(Cell::get) {
        return 1;
    }
    let mt = max_threads();
    if mt <= 1 {
        return 1;
    }
    let active = ACTIVE_RENDERS.load(Ordering::Relaxed).max(1);
    mt.div_ceil(active)
}

/// Runs `f(0..num_chunks)` across the calling thread and up to `max_helpers` rayon workers.
pub fn run_chunks<F>(num_chunks: usize, max_helpers: usize, f: F)
where
    F: Fn(usize) + Sync,
{
    if num_chunks == 0 {
        return;
    }
    let want = if IN_WORKER.with(Cell::get) {
        0
    } else {
        max_helpers.min(num_chunks.saturating_sub(1))
    };
    let lease = HelperLease::try_acquire(want);
    if lease.leased == 0 {
        for idx in 0..num_chunks {
            f(idx);
        }
        return;
    }

    let next = AtomicUsize::new(0);
    rayon::in_place_scope(|s| {
        for _ in 0..lease.leased {
            s.spawn(|_| {
                let _w = InWorkerGuard::enter();
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= num_chunks {
                        break;
                    }
                    f(idx);
                }
            });
        }
        let _w = InWorkerGuard::enter();
        loop {
            let idx = next.fetch_add(1, Ordering::Relaxed);
            if idx >= num_chunks {
                break;
            }
            f(idx);
        }
    });
}

/// Runs `fa` and `fb` concurrently (calling thread runs `fa`, one rayon worker runs `fb`) when
/// budget and helper capacity allow; otherwise runs sequentially.
/// Both branches run with `IN_WORKER = true` so inner loops do not nest additional barriers.
pub fn join<A, B, FA, FB>(fa: FA, fb: FB) -> (A, B)
where
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B + Send,
    A: Send,
    B: Send,
{
    if budget() <= 1 {
        return (fa(), fb());
    }
    let lease = HelperLease::try_acquire(1);
    if lease.leased == 0 {
        return (fa(), fb());
    }
    let mut out_a: Option<A> = None;
    let mut out_b: Option<B> = None;
    let mut fb_opt = Some(fb);
    rayon::in_place_scope(|s| {
        let out_b_ref = &mut out_b;
        s.spawn(move |_| {
            let _w = InWorkerGuard::enter();
            if let Some(f) = fb_opt.take() {
                *out_b_ref = Some(f());
            }
        });
        let _w = InWorkerGuard::enter();
        out_a = Some(fa());
    });
    (out_a.unwrap(), out_b.unwrap())
}

/// Runs `fa` and `fb` concurrently while STILL allowing both branches to use half the thread
/// budget when budget > 4 (used at K=1/K=4 for coarse top-level prev/next split in `field::analyze`).
pub fn join_split<A, B, FA, FB>(fa: FA, fb: FB) -> (A, B)
where
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B + Send,
    A: Send,
    B: Send,
{
    let b = budget();
    if b <= 1 {
        return (fa(), fb());
    }
    if b <= 4 {
        return join(fa, fb);
    }
    let mut out_a: Option<A> = None;
    let mut out_b: Option<B> = None;
    let mut fb_opt = Some(fb);
    ACTIVE_RENDERS.fetch_add(1, Ordering::Relaxed);
    struct DecGuard;
    impl Drop for DecGuard {
        fn drop(&mut self) {
            ACTIVE_RENDERS.fetch_sub(1, Ordering::Relaxed);
        }
    }
    let _dec = DecGuard;
    rayon::in_place_scope(|s| {
        let out_b_ref = &mut out_b;
        s.spawn(move |_| {
            if let Some(f) = fb_opt.take() {
                *out_b_ref = Some(f());
            }
        });
        out_a = Some(fa());
    });
    (out_a.unwrap(), out_b.unwrap())
}

/// Computes `(rows_per_chunk, want_helpers)` given total rows, minimum rows per chunk, and an
/// upper cap on workers.
#[inline]
fn plan_chunks(total: usize, min_chunk: usize, cap_threads: usize) -> (usize, usize) {
    let b = budget().min(cap_threads);
    if total == 0 || b <= 1 || total <= min_chunk {
        return (total.max(1), 0);
    }
    let max_by_min = (total / min_chunk).max(1);
    let target_chunks = (b * 4).min(max_by_min).min(total);
    if target_chunks <= 1 {
        return (total, 0);
    }
    let chunk_size = total.div_ceil(target_chunks);
    let actual_chunks = total.div_ceil(chunk_size);
    (chunk_size, (b - 1).min(actual_chunks - 1))
}

/// Parallel mutable slice processing by logical rows of length `row_len` — 100% safe Rust.
/// Calls `f(start_row, chunk_slice)` for each disjoint strip of rows.
pub fn for_rows_mut<T, F>(slice: &mut [T], row_len: usize, min_rows: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    for_rows_mut_capped(slice, row_len, min_rows, usize::MAX, f);
}

/// Same as `for_rows_mut`, with an explicit thread cap for small loops.
pub fn for_rows_mut_capped<T, F>(
    slice: &mut [T],
    row_len: usize,
    min_rows: usize,
    cap_threads: usize,
    f: F,
) where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync,
{
    if slice.is_empty() || row_len == 0 {
        return;
    }
    let total_rows = slice.len().div_ceil(row_len);
    let (rows_per_chunk, want) = plan_chunks(total_rows, min_rows, cap_threads);
    let lease = HelperLease::try_acquire(want);
    if lease.leased == 0 {
        f(0, slice);
        return;
    }
    let elem_chunk = rows_per_chunk * row_len;
    let queue: Mutex<VecDeque<(usize, &mut [T])>> = Mutex::new(
        slice
            .chunks_mut(elem_chunk)
            .enumerate()
            .map(|(c, ch)| (c * rows_per_chunk, ch))
            .collect(),
    );
    rayon::in_place_scope(|s| {
        for _ in 0..lease.leased {
            s.spawn(|_| {
                let _w = InWorkerGuard::enter();
                loop {
                    let item = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                    let Some((r0, ch)) = item else { break };
                    f(r0, ch);
                }
            });
        }
        let _w = InWorkerGuard::enter();
        loop {
            let item = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
            let Some((r0, ch)) = item else { break };
            f(r0, ch);
        }
    });
}

/// Parallel mutable processing of two zipped slices with matching row counts — 100% safe Rust.
pub fn for_rows_mut2<T, U, F>(
    a: &mut [T],
    b: &mut [U],
    row_len_a: usize,
    row_len_b: usize,
    min_rows: usize,
    f: F,
) where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Sync,
{
    for_rows_mut2_capped(a, b, row_len_a, row_len_b, min_rows, usize::MAX, f);
}

pub fn for_rows_mut2_capped<T, U, F>(
    a: &mut [T],
    b: &mut [U],
    row_len_a: usize,
    row_len_b: usize,
    min_rows: usize,
    cap_threads: usize,
    f: F,
) where
    T: Send,
    U: Send,
    F: Fn(usize, &mut [T], &mut [U]) + Sync,
{
    if row_len_a == 0 || row_len_b == 0 {
        assert!(
            a.is_empty() && b.is_empty(),
            "for_rows_mut2: zero row_len with non-empty slice (a={}, b={}, row_len_a={row_len_a}, row_len_b={row_len_b})",
            a.len(),
            b.len(),
        );
        return;
    }
    let rows_a = a.len().div_ceil(row_len_a);
    let rows_b = b.len().div_ceil(row_len_b);
    assert_eq!(
        rows_a, rows_b,
        "for_rows_mut2: mismatched row counts ({rows_a} vs {rows_b})"
    );
    if rows_a == 0 {
        return;
    }
    let total_rows = rows_a;
    let (rows_per_chunk, want) = plan_chunks(total_rows, min_rows, cap_threads);
    let lease = HelperLease::try_acquire(want);
    if lease.leased == 0 {
        f(0, a, b);
        return;
    }
    let queue: Mutex<VecDeque<(usize, &mut [T], &mut [U])>> = Mutex::new(
        a.chunks_mut(rows_per_chunk * row_len_a)
            .zip(b.chunks_mut(rows_per_chunk * row_len_b))
            .enumerate()
            .map(|(c, (ca, cb))| (c * rows_per_chunk, ca, cb))
            .collect(),
    );
    rayon::in_place_scope(|s| {
        for _ in 0..lease.leased {
            s.spawn(|_| {
                let _w = InWorkerGuard::enter();
                loop {
                    let item = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
                    let Some((r0, ca, cb)) = item else { break };
                    f(r0, ca, cb);
                }
            });
        }
        let _w = InWorkerGuard::enter();
        loop {
            let item = queue.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
            let Some((r0, ca, cb)) = item else { break };
            f(r0, ca, cb);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "mismatched row counts")]
    fn for_rows_mut2_panics_on_mismatched_rows_seq() {
        let mut a = vec![0u32; 20]; // 2 rows of 10
        let mut b = vec![0u32; 15]; // 3 rows of 5
        for_rows_mut2_capped(&mut a, &mut b, 10, 5, 1, 1, |_, _, _| {});
    }

    #[test]
    #[should_panic(expected = "mismatched row counts")]
    fn for_rows_mut2_panics_on_mismatched_rows_par() {
        let mut a = vec![0u32; 1000]; // 100 rows of 10
        let mut b = vec![0u32; 505]; // 101 rows of 5
        for_rows_mut2(&mut a, &mut b, 10, 5, 1, |_, _, _| {});
    }

    #[test]
    #[should_panic(expected = "mismatched row counts")]
    fn for_rows_mut2_panics_when_one_slice_is_empty() {
        let mut a = vec![0u32; 10];
        let mut b: Vec<u32> = Vec::new();
        for_rows_mut2(&mut a, &mut b, 10, 5, 1, |_, _, _| {});
    }

    #[test]
    fn for_rows_mut2_covers_all_rows_including_partial_last_row() {
        // 50 rows of 7 (350) + partial last row of 3 -> 51 rows; b has 51 rows of 2.
        let mut a = vec![0usize; 50 * 7 + 3];
        let mut b = vec![0usize; 51 * 2];
        for_rows_mut2(&mut a, &mut b, 7, 2, 1, |r0, ca, cb| {
            for (dr, (ra, rb)) in ca.chunks_mut(7).zip(cb.chunks_exact_mut(2)).enumerate() {
                let row = r0 + dr;
                for x in ra {
                    *x = row + 1;
                }
                rb[0] = row + 1;
                rb[1] = row + 1;
            }
        });
        assert!(a.iter().all(|&v| v > 0));
        assert!(b.iter().all(|&v| v > 0));
        assert_eq!(*a.last().unwrap(), 51);
        assert_eq!(*b.last().unwrap(), 51);
    }
}
