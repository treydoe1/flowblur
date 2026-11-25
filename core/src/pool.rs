//! Strictly bounded buffer pool for large full-resolution render buffers.
//!
//! Bounds:
//! * Total retained pool memory across the entire process is strictly capped at `MAX_POOL_BYTES`
//!   (256 MiB) and at most `MAX_POOL_SLOTS` (8) buffers per type.
//! * Buffers larger than `MAX_POOL_BYTES / 2` are never retained.
//! * Returning a buffer that would exceed the cap drops it immediately.
//! * Buffers are retained at their initialized length (never `clear()`ed on return), so exact-size
//!   hits require zero allocation, zero `memset`, and zero `unsafe`.

use std::sync::Mutex;

const MAX_POOL_BYTES: usize = 256 * 1024 * 1024;
const MAX_POOL_SLOTS: usize = 8;

struct CombinedPool {
    f32_bufs: Vec<Vec<f32>>,
    vec2_bufs: Vec<Vec<[f32; 2]>>,
    total_bytes: usize,
}

impl CombinedPool {
    const fn new() -> Self {
        CombinedPool {
            f32_bufs: Vec::new(),
            vec2_bufs: Vec::new(),
            total_bytes: 0,
        }
    }

    fn take_from<T: Clone>(bufs: &mut Vec<Vec<T>>, total_bytes: &mut usize, len: usize, fill: T) -> Vec<T> {
        let elem = std::mem::size_of::<T>();
        // Prefer an exact-length hit (zero memset), else smallest-fitting capacity.
        let hit = bufs
            .iter()
            .enumerate()
            .filter(|(_, b)| b.capacity() >= len && b.capacity() <= len.saturating_mul(2).max(len + 65536))
            .min_by_key(|(_, b)| (b.len() != len, b.capacity()))
            .map(|(i, _)| i);
        if let Some(idx) = hit {
            let mut buf = bufs.swap_remove(idx);
            *total_bytes = total_bytes.saturating_sub(buf.capacity() * elem);
            if buf.len() != len {
                buf.resize(len, fill);
            }
            buf
        } else {
            vec![fill; len]
        }
    }

    fn put_into<T>(bufs: &mut Vec<Vec<T>>, total_bytes: &mut usize, buf: Vec<T>) {
        let elem = std::mem::size_of::<T>();
        let bytes = buf.capacity() * elem;
        if bytes == 0 || bytes > MAX_POOL_BYTES / 2 {
            return;
        }
        if bufs.len() < MAX_POOL_SLOTS && *total_bytes + bytes <= MAX_POOL_BYTES {
            *total_bytes += bytes;
            bufs.push(buf);
        }
    }
}

static POOL: Mutex<CombinedPool> = Mutex::new(CombinedPool::new());

pub(crate) struct PooledF32(pub Vec<f32>);

impl PooledF32 {
    /// Returns a buffer of length `len` whose contents may hold previous frame data.
    /// Caller must overwrite every element before reading.
    pub fn take_dirty(len: usize) -> Self {
        let v = {
            let mut p = POOL.lock().unwrap_or_else(|e| e.into_inner());
            let CombinedPool { f32_bufs, total_bytes, .. } = &mut *p;
            CombinedPool::take_from(f32_bufs, total_bytes, len, 0.0f32)
        };
        PooledF32(v)
    }
}

impl Drop for PooledF32 {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.0);
        let mut p = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let CombinedPool { f32_bufs, total_bytes, .. } = &mut *p;
        CombinedPool::put_into(f32_bufs, total_bytes, v);
    }
}

pub(crate) struct PooledVec2(pub Vec<[f32; 2]>);

impl PooledVec2 {
    /// Returns a buffer of length `len` whose contents may hold previous frame data.
    /// Caller must overwrite every element before reading.
    pub fn take_dirty(len: usize) -> Self {
        let v = {
            let mut p = POOL.lock().unwrap_or_else(|e| e.into_inner());
            let CombinedPool { vec2_bufs, total_bytes, .. } = &mut *p;
            CombinedPool::take_from(vec2_bufs, total_bytes, len, [0.0f32; 2])
        };
        PooledVec2(v)
    }
}

impl Drop for PooledVec2 {
    fn drop(&mut self) {
        let v = std::mem::take(&mut self.0);
        let mut p = POOL.lock().unwrap_or_else(|e| e.into_inner());
        let CombinedPool { vec2_bufs, total_bytes, .. } = &mut *p;
        CombinedPool::put_into(vec2_bufs, total_bytes, v);
    }
}
