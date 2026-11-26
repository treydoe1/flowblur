//! macOS Metal buffer synchronization, validation, readback, writeback, and row copy for
//! Premiere Pro's GPU path.
//!
//! Premiere hands GPU filters `id<MTLDevice>`, `id<MTLCommandQueue>`, and per-frame `id<MTLBuffer>`
//! handles. This module:
//! * Retains the host's `MTLDevice` and `MTLCommandQueue` for the lifetime of one filter instance
//!   (no process-global GPU state).
//! * Runs every Metal / Objective-C operation (including staging buffer teardown) inside an
//!   `autoreleasepool` so pool-less host worker threads never leak Objective-C references.
//! * Verifies that raw buffer pointers actually conform to the `MTLBuffer` protocol before
//!   treating them as Metal buffers.
//! * Bounds every individual GPU wait by `METAL_PER_WAIT_TIMEOUT` (5 s) and cumulative GPU wait
//!   time across one `Render` call by `METAL_RENDER_WAIT_BUDGET` (30 s) via `addCompletedHandler`
//!   + `Condvar::wait_timeout` (never blocks indefinitely), ending the render on the first actual
//!   timeout and reporting whether a timed-out command buffer still targets an output buffer so
//!   the caller never disposes in-flight GPU memory.
//! * Handles all Metal storage modes transparently:
//!   - `MTLStorageModeShared`: zero-copy CPU read/write after FIFO queue sync.
//!   - `MTLStorageModeManaged`: `synchronizeResource` before CPU read, `didModifyRange` after CPU write.
//!   - `MTLStorageModePrivate`: blits to/from a temporary `StorageModeShared` staging buffer inside
//!     the autorelease pool and waits up to the render deadline.

use std::cell::Cell;
use std::ffi::{CStr, c_void};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use block2::RcBlock;
use tblur_core as core;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyObject, AnyProtocol, ProtocolObject};
use objc2::{msg_send, sel};
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLDevice, MTLResource, MTLResourceOptions, MTLStorageMode,
};

use crate::pixels::{self, Alpha, ImageDesc, Layout};

/// Maximum time we wait for any single Metal synchronization / blit command buffer.
/// Prevents a hung GPU queue from deadlocking Premiere's render thread.
pub const METAL_PER_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

/// Maximum cumulative GPU wait time allowed across all Metal waits within one `Render` call.
/// Allows a busy but healthy GPU queue (e.g. ~10 private-storage waits in a single render) to
/// complete without false timeouts while still bounding worst-case latency.
pub const METAL_RENDER_WAIT_BUDGET: Duration = Duration::from_secs(30);

/// Structured error from Metal buffer operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalError {
    /// A GPU wait hit the render deadline while a command buffer was still in flight.
    /// When `targets_output` is true, the in-flight command buffer writes into the caller's
    /// destination buffer, which MUST NOT be disposed while queued.
    Timeout {
        op: &'static str,
        targets_output: bool,
    },
    /// The host pointer did not conform to `MTLBuffer`.
    NotBuffer,
    /// Invalid frame dimensions, row bytes, memoryless mode, or backing `MTLBuffer` too small.
    BadGeometry(String),
    /// Any other Metal / pixel codec error.
    Other(String),
}

impl MetalError {
    pub const fn is_timeout(&self) -> bool {
        matches!(self, MetalError::Timeout { .. })
    }

    pub const fn targets_output(&self) -> bool {
        matches!(self, MetalError::Timeout { targets_output: true, .. })
    }

    pub fn summary(&self) -> String {
        match self {
            MetalError::Timeout { op, targets_output } => {
                if *targets_output {
                    format!("metal_timeout({op},in_flight_out)")
                } else {
                    format!("metal_timeout({op})")
                }
            }
            MetalError::NotBuffer => "not_metal_buffer".to_string(),
            MetalError::BadGeometry(s) => s.clone(),
            MetalError::Other(s) => s.clone(),
        }
    }
}

/// Per-instance retained Metal handles. Immutable after creation, so concurrent Premiere GPU
/// callbacks sharing `&MetalContext` are data-race-free.
pub struct MetalContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    per_wait_timeout: Duration,
    render_wait_budget: Duration,
}

// SAFETY: Apple's Metal framework documents `MTLDevice` and `MTLCommandQueue` as thread-safe for
// concurrent immutable use across threads (allocating buffers and creating command buffers from a
// shared `MTLCommandQueue` is thread-safe).
unsafe impl Send for MetalContext {}
unsafe impl Sync for MetalContext {}

/// Borrowed descriptor of one GPU frame's `MTLBuffer` and pixel geometry.
#[derive(Clone, Copy, Debug)]
pub struct MetalFrameDesc {
    pub ptr: *const c_void,
    pub width: usize,
    pub height: usize,
    pub row_bytes: i32,
    pub layout: Layout,
}

impl MetalFrameDesc {
    /// Validates geometry and returns `(ImageDesc, required_byte_len)`.
    pub fn validated_desc(&self) -> Result<(ImageDesc, usize), MetalError> {
        if self.ptr.is_null() {
            return Err(MetalError::BadGeometry("null_metal_buffer".to_string()));
        }
        if self.width == 0 || self.height == 0 {
            return Err(MetalError::BadGeometry("empty_dimensions".to_string()));
        }
        if self.row_bytes <= 0 {
            return Err(MetalError::BadGeometry(format!(
                "non_positive_row_bytes:{}",
                self.row_bytes
            )));
        }
        let bpp = self.layout.bytes_per_pixel();
        let min_row = self
            .width
            .checked_mul(bpp)
            .ok_or_else(|| MetalError::BadGeometry("row_bytes_overflow".to_string()))?;
        let stride = self.row_bytes as usize;
        if stride < min_row {
            return Err(MetalError::BadGeometry(format!(
                "row_bytes_too_small:{stride}<{min_row}"
            )));
        }
        let needed = (self.height - 1)
            .checked_mul(stride)
            .and_then(|n| n.checked_add(min_row))
            .ok_or_else(|| MetalError::BadGeometry("buffer_size_overflow".to_string()))?;
        let desc = ImageDesc {
            width: self.width,
            height: self.height,
            rowbytes: self.row_bytes as isize,
            layout: self.layout,
            alpha: Alpha::Straight,
        };
        Ok((desc, needed))
    }
}

/// Human-readable name of a Metal storage mode for diagnostics logs.
pub fn storage_mode_name(mode: MTLStorageMode) -> &'static str {
    if mode == MTLStorageMode::Shared {
        "shared"
    } else if mode == MTLStorageMode::Managed {
        "managed"
    } else if mode == MTLStorageMode::Private {
        "private"
    } else if mode == MTLStorageMode::Memoryless {
        "memoryless"
    } else {
        "other"
    }
}

/// Returns the Objective-C runtime class name of a raw pointer, or `"null"` / `"unknown"`.
///
/// # Safety
/// If `ptr` is non-null, it must point to a live Objective-C object (`id`).
pub unsafe fn objc_class_name(ptr: *const c_void) -> String {
    if ptr.is_null() {
        return "null".to_string();
    }
    autoreleasepool(|_| {
        // SAFETY: Caller guarantees `ptr` points to a live Objective-C object; `object_getClassName`
        // returns a static NUL-terminated C string owned by the Objective-C runtime.
        let cname = unsafe { objc2::ffi::object_getClassName(ptr as *mut AnyObject) };
        if cname.is_null() {
            return "unknown".to_string();
        }
        // SAFETY: `object_getClassName` returns a valid NUL-terminated C string.
        unsafe { CStr::from_ptr(cname).to_string_lossy().into_owned() }
    })
}

/// Returns `true` iff `ptr` is a non-null Objective-C object conforming to the `MTLBuffer`
/// protocol.
///
/// # Safety
/// If `ptr` is non-null, it must point to a live Objective-C object (`id`).
pub unsafe fn is_metal_buffer(ptr: *const c_void) -> bool {
    if ptr.is_null() {
        return false;
    }
    autoreleasepool(|_| {
        // SAFETY: Caller guarantees `ptr` is a live Objective-C object (`id`).
        let obj: &AnyObject = unsafe { &*(ptr as *const AnyObject) };
        if let Some(proto) = AnyProtocol::get(c"MTLBuffer") {
            // SAFETY: Standard `-[NSObject conformsToProtocol:]` check against `MTLBuffer`.
            let conforms: bool = unsafe { msg_send![obj, conformsToProtocol: proto] };
            return conforms;
        }
        // Defensive fallback if `MTLBuffer` protocol metadata were not registered yet.
        // SAFETY: Standard `-[NSObject respondsToSelector:]` queries on a live ObjC object.
        unsafe {
            let has_len: bool = msg_send![obj, respondsToSelector: sel!(length)];
            let has_mode: bool = msg_send![obj, respondsToSelector: sel!(storageMode)];
            let has_contents: bool = msg_send![obj, respondsToSelector: sel!(contents)];
            has_len && has_mode && has_contents
        }
    })
}

/// Returns the `MTLStorageMode` of a raw `id<MTLBuffer>`, or `None` if null or not an `MTLBuffer`.
///
/// # Safety
/// If `ptr` is non-null, it must point to a live Objective-C object (`id`).
pub unsafe fn buffer_storage_mode(ptr: *const c_void) -> Option<MTLStorageMode> {
    if unsafe { !is_metal_buffer(ptr) } {
        return None;
    }
    autoreleasepool(|_| {
        // SAFETY: `is_metal_buffer(ptr)` confirmed `ptr` is a live `id<MTLBuffer>`.
        let buf: &ProtocolObject<dyn MTLBuffer> =
            unsafe { &*(ptr as *const ProtocolObject<dyn MTLBuffer>) };
        Some(buf.storageMode())
    })
}

impl MetalContext {
    /// Retains the host-provided `MTLDevice` (or derives it from `queue`) and `MTLCommandQueue`.
    ///
    /// # Safety
    /// `queue_ptr` must be a valid live `id<MTLCommandQueue>`. If `device_ptr` is non-null, it
    /// must be a valid live `id<MTLDevice>`.
    pub unsafe fn from_raw_handles(
        device_ptr: *mut c_void,
        queue_ptr: *mut c_void,
    ) -> Result<Self, String> {
        if queue_ptr.is_null() {
            return Err("null_command_queue".to_string());
        }
        autoreleasepool(|_| {
            // SAFETY: `queue_ptr` is a valid host `id<MTLCommandQueue>`. `Retained::retain`
            // increments its reference count so dropping `MetalContext` balances it without
            // over-releasing Premiere's queue.
            let queue: Retained<ProtocolObject<dyn MTLCommandQueue>> = unsafe {
                Retained::retain(queue_ptr as *mut ProtocolObject<dyn MTLCommandQueue>)
            }
            .ok_or_else(|| "retain_queue_failed".to_string())?;

            let device: Retained<ProtocolObject<dyn MTLDevice>> = if !device_ptr.is_null() {
                // SAFETY: `device_ptr` is a valid host `id<MTLDevice>`, retained cleanly.
                unsafe { Retained::retain(device_ptr as *mut ProtocolObject<dyn MTLDevice>) }
                    .ok_or_else(|| "retain_device_failed".to_string())?
            } else {
                queue.device()
            };

            Ok(Self {
                device,
                queue,
                per_wait_timeout: METAL_PER_WAIT_TIMEOUT,
                render_wait_budget: METAL_RENDER_WAIT_BUDGET,
            })
        })
    }

    pub const fn per_wait_timeout(&self) -> Duration {
        self.per_wait_timeout
    }

    pub const fn render_wait_budget(&self) -> Duration {
        self.render_wait_budget
    }

    #[cfg(test)]
    pub fn set_wait_limits(&mut self, per_wait: Duration, render_budget: Duration) {
        self.per_wait_timeout = per_wait;
        self.render_wait_budget = render_budget;
    }

    /// Validates that `frame` describes a non-null, non-memoryless `MTLBuffer` whose backing
    /// length covers `(width, height, row_bytes, layout)`. Does NOT access pixel contents or
    /// enqueue GPU work.
    ///
    /// # Safety
    /// If `frame.ptr` is non-null, it must point to a live Objective-C object (`id`).
    pub unsafe fn validate_buffer(
        frame: &MetalFrameDesc,
    ) -> Result<(ImageDesc, usize, MTLStorageMode), MetalError> {
        let (desc, needed) = frame.validated_desc()?;
        autoreleasepool(|_| {
            // SAFETY: Caller guarantees `frame.ptr` is a live ObjC object; check `MTLBuffer`.
            if unsafe { !is_metal_buffer(frame.ptr) } {
                return Err(MetalError::NotBuffer);
            }
            // SAFETY: `is_metal_buffer` confirmed `frame.ptr` conforms to `MTLBuffer`.
            let buf: &ProtocolObject<dyn MTLBuffer> =
                unsafe { &*(frame.ptr as *const ProtocolObject<dyn MTLBuffer>) };
            let actual = buf.length();
            if actual < needed {
                return Err(MetalError::BadGeometry(format!(
                    "buffer_too_small:{actual}<{needed}"
                )));
            }
            let mode = buf.storageMode();
            if mode == MTLStorageMode::Memoryless {
                return Err(MetalError::BadGeometry("memoryless_buffer".to_string()));
            }
            Ok((desc, needed, mode))
        })
    }

    /// Commits `cmd_buf` and waits up to `min(self.per_wait_timeout, remaining_budget)` via a
    /// completion handler + condvar, deducting elapsed GPU wait time from `wait_budget` so:
    /// * Any single wait is bounded by `METAL_PER_WAIT_TIMEOUT` (5 s).
    /// * Cumulative GPU wait across one `Render` call never exceeds `METAL_RENDER_WAIT_BUDGET` (30 s).
    /// * CPU optical flow time does not eat into the GPU wait budget.
    /// * The first wait that actually times out zeroes `wait_budget` and ends the render.
    fn commit_and_wait(
        &self,
        cmd_buf: &ProtocolObject<dyn MTLCommandBuffer>,
        wait_budget: &Cell<Duration>,
        op: &'static str,
        targets_output: bool,
    ) -> Result<(), MetalError> {
        let budget = wait_budget.get();
        if budget.is_zero() {
            return Err(MetalError::Timeout { op, targets_output: false });
        }
        let wait_limit = self.per_wait_timeout.min(budget);
        let state = Arc::new((Mutex::new(false), Condvar::new()));
        let state_cb = Arc::clone(&state);
        let block = RcBlock::new(move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| {
            let (lock, cvar) = &*state_cb;
            let mut done = lock.lock().unwrap_or_else(|e| e.into_inner());
            *done = true;
            cvar.notify_all();
        });
        // SAFETY: `RcBlock::as_ptr(&block)` yields a valid pointer to a reference-counted ObjC
        // block matching `MTLCommandBufferHandler`; Metal retains/copies it before returning, and
        // the closure owns its own `Arc` clone so even a timeout cannot cause a use-after-free.
        unsafe {
            cmd_buf.addCompletedHandler(RcBlock::as_ptr(&block));
        }
        cmd_buf.commit();

        // Pause core's active-render count while blocked on GPU completion so sibling CPU threads
        // keep full parallelism.
        let _pause = core::par::pause_render();
        let wait_start = Instant::now();
        let (lock, cvar) = &*state;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let (done, _) = cvar
            .wait_timeout_while(guard, wait_limit, |done| !*done)
            .unwrap_or_else(|e| e.into_inner());
        wait_budget.set(budget.saturating_sub(wait_start.elapsed()));
        if !*done {
            wait_budget.set(Duration::ZERO);
            return Err(MetalError::Timeout { op, targets_output });
        }
        if cmd_buf.status() == MTLCommandBufferStatus::Error {
            let msg = cmd_buf
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            return Err(MetalError::Other(format!("metal_exec_err:{msg}")));
        }
        Ok(())
    }

    /// Drains all previously committed GPU commands on Premiere's FIFO command queue.
    pub fn sync_queue(&self, wait_budget: &Cell<Duration>) -> Result<(), MetalError> {
        autoreleasepool(|_| {
            let cmd_buf = self
                .queue
                .commandBuffer()
                .ok_or_else(|| MetalError::Other("command_buffer_null".to_string()))?;
            self.commit_and_wait(&cmd_buf, wait_budget, "sync_queue", false)
        })
    }

    /// Ensures upstream GPU work for `frame` has completed, makes its bytes CPU-readable
    /// (zero-copy for `Shared`, `synchronizeResource` for `Managed`, blit-to-shared staging for
    /// `Private`), and invokes `f(bytes, &desc)`. Runs entirely inside an `autoreleasepool` so
    /// pool-less host threads never leak Metal temporaries.
    ///
    /// # Safety
    /// If `frame.ptr` is non-null, it must point to a live Objective-C object (`id`); if it is an
    /// `MTLBuffer`, it must belong to the same Metal device as this context.
    pub unsafe fn with_readable_bytes<R>(
        &self,
        frame: &MetalFrameDesc,
        queue_synced: &Cell<bool>,
        wait_budget: &Cell<Duration>,
        f: impl FnOnce(&[u8], &ImageDesc) -> R,
    ) -> Result<(R, &'static str), MetalError> {
        autoreleasepool(|_| {
            // SAFETY: Caller guarantees `frame.ptr` is a live ObjC object; validates `MTLBuffer`,
            // non-Memoryless storage mode, and `length >= needed`.
            let (desc, needed, mode) = unsafe { Self::validate_buffer(frame) }?;
            // SAFETY: `validate_buffer` confirmed `frame.ptr` is a live `id<MTLBuffer>`.
            let buf: &ProtocolObject<dyn MTLBuffer> =
                unsafe { &*(frame.ptr as *const ProtocolObject<dyn MTLBuffer>) };
            let mode_name = storage_mode_name(mode);

            if mode == MTLStorageMode::Shared {
                if !queue_synced.get() {
                    self.sync_queue(wait_budget)?;
                    queue_synced.set(true);
                }
                let ptr = buf.contents().as_ptr() as *const u8;
                // SAFETY: `buf` is `StorageModeShared`, upstream GPU work on `self.queue` has
                // completed (`sync_queue`), and `needed <= buf.length()`.
                let bytes = unsafe { std::slice::from_raw_parts(ptr, needed) };
                Ok((f(bytes, &desc), mode_name))
            } else if mode == MTLStorageMode::Managed {
                let cmd_buf = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| MetalError::Other("command_buffer_null".to_string()))?;
                let blit = cmd_buf
                    .blitCommandEncoder()
                    .ok_or_else(|| MetalError::Other("blit_encoder_null".to_string()))?;
                blit.synchronizeResource(ProtocolObject::from_ref(buf));
                blit.endEncoding();
                self.commit_and_wait(&cmd_buf, wait_budget, "read_managed_sync", false)?;
                queue_synced.set(true);
                let ptr = buf.contents().as_ptr() as *const u8;
                // SAFETY: `buf` is `StorageModeManaged`, `synchronizeResource` completed on GPU,
                // and `needed <= buf.length()`.
                let bytes = unsafe { std::slice::from_raw_parts(ptr, needed) };
                Ok((f(bytes, &desc), mode_name))
            } else {
                // `MTLStorageMode::Private`: staging buffer is allocated, read, and dropped
                // strictly inside this `autoreleasepool` so `dealloc` never leaks into a missing
                // thread pool.
                let staging = self
                    .device
                    .newBufferWithLength_options(needed, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| MetalError::Other("alloc_staging_read_failed".to_string()))?;
                let cmd_buf = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| MetalError::Other("command_buffer_null".to_string()))?;
                let blit = cmd_buf
                    .blitCommandEncoder()
                    .ok_or_else(|| MetalError::Other("blit_encoder_null".to_string()))?;
                // SAFETY: `needed <= buf.length()` and `staging.length() >= needed`; offsets are 0.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        buf, 0, &staging, 0, needed,
                    );
                }
                blit.endEncoding();
                self.commit_and_wait(&cmd_buf, wait_budget, "read_private_blit", false)?;
                queue_synced.set(true);
                let ptr = staging.contents().as_ptr() as *const u8;
                // SAFETY: `staging` is a live `StorageModeShared` buffer of at least `needed`
                // bytes and the blit command buffer has completed.
                let bytes = unsafe { std::slice::from_raw_parts(ptr, needed) };
                Ok((f(bytes, &desc), mode_name))
            }
        })
    }

    /// Writes premultiplied f32 RGBA pixels into the destination Metal buffer, handling `Shared`,
    /// `Managed`, and `Private` storage modes. Runs entirely inside an `autoreleasepool`.
    /// On a timeout during a `Private` blit, returns `MetalError::Timeout { targets_output: true }`.
    ///
    /// # Safety
    /// If `frame.ptr` is non-null, it must point to a live Objective-C object (`id`); if it is an
    /// `MTLBuffer`, it must belong to this context's device and not alias any concurrently read
    /// buffer.
    pub unsafe fn write_rgba_to_buffer(
        &self,
        rgba: &[f32],
        frame: &MetalFrameDesc,
        queue_synced: &Cell<bool>,
        wait_budget: &Cell<Duration>,
    ) -> Result<&'static str, MetalError> {
        autoreleasepool(|_| {
            // SAFETY: Caller guarantees `frame.ptr` is a live ObjC object; validates `MTLBuffer`,
            // non-Memoryless mode, and `length >= needed`.
            let (desc, needed, mode) = unsafe { Self::validate_buffer(frame) }?;
            // SAFETY: `validate_buffer` confirmed `frame.ptr` is a live `id<MTLBuffer>`.
            let buf: &ProtocolObject<dyn MTLBuffer> =
                unsafe { &*(frame.ptr as *const ProtocolObject<dyn MTLBuffer>) };
            let mode_name = storage_mode_name(mode);

            if mode == MTLStorageMode::Shared {
                if !queue_synced.get() {
                    self.sync_queue(wait_budget)?;
                    queue_synced.set(true);
                }
                let ptr = buf.contents().as_ptr() as *mut u8;
                // SAFETY: `buf` is `StorageModeShared`, prior queue work completed, caller
                // guarantees exclusive write access, and `needed <= buf.length()`.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, needed) };
                pixels::write_rgba(rgba, dst, &desc)
                    .map_err(|e| MetalError::Other(format!("write_rgba_err:{e:?}")))?;
                Ok(mode_name)
            } else if mode == MTLStorageMode::Managed {
                if !queue_synced.get() {
                    self.sync_queue(wait_budget)?;
                    queue_synced.set(true);
                }
                let ptr = buf.contents().as_ptr() as *mut u8;
                // SAFETY: `buf` is `StorageModeManaged`, prior queue work completed, caller
                // guarantees exclusive write access, and `needed <= buf.length()`.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, needed) };
                pixels::write_rgba(rgba, dst, &desc)
                    .map_err(|e| MetalError::Other(format!("write_rgba_err:{e:?}")))?;
                buf.didModifyRange(NSRange::new(0, needed));
                Ok(mode_name)
            } else {
                // `MTLStorageMode::Private`: write into a `StorageModeShared` staging buffer and
                // blit into `buf`. Metal retains both `staging` and `buf` for the command buffer's
                // lifetime, and `targets_output: true` tells the caller not to dispose the output
                // PPix if the wait times out.
                let staging = self
                    .device
                    .newBufferWithLength_options(needed, MTLResourceOptions::StorageModeShared)
                    .ok_or_else(|| MetalError::Other("alloc_staging_write_failed".to_string()))?;
                let ptr = staging.contents().as_ptr() as *mut u8;
                // SAFETY: `staging` is freshly allocated `StorageModeShared` of `needed` bytes and
                // not yet visible to the GPU.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr, needed) };
                pixels::write_rgba(rgba, dst, &desc)
                    .map_err(|e| MetalError::Other(format!("write_rgba_err:{e:?}")))?;

                let cmd_buf = self
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| MetalError::Other("command_buffer_null".to_string()))?;
                let blit = cmd_buf
                    .blitCommandEncoder()
                    .ok_or_else(|| MetalError::Other("blit_encoder_null".to_string()))?;
                // SAFETY: `staging.length() >= needed` and `buf.length() >= needed`; offsets are 0.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &staging, 0, buf, 0, needed,
                    );
                }
                blit.endEncoding();
                self.commit_and_wait(&cmd_buf, wait_budget, "write_private_blit", true)?;
                queue_synced.set(true);
                Ok(mode_name)
            }
        })
    }

    /// Copies every active pixel row from `src` to `dst` on the GPU command queue, supporting
    /// differing `row_bytes` (while leaving destination row padding untouched) and any non-memoryless
    /// storage modes (`Shared`, `Managed`, `Private`).
    ///
    /// # Safety
    /// If non-null, `src.ptr` and `dst.ptr` must point to live Objective-C objects (`id`) belonging
    /// to this context's Metal device.
    pub unsafe fn copy_frame_rows(
        &self,
        src: &MetalFrameDesc,
        dst: &MetalFrameDesc,
        wait_budget: &Cell<Duration>,
    ) -> Result<(), MetalError> {
        autoreleasepool(|_| {
            // SAFETY: Caller guarantees both pointers are live ObjC objects; validates `MTLBuffer`,
            // non-Memoryless mode, and `length >= needed`.
            let (s_desc, s_needed, _) = unsafe { Self::validate_buffer(src) }?;
            let (d_desc, _, _) = unsafe { Self::validate_buffer(dst) }?;

            if s_desc.layout != d_desc.layout
                || s_desc.width != d_desc.width
                || s_desc.height != d_desc.height
            {
                return Err(MetalError::BadGeometry("copy_frame_mismatch".to_string()));
            }
            if src.ptr == dst.ptr {
                return Ok(());
            }

            // SAFETY: `validate_buffer` confirmed both pointers conform to `MTLBuffer`.
            let src_buf: &ProtocolObject<dyn MTLBuffer> =
                unsafe { &*(src.ptr as *const ProtocolObject<dyn MTLBuffer>) };
            let dst_buf: &ProtocolObject<dyn MTLBuffer> =
                unsafe { &*(dst.ptr as *const ProtocolObject<dyn MTLBuffer>) };

            let cmd_buf = self
                .queue
                .commandBuffer()
                .ok_or_else(|| MetalError::Other("command_buffer_null".to_string()))?;
            let blit = cmd_buf
                .blitCommandEncoder()
                .ok_or_else(|| MetalError::Other("blit_encoder_null".to_string()))?;

            if src.row_bytes == dst.row_bytes {
                // SAFETY: `s_needed <= src_buf.length()` and `s_needed == d_needed <= dst_buf.length()`.
                unsafe {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        src_buf, 0, dst_buf, 0, s_needed,
                    );
                }
            } else {
                let row_active = s_desc.width * s_desc.layout.bytes_per_pixel();
                let s_stride = src.row_bytes as usize;
                let d_stride = dst.row_bytes as usize;
                for y in 0..s_desc.height {
                    let s_off = y * s_stride;
                    let d_off = y * d_stride;
                    // SAFETY: `validate_buffer` guaranteed `(height - 1) * stride + row_active <= length`
                    // on both buffers, so every row slice `[off..off + row_active]` is in bounds.
                    unsafe {
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            src_buf, s_off, dst_buf, d_off, row_active,
                        );
                    }
                }
            }
            blit.endEncoding();
            self.commit_and_wait(&cmd_buf, wait_budget, "copy_frame_rows", true)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_foundation::NSString;
    use objc2_metal::MTLCreateSystemDefaultDevice;

    fn make_shared_buf(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let ptr = NonNull::new(bytes.as_ptr() as *mut c_void).expect("non-empty");
        // SAFETY: `bytes` is a valid readable slice of `bytes.len()` bytes.
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

    fn make_private_buf_from_bytes(
        device: &ProtocolObject<dyn MTLDevice>,
        queue: &ProtocolObject<dyn MTLCommandQueue>,
        bytes: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        autoreleasepool(|_| {
            let staging = make_shared_buf(device, bytes);
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

    #[test]
    fn metal_read_and_write_shared_and_private_buffers_with_padding() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device required on macOS");
        let queue = device.newCommandQueue().expect("command queue");
        let ctx = unsafe {
            MetalContext::from_raw_handles(
                Retained::as_ptr(&device) as *mut c_void,
                Retained::as_ptr(&queue) as *mut c_void,
            )
        }
        .expect("MetalContext");
        let budget = Cell::new(METAL_RENDER_WAIT_BUDGET);

        // 3x2 BGRA32f with padded row_bytes = 64 (3*16 = 48 active + 16 pad).
        let (w, h, rb) = (3usize, 2usize, 64i32);
        let mut raw = vec![0x77u8; h * rb as usize];
        for y in 0..h {
            for x in 0..w {
                let off = y * rb as usize + x * 16;
                let px: [f32; 4] = [
                    (x + 1) as f32 * 0.1,
                    (y + 1) as f32 * 0.25,
                    0.5 + x as f32 * 0.1,
                    1.0,
                ];
                for (c, v) in px.iter().enumerate() {
                    raw[off + c * 4..off + c * 4 + 4].copy_from_slice(&v.to_ne_bytes());
                }
            }
        }

        // 1. Shared buffer round-trip.
        let shared_in = make_shared_buf(&device, &raw);
        let shared_out = device
            .newBufferWithLength_options(raw.len(), MTLResourceOptions::StorageModeShared)
            .expect("shared out");
        let synced = Cell::new(false);
        let in_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&shared_in) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb,
            layout: Layout::Bgra32f,
        };
        let mut rgba = Vec::new();
        let (res, mode) = unsafe {
            ctx.with_readable_bytes(&in_desc, &synced, &budget, |bytes, desc| {
                pixels::read_rgba_and_luma(bytes, desc, &mut rgba)
            })
        }
        .unwrap();
        assert_eq!(mode, "shared");
        assert!(synced.get());
        res.unwrap();

        let out_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&shared_out) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb,
            layout: Layout::Bgra32f,
        };
        let out_mode =
            unsafe { ctx.write_rgba_to_buffer(&rgba, &out_desc, &synced, &budget) }.unwrap();
        assert_eq!(out_mode, "shared");

        // 2. Private buffer round-trip (blit-to-shared read + shared-to-private write).
        let priv_in = make_private_buf_from_bytes(&device, &queue, &raw);
        let priv_out = device
            .newBufferWithLength_options(raw.len(), MTLResourceOptions::StorageModePrivate)
            .expect("private out");
        let synced2 = Cell::new(false);
        let priv_in_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&priv_in) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb,
            layout: Layout::Bgra32f,
        };
        let mut rgba_priv = Vec::new();
        let (res_p, mode_p) = unsafe {
            ctx.with_readable_bytes(&priv_in_desc, &synced2, &budget, |bytes, desc| {
                pixels::read_rgba_and_luma(bytes, desc, &mut rgba_priv)
            })
        }
        .unwrap();
        assert_eq!(mode_p, "private");
        res_p.unwrap();
        assert_eq!(rgba_priv, rgba);

        let priv_out_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&priv_out) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb,
            layout: Layout::Bgra32f,
        };
        let out_mode_p =
            unsafe { ctx.write_rgba_to_buffer(&rgba_priv, &priv_out_desc, &synced2, &budget) }
                .unwrap();
        assert_eq!(out_mode_p, "private");

        let (back_bytes, _) = unsafe {
            ctx.with_readable_bytes(&priv_out_desc, &synced2, &budget, |bytes, _| {
                bytes.to_vec()
            })
        }
        .unwrap();
        for y in 0..h {
            let off = y * rb as usize;
            assert_eq!(&back_bytes[off..off + w * 16], &raw[off..off + w * 16]);
        }
    }

    #[test]
    fn metal_shared_read_waits_for_async_gpu_write_on_queue() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device required on macOS");
        let queue = device.newCommandQueue().expect("command queue");
        let ctx = unsafe {
            MetalContext::from_raw_handles(
                Retained::as_ptr(&device) as *mut c_void,
                Retained::as_ptr(&queue) as *mut c_void,
            )
        }
        .expect("MetalContext");

        let (w, h) = (256usize, 256usize);
        let rb = (w * 16) as i32;
        let len = h * rb as usize;
        let src_bytes: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) & 0xff) as u8).collect();
        let src_buf = make_shared_buf(&device, &src_bytes);
        let dst_buf = device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .expect("dst shared");

        let cmd_buf = queue.commandBuffer().expect("cmd_buf");
        let blit = cmd_buf.blitCommandEncoder().expect("blit");
        unsafe {
            blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                &src_buf, 0, &dst_buf, 0, len,
            );
        }
        blit.endEncoding();
        cmd_buf.commit();

        let synced = Cell::new(false);
        let desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&dst_buf) as *const c_void,
            width: w,
            height: h,
            row_bytes: rb,
            layout: Layout::Bgra32f,
        };
        let budget = Cell::new(METAL_RENDER_WAIT_BUDGET);
        let (got, mode) =
            unsafe { ctx.with_readable_bytes(&desc, &synced, &budget, |b, _| b.to_vec()) }
                .unwrap();
        assert_eq!(mode, "shared");
        assert!(synced.get());
        assert_eq!(got, src_bytes);
    }

    #[test]
    fn is_metal_buffer_rejects_non_buffer_objc_objects() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device required on macOS");
        let queue = device.newCommandQueue().expect("command queue");
        let buf = device
            .newBufferWithLength_options(64, MTLResourceOptions::StorageModeShared)
            .expect("MTLBuffer");
        let ns_str = NSString::from_str("not a buffer");

        unsafe {
            assert!(is_metal_buffer(Retained::as_ptr(&buf) as *const c_void));
            assert!(!is_metal_buffer(Retained::as_ptr(&device) as *const c_void));
            assert!(!is_metal_buffer(Retained::as_ptr(&queue) as *const c_void));
            assert!(!is_metal_buffer(Retained::as_ptr(&ns_str) as *const c_void));
            assert!(!is_metal_buffer(std::ptr::null()));
        }
    }

    #[test]
    fn copy_frame_rows_across_differing_row_bytes_and_storage_modes() {
        let device = MTLCreateSystemDefaultDevice().expect("Metal device required on macOS");
        let queue = device.newCommandQueue().expect("command queue");
        let ctx = unsafe {
            MetalContext::from_raw_handles(
                Retained::as_ptr(&device) as *mut c_void,
                Retained::as_ptr(&queue) as *mut c_void,
            )
        }
        .expect("MetalContext");

        let (w, h) = (17usize, 9usize);
        let src_rb = w * 16 + 16;
        let dst_rb = w * 16 + 80;
        let mut src_raw = vec![0xAAu8; h * src_rb];
        for y in 0..h {
            for x in 0..w * 16 {
                src_raw[y * src_rb + x] = ((y * 73 + x * 19) & 0xff) as u8;
            }
        }
        let src_priv = make_private_buf_from_bytes(&device, &queue, &src_raw);
        let dst_init = vec![0x55u8; h * dst_rb];
        let dst_shared = make_shared_buf(&device, &dst_init);

        let src_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&src_priv) as *const c_void,
            width: w,
            height: h,
            row_bytes: src_rb as i32,
            layout: Layout::Bgra32f,
        };
        let dst_desc = MetalFrameDesc {
            ptr: Retained::as_ptr(&dst_shared) as *const c_void,
            width: w,
            height: h,
            row_bytes: dst_rb as i32,
            layout: Layout::Bgra32f,
        };
        let budget = Cell::new(METAL_RENDER_WAIT_BUDGET);
        unsafe { ctx.copy_frame_rows(&src_desc, &dst_desc, &budget) }.unwrap();

        let synced = Cell::new(false);
        let (got, _) =
            unsafe { ctx.with_readable_bytes(&dst_desc, &synced, &budget, |b, _| b.to_vec()) }
                .unwrap();
        for y in 0..h {
            assert_eq!(
                &got[y * dst_rb..y * dst_rb + w * 16],
                &src_raw[y * src_rb..y * src_rb + w * 16],
                "row {y} mismatch"
            );
            // Padding in destination rows (except beyond required_len on last row) stays untouched.
            if y + 1 < h {
                assert!(
                    got[y * dst_rb + w * 16..(y + 1) * dst_rb]
                        .iter()
                        .all(|&b| b == 0x55),
                    "row {y} padding modified"
                );
            }
        }
    }

    #[test]
    fn private_read_does_not_leak_device_retain_count_without_pool() {
        // Item 3 regression: run 2,000 private-storage reads + writes on a fresh OS thread with
        // NO outer autorelease pool, and confirm `MTLDevice`'s retainCount stays bounded (no
        // ~1/iter leak).
        std::thread::spawn(|| {
            let device = MTLCreateSystemDefaultDevice().expect("Metal device required on macOS");
            let queue = device.newCommandQueue().expect("command queue");
            let ctx = unsafe {
                MetalContext::from_raw_handles(
                    Retained::as_ptr(&device) as *mut c_void,
                    Retained::as_ptr(&queue) as *mut c_void,
                )
            }
            .expect("MetalContext");

            let raw = vec![0u8; 2 * 32];
            let priv_buf = make_private_buf_from_bytes(&device, &queue, &raw);
            let desc = MetalFrameDesc {
                ptr: Retained::as_ptr(&priv_buf) as *const c_void,
                width: 2,
                height: 2,
                row_bytes: 32,
                layout: Layout::Bgra32f,
            };
            let rgba = vec![0.5f32; 16];

            // Warm up once, then measure retainCount before & after 2,000 iterations.
            let warm_b = Cell::new(METAL_RENDER_WAIT_BUDGET);
            unsafe {
                ctx.with_readable_bytes(&desc, &Cell::new(false), &warm_b, |b, _| b.len())
                    .unwrap();
            }
            let rc_of = |d: &ProtocolObject<dyn MTLDevice>| -> usize {
                unsafe { msg_send![d, retainCount] }
            };
            let r0 = rc_of(&device);
            for _ in 0..2000 {
                let b = Cell::new(METAL_RENDER_WAIT_BUDGET);
                let s = Cell::new(false);
                unsafe {
                    ctx.with_readable_bytes(&desc, &s, &b, |bytes, _| bytes.len())
                        .unwrap();
                    ctx.write_rgba_to_buffer(&rgba, &desc, &s, &b).unwrap();
                }
            }
            // Allow any async Metal completion handler teardown to finish.
            ctx.sync_queue(&Cell::new(METAL_RENDER_WAIT_BUDGET)).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            let r1 = rc_of(&device);
            let delta = r1.saturating_sub(r0);
            assert!(
                delta <= 16,
                "MTLDevice retainCount leaked on pool-less thread: {r0} -> {r1} (+{delta} over 2000 iters)"
            );
        })
        .join()
        .unwrap();
    }
}
