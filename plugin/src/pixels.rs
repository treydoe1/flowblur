//! Host pixel buffers <-> the core's f32 premultiplied RGBA, as plain functions over byte slices.
//!
//! Nothing in here touches the Adobe API, so it is fully unit-tested (see the tests below).
//!
//! # What the host hands us, and how sure we are
//!
//! Sources: AE = https://ae-plugins.docsforadobe.dev, PPro = https://ppro-plugins.docsforadobe.dev
//!
//! CONFIRMED from documentation:
//! * AE worlds are ARGB: "sequential words each containing Alpha, Red, Green, Blue from the low byte
//!   to the high byte" (AE: effect-basics/PF_EffectWorld). Same order for 8 and 16 bpc
//!   (`PF_Pixel8` / `PF_Pixel16` are {alpha, red, green, blue}).
//! * 16-bpc white is 32768, not 65535 (PPro: universals/pixel-formats-and-color-spaces: "black at 0
//!   to white at 32768, matching After Effects and Photoshop 16-bit formats").
//! * Rows may be padded: "Don't assume that you can get to the next scanline ... find the beginning
//!   of each row using height and rowbytes" (AE: PF_EffectWorld); "Honor the rowbytes"
//!   (PPro: universals/video-frames). With field rendering Premiere halves height and doubles
//!   rowbytes (AE: ppro/basic-host-differences) - handled because we never assume rowbytes.
//! * Premiere's native formats list "BGRA, ARGB, and VUYA ... in order of increasing memory address
//!   from left to right" (PPro: pixel-formats-and-color-spaces). So BGRA_4444_8u is bytes B,G,R,A
//!   and BGRA_4444_32f is f32 B,G,R,A: the float format has the SAME channel order as the 8-bit one.
//! * Premiere hands AE-API effects 8-bit only unless more formats are registered with the PF Pixel
//!   Format suite at global setup (AE: ppro/bigger-differences), and AE 16-bpc rendering and SmartFX
//!   are unsupported in Premiere (AE: ppro/bigger-differences, ppro/unsupported-features).
//!
//! INFERRED from documentation (not stated outright) - CHECK IN THE HOST:
//! * Premiere BGRA_4444_8u / BGRA_4444_32f are STRAIGHT alpha. The docs never say so, but they list
//!   BGRP_4444_8u separately as "Premultiplied alpha" (PPro: pixel-formats-and-color-spaces), which
//!   implies plain BGRA is not. We premultiply on the way in and un-premultiply on the way out.
//! * After Effects worlds are PREMULTIPLIED (with black). The effect docs never state it; the AEGP
//!   docs offer `AEGP_CopyBits_HQ_Straight` as the special straight-alpha variant of CopyBits_HQ,
//!   and the PF `premultiply` utility exists to convert to/from straight. Treated as passthrough.
//! * Row orientation: PPro native uncompressed frames have a lower-left origin
//!   (PPro: pixel-formats-and-color-spaces). What an AE-API effect sees is the PF_EffectWorld's
//!   `data` + `rowbytes`, where row y starts at data + y*rowbytes (rowbytes may be negative; the
//!   bindings' `Layer::buffer()` returns the slice from the lowest address). We honour the sign, so
//!   row 0 is whatever the host calls row 0. If Premiere's row 0 were in fact the bottom row, every
//!   world (input, output, neighbours) is flipped the same way, and motion blur is symmetric under a
//!   vertical flip, so the output is still correct; only the Motion Vectors view's colour coding of
//!   vertical direction would be mirrored.
//! * 32f values may exceed 1.0 (HDR / super-whites); we pass them through unclamped. Non-finite
//!   input values are replaced by 0 because the core requires finite input.

/// Pixel layouts we understand.
use tblur_core::par;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// After Effects 8 bpc: bytes A,R,G,B; 0..=255.
    Argb8,
    /// After Effects 16 bpc: u16 A,R,G,B (native endian); 0..=32768.
    Argb16,
    /// Premiere BGRA_4444_8u: bytes B,G,R,A; 0..=255.
    Bgra8,
    /// Premiere GPU_BGRA_4444_16f: IEEE 754 binary16 B,G,R,A (native endian); nominal 0..=1, may exceed.
    Bgra16f,
    /// Premiere BGRA_4444_32f / GPU_BGRA_4444_32f: f32 B,G,R,A (native endian); nominal 0..=1, may exceed.
    Bgra32f,
}

impl Layout {
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            Layout::Argb8 | Layout::Bgra8 => 4,
            Layout::Argb16 | Layout::Bgra16f => 8,
            Layout::Bgra32f => 16,
        }
    }
    /// Memory position of (R, G, B, A) within one pixel, in channel units (used by the tests to
    /// build buffers independently of the codecs).
    #[cfg(test)]
    const fn rgba_slots(self) -> [usize; 4] {
        match self {
            Layout::Argb8 | Layout::Argb16 => [1, 2, 3, 0],
            Layout::Bgra8 | Layout::Bgra16f | Layout::Bgra32f => [2, 1, 0, 3],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Alpha {
    Premultiplied,
    Straight,
}

/// Geometry and format of one host buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageDesc {
    pub width: usize,
    pub height: usize,
    /// Signed byte distance from row y to row y+1, as in `PF_EffectWorld::rowbytes`.
    pub rowbytes: isize,
    pub layout: Layout,
    pub alpha: Alpha,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelError {
    Empty,
    RowbytesTooSmall,
    BufferTooSmall,
    SizeMismatch,
}

impl ImageDesc {
    fn stride(&self) -> usize {
        self.rowbytes.unsigned_abs()
    }
    /// Byte offset of row `y` inside a slice that starts at the buffer's lowest address
    /// (which is what `Layer::buffer()` returns for both signs of rowbytes).
    fn row_offset(&self, y: usize) -> usize {
        if self.rowbytes >= 0 { y * self.stride() } else { (self.height - 1 - y) * self.stride() }
    }
    /// Checked minimum slice length needed for this image; returns `None` on arithmetic overflow.
    fn checked_required_len(&self) -> Option<usize> {
        if self.width == 0 || self.height == 0 {
            return Some(0);
        }
        let row_bytes = self.width.checked_mul(self.layout.bytes_per_pixel())?;
        (self.height - 1).checked_mul(self.stride())?.checked_add(row_bytes)
    }
    /// Minimum slice length needed for this image (saturating on overflow).
    pub fn required_len(&self) -> usize {
        self.checked_required_len().unwrap_or(usize::MAX)
    }
    pub fn validate(&self, len: usize) -> Result<(), PixelError> {
        if self.width == 0 || self.height == 0 {
            return Err(PixelError::Empty);
        }
        let row_bytes = self.width.checked_mul(self.layout.bytes_per_pixel()).ok_or(PixelError::RowbytesTooSmall)?;
        if self.stride() < row_bytes {
            return Err(PixelError::RowbytesTooSmall);
        }
        let req = self.checked_required_len().ok_or(PixelError::BufferTooSmall)?;
        if len < req {
            return Err(PixelError::BufferTooSmall);
        }
        Ok(())
    }
}

#[inline]
fn finite_or_zero(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

// ── per-layout pixel codecs (monomorphised so the inner loops carry no format branches) ──

/// One pixel layout: load returns straight-or-premultiplied (as stored) R,G,B,A floats;
/// store takes R,G,B,A floats already converted to the stored alpha convention.
trait Codec: Sync {
    const BPP: usize;
    fn load(px: &[u8]) -> [f32; 4];
    fn store(c: [f32; 4], px: &mut [u8]);
}

struct Argb8;
struct Argb16;
struct Bgra8;
struct Bgra16f;
struct Bgra32f;

/// Converts an IEEE 754 binary16 (half-precision float) bit pattern to `f32`.
/// Handles zeros, subnormals, normals, infinities, and NaNs exactly.
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x03ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // Normalize subnormal: bit 10 is at leading_zeros == 21 in u32.
        let lz = frac.leading_zeros() - 21;
        let new_exp = 127 - 14 - lz;
        let new_frac = (frac << lz) & 0x03ff;
        f32::from_bits(sign | (new_exp << 23) | (new_frac << 13))
    } else if exp == 31 {
        // Inf (frac == 0) or NaN (frac != 0).
        f32::from_bits(sign | 0x7f80_0000 | (frac << 13))
    } else {
        f32::from_bits(sign | ((exp + (127 - 15)) << 23) | (frac << 13))
    }
}

/// Converts an `f32` to IEEE 754 binary16 using round-to-nearest-even.
/// Finite values beyond the binary16 range (`±65504.0`) clamp to `±65504.0` rather than
/// synthesizing infinities in pixel buffers; non-finite inputs map to IEEE 754 Inf/qNaN.
#[inline(always)]
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        return if frac != 0 { sign | 0x7e00 } else { sign | 0x7c00 };
    }
    let mut new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7bff;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let mant = frac | 0x0080_0000;
        let shift = (14 - new_exp) as u32;
        let half = 1u32 << (shift - 1);
        let mask = (1u32 << shift) - 1;
        let rem = mant & mask;
        let mut res = mant >> shift;
        if rem > half || (rem == half && (res & 1) != 0) {
            res += 1;
        }
        return sign | (res as u16);
    }
    let rem = frac & 0x1fff;
    let mut mant = frac >> 13;
    if rem > 0x1000 || (rem == 0x1000 && (mant & 1) != 0) {
        mant += 1;
        if mant == 0x0400 {
            mant = 0;
            new_exp += 1;
            if new_exp >= 31 {
                return sign | 0x7bff;
            }
        }
    }
    sign | ((new_exp as u16) << 10) | (mant as u16)
}

#[inline(always)]
fn f16f(px: &[u8], slot: usize) -> f32 {
    finite_or_zero(f16_to_f32(u16::from_ne_bytes([px[2 * slot], px[2 * slot + 1]])))
}

#[inline(always)]
fn u8f(v: u8) -> f32 {
    v as f32 * (1.0 / 255.0)
}
#[inline(always)]
fn f8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}
#[inline(always)]
fn u16f(px: &[u8], slot: usize) -> f32 {
    u16::from_ne_bytes([px[2 * slot], px[2 * slot + 1]]) as f32 * (1.0 / 32768.0)
}
#[inline(always)]
fn f16(v: f32, px: &mut [u8], slot: usize) {
    let v = (v.clamp(0.0, 1.0) * 32768.0 + 0.5) as u16;
    px[2 * slot..2 * slot + 2].copy_from_slice(&v.to_ne_bytes());
}
#[inline(always)]
fn f32f(px: &[u8], slot: usize) -> f32 {
    finite_or_zero(f32::from_ne_bytes([px[4 * slot], px[4 * slot + 1], px[4 * slot + 2], px[4 * slot + 3]]))
}

impl Codec for Argb8 {
    const BPP: usize = 4;
    #[inline(always)]
    fn load(px: &[u8]) -> [f32; 4] {
        [u8f(px[1]), u8f(px[2]), u8f(px[3]), u8f(px[0])]
    }
    #[inline(always)]
    fn store(c: [f32; 4], px: &mut [u8]) {
        px[..4].copy_from_slice(&[f8(c[3]), f8(c[0]), f8(c[1]), f8(c[2])]);
    }
}
impl Codec for Bgra8 {
    const BPP: usize = 4;
    #[inline(always)]
    fn load(px: &[u8]) -> [f32; 4] {
        [u8f(px[2]), u8f(px[1]), u8f(px[0]), u8f(px[3])]
    }
    #[inline(always)]
    fn store(c: [f32; 4], px: &mut [u8]) {
        px[..4].copy_from_slice(&[f8(c[2]), f8(c[1]), f8(c[0]), f8(c[3])]);
    }
}
impl Codec for Argb16 {
    const BPP: usize = 8;
    #[inline(always)]
    fn load(px: &[u8]) -> [f32; 4] {
        [u16f(px, 1), u16f(px, 2), u16f(px, 3), u16f(px, 0)]
    }
    #[inline(always)]
    fn store(c: [f32; 4], px: &mut [u8]) {
        f16(c[3], px, 0);
        f16(c[0], px, 1);
        f16(c[1], px, 2);
        f16(c[2], px, 3);
    }
}
impl Codec for Bgra16f {
    const BPP: usize = 8;
    #[inline(always)]
    fn load(px: &[u8]) -> [f32; 4] {
        [f16f(px, 2), f16f(px, 1), f16f(px, 0), f16f(px, 3)]
    }
    #[inline(always)]
    fn store(c: [f32; 4], px: &mut [u8]) {
        for (slot, v) in [(2, c[0]), (1, c[1]), (0, c[2]), (3, c[3])] {
            px[2 * slot..2 * slot + 2].copy_from_slice(&f32_to_f16(v).to_ne_bytes());
        }
    }
}
impl Codec for Bgra32f {
    const BPP: usize = 16;
    #[inline(always)]
    fn load(px: &[u8]) -> [f32; 4] {
        [f32f(px, 2), f32f(px, 1), f32f(px, 0), f32f(px, 3)]
    }
    #[inline(always)]
    fn store(c: [f32; 4], px: &mut [u8]) {
        for (slot, v) in [(2, c[0]), (1, c[1]), (0, c[2]), (3, c[3])] {
            px[4 * slot..4 * slot + 4].copy_from_slice(&v.to_ne_bytes());
        }
    }
}

/// Stored pixel -> premultiplied RGBA (the exact arithmetic both read paths share, so the
/// direct luma decode reproduces read_rgba + luma_from_rgba bit for bit).
#[inline(always)]
fn load_premul<C: Codec>(px: &[u8], straight: bool) -> [f32; 4] {
    let mut c = C::load(px);
    if straight {
        let a = c[3];
        c[0] *= a;
        c[1] *= a;
        c[2] *= a;
    }
    c
}

/// Dispatches a generic row kernel on the runtime layout.
macro_rules! with_codec {
    ($layout:expr, $f:ident ( $($arg:expr),* )) => {
        match $layout {
            Layout::Argb8 => $f::<Argb8>($($arg),*),
            Layout::Argb16 => $f::<Argb16>($($arg),*),
            Layout::Bgra8 => $f::<Bgra8>($($arg),*),
            Layout::Bgra16f => $f::<Bgra16f>($($arg),*),
            Layout::Bgra32f => $f::<Bgra32f>($($arg),*),
        }
    };
}

/// Host buffer -> tightly packed premultiplied RGBA f32 (len = width*height*4). `out` is resized.
/// Rows are converted in parallel on rayon's global pool (shared with the core).
pub fn read_rgba(src: &[u8], desc: &ImageDesc, out: &mut Vec<f32>) -> Result<(), PixelError> {
    desc.validate(src.len())?;
    let rgba_len = desc.width.checked_mul(desc.height).and_then(|n| n.checked_mul(4)).ok_or(PixelError::BufferTooSmall)?;
    out.clear();
    out.resize(rgba_len, 0.0);
    #[cfg(target_arch = "aarch64")]
    if desc.layout == Layout::Bgra8 {
        read_rgba_bgra8_neon(src, desc, out);
        return Ok(());
    }
    with_codec!(desc.layout, read_rgba_rows(src, desc, out));
    Ok(())
}

/// Host buffer -> tightly packed premultiplied RGBA f32, luma plane (len = width*height), and
/// largest alpha in a single parallel pass. Reproduces `read_rgba` + `core::luma_from_rgba` +
/// `max_alpha` bit for bit.
pub fn read_rgba_and_luma(src: &[u8], desc: &ImageDesc, out: &mut Vec<f32>) -> Result<(Vec<f32>, f32), PixelError> {
    desc.validate(src.len())?;
    let pixels = desc.width.checked_mul(desc.height).ok_or(PixelError::BufferTooSmall)?;
    let rgba_len = pixels.checked_mul(4).ok_or(PixelError::BufferTooSmall)?;
    out.clear();
    out.resize(rgba_len, 0.0);
    let mut luma = vec![0.0f32; pixels];
    #[cfg(target_arch = "aarch64")]
    if desc.layout == Layout::Bgra8 {
        let max_a = read_rgba_and_luma_bgra8_neon(src, desc, out, &mut luma);
        return Ok((luma, max_a));
    }
    let max_a = with_codec!(desc.layout, read_rgba_and_luma_rows(src, desc, out, &mut luma));
    Ok((luma, max_a))
}

fn read_rgba_rows<C: Codec>(src: &[u8], desc: &ImageDesc, out: &mut [f32]) {
    let (w, straight) = (desc.width, desc.alpha == Alpha::Straight);
    par::for_rows_mut(out, w * 4, 32, |y0, chunk| {
        for (dy, dst) in chunk.chunks_exact_mut(w * 4).enumerate() {
            let y = y0 + dy;
            let off = desc.row_offset(y);
            let row = &src[off..off + w * C::BPP];
            for (px, d) in row.chunks_exact(C::BPP).zip(dst.chunks_exact_mut(4)) {
                d.copy_from_slice(&load_premul::<C>(px, straight));
            }
        }
    });
}

fn read_rgba_and_luma_rows<C: Codec>(src: &[u8], desc: &ImageDesc, out: &mut [f32], luma: &mut [f32]) -> f32 {
    let [wr, wg, wb] = tblur_core::LUMA_WEIGHTS;
    let (w, straight) = (desc.width, desc.alpha == Alpha::Straight);
    let row_max_sync = std::sync::atomic::AtomicU32::new(0);
    par::for_rows_mut2(out, luma, w * 4, w, 32, |y0, rchunk, lchunk| {
        let mut local_max = 0.0f32;
        for (dy, (dst, ldst)) in rchunk.chunks_exact_mut(w * 4).zip(lchunk.chunks_exact_mut(w)).enumerate() {
            let y = y0 + dy;
            let off = desc.row_offset(y);
            let row = &src[off..off + w * C::BPP];
            for ((px, d), l) in row.chunks_exact(C::BPP).zip(dst.chunks_exact_mut(4)).zip(ldst.iter_mut()) {
                let c = load_premul::<C>(px, straight);
                d.copy_from_slice(&c);
                *l = wr * c[0] + wg * c[1] + wb * c[2];
                local_max = local_max.max(c[3]);
            }
        }
        // Atomic max over non-negative f32s (IEEE-754 order matches u32 order for >= 0.0).
        if local_max > 0.0 {
            let bits = local_max.to_bits();
            row_max_sync.fetch_max(bits, std::sync::atomic::Ordering::Relaxed);
        }
    });
    f32::from_bits(row_max_sync.load(std::sync::atomic::Ordering::Relaxed))
}

/// Host buffer -> luma plane (len = width*height) plus the largest alpha, in one parallel pass
/// and without materialising RGBA. Luma uses `tblur_core::LUMA_WEIGHTS` on the premultiplied,
/// gamma-encoded values with the same arithmetic as `read_rgba` + `tblur_core::luma_from_rgba`.
pub fn read_luma(src: &[u8], desc: &ImageDesc) -> Result<(Vec<f32>, f32), PixelError> {
    desc.validate(src.len())?;
    let pixels = desc.width.checked_mul(desc.height).ok_or(PixelError::BufferTooSmall)?;
    let mut luma = vec![0f32; pixels];
    #[cfg(target_arch = "aarch64")]
    if desc.layout == Layout::Bgra8 {
        let max_a = read_luma_bgra8_neon(src, desc, &mut luma);
        return Ok((luma, max_a));
    }
    let max_a = with_codec!(desc.layout, read_luma_rows(src, desc, &mut luma));
    Ok((luma, max_a))
}

fn read_luma_rows<C: Codec>(src: &[u8], desc: &ImageDesc, luma: &mut [f32]) -> f32 {
    let [wr, wg, wb] = tblur_core::LUMA_WEIGHTS;
    let (w, h, straight) = (desc.width, desc.height, desc.alpha == Alpha::Straight);
    let mut row_max = vec![0.0f32; h];
    par::for_rows_mut2(luma, &mut row_max, w, 1, 32, |y0, lchunk, mchunk| {
        for (dy, (dst, slot)) in lchunk.chunks_exact_mut(w).zip(mchunk.iter_mut()).enumerate() {
            let y = y0 + dy;
            let off = desc.row_offset(y);
            let row = &src[off..off + w * C::BPP];
            let mut max_a = 0f32;
            for (px, l) in row.chunks_exact(C::BPP).zip(dst.iter_mut()) {
                let c = load_premul::<C>(px, straight);
                *l = wr * c[0] + wg * c[1] + wb * c[2];
                max_a = max_a.max(c[3]);
            }
            *slot = max_a;
        }
    });
    row_max.into_iter().fold(0.0f32, f32::max)
}

/// Tightly packed premultiplied RGBA f32 -> host buffer. Only the `width*bpp` pixel bytes of each
/// row are written; row padding is left untouched. Rows are written in parallel.
pub fn write_rgba(src: &[f32], dst: &mut [u8], desc: &ImageDesc) -> Result<(), PixelError> {
    desc.validate(dst.len())?;
    let expected = desc.width.checked_mul(desc.height).and_then(|n| n.checked_mul(4)).ok_or(PixelError::SizeMismatch)?;
    if src.len() != expected {
        return Err(PixelError::SizeMismatch);
    }
    #[cfg(target_arch = "aarch64")]
    if desc.layout == Layout::Bgra8 {
        write_rgba_bgra8_neon(src, dst, desc);
        return Ok(());
    }
    with_codec!(desc.layout, write_rgba_rows(src, dst, desc));
    Ok(())
}

// ── aarch64 NEON fast paths for Premiere BGRA_4444_8u (bit-exact with scalar Bgra8) ──

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn u8x16_to_f32x4x4(v: std::arch::aarch64::uint8x16_t, scale: std::arch::aarch64::float32x4_t) -> [std::arch::aarch64::float32x4_t; 4] {
    use std::arch::aarch64::*;
    unsafe {
        let lo16 = vmovl_u8(vget_low_u8(v));
        let hi16 = vmovl_u8(vget_high_u8(v));
        [
            vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo16))), scale),
            vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo16))), scale),
            vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi16))), scale),
            vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi16))), scale),
        ]
    }
}

#[cfg(target_arch = "aarch64")]
fn read_rgba_bgra8_neon(src: &[u8], desc: &ImageDesc, out: &mut [f32]) {
    use std::arch::aarch64::*;
    let (w, straight) = (desc.width, desc.alpha == Alpha::Straight);
    par::for_rows_mut(out, w * 4, 32, |y0, chunk| {
        for (dy, dst) in chunk.chunks_exact_mut(w * 4).enumerate() {
            let y = y0 + dy;
            let off = desc.row_offset(y);
            let row = &src[off..off + w * 4];
            let n = (row.len() / 4).min(dst.len() / 4);
            let mut x = 0usize;
            unsafe {
                let vscale = vdupq_n_f32(1.0 / 255.0);
                while x + 16 <= n {
                    let bgra = vld4q_u8(row.as_ptr().add(x * 4));
                    let bs = u8x16_to_f32x4x4(bgra.0, vscale);
                    let gs = u8x16_to_f32x4x4(bgra.1, vscale);
                    let rs = u8x16_to_f32x4x4(bgra.2, vscale);
                    let a_s = u8x16_to_f32x4x4(bgra.3, vscale);
                    let dp = dst.as_mut_ptr().add(x * 4);
                    for q in 0..4 {
                        let (mut r, mut g, mut b, a) = (rs[q], gs[q], bs[q], a_s[q]);
                        if straight {
                            r = vmulq_f32(r, a);
                            g = vmulq_f32(g, a);
                            b = vmulq_f32(b, a);
                        }
                        vst4q_f32(dp.add(q * 16), float32x4x4_t(r, g, b, a));
                    }
                    x += 16;
                }
            }
            while x < n {
                let c = load_premul::<Bgra8>(&row[x * 4..x * 4 + 4], straight);
                dst[x * 4..x * 4 + 4].copy_from_slice(&c);
                x += 1;
            }
        }
    });
}

#[cfg(target_arch = "aarch64")]
fn read_rgba_and_luma_bgra8_neon(src: &[u8], desc: &ImageDesc, out: &mut [f32], luma: &mut [f32]) -> f32 {
    use std::arch::aarch64::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    let [wr, wg, wb] = tblur_core::LUMA_WEIGHTS;
    let (w, straight) = (desc.width, desc.alpha == Alpha::Straight);
    let max_bits = AtomicU32::new(0);
    par::for_rows_mut2(out, luma, w * 4, w, 32, |y0, rchunk, lchunk| {
        let mut local_max = 0.0f32;
        for (dy, (dst, ldst)) in rchunk.chunks_exact_mut(w * 4).zip(lchunk.chunks_exact_mut(w)).enumerate() {
            let y = y0 + dy;
            let off = desc.row_offset(y);
            let row = &src[off..off + w * 4];
            let n = (row.len() / 4).min(dst.len() / 4).min(ldst.len());
            let mut x = 0usize;
            unsafe {
                let vscale = vdupq_n_f32(1.0 / 255.0);
                let vwr = vdupq_n_f32(wr);
                let vwg = vdupq_n_f32(wg);
                let vwb = vdupq_n_f32(wb);
                let mut vmax_a = vdupq_n_u8(0);
                while x + 16 <= n {
                    let bgra = vld4q_u8(row.as_ptr().add(x * 4));
                    vmax_a = vmaxq_u8(vmax_a, bgra.3);
                    let bs = u8x16_to_f32x4x4(bgra.0, vscale);
                    let gs = u8x16_to_f32x4x4(bgra.1, vscale);
                    let rs = u8x16_to_f32x4x4(bgra.2, vscale);
                    let a_s = u8x16_to_f32x4x4(bgra.3, vscale);
                    let dp = dst.as_mut_ptr().add(x * 4);
                    let lp = ldst.as_mut_ptr().add(x);
                    for q in 0..4 {
                        let (mut r, mut g, mut b, a) = (rs[q], gs[q], bs[q], a_s[q]);
                        if straight {
                            r = vmulq_f32(r, a);
                            g = vmulq_f32(g, a);
                            b = vmulq_f32(b, a);
                        }
                        vst4q_f32(dp.add(q * 16), float32x4x4_t(r, g, b, a));
                        let vl = vaddq_f32(vaddq_f32(vmulq_f32(vwr, r), vmulq_f32(vwg, g)), vmulq_f32(vwb, b));
                        vst1q_f32(lp.add(q * 4), vl);
                    }
                    x += 16;
                }
                let ma = vmaxvq_u8(vmax_a) as f32 * (1.0 / 255.0);
                local_max = local_max.max(ma);
            }
            while x < n {
                let c = load_premul::<Bgra8>(&row[x * 4..x * 4 + 4], straight);
                dst[x * 4..x * 4 + 4].copy_from_slice(&c);
                ldst[x] = wr * c[0] + wg * c[1] + wb * c[2];
                local_max = local_max.max(c[3]);
                x += 1;
            }
        }
        if local_max > 0.0 {
            max_bits.fetch_max(local_max.to_bits(), Ordering::Relaxed);
        }
    });
    f32::from_bits(max_bits.load(Ordering::Relaxed))
}

#[cfg(target_arch = "aarch64")]
fn read_luma_bgra8_neon(src: &[u8], desc: &ImageDesc, luma: &mut [f32]) -> f32 {
    use std::arch::aarch64::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    let [wr, wg, wb] = tblur_core::LUMA_WEIGHTS;
    let (w, straight) = (desc.width, desc.alpha == Alpha::Straight);
    let max_bits = AtomicU32::new(0);
    par::for_rows_mut(luma, w, 32, |y0, lchunk| {
        let mut local_max = 0.0f32;
        for (dy, dst) in lchunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let off = desc.row_offset(y);
            let row = &src[off..off + w * 4];
            let n = (row.len() / 4).min(dst.len());
            let mut x = 0usize;
            unsafe {
                let vscale = vdupq_n_f32(1.0 / 255.0);
                let vwr = vdupq_n_f32(wr);
                let vwg = vdupq_n_f32(wg);
                let vwb = vdupq_n_f32(wb);
                let mut vmax_a = vdupq_n_u8(0);
                while x + 16 <= n {
                    let bgra = vld4q_u8(row.as_ptr().add(x * 4));
                    vmax_a = vmaxq_u8(vmax_a, bgra.3);
                    let bs = u8x16_to_f32x4x4(bgra.0, vscale);
                    let gs = u8x16_to_f32x4x4(bgra.1, vscale);
                    let rs = u8x16_to_f32x4x4(bgra.2, vscale);
                    let a_s = u8x16_to_f32x4x4(bgra.3, vscale);
                    let lp = dst.as_mut_ptr().add(x);
                    for q in 0..4 {
                        let (mut r, mut g, mut b, a) = (rs[q], gs[q], bs[q], a_s[q]);
                        if straight {
                            r = vmulq_f32(r, a);
                            g = vmulq_f32(g, a);
                            b = vmulq_f32(b, a);
                        }
                        let vl = vaddq_f32(vaddq_f32(vmulq_f32(vwr, r), vmulq_f32(vwg, g)), vmulq_f32(vwb, b));
                        vst1q_f32(lp.add(q * 4), vl);
                    }
                    x += 16;
                }
                let ma = vmaxvq_u8(vmax_a) as f32 * (1.0 / 255.0);
                local_max = local_max.max(ma);
            }
            while x < n {
                let c = load_premul::<Bgra8>(&row[x * 4..x * 4 + 4], straight);
                dst[x] = wr * c[0] + wg * c[1] + wb * c[2];
                local_max = local_max.max(c[3]);
                x += 1;
            }
        }
        if local_max > 0.0 {
            max_bits.fetch_max(local_max.to_bits(), Ordering::Relaxed);
        }
    });
    f32::from_bits(max_bits.load(Ordering::Relaxed))
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn finite_or_zero_x4(v: std::arch::aarch64::float32x4_t, vzero: std::arch::aarch64::float32x4_t) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let mask = vceqq_f32(vsubq_f32(v, v), vzero);
        vbslq_f32(mask, v, vzero)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn f32x4_to_u8_lane(
    v: std::arch::aarch64::float32x4_t,
    vzero: std::arch::aarch64::float32x4_t,
    vone: std::arch::aarch64::float32x4_t,
    v255: std::arch::aarch64::float32x4_t,
    v05: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::uint16x4_t {
    use std::arch::aarch64::*;
    unsafe {
        let cl = vminq_f32(vmaxq_f32(v, vzero), vone);
        let q = vaddq_f32(vmulq_f32(cl, v255), v05);
        vqmovn_u32(vcvtq_u32_f32(q))
    }
}

#[cfg(target_arch = "aarch64")]
fn write_rgba_bgra8_neon(src: &[f32], dst: &mut [u8], desc: &ImageDesc) {
    use std::arch::aarch64::*;
    let (w, h, straight) = (desc.width, desc.height, desc.alpha == Alpha::Straight);
    let stride = desc.stride();
    let used = desc.required_len();
    par::for_rows_mut(&mut dst[..used], stride, 32, |m0, chunk| {
        for (dm, row) in chunk.chunks_mut(stride).enumerate() {
            let m = m0 + dm;
            if m >= h {
                break;
            }
            let y = if desc.rowbytes >= 0 { m } else { h - 1 - m };
            let s_row = &src[y * w * 4..(y + 1) * w * 4];
            let d_row = &mut row[..w * 4];
            let n = (s_row.len() / 4).min(d_row.len() / 4);
            let mut x = 0usize;
            unsafe {
                let vzero = vdupq_n_f32(0.0);
                let vone = vdupq_n_f32(1.0);
                let vmin_pos = vdupq_n_f32(f32::MIN_POSITIVE);
                let v255 = vdupq_n_f32(255.0);
                let v05 = vdupq_n_f32(0.5);
                while x + 16 <= n {
                    let mut b16 = [vdup_n_u16(0); 4];
                    let mut g16 = [vdup_n_u16(0); 4];
                    let mut r16 = [vdup_n_u16(0); 4];
                    let mut a16 = [vdup_n_u16(0); 4];
                    let sp = s_row.as_ptr().add(x * 4);
                    for q in 0..4 {
                        let rgba = vld4q_f32(sp.add(q * 16));
                        let mut r = finite_or_zero_x4(rgba.0, vzero);
                        let mut g = finite_or_zero_x4(rgba.1, vzero);
                        let mut b = finite_or_zero_x4(rgba.2, vzero);
                        let a = finite_or_zero_x4(rgba.3, vzero);
                        if straight {
                            let pos = vcgeq_f32(a, vmin_pos);
                            let inv = vdivq_f32(vone, a);
                            r = vbslq_f32(pos, finite_or_zero_x4(vmulq_f32(r, inv), vzero), vzero);
                            g = vbslq_f32(pos, finite_or_zero_x4(vmulq_f32(g, inv), vzero), vzero);
                            b = vbslq_f32(pos, finite_or_zero_x4(vmulq_f32(b, inv), vzero), vzero);
                        }
                        b16[q] = f32x4_to_u8_lane(b, vzero, vone, v255, v05);
                        g16[q] = f32x4_to_u8_lane(g, vzero, vone, v255, v05);
                        r16[q] = f32x4_to_u8_lane(r, vzero, vone, v255, v05);
                        a16[q] = f32x4_to_u8_lane(a, vzero, vone, v255, v05);
                    }
                    let b8 = vcombine_u8(vqmovn_u16(vcombine_u16(b16[0], b16[1])), vqmovn_u16(vcombine_u16(b16[2], b16[3])));
                    let g8 = vcombine_u8(vqmovn_u16(vcombine_u16(g16[0], g16[1])), vqmovn_u16(vcombine_u16(g16[2], g16[3])));
                    let r8 = vcombine_u8(vqmovn_u16(vcombine_u16(r16[0], r16[1])), vqmovn_u16(vcombine_u16(r16[2], r16[3])));
                    let a8 = vcombine_u8(vqmovn_u16(vcombine_u16(a16[0], a16[1])), vqmovn_u16(vcombine_u16(a16[2], a16[3])));
                    vst4q_u8(d_row.as_mut_ptr().add(x * 4), uint8x16x4_t(b8, g8, r8, a8));
                    x += 16;
                }
            }
            while x < n {
                let s = &s_row[x * 4..x * 4 + 4];
                let mut c = [finite_or_zero(s[0]), finite_or_zero(s[1]), finite_or_zero(s[2]), finite_or_zero(s[3])];
                if straight {
                    let a = c[3];
                    if a >= f32::MIN_POSITIVE {
                        let inv = 1.0 / a;
                        c[0] = finite_or_zero(c[0] * inv);
                        c[1] = finite_or_zero(c[1] * inv);
                        c[2] = finite_or_zero(c[2] * inv);
                    } else {
                        c = [0.0, 0.0, 0.0, if a > 0.0 { 0.0 } else { a }];
                    }
                }
                Bgra8::store(c, &mut d_row[x * 4..x * 4 + 4]);
                x += 1;
            }
        }
    });
}

fn write_rgba_rows<C: Codec>(src: &[f32], dst: &mut [u8], desc: &ImageDesc) {
    let (w, h, straight) = (desc.width, desc.height, desc.alpha == Alpha::Straight);
    let stride = desc.stride();
    let used = desc.required_len();
    par::for_rows_mut(&mut dst[..used], stride, 32, |m0, chunk| {
        for (dm, row) in chunk.chunks_mut(stride).enumerate() {
            let m = m0 + dm;
            if m >= h {
                break;
            }
            let y = if desc.rowbytes >= 0 { m } else { h - 1 - m };
            let s_row = &src[y * w * 4..(y + 1) * w * 4];
            for (px, s) in row[..w * C::BPP].chunks_exact_mut(C::BPP).zip(s_row.chunks_exact(4)) {
                let mut c = [finite_or_zero(s[0]), finite_or_zero(s[1]), finite_or_zero(s[2]), finite_or_zero(s[3])];
                if straight {
                    let a = c[3];
                    if a >= f32::MIN_POSITIVE {
                        let inv = 1.0 / a;
                        c[0] = finite_or_zero(c[0] * inv);
                        c[1] = finite_or_zero(c[1] * inv);
                        c[2] = finite_or_zero(c[2] * inv);
                    } else {
                        c = [0.0, 0.0, 0.0, if a > 0.0 { 0.0 } else { a }];
                    }
                }
                C::store(c, px);
            }
        }
    });
}

/// Format-agnostic row copy between two worlds of the same width/height/pixel format (the
/// pass-through fallback). Copies `row_len` bytes per row, honouring each side's signed rowbytes.
/// Works for formats we cannot otherwise decode, since it never interprets the pixels.
pub fn copy_rows(src: &[u8], src_rowbytes: isize, dst: &mut [u8], dst_rowbytes: isize, height: usize, row_len: usize) -> Result<(), PixelError> {
    if height == 0 || row_len == 0 {
        return Err(PixelError::Empty);
    }
    let (ss, ds) = (src_rowbytes.unsigned_abs(), dst_rowbytes.unsigned_abs());
    if ss < row_len || ds < row_len {
        return Err(PixelError::RowbytesTooSmall);
    }
    let need_s = (height - 1).checked_mul(ss).and_then(|v| v.checked_add(row_len)).ok_or(PixelError::BufferTooSmall)?;
    let need_d = (height - 1).checked_mul(ds).and_then(|v| v.checked_add(row_len)).ok_or(PixelError::BufferTooSmall)?;
    if src.len() < need_s || dst.len() < need_d {
        return Err(PixelError::BufferTooSmall);
    }
    for y in 0..height {
        let so = if src_rowbytes >= 0 { y * ss } else { (height - 1 - y) * ss };
        let d_o = if dst_rowbytes >= 0 { y * ds } else { (height - 1 - y) * ds };
        dst[d_o..d_o + row_len].copy_from_slice(&src[so..so + row_len]);
    }
    Ok(())
}

/// Largest alpha in a packed RGBA f32 image (used to reject all-transparent "frames" a host may
/// return when checking out beyond the end of the media).
pub fn max_alpha(rgba: &[f32]) -> f32 {
    rgba.chunks_exact(4).fold(0.0f32, |m, p| m.max(p[3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(w: usize, h: usize, rb: isize, layout: Layout, alpha: Alpha) -> ImageDesc {
        ImageDesc { width: w, height: h, rowbytes: rb, layout, alpha }
    }

    /// Deterministic opaque-ish test pattern, values exactly representable in 8 bits.
    fn pattern8(w: usize, h: usize) -> Vec<[u8; 4]> {
        (0..w * h).map(|i| [(i * 37 % 256) as u8, (i * 91 % 256) as u8, (i * 13 % 256) as u8, (255 - i * 7 % 200) as u8]).collect()
    }

    #[test]
    fn bgra8_channel_order_and_premultiply() {
        // One pixel B=10 G=20 R=30 A=128 (straight).
        let buf = [10u8, 20, 30, 128];
        let d = desc(1, 1, 4, Layout::Bgra8, Alpha::Straight);
        let mut out = Vec::new();
        read_rgba(&buf, &d, &mut out).unwrap();
        let a = 128.0 / 255.0;
        let exp = [30.0 / 255.0 * a, 20.0 / 255.0 * a, 10.0 / 255.0 * a, a];
        for i in 0..4 {
            assert!((out[i] - exp[i]).abs() < 1e-6, "{i}: {} vs {}", out[i], exp[i]);
        }
        let mut back = [0u8; 4];
        write_rgba(&out, &mut back, &d).unwrap();
        assert_eq!(back, buf);
    }

    #[test]
    fn argb8_channel_order() {
        let buf = [200u8, 1, 2, 3]; // A R G B, premultiplied
        let d = desc(1, 1, 4, Layout::Argb8, Alpha::Premultiplied);
        let mut out = Vec::new();
        read_rgba(&buf, &d, &mut out).unwrap();
        let exp = [1.0 / 255.0, 2.0 / 255.0, 3.0 / 255.0, 200.0 / 255.0]; // R G B A
        for i in 0..4 {
            assert!((out[i] - exp[i]).abs() < 1e-6, "{i}: {} vs {}", out[i], exp[i]);
        }
        let mut back = [0u8; 4];
        write_rgba(&out, &mut back, &d).unwrap();
        assert_eq!(back, buf);
    }

    #[test]
    fn argb16_white_is_32768() {
        let mut buf = Vec::new();
        for v in [32768u16, 32768, 16384, 0] {
            buf.extend_from_slice(&v.to_ne_bytes());
        }
        let d = desc(1, 1, 8, Layout::Argb16, Alpha::Premultiplied);
        let mut out = Vec::new();
        read_rgba(&buf, &d, &mut out).unwrap();
        assert_eq!(out, vec![1.0, 0.5, 0.0, 1.0]); // R G B A
        // Over-range values clamp to 32768, never wrap to 65535-ish garbage.
        let mut back = vec![0u8; 8];
        write_rgba(&[2.0, 1.0, 0.5, 1.0], &mut back, &d).unwrap();
        let vals: Vec<u16> = back.chunks(2).map(|c| u16::from_ne_bytes([c[0], c[1]])).collect();
        assert_eq!(vals, vec![32768, 32768, 32768, 16384]); // A R G B
    }

    #[test]
    fn bgra32f_order_and_values_above_one_survive() {
        let px = [0.25f32, 0.5, 3.5, 1.0]; // B G R A, straight
        let buf: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let d = desc(1, 1, 16, Layout::Bgra32f, Alpha::Straight);
        let mut out = Vec::new();
        read_rgba(&buf, &d, &mut out).unwrap();
        assert_eq!(out, vec![3.5, 0.5, 0.25, 1.0]);
        let mut back = vec![0u8; 16];
        write_rgba(&out, &mut back, &d).unwrap();
        assert_eq!(back, buf);
    }

    #[test]
    fn bgra32f_straight_alpha_roundtrip_and_nonfinite() {
        let px = [0.2f32, 0.4, 1.6, 0.5, f32::NAN, 1.0, f32::INFINITY, 1.0];
        let buf: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
        let d = desc(2, 1, 32, Layout::Bgra32f, Alpha::Straight);
        let mut out = Vec::new();
        read_rgba(&buf, &d, &mut out).unwrap();
        assert_eq!(&out[0..4], &[0.8, 0.2, 0.1, 0.5]); // premultiplied
        assert!(out.iter().all(|v| v.is_finite()));
        let mut back = vec![0u8; 32];
        write_rgba(&out, &mut back, &d).unwrap();
        let vals: Vec<f32> = back.chunks(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        for (a, b) in vals[0..4].iter().zip(&px[0..4]) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_alpha_unpremultiplies_to_black() {
        let d = desc(1, 1, 4, Layout::Bgra8, Alpha::Straight);
        let mut back = [9u8; 4];
        write_rgba(&[0.3, 0.3, 0.3, 0.0], &mut back, &d).unwrap();
        assert_eq!(back, [0, 0, 0, 0]);
    }

    fn roundtrip_padded(layout: Layout, alpha: Alpha, negative: bool) {
        let (w, h) = (5usize, 3usize);
        let bpp = layout.bytes_per_pixel();
        let stride = w * bpp + 13; // odd padding
        let rb = if negative { -(stride as isize) } else { stride as isize };
        let d = desc(w, h, rb, layout, alpha);
        let pat = pattern8(w, h);
        // Build the host buffer by hand, with padding bytes set to a sentinel.
        let mut buf = vec![0xEEu8; d.required_len() + 13];
        for y in 0..h {
            let off = if negative { (h - 1 - y) * stride } else { y * stride };
            for x in 0..w {
                let [r, g, b, a] = pat[y * w + x];
                let slots = layout.rgba_slots();
                let rgba = [r, g, b, a];
                for i in 0..4 {
                    let s = slots[i];
                    match layout {
                        Layout::Argb8 | Layout::Bgra8 => buf[off + x * bpp + s] = rgba[i],
                        Layout::Argb16 => {
                            let v = (rgba[i] as u32 * 32768 / 255) as u16;
                            buf[off + x * bpp + 2 * s..off + x * bpp + 2 * s + 2].copy_from_slice(&v.to_ne_bytes());
                        }
                        Layout::Bgra16f => {
                            let v = f32_to_f16(rgba[i] as f32 / 255.0);
                            buf[off + x * bpp + 2 * s..off + x * bpp + 2 * s + 2].copy_from_slice(&v.to_ne_bytes());
                        }
                        Layout::Bgra32f => {
                            let v = rgba[i] as f32 / 255.0;
                            buf[off + x * bpp + 4 * s..off + x * bpp + 4 * s + 4].copy_from_slice(&v.to_ne_bytes());
                        }
                    }
                }
            }
        }
        let mut rgba = Vec::new();
        read_rgba(&buf, &d, &mut rgba).unwrap();
        assert_eq!(rgba.len(), w * h * 4);
        // Row 0 of the packed output is host row 0 regardless of the rowbytes sign.
        let expect_r = |i: usize| {
            let [r, _, _, a] = pat[i];
            let k = if alpha == Alpha::Straight { a as f32 / 255.0 } else { 1.0 };
            r as f32 / 255.0 * k
        };
        assert!((rgba[0] - expect_r(0)).abs() < 1e-3);
        assert!((rgba[(w * (h - 1)) * 4] - expect_r(w * (h - 1))).abs() < 1e-3);
        // Direct luma decode == read_rgba + core::luma_from_rgba (same arithmetic, so exact up
        // to float rounding), and its max alpha == max over the RGBA image.
        let (luma, max_a) = read_luma(&buf, &d).unwrap();
        let reference = tblur_core::luma_from_rgba(tblur_core::Frame { width: w, height: h, data: &rgba });
        assert_eq!(luma.len(), reference.len());
        for (a, b) in luma.iter().zip(&reference) {
            assert!((a - b).abs() <= 1e-6 * b.abs().max(1.0), "{layout:?} {alpha:?} neg={negative}: {a} vs {b}");
        }
        assert_eq!(max_a, max_alpha(&rgba));
        let mut back = vec![0xEEu8; buf.len()];
        write_rgba(&rgba, &mut back, &d).unwrap();
        if matches!(layout, Layout::Bgra32f | Layout::Bgra16f) && alpha == Alpha::Straight {
            // c*a/a is not exact in floating point: compare pixel values with tolerance (padding must be untouched).
            for y in 0..h {
                let off = if negative { (h - 1 - y) * stride } else { y * stride };
                if layout == Layout::Bgra32f {
                    for (a, b) in back[off..off + w * bpp].chunks(4).zip(buf[off..off + w * bpp].chunks(4)) {
                        let (a, b) = (f32::from_ne_bytes(a.try_into().unwrap()), f32::from_ne_bytes(b.try_into().unwrap()));
                        assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
                    }
                } else {
                    for (a, b) in back[off..off + w * bpp].chunks(2).zip(buf[off..off + w * bpp].chunks(2)) {
                        let (a, b) = (f16_to_f32(u16::from_ne_bytes([a[0], a[1]])), f16_to_f32(u16::from_ne_bytes([b[0], b[1]])));
                        assert!((a - b).abs() <= 2e-3, "{a} vs {b}");
                    }
                }
                assert_eq!(&back[off + w * bpp..off + stride], &buf[off + w * bpp..off + stride]);
            }
        } else {
            assert_eq!(back, buf, "{layout:?} {alpha:?} negative={negative}");
        }
    }

    #[test]
    fn roundtrip_all_formats_with_padding_both_rowbyte_signs() {
        for layout in [Layout::Argb8, Layout::Argb16, Layout::Bgra8, Layout::Bgra16f, Layout::Bgra32f] {
            for alpha in [Alpha::Premultiplied, Alpha::Straight] {
                for neg in [false, true] {
                    roundtrip_padded(layout, alpha, neg);
                }
            }
        }
    }

    #[test]
    fn rejects_bad_geometry() {
        let mut out = Vec::new();
        assert_eq!(read_rgba(&[0; 16], &desc(4, 1, 8, Layout::Bgra8, Alpha::Straight), &mut out), Err(PixelError::RowbytesTooSmall));
        assert_eq!(read_rgba(&[0; 15], &desc(2, 2, 8, Layout::Bgra8, Alpha::Straight), &mut out), Err(PixelError::BufferTooSmall));
        assert_eq!(read_rgba(&[], &desc(0, 2, 8, Layout::Bgra8, Alpha::Straight), &mut out), Err(PixelError::Empty));
        assert_eq!(write_rgba(&[0.0; 4], &mut [0; 16], &desc(2, 2, 8, Layout::Bgra8, Alpha::Straight)), Err(PixelError::SizeMismatch));
    }

    #[test]
    fn rejects_overflowing_dimensions() {
        let mut out = Vec::new();
        let mut dst = [0u8; 64];
        let cases = [
            // Exact width*4 wrap-around case (2^62 + 1) * 4 == 4 in wrapping usize arithmetic.
            desc((1usize << 62) + 1, 1, 4, Layout::Bgra8, Alpha::Straight),
            desc((1usize << 62) + 1, 1, 16, Layout::Bgra8, Alpha::Premultiplied),
            desc(usize::MAX, 1, 4, Layout::Bgra8, Alpha::Straight),
            desc((1usize << 61) + 1, 1, 8, Layout::Argb16, Alpha::Premultiplied),
            desc((1usize << 60) + 1, 1, 16, Layout::Bgra32f, Alpha::Straight),
            // Height * stride overflow.
            desc(1, (1usize << 62) + 1, 4, Layout::Bgra8, Alpha::Straight),
            desc(1, usize::MAX, 8, Layout::Bgra8, Alpha::Straight),
            desc(1usize << 33, 1usize << 33, isize::MAX, Layout::Bgra8, Alpha::Straight),
        ];
        for d in cases {
            assert!(read_rgba(&[0u8; 64], &d, &mut out).is_err(), "read_rgba should reject {d:?}");
            assert!(read_rgba_and_luma(&[0u8; 64], &d, &mut out).is_err(), "read_rgba_and_luma should reject {d:?}");
            assert!(read_luma(&[0u8; 64], &d).is_err(), "read_luma should reject {d:?}");
            assert!(write_rgba(&[0.0f32; 4], &mut dst, &d).is_err(), "write_rgba should reject {d:?}");
        }
        assert_eq!(copy_rows(&[0u8; 16], 4, &mut dst, 4, (1usize << 62) + 1, 4), Err(PixelError::BufferTooSmall));
    }

    #[test]
    fn copy_rows_handles_different_strides_and_signs() {
        let (h, row) = (3usize, 6usize);
        let src: Vec<u8> = (0..(h * 8) as u8).collect();
        let mut dst = vec![0u8; h * 10];
        copy_rows(&src, 8, &mut dst, -10, h, row).unwrap();
        for y in 0..h {
            assert_eq!(&dst[(h - 1 - y) * 10..(h - 1 - y) * 10 + row], &src[y * 8..y * 8 + row]);
        }
    }

    #[test]
    fn bgra8_neon_matches_scalar_all_widths_and_edges() {
        for &w in &[1usize, 15, 16, 17, 33, 64] {
            let h = 4usize;
            for alpha in [Alpha::Straight, Alpha::Premultiplied] {
                for neg in [false, true] {
                    let stride = w * 4 + 12;
                    let rb = if neg { -(stride as isize) } else { stride as isize };
                    let d = desc(w, h, rb, Layout::Bgra8, alpha);
                    let mut buf = vec![0xAAu8; d.required_len() + 12];
                    for y in 0..h {
                        let off = d.row_offset(y);
                        for x in 0..w {
                            let i = y * w + x;
                            let a = if i % 11 == 0 { 0u8 } else { ((i * 53 + 17) % 256) as u8 };
                            buf[off + x * 4] = ((i * 19) % 256) as u8;
                            buf[off + x * 4 + 1] = ((i * 73) % 256) as u8;
                            buf[off + x * 4 + 2] = ((i * 131) % 256) as u8;
                            buf[off + x * 4 + 3] = a;
                        }
                    }

                    // Scalar reference
                    let mut ref_rgba = vec![0.0f32; w * h * 4];
                    read_rgba_rows::<Bgra8>(&buf, &d, &mut ref_rgba);
                    let mut ref_luma = vec![0.0f32; w * h];
                    let ref_max_a = read_luma_rows::<Bgra8>(&buf, &d, &mut ref_luma);

                    // Public API (dispatches to NEON on aarch64)
                    let mut got_rgba = Vec::new();
                    read_rgba(&buf, &d, &mut got_rgba).unwrap();
                    assert_eq!(got_rgba, ref_rgba, "read_rgba w={w} {alpha:?} neg={neg}");

                    let (got_luma, got_max_a) = read_luma(&buf, &d).unwrap();
                    assert_eq!(got_luma, ref_luma, "read_luma w={w} {alpha:?} neg={neg}");
                    assert_eq!(got_max_a, ref_max_a, "read_luma max_a w={w} {alpha:?} neg={neg}");

                    let mut fused_rgba = Vec::new();
                    let (fused_luma, fused_max_a) = read_rgba_and_luma(&buf, &d, &mut fused_rgba).unwrap();
                    assert_eq!(fused_rgba, ref_rgba, "fused rgba w={w} {alpha:?} neg={neg}");
                    assert_eq!(fused_luma, ref_luma, "fused luma w={w} {alpha:?} neg={neg}");
                    assert_eq!(fused_max_a, ref_max_a, "fused max_a w={w} {alpha:?} neg={neg}");

                    // Write with edge floats (NaN, Inf, -Inf, subnormal alpha, f32::MAX/MIN, negative, >1.0, alpha=0)
                    let mut test_src = ref_rgba.clone();
                    if w * h >= 4 {
                        test_src[0] = f32::NAN;
                        test_src[1] = f32::INFINITY;
                        test_src[2] = -0.25;
                        test_src[3] = 1.5;
                        test_src[4] = 0.5;
                        test_src[5] = 0.5;
                        test_src[6] = 0.5;
                        test_src[7] = 0.0;
                        test_src[8] = 0.0;
                        test_src[9] = 1.0;
                        test_src[10] = f32::MAX;
                        test_src[11] = 1e-40;
                        test_src[12] = f32::NEG_INFINITY;
                        test_src[13] = f32::MIN;
                        test_src[14] = -0.0;
                        test_src[15] = -0.5;
                    }
                    let mut ref_out = vec![0x55u8; buf.len()];
                    write_rgba_rows::<Bgra8>(&test_src, &mut ref_out, &d);
                    let mut got_out = vec![0x55u8; buf.len()];
                    write_rgba(&test_src, &mut got_out, &d).unwrap();
                    assert_eq!(got_out, ref_out, "write_rgba w={w} {alpha:?} neg={neg}");
                }
            }
        }
    }

    #[test]
    fn f16_round_trip_all_finite_values_and_edges() {
        for bits in 0u16..=u16::MAX {
            let exp = (bits >> 10) & 0x1f;
            let f = f16_to_f32(bits);
            if exp == 31 {
                if (bits & 0x03ff) == 0 {
                    assert!(f.is_infinite(), "bits=0x{bits:04x}");
                } else {
                    assert!(f.is_nan(), "bits=0x{bits:04x}");
                }
            } else {
                assert!(f.is_finite(), "bits=0x{bits:04x}");
                let back = f32_to_f16(f);
                assert_eq!(back, bits, "round-trip failed for 0x{bits:04x} (f={f}) -> 0x{back:04x}");
            }
        }
        // Known exact constants.
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        // Round-to-nearest-even tie checks around 1.0 (ULP = 2^-10).
        let half_ulp = 2.0f32.powi(-11);
        assert_eq!(f32_to_f16(1.0 + half_ulp), 0x3c00); // ties to even (LSB 0)
        assert_eq!(f32_to_f16(1.0 + 3.0 * half_ulp), 0x3c02); // ties to even (LSB 0)
        // Finite overflow clamps to max finite (65504.0), never Inf.
        assert_eq!(f32_to_f16(100_000.0), 0x7bff);
        assert_eq!(f32_to_f16(-100_000.0), 0xfbff);
    }

    #[test]
    fn bgra16f_order_premultiply_hdr_and_padding() {
        let px = [0.25f32, 0.5, 3.5, 0.5]; // B G R A, straight
        let mut buf = vec![0xA5u8; 32]; // 1x2 with rowbytes 16 (8 bytes padding)
        for row in 0..2 {
            for (slot, &v) in px.iter().enumerate() {
                let off = row * 16 + slot * 2;
                buf[off..off + 2].copy_from_slice(&f32_to_f16(v).to_ne_bytes());
            }
        }
        let d = desc(1, 2, 16, Layout::Bgra16f, Alpha::Straight);
        let mut rgba = Vec::new();
        let (luma, max_a) = read_rgba_and_luma(&buf, &d, &mut rgba).unwrap();
        assert_eq!(max_a, 0.5);
        assert_eq!(rgba, vec![1.75, 0.25, 0.125, 0.5, 1.75, 0.25, 0.125, 0.5]);
        let (luma2, max_a2) = read_luma(&buf, &d).unwrap();
        assert_eq!(luma, luma2);
        assert_eq!(max_a, max_a2);

        let mut out = vec![0x3Cu8; 32];
        write_rgba(&rgba, &mut out, &d).unwrap();
        assert_eq!(&out[0..8], &buf[0..8]);
        assert_eq!(&out[16..24], &buf[16..24]);
        // Row padding untouched.
        assert!(out[8..16].iter().all(|&b| b == 0x3C));
        assert!(out[24..32].iter().all(|&b| b == 0x3C));
    }

    #[test]
    fn subnormal_alpha_and_extreme_hdr_produce_finite_outputs() {
        // Item 6 regression: subnormal straight alphas (< f32::MIN_POSITIVE) and huge HDR RGB
        // divided by tiny normal alphas must never produce Inf or NaN in 32f or 16f output.
        for a in [1e-45f32, 1e-40, 1e-39, f32::MIN_POSITIVE * 0.5, f32::MIN_POSITIVE, 1e-30, 1e-10] {
            let px: [f32; 4] = [1.0, 1.0, 1.0, a]; // B G R A straight
            let buf: Vec<u8> = px.iter().flat_map(|v| v.to_ne_bytes()).collect();
            let d32 = desc(1, 1, 16, Layout::Bgra32f, Alpha::Straight);
            let mut rgba = Vec::new();
            read_rgba_and_luma(&buf, &d32, &mut rgba).unwrap();

            let mut out32 = vec![0u8; 16];
            write_rgba(&rgba, &mut out32, &d32).unwrap();
            for chunk in out32.chunks_exact(4) {
                let v = f32::from_ne_bytes(chunk.try_into().unwrap());
                assert!(v.is_finite(), "32f non-finite {v} for alpha={a:e}");
            }

            let d16 = desc(1, 1, 8, Layout::Bgra16f, Alpha::Straight);
            let mut out16 = vec![0u8; 8];
            write_rgba(&rgba, &mut out16, &d16).unwrap();
            for chunk in out16.chunks_exact(2) {
                let v = f16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]]));
                assert!(v.is_finite(), "16f non-finite {v} for alpha={a:e}");
            }
        }

        // Premultiplied HDR pixel whose un-premultiply overflows f32::MAX must also clamp/zero
        // cleanly to finite values in 32f and 16f.
        let extreme = [f32::MAX, -f32::MAX, 1e30, f32::MIN_POSITIVE];
        let mut out32 = vec![0u8; 16];
        write_rgba(&extreme, &mut out32, &desc(1, 1, 16, Layout::Bgra32f, Alpha::Straight)).unwrap();
        for chunk in out32.chunks_exact(4) {
            assert!(f32::from_ne_bytes(chunk.try_into().unwrap()).is_finite());
        }
        let mut out16 = vec![0u8; 8];
        write_rgba(&extreme, &mut out16, &desc(1, 1, 8, Layout::Bgra16f, Alpha::Straight)).unwrap();
        for chunk in out16.chunks_exact(2) {
            assert!(f16_to_f32(u16::from_ne_bytes([chunk[0], chunk[1]])).is_finite());
        }
    }
}
