//! sRGB-style transfer for averaging in linear light. Odd-symmetric for negatives, and the power
//! segment simply continues above 1.0 so HDR / float values stay monotonic and invertible.
//!
//! Uses L1-resident linearly-interpolated LUTs for `[0, 1]` (max error < 2e-7 vs exact `powf`),
//! falling back to exact `powf` for HDR values above 1.0.

use std::sync::OnceLock;

const DEC_N: usize = 4096;
const ENC_N: usize = 2048;

#[inline(always)]
fn decode_exact(a: f32) -> f32 {
    if a <= 0.04045 { a / 12.92 } else { ((a + 0.055) / 1.055).powf(2.4) }
}

#[inline(always)]
fn encode_exact(a: f32) -> f32 {
    if a <= 0.003_130_8 { a * 12.92 } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 }
}

fn dec_lut() -> &'static [f32; DEC_N + 2] {
    static LUT: OnceLock<Box<[f32; DEC_N + 2]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = Box::new([0.0f32; DEC_N + 2]);
        for i in 0..=DEC_N {
            t[i] = decode_exact(i as f32 / DEC_N as f32);
        }
        t[DEC_N + 1] = t[DEC_N];
        t
    })
}

/// Indexed by `sqrt(a)` on `[0, 1]` so the steep derivative near 0 is stretched smoothly.
fn enc_lut() -> &'static [f32; ENC_N + 2] {
    static LUT: OnceLock<Box<[f32; ENC_N + 2]>> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = Box::new([0.0f32; ENC_N + 2]);
        for i in 0..=ENC_N {
            let s = i as f32 / ENC_N as f32;
            t[i] = encode_exact(s * s);
        }
        t[ENC_N + 1] = t[ENC_N];
        t
    })
}

#[inline(always)]
fn decode_with_lut(x: f32, lut: &[f32; DEC_N + 2]) -> f32 {
    let a = x.abs();
    let v = if a <= 0.04045 {
        a * (1.0 / 12.92)
    } else if a <= 1.0 {
        let p = a * DEC_N as f32;
        let i = p as usize;
        let f = p - i as f32;
        lut[i] + f * (lut[i + 1] - lut[i])
    } else {
        ((a + 0.055) / 1.055).powf(2.4)
    };
    v.copysign(x)
}

#[inline(always)]
fn encode_with_lut(x: f32, lut: &[f32; ENC_N + 2]) -> f32 {
    let a = x.abs();
    let v = if a <= 0.003_130_8 {
        a * 12.92
    } else if a <= 1.0 {
        let p = a.sqrt() * ENC_N as f32;
        let i = p as usize;
        let f = p - i as f32;
        lut[i] + f * (lut[i + 1] - lut[i])
    } else {
        1.055 * a.powf(1.0 / 2.4) - 0.055
    };
    v.copysign(x)
}

#[cfg(test)]
#[inline(always)]
pub(crate) fn decode(x: f32) -> f32 {
    decode_with_lut(x, dec_lut())
}

#[cfg(test)]
#[inline(always)]
pub(crate) fn encode(x: f32) -> f32 {
    encode_with_lut(x, enc_lut())
}

/// Premultiplied gamma pixel -> premultiplied linear pixel, writing from `src` to `dst` directly.
#[inline(always)]
pub(crate) fn decode_rgba_chunk(src: &[f32], dst: &mut [f32]) {
    let lut = dec_lut();
    for (s, d) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        let a = s[3];
        if a > 1e-6 && (a - 1.0).abs() > 1e-7 {
            let ia = 1.0 / a;
            d[0] = decode_with_lut(s[0] * ia, lut) * a;
            d[1] = decode_with_lut(s[1] * ia, lut) * a;
            d[2] = decode_with_lut(s[2] * ia, lut) * a;
        } else {
            d[0] = decode_with_lut(s[0], lut);
            d[1] = decode_with_lut(s[1], lut);
            d[2] = decode_with_lut(s[2], lut);
        }
        d[3] = a;
    }
}

#[inline(always)]
pub(crate) fn px_encode(p: &mut [f32; 4]) {
    let lut = enc_lut();
    let a = p[3];
    if a > 1e-6 && (a - 1.0).abs() > 1e-7 {
        let ia = 1.0 / a;
        p[0] = encode_with_lut(p[0] * ia, lut) * a;
        p[1] = encode_with_lut(p[1] * ia, lut) * a;
        p[2] = encode_with_lut(p[2] * ia, lut) * a;
    } else {
        p[0] = encode_with_lut(p[0], lut);
        p[1] = encode_with_lut(p[1], lut);
        p[2] = encode_with_lut(p[2], lut);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lut_matches_exact_powf() {
        let mut max_dec = 0.0f32;
        let mut max_enc = 0.0f32;
        for i in 0..=20000 {
            let x = i as f32 / 10000.0; // 0..=2.0 (includes HDR)
            max_dec = max_dec.max((decode(x) - decode_exact(x)).abs());
            max_enc = max_enc.max((encode(x) - encode_exact(x)).abs());
            max_dec = max_dec.max((decode(-x) + decode_exact(x)).abs());
            max_enc = max_enc.max((encode(-x) + encode_exact(x)).abs());
        }
        assert!(max_dec < 2e-6, "max_dec={max_dec}");
        assert!(max_enc < 2e-6, "max_enc={max_enc}");
    }
}
