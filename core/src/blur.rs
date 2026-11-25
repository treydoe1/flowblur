//! Reconstruction-filter motion blur (McGuire et al. 2012, Guertin et al. 2014 refinements) with an
//! energy-conserving, depth-free scatter-as-gather weighting that models a box shutter.

use crate::Quality;
use crate::field::Analysis;
use crate::par;
use crate::pool::PooledVec2;

pub(crate) const TILE: usize = 16;

/// Guided (joint-bilateral) upsampling of the analysis velocity to one full-resolution pixel.
pub(crate) struct Upsampler<'a> {
    pub an: &'a Analysis,
    pub luma: &'a [f32],
    pub w: usize,
    sx: f32,
    sy: f32,
    mx: f32,
    my: f32,
    inv_sig2: f32,
}

impl<'a> Upsampler<'a> {
    pub fn new(an: &'a Analysis, luma: &'a [f32], w: usize, h: usize) -> Self {
        let sig = 0.25 * crate::field::NORM_STD;
        Upsampler {
            an,
            luma,
            w,
            sx: an.wa as f32 / w as f32,
            sy: an.ha as f32 / h as f32,
            mx: w as f32 / an.wa as f32,
            my: h as f32 / an.ha as f32,
            inv_sig2: 1.0 / (sig * sig),
        }
    }

    #[inline(always)]
    fn vel_taps(&self, x0: usize, x1: usize, fx: f32, r0: usize, r1: usize, fy: f32, yl: f32) -> [f32; 2] {
        let an = self.an;
        let i00 = r0 + x0;
        let i10 = r0 + x1;
        let i01 = r1 + x0;
        let i11 = r1 + x1;

        let d0 = yl - an.ya.d[i00];
        let d1 = yl - an.ya.d[i10];
        let d2 = yl - an.ya.d[i01];
        let d3 = yl - an.ya.d[i11];
        let e0 = (-(d0 * d0) * self.inv_sig2).exp();
        let e1 = (-(d1 * d1) * self.inv_sig2).exp();
        let e2 = (-(d2 * d2) * self.inv_sig2).exp();
        let e3 = (-(d3 * d3) * self.inv_sig2).exp();

        let omx = 1.0 - fx;
        let omy = 1.0 - fy;
        let b0 = omx * omy;
        let b1 = fx * omy;
        let b2 = omx * fy;
        let b3 = fx * fy;

        let w0 = b0 * e0;
        let w1 = b1 * e1;
        let w2 = b2 * e2;
        let w3 = b3 * e3;

        let u0 = an.vu.d[i00];
        let u1 = an.vu.d[i10];
        let u2 = an.vu.d[i01];
        let u3 = an.vu.d[i11];

        let v0 = an.vv.d[i00];
        let v1 = an.vv.d[i10];
        let v2 = an.vv.d[i01];
        let v3 = an.vv.d[i11];

        let sw = ((w0 + w1) + w2) + w3;
        let (u, v) = if sw > 1e-4 {
            let su = ((w0 * u0 + w1 * u1) + w2 * u2) + w3 * u3;
            let sv = ((w0 * v0 + w1 * v1) + w2 * v2) + w3 * v3;
            (su / sw, sv / sw)
        } else {
            let bu = ((b0 * u0 + b1 * u1) + b2 * u2) + b3 * u3;
            let bv = ((b0 * v0 + b1 * v1) + b2 * v2) + b3 * v3;
            (bu, bv)
        };
        [u * self.mx, v * self.my]
    }

    /// Velocity at full-res pixel (x, y), in full-res pixels per frame.
    #[inline(always)]
    pub fn vel(&self, x: usize, y: usize) -> [f32; 2] {
        let an = self.an;
        let ax = ((x as f32 + 0.5) * self.sx - 0.5).clamp(0.0, (an.wa - 1) as f32);
        let ay = ((y as f32 + 0.5) * self.sy - 0.5).clamp(0.0, (an.ha - 1) as f32);
        let x0 = ax as usize;
        let y0 = ay as usize;
        let x1 = (x0 + 1).min(an.wa - 1);
        let y1 = (y0 + 1).min(an.ha - 1);
        let fx = ax - x0 as f32;
        let fy = ay - y0 as f32;
        let yl = self.luma[y * self.w + x] * an.norm.0 + an.norm.1;
        self.vel_taps(x0, x1, fx, y0 * an.wa, y1 * an.wa, fy, yl)
    }
}

/// Compact per-pixel half-extent field `[hx, hy]` (8 bytes/px, pooled to avoid page faults).
pub(crate) struct HField {
    pub w: usize,
    pub h: usize,
    pub hv: PooledVec2,
    pub tw: usize,
    pub th: usize,
    pub tmax: Vec<[f32; 2]>,
    pub max_len: f32,
}

#[inline(always)]
fn len2(v: [f32; 2]) -> f32 {
    v[0] * v[0] + v[1] * v[1]
}

/// Full-resolution half-extent field h = v * blur/2 (clamped to `hmax`) and its per-tile maxima.
pub(crate) fn build_hfield(up: &Upsampler, w: usize, h: usize, mul: f32, hmax: f32) -> HField {
    let tw = w.div_ceil(TILE);
    let th = h.div_ceil(TILE);
    let an = up.an;
    let max_ax = (an.wa - 1) as f32;
    let x_taps: Vec<(usize, usize, f32)> = (0..w)
        .map(|x| {
            let ax = ((x as f32 + 0.5) * up.sx - 0.5).clamp(0.0, max_ax);
            let x0 = ax as usize;
            let x1 = (x0 + 1).min(an.wa - 1);
            (x0, x1, ax - x0 as f32)
        })
        .collect();
    let max_ay = (an.ha - 1) as f32;
    let (norm_g, norm_o) = an.norm;

    let mut hv = PooledVec2::take_dirty(w * h);
    let mut tmax = vec![[0.0f32; 2]; tw * th];
    par::for_rows_mut2(&mut hv.0, &mut tmax, w * TILE, tw, 2, |ty0, hchunk, tchunk| {
        for (dty, (strip, trow)) in hchunk.chunks_mut(w * TILE).zip(tchunk.chunks_exact_mut(tw)).enumerate() {
            let ty = ty0 + dty;
            let rows = strip.len() / w;
            for r in 0..rows {
                let y = ty * TILE + r;
                let ay = ((y as f32 + 0.5) * up.sy - 0.5).clamp(0.0, max_ay);
                let y0 = ay as usize;
                let y1 = (y0 + 1).min(an.ha - 1);
                let fy = ay - y0 as f32;
                let r0 = y0 * an.wa;
                let r1 = y1 * an.wa;
                let lrow = &up.luma[y * w..(y + 1) * w];
                let hrow = &mut strip[r * w..(r + 1) * w];
                for x in 0..w {
                    let (x0, x1, fx) = x_taps[x];
                    let yl = lrow[x] * norm_g + norm_o;
                    let v = up.vel_taps(x0, x1, fx, r0, r1, fy, yl);
                    let mut hh = [v[0] * mul, v[1] * mul];
                    let l = len2(hh).sqrt();
                    if l > hmax {
                        let k = hmax / l;
                        hh = [hh[0] * k, hh[1] * k];
                    }
                    hrow[x] = hh;
                    let t = &mut trow[x / TILE];
                    if len2(hh) > len2(*t) {
                        *t = hh;
                    }
                }
            }
        }
    });
    let max_len = tmax.iter().map(|&t| len2(t)).fold(0.0f32, f32::max).sqrt();
    HField { w, h, hv, tw, th, tmax, max_len }
}

/// Builds an HField from an explicit half-extent field (tests).
#[cfg(test)]
pub(crate) fn hfield_from(w: usize, h: usize, hv_raw: Vec<[f32; 2]>) -> HField {
    let tw = w.div_ceil(TILE);
    let th = h.div_ceil(TILE);
    let mut tmax = vec![[0.0f32; 2]; tw * th];
    for y in 0..h {
        for x in 0..w {
            let hh = hv_raw[y * w + x];
            let t = &mut tmax[(y / TILE) * tw + x / TILE];
            if len2(hh) > len2(*t) {
                *t = hh;
            }
        }
    }
    let max_len = tmax.iter().map(|&t| len2(t)).fold(0.0f32, f32::max).sqrt();
    HField { w, h, hv: PooledVec2(hv_raw), tw, th, tmax, max_len }
}

/// Neighbour-max: per tile, the longest tile-max vector among tiles whose streak can reach it.
/// Returned as `[nx, ny, ln]` (unit direction + length).
fn neighbor_max(f: &HField) -> Vec<[f32; 3]> {
    let (tw, th) = (f.tw as isize, f.th as isize);
    let r = (f.max_len / TILE as f32).ceil() as isize + 1;
    let k = TILE as f32;
    let lim2 = (1.5 * k) * (1.5 * k);
    let mut nm = vec![[0.0f32; 3]; f.tw * f.th];
    par::for_rows_mut(&mut nm, f.tw, 4, |ty0, chunk| {
        for (dty, row) in chunk.chunks_exact_mut(f.tw).enumerate() {
            let ty = (ty0 + dty) as isize;
            for (tx_u, slot) in row.iter_mut().enumerate() {
                let tx = tx_u as isize;
                let t = (ty0 + dty) * f.tw + tx_u;
                let mut best = f.tmax[t];
                let mut bl = len2(best);
                let y_lo = (ty - r).max(0);
                let y_hi = (ty + r).min(th - 1);
                let x_lo = (tx - r).max(0);
                let x_hi = (tx + r).min(tw - 1);
                for yy in y_lo..=y_hi {
                    let dy = yy - ty;
                    let row_off = (yy * tw) as usize;
                    for xx in x_lo..=x_hi {
                        let hv = f.tmax[row_off + xx as usize];
                        let l2 = len2(hv);
                        if l2 <= bl {
                            continue;
                        }
                        let dx = xx - tx;
                        let d = [-(dx as f32) * k, -(dy as f32) * k];
                        let p = ((d[0] * hv[0] + d[1] * hv[1]) / l2).clamp(-1.0, 1.0);
                        let e = [d[0] - p * hv[0], d[1] - p * hv[1]];
                        if len2(e) <= lim2 {
                            best = hv;
                            bl = l2;
                        }
                    }
                }
                let l = bl.sqrt();
                *slot = if l > 0.0 { [best[0] / l, best[1] / l, l] } else { [0.0, 0.0, 0.0] };
            }
        }
    });
    nm
}

#[inline(always)]
fn hash(x: u32, y: u32, salt: u32) -> u32 {
    let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ salt.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x297a_2d39);
    h ^= h >> 15;
    h
}

#[inline(always)]
fn unit(h: u32) -> f32 {
    (h >> 8) as f32 * (1.0 / 16_777_216.0)
}

#[inline(always)]
fn sat(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

pub(crate) struct QualityCfg {
    spp: f32,
    cap: usize,
    min: usize,
}

pub(crate) fn qcfg(q: Quality) -> QualityCfg {
    match q {
        Quality::Draft => QualityCfg { spp: 0.5, cap: 16, min: 3 },
        Quality::Normal => QualityCfg { spp: 1.0, cap: 48, min: 4 },
        Quality::High => QualityCfg { spp: 1.5, cap: 128, min: 5 },
    }
}

#[inline(always)]
fn coverage(dx: f32, dy: f32, hx: f32, hy: f32, l: f32, inv_l: f32, inv_soft: f32) -> f32 {
    if l < 0.25 {
        return 0.0;
    }
    let along = (dx * hx + dy * hy).abs() * inv_l;
    let perp = (dx * hy - dy * hx).abs() * inv_l;
    sat(1.75 + 0.15 * along - perp) * sat((l - along) * inv_soft + 0.5)
}

/// Scalar sampling line (used on non-aarch64 and as test reference on aarch64).
#[cfg(any(test, not(target_arch = "aarch64")))]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn line_scalar(
    f: &HField,
    col: &[f32],
    qx: f32,
    qy: f32,
    hq: [f32; 2],
    lq: f32,
    dir: [f32; 2],
    half: f32,
    n: usize,
    jit: f32,
    acc: &mut [f32; 4],
    wsum: &mut f32,
) {
    let dsp = 2.0 * half / n as f32;
    let inv_soft = 1.0 / dsp.max(1.0);
    let (wm, hm) = ((f.w - 1) as f32, (f.h - 1) as f32);
    let w = f.w;
    let lq_valid = lq >= 0.25;
    let inv_lq = if lq_valid { 1.0 / lq } else { 0.0 };
    let half_dsp = 0.5 * dsp;
    let half_dsp_inv_lq = half_dsp * inv_lq;
    let hv = &f.hv.0;

    for i in 0..n {
        let t = -half + (i as f32 + 0.5 + jit) * dsp;
        let sx = (qx + t * dir[0]).round().clamp(0.0, wm);
        let sy = (qy + t * dir[1]).round().clamp(0.0, hm);
        let si = sy as usize * w + sx as usize;
        let hx = hv[si];
        let lx = len2(hx).sqrt();
        let inv_lx = if lx >= 0.25 { 1.0 / lx } else { 0.0 };
        let dx = qx - sx;
        let dy = qy - sy;
        let fa = coverage(dx, dy, hx[0], hx[1], lx, inv_lx, inv_soft);
        let fb = if lq_valid { coverage(dx, dy, hq[0], hq[1], lq, inv_lq, inv_soft) } else { 0.0 };
        let wgt = if fa > 0.0 && (fb <= 0.0 || lx >= lq) {
            fa * half_dsp * inv_lx
        } else if fb > 0.0 {
            fb * half_dsp_inv_lq
        } else {
            continue;
        };
        let c = &col[si * 4..si * 4 + 4];
        acc[0] += wgt * c[0];
        acc[1] += wgt * c[1];
        acc[2] += wgt * c[2];
        acc[3] += wgt * c[3];
        *wsum += wgt;
    }
}

/// 4-sample NEON-vectorized sampling line for aarch64, reading compact `[f32; 2]` HField.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn line_neon(
    f: &HField,
    col: &[f32],
    qx: f32,
    qy: f32,
    hq: [f32; 2],
    lq: f32,
    dir: [f32; 2],
    half: f32,
    n: usize,
    jit: f32,
    acc: &mut [f32; 4],
    wsum: &mut f32,
) {
    use std::arch::aarch64::*;
    unsafe {
        let dsp = 2.0 * half / n as f32;
        let inv_soft = 1.0 / dsp.max(1.0);
        let (wm, hm) = ((f.w - 1) as f32, (f.h - 1) as f32);
        let w = f.w;
        let h = f.h;
        let npix = w.checked_mul(h).expect("HField dimensions overflow");
        assert!(w > 0 && h > 0 && npix <= u32::MAX as usize && f.hv.0.len() >= npix && col.len() >= npix * 4);
        let lq_valid = lq >= 0.25;
        let inv_lq = if lq_valid { 1.0 / lq } else { 0.0 };
        let half_dsp = 0.5 * dsp;
        let half_dsp_inv_lq = half_dsp * inv_lq;

        let vqx = vdupq_n_f32(qx);
        let vqy = vdupq_n_f32(qy);
        let vdirx = vdupq_n_f32(dir[0]);
        let vdiry = vdupq_n_f32(dir[1]);
        let vzero = vdupq_n_f32(0.0);
        let vone = vdupq_n_f32(1.0);
        let vwm = vdupq_n_f32(wm);
        let vhm = vdupq_n_f32(hm);
        let vw_u32 = vdupq_n_u32(w as u32);
        let vwm1_u32 = vdupq_n_u32((w - 1) as u32);
        let vhm1_u32 = vdupq_n_u32((h - 1) as u32);
        let v175 = vdupq_n_f32(1.75);
        let v015 = vdupq_n_f32(0.15);
        let v05 = vdupq_n_f32(0.5);
        let v025 = vdupq_n_f32(0.25);
        let vinv_soft = vdupq_n_f32(inv_soft);
        let vdsp = vdupq_n_f32(dsp);
        let vhalf_dsp = vdupq_n_f32(half_dsp);
        let vhqx = vdupq_n_f32(hq[0]);
        let vhqy = vdupq_n_f32(hq[1]);
        let vlq = vdupq_n_f32(lq);
        let vinv_lq = vdupq_n_f32(inv_lq);
        let vhalf_dsp_inv_lq = vdupq_n_f32(half_dsp_inv_lq);

        let mut vacc0 = vld1q_f32(acc.as_ptr());
        let mut vacc1 = vzero;
        let mut ws = *wsum;
        let hv_ptr = f.hv.0.as_ptr();
        let col_ptr = col.as_ptr();

        let vneg_half = vdupq_n_f32(-half);
        let v05jit = vdupq_n_f32(0.5 + jit);
        let vfour = vdupq_n_f32(4.0);
        let idx0: [f32; 4] = [0.0, 1.0, 2.0, 3.0];
        let mut vi = vld1q_f32(idx0.as_ptr());

        let mut i = 0usize;
        while i + 4 <= n {
            let vt = vaddq_f32(vneg_half, vmulq_f32(vaddq_f32(vi, v05jit), vdsp));
            let vsx = vminq_f32(vmaxq_f32(vrndaq_f32(vaddq_f32(vqx, vmulq_f32(vt, vdirx))), vzero), vwm);
            let vsy = vminq_f32(vmaxq_f32(vrndaq_f32(vaddq_f32(vqy, vmulq_f32(vt, vdiry))), vzero), vhm);
            vi = vaddq_f32(vi, vfour);

            let vix = vminq_u32(vcvtq_u32_f32(vsx), vwm1_u32);
            let viy = vminq_u32(vcvtq_u32_f32(vsy), vhm1_u32);
            let vsi = vmlaq_u32(vix, viy, vw_u32);
            let si0 = vgetq_lane_u32::<0>(vsi) as usize;
            let si1 = vgetq_lane_u32::<1>(vsi) as usize;
            let si2 = vgetq_lane_u32::<2>(vsi) as usize;
            let si3 = vgetq_lane_u32::<3>(vsi) as usize;

            // Load 4 compact [hx, hy] entries (8 bytes each) and deinterleave into vhx, vhy.
            let h01 = vcombine_f32(vld1_f32(hv_ptr.add(si0) as *const f32), vld1_f32(hv_ptr.add(si1) as *const f32));
            let h23 = vcombine_f32(vld1_f32(hv_ptr.add(si2) as *const f32), vld1_f32(hv_ptr.add(si3) as *const f32));
            let vhx = vuzp1q_f32(h01, h23);
            let vhy = vuzp2q_f32(h01, h23);

            let vlx = vsqrtq_f32(vaddq_f32(vmulq_f32(vhx, vhx), vmulq_f32(vhy, vhy)));
            let valid_a = vcgeq_f32(vlx, v025);
            let vinvlx = vbslq_f32(valid_a, vdivq_f32(vone, vlx), vzero);

            let vdx = vsubq_f32(vqx, vsx);
            let vdy = vsubq_f32(vqy, vsy);

            // Coverage fa (x's streak covers q)
            let along_a = vmulq_f32(vabsq_f32(vaddq_f32(vmulq_f32(vdx, vhx), vmulq_f32(vdy, vhy))), vinvlx);
            let perp_a = vmulq_f32(vabsq_f32(vsubq_f32(vmulq_f32(vdx, vhy), vmulq_f32(vdy, vhx))), vinvlx);
            let cap_a = vminq_f32(vmaxq_f32(vsubq_f32(vaddq_f32(v175, vmulq_f32(v015, along_a)), perp_a), vzero), vone);
            let end_a = vminq_f32(vmaxq_f32(vaddq_f32(vmulq_f32(vsubq_f32(vlx, along_a), vinv_soft), v05), vzero), vone);
            let vfa = vbslq_f32(valid_a, vmulq_f32(cap_a, end_a), vzero);

            // Coverage fb (q's streak covers x)
            let vfb = if lq_valid {
                let along_b = vmulq_f32(vabsq_f32(vaddq_f32(vmulq_f32(vdx, vhqx), vmulq_f32(vdy, vhqy))), vinv_lq);
                let perp_b = vmulq_f32(vabsq_f32(vsubq_f32(vmulq_f32(vdx, vhqy), vmulq_f32(vdy, vhqx))), vinv_lq);
                let cap_b = vminq_f32(vmaxq_f32(vsubq_f32(vaddq_f32(v175, vmulq_f32(v015, along_b)), perp_b), vzero), vone);
                let end_b = vminq_f32(vmaxq_f32(vaddq_f32(vmulq_f32(vsubq_f32(vlq, along_b), vinv_soft), v05), vzero), vone);
                vmulq_f32(cap_b, end_b)
            } else {
                vzero
            };

            let fa_pos = vcgtq_f32(vfa, vzero);
            let fb_pos = vcgtq_f32(vfb, vzero);
            let use_a = vandq_u32(fa_pos, vorrq_u32(vmvnq_u32(fb_pos), vcgeq_f32(vlx, vlq)));
            let w_a = vmulq_f32(vmulq_f32(vfa, vhalf_dsp), vinvlx);
            let w_b = vbslq_f32(fb_pos, vmulq_f32(vfb, vhalf_dsp_inv_lq), vzero);
            let vwgt = vbslq_f32(use_a, w_a, w_b);

            if vmaxvq_f32(vwgt) > 0.0 {
                let c0 = vld1q_f32(col_ptr.add(si0 * 4));
                let c1 = vld1q_f32(col_ptr.add(si1 * 4));
                let c2 = vld1q_f32(col_ptr.add(si2 * 4));
                let c3 = vld1q_f32(col_ptr.add(si3 * 4));
                vacc0 = vfmaq_laneq_f32::<0>(vacc0, c0, vwgt);
                vacc1 = vfmaq_laneq_f32::<1>(vacc1, c1, vwgt);
                vacc0 = vfmaq_laneq_f32::<2>(vacc0, c2, vwgt);
                vacc1 = vfmaq_laneq_f32::<3>(vacc1, c3, vwgt);
                ws += vaddvq_f32(vwgt);
            }

            i += 4;
        }

        // Scalar tail for remaining 0..3 samples.
        vst1q_f32(acc.as_mut_ptr(), vaddq_f32(vacc0, vacc1));
        *wsum = ws;
        while i < n {
            let t = -half + (i as f32 + 0.5 + jit) * dsp;
            let sx = (qx + t * dir[0]).round().clamp(0.0, wm);
            let sy = (qy + t * dir[1]).round().clamp(0.0, hm);
            let si = (sy as usize).min(h - 1) * w + (sx as usize).min(w - 1);
            let hx = f.hv.0[si];
            let lx = len2(hx).sqrt();
            let inv_lx = if lx >= 0.25 { 1.0 / lx } else { 0.0 };
            let dx = qx - sx;
            let dy = qy - sy;
            let fa = coverage(dx, dy, hx[0], hx[1], lx, inv_lx, inv_soft);
            let fb = if lq_valid { coverage(dx, dy, hq[0], hq[1], lq, inv_lq, inv_soft) } else { 0.0 };
            let wgt = if fa > 0.0 && (fb <= 0.0 || lx >= lq) {
                fa * half_dsp * inv_lx
            } else if fb > 0.0 {
                fb * half_dsp_inv_lq
            } else {
                i += 1;
                continue;
            };
            let c = &col[si * 4..si * 4 + 4];
            acc[0] += wgt * c[0];
            acc[1] += wgt * c[1];
            acc[2] += wgt * c[2];
            acc[3] += wgt * c[3];
            *wsum += wgt;
            i += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn line(
    f: &HField,
    col: &[f32],
    qx: f32,
    qy: f32,
    hq: [f32; 2],
    lq: f32,
    dir: [f32; 2],
    half: f32,
    n: usize,
    jit: f32,
    acc: &mut [f32; 4],
    wsum: &mut f32,
) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        line_neon(f, col, qx, qy, hq, lq, dir, half, n, jit, acc, wsum);
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        line_scalar(f, col, qx, qy, hq, lq, dir, half, n, jit, acc, wsum);
    }
}

/// Gathers the blurred image. `col` is the colour the averaging happens in (linear or gamma);
/// `orig` the untouched input (for bit-exact copies). `encode` maps a blurred `col` pixel back.
pub(crate) fn gather<E>(f: &HField, col: &[f32], orig: &[f32], q: Quality, out: &mut [f32], encode: E)
where
    E: Fn(&mut [f32; 4]) + Sync,
{
    let nm = neighbor_max(f);
    let qc = qcfg(q);
    let w = f.w;
    let h = f.h;
    let wm = (w - 1) as f32;
    let hm = (h - 1) as f32;
    let jr = (TILE / 4) as f32;
    let hv = &f.hv.0;
    par::for_rows_mut(out, w * 4, 8, |y0, chunk| {
        for (dy, orow) in chunk.chunks_exact_mut(w * 4).enumerate() {
            let y = y0 + dy;
            let yf = y as f32;
            let row_base = y * w;
            for x in 0..w {
                let pi = row_base + x;
                let xf = x as f32;
                let hj = hash(x as u32, y as u32, 1);
                let jx = (xf + (unit(hj) * 2.0 - 1.0) * jr).clamp(0.0, wm) as usize;
                let jy = (yf + (unit(hash(x as u32, y as u32, 2)) * 2.0 - 1.0) * jr).clamp(0.0, hm) as usize;
                let nm_e = nm[(jy / TILE) * f.tw + jx / TILE];
                let hq = hv[pi];
                let lq = len2(hq).sqrt();
                let (dn, ln) = if lq > nm_e[2] {
                    ([hq[0] / lq, hq[1] / lq], lq)
                } else {
                    ([nm_e[0], nm_e[1]], nm_e[2])
                };
                if ln < 0.5 {
                    orow[x * 4..x * 4 + 4].copy_from_slice(&orig[pi * 4..pi * 4 + 4]);
                    continue;
                }
                let jit = unit(hash(x as u32, y as u32, 3)) - 0.5;
                let mut acc = [0.0f32; 4];
                let mut wsum = 0.0f32;
                let second = lq >= 0.5 && (hq[0] * dn[0] + hq[1] * dn[1]) / lq < 0.985;
                let nn = ((qc.spp * 2.0 * ln).ceil() as usize).clamp(qc.min, qc.cap);
                line(f, col, xf, yf, hq, lq, dn, ln, nn, jit, &mut acc, &mut wsum);
                if second {
                    let dq = [hq[0] / lq, hq[1] / lq];
                    let nq = ((qc.spp * 2.0 * lq).ceil() as usize).clamp(qc.min, qc.cap);
                    line(f, col, xf, yf, hq, lq, dq, lq, nq, -jit, &mut acc, &mut wsum);
                }
                if wsum < 1e-6 {
                    orow[x * 4..x * 4 + 4].copy_from_slice(&orig[pi * 4..pi * 4 + 4]);
                    continue;
                }
                let rest = (1.0 - wsum).max(0.0);
                let norm = 1.0 / wsum.max(1.0);
                let c = &col[pi * 4..pi * 4 + 4];
                let mut o = [
                    (acc[0] + rest * c[0]) * norm,
                    (acc[1] + rest * c[1]) * norm,
                    (acc[2] + rest * c[2]) * norm,
                    (acc[3] + rest * c[3]) * norm,
                ];
                encode(&mut o);
                orow[x * 4..x * 4 + 4].copy_from_slice(&o);
            }
        }
    });
}

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_tests {
    use super::*;

    #[test]
    fn line_neon_matches_scalar() {
        let (w, h) = (32, 24);
        let mut hv_raw = Vec::with_capacity(w * h);
        let mut col = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let vx = ((x as f32 - 10.0) * 0.4).sin() * 6.0;
                let vy = ((y as f32 - 8.0) * 0.3).cos() * 4.0;
                hv_raw.push([vx, vy]);
                col.push((x as f32) / (w as f32));
                col.push((y as f32) / (h as f32));
                col.push(((x + y) % 17) as f32 / 16.0);
                col.push(1.0);
            }
        }
        let hf = hfield_from(w, h, hv_raw);
        for n in [3, 4, 5, 7, 16, 25] {
            for &(qx, qy) in &[(0usize, 0usize), (15, 11), (31, 23)] {
                let hq = hf.hv.0[qy * w + qx];
                let lq = len2(hq).sqrt();
                let mut acc_s = [0.0f32; 4];
                let mut ws_s = 0.0f32;
                let mut acc_n = [0.0f32; 4];
                let mut ws_n = 0.0f32;
                let dir = [0.6f32, 0.8f32];
                line_scalar(&hf, &col, qx as f32, qy as f32, hq, lq, dir, 6.0, n, 0.17, &mut acc_s, &mut ws_s);
                unsafe {
                    line_neon(&hf, &col, qx as f32, qy as f32, hq, lq, dir, 6.0, n, 0.17, &mut acc_n, &mut ws_n);
                }
                assert!((ws_s - ws_n).abs() < 1e-5, "n={n} ws {ws_s} vs {ws_n}");
                for k in 0..4 {
                    assert!((acc_s[k] - acc_n[k]).abs() < 1e-5, "n={n} k={k} {} vs {}", acc_s[k], acc_n[k]);
                }
            }
        }
    }

    #[test]
    fn line_neon_small_images_and_nonfinite_inputs_safe() {
        for &(w, h) in &[(1usize, 1usize), (2, 2), (3, 3), (7, 5), (15, 3)] {
            let hf = hfield_from(w, h, vec![[3.0, -2.0]; w * h]);
            let col = vec![0.5f32; w * h * 4];
            for dir in [[1.0f32, 0.0], [f32::NAN, 0.0], [f32::INFINITY, f32::NEG_INFINITY], [1e30, -1e30]] {
                let mut acc = [0.0f32; 4];
                let mut ws = 0.0f32;
                unsafe {
                    line_neon(&hf, &col, 0.0, 0.0, [3.0, -2.0], 3.6, dir, 5.0, 9, 0.1, &mut acc, &mut ws);
                }
            }
        }
    }
}
