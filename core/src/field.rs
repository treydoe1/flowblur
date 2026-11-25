//! Turns raw DIS flows into one blur-ready velocity field at analysis resolution.
//!
//! Steps: exposure normalisation of all lumas, DIS per neighbour, 3x3 median outlier removal,
//! per-pixel confidence from a contrast-relative photometric residual, scene-cut rejection,
//! confidence-weighted central difference, confidence-weighted edge-aware smoothing (normalised
//! domain-transform filter guided by the current luma), confidence gate and dead zone.

use crate::dis::{DisCfg, Level, build_pyramid, dis_flow, levels_for, precompute_tmpls};
use crate::par;
use crate::plane::{Plane, det_sum, gaussian_blur, gradients, median3, resample_area};
use crate::{Luma, Neighbor, Params, Quality};

/// Target standard deviation of the normalised analysis luma. All luma-unit constants below are
/// relative to this, which makes them independent of the footage's exposure and contrast.
pub(crate) const NORM_STD: f32 = 0.2;

/// Smallest `Neighbor::frame_distance` the analysis accepts; smaller values are raised to it.
pub const MIN_FRAME_DISTANCE: f32 = 1.0 / 64.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct SensMap {
    pub sigma_s: f32,
    pub c_lo: f32,
    pub c_hi: f32,
    pub dead: f32,
}

pub(crate) fn sens_map(sens: f32) -> SensMap {
    let t = (sens / 100.0).clamp(0.0, 1.0);
    let u = 1.0 - t;
    let c_lo = 0.6 * u;
    SensMap { sigma_s: 2.0 + 28.0 * u.powf(1.5), c_lo, c_hi: c_lo + 0.03 + 0.25 * u, dead: 0.03 + 1.2 * u * u }
}

pub(crate) fn analysis_long_edge(q: Quality) -> usize {
    match q {
        Quality::Draft => 384,
        Quality::Normal => 640,
        Quality::High => 1024,
    }
}

fn dis_cfg(q: Quality) -> DisCfg {
    match q {
        Quality::Draft => DisCfg { stride: 4, iters: 12 },
        Quality::Normal => DisCfg { stride: 4, iters: 16 },
        Quality::High => DisCfg { stride: 3, iters: 24 },
    }
}

pub(crate) struct Analysis {
    pub wa: usize,
    pub ha: usize,
    /// Velocity in analysis pixels per frame interval.
    pub vu: Plane,
    pub vv: Plane,
    /// Normalised current luma at analysis resolution (guide for upsampling).
    pub ya: Plane,
    /// Affine (gain, offset) that normalises raw luma values.
    pub norm: (f32, f32),
    pub used_prev: bool,
    pub used_next: bool,
    /// Fraction of structured pixels explained by each neighbour (diagnostics, tests).
    #[allow(dead_code)]
    pub explained: [f32; 2],
}

#[inline(always)]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Forward-backward check (Sundaram et al. 2010): following the forward flow and then the backward
/// flow must return close to the start.
fn fb_consistency(c: &mut Plane, fu: &Plane, fv: &Plane, bu: &Plane, bv: &Plane) {
    let w = c.w;
    let min_r = (16384 / w.max(1)).max(8);
    par::for_rows_mut(&mut c.d, w, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let yf = y as f32;
            let base = y * w;
            for (x, o) in row.iter_mut().enumerate() {
                let i = base + x;
                let (u, v) = (fu.d[i], fv.d[i]);
                let (px, py) = (x as f32 + u, yf + v);
                let (b0, b1) = (bu.bilinear(px, py), bv.bilinear(px, py));
                let e2 = (u + b0) * (u + b0) + (v + b1) * (v + b1);
                let tol2 = 0.01 * (u * u + v * v + b0 * b0 + b1 * b1) + 0.5;
                if e2 > tol2 {
                    *o *= (-(e2 - tol2) / tol2).exp();
                }
            }
        }
    });
}

/// Gradient pair of a lightly smoothed luma: the matching feature for confidence.
pub(crate) struct Grad {
    x: Plane,
    y: Plane,
}

fn grad_feature_from_smooth(smooth: &Plane) -> Grad {
    let (x, y) = gradients(smooth);
    Grad { x, y }
}

fn grad_feature(p: &Plane) -> Grad {
    grad_feature_from_smooth(&gaussian_blur(p, 0.7))
}

/// Feature residual of `flow` at pixel i, relative to the local gradient energy `lc` (+ noise floor).
#[inline(always)]
fn resid(g0: &Grad, g1: &Grad, lc: &Plane, x: usize, y: usize, u: f32, v: f32) -> f32 {
    let i = y * g0.x.w + x;
    let (px, py) = (x as f32 + u, y as f32 + v);
    let dx = g1.x.bilinear(px, py) - g0.x.d[i];
    let dy = g1.y.bilinear(px, py) - g0.y.d[i];
    (dx * dx + dy * dy).sqrt() / (lc.d[i] + 0.004)
}

/// Contrast-relative photometric confidence of `flow` (cur -> neighbour) plus the structure-weighted
/// fraction of the image it explains.
fn confidence(g0: &Grad, lc: &Plane, sw: &Plane, sw_den: f64, g1: &Grad, fu: &Plane, fv: &Plane) -> (Plane, f32) {
    let (w, h) = (lc.w, lc.h);
    let min_r = (16384 / w.max(1)).max(8);
    let mut c = vec![0.0f32; w * h];
    par::for_rows_mut(&mut c, w, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let base = y * w;
            for (x, o) in row.iter_mut().enumerate() {
                let i = base + x;
                let rn = resid(g0, g1, lc, x, y, fu.d[i], fv.d[i]);
                *o = (-(rn * rn) / (0.7 * 0.7)).exp();
            }
        }
    });
    // Scene-cut measure: how much of the image a *spatially coherent* version of the flow explains.
    let su = gaussian_blur(fu, 4.0);
    let sv = gaussian_blur(fv, 4.0);
    let num = det_sum(h, |y| {
        let base = y * w;
        let mut s = 0.0f64;
        for x in 0..w {
            let i = base + x;
            let r = resid(g0, g1, lc, x, y, su.d[i], sv.d[i]);
            if r < 0.6 {
                s += sw.d[i] as f64;
            }
        }
        s
    });
    (Plane::from_vec(w, h, c), (num / sw_den.max(1e-9)) as f32)
}

/// Cache-blocked transpose into a preallocated output slice `out` (length `w * h`).
fn transpose4_into(d: &[[f32; 4]], out: &mut [[f32; 4]], w: usize, h: usize) {
    const BLK: usize = 16;
    let min_r = (16384 / h.max(1)).max(BLK);
    par::for_rows_mut(out, h, min_r, |x0, chunk| {
        let nx = chunk.len() / h;
        let mut bx = 0usize;
        while bx < nx {
            let bx_end = (bx + BLK).min(nx);
            let mut by = 0usize;
            while by < h {
                let by_end = (by + BLK).min(h);
                for x_off in bx..bx_end {
                    let x = x0 + x_off;
                    let dst_row = x_off * h;
                    for y in by..by_end {
                        chunk[dst_row + y] = d[y * w + x];
                    }
                }
                by = by_end;
            }
            bx = bx_end;
        }
    });
}

#[inline(always)]
fn dt_line_pass(r: &mut [[f32; 4]], dr: &[f32], v: &mut [f32], n: usize, la: f32) {
    if n <= 1 {
        return;
    }
    assert!(r.len() >= n && dr.len() >= n && v.len() >= n);
    for x in 1..n {
        v[x] = (la * dr[x]).exp();
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let rp = r.as_mut_ptr() as *mut f32;
        let mut prev = vld1q_f32(rp);
        for x in 1..n {
            let cur = vld1q_f32(rp.add(x * 4));
            let next = vaddq_f32(cur, vmulq_n_f32(vsubq_f32(prev, cur), v[x]));
            vst1q_f32(rp.add(x * 4), next);
            prev = next;
        }
        prev = vld1q_f32(rp.add((n - 1) * 4));
        for x in (0..n - 1).rev() {
            let cur = vld1q_f32(rp.add(x * 4));
            let next = vaddq_f32(cur, vmulq_n_f32(vsubq_f32(prev, cur), v[x + 1]));
            vst1q_f32(rp.add(x * 4), next);
            prev = next;
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for x in 1..n {
            let (a, b) = r.split_at_mut(x);
            let (p, c) = (&a[x - 1], &mut b[0]);
            let vx = v[x];
            for k in 0..4 {
                c[k] += vx * (p[k] - c[k]);
            }
        }
        for x in (0..n - 1).rev() {
            let (a, b) = r.split_at_mut(x + 1);
            let (c, p) = (&mut a[x], &b[0]);
            let vx = v[x + 1];
            for k in 0..4 {
                c[k] += vx * (p[k] - c[k]);
            }
        }
    }
}

/// Recursive-filter domain transform (Gastal & Oliveira 2011) applied jointly to four interleaved
/// channels, guided by `guide`.
fn dt_filter4(data: &mut Vec<[f32; 4]>, guide: &Plane, sigma_s: f32, sigma_r: f32, iters: usize) {
    let (w, h) = (guide.w, guide.h);
    let ratio = sigma_s / sigma_r;
    // Domain-transform derivatives; the vertical one is stored transposed.
    let mut dh = vec![1.0f32; w * h];
    par::for_rows_mut(&mut dh, w, (16384 / w.max(1)).max(8), |y0, chunk| {
        for (dy, r) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let grow = &guide.d[y * w..(y + 1) * w];
            for x in 1..w {
                r[x] = 1.0 + ratio * (grow[x] - grow[x - 1]).abs();
            }
        }
    });
    let mut dvt = vec![1.0f32; w * h];
    par::for_rows_mut(&mut dvt, h, (16384 / h.max(1)).max(8), |x0, chunk| {
        for (dx, r) in chunk.chunks_exact_mut(h).enumerate() {
            let x = x0 + dx;
            for y in 1..h {
                r[y] = 1.0 + ratio * (guide.d[y * w + x] - guide.d[(y - 1) * w + x]).abs();
            }
        }
    });
    let pass = |d: &mut [[f32; 4]], dd: &[f32], n: usize, la: f32| {
        let min_r = (16384 / n.max(1)).max(8);
        par::for_rows_mut(d, n, min_r, |r0, chunk| {
            let mut v = vec![0.0f32; n];
            for (dr_i, r) in chunk.chunks_exact_mut(n).enumerate() {
                let dr = &dd[(r0 + dr_i) * n..(r0 + dr_i + 1) * n];
                dt_line_pass(r, dr, &mut v, n, la);
            }
        });
    };
    // Reuse a single scratch transpose buffer across all iterations.
    let mut t = vec![[0.0f32; 4]; w * h];
    let nf = iters as f32;
    for k in 0..iters {
        let sig = sigma_s * 3f32.sqrt() * 2f32.powf(nf - k as f32 - 1.0) / (4f32.powf(nf) - 1.0).sqrt();
        let la = -(2f32.sqrt()) / sig; // ln(a)
        pass(data, &dh, w, la);
        transpose4_into(data, &mut t, w, h);
        pass(&mut t, &dvt, h, la);
        transpose4_into(&t, data, h, w);
    }
}

struct NbIn {
    pyr: Vec<Level>,
    g: Grad,
    fd: f32,
    sign: f32,
}

/// Full analysis. Returns None when no neighbour is usable.
pub(crate) fn analyze(cur_luma: Luma, prev: Option<Neighbor>, next: Option<Neighbor>, params: &Params) -> Option<Analysis> {
    let (w, h) = (cur_luma.width, cur_luma.height);
    let le = analysis_long_edge(params.quality);
    let s = (le as f32 / w.max(h) as f32).min(1.0);
    let wa = ((w as f32 * s).round() as usize).clamp(1, w);
    let ha = ((h as f32 * s).round() as usize).clamp(1, h);
    if wa < crate::dis::PS + 2 || ha < crate::dis::PS + 2 {
        return None;
    }
    let raw0 = resample_area(cur_luma.data, w, h, wa, ha);
    // Exposure normalisation from the current frame's statistics, applied to all lumas alike.
    let n = (wa * ha) as f64;
    let mean = det_sum(ha, |y| raw0.d[y * wa..(y + 1) * wa].iter().map(|&v| v as f64).sum()) / n;
    let var = det_sum(ha, |y| raw0.d[y * wa..(y + 1) * wa].iter().map(|&v| (v as f64 - mean).powi(2)).sum()) / n;
    let gain = NORM_STD / (var.sqrt() as f32).max(1e-4);
    let off = 0.5 - mean as f32 * gain;
    let normalize = |mut p: Plane| {
        for v in &mut p.d {
            *v = *v * gain + off;
        }
        p
    };
    let y0 = normalize(raw0);

    let mk_nb = |nb: Option<Neighbor<'_>>, sign: f32| -> Option<(Plane, f32, f32)> {
        let nb = nb?;
        if nb.luma.width != w || nb.luma.height != h || nb.luma.data.len() != w * h || !nb.frame_distance.is_finite() {
            return None;
        }
        let fd = nb.frame_distance.max(MIN_FRAME_DISTANCE);
        Some((normalize(resample_area(nb.luma.data, w, h, wa, ha)), fd, sign))
    };
    let (np, nn) = par::join_split(|| mk_nb(prev, -1.0), || mk_nb(next, 1.0));
    if np.is_none() && nn.is_none() {
        return None;
    }
    let need = |fd: f32| levels_for(wa, ha, 0.15 * wa as f32 * fd.max(1.0));
    let nlev = [np.as_ref(), nn.as_ref()].iter().flatten().map(|n| need(n.1)).max().unwrap();

    let cfg = dis_cfg(params.quality);
    let build_cur_pyr = || {
        let pyr = build_pyramid(y0.clone(), nlev, true);
        let tmpls = precompute_tmpls(&pyr, cfg.stride);
        (pyr, tmpls)
    };
    let build_cur_feats = || {
        // Compute gaussian_blur(&y0, 0.7) ONCE and reuse as both grad_feature input and dt_filter4 guide.
        let guide = gaussian_blur(&y0, 0.7);
        let g0 = grad_feature_from_smooth(&guide);
        let sq = Plane::from_vec(wa, ha, g0.x.d.iter().zip(&g0.y.d).map(|(a, b)| a * a + b * b).collect());
        let lc = gaussian_blur(&sq, 1.0);
        let lc = Plane::from_vec(wa, ha, lc.d.iter().map(|v| v.sqrt()).collect());
        // Structure weight, relative to the frame's mean gradient magnitude.
        let (gx, gy) = gradients(&gaussian_blur(&y0, 1.0));
        let gm: Vec<f32> = gx.d.iter().zip(&gy.d).map(|(a, b)| (a * a + b * b).sqrt()).collect();
        let mg = det_sum(ha, |y| gm[y * wa..(y + 1) * wa].iter().map(|&v| v as f64).sum()) / n;
        let gth = (0.5 * mg as f32).max(1e-4);
        let sw = Plane::from_vec(wa, ha, gm.iter().map(|g| 0.01 + g * g / (g * g + gth * gth)).collect());
        let sw_den = det_sum(ha, |y| sw.d[y * wa..(y + 1) * wa].iter().map(|&v| v as f64).sum());
        (guide, g0, lc, sw, sw_den)
    };
    let build_nb = |x: Option<(Plane, f32, f32)>| {
        x.map(|(p, fd, sign)| {
            let g = grad_feature(&p);
            let lv = need(fd);
            let pyr = build_pyramid(p, lv, true);
            NbIn { pyr, g, fd, sign }
        })
    };
    let (((p0, tmpls0), (guide, g0, lc, sw, sw_den)), nbs) = par::join_split(
        || par::join_split(build_cur_pyr, build_cur_feats),
        || par::join_split(|| build_nb(np), || build_nb(nn)),
    );

    // Flow + confidence per neighbour.
    let run = |nb: &Option<NbIn>| -> Option<(Plane, Plane, Plane, f32, f32, f32)> {
        let nb = nb.as_ref()?;
        // Forward (cur -> neighbour) and backward (neighbour -> cur) flow.
        let ((u, v), (bu, bv)) = par::join_split(
            || {
                let (u, v) = dis_flow(&p0, Some(&tmpls0), &nb.pyr, nb.pyr.len(), cfg);
                (median3(&u), median3(&v))
            },
            || {
                let (u, v) = dis_flow(&nb.pyr, None, &p0, nb.pyr.len(), cfg);
                (median3(&u), median3(&v))
            },
        );
        let (mut c, expl) = confidence(&g0, &lc, &sw, sw_den, &nb.g, &u, &v);
        fb_consistency(&mut c, &u, &v, &bu, &bv);
        Some((u, v, c, expl, nb.fd, nb.sign))
    };
    let (rp, rn) = par::join_split(|| run(&nbs.0), || run(&nbs.1));
    drop((p0, tmpls0, nbs, g0, lc));
    let mut explained = [0.0f32; 2];
    const CUT_FRACTION: f32 = 0.35;
    let keep = |r: Option<(Plane, Plane, Plane, f32, f32, f32)>, slot: usize, ex: &mut [f32; 2]| {
        r.and_then(|r| {
            ex[slot] = r.3;
            if r.3 >= CUT_FRACTION { Some(r) } else { None }
        })
    };
    let rp = keep(rp, 0, &mut explained);
    let rn = keep(rn, 1, &mut explained);
    if rp.is_none() && rn.is_none() {
        return None;
    }
    let used_prev = rp.is_some();
    let used_next = rn.is_some();

    // Confidence-weighted central difference.
    let sm = sens_map(params.motion_sensitivity);
    let mut ch = vec![[0.0f32; 4]; wa * ha];
    let min_r = (16384 / wa.max(1)).max(8);
    par::for_rows_mut(&mut ch, wa, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(wa).enumerate() {
            let y = y0 + dy;
            for (x, o) in row.iter_mut().enumerate() {
                let i = y * wa + x;
                let (mut su, mut sv, mut sc, mut cmax) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
                for r in [&rp, &rn].into_iter().flatten() {
                    let cw = r.2.d[i] + 1e-3;
                    let k = r.5 / r.4;
                    su += cw * r.0.d[i] * k;
                    sv += cw * r.1.d[i] * k;
                    sc += cw;
                    cmax = cmax.max(r.2.d[i]);
                }
                let st = sw.d[i];
                let wgt = st * cmax;
                *o = [wgt * su / sc, wgt * sv / sc, wgt, st];
            }
        }
    });
    dt_filter4(&mut ch, &guide, sm.sigma_s, 0.1 * NORM_STD / 0.2, 3);

    let mut vu = vec![0.0f32; wa * ha];
    let mut vv = vec![0.0f32; wa * ha];
    par::for_rows_mut2(&mut vu, &mut vv, wa, wa, min_r, |y0, cu, cv| {
        for (dy, (ru, rv)) in cu.chunks_exact_mut(wa).zip(cv.chunks_exact_mut(wa)).enumerate() {
            let y = y0 + dy;
            for x in 0..wa {
                let i = y * wa + x;
                let [nu, nv, wsum, ssum] = ch[i];
                if wsum <= 1e-7 {
                    continue;
                }
                let u = nu / wsum;
                let v = nv / wsum;
                let cs = wsum / ssum.max(1e-7);
                let g = smoothstep(sm.c_lo, sm.c_hi, cs);
                let len = (u * u + v * v).sqrt();
                let shrink = if len > 0.0 { ((len - sm.dead).max(0.0) / len) * g } else { 0.0 };
                ru[x] = u * shrink;
                rv[x] = v * shrink;
            }
        }
    });
    Some(Analysis {
        wa,
        ha,
        vu: Plane::from_vec(wa, ha, vu),
        vv: Plane::from_vec(wa, ha, vv),
        ya: y0,
        norm: (gain, off),
        used_prev,
        used_next,
        explained,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dt_line_pass_handles_small_and_empty_lengths() {
        for &n in &[0usize, 1, 2, 3, 7, 16] {
            let mut r: Vec<[f32; 4]> = (0..n).map(|i| [i as f32, (i * 2) as f32, 0.5, 1.0]).collect();
            let dr = vec![1.25f32; n];
            let mut v = vec![0.0f32; n];
            dt_line_pass(&mut r, &dr, &mut v, n, -0.5);
            assert!(r.iter().flatten().all(|x| x.is_finite()));
        }
    }
}
