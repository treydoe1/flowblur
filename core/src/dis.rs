//! Dense Inverse Search optical flow (Kroeger et al. 2016), written from scratch.
//!
//! Coarse-to-fine over a Gaussian pyramid. Per level: an 8x8 patch grid with configurable stride,
//! each patch refined by inverse-compositional Gauss-Newton on mean-normalised intensities, with
//! spatial propagation of candidates from the neighbouring patch (row pass, then column pass),
//! rejection of updates that move too far or worsen the fit, and densification by weighting
//! overlapping patches with their per-pixel photometric fit.

use crate::par;
use crate::plane::{Plane, downsample2, gradients, upsample_flow};

pub(crate) const PS: usize = 8;
const PN: usize = PS * PS;

pub(crate) struct Level {
    pub img: Plane,
    pub gx: Plane,
    pub gy: Plane,
}

/// Pyramid with `n` levels (index 0 = finest). Gradients only when `grads` (template side).
pub(crate) fn build_pyramid(base: Plane, n: usize, grads: bool) -> Vec<Level> {
    let mut out: Vec<Level> = Vec::with_capacity(n);
    let mut cur = base;
    for i in 0..n {
        let next = if i + 1 < n { Some(downsample2(&cur)) } else { None };
        let (gx, gy) = if grads { gradients(&cur) } else { (Plane::new(0, 0), Plane::new(0, 0)) };
        out.push(Level { img: cur, gx, gy });
        match next {
            Some(nx) => cur = nx,
            None => break,
        }
    }
    out
}

/// Exhaustive integer search radius used at the coarsest level.
pub(crate) const SEARCH_R: i32 = 8;

/// Number of pyramid levels so that a displacement of `need_px` (analysis pixels) is within the
/// exhaustive search radius at the coarsest level, limited so the coarsest level keeps at least
/// 2.5 patches across its short side.
pub(crate) fn levels_for(w: usize, h: usize, need_px: f32) -> usize {
    let mut k = 0usize;
    while need_px / (1u32 << k) as f32 > SEARCH_R as f32 - 1.0 {
        let nw = w >> (k + 1);
        let nh = h >> (k + 1);
        if nw < 5 * PS / 2 || nh < 5 * PS / 2 {
            break;
        }
        k += 1;
    }
    k + 1
}

#[derive(Clone, Copy)]
pub(crate) struct DisCfg {
    pub stride: usize,
    pub iters: usize,
}

fn grid(n: usize, stride: usize) -> Vec<usize> {
    if n <= PS {
        return vec![0];
    }
    let mut v: Vec<usize> = (0..=n - PS).step_by(stride).collect();
    if *v.last().unwrap() != n - PS {
        v.push(n - PS);
    }
    v
}

#[repr(align(16))]
pub(crate) struct Tmpl {
    t: [f32; PN], // mean-subtracted template
    gx: [f32; PN],
    gy: [f32; PN],
    hinv: [f32; 3], // symmetric inverse (xx, xy, yy)
    tn: f32,        // template energy
    mt: f32,        // raw template mean (reused by refine_patch)
}

pub(crate) struct LevelTmpls {
    xs: Vec<usize>,
    ys: Vec<usize>,
    tmpls: Vec<Tmpl>,
}

/// Precomputes patch templates for a single pyramid level.
fn precompute_level_tmpls(l: &Level, stride: usize) -> LevelTmpls {
    let (w, h) = (l.img.w, l.img.h);
    if w < PS || h < PS {
        return LevelTmpls { xs: Vec::new(), ys: Vec::new(), tmpls: Vec::new() };
    }
    let xs = grid(w, stride);
    let ys = grid(h, stride);
    let (nx, ny) = (xs.len(), ys.len());
    let mut tmpls: Vec<Tmpl> = (0..nx * ny)
        .map(|_| Tmpl {
            t: [0.0; PN],
            gx: [0.0; PN],
            gy: [0.0; PN],
            hinv: [0.0; 3],
            tn: 0.0,
            mt: 0.0,
        })
        .collect();
    let min_r = (256 / nx.max(1)).max(4);
    par::for_rows_mut(&mut tmpls, nx, min_r, |j0, chunk| {
        for (dj, row) in chunk.chunks_exact_mut(nx).enumerate() {
            let y0 = ys[j0 + dj];
            for (i, slot) in row.iter_mut().enumerate() {
                *slot = make_tmpl(l, xs[i], y0);
            }
        }
    });
    LevelTmpls { xs, ys, tmpls }
}

/// Precomputes patch templates for every level of a pyramid so they are built ONCE and reused
/// across coarsest search, Pass 1, Pass 2, and (for `p0`) across both prev and next directions.
pub(crate) fn precompute_tmpls(pyr: &[Level], stride: usize) -> Vec<LevelTmpls> {
    pyr.iter().map(|l| precompute_level_tmpls(l, stride)).collect()
}

fn make_tmpl(l: &Level, x0: usize, y0: usize) -> Tmpl {
    let mut t = [0.0f32; PN];
    let mut gx = [0.0f32; PN];
    let mut gy = [0.0f32; PN];
    let (mut mt, mut mx, mut my) = (0.0f32, 0.0f32, 0.0f32);
    let w = l.img.w;
    for j in 0..PS {
        let r = (y0 + j) * w + x0;
        let it = &l.img.d[r..r + PS];
        let igx = &l.gx.d[r..r + PS];
        let igy = &l.gy.d[r..r + PS];
        let o = j * PS;
        for i in 0..PS {
            let tv = it[i];
            let gxv = igx[i];
            let gyv = igy[i];
            t[o + i] = tv;
            gx[o + i] = gxv;
            gy[o + i] = gyv;
            mt += tv;
            mx += gxv;
            my += gyv;
        }
    }
    let inv = 1.0 / PN as f32;
    mt *= inv;
    mx *= inv;
    my *= inv;
    let (mut hxx, mut hxy, mut hyy, mut tn) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for k in 0..PN {
        let tv = t[k] - mt;
        let gxv = gx[k] - mx;
        let gyv = gy[k] - my;
        t[k] = tv;
        gx[k] = gxv;
        gy[k] = gyv;
        hxx += gxv * gxv;
        hxy += gxv * gyv;
        hyy += gyv * gyv;
        tn += tv * tv;
    }
    // Tikhonov term keeps flat / 1-D patches from producing wild updates.
    let eps = 2e-4;
    hxx += eps;
    hyy += eps;
    let det = hxx * hyy - hxy * hxy;
    let id = 1.0 / det;
    Tmpl { t, gx, gy, hinv: [hyy * id, -hxy * id, hxx * id], tn, mt }
}

/// Path selection for `warp`, split out so overflow and non-finite inputs can be unit-tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WarpPath {
    IntFast(usize, usize),
    BilinearFast(usize, usize),
    Clamped,
}

#[inline(always)]
fn choose_warp_path(w: usize, h: usize, bx: f32, by: f32, ax: f32, ay: f32, ix: isize, iy: isize) -> WarpPath {
    if !bx.is_finite() || !by.is_finite() || ix < 0 || iy < 0 {
        return WarpPath::Clamped;
    }
    if ax == 0.0 && ay == 0.0 {
        if w >= PS && h >= PS && ix <= (w - PS) as isize && iy <= (h - PS) as isize {
            return WarpPath::IntFast(ix as usize, iy as usize);
        }
    }
    if w > PS && h > PS && ix <= (w - PS - 1) as isize && iy <= (h - PS - 1) as isize {
        return WarpPath::BilinearFast(ix as usize, iy as usize);
    }
    WarpPath::Clamped
}

/// Bilinearly warps the 8x8 patch at (x0,y0)+u out of `img`; returns the patch mean.
#[inline(always)]
fn warp(img: &Plane, x0: usize, y0: usize, u: [f32; 2], out: &mut [f32; PN]) -> f32 {
    let bx = x0 as f32 + u[0];
    let by = y0 as f32 + u[1];
    let fx = bx.floor();
    let fy = by.floor();
    let ax = bx - fx;
    let ay = by - fy;
    let ix = fx as isize;
    let iy = fy as isize;
    let wu = img.w;
    let hu = img.h;
    debug_assert_eq!(img.d.len(), wu * hu);

    match choose_warp_path(wu, hu, bx, by, ax, ay, ix, iy) {
        WarpPath::IntFast(ixu, iyu) => {
            let mut sum = 0.0f32;
            for j in 0..PS {
                let r0 = &img.d[(iyu + j) * wu + ixu..(iyu + j) * wu + ixu + PS];
                let o = &mut out[j * PS..j * PS + PS];
                o.copy_from_slice(r0);
                for &v in r0 {
                    sum += v;
                }
            }
            sum / PN as f32
        }
        WarpPath::BilinearFast(ixu, iyu) => {
            let w00 = (1.0 - ax) * (1.0 - ay);
            let w10 = ax * (1.0 - ay);
            let w01 = (1.0 - ax) * ay;
            let w11 = ax * ay;
            #[cfg(target_arch = "aarch64")]
            {
                assert!(img.d.len() >= wu * hu);
                unsafe {
                    use std::arch::aarch64::*;
                    let src = img.d.as_ptr();
                    let dst = out.as_mut_ptr();
                    let vw00 = vdupq_n_f32(w00);
                    let vw10 = vdupq_n_f32(w10);
                    let vw01 = vdupq_n_f32(w01);
                    let vw11 = vdupq_n_f32(w11);
                    let mut sum = 0.0f32;
                    for j in 0..PS {
                        let p0 = src.add((iyu + j) * wu + ixu);
                        let p1 = src.add((iyu + j + 1) * wu + ixu);
                        // Left-associative unfused sum matching scalar Rust bit-for-bit.
                        let a0 = vaddq_f32(
                            vaddq_f32(
                                vaddq_f32(vmulq_f32(vw00, vld1q_f32(p0)), vmulq_f32(vw10, vld1q_f32(p0.add(1)))),
                                vmulq_f32(vw01, vld1q_f32(p1)),
                            ),
                            vmulq_f32(vw11, vld1q_f32(p1.add(1))),
                        );
                        let a1 = vaddq_f32(
                            vaddq_f32(
                                vaddq_f32(vmulq_f32(vw00, vld1q_f32(p0.add(4))), vmulq_f32(vw10, vld1q_f32(p0.add(5)))),
                                vmulq_f32(vw01, vld1q_f32(p1.add(4))),
                            ),
                            vmulq_f32(vw11, vld1q_f32(p1.add(5))),
                        );
                        vst1q_f32(dst.add(j * PS), a0);
                        vst1q_f32(dst.add(j * PS + 4), a1);
                        sum += vgetq_lane_f32::<0>(a0);
                        sum += vgetq_lane_f32::<1>(a0);
                        sum += vgetq_lane_f32::<2>(a0);
                        sum += vgetq_lane_f32::<3>(a0);
                        sum += vgetq_lane_f32::<0>(a1);
                        sum += vgetq_lane_f32::<1>(a1);
                        sum += vgetq_lane_f32::<2>(a1);
                        sum += vgetq_lane_f32::<3>(a1);
                    }
                    return sum / PN as f32;
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                let mut sum = 0.0;
                for j in 0..PS {
                    let r0 = &img.d[(iyu + j) * wu + ixu..(iyu + j) * wu + ixu + PS + 1];
                    let r1 = &img.d[(iyu + j + 1) * wu + ixu..(iyu + j + 1) * wu + ixu + PS + 1];
                    for i in 0..PS {
                        let v = w00 * r0[i] + w10 * r0[i + 1] + w01 * r1[i] + w11 * r1[i + 1];
                        out[j * PS + i] = v;
                        sum += v;
                    }
                }
                sum / PN as f32
            }
        }
        WarpPath::Clamped => {
            let w00 = (1.0 - ax) * (1.0 - ay);
            let w10 = ax * (1.0 - ay);
            let w01 = (1.0 - ax) * ay;
            let w11 = ax * ay;
            let mut sum = 0.0;
            for j in 0..PS {
                let yy = iy.saturating_add(j as isize);
                let yy1 = yy.saturating_add(1);
                for i in 0..PS {
                    let xx = ix.saturating_add(i as isize);
                    let xx1 = xx.saturating_add(1);
                    let v = w00 * img.get_c(xx, yy)
                        + w10 * img.get_c(xx1, yy)
                        + w01 * img.get_c(xx, yy1)
                        + w11 * img.get_c(xx1, yy1);
                    out[j * PS + i] = v;
                    sum += v;
                }
            }
            sum / PN as f32
        }
    }
}

/// Fraction of the patch at (x0,y0)+u that falls outside the image.
#[inline(always)]
fn out_frac(img: &Plane, x0: usize, y0: usize, u: [f32; 2]) -> f32 {
    let inside = |p: f32, n: usize| -> f32 {
        let lo = p.max(0.0);
        let hi = (p + PS as f32).min(n as f32);
        ((hi - lo) / PS as f32).clamp(0.0, 1.0)
    };
    1.0 - inside(x0 as f32 + u[0], img.w) * inside(y0 as f32 + u[1], img.h)
}

#[inline(always)]
fn ssd_64(buf: &[f32; PN], t: &[f32; PN], m: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let bp = buf.as_ptr();
        let tp = t.as_ptr();
        let vm = vdupq_n_f32(m);
        let mut s = 0.0f32;
        let mut k = 0usize;
        while k < PN {
            let r = vsubq_f32(vsubq_f32(vld1q_f32(bp.add(k)), vm), vld1q_f32(tp.add(k)));
            let r2 = vmulq_f32(r, r);
            s += vgetq_lane_f32::<0>(r2);
            s += vgetq_lane_f32::<1>(r2);
            s += vgetq_lane_f32::<2>(r2);
            s += vgetq_lane_f32::<3>(r2);
            k += 4;
        }
        s
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut s = 0.0f32;
        for k in 0..PN {
            let r = (buf[k] - m) - t[k];
            s += r * r;
        }
        s
    }
}

#[inline(always)]
fn gn_rhs_64(buf: &[f32; PN], tm: &Tmpl, m: f32) -> (f32, f32) {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        use std::arch::aarch64::*;
        let bp = buf.as_ptr();
        let tp = tm.t.as_ptr();
        let gxp = tm.gx.as_ptr();
        let gyp = tm.gy.as_ptr();
        let vm = vdupq_n_f32(m);
        let (mut bx, mut by) = (0.0f32, 0.0f32);
        let mut k = 0usize;
        while k < PN {
            let r = vsubq_f32(vsubq_f32(vld1q_f32(bp.add(k)), vm), vld1q_f32(tp.add(k)));
            let px = vmulq_f32(vld1q_f32(gxp.add(k)), r);
            let py = vmulq_f32(vld1q_f32(gyp.add(k)), r);
            bx += vgetq_lane_f32::<0>(px);
            bx += vgetq_lane_f32::<1>(px);
            bx += vgetq_lane_f32::<2>(px);
            bx += vgetq_lane_f32::<3>(px);
            by += vgetq_lane_f32::<0>(py);
            by += vgetq_lane_f32::<1>(py);
            by += vgetq_lane_f32::<2>(py);
            by += vgetq_lane_f32::<3>(py);
            k += 4;
        }
        (bx, by)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let (mut bx, mut by) = (0.0f32, 0.0f32);
        for k in 0..PN {
            let r = (buf[k] - m) - tm.t[k];
            bx += tm.gx[k] * r;
            by += tm.gy[k] * r;
        }
        (bx, by)
    }
}

/// Zero-mean SSD of the template against the warped patch. The out-of-frame part of the patch is
/// edge-clamped and would match flat-ish templates suspiciously well, so it is charged as a
/// mismatch: without this, content leaving the frame produces wild vectors that propagate inward.
#[inline(always)]
fn cost(tm: &Tmpl, img: &Plane, x0: usize, y0: usize, u: [f32; 2], buf: &mut [f32; PN]) -> (f32, f32) {
    let m = warp(img, x0, y0, u, buf);
    let mut s = ssd_64(buf, &tm.t, m);
    let fo = out_frac(img, x0, y0, u);
    if fo > 0.0 {
        s = s * (1.0 + 4.0 * fo) + 2.0 * fo * (tm.tn + 1e-3);
    }
    (s, m)
}

#[derive(Clone, Copy, Default)]
struct PatchRes {
    u: [f32; 2],
    off: f32, // mean(warped) - mean(template) at the final u
}

/// Exhaustive zero-mean SSD search over integer offsets in [-r, r]^2 around `c`, with a tiny
/// preference for small motion to resolve flat / ambiguous patches.
/// Returns the best offset and its cost relative to the template energy.
fn search_patch(tm: &Tmpl, i1: &Plane, x0: usize, y0: usize, c: [f32; 2], r: i32) -> ([f32; 2], f32) {
    let mut buf = [0.0f32; PN];
    let lam = 1e-4 * (tm.tn + 1e-3);
    let mut best = c;
    let mut bc = f32::INFINITY;
    for dy in -r..=r {
        for dx in -r..=r {
            let u = [c[0] + dx as f32, c[1] + dy as f32];
            let (cc, _) = cost(tm, i1, x0, y0, u, &mut buf);
            let cc = cc + lam * (u[0] * u[0] + u[1] * u[1]);
            if cc < bc {
                bc = cc;
                best = u;
            }
        }
    }
    (best, bc / (tm.tn + 1e-4))
}

fn refine_patch(
    tm: &Tmpl,
    i1: &Plane,
    x0: usize,
    y0: usize,
    cands: &[[f32; 2]],
    iters: usize,
) -> PatchRes {
    let mut buf = [0.0f32; PN];
    let mt = tm.mt;
    // Pick the best starting candidate, tracking what `buf` currently holds so iter 0 can skip warp.
    let mut start = cands[0];
    let (mut c0, mut m0) = cost(tm, i1, x0, y0, start, &mut buf);
    let mut buf_u = start;
    let mut buf_m = m0;
    for &c in &cands[1..] {
        if (c[0] == start[0] && c[1] == start[1]) || (c[0] == buf_u[0] && c[1] == buf_u[1]) {
            continue;
        }
        let (cc, mm) = cost(tm, i1, x0, y0, c, &mut buf);
        buf_u = c;
        buf_m = mm;
        if cc < c0 {
            c0 = cc;
            m0 = mm;
            start = c;
        }
    }
    let mut u = start;
    let mut buf_valid = u[0] == buf_u[0] && u[1] == buf_u[1];
    for _ in 0..iters {
        let m = if buf_valid {
            buf_m
        } else {
            buf_m = warp(i1, x0, y0, u, &mut buf);
            buf_m
        };
        let (bx, by) = gn_rhs_64(&buf, tm, m);
        let dx = tm.hinv[0] * bx + tm.hinv[1] * by;
        let dy = tm.hinv[1] * bx + tm.hinv[2] * by;
        u[0] -= dx;
        u[1] -= dy;
        buf_valid = dx == 0.0 && dy == 0.0;
        if dx * dx + dy * dy < 1e-6 {
            break;
        }
    }
    let du = [u[0] - start[0], u[1] - start[1]];
    let (c1, m1) = if buf_valid {
        let mut s = ssd_64(&buf, &tm.t, buf_m);
        let fo = out_frac(i1, x0, y0, u);
        if fo > 0.0 {
            s = s * (1.0 + 4.0 * fo) + 2.0 * fo * (tm.tn + 1e-3);
        }
        (s, buf_m)
    } else {
        cost(tm, i1, x0, y0, u, &mut buf)
    };
    // Reject when the fit got worse (or became NaN) or the update is wild.
    let improved = matches!(c1.partial_cmp(&c0), Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal));
    if !improved || du[0] * du[0] + du[1] * du[1] > (PS * PS) as f32 || !u[0].is_finite() || !u[1].is_finite() {
        return PatchRes { u: start, off: m0 - mt };
    }
    PatchRes { u, off: m1 - mt }
}

/// Densify the patch flows into a per-pixel field.
fn densify(l0: &Level, i1: &Plane, xs: &[usize], ys: &[usize], pr: &[PatchRes]) -> (Plane, Plane) {
    let (w, h) = (l0.img.w, l0.img.h);
    let nx = xs.len();
    let range = |v: &[usize], p: usize| -> (usize, usize) {
        let hi = match v.binary_search(&p) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let mut lo = hi;
        while lo > 0 && v[lo - 1] + PS > p {
            lo -= 1;
        }
        (lo, hi)
    };
    let xr: Vec<(usize, usize)> = (0..w).map(|x| range(xs, x)).collect();
    let mut fu = vec![0.0f32; w * h];
    let mut fv = vec![0.0f32; w * h];
    let min_r = (8192 / w.max(1)).max(8);
    let wm = (w - 1) as f32;
    let hm = (h - 1) as f32;
    let i1d = &i1.d;
    assert!(w > 0 && h > 0 && i1d.len() == w * h);
    par::for_rows_mut2(&mut fu, &mut fv, w, w, min_r, |y0, cu, cv| {
        for (dy, (ru, rv)) in cu.chunks_exact_mut(w).zip(cv.chunks_exact_mut(w)).enumerate() {
            let y = y0 + dy;
            let yf = y as f32;
            let (ylo, yhi) = range(ys, y);
            let l0_row = &l0.img.d[y * w..(y + 1) * w];
            for x in 0..w {
                let (xlo, xhi) = xr[x];
                let xf = x as f32;
                let i0 = l0_row[x];
                let (mut su, mut sv, mut sw) = (0.0f32, 0.0f32, 0.0f32);
                for j in ylo..=yhi {
                    let pr_row = &pr[j * nx..];
                    for i in xlo..=xhi {
                        let p = &pr_row[i];
                        let px = (xf + p.u[0]).clamp(0.0, wm);
                        let py = (yf + p.u[1]).clamp(0.0, hm);
                        let bx = (px as usize).min(w - 1);
                        let by = (py as usize).min(h - 1);
                        let bx1 = (bx + 1).min(w - 1);
                        let by1 = (by + 1).min(h - 1);
                        let ax = px - bx as f32;
                        let ay = py - by as f32;
                        let r0 = by * w;
                        let r1 = by1 * w;
                        // SAFETY: bx, bx1 < w and by, by1 < h (guaranteed by .min(w-1)/(h-1)), and i1d.len() == w * h.
                        let (d00, d10, d01, d11) = unsafe {
                            (
                                *i1d.get_unchecked(r0 + bx),
                                *i1d.get_unchecked(r0 + bx1),
                                *i1d.get_unchecked(r1 + bx),
                                *i1d.get_unchecked(r1 + bx1),
                            )
                        };
                        let a = d00 + ax * (d10 - d00);
                        let b = d01 + ax * (d11 - d01);
                        let sample = a + ay * (b - a);
                        let d = sample - i0 - p.off;
                        let wt = 1.0 / (d.abs() * 255.0).max(1.0);
                        su += wt * p.u[0];
                        sv += wt * p.u[1];
                        sw += wt;
                    }
                }
                if sw > 0.0 {
                    ru[x] = su / sw;
                    rv[x] = sv / sw;
                }
            }
        }
    });
    (Plane::from_vec(w, h, fu), Plane::from_vec(w, h, fv))
}

/// Flow from `p0` (template, with optional precomputed templates `tmpls0`) to `p1`.
/// When `tmpls0` is `None`, each level's templates are built on demand and dropped before the
/// next level to minimise peak memory.
/// Both pyramids must have at least `n` levels. Returns (u, v) at level-0 resolution.
pub(crate) fn dis_flow(
    p0: &[Level],
    tmpls0: Option<&[LevelTmpls]>,
    p1: &[Level],
    n: usize,
    cfg: DisCfg,
) -> (Plane, Plane) {
    let n = n.min(p0.len()).min(p1.len()).max(1);
    let mut flow: Option<(Plane, Plane)> = None;
    for lev in (0..n).rev() {
        let l0 = &p0[lev];
        let i1 = &p1[lev].img;
        let (w, h) = (l0.img.w, l0.img.h);
        let (fu, fv) = match flow.take() {
            None => (Plane::new(w, h), Plane::new(w, h)),
            Some((u, v)) => (upsample_flow(&u, w, h, 2.0), upsample_flow(&v, w, h, 2.0)),
        };
        if w < PS || h < PS {
            flow = Some((fu, fv));
            continue;
        }
        let owned_lt;
        let lt = match tmpls0 {
            Some(ts) => &ts[lev],
            None => {
                owned_lt = precompute_level_tmpls(l0, cfg.stride);
                &owned_lt
            }
        };
        let xs = &lt.xs;
        let ys = &lt.ys;
        let tmpls = &lt.tmpls;
        let (nx, ny) = (xs.len(), ys.len());
        let half = PS as f32 * 0.5 - 0.5;
        let mut init: Vec<[f32; 2]> = (0..nx * ny)
            .map(|p| {
                let cx = xs[p % nx] as f32 + half;
                let cy = ys[p / nx] as f32 + half;
                [fu.bilinear(cx, cy), fv.bilinear(cx, cy)]
            })
            .collect();
        let it1 = cfg.iters.div_ceil(2).max(1);
        if lev == n - 1 {
            // Coarsest level: exhaustive integer search per patch, then a 5x5 median over the patch
            // grid.
            let mut found = vec![([0.0f32; 2], 0.0f32); nx * ny];
            par::for_rows_mut(&mut found, 1, 8, |p0_idx, chunk| {
                for (dp, slot) in chunk.iter_mut().enumerate() {
                    let p = p0_idx + dp;
                    *slot = search_patch(&tmpls[p], i1, xs[p % nx], ys[p / nx], init[p], SEARCH_R);
                }
            });
            let valid: Vec<bool> = found.iter().map(|f| f.1 < 0.25).collect();
            let mut a = Vec::with_capacity(25);
            let mut b = Vec::with_capacity(25);
            init = (0..nx * ny)
                .map(|p| {
                    let (i, j) = ((p % nx) as isize, (p / nx) as isize);
                    let r_max = nx.max(ny) as isize;
                    let mut r = 2isize;
                    loop {
                        a.clear();
                        b.clear();
                        let j_lo = (j - r).max(0);
                        let j_hi = (j + r).min(ny as isize - 1);
                        let i_lo = (i - r).max(0);
                        let i_hi = (i + r).min(nx as isize - 1);
                        for jj in j_lo..=j_hi {
                            let row_q = jj as usize * nx;
                            for ii in i_lo..=i_hi {
                                let q = row_q + ii as usize;
                                if valid[q] {
                                    a.push(found[q].0[0]);
                                    b.push(found[q].0[1]);
                                }
                            }
                        }
                        if a.len() >= 3 || r >= r_max {
                            if a.is_empty() {
                                break [0.0, 0.0];
                            }
                            a.sort_by(|x, y| x.total_cmp(y));
                            b.sort_by(|x, y| x.total_cmp(y));
                            break [a[a.len() / 2], b[b.len() / 2]];
                        }
                        r *= 2;
                    }
                })
                .collect();
        }
        // Pass 1: rows, left to right, propagating from the left neighbour.
        let mut pr: Vec<PatchRes> = vec![PatchRes::default(); nx * ny];
        let min_p1 = (128 / nx.max(1)).max(2);
        par::for_rows_mut(&mut pr, nx, min_p1, |j0, chunk| {
            for (dj, row) in chunk.chunks_exact_mut(nx).enumerate() {
                let j = j0 + dj;
                let y0 = ys[j];
                let t_row = &tmpls[j * nx..(j + 1) * nx];
                let init_row = &init[j * nx..(j + 1) * nx];
                for i in 0..nx {
                    let own = init_row[i];
                    let res = if i > 0 {
                        refine_patch(&t_row[i], i1, xs[i], y0, &[own, row[i - 1].u], it1)
                    } else {
                        refine_patch(&t_row[i], i1, xs[i], y0, &[own], it1)
                    };
                    row[i] = res;
                }
            }
        });
        // Pass 2: columns, bottom to top, propagating from the patch below.
        let mut cols = vec![PatchRes::default(); nx * ny];
        let min_p2 = (128 / ny.max(1)).max(2);
        par::for_rows_mut(&mut cols, ny, min_p2, |i0, chunk| {
            for (di, col) in chunk.chunks_exact_mut(ny).enumerate() {
                let i = i0 + di;
                let x0 = xs[i];
                for j in 0..ny {
                    col[j] = pr[j * nx + i];
                }
                for j in (0..ny).rev() {
                    let own = col[j].u;
                    let tm = &tmpls[j * nx + i];
                    col[j] = if j + 1 < ny {
                        refine_patch(tm, i1, x0, ys[j], &[own, col[j + 1].u], it1)
                    } else {
                        refine_patch(tm, i1, x0, ys[j], &[own], it1)
                    };
                }
            }
        });
        for i in 0..nx {
            let col = &cols[i * ny..(i + 1) * ny];
            for j in 0..ny {
                pr[j * nx + i] = col[j];
            }
        }
        flow = Some(densify(l0, i1, xs, ys, &pr));
    }
    flow.unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_handles_extreme_nonfinite_and_small_images_without_oob_or_overflow() {
        let extreme_vals = [
            0.0f32,
            0.5,
            -0.5,
            1.0,
            -1.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            1e30,
            -1e30,
            f32::MAX,
            f32::MIN,
            (isize::MAX as f32),
            -(isize::MAX as f32),
        ];
        for &(w, h) in &[(1usize, 1usize), (2, 2), (7, 7), (8, 8), (9, 9), (10, 10), (16, 12)] {
            let data: Vec<f32> = (0..w * h).map(|i| (i as f32 + 1.0) / (w * h) as f32).collect();
            let img = Plane::from_vec(w, h, data);
            let mut out = [0.0f32; PN];
            for &ux in &extreme_vals {
                for &uy in &extreme_vals {
                    let bx = ux;
                    let by = uy;
                    let fx = bx.floor();
                    let fy = by.floor();
                    let ax = bx - fx;
                    let ay = by - fy;
                    let ix = fx as isize;
                    let iy = fy as isize;
                    let path = choose_warp_path(w, h, bx, by, ax, ay, ix, iy);
                    if !ux.is_finite() || !uy.is_finite() || ux.abs() > 1e6 || uy.abs() > 1e6 {
                        assert_eq!(path, WarpPath::Clamped, "w={w} h={h} u=[{ux},{uy}] took {path:?}");
                    }
                    if w < PS || h < PS {
                        assert_eq!(path, WarpPath::Clamped, "w={w} h={h} smaller than PS took {path:?}");
                    }
                    if w == PS || h == PS {
                        assert!(!matches!(path, WarpPath::BilinearFast(..)), "w={w} h={h} took BilinearFast");
                    }
                    let _ = warp(&img, 0, 0, [ux, uy], &mut out);
                }
            }
            // Exact boundary cases on w=9, h=9 and w=10, h=10
            if w == 9 && h == 9 {
                assert_eq!(choose_warp_path(w, h, 0.0, 0.0, 0.0, 0.0, 0, 0), WarpPath::IntFast(0, 0));
                assert_eq!(choose_warp_path(w, h, 1.0, 1.0, 0.0, 0.0, 1, 1), WarpPath::IntFast(1, 1));
                assert_eq!(choose_warp_path(w, h, 0.5, 0.5, 0.5, 0.5, 0, 0), WarpPath::BilinearFast(0, 0));
                assert_eq!(choose_warp_path(w, h, 1.5, 0.5, 0.5, 0.5, 1, 0), WarpPath::Clamped);
            }
        }
    }
}
