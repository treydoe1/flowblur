//! Single-channel f32 image plane and the basic filters used by the flow stage.

use crate::par;

#[derive(Clone, Debug)]
pub(crate) struct Plane {
    pub w: usize,
    pub h: usize,
    pub d: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize) -> Self {
        Plane { w, h, d: vec![0.0; w * h] }
    }

    pub fn from_vec(w: usize, h: usize, d: Vec<f32>) -> Self {
        debug_assert_eq!(d.len(), w * h);
        Plane { w, h, d }
    }

    #[inline(always)]
    pub fn get_c(&self, x: isize, y: isize) -> f32 {
        let xx = x.clamp(0, self.w as isize - 1) as usize;
        let yy = y.clamp(0, self.h as isize - 1) as usize;
        self.d[yy * self.w + xx]
    }

    /// Bilinear sample at continuous pixel coordinates (pixel centres at integers), clamped to the edge.
    #[inline(always)]
    pub fn bilinear(&self, x: f32, y: f32) -> f32 {
        let w = self.w;
        let h = self.h;
        let xm = (w - 1) as f32;
        let ym = (h - 1) as f32;
        let xf = x.clamp(0.0, xm);
        let yf = y.clamp(0.0, ym);
        let x0 = xf as usize;
        let y0 = yf as usize;
        let x1 = (x0 + 1).min(w - 1);
        let y1 = (y0 + 1).min(h - 1);
        let ax = xf - x0 as f32;
        let ay = yf - y0 as f32;
        let r0 = y0 * w;
        let r1 = y1 * w;
        let d00 = self.d[r0 + x0];
        let d10 = self.d[r0 + x1];
        let d01 = self.d[r1 + x0];
        let d11 = self.d[r1 + x1];
        let a = d00 + ax * (d10 - d00);
        let b = d01 + ax * (d11 - d01);
        a + ay * (b - a)
    }
}

/// Per-output-sample list of (first source index, weights) for an area (box) resample of one axis.
fn area_weights(src_n: usize, dst_n: usize) -> Vec<(usize, Vec<f32>)> {
    let f = src_n as f64 / dst_n as f64;
    (0..dst_n)
        .map(|i| {
            let a = i as f64 * f;
            let b = ((i + 1) as f64 * f).min(src_n as f64);
            let j0 = a.floor() as usize;
            let j1 = (b.ceil() as usize).min(src_n).max(j0 + 1);
            let mut ws = Vec::with_capacity(j1 - j0);
            for j in j0..j1 {
                let lo = (j as f64).max(a);
                let hi = ((j + 1) as f64).min(b);
                ws.push(((hi - lo).max(0.0) / (b - a)) as f32);
            }
            (j0, ws)
        })
        .collect()
}

/// Area-average resample of a plane given as a raw slice. Intended for downscaling.
pub(crate) fn resample_area(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Plane {
    if sw == dw && sh == dh {
        return Plane::from_vec(sw, sh, src.to_vec());
    }
    let wx = area_weights(sw, dw);
    let wy = area_weights(sh, dh);
    let min_r = (16384 / dw.max(1)).max(8);
    let mut tmp = vec![0.0f32; dw * sh];
    par::for_rows_mut(&mut tmp, dw, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(dw).enumerate() {
            let y = y0 + dy;
            let s = &src[y * sw..(y + 1) * sw];
            for (x, o) in row.iter_mut().enumerate() {
                let (j0, ws) = &wx[x];
                let mut acc = 0.0;
                for (k, w) in ws.iter().enumerate() {
                    acc += w * s[j0 + k];
                }
                *o = acc;
            }
        }
    });
    let mut out = vec![0.0f32; dw * dh];
    par::for_rows_mut(&mut out, dw, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(dw).enumerate() {
            let y = y0 + dy;
            let (j0, ws) = &wy[y];
            for (k, &w) in ws.iter().enumerate() {
                let s = &tmp[(j0 + k) * dw..(j0 + k + 1) * dw];
                for (o, &v) in row.iter_mut().zip(s) {
                    *o += w * v;
                }
            }
        }
    });
    Plane::from_vec(dw, dh, out)
}

fn gauss_kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as usize;
    let mut k: Vec<f32> = (0..=2 * r)
        .map(|i| {
            let x = i as f32 - r as f32;
            (-(x * x) / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gauss_h_row_neon(s: &[f32], row: &mut [f32], k: &[f32], r: usize) {
    use std::arch::aarch64::*;
    let w = s.len();
    if w == 0 {
        return;
    }
    let klen = k.len();
    assert_eq!(row.len(), w);
    assert_eq!(klen, 2 * r + 1);
    // Left border
    for x in 0..r.min(w) {
        let xi = x as isize;
        let ri = r as isize;
        let mut acc = 0.0f32;
        for (i, &kv) in k.iter().enumerate() {
            let xx = (xi + i as isize - ri).clamp(0, w as isize - 1) as usize;
            acc += kv * s[xx];
        }
        row[x] = acc;
    }
    // Interior 4-wide NEON
    let mut x = r;
    let end = w.saturating_sub(r);
    unsafe {
        let sp = s.as_ptr();
        let rp = row.as_mut_ptr();
        while x + 4 <= end {
            let base = x - r;
            let mut acc = vdupq_n_f32(0.0);
            for (i, &kv) in k.iter().enumerate() {
                let sv = vld1q_f32(sp.add(base + i));
                acc = vaddq_f32(acc, vmulq_n_f32(sv, kv));
            }
            vst1q_f32(rp.add(x), acc);
            x += 4;
        }
    }
    // Interior scalar tail
    while x < end {
        let base = x - r;
        let mut acc = 0.0f32;
        for i in 0..klen {
            acc += k[i] * s[base + i];
        }
        row[x] = acc;
        x += 1;
    }
    // Right border
    for x in end.max(r.min(w))..w {
        let xi = x as isize;
        let ri = r as isize;
        let mut acc = 0.0f32;
        for (i, &kv) in k.iter().enumerate() {
            let xx = (xi + i as isize - ri).clamp(0, w as isize - 1) as usize;
            acc += kv * s[xx];
        }
        row[x] = acc;
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn gauss_v_row_neon(tmp: &[f32], row: &mut [f32], y: usize, w: usize, h: usize, k: &[f32], r: isize) {
    use std::arch::aarch64::*;
    if w == 0 || h == 0 {
        return;
    }
    assert!(row.len() >= w && tmp.len() >= w * h);
    let klen = k.len();
    let mut row_offs = [0usize; 32];
    let use_stack = klen <= 32;
    let mut heap_offs = Vec::new();
    let offs: &[usize] = if use_stack {
        for i in 0..klen {
            let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
            row_offs[i] = yy * w;
        }
        &row_offs[..klen]
    } else {
        heap_offs.reserve(klen);
        for i in 0..klen {
            let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
            heap_offs.push(yy * w);
        }
        &heap_offs
    };

    let mut x = 0usize;
    unsafe {
        let tp = tmp.as_ptr();
        let rp = row.as_mut_ptr();
        while x + 8 <= w {
            let mut a0 = vdupq_n_f32(0.0);
            let mut a1 = vdupq_n_f32(0.0);
            for i in 0..klen {
                let p = tp.add(offs[i] + x);
                let kv = k[i];
                a0 = vaddq_f32(a0, vmulq_n_f32(vld1q_f32(p), kv));
                a1 = vaddq_f32(a1, vmulq_n_f32(vld1q_f32(p.add(4)), kv));
            }
            vst1q_f32(rp.add(x), a0);
            vst1q_f32(rp.add(x + 4), a1);
            x += 8;
        }
        while x + 4 <= w {
            let mut a0 = vdupq_n_f32(0.0);
            for i in 0..klen {
                let p = tp.add(offs[i] + x);
                a0 = vaddq_f32(a0, vmulq_n_f32(vld1q_f32(p), k[i]));
            }
            vst1q_f32(rp.add(x), a0);
            x += 4;
        }
    }
    while x < w {
        let mut acc = 0.0f32;
        for i in 0..klen {
            acc += k[i] * tmp[offs[i] + x];
        }
        row[x] = acc;
        x += 1;
    }
}

/// Separable Gaussian blur with clamped borders.
pub(crate) fn gaussian_blur(p: &Plane, sigma: f32) -> Plane {
    if sigma <= 0.0 {
        return p.clone();
    }
    let k = gauss_kernel(sigma);
    let r_u = k.len() / 2;
    let r = r_u as isize;
    let (w, h) = (p.w, p.h);
    let min_r = (16384 / w.max(1)).max(8);
    let mut tmp = vec![0.0f32; w * h];
    par::for_rows_mut(&mut tmp, w, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let s = &p.d[y * w..(y + 1) * w];
            #[cfg(target_arch = "aarch64")]
            unsafe {
                gauss_h_row_neon(s, row, &k, r_u);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for (x, o) in row.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    let xi = x as isize;
                    if xi >= r && xi + r < w as isize {
                        let base = (xi - r) as usize;
                        for (i, kv) in k.iter().enumerate() {
                            acc += kv * s[base + i];
                        }
                    } else {
                        for (i, kv) in k.iter().enumerate() {
                            let xx = (xi + i as isize - r).clamp(0, w as isize - 1) as usize;
                            acc += kv * s[xx];
                        }
                    }
                    *o = acc;
                }
            }
        }
    });
    let mut out = vec![0.0f32; w * h];
    par::for_rows_mut(&mut out, w, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            #[cfg(target_arch = "aarch64")]
            unsafe {
                gauss_v_row_neon(&tmp, row, y, w, h, &k, r);
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for (i, kv) in k.iter().enumerate() {
                    let yy = (y as isize + i as isize - r).clamp(0, h as isize - 1) as usize;
                    let s = &tmp[yy * w..(yy + 1) * w];
                    for (o, v) in row.iter_mut().zip(s) {
                        *o += kv * v;
                    }
                }
            }
        }
    });
    Plane::from_vec(w, h, out)
}

/// Gaussian pre-filter then 2x2 box decimation. Coarse pixel i is centred on fine coordinate 2i+0.5.
pub(crate) fn downsample2(p: &Plane) -> Plane {
    let b = gaussian_blur(p, 0.8);
    let w2 = (p.w / 2).max(1);
    let h2 = (p.h / 2).max(1);
    let min_r = (16384 / w2.max(1)).max(8);
    let mut out = vec![0.0f32; w2 * h2];
    par::for_rows_mut(&mut out, w2, min_r, |yr0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w2).enumerate() {
            let y = yr0 + dy;
            let y0 = (2 * y).min(p.h - 1);
            let y1 = (2 * y + 1).min(p.h - 1);
            let r0 = &b.d[y0 * p.w..(y0 + 1) * p.w];
            let r1 = &b.d[y1 * p.w..(y1 + 1) * p.w];
            for (x, o) in row.iter_mut().enumerate() {
                let x0 = (2 * x).min(p.w - 1);
                let x1 = (2 * x + 1).min(p.w - 1);
                *o = 0.25 * (r0[x0] + r0[x1] + r1[x0] + r1[x1]);
            }
        }
    });
    Plane::from_vec(w2, h2, out)
}

/// Bilinear upsample of a coarse flow component to (w, h), inverse of `downsample2`'s alignment,
/// multiplied by `mul` (2.0 for a pyramid step).
pub(crate) fn upsample_flow(c: &Plane, w: usize, h: usize, mul: f32) -> Plane {
    let sx = c.w as f32 / w as f32;
    let sy = c.h as f32 / h as f32;
    let xm = (c.w - 1) as f32;
    let ym = (c.h - 1) as f32;
    let x_taps: Vec<(usize, usize, f32)> = (0..w)
        .map(|x| {
            let xf = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, xm);
            let x0 = xf as usize;
            let x1 = (x0 + 1).min(c.w - 1);
            (x0, x1, xf - x0 as f32)
        })
        .collect();
    let min_r = (16384 / w.max(1)).max(8);
    let mut out = vec![0.0f32; w * h];
    par::for_rows_mut(&mut out, w, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let yf = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, ym);
            let cy0 = yf as usize;
            let cy1 = (cy0 + 1).min(c.h - 1);
            let ay = yf - cy0 as f32;
            let r0 = &c.d[cy0 * c.w..(cy0 + 1) * c.w];
            let r1 = &c.d[cy1 * c.w..(cy1 + 1) * c.w];
            for (x, o) in row.iter_mut().enumerate() {
                let (x0, x1, ax) = x_taps[x];
                let a = r0[x0] + ax * (r0[x1] - r0[x0]);
                let b = r1[x0] + ax * (r1[x1] - r1[x0]);
                *o = (a + ay * (b - a)) * mul;
            }
        }
    });
    Plane::from_vec(w, h, out)
}

/// Sobel gradients normalised to intensity units per pixel.
pub(crate) fn gradients(p: &Plane) -> (Plane, Plane) {
    let (w, h) = (p.w, p.h);
    let min_r = (16384 / w.max(1)).max(8);
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    par::for_rows_mut2(&mut gx, &mut gy, w, w, min_r, |y0, cx, cy| {
        for (dy, (rx, ry)) in cx.chunks_exact_mut(w).zip(cy.chunks_exact_mut(w)).enumerate() {
            let y = y0 + dy;
            let y_prev = y.saturating_sub(1);
            let y_next = (y + 1).min(h - 1);
            let rp = &p.d[y_prev * w..(y_prev + 1) * w];
            let rc = &p.d[y * w..(y + 1) * w];
            let rn = &p.d[y_next * w..(y_next + 1) * w];

            // x = 0
            {
                let x1 = 1.min(w - 1);
                let (a, b, c) = (rp[0], rp[0], rp[x1]);
                let (d, f) = (rc[0], rc[x1]);
                let (g, hh, i) = (rn[0], rn[0], rn[x1]);
                rx[0] = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) * 0.125;
                ry[0] = ((g + 2.0 * hh + i) - (a + 2.0 * b + c)) * 0.125;
            }
            if w > 2 {
                let mut x = 1usize;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let vtwo = vdupq_n_f32(2.0);
                    let vs = vdupq_n_f32(0.125);
                    let rpp = rp.as_ptr();
                    let rcp = rc.as_ptr();
                    let rnp = rn.as_ptr();
                    let rxp = rx.as_mut_ptr();
                    let ryp = ry.as_mut_ptr();
                    while x + 4 <= w - 1 {
                        let a = vld1q_f32(rpp.add(x - 1));
                        let b = vld1q_f32(rpp.add(x));
                        let c = vld1q_f32(rpp.add(x + 1));
                        let d = vld1q_f32(rcp.add(x - 1));
                        let f = vld1q_f32(rcp.add(x + 1));
                        let g = vld1q_f32(rnp.add(x - 1));
                        let hh = vld1q_f32(rnp.add(x));
                        let i = vld1q_f32(rnp.add(x + 1));
                        let gx_v = vmulq_f32(
                            vsubq_f32(vaddq_f32(vaddq_f32(c, vmulq_f32(vtwo, f)), i), vaddq_f32(vaddq_f32(a, vmulq_f32(vtwo, d)), g)),
                            vs,
                        );
                        let gy_v = vmulq_f32(
                            vsubq_f32(vaddq_f32(vaddq_f32(g, vmulq_f32(vtwo, hh)), i), vaddq_f32(vaddq_f32(a, vmulq_f32(vtwo, b)), c)),
                            vs,
                        );
                        vst1q_f32(rxp.add(x), gx_v);
                        vst1q_f32(ryp.add(x), gy_v);
                        x += 4;
                    }
                }
                while x < w - 1 {
                    let (a, b, c) = (rp[x - 1], rp[x], rp[x + 1]);
                    let (d, f) = (rc[x - 1], rc[x + 1]);
                    let (g, hh, i) = (rn[x - 1], rn[x], rn[x + 1]);
                    rx[x] = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) * 0.125;
                    ry[x] = ((g + 2.0 * hh + i) - (a + 2.0 * b + c)) * 0.125;
                    x += 1;
                }
            }
            if w > 1 {
                let x = w - 1;
                let (a, b, c) = (rp[x - 1], rp[x], rp[x]);
                let (d, f) = (rc[x - 1], rc[x]);
                let (g, hh, i) = (rn[x - 1], rn[x], rn[x]);
                rx[x] = ((c + 2.0 * f + i) - (a + 2.0 * d + g)) * 0.125;
                ry[x] = ((g + 2.0 * hh + i) - (a + 2.0 * b + c)) * 0.125;
            }
        }
    });
    (Plane::from_vec(w, h, gx), Plane::from_vec(w, h, gy))
}

/// Branchless 19-comparator median-of-9 network (exact for all finite floats).
#[inline(always)]
fn median9_scalar(mut p0: f32, mut p1: f32, mut p2: f32, mut p3: f32, mut p4: f32, mut p5: f32, mut p6: f32, mut p7: f32, mut p8: f32) -> f32 {
    macro_rules! cswap {
        ($a:expr, $b:expr) => {
            let lo = $a.min($b);
            let hi = $a.max($b);
            $a = lo;
            $b = hi;
        };
    }
    // Sort 3 rows of 3
    cswap!(p1, p2); cswap!(p4, p5); cswap!(p7, p8);
    cswap!(p0, p1); cswap!(p3, p4); cswap!(p6, p7);
    cswap!(p1, p2); cswap!(p4, p5); cswap!(p7, p8);
    // Max of mins -> p6
    p3 = p0.max(p3);
    p6 = p3.max(p6);
    // Min of maxes -> p2
    p5 = p5.min(p8);
    p2 = p2.min(p5);
    // Median of row medians (p1, p4, p7) -> p4
    cswap!(p1, p4);
    p4 = p4.min(p7);
    p4 = p1.max(p4);
    // Median of (p6, p4, p2) -> result
    cswap!(p6, p4);
    p4 = p4.min(p2);
    p6.max(p4)
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn median9_neon(
    mut p0: std::arch::aarch64::float32x4_t,
    mut p1: std::arch::aarch64::float32x4_t,
    mut p2: std::arch::aarch64::float32x4_t,
    mut p3: std::arch::aarch64::float32x4_t,
    mut p4: std::arch::aarch64::float32x4_t,
    mut p5: std::arch::aarch64::float32x4_t,
    mut p6: std::arch::aarch64::float32x4_t,
    mut p7: std::arch::aarch64::float32x4_t,
    mut p8: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    use std::arch::aarch64::*;
    unsafe {
        macro_rules! cswap {
            ($a:expr, $b:expr) => {
                let lo = vminq_f32($a, $b);
                let hi = vmaxq_f32($a, $b);
                $a = lo;
                $b = hi;
            };
        }
        cswap!(p1, p2); cswap!(p4, p5); cswap!(p7, p8);
        cswap!(p0, p1); cswap!(p3, p4); cswap!(p6, p7);
        cswap!(p1, p2); cswap!(p4, p5); cswap!(p7, p8);
        p3 = vmaxq_f32(p0, p3);
        p6 = vmaxq_f32(p3, p6);
        p5 = vminq_f32(p5, p8);
        p2 = vminq_f32(p2, p5);
        cswap!(p1, p4);
        p4 = vminq_f32(p4, p7);
        p4 = vmaxq_f32(p1, p4);
        cswap!(p6, p4);
        p4 = vminq_f32(p4, p2);
        vmaxq_f32(p6, p4)
    }
}

/// 3x3 median filter (clamped borders).
pub(crate) fn median3(p: &Plane) -> Plane {
    let (w, h) = (p.w, p.h);
    let min_r = (8192 / w.max(1)).max(8);
    let mut out = vec![0.0f32; w * h];
    par::for_rows_mut(&mut out, w, min_r, |y0, chunk| {
        for (dy, row) in chunk.chunks_exact_mut(w).enumerate() {
            let y = y0 + dy;
            let yp = y.saturating_sub(1);
            let yn = (y + 1).min(h - 1);
            let rp = &p.d[yp * w..(yp + 1) * w];
            let rc = &p.d[y * w..(y + 1) * w];
            let rn = &p.d[yn * w..(yn + 1) * w];

            // x = 0
            {
                let x1 = 1.min(w - 1);
                row[0] = median9_scalar(rp[0], rp[0], rp[x1], rc[0], rc[0], rc[x1], rn[0], rn[0], rn[x1]);
            }
            if w > 2 {
                let mut x = 1usize;
                #[cfg(target_arch = "aarch64")]
                unsafe {
                    use std::arch::aarch64::*;
                    let rpp = rp.as_ptr();
                    let rcp = rc.as_ptr();
                    let rnp = rn.as_ptr();
                    let outp = row.as_mut_ptr();
                    while x + 4 <= w - 1 {
                        let m = median9_neon(
                            vld1q_f32(rpp.add(x - 1)),
                            vld1q_f32(rpp.add(x)),
                            vld1q_f32(rpp.add(x + 1)),
                            vld1q_f32(rcp.add(x - 1)),
                            vld1q_f32(rcp.add(x)),
                            vld1q_f32(rcp.add(x + 1)),
                            vld1q_f32(rnp.add(x - 1)),
                            vld1q_f32(rnp.add(x)),
                            vld1q_f32(rnp.add(x + 1)),
                        );
                        vst1q_f32(outp.add(x), m);
                        x += 4;
                    }
                }
                while x < w - 1 {
                    row[x] = median9_scalar(
                        rp[x - 1], rp[x], rp[x + 1],
                        rc[x - 1], rc[x], rc[x + 1],
                        rn[x - 1], rn[x], rn[x + 1],
                    );
                    x += 1;
                }
            }
            if w > 1 {
                let x = w - 1;
                row[x] = median9_scalar(
                    rp[x - 1], rp[x], rp[x],
                    rc[x - 1], rc[x], rc[x],
                    rn[x - 1], rn[x], rn[x],
                );
            }
        }
    });
    Plane::from_vec(w, h, out)
}

/// Deterministic sum: per-row sequential sums, then a sequential sum of rows.
pub(crate) fn det_sum<F>(h: usize, f: F) -> f64
where
    F: Fn(usize) -> f64 + Sync + Send,
{
    let mut rows = vec![0.0f64; h];
    par::for_rows_mut(&mut rows, 1, 32, |y0, chunk| {
        for (dy, slot) in chunk.iter_mut().enumerate() {
            *slot = f(y0 + dy);
        }
    });
    rows.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median9_network_exact_on_all_permutations() {
        let mut perm = [0u8, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut count = 0usize;
        loop {
            let m = median9_scalar(
                perm[0] as f32, perm[1] as f32, perm[2] as f32,
                perm[3] as f32, perm[4] as f32, perm[5] as f32,
                perm[6] as f32, perm[7] as f32, perm[8] as f32,
            );
            assert_eq!(m, 4.0, "failed on {perm:?}");
            count += 1;
            // Next lexicographical permutation
            let mut i = 8;
            while i > 0 && perm[i - 1] >= perm[i] {
                i -= 1;
            }
            if i == 0 {
                break;
            }
            let mut j = 8;
            while perm[j] <= perm[i - 1] {
                j -= 1;
            }
            perm.swap(i - 1, j);
            perm[i..].reverse();
        }
        assert_eq!(count, 362_880);
    }

    #[test]
    fn plane_filters_handle_small_and_odd_dimensions() {
        for &(w, h) in &[(1usize, 1usize), (2, 2), (3, 3), (4, 2), (5, 3), (6, 4), (7, 5), (8, 8), (9, 5), (15, 7), (17, 9)] {
            let data: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 101) as f32 * 0.01).collect();
            let p = Plane::from_vec(w, h, data);
            let b0 = gaussian_blur(&p, 0.7);
            let b1 = gaussian_blur(&p, 4.0);
            let (gx, gy) = gradients(&p);
            let m = median3(&p);
            assert_eq!(b0.d.len(), w * h);
            assert_eq!(b1.d.len(), w * h);
            assert_eq!(gx.d.len(), w * h);
            assert_eq!(gy.d.len(), w * h);
            assert_eq!(m.d.len(), w * h);
            assert!(b0.d.iter().all(|v| v.is_finite()));
            assert!(b1.d.iter().all(|v| v.is_finite()));
            assert!(gx.d.iter().all(|v| v.is_finite()));
            assert!(gy.d.iter().all(|v| v.is_finite()));
            assert!(m.d.iter().all(|v| v.is_finite()));
        }
    }
}
