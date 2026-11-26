//! T Blur core: optical-flow-driven motion blur.
//!
//! Pipeline (see the module docs for details):
//! 1. `field`: luma at analysis resolution, exposure-normalised; DIS optical flow (`dis`) to each
//!    neighbour; outlier removal; contrast-relative confidence; scene-cut rejection; one velocity per
//!    pixel; confidence-weighted edge-aware smoothing; Motion Sensitivity gate and dead zone.
//! 2. `blur`: guided upsampling of the velocity to full resolution, tile-max / neighbour-max, and an
//!    energy-conserving gather along the dominant (and, if different, the pixel's own) motion.

mod blur;
mod color;
mod dis;
mod field;
pub mod par;
mod plane;
mod pool;
mod viz;
#[cfg(test)]
mod tests;

use std::time::Instant;

pub use field::MIN_FRAME_DISTANCE;

/// Full-colour image, borrowed. Premultiplied RGBA, f32, interleaved (R,G,B,A), row-major, top-left
/// origin, no row padding: data.len() == width*height*4. Colour is display-referred (gamma-encoded,
/// as the host delivers it), nominally 0..=1 but may exceed 1 for HDR/float footage. Finite values only.
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [f32],
}

/// Single-channel gamma-encoded luma plane, borrowed. data.len() == width*height. Same size as the current frame.
#[derive(Clone, Copy)]
pub struct Luma<'a> {
    pub width: usize,
    pub height: usize,
    pub data: &'a [f32],
}

/// Rec.709 weights used wherever a luma plane is built: Y = 0.2126 R + 0.7152 G + 0.0722 B
/// (on the gamma-encoded, premultiplied values).
pub const LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

#[derive(Clone, Copy)]
pub struct Neighbor<'a> {
    pub luma: Luma<'a>,
    /// Output-frame intervals between the current frame and this neighbour, > 0. Core divides the
    /// measured displacement by this to get motion per output frame interval (the unit Blur
    /// Amount refers to). Values below 1 are valid: a neighbour 0.2 output frames away (the
    /// adjacent source frame of 120 fps footage on a 24 fps timeline) has its displacement
    /// multiplied by 5. Values below `MIN_FRAME_DISTANCE` (1/64) are raised to it; a non-finite
    /// value makes the neighbour unusable.
    pub frame_distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quality {
    Draft,
    Normal,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Output,
    MotionVectors,
}

#[derive(Clone, Copy, Debug)]
pub struct Params {
    /// Shutter-open time as a fraction of one frame interval. 0.5 = 180 degree shutter, 1.0 = 360.
    /// Range 0.0..=4.0. 0 => exact passthrough.
    pub blur_amount: f32,
    /// 0.0..=100.0, default 70. 0 => no blur. Higher trusts finer / smaller motion.
    ///
    /// Exact mapping, with t = sensitivity / 100:
    /// * edge-aware smoothing of the vector field, spatial sigma in analysis pixels:
    ///   `2 + 28 (1-t)^1.5` (100 -> 2, 70 -> 6.6, 0 -> 30);
    /// * confidence gate on the locally averaged match confidence (0..1): no motion below
    ///   `c_lo = 0.6 (1-t)`, full motion above `c_lo + 0.03 + 0.25 (1-t)` (100 -> 0.00..0.03,
    ///   70 -> 0.18..0.285), smoothstep in between;
    /// * dead zone: vector length is soft-thresholded by `0.03 + 1.2 (1-t)^2` analysis pixels per
    ///   frame (100 -> 0.03, 70 -> 0.14, 0 -> 1.23).
    ///
    /// All three terms are monotonic, so lowering the value never adds blur.
    pub motion_sensitivity: f32,
    /// Maximum blur streak length as a percentage of the frame's longer side. 0.0..=50.0, default 15.
    pub max_blur_percent: f32,
    pub quality: Quality,
    /// Average colours in linear light (default true) rather than on gamma-encoded values.
    pub linear_light: bool,
    pub view: View,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            blur_amount: 0.5,
            motion_sensitivity: 70.0,
            max_blur_percent: 15.0,
            quality: Quality::Normal,
            linear_light: true,
            view: View::Output,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RenderStats {
    pub flow_ms: f32,
    pub blur_ms: f32,
    pub used_prev: bool,
    pub used_next: bool,
    pub max_velocity_px: f32,
}

fn copy_through(src: &[f32], out: &mut [f32]) {
    let n = out.len().min(src.len());
    out[..n].copy_from_slice(&src[..n]);
}

/// Writes the motion-blurred `cur` into `out` (same layout and length as cur.data). Never panics on
/// valid input. With no usable neighbour, zero blur_amount or zero sensitivity, `out` is a bit-exact
/// copy of cur.data.
pub fn render(cur: Frame, prev: Option<Neighbor>, next: Option<Neighbor>, params: &Params, out: &mut [f32]) -> RenderStats {
    if !preflight(cur, prev, next, params, out) {
        copy_through(cur.data, out);
        return RenderStats::default();
    }
    let l = luma_from_rgba(cur);
    render_with_luma(cur, Luma { width: cur.width, height: cur.height, data: &l }, prev, next, params, out)
}

/// Cheap checks shared by both entry points: true when there is any chance of doing work.
fn preflight(cur: Frame, prev: Option<Neighbor>, next: Option<Neighbor>, params: &Params, out: &[f32]) -> bool {
    let n = cur.width.checked_mul(cur.height).and_then(|v| v.checked_mul(4));
    n == Some(cur.data.len())
        && out.len() == cur.data.len()
        && cur.width >= 2
        && cur.height >= 2
        && params.blur_amount.is_finite()
        && params.blur_amount > 0.0
        && params.motion_sensitivity.is_finite()
        && params.motion_sensitivity > 0.0
        && (prev.is_some() || next.is_some())
}

/// Same as `render`, except that motion is estimated from `cur_luma` instead of a luma derived from
/// `cur`. `cur_luma` has the same dimensions as `cur`. Colour for the blur still comes from `cur`.
pub fn render_with_luma(
    cur: Frame,
    cur_luma: Luma,
    prev: Option<Neighbor>,
    next: Option<Neighbor>,
    params: &Params,
    out: &mut [f32],
) -> RenderStats {
    let _render_guard = par::enter_render();
    let mut st = RenderStats::default();
    if !preflight(cur, prev, next, params, out)
        || cur_luma.width != cur.width
        || cur_luma.height != cur.height
        || cur_luma.data.len() != cur.width * cur.height
    {
        copy_through(cur.data, out);
        return st;
    }
    let t0 = Instant::now();
    let an = field::analyze(cur_luma, prev, next, params);
    st.flow_ms = t0.elapsed().as_secs_f32() * 1000.0;
    let Some(an) = an else {
        copy_through(cur.data, out);
        return st;
    };
    finish_blur(cur, cur_luma.data, an, params, out, &mut st, None);
    st
}

/// Same as `render_with_luma`, but takes ownership of the luma buffers so neighbour lumas are
/// freed immediately after optical flow and `cur_luma` is freed right after `build_hfield`,
/// reducing peak memory during full-resolution blur (by 96 MB per concurrent 4K frame) with
/// identical output and no speed cost.
pub fn render_with_owned_luma(
    cur: Frame,
    cur_luma: Vec<f32>,
    prev: Option<(Vec<f32>, f32)>,
    next: Option<(Vec<f32>, f32)>,
    params: &Params,
    out: &mut [f32],
) -> RenderStats {
    let _render_guard = par::enter_render();
    let mut st = RenderStats::default();
    let (w, h) = (cur.width, cur.height);
    let cur_l = Luma { width: w, height: h, data: &cur_luma };
    let p_nb = prev.as_ref().map(|(d, fd)| Neighbor { luma: Luma { width: w, height: h, data: d }, frame_distance: *fd });
    let n_nb = next.as_ref().map(|(d, fd)| Neighbor { luma: Luma { width: w, height: h, data: d }, frame_distance: *fd });
    if !preflight(cur, p_nb, n_nb, params, out) || cur_luma.len() != w * h {
        copy_through(cur.data, out);
        return st;
    }
    let t0 = Instant::now();
    let an = field::analyze(cur_l, p_nb, n_nb, params);
    st.flow_ms = t0.elapsed().as_secs_f32() * 1000.0;
    // Drop neighbour lumas before full-resolution blur allocates HField / linear buffers.
    drop(prev);
    drop(next);
    let Some(an) = an else {
        copy_through(cur.data, out);
        return st;
    };
    finish_blur(cur, &[], an, params, out, &mut st, Some(cur_luma));
    st
}

fn finish_blur(
    cur: Frame,
    cur_luma_ref: &[f32],
    an: field::Analysis,
    params: &Params,
    out: &mut [f32],
    st: &mut RenderStats,
    mut owned_cur_luma: Option<Vec<f32>>,
) {
    let (w, h) = (cur.width, cur.height);
    st.used_prev = an.used_prev;
    st.used_next = an.used_next;
    let (mx, my) = (w as f32 / an.wa as f32, h as f32 / an.ha as f32);
    let mut mv2 = 0.0f32;
    for (&u, &v) in an.vu.d.iter().zip(&an.vv.d) {
        let m = (u * mx) * (u * mx) + (v * my) * (v * my);
        if m > mv2 {
            mv2 = m;
        }
    }
    st.max_velocity_px = mv2.sqrt();

    let t1 = Instant::now();
    match params.view {
        View::MotionVectors => {
            let luma_slice = owned_cur_luma.as_deref().unwrap_or(cur_luma_ref);
            let up = blur::Upsampler::new(&an, luma_slice, w, h);
            let wheel = viz::Wheel::new();
            let scale = 1.0 / (viz::VIZ_FULL_SCALE * w.max(h) as f32);
            par::for_rows_mut(out, w * 4, 16, |y0, chunk| {
                for (dy, row) in chunk.chunks_exact_mut(w * 4).enumerate() {
                    let y = y0 + dy;
                    for x in 0..w {
                        let v = up.vel(x, y);
                        let c = wheel.color(v[0] * scale, v[1] * scale);
                        row[x * 4..x * 4 + 4].copy_from_slice(&[c[0], c[1], c[2], 1.0]);
                    }
                }
            });
        }
        View::Output => {
            if st.max_velocity_px * params.blur_amount * 0.5 < 0.5 {
                copy_through(cur.data, out);
            } else {
                let hmax = 0.5 * params.max_blur_percent.clamp(0.0, 50.0) / 100.0 * w.max(h) as f32;
                let hf = {
                    let luma_slice = owned_cur_luma.as_deref().unwrap_or(cur_luma_ref);
                    let up = blur::Upsampler::new(&an, luma_slice, w, h);
                    blur::build_hfield(&up, w, h, params.blur_amount.min(4.0) * 0.5, hmax)
                };
                // Free analysis planes and owned current luma before allocating `lin` and gathering.
                drop(an);
                drop(owned_cur_luma.take());
                if params.linear_light {
                    let mut lin = pool::PooledF32::take_dirty(cur.data.len());
                    par::for_rows_mut(&mut lin.0, w * 4, 16, |y0, dst| {
                        let off = y0 * w * 4;
                        color::decode_rgba_chunk(&cur.data[off..off + dst.len()], dst);
                    });
                    blur::gather(&hf, &lin.0, cur.data, params.quality, out, color::px_encode);
                } else {
                    blur::gather(&hf, cur.data, cur.data, params.quality, out, |_| {});
                }
            }
        }
    }
    st.blur_ms = t1.elapsed().as_secs_f32() * 1000.0;
}

/// Builds a luma plane from a Frame.
pub fn luma_from_rgba(frame: Frame) -> Vec<f32> {
    let [a, b, c] = LUMA_WEIGHTS;
    let n = (frame.width * frame.height).min(frame.data.len() / 4);
    let mut out = vec![0.0f32; n];
    par::for_rows_mut(&mut out, 4096, 8, |k0, chunk| {
        let base = k0 * 4096;
        for (i, v) in chunk.iter_mut().enumerate() {
            let p = &frame.data[(base + i) * 4..(base + i) * 4 + 3];
            *v = a * p[0] + b * p[1] + c * p[2];
        }
    });
    out
}

/// True when `b` is a repeat of `a` (held / duplicated frame): subsampled mean absolute luma
/// difference below a very small threshold (bit-identical or near it; ordinary sensor noise on a
/// static shot must NOT count as duplicate).
///
/// Threshold: mean |a-b| over a ~64k-sample regular subsample below 2e-4 (about 1/20 of an 8-bit
/// code value). Mismatched sizes are never duplicates.
pub fn is_duplicate_frame(a: Luma, b: Luma) -> bool {
    if a.width != b.width || a.height != b.height || a.data.len() != b.data.len() || a.data.len() != a.width * a.height {
        return false;
    }
    if a.data.is_empty() {
        return true;
    }
    let n = a.data.len();
    let step = (n / 65_536).max(1);
    // Odd stride so the subsample does not lock onto columns.
    let step = if step > 1 && step.is_multiple_of(2) { step + 1 } else { step };
    let mut s = 0.0f64;
    let mut k = 0usize;
    let mut i = 0usize;
    while i < n {
        s += (a.data[i] - b.data[i]).abs() as f64;
        k += 1;
        i += step;
    }
    s / (k as f64) < 2e-4
}
