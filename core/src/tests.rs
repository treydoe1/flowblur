//! Synthetic-scene tests. Run with `cargo test --release -- --nocapture` to see the measured numbers.

use crate::*;

/// Band-limited analytic texture (sum of sinusoids), so sub-pixel shifts are exact.
struct Tex {
    waves: Vec<(f32, f32, f32, f32)>, // kx, ky, phase, amp
}

impl Tex {
    fn new(seed: u64) -> Self {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f32 / (1u64 << 53) as f32
        };
        // Log-spaced frequencies (1/f-like spectrum, as in natural images) from 1/250 to 1/4 cycles/px.
        let waves = (0..32)
            .map(|_| {
                let f = 0.004 * (62.5f32).powf(rnd());
                let a = rnd() * std::f32::consts::TAU;
                let k = f * std::f32::consts::TAU;
                (k * a.cos(), k * a.sin(), rnd() * std::f32::consts::TAU, 0.3 + rnd())
            })
            .collect::<Vec<_>>();
        let norm: f32 = waves.iter().map(|w| w.3).sum();
        Tex { waves: waves.into_iter().map(|(a, b, c, d)| (a, b, c, d / norm)).collect() }
    }
    fn at(&self, x: f32, y: f32) -> f32 {
        let mut v = 0.0;
        for &(kx, ky, p, a) in &self.waves {
            v += a * (kx * x + ky * y + p).sin();
        }
        0.5 + 0.6 * v
    }
}

fn grey_frame(w: usize, h: usize, f: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let mut d = vec![0.0f32; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let v = f(x as f32, y as f32);
            d[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[v, v, v, 1.0]);
        }
    }
    d
}

fn luma(w: usize, h: usize, d: &[f32]) -> Vec<f32> {
    luma_from_rgba(Frame { width: w, height: h, data: d })
}

fn nb<'a>(w: usize, h: usize, l: &'a [f32], fd: f32) -> Neighbor<'a> {
    Neighbor { luma: Luma { width: w, height: h, data: l }, frame_distance: fd }
}

fn analyze(w: usize, h: usize, cur: &[f32], prev: Option<(&[f32], f32)>, next: Option<(&[f32], f32)>, p: &Params) -> Option<field::Analysis> {
    field::analyze(Luma { width: w, height: h, data: cur }, prev.map(|(l, fd)| nb(w, h, l, fd)), next.map(|(l, fd)| nb(w, h, l, fd)), p)
}

fn sens(s: f32) -> Params {
    Params { motion_sensitivity: s, ..Params::default() }
}

/// Mean endpoint error of the analysis velocity against `truth`, over pixels where `mask` holds.
fn epe(an: &field::Analysis, truth: impl Fn(usize, usize) -> [f32; 2], mask: impl Fn(usize, usize) -> bool) -> f32 {
    let (mut s, mut n) = (0.0f64, 0usize);
    for y in 0..an.ha {
        for x in 0..an.wa {
            if !mask(x, y) {
                continue;
            }
            let t = truth(x, y);
            let i = y * an.wa + x;
            let (du, dv) = (an.vu.d[i] - t[0], an.vv.d[i] - t[1]);
            s += ((du * du + dv * dv) as f64).sqrt();
            n += 1;
        }
    }
    (s / n.max(1) as f64) as f32
}

const W: usize = 640;
const H: usize = 360;

#[test]
fn global_translation_epe() {
    let t = Tex::new(1);
    let cur = luma(W, H, &grey_frame(W, H, |x, y| t.at(x, y)));
    let mut worst_rel: f32 = 0.0;
    for d in [[0.3f32, 0.2], [1.5, -0.7], [5.0, 3.0], [20.0, -8.0], [60.0, 25.0], [96.0, 0.0]] {
        let prev = luma(W, H, &grey_frame(W, H, |x, y| t.at(x + d[0], y + d[1])));
        let next = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - d[0], y - d[1])));
        let border = (d[0].abs().max(d[1].abs()) as usize) + 16;
        let mask = |x: usize, y: usize| x >= border && y >= border && x + border < W && y + border < H;
        let mut line = format!("translation ({:5.1},{:5.1}) |d|={:6.2}:", d[0], d[1], (d[0] * d[0] + d[1] * d[1]).sqrt());
        for s in [100.0, 70.0] {
            let both = analyze(W, H, &cur, Some((&prev, 1.0)), Some((&next, 1.0)), &sens(s)).unwrap();
            let e = epe(&both, |_, _| d, mask);
            line += &format!("  EPE sens{} {:.3}", s as i32, e);
            if s == 100.0 {
                let mag = (d[0] * d[0] + d[1] * d[1]).sqrt();
                worst_rel = worst_rel.max(e / mag.max(1.0));
                assert!(e < 0.1 + 0.02 * mag, "EPE {e} too large for {d:?}");
            }
        }
        println!("{line}");
    }
    println!("worst relative EPE (sens 100): {worst_rel:.4}");
}

/// Textured 160x160 object moving by `d` per frame over a static textured background.
fn object_scene(d: [f32; 2]) -> (Vec<f32>, Vec<f32>, Vec<f32>, impl Fn(f32, f32) -> bool + Copy) {
    let bg = Tex::new(2);
    let ob = Tex::new(3);
    let (ox, oy, sz) = (240.0f32, 100.0f32, 160.0f32);
    let frame = |k: f32| {
        grey_frame(W, H, |x, y| {
            let (px, py) = (ox + k * d[0], oy + k * d[1]);
            if x >= px && x < px + sz && y >= py && y < py + sz {
                // object texture is brighter and moves with the object
                0.25 + 0.75 * ob.at(x - px, y - py)
            } else {
                0.75 * bg.at(x, y)
            }
        })
    };
    let inside = move |x: f32, y: f32| x >= ox && x < ox + sz && y >= oy && y < oy + sz;
    (frame(-1.0), frame(0.0), frame(1.0), inside)
}

#[test]
fn object_over_background() {
    let d = [8.0f32, 3.0];
    let (p, c, n, inside) = object_scene(d);
    let (lp, lc, ln) = (luma(W, H, &p), luma(W, H, &c), luma(W, H, &n));
    for s in [100.0, 70.0, 30.0] {
    let an = analyze(W, H, &lc, Some((&lp, 1.0)), Some((&ln, 1.0)), &sens(s)).unwrap();
    println!("-- sensitivity {s}");
    let (ox, oy, sz) = (240usize, 100usize, 160usize);
    let dist_out = |x: usize, y: usize| -> f32 {
        let dx = (ox as f32 - x as f32).max(x as f32 - (ox + sz - 1) as f32).max(0.0);
        let dy = (oy as f32 - y as f32).max(y as f32 - (oy + sz - 1) as f32).max(0.0);
        (dx * dx + dy * dy).sqrt()
    };
    let in_obj = epe(&an, |_, _| d, |x, y| x >= ox + 12 && x < ox + sz - 12 && y >= oy + 12 && y < oy + sz - 12);
    let in_bg = epe(&an, |_, _| [0.0, 0.0], |x, y| !inside(x as f32, y as f32) && dist_out(x, y) >= 12.0 && x > 16 && y > 16 && x < W - 16 && y < H - 16);
    println!("object EPE (>=12px inside): {in_obj:.3} px, background EPE (>=12px outside): {in_bg:.3} px");
    // Bleed profile: mean |v| in rings outside the silhouette.
    let mut prof = String::new();
    let mut bleed = 0;
    for r in 1..=16 {
        let (mut s, mut k) = (0.0f32, 0);
        for y in 0..H {
            for x in 0..W {
                let dd = dist_out(x, y);
                if dd > (r - 1) as f32 && dd <= r as f32 {
                    let i = y * W + x;
                    s += (an.vu.d[i].powi(2) + an.vv.d[i].powi(2)).sqrt();
                    k += 1;
                }
            }
        }
        let m = s / k as f32;
        if m > 0.1 * 8.54 {
            bleed = r;
        }
        prof += &format!(" {r}:{m:.2}");
    }
    // And the inner side: does the object's motion hold up to its silhouette?
    let mut inner = String::new();
    for r in 0..6usize {
        let (mut s, mut k) = (0.0f32, 0);
        for y in oy..oy + sz {
            for x in ox..ox + sz {
                let e = (x - ox).min(ox + sz - 1 - x).min(y - oy).min(oy + sz - 1 - y);
                if e == r {
                    let i = y * W + x;
                    s += (an.vu.d[i].powi(2) + an.vv.d[i].powi(2)).sqrt();
                    k += 1;
                }
            }
        }
        inner += &format!(" {r}:{:.2}", s / k as f32);
    }
    println!("mean |v| (true 8.54) in 1-px rings outside silhouette:{prof}");
    println!("mean |v| in rings inside silhouette (0 = edge pixel):{inner}");
    println!("bleed distance (|v| > 10% of object speed): {bleed} px");
    // The sensitivity dead zone shortens every vector by `dead`, by design.
    assert!(in_obj < field::sens_map(s).dead + 0.3 && in_bg < 0.2);
    assert!(bleed <= 4, "vectors bleed {bleed} px past the silhouette");
    }
}

#[test]
fn static_scene_is_bit_exact() {
    let t = Tex::new(4);
    let c = grey_frame(W, H, |x, y| t.at(x, y));
    let l = luma(W, H, &c);
    let mut out = vec![0.0; c.len()];
    let st = render(Frame { width: W, height: H, data: &c }, Some(nb(W, H, &l, 1.0)), Some(nb(W, H, &l, 1.0)), &Params::default(), &mut out);
    assert!(st.used_prev && st.used_next);
    assert!(out.iter().zip(&c).all(|(a, b)| a.to_bits() == b.to_bits()), "static scene not bit-exact");
    println!("static: max velocity {:.4} px", st.max_velocity_px);
}

#[test]
fn trivial_passthroughs_are_bit_exact() {
    let t = Tex::new(5);
    let c = grey_frame(W, H, |x, y| t.at(x, y));
    let n = grey_frame(W, H, |x, y| t.at(x - 10.0, y));
    let ln = luma(W, H, &n);
    let f = Frame { width: W, height: H, data: &c };
    let exact = |p: Params, nx: Option<Neighbor>| {
        let mut out = vec![1.0; c.len()];
        render(f, None, nx, &p, &mut out);
        out.iter().zip(&c).all(|(a, b)| a.to_bits() == b.to_bits())
    };
    assert!(exact(Params { blur_amount: 0.0, ..Params::default() }, Some(nb(W, H, &ln, 1.0))));
    assert!(exact(Params { motion_sensitivity: 0.0, ..Params::default() }, Some(nb(W, H, &ln, 1.0))));
    assert!(exact(Params::default(), None));
    assert!(!exact(Params::default(), Some(nb(W, H, &ln, 1.0))));
    // Malformed input never panics.
    let mut short = vec![0.0; 10];
    render(f, None, Some(nb(W, H, &ln, 1.0)), &Params::default(), &mut short);
    render(f, None, Some(nb(W - 1, H, &ln[..(W - 1) * H], 1.0)), &Params::default(), &mut vec![0.0; c.len()]);
    render(Frame { width: 3, height: 3, data: &c[..36] }, None, Some(nb(3, 3, &ln[..9], 1.0)), &Params::default(), &mut [0.0; 36]);
}

/// Row profile of a vertical step edge moving horizontally by `d` px/frame.
fn step_profile(d: f32, blur: f32, linear: bool) -> (Vec<f32>, Vec<f32>) {
    let t = Tex::new(6);
    let e = W as f32 / 2.0;
    let t = &t;
    let img = |s: f32| {
        move |x: f32, _y: f32| {
            let xs = x - s;
            let base = if xs >= e { 0.8 } else { 0.2 };
            // weak x-only texture away from the edge keeps the flow well posed everywhere
            let tex = if (xs - e).abs() > 60.0 { 0.06 * (t.at(xs, 0.0) - 0.5) } else { 0.0 };
            base + tex
        }
    };
    let c = grey_frame(W, H, img(0.0));
    let p = luma(W, H, &grey_frame(W, H, img(-d)));
    let n = luma(W, H, &grey_frame(W, H, img(d)));
    let mut out = vec![0.0; c.len()];
    let prm = Params { blur_amount: blur, linear_light: linear, ..Params::default() };
    render(Frame { width: W, height: H, data: &c }, Some(nb(W, H, &p, 1.0)), Some(nb(W, H, &n, 1.0)), &prm, &mut out);
    let prof = |buf: &[f32]| -> Vec<f32> {
        (0..W)
            .map(|x| {
                let mut s = 0.0;
                for y in 100..260 {
                    let v = buf[(y * W + x) * 4];
                    s += if linear { color::decode(v) } else { v };
                }
                s / 160.0
            })
            .collect()
    };
    (prof(&c), prof(&out))
}

#[test]
fn step_edge_blur_length_and_energy() {
    for (d, blur, linear) in [(20.0, 0.5, false), (20.0, 1.0, false), (12.0, 2.0, false), (20.0, 1.0, true)] {
        let (pin, pout) = step_profile(d, blur, linear);
        let lo = pin[W / 2 - 50];
        let hi = pin[W / 2 + 50];
        let f = |v: f32| (v - lo) / (hi - lo);
        let x10 = (W / 2 - 50..W / 2 + 50).find(|&x| f(pout[x]) >= 0.1).unwrap();
        let x90 = (W / 2 - 50..W / 2 + 50).find(|&x| f(pout[x]) >= 0.9).unwrap();
        let w1090 = (x90 - x10) as f32;
        let expect_len = blur * d;
        let measured_len = w1090 / 0.8;
        let range = W / 2 - 55..W / 2 + 55;
        let mi: f32 = pin[range.clone()].iter().sum::<f32>() / 110.0;
        let mo: f32 = pout[range].iter().sum::<f32>() / 110.0;
        println!(
            "step d={d} blur={blur} linear={linear}: expected streak {expect_len:.1} px, measured {measured_len:.1} px (10-90 {w1090} px); mean in {mi:.4} out {mo:.4} ({:+.2}%)",
            100.0 * (mo - mi) / mi
        );
        assert!((measured_len - expect_len).abs() <= 0.15 * expect_len + 1.5, "streak length off");
        assert!(((mo - mi) / mi).abs() < 0.01, "brightness not conserved");
    }
}

/// The three physical cases on an injected velocity field (no flow involved).
#[test]
fn blur_physics_cases() {
    let (w, h) = (240usize, 48usize);
    let hl = 10.0f32;
    for q in [Quality::Normal, Quality::High] {
        // Object pixels x in [100, 140) colour 1 moving +x with half-extent hl, static background 0.
        let mut col = vec![0.0f32; w * h * 4];
        let mut hv = vec![[0.0f32; 2]; w * h];
        for y in 0..h {
            for x in 100..140 {
                col[(y * w + x) * 4] = 1.0;
                hv[y * w + x] = [hl, 0.0];
            }
            for x in 0..w {
                col[(y * w + x) * 4 + 3] = 1.0;
            }
        }
        let f = blur::hfield_from(w, h, hv);
        let mut out = vec![0.0; col.len()];
        blur::gather(&f, &col, &col, q, &mut out, |_| {});
        let at = |x: usize| (12..36).map(|y| out[(y * w + x) * 4]).sum::<f32>() / 24.0;
        let mut worst = 0.0f32;
        let mut s = String::new();
        for k in 1..=10usize {
            // background ahead: pixel 139+k, distance d = k - 0.5 from the leading edge
            let dd = k as f32 - 0.5;
            let expect = ((hl - dd) / (2.0 * hl)).max(0.0);
            let got = at(139 + k);
            worst = worst.max((got - expect).abs());
            // just inside trailing edge: pixel 100 + k - 1, background weight (H-d)/2H
            let got_t = 1.0 - at(100 + k - 1);
            worst = worst.max((got_t - expect).abs());
            // background behind trailing edge: pixel 100 - k gets object colour (H-d)/2H
            let got_b = at(100 - k);
            worst = worst.max((got_b - expect).abs());
            s += &format!(" d={dd}: ahead {got:.3} / inside-trailing {got_t:.3} / behind {got_b:.3} (expect {expect:.3});");
        }
        println!("{q:?}:{s}");
        println!("{q:?}: worst abs error vs (H-d)/2H = {worst:.3}");
        assert!(worst < 0.06);

        // Uniform translation of a sinusoid: W=1 and a box filter of length 2H.
        let per = 37.0f32;
        let mut col = vec![0.0f32; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let v = 0.5 + 0.4 * (std::f32::consts::TAU * x as f32 / per).sin();
                col[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&[v, v, v, 1.0]);
            }
        }
        let f = blur::hfield_from(w, h, vec![[hl, 0.0]; w * h]);
        let mut out = vec![0.0; col.len()];
        blur::gather(&f, &col, &col, q, &mut out, |_| {});
        let len = 2.0 * hl;
        let a = std::f32::consts::PI * len / per;
        let atten = a.sin() / a;
        let (mut err, mut n) = (0.0f32, 0);
        for y in 10..38 {
            for x in 40..200 {
                let expect = 0.5 + 0.4 * atten * (std::f32::consts::TAU * x as f32 / per).sin();
                err += (out[(y * w + x) * 4] - expect).abs();
                n += 1;
            }
        }
        println!("{q:?}: uniform translation box-blur mean abs error {:.4}", err / n as f32);
        assert!(err / (n as f32) < 0.02);
    }
}

#[test]
fn scene_cut_and_one_sided() {
    let t = Tex::new(7);
    let u = Tex::new(99);
    let d = [6.0f32, -2.0];
    let c = grey_frame(W, H, |x, y| t.at(x, y));
    let lc = luma(W, H, &c);
    let lp = luma(W, H, &grey_frame(W, H, |x, y| t.at(x + d[0], y + d[1])));
    let ln = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - d[0], y - d[1])));
    let lcut = luma(W, H, &grey_frame(W, H, |x, y| u.at(x * 1.3, y * 0.8)));
    let p = Params::default();
    let mask = |x: usize, y: usize| x >= 24 && y >= 24 && x + 24 < W && y + 24 < H;
    // One-sided, each direction.
    let a = analyze(W, H, &lc, Some((&lp, 1.0)), None, &p).unwrap();
    let b = analyze(W, H, &lc, None, Some((&ln, 1.0)), &p).unwrap();
    let ea = epe(&a, |_, _| d, mask);
    let eb = epe(&b, |_, _| d, mask);
    println!("one-sided prev-only EPE {ea:.3}, next-only EPE {eb:.3} (|d| = 6.32, includes 0.14 px dead zone)");
    assert!(a.used_prev && !a.used_next && b.used_next && !b.used_prev);
    assert!(ea < 0.3 && eb < 0.3);
    // Scene cut on the next side.
    let cut = analyze(W, H, &lc, Some((&lp, 1.0)), Some((&lcut, 1.0)), &p).unwrap();
    println!("scene cut: explained prev {:.3}, next(cut) {:.3}", cut.explained[0], cut.explained[1]);
    assert!(cut.used_prev && !cut.used_next);
    assert!(epe(&cut, |_, _| d, mask) < 0.3);
    // Only the unrelated neighbour: passthrough.
    let mut out = vec![0.0; c.len()];
    let st = render(Frame { width: W, height: H, data: &c }, None, Some(nb(W, H, &lcut, 1.0)), &p, &mut out);
    assert!(!st.used_next && !st.used_prev);
    assert!(out.iter().zip(&c).all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn frame_distance_scaling() {
    let t = Tex::new(8);
    let d = [5.0f32, 2.0];
    let lc = luma(W, H, &grey_frame(W, H, |x, y| t.at(x, y)));
    let lp = luma(W, H, &grey_frame(W, H, |x, y| t.at(x + d[0], y + d[1])));
    let ln2 = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - 2.0 * d[0], y - 2.0 * d[1])));
    let ln3 = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - 3.0 * d[0], y - 3.0 * d[1])));
    let mask = |x: usize, y: usize| x >= 40 && y >= 40 && x + 40 < W && y + 40 < H;
    let a = analyze(W, H, &lc, None, Some((&ln2, 2.0)), &Params::default()).unwrap();
    let b = analyze(W, H, &lc, Some((&lp, 1.0)), Some((&ln3, 3.0)), &Params::default()).unwrap();
    let (ea, eb) = (epe(&a, |_, _| d, mask), epe(&b, |_, _| d, mask));
    println!("frame_distance: next at fd=2 EPE {ea:.3}; prev fd=1 + next fd=3 EPE {eb:.3} (per-frame |d| = 5.39)");
    assert!(ea < 0.3 && eb < 0.3);
}

/// Adjacent source frames of high-frame-rate footage: each neighbour is 0.2 output frames away,
/// so a per-source-frame translation of d must come out as 5d per output frame.
#[test]
fn fractional_frame_distance() {
    let t = Tex::new(10);
    let d = [1.0f32, 0.4];
    let lc = luma(W, H, &grey_frame(W, H, |x, y| t.at(x, y)));
    let lp = luma(W, H, &grey_frame(W, H, |x, y| t.at(x + d[0], y + d[1])));
    let ln = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - d[0], y - d[1])));
    let mask = |x: usize, y: usize| x >= 40 && y >= 40 && x + 40 < W && y + 40 < H;
    let v = [5.0 * d[0], 5.0 * d[1]];
    let an = analyze(W, H, &lc, Some((&lp, 0.2)), Some((&ln, 0.2)), &sens(100.0)).unwrap();
    let e = epe(&an, |_, _| v, mask);
    // Same scene with neighbours 1 frame away: velocity d, and the fd=0.2 result is ~5x it.
    let an1 = analyze(W, H, &lc, Some((&lp, 1.0)), Some((&ln, 1.0)), &sens(100.0)).unwrap();
    let e1 = epe(&an1, |_, _| d, mask);
    println!("fractional frame distance: fd=0.2 EPE {e:.3} vs truth (5.0,2.0); fd=1 EPE {e1:.3} vs (1.0,0.4)");
    assert!(e < 0.3, "EPE {e}");
    assert!(e1 < 0.06, "EPE {e1}");
    // Values below the floor are clamped, not rejected.
    let tiny = analyze(W, H, &lc, Some((&lp, 1e-6)), None, &sens(100.0)).unwrap();
    assert!(tiny.used_prev);
}

#[test]
fn duplicate_detection() {
    let t = Tex::new(9);
    let a = luma(W, H, &grey_frame(W, H, |x, y| t.at(x, y)));
    let mut s: u32 = 12345;
    let mut noise = || {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        ((s >> 8) as f32 / 16_777_216.0 - 0.5) * 0.01 // mild noise, +-0.5%
    };
    let noisy: Vec<f32> = a.iter().map(|v| v + noise()).collect();
    let moved = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - 1.0, y)));
    fn l(d: &[f32]) -> Luma<'_> {
        Luma { width: W, height: H, data: d }
    }
    assert!(is_duplicate_frame(l(&a), l(&a.clone())));
    // A re-encoded repeat where 2% of pixels differ by one 8-bit code value still counts.
    let q: Vec<f32> = a.iter().map(|v| (v * 255.0).round() / 255.0).collect();
    let q2: Vec<f32> = q.iter().enumerate().map(|(i, v)| if i % 50 == 7 { v + 1.0 / 255.0 } else { *v }).collect();
    assert!(is_duplicate_frame(l(&q), l(&q2)));
    assert!(!is_duplicate_frame(l(&a), l(&noisy)));
    assert!(!is_duplicate_frame(l(&a), l(&moved)));
    assert!(!is_duplicate_frame(l(&a), Luma { width: W - 1, height: H, data: &a[..(W - 1) * H] }));
}

#[test]
fn determinism() {
    let (p, c, n, _) = object_scene([9.0, -4.0]);
    let (lp, ln) = (luma(W, H, &p), luma(W, H, &n));
    let f = Frame { width: W, height: H, data: &c };
    let mut o1 = vec![0.0; c.len()];
    let mut o2 = vec![0.0; c.len()];
    for q in [Quality::Draft, Quality::Normal, Quality::High] {
        let prm = Params { quality: q, blur_amount: 1.0, ..Params::default() };
        render(f, Some(nb(W, H, &lp, 1.0)), Some(nb(W, H, &ln, 1.0)), &prm, &mut o1);
        render(f, Some(nb(W, H, &lp, 1.0)), Some(nb(W, H, &ln, 1.0)), &prm, &mut o2);
        assert!(o1.iter().zip(&o2).all(|(a, b)| a.to_bits() == b.to_bits()), "{q:?} not deterministic");
        assert!(o1.iter().zip(&c).any(|(a, b)| a != b));
    }
}

#[test]
fn sensitivity_is_monotonic() {
    let (p, c, n, _) = object_scene([6.0, 2.0]);
    let (lp, ln) = (luma(W, H, &p), luma(W, H, &n));
    let f = Frame { width: W, height: H, data: &c };
    let mut last = -1.0f64;
    let mut line = String::from("sensitivity -> mean |out-in|:");
    for s in [0.0, 10.0, 30.0, 50.0, 70.0, 85.0, 100.0] {
        let mut o = vec![0.0; c.len()];
        render(f, Some(nb(W, H, &lp, 1.0)), Some(nb(W, H, &ln, 1.0)), &sens(s), &mut o);
        let diff: f64 = o.iter().zip(&c).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / c.len() as f64;
        line += &format!(" {s}:{diff:.5}");
        if s == 0.0 {
            assert_eq!(diff, 0.0);
        }
        assert!(diff >= last * 0.98, "blur decreased when raising sensitivity to {s}");
        last = diff;
    }
    println!("{line}");
}

/// Exposure invariance: a strongly contrast-changed analysis luma (gain + gamma, and a flat
/// log-like curve), applied to the current and the neighbour lumas alike, must give nearly the
/// same velocity field and output.
#[test]
fn exposure_invariance_with_render_with_luma() {
    let (p, c, n, inside) = object_scene([7.0, 2.0]);
    let (lp, lc, ln) = (luma(W, H, &p), luma(W, H, &c), luma(W, H, &n));
    let base = analyze(W, H, &lc, Some((&lp, 1.0)), Some((&ln, 1.0)), &Params::default()).unwrap();
    let f = Frame { width: W, height: H, data: &c };
    let mut ob = vec![0.0; c.len()];
    render_with_luma(f, Luma { width: W, height: H, data: &lc }, Some(nb(W, H, &lp, 1.0)), Some(nb(W, H, &ln, 1.0)), &Params::default(), &mut ob);
    let curves: [(&str, fn(f32) -> f32); 3] = [
        ("gain 2.5 * x^2.2", |v| 2.5 * v.max(0.0).powf(2.2)),
        ("log-like 0.35 + 0.25 x^0.5", |v| 0.35 + 0.25 * v.max(0.0).powf(0.5)),
        ("gain 0.2 * x^0.6", |v| 0.2 * v.max(0.0).powf(0.6)),
    ];
    for (name, g) in curves {
        let m = |l: &[f32]| l.iter().map(|&v| g(v)).collect::<Vec<f32>>();
        let (mp, mc, mn) = (m(&lp), m(&lc), m(&ln));
        let a = analyze(W, H, &mc, Some((&mp, 1.0)), Some((&mn, 1.0)), &Params::default()).unwrap();
        let mut dv = 0.0f64;
        let mut dvo = 0.0f64;
        let mut no = 0;
        for i in 0..W * H {
            let e = ((a.vu.d[i] - base.vu.d[i]).powi(2) + (a.vv.d[i] - base.vv.d[i]).powi(2)).sqrt() as f64;
            dv += e;
            if inside((i % W) as f32, (i / W) as f32) {
                dvo += e;
                no += 1;
            }
        }
        let mut o = vec![0.0; c.len()];
        render_with_luma(f, Luma { width: W, height: H, data: &mc }, Some(nb(W, H, &mp, 1.0)), Some(nb(W, H, &mn, 1.0)), &Params::default(), &mut o);
        let od: f64 = o.iter().zip(&ob).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / o.len() as f64;
        let blur_energy: f64 = ob.iter().zip(&c).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / c.len() as f64;
        println!(
            "exposure '{name}': mean |dv| {:.4} px (object {:.4}), explained {:.2}/{:.2} vs {:.2}/{:.2}, mean |out diff| {od:.5} (blur itself changes the image by {blur_energy:.5})",
            dv / (W * H) as f64,
            dvo / no as f64,
            a.explained[0],
            a.explained[1],
            base.explained[0],
            base.explained[1]
        );
        let omax = o.iter().zip(&ob).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
        println!("    max |out diff| {omax:.4}");
        assert!(dv / ((W * H) as f64) < 0.1);
        assert!(od < 3e-4 && od < 0.3 * blur_energy);
    }
}

/// One-sided (next only) large motion in all directions, including content leaving the frame on
/// the left, which once produced garbage that propagated inward.
#[test]
fn large_translation_all_directions() {
    let t = Tex::new(1);
    let cur = luma(W, H, &grey_frame(W, H, |x, y| t.at(x, y)));
    for d in [[-45.0f32, 0.0], [-60.0, -25.0], [-96.0, 10.0], [96.0, 0.0], [0.0, -54.0], [0.0, 54.0], [3.0, 1.0]] {
        let next = luma(W, H, &grey_frame(W, H, |x, y| t.at(x - d[0], y - d[1])));
        let border = (d[0].abs().max(d[1].abs()) as usize) + 16;
        let mask = |x: usize, y: usize| x >= border && y >= border && x + border < W && y + border < H;
        let a = analyze(W, H, &cur, None, Some((&next, 1.0)), &sens(100.0)).unwrap();
        let e = epe(&a, |_, _| d, mask);
        println!("large one-sided translation {d:?}: EPE {e:.3}");
        assert!(e < 0.2);
    }
}

/// 16 concurrent foreign host threads rendering simultaneously must produce bit-identical output
/// to a standalone render, exercising `par::enter_render`, helper leasing, and buffer pooling.
#[test]
fn concurrent_k16_is_bit_exact() {
    use std::sync::{Arc, Barrier};
    let (p, c, n, _) = object_scene([9.0, -4.0]);
    let (lp, ln) = (Arc::new(luma(W, H, &p)), Arc::new(luma(W, H, &n)));
    let c = Arc::new(c);
    let prm = Params { quality: Quality::Normal, blur_amount: 1.0, ..Params::default() };

    let mut baseline = vec![0.0f32; c.len()];
    render(
        Frame { width: W, height: H, data: &c },
        Some(nb(W, H, &lp, 1.0)),
        Some(nb(W, H, &ln, 1.0)),
        &prm,
        &mut baseline,
    );
    let baseline = Arc::new(baseline);

    let k = 16usize;
    let bar = Arc::new(Barrier::new(k));
    let mut handles = Vec::with_capacity(k);
    for tid in 0..k {
        let (c, lp, ln, base, bar) = (
            Arc::clone(&c),
            Arc::clone(&lp),
            Arc::clone(&ln),
            Arc::clone(&baseline),
            Arc::clone(&bar),
        );
        handles.push(std::thread::spawn(move || {
            let mut out = vec![0.0f32; c.len()];
            bar.wait();
            for iter in 0..3 {
                render(
                    Frame { width: W, height: H, data: &c },
                    Some(nb(W, H, &lp, 1.0)),
                    Some(nb(W, H, &ln, 1.0)),
                    &prm,
                    &mut out,
                );
                assert!(
                    out.iter().zip(base.iter()).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "thread {tid} iter {iter} diverged from single-thread render"
                );
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
}

/// Panics inside `par::for_rows_mut`, `par::join`, and `par::join_split` must unwind cleanly
/// under `catch_unwind` without aborting, deadlocking, or leaking render/helper slots.
#[test]
fn par_worker_panic_unwinds_cleanly() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut buf = vec![0u32; 1024];
    let r1 = catch_unwind(AssertUnwindSafe(|| {
        let _g = crate::par::enter_render();
        crate::par::for_rows_mut(&mut buf, 16, 1, |r0, chunk| {
            let rows = chunk.len() / 16;
            if r0 <= 32 && 32 < r0 + rows {
                panic!("synthetic worker panic in for_rows_mut");
            }
            for x in chunk {
                *x = 1;
            }
        });
    }));
    assert!(r1.is_err(), "expected panic to propagate from for_rows_mut");

    let r2 = catch_unwind(AssertUnwindSafe(|| {
        let _g = crate::par::enter_render();
        crate::par::join(
            || 1usize,
            || -> usize { panic!("synthetic worker panic in join") },
        )
    }));
    assert!(r2.is_err(), "expected panic to propagate from join");

    let r3 = catch_unwind(AssertUnwindSafe(|| {
        let _g = crate::par::enter_render();
        crate::par::join_split(
            || 1usize,
            || -> usize { panic!("synthetic worker panic in join_split") },
        )
    }));
    assert!(r3.is_err(), "expected panic to propagate from join_split");

    // Subsequent render must succeed normally after panics.
    let (p, c, n, _) = object_scene([6.0, 2.0]);
    let (lp, ln) = (luma(W, H, &p), luma(W, H, &n));
    let mut out = vec![0.0f32; c.len()];
    let st = render(
        Frame { width: W, height: H, data: &c },
        Some(nb(W, H, &lp, 1.0)),
        Some(nb(W, H, &ln, 1.0)),
        &Params::default(),
        &mut out,
    );
    assert!(st.used_prev && st.used_next);
}

