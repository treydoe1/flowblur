//! tblur-cli: run the T Blur core on PNG frames.
//!
//! Single frame:  tblur-cli [--prev a.png] --cur b.png [--next c.png] --out o.png [options]
//! Sequence:      tblur-cli --seq <dir> --out-dir <dir> [options]
//! Options: --blur F  --sens F  --max-blur F  --quality draft|normal|high  --view output|vectors
//!          --no-linear  --bench N (render each frame N extra times and report min/median)
//!          --fd F  frame distance of both neighbours in output frames (default 1.0; may be < 1,
//!                  e.g. 0.2 for adjacent 120 fps source frames on a 24 fps timeline). In --seq
//!                  mode it multiplies the per-neighbour distance in input frames.

use tblur_core::*;
use std::path::{Path, PathBuf};
use std::process::exit;
use std::time::Instant;

struct Img {
    w: usize,
    h: usize,
    rgba: Vec<f32>,
    luma: Vec<f32>,
}

fn load(p: &Path) -> Img {
    let im = image::open(p).unwrap_or_else(|e| {
        eprintln!("cannot read {}: {e}", p.display());
        exit(1)
    });
    let im = im.into_rgba32f();
    let (w, h) = (im.width() as usize, im.height() as usize);
    let mut rgba = im.into_raw();
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3];
        px[0] *= a;
        px[1] *= a;
        px[2] *= a;
    }
    let luma = luma_from_rgba(Frame { width: w, height: h, data: &rgba });
    Img { w, h, rgba, luma }
}

fn save(p: &Path, w: usize, h: usize, data: &[f32]) {
    let mut buf = Vec::with_capacity(w * h * 3);
    for px in data.chunks_exact(4) {
        let a = px[3];
        let ia = if a > 1e-6 { 1.0 / a } else { 0.0 };
        for c in 0..3 {
            let v = if a > 1e-6 { px[c] * ia } else { px[c] };
            buf.push((v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
        }
    }
    image::save_buffer(p, &buf, w as u32, h as u32, image::ExtendedColorType::Rgb8).unwrap_or_else(|e| {
        eprintln!("cannot write {}: {e}", p.display());
        exit(1)
    });
}

fn usage() -> ! {
    eprintln!(
        "usage:\n  tblur-cli [--prev a.png] --cur b.png [--next c.png] --out o.png [opts]\n  tblur-cli --seq <dir> --out-dir <dir> [opts]\nopts: --blur F --sens F --max-blur F --quality draft|normal|high --view output|vectors --no-linear --bench N --fd F"
    );
    exit(2)
}

fn neighbor(img: &Img, fd: f32) -> Neighbor<'_> {
    Neighbor { luma: Luma { width: img.w, height: img.h, data: &img.luma }, frame_distance: fd }
}

fn run_one(name: &str, cur: &Img, prev: Option<Neighbor>, next: Option<Neighbor>, p: &Params, bench: usize, out: &mut [f32]) {
    let frame = Frame { width: cur.w, height: cur.h, data: &cur.rgba };
    let t = Instant::now();
    let st = render(frame, prev, next, p, out);
    let wall = t.elapsed().as_secs_f32() * 1000.0;
    print!(
        "{name}: {}x{} flow {:.1} ms, blur {:.1} ms, wall {:.1} ms, prev {} next {}, max velocity {:.1} px/frame",
        cur.w, cur.h, st.flow_ms, st.blur_ms, wall, st.used_prev, st.used_next, st.max_velocity_px
    );
    if bench > 0 {
        let mut ts: Vec<f32> = (0..bench)
            .map(|_| {
                let t = Instant::now();
                render(frame, prev, next, p, out);
                t.elapsed().as_secs_f32() * 1000.0
            })
            .collect();
        ts.sort_by(|a, b| a.total_cmp(b));
        print!(" | bench x{bench}: min {:.1} ms, median {:.1} ms", ts[0], ts[ts.len() / 2]);
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut p = Params::default();
    let (mut prev, mut cur, mut next, mut out, mut seq, mut out_dir) = (None, None, None, None, None, None);
    let mut bench = 0usize;
    let mut fd = 1.0f32;
    let mut i = 0;
    let val = |i: &mut usize| -> String {
        *i += 1;
        args.get(*i).cloned().unwrap_or_else(|| usage())
    };
    let num = |s: String| -> f32 { s.parse().unwrap_or_else(|_| usage()) };
    while i < args.len() {
        match args[i].as_str() {
            "--prev" => prev = Some(PathBuf::from(val(&mut i))),
            "--cur" => cur = Some(PathBuf::from(val(&mut i))),
            "--next" => next = Some(PathBuf::from(val(&mut i))),
            "--out" => out = Some(PathBuf::from(val(&mut i))),
            "--seq" => seq = Some(PathBuf::from(val(&mut i))),
            "--out-dir" => out_dir = Some(PathBuf::from(val(&mut i))),
            "--blur" => p.blur_amount = num(val(&mut i)),
            "--sens" => p.motion_sensitivity = num(val(&mut i)),
            "--max-blur" => p.max_blur_percent = num(val(&mut i)),
            "--bench" => bench = num(val(&mut i)) as usize,
            "--fd" => {
                fd = num(val(&mut i));
                if !(fd.is_finite() && fd > 0.0) {
                    eprintln!("--fd must be a positive number");
                    exit(2)
                }
            }
            "--quality" => {
                p.quality = match val(&mut i).as_str() {
                    "draft" => Quality::Draft,
                    "normal" => Quality::Normal,
                    "high" => Quality::High,
                    _ => usage(),
                }
            }
            "--view" => {
                p.view = match val(&mut i).as_str() {
                    "output" => View::Output,
                    "vectors" => View::MotionVectors,
                    _ => usage(),
                }
            }
            "--no-linear" => p.linear_light = false,
            _ => usage(),
        }
        i += 1;
    }

    if let (Some(dir), Some(od)) = (seq, out_dir) {
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| {
                eprintln!("cannot list {}: {e}", dir.display());
                exit(1)
            })
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e.eq_ignore_ascii_case("png")))
            .collect();
        files.sort();
        std::fs::create_dir_all(&od).unwrap_or_else(|e| {
            eprintln!("cannot create {}: {e}", od.display());
            exit(1)
        });
        let imgs: Vec<Img> = files.iter().map(|f| load(f)).collect();
        let mut buf = Vec::new();
        for k in 0..imgs.len() {
            let c = &imgs[k];
            let lum = |j: usize| Luma { width: imgs[j].w, height: imgs[j].h, data: &imgs[j].luma };
            // Skip held (duplicated) frames when picking neighbours, as a host would.
            let mut pv = None;
            let mut j = k;
            while j > 0 {
                j -= 1;
                if !is_duplicate_frame(lum(k), lum(j)) {
                    pv = Some(neighbor(&imgs[j], (k - j) as f32 * fd));
                    break;
                }
            }
            let mut nx = None;
            let mut j = k;
            while j + 1 < imgs.len() {
                j += 1;
                if !is_duplicate_frame(lum(k), lum(j)) {
                    nx = Some(neighbor(&imgs[j], (j - k) as f32 * fd));
                    break;
                }
            }
            buf.resize(c.rgba.len(), 0.0);
            let name = files[k].file_name().unwrap().to_string_lossy().to_string();
            run_one(&name, c, pv, nx, &p, bench, &mut buf);
            save(&od.join(&name), c.w, c.h, &buf);
        }
        return;
    }

    let (Some(cur), Some(out)) = (cur, out) else { usage() };
    let c = load(&cur);
    let pi = prev.map(|f| load(&f));
    let ni = next.map(|f| load(&f));
    let mut buf = vec![0.0f32; c.rgba.len()];
    run_one(
        &cur.file_name().unwrap().to_string_lossy(),
        &c,
        pi.as_ref().map(|i| neighbor(i, fd)),
        ni.as_ref().map(|i| neighbor(i, fd)),
        &p,
        bench,
        &mut buf,
    );
    save(&out, c.w, c.h, &buf);
}
