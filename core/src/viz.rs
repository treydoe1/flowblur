//! Standard optical-flow colour wheel (Baker et al., Middlebury): hue = direction,
//! saturation = magnitude relative to a fixed scale, white = no motion.

/// Magnitude (as a fraction of the frame's longer side, per frame interval) that maps to full saturation.
pub(crate) const VIZ_FULL_SCALE: f32 = 0.025;

const NCOLS: usize = 55;

fn wheel() -> [[f32; 3]; NCOLS] {
    let (ry, yg, gc, cb, bm, mr) = (15, 6, 4, 11, 13, 6);
    let mut w = [[0.0f32; 3]; NCOLS];
    let mut k = 0;
    for i in 0..ry {
        w[k] = [1.0, i as f32 / ry as f32, 0.0];
        k += 1;
    }
    for i in 0..yg {
        w[k] = [1.0 - i as f32 / yg as f32, 1.0, 0.0];
        k += 1;
    }
    for i in 0..gc {
        w[k] = [0.0, 1.0, i as f32 / gc as f32];
        k += 1;
    }
    for i in 0..cb {
        w[k] = [0.0, 1.0 - i as f32 / cb as f32, 1.0];
        k += 1;
    }
    for i in 0..bm {
        w[k] = [i as f32 / bm as f32, 0.0, 1.0];
        k += 1;
    }
    for i in 0..mr {
        w[k] = [1.0, 0.0, 1.0 - i as f32 / mr as f32];
        k += 1;
    }
    w
}

pub(crate) struct Wheel([[f32; 3]; NCOLS]);

impl Wheel {
    pub fn new() -> Self {
        Wheel(wheel())
    }

    /// `u, v` already divided by the full-scale magnitude. Returns gamma-encoded RGB.
    pub fn color(&self, u: f32, v: f32) -> [f32; 3] {
        let rad = (u * u + v * v).sqrt().min(1.0);
        let a = (-v).atan2(-u) / std::f32::consts::PI;
        let fk = (a + 1.0) / 2.0 * (NCOLS - 1) as f32;
        let k0 = (fk.floor() as usize).min(NCOLS - 1);
        let k1 = (k0 + 1) % NCOLS;
        let f = fk - k0 as f32;
        let mut out = [0.0f32; 3];
        for c in 0..3 {
            let col = (1.0 - f) * self.0[k0][c] + f * self.0[k1][c];
            out[c] = 1.0 - rad * (1.0 - col);
        }
        out
    }
}
