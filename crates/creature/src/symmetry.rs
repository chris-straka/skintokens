//! The body's mirror plane, found from the body itself.
//!
//! Everything that reads a creature's parts takes its left and right from
//! the plane x = 0. A cleaned mesh should stand that way; when the clean
//! step could not tell which way a model faces (an unclear pose, hands on
//! the ground) it can leave it a quarter turn off, and then an arm and a
//! leg of one side read as a left and a right.

use skintokens::geom::{self, KdTree, V3};

pub struct Mirror {
    /// Turn about the up axis (degrees) that brings the mirror plane's
    /// normal onto +X: x' = cos * x + sin * z, z' = -sin * x + cos * z.
    pub yaw_deg: f64,
    /// Where the plane then stands on the x axis.
    pub offset: f64,
    /// Mean distance from the body's mirror image back to the body, as a
    /// share of its longest side: for the plane found, and for the best
    /// plane that faces along x as the mesh is.
    pub score: f64,
    pub score_as_is: f64,
}

pub fn turn(p: V3, yaw_deg: f64, offset: f64) -> V3 {
    let (s, c) = yaw_deg.to_radians().sin_cos();
    [c * p[0] + s * p[2] - offset, p[1], -s * p[0] + c * p[2]]
}

pub fn mirror_plane(verts: &[V3]) -> Mirror {
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in verts {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let size = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max).max(1e-9);
    let tree = KdTree::new(verts.to_vec());
    let step = (verts.len() / 1200).max(1);
    let sample: Vec<V3> = verts.iter().step_by(step).copied().collect();
    let score = |yaw: f64, off: f64| -> f64 {
        let (s, c) = yaw.to_radians().sin_cos();
        let n = [c, 0.0, s];
        let mut sum = 0.0;
        for p in &sample {
            let d = geom::dot(*p, n) - off;
            let q = geom::sub(*p, geom::scale(n, 2.0 * d));
            sum += tree.knn(q, 1)[0].0;
        }
        sum / sample.len() as f64 / size
    };
    let best_at = |yaw: f64| -> (f64, f64) {
        let (s, c) = yaw.to_radians().sin_cos();
        let mut proj: Vec<f64> = sample.iter().map(|p| c * p[0] + s * p[2]).collect();
        proj.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = (proj[0] + proj[proj.len() - 1]) / 2.0;
        let med = proj[proj.len() / 2];
        let mut best = (f64::INFINITY, 0.0);
        for k in 0..=6 {
            let off = mid + (med - mid) * k as f64 / 3.0 - (med - mid) / 2.0;
            let sc = score(yaw, off);
            if sc < best.0 {
                best = (sc, off);
            }
        }
        best
    };
    let as_is = best_at(0.0);
    let mut best = (as_is.0, 0.0, as_is.1);
    let mut yaw = -87.0;
    while yaw <= 90.0 {
        if yaw != 0.0 {
            let (sc, off) = best_at(yaw);
            if sc < best.0 {
                best = (sc, yaw, off);
            }
        }
        yaw += 3.0;
    }
    // a finer look round the best
    for d in [-2.0, -1.0, 1.0, 2.0] {
        let (sc, off) = best_at(best.1 + d);
        if sc < best.0 {
            best = (sc, best.1 + d, off);
        }
    }
    Mirror {
        yaw_deg: best.1,
        offset: best.2,
        score: best.0,
        score_as_is: as_is.0,
    }
}
