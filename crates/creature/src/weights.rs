//! Skin weights from distance through the body. Each bone's distance to
//! every cell of the solid is measured through the inside only, so a leg
//! is far from the other leg even where their surfaces touch, and a
//! vertex belongs to the bone it is nearest to that way. Next to a joint
//! it is shared with the neighbouring bone, over a band as wide as the
//! body is thick there.

use skintokens::geom::{self, V3};

use crate::parts::{geodesic, Body};
use crate::skeleton::Skeleton;

/// Width of the band a joint's two bones share, in body thicknesses (the
/// inscribed radius at the joint).
pub const BLEND: f64 = 0.6;

/// Smoothing passes along the surface after the bands are laid.
const SMOOTH_PASSES: usize = 2;

pub struct Skin {
    /// Per vertex: up to four (bone, weight), weights summing to 1.
    pub joints: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}

/// Cells a bone passes through.
fn bone_cells(body: &Body, head: V3, tail: V3) -> Vec<usize> {
    let s = &body.solid;
    let len = geom::norm(geom::sub(tail, head));
    let n = ((len / (0.5 * s.cell)).ceil() as usize).max(1);
    let mut out = vec![];
    for k in 0..=n {
        let p = geom::add(head, geom::scale(geom::sub(tail, head), k as f64 / n as f64));
        if let Some(c) = s.nearest_inside(p, 4) {
            if out.last() != Some(&c) {
                out.push(c);
            }
        }
    }
    out
}

/// Distance field of every bone, on the inside cells only.
pub struct Fields {
    /// Grid cell -> index among inside cells (u32::MAX outside).
    slot: Vec<u32>,
    /// Per bone, per inside cell: distance in cells.
    dist: Vec<Vec<f32>>,
}

impl Fields {
    pub fn build(body: &Body, sk: &Skeleton) -> Fields {
        let s = &body.solid;
        let mut slot = vec![u32::MAX; s.len()];
        let mut inside = vec![];
        for i in 0..s.len() {
            if s.inside[i] {
                slot[i] = inside.len() as u32;
                inside.push(i);
            }
        }
        let nb = sk.bones.len();
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(nb.max(1));
        let mut dist: Vec<Vec<f32>> = vec![vec![]; nb];
        let chunk = nb.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (ci, out) in dist.chunks_mut(chunk).enumerate() {
                let inside = &inside;
                sc.spawn(move || {
                    for (k, o) in out.iter_mut().enumerate() {
                        let b = &sk.bones[ci * chunk + k];
                        let field = geodesic(s, &bone_cells(body, b.head, b.tail));
                        *o = inside.iter().map(|&i| field[i]).collect();
                    }
                });
            }
        });
        Fields { slot, dist }
    }

    /// Each bone's distance (cells) at point `p`: blended between the
    /// inside cells around it, so it changes smoothly across cell borders.
    pub fn at(&self, body: &Body, p: V3, out: &mut [f32]) {
        let s = &body.solid;
        let g = [0, 1, 2].map(|a| (p[a] - s.origin[a]) / s.cell - 0.5);
        let base = g.map(|x| x.floor());
        out.iter_mut().for_each(|o| *o = 0.0);
        let mut total = 0.0f32;
        for corner in 0..8 {
            let mut c = [0usize; 3];
            let mut w = 1.0f64;
            let mut ok = true;
            for a in 0..3 {
                let hi = corner >> a & 1;
                let v = base[a] as i64 + hi as i64;
                if v < 0 || v >= s.dims[a] as i64 {
                    ok = false;
                    break;
                }
                c[a] = v as usize;
                let f = g[a] - base[a];
                w *= if hi == 1 { f } else { 1.0 - f };
            }
            if !ok || w <= 1e-6 {
                continue;
            }
            let sl = self.slot[s.idx(c)];
            if sl == u32::MAX || !self.dist[0][sl as usize].is_finite() {
                continue;
            }
            for (o, d) in out.iter_mut().zip(&self.dist) {
                *o += w as f32 * d[sl as usize];
            }
            total += w as f32;
        }
        if total > 0.0 {
            out.iter_mut().for_each(|o| *o /= total);
        } else if let Some(c) = s.nearest_inside(p, 6) {
            let sl = self.slot[c] as usize;
            let off = (geom::norm(geom::sub(s.center(c), p)) / s.cell) as f32;
            for (o, d) in out.iter_mut().zip(&self.dist) {
                *o = d[sl] + off;
            }
        } else {
            out.iter_mut().for_each(|o| *o = f32::INFINITY);
        }
    }
}

/// Weights for `verts` (any order; split copies of a position get the same
/// weights). `blend`: see `BLEND`.
pub fn skin(body: &Body, sk: &Skeleton, verts: &[V3], tris: &[[u32; 3]], blend: f64) -> Skin {
    let nb = sk.bones.len();
    let fields = Fields::build(body, sk);
    let (id, reps) = skintokens::mesh::weld(verts, 1e-5 * body.size.max(1e-9));
    // bones that may share a vertex with bone b: its parent, its children,
    // and the other children of its parent
    let mut near: Vec<Vec<usize>> = vec![vec![]; nb];
    for b in 0..nb {
        if let Some(p) = sk.bones[b].parent {
            near[b].push(p);
            near[p].push(b);
            for c in 0..nb {
                if c != b && sk.bones[c].parent == Some(p) {
                    near[b].push(c);
                }
            }
        }
    }
    let cell = body.solid.cell;
    let mut w: Vec<Vec<(u16, f32)>> = Vec::with_capacity(reps.len());
    let mut d = vec![0f32; nb];
    for &p in &reps {
        fields.at(body, p, &mut d);
        let a = (0..nb).min_by(|&x, &y| d[x].partial_cmp(&d[y]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0);
        let mut row: Vec<(u16, f32)> = vec![(a as u16, 1.0)];
        if d[a].is_finite() {
            for &b in &near[a] {
                // the joint the two share: the head of whichever is the child
                // (two children of one parent: the nearer one's head)
                let joint = if sk.bones[b].parent == Some(a) { b } else { a };
                let band = (blend * sk.bones[joint].radius / cell).max(1.0) as f32;
                let t = 1.0 - (d[b] - d[a]) / band;
                if t > 0.0 {
                    row.push((b as u16, t * t * (3.0 - 2.0 * t)));
                }
            }
        }
        w.push(row);
    }
    // one smoothing pass along the surface takes out the cells' stair steps
    let mut nbrs: Vec<Vec<u32>> = vec![vec![]; reps.len()];
    for t in tris {
        let v = t.map(|i| id[i as usize]);
        for k in 0..3 {
            let (x, y) = (v[k], v[(k + 1) % 3]);
            if x != y {
                nbrs[x as usize].push(y);
                nbrs[y as usize].push(x);
            }
        }
    }
    let norm = |row: &mut Vec<(u16, f32)>| {
        row.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        row.truncate(4);
        let s: f32 = row.iter().map(|x| x.1).sum();
        row.iter_mut().for_each(|x| x.1 /= s.max(1e-12));
    };
    w.iter_mut().for_each(norm);
    for ns in nbrs.iter_mut() {
        ns.sort_unstable();
        ns.dedup();
    }
    for _ in 0..SMOOTH_PASSES {
    let old = w.clone();
    for (v, ns) in nbrs.iter().enumerate() {
        if ns.is_empty() {
            continue;
        }
        let mut acc: Vec<(u16, f32)> = old[v].iter().map(|&(j, x)| (j, 0.5 * x)).collect();
        let share = 0.5 / ns.len() as f32;
        for &n in ns.iter() {
            for &(j, x) in &old[n as usize] {
                match acc.iter_mut().find(|a| a.0 == j) {
                    Some(a) => a.1 += share * x,
                    None => acc.push((j, share * x)),
                }
            }
        }
        norm(&mut acc);
        w[v] = acc;
    }
    }
    let mut joints = Vec::with_capacity(verts.len());
    let mut weights = Vec::with_capacity(verts.len());
    for &i in &id {
        let row = &w[i as usize];
        let mut j4 = [0u16; 4];
        let mut w4 = [0f32; 4];
        for (k, &(j, x)) in row.iter().enumerate() {
            if x > 1e-4 {
                j4[k] = j;
                w4[k] = x;
            }
        }
        let s: f32 = w4.iter().sum();
        w4.iter_mut().for_each(|x| *x /= s.max(1e-12));
        joints.push(j4);
        weights.push(w4);
    }
    Skin { joints, weights }
}
