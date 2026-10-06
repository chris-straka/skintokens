//! What upstream's data pipeline does to a mesh before the model sees it
//! (`src/rig_package/parser/bpy.py` extract_mesh, `src/data/augment.py`
//! trim/affine/normalize, `src/data/sampler.py` + `rig_package/utils.py`
//! sampling, and `Asset.from_data`'s inverse-distance skin transfer), MIT.
//!
//! The model frame is Blender's: glTF (x, y, z) -> (x, -z, y), Z up.

use crate::geom::{self, KdTree, V3};
use skintokens_nn::Rng;

/// glTF (Y up) -> model frame (Blender, Z up).
pub fn to_model(p: V3) -> V3 {
    [p[0], -p[2], p[1]]
}

/// Model frame -> glTF.
pub fn from_model(p: V3) -> V3 {
    [p[0], p[2], -p[1]]
}

pub struct Mesh {
    pub vertices: Vec<V3>,
    pub faces: Vec<[u32; 3]>,
    /// Start offset of each part in `vertices`.
    pub offsets: Vec<usize>,
}

impl Mesh {
    /// Concatenates parts (already in the model frame). Faces with a repeated
    /// vertex or the same vertex set as an earlier face are dropped, as
    /// Blender's importer does.
    pub fn from_parts(parts: &[(Vec<V3>, Vec<[u32; 3]>)]) -> Mesh {
        let mut vertices = vec![];
        let mut faces = vec![];
        let mut offsets = vec![];
        for (v, f) in parts {
            let base = vertices.len() as u32;
            offsets.push(vertices.len());
            vertices.extend_from_slice(v);
            let mut seen = std::collections::HashSet::new();
            for t in f {
                if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] || t.iter().any(|&i| i as usize >= v.len()) {
                    continue;
                }
                let mut k = *t;
                k.sort();
                if seen.insert(k) {
                    faces.push([t[0] + base, t[1] + base, t[2] + base]);
                }
            }
        }
        Mesh { vertices, faces, offsets }
    }

    /// trimesh face normals (unit; zero for degenerate faces).
    pub fn face_normals(&self) -> Vec<V3> {
        self.faces
            .iter()
            .map(|f| {
                let [a, b, c] = f.map(|i| self.vertices[i as usize]);
                let n = geom::cross(geom::sub(b, a), geom::sub(c, a));
                let l = geom::norm(n);
                if l > 0.0 {
                    geom::scale(n, 1.0 / l)
                } else {
                    [0.0; 3]
                }
            })
            .collect()
    }

    /// trimesh vertex normals: face normals weighted by the corner angle,
    /// degenerate faces skipped, unit length (zero if unreferenced).
    pub fn vertex_normals(&self, fnorm: &[V3]) -> Vec<V3> {
        let mut acc = vec![[0.0f64; 3]; self.vertices.len()];
        for (f, n) in self.faces.iter().zip(fnorm) {
            if geom::dot(*n, *n) <= 0.5 {
                continue;
            }
            let p = f.map(|i| self.vertices[i as usize]);
            for k in 0..3 {
                let a = geom::sub(p[(k + 1) % 3], p[k]);
                let b = geom::sub(p[(k + 2) % 3], p[k]);
                let (la, lb) = (geom::norm(a), geom::norm(b));
                let ang = if la > 0.0 && lb > 0.0 { (geom::dot(a, b) / (la * lb)).clamp(-1.0, 1.0).acos() } else { 0.0 };
                let v = &mut acc[f[k] as usize];
                for d in 0..3 {
                    v[d] += n[d] * ang;
                }
            }
        }
        acc.into_iter()
            .map(|v| {
                let l = geom::norm(v);
                if l > 0.0 {
                    geom::scale(v, 1.0 / l)
                } else {
                    [0.0; 3]
                }
            })
            .collect()
    }
}

/// Upstream's predict-time affine: center the bounds (of vertices and, for
/// a given skeleton, joints) and scale the longest extent into [-1, 1].
///
/// Upstream composes the matrix in float32 (`_trans_to_m`, `_scale_to_m`)
/// and applies it to float64 points; that rounding is kept because joints
/// that sit exactly on a token bin boundary otherwise round the other way.
#[derive(Clone, Copy, Debug)]
pub struct Normalize {
    pub center: V3,
    pub scale: f64,
    /// float32-rounded 1/scale and offset (-center/scale).
    k: f64,
    t: V3,
}

impl Normalize {
    pub fn fit<'a>(points: impl Iterator<Item = &'a V3>) -> Normalize {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in points {
            for a in 0..3 {
                lo[a] = lo[a].min(p[a]);
                hi[a] = hi[a].max(p[a]);
            }
        }
        let center = geom::scale(geom::add(lo, hi), 0.5);
        let ext = geom::sub(hi, lo);
        let scale = (ext[0].max(ext[1]).max(ext[2]) / 2.0).max(1e-12);
        let k = (1.0 / scale) as f32;
        let t = center.map(|c| (k * (-c as f32)) as f64);
        Normalize { center, scale, k: k as f64, t }
    }
    pub fn apply(&self, p: V3) -> V3 {
        [p[0] * self.k + self.t[0], p[1] * self.k + self.t[1], p[2] * self.k + self.t[2]]
    }
    pub fn invert(&self, p: V3) -> V3 {
        [(p[0] - self.t[0]) / self.k, (p[1] - self.t[1]) / self.k, (p[2] - self.t[2]) / self.k]
    }
}

pub const NUM_SAMPLES: usize = 54000;

/// Upstream's `mix` sampler at predict time: its config names 16384 vertex
/// samples, but the call for the main cloud does not pass that count, so
/// all 54000 points are area-weighted surface samples with face normals.
/// Positions in the normalized frame. (`vnorm` is used only for meshes
/// without area.)
pub fn sample(vertices: &[V3], faces: &[[u32; 3]], vnorm: &[V3], fnorm: &[V3], rng: &mut Rng) -> Vec<[f32; 6]> {
    let mut out = Vec::with_capacity(NUM_SAMPLES);
    let rest = NUM_SAMPLES;
    let mut cum = Vec::with_capacity(faces.len());
    let mut total = 0.0;
    for f in faces {
        let [a, b, c] = f.map(|i| vertices[i as usize]);
        total += geom::norm(geom::cross(geom::sub(b, a), geom::sub(c, a)));
        cum.push(total);
    }
    if total <= 0.0 || faces.is_empty() {
        // no area: repeat vertices
        while out.len() < NUM_SAMPLES {
            let i = rng.below(vertices.len());
            let (v, n) = (vertices[i], vnorm[i]);
            out.push([v[0] as f32, v[1] as f32, v[2] as f32, n[0] as f32, n[1] as f32, n[2] as f32]);
        }
        return out;
    }
    for _ in 0..rest {
        let pick = rng.uniform() * total;
        let fi = cum.partition_point(|&c| c < pick).min(faces.len() - 1);
        let f = faces[fi];
        let [a, b, c] = f.map(|i| vertices[i as usize]);
        let (mut r1, mut r2) = (rng.uniform(), rng.uniform());
        if r1 + r2 > 1.0 {
            r1 = (r1 - 1.0).abs();
            r2 = (r2 - 1.0).abs();
        }
        let p = geom::add(a, geom::add(geom::scale(geom::sub(b, a), r1), geom::scale(geom::sub(c, a), r2)));
        let n = fnorm[fi];
        out.push([p[0] as f32, p[1] as f32, p[2] as f32, n[0] as f32, n[1] as f32, n[2] as f32]);
    }
    out
}

/// `Asset.from_data`: per-vertex skin from the sampled points' skin by
/// inverse distance over the 8 nearest samples.
pub fn transfer_skin(samples: &[[f32; 6]], skin: &[Vec<f32>], vertices: &[V3]) -> Vec<Vec<f32>> {
    let pts: Vec<V3> = samples.iter().map(|s| [s[0] as f64, s[1] as f64, s[2] as f64]).collect();
    let tree = KdTree::new(pts);
    let j = skin.first().map(|r| r.len()).unwrap_or(0);
    let k = 8.min(samples.len());
    vertices
        .iter()
        .map(|&v| {
            let nn = tree.knn(v, k);
            let w: Vec<f64> = nn.iter().map(|(d, _)| 1.0 / (d + 1e-8)).collect();
            let ws: f64 = w.iter().sum();
            let mut out = vec![0f32; j];
            for ((_, i), wi) in nn.iter().zip(&w) {
                let f = (wi / ws) as f32;
                for (o, s) in out.iter_mut().zip(&skin[*i]) {
                    *o += f * s;
                }
            }
            out
        })
        .collect()
}

/// Top-4 influences per vertex, renormalized over those four (upstream's
/// group_per_vertex=4), then what Blender's glTF exporter does to them:
/// influences <= 1e-4 dropped and the rest renormalized. Returns (joint
/// indices, weights).
pub fn top4(skin: &[Vec<f32>]) -> (Vec<[u16; 4]>, Vec<[f32; 4]>) {
    let mut js = Vec::with_capacity(skin.len());
    let mut ws = Vec::with_capacity(skin.len());
    for row in skin {
        let mut idx: Vec<usize> = (0..row.len()).collect();
        idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
        let k = idx.len().min(4);
        let s: f32 = idx[..k].iter().map(|&i| row[i].max(0.0)).sum();
        let mut j4 = [0u16; 4];
        let mut w4 = [0f32; 4];
        for t in 0..k {
            j4[t] = idx[t] as u16;
            w4[t] = if s > 0.0 { row[idx[t]].max(0.0) / s } else if t == 0 { 1.0 } else { 0.0 };
        }
        // Blender export: drop <= 1e-4 (the top one always survives here), renormalize
        for t in 1..4 {
            if w4[t] <= 1e-4 {
                w4[t] = 0.0;
            }
        }
        let s2: f32 = w4.iter().sum();
        if s2 > 0.0 {
            w4.iter_mut().for_each(|w| *w /= s2);
        }
        // unused slots: joint 0, weight 0 (glTF convention)
        for t in 0..4 {
            if w4[t] <= 0.0 {
                j4[t] = 0;
                w4[t] = 0.0;
            }
        }
        js.push(j4);
        ws.push(w4);
    }
    (js, ws)
}

/// Upstream `trim_skeleton`: drop joints that carry no skin unless they are
/// needed to keep branches. Returns kept joint indices (in order).
pub fn trim(parents: &[i64], has_skin: &[bool]) -> Vec<usize> {
    let n = parents.len();
    if !has_skin.iter().any(|&h| h) {
        return (0..n).collect();
    }
    let mut sons = vec![vec![]; n];
    let order = dfs_order(parents);
    for &u in &order {
        if parents[u] >= 0 {
            sons[parents[u] as usize].push(u);
        }
    }
    let mut sub = vec![false; n];
    for &u in order.iter().rev() {
        sub[u] = has_skin[u] || sons[u].iter().any(|&v| sub[v]);
    }
    let good: Vec<Vec<usize>> = (0..n).map(|u| sons[u].iter().copied().filter(|&v| sub[v]).collect()).collect();
    let mut keep = vec![false; n];
    for &u in &order {
        if has_skin[u] {
            keep[u] = true;
        } else {
            let p = parents[u];
            if good[u].len() >= 2 {
                keep[u] = true;
            } else if good[u].len() == 1 && p != -1 {
                let gp = &good[p as usize];
                if gp.len() >= 2 || (gp.len() == 1 && gp[0] != u) {
                    keep[u] = true;
                }
            }
        }
    }
    (0..n).filter(|&i| keep[i]).collect()
}

/// Upstream `Asset.dfs_order`.
pub fn dfs_order(parents: &[i64]) -> Vec<usize> {
    let n = parents.len();
    let mut sons = vec![vec![]; n];
    let mut stack = vec![];
    for (i, &p) in parents.iter().enumerate() {
        if p == -1 {
            stack.push(i);
        } else {
            sons[p as usize].push(i);
        }
    }
    let mut order = vec![];
    while let Some(u) = stack.pop() {
        order.push(u);
        for &s in sons[u].iter().rev() {
            stack.push(s);
        }
    }
    order
}

/// Parents after keeping only `kept` joints: nearest kept ancestor.
pub fn reparent(parents: &[i64], kept: &[usize]) -> Vec<i64> {
    let pos: std::collections::HashMap<usize, usize> = kept.iter().enumerate().map(|(i, &k)| (k, i)).collect();
    kept.iter()
        .map(|&k| {
            let mut p = parents[k];
            while p >= 0 {
                if let Some(&i) = pos.get(&(p as usize)) {
                    return i as i64;
                }
                p = parents[p as usize];
            }
            -1
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_keeps_skinned_and_branch_joints() {
        // root(no skin) -> a(skin) -> leaf(no skin); root -> b(skin)
        let parents = [-1, 0, 1, 0];
        let kept = trim(&parents, &[false, true, false, true]);
        assert_eq!(kept, vec![0, 1, 3]);
        assert_eq!(reparent(&parents, &kept), vec![-1, 0, 0]);
    }

    #[test]
    fn top4_normalizes() {
        let (j, w) = top4(&[vec![0.1, 0.5, 0.0, 0.2, 0.05, 0.3]]);
        assert_eq!(j[0], [1, 5, 3, 0]);
        assert!((w[0].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
