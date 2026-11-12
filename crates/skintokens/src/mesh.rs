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
    sample_with_faces(vertices, faces, vnorm, fnorm, rng).0
}

/// `sample`, also giving the face each sample lies on (`NO_FACE` for a
/// mesh without area, whose samples are vertices). Same draws, same points.
pub fn sample_with_faces(vertices: &[V3], faces: &[[u32; 3]], vnorm: &[V3], fnorm: &[V3], rng: &mut Rng) -> (Vec<[f32; 6]>, Vec<u32>) {
    let mut out = Vec::with_capacity(NUM_SAMPLES);
    let mut on = Vec::with_capacity(NUM_SAMPLES);
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
            on.push(NO_FACE);
        }
        return (out, on);
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
        on.push(fi as u32);
    }
    (out, on)
}

pub const NO_FACE: u32 = u32::MAX;

/// `Asset.from_data`: per-vertex skin from the sampled points' skin by
/// inverse distance over the 8 nearest samples, measured in a straight line
/// (upstream's transfer; `--transfer straight`).
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

/// How the samples' skin reaches the mesh vertices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    /// Upstream's: the 8 nearest samples in a straight line through space,
    /// whatever surface they lie on. Parts that nearly touch (a coat beside
    /// a leg, two thighs) take each other's weights.
    Straight,
    /// The 8 nearest samples (straight line) that lie on the vertex's own
    /// connected piece (welded across UV seams) and whose surface faces the
    /// same side as the vertex (normals less than 90 degrees apart).
    Facing,
    /// The 8 nearest samples along the surface: a walk over the welded
    /// mesh's edges to a corner of the sample's face, then straight across
    /// that face.
    Surface,
}

impl Transfer {
    /// What `rig` and `skin` use unless told otherwise (`--transfer`,
    /// SKINTOKENS_TRANSFER). Upstream's: on four generated humans the
    /// surface copy changed almost no vertex and no fault
    /// (docs/evaluation.md, "Weights along the surface").
    pub const DEFAULT: Transfer = Transfer::Straight;
    pub const ALL: [Transfer; 3] = [Transfer::Straight, Transfer::Facing, Transfer::Surface];

    pub fn parse(s: &str) -> Option<Transfer> {
        Transfer::ALL.into_iter().find(|t| t.name() == s)
    }

    pub fn name(self) -> &'static str {
        match self {
            Transfer::Straight => "straight",
            Transfer::Facing => "facing",
            Transfer::Surface => "surface",
        }
    }
}

/// What a transfer did, for the report.
#[derive(Clone, Debug, Default)]
pub struct TransferStats {
    pub vertices: usize,
    /// Distinct positions after welding UV seams.
    pub welded: usize,
    /// Connected pieces of the welded mesh.
    pub pieces: usize,
    /// Vertices that found no surface-near sample and took the straight
    /// line's 8 nearest instead.
    pub fallback: usize,
    /// Vertices that found fewer than 8 surface-near samples (and used
    /// what they found).
    pub short: usize,
}

/// Samples a vertex borrows from.
const K: usize = 8;
/// Welding distance (normalized frame, longest side 2).
const WELD_EPS: f64 = 1e-5;
/// Farthest walk along the surface (normalized frame: about 9 cm on a
/// 1.8 m character); a vertex with nothing closer falls back.
const SURFACE_REACH: f64 = 0.1;
/// Straight-line candidates the facing test picks its 8 from.
const FACING_POOL: usize = 48;

/// Per-vertex skin from per-sample skin by `method` (all inverse distance
/// over up to 8 samples). `sample_faces` from `sample_with_faces`.
pub fn transfer(
    method: Transfer,
    samples: &[[f32; 6]],
    sample_faces: &[u32],
    skin: &[Vec<f32>],
    vertices: &[V3],
    faces: &[[u32; 3]],
) -> (Vec<Vec<f32>>, TransferStats) {
    let mut st = TransferStats { vertices: vertices.len(), ..Default::default() };
    if method == Transfer::Straight || samples.is_empty() {
        return (transfer_skin(samples, skin, vertices), st);
    }
    let surf = Welded::new(vertices, faces);
    st.welded = surf.reps.len();
    st.pieces = surf.pieces;
    let pts: Vec<V3> = samples.iter().map(|s| [s[0] as f64, s[1] as f64, s[2] as f64]).collect();
    let tree = KdTree::new(pts.clone());
    let j = skin.first().map(|r| r.len()).unwrap_or(0);
    let blend = |nn: &[(f64, usize)]| -> Vec<f32> {
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
    };
    // one search per welded position; split copies share it
    let picks: Vec<Option<Vec<(f64, usize)>>> = match method {
        Transfer::Straight => unreachable!(),
        Transfer::Facing => {
            let piece_of_face: Vec<u32> = faces.iter().map(|f| surf.piece[surf.id[f[0] as usize] as usize]).collect();
            par_map(surf.reps.len(), |w| {
                if surf.faces_of[w].is_empty() {
                    return None;
                }
                let (p, n) = (surf.reps[w], surf.normals[w]);
                let piece = surf.piece[w];
                let has_n = geom::dot(n, n) > 0.5;
                let nn: Vec<(f64, usize)> = tree
                    .knn(p, FACING_POOL.min(pts.len()))
                    .into_iter()
                    .filter(|&(_, s)| {
                        let f = sample_faces[s];
                        if f == NO_FACE {
                            return true;
                        }
                        let sn = [samples[s][3] as f64, samples[s][4] as f64, samples[s][5] as f64];
                        piece_of_face[f as usize] == piece && (!has_n || geom::dot(sn, n) > 0.0)
                    })
                    .take(K)
                    .collect();
                (!nn.is_empty()).then_some(nn)
            })
        }
        Transfer::Surface => {
            // samples per face
            let mut start = vec![0u32; faces.len() + 1];
            for &f in sample_faces {
                if f != NO_FACE {
                    start[f as usize + 1] += 1;
                }
            }
            for i in 0..faces.len() {
                start[i + 1] += start[i];
            }
            let mut fill = start.clone();
            let mut by_face = vec![0u32; start[faces.len()] as usize];
            for (s, &f) in sample_faces.iter().enumerate() {
                if f != NO_FACE {
                    by_face[fill[f as usize] as usize] = s as u32;
                    fill[f as usize] += 1;
                }
            }
            let on_face = |f: u32| &by_face[start[f as usize] as usize..start[f as usize + 1] as usize];
            par_map_with(
                surf.reps.len(),
                || (vec![f64::INFINITY; surf.reps.len()], Vec::<u32>::new()),
                |(dist, touched), w| {
                    if surf.faces_of[w].is_empty() {
                        return None;
                    }
                    let nn = surf.walk(w, dist, touched, |f| on_face(f), &pts, SURFACE_REACH);
                    (!nn.is_empty()).then_some(nn)
                },
            )
        }
    };
    let out = vertices
        .iter()
        .enumerate()
        .map(|(i, &v)| match &picks[surf.id[i] as usize] {
            Some(nn) => {
                if nn.len() < K.min(samples.len()) {
                    st.short += 1;
                }
                blend(nn)
            }
            None => {
                st.fallback += 1;
                blend(&tree.knn(v, K.min(samples.len())))
            }
        })
        .collect();
    (out, st)
}

/// A mesh welded across its UV seams: one id per position, the edges
/// between them, each position's faces and normal, and connected pieces.
struct Welded {
    /// Welded id of each vertex.
    id: Vec<u32>,
    reps: Vec<V3>,
    /// (neighbour, edge length)
    adj: Vec<Vec<(u32, f64)>>,
    faces_of: Vec<Vec<u32>>,
    normals: Vec<V3>,
    piece: Vec<u32>,
    pieces: usize,
}

impl Welded {
    fn new(vertices: &[V3], faces: &[[u32; 3]]) -> Welded {
        let (id, reps) = weld(vertices, WELD_EPS);
        let n = reps.len();
        let mut adj: Vec<Vec<(u32, f64)>> = vec![vec![]; n];
        let mut faces_of: Vec<Vec<u32>> = vec![vec![]; n];
        let mut acc = vec![[0.0f64; 3]; n];
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn root(p: &mut [u32], mut x: u32) -> u32 {
            while p[x as usize] != x {
                p[x as usize] = p[p[x as usize] as usize];
                x = p[x as usize];
            }
            x
        }
        for (fi, f) in faces.iter().enumerate() {
            let w = f.map(|i| id[i as usize]);
            let [a, b, c] = w.map(|i| reps[i as usize]);
            let fnrm = geom::cross(geom::sub(b, a), geom::sub(c, a)); // area-weighted
            for k in 0..3 {
                let (x, y) = (w[k], w[(k + 1) % 3]);
                if !faces_of[x as usize].contains(&(fi as u32)) {
                    faces_of[x as usize].push(fi as u32);
                    acc[x as usize] = geom::add(acc[x as usize], fnrm);
                }
                if x != y {
                    let d = geom::norm(geom::sub(reps[x as usize], reps[y as usize]));
                    adj[x as usize].push((y, d));
                    adj[y as usize].push((x, d));
                    let (rx, ry) = (root(&mut parent, x), root(&mut parent, y));
                    if rx != ry {
                        parent[rx as usize] = ry;
                    }
                }
            }
        }
        for a in adj.iter_mut() {
            a.sort_by_key(|e| e.0);
            a.dedup_by_key(|e| e.0);
        }
        let mut piece_id = std::collections::HashMap::new();
        let piece: Vec<u32> = (0..n as u32)
            .map(|x| {
                let r = root(&mut parent, x);
                let k = piece_id.len() as u32;
                *piece_id.entry(r).or_insert(k)
            })
            .collect();
        let normals = acc
            .into_iter()
            .map(|v| {
                let l = geom::norm(v);
                if l > 0.0 {
                    geom::scale(v, 1.0 / l)
                } else {
                    [0.0; 3]
                }
            })
            .collect();
        Welded { id, reps, adj, faces_of, normals, piece, pieces: piece_id.len() }
    }

    /// The K nearest samples to welded position `w` along the surface
    /// (distance, sample), nearest first, out to `reach`: Dijkstra over the
    /// welded edges; a sample on a face at corner c is d(c) + |sample - c|
    /// (exact on the faces around `w`, an upper bound beyond).
    fn walk<'a>(
        &self,
        w: usize,
        dist: &mut [f64],
        touched: &mut Vec<u32>,
        on_face: impl Fn(u32) -> &'a [u32],
        pts: &[V3],
        reach: f64,
    ) -> Vec<(f64, usize)> {
        use std::cmp::Reverse;
        use std::collections::{BinaryHeap, HashMap};
        let mut heap = BinaryHeap::new();
        let mut cand: HashMap<u32, f64> = HashMap::new();
        let mut scratch: Vec<f64> = vec![];
        dist[w] = 0.0;
        touched.push(w as u32);
        heap.push(Reverse((0f64.to_bits(), w as u32)));
        while let Some(Reverse((db, c))) = heap.pop() {
            let dc = f64::from_bits(db);
            if dc > dist[c as usize] {
                continue; // stale
            }
            if cand.len() >= K {
                scratch.clear();
                scratch.extend(cand.values());
                let (_, kth, _) = scratch.select_nth_unstable_by(K - 1, |a, b| a.partial_cmp(b).unwrap());
                if *kth <= dc {
                    break;
                }
            }
            let pc = self.reps[c as usize];
            for &f in &self.faces_of[c as usize] {
                for &s in on_face(f) {
                    let d = dc + geom::norm(geom::sub(pts[s as usize], pc));
                    let e = cand.entry(s).or_insert(f64::INFINITY);
                    if d < *e {
                        *e = d;
                    }
                }
            }
            for &(nb, len) in &self.adj[c as usize] {
                let nd = dc + len;
                if nd <= reach && nd < dist[nb as usize] {
                    if dist[nb as usize].is_infinite() {
                        touched.push(nb);
                    }
                    dist[nb as usize] = nd;
                    heap.push(Reverse((nd.to_bits(), nb)));
                }
            }
        }
        for &t in touched.iter() {
            dist[t as usize] = f64::INFINITY;
        }
        touched.clear();
        let mut out: Vec<(f64, usize)> = cand.into_iter().filter(|&(_, d)| d <= reach).map(|(s, d)| (d, s as usize)).collect();
        out.sort_by(|a, b| a.partial_cmp(b).unwrap());
        out.truncate(K);
        out
    }
}

/// Welded id per vertex and the welded positions: vertices within `eps` of
/// a kept position share it (a generated mesh is split at every UV seam).
pub fn weld(vertices: &[V3], eps: f64) -> (Vec<u32>, Vec<V3>) {
    let mut grid: std::collections::HashMap<[i64; 3], Vec<u32>> = std::collections::HashMap::new();
    let mut reps: Vec<V3> = vec![];
    let cell = |p: V3| p.map(|x| (x / eps).floor() as i64);
    let ids = vertices
        .iter()
        .map(|&v| {
            let c = cell(v);
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(list) = grid.get(&[c[0] + dx, c[1] + dy, c[2] + dz]) {
                            if let Some(&w) = list.iter().find(|&&w| geom::dist2(reps[w as usize], v) <= eps * eps) {
                                return w;
                            }
                        }
                    }
                }
            }
            let w = reps.len() as u32;
            reps.push(v);
            grid.entry(c).or_default().push(w);
            w
        })
        .collect();
    (ids, reps)
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4).min(16)
}

fn par_map<T: Send>(n: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    par_map_with(n, || (), |_, i| f(i))
}

/// `(0..n).map(f)` on all cores, each thread with its own scratch from `init`.
fn par_map_with<T: Send, S>(n: usize, init: impl Fn() -> S + Sync, f: impl Fn(&mut S, usize) -> T + Sync) -> Vec<T> {
    let t = threads().max(1);
    let chunk = n.div_ceil(t).max(1);
    let mut parts: Vec<Vec<T>> = Vec::new();
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..n)
            .step_by(chunk)
            .map(|lo| {
                let (init, f) = (&init, &f);
                sc.spawn(move || {
                    let mut s = init();
                    (lo..(lo + chunk).min(n)).map(|i| f(&mut s, i)).collect::<Vec<T>>()
                })
            })
            .collect();
        parts = hs.into_iter().map(|h| h.join().expect("transfer thread panicked")).collect();
    });
    parts.into_iter().flatten().collect()
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

    /// An open tube along z around (cx, 0), split at a UV seam (the seam
    /// column is duplicated, as a generated mesh is), angle 0 facing +x.
    fn tube(cx: f64, r: f64, h: f64, around: usize, rings: usize, base: u32) -> (Vec<V3>, Vec<[u32; 3]>) {
        let mut v = vec![];
        for i in 0..=rings {
            let z = h * i as f64 / rings as f64 - h / 2.0;
            for k in 0..=around {
                let a = std::f64::consts::TAU * k as f64 / around as f64;
                v.push([cx + r * a.cos(), r * a.sin(), z]);
            }
        }
        let row = around as u32 + 1;
        let mut f = vec![];
        for i in 0..rings as u32 {
            for k in 0..around as u32 {
                let (a, b, c, d) = (i * row + k, i * row + k + 1, (i + 1) * row + k + 1, (i + 1) * row + k);
                f.push([base + a, base + b, base + c]);
                f.push([base + a, base + c, base + d]);
            }
        }
        (v, f)
    }

    /// Two tubes side by side, 4 mm apart at 10 cm radius (in a frame
    /// whose longest side is 2), each skinned to its own bone with no
    /// error at the samples: the most weight any vertex takes from the
    /// other tube's bone.
    fn tube_bleed(how: Transfer) -> f32 {
        let (r, gap) = (0.1, 0.004);
        let (va, fa) = tube(-(r + gap / 2.0), r, 2.0, 32, 40, 0);
        let (vb, fb) = tube(r + gap / 2.0, r, 2.0, 32, 40, va.len() as u32);
        let m = Mesh { vertices: [va.clone(), vb].concat(), faces: [fa.clone(), fb].concat(), offsets: vec![0, va.len()] };
        let fnorm = m.face_normals();
        let vnorm = m.vertex_normals(&fnorm);
        let (samples, on) = sample_with_faces(&m.vertices, &m.faces, &vnorm, &fnorm, &mut Rng::new(7));
        let skin: Vec<Vec<f32>> = on.iter().map(|&f| if (f as usize) < fa.len() { vec![1.0, 0.0] } else { vec![0.0, 1.0] }).collect();
        let (w, st) = transfer(how, &samples, &on, &skin, &m.vertices, &m.faces);
        if how != Transfer::Straight {
            assert_eq!(st.pieces, 2);
            assert_eq!(st.welded, 2 * 41 * 32, "UV seam welded");
            assert_eq!(st.fallback, 0);
        }
        w.iter().enumerate().map(|(i, row)| if i < va.len() { row[1] } else { row[0] }).fold(0.0, f32::max)
    }

    #[test]
    fn touching_tubes_keep_their_own_bone() {
        // upstream's straight-line copy mixes the tubes where they face
        // each other (this is the fault the surface copy is for)
        let straight = tube_bleed(Transfer::Straight);
        assert!(straight > 0.2, "straight copy should bleed here, got {straight}");
        for how in [Transfer::Surface, Transfer::Facing] {
            let b = tube_bleed(how);
            assert!(b < 1e-6, "{how:?}: a vertex took {b} of the other tube's bone");
        }
    }

    #[test]
    fn surface_walk_matches_straight_on_one_flat_sheet() {
        // on a flat sheet the walk finds the same 8 samples as a straight
        // line (away from the border), so nothing changes where nothing touches
        let n = 30usize;
        let mut v = vec![];
        let mut f = vec![];
        for i in 0..=n {
            for k in 0..=n {
                v.push([i as f64 / n as f64, k as f64 / n as f64, 0.0]);
            }
        }
        let row = n as u32 + 1;
        for i in 0..n as u32 {
            for k in 0..n as u32 {
                let a = i * row + k;
                f.push([a, a + row, a + row + 1]);
                f.push([a, a + row + 1, a + 1]);
            }
        }
        let m = Mesh { vertices: v, faces: f, offsets: vec![0] };
        let fnorm = m.face_normals();
        let vnorm = m.vertex_normals(&fnorm);
        let (samples, on) = sample_with_faces(&m.vertices, &m.faces, &vnorm, &fnorm, &mut Rng::new(1));
        let skin: Vec<Vec<f32>> = samples.iter().map(|s| vec![s[0], 1.0 - s[0]]).collect();
        let a = transfer_skin(&samples, &skin, &m.vertices);
        let (b, _) = transfer(Transfer::Surface, &samples, &on, &skin, &m.vertices, &m.faces);
        let worst = a.iter().zip(&b).map(|(x, y)| (x[0] - y[0]).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "flat sheet: surface differs from straight by {worst}");
    }

    #[test]
    fn top4_normalizes() {
        let (j, w) = top4(&[vec![0.1, 0.5, 0.0, 0.2, 0.05, 0.3]]);
        assert_eq!(j[0], [1, 5, 3, 0]);
        assert!((w[0].iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
