//! Small f64 geometry helpers: glTF column-major 4x4 matrices, rotations,
//! and a k-d tree for nearest-neighbour queries.

pub type V3 = [f64; 3];
/// Column-major 4x4 (glTF `matrix` layout): m[col][row].
pub type M4 = [[f64; 4]; 4];

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
pub fn dist2(a: V3, b: V3) -> f64 {
    let d = sub(a, b);
    dot(d, d)
}

pub const IDENTITY: M4 = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]];

pub fn mul(a: &M4, b: &M4) -> M4 {
    let mut r = [[0.0; 4]; 4];
    for c in 0..4 {
        for row in 0..4 {
            r[c][row] = (0..4).map(|k| a[k][row] * b[c][k]).sum();
        }
    }
    r
}

pub fn apply(m: &M4, p: V3) -> V3 {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2] + m[3][0],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2] + m[3][1],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2] + m[3][2],
    ]
}

pub fn apply_dir(m: &M4, p: V3) -> V3 {
    [
        m[0][0] * p[0] + m[1][0] * p[1] + m[2][0] * p[2],
        m[0][1] * p[0] + m[1][1] * p[1] + m[2][1] * p[2],
        m[0][2] * p[0] + m[1][2] * p[1] + m[2][2] * p[2],
    ]
}

pub fn is_identity(m: &M4) -> bool {
    (0..4).all(|c| (0..4).all(|r| (m[c][r] - IDENTITY[c][r]).abs() < 1e-9))
}

/// General 4x4 inverse (cofactors).
pub fn inverse(m: &M4) -> M4 {
    let a: Vec<f64> = (0..16).map(|i| m[i / 4][i % 4]).collect(); // column-major flat
    let mut inv = [0.0f64; 16];
    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
    let det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
    let d = if det.abs() < 1e-300 { 0.0 } else { 1.0 / det };
    let mut r = [[0.0; 4]; 4];
    for i in 0..16 {
        r[i / 4][i % 4] = inv[i] * d;
    }
    r
}

/// Rotation (columns = axes) + translation -> M4.
pub fn from_rt(cols: [V3; 3], t: V3) -> M4 {
    [
        [cols[0][0], cols[0][1], cols[0][2], 0.0],
        [cols[1][0], cols[1][1], cols[1][2], 0.0],
        [cols[2][0], cols[2][1], cols[2][2], 0.0],
        [t[0], t[1], t[2], 1.0],
    ]
}

/// glTF TRS -> M4 (quaternion x, y, z, w).
pub fn from_trs(t: V3, q: [f64; 4], s: V3) -> M4 {
    let [x, y, z, w] = q;
    let r = [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w)],
        [2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w)],
        [2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y)],
    ];
    from_rt([scale(r[0], s[0]), scale(r[1], s[1]), scale(r[2], s[2])], t)
}

/// Rotation part (orthonormal columns) -> quaternion (x, y, z, w).
pub fn quat_from_cols(c: [V3; 3]) -> [f64; 4] {
    // m[row][col]
    let m = |r: usize, k: usize| c[k][r];
    let tr = m(0, 0) + m(1, 1) + m(2, 2);
    let q = if tr > 0.0 {
        let s = (tr + 1.0).sqrt() * 2.0;
        [(m(2, 1) - m(1, 2)) / s, (m(0, 2) - m(2, 0)) / s, (m(1, 0) - m(0, 1)) / s, 0.25 * s]
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let s = (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt() * 2.0;
        [0.25 * s, (m(0, 1) + m(1, 0)) / s, (m(0, 2) + m(2, 0)) / s, (m(2, 1) - m(1, 2)) / s]
    } else if m(1, 1) > m(2, 2) {
        let s = (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt() * 2.0;
        [(m(0, 1) + m(1, 0)) / s, 0.25 * s, (m(1, 2) + m(2, 1)) / s, (m(0, 2) - m(2, 0)) / s]
    } else {
        let s = (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt() * 2.0;
        [(m(0, 2) + m(2, 0)) / s, (m(1, 2) + m(2, 1)) / s, 0.25 * s, (m(1, 0) - m(0, 1)) / s]
    };
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let q = [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
    if q[3] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    }
}

pub fn cols(m: &M4) -> [V3; 3] {
    [[m[0][0], m[0][1], m[0][2]], [m[1][0], m[1][1], m[1][2]], [m[2][0], m[2][1], m[2][2]]]
}

pub fn translation(m: &M4) -> V3 {
    [m[3][0], m[3][1], m[3][2]]
}

/// Blender's `vec_roll_to_mat3_normalized` with roll 0: the rest rotation
/// of an edit bone pointing along `nor` (columns = bone X, Y, Z; Y = nor).
pub fn bone_rotation(nor: V3) -> [V3; 3] {
    const SAFE: f64 = 6.1e-3;
    const CRITICAL: f64 = 2.5e-4;
    let [x, y, z] = nor;
    let mut theta = 1.0 + y;
    let theta_alt = x * x + z * z;
    // b[col][row]
    let mut b = [[0.0f64; 3]; 3];
    if theta > SAFE || theta_alt > CRITICAL {
        b[0][1] = -x;
        b[1][0] = x;
        b[1][1] = y;
        b[1][2] = z;
        b[2][1] = -z;
        if theta <= SAFE {
            theta = theta_alt * 0.5 + theta_alt * theta_alt * 0.125;
        }
        b[0][0] = 1.0 - x * x / theta;
        b[2][2] = 1.0 - z * z / theta;
        b[2][0] = -x * z / theta;
        b[0][2] = -x * z / theta;
    } else {
        b = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    [b[0], b[1], b[2]]
}

/// Static k-d tree over 3D points (indices into the original array).
pub struct KdTree {
    pts: Vec<V3>,
    nodes: Vec<(usize, u8)>, // implicit tree over a permutation: (point index, split axis)
}

impl KdTree {
    pub fn new(pts: Vec<V3>) -> Self {
        let mut idx: Vec<usize> = (0..pts.len()).collect();
        let mut nodes = vec![(0usize, 0u8); pts.len()];
        fn build(pts: &[V3], idx: &mut [usize], out: &mut [(usize, u8)], depth: usize) {
            if idx.is_empty() {
                return;
            }
            // split on the widest axis of this cell
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            for &i in idx.iter() {
                for a in 0..3 {
                    lo[a] = lo[a].min(pts[i][a]);
                    hi[a] = hi[a].max(pts[i][a]);
                }
            }
            let axis = (0..3).max_by(|&a, &b| (hi[a] - lo[a]).partial_cmp(&(hi[b] - lo[b])).unwrap()).unwrap();
            let mid = idx.len() / 2;
            idx.select_nth_unstable_by(mid, |&a, &b| pts[a][axis].partial_cmp(&pts[b][axis]).unwrap());
            out[mid] = (idx[mid], axis as u8);
            let (l, r) = idx.split_at_mut(mid);
            let (ol, or) = out.split_at_mut(mid);
            build(pts, l, ol, depth + 1);
            build(pts, &mut r[1..], &mut or[1..], depth + 1);
        }
        build(&pts, &mut idx, &mut nodes, 0);
        KdTree { pts, nodes }
    }

    /// k nearest (distance, index), nearest first.
    pub fn knn(&self, q: V3, k: usize) -> Vec<(f64, usize)> {
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
        self.search(0, self.nodes.len(), q, k, &mut best);
        best.iter().map(|&(d2, i)| (d2.sqrt(), i)).collect()
    }

    fn search(&self, lo: usize, hi: usize, q: V3, k: usize, best: &mut Vec<(f64, usize)>) {
        if lo >= hi {
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let (pi, axis) = self.nodes[mid];
        let p = self.pts[pi];
        let d2 = dist2(p, q);
        if best.len() < k || d2 < best[best.len() - 1].0 {
            let pos = best.partition_point(|x| x.0 <= d2);
            best.insert(pos, (d2, pi));
            best.truncate(k);
        }
        let diff = q[axis as usize] - p[axis as usize];
        let (near, far) = if diff < 0.0 { ((lo, mid), (mid + 1, hi)) } else { ((mid + 1, hi), (lo, mid)) };
        self.search(near.0, near.1, q, k, best);
        if best.len() < k || diff * diff < best[best.len() - 1].0 {
            self.search(far.0, far.1, q, k, best);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kdtree_matches_brute_force() {
        let mut r = skintokens_nn::Rng::new(3);
        let pts: Vec<V3> = (0..2000).map(|_| [r.uniform(), r.uniform(), r.uniform()]).collect();
        let t = KdTree::new(pts.clone());
        for _ in 0..50 {
            let q = [r.uniform(), r.uniform(), r.uniform()];
            let got = t.knn(q, 8);
            let mut all: Vec<(f64, usize)> = pts.iter().enumerate().map(|(i, p)| (dist2(*p, q).sqrt(), i)).collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            assert_eq!(got.iter().map(|x| x.1).collect::<Vec<_>>(), all[..8].iter().map(|x| x.1).collect::<Vec<_>>());
        }
    }

    #[test]
    fn inverse_and_quat() {
        let m = from_trs([1.0, 2.0, 3.0], [0.1825742, 0.3651484, 0.5477226, 0.7302967], [1.0, 1.0, 1.0]);
        let p = mul(&m, &inverse(&m));
        assert!(is_identity(&p));
        let q = quat_from_cols(cols(&m));
        assert!((q[0] - 0.1825742).abs() < 1e-6 && (q[3] - 0.7302967).abs() < 1e-6);
        let b = bone_rotation([0.0, 0.0, 1.0]);
        assert!((b[1][2] - 1.0).abs() < 1e-12);
    }
}
