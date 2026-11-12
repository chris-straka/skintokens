//! The body as a solid: which cells of a grid are inside the mesh, and how
//! deep. Generated meshes are one fused shell with holes, loose flakes and
//! parts pushed through each other, so "inside" is voted: along each axis
//! the surface crossings are counted with their facing (a winding number,
//! which overlapping shells don't cancel), a column whose count doesn't
//! close is read from both ends, and a cell is inside when two of the three
//! axes say so. Cells the surface passes through always count, so a tail
//! or a wing thinner than a cell stays connected.

use skintokens::geom::{self, V3};

pub struct Solid {
    pub dims: [usize; 3],
    /// Corner of cell (0, 0, 0).
    pub origin: V3,
    pub cell: f64,
    pub inside: Vec<bool>,
    /// Distance to the outside, in cells (chamfer 3-4-5, about Euclidean).
    pub depth: Vec<f32>,
}

pub const NEIGHBOURS: [[i32; 3]; 26] = {
    let mut n = [[0i32; 3]; 26];
    let mut k = 0;
    let mut x = -1;
    while x <= 1 {
        let mut y = -1;
        while y <= 1 {
            let mut z = -1;
            while z <= 1 {
                if x != 0 || y != 0 || z != 0 {
                    n[k] = [x, y, z];
                    k += 1;
                }
                z += 1;
            }
            y += 1;
        }
        x += 1;
    }
    n
};

impl Solid {
    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
    pub fn idx(&self, c: [usize; 3]) -> usize {
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }
    pub fn coords(&self, i: usize) -> [usize; 3] {
        [i % self.dims[0], (i / self.dims[0]) % self.dims[1], i / (self.dims[0] * self.dims[1])]
    }
    pub fn center(&self, i: usize) -> V3 {
        let c = self.coords(i);
        [0, 1, 2].map(|a| self.origin[a] + (c[a] as f64 + 0.5) * self.cell)
    }
    /// The cell holding `p`, clamped into the grid.
    pub fn cell_of(&self, p: V3) -> [usize; 3] {
        [0, 1, 2].map(|a| (((p[a] - self.origin[a]) / self.cell).floor().max(0.0) as usize).min(self.dims[a] - 1))
    }
    /// Neighbour of cell `i` by `d`, if in the grid.
    pub fn step(&self, i: usize, d: [i32; 3]) -> Option<usize> {
        let c = self.coords(i);
        let mut o = [0usize; 3];
        for a in 0..3 {
            let v = c[a] as i64 + d[a] as i64;
            if v < 0 || v >= self.dims[a] as i64 {
                return None;
            }
            o[a] = v as usize;
        }
        Some(self.idx(o))
    }
    pub fn is_inside(&self, p: V3) -> bool {
        (0..3).all(|a| p[a] >= self.origin[a] && p[a] < self.origin[a] + self.dims[a] as f64 * self.cell) && self.inside[self.idx(self.cell_of(p))]
    }
    /// The inside cell nearest to `p` (itself if inside), searching outward
    /// up to `reach` cells.
    pub fn nearest_inside(&self, p: V3, reach: usize) -> Option<usize> {
        let c = self.cell_of(p);
        let i = self.idx(c);
        if self.inside[i] {
            return Some(i);
        }
        let mut best: Option<(f64, usize)> = None;
        for r in 1..=reach as i64 {
            for dz in -r..=r {
                for dy in -r..=r {
                    for dx in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        let q = [c[0] as i64 + dx, c[1] as i64 + dy, c[2] as i64 + dz];
                        if (0..3).any(|a| q[a] < 0 || q[a] >= self.dims[a] as i64) {
                            continue;
                        }
                        let j = self.idx([q[0] as usize, q[1] as usize, q[2] as usize]);
                        if self.inside[j] {
                            let d = geom::dist2(self.center(j), p);
                            if best.is_none_or(|b| d < b.0) {
                                best = Some((d, j));
                            }
                        }
                    }
                }
            }
            if let Some(b) = best {
                // a nearer cell can still sit in the next shell's corners
                if (r as f64) * self.cell > b.0.sqrt() || r == reach as i64 {
                    return Some(b.1);
                }
            }
        }
        best.map(|b| b.1)
    }

    /// `res` cells along the longest side, two cells of margin around.
    pub fn build(verts: &[V3], tris: &[[u32; 3]], res: usize) -> Solid {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for v in verts {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        let ext = geom::sub(hi, lo);
        let cell = (ext[0].max(ext[1]).max(ext[2]) / res as f64).max(1e-9);
        let pad = 2.0;
        let origin = [0, 1, 2].map(|a| lo[a] - pad * cell);
        let dims = [0, 1, 2].map(|a| (ext[a] / cell).ceil() as usize + 2 * pad as usize + 1);
        let mut s = Solid { dims, origin, cell, inside: vec![], depth: vec![] };
        let n = s.len();
        let mut surface = vec![false; n];
        for t in tris {
            let [a, b, c] = t.map(|i| verts[i as usize]);
            let longest = geom::norm(geom::sub(b, a)).max(geom::norm(geom::sub(c, a))).max(geom::norm(geom::sub(c, b)));
            let k = ((longest / (0.5 * cell)).ceil() as usize).max(1);
            for i in 0..=k {
                for j in 0..=(k - i) {
                    let (u, v) = (i as f64 / k as f64, j as f64 / k as f64);
                    let p = geom::add(a, geom::add(geom::scale(geom::sub(b, a), u), geom::scale(geom::sub(c, a), v)));
                    let ci = s.idx(s.cell_of(p));
                    surface[ci] = true;
                }
            }
        }
        let mut votes = vec![0u8; n];
        for axis in 0..3 {
            s.vote(verts, tris, axis, &mut votes);
        }
        s.inside = (0..n).map(|i| votes[i] >= 2 || surface[i]).collect();
        s.keep_body();
        s.depth = s.chamfer();
        s
    }

    /// One axis's say: cells whose winding count is positive read from
    /// both ends of their column get a vote.
    fn vote(&self, verts: &[V3], tris: &[[u32; 3]], axis: usize, votes: &mut [u8]) {
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let (nu, nv, na) = (self.dims[u], self.dims[v], self.dims[axis]);
        // crossings per column: (depth along the axis in cells, +1 entering / -1 leaving)
        let mut cols: Vec<Vec<(f32, i8)>> = vec![vec![]; nu * nv];
        // column centres sit a hair off the cell centres so they don't run
        // exactly along mesh edges
        let (ju, jv) = (0.5 + 1.3e-4, 0.5 - 0.7e-4);
        for t in tris {
            let p = t.map(|i| verts[i as usize]);
            let n = geom::cross(geom::sub(p[1], p[0]), geom::sub(p[2], p[0]));
            if n[axis].abs() < 1e-14 {
                continue;
            }
            let sign: i8 = if n[axis] < 0.0 { 1 } else { -1 };
            let q = p.map(|x| [(x[u] - self.origin[u]) / self.cell, (x[v] - self.origin[v]) / self.cell, (x[axis] - self.origin[axis]) / self.cell]);
            let (ulo, uhi) = (q.iter().map(|x| x[0]).fold(f64::INFINITY, f64::min), q.iter().map(|x| x[0]).fold(f64::NEG_INFINITY, f64::max));
            let (vlo, vhi) = (q.iter().map(|x| x[1]).fold(f64::INFINITY, f64::min), q.iter().map(|x| x[1]).fold(f64::NEG_INFINITY, f64::max));
            let d = (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (q[1][1] - q[0][1]);
            if d.abs() < 1e-14 {
                continue;
            }
            let (i0, i1) = (((ulo - ju).ceil().max(0.0)) as usize, ((uhi - ju).floor().min(nu as f64 - 1.0)).max(-1.0) as i64);
            let (j0, j1) = (((vlo - jv).ceil().max(0.0)) as usize, ((vhi - jv).floor().min(nv as f64 - 1.0)).max(-1.0) as i64);
            for j in j0 as i64..=j1 {
                for i in i0 as i64..=i1 {
                    let (x, y) = (i as f64 + ju, j as f64 + jv);
                    let w1 = ((x - q[0][0]) * (q[2][1] - q[0][1]) - (q[2][0] - q[0][0]) * (y - q[0][1])) / d;
                    let w2 = ((q[1][0] - q[0][0]) * (y - q[0][1]) - (x - q[0][0]) * (q[1][1] - q[0][1])) / d;
                    if w1 < 0.0 || w2 < 0.0 || w1 + w2 > 1.0 {
                        continue;
                    }
                    let z = q[0][2] + w1 * (q[1][2] - q[0][2]) + w2 * (q[2][2] - q[0][2]);
                    cols[j as usize * nu + i as usize].push((z as f32, sign));
                }
            }
        }
        let mut c = [0usize; 3];
        for j in 0..nv {
            for i in 0..nu {
                let col = &mut cols[j * nu + i];
                if col.is_empty() {
                    continue;
                }
                col.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                // winding in front of each crossing, read forwards and backwards
                let mut fwd = vec![0i32; col.len() + 1];
                for (k, x) in col.iter().enumerate() {
                    fwd[k + 1] = fwd[k] + x.1 as i32;
                }
                let mut back = vec![0i32; col.len() + 1];
                for k in (0..col.len()).rev() {
                    back[k] = back[k + 1] - col[k].1 as i32;
                }
                let mut k = 0;
                for a in 0..na {
                    let z = a as f32 + 0.5;
                    while k < col.len() && col[k].0 < z {
                        k += 1;
                    }
                    if fwd[k] > 0 && back[k] > 0 {
                        c[u] = i;
                        c[v] = j;
                        c[axis] = a;
                        votes[self.idx(c)] += 1;
                    }
                }
            }
        }
    }

    /// Drops loose flakes: inside cells not joined (through faces, edges or
    /// corners) to a piece holding at least 2% of the largest one's cells.
    fn keep_body(&mut self) {
        let n = self.len();
        let mut label = vec![u32::MAX; n];
        let mut sizes: Vec<usize> = vec![];
        let mut stack = vec![];
        for s in 0..n {
            if !self.inside[s] || label[s] != u32::MAX {
                continue;
            }
            let id = sizes.len() as u32;
            let mut count = 0;
            label[s] = id;
            stack.push(s);
            while let Some(i) = stack.pop() {
                count += 1;
                for d in NEIGHBOURS {
                    if let Some(j) = self.step(i, d) {
                        if self.inside[j] && label[j] == u32::MAX {
                            label[j] = id;
                            stack.push(j);
                        }
                    }
                }
            }
            sizes.push(count);
        }
        let most = sizes.iter().copied().max().unwrap_or(0);
        for i in 0..n {
            if self.inside[i] && sizes[label[i] as usize] * 50 < most {
                self.inside[i] = false;
            }
        }
    }

    /// Distance to the nearest outside cell, in cells (two-pass 3-4-5 chamfer).
    fn chamfer(&self) -> Vec<f32> {
        let n = self.len();
        let big = i32::MAX / 2;
        let mut d: Vec<i32> = self.inside.iter().map(|&b| if b { big } else { 0 }).collect();
        let cost = |o: [i32; 3]| match o[0].abs() + o[1].abs() + o[2].abs() {
            1 => 3,
            2 => 4,
            _ => 5,
        };
        let before: Vec<[i32; 3]> = NEIGHBOURS.iter().copied().filter(|o| (o[2], o[1], o[0]) < (0, 0, 0)).collect();
        let after: Vec<[i32; 3]> = NEIGHBOURS.iter().copied().filter(|o| (o[2], o[1], o[0]) > (0, 0, 0)).collect();
        for i in 0..n {
            if d[i] == 0 {
                continue;
            }
            for &o in &before {
                match self.step(i, o) {
                    Some(j) => d[i] = d[i].min(d[j] + cost(o)),
                    None => d[i] = d[i].min(cost(o)),
                }
            }
        }
        for i in (0..n).rev() {
            if d[i] == 0 {
                continue;
            }
            for &o in &after {
                match self.step(i, o) {
                    Some(j) => d[i] = d[i].min(d[j] + cost(o)),
                    None => d[i] = d[i].min(cost(o)),
                }
            }
        }
        d.into_iter().map(|x| x as f32 / 3.0).collect()
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// A closed box from `lo` to `hi`, faces outward.
    pub fn cuboid(lo: V3, hi: V3, base: u32) -> (Vec<V3>, Vec<[u32; 3]>) {
        let v: Vec<V3> = (0..8).map(|i| [if i & 1 == 0 { lo[0] } else { hi[0] }, if i & 2 == 0 { lo[1] } else { hi[1] }, if i & 4 == 0 { lo[2] } else { hi[2] }]).collect();
        let quads = [[0, 4, 6, 2], [1, 3, 7, 5], [0, 1, 5, 4], [2, 6, 7, 3], [0, 2, 3, 1], [4, 5, 7, 6]];
        let mut f = vec![];
        for q in quads {
            f.push([base + q[0], base + q[1], base + q[2]]);
            f.push([base + q[0], base + q[2], base + q[3]]);
        }
        (v, f)
    }

    #[test]
    fn a_box_is_full_and_its_outside_empty() {
        let (v, f) = cuboid([0.0, 0.0, 0.0], [1.0, 2.0, 0.5], 0);
        let s = Solid::build(&v, &f, 40);
        assert!(s.is_inside([0.5, 1.0, 0.25]));
        assert!(!s.is_inside([1.2, 1.0, 0.25]));
        let filled = s.inside.iter().filter(|&&b| b).count() as f64 * s.cell.powi(3);
        assert!((filled - 1.0).abs() < 0.25, "volume {filled}");
        // deepest in the middle of the thin side: about 0.25 deep
        let deep = s.depth.iter().cloned().fold(0.0, f32::max) as f64 * s.cell;
        assert!((deep - 0.25).abs() < 0.08, "depth {deep}");
    }

    #[test]
    fn a_hole_and_an_overlap_don_t_empty_it() {
        // two boxes pushed through each other, one with a face missing
        let (mut v, mut f) = cuboid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 0);
        let (v2, f2) = cuboid([0.5, 0.2, 0.2], [2.0, 0.8, 0.8], 8);
        v.extend(v2);
        f.extend(f2);
        f.truncate(f.len() - 2); // the second box loses its +z face
        let s = Solid::build(&v, &f, 48);
        for p in [[0.25, 0.5, 0.5], [0.75, 0.5, 0.5], [1.5, 0.5, 0.5]] {
            assert!(s.is_inside(p), "{p:?} should be inside");
        }
        assert!(!s.is_inside([1.5, 0.5, 1.2]));
        assert!(!s.is_inside([1.5, 0.95, 0.5]));
    }
}
