//! The body's parts, read from its volume. Distance through the inside is
//! measured from the deepest point of the body; cells at about the same
//! distance form shells, and a shell falls into separate pieces wherever
//! the body does (two pieces past the chest: the neck and a leg). Linking
//! each piece to the one it grew out of gives a tree, and its long
//! branches are the limbs, the tail, the neck, the wings. Each branch
//! carries the middle of its pieces as a line down its centre, which lies
//! inside the body by construction, and its thickness along that line.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use skintokens::geom::{self, V3};

use crate::solid::{Solid, NEIGHBOURS};

/// One piece of one shell.
pub struct Node {
    pub band: usize,
    pub cells: Vec<u32>,
    pub centroid: V3,
    /// Deepest cell's distance to the outside (world units): the radius of
    /// the largest ball that fits here.
    pub radius: f64,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Farthest distance from the seed in this node's subtree (cells).
    pub far: f32,
    /// The cell where that distance is reached.
    pub far_cell: u32,
    /// Cells in the subtree.
    pub volume: usize,
}

/// A branch of the tree between two forks (or a fork and a tip).
#[derive(Clone)]
pub struct Part {
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    /// Shell pieces from the base outward.
    pub nodes: Vec<usize>,
    /// Centre line from the base (where it leaves its parent) to the tip.
    pub line: Vec<V3>,
    /// Thickness (inscribed radius) at each point of `line`.
    pub radius: Vec<f64>,
    pub length: f64,
    /// Cells of the part itself, bumps too small to be parts included.
    pub cells: usize,
    /// Lowest and highest point of the part's cells (y).
    pub lo_y: f64,
    pub hi_y: f64,
    /// Spread of its cells along their three principal directions (standard
    /// deviations, largest first): a sheet has a small third.
    pub spread: [f64; 3],
}

impl Part {
    pub fn base(&self) -> V3 {
        self.line[0]
    }
    pub fn tip(&self) -> V3 {
        *self.line.last().unwrap()
    }
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
    /// Point at arc length `s` from the base, and the thickness there.
    pub fn at(&self, s: f64) -> (V3, f64) {
        let mut left = s.max(0.0);
        for k in 1..self.line.len() {
            let seg = geom::norm(geom::sub(self.line[k], self.line[k - 1]));
            if left <= seg || k + 1 == self.line.len() {
                let t = if seg > 0.0 { (left / seg).min(1.0) } else { 0.0 };
                let p = geom::add(self.line[k - 1], geom::scale(geom::sub(self.line[k], self.line[k - 1]), t));
                return (p, self.radius[k - 1] + (self.radius[k] - self.radius[k - 1]) * t);
            }
            left -= seg;
        }
        (self.line[0], self.radius[0])
    }
}

pub struct Body {
    pub solid: Solid,
    /// Distance through the inside from the seed, in cells (infinite: not
    /// reached).
    pub dist: Vec<f32>,
    pub seed: usize,
    pub nodes: Vec<Node>,
    pub parts: Vec<Part>,
    /// Part of each inside cell (u32::MAX outside or unreached).
    pub part_of: Vec<u32>,
    /// Longest side of the body.
    pub size: f64,
    pub lo: V3,
    pub hi: V3,
}

/// Shell thickness, in cells.
pub const BAND: f32 = 3.0;

/// What makes a branch a part: at least this long (fraction of the body's
/// longest side) ...
pub const MIN_LENGTH: f64 = 0.07;
/// ... this much of the body's volume ...
pub const MIN_VOLUME: f64 = 0.002;
/// ... and at least this long measured in the thickness of what it grows
/// out of (a broad bump is not a limb).
pub const MIN_REACH_IN_RADII: f64 = 0.8;

impl Body {
    pub fn build(verts: &[V3], tris: &[[u32; 3]], res: usize) -> Body {
        let solid = Solid::build(verts, tris, res);
        let n = solid.len();
        let seed = (0..n).filter(|&i| solid.inside[i]).max_by(|&a, &b| solid.depth[a].partial_cmp(&solid.depth[b]).unwrap().then(b.cmp(&a))).expect("an empty solid");
        let dist = geodesic(&solid, &[seed]);
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for v in verts {
            for a in 0..3 {
                lo[a] = lo[a].min(v[a]);
                hi[a] = hi[a].max(v[a]);
            }
        }
        let size = (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max);
        let mut b = Body { solid, dist, seed, nodes: vec![], parts: vec![], part_of: vec![u32::MAX; n], size, lo, hi };
        b.shells();
        b.branches();
        b
    }

    /// Cuts the distance field into shells and links their pieces.
    fn shells(&mut self) {
        let s = &self.solid;
        let n = s.len();
        let band_of = |i: usize| (self.dist[i] / BAND) as usize;
        let mut node_of = vec![u32::MAX; n];
        let mut order: Vec<usize> = (0..n).filter(|&i| self.dist[i].is_finite()).collect();
        order.sort_by(|&a, &b| self.dist[a].partial_cmp(&self.dist[b]).unwrap().then(a.cmp(&b)));
        let mut stack = vec![];
        for &st in &order {
            if node_of[st] != u32::MAX {
                continue;
            }
            let id = self.nodes.len() as u32;
            let band = band_of(st);
            let mut cells = vec![];
            node_of[st] = id;
            stack.push(st);
            while let Some(i) = stack.pop() {
                cells.push(i as u32);
                for d in NEIGHBOURS {
                    if let Some(j) = s.step(i, d) {
                        if node_of[j] == u32::MAX && self.dist[j].is_finite() && band_of(j) == band {
                            node_of[j] = id;
                            stack.push(j);
                        }
                    }
                }
            }
            let mut c = [0.0; 3];
            let mut deep = 0f32;
            let mut far = (0f32, st as u32);
            for &i in &cells {
                c = geom::add(c, s.center(i as usize));
                deep = deep.max(s.depth[i as usize]);
                if self.dist[i as usize] > far.0 {
                    far = (self.dist[i as usize], i);
                }
            }
            let volume = cells.len();
            self.nodes.push(Node {
                band,
                centroid: geom::scale(c, 1.0 / cells.len() as f64),
                radius: deep as f64 * s.cell,
                cells,
                parent: None,
                children: vec![],
                far: far.0,
                far_cell: far.1,
                volume,
            });
        }
        // each piece grew out of the piece of the shell below it that it
        // touches most
        for id in 0..self.nodes.len() {
            if self.nodes[id].band == 0 {
                continue;
            }
            let mut touch: Vec<(u32, usize)> = vec![];
            for &i in &self.nodes[id].cells {
                for d in NEIGHBOURS {
                    if let Some(j) = s.step(i as usize, d) {
                        let o = node_of[j];
                        if o != u32::MAX && self.nodes[o as usize].band + 1 == self.nodes[id].band {
                            match touch.iter_mut().find(|t| t.0 == o) {
                                Some(t) => t.1 += 1,
                                None => touch.push((o, 1)),
                            }
                        }
                    }
                }
            }
            if let Some(&(p, _)) = touch.iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))) {
                self.nodes[id].parent = Some(p as usize);
                self.nodes[p as usize].children.push(id);
            }
        }
        // subtree reach and volume, leaves first (nodes are in distance order)
        for id in (0..self.nodes.len()).rev() {
            if let Some(p) = self.nodes[id].parent {
                let (far, far_cell, vol) = (self.nodes[id].far, self.nodes[id].far_cell, self.nodes[id].volume);
                let pn = &mut self.nodes[p];
                pn.volume += vol;
                if far > pn.far {
                    pn.far = far;
                    pn.far_cell = far_cell;
                }
            }
        }
        // the per-cell map is rebuilt per part in `branches`
        self.part_of = node_of;
    }

    /// Whether the branch starting at `child` is a part of its own.
    fn is_part(&self, child: usize) -> bool {
        let c = &self.nodes[child];
        let Some(p) = c.parent else { return true };
        let cell = self.solid.cell;
        let reach = (c.far - c.band as f32 * BAND) as f64 * cell;
        let total = self.nodes.iter().filter(|n| n.parent.is_none()).map(|n| n.volume).sum::<usize>();
        reach >= MIN_LENGTH * self.size && c.volume as f64 >= MIN_VOLUME * total as f64 && reach >= MIN_REACH_IN_RADII * self.nodes[p].radius
    }

    /// Walks the tree from the seed and cuts it into parts at its forks.
    fn branches(&mut self) {
        let node_of = std::mem::replace(&mut self.part_of, vec![]);
        let mut part_of_node = vec![u32::MAX; self.nodes.len()];
        let roots: Vec<usize> = (0..self.nodes.len()).filter(|&i| self.nodes[i].parent.is_none()).collect();
        let mut parts: Vec<Part> = vec![];
        // (first node, parent part)
        let mut todo: Vec<(usize, Option<usize>)> = roots.iter().map(|&r| (r, None)).collect();
        while let Some((first, parent)) = todo.pop() {
            let id = parts.len();
            let mut nodes = vec![];
            let mut cur = first;
            let mut forks: Vec<usize> = vec![];
            loop {
                nodes.push(cur);
                part_of_node[cur] = id as u32;
                let kids: Vec<usize> = self.nodes[cur].children.iter().copied().filter(|&k| self.is_part(k)).collect();
                // bumps: everything below a child that is not a part stays with this one
                let mut bumps: Vec<usize> = self.nodes[cur].children.iter().copied().filter(|k| !kids.contains(k)).collect();
                while let Some(b) = bumps.pop() {
                    part_of_node[b] = id as u32;
                    bumps.extend(self.nodes[b].children.iter().copied());
                }
                match kids.len() {
                    0 => break,
                    1 => cur = kids[0],
                    _ => {
                        forks = kids;
                        break;
                    }
                }
            }
            if let Some(p) = parent {
                parts[p].children.push(id);
            }
            for f in forks {
                todo.push((f, Some(id)));
            }
            parts.push(Part { parent, children: vec![], nodes, line: vec![], radius: vec![], length: 0.0, cells: 0, lo_y: 0.0, hi_y: 0.0, spread: [0.0; 3] });
        }
        self.part_of = node_of.iter().map(|&nd| if nd == u32::MAX { u32::MAX } else { part_of_node[nd as usize] }).collect();
        // centre lines
        for id in 0..parts.len() {
            let mut line: Vec<V3> = parts[id].nodes.iter().map(|&n| self.nodes[n].centroid).collect();
            let mut radius: Vec<f64> = parts[id].nodes.iter().map(|&n| self.nodes[n].radius).collect();
            let last = *parts[id].nodes.last().unwrap();
            if parts[id].children.is_empty() {
                // a tip: run the line out to the farthest cell
                let tip = self.solid.center(self.nodes[last].far_cell as usize);
                if geom::norm(geom::sub(tip, *line.last().unwrap())) > 0.5 * self.solid.cell {
                    line.push(tip);
                    radius.push(self.solid.depth[self.nodes[last].far_cell as usize] as f64 * self.solid.cell);
                }
            }
            if let Some(p) = parts[id].parent {
                let pn = *parts[p].nodes.last().unwrap();
                line.insert(0, self.nodes[pn].centroid);
                radius.insert(0, self.nodes[pn].radius);
            }
            // smooth the middle, keep the ends
            for _ in 0..2 {
                let old = line.clone();
                for k in 1..line.len().saturating_sub(1) {
                    line[k] = geom::scale(geom::add(geom::add(old[k - 1], geom::scale(old[k], 2.0)), old[k + 1]), 0.25);
                }
            }
            parts[id].length = (1..line.len()).map(|k| geom::norm(geom::sub(line[k], line[k - 1]))).sum();
            parts[id].line = line;
            parts[id].radius = radius;
        }
        // per-part cell statistics
        let np = parts.len();
        let mut sum = vec![[0.0f64; 3]; np];
        let mut cnt = vec![0usize; np];
        let mut lo = vec![f64::INFINITY; np];
        let mut hi = vec![f64::NEG_INFINITY; np];
        for i in 0..self.part_of.len() {
            let p = self.part_of[i];
            if p == u32::MAX {
                continue;
            }
            let c = self.solid.center(i);
            sum[p as usize] = geom::add(sum[p as usize], c);
            cnt[p as usize] += 1;
            lo[p as usize] = lo[p as usize].min(c[1]);
            hi[p as usize] = hi[p as usize].max(c[1]);
        }
        let mut cov = vec![[[0.0f64; 3]; 3]; np];
        for i in 0..self.part_of.len() {
            let p = self.part_of[i];
            if p == u32::MAX {
                continue;
            }
            let d = geom::sub(self.solid.center(i), geom::scale(sum[p as usize], 1.0 / cnt[p as usize] as f64));
            for a in 0..3 {
                for b in 0..3 {
                    cov[p as usize][a][b] += d[a] * d[b];
                }
            }
        }
        for id in 0..np {
            parts[id].cells = cnt[id];
            parts[id].lo_y = lo[id] - 0.5 * self.solid.cell;
            parts[id].hi_y = hi[id] + 0.5 * self.solid.cell;
            let n = cnt[id].max(1) as f64;
            let m = cov[id].map(|r| r.map(|x| x / n));
            let mut e = eigenvalues_sym3(m).map(|x| x.max(0.0).sqrt());
            e.sort_by(|a, b| b.partial_cmp(a).unwrap());
            parts[id].spread = e;
        }
        self.parts = parts;
    }
}

/// Distance through the inside from `seeds` to every inside cell, in cells
/// (steps to the 26 neighbours cost their length).
pub fn geodesic(s: &Solid, seeds: &[usize]) -> Vec<f32> {
    let mut dist = vec![f32::INFINITY; s.len()];
    let mut heap = BinaryHeap::new();
    for &i in seeds {
        dist[i] = 0.0;
        heap.push(Reverse((0u32, i as u32)));
    }
    let cost = |d: [i32; 3]| match d[0].abs() + d[1].abs() + d[2].abs() {
        1 => 1.0f32,
        2 => std::f32::consts::SQRT_2,
        _ => 1.732_050_8,
    };
    while let Some(Reverse((db, i))) = heap.pop() {
        let di = f32::from_bits(db);
        if di > dist[i as usize] {
            continue;
        }
        for d in NEIGHBOURS {
            if let Some(j) = s.step(i as usize, d) {
                if s.inside[j] {
                    let nd = di + cost(d);
                    if nd < dist[j] {
                        dist[j] = nd;
                        heap.push(Reverse((nd.to_bits(), j as u32)));
                    }
                }
            }
        }
    }
    dist
}

/// Eigenvalues of a symmetric 3x3 matrix (trigonometric solution).
pub fn eigenvalues_sym3(m: [[f64; 3]; 3]) -> [f64; 3] {
    let p1 = m[0][1] * m[0][1] + m[0][2] * m[0][2] + m[1][2] * m[1][2];
    let q = (m[0][0] + m[1][1] + m[2][2]) / 3.0;
    let p2 = (m[0][0] - q).powi(2) + (m[1][1] - q).powi(2) + (m[2][2] - q).powi(2) + 2.0 * p1;
    let p = (p2 / 6.0).sqrt();
    if p < 1e-18 {
        return [q, q, q];
    }
    let b = |i: usize, j: usize| (m[i][j] - if i == j { q } else { 0.0 }) / p;
    let det = b(0, 0) * (b(1, 1) * b(2, 2) - b(1, 2) * b(2, 1)) - b(0, 1) * (b(1, 0) * b(2, 2) - b(1, 2) * b(2, 0)) + b(0, 2) * (b(1, 0) * b(2, 1) - b(1, 1) * b(2, 0));
    let phi = (det / 2.0).clamp(-1.0, 1.0).acos() / 3.0;
    let e1 = q + 2.0 * p * phi.cos();
    let e3 = q + 2.0 * p * (phi + 2.0 * std::f64::consts::PI / 3.0).cos();
    [e1, 3.0 * q - e1 - e3, e3]
}
