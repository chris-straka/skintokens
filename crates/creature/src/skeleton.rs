//! A skeleton laid along the body's own centre lines: hips and spine
//! through the trunk, a chain down every limb, the tail, the neck and
//! head, the wings. Joints sit where a limb's line turns (a knee, a hock)
//! or, where it runs straight, at set fractions of its length; a left and
//! a right limb get the same joints. Every joint is on a centre line, so
//! every bone is inside the body.

use skintokens::geom::{self, V3};

use crate::parts::Body;
use crate::roles::{self, Anatomy, Kind, Limb, Role, Side};

#[derive(Clone, Debug)]
pub struct Bone {
    pub name: String,
    pub parent: Option<usize>,
    pub head: V3,
    pub tail: V3,
    /// Index into the anatomy's limbs; None for hips and spine.
    pub limb: Option<usize>,
    /// Hips or spine. (A bone of a checked rig may be neither: it lies
    /// along no part of the body.)
    pub trunk: bool,
    /// Place in its chain, from the body outward (0 first).
    pub index: usize,
    /// Thickness of the body at the head (inscribed radius).
    pub radius: f64,
}

pub struct Skeleton {
    pub bones: Vec<Bone>,
}

/// A limb's own length (past the body it grows from) per spine-sized bone.
const SPINE_BONE: f64 = 0.13;
const TAIL_BONE: f64 = 0.09;

/// Interior joints of a leg or an arm: arc lengths (from the limb's start,
/// as fractions of its own length) where the centre line turns most, at
/// least `min_turn` degrees and `spacing` apart; filled up to `want` with
/// set fractions where the line runs straight.
fn bends(l: &Limb, want: usize, most: usize, defaults: &[f64]) -> Vec<f64> {
    let mut v = bends_min(l, most, &[], 16.0);
    let spacing = 0.16;
    for &d in defaults {
        if v.len() >= want {
            break;
        }
        if v.iter().all(|&g| (g - d).abs() >= spacing) {
            v.push(d);
        }
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

fn bends_min(l: &Limb, most: usize, defaults: &[f64], min_turn: f64) -> Vec<f64> {
    let want = 0;
    let own = l.length - l.start;
    if own <= 0.0 {
        return vec![];
    }
    let n = 48usize;
    let pts: Vec<V3> = (0..=n).map(|k| l.at(l.start + own * k as f64 / n as f64).0).collect();
    let w = 5usize; // about a tenth of the limb each way
    let mut turn = vec![0.0f64; n + 1];
    for k in w..=n - w {
        let a = geom::sub(pts[k], pts[k - w]);
        let b = geom::sub(pts[k + w], pts[k]);
        let c = geom::dot(a, b) / (geom::norm(a) * geom::norm(b)).max(1e-12);
        turn[k] = c.clamp(-1.0, 1.0).acos().to_degrees();
    }
    let spacing = 0.16;
    let mut peaks: Vec<(f64, f64)> = (w..=n - w)
        .filter(|&k| turn[k] >= min_turn && turn[k] >= turn[k - 1] && turn[k] >= turn[k + 1])
        .map(|k| (turn[k], k as f64 / n as f64))
        .filter(|&(_, f)| (0.14..=0.88).contains(&f))
        .collect();
    peaks.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let mut out: Vec<f64> = vec![];
    for (_, f) in peaks {
        if out.len() < most && out.iter().all(|&g| (g - f).abs() >= spacing) {
            out.push(f);
        }
    }
    for &d in defaults {
        if out.len() >= want {
            break;
        }
        if out.iter().all(|&g| (g - d).abs() >= spacing) {
            out.push(d);
        }
    }
    out.sort_by(|a, b| a.partial_cmp(b).unwrap());
    out
}

/// Joint positions of one limb as fractions of its own length (0 = where
/// it leaves the body), and the stem of its bones' names.
fn limb_joints(l: &Limb, body: &Body, kind: Kind) -> Vec<f64> {
    let own = l.length - l.start;
    let size = body.size;
    match l.role {
        Role::Leg => {
            let most = if kind == Kind::FourLegged || kind == Kind::ManyLimbed { 3 } else { 2 };
            let mut v = vec![0.0];
            v.extend(bends(l, 2, most, &[0.45, 0.8, 0.25]));
            v
        }
        Role::Arm => {
            let mut v = vec![0.0];
            v.extend(bends(l, 2, 2, &[0.45, 0.82]));
            v
        }
        Role::Tail => {
            let n = ((own / (TAIL_BONE * size)).round() as usize).clamp(2, 5);
            (0..n).map(|k| k as f64 / n as f64).collect()
        }
        Role::Head => {
            if own < 0.18 * size {
                return vec![0.0];
            }
            // the skull starts where the line turns down into the head (a
            // horse's poll), else where the neck is thinnest
            let n = 40;
            let (mut best, mut at) = (f64::INFINITY, 0.5);
            for k in (n * 3 / 10)..=(3 * n / 4) {
                let f = k as f64 / n as f64;
                let r = l.at(l.start + own * f).1;
                if r < best - 1e-9 {
                    best = r;
                    at = f;
                }
            }
            if let Some(&turn) = bends_min(l, 1, &[], 25.0).iter().find(|f| (0.3..=0.8).contains(*f)) {
                at = turn;
            }
            if at * own >= 0.2 * size {
                vec![0.0, at * 0.5, at]
            } else {
                vec![0.0, at]
            }
        }
        Role::Wing => vec![0.0, 0.35, 0.68],
        Role::Extra => {
            let n = ((own / (0.15 * size)).round() as usize).clamp(1, 3);
            (0..n).map(|k| k as f64 / n as f64).collect()
        }
    }
}

fn stem_and_side(l: &Limb) -> (String, &'static str) {
    let side = match l.side {
        Side::Left => ".L",
        Side::Right => ".R",
        Side::Centre => "",
    };
    (l.name.trim_end_matches(".L").trim_end_matches(".R").to_string(), side)
}

impl Skeleton {
    pub fn fit(body: &Body, an: &Anatomy) -> Skeleton {
        let mut bones: Vec<Bone> = vec![];
        let thick = |p: V3| {
            let c = body.solid.nearest_inside(p, 4).unwrap_or(body.seed);
            (body.solid.depth[c] as f64 * body.solid.cell).max(1.5 * body.solid.cell)
        };
        // --- hips and spine
        let ls = roles::length(&an.spine);
        let n_spine = ((ls / (SPINE_BONE * body.size)).round() as usize).clamp(1, 4);
        let step = ls / (n_spine + 1) as f64;
        for k in 0..=n_spine {
            let head = roles::at(&an.spine, &an.spine_radius, k as f64 * step).0;
            let tail = roles::at(&an.spine, &an.spine_radius, (k + 1) as f64 * step).0;
            bones.push(Bone {
                name: if k == 0 { "DEF-hips".into() } else { format!("DEF-spine.{k:03}") },
                parent: if k == 0 { None } else { Some(k - 1) },
                head,
                tail,
                limb: None,
                trunk: true,
                index: k,
                radius: thick(head),
            });
        }
        // --- limbs: joint fractions, the same on a left and its right
        let mut fracs: Vec<Vec<f64>> = an.limbs.iter().map(|l| limb_joints(l, body, an.kind)).collect();
        for i in 0..an.limbs.len() {
            for j in i + 1..an.limbs.len() {
                let (a, b) = (&an.limbs[i], &an.limbs[j]);
                if stem_and_side(a).0 == stem_and_side(b).0 && a.side != b.side && a.side != Side::Centre && b.side != Side::Centre {
                    let (fa, fb) = (fracs[i].clone(), fracs[j].clone());
                    let both = if fa.len() == fb.len() {
                        fa.iter().zip(&fb).map(|(x, y)| 0.5 * (x + y)).collect()
                    } else if fa.len() > fb.len() {
                        fa
                    } else {
                        fb
                    };
                    fracs[i] = both.clone();
                    fracs[j] = both;
                }
            }
        }
        for (li, l) in an.limbs.iter().enumerate() {
            let own = l.length - l.start;
            // hangs from the spine bone its base is nearest to
            let base = l.base();
            let parent = (0..=n_spine)
                .min_by(|&a, &b| {
                    let d = |k: usize| seg_dist(base, bones[k].head, bones[k].tail);
                    d(a).partial_cmp(&d(b)).unwrap()
                })
                .unwrap();
            let (stem, side) = stem_and_side(l);
            let f = &fracs[li];
            let mut prev = parent;
            for (k, &fk) in f.iter().enumerate() {
                let head = l.at(l.start + own * fk).0;
                let tail = if k + 1 < f.len() { l.at(l.start + own * f[k + 1]).0 } else { l.tip() };
                let name = if l.role == Role::Head && k + 1 == f.len() {
                    "DEF-head".to_string()
                } else if f.len() == 1 || (l.role == Role::Head && f.len() == 2) {
                    format!("DEF-{stem}{side}")
                } else {
                    format!("DEF-{stem}.{:03}{side}", k + 1)
                };
                bones.push(Bone { name, parent: Some(prev), head, tail, limb: Some(li), trunk: false, index: k, radius: thick(head) });
                prev = bones.len() - 1;
            }
        }
        // a bone too short to turn about is given a little length
        for b in bones.iter_mut() {
            if geom::norm(geom::sub(b.tail, b.head)) < 0.5 * body.solid.cell {
                b.tail = geom::add(b.head, [0.0, body.solid.cell, 0.0]);
            }
        }
        Skeleton { bones }
    }

    pub fn children(&self, b: usize) -> Vec<usize> {
        (0..self.bones.len()).filter(|&i| self.bones[i].parent == Some(b)).collect()
    }
}

pub fn seg_dist(p: V3, a: V3, b: V3) -> f64 {
    let ab = geom::sub(b, a);
    let t = (geom::dot(geom::sub(p, a), ab) / geom::dot(ab, ab).max(1e-18)).clamp(0.0, 1.0);
    geom::norm(geom::sub(p, geom::add(a, geom::scale(ab, t))))
}
