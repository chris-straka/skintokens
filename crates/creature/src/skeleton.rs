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
    /// A limb's bone that lies inside the body the limb grows from (a
    /// shoulder, a hip): it swings the limb, it is not the upper arm.
    pub inside: bool,
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
    let own = l.length - l.emerge;
    if own <= 0.0 {
        return vec![];
    }
    let n = 48usize;
    let pts: Vec<V3> = (0..=n)
        .map(|k| l.at(l.emerge + own * k as f64 / n as f64).0)
        .collect();
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

/// Joint positions of one limb as fractions of its length outside the
/// body (0 = where it comes out; negative = inside, toward the middle of
/// the body).
fn limb_joints(l: &Limb, body: &Body, kind: Kind) -> Vec<f64> {
    let own = l.length - l.emerge;
    let size = body.size;
    // A leg or an arm swings from the shoulder or the hip, inside the
    // body: an upper bone from half way in to where the limb comes out,
    // when there is that much body to cross.
    // (The limb's line starts on the trunk's own centre line, so "half way
    // in" is half way between the spine and the surface: about where a
    // shoulder or a hip joint is.)
    let upper = || -> Vec<f64> {
        let inside = l.emerge;
        if inside >= 0.1 * size && own > 0.0 {
            vec![-0.5 * inside / own, 0.0]
        } else {
            vec![0.0]
        }
    };
    match l.role {
        Role::Leg => {
            let most = if kind == Kind::FourLegged || kind == Kind::ManyLimbed {
                3
            } else {
                2
            };
            // the foot: a joint where the leg comes within a tenth of the
            // body's height of the ground (an ankle, a pastern), so the last
            // bone is the foot and stands on the ground
            let (ground, height) = (body.lo[1], body.hi[1] - body.lo[1]);
            let n = 40;
            let foot = (0..=n)
                .map(|k| 0.78 + 0.15 * k as f64 / n as f64)
                .find(|&f| l.at(l.emerge + own * f).0[1] <= ground + 0.1 * height)
                .unwrap_or(0.93);
            let mut v = upper();
            v.extend(
                bends(l, 1, most, &[0.45, 0.25])
                    .into_iter()
                    .filter(|&f| f <= foot - 0.12),
            );
            v.push(foot);
            v
        }
        Role::Arm => {
            let mut v = upper();
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
                let r = l.at(l.emerge + own * f).1;
                if r < best - 1e-9 {
                    best = r;
                    at = f;
                }
            }
            if let Some(&turn) = bends_min(l, 1, &[], 25.0)
                .iter()
                .find(|f| (0.3..=0.8).contains(*f))
            {
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
    (
        l.name
            .trim_end_matches(".L")
            .trim_end_matches(".R")
            .to_string(),
        side,
    )
}

/// Where another rig of this body (the model's prediction) puts its
/// joints along a leg or an arm, as fractions of the limb's length outside
/// the body: the heads of that rig's bones that lie on the limb's centre
/// line, at least two of them, well apart. The model has seen where knees
/// and hocks are; the centre line knows where the inside of the body is.
fn hinted(l: &Limb, body: &Body, hint: &[Bone], most: usize) -> Option<Vec<f64>> {
    let own = l.length - l.emerge;
    if own <= 0.0 {
        return None;
    }
    let mut at: Vec<f64> = vec![];
    for b in hint {
        // nearest point of the limb's line to this joint
        let (mut best, mut s_best, mut s) = (f64::INFINITY, 0.0, 0.0);
        for k in 1..l.line.len() {
            let (p, q) = (l.line[k - 1], l.line[k]);
            let pq = geom::sub(q, p);
            let len = geom::norm(pq);
            let t = if len > 0.0 {
                (geom::dot(geom::sub(b.head, p), pq) / (len * len)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let d = geom::norm(geom::sub(b.head, geom::add(p, geom::scale(pq, t))));
            if d < best {
                best = d;
                s_best = s + t * len;
            }
            s += len;
        }
        let r = l.at(s_best).1;
        let f = (s_best - l.emerge) / own;
        if best <= (1.5 * r).max(0.04 * body.size) && (0.12..=0.9).contains(&f) {
            at.push(f);
        }
    }
    at.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut out: Vec<f64> = vec![];
    for f in at {
        if out.last().is_none_or(|&g| f - g >= 0.14) {
            out.push(f);
        }
    }
    if out.len() < 2 {
        return None;
    }
    while out.len() > most {
        // drop the joint nearest to a neighbour
        let k = (0..out.len()).min_by(|&a, &b| {
            let gap = |i: usize| {
                (if i > 0 { out[i] - out[i - 1] } else { out[i] }).min(if i + 1 < out.len() {
                    out[i + 1] - out[i]
                } else {
                    1.0 - out[i]
                })
            };
            gap(a).partial_cmp(&gap(b)).unwrap()
        });
        out.remove(k.unwrap());
    }
    Some(out)
}

impl Skeleton {
    /// `hint`: another rig's bones (see `hinted`), used for where a leg's
    /// or an arm's joints go along it.
    pub fn fit_with(body: &Body, an: &Anatomy, hint: Option<&[Bone]>) -> Skeleton {
        let mut sk = Self::fit_inner(body, an, hint);
        sk.bones.iter_mut().for_each(|b| b.trunk = b.limb.is_none());
        sk
    }

    pub fn fit(body: &Body, an: &Anatomy) -> Skeleton {
        Self::fit_with(body, an, None)
    }

    fn fit_inner(body: &Body, an: &Anatomy, hint: Option<&[Bone]>) -> Skeleton {
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
                name: if k == 0 {
                    "DEF-hips".into()
                } else {
                    format!("DEF-spine.{k:03}")
                },
                parent: if k == 0 { None } else { Some(k - 1) },
                head,
                tail,
                limb: None,
                trunk: true,
                index: k,
                inside: false,
                radius: thick(head),
            });
        }
        // --- limbs: joint fractions, the same on a left and its right
        let mut fracs: Vec<Vec<f64>> = an
            .limbs
            .iter()
            .map(|l| limb_joints(l, body, an.kind))
            .collect();
        let hinted_fracs = |fracs: &mut Vec<Vec<f64>>| {
            if let Some(h) = hint {
                for (li, l) in an.limbs.iter().enumerate() {
                    if l.role != Role::Leg && l.role != Role::Arm {
                        continue;
                    }
                    let most = if l.role == Role::Leg
                        && (an.kind == Kind::FourLegged || an.kind == Kind::ManyLimbed)
                    {
                        3
                    } else {
                        2
                    };
                    if let Some(mid) = hinted(l, body, h, most) {
                        // keep what comes before the limb is out of the body (the
                        // shoulder or hip) and, on a leg, the foot joint
                        let mut v: Vec<f64> =
                            fracs[li].iter().copied().filter(|&f| f <= 0.0).collect();
                        if l.role == Role::Leg {
                            let foot = *fracs[li].last().unwrap();
                            v.extend(mid.into_iter().filter(|&f| f <= foot - 0.12));
                            v.push(foot);
                        } else {
                            v.extend(mid);
                        }
                        fracs[li] = v;
                    }
                }
            }
        };
        hinted_fracs(&mut fracs);
        // no bone shorter than 6% of the body's longest side: a short leg
        // gets fewer bones, not small ones. The first joint stays; so does a
        // leg's foot joint (the joint before it goes instead).
        for (li, l) in an.limbs.iter().enumerate() {
            let own = l.length - l.emerge;
            let min = 0.06 * body.size / own.max(1e-9);
            let f = &fracs[li];
            let mut kept: Vec<f64> = vec![f[0]];
            for (k, &x) in f.iter().enumerate().skip(1) {
                let last = *kept.last().unwrap();
                if x - last >= min {
                    kept.push(x);
                } else if l.role == Role::Leg && k + 1 == f.len() && kept.len() > 1 {
                    *kept.last_mut().unwrap() = x;
                }
            }
            // the last bone, to the tip, has to be a bone too
            while kept.len() > 1 && 1.0 - *kept.last().unwrap() < 0.5 * min {
                kept.pop();
            }
            fracs[li] = kept;
        }
        for i in 0..an.limbs.len() {
            for j in i + 1..an.limbs.len() {
                let (a, b) = (&an.limbs[i], &an.limbs[j]);
                if stem_and_side(a).0 == stem_and_side(b).0
                    && a.side != b.side
                    && a.side != Side::Centre
                    && b.side != Side::Centre
                {
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
        // each limb hangs from the spine bone its base is nearest to; a
        // left and its right from the same one (nearest to the middle of
        // their two bases)
        let mut hang: Vec<usize> = vec![0; an.limbs.len()];
        for (li, l) in an.limbs.iter().enumerate() {
            let mut base = l.base();
            for (lj, m) in an.limbs.iter().enumerate() {
                if lj != li
                    && stem_and_side(m).0 == stem_and_side(l).0
                    && m.side != l.side
                    && m.side != Side::Centre
                    && l.side != Side::Centre
                {
                    base = geom::scale(geom::add(l.base(), m.base()), 0.5);
                }
            }
            hang[li] = (0..=n_spine)
                .min_by(|&a, &b| {
                    let d = |k: usize| seg_dist(base, bones[k].head, bones[k].tail);
                    d(a).partial_cmp(&d(b)).unwrap().then(a.cmp(&b))
                })
                .unwrap();
        }
        for (li, l) in an.limbs.iter().enumerate() {
            let own = l.length - l.emerge;
            let parent = hang[li];
            let (stem, side) = stem_and_side(l);
            let f = &fracs[li];
            let mut prev = parent;
            for (k, &fk) in f.iter().enumerate() {
                let head = l.at(l.emerge + own * fk).0;
                let tail = if k + 1 < f.len() {
                    l.at(l.emerge + own * f[k + 1]).0
                } else {
                    l.tip()
                };
                let name = if l.role == Role::Head && k + 1 == f.len() {
                    "DEF-head".to_string()
                } else if f.len() == 1 || (l.role == Role::Head && f.len() == 2) {
                    format!("DEF-{stem}{side}")
                } else {
                    format!("DEF-{stem}.{:03}{side}", k + 1)
                };
                bones.push(Bone {
                    name,
                    parent: Some(prev),
                    head,
                    tail,
                    limb: Some(li),
                    trunk: false,
                    index: k,
                    inside: fk < -1e-9,
                    radius: thick(head),
                });
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
        (0..self.bones.len())
            .filter(|&i| self.bones[i].parent == Some(b))
            .collect()
    }
}

pub fn seg_dist(p: V3, a: V3, b: V3) -> f64 {
    let ab = geom::sub(b, a);
    let t = (geom::dot(geom::sub(p, a), ab) / geom::dot(ab, ab).max(1e-18)).clamp(0.0, 1.0);
    geom::norm(geom::sub(p, geom::add(a, geom::scale(ab, t))))
}
