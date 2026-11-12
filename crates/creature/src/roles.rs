//! What each part of the body is: which branches are legs, arms, the head
//! and neck, the tail, wings, and which are only bumps (ears, toes, fur),
//! and from those what kind of creature it is. Read from where a branch
//! starts and ends, how long and how flat it is; no skeleton is involved.
//!
//! The frame is the cleaned mesh's: +Y up, +Z forward, +X the creature's
//! left, standing on the ground, mirrored about x = 0.

use skintokens::geom::{self, V3};

use crate::parts::Body;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Leg,
    Arm,
    /// Neck and head as one chain.
    Head,
    Tail,
    Wing,
    /// A long appendage that is none of the above (an antenna, a tendril,
    /// something carried).
    Extra,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
    Centre,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    FourLegged,
    TwoLegged,
    Winged,
    ManyLimbed,
    /// Nothing it stands on: a cloak to the ground, a fish, a cloud.
    Legless,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::FourLegged => "four-legged",
            Kind::TwoLegged => "two-legged",
            Kind::Winged => "winged",
            Kind::ManyLimbed => "many-limbed",
            Kind::Legless => "legless",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Limb {
    pub role: Role,
    pub side: Side,
    /// Legs of a four-legged body: the front pair.
    pub front: bool,
    pub name: String,
    /// Part ids from the body outward.
    pub parts: Vec<usize>,
    /// Centre line from the middle of the body it grows out of to its tip.
    pub line: Vec<V3>,
    pub radius: Vec<f64>,
    pub length: f64,
    /// Arc length at which its own first piece begins: from here on the
    /// line is the limb's, though still inside the body it grows out of
    /// (a shoulder, a haunch).
    pub start: f64,
    /// Arc length at which it has come out of that body: the body around
    /// the line is no thicker than the limb.
    pub emerge: f64,
}

impl Limb {
    pub fn base(&self) -> V3 {
        self.line[0]
    }
    pub fn tip(&self) -> V3 {
        *self.line.last().unwrap()
    }
    /// Point and thickness at arc length `s`.
    pub fn at(&self, s: f64) -> (V3, f64) {
        at(&self.line, &self.radius, s)
    }
}

pub fn at(line: &[V3], radius: &[f64], s: f64) -> (V3, f64) {
    let mut left = s.max(0.0);
    for k in 1..line.len() {
        let seg = geom::norm(geom::sub(line[k], line[k - 1]));
        if left <= seg || k + 1 == line.len() {
            let t = if seg > 0.0 { (left / seg).min(1.0) } else { 0.0 };
            let p = geom::add(line[k - 1], geom::scale(geom::sub(line[k], line[k - 1]), t));
            return (p, radius[k - 1] + (radius[k] - radius[k - 1]) * t);
        }
        left -= seg;
    }
    (line[0], radius[0])
}

pub fn length(line: &[V3]) -> f64 {
    (1..line.len()).map(|k| geom::norm(geom::sub(line[k], line[k - 1]))).sum()
}

pub struct Anatomy {
    pub kind: Kind,
    /// The body stands up (spine nearer vertical than level).
    pub upright: bool,
    pub limbs: Vec<Limb>,
    /// Parts that are the body itself.
    pub trunk: Vec<usize>,
    /// The spine: from the hips (or the lowest or rearmost point of a
    /// legless body) to where the neck starts (or the top or front).
    pub spine: Vec<V3>,
    pub spine_radius: Vec<f64>,
    pub ground: f64,
    pub height: f64,
    /// Role of each part: index into `limbs`, or None for the trunk.
    pub limb_of_part: Vec<Option<usize>>,
    /// +1 when the body faces +Z as a cleaned mesh should, -1 when its toes
    /// say it faces -Z (the clean step turned it the wrong way).
    pub forward: f64,
}

/// A branch counts as a limb of its own (not a toe, an ear, a tuft) when
/// its reach is this fraction of the body's longest side ...
const MAJOR_LENGTH: f64 = 0.15;
/// ... and it holds this fraction of the body's cells.
const MAJOR_VOLUME: f64 = 0.008;
/// A minor branch this long still gets a chain (an antenna, a tendril).
const EXTRA_LENGTH: f64 = 0.22;
/// A minor branch shorter than this at the end of a limb is a digit: the
/// limb's line runs out through the straightest one.
const DIGIT_LENGTH: f64 = 0.12;

struct Tree<'a> {
    body: &'a Body,
    /// Longest way down from each part's base to a tip.
    reach: Vec<f64>,
    /// Cells in each part and everything below it.
    volume: Vec<usize>,
    total: usize,
}

impl Tree<'_> {
    fn major(&self, p: usize) -> bool {
        self.reach[p] >= MAJOR_LENGTH * self.body.size && self.volume[p] as f64 >= MAJOR_VOLUME * self.total as f64
    }
}

impl Anatomy {
    pub fn read(body: &Body) -> Anatomy {
        let np = body.parts.len();
        let mut reach = vec![0.0; np];
        let mut volume = vec![0usize; np];
        // children come after their parents in `parts`
        for p in (0..np).rev() {
            let own = body.parts[p].length;
            let below = body.parts[p].children.iter().map(|&c| reach[c]).fold(0.0, f64::max);
            reach[p] = own + below;
            volume[p] = body.parts[p].cells + body.parts[p].children.iter().map(|&c| volume[c]).sum::<usize>();
        }
        let total: usize = (0..np).filter(|&p| body.parts[p].parent.is_none()).map(|p| volume[p]).sum();
        let t = Tree { body, reach, volume, total };

        // trunk: the root, and any part that forks into two or more major branches
        let mut is_trunk = vec![false; np];
        let mut chains: Vec<Vec<usize>> = vec![];
        let mut todo: Vec<usize> = (0..np).filter(|&p| body.parts[p].parent.is_none()).collect();
        for &r in &todo {
            is_trunk[r] = true;
        }
        while let Some(p) = todo.pop() {
            for &c in &body.parts[p].children {
                let majors = body.parts[c].children.iter().filter(|&&k| t.major(k)).count();
                if majors >= 2 {
                    is_trunk[c] = true;
                    todo.push(c);
                } else if t.major(c) || t.reach[c] >= EXTRA_LENGTH * body.size {
                    chains.push(chain_from(&t, c));
                }
            }
        }
        // a chain that ran into a fork of two majors further out stops there;
        // what lies beyond is found from that trunk part
        let mut k = 0;
        while k < chains.len() {
            let last = *chains[k].last().unwrap();
            let majors: Vec<usize> = body.parts[last].children.iter().copied().filter(|&c| t.major(c)).collect();
            if majors.len() >= 2 {
                // only reachable when chain_from stopped at such a part
                is_trunk[last] = true;
                chains[k].pop();
                for &c in &body.parts[last].children {
                    if t.major(c) || t.reach[c] >= EXTRA_LENGTH * body.size {
                        chains.push(chain_from(&t, c));
                    }
                }
                if chains[k].is_empty() {
                    chains.remove(k);
                    continue;
                }
            }
            k += 1;
        }

        let ground = body.lo[1];
        let height = body.hi[1] - body.lo[1];
        let size = body.size;
        let mut limbs: Vec<Limb> = chains
            .into_iter()
            .map(|parts| {
                let mut line: Vec<V3> = vec![];
                let mut radius: Vec<f64> = vec![];
                for (k, &p) in parts.iter().enumerate() {
                    let skip = if k == 0 { 0 } else { 1 };
                    line.extend_from_slice(&body.parts[p].line[skip..]);
                    radius.extend_from_slice(&body.parts[p].radius[skip..]);
                }
                let length = length(&line);
                // it has left the body where the line passes the first piece of its own
                let start = if line.len() > 1 { geom::norm(geom::sub(line[1], line[0])) } else { 0.0 };
                let mut l = Limb { role: Role::Extra, side: Side::Centre, front: false, name: String::new(), parts, line, radius, length, start: start.min(0.4 * length), emerge: 0.0 };
                l.emerge = l.start.max(emerges(&l)).min(0.4 * length);
                l
            })
            .collect();

        // --- what each limb is
        let mid = 0.07 * size;
        let grounded = |l: &Limb| {
            let low = l.parts.iter().map(|&p| body.parts[p].lo_y).fold(f64::INFINITY, f64::min);
            low <= ground + 0.07 * height && l.tip()[1] <= ground + 0.18 * height && l.base()[1] - l.tip()[1] >= 0.12 * height
        };
        let flat = |l: &Limb| {
            let p = &body.parts[*l.parts.last().unwrap()];
            p.spread[1] >= 0.06 * size && p.spread[2] < 0.3 * p.spread[1]
        };
        // which side a limb is on: where its line runs once it has left the
        // body, against a margin that a narrow animal's legs still clear
        let width = body.hi[0] - body.lo[0];
        let off = (0.015 * size).max(0.06 * width);
        for l in limbs.iter_mut() {
            let own: Vec<f64> = l.line.iter().skip(1).map(|p| p[0]).collect();
            let x = if own.is_empty() { l.tip()[0] } else { own.iter().sum::<f64>() / own.len() as f64 };
            l.side = if x > off {
                Side::Left
            } else if x < -off {
                Side::Right
            } else {
                Side::Centre
            };
            if grounded(l) {
                l.role = Role::Leg;
            } else if l.side != Side::Centre && flat(l) {
                l.role = Role::Wing;
            }
        }
        // a pair of legs has a left and a right: a grounded limb on the
        // midline is a tail or a hem, not a leg
        for l in limbs.iter_mut() {
            if l.role == Role::Leg && l.side == Side::Centre {
                l.role = Role::Extra;
            }
        }
        // wings come in pairs: one flat thing on one side is carried, not flown with
        for side in [Side::Left, Side::Right] {
            let other = if side == Side::Left { Side::Right } else { Side::Left };
            if !limbs.iter().any(|l| l.role == Role::Wing && l.side == other) {
                for l in limbs.iter_mut().filter(|l| l.role == Role::Wing && l.side == side) {
                    l.role = Role::Extra;
                }
            }
        }
        let legs: Vec<usize> = (0..limbs.len()).filter(|&i| limbs[i].role == Role::Leg).collect();
        // posture: the line from where the hind (or lower) legs leave the
        // body to where the front (or upper) legs or the head do
        let mut upright = body.hi[1] - body.lo[1] > body.hi[2] - body.lo[2];
        if legs.len() >= 4 {
            let mut by_z = legs.clone();
            // (the toes are read after this, so front and hind are swapped below if it faces -Z)
            by_z.sort_by(|&a, &b| limbs[a].at(limbs[a].emerge).0[2].partial_cmp(&limbs[b].at(limbs[b].emerge).0[2]).unwrap());
            let mean = |ids: &[usize]| {
                let mut c = [0.0; 3];
                for &i in ids {
                    c = geom::add(c, limbs[i].at(limbs[i].emerge).0);
                }
                geom::scale(c, 1.0 / ids.len() as f64)
            };
            let (hind, fore) = (mean(&by_z[..2]), mean(&by_z[by_z.len() - 2..]));
            let d = geom::sub(fore, hind);
            upright = d[1].abs() > d[2].abs();
            if upright && legs.len() == 4 {
                // standing up with its hands on the ground: the upper pair are arms
                let mut by_y = legs.clone();
                by_y.sort_by(|&a, &b| limbs[a].at(limbs[a].emerge).0[1].partial_cmp(&limbs[b].at(limbs[b].emerge).0[1]).unwrap());
                for &i in &by_y[2..] {
                    limbs[i].role = Role::Arm;
                }
            } else {
                for &i in &by_z[by_z.len() - 2..] {
                    limbs[i].front = true;
                }
            }
        }
        // which way it faces: toes point forward. The cleaned mesh should
        // face +Z; where the clean step measured it wrong, read the body
        // the other way round rather than call the head a tail.
        let mut toes = 0.0;
        for &i in &legs {
            let l = &limbs[i];
            let d = geom::sub(l.tip(), l.at(0.85 * l.length).0);
            toes += d[2] / l.length.max(1e-9);
        }
        let fwd = if !legs.is_empty() && toes < -0.03 * legs.len() as f64 { -1.0 } else { 1.0 };
        if fwd < 0.0 {
            for l in limbs.iter_mut().filter(|l| l.role == Role::Leg) {
                l.front = !l.front && !upright && legs.len() >= 4;
            }
            for l in limbs.iter_mut() {
                l.side = match l.side {
                    Side::Left => Side::Right,
                    Side::Right => Side::Left,
                    Side::Centre => Side::Centre,
                };
            }
        }
        // head and tail: limbs on the midline
        let centre: Vec<usize> = (0..limbs.len()).filter(|&i| limbs[i].role == Role::Extra && limbs[i].side == Side::Centre && limbs[i].base()[0].abs() <= 2.0 * mid).collect();
        // the head: on a body that stands up, the midline limb that starts
        // highest and goes up or forward (a head may hang); on a level body
        // the one that reaches farthest forward
        let ahead_of: Vec<f64> = limbs.iter().map(|l| fwd * (l.tip()[2] - l.base()[2])).collect();
        let ahead = |i: usize| ahead_of[i];
        let head = if upright {
            centre
                .iter()
                .copied()
                .filter(|&i| limbs[i].base()[1] > ground + 0.5 * height && (limbs[i].tip()[1] > limbs[i].base()[1] || ahead(i) > 0.05 * size))
                .max_by(|&a, &b| limbs[a].base()[1].partial_cmp(&limbs[b].base()[1]).unwrap())
        } else {
            centre.iter().copied().filter(|&i| ahead(i) > 0.05 * size).max_by(|&a, &b| (fwd * limbs[a].tip()[2]).partial_cmp(&(fwd * limbs[b].tip()[2])).unwrap())
        };
        if let Some(h) = head {
            limbs[h].role = Role::Head;
        }
        let tail = centre
            .iter()
            .copied()
            .filter(|&i| Some(i) != head && ahead(i) < -0.05 * size)
            .min_by(|&a, &b| (fwd * limbs[a].tip()[2]).partial_cmp(&(fwd * limbs[b].tip()[2])).unwrap());
        if let Some(tl) = tail {
            limbs[tl].role = Role::Tail;
        }
        // arms: a left and a right limb high on a body that stands up
        if upright {
            for l in limbs.iter_mut() {
                if l.role == Role::Extra && l.side != Side::Centre && l.at(l.emerge).0[1] > ground + 0.45 * height && l.length >= 0.2 * height {
                    l.role = Role::Arm;
                }
            }
        }
        // one arm a side: the one with the most body to it; rags and
        // tendrils hanging off the shoulders are extras
        for side in [Side::Left, Side::Right] {
            let vol = |l: &Limb| l.parts.iter().map(|&p| t.volume[p]).max().unwrap_or(0);
            let best = (0..limbs.len()).filter(|&i| limbs[i].role == Role::Arm && limbs[i].side == side).max_by_key(|&i| vol(&limbs[i]));
            for i in 0..limbs.len() {
                if limbs[i].role == Role::Arm && limbs[i].side == side && Some(i) != best {
                    limbs[i].role = Role::Extra;
                }
            }
        }
        // minor branches that got a chain only stay if nothing else claimed them
        limbs.retain(|l| l.role != Role::Extra || l.length >= EXTRA_LENGTH * size);

        let count = |r: Role| limbs.iter().filter(|l| l.role == r).count();
        let kind = if count(Role::Wing) >= 2 {
            Kind::Winged
        } else {
            match count(Role::Leg) {
                0 | 1 => Kind::Legless,
                2 | 3 => Kind::TwoLegged,
                4 => Kind::FourLegged,
                _ => Kind::ManyLimbed,
            }
        };
        // names
        let mut seen: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for l in limbs.iter_mut() {
            let stem = match l.role {
                Role::Leg if kind == Kind::FourLegged => {
                    if l.front {
                        "leg_front"
                    } else {
                        "leg_hind"
                    }
                }
                Role::Leg => "leg",
                Role::Arm => "arm",
                Role::Head => "neck",
                Role::Tail => "tail",
                Role::Wing => "wing",
                Role::Extra => "extra",
            };
            let side = match l.side {
                Side::Left => ".L",
                Side::Right => ".R",
                Side::Centre => "",
            };
            let n = seen.entry(format!("{stem}{side}")).or_insert(0);
            *n += 1;
            l.name = if *n == 1 { format!("{stem}{side}") } else { format!("{stem}{}{side}", *n) };
        }
        let mut limb_of_part = vec![None; np];
        for (i, l) in limbs.iter().enumerate() {
            for &p in &l.parts {
                mark_below(body, p, i, &mut limb_of_part);
            }
        }
        let trunk: Vec<usize> = (0..np).filter(|&p| limb_of_part[p].is_none()).collect();
        let (spine, spine_radius) = spine_line(body, &limbs, &limb_of_part, upright, fwd);
        Anatomy { kind, upright, limbs, trunk, spine, spine_radius, ground, height, limb_of_part, forward: fwd }
    }
}

/// Arc length at which a limb's line has come out of the body it grows
/// from: where the body around the line is no more than a quarter thicker
/// than the limb is along its outer part. (The line starts in the middle of
/// the trunk; a leg's first joint belongs at the shoulder, not there.)
fn emerges(l: &Limb) -> f64 {
    let n = 40;
    let mut outer: Vec<f64> = (n * 2 / 5..=n * 9 / 10).map(|k| l.at(l.length * k as f64 / n as f64).1).collect();
    outer.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let typical = outer[outer.len() / 2];
    for k in 0..=n * 2 / 5 {
        let s = l.length * k as f64 / n as f64;
        if l.at(s).1 <= 1.25 * typical {
            return s;
        }
    }
    0.4 * l.length
}

fn mark_below(body: &Body, p: usize, limb: usize, out: &mut [Option<usize>]) {
    if out[p].is_some() {
        return;
    }
    out[p] = Some(limb);
    for &c in &body.parts[p].children {
        mark_below(body, c, limb, out);
    }
}

/// The parts of the limb that starts at `first`: on through single major
/// children, then out through the straightest digit.
fn chain_from(t: &Tree, first: usize) -> Vec<usize> {
    let body = t.body;
    let mut chain = vec![first];
    let mut cur = first;
    loop {
        let kids = &body.parts[cur].children;
        let majors: Vec<usize> = kids.iter().copied().filter(|&c| t.major(c)).collect();
        if majors.len() >= 2 {
            break;
        }
        if majors.len() == 1 {
            // two forks of majors in a row are the caller's to split
            cur = majors[0];
            chain.push(cur);
            continue;
        }
        // digits: carry the line on through the one that runs straightest
        let line = &body.parts[cur].line;
        if line.len() < 2 {
            break;
        }
        let dir = geom::sub(line[line.len() - 1], line[line.len().saturating_sub(4).min(line.len() - 2)]);
        let dl = geom::norm(dir);
        let best = kids
            .iter()
            .copied()
            .filter(|&c| body.parts[c].children.is_empty() && body.parts[c].length < DIGIT_LENGTH * body.size)
            .map(|c| {
                let d = geom::sub(body.parts[c].tip(), body.parts[c].base());
                (geom::dot(d, dir) / (geom::norm(d) * dl).max(1e-12), c)
            })
            .filter(|x| x.0 > 0.5)
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        if let Some((_, c)) = best {
            chain.push(c);
        }
        break;
    }
    chain
}

/// The spine through the trunk parts. With legs: from where the hind (or
/// the only) legs leave the body to where the neck does, else the front
/// legs or arms, else the far end of the trunk. Legless: from the lowest
/// (upright) or rearmost point of the trunk to its highest or foremost.
fn spine_line(body: &Body, limbs: &[Limb], limb_of_part: &[Option<usize>], upright: bool, fwd: f64) -> (Vec<V3>, Vec<f64>) {
    // attachment: the trunk part a limb grows from and the point where it does
    let attach = |l: &Limb| (body.parts[l.parts[0]].parent.unwrap_or(0), l.base());
    let legs: Vec<&Limb> = limbs.iter().filter(|l| l.role == Role::Leg).collect();
    let trunk_leaves: Vec<usize> = (0..body.parts.len()).filter(|&p| limb_of_part[p].is_none() && body.parts[p].children.iter().all(|&c| limb_of_part[c].is_some())).collect();
    let end_of = |p: usize| *body.parts[p].line.last().unwrap();
    let score_far = |p: usize, sign: f64| {
        let e = end_of(p);
        sign * if upright { e[1] } else { fwd * e[2] }
    };
    let low: (usize, V3) = if !legs.is_empty() {
        // the hind pair: the legs that leave the body lowest (upright) or rearmost
        let key = |l: &&Limb| if upright { l.base()[1] } else { fwd * l.base()[2] };
        let l = legs.iter().min_by(|a, b| key(a).partial_cmp(&key(b)).unwrap()).unwrap();
        attach(l)
    } else {
        let p = trunk_leaves.iter().copied().min_by(|&a, &b| score_far(a, 1.0).partial_cmp(&score_far(b, 1.0)).unwrap()).unwrap_or(0);
        (p, end_of(p))
    };
    let high: (usize, V3) = if let Some(h) = limbs.iter().find(|l| l.role == Role::Head) {
        attach(h)
    } else if let Some(a) = limbs.iter().filter(|l| l.role == Role::Arm || (l.role == Role::Leg && l.front)).next() {
        attach(a)
    } else {
        let p = trunk_leaves.iter().copied().filter(|&p| p != low.0 || trunk_leaves.len() == 1).max_by(|&a, &b| score_far(a, 1.0).partial_cmp(&score_far(b, 1.0)).unwrap()).unwrap_or(0);
        (p, end_of(p))
    };
    // path through the part tree: up from `low` to the common ancestor, down to `high`
    let up = |mut p: usize| {
        let mut v = vec![p];
        while let Some(q) = body.parts[p].parent {
            v.push(q);
            p = q;
        }
        v
    };
    let (a, b) = (up(low.0), up(high.0));
    let meet = a.iter().copied().find(|p| b.contains(p)).unwrap_or(0);
    let mut line: Vec<V3> = vec![];
    let mut radius: Vec<f64> = vec![];
    let push = |pts: Vec<(V3, f64)>, line: &mut Vec<V3>, radius: &mut Vec<f64>| {
        for (p, r) in pts {
            if line.last().is_none_or(|q| geom::norm(geom::sub(*q, p)) > 1e-9) {
                line.push(p);
                radius.push(r);
            }
        }
    };
    let pts = |p: usize| -> Vec<(V3, f64)> { body.parts[p].line.iter().copied().zip(body.parts[p].radius.iter().copied()).collect() };
    // going up toward the seed: each part's line reversed
    for &p in a.iter().take_while(|&&p| p != meet) {
        let mut v = pts(p);
        v.reverse();
        push(v, &mut line, &mut radius);
    }
    if low.0 == meet && high.0 == meet {
        push(pts(meet), &mut line, &mut radius);
        // one part: order it low to high
        let key = |p: V3| if upright { p[1] } else { fwd * p[2] };
        if line.len() >= 2 && key(line[0]) > key(*line.last().unwrap()) {
            line.reverse();
            radius.reverse();
        }
    } else if low.0 == meet {
        // the low end sits in an ancestor of the high end: start where the
        // low limbs leave it, which is that part's end
        push(vec![(end_of(meet), *body.parts[meet].radius.last().unwrap())], &mut line, &mut radius);
    } else if high.0 == meet {
        push(vec![(end_of(meet), *body.parts[meet].radius.last().unwrap())], &mut line, &mut radius);
    }
    let down: Vec<usize> = b.iter().copied().take_while(|&p| p != meet).collect();
    for &p in down.iter().rev() {
        push(pts(p), &mut line, &mut radius);
    }
    if line.len() < 2 {
        // a body that is all one piece: the seed to the high point
        let s = body.solid.center(body.seed);
        line = vec![s, high.1];
        radius = vec![body.solid.depth[body.seed] as f64 * body.solid.cell; 2];
        if geom::norm(geom::sub(line[1], line[0])) < 1e-6 {
            line[1] = geom::add(line[0], if upright { [0.0, 0.1 * body.size, 0.0] } else { [0.0, 0.0, fwd * 0.1 * body.size] });
        }
    }
    (line, radius)
}
