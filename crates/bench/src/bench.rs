//! The measurements: a route's rig against the artist's rig of the same
//! mesh. Everything is done in the route's space; lengths are reported
//! as shares of the body's length (the longest side of the rest pose).

use serde_json::{json, Value};
use skintokens::geom::{self, KdTree, M4, V3};

use crate::rig::{m3_apply, m3_mul, m3_t, rotation_of, Rig, M3};

// ------------------------------------------------------------- small maths

fn bbox(pts: &[V3]) -> (V3, V3) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in pts {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

fn longest(pts: &[V3]) -> f64 {
    let (lo, hi) = bbox(pts);
    (0..3).map(|a| hi[a] - lo[a]).fold(0.0, f64::max)
}

fn seg_dist(p: V3, a: V3, b: V3) -> f64 {
    let ab = geom::sub(b, a);
    let l2 = geom::dot(ab, ab);
    if l2 < 1e-18 {
        return geom::norm(geom::sub(p, a));
    }
    let u = (geom::dot(geom::sub(p, a), ab) / l2).clamp(0.0, 1.0);
    geom::norm(geom::sub(p, geom::add(a, geom::scale(ab, u))))
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let k = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[k]
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

fn r4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

/// A share of the body's length as a percentage with two decimals.
fn pct(x: f64) -> f64 {
    (x * 1e4).round() / 1e2
}

// ---------------------------------------------------------------- aligning

/// How the artist's rest mesh sits in the route's file: a quarter turn
/// about the up axis, a uniform scale and a shift. A route may turn a
/// model to face forward or put it on the ground; it does not otherwise
/// move it.
pub struct Align {
    pub m: M4,
    pub yaw_deg: f64,
    pub scale: f64,
    /// Mean and largest distance from an artist's vertex to the nearest
    /// vertex of the route's mesh, in body lengths.
    pub fit: f64,
    pub fit_max: f64,
}

pub fn align(truth_rest: &[V3], route_rest: &[V3]) -> Align {
    let tree = KdTree::new(route_rest.to_vec());
    let len_r = longest(route_rest);
    let (rlo, rhi) = bbox(route_rest);
    let rc = geom::scale(geom::add(rlo, rhi), 0.5);
    let mut best: Option<Align> = None;
    for fitted in [false, true] {
        for quarter in 0..4 {
            let th = quarter as f64 * std::f64::consts::FRAC_PI_2;
            let (s, c) = th.sin_cos();
            let rot: M4 = [
                [c, 0.0, -s, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [s, 0.0, c, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ];
            let mut m = rot;
            let mut scale = 1.0;
            if fitted {
                let turned: Vec<V3> = truth_rest.iter().map(|p| geom::apply(&rot, *p)).collect();
                scale = len_r / longest(&turned).max(1e-12);
                let (lo, hi) = bbox(&turned);
                let tc = geom::scale(geom::add(lo, hi), 0.5 * scale);
                let sm: M4 = [
                    [scale, 0.0, 0.0, 0.0],
                    [0.0, scale, 0.0, 0.0],
                    [0.0, 0.0, scale, 0.0],
                    [rc[0] - tc[0], rc[1] - tc[1], rc[2] - tc[2], 1.0],
                ];
                m = geom::mul(&sm, &rot);
            }
            let (mut sum, mut worst) = (0.0, 0.0f64);
            for p in truth_rest {
                let d = tree.knn(geom::apply(&m, *p), 1)[0].0;
                sum += d;
                worst = worst.max(d);
            }
            let fit = sum / truth_rest.len() as f64 / len_r;
            let cand = Align {
                m,
                yaw_deg: quarter as f64 * 90.0,
                scale,
                fit,
                fit_max: worst / len_r,
            };
            if best
                .as_ref()
                .map(|b| cand.fit < b.fit - 1e-9)
                .unwrap_or(true)
            {
                best = Some(cand);
            }
        }
        // as it is, or a plain quarter turn, already fits: no need to guess a scale
        if best.as_ref().unwrap().fit < 1e-3 {
            break;
        }
    }
    let mut best = best.unwrap();
    // A route may also turn a model by a few degrees to face forward as
    // it measures "forward": refine about the up axis on nearest pairs.
    for _ in 0..30 {
        if best.fit < 1e-6 {
            break;
        }
        let moved: Vec<V3> = truth_rest
            .iter()
            .map(|p| geom::apply(&best.m, *p))
            .collect();
        let near: Vec<V3> = moved
            .iter()
            .map(|p| route_rest[tree.knn(*p, 1)[0].1])
            .collect();
        let n = moved.len() as f64;
        let ct = geom::scale(
            moved.iter().fold([0.0; 3], |a, p| geom::add(a, *p)),
            1.0 / n,
        );
        let cr = geom::scale(near.iter().fold([0.0; 3], |a, p| geom::add(a, *p)), 1.0 / n);
        let (mut sn, mut cs, mut aa) = (0.0, 0.0, 0.0);
        for (p, q) in moved.iter().zip(&near) {
            let (a, b) = (geom::sub(*p, ct), geom::sub(*q, cr));
            sn += a[2] * b[0] - a[0] * b[2];
            cs += a[0] * b[0] + a[2] * b[2];
            aa += geom::dot(a, a);
        }
        let th = sn.atan2(cs);
        let (s, c) = th.sin_cos();
        let rot: M4 = [
            [c, 0.0, -s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let mut ab = 0.0;
        for (p, q) in moved.iter().zip(&near) {
            ab += geom::dot(geom::apply_dir(&rot, geom::sub(*p, ct)), geom::sub(*q, cr));
        }
        let k = if aa > 1e-18 {
            (ab / aa).clamp(0.5, 2.0)
        } else {
            1.0
        };
        let rc = geom::scale(geom::apply_dir(&rot, ct), k);
        let step: M4 = [
            [k * rot[0][0], 0.0, k * rot[0][2], 0.0],
            [0.0, k, 0.0, 0.0],
            [k * rot[2][0], 0.0, k * rot[2][2], 0.0],
            [cr[0] - rc[0], cr[1] - rc[1], cr[2] - rc[2], 1.0],
        ];
        let m = geom::mul(&step, &best.m);
        let (mut sum, mut worst) = (0.0, 0.0f64);
        for p in truth_rest {
            let d = tree.knn(geom::apply(&m, *p), 1)[0].0;
            sum += d;
            worst = worst.max(d);
        }
        let fit = sum / n / len_r;
        if fit >= best.fit - 1e-9 {
            break;
        }
        best = Align {
            m,
            yaw_deg: best.yaw_deg + th.to_degrees(),
            scale: best.scale * k,
            fit,
            fit_max: worst / len_r,
        };
    }
    best
}

// ------------------------------------------------------------------- roles

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Role {
    Trunk,
    Neck,
    Head,
    Jaw,
    Leg,
    Toe,
    Tail,
    Wing,
    Other,
}

impl Role {
    pub fn name(self) -> &'static str {
        match self {
            Role::Trunk => "trunk",
            Role::Neck => "neck",
            Role::Head => "head",
            Role::Jaw => "jaw",
            Role::Leg => "leg",
            Role::Toe => "toe",
            Role::Tail => "tail",
            Role::Wing => "wing",
            Role::Other => "other",
        }
    }
    pub fn parse(s: &str) -> Option<Role> {
        [
            Role::Trunk,
            Role::Neck,
            Role::Head,
            Role::Jaw,
            Role::Leg,
            Role::Toe,
            Role::Tail,
            Role::Wing,
            Role::Other,
        ]
        .into_iter()
        .find(|r| r.name() == s)
    }
    /// The joints a rig is judged on. Toes, ears and the like are listed
    /// when a route lacks them, but their distance is not in the mean.
    /// The trunk's joints are reported apart: where an artist puts a
    /// body or hips pivot is a habit, not something a rig can get wrong.
    pub fn main(self) -> bool {
        !matches!(self, Role::Toe | Role::Other)
    }
}

/// What a hand-made bone is, read from its name.
pub fn role_of(name: &str) -> Role {
    let n = name.to_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| n.contains(w));
    if has(&["jaw", "mouth", "chin", "beak", "mandible"]) {
        Role::Jaw
    } else if has(&["tail"]) {
        Role::Tail
    } else if has(&["wing"]) {
        Role::Wing
    } else if has(&["toe", "finger", "claw", "digit", "thumb"]) {
        Role::Toe
    } else if has(&["neck"]) {
        Role::Neck
    } else if has(&["head", "face", "nose", "skull", "snout"]) {
        Role::Head
    } else if has(&[
        "leg", "arm", "foot", "feet", "hand", "thigh", "shin", "calf", "paw", "hoof", "knee",
        "elbow", "ankle",
    ]) {
        Role::Leg
    } else if has(&["shoulders"]) {
        Role::Trunk
    } else if has(&["shoulder", "clavicle", "scapula"]) {
        Role::Leg
    } else if has(&[
        "ear", "eye", "tongue", "antenna", "horn", "hair", "whisker", "pole", "target", "sting",
    ]) {
        Role::Other
    } else if has(&[
        "body", "hip", "spine", "back", "torso", "chest", "pelvis", "abdomen", "thorax", "root",
        "belly", "waist",
    ]) {
        Role::Trunk
    } else {
        Role::Other
    }
}

// --------------------------------------------------------------- the truth

/// The artist's rig in the route's space.
pub struct Truth<'a> {
    pub rig: &'a Rig,
    pub a: M4,
    /// Body length in the route's space.
    pub len: f64,
    pub rest_pos: Vec<V3>,
    pub rest_inv: Vec<M4>,
    pub rest_rot: Vec<M3>,
    pub joint_pos: Vec<V3>,
    pub roles: Vec<Role>,
    /// Bones that hold skin.
    pub skin_bone: Vec<bool>,
    /// Bones a route's bone may follow: those that hold skin and the
    /// ones above them (a body or pelvis bone that only carries others).
    /// Not handles that hold nothing and carry nothing (IK targets).
    pub leads: Vec<bool>,
    /// The lines each skin bone lies along (to each child, or for a last
    /// bone to the middle of the skin it holds).
    pub lines: Vec<Vec<(V3, V3)>>,
}

fn bone_lines(rig: &Rig, pos: &[V3], verts: &[V3], len: f64) -> Vec<Vec<(V3, V3)>> {
    let mut held: Vec<(V3, f64)> = vec![([0.0; 3], 0.0); pos.len()];
    for (v, &d) in rig.dominant.iter().enumerate() {
        if d != usize::MAX {
            held[d].0 = geom::add(held[d].0, verts[v]);
            held[d].1 += 1.0;
        }
    }
    (0..pos.len())
        .map(|j| {
            let mut out: Vec<(V3, V3)> = rig.children[j]
                .iter()
                .map(|&c| (pos[j], pos[c]))
                .filter(|(a, b)| geom::norm(geom::sub(*b, *a)) > 1e-4 * len)
                .collect();
            if out.is_empty() {
                if held[j].1 > 0.0 {
                    let c = geom::scale(held[j].0, 1.0 / held[j].1);
                    if geom::norm(geom::sub(c, pos[j])) > 1e-4 * len {
                        out.push((pos[j], c));
                    }
                }
            }
            if out.is_empty() {
                if let Some(p) = rig.parent[j] {
                    let d = geom::sub(pos[j], pos[p]);
                    if geom::norm(d) > 1e-4 * len {
                        out.push((pos[j], geom::add(pos[j], geom::scale(d, 0.5))));
                    }
                }
            }
            if out.is_empty() {
                out.push((pos[j], pos[j]));
            }
            out
        })
        .collect()
}

impl<'a> Truth<'a> {
    pub fn new(rig: &'a Rig, al: &Align, roles_given: &Value) -> Truth<'a> {
        let rest_pos: Vec<V3> = rig
            .rest_pos
            .iter()
            .map(|p| geom::apply(&al.m, *p))
            .collect();
        let rest_world: Vec<M4> = rig
            .joints
            .iter()
            .map(|&n| geom::mul(&al.m, &rig.rest_world[n]))
            .collect();
        let joint_pos: Vec<V3> = rest_world.iter().map(geom::translation).collect();
        let len = longest(&rest_pos);
        let mut roles: Vec<Role> = rig.names.iter().map(|n| role_of(n)).collect();
        // a shoulder that carries a wing is the wing's first bone
        for j in 0..roles.len() {
            if roles[j] == Role::Leg
                && rig.names[j].to_lowercase().contains("shoulder")
                && rig.children[j].iter().any(|&c| roles[c] == Role::Wing)
            {
                roles[j] = Role::Wing;
            }
        }
        for (j, name) in rig.names.iter().enumerate() {
            if let Some(r) = roles_given[name].as_str().and_then(Role::parse) {
                roles[j] = r;
            }
        }
        let lines = bone_lines(rig, &joint_pos, &rest_pos, len);
        let skin_bone: Vec<bool> = rig.mass.iter().map(|&m| m > 1e-6).collect();
        let mut leads = skin_bone.clone();
        for &j in rig.joint_order.iter().rev() {
            if leads[j] {
                if let Some(p) = rig.parent[j] {
                    leads[p] = true;
                }
            }
        }
        Truth {
            rig,
            a: al.m,
            len,
            rest_inv: rest_world.iter().map(geom::inverse).collect(),
            rest_rot: rest_world.iter().map(rotation_of).collect(),
            skin_bone,
            leads,
            rest_pos,
            joint_pos,
            roles,
            lines,
        }
    }

    /// Joint matrices and the skin at a time of a clip, in the route's space.
    pub fn frame(&self, clip: usize, t: f64) -> (Vec<M4>, Vec<V3>) {
        let nodes = self.rig.pose_nodes(&self.rig.clips[clip], t);
        let skin: Vec<V3> = self
            .rig
            .skin_nodes(&nodes)
            .iter()
            .map(|p| geom::apply(&self.a, *p))
            .collect();
        let jw = self
            .rig
            .joints
            .iter()
            .map(|&n| geom::mul(&self.a, &nodes[n]))
            .collect();
        (jw, skin)
    }
}

// ----------------------------------------------------------------- mapping

pub struct Mapping {
    /// The artist's bone each of the route's bones follows, by where it lies.
    pub by_line: Vec<usize>,
    /// How far the route's bone lies from that bone's line (body lengths).
    pub off: Vec<f64>,
    /// The same choice made from the skin each bone holds (a cross-check).
    pub by_skin: Vec<usize>,
}

/// Beyond this far from every bone of the artist's (in body lengths), a
/// route's bone lies along nothing the artist rigged.
pub const ALONG_NOTHING: f64 = 0.12;

pub fn map_bones(truth: &Truth, route: &Rig, r2t: &[usize]) -> Mapping {
    let pos = route.joint_rest();
    let lines = bone_lines(route, &pos, &route.rest_pos, truth.len);
    let n = route.joints.len();
    let mut by_line = vec![0; n];
    let mut off = vec![0.0; n];
    for j in 0..n {
        let mut best = (f64::INFINITY, 0usize, 0.0);
        for a in 0..truth.rig.joints.len() {
            if !truth.leads[a] {
                continue;
            }
            let (mut cost, mut dist) = (0.0, 0.0);
            for &(p, q) in &lines[j] {
                let dir = geom::sub(q, p);
                let dl = geom::norm(dir);
                let mut seg_best = (f64::INFINITY, 0.0);
                for &(la, lb) in &truth.lines[a] {
                    let d = [0.25, 0.5, 0.75]
                        .iter()
                        .map(|u| seg_dist(geom::add(p, geom::scale(dir, *u)), la, lb))
                        .sum::<f64>()
                        / 3.0
                        / truth.len;
                    let ld = geom::sub(lb, la);
                    let ll = geom::norm(ld);
                    let turn = if dl > 1e-9 && ll > 1e-9 {
                        1.0 - geom::dot(dir, ld) / (dl * ll)
                    } else {
                        0.0
                    };
                    let c = d + 0.15 * turn;
                    if c < seg_best.0 {
                        seg_best = (c, d);
                    }
                }
                cost += seg_best.0;
                dist += seg_best.1;
            }
            cost /= lines[j].len() as f64;
            dist /= lines[j].len() as f64;
            if cost < best.0 {
                best = (cost, a, dist);
            }
        }
        by_line[j] = best.1;
        off[j] = best.2;
    }
    // by skin: the artist's bone that holds most of what this bone holds
    let na = truth.rig.joints.len();
    let mut share = vec![vec![0.0f64; na]; n];
    for (r, list) in route.inf.iter().enumerate() {
        let t = r2t[r];
        for x in list {
            for y in &truth.rig.inf[t] {
                share[x.joint][y.joint] += x.w * y.w;
            }
        }
    }
    let mut by_skin = by_line.clone();
    for &j in &route.joint_order {
        let (mut best, mut arg) = (0.0, usize::MAX);
        for a in 0..na {
            if share[j][a] > best {
                best = share[j][a];
                arg = a;
            }
        }
        by_skin[j] = if arg != usize::MAX && best > 1e-6 {
            arg
        } else {
            match route.parent[j] {
                Some(p) => by_skin[p],
                None => by_line[j],
            }
        };
    }
    Mapping {
        by_line,
        off,
        by_skin,
    }
}

// ------------------------------------------------------------ the skeleton

pub fn skeleton(truth: &Truth, route: &Rig, map: &Mapping) -> Value {
    let rpos = route.joint_rest();
    let na = truth.rig.joints.len();
    let mut followers: Vec<Vec<usize>> = vec![vec![]; na];
    for (j, &a) in map.by_line.iter().enumerate() {
        if map.off[j] <= ALONG_NOTHING {
            followers[a].push(j);
        }
    }
    let near = |a: usize, among: &[usize]| -> Option<(f64, usize)> {
        among
            .iter()
            .map(|&j| {
                (
                    geom::norm(geom::sub(rpos[j], truth.joint_pos[a])) / truth.len,
                    j,
                )
            })
            .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
    };
    let mut joints = vec![];
    let (mut errs, mut trunk, mut unmatched) = (vec![], vec![], 0);
    for a in 0..na {
        if !truth.skin_bone[a] {
            continue;
        }
        // a joint of the route's at this joint: one whose bone lies along
        // this bone, the one above it or one below it
        let mut cand: Vec<usize> = followers[a].clone();
        if let Some(p) = truth.rig.parent[a] {
            cand.extend(&followers[p]);
        }
        for &c in &truth.rig.children[a] {
            cand.extend(&followers[c]);
        }
        let mut found = near(a, &cand);
        let has = found.is_some();
        if !has {
            // none: the rig stops short of this joint. Its error is the
            // distance back to where the rig does have a bone.
            let mut up = truth.rig.parent[a];
            while let Some(p) = up {
                if !followers[p].is_empty() {
                    found = near(a, &followers[p]);
                    break;
                }
                up = truth.rig.parent[p];
            }
            if found.is_none() {
                found = near(a, &(0..rpos.len()).collect::<Vec<_>>());
            }
        }
        let (e, j) = found.unwrap_or((1.0, 0));
        let role = truth.roles[a];
        if role == Role::Trunk {
            trunk.push(e);
        } else if role.main() {
            errs.push(e);
            if !has {
                unmatched += 1;
            }
        }
        joints.push(
            json!({"joint": truth.rig.names[a], "role": role.name(), "error_pct": pct(e),
            "nearest": route.names.get(j), "has_a_joint": has}),
        );
    }
    // chains: bones of one role joined through the hierarchy (a bone
    // that holds no skin but carries the next one stays in its chain)
    let mut chain_of = vec![usize::MAX; na];
    let mut chains: Vec<Vec<usize>> = vec![];
    for &a in &truth.rig.joint_order {
        if !truth.leads[a] {
            continue;
        }
        let joined = truth.rig.parent[a]
            .filter(|&p| truth.roles[p] == truth.roles[a] && chain_of[p] != usize::MAX);
        match joined {
            Some(p) => {
                chain_of[a] = chain_of[p];
                chains[chain_of[p]].push(a);
            }
            None => {
                chain_of[a] = chains.len();
                chains.push(vec![a]);
            }
        }
    }
    let mut chain_json = vec![];
    let mut missing = vec![];
    let mut short = vec![];
    for c in &chains {
        if !c.iter().any(|&a| truth.skin_bone[a]) {
            continue;
        }
        let role = truth.roles[c[0]];
        let covered = c.iter().filter(|&&a| !followers[a].is_empty()).count();
        let label = if c.len() == 1 {
            format!("{} ({})", role.name(), truth.rig.names[c[0]])
        } else {
            format!(
                "{} ({} to {}, {} bones)",
                role.name(),
                truth.rig.names[c[0]],
                truth.rig.names[*c.last().unwrap()],
                c.len()
            )
        };
        chain_json.push(json!({"chain": label, "role": role.name(), "bones": c.len(), "with_a_bone_along": covered}));
        if covered == 0 {
            missing.push(label);
        } else if covered < c.len() {
            short.push(format!("{label}: {covered} of {}", c.len()));
        }
    }
    // the route's bones that lie along nothing of the artist's
    let is_extra = |j: usize| map.off[j] > ALONG_NOTHING && route.mass[j] > 1e-6;
    let extra: Vec<usize> = (0..rpos.len()).filter(|&j| is_extra(j)).collect();
    let extra_chains = extra
        .iter()
        .filter(|&&j| route.parent[j].map(|p| !is_extra(p)).unwrap_or(true))
        .count();
    let mut sorted = errs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    json!({
        "joint_error_pct": pct(mean(&errs)),
        "joint_error_max_pct": pct(sorted.last().copied().unwrap_or(0.0)),
        "limb_joints": errs.len(),
        "limb_joints_without_a_joint": unmatched,
        "trunk_joint_error_pct": pct(mean(&trunk)),
        "bones": {"artist": truth.skin_bone.iter().filter(|x| **x).count(), "route": route.joints.len()},
        "missing_chains": missing,
        "short_chains": short,
        "extra_chains": extra_chains,
        "extra_bones": extra.iter().map(|&j| route.names[j].clone()).collect::<Vec<_>>(),
        "chains": chain_json,
        "joints": joints,
    })
}

// ----------------------------------------------------------------- driving

#[derive(Clone, Copy, PartialEq)]
pub enum Drive {
    /// Each bone turns as the artist's bone it follows turns; it stays on
    /// its own parent, so a joint in the wrong place shows.
    Turn,
    /// Each bone moves bodily with the artist's bone it follows: where
    /// its joints are no longer matters, only its weights.
    Carry,
}

/// A world matrix per joint of the route for one frame of the artist's.
pub fn drive(truth: &Truth, route: &Rig, follow: &[usize], now: &[M4], how: Drive) -> Vec<M4> {
    let n = route.joints.len();
    let mut out = vec![geom::IDENTITY; n];
    let mut turn: Vec<M3> = vec![[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]; n];
    for &j in &route.joint_order {
        let a = follow[j];
        let rest = route.rest_world[route.joints[j]];
        let delta = geom::mul(&now[a], &truth.rest_inv[a]);
        if how == Drive::Carry {
            out[j] = geom::mul(&delta, &rest);
            continue;
        }
        let rd = m3_mul(&rotation_of(&now[a]), &m3_t(&truth.rest_rot[a]));
        turn[j] = rd;
        let lin = m3_mul(&rd, &geom::cols(&rest));
        let p = geom::translation(&rest);
        let at = match route.parent[j] {
            None => geom::apply(&delta, p),
            Some(pj) => {
                let pr = geom::translation(&route.rest_world[route.joints[pj]]);
                let mut at = geom::add(
                    geom::translation(&out[pj]),
                    m3_apply(&turn[pj], geom::sub(p, pr)),
                );
                // Where the artist slides this bone along its parent (a
                // body bobbing on its root, a foot on its handle), and
                // the route's bone hangs from the bone that follows that
                // parent, it slides the same way.
                let ap = follow[pj];
                if truth.rig.parent[a] == Some(ap) {
                    let rest_off = geom::sub(truth.joint_pos[a], truth.joint_pos[ap]);
                    let now_off =
                        geom::sub(geom::translation(&now[a]), geom::translation(&now[ap]));
                    at = geom::add(at, geom::sub(now_off, m3_apply(&turn[pj], rest_off)));
                }
                at
            }
        };
        out[j] = geom::from_rt(lin, at);
    }
    out
}

/// The rotation that best carries the vectors `a` onto `b` (weighted
/// least squares; Horn's quaternion method).
fn best_rotation(pairs: &[(V3, V3, f64)]) -> M3 {
    let mut s = [[0.0f64; 3]; 3];
    for (a, b, w) in pairs {
        for r in 0..3 {
            for c in 0..3 {
                s[r][c] += w * a[r] * b[c];
            }
        }
    }
    let mut n = [
        [
            s[0][0] + s[1][1] + s[2][2],
            s[1][2] - s[2][1],
            s[2][0] - s[0][2],
            s[0][1] - s[1][0],
        ],
        [
            s[1][2] - s[2][1],
            s[0][0] - s[1][1] - s[2][2],
            s[0][1] + s[1][0],
            s[2][0] + s[0][2],
        ],
        [
            s[2][0] - s[0][2],
            s[0][1] + s[1][0],
            -s[0][0] + s[1][1] - s[2][2],
            s[1][2] + s[2][1],
        ],
        [
            s[0][1] - s[1][0],
            s[2][0] + s[0][2],
            s[1][2] + s[2][1],
            -s[0][0] - s[1][1] + s[2][2],
        ],
    ];
    // Jacobi: the eigenvector of the largest eigenvalue is the quaternion (w, x, y, z)
    let mut v = [[0.0f64; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _ in 0..60 {
        let mut off = 0.0;
        for i in 0..4 {
            for j in (i + 1)..4 {
                off += n[i][j] * n[i][j];
            }
        }
        if off < 1e-24 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                if n[p][q].abs() < 1e-300 {
                    continue;
                }
                let th = 0.5 * (2.0 * n[p][q]).atan2(n[q][q] - n[p][p]);
                let (sn, cs) = th.sin_cos();
                for k in 0..4 {
                    let (a, b) = (n[k][p], n[k][q]);
                    n[k][p] = cs * a - sn * b;
                    n[k][q] = sn * a + cs * b;
                }
                for k in 0..4 {
                    let (a, b) = (n[p][k], n[q][k]);
                    n[p][k] = cs * a - sn * b;
                    n[q][k] = sn * a + cs * b;
                }
                for row in v.iter_mut() {
                    let (a, b) = (row[p], row[q]);
                    row[p] = cs * a - sn * b;
                    row[q] = sn * a + cs * b;
                }
            }
        }
    }
    let top = (0..4)
        .max_by(|&a, &b| n[a][a].partial_cmp(&n[b][b]).unwrap())
        .unwrap();
    let (w, x, y, z) = (v[0][top], v[1][top], v[2][top], v[3][top]);
    let l = (w * w + x * x + y * y + z * z).sqrt().max(1e-12);
    let (w, x, y, z) = (w / l, x / l, y / l, z / l);
    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ],
        [
            2.0 * (x * y - z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z + x * w),
        ],
        [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

/// What the fit needs of a route's rig, gathered once.
pub struct Fitter {
    /// Per joint: the vertices held by it or by a joint below it.
    under: Vec<Vec<usize>>,
    /// Depth-first numbering: joint `k` is `j` or below it when
    /// `tin[j] <= tin[k] && tin[k] < tout[j]`.
    tin: Vec<usize>,
    tout: Vec<usize>,
    pos: Vec<V3>,
    sweeps: usize,
}

impl Fitter {
    pub fn new(route: &Rig, sweeps: usize) -> Fitter {
        let n = route.joints.len();
        let (mut tin, mut tout) = (vec![0; n], vec![0; n]);
        let mut clock = 0;
        let mut stack: Vec<(usize, bool)> = (0..n)
            .filter(|&j| route.parent[j].is_none())
            .map(|j| (j, false))
            .collect();
        while let Some((j, done)) = stack.pop() {
            if done {
                tout[j] = clock;
                continue;
            }
            tin[j] = clock;
            clock += 1;
            stack.push((j, true));
            for &c in &route.children[j] {
                stack.push((c, false));
            }
        }
        let mut under = vec![vec![]; n];
        for (v, list) in route.inf.iter().enumerate() {
            let mut seen: Vec<usize> = vec![];
            for x in list {
                let mut k = Some(x.joint);
                while let Some(j) = k {
                    if !seen.contains(&j) {
                        seen.push(j);
                        under[j].push(v);
                    }
                    k = route.parent[j];
                }
            }
        }
        Fitter {
            under,
            tin,
            tout,
            pos: route.joint_rest(),
            sweeps,
        }
    }
}

/// The route's rig doing its best to follow one frame of the artist's.
/// It starts from `start` (a world matrix per joint: the artist's
/// rotations carried across). Then, a few times over, each joint in
/// turn is rotated about its own place, everything below it turning
/// with it, to the rotation that leaves the whole skin nearest to
/// `target` (the artist's skin for each of the route's vertices); a
/// joint with no parent may also move. A step is kept only when the
/// skin gets nearer, so the result is never farther than the start.
pub fn fit(route: &Rig, fz: &Fitter, start: &[M4], target: &[V3]) -> Vec<M4> {
    let n = route.joints.len();
    // state: each joint's turn from rest, and where it is
    let rest_rot: Vec<M3> = (0..n)
        .map(|j| rotation_of(&route.rest_world[route.joints[j]]))
        .collect();
    let mut rot: Vec<M3> = (0..n)
        .map(|j| m3_mul(&rotation_of(&start[j]), &m3_t(&rest_rot[j])))
        .collect();
    let mut at: Vec<V3> = start.iter().map(geom::translation).collect();
    let place = |rot: &[M3], at: &[V3], k: usize, x: V3| {
        geom::add(at[k], m3_apply(&rot[k], geom::sub(x, fz.pos[k])))
    };
    for _ in 0..fz.sweeps {
        let mut moved = 0.0f64;
        for &j in &route.joint_order {
            let verts = &fz.under[j];
            if verts.is_empty() {
                continue;
            }
            let rj_t = m3_t(&rot[j]);
            let below = |k: usize| fz.tin[j] <= fz.tin[k] && fz.tin[k] < fz.tout[j];
            // per vertex: what this joint's turn moves (in the joint's
            // frame), and where that part has to land
            let mut a: Vec<V3> = Vec::with_capacity(verts.len());
            let mut b: Vec<V3> = Vec::with_capacity(verts.len());
            let mut m: Vec<f64> = Vec::with_capacity(verts.len());
            for &v in verts {
                let x = route.rest_pos[v];
                let (mut av, mut rest, mut mv) = ([0.0; 3], target[v], 0.0);
                for i in &route.inf[v] {
                    let p = place(&rot, &at, i.joint, x);
                    if below(i.joint) {
                        av = geom::add(av, geom::scale(m3_apply(&rj_t, geom::sub(p, at[j])), i.w));
                        mv += i.w;
                    } else {
                        rest = geom::sub(rest, geom::scale(p, i.w));
                    }
                }
                a.push(av);
                b.push(geom::sub(rest, geom::scale(at[j], mv)));
                m.push(mv);
            }
            let err = |r: &M3, c: V3| -> f64 {
                (0..a.len())
                    .map(|i| geom::dist2(geom::add(m3_apply(r, a[i]), geom::scale(c, m[i])), b[i]))
                    .sum()
            };
            let before = err(&rot[j], [0.0; 3]);
            let spread = (a.iter().map(|x| geom::dot(*x, *x)).sum::<f64>() / a.len() as f64).sqrt();
            let mut shift = [0.0; 3];
            let mut r_new = rot[j];
            for _ in 0..(if route.parent[j].is_none() { 3 } else { 1 }) {
                let mut pairs: Vec<(V3, V3, f64)> = (0..a.len())
                    .map(|i| (a[i], geom::sub(b[i], geom::scale(shift, m[i])), 1.0))
                    .collect();
                // a thin limb's spin about its own length is not told by
                // its skin: hold it lightly where it is
                for k in 0..3 {
                    let e = geom::scale(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]][k],
                        spread,
                    );
                    pairs.push((e, m3_apply(&rot[j], e), 0.01 * a.len() as f64));
                }
                r_new = best_rotation(&pairs);
                if route.parent[j].is_none() {
                    let (mut num, mut den) = ([0.0; 3], 0.0);
                    for i in 0..a.len() {
                        num = geom::add(
                            num,
                            geom::scale(geom::sub(b[i], m3_apply(&r_new, a[i])), m[i]),
                        );
                        den += m[i] * m[i];
                    }
                    if den > 1e-12 {
                        shift = geom::scale(num, 1.0 / den);
                    }
                }
            }
            let after = err(&r_new, shift);
            if after >= before - 1e-15 {
                continue;
            }
            moved += before - after;
            // turn everything below about this joint, then move it
            let step = m3_mul(&r_new, &rj_t);
            let pivot = at[j];
            for k in 0..n {
                if below(k) {
                    rot[k] = m3_mul(&step, &rot[k]);
                    at[k] = geom::add(
                        geom::add(pivot, m3_apply(&step, geom::sub(at[k], pivot))),
                        shift,
                    );
                }
            }
        }
        if moved < 1e-12 {
            break;
        }
    }
    (0..n)
        .map(|j| {
            geom::from_rt(
                m3_mul(&rot[j], &geom::cols(&route.rest_world[route.joints[j]])),
                at[j],
            )
        })
        .collect()
}

// ------------------------------------------------------------ skin and feet

pub struct Frames {
    pub clip: Vec<usize>,
    pub time: Vec<f64>,
    pub truth_joints: Vec<Vec<M4>>,
    pub truth_skin: Vec<Vec<V3>>,
}

pub fn frames(truth: &Truth, max: usize) -> Frames {
    let mut f = Frames {
        clip: vec![],
        time: vec![],
        truth_joints: vec![],
        truth_skin: vec![],
    };
    for (ci, clip) in truth.rig.clips.iter().enumerate() {
        for t in Rig::frame_times(clip, max) {
            let (jw, skin) = truth.frame(ci, t);
            f.clip.push(ci);
            f.time.push(t);
            f.truth_joints.push(jw);
            f.truth_skin.push(skin);
        }
    }
    f
}

pub struct SkinRun {
    pub json: Value,
    /// The route's skin per frame (its own vertex order).
    pub skins: Vec<Vec<V3>>,
    pub joints: Vec<Vec<M4>>,
}

/// How far the route's skin lands from the artist's on every frame.
/// `t2r`: for each of the artist's vertices, the same vertex of the
/// route's mesh.
pub fn skin_error(
    truth: &Truth,
    route: &Rig,
    fr: &Frames,
    t2r: &[usize],
    pose: impl Fn(usize) -> Vec<M4>,
) -> SkinRun {
    let nf = fr.time.len();
    let nv = truth.rest_pos.len();
    let mut skins = Vec::with_capacity(nf);
    let mut joints = Vec::with_capacity(nf);
    let mut per_frame: Vec<(f64, f64, f64)> = vec![];
    let nb = truth.rig.joints.len();
    let mut region_sum = vec![0.0; nb];
    let mut region_n = vec![0.0; nb];
    let mut worst = (0usize, -1.0);
    let mut worst_regions: Vec<(f64, f64)> = vec![];
    for f in 0..nf {
        let jw = pose(f);
        let skin = route.skin(&jw);
        let mut e: Vec<f64> = (0..nv)
            .map(|v| geom::norm(geom::sub(skin[t2r[v]], fr.truth_skin[f][v])) / truth.len)
            .collect();
        let mut rs = vec![(0.0, 0.0); nb];
        for v in 0..nv {
            let d = truth.rig.dominant[v];
            if d != usize::MAX {
                region_sum[d] += e[v];
                region_n[d] += 1.0;
                rs[d].0 += e[v];
                rs[d].1 += 1.0;
            }
        }
        let m = mean(&e);
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95 = percentile(&e, 0.95);
        if p95 > worst.1 {
            worst = (f, p95);
            worst_regions = rs;
        }
        per_frame.push((m, p95, *e.last().unwrap_or(&0.0)));
        skins.push(skin);
        joints.push(jw);
    }
    // per clip, then the clips' mean: a long idle counts as much as a short attack
    let mut clips = vec![];
    let (mut cm, mut cp) = (vec![], vec![]);
    for (ci, clip) in truth.rig.clips.iter().enumerate() {
        let idx: Vec<usize> = (0..nf).filter(|&f| fr.clip[f] == ci).collect();
        if idx.is_empty() {
            continue;
        }
        let m = mean(&idx.iter().map(|&f| per_frame[f].0).collect::<Vec<_>>());
        let p = mean(&idx.iter().map(|&f| per_frame[f].1).collect::<Vec<_>>());
        let mx = idx.iter().map(|&f| per_frame[f].2).fold(0.0, f64::max);
        cm.push(m);
        cp.push(p);
        clips.push(json!({"clip": clip.name, "frames": idx.len(), "mean_pct": pct(m), "p95_pct": pct(p), "max_pct": pct(mx),
            "per_frame_p95_pct": idx.iter().map(|&f| pct(per_frame[f].1)).collect::<Vec<_>>()}));
    }
    let mut regions: Vec<(String, f64)> = (0..nb)
        .filter(|&b| region_n[b] > 0.0)
        .map(|b| (truth.rig.names[b].clone(), region_sum[b] / region_n[b]))
        .collect();
    regions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let where_worst = (0..nb)
        .filter(|&b| worst_regions.get(b).map(|x| x.1 > 0.0).unwrap_or(false))
        .max_by(|&a, &b| {
            (worst_regions[a].0 / worst_regions[a].1)
                .partial_cmp(&(worst_regions[b].0 / worst_regions[b].1))
                .unwrap()
        })
        .map(|b| truth.rig.names[b].clone());
    let json = json!({
        "mean_pct": pct(mean(&cm)),
        "p95_pct": pct(mean(&cp)),
        "max_pct": pct(per_frame.iter().map(|x| x.2).fold(0.0, f64::max)),
        "worst_frame": if nf > 0 { json!({"clip": truth.rig.clips[fr.clip[worst.0]].name, "time": r4(fr.time[worst.0]), "index": worst.0,
            "p95_pct": pct(worst.1), "where": where_worst}) } else { Value::Null },
        "where": regions.iter().take(6).map(|(n, e)| json!({"under": n, "mean_pct": pct(*e)})).collect::<Vec<_>>(),
        "clips": clips,
    });
    SkinRun {
        json,
        skins,
        joints,
    }
}

/// The feet: skin held by a leg bone that stands on the ground in the
/// rest pose, gathered into feet by where it stands. While the artist's
/// foot is on the ground, how far the route's is from it across the
/// ground, and how far under or over it.
pub fn feet(truth: &Truth, fr: &Frames, run: &SkinRun, t2r: &[usize]) -> Value {
    let (lo, _) = bbox(&truth.rest_pos);
    let ground = lo[1];
    let len = truth.len;
    let sole: Vec<usize> = (0..truth.rest_pos.len())
        .filter(|&v| {
            let d = truth.rig.dominant[v];
            d != usize::MAX
                && matches!(truth.roles[d], Role::Leg | Role::Toe)
                && truth.rest_pos[v][1] - ground < 0.05 * len
        })
        .collect();
    // gather into feet: soles within 6% of the body's length of each other
    let mut foot = vec![usize::MAX; sole.len()];
    let mut n_feet = 0;
    for i in 0..sole.len() {
        if foot[i] != usize::MAX {
            continue;
        }
        foot[i] = n_feet;
        let mut stack = vec![i];
        while let Some(k) = stack.pop() {
            for m in 0..sole.len() {
                if foot[m] == usize::MAX {
                    let (a, b) = (truth.rest_pos[sole[k]], truth.rest_pos[sole[m]]);
                    if ((a[0] - b[0]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() < 0.06 * len {
                        foot[m] = n_feet;
                        stack.push(m);
                    }
                }
            }
        }
        n_feet += 1;
    }
    let (mut slide, mut sink, mut lift) = (vec![], 0.0f64, 0.0f64);
    let mut grounded = 0usize;
    for f in 0..fr.time.len() {
        for k in 0..n_feet {
            let vs: Vec<usize> = (0..sole.len())
                .filter(|&i| foot[i] == k)
                .map(|i| sole[i])
                .collect();
            let t_low = vs
                .iter()
                .map(|&v| fr.truth_skin[f][v][1])
                .fold(f64::INFINITY, f64::min);
            if t_low - ground > 0.02 * len {
                continue;
            }
            grounded += 1;
            let r_low = vs
                .iter()
                .map(|&v| run.skins[f][t2r[v]][1])
                .fold(f64::INFINITY, f64::min);
            let (mut ct, mut cr) = ([0.0; 3], [0.0; 3]);
            for &v in &vs {
                ct = geom::add(ct, fr.truth_skin[f][v]);
                cr = geom::add(cr, run.skins[f][t2r[v]]);
            }
            let d = geom::scale(geom::sub(cr, ct), 1.0 / vs.len() as f64);
            slide.push((d[0] * d[0] + d[2] * d[2]).sqrt() / len);
            sink = sink.max((t_low - r_low) / len);
            lift = lift.max((r_low - t_low) / len);
        }
    }
    let m = mean(&slide);
    slide.sort_by(|a, b| a.partial_cmp(b).unwrap());
    json!({
        "feet": n_feet,
        "frames_on_the_ground": grounded,
        "off_mean_pct": pct(m),
        "off_p95_pct": pct(percentile(&slide, 0.95)),
        "sink_max_pct": pct(sink),
        "lift_max_pct": pct(lift),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_rotation_finds_a_known_turn() {
        // 40 degrees about a slanted axis, weights uneven
        let axis = {
            let a = [0.3, 0.8, -0.5];
            geom::scale(a, 1.0 / geom::norm(a))
        };
        let (s, c) = 40f64.to_radians().sin_cos();
        let turn = |v: V3| -> V3 {
            let k = geom::cross(axis, v);
            let d = geom::dot(axis, v);
            [
                v[0] * c + k[0] * s + axis[0] * d * (1.0 - c),
                v[1] * c + k[1] * s + axis[1] * d * (1.0 - c),
                v[2] * c + k[2] * s + axis[2] * d * (1.0 - c),
            ]
        };
        let pts = [
            [1.0, 0.2, 0.0],
            [0.0, 1.0, 0.3],
            [-0.4, 0.1, 1.0],
            [0.7, -0.6, 0.2],
            [0.1, 0.1, -0.9],
        ];
        let pairs: Vec<(V3, V3, f64)> = pts
            .iter()
            .enumerate()
            .map(|(i, p)| (*p, turn(*p), 1.0 + i as f64))
            .collect();
        let r = best_rotation(&pairs);
        for p in pts {
            let got = m3_apply(&r, p);
            let want = turn(p);
            assert!(
                geom::norm(geom::sub(got, want)) < 1e-9,
                "{got:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn align_finds_a_turned_scaled_moved_mesh() {
        let truth: Vec<V3> = (0..200)
            .map(|i| {
                let t = i as f64 * 0.37;
                [
                    t.sin() * 0.3 + 0.1 * (t * 1.7).cos(),
                    (t * 0.61).cos() * 0.5 + 0.5,
                    t.cos() * 0.9 + 0.2 * (t * 2.3).sin(),
                ]
            })
            .collect();
        // a quarter turn and three degrees more, 4% larger, moved
        let (s, c) = 93f64.to_radians().sin_cos();
        let route: Vec<V3> = truth
            .iter()
            .map(|p| {
                [
                    1.04 * (c * p[0] + s * p[2]) + 0.2,
                    1.04 * p[1],
                    1.04 * (-s * p[0] + c * p[2]) - 0.1,
                ]
            })
            .collect();
        let al = align(&truth, &route);
        assert!(al.fit < 1e-4, "fit {}", al.fit);
        assert!((al.scale - 1.04).abs() < 2e-3, "scale {}", al.scale);
        assert!((al.yaw_deg - 93.0).abs() < 0.5, "yaw {}", al.yaw_deg);
    }

    #[test]
    fn roles_are_read_from_names() {
        assert_eq!(role_of("b_LeftForeArm_010"), Role::Leg); // not an ear
        assert_eq!(role_of("Tail3"), Role::Tail);
        assert_eq!(role_of("Shoulders"), Role::Trunk);
        assert_eq!(role_of("Shoulder.L"), Role::Leg);
        assert_eq!(role_of("Jaw"), Role::Jaw);
        assert_eq!(role_of("Wing2.R"), Role::Wing);
        assert_eq!(role_of("PoleTarget.L"), Role::Other);
    }
}
