//! A skeleton judged with no artist's rig to hold it against, and with
//! no reading of the body by the tool that made it: only the rig's own
//! joints, the skin they hold and the mirror plane.
//!
//! What it asks is what any animal's skeleton has: a left for every
//! right, a head, arms on a body that stands up, and names that agree
//! with where a limb joins the trunk.

use serde_json::{json, Value};
use skintokens_geom::geom::{self, V3};

use crate::rig::Rig;

struct Limb {
    root: usize,
    /// Joints from the root to the farthest tip.
    path: Vec<usize>,
    joints: usize,
    length: f64,
    mass: f64,
    /// +1 on +X, -1 on -X.
    side: i32,
    attach: V3,
    tip: V3,
    partner: Option<usize>,
    kind: &'static str,
}

fn named(name: &str) -> Option<&'static str> {
    let n = name.to_lowercase();
    for (word, kind) in [
        ("tail", "tail"),
        ("neck", "neck"),
        ("head", "head"),
        ("spine", "trunk"),
        ("hips", "trunk"),
        ("wing", "wing"),
        ("arm", "arm"),
        ("leg", "leg"),
    ] {
        if n.contains(word) {
            return Some(kind);
        }
    }
    None
}

fn r3(x: f64) -> f64 {
    (x * 1e3).round() / 1e3
}

pub fn check(rig: &Rig) -> Value {
    let n = rig.joints.len();
    let pos = rig.joint_rest();
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for p in &rig.rest_pos {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let size = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let len = size.iter().cloned().fold(0.0, f64::max);
    let mut xs: Vec<f64> = rig.rest_pos.iter().map(|p| p[0]).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    let x0 = xs[xs.len() / 2];
    let (mut lxlo, mut lxhi) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in rig
        .rest_pos
        .iter()
        .filter(|p| p[1] - lo[1] <= 0.40 * size[1])
    {
        lxlo = lxlo.min(p[0]);
        lxhi = lxhi.max(p[0]);
    }
    let stance_w = (if lxhi > lxlo { lxhi - lxlo } else { size[0] }).min(0.5 * len);
    let tol = (0.015 * len).max(0.10 * stance_w);

    // joints that move skin
    let mut live = vec![false; n];
    for &j in rig.joint_order.iter().rev() {
        if rig.mass[j] > 0.5 {
            live[j] = true;
        }
        if live[j] {
            if let Some(p) = rig.parent[j] {
                live[p] = true;
            }
        }
    }
    let kids = |j: usize| -> Vec<usize> {
        rig.children[j]
            .iter()
            .copied()
            .filter(|&c| live[c])
            .collect()
    };
    let below = |j: usize| -> Vec<usize> {
        let mut out = vec![];
        let mut stack = vec![j];
        while let Some(k) = stack.pop() {
            out.push(k);
            stack.extend(kids(k));
        }
        out
    };
    // the middle of the skin each joint holds
    let mut held = vec![([0.0f64; 3], 0.0f64); n];
    for (v, list) in rig.inf.iter().enumerate() {
        for i in list {
            held[i.joint].0 = geom::add(held[i.joint].0, geom::scale(rig.rest_pos[v], i.w));
            held[i.joint].1 += i.w;
        }
    }
    let skin_of = |js: &[usize]| -> (V3, f64) {
        let (mut c, mut m) = ([0.0; 3], 0.0);
        for &j in js {
            c = geom::add(c, held[j].0);
            m += held[j].1;
        }
        (if m > 0.0 { geom::scale(c, 1.0 / m) } else { c }, m)
    };

    // From the top of the tree down: what hangs off a joint is a side
    // limb when all of it lies on one side of the mirror plane (its first
    // joint may sit on the plane) and the middle of its skin is off the
    // plane. Anything else carries on the centre line.
    let mut side = vec![false; n];
    let mut roots = vec![];
    let mut swapped: Vec<String> = vec![];
    let mut stack: Vec<usize> = (0..n)
        .filter(|&j| live[j] && rig.parent[j].is_none())
        .collect();
    while let Some(j) = stack.pop() {
        for c in kids(j) {
            let sub = below(c);
            let xs: Vec<f64> = sub.iter().map(|&k| pos[k][0] - x0).collect();
            let pos_side = xs.iter().all(|x| *x > -tol / 2.0);
            let neg_side = xs.iter().all(|x| *x < tol / 2.0);
            let one_side = pos_side || neg_side;
            let j_mean = xs.iter().sum::<f64>() / xs.len().max(1) as f64;
            let (mid, mass) = skin_of(&sub);
            if one_side && mass > 0.0 && (mid[0] - x0).abs() > tol {
                for &k in &sub {
                    side[k] = true;
                }
                if j_mean.abs() > tol / 2.0 && (mid[0] - x0) * j_mean < 0.0 {
                    swapped.push(format!(
                        "{} sits on the {} side while the skin it holds lies on the {} side",
                        rig.names[c],
                        if j_mean > 0.0 { "+x" } else { "-x" },
                        if mid[0] - x0 > 0.0 { "+x" } else { "-x" }
                    ));
                }
                roots.push(c);
            } else {
                stack.push(c);
            }
        }
    }

    let mut limbs: Vec<Limb> = vec![];
    for &r in &roots {
        // the farthest tip by length along the bones
        let mut best: (f64, Vec<usize>) = (-1.0, vec![r]);
        let mut stack = vec![(r, 0.0, vec![r])];
        let (mut count, mut sum_x) = (0usize, 0.0);
        while let Some((j, d, path)) = stack.pop() {
            count += 1;
            sum_x += pos[j][0] - x0;
            let ks = kids(j);
            if ks.is_empty() && d > best.0 {
                best = (d, path.clone());
            }
            for c in ks {
                let mut p = path.clone();
                p.push(c);
                stack.push((c, d + geom::norm(geom::sub(pos[c], pos[j])), p));
            }
        }
        let tip = pos[*best.1.last().unwrap()];
        let (_, mass) = skin_of(&below(r));
        limbs.push(Limb {
            root: r,
            joints: best.1.len(),
            path: best.1,
            length: best.0.max(0.0),
            mass,
            side: if sum_x / count as f64 >= 0.0 { 1 } else { -1 },
            attach: pos[r],
            tip,
            partner: None,
            kind: "limb",
        });
    }

    let mut findings: Vec<Value> = vec![];
    let mut fail = |code: &str, detail: String| {
        findings.push(json!({"code": code, "severity": "fail", "detail": detail}))
    };
    for msg in swapped {
        fail("B_SWAPPED", msg);
    }
    let label = |l: &Limb| -> String {
        format!(
            "{} ({} joints, on the {})",
            rig.names[l.root],
            l.joints,
            if l.side > 0 { "+x side" } else { "-x side" }
        )
    };

    let step = (rig.rest_pos.len() / 4000).max(1);
    let tree = geom::KdTree::new(rig.rest_pos.clone());
    let sample: Vec<V3> = rig.rest_pos.iter().step_by(step).copied().collect();
    let asym = sample
        .iter()
        .map(|p| tree.knn([2.0 * x0 - p[0], p[1], p[2]], 1)[0].0)
        .sum::<f64>()
        / sample.len() as f64
        / len;
    let lopsided_body = asym > 0.02;

    // left and right: each limb has a partner that joins the trunk at the
    // mirrored place, with as many joints (one more or less) and about
    // the same length
    let mirror = |p: V3| [2.0 * x0 - p[0], p[1], p[2]];
    let mut cand: Vec<(f64, usize, usize)> = vec![];
    for a in 0..limbs.len() {
        for b in 0..limbs.len() {
            if limbs[a].side > 0 && limbs[b].side < 0 {
                cand.push((
                    geom::norm(geom::sub(limbs[a].attach, mirror(limbs[b].attach))),
                    a,
                    b,
                ));
            }
        }
    }
    cand.sort_by(|x, y| x.0.total_cmp(&y.0));
    for (d, a, b) in cand {
        // a body in a lopsided pose holds its shoulders unevenly
        if d > (if lopsided_body { 0.25 } else { 0.15 }) * len
            || limbs[a].partner.is_some()
            || limbs[b].partner.is_some()
        {
            continue;
        }
        limbs[a].partner = Some(b);
        limbs[b].partner = Some(a);
        let (la, lb) = (&limbs[a], &limbs[b]);
        let (lo_len, hi_len) = (la.length.min(lb.length), la.length.max(lb.length));
        let ratio = if lo_len > 1e-9 {
            hi_len / lo_len
        } else if hi_len > 0.05 * len {
            f64::INFINITY
        } else {
            1.0
        };
        if (la.joints as i64 - lb.joints as i64).abs() > 1 || ratio > 1.5 {
            fail(
                "B_LOPSIDED",
                format!(
                    "a left and its right differ: {} is {:.2} long, {} is {:.2} long",
                    label(la),
                    la.length,
                    label(lb),
                    lb.length
                ),
            );
        }
        if !lopsided_body {
            let (lo_mass, hi_mass) = (la.mass.min(lb.mass), la.mass.max(lb.mass));
            if hi_mass >= 5.0 && lo_mass / hi_mass < 0.35 {
                fail(
                    "B_FALLOFF",
                    format!(
                        "a left and its right hold very different amounts of skin: {} holds {:.1}, {} holds {:.1}",
                        label(la),
                        la.mass,
                        label(lb),
                        lb.mass
                    ),
                );
            }
        }
    }
    let mut unpaired: Vec<usize> = (0..limbs.len())
        .filter(|&i| limbs[i].partner.is_none())
        .collect();

    // the centre line and which way the body stands
    let centre: Vec<usize> = (0..n).filter(|&j| live[j] && !side[j]).collect();
    let mut trunk_lo = [f64::INFINITY; 3];
    let mut trunk_hi = [f64::NEG_INFINITY; 3];
    for (v, &d) in rig.dominant.iter().enumerate() {
        if d != usize::MAX && !side[d] {
            for a in 0..3 {
                trunk_lo[a] = trunk_lo[a].min(rig.rest_pos[v][a]);
                trunk_hi[a] = trunk_hi[a].max(rig.rest_pos[v][a]);
            }
        }
    }
    let upright = trunk_hi[1] - trunk_lo[1] > trunk_hi[2] - trunk_lo[2];
    let axis = if upright { 1 } else { 2 };
    for &j in &centre {
        let stray = (pos[j][0] - x0).abs();
        if stray > 0.15 * len {
            fail(
                "B_OFF_CENTRE",
                format!("{} is on the centre line of the skeleton and lies {:.0}% of the body's length off the mirror plane", rig.names[j], 100.0 * stray / len),
            );
        }
    }

    // where the paired limbs join, along the body's axis
    let pairs: Vec<(usize, usize)> = (0..limbs.len())
        .filter(|&a| limbs[a].side > 0 && limbs[a].partner.is_some())
        .map(|a| (a, limbs[a].partner.unwrap()))
        .collect();
    let joins: Vec<f64> = pairs
        .iter()
        .map(|p| (limbs[p.0].attach[axis] + limbs[p.1].attach[axis]) / 2.0)
        .collect();
    let jmax = joins.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    // A head: a bone on the centre line with no limb hanging from it or
    // from anything below it, holding at least a hundredth of the skin,
    // whose skin lies past the last pair of limbs toward the head's end
    // (up for a body that stands; forward, +Z, for one that is level).
    let verts = rig.rest_pos.len() as f64;
    let mut has_head = false;
    for &j in &centre {
        let sub = below(j);
        if sub.iter().any(|&k| side[k]) {
            continue;
        }
        let (mid, mass) = skin_of(&sub);
        if mass >= 0.01 * verts && (pairs.is_empty() || mid[axis] > jmax - 0.1 * len) {
            has_head = true;
        }
    }
    // A body in a lopsided pose may hold its head to one side: then the
    // head is the one limb without a partner that joins the centre line
    // past the last pair and holds a head's worth of skin.
    let mut leaning_head = None;
    if !has_head && lopsided_body {
        leaning_head = unpaired
            .iter()
            .copied()
            .filter(|&i| {
                let (_, mass) = skin_of(&below(limbs[i].root));
                mass >= 0.01 * verts
                    && (pairs.is_empty() || limbs[i].attach[axis] > jmax - 0.1 * len)
            })
            .max_by(|&a, &b| limbs[a].attach[axis].total_cmp(&limbs[b].attach[axis]));
        if let Some(h) = leaning_head {
            has_head = true;
            limbs[h].kind = "head";
            unpaired.retain(|&i| i != h);
        }
    }
    for &i in &unpaired {
        fail(
            "B_LOPSIDED",
            format!("{} has no partner on the other side", label(&limbs[i])),
        );
    }
    if !has_head && !pairs.is_empty() {
        fail(
            "B_NO_HEAD",
            format!(
                "no bone of its own on the centre line holds skin past the {} limbs: nothing to turn a head with",
                if upright { "upper" } else { "front" }
            ),
        );
    }

    // kinds, by where a pair joins the trunk
    let (mut clo, mut chi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &j in &centre {
        clo = clo.min(pos[j][axis]);
        chi = chi.max(pos[j][axis]);
    }
    let span = (chi - clo).max(1e-9);
    let top = joins.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    for (k, p) in pairs.iter().enumerate() {
        let t = (joins[k] - clo) / span;
        let kind = if upright {
            // with two pairs or more the top one is the arms; a lone pair
            // is arms only if it joins the upper part of the centre line
            if (pairs.len() >= 2 && joins[k] >= top - 1e-9) || (pairs.len() == 1 && t > 0.4) {
                "arm"
            } else {
                "leg"
            }
        } else if t > 0.5 {
            "front leg"
        } else {
            "hind leg"
        };
        limbs[p.0].kind = kind;
        limbs[p.1].kind = kind;
    }
    if upright && !pairs.iter().any(|p| limbs[p.0].kind == "arm") {
        fail(
            "B_NO_ARMS",
            "the body stands up and no pair of limbs joins the upper part of its trunk".into(),
        );
    }
    // Feet: where the body stands on the ground off the mirror plane,
    // the skin there is held by a side limb, not by the trunk. (A hem or
    // a tail that touches the ground on the plane is not a foot.)
    let ground = lo[1];
    let standing: Vec<usize> = (0..rig.rest_pos.len())
        .filter(|&v| rig.rest_pos[v][1] - ground < 0.04 * size[1] && rig.dominant[v] != usize::MAX)
        .collect();
    let mid_seam = 0.012 * len;
    let mut patch = vec![usize::MAX; standing.len()];
    let mut n_patches = 0;
    let cell = 0.05 * len;
    let mut grid: std::collections::HashMap<(i64, i64), Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &v) in standing.iter().enumerate() {
        let p = rig.rest_pos[v];
        grid.entry(((p[0] / cell).floor() as i64, (p[2] / cell).floor() as i64))
            .or_default()
            .push(i);
    }
    for i in 0..standing.len() {
        if patch[i] != usize::MAX {
            continue;
        }
        patch[i] = n_patches;
        let mut stack = vec![i];
        while let Some(k) = stack.pop() {
            let p = rig.rest_pos[standing[k]];
            let (gx, gz) = ((p[0] / cell).floor() as i64, (p[2] / cell).floor() as i64);
            for dx in -1..=1 {
                for dz in -1..=1 {
                    for &m in grid.get(&(gx + dx, gz + dz)).into_iter().flatten() {
                        if patch[m] == usize::MAX {
                            let q = rig.rest_pos[standing[m]];
                            let (px, qx) = (p[0] - x0, q[0] - x0);
                            let cross_midline = px.abs() > mid_seam
                                && qx.abs() > mid_seam
                                && px.signum() != qx.signum();
                            if !cross_midline
                                && ((p[0] - q[0]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() < cell
                            {
                                patch[m] = n_patches;
                                stack.push(m);
                            }
                        }
                    }
                }
            }
        }
        n_patches += 1;
    }
    if !limbs.is_empty() && limbs.iter().all(|l| l.joints < 2) {
        fail(
            "B_NO_LEG",
            "every side limb has only 1 joint: nothing can bend a knee, elbow or ankle".into(),
        );
    }
    let foot_tol = (0.015 * len).max(0.06 * stance_w);
    let mut feet = 0;
    for k in 0..n_patches {
        let vs: Vec<usize> = (0..standing.len())
            .filter(|&i| patch[i] == k)
            .map(|i| standing[i])
            .collect();
        if vs.len() < 3 {
            continue;
        }
        let xs: Vec<f64> = vs.iter().map(|&v| rig.rest_pos[v][0] - x0).collect();
        let mid = xs.iter().sum::<f64>() / xs.len() as f64;
        let touches_seam = xs.iter().any(|x| x.abs() <= mid_seam);
        let one_side =
            xs.iter().all(|x| *x > -foot_tol / 2.0) || xs.iter().all(|x| *x < foot_tol / 2.0);
        if touches_seam || !one_side || mid.abs() <= foot_tol {
            continue;
        }
        feet += 1;
        let patch_side = if mid > 0.0 { 1 } else { -1 };
        let foot_pos = {
            let mut c = [0.0; 3];
            for &v in &vs {
                c = geom::add(c, rig.rest_pos[v]);
            }
            geom::scale(c, 1.0 / vs.len() as f64)
        };
        let on_limb = vs.iter().filter(|&&v| side[rig.dominant[v]]).count();
        let near_joint = limbs.iter().filter(|l| l.side == patch_side).any(|l| {
            l.path
                .iter()
                .any(|&j| geom::norm(geom::sub(pos[j], foot_pos)) < 0.30 * len)
        });
        if on_limb * 2 < vs.len() || !near_joint {
            let mut count: std::collections::BTreeMap<usize, usize> =
                std::collections::BTreeMap::new();
            for &v in &vs {
                let d = rig.dominant[v];
                if d < rig.names.len() {
                    *count.entry(d).or_default() += 1;
                }
            }
            let holder = count
                .into_iter()
                .max_by(|(j1, c1), (j2, c2)| c1.cmp(c2).then_with(|| j2.cmp(j1)))
                .map(|(j, _)| rig.names[j].as_str())
                .unwrap_or("unweighted");
            let z = vs.iter().map(|&v| rig.rest_pos[v][2]).sum::<f64>() / vs.len() as f64;
            fail(
                "B_NO_LEG",
                format!(
                    "the body stands on a foot on its {} side ({:.0}% of its length from the back) and has no limb joint near it (held by {})",
                    if mid > 0.0 { "+x" } else { "-x" },
                    100.0 * (z - lo[2]) / size[2].max(1e-9),
                    holder
                ),
            );
        }
    }

    // names against places
    for (i, l) in limbs.iter().enumerate() {
        if Some(i) == leaning_head {
            continue;
        }
        let Some(name) = named(&rig.names[l.root]) else {
            continue;
        };
        let wrong = match (name, l.kind) {
            ("tail", _) | ("neck", _) | ("head", _) | ("trunk", _) => {
                Some("is named for the centre line and hangs off the side as a limb".to_string())
            }
            ("leg", "arm") => Some(
                "is named a leg and joins the upper half of a standing trunk, where arms do"
                    .to_string(),
            ),
            ("arm", "leg") => Some(
                "is named an arm and joins the lower half of a standing trunk, where legs do"
                    .to_string(),
            ),
            _ => None,
        };
        if let Some(w) = wrong {
            fail("B_KIND", format!("{} {w}", label(l)));
        }
    }

    let limbs_json: Vec<Value> = limbs
        .iter()
        .map(|l| {
            json!({
                "from": rig.names[l.root], "to": rig.names[*l.path.last().unwrap()], "joints": l.joints, "length": r3(l.length),
                "skin_mass": r3(l.mass),
                "side": if l.side > 0 { "+x" } else { "-x" }, "kind": l.kind,
                "partner": l.partner.map(|p| rig.names[limbs[p].root].clone()),
                "reaches_the_ground": l.tip[1] - ground < 0.12 * size[1],
            })
        })
        .collect();
    // A body that is itself lopsided (a hand, a figure with one arm
    // under a cloak) cannot have a left for every right: how far the
    // mesh is from its own mirror image says so, and its lopsided
    // findings are then warnings.
    if lopsided_body {
        for f in findings.iter_mut() {
            if f["code"] == json!("B_LOPSIDED") {
                f["severity"] = json!("warn");
                f["detail"] = json!(format!(
                    "{} (the body itself is lopsided: it lies {:.1}% of its length from its mirror image)",
                    f["detail"].as_str().unwrap_or(""),
                    100.0 * asym
                ));
            }
        }
        // a pose can move a limb off its mirror place; it cannot change
        // how many limbs a side has
        let (right, left) = (
            limbs
                .iter()
                .filter(|l| l.side > 0 && l.kind != "head")
                .count(),
            limbs
                .iter()
                .filter(|l| l.side < 0 && l.kind != "head")
                .count(),
        );
        if right != left {
            findings.push(json!({"code": "B_LOPSIDED", "severity": "fail",
                "detail": format!("{right} limbs on the +x side and {left} on the -x side")}));
        }
    }
    let pass = !findings.iter().any(|f| f["severity"] == json!("fail"));
    json!({
        "format": "creature-bench-anatomy/1",
        "pass": pass,
        "body_off_its_mirror_image_pct": (asym * 1e4).round() / 1e2,
        "stands": if upright { "upright" } else { "level" },
        "limb_pairs": pairs.len(),
        "limbs_without_a_partner": unpaired.len(),
        "head": has_head,
        "ground_patches_off_the_plane": feet,
        "centre_joints": centre.len(),
        "findings": findings,
        "limbs": limbs_json,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rig::{Influence, Trs};
    use std::collections::HashMap;

    fn test_quad_rig(swap_lr: bool, stub_legs: bool) -> Rig {
        // Simple symmetric quadruped: root(0), head(1), legL0(2)->legL1(3), legR0(4)->legR1(5)
        let jpos = if stub_legs {
            vec![
                [0.0, 0.5, 0.0],
                [0.0, 0.6, 0.5],
                [0.2, 0.4, 0.2],
                [-0.2, 0.4, 0.2],
            ]
        } else {
            vec![
                [0.0, 0.5, 0.0],
                [0.0, 0.6, 0.5],
                [0.2, 0.4, 0.2],
                [0.2, 0.05, 0.2],
                [-0.2, 0.4, 0.2],
                [-0.2, 0.05, 0.2],
            ]
        };
        let nj = jpos.len();
        let trs: Vec<Trs> = jpos
            .iter()
            .map(|&t| Trs {
                t,
                r: [0.0, 0.0, 0.0, 1.0],
                s: [1.0, 1.0, 1.0],
            })
            .collect();
        let rest_world: Vec<geom::M4> = trs.iter().map(|x| x.matrix()).collect();
        let parent = if stub_legs {
            vec![None, Some(0), Some(0), Some(0)]
        } else {
            vec![None, Some(0), Some(0), Some(2), Some(0), Some(4)]
        };
        let mut children = vec![vec![]; nj];
        for (j, p) in parent.iter().enumerate() {
            if let Some(p) = *p {
                children[p].push(j);
            }
        }
        let mut rest_pos = vec![];
        let mut dominant = vec![];
        // Trunk & head vertices (100)
        for k in 0..80 {
            let z = -0.4 + 0.8 * (k as f64 / 80.0);
            let x = if k % 2 == 0 { 0.05 } else { -0.05 };
            rest_pos.push([x, 0.5, z]);
            dominant.push(0);
        }
        for k in 0..20 {
            let x = if k % 2 == 0 { 0.03 } else { -0.03 };
            rest_pos.push([x, 0.6, 0.45 + 0.01 * k as f64]);
            dominant.push(1);
        }
        // Left & right feet on ground y = 0.0
        let (jl, jr) = if stub_legs { (2, 3) } else { (3, 5) };
        let (hold_l, hold_r) = if swap_lr { (jr, jl) } else { (jl, jr) };
        for k in 0..10 {
            rest_pos.push([0.18 + 0.005 * k as f64, 0.0, 0.18 + 0.005 * k as f64]);
            dominant.push(hold_l);
        }
        for k in 0..10 {
            rest_pos.push([-0.18 - 0.005 * k as f64, 0.0, 0.18 + 0.005 * k as f64]);
            dominant.push(hold_r);
        }
        let mut mass = vec![1.0; nj];
        let inf: Vec<Vec<Influence>> = dominant
            .iter()
            .map(|&j| {
                mass[j] += 1.0;
                vec![Influence {
                    joint: j,
                    w: 1.0,
                    q: [0.0; 3],
                }]
            })
            .collect();
        Rig {
            glb: skintokens_geom::glb::Glb {
                json: serde_json::json!({}),
                bin: vec![],
            },
            node_parent: parent.clone(),
            order: (0..nj).collect(),
            rest: trs,
            rest_world,
            joints: (0..nj).collect(),
            joint_of: (0..nj).map(|i| (i, i)).collect::<HashMap<_, _>>(),
            names: (0..nj).map(|i| format!("b{i}")).collect(),
            parent,
            children,
            joint_order: (0..nj).collect(),
            inf,
            tris: vec![[0, 1, 2]],
            rest_pos,
            mass,
            dominant,
            clips: vec![],
        }
    }

    #[test]
    fn clean_quad_passes_while_swapped_and_stub_legs_fail() {
        let ok = check(&test_quad_rig(false, false));
        assert_eq!(ok["pass"], json!(true), "{ok}");

        let sw = check(&test_quad_rig(true, false));
        assert_eq!(sw["pass"], json!(false), "{sw}");
        assert!(sw["findings"].to_string().contains("B_SWAPPED"), "{sw}");

        let st = check(&test_quad_rig(false, true));
        assert_eq!(st["pass"], json!(false), "{st}");
        assert!(st["findings"].to_string().contains("B_NO_LEG"), "{st}");
    }

    #[test]
    fn no_leg_holder_tie_break_is_deterministic() {
        let mut rig = test_quad_rig(false, true);
        // Set up equal vertex counts (5 each on joint 0 and joint 1) for the +x foot patch
        let n = rig.rest_pos.len();
        for i in 0..5 {
            let idx = n - 20 + i;
            rig.dominant[idx] = 0;
            rig.dominant[idx + 5] = 1;
        }
        let first = check(&rig)["findings"].to_string();
        for _ in 0..40 {
            assert_eq!(check(&rig)["findings"].to_string(), first);
        }
    }
}
