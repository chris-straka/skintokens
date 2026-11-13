//! A skeleton judged with no artist's rig to hold it against, and with
//! no reading of the body by the tool that made it: only the rig's own
//! joints, the skin they hold and the mirror plane.
//!
//! What it asks is what any animal's skeleton has: a left for every
//! right, a head, arms on a body that stands up, and names that agree
//! with where a limb joins the trunk.

use serde_json::{json, Value};
use skintokens::geom::{self, V3};

use crate::rig::Rig;

struct Limb {
    root: usize,
    /// Joints from the root to the farthest tip.
    path: Vec<usize>,
    joints: usize,
    length: f64,
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
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let x0 = xs[xs.len() / 2];
    let tol = (0.015 * len).max(0.10 * size[0]);

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
    let mut stack: Vec<usize> = (0..n)
        .filter(|&j| live[j] && rig.parent[j].is_none())
        .collect();
    while let Some(j) = stack.pop() {
        for c in kids(j) {
            let sub = below(c);
            let xs: Vec<f64> = sub.iter().map(|&k| pos[k][0] - x0).collect();
            let one_side = xs.iter().all(|x| *x > -tol / 2.0) || xs.iter().all(|x| *x < tol / 2.0);
            let (mid, mass) = skin_of(&sub);
            if one_side && mass > 0.0 && (mid[0] - x0).abs() > tol {
                for &k in &sub {
                    side[k] = true;
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
        limbs.push(Limb {
            root: r,
            joints: best.1.len(),
            path: best.1,
            length: best.0.max(0.0),
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
    let label = |l: &Limb| -> String {
        format!(
            "{} ({} joints, on the {})",
            rig.names[l.root],
            l.joints,
            if l.side > 0 { "+x side" } else { "-x side" }
        )
    };

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
    cand.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    for (d, a, b) in cand {
        if d > 0.15 * len || limbs[a].partner.is_some() || limbs[b].partner.is_some() {
            continue;
        }
        limbs[a].partner = Some(b);
        limbs[b].partner = Some(a);
        let (la, lb) = (&limbs[a], &limbs[b]);
        let ratio = if la.length.min(lb.length) > 1e-9 {
            la.length.max(lb.length) / la.length.min(lb.length)
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
    }
    for l in &limbs {
        if l.partner.is_none() {
            fail(
                "B_LOPSIDED",
                format!("{} has no partner on the other side", label(l)),
            );
        }
    }

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
                            if ((p[0] - q[0]).powi(2) + (p[2] - q[2]).powi(2)).sqrt() < cell {
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
        let one_side = xs.iter().all(|x| *x > -tol / 2.0) || xs.iter().all(|x| *x < tol / 2.0);
        if !one_side || mid.abs() <= tol {
            continue;
        }
        feet += 1;
        let on_limb = vs.iter().filter(|&&v| side[rig.dominant[v]]).count();
        if on_limb * 2 < vs.len() {
            let mut count: std::collections::HashMap<usize, usize> =
                std::collections::HashMap::new();
            for &v in &vs {
                *count.entry(rig.dominant[v]).or_default() += 1;
            }
            let holder = count.iter().max_by_key(|x| *x.1).map(|x| *x.0).unwrap();
            let z = vs.iter().map(|&v| rig.rest_pos[v][2]).sum::<f64>() / vs.len() as f64;
            fail(
                "B_NO_LEG",
                format!(
                    "the body stands on a foot on its {} side ({} of its length from the back) and the bone that holds it, {}, is on the centre line: that leg has no bones",
                    if mid > 0.0 { "+x" } else { "-x" },
                    format!("{:.0}%", 100.0 * (z - lo[2]) / size[2].max(1e-9)),
                    rig.names[holder]
                ),
            );
        }
    }

    // names against places
    for l in &limbs {
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
                "side": if l.side > 0 { "+x" } else { "-x" }, "kind": l.kind,
                "partner": l.partner.map(|p| rig.names[limbs[p].root].clone()),
                "reaches_the_ground": l.tip[1] - ground < 0.12 * size[1],
            })
        })
        .collect();
    let pass = findings.is_empty();
    json!({
        "format": "creature-bench-anatomy/1",
        "pass": pass,
        "stands": if upright { "upright" } else { "level" },
        "limb_pairs": pairs.len(),
        "limbs_without_a_partner": limbs.iter().filter(|l| l.partner.is_none()).count(),
        "head": has_head,
        "ground_patches_off_the_plane": feet,
        "centre_joints": centre.len(),
        "findings": findings,
        "limbs": limbs_json,
    })
}
