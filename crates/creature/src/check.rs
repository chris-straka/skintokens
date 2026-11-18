//! The skeleton check: does a rig's skeleton fit the body it is in? The
//! body's parts are read from the mesh alone (`roles`), then every bone is
//! laid against them. Each leg, arm, tail, wing and the neck and head must
//! have a chain of bones along it, the bones must lie inside the body, and
//! a left limb and its right must have the same bones. The bend checker
//! cannot see any of this: a rig with no bones in its tail bends nothing
//! there and so tears nothing.

use serde_json::{json, Value};
use skintokens::geom::{self, V3};
use skintokens::glb::Glb;

use crate::parts::Body;
use crate::roles::{self, Anatomy, Role, Side};
use crate::skeleton::{seg_dist, Bone, Skeleton};

/// A rig's joints as bones in world space: names, parents, heads, tails.
/// A bone runs to the child that carries its chain on (the one farthest
/// along the way it came), a leaf 0.6 of its parent's length further.
///
/// Joints that hold no skin and carry no joint that does are left out:
/// they are the end markers an exporter writes past a foot or a head
/// (`*_end`), not bones that move anything. Ancestor container joints that
/// hold no skin themselves (`root`, `_rootJoint`) stay in the hierarchy
/// with `deforms = false` so they are not judged as body segments.
pub fn read_rig(g: &Glb) -> anyhow::Result<Vec<Bone>> {
    let joints: Vec<usize> = g.json["skins"][0]["joints"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("the file has no skin"))?
        .iter()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    let worlds = g.world_matrices();
    let node_parent = g.parents();
    let slot: std::collections::HashMap<usize, usize> =
        joints.iter().enumerate().map(|(k, &n)| (n, k)).collect();
    let parent: Vec<Option<usize>> = joints
        .iter()
        .map(|&n| {
            let mut p = node_parent[n];
            while let Some(x) = p {
                if let Some(&k) = slot.get(&x) {
                    return Some(k);
                }
                p = node_parent[x];
            }
            None
        })
        .collect();
    let head: Vec<V3> = joints
        .iter()
        .map(|&n| geom::translation(&worlds[n]))
        .collect();
    let n = joints.len();
    let mut tail = head.clone();
    let mut len = vec![0.0; n];
    let kids = |j: usize| {
        (0..n)
            .filter(|&k| parent[k] == Some(j) && geom::norm(geom::sub(head[k], head[j])) > 1e-9)
            .collect::<Vec<_>>()
    };
    // Blender's convention (a bone runs along its joint's local +Y): held
    // when four in five joints with one child point their +Y at it. Then a
    // leaf runs along its own +Y, as weightforge reads it.
    let yaxis: Vec<V3> = joints
        .iter()
        .map(|&nd| {
            let y = geom::cols(&worlds[nd])[1];
            let l = geom::norm(y);
            if l > 0.0 {
                geom::scale(y, 1.0 / l)
            } else {
                [0.0, 1.0, 0.0]
            }
        })
        .collect();
    let (mut agree, mut total) = (0, 0);
    for j in 0..n {
        let ch = kids(j);
        if ch.len() == 1 {
            let d = geom::sub(head[ch[0]], head[j]);
            total += 1;
            if geom::dot(geom::scale(d, 1.0 / geom::norm(d)), yaxis[j]) > 0.98 {
                agree += 1;
            }
        }
    }
    let y_convention = total > 0 && agree * 10 >= total * 8;
    for j in 0..n {
        let ch = kids(j);
        if ch.is_empty() {
            continue;
        }
        if y_convention {
            let l = ch
                .iter()
                .map(|&k| geom::dot(geom::sub(head[k], head[j]), yaxis[j]))
                .fold(0.0, f64::max);
            if l > 1e-9 {
                tail[j] = geom::add(head[j], geom::scale(yaxis[j], l));
                len[j] = l;
                continue;
            }
        }
        let dir = match parent[j] {
            Some(p) if geom::norm(geom::sub(head[j], head[p])) > 1e-9 => geom::scale(
                geom::sub(head[j], head[p]),
                1.0 / geom::norm(geom::sub(head[j], head[p])),
            ),
            _ => [0.0; 3],
        };
        let key = |k: usize| {
            geom::dot(geom::sub(head[k], head[j]), dir)
                + 1e-3 * geom::norm(geom::sub(head[k], head[j]))
        };
        let best = ch
            .iter()
            .copied()
            .max_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap())
            .unwrap();
        tail[j] = head[best];
        len[j] = geom::norm(geom::sub(tail[j], head[j]));
    }
    for j in 0..n {
        if len[j] > 0.0 {
            continue;
        }
        let plen = parent[j].map(|p| len[p]).unwrap_or(0.0);
        let l = if plen > 1e-9 { 0.6 * plen } else { 0.05 };
        let dir = match parent[j] {
            _ if y_convention => yaxis[j],
            Some(p) if geom::norm(geom::sub(head[j], head[p])) > 1e-9 => geom::scale(
                geom::sub(head[j], head[p]),
                1.0 / geom::norm(geom::sub(head[j], head[p])),
            ),
            _ => yaxis[j],
        };
        tail[j] = geom::add(head[j], geom::scale(dir, l));
    }
    let mut mass = vec![0.0f64; n];
    if let Ok(parts) = g.parts(skintokens::glb::Frame::SkinBind) {
        for p in &parts {
            if let (Some(js), Some(ws)) = (&p.joints, &p.weights) {
                for (j, w) in js.iter().zip(ws.iter()) {
                    for k in 0..4 {
                        let s = j[k] as usize;
                        if s < n {
                            mass[s] += w[k] as f64;
                        }
                    }
                }
            }
        }
    }
    let bones: Vec<Bone> = (0..n)
        .map(|j| Bone {
            name: g.nodes()[joints[j]]["name"]
                .as_str()
                .unwrap_or("joint")
                .to_string(),
            parent: parent[j],
            head: head[j],
            tail: tail[j],
            limb: None,
            trunk: false,
            index: 0,
            inside: false,
            radius: 0.0,
            deforms: mass[j] > 1e-6
                && g.nodes()[joints[j]]["name"].as_str() != Some("neutral_bone"),
        })
        .collect();
    // keep a joint that holds skin, or carries one that does
    let mut keep: Vec<bool> = mass.iter().map(|&m| m > 1e-6).collect();
    loop {
        let mut grew = false;
        for j in 0..n {
            if !keep[j] && (0..n).any(|k| keep[k] && parent[k] == Some(j)) {
                keep[j] = true;
                grew = true;
            }
        }
        if !grew {
            break;
        }
    }
    if keep.iter().all(|&k| k) {
        return Ok(bones);
    }
    let remap: Vec<Option<usize>> = {
        let mut m = vec![None; n];
        let mut next = 0;
        for j in 0..n {
            if keep[j] {
                m[j] = Some(next);
                next += 1;
            }
        }
        m
    };
    Ok(bones
        .into_iter()
        .enumerate()
        .filter(|(j, _)| keep[*j])
        .map(|(_j, mut b)| {
            let mut p = b.parent;
            while let Some(x) = p {
                if keep[x] {
                    break;
                }
                p = parent[x];
            }
            b.parent = p.and_then(|x| remap[x]);
            b
        })
        .collect())
}

/// The point of segment `ab` nearest to `p`.
fn seg_point(p: V3, a: V3, b: V3) -> V3 {
    let ab = geom::sub(b, a);
    let l2 = geom::dot(ab, ab);
    let t = if l2 > 0.0 {
        (geom::dot(geom::sub(p, a), ab) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    geom::add(a, geom::scale(ab, t))
}

/// Distance from `p` to a polyline and the arc length of the nearest point.
fn line_dist(p: V3, line: &[V3]) -> (f64, f64) {
    let mut best = (f64::INFINITY, 0.0);
    let mut s = 0.0;
    for k in 1..line.len() {
        let (a, b) = (line[k - 1], line[k]);
        let ab = geom::sub(b, a);
        let l = geom::norm(ab);
        let t = if l > 0.0 {
            (geom::dot(geom::sub(p, a), ab) / (l * l)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let d = geom::norm(geom::sub(p, geom::add(a, geom::scale(ab, t))));
        if d < best.0 {
            best = (d, s + t * l);
        }
        s += l;
    }
    best
}

pub struct Verdict {
    pub score: f64,
    pub report: Value,
    /// The rig's bones with the part each lies along (for the pose set).
    pub skeleton: Skeleton,
    pub failing: usize,
}

/// Bones needed along a limb for it to bend as that limb does.
fn needed(r: Role) -> usize {
    match r {
        Role::Leg | Role::Arm | Role::Wing | Role::Tail => 2,
        Role::Head | Role::Extra => 1,
    }
}

pub fn check(body: &Body, an: &Anatomy, mut bones: Vec<Bone>) -> Verdict {
    let size = body.size;
    let nb = bones.len();
    // A file stores no bone ends. A last bone's length is a guess (0.6 of
    // its parent's): it is taken to end where it leaves the body, if it
    // runs inside for a while first.
    for b in 0..nb {
        if bones.iter().any(|c| c.parent == Some(b)) {
            continue;
        }
        let (h, t) = (bones[b].head, bones[b].tail);
        let steps = 12;
        let inside = |k: usize| {
            body.solid
                .nearest_inside(
                    geom::add(h, geom::scale(geom::sub(t, h), k as f64 / steps as f64)),
                    1,
                )
                .is_some()
        };
        if let Some(out) = (1..=steps).find(|&k| !inside(k)) {
            if out >= 4 {
                bones[b].tail = geom::add(
                    h,
                    geom::scale(geom::sub(t, h), (out - 1) as f64 / steps as f64),
                );
            }
        }
    }
    let reach = |r: f64| (1.5 * r).max(0.04 * size);
    // --- which part each bone lies along: the limb (or the spine) most of
    // its length runs beside
    const SAMPLES: usize = 7;
    let mut along = vec![0.0f64; nb]; // arc length of the bone's middle on its limb
    for (bi, b) in bones.iter_mut().enumerate() {
        let pts: Vec<V3> = (0..SAMPLES)
            .map(|k| {
                geom::add(
                    b.head,
                    geom::scale(geom::sub(b.tail, b.head), (k as f64 + 0.5) / SAMPLES as f64),
                )
            })
            .collect();
        // votes[0] = spine, votes[1 + limb]
        let mut votes = vec![0usize; an.limbs.len() + 1];
        let mut arc = vec![0.0f64; an.limbs.len() + 1];
        for &p in &pts {
            // The part of the body this point of the bone sits in decides
            // first: a wing's bones run along its front edge, far from the
            // membrane's centre line, while still sitting inside the wing.
            let owner = body
                .solid
                .nearest_inside(p, 1)
                .map(|c| body.part_of[c])
                .filter(|&pt| pt != u32::MAX)
                .map(|pt| an.limb_of_part[pt as usize]);
            if let Some(owner) = owner {
                let w = owner.map(|l| l + 1).unwrap_or(0);
                let s = match owner {
                    Some(l) => line_dist(p, &an.limbs[l].line).1,
                    None => line_dist(p, &an.spine).1,
                };
                votes[w] += 1;
                arc[w] += s;
                continue;
            }
            let mut best: Option<(f64, usize, f64)> = None;
            // the nearest line among those the point is within reach of
            // (nearest in plain distance: a bone on a thin limb's own centre
            // line is that limb's, however thick the limb beside it)
            let (d, s) = line_dist(p, &an.spine);
            let r = roles::at(&an.spine, &an.spine_radius, s).1;
            if d <= reach(r) {
                best = Some((d, 0, s));
            }
            for (li, l) in an.limbs.iter().enumerate() {
                let (d, s) = line_dist(p, &l.line);
                // its line starts on the trunk's centre line: the first
                // third of the way out is the trunk's
                if s < 0.35 * l.emerge {
                    continue;
                }
                let r = l.at(s).1;
                if d <= reach(r) && best.is_none_or(|b| d < b.0) {
                    best = Some((d, li + 1, s));
                }
            }
            if let Some((_, w, s)) = best {
                votes[w] += 1;
                arc[w] += s;
            }
        }
        let (w, &n) = votes.iter().enumerate().max_by_key(|x| *x.1).unwrap();
        if n * 2 > SAMPLES {
            along[bi] = arc[w] / n as f64;
            if w == 0 {
                b.trunk = true;
            } else {
                b.limb = Some(w - 1);
                b.inside = along[bi] < an.limbs[w - 1].emerge;
            }
        }
    }
    // order within each limb, from the body outward
    for li in 0..an.limbs.len() {
        let mut ids: Vec<usize> = (0..nb).filter(|&b| bones[b].limb == Some(li)).collect();
        ids.sort_by(|&a, &b| along[a].partial_cmp(&along[b]).unwrap());
        for (k, &b) in ids.iter().enumerate() {
            bones[b].index = k;
        }
    }
    let mut trunk: Vec<usize> = (0..nb).filter(|&b| bones[b].trunk).collect();
    trunk.sort_by(|&a, &b| along[a].partial_cmp(&along[b]).unwrap());
    for (k, &b) in trunk.iter().enumerate() {
        bones[b].index = k;
    }

    let mut findings: Vec<Value> = vec![];
    // --- every limb has a chain
    let covered = |line: &[V3],
                   radius: &[f64],
                   from: f64,
                   to: f64,
                   ids: &[usize],
                   limb: Option<usize>|
     -> f64 {
        let owner_of = |q: V3| {
            body.solid
                .nearest_inside(q, 1)
                .map(|c| body.part_of[c])
                .filter(|&pt| pt != u32::MAX)
                .and_then(|pt| an.limb_of_part[pt as usize])
        };
        let n = 24;
        let mut hit = 0;
        for k in 0..n {
            let s = from + (to - from) * (k as f64 + 0.5) / n as f64;
            let (p, r) = roles::at(line, radius, s);
            let near = ids
                .iter()
                .any(|&b| seg_dist(p, bones[b].head, bones[b].tail) <= reach(r));
            let through = !near
                && limb.is_some_and(|li| {
                    owner_of(p) == Some(li)
                        && ids.iter().any(|&b| {
                            let q = seg_point(p, bones[b].head, bones[b].tail);
                            owner_of(q) == Some(li) && (line_dist(q, line).1 - s).abs() <= 0.3 * to
                        })
                });
            if near || through {
                hit += 1;
            }
        }
        hit as f64 / n as f64
    };
    let continues = |li: usize, lj: usize| -> bool {
        if li == lj || an.limbs[lj].parts.is_empty() {
            return false;
        }
        let mut p = body.parts[an.limbs[lj].parts[0]].parent;
        while let Some(x) = p {
            if an.limbs[li].parts.contains(&x) {
                return true;
            }
            p = body.parts[x].parent;
        }
        false
    };
    let inside_cells = body.solid.inside.iter().filter(|&&b| b).count().max(1);
    let mut limb_scores = vec![];
    let mut limb_rows = vec![];
    for (li, l) in an.limbs.iter().enumerate() {
        let ids: Vec<usize> = (0..nb)
            .filter(|&b| bones[b].limb == Some(li) && bones[b].deforms)
            .collect();
        // bones are asked for where the limb is out of the body; what a
        // rig has inside it (a shoulder blade, a thigh in the haunch) counts
        // as the limb's but is not demanded
        let cov = covered(&l.line, &l.radius, l.emerge, l.length, &ids, Some(li));
        let mut chain: Vec<usize> = ids.clone();
        for &b in &ids {
            let mut p = bones[b].parent;
            // An IK foot hung directly under the trajectory root meets its
            // shin's tail at the ankle rather than through `parent`.
            if p.is_some_and(|r| bones[r].parent.is_none() && !bones[r].deforms) {
                if let Some(shin) = (0..nb)
                    .filter(|&k| k != b && bones[k].deforms && bones[k].limb.is_none())
                    .min_by(|&a, &c| {
                        geom::norm(geom::sub(bones[a].tail, bones[b].head))
                            .partial_cmp(&geom::norm(geom::sub(bones[c].tail, bones[b].head)))
                            .unwrap()
                    })
                    .filter(|&k| geom::norm(geom::sub(bones[k].tail, bones[b].head)) <= 0.05 * size)
                {
                    p = Some(shin);
                }
            }
            let mut up = 0;
            while let Some(x) = p {
                if bones[x].parent.is_none() && !bones[x].deforms {
                    break;
                }
                let (d, s) = line_dist(bones[x].head, &l.line);
                let near = d <= reach(roles::at(&l.line, &l.radius, s).1).max(0.08 * size);
                let (sd, ss) = line_dist(bones[x].head, &an.spine);
                let off_spine = sd > reach(roles::at(&an.spine, &an.spine_radius, ss).1)
                    || (l.side != Side::Centre && bones[x].head[0].abs() > roles::stance_off(body));
                if bones[x].limb.is_none() && (near || off_spine) && up < 2 {
                    if !chain.contains(&x) {
                        chain.push(x);
                    }
                    up += 1;
                    p = bones[x].parent;
                } else {
                    break;
                }
            }
        }
        for lj in 0..an.limbs.len() {
            if continues(li, lj) {
                for b in (0..nb).filter(|&b| bones[b].limb == Some(lj) && bones[b].deforms) {
                    if !chain.contains(&b) {
                        chain.push(b);
                    }
                }
            }
        }
        let enough = chain.len() >= needed(l.role);
        let s = (cov / 0.8).min(1.0) * if enough { 1.0 } else { 0.5 };
        let cells: usize = l.parts.iter().map(|&p| body.parts[p].cells).sum();
        let share = (cells as f64 / inside_cells as f64 * 1e4).round() / 1e4;
        // A tiny, short non-leg nub (< 1.5% of the body and < 20% of body
        // size outside the trunk: a pug's curled tail, a wyvern's folded
        // fore-claws) may legitimately ride on its parent bone. Weight is
        // fixed per limb (never conditioned on ids.is_empty()) so adding a
        // bone along any limb always raises or keeps the score.
        let max_minor_len = if l.role == Role::Tail { 0.20 } else { 0.30 };
        let minor =
            l.role != Role::Leg && share < 0.015 && (l.length - l.emerge) < max_minor_len * size;
        let weight = if minor {
            0.05
        } else {
            match l.role {
                Role::Extra => 0.15,
                Role::Tail => 0.5,
                _ => 1.0,
            }
        };
        limb_scores.push((s, weight));
        limb_rows.push(
            json!({"limb": l.name, "bones": ids.len(), "covered": (cov * 100.0).round() / 100.0, "body_share": share}),
        );
        if cov < 0.5 || ids.is_empty() {
            let sev = if l.role == Role::Extra || minor {
                "warn"
            } else {
                "fail"
            };
            findings.push(
                json!({"code": "S_NO_CHAIN", "severity": sev, "part": l.name,
                "detail": format!("bones run along {:.0}% of it", cov * 100.0)}),
            );
        } else if !enough {
            let sev = if minor { "warn" } else { "fail" };
            findings.push(json!({"code": "S_SHORT_CHAIN", "severity": sev, "part": l.name,
                "detail": format!("{} bone(s) where it needs {} to bend", chain.len(), needed(l.role))}));
        }
    }
    let wsum: f64 = limb_scores.iter().map(|x| x.1).sum();
    let limbs = if wsum > 0.0 {
        limb_scores.iter().map(|x| x.0 * x.1).sum::<f64>() / wsum
    } else {
        1.0
    };
    let spine_cov = covered(
        &an.spine,
        &an.spine_radius,
        0.0,
        roles::length(&an.spine),
        &trunk,
        None,
    );
    if spine_cov < 0.5 {
        findings.push(json!({"code": "S_NO_CHAIN", "severity": "fail", "part": "spine", "detail": format!("bones run along {:.0}% of it", spine_cov * 100.0)}));
    }
    // --- bones lie inside the body (deforming bones only; trajectory root containers sit on the ground)
    let mut inside_all = (0usize, 0usize);
    let mut outside_bones = vec![];
    let deforming_nb = bones.iter().filter(|b| b.deforms).count().max(1);
    for b in bones.iter().filter(|b| b.deforms) {
        let mut inn = 0;
        for k in 0..SAMPLES {
            let p = geom::add(
                b.head,
                geom::scale(geom::sub(b.tail, b.head), (k as f64 + 0.5) / SAMPLES as f64),
            );
            if body.solid.nearest_inside(p, 1).is_some() {
                inn += 1;
            }
        }
        inside_all.0 += inn;
        inside_all.1 += SAMPLES;
        if inn * 10 < SAMPLES * 7 {
            outside_bones.push(b.name.clone());
        }
    }
    let inside = if inside_all.1 > 0 {
        inside_all.0 as f64 / inside_all.1 as f64
    } else {
        1.0
    };
    if !outside_bones.is_empty() {
        findings.push(json!({"code": "S_OUTSIDE", "severity": if outside_bones.len() * 10 > deforming_nb { "fail" } else { "warn" }, "part": "*",
            "detail": format!("{} of {} bones run outside the body: {}", outside_bones.len(), deforming_nb, outside_bones.iter().take(6).cloned().collect::<Vec<_>>().join(", "))}));
    }
    // --- a left and its right match
    let stem = |n: &str| n.trim_end_matches(".L").trim_end_matches(".R").to_string();
    let mut pairs = (0usize, 0usize);
    for (i, a) in an.limbs.iter().enumerate() {
        for (j, b) in an.limbs.iter().enumerate().skip(i + 1) {
            if a.side != Side::Centre
                && b.side != Side::Centre
                && a.side != b.side
                && stem(&a.name) == stem(&b.name)
                && a.role != Role::Extra
            {
                let (na, nb_) = (
                    (0..nb)
                        .filter(|&x| bones[x].limb == Some(i) && bones[x].deforms)
                        .count(),
                    (0..nb)
                        .filter(|&x| bones[x].limb == Some(j) && bones[x].deforms)
                        .count(),
                );
                pairs.1 += 1;
                if na.abs_diff(nb_) <= 1 {
                    pairs.0 += 1;
                    if na != nb_ {
                        findings.push(json!({"code": "S_LOPSIDED", "severity": "warn", "part": stem(&a.name), "detail": format!("{na} bones on one side, {nb_} on the other")}));
                    }
                } else {
                    findings.push(json!({"code": "S_LOPSIDED", "severity": "fail", "part": stem(&a.name), "detail": format!("{na} bones on one side, {nb_} on the other")}));
                }
            }
        }
    }
    let symmetry = if pairs.1 > 0 {
        pairs.0 as f64 / pairs.1 as f64
    } else {
        1.0
    };
    let stray = (0..nb)
        .filter(|&b| bones[b].deforms && !bones[b].trunk && bones[b].limb.is_none())
        .count();
    let score = 100.0
        * (0.55 * limbs + 0.15 * (spine_cov / 0.8).min(1.0) + 0.15 * inside + 0.15 * symmetry);
    let failing = findings.iter().filter(|f| f["severity"] == "fail").count();
    let report = json!({
        "format": "creature-skeleton-check/1",
        "kind": an.kind.name(),
        "upright": an.upright,
        "faces": if an.forward > 0.0 { "+z" } else { "-z" },
        "score": (score * 10.0).round() / 10.0,
        "pass": failing == 0,
        "bones": nb,
        "limbs": limb_rows,
        "spine_covered": (spine_cov * 100.0).round() / 100.0,
        "inside": (inside * 1000.0).round() / 1000.0,
        "sides_match": (symmetry * 100.0).round() / 100.0,
        "bones_along_nothing": stray,
        "findings": findings,
    });
    Verdict {
        score,
        report,
        skeleton: Skeleton { bones },
        failing,
    }
}
