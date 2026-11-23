//! Can It Code skeleton pruning: cut leaves stuck outside the mesh volume
//! and (when not `--class hero` / `--keep-digits`) fold dense digit/cloth fans
//! into their parent bone, transferring weights conservatively to kept ancestors.

use std::path::PathBuf;

use anyhow::{anyhow, Result};
use serde_json::json;
use skintokens_geom::geom::{self, V3};
use skintokens_geom::glb::{Frame, Glb};

use crate::{parts, Args};

fn len2(v: V3) -> f64 {
    geom::dot(v, v)
}

fn dist(a: V3, b: V3) -> f64 {
    geom::dist2(a, b).sqrt()
}

fn point_triangle_dist_sq(p: V3, a: V3, b: V3, c: V3) -> f64 {
    let ab = geom::sub(b, a);
    let ac = geom::sub(c, a);
    let ap = geom::sub(p, a);

    let d1 = geom::dot(ab, ap);
    let d2 = geom::dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return len2(ap);
    }

    let bp = geom::sub(p, b);
    let d3 = geom::dot(ab, bp);
    let d4 = geom::dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return len2(bp);
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let q = geom::add(a, geom::scale(ab, v));
        return len2(geom::sub(p, q));
    }

    let cp = geom::sub(p, c);
    let d5 = geom::dot(ab, cp);
    let d6 = geom::dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return len2(cp);
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let q = geom::add(a, geom::scale(ac, w));
        return len2(geom::sub(p, q));
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let bc = geom::sub(c, b);
        let q = geom::add(b, geom::scale(bc, w));
        return len2(geom::sub(p, q));
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let q = geom::add(a, geom::add(geom::scale(ab, v), geom::scale(ac, w)));
    len2(geom::sub(p, q))
}

fn nearest_surface_distance(verts: &[V3], tris: &[[u32; 3]], p: V3) -> f64 {
    let mut best_sq = f64::INFINITY;
    for &[ia, ib, ic] in tris {
        let (Some(&a), Some(&b), Some(&c)) = (
            verts.get(ia as usize),
            verts.get(ib as usize),
            verts.get(ic as usize),
        ) else {
            continue;
        };
        let d2 = point_triangle_dist_sq(p, a, b, c);
        if d2 < best_sq {
            best_sq = d2;
        }
    }
    best_sq.sqrt()
}

pub fn cmd_prune(a: &Args) -> Result<()> {
    let input = PathBuf::from(
        a.pos
            .first()
            .ok_or_else(|| anyhow!("prune needs IN.glb OUT.glb"))?,
    );
    let output = PathBuf::from(
        a.pos
            .get(1)
            .ok_or_else(|| anyhow!("prune needs IN.glb OUT.glb"))?,
    );
    let res: usize = a
        .flags
        .get("res")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(128);
    let klass = a
        .flags
        .get("class")
        .map(|s| s.trim().to_ascii_lowercase())
        .unwrap_or_else(|| "monster".to_string());
    let keep_digits = a.flags.contains_key("keep-digits") || klass == "hero";

    let mut g = Glb::read(&input)?;
    let ski = g.skin_info()?;
    let worlds = g.world_matrices();

    let mut verts = Vec::new();
    let mut tris = Vec::new();
    for p in g.parts(Frame::Rest)? {
        let base = verts.len() as u32;
        verts.extend_from_slice(&p.positions);
        tris.extend(
            p.triangles
                .iter()
                .map(|t| [t[0] + base, t[1] + base, t[2] + base]),
        );
    }
    if tris.is_empty() {
        anyhow::bail!("{} has no triangles", input.display());
    }

    let body = parts::Body::build(&verts, &tris, res);
    let height = (body.hi[1] - body.lo[1]).max(1e-6);
    let nj = ski.nodes.len();
    let heads: Vec<V3> = ski
        .nodes
        .iter()
        .map(|&n| geom::translation(&worlds[n]))
        .collect();
    let jparent: Vec<Option<usize>> = ski
        .parents
        .iter()
        .map(|&p| if p < 0 { None } else { Some(p as usize) })
        .collect();

    let mut keep = vec![true; nj];
    let kids_of = |k: &[bool], p: usize| -> Vec<usize> {
        (0..nj).filter(|&c| k[c] && jparent[c] == Some(p)).collect()
    };
    let subtree = |k: &[bool], root: usize| -> Vec<usize> {
        let mut out = Vec::new();
        let mut stack = vec![root];
        while let Some(i) = stack.pop() {
            out.push(i);
            for c in 0..nj {
                if k[c] && jparent[c] == Some(i) {
                    stack.push(c);
                }
            }
        }
        out
    };

    let mut fingers_folded = 0usize;
    if !keep_digits {
        let tiny = 0.03 * height;
        let hand = 0.25 * height;
        for i in 0..nj {
            if !keep[i] || jparent[i].is_none() {
                continue;
            }
            let digits: Vec<Vec<usize>> = kids_of(&keep, i)
                .into_iter()
                .map(|c| subtree(&keep, c))
                .filter(|sub| sub.iter().all(|&j| dist(heads[j], heads[i]) <= hand))
                .collect();
            if digits.len() >= 3 {
                for sub in digits {
                    for j in sub {
                        if keep[j] {
                            keep[j] = false;
                            fingers_folded += 1;
                        }
                    }
                }
            }
        }
        loop {
            let mut changed = false;
            for i in 0..nj {
                if !keep[i] {
                    continue;
                }
                for c in kids_of(&keep, i) {
                    if kids_of(&keep, c).is_empty() && dist(heads[c], heads[i]) <= tiny {
                        keep[c] = false;
                        fingers_folded += 1;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    let clearance = 0.02 * height;
    let is_outside = |p: V3| -> bool {
        !body.solid.is_inside(p) && nearest_surface_distance(&verts, &tris, p) > clearance
    };
    let mut outside_cut = 0usize;
    loop {
        let mut changed = false;
        for i in 0..nj {
            if !keep[i] || jparent[i].is_none() {
                continue;
            }
            if kids_of(&keep, i).is_empty() && is_outside(heads[i]) {
                keep[i] = false;
                outside_cut += 1;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let outside_left = (0..nj)
        .filter(|&i| keep[i] && jparent[i].is_some() && is_outside(heads[i]))
        .count();

    let prune_nodes: Vec<usize> = (0..nj)
        .filter(|&i| !keep[i])
        .map(|i| ski.nodes[i])
        .collect();
    let pruned_names = skintokens_geom::prune::prune_skin_joints(&mut g, &prune_nodes)?;
    g.write(&output)?;

    let kept_count = keep.iter().filter(|&&k| k).count();
    let report = json!({
        "format": "creature-prune/1",
        "class": klass,
        "keep_digits": keep_digits,
        "joints_before": nj,
        "joints_after": kept_count,
        "fingers_folded": fingers_folded,
        "outside_cut": outside_cut,
        "outside_left": outside_left,
        "pruned": pruned_names,
    });
    if let Some(jp) = a.flags.get("json") {
        std::fs::write(jp, serde_json::to_string_pretty(&report)?)?;
    }
    println!(
        "pruned {} joints (digits {}, outside {}; kept {}/{} joints)",
        pruned_names.len(),
        fingers_folded,
        outside_cut,
        kept_count,
        nj
    );
    Ok(())
}
