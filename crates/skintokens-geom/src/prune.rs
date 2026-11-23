//! Joint pruning and conservative skin-weight transfer on GLB containers.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::geom::M4;
use crate::glb::Glb;

/// Prune `prune_nodes` (glTF node indices) from every skin and node hierarchy in `g`.
///
/// Any vertex skin weight referencing a pruned joint is conservatively reassigned
/// to that joint's nearest kept ancestor in the node hierarchy, merging duplicate
/// influences, keeping at most the top 4 per vertex, and renormalizing positive sums
/// to 1.0. Inverse bind matrices, node child links, and animation channels are
/// updated to match. Returns the removed joint node names in node order.
pub fn prune_skin_joints(g: &mut Glb, prune_nodes: &[usize]) -> Result<Vec<String>> {
    if prune_nodes.is_empty() {
        return Ok(Vec::new());
    }
    let n_nodes = g.nodes().len();
    let prune_set: BTreeSet<usize> = prune_nodes
        .iter()
        .copied()
        .filter(|&n| n < n_nodes)
        .collect();
    if prune_set.is_empty() {
        return Ok(Vec::new());
    }
    let is_pruned = |n: usize| prune_set.contains(&n);

    let parents = g.parents();
    let pruned_names: Vec<String> = prune_set
        .iter()
        .map(|&n| {
            g.nodes()[n]
                .get("name")
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("node_{n}"))
        })
        .collect();

    struct PendingPrim {
        mesh: usize,
        prim: usize,
        joints: Vec<[u16; 4]>,
        weights: Vec<[f32; 4]>,
    }
    struct PendingSkin {
        skin: usize,
        joints: Vec<usize>,
        ibms: Vec<M4>,
        skel_root: Option<usize>,
    }
    type JointWeightSet = (Vec<[u16; 4]>, Vec<[f32; 4]>);

    let mut pending_prims: Vec<PendingPrim> = Vec::new();
    let mut pending_skins: Vec<PendingSkin> = Vec::new();
    {
        let doc = g.doc()?;
        let mut updated_mesh_skin: BTreeSet<(usize, usize)> = BTreeSet::new();

        for skin in doc.skins() {
            let si = skin.index();
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
            let new_joints: Vec<usize> =
                joints.iter().copied().filter(|j| !is_pruned(*j)).collect();
            if new_joints.is_empty() {
                bail!("pruning would empty skin {si}");
            }
            let slot_of_new: BTreeMap<usize, u16> = new_joints
                .iter()
                .enumerate()
                .map(|(idx, &n)| (n, idx as u16))
                .collect();

            let kept_ancestor_slot = |start: usize| -> Option<u16> {
                let mut cur = Some(start);
                let mut steps = 0usize;
                while let Some(n) = cur {
                    if !is_pruned(n) {
                        if let Some(&slot) = slot_of_new.get(&n) {
                            return Some(slot);
                        }
                    }
                    if steps >= n_nodes {
                        break;
                    }
                    steps += 1;
                    cur = parents.get(n).copied().flatten();
                }
                None
            };

            let slot_map: Vec<Option<u16>> =
                joints.iter().map(|&j| kept_ancestor_slot(j)).collect();

            let ibms: Vec<M4> = match skin.reader(g.buffer()).read_inverse_bind_matrices() {
                Some(it) => it
                    .map(|m| std::array::from_fn(|c| std::array::from_fn(|r| m[c][r] as f64)))
                    .collect(),
                None => vec![crate::geom::IDENTITY; joints.len()],
            };
            if ibms.len() != joints.len() {
                bail!(
                    "skin {si} inverseBindMatrices count ({}) != joints count ({})",
                    ibms.len(),
                    joints.len()
                );
            }
            let kept_ibms: Vec<M4> = joints
                .iter()
                .zip(ibms)
                .filter(|(j, _)| !is_pruned(**j))
                .map(|(_, m)| m)
                .collect();

            for node in doc.nodes() {
                if node.skin().map(|s| s.index()) != Some(si) {
                    continue;
                }
                let Some(mesh) = node.mesh() else { continue };
                if !updated_mesh_skin.insert((si, mesh.index())) {
                    continue;
                }
                for (pi, prim) in mesh.primitives().enumerate() {
                    let r = prim.reader(g.buffer());
                    let mut sets: Vec<JointWeightSet> = Vec::new();
                    for set in 0..4 {
                        match (r.read_joints(set), r.read_weights(set)) {
                            (Some(j), Some(w)) => {
                                sets.push((j.into_u16().collect(), w.into_f32().collect()))
                            }
                            _ => break,
                        }
                    }
                    if sets.is_empty() {
                        continue;
                    }
                    let nv = sets[0].0.len();
                    let mut out_joints = Vec::with_capacity(nv);
                    let mut out_weights = Vec::with_capacity(nv);
                    for vi in 0..nv {
                        let mut acc: Vec<(u16, f64)> = Vec::new();
                        for (js, ws) in &sets {
                            let Some(jr) = js.get(vi) else { continue };
                            let Some(wr) = ws.get(vi) else { continue };
                            for k in 0..4 {
                                let w = wr[k] as f64;
                                let old_slot = jr[k] as usize;
                                if !w.is_finite() || w <= 0.0 {
                                    continue;
                                }
                                let Some(new_slot) = slot_map.get(old_slot).copied().flatten()
                                else {
                                    continue;
                                };
                                match acc.iter_mut().find(|(s, _)| *s == new_slot) {
                                    Some(entry) => entry.1 += w,
                                    None => acc.push((new_slot, w)),
                                }
                            }
                        }
                        acc.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                        acc.truncate(4);
                        let total: f64 = acc.iter().map(|(_, w)| *w).sum();
                        let mut jrow = [0u16; 4];
                        let mut wrow = [0.0f32; 4];
                        if total > 0.0 {
                            for (c, (slot, w)) in acc.iter().enumerate() {
                                jrow[c] = *slot;
                                wrow[c] = (*w / total) as f32;
                            }
                            let sum32: f32 = wrow.iter().sum();
                            if sum32 > 0.0 {
                                wrow[0] += 1.0 - sum32;
                            }
                        }
                        out_joints.push(jrow);
                        out_weights.push(wrow);
                    }
                    pending_prims.push(PendingPrim {
                        mesh: mesh.index(),
                        prim: pi,
                        joints: out_joints,
                        weights: out_weights,
                    });
                }
            }

            let skel_root = skin
                .skeleton()
                .map(|n| n.index())
                .filter(|&n| !is_pruned(n))
                .or_else(|| new_joints.first().copied());
            pending_skins.push(PendingSkin {
                skin: si,
                joints: new_joints,
                ibms: kept_ibms,
                skel_root,
            });
        }
    }

    let mut prim_updates: Vec<(usize, usize, usize, usize)> =
        Vec::with_capacity(pending_prims.len());
    for p in pending_prims {
        let (nja, nwa) = g.push_skin_attrs(&p.joints, &p.weights)?;
        prim_updates.push((p.mesh, p.prim, nja, nwa));
    }
    let mut skin_updates: Vec<(usize, Vec<usize>, usize, Option<usize>)> =
        Vec::with_capacity(pending_skins.len());
    for s in pending_skins {
        let new_ibm_acc = g.push_mat4s(&s.ibms)?;
        skin_updates.push((s.skin, s.joints, new_ibm_acc, s.skel_root));
    }

    for (mi, pi, nja, nwa) in prim_updates {
        let meshes = g
            .json
            .get_mut("meshes")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| anyhow!("missing meshes array"))?;
        let prim = meshes
            .get_mut(mi)
            .and_then(|m| m.get_mut("primitives"))
            .and_then(|p| p.as_array_mut())
            .and_then(|ps| ps.get_mut(pi))
            .ok_or_else(|| anyhow!("missing primitive {mi}:{pi}"))?;
        let attrs = prim
            .get_mut("attributes")
            .and_then(|a| a.as_object_mut())
            .ok_or_else(|| anyhow!("missing attributes on {mi}:{pi}"))?;
        attrs.insert("JOINTS_0".into(), json!(nja));
        attrs.insert("WEIGHTS_0".into(), json!(nwa));
        for set in 1..4 {
            attrs.remove(&format!("JOINTS_{set}"));
            attrs.remove(&format!("WEIGHTS_{set}"));
        }
    }

    for (si, new_joints, new_ibm_acc, skel_root) in skin_updates {
        let skins = g
            .json
            .get_mut("skins")
            .and_then(|v| v.as_array_mut())
            .ok_or_else(|| anyhow!("missing skins array"))?;
        let skin_obj = skins
            .get_mut(si)
            .and_then(|s| s.as_object_mut())
            .ok_or_else(|| anyhow!("missing skin {si}"))?;
        skin_obj.insert(
            "joints".into(),
            Value::Array(new_joints.into_iter().map(|j| json!(j)).collect()),
        );
        skin_obj.insert("inverseBindMatrices".into(), json!(new_ibm_acc));
        if let Some(root) = skel_root {
            skin_obj.insert("skeleton".into(), json!(root));
        } else {
            skin_obj.remove("skeleton");
        }
    }

    if let Some(nodes) = g.json.get_mut("nodes").and_then(|v| v.as_array_mut()) {
        for node in nodes.iter_mut() {
            if let Some(children) = node.get_mut("children").and_then(|v| v.as_array_mut()) {
                children.retain(|c| c.as_u64().map(|x| !is_pruned(x as usize)).unwrap_or(true));
            }
        }
    }
    if let Some(scenes) = g.json.get_mut("scenes").and_then(|v| v.as_array_mut()) {
        for sc in scenes.iter_mut() {
            if let Some(nodes) = sc.get_mut("nodes").and_then(|v| v.as_array_mut()) {
                nodes.retain(|c| c.as_u64().map(|x| !is_pruned(x as usize)).unwrap_or(true));
            }
        }
    }

    // Remove animation channels targeting pruned nodes so exported GLB remains valid.
    if let Some(anims) = g.json.get_mut("animations").and_then(|v| v.as_array_mut()) {
        for anim in anims.iter_mut() {
            if let Some(chans) = anim.get_mut("channels").and_then(|v| v.as_array_mut()) {
                chans.retain(|ch| {
                    ch.get("target")
                        .and_then(|t| t.get("node"))
                        .and_then(|n| n.as_u64())
                        .map(|n| !is_pruned(n as usize))
                        .unwrap_or(true)
                });
            }
        }
        anims.retain(|anim| {
            anim.get("channels")
                .and_then(|v| v.as_array())
                .map(|c| !c.is_empty())
                .unwrap_or(false)
        });
    }

    g.doc()?;
    Ok(pruned_names)
}
