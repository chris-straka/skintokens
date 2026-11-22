//! Pure-MIT GLB skeleton/skin editing and `skintokens-joints/1` helpers.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::geom::{self, M4, V3};
use crate::glb::{Glb, Part};

/// Removes skins, animations, skinning attributes and detaches old joint
/// nodes from the scene (an already rigged input is re-rigged from scratch).
/// A node that is not a joint but hangs under one (eyes, a helmet, a weapon
/// parented to a bone) is moved up to its nearest ancestor that is not a
/// joint, or to the scene root, where it keeps its place in the world.
pub fn strip_rig(g: &mut Glb) -> Result<()> {
    let mut joints = BTreeSet::new();
    for s in g
        .json
        .get("skins")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        for j in s
            .get("joints")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
        {
            if let Some(j) = j.as_u64() {
                joints.insert(j);
            }
        }
    }
    // nodes that would be cut off with their joint: (node, new parent or
    // None for the scene root, its matrix under that parent)
    let worlds = g.world_matrices();
    let parents = g.parents();
    let mut rehome: Vec<(usize, Option<usize>, M4)> = vec![];
    for (i, p) in parents.iter().enumerate() {
        let Some(p) = *p else { continue };
        if joints.contains(&(i as u64)) || !joints.contains(&(p as u64)) {
            continue;
        }
        let mut up = parents.get(p).copied().flatten();
        let mut steps = 0usize;
        while let Some(a) = up {
            if steps >= parents.len() {
                up = None;
                break;
            }
            steps += 1;
            if !joints.contains(&(a as u64)) {
                break;
            }
            up = parents.get(a).copied().flatten();
        }
        let local = match up {
            Some(a) => geom::mul(&geom::inverse(&worlds[a]), &worlds[i]),
            None => worlds[i],
        };
        rehome.push((i, up, local));
    }
    let sc = g.json.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let obj = g
        .json
        .as_object_mut()
        .ok_or_else(|| anyhow!("GLB JSON root is not an object"))?;
    obj.remove("skins");
    obj.remove("animations");
    if let Some(nodes) = obj.get_mut("nodes").and_then(|v| v.as_array_mut()) {
        for nd in nodes.iter_mut() {
            if let Some(o) = nd.as_object_mut() {
                o.remove("skin");
                if let Some(c) = o.get_mut("children").and_then(|c| c.as_array_mut()) {
                    c.retain(|x| !x.as_u64().map(|v| joints.contains(&v)).unwrap_or(false));
                    if c.is_empty() {
                        o.remove("children");
                    }
                }
            }
        }
    }
    if let Some(scenes) = obj.get_mut("scenes").and_then(|v| v.as_array_mut()) {
        for scene in scenes.iter_mut() {
            if let Some(c) = scene.get_mut("nodes").and_then(|v| v.as_array_mut()) {
                c.retain(|x| !x.as_u64().map(|v| joints.contains(&v)).unwrap_or(false));
            }
        }
    }
    let moved: BTreeSet<u64> = rehome.iter().map(|r| r.0 as u64).collect();
    if let Some(nodes) = obj.get_mut("nodes").and_then(|v| v.as_array_mut()) {
        for nd in nodes.iter_mut() {
            if let Some(c) = nd.get_mut("children").and_then(|c| c.as_array_mut()) {
                c.retain(|x| !x.as_u64().map(|v| moved.contains(&v)).unwrap_or(false));
            }
        }
    }
    for (i, up, local) in rehome {
        let flat: Vec<f64> = local.iter().flat_map(|c| c.iter().copied()).collect();
        if let Some(o) = obj
            .get_mut("nodes")
            .and_then(|v| v.as_array_mut())
            .and_then(|n| n.get_mut(i))
            .and_then(|v| v.as_object_mut())
        {
            for k in ["translation", "rotation", "scale"] {
                o.remove(k);
            }
            o.insert("matrix".into(), json!(flat));
        }
        match up {
            Some(a) => {
                let node_obj = obj
                    .get_mut("nodes")
                    .and_then(|v| v.as_array_mut())
                    .and_then(|n| n.get_mut(a))
                    .and_then(|v| v.as_object_mut())
                    .ok_or_else(|| anyhow!("missing rehome ancestor node {a}"))?;
                let c = node_obj
                    .entry("children")
                    .or_insert_with(|| json!([]))
                    .as_array_mut()
                    .ok_or_else(|| anyhow!("node {a} 'children' is not an array"))?;
                c.push(json!(i));
            }
            None => {
                if let Some(c) = obj
                    .get_mut("scenes")
                    .and_then(|v| v.as_array_mut())
                    .and_then(|s| s.get_mut(sc))
                    .and_then(|s| s.as_object_mut())
                    .and_then(|s| s.get_mut("nodes"))
                    .and_then(|v| v.as_array_mut())
                {
                    c.push(json!(i));
                }
            }
        }
    }
    if let Some(meshes) = obj.get_mut("meshes").and_then(|v| v.as_array_mut()) {
        for m in meshes.iter_mut() {
            if let Some(prims) = m.get_mut("primitives").and_then(|v| v.as_array_mut()) {
                for p in prims.iter_mut() {
                    if let Some(a) = p.get_mut("attributes").and_then(|v| v.as_object_mut()) {
                        a.retain(|k, _| !k.starts_with("JOINTS_") && !k.starts_with("WEIGHTS_"));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Adds a skeleton and its skin to `g` (any old rig already stripped): the
/// joints as nodes under a new `Armature` node at the scene root, one skin
/// per distinct mesh placement, and JOINTS_0/WEIGHTS_0 on every primitive.
/// `worlds` are the joints' rest matrices in glTF world space, `parents`
/// -1 for a root; `js`/`ws` run over the vertices of `parts` in order.
pub fn attach_skeleton(
    g: &mut Glb,
    parts: &[Part],
    names: &[String],
    parents: &[i64],
    worlds: &[M4],
    js: &[[u16; 4]],
    ws: &[[f32; 4]],
) -> Result<()> {
    if names.len() != parents.len() || names.len() != worlds.len() {
        bail!(
            "skeleton slice length mismatch: names {}, parents {}, worlds {}",
            names.len(),
            parents.len(),
            worlds.len()
        );
    }
    if js.len() != ws.len() {
        bail!(
            "skin slice length mismatch: joints {} vs weights {}",
            js.len(),
            ws.len()
        );
    }
    let needed: usize = parts.iter().map(|p| p.positions.len()).sum();
    if needed > js.len() {
        bail!(
            "skin attributes too short: need {needed} vertices, got {}",
            js.len()
        );
    }
    let node_worlds = g.world_matrices();
    // joint nodes under a new armature node at the scene root
    let base = g.nodes().len();
    let arm = base + names.len();
    let mut kids: Vec<Vec<usize>> = vec![vec![]; names.len()];
    for (i, &p) in parents.iter().enumerate() {
        if p >= 0 {
            let pi = p as usize;
            if pi >= names.len() {
                bail!("parent index {pi} out of range for {} joints", names.len());
            }
            kids[pi].push(i);
        }
    }
    let mut new_nodes = vec![];
    for i in 0..names.len() {
        let local = if parents[i] >= 0 {
            geom::mul(&geom::inverse(&worlds[parents[i] as usize]), &worlds[i])
        } else {
            worlds[i]
        };
        let q = geom::quat_from_cols(geom::cols(&local));
        let t = geom::translation(&local);
        let mut nd = json!({"name": names[i], "translation": t, "rotation": q});
        if !kids[i].is_empty() {
            nd["children"] = json!(kids[i].iter().map(|&k| base + k).collect::<Vec<_>>());
        }
        new_nodes.push(nd);
    }
    let roots: Vec<usize> = (0..names.len())
        .filter(|&i| parents[i] < 0)
        .map(|i| base + i)
        .collect();
    let first_root = *roots
        .first()
        .ok_or_else(|| anyhow!("skeleton has no root joint (-1 parent)"))?;
    new_nodes.push(json!({"name": "Armature", "children": roots}));

    let si = g.json.get("scene").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if g.json.get("nodes").and_then(|v| v.as_array()).is_none() {
        bail!("no nodes");
    }
    if g.json
        .get("scenes")
        .and_then(|v| v.get(si))
        .and_then(|s| s.get("nodes"))
        .and_then(|v| v.as_array())
        .is_none()
    {
        bail!("GLB has no scene to add the armature to");
    }

    // one skin per distinct mesh-node world matrix (inverse bind = joint^-1 * mesh)
    let mut skins: Vec<(M4, usize)> = vec![];
    let joint_nodes: Vec<usize> = (0..names.len()).map(|i| base + i).collect();
    let mut node_skin: BTreeMap<usize, usize> = BTreeMap::new();
    for p in parts {
        if p.node >= node_worlds.len() {
            bail!("part node {} out of range", p.node);
        }
        if node_skin.contains_key(&p.node) {
            continue;
        }
        let mw = node_worlds[p.node];
        let k = match skins
            .iter()
            .position(|(m, _)| (0..4).all(|c| (0..4).all(|r| (m[c][r] - mw[c][r]).abs() < 1e-9)))
        {
            Some(k) => k,
            None => {
                let ibm: Vec<M4> = worlds
                    .iter()
                    .map(|w| geom::mul(&geom::inverse(w), &mw))
                    .collect();
                let acc = g.push_mat4s(&ibm)?;
                skins.push((mw, acc));
                skins.len() - 1
            }
        };
        node_skin.insert(p.node, k);
    }

    let nodes = g
        .json
        .get_mut("nodes")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| anyhow!("no nodes"))?;
    nodes.extend(new_nodes);
    let sn = g
        .json
        .get_mut("scenes")
        .and_then(|s| s.get_mut(si))
        .and_then(|s| s.get_mut("nodes"))
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| anyhow!("GLB has no scene to add the armature to"))?;
    sn.push(json!(arm));

    g.json["skins"] = json!(skins
        .iter()
        .map(|(_, acc)| json!({"joints": joint_nodes, "inverseBindMatrices": acc, "skeleton": first_root}))
        .collect::<Vec<_>>());
    for (&n, &k) in &node_skin {
        let node_obj = g
            .json
            .get_mut("nodes")
            .and_then(|v| v.get_mut(n))
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| anyhow!("missing mesh node {n}"))?;
        node_obj.insert("skin".into(), json!(k));
    }
    let mut offset = 0;
    for p in parts.iter() {
        let (s, e) = (offset, offset + p.positions.len());
        offset = e;
        let (ja, wa) = g.push_skin_attrs(&js[s..e], &ws[s..e])?;
        let attrs = g
            .json
            .get_mut("meshes")
            .and_then(|m| m.get_mut(p.mesh))
            .and_then(|m| m.get_mut("primitives"))
            .and_then(|pr| pr.get_mut(p.prim))
            .and_then(|pr| pr.get_mut("attributes"))
            .and_then(|a| a.as_object_mut())
            .ok_or_else(|| {
                anyhow!(
                    "missing primitive attributes on mesh {} prim {}",
                    p.mesh,
                    p.prim
                )
            })?;
        attrs.insert("JOINTS_0".into(), json!(ja));
        attrs.insert("WEIGHTS_0".into(), json!(wa));
    }
    Ok(())
}

/// Twist/helper bones (motionforge's DEF-*_twist.*): {helper node: driver
/// node}. Marked by node extras.helper_bone.driver, else by the name
/// `<driver>_twist.<side>`.
pub fn helper_joints(g: &Glb, joint_nodes: &[usize]) -> BTreeMap<usize, usize> {
    let nodes = g.nodes();
    let names: Vec<String> = joint_nodes
        .iter()
        .map(|&n| {
            nodes
                .get(n)
                .and_then(|nd| nd.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()
        })
        .collect();
    let mut out = BTreeMap::new();
    for (k, &n) in joint_nodes.iter().enumerate() {
        let mut driver = nodes
            .get(n)
            .and_then(|nd| nd.get("extras"))
            .and_then(|e| e.get("helper_bone"))
            .and_then(|h| h.get("driver"))
            .and_then(|v| v.as_str())
            .map(String::from);
        if driver.is_none() {
            if let Some((base, side)) = names[k].rsplit_once('.') {
                if (side == "L" || side == "R") && base.ends_with("_twist") {
                    driver = Some(format!("{}.{side}", &base[..base.len() - "_twist".len()]));
                }
            }
        }
        if let Some(d) = driver {
            if let Some(di) = names.iter().position(|x| *x == d) {
                if di != k {
                    out.insert(n, joint_nodes[di]);
                }
            }
        }
    }
    out
}

/// Writes rows into an existing accessor in place (its type and layout
/// kept), as the previous wrapper did.
pub fn set_accessor(g: &mut Glb, acc: usize, rows: &[[f64; 4]]) -> Result<()> {
    let a = g
        .json
        .get("accessors")
        .and_then(|v| v.get(acc))
        .ok_or_else(|| anyhow!("missing accessor {acc}"))?
        .clone();
    if a.get("sparse").is_some() {
        bail!("sparse skin accessors are not supported");
    }
    let bvi = a
        .get("bufferView")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| anyhow!("accessor without bufferView"))? as usize;
    let bv = g
        .json
        .get("bufferViews")
        .and_then(|v| v.get(bvi))
        .ok_or_else(|| anyhow!("missing bufferView {bvi}"))?
        .clone();
    let ct = a.get("componentType").and_then(|v| v.as_u64()).unwrap_or(0);
    let (size, max) = match ct {
        5121 => (1usize, 255.0),
        5123 => (2, 65535.0),
        5125 => (4, 4294967295.0),
        5126 => (4, 0.0),
        _ => bail!("unsupported component type {ct}"),
    };
    let nc = 4usize;
    let bv_off = bv.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let a_off = a.get("byteOffset").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let start = bv_off
        .checked_add(a_off)
        .ok_or_else(|| anyhow!("accessor offset overflow"))?;
    let stride = bv
        .get("byteStride")
        .and_then(|v| v.as_u64())
        .map(|s| s as usize)
        .unwrap_or(size * nc);
    let normalized = a
        .get("normalized")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let count = a.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    if rows.len() != count {
        bail!("accessor count mismatch");
    }
    let mut mn = [f64::INFINITY; 4];
    let mut mx = [f64::NEG_INFINITY; 4];
    for (k, r) in rows.iter().enumerate() {
        for c in 0..nc {
            let o = start
                .checked_add(
                    k.checked_mul(stride)
                        .ok_or_else(|| anyhow!("offset overflow"))?,
                )
                .and_then(|x| x.checked_add(c * size))
                .ok_or_else(|| anyhow!("offset overflow"))?;
            let v = if normalized && ct != 5126 {
                (r[c] * max).round()
            } else {
                r[c]
            };
            let bytes: Vec<u8> = match ct {
                5121 => vec![v.clamp(0.0, 255.0) as u8],
                5123 => (v.clamp(0.0, 65535.0) as u16).to_le_bytes().to_vec(),
                5125 => (v.max(0.0) as u32).to_le_bytes().to_vec(),
                _ => (v as f32).to_le_bytes().to_vec(),
            };
            let end = o
                .checked_add(size)
                .ok_or_else(|| anyhow!("accessor end overflow"))?;
            g.bin
                .get_mut(o..end)
                .ok_or_else(|| anyhow!("accessor out of range"))?
                .copy_from_slice(&bytes);
            mn[c] = mn[c].min(v);
            mx[c] = mx[c].max(v);
        }
    }
    if a.get("min").is_some() || a.get("max").is_some() {
        let acc_obj = g
            .json
            .get_mut("accessors")
            .and_then(|v| v.get_mut(acc))
            .and_then(|v| v.as_object_mut())
            .ok_or_else(|| anyhow!("missing accessor object {acc}"))?;
        acc_obj.insert("min".into(), json!(mn));
        acc_obj.insert("max".into(), json!(mx));
    }
    Ok(())
}

/// skintokens-joints/1 (same schema as unirig-joints/1): first skin, joints
/// in skin order, heads from the inverse bind matrices, normalizer over
/// every primitive's POSITION as stored.
pub fn joints_doc(g: &Glb, subject: Option<&str>, source: &str) -> Result<Value> {
    let doc = g.doc()?;
    let skin = doc.skins().next().ok_or_else(|| anyhow!("no skin"))?;
    let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
    let ibm: Vec<[[f32; 4]; 4]> = skin
        .reader(g.buffer())
        .read_inverse_bind_matrices()
        .map(|m| m.collect())
        .unwrap_or_else(|| {
            vec![
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ];
                joints.len()
            ]
        });
    let parent_of = g.parents();
    let index: BTreeMap<usize, usize> = joints.iter().enumerate().map(|(k, &j)| (j, k)).collect();
    let parents: Vec<Option<usize>> = joints
        .iter()
        .map(|&j| {
            let mut p = parent_of.get(j).copied().flatten();
            let mut steps = 0usize;
            while let Some(x) = p {
                if steps >= parent_of.len() {
                    break;
                }
                steps += 1;
                if let Some(&k) = index.get(&x) {
                    return Some(k);
                }
                p = parent_of.get(x).copied().flatten();
            }
            None
        })
        .collect();
    let heads: Vec<V3> = ibm
        .iter()
        .map(|m| {
            let m: M4 = std::array::from_fn(|c| std::array::from_fn(|r| m[c][r] as f64));
            geom::translation(&geom::inverse(&m))
        })
        .collect();
    let nodes = g.nodes();
    let names: Vec<String> = joints
        .iter()
        .map(|&j| {
            nodes
                .get(j)
                .and_then(|nd| nd.get("name"))
                .and_then(|v| v.as_str())
                .map(String::from)
                .unwrap_or_else(|| format!("joint_{j}"))
        })
        .collect();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for m in doc.meshes() {
        for p in m.primitives() {
            if let Some(ps) = p.reader(g.buffer()).read_positions() {
                for v in ps {
                    for a in 0..3 {
                        lo[a] = lo[a].min(v[a] as f64);
                        hi[a] = hi[a].max(v[a] as f64);
                    }
                }
            }
        }
    }
    if !lo[0].is_finite() {
        bail!("no mesh positions");
    }
    let center = geom::scale(geom::add(lo, hi), 0.5);
    let ext = geom::sub(hi, lo);
    let s = ext[0].max(ext[1]).max(ext[2]) / 2.0;
    let scale = if s != 0.0 { s } else { 1.0 };
    let mut kids: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, p) in parents.iter().enumerate() {
        if let Some(p) = p {
            kids.entry(*p).or_default().push(i);
        }
    }
    let js: Vec<Value> = (0..joints.len())
        .map(|i| {
            let k = kids.get(&i).cloned().unwrap_or_default();
            json!({
                "name": names[i],
                "parent": parents[i],
                "world": heads[i],
                "normalized": geom::scale(geom::sub(heads[i], center), 1.0 / scale),
                "tail_world": if k.len() == 1 { json!(heads[k[0]]) } else { Value::Null },
                "confidence": Value::Null,
            })
        })
        .collect();
    Ok(json!({
        "format": "skintokens-joints/1",
        "subject": subject,
        "seed": Value::Null,
        "source": source,
        "frame": "source mesh frame as stored (glTF, Y-up); positions in source units",
        "normalizer": {"center": center, "scale": scale,
                       "formula": "(world - center) / scale over the mesh bbox; longest extent maps to [-1, 1]"},
        "joints": js,
    }))
}
