//! The three commands' work: `rig` (skeleton + weights for a mesh), `skin`
//! (new weights for the input's own skeleton) and `joints` (rigforge hints).

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use skintokens_nn::Rng;

use crate::generate::GenConfig;
use crate::geom::{self, M4, V3};
use crate::glb::{Frame, Glb};
use crate::mesh::{self, Mesh, Normalize};
use crate::model::TokenRig;
use crate::naming;
use crate::tokenizer;

/// Normalized model input for one GLB.
struct Prepared {
    mesh: Mesh,
    norm: Normalize,
    /// Normalized vertices (model frame).
    verts: Vec<V3>,
    samples: Vec<[f32; 6]>,
}

fn prepare(parts: Vec<(Vec<V3>, Vec<[u32; 3]>)>, joints: &[V3], rng: &mut Rng) -> Prepared {
    let mesh = Mesh::from_parts(&parts);
    let norm = Normalize::fit(mesh.vertices.iter().chain(joints.iter()));
    let verts: Vec<V3> = mesh.vertices.iter().map(|&v| norm.apply(v)).collect();
    let nm = Mesh { vertices: verts.clone(), faces: mesh.faces.clone(), offsets: mesh.offsets.clone() };
    let fnorm = nm.face_normals();
    let vnorm = nm.vertex_normals(&fnorm);
    let samples = mesh::sample(&verts, &nm.faces, &vnorm, &fnorm, rng);
    if let Ok(p) = std::env::var("SKINTOKENS_DUMP_SAMPLES") {
        let flat: Vec<f32> = samples.iter().flatten().copied().collect();
        let _ = candle_core::Tensor::from_vec(flat, (samples.len(), 6), &candle_core::Device::Cpu).and_then(|t| t.write_npy(p));
    }
    Prepared { mesh, norm, verts, samples }
}

pub fn weights_summary(js: &[[u16; 4]], ws: &[[f32; 4]]) -> Value {
    let _ = js;
    let max_inf = ws.iter().map(|w| w.iter().filter(|&&x| x > 1e-6).count()).max().unwrap_or(0);
    let err = ws.iter().map(|w| (w.iter().sum::<f32>() - 1.0).abs() as f64).fold(0.0, f64::max);
    json!({"verts": ws.len(), "max_influences": max_inf, "max_sum_error": err})
}

// ------------------------------------------------------------------ rig

pub struct RigResult {
    pub glb: Glb,
    pub details: Value,
    /// Humanoid core names that could not be placed (empty = fine).
    pub missing: Vec<String>,
}

/// Upstream `make_asset` tails for a generated skeleton (no stored tails):
/// as written upstream every non-root bone points away from its parent
/// with half the mean bone length (its single-child branch never fires
/// because children are listed twice), the root points up +Z; joints that
/// sit on their parent are nudged by 1% of that length.
fn bone_frames(joints: &mut [V3], parents: &[i64], rng: &mut Rng) -> Vec<[V3; 3]> {
    let n = joints.len();
    let root = parents.iter().position(|&p| p == -1).unwrap_or(0);
    let mut sum = 0.0;
    for i in 0..n {
        if parents[i] >= 0 {
            sum += geom::norm(geom::sub(joints[i], joints[parents[i] as usize]));
        }
    }
    let length = if n <= 1 { 1.0 } else { sum / ((n - 1).max(1) as f64) * 0.5 };
    for i in 0..n {
        let p = parents[i];
        if p < 0 {
            continue;
        }
        if geom::norm(geom::sub(joints[i], joints[p as usize])) <= length * 1e-2 {
            let m = length.max(1e-5);
            for a in 0..3 {
                joints[i][a] += rng.normal() * m * 1e-2;
            }
        }
    }
    let mut tails = joints.to_vec();
    for i in 0..n {
        let p = parents[i];
        if p >= 0 {
            let d = geom::sub(joints[i], joints[p as usize]);
            let l = geom::norm(d);
            let d = if l < 1e-6 { [0.0, 0.0, 1.0] } else { geom::scale(d, 1.0 / l) };
            tails[i] = geom::add(joints[i], geom::scale(d, length));
        }
    }
    tails[root] = geom::add(joints[root], [0.0, 0.0, length]);
    (0..n)
        .map(|i| {
            let d = geom::sub(tails[i], joints[i]);
            let l = geom::norm(d);
            geom::bone_rotation(if l > 0.0 { geom::scale(d, 1.0 / l) } else { [0.0, 1.0, 0.0] })
        })
        .collect()
}

/// Model-frame (Blender) rotation columns -> glTF world columns. Blender's
/// exporter converts the world axes only; bones keep their local frame
/// (Y along the bone).
fn rot_to_gltf(c: [V3; 3]) -> [V3; 3] {
    [mesh::from_model(c[0]), mesh::from_model(c[1]), mesh::from_model(c[2])]
}

/// Removes skins, animations, skinning attributes and detaches old joint
/// nodes from the scene (an already rigged input is re-rigged from scratch).
fn strip_rig(g: &mut Glb) {
    let mut joints = std::collections::HashSet::new();
    for s in g.json["skins"].as_array().into_iter().flatten() {
        for j in s["joints"].as_array().into_iter().flatten() {
            if let Some(j) = j.as_u64() {
                joints.insert(j);
            }
        }
    }
    let obj = g.json.as_object_mut().unwrap();
    obj.remove("skins");
    obj.remove("animations");
    for nd in g.json["nodes"].as_array_mut().into_iter().flatten() {
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
    for sc in g.json["scenes"].as_array_mut().into_iter().flatten() {
        if let Some(c) = sc["nodes"].as_array_mut() {
            c.retain(|x| !x.as_u64().map(|v| joints.contains(&v)).unwrap_or(false));
        }
    }
    for m in g.json["meshes"].as_array_mut().into_iter().flatten() {
        for p in m["primitives"].as_array_mut().into_iter().flatten() {
            if let Some(a) = p["attributes"].as_object_mut() {
                a.retain(|k, _| !k.starts_with("JOINTS_") && !k.starts_with("WEIGHTS_"));
            }
        }
    }
}

pub fn rig(input: &Glb, model: &TokenRig, cfg: &GenConfig, seed: u64, humanoid: bool) -> Result<RigResult> {
    let parts = input.parts(Frame::World)?;
    let mut rng = Rng::new(seed);
    let model_parts: Vec<(Vec<V3>, Vec<[u32; 3]>)> =
        parts.iter().map(|p| (p.positions.iter().map(|&v| mesh::to_model(v)).collect(), p.triangles.clone())).collect();
    let prep = prepare(model_parts, &[], &mut rng);
    let start = vec![tokenizer::BOS, tokenizer::CLS_ARTICULATION];
    let out = model.run(&prep.samples, &start, cfg, rng.next_u64())?;
    let sk = &out.skeleton;
    let skin = mesh::transfer_skin(&prep.samples, &out.skin, &prep.verts);
    let (js, ws) = mesh::top4(&skin);

    // skeleton in the input's world space
    let mut heads_model: Vec<V3> = sk.joints.iter().map(|j| prep.norm.invert([j[0] as f64, j[1] as f64, j[2] as f64])).collect();
    let rots = bone_frames(&mut heads_model, &sk.parents, &mut rng);
    let worlds: Vec<M4> = heads_model.iter().zip(&rots).map(|(&h, &r)| geom::from_rt(rot_to_gltf(r), mesh::from_model(h))).collect();

    let mut g = Glb { json: input.json.clone(), bin: input.bin.clone() };
    strip_rig(&mut g);
    let node_worlds = g.world_matrices();
    // names: humanoid naming works on heads in the first mesh's bind space
    let mut names: Vec<String> = sk.names.clone();
    let mut missing = vec![];
    let mut details = json!({"joints_raw": names.len()});
    if humanoid {
        let m0 = geom::inverse(&node_worlds[parts[0].node]);
        let heads: Vec<[f64; 3]> = worlds.iter().map(|w| geom::apply(&m0, geom::translation(w))).collect();
        let parents: Vec<Option<usize>> = sk.parents.iter().map(|&p| if p < 0 { None } else { Some(p as usize) }).collect();
        let (mapping, miss) = naming::name_humanoid(&names, &parents, &heads);
        if !miss.is_empty() {
            details["missing"] = json!(miss);
            missing = miss;
        } else {
            names = names.iter().map(|n| mapping.get(n).cloned().unwrap_or_else(|| n.clone())).collect();
            details["naming"] = json!("mixamo (structural)");
        }
    }

    // joint nodes under a new armature node at the scene root
    let base = g.nodes().len();
    let arm = base + names.len();
    let mut kids: Vec<Vec<usize>> = vec![vec![]; names.len()];
    for (i, &p) in sk.parents.iter().enumerate() {
        if p >= 0 {
            kids[p as usize].push(i);
        }
    }
    let mut new_nodes = vec![];
    for i in 0..names.len() {
        let local = if sk.parents[i] >= 0 { geom::mul(&geom::inverse(&worlds[sk.parents[i] as usize]), &worlds[i]) } else { worlds[i] };
        let q = geom::quat_from_cols(geom::cols(&local));
        let t = geom::translation(&local);
        let mut nd = json!({"name": names[i], "translation": t, "rotation": q});
        if !kids[i].is_empty() {
            nd["children"] = json!(kids[i].iter().map(|&k| base + k).collect::<Vec<_>>());
        }
        new_nodes.push(nd);
    }
    let roots: Vec<usize> = (0..names.len()).filter(|&i| sk.parents[i] < 0).map(|i| base + i).collect();
    new_nodes.push(json!({"name": "Armature", "children": roots}));
    let nodes = g.json["nodes"].as_array_mut().ok_or_else(|| anyhow!("no nodes"))?;
    nodes.extend(new_nodes);
    let si = g.json["scene"].as_u64().unwrap_or(0) as usize;
    if let Some(sn) = g.json["scenes"].get_mut(si).and_then(|s| s["nodes"].as_array_mut()) {
        sn.push(json!(arm));
    } else {
        bail!("GLB has no scene to add the armature to");
    }
    // one skin per distinct mesh-node world matrix (inverse bind = joint^-1 * mesh)
    let mut skins: Vec<(M4, usize)> = vec![];
    let joint_nodes: Vec<usize> = (0..names.len()).map(|i| base + i).collect();
    let mut node_skin: HashMap<usize, usize> = HashMap::new();
    for p in &parts {
        if node_skin.contains_key(&p.node) {
            continue;
        }
        let mw = node_worlds[p.node];
        let k = match skins.iter().position(|(m, _)| (0..4).all(|c| (0..4).all(|r| (m[c][r] - mw[c][r]).abs() < 1e-9))) {
            Some(k) => k,
            None => {
                let ibm: Vec<M4> = worlds.iter().map(|w| geom::mul(&geom::inverse(w), &mw)).collect();
                let acc = g.push_mat4s(&ibm);
                skins.push((mw, acc));
                skins.len() - 1
            }
        };
        node_skin.insert(p.node, k);
    }
    g.json["skins"] = json!(skins
        .iter()
        .map(|(_, acc)| json!({"joints": joint_nodes, "inverseBindMatrices": acc, "skeleton": roots[0]}))
        .collect::<Vec<_>>());
    for (&n, &k) in &node_skin {
        g.json["nodes"][n]["skin"] = json!(k);
    }
    for (pi, p) in parts.iter().enumerate() {
        let (s, e) = (prep.mesh.offsets[pi], prep.mesh.offsets[pi] + p.positions.len());
        let (ja, wa) = g.push_skin_attrs(&js[s..e], &ws[s..e]);
        let attrs = &mut g.json["meshes"][p.mesh]["primitives"][p.prim]["attributes"];
        attrs["JOINTS_0"] = json!(ja);
        attrs["WEIGHTS_0"] = json!(wa);
    }
    g.doc()?;
    details["weights"] = weights_summary(&js, &ws);
    details["tokens"] = json!(out.ids.len());
    Ok(RigResult { glb: g, details, missing })
}

// ------------------------------------------------------------------ skin

/// Twist/helper bones (motionforge's DEF-*_twist.*): {helper node: driver
/// node}. Marked by node extras.hll_helper.driver, else by the name
/// `<driver>_twist.<side>`.
pub fn helper_joints(g: &Glb, joint_nodes: &[usize]) -> HashMap<usize, usize> {
    let names: Vec<String> = joint_nodes.iter().map(|&n| g.nodes()[n]["name"].as_str().unwrap_or("").to_string()).collect();
    let mut out = HashMap::new();
    for (k, &n) in joint_nodes.iter().enumerate() {
        let mut driver = g.nodes()[n]["extras"]["hll_helper"]["driver"].as_str().map(String::from);
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
fn set_accessor(g: &mut Glb, acc: usize, rows: &[[f64; 4]]) -> Result<()> {
    let a = g.json["accessors"][acc].clone();
    if a.get("sparse").is_some() {
        bail!("sparse skin accessors are not supported");
    }
    let bv = &g.json["bufferViews"][a["bufferView"].as_u64().ok_or_else(|| anyhow!("accessor without bufferView"))? as usize];
    let ct = a["componentType"].as_u64().unwrap_or(0);
    let (size, max) = match ct {
        5121 => (1usize, 255.0),
        5123 => (2, 65535.0),
        5125 => (4, 4294967295.0),
        5126 => (4, 0.0),
        _ => bail!("unsupported component type {ct}"),
    };
    let nc = 4usize;
    let start = bv["byteOffset"].as_u64().unwrap_or(0) as usize + a["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = bv["byteStride"].as_u64().map(|s| s as usize).unwrap_or(size * nc);
    let normalized = a["normalized"].as_bool().unwrap_or(false);
    if rows.len() != a["count"].as_u64().unwrap_or(0) as usize {
        bail!("accessor count mismatch");
    }
    let mut mn = [f64::INFINITY; 4];
    let mut mx = [f64::NEG_INFINITY; 4];
    for (k, r) in rows.iter().enumerate() {
        for c in 0..nc {
            let o = start + k * stride + c * size;
            let v = if normalized && ct != 5126 { (r[c] * max).round() } else { r[c] };
            let bytes: Vec<u8> = match ct {
                5121 => vec![v.clamp(0.0, 255.0) as u8],
                5123 => (v.clamp(0.0, 65535.0) as u16).to_le_bytes().to_vec(),
                5125 => (v.max(0.0) as u32).to_le_bytes().to_vec(),
                _ => (v as f32).to_le_bytes().to_vec(),
            };
            g.bin.get_mut(o..o + size).ok_or_else(|| anyhow!("accessor out of range"))?.copy_from_slice(&bytes);
            mn[c] = mn[c].min(v);
            mx[c] = mx[c].max(v);
        }
    }
    if a.get("min").is_some() || a.get("max").is_some() {
        g.json["accessors"][acc]["min"] = json!(mn);
        g.json["accessors"][acc]["max"] = json!(mx);
    }
    Ok(())
}

pub struct SkinResult {
    pub glb: Glb,
    pub details: Value,
}

pub fn skin(input: &Glb, model: &TokenRig, cfg: &GenConfig, seed: u64) -> Result<SkinResult> {
    let info = input.skin_info()?;
    let skin_joints: Vec<usize> = input.json["skins"][0]["joints"].as_array().into_iter().flatten().filter_map(|v| v.as_u64().map(|x| x as usize)).collect();
    let helpers = helper_joints(input, &skin_joints);
    let parts = input.parts(Frame::SkinBind)?;
    // joints the model sees: no helpers (their weight counts for the driver)
    let visible: Vec<usize> = (0..info.nodes.len()).filter(|&i| !helpers.contains_key(&info.nodes[i])).collect();
    let vis_parents = mesh::reparent(&info.parents, &visible);
    let slot_to_vis: HashMap<usize, usize> = visible.iter().enumerate().map(|(vi, &i)| (info.skin_slot[i], vi)).collect();
    let node_to_slot: HashMap<usize, usize> = info.nodes.iter().zip(&info.skin_slot).map(|(&n, &s)| (n, s)).collect();
    let mut weight_sum = vec![0f64; visible.len()];
    for p in &parts {
        if let (Some(j), Some(w)) = (&p.joints, &p.weights) {
            for (jr, wr) in j.iter().zip(w) {
                for c in 0..4 {
                    let mut slot = jr[c] as usize;
                    if let Some(n) = skin_joints.get(slot) {
                        if let Some(d) = helpers.get(n) {
                            slot = node_to_slot[d];
                        }
                    }
                    if let Some(&vi) = slot_to_vis.get(&slot) {
                        weight_sum[vi] += wr[c] as f64;
                    }
                }
            }
        }
    }
    let has_skin: Vec<bool> = weight_sum.iter().map(|&s| s > 1e-6).collect();
    let kept_vis = mesh::trim(&vis_parents, &has_skin);
    let kept: Vec<usize> = kept_vis.iter().map(|&v| visible[v]).collect();
    let parents = mesh::reparent(&vis_parents, &kept_vis);
    if parents.iter().filter(|&&p| p < 0).count() != 1 || parents[0] != -1 {
        bail!("the skeleton must have exactly one root joint");
    }
    let arm = |v: V3| mesh::to_model(geom::apply(&info.to_armature, v).map(|x| x as f32 as f64));
    let joints_model: Vec<V3> = kept.iter().map(|&i| arm(info.heads[i])).collect();
    let mut rng = Rng::new(seed);
    let model_parts: Vec<(Vec<V3>, Vec<[u32; 3]>)> = parts.iter().map(|p| (p.positions.iter().map(|&v| arm(v)).collect(), p.triangles.clone())).collect();
    let prep = prepare(model_parts, &joints_model, &mut rng);
    let jn: Vec<[f64; 3]> = joints_model.iter().map(|&j| prep.norm.apply(j)).collect();
    let mut start = tokenizer::tokenize(&jn, &parents, tokenizer::CLS_ARTICULATION);
    // developer hooks for parity experiments
    if let Ok(p) = std::env::var("SKINTOKENS_DUMP_TOKENS") {
        let t: Vec<i64> = start.iter().map(|&x| x as i64).collect();
        let _ = candle_core::Tensor::from_vec(t.clone(), t.len(), &candle_core::Device::Cpu).and_then(|t| t.write_npy(p));
    }
    if let Ok(p) = std::env::var("SKINTOKENS_START_TOKENS") {
        let t = candle_core::Tensor::read_npy(p)?.to_dtype(candle_core::DType::I64)?.to_vec1::<i64>()?;
        start = t.into_iter().map(|x| x as u32).collect();
    }
    let out = model.run(&prep.samples, &start, cfg, rng.next_u64())?;
    if out.skin.first().map(|r| r.len()) != Some(kept.len()) {
        bail!("model returned {} joints for {} given", out.skin.first().map(|r| r.len()).unwrap_or(0), kept.len());
    }
    let skin = mesh::transfer_skin(&prep.samples, &out.skin, &prep.verts);
    let (js, ws) = mesh::top4(&skin);
    let slots: Vec<usize> = kept.iter().map(|&i| info.skin_slot[i]).collect();
    let mut g = Glb { json: input.json.clone(), bin: input.bin.clone() };
    for (pi, p) in parts.iter().enumerate() {
        let (s, e) = (prep.mesh.offsets[pi], prep.mesh.offsets[pi] + p.positions.len());
        let attrs = &input.json["meshes"][p.mesh]["primitives"][p.prim]["attributes"];
        let (ja, wa) = (attrs["JOINTS_0"].as_u64(), attrs["WEIGHTS_0"].as_u64());
        let (Some(ja), Some(wa)) = (ja, wa) else { bail!("skinned primitive without JOINTS_0/WEIGHTS_0") };
        let jrows: Vec<[f64; 4]> = js[s..e]
            .iter()
            .zip(&ws[s..e])
            .map(|(j, w)| {
                let mut r = [0.0; 4];
                for c in 0..4 {
                    r[c] = if w[c] > 0.0 { slots[j[c] as usize] as f64 } else { 0.0 };
                }
                r
            })
            .collect();
        let wrows: Vec<[f64; 4]> = ws[s..e].iter().map(|w| w.map(|x| x as f64)).collect();
        set_accessor(&mut g, ja as usize, &jrows)?;
        set_accessor(&mut g, wa as usize, &wrows)?;
    }
    g.doc()?;
    let details = json!({
        "joints_model": kept.len(),
        "joints_trimmed": visible.len() - kept.len(),
        "helpers_unweighted": helpers.len(),
        "weights": weights_summary(&js, &ws),
        "tokens": out.ids.len(),
    });
    Ok(SkinResult { glb: g, details })
}

// ---------------------------------------------------------------- joints

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
        .unwrap_or_else(|| vec![[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]; joints.len()]);
    let parent_of = g.parents();
    let index: HashMap<usize, usize> = joints.iter().enumerate().map(|(k, &j)| (j, k)).collect();
    let parents: Vec<Option<usize>> = joints
        .iter()
        .map(|&j| {
            let mut p = parent_of[j];
            while let Some(x) = p {
                if let Some(&k) = index.get(&x) {
                    return Some(k);
                }
                p = parent_of[x];
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
    let names: Vec<String> = joints.iter().map(|&j| g.nodes()[j]["name"].as_str().map(String::from).unwrap_or(format!("joint_{j}"))).collect();
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
    let mut kids: HashMap<usize, Vec<usize>> = HashMap::new();
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
