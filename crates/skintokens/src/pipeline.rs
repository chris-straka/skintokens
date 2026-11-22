//! The three commands' work: `rig` (skeleton + weights for a mesh), `skin`
//! (new weights for the input's own skeleton) and `joints` (rigforge hints).

use std::collections::HashMap;

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use skintokens_nn::Rng;

use crate::generate::GenConfig;
use crate::geom::{self, M4, V3};
use crate::glb::{Frame, Glb};
use crate::mesh::{self, Mesh, Normalize, Transfer};
use crate::model::{self, TokenRig};
use crate::naming;
use crate::tokenizer;

pub use skintokens_geom::rig_edit::{
    attach_skeleton, helper_joints, joints_doc, set_accessor, strip_rig,
};

/// Normalized model input for one GLB.
struct Prepared {
    mesh: Mesh,
    norm: Normalize,
    /// Normalized vertices (model frame).
    verts: Vec<V3>,
    samples: Vec<[f32; 6]>,
    /// Face each sample lies on.
    sample_faces: Vec<u32>,
}

fn prepare(parts: Vec<(Vec<V3>, Vec<[u32; 3]>)>, joints: &[V3], rng: &mut Rng) -> Prepared {
    let mesh = Mesh::from_parts(&parts);
    let norm = Normalize::fit(mesh.vertices.iter().chain(joints.iter()));
    let verts: Vec<V3> = mesh.vertices.iter().map(|&v| norm.apply(v)).collect();
    let nm = Mesh {
        vertices: verts.clone(),
        faces: mesh.faces.clone(),
        offsets: mesh.offsets.clone(),
    };
    let fnorm = nm.face_normals();
    let vnorm = nm.vertex_normals(&fnorm);
    let (samples, sample_faces) = mesh::sample_with_faces(&verts, &nm.faces, &vnorm, &fnorm, rng);
    if let Ok(p) = std::env::var("SKINTOKENS_DUMP_SAMPLES") {
        let flat: Vec<f32> = samples.iter().flatten().copied().collect();
        let _ = candle_core::Tensor::from_vec(flat, (samples.len(), 6), &candle_core::Device::Cpu)
            .and_then(|t| t.write_npy(p));
    }
    Prepared {
        mesh,
        norm,
        verts,
        samples,
        sample_faces,
    }
}

impl Prepared {
    /// Per-vertex skin from the model's per-sample skin.
    fn transfer(&self, how: Transfer, skin: &[Vec<f32>]) -> (Vec<Vec<f32>>, Value) {
        let t = std::time::Instant::now();
        let (out, st) = mesh::transfer(
            how,
            &self.samples,
            &self.sample_faces,
            skin,
            &self.verts,
            &self.mesh.faces,
        );
        let mut d = json!({"method": how.name(), "seconds": (t.elapsed().as_secs_f64() * 100.0).round() / 100.0});
        if how != Transfer::Straight {
            d["welded_vertices"] = json!(st.welded);
            d["pieces"] = json!(st.pieces);
            d["fell_back_to_straight"] = json!(st.fallback);
            d["fewer_than_8_samples"] = json!(st.short);
        }
        (out, d)
    }
}

/// `model.run`, or its result kept from an earlier run when
/// SKINTOKENS_PRED_CACHE names a file prefix: the first run writes
/// <prefix>.ids.npy and <prefix>.skin.npy, later runs with the same input
/// and seed read them instead of running the model (developer hook, so two
/// transfers can be compared on one prediction).
fn run_model(
    model: &TokenRig,
    prep: &Prepared,
    start: &[u32],
    cfg: &GenConfig,
    seed: u64,
) -> Result<model::Output> {
    use candle_core::{Device, Tensor};
    let Ok(prefix) = std::env::var("SKINTOKENS_PRED_CACHE") else {
        return model.run(&prep.samples, start, cfg, seed);
    };
    let (ids_p, skin_p) = (format!("{prefix}.ids.npy"), format!("{prefix}.skin.npy"));
    if std::path::Path::new(&skin_p).exists() {
        let ids: Vec<u32> = Tensor::read_npy(&ids_p)?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|x| x as u32)
            .collect();
        let skin: Vec<Vec<f32>> = Tensor::read_npy(&skin_p)?.to_vec2::<f32>()?;
        if skin.len() != prep.samples.len() || !ids.starts_with(start) {
            bail!("SKINTOKENS_PRED_CACHE {prefix}: kept prediction does not match this input");
        }
        let w = ids
            .iter()
            .position(|&t| t == tokenizer::EOS)
            .ok_or_else(|| anyhow!("kept ids have no skeleton end"))?;
        let skeleton = tokenizer::detokenize(&ids[..=w])?;
        return Ok(model::Output {
            ids,
            skeleton,
            skin,
            at_limit: false,
        });
    }
    let out = model.run(&prep.samples, start, cfg, seed)?;
    let ids: Vec<i64> = out.ids.iter().map(|&x| x as i64).collect();
    Tensor::from_vec(ids.clone(), ids.len(), &Device::Cpu)?.write_npy(&ids_p)?;
    let j = out.skin.first().map(|r| r.len()).unwrap_or(0);
    let flat: Vec<f32> = out.skin.iter().flatten().copied().collect();
    Tensor::from_vec(flat, (out.skin.len(), j), &Device::Cpu)?.write_npy(&skin_p)?;
    Ok(out)
}

pub fn weights_summary(js: &[[u16; 4]], ws: &[[f32; 4]]) -> Value {
    let _ = js;
    let max_inf = ws
        .iter()
        .map(|w| w.iter().filter(|&&x| x > 1e-6).count())
        .max()
        .unwrap_or(0);
    let err = ws
        .iter()
        .map(|w| (w.iter().sum::<f32>() - 1.0).abs() as f64)
        .fold(0.0, f64::max);
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
    let length = if n <= 1 {
        1.0
    } else {
        sum / ((n - 1).max(1) as f64) * 0.5
    };
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
            let d = if l < 1e-6 {
                [0.0, 0.0, 1.0]
            } else {
                geom::scale(d, 1.0 / l)
            };
            tails[i] = geom::add(joints[i], geom::scale(d, length));
        }
    }
    if !joints.is_empty() {
        tails[root] = geom::add(joints[root], [0.0, 0.0, length]);
    }
    (0..n)
        .map(|i| {
            let d = geom::sub(tails[i], joints[i]);
            let l = geom::norm(d);
            geom::bone_rotation(if l > 0.0 {
                geom::scale(d, 1.0 / l)
            } else {
                [0.0, 1.0, 0.0]
            })
        })
        .collect()
}

/// Model-frame (Blender) rotation columns -> glTF world columns. Blender's
/// exporter converts the world axes only; bones keep their local frame
/// (Y along the bone).
fn rot_to_gltf(c: [V3; 3]) -> [V3; 3] {
    [
        mesh::from_model(c[0]),
        mesh::from_model(c[1]),
        mesh::from_model(c[2]),
    ]
}

pub fn rig(
    input: &Glb,
    model: &TokenRig,
    cfg: &GenConfig,
    seed: u64,
    humanoid: bool,
    how: Transfer,
) -> Result<RigResult> {
    let parts = input.parts(Frame::World)?;
    let first_part = parts.first().ok_or_else(|| anyhow!("no mesh part"))?;
    let mut rng = Rng::new(seed);
    let model_parts: Vec<(Vec<V3>, Vec<[u32; 3]>)> = parts
        .iter()
        .map(|p| {
            (
                p.positions.iter().map(|&v| mesh::to_model(v)).collect(),
                p.triangles.clone(),
            )
        })
        .collect();
    let prep = prepare(model_parts, &[], &mut rng);
    let start = vec![tokenizer::BOS, tokenizer::CLS_ARTICULATION];
    let out = run_model(model, &prep, &start, cfg, rng.next_u64())?;
    let sk = &out.skeleton;
    let (skin, transfer) = prep.transfer(how, &out.skin);
    let (js, ws) = mesh::top4(&skin);

    // skeleton in the input's world space
    let mut heads_model: Vec<V3> = sk
        .joints
        .iter()
        .map(|j| prep.norm.invert([j[0] as f64, j[1] as f64, j[2] as f64]))
        .collect();
    let rots = bone_frames(&mut heads_model, &sk.parents, &mut rng);
    let worlds: Vec<M4> = heads_model
        .iter()
        .zip(&rots)
        .map(|(&h, &r)| geom::from_rt(rot_to_gltf(r), mesh::from_model(h)))
        .collect();

    let mut g = Glb {
        json: input.json.clone(),
        bin: input.bin.clone(),
    };
    strip_rig(&mut g)?;
    let node_worlds = g.world_matrices();
    // names: humanoid naming works on heads in the first mesh's bind space
    let mut names: Vec<String> = sk.names.clone();
    let mut missing = vec![];
    let mut details = json!({"joints_raw": names.len()});
    if humanoid {
        let m0 = geom::inverse(
            node_worlds
                .get(first_part.node)
                .ok_or_else(|| anyhow!("first part node {} out of range", first_part.node))?,
        );
        let heads: Vec<[f64; 3]> = worlds
            .iter()
            .map(|w| geom::apply(&m0, geom::translation(w)))
            .collect();
        let parents: Vec<Option<usize>> = sk
            .parents
            .iter()
            .map(|&p| if p < 0 { None } else { Some(p as usize) })
            .collect();
        let (mapping, miss) = naming::name_humanoid(&names, &parents, &heads);
        if !miss.is_empty() {
            details["missing"] = json!(miss);
            missing = miss;
        } else {
            names = names
                .iter()
                .map(|n| mapping.get(n).cloned().unwrap_or_else(|| n.clone()))
                .collect();
            details["naming"] = json!("mixamo (structural)");
        }
    }

    attach_skeleton(&mut g, &parts, &names, &sk.parents, &worlds, &js, &ws)?;
    g.doc()?;
    details["weights"] = weights_summary(&js, &ws);
    details["tokens"] = json!(out.ids.len());
    details["skeleton_cut_at_token_limit"] = json!(out.at_limit);
    details["transfer"] = transfer;
    details["last_joint"] = json!(names.last());
    details["full_skin"] = json!(model::full_skin());
    Ok(RigResult {
        glb: g,
        details,
        missing,
    })
}

// ------------------------------------------------------------------ skin

pub struct SkinResult {
    pub glb: Glb,
    pub details: Value,
}

pub fn skin(
    input: &Glb,
    model: &TokenRig,
    cfg: &GenConfig,
    seed: u64,
    how: Transfer,
) -> Result<SkinResult> {
    let info = input.skin_info()?;
    let skin_joints: Vec<usize> = input
        .json
        .get("skins")
        .and_then(|s| s.get(0))
        .and_then(|s| s.get("joints"))
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
        .filter_map(|v| v.as_u64().map(|x| x as usize))
        .collect();
    let helpers = helper_joints(input, &skin_joints);
    let parts = input.parts(Frame::SkinBind)?;
    // joints the model sees: no helpers (their weight counts for the driver)
    let visible: Vec<usize> = (0..info.nodes.len())
        .filter(|&i| !helpers.contains_key(&info.nodes[i]))
        .collect();
    let vis_parents = mesh::reparent(&info.parents, &visible);
    let slot_to_vis: HashMap<usize, usize> = visible
        .iter()
        .enumerate()
        .map(|(vi, &i)| (info.skin_slot[i], vi))
        .collect();
    let node_to_slot: HashMap<usize, usize> = info
        .nodes
        .iter()
        .zip(&info.skin_slot)
        .map(|(&n, &s)| (n, s))
        .collect();
    let mut weight_sum = vec![0f64; visible.len()];
    for p in &parts {
        if let (Some(j), Some(w)) = (&p.joints, &p.weights) {
            for (jr, wr) in j.iter().zip(w) {
                for c in 0..4 {
                    let mut slot = jr[c] as usize;
                    if let Some(n) = skin_joints.get(slot) {
                        if let Some(d) = helpers.get(n) {
                            if let Some(&ds) = node_to_slot.get(d) {
                                slot = ds;
                            }
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
    if parents.is_empty() || parents.iter().filter(|&&p| p < 0).count() != 1 || parents[0] != -1 {
        bail!("the skeleton must have exactly one root joint");
    }
    let arm = |v: V3| mesh::to_model(geom::apply(&info.to_armature, v).map(|x| x as f32 as f64));
    let joints_model: Vec<V3> = kept.iter().map(|&i| arm(info.heads[i])).collect();
    let mut rng = Rng::new(seed);
    let model_parts: Vec<(Vec<V3>, Vec<[u32; 3]>)> = parts
        .iter()
        .map(|p| {
            (
                p.positions.iter().map(|&v| arm(v)).collect(),
                p.triangles.clone(),
            )
        })
        .collect();
    let prep = prepare(model_parts, &joints_model, &mut rng);
    let jn: Vec<[f64; 3]> = joints_model.iter().map(|&j| prep.norm.apply(j)).collect();
    let mut start = tokenizer::tokenize(&jn, &parents, tokenizer::CLS_ARTICULATION);
    // developer hooks for parity experiments
    if let Ok(p) = std::env::var("SKINTOKENS_DUMP_TOKENS") {
        let t: Vec<i64> = start.iter().map(|&x| x as i64).collect();
        let _ = candle_core::Tensor::from_vec(t.clone(), t.len(), &candle_core::Device::Cpu)
            .and_then(|t| t.write_npy(p));
    }
    if let Ok(p) = std::env::var("SKINTOKENS_START_TOKENS") {
        let t = candle_core::Tensor::read_npy(p)?
            .to_dtype(candle_core::DType::I64)?
            .to_vec1::<i64>()?;
        start = t.into_iter().map(|x| x as u32).collect();
    }
    let out = run_model(model, &prep, &start, cfg, rng.next_u64())?;
    if out.skin.first().map(|r| r.len()) != Some(kept.len()) {
        bail!(
            "model returned {} joints for {} given",
            out.skin.first().map(|r| r.len()).unwrap_or(0),
            kept.len()
        );
    }
    let (skin, transfer) = prep.transfer(how, &out.skin);
    let (js, ws) = mesh::top4(&skin);
    let slots: Vec<usize> = kept.iter().map(|&i| info.skin_slot[i]).collect();
    let mut g = Glb {
        json: input.json.clone(),
        bin: input.bin.clone(),
    };
    for (pi, p) in parts.iter().enumerate() {
        let (s, e) = (
            prep.mesh.offsets[pi],
            prep.mesh.offsets[pi] + p.positions.len(),
        );
        if e > js.len() || e > ws.len() {
            bail!("predicted skin attributes shorter than mesh part vertex count");
        }
        let attrs = input
            .json
            .get("meshes")
            .and_then(|m| m.get(p.mesh))
            .and_then(|m| m.get("primitives"))
            .and_then(|pr| pr.get(p.prim))
            .and_then(|pr| pr.get("attributes"))
            .ok_or_else(|| anyhow!("missing attributes on mesh {} prim {}", p.mesh, p.prim))?;
        let (Some(ja), Some(wa)) = (
            attrs.get("JOINTS_0").and_then(|v| v.as_u64()),
            attrs.get("WEIGHTS_0").and_then(|v| v.as_u64()),
        ) else {
            bail!("skinned primitive without JOINTS_0/WEIGHTS_0")
        };
        let jrows: Vec<[f64; 4]> = js[s..e]
            .iter()
            .zip(&ws[s..e])
            .map(|(j, w)| {
                let mut r = [0.0; 4];
                for c in 0..4 {
                    r[c] = if w[c] > 0.0 {
                        *slots
                            .get(j[c] as usize)
                            .ok_or_else(|| anyhow!("predicted joint index {} out of range", j[c]))?
                            as f64
                    } else {
                        0.0
                    };
                }
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
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
        "transfer": transfer,
    });
    Ok(SkinResult { glb: g, details })
}
