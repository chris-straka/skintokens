//! Developer checks against tensors dumped from the Python reference
//! (`parity/dump_ref.py` at commit ac86785, run in the old upstream venv):
//! `skintokens parity-net REFDIR [gen]`, `skintokens parity-input IN DIR rig|skin`.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::generate::{self, GenConfig, Grammar};
use crate::model::{TokenRig, TOKENS_PER_SKIN};

fn npy(dir: &Path, name: &str) -> Result<Tensor> {
    Tensor::read_npy(dir.join(format!("{name}.npy"))).with_context(|| format!("{name}.npy"))
}

fn ids(dir: &Path, name: &str) -> Result<Vec<u32>> {
    Ok(npy(dir, name)?.to_dtype(DType::I64)?.to_vec1::<i64>()?.into_iter().map(|v| v as u32).collect())
}

fn idx(dir: &Path, name: &str) -> Result<Vec<usize>> {
    Ok(npy(dir, name)?.to_dtype(DType::I64)?.flatten_all()?.to_vec1::<i64>()?.into_iter().map(|v| v as usize).collect())
}

/// max |a-b|, max |a-b| / max |b|, mean |a-b|, and cosine similarity.
pub fn diff(name: &str, a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?;
    let d = (&a - &b)?.abs()?;
    let max = d.max(0)?.to_scalar::<f32>()?;
    let mean = d.mean(0)?.to_scalar::<f32>()?;
    let bmax = b.abs()?.max(0)?.to_scalar::<f32>()?;
    let cos = ((&a * &b)?.sum_all()?.to_scalar::<f32>()?)
        / ((a.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt() * (b.sqr()?.sum_all()?.to_scalar::<f32>()?).sqrt());
    println!("{name:<28} max {max:.3e}  rel {:.3e}  mean {mean:.3e}  cos {cos:.7}", max / bmax.max(1e-12));
    Ok(max)
}

pub fn run(model: &TokenRig, dir: &Path, gen: bool) -> Result<()> {
    let dev = &model.device;
    let sampled = npy(dir, "sampled")?.to_device(dev)?;
    let points: Vec<[f32; 3]> = skintokens_nn::to_points(&sampled)?;
    // FPS on the same random picks
    let choice = idx(dir, "enc_choice")?;
    let eq_ref = idx(dir, "enc_query")?;
    let eq = skintokens_encoder::ShapeEncoder::fps_queries(&points, &choice);
    let same = eq.iter().zip(&eq_ref).filter(|(a, b)| a == b).count();
    println!("encoder FPS indices          {same}/{} identical", eq_ref.len());
    let t = std::time::Instant::now();
    skintokens_nn::trace_enable(true);
    let (lat, mc) = model.mesh_cond(&sampled, &eq_ref)?;
    let _cl = model.vae.encode_cond(&sampled, &idx(dir, "vae_query")?)?;
    for (name, tns) in skintokens_nn::trace_take() {
        let f = dir.join("layers").join(format!("{name}.npy"));
        if f.exists() {
            diff(&format!("  {name}"), &tns, &Tensor::read_npy(&f)?)?;
        }
    }
    skintokens_nn::trace_enable(false);
    diff("encoder latents", &lat.squeeze(0)?, &npy(dir, "enc_latents")?)?;
    diff("mesh_cond (output_proj)", &mc.squeeze(0)?, &npy(dir, "mesh_cond")?)?;
    eprintln!("  encoder {:?}", t.elapsed());
    let t = std::time::Instant::now();
    let cl = model.vae.encode_cond(&sampled, &idx(dir, "vae_query")?)?;
    diff("VAE cond_latents", &cl.squeeze(0)?, &npy(dir, "cond_latents")?)?;
    eprintln!("  vae cond {:?}", t.elapsed());
    // LLM on the reference condition, fixed prefix
    let mc_ref = npy(dir, "mesh_cond")?.to_device(dev)?.unsqueeze(0)?;
    let prefix = ids(dir, "prefix_ids")?;
    let x = model.prompt(&mc_ref, &prefix)?;
    let mut cache = model.llm.new_cache();
    let t = std::time::Instant::now();
    let logits = model.llm.forward(&x, &mut cache, true)?.squeeze(0)?;
    let n = prefix.len();
    let logits = logits.narrow(0, logits.dim(0)? - n - 1, n + 1)?;
    diff("LLM logits (fixed prefix)", &logits, &npy(dir, "prefix_logits")?)?;
    let am: Vec<u32> = logits.argmax(1)?.to_vec1()?;
    let am_ref: Vec<u32> = npy(dir, "prefix_logits")?.argmax(1)?.to_vec1()?;
    println!("LLM argmax per position      {}/{} identical", am.iter().zip(&am_ref).filter(|(a, b)| a == b).count(), am.len());
    eprintln!("  llm prefill {:?}", t.elapsed());
    // VAE decoder for fixed tokens, reference condition
    if dir.join("dec_ids.npy").exists() {
        let cl_ref = npy(dir, "cond_latents")?.to_device(dev)?.unsqueeze(0)?;
        let t = std::time::Instant::now();
        skintokens_nn::trace_enable(true);
        let (_, skin) = model.decode(&ids(dir, "dec_ids")?, &sampled, &cl_ref)?.context("decode")?;
        let mut seen = std::collections::HashSet::new();
        for (name, tns) in skintokens_nn::trace_take() {
            let f = dir.join("layers").join(format!("{name}.npy"));
            if f.exists() && seen.insert(name.clone()) {
                let r = Tensor::read_npy(&f)?;
                if std::env::var("SKINTOKENS_PARITY_VALUES").is_ok() {
                    let a: Vec<f32> = tns.flatten_all()?.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.narrow(0, 0, 6)?.to_vec1()?;
                    let b: Vec<f32> = r.flatten_all()?.narrow(0, 0, 6)?.to_vec1()?;
                    println!("    {name} {:?} rust {a:?}\n    ref {:?} {b:?}", tns.dims(), r.dims());
                }
                diff(&format!("  {name}"), &tns, &r)?;
            }
        }
        skintokens_nn::trace_enable(false);
        let j = skin[0].len();
        let flat: Vec<f32> = skin.into_iter().flatten().collect();
        let st = Tensor::from_vec(flat, (sampled.dim(0)?, j), &Device::Cpu)?;
        diff("VAE decoder skin", &st, &npy(dir, "dec_skin")?.narrow(1, 0, j)?)?;
        eprintln!("  decode {j} joints {:?}", t.elapsed());
    }
    if gen {
        let start = ids(dir, "start_tokens")?;
        let prompt = model.prompt(&mc_ref, &start)?;
        let g = Grammar { init: start.clone(), eos_lm: model.eos_lm, tokens_per_skin: TOKENS_PER_SKIN };
        let cmp = |name: &str, got: &[u32]| -> Result<()> {
            let want = ids(dir, name)?;
            let mut full = start.clone();
            full.extend_from_slice(got);
            let first = full.iter().zip(&want).position(|(a, b)| a != b);
            println!("{name:<28} len {} vs {}  first difference: {:?}", full.len(), want.len(), first);
            Ok(())
        };
        let t = std::time::Instant::now();
        let cfg = GenConfig { num_beams: 1, do_sample: false, max_new_tokens: Some(400), ..Default::default() };
        cmp("greedy_ids", &generate::greedy(&model.llm, &prompt, &cfg, &g)?)?;
        eprintln!("  greedy {:?}", t.elapsed());
        let t = std::time::Instant::now();
        let cfg = GenConfig { num_beams: 10, do_sample: false, ..Default::default() };
        let mut rng = skintokens_nn::Rng::new(0);
        cmp("beam_ids", &generate::beam_search(&model.llm, &prompt, &cfg, &g, &mut rng)?)?;
        eprintln!("  beam {:?}", t.elapsed());
    }
    Ok(())
}

fn save_rows(dir: &Path, name: &str, rows: &[[f64; 3]]) -> Result<()> {
    let flat: Vec<f32> = rows.iter().flatten().map(|&v| v as f32).collect();
    Tensor::from_vec(flat, (rows.len(), 3), &Device::Cpu)?.write_npy(dir.join(format!("{name}.npy")))?;
    Ok(())
}

/// Writes what the Rust input pipeline feeds the model (mesh in the model
/// frame, normals, normalized vertices, skin-mode joints and prompt tokens)
/// for comparison with dump_ref.py's asset_* / norm_* / start_tokens.
pub fn dump_input(g: &crate::glb::Glb, dir: &Path, skin: bool) -> Result<()> {
    use crate::glb::Frame;
    use crate::mesh::{self, Mesh, Normalize};
    std::fs::create_dir_all(dir)?;
    let parts = g.parts(if skin { Frame::SkinBind } else { Frame::World })?;
    let to_arm = if skin { g.skin_info()?.to_armature } else { crate::geom::IDENTITY };
    let arm = |v: [f64; 3]| mesh::to_model(crate::geom::apply(&to_arm, v).map(|x| if skin { x as f32 as f64 } else { x }));
    let mp: Vec<_> = parts.iter().map(|p| (p.positions.iter().map(|&v| arm(v)).collect::<Vec<_>>(), p.triangles.clone())).collect();
    let m = Mesh::from_parts(&mp);
    save_rows(dir, "asset_vertices", &m.vertices)?;
    let faces: Vec<i64> = m.faces.iter().flatten().map(|&f| f as i64).collect();
    Tensor::from_vec(faces, (m.faces.len(), 3), &Device::Cpu)?.write_npy(dir.join("asset_faces.npy"))?;
    let fnorm = m.face_normals();
    save_rows(dir, "asset_face_normals", &fnorm)?;
    save_rows(dir, "asset_vertex_normals", &m.vertex_normals(&fnorm))?;
    let mut joints = vec![];
    if skin {
        let info = g.skin_info()?;
        joints = info.heads.iter().map(|&h| arm(h)).collect();
        save_rows(dir, "asset_joints", &joints)?;
        let par: Vec<i64> = info.parents.clone();
        Tensor::from_vec(par.clone(), par.len(), &Device::Cpu)?.write_npy(dir.join("asset_parents.npy"))?;
        let norm = Normalize::fit(m.vertices.iter().chain(joints.iter()));
        let jn: Vec<[f64; 3]> = joints.iter().map(|&j| norm.apply(j)).collect();
        save_rows(dir, "norm_joints", &jn)?;
        let toks: Vec<i64> = crate::tokenizer::tokenize(&jn, &par, crate::tokenizer::CLS_ARTICULATION).into_iter().map(|t| t as i64).collect();
        Tensor::from_vec(toks.clone(), toks.len(), &Device::Cpu)?.write_npy(dir.join("start_tokens.npy"))?;
    }
    let norm = Normalize::fit(m.vertices.iter().chain(joints.iter()));
    let nv: Vec<[f64; 3]> = m.vertices.iter().map(|&v| norm.apply(v)).collect();
    save_rows(dir, "norm_vertices", &nv)?;
    println!("wrote {}", dir.display());
    Ok(())
}
