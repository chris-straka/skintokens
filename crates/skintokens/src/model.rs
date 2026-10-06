//! TokenRig: shape encoder -> Qwen3 -> skeleton + SkinTokens -> FSQ-CVAE
//! decoder -> per-point skin weights. Weights come straight from the
//! official `grpo_1400.ckpt` (it also carries the VAE).

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use skintokens_encoder::ShapeEncoder;
use skintokens_nn::{Linear, RmsNorm, Rng, Weights};

use crate::generate::{self, GenConfig, Grammar};
use crate::qwen::{Config, Qwen3};
use crate::tokenizer;
use crate::vae::SkinVae;

pub const TOKENS_PER_SKIN: usize = 4;
pub const TOKENS_SKIN_COND: usize = 384;

pub struct TokenRig {
    pub encoder: ShapeEncoder,
    proj: Linear,
    proj_norm: RmsNorm,
    pub llm: Qwen3,
    pub vae: SkinVae,
    pub device: Device,
    pub dtype: DType,
    pub eos_lm: u32,
}

/// What generation produced.
pub struct Output {
    /// Prompt + generated ids (upstream's `output_ids`).
    pub ids: Vec<u32>,
    pub skeleton: tokenizer::Skeleton,
    /// (N, J) weights for the sampled points, upstream's `skin_pred`.
    pub skin: Vec<Vec<f32>>,
}

impl TokenRig {
    pub fn load(ckpt: &Path, device: &Device, dtype: DType) -> Result<Self> {
        let w = Weights::load_ckpt(ckpt, dtype, device)?;
        let encoder = ShapeEncoder::load(&w.pp("mesh_encoder.encoder"), 8, 8, 8, false, 512).context("shape encoder")?;
        let vae = SkinVae::load(&w.pp("vae.model"), 12, 2, 10, vec![8, 8, 8, 8, 8], 8).context("skin VAE")?;
        let llm = Qwen3::load(&w.pp("transformer"), Config::default()).context("Qwen3")?;
        let vocab = llm.vocab();
        let expect = tokenizer::VOCAB as usize + vae.fsq.codebook_size() + 1;
        if vocab != expect {
            bail!("checkpoint vocabulary {vocab} != tokenizer {expect}");
        }
        Ok(Self {
            encoder,
            proj: Linear::load(&w, "output_proj.0", true)?,
            proj_norm: RmsNorm::load(&w, "output_proj.1", f32::EPSILON)?,
            llm,
            vae,
            device: device.clone(),
            dtype,
            eos_lm: (vocab - 1) as u32,
        })
    }

    /// (N, 6) [positions, normals] in the normalized model frame -> device tensor.
    pub fn cond_tensor(&self, pts: &[[f32; 6]]) -> Result<Tensor> {
        let flat: Vec<f32> = pts.iter().flatten().copied().collect();
        Ok(Tensor::from_vec(flat, (pts.len(), 6), &self.device)?)
    }

    /// Learned mesh condition (1, 512, hidden).
    pub fn mesh_cond(&self, cond: &Tensor, query: &[usize]) -> Result<(Tensor, Tensor)> {
        let pc = cond.narrow(1, 0, 3)?;
        let nr = cond.narrow(1, 3, 3)?;
        let lat = self.encoder.forward(&pc, &nr, query)?;
        let mc = self.proj_norm.forward(&self.proj.forward(&lat)?)?;
        Ok((lat, mc))
    }

    pub fn prompt(&self, mesh_cond: &Tensor, start: &[u32]) -> Result<Tensor> {
        let e = self.llm.embed_ids(&[start.to_vec()])?;
        Ok(Tensor::cat(&[&mesh_cond.to_dtype(self.dtype)?, &e], 1)?)
    }

    /// Decodes skin for `ids` (prompt + generated): (N, J) per sampled point.
    /// `None` when the token count does not cover every joint (upstream
    /// then has no skin either).
    pub fn decode(&self, ids: &[u32], cond: &Tensor, cond_latents: &Tensor) -> Result<Option<(tokenizer::Skeleton, Vec<Vec<f32>>)>> {
        let w = ids.iter().position(|&t| t == tokenizer::EOS).context("no skeleton eos in the generated tokens")?;
        let sk = tokenizer::detokenize(&ids[..=w])?;
        let j = sk.joints.len();
        let skin_ids = &ids[w + 1..];
        if skin_ids.len() < j * TOKENS_PER_SKIN {
            return Ok(None);
        }
        let idx: Vec<i64> = skin_ids[..j * TOKENS_PER_SKIN].iter().map(|&t| t as i64 - tokenizer::VOCAB as i64).collect();
        let queries = self.vae.decoder_queries(cond)?;
        let n = cond.dim(0)?;
        let mut cols: Vec<Tensor> = Vec::with_capacity(j);
        let batch = decode_batch(&self.device);
        let mut s = 0;
        let jmax = std::env::var("SKINTOKENS_PARITY_JOINTS").ok().and_then(|v| v.parse().ok()).unwrap_or(j);
        while s < j.min(jmax) {
            let b = batch.min(j.min(jmax) - s);
            let z = self.vae.fsq.indices_to_codes(&idx[s * TOKENS_PER_SKIN..(s + b) * TOKENS_PER_SKIN], &self.device, self.dtype)?;
            let z = z.reshape((b, TOKENS_PER_SKIN, ()))?;
            // Pull each batch back to the host: candle queues Metal work
            // without waiting, and 28 joints' (N, 3072) temporaries in
            // flight exhaust memory (seen: garbage weights, 60x slower).
            cols.push(self.vae.decode(&z, cond_latents, &queries)?.to_device(&Device::Cpu)?);
            self.device.synchronize()?;
            s += b;
        }
        let skin = Tensor::cat(&cols, 1)?.to_dtype(DType::F32)?;
        let skin: Vec<Vec<f32>> = skin.to_vec2()?;
        debug_assert_eq!(skin.len(), n);
        Ok(Some((sk, skin)))
    }

    /// Full run on one normalized cloud. `start` = prompt tokens (class head,
    /// or a whole skeleton for skin-only).
    pub fn run(&self, pts: &[[f32; 6]], start: &[u32], cfg: &GenConfig, seed: u64) -> Result<Output> {
        let cond = self.cond_tensor(pts)?;
        let points: Vec<[f32; 3]> = pts.iter().map(|p| [p[0], p[1], p[2]]).collect();
        let mut rng = Rng::new(seed);
        let vq = SkinVae::select_queries(&points, TOKENS_SKIN_COND, &mut rng);
        let cond_latents = self.vae.encode_cond(&cond, &vq)?;
        let eq = self.encoder.select_queries(&points, 0);
        let (_, mc) = self.mesh_cond(&cond, &eq)?;
        let prompt = self.prompt(&mc, start)?;
        let g = Grammar { init: start.to_vec(), eos_lm: self.eos_lm, tokens_per_skin: TOKENS_PER_SKIN };
        let gen = if cfg.num_beams == 1 && !cfg.do_sample {
            generate::greedy(&self.llm, &prompt, cfg, &g)?
        } else {
            generate::beam_search(&self.llm, &prompt, cfg, &g, &mut rng)?
        };
        let mut ids = start.to_vec();
        ids.extend(gen);
        match self.decode(&ids, &cond, &cond_latents)? {
            Some((skeleton, skin)) => Ok(Output { ids, skeleton, skin }),
            None => bail!("the model did not produce skin tokens for every joint"),
        }
    }
}

/// Joints decoded per batch: each one holds a (N, 3072) feed-forward
/// activation, so keep it small.
fn decode_batch(dev: &Device) -> usize {
    std::env::var("SKINTOKENS_DECODE_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(if dev.is_metal() { 2 } else { 1 })
}
