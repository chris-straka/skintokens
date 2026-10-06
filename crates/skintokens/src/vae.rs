//! FSQ-CVAE skin decoder (upstream `src/model/skin_vae/`, MIT; the DiT
//! block follows diffusers' HunyuanDiT block, Apache-2.0; FSQ follows
//! lucidrains/vector-quantize-pytorch, MIT).
//!
//! Inference uses three parts: the condition encoder (mesh points ->
//! 384 condition latents), FSQ `indices_to_codes` (4 SkinTokens per joint
//! -> latents) and the decoder (latents + condition -> per-point weight).

use anyhow::Result;
use candle_core::{DType, Tensor, D};
use skintokens_nn::{attention, fps, trace, LayerNorm, Linear, Weights};

/// FrequencyPositionalEmbedding(num_freqs, logspace, include_pi, use_pmpe):
/// [x, sin(x f) + sin(x pi/2 + phase), cos(x f) + cos(x pi/2 + phase)].
/// Computed on the CPU: Metal's sin/cos are not accurate at these
/// arguments (up to ~400 rad).
pub fn freq_embed(x: &Tensor, num_freqs: usize) -> Result<Tensor> {
    let out_dev = x.device().clone();
    let x = x.to_device(&candle_core::Device::Cpu)?;
    let dev = &candle_core::Device::Cpu;
    let pi = std::f32::consts::PI;
    let f: Vec<f32> = (0..num_freqs).map(|i| 2f32.powi(i as i32) * pi).collect();
    let n = num_freqs as f32;
    let ph: Vec<f32> = (0..num_freqs)
        .map(|i| {
            let i1 = (i + 1) as f32;
            (n.powf(1.0 - i1 / n) + i1 / n) * pi * 2.0
        })
        .collect();
    // Upstream casts the whole VAE to bf16 (`vae.to(torch.bfloat16)`),
    // which rounds these buffers too; inference uses the rounded values.
    let f = Tensor::from_vec(f, num_freqs, dev)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let ph = Tensor::from_vec(ph, num_freqs, dev)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let x = x.to_dtype(DType::F32)?;
    let mut shape = x.dims().to_vec();
    *shape.last_mut().unwrap() *= num_freqs;
    let xe = x.unsqueeze(D::Minus1)?;
    let e = xe.broadcast_mul(&f)?.reshape(shape.clone())?;
    let p = (xe * (pi as f64 * 0.5))?.broadcast_add(&ph)?.reshape(shape)?;
    let s = (e.sin()? + p.sin()?)?;
    let c = (e.cos()? + p.cos()?)?;
    Ok(Tensor::cat(&[x, s, c], D::Minus1)?.to_device(&out_dev)?)
}

fn merge(x: &Tensor) -> Result<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    Ok(x.transpose(1, 2)?.reshape((b, l, h * d))?)
}

/// diffusers `Attention` driven by Tripo2AttnProcessor: heads are split
/// first, then q/k/v (self) or k/v (cross) inside each head.
struct Attn {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    to_out: Linear,
    norm_cross: Option<LayerNorm>,
    heads: usize,
}

impl Attn {
    fn load(w: &Weights, heads: usize) -> Result<Self> {
        Ok(Self {
            to_q: Linear::load_auto(w, "to_q")?,
            to_k: Linear::load_auto(w, "to_k")?,
            to_v: Linear::load_auto(w, "to_v")?,
            to_out: Linear::load(w, "to_out.0", true)?,
            norm_cross: if w.has("norm_cross.weight") { Some(LayerNorm::load(w, "norm_cross", 1e-5)?) } else { None },
            heads,
        })
    }

    fn forward(&self, x: &Tensor, enc: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let h = self.heads;
        let q = self.to_q.forward(x)?;
        let (q, k, v) = match enc {
            None => {
                let k = self.to_k.forward(x)?;
                let v = self.to_v.forward(x)?;
                let d = c / h;
                let qkv = Tensor::cat(&[q, k, v], D::Minus1)?.reshape((b, l, h, 3, d))?;
                let p = |i| -> Result<Tensor> { Ok(qkv.narrow(3, i, 1)?.squeeze(3)?.transpose(1, 2)?) };
                (p(0)?, p(1)?, p(2)?)
            }
            Some(e) => {
                let e = match &self.norm_cross {
                    Some(n) => n.forward(e)?,
                    None => e.clone(),
                };
                let s = e.dims3()?.1;
                let k = self.to_k.forward(&e)?;
                let v = self.to_v.forward(&e)?;
                let d = k.dims3()?.2 / h;
                let kv = Tensor::cat(&[k, v], D::Minus1)?.reshape((b, s, h, 2, d))?;
                let k = kv.narrow(3, 0, 1)?.squeeze(3)?.transpose(1, 2)?;
                let v = kv.narrow(3, 1, 1)?.squeeze(3)?.transpose(1, 2)?;
                let q = q.reshape((b, l, h, d))?.transpose(1, 2)?;
                (q, k, v)
            }
        };
        let d = q.dims4()?.3;
        let o = attention(&q, &k, &v, 1.0 / (d as f32).sqrt(), false)?;
        self.to_out.forward(&merge(&o)?)
    }
}

/// DiTBlock with either self-attention (norm1/attn1) or cross-attention
/// (norm2/attn2), then a GELU feed-forward (norm3/ff).
struct DitBlock {
    norm1: Option<LayerNorm>,
    attn1: Option<Attn>,
    norm2: Option<LayerNorm>,
    attn2: Option<Attn>,
    norm3: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl DitBlock {
    fn load(w: &Weights, heads: usize) -> Result<Self> {
        let selfa = w.has("attn1.to_q.weight");
        let cross = w.has("attn2.to_q.weight");
        Ok(Self {
            norm1: if selfa { Some(LayerNorm::load(w, "norm1", 1e-5)?) } else { None },
            attn1: if selfa { Some(Attn::load(&w.pp("attn1"), heads)?) } else { None },
            norm2: if cross { Some(LayerNorm::load(w, "norm2", 1e-5)?) } else { None },
            attn2: if cross { Some(Attn::load(&w.pp("attn2"), heads)?) } else { None },
            norm3: LayerNorm::load(w, "norm3", 1e-5)?,
            ff_in: Linear::load(w, "ff.net.0.proj", true)?,
            ff_out: Linear::load(w, "ff.net.2", true)?,
        })
    }

    fn forward(&self, x: &Tensor, enc: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.clone();
        if let (Some(n), Some(a)) = (&self.norm1, &self.attn1) {
            x = (&x + a.forward(&n.forward(&x)?, None)?)?;
        }
        if let (Some(n), Some(a)) = (&self.norm2, &self.attn2) {
            x = (&x + a.forward(&n.forward(&x)?, enc)?)?;
        }
        let f = self.ff_out.forward(&self.ff_in.forward(&self.norm3.forward(&x)?)?.gelu_erf()?)?;
        Ok((x + f)?)
    }
}

/// FSQ codebook: levels -> codes in [-1, 1], then `project_out`.
pub struct Fsq {
    levels: Vec<u32>,
    project_out: Linear,
}

impl Fsq {
    pub fn codebook_size(&self) -> usize {
        self.levels.iter().map(|&l| l as usize).product()
    }

    /// indices (any count) -> (count, dim) latents. Indices outside the
    /// codebook wrap per level exactly as upstream's integer arithmetic does
    /// (it matters: upstream feeds the final EOS id through here).
    pub fn indices_to_codes(&self, idx: &[i64], dev: &candle_core::Device, dtype: DType) -> Result<Tensor> {
        let mut codes = Vec::with_capacity(idx.len() * self.levels.len());
        for &i in idx {
            let mut basis = 1i64;
            for &l in &self.levels {
                let l = l as i64;
                let li = (i.div_euclid(basis)).rem_euclid(l);
                let half = l / 2;
                codes.push((li - half) as f32 / half as f32);
                basis *= l;
            }
        }
        let t = Tensor::from_vec(codes, (idx.len(), self.levels.len()), dev)?.to_dtype(dtype)?;
        self.project_out.forward(&t)
    }
}

pub struct SkinVae {
    // condition encoder (Tripo2Encoder without learned queries)
    cond_proj_in: Linear,
    cond_blocks: Vec<DitBlock>,
    cond_norm_out: LayerNorm,
    cond_quant: Linear,
    // decoder (Tripo2Decoder)
    post_quant: Linear,
    dec_blocks: Vec<DitBlock>,
    proj_query: Linear,
    dec_norm_out: LayerNorm,
    proj_out: Linear,
    pub fsq: Fsq,
    pub num_freqs: usize,
    dtype: DType,
}

impl SkinVae {
    /// `w` points at `vae.model` in the TokenRig checkpoint.
    pub fn load(w: &Weights, heads: usize, enc_layers: usize, dec_layers: usize, levels: Vec<u32>, num_freqs: usize) -> Result<Self> {
        let ce = w.pp("cond_encoder");
        let de = w.pp("decoder");
        Ok(Self {
            cond_proj_in: Linear::load(&ce, "proj_in", true)?,
            cond_blocks: (0..enc_layers + 1).map(|i| DitBlock::load(&ce.pp(format!("blocks.{i}")), heads)).collect::<Result<_>>()?,
            cond_norm_out: LayerNorm::load(&ce, "norm_out", 1e-5)?,
            cond_quant: Linear::load(w, "cond_quant", true)?,
            post_quant: Linear::load(w, "post_quant", true)?,
            dec_blocks: (0..dec_layers + 1).map(|i| DitBlock::load(&de.pp(format!("blocks.{i}")), heads)).collect::<Result<_>>()?,
            proj_query: Linear::load(&de, "proj_query", true)?,
            dec_norm_out: LayerNorm::load(&de, "norm_out", 1e-5)?,
            proj_out: Linear::load(&de, "proj_out", true)?,
            fsq: Fsq { levels, project_out: Linear::load(w, "FSQ.project_out", true)? },
            num_freqs,
            dtype: w.dtype,
        })
    }

    fn embed_points(&self, pts: &Tensor) -> Result<Tensor> {
        let pos = pts.narrow(D::Minus1, 0, 3)?;
        let feat = pts.narrow(D::Minus1, 3, pts.dim(D::Minus1)? - 3)?.to_dtype(DType::F32)?;
        Ok(Tensor::cat(&[freq_embed(&pos, self.num_freqs)?, feat], D::Minus1)?.to_dtype(self.dtype)?)
    }

    /// Upstream `_sample_features`: `4 * n` random points, farthest point
    /// sampling to `n`. Returns indices into the cloud.
    pub fn select_queries(points: &[[f32; 3]], n: usize, rng: &mut skintokens_nn::Rng) -> Vec<usize> {
        let choice = rng.choice(points.len(), n * 4);
        let picked: Vec<[f32; 3]> = choice.iter().map(|&i| points[i]).collect();
        fps(&picked, 0.25).into_iter().map(|i| choice[i]).collect()
    }

    /// cond: (N, 6) [positions, normals]; query: indices of the condition
    /// tokens. Returns (1, len(query), latent).
    pub fn encode_cond(&self, cond: &Tensor, query: &[usize]) -> Result<Tensor> {
        let cond = &cond.contiguous()?;
        let idx = Tensor::from_vec(query.iter().map(|&i| i as u32).collect::<Vec<_>>(), query.len(), cond.device())?;
        let eq = self.embed_points(&cond.index_select(&idx, 0)?)?;
        trace("vae.embed_q", &eq);
        let q = self.cond_proj_in.forward(&eq)?.unsqueeze(0)?;
        let kv = self.cond_proj_in.forward(&self.embed_points(cond)?)?.unsqueeze(0)?;
        trace("vae.proj_in.0", &q.squeeze(0)?);
        trace("vae.proj_in.1", &kv.squeeze(0)?);
        let mut x = self.cond_blocks[0].forward(&q, Some(&kv))?;
        trace("vae.block0.0", &x.squeeze(0)?);
        for (i, b) in self.cond_blocks[1..].iter().enumerate() {
            x = b.forward(&x, None)?;
            trace(&format!("vae.block{}.0", i + 1), &x.squeeze(0)?);
        }
        self.cond_quant.forward(&self.cond_norm_out.forward(&x)?)
    }

    /// Embedded decoder queries for a point cloud: (1, N, width). Shared by
    /// every joint.
    pub fn decoder_queries(&self, cond: &Tensor) -> Result<Tensor> {
        Ok(self.proj_query.forward(&self.embed_points(cond)?)?.unsqueeze(0)?)
    }

    /// z: (b, tokens, latent) codes for b joints; cond_latents: (1, 384,
    /// latent); queries from `decoder_queries`. Returns (N, b) weights in
    /// [0, 1] (sigmoid).
    pub fn decode(&self, z: &Tensor, cond_latents: &Tensor, queries: &Tensor) -> Result<Tensor> {
        let b = z.dim(0)?;
        let cl = cond_latents.broadcast_as((b, cond_latents.dim(1)?, cond_latents.dim(2)?))?;
        trace("dec.z", &z.get(0)?);
        let mut h = self.post_quant.forward(&Tensor::cat(&[z, &cl.contiguous()?], 1)?)?;
        trace("dec.post_quant", &h.get(0)?);
        let (last, selfs) = self.dec_blocks.split_last().unwrap();
        for blk in selfs {
            h = blk.forward(&h, None)?;
        }
        trace("dec.kv", &h.get(0)?);
        let (_, n, c) = queries.dims3()?;
        trace("dec.q", &queries.get(0)?.narrow(0, 0, 2000.min(n))?);
        // The query side is independent per point: chunk it to bound memory
        // (each point carries a 3072-wide feed-forward activation).
        let chunk = decode_chunk();
        let mut outs = Vec::new();
        let mut st = 0;
        while st < n {
            let m = chunk.min(n - st);
            let q = queries.narrow(1, st, m)?.broadcast_as((b, m, c))?.contiguous()?;
            let l = last.forward(&q, Some(&h))?;
            if st == 0 {
                trace("dec.cross", &l.get(0)?.narrow(0, 0, 2000.min(m))?);
            }
            let logits = self.proj_out.forward(&self.dec_norm_out.forward(&l)?)?;
            let sg = candle_nn::ops::sigmoid(&logits.to_dtype(DType::F32)?)?.squeeze(2)?;
            outs.push(sg.to_device(&candle_core::Device::Cpu)?);
            st += m;
        }
        let all = Tensor::cat(&outs, 1)?;
        trace("dec.logits", &all.get(0)?);
        Ok(all.t()?)
    }
}

fn decode_chunk() -> usize {
    std::env::var("SKINTOKENS_DECODE_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(16384)
}
