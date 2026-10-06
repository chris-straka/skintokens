// SPDX-License-Identifier: GPL-3.0-only
//
// skintokens-encoder: the point-cloud shape encoder TokenRig conditions on.
//
// This is a Rust port of SkinTokens' `src/model/michelangelo/` modules
// (ShapeAsLatentPerceiverEncoder, CrossAttentionEncoder, the residual
// attention blocks and FourierEmbedder), which derive from
// NeuralCarver/Michelangelo, licensed GPL-3.0. This crate is therefore
// licensed GPL-3.0 as well (see COPYING); the rest of the skintokens
// workspace is MIT. The `skintokens` binary links this crate, so the
// binary as a whole is distributed under GPL-3.0.
//
// Copyright (C) 2026 the skintokens contributors.
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License version 3 as published
// by the Free Software Foundation. It is distributed WITHOUT ANY WARRANTY;
// see the GNU General Public License for details.

use anyhow::Result;
use candle_core::{DType, Tensor};
use skintokens_nn::{attention, fps, trace, LayerNorm, Linear, Rng, Weights};

/// sin/cos features at frequencies 2^0..2^(n-1) (times pi if `include_pi`),
/// with the input first: [x, sin(x f), cos(x f)], dim-major.
/// Computed on the CPU: Metal's sin/cos are not accurate for large arguments.
pub fn fourier_embed(x: &Tensor, num_freqs: usize, include_pi: bool) -> Result<Tensor> {
    let out_dev = x.device().clone();
    let x = x.to_device(&candle_core::Device::Cpu)?;
    let dev = &candle_core::Device::Cpu;
    let pi = if include_pi { std::f32::consts::PI } else { 1.0 };
    let f: Vec<f32> = (0..num_freqs).map(|i| 2f32.powi(i as i32) * pi).collect();
    let f = Tensor::from_vec(f, num_freqs, dev)?;
    let x = x.to_dtype(DType::F32)?;
    let dims = x.dims().to_vec();
    let mut e_shape = dims.clone();
    *e_shape.last_mut().unwrap() *= num_freqs;
    let e = x.unsqueeze(candle_core::D::Minus1)?.broadcast_mul(&f)?.reshape(e_shape)?;
    Ok(Tensor::cat(&[x, e.sin()?, e.cos()?], candle_core::D::Minus1)?.to_device(&out_dev)?)
}

struct Mlp {
    c_fc: Linear,
    c_proj: Linear,
}

impl Mlp {
    fn load(w: &Weights) -> Result<Self> {
        Ok(Self { c_fc: Linear::load(w, "c_fc", true)?, c_proj: Linear::load(w, "c_proj", true)? })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.c_proj.forward(&self.c_fc.forward(x)?.gelu_erf()?)
    }
}

/// (B, L, H*D) -> (B, H, L, D)
fn heads(x: &Tensor, h: usize) -> Result<Tensor> {
    let (b, l, c) = x.dims3()?;
    Ok(x.reshape((b, l, h, c / h))?.transpose(1, 2)?)
}

/// (B, H, L, D) -> (B, L, H*D)
fn merge(x: &Tensor) -> Result<Tensor> {
    let (b, h, l, d) = x.dims4()?;
    Ok(x.transpose(1, 2)?.reshape((b, l, h * d))?)
}

/// Self-attention block: x + attn(ln_1 x); x + mlp(ln_2 x). The fused qkv
/// projection splits per head into [q | k | v].
struct ResidualAttentionBlock {
    c_qkv: Linear,
    c_proj: Linear,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    mlp: Mlp,
    heads: usize,
}

impl ResidualAttentionBlock {
    fn load(w: &Weights, heads: usize) -> Result<Self> {
        Ok(Self {
            c_qkv: Linear::load_auto(w, "attn.c_qkv")?,
            c_proj: Linear::load(w, "attn.c_proj", true)?,
            ln_1: LayerNorm::load(w, "ln_1", 1e-5)?,
            ln_2: LayerNorm::load(w, "ln_2", 1e-5)?,
            mlp: Mlp::load(&w.pp("mlp"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, l, c) = x.dims3()?;
        let qkv = self.c_qkv.forward(&self.ln_1.forward(x)?)?;
        let d = c / self.heads;
        let qkv = qkv.reshape((b, l, self.heads, 3, d))?;
        let pick = |i: usize| -> Result<Tensor> { Ok(qkv.narrow(3, i, 1)?.squeeze(3)?.transpose(1, 2)?) };
        let o = attention(&pick(0)?, &pick(1)?, &pick(2)?, 1.0 / (d as f32).sqrt(), false)?;
        let x = (x + self.c_proj.forward(&merge(&o)?)?)?;
        Ok((&x + self.mlp.forward(&self.ln_2.forward(&x)?)?)?)
    }
}

/// Cross-attention block: x + attn(ln_1 x, ln_2 data); x + mlp(ln_3 x). The
/// key/value projection splits per head into [k | v].
struct ResidualCrossAttentionBlock {
    c_q: Linear,
    c_kv: Linear,
    c_proj: Linear,
    ln_1: LayerNorm,
    ln_2: LayerNorm,
    ln_3: LayerNorm,
    mlp: Mlp,
    heads: usize,
}

impl ResidualCrossAttentionBlock {
    fn load(w: &Weights, heads: usize) -> Result<Self> {
        Ok(Self {
            c_q: Linear::load_auto(w, "attn.c_q")?,
            c_kv: Linear::load_auto(w, "attn.c_kv")?,
            c_proj: Linear::load(w, "attn.c_proj", true)?,
            ln_1: LayerNorm::load(w, "ln_1", 1e-5)?,
            ln_2: LayerNorm::load(w, "ln_2", 1e-5)?,
            ln_3: LayerNorm::load(w, "ln_3", 1e-5)?,
            mlp: Mlp::load(&w.pp("mlp"))?,
            heads,
        })
    }

    fn forward(&self, x: &Tensor, data: &Tensor) -> Result<Tensor> {
        let q = heads(&self.c_q.forward(&self.ln_1.forward(x)?)?, self.heads)?;
        let kv = self.c_kv.forward(&self.ln_2.forward(data)?)?;
        let (b, n, c2) = kv.dims3()?;
        let d = c2 / self.heads / 2;
        let kv = kv.reshape((b, n, self.heads, 2, d))?;
        let k = kv.narrow(3, 0, 1)?.squeeze(3)?.transpose(1, 2)?;
        let v = kv.narrow(3, 1, 1)?.squeeze(3)?.transpose(1, 2)?;
        let o = attention(&q, &k, &v, 1.0 / (d as f32).sqrt(), false)?;
        let x = (x + self.c_proj.forward(&merge(&o)?)?)?;
        Ok((&x + self.mlp.forward(&self.ln_3.forward(&x)?)?)?)
    }
}

/// ShapeAsLatentPerceiverEncoder (no learned queries, full input): queries
/// are `token_num` farthest-point samples of the cloud, they cross-attend
/// to every point, then `layers` self-attention blocks and ln_post.
pub struct ShapeEncoder {
    input_proj: Linear,
    cross: ResidualCrossAttentionBlock,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: Option<LayerNorm>,
    pub num_freqs: usize,
    pub include_pi: bool,
    pub token_num: usize,
    pub width: usize,
}

impl ShapeEncoder {
    /// `w` points at `mesh_encoder.encoder` in the TokenRig checkpoint.
    pub fn load(w: &Weights, heads: usize, layers: usize, num_freqs: usize, include_pi: bool, token_num: usize) -> Result<Self> {
        let input_proj = Linear::load(w, "input_proj", true)?;
        let width = input_proj.out_dim();
        Ok(Self {
            input_proj,
            cross: ResidualCrossAttentionBlock::load(&w.pp("cross_attn"), heads)?,
            blocks: (0..layers)
                .map(|i| ResidualAttentionBlock::load(&w.pp(format!("self_attn.resblocks.{i}")), heads))
                .collect::<Result<_>>()?,
            ln_post: if w.has("ln_post.weight") { Some(LayerNorm::load(w, "ln_post", 1e-5)?) } else { None },
            num_freqs,
            include_pi,
            token_num,
            width,
        })
    }

    /// Query selection as upstream: `4 * token_num` random points (without
    /// replacement unless the cloud is smaller), then farthest point
    /// sampling down to a quarter. Upstream draws with a fixed seed in eval
    /// mode; this does the same with its own generator.
    pub fn select_queries(&self, points: &[[f32; 3]], seed: u64) -> Vec<usize> {
        let choice = Rng::new(seed).choice(points.len(), self.token_num * 4);
        Self::fps_queries(points, &choice)
    }

    /// `choice[fps(points[choice])]`.
    pub fn fps_queries(points: &[[f32; 3]], choice: &[usize]) -> Vec<usize> {
        let picked: Vec<[f32; 3]> = choice.iter().map(|&i| points[i]).collect();
        fps(&picked, 0.25).into_iter().map(|i| choice[i]).collect()
    }

    fn embed(&self, pc: &Tensor, feats: &Tensor, dtype: DType) -> Result<Tensor> {
        let e = fourier_embed(pc, self.num_freqs, self.include_pi)?;
        let x = Tensor::cat(&[e, feats.to_dtype(DType::F32)?], candle_core::D::Minus1)?.to_dtype(dtype)?;
        self.input_proj.forward(&x)
    }

    /// pc, feats: (N, 3) one cloud; query: indices into it. Returns the
    /// latents (1, token_num, width).
    pub fn forward(&self, pc: &Tensor, feats: &Tensor, query: &[usize]) -> Result<Tensor> {
        // index_select needs contiguous input on Metal
        let (pc, feats) = (&pc.contiguous()?, &feats.contiguous()?);
        let dtype = self.input_proj.dtype();
        let data = self.embed(pc, feats, dtype)?.unsqueeze(0)?;
        trace("enc.input_proj.0", &data.squeeze(0)?);
        let idx = Tensor::from_vec(query.iter().map(|&i| i as u32).collect::<Vec<_>>(), query.len(), pc.device())?;
        let q = self.embed(&pc.index_select(&idx, 0)?, &feats.index_select(&idx, 0)?, dtype)?.unsqueeze(0)?;
        trace("enc.input_proj.1", &q.squeeze(0)?);
        let mut x = self.cross.forward(&q, &data)?;
        trace("enc.cross.0", &x.squeeze(0)?);
        for (i, b) in self.blocks.iter().enumerate() {
            x = b.forward(&x)?;
            trace(&format!("enc.block{i}.0"), &x.squeeze(0)?);
        }
        match &self.ln_post {
            Some(ln) => ln.forward(&x),
            None => Ok(x),
        }
    }
}
