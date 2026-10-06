//! TokenRig's language model: the Qwen3 architecture (Apache-2.0 config
//! from Qwen/Qwen3-0.6B, hidden size changed to 896 and the vocabulary to
//! TokenRig's) with the fine-tuned weights from the TokenRig checkpoint.
//! Written for this use: inputs_embeds prefill, a batch of beams, and a KV
//! cache that can be reordered between steps.

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use skintokens_nn::{attention, Linear, RmsNorm, Weights};

pub struct Config {
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub eps: f32,
}

impl Default for Config {
    fn default() -> Self {
        Config { layers: 28, heads: 16, kv_heads: 8, head_dim: 128, rope_theta: 1_000_000.0, eps: 1e-6 }
    }
}

struct Layer {
    input_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_norm: RmsNorm,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// Per-layer cached keys/values, (B, kv_heads, T, head_dim).
#[derive(Clone)]
pub struct Cache {
    pub kv: Vec<Option<(Tensor, Tensor)>>,
    pub len: usize,
}

impl Cache {
    /// Repeats the batch (one prefilled prompt -> n beams).
    pub fn expand(&self, n: usize) -> Result<Cache> {
        let kv = self
            .kv
            .iter()
            .map(|e| {
                e.as_ref()
                    .map(|(k, v)| -> Result<(Tensor, Tensor)> {
                        let rep = |t: &Tensor| -> Result<Tensor> {
                            let (_, h, l, d) = t.dims4()?;
                            Ok(t.broadcast_as((n, h, l, d))?.contiguous()?)
                        };
                        Ok((rep(k)?, rep(v)?))
                    })
                    .transpose()
            })
            .collect::<Result<_>>()?;
        Ok(Cache { kv, len: self.len })
    }

    /// Keeps batch rows `idx` (beam reordering).
    pub fn select(&mut self, idx: &[usize]) -> Result<()> {
        let dev = self.kv.iter().flatten().next().map(|(k, _)| k.device().clone());
        let Some(dev) = dev else { return Ok(()) };
        let t = Tensor::from_vec(idx.iter().map(|&i| i as u32).collect::<Vec<_>>(), idx.len(), &dev)?;
        for e in self.kv.iter_mut().flatten() {
            *e = (e.0.index_select(&t, 0)?, e.1.index_select(&t, 0)?);
        }
        Ok(())
    }
}

pub struct Qwen3 {
    cfg: Config,
    embed: Tensor,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    inv_freq: Vec<f32>,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen3 {
    /// `w` points at `transformer` (HF Qwen3ForCausalLM) in the checkpoint.
    pub fn load(w: &Weights, cfg: Config) -> Result<Self> {
        let m = w.pp("model");
        let layers = (0..cfg.layers)
            .map(|i| {
                let l = m.pp(format!("layers.{i}"));
                Ok(Layer {
                    input_norm: RmsNorm::load(&l, "input_layernorm", cfg.eps)?,
                    q: Linear::load(&l, "self_attn.q_proj", false)?,
                    k: Linear::load(&l, "self_attn.k_proj", false)?,
                    v: Linear::load(&l, "self_attn.v_proj", false)?,
                    o: Linear::load(&l, "self_attn.o_proj", false)?,
                    q_norm: RmsNorm::load(&l, "self_attn.q_norm", cfg.eps)?,
                    k_norm: RmsNorm::load(&l, "self_attn.k_norm", cfg.eps)?,
                    post_norm: RmsNorm::load(&l, "post_attention_layernorm", cfg.eps)?,
                    gate: Linear::load(&l, "mlp.gate_proj", false)?,
                    up: Linear::load(&l, "mlp.up_proj", false)?,
                    down: Linear::load(&l, "mlp.down_proj", false)?,
                })
            })
            .collect::<Result<_>>()?;
        let embed = m.get("embed_tokens.weight")?;
        // Qwen3-0.6B ties the head to the embeddings (the checkpoint stores both, equal).
        let lm_head = if w.has("lm_head.weight") { Linear::load(w, "lm_head", false)? } else { Linear::new(&embed, None)? };
        let half = cfg.head_dim / 2;
        // Upstream casts the transformer to bf16, which rounds the rotary
        // `inv_freq` buffer as well; inference uses the rounded values.
        let inv_freq: Vec<f32> = (0..half).map(|i| (1.0 / cfg.rope_theta.powf(2.0 * i as f64 / cfg.head_dim as f64)) as f32).collect();
        let inv_freq = Tensor::from_vec(inv_freq, half, &candle_core::Device::Cpu)?.to_dtype(DType::BF16)?.to_dtype(DType::F32)?.to_vec1()?;
        Ok(Self {
            norm: RmsNorm::load(&m, "norm", cfg.eps)?,
            layers,
            embed,
            lm_head,
            inv_freq,
            device: w.device.clone(),
            dtype: w.dtype,
            cfg,
        })
    }

    pub fn vocab(&self) -> usize {
        self.embed.dim(0).unwrap()
    }

    pub fn hidden(&self) -> usize {
        self.embed.dim(1).unwrap()
    }

    pub fn new_cache(&self) -> Cache {
        Cache { kv: vec![None; self.cfg.layers], len: 0 }
    }

    /// Token ids -> embeddings (B, L, hidden).
    pub fn embed_ids(&self, ids: &[Vec<u32>]) -> Result<Tensor> {
        let b = ids.len();
        let l = ids[0].len();
        let flat: Vec<u32> = ids.iter().flatten().copied().collect();
        let t = Tensor::from_vec(flat, b * l, &self.device)?;
        Ok(self.embed.index_select(&t, 0)?.reshape((b, l, self.hidden()))?)
    }

    fn rope(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let half = self.inv_freq.len();
        let mut f = Vec::with_capacity(len * half);
        for p in start..start + len {
            for &w in &self.inv_freq {
                f.push(p as f32 * w);
            }
        }
        let f = Tensor::from_vec(f, (len, half), &self.device)?;
        let f = Tensor::cat(&[&f, &f], 1)?;
        Ok((f.cos()?, f.sin()?))
    }

    /// HF apply_rotary_pos_emb (rotate_half), x: (B, H, L, D).
    fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let d = x.dim(D::Minus1)?;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
        let rot = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let dt = x.dtype();
        let xf = x.to_dtype(DType::F32)?;
        let out = (xf.broadcast_mul(cos)? + rot.to_dtype(DType::F32)?.broadcast_mul(sin)?)?;
        Ok(out.to_dtype(dt)?)
    }

    /// Runs `x` (B, L, hidden) at positions cache.len.. and returns the
    /// last position's logits (B, vocab) in f32, or all positions if `all`.
    pub fn forward(&self, x: &Tensor, cache: &mut Cache, all: bool) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let (cos, sin) = self.rope(cache.len, l)?;
        let causal = cache.len == 0 && l > 1;
        let (h, hk, d) = (self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim);
        let mut x = x.to_dtype(self.dtype)?;
        for (i, ly) in self.layers.iter().enumerate() {
            let hn = ly.input_norm.forward(&x)?;
            let q = ly.q.forward(&hn)?.reshape((b, l, h, d))?;
            let k = ly.k.forward(&hn)?.reshape((b, l, hk, d))?;
            let v = ly.v.forward(&hn)?.reshape((b, l, hk, d))?.transpose(1, 2)?.contiguous()?;
            let q = Self::apply_rope(&ly.q_norm.forward(&q)?.transpose(1, 2)?, &cos, &sin)?;
            let k = Self::apply_rope(&ly.k_norm.forward(&k)?.transpose(1, 2)?, &cos, &sin)?.contiguous()?;
            let (k, v) = match cache.kv[i].take() {
                Some((pk, pv)) => (Tensor::cat(&[&pk, &k], 2)?, Tensor::cat(&[&pv, &v], 2)?),
                None => (k, v),
            };
            let o = attention(&q, &k, &v, 1.0 / (d as f32).sqrt(), causal)?;
            cache.kv[i] = Some((k, v));
            let o = o.transpose(1, 2)?.reshape((b, l, h * d))?;
            x = (x + ly.o.forward(&o)?)?;
            let hn = ly.post_norm.forward(&x)?;
            let m = ly.down.forward(&(ly.gate.forward(&hn)?.silu()? * ly.up.forward(&hn)?)?)?;
            x = (x + m)?;
        }
        cache.len += l;
        let x = if all { x } else { x.narrow(1, l - 1, 1)? };
        let logits = self.lm_head.forward(&self.norm.forward(&x)?)?.to_dtype(DType::F32)?;
        if all {
            Ok(logits)
        } else {
            Ok(logits.squeeze(1)?)
        }
    }
}

/// KV storage for beam search without copies: the prompt's keys/values are
/// kept once, every generated token gets a slot in a pool written in place,
/// and each beam attends to the prompt plus the slots on its own path (a
/// mask), so reordering beams is bookkeeping on the host.
pub struct BeamCache {
    /// per layer: prompt K^T (Hkv, D, P) and V (Hkv, P, D)
    prompt: Vec<(Tensor, Tensor)>,
    /// per layer: pool K^T (Hkv, D, cap) and V (Hkv, cap, D)
    pool: Vec<(Tensor, Tensor)>,
    pub cap: usize,
    pub used: usize,
    pub prompt_len: usize,
}

impl Qwen3 {
    /// Turns a prefilled single-sequence cache into a beam cache.
    pub fn beam_cache(&self, c: &Cache, cap: usize) -> Result<BeamCache> {
        let mut prompt = vec![];
        let mut pool = vec![];
        for e in &c.kv {
            let (k, v) = e.as_ref().ok_or_else(|| anyhow::anyhow!("cache not prefilled"))?;
            let k = k.squeeze(0)?; // (Hkv, P, D)
            let v = v.squeeze(0)?.contiguous()?;
            prompt.push((k.t()?.contiguous()?, v));
            pool.push(self.empty_pool(cap)?);
        }
        Ok(BeamCache { prompt, pool, cap, used: 0, prompt_len: c.len })
    }

    fn empty_pool(&self, cap: usize) -> Result<(Tensor, Tensor)> {
        let (hk, d) = (self.cfg.kv_heads, self.cfg.head_dim);
        Ok((Tensor::zeros((hk, d, cap), self.dtype, &self.device)?, Tensor::zeros((hk, cap, d), self.dtype, &self.device)?))
    }

    /// One decoding step for B beams. `ids`: each beam's newest token;
    /// `paths`: each beam's pool slots in order, the last being the slot for
    /// this token (= cache.used + b). Returns logits (B, vocab) f32.
    pub fn beam_step(&self, ids: &[u32], paths: &[Vec<usize>], cache: &mut BeamCache) -> Result<Tensor> {
        let b = ids.len();
        if cache.used + b > cache.cap {
            let grow = cache.cap.max(b);
            for p in cache.pool.iter_mut() {
                let (zk, zv) = self.empty_pool(grow)?;
                *p = (Tensor::cat(&[&p.0, &zk], 2)?, Tensor::cat(&[&p.1, &zv], 1)?);
            }
            cache.cap += grow;
        }
        let (h, hk, d) = (self.cfg.heads, self.cfg.kv_heads, self.cfg.head_dim);
        let g = h / hk;
        let pl = cache.prompt_len;
        let cap = cache.cap;
        // additive mask over [prompt | pool] for each (beam, group) row
        let mut mask = vec![f32::NEG_INFINITY; b * (pl + cap)];
        for (bi, path) in paths.iter().enumerate() {
            let row = &mut mask[bi * (pl + cap)..(bi + 1) * (pl + cap)];
            row[..pl].iter_mut().for_each(|m| *m = 0.0);
            for &s in path {
                row[pl + s] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask, (b, 1, pl + cap), &self.device)?
            .broadcast_as((b, g, pl + cap))?
            .reshape((1, b * g, pl + cap))?
            .to_dtype(DType::F32)?;
        let (cos, sin) = self.rope(pl + cache.used / b.max(1), 1)?;
        let mut x = self.embed_ids(&ids.iter().map(|&t| vec![t]).collect::<Vec<_>>())?.to_dtype(self.dtype)?;
        let scale = 1.0 / (d as f64).sqrt();
        for (i, ly) in self.layers.iter().enumerate() {
            let hn = ly.input_norm.forward(&x)?;
            let q = ly.q.forward(&hn)?.reshape((b, 1, h, d))?;
            let k = ly.k.forward(&hn)?.reshape((b, 1, hk, d))?;
            let v = ly.v.forward(&hn)?.reshape((b, hk, d))?;
            let q = Self::apply_rope(&ly.q_norm.forward(&q)?.transpose(1, 2)?, &cos, &sin)?; // (b, h, 1, d)
            let k = Self::apply_rope(&ly.k_norm.forward(&k)?.transpose(1, 2)?, &cos, &sin)?; // (b, hk, 1, d)
            // write this step's keys/values into slots used..used+b
            let (pk, pv) = &cache.pool[i];
            pk.slice_set(&k.squeeze(2)?.permute((1, 2, 0))?.contiguous()?, 2, cache.used)?; // (hk, d, b)
            pv.slice_set(&v.transpose(0, 1)?.contiguous()?, 1, cache.used)?; // (hk, b, d)
            // queries grouped by kv head: (hk, b*g, d)
            let qg = q.squeeze(2)?.reshape((b, hk, g, d))?.permute((1, 0, 2, 3))?.reshape((hk, b * g, d))?.contiguous()?;
            let (kpt, vp) = &cache.prompt[i];
            let sp = qg.matmul(kpt)?;
            let sg = qg.matmul(pk)?;
            let s = (Tensor::cat(&[&sp, &sg], 2)?.to_dtype(DType::F32)? * scale)?.broadcast_add(&mask)?;
            let p = candle_nn::ops::softmax_last_dim(&s)?.to_dtype(self.dtype)?;
            let o = (p.narrow(2, 0, pl)?.contiguous()?.matmul(vp)? + p.narrow(2, pl, cap)?.contiguous()?.matmul(pv)?)?;
            let o = o.reshape((hk, b, g, d))?.permute((1, 0, 2, 3))?.reshape((b, 1, h * d))?;
            x = (x + ly.o.forward(&o)?)?;
            let hn = ly.post_norm.forward(&x)?;
            let m = ly.down.forward(&(ly.gate.forward(&hn)?.silu()? * ly.up.forward(&hn)?)?)?;
            x = (x + m)?;
        }
        cache.used += b;
        Ok(self.lm_head.forward(&self.norm.forward(&x)?)?.to_dtype(DType::F32)?.squeeze(1)?)
    }
}
