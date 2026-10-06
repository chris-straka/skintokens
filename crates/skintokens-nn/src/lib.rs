//! Shared tensor building blocks for the SkinTokens port.
//!
//! Everything here is plain MIT code: checkpoint access, linear and norm
//! layers, scaled dot-product attention (fused Metal kernel when it
//! applies, a chunked matmul-softmax path otherwise) and farthest point
//! sampling as upstream's `src/model/utils.py` does it.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor, D};

/// Named tensors from a PyTorch Lightning checkpoint (or any map), handed
/// out in the compute dtype on the compute device.
#[derive(Clone)]
pub struct Weights {
    map: Arc<HashMap<String, Tensor>>,
    prefix: String,
    pub dtype: DType,
    pub device: Device,
}

impl Weights {
    /// Reads `state_dict` from a `.ckpt` (torch zip pickle) without Python.
    /// `_orig_mod.` (torch.compile wrappers) is dropped from the names.
    pub fn load_ckpt(path: &std::path::Path, dtype: DType, device: &Device) -> Result<Self> {
        let all = candle_core::pickle::read_all_with_key(path, Some("state_dict"))
            .with_context(|| format!("reading {}", path.display()))?;
        let map = all.into_iter().map(|(k, v)| (k.replace("_orig_mod.", ""), v)).collect();
        Ok(Self::from_map(map, dtype, device))
    }

    pub fn from_map(map: HashMap<String, Tensor>, dtype: DType, device: &Device) -> Self {
        Self { map: Arc::new(map), prefix: String::new(), dtype, device: device.clone() }
    }

    pub fn pp(&self, s: impl AsRef<str>) -> Self {
        let mut w = self.clone();
        w.prefix = self.full(s.as_ref());
        w
    }

    fn full(&self, name: &str) -> String {
        if self.prefix.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.prefix, name)
        }
    }

    pub fn has(&self, name: &str) -> bool {
        self.map.contains_key(&self.full(name))
    }

    pub fn names(&self) -> Vec<String> {
        let p = if self.prefix.is_empty() { String::new() } else { format!("{}.", self.prefix) };
        self.map.keys().filter(|k| k.starts_with(&p)).cloned().collect()
    }

    pub fn get(&self, name: &str) -> Result<Tensor> {
        let full = self.full(name);
        let t = self.map.get(&full).ok_or_else(|| anyhow!("checkpoint has no tensor {full}"))?;
        Ok(t.to_dtype(self.dtype)?.to_device(&self.device)?)
    }

    pub fn get_shape(&self, name: &str, shape: &[usize]) -> Result<Tensor> {
        let t = self.get(name)?;
        if t.dims() != shape {
            return Err(anyhow!("{}: expected shape {:?}, got {:?}", self.full(name), shape, t.dims()));
        }
        Ok(t)
    }
}

/// `nn.Linear`: y = x W^T + b over the last dimension.
/// The weight is kept transposed and contiguous, (in, out): Metal matmul
/// against a transposed view gave wrong results for a small inner
/// dimension (FSQ's 5 -> 512 projection).
#[derive(Clone)]
pub struct Linear {
    pub wt: Tensor,
    pub b: Option<Tensor>,
}

impl Linear {
    pub fn new(w: &Tensor, b: Option<Tensor>) -> Result<Self> {
        Ok(Self { wt: w.t()?.contiguous()?, b })
    }

    pub fn load(w: &Weights, name: &str, bias: bool) -> Result<Self> {
        let ww = w.pp(name);
        Self::new(&ww.get("weight")?, if bias { Some(ww.get("bias")?) } else { None })
    }

    /// Bias if the checkpoint has one.
    pub fn load_auto(w: &Weights, name: &str) -> Result<Self> {
        let ww = w.pp(name);
        Self::load(w, name, ww.has("bias"))
    }

    pub fn out_dim(&self) -> usize {
        self.wt.dims()[1]
    }

    pub fn dtype(&self) -> DType {
        self.wt.dtype()
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let inn = *dims.last().unwrap();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let y = x.reshape((rows, inn))?.contiguous()?.matmul(&self.wt)?;
        let y = match &self.b {
            Some(b) => y.broadcast_add(b)?,
            None => y,
        };
        let mut out = dims;
        *out.last_mut().unwrap() = self.out_dim();
        Ok(y.reshape(out)?)
    }
}

/// `nn.LayerNorm` (affine), computed in f32 like torch's fp32 layer norms.
#[derive(Clone)]
pub struct LayerNorm {
    pub w: Tensor,
    pub b: Tensor,
    pub eps: f32,
}

impl LayerNorm {
    pub fn load(w: &Weights, name: &str, eps: f32) -> Result<Self> {
        let ww = w.pp(name);
        Ok(Self { w: ww.get("weight")?.to_dtype(DType::F32)?, b: ww.get("bias")?.to_dtype(DType::F32)?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let x = x.to_dtype(DType::F32)?.contiguous()?;
        let y = if std::env::var("SKINTOKENS_SLOW_LN").is_ok() {
            let mean = x.mean_keepdim(D::Minus1)?;
            let xc = x.broadcast_sub(&mean)?;
            let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
            xc.broadcast_div(&(var + self.eps as f64)?.sqrt()?)?.broadcast_mul(&self.w)?.broadcast_add(&self.b)?
        } else {
            candle_nn::ops::layer_norm(&x, &self.w, &self.b, self.eps)?
        };
        Ok(y.to_dtype(dt)?)
    }
}

/// RMSNorm with a weight (Qwen3, `nn.RMSNorm`), computed in f32.
#[derive(Clone)]
pub struct RmsNorm {
    pub w: Tensor,
    pub eps: f32,
}

impl RmsNorm {
    pub fn load(w: &Weights, name: &str, eps: f32) -> Result<Self> {
        Ok(Self { w: w.pp(name).get("weight")?.to_dtype(DType::F32)?, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dt = x.dtype();
        let y = candle_nn::ops::rms_norm(&x.to_dtype(DType::F32)?.contiguous()?, &self.w, self.eps)?;
        Ok(y.to_dtype(dt)?)
    }
}

fn use_sdpa() -> bool {
    std::env::var("SKINTOKENS_NO_SDPA").is_err()
}

/// Rows of queries per chunk on the matmul path (bounds the score matrix).
const CHUNK: usize = 4096;

/// Scaled dot-product attention. q: (B, H, Lq, D), k/v: (B, Hkv, Lk, D)
/// with H a multiple of Hkv. Returns (B, H, Lq, D). `causal` masks key j > i
/// (only used with Lq == Lk, no cache).
pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    let (_, h, lq, d) = q.dims4()?;
    let hkv = k.dims4()?.1;
    if q.device().is_metal() && use_sdpa() && matches!(d, 32 | 64 | 72 | 80 | 96 | 128 | 256) && (!causal || lq > 1) {
        let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
        return Ok(candle_nn::ops::sdpa(&q, &k, &v, None, causal, scale, 1.0)?);
    }
    let (k, v) = if hkv != h {
        let r = h / hkv;
        let rep = |t: &Tensor| -> Result<Tensor> {
            let (b, hk, l, dd) = t.dims4()?;
            Ok(t.unsqueeze(2)?.expand((b, hk, r, l, dd))?.reshape((b, hk * r, l, dd))?)
        };
        (rep(k)?, rep(v)?)
    } else {
        (k.clone(), v.clone())
    };
    let kt = k.t()?.contiguous()?;
    let v = v.contiguous()?;
    let lk = k.dims4()?.2;
    let mut outs = Vec::new();
    let mut s = 0;
    while s < lq {
        let n = CHUNK.min(lq - s);
        let qc = q.narrow(2, s, n)?.contiguous()?;
        let mut att = (qc.matmul(&kt)? * scale as f64)?;
        if causal {
            let mask: Vec<f32> = (0..n)
                .flat_map(|i| (0..lk).map(move |j| if j > s + i { f32::NEG_INFINITY } else { 0.0 }))
                .collect();
            let mask = Tensor::from_vec(mask, (n, lk), q.device())?.to_dtype(att.dtype())?;
            att = att.broadcast_add(&mask)?;
        }
        let att = candle_nn::ops::softmax_last_dim(&att.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        outs.push(att.matmul(&v)?);
        s += n;
    }
    Ok(Tensor::cat(&outs, 2)?)
}

/// Farthest point sampling as upstream `fps` (start at index 0, squared
/// distances in f32, first index of the maximum). Returns `round(n *
/// ratio)` indices into `points`.
pub fn fps(points: &[[f32; 3]], ratio: f32) -> Vec<usize> {
    let n = points.len();
    if n == 0 {
        return vec![];
    }
    let m = ((n as f64 * ratio as f64).round() as usize).clamp(1, n);
    let mut dist = vec![f32::INFINITY; n];
    let mut out = Vec::with_capacity(m);
    let mut far = 0usize;
    for _ in 0..m {
        out.push(far);
        let c = points[far];
        let mut best = (f32::NEG_INFINITY, 0usize);
        for (i, p) in points.iter().enumerate() {
            let dx = p[0] - c[0];
            let dy = p[1] - c[1];
            let dz = p[2] - c[2];
            let dd = dx * dx + dy * dy + dz * dz;
            if dd < dist[i] {
                dist[i] = dd;
            }
            if dist[i] > best.0 {
                best = (dist[i], i);
            }
        }
        far = best.1;
    }
    out
}

/// Rows of a (N, C) f32 tensor as Vec<Vec<f32>> helpers.
pub fn to_points(t: &Tensor) -> Result<Vec<[f32; 3]>> {
    let v: Vec<Vec<f32>> = t.narrow(D::Minus1, 0, 3)?.to_dtype(DType::F32)?.to_vec2()?;
    Ok(v.into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// Picks the best compute device: Metal unless SKINTOKENS_DEVICE=cpu.
pub fn pick_device() -> Result<Device> {
    match std::env::var("SKINTOKENS_DEVICE").as_deref() {
        Ok("cpu") => Ok(Device::Cpu),
        Ok("metal") | Ok("mps") | Ok("gpu") => Ok(Device::new_metal(0)?),
        _ => Ok(Device::new_metal(0).unwrap_or(Device::Cpu)),
    }
}

/// Small seeded RNG (SplitMix64) for sampling. Not NumPy's stream: the
/// Python reference is not reproducible run to run either (unseeded draws
/// in the VAE, MPS kernels), so parity is checked on fixed inputs.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform integer in [0, n).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Exponential(1) draw.
    pub fn exponential(&mut self) -> f64 {
        -(1.0 - self.uniform()).max(1e-300).ln()
    }

    /// Random permutation of 0..n.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.below(i + 1);
            v.swap(i, j);
        }
        v
    }

    /// `k` indices from 0..n: without replacement when k <= n, else with
    /// replacement (NumPy's `choice(n, k, replace=k > n)`).
    pub fn choice(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k > n {
            return (0..k).map(|_| self.below(n)).collect();
        }
        let mut v: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + self.below(n - i);
            v.swap(i, j);
        }
        v.truncate(k);
        v
    }
}

static TRACE: std::sync::Mutex<Option<Vec<(String, Tensor)>>> = std::sync::Mutex::new(None);

/// Developer tracing of intermediate activations (parity checks only).
pub fn trace_enable(on: bool) {
    *TRACE.lock().unwrap() = if on { Some(vec![]) } else { None };
}

pub fn trace(name: &str, t: &Tensor) {
    if let Some(v) = TRACE.lock().unwrap().as_mut() {
        v.push((name.to_string(), t.clone()));
    }
}

pub fn trace_take() -> Vec<(String, Tensor)> {
    TRACE.lock().unwrap().as_mut().map(std::mem::take).unwrap_or_default()
}
