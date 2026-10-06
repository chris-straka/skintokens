//! Token generation as upstream runs it: Hugging Face transformers 5.18
//! `generate` with `num_beams=10, do_sample=True` (beam sampling), its
//! logits processors (repetition penalty, then TokenRig's vocabulary
//! switch, then temperature / top-k / top-p) applied to log-probabilities,
//! and the same beam bookkeeping (finished-beam pool, length penalty 1,
//! early-stop heuristic). Greedy decoding is here for the deterministic
//! parity check.

use anyhow::Result;
use candle_core::Tensor;
use skintokens_nn::Rng;

use crate::qwen::Qwen3;
use crate::tokenizer;

#[derive(Clone, Debug)]
pub struct GenConfig {
    pub num_beams: usize,
    pub do_sample: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    /// Upstream passes max_length=2048 with inputs_embeds, which leaves
    /// 2048 - prompt length new tokens.
    pub max_length: usize,
    pub max_new_tokens: Option<usize>,
}

impl Default for GenConfig {
    /// demo.py's defaults (SKINTOKENS_NUM_BEAMS / SKINTOKENS_GREEDY for experiments).
    fn default() -> Self {
        let env = |k: &str| std::env::var(k).ok();
        GenConfig {
            num_beams: env("SKINTOKENS_NUM_BEAMS").and_then(|v| v.parse().ok()).unwrap_or(10),
            do_sample: env("SKINTOKENS_NO_SAMPLE").is_none(),
            top_k: 5,
            top_p: 0.95,
            temperature: 1.0,
            repetition_penalty: 2.0,
            max_length: 2048,
            max_new_tokens: None,
        }
    }
}

/// TokenRig's VocabSwitchingLogitsProcessor: the skeleton grammar until the
/// skeleton's eos, then only SkinTokens, then the LM eos once
/// `(len - eos_pos) == joints * tokens_per_skin` (note: that is one skin
/// token short of joints * tokens_per_skin; upstream behaves so and the
/// last code of the last joint is then the eos id, kept for parity).
pub struct Grammar {
    pub init: Vec<u32>,
    pub eos_lm: u32,
    pub tokens_per_skin: usize,
}

impl Grammar {
    pub fn apply(&self, generated: &[u32], scores: &mut [f32]) {
        let mut seq = self.init.clone();
        seq.extend_from_slice(generated);
        let ninf = f32::NEG_INFINITY;
        let v = scores.len();
        let mut mask = vec![ninf; v];
        if let Some(w) = seq.iter().position(|&t| t == tokenizer::EOS) {
            for m in mask.iter_mut().skip(tokenizer::EOS as usize) {
                *m = 0.0;
            }
            let j = tokenizer::bones_in_sequence(&seq);
            if seq.len() - w == j * self.tokens_per_skin {
                mask.iter_mut().for_each(|m| *m = ninf);
                mask[self.eos_lm as usize] = 0.0;
            } else {
                mask[self.eos_lm as usize] = ninf;
            }
        } else {
            for t in tokenizer::next_possible(&seq) {
                mask[t as usize] = 0.0;
            }
        }
        for (s, m) in scores.iter_mut().zip(mask) {
            *s += m;
        }
    }
}

fn log_softmax(x: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let s: f32 = x.iter().map(|&v| (v - m).exp()).sum();
    let l = m + s.ln();
    x.iter_mut().for_each(|v| *v -= l);
}

fn repetition_penalty(generated: &[u32], scores: &mut [f32], p: f32) {
    if p == 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &t in generated {
        if seen.insert(t) {
            let s = &mut scores[t as usize];
            *s = if *s < 0.0 { *s * p } else { *s / p };
        }
    }
}

fn top_k(scores: &mut [f32], k: usize) {
    if k == 0 || k >= scores.len() {
        return;
    }
    let mut v: Vec<f32> = scores.to_vec();
    v.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let thr = v[k - 1];
    scores.iter_mut().for_each(|s| {
        if *s < thr {
            *s = f32::NEG_INFINITY
        }
    });
}

fn top_p(scores: &mut [f32], p: f32, min_keep: usize) {
    if p >= 1.0 {
        return;
    }
    let mut finite: Vec<(f32, usize)> = scores.iter().cloned().enumerate().filter(|(_, s)| s.is_finite()).map(|(i, s)| (s, i)).collect();
    // ascending; torch.sort is not stable but ties only reorder equal values
    finite.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    let m = finite.iter().map(|x| x.0).fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = finite.iter().map(|x| (x.0 - m).exp()).sum();
    let n = finite.len();
    let mut cum = 0f32;
    for (k, &(s, i)) in finite.iter().enumerate() {
        cum += (s - m).exp() / z;
        if cum <= 1.0 - p && k + min_keep < n {
            scores[i] = f32::NEG_INFINITY;
        }
    }
}

/// Processors in transformers' order for this config.
fn process(cfg: &GenConfig, g: &Grammar, generated: &[u32], scores: &mut [f32], min_keep: usize) {
    repetition_penalty(generated, scores, cfg.repetition_penalty);
    g.apply(generated, scores);
    if cfg.do_sample {
        if cfg.temperature != 1.0 {
            scores.iter_mut().for_each(|s| *s /= cfg.temperature);
        }
        top_k(scores, cfg.top_k.max(min_keep));
        top_p(scores, cfg.top_p, min_keep);
    }
}

fn row(t: &Tensor, i: usize) -> Result<Vec<f32>> {
    Ok(t.get(i)?.to_vec1()?)
}

/// Greedy decoding (num_beams=1, no sampling): processors on raw logits,
/// argmax. Returns the generated tokens (eos included).
pub fn greedy(model: &Qwen3, prompt: &Tensor, cfg: &GenConfig, g: &Grammar) -> Result<Vec<u32>> {
    let mut cache = model.new_cache();
    let mut logits = model.forward(prompt, &mut cache, false)?;
    let max_new = cfg.max_new_tokens.unwrap_or(cfg.max_length.saturating_sub(prompt.dim(1)?));
    let mut out = vec![];
    for _ in 0..max_new {
        let mut s = row(&logits, 0)?;
        process(cfg, g, &out, &mut s, 1);
        let (t, _) = s.iter().enumerate().fold((0, f32::NEG_INFINITY), |a, (i, &v)| if v > a.1 { (i, v) } else { a });
        out.push(t as u32);
        if t as u32 == g.eos_lm {
            break;
        }
        let e = model.embed_ids(&[vec![t as u32]])?;
        logits = model.forward(&e, &mut cache, false)?;
    }
    Ok(out)
}

/// Beam search (sampling or not) exactly as transformers 5.18
/// `_beam_search` for batch size 1, decoder-only, prompt given as
/// embeddings (so the generated ids start empty).
pub fn beam_search(model: &Qwen3, prompt: &Tensor, cfg: &GenConfig, g: &Grammar, rng: &mut Rng) -> Result<Vec<u32>> {
    let nb = cfg.num_beams;
    let keep = 2 * nb; // max(2, 1 + n_eos) * num_beams with one eos id
    let max_length = match cfg.max_new_tokens {
        Some(n) => n,
        None => cfg.max_length.saturating_sub(prompt.dim(1)?),
    };
    let min_keep = 2; // n_eos + 1 for beam methods
    let lp = 1.0f32; // length_penalty

    let mut cache = model.new_cache();
    let first = model.forward(prompt, &mut cache, false)?;
    let mut cache = model.beam_cache(&cache, nb * 256)?;
    // pool slots of each running beam's generated tokens
    let mut paths: Vec<Vec<usize>> = vec![vec![]; nb];
    let first_row = row(&first, 0)?;
    let vocab = first_row.len();

    let mut running: Vec<Vec<u32>> = vec![vec![]; nb];
    let mut running_scores = vec![-1e9f32; nb];
    running_scores[0] = 0.0;
    let mut finished: Vec<Vec<u32>> = vec![vec![]; nb];
    let mut finished_scores = vec![-1e9f32; nb];
    let mut is_finished = vec![false; nb];
    let mut heuristic_unsatisfied = true;
    let mut cur_len = 0usize;
    let mut rows: Vec<Vec<f32>> = vec![first_row; nb];

    let prof = std::env::var("SKINTOKENS_PROFILE").is_ok();
    let (mut t_cpu, mut t_gpu, mut t_cache) = (0f64, 0f64, 0f64);
    loop {
        let tc = std::time::Instant::now();
        // b. log-probs, processors, accumulate
        let mut acc = Vec::with_capacity(nb * vocab);
        for (b, r) in rows.iter_mut().enumerate() {
            log_softmax(r);
            process(cfg, g, &running[b], r, min_keep);
            acc.extend(r.iter().map(|&x| x + running_scores[b]));
        }
        // c. top-K continuations (multinomial without replacement = exponential race)
        let cand: Vec<usize> = if cfg.do_sample {
            let m = acc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut keys: Vec<(f64, usize)> = acc
                .iter()
                .enumerate()
                .map(|(i, &a)| {
                    let p = ((a - m) as f64).exp() as f32; // f32 softmax underflow -> zero probability
                    let k = if p > 0.0 { (a - m) as f64 - rng.exponential().ln() } else { f64::NEG_INFINITY };
                    (k, i)
                })
                .collect();
            keys.select_nth_unstable_by(keep - 1, |x, y| y.0.partial_cmp(&x.0).unwrap().then(x.1.cmp(&y.1)));
            keys.truncate(keep);
            keys.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap().then(x.1.cmp(&y.1)));
            keys.into_iter().map(|x| x.1).collect()
        } else {
            let mut idx: Vec<usize> = (0..acc.len()).collect();
            idx.select_nth_unstable_by(keep - 1, |&x, &y| acc[y].partial_cmp(&acc[x]).unwrap().then(x.cmp(&y)));
            idx.truncate(keep);
            idx.sort_by(|&x, &y| acc[y].partial_cmp(&acc[x]).unwrap().then(x.cmp(&y)));
            idx
        };
        let cand_scores: Vec<f32> = cand.iter().map(|&i| acc[i]).collect();
        let cand_beam: Vec<usize> = cand.iter().map(|&i| i / vocab).collect();
        let cand_seq: Vec<Vec<u32>> = cand
            .iter()
            .map(|&i| {
                let mut s = running[i / vocab].clone();
                s.push((i % vocab) as u32);
                s
            })
            .collect();
        // d. stopping criteria per candidate
        let hits: Vec<bool> = cand_seq.iter().map(|s| *s.last().unwrap() == g.eos_lm || s.len() >= max_length).collect();
        // e. next running beams: best non-finished
        let run_scores: Vec<f32> = cand_scores.iter().zip(&hits).map(|(&s, &h)| s + if h { -1e9 } else { 0.0 }).collect();
        let mut order: Vec<usize> = (0..keep).collect();
        order.sort_by(|&x, &y| run_scores[y].partial_cmp(&run_scores[x]).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
        let next: Vec<usize> = order[..nb].to_vec();
        // f. finished pool
        let denom = ((cur_len + 1) as f32).powf(lp);
        let mut merged: Vec<(f32, Vec<u32>, bool)> = (0..nb).map(|b| (finished_scores[b], finished[b].clone(), is_finished[b])).collect();
        for k in 0..keep {
            let just = hits[k] && k < nb;
            let mut s = cand_scores[k] / denom;
            if !heuristic_unsatisfied {
                s += -1e9;
            }
            if !just {
                s += -1e9;
            }
            merged.push((s, cand_seq[k].clone(), just));
        }
        let mut morder: Vec<usize> = (0..merged.len()).collect();
        morder.sort_by(|&x, &y| merged[y].0.partial_cmp(&merged[x].0).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(&y)));
        let mut nf = Vec::with_capacity(nb);
        for &i in &morder[..nb] {
            nf.push(merged[i].clone());
        }
        finished_scores = nf.iter().map(|x| x.0).collect();
        finished = nf.iter().map(|x| x.1.clone()).collect();
        is_finished = nf.iter().map(|x| x.2).collect();

        running = next.iter().map(|&k| cand_seq[k].clone()).collect();
        running_scores = next.iter().map(|&k| run_scores[k]).collect();
        let src: Vec<usize> = next.iter().map(|&k| cand_beam[k]).collect();
        cur_len += 1;

        // early-stop heuristic (early_stopping=False)
        let best = running_scores[0] / (cur_len as f32).powf(lp);
        let min_fin = finished_scores.iter().cloned().fold(f32::INFINITY, f32::min);
        heuristic_unsatisfied = heuristic_unsatisfied && is_finished.iter().any(|&f| best > if f { min_fin } else { -1e9 });
        let all_hit = hits.iter().all(|&h| h);
        if !heuristic_unsatisfied || all_hit {
            break;
        }
        t_cpu += tc.elapsed().as_secs_f64();
        // next forward pass for the running beams
        let tk = std::time::Instant::now();
        let base = cache.used;
        paths = src.iter().enumerate().map(|(bi, &s)| {
            let mut p = paths[s].clone();
            p.push(base + bi);
            p
        }).collect();
        t_cache += tk.elapsed().as_secs_f64();
        let tg = std::time::Instant::now();
        let last: Vec<u32> = running.iter().map(|s| *s.last().unwrap()).collect();
        let logits = model.beam_step(&last, &paths, &mut cache)?;
        rows = (0..nb).map(|b| row(&logits, b)).collect::<Result<_>>()?;
        t_gpu += tg.elapsed().as_secs_f64();
    }
    if prof {
        eprintln!("  beam search: cpu {t_cpu:.1} s, cache reorder {t_cache:.1} s, forward+readback {t_gpu:.1} s, {cur_len} steps");
    }
    Ok(finished.into_iter().next().unwrap_or_default())
}
