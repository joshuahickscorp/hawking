//! On-GPU sampling — wedge 3 (Phase 2.5).
//!
//! Phase 0 ships a CPU reference; the Metal kernels in
//! `shaders/sample.metal` arrive in Phase 2.5 and hold the same Rust
//! signatures, so the model layer is unchanged when we swap.
//!
//! Once Metal kernels land, logits never leave the GPU; only the
//! sampled token id crosses the bus. Eliminates the per-token CPU↔GPU
//! sync that dominates llama.cpp's decode loop above ~50 tok/s.

use crate::engine::SamplingParams;
use rand::Rng;
use rand_pcg::Pcg64Mcg;

pub struct Sampler {
    rng: Pcg64Mcg,
    /// rolling history of recently emitted tokens for repetition penalty
    recent: Vec<u32>,
    /// fixed window size for repetition penalty
    rep_window: usize,
}

impl Sampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Pcg64Mcg::new(seed as u128),
            recent: Vec::new(),
            rep_window: 64,
        }
    }

    pub fn record(&mut self, token: u32) {
        self.recent.push(token);
        if self.recent.len() > self.rep_window {
            let n = self.recent.len() - self.rep_window;
            self.recent.drain(0..n);
        }
    }

    /// Sample one token from logits given `params`. Mutates `logits`
    /// (temp-scaling, repetition-penalty, etc. happen in-place).
    pub fn sample(&mut self, logits: &mut [f32], params: &SamplingParams) -> u32 {
        // 1. repetition penalty
        if params.repetition_penalty != 1.0 {
            for &t in &self.recent {
                let i = t as usize;
                if i < logits.len() {
                    let v = logits[i];
                    logits[i] = if v >= 0.0 {
                        v / params.repetition_penalty
                    } else {
                        v * params.repetition_penalty
                    };
                }
            }
        }
        // 2. temperature: temp == 0 → argmax (greedy)
        if params.temperature <= 0.0 {
            return argmax(logits);
        }
        for v in logits.iter_mut() {
            *v /= params.temperature;
        }

        // 3. softmax
        let mut probs = logits.to_vec();
        softmax(&mut probs);

        // 4. top-K
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if (params.top_k as usize) > 0 && (params.top_k as usize) < indexed.len() {
            indexed.truncate(params.top_k as usize);
        }

        // 5. top-P
        if params.top_p > 0.0 && params.top_p < 1.0 {
            let mut cum = 0.0f32;
            let mut cutoff = indexed.len();
            for (k, (_, p)) in indexed.iter().enumerate() {
                cum += *p;
                if cum >= params.top_p {
                    cutoff = k + 1;
                    break;
                }
            }
            indexed.truncate(cutoff);
        }

        // 6. renormalize and draw
        let total: f32 = indexed.iter().map(|(_, p)| *p).sum();
        let r: f32 = self.rng.gen::<f32>() * total;
        let mut acc = 0.0f32;
        for (idx, p) in &indexed {
            acc += *p;
            if r <= acc {
                return *idx as u32;
            }
        }
        indexed.last().map(|(i, _)| *i as u32).unwrap_or(0)
    }
}

fn argmax(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > bv {
            best = i;
            bv = v;
        }
    }
    best as u32
}

fn softmax(xs: &mut [f32]) {
    let mut m = f32::NEG_INFINITY;
    for &v in xs.iter() {
        if v > m {
            m = v;
        }
    }
    let mut sum = 0.0f32;
    for v in xs.iter_mut() {
        *v = (*v - m).exp();
        sum += *v;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for v in xs.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_argmax() {
        let mut s = Sampler::new(42);
        let mut logits = vec![1.0, 5.0, 2.0, 0.0];
        let p = SamplingParams {
            temperature: 0.0,
            ..SamplingParams::default()
        };
        assert_eq!(s.sample(&mut logits, &p), 1);
    }
}
