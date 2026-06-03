//! Attention kernels: standard MHA (Qwen, Llama, etc.) and Multi-head Latent Attention
//! (DeepSeek-V2-Lite, V2, V3). MLA carries a compressed KV cache that decompresses on read.
//!
//! `mha_decode_step` is the CPU reference used by the non-TCB fallback (temp>0 sampling,
//! `DISMANTLE_FORCE_CPU=1`) and the `DISMANTLE_QWEN_ATTN_CAPTURE=1` oracle.
//! The default fast path (`forward_token_greedy_tcb`) dispatches the GPU kernel
//! `mha_decode_f32` via `kernels::mha_decode_f32_tcb`.
//! Metal sources: `shaders/mha.metal` (MHA), `shaders/attn.metal` (MLA flash-decode).

use crate::kernels::{logit_softcap_inplace, softmax_inplace};
use crate::Result;

/// Standard multi-head attention for one new token (decode step).
///
/// Inputs:
///   - q:        (n_heads, head_dim) — query for the just-produced token
///   - k_cache:  (seq_len, n_kv_heads, head_dim) — past keys
///   - v_cache:  (seq_len, n_kv_heads, head_dim) — past values
///   - n_heads, n_kv_heads: ratio determines GQA grouping
///   - head_dim
/// Output:
///   - out:      (n_heads, head_dim)
pub fn mha_decode_step(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    out: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);

    let group_size = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0.0f32; seq_len];
    for h in 0..n_heads {
        let kv_h = h / group_size;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        // Compute attention scores against every position.
        for t in 0..seq_len {
            let off = t * (n_kv_heads * head_dim) + kv_h * head_dim;
            let k = &k_cache[off..off + head_dim];
            let mut s = 0.0f32;
            for i in 0..head_dim {
                s += q_head[i] * k[i];
            }
            scores[t] = s * scale;
        }
        softmax_inplace(&mut scores);
        // Weighted sum of values.
        let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_head.iter_mut() {
            *v = 0.0;
        }
        for t in 0..seq_len {
            let off = t * (n_kv_heads * head_dim) + kv_h * head_dim;
            let val = &v_cache[off..off + head_dim];
            let w = scores[t];
            for i in 0..head_dim {
                out_head[i] += w * val[i];
            }
        }
    }
    Ok(())
}

/// Recompute the per-head post-softmax attention distributions for one
/// decode/prefill query, matching [`mha_decode_step`]'s arithmetic exactly.
///
/// **Oracle-only.** This is *not* on any production path — it is called by
/// the L1.1 attention-mass capture instrument
/// ([`crate::stateful::attn_capture`]) only when
/// `DISMANTLE_QWEN_ATTN_CAPTURE=1`, to observe the same softmax weights
/// `mha_decode_step` materializes internally and discards. Returns a
/// `Vec` of `n_heads` distributions, each length `seq_len`, in
/// retained-position order (index 0 == oldest cached position).
pub fn mha_decode_step_weights(
    q: &[f32],
    k_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
) -> Vec<Vec<f32>> {
    let group_size = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let kv_h = h / group_size;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; seq_len];
        for t in 0..seq_len {
            let off = t * (n_kv_heads * head_dim) + kv_h * head_dim;
            let k = &k_cache[off..off + head_dim];
            let mut s = 0.0f32;
            for i in 0..head_dim {
                s += q_head[i] * k[i];
            }
            scores[t] = s * scale;
        }
        softmax_inplace(&mut scores);
        out.push(scores);
    }
    out
}

/// Gemma-2 variant of [`mha_decode_step`] with a configurable attention
/// `scale` and optional attention-logit soft-capping.
///
/// Gemma-2 differs from a vanilla GQA decode step in two ways:
///   - the score scale is `1/sqrt(query_pre_attn_scalar)`, which is not
///     always `1/sqrt(head_dim)` (they coincide for Gemma-2-2B where
///     query_pre_attn_scalar == head_dim == 256, but diverge on 9B/27B)
///   - scores are soft-capped (`cap·tanh(score/cap)`, cap≈50) *after*
///     scaling and *before* softmax
///
/// `attn_softcap <= 0` disables capping, recovering a plain scaled MHA
/// step. Passing `scale = 1/sqrt(head_dim)` and `attn_softcap = 0` makes
/// this bit-equivalent to [`mha_decode_step`].
#[allow(clippy::too_many_arguments)]
pub fn mha_decode_step_gemma(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    scale: f32,
    attn_softcap: f32,
    out: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(q.len(), n_heads * head_dim);
    debug_assert_eq!(out.len(), n_heads * head_dim);

    let group_size = n_heads / n_kv_heads;
    let mut scores = vec![0.0f32; seq_len];
    for h in 0..n_heads {
        let kv_h = h / group_size;
        let q_head = &q[h * head_dim..(h + 1) * head_dim];
        for t in 0..seq_len {
            let off = t * (n_kv_heads * head_dim) + kv_h * head_dim;
            let k = &k_cache[off..off + head_dim];
            let mut s = 0.0f32;
            for i in 0..head_dim {
                s += q_head[i] * k[i];
            }
            scores[t] = s * scale;
        }
        // Soft-cap the scaled scores before softmax (no-op when cap<=0).
        logit_softcap_inplace(&mut scores, attn_softcap);
        softmax_inplace(&mut scores);
        let out_head = &mut out[h * head_dim..(h + 1) * head_dim];
        for v in out_head.iter_mut() {
            *v = 0.0;
        }
        for t in 0..seq_len {
            let off = t * (n_kv_heads * head_dim) + kv_h * head_dim;
            let val = &v_cache[off..off + head_dim];
            let w = scores[t];
            for i in 0..head_dim {
                out_head[i] += w * val[i];
            }
        }
    }
    Ok(())
}

/// Multi-head Latent Attention (DeepSeek-V2 family) — reference path.
///
/// MLA compresses K and V into a single low-rank `c_kv` latent on
/// prefill; on read, the kernel reconstructs head-shape K/V via the
/// upproject matrices `w_uk`, `w_uv`. Phase 0 reference reconstructs
/// in fp32 then runs the same softmax-attention as MHA.
///
/// Inputs:
///   - q:        (n_heads, qk_rope_head_dim + qk_nope_head_dim)
///   - c_kv:     (seq_len, kv_lora_rank) — latent KV cache
///   - k_pe:     (seq_len, qk_rope_head_dim) — separate rope-positional K
///   - w_uk, w_uv: upproject matrices, shape (n_heads, head_dim, kv_lora_rank)
pub fn mla_decode_step(
    q: &[f32],
    c_kv: &[f32],
    k_pe: &[f32],
    w_uk: &[f32],
    w_uv: &[f32],
    n_heads: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    kv_lora_rank: usize,
    seq_len: usize,
    out: &mut [f32],
) -> Result<()> {
    debug_assert_eq!(q.len(), n_heads * (qk_nope_head_dim + qk_rope_head_dim));
    debug_assert_eq!(out.len(), n_heads * v_head_dim);

    let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
    let scale = 1.0 / (q_head_dim as f32).sqrt();

    let mut scores = vec![0.0f32; seq_len];
    let mut k_full = vec![0.0f32; q_head_dim];
    let mut v_full = vec![0.0f32; v_head_dim];

    for h in 0..n_heads {
        let q_head = &q[h * q_head_dim..(h + 1) * q_head_dim];

        // Per-head upproject matrices: w_uk[h] is (qk_nope_head_dim, kv_lora_rank).
        let uk_off = h * qk_nope_head_dim * kv_lora_rank;
        let uv_off = h * v_head_dim * kv_lora_rank;
        let uk = &w_uk[uk_off..uk_off + qk_nope_head_dim * kv_lora_rank];
        let uv = &w_uv[uv_off..uv_off + v_head_dim * kv_lora_rank];

        for t in 0..seq_len {
            let c = &c_kv[t * kv_lora_rank..(t + 1) * kv_lora_rank];
            // Reconstruct K's nope half: K_nope = uk @ c
            for i in 0..qk_nope_head_dim {
                let mut acc = 0.0f32;
                for j in 0..kv_lora_rank {
                    acc += uk[i * kv_lora_rank + j] * c[j];
                }
                k_full[i] = acc;
            }
            // Splice rope K (shared across heads) into the rope half.
            let kpe = &k_pe[t * qk_rope_head_dim..(t + 1) * qk_rope_head_dim];
            for i in 0..qk_rope_head_dim {
                k_full[qk_nope_head_dim + i] = kpe[i];
            }
            // dot(q_head, k_full)
            let mut s = 0.0f32;
            for i in 0..q_head_dim {
                s += q_head[i] * k_full[i];
            }
            scores[t] = s * scale;
        }
        softmax_inplace(&mut scores);

        let out_head = &mut out[h * v_head_dim..(h + 1) * v_head_dim];
        for v in out_head.iter_mut() {
            *v = 0.0;
        }
        for t in 0..seq_len {
            let c = &c_kv[t * kv_lora_rank..(t + 1) * kv_lora_rank];
            // Reconstruct V via uv @ c.
            for i in 0..v_head_dim {
                let mut acc = 0.0f32;
                for j in 0..kv_lora_rank {
                    acc += uv[i * kv_lora_rank + j] * c[j];
                }
                v_full[i] = acc;
            }
            let w = scores[t];
            for i in 0..v_head_dim {
                out_head[i] += w * v_full[i];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_vec(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    /// Gemma attention with cap disabled and the default scale must match
    /// the vanilla MHA step bit-for-bit (same arithmetic order).
    #[test]
    fn gemma_attn_cap_off_matches_mha() {
        let (n_heads, n_kv_heads, head_dim, seq_len) = (4, 2, 8, 5);
        let q = rng_vec(n_heads * head_dim, 1);
        let k = rng_vec(seq_len * n_kv_heads * head_dim, 2);
        let v = rng_vec(seq_len * n_kv_heads * head_dim, 3);

        let mut out_ref = vec![0.0f32; n_heads * head_dim];
        mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut out_ref).unwrap();

        let mut out_gemma = vec![0.0f32; n_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        mha_decode_step_gemma(
            &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, scale, 0.0, &mut out_gemma,
        )
        .unwrap();

        for i in 0..out_ref.len() {
            assert_eq!(
                out_ref[i].to_bits(),
                out_gemma[i].to_bits(),
                "i={i}: mha={} gemma={}",
                out_ref[i],
                out_gemma[i]
            );
        }
    }

    /// With a finite cap, attention output stays a valid convex
    /// combination of the value rows (softmax weights still sum to 1),
    /// so every output element is within the min/max of that head's V.
    #[test]
    fn gemma_attn_softcap_is_convex() {
        let (n_heads, n_kv_heads, head_dim, seq_len) = (2, 1, 4, 6);
        let q = rng_vec(n_heads * head_dim, 7);
        let k = rng_vec(seq_len * n_kv_heads * head_dim, 8);
        let v = rng_vec(seq_len * n_kv_heads * head_dim, 9);
        let mut out = vec![0.0f32; n_heads * head_dim];
        let scale = 1.0 / (head_dim as f32).sqrt();
        mha_decode_step_gemma(
            &q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, scale, 50.0, &mut out,
        )
        .unwrap();
        for d in 0..head_dim {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for t in 0..seq_len {
                let val = v[t * head_dim + d];
                lo = lo.min(val);
                hi = hi.max(val);
            }
            for h in 0..n_heads {
                let o = out[h * head_dim + d];
                assert!(o >= lo - 1e-4 && o <= hi + 1e-4, "out {o} not in [{lo},{hi}]");
            }
        }
    }

    /// The L1.1 oracle's `mha_decode_step_weights` must reproduce exactly
    /// the post-softmax weights `mha_decode_step` uses internally: applying
    /// them to V by hand must reconstruct `mha_decode_step`'s output
    /// bit-for-bit. (Same scale, same softmax, same arithmetic order.)
    #[test]
    fn weights_reconstruct_mha_output() {
        let (n_heads, n_kv_heads, head_dim, seq_len) = (4, 2, 8, 7);
        let q = rng_vec(n_heads * head_dim, 11);
        let k = rng_vec(seq_len * n_kv_heads * head_dim, 12);
        let v = rng_vec(seq_len * n_kv_heads * head_dim, 13);

        let mut out_ref = vec![0.0f32; n_heads * head_dim];
        mha_decode_step(&q, &k, &v, n_heads, n_kv_heads, head_dim, seq_len, &mut out_ref).unwrap();

        let w = mha_decode_step_weights(&q, &k, n_heads, n_kv_heads, head_dim, seq_len);
        assert_eq!(w.len(), n_heads);
        let group_size = n_heads / n_kv_heads;
        let mut out_w = vec![0.0f32; n_heads * head_dim];
        for h in 0..n_heads {
            assert_eq!(w[h].len(), seq_len);
            // weights are a probability distribution
            let sum: f32 = w[h].iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "head {h} weights sum {sum}");
            let kv_h = h / group_size;
            let out_head = &mut out_w[h * head_dim..(h + 1) * head_dim];
            for t in 0..seq_len {
                let off = t * (n_kv_heads * head_dim) + kv_h * head_dim;
                let val = &v[off..off + head_dim];
                let ww = w[h][t];
                for i in 0..head_dim {
                    out_head[i] += ww * val[i];
                }
            }
        }
        // Same arithmetic order as mha_decode_step's Phase 4 → bit-identical.
        for i in 0..out_ref.len() {
            assert_eq!(
                out_ref[i].to_bits(),
                out_w[i].to_bits(),
                "i={i}: mha={} weights-applied={}",
                out_ref[i],
                out_w[i]
            );
        }
    }
}
