//! Attention kernels.
//!
//! Two flavors: standard MHA (Qwen-MoE) and Multi-head Latent
//! Attention (DeepSeek-V2-Lite, V2, V3). MLA carries a compressed KV
//! cache that decompresses on read — no public Metal MLA kernel
//! exists; we write one in Phase 3.
//!
//! Phase 0 ships a CPU reference so the model produces coherent text
//! before any Metal attention work. The Phase 3 Metal kernels live in
//! `shaders/attn.metal`.

use crate::kernels::softmax_inplace;
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
