//! Layer-major BF16 SOURCE activation capture for Qwen3-Coder-30B-A3B.
//!
//! Resource contract (intentionally distinct from the co-resident memory gate):
//!
//! * Capture does **not** resident-load the 56.9 GiB BF16 source.
//! * Loop is inverted: for each layer, range-read that layer's weights, push
//!   every probe token through the layer, record routes + retained hiddens,
//!   then free the layer weights.
//! * Working set is ~one layer of MoE (~1.1–2.2 GiB depending on widen) plus
//!   the full residual stream for all tokens (0.67 GiB at the sealed corpus).
//!
//! Captured router inputs are the post-attention RMSNorm vectors (same surface
//! as the existing complete-binary all-layer route capture). Output layout is
//! schema-compatible with that capture so the activation-weighted SVD repack
//! can consume it without changes.

use crate::artifact::widen_native;
use crate::attn::mha_decode_step;
use crate::kernels::{add_inplace, argmax_f32, rmsnorm, silu_mul, softmax_inplace};
use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

pub const QWEN30_LAYERS: usize = 48;
pub const QWEN30_HIDDEN: usize = 2048;
pub const QWEN30_HEADS: usize = 32;
pub const QWEN30_KV_HEADS: usize = 4;
pub const QWEN30_HEAD_DIM: usize = 128;
pub const QWEN30_EXPERTS: usize = 128;
pub const QWEN30_TOP_K: usize = 8;
pub const QWEN30_MOE_INTERMEDIATE: usize = 768;
pub const QWEN30_VOCAB: usize = 151_936;
pub const QWEN30_ROPE_THETA: f32 = 10_000_000.0;
pub const QWEN30_RMS_EPS: f32 = 1e-6;
/// Soft upper bound declared by this contract (single-digit GiB). Exceeding
/// this means the implementation has effectively resident-loaded the source.
pub const STREAMED_PEAK_RSS_HARD_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;
/// Approximate per-layer MoE BF16 payload (attention + 128 experts).
pub const PER_LAYER_MOE_BF16_BYTES: u64 = 1_200_000_000;

fn model_err(msg: impl Into<String>) -> Error {
    Error::Model(msg.into())
}

#[inline]
fn bf16_le_to_f32(bytes: &[u8]) -> f32 {
    debug_assert!(bytes.len() >= 2);
    f32::from_bits((u16::from_le_bytes([bytes[0], bytes[1]]) as u32) << 16)
}

/// BF16-LE → f32 widen of a row-major `[rows, cols]` matrix into a fresh buffer.
pub fn widen_bf16_mat(weight_le: &[u8], rows: usize, cols: usize) -> Result<Vec<f32>> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| model_err("widen_bf16_mat size overflow"))?;
    let mut out = vec![0.0f32; n];
    widen_bf16_into(weight_le, rows, cols, &mut out)?;
    Ok(out)
}

/// BF16-LE → f32 widen into a caller-owned buffer (avoids per-call alloc when reused).
pub fn widen_bf16_into(
    weight_le: &[u8],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) -> Result<()> {
    let n = rows
        .checked_mul(cols)
        .ok_or_else(|| model_err("widen_bf16_into size overflow"))?;
    let expect = n
        .checked_mul(2)
        .ok_or_else(|| model_err("widen_bf16_into byte overflow"))?;
    if weight_le.len() < expect {
        return Err(model_err(format!(
            "widen_bf16_into weight bytes {} < {expect}",
            weight_le.len()
        )));
    }
    if out.len() < n {
        return Err(model_err(format!(
            "widen_bf16_into out len {} < {n}",
            out.len()
        )));
    }
    // Memory-bandwidth bound. Parallelise only for large tensors; avoid
    // thread-spawn storms when widening 128 experts × 3 mats per layer.
    const PARALLEL_THRESHOLD: usize = 512 * 1024; // elements
    if n < PARALLEL_THRESHOLD {
        for i in 0..n {
            let b = i * 2;
            out[i] = bf16_le_to_f32(&weight_le[b..b + 2]);
        }
        return Ok(());
    }
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .clamp(1, 8);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out[..n].chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            let w = weight_le;
            scope.spawn(move || {
                for (i, o) in out_chunk.iter_mut().enumerate() {
                    let idx = base + i;
                    let b = idx * 2;
                    *o = bf16_le_to_f32(&w[b..b + 2]);
                }
            });
        }
    });
    Ok(())
}

/// Run one expert's gate/up/down as batched GEMMs against gathered token rows.
///
/// Widens BF16 → f32 for this expert only (then drops), so peak resident expert
/// weights stay ~18 MiB rather than ~2.3 GiB for all 128 experts at once.
fn expert_batched_moe(
    expert: &ExpertWeights,
    members: &[(usize, f32)],
    all_router_in: &[f32],
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
    // Reusable scratch (sized for max members across callers).
    x_g: &mut [f32],
    gu_out: &mut [f32],
    act: &mut [f32],
    down: &mut [f32],
    w_gu: &mut [f32],
    w_down: &mut [f32],
) -> Result<()> {
    let n = members.len();
    if n == 0 {
        return Ok(());
    }
    let x_g = &mut x_g[..n * h];
    for (i, &(t, _)) in members.iter().enumerate() {
        x_g[i * h..(i + 1) * h].copy_from_slice(&all_router_in[t * h..(t + 1) * h]);
    }

    // Stack gate+up into one (2*inter) × h matrix → one GEMM instead of two.
    let gu_rows = 2 * inter;
    let w_gu = &mut w_gu[..gu_rows * h];
    widen_bf16_into(&expert.gate, inter, h, &mut w_gu[..inter * h])?;
    widen_bf16_into(&expert.up, inter, h, &mut w_gu[inter * h..gu_rows * h])?;
    let gu_out = &mut gu_out[..n * gu_rows];
    gemm_f32(w_gu, gu_rows, h, x_g, n, gu_out)?;

    let act = &mut act[..n * inter];
    for i in 0..n {
        let row = &gu_out[i * gu_rows..(i + 1) * gu_rows];
        let g = &row[..inter];
        let u = &row[inter..];
        let a = &mut act[i * inter..(i + 1) * inter];
        for j in 0..inter {
            let gv = g[j];
            a[j] = (gv / (1.0 + (-gv).exp())) * u[j];
        }
    }

    let w_down = &mut w_down[..h * inter];
    widen_bf16_into(&expert.down, h, inter, w_down)?;
    let down = &mut down[..n * h];
    gemm_f32(w_down, h, inter, act, n, down)?;

    for (i, &(t, w)) in members.iter().enumerate() {
        let src = &down[i * h..(i + 1) * h];
        let dst = &mut moe_out[t * h..(t + 1) * h];
        for j in 0..h {
            dst[j] += src[j] * w;
        }
    }
    Ok(())
}

/// Parallel per-expert MoE over the flat token corpus at one layer.
fn moe_all_experts_parallel(
    layer: &mut LoadedLayer,
    expert_members: &[Vec<(usize, f32)>],
    all_router_in: &[f32],
    total_tokens: usize,
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
) -> Result<()> {
    moe_out.fill(0.0);

    // Active experts only.
    let active: Vec<usize> = (0..QWEN30_EXPERTS)
        .filter(|&e| !expert_members[e].is_empty())
        .collect();
    for e in 0..QWEN30_EXPERTS {
        if expert_members[e].is_empty() {
            layer.experts[e].gate = Vec::new();
            layer.experts[e].up = Vec::new();
            layer.experts[e].down = Vec::new();
        }
    }
    if active.is_empty() {
        return Ok(());
    }

    // Cap workers by corpus size so partial moe_out buffers stay in budget.
    // Each worker holds a private total_tokens×h partial (~4 B/elem).
    let max_by_mem = if total_tokens > 40_000 {
        2usize
    } else if total_tokens > 4_000 {
        4
    } else {
        8
    };
    let n_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, max_by_mem)
        .min(active.len());

    if n_workers == 1 {
        let max_n = active
            .iter()
            .map(|&e| expert_members[e].len())
            .max()
            .unwrap_or(0);
        let mut x_g = vec![0.0f32; max_n * h];
        let mut gu_out = vec![0.0f32; max_n * 2 * inter];
        let mut act = vec![0.0f32; max_n * inter];
        let mut down = vec![0.0f32; max_n * h];
        let mut w_gu = vec![0.0f32; 2 * inter * h];
        let mut w_down = vec![0.0f32; h * inter];
        for &e in &active {
            expert_batched_moe(
                &layer.experts[e],
                &expert_members[e],
                all_router_in,
                h,
                inter,
                moe_out,
                &mut x_g,
                &mut gu_out,
                &mut act,
                &mut down,
                &mut w_gu,
                &mut w_down,
            )?;
            layer.experts[e].gate = Vec::new();
            layer.experts[e].up = Vec::new();
            layer.experts[e].down = Vec::new();
        }
        return Ok(());
    }

    // Partition active experts across workers; each writes a private partial.
    let chunk = active.len().div_ceil(n_workers);
    let mut partials: Vec<Vec<f32>> = (0..n_workers)
        .map(|_| vec![0.0f32; total_tokens * h])
        .collect();
    let err: Mutex<Option<String>> = Mutex::new(None);

    // Immutable expert view for the parallel section; BF16 free happens after.
    {
        let experts: &[ExpertWeights] = &layer.experts;
        std::thread::scope(|scope| {
            for (wi, partial) in partials.iter_mut().enumerate() {
                let start = wi * chunk;
                if start >= active.len() {
                    break;
                }
                let end = (start + chunk).min(active.len());
                let my_experts = &active[start..end];
                let expert_members = expert_members;
                let all_router_in = all_router_in;
                let err = &err;
                scope.spawn(move || {
                    let max_n = my_experts
                        .iter()
                        .map(|&e| expert_members[e].len())
                        .max()
                        .unwrap_or(0);
                    let mut x_g = vec![0.0f32; max_n * h];
                    let mut gu_out = vec![0.0f32; max_n * 2 * inter];
                    let mut act = vec![0.0f32; max_n * inter];
                    let mut down = vec![0.0f32; max_n * h];
                    let mut w_gu = vec![0.0f32; 2 * inter * h];
                    let mut w_down = vec![0.0f32; h * inter];
                    for &e in my_experts {
                        if let Err(err_e) = expert_batched_moe(
                            &experts[e],
                            &expert_members[e],
                            all_router_in,
                            h,
                            inter,
                            partial,
                            &mut x_g,
                            &mut gu_out,
                            &mut act,
                            &mut down,
                            &mut w_gu,
                            &mut w_down,
                        ) {
                            if let Ok(mut g) = err.lock() {
                                *g = Some(err_e.to_string());
                            }
                            return;
                        }
                    }
                });
            }
        });
    }

    if let Some(msg) = err.into_inner().unwrap_or(None) {
        return Err(model_err(msg));
    }

    // Reduce partials.
    for partial in &partials {
        for (d, s) in moe_out.iter_mut().zip(partial.iter()) {
            *d += *s;
        }
    }
    // Free all expert BF16 now that every active expert has been consumed.
    for e in 0..QWEN30_EXPERTS {
        layer.experts[e].gate = Vec::new();
        layer.experts[e].up = Vec::new();
        layer.experts[e].down = Vec::new();
    }
    Ok(())
}

#[cfg(target_os = "macos")]
mod accelerate_gemm {
    /// CBLAS row-major order / transpose enums (from cblas.h).
    const CBLAS_ROW_MAJOR: i32 = 101;
    const CBLAS_NO_TRANS: i32 = 111;
    const CBLAS_TRANS: i32 = 112;

    #[link(name = "Accelerate", kind = "framework")]
    extern "C" {
        fn cblas_sgemm(
            order: i32,
            transa: i32,
            transb: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    /// Batched matmul: for each of `n_batch` rows of `x` (`[n_batch, cols]`),
    /// compute `out[b] = W @ x[b]` where `W` is row-major `[rows, cols]`.
    ///
    /// Equivalently `Out = X @ Wᵀ` with X/Out row-major.
    pub fn gemm_w_times_x(
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        n_batch: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(w.len(), rows * cols);
        debug_assert_eq!(x.len(), n_batch * cols);
        debug_assert_eq!(out.len(), n_batch * rows);
        if n_batch == 0 || rows == 0 || cols == 0 {
            return;
        }
        // C = A * Bᵀ with A=X (M×K), B=W (N×K), C=Out (M×N)
        // M=n_batch, N=rows, K=cols
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_TRANS,
                n_batch as i32,
                rows as i32,
                cols as i32,
                1.0,
                x.as_ptr(),
                cols as i32,
                w.as_ptr(),
                cols as i32,
                0.0,
                out.as_mut_ptr(),
                rows as i32,
            );
        }
    }

    /// `Out = scores @ V` with scores `[seq, seq]` and V `[seq, head_dim]`,
    /// both row-major. Used by causal prefill attention.
    pub fn gemm_scores_times_v(
        scores: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(scores.len(), seq_len * seq_len);
        debug_assert_eq!(v.len(), seq_len * head_dim);
        debug_assert_eq!(out.len(), seq_len * head_dim);
        if seq_len == 0 || head_dim == 0 {
            return;
        }
        // C = A * B: A=scores (M×K), B=V (K×N), C=out (M×N)
        // M=seq, N=head_dim, K=seq
        unsafe {
            cblas_sgemm(
                CBLAS_ROW_MAJOR,
                CBLAS_NO_TRANS,
                CBLAS_NO_TRANS,
                seq_len as i32,
                head_dim as i32,
                seq_len as i32,
                1.0,
                scores.as_ptr(),
                seq_len as i32,
                v.as_ptr(),
                head_dim as i32,
                0.0,
                out.as_mut_ptr(),
                head_dim as i32,
            );
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod accelerate_gemm {
    /// Portable fallback: parallel GEMV over batch (no Accelerate).
    pub fn gemm_w_times_x(
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        n_batch: usize,
        out: &mut [f32],
    ) {
        for b in 0..n_batch {
            let xb = &x[b * cols..(b + 1) * cols];
            let ob = &mut out[b * rows..(b + 1) * rows];
            for r in 0..rows {
                let row = &w[r * cols..(r + 1) * cols];
                let mut acc = 0.0f32;
                for c in 0..cols {
                    acc += row[c] * xb[c];
                }
                ob[r] = acc;
            }
        }
    }

    pub fn gemm_scores_times_v(
        scores: &[f32],
        v: &[f32],
        seq_len: usize,
        head_dim: usize,
        out: &mut [f32],
    ) {
        for b in 0..seq_len {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for t in 0..seq_len {
                    acc += scores[b * seq_len + t] * v[t * head_dim + d];
                }
                out[b * head_dim + d] = acc;
            }
        }
    }
}

/// Causal multi-head attention prefill for a full sequence, using Accelerate
/// GEMM for the QKᵀ and PV matmuls. Numerically close to repeated
/// [`mha_decode_step`] calls (float reassociation only).
///
/// Layouts match `mha_decode_step`:
///   q:     (seq, n_heads * head_dim)
///   k/v:   (seq, n_kv_heads * head_dim)
///   out:   (seq, n_heads * head_dim)
fn mha_prefill_causal(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    out: &mut [f32],
) -> Result<()> {
    if seq_len == 0 {
        return Ok(());
    }
    let q_dim = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;
    if q.len() != seq_len * q_dim
        || k.len() != seq_len * kv_dim
        || v.len() != seq_len * kv_dim
        || out.len() != seq_len * q_dim
    {
        return Err(model_err("mha_prefill_causal geometry mismatch"));
    }
    // Short sequences: the decode-step loop is fine and avoids scratch alloc.
    if seq_len <= 32 {
        for pos in 0..seq_len {
            mha_decode_step(
                &q[pos * q_dim..(pos + 1) * q_dim],
                &k[..(pos + 1) * kv_dim],
                &v[..(pos + 1) * kv_dim],
                n_heads,
                n_kv_heads,
                head_dim,
                pos + 1,
                &mut out[pos * q_dim..(pos + 1) * q_dim],
            )?;
        }
        return Ok(());
    }

    let group_size = n_heads / n_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Per-head Q/K/V scratch (contiguous for GEMM).
    let mut q_h = vec![0.0f32; seq_len * head_dim];
    let mut k_h = vec![0.0f32; seq_len * head_dim];
    let mut v_h = vec![0.0f32; seq_len * head_dim];
    let mut scores = vec![0.0f32; seq_len * seq_len];
    let mut ctx = vec![0.0f32; seq_len * head_dim];

    for h in 0..n_heads {
        let kv_h = h / group_size;
        // Gather head slices into contiguous [seq, head_dim].
        for t in 0..seq_len {
            let qs = t * q_dim + h * head_dim;
            let ks = t * kv_dim + kv_h * head_dim;
            q_h[t * head_dim..(t + 1) * head_dim]
                .copy_from_slice(&q[qs..qs + head_dim]);
            k_h[t * head_dim..(t + 1) * head_dim]
                .copy_from_slice(&k[ks..ks + head_dim]);
            v_h[t * head_dim..(t + 1) * head_dim]
                .copy_from_slice(&v[ks..ks + head_dim]);
        }
        // scores = Q @ Kᵀ   →  [seq, seq]
        // Using gemm_w_times_x with W=K (rows=seq, cols=head_dim), X=Q (n_batch=seq):
        //   out[b] = K @ q[b]  gives scores as [seq, seq] with scores[b, t] = q[b]·k[t]
        // Wait: gemm_w_times_x(W, rows, cols, x, n_batch) does out[b] = W @ x[b]
        // so out[b,r] = W[r,:] · x[b,:]. If W=K (seq × dim), out[b,t] = k[t]·q[b] = scores[b,t].
        accelerate_gemm::gemm_w_times_x(&k_h, seq_len, head_dim, &q_h, seq_len, &mut scores);
        // Scale + causal mask + softmax per query row.
        for b in 0..seq_len {
            let row = &mut scores[b * seq_len..(b + 1) * seq_len];
            let mut max_v = f32::NEG_INFINITY;
            for t in 0..=b {
                row[t] *= scale;
                if row[t] > max_v {
                    max_v = row[t];
                }
            }
            for t in (b + 1)..seq_len {
                row[t] = 0.0; // masked; zero after softmax too
            }
            let mut sum = 0.0f32;
            for t in 0..=b {
                let e = (row[t] - max_v).exp();
                row[t] = e;
                sum += e;
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for t in 0..=b {
                row[t] *= inv;
            }
            for t in (b + 1)..seq_len {
                row[t] = 0.0;
            }
        }
        // ctx = scores @ V   →  [seq, head_dim]
        // gemm: out[b] = Vᵀ?  We want ctx[b,d] = sum_t scores[b,t] * v[t,d]
        // = (scores @ V) with V as [seq, dim].
        // gemm_x_times_wt needs Wt = Vᵀ ([dim, seq]), OR:
        // Use C = scores * V with NoTrans×NoTrans if V is [seq, dim] as B with
        // M=seq, N=dim, K=seq, ldb=dim.
        accelerate_gemm::gemm_scores_times_v(&scores, &v_h, seq_len, head_dim, &mut ctx);
        for t in 0..seq_len {
            let os = t * q_dim + h * head_dim;
            out[os..os + head_dim].copy_from_slice(&ctx[t * head_dim..(t + 1) * head_dim]);
        }
    }
    Ok(())
}

/// F32 GEMM: `out[b] = W @ x[b]` for `b in 0..n_batch`.
/// `W` is row-major `[rows, cols]`; `x` is `[n_batch, cols]`; `out` is `[n_batch, rows]`.
pub fn gemm_f32(
    w: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
    n_batch: usize,
    out: &mut [f32],
) -> Result<()> {
    if w.len() < rows * cols {
        return Err(model_err(format!(
            "gemm_f32 W len {} < rows*cols {}",
            w.len(),
            rows * cols
        )));
    }
    if x.len() != n_batch * cols || out.len() != n_batch * rows {
        return Err(model_err(format!(
            "gemm_f32 geometry: x={} out={} n_batch={n_batch} rows={rows} cols={cols}",
            x.len(),
            out.len()
        )));
    }
    accelerate_gemm::gemm_w_times_x(w, rows, cols, x, n_batch, out);
    Ok(())
}

/// Row-major BF16 GEMM: widen once, then `out[b] = W @ x[b]`.
pub fn gemm_bf16(
    weight_le: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    n_batch: usize,
    out: &mut [f32],
) -> Result<()> {
    let w = widen_bf16_mat(weight_le, rows, cols)?;
    gemm_f32(&w, rows, cols, x, n_batch, out)
}

/// Row-major GEMV: `out = W @ x` with W stored as little-endian BF16 rows.
///
/// Converts BF16 on the fly (no full-matrix f32 allocation). Used by the
/// single-token decode path where weights are not reused enough to amortise a
/// widen. Capture uses [`gemm_bf16`] / [`gemm_f32`] instead so each expert is
/// widened once and multiplied against a token batch.
pub fn gemv_bf16(weight_le: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    if x.len() != cols || out.len() != rows {
        return Err(model_err(format!(
            "gemv_bf16 geometry: x={} out={} rows={rows} cols={cols}",
            x.len(),
            out.len()
        )));
    }
    let expect = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(2))
        .ok_or_else(|| model_err("gemv_bf16 size overflow"))?;
    if weight_le.len() < expect {
        return Err(model_err(format!(
            "gemv_bf16 weight bytes {} < {expect}",
            weight_le.len()
        )));
    }
    // Large single matvec (e.g. lm_head): widen once and hit Accelerate GEMM.
    // Small projections: on-the-fly convert keeps the decode working set lean.
    const WIDEN_THRESHOLD_ELEMS: usize = 256 * 1024;
    if rows.saturating_mul(cols) >= WIDEN_THRESHOLD_ELEMS {
        return gemm_bf16(weight_le, rows, cols, x, 1, out);
    }
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 16);
    let chunk = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out.chunks_mut(chunk).enumerate() {
            let row0 = t * chunk;
            let w = weight_le;
            let x = x;
            scope.spawn(move || {
                for (i, o) in out_chunk.iter_mut().enumerate() {
                    let row = row0 + i;
                    if row >= rows {
                        break;
                    }
                    let base = row * cols * 2;
                    let mut acc = 0.0f32;
                    let mut c = 0usize;
                    while c + 4 <= cols {
                        acc += bf16_le_to_f32(&w[base + c * 2..]) * x[c]
                            + bf16_le_to_f32(&w[base + (c + 1) * 2..]) * x[c + 1]
                            + bf16_le_to_f32(&w[base + (c + 2) * 2..]) * x[c + 2]
                            + bf16_le_to_f32(&w[base + (c + 3) * 2..]) * x[c + 3];
                        c += 4;
                    }
                    while c < cols {
                        acc += bf16_le_to_f32(&w[base + c * 2..]) * x[c];
                        c += 1;
                    }
                    *o = acc;
                }
            });
        }
    });
    Ok(())
}

/// Row-major GEMV with f32 weights (after a one-shot BF16 widen of a layer tensor).
pub fn gemv_f32_rows(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
    if x.len() != cols || out.len() != rows || w.len() < rows * cols {
        return Err(model_err(format!(
            "gemv_f32 geometry: w={} x={} out={} rows={rows} cols={cols}",
            w.len(),
            x.len(),
            out.len()
        )));
    }
    gemm_f32(w, rows, cols, x, 1, out)
}

#[derive(Clone, Debug)]
struct TensorLoc {
    shard: PathBuf,
    /// Absolute byte offset of tensor payload inside the shard file.
    data_offset: u64,
    nbytes: usize,
    shape: Vec<usize>,
    dtype: String,
}

/// Index over the source BF16 safetensors shards. Opens shard headers once;
/// tensor payloads are range-read on demand and never bulk-resident.
pub struct SourceBf16Index {
    pub model_dir: PathBuf,
    map: HashMap<String, TensorLoc>,
    /// Opened shard handles for positioned reads (not full-file loads).
    handles: Mutex<HashMap<PathBuf, File>>,
}

impl SourceBf16Index {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let index_bytes = std::fs::read(&index_path).map_err(|e| {
            model_err(format!(
                "cannot read safetensors index {}: {e}",
                index_path.display()
            ))
        })?;
        let index: Value = serde_json::from_slice(&index_bytes)
            .map_err(|e| model_err(format!("safetensors index is not JSON: {e}")))?;
        let weight_map = index
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| model_err("safetensors index lacks weight_map"))?;

        let mut by_shard: HashMap<String, Vec<String>> = HashMap::new();
        for (name, shard_v) in weight_map {
            let shard = shard_v
                .as_str()
                .ok_or_else(|| model_err(format!("weight_map entry {name} is not a string")))?;
            by_shard.entry(shard.to_string()).or_default().push(name.clone());
        }

        let mut map = HashMap::new();
        for (shard_name, names) in by_shard {
            let shard_path = model_dir.join(&shard_name);
            let header = read_safetensors_header(&shard_path)?;
            let header_len = header.header_nbytes;
            for name in names {
                let info = header
                    .tensors
                    .get(&name)
                    .ok_or_else(|| model_err(format!("shard {shard_name} lacks tensor {name}")))?;
                if info.dtype != "BF16" && info.dtype != "BFLOAT16" {
                    return Err(model_err(format!(
                        "tensor {name} dtype {} is not BF16",
                        info.dtype
                    )));
                }
                let (begin, end) = info.data_offsets;
                if end < begin {
                    return Err(model_err(format!("tensor {name} has inverted data_offsets")));
                }
                let nbytes = (end - begin) as usize;
                map.insert(
                    name,
                    TensorLoc {
                        shard: shard_path.clone(),
                        data_offset: 8 + header_len + begin,
                        nbytes,
                        shape: info.shape.clone(),
                        dtype: info.dtype.clone(),
                    },
                );
            }
        }
        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            map,
            handles: Mutex::new(HashMap::new()),
        })
    }

    pub fn tensor_count(&self) -> usize {
        self.map.len()
    }

    pub fn require(&self, name: &str) -> Result<&TensorLoc> {
        self.map
            .get(name)
            .ok_or_else(|| model_err(format!("source index lacks tensor {name}")))
    }

    /// Range-read a tensor's raw BF16 payload. Does not keep other tensors resident.
    ///
    /// On Unix this uses `pread` (`read_exact_at`) so concurrent callers on the
    /// same shard do not need to serialize seeks. That is what makes parallel
    /// expert loads in [`LoadedLayer::load`] safe and fast.
    pub fn read_raw(&self, name: &str) -> Result<Vec<u8>> {
        let loc = self.require(name)?;
        // Avoid zero-fill: pread overwrites every byte. Zeroing ~1.2 GiB/layer × 48
        // was a dominant sys-time cost on the capture path.
        let mut buf = Vec::with_capacity(loc.nbytes);
        unsafe {
            buf.set_len(loc.nbytes);
        }
        self.read_raw_into(loc, name, &mut buf)?;
        Ok(buf)
    }

    fn ensure_shard_open(&self, shard: &Path) -> Result<()> {
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| model_err("source shard handle map poisoned"))?;
        if !handles.contains_key(shard) {
            let f = File::open(shard)
                .map_err(|e| model_err(format!("cannot open shard {}: {e}", shard.display())))?;
            handles.insert(shard.to_path_buf(), f);
        }
        Ok(())
    }

    fn read_raw_into(&self, loc: &TensorLoc, name: &str, buf: &mut [u8]) -> Result<()> {
        if buf.len() < loc.nbytes {
            return Err(model_err(format!(
                "read_raw_into buffer {} < {}",
                buf.len(),
                loc.nbytes
            )));
        }
        self.ensure_shard_open(&loc.shard)?;
        // Clone the fd under the lock, then release so concurrent preads do not
        // serialize on the handle map.
        let file = {
            let handles = self
                .handles
                .lock()
                .map_err(|_| model_err("source shard handle map poisoned"))?;
            let f = handles
                .get(&loc.shard)
                .ok_or_else(|| model_err(format!("shard {} not open", loc.shard.display())))?;
            f.try_clone().map_err(|e| {
                model_err(format!(
                    "cannot clone shard handle {}: {e}",
                    loc.shard.display()
                ))
            })?
        };
        #[cfg(unix)]
        {
            // pread: position is an argument, concurrent readers are fine.
            file.read_exact_at(&mut buf[..loc.nbytes], loc.data_offset)
                .map_err(|e| {
                    model_err(format!(
                        "range-read {} ({} bytes) from {} @ {}: {e}",
                        name,
                        loc.nbytes,
                        loc.shard.display(),
                        loc.data_offset
                    ))
                })?;
        }
        #[cfg(not(unix))]
        {
            let mut f = file;
            f.seek(SeekFrom::Start(loc.data_offset)).map_err(|e| {
                model_err(format!(
                    "seek {} @ {}: {e}",
                    loc.shard.display(),
                    loc.data_offset
                ))
            })?;
            f.read_exact(&mut buf[..loc.nbytes]).map_err(|e| {
                model_err(format!(
                    "range-read {} ({} bytes) from {}: {e}",
                    name,
                    loc.nbytes,
                    loc.shard.display()
                ))
            })?;
        }
        Ok(())
    }

    /// Parallel range-read of many tensors (used to load a layer's 128 experts).
    pub fn read_raw_many(&self, names: &[String]) -> Result<Vec<Vec<u8>>> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        // Ensure every needed shard is open before the parallel section.
        for name in names {
            let loc = self.require(name)?;
            self.ensure_shard_open(&loc.shard)?;
        }
        let n = names.len();
        let n_workers = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(4)
            .clamp(1, 16)
            .min(n);
        let chunk = n.div_ceil(n_workers);
        let mut out: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        let err: Mutex<Option<String>> = Mutex::new(None);
        std::thread::scope(|scope| {
            for (wi, out_chunk) in out.chunks_mut(chunk).enumerate() {
                let base = wi * chunk;
                let names = names;
                let err = &err;
                let index = self;
                scope.spawn(move || {
                    for (i, slot) in out_chunk.iter_mut().enumerate() {
                        let name = &names[base + i];
                        match index.read_raw(name) {
                            Ok(buf) => *slot = Some(buf),
                            Err(e) => {
                                if let Ok(mut g) = err.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                        }
                    }
                });
            }
        });
        if let Some(msg) = err.into_inner().unwrap_or(None) {
            return Err(model_err(msg));
        }
        out.into_iter()
            .map(|o| o.ok_or_else(|| model_err("parallel read_raw_many missing result")))
            .collect()
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>> {
        let raw = self.read_raw(name)?;
        widen_native("native.bf16", &raw)
    }

    /// Read a single embedding row without loading the full embedding table.
    pub fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= QWEN30_VOCAB {
            return Err(model_err(format!("token {token} outside vocabulary")));
        }
        let loc = self.require("model.embed_tokens.weight")?;
        if loc.shape != [QWEN30_VOCAB, QWEN30_HIDDEN] {
            return Err(model_err(format!(
                "embed_tokens shape {:?} is not [{QWEN30_VOCAB}, {QWEN30_HIDDEN}]",
                loc.shape
            )));
        }
        let row_bytes = QWEN30_HIDDEN * 2;
        let offset = loc
            .data_offset
            .checked_add((token as u64).checked_mul(row_bytes as u64).ok_or_else(|| {
                model_err("embed row offset overflow")
            })?)
            .ok_or_else(|| model_err("embed absolute offset overflow"))?;
        let mut handles = self
            .handles
            .lock()
            .map_err(|_| model_err("source shard handle map poisoned"))?;
        let file = if let Some(f) = handles.get_mut(&loc.shard) {
            f
        } else {
            let f = File::open(&loc.shard).map_err(|e| {
                model_err(format!("cannot open shard {}: {e}", loc.shard.display()))
            })?;
            handles.insert(loc.shard.clone(), f);
            handles.get_mut(&loc.shard).unwrap()
        };
        file.seek(SeekFrom::Start(offset)).map_err(|e| {
            model_err(format!("seek embed row {token}: {e}"))
        })?;
        let mut buf = vec![0u8; row_bytes];
        file.read_exact(&mut buf)
            .map_err(|e| model_err(format!("read embed row {token}: {e}")))?;
        widen_native("native.bf16", &buf)
    }
}

struct SafetensorsHeader {
    header_nbytes: u64,
    tensors: HashMap<String, SafetensorsTensorInfo>,
}

struct SafetensorsTensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: (u64, u64),
}

fn read_safetensors_header(path: &Path) -> Result<SafetensorsHeader> {
    let mut file = File::open(path)
        .map_err(|e| model_err(format!("cannot open {}: {e}", path.display())))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| model_err(format!("cannot read header length of {}: {e}", path.display())))?;
    let header_nbytes = u64::from_le_bytes(len_buf);
    if header_nbytes == 0 || header_nbytes > 64 * 1024 * 1024 {
        return Err(model_err(format!(
            "implausible safetensors header length {header_nbytes} in {}",
            path.display()
        )));
    }
    let mut raw = vec![0u8; header_nbytes as usize];
    file.read_exact(&mut raw)
        .map_err(|e| model_err(format!("cannot read header of {}: {e}", path.display())))?;
    let value: Value = serde_json::from_slice(&raw)
        .map_err(|e| model_err(format!("safetensors header JSON invalid in {}: {e}", path.display())))?;
    let object = value
        .as_object()
        .ok_or_else(|| model_err(format!("safetensors header is not an object in {}", path.display())))?;
    let mut tensors = HashMap::new();
    for (name, info_v) in object {
        if name == "__metadata__" {
            continue;
        }
        let info = info_v
            .as_object()
            .ok_or_else(|| model_err(format!("tensor {name} header is not an object")))?;
        let dtype = info
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| model_err(format!("tensor {name} lacks dtype")))?
            .to_string();
        let shape = info
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks shape")))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| model_err(format!("tensor {name} has non-integer shape")))
            })
            .collect::<Result<Vec<_>>>()?;
        let offsets = info
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| model_err(format!("tensor {name} lacks data_offsets")))?;
        if offsets.len() != 2 {
            return Err(model_err(format!("tensor {name} data_offsets is not a pair")));
        }
        let begin = offsets[0]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[0] invalid")))?;
        let end = offsets[1]
            .as_u64()
            .ok_or_else(|| model_err(format!("tensor {name} data_offsets[1] invalid")))?;
        tensors.insert(
            name.clone(),
            SafetensorsTensorInfo {
                dtype,
                shape,
                data_offsets: (begin, end),
            },
        );
    }
    Ok(SafetensorsHeader {
        header_nbytes,
        tensors,
    })
}

fn layer_name(layer: usize, suffix: &str) -> String {
    format!("model.layers.{layer}.{suffix}")
}

fn expert_name(layer: usize, expert: usize, role: &str) -> String {
    format!("model.layers.{layer}.mlp.experts.{expert}.{role}.weight")
}

/// One loaded transformer layer, held only for the duration of that layer's
/// pass over the corpus, then dropped.
pub struct LoadedLayer {
    pub layer: usize,
    pub input_layernorm: Vec<f32>,
    pub post_attention_layernorm: Vec<f32>,
    pub q_proj: Vec<u8>,
    pub k_proj: Vec<u8>,
    pub v_proj: Vec<u8>,
    pub o_proj: Vec<u8>,
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub router: Vec<u8>,
    /// gate/up/down raw BF16 payloads per expert.
    pub experts: Vec<ExpertWeights>,
    pub resident_bytes: u64,
}

pub struct ExpertWeights {
    pub gate: Vec<u8>,
    pub up: Vec<u8>,
    pub down: Vec<u8>,
}

impl LoadedLayer {
    pub fn load(index: &SourceBf16Index, layer: usize) -> Result<Self> {
        if layer >= QWEN30_LAYERS {
            return Err(model_err(format!("layer {layer} out of range")));
        }
        let input_layernorm = index.read_f32(&layer_name(layer, "input_layernorm.weight"))?;
        let post_attention_layernorm =
            index.read_f32(&layer_name(layer, "post_attention_layernorm.weight"))?;
        let q_norm = index.read_f32(&layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = index.read_f32(&layer_name(layer, "self_attn.k_norm.weight"))?;
        let q_proj = index.read_raw(&layer_name(layer, "self_attn.q_proj.weight"))?;
        let k_proj = index.read_raw(&layer_name(layer, "self_attn.k_proj.weight"))?;
        let v_proj = index.read_raw(&layer_name(layer, "self_attn.v_proj.weight"))?;
        let o_proj = index.read_raw(&layer_name(layer, "self_attn.o_proj.weight"))?;
        let router = index.read_raw(&layer_name(layer, "mlp.gate.weight"))?;
        // 128 experts × gate/up/down: parallel pread across shards.
        let mut expert_names = Vec::with_capacity(QWEN30_EXPERTS * 3);
        for expert in 0..QWEN30_EXPERTS {
            expert_names.push(expert_name(layer, expert, "gate_proj"));
            expert_names.push(expert_name(layer, expert, "up_proj"));
            expert_names.push(expert_name(layer, expert, "down_proj"));
        }
        let mut expert_payloads = index.read_raw_many(&expert_names)?;
        if expert_payloads.len() != QWEN30_EXPERTS * 3 {
            return Err(model_err(format!(
                "expert payload count {} != {}",
                expert_payloads.len(),
                QWEN30_EXPERTS * 3
            )));
        }
        let mut experts = Vec::with_capacity(QWEN30_EXPERTS);
        let mut resident = (input_layernorm.len()
            + post_attention_layernorm.len()
            + q_norm.len()
            + k_norm.len())
            * 4
            + q_proj.len()
            + k_proj.len()
            + v_proj.len()
            + o_proj.len()
            + router.len();
        // Drain triples (gate, up, down) without cloning payloads.
        let mut payloads = expert_payloads.drain(..);
        for _ in 0..QWEN30_EXPERTS {
            let gate = payloads.next().unwrap();
            let up = payloads.next().unwrap();
            let down = payloads.next().unwrap();
            resident += gate.len() + up.len() + down.len();
            experts.push(ExpertWeights { gate, up, down });
        }
        Ok(Self {
            layer,
            input_layernorm,
            post_attention_layernorm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            router,
            experts,
            resident_bytes: resident as u64,
        })
    }
}

/// Apply Qwen3 NeoX / rotate_half RoPE to one head vector in place.
pub fn rope_neox_inplace(x: &mut [f32], pos: u32, base: f32) {
    let head_dim = x.len();
    let half = head_dim / 2;
    for i in 0..half {
        let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
        let (sin, cos) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

fn rmsnorm_rows(x: &mut [f32], weight: &[f32], n_heads: usize, head_dim: usize) -> Result<()> {
    if x.len() != n_heads * head_dim || weight.len() != head_dim {
        return Err(model_err("rmsnorm_rows geometry mismatch"));
    }
    let mut tmp = vec![0.0f32; head_dim];
    for h in 0..n_heads {
        let start = h * head_dim;
        let row = &x[start..start + head_dim];
        rmsnorm(row, weight, QWEN30_RMS_EPS, &mut tmp);
        x[start..start + head_dim].copy_from_slice(&tmp);
    }
    Ok(())
}

/// Top-k over softmax with `norm_topk_prob=true` (renormalize selected weights).
pub fn router_topk_norm(
    logits: &[f32],
    top_k: usize,
) -> Result<(Vec<u32>, Vec<f32>)> {
    if logits.len() != QWEN30_EXPERTS {
        return Err(model_err(format!(
            "router logits len {} != {QWEN30_EXPERTS}",
            logits.len()
        )));
    }
    let mut probs = logits.to_vec();
    softmax_inplace(&mut probs);
    let mut ids = Vec::with_capacity(top_k);
    let mut weights = Vec::with_capacity(top_k);
    let mut work = probs;
    for _ in 0..top_k {
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in work.iter().enumerate() {
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        ids.push(best_i as u32);
        weights.push(best_v);
        work[best_i] = f32::NEG_INFINITY;
    }
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(model_err("router top-k weight sum is non-positive"));
    }
    for w in &mut weights {
        *w /= sum;
    }
    Ok((ids, weights))
}

/// Per-token capture surface for one layer (matches complete-binary capture).
#[derive(Clone, Debug)]
pub struct LayerTokenCapture {
    pub layer: usize,
    pub selected_expert_ids: Vec<u32>,
    pub normalized_route_weights: Vec<f32>,
    pub router_input_hidden: Vec<f32>,
}

/// Residuals for every token of one probe: length `seq * hidden`.
pub type ProbeHidden = Vec<f32>;

/// Run one loaded layer over one probe sequence (causal attention within the probe).
///
/// Dense projections (Q/K/V/O, router) are one GEMM over the probe's tokens.
/// MoE experts gather tokens that route to them and run one GEMM each. For the
/// multi-probe capture path prefer [`capture_all_layers`], which batches MoE
/// across the whole corpus at the layer so each expert's weights are read once.
///
/// Returns updated residuals and per-token route/hidden captures for this layer.
pub fn forward_layer_probe(
    layer: &LoadedLayer,
    hidden: &mut ProbeHidden,
    seq_len: usize,
) -> Result<Vec<LayerTokenCapture>> {
    if seq_len == 0 {
        return Ok(Vec::new());
    }
    if hidden.len() != seq_len * QWEN30_HIDDEN {
        return Err(model_err(format!(
            "probe hidden len {} != seq {seq_len} * {QWEN30_HIDDEN}",
            hidden.len()
        )));
    }

    let q_dim = QWEN30_HEADS * QWEN30_HEAD_DIM;
    let kv_dim = QWEN30_KV_HEADS * QWEN30_HEAD_DIM;
    let h = QWEN30_HIDDEN;
    let inter = QWEN30_MOE_INTERMEDIATE;

    // --- Attention: batch Q/K/V over all positions, causal MHA, batch O. ---
    let mut x_norm = vec![0.0f32; seq_len * h];
    for pos in 0..seq_len {
        let x = &hidden[pos * h..(pos + 1) * h];
        rmsnorm(
            x,
            &layer.input_layernorm,
            QWEN30_RMS_EPS,
            &mut x_norm[pos * h..(pos + 1) * h],
        );
    }

    let q_w = widen_bf16_mat(&layer.q_proj, q_dim, h)?;
    let k_w = widen_bf16_mat(&layer.k_proj, kv_dim, h)?;
    let v_w = widen_bf16_mat(&layer.v_proj, kv_dim, h)?;
    let mut q = vec![0.0f32; seq_len * q_dim];
    let mut k_cache = vec![0.0f32; seq_len * kv_dim];
    let mut v_cache = vec![0.0f32; seq_len * kv_dim];
    gemm_f32(&q_w, q_dim, h, &x_norm, seq_len, &mut q)?;
    gemm_f32(&k_w, kv_dim, h, &x_norm, seq_len, &mut k_cache)?;
    gemm_f32(&v_w, kv_dim, h, &x_norm, seq_len, &mut v_cache)?;
    drop(q_w);
    drop(k_w);
    drop(v_w);

    for pos in 0..seq_len {
        let q_row = &mut q[pos * q_dim..(pos + 1) * q_dim];
        let k_row = &mut k_cache[pos * kv_dim..(pos + 1) * kv_dim];
        rmsnorm_rows(q_row, &layer.q_norm, QWEN30_HEADS, QWEN30_HEAD_DIM)?;
        rmsnorm_rows(k_row, &layer.k_norm, QWEN30_KV_HEADS, QWEN30_HEAD_DIM)?;
        for head in 0..QWEN30_HEADS {
            let start = head * QWEN30_HEAD_DIM;
            rope_neox_inplace(
                &mut q_row[start..start + QWEN30_HEAD_DIM],
                pos as u32,
                QWEN30_ROPE_THETA,
            );
        }
        for head in 0..QWEN30_KV_HEADS {
            let start = head * QWEN30_HEAD_DIM;
            rope_neox_inplace(
                &mut k_row[start..start + QWEN30_HEAD_DIM],
                pos as u32,
                QWEN30_ROPE_THETA,
            );
        }
    }

    let mut attn = vec![0.0f32; seq_len * q_dim];
    mha_prefill_causal(
        &q,
        &k_cache,
        &v_cache,
        QWEN30_HEADS,
        QWEN30_KV_HEADS,
        QWEN30_HEAD_DIM,
        seq_len,
        &mut attn,
    )?;
    drop(q);
    drop(k_cache);
    drop(v_cache);

    let o_w = widen_bf16_mat(&layer.o_proj, h, q_dim)?;
    let mut attn_proj = vec![0.0f32; seq_len * h];
    gemm_f32(&o_w, h, q_dim, &attn, seq_len, &mut attn_proj)?;
    drop(o_w);
    drop(attn);
    for pos in 0..seq_len {
        add_inplace(
            &mut hidden[pos * h..(pos + 1) * h],
            &attn_proj[pos * h..(pos + 1) * h],
        );
    }
    drop(attn_proj);

    // Router input = post-attention RMSNorm(x) for every position.
    for pos in 0..seq_len {
        let x = &hidden[pos * h..(pos + 1) * h];
        rmsnorm(
            x,
            &layer.post_attention_layernorm,
            QWEN30_RMS_EPS,
            &mut x_norm[pos * h..(pos + 1) * h],
        );
    }

    let router_w = widen_bf16_mat(&layer.router, QWEN30_EXPERTS, h)?;
    let mut router_logits = vec![0.0f32; seq_len * QWEN30_EXPERTS];
    gemm_f32(
        &router_w,
        QWEN30_EXPERTS,
        h,
        &x_norm,
        seq_len,
        &mut router_logits,
    )?;
    drop(router_w);

    let mut routes: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(seq_len);
    let mut expert_members: Vec<Vec<(usize, f32)>> = vec![Vec::new(); QWEN30_EXPERTS];
    for pos in 0..seq_len {
        let (ids, weights) = router_topk_norm(
            &router_logits[pos * QWEN30_EXPERTS..(pos + 1) * QWEN30_EXPERTS],
            QWEN30_TOP_K,
        )?;
        for (&e, &w) in ids.iter().zip(weights.iter()) {
            expert_members[e as usize].push((pos, w));
        }
        routes.push((ids, weights));
    }
    drop(router_logits);

    // MoE: one GEMM per expert over the tokens that selected it.
    let mut moe_out = vec![0.0f32; seq_len * h];
    for e in 0..QWEN30_EXPERTS {
        let members = &expert_members[e];
        if members.is_empty() {
            continue;
        }
        let n = members.len();
        let mut x_g = vec![0.0f32; n * h];
        for (i, &(pos, _)) in members.iter().enumerate() {
            x_g[i * h..(i + 1) * h].copy_from_slice(&x_norm[pos * h..(pos + 1) * h]);
        }
        let expert_w = &layer.experts[e];
        let gate_w = widen_bf16_mat(&expert_w.gate, inter, h)?;
        let up_w = widen_bf16_mat(&expert_w.up, inter, h)?;
        let mut gate = vec![0.0f32; n * inter];
        let mut up = vec![0.0f32; n * inter];
        gemm_f32(&gate_w, inter, h, &x_g, n, &mut gate)?;
        gemm_f32(&up_w, inter, h, &x_g, n, &mut up)?;
        drop(gate_w);
        drop(up_w);
        drop(x_g);

        let mut act = vec![0.0f32; n * inter];
        for i in 0..n {
            silu_mul(
                &gate[i * inter..(i + 1) * inter],
                &up[i * inter..(i + 1) * inter],
                &mut act[i * inter..(i + 1) * inter],
            );
        }
        drop(gate);
        drop(up);

        let down_w = widen_bf16_mat(&expert_w.down, h, inter)?;
        let mut down = vec![0.0f32; n * h];
        gemm_f32(&down_w, h, inter, &act, n, &mut down)?;
        drop(down_w);
        drop(act);

        for (i, &(pos, w)) in members.iter().enumerate() {
            let src = &down[i * h..(i + 1) * h];
            let dst = &mut moe_out[pos * h..(pos + 1) * h];
            for j in 0..h {
                dst[j] += src[j] * w;
            }
        }
    }

    let mut captures = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        add_inplace(
            &mut hidden[pos * h..(pos + 1) * h],
            &moe_out[pos * h..(pos + 1) * h],
        );
        let (ids, weights) = routes[pos].clone();
        captures.push(LayerTokenCapture {
            layer: layer.layer,
            selected_expert_ids: ids,
            normalized_route_weights: weights,
            router_input_hidden: x_norm[pos * h..(pos + 1) * h].to_vec(),
        });
    }
    Ok(captures)
}

/// Embed every probe's tokens into residual streams (range-read rows only).
pub fn embed_probes(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
) -> Result<Vec<ProbeHidden>> {
    let mut out = Vec::with_capacity(probes.len());
    for (_, tokens) in probes {
        let mut h = Vec::with_capacity(tokens.len() * QWEN30_HIDDEN);
        for &tok in tokens {
            let row = index.embed_row(tok)?;
            h.extend_from_slice(&row);
        }
        out.push(h);
    }
    Ok(out)
}

/// Final RMSNorm + lm_head logits for the last residual of a sequence.
pub fn logits_from_final_hidden(
    index: &SourceBf16Index,
    hidden: &[f32],
) -> Result<Vec<f32>> {
    if hidden.len() != QWEN30_HIDDEN {
        return Err(model_err("final hidden width mismatch"));
    }
    let norm_w = index.read_f32("model.norm.weight")?;
    let mut normed = vec![0.0f32; QWEN30_HIDDEN];
    rmsnorm(hidden, &norm_w, QWEN30_RMS_EPS, &mut normed);
    // lm_head is large (~593 MiB BF16). Load, matvec, free.
    let lm_head = index.read_raw("lm_head.weight")?;
    let mut logits = vec![0.0f32; QWEN30_VOCAB];
    gemv_bf16(&lm_head, QWEN30_VOCAB, QWEN30_HIDDEN, &normed, &mut logits)?;
    drop(lm_head);
    Ok(logits)
}

/// Layer-major full forward over all probes: returns per-probe per-layer per-token captures
/// and leaves `hiddens` as the final residuals.
///
/// Per layer this:
/// 1. Streams attention probe-by-probe (causal within probe) with batched Q/K/V/O GEMMs
/// 2. Packs every token's post-attention RMSNorm (router input) into one matrix
/// 3. Runs one router GEMM over the whole corpus at this layer
/// 4. Per expert: gather tokens that selected it, one gate/up/down GEMM, scatter-add
///
/// That is the point of layer-major: each expert's weights are widened and multiplied
/// once per layer, not once per token.
pub fn capture_all_layers(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
    hiddens: &mut [ProbeHidden],
    mut on_layer: Option<&mut dyn FnMut(usize, u64)>,
) -> Result<Vec<Vec<Vec<LayerTokenCapture>>>> {
    if hiddens.len() != probes.len() {
        return Err(model_err("hiddens/probes length mismatch"));
    }
    // captures[probe][token][layer]
    let mut captures: Vec<Vec<Vec<LayerTokenCapture>>> = probes
        .iter()
        .map(|(_, toks)| (0..toks.len()).map(|_| Vec::with_capacity(QWEN30_LAYERS)).collect())
        .collect();

    let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
    let h = QWEN30_HIDDEN;
    let inter = QWEN30_MOE_INTERMEDIATE;
    let q_dim = QWEN30_HEADS * QWEN30_HEAD_DIM;
    let kv_dim = QWEN30_KV_HEADS * QWEN30_HEAD_DIM;

    // Scratch sized for the largest probe (attention) and the full corpus (MoE).
    let max_seq = probes.iter().map(|(_, t)| t.len()).max().unwrap_or(0);
    let mut scratch_x_norm = vec![0.0f32; max_seq * h];
    let mut scratch_q = vec![0.0f32; max_seq * q_dim];
    let mut scratch_k = vec![0.0f32; max_seq * kv_dim];
    let mut scratch_v = vec![0.0f32; max_seq * kv_dim];
    let mut scratch_attn = vec![0.0f32; max_seq * q_dim];
    let mut scratch_attn_proj = vec![0.0f32; max_seq * h];

    // MoE output + router surfaces. Expert f32 weights and gather scratch are
    // allocated inside moe_all_experts_parallel (per-worker, one-expert-at-a-time).
    let mut moe_out = vec![0.0f32; total_tokens * h];
    let mut all_router_in = vec![0.0f32; total_tokens * h];
    let mut router_logits = vec![0.0f32; total_tokens * QWEN30_EXPERTS];

    for layer_idx in 0..QWEN30_LAYERS {
        let mut layer = LoadedLayer::load(index, layer_idx)?;
        let resident = layer.resident_bytes;
        if let Some(cb) = on_layer.as_mut() {
            cb(layer_idx, resident);
        }

        // token_index[t] = (probe_i, pos)
        let mut token_index: Vec<(usize, usize)> = Vec::with_capacity(total_tokens);

        // Widen dense projections once for this layer; free BF16 payloads.
        // Experts stay BF16 until Phase 3 and are widened one-at-a-time there.
        let q_w = widen_bf16_mat(&layer.q_proj, q_dim, h)?;
        let k_w = widen_bf16_mat(&layer.k_proj, kv_dim, h)?;
        let v_w = widen_bf16_mat(&layer.v_proj, kv_dim, h)?;
        let o_w = widen_bf16_mat(&layer.o_proj, h, q_dim)?;
        let router_w = widen_bf16_mat(&layer.router, QWEN30_EXPERTS, h)?;
        layer.q_proj = Vec::new();
        layer.k_proj = Vec::new();
        layer.v_proj = Vec::new();
        layer.o_proj = Vec::new();
        layer.router = Vec::new();

        // --- Phase 1: attention per probe (probe-local causal), collect router inputs. ---
        let mut flat_t = 0usize;
        for (pi, (_, tokens)) in probes.iter().enumerate() {
            let seq_len = tokens.len();
            if seq_len == 0 {
                continue;
            }
            let hidden = &mut hiddens[pi];
            if hidden.len() != seq_len * h {
                return Err(model_err(format!(
                    "layer {layer_idx} probe {pi}: hidden len {} != {seq_len}*{h}",
                    hidden.len()
                )));
            }

            let x_norm = &mut scratch_x_norm[..seq_len * h];
            for pos in 0..seq_len {
                rmsnorm(
                    &hidden[pos * h..(pos + 1) * h],
                    &layer.input_layernorm,
                    QWEN30_RMS_EPS,
                    &mut x_norm[pos * h..(pos + 1) * h],
                );
            }

            let q = &mut scratch_q[..seq_len * q_dim];
            let k_cache = &mut scratch_k[..seq_len * kv_dim];
            let v_cache = &mut scratch_v[..seq_len * kv_dim];
            gemm_f32(&q_w, q_dim, h, x_norm, seq_len, q)?;
            gemm_f32(&k_w, kv_dim, h, x_norm, seq_len, k_cache)?;
            gemm_f32(&v_w, kv_dim, h, x_norm, seq_len, v_cache)?;

            for pos in 0..seq_len {
                let q_row = &mut q[pos * q_dim..(pos + 1) * q_dim];
                let k_row = &mut k_cache[pos * kv_dim..(pos + 1) * kv_dim];
                rmsnorm_rows(q_row, &layer.q_norm, QWEN30_HEADS, QWEN30_HEAD_DIM)?;
                rmsnorm_rows(k_row, &layer.k_norm, QWEN30_KV_HEADS, QWEN30_HEAD_DIM)?;
                for head in 0..QWEN30_HEADS {
                    let start = head * QWEN30_HEAD_DIM;
                    rope_neox_inplace(
                        &mut q_row[start..start + QWEN30_HEAD_DIM],
                        pos as u32,
                        QWEN30_ROPE_THETA,
                    );
                }
                for head in 0..QWEN30_KV_HEADS {
                    let start = head * QWEN30_HEAD_DIM;
                    rope_neox_inplace(
                        &mut k_row[start..start + QWEN30_HEAD_DIM],
                        pos as u32,
                        QWEN30_ROPE_THETA,
                    );
                }
            }

            let attn = &mut scratch_attn[..seq_len * q_dim];
            mha_prefill_causal(
                q,
                k_cache,
                v_cache,
                QWEN30_HEADS,
                QWEN30_KV_HEADS,
                QWEN30_HEAD_DIM,
                seq_len,
                attn,
            )?;

            let attn_proj = &mut scratch_attn_proj[..seq_len * h];
            gemm_f32(&o_w, h, q_dim, attn, seq_len, attn_proj)?;
            for pos in 0..seq_len {
                add_inplace(
                    &mut hidden[pos * h..(pos + 1) * h],
                    &attn_proj[pos * h..(pos + 1) * h],
                );
            }

            // Post-attention RMSNorm → router input (capture surface).
            for pos in 0..seq_len {
                rmsnorm(
                    &hidden[pos * h..(pos + 1) * h],
                    &layer.post_attention_layernorm,
                    QWEN30_RMS_EPS,
                    &mut all_router_in[flat_t * h..(flat_t + 1) * h],
                );
                token_index.push((pi, pos));
                flat_t += 1;
            }
        }
        debug_assert_eq!(flat_t, total_tokens);
        drop(q_w);
        drop(k_w);
        drop(v_w);
        drop(o_w);

        // --- Phase 2: one router GEMM over all corpus tokens at this layer. ---
        let t_all = total_tokens;
        gemm_f32(
            &router_w,
            QWEN30_EXPERTS,
            h,
            &all_router_in,
            t_all,
            &mut router_logits,
        )?;
        drop(router_w);

        let mut routes: Vec<(Vec<u32>, Vec<f32>)> = Vec::with_capacity(t_all);
        let mut expert_members: Vec<Vec<(usize, f32)>> = vec![Vec::new(); QWEN30_EXPERTS];
        for t in 0..t_all {
            let (ids, weights) = router_topk_norm(
                &router_logits[t * QWEN30_EXPERTS..(t + 1) * QWEN30_EXPERTS],
                QWEN30_TOP_K,
            )?;
            for (&e, &w) in ids.iter().zip(weights.iter()) {
                expert_members[e as usize].push((t, w));
            }
            routes.push((ids, weights));
        }

        // --- Phase 3: parallel per-expert batched GEMM; widen only active experts. ---
        moe_all_experts_parallel(
            &mut layer,
            &expert_members,
            &all_router_in,
            t_all,
            h,
            inter,
            &mut moe_out,
        )?;

        // Apply MoE residual and record captures.
        for (t, &(pi, pos)) in token_index.iter().enumerate() {
            add_inplace(
                &mut hiddens[pi][pos * h..(pos + 1) * h],
                &moe_out[t * h..(t + 1) * h],
            );
            let (ids, weights) = std::mem::take(&mut routes[t]);
            captures[pi][pos].push(LayerTokenCapture {
                layer: layer_idx,
                selected_expert_ids: ids,
                normalized_route_weights: weights,
                router_input_hidden: all_router_in[t * h..(t + 1) * h].to_vec(),
            });
        }

        // Explicit free before next layer load.
        drop(layer);
    }
    Ok(captures)
}

/// One layer's KV cache (f32), layout `(seq, kv_heads * head_dim)`.
struct LayerKv {
    k: Vec<f32>,
    v: Vec<f32>,
    seq: usize,
}

impl LayerKv {
    fn new() -> Self {
        Self {
            k: Vec::new(),
            v: Vec::new(),
            seq: 0,
        }
    }
}

/// Run one position through a loaded layer, appending to that layer's KV cache.
/// Updates `x` (length `hidden`) in place to the post-MoE residual.
fn forward_layer_token(
    layer: &LoadedLayer,
    x: &mut [f32],
    pos: usize,
    kv: &mut LayerKv,
) -> Result<LayerTokenCapture> {
    if x.len() != QWEN30_HIDDEN {
        return Err(model_err("forward_layer_token residual width mismatch"));
    }
    let q_dim = QWEN30_HEADS * QWEN30_HEAD_DIM;
    let kv_dim = QWEN30_KV_HEADS * QWEN30_HEAD_DIM;
    if pos != kv.seq {
        return Err(model_err(format!(
            "KV position mismatch: pos={pos} kv.seq={}",
            kv.seq
        )));
    }
    let mut x_norm = vec![0.0f32; QWEN30_HIDDEN];
    let mut q = vec![0.0f32; q_dim];
    let mut k = vec![0.0f32; kv_dim];
    let mut v = vec![0.0f32; kv_dim];
    let mut attn = vec![0.0f32; q_dim];
    let mut attn_proj = vec![0.0f32; QWEN30_HIDDEN];
    let mut router_logits = vec![0.0f32; QWEN30_EXPERTS];
    let mut gate = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut up = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut act = vec![0.0f32; QWEN30_MOE_INTERMEDIATE];
    let mut down = vec![0.0f32; QWEN30_HIDDEN];
    let mut moe_combined = vec![0.0f32; QWEN30_HIDDEN];

    rmsnorm(x, &layer.input_layernorm, QWEN30_RMS_EPS, &mut x_norm);
    gemv_bf16(&layer.q_proj, q_dim, QWEN30_HIDDEN, &x_norm, &mut q)?;
    gemv_bf16(&layer.k_proj, kv_dim, QWEN30_HIDDEN, &x_norm, &mut k)?;
    gemv_bf16(&layer.v_proj, kv_dim, QWEN30_HIDDEN, &x_norm, &mut v)?;
    rmsnorm_rows(&mut q, &layer.q_norm, QWEN30_HEADS, QWEN30_HEAD_DIM)?;
    rmsnorm_rows(&mut k, &layer.k_norm, QWEN30_KV_HEADS, QWEN30_HEAD_DIM)?;
    for h in 0..QWEN30_HEADS {
        let start = h * QWEN30_HEAD_DIM;
        rope_neox_inplace(
            &mut q[start..start + QWEN30_HEAD_DIM],
            pos as u32,
            QWEN30_ROPE_THETA,
        );
    }
    for h in 0..QWEN30_KV_HEADS {
        let start = h * QWEN30_HEAD_DIM;
        rope_neox_inplace(
            &mut k[start..start + QWEN30_HEAD_DIM],
            pos as u32,
            QWEN30_ROPE_THETA,
        );
    }
    kv.k.extend_from_slice(&k);
    kv.v.extend_from_slice(&v);
    kv.seq += 1;
    mha_decode_step(
        &q,
        &kv.k,
        &kv.v,
        QWEN30_HEADS,
        QWEN30_KV_HEADS,
        QWEN30_HEAD_DIM,
        kv.seq,
        &mut attn,
    )?;
    gemv_bf16(&layer.o_proj, QWEN30_HIDDEN, q_dim, &attn, &mut attn_proj)?;
    add_inplace(x, &attn_proj);
    rmsnorm(x, &layer.post_attention_layernorm, QWEN30_RMS_EPS, &mut x_norm);
    gemv_bf16(
        &layer.router,
        QWEN30_EXPERTS,
        QWEN30_HIDDEN,
        &x_norm,
        &mut router_logits,
    )?;
    let (ids, weights) = router_topk_norm(&router_logits, QWEN30_TOP_K)?;
    moe_combined.fill(0.0);
    for (&expert, &w) in ids.iter().zip(weights.iter()) {
        let expert_w = layer
            .experts
            .get(expert as usize)
            .ok_or_else(|| model_err(format!("route expert {expert} out of range")))?;
        gemv_bf16(
            &expert_w.gate,
            QWEN30_MOE_INTERMEDIATE,
            QWEN30_HIDDEN,
            &x_norm,
            &mut gate,
        )?;
        gemv_bf16(
            &expert_w.up,
            QWEN30_MOE_INTERMEDIATE,
            QWEN30_HIDDEN,
            &x_norm,
            &mut up,
        )?;
        silu_mul(&gate, &up, &mut act);
        gemv_bf16(
            &expert_w.down,
            QWEN30_HIDDEN,
            QWEN30_MOE_INTERMEDIATE,
            &act,
            &mut down,
        )?;
        for i in 0..QWEN30_HIDDEN {
            moe_combined[i] += down[i] * w;
        }
    }
    add_inplace(x, &moe_combined);
    Ok(LayerTokenCapture {
        layer: layer.layer,
        selected_expert_ids: ids,
        normalized_route_weights: weights,
        router_input_hidden: x_norm,
    })
}

/// Greedy decode with the source one-user chat template.
///
/// Layer-major: prefill streams each layer once over the prompt (keeping a
/// small per-layer KV), then each new token re-streams the 48 layers once
/// with only the new position. Never co-resident-loads the full source.
pub fn greedy_decode_user_prompt(
    index: &SourceBf16Index,
    tokenizer_path: &Path,
    user_text: &str,
    max_new_tokens: usize,
) -> Result<GreedyDecodeResult> {
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
        model_err(format!(
            "cannot load tokenizer {}: {e}",
            tokenizer_path.display()
        ))
    })?;
    let rendered = format!("<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n");
    let encoding = tokenizer
        .encode(rendered.as_str(), false)
        .map_err(|e| model_err(format!("tokenizer encode failed: {e}")))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    if prompt_ids.is_empty() {
        return Err(model_err("chat-template encoding produced no tokens"));
    }

    // Residuals for the current sequence (grows with generation).
    let mut residuals: Vec<Vec<f32>> = Vec::with_capacity(prompt_ids.len() + max_new_tokens);
    for &tok in &prompt_ids {
        residuals.push(index.embed_row(tok)?);
    }
    // Per-layer KV across the whole decode session.
    let mut kvs: Vec<LayerKv> = (0..QWEN30_LAYERS).map(|_| LayerKv::new()).collect();

    // Prefill: for each layer, push every prompt position through it.
    for layer_idx in 0..QWEN30_LAYERS {
        let layer = LoadedLayer::load(index, layer_idx)?;
        for pos in 0..prompt_ids.len() {
            forward_layer_token(&layer, &mut residuals[pos], pos, &mut kvs[layer_idx])?;
        }
        drop(layer);
    }

    let mut generated = Vec::new();
    let mut first_token_top10 = Vec::new();
    let eos = [151645u32, 151643u32];

    for _step in 0..max_new_tokens {
        let last = residuals.last().ok_or_else(|| model_err("empty residual"))?;
        let logits = logits_from_final_hidden(index, last)?;
        let next = argmax_f32(&logits);
        if generated.is_empty() {
            let mut ranked: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            ranked.truncate(10);
            first_token_top10 = ranked;
        }
        generated.push(next);
        if eos.contains(&next) {
            break;
        }
        // Append embed of new token and stream it through every layer.
        let pos = residuals.len();
        residuals.push(index.embed_row(next)?);
        for layer_idx in 0..QWEN30_LAYERS {
            let layer = LoadedLayer::load(index, layer_idx)?;
            forward_layer_token(&layer, &mut residuals[pos], pos, &mut kvs[layer_idx])?;
            drop(layer);
        }
    }

    let cont_text = tokenizer
        .decode(&generated, true)
        .map_err(|e| model_err(format!("tokenizer decode failed: {e}")))?;
    Ok(GreedyDecodeResult {
        prompt_token_count: prompt_ids.len(),
        prompt_token_ids: prompt_ids,
        generated_token_ids: generated,
        continuation_text: cont_text,
        rendered_prompt: rendered,
        first_token_top10,
    })
}

#[derive(Clone, Debug)]
pub struct GreedyDecodeResult {
    pub prompt_token_count: usize,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub continuation_text: String,
    pub rendered_prompt: String,
    pub first_token_top10: Vec<(u32, f32)>,
}

/// True when the continuation is a coherent capital-of-France answer.
///
/// Accepts top-1 `" Paris"` / `"Paris"` and multi-token variants such as
/// `"The capital of France is Paris"`. Rejects `"Wien swiper"` degeneration.
pub fn is_coherent_paris_continuation(text: &str) -> bool {
    let t = text.trim_start();
    if t.is_empty() {
        return false;
    }
    let lower = t.to_ascii_lowercase();
    if lower.contains("wien") || lower.contains("swiper") {
        return false;
    }
    // Degenerate pure repetition (same word thrice) is not coherent.
    let words: Vec<&str> = lower.split_whitespace().collect();
    if words.len() >= 3 && words[0] == words[1] && words[1] == words[2] {
        return false;
    }
    if lower.starts_with("paris")
        || lower.starts_with("**paris")
        || lower.starts_with("*paris")
    {
        return true;
    }
    // Multi-token correct answers.
    if lower.contains("paris")
        && (lower.contains("france")
            || lower.starts_with("the capital")
            || lower.starts_with("it's paris")
            || lower.starts_with("it is paris"))
    {
        return true;
    }
    false
}

/// Process peak RSS in bytes (macOS: `ru_maxrss` is already bytes).
pub fn peak_rss_bytes() -> u64 {
    #[cfg(unix)]
    {
        // Layout matches cost_ledger::sample_page_faults (Darwin timeval padding).
        #[cfg(target_os = "macos")]
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TimeVal {
            tv_sec: i64,
            tv_usec: i32,
            _pad: i32,
        }
        #[cfg(not(target_os = "macos"))]
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct TimeVal {
            tv_sec: i64,
            tv_usec: i64,
        }
        #[repr(C)]
        struct Rusage {
            ru_utime: TimeVal,
            ru_stime: TimeVal,
            ru_maxrss: i64,
            ru_ixrss: i64,
            ru_idrss: i64,
            ru_isrss: i64,
            ru_minflt: i64,
            ru_majflt: i64,
            _pad: [i64; 8],
        }
        extern "C" {
            fn getrusage(who: i32, usage: *mut Rusage) -> i32;
        }
        const RUSAGE_SELF: i32 = 0;
        #[cfg(target_os = "macos")]
        let zero_tv = TimeVal {
            tv_sec: 0,
            tv_usec: 0,
            _pad: 0,
        };
        #[cfg(not(target_os = "macos"))]
        let zero_tv = TimeVal {
            tv_sec: 0,
            tv_usec: 0,
        };
        let mut u = Rusage {
            ru_utime: zero_tv,
            ru_stime: zero_tv,
            ru_maxrss: 0,
            ru_ixrss: 0,
            ru_idrss: 0,
            ru_isrss: 0,
            ru_minflt: 0,
            ru_majflt: 0,
            _pad: [0; 8],
        };
        let rc = unsafe { getrusage(RUSAGE_SELF, &mut u) };
        if rc != 0 {
            return 0;
        }
        u.ru_maxrss.max(0) as u64
    }
    #[cfg(not(unix))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn router_topk_renormalizes() {
        let mut logits = vec![-10.0f32; QWEN30_EXPERTS];
        logits[3] = 5.0;
        logits[7] = 4.0;
        logits[11] = 3.0;
        logits[13] = 2.0;
        logits[17] = 1.0;
        logits[19] = 0.5;
        logits[23] = 0.25;
        logits[29] = 0.1;
        let (ids, w) = router_topk_norm(&logits, 8).unwrap();
        assert_eq!(ids.len(), 8);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(ids[0], 3);
    }

    #[test]
    fn paris_coherence_accepts_variants() {
        assert!(is_coherent_paris_continuation(" Paris"));
        assert!(is_coherent_paris_continuation("Paris is the capital"));
        assert!(!is_coherent_paris_continuation(" Wien swiper swiper"));
        assert!(!is_coherent_paris_continuation("swiper Wien"));
    }

    #[test]
    fn gemm_f32_matches_rowwise_gemv() {
        // W is [rows=3, cols=4]; batch of 5 input rows.
        let w: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let rows = 3usize;
        let cols = 4usize;
        let n_batch = 5usize;
        let x: Vec<f32> = (0..n_batch * cols)
            .map(|i| (i as f32) * 0.07 - 0.3)
            .collect();
        let mut out = vec![0.0f32; n_batch * rows];
        gemm_f32(&w, rows, cols, &x, n_batch, &mut out).unwrap();
        for b in 0..n_batch {
            let mut expect = vec![0.0f32; rows];
            gemv_f32_rows(&w, rows, cols, &x[b * cols..(b + 1) * cols], &mut expect).unwrap();
            for r in 0..rows {
                let got = out[b * rows + r];
                let e = expect[r];
                let abs = (got - e).abs();
                let rel = abs / e.abs().max(1e-6);
                assert!(
                    abs < 1e-4 || rel < 1e-4,
                    "batch {b} row {r}: got {got} expect {e} abs={abs} rel={rel}"
                );
            }
        }
    }

    #[test]
    fn gemm_bf16_matches_gemv_bf16_small() {
        let rows = 8usize;
        let cols = 16usize;
        let n_batch = 4usize;
        // Pack simple f32 values as BF16-LE (truncate mantissa).
        let mut weight_le = Vec::with_capacity(rows * cols * 2);
        for i in 0..rows * cols {
            let v = ((i as f32) * 0.01) - 0.4;
            let bits = (v.to_bits() >> 16) as u16;
            weight_le.extend_from_slice(&bits.to_le_bytes());
        }
        let x: Vec<f32> = (0..n_batch * cols)
            .map(|i| (i as f32) * 0.03 - 0.2)
            .collect();
        let mut out = vec![0.0f32; n_batch * rows];
        gemm_bf16(&weight_le, rows, cols, &x, n_batch, &mut out).unwrap();
        for b in 0..n_batch {
            let mut expect = vec![0.0f32; rows];
            gemv_bf16(
                &weight_le,
                rows,
                cols,
                &x[b * cols..(b + 1) * cols],
                &mut expect,
            )
            .unwrap();
            for r in 0..rows {
                let got = out[b * rows + r];
                let e = expect[r];
                let abs = (got - e).abs();
                assert!(
                    abs < 1e-3,
                    "batch {b} row {r}: got {got} expect {e} abs={abs}"
                );
            }
        }
    }
}
