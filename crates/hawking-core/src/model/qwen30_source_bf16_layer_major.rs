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
use std::time::Instant;

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
    // Unrolled 8-wide conversion keeps the inner loop tight (no per-elem fn call).
    const PARALLEL_THRESHOLD: usize = 256 * 1024; // elements
    if n < PARALLEL_THRESHOLD {
        widen_bf16_slice(weight_le, &mut out[..n]);
        return Ok(());
    }
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .clamp(1, 12);
    let chunk = n.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (t, out_chunk) in out[..n].chunks_mut(chunk).enumerate() {
            let base = t * chunk;
            let w = weight_le;
            scope.spawn(move || {
                let byte_base = base * 2;
                let byte_end = byte_base + out_chunk.len() * 2;
                widen_bf16_slice(&w[byte_base..byte_end], out_chunk);
            });
        }
    });
    Ok(())
}

/// BF16-LE → f32 for a contiguous slice; 8-wide unrolled body.
#[inline]
fn widen_bf16_slice(weight_le: &[u8], out: &mut [f32]) {
    let n = out.len();
    debug_assert!(weight_le.len() >= n * 2);
    let mut i = 0usize;
    while i + 8 <= n {
        let b = i * 2;
        out[i] = f32::from_bits((u16::from_le_bytes([weight_le[b], weight_le[b + 1]]) as u32) << 16);
        out[i + 1] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 2], weight_le[b + 3]]) as u32) << 16,
        );
        out[i + 2] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 4], weight_le[b + 5]]) as u32) << 16,
        );
        out[i + 3] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 6], weight_le[b + 7]]) as u32) << 16,
        );
        out[i + 4] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 8], weight_le[b + 9]]) as u32) << 16,
        );
        out[i + 5] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 10], weight_le[b + 11]]) as u32) << 16,
        );
        out[i + 6] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 12], weight_le[b + 13]]) as u32) << 16,
        );
        out[i + 7] = f32::from_bits(
            (u16::from_le_bytes([weight_le[b + 14], weight_le[b + 15]]) as u32) << 16,
        );
        i += 8;
    }
    while i < n {
        let b = i * 2;
        out[i] = f32::from_bits((u16::from_le_bytes([weight_le[b], weight_le[b + 1]]) as u32) << 16);
        i += 1;
    }
}

/// Run one expert's gate/up/down as batched GEMMs against gathered token rows.
///
/// Widens BF16 → f32 for this expert only (then drops), so peak resident expert
/// weights stay ~18 MiB rather than ~2.3 GiB for all 128 experts at once.
/// Leaves the weighted expert contribution in `down[0..n*h]` (already scaled by
/// route weight) and does **not** scatter — callers scatter into the shared
/// residual under whatever concurrency scheme they use.
fn expert_batched_moe_compute(
    expert: &ExpertWeights,
    members: &[(usize, f32)],
    all_router_in: &[f32],
    h: usize,
    inter: usize,
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

    // Fold route weight into down so scatter is a plain add.
    for (i, &(_, w)) in members.iter().enumerate() {
        let row = &mut down[i * h..(i + 1) * h];
        for j in 0..h {
            row[j] *= w;
        }
    }
    Ok(())
}

#[inline]
fn scatter_expert_down(
    members: &[(usize, f32)],
    down: &[f32],
    h: usize,
    moe_out: &mut [f32],
) {
    for (i, &(t, _)) in members.iter().enumerate() {
        let src = &down[i * h..(i + 1) * h];
        let dst = &mut moe_out[t * h..(t + 1) * h];
        for j in 0..h {
            dst[j] += src[j];
        }
    }
}

/// Serial expert path: compute + scatter into `moe_out` with no locking.
fn expert_batched_moe(
    expert: &ExpertWeights,
    members: &[(usize, f32)],
    all_router_in: &[f32],
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
    x_g: &mut [f32],
    gu_out: &mut [f32],
    act: &mut [f32],
    down: &mut [f32],
    w_gu: &mut [f32],
    w_down: &mut [f32],
) -> Result<()> {
    expert_batched_moe_compute(
        expert,
        members,
        all_router_in,
        h,
        inter,
        x_g,
        gu_out,
        act,
        down,
        w_gu,
        w_down,
    )?;
    scatter_expert_down(members, down, h, moe_out);
    Ok(())
}

/// Host baseline: serial experts, one Accelerate GEMM pair per expert.
///
/// Parallelism is inside Accelerate sgemm (VECLIB threads). Holding all experts'
/// down buffers at once is tokens×top_k×h×4 ≈ 7.8 GiB on the sealed corpus and
/// blows the 12 GiB RSS cap; one-expert-at-a-time keeps expert scratch to
/// ~O(max_n×h). Dispatch count: 2 × n_active per layer (gate_up + down).
fn moe_all_experts_host(
    layer: &mut LoadedLayer,
    expert_members: &[Vec<(usize, f32)>],
    all_router_in: &[f32],
    active: &[usize],
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
) -> Result<()> {
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
    for &e in active {
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
    Ok(())
}

/// Grouped expert MoE via MPS multi-encode (one commit/wait per chunk).
///
/// Tokens are already sorted by expert in `expert_members[e]`. Experts are
/// chunked by **sum of membership rows** (no zero-pad) so scratch stays inside
/// the 12 GiB streamed RSS cap:
///   1. gather X rows contiguously per expert (no pad)
///   2. widen W_gu for the chunk
///   3. ONE command-buffer: encode B variable-M GEMMs, single wait → gu_out
///   4. SiLU·up on host
///   5. widen W_down; ONE CB multi-encode → down
///   6. route-weight + scatter into moe_out
///
/// When a chunk has uniform M (or B==1), uses the true batched
/// `MPSMatrixMultiplication.batchSize` API. Otherwise uses var-M multi-encode
/// into one command buffer (still O(1) waits, not O(B)).
///
/// Target dispatches/layer: 2 × n_chunks  (≪ 2 × 128).
#[cfg(target_os = "macos")]
fn moe_all_experts_grouped_mps(
    layer: &mut LoadedLayer,
    expert_members: &[Vec<(usize, f32)>],
    all_router_in: &[f32],
    active: &[usize],
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
    gpu: &mut CaptureMetalGemm,
) -> Result<()> {
    if active.is_empty() {
        return Ok(());
    }
    let gu_rows = 2 * inter;
    // Host-side X rows budget. Peak live scratch ≈ X + gu_out + MPS pool copy
    // of same ≈ 2×(sum_n*h + sum_n*gu_rows)*4. Full-corpus host baseline peaks
    // ~9.6 GiB; 400 MiB X budget keeps capture peak_rss_bytes under 12 GiB
    // (measured 10.8 GiB at 220 MiB; room for larger chunks / fewer waits).
    const ROW_SCRATCH_BUDGET_BYTES: usize = 400 * 1024 * 1024;
    let max_sum_n = (ROW_SCRATCH_BUDGET_BYTES / (h.saturating_mul(4)).max(1)).max(1);

    let mut i = 0usize;
    while i < active.len() {
        let mut j = i;
        let mut sum_n = 0usize;
        while j < active.len() {
            let n = expert_members[active[j]].len();
            if sum_n > 0 && sum_n + n > max_sum_n {
                break;
            }
            sum_n += n;
            j += 1;
            // Prefer larger chunks for GPU occupancy, but never exceed budget.
            if sum_n >= max_sum_n {
                break;
            }
        }
        let chunk = &active[i..j];
        let b = chunk.len();
        if sum_n == 0 {
            for &e in chunk {
                layer.experts[e].gate = Vec::new();
                layer.experts[e].up = Vec::new();
                layer.experts[e].down = Vec::new();
            }
            i = j;
            continue;
        }

        // Contiguous unpadded packs: offsets[bi] is start row of expert bi.
        let mut counts = vec![0usize; b];
        let mut offsets = vec![0usize; b];
        let mut row = 0usize;
        for (bi, &e) in chunk.iter().enumerate() {
            offsets[bi] = row;
            counts[bi] = expert_members[e].len();
            row += counts[bi];
        }
        debug_assert_eq!(row, sum_n);

        let mut x_pack = vec![0.0f32; sum_n * h];
        let mut w_gu = vec![0.0f32; b * gu_rows * h];
        for (bi, &e) in chunk.iter().enumerate() {
            let members = &expert_members[e];
            let base = offsets[bi] * h;
            for (ri, &(t, _)) in members.iter().enumerate() {
                x_pack[base + ri * h..base + (ri + 1) * h]
                    .copy_from_slice(&all_router_in[t * h..(t + 1) * h]);
            }
            let w_base = bi * gu_rows * h;
            widen_bf16_into(
                &layer.experts[e].gate,
                inter,
                h,
                &mut w_gu[w_base..w_base + inter * h],
            )?;
            widen_bf16_into(
                &layer.experts[e].up,
                inter,
                h,
                &mut w_gu[w_base + inter * h..w_base + gu_rows * h],
            )?;
            layer.experts[e].gate = Vec::new();
            layer.experts[e].up = Vec::new();
        }

        let mut gu_out = vec![0.0f32; sum_n * gu_rows];
        gpu.gemm_x_wt_grouped_var_m(
            &x_pack,
            &w_gu,
            &mut gu_out,
            &counts,
            &offsets,
            gu_rows,
            h,
        )?;
        drop(x_pack);
        drop(w_gu);

        // SiLU(gate)*up → act [sum_n, inter]
        let mut act = vec![0.0f32; sum_n * inter];
        for ri in 0..sum_n {
            let gu = &gu_out[ri * gu_rows..(ri + 1) * gu_rows];
            let g = &gu[..inter];
            let u = &gu[inter..];
            let a = &mut act[ri * inter..(ri + 1) * inter];
            for j in 0..inter {
                let gv = g[j];
                a[j] = (gv / (1.0 + (-gv).exp())) * u[j];
            }
        }
        drop(gu_out);

        let mut w_down = vec![0.0f32; b * h * inter];
        for (bi, &e) in chunk.iter().enumerate() {
            let w_base = bi * h * inter;
            widen_bf16_into(
                &layer.experts[e].down,
                h,
                inter,
                &mut w_down[w_base..w_base + h * inter],
            )?;
            layer.experts[e].down = Vec::new();
        }

        let mut down = vec![0.0f32; sum_n * h];
        gpu.gemm_x_wt_grouped_var_m(&act, &w_down, &mut down, &counts, &offsets, h, inter)?;
        drop(act);
        drop(w_down);

        for (bi, &e) in chunk.iter().enumerate() {
            let members = &expert_members[e];
            let base = offsets[bi];
            for (ri, &(t, w)) in members.iter().enumerate() {
                let src = &down[(base + ri) * h..(base + ri + 1) * h];
                let dst = &mut moe_out[t * h..(t + 1) * h];
                for j in 0..h {
                    dst[j] += src[j] * w;
                }
            }
        }
        drop(down);
        i = j;
    }
    Ok(())
}

/// MoE over the flat token corpus at one layer.
///
/// When `gpu` is `Some` and compute mode is metal, uses grouped/batched MPS
/// expert GEMM (O(chunks) dispatches/layer). Otherwise host Accelerate serial
/// experts (baseline). Attention stays on host either way.
fn moe_all_experts_parallel(
    layer: &mut LoadedLayer,
    expert_members: &[Vec<(usize, f32)>],
    all_router_in: &[f32],
    _total_tokens: usize,
    h: usize,
    inter: usize,
    moe_out: &mut [f32],
    gpu: Option<&mut CaptureMetalGemm>,
) -> Result<()> {
    moe_out.fill(0.0);

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

    #[cfg(target_os = "macos")]
    {
        if let Some(gpu) = gpu {
            return moe_all_experts_grouped_mps(
                layer,
                expert_members,
                all_router_in,
                &active,
                h,
                inter,
                moe_out,
                gpu,
            );
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        let _ = gpu;
    }
    moe_all_experts_host(
        layer,
        expert_members,
        all_router_in,
        &active,
        h,
        inter,
        moe_out,
    )
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

/// Compute backend for capture expert GEMMs. Router always stays on host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureComputeBackend {
    /// Accelerate cblas_sgemm on CPU (control / fallback).
    Host,
    /// Grouped/batched MPSMatrixMultiplication for expert MoE GEMMs.
    Metal,
}

impl CaptureComputeBackend {
    /// `HAWKING_CAPTURE_COMPUTE=host|metal` (default: metal on macOS).
    pub fn from_env() -> Self {
        match std::env::var("HAWKING_CAPTURE_COMPUTE")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str()
        {
            "host" | "cpu" | "accelerate" => Self::Host,
            "metal" | "mps" | "gpu" => Self::Metal,
            "" => {
                #[cfg(target_os = "macos")]
                {
                    Self::Metal
                }
                #[cfg(not(target_os = "macos"))]
                {
                    Self::Host
                }
            }
            other => {
                eprintln!(
                    "warning: unknown HAWKING_CAPTURE_COMPUTE={other:?}; using host"
                );
                Self::Host
            }
        }
    }
}

/// Metal/MPS f32 GEMM backend for capture expert matmuls.
///
/// Uses Apple's `MPSMatrixMultiplication` (single and batched via
/// `matrices`/`batchSize`) through a tiny ObjC bridge with a process-global
/// shared-buffer pool + per-call `@autoreleasepool`.
///
/// Router GEMMs stay on host Accelerate — FP reassociation there has flipped
/// top-k three times in this campaign. Attention stays host-parallel (serial
/// Metal attention lost probe parallelism in the prior lane).
#[cfg(target_os = "macos")]
pub struct CaptureMetalGemm {
    device: metal::Device,
    queue: metal::CommandQueue,
}

#[cfg(target_os = "macos")]
extern "C" {
    fn hawking_capture_mps_gemm_x_wt(
        device: *mut std::ffi::c_void,
        queue: *mut std::ffi::c_void,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;
    fn hawking_capture_mps_gemm_x_wt_batched(
        device: *mut std::ffi::c_void,
        queue: *mut std::ffi::c_void,
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        b: u32,
        m: u32,
        n: u32,
        k: u32,
    ) -> i32;
    fn hawking_capture_mps_gemm_x_wt_grouped_varM(
        device: *mut std::ffi::c_void,
        queue: *mut std::ffi::c_void,
        xs: *const *const f32,
        ws: *const *const f32,
        outs: *const *mut f32,
        ms: *const u32,
        b: u32,
        n: u32,
        k: u32,
    ) -> i32;
    fn hawking_capture_mps_dispatch_count() -> u64;
    fn hawking_capture_mps_dispatch_count_reset();
}

#[cfg(target_os = "macos")]
impl CaptureMetalGemm {
    pub fn new() -> Result<Self> {
        use metal::Device;
        let device = Device::system_default()
            .ok_or_else(|| model_err("CaptureMetalGemm: no Metal device"))?;
        let queue = device.new_command_queue();
        Ok(Self { device, queue })
    }

    fn device_ptr(&self) -> *mut std::ffi::c_void {
        use metal::foreign_types::ForeignType;
        self.device.as_ptr() as *mut std::ffi::c_void
    }

    fn queue_ptr(&self) -> *mut std::ffi::c_void {
        use metal::foreign_types::ForeignType;
        self.queue.as_ptr() as *mut std::ffi::c_void
    }

    /// Reset process-lifetime MPS commit/wait counter (for measurement).
    pub fn reset_dispatch_count() {
        unsafe { hawking_capture_mps_dispatch_count_reset() }
    }

    /// Number of MPS command-buffer commit/wait pairs since last reset.
    pub fn dispatch_count() -> u64 {
        unsafe { hawking_capture_mps_dispatch_count() }
    }

    /// `Out = X @ Wᵀ` with X [M,K], W [N,K], Out [M,N].
    pub fn gemm_w_times_x(
        &mut self,
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        n_batch: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if n_batch == 0 || rows == 0 || cols == 0 {
            return Ok(());
        }
        let rc = unsafe {
            hawking_capture_mps_gemm_x_wt(
                self.device_ptr(),
                self.queue_ptr(),
                x.as_ptr(),
                w.as_ptr(),
                out.as_mut_ptr(),
                n_batch as u32,
                rows as u32,
                cols as u32,
            )
        };
        if rc != 0 {
            return Err(model_err(format!(
                "MPS gemm_x_wt failed rc={rc} M={n_batch} N={rows} K={cols}"
            )));
        }
        Ok(())
    }

    /// Batched: for b in 0..B, Out_b = X_b @ W_b^T with identical (M,N,K).
    /// Layouts are matrix-major: matrix then row-major rows.
    pub fn gemm_x_wt_batched(
        &mut self,
        x: &[f32],
        w: &[f32],
        out: &mut [f32],
        batch: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if batch == 0 || m == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        debug_assert_eq!(x.len(), batch * m * k);
        debug_assert_eq!(w.len(), batch * n * k);
        debug_assert_eq!(out.len(), batch * m * n);
        let rc = unsafe {
            hawking_capture_mps_gemm_x_wt_batched(
                self.device_ptr(),
                self.queue_ptr(),
                x.as_ptr(),
                w.as_ptr(),
                out.as_mut_ptr(),
                batch as u32,
                m as u32,
                n as u32,
                k as u32,
            )
        };
        if rc != 0 {
            return Err(model_err(format!(
                "MPS gemm_x_wt_batched failed rc={rc} B={batch} M={m} N={n} K={k}"
            )));
        }
        Ok(())
    }

    /// Grouped var-M: experts share (N,K) but each has its own M_b.
    ///
    /// `x` / `out` are contiguous row packs with expert `bi` starting at
    /// `offsets[bi]` rows; `w` is B contiguous weight matrices of shape [N,K].
    /// Uses true batch API when all M equal; otherwise one multi-encode CB.
    pub fn gemm_x_wt_grouped_var_m(
        &mut self,
        x: &[f32],
        w: &[f32],
        out: &mut [f32],
        counts: &[usize],
        offsets: &[usize],
        n: usize,
        k: usize,
    ) -> Result<()> {
        let b = counts.len();
        if b == 0 || n == 0 || k == 0 {
            return Ok(());
        }
        debug_assert_eq!(offsets.len(), b);
        debug_assert_eq!(w.len(), b * n * k);
        let sum_n: usize = counts.iter().sum();
        if sum_n == 0 {
            return Ok(());
        }
        debug_assert_eq!(x.len(), sum_n * k);
        debug_assert_eq!(out.len(), sum_n * n);

        // Fast path: uniform M → true MPS batch (one encode).
        let m0 = counts[0];
        if counts.iter().all(|&c| c == m0) && m0 > 0 {
            return self.gemm_x_wt_batched(x, w, out, b, m0, n, k);
        }

        // Var-M: pointer arrays into the contiguous packs.
        let mut xs: Vec<*const f32> = Vec::with_capacity(b);
        let mut ws: Vec<*const f32> = Vec::with_capacity(b);
        let mut outs: Vec<*mut f32> = Vec::with_capacity(b);
        let mut ms: Vec<u32> = Vec::with_capacity(b);
        for bi in 0..b {
            let m = counts[bi];
            ms.push(m as u32);
            if m == 0 {
                xs.push(std::ptr::null());
                ws.push(std::ptr::null());
                outs.push(std::ptr::null_mut());
                continue;
            }
            let off = offsets[bi];
            xs.push(unsafe { x.as_ptr().add(off * k) });
            ws.push(unsafe { w.as_ptr().add(bi * n * k) });
            outs.push(unsafe { out.as_mut_ptr().add(off * n) });
        }
        let rc = unsafe {
            hawking_capture_mps_gemm_x_wt_grouped_varM(
                self.device_ptr(),
                self.queue_ptr(),
                xs.as_ptr(),
                ws.as_ptr(),
                outs.as_ptr(),
                ms.as_ptr(),
                b as u32,
                n as u32,
                k as u32,
            )
        };
        if rc != 0 {
            return Err(model_err(format!(
                "MPS gemm_x_wt_grouped_varM failed rc={rc} B={b} N={n} K={k} sum_n={sum_n}"
            )));
        }
        Ok(())
    }
}

#[cfg(not(target_os = "macos"))]
pub struct CaptureMetalGemm;

#[cfg(not(target_os = "macos"))]
impl CaptureMetalGemm {
    pub fn new() -> Result<Self> {
        Err(model_err("CaptureMetalGemm requires macOS"))
    }
    pub fn reset_dispatch_count() {}
    pub fn dispatch_count() -> u64 {
        0
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

/// Default retained router-input rows per (layer, expert) under first-N retention.
///
/// Chosen so organs see enough fit rows to pass the null test (failure is sharp
/// below ~16 rows; flattens above ~32). At N=64 × 128 experts the worst-case
/// unique rows/layer is 8192 (≈64 MiB of f32@2048), which is the documented
/// streamed budget for this capture path.
pub const DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT: usize = 64;

/// Wall-clock buckets for one capture_all_layers run (summed over all layers).
#[derive(Clone, Debug, Default)]
pub struct CaptureTiming {
    pub load_secs: f64,
    pub bf16_widen_dense_secs: f64,
    pub attention_secs: f64,
    pub router_secs: f64,
    pub expert_gemm_secs: f64,
    pub retention_write_secs: f64,
    pub total_secs: f64,
    /// MPS command-buffer commit/wait count for expert GEMMs (0 on host path).
    pub expert_mps_dispatches: u64,
    /// Compute backend label: "host" or "metal_grouped".
    pub compute_backend: String,
}

/// Per-layer first-N saturation: flat token index at which every expert first
/// held `N` retained credits (None if the corpus never filled every expert).
#[derive(Clone, Debug, Default)]
pub struct SaturationStats {
    /// `layer_all_experts_full_at_token[L] = Some(t)` means at flat token `t`
    /// (0-based, inclusive), every expert at layer L first reached N.
    pub layer_all_experts_full_at_token: Vec<Option<usize>>,
    /// Max saturation token index across layers that did saturate.
    pub global_max_saturation_token: Option<usize>,
    /// Fraction of corpus tokens strictly after the global max saturation token.
    pub pct_corpus_after_global_saturation: f64,
}

/// Result of [`capture_all_layers`]: captures plus timing/saturation telemetry.
#[derive(Clone, Debug)]
pub struct CaptureAllLayersResult {
    pub captures: Vec<Vec<Vec<LayerTokenCapture>>>,
    pub timing: CaptureTiming,
    pub saturation: SaturationStats,
}

/// Deterministic per-expert first-N retention for one token.
///
/// Walk tokens in global order. For each top-k expert that still has fewer than
/// `max_per_expert` retained members, credit this token to that expert. Retain
/// the token's router-input hidden if any expert still needed a slot.
///
/// This is deliberately not a random reservoir: the same corpus + same N yields
/// byte-identical retention across runs. Route membership is independent and
/// stays complete for every token.
pub fn credit_expert_first_n_retention(
    expert_retained: &mut [usize],
    selected_expert_ids: &[u32],
    max_per_expert: usize,
) -> bool {
    if max_per_expert == 0 || selected_expert_ids.is_empty() {
        return false;
    }
    let mut retain = false;
    for &expert in selected_expert_ids {
        let e = expert as usize;
        if e >= expert_retained.len() {
            continue;
        }
        if expert_retained[e] < max_per_expert {
            expert_retained[e] += 1;
            retain = true;
        }
    }
    retain
}

/// True iff every expert has at least `max_per_expert` retained credits.
#[inline]
fn all_experts_saturated(expert_retained: &[usize], max_per_expert: usize) -> bool {
    expert_retained.iter().all(|&c| c >= max_per_expert)
}

/// Layer-major full forward over all probes: returns per-probe per-layer per-token captures
/// and leaves `hiddens` as the final residuals.
///
/// Per layer this:
/// 1. Streams attention probe-by-probe (causal within probe) with batched Q/K/V/O GEMMs
/// 2. Packs every token's post-attention RMSNorm (router input) into one matrix
/// 3. Runs one router GEMM over the whole corpus at this layer
/// 4. Per expert: gather tokens that selected it, one gate/up/down GEMM, scatter-add
/// 5. Retains router-input hiddens under **per-expert first-N** (see
///    [`credit_expert_first_n_retention`]): the first `max_hidden_tokens_per_expert`
///    tokens that route to expert E keep their hidden for that layer. Full route
///    membership is always recorded.
///
/// Prefetches layer L+1 weights on a background thread while layer L computes.
///
/// Expert GEMMs always run for the full membership set: residual after MoE feeds
/// the next layer's attention and router, so dropping them after retention
/// saturation would flip later-layer routes (not bit-identical). Saturation is
/// measured and reported so the free "hidden materialization" exit is quantified.
pub fn capture_all_layers(
    index: &SourceBf16Index,
    probes: &[(String, Vec<u32>)],
    hiddens: &mut [ProbeHidden],
    max_hidden_tokens_per_expert: usize,
    mut on_layer: Option<&mut dyn FnMut(usize, u64)>,
) -> Result<CaptureAllLayersResult> {
    if hiddens.len() != probes.len() {
        return Err(model_err("hiddens/probes length mismatch"));
    }
    let wall0 = Instant::now();
    let mut timing = CaptureTiming::default();

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

    // Largest probe length — each attention worker allocates private scratch of this size.
    let max_seq = probes.iter().map(|(_, t)| t.len()).max().unwrap_or(0);

    // MoE output + router surfaces. Expert f32 weights and gather scratch are
    // allocated inside moe_all_experts_parallel.
    let mut moe_out = vec![0.0f32; total_tokens * h];
    let mut all_router_in = vec![0.0f32; total_tokens * h];
    let mut router_logits = vec![0.0f32; total_tokens * QWEN30_EXPERTS];

    let mut layer_sat: Vec<Option<usize>> = vec![None; QWEN30_LAYERS];

    // Expert GEMM backend. Attention stays host-parallel (prior lane measured
    // serial Metal attention slower than 8-way host). Router always host.
    let backend = CaptureComputeBackend::from_env();
    let mut metal_gemm: Option<CaptureMetalGemm> = match backend {
        CaptureComputeBackend::Metal => {
            #[cfg(target_os = "macos")]
            {
                match CaptureMetalGemm::new() {
                    Ok(g) => {
                        CaptureMetalGemm::reset_dispatch_count();
                        timing.compute_backend = "metal_grouped".into();
                        Some(g)
                    }
                    Err(e) => {
                        eprintln!(
                            "warning: Metal GEMM unavailable ({e}); falling back to host Accelerate"
                        );
                        timing.compute_backend = "host".into();
                        None
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                timing.compute_backend = "host".into();
                None
            }
        }
        CaptureComputeBackend::Host => {
            timing.compute_backend = "host".into();
            None
        }
    };

    // Load layer 0; each iteration prefetches L+1 while computing L.
    let t_load0 = Instant::now();
    let mut next_layer: Option<LoadedLayer> = Some(LoadedLayer::load(index, 0)?);
    timing.load_secs += t_load0.elapsed().as_secs_f64();

    for layer_idx in 0..QWEN30_LAYERS {
        let mut layer = next_layer
            .take()
            .ok_or_else(|| model_err(format!("missing layer {layer_idx}")))?;

        // Overlap load(L+1) with compute(L). Join at end of scope yields the next layer.
        let prefetched = std::thread::scope(|scope| -> Result<Option<LoadedLayer>> {
            let prefetch = if layer_idx + 1 < QWEN30_LAYERS {
                let next = layer_idx + 1;
                Some(scope.spawn(move || LoadedLayer::load(index, next)))
            } else {
                None
            };

            let resident = layer.resident_bytes;
            if let Some(cb) = on_layer.as_mut() {
                cb(layer_idx, resident);
            }

            // Widen dense projections once for this layer; free BF16 payloads.
            let t_widen = Instant::now();
            let q_w = widen_bf16_mat(&layer.q_proj, q_dim, h)?;
            let k_w = widen_bf16_mat(&layer.k_proj, kv_dim, h)?;
            let v_w = widen_bf16_mat(&layer.v_proj, kv_dim, h)?;
            let o_w = widen_bf16_mat(&layer.o_proj, h, q_dim)?;
            let router_w = widen_bf16_mat(&layer.router, QWEN30_EXPERTS, h)?;
            timing.bf16_widen_dense_secs += t_widen.elapsed().as_secs_f64();
            layer.q_proj = Vec::new();
            layer.k_proj = Vec::new();
            layer.v_proj = Vec::new();
            layer.o_proj = Vec::new();
            layer.router = Vec::new();

            // --- Phase 1: attention per probe (parallel across probes), collect router inputs. ---
            let t_attn = Instant::now();
            // Precompute flat offsets so workers write disjoint slices of all_router_in.
            let mut probe_flat_start = vec![0usize; probes.len()];
            {
                let mut acc = 0usize;
                for (pi, (_, tokens)) in probes.iter().enumerate() {
                    probe_flat_start[pi] = acc;
                    acc += tokens.len();
                }
                debug_assert_eq!(acc, total_tokens);
            }
            // Build token_index in global order (probe-major, position order).
            let mut token_index: Vec<(usize, usize)> = Vec::with_capacity(total_tokens);
            for (pi, (_, tokens)) in probes.iter().enumerate() {
                for pos in 0..tokens.len() {
                    token_index.push((pi, pos));
                }
            }

            // Parallel attention over probes. Each probe is independent (own residual,
            // own causal window). Shared dense weights are read-only; each worker
            // owns private scratch sized to the largest probe.
            let n_attn_workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
                .clamp(1, 8)
                .min(probes.len().max(1));
            let err_attn: Mutex<Option<String>> = Mutex::new(None);
            // SAFETY: hiddens partitions by probe; all_router_in partitions by flat offset.
            let hiddens_addr = hiddens.as_mut_ptr() as usize;
            let hiddens_len = hiddens.len();
            let router_in_addr = all_router_in.as_mut_ptr() as usize;
            let router_in_len = all_router_in.len();
            let input_ln = &layer.input_layernorm;
            let post_ln = &layer.post_attention_layernorm;
            let q_norm = &layer.q_norm;
            let k_norm = &layer.k_norm;

            std::thread::scope(|scope| {
                let chunk = probes.len().div_ceil(n_attn_workers);
                for wi in 0..n_attn_workers {
                    let start_pi = wi * chunk;
                    if start_pi >= probes.len() {
                        break;
                    }
                    let end_pi = (start_pi + chunk).min(probes.len());
                    let probes = probes;
                    let probe_flat_start = &probe_flat_start;
                    let err_attn = &err_attn;
                    let q_w = &q_w;
                    let k_w = &k_w;
                    let v_w = &v_w;
                    let o_w = &o_w;
                    scope.spawn(move || {
                        let mut x_norm = vec![0.0f32; max_seq * h];
                        let mut q = vec![0.0f32; max_seq * q_dim];
                        let mut k_cache = vec![0.0f32; max_seq * kv_dim];
                        let mut v_cache = vec![0.0f32; max_seq * kv_dim];
                        let mut attn = vec![0.0f32; max_seq * q_dim];
                        let mut attn_proj = vec![0.0f32; max_seq * h];
                        for pi in start_pi..end_pi {
                            let seq_len = probes[pi].1.len();
                            if seq_len == 0 {
                                continue;
                            }
                            // Exclusive probe residual slice.
                            let hidden = unsafe {
                                let base = hiddens_addr as *mut ProbeHidden;
                                &mut *base.add(pi)
                            };
                            if hidden.len() != seq_len * h {
                                if let Ok(mut g) = err_attn.lock() {
                                    *g = Some(format!(
                                        "layer {layer_idx} probe {pi}: hidden len {} != {seq_len}*{h}",
                                        hidden.len()
                                    ));
                                }
                                return;
                            }
                            let x_norm = &mut x_norm[..seq_len * h];
                            for pos in 0..seq_len {
                                rmsnorm(
                                    &hidden[pos * h..(pos + 1) * h],
                                    input_ln,
                                    QWEN30_RMS_EPS,
                                    &mut x_norm[pos * h..(pos + 1) * h],
                                );
                            }
                            let q = &mut q[..seq_len * q_dim];
                            let k_cache = &mut k_cache[..seq_len * kv_dim];
                            let v_cache = &mut v_cache[..seq_len * kv_dim];
                            if let Err(e) = gemm_f32(q_w, q_dim, h, x_norm, seq_len, q) {
                                if let Ok(mut g) = err_attn.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                            if let Err(e) = gemm_f32(k_w, kv_dim, h, x_norm, seq_len, k_cache) {
                                if let Ok(mut g) = err_attn.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                            if let Err(e) = gemm_f32(v_w, kv_dim, h, x_norm, seq_len, v_cache) {
                                if let Ok(mut g) = err_attn.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                            for pos in 0..seq_len {
                                let q_row = &mut q[pos * q_dim..(pos + 1) * q_dim];
                                let k_row = &mut k_cache[pos * kv_dim..(pos + 1) * kv_dim];
                                if let Err(e) =
                                    rmsnorm_rows(q_row, q_norm, QWEN30_HEADS, QWEN30_HEAD_DIM)
                                {
                                    if let Ok(mut g) = err_attn.lock() {
                                        *g = Some(e.to_string());
                                    }
                                    return;
                                }
                                if let Err(e) =
                                    rmsnorm_rows(k_row, k_norm, QWEN30_KV_HEADS, QWEN30_HEAD_DIM)
                                {
                                    if let Ok(mut g) = err_attn.lock() {
                                        *g = Some(e.to_string());
                                    }
                                    return;
                                }
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
                            let attn = &mut attn[..seq_len * q_dim];
                            if let Err(e) = mha_prefill_causal(
                                q,
                                k_cache,
                                v_cache,
                                QWEN30_HEADS,
                                QWEN30_KV_HEADS,
                                QWEN30_HEAD_DIM,
                                seq_len,
                                attn,
                            ) {
                                if let Ok(mut g) = err_attn.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                            let attn_proj = &mut attn_proj[..seq_len * h];
                            if let Err(e) = gemm_f32(o_w, h, q_dim, attn, seq_len, attn_proj) {
                                if let Ok(mut g) = err_attn.lock() {
                                    *g = Some(e.to_string());
                                }
                                return;
                            }
                            for pos in 0..seq_len {
                                add_inplace(
                                    &mut hidden[pos * h..(pos + 1) * h],
                                    &attn_proj[pos * h..(pos + 1) * h],
                                );
                            }
                            let flat0 = probe_flat_start[pi];
                            let router_in = unsafe {
                                std::slice::from_raw_parts_mut(
                                    router_in_addr as *mut f32,
                                    router_in_len,
                                )
                            };
                            for pos in 0..seq_len {
                                let flat_t = flat0 + pos;
                                rmsnorm(
                                    &hidden[pos * h..(pos + 1) * h],
                                    post_ln,
                                    QWEN30_RMS_EPS,
                                    &mut router_in[flat_t * h..(flat_t + 1) * h],
                                );
                            }
                            let _ = hiddens_len; // silence
                        }
                    });
                }
            });
            if let Some(msg) = err_attn.into_inner().unwrap_or(None) {
                return Err(model_err(msg));
            }
            timing.attention_secs += t_attn.elapsed().as_secs_f64();
            drop(q_w);
            drop(k_w);
            drop(v_w);
            drop(o_w);

            // --- Phase 2: router GEMM + top-k for every token. ---
            let t_router = Instant::now();
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
            timing.router_secs += t_router.elapsed().as_secs_f64();

            // --- Phase 3: full MoE residual (required for later-layer routes). ---
            let t_moe = Instant::now();
            moe_all_experts_parallel(
                &mut layer,
                &expert_members,
                &all_router_in,
                t_all,
                h,
                inter,
                &mut moe_out,
                metal_gemm.as_mut(),
            )?;
            timing.expert_gemm_secs += t_moe.elapsed().as_secs_f64();

            // Residual + retention. Route membership always recorded; hiddens only
            // while some expert still has an open first-N slot.
            let t_ret = Instant::now();
            let mut expert_retained = vec![0usize; QWEN30_EXPERTS];
            let mut sat_at: Option<usize> = None;
            for (t, &(pi, pos)) in token_index.iter().enumerate() {
                add_inplace(
                    &mut hiddens[pi][pos * h..(pos + 1) * h],
                    &moe_out[t * h..(t + 1) * h],
                );
                let (ids, weights) = std::mem::take(&mut routes[t]);
                let retain = credit_expert_first_n_retention(
                    &mut expert_retained,
                    &ids,
                    max_hidden_tokens_per_expert,
                );
                if sat_at.is_none()
                    && all_experts_saturated(&expert_retained, max_hidden_tokens_per_expert)
                {
                    sat_at = Some(t);
                }
                captures[pi][pos].push(LayerTokenCapture {
                    layer: layer_idx,
                    selected_expert_ids: ids,
                    normalized_route_weights: weights,
                    router_input_hidden: if retain {
                        all_router_in[t * h..(t + 1) * h].to_vec()
                    } else {
                        Vec::new()
                    },
                });
            }
            layer_sat[layer_idx] = sat_at;
            timing.retention_write_secs += t_ret.elapsed().as_secs_f64();
            drop(layer);

            // Join next-layer load (overlapped with everything above).
            let t_join = Instant::now();
            let next = match prefetch {
                Some(handle) => Some(
                    handle
                        .join()
                        .map_err(|_| model_err("layer prefetch thread panicked"))??,
                ),
                None => None,
            };
            // Only the wait residual counts as load time; pure overlap is free.
            timing.load_secs += t_join.elapsed().as_secs_f64();
            Ok(next)
        })?;

        next_layer = prefetched;
    }

    timing.total_secs = wall0.elapsed().as_secs_f64();
    timing.expert_mps_dispatches = CaptureMetalGemm::dispatch_count();
    // Drop Metal resources before returning so RSS telemetry is clean.
    drop(metal_gemm);
    let global_max = layer_sat.iter().flatten().copied().max();
    let pct_after = match global_max {
        Some(t) if total_tokens > 0 => {
            let after = total_tokens.saturating_sub(t + 1);
            100.0 * after as f64 / total_tokens as f64
        }
        _ => 0.0,
    };
    Ok(CaptureAllLayersResult {
        captures,
        timing,
        saturation: SaturationStats {
            layer_all_experts_full_at_token: layer_sat,
            global_max_saturation_token: global_max,
            pct_corpus_after_global_saturation: pct_after,
        },
    })
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
    fn per_expert_first_n_retention_is_deterministic_and_bounded() {
        // Synthetic routes: tokens cycle through disjoint expert pairs so each
        // expert's first-N set is unambiguous.
        let max_n = 3usize;
        let mut counts = vec![0usize; 8];
        let routes: Vec<Vec<u32>> = (0..20u32)
            .map(|t| vec![t % 4, 4 + (t % 4)])
            .collect();
        let mut retained_mask = Vec::new();
        for ids in &routes {
            retained_mask.push(credit_expert_first_n_retention(&mut counts, ids, max_n));
        }
        // Every expert should have been credited exactly max_n times.
        for (e, &c) in counts.iter().enumerate() {
            assert_eq!(c, max_n, "expert {e} retained count");
        }
        // First 3 tokens that hit expert 0 (t=0,4,8) must be retained; later ones
        // that only hit already-full experts must not.
        assert!(retained_mask[0]);
        assert!(retained_mask[4]);
        assert!(retained_mask[8]);
        // After experts 0 and 4 are full (3 credits each from t=0,4,8), token 12
        // routes to the same pair and must be dropped.
        assert!(!retained_mask[12]);
        // Re-running the same sequence must produce the same mask (determinism).
        let mut counts2 = vec![0usize; 8];
        let mask2: Vec<bool> = routes
            .iter()
            .map(|ids| credit_expert_first_n_retention(&mut counts2, ids, max_n))
            .collect();
        assert_eq!(retained_mask, mask2);
        assert_eq!(counts, counts2);
    }

    #[test]
    fn per_expert_first_n_zero_retains_nothing() {
        let mut counts = vec![0usize; 4];
        assert!(!credit_expert_first_n_retention(&mut counts, &[0, 1], 0));
        assert_eq!(counts, vec![0, 0, 0, 0]);
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
