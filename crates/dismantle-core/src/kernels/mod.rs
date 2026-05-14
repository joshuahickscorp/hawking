//! Rust host code that dispatches `.metal` kernels.
//!
//! Each function in this module corresponds to a kernel in
//! `shaders/*.metal`. The Phase 0 reference path runs everything on
//! the CPU in fp32 — correctness-first; real kernels arrive in Phase
//! 1+. Both implementations share the same Rust signature, so the
//! model layer is unchanged when we swap.
//!
//! All host-side ops here operate on plain `[f32]` slices for the
//! Phase 0 reference path. When the Metal kernels land, the same
//! function names will gain a `Tensor`-shaped overload that takes
//! `MTLBuffer`s.

use half::f16;

// -------- common.metal -------------------------------------------------

/// RMS-normalize a row in-place.
///
/// `out = (x / rms(x)) * weight` where `rms(x) = sqrt(mean(x^2) + eps)`.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), weight.len());
    debug_assert_eq!(x.len(), out.len());
    let n = x.len() as f32;
    let mut sum_sq = 0.0f64;
    for &v in x {
        sum_sq += (v as f64) * (v as f64);
    }
    let rms = ((sum_sq / n as f64) as f32 + eps).sqrt();
    let inv = 1.0 / rms;
    for i in 0..x.len() {
        out[i] = x[i] * inv * weight[i];
    }
}

/// SwiGLU activation: `out = silu(gate) * up` where `silu(x) = x * sigmoid(x)`.
pub fn silu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for i in 0..gate.len() {
        let g = gate[i];
        let s = g / (1.0 + (-g).exp());
        out[i] = s * up[i];
    }
}

/// Softmax in place over a slice. Numerically stable.
pub fn softmax_inplace(xs: &mut [f32]) {
    if xs.is_empty() {
        return;
    }
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
    let inv = 1.0 / sum;
    for v in xs.iter_mut() {
        *v *= inv;
    }
}

/// In-place rotary positional embedding for one (head_dim,) vector at
/// absolute position `pos`, using the standard θᵢ = base^(-2i/dim)
/// schedule. The rotary applies in interleaved pairs: (x_{2i}, x_{2i+1}).
pub fn rope_inplace(x: &mut [f32], pos: u32, base: f32) {
    let head_dim = x.len();
    let half = head_dim / 2;
    for i in 0..half {
        let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
        let (sin, cos) = theta.sin_cos();
        let x0 = x[2 * i];
        let x1 = x[2 * i + 1];
        x[2 * i] = x0 * cos - x1 * sin;
        x[2 * i + 1] = x0 * sin + x1 * cos;
    }
}

/// Phase 2 Wedge 2c — apply RoPE to N rotation vectors at N positions in
/// one call. RoPE is element-wise per (vector, position); this helper
/// makes the multi-token call site obvious without changing the math.
///
/// `xs` is N rotation vectors (each of length head_dim, even). `positions`
/// is N positions (one per vector). `base` is the rope theta-base.
///
/// Equivalent to N sequential calls to `rope_inplace`. Bit-identical.
pub fn rope_inplace_batch(xs: &mut [&mut [f32]], positions: &[u32], base: f32) {
    debug_assert_eq!(
        xs.len(),
        positions.len(),
        "rope_inplace_batch: xs.len()={} positions.len()={}",
        xs.len(),
        positions.len(),
    );
    for (x, &pos) in xs.iter_mut().zip(positions.iter()) {
        rope_inplace(*x, pos, base);
    }
}

/// Look up a token embedding row. `embed` is laid out (vocab, hidden).
pub fn embed_lookup(embed: &[f16], hidden: usize, token_id: u32, out: &mut [f32]) {
    let row = token_id as usize * hidden;
    debug_assert_eq!(out.len(), hidden);
    for i in 0..hidden {
        out[i] = embed[row + i].to_f32();
    }
}

// -------- generic GEMV ------------------------------------------------

/// Row-major GEMV: `out = W @ x`, where `W` is (rows, cols) and `x` is (cols,).
/// Phase 0 reference; replaced by Metal in Phase 1+.
pub fn gemv_f16(w: &[f16], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c].to_f32() * x[c];
        }
        out[r] = acc;
    }
}

/// Row-major GEMV from f32 weights.
pub fn gemv_f32(w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(w.len(), rows * cols);
    debug_assert_eq!(x.len(), cols);
    debug_assert_eq!(out.len(), rows);
    for r in 0..rows {
        let mut acc = 0.0f32;
        let row = &w[r * cols..(r + 1) * cols];
        for c in 0..cols {
            acc += row[c] * x[c];
        }
        out[r] = acc;
    }
}

/// Add `b` into `a` in place.
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    debug_assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// Helper used by tests.
pub fn argmax_f32(xs: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best = i;
            best_v = v;
        }
    }
    best as u32
}

/// Weighted gather of per-(token, expert) outputs back into per-token
/// activations. CPU reference for the `moe_gather_combine` Metal kernel.
///
///   token_out[t, h] = Σ_k weights[t, k] * expert_out[t, k, h]
///
///   expert_out: (n_tokens, top_k, hidden) row-major
///   weights:    (n_tokens, top_k)
///   token_out:  (n_tokens, hidden)
pub fn gather_combine(
    expert_out: &[f32],
    weights: &[f32],
    n_tokens: usize,
    top_k: usize,
    hidden: usize,
    token_out: &mut [f32],
) {
    debug_assert_eq!(expert_out.len(), n_tokens * top_k * hidden);
    debug_assert_eq!(weights.len(), n_tokens * top_k);
    debug_assert_eq!(token_out.len(), n_tokens * hidden);

    for t in 0..n_tokens {
        for h in 0..hidden {
            let mut acc = 0.0f32;
            for k in 0..top_k {
                let w = weights[t * top_k + k];
                let v = expert_out[(t * top_k + k) * hidden + h];
                acc += w * v;
            }
            token_out[t * hidden + h] = acc;
        }
    }
}

/// Per-token softmax + top-K selection. CPU reference for the
/// `moe_topk_gate` Metal kernel. Outputs raw post-softmax probabilities
/// (no top-K renormalization), in the same selection order the Metal
/// kernel uses (mask-and-pick-next-max).
///
///   logits: (n_tokens, n_experts) row-major
///   expert_ids_out: (n_tokens, top_k) — selected expert indices
///   weights_out: (n_tokens, top_k) — softmax probs of those experts
pub fn topk_softmax_batch(
    logits: &[f32],
    n_tokens: usize,
    n_experts: usize,
    top_k: usize,
    expert_ids_out: &mut [u32],
    weights_out: &mut [f32],
) {
    debug_assert_eq!(logits.len(), n_tokens * n_experts);
    debug_assert_eq!(expert_ids_out.len(), n_tokens * top_k);
    debug_assert_eq!(weights_out.len(), n_tokens * top_k);

    let mut work = vec![0.0f32; n_experts];
    for t in 0..n_tokens {
        work.copy_from_slice(&logits[t * n_experts..(t + 1) * n_experts]);
        softmax_inplace(&mut work);
        for k in 0..top_k {
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for i in 0..n_experts {
                if work[i] > best_val {
                    best_val = work[i];
                    best_idx = i;
                }
            }
            expert_ids_out[t * top_k + k] = best_idx as u32;
            weights_out[t * top_k + k] = best_val;
            work[best_idx] = f32::NEG_INFINITY;
        }
    }
}

// -------- Metal-backed paths (Phase 1+) -------------------------------

#[cfg(target_os = "macos")]
mod metal_dispatch {
    use crate::metal::{ArgLayout, CommandBatch, DecodeArena, KernelArgBuffer, MetalContext, PinnedBuffer, TokenCommandBuffer};
    use crate::{Error, Result};
    use half::f16;

    // Reduction kernels in this module are written for tg_size=256 (the
    // shader's stride>>=1 pairwise reduction requires a power of two).
    const TG_SIZE: u32 = 256;

    /// Q4_K_M-weight × fp32-vec → fp32 GEMV, dispatching the
    /// dense-path `gemm_q4_k_m_fused` kernel in `shaders/quant.metal`.
    /// Wedge 2 / H2.4 — dequant is fused inside the FMA loop in
    /// threadgroup memory; weights stay 4-bit in DRAM.
    pub fn gemv_q4_k_m(
        ctx: &MetalContext,
        w_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_gemv(ctx, "gemm_q4_k_m_fused", w_bytes, rows, cols, x, out)
    }

    /// v0.3.0 — simdgroup_matrix variant of gemv_q4_k_m.  Dispatches
    /// `gemm_q4_k_m_fused_simd`; selected via kernel-profile
    /// `gemm_q4_k_schedule = "simdgroup"`.
    pub fn gemv_q4_k_m_simd(
        ctx: &MetalContext,
        w_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd shape: x={} cols={} out={} rows={}",
                x.len(),
                cols,
                out.len(),
                rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd weight bytes: got {} expected {}",
                w_bytes.len(),
                expected_bytes
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;

        // 32 threads (1 simdgroup) per threadgroup, 8 output rows per threadgroup.
        const SIMD_TG: u32 = 32;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows_u32 + ROWS_PER_TG - 1) / ROWS_PER_TG;
        // shmem: W-tile[64] + X-tile[64] + out-tile[64] = 192 floats = 768 bytes.
        let shmem_bytes = 192u64 * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "gemm_q4_k_m_fused_simd",
            (n_tg * SIMD_TG, 1, 1),
            (SIMD_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    /// v0.4.0 — multi-row TG + simd_sum variant.  Dispatches
    /// `gemm_q4_k_m_fused_v2`; selected via kernel-profile
    /// `gemm_q4_k_schedule = "v2"`.
    pub fn gemv_q4_k_m_v2(
        ctx: &MetalContext,
        w_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_gemv_v2(ctx, "gemm_q4_k_m_fused_v2", w_bytes, rows, cols, x, out)
    }

    /// Wedge A — pinned-buffer variant of `gemv_q4_k_m_v2`. Reads Q4_K_M weights
    /// directly from `model_buf` at `w_offset` bytes, skipping the per-call
    /// `new_buffer_with_bytes` memcpy (1.6–11 MB per expert).
    pub fn gemv_q4_k_m_v2_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_gemv_v2_pinned(
            ctx,
            "gemm_q4_k_m_fused_v2",
            model_buf,
            w_offset,
            w_byte_size,
            rows,
            cols,
            x,
            out,
        )
    }

    /// TCB variant of `gemv_q4_k_m_v2_pinned`.
    /// Encodes `gemm_q4_k_m_fused_v2` against existing buffers without
    /// committing; the caller owns the command-buffer boundary.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_v2_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_fused_v2";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(144))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        let end = w_offset
            .checked_add(w_byte_size)
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }
        let x_bytes = cols * std::mem::size_of::<f32>();
        let out_bytes = rows * std::mem::size_of::<f32>();
        if x_buf.length() < x_bytes as u64 || out_buf.length() < out_bytes as u64 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb buffer sizes: x={} expected>={x_bytes} out={} expected>={out_bytes}",
                x_buf.length(),
                out_buf.length()
            )));
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(
            KERNEL,
            (n_tg * V2_TG, 1, 1),
            (V2_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_buffer(3, Some(ab.handle()), 0);
            },
        )
    }

    /// Q3_K-weight × fp32-vec → fp32 GEMV, dispatching `gemm_q3_k_fused_v2`
    /// against a pinned model buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3_k_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q3_k_gemv_pinned(
            ctx,
            "gemm_q3_k_fused_v2",
            model_buf,
            w_offset,
            w_byte_size,
            rows,
            cols,
            x,
            out,
        )
    }

    /// TCB variant of `gemv_q3_k_pinned`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3_k_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q3_k_fused_v2";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(110))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        let end = w_offset
            .checked_add(w_byte_size)
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }
        let x_bytes = cols * std::mem::size_of::<f32>();
        let out_bytes = rows * std::mem::size_of::<f32>();
        if x_buf.length() < x_bytes as u64 || out_buf.length() < out_bytes as u64 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned_tcb buffer sizes: x={} expected>={x_bytes} out={} expected>={out_bytes}",
                x_buf.length(),
                out_buf.length()
            )));
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;
        tcb.dispatch_threads(
            KERNEL,
            (n_tg * V2_TG, 1, 1),
            (V2_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            },
        )
    }

    /// Wedge K — simdmat-optimised pinned-buffer Q4_K_M GEMV. Same signature
    /// as `gemv_q4_k_m_v2_pinned`; dispatches `gemm_q4_k_m_simdmat`.
    /// Selected via `gemm_q4_k_schedule = "simdmat"`.
    ///
    /// Uses 128-thread / 4-row-per-TG geometry (vs v2's 256/8) for better
    /// parallelism on small-row expert shapes.
    pub fn gemv_q4_k_m_simdmat_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_simdmat_pinned(
            ctx,
            model_buf,
            w_offset,
            w_byte_size,
            rows,
            cols,
            x,
            out,
        )
    }

    /// Approach 1 Iter 1 — 256 threads, 8 rows/TG, 8 simdgroups.
    /// Selected via `gemm_q4_k_schedule = "v3_8r"`.
    pub fn gemv_q4_k_m_v3_8r_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_v3_8r_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// Approach 3 — 64 threads, 4 rows/simdgroup (N_R0=4), sumy trick.
    /// Selected via `gemm_q4_k_schedule = "v3_llama"`.
    pub fn gemv_q4_k_m_v3_llama_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_v3_llama_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// v1.1.0 opt-in schedule name for the faithful llama.cpp-style Q4_K port.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_llama_port_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        gemv_q4_k_m_v3_llama_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// Approach 1 Iter 2 — 128 threads, 2 rows/simdgroup (N_R0=2), 8 rows/TG.
    /// Selected via `gemm_q4_k_schedule = "v3_dual"`.
    pub fn gemv_q4_k_m_v3_dual_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_v3_dual_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// v0.3.1 — low-level batched encoder for `gemm_q4_k_m_fused_simd`.
    /// Takes pre-allocated Metal buffers; encodes into an existing CommandBatch
    /// without allocation or readback. Use this to coalesce multiple independent
    /// simd GEMVs (e.g. gate + up) into a single command buffer.
    pub(crate) fn encode_gemv_q4_k_m_simd(
        batch: &mut CommandBatch<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const SIMD_TG: u32 = 32;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows_u32 + ROWS_PER_TG - 1) / ROWS_PER_TG;
        let shmem_bytes = 192u64 * std::mem::size_of::<f32>() as u64;
        batch.dispatch_threads(
            "gemm_q4_k_m_fused_simd",
            (n_tg * SIMD_TG, 1, 1),
            (SIMD_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// v0.3.4 — low-level batched encoder for `gemv_f32_attn`.
    /// Takes pre-allocated Metal buffers; encodes into an existing CommandBatch
    /// without allocation or readback. Use this to coalesce two independent
    /// fp32 GEMVs (e.g. q_a_proj + kv_a_proj) into a single command buffer.
    pub(crate) fn encode_gemv_f32_attn_pinned(
        batch: &mut CommandBatch<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        batch.dispatch_threads(
            "gemv_f32_attn",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// v0.3.1 — slice-in / slice-out wrapper: allocates Metal buffers, routes
    /// through `ctx.dispatch_batch { encode_gemv_q4_k_m_simd }`, reads back.
    /// Replaces the standalone `ctx.dispatch_threads` path in
    /// `moe_expert_matmul_dispatch` so simd GEMVs appear in the
    /// dispatch_batch profiling bucket and can later be coalesced.
    pub fn dispatch_gemv_q4_k_m_simd_batched(
        ctx: &MetalContext,
        w_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd shape: x={} cols={} out={} rows={}",
                x.len(),
                cols,
                out.len(),
                rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd weight bytes: got {} expected {}",
                w_bytes.len(),
                expected_bytes
            )));
        }
        let w_buf = ctx.new_buffer_with_bytes(w_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_gemv_q4_k_m_simd(batch, &w_buf, rows, cols, &x_buf, &out_buf)
        })?;
        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    /// v0.3.2 — pair wrapper: allocates x once, encodes gate+up into ONE CommandBatch.
    /// Two Q4_K_M simd GEMVs (w_a, w_b) sharing the same input (x) and output
    /// dimensions coalesce into a single command-buffer commit instead of two.
    pub fn dispatch_gemv_q4_k_m_simd_pair_batched(
        ctx: &MetalContext,
        w_a_bytes: &[u8],
        w_b_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd pair requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out_a.len() != rows || out_b.len() != rows {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd pair shape: x={} cols={} out_a={} out_b={} rows={}",
                x.len(), cols, out_a.len(), out_b.len(), rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_a_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd pair w_a bytes: got {} expected {}",
                w_a_bytes.len(), expected_bytes
            )));
        }
        if w_b_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd pair w_b bytes: got {} expected {}",
                w_b_bytes.len(), expected_bytes
            )));
        }
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let w_a_buf = ctx.new_buffer_with_bytes(w_a_bytes);
        let w_b_buf = ctx.new_buffer_with_bytes(w_b_bytes);
        let out_a_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let out_b_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_gemv_q4_k_m_simd(batch, &w_a_buf, rows, cols, &x_buf, &out_a_buf)?;
            encode_gemv_q4_k_m_simd(batch, &w_b_buf, rows, cols, &x_buf, &out_b_buf)
        })?;
        let ptr_a = out_a_buf.contents() as *const f32;
        let ptr_b = out_b_buf.contents() as *const f32;
        let slice_a = unsafe { std::slice::from_raw_parts(ptr_a, rows) };
        let slice_b = unsafe { std::slice::from_raw_parts(ptr_b, rows) };
        out_a.copy_from_slice(slice_a);
        out_b.copy_from_slice(slice_b);
        Ok(())
    }

    /// v0.3.3 — fused pair+silu: encode gate, up, and silu_mul in ONE CommandBatch.
    /// `a` receives `silu(gate_out) * up_out`; intermediate gate/up buffers stay
    /// on the GPU and are never read back.
    pub fn dispatch_gemv_q4_k_m_simd_pair_silu_batched(
        ctx: &MetalContext,
        w_gate_bytes: &[u8],
        w_up_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        a: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd pair+silu requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || a.len() != rows {
            return Err(Error::Kernel(format!(
                "gemm_q4_k_m_fused_simd pair+silu shape: x={} cols={} a={} rows={}",
                x.len(), cols, a.len(), rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_gate_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "pair+silu w_gate bytes: got {} expected {}", w_gate_bytes.len(), expected_bytes
            )));
        }
        if w_up_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "pair+silu w_up bytes: got {} expected {}", w_up_bytes.len(), expected_bytes
            )));
        }
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let w_gate_buf = ctx.new_buffer_with_bytes(w_gate_bytes);
        let w_up_buf = ctx.new_buffer_with_bytes(w_up_bytes);
        // g_buf and u_buf are device-only intermediates; never read back.
        let g_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let u_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let a_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        // Single CB: gate GEMV, up GEMV, then silu_mul. Metal serializes
        // compute passes within one CB, so g_buf/u_buf are coherent when
        // silu_mul reads them.
        ctx.dispatch_batch(|batch| {
            encode_gemv_q4_k_m_simd(batch, &w_gate_buf, rows, cols, &x_buf, &g_buf)?;
            encode_gemv_q4_k_m_simd(batch, &w_up_buf,  rows, cols, &x_buf, &u_buf)?;
            encode_silu_mul(batch, &g_buf, &u_buf, &a_buf, rows)
        })?;
        let ptr = a_buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, rows) };
        a.copy_from_slice(slice);
        Ok(())
    }

    // ---- Phase 1 / Haul 1 — stubs the haul replaces with bodies ----
    //
    // Each function below is the seam the haul targets. The signature
    // and call-from-host expectations are locked: bodies arrive in
    // `_phase1_haul_manifest.md` G1.1 / G1.2 / G1.3 / G1.4. The haul
    // does NOT change these signatures; doing so would invalidate the
    // call sites in `model::deepseek_v2` and the parity tests in
    // `tests/phase1_kernel_parity.rs`.

    /// G1.1 — RMSNorm via the existing `rmsnorm` kernel in
    /// `shaders/common.metal`. Inputs and outputs are fp32 from the
    /// caller's view; the kernel works in fp16 internally.
    ///
    /// Threadgroup size 256 (kernel uses parallel reduction; must be
    /// power of two ≤ 1024).
    pub fn rmsnorm_metal(
        ctx: &MetalContext,
        x: &[f32],
        weight: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let hidden = x.len();
        if weight.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "rmsnorm_metal shape mismatch: x={} weight={} out={}",
                hidden,
                weight.len(),
                out.len()
            )));
        }

        // Host-side f32 → f16 conversion for the kernel's half I/O.
        // The test path is small (4096 elements); this is dwarfed by
        // the dispatch overhead. Real model paths will keep weights in
        // device memory and skip this round-trip.
        let x_f16: Vec<f16> = x.iter().map(|&v| f16::from_f32(v)).collect();
        let w_f16: Vec<f16> = weight.iter().map(|&v| f16::from_f32(v)).collect();

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&x_f16));
        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(&w_f16));
        let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f16>());

        let hidden_u32 = hidden as u32;
        let eps_f32 = eps;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&x_buf), 0);
            enc.set_buffer(1, Some(&w_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &hidden_u32 as *const u32 as *const _,
            );
            enc.set_bytes(
                4,
                std::mem::size_of::<f32>() as u64,
                &eps_f32 as *const f32 as *const _,
            );
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        // Read back from the shared-storage output buffer. f16 → f32
        // on the host; the parity test compares against the f32 CPU
        // reference at atol=1e-3 (fp16 quant noise floor).
        let out_ptr = out_buf.contents() as *const f16;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
        for i in 0..hidden {
            out[i] = out_slice[i].to_f32();
        }

        Ok(())
    }

    /// G1.2 — fp16 GEMV. Maps to a new `gemv_f16` kernel in
    /// `shaders/common.metal` (added during the haul). Used for the
    /// LM-head projection (vocab × hidden).
    ///
    /// Layout: `w` is row-major `(rows, cols)` fp16; `x` is fp32
    /// converted to fp16 inside the dispatch; `out` is fp32 from the
    /// kernel's fp16 result.
    pub fn gemv_f16_metal(
        ctx: &MetalContext,
        w_f16_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "gemv_f16_metal shape mismatch: x={} rows={} cols={} out={}",
                x.len(),
                rows,
                cols,
                out.len()
            )));
        }
        let expected_w = rows * cols * std::mem::size_of::<f16>();
        if w_f16_bytes.len() != expected_w {
            return Err(Error::Kernel(format!(
                "gemv_f16_metal weight bytes mismatch: got {} expected {}",
                w_f16_bytes.len(),
                expected_w
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_f16_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        // Total threads = rows * tg_size; threadgroup = tg_size →
        // exactly one threadgroup per output row.
        ctx.dispatch_threads(
            "gemv_f16",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// WB pinned variant of `gemv_f16_metal`: takes a pre-uploaded
    /// `&PinnedBuffer` for the weight matrix instead of a host byte
    /// slice. Eliminates the per-dispatch `new_buffer_with_bytes`
    /// memcpy for the LM head (~400 MB / token in DeepSeek-V2-Lite).
    ///
    /// Caller owns `w_buf` (typically held on the model). Shape
    /// constraints identical to `gemv_f16_metal`. Output buffer is
    /// allocated fresh per dispatch (small — `rows * 4` bytes).
    pub fn gemv_f16_metal_pinned(
        ctx: &MetalContext,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "gemv_f16_metal_pinned shape mismatch: x={} rows={} cols={} out={}",
                x.len(),
                rows,
                cols,
                out.len()
            )));
        }
        let expected_w = (rows * cols * std::mem::size_of::<f16>()) as u64;
        if w_buf.length() < expected_w {
            return Err(Error::Kernel(format!(
                "gemv_f16_metal_pinned weight buffer too small: got {} expected {}",
                w_buf.length(),
                expected_w
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "gemv_f16",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Profiled greedy decode primitive: LM-head GEMV followed by GPU
    /// argmax in one command buffer. Only the final token id is read
    /// back to the CPU.
    pub fn gemv_f16_argmax_metal_pinned(
        ctx: &MetalContext,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x: &[f32],
    ) -> Result<u32> {
        if x.len() != cols {
            return Err(Error::Kernel(format!(
                "gemv_f16_argmax_metal_pinned shape mismatch: x={} cols={}",
                x.len(),
                cols
            )));
        }
        if rows == 0 {
            return Err(Error::Kernel(
                "gemv_f16_argmax_metal_pinned requires rows > 0".into(),
            ));
        }
        let expected_w = (rows * cols * std::mem::size_of::<f16>()) as u64;
        if w_buf.length() < expected_w {
            return Err(Error::Kernel(format!(
                "gemv_f16_argmax_metal_pinned weight buffer too small: got {} expected {}",
                w_buf.length(),
                expected_w
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let logits_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "gemv_f16",
                (rows_u32 * TG_SIZE, 1, 1),
                (TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(w_buf), 0);
                    enc.set_buffer(1, Some(&x_buf), 0);
                    enc.set_buffer(2, Some(&logits_buf), 0);
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<u32>() as u64,
                        &rows_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<u32>() as u64,
                        &cols_u32 as *const u32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                },
            )?;
            // v0.5.7-A: parallel 256-thread argmax; needs threadgroup memory for
            // shmem_v (256 floats) and shmem_i (256 uints).
            batch.dispatch_threads("sample_argmax_f32", (256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(&logits_buf), 0);
                enc.set_buffer(1, Some(&token_buf), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, 256 * std::mem::size_of::<f32>() as u64);
                enc.set_threadgroup_memory_length(1, 256 * std::mem::size_of::<u32>() as u64);
            })?;
            Ok(())
        })?;

        let token_ptr = token_buf.contents() as *const u32;
        Ok(unsafe { *token_ptr })
    }

    /// G1.3 — fp32 GEMV for attention's `o_proj`. Maps to a new
    /// `gemv_f32_attn` kernel in `shaders/attn.metal`. The model
    /// layer dequants per-call into a scratch buffer (lazy-dequant
    /// invariant from Phase 0); this kernel reads that scratch as
    /// fp32 weights.
    pub fn gemv_f32_attn_metal(
        ctx: &MetalContext,
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_gemv_f32(ctx, "gemv_f32_attn", w, rows, cols, x, out)
    }

    /// WB pinned variant of `gemv_f32_attn_metal`: takes a pre-uploaded
    /// `&PinnedBuffer` for the weight matrix instead of a host
    /// `&[f32]`. Eliminates the per-dispatch `new_buffer_with_bytes`
    /// memcpy for the 5 attention-projection gemvs (q_a_proj,
    /// q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj — totaling
    /// ~50 MB / token in DeepSeek-V2-Lite at 27 layers).
    pub fn gemv_f32_attn_metal_pinned(
        ctx: &MetalContext,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_gemv_f32_pinned(ctx, "gemv_f32_attn", w_buf, rows, cols, x, out)
    }

    /// v0.3.4 — shared-input pair wrapper: coalesces two independent fp32 GEMVs
    /// (e.g. q_a_proj + kv_a_proj) that read the same `x` into ONE CommandBatch.
    /// Saves one CB commit per attention layer per token vs two standalone calls.
    pub fn dispatch_gemv_f32_attn_pinned_pair_batched(
        ctx: &MetalContext,
        w_a_buf: &PinnedBuffer,
        rows_a: usize,
        w_b_buf: &PinnedBuffer,
        rows_b: usize,
        cols: usize,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) -> Result<()> {
        if x.len() != cols || out_a.len() != rows_a || out_b.len() != rows_b {
            return Err(Error::Kernel(format!(
                "dispatch_gemv_f32_attn_pinned_pair shape: x={} cols={} out_a={} rows_a={} out_b={} rows_b={}",
                x.len(), cols, out_a.len(), rows_a, out_b.len(), rows_b
            )));
        }
        let expected_a = (rows_a * cols * std::mem::size_of::<f32>()) as u64;
        if w_a_buf.length() < expected_a {
            return Err(Error::Kernel(format!(
                "dispatch_gemv_f32_attn_pinned_pair w_a too small: got {} expected {}",
                w_a_buf.length(), expected_a
            )));
        }
        let expected_b = (rows_b * cols * std::mem::size_of::<f32>()) as u64;
        if w_b_buf.length() < expected_b {
            return Err(Error::Kernel(format!(
                "dispatch_gemv_f32_attn_pinned_pair w_b too small: got {} expected {}",
                w_b_buf.length(), expected_b
            )));
        }
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_a_buf = ctx.new_buffer(rows_a * std::mem::size_of::<f32>());
        let out_b_buf = ctx.new_buffer(rows_b * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_gemv_f32_attn_pinned(batch, w_a_buf, rows_a, cols, &x_buf, &out_a_buf)?;
            encode_gemv_f32_attn_pinned(batch, w_b_buf, rows_b, cols, &x_buf, &out_b_buf)
        })?;
        let ptr_a = out_a_buf.contents() as *const f32;
        let ptr_b = out_b_buf.contents() as *const f32;
        let slice_a = unsafe { std::slice::from_raw_parts(ptr_a, rows_a) };
        let slice_b = unsafe { std::slice::from_raw_parts(ptr_b, rows_b) };
        out_a.copy_from_slice(slice_a);
        out_b.copy_from_slice(slice_b);
        Ok(())
    }

    /// G1.4 — fp32 GEMV for the MoE gate-logit projection
    /// (`ffn_gate_inp`). Maps to a new `gemv_f32_moe` kernel in
    /// `shaders/moe.metal`. Tiny (n_routed × hidden = 64 × 2048) but
    /// proves MoE-shaped weight access.
    pub fn gemv_f32_moe_metal(
        ctx: &MetalContext,
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_gemv_f32(ctx, "gemv_f32_moe", w, rows, cols, x, out)
    }

    /// Phase 2 — no-pack batched DeepSeek MoE block. The weight buffer is
    /// the full GGUF mmap (or a test stand-in), and tensor byte offsets
    /// select the fused routed/shared expert tensors in-place. Route IDs
    /// choose experts inside the fused routed tensors, eliminating the
    /// per-token host packing of selected expert bytes.
    ///
    /// The whole MoE subgraph is encoded into one command buffer, so the
    /// routed/shared GEMVs, activations, and accumulation pay one
    /// `commit + wait` instead of one per kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_block_batched_indexed_metal(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        routed_gate_offset: usize,
        routed_up_offset: usize,
        routed_down_offset: usize,
        n_routed_experts: usize,
        route_ids: &[u32],
        route_weights: &[f32],
        shared_gate_offset: Option<usize>,
        shared_up_offset: Option<usize>,
        shared_down_offset: Option<usize>,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        q4k_schedule: &str,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let routes = route_ids.len();
        if routes == 0 {
            return Err(Error::Kernel(
                "moe_block_batched_indexed_metal: no routes".into(),
            ));
        }
        if route_weights.len() != routes {
            return Err(Error::Kernel(format!(
                "moe_block_batched_indexed_metal: {} route ids but {} weights",
                routes,
                route_weights.len()
            )));
        }
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "moe_block_batched_indexed_metal shape: x={} hidden={} out={}",
                x.len(),
                hidden,
                out.len()
            )));
        }
        for &eid in route_ids {
            if eid as usize >= n_routed_experts {
                return Err(Error::Kernel(format!(
                    "moe_block_batched_indexed_metal: route expert {eid} >= {n_routed_experts}"
                )));
            }
        }

        validate_indexed_quant(
            "moe_block_batched_indexed routed_gate_q4",
            model_buf,
            routed_gate_offset,
            n_routed_experts,
            routed_mid,
            hidden,
            256,
            144,
        )?;
        validate_indexed_quant(
            "moe_block_batched_indexed routed_up_q4",
            model_buf,
            routed_up_offset,
            n_routed_experts,
            routed_mid,
            hidden,
            256,
            144,
        )?;
        validate_indexed_quant(
            "moe_block_batched_indexed routed_down_q8",
            model_buf,
            routed_down_offset,
            n_routed_experts,
            hidden,
            routed_mid,
            32,
            34,
        )?;

        let has_shared = shared_gate_offset.is_some()
            || shared_up_offset.is_some()
            || shared_down_offset.is_some();
        if has_shared
            && !(shared_gate_offset.is_some()
                && shared_up_offset.is_some()
                && shared_down_offset.is_some())
        {
            return Err(Error::Kernel(
                "moe_block_batched_indexed_metal: shared offsets must be all Some or all None"
                    .into(),
            ));
        }
        if has_shared {
            validate_indexed_quant(
                "moe_block_batched_indexed shared_gate_q4",
                model_buf,
                shared_gate_offset.unwrap(),
                1,
                shared_mid,
                hidden,
                256,
                144,
            )?;
            validate_indexed_quant(
                "moe_block_batched_indexed shared_up_q4",
                model_buf,
                shared_up_offset.unwrap(),
                1,
                shared_mid,
                hidden,
                256,
                144,
            )?;
            validate_indexed_quant(
                "moe_block_batched_indexed shared_down_q6",
                model_buf,
                shared_down_offset.unwrap(),
                1,
                hidden,
                shared_mid,
                256,
                210,
            )?;
        }

        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let route_weights_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let shared_route_ids = [0u32];
        let shared_route_ids_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&shared_route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));

        let routed_gate_out = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_up_out = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_act = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_out = ctx.new_buffer(routes * hidden * std::mem::size_of::<f32>());
        let final_out = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let shared_gate_out = ctx.new_buffer(shared_mid.max(1) * std::mem::size_of::<f32>());
        let shared_up_out = ctx.new_buffer(shared_mid.max(1) * std::mem::size_of::<f32>());
        let shared_act = ctx.new_buffer(shared_mid.max(1) * std::mem::size_of::<f32>());
        let shared_out = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let q4k_indexed_kernel = match q4k_schedule {
            "v2" | "llama_port" | "per_shape" => "moe_batched_gemm_q4_indexed_v2",
            _ => "moe_batched_gemm_q4_indexed",
        };

        ctx.dispatch_batch(|batch| {
            encode_batched_gemv_indexed(
                batch,
                q4k_indexed_kernel,
                model_buf,
                &route_ids_buf,
                &x_buf,
                &routed_gate_out,
                routed_gate_offset,
                routes,
                routed_mid,
                hidden,
            )?;
            encode_batched_gemv_indexed(
                batch,
                q4k_indexed_kernel,
                model_buf,
                &route_ids_buf,
                &x_buf,
                &routed_up_out,
                routed_up_offset,
                routes,
                routed_mid,
                hidden,
            )?;
            encode_silu_mul(
                batch,
                &routed_gate_out,
                &routed_up_out,
                &routed_act,
                routes * routed_mid,
            )?;
            encode_batched_gemv_indexed(
                batch,
                "moe_batched_gemm_q8_0_indexed",
                model_buf,
                &route_ids_buf,
                &routed_act,
                &routed_out,
                routed_down_offset,
                routes,
                hidden,
                routed_mid,
            )?;

            if let (Some(gate_off), Some(up_off), Some(down_off)) =
                (shared_gate_offset, shared_up_offset, shared_down_offset)
            {
                encode_batched_gemv_indexed(
                    batch,
                    q4k_indexed_kernel,
                    model_buf,
                    &shared_route_ids_buf,
                    &x_buf,
                    &shared_gate_out,
                    gate_off,
                    1,
                    shared_mid,
                    hidden,
                )?;
                encode_batched_gemv_indexed(
                    batch,
                    q4k_indexed_kernel,
                    model_buf,
                    &shared_route_ids_buf,
                    &x_buf,
                    &shared_up_out,
                    up_off,
                    1,
                    shared_mid,
                    hidden,
                )?;
                encode_silu_mul(
                    batch,
                    &shared_gate_out,
                    &shared_up_out,
                    &shared_act,
                    shared_mid,
                )?;
                encode_batched_gemv_indexed(
                    batch,
                    "moe_batched_gemm_q6_k_indexed",
                    model_buf,
                    &shared_route_ids_buf,
                    &shared_act,
                    &shared_out,
                    down_off,
                    1,
                    hidden,
                    shared_mid,
                )?;
            }

            encode_route_accumulate(
                batch,
                &routed_out,
                &route_weights_buf,
                &shared_out,
                &final_out,
                hidden,
                routes,
                has_shared,
            )
        })?;

        copy_f32_buffer(&final_out, out);
        Ok(())
    }

    /// Wedge 1 — Metal MLA decode kernel.
    ///
    /// Replaces the CPU `mla_decode_step` for DeepSeek-V2-family models.
    /// Operates on the compressed KV cache (c_kv, k_pe) rather than the
    /// expanded K/V matrices. kv_b_proj is pinned (loaded once at model
    /// load time via `layer.pinned.kv_b_proj`).
    ///
    /// Buffer layout matches `mla_decode_kernel` in `shaders/attn.metal`.
    /// Dispatch: one workgroup per attention head (grid = n_heads × TG_SIZE).
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decode_metal(
        ctx: &MetalContext,
        q: &[f32],
        c_kv: &[f32],
        k_pe: &[f32],
        kv_b_proj: &PinnedBuffer,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        seq_len: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        if q.len() != n_heads * q_head_dim {
            return Err(Error::Kernel(format!(
                "mla_decode_metal: q.len={} expected {}",
                q.len(),
                n_heads * q_head_dim
            )));
        }
        if c_kv.len() != seq_len * kv_lora_rank {
            return Err(Error::Kernel(format!(
                "mla_decode_metal: c_kv.len={} expected {}",
                c_kv.len(),
                seq_len * kv_lora_rank
            )));
        }
        if k_pe.len() != seq_len * qk_rope_head_dim {
            return Err(Error::Kernel(format!(
                "mla_decode_metal: k_pe.len={} expected {}",
                k_pe.len(),
                seq_len * qk_rope_head_dim
            )));
        }
        let expected_kv_b =
            (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>())
                as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!(
                "mla_decode_metal: kv_b_proj buffer too small: got {} expected {}",
                kv_b_proj.length(),
                expected_kv_b
            )));
        }
        if out.len() != n_heads * v_head_dim {
            return Err(Error::Kernel(format!(
                "mla_decode_metal: out.len={} expected {}",
                out.len(),
                n_heads * v_head_dim
            )));
        }
        if seq_len == 0 {
            return Err(Error::Kernel(
                "mla_decode_metal: seq_len must be >= 1".into(),
            ));
        }

        let q_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(q));
        let c_kv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(c_kv));
        let k_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(k_pe));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());

        let n_heads_u32 = n_heads as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let v_head_u32 = v_head_dim as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let seq_len_u32 = seq_len as u32;

        // Threadgroup slots:
        //   0 — q_nope_proj: kv_lora_rank floats
        //   1 — scores:      seq_len floats
        //   2 — c_kv_wt:     kv_lora_rank floats
        let q_nope_proj_bytes = (kv_lora_rank as u64) * std::mem::size_of::<f32>() as u64;
        let scores_bytes = (seq_len as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "mla_decode_kernel",
            (n_heads_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&q_buf), 0);
                enc.set_buffer(1, Some(&c_kv_buf), 0);
                enc.set_buffer(2, Some(&k_pe_buf), 0);
                enc.set_buffer(3, Some(kv_b_proj), 0);
                enc.set_buffer(4, Some(&out_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &n_heads_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &qk_nope_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    7,
                    std::mem::size_of::<u32>() as u64,
                    &qk_rope_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    8,
                    std::mem::size_of::<u32>() as u64,
                    &v_head_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    9,
                    std::mem::size_of::<u32>() as u64,
                    &kv_lora_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    10,
                    std::mem::size_of::<u32>() as u64,
                    &seq_len_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    11,
                    std::mem::size_of::<f32>() as u64,
                    &scale as *const f32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                enc.set_threadgroup_memory_length(1, scores_bytes);
                enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
            },
        )?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Wedge L — flash attention decode using online softmax (MLA-aware).
    ///
    /// Replaces phases 1-3 of `mla_decode_metal` with a tiled flash loop that
    /// never materialises the full seq_len scores array. TG shmem drops from
    /// O(seq_len) to O(FLASH_TG=128) floats, enabling higher GPU occupancy.
    ///
    /// Interface is identical to `mla_decode_metal`; caller selects via
    /// `profile.selected.attn_block_schedule == "flash"`.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn_decode_metal(
        ctx: &MetalContext,
        q: &[f32],
        c_kv: &[f32],
        k_pe: &[f32],
        kv_b_proj: &PinnedBuffer,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        seq_len: usize,
        scale: f32,
        out: &mut [f32],
    ) -> Result<()> {
        const FLASH_TG: u32 = 128;

        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        if q.len() != n_heads * q_head_dim {
            return Err(Error::Kernel(format!(
                "flash_attn_decode_metal: q.len={} expected {}",
                q.len(),
                n_heads * q_head_dim
            )));
        }
        if c_kv.len() != seq_len * kv_lora_rank {
            return Err(Error::Kernel(format!(
                "flash_attn_decode_metal: c_kv.len={} expected {}",
                c_kv.len(),
                seq_len * kv_lora_rank
            )));
        }
        if k_pe.len() != seq_len * qk_rope_head_dim {
            return Err(Error::Kernel(format!(
                "flash_attn_decode_metal: k_pe.len={} expected {}",
                k_pe.len(),
                seq_len * qk_rope_head_dim
            )));
        }
        let expected_kv_b =
            (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>())
                as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!(
                "flash_attn_decode_metal: kv_b_proj buffer too small: got {} expected {}",
                kv_b_proj.length(),
                expected_kv_b
            )));
        }
        if out.len() != n_heads * v_head_dim {
            return Err(Error::Kernel(format!(
                "flash_attn_decode_metal: out.len={} expected {}",
                out.len(),
                n_heads * v_head_dim
            )));
        }
        if seq_len == 0 {
            return Err(Error::Kernel(
                "flash_attn_decode_metal: seq_len must be >= 1".into(),
            ));
        }

        let q_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(q));
        let c_kv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(c_kv));
        let k_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(k_pe));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());

        let n_heads_u32 = n_heads as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let v_head_u32 = v_head_dim as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let seq_len_u32 = seq_len as u32;

        let f32_size = std::mem::size_of::<f32>() as u64;
        // slot 0: q_nope_proj[kv_lora_rank]
        let q_nope_proj_bytes = kv_lora_rank as u64 * f32_size;
        // slot 1: acc[kv_lora_rank]
        let acc_bytes = kv_lora_rank as u64 * f32_size;
        // slot 2: scores_tile[FLASH_TG]
        let scores_tile_bytes = FLASH_TG as u64 * f32_size;
        // slot 3: state[8] = {m, l, corr, m_tile, simd0..3_max}
        let state_bytes = 8u64 * f32_size;

        ctx.dispatch_threads(
            "flash_attn_decode_kernel",
            (n_heads_u32 * FLASH_TG, 1, 1),
            (FLASH_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&q_buf), 0);
                enc.set_buffer(1, Some(&c_kv_buf), 0);
                enc.set_buffer(2, Some(&k_pe_buf), 0);
                enc.set_buffer(3, Some(kv_b_proj), 0);
                enc.set_buffer(4, Some(&out_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &n_heads_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &qk_nope_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    7,
                    std::mem::size_of::<u32>() as u64,
                    &qk_rope_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    8,
                    std::mem::size_of::<u32>() as u64,
                    &v_head_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    9,
                    std::mem::size_of::<u32>() as u64,
                    &kv_lora_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    10,
                    std::mem::size_of::<u32>() as u64,
                    &seq_len_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    11,
                    std::mem::size_of::<f32>() as u64,
                    &scale as *const f32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                enc.set_threadgroup_memory_length(1, acc_bytes);
                enc.set_threadgroup_memory_length(2, scores_tile_bytes);
                enc.set_threadgroup_memory_length(3, state_bytes);
            },
        )?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Wedge 3 — Layer-CB: batch mla_decode_kernel + gemv_f32_attn (o_proj)
    /// into one command buffer. Saves one commit+wait per attention layer
    /// (27 fewer roundtrips per token on DeepSeek-V2-Lite).
    ///
    /// The intermediate `attn_out` buffer stays in GPU memory between the two
    /// kernels; Metal guarantees sequential execution within a command buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decode_and_o_proj_metal(
        ctx: &MetalContext,
        q: &[f32],
        c_kv: &[f32],
        k_pe: &[f32],
        kv_b_proj: &PinnedBuffer,
        o_proj: &PinnedBuffer,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        seq_len: usize,
        scale: f32,
        hidden: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        let attn_out_len = n_heads * v_head_dim;
        let o_proj_cols = attn_out_len;
        if q.len() != n_heads * q_head_dim {
            return Err(Error::Kernel(format!(
                "mla_decode_and_o_proj_metal: q.len={} expected {}",
                q.len(),
                n_heads * q_head_dim
            )));
        }
        if seq_len == 0 {
            return Err(Error::Kernel(
                "mla_decode_and_o_proj_metal: seq_len must be >= 1".into(),
            ));
        }
        if out.len() != hidden {
            return Err(Error::Kernel(format!(
                "mla_decode_and_o_proj_metal: out.len={} expected hidden={}",
                out.len(),
                hidden
            )));
        }
        let expected_kv_b =
            (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>())
                as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!(
                "mla_decode_and_o_proj_metal: kv_b_proj too small: {} < {}",
                kv_b_proj.length(),
                expected_kv_b
            )));
        }
        let expected_o_proj = (hidden * o_proj_cols * std::mem::size_of::<f32>()) as u64;
        if o_proj.length() < expected_o_proj {
            return Err(Error::Kernel(format!(
                "mla_decode_and_o_proj_metal: o_proj too small: {} < {}",
                o_proj.length(),
                expected_o_proj
            )));
        }

        let q_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(q));
        let c_kv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(c_kv));
        let k_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(k_pe));
        // Intermediate attn_out stays in GPU memory — shared between mla_decode and o_proj.
        let attn_out_buf = ctx.new_buffer(attn_out_len * std::mem::size_of::<f32>());
        let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let n_heads_u32 = n_heads as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let v_head_u32 = v_head_dim as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let seq_len_u32 = seq_len as u32;
        let hidden_u32 = hidden as u32;
        let o_proj_cols_u32 = o_proj_cols as u32;

        let q_nope_proj_bytes = (kv_lora_rank as u64) * std::mem::size_of::<f32>() as u64;
        let scores_bytes = (seq_len as u64) * std::mem::size_of::<f32>() as u64;
        let shmem_bytes = TG_SIZE as u64 * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_batch(|batch| {
            // Kernel 1: mla_decode_kernel → writes attn_out_buf.
            batch.dispatch_threads(
                "mla_decode_kernel",
                (n_heads_u32 * TG_SIZE, 1, 1),
                (TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&q_buf), 0);
                    enc.set_buffer(1, Some(&c_kv_buf), 0);
                    enc.set_buffer(2, Some(&k_pe_buf), 0);
                    enc.set_buffer(3, Some(kv_b_proj), 0);
                    enc.set_buffer(4, Some(&attn_out_buf), 0);
                    enc.set_bytes(
                        5,
                        std::mem::size_of::<u32>() as u64,
                        &n_heads_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        6,
                        std::mem::size_of::<u32>() as u64,
                        &qk_nope_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        7,
                        std::mem::size_of::<u32>() as u64,
                        &qk_rope_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        8,
                        std::mem::size_of::<u32>() as u64,
                        &v_head_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        9,
                        std::mem::size_of::<u32>() as u64,
                        &kv_lora_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        10,
                        std::mem::size_of::<u32>() as u64,
                        &seq_len_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        11,
                        std::mem::size_of::<f32>() as u64,
                        &scale as *const f32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                    enc.set_threadgroup_memory_length(1, scores_bytes);
                    enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
                },
            )?;

            // Kernel 2: gemv_f32_attn (o_proj) — reads attn_out_buf, writes out_buf.
            // Metal serializes these within the command buffer.
            batch.dispatch_threads(
                "gemv_f32_attn",
                (hidden_u32 * TG_SIZE, 1, 1),
                (TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(o_proj), 0);
                    enc.set_buffer(1, Some(&attn_out_buf), 0);
                    enc.set_buffer(2, Some(&out_buf), 0);
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<u32>() as u64,
                        &hidden_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<u32>() as u64,
                        &o_proj_cols_u32 as *const u32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                },
            )?;

            Ok(())
        })?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Wedge 4 — Decode-Arena variant of `mla_decode_and_o_proj_metal`.
    /// Uses pre-allocated arena buffers for attn_out and final out, and
    /// writes q/c_kv/k_pe into arena buffers via direct CPU memcpy.
    /// Eliminates all 5 per-dispatch Metal buffer allocations.
    ///
    /// The arena must already hold c_kv/k_pe data up to `seq_len` entries
    /// (written by the caller via `arena.append_c_kv` / `append_k_pe`),
    /// and q written via `arena.write_q`.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decode_and_o_proj_arena_metal(
        ctx: &MetalContext,
        arena: &DecodeArena,
        kv_b_proj: &PinnedBuffer,
        o_proj: &PinnedBuffer,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        seq_len: usize,
        scale: f32,
        hidden: usize,
        out: &mut [f32],
    ) -> Result<()> {
        if seq_len == 0 {
            return Err(Error::Kernel(
                "mla_decode_and_o_proj_arena_metal: seq_len must be >= 1".into(),
            ));
        }
        if out.len() != hidden {
            return Err(Error::Kernel(format!(
                "mla_decode_and_o_proj_arena_metal: out.len={} != hidden={}",
                out.len(),
                hidden
            )));
        }

        let n_heads_u32 = n_heads as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let v_head_u32 = v_head_dim as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let seq_len_u32 = seq_len as u32;
        let hidden_u32 = hidden as u32;
        let o_proj_cols_u32 = (n_heads * v_head_dim) as u32;

        let q_nope_proj_bytes = (kv_lora_rank as u64) * std::mem::size_of::<f32>() as u64;
        let scores_bytes = (seq_len as u64) * std::mem::size_of::<f32>() as u64;
        let shmem_bytes = TG_SIZE as u64 * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "mla_decode_kernel",
                (n_heads_u32 * TG_SIZE, 1, 1),
                (TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&arena.q), 0);
                    enc.set_buffer(1, Some(&arena.c_kv), 0);
                    enc.set_buffer(2, Some(&arena.k_pe), 0);
                    enc.set_buffer(3, Some(kv_b_proj), 0);
                    enc.set_buffer(4, Some(&arena.attn_out), 0);
                    enc.set_bytes(
                        5,
                        std::mem::size_of::<u32>() as u64,
                        &n_heads_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        6,
                        std::mem::size_of::<u32>() as u64,
                        &qk_nope_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        7,
                        std::mem::size_of::<u32>() as u64,
                        &qk_rope_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        8,
                        std::mem::size_of::<u32>() as u64,
                        &v_head_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        9,
                        std::mem::size_of::<u32>() as u64,
                        &kv_lora_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        10,
                        std::mem::size_of::<u32>() as u64,
                        &seq_len_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        11,
                        std::mem::size_of::<f32>() as u64,
                        &scale as *const f32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                    enc.set_threadgroup_memory_length(1, scores_bytes);
                    enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
                },
            )?;

            batch.dispatch_threads(
                "gemv_f32_attn",
                (hidden_u32 * TG_SIZE, 1, 1),
                (TG_SIZE, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(o_proj), 0);
                    enc.set_buffer(1, Some(&arena.attn_out), 0);
                    enc.set_buffer(2, Some(&arena.out), 0);
                    enc.set_bytes(
                        3,
                        std::mem::size_of::<u32>() as u64,
                        &hidden_u32 as *const u32 as *const _,
                    );
                    enc.set_bytes(
                        4,
                        std::mem::size_of::<u32>() as u64,
                        &o_proj_cols_u32 as *const u32 as *const _,
                    );
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                },
            )?;

            Ok(())
        })?;

        arena.read_out(out);
        Ok(())
    }

    fn validate_indexed_quant(
        name: &str,
        model_buf: &PinnedBuffer,
        base_offset: usize,
        matrices: usize,
        rows: usize,
        cols: usize,
        block_elems: usize,
        block_bytes: usize,
    ) -> Result<()> {
        if matrices == 0 {
            return Err(Error::Kernel(format!("{name}: matrices must be > 0")));
        }
        if cols % block_elems != 0 {
            return Err(Error::Kernel(format!(
                "{name}: cols must be multiple of {block_elems}; got {cols}"
            )));
        }
        let expected = matrices
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols / block_elems))
            .and_then(|v| v.checked_mul(block_bytes))
            .ok_or_else(|| Error::Kernel(format!("{name}: byte-size overflow")))?;
        let end = base_offset
            .checked_add(expected)
            .ok_or_else(|| Error::Kernel(format!("{name}: byte-range overflow")))?;
        if end as u64 > model_buf.length() {
            return Err(Error::Kernel(format!(
                "{name}: byte range [{base_offset}, {end}) exceeds model buffer {}",
                model_buf.length()
            )));
        }
        Ok(())
    }

    /// Parity-test helper: dispatch `moe_batched_gemm_q4_indexed` (scalar)
    /// or `moe_batched_gemm_q4_indexed_v2` directly against a byte slice
    /// containing the full fused expert tensor.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_batched_gemm_q4_indexed_v2t_raw(
        ctx: &MetalContext,
        w_all_bytes: &[u8],
        base_offset: usize,
        route_ids: &[u32],
        x: &[f32],
        routes: usize,
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_batched_gemv_indexed(
                batch,
                "moe_batched_gemm_q4_indexed_v2t",
                &model_buf,
                &route_ids_buf,
                &x_buf,
                &out_buf,
                base_offset,
                routes,
                rows,
                cols,
            )
        })?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_batched_gemm_q4_indexed_v2t_gu_raw(
        ctx: &MetalContext,
        w_all_bytes: &[u8],
        gate_offset: usize,
        up_offset: usize,
        route_ids: &[u32],
        x: &[f32],
        routes: usize,
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let model_buf    = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf        = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf      = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64   = up_offset   as u64;
        let routes_u32  = routes as u32;
        let rows_u32    = rows   as u32;
        let cols_u32    = cols   as u32;
        let tg_size     = TG_SIZE as u32;
        let n_tg_x      = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "moe_batched_gemm_q4_indexed_v2t_gu",
                (n_tg_x * tg_size, routes_u32, 1),
                (tg_size, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&model_buf),    0);
                    enc.set_buffer(1, Some(&route_ids_buf), 0);
                    enc.set_buffer(2, Some(&x_buf),         0);
                    enc.set_buffer(3, Some(&out_buf),       0);
                    enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
                    enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64   as *const u64 as *const _);
                    enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &routes_u32 as *const u32 as *const _);
                    enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &rows_u32   as *const u32 as *const _);
                    enc.set_bytes(8, std::mem::size_of::<u32>() as u64, &cols_u32   as *const u32 as *const _);
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                },
            )
        })?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Raw dispatch of `moe_batched_gemm_q4_indexed_v2t_gu_v2` for parity tests.
    /// Output is silu(gate) * up, same layout as v2t_gu_raw.
    pub fn moe_batched_gemm_q4_indexed_v2t_gu_v2_raw(
        ctx: &MetalContext,
        w_all_bytes: &[u8],
        gate_offset: usize,
        up_offset: usize,
        route_ids: &[u32],
        x: &[f32],
        routes: usize,
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let model_buf     = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf         = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf       = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64   = up_offset   as u64;
        let routes_u32  = routes as u32;
        let rows_u32    = rows   as u32;
        let cols_u32    = cols   as u32;
        let tg_size     = TG_SIZE as u32;
        let n_tg_x      = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads(
                "moe_batched_gemm_q4_indexed_v2t_gu_v2",
                (n_tg_x * tg_size, routes_u32, 1),
                (tg_size, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&model_buf),     0);
                    enc.set_buffer(1, Some(&route_ids_buf), 0);
                    enc.set_buffer(2, Some(&x_buf),          0);
                    enc.set_buffer(3, Some(&out_buf),        0);
                    enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
                    enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64   as *const u64 as *const _);
                    enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &routes_u32 as *const u32 as *const _);
                    enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &rows_u32   as *const u32 as *const _);
                    enc.set_bytes(8, std::mem::size_of::<u32>() as u64, &cols_u32   as *const u32 as *const _);
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                },
            )
        })?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    pub fn moe_batched_gemm_q4_indexed_v2s_raw(
        ctx: &MetalContext,
        w_all_bytes: &[u8],
        base_offset: usize,
        route_ids: &[u32],
        x: &[f32],
        routes: usize,
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_batched_gemv_indexed(
                batch,
                "moe_batched_gemm_q4_indexed_v2s",
                &model_buf,
                &route_ids_buf,
                &x_buf,
                &out_buf,
                base_offset,
                routes,
                rows,
                cols,
            )
        })?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    pub fn moe_batched_gemm_q4_indexed_raw(
        ctx: &MetalContext,
        use_v2: bool,
        w_all_bytes: &[u8],
        base_offset: usize,
        route_ids: &[u32],
        x: &[f32],
        routes: usize,
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let kernel_name = if use_v2 {
            "moe_batched_gemm_q4_indexed_v2"
        } else {
            "moe_batched_gemm_q4_indexed"
        };
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_batched_gemv_indexed(
                batch,
                kernel_name,
                &model_buf,
                &route_ids_buf,
                &x_buf,
                &out_buf,
                base_offset,
                routes,
                rows,
                cols,
            )
        })?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_batched_gemm_q8_0_indexed_v2t_raw(
        ctx: &MetalContext,
        w_all_bytes: &[u8],
        base_offset: usize,
        route_ids: &[u32],
        x: &[f32],
        routes: usize,
        rows: usize,
        cols: usize,
        out: &mut [f32],
    ) -> Result<()> {
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| {
            encode_batched_gemv_indexed(
                batch,
                "moe_batched_gemm_q8_0_indexed_v2t",
                &model_buf,
                &route_ids_buf,
                &x_buf,
                &out_buf,
                base_offset,
                routes,
                rows,
                cols,
            )
        })?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_batched_gemv_indexed(
        batch: &mut CommandBatch<'_>,
        kernel_name: &str,
        model_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        base_offset: usize,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let base_offset_u64 = base_offset as u64;
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let is_v2t = kernel_name.ends_with("_v2t");
        let is_v2_family = kernel_name.ends_with("_v2") || kernel_name.ends_with("_v2s") || is_v2t;
        let n_tg_x = if is_v2_family { (rows_u32 + 7) / 8 } else { rows_u32 };
        let shmem_bytes = if is_v2t {
            (cols as u64) * std::mem::size_of::<f32>() as u64
        } else if is_v2_family {
            0u64
        } else {
            (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64
        };

        batch.dispatch_threads(
            kernel_name,
            (n_tg_x * tg_size, routes_u32, 1),
            (tg_size, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), 0);
                enc.set_buffer(1, Some(route_ids_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                enc.set_buffer(3, Some(out_buf), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u64>() as u64,
                    &base_offset_u64 as *const u64 as *const _,
                );
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &routes_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    7,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                if !is_v2_family || is_v2t {
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                }
            },
        )
    }

    fn encode_silu_mul(
        batch: &mut CommandBatch<'_>,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        let n_u32 = n as u32;
        batch.dispatch_threads(
            "moe_batched_silu_mul",
            (n_u32, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(gate_buf), 0);
                enc.set_buffer(1, Some(up_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &n_u32 as *const u32 as *const _,
                );
            },
        )
    }

    fn encode_route_accumulate(
        batch: &mut CommandBatch<'_>,
        routed_out: &PinnedBuffer,
        weights: &PinnedBuffer,
        shared_out: &PinnedBuffer,
        out: &PinnedBuffer,
        hidden: usize,
        routes: usize,
        has_shared: bool,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let routes_u32 = routes as u32;
        let has_shared_u32 = u32::from(has_shared);
        batch.dispatch_threads(
            "moe_route_accumulate",
            (hidden_u32, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(routed_out), 0);
                enc.set_buffer(1, Some(weights), 0);
                enc.set_buffer(2, Some(shared_out), 0);
                enc.set_buffer(3, Some(out), 0);
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &routes_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &has_shared_u32 as *const u32 as *const _,
                );
            },
        )
    }

    fn copy_f32_buffer(buf: &PinnedBuffer, out: &mut [f32]) {
        let ptr = buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, out.len()) };
        out.copy_from_slice(slice);
    }

    // Shared dispatch for the two Q4_K_M-fused GEMV kernels (H2.2 in
    // moe.metal, H2.4 in quant.metal). Same kernel body in both files;
    // only the function name differs because the manifest split puts
    // them in different shader modules. tg_size hardcoded to 256
    // (matches the Q4_K_M super-block size — see kernel comments).
    fn dispatch_q4_k_m_gemv(
        ctx: &MetalContext,
        kernel_name: &str,
        w_q4_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{kernel_name} requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{kernel_name} shape: x={} cols={} out={} rows={}",
                x.len(),
                cols,
                out.len(),
                rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_q4_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "{kernel_name} weight bytes: got {} expected {}",
                w_q4_bytes.len(),
                expected_bytes
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            kernel_name,
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // v0.4.0 — v2 dispatch: 256-thread TG, 8 rows per TG (8 simdgroups),
    // simd_sum reduction.  No threadgroup memory needed.
    fn dispatch_q4_k_m_gemv_v2(
        ctx: &MetalContext,
        kernel_name: &str,
        w_q4_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{kernel_name} requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{kernel_name} shape: x={} cols={} out={} rows={}",
                x.len(),
                cols,
                out.len(),
                rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_q4_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "{kernel_name} weight bytes: got {} expected {}",
                w_q4_bytes.len(),
                expected_bytes
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;

        ctx.dispatch_threads(
            kernel_name,
            (n_tg * V2_TG, 1, 1),
            (V2_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                // NO set_threadgroup_memory_length — kernel uses none.
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // Wedge A — pinned-buffer variant of dispatch_q4_k_m_gemv_v2. Uses set_buffer
    // offset instead of new_buffer_with_bytes, eliminating the per-call
    // weight memcpy (1.6–11 MB per expert × 236 calls/token).
    fn dispatch_q4_k_m_gemv_v2_pinned(
        ctx: &MetalContext,
        kernel_name: &str,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned shape: x={} cols={} out={} rows={}",
                x.len(),
                cols,
                out.len(),
                rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;

        ctx.dispatch_threads(
            kernel_name,
            (n_tg * V2_TG, 1, 1),
            (V2_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                // NO set_threadgroup_memory_length — kernel uses none.
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    fn dispatch_q3_k_gemv_pinned(
        ctx: &MetalContext,
        kernel_name: &str,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned shape: x={} cols={} out={} rows={}",
                x.len(),
                cols,
                out.len(),
                rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 110;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;

        ctx.dispatch_threads(
            kernel_name,
            (n_tg * V2_TG, 1, 1),
            (V2_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // Wedge K dispatcher — gemm_q4_k_m_simdmat geometry: 128 threads per TG
    // (4 simdgroups × 32), 4 rows per TG, grid=(ceil(rows/4)*128, 1, 1).
    fn dispatch_q4_k_m_simdmat_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_simdmat";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned shape: x={} cols={} out={} rows={}",
                x.len(), cols, out.len(), rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const SM_TG: u32 = 128;  // 4 simdgroups × 32 threads
        const SM_ROWS: u32 = 4;  // 1 simdgroup per row, 4 rows per TG
        let n_tg = (rows_u32 + SM_ROWS - 1) / SM_ROWS;

        ctx.dispatch_threads(
            KERNEL,
            (n_tg * SM_TG, 1, 1),
            (SM_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // Wedge K Approach 1 Iter 1 — v3_8r: 256 threads per TG (8 simdgroups),
    // 8 rows per TG, grid=(ceil(rows/8)*256, 1, 1).
    fn dispatch_q4_k_m_v3_8r_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_8r";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned shape: x={} cols={} out={} rows={}",
                x.len(), cols, out.len(), rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = (rows_u32 + V3_ROWS - 1) / V3_ROWS;

        ctx.dispatch_threads(
            KERNEL,
            (n_tg * V3_TG, 1, 1),
            (V3_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // Wedge K Approach 1 Iter 2 — v3_dual: 128 threads per TG (4 simdgroups),
    // 2 rows per simdgroup (N_R0=2), 8 rows per TG.
    // grid=(ceil(rows/8)*128, 1, 1). Amortizes activation load over 2 rows.
    fn dispatch_q4_k_m_v3_dual_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_dual";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned shape: x={} cols={} out={} rows={}",
                x.len(), cols, out.len(), rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const DUAL_TG: u32 = 128;
        const DUAL_ROWS: u32 = 8;
        let n_tg = (rows_u32 + DUAL_ROWS - 1) / DUAL_ROWS;

        ctx.dispatch_threads(
            KERNEL,
            (n_tg * DUAL_TG, 1, 1),
            (DUAL_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // Approach 3 — v3_llama: 64 threads per TG (2 simdgroups), 4 rows per
    // simdgroup (N_R0=4), sumy trick for min correction.
    // grid=(ceil(rows/8)*64, 1, 1). Faithful llama.cpp port.
    fn dispatch_q4_k_m_v3_llama_pinned(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_llama";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}"
            )));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned shape: x={} cols={} out={} rows={}",
                x.len(), cols, out.len(), rows
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}"
            )));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!(
                "{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}",
                model_buf.length()
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const LLAMA_TG: u32 = 64;   // 2 simdgroups × 32 threads
        const LLAMA_ROWS: u32 = 8;  // 2 simdgroups × 4 rows each
        let n_tg = (rows_u32 + LLAMA_ROWS - 1) / LLAMA_ROWS;

        ctx.dispatch_threads(
            KERNEL,
            (n_tg * LLAMA_TG, 1, 1),
            (LLAMA_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), w_offset as u64);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // WB shared pinned dispatch for the f32 GEMV kernels. Same kernel
    // signature as the byte-slice path; only the weight upload changes
    // (pre-uploaded Buffer instead of fresh `new_buffer_with_bytes`).
    fn dispatch_gemv_f32_pinned(
        ctx: &MetalContext,
        kernel_name: &str,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned shape mismatch: x={} rows={} cols={} out={}",
                x.len(),
                rows,
                cols,
                out.len()
            )));
        }
        let expected_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;
        if w_buf.length() < expected_bytes {
            return Err(Error::Kernel(format!(
                "{kernel_name}_pinned weight buffer too small: got {} expected {}",
                w_buf.length(),
                expected_bytes
            )));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            kernel_name,
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // Shared dispatch for the two f32 GEMV variants (attn o_proj, moe gate
    // logits). Same kernel body in their respective shader files; only the
    // function name differs because the manifest splits them across
    // shaders/{attn,moe}.metal as separate gates.
    fn dispatch_gemv_f32(
        ctx: &MetalContext,
        kernel_name: &str,
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!(
                "{kernel_name} shape mismatch: x={} rows={} cols={} out={}",
                x.len(),
                rows,
                cols,
                out.len()
            )));
        }
        if w.len() != rows * cols {
            return Err(Error::Kernel(format!(
                "{kernel_name} weight len mismatch: got {} expected {}",
                w.len(),
                rows * cols
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            kernel_name,
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Wedge B — TCB variant of add_inplace_metal. Encodes into `tcb` without
    /// committing. Caller commits when a batch boundary is appropriate.
    pub fn add_inplace_metal_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        a_buf: &PinnedBuffer,
        b_buf: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        let n_u32 = n as u32;
        let n_tg = (n_u32 + TG_SIZE - 1) / TG_SIZE;
        tcb.dispatch_threads(
            "add_inplace",
            (n_tg * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(a_buf), 0);
                enc.set_buffer(1, Some(b_buf), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &n_u32 as *const u32 as *const _,
                );
            },
        )
    }

    // ── v0.5.6 buffer-arg dispatcher siblings ─────────────────────────────
    //
    // Each function below is a "buf" sibling of an existing dispatcher.
    // The difference: callers pass pre-existing Metal Buffers instead of
    // having the dispatcher allocate per-call. Same kernel, same binding
    // scheme — only the buffer-allocation boilerplate is removed.

    /// v0.5.6 — buffer-arg sibling of `rmsnorm_metal`.
    /// Takes pre-existing f16 Metal Buffers; skips the Vec→Buffer round-trip.
    /// Same kernel `"rmsnorm"`, same binding scheme (buf0=x, buf1=weight,
    /// buf2=out, bytes3=hidden, bytes4=eps, tg0=shmem).
    pub fn rmsnorm_metal_buf(
        ctx: &MetalContext,
        x_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<u32>() as u64,
                &hidden_u32 as *const u32 as *const _,
            );
            enc.set_bytes(
                4,
                std::mem::size_of::<f32>() as u64,
                &eps as *const f32 as *const _,
            );
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Wedge B — TCB variant of rmsnorm for the f32 residual stream.
    /// Uses `"rmsnorm_f32"` kernel (f32 x, f32 weight → f32 out). Encodes into
    /// `tcb` without committing. Caller commits when a batch boundary is appropriate.
    pub fn rmsnorm_metal_buf_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("rmsnorm_f32", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.5.6 — buffer-arg variant of the f16 silu_mul kernel.
    /// Takes pre-existing f16 Metal Buffers. Kernel `"silu_mul"` in
    /// common.metal: out[i] = silu(gate[i]) * up[i], f16 I/O, f32 internal.
    pub fn silu_mul_metal_buf(
        ctx: &MetalContext,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        let n_u32 = n as u32;
        let n_tg = (n_u32 + TG_SIZE - 1) / TG_SIZE;
        ctx.dispatch_threads(
            "silu_mul",
            (n_tg * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(gate_buf), 0);
                enc.set_buffer(1, Some(up_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &n_u32 as *const u32 as *const _,
                );
            },
        )
    }

    // add_inplace_metal_buf: SKIPPED — existing `add_inplace_metal` already
    // takes PinnedBuffer args (it IS the buf variant). No wrapper needed.

    /// v0.5.6 — buffer-arg sibling of `gemv_f32_attn_metal`.
    /// `w` is still a host slice (allocates a temp buffer); `x_buf` and
    /// `y_buf` are pre-existing Metal Buffers. Same kernel `"gemv_f32_attn"`.
    pub fn gemv_f32_attn_metal_buf(
        ctx: &MetalContext,
        w: &[f32],
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        if w.len() != rows * cols {
            return Err(Error::Kernel(format!(
                "gemv_f32_attn_metal_buf weight len mismatch: got {} expected {}",
                w.len(), rows * cols
            )));
        }
        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads(
            "gemv_f32_attn",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// v0.5.6 — buffer-arg sibling of `gemv_f32_attn_metal_pinned`.
    /// All three matrix buffers are pre-existing; no allocation inside.
    /// Same kernel `"gemv_f32_attn"`.
    pub fn gemv_f32_attn_metal_pinned_buf(
        ctx: &MetalContext,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads(
            "gemv_f32_attn",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// v0.5.6 — buffer-arg sibling of `dispatch_gemv_f32_attn_pinned_pair_batched`.
    /// All buffers are pre-existing; dispatches two `"gemv_f32_attn"` kernels
    /// in a single CommandBatch, sharing the same x_buf.
    pub fn gemv_f32_attn_pair_metal_buf(
        ctx: &MetalContext,
        w_a_buf: &PinnedBuffer,
        rows_a: usize,
        w_b_buf: &PinnedBuffer,
        rows_b: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_a_buf: &PinnedBuffer,
        out_b_buf: &PinnedBuffer,
    ) -> Result<()> {
        ctx.dispatch_batch(|batch| {
            encode_gemv_f32_attn_pinned(batch, w_a_buf, rows_a, cols, x_buf, out_a_buf)?;
            encode_gemv_f32_attn_pinned(batch, w_b_buf, rows_b, cols, x_buf, out_b_buf)
        })
    }

    /// v0.5.6 — buffer-arg sibling of `gemv_f32_moe_metal`.
    /// `w` is still a host slice (allocates a temp buffer); `x_buf` and
    /// `y_buf` are pre-existing Metal Buffers. Same kernel `"gemv_f32_moe"`.
    pub fn gemv_f32_moe_metal_buf(
        ctx: &MetalContext,
        w: &[f32],
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        if w.len() != rows * cols {
            return Err(Error::Kernel(format!(
                "gemv_f32_moe_metal_buf weight len mismatch: got {} expected {}",
                w.len(), rows * cols
            )));
        }
        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads(
            "gemv_f32_moe",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// v0.5.6 — buffer-arg sibling of `moe_grouped_gemm_q4_metal`.
    /// `w_q4_bytes` is still a host slice (allocates a temp buffer);
    /// `x_buf` and `y_buf` are pre-existing Metal Buffers.
    /// Same kernel `"moe_grouped_gemm_q4"`.
    pub fn moe_grouped_gemm_q4_metal_buf(
        ctx: &MetalContext,
        w_q4_bytes: &[u8],
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!(
                "moe_grouped_gemm_q4_metal_buf requires cols % 256 == 0; got cols={cols}"
            )));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_q4_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!(
                "moe_grouped_gemm_q4_metal_buf weight bytes: got {} expected {}",
                w_q4_bytes.len(), expected_bytes
            )));
        }
        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads(
            "moe_grouped_gemm_q4",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    // ── end v0.5.6 buffer-arg dispatcher siblings ─────────────────────────

    // ── v0.5.7 GPU sampling dispatchers ──────────────────────────────────────

    // ── end v0.5.7 GPU sampling dispatchers ──────────────────────────────────

    // ── v0.5.8 fused RMSNorm+GEMV dispatchers ────────────────────────────────

    // ── v0.5.9 fp16 activation kernel dispatchers ─────────────────────────────

    // ── end v0.5.9 fp16 activation kernel dispatchers ─────────────────────────

    // ── v0.5.10 fp16 Q-format kernel dispatchers ──────────────────────────────

    // ── end v0.5.10 fp16 Q-format kernel dispatchers ──────────────────────────

    // ── v1.0.0-C: TokenCommandBuffer variants for attention + FFN kernels ─────
    // These functions encode kernels into an external or internal TCB rather
    // than calling ctx.dispatch_threads/dispatch_batch. TCB commits are NOT
    // counted toward dispatch_commits_per_token, enabling the target ≤30/token.

    /// Encode one f32 GEMV (pinned w, arena x → arena out) into TCB.
    /// Reuses the `gemv_f32_attn` kernel; no ctx.dispatch_threads call.
    pub fn gemv_f32_attn_pinned_buf_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(
            "gemv_f32_attn",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_buffer(3, Some(ab.handle()), 0);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// Encode two f32 GEMVs sharing x (q_a_proj + kv_a_proj) into TCB.
    /// Both kernels encode sequentially; single commit by caller.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_f32_attn_pair_arena_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_a_buf: &PinnedBuffer,
        rows_a: usize,
        w_b_buf: &PinnedBuffer,
        rows_b: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_a_buf: &PinnedBuffer,
        out_b_buf: &PinnedBuffer,
    ) -> Result<()> {
        gemv_f32_attn_pinned_buf_tcb(tcb, w_a_buf, rows_a, cols, x_buf, out_a_buf)?;
        gemv_f32_attn_pinned_buf_tcb(tcb, w_b_buf, rows_b, cols, x_buf, out_b_buf)
    }

    /// Apply f32 RoPE in-place to the rope slice of every Q head.
    pub fn rope_q_f32_inplace_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q_buf: &PinnedBuffer,
        n_heads: usize,
        q_head_dim: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        pos: u32,
        base: f32,
    ) -> Result<()> {
        let n_heads_u32 = n_heads as u32;
        let q_head_u32 = q_head_dim as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let total_pairs = n_heads_u32 * (qk_rope_u32 / 2);
        let tg = TG_SIZE.min(total_pairs.max(1));
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[
            ArgLayout::U32, ArgLayout::U32, ArgLayout::U32,
            ArgLayout::U32, ArgLayout::U32, ArgLayout::F32,
        ])?;
        ab.set_u32(0, n_heads_u32);
        ab.set_u32(1, q_head_u32);
        ab.set_u32(2, qk_nope_u32);
        ab.set_u32(3, qk_rope_u32);
        ab.set_u32(4, pos);
        ab.set_f32(5, base);
        tcb.dispatch_threads("rope_q_f32_inplace", (total_pairs, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(q_buf), 0);
            enc.set_buffer(1, Some(ab.handle()), 0);
        })
    }

    /// Apply f32 RoPE in-place to a contiguous slice inside a larger f32 buffer.
    pub fn rope_slice_f32_inplace_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        buf: &PinnedBuffer,
        offset_f32: usize,
        head_dim: usize,
        pos: u32,
        base: f32,
    ) -> Result<()> {
        let offset_u32 = offset_f32 as u32;
        let head_dim_u32 = head_dim as u32;
        let half_dim = head_dim_u32 / 2;
        let tg = TG_SIZE.min(half_dim.max(1));
        tcb.dispatch_threads("rope_slice_f32_inplace", (half_dim, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(buf), 0);
            enc.set_bytes(1, std::mem::size_of::<u32>() as u64, &offset_u32 as *const u32 as *const _);
            enc.set_bytes(2, std::mem::size_of::<u32>() as u64, &head_dim_u32 as *const u32 as *const _);
            enc.set_bytes(3, std::mem::size_of::<u32>() as u64, &pos as *const u32 as *const _);
            enc.set_bytes(4, std::mem::size_of::<f32>() as u64, &base as *const f32 as *const _);
        })
    }

    /// Append one KV entry to persistent GPU KV buffers (GPU-resident KV cache).
    /// Encodes kv_append_f32 kernel into the provided TCB (no commit).
    /// src_c_kv_normed: c_kv_normed_buf (kv_lora_rank f32).
    /// src_kv_a_out: kv_a_out_buf (kv_a_dim f32; k_pe is at [kv_lora_rank..]).
    /// dst_c_kv / dst_k_pe: persistent per-layer GPU buffers (max_seq capacity).
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src_c_kv_normed: &PinnedBuffer,
        src_kv_a_out: &PinnedBuffer,
        dst_c_kv: &PinnedBuffer,
        dst_k_pe: &PinnedBuffer,
        seq_slot: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
    ) -> Result<()> {
        let seq_slot_u32 = seq_slot as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let rope_u32 = qk_rope_head_dim as u32;
        let n_threads = kv_lora_rank.max(qk_rope_head_dim) as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, seq_slot_u32);
        ab.set_u32(1, kv_lora_u32);
        ab.set_u32(2, rope_u32);
        tcb.dispatch_threads(
            "kv_append_f32",
            (n_threads, 1, 1),
            (64u32.min(n_threads), 1, 1),
            |enc| {
                enc.set_buffer(0, Some(src_c_kv_normed), 0);
                enc.set_buffer(1, Some(src_kv_a_out), 0);
                enc.set_buffer(2, Some(dst_c_kv), 0);
                enc.set_buffer(3, Some(dst_k_pe), 0);
                enc.set_buffer(4, Some(ab.handle()), 0);
            },
        )
    }

    /// Encode mla_decode_kernel + o_proj gemv into external TCB.
    /// Reads arena.q / c_kv / k_pe; writes arena.attn_out / arena.out.
    /// c_kv and k_pe are passed explicitly so callers can use persistent GPU
    /// KV buffers (GPU-resident KV cache) or arena scratch buffers.
    /// No commit — caller commits the TCB when ready.
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decode_and_o_proj_arena_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        arena: &DecodeArena,
        kv_b_proj: &PinnedBuffer,
        o_proj: &PinnedBuffer,
        c_kv: &PinnedBuffer,
        k_pe: &PinnedBuffer,
        n_heads: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        kv_lora_rank: usize,
        seq_len: usize,
        scale: f32,
        hidden: usize,
    ) -> Result<()> {
        let n_heads_u32 = n_heads as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let v_head_u32 = v_head_dim as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let seq_len_u32 = seq_len as u32;
        let q_nope_proj_bytes = (kv_lora_rank as u64) * std::mem::size_of::<f32>() as u64;
        let scores_bytes = (seq_len as u64) * std::mem::size_of::<f32>() as u64;

        tcb.dispatch_threads(
            "mla_decode_kernel",
            (n_heads_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&arena.q), 0);
                enc.set_buffer(1, Some(c_kv), 0);
                enc.set_buffer(2, Some(k_pe), 0);
                enc.set_buffer(3, Some(kv_b_proj), 0);
                enc.set_buffer(4, Some(&arena.attn_out), 0);
                enc.set_bytes(5, std::mem::size_of::<u32>() as u64, &n_heads_u32 as *const u32 as *const _);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &qk_nope_u32 as *const u32 as *const _);
                enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &qk_rope_u32 as *const u32 as *const _);
                enc.set_bytes(8, std::mem::size_of::<u32>() as u64, &v_head_u32 as *const u32 as *const _);
                enc.set_bytes(9, std::mem::size_of::<u32>() as u64, &kv_lora_u32 as *const u32 as *const _);
                enc.set_bytes(10, std::mem::size_of::<u32>() as u64, &seq_len_u32 as *const u32 as *const _);
                enc.set_bytes(11, std::mem::size_of::<f32>() as u64, &scale as *const f32 as *const _);
                enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                enc.set_threadgroup_memory_length(1, scores_bytes);
                enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
            },
        )?;
        // o_proj pinned as f16; use gemv_f16_simdmat (half w × float x → float y).
        // Cols = n_heads × v_head_dim = 2048 and rows = hidden = 2048 (both % 8 == 0).
        gemv_f16_simdmat_tcb(
            tcb, o_proj, hidden, n_heads * v_head_dim, &arena.attn_out, &arena.out,
        )
    }

    /// Encode one f32 MoE gate-logit GEMV (mmap-pinned w, buffer x → buffer out) into TCB.
    pub fn gemv_f32_moe_pinned_buf_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(
            "gemv_f32_moe",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_buffer(3, Some(ab.handle()), 0);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// TCB version of encode_batched_gemv_indexed (private helper).
    fn encode_batched_gemv_indexed_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        kernel_name: &str,
        model_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        base_offset: usize,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let base_offset_u64 = base_offset as u64;
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let is_v2t = kernel_name.ends_with("_v2t");
        let is_v2_family = kernel_name.ends_with("_v2") || kernel_name.ends_with("_v2s") || is_v2t;
        let n_tg_x = if is_v2_family { (rows_u32 + 7) / 8 } else { rows_u32 };
        let shmem_bytes = if is_v2t {
            // x_cache: cols floats in threadgroup SRAM
            (cols as u64) * std::mem::size_of::<f32>() as u64
        } else if is_v2_family {
            0u64
        } else {
            (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64
        };
        tcb.dispatch_threads(
            kernel_name,
            (n_tg_x * tg_size, routes_u32, 1),
            (tg_size, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), 0);
                enc.set_buffer(1, Some(route_ids_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                enc.set_buffer(3, Some(out_buf), 0);
                enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &base_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, std::mem::size_of::<u32>() as u64, &routes_u32 as *const u32 as *const _);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                if !is_v2_family || is_v2t {
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                }
            },
        )
    }

    // Serial variant of encode_batched_gemv_indexed_tcb. Dispatches one route per
    // kernel submission so each expert's weight slab is read as a single stream,
    // improving L2 hit rate vs. the 6-stream parallel baseline.
    // x_buf is route-major (x[route*cols..+cols]); out_buf is route-major
    // (out[route*rows..+rows]). Buffer offsets route both to slot 0 in each dispatch.
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_gemv_indexed_serial_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        kernel_name: &str,
        model_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        base_offset: usize,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let base_offset_u64 = base_offset as u64;
        let routes_one      = 1u32;
        let rows_u32        = rows as u32;
        let cols_u32        = cols as u32;
        let tg_size         = TG_SIZE as u32;
        let is_v2t      = kernel_name.ends_with("_v2t");
        let is_v2_family = kernel_name.ends_with("_v2")
            || kernel_name.ends_with("_v2s")
            || is_v2t;
        let n_tg_x = if is_v2_family { (rows_u32 + 7) / 8 } else { rows_u32 };
        let shmem_bytes = if is_v2t {
            (cols as u64) * std::mem::size_of::<f32>() as u64
        } else if is_v2_family {
            0u64
        } else {
            (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64
        };
        for route_i in 0..routes {
            let ids_off = (route_i * std::mem::size_of::<u32>()) as u64;
            // x is route-major: offset x_buf so x[0*cols..] = original x[route_i*cols..].
            let x_off   = (route_i * cols * std::mem::size_of::<f32>()) as u64;
            let out_off = (route_i * rows * std::mem::size_of::<f32>()) as u64;
            tcb.dispatch_threads(
                kernel_name,
                (n_tg_x * tg_size, 1, 1),
                (tg_size, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(model_buf),     0);
                    enc.set_buffer(1, Some(route_ids_buf), ids_off);
                    enc.set_buffer(2, Some(x_buf),         x_off);
                    enc.set_buffer(3, Some(out_buf),       out_off);
                    enc.set_bytes(4, 8, &base_offset_u64 as *const u64 as *const _);
                    enc.set_bytes(5, 4, &routes_one as *const u32 as *const _);
                    enc.set_bytes(6, 4, &rows_u32   as *const u32 as *const _);
                    enc.set_bytes(7, 4, &cols_u32   as *const u32 as *const _);
                    if !is_v2_family || is_v2t {
                        enc.set_threadgroup_memory_length(0, shmem_bytes);
                    }
                },
            )?;
        }
        Ok(())
    }

    pub fn silu_mul_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        let n_u32 = n as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32])?;
        ab.set_u32(0, n_u32);
        tcb.dispatch_threads("moe_batched_silu_mul", (n_u32, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(gate_buf), 0);
            enc.set_buffer(1, Some(up_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_batched_gemv_fused_gu_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        act_buf: &PinnedBuffer,
        gate_offset: usize,
        up_offset: usize,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64   = up_offset   as u64;
        let routes_u32 = routes as u32;
        let rows_u32   = rows   as u32;
        let cols_u32   = cols   as u32;
        let tg_size    = TG_SIZE as u32;
        let n_tg_x     = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "moe_batched_gemm_q4_indexed_v2t_gu",
            (n_tg_x * tg_size, routes_u32, 1),
            (tg_size, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf),    0);
                enc.set_buffer(1, Some(route_ids_buf), 0);
                enc.set_buffer(2, Some(x_buf),         0);
                enc.set_buffer(3, Some(act_buf),       0);
                enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64   as *const u64 as *const _);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &routes_u32 as *const u32 as *const _);
                enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &rows_u32   as *const u32 as *const _);
                enc.set_bytes(8, std::mem::size_of::<u32>() as u64, &cols_u32   as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    // v2t_gu_v2: same signature as encode_batched_gemv_fused_gu_tcb but dispatches
    // moe_batched_gemm_q4_indexed_v2t_gu_v2 (sumy trick + scale preload +
    // paired nibble reads — Phase 2 optimisation).
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_gemv_fused_gu_v2_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        act_buf: &PinnedBuffer,
        gate_offset: usize,
        up_offset: usize,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64   = up_offset   as u64;
        let routes_u32 = routes as u32;
        let rows_u32   = rows   as u32;
        let cols_u32   = cols   as u32;
        let tg_size    = TG_SIZE as u32;
        let n_tg_x     = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "moe_batched_gemm_q4_indexed_v2t_gu_v2",
            (n_tg_x * tg_size, routes_u32, 1),
            (tg_size, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf),     0);
                enc.set_buffer(1, Some(route_ids_buf), 0);
                enc.set_buffer(2, Some(x_buf),         0);
                enc.set_buffer(3, Some(act_buf),        0);
                enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64   as *const u64 as *const _);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &routes_u32 as *const u32 as *const _);
                enc.set_bytes(7, std::mem::size_of::<u32>() as u64, &rows_u32   as *const u32 as *const _);
                enc.set_bytes(8, std::mem::size_of::<u32>() as u64, &cols_u32   as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    // Serial variant: dispatches one route at a time so each expert's weights
    // (gate+up = ~3MB) are read as a single sequential stream that fits in L2,
    // avoiding the cache-thrashing caused by 6 simultaneous scattered expert streams.
    // Requires a single command buffer (Pillar 2) to amortise per-dispatch overhead.
    #[allow(clippy::too_many_arguments)]
    fn encode_batched_gemv_fused_gu_serial_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        act_buf: &PinnedBuffer,
        gate_offset: usize,
        up_offset: usize,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64   = up_offset   as u64;
        let routes_one      = 1u32;
        let rows_u32        = rows as u32;
        let cols_u32        = cols as u32;
        let tg_size         = TG_SIZE as u32;
        let n_tg_x          = (rows_u32 + 7) / 8;
        let shmem_bytes     = (cols as u64) * std::mem::size_of::<f32>() as u64;
        for route_i in 0..routes {
            // Offset route_ids so route_ids[0] = expert for this route.
            let ids_off = (route_i * std::mem::size_of::<u32>()) as u64;
            // Offset act_buf so this route writes to act[route_i * rows .. +rows].
            let act_off = (route_i * rows * std::mem::size_of::<f32>()) as u64;
            // x (hidden state) is the same for all routes — no offset needed.
            tcb.dispatch_threads(
                "moe_batched_gemm_q4_indexed_v2t_gu",
                (n_tg_x * tg_size, 1, 1),
                (tg_size, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(model_buf),     0);
                    enc.set_buffer(1, Some(route_ids_buf), ids_off);
                    enc.set_buffer(2, Some(x_buf),         0);
                    enc.set_buffer(3, Some(act_buf),       act_off);
                    enc.set_bytes(4, 8, &gate_offset_u64 as *const u64 as *const _);
                    enc.set_bytes(5, 8, &up_offset_u64   as *const u64 as *const _);
                    enc.set_bytes(6, 4, &routes_one as *const u32 as *const _);
                    enc.set_bytes(7, 4, &rows_u32   as *const u32 as *const _);
                    enc.set_bytes(8, 4, &cols_u32   as *const u32 as *const _);
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                },
            )?;
        }
        Ok(())
    }

    fn encode_route_accumulate_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        routed_out: &PinnedBuffer,
        weights: &PinnedBuffer,
        shared_out: &PinnedBuffer,
        out: &PinnedBuffer,
        hidden: usize,
        routes: usize,
        has_shared: bool,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let routes_u32 = routes as u32;
        let has_shared_u32 = u32::from(has_shared);
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_u32(1, routes_u32);
        ab.set_u32(2, has_shared_u32);
        tcb.dispatch_threads("moe_route_accumulate", (hidden_u32, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(routed_out), 0);
            enc.set_buffer(1, Some(weights), 0);
            enc.set_buffer(2, Some(shared_out), 0);
            enc.set_buffer(3, Some(out), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    pub fn moe_topk_gate_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        logits_buf: &PinnedBuffer,
        route_ids_buf: &PinnedBuffer,
        route_weights_buf: &PinnedBuffer,
        n_experts: usize,
        top_k: usize,
    ) -> Result<()> {
        if top_k == 0 {
            return Err(Error::Kernel("moe_topk_gate_tcb: top_k must be > 0".into()));
        }
        let n_experts_u32 = n_experts as u32;
        let top_k_u32 = top_k as u32;
        let shmem_bytes = (n_experts as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n_experts_u32);
        ab.set_u32(1, top_k_u32);
        tcb.dispatch_threads("moe_topk_gate", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(logits_buf), 0);
            enc.set_buffer(1, Some(route_ids_buf), 0);
            enc.set_buffer(2, Some(route_weights_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_shared_only_indexed_tcb_with_scratch(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        shared_route_ids_buf: &PinnedBuffer,
        shared_gate_offset: usize,
        shared_up_offset: usize,
        shared_down_offset: usize,
        hidden: usize,
        shared_mid: usize,
        q4k_schedule: &str,
        shared_down_kernel: &str,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        shared_gate_out: &PinnedBuffer,
        shared_up_out: &PinnedBuffer,
        shared_act: &PinnedBuffer,
    ) -> Result<()> {
        let q4k_indexed_kernel = match q4k_schedule {
            "v2" | "llama_port" | "per_shape" => "moe_batched_gemm_q4_indexed_v2",
            "v2s" => "moe_batched_gemm_q4_indexed_v2s",
            "v2t" | "v2t_gu" | "v2t_gu_serial" | "v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t",
            _ => "moe_batched_gemm_q4_indexed",
        };

        if q4k_schedule == "v2t_gu_v2" {
            encode_batched_gemv_fused_gu_v2_tcb(
                tcb, model_buf, shared_route_ids_buf, x_buf, shared_act,
                shared_gate_offset, shared_up_offset, 1, shared_mid, hidden,
            )?;
        } else if q4k_schedule == "v2t_gu" || q4k_schedule == "v2t_gu_serial" {
            encode_batched_gemv_fused_gu_tcb(
                tcb, model_buf, shared_route_ids_buf, x_buf, shared_act,
                shared_gate_offset, shared_up_offset, 1, shared_mid, hidden,
            )?;
        } else {
            encode_batched_gemv_indexed_tcb(
                tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf,
                shared_gate_out, shared_gate_offset, 1, shared_mid, hidden,
            )?;
            encode_batched_gemv_indexed_tcb(
                tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf,
                shared_up_out, shared_up_offset, 1, shared_mid, hidden,
            )?;
            silu_mul_tcb(tcb, shared_gate_out, shared_up_out, shared_act, shared_mid)?;
        }

        encode_batched_gemv_indexed_tcb(
            tcb, shared_down_kernel, model_buf, shared_route_ids_buf,
            shared_act, out_buf, shared_down_offset, 1, hidden, shared_mid,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_block_batched_indexed_tcb_with_scratch(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        routed_gate_offset: usize,
        routed_up_offset: usize,
        routed_down_offset: usize,
        route_ids_buf: &PinnedBuffer,
        route_weights_buf: &PinnedBuffer,
        routes: usize,
        shared_route_ids_buf: &PinnedBuffer,
        shared_gate_offset: Option<usize>,
        shared_up_offset: Option<usize>,
        shared_down_offset: Option<usize>,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        q4k_schedule: &str,
        routed_down_kernel: &str,
        shared_down_kernel: &str,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        routed_gate_out: &PinnedBuffer,
        routed_up_out: &PinnedBuffer,
        routed_act: &PinnedBuffer,
        routed_out: &PinnedBuffer,
        shared_gate_out: &PinnedBuffer,
        shared_up_out: &PinnedBuffer,
        shared_act: &PinnedBuffer,
        shared_out: &PinnedBuffer,
    ) -> Result<()> {
        if routes == 0 {
            return Err(Error::Kernel("encode_moe_block_batched_indexed_tcb_with_scratch: no routes".into()));
        }

        let has_shared = shared_gate_offset.is_some()
            || shared_up_offset.is_some()
            || shared_down_offset.is_some();

        let q4k_indexed_kernel = match q4k_schedule {
            "v2" | "llama_port" | "per_shape" => "moe_batched_gemm_q4_indexed_v2",
            "v2s" => "moe_batched_gemm_q4_indexed_v2s",
            "v2t" | "v2t_gu" | "v2t_gu_serial" | "v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t",
            _ => "moe_batched_gemm_q4_indexed",
        };
        let use_fused_gu_v2  = q4k_schedule == "v2t_gu_v2";
        let use_fused_gu     = q4k_schedule == "v2t_gu";
        // Serial: dispatch one expert at a time so each expert's weight slab (~3 MB
        // gate+up) is a single sequential stream. Eliminates 6-stream L2 thrashing.
        // Effective only when combined with a single command buffer (Pillar 2).
        let use_serial_gu    = q4k_schedule == "v2t_gu_serial";

        if use_serial_gu {
            encode_batched_gemv_fused_gu_serial_tcb(
                tcb, model_buf, route_ids_buf, x_buf, routed_act,
                routed_gate_offset, routed_up_offset, routes, routed_mid, hidden,
            )?;
        } else if use_fused_gu_v2 {
            encode_batched_gemv_fused_gu_v2_tcb(
                tcb, model_buf, route_ids_buf, x_buf, routed_act,
                routed_gate_offset, routed_up_offset, routes, routed_mid, hidden,
            )?;
        } else if use_fused_gu {
            encode_batched_gemv_fused_gu_tcb(
                tcb, model_buf, route_ids_buf, x_buf, routed_act,
                routed_gate_offset, routed_up_offset, routes, routed_mid, hidden,
            )?;
        } else {
            encode_batched_gemv_indexed_tcb(
                tcb, q4k_indexed_kernel, model_buf, route_ids_buf, x_buf,
                routed_gate_out, routed_gate_offset, routes, routed_mid, hidden,
            )?;
            encode_batched_gemv_indexed_tcb(
                tcb, q4k_indexed_kernel, model_buf, route_ids_buf, x_buf,
                routed_up_out, routed_up_offset, routes, routed_mid, hidden,
            )?;
            silu_mul_tcb(tcb, routed_gate_out, routed_up_out, routed_act, routes * routed_mid)?;
        }

        // Down projection: also serial when using v2t_gu_serial to fix the same
        // L2 thrashing on the down-projection weight slabs.
        if use_serial_gu {
            encode_batched_gemv_indexed_serial_tcb(
                tcb, routed_down_kernel, model_buf, route_ids_buf,
                routed_act, routed_out, routed_down_offset, routes, hidden, routed_mid,
            )?;
        } else {
            encode_batched_gemv_indexed_tcb(
                tcb, routed_down_kernel, model_buf, route_ids_buf,
                routed_act, routed_out, routed_down_offset, routes, hidden, routed_mid,
            )?;
        }

        if let (Some(gate_off), Some(up_off), Some(down_off)) =
            (shared_gate_offset, shared_up_offset, shared_down_offset)
        {
            // Shared expert always routes=1, so serial == parallel. Use the
            // appropriate fused_gu variant when any gu schedule is selected.
            if use_fused_gu_v2 {
                encode_batched_gemv_fused_gu_v2_tcb(
                    tcb, model_buf, shared_route_ids_buf, x_buf, shared_act,
                    gate_off, up_off, 1, shared_mid, hidden,
                )?;
            } else if use_fused_gu || use_serial_gu {
                encode_batched_gemv_fused_gu_tcb(
                    tcb, model_buf, shared_route_ids_buf, x_buf, shared_act,
                    gate_off, up_off, 1, shared_mid, hidden,
                )?;
            } else {
                encode_batched_gemv_indexed_tcb(
                    tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf,
                    shared_gate_out, gate_off, 1, shared_mid, hidden,
                )?;
                encode_batched_gemv_indexed_tcb(
                    tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf,
                    shared_up_out, up_off, 1, shared_mid, hidden,
                )?;
                silu_mul_tcb(tcb, shared_gate_out, shared_up_out, shared_act, shared_mid)?;
            }
            encode_batched_gemv_indexed_tcb(
                tcb, shared_down_kernel, model_buf, shared_route_ids_buf,
                shared_act, shared_out, down_off, 1, hidden, shared_mid,
            )?;
        }

        encode_route_accumulate_tcb(
            tcb, routed_out, route_weights_buf, shared_out, out_buf,
            hidden, routes, has_shared,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_moe_block_batched_indexed_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        routed_gate_offset: usize,
        routed_up_offset: usize,
        routed_down_offset: usize,
        route_ids_buf: &PinnedBuffer,
        route_weights_buf: &PinnedBuffer,
        routes: usize,
        shared_route_ids_buf: &PinnedBuffer,
        shared_gate_offset: Option<usize>,
        shared_up_offset: Option<usize>,
        shared_down_offset: Option<usize>,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        q4k_schedule: &str,
        routed_down_kernel: &str,
        shared_down_kernel: &str,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<Vec<PinnedBuffer>> {
        let routed_gate_out = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_up_out = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_act = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_out = ctx.new_buffer(routes * hidden * std::mem::size_of::<f32>());
        let shared_gate_out = ctx.new_buffer(shared_mid.max(1) * std::mem::size_of::<f32>());
        let shared_up_out = ctx.new_buffer(shared_mid.max(1) * std::mem::size_of::<f32>());
        let shared_act = ctx.new_buffer(shared_mid.max(1) * std::mem::size_of::<f32>());
        let shared_out = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        encode_moe_block_batched_indexed_tcb_with_scratch(
            tcb, model_buf,
            routed_gate_offset, routed_up_offset, routed_down_offset,
            route_ids_buf, route_weights_buf, routes, shared_route_ids_buf,
            shared_gate_offset, shared_up_offset, shared_down_offset,
            hidden, routed_mid, shared_mid, q4k_schedule,
            routed_down_kernel, shared_down_kernel, x_buf, out_buf,
            &routed_gate_out, &routed_up_out, &routed_act, &routed_out,
            &shared_gate_out, &shared_up_out, &shared_act, &shared_out,
        )?;

        Ok(vec![
            routed_gate_out,
            routed_up_out,
            routed_act,
            routed_out,
            shared_gate_out,
            shared_up_out,
            shared_act,
            shared_out,
        ])
    }

    /// v1.0.0-C: MoE block via internal TCB (zero counted dispatches).
    /// Functionally identical to `moe_block_batched_indexed_metal` but uses
    /// TokenCommandBuffer internally so stats.commits is NOT incremented.
    /// `x_buf` and `out_buf` are pre-allocated arena buffers.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_block_batched_indexed_tcb(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        routed_gate_offset: usize,
        routed_up_offset: usize,
        routed_down_offset: usize,
        _n_routed_experts: usize,
        route_ids: &[u32],
        route_weights: &[f32],
        shared_gate_offset: Option<usize>,
        shared_up_offset: Option<usize>,
        shared_down_offset: Option<usize>,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        q4k_schedule: &str,
        routed_down_kernel: &str,
        shared_down_kernel: &str,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let routes = route_ids.len();
        if routes == 0 {
            return Err(Error::Kernel("moe_block_batched_indexed_tcb: no routes".into()));
        }

        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(route_ids));
        let route_weights_buf =
            ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let shared_route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&[0u32]));

        let mut tcb = TokenCommandBuffer::new(ctx);
        let _temp_buffers = encode_moe_block_batched_indexed_tcb(
            &mut tcb, ctx, model_buf,
            routed_gate_offset, routed_up_offset, routed_down_offset,
            &route_ids_buf, &route_weights_buf, routes, &shared_route_ids_buf,
            shared_gate_offset, shared_up_offset, shared_down_offset,
            hidden, routed_mid, shared_mid, q4k_schedule,
            routed_down_kernel, shared_down_kernel, x_buf, out_buf,
        )?;
        tcb.commit_and_wait()?;
        // temp buffers dropped here, after GPU is done
        Ok(())
    }

    // ── v1.0.0-D: embed lookup writing f32 residual directly to GPU buffer ──

    /// Encode embed_lookup_f32 into TCB: reads f16 embed table at row `token`,
    /// writes hidden f32 values into x_buf. Zero counted dispatches.
    pub fn embed_lookup_metal_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        embed_buf: &PinnedBuffer,
        token: u32,
        hidden: usize,
        x_buf: &PinnedBuffer,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let tg = TG_SIZE.min(hidden_u32);
        tcb.dispatch_threads(
            "embed_lookup_f32",
            (hidden_u32, 1, 1),
            (tg, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(embed_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &token as *const u32 as *const _,
                );
            },
        )
    }

    // ── v1.0.0-E: GPU argmax sampling dispatchers ────────────────────────────

    /// LM-head GEMV via TCB: w_buf (rows×cols f16) × x_buf (cols f32) → y_buf (rows f32).
    /// Zero counted dispatches. Used for the final LM-head projection in the greedy path.
    pub fn gemv_f16_metal_buf_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "gemv_f16",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// GPU greedy argmax via TCB: logits_buf (vocab f32) → token_buf (u32).
    /// Zero counted dispatches. Grid and threadgroup are both (256, 1, 1) to
    /// match the sample_argmax_f32 kernel's two-phase 256-thread reduction.
    pub fn sample_argmax_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        logits_buf: &PinnedBuffer,
        token_buf: &PinnedBuffer,
        vocab: usize,
    ) -> Result<()> {
        let vocab_u32 = vocab as u32;
        let shmem_f = 256 * std::mem::size_of::<f32>() as u64;
        let shmem_u = 256 * std::mem::size_of::<u32>() as u64;
        tcb.dispatch_threads("sample_argmax_f32", (256, 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(logits_buf), 0);
            enc.set_buffer(1, Some(token_buf), 0);
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &vocab_u32 as *const u32 as *const _,
            );
            enc.set_threadgroup_memory_length(0, shmem_f);
            enc.set_threadgroup_memory_length(1, shmem_u);
        })
    }

    // ── end v1.0.0-E ─────────────────────────────────────────────────────────

    // ── v1.0.0-G: rmsnorm-gemv fusion TCB dispatchers ────────────────────────

    /// Fused rmsnorm + f32 GEMV for attention projections (q_a, kv_a).
    /// Reads x_buf (f32 raw residual), applies attn rmsnorm with weight_buf,
    /// writes out_buf (f32 rows). Grid = (rows * TG_SIZE, 1, 1).
    /// Eliminates the standalone rmsnorm_metal_buf_tcb in mini-TCB α.
    /// Zero counted dispatches.
    pub fn rmsnorm_gemv_f32_attn_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        out_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "rmsnorm_gemv_f32_attn_pinned",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(weight_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<f32>() as u64,
                    &eps as *const f32 as *const _,
                );
                enc.set_buffer(4, Some(out_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// f16-weight variant: same binding layout as rmsnorm_gemv_f32_attn_pinned_tcb
    /// but w_buf holds f16 bytes. Halves weight bandwidth for q_a and kv_a projections.
    pub fn rmsnorm_gemv_f16w_attn_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        out_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "rmsnorm_gemv_f16w_attn_pinned",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(weight_buf), 0);
                enc.set_bytes(3, std::mem::size_of::<f32>() as u64, &eps as *const f32 as *const _);
                enc.set_buffer(4, Some(out_buf), 0);
                enc.set_bytes(5, std::mem::size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    /// v2.2.0-T2.14 — v2t-pattern dispatch: 8 rows per threadgroup, one simdgroup
    /// per row, threadgroup `xw_cache` for once-per-TG rmsnorm-scaled activation.
    /// Requires rows % 8 == 0 and cols % 32 == 0.
    pub fn rmsnorm_gemv_f16w_attn_pinned_v2t_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        out_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if rows % 8 != 0 || cols % 32 != 0 {
            return Err(crate::error::Error::Kernel(format!(
                "rmsnorm_gemv_f16w_attn_pinned_v2t requires rows%8==0 and cols%32==0; rows={rows} cols={cols}"
            )));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_tgs = (rows / 8) as u32;
        let shmem_bytes = 16u64 * std::mem::size_of::<f32>() as u64;
        let xw_cache_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "rmsnorm_gemv_f16w_attn_pinned_v2t",
            (n_tgs * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(weight_buf), 0);
                enc.set_bytes(3, std::mem::size_of::<f32>() as u64, &eps as *const f32 as *const _);
                enc.set_buffer(4, Some(out_buf), 0);
                enc.set_bytes(5, std::mem::size_of::<u32>() as u64, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(6, std::mem::size_of::<u32>() as u64, &cols_u32 as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
                enc.set_threadgroup_memory_length(1, xw_cache_bytes);
            },
        )
    }

    // ── end v1.0.0-G ─────────────────────────────────────────────────────────

    // ── v1.0.0-H: simdgroup_matrix GEMV dispatchers (Path 2) ─────────────────

    /// simdgroup_matrix GEMV: w (rows×cols f32) × x (cols f32) → y (rows f32).
    /// One SIMD group (32 threads) per threadgroup; each handles 8 output rows.
    /// Requires cols % 8 == 0. Grid = (ceil(rows/8)*32, 1, 1), TG = (32, 1, 1).
    /// Zero counted dispatches.
    pub fn gemv_simdgroup_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        if cols % 8 != 0 {
            return Err(crate::error::Error::Kernel(format!(
                "gemv_simdgroup_f32 requires cols % 8 == 0; cols={cols}"
            )));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_groups = rows.div_ceil(8) as u32;
        let scratch_bytes = 192u64 * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "gemv_simdgroup_f32",
            (n_groups * 32, 1, 1),
            (32, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, scratch_bytes);
            },
        )
    }

    // ── end v1.0.0-H ─────────────────────────────────────────────────────────

    // ── v1.1.0-X: simdgroup_matrix LM-head GEMV (f16 weights) ────────────────

    /// LM-head GEMV via simdgroup_matrix: w (rows×cols f16) × x (cols f32) → y (rows f32).
    /// Mixed-precision: half A × half B + float C → float D.
    /// One SIMD group (32 threads) per threadgroup; each handles 8 output rows.
    /// Requires cols % 8 == 0. Grid = (ceil(rows/8)*32, 1, 1), TG = (32, 1, 1).
    pub fn gemv_f16_simdmat_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        if cols % 8 != 0 {
            return Err(crate::error::Error::Kernel(format!(
                "gemv_f16_simdmat requires cols % 8 == 0; cols={cols}"
            )));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_groups = rows.div_ceil(8) as u32;
        // 3 × 64 floats: W tile + X tile + result tile
        let shmem_bytes: u64 = 192 * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "gemv_f16_simdmat",
            (n_groups * 32, 1, 1),
            (32, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    // ── end v1.1.0-X ─────────────────────────────────────────────────────────


    // ── Phase 5C.2: f32→f16 norm output + f16-activation LM head ─────────────

    /// f32 residual → f16 normed activation (Phase 5C.2).
    /// Dispatches `rmsnorm_f32_to_f16`: reads f32 x, f32 weight → writes half* out.
    /// Variance accumulator stays f32. Used when kernel profile x_norm_dtype="f16".
    /// Same ArgbufRmsnorm pattern as rmsnorm_metal_buf_tcb. out_buf must be
    /// pre-allocated as hidden × sizeof(f16) bytes (arena.x_norm_f16_buf).
    pub fn rmsnorm_f32_to_f16_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("rmsnorm_f32_to_f16", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(ab.handle()), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// f16-weight × f16-activation GEMV → f32 output (Phase 5C.2).
    /// Dispatches `gemv_f16_f16in`: same binding layout as gemv_f16_metal_buf_tcb
    /// except x_buf holds f16 values (arena.x_norm_f16_buf). Output y_buf is f32.
    /// MAC accumulates in f32. Used for the LM head GEMV when x_norm_dtype="f16".
    pub fn gemv_f16_f16in_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        y_buf: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads(
            "gemv_f16_f16in",
            (rows_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(y_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    // ── end Phase 5C.2 ────────────────────────────────────────────────────────
}


#[cfg(target_os = "macos")]
pub use metal_dispatch::*;

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn rmsnorm_unit_weight() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 1.0];
        let mut out = [0.0; 4];
        rmsnorm(&x, &w, 1e-6, &mut out);
        // RMS = sqrt(30/4) = sqrt(7.5)
        let rms = (7.5f32).sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] / rms).abs() < 1e-4);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut xs = [1.0, 2.0, 3.0, 4.0];
        softmax_inplace(&mut xs);
        let sum: f32 = xs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn gemv_round_trip() {
        let w_f32 = [1.0, 0.0, 0.0, 1.0];
        let w: Vec<f16> = w_f32.iter().map(|&v| f16::from_f32(v)).collect();
        let x = [3.0, 5.0];
        let mut out = [0.0; 2];
        gemv_f16(&w, 2, 2, &x, &mut out);
        assert!((out[0] - 3.0).abs() < 1e-5);
        assert!((out[1] - 5.0).abs() < 1e-5);
    }
}
