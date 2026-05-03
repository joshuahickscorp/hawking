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
    use crate::metal::{CommandBatch, MetalContext, PinnedBuffer};
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
            batch.dispatch_threads("sample_argmax_f32", (1, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(&logits_buf), 0);
                enc.set_buffer(1, Some(&token_buf), 0);
                enc.set_bytes(
                    2,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
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

    /// H2.1 — top-K softmax gate over routed-expert logits. Maps to
    /// `moe_topk_gate` in `shaders/moe.metal`. One workgroup per token.
    /// Outputs raw post-softmax probabilities of the top-k experts (no
    /// top-k renormalization) and their integer expert indices.
    pub fn moe_topk_gate_metal(
        ctx: &MetalContext,
        logits: &[f32],
        n_tokens: usize,
        n_experts: usize,
        top_k: usize,
        expert_ids: &mut [u32],
        weights: &mut [f32],
    ) -> Result<()> {
        if logits.len() != n_tokens * n_experts {
            return Err(Error::Kernel(format!(
                "moe_topk_gate_metal logits shape: got {} expected {}",
                logits.len(),
                n_tokens * n_experts
            )));
        }
        if expert_ids.len() != n_tokens * top_k || weights.len() != n_tokens * top_k {
            return Err(Error::Kernel(format!(
                "moe_topk_gate_metal output shape: ids={} weights={} expected {}",
                expert_ids.len(),
                weights.len(),
                n_tokens * top_k
            )));
        }

        let logits_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(logits));
        let ids_buf = ctx.new_buffer(n_tokens * top_k * std::mem::size_of::<u32>());
        let weights_buf = ctx.new_buffer(n_tokens * top_k * std::mem::size_of::<f32>());

        let n_experts_u32 = n_experts as u32;
        let top_k_u32 = top_k as u32;
        let n_tokens_u32 = n_tokens as u32;
        let shmem_bytes = (n_experts as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "moe_topk_gate",
            (n_tokens_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&logits_buf), 0);
                enc.set_buffer(1, Some(&ids_buf), 0);
                enc.set_buffer(2, Some(&weights_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &n_experts_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &top_k_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )?;

        let ids_ptr = ids_buf.contents() as *const u32;
        let ids_slice = unsafe { std::slice::from_raw_parts(ids_ptr, n_tokens * top_k) };
        expert_ids.copy_from_slice(ids_slice);

        let weights_ptr = weights_buf.contents() as *const f32;
        let weights_slice = unsafe { std::slice::from_raw_parts(weights_ptr, n_tokens * top_k) };
        weights.copy_from_slice(weights_slice);

        Ok(())
    }

    /// H2.3 — weighted gather of per-(token, expert) outputs into
    /// per-token activations. Maps to `moe_gather_combine` in
    /// `shaders/moe.metal`. 2D dispatch: one thread per (token, hidden)
    /// pair; loops over top_k experts internally.
    pub fn moe_gather_combine_metal(
        ctx: &MetalContext,
        expert_out: &[f32],
        weights: &[f32],
        n_tokens: usize,
        top_k: usize,
        hidden: usize,
        token_out: &mut [f32],
    ) -> Result<()> {
        if expert_out.len() != n_tokens * top_k * hidden {
            return Err(Error::Kernel(format!(
                "moe_gather_combine_metal expert_out shape: got {} expected {}",
                expert_out.len(),
                n_tokens * top_k * hidden
            )));
        }
        if weights.len() != n_tokens * top_k {
            return Err(Error::Kernel(format!(
                "moe_gather_combine_metal weights shape: got {} expected {}",
                weights.len(),
                n_tokens * top_k
            )));
        }
        if token_out.len() != n_tokens * hidden {
            return Err(Error::Kernel(format!(
                "moe_gather_combine_metal token_out shape: got {} expected {}",
                token_out.len(),
                n_tokens * hidden
            )));
        }

        let expert_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(expert_out));
        let weights_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(weights));
        let out_buf = ctx.new_buffer(n_tokens * hidden * std::mem::size_of::<f32>());

        let hidden_u32 = hidden as u32;
        let top_k_u32 = top_k as u32;

        // 2D dispatch: grid (hidden, n_tokens, 1), tg (256, 1, 1).
        // Metal's non-uniform threadgroup variant lets `hidden` be any
        // value; threads with gid.x >= hidden return early.
        let grid_x = ((hidden + TG_SIZE as usize - 1) / TG_SIZE as usize) * TG_SIZE as usize;
        ctx.dispatch_threads(
            "moe_gather_combine",
            (grid_x as u32, n_tokens as u32, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&expert_buf), 0);
                enc.set_buffer(1, Some(&weights_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &top_k_u32 as *const u32 as *const _,
                );
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, n_tokens * hidden) };
        token_out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Deterministic greedy argmax over fp32 logits. This is deliberately
    /// simple: it proves the token-only GPU readback contract before the
    /// LM-head path starts keeping logits resident.
    pub fn sample_argmax_f32_metal(ctx: &MetalContext, logits: &[f32]) -> Result<u32> {
        if logits.is_empty() {
            return Err(Error::Kernel("sample_argmax_f32_metal empty logits".into()));
        }
        if logits.len() > u32::MAX as usize {
            return Err(Error::Kernel(format!(
                "sample_argmax_f32_metal logits too large: {}",
                logits.len()
            )));
        }

        let logits_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(logits));
        let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());
        let n_u32 = logits.len() as u32;

        ctx.dispatch_threads("sample_argmax_f32", (1, 1, 1), (1, 1, 1), |enc| {
            enc.set_buffer(0, Some(&logits_buf), 0);
            enc.set_buffer(1, Some(&token_buf), 0);
            enc.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &n_u32 as *const u32 as *const _,
            );
        })?;

        let token_ptr = token_buf.contents() as *const u32;
        Ok(unsafe { *token_ptr })
    }

    /// H2.2 — fp32 GEMV with Q4_K_M weights, dequant fused inside the
    /// FMA loop. Maps to `moe_grouped_gemm_q4` in `shaders/moe.metal`.
    /// One workgroup per output row, tg_size=256 (matches the Q4_K_M
    /// super-block size). cols must be a multiple of 256.
    pub fn moe_grouped_gemm_q4_metal(
        ctx: &MetalContext,
        w_q4_bytes: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        dispatch_q4_k_m_gemv(ctx, "moe_grouped_gemm_q4", w_q4_bytes, rows, cols, x, out)
    }

    /// Phase 2 — batched Q4_K GEMV for selected routed/shared experts.
    /// `w_q4_bytes` is `routes` consecutive `(rows, cols)` matrices.
    /// The same input vector `x` is multiplied by each route matrix.
    pub fn moe_batched_gemm_q4_metal(
        ctx: &MetalContext,
        w_q4_bytes: &[u8],
        routes: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        validate_batched_quant(
            "moe_batched_gemm_q4",
            w_q4_bytes,
            routes,
            rows,
            cols,
            256,
            144,
        )?;
        if x.len() != cols || out.len() != routes * rows {
            return Err(Error::Kernel(format!(
                "moe_batched_gemm_q4 shape: x={} cols={} out={} expected {}",
                x.len(),
                cols,
                out.len(),
                routes * rows
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        dispatch_batched_gemv(
            ctx,
            "moe_batched_gemm_q4",
            &w_buf,
            &x_buf,
            &out_buf,
            routes,
            rows,
            cols,
        )?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Phase 2 — batched Q8_0 GEMV. `x` is route-major
    /// `(routes, cols)`, matching the routed activation matrix.
    pub fn moe_batched_gemm_q8_0_metal(
        ctx: &MetalContext,
        w_q8_bytes: &[u8],
        routes: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        validate_batched_quant(
            "moe_batched_gemm_q8_0",
            w_q8_bytes,
            routes,
            rows,
            cols,
            32,
            34,
        )?;
        if x.len() != routes * cols || out.len() != routes * rows {
            return Err(Error::Kernel(format!(
                "moe_batched_gemm_q8_0 shape: x={} expected {} out={} expected {}",
                x.len(),
                routes * cols,
                out.len(),
                routes * rows
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q8_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        dispatch_batched_gemv(
            ctx,
            "moe_batched_gemm_q8_0",
            &w_buf,
            &x_buf,
            &out_buf,
            routes,
            rows,
            cols,
        )?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Phase 2 — batched Q6_K GEMV. `x` is route-major
    /// `(routes, cols)`.
    pub fn moe_batched_gemm_q6_k_metal(
        ctx: &MetalContext,
        w_q6_bytes: &[u8],
        routes: usize,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        validate_batched_quant(
            "moe_batched_gemm_q6_k",
            w_q6_bytes,
            routes,
            rows,
            cols,
            256,
            210,
        )?;
        if x.len() != routes * cols || out.len() != routes * rows {
            return Err(Error::Kernel(format!(
                "moe_batched_gemm_q6_k shape: x={} expected {} out={} expected {}",
                x.len(),
                routes * cols,
                out.len(),
                routes * rows
            )));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q6_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        dispatch_batched_gemv(
            ctx,
            "moe_batched_gemm_q6_k",
            &w_buf,
            &x_buf,
            &out_buf,
            routes,
            rows,
            cols,
        )?;
        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Phase 2 — batched DeepSeek MoE block for the real Q4/Q8/Q6
    /// expert layout. Routed gate/up use Q4_K, routed down uses Q8_0;
    /// shared gate/up use Q4_K and shared down uses Q6_K.
    pub fn moe_block_batched_metal(
        ctx: &MetalContext,
        routed_gate_q4: &[u8],
        routed_up_q4: &[u8],
        routed_down_q8: &[u8],
        route_weights: &[f32],
        shared_gate_q4: Option<&[u8]>,
        shared_up_q4: Option<&[u8]>,
        shared_down_q6: Option<&[u8]>,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let routes = route_weights.len();
        if routes == 0 {
            return Err(Error::Kernel("moe_block_batched_metal: no routes".into()));
        }
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "moe_block_batched_metal shape: x={} hidden={} out={}",
                x.len(),
                hidden,
                out.len()
            )));
        }

        validate_batched_quant(
            "moe_block_batched routed_gate_q4",
            routed_gate_q4,
            routes,
            routed_mid,
            hidden,
            256,
            144,
        )?;
        validate_batched_quant(
            "moe_block_batched routed_up_q4",
            routed_up_q4,
            routes,
            routed_mid,
            hidden,
            256,
            144,
        )?;
        validate_batched_quant(
            "moe_block_batched routed_down_q8",
            routed_down_q8,
            routes,
            hidden,
            routed_mid,
            32,
            34,
        )?;

        let has_shared =
            shared_gate_q4.is_some() || shared_up_q4.is_some() || shared_down_q6.is_some();
        if has_shared
            && !(shared_gate_q4.is_some() && shared_up_q4.is_some() && shared_down_q6.is_some())
        {
            return Err(Error::Kernel(
                "moe_block_batched_metal: shared tensors must be all Some or all None".into(),
            ));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let weights_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));

        let routed_gate_buf = ctx.new_buffer_with_bytes(routed_gate_q4);
        let routed_up_buf = ctx.new_buffer_with_bytes(routed_up_q4);
        let routed_down_buf = ctx.new_buffer_with_bytes(routed_down_q8);

        let routed_gate_out = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_up_out = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_act = ctx.new_buffer(routes * routed_mid * std::mem::size_of::<f32>());
        let routed_out = ctx.new_buffer(routes * hidden * std::mem::size_of::<f32>());
        let final_out = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        dispatch_batched_gemv(
            ctx,
            "moe_batched_gemm_q4",
            &routed_gate_buf,
            &x_buf,
            &routed_gate_out,
            routes,
            routed_mid,
            hidden,
        )?;
        dispatch_batched_gemv(
            ctx,
            "moe_batched_gemm_q4",
            &routed_up_buf,
            &x_buf,
            &routed_up_out,
            routes,
            routed_mid,
            hidden,
        )?;
        dispatch_silu_mul(
            ctx,
            &routed_gate_out,
            &routed_up_out,
            &routed_act,
            routes * routed_mid,
        )?;
        dispatch_batched_gemv(
            ctx,
            "moe_batched_gemm_q8_0",
            &routed_down_buf,
            &routed_act,
            &routed_out,
            routes,
            hidden,
            routed_mid,
        )?;

        let shared_out = if let (Some(gate), Some(up), Some(down)) =
            (shared_gate_q4, shared_up_q4, shared_down_q6)
        {
            validate_batched_quant(
                "moe_block_batched shared_gate_q4",
                gate,
                1,
                shared_mid,
                hidden,
                256,
                144,
            )?;
            validate_batched_quant(
                "moe_block_batched shared_up_q4",
                up,
                1,
                shared_mid,
                hidden,
                256,
                144,
            )?;
            validate_batched_quant(
                "moe_block_batched shared_down_q6",
                down,
                1,
                hidden,
                shared_mid,
                256,
                210,
            )?;

            let shared_gate_buf = ctx.new_buffer_with_bytes(gate);
            let shared_up_buf = ctx.new_buffer_with_bytes(up);
            let shared_down_buf = ctx.new_buffer_with_bytes(down);
            let shared_gate_out = ctx.new_buffer(shared_mid * std::mem::size_of::<f32>());
            let shared_up_out = ctx.new_buffer(shared_mid * std::mem::size_of::<f32>());
            let shared_act = ctx.new_buffer(shared_mid * std::mem::size_of::<f32>());
            let shared_out = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

            dispatch_batched_gemv(
                ctx,
                "moe_batched_gemm_q4",
                &shared_gate_buf,
                &x_buf,
                &shared_gate_out,
                1,
                shared_mid,
                hidden,
            )?;
            dispatch_batched_gemv(
                ctx,
                "moe_batched_gemm_q4",
                &shared_up_buf,
                &x_buf,
                &shared_up_out,
                1,
                shared_mid,
                hidden,
            )?;
            dispatch_silu_mul(
                ctx,
                &shared_gate_out,
                &shared_up_out,
                &shared_act,
                shared_mid,
            )?;
            dispatch_batched_gemv(
                ctx,
                "moe_batched_gemm_q6_k",
                &shared_down_buf,
                &shared_act,
                &shared_out,
                1,
                hidden,
                shared_mid,
            )?;
            shared_out
        } else {
            ctx.new_buffer(hidden * std::mem::size_of::<f32>())
        };

        dispatch_route_accumulate(
            ctx,
            &routed_out,
            &weights_buf,
            &shared_out,
            &final_out,
            hidden,
            routes,
            has_shared,
        )?;
        copy_f32_buffer(&final_out, out);
        Ok(())
    }

    /// Stage 1a of the strict single-launch fused MoE wedge.
    ///
    /// One-expert variant: gate / up / down all Q4_K, no top-K, no shared
    /// expert. The whole MoE block (gate matmul, up matmul, SwiGLU,
    /// down matmul) runs in ONE Metal grid via `moe_block_fused_q4_one`
    /// — workgroup-per-output-row, intermediate vector cached in
    /// threadgroup memory.
    ///
    /// Purpose: prove the single-launch design at parity vs the per-step
    /// reference path (`gemv_f32` × 3 + `silu_mul`). Stage 1b extends to
    /// top-K + Q8_0 down + Q6_K shared.
    ///
    /// Constraints:
    ///   - `hidden % 256 == 0` and `mid % 256 == 0` (Q4_K super-block).
    ///   - `gate_w_q4` / `up_w_q4` shaped `(mid, hidden)` Q4_K (
    ///     `mid * (hidden/256) * 144` bytes each).
    ///   - `down_w_q4` shaped `(hidden, mid)` Q4_K (
    ///     `hidden * (mid/256) * 144` bytes).
    ///   - Threadgroup memory budget: `mid` floats for the intermediate
    ///     plus 256 floats for the reduction. M3 Pro tg memory is 32 KB
    ///     so `mid <= ~7900` fits comfortably (DeepSeek-V2-Lite's
    ///     `moe_intermediate=1408` is well under the limit).
    pub fn moe_block_fused_q4_one_metal(
        ctx: &MetalContext,
        gate_w_q4: &[u8],
        up_w_q4: &[u8],
        down_w_q4: &[u8],
        hidden: usize,
        mid: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_one_metal shape: x={} hidden={} out={}",
                x.len(),
                hidden,
                out.len()
            )));
        }
        if hidden % 256 != 0 || mid % 256 != 0 {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_one_metal: hidden ({hidden}) and mid ({mid}) \
                 must be 256-aligned"
            )));
        }
        let hidden_blocks = hidden / 256;
        let mid_blocks = mid / 256;
        let expected_gate_up = mid * hidden_blocks * 144;
        let expected_down = hidden * mid_blocks * 144;
        if gate_w_q4.len() != expected_gate_up
            || up_w_q4.len() != expected_gate_up
            || down_w_q4.len() != expected_down
        {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_one_metal weight bytes: gate={} up={} down={} \
                 expected gate=up={} down={}",
                gate_w_q4.len(),
                up_w_q4.len(),
                down_w_q4.len(),
                expected_gate_up,
                expected_down
            )));
        }

        let gate_buf = ctx.new_buffer_with_bytes(gate_w_q4);
        let up_buf = ctx.new_buffer_with_bytes(up_w_q4);
        let down_buf = ctx.new_buffer_with_bytes(down_w_q4);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let hidden_u32 = hidden as u32;
        let mid_u32 = mid as u32;
        let intermed_bytes = (mid as u64) * std::mem::size_of::<f32>() as u64;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "moe_block_fused_q4_one",
            (hidden_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&gate_buf), 0);
                enc.set_buffer(1, Some(&up_buf), 0);
                enc.set_buffer(2, Some(&down_buf), 0);
                enc.set_buffer(3, Some(&x_buf), 0);
                enc.set_buffer(4, Some(&out_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &mid_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, intermed_bytes);
                enc.set_threadgroup_memory_length(1, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Stage 1b of the strict single-launch fused MoE wedge — TOP-K
    /// variant.
    ///
    /// Same workgroup-per-output-row design as Stage 1a, but the
    /// kernel iterates K experts per workgroup, accumulating their
    /// weighted contributions in-thread. Inputs follow the indexed
    /// no-pack convention: `gate_w_q4` / `up_w_q4` / `down_w_q4` are
    /// the FULL fused-expert tensors (`n_experts` slabs each), and
    /// `expert_ids[k]` selects the slab for the k-th iteration.
    ///
    /// Constraints (in addition to Stage 1a's):
    ///   - `n_experts >= max(expert_ids) + 1`
    ///   - `expert_ids.len() == route_weights.len() == top_k >= 1`
    ///   - `gate_w_q4` and `up_w_q4` are
    ///     `n_experts * mid * (hidden/256) * 144` bytes each
    ///   - `down_w_q4` is `n_experts * hidden * (mid/256) * 144` bytes
    #[allow(clippy::too_many_arguments)]
    pub fn moe_block_fused_q4_topk_metal(
        ctx: &MetalContext,
        gate_w_q4: &[u8],
        up_w_q4: &[u8],
        down_w_q4: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        n_experts: usize,
        hidden: usize,
        mid: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let top_k = expert_ids.len();
        if top_k == 0 || top_k != route_weights.len() {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_topk_metal: expert_ids.len={} route_weights.len={}",
                top_k,
                route_weights.len()
            )));
        }
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_topk_metal shape: x={} hidden={} out={}",
                x.len(),
                hidden,
                out.len()
            )));
        }
        if hidden % 256 != 0 || mid % 256 != 0 {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_topk_metal: hidden ({hidden}) and mid ({mid}) \
                 must be 256-aligned"
            )));
        }
        for &eid in expert_ids {
            if (eid as usize) >= n_experts {
                return Err(Error::Kernel(format!(
                    "moe_block_fused_q4_topk_metal: expert id {eid} >= n_experts {n_experts}"
                )));
            }
        }
        let hidden_blocks = hidden / 256;
        let mid_blocks = mid / 256;
        let expected_gate_up = n_experts * mid * hidden_blocks * 144;
        let expected_down = n_experts * hidden * mid_blocks * 144;
        if gate_w_q4.len() != expected_gate_up
            || up_w_q4.len() != expected_gate_up
            || down_w_q4.len() != expected_down
        {
            return Err(Error::Kernel(format!(
                "moe_block_fused_q4_topk_metal weight bytes: gate={} up={} down={} \
                 expected gate=up={} down={}",
                gate_w_q4.len(),
                up_w_q4.len(),
                down_w_q4.len(),
                expected_gate_up,
                expected_down
            )));
        }

        let gate_buf = ctx.new_buffer_with_bytes(gate_w_q4);
        let up_buf = ctx.new_buffer_with_bytes(up_w_q4);
        let down_buf = ctx.new_buffer_with_bytes(down_w_q4);
        let ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(expert_ids));
        let weights_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let hidden_u32 = hidden as u32;
        let mid_u32 = mid as u32;
        let top_k_u32 = top_k as u32;
        let intermed_bytes = (mid as u64) * std::mem::size_of::<f32>() as u64;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "moe_block_fused_q4_topk",
            (hidden_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&gate_buf), 0);
                enc.set_buffer(1, Some(&up_buf), 0);
                enc.set_buffer(2, Some(&down_buf), 0);
                enc.set_buffer(3, Some(&ids_buf), 0);
                enc.set_buffer(4, Some(&weights_buf), 0);
                enc.set_buffer(5, Some(&x_buf), 0);
                enc.set_buffer(6, Some(&out_buf), 0);
                enc.set_bytes(
                    7,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    8,
                    std::mem::size_of::<u32>() as u64,
                    &mid_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    9,
                    std::mem::size_of::<u32>() as u64,
                    &top_k_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, intermed_bytes);
                enc.set_threadgroup_memory_length(1, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Stage 1c — strict single-launch fused MoE block matching DeepSeek-V2-Lite's
    /// production layout: Q4_K gate/up + Q8_0 down for routed experts, Q4_K gate/up
    /// + Q6_K down for the always-on shared expert.
    ///
    /// Grid: `hidden` workgroups × TG_SIZE=256 threads. Each workgroup computes one
    /// output element, iterating K routed experts then the shared expert, reusing
    /// the same threadgroup `intermed` buffer across phases.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_block_fused_v2lite_metal(
        ctx: &MetalContext,
        routed_gate_q4: &[u8],
        routed_up_q4: &[u8],
        routed_down_q8: &[u8],
        shared_gate_q4: &[u8],
        shared_up_q4: &[u8],
        shared_down_q6: &[u8],
        expert_ids: &[u32],
        route_weights: &[f32],
        n_experts: usize,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let top_k = expert_ids.len();
        if top_k == 0 || top_k != route_weights.len() {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal: expert_ids.len={} route_weights.len={}",
                top_k,
                route_weights.len()
            )));
        }
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal shape: x={} hidden={} out={}",
                x.len(),
                hidden,
                out.len()
            )));
        }
        // Per-quant alignment: Q4_K cols (hidden) need 256-block,
        // Q8_0 down cols (routed_mid) need 32-block, Q6_K down cols
        // (shared_mid) need 256-block. The kernel itself iterates the
        // right block size per quant; the validation just guards the
        // host-side stride math.
        if hidden % 256 != 0 || routed_mid % 32 != 0 || shared_mid % 256 != 0 {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal: hidden ({hidden}) must be 256-aligned, \
                 routed_mid ({routed_mid}) must be 32-aligned (Q8_0 down), \
                 shared_mid ({shared_mid}) must be 256-aligned (Q6_K down)"
            )));
        }
        for &eid in expert_ids {
            if (eid as usize) >= n_experts {
                return Err(Error::Kernel(format!(
                    "moe_block_fused_v2lite_metal: expert id {eid} >= n_experts {n_experts}"
                )));
            }
        }
        let hidden_blocks = hidden / 256;
        let expected_routed_gate_up = n_experts * routed_mid * hidden_blocks * 144;
        let expected_routed_down = n_experts * hidden * (routed_mid / 32) * 34;
        let expected_shared_gate_up = shared_mid * hidden_blocks * 144;
        let expected_shared_down = hidden * (shared_mid / 256) * 210;
        if routed_gate_q4.len() != expected_routed_gate_up
            || routed_up_q4.len() != expected_routed_gate_up
        {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal routed gate/up bytes: gate={} up={} expected={}",
                routed_gate_q4.len(),
                routed_up_q4.len(),
                expected_routed_gate_up
            )));
        }
        if routed_down_q8.len() != expected_routed_down {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal routed down bytes: got={} expected={}",
                routed_down_q8.len(),
                expected_routed_down
            )));
        }
        if shared_gate_q4.len() != expected_shared_gate_up
            || shared_up_q4.len() != expected_shared_gate_up
        {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal shared gate/up bytes: gate={} up={} expected={}",
                shared_gate_q4.len(),
                shared_up_q4.len(),
                expected_shared_gate_up
            )));
        }
        if shared_down_q6.len() != expected_shared_down {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_metal shared down bytes: got={} expected={}",
                shared_down_q6.len(),
                expected_shared_down
            )));
        }

        let rg_buf = ctx.new_buffer_with_bytes(routed_gate_q4);
        let ru_buf = ctx.new_buffer_with_bytes(routed_up_q4);
        let rd_buf = ctx.new_buffer_with_bytes(routed_down_q8);
        let sg_buf = ctx.new_buffer_with_bytes(shared_gate_q4);
        let su_buf = ctx.new_buffer_with_bytes(shared_up_q4);
        let sd_buf = ctx.new_buffer_with_bytes(shared_down_q6);
        let ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(expert_ids));
        let wts_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let hidden_u32 = hidden as u32;
        let routed_mid_u32 = routed_mid as u32;
        let shared_mid_u32 = shared_mid as u32;
        let top_k_u32 = top_k as u32;
        let intermed_bytes =
            std::cmp::max(routed_mid, shared_mid) as u64 * std::mem::size_of::<f32>() as u64;
        let shmem_bytes = TG_SIZE as u64 * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "moe_block_fused_v2lite",
            (hidden_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&rg_buf), 0);
                enc.set_buffer(1, Some(&ru_buf), 0);
                enc.set_buffer(2, Some(&rd_buf), 0);
                enc.set_buffer(3, Some(&sg_buf), 0);
                enc.set_buffer(4, Some(&su_buf), 0);
                enc.set_buffer(5, Some(&sd_buf), 0);
                enc.set_buffer(6, Some(&ids_buf), 0);
                enc.set_buffer(7, Some(&wts_buf), 0);
                enc.set_buffer(8, Some(&x_buf), 0);
                enc.set_buffer(9, Some(&out_buf), 0);
                enc.set_bytes(
                    10,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    11,
                    std::mem::size_of::<u32>() as u64,
                    &routed_mid_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    12,
                    std::mem::size_of::<u32>() as u64,
                    &shared_mid_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    13,
                    std::mem::size_of::<u32>() as u64,
                    &top_k_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, intermed_bytes);
                enc.set_threadgroup_memory_length(1, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Stage B.4 — indexed production dispatcher for `moe_block_fused_v2lite`.
    /// Takes the whole GGUF mmap already on GPU (`model_buf`) plus per-tensor
    /// byte offsets, matching the no-copy indexed convention of the batched
    /// path. No buffer uploads per dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn moe_block_fused_v2lite_indexed_metal(
        ctx: &MetalContext,
        model_buf: &PinnedBuffer,
        routed_gate_offset: usize,
        routed_up_offset: usize,
        routed_down_offset: usize,
        shared_gate_offset: usize,
        shared_up_offset: usize,
        shared_down_offset: usize,
        expert_ids: &[u32],
        route_weights: &[f32],
        n_experts: usize,
        hidden: usize,
        routed_mid: usize,
        shared_mid: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        let top_k = expert_ids.len();
        if top_k == 0 || top_k != route_weights.len() {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_indexed_metal: expert_ids.len={} route_weights.len={}",
                top_k,
                route_weights.len()
            )));
        }
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_indexed_metal shape: x={} hidden={} out={}",
                x.len(),
                hidden,
                out.len()
            )));
        }
        // See moe_block_fused_v2lite_metal for the alignment rationale.
        if hidden % 256 != 0 || routed_mid % 32 != 0 || shared_mid % 256 != 0 {
            return Err(Error::Kernel(format!(
                "moe_block_fused_v2lite_indexed_metal: hidden ({hidden}) must be 256-aligned, \
                 routed_mid ({routed_mid}) must be 32-aligned (Q8_0 down), \
                 shared_mid ({shared_mid}) must be 256-aligned (Q6_K down)"
            )));
        }
        for &eid in expert_ids {
            if (eid as usize) >= n_experts {
                return Err(Error::Kernel(format!(
                    "moe_block_fused_v2lite_indexed_metal: expert id {eid} >= n_experts {n_experts}"
                )));
            }
        }

        let ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(expert_ids));
        let wts_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(hidden * std::mem::size_of::<f32>());

        let hidden_u32 = hidden as u32;
        let routed_mid_u32 = routed_mid as u32;
        let shared_mid_u32 = shared_mid as u32;
        let top_k_u32 = top_k as u32;
        let routed_gate_off = routed_gate_offset as u64;
        let routed_up_off = routed_up_offset as u64;
        let routed_down_off = routed_down_offset as u64;
        let shared_gate_off = shared_gate_offset as u64;
        let shared_up_off = shared_up_offset as u64;
        let shared_down_off = shared_down_offset as u64;
        let intermed_bytes =
            std::cmp::max(routed_mid, shared_mid) as u64 * std::mem::size_of::<f32>() as u64;
        let shmem_bytes = TG_SIZE as u64 * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            "moe_block_fused_v2lite_indexed",
            (hidden_u32 * TG_SIZE, 1, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(model_buf), 0);
                enc.set_buffer(1, Some(&ids_buf), 0);
                enc.set_buffer(2, Some(&wts_buf), 0);
                enc.set_buffer(3, Some(&x_buf), 0);
                enc.set_buffer(4, Some(&out_buf), 0);
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &hidden_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    6,
                    std::mem::size_of::<u32>() as u64,
                    &routed_mid_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    7,
                    std::mem::size_of::<u32>() as u64,
                    &shared_mid_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    8,
                    std::mem::size_of::<u32>() as u64,
                    &top_k_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    9,
                    std::mem::size_of::<u64>() as u64,
                    &routed_gate_off as *const u64 as *const _,
                );
                enc.set_bytes(
                    10,
                    std::mem::size_of::<u64>() as u64,
                    &routed_up_off as *const u64 as *const _,
                );
                enc.set_bytes(
                    11,
                    std::mem::size_of::<u64>() as u64,
                    &routed_down_off as *const u64 as *const _,
                );
                enc.set_bytes(
                    12,
                    std::mem::size_of::<u64>() as u64,
                    &shared_gate_off as *const u64 as *const _,
                );
                enc.set_bytes(
                    13,
                    std::mem::size_of::<u64>() as u64,
                    &shared_up_off as *const u64 as *const _,
                );
                enc.set_bytes(
                    14,
                    std::mem::size_of::<u64>() as u64,
                    &shared_down_off as *const u64 as *const _,
                );
                enc.set_threadgroup_memory_length(0, intermed_bytes);
                enc.set_threadgroup_memory_length(1, shmem_bytes);
            },
        )?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, hidden) };
        out.copy_from_slice(out_slice);

        Ok(())
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

        ctx.dispatch_batch(|batch| {
            encode_batched_gemv_indexed(
                batch,
                "moe_batched_gemm_q4_indexed",
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
                "moe_batched_gemm_q4_indexed",
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
                    "moe_batched_gemm_q4_indexed",
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
                    "moe_batched_gemm_q4_indexed",
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

    fn validate_batched_quant(
        name: &str,
        bytes: &[u8],
        routes: usize,
        rows: usize,
        cols: usize,
        block_elems: usize,
        block_bytes: usize,
    ) -> Result<()> {
        if routes == 0 {
            return Err(Error::Kernel(format!("{name}: routes must be > 0")));
        }
        if cols % block_elems != 0 {
            return Err(Error::Kernel(format!(
                "{name}: cols must be multiple of {block_elems}; got {cols}"
            )));
        }
        let expected = routes * rows * (cols / block_elems) * block_bytes;
        if bytes.len() != expected {
            return Err(Error::Kernel(format!(
                "{name}: got {} weight bytes expected {}",
                bytes.len(),
                expected
            )));
        }
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
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        batch.dispatch_threads(
            kernel_name,
            (rows_u32 * TG_SIZE, routes_u32, 1),
            (TG_SIZE, 1, 1),
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
                enc.set_threadgroup_memory_length(0, shmem_bytes);
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

    fn dispatch_batched_gemv(
        ctx: &MetalContext,
        kernel_name: &str,
        w_buf: &PinnedBuffer,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        routes: usize,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(
            kernel_name,
            (rows_u32 * TG_SIZE, routes_u32, 1),
            (TG_SIZE, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(out_buf), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &routes_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    4,
                    std::mem::size_of::<u32>() as u64,
                    &rows_u32 as *const u32 as *const _,
                );
                enc.set_bytes(
                    5,
                    std::mem::size_of::<u32>() as u64,
                    &cols_u32 as *const u32 as *const _,
                );
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            },
        )
    }

    fn dispatch_silu_mul(
        ctx: &MetalContext,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        let n_u32 = n as u32;
        ctx.dispatch_threads(
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

    fn dispatch_route_accumulate(
        ctx: &MetalContext,
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
        ctx.dispatch_threads(
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
