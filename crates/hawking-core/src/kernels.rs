use half::f16;

#[path = "kernels_megakernel.rs"]
pub mod megakernel;

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

/// GeGLU activation: `out = gelu_tanh(gate) * up`.
///
/// Uses the tanh approximation of GELU (`gelu_pytorch_tanh`), which is
/// what Gemma-2 was trained with:
///   gelu(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
pub fn gelu_mul(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // √(2/π)
    for i in 0..gate.len() {
        let x = gate[i];
        let inner = SQRT_2_OVER_PI * (x + 0.044715 * x * x * x);
        let g = 0.5 * x * (1.0 + inner.tanh());
        out[i] = g * up[i];
    }
}

/// Logit soft-capping: `xs[i] = cap · tanh(xs[i] / cap)` in place.
///
/// Gemma-2 caps both the attention scores (cap≈50) and the final logits
/// (cap≈30). `cap <= 0` is a no-op (capping disabled). Bounds the output
/// to (−cap, cap) while staying ~linear near 0.
pub fn logit_softcap_inplace(xs: &mut [f32], cap: f32) {
    if cap <= 0.0 {
        return;
    }
    let inv = 1.0 / cap;
    for v in xs.iter_mut() {
        *v = cap * (*v * inv).tanh();
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
/// schedule. Llama/Qwen/Gemma GGUF tensors use NEOX pairing: dimension
/// `i` rotates with dimension `i + head_dim/2`.
pub fn rope_inplace(x: &mut [f32], pos: u32, base: f32) {
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

/// Llama-3.1+ NTK-aware RoPE frequency-rescaling parameters.
///
/// Comes from GGUF metadata `llama.rope.scaling.{factor, low_freq_factor,
/// high_freq_factor, original_context_length}` when
/// `llama.rope.scaling.type == "llama3"`. Absent for Llama-3.0, Qwen2,
/// and DeepSeek-V2, in which case the unscaled [`rope_inplace`] path is
/// used.
///
/// Reference: llama.cpp `llama-model.cpp::llama_init_freqs_llama3`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Llama3RopeScaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: u32,
}

/// In-place RoPE with optional Llama-3.1+ NTK-aware frequency rescale.
/// When `scaling` is `None`, this is bit-identical to [`rope_inplace`].
///
/// Scaling rule per half-pair (interpreted as a wavelength gate):
/// - wavelen < high_wavelen  → freq unchanged   (high-freq pairs stay)
/// - wavelen > low_wavelen   → freq / factor     (long-context tail)
/// - in between              → smooth linear interpolation between the two
pub fn rope_inplace_scaled(x: &mut [f32], pos: u32, base: f32, scaling: Option<Llama3RopeScaling>) {
    let Some(s) = scaling else {
        // Delegate so the Qwen2 / DeepSeek-V2 unscaled paths stay
        // bit-identical to the baselines captured against `rope_inplace`.
        rope_inplace(x, pos, base);
        return;
    };
    let head_dim = x.len();
    let half = head_dim / 2;
    let two_pi = std::f32::consts::TAU;
    for i in 0..half {
        let inv_freq = base.powf(2.0 * i as f32 / head_dim as f32);
        let freq = 1.0 / inv_freq;
        let wavelen = two_pi / freq;
        let low_wavelen = s.original_max_position_embeddings as f32 / s.low_freq_factor;
        let high_wavelen = s.original_max_position_embeddings as f32 / s.high_freq_factor;
        let freq_eff = if wavelen < high_wavelen {
            freq
        } else if wavelen > low_wavelen {
            freq / s.factor
        } else {
            let smooth = (s.original_max_position_embeddings as f32 / wavelen - s.low_freq_factor) / (s.high_freq_factor - s.low_freq_factor);
            (1.0 - smooth) * (freq / s.factor) + smooth * freq
        };
        let theta = pos as f32 * freq_eff;
        let (sin, cos) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

/// Phi-3 "longrope" (su-scaled) RoPE, NEOX pairing.
///
/// Difference from [`rope_inplace`]:
///   - **Per-dimension frequency rescale + mscale**: each pair's inverse
///     frequency is divided by `ext_factors[i]` (the short_factor or
///     long_factor array Phi-3.5 ships as a GGUF tensor), and the
///     resulting cos/sin are scaled by `mscale` (the long-context
///     attention factor).
///
/// `ext_factors.len()` must equal `head_dim/2`. With all factors == 1.0
/// and `mscale == 1.0` this is plain NEOX RoPE.
pub fn rope_inplace_longrope(x: &mut [f32], pos: u32, base: f32, ext_factors: &[f32], mscale: f32) {
    let head_dim = x.len();
    let half = head_dim / 2;
    debug_assert_eq!(ext_factors.len(), half);
    for i in 0..half {
        let inv_freq = 1.0 / (ext_factors[i] * base.powf(2.0 * i as f32 / head_dim as f32));
        let theta = pos as f32 * inv_freq;
        let (sin, cos) = theta.sin_cos();
        let sin = sin * mscale;
        let cos = cos * mscale;
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

/// Phase 2 Wedge 2c -- apply RoPE to N rotation vectors at N positions in
/// one call. RoPE is element-wise per (vector, position); this helper
/// makes the multi-token call site obvious without changing the math.
///
/// `xs` is N rotation vectors (each of length head_dim, even). `positions`
/// is N positions (one per vector). `base` is the rope theta-base.
///
/// Equivalent to N sequential calls to `rope_inplace`. Bit-identical.
pub fn rope_inplace_batch(xs: &mut [&mut [f32]], positions: &[u32], base: f32) {
    debug_assert_eq!(xs.len(), positions.len(), "rope_inplace_batch: xs.len()={} positions.len()={}", xs.len(), positions.len(),);
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
pub fn gather_combine(expert_out: &[f32], weights: &[f32], n_tokens: usize, top_k: usize, hidden: usize, token_out: &mut [f32]) {
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
///   expert_ids_out: (n_tokens, top_k) -- selected expert indices
///   weights_out: (n_tokens, top_k) -- softmax probs of those experts
pub fn topk_softmax_batch(logits: &[f32], n_tokens: usize, n_experts: usize, top_k: usize, expert_ids_out: &mut [u32], weights_out: &mut [f32]) {
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

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
mod metal_dispatch {
    use crate::metal::{ArgLayout, CommandBatch, DecodeArena, KernelArgBuffer, MetalContext, PinnedBuffer, TokenCommandBuffer};
    use crate::{Error, Result};
    use half::f16;

    /// Extension trait that collapses the verbose
    /// `set_bytes(i, size_of::<T>() as u64, &v as *const T as *const _)`
    /// scalar-binding idiom to a single call. `#[inline(always)]` makes this
    /// identical codegen to the inline form -- a pure LOC/readability
    /// consolidation with no behavioral or performance change.
    trait SetScalar {
        fn set_u32(&self, index: u64, value: u32);
        fn set_f32(&self, index: u64, value: f32);
    }
    impl SetScalar for ::metal::ComputeCommandEncoderRef {
        #[inline(always)]
        fn set_u32(&self, index: u64, value: u32) {
            self.set_bytes(index, std::mem::size_of::<u32>() as u64, &value as *const u32 as *const _);
        }
        #[inline(always)]
        fn set_f32(&self, index: u64, value: f32) {
            self.set_bytes(index, std::mem::size_of::<f32>() as u64, &value as *const f32 as *const _);
        }
    }

    // Reduction kernels in this module are written for tg_size=256 (the
    // shader's stride>>=1 pairwise reduction requires a power of two).
    const TG_SIZE: u32 = 256;

    /// Matches `struct ArgbufRowsCols { uint rows; uint cols; }` in
    /// `shaders/common.metal`. Several kernels (gemv_f32_attn, gemv_f32_moe,
    /// gemm_q4_k_m_fused_v2, gemm_q3_k_fused_v2) read this packed struct from
    /// buffer 3. Dispatchers must send 8 bytes via a single `set_bytes(3, ...)`
    /// — not two separate `set_bytes(3, u32)` + `set_bytes(4, u32)` calls,
    /// which leaves `args.cols` undefined in the synthetic test setup.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct ArgbufRowsCols {
        rows: u32,
        cols: u32,
    }

    /// Matches `ArgbufQkvRopeAppend` in `shaders/quant.metal`.
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct ArgbufQkvRopeAppend {
        q_rows: u32,
        kv_rows: u32,
        cols: u32,
        n_q_heads: u32,
        n_k_heads: u32,
        head_dim: u32,
        pos: u32,
        kv_off: u32,
        has_q_bias: u32,
        has_k_bias: u32,
        has_v_bias: u32,
        base: f32,
    }

    /// Q4_K_M-weight × fp32-vec → fp32 GEMV, dispatching the
    /// dense-path `gemm_q4_k_m_fused` kernel in `shaders/quant.metal`.
    /// Wedge 2 / H2.4 -- dequant is fused inside the FMA loop in
    /// threadgroup memory; weights stay 4-bit in DRAM.
    pub fn gemv_q4_k_m(ctx: &MetalContext, w_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_gemv(ctx, "gemm_q4_k_m_fused", w_bytes, rows, cols, x, out)
    }

    /// v0.3.0 -- simdgroup_matrix variant of gemv_q4_k_m.  Dispatches
    /// `gemm_q4_k_m_fused_simd`; selected via kernel-profile
    /// `gemm_q4_k_schedule = "simdgroup"`.
    pub fn gemv_q4_k_m_simd(ctx: &MetalContext, w_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd weight bytes: got {} expected {}", w_bytes.len(), expected_bytes)));
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

        ctx.dispatch_threads("gemm_q4_k_m_fused_simd", (n_tg * SIMD_TG, 1, 1), (SIMD_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    /// v0.4.0 -- multi-row TG + simd_sum variant.  Dispatches
    /// `gemm_q4_k_m_fused_v2`; selected via kernel-profile
    /// `gemm_q4_k_schedule = "v2"`.
    pub fn gemv_q4_k_m_v2(ctx: &MetalContext, w_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_gemv_v2(ctx, "gemm_q4_k_m_fused_v2", w_bytes, rows, cols, x, out)
    }

    /// Wedge A -- pinned-buffer variant of `gemv_q4_k_m_v2`. Reads Q4_K_M weights
    /// directly from `model_buf` at `w_offset` bytes, skipping the per-call
    /// `new_buffer_with_bytes` memcpy (1.6–11 MB per expert).
    pub fn gemv_q4_k_m_v2_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_gemv_v2_pinned(ctx, "gemm_q4_k_m_fused_v2", model_buf, w_offset, w_byte_size, rows, cols, x, out)
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
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        let end = w_offset.checked_add(w_byte_size).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let x_bytes = cols * std::mem::size_of::<f32>();
        let out_bytes = rows * std::mem::size_of::<f32>();
        if x_buf.length() < x_bytes as u64 || out_buf.length() < out_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} expected>={x_bytes} out={} expected>={out_bytes}", x_buf.length(), out_buf.length())));
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    /// P3 v3w — Batched Q4_K_M GEMM widened to B in 1..=8. Same shmem
    /// staging as v3 but with two float4 partial accumulators so a
    /// single dispatch can amortize one weight read across 8 tokens.
    /// Shmem tile is B*256 floats (8 KB at B=8).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb supports batch in 1..=8; got {batch}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} need={} y={} need={}", x_batch_buf.length(), x_bytes, y_batch_buf.length(), y_bytes,)));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        const V3_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        let shmem_bytes = (batch * 256 * std::mem::size_of::<f32>()) as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        ab.set_u32(2, batch_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_batch_buf), 0);
            enc.set_buffer(2, Some(y_batch_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Batched Q4_K GEMM with PRE-DECODED sub-block scales — same as
    /// `gemm_q4_k_m_batched_v3w_pinned_tcb` but reads `ds/dm` from a predec
    /// scale table (built via `predecode_q4_k_scale_table`) instead of decoding
    /// the Q4_K header per element. Brings the single-path predec win to the
    /// batched decode/verify path. `scales_buf` holds `rows*blocks_per_row*16`
    /// f32 (16 floats/block); `scales_offset` is its byte offset (usually 0,
    /// one buffer per tensor).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_predec_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 1..=8 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let scales_need = (scales_offset + rows * blocks_per_row * 16 * std::mem::size_of::<f32>()) as u64;
        if scales_buf.length() < scales_need {
            return Err(Error::Kernel(format!("{KERNEL}: scales buf {} < need {}", scales_buf.length(), scales_need)));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}: x/y buffer too small")));
        }
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows as u32);
        ab.set_u32(1, cols as u32);
        ab.set_u32(2, batch as u32);
        const V3_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        let shmem_bytes = (batch * 256 * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_batch_buf), 0);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// `gemm_q4_k_m_batched_v3w_predec_pinned_tcb` with a BYTE OFFSET into the
    /// x_batch buffer, so a slot-major activation block (e.g. the RWKV-7 `xs`
    /// lerp buffer laid out `(slot, B, n)`) can feed the batched GEMM without a
    /// copy: the kernel reads its `(B, cols)` input starting at `x_off_bytes`.
    /// Identical dispatch/geometry to the base function; only the x binding is
    /// offset (the same trick `gemv_q4_k_v4_predec_xoff_pinned_tcb` uses for the
    /// single-stream slot-major path). `x_off_bytes` must keep the whole
    /// `batch*cols*4` window in-bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_predec_xoff_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        x_off_bytes: usize,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 1..=8 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let scales_need = (scales_offset + rows * blocks_per_row * 16 * std::mem::size_of::<f32>()) as u64;
        if scales_buf.length() < scales_need {
            return Err(Error::Kernel(format!("{KERNEL}: scales buf {} < need {}", scales_buf.length(), scales_need)));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < (x_off_bytes + x_bytes) as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}: x/y buffer too small")));
        }
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows as u32);
        ab.set_u32(1, cols as u32);
        ab.set_u32(2, batch as u32);
        const V3_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        let shmem_bytes = (batch * 256 * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_batch_buf), x_off_bytes as u64);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// B=1..16 extension of `gemm_q4_k_m_batched_v3w_predec_pinned_tcb`.
    ///
    /// Uses two additional float4 accumulators (partial_lo2, partial_hi2) for
    /// slots 8..15. Threadgroup shmem scales to B*256*4 bytes (16 KiB at B=16,
    /// within the M3 Pro 32 KiB limit). All other dispatch geometry is identical
    /// to the original v3w_predec kernel. Intentionally dead code until
    /// MAX_MULTISEQ_SLOTS is raised beyond 8.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_predec_b16_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w_predec_b16";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(1..=16).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 1..=16 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let scales_need = (scales_offset + rows * blocks_per_row * 16 * std::mem::size_of::<f32>()) as u64;
        if scales_buf.length() < scales_need {
            return Err(Error::Kernel(format!("{KERNEL}: scales buf {} < need {}", scales_buf.length(), scales_need)));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}: x/y buffer too small")));
        }
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows as u32);
        ab.set_u32(1, cols as u32);
        ab.set_u32(2, batch as u32);
        const V3_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        let shmem_bytes = (batch * 256 * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_batch_buf), 0);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Barrier-free drop-in replacement for `gemm_q4_k_m_batched_v3w_predec_pinned_tcb`.
    ///
    /// The v3w_predec kernel stages B activation vectors in threadgroup shmem
    /// behind two barriers per block (16 barriers/projection for hidden=2048).
    /// This kernel reads x directly from device memory — zero shmem, zero
    /// barriers — and processes 16 rows/TG via 2-row ILP (was 8 rows/TG).
    ///
    /// Same buffer layout and I/O contract as v3w_predec. Bit-identical
    /// output when reduction order is preserved (validated by parity test).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v4r_predec_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v4r_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(2..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 2..=8 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let scales_need = (scales_offset + rows * blocks_per_row * 16 * std::mem::size_of::<f32>()) as u64;
        if scales_buf.length() < scales_need {
            return Err(Error::Kernel(format!("{KERNEL}: scales buf {} < need {}", scales_buf.length(), scales_need)));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}: x/y buffer too small")));
        }
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows as u32);
        ab.set_u32(1, cols as u32);
        ab.set_u32(2, batch as u32);
        const V4R_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 16;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        // No threadgroup memory — barrier-free kernel.
        tcb.dispatch_threads(KERNEL, (n_tg * V4R_TG, 1, 1), (V4R_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_batch_buf), 0);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// P1 — simdgroup-matrix (MMA) twin of `gemm_q4_k_m_batched_v3w_pinned_tcb`.
    /// Same Q4_K dequant→threadgroup staging + identical I/O contract, but the
    /// inner product runs on hardware `simdgroup_matrix<float,8,8>` tiles.
    /// Geometry differs: ONE simdgroup (32 threads) per threadgroup, 8 rows/TG,
    /// fixed 576-f32 shmem (independent of batch). MMA wins on rows>cols (ffn
    /// gate/up); the caller shape-gates it (silicon #8 /
    /// plans/prefill_mma_build_plan.md).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_mma_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w_mma";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb supports batch in 1..=8; got {batch}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} need={} y={} need={}", x_batch_buf.length(), x_bytes, y_batch_buf.length(), y_bytes,)));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        // MMA geometry: one simdgroup (32 threads) per TG, 8 rows/TG; shmem is
        // fixed 576 f32 (Ws[256]+Xs[256]+Os[64]), NOT batch*256.
        const MMA_TG: u32 = 32;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        let shmem_bytes = (576 * std::mem::size_of::<f32>()) as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        ab.set_u32(2, batch_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * MMA_TG, 1, 1), (MMA_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_batch_buf), 0);
            enc.set_buffer(2, Some(y_batch_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Track 3.5 — SwiGLU-fused ffn_down via v3w_predec (B=5..8).
    /// Replaces the (ffn_gate GEMM + ffn_up GEMM + silu_mul + ffn_down GEMM)
    /// sequence with (ffn_gate GEMM + ffn_up GEMM + ffn_down_swiglu GEMM).
    /// Saves 1 dispatch/layer by inlining silu(gate)*up into x_tile loading.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_predec_swiglu_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        gate_batch_buf: &PinnedBuffer, // (batch, cols) f32 gate activations
        up_batch_buf: &PinnedBuffer,   // (batch, cols) f32 up activations
        y_batch_buf: &PinnedBuffer,    // (batch, rows) f32 ffn_down output
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w_predec_swiglu";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 1..=8 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        ab.set_u32(2, batch_u32);
        const V3_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        let shmem_bytes = (batch * 256 * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(gate_batch_buf), 0);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_buffer(5, Some(up_batch_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Track 3.5 — SwiGLU-fused ffn_down via v4r_predec (B=2..4).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v4r_predec_swiglu_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        gate_batch_buf: &PinnedBuffer,
        up_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v4r_predec_swiglu";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(2..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 2..=8 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        ab.set_u32(2, batch_u32);
        const V4_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 16;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * V4_TG, 1, 1), (V4_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(gate_batch_buf), 0);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_buffer(5, Some(up_batch_buf), 0);
        })
    }

    /// Track 3.15-style FFN tail helper for Q4_K predec batched paths.
    ///
    /// This intentionally composes two ordered dispatches:
    ///
    /// 1. `ffn_down = W_down * (silu(gate) * up)` via the existing Q4_K
    ///    predec SwiGLU down kernels.
    /// 2. `x += ffn_down; x_norm = rmsnorm(x, norm_weight)` via the existing
    ///    batched add+rmsnorm kernel.
    ///
    /// A single Metal dispatch is not a safe small extension here: the current
    /// down GEMV is row-parallel across threadgroups, while RMSNorm needs a
    /// hidden-wide reduction after every row has been written.
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_down_swiglu_add_rmsnorm_ffn_q4k_predec_batched_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        gate_batch_buf: &PinnedBuffer,
        up_batch_buf: &PinnedBuffer,
        residual_x_batch_buf: &PinnedBuffer,
        norm_weight_buf: &PinnedBuffer,
        x_norm_batch_buf: &PinnedBuffer,
        eps: f32,
        ffn_down_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if batch > 8 {
            return Err(Error::Kernel(format!("ffn_down_swiglu_add_rmsnorm_ffn_q4k_predec_batched_tcb supports batch in 1..=8; got {batch}")));
        }

        if batch == 1 {
            gemv_q4_k_v4_predec_swiglu_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_buf, scales_offset, rows, cols, gate_batch_buf, up_batch_buf, ffn_down_batch_buf)?;
        } else if batch <= 4 || crate::env_on("HAWKING_QWEN_MULTISEQ_V4R_HIGHB") {
            // Wave-R0: B=5..8 fused FFN-down SwiGLU also routes through the
            // barrier-free v4r_predec swiglu (p0_hi/p1_hi cover slots 4-7) when the
            // multiseq-v4r-highB flag is set; gated by the same clean aggregate bench.
            gemm_q4_k_m_batched_v4r_predec_swiglu_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_buf, scales_offset, rows, cols, batch, gate_batch_buf, up_batch_buf, ffn_down_batch_buf)?;
        } else {
            gemm_q4_k_m_batched_v3w_predec_swiglu_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_buf, scales_offset, rows, cols, batch, gate_batch_buf, up_batch_buf, ffn_down_batch_buf)?;
        }

        add_rmsnorm_fused_batched_tcb(tcb, residual_x_batch_buf, ffn_down_batch_buf, norm_weight_buf, x_norm_batch_buf, eps, rows, batch)
    }

    /// P1 — predec MMA twin: `gemm_q4_k_m_batched_v3w_predec_pinned_tcb` with the
    /// simdgroup-matrix inner product. Reads pre-decoded (ds,dm) scales at buffer
    /// slot 1 (x/y/args shift to 2/3/4). Same MMA geometry (32 threads/TG, 8
    /// rows/TG, 576-f32 shmem). This is the twin that moves shipped prefill,
    /// since the batched path is predec-default-ON (Option B in the build plan).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3w_mma_predec_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3w_mma_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0 ({cols})")));
        }
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}: batch must be 1..=8 ({batch})")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: w bytes {w_byte_size} != {expected_bytes}")));
        }
        let scales_need = (scales_offset + rows * blocks_per_row * 16 * std::mem::size_of::<f32>()) as u64;
        if scales_buf.length() < scales_need {
            return Err(Error::Kernel(format!("{KERNEL}: scales buf {} < need {}", scales_buf.length(), scales_need)));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}: x/y buffer too small")));
        }
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows as u32);
        ab.set_u32(1, cols as u32);
        ab.set_u32(2, batch as u32);
        const MMA_TG: u32 = 32;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows as u32).div_ceil(ROWS_PER_TG);
        let shmem_bytes = (576 * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (n_tg * MMA_TG, 1, 1), (MMA_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_batch_buf), 0);
            enc.set_buffer(3, Some(y_batch_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// P3 v3 — Batched Q4_K_M GEMM with cooperative shmem activation
    /// staging. Same args + layout as v2, but adds a 4 KB threadgroup
    /// tile so all 8 rows in a TG read the activation block from shmem
    /// (single-cycle L1) instead of B separate DRAM loads per thread.
    /// Fixes the cols-large performance cliff (ffn_down 2048×11008
    /// where v2 = sequential GEMV in the microbench).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v3_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v3";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        if !(1..=4).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb supports batch in 1..=4; got {batch}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} need={} y={} need={}", x_batch_buf.length(), x_bytes, y_batch_buf.length(), y_bytes,)));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        const V3_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        // shmem: B × 256 floats. At B=4 → 4 KiB.
        let shmem_bytes = (batch * 256 * std::mem::size_of::<f32>()) as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        ab.set_u32(2, batch_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_batch_buf), 0);
            enc.set_buffer(2, Some(y_batch_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// P3 — Batched Q4_K_M GEMM: one weight applied to B activation
    /// vectors in parallel. Reads the weight matrix once and produces B
    /// output rows worth of dot products per row. Bandwidth amortized
    /// near-linearly across B until compute-bound. Supported B: 1..=4.
    ///
    /// Layouts:
    ///   `x_batch`: (B, cols) f32, row-major
    ///   `y_batch`: (B, rows) f32, row-major
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_m_batched_v2_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        batch: usize,
        x_batch_buf: &PinnedBuffer,
        y_batch_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_batched_v2";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        if !(1..=4).contains(&batch) {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb supports batch in 1..=4; got {batch}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let x_bytes = batch * cols * std::mem::size_of::<f32>();
        let y_bytes = batch * rows * std::mem::size_of::<f32>();
        if x_batch_buf.length() < x_bytes as u64 || y_batch_buf.length() < y_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} need={} y={} need={}", x_batch_buf.length(), x_bytes, y_batch_buf.length(), y_bytes,)));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let batch_u32 = batch as u32;
        const V2_TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        ab.set_u32(2, batch_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_batch_buf), 0);
            enc.set_buffer(2, Some(y_batch_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    /// P2 — Wedge K Q4_K GEMV (scale + activation preload, paired-nibble
    /// reads). TCB-encoded variant of `gemv_q4_k_m_simdmat_pinned`.
    /// Geometry: 128 threads/TG, 4 rows/TG. Per the kernel comment, this
    /// improves small-row shapes (e.g. Qwen attn k/v_proj rows=256) over
    /// the v2 (8 rows/TG) baseline. Same buffer layout as v2.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_simdmat_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_simdmat";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const SM_TG: u32 = 128;
        const SM_ROWS: u32 = 4;
        let n_tg = rows_u32.div_ceil(SM_ROWS);
        tcb.dispatch_threads(KERNEL, (n_tg * SM_TG, 1, 1), (SM_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })
    }

    /// P2 — Wedge K-pattern Q4_K GEMV in v3 8-rows-per-TG geometry.
    /// Same scale/activation preload + paired-nibble reads as simdmat
    /// but with 8 rows/TG (256 threads, 8 simdgroups) → fewer TGs;
    /// candidate for larger-row shapes like Qwen FFN gate/up.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_v3_8r_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_8r";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = rows_u32.div_ceil(V3_ROWS);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })
    }

    /// Pre-decode the 8 sub-block (scale, min) f32 pairs for every Q4_K block
    /// in `w_q4_bytes` into a flat host-side `Vec<f32>`.
    ///
    /// Layout per block (16 f32 = 64 bytes):
    ///   `out[block_idx*16 + sub*2 + 0]` = ds[sub] = (f32)d    * (f32)sb[sub]
    ///   `out[block_idx*16 + sub*2 + 1]` = dm[sub] = (f32)dmin * (f32)mb[sub]
    /// for `sub` in 0..8, where `d` / `dmin` are the f16 block header and
    /// `sb` / `mb` are the 6-bit sub-block scale/min indices decoded from
    /// bytes 4..16 of each 144-byte Q4_K block.
    ///
    /// Math is exactly equivalent to the inline decode in `gemm_q4_k_m_v3_8r`:
    /// `gemm_q4_k_v4_predec` reads (ds, dm) from this table at the same
    /// fp16→f32 / uchar→f32 widening order, so the output is bit-identical.
    ///
    /// Intended use: call once at load time when pinning a Q4_K weight tensor
    /// and upload the result as a `PinnedBuffer` alongside the existing Q4_K
    /// weight buffer. Total table size = `(w_q4_bytes.len() / 144) * 64`
    /// bytes (= 0.444× the Q4_K weight size).
    pub fn predecode_q4_k_scale_table(w_q4_bytes: &[u8]) -> Vec<f32> {
        debug_assert_eq!(w_q4_bytes.len() % 144, 0, "predecode_q4_k_scale_table: byte len {} not a multiple of 144", w_q4_bytes.len());
        let n_blocks = w_q4_bytes.len() / 144;
        let mut out = vec![0.0f32; n_blocks * 16];
        for b in 0..n_blocks {
            let bo = b * 144;
            // Block header: d (f16 LE), dmin (f16 LE).
            let d_bits = u16::from_le_bytes([w_q4_bytes[bo], w_q4_bytes[bo + 1]]);
            let dmin_bits = u16::from_le_bytes([w_q4_bytes[bo + 2], w_q4_bytes[bo + 3]]);
            let d = half::f16::from_bits(d_bits).to_f32();
            let dmin = half::f16::from_bits(dmin_bits).to_f32();
            // 6-bit sub-block indices: same unpack as the shader.
            //   sub 0..4: low 6 bits of bytes [4+sub] / [8+sub]
            //   sub 4..8: low 4 bits of bytes [12+j]  | (high-2-bits of [4+j]/[8+j] << 4)
            let mut sb = [0u8; 8];
            let mut mb = [0u8; 8];
            for sub in 0..4 {
                sb[sub] = w_q4_bytes[bo + 4 + sub] & 0x3F;
                mb[sub] = w_q4_bytes[bo + 8 + sub] & 0x3F;
            }
            for j in 0..4 {
                let b12 = w_q4_bytes[bo + 12 + j];
                let b4 = w_q4_bytes[bo + 4 + j];
                let b8 = w_q4_bytes[bo + 8 + j];
                sb[4 + j] = (b12 & 0x0F) | ((b4 >> 6) << 4);
                mb[4 + j] = (b12 >> 4) | ((b8 >> 6) << 4);
            }
            let so = b * 16;
            for sub in 0..8 {
                out[so + sub * 2] = d * (sb[sub] as f32);
                out[so + sub * 2 + 1] = dmin * (mb[sub] as f32);
            }
        }
        out
    }

    /// f16 variant of [`predecode_q4_k_scale_table`] (worklist 1.2). Stores the
    /// pre-decoded (ds, dm) pairs as `half::f16` (32 B/block vs the f32 table's
    /// 64 B), cutting the dominant-GEMV predec scale bandwidth ~17% (160 vs
    /// 192 B/block effective). The matching kernel reads `half` and widens to
    /// float in-register. Parity is atol-1e-3 fp16 (scale rounding), NOT
    /// bit-identical — gate via a Rust atol parity test, not the greedy gate.
    pub fn predecode_q4_k_scale_table_f16(w_q4_bytes: &[u8]) -> Vec<half::f16> {
        predecode_q4_k_scale_table(w_q4_bytes).into_iter().map(half::f16::from_f32).collect()
    }

    /// Q4_K decode GEMV with pre-decoded sub-block scales (v4_predec).
    ///
    /// Identical math to `gemv_q4_k_m_v3_8r_pinned_tcb` (same v3_8r geometry:
    /// 256 threads/TG, 8 simdgroups, 8 rows/TG) but reads the 8 sub-block
    /// (ds, dm) f32 pairs per block from a parallel pre-decoded table
    /// (`scales_buf`) instead of decoding them inline from the packed 6-bit
    /// indices every call.
    ///
    /// Build the table once at load time via `predecode_q4_k_scale_table`
    /// and pin it as a `PinnedBuffer`. Expected `scales_buf` length is
    /// `rows * (cols / 256) * 16 * sizeof(f32)`.
    ///
    /// **Private API entry point** — not yet wired into the production
    /// forward pass; that's the consolidation step.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb scales oob: {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        // path-to-50: 2-rows-per-simdgroup variant (2 accumulator chains, shared
        // x load) for better DRAM-latency hiding. Default-on (paired bench +6.2%
        // bit-identical vs 1-row predec, 2026-05-30); opt out HAWKING_QWEN_PREDEC_2R=0.
        let use_2r = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| std::env::var_os("HAWKING_QWEN_PREDEC_2R").map(|v| v != "0").unwrap_or(true))
        };
        // Stage 2: 4-rows-per-simdgroup variant (4 accumulator chains). Opt-in,
        // takes precedence over 2r when set. Bit-identical (same per-row math).
        let use_4r = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| std::env::var_os("HAWKING_QWEN_PREDEC_4R").map(|v| v != "0").unwrap_or(false))
        };
        let (dispatch_kernel, rows_per_tg): (&str, u32) = if use_4r {
            ("gemm_q4_k_v4_predec_4r", 32)
        } else if use_2r {
            ("gemm_q4_k_v4_predec_2r", 16)
        } else {
            (KERNEL, 8)
        };
        let n_tg = rows_u32.div_ceil(rows_per_tg);
        tcb.dispatch_threads(dispatch_kernel, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Q4_K predec GEMV reading the activation vector at a byte OFFSET into
    /// `x_buf` (everything else identical to [`gemv_q4_k_v4_predec_pinned_tcb`]).
    ///
    /// The predec kernels index `x[b*256 + k*32 + lane]` from the bound buffer
    /// base, so binding `x_buf` at `x_off_bytes` makes the kernel consume the
    /// `cols`-wide slice starting there with no copy. This lets a slot-major
    /// activation block (e.g. the RWKV-7 time-mix `xs` buffer, where slot `s`
    /// lives at `s*cols` floats) feed each projection directly. Same 2r/4r
    /// auto-selection and the same bit-identical per-row MAC.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_xoff_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_xoff_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_xoff_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_xoff_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_xoff_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_xoff_pinned_tcb scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_xoff_pinned_tcb scales oob: {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        // Match the default-on 2r / opt-in 4r selection of the non-offset path
        // (bit-identical per-row math; geometry only).
        let use_2r = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| std::env::var_os("HAWKING_QWEN_PREDEC_2R").map(|v| v != "0").unwrap_or(true))
        };
        let use_4r = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| std::env::var_os("HAWKING_QWEN_PREDEC_4R").map(|v| v != "0").unwrap_or(false))
        };
        let (dispatch_kernel, rows_per_tg): (&str, u32) = if use_4r {
            ("gemm_q4_k_v4_predec_4r", 32)
        } else if use_2r {
            ("gemm_q4_k_v4_predec_2r", 16)
        } else {
            (KERNEL, 8)
        };
        let n_tg = rows_u32.div_ceil(rows_per_tg);
        tcb.dispatch_threads(dispatch_kernel, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), x_off_bytes as u64);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track 3.14 tail fusion: Q4_K predec GEMV plus residual add.
    ///
    /// Dispatches `gemm_q4_k_v4_predec_2r_add`, which is the same 2-row-ILP
    /// predec GEMV math as `gemm_q4_k_v4_predec_2r` but writes
    /// `residual[row] += dot(row, x)` instead of materializing a temporary
    /// output vector. Intended for the Qwen B=1 `o_proj -> residual` tail.
    ///
    /// The RMSNorm that follows still needs its own dispatch because it reduces
    /// across the fully updated residual vector.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_2r_add_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        residual_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_2r_add";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb scales oob: {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let input_bytes = cols.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb input overflow")))?;
        if x_buf.length() < input_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb x buf too small: got {} need {input_bytes}", x_buf.length())));
        }
        let residual_bytes = rows.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb residual overflow")))?;
        if residual_buf.length() < residual_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb residual buf too small: got {} need {residual_bytes}", residual_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 16;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(residual_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track B4 — 4-row-per-simdgroup variant of `gemv_q4_k_v4_predec_2r_add`.
    /// Opt-in via `HAWKING_QWEN_OPROJ_4R=1`.  Inline scale reads, 32 rows/TG,
    /// 4 FMA chains → higher ILP vs 2r (preloaded, 16 rows/TG, 2 chains).
    /// Bit-identical to 2r_add (same per-accumulator FMA order).
    ///
    /// Grid: `(ceil(rows/32) × 256, 1, 1)`  TG: `(256, 1, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_4r_add_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        residual_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_4r_add";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: oob {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}: scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: scales oob {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 32; // 8 simdgroups × 4 rows each
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(residual_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track D6 — f16-scales variant of `gemv_q4_k_v4_predec_2r_add`.
    /// Same 2-row in-place residual-add geometry as `2r_add` but reads scales
    /// as half* (2 B each). Enables oproj_add_rmsnorm_fuse when f16s active.
    ///
    /// Grid: `(ceil(rows/16) × 256, 1, 1)`  TG: `(256, 1, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_2r_add_f16s_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        residual_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_2r_add_f16s";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: oob {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 16;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(residual_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track D6 — f16-scales + 4r geometry variant of `gemv_q4_k_v4_predec_4r_add`.
    /// Combines inline half→float scale casts (D6) with 32 rows/TG (half the TG
    /// count of 2r_add_f16s). Active when PREDEC_F16SCALES=1 AND PREDEC_4R/OPROJ_4R.
    ///
    /// Grid: `(ceil(rows/32) × 256, 1, 1)`  TG: `(256, 1, 1)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_4r_add_f16s_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        residual_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_4r_add_f16s";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: oob {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 32;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(residual_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track 3.14 o_proj tail helper: Q4_K predec 2r GEMV adds directly into
    /// `residual_buf`, then `rmsnorm_f32` writes `x_norm_buf` from the updated
    /// residual. Equivalent to `gemv_q4_k_v4_predec_pinned_tcb` into a temp
    /// followed by `add_rmsnorm_fused_tcb`, for the f32-predec 2r path.
    /// When `HAWKING_QWEN_OPROJ_4R=1`, uses the 4r_add variant for better ILP.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_2r_add_rmsnorm_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        residual_buf: &PinnedBuffer,
        norm_weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        eps: f32,
    ) -> Result<()> {
        // Track B4: opt-in via HAWKING_QWEN_OPROJ_4R=1
        static USE_4R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let use_4r = *USE_4R.get_or_init(|| std::env::var_os("HAWKING_QWEN_OPROJ_4R").map(|v| v != "0").unwrap_or(false));
        if use_4r {
            gemv_q4_k_v4_predec_4r_add_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_buf, scales_offset, rows, cols, x_buf, residual_buf)?;
        } else {
            gemv_q4_k_v4_predec_2r_add_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_buf, scales_offset, rows, cols, x_buf, residual_buf)?;
        }
        rmsnorm_metal_buf_tcb(tcb, residual_buf, norm_weight_buf, eps, rows, x_norm_buf)
    }

    /// Track D6 — f16-scales variant of `gemv_q4_k_v4_predec_2r_add_rmsnorm_tcb`.
    /// Uses half* scale reads (2 B each) to halve scale bandwidth vs the f32 path.
    /// When `HAWKING_QWEN_PREDEC_4R=1`, uses `4r_add_f16s` (32 rows/TG) for
    /// additional TG-scheduling savings. Enables oproj_add_rmsnorm_fuse in fast
    /// profile (`PREDEC_F16SCALES=1`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_add_rmsnorm_f16s_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_f16_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        residual_buf: &PinnedBuffer,
        norm_weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        eps: f32,
        use_4r: bool,
    ) -> Result<()> {
        if use_4r {
            gemv_q4_k_v4_predec_4r_add_f16s_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_f16_buf, scales_offset, rows, cols, x_buf, residual_buf)?;
        } else {
            gemv_q4_k_v4_predec_2r_add_f16s_pinned_tcb(tcb, model_buf, w_offset, w_byte_size, scales_f16_buf, scales_offset, rows, cols, x_buf, residual_buf)?;
        }
        rmsnorm_metal_buf_tcb(tcb, residual_buf, norm_weight_buf, eps, rows, x_norm_buf)
    }

    /// Track D5 — 4r × f16-scales single GEMV.
    /// Combines predec_4r geometry (32 rows/TG, inline scale reads) with half*
    /// scale tables. For o_proj (2048 rows × 2048 cols): 64 TGs vs 128 (2r_f16s).
    /// Active when HAWKING_QWEN_PREDEC_F16SCALES=1 AND HAWKING_QWEN_PREDEC_4R=1.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_4r_f16s_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_4r_f16s";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<half::f16>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb scales oob: {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 32; // 8 simdgroups × 4 rows (4r geometry)
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track 3.5 — SwiGLU-fused Q4_K predec GEMV (B=1).
    ///
    /// Fuses `silu(gate) * up` inline into the predec Q4_K GEMV, eliminating
    /// the separate `silu_mul_tcb` dispatch. Dispatches `_2r_swiglu` (default),
    /// `_4r_swiglu` (opt-in via `HAWKING_QWEN_PREDEC_4R=1`), or `_swiglu`
    /// (1-row base, opt-out via `HAWKING_QWEN_PREDEC_2R=0`).
    ///
    /// Buffer layout (extra gate/up vs base predec):
    ///   0: w_q4  1: scales  2: gate  3: up  4: y  5: rows  6: cols
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_swiglu_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const BASE: &str = "gemm_q4_k_v4_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{BASE}_swiglu: cols % 256 != 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{BASE}_swiglu: byte overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{BASE}_swiglu: bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{BASE}_swiglu: oob {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kernel(format!("{BASE}_swiglu: scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{BASE}_swiglu: scales oob {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        // 4r is the default for swiglu (ffn_down) — 4 FMA chains vs 2 improves ILP
        // at the cost of inline scale reads (no preload).  Set
        // HAWKING_QWEN_SWIGLU_4R=0 to fall back to the preloaded-2r variant.
        // The legacy HAWKING_QWEN_PREDEC_4R opt-in is superseded for this path.
        let use_4r = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| !std::env::var_os("HAWKING_QWEN_SWIGLU_4R").map(|v| v == "0").unwrap_or(false))
        };
        let use_2r = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| std::env::var_os("HAWKING_QWEN_PREDEC_2R").map(|v| v != "0").unwrap_or(true))
        };
        let (dispatch_kernel, rows_per_tg): (&str, u32) = if use_4r {
            ("gemm_q4_k_v4_predec_4r_swiglu", 32)
        } else if use_2r {
            ("gemm_q4_k_v4_predec_2r_swiglu", 16)
        } else {
            ("gemm_q4_k_v4_predec_swiglu", 8)
        };
        let n_tg = rows_u32.div_ceil(rows_per_tg);
        tcb.dispatch_threads(dispatch_kernel, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(gate_buf), 0);
            enc.set_buffer(3, Some(up_buf), 0);
            enc.set_buffer(4, Some(out_buf), 0);
            enc.set_u32(5, rows_u32);
            enc.set_u32(6, cols_u32);
        })
    }

    /// Track D1 — f16-scales SwiGLU-fused ffn_down (4r geometry).
    ///
    /// f16-scales variant of [`gemv_q4_k_v4_predec_swiglu_pinned_tcb`]. Reads
    /// the predecoded scale table as `half` instead of `f32`, cutting the
    /// scale-table bandwidth for ffn_down from 5.6 MB → 2.8 MB per token (Qwen-3B,
    /// 2048 rows × 11008 cols). Uses 4r geometry (32 rows/TG). NOT bit-identical
    /// (f16 scale rounding ≈5e-4 relative). Only active under
    /// `HAWKING_QWEN_PREDEC_F16SCALES=1`.
    ///
    /// Scale table layout: `rows * (cols/256) * 16 * sizeof(half)` bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_f16s_swiglu_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_f16s_4r_swiglu";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: cols % 256 != 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: byte overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: oob {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        // f16 scale table: 16 halfs/block = 32 bytes/block (vs 64 for f32).
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<half::f16>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}: scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: scales oob {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 32; // 8 simdgroups × 4 rows (4r geometry)
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(gate_buf), 0);
            enc.set_buffer(3, Some(up_buf), 0);
            enc.set_buffer(4, Some(out_buf), 0);
            enc.set_u32(5, rows_u32);
            enc.set_u32(6, cols_u32);
        })
    }

    /// Q4_K decode GEMV with pre-decoded sub-block scales stored as **f16**
    /// (Stage-2 bandwidth lever, `_2r_f16s`).
    ///
    /// Identical 2-row-ILP math + geometry to the default `_2r` predec kernel,
    /// but the pre-decoded `(ds, dm)` pairs are read as `half` (2 B) instead of
    /// f32 (4 B), cutting the predec scale table 192→160 B/block (−17%) on the
    /// bandwidth-bound Q4_K GEMV (the profiling-confirmed ~76%-of-decode-time
    /// wall). Scales widen to f32 in register.
    ///
    /// **NOT bit-identical** to the f32 predec path — the f16 scale rounding
    /// perturbs each `(d*scale)` by ~half-mantissa (≈5e-4 relative), so this is
    /// gated on a quality check (relative parity), not exact equality. Build the
    /// table via [`predecode_q4_k_scale_table_f16`] (16 halfs/block); expected
    /// `scales_buf` length is `rows * (cols / 256) * 16 * sizeof(f16)`.
    ///
    /// **Private API entry point** — not yet wired into the production forward
    /// pass; production opt-in is `HAWKING_QWEN_PREDEC_F16SCALES=1`, gated on
    /// the on-GPU relative-parity + paired bench (must clear the quality bar
    /// before the bandwidth win is bankable).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_v4_predec_2r_f16s_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_v4_predec_2r_f16s";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        // f16 scale table: 16 halfs/block, sizeof(f16) = 2 bytes.
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<half::f16>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb scales oob: {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 16; // 8 simdgroups × 2 rows (2r geometry)
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Q3_K decode GEMV with pre-decoded sub-block scales (byte-cut Stage 3).
    ///
    /// The fast Q3_K GEMV the oracle byte-cut win was blocked on: a Q3_K model
    /// otherwise runs the generic dequant path (~19 dec_tps vs ~32 on the Q4_K
    /// fast stack, because predec/2r are Q4_K-specific). Same 8-rows-per-TG
    /// geometry as `gemm_q3_k_fused_v2` but reads the 16 per-sub-block
    /// `d * scale[i]` f32 values per block from a parallel pre-decoded table
    /// (`scales_buf`) instead of unpacking the packed 6-bit scales + super-block
    /// `d` inline every call. Bit-identical to `gemm_q3_k_fused_v2`.
    ///
    /// Build the table once at load time via
    /// [`crate::quant::predecode_q3_k_scale_table`] and pin it. Q3_K is
    /// symmetric (no min term): the table is `rows * (cols / 256) * 16` f32.
    /// Q3_K block is 110 bytes (vs Q4_K's 144).
    ///
    /// **Private API entry point** — not yet wired into the production forward
    /// pass (hawking's dense path serves Q4_K_M today); that's the byte-cut
    /// consolidation step, gated on the on-GPU parity + paired bench.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3_k_v4_predec_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        scales_buf: &PinnedBuffer,
        scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q3_k_v4_predec";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(110)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let expected_scale_bytes = rows
            .checked_mul(blocks_per_row)
            .and_then(|v| v.checked_mul(16))
            .and_then(|v| v.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb scale overflow")))?;
        if scales_offset + expected_scale_bytes > scales_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb scales oob: {scales_offset}+{expected_scale_bytes} > {}", scales_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(scales_buf), scales_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch_q4_predec_pair_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        kernel: &str,
        rows_per_tg: u32,
        scale_elem_bytes: usize,
        model_buf: &PinnedBuffer,
        g_offset: usize,
        g_byte_size: usize,
        g_scales_buf: &PinnedBuffer,
        g_scales_offset: usize,
        u_offset: usize,
        u_byte_size: usize,
        u_scales_buf: &PinnedBuffer,
        u_scales_offset: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        g_out_buf: &PinnedBuffer,
        u_out_buf: &PinnedBuffer,
    ) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{kernel} requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{kernel} overflow")))?;
        let expected_scale_bytes =
            rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(16)).and_then(|v| v.checked_mul(scale_elem_bytes)).ok_or_else(|| Error::Kernel(format!("{kernel} scale overflow")))?;
        for (tag, bytes, off, sc_buf, sc_off) in [("gate", g_byte_size, g_offset, g_scales_buf, g_scales_offset), ("up", u_byte_size, u_offset, u_scales_buf, u_scales_offset)] {
            if bytes != expected_bytes {
                return Err(Error::Kernel(format!("{kernel} {tag} bytes mismatch: got {bytes} expected {expected_bytes}")));
            }
            if off + bytes > model_buf.length() as usize {
                return Err(Error::Kernel(format!("{kernel} {tag} oob: {off}+{bytes} > {}", model_buf.length())));
            }
            if sc_off + expected_scale_bytes > sc_buf.length() as usize {
                return Err(Error::Kernel(format!("{kernel} {tag} scales oob: {sc_off}+{expected_scale_bytes} > {}", sc_buf.length())));
            }
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg = rows_u32.div_ceil(rows_per_tg);
        tcb.dispatch_threads(kernel, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), g_offset as u64);
            enc.set_buffer(1, Some(g_scales_buf), g_scales_offset as u64);
            enc.set_buffer(2, Some(model_buf), u_offset as u64);
            enc.set_buffer(3, Some(u_scales_buf), u_scales_offset as u64);
            enc.set_buffer(4, Some(x_buf), 0);
            enc.set_buffer(5, Some(g_out_buf), 0);
            enc.set_buffer(6, Some(u_out_buf), 0);
            enc.set_u32(7, rows_u32);
            enc.set_u32(8, cols_u32);
        })
    }

    macro_rules! q4_predec_pair_kernel {
        ($(#[$meta:meta])* $fn_name:ident, $kernel:literal, $rows_per_tg:expr, $scale_ty:ty) => {
            $(#[$meta])*
            #[allow(clippy::too_many_arguments)]
            pub fn $fn_name(
                tcb: &mut TokenCommandBuffer<'_>,
                model_buf: &PinnedBuffer,
                g_offset: usize,
                g_byte_size: usize,
                g_scales_buf: &PinnedBuffer,
                g_scales_offset: usize,
                u_offset: usize,
                u_byte_size: usize,
                u_scales_buf: &PinnedBuffer,
                u_scales_offset: usize,
                rows: usize,
                cols: usize,
                x_buf: &PinnedBuffer,
                g_out_buf: &PinnedBuffer,
                u_out_buf: &PinnedBuffer,
            ) -> Result<()> {
                dispatch_q4_predec_pair_tcb(
                    tcb,
                    $kernel,
                    $rows_per_tg,
                    std::mem::size_of::<$scale_ty>(),
                    model_buf,
                    g_offset,
                    g_byte_size,
                    g_scales_buf,
                    g_scales_offset,
                    u_offset,
                    u_byte_size,
                    u_scales_buf,
                    u_scales_offset,
                    rows,
                    cols,
                    x_buf,
                    g_out_buf,
                    u_out_buf,
                )
            }
        };
    }

    q4_predec_pair_kernel!(
        /// Fused gate+up Q4_K predecode GEMV; preserves the 1-row pair shader.
        gemv_q4_k_v4_predec_pair_pinned_tcb,
        "gemm_q4_k_v4_predec_pair",
        8,
        f32
    );

    q4_predec_pair_kernel!(
        /// 2-row-per-simdgroup gate+up Q4_K predecode pair.
        gemv_q4_k_v4_predec_pair_2r_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_2r",
        16,
        f32
    );

    q4_predec_pair_kernel!(
        /// 2-row inline-scale gate+up Q4_K predecode pair.
        gemv_q4_k_v4_predec_pair_2r_inline_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_2r_inline",
        16,
        f32
    );

    q4_predec_pair_kernel!(
        /// 2-row inline-scale gate+up pair with activation preload removed.
        gemv_q4_k_v4_predec_pair_2r_inline_nox_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_2r_inline_nox",
        16,
        f32
    );

    q4_predec_pair_kernel!(
        /// 3-row-per-simdgroup gate+up Q4_K predecode pair.
        gemv_q4_k_v4_predec_pair_3r_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_3r",
        24,
        f32
    );

    q4_predec_pair_kernel!(
        /// 4-row-per-simdgroup gate+up Q4_K predecode pair.
        gemv_q4_k_v4_predec_pair_4r_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_4r",
        32,
        f32
    );

    q4_predec_pair_kernel!(
        /// 8-row-per-simdgroup gate+up Q4_K predecode pair.
        gemv_q4_k_v4_predec_pair_8r_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_8r",
        64,
        f32
    );

    q4_predec_pair_kernel!(
        /// f16-scale gate+up Q4_K predecode pair.
        gemv_q4_k_v4_predec_pair_f16s_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_f16s",
        8,
        half::f16
    );

    q4_predec_pair_kernel!(
        /// f16-scale gate+up pair with activation preload removed.
        gemv_q4_k_v4_predec_pair_f16s_nox_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_f16s_nox",
        8,
        half::f16
    );

    q4_predec_pair_kernel!(
        /// f16-scale gate+up pair with half-register scale handling.
        gemv_q4_k_v4_predec_pair_f16s_halfreg_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_f16s_halfreg",
        8,
        half::f16
    );

    q4_predec_pair_kernel!(
        /// 2-row inline geometry plus f16 scale tables.
        gemv_q4_k_v4_predec_pair_2r_inline_f16s_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_2r_inline_f16s",
        16,
        half::f16
    );

    q4_predec_pair_kernel!(
        /// 4-row geometry plus f16 scale tables.
        gemv_q4_k_v4_predec_pair_4r_f16s_pinned_tcb,
        "gemm_q4_k_v4_predec_pair_4r_f16s",
        32,
        half::f16
    );

    /// Q4K_FAST v1 — Q4_K with sub-block-contiguous re-layout.
    ///
    /// Weight buffer layout: 160 bytes per 256-element block. Per sub-block
    /// (32 elements):
    ///
    /// ```text
    ///   bytes [k*20 + 0 ..k*20 + 2]   sub_scale (fp16) = d * sb_idx[k]
    ///   bytes [k*20 + 2 ..k*20 + 4]   sub_min   (fp16) = dmin * mb_idx[k]
    ///   bytes [k*20 + 4 ..k*20 + 20]  16 bytes; 32 4-bit values, where
    ///                                 element 2i lives in the low nibble
    ///                                 of byte i and element 2i+1 in the
    ///                                 high nibble.
    /// ```
    ///
    /// Same dispatch geometry as v3_8r (8 rows/TG, 256 threads/TG, 8
    /// simdgroups). Output is bit-identical to `gemv_q4_k_m_v3_8r_pinned_tcb`
    /// when applied to a `q4k_fast`-converted tensor whose per-sub-block
    /// products `d*sb_idx[k]` and `dmin*mb_idx[k]` are exactly representable
    /// in fp16 (the parity-test invariant; covered by
    /// `tests/q4k_fast_parity.rs`).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_fast_v1_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_fast_v1";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(160)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = rows_u32.div_ceil(V3_ROWS);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })
    }

    /// W4A8 prototype — Q4_K weight × int8 activation GEMV at v3_8r geometry.
    /// Activation is per-block (256-element) int8 + f32 scale, expected to
    /// be quantized CPU-side once per layer via `quantize_to_int8_per_block`.
    /// Bandwidth on the activation buffer drops 4× vs `gemv_q4_k_m_v3_8r`;
    /// kernel structure is otherwise identical so the per-call delta is
    /// purely the activation BW saving.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_a8_v3_8r_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_int8_buf: &PinnedBuffer,
        x_scales_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_a8_v3_8r";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        let x_bytes = cols * std::mem::size_of::<i8>();
        let scales_bytes = blocks_per_row * std::mem::size_of::<f32>();
        if x_int8_buf.length() < x_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb x_int8 buffer too small: got {} need {}", x_int8_buf.length(), x_bytes,)));
        }
        if x_scales_buf.length() < scales_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb x_scales buffer too small: got {} need {}", x_scales_buf.length(), scales_bytes,)));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = rows_u32.div_ceil(V3_ROWS);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_int8_buf), 0);
            enc.set_buffer(2, Some(x_scales_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// W4A8 per-channel — Q4_K weight × int8 activation GEMV at v3_8r geometry,
    /// but with ONE f32 scale PER ACTIVATION CHANNEL instead of per 256-element
    /// block. Pairs with `quantize_to_int8_per_channel` (CPU) for the
    /// activation side. Rationale + reconstruction-RMSE evidence in
    /// memory/w4a8_quality_redesign_2026_05_26.md and
    /// memory/w4a8_activation_distribution_2026_05_26.md.
    ///
    /// `x_scales_buf` size: `cols * sizeof(f32)` (vs `(cols/256) * sizeof(f32)`
    /// for the per-block variant).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_q4_k_a8_v3_8r_per_channel_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_int8_buf: &PinnedBuffer,
        x_scales_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_a8_v3_8r_per_channel";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        let x_bytes = cols * std::mem::size_of::<i8>();
        let scales_bytes = cols * std::mem::size_of::<f32>();
        if x_int8_buf.length() < x_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb x_int8 buffer too small: got {} need {}", x_int8_buf.length(), x_bytes,)));
        }
        if x_scales_buf.length() < scales_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb x_scales buffer too small: got {} need {}", x_scales_buf.length(), scales_bytes,)));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = rows_u32.div_ceil(V3_ROWS);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_int8_buf), 0);
            enc.set_buffer(2, Some(x_scales_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// GPU-side per-block int8 quantization of a length-`n` f32 activation
    /// to int8 + per-256-elem f32 scales, matching the CPU reference
    /// (`quantize_to_int8_per_block`) bit-identically. Production W4A8
    /// path uses this to avoid the GPU→CPU readback after rmsnorm.
    ///
    /// Requires `n % 256 == 0`. Writes `n` bytes to `x_int8_buf` and
    /// `n / 256` f32 to `x_scales_buf`.
    pub fn quantize_f32_to_int8_per_block_tcb(tcb: &mut TokenCommandBuffer<'_>, x_buf: &PinnedBuffer, x_int8_buf: &PinnedBuffer, x_scales_buf: &PinnedBuffer, n: usize) -> Result<()> {
        const KERNEL: &str = "quantize_f32_to_int8_per_block";
        if n % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb requires n % 256 == 0; got n={n}")));
        }
        if x_buf.length() < (n * std::mem::size_of::<f32>()) as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_buf too small: got {} need {}", x_buf.length(), n * std::mem::size_of::<f32>(),)));
        }
        if x_int8_buf.length() < n as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_int8_buf too small: got {} need {}", x_int8_buf.length(), n,)));
        }
        let n_blocks = n / 256;
        let scales_bytes = n_blocks * std::mem::size_of::<f32>();
        if x_scales_buf.length() < scales_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_scales_buf too small: got {} need {}", x_scales_buf.length(), scales_bytes,)));
        }
        const TG: u32 = 256;
        let grid_x = n as u32;
        let shmem_bytes = (TG as usize * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (grid_x, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(x_int8_buf), 0);
            enc.set_buffer(2, Some(x_scales_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// AWQ Option B: GPU-side fused activation-divide + per-block int8
    /// quantization. Same shape as `quantize_f32_to_int8_per_block_tcb` but
    /// divides each input element by the matching entry of a per-channel
    /// smoothing vector `s_buf` (length `n`) BEFORE computing the per-block
    /// `max|x|/127` scale. Pairs with offline-baked Q4_K weights
    /// (`W' = W * s`) produced by `tools/awq_bake/`.
    pub fn quantize_f32_to_int8_per_block_scaled_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        s_buf: &PinnedBuffer,
        x_int8_buf: &PinnedBuffer,
        x_scales_buf: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        const KERNEL: &str = "quantize_f32_to_int8_per_block_scaled";
        if n % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb requires n % 256 == 0; got n={n}")));
        }
        let f32_bytes = (n * std::mem::size_of::<f32>()) as u64;
        if x_buf.length() < f32_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_buf too small: got {} need {}", x_buf.length(), f32_bytes,)));
        }
        if s_buf.length() < f32_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_tcb s_buf too small: got {} need {}", s_buf.length(), f32_bytes,)));
        }
        if x_int8_buf.length() < n as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_int8_buf too small: got {} need {}", x_int8_buf.length(), n,)));
        }
        let n_blocks = n / 256;
        let scales_bytes = n_blocks * std::mem::size_of::<f32>();
        if x_scales_buf.length() < scales_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_scales_buf too small: got {} need {}", x_scales_buf.length(), scales_bytes,)));
        }
        const TG: u32 = 256;
        let grid_x = n as u32;
        let shmem_bytes = (TG as usize * std::mem::size_of::<f32>()) as u64;
        tcb.dispatch_threads(KERNEL, (grid_x, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(s_buf), 0);
            enc.set_buffer(2, Some(x_int8_buf), 0);
            enc.set_buffer(3, Some(x_scales_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// GPU-side per-CHANNEL int8 quantization. Pairs with the per-channel
    /// W4A8 path: uses STATIC scales pinned from a calibration pass (e.g.
    /// reports/w4a8_lmhead_calibration_2026_05_26.json on Qwen-3B). Scales
    /// are an INPUT here (read-only); only the int8 output buffer is written.
    ///
    /// Matches the CPU reference `quantize_to_int8_per_channel`:
    ///   q[i] = round(x[i] / scales[i]).clamp(-127, 127)
    ///
    /// `scales_buf` must hold at least `n * sizeof(f32)` bytes (one scale
    /// per channel/element). `x_int8_buf` must hold at least `n` bytes.
    pub fn quantize_f32_to_int8_per_channel_tcb(tcb: &mut TokenCommandBuffer<'_>, x_buf: &PinnedBuffer, scales_buf: &PinnedBuffer, x_int8_buf: &PinnedBuffer, n: usize) -> Result<()> {
        const KERNEL: &str = "quantize_f32_to_int8_per_channel";
        if x_buf.length() < (n * std::mem::size_of::<f32>()) as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_buf too small: got {} need {}", x_buf.length(), n * std::mem::size_of::<f32>(),)));
        }
        if scales_buf.length() < (n * std::mem::size_of::<f32>()) as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb scales_buf too small: got {} need {}", scales_buf.length(), n * std::mem::size_of::<f32>(),)));
        }
        if x_int8_buf.length() < n as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_tcb x_int8_buf too small: got {} need {}", x_int8_buf.length(), n,)));
        }
        const TG: u32 = 256;
        let grid_x = (n as u32).next_multiple_of(TG);
        let n_u32 = n as u32;
        tcb.dispatch_threads(KERNEL, (grid_x, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(scales_buf), 0);
            enc.set_buffer(2, Some(x_int8_buf), 0);
            enc.set_u32(3, n_u32);
        })
    }

    /// CPU-side per-block int8 quantization of a length-cols f32 activation
    /// vector. Splits into ceil(cols/256) blocks, computes `scale = max|x|/127`
    /// per block, encodes `x_int8[i] = round(x[i] / scale)` clamped to
    /// [-127, 127]. Returns (int8 bytes, f32 scales). Used by the W4A8
    /// prototype to feed `gemm_q4_k_a8_v3_8r_pinned_tcb`.
    pub fn quantize_to_int8_per_block(x: &[f32], block_size: usize) -> (Vec<i8>, Vec<f32>) {
        let blocks = x.len().div_ceil(block_size);
        let mut out_int8 = vec![0i8; x.len()];
        let mut scales = vec![0.0f32; blocks];
        for b in 0..blocks {
            let lo = b * block_size;
            let hi = (lo + block_size).min(x.len());
            let mut max_abs = 0.0f32;
            for &v in &x[lo..hi] {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales[b] = scale;
            for i in lo..hi {
                let q = (x[i] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                out_int8[i] = q;
            }
        }
        (out_int8, scales)
    }

    /// CPU reference for the AWQ Option B fused divide-and-quantize. Same
    /// semantics as `quantize_to_int8_per_block` but pre-divides each element
    /// by the matching entry of a per-channel smoothing vector `s` BEFORE
    /// computing the per-block scale. Used by the parity test for
    /// `quantize_f32_to_int8_per_block_scaled`.
    pub fn quantize_to_int8_per_block_scaled(x: &[f32], s: &[f32], block_size: usize) -> (Vec<i8>, Vec<f32>) {
        assert_eq!(x.len(), s.len(), "quantize_to_int8_per_block_scaled: x.len()={} != s.len()={}", x.len(), s.len(),);
        let blocks = x.len().div_ceil(block_size);
        let mut out_int8 = vec![0i8; x.len()];
        let mut scales = vec![0.0f32; blocks];
        for b in 0..blocks {
            let lo = b * block_size;
            let hi = (lo + block_size).min(x.len());
            let mut max_abs = 0.0f32;
            for i in lo..hi {
                let sv = s[i];
                let inv_s = if sv > 1e-12 { 1.0 / sv } else { 0.0 };
                let scaled = x[i] * inv_s;
                let a = scaled.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales[b] = scale;
            for i in lo..hi {
                let sv = s[i];
                let inv_s = if sv > 1e-12 { 1.0 / sv } else { 0.0 };
                let scaled = x[i] * inv_s;
                let q = (scaled * inv_scale).round().clamp(-127.0, 127.0) as i8;
                out_int8[i] = q;
            }
        }
        (out_int8, scales)
    }

    /// CPU-side PER-CHANNEL int8 quantization of a length-cols f32 activation
    /// vector. Each channel gets its OWN scale (one f32 per element) computed
    /// from a running per-channel max|x| estimate provided by the caller.
    ///
    /// For the static-calibration use case, `channel_scales[c]` is the
    /// pre-computed `max|x_c| / 127` over a calibration corpus (e.g., the
    /// 180-sample analysis in
    /// memory/w4a8_activation_distribution_2026_05_26.md). For the dynamic
    /// use case (where we have to recompute every token), `channel_scales` is
    /// derived from the current activation itself — equivalent to per-block
    /// with block_size=1, i.e., trivially scaled by |x[c]|, which makes the
    /// int8 quantum meaningless (every element rounds to ±127). The
    /// production wire-up therefore uses STATIC scales from calibration plus
    /// a per-token rescaling guard.
    ///
    /// Returns int8 bytes (one per channel). The caller owns `channel_scales`.
    /// Bit-identical with the per-block path when each block_size=1; quality
    /// improvement comes from the per-channel scales being chosen from a
    /// CORPUS distribution, not a single token's max.
    pub fn quantize_to_int8_per_channel(x: &[f32], channel_scales: &[f32]) -> Vec<i8> {
        assert_eq!(x.len(), channel_scales.len(), "quantize_to_int8_per_channel: x.len()={} != scales.len()={}", x.len(), channel_scales.len());
        let mut out = vec![0i8; x.len()];
        for i in 0..x.len() {
            let s = channel_scales[i];
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let q = (x[i] * inv).round().clamp(-127.0, 127.0) as i8;
            out[i] = q;
        }
        out
    }

    /// Convenience: derive per-channel scales from a single-vector max — i.e.,
    /// `channel_scales[c] = |x[c]| / 127`. Useful only as a parity-test fixture
    /// (the dynamic case degenerates to "every element saturates at ±127");
    /// production calibration-based scales come from a corpus pass and are
    /// stored in the model/profile.
    pub fn per_channel_scales_from_abs(x: &[f32]) -> Vec<f32> {
        x.iter()
            .map(|&v| {
                let a = v.abs();
                if a > 0.0 {
                    a / 127.0
                } else {
                    1.0
                }
            })
            .collect()
    }

    /// P2 — v3_llama: 2 simdgroups × 4-rows-each per TG (TG=64, 8 rows/TG).
    /// Lower per-TG occupancy + higher TG count compared to v3_8r;
    /// candidate for shapes where the GPU scheduler benefits from more
    /// independent threadgroups.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_v3_llama_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_llama";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 64;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = rows_u32.div_ceil(ROWS_PER_TG);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })
    }

    /// P2 — Q6_K-weight × fp32-vec → fp32 GEMV against pinned model
    /// buffer + byte offset window. Same dispatch shape as
    /// `gemv_q4_k_m_v2_pinned_tcb`: 8 rows per TG, 32 threads/row.
    /// Replaces the f16-dequant fallback for Q6_K weights in Q4_K_M
    /// mix-quant GGUFs; saves ~2.46× bandwidth on those layers.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_fused_v2";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        let end = w_offset.checked_add(w_byte_size).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb offset oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = rows_u32.div_ceil(8);
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    /// Track 3.5 — SwiGLU-fused Q6_K GEMV.
    ///
    /// Identical to [`gemv_q6_k_pinned_tcb`] but fuses `silu(gate) * up` inline as
    /// the activation, eliminating the preceding `silu_mul_tcb` dispatch.
    /// Saves 1 dispatch/layer × n_layers on the default Q4_K_M path where
    /// `ffn_down` is Q6_K.
    ///
    /// Buffer layout in the kernel (differs from base):
    ///   0: w_q6  1: gate  2: up  3: y  4: ArgbufRowsCols{rows, cols}
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_swiglu_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_fused_v2_swiglu";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}: weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        let end = w_offset.checked_add(w_byte_size).ok_or_else(|| Error::Kernel(format!("{KERNEL}: offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}: offset oob: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        // Track D8: opt-in 4r variant (32 rows/TG = 64 TGs for Qwen-3B ffn_down).
        // Better memory latency hiding (4 independent Q6K weight streams per thread).
        // Opt-in via HAWKING_QWEN_Q6K_SWIGLU_4R=1; subsumes 2r when set.
        let use_4r = {
            static E4R: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E4R.get_or_init(|| std::env::var_os("HAWKING_QWEN_Q6K_SWIGLU_4R").map(|v| v != "0").unwrap_or(false))
        };
        // Track D7: default to the 2r variant (16 rows/TG, 128 TGs for Qwen-3B
        // ffn_down vs 256 for 1r). Opt-out: HAWKING_QWEN_Q6K_SWIGLU_2R=0.
        let use_2r = !use_4r && {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| !std::env::var_os("HAWKING_QWEN_Q6K_SWIGLU_2R").map(|v| v == "0").unwrap_or(false))
        };
        let (dispatch_kernel, rows_per_tg): (&str, u32) = if use_4r {
            ("gemm_q6_k_fused_v2_swiglu_4r", 32)
        } else if use_2r {
            ("gemm_q6_k_fused_v2_swiglu_2r", 16)
        } else {
            (KERNEL, 8)
        };
        let n_tg = rows_u32.div_ceil(rows_per_tg);
        tcb.dispatch_threads(dispatch_kernel, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(gate_buf), 0);
            enc.set_buffer(2, Some(up_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track D7 — Direct dispatch for 1r Q6K swiglu (parity reference).
    /// Bypasses the OnceLock geometry selection; always uses the 1r kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_swiglu_1r_direct_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_fused_v2_swiglu";
        let blocks_per_row = cols / 256;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_tg = rows_u32.div_ceil(8);
        let _ = blocks_per_row;
        let _ = w_byte_size;
        tcb.dispatch_threads(KERNEL, (n_tg * 256, 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(gate_buf), 0);
            enc.set_buffer(2, Some(up_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track D7 — Direct dispatch for 2r Q6K swiglu (Track D7 kernel).
    /// Bypasses the OnceLock geometry selection; always uses the 2r kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_swiglu_2r_direct_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_fused_v2_swiglu_2r";
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let _ = w_byte_size;
        let n_tg = rows_u32.div_ceil(16);
        tcb.dispatch_threads(KERNEL, (n_tg * 256, 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(gate_buf), 0);
            enc.set_buffer(2, Some(up_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track D8 — Direct dispatch for 4r Q6K swiglu.
    /// Bypasses OnceLock; always uses `gemm_q6_k_fused_v2_swiglu_4r` (32 rows/TG).
    /// Used by the D8 parity test to compare 4r vs 2r/1r directly.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_swiglu_4r_direct_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        gate_buf: &PinnedBuffer,
        up_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_fused_v2_swiglu_4r";
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let _ = w_byte_size;
        let n_tg = rows_u32.div_ceil(32);
        tcb.dispatch_threads(KERNEL, (n_tg * 256, 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(gate_buf), 0);
            enc.set_buffer(2, Some(up_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_u32(4, rows_u32);
            enc.set_u32(5, cols_u32);
        })
    }

    /// Track 3.8 — Fused K+V Q6_K GEMV pair (`gemm_q6_k_kv_pair`).
    ///
    /// Computes both K and V projections in one dispatch, sharing the `x_norm`
    /// read. Saves 1 dispatch/layer × n_layers (28 on Qwen-3B).
    ///
    /// Both K and V must be Q6_K and the same shape (`rows` × `cols`).
    /// The caller binds the same pinned model buffer at two byte offsets.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_kv_pair_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        v_offset: usize,
        v_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        yk_buf: &PinnedBuffer,
        yv_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_kv_pair";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: byte-size overflow")))?;
        for (label, off, sz) in [("k", k_offset, k_byte_size), ("v", v_offset, v_byte_size)] {
            if sz != expected_bytes {
                return Err(Error::Kernel(format!("{KERNEL}: {label} weight bytes: got {sz} expected {expected_bytes}")));
            }
            let end = off.checked_add(sz).ok_or_else(|| Error::Kernel(format!("{KERNEL}: {label} offset overflow")))?;
            if end > model_buf.length() as usize {
                return Err(Error::Kernel(format!("{KERNEL}: {label} offset oob: {off}+{sz} > {}", model_buf.length())));
            }
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        const TG: u32 = 256;
        let n_tg = rows_u32.div_ceil(8);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), k_offset as u64);
            enc.set_buffer(1, Some(model_buf), v_offset as u64);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(yk_buf), 0);
            enc.set_buffer(4, Some(yv_buf), 0);
            enc.set_bytes(5, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
        })
    }

    /// Track 3.9 — Cross-dtype K+V pair: K=Q4_K (predec) + V=Q6_K (inline).
    ///
    /// One dispatch computes both K and V projections when they have different
    /// quantization types (k_proj=Q4_K, v_proj=Q6_K). Saves 1 dispatch/layer.
    /// Grid: 2 × ceil(kv_dim/8) × 256.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_predec_q6k_pair_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        yk_buf: &PinnedBuffer,
        yv_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_predec_q6k_pair";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let k_expected = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: k overflow")))?;
        let v_expected = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: v overflow")))?;
        if k_byte_size != k_expected {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {k_expected}")));
        }
        if v_byte_size != v_expected {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {v_expected}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg = rows_u32.div_ceil(8);
        tcb.dispatch_threads(KERNEL, (2 * n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), k_offset as u64);
            enc.set_buffer(1, Some(k_scales), 0);
            enc.set_buffer(2, Some(model_buf), v_offset as u64);
            enc.set_buffer(3, Some(x_buf), 0);
            enc.set_buffer(4, Some(yk_buf), 0);
            enc.set_buffer(5, Some(yv_buf), 0);
            enc.set_u32(6, rows_u32);
            enc.set_u32(7, cols_u32);
        })
    }

    /// Track 3.10 — Fused Q+K+V Q4_K predec triple.
    ///
    /// All three projections use Q4_K predec format and share the dispatch overhead.
    /// Replaces the separate Q dispatch + KV-pair dispatch (2 dispatches → 1).
    /// Q has q_rows (q_dim), K and V each have kv_rows (kv_dim).
    /// Grid: (ceil(q_rows/8) + 2*ceil(kv_rows/8)) × 256.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_predec_qkv_triple_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        v_scales: &PinnedBuffer,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        yq_buf: &PinnedBuffer,
        yk_buf: &PinnedBuffer,
        yv_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_predec_qkv_triple";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q overflow")))?;
        let k_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: k overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != k_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {k_exp}")));
        }
        if v_byte_size != k_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {k_exp}")));
        }
        let q_rows_u32 = q_rows as u32;
        let kv_rows_u32 = kv_rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg_q = q_rows_u32.div_ceil(8);
        let n_tg_kv = kv_rows_u32.div_ceil(8);
        let total_tg = n_tg_q + 2 * n_tg_kv;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(v_scales), 0);
            enc.set_buffer(6, Some(x_buf), 0);
            enc.set_buffer(7, Some(yq_buf), 0);
            enc.set_buffer(8, Some(yk_buf), 0);
            enc.set_buffer(9, Some(yv_buf), 0);
            enc.set_u32(10, q_rows_u32);
            enc.set_u32(11, kv_rows_u32);
            enc.set_u32(12, cols_u32);
        })
    }

    /// Track 3.11 — Mixed Q(Q4K predec)+K(Q4K predec)+V(Q6K) triple.
    ///
    /// For layers where q and k are Q4_K predec but v is Q6_K. All three in
    /// one dispatch, saving 1 vs Q-separate + cross-dtype-pair.
    /// Grid: (ceil(q_rows/8) + 2*ceil(kv_rows/8)) × 256.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_q4k_q6k_triple_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        yq_buf: &PinnedBuffer,
        yk_buf: &PinnedBuffer,
        yv_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_q4k_q6k_triple";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q overflow")))?;
        let k_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: k overflow")))?;
        let v_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: v overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != k_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {k_exp}")));
        }
        if v_byte_size != v_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {v_exp}")));
        }
        let q_rows_u32 = q_rows as u32;
        let kv_rows_u32 = kv_rows as u32;
        let cols_u32 = cols as u32;
        const TG: u32 = 256;
        let n_tg_q = q_rows_u32.div_ceil(8);
        let n_tg_kv = kv_rows_u32.div_ceil(8);
        let total_tg = n_tg_q + 2 * n_tg_kv;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(x_buf), 0);
            enc.set_buffer(6, Some(yq_buf), 0);
            enc.set_buffer(7, Some(yk_buf), 0);
            enc.set_buffer(8, Some(yv_buf), 0);
            enc.set_u32(9, q_rows_u32);
            enc.set_u32(10, kv_rows_u32);
            enc.set_u32(11, cols_u32);
        })
    }

    fn validate_qkv_rope_append_shape(kernel: &str, q_rows: usize, kv_rows: usize, cols: usize, n_q_heads: usize, n_k_heads: usize, head_dim: usize, kv_off: usize) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{kernel}: requires cols % 256 == 0; got cols={cols}")));
        }
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::Kernel(format!("{kernel}: head_dim must be non-zero and even; got {head_dim}")));
        }
        if q_rows != n_q_heads * head_dim {
            return Err(Error::Kernel(format!("{kernel}: q_rows={q_rows} != n_q_heads({n_q_heads})*head_dim({head_dim})")));
        }
        if kv_rows != n_k_heads * head_dim {
            return Err(Error::Kernel(format!("{kernel}: kv_rows={kv_rows} != n_k_heads({n_k_heads})*head_dim({head_dim})")));
        }
        if q_rows % 2 != 0 || kv_rows % 2 != 0 {
            return Err(Error::Kernel(format!("{kernel}: q_rows and kv_rows must be even; got {q_rows}/{kv_rows}")));
        }
        if kv_off > u32::MAX as usize {
            return Err(Error::Kernel(format!("{kernel}: kv_off={kv_off} exceeds u32 addressable elements")));
        }
        Ok(())
    }

    /// Track 3.12/3.13 — Q4K/Q4K/Q4K triple with inline Q/K bias+RoPE and
    /// direct f32 KV-cache append (+ optional V bias).
    ///
    /// Replaces `gemv_q4k_predec_qkv_triple_pinned_tcb` +
    /// `rope_qk_f32_b1_bias_tcb` + `kv_append_vbias_f32_tcb` with one dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_predec_qkv_rope_append_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        v_scales: &PinnedBuffer,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        rope_base: f32,
        kv_off: usize,
        x_buf: &PinnedBuffer,
        q_buf: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_predec_qkv_rope_append";
        validate_qkv_rope_append_shape(KERNEL, q_rows, kv_rows, cols, n_q_heads, n_k_heads, head_dim, kv_off)?;
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q byte overflow")))?;
        let kv_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: kv byte overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {kv_exp}")));
        }
        if v_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {kv_exp}")));
        }
        let args = ArgbufQkvRopeAppend {
            q_rows: q_rows as u32,
            kv_rows: kv_rows as u32,
            cols: cols as u32,
            n_q_heads: n_q_heads as u32,
            n_k_heads: n_k_heads as u32,
            head_dim: head_dim as u32,
            pos,
            kv_off: kv_off as u32,
            has_q_bias: q_bias_buf.is_some() as u32,
            has_k_bias: k_bias_buf.is_some() as u32,
            has_v_bias: v_bias_buf.is_some() as u32,
            base: rope_base,
        };
        let q_bias = q_bias_buf.unwrap_or(q_buf);
        let k_bias = k_bias_buf.unwrap_or(q_buf);
        let v_bias = v_bias_buf.unwrap_or(q_buf);
        const TG: u32 = 256;
        let q_pair_tg = ((q_rows / 2) as u32).div_ceil(8);
        let k_pair_tg = ((kv_rows / 2) as u32).div_ceil(8);
        let v_tg = (kv_rows as u32).div_ceil(8);
        let total_tg = q_pair_tg + k_pair_tg + v_tg;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(v_scales), 0);
            enc.set_buffer(6, Some(x_buf), 0);
            enc.set_buffer(7, Some(q_buf), 0);
            enc.set_buffer(8, Some(k_cache), 0);
            enc.set_buffer(9, Some(v_cache), 0);
            enc.set_buffer(10, Some(q_bias), 0);
            enc.set_buffer(11, Some(k_bias), 0);
            enc.set_buffer(12, Some(v_bias), 0);
            enc.set_bytes(13, std::mem::size_of::<ArgbufQkvRopeAppend>() as u64, &args as *const ArgbufQkvRopeAppend as *const _);
        })
    }

    /// Track C28 — 4r variant of gemv_q4k_predec_qkv_rope_append_pinned_tcb.
    /// Q and K use 4 rows/simdgroup (2 RoPE pairs); V uses 2 rows/simdgroup.
    /// Total TGs for Qwen-3B: 160 vs 320 — same 1 dispatch, half scheduling overhead.
    /// Requires q_rows % 4 == 0 and kv_rows % 4 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_predec_qkv_rope_append_4r_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        v_scales: &PinnedBuffer,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        rope_base: f32,
        kv_off: usize,
        x_buf: &PinnedBuffer,
        q_buf: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_predec_qkv_rope_append_4r";
        validate_qkv_rope_append_shape(KERNEL, q_rows, kv_rows, cols, n_q_heads, n_k_heads, head_dim, kv_off)?;
        if q_rows % 4 != 0 || kv_rows % 4 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: q_rows ({q_rows}) and kv_rows ({kv_rows}) must be divisible by 4")));
        }
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q byte overflow")))?;
        let kv_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: kv byte overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {kv_exp}")));
        }
        if v_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {kv_exp}")));
        }
        let args = ArgbufQkvRopeAppend {
            q_rows: q_rows as u32,
            kv_rows: kv_rows as u32,
            cols: cols as u32,
            n_q_heads: n_q_heads as u32,
            n_k_heads: n_k_heads as u32,
            head_dim: head_dim as u32,
            pos,
            kv_off: kv_off as u32,
            has_q_bias: q_bias_buf.is_some() as u32,
            has_k_bias: k_bias_buf.is_some() as u32,
            has_v_bias: v_bias_buf.is_some() as u32,
            base: rope_base,
        };
        let q_bias = q_bias_buf.unwrap_or(q_buf);
        let k_bias = k_bias_buf.unwrap_or(q_buf);
        let v_bias = v_bias_buf.unwrap_or(q_buf);
        const TG: u32 = 256;
        let q_quad_tg = ((q_rows / 4) as u32).div_ceil(8);
        let k_quad_tg = ((kv_rows / 4) as u32).div_ceil(8);
        let v_pair_tg = ((kv_rows / 2) as u32).div_ceil(8);
        let total_tg = q_quad_tg + k_quad_tg + v_pair_tg;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(v_scales), 0);
            enc.set_buffer(6, Some(x_buf), 0);
            enc.set_buffer(7, Some(q_buf), 0);
            enc.set_buffer(8, Some(k_cache), 0);
            enc.set_buffer(9, Some(v_cache), 0);
            enc.set_buffer(10, Some(q_bias), 0);
            enc.set_buffer(11, Some(k_bias), 0);
            enc.set_buffer(12, Some(v_bias), 0);
            enc.set_bytes(13, std::mem::size_of::<ArgbufQkvRopeAppend>() as u64, &args as *const ArgbufQkvRopeAppend as *const _);
        })
    }

    /// Track D3 — f16-scales 2r variant of gemv_q4k_predec_qkv_rope_append_pinned_tcb.
    /// q_scales/k_scales/v_scales hold half-precision (f16) predecoded tables —
    /// half the bandwidth of the f32 versions. Otherwise identical geometry (320 TGs
    /// for Qwen-3B). Scale buffer size: rows * (cols/256) * 16 * sizeof(f16).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_predec_qkv_rope_append_f16s_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        v_scales: &PinnedBuffer,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        rope_base: f32,
        kv_off: usize,
        x_buf: &PinnedBuffer,
        q_buf: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_predec_qkv_rope_append_f16s";
        validate_qkv_rope_append_shape(KERNEL, q_rows, kv_rows, cols, n_q_heads, n_k_heads, head_dim, kv_off)?;
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q byte overflow")))?;
        let kv_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: kv byte overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {kv_exp}")));
        }
        if v_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {kv_exp}")));
        }
        let args = ArgbufQkvRopeAppend {
            q_rows: q_rows as u32,
            kv_rows: kv_rows as u32,
            cols: cols as u32,
            n_q_heads: n_q_heads as u32,
            n_k_heads: n_k_heads as u32,
            head_dim: head_dim as u32,
            pos,
            kv_off: kv_off as u32,
            has_q_bias: q_bias_buf.is_some() as u32,
            has_k_bias: k_bias_buf.is_some() as u32,
            has_v_bias: v_bias_buf.is_some() as u32,
            base: rope_base,
        };
        let q_bias = q_bias_buf.unwrap_or(q_buf);
        let k_bias = k_bias_buf.unwrap_or(q_buf);
        let v_bias = v_bias_buf.unwrap_or(q_buf);
        const TG: u32 = 256;
        let q_pair_tg = ((q_rows / 2) as u32).div_ceil(8);
        let k_pair_tg = ((kv_rows / 2) as u32).div_ceil(8);
        let v_tg = (kv_rows as u32).div_ceil(8);
        let total_tg = q_pair_tg + k_pair_tg + v_tg;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(v_scales), 0);
            enc.set_buffer(6, Some(x_buf), 0);
            enc.set_buffer(7, Some(q_buf), 0);
            enc.set_buffer(8, Some(k_cache), 0);
            enc.set_buffer(9, Some(v_cache), 0);
            enc.set_buffer(10, Some(q_bias), 0);
            enc.set_buffer(11, Some(k_bias), 0);
            enc.set_buffer(12, Some(v_bias), 0);
            enc.set_bytes(13, std::mem::size_of::<ArgbufQkvRopeAppend>() as u64, &args as *const ArgbufQkvRopeAppend as *const _);
        })
    }

    /// Track D3 — f16-scales 4r variant of gemv_q4k_predec_qkv_rope_append_4r_pinned_tcb.
    /// Combines C28 (4r = 160 TGs for Qwen-3B) with D3 (f16 scale bandwidth).
    /// Requires q_rows % 4 == 0 and kv_rows % 4 == 0.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_predec_qkv_rope_append_4r_f16s_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        v_scales: &PinnedBuffer,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        rope_base: f32,
        kv_off: usize,
        x_buf: &PinnedBuffer,
        q_buf: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_predec_qkv_rope_append_4r_f16s";
        validate_qkv_rope_append_shape(KERNEL, q_rows, kv_rows, cols, n_q_heads, n_k_heads, head_dim, kv_off)?;
        if q_rows % 4 != 0 || kv_rows % 4 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}: q_rows ({q_rows}) and kv_rows ({kv_rows}) must be divisible by 4")));
        }
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q byte overflow")))?;
        let kv_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: kv byte overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {kv_exp}")));
        }
        if v_byte_size != kv_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {kv_exp}")));
        }
        let args = ArgbufQkvRopeAppend {
            q_rows: q_rows as u32,
            kv_rows: kv_rows as u32,
            cols: cols as u32,
            n_q_heads: n_q_heads as u32,
            n_k_heads: n_k_heads as u32,
            head_dim: head_dim as u32,
            pos,
            kv_off: kv_off as u32,
            has_q_bias: q_bias_buf.is_some() as u32,
            has_k_bias: k_bias_buf.is_some() as u32,
            has_v_bias: v_bias_buf.is_some() as u32,
            base: rope_base,
        };
        let q_bias = q_bias_buf.unwrap_or(q_buf);
        let k_bias = k_bias_buf.unwrap_or(q_buf);
        let v_bias = v_bias_buf.unwrap_or(q_buf);
        const TG: u32 = 256;
        let q_quad_tg = ((q_rows / 4) as u32).div_ceil(8);
        let k_quad_tg = ((kv_rows / 4) as u32).div_ceil(8);
        let v_pair_tg = ((kv_rows / 2) as u32).div_ceil(8);
        let total_tg = q_quad_tg + k_quad_tg + v_pair_tg;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(v_scales), 0);
            enc.set_buffer(6, Some(x_buf), 0);
            enc.set_buffer(7, Some(q_buf), 0);
            enc.set_buffer(8, Some(k_cache), 0);
            enc.set_buffer(9, Some(v_cache), 0);
            enc.set_buffer(10, Some(q_bias), 0);
            enc.set_buffer(11, Some(k_bias), 0);
            enc.set_buffer(12, Some(v_bias), 0);
            enc.set_bytes(13, std::mem::size_of::<ArgbufQkvRopeAppend>() as u64, &args as *const ArgbufQkvRopeAppend as *const _);
        })
    }

    /// Track 3.12/3.13 mixed variant: Q/K are Q4_K predec, V is Q6_K.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4k_q4k_q6k_rope_append_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        q_offset: usize,
        q_byte_size: usize,
        q_scales: &PinnedBuffer,
        k_offset: usize,
        k_byte_size: usize,
        k_scales: &PinnedBuffer,
        v_offset: usize,
        v_byte_size: usize,
        q_rows: usize,
        kv_rows: usize,
        cols: usize,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        rope_base: f32,
        kv_off: usize,
        x_buf: &PinnedBuffer,
        q_buf: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4k_q4k_q6k_rope_append";
        validate_qkv_rope_append_shape(KERNEL, q_rows, kv_rows, cols, n_q_heads, n_k_heads, head_dim, kv_off)?;
        let blocks_per_row = cols / 256;
        let q_exp = q_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: q byte overflow")))?;
        let k_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: k byte overflow")))?;
        let v_exp = kv_rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}: v byte overflow")))?;
        if q_byte_size != q_exp {
            return Err(Error::Kernel(format!("{KERNEL}: q bytes: got {q_byte_size} expected {q_exp}")));
        }
        if k_byte_size != k_exp {
            return Err(Error::Kernel(format!("{KERNEL}: k bytes: got {k_byte_size} expected {k_exp}")));
        }
        if v_byte_size != v_exp {
            return Err(Error::Kernel(format!("{KERNEL}: v bytes: got {v_byte_size} expected {v_exp}")));
        }
        let args = ArgbufQkvRopeAppend {
            q_rows: q_rows as u32,
            kv_rows: kv_rows as u32,
            cols: cols as u32,
            n_q_heads: n_q_heads as u32,
            n_k_heads: n_k_heads as u32,
            head_dim: head_dim as u32,
            pos,
            kv_off: kv_off as u32,
            has_q_bias: q_bias_buf.is_some() as u32,
            has_k_bias: k_bias_buf.is_some() as u32,
            has_v_bias: v_bias_buf.is_some() as u32,
            base: rope_base,
        };
        let q_bias = q_bias_buf.unwrap_or(q_buf);
        let k_bias = k_bias_buf.unwrap_or(q_buf);
        let v_bias = v_bias_buf.unwrap_or(q_buf);
        const TG: u32 = 256;
        let q_pair_tg = ((q_rows / 2) as u32).div_ceil(8);
        let k_pair_tg = ((kv_rows / 2) as u32).div_ceil(8);
        let v_tg = (kv_rows as u32).div_ceil(8);
        let total_tg = q_pair_tg + k_pair_tg + v_tg;
        tcb.dispatch_threads(KERNEL, (total_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), q_offset as u64);
            enc.set_buffer(1, Some(q_scales), 0);
            enc.set_buffer(2, Some(model_buf), k_offset as u64);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(model_buf), v_offset as u64);
            enc.set_buffer(5, Some(x_buf), 0);
            enc.set_buffer(6, Some(q_buf), 0);
            enc.set_buffer(7, Some(k_cache), 0);
            enc.set_buffer(8, Some(v_cache), 0);
            enc.set_buffer(9, Some(q_bias), 0);
            enc.set_buffer(10, Some(k_bias), 0);
            enc.set_buffer(11, Some(v_bias), 0);
            enc.set_bytes(12, std::mem::size_of::<ArgbufQkvRopeAppend>() as u64, &args as *const ArgbufQkvRopeAppend as *const _);
        })
    }

    /// Q3_K-weight × fp32-vec → fp32 GEMV, dispatching `gemm_q3_k_fused_v2`
    /// against a pinned model buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3_k_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q3_k_gemv_pinned(ctx, "gemm_q3_k_fused_v2", model_buf, w_offset, w_byte_size, rows, cols, x, out)
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
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(110)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        let end = w_offset.checked_add(w_byte_size).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let x_bytes = cols * std::mem::size_of::<f32>();
        let out_bytes = rows * std::mem::size_of::<f32>();
        if x_buf.length() < x_bytes as u64 || out_buf.length() < out_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} expected>={x_bytes} out={} expected>={out_bytes}", x_buf.length(), out_buf.length())));
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        // gemm_q3_k_fused_v2 reads ONE `ArgbufRowsCols` struct at buffer(3), not
        // two separate uints — pack it like gemv_q3_k_pinned does. (Writing two
        // set_bytes at 3/4 left args.cols=0 → blocks_per_row=0 → all-zero output.)
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;
        tcb.dispatch_threads(KERNEL, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
        })
    }

    /// 2-row-ILP FUSED Q3_K GEMV — dispatches `gemm_q3_k_fused_2r`. Same buffer
    /// layout and `ArgbufRowsCols` binding as `gemv_q3_k_pinned_tcb`, but 16
    /// rows/TG (8 simdgroups x 2 rows) with two accumulator chains sharing the
    /// `x` load. Bit-identical per-row to `gemm_q3_k_fused_v2`; the byte-cut
    /// speed lever (no scale table — fewest bytes).
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q3_k_fused_2r_pinned_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q3_k_fused_2r";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(110)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        let end = w_offset.checked_add(w_byte_size).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_tcb offset overflow")))?;
        if end > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }
        let x_bytes = cols * std::mem::size_of::<f32>();
        let out_bytes = rows * std::mem::size_of::<f32>();
        if x_buf.length() < x_bytes as u64 || out_buf.length() < out_bytes as u64 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_tcb buffer sizes: x={} expected>={x_bytes} out={} expected>={out_bytes}", x_buf.length(), out_buf.length())));
        }

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        // Same packed ArgbufRowsCols binding as gemm_q3_k_fused_v2 (buffer 3).
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        const TG: u32 = 256;
        // 16 rows/TG (8 simdgroups x 2 rows).
        let n_tg = rows_u32.div_ceil(16);
        tcb.dispatch_threads(KERNEL, (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
        })
    }

    /// Wedge K -- simdmat-optimised pinned-buffer Q4_K_M GEMV. Same signature
    /// as `gemv_q4_k_m_v2_pinned`; dispatches `gemm_q4_k_m_simdmat`.
    /// Selected via `gemm_q4_k_schedule = "simdmat"`.
    ///
    /// Uses 128-thread / 4-row-per-TG geometry (vs v2's 256/8) for better
    /// parallelism on small-row expert shapes.
    pub fn gemv_q4_k_m_simdmat_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_simdmat_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// Approach 1 Iter 1 -- 256 threads, 8 rows/TG, 8 simdgroups.
    /// Selected via `gemm_q4_k_schedule = "v3_8r"`.
    pub fn gemv_q4_k_m_v3_8r_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_v3_8r_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// Approach 3 -- 64 threads, 4 rows/simdgroup (N_R0=4), sumy trick.
    /// Selected via `gemm_q4_k_schedule = "v3_llama"`.
    pub fn gemv_q4_k_m_v3_llama_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_v3_llama_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// v1.1.0 opt-in schedule name for the faithful llama.cpp-style Q4_K port.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_llama_port_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        gemv_q4_k_m_v3_llama_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// Approach 1 Iter 2 -- 128 threads, 2 rows/simdgroup (N_R0=2), 8 rows/TG.
    /// Selected via `gemm_q4_k_schedule = "v3_dual"`.
    pub fn gemv_q4_k_m_v3_dual_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_q4_k_m_v3_dual_pinned(ctx, model_buf, w_offset, w_byte_size, rows, cols, x, out)
    }

    /// v0.3.1 -- low-level batched encoder for `gemm_q4_k_m_fused_simd`.
    /// Takes pre-allocated Metal buffers; encodes into an existing CommandBatch
    /// without allocation or readback. Use this to coalesce multiple independent
    /// simd GEMVs (e.g. gate + up) into a single command buffer.
    pub(crate) fn encode_gemv_q4_k_m_simd(batch: &mut CommandBatch<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const SIMD_TG: u32 = 32;
        const ROWS_PER_TG: u32 = 8;
        let n_tg = (rows_u32 + ROWS_PER_TG - 1) / ROWS_PER_TG;
        let shmem_bytes = 192u64 * std::mem::size_of::<f32>() as u64;
        batch.dispatch_threads("gemm_q4_k_m_fused_simd", (n_tg * SIMD_TG, 1, 1), (SIMD_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.3.4 -- low-level batched encoder for `gemv_f32_attn`.
    /// Takes pre-allocated Metal buffers; encodes into an existing CommandBatch
    /// without allocation or readback. Use this to coalesce two independent
    /// fp32 GEMVs (e.g. q_a_proj + kv_a_proj) into a single command buffer.
    pub(crate) fn encode_gemv_f32_attn_pinned(batch: &mut CommandBatch<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        batch.dispatch_threads("gemv_f32_attn", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.3.1 -- slice-in / slice-out wrapper: allocates Metal buffers, routes
    /// through `ctx.dispatch_batch { encode_gemv_q4_k_m_simd }`, reads back.
    /// Replaces the standalone `ctx.dispatch_threads` path in
    /// `moe_expert_matmul_dispatch` so simd GEMVs appear in the
    /// dispatch_batch profiling bucket and can later be coalesced.
    pub fn dispatch_gemv_q4_k_m_simd_batched(ctx: &MetalContext, w_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd weight bytes: got {} expected {}", w_bytes.len(), expected_bytes)));
        }
        let w_buf = ctx.new_buffer_with_bytes(w_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| encode_gemv_q4_k_m_simd(batch, &w_buf, rows, cols, &x_buf, &out_buf))?;
        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    /// v0.3.2 -- pair wrapper: allocates x once, encodes gate+up into ONE CommandBatch.
    /// Two Q4_K_M simd GEMVs (w_a, w_b) sharing the same input (x) and output
    /// dimensions coalesce into a single command-buffer commit instead of two.
    pub fn dispatch_gemv_q4_k_m_simd_pair_batched(ctx: &MetalContext, w_a_bytes: &[u8], w_b_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out_a: &mut [f32], out_b: &mut [f32]) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd pair requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out_a.len() != rows || out_b.len() != rows {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd pair shape: x={} cols={} out_a={} out_b={} rows={}", x.len(), cols, out_a.len(), out_b.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_a_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd pair w_a bytes: got {} expected {}", w_a_bytes.len(), expected_bytes)));
        }
        if w_b_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd pair w_b bytes: got {} expected {}", w_b_bytes.len(), expected_bytes)));
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

    /// v0.3.3 -- fused pair+silu: encode gate, up, and silu_mul in ONE CommandBatch.
    /// `a` receives `silu(gate_out) * up_out`; intermediate gate/up buffers stay
    /// on the GPU and are never read back.
    pub fn dispatch_gemv_q4_k_m_simd_pair_silu_batched(ctx: &MetalContext, w_gate_bytes: &[u8], w_up_bytes: &[u8], rows: usize, cols: usize, x: &[f32], a: &mut [f32]) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd pair+silu requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || a.len() != rows {
            return Err(Error::Kernel(format!("gemm_q4_k_m_fused_simd pair+silu shape: x={} cols={} a={} rows={}", x.len(), cols, a.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_gate_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("pair+silu w_gate bytes: got {} expected {}", w_gate_bytes.len(), expected_bytes)));
        }
        if w_up_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("pair+silu w_up bytes: got {} expected {}", w_up_bytes.len(), expected_bytes)));
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
            encode_gemv_q4_k_m_simd(batch, &w_up_buf, rows, cols, &x_buf, &u_buf)?;
            encode_silu_mul(batch, &g_buf, &u_buf, &a_buf, rows)
        })?;
        let ptr = a_buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, rows) };
        a.copy_from_slice(slice);
        Ok(())
    }

    //
    // Each function below is the seam the haul targets. The signature
    // and call-from-host expectations are locked: bodies arrive in
    // `_phase1_haul_manifest.md` G1.1 / G1.2 / G1.3 / G1.4. The haul
    // does NOT change these signatures; doing so would invalidate the
    // call sites in `model::deepseek_v2` and the parity tests in
    // `tests/phase1_kernel_parity.rs`.

    /// G1.1 -- RMSNorm via the existing `rmsnorm` kernel in
    /// `shaders/common.metal`. Inputs and outputs are fp32 from the
    /// caller's view; the kernel works in fp16 internally.
    ///
    /// Threadgroup size 256 (kernel uses parallel reduction; must be
    /// power of two ≤ 1024).
    pub fn rmsnorm_metal(ctx: &MetalContext, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        let hidden = x.len();
        if weight.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!("rmsnorm_metal shape mismatch: x={} weight={} out={}", hidden, weight.len(), out.len())));
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
            enc.set_u32(3, hidden_u32);
            enc.set_f32(4, eps_f32);
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

    /// G1.2 -- fp16 GEMV. Maps to a new `gemv_f16` kernel in
    /// `shaders/common.metal` (added during the haul). Used for the
    /// LM-head projection (vocab × hidden).
    ///
    /// Layout: `w` is row-major `(rows, cols)` fp16; `x` is fp32
    /// converted to fp16 inside the dispatch; `out` is fp32 from the
    /// kernel's fp16 result.
    pub fn gemv_f16_metal(ctx: &MetalContext, w_f16_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("gemv_f16_metal shape mismatch: x={} rows={} cols={} out={}", x.len(), rows, cols, out.len())));
        }
        let expected_w = rows * cols * std::mem::size_of::<f16>();
        if w_f16_bytes.len() != expected_w {
            return Err(Error::Kernel(format!("gemv_f16_metal weight bytes mismatch: got {} expected {}", w_f16_bytes.len(), expected_w)));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_f16_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        // Total threads = rows * tg_size; threadgroup = tg_size →
        // exactly one threadgroup per output row.
        ctx.dispatch_threads("gemv_f16", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

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
    /// allocated fresh per dispatch (small -- `rows * 4` bytes).
    pub fn gemv_f16_metal_pinned(ctx: &MetalContext, w_buf: &PinnedBuffer, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("gemv_f16_metal_pinned shape mismatch: x={} rows={} cols={} out={}", x.len(), rows, cols, out.len())));
        }
        let expected_w = (rows * cols * std::mem::size_of::<f16>()) as u64;
        if w_buf.length() < expected_w {
            return Err(Error::Kernel(format!("gemv_f16_metal_pinned weight buffer too small: got {} expected {}", w_buf.length(), expected_w)));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads("gemv_f16", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Profiled greedy decode primitive: LM-head GEMV followed by GPU
    /// argmax in one command buffer. Only the final token id is read
    /// back to the CPU.
    pub fn gemv_f16_argmax_metal_pinned(ctx: &MetalContext, w_buf: &PinnedBuffer, rows: usize, cols: usize, x: &[f32]) -> Result<u32> {
        if x.len() != cols {
            return Err(Error::Kernel(format!("gemv_f16_argmax_metal_pinned shape mismatch: x={} cols={}", x.len(), cols)));
        }
        if rows == 0 {
            return Err(Error::Kernel("gemv_f16_argmax_metal_pinned requires rows > 0".into()));
        }
        let expected_w = (rows * cols * std::mem::size_of::<f16>()) as u64;
        if w_buf.length() < expected_w {
            return Err(Error::Kernel(format!("gemv_f16_argmax_metal_pinned weight buffer too small: got {} expected {}", w_buf.length(), expected_w)));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let logits_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let token_buf = ctx.new_buffer(std::mem::size_of::<u32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads("gemv_f16", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(&x_buf), 0);
                enc.set_buffer(2, Some(&logits_buf), 0);
                enc.set_u32(3, rows_u32);
                enc.set_u32(4, cols_u32);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })?;
            // v0.5.7-A: parallel 256-thread argmax; needs threadgroup memory for
            // shmem_v (256 floats) and shmem_i (256 uints).
            batch.dispatch_threads("sample_argmax_f32", (256, 1, 1), (256, 1, 1), |enc| {
                enc.set_buffer(0, Some(&logits_buf), 0);
                enc.set_buffer(1, Some(&token_buf), 0);
                enc.set_u32(2, rows_u32);
                enc.set_threadgroup_memory_length(0, 256 * std::mem::size_of::<f32>() as u64);
                enc.set_threadgroup_memory_length(1, 256 * std::mem::size_of::<u32>() as u64);
            })?;
            Ok(())
        })?;

        let token_ptr = token_buf.contents() as *const u32;
        Ok(unsafe { *token_ptr })
    }

    /// G1.3 -- fp32 GEMV for attention's `o_proj`. Maps to a new
    /// `gemv_f32_attn` kernel in `shaders/attn.metal`. The model
    /// layer dequants per-call into a scratch buffer (lazy-dequant
    /// invariant from Phase 0); this kernel reads that scratch as
    /// fp32 weights.
    pub fn gemv_f32_attn_metal(ctx: &MetalContext, w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_gemv_f32(ctx, "gemv_f32_attn", w, rows, cols, x, out)
    }

    /// WB pinned variant of `gemv_f32_attn_metal`: takes a pre-uploaded
    /// `&PinnedBuffer` for the weight matrix instead of a host
    /// `&[f32]`. Eliminates the per-dispatch `new_buffer_with_bytes`
    /// memcpy for the 5 attention-projection gemvs (q_a_proj,
    /// q_b_proj, kv_a_proj_with_mqa, kv_b_proj, o_proj -- totaling
    /// ~50 MB / token in DeepSeek-V2-Lite at 27 layers).
    pub fn gemv_f32_attn_metal_pinned(ctx: &MetalContext, w_buf: &PinnedBuffer, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_gemv_f32_pinned(ctx, "gemv_f32_attn", w_buf, rows, cols, x, out)
    }

    /// v0.3.4 -- shared-input pair wrapper: coalesces two independent fp32 GEMVs
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
                x.len(),
                cols,
                out_a.len(),
                rows_a,
                out_b.len(),
                rows_b
            )));
        }
        let expected_a = (rows_a * cols * std::mem::size_of::<f32>()) as u64;
        if w_a_buf.length() < expected_a {
            return Err(Error::Kernel(format!("dispatch_gemv_f32_attn_pinned_pair w_a too small: got {} expected {}", w_a_buf.length(), expected_a)));
        }
        let expected_b = (rows_b * cols * std::mem::size_of::<f32>()) as u64;
        if w_b_buf.length() < expected_b {
            return Err(Error::Kernel(format!("dispatch_gemv_f32_attn_pinned_pair w_b too small: got {} expected {}", w_b_buf.length(), expected_b)));
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

    /// G1.4 -- fp32 GEMV for the MoE gate-logit projection
    /// (`ffn_gate_inp`). Maps to a new `gemv_f32_moe` kernel in
    /// `shaders/moe.metal`. Tiny (n_routed × hidden = 64 × 2048) but
    /// proves MoE-shaped weight access.
    pub fn gemv_f32_moe_metal(ctx: &MetalContext, w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        dispatch_gemv_f32(ctx, "gemv_f32_moe", w, rows, cols, x, out)
    }

    /// Phase 2 -- no-pack batched DeepSeek MoE block. The weight buffer is
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
            return Err(Error::Kernel("moe_block_batched_indexed_metal: no routes".into()));
        }
        if route_weights.len() != routes {
            return Err(Error::Kernel(format!("moe_block_batched_indexed_metal: {} route ids but {} weights", routes, route_weights.len())));
        }
        if x.len() != hidden || out.len() != hidden {
            return Err(Error::Kernel(format!("moe_block_batched_indexed_metal shape: x={} hidden={} out={}", x.len(), hidden, out.len())));
        }
        for &eid in route_ids {
            if eid as usize >= n_routed_experts {
                return Err(Error::Kernel(format!("moe_block_batched_indexed_metal: route expert {eid} >= {n_routed_experts}")));
            }
        }

        validate_indexed_quant("moe_block_batched_indexed routed_gate_q4", model_buf, routed_gate_offset, n_routed_experts, routed_mid, hidden, 256, 144)?;
        validate_indexed_quant("moe_block_batched_indexed routed_up_q4", model_buf, routed_up_offset, n_routed_experts, routed_mid, hidden, 256, 144)?;
        validate_indexed_quant("moe_block_batched_indexed routed_down_q8", model_buf, routed_down_offset, n_routed_experts, hidden, routed_mid, 32, 34)?;

        let has_shared = shared_gate_offset.is_some() || shared_up_offset.is_some() || shared_down_offset.is_some();
        if has_shared && !(shared_gate_offset.is_some() && shared_up_offset.is_some() && shared_down_offset.is_some()) {
            return Err(Error::Kernel("moe_block_batched_indexed_metal: shared offsets must be all Some or all None".into()));
        }
        if has_shared {
            validate_indexed_quant("moe_block_batched_indexed shared_gate_q4", model_buf, shared_gate_offset.unwrap(), 1, shared_mid, hidden, 256, 144)?;
            validate_indexed_quant("moe_block_batched_indexed shared_up_q4", model_buf, shared_up_offset.unwrap(), 1, shared_mid, hidden, 256, 144)?;
            validate_indexed_quant("moe_block_batched_indexed shared_down_q6", model_buf, shared_down_offset.unwrap(), 1, hidden, shared_mid, 256, 210)?;
        }

        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let route_weights_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let shared_route_ids = [0u32];
        let shared_route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(&shared_route_ids));
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
            encode_batched_gemv_indexed(batch, q4k_indexed_kernel, model_buf, &route_ids_buf, &x_buf, &routed_gate_out, routed_gate_offset, routes, routed_mid, hidden)?;
            encode_batched_gemv_indexed(batch, q4k_indexed_kernel, model_buf, &route_ids_buf, &x_buf, &routed_up_out, routed_up_offset, routes, routed_mid, hidden)?;
            encode_silu_mul(batch, &routed_gate_out, &routed_up_out, &routed_act, routes * routed_mid)?;
            encode_batched_gemv_indexed(batch, "moe_batched_gemm_q8_0_indexed", model_buf, &route_ids_buf, &routed_act, &routed_out, routed_down_offset, routes, hidden, routed_mid)?;

            if let (Some(gate_off), Some(up_off), Some(down_off)) = (shared_gate_offset, shared_up_offset, shared_down_offset) {
                encode_batched_gemv_indexed(batch, q4k_indexed_kernel, model_buf, &shared_route_ids_buf, &x_buf, &shared_gate_out, gate_off, 1, shared_mid, hidden)?;
                encode_batched_gemv_indexed(batch, q4k_indexed_kernel, model_buf, &shared_route_ids_buf, &x_buf, &shared_up_out, up_off, 1, shared_mid, hidden)?;
                encode_silu_mul(batch, &shared_gate_out, &shared_up_out, &shared_act, shared_mid)?;
                encode_batched_gemv_indexed(batch, "moe_batched_gemm_q6_k_indexed", model_buf, &shared_route_ids_buf, &shared_act, &shared_out, down_off, 1, hidden, shared_mid)?;
            }

            encode_route_accumulate(batch, &routed_out, &route_weights_buf, &shared_out, &final_out, hidden, routes, has_shared)
        })?;

        copy_f32_buffer(&final_out, out);
        Ok(())
    }

    /// Wedge 1 -- Metal MLA decode kernel.
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
            return Err(Error::Kernel(format!("mla_decode_metal: q.len={} expected {}", q.len(), n_heads * q_head_dim)));
        }
        if c_kv.len() != seq_len * kv_lora_rank {
            return Err(Error::Kernel(format!("mla_decode_metal: c_kv.len={} expected {}", c_kv.len(), seq_len * kv_lora_rank)));
        }
        if k_pe.len() != seq_len * qk_rope_head_dim {
            return Err(Error::Kernel(format!("mla_decode_metal: k_pe.len={} expected {}", k_pe.len(), seq_len * qk_rope_head_dim)));
        }
        let expected_kv_b = (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>()) as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!("mla_decode_metal: kv_b_proj buffer too small: got {} expected {}", kv_b_proj.length(), expected_kv_b)));
        }
        if out.len() != n_heads * v_head_dim {
            return Err(Error::Kernel(format!("mla_decode_metal: out.len={} expected {}", out.len(), n_heads * v_head_dim)));
        }
        if seq_len == 0 {
            return Err(Error::Kernel("mla_decode_metal: seq_len must be >= 1".into()));
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
        //   0 -- q_nope_proj: kv_lora_rank floats
        //   1 -- scores:      seq_len floats
        //   2 -- c_kv_wt:     kv_lora_rank floats
        let q_nope_proj_bytes = (kv_lora_rank as u64) * std::mem::size_of::<f32>() as u64;
        let scores_bytes = (seq_len as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads("mla_decode_kernel", (n_heads_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&c_kv_buf), 0);
            enc.set_buffer(2, Some(&k_pe_buf), 0);
            enc.set_buffer(3, Some(kv_b_proj), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_u32(5, n_heads_u32);
            enc.set_u32(6, qk_nope_u32);
            enc.set_u32(7, qk_rope_u32);
            enc.set_u32(8, v_head_u32);
            enc.set_u32(9, kv_lora_u32);
            enc.set_u32(10, seq_len_u32);
            enc.set_f32(11, scale);
            enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
            enc.set_threadgroup_memory_length(1, scores_bytes);
            enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
        })?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Q8 KV variant of `mla_decode_metal`.
    ///
    /// Same semantics, same I/O — except `c_kv_q8` is the Q8_0-packed
    /// latent cache instead of `&[f32]`. Per-row byte count must equal
    /// `(kv_lora_rank / 32) * 34`. `kv_lora_rank` must be a multiple of 32.
    ///
    /// k_pe stays f32 — positional-embedding precision matters and the
    /// bandwidth contribution is small (qk_rope_head_dim ≪ kv_lora_rank).
    ///
    /// The match for the f32 `mla_decode_metal` is ATOL ~5e-3 (per-block
    /// f16 scale + round-to-nearest int8 introduces bounded error).
    #[allow(clippy::too_many_arguments)]
    pub fn mla_decode_q8kv_metal(
        ctx: &MetalContext,
        q: &[f32],
        c_kv_q8: &[u8],
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
        if kv_lora_rank % 32 != 0 {
            return Err(Error::Kernel(format!("mla_decode_q8kv_metal: kv_lora_rank {kv_lora_rank} not multiple of 32")));
        }
        let row_bytes = (kv_lora_rank / 32) * 34;
        let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
        if q.len() != n_heads * q_head_dim {
            return Err(Error::Kernel(format!("mla_decode_q8kv_metal: q.len={} expected {}", q.len(), n_heads * q_head_dim)));
        }
        if c_kv_q8.len() < seq_len * row_bytes {
            return Err(Error::Kernel(format!("mla_decode_q8kv_metal: c_kv_q8 len={} need at least {}", c_kv_q8.len(), seq_len * row_bytes)));
        }
        if k_pe.len() != seq_len * qk_rope_head_dim {
            return Err(Error::Kernel(format!("mla_decode_q8kv_metal: k_pe.len={} expected {}", k_pe.len(), seq_len * qk_rope_head_dim)));
        }
        let expected_kv_b = (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>()) as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!("mla_decode_q8kv_metal: kv_b_proj buffer too small: got {} expected {}", kv_b_proj.length(), expected_kv_b)));
        }
        if out.len() != n_heads * v_head_dim {
            return Err(Error::Kernel(format!("mla_decode_q8kv_metal: out.len={} expected {}", out.len(), n_heads * v_head_dim)));
        }
        if seq_len == 0 {
            return Err(Error::Kernel("mla_decode_q8kv_metal: seq_len must be >= 1".into()));
        }

        let q_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(q));
        let c_kv_buf = ctx.new_buffer_with_bytes(c_kv_q8);
        let k_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(k_pe));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());

        let n_heads_u32 = n_heads as u32;
        let qk_nope_u32 = qk_nope_head_dim as u32;
        let qk_rope_u32 = qk_rope_head_dim as u32;
        let v_head_u32 = v_head_dim as u32;
        let kv_lora_u32 = kv_lora_rank as u32;
        let seq_len_u32 = seq_len as u32;

        let q_nope_proj_bytes = (kv_lora_rank as u64) * std::mem::size_of::<f32>() as u64;
        let scores_bytes = (seq_len as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads("mla_decode_kernel_q8kv", (n_heads_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&c_kv_buf), 0);
            enc.set_buffer(2, Some(&k_pe_buf), 0);
            enc.set_buffer(3, Some(kv_b_proj), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_u32(5, n_heads_u32);
            enc.set_u32(6, qk_nope_u32);
            enc.set_u32(7, qk_rope_u32);
            enc.set_u32(8, v_head_u32);
            enc.set_u32(9, kv_lora_u32);
            enc.set_u32(10, seq_len_u32);
            enc.set_f32(11, scale);
            enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
            enc.set_threadgroup_memory_length(1, scores_bytes);
            enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
        })?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// One-token GPU-side Q8_0 quantize-and-append for the latent KV cache.
    ///
    /// This is the standalone "single token in, Q8 bytes out" path used by
    /// parity tests. Production code calls the TCB variant
    /// `kv_append_q8_0_f32_tcb` (see below) to chain into a multi-kernel
    /// command buffer. Both go through the same `kv_append_q8_0_f32` shader.
    ///
    /// `c_kv_normed` (kv_lora_rank f32) is quantized to Q8_0 and written to
    /// `dst_c_kv_q8` at slot `seq_slot`. `kv_a_out[kv_lora_rank..]` (the
    /// k_pe slice) is copied verbatim to `dst_k_pe` at slot `seq_slot`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_q8_0_f32_metal(
        ctx: &MetalContext,
        c_kv_normed: &[f32],
        kv_a_out: &[f32],
        dst_c_kv_q8: &mut [u8],
        dst_k_pe: &mut [f32],
        seq_slot: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        max_seq: usize,
    ) -> Result<()> {
        if kv_lora_rank % 32 != 0 {
            return Err(Error::Kernel(format!("kv_append_q8_0_f32: kv_lora_rank {kv_lora_rank} not multiple of 32")));
        }
        let n_blocks = kv_lora_rank / 32;
        let row_bytes = n_blocks * 34;
        if c_kv_normed.len() != kv_lora_rank {
            return Err(Error::Kernel(format!("kv_append_q8_0_f32: c_kv_normed.len={} expected {}", c_kv_normed.len(), kv_lora_rank)));
        }
        if kv_a_out.len() < kv_lora_rank + qk_rope_head_dim {
            return Err(Error::Kernel(format!("kv_append_q8_0_f32: kv_a_out.len={} need {}", kv_a_out.len(), kv_lora_rank + qk_rope_head_dim)));
        }
        if dst_c_kv_q8.len() < max_seq * row_bytes {
            return Err(Error::Kernel(format!("kv_append_q8_0_f32: dst_c_kv_q8.len={} need {}", dst_c_kv_q8.len(), max_seq * row_bytes)));
        }
        if dst_k_pe.len() < max_seq * qk_rope_head_dim {
            return Err(Error::Kernel(format!("kv_append_q8_0_f32: dst_k_pe.len={} need {}", dst_k_pe.len(), max_seq * qk_rope_head_dim)));
        }
        if seq_slot >= max_seq {
            return Err(Error::Kernel(format!("kv_append_q8_0_f32: seq_slot {seq_slot} >= max_seq {max_seq}")));
        }

        let src_c_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(c_kv_normed));
        let src_kv_a_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(kv_a_out));
        let dst_c_buf = ctx.new_buffer_with_bytes(dst_c_kv_q8);
        let dst_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(dst_k_pe));

        // Argbuf: { seq_slot, kv_lora_rank, qk_rope_head_dim } as 3 packed u32s.
        let args: [u32; 3] = [seq_slot as u32, kv_lora_rank as u32, qk_rope_head_dim as u32];

        let absmax_bytes = 32u64 * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("kv_append_q8_0_f32", (n_blocks as u32 * 32, 1, 1), (32, 1, 1), |enc| {
            enc.set_buffer(0, Some(&src_c_buf), 0);
            enc.set_buffer(1, Some(&src_kv_a_buf), 0);
            enc.set_buffer(2, Some(&dst_c_buf), 0);
            enc.set_buffer(3, Some(&dst_pe_buf), 0);
            enc.set_bytes(4, std::mem::size_of::<[u32; 3]>() as u64, args.as_ptr() as *const _);
            enc.set_threadgroup_memory_length(0, absmax_bytes);
        })?;

        copy_u8_buffer(&dst_c_buf, dst_c_kv_q8);
        copy_f32_buffer(&dst_pe_buf, dst_k_pe);
        Ok(())
    }

    /// Wedge L -- flash attention decode using online softmax (MLA-aware).
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
            return Err(Error::Kernel(format!("flash_attn_decode_metal: q.len={} expected {}", q.len(), n_heads * q_head_dim)));
        }
        if c_kv.len() != seq_len * kv_lora_rank {
            return Err(Error::Kernel(format!("flash_attn_decode_metal: c_kv.len={} expected {}", c_kv.len(), seq_len * kv_lora_rank)));
        }
        if k_pe.len() != seq_len * qk_rope_head_dim {
            return Err(Error::Kernel(format!("flash_attn_decode_metal: k_pe.len={} expected {}", k_pe.len(), seq_len * qk_rope_head_dim)));
        }
        let expected_kv_b = (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>()) as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!("flash_attn_decode_metal: kv_b_proj buffer too small: got {} expected {}", kv_b_proj.length(), expected_kv_b)));
        }
        if out.len() != n_heads * v_head_dim {
            return Err(Error::Kernel(format!("flash_attn_decode_metal: out.len={} expected {}", out.len(), n_heads * v_head_dim)));
        }
        if seq_len == 0 {
            return Err(Error::Kernel("flash_attn_decode_metal: seq_len must be >= 1".into()));
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

        ctx.dispatch_threads("flash_attn_decode_kernel", (n_heads_u32 * FLASH_TG, 1, 1), (FLASH_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&q_buf), 0);
            enc.set_buffer(1, Some(&c_kv_buf), 0);
            enc.set_buffer(2, Some(&k_pe_buf), 0);
            enc.set_buffer(3, Some(kv_b_proj), 0);
            enc.set_buffer(4, Some(&out_buf), 0);
            enc.set_u32(5, n_heads_u32);
            enc.set_u32(6, qk_nope_u32);
            enc.set_u32(7, qk_rope_u32);
            enc.set_u32(8, v_head_u32);
            enc.set_u32(9, kv_lora_u32);
            enc.set_u32(10, seq_len_u32);
            enc.set_f32(11, scale);
            enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
            enc.set_threadgroup_memory_length(1, acc_bytes);
            enc.set_threadgroup_memory_length(2, scores_tile_bytes);
            enc.set_threadgroup_memory_length(3, state_bytes);
        })?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Wedge 3 -- Layer-CB: batch mla_decode_kernel + gemv_f32_attn (o_proj)
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
            return Err(Error::Kernel(format!("mla_decode_and_o_proj_metal: q.len={} expected {}", q.len(), n_heads * q_head_dim)));
        }
        if seq_len == 0 {
            return Err(Error::Kernel("mla_decode_and_o_proj_metal: seq_len must be >= 1".into()));
        }
        if out.len() != hidden {
            return Err(Error::Kernel(format!("mla_decode_and_o_proj_metal: out.len={} expected hidden={}", out.len(), hidden)));
        }
        let expected_kv_b = (n_heads * (qk_nope_head_dim + v_head_dim) * kv_lora_rank * std::mem::size_of::<f32>()) as u64;
        if kv_b_proj.length() < expected_kv_b {
            return Err(Error::Kernel(format!("mla_decode_and_o_proj_metal: kv_b_proj too small: {} < {}", kv_b_proj.length(), expected_kv_b)));
        }
        let expected_o_proj = (hidden * o_proj_cols * std::mem::size_of::<f32>()) as u64;
        if o_proj.length() < expected_o_proj {
            return Err(Error::Kernel(format!("mla_decode_and_o_proj_metal: o_proj too small: {} < {}", o_proj.length(), expected_o_proj)));
        }

        let q_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(q));
        let c_kv_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(c_kv));
        let k_pe_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(k_pe));
        // Intermediate attn_out stays in GPU memory -- shared between mla_decode and o_proj.
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
            batch.dispatch_threads("mla_decode_kernel", (n_heads_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(&q_buf), 0);
                enc.set_buffer(1, Some(&c_kv_buf), 0);
                enc.set_buffer(2, Some(&k_pe_buf), 0);
                enc.set_buffer(3, Some(kv_b_proj), 0);
                enc.set_buffer(4, Some(&attn_out_buf), 0);
                enc.set_u32(5, n_heads_u32);
                enc.set_u32(6, qk_nope_u32);
                enc.set_u32(7, qk_rope_u32);
                enc.set_u32(8, v_head_u32);
                enc.set_u32(9, kv_lora_u32);
                enc.set_u32(10, seq_len_u32);
                enc.set_f32(11, scale);
                enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                enc.set_threadgroup_memory_length(1, scores_bytes);
                enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
            })?;

            // Kernel 2: gemv_f32_attn (o_proj) -- reads attn_out_buf, writes out_buf.
            // Metal serializes these within the command buffer.
            batch.dispatch_threads("gemv_f32_attn", (hidden_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(o_proj), 0);
                enc.set_buffer(1, Some(&attn_out_buf), 0);
                enc.set_buffer(2, Some(&out_buf), 0);
                enc.set_u32(3, hidden_u32);
                enc.set_u32(4, o_proj_cols_u32);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })?;

            Ok(())
        })?;

        copy_f32_buffer(&out_buf, out);
        Ok(())
    }

    /// Wedge 4 -- Decode-Arena variant of `mla_decode_and_o_proj_metal`.
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
            return Err(Error::Kernel("mla_decode_and_o_proj_arena_metal: seq_len must be >= 1".into()));
        }
        if out.len() != hidden {
            return Err(Error::Kernel(format!("mla_decode_and_o_proj_arena_metal: out.len={} != hidden={}", out.len(), hidden)));
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
            batch.dispatch_threads("mla_decode_kernel", (n_heads_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(&arena.q), 0);
                enc.set_buffer(1, Some(&arena.c_kv), 0);
                enc.set_buffer(2, Some(&arena.k_pe), 0);
                enc.set_buffer(3, Some(kv_b_proj), 0);
                enc.set_buffer(4, Some(&arena.attn_out), 0);
                enc.set_u32(5, n_heads_u32);
                enc.set_u32(6, qk_nope_u32);
                enc.set_u32(7, qk_rope_u32);
                enc.set_u32(8, v_head_u32);
                enc.set_u32(9, kv_lora_u32);
                enc.set_u32(10, seq_len_u32);
                enc.set_f32(11, scale);
                enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
                enc.set_threadgroup_memory_length(1, scores_bytes);
                enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
            })?;

            batch.dispatch_threads("gemv_f32_attn", (hidden_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
                enc.set_buffer(0, Some(o_proj), 0);
                enc.set_buffer(1, Some(&arena.attn_out), 0);
                enc.set_buffer(2, Some(&arena.out), 0);
                enc.set_u32(3, hidden_u32);
                enc.set_u32(4, o_proj_cols_u32);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })?;

            Ok(())
        })?;

        arena.read_out(out);
        Ok(())
    }

    fn validate_indexed_quant(name: &str, model_buf: &PinnedBuffer, base_offset: usize, matrices: usize, rows: usize, cols: usize, block_elems: usize, block_bytes: usize) -> Result<()> {
        if matrices == 0 {
            return Err(Error::Kernel(format!("{name}: matrices must be > 0")));
        }
        if cols % block_elems != 0 {
            return Err(Error::Kernel(format!("{name}: cols must be multiple of {block_elems}; got {cols}")));
        }
        let expected =
            matrices.checked_mul(rows).and_then(|v| v.checked_mul(cols / block_elems)).and_then(|v| v.checked_mul(block_bytes)).ok_or_else(|| Error::Kernel(format!("{name}: byte-size overflow")))?;
        let end = base_offset.checked_add(expected).ok_or_else(|| Error::Kernel(format!("{name}: byte-range overflow")))?;
        if end as u64 > model_buf.length() {
            return Err(Error::Kernel(format!("{name}: byte range [{base_offset}, {end}) exceeds model buffer {}", model_buf.length())));
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
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| encode_batched_gemv_indexed(batch, "moe_batched_gemm_q4_indexed_v2t", &model_buf, &route_ids_buf, &x_buf, &out_buf, base_offset, routes, rows, cols))?;
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
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64 = up_offset as u64;
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let n_tg_x = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads("moe_batched_gemm_q4_indexed_v2t_gu", (n_tg_x * tg_size, routes_u32, 1), (tg_size, 1, 1), |enc| {
                enc.set_buffer(0, Some(&model_buf), 0);
                enc.set_buffer(1, Some(&route_ids_buf), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&out_buf), 0);
                enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64 as *const u64 as *const _);
                enc.set_u32(6, routes_u32);
                enc.set_u32(7, rows_u32);
                enc.set_u32(8, cols_u32);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })
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
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        let gate_offset_u64 = gate_offset as u64;
        let up_offset_u64 = up_offset as u64;
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let n_tg_x = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_batch(|batch| {
            batch.dispatch_threads("moe_batched_gemm_q4_indexed_v2t_gu_v2", (n_tg_x * tg_size, routes_u32, 1), (tg_size, 1, 1), |enc| {
                enc.set_buffer(0, Some(&model_buf), 0);
                enc.set_buffer(1, Some(&route_ids_buf), 0);
                enc.set_buffer(2, Some(&x_buf), 0);
                enc.set_buffer(3, Some(&out_buf), 0);
                enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64 as *const u64 as *const _);
                enc.set_u32(6, routes_u32);
                enc.set_u32(7, rows_u32);
                enc.set_u32(8, cols_u32);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })
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
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| encode_batched_gemv_indexed(batch, "moe_batched_gemm_q4_indexed_v2s", &model_buf, &route_ids_buf, &x_buf, &out_buf, base_offset, routes, rows, cols))?;
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
        let kernel_name = if use_v2 { "moe_batched_gemm_q4_indexed_v2" } else { "moe_batched_gemm_q4_indexed" };
        let model_buf = ctx.new_buffer_with_bytes(w_all_bytes);
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| encode_batched_gemv_indexed(batch, kernel_name, &model_buf, &route_ids_buf, &x_buf, &out_buf, base_offset, routes, rows, cols))?;
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
        let route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<u32, u8>(route_ids));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(out.len() * std::mem::size_of::<f32>());
        ctx.dispatch_batch(|batch| encode_batched_gemv_indexed(batch, "moe_batched_gemm_q8_0_indexed_v2t", &model_buf, &route_ids_buf, &x_buf, &out_buf, base_offset, routes, rows, cols))?;
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

        batch.dispatch_threads(kernel_name, (n_tg_x * tg_size, routes_u32, 1), (tg_size, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), 0);
            enc.set_buffer(1, Some(route_ids_buf), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &base_offset_u64 as *const u64 as *const _);
            enc.set_u32(5, routes_u32);
            enc.set_u32(6, rows_u32);
            enc.set_u32(7, cols_u32);
            if !is_v2_family || is_v2t {
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            }
        })
    }

    fn encode_silu_mul(batch: &mut CommandBatch<'_>, gate_buf: &PinnedBuffer, up_buf: &PinnedBuffer, out_buf: &PinnedBuffer, n: usize) -> Result<()> {
        let n_u32 = n as u32;
        batch.dispatch_threads("moe_batched_silu_mul", (n_u32, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(gate_buf), 0);
            enc.set_buffer(1, Some(up_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, n_u32);
        })
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
        batch.dispatch_threads("moe_route_accumulate", (hidden_u32, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(routed_out), 0);
            enc.set_buffer(1, Some(weights), 0);
            enc.set_buffer(2, Some(shared_out), 0);
            enc.set_buffer(3, Some(out), 0);
            enc.set_u32(4, hidden_u32);
            enc.set_u32(5, routes_u32);
            enc.set_u32(6, has_shared_u32);
        })
    }

    fn copy_f32_buffer(buf: &PinnedBuffer, out: &mut [f32]) {
        let ptr = buf.contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, out.len()) };
        out.copy_from_slice(slice);
    }

    fn copy_u8_buffer(buf: &PinnedBuffer, out: &mut [u8]) {
        let ptr = buf.contents() as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr, out.len()) };
        out.copy_from_slice(slice);
    }

    // Shared dispatch for the two Q4_K_M-fused GEMV kernels (H2.2 in
    // moe.metal, H2.4 in quant.metal). Same kernel body in both files;
    // only the function name differs because the manifest split puts
    // them in different shader modules. tg_size hardcoded to 256
    // (matches the Q4_K_M super-block size -- see kernel comments).
    fn dispatch_q4_k_m_gemv(ctx: &MetalContext, kernel_name: &str, w_q4_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{kernel_name} requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{kernel_name} shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_q4_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("{kernel_name} weight bytes: got {} expected {}", w_q4_bytes.len(), expected_bytes)));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(kernel_name, (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // v0.4.0 -- v2 dispatch: 256-thread TG, 8 rows per TG (8 simdgroups),
    // simd_sum reduction.  No threadgroup memory needed.
    fn dispatch_q4_k_m_gemv_v2(ctx: &MetalContext, kernel_name: &str, w_q4_bytes: &[u8], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{kernel_name} requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{kernel_name} shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_q4_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("{kernel_name} weight bytes: got {} expected {}", w_q4_bytes.len(), expected_bytes)));
        }

        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;

        ctx.dispatch_threads(kernel_name, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
            // NO set_threadgroup_memory_length -- kernel uses none.
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // Wedge A -- pinned-buffer variant of dispatch_q4_k_m_gemv_v2. Uses set_buffer
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
            return Err(Error::Kernel(format!("{kernel_name}_pinned requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{kernel_name}_pinned shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{kernel_name}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{kernel_name}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;

        ctx.dispatch_threads(kernel_name, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
            // NO set_threadgroup_memory_length -- kernel uses none.
        })?;

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
            return Err(Error::Kernel(format!("{kernel_name}_pinned requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{kernel_name}_pinned shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 110;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{kernel_name}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{kernel_name}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        const V2_TG: u32 = 256;
        let n_tg = (rows_u32 + 7) / 8;

        ctx.dispatch_threads(kernel_name, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // Wedge K dispatcher -- gemm_q4_k_m_simdmat geometry: 128 threads per TG
    // (4 simdgroups × 32), 4 rows per TG, grid=(ceil(rows/4)*128, 1, 1).
    fn dispatch_q4_k_m_simdmat_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_simdmat";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{KERNEL}_pinned shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const SM_TG: u32 = 128; // 4 simdgroups × 32 threads
        const SM_ROWS: u32 = 4; // 1 simdgroup per row, 4 rows per TG
        let n_tg = (rows_u32 + SM_ROWS - 1) / SM_ROWS;

        ctx.dispatch_threads(KERNEL, (n_tg * SM_TG, 1, 1), (SM_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // Wedge K Approach 1 Iter 1 -- v3_8r: 256 threads per TG (8 simdgroups),
    // 8 rows per TG, grid=(ceil(rows/8)*256, 1, 1).
    fn dispatch_q4_k_m_v3_8r_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_8r";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{KERNEL}_pinned shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = (rows_u32 + V3_ROWS - 1) / V3_ROWS;

        ctx.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // Wedge K Approach 1 Iter 2 -- v3_dual: 128 threads per TG (4 simdgroups),
    // 2 rows per simdgroup (N_R0=2), 8 rows per TG.
    // grid=(ceil(rows/8)*128, 1, 1). Amortizes activation load over 2 rows.
    fn dispatch_q4_k_m_v3_dual_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_dual";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{KERNEL}_pinned shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const DUAL_TG: u32 = 128;
        const DUAL_ROWS: u32 = 8;
        let n_tg = (rows_u32 + DUAL_ROWS - 1) / DUAL_ROWS;

        ctx.dispatch_threads(KERNEL, (n_tg * DUAL_TG, 1, 1), (DUAL_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // Approach 3 -- v3_llama: 64 threads per TG (2 simdgroups), 4 rows per
    // simdgroup (N_R0=4), sumy trick for min correction.
    // grid=(ceil(rows/8)*64, 1, 1). Faithful llama.cpp port.
    fn dispatch_q4_k_m_v3_llama_pinned(ctx: &MetalContext, model_buf: &PinnedBuffer, w_offset: usize, w_byte_size: usize, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_llama";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned requires cols % 256 == 0; got cols={cols}")));
        }
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{KERNEL}_pinned shape: x={} cols={} out={} rows={}", x.len(), cols, out.len(), rows)));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        if w_offset + w_byte_size > model_buf.length() as usize {
            return Err(Error::Kernel(format!("{KERNEL}_pinned offset out of bounds: {w_offset}+{w_byte_size} > {}", model_buf.length())));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const LLAMA_TG: u32 = 64; // 2 simdgroups × 32 threads
        const LLAMA_ROWS: u32 = 8; // 2 simdgroups × 4 rows each
        let n_tg = (rows_u32 + LLAMA_ROWS - 1) / LLAMA_ROWS;

        ctx.dispatch_threads(KERNEL, (n_tg * LLAMA_TG, 1, 1), (LLAMA_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);
        Ok(())
    }

    // WB shared pinned dispatch for the f32 GEMV kernels. Same kernel
    // signature as the byte-slice path; only the weight upload changes
    // (pre-uploaded Buffer instead of fresh `new_buffer_with_bytes`).
    fn dispatch_gemv_f32_pinned(ctx: &MetalContext, kernel_name: &str, w_buf: &PinnedBuffer, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{kernel_name}_pinned shape mismatch: x={} rows={} cols={} out={}", x.len(), rows, cols, out.len())));
        }
        let expected_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;
        if w_buf.length() < expected_bytes {
            return Err(Error::Kernel(format!("{kernel_name}_pinned weight buffer too small: got {} expected {}", w_buf.length(), expected_bytes)));
        }

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(kernel_name, (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    // Shared dispatch for the two f32 GEMV variants (attn o_proj, moe gate
    // logits). Same kernel body in their respective shader files; only the
    // function name differs because the manifest splits them across
    // shaders/{attn,moe}.metal as separate gates.
    fn dispatch_gemv_f32(ctx: &MetalContext, kernel_name: &str, w: &[f32], rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        if x.len() != cols || out.len() != rows {
            return Err(Error::Kernel(format!("{kernel_name} shape mismatch: x={} rows={} cols={} out={}", x.len(), rows, cols, out.len())));
        }
        if w.len() != rows * cols {
            return Err(Error::Kernel(format!("{kernel_name} weight len mismatch: got {} expected {}", w.len(), rows * cols)));
        }

        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let args = ArgbufRowsCols { rows: rows_u32, cols: cols_u32 };
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;

        ctx.dispatch_threads(kernel_name, (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&x_buf), 0);
            enc.set_buffer(2, Some(&out_buf), 0);
            enc.set_bytes(3, std::mem::size_of::<ArgbufRowsCols>() as u64, &args as *const ArgbufRowsCols as *const _);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        let out_ptr = out_buf.contents() as *const f32;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        out.copy_from_slice(out_slice);

        Ok(())
    }

    /// Wedge B -- TCB variant of add_inplace_metal. Encodes into `tcb` without
    /// committing. Caller commits when a batch boundary is appropriate.
    pub fn add_inplace_metal_tcb(tcb: &mut TokenCommandBuffer<'_>, a_buf: &PinnedBuffer, b_buf: &PinnedBuffer, n: usize) -> Result<()> {
        let n_u32 = n as u32;
        let n_tg = (n_u32 + TG_SIZE - 1) / TG_SIZE;
        tcb.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(a_buf), 0);
            enc.set_buffer(1, Some(b_buf), 0);
            enc.set_u32(2, n_u32);
        })
    }

    // ── v0.5.6 buffer-arg dispatcher siblings ─────────────────────────────
    //
    // Each function below is a "buf" sibling of an existing dispatcher.
    // The difference: callers pass pre-existing Metal Buffers instead of
    // having the dispatcher allocate per-call. Same kernel, same binding
    // scheme -- only the buffer-allocation boilerplate is removed.

    /// v0.5.6 -- buffer-arg sibling of `rmsnorm_metal`.
    /// Takes pre-existing f16 Metal Buffers; skips the Vec→Buffer round-trip.
    /// Same kernel `"rmsnorm"`, same binding scheme (buf0=x, buf1=weight,
    /// buf2=out, bytes3=hidden, bytes4=eps, tg0=shmem).
    pub fn rmsnorm_metal_buf(ctx: &MetalContext, x_buf: &PinnedBuffer, weight_buf: &PinnedBuffer, eps: f32, hidden: usize, out_buf: &PinnedBuffer) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("rmsnorm", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, hidden_u32);
            enc.set_f32(4, eps);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Wedge B -- TCB variant of rmsnorm for the f32 residual stream.
    /// Uses `"rmsnorm_f32"` kernel (f32 x, f32 weight → f32 out). Encodes into
    /// `tcb` without committing. Caller commits when a batch boundary is appropriate.
    pub fn rmsnorm_metal_buf_tcb(tcb: &mut TokenCommandBuffer<'_>, x_buf: &PinnedBuffer, weight_buf: &PinnedBuffer, eps: f32, hidden: usize, out_buf: &PinnedBuffer) -> Result<()> {
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

    /// Session F (sketch) — fused add_inplace + rmsnorm_f32 dispatcher.
    ///
    /// Replaces the back-to-back pair
    /// `add_inplace_metal_tcb(&x, &attn_out)` + `rmsnorm_metal_buf_tcb(&x, w, eps, h, &x_norm)`
    /// with a single dispatch of `add_rmsnorm_fused`.
    ///
    /// Effect on x_buf: same as the unfused pair (x += attn_out, then x_norm = norm(x)).
    /// Eliminates one dispatch and one full DRAM pass over `x`.
    ///
    /// Opt-in only: gated behind the `HAWKING_FUSED_ADD_RMSNORM` env var at
    /// call sites in `deepseek_v2.rs`. Parity test:
    /// `tests/rmsnorm_fused_parity.rs`.
    pub fn add_rmsnorm_fused_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        attn_out_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("add_rmsnorm_fused", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(attn_out_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_buffer(3, Some(x_norm_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// W4A8 fusion (2026-05-24): single-dispatch version of
    /// `add_rmsnorm_fused_tcb` + `quantize_f32_to_int8_per_block_tcb`. Same
    /// add+rmsnorm semantics, also writes per-256-block int8 + f32 scales
    /// of the normalized output.
    ///
    /// Replaces two dispatches per layer × 2 sites per layer = 72 dispatches
    /// per decode token on Qwen-3B (36 layers). Bit-identical to the unfused
    /// pair (parity test: `tests/add_rmsnorm_fused_q8_parity.rs`).
    ///
    /// Requires `hidden % 256 == 0`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_fused_q8_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        attn_out_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        x_norm_int8_buf: &PinnedBuffer,
        x_norm_scales_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()> {
        if hidden % 256 != 0 {
            return Err(Error::Kernel(format!("add_rmsnorm_fused_q8_tcb requires hidden % 256 == 0; got hidden={hidden}")));
        }
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("add_rmsnorm_fused_q8", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(attn_out_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_buffer(3, Some(x_norm_buf), 0);
            enc.set_buffer(4, Some(x_norm_int8_buf), 0);
            enc.set_buffer(5, Some(x_norm_scales_buf), 0);
            enc.set_buffer(6, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// AWQ Option B variant of `add_rmsnorm_fused_q8_tcb`. Same residual+norm+
    /// int8-quantize fusion but the phase-3 quantize divides each `x_norm`
    /// element by the matching entry of a per-channel smoothing vector `s_buf`
    /// (length `hidden`) before computing the per-block scale. The stored
    /// `x_norm` is unchanged so f32 fallback consumers still see the canonical
    /// normalized activation.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_fused_q8_scaled_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        attn_out_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        x_norm_int8_buf: &PinnedBuffer,
        x_norm_scales_buf: &PinnedBuffer,
        s_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()> {
        if hidden % 256 != 0 {
            return Err(Error::Kernel(format!("add_rmsnorm_fused_q8_scaled_tcb requires hidden % 256 == 0; got hidden={hidden}")));
        }
        let f32_bytes = (hidden * std::mem::size_of::<f32>()) as u64;
        if s_buf.length() < f32_bytes {
            return Err(Error::Kernel(format!("add_rmsnorm_fused_q8_scaled_tcb s_buf too small: got {} need {}", s_buf.length(), f32_bytes,)));
        }
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("add_rmsnorm_fused_q8_scaled", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(attn_out_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_buffer(3, Some(x_norm_buf), 0);
            enc.set_buffer(4, Some(x_norm_int8_buf), 0);
            enc.set_buffer(5, Some(x_norm_scales_buf), 0);
            enc.set_buffer(6, Some(s_buf), 0);
            enc.set_buffer(7, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.5.6 -- buffer-arg variant of the f16 silu_mul kernel.
    /// Takes pre-existing f16 Metal Buffers. Kernel `"silu_mul"` in
    /// common.metal: out[i] = silu(gate[i]) * up[i], f16 I/O, f32 internal.
    pub fn silu_mul_metal_buf(ctx: &MetalContext, gate_buf: &PinnedBuffer, up_buf: &PinnedBuffer, out_buf: &PinnedBuffer, n: usize) -> Result<()> {
        let n_u32 = n as u32;
        let n_tg = (n_u32 + TG_SIZE - 1) / TG_SIZE;
        ctx.dispatch_threads("silu_mul", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(gate_buf), 0);
            enc.set_buffer(1, Some(up_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, n_u32);
        })
    }

    // add_inplace_metal_buf: SKIPPED -- existing `add_inplace_metal` already
    // takes PinnedBuffer args (it IS the buf variant). No wrapper needed.

    /// v0.5.6 -- buffer-arg sibling of `gemv_f32_attn_metal`.
    /// `w` is still a host slice (allocates a temp buffer); `x_buf` and
    /// `y_buf` are pre-existing Metal Buffers. Same kernel `"gemv_f32_attn"`.
    pub fn gemv_f32_attn_metal_buf(ctx: &MetalContext, w: &[f32], rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        if w.len() != rows * cols {
            return Err(Error::Kernel(format!("gemv_f32_attn_metal_buf weight len mismatch: got {} expected {}", w.len(), rows * cols)));
        }
        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("gemv_f32_attn", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.5.6 -- buffer-arg sibling of `gemv_f32_attn_metal_pinned`.
    /// All three matrix buffers are pre-existing; no allocation inside.
    /// Same kernel `"gemv_f32_attn"`.
    pub fn gemv_f32_attn_metal_pinned_buf(ctx: &MetalContext, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("gemv_f32_attn", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.5.6 -- buffer-arg sibling of `dispatch_gemv_f32_attn_pinned_pair_batched`.
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

    /// v0.5.6 -- buffer-arg sibling of `gemv_f32_moe_metal`.
    /// `w` is still a host slice (allocates a temp buffer); `x_buf` and
    /// `y_buf` are pre-existing Metal Buffers. Same kernel `"gemv_f32_moe"`.
    pub fn gemv_f32_moe_metal_buf(ctx: &MetalContext, w: &[f32], rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        if w.len() != rows * cols {
            return Err(Error::Kernel(format!("gemv_f32_moe_metal_buf weight len mismatch: got {} expected {}", w.len(), rows * cols)));
        }
        let w_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("gemv_f32_moe", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v0.5.6 -- buffer-arg sibling of `moe_grouped_gemm_q4_metal`.
    /// `w_q4_bytes` is still a host slice (allocates a temp buffer);
    /// `x_buf` and `y_buf` are pre-existing Metal Buffers.
    /// Same kernel `"moe_grouped_gemm_q4"`.
    pub fn moe_grouped_gemm_q4_metal_buf(ctx: &MetalContext, w_q4_bytes: &[u8], rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("moe_grouped_gemm_q4_metal_buf requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows * blocks_per_row * 144;
        if w_q4_bytes.len() != expected_bytes {
            return Err(Error::Kernel(format!("moe_grouped_gemm_q4_metal_buf weight bytes: got {} expected {}", w_q4_bytes.len(), expected_bytes)));
        }
        let w_buf = ctx.new_buffer_with_bytes(w_q4_bytes);
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        ctx.dispatch_threads("moe_grouped_gemm_q4", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
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
    pub fn gemv_f32_attn_pinned_buf_tcb(tcb: &mut TokenCommandBuffer<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads("gemv_f32_attn", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
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
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
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

    /// Track 3.6 — B=1 fused Q+K RoPE with optional bias addition.
    ///
    /// Replaces four dispatches per layer in `forward_token_greedy_tcb`
    /// (add_inplace q_bias, rope_q, add_inplace k_bias, rope_k) with ONE.
    /// Saves 3 dispatches/layer × n_layers (= 84 on Qwen-3B).
    ///
    /// `q_bias_buf` / `k_bias_buf` are read only when the corresponding
    /// `has_q_bias` / `has_k_bias` flag is set (pass any valid buffer when
    /// the flag is 0 — the kernel will not read it).
    ///
    /// Grid: `(n_q_heads + n_k_heads) × (head_dim/2)` threads.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_f32_b1_bias_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q_buf: &PinnedBuffer,
        k_buf: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        base: f32,
    ) -> Result<()> {
        let pairs_per_head = (head_dim / 2) as u32;
        let q_total = n_q_heads as u32 * pairs_per_head;
        let total = q_total + n_k_heads as u32 * pairs_per_head;
        let tg = TG_SIZE.min(total.max(1));
        let has_q = if q_bias_buf.is_some() { 1u32 } else { 0u32 };
        let has_k = if k_bias_buf.is_some() { 1u32 } else { 0u32 };
        let mut ab = KernelArgBuffer::new(
            tcb.ctx,
            &[
                ArgLayout::U32, // n_q_heads
                ArgLayout::U32, // n_k_heads
                ArgLayout::U32, // head_dim
                ArgLayout::U32, // pos
                ArgLayout::F32, // base
                ArgLayout::U32, // has_q_bias
                ArgLayout::U32, // has_k_bias
            ],
        )?;
        ab.set_u32(0, n_q_heads as u32);
        ab.set_u32(1, n_k_heads as u32);
        ab.set_u32(2, head_dim as u32);
        ab.set_u32(3, pos);
        ab.set_f32(4, base);
        ab.set_u32(5, has_q);
        ab.set_u32(6, has_k);
        // Use q_buf as a dummy for missing bias buffers (kernel won't read it
        // when has_q/k_bias=0).
        let qb = q_bias_buf.unwrap_or(q_buf);
        let kb = k_bias_buf.unwrap_or(k_buf);
        tcb.dispatch_threads("rope_qk_f32_b1_bias", (total, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(q_buf), 0);
            enc.set_buffer(1, Some(k_buf), 0);
            enc.set_buffer(2, Some(qb), 0);
            enc.set_buffer(3, Some(kb), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// Apply f32 RoPE in-place to a contiguous slice inside a larger f32 buffer.
    pub fn rope_slice_f32_inplace_tcb(tcb: &mut TokenCommandBuffer<'_>, buf: &PinnedBuffer, offset_f32: usize, head_dim: usize, pos: u32, base: f32) -> Result<()> {
        let offset_u32 = offset_f32 as u32;
        let head_dim_u32 = head_dim as u32;
        let half_dim = head_dim_u32 / 2;
        let tg = TG_SIZE.min(half_dim.max(1));
        tcb.dispatch_threads("rope_slice_f32_inplace", (half_dim, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(buf), 0);
            enc.set_u32(1, offset_u32);
            enc.set_u32(2, head_dim_u32);
            enc.set_u32(3, pos);
            enc.set_f32(4, base);
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
        tcb.dispatch_threads("kv_append_f32", (n_threads, 1, 1), (64u32.min(n_threads), 1, 1), |enc| {
            enc.set_buffer(0, Some(src_c_kv_normed), 0);
            enc.set_buffer(1, Some(src_kv_a_out), 0);
            enc.set_buffer(2, Some(dst_c_kv), 0);
            enc.set_buffer(3, Some(dst_k_pe), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// P1b: standard GQA multi-head attention for one decode step.
    /// Encodes `mha_decode_f32` into the supplied TCB. One TG per query
    /// head; TG size 64. Caller commits.
    ///
    /// `k_off_bytes` / `v_off_bytes` are byte offsets into the K/V cache
    /// buffer (used to address one layer's window when a single buffer
    /// holds all layers' KV cache).
    ///
    /// Buffer roles match the shader:
    ///   q       (n_heads, head_dim) f32
    ///   k_cache (seq_len, n_kv_heads, head_dim) f32 -- after applying k_off_bytes
    ///   v_cache (seq_len, n_kv_heads, head_dim) f32 -- after applying v_off_bytes
    ///   out     (n_heads, head_dim) f32
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_f32_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);

        // TG=128 matches Qwen-3B head_dim (128), so Phase 4 (per-output-element
        // accumulation) achieves full TG occupancy.
        const TG_SIZE: u32 = 128;
        let shmem_bytes = ((seq_len + TG_SIZE as usize) * std::mem::size_of::<f32>()) as u64;

        tcb.dispatch_threads("mha_decode_f32", (n_heads as u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Phase 2.3 — encode `mha_decode_flash_f32` (GQA online-softmax flash
    /// decode) into the per-token TCB. Drop-in numerical equivalent of
    /// [`Self::mha_decode_f32_tcb`] with the SAME arg list, the SAME 5-field
    /// `ArgbufMhaDecode`, and the SAME buffer bindings — only the dispatched
    /// kernel and the threadgroup-memory layout differ. The flash kernel does
    /// not materialize `scores[seq_len]`; its threadgroup memory is constant
    /// (`head_dim + FLASH_TG + 8` floats), so it removes the ~7800-token shmem
    /// cap that `mha_decode_f32` hits at the 32 KB ceiling. Default-off behind
    /// `HAWKING_QWEN_FLASH_ATTN`; not bit-identical to the materialize path
    /// (online softmax reorders the sum tile-wise — a reduction reorder).
    ///
    /// `FLASH_TG = 128` is load-bearing: it is exactly 4 simdgroups (the
    /// kernel's `state[4..7]` reduction assumes `FLASH_NSG = 4`) and matches
    /// Qwen2.5-3B `head_dim = 128` for full Phase-4 occupancy.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_flash_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_flash_f32_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);

        // FLASH_TG=128 = 4 simdgroups; matches Qwen-3B head_dim (128).
        // Threadgroup memory is constant (ctx-independent): no O(seq) scores.
        const FLASH_TG: u32 = 128;
        let f32_sz = std::mem::size_of::<f32>() as u64;
        // slot 0: acc[head_dim]
        let acc_bytes = head_dim as u64 * f32_sz;
        // slot 1: scores_tile[FLASH_TG]
        let scores_tile_bytes = FLASH_TG as u64 * f32_sz;
        // slot 2: state[8] = {m_run, l_run, corr, m_bc, simd0..3_max}
        let state_bytes = 8u64 * f32_sz;

        tcb.dispatch_threads("mha_decode_flash_f32", (n_heads as u32 * FLASH_TG, 1, 1), (FLASH_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_threadgroup_memory_length(0, acc_bytes);
            enc.set_threadgroup_memory_length(1, scores_tile_bytes);
            enc.set_threadgroup_memory_length(2, state_bytes);
        })
    }

    /// Wave-R6: flash decode reading a HALF K/V cache. Identical online-softmax
    /// + constant-threadgroup-memory structure as [`mha_decode_flash_f32_tcb`]
    /// (no O(seq) shmem ⇒ runs at 32K), but K/V are `half` (widened in-register),
    /// halving the dominant KV byte stream at depth. `k_off_bytes`/`v_off_bytes`
    /// are HALF-stride byte offsets (the f16 cache layout). q/out stay f32.
    /// Numerically equals `mha_decode_f16kv` up to the online-softmax reorder
    /// (gate atol 1e-3 + rtol 1e-4). Opt-in `HAWKING_QWEN_FLASH_F16KV`.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_flash_f16kv_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_flash_f16kv_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);

        const FLASH_TG: u32 = 128;
        let f32_sz = std::mem::size_of::<f32>() as u64;
        let acc_bytes = head_dim as u64 * f32_sz;
        let scores_tile_bytes = FLASH_TG as u64 * f32_sz;
        let state_bytes = 8u64 * f32_sz;

        tcb.dispatch_threads("mha_decode_flash_f16kv", (n_heads as u32 * FLASH_TG, 1, 1), (FLASH_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_threadgroup_memory_length(0, acc_bytes);
            enc.set_threadgroup_memory_length(1, scores_tile_bytes);
            enc.set_threadgroup_memory_length(2, state_bytes);
        })
    }

    /// Track 5.3 — quantize the per-token f32 K/V slices to int4 (per-row
    /// symmetric: head_dim=128 → 64 packed bytes + 1 f16 scale per kv-head row)
    /// and append into the layer's int4 KV cache. `dst_row_base` =
    /// `(layer*max_seq + seq_slot) * n_kv_heads` (ROW units). One TG per (row, K|V).
    #[allow(clippy::too_many_arguments)]
    pub fn kv_quant_int4_append_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src_k: &PinnedBuffer,
        src_v: &PinnedBuffer,
        k_packed: &PinnedBuffer,
        k_scales: &PinnedBuffer,
        v_packed: &PinnedBuffer,
        v_scales: &PinnedBuffer,
        n_kv_heads: usize,
        head_dim: usize,
        dst_row_base: usize,
    ) -> Result<()> {
        let kv_dim = (n_kv_heads * head_dim) as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, kv_dim);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, dst_row_base as u32);
        let hd = head_dim as u32;
        let red_bytes = hd as u64 * std::mem::size_of::<f32>() as u64;
        // dispatch_threads takes TOTAL thread counts: x = n_kv_heads TGs × hd
        // threads/TG, y = 2 (K|V) TGs × 1. tg = (hd,1,1) ⇒ tg_id.x ∈ [0,n_kv_heads),
        // tg_id.y ∈ {0,1}.
        tcb.dispatch_threads("kv_quant_int4_append", (n_kv_heads as u32 * hd, 2, 1), (hd, 1, 1), |enc| {
            enc.set_buffer(0, Some(src_k), 0);
            enc.set_buffer(1, Some(src_v), 0);
            enc.set_buffer(2, Some(k_packed), 0);
            enc.set_buffer(3, Some(k_scales), 0);
            enc.set_buffer(4, Some(v_packed), 0);
            enc.set_buffer(5, Some(v_scales), 0);
            enc.set_buffer(6, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, red_bytes);
        })
    }

    /// Track 5.3 — flash decode over an int4 (per-row symmetric) K/V cache.
    /// Same constant-shmem online-softmax structure as
    /// [`mha_decode_flash_f16kv_tcb`] (runs at 32K), but K/V rows are 64 packed
    /// bytes + one f16 scale, dequantized in-register. `k_packed`/`v_packed` are
    /// byte planes (HALF=head_dim/2 bytes/row), `k_scales`/`v_scales` one half/row.
    /// q/out stay f32. NOT bit-identical (int4 quant) — gate cosine ≥ 0.998.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_flash_int4kv_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_packed: &PinnedBuffer,
        k_packed_off_bytes: usize,
        k_scales: &PinnedBuffer,
        k_scales_off_elems: usize,
        v_packed: &PinnedBuffer,
        v_packed_off_bytes: usize,
        v_scales: &PinnedBuffer,
        v_scales_off_elems: usize,
        out: &PinnedBuffer,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_flash_int4kv_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);

        const FLASH_TG: u32 = 128;
        let f32_sz = std::mem::size_of::<f32>() as u64;
        let half_sz = std::mem::size_of::<half::f16>() as u64;
        let acc_bytes = head_dim as u64 * f32_sz;
        let scores_tile_bytes = FLASH_TG as u64 * f32_sz;
        let state_bytes = 8u64 * f32_sz;

        tcb.dispatch_threads("mha_decode_flash_int4kv", (n_heads as u32 * FLASH_TG, 1, 1), (FLASH_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_packed), k_packed_off_bytes as u64);
            enc.set_buffer(3, Some(k_scales), (k_scales_off_elems as u64) * half_sz);
            enc.set_buffer(4, Some(v_packed), v_packed_off_bytes as u64);
            enc.set_buffer(5, Some(v_scales), (v_scales_off_elems as u64) * half_sz);
            enc.set_buffer(6, Some(out), 0);
            enc.set_threadgroup_memory_length(0, acc_bytes);
            enc.set_threadgroup_memory_length(1, scores_tile_bytes);
            enc.set_threadgroup_memory_length(2, state_bytes);
        })
    }

    /// #15 redesign — fold one token's per-channel max|x| into the calibration
    /// scale table (run once per prefill token before finalize). grid=(nkvh*hd,2,1).
    /// The scale tables MUST be zeroed before the first call (new_buffer is not
    /// zero-initialized); the host divides the finalized maxima by 7.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_int4_calib_max_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src_k: &PinnedBuffer,
        src_v: &PinnedBuffer,
        k_chan_scales: &PinnedBuffer,
        v_chan_scales: &PinnedBuffer,
        n_kv_heads: usize,
        head_dim: usize,
        scale_row_base: usize,
    ) -> Result<()> {
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, head_dim as u32);
        ab.set_u32(1, scale_row_base as u32);
        let hd = head_dim as u32;
        tcb.dispatch_threads("kv_int4_calib_max", (n_kv_heads as u32 * hd, 2, 1), (hd, 1, 1), |enc| {
            enc.set_buffer(0, Some(src_k), 0);
            enc.set_buffer(1, Some(src_v), 0);
            enc.set_buffer(2, Some(k_chan_scales), 0);
            enc.set_buffer(3, Some(v_chan_scales), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// #15 redesign — per-channel int4 append (uses fixed per-channel scales,
    /// no row reduction). Bit-layout identical to `kv_quant_int4_append_tcb`.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_quant_int4_append_pc_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src_k: &PinnedBuffer,
        src_v: &PinnedBuffer,
        k_packed: &PinnedBuffer,
        k_chan_scales: &PinnedBuffer,
        v_packed: &PinnedBuffer,
        v_chan_scales: &PinnedBuffer,
        n_kv_heads: usize,
        head_dim: usize,
        dst_row_base: usize,
        scale_row_base: usize,
    ) -> Result<()> {
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, head_dim as u32);
        ab.set_u32(1, dst_row_base as u32);
        ab.set_u32(2, scale_row_base as u32);
        let hd = head_dim as u32;
        let red_bytes = hd as u64 * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("kv_quant_int4_append_pc", (n_kv_heads as u32 * hd, 2, 1), (hd, 1, 1), |enc| {
            enc.set_buffer(0, Some(src_k), 0);
            enc.set_buffer(1, Some(src_v), 0);
            enc.set_buffer(2, Some(k_packed), 0);
            enc.set_buffer(3, Some(k_chan_scales), 0);
            enc.set_buffer(4, Some(v_packed), 0);
            enc.set_buffer(5, Some(v_chan_scales), 0);
            enc.set_buffer(6, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, red_bytes);
        })
    }

    /// #15 redesign — per-channel int4 flash decode. `scale_row_base` is passed
    /// as a scalar buffer(7) (= layer*n_kv_heads*head_dim, in CHANNEL units).
    /// Each nibble is scaled by its channel's fixed scale. q/out stay f32.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_flash_int4kv_pc_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_packed: &PinnedBuffer,
        k_packed_off_bytes: usize,
        k_chan_scales: &PinnedBuffer,
        v_packed: &PinnedBuffer,
        v_packed_off_bytes: usize,
        v_chan_scales: &PinnedBuffer,
        out: &PinnedBuffer,
        scale_row_base: usize,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_flash_int4kv_pc_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);
        let srb = scale_row_base as u32;
        let srb_buf = tcb.ctx.new_buffer_with_bytes(&srb.to_le_bytes());
        const FLASH_TG: u32 = 128;
        let f32_sz = std::mem::size_of::<f32>() as u64;
        let acc_bytes = head_dim as u64 * f32_sz;
        let scores_tile_bytes = FLASH_TG as u64 * f32_sz;
        let state_bytes = 8u64 * f32_sz;
        tcb.dispatch_threads("mha_decode_flash_int4kv_pc", (n_heads as u32 * FLASH_TG, 1, 1), (FLASH_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_packed), k_packed_off_bytes as u64);
            enc.set_buffer(3, Some(k_chan_scales), 0);
            enc.set_buffer(4, Some(v_packed), v_packed_off_bytes as u64);
            enc.set_buffer(5, Some(v_chan_scales), 0);
            enc.set_buffer(6, Some(out), 0);
            enc.set_buffer(7, Some(&srb_buf), 0);
            enc.set_threadgroup_memory_length(0, acc_bytes);
            enc.set_threadgroup_memory_length(1, scores_tile_bytes);
            enc.set_threadgroup_memory_length(2, state_bytes);
        })
    }

    /// P1f: encode `memcpy_f32_off` — copy `n` f32 elements from
    /// `src[src_off..]` into `dst[dst_off..]`. Used by the dense (GQA)
    /// KV append path to write the per-token K/V slice into the
    /// per-layer cache window at `(layer * max_seq + seq_slot) * kv_dim`.
    /// Track 3.7 — Fused KV-cache append with optional V-bias addition.
    ///
    /// Replaces three dispatches per layer (v_bias add + k_append + v_append)
    /// with ONE, saving 2 dispatches/layer × n_layers = 56 on Qwen-3B.
    ///
    /// `k_tok` is written verbatim (K bias was already handled by
    /// `rope_qk_f32_b1_bias_tcb`). `v_tok` has `v_bias` added before writing.
    /// Pass `v_bias_buf=None` when the model has no V bias.
    #[allow(clippy::too_many_arguments)]
    pub fn kv_append_vbias_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        k_tok: &PinnedBuffer,
        v_tok: &PinnedBuffer,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
        kv_dim: usize,
        kv_off: usize, // element offset into k_cache/v_cache
    ) -> Result<()> {
        let kv_dim_u32 = kv_dim as u32;
        let kv_off_u32 = kv_off as u32;
        let has_v = if v_bias_buf.is_some() { 1u32 } else { 0u32 };
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, kv_dim_u32);
        ab.set_u32(1, kv_off_u32);
        ab.set_u32(2, has_v);
        let vb = v_bias_buf.unwrap_or(v_tok); // dummy when has_v=0
        let tg = TG_SIZE.min(kv_dim_u32.max(1));
        tcb.dispatch_threads("kv_append_vbias_f32", (kv_dim_u32, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(k_tok), 0);
            enc.set_buffer(1, Some(v_tok), 0);
            enc.set_buffer(2, Some(vb), 0);
            enc.set_buffer(3, Some(k_cache), 0);
            enc.set_buffer(4, Some(v_cache), 0);
            enc.set_buffer(5, Some(ab.handle()), 0);
        })
    }

    /// Track B6 — Fused RoPE(Q+K) + KV-cache append (+ V-bias), one dispatch.
    ///
    /// Combines `rope_qk_f32_b1_bias_tcb` + `kv_append_vbias_f32_tcb` into a
    /// single kernel call, saving 1 dispatch/layer (36 on Qwen-3B 36-layer).
    ///
    /// Thread partition (grid = q_pairs + k_pairs + kv_dim):
    /// - Q rope+bias section: threads [0, q_pairs) → in-place q_buf update
    /// - K rope+bias section: threads [q_pairs, q_pairs+k_pairs) → k_tok read-only,
    ///   rotated result written to k_cache[kv_off..]. k_tok left in pre-rope state.
    /// - V section: threads [q_pairs+k_pairs, +kv_dim) → v_tok+v_bias → v_cache[kv_off..]
    ///
    /// Opt-in via `HAWKING_QWEN_ROPE_KV_FUSE=1` (default off; bench first).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_kv_append_vbias_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q_buf: &PinnedBuffer,
        k_tok: &PinnedBuffer,
        v_tok: &PinnedBuffer,
        q_bias_buf: Option<&PinnedBuffer>,
        k_bias_buf: Option<&PinnedBuffer>,
        v_bias_buf: Option<&PinnedBuffer>,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        pos: u32,
        base: f32,
        kv_dim: usize,
        kv_off: usize, // element offset into k_cache / v_cache
    ) -> Result<()> {
        let pairs_per_head = (head_dim / 2) as u32;
        let q_pairs = n_q_heads as u32 * pairs_per_head;
        let k_pairs = n_k_heads as u32 * pairs_per_head;
        let total = q_pairs + k_pairs + kv_dim as u32;
        let tg = TG_SIZE.min(total.max(1));
        let has_q = if q_bias_buf.is_some() { 1u32 } else { 0u32 };
        let has_k = if k_bias_buf.is_some() { 1u32 } else { 0u32 };
        let has_v = if v_bias_buf.is_some() { 1u32 } else { 0u32 };
        let mut ab = KernelArgBuffer::new(
            tcb.ctx,
            &[
                ArgLayout::U32, // n_q_heads
                ArgLayout::U32, // n_k_heads
                ArgLayout::U32, // head_dim
                ArgLayout::U32, // pos
                ArgLayout::F32, // base
                ArgLayout::U32, // has_q_bias
                ArgLayout::U32, // has_k_bias
                ArgLayout::U32, // has_v_bias
                ArgLayout::U32, // kv_dim
                ArgLayout::U32, // kv_off
            ],
        )?;
        ab.set_u32(0, n_q_heads as u32);
        ab.set_u32(1, n_k_heads as u32);
        ab.set_u32(2, head_dim as u32);
        ab.set_u32(3, pos);
        ab.set_f32(4, base);
        ab.set_u32(5, has_q);
        ab.set_u32(6, has_k);
        ab.set_u32(7, has_v);
        ab.set_u32(8, kv_dim as u32);
        ab.set_u32(9, kv_off as u32);
        // Dummy buffers for unused bias slots (kernel won't read them).
        let qb = q_bias_buf.unwrap_or(q_buf);
        let kb = k_bias_buf.unwrap_or(k_tok);
        let vb = v_bias_buf.unwrap_or(v_tok);
        tcb.dispatch_threads("rope_qk_kv_append_vbias_f32", (total, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(q_buf), 0);
            enc.set_buffer(1, Some(k_tok), 0);
            enc.set_buffer(2, Some(v_tok), 0);
            enc.set_buffer(3, Some(qb), 0);
            enc.set_buffer(4, Some(kb), 0);
            enc.set_buffer(5, Some(vb), 0);
            enc.set_buffer(6, Some(k_cache), 0);
            enc.set_buffer(7, Some(v_cache), 0);
            enc.set_buffer(8, Some(ab.handle()), 0);
        })
    }

    /// Both offsets are element units.
    pub fn memcpy_f32_off_tcb(tcb: &mut TokenCommandBuffer<'_>, src: &PinnedBuffer, dst: &PinnedBuffer, src_off: usize, dst_off: usize, n: usize) -> Result<()> {
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, src_off as u32);
        ab.set_u32(2, dst_off as u32);
        let n_u32 = n as u32;
        let n_tg = n_u32.div_ceil(TG_SIZE);
        tcb.dispatch_threads("memcpy_f32_off", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_buffer(2, Some(ab.handle()), 0);
        })
    }

    /// R3 — batched KV scatter-append over B multi-seq slots. ONE dispatch (K+V)
    /// replaces the per-slot `memcpy_f32_off_tcb` loop (2B → 1 per layer). Each
    /// slot bi copies kv_dim K and V elems from src[bi*kv_dim] into its STABLE
    /// region (regions[bi]) at positions[bi] within layer `layer_off_elems`.
    /// Byte-identical (pure copy). `regions`/`positions` are u32 buffers (B each).
    #[allow(clippy::too_many_arguments)]
    pub fn kv_scatter_append_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src_k: &PinnedBuffer,
        src_v: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        v_cache: &PinnedBuffer,
        regions: &PinnedBuffer,
        positions: &PinnedBuffer,
        kv_dim: usize,
        b: usize,
        slot_stride_elems: usize,
        layer_off_elems: usize,
    ) -> Result<()> {
        let total = (b as u32) * (kv_dim as u32);
        if total == 0 {
            return Ok(());
        }
        let n_tg = total.div_ceil(TG_SIZE);
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, kv_dim as u32);
        ab.set_u32(1, b as u32);
        ab.set_u32(2, slot_stride_elems as u32);
        ab.set_u32(3, layer_off_elems as u32);
        tcb.dispatch_threads("kv_scatter_append_multiseq", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(src_k), 0);
            enc.set_buffer(1, Some(src_v), 0);
            enc.set_buffer(2, Some(k_cache), 0);
            enc.set_buffer(3, Some(v_cache), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_buffer(5, Some(regions), 0);
            enc.set_buffer(6, Some(positions), 0);
        })
    }

    /// f16-KV variant of [`mha_decode_f32_tcb`] (Phase 2.1-a). Identical
    /// args/geometry; the dispatched kernel reads k/v as `half`. `k_off_bytes`/
    /// `v_off_bytes` are BYTE offsets into the *half* cache — the caller
    /// computes them with `size_of::<half::f16>()` (= 2), e.g. via
    /// `DenseDecodeArena::kv_f16_layer_byte_offset`. Q stays f32, out stays
    /// f32. Default-off lever (HAWKING_QWEN_F16_KV). No commit.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_f16kv_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_f16kv_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);

        const TG_SIZE: u32 = 128;
        let shmem_bytes = ((seq_len + TG_SIZE as usize) * std::mem::size_of::<f32>()) as u64;

        tcb.dispatch_threads("mha_decode_f16kv", (n_heads as u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// f16-KV variant of [`mha_decode_f32_batched_tcb`] (Phase 2.1-a). The
    /// batched-prefill producer for the f16 cache; same geometry, half k/v.
    /// `k_off_bytes`/`v_off_bytes` are BYTE offsets into the *half* cache.
    /// Q/out stay f32. Default-off lever. No commit.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_f16kv_batched_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        p0: usize,
        batch: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_f16kv_batched_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let max_seq_len = p0 + batch; // largest batch's seq_len

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, p0 as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_heads as u32);
        ab.set_u32(3, n_kv_heads as u32);
        ab.set_u32(4, group_size);
        ab.set_f32(5, scale);

        const TG_SIZE_MHA: u32 = 128;
        let shmem_bytes = ((max_seq_len + TG_SIZE_MHA as usize) * std::mem::size_of::<f32>()) as u64;

        tcb.dispatch_threads("mha_decode_f16kv_batched", (n_heads as u32 * TG_SIZE_MHA, batch as u32, 1), (TG_SIZE_MHA, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// f32->f16 KV-append: clone of [`memcpy_f32_off_tcb`] writing f32 `src`
    /// into a `half` `dst` at ELEMENT offset `dst_off` (dst_off indexes half
    /// elements, identical convention to memcpy_f32_off_tcb). Uses the
    /// module-level `TG_SIZE` like its sibling. Default-off lever. No commit.
    pub fn memcpy_f32_to_f16_off_tcb(tcb: &mut TokenCommandBuffer<'_>, src: &PinnedBuffer, dst: &PinnedBuffer, src_off: usize, dst_off: usize, n: usize) -> Result<()> {
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, src_off as u32);
        ab.set_u32(2, dst_off as u32);
        let n_u32 = n as u32;
        let n_tg = n_u32.div_ceil(TG_SIZE);
        tcb.dispatch_threads("memcpy_f32_to_f16_off", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), 0);
            enc.set_buffer(2, Some(ab.handle()), 0);
        })
    }

    /// Encode mla_decode_kernel + o_proj gemv into external TCB.
    /// Reads arena.q / c_kv / k_pe; writes arena.attn_out / arena.out.
    /// c_kv and k_pe are passed explicitly so callers can use persistent GPU
    /// KV buffers (GPU-resident KV cache) or arena scratch buffers.
    /// No commit -- caller commits the TCB when ready.
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

        tcb.dispatch_threads("mla_decode_kernel", (n_heads_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(&arena.q), 0);
            enc.set_buffer(1, Some(c_kv), 0);
            enc.set_buffer(2, Some(k_pe), 0);
            enc.set_buffer(3, Some(kv_b_proj), 0);
            enc.set_buffer(4, Some(&arena.attn_out), 0);
            enc.set_u32(5, n_heads_u32);
            enc.set_u32(6, qk_nope_u32);
            enc.set_u32(7, qk_rope_u32);
            enc.set_u32(8, v_head_u32);
            enc.set_u32(9, kv_lora_u32);
            enc.set_u32(10, seq_len_u32);
            enc.set_f32(11, scale);
            enc.set_threadgroup_memory_length(0, q_nope_proj_bytes);
            enc.set_threadgroup_memory_length(1, scores_bytes);
            enc.set_threadgroup_memory_length(2, q_nope_proj_bytes);
        })?;
        // o_proj pinned as f16; use gemv_f16_simdmat (half w × float x → float y).
        // Cols = n_heads × v_head_dim = 2048 and rows = hidden = 2048 (both % 8 == 0).
        gemv_f16_simdmat_tcb(tcb, o_proj, hidden, n_heads * v_head_dim, &arena.attn_out, &arena.out)
    }

    /// Encode one f32 MoE gate-logit GEMV (mmap-pinned w, buffer x → buffer out) into TCB.
    pub fn gemv_f32_moe_pinned_buf_tcb(tcb: &mut TokenCommandBuffer<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads("gemv_f32_moe", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
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
        // Session J sketch: opt-in Q8_0 down "wide-N" kernels.
        // - HAWKING_Q8_DOWN_W4=1 → 4 rows/simdgroup, 32 rows/TG (rows%32==0).
        // - HAWKING_Q8_DOWN_W2=1 → 2 rows/simdgroup, 16 rows/TG (rows%16==0).
        // W4 takes precedence over W2. Both only apply to the Q8_0 _v2t down
        // kernel; fall back to default _v2t otherwise. No default changed.
        let q8_w4_opt_in = std::env::var_os("HAWKING_Q8_DOWN_W4").map(|v| v == "1" || v == "true").unwrap_or(false);
        let q8_w2_opt_in = std::env::var_os("HAWKING_Q8_DOWN_W2").map(|v| v == "1" || v == "true").unwrap_or(false);
        let use_q8_w4 = q8_w4_opt_in && kernel_name == "moe_batched_gemm_q8_0_indexed_v2t" && rows_u32 % 32 == 0;
        let use_q8_w2 = !use_q8_w4 && q8_w2_opt_in && kernel_name == "moe_batched_gemm_q8_0_indexed_v2t" && rows_u32 % 16 == 0;
        let effective_kernel: &str = if use_q8_w4 {
            "moe_batched_gemm_q8_0_indexed_v2t_w4"
        } else if use_q8_w2 {
            "moe_batched_gemm_q8_0_indexed_v2t_w2"
        } else {
            kernel_name
        };
        let n_tg_x = if use_q8_w4 {
            (rows_u32 + 31) / 32
        } else if use_q8_w2 {
            (rows_u32 + 15) / 16
        } else if is_v2_family {
            (rows_u32 + 7) / 8
        } else {
            rows_u32
        };
        let shmem_bytes = if is_v2t || use_q8_w2 || use_q8_w4 {
            // x_cache: cols floats in threadgroup SRAM
            (cols as u64) * std::mem::size_of::<f32>() as u64
        } else if is_v2_family {
            0u64
        } else {
            (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64
        };
        tcb.dispatch_threads(effective_kernel, (n_tg_x * tg_size, routes_u32, 1), (tg_size, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), 0);
            enc.set_buffer(1, Some(route_ids_buf), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(out_buf), 0);
            enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &base_offset_u64 as *const u64 as *const _);
            enc.set_u32(5, routes_u32);
            enc.set_u32(6, rows_u32);
            enc.set_u32(7, cols_u32);
            if !is_v2_family || is_v2t || use_q8_w2 || use_q8_w4 {
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            }
        })
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
        let routes_one = 1u32;
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
        for route_i in 0..routes {
            let ids_off = (route_i * std::mem::size_of::<u32>()) as u64;
            // x is route-major: offset x_buf so x[0*cols..] = original x[route_i*cols..].
            let x_off = (route_i * cols * std::mem::size_of::<f32>()) as u64;
            let out_off = (route_i * rows * std::mem::size_of::<f32>()) as u64;
            tcb.dispatch_threads(kernel_name, (n_tg_x * tg_size, 1, 1), (tg_size, 1, 1), |enc| {
                enc.set_buffer(0, Some(model_buf), 0);
                enc.set_buffer(1, Some(route_ids_buf), ids_off);
                enc.set_buffer(2, Some(x_buf), x_off);
                enc.set_buffer(3, Some(out_buf), out_off);
                enc.set_bytes(4, 8, &base_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, 4, &routes_one as *const u32 as *const _);
                enc.set_bytes(6, 4, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(7, 4, &cols_u32 as *const u32 as *const _);
                if !is_v2_family || is_v2t {
                    enc.set_threadgroup_memory_length(0, shmem_bytes);
                }
            })?;
        }
        Ok(())
    }

    pub fn silu_mul_tcb(tcb: &mut TokenCommandBuffer<'_>, gate_buf: &PinnedBuffer, up_buf: &PinnedBuffer, out_buf: &PinnedBuffer, n: usize) -> Result<()> {
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
        let up_offset_u64 = up_offset as u64;
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let n_tg_x = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("moe_batched_gemm_q4_indexed_v2t_gu", (n_tg_x * tg_size, routes_u32, 1), (tg_size, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), 0);
            enc.set_buffer(1, Some(route_ids_buf), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(act_buf), 0);
            enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
            enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64 as *const u64 as *const _);
            enc.set_u32(6, routes_u32);
            enc.set_u32(7, rows_u32);
            enc.set_u32(8, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    // v2t_gu_v2: same signature as encode_batched_gemv_fused_gu_tcb but dispatches
    // moe_batched_gemm_q4_indexed_v2t_gu_v2 (sumy trick + scale preload +
    // paired nibble reads -- Phase 2 optimisation).
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
        let up_offset_u64 = up_offset as u64;
        let routes_u32 = routes as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let n_tg_x = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("moe_batched_gemm_q4_indexed_v2t_gu_v2", (n_tg_x * tg_size, routes_u32, 1), (tg_size, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), 0);
            enc.set_buffer(1, Some(route_ids_buf), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(act_buf), 0);
            enc.set_bytes(4, std::mem::size_of::<u64>() as u64, &gate_offset_u64 as *const u64 as *const _);
            enc.set_bytes(5, std::mem::size_of::<u64>() as u64, &up_offset_u64 as *const u64 as *const _);
            enc.set_u32(6, routes_u32);
            enc.set_u32(7, rows_u32);
            enc.set_u32(8, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
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
        let up_offset_u64 = up_offset as u64;
        let routes_one = 1u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let tg_size = TG_SIZE as u32;
        let n_tg_x = (rows_u32 + 7) / 8;
        let shmem_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        for route_i in 0..routes {
            // Offset route_ids so route_ids[0] = expert for this route.
            let ids_off = (route_i * std::mem::size_of::<u32>()) as u64;
            // Offset act_buf so this route writes to act[route_i * rows .. +rows].
            let act_off = (route_i * rows * std::mem::size_of::<f32>()) as u64;
            // x (hidden state) is the same for all routes -- no offset needed.
            tcb.dispatch_threads("moe_batched_gemm_q4_indexed_v2t_gu", (n_tg_x * tg_size, 1, 1), (tg_size, 1, 1), |enc| {
                enc.set_buffer(0, Some(model_buf), 0);
                enc.set_buffer(1, Some(route_ids_buf), ids_off);
                enc.set_buffer(2, Some(x_buf), 0);
                enc.set_buffer(3, Some(act_buf), act_off);
                enc.set_bytes(4, 8, &gate_offset_u64 as *const u64 as *const _);
                enc.set_bytes(5, 8, &up_offset_u64 as *const u64 as *const _);
                enc.set_bytes(6, 4, &routes_one as *const u32 as *const _);
                enc.set_bytes(7, 4, &rows_u32 as *const u32 as *const _);
                enc.set_bytes(8, 4, &cols_u32 as *const u32 as *const _);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })?;
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
            encode_batched_gemv_fused_gu_v2_tcb(tcb, model_buf, shared_route_ids_buf, x_buf, shared_act, shared_gate_offset, shared_up_offset, 1, shared_mid, hidden)?;
        } else if q4k_schedule == "v2t_gu" || q4k_schedule == "v2t_gu_serial" {
            encode_batched_gemv_fused_gu_tcb(tcb, model_buf, shared_route_ids_buf, x_buf, shared_act, shared_gate_offset, shared_up_offset, 1, shared_mid, hidden)?;
        } else {
            encode_batched_gemv_indexed_tcb(tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf, shared_gate_out, shared_gate_offset, 1, shared_mid, hidden)?;
            encode_batched_gemv_indexed_tcb(tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf, shared_up_out, shared_up_offset, 1, shared_mid, hidden)?;
            silu_mul_tcb(tcb, shared_gate_out, shared_up_out, shared_act, shared_mid)?;
        }

        encode_batched_gemv_indexed_tcb(tcb, shared_down_kernel, model_buf, shared_route_ids_buf, shared_act, out_buf, shared_down_offset, 1, hidden, shared_mid)
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

        let has_shared = shared_gate_offset.is_some() || shared_up_offset.is_some() || shared_down_offset.is_some();

        let q4k_indexed_kernel = match q4k_schedule {
            "v2" | "llama_port" | "per_shape" => "moe_batched_gemm_q4_indexed_v2",
            "v2s" => "moe_batched_gemm_q4_indexed_v2s",
            "v2t" | "v2t_gu" | "v2t_gu_serial" | "v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t",
            _ => "moe_batched_gemm_q4_indexed",
        };
        let use_fused_gu_v2 = q4k_schedule == "v2t_gu_v2";
        let use_fused_gu = q4k_schedule == "v2t_gu";
        // Serial: dispatch one expert at a time so each expert's weight slab (~3 MB
        // gate+up) is a single sequential stream. Eliminates 6-stream L2 thrashing.
        // Effective only when combined with a single command buffer (Pillar 2).
        let use_serial_gu = q4k_schedule == "v2t_gu_serial";

        if use_serial_gu {
            encode_batched_gemv_fused_gu_serial_tcb(tcb, model_buf, route_ids_buf, x_buf, routed_act, routed_gate_offset, routed_up_offset, routes, routed_mid, hidden)?;
        } else if use_fused_gu_v2 {
            encode_batched_gemv_fused_gu_v2_tcb(tcb, model_buf, route_ids_buf, x_buf, routed_act, routed_gate_offset, routed_up_offset, routes, routed_mid, hidden)?;
        } else if use_fused_gu {
            encode_batched_gemv_fused_gu_tcb(tcb, model_buf, route_ids_buf, x_buf, routed_act, routed_gate_offset, routed_up_offset, routes, routed_mid, hidden)?;
        } else {
            encode_batched_gemv_indexed_tcb(tcb, q4k_indexed_kernel, model_buf, route_ids_buf, x_buf, routed_gate_out, routed_gate_offset, routes, routed_mid, hidden)?;
            encode_batched_gemv_indexed_tcb(tcb, q4k_indexed_kernel, model_buf, route_ids_buf, x_buf, routed_up_out, routed_up_offset, routes, routed_mid, hidden)?;
            silu_mul_tcb(tcb, routed_gate_out, routed_up_out, routed_act, routes * routed_mid)?;
        }

        // Down projection: also serial when using v2t_gu_serial to fix the same
        // L2 thrashing on the down-projection weight slabs.
        if use_serial_gu {
            encode_batched_gemv_indexed_serial_tcb(tcb, routed_down_kernel, model_buf, route_ids_buf, routed_act, routed_out, routed_down_offset, routes, hidden, routed_mid)?;
        } else {
            encode_batched_gemv_indexed_tcb(tcb, routed_down_kernel, model_buf, route_ids_buf, routed_act, routed_out, routed_down_offset, routes, hidden, routed_mid)?;
        }

        if let (Some(gate_off), Some(up_off), Some(down_off)) = (shared_gate_offset, shared_up_offset, shared_down_offset) {
            // Shared expert always routes=1, so serial == parallel. Use the
            // appropriate fused_gu variant when any gu schedule is selected.
            if use_fused_gu_v2 {
                encode_batched_gemv_fused_gu_v2_tcb(tcb, model_buf, shared_route_ids_buf, x_buf, shared_act, gate_off, up_off, 1, shared_mid, hidden)?;
            } else if use_fused_gu || use_serial_gu {
                encode_batched_gemv_fused_gu_tcb(tcb, model_buf, shared_route_ids_buf, x_buf, shared_act, gate_off, up_off, 1, shared_mid, hidden)?;
            } else {
                encode_batched_gemv_indexed_tcb(tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf, shared_gate_out, gate_off, 1, shared_mid, hidden)?;
                encode_batched_gemv_indexed_tcb(tcb, q4k_indexed_kernel, model_buf, shared_route_ids_buf, x_buf, shared_up_out, up_off, 1, shared_mid, hidden)?;
                silu_mul_tcb(tcb, shared_gate_out, shared_up_out, shared_act, shared_mid)?;
            }
            encode_batched_gemv_indexed_tcb(tcb, shared_down_kernel, model_buf, shared_route_ids_buf, shared_act, shared_out, down_off, 1, hidden, shared_mid)?;
        }

        encode_route_accumulate_tcb(tcb, routed_out, route_weights_buf, shared_out, out_buf, hidden, routes, has_shared)
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
            tcb,
            model_buf,
            routed_gate_offset,
            routed_up_offset,
            routed_down_offset,
            route_ids_buf,
            route_weights_buf,
            routes,
            shared_route_ids_buf,
            shared_gate_offset,
            shared_up_offset,
            shared_down_offset,
            hidden,
            routed_mid,
            shared_mid,
            q4k_schedule,
            routed_down_kernel,
            shared_down_kernel,
            x_buf,
            out_buf,
            &routed_gate_out,
            &routed_up_out,
            &routed_act,
            &routed_out,
            &shared_gate_out,
            &shared_up_out,
            &shared_act,
            &shared_out,
        )?;

        Ok(vec![routed_gate_out, routed_up_out, routed_act, routed_out, shared_gate_out, shared_up_out, shared_act, shared_out])
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
        let route_weights_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(route_weights));
        let shared_route_ids_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice(&[0u32]));

        let mut tcb = TokenCommandBuffer::new(ctx);
        let _temp_buffers = encode_moe_block_batched_indexed_tcb(
            &mut tcb,
            ctx,
            model_buf,
            routed_gate_offset,
            routed_up_offset,
            routed_down_offset,
            &route_ids_buf,
            &route_weights_buf,
            routes,
            &shared_route_ids_buf,
            shared_gate_offset,
            shared_up_offset,
            shared_down_offset,
            hidden,
            routed_mid,
            shared_mid,
            q4k_schedule,
            routed_down_kernel,
            shared_down_kernel,
            x_buf,
            out_buf,
        )?;
        tcb.commit_and_wait()?;
        // temp buffers dropped here, after GPU is done
        Ok(())
    }

    // ── v1.0.0-D: embed lookup writing f32 residual directly to GPU buffer ──

    /// Encode embed_lookup_f32 into TCB: reads f16 embed table at row `token`,
    /// writes hidden f32 values into x_buf. Zero counted dispatches.
    pub fn embed_lookup_metal_f32_tcb(tcb: &mut TokenCommandBuffer<'_>, embed_buf: &PinnedBuffer, token: u32, hidden: usize, x_buf: &PinnedBuffer) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let tg = TG_SIZE.min(hidden_u32);
        tcb.dispatch_threads("embed_lookup_f32", (hidden_u32, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(embed_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_u32(2, hidden_u32);
            enc.set_u32(3, token);
        })
    }

    /// Track B7 — Fused embedding lookup + layer-0 RMSNorm (single dispatch).
    ///
    /// Replaces `embed_lookup_metal_f32_tcb(embed, token → x)` immediately
    /// followed by `rmsnorm_metal_buf_tcb(x, weight, eps → x_norm)` with one
    /// dispatch of `embed_lookup_rmsnorm_f32`. Saves 1 dispatch (292 → 291
    /// at default settings). Default-ON; opt-out via
    /// `HAWKING_QWEN_EMBED_RMSNORM_FUSE=0`.
    ///
    /// Phase 1 loads embed → x (device) and accumulates partial squares.
    /// Phase 2 re-reads x (L1 cache hit on Apple Silicon) to write x_norm —
    /// bit-identical to the two-dispatch reference. No hidden-size cap.
    pub fn embed_lookup_rmsnorm_f32_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        embed_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        token: u32,
        hidden: usize,
        eps: f32,
        x_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
    ) -> Result<()> {
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden as u32);
        ab.set_u32(1, token);
        ab.set_f32(2, eps);
        tcb.dispatch_threads("embed_lookup_rmsnorm_f32", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(embed_buf), 0);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_buffer(3, Some(x_norm_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    // ── v1.0.0-E: GPU argmax sampling dispatchers ────────────────────────────

    /// LM-head GEMV via TCB: w_buf (rows×cols f16) × x_buf (cols f32) → y_buf (rows f32).
    /// Zero counted dispatches. Used for the final LM-head projection in the greedy path.
    pub fn gemv_f16_metal_buf_tcb(tcb: &mut TokenCommandBuffer<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("gemv_f16", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// GPU greedy argmax via TCB: logits_buf (vocab f32) → token_buf (u32).
    /// Zero counted dispatches. Grid and threadgroup are both (256, 1, 1) to
    /// match the sample_argmax_f32 kernel's two-phase 256-thread reduction.
    pub fn sample_argmax_f32_tcb(tcb: &mut TokenCommandBuffer<'_>, logits_buf: &PinnedBuffer, token_buf: &PinnedBuffer, vocab: usize) -> Result<()> {
        let vocab_u32 = vocab as u32;
        let shmem_f = 256 * std::mem::size_of::<f32>() as u64;
        let shmem_u = 256 * std::mem::size_of::<u32>() as u64;
        tcb.dispatch_threads("sample_argmax_f32", (256, 1, 1), (256, 1, 1), |enc| {
            enc.set_buffer(0, Some(logits_buf), 0);
            enc.set_buffer(1, Some(token_buf), 0);
            enc.set_u32(2, vocab_u32);
            enc.set_threadgroup_memory_length(0, shmem_f);
            enc.set_threadgroup_memory_length(1, shmem_u);
        })
    }

    /// Batched greedy argmax: one thread group (256 threads) per slot.
    /// `logits_buf` is `(batch, vocab)` row-major f32.
    /// `tokens_buf` receives `batch` u32 token ids.
    /// Grid: (batch * 256, 1, 1). Thread groups: (256, 1, 1).
    pub fn sample_argmax_f32_batched_tcb(tcb: &mut TokenCommandBuffer<'_>, logits_buf: &PinnedBuffer, tokens_buf: &PinnedBuffer, vocab: usize, batch: usize) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let need_logits = (batch * vocab * std::mem::size_of::<f32>()) as u64;
        let need_tokens = (batch * std::mem::size_of::<u32>()) as u64;
        if logits_buf.length() < need_logits {
            return Err(Error::Kernel(format!("sample_argmax_f32_batched: logits buf {} < need {}", logits_buf.length(), need_logits)));
        }
        if tokens_buf.length() < need_tokens {
            return Err(Error::Kernel(format!("sample_argmax_f32_batched: tokens buf {} < need {}", tokens_buf.length(), need_tokens)));
        }
        let vocab_u32 = vocab as u32;
        let batch_u32 = batch as u32;
        const TG: u32 = 256;
        let shmem_f = TG as u64 * std::mem::size_of::<f32>() as u64;
        let shmem_u = TG as u64 * std::mem::size_of::<u32>() as u64;
        tcb.dispatch_threads("sample_argmax_f32_batched", (batch as u32 * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(logits_buf), 0);
            enc.set_buffer(1, Some(tokens_buf), 0);
            enc.set_u32(2, vocab_u32);
            enc.set_u32(3, batch_u32);
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
        tcb.dispatch_threads("rmsnorm_gemv_f32_attn_pinned", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_f32(3, eps);
            enc.set_buffer(4, Some(out_buf), 0);
            enc.set_u32(5, rows_u32);
            enc.set_u32(6, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
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
        tcb.dispatch_threads("rmsnorm_gemv_f16w_attn_pinned", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_f32(3, eps);
            enc.set_buffer(4, Some(out_buf), 0);
            enc.set_u32(5, rows_u32);
            enc.set_u32(6, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// v2.2.0-T2.14 -- v2t-pattern dispatch: 8 rows per threadgroup, one simdgroup
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
            return Err(crate::error::Error::Kernel(format!("rmsnorm_gemv_f16w_attn_pinned_v2t requires rows%8==0 and cols%32==0; rows={rows} cols={cols}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_tgs = (rows / 8) as u32;
        let shmem_bytes = 16u64 * std::mem::size_of::<f32>() as u64;
        let xw_cache_bytes = (cols as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("rmsnorm_gemv_f16w_attn_pinned_v2t", (n_tgs * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_f32(3, eps);
            enc.set_buffer(4, Some(out_buf), 0);
            enc.set_u32(5, rows_u32);
            enc.set_u32(6, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
            enc.set_threadgroup_memory_length(1, xw_cache_bytes);
        })
    }

    // ── end v1.0.0-G ─────────────────────────────────────────────────────────

    // ── v1.0.0-H: simdgroup_matrix GEMV dispatchers (Path 2) ─────────────────

    /// simdgroup_matrix GEMV: w (rows×cols f32) × x (cols f32) → y (rows f32).
    /// One SIMD group (32 threads) per threadgroup; each handles 8 output rows.
    /// Requires cols % 8 == 0. Grid = (ceil(rows/8)*32, 1, 1), TG = (32, 1, 1).
    /// Zero counted dispatches.
    pub fn gemv_simdgroup_f32_tcb(tcb: &mut TokenCommandBuffer<'_>, w_buf: &PinnedBuffer, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer, rows: usize, cols: usize) -> Result<()> {
        if cols % 8 != 0 {
            return Err(crate::error::Error::Kernel(format!("gemv_simdgroup_f32 requires cols % 8 == 0; cols={cols}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_groups = rows.div_ceil(8) as u32;
        let scratch_bytes = 192u64 * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("gemv_simdgroup_f32", (n_groups * 32, 1, 1), (32, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, scratch_bytes);
        })
    }

    // ── end v1.0.0-H ─────────────────────────────────────────────────────────

    // ── v1.1.0-X: simdgroup_matrix LM-head GEMV (f16 weights) ────────────────

    /// LM-head GEMV via simdgroup_matrix: w (rows×cols f16) × x (cols f32) → y (rows f32).
    /// Mixed-precision: half A × half B + float C → float D.
    /// One SIMD group (32 threads) per threadgroup; each handles 8 output rows.
    /// Requires cols % 8 == 0. Grid = (ceil(rows/8)*32, 1, 1), TG = (32, 1, 1).
    pub fn gemv_f16_simdmat_tcb(tcb: &mut TokenCommandBuffer<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        if cols % 8 != 0 {
            return Err(crate::error::Error::Kernel(format!("gemv_f16_simdmat requires cols % 8 == 0; cols={cols}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let n_groups = rows.div_ceil(8) as u32;
        // 3 × 64 floats: W tile + X tile + result tile
        let shmem_bytes: u64 = 192 * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("gemv_f16_simdmat", (n_groups * 32, 1, 1), (32, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    // ── end v1.1.0-X ─────────────────────────────────────────────────────────

    // ── Phase 5C.2: f32→f16 norm output + f16-activation LM head ─────────────

    /// f32 residual → f16 normed activation (Phase 5C.2).
    /// Dispatches `rmsnorm_f32_to_f16`: reads f32 x, f32 weight → writes half* out.
    /// Variance accumulator stays f32. Used when kernel profile x_norm_dtype="f16".
    /// Same ArgbufRmsnorm pattern as rmsnorm_metal_buf_tcb. out_buf must be
    /// pre-allocated as hidden × sizeof(f16) bytes (arena.x_norm_f16_buf).
    pub fn rmsnorm_f32_to_f16_tcb(tcb: &mut TokenCommandBuffer<'_>, x_buf: &PinnedBuffer, weight_buf: &PinnedBuffer, eps: f32, hidden: usize, out_buf: &PinnedBuffer) -> Result<()> {
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
    pub fn gemv_f16_f16in_tcb(tcb: &mut TokenCommandBuffer<'_>, w_buf: &PinnedBuffer, rows: usize, cols: usize, x_buf: &PinnedBuffer, y_buf: &PinnedBuffer) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("gemv_f16_f16in", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), 0);
            enc.set_buffer(2, Some(y_buf), 0);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    // ── end Phase 5C.2 ────────────────────────────────────────────────────────

    // ── P3 — Offset-variant dispatchers for batched prefill ───────────────
    //
    // Thin wrappers around the existing single-token dispatchers that accept
    // byte offsets on the per-token input/output buffers. Used by
    // `forward_tokens_batch_tcb` to slice B-wide arena buffers into B
    // single-token windows. The compiled kernel is identical; only the GPU
    // buffer base pointer is shifted.

    pub fn embed_lookup_metal_f32_off_tcb(tcb: &mut TokenCommandBuffer<'_>, embed_buf: &PinnedBuffer, token: u32, hidden: usize, x_buf: &PinnedBuffer, x_off_bytes: usize) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let tg = TG_SIZE.min(hidden_u32);
        tcb.dispatch_threads("embed_lookup_f32", (hidden_u32, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(embed_buf), 0);
            enc.set_buffer(1, Some(x_buf), x_off_bytes as u64);
            enc.set_u32(2, hidden_u32);
            enc.set_u32(3, token);
        })
    }

    /// Track 3.2: batched embed lookup — B tokens in one dispatch.
    /// Saves B-1 dispatches vs the per-slot `embed_lookup_metal_f32_off_tcb` loop.
    pub fn embed_lookup_f32_batched_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        embed_buf: &PinnedBuffer,  // (vocab, hidden) f16
        tokens_buf: &PinnedBuffer, // (B,) u32 token ids
        out_buf: &PinnedBuffer,    // (B, hidden) f32
        hidden: usize,
        b: usize,
    ) -> Result<()> {
        if b == 0 {
            return Ok(());
        }
        let total = (b * hidden) as u32;
        let tg = TG_SIZE.min(total);
        tcb.dispatch_threads("embed_lookup_f32_batched", (total, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(embed_buf), 0);
            enc.set_buffer(1, Some(tokens_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_u32(3, hidden as u32);
            enc.set_u32(4, b as u32);
        })
    }

    /// Track 3.2: batched cold rmsnorm — B rows in one dispatch, no residual add.
    /// Saves B-1 dispatches vs the per-slot `rmsnorm_metal_buf_off_tcb` loop.
    pub fn rmsnorm_f32_batched_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,      // (B, hidden) f32 input
        weight_buf: &PinnedBuffer, // (hidden,) f32 scale
        out_buf: &PinnedBuffer,    // (B, hidden) f32 output
        eps: f32,
        hidden: usize,
        b: usize,
    ) -> Result<()> {
        if b == 0 {
            return Ok(());
        }
        let shmem = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden as u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("rmsnorm_f32_batched", (TG_SIZE * b as u32, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(out_buf), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem);
        })
    }

    pub fn rmsnorm_metal_buf_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        weight_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("rmsnorm_f32", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), x_off_bytes as u64);
            enc.set_buffer(1, Some(weight_buf), 0);
            enc.set_buffer(2, Some(out_buf), out_off_bytes as u64);
            enc.set_buffer(3, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn add_rmsnorm_fused_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        attn_out_buf: &PinnedBuffer,
        attn_off_bytes: usize,
        weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        x_norm_off_bytes: usize,
        eps: f32,
        hidden: usize,
    ) -> Result<()> {
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("add_rmsnorm_fused", (TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), x_off_bytes as u64);
            enc.set_buffer(1, Some(attn_out_buf), attn_off_bytes as u64);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_buffer(3, Some(x_norm_buf), x_norm_off_bytes as u64);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    pub fn add_inplace_metal_off_tcb(tcb: &mut TokenCommandBuffer<'_>, a_buf: &PinnedBuffer, a_off_bytes: usize, b_buf: &PinnedBuffer, n: usize) -> Result<()> {
        let n_u32 = n as u32;
        let n_tg = n_u32.div_ceil(TG_SIZE);
        tcb.dispatch_threads("add_inplace", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(a_buf), a_off_bytes as u64);
            enc.set_buffer(1, Some(b_buf), 0);
            enc.set_u32(2, n_u32);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rope_q_f32_inplace_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q_buf: &PinnedBuffer,
        q_off_bytes: usize,
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
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, n_heads_u32);
        ab.set_u32(1, q_head_u32);
        ab.set_u32(2, qk_nope_u32);
        ab.set_u32(3, qk_rope_u32);
        ab.set_u32(4, pos);
        ab.set_f32(5, base);
        tcb.dispatch_threads("rope_q_f32_inplace", (total_pairs, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(q_buf), q_off_bytes as u64);
            enc.set_buffer(1, Some(ab.handle()), 0);
        })
    }

    /// R2 — batched RoPE over B multi-seq slots, each at its own `positions[bi]`.
    /// One dispatch replaces the per-slot `rope_q_f32_inplace_off_tcb` loop
    /// (2B → 2 per layer). Bit-identical (rope is elementwise). `x_buf` is laid
    /// out [B, slot_stride] f32; call once for Q (n_heads, slot_stride=q_dim) and
    /// once for K (n_kv_heads, slot_stride=kv_dim). `positions` is a u32 buffer
    /// (B entries). Full-head rope (qk_nope_dim = 0), matching the dense path.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_f32_batched_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        positions: &PinnedBuffer,
        n_heads: usize,
        head_dim: usize,
        slot_stride_elems: usize,
        b: usize,
        base: f32,
    ) -> Result<()> {
        let pairs_per_head = (head_dim / 2) as u32;
        let total = (b as u32) * (n_heads as u32) * pairs_per_head;
        if total == 0 {
            return Ok(());
        }
        let n_tg = total.div_ceil(TG_SIZE);
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, n_heads as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, slot_stride_elems as u32);
        ab.set_u32(3, b as u32);
        ab.set_f32(4, base);
        tcb.dispatch_threads("rope_f32_batched_multiseq", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(ab.handle()), 0);
            enc.set_buffer(2, Some(positions), 0);
        })
    }

    /// Fused Q+K RoPE (Track 3.4): one dispatch/layer instead of two.
    /// Saves 28 dispatches on Qwen-3B (1/layer × 28 layers).
    /// Bit-identical to calling rope_f32_batched_multiseq_tcb twice.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_qk_f32_batched_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q_buf: &PinnedBuffer,
        k_buf: &PinnedBuffer,
        positions: &PinnedBuffer,
        n_q_heads: usize,
        n_k_heads: usize,
        head_dim: usize,
        q_slot_stride: usize,
        k_slot_stride: usize,
        b: usize,
        base: f32,
    ) -> Result<()> {
        let pairs_per_head = head_dim / 2;
        let total = b * (n_q_heads + n_k_heads) * pairs_per_head;
        if total == 0 {
            return Ok(());
        }
        // ArgbufRopeQKMultiseq: n_q_heads, n_k_heads, head_dim, q_slot_stride, k_slot_stride, b, base
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, n_q_heads as u32);
        ab.set_u32(1, n_k_heads as u32);
        ab.set_u32(2, head_dim as u32);
        ab.set_u32(3, q_slot_stride as u32);
        ab.set_u32(4, k_slot_stride as u32);
        ab.set_u32(5, b as u32);
        ab.set_f32(6, base);
        let n_tg = (total as u32).div_ceil(TG_SIZE);
        tcb.dispatch_threads("rope_qk_f32_batched_multiseq", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(q_buf), 0);
            enc.set_buffer(1, Some(k_buf), 0);
            enc.set_buffer(2, Some(ab.handle()), 0);
            enc.set_buffer(3, Some(positions), 0);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_f32_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        q_off_bytes: usize,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        out_off_bytes: usize,
        seq_len: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_f32_off_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, seq_len as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_f32(4, scale);

        const TG_SIZE_MHA: u32 = 128;
        let shmem_bytes = ((seq_len + TG_SIZE_MHA as usize) * std::mem::size_of::<f32>()) as u64;

        tcb.dispatch_threads("mha_decode_f32", (n_heads as u32 * TG_SIZE_MHA, 1, 1), (TG_SIZE_MHA, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), q_off_bytes as u64);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), out_off_bytes as u64);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn silu_mul_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        gate_buf: &PinnedBuffer,
        gate_off_bytes: usize,
        up_buf: &PinnedBuffer,
        up_off_bytes: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
        n: usize,
    ) -> Result<()> {
        let n_u32 = n as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32])?;
        ab.set_u32(0, n_u32);
        tcb.dispatch_threads("moe_batched_silu_mul", (n_u32, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(gate_buf), gate_off_bytes as u64);
            enc.set_buffer(1, Some(up_buf), up_off_bytes as u64);
            enc.set_buffer(2, Some(out_buf), out_off_bytes as u64);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q4_k_m_v3_8r_pinned_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q4_k_m_v3_8r";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_off_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(144)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_off_tcb overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_off_tcb bytes mismatch: got {w_byte_size} expected {expected_bytes}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V3_TG: u32 = 256;
        const V3_ROWS: u32 = 8;
        let n_tg = rows_u32.div_ceil(V3_ROWS);
        tcb.dispatch_threads(KERNEL, (n_tg * V3_TG, 1, 1), (V3_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), x_off_bytes as u64);
            enc.set_buffer(2, Some(out_buf), out_off_bytes as u64);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemv_q6_k_pinned_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
    ) -> Result<()> {
        const KERNEL: &str = "gemm_q6_k_fused_v2";
        if cols % 256 != 0 {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_off_tcb requires cols % 256 == 0; got cols={cols}")));
        }
        let blocks_per_row = cols / 256;
        let expected_bytes = rows.checked_mul(blocks_per_row).and_then(|v| v.checked_mul(210)).ok_or_else(|| Error::Kernel(format!("{KERNEL}_pinned_off_tcb byte-size overflow")))?;
        if w_byte_size != expected_bytes {
            return Err(Error::Kernel(format!("{KERNEL}_pinned_off_tcb weight bytes: got {w_byte_size} expected {expected_bytes}")));
        }
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        const V2_TG: u32 = 256;
        let n_tg = rows_u32.div_ceil(8);
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(KERNEL, (n_tg * V2_TG, 1, 1), (V2_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(model_buf), w_offset as u64);
            enc.set_buffer(1, Some(x_buf), x_off_bytes as u64);
            enc.set_buffer(2, Some(out_buf), out_off_bytes as u64);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    pub fn gemv_f16_metal_buf_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_buf: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        y_buf: &PinnedBuffer,
        y_off_bytes: usize,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        tcb.dispatch_threads("gemv_f16", (rows_u32 * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(w_buf), 0);
            enc.set_buffer(1, Some(x_buf), x_off_bytes as u64);
            enc.set_buffer(2, Some(y_buf), y_off_bytes as u64);
            enc.set_u32(3, rows_u32);
            enc.set_u32(4, cols_u32);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }
    // ── end P3 offset variants ────────────────────────────────────────────

    // ── P3 — Batched per-layer-op dispatchers ─────────────────────────
    //
    // These collapse the B-times-sequential dispatches in the batched
    // prefill loop into single dispatches that cover all B rows. Same
    // math, fewer kernel launches.

    /// P3 — Batched MHA decode: one dispatch handles all B query tokens.
    /// 2D grid (n_heads, B) of TGs. Each TG computes attention for one
    /// (head, batch_elem) using its own causal seq_len = p0 + b + 1.
    #[allow(clippy::too_many_arguments)]
    pub fn mha_decode_f32_batched_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        p0: usize,
        batch: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_f32_batched_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();
        let max_seq_len = p0 + batch; // largest batch's seq_len

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, p0 as u32);
        ab.set_u32(1, head_dim as u32);
        ab.set_u32(2, n_heads as u32);
        ab.set_u32(3, n_kv_heads as u32);
        ab.set_u32(4, group_size);
        ab.set_f32(5, scale);

        const TG_SIZE_MHA: u32 = 128;
        let shmem_bytes = ((max_seq_len + TG_SIZE_MHA as usize) * std::mem::size_of::<f32>()) as u64;

        tcb.dispatch_threads("mha_decode_f32_batched", (n_heads as u32 * TG_SIZE_MHA, batch as u32, 1), (TG_SIZE_MHA, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    /// Continuous-batching multi-seq decode: B INDEPENDENT sequences in one
    /// dispatch. Each batch element `bi` has its own position (`positions[bi]`,
    /// a u32 buffer of length `batch`) and its own slot-strided K/V region at
    /// element offset `bi * kv_slot_stride_elems`. `max_seq` (= max position + 1)
    /// sizes the shared scores shmem. See `mha_decode_f32_batched_multiseq` in
    /// shaders/mha.metal. The softmax math is byte-identical to
    /// `mha_decode_f32_batched_tcb`; only the per-slot K/V base + per-slot SEQ
    /// differ.
    pub fn mha_decode_f32_batched_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        q: &PinnedBuffer,
        k_cache: &PinnedBuffer,
        k_off_bytes: usize,
        v_cache: &PinnedBuffer,
        v_off_bytes: usize,
        out: &PinnedBuffer,
        positions: &PinnedBuffer,
        regions: &PinnedBuffer,
        max_seq: usize,
        kv_slot_stride_elems: usize,
        batch: usize,
        head_dim: usize,
        n_heads: usize,
        n_kv_heads: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Metal(format!("mha_decode_f32_batched_multiseq_tcb: n_heads ({n_heads}) must be a multiple of n_kv_heads ({n_kv_heads})")));
        }
        let group_size = (n_heads / n_kv_heads) as u32;
        let scale = 1.0_f32 / (head_dim as f32).sqrt();

        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, head_dim as u32);
        ab.set_u32(1, n_heads as u32);
        ab.set_u32(2, n_kv_heads as u32);
        ab.set_u32(3, group_size);
        ab.set_u32(4, kv_slot_stride_elems as u32);
        ab.set_f32(5, scale);

        const TG_SIZE_MHA: u32 = 128;
        let shmem_bytes = ((max_seq + TG_SIZE_MHA as usize) * std::mem::size_of::<f32>()) as u64;

        tcb.dispatch_threads("mha_decode_f32_batched_multiseq", (n_heads as u32 * TG_SIZE_MHA, batch as u32, 1), (TG_SIZE_MHA, 1, 1), |enc| {
            enc.set_buffer(0, Some(ab.handle()), 0);
            enc.set_buffer(1, Some(q), 0);
            enc.set_buffer(2, Some(k_cache), k_off_bytes as u64);
            enc.set_buffer(3, Some(v_cache), v_off_bytes as u64);
            enc.set_buffer(4, Some(out), 0);
            enc.set_buffer(5, Some(positions), 0);
            enc.set_buffer(6, Some(regions), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    pub fn add_rmsnorm_fused_batched_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        attn_out_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        x_norm_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
        batch: usize,
    ) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let hidden_u32 = hidden as u32;
        let shmem_bytes = (TG_SIZE as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden_u32);
        ab.set_f32(1, eps);
        let total_threads = (batch as u32) * TG_SIZE;
        tcb.dispatch_threads("add_rmsnorm_fused_batched", (total_threads, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(x_buf), 0);
            enc.set_buffer(1, Some(attn_out_buf), 0);
            enc.set_buffer(2, Some(weight_buf), 0);
            enc.set_buffer(3, Some(x_norm_buf), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })
    }

    pub fn add_inplace_broadcast_tcb(tcb: &mut TokenCommandBuffer<'_>, a_buf: &PinnedBuffer, bias_buf: &PinnedBuffer, dim: usize, batch: usize) -> Result<()> {
        if batch == 0 {
            return Ok(());
        }
        let n = (dim * batch) as u32;
        let dim_u32 = dim as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n);
        ab.set_u32(1, dim_u32);
        let n_tg = n.div_ceil(TG_SIZE);
        tcb.dispatch_threads("add_inplace_broadcast", (n_tg * TG_SIZE, 1, 1), (TG_SIZE, 1, 1), |enc| {
            enc.set_buffer(0, Some(a_buf), 0);
            enc.set_buffer(1, Some(bias_buf), 0);
            enc.set_buffer(2, Some(ab.handle()), 0);
        })
    }

    // ── TQ G4 bitslice decode (the `tq` feature) ────────────────────────────
    //
    // Host driver for `shaders/strand_bitslice.metal::strand_bitslice_decode`,
    // ported from `vendor/strand-decode-kernel/src/metal.rs::BitsliceGpu::decode_q12`.
    // Decode-only (Q12 ints out) — the cleanest identity surface: no MAC/reduce
    // float confound, so the GPU output is held byte-for-byte equal to the CPU
    // oracle `strand_quant::decode::decode_tensor_fixed` (see
    // `tests/tq_trellis_parity.rs`). The fused GEMV/GEMM kernels in the same
    // shader family are a follow-on; this is the bit-identity gate.

    /// GPU `sizeof(BitsliceEntry)` via the `strand_bitslice_entry_sizeof` probe
    /// kernel. The host asserts this equals `size_of::<crate::tq_gpu::BitsliceEntry>()`
    /// before any decode dispatch — the table stride is NEVER hardcoded (the MSL
    /// struct and the Rust `#[repr(C)]` mirror must agree or the row-major table
    /// read diverges). Mirrors `BitsliceGpu::gpu_entry_sizeof`.
    #[cfg(feature = "tq")]
    pub(crate) fn strand_bitslice_entry_sizeof(ctx: &MetalContext) -> Result<u32> {
        let out = ctx.new_buffer(std::mem::size_of::<u32>());
        ctx.dispatch_threads("strand_bitslice_entry_sizeof", (1, 1, 1), (1, 1, 1), |enc| {
            enc.set_buffer(0, Some(&out), 0);
        })?;
        Ok(unsafe { *(out.contents() as *const u32) })
    }

    /// Runtime stride probe for the 40-byte compact compute-for-memory table.
    #[cfg(feature = "tq")]
    pub(crate) fn strand_bitslice_compact_entry_sizeof(ctx: &MetalContext) -> Result<u32> {
        let out = ctx.new_buffer(std::mem::size_of::<u32>());
        ctx.dispatch_threads("strand_bitslice_compact_entry_sizeof", (1, 1, 1), (1, 1, 1), |enc| enc.set_buffer(0, Some(&out), 0))?;
        Ok(unsafe { *(out.contents() as *const u32) })
    }

    /// Decode a STRAND/TQ tensor's trellis-coded payload to its Q12 weights on the
    /// GPU via the G4 bitslice kernel, returning a `Vec<i32>` of length `total`.
    ///
    /// `payload` is the tensor's contiguous k-bit symbol stream (LSB-first); `tbl`
    /// is the per-block table from [`crate::tq_gpu::bake_bitslice_entries`]; `lut`
    /// is the `2^L` Q12 codebook (`strand_quant::codebook::codebook_lut(l_bits)`,
    /// which equals `TrellisConfig::codebook()` for the default `StoredLut` mode).
    ///
    /// Grid = all blocks (one thread owns one block-stream end-to-end), 256
    /// threads/threadgroup, `2^L` Q12 LUT staged once into threadgroup memory. The
    /// payload buffer is zero-padded to a 4-byte word boundary + 8 bytes (the
    /// `WordReader` contract — the kernel's whole-word tail loads must stay in
    /// bounds). Output is bit-identical to `decode_tensor_fixed`.
    ///
    /// Errors if the GPU/host `sizeof(BitsliceEntry)` probe disagrees, or if `lut`
    /// is not exactly `2^l_bits` long.
    #[cfg(feature = "tq")]
    pub(crate) fn decode_strand_bitslice(ctx: &MetalContext, payload: &[u8], tbl: &[crate::tq_gpu::BitsliceEntry], lut: &[i32], total: usize, k_bits: u32, l_bits: u32) -> Result<Vec<i32>> {
        if lut.len() != (1usize << l_bits) {
            return Err(Error::Kernel(format!("decode_strand_bitslice: LUT has {} entries, expected 2^{l_bits} = {}", lut.len(), 1usize << l_bits)));
        }
        // The un-hardcoded stride probe: GPU sizeof(BitsliceEntry) must equal the
        // host #[repr(C)] size or the row-major table read diverges silently.
        let gpu_sz = strand_bitslice_entry_sizeof(ctx)? as usize;
        let host_sz = std::mem::size_of::<crate::tq_gpu::BitsliceEntry>();
        if gpu_sz != host_sz {
            return Err(Error::Kernel(format!(
                "decode_strand_bitslice: GPU sizeof(BitsliceEntry)={gpu_sz} != host {host_sz}; \
                 table stride would diverge"
            )));
        }

        // buffer(0): payload, padded to a word boundary + 8 zero bytes.
        let padded_len = payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..payload.len()].copy_from_slice(payload);
        let w_buf = ctx.new_buffer_with_bytes(&padded);

        // buffer(1): decode-only Q12 output, total i32s.
        let out_buf = ctx.new_buffer(total.max(1) * std::mem::size_of::<i32>());

        // buffer(2): the BitsliceEntry table. #[repr(C)], all-POD fields → a flat
        // byte reinterpret is the upload (matches the reference `upload`).
        let tbl_bytes: &[u8] = unsafe { std::slice::from_raw_parts(tbl.as_ptr() as *const u8, std::mem::size_of_val(tbl)) };
        let tbl_buf = ctx.new_buffer_with_bytes(tbl_bytes);

        // buffer(6): the 2^L Q12 codebook.
        let lut_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(lut));

        let n_blocks = tbl.len() as u32;
        const TG: u32 = 256;
        let n_tg = n_blocks.div_ceil(TG).max(1);
        let shmem_bytes = ((1usize << l_bits) * std::mem::size_of::<i32>()) as u64;

        ctx.dispatch_threads("strand_bitslice_decode", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&w_buf), 0);
            enc.set_buffer(1, Some(&out_buf), 0);
            enc.set_buffer(2, Some(&tbl_buf), 0);
            enc.set_u32(3, n_blocks);
            enc.set_u32(4, k_bits);
            enc.set_u32(5, l_bits);
            enc.set_buffer(6, Some(&lut_buf), 0);
            enc.set_threadgroup_memory_length(0, shmem_bytes);
        })?;

        let ptr = out_buf.contents() as *const i32;
        Ok(unsafe { std::slice::from_raw_parts(ptr, total) }.to_vec())
    }

    // ── TQ G4 bitslice GEMV / GEMM stubs ────────────────────────────────────
    //
    // Placeholder entry points for the fused decode-and-GEMV / decode-and-GEMM
    // kernels that will replace the two-pass (decode then GEMV) path once the
    // fused `strand_bitslice_gemv` / `strand_bitslice_gemm` Metal shaders land
    // in `vendor/strand-decode-kernel/src/metal.rs`. The signatures are frozen;
    // only the bodies are stubs — callers can wire up the dispatch paths without
    // waiting for the kernel to exist.

    /// Fused TQ decode-and-GEMV: decode a STRAND-encoded weight matrix from
    /// `prepared` and multiply by the activations in `x_buf`, writing the result
    /// to `out_buf`. `partials_buf` is a scratch buffer of at least
    /// `n_blocks * sizeof(f32)` bytes (one partial sum per block).
    ///
    /// Two-pass Metal dispatch in a single `dispatch_batch` (one command buffer,
    /// one commit):
    ///   1. `strand_bitslice_gemv_partials` — one thread per block, accumulates a
    ///      partial dot-product into `partials_buf`.
    ///   2. `strand_bitslice_reduce_rows`   — sums the per-block partials for each
    ///      row into `out_buf`.
    ///
    /// Buffer layout mirrors `BitsliceGpu::matvec_dispatch` in
    /// `vendor/strand-decode-kernel/src/metal.rs`.
    #[cfg(feature = "tq")]
    #[allow(dead_code)]
    pub(crate) fn strand_bitslice_gemv(ctx: &MetalContext, prepared: &crate::tq_gpu::TqPreparedGpu, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer, partials_buf: &PinnedBuffer) -> Result<()> {
        // Upload the per-block seek table and the Q12 codebook LUT.
        let tbl_bytes: &[u8] = unsafe { std::slice::from_raw_parts(prepared.entries.as_ptr() as *const u8, std::mem::size_of_val(prepared.entries.as_slice())) };
        let tbl_buf = ctx.new_buffer_with_bytes(tbl_bytes);
        let lut_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&prepared.lut_q12));

        // Upload the payload (bit-stream), padded to a 4-byte word boundary + 8
        // zero bytes so the WordReader's whole-word tail loads stay in bounds.
        let padded_len = prepared.payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..prepared.payload.len()].copy_from_slice(&prepared.payload);
        let w_buf = ctx.new_buffer_with_bytes(&padded);

        let n_blocks = prepared.entries.len() as u32;
        let cols = prepared.cols as u32;
        let rows = prepared.rows as u32;
        let k_bits = prepared.k_bits;
        let l_bits = prepared.l_bits;
        let bpr = cols / 256; // blocks per row

        // threadgroup shmem: stage the 2^L Q12 LUT once per threadgroup.
        let shmem_bytes = ((1usize << l_bits) * std::mem::size_of::<i32>()) as u64;

        // Pass 1: gemv_partials — grid = one thread per block, tg = 256.
        const TG: u32 = 256;
        let n_tg_partials = n_blocks.div_ceil(TG).max(1);
        // Pass 2: reduce_rows — grid = one thread per row, tg = 256.
        let n_tg_reduce = rows.div_ceil(TG).max(1);

        ctx.dispatch_batch(|batch| {
            // ── Pass 1: strand_bitslice_gemv_partials ──────────────────────
            // buffer(0): w_bits  (payload)
            // buffer(1): x       (activation f32 vector, length = cols)
            // buffer(2): partials (scratch f32, one per block)
            // buffer(3): tbl     (BitsliceEntry seek table)
            // buffer(4): n_blocks (constant u32)
            // buffer(5): cols    (constant u32)
            // buffer(6): k_bits  (constant u32)
            // buffer(7): l_bits  (constant u32)
            // buffer(8): lut_q12 (2^L i32 codebook)
            // shmem(0):  2^L * sizeof(i32) — staged codebook
            batch.dispatch_threads("strand_bitslice_gemv_partials", (n_tg_partials * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(x_buf), 0);
                enc.set_buffer(2, Some(partials_buf), 0);
                enc.set_buffer(3, Some(&tbl_buf), 0);
                enc.set_u32(4, n_blocks);
                enc.set_u32(5, cols);
                enc.set_u32(6, k_bits);
                enc.set_u32(7, l_bits);
                enc.set_buffer(8, Some(&lut_buf), 0);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })?;
            // ── Pass 2: strand_bitslice_reduce_rows ───────────────────────
            // buffer(0): partials (one f32 per block, from pass 1)
            // buffer(1): y       (output f32, length = rows)
            // buffer(2): rows    (constant u32)
            // buffer(3): bpr     (blocks per row = cols / 256, constant u32)
            batch.dispatch_threads("strand_bitslice_reduce_rows", (n_tg_reduce * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(partials_buf), 0);
                enc.set_buffer(1, Some(out_buf), 0);
                enc.set_u32(2, rows);
                enc.set_u32(3, bpr);
            })
        })
    }

    /// Shared inner of the TCB TQ GEMV: the (optional) RHT-cols activation
    /// transform plus the `strand_bitslice_gemv_partials` pass. Returns the byte
    /// offset the partials pass actually read its activation from — 0 when the RHT
    /// transform ran (it writes the transformed vector to `gpu.rht_x_buf` at
    /// element 0), else the caller's `x_off_bytes`. The reduce pass differs between
    /// the base (overwrite) and residual (accumulate) wrappers, so it is NOT done
    /// here. (GAP 1: serves the `--rht-cols` quality recipe — the bitslice GEMV
    /// dots rotated weights against `T(x)`, exactly `outlier_mac::matvec_rht`'s
    /// col path, with the activation transform computed ONCE on GPU.)
    #[cfg(feature = "tq")]
    fn strand_bitslice_partials_tcb(tcb: &mut TokenCommandBuffer<'_>, gpu: &crate::tq_gpu::TqGpuReady, x_buf: &PinnedBuffer, x_off_bytes: usize) -> Result<()> {
        const TG: u32 = 256;
        // RHT-cols: transform the activation ONCE (x@x_off → rht_x_buf@0). The
        // partials pass then reads rht_x_buf at offset 0 (rht_x_buf is cols-long).
        if gpu.rht_mode == 2 {
            let rht_x = gpu.rht_x_buf.as_ref().ok_or_else(|| crate::Error::Metal("RhtMode::Cols TqGpuReady missing rht_x_buf".into()))?;
            let seed_lo = (gpu.rht_seed & 0xffff_ffff) as u32;
            let seed_hi = (gpu.rht_seed >> 32) as u32;
            let x_base_elems = (x_off_bytes / std::mem::size_of::<f32>()) as u32;
            let n_blocks = gpu.rht_n_blocks.max(1);
            let n_tg = n_blocks.div_ceil(TG).max(1);
            tcb.dispatch_threads("strand_rht_forward_cols", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf), 0);
                enc.set_buffer(1, Some(rht_x), 0);
                enc.set_u32(2, gpu.cols);
                enc.set_u32(3, seed_lo);
                enc.set_u32(4, seed_hi);
                enc.set_u32(5, x_base_elems);
            })?;
        }
        // The activation the partials pass dots against: the transformed scratch
        // (offset 0) when RHT ran, else the caller's raw slice.
        let (act_buf, act_off): (&PinnedBuffer, u64) = if gpu.rht_mode == 2 { (gpu.rht_x_buf.as_ref().unwrap(), 0) } else { (x_buf, x_off_bytes as u64) };
        let kernel = match gpu.runtime_path {
            crate::tq_gpu::TqRuntimePath::Stored => "strand_bitslice_gemv_partials",
            crate::tq_gpu::TqRuntimePath::CompactMetadata => "strand_bitslice_gemv_partials_compact",
            crate::tq_gpu::TqRuntimePath::HashedQuantile => "strand_bitslice_gemv_partials_hashed",
            crate::tq_gpu::TqRuntimePath::ComputedAcklam => "strand_bitslice_gemv_partials_computed",
        };
        tcb.dispatch_threads(kernel, (gpu.n_tg_partials * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&gpu.w_buf), 0);
            enc.set_buffer(1, Some(act_buf), act_off);
            enc.set_buffer(2, Some(&gpu.partials_buf), 0);
            enc.set_buffer(3, Some(&gpu.tbl_buf), 0);
            enc.set_u32(4, gpu.n_blocks);
            enc.set_u32(5, gpu.cols);
            enc.set_u32(6, gpu.k_bits);
            enc.set_u32(7, gpu.l_bits);
            enc.set_buffer(8, Some(&gpu.lut_buf), 0);
            if gpu.shmem_bytes != 0 {
                enc.set_threadgroup_memory_length(0, gpu.shmem_bytes);
            }
        })
    }

    /// The OUTL sparse-correction pass (GAP 1): `y[row] += resid * x_raw[col]` over
    /// this tensor's pre-resolved outliers, using the RAW (un-transformed)
    /// activation — exactly `outlier_mac::matvec_rht`'s residual loop. No-op when
    /// the tensor has no outliers. `x_off_bytes` / `out_off_bytes` are byte offsets
    /// into the raw activation and the output (same slices the GEMV used).
    #[cfg(feature = "tq")]
    fn strand_outlier_correct_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        gpu: &crate::tq_gpu::TqGpuReady,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
    ) -> Result<()> {
        if gpu.n_outl == 0 {
            return Ok(());
        }
        const TG: u32 = 256;
        let n_tg = gpu.n_outl.div_ceil(TG).max(1);
        // Bind BOTH buffers at offset 0 and carry the element offsets as constants:
        // the kernel indexes `x_raw[x_base_elems + col]` / `y[y_base_elems + row]`,
        // so the offset must NOT also be applied at bind time (that would
        // double-count it). This matters for the residual/multiseq paths where
        // out_off_bytes != 0.
        let x_base_elems = (x_off_bytes / std::mem::size_of::<f32>()) as u32;
        let y_base_elems = (out_off_bytes / std::mem::size_of::<f32>()) as u32;
        tcb.dispatch_threads("strand_outlier_correct", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&gpu.outl_buf), 0);
            enc.set_buffer(1, Some(out_buf), 0);
            enc.set_buffer(2, Some(x_buf), 0);
            enc.set_u32(3, gpu.n_outl);
            enc.set_u32(4, x_base_elems);
            enc.set_u32(5, y_base_elems);
        })
    }

    /// TCB-compatible TQ GEMV: encodes the (optional) RHT-cols transform, the
    /// bitslice-GEMV partials + reduce (OVERWRITE), and the (optional) OUTL sparse
    /// correction into `tcb` using pre-uploaded [`crate::tq_gpu::TqGpuReady`]
    /// buffers. Zero per-inference allocations. `x_off_bytes` / `out_off_bytes` are
    /// Metal buffer byte offsets for the activation input and output slices.
    ///
    /// Serves the FULL quality recipe: raw Q12 (RhtMode::None), `--rht-cols`
    /// (RhtMode::Cols → activation transformed once on GPU), and `--outlier-channel`
    /// (sparse correction in the un-rotated domain) — bit-faithful (within fp
    /// reduction grouping) to `crate::tq::StrandTensor::matvec` / `outlier_mac`.
    #[cfg(feature = "tq")]
    pub(crate) fn strand_bitslice_gemv_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        gpu: &crate::tq_gpu::TqGpuReady,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
    ) -> Result<()> {
        const TG: u32 = 256;
        strand_bitslice_partials_tcb(tcb, gpu, x_buf, x_off_bytes)?;
        // Reduce (OVERWRITE): seeds `out`.
        tcb.dispatch_threads("strand_bitslice_reduce_rows", (gpu.n_tg_reduce * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&gpu.partials_buf), 0);
            enc.set_buffer(1, Some(out_buf), out_off_bytes as u64);
            enc.set_u32(2, gpu.rows);
            enc.set_u32(3, gpu.bpr);
        })?;
        // OUTL sparse correction (+=) on the RAW activation, after `out` is seeded.
        strand_outlier_correct_tcb(tcb, gpu, x_buf, x_off_bytes, out_buf, out_off_bytes)
    }

    /// TCB-compatible TQ GEMV that ACCUMULATES into `out` (the residual second
    /// pass of the two-part serving recipe — see `strand_bitslice_reduce_rows_accum`).
    /// Identical to [`strand_bitslice_gemv_tcb`] except the reduce pass adds into
    /// `out[gidx]` instead of overwriting it, so calling
    ///   strand_bitslice_gemv_tcb(base, x, out);          // seeds out
    ///   strand_bitslice_gemv_tcb_accum(residual, x, out) // out += residual·x
    /// yields `y = decode(base)·x + decode(residual)·x`, the decoded-sum the
    /// residual STRAND bake targets, with both passes kept compressed in RAM.
    /// `out` MUST already hold the base pass's result (or be zeroed). Each pass
    /// applies its own RHT-cols transform and OUTL correction. Zero per-inference
    /// allocations. `x_off_bytes` / `out_off_bytes` are Metal buffer byte offsets.
    #[cfg(feature = "tq")]
    pub(crate) fn strand_bitslice_gemv_tcb_accum(
        tcb: &mut TokenCommandBuffer<'_>,
        gpu: &crate::tq_gpu::TqGpuReady,
        x_buf: &PinnedBuffer,
        x_off_bytes: usize,
        out_buf: &PinnedBuffer,
        out_off_bytes: usize,
    ) -> Result<()> {
        const TG: u32 = 256;
        strand_bitslice_partials_tcb(tcb, gpu, x_buf, x_off_bytes)?;
        // Reduce (ACCUMULATE): y[gidx] += acc.
        tcb.dispatch_threads("strand_bitslice_reduce_rows_accum", (gpu.n_tg_reduce * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&gpu.partials_buf), 0);
            enc.set_buffer(1, Some(out_buf), out_off_bytes as u64);
            enc.set_u32(2, gpu.rows);
            enc.set_u32(3, gpu.bpr);
        })?;
        strand_outlier_correct_tcb(tcb, gpu, x_buf, x_off_bytes, out_buf, out_off_bytes)
    }

    /// Shared B=1..=8 TQ batch-major projection for speculative verification.
    /// One Metal thread decodes one 256-weight block once, then reuses the
    /// decoded f32 weight across all activation rows. `x_buf` and `out_buf` are
    /// contiguous `(batch, cols)` / `(batch, rows)` arena buffers. The optional
    /// RHT-cols transform, OUTL correction, runtime codebook path, and residual
    /// accumulate semantics are the same recipe components as the GEMV path.
    #[cfg(feature = "tq")]
    fn strand_bitslice_gemm_small_tcb_inner(
        tcb: &mut TokenCommandBuffer<'_>,
        gpu: &crate::tq_gpu::TqGpuReady,
        x_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        batch: usize,
        accumulate: bool,
    ) -> Result<()> {
        if !(1..=8).contains(&batch) {
            return Err(Error::Kernel(format!("strand_bitslice_gemm_small_tcb: batch must be in 1..=8 (got {batch})")));
        }
        const TG: u32 = 256;
        let batch_u32 = batch as u32;

        // RHT-cols over the batch-major activation matrix. Each work item owns
        // one 256-wide block, preserving the scalar kernel's butterfly order.
        if gpu.rht_mode == 2 {
            let rht_x = gpu.rht_x_buf.as_ref().ok_or_else(|| Error::Metal("RhtMode::Cols TqGpuReady missing batched rht_x_buf".into()))?;
            let seed_lo = (gpu.rht_seed & 0xffff_ffff) as u32;
            let seed_hi = (gpu.rht_seed >> 32) as u32;
            let work = gpu.rht_n_blocks.saturating_mul(batch_u32);
            let n_tg = work.div_ceil(TG).max(1);
            tcb.dispatch_threads("strand_rht_forward_cols_batched", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(x_buf), 0);
                enc.set_buffer(1, Some(rht_x), 0);
                enc.set_u32(2, gpu.cols);
                enc.set_u32(3, seed_lo);
                enc.set_u32(4, seed_hi);
                enc.set_u32(5, batch_u32);
            })?;
        }
        let act_buf = if gpu.rht_mode == 2 { gpu.rht_x_buf.as_ref().unwrap() } else { x_buf };

        tcb.dispatch_threads(gpu.runtime_path.small_batch_kernel_name(), (gpu.n_tg_partials * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&gpu.w_buf), 0);
            enc.set_buffer(1, Some(act_buf), 0);
            enc.set_buffer(2, Some(&gpu.partials_buf), 0);
            enc.set_buffer(3, Some(&gpu.tbl_buf), 0);
            enc.set_u32(4, gpu.n_blocks);
            enc.set_u32(5, gpu.cols);
            enc.set_u32(6, gpu.k_bits);
            enc.set_u32(7, gpu.l_bits);
            enc.set_buffer(8, Some(&gpu.lut_buf), 0);
            enc.set_u32(9, batch_u32);
            if gpu.shmem_bytes != 0 {
                enc.set_threadgroup_memory_length(0, gpu.shmem_bytes);
            }
        })?;

        let n_out = gpu.rows.saturating_mul(batch_u32);
        let n_tg_reduce = n_out.div_ceil(TG).max(1);
        let reduce_kernel = if accumulate { "strand_bitslice_reduce_rows_small_batch_accum" } else { "strand_bitslice_reduce_rows_small_batch" };
        tcb.dispatch_threads(reduce_kernel, (n_tg_reduce * TG, 1, 1), (TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(&gpu.partials_buf), 0);
            enc.set_buffer(1, Some(out_buf), 0);
            enc.set_u32(2, gpu.rows);
            enc.set_u32(3, gpu.bpr);
            enc.set_u32(4, gpu.n_blocks);
            enc.set_u32(5, batch_u32);
        })?;

        if gpu.n_outl != 0 {
            let work = gpu.n_outl.saturating_mul(batch_u32);
            let n_tg = work.div_ceil(TG).max(1);
            tcb.dispatch_threads("strand_outlier_correct_batched", (n_tg * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&gpu.outl_buf), 0);
                enc.set_buffer(1, Some(out_buf), 0);
                enc.set_buffer(2, Some(x_buf), 0);
                enc.set_u32(3, gpu.n_outl);
                enc.set_u32(4, gpu.rows);
                enc.set_u32(5, gpu.cols);
                enc.set_u32(6, batch_u32);
            })?;
        }
        Ok(())
    }

    /// Overwrite-form B=1..=8 TQ batch-major GEMM.
    #[cfg(feature = "tq")]
    pub(crate) fn strand_bitslice_gemm_small_tcb(tcb: &mut TokenCommandBuffer<'_>, gpu: &crate::tq_gpu::TqGpuReady, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer, batch: usize) -> Result<()> {
        strand_bitslice_gemm_small_tcb_inner(tcb, gpu, x_buf, out_buf, batch, false)
    }

    /// Accumulate-form B=1..=8 TQ batch-major GEMM for the residual second pass.
    #[cfg(feature = "tq")]
    pub(crate) fn strand_bitslice_gemm_small_tcb_accum(tcb: &mut TokenCommandBuffer<'_>, gpu: &crate::tq_gpu::TqGpuReady, x_buf: &PinnedBuffer, out_buf: &PinnedBuffer, batch: usize) -> Result<()> {
        strand_bitslice_gemm_small_tcb_inner(tcb, gpu, x_buf, out_buf, batch, true)
    }

    /// Fused TQ decode-and-GEMM: decode a STRAND-encoded weight matrix from
    /// `prepared` and multiply by the `batch`-wide activation matrix in
    /// `xt_buf` (column-major / transposed: row = feature, stride = batch), writing
    /// `rows * batch` f32 results to `out_buf`. `partials_buf` is scratch of at
    /// least `n_blocks * batch * sizeof(f32)` bytes.
    ///
    /// Selects the right kernel variant based on `batch`:
    ///   4  → `strand_bitslice_gemm_partials_b4`
    ///   16 → `strand_bitslice_gemm_partials_b16`
    ///   64 → `strand_bitslice_gemm_partials_b64`
    ///
    /// Two-pass Metal dispatch in a single `dispatch_batch`, mirroring
    /// `BitsliceGpu::gemm_dispatch` in `vendor/strand-decode-kernel/src/metal.rs`.
    #[cfg(feature = "tq")]
    #[allow(dead_code)]
    pub(crate) fn strand_bitslice_gemm(
        ctx: &MetalContext,
        prepared: &crate::tq_gpu::TqPreparedGpu,
        xt_buf: &PinnedBuffer,
        out_buf: &PinnedBuffer,
        partials_buf: &PinnedBuffer,
        batch: u32,
    ) -> Result<()> {
        let kernel_name = match batch {
            4 => "strand_bitslice_gemm_partials_b4",
            16 => "strand_bitslice_gemm_partials_b16",
            64 => "strand_bitslice_gemm_partials_b64",
            _ => return Err(Error::Kernel(format!("strand_bitslice_gemm: batch must be 4, 16, or 64 (got {batch})"))),
        };

        // Upload the per-block seek table and the Q12 codebook LUT.
        let tbl_bytes: &[u8] = unsafe { std::slice::from_raw_parts(prepared.entries.as_ptr() as *const u8, std::mem::size_of_val(prepared.entries.as_slice())) };
        let tbl_buf = ctx.new_buffer_with_bytes(tbl_bytes);
        let lut_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<i32, u8>(&prepared.lut_q12));

        // Payload padded to word boundary + 8 zero bytes (WordReader contract).
        let padded_len = prepared.payload.len().div_ceil(4) * 4 + 8;
        let mut padded = vec![0u8; padded_len];
        padded[..prepared.payload.len()].copy_from_slice(&prepared.payload);
        let w_buf = ctx.new_buffer_with_bytes(&padded);

        let n_blocks = prepared.entries.len() as u32;
        let cols = prepared.cols as u32;
        let rows = prepared.rows as u32;
        let k_bits = prepared.k_bits;
        let l_bits = prepared.l_bits;
        let bpr = cols / 256; // blocks per row

        // threadgroup shmem: 2^L Q12 LUT staged once per threadgroup.
        let shmem_bytes = ((1usize << l_bits) * std::mem::size_of::<i32>()) as u64;

        const TG: u32 = 256;
        // Pass 1 grid: one thread per block (same as GEMV).
        let n_tg_partials = n_blocks.div_ceil(TG).max(1);
        // Pass 2 grid: one thread per output element (rows × batch).
        let n_out = (rows as u64) * (batch as u64);
        let n_tg_reduce = (n_out as u32).div_ceil(TG).max(1);

        ctx.dispatch_batch(|batch_cb| {
            // ── Pass 1: strand_bitslice_gemm_partials_b{4,16,64} ──────────
            // Same buffer layout as gemv_partials; xt_buf carries the
            // column-major activation matrix (col = one activation vector).
            batch_cb.dispatch_threads(kernel_name, (n_tg_partials * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(&w_buf), 0);
                enc.set_buffer(1, Some(xt_buf), 0);
                enc.set_buffer(2, Some(partials_buf), 0);
                enc.set_buffer(3, Some(&tbl_buf), 0);
                enc.set_u32(4, n_blocks);
                enc.set_u32(5, cols);
                enc.set_u32(6, k_bits);
                enc.set_u32(7, l_bits);
                enc.set_buffer(8, Some(&lut_buf), 0);
                enc.set_threadgroup_memory_length(0, shmem_bytes);
            })?;
            // ── Pass 2: strand_bitslice_reduce_rows_gemm ──────────────────
            // buffer(0): partials (n_blocks * batch f32)
            // buffer(1): y       (rows * batch f32)
            // buffer(2): rows    (constant u32)
            // buffer(3): bpr     (blocks per row, constant u32)
            // buffer(4): batch   (constant u32)
            batch_cb.dispatch_threads("strand_bitslice_reduce_rows_gemm", (n_tg_reduce * TG, 1, 1), (TG, 1, 1), |enc| {
                enc.set_buffer(0, Some(partials_buf), 0);
                enc.set_buffer(1, Some(out_buf), 0);
                enc.set_u32(2, rows);
                enc.set_u32(3, bpr);
                enc.set_u32(4, batch);
            })
        })
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

    #[test]
    fn longrope_neox_unit_factors_is_plain_neox() {
        // With factors == 1 and mscale == 1, rope_inplace_longrope is
        // plain NEOX RoPE: verify against the closed form on head_dim=8.
        let head_dim = 8usize;
        let half = head_dim / 2;
        let base = 10_000.0f32;
        let pos = 5u32;
        let factors = vec![1.0f32; half];
        let mut x: Vec<f32> = (0..head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let orig = x.clone();
        rope_inplace_longrope(&mut x, pos, base, &factors, 1.0);
        for i in 0..half {
            let inv_freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);
            let theta = pos as f32 * inv_freq;
            let (s, c) = theta.sin_cos();
            let x0 = orig[i];
            let x1 = orig[i + half];
            assert!((x[i] - (x0 * c - x1 * s)).abs() < 1e-6);
            assert!((x[i + half] - (x0 * s + x1 * c)).abs() < 1e-6);
        }
    }

    #[test]
    fn longrope_factor_lowers_frequency() {
        // A larger ext_factor divides the inverse frequency, so the
        // rotation angle shrinks. At pos=1, dim i=1, factor 2 vs 1 must
        // halve the effective angle (inv_freq scales by 1/factor).
        let head_dim = 8usize;
        let half = head_dim / 2;
        let base = 10_000.0f32;
        let i = 1usize;
        let inv_freq = 1.0 / base.powf(2.0 * i as f32 / head_dim as f32);

        let mut a = vec![0.0f32; head_dim];
        a[i] = 1.0; // (x_i, x_{i+half}) = (1, 0) → reads out (cos, sin)
        let mut b = a.clone();
        let mut f1 = vec![1.0f32; half];
        let mut f2 = vec![1.0f32; half];
        f1[i] = 1.0;
        f2[i] = 2.0;
        rope_inplace_longrope(&mut a, 1, base, &f1, 1.0);
        rope_inplace_longrope(&mut b, 1, base, &f2, 1.0);
        let angle_a = a[i + half].atan2(a[i]); // = inv_freq
        let angle_b = b[i + half].atan2(b[i]); // = inv_freq / 2
        assert!((angle_a - inv_freq).abs() < 1e-6);
        assert!((angle_b - inv_freq / 2.0).abs() < 1e-6);
    }

    #[test]
    fn gelu_mul_matches_reference() {
        let gate = [0.0f32, 1.0, -1.0, 2.5];
        let up = [1.0f32, 2.0, 3.0, 0.5];
        let mut out = [0.0f32; 4];
        gelu_mul(&gate, &up, &mut out);
        // Reference gelu_tanh computed independently.
        let gelu = |x: f32| {
            let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        };
        for i in 0..4 {
            let expect = gelu(gate[i]) * up[i];
            assert!((out[i] - expect).abs() < 1e-6, "i={i}: got {} want {expect}", out[i]);
        }
        // gelu(0) == 0, so out[0] must be exactly 0.
        assert_eq!(out[0], 0.0);
    }

    #[test]
    fn logit_softcap_bounds_and_noop() {
        // cap<=0 is a no-op.
        let mut a = [5.0f32, -3.0, 100.0];
        logit_softcap_inplace(&mut a, 0.0);
        assert_eq!(a, [5.0, -3.0, 100.0]);

        // With cap=30, output is bounded to (-30, 30) and monotone.
        let cap = 30.0f32;
        let mut b = [0.0f32, 30.0, 1000.0, -1000.0];
        logit_softcap_inplace(&mut b, cap);
        assert!((b[0] - 0.0).abs() < 1e-6); // tanh(0)=0
        assert!((b[1] - cap * (1.0f32).tanh()).abs() < 1e-5);
        // tanh saturates to exactly 1.0 in f32 for large args, so the
        // capped value reaches cap; assert bounded (<=) and near-cap.
        assert!(b[2] <= cap && b[2] > cap - 1e-2);
        assert!(b[3] >= -cap && b[3] < -cap + 1e-2);
    }

    /// `rope_inplace_scaled(..., None)` must be bit-identical to the
    /// unscaled `rope_inplace` so Qwen2 / DeepSeek-V2 paths can swap
    /// without behavioural change.
    #[test]
    fn rope_scaled_none_matches_unscaled() {
        let mut rng_state: u32 = 0xC0FFEEu32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((rng_state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        };
        let head_dim = 128;
        let a: Vec<f32> = (0..head_dim).map(|_| next()).collect();
        for &(pos, base) in &[(0u32, 1_000_000.0f32), (37, 500_000.0), (4096, 1_000_000.0)] {
            let mut a_unscaled = a.clone();
            let mut a_scaled = a.clone();
            rope_inplace(&mut a_unscaled, pos, base);
            rope_inplace_scaled(&mut a_scaled, pos, base, None);
            for i in 0..head_dim {
                assert_eq!(a_unscaled[i].to_bits(), a_scaled[i].to_bits(), "rope_scaled(None) diverged from rope_inplace at pos={pos} base={base} i={i}");
            }
        }
    }

    /// Llama-3.1 reference parameters. Verify the three regimes:
    ///   (a) high-frequency (small i): freq unchanged → angle = pos * freq
    ///   (b) low-frequency  (large i): freq divided by `factor`
    ///   (c) middle band: smooth interpolation between (a) and (b)
    #[test]
    fn rope_scaled_llama3_regimes() {
        let head_dim = 64usize;
        let base = 500_000.0f32;
        let pos = 1u32; // pos=1 makes the rotated angle exactly equal to freq_eff
        let scaling = Llama3RopeScaling { factor: 8.0, low_freq_factor: 1.0, high_freq_factor: 4.0, original_max_position_embeddings: 8192 };

        // Use a NEOX vector of pairs (cos₀=1, sin₀=0) per half-pair so that after
        // one rotation step the resulting (x0, x1) = (cos θ, sin θ) — i.e. we
        // can read freq_eff[i] directly off the output without inversion.
        let mut x = vec![0.0f32; head_dim];
        let half = head_dim / 2;
        for i in 0..head_dim / 2 {
            x[i] = 1.0;
            x[i + half] = 0.0;
        }
        rope_inplace_scaled(&mut x, pos, base, Some(scaling));

        let two_pi = std::f32::consts::TAU;
        let low_wavelen = scaling.original_max_position_embeddings as f32 / scaling.low_freq_factor;
        let high_wavelen = scaling.original_max_position_embeddings as f32 / scaling.high_freq_factor;

        let mut saw_unscaled = false;
        let mut saw_scaled = false;
        let mut saw_smooth = false;
        for i in 0..head_dim / 2 {
            let inv_freq = base.powf(2.0 * i as f32 / head_dim as f32);
            let freq = 1.0 / inv_freq;
            let wavelen = two_pi / freq;
            let recovered_freq_eff = x[i + half].atan2(x[i]); // since pos=1, θ = freq_eff
            if wavelen < high_wavelen {
                // Regime (a): unchanged.
                assert!((recovered_freq_eff - freq).abs() < 1e-5, "i={i}: expected unscaled freq={freq}, got {recovered_freq_eff}");
                saw_unscaled = true;
            } else if wavelen > low_wavelen {
                // Regime (b): freq / factor.
                let expected = freq / scaling.factor;
                assert!((recovered_freq_eff - expected).abs() < 1e-5, "i={i}: expected freq/factor={expected}, got {recovered_freq_eff}");
                saw_scaled = true;
            } else {
                // Regime (c): smooth.
                let smooth = (scaling.original_max_position_embeddings as f32 / wavelen - scaling.low_freq_factor) / (scaling.high_freq_factor - scaling.low_freq_factor);
                let expected = (1.0 - smooth) * (freq / scaling.factor) + smooth * freq;
                assert!((recovered_freq_eff - expected).abs() < 1e-5, "i={i}: expected smooth={expected}, got {recovered_freq_eff}");
                saw_smooth = true;
            }
        }
        // Confirm the test actually exercised all three regimes.
        assert!(saw_unscaled && saw_scaled && saw_smooth, "test did not cover all three regimes");
    }
}

/// Residual two-part STRAND **serving** parity gate (HAWKING_TQ_RESIDUAL).
///
/// The quality breakthrough bakes `W ≈ STRAND_b1(W) + STRAND_b2(W − STRAND_b1(W))`
/// and `residual_bake.py` materialises the DECODED SUM `decode(base) + decode(res)`
/// as f16. For SERVING we keep BOTH passes COMPRESSED and sum them at GEMV time:
/// `y = bitslice_gemv(base, x) + bitslice_gemv(residual, x)` — base via
/// `strand_bitslice_gemv_tcb` (seeds `out`), residual via
/// `strand_bitslice_gemv_tcb_accum` (`out += residual·x`). This gate asserts that
/// GPU two-part result equals the CPU decoded-sum GEMV (the exact quantity
/// `residual_bake.py` yields) within fp tolerance, on a synthetic tensor with
/// `in_features % 256 == 0` (the deploy alignment invariant).
///
/// Both passes use `RhtMode::None` (raw, unrotated) — which is exactly what the
/// bitslice GEMV kernel serves (it decodes raw Q12 and dots directly; it does NOT
/// apply the RHT-cols activation transform or OUTL overwrites). So a `--no-rht`,
/// no-outlier two-part bake is the artifact this serving path reproduces
/// bit-faithfully; an RHT-cols/OUTL bake (what `residual_bake.py` emits today)
/// would need those serving steps wired separately — see the report.
#[cfg(all(test, target_os = "macos", feature = "tq"))]
mod residual_serve_tests {
    use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use crate::tq_gpu::{bake_bitslice_entries, bake_compact_bitslice_entries, TqGpuReady, TqPreparedGpu};
    use strand_quant::decode::decode_tensor_fixed;
    use strand_quant::encode::{encode_tensor, EncodedTensor};
    use strand_quant::TrellisConfig;

    fn synth_w(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed as f32) * 0.0137).sin() * 0.5).collect()
    }
    fn synth_x(n: usize, seed: u64) -> Vec<f32> {
        (0..n).map(|i| ((i as f32 + seed as f32) * 0.07).cos()).collect()
    }

    /// Build a `TqPreparedGpu` straight from a raw `EncodedTensor` (RhtMode::None),
    /// mirroring `TqPreparedGpu::from_strand_tensor` without needing a StrandTensor.
    fn prepare(enc: &EncodedTensor, cfg: &TrellisConfig, rows: usize, cols: usize) -> TqPreparedGpu {
        let entries = bake_bitslice_entries(enc, cfg).expect("scalar bake (n<=256 per block)");
        let compact_entries = bake_compact_bitslice_entries(enc, cfg).expect("compact scalar bake");
        TqPreparedGpu {
            payload: enc.bits.clone(),
            entries,
            compact_entries,
            lut_q12: cfg.codebook().into_owned(),
            k_bits: cfg.k_bits,
            l_bits: cfg.l_bits,
            rows,
            cols,
            rht_mode: 0,
            rht_seed: 0,
            outliers: Vec::new(),
            bpw: cfg.k_bits as f32 / cfg.vec_dim() as f32,
        }
    }

    /// Decode an EncodedTensor to f32 weights (Q12 → f32) the way the decoded-sum
    /// reference (residual_bake.py) does: `decode_tensor_fixed` then `* 1/2^shift`.
    fn decode_f32(enc: &EncodedTensor, cfg: &TrellisConfig) -> Vec<f32> {
        let inv = crate::tq::q12_to_f32();
        decode_tensor_fixed(enc, cfg).into_iter().map(|q| q as f32 * inv).collect()
    }

    fn read_back(buf: &PinnedBuffer, n: usize) -> Vec<f32> {
        let p = buf.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
    }

    /// Core gate: GPU two-part GEMV == CPU decoded-sum GEMV within fp tolerance.
    /// Returns (max_abs_err, max_rel_err) for reporting.
    fn run_case(ctx: &MetalContext, rows: usize, cols: usize, b1: f64, b2: f64, seed: u64) -> (f32, f32) {
        assert_eq!(cols % 256, 0, "deploy invariant: in_features % 256 == 0");
        let total = rows * cols;

        // ── Residual bake (in-process), RhtMode::None ──────────────────────────
        // Pass 1: base STRAND of W.
        let w = synth_w(total, seed);
        let cfg_b = TrellisConfig::for_bpw(b1);
        let enc_base = encode_tensor(&w, &cfg_b);
        let wh1 = decode_f32(&enc_base, &cfg_b); // decode(base)

        // Pass 2: residual STRAND of (W − decode(base)).
        let resid: Vec<f32> = w.iter().zip(&wh1).map(|(a, b)| a - b).collect();
        let cfg_r = TrellisConfig::for_bpw(b2);
        let enc_res = encode_tensor(&resid, &cfg_r);
        let rh = decode_f32(&enc_res, &cfg_r); // decode(residual)

        // Decoded SUM — exactly what residual_bake.py writes (out = Wh1 + Rh).
        let w_sum: Vec<f32> = wh1.iter().zip(&rh).map(|(a, b)| a + b).collect();

        // CPU reference: y_ref = W_sum · x  (one summed-weight dot per row).
        let x = synth_x(cols, seed ^ 0x5a5a);
        let mut y_ref = vec![0.0f32; rows];
        for o in 0..rows {
            let row = &w_sum[o * cols..(o + 1) * cols];
            let mut acc = 0.0f32;
            for i in 0..cols {
                acc += row[i] * x[i];
            }
            y_ref[o] = acc;
        }

        // ── GPU two-part serve: base (overwrite) then residual (accumulate) ────
        let prep_base = prepare(&enc_base, &cfg_b, rows, cols);
        let prep_res = prepare(&enc_res, &cfg_r, rows, cols);
        let gpu_base: TqGpuReady = prep_base.upload_to_gpu(ctx).expect("upload base");
        let gpu_res: TqGpuReady = prep_res.upload_to_gpu(ctx).expect("upload residual");

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());

        let mut tcb = TokenCommandBuffer::new(ctx);
        // base pass seeds out; residual pass accumulates: out = base·x + res·x.
        super::strand_bitslice_gemv_tcb(&mut tcb, &gpu_base, &x_buf, 0, &out_buf, 0).expect("base gemv");
        super::strand_bitslice_gemv_tcb_accum(&mut tcb, &gpu_res, &x_buf, 0, &out_buf, 0).expect("residual gemv accum");
        tcb.commit_and_wait().expect("commit");

        let y_gpu = read_back(&out_buf, rows);

        // Error vs the decoded-sum reference (only fp reduction order differs:
        // two per-pass accumulations on GPU vs one summed-weight dot on CPU).
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        for o in 0..rows {
            let abs = (y_gpu[o] - y_ref[o]).abs();
            let rel = abs / (1.0 + y_ref[o].abs());
            max_abs = max_abs.max(abs);
            max_rel = max_rel.max(rel);
        }
        (max_abs, max_rel)
    }

    /// The two-part GPU serve matches the decoded-sum across the deploy bit-pairs
    /// and a 7B-shaped projection (cols ∈ {3584, 18944} are %256==0; we use a few
    /// rows to keep the synthetic encode fast). Tolerance is generous-but-tight:
    /// these are f32 dot products differing only in reduction grouping.
    #[test]
    fn residual_two_part_gemv_matches_decoded_sum() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[residual_serve] no Metal device; skipping two-part serve gate");
            return;
        };

        // (rows, cols, b1, b2): the proven residual pairs (3+2, 2+2) and a couple
        // of shapes including 7B in_features (3584) and FFN width (18944).
        let cases: [(usize, usize, f64, f64); 5] = [
            (8, 256, 3.0, 2.0),
            (8, 512, 3.0, 2.0),
            (8, 256, 2.0, 2.0),
            (4, 3584, 3.0, 2.0),  // 7B attn in_features
            (2, 18944, 3.0, 2.0), // 7B FFN in_features
        ];
        // Per-element relative tolerance. A summed dot of `cols` f32 terms vs two
        // per-pass dots accumulates ~cols * eps rounding; 2e-3 covers cols≈19k.
        const REL_TOL: f32 = 2e-3;

        let mut worst_abs = 0.0f32;
        let mut worst_rel = 0.0f32;
        for &(rows, cols, b1, b2) in &cases {
            let (max_abs, max_rel) = run_case(&ctx, rows, cols, b1, b2, 0xC0FFEE);
            println!("[residual_serve] {rows}x{cols} base{b1}+res{b2}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
            assert!(max_rel <= REL_TOL, "{rows}x{cols} base{b1}+res{b2}: max_rel {max_rel:.3e} > {REL_TOL:.1e} (GPU two-part vs decoded-sum)");
            worst_abs = worst_abs.max(max_abs);
            worst_rel = worst_rel.max(max_rel);
        }
        println!("[residual_serve] PASS — GPU two-part GEMV == decoded-sum across {} cases; worst max_abs={worst_abs:.3e} worst max_rel={worst_rel:.3e}", cases.len());
    }

    #[test]
    fn tq_runtime_paths_are_gpu_bit_identical() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[tq_runtime_path] no Metal device; skipping policy parity gate");
            return;
        };
        let (rows, cols) = (8usize, 512usize);
        let cfg = TrellisConfig::for_bpw_l(3.0, 10);
        let w = synth_w(rows * cols, 0xF10F5);
        let enc = strand_quant::encode::encode_tensor_with(&w, &cfg, &strand_quant::encode::EncodeOpts { tail_biting: true, affine_min: true, ..Default::default() });
        let prepared = prepare(&enc, &cfg, rows, cols);
        let x = synth_x(cols, 0xC0DE);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));

        let mut reference: Option<Vec<u32>> = None;
        for path in [crate::TqRuntimePath::Stored, crate::TqRuntimePath::CompactMetadata, crate::TqRuntimePath::HashedQuantile, crate::TqRuntimePath::ComputedAcklam] {
            let gpu = prepared.upload_to_gpu_with_path(&ctx, path).unwrap_or_else(|e| panic!("{path:?} upload failed: {e}"));
            let out = ctx.new_buffer(rows * std::mem::size_of::<f32>());
            let mut tcb = TokenCommandBuffer::new(&ctx);
            super::strand_bitslice_gemv_tcb(&mut tcb, &gpu, &x_buf, 0, &out, 0).unwrap_or_else(|e| panic!("{path:?} dispatch failed: {e}"));
            tcb.commit_and_wait().unwrap();
            let bits: Vec<u32> = read_back(&out, rows).into_iter().map(f32::to_bits).collect();
            if let Some(want) = &reference {
                assert_eq!(&bits, want, "{path:?} changed fused GEMV float bits");
            } else {
                reference = Some(bits);
            }
        }
    }

    /// Guard that the residual term is actually LIVE: the two-part serve must
    /// differ from the base-only serve (otherwise an accidental no-op residual
    /// would pass the sum test trivially when the residual is tiny). We assert the
    /// residual changes the output by more than fp noise.
    #[test]
    fn residual_pass_is_not_a_noop() {
        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[residual_serve] no Metal device; skipping residual-live gate");
            return;
        };
        let (rows, cols) = (8usize, 512usize);
        let total = rows * cols;
        let seed = 7u64;
        let w = synth_w(total, seed);
        let cfg_b = TrellisConfig::for_bpw(2.0); // coarse base ⇒ meaningful residual
        let enc_base = encode_tensor(&w, &cfg_b);
        let wh1 = decode_f32(&enc_base, &cfg_b);
        let resid: Vec<f32> = w.iter().zip(&wh1).map(|(a, b)| a - b).collect();
        let cfg_r = TrellisConfig::for_bpw(2.0);
        let enc_res = encode_tensor(&resid, &cfg_r);

        let x = synth_x(cols, seed ^ 0x1234);
        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));

        let gpu_base = prepare(&enc_base, &cfg_b, rows, cols).upload_to_gpu(&ctx).unwrap();
        let gpu_res = prepare(&enc_res, &cfg_r, rows, cols).upload_to_gpu(&ctx).unwrap();

        // base-only.
        let out_base = ctx.new_buffer(rows * 4);
        let mut tcb = TokenCommandBuffer::new(&ctx);
        super::strand_bitslice_gemv_tcb(&mut tcb, &gpu_base, &x_buf, 0, &out_base, 0).unwrap();
        tcb.commit_and_wait().unwrap();
        let y_base = read_back(&out_base, rows);

        // base + residual.
        let out_sum = ctx.new_buffer(rows * 4);
        let mut tcb2 = TokenCommandBuffer::new(&ctx);
        super::strand_bitslice_gemv_tcb(&mut tcb2, &gpu_base, &x_buf, 0, &out_sum, 0).unwrap();
        super::strand_bitslice_gemv_tcb_accum(&mut tcb2, &gpu_res, &x_buf, 0, &out_sum, 0).unwrap();
        tcb2.commit_and_wait().unwrap();
        let y_sum = read_back(&out_sum, rows);

        let max_delta = (0..rows).map(|o| (y_sum[o] - y_base[o]).abs()).fold(0.0f32, f32::max);
        println!("[residual_serve] residual contribution max_delta={max_delta:.3e}");
        assert!(max_delta > 1e-5, "residual pass changed output by only {max_delta:.3e} — residual term is a no-op?");
    }

    /// Full file→loader→serve loop: write base + residual `.tq` STR2 archives (the
    /// SAME format residual_tq.py emits), read them back through the production
    /// loader `crate::tq::read_strand`, build `TqPreparedGpu::from_strand_tensor`
    /// for each, run the GPU two-part GEMV, and assert it equals the CPU decoded-sum
    /// GEMV over `decode_q12_raw(base) + decode_q12_raw(residual)`. This exercises
    /// the loader path (task step 2), not just raw EncodedTensors.
    #[test]
    fn residual_file_round_trip_two_part_serves_decoded_sum() {
        use strand_quant::format::{write_strand_v2, PackedTensor, PackedTensorV2};

        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[residual_serve] no Metal device; skipping file round-trip gate");
            return;
        };

        let name = "model.layers.0.mlp.down_proj.weight";
        let (rows, cols) = (8usize, 512usize); // cols % 256 == 0
        let total = rows * cols;
        let w = synth_w(total, 0xBEEF);

        // base pass.
        let cfg_b = TrellisConfig::for_bpw(3.0);
        let enc_base = encode_tensor(&w, &cfg_b);
        let wh1 = decode_f32(&enc_base, &cfg_b);
        // residual pass on W − decode(base).
        let resid: Vec<f32> = w.iter().zip(&wh1).map(|(a, b)| a - b).collect();
        let cfg_r = TrellisConfig::for_bpw(2.0);
        let enc_res = encode_tensor(&resid, &cfg_r);

        // Write two real STR2 archives (no RHT, no OUTL — the served contract).
        let shape = [rows as u64, cols as u64];
        let pack = |enc: &EncodedTensor, cfg: &TrellisConfig| {
            write_strand_v2(
                &[PackedTensorV2 {
                    base: PackedTensor { name, shape: &shape, rht_seed: 0, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc },
                    block_len: cfg.block_len as u32,
                }],
                [0u8; 32],
                true,
            )
            .expect("write_strand_v2")
        };
        let base_bytes = pack(&enc_base, &cfg_b);
        let res_bytes = pack(&enc_res, &cfg_r);

        // Read both back through the PRODUCTION loader.
        let base_store = crate::tq::read_strand(&base_bytes).expect("read base .tq");
        let res_store = crate::tq::read_strand(&res_bytes).expect("read residual .tq");
        assert_eq!(base_store.len(), 1);
        assert_eq!(res_store.len(), 1);
        let st_base = &base_store[0];
        let st_res = &res_store[0];
        assert_eq!((st_base.out_features, st_base.in_features), (rows, cols));
        assert_eq!(st_base.rht_mode, crate::tq::RhtMode::None);

        // CPU decoded-sum reference from the LOADED tensors (decode_q12_raw → f32).
        let inv = crate::tq::q12_to_f32();
        let qb = st_base.decode_q12_raw();
        let qr = st_res.decode_q12_raw();
        let x = synth_x(cols, 0x1357);
        let mut y_ref = vec![0.0f32; rows];
        for o in 0..rows {
            let mut acc = 0.0f32;
            for i in 0..cols {
                let wsum = (qb[o * cols + i] as f32 + qr[o * cols + i] as f32) * inv;
                acc += wsum * x[i];
            }
            y_ref[o] = acc;
        }

        // GPU two-part serve from the LOADED tensors via from_strand_tensor.
        let gpu_base = TqPreparedGpu::from_strand_tensor(st_base).expect("prep base").upload_to_gpu(&ctx).expect("upload base");
        let gpu_res = TqPreparedGpu::from_strand_tensor(st_res).expect("prep res").upload_to_gpu(&ctx).expect("upload res");

        let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
        let out_buf = ctx.new_buffer(rows * std::mem::size_of::<f32>());
        let mut tcb = TokenCommandBuffer::new(&ctx);
        super::strand_bitslice_gemv_tcb(&mut tcb, &gpu_base, &x_buf, 0, &out_buf, 0).unwrap();
        super::strand_bitslice_gemv_tcb_accum(&mut tcb, &gpu_res, &x_buf, 0, &out_buf, 0).unwrap();
        tcb.commit_and_wait().unwrap();
        let y_gpu = read_back(&out_buf, rows);

        let mut max_rel = 0.0f32;
        for o in 0..rows {
            let abs = (y_gpu[o] - y_ref[o]).abs();
            max_rel = max_rel.max(abs / (1.0 + y_ref[o].abs()));
        }
        println!("[residual_serve] file round-trip {rows}x{cols} 3+2: max_rel={max_rel:.3e}");
        assert!(max_rel <= 2e-3, "file-loaded two-part serve max_rel {max_rel:.3e} > 2e-3 vs decoded-sum");
    }

    /// GAP 1: a tensor baked WITH `--rht-cols` + `--outlier-channel` (the ACTUAL
    /// quality recipe `residual_bake.py` / the audit ladder use) serves on the GPU
    /// bitslice path bit-faithfully vs the CPU decode.
    ///
    /// Builds a real STR2 `Cols` archive with an OUTL section in-process (the same
    /// wire `write_strand_v2_rht` + `append_outl` produce), reads it back through
    /// the production loader (`crate::tq::read_strand`) → `StrandTensor` with
    /// `RhtMode::Cols` + outliers, takes `StrandTensor::matvec(x)` as the CPU
    /// reference, then serves the same tensor on GPU via `from_strand_tensor` →
    /// `upload_to_gpu` → `strand_bitslice_gemv_tcb` (which now runs the GPU RHT-cols
    /// activation transform + the OUTL sparse correction). `in_features % 256 == 0`
    /// (the deploy/GPU-FWHT invariant). This is the gate that the actual quality
    /// recipe can be served on GPU — the unlock GAP 1 targets.
    #[test]
    fn rht_cols_outlier_serves_bit_faithfully_vs_cpu() {
        use crate::tq::{read_strand, RhtMode};
        use std::io::Write as _;
        use strand_quant::format::{write_strand_v2_rht, PackedTensor, PackedTensorV2};
        use strand_quant::outlier_wire::{append_outl, OutlierWire};
        use strand_quant::rht::{rht_forward_cols, RhtConfig};

        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[residual_serve] no Metal device; skipping rht-cols+OUTL gate");
            return;
        };

        // 7B-shaped in_features (3584 attn, both % 256 == 0); a few rows to keep the
        // synthetic encode fast. (896 — the 0.5B width — is NOT %256, so the GPU
        // RHT-cols path intentionally refuses it; that refusal is asserted below.)
        for &(out_f, in_f) in &[(6usize, 256usize), (4usize, 3584usize)] {
            let name = "model.layers.0.mlp.down_proj.weight";
            let n = out_f * in_f;
            let seed = strand_quant::gate_utils::rht_seed_for(name);
            let gt = synth_w(n, 0xC0FFEE);

            // Outlier selection: top-|w| 1%, quantised exactly like the baker.
            let k = ((1.0f64 / 100.0) * n as f64).round().max(1.0) as usize;
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by(|&a, &b| gt[b].abs().partial_cmp(&gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
            let idx: Vec<usize> = order[..k].to_vec();
            let ob = 8u32;
            let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
            let levels = ((1i64 << (ob - 1)) - 1) as f32;
            let codes: Vec<i32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() as i32).collect();

            // Bulk = ground truth with outlier positions zeroed, column-rotated.
            let mut bulk = gt.clone();
            for &i in &idx {
                bulk[i] = 0.0;
            }
            let rcfg = RhtConfig::from_seed(seed);
            let work = rht_forward_cols(&bulk, &rcfg, in_f);
            let cfg = TrellisConfig::for_bpw(3.0);
            let mut enc = encode_tensor(&work, &cfg);
            enc.has_rht_seed = true;

            let shape = [out_f as u64, in_f as u64];
            let packed = PackedTensorV2 {
                base: PackedTensor { name, shape: &shape, rht_seed: seed, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
                block_len: cfg.block_len as u32,
            };
            let buf = write_strand_v2_rht(&[packed], [0u8; 32], true, false, &[true]).expect("write_strand_v2_rht");
            let mut path = std::env::temp_dir();
            path.push(format!("tq_gpu_rhtcols_outl_{}_{}_{in_f}.tq", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
            {
                let mut f = std::fs::File::create(&path).expect("create temp .tq");
                f.write_all(&buf).expect("write temp .tq");
                f.sync_all().ok();
            }
            let wire = OutlierWire::from_selection(n, idx.clone(), codes, omax, ob);
            append_outl(&path, &[Some(wire)]).expect("append outl");
            let bytes = std::fs::read(&path).expect("re-read .tq");
            let _ = std::fs::remove_file(&path);

            let tensors = read_strand(&bytes).expect("read_strand cols+OUTL");
            let st = &tensors[0];
            assert_eq!(st.rht_mode, RhtMode::Cols, "must be a Cols archive");
            assert_eq!(st.outliers.len(), k, "OUTL must round-trip");

            // CPU reference: the production StrandTensor serve (un-rotated patched).
            let x = synth_x(in_f, 0x1357);
            let y_ref = st.matvec(&x);

            // GPU serve: from_strand_tensor (precomputes outlier resids + RHT seed)
            // → upload → strand_bitslice_gemv_tcb (RHT-cols transform + GEMV + OUTL).
            let prep = TqPreparedGpu::from_strand_tensor(st).expect("prep cols+OUTL");
            assert_eq!(prep.rht_mode, 2, "RhtMode::Cols → 2");
            assert_eq!(prep.outliers.len(), k, "outlier resids precomputed");
            let gpu: TqGpuReady = prep.upload_to_gpu(&ctx).expect("upload cols+OUTL");
            assert!(gpu.rht_x_buf.is_some(), "Cols needs the rht_x scratch");
            assert_eq!(gpu.n_outl, k as u32);

            let x_buf = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
            let out_buf = ctx.new_buffer(out_f * std::mem::size_of::<f32>());
            let mut tcb = TokenCommandBuffer::new(&ctx);
            super::strand_bitslice_gemv_tcb(&mut tcb, &gpu, &x_buf, 0, &out_buf, 0).expect("gpu cols+OUTL gemv");
            tcb.commit_and_wait().expect("commit");
            let y_gpu = read_back(&out_buf, out_f);

            let mut max_rel = 0.0f32;
            for o in 0..out_f {
                let abs = (y_gpu[o] - y_ref[o]).abs();
                max_rel = max_rel.max(abs / (1.0 + y_ref[o].abs()));
            }
            println!("[residual_serve] rht-cols+OUTL {out_f}x{in_f}: max_rel={max_rel:.3e} (k={k} outliers)");
            // f32 FWHT + GEMV reduction grouping vs the CPU's row-major dot: same
            // 2e-3 budget as the other serve gates (cols up to ~3584).
            assert!(max_rel <= 2e-3, "{out_f}x{in_f} rht-cols+OUTL GPU serve max_rel {max_rel:.3e} > 2e-3 vs CPU matvec");
        }

        // The GPU RHT-cols path is 256-wide; an unaligned in_features (the 0.5B's
        // 896) must REFUSE the GPU upload rather than serve a divergent transform.
        // (The STR2 writer ALSO enforces in_features % block_len == 0, so such an
        // archive can't even be written today — the upload guard is the defensive
        // backstop. We exercise it by hand-building a Cols TqPreparedGpu at cols=896
        // from a `--no-rht` 896-wide encode and flipping rht_mode to Cols.)
        {
            let (out_f, in_f) = (4usize, 896usize); // 896 % 256 == 128 ≠ 0
            let cfg = TrellisConfig::for_bpw(3.0);
            let enc = encode_tensor(&synth_w(out_f * in_f, 7), &cfg);
            let mut prep = prepare(&enc, &cfg, out_f, in_f);
            prep.rht_mode = 2; // pretend Cols on an unaligned width
            prep.rht_seed = 0xABCD;
            let res = prep.upload_to_gpu(&ctx);
            assert!(res.is_err(), "Cols with in_features%256!=0 (896) must refuse the GPU RHT path");
        }
    }

    /// GAP 1/2 offset guard: the RHT-cols transform + OUTL correction must honour a
    /// NON-ZERO `out_off_bytes` (and `x_off_bytes`) — the layout the rwkv7/Qwen
    /// multiseq batched path uses (`out_off_b = bi*rows*f`). Serves one Cols+OUTL
    /// tensor into the SECOND row-slot of a 2-slot output buffer (and from the
    /// second slot of a 2-slot x buffer) and asserts it equals the offset-0 serve.
    /// This catches a double-applied offset (binding at the byte offset AND adding
    /// the element offset in-kernel).
    #[test]
    fn rht_cols_outlier_honours_output_offset() {
        use crate::tq::{read_strand, RhtMode};
        use std::io::Write as _;
        use strand_quant::format::{write_strand_v2_rht, PackedTensor, PackedTensorV2};
        use strand_quant::outlier_wire::{append_outl, OutlierWire};
        use strand_quant::rht::{rht_forward_cols, RhtConfig};

        let Ok(ctx) = MetalContext::new() else {
            eprintln!("[residual_serve] no Metal device; skipping offset guard");
            return;
        };
        let name = "model.layers.0.mlp.down_proj.weight";
        let (out_f, in_f) = (6usize, 512usize);
        let n = out_f * in_f;
        let seed = strand_quant::gate_utils::rht_seed_for(name);
        let gt = synth_w(n, 0xD00D);
        let k = ((1.0f64 / 100.0) * n as f64).round().max(1.0) as usize;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| gt[b].abs().partial_cmp(&gt[a].abs()).unwrap_or(std::cmp::Ordering::Equal));
        let idx: Vec<usize> = order[..k].to_vec();
        let ob = 8u32;
        let omax = idx.iter().fold(0f32, |m, &i| m.max(gt[i].abs())).max(1e-12);
        let levels = ((1i64 << (ob - 1)) - 1) as f32;
        let codes: Vec<i32> = idx.iter().map(|&i| (gt[i] / omax * levels).round() as i32).collect();
        let mut bulk = gt.clone();
        for &i in &idx {
            bulk[i] = 0.0;
        }
        let rcfg = RhtConfig::from_seed(seed);
        let work = rht_forward_cols(&bulk, &rcfg, in_f);
        let cfg = TrellisConfig::for_bpw(3.0);
        let mut enc = encode_tensor(&work, &cfg);
        enc.has_rht_seed = true;
        let shape = [out_f as u64, in_f as u64];
        let packed = PackedTensorV2 {
            base: PackedTensor { name, shape: &shape, rht_seed: seed, l_bits: cfg.l_bits as u8, k_bits: cfg.k_bits as u8, vec_dim: cfg.vec_dim() as u8, enc: &enc },
            block_len: cfg.block_len as u32,
        };
        let buf = write_strand_v2_rht(&[packed], [0u8; 32], true, false, &[true]).expect("write");
        let mut path = std::env::temp_dir();
        path.push(format!("tq_offset_guard_{}.tq", std::process::id()));
        {
            let mut f = std::fs::File::create(&path).expect("create");
            f.write_all(&buf).expect("write");
            f.sync_all().ok();
        }
        let wire = OutlierWire::from_selection(n, idx.clone(), codes, omax, ob);
        append_outl(&path, &[Some(wire)]).expect("append outl");
        let bytes = std::fs::read(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        let tensors = read_strand(&bytes).expect("read_strand");
        let st = &tensors[0];
        assert_eq!(st.rht_mode, RhtMode::Cols);
        assert!(!st.outliers.is_empty());

        let gpu = TqPreparedGpu::from_strand_tensor(st).unwrap().upload_to_gpu(&ctx).unwrap();
        let x = synth_x(in_f, 0x2468);

        // Offset-0 baseline.
        let x0 = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x));
        let out0 = ctx.new_buffer(out_f * 4);
        let mut t0 = TokenCommandBuffer::new(&ctx);
        super::strand_bitslice_gemv_tcb(&mut t0, &gpu, &x0, 0, &out0, 0).unwrap();
        t0.commit_and_wait().unwrap();
        let y0 = read_back(&out0, out_f);

        // Slot-1 serve: x in the second of two cols-slots, out into the second of
        // two rows-slots (the multiseq stride layout). Must equal y0 exactly.
        let f = std::mem::size_of::<f32>();
        let mut x2 = vec![0.0f32; 2 * in_f];
        x2[in_f..].copy_from_slice(&x);
        let x2b = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&x2));
        let out2 = ctx.new_buffer(2 * out_f * 4);
        let mut t1 = TokenCommandBuffer::new(&ctx);
        super::strand_bitslice_gemv_tcb(&mut t1, &gpu, &x2b, in_f * f, &out2, out_f * f).unwrap();
        t1.commit_and_wait().unwrap();
        let y2_all = read_back(&out2, 2 * out_f);
        let y2 = &y2_all[out_f..];

        for o in 0..out_f {
            assert!((y2[o] - y0[o]).abs() <= 1e-5 * (1.0 + y0[o].abs()), "row {o}: offset serve {} != offset-0 {} (double-applied offset?)", y2[o], y0[o]);
        }
        // Slot-0 of out2 must be untouched (the offset serve wrote only slot 1).
        for o in 0..out_f {
            assert_eq!(y2_all[o], 0.0, "slot 0 must be untouched by an offset serve");
        }
        println!("[residual_serve] offset guard {out_f}x{in_f}: slot-1 serve == slot-0 baseline");
    }
}
