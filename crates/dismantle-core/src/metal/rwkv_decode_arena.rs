//! RWKV-7 GPU decode arena + WKV-7 / glue kernel dispatchers.
//!
//! RWKV-7 is a state-space model: it carries a FIXED per-layer recurrent state
//! (one `head_size×head_size` matrix per head) instead of a growing KV cache, so
//! decode is O(1) in context. This arena holds that state in persistent GPU
//! buffers — `wkv_state` (the constant ~6 MiB), plus the two token-shift state
//! planes — alongside the small per-token scratch buffers for one decode step.
//! Nothing here grows with sequence length; the buffers are allocated once and
//! advanced in place each step (the realization of the "flat decode" property on
//! the GPU).
//!
//! All scratch is f32 so the GPU path is bit-for-bit (within f32 tolerance)
//! against the CPU reference `rwkv7.rs::forward_token`, which dequantizes every
//! weight to f32 and uses `gemv_f32`. The projection GEMVs reuse the existing
//! `gemv_f32_attn` kernel (identical f32 MAC); the only novel kernel is the
//! WKV-7 recurrence (`rwkv7_wkv_decode`) plus a handful of elementwise glue
//! kernels for the RWKV-specific token-shift/lerp/activations.

#[cfg(target_os = "macos")]
pub use imp::{
    rwkv7_add_into_tcb, rwkv7_channel_mix_shift_tcb, rwkv7_copy_tcb, rwkv7_decay_act_tcb,
    rwkv7_gemv_f32_off_tcb, rwkv7_kk_kmix_tcb, rwkv7_layernorm_tcb, rwkv7_relu_sq_inplace_tcb,
    rwkv7_sigmoid_bias_tcb, rwkv7_sigmoid_inplace_tcb, rwkv7_tanh_inplace_tcb,
    rwkv7_token_shift_lerp_tcb, rwkv7_value_residual_mix_tcb, rwkv7_wkv_decode_tcb,
    RwkvDecodeArena,
};

#[cfg(target_os = "macos")]
mod imp {
    use crate::metal::argbuf::{ArgLayout, KernelArgBuffer};
    use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
    use crate::Result;

    /// Threadgroup width for the LayerNorm reduction kernel (matches the rest of
    /// the codebase's 256-thread single-TG norm convention).
    const LN_TG: u32 = 256;

    /// Local scalar-binding helper (the codebase's `SetScalar` is private to
    /// `kernels::metal_dispatch`). Same `set_bytes` idiom, `#[inline(always)]` so
    /// codegen is identical to the inline form.
    trait SetScalar {
        fn set_u32(&self, index: u64, value: u32);
    }
    impl SetScalar for metal::ComputeCommandEncoderRef {
        #[inline(always)]
        fn set_u32(&self, index: u64, value: u32) {
            self.set_bytes(
                index,
                std::mem::size_of::<u32>() as u64,
                &value as *const u32 as *const _,
            );
        }
    }

    /// Per-layer recurrent state + per-token scratch for the RWKV-7 GPU decode.
    ///
    /// The three `*_state` buffers are PERSISTENT across tokens (the KV-cache
    /// replacement); everything else is per-token scratch, overwritten each step.
    /// Sizes are constant in context — the headline "no growing KV" guarantee.
    pub struct RwkvDecodeArena {
        // ── persistent recurrent state (advanced in place, never grows) ──
        /// `n_layer * head_count * hs * hs` floats — the per-head S matrices.
        pub wkv_state: PinnedBuffer,
        /// `n_layer * n_embd` — previous token's att-branch post-LN hidden.
        pub att_shift: PinnedBuffer,
        /// `n_layer * n_embd` — previous token's ffn-branch post-LN hidden.
        pub ffn_shift: PinnedBuffer,

        // ── per-token scratch (n_embd unless noted) ──
        pub x: PinnedBuffer,
        pub att_in: PinnedBuffer,
        pub ffn_inp: PinnedBuffer,
        pub ffn_in: PinnedBuffer,
        pub x_norm: PinnedBuffer,
        /// `[6 * n_embd]` slot-major lerped activations (r,w,k,v,a,g).
        pub xs: PinnedBuffer,
        pub xk_ffn: PinnedBuffer,
        pub r: PinnedBuffer,
        pub w: PinnedBuffer,
        pub w_raw: PinnedBuffer,
        pub k: PinnedBuffer,
        pub v: PinnedBuffer,
        pub a: PinnedBuffer,
        pub a_op: PinnedBuffer,
        pub b_op: PinnedBuffer,
        pub gate: PinnedBuffer,
        pub out_wkv: PinnedBuffer,
        pub cur: PinnedBuffer,
        pub cmix: PinnedBuffer,
        pub v_first: PinnedBuffer,
        pub v_mix: PinnedBuffer,
        /// LoRA low-rank scratch.
        pub w_lo: PinnedBuffer,
        pub a_lo: PinnedBuffer,
        pub v_lo: PinnedBuffer,
        pub g_lo: PinnedBuffer,
        /// Channel-mix intermediate `[n_ff]`.
        pub ffn_k: PinnedBuffer,
        /// LM-head output `[vocab]` and greedy argmax token.
        pub logits: PinnedBuffer,
        pub token: PinnedBuffer,

        // dims
        pub n_layer: usize,
        pub n_embd: usize,
        pub n_ff: usize,
        pub head_size: usize,
        pub head_count: usize,
        pub vocab_size: usize,
        pub decay_lora: usize,
        pub iclr_lora: usize,
        pub value_res_lora: usize,
        pub gate_lora: usize,
    }

    impl RwkvDecodeArena {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            ctx: &MetalContext,
            n_layer: usize,
            n_embd: usize,
            n_ff: usize,
            head_size: usize,
            head_count: usize,
            vocab_size: usize,
            decay_lora: usize,
            iclr_lora: usize,
            value_res_lora: usize,
            gate_lora: usize,
        ) -> Self {
            let f = std::mem::size_of::<f32>();
            let nb = |elems: usize| ctx.new_buffer(elems.max(1) * f);
            let s_per_layer = head_count * head_size * head_size;
            Self {
                wkv_state: nb(n_layer * s_per_layer),
                att_shift: nb(n_layer * n_embd),
                ffn_shift: nb(n_layer * n_embd),

                x: nb(n_embd),
                att_in: nb(n_embd),
                ffn_inp: nb(n_embd),
                ffn_in: nb(n_embd),
                x_norm: nb(n_embd),
                xs: nb(6 * n_embd),
                xk_ffn: nb(n_embd),
                r: nb(n_embd),
                w: nb(n_embd),
                w_raw: nb(n_embd),
                k: nb(n_embd),
                v: nb(n_embd),
                a: nb(n_embd),
                a_op: nb(n_embd),
                b_op: nb(n_embd),
                gate: nb(n_embd),
                out_wkv: nb(n_embd),
                cur: nb(n_embd),
                cmix: nb(n_embd),
                v_first: nb(n_embd),
                v_mix: nb(n_embd),
                w_lo: nb(decay_lora),
                a_lo: nb(iclr_lora),
                v_lo: nb(value_res_lora),
                g_lo: nb(gate_lora),
                ffn_k: nb(n_ff),
                logits: nb(vocab_size),
                token: ctx.new_buffer(std::mem::size_of::<u32>()),

                n_layer,
                n_embd,
                n_ff,
                head_size,
                head_count,
                vocab_size,
                decay_lora,
                iclr_lora,
                value_res_lora,
                gate_lora,
            }
        }

        /// Byte offset of `layer`'s window in the WKV state plane.
        pub fn wkv_layer_byte_offset(&self, layer: usize) -> usize {
            layer * self.head_count * self.head_size * self.head_size * std::mem::size_of::<f32>()
        }
        /// Byte offset of `layer`'s window in a token-shift state plane.
        pub fn shift_layer_byte_offset(&self, layer: usize) -> usize {
            layer * self.n_embd * std::mem::size_of::<f32>()
        }

        /// Zero the persistent recurrent state (start of a fresh sequence).
        /// Shared-storage buffers on Apple Silicon are CPU-visible, so this is a
        /// plain memset on the unified-memory backing.
        pub fn reset_state(&self) {
            for (buf, elems) in [
                (
                    &self.wkv_state,
                    self.n_layer * self.head_count * self.head_size * self.head_size,
                ),
                (&self.att_shift, self.n_layer * self.n_embd),
                (&self.ffn_shift, self.n_layer * self.n_embd),
            ] {
                let ptr = buf.contents() as *mut f32;
                unsafe { std::ptr::write_bytes(ptr, 0, elems) };
            }
        }
    }

    // ── glue-kernel dispatchers (TCB; no counted commits) ────────────────────

    fn dispatch_n(
        tcb: &mut TokenCommandBuffer<'_>,
        kernel: &str,
        n: u32,
        encode: impl FnOnce(&metal::ComputeCommandEncoderRef),
    ) -> Result<()> {
        let tg = LN_TG.min(n.max(1));
        let n_tg = n.div_ceil(tg).max(1);
        tcb.dispatch_threads(kernel, (n_tg * tg, 1, 1), (tg, 1, 1), encode)
    }

    /// f32 GEMV (`gemv_f32_attn`) with an OFFSET into the x buffer, so a
    /// slot-major activation block can feed each projection without copying the
    /// slot out. `w` is `[rows,cols]` row-major f32; `x` is read starting at
    /// `x_off_bytes`; `out` is written at offset 0. Reuses the proven attn GEMV
    /// kernel (identical f32 MAC to the CPU `gemv_f32`).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_gemv_f32_off_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x: &PinnedBuffer,
        x_off_bytes: usize,
        out: &PinnedBuffer,
    ) -> Result<()> {
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        let shmem = (LN_TG as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, rows_u32);
        ab.set_u32(1, cols_u32);
        tcb.dispatch_threads(
            "gemv_f32_attn",
            (rows_u32 * LN_TG, 1, 1),
            (LN_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(w), 0);
                enc.set_buffer(1, Some(x), x_off_bytes as u64);
                enc.set_buffer(2, Some(out), 0);
                enc.set_buffer(3, Some(ab.handle()), 0);
                enc.set_threadgroup_memory_length(0, shmem);
            },
        )
    }

    /// token-shift + per-slot lerp (time-mix). Writes `xs` slot-major. `x_prev`
    /// is the per-layer token-shift state plane, read at `x_prev_off_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_token_shift_lerp_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        att_in: &PinnedBuffer,
        x_prev: &PinnedBuffer,
        x_prev_off_bytes: usize,
        lerp: &PinnedBuffer,
        xs: &PinnedBuffer,
        n: usize,
        n_slots: usize,
        fresh: bool,
    ) -> Result<()> {
        let mut ab =
            KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, n_slots as u32);
        ab.set_u32(2, fresh as u32);
        dispatch_n(tcb, "rwkv7_token_shift_lerp", n as u32, |enc| {
            enc.set_buffer(0, Some(att_in), 0);
            enc.set_buffer(1, Some(x_prev), x_prev_off_bytes as u64);
            enc.set_buffer(2, Some(lerp), 0);
            enc.set_buffer(3, Some(xs), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// channel-mix token-shift + single lerp. `x_prev` is the per-layer ffn
    /// token-shift state plane, read at `x_prev_off_bytes`.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_channel_mix_shift_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        ffn_in: &PinnedBuffer,
        x_prev: &PinnedBuffer,
        x_prev_off_bytes: usize,
        lerp_k: &PinnedBuffer,
        xk: &PinnedBuffer,
        n: usize,
        fresh: bool,
    ) -> Result<()> {
        let mut ab =
            KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, 1);
        ab.set_u32(2, fresh as u32);
        dispatch_n(tcb, "rwkv7_channel_mix_shift", n as u32, |enc| {
            enc.set_buffer(0, Some(ffn_in), 0);
            enc.set_buffer(1, Some(x_prev), x_prev_off_bytes as u64);
            enc.set_buffer(2, Some(lerp_k), 0);
            enc.set_buffer(3, Some(xk), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// tanh in place over `n` elements.
    pub fn rwkv7_tanh_inplace_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_tanh_inplace", n as u32, |enc| {
            enc.set_buffer(0, Some(x), 0);
            enc.set_u32(1, n as u32);
        })
    }

    /// decay activation: w = exp(-0.606531 * sigmoid(w_raw + w0)).
    pub fn rwkv7_decay_act_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_raw: &PinnedBuffer,
        w0: &PinnedBuffer,
        w: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_decay_act", n as u32, |enc| {
            enc.set_buffer(0, Some(w_raw), 0);
            enc.set_buffer(1, Some(w0), 0);
            enc.set_buffer(2, Some(w), 0);
            enc.set_u32(3, n as u32);
        })
    }

    /// sigmoid-with-bias in place: x = sigmoid(x + bias).
    pub fn rwkv7_sigmoid_bias_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &PinnedBuffer,
        bias: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_sigmoid_bias", n as u32, |enc| {
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(bias), 0);
            enc.set_u32(2, n as u32);
        })
    }

    /// plain sigmoid in place: x = sigmoid(x).
    pub fn rwkv7_sigmoid_inplace_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_sigmoid_inplace", n as u32, |enc| {
            enc.set_buffer(0, Some(x), 0);
            enc.set_u32(1, n as u32);
        })
    }

    /// value-residual mix: v += (v_first - v) * sigmoid(v_mix + v0).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_value_residual_mix_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        v: &PinnedBuffer,
        v_first: &PinnedBuffer,
        v_mix: &PinnedBuffer,
        v0: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_value_residual_mix", n as u32, |enc| {
            enc.set_buffer(0, Some(v), 0);
            enc.set_buffer(1, Some(v_first), 0);
            enc.set_buffer(2, Some(v_mix), 0);
            enc.set_buffer(3, Some(v0), 0);
            enc.set_u32(4, n as u32);
        })
    }

    /// kk = l2norm_per_head(k*k_k); k += (a-1)*(k*k_a); a_op=-kk; b_op=kk*a.
    /// One threadgroup per head (hs threads).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_kk_kmix_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        k: &PinnedBuffer,
        k_k: &PinnedBuffer,
        k_a: &PinnedBuffer,
        a: &PinnedBuffer,
        a_op: &PinnedBuffer,
        b_op: &PinnedBuffer,
        head_size: usize,
        head_count: usize,
    ) -> Result<()> {
        let hs = head_size as u32;
        let shmem = (hs as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32])?;
        ab.set_u32(0, hs);
        tcb.dispatch_threads(
            "rwkv7_kk_kmix",
            (head_count as u32 * hs, 1, 1),
            (hs, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(k), 0);
                enc.set_buffer(1, Some(k_k), 0);
                enc.set_buffer(2, Some(k_a), 0);
                enc.set_buffer(3, Some(a), 0);
                enc.set_buffer(4, Some(a_op), 0);
                enc.set_buffer(5, Some(b_op), 0);
                enc.set_buffer(6, Some(ab.handle()), 0);
                enc.set_threadgroup_memory_length(0, shmem);
            },
        )
    }

    /// The WKV-7 single-step recurrence + per-head group-norm + bonus + gate.
    /// `state` is the persistent S plane; `state_off_bytes` selects the layer.
    /// All vector inputs are full `n_embd` buffers indexed per-head inside the
    /// kernel. `gate` may alias any buffer when `has_gate` is false (unused).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_wkv_decode_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        state: &PinnedBuffer,
        state_off_bytes: usize,
        r: &PinnedBuffer,
        w: &PinnedBuffer,
        k: &PinnedBuffer,
        v: &PinnedBuffer,
        a_op: &PinnedBuffer,
        b_op: &PinnedBuffer,
        r_k: &PinnedBuffer,
        ln_w: &PinnedBuffer,
        ln_b: &PinnedBuffer,
        gate: &PinnedBuffer,
        out: &PinnedBuffer,
        head_size: usize,
        head_count: usize,
        gn_eps: f32,
        has_gate: bool,
    ) -> Result<()> {
        let hs = head_size as u32;
        let shmem = (hs as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab =
            KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32, ArgLayout::U32])?;
        ab.set_u32(0, hs);
        ab.set_f32(1, gn_eps);
        ab.set_u32(2, has_gate as u32);
        tcb.dispatch_threads(
            "rwkv7_wkv_decode",
            (head_count as u32 * hs, 1, 1),
            (hs, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(state), state_off_bytes as u64);
                enc.set_buffer(1, Some(r), 0);
                enc.set_buffer(2, Some(w), 0);
                enc.set_buffer(3, Some(k), 0);
                enc.set_buffer(4, Some(v), 0);
                enc.set_buffer(5, Some(a_op), 0);
                enc.set_buffer(6, Some(b_op), 0);
                enc.set_buffer(7, Some(r_k), 0);
                enc.set_buffer(8, Some(ln_w), 0);
                enc.set_buffer(9, Some(ln_b), 0);
                enc.set_buffer(10, Some(gate), 0);
                enc.set_buffer(11, Some(out), 0);
                enc.set_buffer(12, Some(ab.handle()), 0);
                enc.set_threadgroup_memory_length(0, shmem);
            },
        )
    }

    /// channel-mix activation in place: k = relu(k)^2 over `n_ff` elements.
    pub fn rwkv7_relu_sq_inplace_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        k: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_relu_sq_inplace", n as u32, |enc| {
            enc.set_buffer(0, Some(k), 0);
            enc.set_u32(1, n as u32);
        })
    }

    /// out = a + b (fresh destination).
    pub fn rwkv7_add_into_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        a: &PinnedBuffer,
        b: &PinnedBuffer,
        out: &PinnedBuffer,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_add_into", n as u32, |enc| {
            enc.set_buffer(0, Some(a), 0);
            enc.set_buffer(1, Some(b), 0);
            enc.set_buffer(2, Some(out), 0);
            enc.set_u32(3, n as u32);
        })
    }

    /// LayerNorm (weight+bias, population variance). Single-TG, grid-strided.
    /// `x_off_bytes` / `out_off_bytes` select a layer window when the buffer is
    /// a multi-layer plane (e.g. the token-shift state is read with offset 0 for
    /// scratch but the norm output may target a per-layer plane).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_layernorm_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &PinnedBuffer,
        x_off_bytes: usize,
        weight: &PinnedBuffer,
        bias: &PinnedBuffer,
        out: &PinnedBuffer,
        out_off_bytes: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        let shmem = (LN_TG as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden as u32);
        ab.set_f32(1, eps);
        tcb.dispatch_threads("rwkv7_layernorm", (LN_TG, 1, 1), (LN_TG, 1, 1), |enc| {
            enc.set_buffer(0, Some(x), x_off_bytes as u64);
            enc.set_buffer(1, Some(weight), 0);
            enc.set_buffer(2, Some(bias), 0);
            enc.set_buffer(3, Some(out), out_off_bytes as u64);
            enc.set_buffer(4, Some(ab.handle()), 0);
            enc.set_threadgroup_memory_length(0, shmem);
        })
    }

    /// dst[off] = src (copy `n` elems into a possibly-offset destination plane).
    pub fn rwkv7_copy_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src: &PinnedBuffer,
        dst: &PinnedBuffer,
        dst_off_bytes: usize,
        n: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_copy", n as u32, |enc| {
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst), dst_off_bytes as u64);
            enc.set_u32(2, n as u32);
        })
    }
}
