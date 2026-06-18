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
    rwkv7_add_into_flat_tcb, rwkv7_add_into_tcb, rwkv7_channel_mix_shift_multiseq_tcb,
    rwkv7_channel_mix_shift_tcb, rwkv7_copy_tcb, rwkv7_decay_act_multiseq_tcb, rwkv7_decay_act_tcb,
    rwkv7_gemv_f32_off_tcb, rwkv7_gemv_f32_xoff_yoff_tcb, rwkv7_kk_kmix_multiseq_tcb,
    rwkv7_kk_kmix_tcb, rwkv7_layernorm_multiseq_tcb, rwkv7_layernorm_tcb,
    rwkv7_relu_sq_inplace_tcb, rwkv7_shift_writeback_multiseq_tcb, rwkv7_sigmoid_bias_multiseq_tcb,
    rwkv7_sigmoid_bias_tcb, rwkv7_sigmoid_inplace_tcb, rwkv7_tanh_inplace_tcb,
    rwkv7_token_shift_lerp_multiseq_tcb, rwkv7_token_shift_lerp_tcb,
    rwkv7_value_residual_mix_multiseq_tcb, rwkv7_value_residual_mix_tcb,
    rwkv7_wkv_decode_multiseq_tcb, rwkv7_wkv_decode_tcb, RwkvDecodeArena,
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

    /// Per-layer recurrent state + per-token scratch for the RWKV-7 GPU decode
    /// (single-stream or B-stream continuous batch — see `new_with_batch`).
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
        /// Number of independent streams the arena is sized for (1 = the
        /// single-stream decode; B = the continuous-batch multiseq decode).
        /// EVERY buffer above is sized for `batch` streams; the single-stream
        /// dispatchers index slot 0 (offset 0) and are unchanged.
        pub batch: usize,
    }

    impl RwkvDecodeArena {
        /// Single-stream arena (B = 1). Thin wrapper over `new_with_batch`.
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
            Self::new_with_batch(
                ctx,
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
                1,
            )
        }

        /// Continuous-batch arena for `batch` INDEPENDENT streams.
        ///
        /// Buffer layouts (the single-stream `new` is exactly `batch = 1`):
        /// - State planes (`wkv_state`, `att_shift`, `ffn_shift`): STREAM-major,
        ///   `[stream][layer][..]`. The WKV recurrence + token-shift select one
        ///   `(stream, layer)` window; per-stream state never mixes.
        /// - Activation scratch that feeds the batched projection GEMV (`x`,
        ///   `att_in`, `out_wkv`, `xk_ffn`, `ffn_k`, `r`, `w`, `k`, `v`, `a`, …):
        ///   `(B, dim)` ROW-major, i.e. `buf[b*dim + i]` — exactly the layout
        ///   `gemm_q4_k_m_batched_v3w_predec` reads (`x_batch[b*cols + off]`).
        /// - `xs` (the 6-slot lerp output): `(slot, B, n)` so each slot's
        ///   `(B, n)` block is contiguous and feeds one batched GEMV directly
        ///   (slot byte offset = `slot * B * n * 4`).
        /// - `logits`: `(B, vocab)`.
        #[allow(clippy::too_many_arguments)]
        pub fn new_with_batch(
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
            batch: usize,
        ) -> Self {
            let f = std::mem::size_of::<f32>();
            let b = batch.max(1);
            let nb = |elems: usize| ctx.new_buffer(elems.max(1) * f);
            let s_per_layer = head_count * head_size * head_size;
            Self {
                wkv_state: nb(b * n_layer * s_per_layer),
                att_shift: nb(b * n_layer * n_embd),
                ffn_shift: nb(b * n_layer * n_embd),

                x: nb(b * n_embd),
                att_in: nb(b * n_embd),
                ffn_inp: nb(b * n_embd),
                ffn_in: nb(b * n_embd),
                x_norm: nb(b * n_embd),
                xs: nb(6 * b * n_embd),
                xk_ffn: nb(b * n_embd),
                r: nb(b * n_embd),
                w: nb(b * n_embd),
                w_raw: nb(b * n_embd),
                k: nb(b * n_embd),
                v: nb(b * n_embd),
                a: nb(b * n_embd),
                a_op: nb(b * n_embd),
                b_op: nb(b * n_embd),
                gate: nb(b * n_embd),
                out_wkv: nb(b * n_embd),
                cur: nb(b * n_embd),
                cmix: nb(b * n_embd),
                v_first: nb(b * n_embd),
                v_mix: nb(b * n_embd),
                w_lo: nb(b * decay_lora),
                a_lo: nb(b * iclr_lora),
                v_lo: nb(b * value_res_lora),
                g_lo: nb(b * gate_lora),
                ffn_k: nb(b * n_ff),
                logits: nb(b * vocab_size),
                token: ctx.new_buffer(b * std::mem::size_of::<u32>()),

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
                batch: b,
            }
        }

        /// Byte offset of `layer`'s window in the (single-stream) WKV state
        /// plane. For the multiseq arena use `wkv_slot_layer_byte_offset`.
        pub fn wkv_layer_byte_offset(&self, layer: usize) -> usize {
            layer * self.head_count * self.head_size * self.head_size * std::mem::size_of::<f32>()
        }
        /// Byte offset of `layer`'s window in a (single-stream) token-shift plane.
        pub fn shift_layer_byte_offset(&self, layer: usize) -> usize {
            layer * self.n_embd * std::mem::size_of::<f32>()
        }

        /// Per-stream WKV state window: byte offset of `(slot, layer)` in the
        /// stream-major `wkv_state` plane. `slot` in `0..batch`.
        pub fn wkv_slot_layer_byte_offset(&self, slot: usize, layer: usize) -> usize {
            let s_per_layer = self.head_count * self.head_size * self.head_size;
            (slot * self.n_layer + layer) * s_per_layer * std::mem::size_of::<f32>()
        }
        /// Per-stream token-shift window: byte offset of `(slot, layer)` in a
        /// stream-major `att_shift`/`ffn_shift` plane. `slot` in `0..batch`.
        pub fn shift_slot_layer_byte_offset(&self, slot: usize, layer: usize) -> usize {
            (slot * self.n_layer + layer) * self.n_embd * std::mem::size_of::<f32>()
        }
        /// Byte offset of `slot`'s `(B, n)` block within the `(slot, B, n)` `xs`
        /// lerp buffer — the contiguous activation block feeding one batched
        /// projection GEMV for projection-slot `proj_slot` (r,w,k,v,a,g).
        pub fn xs_proj_slot_byte_offset(&self, proj_slot: usize) -> usize {
            proj_slot * self.batch * self.n_embd * std::mem::size_of::<f32>()
        }

        /// Zero the persistent recurrent state for ALL `batch` streams (start of
        /// B fresh sequences). Shared-storage buffers on Apple Silicon are
        /// CPU-visible, so this is a plain memset on the unified-memory backing.
        pub fn reset_state(&self) {
            let s_per_layer = self.head_count * self.head_size * self.head_size;
            for (buf, elems) in [
                (&self.wkv_state, self.batch * self.n_layer * s_per_layer),
                (&self.att_shift, self.batch * self.n_layer * self.n_embd),
                (&self.ffn_shift, self.batch * self.n_layer * self.n_embd),
            ] {
                let ptr = buf.contents() as *mut f32;
                unsafe { std::ptr::write_bytes(ptr, 0, elems) };
            }
        }

        /// Zero the recurrent state of ONE stream (its sequence finished; reuse
        /// the slot for a new sequence without disturbing the other streams —
        /// the continuous-batch reuse path). `slot` in `0..batch`.
        pub fn reset_slot(&self, slot: usize) {
            let s_per_layer = self.head_count * self.head_size * self.head_size;
            // wkv: stream-major, the slot's whole n_layer window is contiguous.
            unsafe {
                let base =
                    (self.wkv_state.contents() as *mut f32).add(slot * self.n_layer * s_per_layer);
                std::ptr::write_bytes(base, 0, self.n_layer * s_per_layer);
                for buf in [&self.att_shift, &self.ffn_shift] {
                    let p = (buf.contents() as *mut f32).add(slot * self.n_layer * self.n_embd);
                    std::ptr::write_bytes(p, 0, self.n_layer * self.n_embd);
                }
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

    /// `rwkv7_gemv_f32_off_tcb` with a y (output) byte offset too, so a (B,rows)
    /// output block can be filled one stream at a time (`out[bi*rows ..]`). Used
    /// by the B-stream F32 projection loop (LoRA + the all-F32 GGUF fallback);
    /// the single-stream callers keep the zero-output-offset helper above.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_gemv_f32_xoff_yoff_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w: &PinnedBuffer,
        rows: usize,
        cols: usize,
        x: &PinnedBuffer,
        x_off_bytes: usize,
        out: &PinnedBuffer,
        out_off_bytes: usize,
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
                enc.set_buffer(2, Some(out), out_off_bytes as u64);
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

    // ── RWKV-7 CONTINUOUS-BATCH (multi-seq) dispatchers ──────────────────────
    //
    // Each is the B-stream twin of one of the single-stream dispatchers above.
    // The LAYOUT CONTRACT (see `RwkvDecodeArena::new_with_batch`):
    //   - activation buffers (att_in, r, w, k, v, a, a_op, b_op, gate, out, ffn_in,
    //     xk, ...) are (B, dim) ROW-major and passed at offset 0.
    //   - the WKV state plane is STREAM-major; the multiseq WKV kernel indexes the
    //     stream itself, so it takes the whole plane + the per-stream/per-layer
    //     strides (`state_stream_stride`, `state_layer_base`).
    //   - the two token-shift state planes are STREAM-major [stream][layer][n];
    //     the shift kernels read them with the per-stream stride `n_layer*n` from
    //     the (stream 0, layer li) base, and the write-back scatters (B,n) back.
    //   - xs (lerp output) is (slot, B, n).
    // The pure-elementwise glue (tanh / sigmoid_inplace / relu_sq / copy / add)
    // needs NO multiseq variant — the single-stream dispatcher already takes `n`,
    // so the caller passes `n = B*dim` and the op is identical per element.

    /// B-stream token-shift + per-slot lerp. `att_in` is (B,n); `shift_plane` is
    /// the STREAM-major att_shift plane passed at the (stream 0, layer li) byte
    /// base; `stream_stride_elems` = n_layer*n strides between streams. Writes
    /// `xs` in (slot, B, n) layout.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_token_shift_lerp_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        att_in: &PinnedBuffer,
        shift_plane: &PinnedBuffer,
        shift_base_bytes: usize,
        stream_stride_elems: usize,
        lerp: &PinnedBuffer,
        xs: &PinnedBuffer,
        n: usize,
        n_slots: usize,
        batch: usize,
        fresh: bool,
    ) -> Result<()> {
        let total = (batch * n) as u32;
        let mut ab = KernelArgBuffer::new(
            tcb.ctx,
            &[
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
            ],
        )?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, n_slots as u32);
        ab.set_u32(2, batch as u32);
        ab.set_u32(3, fresh as u32);
        ab.set_u32(4, stream_stride_elems as u32);
        dispatch_n(tcb, "rwkv7_token_shift_lerp_multiseq", total, |enc| {
            enc.set_buffer(0, Some(att_in), 0);
            enc.set_buffer(1, Some(shift_plane), shift_base_bytes as u64);
            enc.set_buffer(2, Some(lerp), 0);
            enc.set_buffer(3, Some(xs), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// B-stream channel-mix token-shift + single lerp. Same stream-major `x_prev`
    /// contract as `rwkv7_token_shift_lerp_multiseq_tcb`; writes `xk` (B,n).
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_channel_mix_shift_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        ffn_in: &PinnedBuffer,
        shift_plane: &PinnedBuffer,
        shift_base_bytes: usize,
        stream_stride_elems: usize,
        lerp_k: &PinnedBuffer,
        xk: &PinnedBuffer,
        n: usize,
        batch: usize,
        fresh: bool,
    ) -> Result<()> {
        let total = (batch * n) as u32;
        let mut ab = KernelArgBuffer::new(
            tcb.ctx,
            &[
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
            ],
        )?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, 1);
        ab.set_u32(2, batch as u32);
        ab.set_u32(3, fresh as u32);
        ab.set_u32(4, stream_stride_elems as u32);
        dispatch_n(tcb, "rwkv7_channel_mix_shift_multiseq", total, |enc| {
            enc.set_buffer(0, Some(ffn_in), 0);
            enc.set_buffer(1, Some(shift_plane), shift_base_bytes as u64);
            enc.set_buffer(2, Some(lerp_k), 0);
            enc.set_buffer(3, Some(xk), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// Scatter a (B,n) row-major plane (`src` = att_in/ffn_in) into the
    /// STREAM-major token-shift state plane for layer li (the B-stream analogue of
    /// `rwkv7_copy_tcb` into a per-layer window). `dst_plane` is passed at the
    /// (stream 0, layer li) byte base; `stream_stride_elems` = n_layer*n.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_shift_writeback_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        src: &PinnedBuffer,
        dst_plane: &PinnedBuffer,
        dst_base_bytes: usize,
        stream_stride_elems: usize,
        n: usize,
        batch: usize,
    ) -> Result<()> {
        let total = (batch * n) as u32;
        let mut ab =
            KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, batch as u32);
        ab.set_u32(2, stream_stride_elems as u32);
        dispatch_n(tcb, "rwkv7_shift_writeback_multiseq", total, |enc| {
            enc.set_buffer(0, Some(src), 0);
            enc.set_buffer(1, Some(dst_plane), dst_base_bytes as u64);
            enc.set_buffer(2, Some(ab.handle()), 0);
        })
    }

    /// B independent LayerNorms (one threadgroup per stream). `x`/`out` are (B,n)
    /// row-major; `weight`/`bias` are (n,) shared. The `x_off_bytes`/`out_off_bytes`
    /// select a layer window when the buffer is a multi-layer plane.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_layernorm_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &PinnedBuffer,
        x_off_bytes: usize,
        weight: &PinnedBuffer,
        bias: &PinnedBuffer,
        out: &PinnedBuffer,
        out_off_bytes: usize,
        hidden: usize,
        batch: usize,
        eps: f32,
    ) -> Result<()> {
        let shmem = (LN_TG as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab =
            KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::F32])?;
        ab.set_u32(0, hidden as u32);
        ab.set_u32(1, batch as u32);
        ab.set_f32(2, eps);
        // One threadgroup per stream → grid = (batch * LN_TG).
        tcb.dispatch_threads(
            "rwkv7_layernorm_multiseq",
            (batch as u32 * LN_TG, 1, 1),
            (LN_TG, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(x), x_off_bytes as u64);
                enc.set_buffer(1, Some(weight), 0);
                enc.set_buffer(2, Some(bias), 0);
                enc.set_buffer(3, Some(out), out_off_bytes as u64);
                enc.set_buffer(4, Some(ab.handle()), 0);
                enc.set_threadgroup_memory_length(0, shmem);
            },
        )
    }

    /// B-stream kk / k-mix. One threadgroup per (stream, head). `k`/`a`/`a_op`/
    /// `b_op` are (B,n); `k_k`/`k_a` are (n,) shared.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_kk_kmix_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        k: &PinnedBuffer,
        k_k: &PinnedBuffer,
        k_a: &PinnedBuffer,
        a: &PinnedBuffer,
        a_op: &PinnedBuffer,
        b_op: &PinnedBuffer,
        head_size: usize,
        head_count: usize,
        batch: usize,
    ) -> Result<()> {
        let hs = head_size as u32;
        let shmem = (hs as u64) * std::mem::size_of::<f32>() as u64;
        let mut ab =
            KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, hs);
        ab.set_u32(1, head_count as u32);
        ab.set_u32(2, (head_count * head_size) as u32); // args.n = n_embd, not batch
                                                        // Grid: (B * head_count * hs); threadgroup = hs; tg index = b*head_count+head.
        tcb.dispatch_threads(
            "rwkv7_kk_kmix_multiseq",
            (batch as u32 * head_count as u32 * hs, 1, 1),
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

    /// B-stream WKV-7 recurrence + per-head group-norm + bonus + gate. `state` is
    /// the whole STREAM-major plane; `state_stream_stride_elems` strides between
    /// streams (= n_layer*head_count*hs*hs) and `state_layer_base_elems` is this
    /// layer's element offset within a stream window (= layer*head_count*hs*hs).
    /// All activation inputs/outputs are (B,n); `r_k`/`ln_w`/`ln_b` are (n,) shared.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_wkv_decode_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        state: &PinnedBuffer,
        state_stream_stride_elems: usize,
        state_layer_base_elems: usize,
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
        n: usize,
        head_size: usize,
        head_count: usize,
        batch: usize,
        gn_eps: f32,
        has_gate: bool,
    ) -> Result<()> {
        let hs = head_size as u32;
        let shmem = (hs as u64) * std::mem::size_of::<f32>() as u64;
        // ArgbufRwkv7WkvMs { head_size, head_count, n, batch, state_stream_stride,
        //                    state_layer_base, gn_eps, has_gate }
        let mut ab = KernelArgBuffer::new(
            tcb.ctx,
            &[
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::U32,
                ArgLayout::F32,
                ArgLayout::U32,
            ],
        )?;
        ab.set_u32(0, hs);
        ab.set_u32(1, head_count as u32);
        ab.set_u32(2, n as u32);
        ab.set_u32(3, batch as u32);
        ab.set_u32(4, state_stream_stride_elems as u32);
        ab.set_u32(5, state_layer_base_elems as u32);
        ab.set_f32(6, gn_eps);
        ab.set_u32(7, has_gate as u32);
        // Grid: (B * head_count * hs); threadgroup = hs; tg index = b*head_count+head.
        tcb.dispatch_threads(
            "rwkv7_wkv_decode_multiseq",
            (batch as u32 * head_count as u32 * hs, 1, 1),
            (hs, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(state), 0);
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

    /// B-stream decay activation: w = exp(-0.606531 * sigmoid(w_raw + w0[i%n])).
    /// `w_raw`/`w` are (B,n); `w0` is (n,) shared.
    pub fn rwkv7_decay_act_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        w_raw: &PinnedBuffer,
        w0: &PinnedBuffer,
        w: &PinnedBuffer,
        n: usize,
        batch: usize,
    ) -> Result<()> {
        let total = (batch * n) as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, batch as u32);
        dispatch_n(tcb, "rwkv7_decay_act_multiseq", total, |enc| {
            enc.set_buffer(0, Some(w_raw), 0);
            enc.set_buffer(1, Some(w0), 0);
            enc.set_buffer(2, Some(w), 0);
            enc.set_buffer(3, Some(ab.handle()), 0);
        })
    }

    /// B-stream sigmoid-with-bias in place: x = sigmoid(x + bias[i%n]). `x` is
    /// (B,n); `bias` is (n,) shared.
    pub fn rwkv7_sigmoid_bias_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        x: &PinnedBuffer,
        bias: &PinnedBuffer,
        n: usize,
        batch: usize,
    ) -> Result<()> {
        let total = (batch * n) as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, batch as u32);
        dispatch_n(tcb, "rwkv7_sigmoid_bias_multiseq", total, |enc| {
            enc.set_buffer(0, Some(x), 0);
            enc.set_buffer(1, Some(bias), 0);
            enc.set_buffer(2, Some(ab.handle()), 0);
        })
    }

    /// B-stream value-residual mix: v += (v_first - v) * sigmoid(v_mix + v0[i%n]).
    /// `v`/`v_first`/`v_mix` are (B,n) (v_first is per-stream); `v0` is (n,) shared.
    #[allow(clippy::too_many_arguments)]
    pub fn rwkv7_value_residual_mix_multiseq_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        v: &PinnedBuffer,
        v_first: &PinnedBuffer,
        v_mix: &PinnedBuffer,
        v0: &PinnedBuffer,
        n: usize,
        batch: usize,
    ) -> Result<()> {
        let total = (batch * n) as u32;
        let mut ab = KernelArgBuffer::new(tcb.ctx, &[ArgLayout::U32, ArgLayout::U32])?;
        ab.set_u32(0, n as u32);
        ab.set_u32(1, batch as u32);
        dispatch_n(tcb, "rwkv7_value_residual_mix_multiseq", total, |enc| {
            enc.set_buffer(0, Some(v), 0);
            enc.set_buffer(1, Some(v_first), 0);
            enc.set_buffer(2, Some(v_mix), 0);
            enc.set_buffer(3, Some(v0), 0);
            enc.set_buffer(4, Some(ab.handle()), 0);
        })
    }

    /// out[g] = a[g] + b[g] over a flat (B*n) range — B-stream residual add.
    pub fn rwkv7_add_into_flat_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        a: &PinnedBuffer,
        b: &PinnedBuffer,
        out: &PinnedBuffer,
        total: usize,
    ) -> Result<()> {
        dispatch_n(tcb, "rwkv7_add_into_flat", total as u32, |enc| {
            enc.set_buffer(0, Some(a), 0);
            enc.set_buffer(1, Some(b), 0);
            enc.set_buffer(2, Some(out), 0);
            enc.set_u32(3, total as u32);
        })
    }
}
