#[cfg(target_os = "macos")]
pub use arena_imp::DenseDecodeArena;

#[cfg(target_os = "macos")]
mod arena_imp {
    use crate::metal::{MetalContext, PinnedBuffer};

    pub struct DenseDecodeArena {
        pub q_buf: PinnedBuffer,
        pub k_token_buf: PinnedBuffer,
        pub v_token_buf: PinnedBuffer,
        pub k_cache_buf: PinnedBuffer,
        pub v_cache_buf: PinnedBuffer,
        // f16 KV cache (HAWKING_QWEN_F16_KV=1). Lazy-init via
        // `ensure_f16_kv` — None until the flag is observed, then allocated
        // once at half the f32 cache footprint. When ON these become the
        // single source of truth for K/V; the f32 buffers above stay
        // allocated but unread (kept so the flag-OFF path is byte-identical
        // and so arena construction need not know the flag). Layout mirrors
        // the f32 cache: layer-major, n_layers * max_seq * kv_dim halfs.
        pub k_cache_f16_buf: Option<PinnedBuffer>,
        pub v_cache_f16_buf: Option<PinnedBuffer>,
        // int4 KV cache (HAWKING_QWEN_INT4_KV=1). Lazy-init via `ensure_int4_kv`.
        // Per-row symmetric int4: each kv-head row is head_dim/2 packed bytes +
        // one f16 scale. Layer-major: rows = n_layers * max_seq * n_kv_heads.
        pub k_cache_int4_packed: Option<PinnedBuffer>,
        pub v_cache_int4_packed: Option<PinnedBuffer>,
        pub k_cache_int4_scales: Option<PinnedBuffer>,
        pub v_cache_int4_scales: Option<PinnedBuffer>,
        pub attn_out_buf: PinnedBuffer,
        pub x_buf: PinnedBuffer,
        pub x_norm_buf: PinnedBuffer,
        pub ffn_gate_buf: PinnedBuffer,
        pub ffn_up_buf: PinnedBuffer,
        pub ffn_act_buf: PinnedBuffer,
        pub ffn_down_buf: PinnedBuffer,
        pub o_proj_out_buf: PinnedBuffer,
        pub logits_buf: PinnedBuffer,
        pub token_buf: PinnedBuffer,
        // Greedy token-only lane: B u32 token ids from batched GPU argmax.
        // Sized for max_batch; never reallocated.
        pub token_batch_buf: PinnedBuffer,

        // W4A8 scratch (HAWKING_QWEN_W4A8=1). Lazy-init via `ensure_w4a8`
        // — None until the flag is observed, then allocated once. Sized
        // for the three quantize sites:
        //   x_norm_int8/scales:    hidden bytes / hidden/256 f32 (q_proj,
        //                          ffn_gate, ffn_up, LM head — quant once
        //                          per rmsnorm, dispatch up to 4 GEMVs).
        //   attn_out_int8/scales:  q_dim bytes (o_proj).
        //   ffn_act_int8/scales:   intermediate bytes (ffn_down).
        pub x_norm_int8: Option<PinnedBuffer>,
        pub x_norm_scales: Option<PinnedBuffer>,
        pub attn_out_int8: Option<PinnedBuffer>,
        pub attn_out_scales: Option<PinnedBuffer>,
        pub ffn_act_int8: Option<PinnedBuffer>,
        pub ffn_act_scales: Option<PinnedBuffer>,

        // P3 — B-wide scratch buffers for batched prefill
        // (forward_tokens_batch_tcb). Sized for max_batch.
        // Same dims as per-token buffers but with leading B stride.
        pub max_batch: usize,
        pub q_buf_batch: PinnedBuffer,
        pub k_token_buf_batch: PinnedBuffer,
        pub v_token_buf_batch: PinnedBuffer,
        pub attn_out_buf_batch: PinnedBuffer,
        pub x_buf_batch: PinnedBuffer,
        pub x_norm_buf_batch: PinnedBuffer,
        pub ffn_gate_buf_batch: PinnedBuffer,
        pub ffn_up_buf_batch: PinnedBuffer,
        pub ffn_act_buf_batch: PinnedBuffer,
        pub ffn_down_buf_batch: PinnedBuffer,
        pub o_proj_out_buf_batch: PinnedBuffer,

        pub n_layers: usize,
        pub n_heads: usize,
        pub n_kv_heads: usize,
        pub head_dim: usize,
        pub hidden: usize,
        pub intermediate: usize,
        pub vocab_size: usize,
        pub max_seq: usize,
    }

    impl DenseDecodeArena {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            ctx: &MetalContext,
            n_layers: usize,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            hidden: usize,
            intermediate: usize,
            vocab_size: usize,
            max_seq: usize,
        ) -> Self {
            // max_batch=8 — covers the v3w widened kernel (B in 1..=8).
            // Cost vs B=4: doubles the B-wide scratch buffer footprint,
            // still trivial vs weight memory.
            Self::new_with_batch(
                ctx,
                n_layers,
                n_heads,
                n_kv_heads,
                head_dim,
                hidden,
                intermediate,
                vocab_size,
                max_seq,
                8,
            )
        }

        #[allow(clippy::too_many_arguments)]
        pub fn new_with_batch(
            ctx: &MetalContext,
            n_layers: usize,
            n_heads: usize,
            n_kv_heads: usize,
            head_dim: usize,
            hidden: usize,
            intermediate: usize,
            vocab_size: usize,
            max_seq: usize,
            max_batch: usize,
        ) -> Self {
            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;
            let kv_cache_bytes_per_layer = max_seq * kv_dim * std::mem::size_of::<f32>();
            let total_kv_bytes = n_layers * kv_cache_bytes_per_layer;
            let f32_bytes = std::mem::size_of::<f32>();
            let b = max_batch.max(1);

            Self {
                q_buf: ctx.new_buffer(q_dim * f32_bytes),
                k_token_buf: ctx.new_buffer(kv_dim * f32_bytes),
                v_token_buf: ctx.new_buffer(kv_dim * f32_bytes),
                k_cache_buf: ctx.new_buffer(total_kv_bytes),
                v_cache_buf: ctx.new_buffer(total_kv_bytes),
                // f16 KV — lazily allocated by `ensure_f16_kv` only when
                // HAWKING_QWEN_F16_KV=1; None keeps the OFF path byte-identical.
                k_cache_f16_buf: None,
                v_cache_f16_buf: None,
                k_cache_int4_packed: None,
                v_cache_int4_packed: None,
                k_cache_int4_scales: None,
                v_cache_int4_scales: None,
                attn_out_buf: ctx.new_buffer(q_dim * f32_bytes),
                x_buf: ctx.new_buffer(hidden * f32_bytes),
                x_norm_buf: ctx.new_buffer(hidden * f32_bytes),
                ffn_gate_buf: ctx.new_buffer(intermediate * f32_bytes),
                ffn_up_buf: ctx.new_buffer(intermediate * f32_bytes),
                ffn_act_buf: ctx.new_buffer(intermediate * f32_bytes),
                ffn_down_buf: ctx.new_buffer(hidden * f32_bytes),
                o_proj_out_buf: ctx.new_buffer(hidden * f32_bytes),
                logits_buf: ctx.new_buffer(vocab_size * f32_bytes),
                token_buf: ctx.new_buffer(std::mem::size_of::<u32>()),
                token_batch_buf: ctx.new_buffer(b * std::mem::size_of::<u32>()),

                max_batch: b,
                q_buf_batch: ctx.new_buffer(b * q_dim * f32_bytes),
                k_token_buf_batch: ctx.new_buffer(b * kv_dim * f32_bytes),
                v_token_buf_batch: ctx.new_buffer(b * kv_dim * f32_bytes),
                attn_out_buf_batch: ctx.new_buffer(b * q_dim * f32_bytes),
                x_buf_batch: ctx.new_buffer(b * hidden * f32_bytes),
                x_norm_buf_batch: ctx.new_buffer(b * hidden * f32_bytes),
                ffn_gate_buf_batch: ctx.new_buffer(b * intermediate * f32_bytes),
                ffn_up_buf_batch: ctx.new_buffer(b * intermediate * f32_bytes),
                ffn_act_buf_batch: ctx.new_buffer(b * intermediate * f32_bytes),
                ffn_down_buf_batch: ctx.new_buffer(b * hidden * f32_bytes),
                o_proj_out_buf_batch: ctx.new_buffer(b * hidden * f32_bytes),

                x_norm_int8: None,
                x_norm_scales: None,
                attn_out_int8: None,
                attn_out_scales: None,
                ffn_act_int8: None,
                ffn_act_scales: None,

                n_layers,
                n_heads,
                n_kv_heads,
                head_dim,
                hidden,
                intermediate,
                vocab_size,
                max_seq,
            }
        }

        pub fn kv_layer_byte_offset(&self, layer: usize) -> usize {
            layer * self.max_seq * self.n_kv_heads * self.head_dim * std::mem::size_of::<f32>()
        }

        /// f16 counterpart of [`kv_layer_byte_offset`]: byte offset of
        /// `layer`'s window in the f16 KV cache. Identical element math, half
        /// the element size — callers pass this as k_off_bytes / v_off_bytes
        /// to `mha_decode_f16kv_tcb` / `mha_decode_f16kv_batched_tcb`.
        pub fn kv_f16_layer_byte_offset(&self, layer: usize) -> usize {
            layer
                * self.max_seq
                * self.n_kv_heads
                * self.head_dim
                * std::mem::size_of::<half::f16>()
        }

        /// Lazy-init the W4A8 scratch buffers. Called once on the first
        /// forward pass when `HAWKING_QWEN_W4A8=1`. No-op on subsequent
        /// calls. Total footprint at Qwen-3B: ~16 KB (negligible vs the
        /// ~1.6 GB weight + 60 MB KV cache).
        pub fn ensure_w4a8(&mut self, ctx: &MetalContext) {
            if self.x_norm_int8.is_some() {
                return;
            }
            let q_dim = self.n_heads * self.head_dim;
            let f32_bytes = std::mem::size_of::<f32>();
            let h_blocks = self.hidden / 256;
            let q_blocks = q_dim / 256;
            let ffn_blocks = self.intermediate / 256;
            self.x_norm_int8 = Some(ctx.new_buffer(self.hidden));
            self.x_norm_scales = Some(ctx.new_buffer(h_blocks * f32_bytes));
            self.attn_out_int8 = Some(ctx.new_buffer(q_dim));
            self.attn_out_scales = Some(ctx.new_buffer(q_blocks * f32_bytes));
            self.ffn_act_int8 = Some(ctx.new_buffer(self.intermediate));
            self.ffn_act_scales = Some(ctx.new_buffer(ffn_blocks * f32_bytes));
        }

        /// Lazy-init the f16 KV cache buffers. Called once on the first
        /// forward pass when `HAWKING_QWEN_F16_KV=1`. No-op on subsequent
        /// calls. Footprint = 2 * n_layers * max_seq * kv_dim * 2 bytes
        /// (half the f32 cache). The f32 `k_cache_buf` / `v_cache_buf`
        /// remain allocated but unread while the flag is on.
        pub fn ensure_f16_kv(&mut self, ctx: &MetalContext) {
            if self.k_cache_f16_buf.is_some() {
                return;
            }
            let kv_dim = self.n_kv_heads * self.head_dim;
            let f16_bytes = std::mem::size_of::<half::f16>();
            let total_kv_f16_bytes = self.n_layers * self.max_seq * kv_dim * f16_bytes;
            self.k_cache_f16_buf = Some(ctx.new_buffer(total_kv_f16_bytes));
            self.v_cache_f16_buf = Some(ctx.new_buffer(total_kv_f16_bytes));
        }

        /// Byte offset of `layer`'s window in an int4 PACKED plane (head_dim/2
        /// bytes/row; rows = max_seq * n_kv_heads per layer).
        pub fn kv_int4_layer_byte_offset(&self, layer: usize) -> usize {
            layer * self.max_seq * self.n_kv_heads * (self.head_dim / 2)
        }

        /// ROW offset of `layer`'s window in an int4 SCALES plane (one f16/row).
        pub fn kv_int4_layer_scale_offset(&self, layer: usize) -> usize {
            layer * self.max_seq * self.n_kv_heads
        }

        /// First ROW index for a per-token int4 append at (layer, seq_slot):
        /// (layer*max_seq + seq_slot) * n_kv_heads. Pass as `dst_row_base`.
        pub fn kv_int4_dst_row_base(&self, layer: usize, seq_slot: usize) -> usize {
            (layer * self.max_seq + seq_slot) * self.n_kv_heads
        }

        /// Lazy-init the int4 KV cache (HAWKING_QWEN_INT4_KV=1). ~1/4 the f32
        /// cache: packed = 2 * rows * (head_dim/2) bytes + scales = 2 * rows * 2.
        pub fn ensure_int4_kv(&mut self, ctx: &MetalContext) {
            if self.k_cache_int4_packed.is_some() {
                return;
            }
            let rows = self.n_layers * self.max_seq * self.n_kv_heads;
            let packed_bytes = rows * (self.head_dim / 2);
            let scales_bytes = rows * std::mem::size_of::<half::f16>();
            self.k_cache_int4_packed = Some(ctx.new_buffer(packed_bytes));
            self.v_cache_int4_packed = Some(ctx.new_buffer(packed_bytes));
            self.k_cache_int4_scales = Some(ctx.new_buffer(scales_bytes));
            self.v_cache_int4_scales = Some(ctx.new_buffer(scales_bytes));
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct DenseDecodeArena {
    _priv: std::marker::PhantomData<()>,
}
