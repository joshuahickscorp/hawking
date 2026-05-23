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
            Self::new_with_batch(
                ctx, n_layers, n_heads, n_kv_heads, head_dim,
                hidden, intermediate, vocab_size, max_seq, 4,
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
            let kv_cache_bytes_per_layer =
                max_seq * kv_dim * std::mem::size_of::<f32>();
            let total_kv_bytes = n_layers * kv_cache_bytes_per_layer;
            let f32_bytes = std::mem::size_of::<f32>();
            let b = max_batch.max(1);

            Self {
                q_buf: ctx.new_buffer(q_dim * f32_bytes),
                k_token_buf: ctx.new_buffer(kv_dim * f32_bytes),
                v_token_buf: ctx.new_buffer(kv_dim * f32_bytes),
                k_cache_buf: ctx.new_buffer(total_kv_bytes),
                v_cache_buf: ctx.new_buffer(total_kv_bytes),
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
    }
}

#[cfg(not(target_os = "macos"))]
pub struct DenseDecodeArena {
    _priv: std::marker::PhantomData<()>,
}
