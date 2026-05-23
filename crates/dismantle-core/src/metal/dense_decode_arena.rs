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
            let q_dim = n_heads * head_dim;
            let kv_dim = n_kv_heads * head_dim;
            let kv_cache_bytes_per_layer =
                max_seq * kv_dim * std::mem::size_of::<f32>();
            let total_kv_bytes = n_layers * kv_cache_bytes_per_layer;

            Self {
                q_buf: ctx.new_buffer(q_dim * std::mem::size_of::<f32>()),
                k_token_buf: ctx.new_buffer(kv_dim * std::mem::size_of::<f32>()),
                v_token_buf: ctx.new_buffer(kv_dim * std::mem::size_of::<f32>()),
                k_cache_buf: ctx.new_buffer(total_kv_bytes),
                v_cache_buf: ctx.new_buffer(total_kv_bytes),
                attn_out_buf: ctx.new_buffer(q_dim * std::mem::size_of::<f32>()),
                x_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                x_norm_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                ffn_gate_buf: ctx.new_buffer(intermediate * std::mem::size_of::<f32>()),
                ffn_up_buf: ctx.new_buffer(intermediate * std::mem::size_of::<f32>()),
                ffn_act_buf: ctx.new_buffer(intermediate * std::mem::size_of::<f32>()),
                ffn_down_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                o_proj_out_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                logits_buf: ctx.new_buffer(vocab_size * std::mem::size_of::<f32>()),
                token_buf: ctx.new_buffer(std::mem::size_of::<u32>()),

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
