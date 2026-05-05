//! Decode-arena: pre-allocated Metal buffer pool for the MLA attention
//! hot path. Eliminates per-dispatch buffer allocations by reusing
//! fixed-size GPU buffers across decode steps.
//!
//! On unified-memory Apple Silicon, `StorageModeShared` buffers are
//! CPU-writable via their `contents()` pointer with no GPU round-trip.
//! `write_buffer_bytes` fills them in O(n) without allocation overhead.
//!
//! Buffers sized at construction:
//!   q         — n_heads × q_head_dim f32 (fixed across all steps)
//!   c_kv      — max_seq × kv_lora_rank f32 (grown once per step via append)
//!   k_pe      — max_seq × qk_rope_head_dim f32
//!   attn_out  — n_heads × v_head_dim f32 (output of mla_decode_kernel)
//!   out       — hidden f32 (output of o_proj gemv)

#[cfg(target_os = "macos")]
pub use arena_imp::DecodeArena;

#[cfg(target_os = "macos")]
mod arena_imp {
    use crate::metal::{MetalContext, PinnedBuffer};

    pub struct DecodeArena {
        /// Fixed-size query buffer — filled from CPU before each dispatch.
        pub q: PinnedBuffer,
        /// Growing sequence buffer for compressed KV latent (c_kv).
        /// Pre-allocated to max_seq capacity; filled incrementally.
        pub c_kv: PinnedBuffer,
        /// Growing sequence buffer for RoPE-position K (k_pe).
        pub k_pe: PinnedBuffer,
        /// Output of mla_decode_kernel (constant shape).
        pub attn_out: PinnedBuffer,
        /// Output of o_proj gemv (constant shape = hidden).
        pub out: PinnedBuffer,
        /// Residual stream scratch — hidden × f32. Phase 4d buffer-arg path.
        pub x_buf: PinnedBuffer,
        /// RMSNorm output scratch — hidden × f32.
        pub x_norm_buf: PinnedBuffer,
        /// FFN output scratch — hidden × f32.
        pub ffn_out_buf: PinnedBuffer,
        /// MoE gate logits scratch — n_routed_experts × f32.
        pub moe_logits_buf: PinnedBuffer,
        /// Phase 7 bridge: f16 residual stream — hidden × f16. Written before
        /// each rmsnorm step so f16 bridge kernels can read pre-norm activations.
        pub x_f16_buf: PinnedBuffer,
        /// Cached sizes for bounds-checking at dispatch time.
        pub n_heads: usize,
        pub q_head_dim: usize,
        pub v_head_dim: usize,
        pub kv_lora_rank: usize,
        pub qk_rope_head_dim: usize,
        pub hidden: usize,
        pub max_seq: usize,
        pub n_routed_experts: usize,
    }

    impl DecodeArena {
        pub fn new(
            ctx: &MetalContext,
            n_heads: usize,
            qk_nope_head_dim: usize,
            qk_rope_head_dim: usize,
            v_head_dim: usize,
            kv_lora_rank: usize,
            hidden: usize,
            max_seq: usize,
            n_routed_experts: usize,
        ) -> Self {
            let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
            Self {
                q: ctx.new_buffer(n_heads * q_head_dim * std::mem::size_of::<f32>()),
                c_kv: ctx.new_buffer(max_seq * kv_lora_rank * std::mem::size_of::<f32>()),
                k_pe: ctx.new_buffer(max_seq * qk_rope_head_dim * std::mem::size_of::<f32>()),
                attn_out: ctx.new_buffer(n_heads * v_head_dim * std::mem::size_of::<f32>()),
                out: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                x_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                x_norm_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                ffn_out_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                moe_logits_buf: ctx.new_buffer(n_routed_experts.max(1) * std::mem::size_of::<f32>()),
                x_f16_buf: ctx.new_buffer(hidden * std::mem::size_of::<half::f16>()),
                n_heads,
                q_head_dim,
                v_head_dim,
                kv_lora_rank,
                qk_rope_head_dim,
                hidden,
                max_seq,
                n_routed_experts,
            }
        }

        /// Write the query vector into the arena q buffer.
        pub fn write_q(&self, q: &[f32]) {
            MetalContext::write_buffer_bytes(&self.q, bytemuck::cast_slice(q));
        }

        /// Append a single new c_kv entry at `seq_pos` (0-based).
        pub fn append_c_kv(&self, seq_pos: usize, entry: &[f32]) {
            let off = seq_pos * self.kv_lora_rank * std::mem::size_of::<f32>();
            let ptr = unsafe { (self.c_kv.contents() as *mut u8).add(off) as *mut f32 };
            unsafe { ptr.copy_from_nonoverlapping(entry.as_ptr(), entry.len()) };
        }

        /// Append a single new k_pe entry at `seq_pos`.
        pub fn append_k_pe(&self, seq_pos: usize, entry: &[f32]) {
            let off = seq_pos * self.qk_rope_head_dim * std::mem::size_of::<f32>();
            let ptr = unsafe { (self.k_pe.contents() as *mut u8).add(off) as *mut f32 };
            unsafe { ptr.copy_from_nonoverlapping(entry.as_ptr(), entry.len()) };
        }

        /// Read `hidden` f32 values out of the `out` buffer back to CPU.
        pub fn read_out(&self, dst: &mut [f32]) {
            let ptr = self.out.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }

        /// Write the residual stream into x_buf.
        pub fn write_x(&self, x: &[f32]) {
            MetalContext::write_buffer_bytes(&self.x_buf, bytemuck::cast_slice(x));
        }

        /// Read the residual stream back from x_buf to CPU.
        pub fn read_x(&self, dst: &mut [f32]) {
            let ptr = self.x_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }

        /// Convert f32 residual to f16 and write into x_f16_buf.
        /// Called before each rmsnorm step so f16 bridge kernels can read
        /// the pre-norm residual as half-precision.
        pub fn write_x_f16(&self, x: &[f32]) {
            let ptr = self.x_f16_buf.contents() as *mut half::f16;
            for (i, &v) in x.iter().enumerate() {
                unsafe { ptr.add(i).write(half::f16::from_f32(v)) };
            }
        }

        /// Write into x_norm_buf (rmsnorm output scratch).
        pub fn write_x_norm(&self, x: &[f32]) {
            MetalContext::write_buffer_bytes(&self.x_norm_buf, bytemuck::cast_slice(x));
        }

        /// Read x_norm_buf back to CPU.
        pub fn read_x_norm(&self, dst: &mut [f32]) {
            let ptr = self.x_norm_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }

        /// Write into ffn_out_buf (also used as a delta buffer for add_inplace in Wedge B).
        pub fn write_ffn_out(&self, x: &[f32]) {
            MetalContext::write_buffer_bytes(&self.ffn_out_buf, bytemuck::cast_slice(x));
        }

        /// Read ffn_out_buf back to CPU.
        pub fn read_ffn_out(&self, dst: &mut [f32]) {
            let ptr = self.ffn_out_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }

        /// Read moe_logits_buf back to CPU.
        pub fn read_moe_logits(&self, dst: &mut [f32]) {
            let ptr = self.moe_logits_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.n_routed_experts) };
            dst.copy_from_slice(src);
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct DecodeArena {
    _priv: (),
}
