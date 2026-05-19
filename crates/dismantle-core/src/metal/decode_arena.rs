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
        /// MoE top-k route IDs scratch — top_k_routed × u32.
        pub moe_route_ids_buf: PinnedBuffer,
        /// MoE top-k route weights scratch — top_k_routed × f32.
        pub moe_route_weights_buf: PinnedBuffer,
        /// Shared expert route ID scratch — always [0].
        pub shared_route_ids_buf: PinnedBuffer,
        /// MoE routed gate GEMV output scratch — top_k_routed × moe_intermediate.
        pub moe_routed_gate_out_buf: PinnedBuffer,
        /// MoE routed up GEMV output scratch — top_k_routed × moe_intermediate.
        pub moe_routed_up_out_buf: PinnedBuffer,
        /// MoE routed activation scratch — top_k_routed × moe_intermediate.
        pub moe_routed_act_buf: PinnedBuffer,
        /// MoE routed down GEMV output scratch — top_k_routed × hidden.
        pub moe_routed_out_buf: PinnedBuffer,
        /// MoE shared gate GEMV output scratch — n_shared_experts × moe_intermediate.
        pub moe_shared_gate_out_buf: PinnedBuffer,
        /// MoE shared up GEMV output scratch — n_shared_experts × moe_intermediate.
        pub moe_shared_up_out_buf: PinnedBuffer,
        /// MoE shared activation scratch — n_shared_experts × moe_intermediate.
        pub moe_shared_act_buf: PinnedBuffer,
        /// MoE shared down GEMV output scratch — hidden.
        pub moe_shared_out_buf: PinnedBuffer,
        /// Dense FFN gate GEMV output scratch — ffn_intermediate.
        pub dense_gate_out_buf: PinnedBuffer,
        /// Dense FFN up GEMV output scratch — ffn_intermediate.
        pub dense_up_out_buf: PinnedBuffer,
        /// Dense FFN activation scratch — ffn_intermediate.
        pub dense_act_buf: PinnedBuffer,
        /// v1.0.0-C: q-LoRA intermediate — q_lora_rank × f32.
        /// Output of q_a_proj GEMV; input to q_a_norm.
        pub q_lora_buf: PinnedBuffer,
        /// v1.0.0-C: kv-A projection output — (kv_lora_rank + qk_rope_head_dim) × f32.
        /// Output of kv_a_proj GEMV; split into c_kv_normed and k_pe_raw.
        pub kv_a_out_buf: PinnedBuffer,
        /// v1.0.0-C: normed q-LoRA — q_lora_rank × f32.
        /// Output of q_a_norm rmsnorm; input to q_b_proj GEMV.
        pub q_lora_normed_buf: PinnedBuffer,
        /// v1.0.0-C: normed kv-A latent — kv_lora_rank × f32.
        /// Output of kv_a_norm rmsnorm; read by CPU for mla_kv_append.
        pub c_kv_normed_buf: PinnedBuffer,
        /// v1.2.0-9: Route ID history for expert access stats.
        /// Layout: `route_history_buf[layer * top_k_routed + i]` = i-th routed expert id
        /// for the given layer in the most recent token.  Written per-token by a
        /// blit copy inside the TokenCommandBuffer; read by the CPU after commit.
        /// Size: n_moe_layers × top_k_routed × sizeof(u32).
        pub route_history_buf: PinnedBuffer,
        /// Phase 5A: per-slot final-norm output buffers used by `forward_tokens_batched_tcb`.
        /// After each token's final rmsnorm, x_norm_buf is blitted into slot[ki] so all
        /// K final norms survive until the single global TCB commits and LM heads are run.
        /// Size: max_batch_size × hidden × sizeof(f32).
        pub batch_x_norm_buf: Vec<PinnedBuffer>,
        /// path-to-125 Phase A1.1 — per-K residual stream buffers for the
        /// K-batched verify path (`forward_tokens_batched_parallel_k`).
        /// Token ki's running residual is restored to `arena.x_buf` at
        /// the top of each layer's Phase A and Phase C, and saved back
        /// after Phase C's MoE add_inplace. Size: max_batch_size × hidden × sizeof(f32).
        pub batch_x_buf: Vec<PinnedBuffer>,
        /// path-to-125 Phase A1.1 — per-K MoE output buffers for the
        /// K-batched verify path. Token ki's previous-layer MoE output
        /// is restored to `arena.ffn_out_buf` before phase1's
        /// add_inplace(x_buf, ffn_out_buf) residual at li > 0.
        /// Size: max_batch_size × hidden × sizeof(f32).
        pub batch_ffn_out_buf: Vec<PinnedBuffer>,
        /// path-to-125 Phase A1.1 — packed K queries for one K-batched
        /// MLA dispatch per layer. `arena.q` is blitted into
        /// `batch_q_packed[ki * n_heads * q_head_dim]` after each
        /// token's phase2. Size: max_batch_size × n_heads × q_head_dim × sizeof(f32).
        pub batch_q_packed: PinnedBuffer,
        /// path-to-125 Phase A1.1 — packed K outputs from one K-batched
        /// MLA dispatch per layer. Each token's slice is blitted out
        /// to `arena.attn_out` for the per-token o_proj.
        /// Size: max_batch_size × n_heads × v_head_dim × sizeof(f32).
        pub batch_attn_out_packed: PinnedBuffer,
        /// path-to-125 Phase A1.1 — device-scratch scores buffer for
        /// the K-batched MLA kernel. Size n_heads × max_batch_size ×
        /// max_seq × sizeof(f32). The kernel writes per-(head, kk, t)
        /// scores here so TG memory stays under the M3 Pro 32 KB / core
        /// budget at K=4 (see shaders/mla_decode_kernel_fc_kbatch.metal).
        pub batch_scores_scratch: PinnedBuffer,
        /// Phase 5C.2: f16 normed activation buffer — hidden × f16.
        /// Written by rmsnorm_f32_to_f16 when x_norm_dtype="f16" is set in the kernel
        /// profile. Used as the activation input to the LM head GEMV (gemv_f16_f16in),
        /// halving the hidden-size read bandwidth for the final vocab projection.
        /// The residual stream (x_buf) remains f32; this buffer does NOT cross layer
        /// boundaries. When x_norm_dtype="f32" (default) this buffer is allocated but
        /// never written; the code path routes through x_norm_buf instead.
        pub x_norm_f16_buf: PinnedBuffer,

        /// Path-to-90 lever 2 — Eagle4 GPU capture buffers, populated
        /// by mid-TCB blit + add_inplace at layers 2/13/25/26 when
        /// `eagle4_capture_active` is set. Lets the Wedge C single-
        /// TCB path produce captures without forcing per-layer
        /// commits (~4 ms / token saved at 26 layers × ~150 µs).
        /// All hidden × f32 except none — readers cast as needed.
        pub eagle4_h_low_buf: PinnedBuffer,
        pub eagle4_h_mid_buf: PinnedBuffer,
        pub eagle4_h_high_buf: PinnedBuffer,
        pub eagle4_x_norm_26_buf: PinnedBuffer,
        pub eagle4_h_shared_buf: PinnedBuffer,

        /// Cached sizes for bounds-checking at dispatch time.
        pub max_batch_size: usize,
        pub n_heads: usize,
        pub q_head_dim: usize,
        pub v_head_dim: usize,
        pub kv_lora_rank: usize,
        pub qk_rope_head_dim: usize,
        pub hidden: usize,
        pub max_seq: usize,
        pub n_routed_experts: usize,
        pub top_k_routed: usize,
        pub moe_intermediate: usize,
        pub shared_mid: usize,
        pub q_lora_rank: usize,
        pub kv_a_dim: usize,
        pub n_moe_layers: usize,
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
            top_k_routed: usize,
            moe_intermediate: usize,
            n_shared_experts: usize,
            ffn_intermediate: usize,
            q_lora_rank: usize,
            n_moe_layers: usize,
            max_batch_size: usize,
        ) -> Self {
            let q_head_dim = qk_nope_head_dim + qk_rope_head_dim;
            let kv_a_dim = kv_lora_rank + qk_rope_head_dim;
            let q_lora_sz = q_lora_rank.max(1);
            let top_k_sz = top_k_routed.max(1);
            let shared_mid = (n_shared_experts * moe_intermediate).max(1);
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
                moe_route_ids_buf: ctx.new_buffer(top_k_sz * std::mem::size_of::<u32>()),
                moe_route_weights_buf: ctx.new_buffer(top_k_sz * std::mem::size_of::<f32>()),
                shared_route_ids_buf: {
                    let buf = ctx.new_buffer(std::mem::size_of::<u32>());
                    MetalContext::write_buffer_bytes(&buf, bytemuck::cast_slice(&[0u32]));
                    buf
                },
                moe_routed_gate_out_buf: ctx.new_buffer(top_k_sz * moe_intermediate * std::mem::size_of::<f32>()),
                moe_routed_up_out_buf: ctx.new_buffer(top_k_sz * moe_intermediate * std::mem::size_of::<f32>()),
                moe_routed_act_buf: ctx.new_buffer(top_k_sz * moe_intermediate * std::mem::size_of::<f32>()),
                moe_routed_out_buf: ctx.new_buffer(top_k_sz * hidden * std::mem::size_of::<f32>()),
                moe_shared_gate_out_buf: ctx.new_buffer(shared_mid * std::mem::size_of::<f32>()),
                moe_shared_up_out_buf: ctx.new_buffer(shared_mid * std::mem::size_of::<f32>()),
                moe_shared_act_buf: ctx.new_buffer(shared_mid * std::mem::size_of::<f32>()),
                moe_shared_out_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                dense_gate_out_buf: ctx.new_buffer(ffn_intermediate * std::mem::size_of::<f32>()),
                dense_up_out_buf: ctx.new_buffer(ffn_intermediate * std::mem::size_of::<f32>()),
                dense_act_buf: ctx.new_buffer(ffn_intermediate * std::mem::size_of::<f32>()),
                q_lora_buf: ctx.new_buffer(q_lora_sz * std::mem::size_of::<f32>()),
                kv_a_out_buf: ctx.new_buffer(kv_a_dim * std::mem::size_of::<f32>()),
                q_lora_normed_buf: ctx.new_buffer(q_lora_sz * std::mem::size_of::<f32>()),
                c_kv_normed_buf: ctx.new_buffer(kv_lora_rank * std::mem::size_of::<f32>()),
                route_history_buf: ctx.new_buffer(
                    n_moe_layers.max(1) * top_k_sz * std::mem::size_of::<u32>()
                ),
                batch_x_norm_buf: (0..max_batch_size.max(1))
                    .map(|_| ctx.new_buffer(hidden * std::mem::size_of::<f32>()))
                    .collect(),
                batch_x_buf: (0..max_batch_size.max(1))
                    .map(|_| ctx.new_buffer(hidden * std::mem::size_of::<f32>()))
                    .collect(),
                batch_ffn_out_buf: (0..max_batch_size.max(1))
                    .map(|_| ctx.new_buffer(hidden * std::mem::size_of::<f32>()))
                    .collect(),
                batch_q_packed: ctx.new_buffer(
                    max_batch_size.max(1) * n_heads * q_head_dim * std::mem::size_of::<f32>()
                ),
                batch_attn_out_packed: ctx.new_buffer(
                    max_batch_size.max(1) * n_heads * v_head_dim * std::mem::size_of::<f32>()
                ),
                batch_scores_scratch: ctx.new_buffer(
                    n_heads * max_batch_size.max(1) * max_seq * std::mem::size_of::<f32>()
                ),
                x_norm_f16_buf: ctx.new_buffer(hidden * std::mem::size_of::<half::f16>()),
                eagle4_h_low_buf:     ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                eagle4_h_mid_buf:     ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                eagle4_h_high_buf:    ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                eagle4_x_norm_26_buf: ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                eagle4_h_shared_buf:  ctx.new_buffer(hidden * std::mem::size_of::<f32>()),
                max_batch_size: max_batch_size.max(1),
                n_heads,
                q_head_dim,
                v_head_dim,
                kv_lora_rank,
                qk_rope_head_dim,
                hidden,
                max_seq,
                n_routed_experts,
                top_k_routed,
                moe_intermediate,
                shared_mid,
                q_lora_rank,
                kv_a_dim,
                n_moe_layers,
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

        /// Read moe_shared_out_buf back to CPU. This holds the shared-
        /// expert's per-token contribution (post-down-proj, length =
        /// `hidden`) BEFORE it's summed into ffn_out_buf by the fused
        /// MoE kernel. Used by the path-to-90 Eagle4 GPU capture to
        /// extract h_shared from the production MoE forward without
        /// dispatching a separate shared-only kernel.
        pub fn read_moe_shared_out(&self, dst: &mut [f32]) {
            let ptr = self.moe_shared_out_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }

        /// Path-to-90 lever 2 — readers for the mid-TCB-blit-populated
        /// eagle4 capture buffers. Called by the Eagle4 decode loop
        /// after the single Wedge C TCB commits.
        pub fn read_eagle4_h_low(&self, dst: &mut [f32]) {
            let ptr = self.eagle4_h_low_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }
        pub fn read_eagle4_h_mid(&self, dst: &mut [f32]) {
            let ptr = self.eagle4_h_mid_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }
        pub fn read_eagle4_h_high(&self, dst: &mut [f32]) {
            let ptr = self.eagle4_h_high_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }
        pub fn read_eagle4_x_norm_26(&self, dst: &mut [f32]) {
            let ptr = self.eagle4_x_norm_26_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }
        pub fn read_eagle4_h_shared(&self, dst: &mut [f32]) {
            let ptr = self.eagle4_h_shared_buf.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, self.hidden) };
            dst.copy_from_slice(src);
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub struct DecodeArena {
    _priv: (),
}
