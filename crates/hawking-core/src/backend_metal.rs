//! Concrete Metal backend for the Phase-3.1 compute seam.
//!
//! macOS-only. Binds the platform-neutral traits in [`super`] (the
//! LANDED `backend/mod.rs`, commit 728ab6d) to the existing concrete
//! Metal types:
//!
//! - [`Backend::Buffer`] = [`crate::metal::PinnedBuffer`] (an alias for
//!   `::metal::Buffer`, an `Arc`-like refcounted GPU handle). Keeping
//!   `Buffer` concrete in 3.1 means the decode arena and the ~180 weight
//!   fields stay untouched and hash-identical.
//! - [`Backend::Recorder`]`<'a>` = [`MetalRecorder`]`<'a>`, a thin
//!   by-value wrapper over [`crate::metal::TokenCommandBuffer`]`<'a>`.
//!   The wrapper inherits the inner TCB's *Drop-auto-commit*
//!   (`metal/mod.rs` `impl Drop for TokenCommandBuffer`), so it
//!   deliberately has **no** `Drop` of its own; the only explicit commit
//!   point is [`CommandRecorder::commit_and_wait`], which consumes the
//!   recorder by value, preserving the single-command-buffer-per-token
//!   batching invariant.
//!
//! Every op-trait method is a **thin wrapper** that forwards `&mut
//! rec.tcb` to the corresponding existing `crate::kernels::*_tcb`
//! dispatcher *unchanged* — no kernel body moves into this file. Op
//! methods never commit.
//!
//! # Known seam/kernel impedance points (documented, not silently wrong)
//!
//! Four production kernels need state the LANDED trait surface does not
//! carry. Where a faithful thin forward is impossible, the method either
//! (a) uses the only trait-derivable interpretation, or (b) returns a
//! descriptive `Err` so a mis-call can never corrupt numerics. See the
//! per-method docs and the stream's `risks` for the exact trait tweaks
//! recommended to the seam owner:
//!
//! 1. **GEMV byte-offset addressing.** The Q4K/Q6K/Q3K/W4A8 kernels
//!    address weights as `(model_buf, w_offset, w_byte_size)` — a slice
//!    of one giant pinned mmap. [`BackendGemv::gemv`] only gives
//!    `weight: &Buffer`, so the quantized arms here use `w_offset = 0`,
//!    `w_byte_size = weight.length()` (i.e. `weight` is treated as a
//!    standalone per-tensor buffer). F16/F32 map exactly.
//! 2. **GEMV predec / fast side-tables.** `gemv_proj!` consults
//!    pre-decoded-scale and fast-layout sidecar tables that are not on
//!    [`GemvSpec`]; `spec.predec` / `spec.weight == Q4kFast` therefore
//!    fall through to the bit-identical base kernel (the quality levers
//!    stay OFF by default).
//! 3. **Attention cache offsets + scale.** `mha_decode_f32_tcb` wants
//!    per-layer `k_off_bytes`/`v_off_bytes` and derives its own softmax
//!    scale; [`BackendAttention::attention`] supplies neither offset and
//!    passes a `scale`. The MHA arm forwards `0`/`0` (caches must be
//!    pre-sliced) and drops `scale`; the MLA arm returns `Err`.
//! 4. **MoE block scratch + shared experts.**
//!    `encode_moe_block_batched_indexed_tcb_with_scratch` needs nine
//!    arena scratch buffers, shared-expert routing, and schedule
//!    strings absent from [`BackendMoe::moe_block`]; that method returns
//!    a descriptive `Err`. `moe_topk_gate` maps cleanly.

#![cfg(target_os = "macos")]

use crate::kernels;
use crate::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use crate::{Error, Result};

use super::{
    AttentionKind, Backend, BackendAttention, BackendElementwise, BackendEmbed, BackendGemv, BackendKvCache, BackendMoe, BackendNorm, BackendQuant, BackendRope, BackendSample, CommandRecorder,
    GemvSpec, Op, QuantScheme, RopeLayout, WeightKind,
};

/// Concrete Metal [`Backend`].
///
/// Owns a [`MetalContext`] by value. `MetalContext` is `#[derive(Clone)]`
/// and `Arc`-backed, so cloning is cheap and `recorder()` can borrow the
/// owned context for the lifetime of a token's forward pass.
pub struct MetalBackend {
    ctx: MetalContext,
}

impl MetalBackend {
    /// Wrap an existing [`MetalContext`] as a backend. Cheap: the context
    /// is `Arc`-backed, so this shares the device/queue/library/pipeline
    /// cache with every other holder of the same context.
    #[inline]
    pub fn new(ctx: MetalContext) -> Self {
        Self { ctx }
    }

    /// Borrow the underlying [`MetalContext`].
    #[inline]
    pub fn context(&self) -> &MetalContext {
        &self.ctx
    }
}

/// Per-token command recorder: a by-value wrapper over
/// [`TokenCommandBuffer`].
///
/// Op-trait methods borrow this `&mut` and forward `&mut self.tcb` to the
/// raw `kernels::*_tcb` dispatchers. The wrapper has **no** `Drop`: the
/// inner `TokenCommandBuffer`'s own `Drop` auto-commits any not-yet
/// committed command buffer, so dropping a `MetalRecorder` (e.g. on an
/// early-return error path) commits exactly as the pre-seam code did.
/// The explicit commit path is [`CommandRecorder::commit_and_wait`],
/// which consumes `self` and forwards to
/// `TokenCommandBuffer::commit_and_wait`.
pub struct MetalRecorder<'a> {
    /// The single per-token command buffer. `pub(crate)` so sibling
    /// modules that migrate call sites can reach it during the routing
    /// phase; op methods here forward `&mut self.tcb`.
    pub(crate) tcb: TokenCommandBuffer<'a>,
}

impl<'a> CommandRecorder for MetalRecorder<'a> {
    type Buffer = PinnedBuffer;

    #[inline]
    fn commit_and_wait(self) -> Result<()> {
        // The ONLY commit point. Consumes self -> a forward pass commits
        // exactly once at its tail. `TokenCommandBuffer::commit_and_wait`
        // takes `mut self` and blocks until the GPU finishes.
        self.tcb.commit_and_wait()
    }

    #[inline]
    fn read_u32(&mut self, buf: &Self::Buffer, index: usize) -> Result<u32> {
        // Host-visible (StorageModeShared) readback of the argmax-sampled
        // token id. Mirrors the raw `token_buf.contents() as *const u32`
        // pattern at qwen_dense.rs:4632/4721/4743/4803. The caller is
        // responsible for having committed (and waited on) the work that
        // wrote `buf` before calling this — exactly as the pre-seam code
        // reads the id only after `tcb.commit_and_wait()`.
        let ptr = buf.contents() as *const u32;
        if ptr.is_null() {
            return Err(Error::Metal("MetalRecorder::read_u32: buffer contents() returned null".into()));
        }
        // SAFETY: `buf` is a host-visible shared buffer; `index` must be
        // within its element count (callers read index 0 of token_buf).
        let val = unsafe { *ptr.add(index) };
        Ok(val)
    }

    #[inline]
    fn begin_concurrent_group(&mut self) -> Result<()> {
        self.tcb.begin_concurrent_group()
    }

    #[inline]
    fn end_concurrent_group(&mut self) -> Result<()> {
        self.tcb.end_concurrent_group()
    }
}

impl Backend for MetalBackend {
    type Buffer = PinnedBuffer;
    type Recorder<'a>
        = MetalRecorder<'a>
    where
        Self: 'a;

    #[inline]
    fn recorder(&self) -> Self::Recorder<'_> {
        // Borrows `self.ctx` for the lifetime of the returned recorder,
        // satisfying the GAT `where Self: 'a` bound. `TokenCommandBuffer`
        // opens a fresh command buffer on the context's queue.
        MetalRecorder { tcb: TokenCommandBuffer::new(&self.ctx) }
    }

    #[inline]
    fn supports(&self, _op: Op) -> bool {
        // Fully-featured backend: every logical verb is implemented.
        // (Per-verb caveats for MoE/MLA are surfaced as `Err` at call
        // time, not as a `false` here, because those paths exist but need
        // a richer trait surface — see the module-level docs.)
        true
    }
}

impl BackendGemv for MetalBackend {
    fn gemv(&self, rec: &mut Self::Recorder<'_>, spec: &GemvSpec, weight: &Self::Buffer, x: &Self::Buffer, x_int8: &Self::Buffer, x_scales: &Self::Buffer, out: &Self::Buffer) -> Result<()> {
        let tcb = &mut rec.tcb;
        let rows = spec.rows;
        let cols = spec.cols;

        match spec.weight {
            // ── Q4_K family ──────────────────────────────────────────
            // SEAM GAP #1/#2: the trait gives one `weight` buffer with no
            // offset; the production kernels slice the giant mmap by
            // (offset, byte_size). Treat `weight` as a standalone
            // per-tensor buffer: offset 0, byte_size = weight.length().
            // The predec / fast / f16-scale variants in `gemv_proj!`
            // require sidecar tables not present on `GemvSpec`, so
            // `spec.predec` / `spec.predec_f16_scales` currently fall
            // through to the bit-identical base kernel (levers stay OFF).
            WeightKind::Q4K | WeightKind::Q4kFast => {
                let w_off = 0usize;
                let w_bytes = weight.length() as usize;
                if spec.w4a8 {
                    // W4A8: per-block int8 activation x Q4_K weight.
                    kernels::gemm_q4_k_a8_v3_8r_pinned_tcb(tcb, weight, w_off, w_bytes, rows, cols, x_int8, x_scales, out)
                } else {
                    kernels::gemv_q4_k_m_v3_8r_pinned_tcb(tcb, weight, w_off, w_bytes, rows, cols, x, out)
                }
            }
            // ── Q6_K ─────────────────────────────────────────────────
            WeightKind::Q6K => {
                let w_bytes = weight.length() as usize;
                kernels::gemv_q6_k_pinned_tcb(tcb, weight, 0, w_bytes, rows, cols, x, out)
            }
            // ── Q3_K ─────────────────────────────────────────────────
            WeightKind::Q3K => {
                let w_bytes = weight.length() as usize;
                kernels::gemv_q3_k_pinned_tcb(tcb, weight, 0, w_bytes, rows, cols, x, out)
            }
            // ── f16 weights (no offset — maps to the trait cleanly) ──
            WeightKind::F16 => kernels::gemv_f16_metal_buf_tcb(tcb, weight, rows, cols, x, out),
            // ── f32 weights (attention-side projections) ─────────────
            WeightKind::F32 => kernels::gemv_f32_attn_pinned_buf_tcb(tcb, weight, rows, cols, x, out),
        }
    }
}

impl BackendNorm for MetalBackend {
    #[inline]
    fn rmsnorm(&self, rec: &mut Self::Recorder<'_>, x: &Self::Buffer, weight: &Self::Buffer, out: &Self::Buffer, eps: f32, hidden: usize) -> Result<()> {
        // NB kernel arg order: (tcb, x, weight, eps, hidden, out).
        kernels::rmsnorm_metal_buf_tcb(&mut rec.tcb, x, weight, eps, hidden, out)
    }

    #[inline]
    fn add_rmsnorm(&self, rec: &mut Self::Recorder<'_>, x: &Self::Buffer, attn_out: &Self::Buffer, weight: &Self::Buffer, x_norm: &Self::Buffer, eps: f32, hidden: usize) -> Result<()> {
        // Fused residual-add + rmsnorm. `x` is the f32 residual
        // accumulator and STAYS f32 (this kernel never f16's it).
        kernels::add_rmsnorm_fused_tcb(&mut rec.tcb, x, attn_out, weight, x_norm, eps, hidden)
    }

    #[inline]
    fn add_rmsnorm_q8(
        &self,
        rec: &mut Self::Recorder<'_>,
        x: &Self::Buffer,
        attn_out: &Self::Buffer,
        weight: &Self::Buffer,
        x_norm: &Self::Buffer,
        x_norm_int8: &Self::Buffer,
        x_norm_scales: &Self::Buffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()> {
        kernels::add_rmsnorm_fused_q8_tcb(&mut rec.tcb, x, attn_out, weight, x_norm, x_norm_int8, x_norm_scales, eps, hidden)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn add_rmsnorm_q8_scaled(
        &self,
        rec: &mut Self::Recorder<'_>,
        x: &Self::Buffer,
        attn_out: &Self::Buffer,
        weight: &Self::Buffer,
        x_norm: &Self::Buffer,
        x_norm_int8: &Self::Buffer,
        x_norm_scales: &Self::Buffer,
        s_buf: &Self::Buffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()> {
        // AWQ decode path: per-channel smoothing `s_buf` applied before
        // the int8 quant. kernels/mod.rs:5363.
        kernels::add_rmsnorm_fused_q8_scaled_tcb(&mut rec.tcb, x, attn_out, weight, x_norm, x_norm_int8, x_norm_scales, s_buf, eps, hidden)
    }
}

impl BackendElementwise for MetalBackend {
    #[inline]
    fn add(&self, rec: &mut Self::Recorder<'_>, a: &Self::Buffer, b: &Self::Buffer, n: usize) -> Result<()> {
        // In-place f32 residual add `a[i] += b[i]`. `a` stays f32.
        kernels::add_inplace_metal_tcb(&mut rec.tcb, a, b, n)
    }

    #[inline]
    fn silu_mul(&self, rec: &mut Self::Recorder<'_>, gate: &Self::Buffer, up: &Self::Buffer, out: &Self::Buffer, n: usize) -> Result<()> {
        kernels::silu_mul_tcb(&mut rec.tcb, gate, up, out, n)
    }
}

impl BackendRope for MetalBackend {
    #[inline]
    fn rope(&self, rec: &mut Self::Recorder<'_>, buf: &Self::Buffer, layout: RopeLayout, pos: u32, base: f32) -> Result<()> {
        let tcb = &mut rec.tcb;
        match layout {
            // Full-head rope: rope_q_f32_inplace with nope_dim = 0 and
            // rope_dim = head_dim rotates the entire head (matches
            // qwen_dense.rs:4119 for Q and :4129 for K).
            RopeLayout::Full { n_heads, head_dim } => {
                kernels::rope_q_f32_inplace_tcb(tcb, buf, n_heads, /* q_head_dim     */ head_dim, /* qk_nope_head_dim */ 0, /* qk_rope_head_dim */ head_dim, pos, base)
            }
            // Partial (MLA nope/rope split): rotate only the trailing
            // `rope_dim` slice starting at `offset` floats into `buf`.
            RopeLayout::Partial { offset, rope_dim } => kernels::rope_slice_f32_inplace_tcb(tcb, buf, offset, rope_dim, pos, base),
        }
    }
}

impl BackendAttention for MetalBackend {
    fn attention(
        &self,
        rec: &mut Self::Recorder<'_>,
        kind: AttentionKind,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        out: &Self::Buffer,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        _scale: f32,
    ) -> Result<()> {
        match kind {
            // SEAM GAP #3: mha_decode_f32_tcb wants per-layer
            // k_off_bytes/v_off_bytes and derives its own softmax scale
            // (1/sqrt(head_dim)), ignoring `_scale`. The trait passes
            // neither offset, so we forward 0/0 — correct only when the
            // caller hands us the per-layer KV-cache *slice* (a pre-
            // offset buffer view), which is how the routing stream is
            // expected to bind these. kernels/mod.rs:5779.
            AttentionKind::Mha => kernels::mha_decode_f32_tcb(&mut rec.tcb, q, k_cache, /* k_off_bytes */ 0, v_cache, /* v_off_bytes */ 0, out, seq_len, head_dim, n_heads, n_kv_heads),
            // MLA decode fuses the output projection and reads extra
            // latent/rope buffers; deepseek_v2 drives it through a
            // distinct, wider entry point that does not match this
            // generic surface. Surface a clear error rather than a wrong
            // dispatch until the trait carries the MLA latent buffers.
            AttentionKind::Mla => Err(Error::Metal(
                "MetalBackend::attention: AttentionKind::Mla is not expressible through the \
                 generic attention() surface (MLA fuses the output projection and needs the \
                 latent/rope arena buffers); drive MLA via deepseek_v2's direct mla_decode path \
                 until the trait carries the latent buffers"
                    .into(),
            )),
        }
    }
}

impl BackendKvCache for MetalBackend {
    fn kv_append(&self, rec: &mut Self::Recorder<'_>, k_src: &Self::Buffer, v_src: &Self::Buffer, k_cache: &Self::Buffer, v_cache: &Self::Buffer, seq_slot: usize, kv_dim: usize) -> Result<()> {
        // The dense KV append is two contiguous copies of the current
        // token's K and V into the cache slot — exactly how qwen_dense
        // does it (qwen_dense.rs:4143/4151 use two memcpy_f32_off_tcb,
        // NOT the MLA-specific kv_append_f32_tcb, whose latent-cache
        // signature (kv_lora_rank, qk_rope_head_dim, two dst buffers)
        // does not match this generic surface). The caller passes a
        // k_cache/v_cache buffer already based at the per-layer slice, so
        // the destination offset is `seq_slot * kv_dim` floats and the
        // source offset is 0. memcpy_f32_off_tcb arg order is
        // (tcb, src, dst, src_off, dst_off, n) — kernels/mod.rs:5842.
        let dst_off = seq_slot * kv_dim;
        kernels::memcpy_f32_off_tcb(&mut rec.tcb, k_src, k_cache, 0, dst_off, kv_dim)?;
        kernels::memcpy_f32_off_tcb(&mut rec.tcb, v_src, v_cache, 0, dst_off, kv_dim)?;
        Ok(())
    }

    #[inline]
    fn memcpy(&self, rec: &mut Self::Recorder<'_>, src: &Self::Buffer, dst: &Self::Buffer, src_off: usize, dst_off: usize, n: usize) -> Result<()> {
        // Direct match to the kernel: (tcb, src, dst, src_off, dst_off, n).
        kernels::memcpy_f32_off_tcb(&mut rec.tcb, src, dst, src_off, dst_off, n)
    }
}

impl BackendQuant for MetalBackend {
    fn quantize(&self, rec: &mut Self::Recorder<'_>, scheme: QuantScheme, x: &Self::Buffer, x_int8: &Self::Buffer, scales: &Self::Buffer, n: usize) -> Result<()> {
        let tcb = &mut rec.tcb;
        match scheme {
            QuantScheme::PerBlock => {
                // (tcb, x, x_int8, scales, n) — kernels/mod.rs:1917.
                kernels::quantize_f32_to_int8_per_block_tcb(tcb, x, x_int8, scales, n)
            }
            // PerBlockScaled is the AWQ smoothing-then-quant variant; its
            // kernel takes the per-channel smoothing vector `s_buf`. That
            // buffer is not on the BackendQuant::quantize surface, so this
            // scheme is only reachable via the fused
            // `add_rmsnorm_q8_scaled` path (which threads s_buf). Surface
            // a clear error if a caller asks for it standalone here.
            QuantScheme::PerBlockScaled => Err(Error::Metal(
                "MetalBackend::quantize(PerBlockScaled): the scaled per-block quant needs the \
                 per-channel AWQ smoothing vector `s_buf`, which is not on the quantize() \
                 surface; use BackendNorm::add_rmsnorm_q8_scaled (which threads s_buf) for the \
                 AWQ decode path"
                    .into(),
            )),
            QuantScheme::PerChannel => {
                // NB kernel arg order: (tcb, x, scales, x_int8, n) — the
                // scales buffer comes BEFORE x_int8. kernels/mod.rs:2040.
                kernels::quantize_f32_to_int8_per_channel_tcb(tcb, x, scales, x_int8, n)
            }
        }
    }
}

impl BackendEmbed for MetalBackend {
    #[inline]
    fn embed(&self, rec: &mut Self::Recorder<'_>, embed_table: &Self::Buffer, token: u32, out: &Self::Buffer, hidden: usize) -> Result<()> {
        // NB kernel arg order: (tcb, embed_buf, token, hidden, x_buf).
        kernels::embed_lookup_metal_f32_tcb(&mut rec.tcb, embed_table, token, hidden, out)
    }
}

impl BackendSample for MetalBackend {
    #[inline]
    fn sample_argmax(&self, rec: &mut Self::Recorder<'_>, logits: &Self::Buffer, token_out: &Self::Buffer, vocab: usize) -> Result<()> {
        kernels::sample_argmax_f32_tcb(&mut rec.tcb, logits, token_out, vocab)
    }
}

impl BackendMoe for MetalBackend {
    #[inline]
    fn moe_topk_gate(&self, rec: &mut Self::Recorder<'_>, logits: &Self::Buffer, route_ids: &Self::Buffer, route_weights: &Self::Buffer, n_experts: usize, top_k: usize) -> Result<()> {
        kernels::moe_topk_gate_tcb(&mut rec.tcb, logits, route_ids, route_weights, n_experts, top_k)
    }

    fn moe_block(
        &self,
        _rec: &mut Self::Recorder<'_>,
        _model: &Self::Buffer,
        _route_ids: &Self::Buffer,
        _route_weights: &Self::Buffer,
        _x: &Self::Buffer,
        _out: &Self::Buffer,
        _routed_gate_offset: usize,
        _routed_up_offset: usize,
        _routed_down_offset: usize,
        _hidden: usize,
        _routed_mid: usize,
        _routes: usize,
    ) -> Result<()> {
        // SEAM GAP #4: the production routed-expert block
        // (`encode_moe_block_batched_indexed_tcb_with_scratch`,
        // kernels/mod.rs:6363) requires, beyond what this method is
        // handed: the shared-expert route-id buffer + shared
        // gate/up/down byte offsets, `shared_mid`, three
        // schedule/kernel-name strings (q4k_schedule, routed_down_kernel,
        // shared_down_kernel), and NINE arena scratch buffers
        // (routed_gate_out, routed_up_out, routed_act, routed_out,
        // shared_gate_out, shared_up_out, shared_act, shared_out). None of
        // those are on the BackendMoe::moe_block surface, so a faithful
        // thin forward is impossible. Returning a descriptive error keeps
        // the build green and guarantees no mis-dispatch; route MoE
        // through deepseek_v2's direct call (deepseek_v2.rs:2439) until
        // the trait carries a scratch/shared-expert descriptor.
        Err(Error::Metal(
            "MetalBackend::moe_block: the production routed-expert kernel \
             (encode_moe_block_batched_indexed_tcb_with_scratch) needs shared-expert routing, \
             shared offsets/shared_mid, three schedule strings, and nine arena scratch buffers \
             that are not present on BackendMoe::moe_block; drive MoE via deepseek_v2's direct \
             path until the trait is extended with a scratch/shared-expert descriptor"
                .into(),
        ))
    }
}
