//! Compute-backend seam (Phase 3.1).
//!
//! Platform-NEUTRAL trait + enum definitions for the per-token compute
//! seam. This module is `#[cfg]`-free and references no Metal symbol: the
//! only external dependency is [`crate::Result`] (the error type carries no
//! platform gate). It therefore compiles on every target, including hosts
//! that cannot run the Metal path.
//!
//! The concrete `MetalBackend` (binding `type Buffer = metal::PinnedBuffer`
//! and `type Recorder<'a> = metal::TokenCommandBuffer<'a>`) lives in the
//! macOS-gated sibling `backend/metal.rs` and is wired up by a separate
//! stream. Keeping `Buffer` a concrete type in 3.1 means the decode arena
//! and the ~180 weight fields stay unchanged; genericizing the model over
//! `<B: Backend>` is deferred to a later phase, when a second backend
//! (CPU) forces it.
//!
//! # Design contract
//!
//! - **One supertrait.** [`Backend`] bundles the ~11 op-traits
//!   ([`BackendGemv`], [`BackendNorm`], [`BackendElementwise`],
//!   [`BackendRope`], [`BackendAttention`], [`BackendKvCache`],
//!   [`BackendQuant`], [`BackendEmbed`], [`BackendSample`],
//!   [`BackendMoe`]) plus the [`CommandRecorder`] lifecycle, behind two
//!   associated types: [`Backend::Buffer`] and the GAT
//!   [`Backend::Recorder`].
//!
//! - **One `gemv` verb.** The ~31 distinct `gemv*`/`gemm*` GPU entry
//!   points collapse into the single [`BackendGemv::gemv`] method. The
//!   variant explosion (weight dtype, pre-decoded scales, W4A8 activation
//!   quant, paired/batched geometry, custom fast layout) is data, carried
//!   in [`GemvSpec`] + [`WeightKind`], and resolved inside the impl body
//!   (the existing `gemv_proj!` dispatch ladder) — never as 25 trait
//!   methods.
//!
//! - **Single command buffer per token.** Every op-trait method records
//!   into the *one* per-token recorder it is handed by `&mut`. Op methods
//!   MUST NOT commit; only [`CommandRecorder::commit_and_wait`] commits,
//!   and it consumes the recorder by value, so a forward pass commits
//!   exactly once at its tail. A naive per-op commit shatters this
//!   batching and tanks decode throughput.
//!
//! - **Fused kernels stay fused.** Cross-op fusions surface as their own
//!   verbs ([`BackendNorm::add_rmsnorm`], [`BackendNorm::add_rmsnorm_q8`])
//!   rather than being decomposed into `add` + `rmsnorm`. Decomposition
//!   would change numerics and break bit-identity.
//!
//! All op-trait methods return [`crate::Result`] and take `&mut
//! Self::Recorder<'_>` as their first argument.

use crate::Result;

/// Concrete macOS Metal backend implementing every op trait below.
/// Gated so the platform-neutral trait defs in this module continue to
/// compile on every target while the Metal-only impl is built only on
/// macOS.
///
/// `pub` (was private in 3.1) so the Phase-3.2 [`router`] sibling and the
/// decode call site (`crate::model::qwen_dense`) can name `MetalBackend` /
/// `MetalRecorder`. The module stays macOS-gated, so non-macOS targets
/// still never see a Metal symbol.
#[cfg(target_os = "macos")]
pub mod metal;

/// Phase 3.2 per-op compute router with CPU fallback (macOS-only; it names
/// the concrete Metal backend). `Router { primary: MetalBackend, forced:
/// Option<Op> }` routes each fallback-capable op (rmsnorm, rope) to the
/// primary unless `DISMANTLE_FORCE_CPU_OP` forces it onto the CPU
/// primitive. DEFAULT-UNSET ⇒ pure Metal ⇒ golden hash unchanged.
#[cfg(target_os = "macos")]
pub mod router;

/// The set of primitive compute verbs a [`Backend`] may implement.
///
/// Used by [`Backend::supports`] for capability queries (e.g. a CPU
/// backend that has not yet implemented `Moe` reports `false` for it).
/// These are the *logical* verbs, not the physical kernel count: the
/// dozens of GEMV kernels all live behind the single [`Op::Gemv`] verb,
/// and the fused norm kernels behind [`Op::RmsNorm`] / [`Op::Add`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// RMS normalization, including the fused add+rmsnorm variants.
    RmsNorm,
    /// Rotary position embedding (full or partial-dimension slice).
    Rope,
    /// In-place elementwise residual add (`x[i] += y[i]`).
    Add,
    /// SwiGLU activation: `out[i] = silu(gate[i]) * up[i]`.
    SiluMul,
    /// Matrix-vector product. The single verb behind every weight dtype,
    /// quantization scheme, and dispatch geometry (see [`GemvSpec`]).
    Gemv,
    /// Single-token attention (MHA or MLA; see [`AttentionKind`]).
    Attention,
    /// Append the current token's K/V into the KV cache.
    KvAppend,
    /// Quantize an f32 activation to int8 (see [`QuantScheme`]).
    Quantize,
    /// Embedding-table lookup for a single token id.
    Embed,
    /// Argmax sampling over a logits vector.
    Sample,
    /// Mixture-of-experts routed block (top-k gate + expert GEMVs).
    Moe,
}

/// Weight storage class a [`BackendGemv::gemv`] call dispatches on.
///
/// Mirrors the GGUF dtype arms (plus the custom fast sidecar layout) that
/// the production GEMV ladder switches over. Kept independent of
/// `gguf::GgmlType` so this module stays platform- and format-neutral; the
/// concrete impl maps the model's dtype onto this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightKind {
    /// Q4_K_M block-quantized (the tuned Qwen2.5-3B path).
    Q4K,
    /// Q6_K block-quantized.
    Q6K,
    /// Q3_K block-quantized.
    Q3K,
    /// f16 weights (dequantized-once fallback, or natively f16 tensors).
    F16,
    /// f32 weights (e.g. attention-side projections kept in f32).
    F32,
    /// Custom sub-block-contiguous "fast" Q4_K layout pinned from a
    /// sidecar (160-byte blocks; same dispatch geometry as Q4_K, distinct
    /// memory layout).
    Q4kFast,
}

/// Fully describes one matrix-vector dispatch.
///
/// Carries the problem shape plus the boolean dispatch axes the GEMV
/// ladder selects on. The impl body — not the trait — turns this into a
/// concrete kernel choice, reproducing the existing `gemv_proj!`
/// resolution order (pre-decoded f16 scales → pre-decoded f32 scales →
/// fast layout → default → W4A8 → dtype fallbacks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GemvSpec {
    /// Weight storage class (selects the kernel family).
    pub weight: WeightKind,
    /// Output rows (length of the result vector).
    pub rows: usize,
    /// Input columns (length of the activation vector).
    pub cols: usize,
    /// W4A8: the activation is supplied pre-quantized to per-block int8
    /// (4× less activation bandwidth). Only meaningful for
    /// [`WeightKind::Q4K`].
    pub w4a8: bool,
    /// Read pre-decoded sub-block scales from a pinned table instead of
    /// re-decoding inline every dispatch.
    pub predec: bool,
    /// When [`Self::predec`] is set, read half-width (f16) `(ds, dm)`
    /// scale pairs rather than f32 pairs (~17% less scale traffic; not
    /// bit-identical — a quality trade behind an opt-in lever).
    pub predec_f16_scales: bool,
    /// Paired projection: two output rows-groups share one activation
    /// load (the gate+up fusion).
    pub paired: bool,
    /// Batched (multi-row / GEMM-shaped) geometry rather than the
    /// single-vector path.
    pub batched: bool,
}

/// Rope dimensionality layout for a [`BackendRope::rope`] call.
///
/// `Full` rotates the entire head; `Partial` rotates only the trailing
/// `rope_dim` of a head whose total width is `head_dim` (the MLA
/// nope/rope split), starting at `offset` floats into the buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Rotate the full head dimension across all heads.
    Full {
        /// Number of attention heads.
        n_heads: usize,
        /// Per-head dimension (also the rotation width).
        head_dim: usize,
    },
    /// Rotate only a trailing rope slice of one head (partial-rope MLA).
    Partial {
        /// Float offset into the buffer where the slice begins.
        offset: usize,
        /// Width of the rotated slice.
        rope_dim: usize,
    },
}

/// Attention variant for a [`BackendAttention::attention`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionKind {
    /// Standard (grouped) multi-head attention decode.
    Mha,
    /// Multi-head latent attention decode (DeepSeek-V2 style), fused with
    /// the output projection.
    Mla,
}

/// Activation int8 quantization scheme for [`BackendQuant::quantize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// One scale per 256-element block.
    PerBlock,
    /// Per-block with an additional global rescale.
    PerBlockScaled,
    /// One scale per channel.
    PerChannel,
}

/// Lifecycle of the per-token command recorder (the seam over the
/// concrete `TokenCommandBuffer`).
///
/// A recorder accumulates every op of a single token's forward pass into
/// one command buffer. Op-trait methods borrow it `&mut` and append work;
/// they never commit. The pass commits exactly once via
/// [`Self::commit_and_wait`], which consumes the recorder.
///
/// Concurrent-group and `read_u32` are defined here so call sites depend
/// only on the seam: `read_u32` hides the raw GPU-buffer readback used to
/// pull the sampled token id back to the host, and the concurrent-group
/// pair is a no-op on backends that lack the concept.
pub trait CommandRecorder: Sized {
    /// The buffer type recorded against. Bound to the same concrete type
    /// as [`Backend::Buffer`] by any backend pairing the two.
    type Buffer;

    /// Commit the accumulated command buffer and block until the GPU has
    /// finished. Consumes the recorder so that a forward pass commits
    /// exactly once, at its tail. Op methods MUST NOT commit; this is the
    /// only commit point.
    fn commit_and_wait(self) -> Result<()>;

    /// Read a single `u32` element at `index` from `buf`.
    ///
    /// Hides the raw host-visible readback (`buffer.contents() as *const
    /// u32`) used to pull the argmax-sampled token id back to the host.
    /// Backends with a host-visible address space implement this as a
    /// direct load after the relevant work has completed; the call site
    /// no longer touches raw pointers.
    fn read_u32(&mut self, buf: &Self::Buffer, index: usize) -> Result<u32>;

    /// Open a concurrent dispatch group: subsequent ops in the group may
    /// be scheduled to overlap on the device. The caller asserts the
    /// group's dispatches share no overlapping read-write/write-write
    /// buffer range (e.g. the Q/K/V projection triple writing disjoint
    /// outputs).
    ///
    /// No-op on backends without a concurrent-encoder concept.
    fn begin_concurrent_group(&mut self) -> Result<()> {
        Ok(())
    }

    /// Close the active concurrent group. No-op if none is open, and on
    /// backends without the concept.
    fn end_concurrent_group(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Matrix-vector products — the single GEMV verb.
///
/// All weight dtypes, activation-quant schemes, and dispatch geometries
/// funnel through [`Self::gemv`]; the per-call variant is described by
/// [`GemvSpec`]. The impl body owns the kernel-selection ladder.
pub trait BackendGemv: Backend {
    /// Compute `out = W · x` for the projection described by `spec`.
    ///
    /// When `spec.w4a8` is set, the activation is taken pre-quantized; the
    /// `x_int8` / `x_scales` buffers supply the per-block int8
    /// representation and `x` is ignored. Otherwise the f32 `x` is used
    /// and the int8 inputs may be ignored.
    fn gemv(
        &self,
        rec: &mut Self::Recorder<'_>,
        spec: &GemvSpec,
        weight: &Self::Buffer,
        x: &Self::Buffer,
        x_int8: &Self::Buffer,
        x_scales: &Self::Buffer,
        out: &Self::Buffer,
    ) -> Result<()>;
}

/// RMS normalization, including the fused add+norm verbs.
///
/// The fused variants are first-class verbs, not `add` followed by
/// `rmsnorm`: decomposing them changes numerics and breaks bit-identity.
pub trait BackendNorm: Backend {
    /// `out = rmsnorm(x, weight, eps)` over `hidden` elements.
    fn rmsnorm(
        &self,
        rec: &mut Self::Recorder<'_>,
        x: &Self::Buffer,
        weight: &Self::Buffer,
        out: &Self::Buffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()>;

    /// Fused residual-add then RMS norm:
    /// `x[i] += attn_out[i]; x_norm = rmsnorm(x, weight, eps)`.
    ///
    /// The residual accumulator `x` MUST remain f32 — an f16 residual
    /// corrupts the logits after many layers. The norm output `x_norm` is
    /// written separately so the updated f32 residual is preserved for the
    /// next layer.
    fn add_rmsnorm(
        &self,
        rec: &mut Self::Recorder<'_>,
        x: &Self::Buffer,
        attn_out: &Self::Buffer,
        weight: &Self::Buffer,
        x_norm: &Self::Buffer,
        eps: f32,
        hidden: usize,
    ) -> Result<()>;

    /// As [`Self::add_rmsnorm`], additionally emitting the per-block int8
    /// quantization of the norm output (`x_norm_int8` / `x_norm_scales`)
    /// for an immediately-following W4A8 GEMV. The f32 residual `x` and
    /// f32 `x_norm` are still produced; only the extra int8 pair is added.
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
    ) -> Result<()>;

    /// As [`Self::add_rmsnorm_q8`], with an additional per-channel AWQ
    /// smoothing vector `s_buf` applied before the int8 quantization.
    /// Mirrors the production `add_rmsnorm_fused_q8_scaled_tcb` used on the
    /// AWQ decode path (qwen_dense `awq_active`). The f32 residual `x`
    /// stays f32; only the int8 pair is additionally scaled by `s_buf`.
    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<()>;
}

/// Elementwise residual-stream ops.
pub trait BackendElementwise: Backend {
    /// In-place residual add: `a[i] += b[i]` over `n` elements. `a` is the
    /// f32 residual accumulator and stays f32.
    fn add(
        &self,
        rec: &mut Self::Recorder<'_>,
        a: &Self::Buffer,
        b: &Self::Buffer,
        n: usize,
    ) -> Result<()>;

    /// SwiGLU: `out[i] = silu(gate[i]) * up[i]` over `n` elements.
    fn silu_mul(
        &self,
        rec: &mut Self::Recorder<'_>,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &Self::Buffer,
        n: usize,
    ) -> Result<()>;
}

/// Rotary position embedding (in place).
pub trait BackendRope: Backend {
    /// Rotate `buf` in place for position `pos` with the given
    /// [`RopeLayout`] and rope base (`theta`).
    fn rope(
        &self,
        rec: &mut Self::Recorder<'_>,
        buf: &Self::Buffer,
        layout: RopeLayout,
        pos: u32,
        base: f32,
    ) -> Result<()>;
}

/// Single-token attention decode.
pub trait BackendAttention: Backend {
    /// Decode-step attention over a cache of `seq_len` positions.
    ///
    /// `kind` selects MHA vs MLA. For MLA the call additionally fuses the
    /// output projection (the impl reads the extra latent/rope buffers it
    /// needs from the decode arena bound to the recorder). `q`, the K/V
    /// caches, and `out` are the common surface; `scale` is the softmax
    /// scale.
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
        scale: f32,
    ) -> Result<()>;
}

/// KV-cache writes and intra-buffer copies.
pub trait BackendKvCache: Backend {
    /// Append the current token's K/V into the cache at `seq_slot`.
    fn kv_append(
        &self,
        rec: &mut Self::Recorder<'_>,
        k_src: &Self::Buffer,
        v_src: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        seq_slot: usize,
        kv_dim: usize,
    ) -> Result<()>;

    /// Copy `n` f32 elements from `src[src_off..]` to `dst[dst_off..]`
    /// (used to bridge a CPU-side prefill prefix into the GPU arena).
    fn memcpy(
        &self,
        rec: &mut Self::Recorder<'_>,
        src: &Self::Buffer,
        dst: &Self::Buffer,
        src_off: usize,
        dst_off: usize,
        n: usize,
    ) -> Result<()>;
}

/// Activation quantization (f32 → int8) for the W4A8 GEMV path.
pub trait BackendQuant: Backend {
    /// Quantize `x` (`n` f32 elements) into `x_int8` + `scales` under the
    /// given [`QuantScheme`].
    fn quantize(
        &self,
        rec: &mut Self::Recorder<'_>,
        scheme: QuantScheme,
        x: &Self::Buffer,
        x_int8: &Self::Buffer,
        scales: &Self::Buffer,
        n: usize,
    ) -> Result<()>;
}

/// Embedding-table lookup.
pub trait BackendEmbed: Backend {
    /// Write the `hidden`-wide embedding row for `token` into `out`.
    fn embed(
        &self,
        rec: &mut Self::Recorder<'_>,
        embed_table: &Self::Buffer,
        token: u32,
        out: &Self::Buffer,
        hidden: usize,
    ) -> Result<()>;
}

/// Token sampling.
pub trait BackendSample: Backend {
    /// Argmax over `logits` (`vocab` elements), writing the winning id
    /// into `token_out[0]`. Pair with [`CommandRecorder::read_u32`] to
    /// pull the id back to the host after commit.
    fn sample_argmax(
        &self,
        rec: &mut Self::Recorder<'_>,
        logits: &Self::Buffer,
        token_out: &Self::Buffer,
        vocab: usize,
    ) -> Result<()>;
}

/// Mixture-of-experts routed block.
pub trait BackendMoe: Backend {
    /// Top-k gating: from `logits` produce the selected expert ids and
    /// their normalized weights.
    fn moe_topk_gate(
        &self,
        rec: &mut Self::Recorder<'_>,
        logits: &Self::Buffer,
        route_ids: &Self::Buffer,
        route_weights: &Self::Buffer,
        n_experts: usize,
        top_k: usize,
    ) -> Result<()>;

    /// Run the routed (and optional shared) expert block for the gated
    /// routes, accumulating into `out`. Expert weights are addressed by
    /// byte offsets into the single pinned model buffer `model`.
    fn moe_block(
        &self,
        rec: &mut Self::Recorder<'_>,
        model: &Self::Buffer,
        route_ids: &Self::Buffer,
        route_weights: &Self::Buffer,
        x: &Self::Buffer,
        out: &Self::Buffer,
        routed_gate_offset: usize,
        routed_up_offset: usize,
        routed_down_offset: usize,
        hidden: usize,
        routed_mid: usize,
        routes: usize,
    ) -> Result<()>;
}

/// The compute-backend base: the associated [`Self::Buffer`] /
/// [`Self::Recorder`] types, the recorder lifecycle, and `supports()`.
/// Every op trait ([`BackendGemv`], [`BackendNorm`], …) has this as its
/// supertrait so it can name `Self::Buffer` / `Self::Recorder`.
///
/// The full capability bundle is [`ComputeBackend`] (this base + every op
/// trait); that is the "one bound" a model is generic over. Splitting the
/// base from the bundle avoids a supertrait cycle (`Backend` cannot list
/// the op traits as supertraits while each op trait has `Backend` as its
/// supertrait).
///
/// A concrete backend (e.g. `MetalBackend`, defined in the macOS-gated
/// `backend/metal.rs`) binds `Buffer` and `Recorder` to its concrete
/// types and implements every op trait. Models call op methods through a
/// `&B` plus a `&mut B::Recorder<'_>` threaded through the forward pass.
pub trait Backend: Sized {
    /// The GPU/host buffer handle (e.g. a Metal buffer). Kept concrete in
    /// 3.1 so the decode arena and weight fields are untouched.
    type Buffer;

    /// The per-token command recorder, generic over the borrow of the
    /// backend context it records against (GAT). Its associated
    /// [`CommandRecorder::Buffer`] is the same type as [`Self::Buffer`]
    /// for any well-formed backend.
    type Recorder<'a>: CommandRecorder<Buffer = Self::Buffer>
    where
        Self: 'a;

    /// Begin a fresh per-token recorder. The returned recorder borrows
    /// `self` for the duration of the token's forward pass and is
    /// finalized with [`CommandRecorder::commit_and_wait`].
    fn recorder(&self) -> Self::Recorder<'_>;

    /// Whether this backend implements the given logical op. Lets a
    /// partially-implemented backend (e.g. a CPU backend without MoE)
    /// advertise its capabilities; a fully-featured backend returns
    /// `true` for every variant.
    fn supports(&self, op: Op) -> bool;
}

/// The full compute-backend capability bundle: the [`Backend`] base plus
/// every op trait. This is the single bound a model is generic over
/// (`B: ComputeBackend`). The blanket impl makes any type implementing the
/// base + all op traits a `ComputeBackend` automatically — concrete
/// backends never implement it directly.
pub trait ComputeBackend:
    Backend
    + BackendGemv
    + BackendNorm
    + BackendElementwise
    + BackendRope
    + BackendAttention
    + BackendKvCache
    + BackendQuant
    + BackendEmbed
    + BackendSample
    + BackendMoe
{
}

impl<T> ComputeBackend for T where
    T: Backend
        + BackendGemv
        + BackendNorm
        + BackendElementwise
        + BackendRope
        + BackendAttention
        + BackendKvCache
        + BackendQuant
        + BackendEmbed
        + BackendSample
        + BackendMoe
{
}

