//! Llama-family dense forward pass (Llama-2 / Llama-3.x / Mistral).
//!
//! Architecturally a near-sibling of `qwen_dense`:
//!
//!   - Grouped-query attention (n_heads / n_kv_heads)
//!   - SwiGLU FFN (gate + up + down)
//!   - RMSNorm
//!   - RoPE; Llama-3.1+ adds NTK-aware piecewise frequency rescaling
//!     (see [`crate::kernels::Llama3RopeScaling`]).
//!
//! Two structural differences from Qwen2:
//!
//!   1. No Q/K/V biases (Llama families omit them; Qwen2 carries them).
//!   2. RoPE θ is typically 500_000 (Llama-3) instead of 1_000_000.
//!
//! On macOS the Q4_K projections, the f16 LM head, and rmsnorm run on
//! Metal; the remaining ops (Q6_K weights, attention) use the CPU
//! reference path. The full TCB pinned-buffer + predec arena (Qwen's
//! `forward_token_greedy_tcb`) is a follow-up best done with a real
//! GGUF in hand to bench against.

use super::arch_config::{token_embd_vocab_size, ArchReader};
use super::dispatch::{gemv_f16_dispatch, rmsnorm_dispatch};
use super::weights::{dequant_f16, dequant_f32, dequant_ref_into, tensor_ref, TensorRef};
use crate::attn::mha_decode_step;
use crate::cache::KvCache;
use crate::engine::{Engine, EngineConfig, GenStats, GenerateRequest, StopReason, StreamEvent};
use crate::gguf::{GgmlType, GgufFile};
use crate::kernels::{
    add_inplace, gemv_f32, rope_inplace_normal_with_factors, silu_mul, Llama3RopeScaling,
};
use crate::metal::MetalContext;
#[cfg(target_os = "macos")]
use crate::metal::{
    ReplayBufferBinding, ReplayComputeStage, ReplayableComputeGraph, TokenCommandBuffer,
};
use crate::profile::KernelProfile;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Debug-only numerical checkpoint record for one token forward pass.
///
/// This is intentionally scalar-only by default: it is small enough to
/// capture on a real model, while still matching the `llama-eval-callback`
/// aggregate surfaces used by the independent Llama oracle. An explicit
/// vector-surface selector can attach exactly one f32 surface for elementwise
/// K0 diagnosis. It is populated only when
/// `HAWKING_LLAMA_CHECKPOINT_SUMMARY_PATH` is set at model load time.
#[derive(Debug, Serialize)]
struct LlamaCheckpointRecord {
    position: usize,
    token_id: u32,
    embedding_sum: f64,
    layers: Vec<LlamaLayerCheckpoint>,
    final_norm_sum: f64,
    logits_sum: f64,
    greedy_token_id: u32,
    debug_vector: Option<LlamaCheckpointVector>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    debug_vectors: Vec<LlamaCheckpointVector>,
    #[serde(skip_serializing_if = "Option::is_none")]
    capture_kind: Option<&'static str>,
}

/// Opt-in one-surface vector dump for a numerical parity investigation.
/// Never set by normal inference or performance commands.
#[derive(Debug, Serialize)]
struct LlamaCheckpointVector {
    surface: String,
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct LlamaLayerCheckpoint {
    layer: usize,
    attn_norm_sum: f64,
    q_raw_sum: f64,
    k_raw_sum: f64,
    v_raw_sum: f64,
    q_rope_sum: f64,
    k_rope_sum: f64,
    attn_out_sum: f64,
    ffn_input_sum: f64,
    ffn_norm_sum: f64,
    ffn_gate_sum: f64,
    ffn_up_sum: f64,
    ffn_swiglu_sum: f64,
    ffn_out_sum: f64,
    layer_out_sum: f64,
}

#[inline]
fn checkpoint_sum(values: &[f32]) -> f64 {
    values.iter().map(|&value| value as f64).sum()
}

#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    /// Llama-3.1+ NTK-aware RoPE rescaling. `Some` when the GGUF carries
    /// `llama.rope.scaling.type == "llama3"` and the four
    /// `llama.rope.scaling.*` parameters. Earlier Llama / Mistral GGUFs
    /// leave this `None`.
    pub rope_scaling: Option<Llama3RopeScaling>,
    /// Distinguish Llama-2 / Llama-3 / Mistral for reporting. Carried
    /// from GGUF `general.architecture` verbatim so profile matching
    /// (via [`crate::profile::arch_family`]) and downstream logging both
    /// see the original arch string.
    pub arch: String,
    /// Resolved per-pair RoPE divisors from GGUF `rope_freqs.weight`. Llama
    /// 3.1 converters commonly carry long-context scaling here instead of
    /// the optional `llama.rope.scaling.*` metadata fields.
    pub rope_freq_factors: Option<Vec<f32>>,
}

impl LlamaConfig {
    pub fn from_gguf(g: &GgufFile) -> Result<Self> {
        let arch = g.architecture().unwrap_or("").to_string();
        // Every Llama-family GGUF llama.cpp produces uses the `llama.*`
        // metadata prefix regardless of point release (Llama-2/3/3.1/3.2,
        // Mistral, Phi when ported to llama.cpp's "llama" arch). So a
        // single prefix here covers the whole family.
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());
        let get_str = |k: &str| g.metadata.get(k).and_then(|v| v.as_str());
        // P1-D1: shared core reads via ArchReader; the vocab fallback,
        // rope-scaling and sliding-window extras below keep the closures.
        let r = ArchReader::new(g, "llama");

        let n_layers = r.req_usize("block_count")?;
        let hidden = r.req_usize("embedding_length")?;
        let n_heads = r.req_usize("attention.head_count")?;
        let n_kv_heads = r.opt_usize("attention.head_count_kv", n_heads);
        // Some Llama GGUFs ship an explicit head_dim (e.g. Llama-3.2 1B
        // where hidden=2048 but head_dim=64 with 32 heads); fall back to
        // hidden/n_heads when absent.
        let head_dim = r.opt_usize("attention.key_length", hidden / n_heads);
        let intermediate = r.req_usize("feed_forward_length")?;
        let vocab_size = match get_u32("llama.vocab_size") {
            Some(v) => v as usize,
            // GGUF dim ordering varies; vocab >> hidden in practice, so the max
            // dim on the embed tensor is the vocab size.
            None => token_embd_vocab_size(g, "vocab size not in metadata or token_embd dims")?,
        };
        let rope_theta = r.opt_f32("rope.freq_base", 500_000.0);
        let rms_norm_eps = r.opt_f32("attention.layer_norm_rms_epsilon", 1e-5);
        let max_seq_len = r.opt_usize("context_length", 8192);

        // Sliding-window attention: Mistral-7B-v0.1 windows attention at
        // `llama.attention.sliding_window`; v0.2/v0.3 (our target) and
        // Llama dropped it. This engine runs full causal attention, so a
        // window strictly smaller than the context would be silently
        // wrong on long prompts. Surface it rather than fail quietly.
        if let Some(win) = get_u32("llama.attention.sliding_window") {
            if (win as usize) < max_seq_len {
                eprintln!(
                    "hawking: warning — GGUF declares sliding_window={win} but the \
                     llama engine runs full causal attention; output may drift beyond \
                     {win} tokens of context (use a non-SWA build such as \
                     Mistral-7B-Instruct-v0.3)"
                );
            }
        }

        // RoPE NTK-aware scaling: only honored when scaling.type ==
        // "llama3". Some Llama-3.0 GGUFs leave scaling.type unset; in
        // that case the four scaling params are absent and we fall
        // through to unscaled RoPE.
        let rope_scaling = if get_str("llama.rope.scaling.type") == Some("llama3") {
            let factor = get_f32("llama.rope.scaling.factor");
            let low = get_f32("llama.rope.scaling.low_freq_factor");
            let high = get_f32("llama.rope.scaling.high_freq_factor");
            let orig = get_u32("llama.rope.scaling.original_context_length");
            match (factor, low, high, orig) {
                (Some(factor), Some(low_freq_factor), Some(high_freq_factor), Some(orig_ctx)) => {
                    Some(Llama3RopeScaling {
                        factor,
                        low_freq_factor,
                        high_freq_factor,
                        original_max_position_embeddings: orig_ctx,
                    })
                }
                _ => None,
            }
        } else {
            None
        };
        let rope_freq_factors = match g.tensor("rope_freqs.weight") {
            Some(_) => {
                let factors = dequant_f32(g, "rope_freqs.weight")?;
                if factors.len() != head_dim / 2
                    || factors
                        .iter()
                        .any(|factor| !factor.is_finite() || *factor <= 0.0)
                {
                    return Err(Error::Model(format!(
                        "rope_freqs.weight must contain {} finite positive factors, found {}",
                        head_dim / 2,
                        factors.len()
                    )));
                }
                Some(factors)
            }
            None => None,
        };

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv_heads,
            head_dim,
            intermediate,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            max_seq_len,
            rope_scaling,
            arch,
            rope_freq_factors,
        })
    }
}

pub struct LlamaLayer {
    /// Per-layer norms (eager fp32, small).
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    /// Attention projection weights (read via TensorRef on each forward;
    /// Q4_K reads bytes straight from the mmap for the Metal GEMV).
    pub(crate) q_proj: TensorRef,
    pub(crate) k_proj: TensorRef,
    pub(crate) v_proj: TensorRef,
    pub(crate) o_proj: TensorRef,
    // Llama families omit Q/K/V biases (unlike Qwen2).
    /// FFN weights.
    pub(crate) ffn_gate: TensorRef,
    pub(crate) ffn_up: TensorRef,
    pub(crate) ffn_down: TensorRef,
}

/// Buffer-backed form of the scalar argument block accepted by the frozen
/// b9430 RoPE kernel.  An indirect command buffer cannot capture `set_bytes`,
/// so the replay lane updates this persistent representation once per token.
#[cfg(target_os = "macos")]
#[repr(C)]
struct LlamaB9430ReplayRopeArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    ne03: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    nb03: u64,
    ne0: i32,
    ne1: i32,
    ne2: i32,
    ne3: i32,
    nb0: u64,
    nb1: u64,
    nb2: u64,
    nb3: u64,
    n_past: i32,
    n_dims: i32,
    n_ctx_orig: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    sect_0: i32,
    sect_1: i32,
    sect_2: i32,
    sect_3: i32,
    src2: u8,
}

/// The generic long-context attention argument layout in `mha.metal`.
#[cfg(target_os = "macos")]
#[repr(C)]
struct LlamaB9430ReplayMhaArgs {
    seq_len: u32,
    head_dim: u32,
    n_kv_heads: u32,
    group_size: u32,
    scale: f32,
}

/// Exact f32/f16 cache append parameters.  This matches `ArgbufMemcpyF32`.
#[cfg(target_os = "macos")]
#[repr(C)]
struct LlamaB9430ReplayCopyArgs {
    n: u32,
    src_off: u32,
    dst_off: u32,
}

/// One complete pre-encoded Llama decode graph for each attention authority
/// shape.  The short graph remains the b9430 <=32-token authority; the long
/// graph uses the existing materialized f32 attention implementation.
#[cfg(target_os = "macos")]
struct LlamaB9430Replay {
    short_front: ReplayableComputeGraph,
    short_middle: ReplayableComputeGraph,
    short_back: ReplayableComputeGraph,
    long_front: ReplayableComputeGraph,
    long_middle: ReplayableComputeGraph,
    long_back: ReplayableComputeGraph,
    probe: Option<ReplayableComputeGraph>,
    // ICB-safe resident copies for tensors whose GGUF binding offsets exceed
    // the 32-bit range accepted by this device's ICB compiler. These are
    // RAM-only views of the original GGUF bytes, never persisted or
    // re-quantized.
    q4_windows: Vec<(usize, crate::metal::PinnedBuffer)>,
    q6_windows: Vec<(usize, crate::metal::PinnedBuffer)>,
    rope_args: crate::metal::PinnedBuffer,
    cache_copy_args: crate::metal::PinnedBuffer,
    short_seq_len: crate::metal::PinnedBuffer,
    long_mha_args: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
struct LlamaB9430ReplayScalars {
    hidden: crate::metal::PinnedBuffer,
    intermediate: crate::metal::PinnedBuffer,
    q_dim: crate::metal::PinnedBuffer,
    kv_dim: crate::metal::PinnedBuffer,
    vocab: crate::metal::PinnedBuffer,
    n_heads: crate::metal::PinnedBuffer,
    n_kv_heads: crate::metal::PinnedBuffer,
    eps: crate::metal::PinnedBuffer,
    scale: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
impl LlamaB9430ReplayScalars {
    fn u32(ctx: &MetalContext, value: usize) -> crate::metal::PinnedBuffer {
        ctx.new_buffer_with_bytes(&(value as u32).to_ne_bytes())
    }

    fn new(ctx: &MetalContext, cfg: &LlamaConfig) -> Self {
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        Self {
            hidden: Self::u32(ctx, cfg.hidden),
            intermediate: Self::u32(ctx, cfg.intermediate),
            q_dim: Self::u32(ctx, q_dim),
            kv_dim: Self::u32(ctx, kv_dim),
            vocab: Self::u32(ctx, cfg.vocab_size),
            n_heads: Self::u32(ctx, cfg.n_heads),
            n_kv_heads: Self::u32(ctx, cfg.n_kv_heads),
            eps: ctx.new_buffer_with_bytes(&cfg.rms_norm_eps.to_ne_bytes()),
            scale: ctx
                .new_buffer_with_bytes(&(1.0_f32 / (cfg.head_dim as f32).sqrt()).to_ne_bytes()),
        }
    }
}

#[cfg(target_os = "macos")]
impl LlamaB9430Replay {
    const TG: u32 = 256;
    const MAX_REPLAY_SEQ: usize = 7_936;
    // Layer 30's Q4 query projection is non-finite when replayed by this
    // device's ICB compiler. Keep both of that layer's independently-produced
    // Q and Q6 V projections on the established direct TCB path; the K
    // projection stays replayable between them.
    const DIRECT_Q4_Q_STAGE: usize = 30 * 19 + 1;
    const DIRECT_Q6_V_STAGE: usize = 30 * 19 + 3;

    fn add_stage(stages: &mut Vec<ReplayComputeStage>, stage: ReplayComputeStage) {
        stages.push(if stages.is_empty() {
            stage
        } else {
            stage.with_barrier_before()
        });
    }

    fn q4_stage(
        model: &crate::metal::PinnedBuffer,
        q4_windows: &[(usize, crate::metal::PinnedBuffer)],
        tensor: &TensorRef,
        rows: usize,
        _cols: usize,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
        row_arg: &crate::metal::PinnedBuffer,
        col_arg: &crate::metal::PinnedBuffer,
    ) -> Result<ReplayComputeStage> {
        let (weight, weight_offset) = q4_windows
            .iter()
            .find_map(|(offset, window)| (*offset == tensor.offset).then_some((window, 0usize)))
            .unwrap_or((model, tensor.offset));
        let end = weight_offset
            .checked_add(tensor.byte_size)
            .ok_or_else(|| Error::Model("Llama replay Q4 tensor range overflow".into()))?;
        if end as u64 > weight.length() {
            return Err(Error::Model(
                "Llama replay Q4 tensor is outside its pinned window".into(),
            ));
        }
        Ok(ReplayComputeStage::new(
            "gemm_q4_k_m_llama_b9430",
            ((rows.div_ceil(4) as u32) * 64, 1, 1),
            (64, 1, 1),
            vec![
                ReplayBufferBinding::read(0, weight, weight_offset),
                ReplayBufferBinding::read(1, input, 0),
                ReplayBufferBinding::write(2, output, 0),
                ReplayBufferBinding::read(3, row_arg, 0),
                ReplayBufferBinding::read(4, col_arg, 0),
            ],
        ))
    }

    fn q6_stage(
        weight: &crate::metal::PinnedBuffer,
        tensor: &TensorRef,
        rows: usize,
        _cols: usize,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
        row_arg: &crate::metal::PinnedBuffer,
        col_arg: &crate::metal::PinnedBuffer,
    ) -> Result<ReplayComputeStage> {
        tensor
            .offset
            .checked_add(tensor.byte_size)
            .ok_or_else(|| Error::Model("Llama replay Q6 tensor range overflow".into()))?;
        if tensor.byte_size as u64 > weight.length() {
            return Err(Error::Model(
                "Llama replay Q6 tensor is outside pinned GGUF".into(),
            ));
        }
        Ok(ReplayComputeStage::new(
            "gemm_q6_k_llama_b9430",
            ((rows.div_ceil(4) as u32) * 64, 1, 1),
            (64, 1, 1),
            vec![
                ReplayBufferBinding::read(0, weight, 0),
                ReplayBufferBinding::read(1, input, 0),
                ReplayBufferBinding::write(2, output, 0),
                ReplayBufferBinding::read(3, row_arg, 0),
                ReplayBufferBinding::read(4, col_arg, 0),
            ],
        ))
    }

    fn projection_stage(
        model: &crate::metal::PinnedBuffer,
        q4_windows: &[(usize, crate::metal::PinnedBuffer)],
        q6_windows: &[(usize, crate::metal::PinnedBuffer)],
        tensor: &TensorRef,
        rows: usize,
        cols: usize,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
        row_arg: &crate::metal::PinnedBuffer,
        col_arg: &crate::metal::PinnedBuffer,
    ) -> Result<ReplayComputeStage> {
        match tensor.dtype {
            GgmlType::Q4_K => Self::q4_stage(
                model, q4_windows, tensor, rows, cols, input, output, row_arg, col_arg,
            ),
            GgmlType::Q6_K => {
                let weight = q6_windows
                    .iter()
                    .find_map(|(offset, weight)| (*offset == tensor.offset).then_some(weight))
                    .ok_or_else(|| {
                        Error::Model(format!(
                            "Llama replay has no Q6 mmap window at {}",
                            tensor.offset
                        ))
                    })?;
                Self::q6_stage(weight, tensor, rows, cols, input, output, row_arg, col_arg)
            }
            other => Err(Error::Model(format!(
                "Llama replay has unsupported projection dtype {other:?}"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_graph(
        ctx: &MetalContext,
        cfg: &LlamaConfig,
        layers: &[LlamaLayer],
        raw_head: &TensorRef,
        model: &crate::metal::PinnedBuffer,
        q4_windows: &[(usize, crate::metal::PinnedBuffer)],
        q6_windows: &[(usize, crate::metal::PinnedBuffer)],
        arena: &LlamaB9430ResidentArena,
        scalars: &LlamaB9430ReplayScalars,
        rope_args: &crate::metal::PinnedBuffer,
        cache_copy_args: &crate::metal::PinnedBuffer,
        short_seq_len: &crate::metal::PinnedBuffer,
        _long_mha_args: &crate::metal::PinnedBuffer,
        short_attention: bool,
        long_fattn: bool,
        stage_start: usize,
        stage_limit: Option<usize>,
    ) -> Result<ReplayableComputeGraph> {
        let mut stages = Vec::with_capacity(layers.len() * 19 + 2);
        let h = cfg.hidden;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let cache_grid = (kv_dim as u32).div_ceil(Self::TG) * Self::TG;
        for (li, layer) in layers.iter().enumerate() {
            Self::add_stage(
                &mut stages,
                ReplayComputeStage::new(
                    "rmsnorm_llama_b9430",
                    (1024, 1, 1),
                    (1024, 1, 1),
                    vec![
                        ReplayBufferBinding::read(0, &arena.x, 0),
                        ReplayBufferBinding::read(1, &arena.attn_norm_weights[li], 0),
                        ReplayBufferBinding::write(2, &arena.attn_norm_out, 0),
                        ReplayBufferBinding::read(3, &scalars.hidden, 0),
                        ReplayBufferBinding::read(4, &scalars.eps, 0),
                    ],
                )
                .with_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>()),
            );
            Self::add_stage(
                &mut stages,
                Self::q4_stage(
                    model,
                    q4_windows,
                    &layer.q_proj,
                    q_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.q_raw,
                    &scalars.q_dim,
                    &scalars.hidden,
                )?,
            );
            Self::add_stage(
                &mut stages,
                Self::q4_stage(
                    model,
                    q4_windows,
                    &layer.k_proj,
                    kv_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.k_raw,
                    &scalars.kv_dim,
                    &scalars.hidden,
                )?,
            );
            Self::add_stage(
                &mut stages,
                Self::projection_stage(
                    model,
                    q4_windows,
                    q6_windows,
                    &layer.v_proj,
                    kv_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.v,
                    &scalars.kv_dim,
                    &scalars.hidden,
                )?,
            );
            for (input, output, heads) in [
                (&arena.q_raw, &arena.q_rope, cfg.n_heads),
                (&arena.k_raw, &arena.k_rope, cfg.n_kv_heads),
            ] {
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "rope_norm_llama_b9430",
                        ((heads * cfg.head_dim) as u32, 1, 1),
                        (cfg.head_dim as u32, 1, 1),
                        vec![
                            ReplayBufferBinding::read(0, rope_args, 0),
                            ReplayBufferBinding::read(1, input, 0),
                            ReplayBufferBinding::read(2, &arena.position, 0),
                            ReplayBufferBinding::read(3, &arena.rope_factors, 0),
                            ReplayBufferBinding::write(4, output, 0),
                        ],
                    ),
                );
            }
            for buffer in [&arena.k_rope, &arena.v] {
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "round_f16_llama_b9430",
                        (kv_dim as u32, 1, 1),
                        (Self::TG.min(kv_dim as u32), 1, 1),
                        vec![
                            ReplayBufferBinding::read_write(0, buffer, 0),
                            ReplayBufferBinding::read(1, &scalars.kv_dim, 0),
                        ],
                    ),
                );
            }
            for (source, f32_cache, f16_cache) in [
                (&arena.k_rope, &arena.keys_f32[li], &arena.keys_f16[li]),
                (&arena.v, &arena.values_f32[li], &arena.values_f16[li]),
            ] {
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "llama_b9430_cache_append_f32_f16",
                        (cache_grid, 1, 1),
                        (Self::TG, 1, 1),
                        vec![
                            ReplayBufferBinding::read(0, source, 0),
                            ReplayBufferBinding::write(1, f32_cache, 0),
                            ReplayBufferBinding::write(2, f16_cache, 0),
                            ReplayBufferBinding::read(3, cache_copy_args, 0),
                        ],
                    ),
                );
            }
            if short_attention {
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "mha_decode_llama_b9430_short",
                        ((cfg.n_heads * 32) as u32, 1, 1),
                        (32, 1, 1),
                        vec![
                            ReplayBufferBinding::read(0, &arena.q_rope, 0),
                            ReplayBufferBinding::read(1, &arena.keys_f16[li], 0),
                            ReplayBufferBinding::read(2, &arena.values_f16[li], 0),
                            ReplayBufferBinding::write(3, &arena.attn_out, 0),
                            ReplayBufferBinding::read(4, short_seq_len, 0),
                            ReplayBufferBinding::read(5, &scalars.n_heads, 0),
                            ReplayBufferBinding::read(6, &scalars.n_kv_heads, 0),
                            ReplayBufferBinding::read(7, &scalars.scale, 0),
                        ],
                    )
                    .with_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>()),
                );
            } else if long_fattn {
                // The generic f32 attention graph is exact but far slower
                // than the source f16 FATTN authority at real decode
                // contexts. Keep the same main/reduce grammar used by the
                // validated resident serial lane, with an ICB barrier between
                // the scratch producer and reduction consumer.
                let fattn_grid = (cfg.n_heads * 32 * 32) as u32;
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "mha_decode_llama_b9430_fattn_main",
                        (fattn_grid, 1, 1),
                        (32, 1, 1),
                        vec![
                            ReplayBufferBinding::read(0, &arena.q_rope, 0),
                            ReplayBufferBinding::read(1, &arena.keys_f16[li], 0),
                            ReplayBufferBinding::read(2, &arena.values_f16[li], 0),
                            ReplayBufferBinding::write(3, &arena.fattn_scratch, 0),
                            ReplayBufferBinding::read(4, short_seq_len, 0),
                            ReplayBufferBinding::read(5, &scalars.n_heads, 0),
                            ReplayBufferBinding::read(6, &scalars.n_kv_heads, 0),
                            ReplayBufferBinding::read(7, &scalars.scale, 0),
                        ],
                    )
                    .with_threadgroup_memory_length(0, 1024),
                );
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "mha_decode_llama_b9430_fattn_reduce",
                        (fattn_grid, 1, 1),
                        (1024, 1, 1),
                        vec![
                            ReplayBufferBinding::read(0, &arena.fattn_scratch, 0),
                            ReplayBufferBinding::write(1, &arena.attn_out, 0),
                            ReplayBufferBinding::read(2, &scalars.n_heads, 0),
                        ],
                    ),
                );
            } else {
                Self::add_stage(
                    &mut stages,
                    ReplayComputeStage::new(
                        "mha_decode_f32",
                        ((cfg.n_heads * 128) as u32, 1, 1),
                        (128, 1, 1),
                        vec![
                            ReplayBufferBinding::read(0, _long_mha_args, 0),
                            ReplayBufferBinding::read(1, &arena.q_rope, 0),
                            ReplayBufferBinding::read(2, &arena.keys_f32[li], 0),
                            ReplayBufferBinding::read(3, &arena.values_f32[li], 0),
                            ReplayBufferBinding::write(4, &arena.attn_out, 0),
                        ],
                    )
                    .with_threadgroup_memory_length(
                        0,
                        (arena.max_seq * std::mem::size_of::<f32>())
                            + 128 * std::mem::size_of::<f32>(),
                    ),
                );
            }
            Self::add_stage(
                &mut stages,
                Self::q4_stage(
                    model,
                    q4_windows,
                    &layer.o_proj,
                    h,
                    q_dim,
                    &arena.attn_out,
                    &arena.o,
                    &scalars.hidden,
                    &scalars.q_dim,
                )?,
            );
            Self::add_stage(
                &mut stages,
                ReplayComputeStage::new(
                    "add_inplace",
                    (((h as u32).div_ceil(Self::TG)) * Self::TG, 1, 1),
                    (Self::TG, 1, 1),
                    vec![
                        ReplayBufferBinding::read_write(0, &arena.x, 0),
                        ReplayBufferBinding::read(1, &arena.o, 0),
                        ReplayBufferBinding::read(2, &scalars.hidden, 0),
                    ],
                ),
            );
            Self::add_stage(
                &mut stages,
                ReplayComputeStage::new(
                    "rmsnorm_llama_b9430",
                    (1024, 1, 1),
                    (1024, 1, 1),
                    vec![
                        ReplayBufferBinding::read(0, &arena.x, 0),
                        ReplayBufferBinding::read(1, &arena.ffn_norm_weights[li], 0),
                        ReplayBufferBinding::write(2, &arena.ffn_norm_out, 0),
                        ReplayBufferBinding::read(3, &scalars.hidden, 0),
                        ReplayBufferBinding::read(4, &scalars.eps, 0),
                    ],
                )
                .with_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>()),
            );
            for (tensor, output) in [(&layer.ffn_gate, &arena.gate), (&layer.ffn_up, &arena.up)] {
                Self::add_stage(
                    &mut stages,
                    Self::q4_stage(
                        model,
                        q4_windows,
                        tensor,
                        cfg.intermediate,
                        h,
                        &arena.ffn_norm_out,
                        output,
                        &scalars.intermediate,
                        &scalars.hidden,
                    )?,
                );
            }
            Self::add_stage(
                &mut stages,
                ReplayComputeStage::new(
                    "swiglu_llama_b9430",
                    (cfg.intermediate as u32, 1, 1),
                    (Self::TG.min(cfg.intermediate as u32), 1, 1),
                    vec![
                        ReplayBufferBinding::read(0, &arena.gate, 0),
                        ReplayBufferBinding::read(1, &arena.up, 0),
                        ReplayBufferBinding::write(2, &arena.act, 0),
                        ReplayBufferBinding::read(3, &scalars.intermediate, 0),
                    ],
                ),
            );
            Self::add_stage(
                &mut stages,
                Self::projection_stage(
                    model,
                    q4_windows,
                    q6_windows,
                    &layer.ffn_down,
                    h,
                    cfg.intermediate,
                    &arena.act,
                    &arena.ffn_out,
                    &scalars.hidden,
                    &scalars.intermediate,
                )?,
            );
            Self::add_stage(
                &mut stages,
                ReplayComputeStage::new(
                    "add_inplace",
                    (((h as u32).div_ceil(Self::TG)) * Self::TG, 1, 1),
                    (Self::TG, 1, 1),
                    vec![
                        ReplayBufferBinding::read_write(0, &arena.x, 0),
                        ReplayBufferBinding::read(1, &arena.ffn_out, 0),
                        ReplayBufferBinding::read(2, &scalars.hidden, 0),
                    ],
                ),
            );
        }
        Self::add_stage(
            &mut stages,
            ReplayComputeStage::new(
                "rmsnorm_llama_b9430",
                (1024, 1, 1),
                (1024, 1, 1),
                vec![
                    ReplayBufferBinding::read(0, &arena.x, 0),
                    ReplayBufferBinding::read(1, &arena.final_norm_weight, 0),
                    ReplayBufferBinding::write(2, &arena.attn_norm_out, 0),
                    ReplayBufferBinding::read(3, &scalars.hidden, 0),
                    ReplayBufferBinding::read(4, &scalars.eps, 0),
                ],
            )
            .with_threadgroup_memory_length(0, 32 * std::mem::size_of::<f32>()),
        );
        Self::add_stage(
            &mut stages,
            Self::projection_stage(
                model,
                q4_windows,
                q6_windows,
                raw_head,
                cfg.vocab_size,
                h,
                &arena.attn_norm_out,
                &arena.logits,
                &scalars.vocab,
                &scalars.hidden,
            )?,
        );
        let end = stage_limit.unwrap_or(stages.len()).min(stages.len());
        let start = stage_start.min(end);
        stages = stages.drain(start..end).collect();
        ReplayableComputeGraph::new(ctx, stages)
    }

    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &MetalContext,
        cfg: &LlamaConfig,
        layers: &[LlamaLayer],
        raw_head: &TensorRef,
        model: &crate::metal::PinnedBuffer,
        mmap: &[u8],
        arena: &LlamaB9430ResidentArena,
    ) -> Result<Self> {
        if arena.max_seq > Self::MAX_REPLAY_SEQ {
            return Err(Error::Model(format!(
                "Llama replay needs max_seq <= {}; got {}",
                Self::MAX_REPLAY_SEQ,
                arena.max_seq
            )));
        }
        let scalars = LlamaB9430ReplayScalars::new(ctx, cfg);
        let rope_args = ctx.new_buffer(std::mem::size_of::<LlamaB9430ReplayRopeArgs>());
        let cache_copy_args = ctx.new_buffer(std::mem::size_of::<LlamaB9430ReplayCopyArgs>());
        let short_seq_len = ctx.new_buffer(std::mem::size_of::<u32>());
        let long_mha_args = ctx.new_buffer(std::mem::size_of::<LlamaB9430ReplayMhaArgs>());
        let replay_fattn = crate::env_on("HAWKING_LLAMA_RESIDENT_REPLAY_FATTN_UNSAFE_DIAGNOSTIC");
        let mut q4_windows = Vec::new();
        let mut q6_windows = Vec::new();
        for tensor in layers
            .iter()
            .flat_map(|layer| {
                [
                    &layer.q_proj,
                    &layer.k_proj,
                    &layer.v_proj,
                    &layer.o_proj,
                    &layer.ffn_gate,
                    &layer.ffn_up,
                    &layer.ffn_down,
                ]
            })
            .chain(std::iter::once(raw_head))
        {
            let end = tensor
                .offset
                .checked_add(tensor.byte_size)
                .ok_or_else(|| Error::Model("Llama replay tensor mmap window overflows".into()))?;
            match tensor.dtype {
                GgmlType::Q4_K
                    if tensor.offset > u32::MAX as usize
                        && !q4_windows
                            .iter()
                            .any(|(offset, _)| *offset == tensor.offset) =>
                {
                    let bytes = mmap.get(tensor.offset..end).ok_or_else(|| {
                        Error::Model("Llama replay Q4 mmap window is out of bounds".into())
                    })?;
                    q4_windows.push((tensor.offset, ctx.new_buffer_with_bytes(bytes)));
                }
                GgmlType::Q6_K
                    if !q6_windows
                        .iter()
                        .any(|(offset, _)| *offset == tensor.offset) =>
                {
                    let bytes = mmap.get(tensor.offset..end).ok_or_else(|| {
                        Error::Model("Llama replay Q6 mmap window is out of bounds".into())
                    })?;
                    q6_windows.push((tensor.offset, ctx.new_buffer_with_bytes(bytes)));
                }
                _ => {}
            }
        }
        let short_front = Self::build_graph(
            ctx,
            cfg,
            layers,
            raw_head,
            model,
            &q4_windows,
            &q6_windows,
            arena,
            &scalars,
            &rope_args,
            &cache_copy_args,
            &short_seq_len,
            &long_mha_args,
            true,
            false,
            0,
            Some(Self::DIRECT_Q4_Q_STAGE),
        )?;
        let short_middle = Self::build_graph(
            ctx,
            cfg,
            layers,
            raw_head,
            model,
            &q4_windows,
            &q6_windows,
            arena,
            &scalars,
            &rope_args,
            &cache_copy_args,
            &short_seq_len,
            &long_mha_args,
            true,
            false,
            Self::DIRECT_Q4_Q_STAGE + 1,
            Some(Self::DIRECT_Q6_V_STAGE),
        )?;
        let short_back = Self::build_graph(
            ctx,
            cfg,
            layers,
            raw_head,
            model,
            &q4_windows,
            &q6_windows,
            arena,
            &scalars,
            &rope_args,
            &cache_copy_args,
            &short_seq_len,
            &long_mha_args,
            true,
            false,
            Self::DIRECT_Q6_V_STAGE + 1,
            None,
        )?;
        let long_front = Self::build_graph(
            ctx,
            cfg,
            layers,
            raw_head,
            model,
            &q4_windows,
            &q6_windows,
            arena,
            &scalars,
            &rope_args,
            &cache_copy_args,
            &short_seq_len,
            &long_mha_args,
            false,
            replay_fattn,
            0,
            Some(Self::DIRECT_Q4_Q_STAGE),
        )?;
        let long_middle = Self::build_graph(
            ctx,
            cfg,
            layers,
            raw_head,
            model,
            &q4_windows,
            &q6_windows,
            arena,
            &scalars,
            &rope_args,
            &cache_copy_args,
            &short_seq_len,
            &long_mha_args,
            false,
            replay_fattn,
            Self::DIRECT_Q4_Q_STAGE + 1,
            Some(Self::DIRECT_Q6_V_STAGE),
        )?;
        let long_back = Self::build_graph(
            ctx,
            cfg,
            layers,
            raw_head,
            model,
            &q4_windows,
            &q6_windows,
            arena,
            &scalars,
            &rope_args,
            &cache_copy_args,
            &short_seq_len,
            &long_mha_args,
            false,
            replay_fattn,
            Self::DIRECT_Q6_V_STAGE + 1,
            None,
        )?;
        let probe = std::env::var("HAWKING_LLAMA_RESIDENT_REPLAY_PROBE_STAGES")
            .ok()
            .filter(|value| !value.is_empty())
            .map(|value| {
                value.parse::<usize>().map_err(|_| {
                    Error::Model(format!(
                        "HAWKING_LLAMA_RESIDENT_REPLAY_PROBE_STAGES must be an integer; got {value:?}"
                    ))
                })
            })
            .transpose()?;
        let probe = match probe {
            Some(limit) => Some(Self::build_graph(
                ctx,
                cfg,
                layers,
                raw_head,
                model,
                &q4_windows,
                &q6_windows,
                arena,
                &scalars,
                &rope_args,
                &cache_copy_args,
                &short_seq_len,
                &long_mha_args,
                true,
                false,
                0,
                Some(limit),
            )?),
            None => None,
        };
        Ok(Self {
            short_front,
            short_middle,
            short_back,
            long_front,
            long_middle,
            long_back,
            probe,
            q4_windows,
            q6_windows,
            rope_args,
            cache_copy_args,
            short_seq_len,
            long_mha_args,
        })
    }

    fn rope_args_bytes(cfg: &LlamaConfig, pos: usize) -> Vec<u8> {
        let h = cfg.head_dim;
        let heads = cfg.n_heads;
        let mut bytes = vec![0u8; std::mem::size_of::<LlamaB9430ReplayRopeArgs>()];
        macro_rules! write_field {
            ($field:ident, $value:expr) => {{
                let value = $value;
                let offset = std::mem::offset_of!(LlamaB9430ReplayRopeArgs, $field);
                let destination = &mut bytes[offset..offset + std::mem::size_of_val(&value)];
                destination.copy_from_slice(unsafe {
                    std::slice::from_raw_parts(
                        (&value as *const _) as *const u8,
                        std::mem::size_of_val(&value),
                    )
                });
            }};
        }
        let bytes_per_head = (h * std::mem::size_of::<f32>()) as u64;
        write_field!(ne00, h as i32);
        write_field!(ne01, heads as i32);
        write_field!(ne02, 1_i32);
        write_field!(ne03, 1_i32);
        write_field!(nb00, std::mem::size_of::<f32>() as u64);
        write_field!(nb01, bytes_per_head);
        write_field!(nb02, bytes_per_head * heads as u64);
        write_field!(nb03, bytes_per_head * heads as u64);
        write_field!(ne0, h as i32);
        write_field!(ne1, heads as i32);
        write_field!(ne2, 1_i32);
        write_field!(ne3, 1_i32);
        write_field!(nb0, std::mem::size_of::<f32>() as u64);
        write_field!(nb1, bytes_per_head);
        write_field!(nb2, bytes_per_head * heads as u64);
        write_field!(nb3, bytes_per_head * heads as u64);
        write_field!(n_past, pos as i32);
        write_field!(n_dims, h as i32);
        write_field!(n_ctx_orig, 131_072_i32);
        write_field!(freq_base, cfg.rope_theta);
        write_field!(freq_scale, 1.0_f32);
        write_field!(ext_factor, 0.0_f32);
        write_field!(attn_factor, 1.0_f32);
        write_field!(beta_fast, 32.0_f32);
        write_field!(beta_slow, 1.0_f32);
        write_field!(sect_0, 0_i32);
        write_field!(sect_1, 0_i32);
        write_field!(sect_2, 0_i32);
        write_field!(sect_3, 0_i32);
        write_field!(src2, 1_u8);
        bytes
    }

    fn update(&self, cfg: &LlamaConfig, pos: usize) -> Result<()> {
        // Retain all zero-offset ICB windows for the full replay lifetime.
        let _window_keepalive = (&self.q4_windows, &self.q6_windows);
        let seq_len = pos
            .checked_add(1)
            .ok_or_else(|| Error::Model("Llama replay sequence length overflow".into()))?;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let dst_off = pos
            .checked_mul(kv_dim)
            .ok_or_else(|| Error::Model("Llama replay KV offset overflow".into()))?;
        let copy = LlamaB9430ReplayCopyArgs {
            n: kv_dim as u32,
            src_off: 0,
            dst_off: dst_off as u32,
        };
        let mha = LlamaB9430ReplayMhaArgs {
            seq_len: seq_len as u32,
            head_dim: cfg.head_dim as u32,
            n_kv_heads: cfg.n_kv_heads as u32,
            group_size: (cfg.n_heads / cfg.n_kv_heads) as u32,
            scale: 1.0_f32 / (cfg.head_dim as f32).sqrt(),
        };
        MetalContext::write_buffer_bytes(&self.rope_args, &Self::rope_args_bytes(cfg, pos));
        MetalContext::write_buffer_bytes(&self.cache_copy_args, unsafe {
            std::slice::from_raw_parts(
                (&copy as *const LlamaB9430ReplayCopyArgs).cast::<u8>(),
                std::mem::size_of::<LlamaB9430ReplayCopyArgs>(),
            )
        });
        MetalContext::write_buffer_bytes(&self.short_seq_len, &(seq_len as u32).to_ne_bytes());
        MetalContext::write_buffer_bytes(&self.long_mha_args, unsafe {
            std::slice::from_raw_parts(
                (&mha as *const LlamaB9430ReplayMhaArgs).cast::<u8>(),
                std::mem::size_of::<LlamaB9430ReplayMhaArgs>(),
            )
        });
        Ok(())
    }
}

/// Persistent device-resident working set for the strict Llama b9430 decode
/// lane. It is lazy: the normal hybrid K0 lane allocates none of these buffers
/// unless the resident experiment is explicitly selected.
#[cfg(target_os = "macos")]
struct LlamaB9430ResidentArena {
    max_seq: usize,
    x: crate::metal::PinnedBuffer,
    attn_norm_out: crate::metal::PinnedBuffer,
    ffn_norm_out: crate::metal::PinnedBuffer,
    q_raw: crate::metal::PinnedBuffer,
    q_rope: crate::metal::PinnedBuffer,
    k_raw: crate::metal::PinnedBuffer,
    k_rope: crate::metal::PinnedBuffer,
    v: crate::metal::PinnedBuffer,
    attn_out: crate::metal::PinnedBuffer,
    o: crate::metal::PinnedBuffer,
    gate: crate::metal::PinnedBuffer,
    up: crate::metal::PinnedBuffer,
    act: crate::metal::PinnedBuffer,
    ffn_out: crate::metal::PinnedBuffer,
    logits: crate::metal::PinnedBuffer,
    attn_norm_weights: Vec<crate::metal::PinnedBuffer>,
    ffn_norm_weights: Vec<crate::metal::PinnedBuffer>,
    final_norm_weight: crate::metal::PinnedBuffer,
    rope_factors: crate::metal::PinnedBuffer,
    position: crate::metal::PinnedBuffer,
    keys_f32: Vec<crate::metal::PinnedBuffer>,
    values_f32: Vec<crate::metal::PinnedBuffer>,
    keys_f16: Vec<crate::metal::PinnedBuffer>,
    values_f16: Vec<crate::metal::PinnedBuffer>,
    /// GGML b9430 long FlashAttention partials: [head][head_dim][32] plus
    /// [head][64] online-softmax S/M values. Reused by every layer.
    fattn_scratch: crate::metal::PinnedBuffer,
    /// K6 packed-prefill workspace. These are batch-major B×tensor rows, but
    /// deliberately share the resident `keys_f16` / `values_f16` cache above:
    /// prompt chunks therefore become the exact cache consumed by normal
    /// source-FATTN decode without a second KV representation or CPU copy.
    prefill_max_batch: usize,
    prefill_x: crate::metal::PinnedBuffer,
    prefill_x_norm: crate::metal::PinnedBuffer,
    prefill_q: crate::metal::PinnedBuffer,
    prefill_k: crate::metal::PinnedBuffer,
    prefill_v: crate::metal::PinnedBuffer,
    prefill_attn: crate::metal::PinnedBuffer,
    prefill_o: crate::metal::PinnedBuffer,
    prefill_gate: crate::metal::PinnedBuffer,
    prefill_up: crate::metal::PinnedBuffer,
    prefill_act: crate::metal::PinnedBuffer,
    prefill_down: crate::metal::PinnedBuffer,
    prefill_positions: crate::metal::PinnedBuffer,
    /// One source-FATTN partial plane per prompt item.
    prefill_fattn_scratch: crate::metal::PinnedBuffer,
    replay: Option<LlamaB9430Replay>,
}

#[cfg(target_os = "macos")]
impl LlamaB9430ResidentArena {
    fn new(
        ctx: &MetalContext,
        cfg: &LlamaConfig,
        layers: &[LlamaLayer],
        final_norm: &[f32],
        max_seq: usize,
    ) -> Self {
        let f32_bytes = |n: usize| n * std::mem::size_of::<f32>();
        let hidden = cfg.hidden;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let cache_f32_bytes = f32_bytes(max_seq * kv_dim);
        let cache_f16_bytes = max_seq * kv_dim * std::mem::size_of::<f16>();
        let fattn_scratch_f32 = cfg
            .n_heads
            .checked_mul(cfg.head_dim * 32 + 64)
            .expect("Llama b9430 FlashAttention scratch size overflow");
        // The packed Q4 GEMM supports up to 32 activation rows. Keep the
        // resident default modest; the allocation is tiny beside model/KV
        // storage and this independent knob prevents Qwen's prefill tuning
        // from changing the Llama execution contract.
        let prefill_max_batch = crate::env_usize("HAWKING_LLAMA_PREFILL_BATCH", 8).clamp(2, 32);
        let prefill_fattn_scratch_f32 = prefill_max_batch
            .checked_mul(fattn_scratch_f32)
            .expect("Llama packed-prefill FlashAttention scratch size overflow");
        let pairs = cfg.head_dim / 2;
        let factors = cfg
            .rope_freq_factors
            .as_deref()
            .filter(|factors| factors.len() == pairs)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| vec![1.0; pairs]);
        Self {
            max_seq,
            x: ctx.new_buffer(f32_bytes(hidden)),
            attn_norm_out: ctx.new_buffer(f32_bytes(hidden)),
            ffn_norm_out: ctx.new_buffer(f32_bytes(hidden)),
            q_raw: ctx.new_buffer(f32_bytes(q_dim)),
            q_rope: ctx.new_buffer(f32_bytes(q_dim)),
            k_raw: ctx.new_buffer(f32_bytes(kv_dim)),
            k_rope: ctx.new_buffer(f32_bytes(kv_dim)),
            v: ctx.new_buffer(f32_bytes(kv_dim)),
            attn_out: ctx.new_buffer(f32_bytes(q_dim)),
            o: ctx.new_buffer(f32_bytes(hidden)),
            gate: ctx.new_buffer(f32_bytes(cfg.intermediate)),
            up: ctx.new_buffer(f32_bytes(cfg.intermediate)),
            act: ctx.new_buffer(f32_bytes(cfg.intermediate)),
            ffn_out: ctx.new_buffer(f32_bytes(hidden)),
            logits: ctx.new_buffer(f32_bytes(cfg.vocab_size)),
            attn_norm_weights: layers
                .iter()
                .map(|layer| {
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&layer.attn_norm))
                })
                .collect(),
            ffn_norm_weights: layers
                .iter()
                .map(|layer| {
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&layer.ffn_norm))
                })
                .collect(),
            final_norm_weight: ctx
                .new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(final_norm)),
            rope_factors: ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&factors)),
            position: ctx.new_buffer(std::mem::size_of::<i32>()),
            keys_f32: (0..cfg.n_layers)
                .map(|_| ctx.new_buffer(cache_f32_bytes))
                .collect(),
            values_f32: (0..cfg.n_layers)
                .map(|_| ctx.new_buffer(cache_f32_bytes))
                .collect(),
            keys_f16: (0..cfg.n_layers)
                .map(|_| ctx.new_buffer(cache_f16_bytes))
                .collect(),
            values_f16: (0..cfg.n_layers)
                .map(|_| ctx.new_buffer(cache_f16_bytes))
                .collect(),
            fattn_scratch: ctx.new_buffer(f32_bytes(fattn_scratch_f32)),
            prefill_max_batch,
            prefill_x: ctx.new_buffer(f32_bytes(prefill_max_batch * hidden)),
            prefill_x_norm: ctx.new_buffer(f32_bytes(prefill_max_batch * hidden)),
            prefill_q: ctx.new_buffer(f32_bytes(prefill_max_batch * q_dim)),
            prefill_k: ctx.new_buffer(f32_bytes(prefill_max_batch * kv_dim)),
            prefill_v: ctx.new_buffer(f32_bytes(prefill_max_batch * kv_dim)),
            prefill_attn: ctx.new_buffer(f32_bytes(prefill_max_batch * q_dim)),
            prefill_o: ctx.new_buffer(f32_bytes(prefill_max_batch * hidden)),
            prefill_gate: ctx.new_buffer(f32_bytes(prefill_max_batch * cfg.intermediate)),
            prefill_up: ctx.new_buffer(f32_bytes(prefill_max_batch * cfg.intermediate)),
            prefill_act: ctx.new_buffer(f32_bytes(prefill_max_batch * cfg.intermediate)),
            prefill_down: ctx.new_buffer(f32_bytes(prefill_max_batch * hidden)),
            prefill_positions: ctx.new_buffer(prefill_max_batch * std::mem::size_of::<i32>()),
            prefill_fattn_scratch: ctx.new_buffer(f32_bytes(prefill_fattn_scratch_f32)),
            replay: None,
        }
    }
}

pub struct LlamaDense {
    pub config: LlamaConfig,
    pub tokenizer: Tokenizer,
    pub model_id: String,

    /// Whole-GGUF no-copy Metal buffer. It is declared before `gguf` so the
    /// Metal view releases before the mmap backing it during model teardown.
    /// Every raw Q4_K/Q6_K projection binds its tensor offset into this one
    /// buffer, rather than uploading its weights for each decoded token.
    pub weights_mmap_buf: Option<crate::metal::PinnedBuffer>,
    /// Must precede `gguf`: replay graphs can retain zero-copy tensor windows
    /// whose backing mmap must stay valid until every captured graph is gone.
    #[cfg(target_os = "macos")]
    resident_arena: Option<LlamaB9430ResidentArena>,
    /// mmap keepalive (every TensorRef points into this).
    pub gguf: GgufFile,

    /// Exact f32 embedding rows for the CPU K0 authority path. Converting a
    /// quantized embedding table to f16 before layer 0 changes its RMSNorm
    /// enough to alter the BOS distribution on Q4_K_M Llama-3.1-8B.
    pub embed: Vec<f32>,
    /// Only needed when a tied-output Llama takes the Metal f16 LM-head
    /// path. Untied models keep this optional table empty.
    pub embed_f16_for_metal: Option<Vec<f16>>,
    pub final_norm: Vec<f32>,
    /// `None` ⇒ tied to embed (Llama-3.2-1B is tied; larger Llama-3
    /// variants typically ship an explicit `output.weight`).
    pub lm_head: Option<Vec<f16>>,
    /// Original GGUF output tensor retained for the diagnostic raw-quant
    /// LM-head route. Unlike `lm_head`, this preserves the source Q4_K/Q6_K
    /// representation and lets the native Metal authority kernel operate on
    /// the exact on-disk blocks.
    pub lm_head_raw: Option<TensorRef>,
    /// Exact f32 output rows for CPU K0. The Metal f16 path remains a
    /// separate execution candidate and cannot be called K0 until it has its
    /// own numerical proof.
    pub lm_head_f32: Option<Vec<f32>>,
    pub layers: Vec<LlamaLayer>,

    pub kv: KvCache,
    pub sampler: Sampler,
    pub kernel_profile: Option<KernelProfile>,
    pub _weights_path: PathBuf,
    /// `Some` when a Metal device is available. The hybrid forward path
    /// routes Q4_K projections, the f16 LM head, and rmsnorm through
    /// Metal kernels; everything else (Q6_K weights, attention) stays on
    /// the CPU reference path. The full TCB pinned-buffer + predec arena
    /// (Qwen's `forward_token_greedy_tcb`) is a follow-up.
    pub metal_ctx: Option<MetalContext>,
    /// Optional debug-only scalar checkpoint trace destination. This is never
    /// consulted by the normal decode lane, keeping production measurements
    /// free of trace I/O and reduction work.
    checkpoint_summary_path: Option<PathBuf>,
    /// Optional corpus trace directory.  Each `generate` call writes its own
    /// numbered summary here, letting a single loaded teacher capture many
    /// prompts without overwriting the previous prompt's evidence.
    checkpoint_summary_dir: Option<PathBuf>,
    checkpoint_capture_index: usize,
    /// Optional canonical surfaces such as `layer.17.v_raw`. A comma-separated
    /// selector captures a deliberately bounded paired trace for student
    /// evidence; one selector retains the legacy `debug_vector` JSON shape.
    checkpoint_vector_surfaces: Vec<String>,
    checkpoint_records: Vec<LlamaCheckpointRecord>,
    /// Trace-only execution accounting. The raw Llama path is still hybrid,
    /// so these counters make a CPU fallback impossible to mistake for a GPU
    /// parity run. Atomics are touched only when dispatch tracing is enabled.
    track_execution: bool,
    last_dispatch_count: AtomicUsize,
    /// Physical command buffers committed by the immediately preceding
    /// resident decode forward.  The hybrid path leaves this zero rather than
    /// inventing a complete-token graph count from incidental Metal kernels.
    last_command_buffer_count: AtomicUsize,
    last_cpu_reference_fallback_count: AtomicUsize,
}

impl LlamaDense {
    #[inline]
    fn checkpoint_enabled(&self) -> bool {
        self.checkpoint_summary_path.is_some() || self.checkpoint_summary_dir.is_some()
    }

    #[cfg(target_os = "macos")]
    fn resident_final_ffn_capture_requested(&self) -> bool {
        if !crate::env_on("HAWKING_LLAMA_RESIDENT_CAPTURE_UNSAFE_DIAGNOSTIC")
            || !self.checkpoint_enabled()
            || self.checkpoint_vector_surfaces.is_empty()
        {
            return false;
        }
        let final_layer = self.config.n_layers.saturating_sub(1);
        self.checkpoint_vector_surfaces.iter().all(|surface| {
            surface == &format!("layer.{final_layer}.ffn_norm")
                || surface == &format!("layer.{final_layer}.ffn_out")
        })
    }
    #[inline]
    fn capture_checkpoint_vector(
        &self,
        record: &mut Option<LlamaCheckpointRecord>,
        surface: String,
        values: &[f32],
    ) {
        if !self
            .checkpoint_vector_surfaces
            .iter()
            .any(|candidate| candidate == &surface)
        {
            return;
        }
        if let Some(record) = record.as_mut() {
            let vector = LlamaCheckpointVector {
                surface,
                values: values.to_vec(),
            };
            if self.checkpoint_vector_surfaces.len() == 1 {
                record.debug_vector = Some(vector);
            } else {
                record.debug_vectors.push(vector);
            }
        }
    }

    #[inline]
    fn record_dispatch(&self) {
        if self.track_execution {
            self.last_dispatch_count.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[inline]
    fn record_cpu_fallback(&self) {
        if self.track_execution {
            self.last_cpu_reference_fallback_count
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    fn dequant_ref_into(&self, t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
        dequant_ref_into(&self.gguf.mmap, t, buf)
    }

    fn rmsnorm_dispatch(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        if let Some(ctx) = &self.metal_ctx {
            self.record_dispatch();
            return crate::kernels::rmsnorm_llama_b9430(ctx, x, weight, eps, out);
        }
        self.record_cpu_fallback();
        rmsnorm_dispatch(None, x, weight, eps, out)
    }

    /// Per-layer matmul dispatcher. On macOS with Metal alive, raw Q4_K and
    /// Q6_K weights use their separate llama.cpp b9430 authority kernels.
    /// Other types and the off-macOS path retain the CPU reference fallback.
    fn matmul_q4_dispatch(
        &self,
        t: &TensorRef,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
        scratch: &mut Vec<f32>,
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            let pinned_weights = self.weights_mmap_buf.as_ref();
            if t.dtype == GgmlType::Q4_K {
                self.record_dispatch();
                if let Some(model_buf) = pinned_weights {
                    return crate::kernels::gemv_q4_k_m_llama_b9430_pinned(
                        ctx,
                        model_buf,
                        t.offset,
                        t.byte_size,
                        rows,
                        cols,
                        x,
                        out,
                    );
                }
                let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
                return crate::kernels::gemv_q4_k_m_llama_b9430(ctx, bytes, rows, cols, x, out);
            }
            if t.dtype == GgmlType::Q6_K {
                self.record_dispatch();
                if let Some(model_buf) = pinned_weights {
                    return crate::kernels::gemv_q6_k_llama_b9430_pinned(
                        ctx,
                        model_buf,
                        t.offset,
                        t.byte_size,
                        rows,
                        cols,
                        x,
                        out,
                    );
                }
                let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
                return crate::kernels::gemv_q6_k_llama_b9430(ctx, bytes, rows, cols, x, out);
            }
            if t.dtype == GgmlType::Q5_K {
                if let Some(model_buf) = pinned_weights {
                    self.record_dispatch();
                    return crate::kernels::gemv_q5_k_serial_authority_pinned(
                        ctx,
                        model_buf,
                        t.offset,
                        t.byte_size,
                        rows,
                        cols,
                        x,
                        out,
                    );
                }
            }
        }
        self.record_cpu_fallback();
        if crate::env_on("HAWKING_LLAMA_KQ8_AUTHORITY") {
            let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
            match t.dtype {
                GgmlType::Q4_K => return crate::quant::gemv_q4_k_q8k(bytes, rows, cols, x, out),
                GgmlType::Q6_K => return crate::quant::gemv_q6_k_q8k(bytes, rows, cols, x, out),
                _ => {}
            }
        }
        self.dequant_ref_into(t, scratch)?;
        gemv_f32(scratch, rows, cols, x, out);
        Ok(())
    }

    fn gemv_f16_dispatch(
        &self,
        w_f16: &[f16],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        if self.metal_ctx.is_some() {
            self.record_dispatch();
        } else {
            self.record_cpu_fallback();
        }
        gemv_f16_dispatch(self.metal_ctx.as_ref(), w_f16, rows, cols, x, out)
    }

    fn lm_head_dispatch(&self, rows: usize, cols: usize, x: &[f32], out: &mut [f32]) -> Result<()> {
        // Llama K0 must preserve the Q6_K/Q4_K source blocks in
        // `output.weight`; an f16 materialization cannot be bit-identical.
        // This is Hawking's own native Metal GEMV, not an adapter or CPU
        // fallback. Models without a supported raw output tensor retain the
        // established f16 path below.
        if let (Some(_), Some(raw)) = (self.metal_ctx.as_ref(), self.lm_head_raw.as_ref()) {
            if matches!(raw.dtype, GgmlType::Q4_K | GgmlType::Q6_K) {
                self.record_dispatch();
                let mut scratch = Vec::new();
                return self.matmul_q4_dispatch(raw, rows, cols, x, out, &mut scratch);
            }
        }
        if self.metal_ctx.is_some() {
            let w_f16 = self
                .lm_head
                .as_deref()
                .or(self.embed_f16_for_metal.as_deref())
                .ok_or_else(|| Error::Model("missing f16 LM head for Metal dispatch".into()))?;
            if crate::env_on("HAWKING_LLAMA_GGML_LM_HEAD_AUTHORITY") {
                // Diagnostic-only GPU authority adapter. It determines
                // whether the final f16 LM-head reduction is the remaining
                // Llama K0 blocker; it is never eligible for performance or
                // parity promotion.
                self.record_dispatch();
                return crate::kernels::gemv_f16_ggml_authority(w_f16, rows, cols, x, out);
            }
            return self.gemv_f16_dispatch(w_f16, rows, cols, x, out);
        }
        // The CPU K0 authority must multiply the original dequantized f32
        // values, not a second f16-rounding of the quantized GGUF weights.
        self.record_cpu_fallback();
        let w_f32 = self.lm_head_f32.as_deref().unwrap_or(&self.embed);
        gemv_f32(w_f32, rows, cols, x, out);
        Ok(())
    }

    /// Encode a resident Q4_K projection with the selected geometry.
    ///
    /// The strict b9430 four-row grammar remains the authority default.  The
    /// v3-dual geometry is a single, explicitly preregistered candidate: it
    /// doubles rows per threadgroup while preserving the packed source window
    /// and is admitted only through complete-token parity and wall-time gates.
    #[cfg(target_os = "macos")]
    #[allow(clippy::too_many_arguments)]
    fn resident_q4_tcb(
        tcb: &mut TokenCommandBuffer<'_>,
        model_buf: &crate::metal::PinnedBuffer,
        w_offset: usize,
        w_byte_size: usize,
        rows: usize,
        cols: usize,
        x_buf: &crate::metal::PinnedBuffer,
        out_buf: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_GEOM_V3_DUAL") {
            crate::kernels::gemv_q4_k_m_v3_dual_pinned_tcb(
                tcb,
                model_buf,
                w_offset,
                w_byte_size,
                rows,
                cols,
                x_buf,
                out_buf,
            )
        } else {
            crate::kernels::gemv_q4_k_m_llama_b9430_pinned_tcb(
                tcb,
                model_buf,
                w_offset,
                w_byte_size,
                rows,
                cols,
                x_buf,
                out_buf,
            )
        }
    }

    /// Opt-in resident strict-b9430 decoder. This is deliberately kept out of
    /// checkpoint capture until its own long-context receipt is green: it
    /// removes all intermediate CPU visibility, but uses the same source
    /// Q4_K/Q6_K, RMSNorm, RoPE, SwiGLU, f16-cache and attention kernels.
    #[cfg(target_os = "macos")]
    fn forward_token_resident_b9430(&mut self, token: u32, pos: usize) -> Result<Option<Vec<f32>>> {
        let requested = crate::env_on("HAWKING_LLAMA_RESIDENT_B9430");
        let initial_gate = requested
            && ((!self.checkpoint_enabled() && self.checkpoint_vector_surfaces.is_empty())
                || self.resident_final_ffn_capture_requested())
            && self.config.rope_scaling.is_none()
            && pos == self.kv.seq_len;
        if !initial_gate {
            if requested && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
                eprintln!(
                    "[llama-resident] initial gate: checkpoint={} vector={} rope_scaling={} pos={} kv_seq={}",
                    self.checkpoint_enabled(),
                    !self.checkpoint_vector_surfaces.is_empty(),
                    self.config.rope_scaling.is_some(),
                    pos,
                    self.kv.seq_len,
                );
            }
            return Ok(None);
        }
        let Some(ctx) = self.metal_ctx.as_ref() else {
            if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
                eprintln!("[llama-resident] rejected: no Metal context");
            }
            return Ok(None);
        };
        let Some(model_buf) = self.weights_mmap_buf.as_ref() else {
            if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
                eprintln!("[llama-resident] rejected: no pinned GGUF buffer");
            }
            return Ok(None);
        };
        let Some(raw_head) = self.lm_head_raw.as_ref() else {
            if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
                eprintln!("[llama-resident] rejected: no raw output tensor");
            }
            return Ok(None);
        };
        if !matches!(raw_head.dtype, GgmlType::Q4_K | GgmlType::Q6_K) {
            if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
                eprintln!(
                    "[llama-resident] rejected: unsupported output dtype {:?}",
                    raw_head.dtype
                );
            }
            return Ok(None);
        }
        if let Some((bad_layer, layer)) = self.layers.iter().enumerate().find(|(_, layer)| {
            layer.q_proj.dtype != GgmlType::Q4_K
                || layer.k_proj.dtype != GgmlType::Q4_K
                || !matches!(
                    layer.v_proj.dtype,
                    GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
                )
                || layer.o_proj.dtype != GgmlType::Q4_K
                || layer.ffn_gate.dtype != GgmlType::Q4_K
                || layer.ffn_up.dtype != GgmlType::Q4_K
                || !matches!(layer.ffn_down.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
        }) {
            if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
                eprintln!(
                    "[llama-resident] rejected projection dtypes at layer {bad_layer}: q={:?} k={:?} v={:?} o={:?} gate={:?} up={:?} down={:?}",
                    layer.q_proj.dtype,
                    layer.k_proj.dtype,
                    layer.v_proj.dtype,
                    layer.o_proj.dtype,
                    layer.ffn_gate.dtype,
                    layer.ffn_up.dtype,
                    layer.ffn_down.dtype,
                );
            }
            return Ok(None);
        }
        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!(
                "kv cache full at {}",
                self.kv.seq_len
            )));
        }
        if self.resident_arena.is_none() {
            self.resident_arena = Some(LlamaB9430ResidentArena::new(
                ctx,
                &self.config,
                &self.layers,
                &self.final_norm,
                self.kv.max_seq,
            ));
        }
        let replay_requested = crate::env_on("HAWKING_LLAMA_RESIDENT_REPLAY");
        // The capture path is retained as a diagnostic substrate, but an
        // end-to-end P8 receipt exposed an ICB-only Q6/RoPE corruption on this
        // device. It remains fail-closed in every normal execution path. The
        // second, deliberately named flag exists only to let a repair campaign
        // reproduce the failure and collect a fresh numerical receipt.
        let replay_enabled =
            replay_requested && crate::env_on("HAWKING_LLAMA_RESIDENT_REPLAY_UNSAFE_DIAGNOSTIC");
        if replay_requested && !replay_enabled && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG")
        {
            eprintln!(
                "[llama-resident] replay is quarantined: the complete-token parity receipt is not green",
            );
        }
        if replay_enabled {
            let arena = self
                .resident_arena
                .as_mut()
                .expect("resident arena initialized above");
            if arena.replay.is_none() {
                arena.replay = Some(LlamaB9430Replay::new(
                    ctx,
                    &self.config,
                    &self.layers,
                    raw_head,
                    model_buf,
                    &self.gguf.mmap,
                    arena,
                )?);
            }
            arena
                .replay
                .as_ref()
                .expect("replay initialized above")
                .update(&self.config, pos)?;
        }
        let arena = self
            .resident_arena
            .as_ref()
            .expect("resident arena initialized above");
        let cfg = &self.config;
        let h = cfg.hidden;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let seq_len = pos + 1;
        let kv_off = pos * kv_dim;
        let token_start = (token as usize)
            .checked_mul(h)
            .ok_or_else(|| Error::Model("embedding offset overflow".into()))?;
        let token_end = token_start
            .checked_add(h)
            .ok_or_else(|| Error::Model("embedding end overflow".into()))?;
        let embedding = self
            .embed
            .get(token_start..token_end)
            .ok_or_else(|| Error::Model(format!("token id {token} outside embedding table")))?;
        MetalContext::write_buffer_bytes(&arena.x, bytemuck::cast_slice::<f32, u8>(embedding));
        let position =
            i32::try_from(pos).map_err(|_| Error::Model(format!("position {pos} exceeds i32")))?;
        MetalContext::write_buffer_bytes(&arena.position, &position.to_ne_bytes());

        let mut tcb = TokenCommandBuffer::new(ctx);
        // The source-style f16 FlashAttention consumes only the f16 resident
        // cache, so it has no need for the two generic f32 cache blits below.
        // That also lets this diagnostic use one ordered compute encoder for
        // the full device-resident token.  Both levers remain opt-in until a
        // complete-token parity receipt promotes them.
        let source_fattn = crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN");
        let source_fattn_fused_kv =
            source_fattn && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN_FUSED_KV");
        let source_fattn_fused_kv_rope = source_fattn_fused_kv
            && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN_FUSED_KV_ROPE");
        let source_fattn_fused_qkv_rope = source_fattn_fused_kv_rope
            && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN_FUSED_QKV_ROPE");
        let serial_generic =
            !source_fattn && crate::env_on("HAWKING_LLAMA_RESIDENT_SERIAL_GENERIC");
        let serial_encoder = (source_fattn
            && crate::env_on("HAWKING_LLAMA_RESIDENT_SERIAL_ENCODER"))
            || serial_generic;
        // A replayable ICB graph owns its own encoder. Opening the direct
        // serial encoder first prevents Metal from executing the graph at all;
        // keep serial ordering for the direct lane, but let the replay lane
        // submit its captured graph directly under its existing diagnostic gate.
        if serial_encoder && !replay_enabled {
            tcb.begin_serial_group()?;
        }
        if replay_enabled {
            let replay = arena.replay.as_ref().expect("replay initialized above");
            if let Some(probe) = replay.probe.as_ref() {
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(probe)?;
            } else if seq_len <= 32 {
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(&replay.short_front)?;
                let layer = &self.layers[30];
                Self::resident_q4_tcb(
                    &mut tcb,
                    model_buf,
                    layer.q_proj.offset,
                    layer.q_proj.byte_size,
                    q_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.q_raw,
                )?;
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(&replay.short_middle)?;
                crate::kernels::gemv_q6_k_llama_b9430_pinned_tcb(
                    &mut tcb,
                    model_buf,
                    layer.v_proj.offset,
                    layer.v_proj.byte_size,
                    kv_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.v,
                )?;
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(&replay.short_back)?;
            } else {
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(&replay.long_front)?;
                let layer = &self.layers[30];
                crate::kernels::gemv_q4_k_m_llama_b9430_pinned_tcb(
                    &mut tcb,
                    model_buf,
                    layer.q_proj.offset,
                    layer.q_proj.byte_size,
                    q_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.q_raw,
                )?;
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(&replay.long_middle)?;
                crate::kernels::gemv_q6_k_llama_b9430_pinned_tcb(
                    &mut tcb,
                    model_buf,
                    layer.v_proj.offset,
                    layer.v_proj.byte_size,
                    kv_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.v,
                )?;
                tcb.end_concurrent_group()?;
                tcb.execute_replayable_graph(&replay.long_back)?;
            }
        } else {
            let fuse_residual_norm = crate::env_on("HAWKING_LLAMA_RESIDENT_FUSED_RESIDUAL_NORM");
            let fuse_gate_up = crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_FUSED_GATE_UP");
            for (li, layer) in self.layers.iter().enumerate() {
                if !fuse_residual_norm || li == 0 {
                    crate::kernels::rmsnorm_llama_b9430_tcb(
                        &mut tcb,
                        &arena.x,
                        &arena.attn_norm_weights[li],
                        cfg.rms_norm_eps,
                        h,
                        &arena.attn_norm_out,
                    )?;
                }
                let concurrent_projections =
                    !serial_encoder && crate::env_on("HAWKING_LLAMA_RESIDENT_CONCURRENT");
                if concurrent_projections {
                    tcb.begin_concurrent_group()?;
                }
                crate::kernels::gemv_q4_k_m_llama_b9430_pinned_tcb(
                    &mut tcb,
                    model_buf,
                    layer.q_proj.offset,
                    layer.q_proj.byte_size,
                    q_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.q_raw,
                )?;
                Self::resident_q4_tcb(
                    &mut tcb,
                    model_buf,
                    layer.k_proj.offset,
                    layer.k_proj.byte_size,
                    kv_dim,
                    h,
                    &arena.attn_norm_out,
                    &arena.k_raw,
                )?;
                match layer.v_proj.dtype {
                    GgmlType::Q4_K => Self::resident_q4_tcb(
                        &mut tcb,
                        model_buf,
                        layer.v_proj.offset,
                        layer.v_proj.byte_size,
                        kv_dim,
                        h,
                        &arena.attn_norm_out,
                        &arena.v,
                    )?,
                    GgmlType::Q6_K => crate::kernels::gemv_q6_k_llama_b9430_pinned_tcb(
                        &mut tcb,
                        model_buf,
                        layer.v_proj.offset,
                        layer.v_proj.byte_size,
                        kv_dim,
                        h,
                        &arena.attn_norm_out,
                        &arena.v,
                    )?,
                    // Selected value projections in common Q4_K_M releases
                    // are Q5_K. Keep this packed GEMV in the resident command
                    // buffer rather than silently demoting the entire token to
                    // the hybrid execution path.
                    GgmlType::Q5_K => crate::kernels::gemv_q5_k_serial_authority_pinned_tcb(
                        &mut tcb,
                        model_buf,
                        layer.v_proj.offset,
                        layer.v_proj.byte_size,
                        kv_dim,
                        h,
                        &arena.attn_norm_out,
                        &arena.v,
                    )?,
                    _ => return Ok(None),
                }
                if concurrent_projections {
                    tcb.end_concurrent_group()?;
                }
                if source_fattn_fused_qkv_rope {
                    crate::kernels::rope_norm_llama_b9430_qkv_cache_f16_tcb(
                        &mut tcb,
                        &arena.q_raw,
                        &arena.k_raw,
                        &arena.v,
                        &arena.q_rope,
                        &arena.keys_f16[li],
                        &arena.values_f16[li],
                        &arena.position,
                        &arena.rope_factors,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        pos as u32,
                        cfg.rope_theta,
                        kv_off,
                    )?;
                } else {
                    crate::kernels::rope_norm_llama_b9430_tcb(
                        &mut tcb,
                        &arena.q_raw,
                        &arena.q_rope,
                        &arena.position,
                        &arena.rope_factors,
                        cfg.n_heads,
                        cfg.head_dim,
                        pos as u32,
                        cfg.rope_theta,
                    )?;
                }
                if source_fattn_fused_qkv_rope {
                    // Q-RoPE, K-RoPE, and f16 K/V cache append were fused above.
                } else if source_fattn_fused_kv_rope {
                    crate::kernels::rope_norm_llama_b9430_cache_kv_f16_tcb(
                        &mut tcb,
                        &arena.k_raw,
                        &arena.v,
                        &arena.keys_f16[li],
                        &arena.values_f16[li],
                        &arena.position,
                        &arena.rope_factors,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        pos as u32,
                        cfg.rope_theta,
                        kv_off,
                    )?;
                } else {
                    crate::kernels::rope_norm_llama_b9430_tcb(
                        &mut tcb,
                        &arena.k_raw,
                        &arena.k_rope,
                        &arena.position,
                        &arena.rope_factors,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        pos as u32,
                        cfg.rope_theta,
                    )?;
                }
                // Mirror the GGML SET_ROWS f16 storage seam before either cache
                // representation consumes K/V. The f32 cache is its exact
                // f16-rounded expansion for the generic >32-token kernel.
                if !source_fattn_fused_kv {
                    crate::kernels::round_f16_llama_b9430_tcb(&mut tcb, &arena.k_rope, kv_dim)?;
                    crate::kernels::round_f16_llama_b9430_tcb(&mut tcb, &arena.v, kv_dim)?;
                }
                if source_fattn_fused_qkv_rope || source_fattn_fused_kv_rope {
                    // K-RoPE and both f16 cache writes were fused immediately
                    // above; no intermediate K vector is materialized in this lane.
                } else if source_fattn_fused_kv {
                    crate::kernels::llama_b9430_cache_append_kv_f16_tcb(
                        &mut tcb,
                        &arena.k_rope,
                        &arena.v,
                        &arena.keys_f16[li],
                        &arena.values_f16[li],
                        kv_off,
                        kv_dim,
                    )?;
                } else if serial_generic {
                    crate::kernels::llama_b9430_cache_append_f32_f16_tcb(
                        &mut tcb,
                        &arena.k_rope,
                        &arena.keys_f32[li],
                        &arena.keys_f16[li],
                        0,
                        kv_off,
                        kv_dim,
                    )?;
                    crate::kernels::llama_b9430_cache_append_f32_f16_tcb(
                        &mut tcb,
                        &arena.v,
                        &arena.values_f32[li],
                        &arena.values_f16[li],
                        0,
                        kv_off,
                        kv_dim,
                    )?;
                } else if !source_fattn {
                    tcb.copy_buffer_bytes(
                        &arena.k_rope,
                        0,
                        &arena.keys_f32[li],
                        (kv_off * std::mem::size_of::<f32>()) as u64,
                        (kv_dim * std::mem::size_of::<f32>()) as u64,
                    )?;
                    tcb.copy_buffer_bytes(
                        &arena.v,
                        0,
                        &arena.values_f32[li],
                        (kv_off * std::mem::size_of::<f32>()) as u64,
                        (kv_dim * std::mem::size_of::<f32>()) as u64,
                    )?;
                }
                if !serial_generic && !source_fattn_fused_kv {
                    crate::kernels::memcpy_f32_to_f16_off_tcb(
                        &mut tcb,
                        &arena.k_rope,
                        &arena.keys_f16[li],
                        0,
                        kv_off,
                        kv_dim,
                    )?;
                    crate::kernels::memcpy_f32_to_f16_off_tcb(
                        &mut tcb,
                        &arena.v,
                        &arena.values_f16[li],
                        0,
                        kv_off,
                        kv_dim,
                    )?;
                }
                if seq_len <= 32 && cfg.head_dim == 128 {
                    crate::kernels::mha_decode_llama_b9430_short_tcb(
                        &mut tcb,
                        &arena.q_rope,
                        &arena.keys_f16[li],
                        &arena.values_f16[li],
                        &arena.attn_out,
                        seq_len,
                        cfg.head_dim,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                    )?;
                } else if source_fattn {
                    // Diagnostic-only B=1 bridge for the packed-prefill FATTN
                    // primitive. It validates its pipeline, cache layout, and
                    // source arithmetic against the exact decode lane before the
                    // layer-first B>1 graph selects it. Never enabled by default.
                    if crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_FATTN_PREFILL_PROBE") {
                        crate::kernels::mha_decode_llama_b9430_fattn_prefill_tcb(
                            &mut tcb,
                            &arena.q_rope,
                            &arena.keys_f16[li],
                            &arena.values_f16[li],
                            &arena.fattn_scratch,
                            &arena.attn_out,
                            seq_len - 1,
                            1,
                            cfg.head_dim,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                        )?;
                    } else {
                        crate::kernels::mha_decode_llama_b9430_fattn_tcb(
                            &mut tcb,
                            &arena.q_rope,
                            &arena.keys_f16[li],
                            &arena.values_f16[li],
                            &arena.fattn_scratch,
                            &arena.attn_out,
                            seq_len,
                            cfg.head_dim,
                            cfg.n_heads,
                            cfg.n_kv_heads,
                        )?;
                    }
                } else if crate::env_on("HAWKING_LLAMA_RESIDENT_FLASH_F16_ATTN") {
                    crate::kernels::mha_decode_flash_f16kv_tcb(
                        &mut tcb,
                        &arena.q_rope,
                        &arena.keys_f16[li],
                        0,
                        &arena.values_f16[li],
                        0,
                        &arena.attn_out,
                        seq_len,
                        cfg.head_dim,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                    )?;
                } else if crate::env_on("HAWKING_LLAMA_RESIDENT_FLASH_F32_ATTN") {
                    crate::kernels::mha_decode_flash_f32_tcb(
                        &mut tcb,
                        &arena.q_rope,
                        &arena.keys_f32[li],
                        0,
                        &arena.values_f32[li],
                        0,
                        &arena.attn_out,
                        seq_len,
                        cfg.head_dim,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                    )?;
                } else if crate::env_on("HAWKING_LLAMA_RESIDENT_F16_ATTN") {
                    // The resident cache retains the exact f16 SET_ROWS image;
                    // this long-context kernel consumes it directly. It remains
                    // an explicit candidate until the 64-token output hash and
                    // full-vector checkpoints promote it.
                    crate::kernels::mha_decode_f16kv_tcb(
                        &mut tcb,
                        &arena.q_rope,
                        &arena.keys_f16[li],
                        0,
                        &arena.values_f16[li],
                        0,
                        &arena.attn_out,
                        seq_len,
                        cfg.head_dim,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                    )?;
                } else {
                    crate::kernels::mha_decode_f32_tcb(
                        &mut tcb,
                        &arena.q_rope,
                        &arena.keys_f32[li],
                        0,
                        &arena.values_f32[li],
                        0,
                        &arena.attn_out,
                        seq_len,
                        cfg.head_dim,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                    )?;
                }
                Self::resident_q4_tcb(
                    &mut tcb,
                    model_buf,
                    layer.o_proj.offset,
                    layer.o_proj.byte_size,
                    h,
                    q_dim,
                    &arena.attn_out,
                    &arena.o,
                )?;
                if fuse_residual_norm {
                    crate::kernels::add_rmsnorm_llama_b9430_tcb(
                        &mut tcb,
                        &arena.x,
                        &arena.o,
                        &arena.ffn_norm_weights[li],
                        cfg.rms_norm_eps,
                        h,
                        &arena.ffn_norm_out,
                    )?;
                } else {
                    crate::kernels::add_inplace_metal_tcb(&mut tcb, &arena.x, &arena.o, h)?;
                    crate::kernels::rmsnorm_llama_b9430_tcb(
                        &mut tcb,
                        &arena.x,
                        &arena.ffn_norm_weights[li],
                        cfg.rms_norm_eps,
                        h,
                        &arena.ffn_norm_out,
                    )?;
                }
                if fuse_gate_up {
                    crate::kernels::gemv_q4_k_m_llama_b9430_pair_pinned_tcb(
                        &mut tcb,
                        model_buf,
                        layer.ffn_gate.offset,
                        layer.ffn_gate.byte_size,
                        model_buf,
                        layer.ffn_up.offset,
                        layer.ffn_up.byte_size,
                        cfg.intermediate,
                        h,
                        &arena.ffn_norm_out,
                        &arena.gate,
                        &arena.up,
                    )?;
                } else {
                    if concurrent_projections {
                        tcb.begin_concurrent_group()?;
                    }
                    Self::resident_q4_tcb(
                        &mut tcb,
                        model_buf,
                        layer.ffn_gate.offset,
                        layer.ffn_gate.byte_size,
                        cfg.intermediate,
                        h,
                        &arena.ffn_norm_out,
                        &arena.gate,
                    )?;
                    Self::resident_q4_tcb(
                        &mut tcb,
                        model_buf,
                        layer.ffn_up.offset,
                        layer.ffn_up.byte_size,
                        cfg.intermediate,
                        h,
                        &arena.ffn_norm_out,
                        &arena.up,
                    )?;
                    if concurrent_projections {
                        tcb.end_concurrent_group()?;
                    }
                }
                crate::kernels::swiglu_llama_b9430_tcb(
                    &mut tcb,
                    &arena.gate,
                    &arena.up,
                    &arena.act,
                    cfg.intermediate,
                )?;
                match layer.ffn_down.dtype {
                    GgmlType::Q4_K => Self::resident_q4_tcb(
                        &mut tcb,
                        model_buf,
                        layer.ffn_down.offset,
                        layer.ffn_down.byte_size,
                        h,
                        cfg.intermediate,
                        &arena.act,
                        &arena.ffn_out,
                    )?,
                    GgmlType::Q6_K => crate::kernels::gemv_q6_k_llama_b9430_pinned_tcb(
                        &mut tcb,
                        model_buf,
                        layer.ffn_down.offset,
                        layer.ffn_down.byte_size,
                        h,
                        cfg.intermediate,
                        &arena.act,
                        &arena.ffn_out,
                    )?,
                    _ => return Ok(None),
                }
                if fuse_residual_norm {
                    let next_weight = if li + 1 < self.layers.len() {
                        &arena.attn_norm_weights[li + 1]
                    } else {
                        &arena.final_norm_weight
                    };
                    crate::kernels::add_rmsnorm_llama_b9430_tcb(
                        &mut tcb,
                        &arena.x,
                        &arena.ffn_out,
                        next_weight,
                        cfg.rms_norm_eps,
                        h,
                        &arena.attn_norm_out,
                    )?;
                } else {
                    crate::kernels::add_inplace_metal_tcb(&mut tcb, &arena.x, &arena.ffn_out, h)?;
                }
            }
            if !fuse_residual_norm {
                crate::kernels::rmsnorm_llama_b9430_tcb(
                    &mut tcb,
                    &arena.x,
                    &arena.final_norm_weight,
                    cfg.rms_norm_eps,
                    h,
                    &arena.attn_norm_out,
                )?;
            }
            match raw_head.dtype {
                GgmlType::Q4_K => Self::resident_q4_tcb(
                    &mut tcb,
                    model_buf,
                    raw_head.offset,
                    raw_head.byte_size,
                    cfg.vocab_size,
                    h,
                    &arena.attn_norm_out,
                    &arena.logits,
                )?,
                GgmlType::Q6_K => crate::kernels::gemv_q6_k_llama_b9430_pinned_tcb(
                    &mut tcb,
                    model_buf,
                    raw_head.offset,
                    raw_head.byte_size,
                    cfg.vocab_size,
                    h,
                    &arena.attn_norm_out,
                    &arena.logits,
                )?,
                _ => return Ok(None),
            }
        }
        if serial_encoder && !replay_enabled {
            tcb.end_concurrent_group()?;
        }
        let dispatches = tcb.dispatch_count();
        tcb.commit_and_wait()?;
        if replay_enabled && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_DIAG") {
            let report = |name: &str, buffer: &crate::metal::PinnedBuffer, n: usize| {
                let values =
                    unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n) };
                let bad = values.iter().position(|value| !value.is_finite());
                eprintln!(
                    "[llama-resident] replay pos={pos} {name}: finite={}/{} first_bad={bad:?}",
                    values.iter().filter(|value| value.is_finite()).count(),
                    values.len(),
                );
            };
            report("x", &arena.x, h);
            report("final_norm", &arena.attn_norm_out, h);
            report("q_raw", &arena.q_raw, q_dim);
            report("k_raw", &arena.k_raw, kv_dim);
            report("q", &arena.q_rope, q_dim);
            report("k", &arena.k_rope, kv_dim);
            report("v", &arena.v, kv_dim);
            report("attn", &arena.attn_out, q_dim);
            report("ffn", &arena.ffn_out, h);
            report("logits", &arena.logits, cfg.vocab_size);
        }
        let resident_capture = self.resident_final_ffn_capture_requested().then(|| unsafe {
            (
                std::slice::from_raw_parts(arena.ffn_norm_out.contents() as *const f32, h).to_vec(),
                std::slice::from_raw_parts(arena.ffn_out.contents() as *const f32, h).to_vec(),
            )
        });
        self.kv.seq_len += 1;
        if self.track_execution {
            self.last_dispatch_count
                .store(dispatches, Ordering::Relaxed);
            self.last_command_buffer_count.store(1, Ordering::Relaxed);
            self.last_cpu_reference_fallback_count
                .store(0, Ordering::Relaxed);
        }
        let logits = unsafe {
            std::slice::from_raw_parts(arena.logits.contents() as *const f32, cfg.vocab_size)
                .to_vec()
        };
        if let Some((ffn_norm, ffn_out)) = resident_capture {
            let final_layer = cfg.n_layers.saturating_sub(1);
            let mut record = Some(LlamaCheckpointRecord {
                position: pos,
                token_id: token,
                embedding_sum: 0.0,
                layers: Vec::new(),
                final_norm_sum: 0.0,
                logits_sum: 0.0,
                greedy_token_id: 0,
                debug_vector: None,
                debug_vectors: Vec::new(),
                capture_kind: Some("resident_final_ffn_vectors_only"),
            });
            self.capture_checkpoint_vector(
                &mut record,
                format!("layer.{final_layer}.ffn_norm"),
                &ffn_norm,
            );
            self.capture_checkpoint_vector(
                &mut record,
                format!("layer.{final_layer}.ffn_out"),
                &ffn_out,
            );
            self.checkpoint_records
                .push(record.expect("resident capture record"));
        }
        if let Some(path) =
            std::env::var_os("HAWKING_LLAMA_RESIDENT_LOGITS_PATH").filter(|path| !path.is_empty())
        {
            let receipt = serde_json::json!({
                "schema": "hawking.tg.llama_resident_logits.v1",
                "position": pos,
                "token_id": token,
                "values": &logits,
            });
            let bytes = serde_json::to_vec(&receipt)
                .map_err(|err| Error::Model(format!("serialize resident logits receipt: {err}")))?;
            std::fs::write(path, bytes)?;
        }
        Ok(Some(logits))
    }

    /// K6: process a contiguous long-prompt chunk layer-first in the resident
    /// Llama arena. Q4 projections are true batch GEMMs; the source RoPE/f16
    /// cache image and source-FATTN causal limit remain per prompt position.
    /// The normal decoder consumes those same f16 cache planes immediately
    /// afterward, so this introduces neither a second KV format nor a CPU KV
    /// materialization.
    #[cfg(target_os = "macos")]
    fn prefill_tokens_resident_b9430_packed(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<bool> {
        if !crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_PACKED_PREFILL")
            || !crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_PACKED_PREFILL_UNSAFE_DIAGNOSTIC")
            || !crate::env_on("HAWKING_LLAMA_RESIDENT_B9430")
            || !crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_LONG_FATTN")
            || self.checkpoint_enabled()
            || !self.checkpoint_vector_surfaces.is_empty()
            || self.config.rope_scaling.is_some()
            || tokens.len() < 2
            // The source FATTN grammar is deliberately only the long-context
            // path. Retain the established <=32 short attention authority.
            || start_pos < 32
            || start_pos != self.kv.seq_len
        {
            return Ok(false);
        }
        let Some(ctx) = self.metal_ctx.as_ref() else {
            return Ok(false);
        };
        let Some(model_buf) = self.weights_mmap_buf.as_ref() else {
            return Ok(false);
        };
        if tokens.len() > 8 || start_pos + tokens.len() > self.kv.max_seq {
            return Ok(false);
        }
        if self.layers.iter().any(|layer| {
            !matches!(layer.q_proj.dtype, GgmlType::Q4_K)
                || !matches!(layer.k_proj.dtype, GgmlType::Q4_K)
                || !matches!(layer.v_proj.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
                || !matches!(layer.o_proj.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
                || !matches!(layer.ffn_gate.dtype, GgmlType::Q4_K)
                || !matches!(layer.ffn_up.dtype, GgmlType::Q4_K)
                || !matches!(layer.ffn_down.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
        }) {
            return Ok(false);
        }
        if self.resident_arena.is_none() {
            self.resident_arena = Some(LlamaB9430ResidentArena::new(
                ctx,
                &self.config,
                &self.layers,
                &self.final_norm,
                self.kv.max_seq,
            ));
        }
        let arena = self
            .resident_arena
            .as_ref()
            .expect("resident arena initialized");
        let b = tokens.len();
        if b > arena.prefill_max_batch {
            return Ok(false);
        }
        let cfg = &self.config;
        let h = cfg.hidden;
        let q_dim = cfg.n_heads * cfg.head_dim;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let mid = cfg.intermediate;
        let f32_bytes = std::mem::size_of::<f32>();
        let h_bytes = h * f32_bytes;
        let q_bytes = q_dim * f32_bytes;
        let kv_bytes = kv_dim * f32_bytes;
        let mid_bytes = mid * f32_bytes;

        // The embedded f32 authority table is already resident in process.
        // Copy only B activation rows into persistent shared storage; this is
        // a prefill-only ingress, not a per-decode allocation/readback.
        let x_ptr = arena.prefill_x.contents() as *mut f32;
        let pos_ptr = arena.prefill_positions.contents() as *mut i32;
        for (bi, &token) in tokens.iter().enumerate() {
            let begin = (token as usize)
                .checked_mul(h)
                .ok_or_else(|| Error::Model("packed prefill embedding offset overflow".into()))?;
            let embedding = self.embed.get(begin..begin + h).ok_or_else(|| {
                Error::Model(format!(
                    "packed prefill token id {token} outside embedding table"
                ))
            })?;
            unsafe {
                std::ptr::copy_nonoverlapping(embedding.as_ptr(), x_ptr.add(bi * h), h);
                *pos_ptr.add(bi) = i32::try_from(start_pos + bi)
                    .map_err(|_| Error::Model("packed prefill position exceeds i32".into()))?;
            }
        }

        let mut tcb = TokenCommandBuffer::new(ctx);
        crate::kernels::rmsnorm_f32_batched_tcb(
            &mut tcb,
            &arena.prefill_x,
            &arena.attn_norm_weights[0],
            &arena.prefill_x_norm,
            cfg.rms_norm_eps,
            h,
            b,
        )?;

        // Q4 rows are evaluated once across B activations. Q6 remains one
        // source GEMV per row until a strict batched Q6 kernel is available;
        // that preserves the current source representation rather than
        // dequantizing or changing quantization tier for prefill.
        let batch_projection = |tcb: &mut TokenCommandBuffer<'_>,
                                tensor: &TensorRef,
                                rows: usize,
                                cols: usize,
                                input: &crate::metal::PinnedBuffer,
                                input_stride: usize,
                                output: &crate::metal::PinnedBuffer,
                                output_stride: usize|
         -> Result<()> {
            match tensor.dtype {
                GgmlType::Q4_K => crate::kernels::gemv_q4_k_m_llama_b9430_batched_pinned_tcb(
                    tcb,
                    model_buf,
                    tensor.offset,
                    tensor.byte_size,
                    rows,
                    cols,
                    b,
                    input,
                    output,
                ),
                GgmlType::Q6_K => {
                    for bi in 0..b {
                        crate::kernels::gemv_q6_k_llama_b9430_pinned_off_tcb(
                            tcb,
                            model_buf,
                            tensor.offset,
                            tensor.byte_size,
                            rows,
                            cols,
                            input,
                            bi * input_stride,
                            output,
                            bi * output_stride,
                        )?;
                    }
                    Ok(())
                }
                other => Err(Error::Model(format!(
                    "packed prefill unsupported projection dtype {other:?}"
                ))),
            }
        };

        for (li, layer) in self.layers.iter().enumerate() {
            batch_projection(
                &mut tcb,
                &layer.q_proj,
                q_dim,
                h,
                &arena.prefill_x_norm,
                h_bytes,
                &arena.prefill_q,
                q_bytes,
            )?;
            batch_projection(
                &mut tcb,
                &layer.k_proj,
                kv_dim,
                h,
                &arena.prefill_x_norm,
                h_bytes,
                &arena.prefill_k,
                kv_bytes,
            )?;
            batch_projection(
                &mut tcb,
                &layer.v_proj,
                kv_dim,
                h,
                &arena.prefill_x_norm,
                h_bytes,
                &arena.prefill_v,
                kv_bytes,
            )?;

            // Keep source RoPE factors and source half rounding. The packed
            // activations are sliced only at the operator boundary.
            for bi in 0..b {
                crate::kernels::rope_norm_llama_b9430_qkv_cache_f16_off_tcb(
                    &mut tcb,
                    &arena.prefill_q,
                    bi * q_bytes,
                    &arena.prefill_k,
                    bi * kv_bytes,
                    &arena.prefill_v,
                    bi * kv_bytes,
                    &arena.prefill_q,
                    bi * q_bytes,
                    &arena.keys_f16[li],
                    &arena.values_f16[li],
                    &arena.prefill_positions,
                    bi * std::mem::size_of::<i32>(),
                    &arena.rope_factors,
                    cfg.n_heads,
                    cfg.n_kv_heads,
                    cfg.head_dim,
                    (start_pos + bi) as u32,
                    cfg.rope_theta,
                    (start_pos + bi) * kv_dim,
                )?;
            }
            crate::kernels::mha_decode_llama_b9430_fattn_prefill_tcb(
                &mut tcb,
                &arena.prefill_q,
                &arena.keys_f16[li],
                &arena.values_f16[li],
                &arena.prefill_fattn_scratch,
                &arena.prefill_attn,
                start_pos,
                b,
                cfg.head_dim,
                cfg.n_heads,
                cfg.n_kv_heads,
            )?;
            batch_projection(
                &mut tcb,
                &layer.o_proj,
                h,
                q_dim,
                &arena.prefill_attn,
                q_bytes,
                &arena.prefill_o,
                h_bytes,
            )?;
            crate::kernels::add_rmsnorm_fused_batched_tcb(
                &mut tcb,
                &arena.prefill_x,
                &arena.prefill_o,
                &arena.ffn_norm_weights[li],
                &arena.prefill_x_norm,
                cfg.rms_norm_eps,
                h,
                b,
            )?;
            batch_projection(
                &mut tcb,
                &layer.ffn_gate,
                mid,
                h,
                &arena.prefill_x_norm,
                h_bytes,
                &arena.prefill_gate,
                mid_bytes,
            )?;
            batch_projection(
                &mut tcb,
                &layer.ffn_up,
                mid,
                h,
                &arena.prefill_x_norm,
                h_bytes,
                &arena.prefill_up,
                mid_bytes,
            )?;
            crate::kernels::swiglu_llama_b9430_tcb(
                &mut tcb,
                &arena.prefill_gate,
                &arena.prefill_up,
                &arena.prefill_act,
                b * mid,
            )?;
            batch_projection(
                &mut tcb,
                &layer.ffn_down,
                h,
                mid,
                &arena.prefill_act,
                mid_bytes,
                &arena.prefill_down,
                h_bytes,
            )?;
            let next_norm = if li + 1 < cfg.n_layers {
                &arena.attn_norm_weights[li + 1]
            } else {
                &arena.final_norm_weight
            };
            crate::kernels::add_rmsnorm_fused_batched_tcb(
                &mut tcb,
                &arena.prefill_x,
                &arena.prefill_down,
                next_norm,
                &arena.prefill_x_norm,
                cfg.rms_norm_eps,
                h,
                b,
            )?;
        }
        tcb.commit_and_wait()?;
        self.kv.seq_len += b;
        Ok(true)
    }

    /// Forward one token at position `pos`. Appends K/V at the current
    /// `kv.seq_len` slot and bumps `seq_len`. On macOS the Q4_K
    /// projections, the f16 LM head, and rmsnorm run on Metal; the rest
    /// (Q6_K weights, attention) uses the CPU reference path.
    ///
    /// Mirrors `qwen_dense::forward_token`, with two differences:
    ///   - no Q/K/V bias adds (Llama families omit them)
    ///   - RoPE consumes the GGUF's resolved `rope_freqs.weight` factors
    ///     when present, falling back to Llama-3.1 metadata scaling.
    pub(crate) fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        if self.track_execution {
            self.last_dispatch_count.store(0, Ordering::Relaxed);
            self.last_command_buffer_count.store(0, Ordering::Relaxed);
            self.last_cpu_reference_fallback_count
                .store(0, Ordering::Relaxed);
        }
        #[cfg(target_os = "macos")]
        if let Some(logits) = self.forward_token_resident_b9430(token, pos)? {
            return Ok(logits);
        }
        let cfg = &self.config;
        let h = cfg.hidden;
        let head_dim = cfg.head_dim;
        let n_heads = cfg.n_heads;
        let n_kv_heads = cfg.n_kv_heads;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let rope_scaling = cfg.rope_scaling;
        let rope_freq_factors = cfg.rope_freq_factors.as_deref();
        let rope_theta = cfg.rope_theta;
        let rms_eps = cfg.rms_norm_eps;
        let n_layers = cfg.n_layers;
        let mid = cfg.intermediate;
        let vocab_size = cfg.vocab_size;

        let embedding_start = token as usize * h;
        let mut x = self.embed[embedding_start..embedding_start + h].to_vec();
        let mut checkpoint = self.checkpoint_enabled().then(|| LlamaCheckpointRecord {
            position: pos,
            token_id: token,
            embedding_sum: checkpoint_sum(&x),
            layers: Vec::with_capacity(n_layers),
            final_norm_sum: 0.0,
            logits_sum: 0.0,
            greedy_token_id: 0,
            debug_vector: None,
            debug_vectors: Vec::new(),
            capture_kind: None,
        });
        if checkpoint.is_some() {
            self.capture_checkpoint_vector(&mut checkpoint, "embedding".into(), &x);
        }

        let mut scratch = Vec::<f32>::new();

        let stride = n_kv_heads * head_dim;
        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model(format!(
                "kv cache full at {}",
                self.kv.max_seq
            )));
        }
        let kv_off = self.kv.seq_len * stride;
        let mha_seq_len = self.kv.seq_len + 1;

        for li in 0..n_layers {
            // Per-layer weights are accessed in place via `self.layers[li]`.
            // Both the dispatch-method receiver and the weight argument are
            // shared borrows of `self`, so no clone is needed; the borrows
            // end before the `&mut self.kv` write below. (The earlier
            // version cloned every norm + TensorRef per layer per token —
            // hidden-sized allocations on the decode hot path.)
            let mut x_norm = vec![0.0f32; h];
            let mut q_full = vec![0.0f32; q_dim];
            let mut k_token = vec![0.0f32; kv_dim];
            let mut v_token = vec![0.0f32; kv_dim];
            let mut q_raw = vec![0.0f32; q_dim];
            let mut k_raw = vec![0.0f32; kv_dim];
            let attn_rmsnorm_qkv_rope_batched = if let (Some(ctx), Some(model_buf)) =
                (self.metal_ctx.as_ref(), self.weights_mmap_buf.as_ref())
            {
                let layer = &self.layers[li];
                if crate::env_on("HAWKING_LLAMA_BATCH_ATTN_NORM")
                    && rope_scaling.is_none()
                    && layer.q_proj.dtype == GgmlType::Q4_K
                    && layer.k_proj.dtype == GgmlType::Q4_K
                    && layer.v_proj.dtype == GgmlType::Q4_K
                {
                    // RMSNorm plus the five source-equivalent Q/K/V+RoPE
                    // kernels share one command buffer. Count every kernel
                    // so the K0 execution receipt remains structural.
                    for _ in 0..6 {
                        self.record_dispatch();
                    }
                    crate::kernels::llama_b9430_rmsnorm_qkv_rope_q4_pinned(
                        ctx,
                        model_buf,
                        &x,
                        &layer.attn_norm,
                        rms_eps,
                        &mut x_norm,
                        layer.q_proj.offset,
                        layer.q_proj.byte_size,
                        q_dim,
                        layer.k_proj.offset,
                        layer.k_proj.byte_size,
                        layer.v_proj.offset,
                        layer.v_proj.byte_size,
                        kv_dim,
                        &mut q_raw,
                        &mut k_raw,
                        &mut v_token,
                        &mut q_full,
                        &mut k_token,
                        head_dim,
                        pos as u32,
                        rope_theta,
                        rope_freq_factors,
                    )?;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !attn_rmsnorm_qkv_rope_batched {
                self.rmsnorm_dispatch(&x, &self.layers[li].attn_norm, rms_eps, &mut x_norm)?;
            }
            let attn_norm_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&x_norm));
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.attn_norm"),
                    &x_norm,
                );
            }

            let qkv_rope_batched = attn_rmsnorm_qkv_rope_batched
                || if let (Some(ctx), Some(model_buf)) =
                    (self.metal_ctx.as_ref(), self.weights_mmap_buf.as_ref())
                {
                    let layer = &self.layers[li];
                    if rope_scaling.is_none()
                        && layer.q_proj.dtype == GgmlType::Q4_K
                        && layer.k_proj.dtype == GgmlType::Q4_K
                        && layer.v_proj.dtype == GgmlType::Q4_K
                    {
                        for _ in 0..5 {
                            self.record_dispatch();
                        }
                        crate::kernels::llama_b9430_qkv_rope_q4_pinned(
                            ctx,
                            model_buf,
                            layer.q_proj.offset,
                            layer.q_proj.byte_size,
                            q_dim,
                            layer.k_proj.offset,
                            layer.k_proj.byte_size,
                            layer.v_proj.offset,
                            layer.v_proj.byte_size,
                            kv_dim,
                            h,
                            &x_norm,
                            &mut q_raw,
                            &mut k_raw,
                            &mut v_token,
                            &mut q_full,
                            &mut k_token,
                            head_dim,
                            pos as u32,
                            rope_theta,
                            rope_freq_factors,
                        )?;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
            if !qkv_rope_batched {
                self.matmul_q4_dispatch(
                    &self.layers[li].q_proj,
                    q_dim,
                    h,
                    &x_norm,
                    &mut q_full,
                    &mut scratch,
                )?;
                self.matmul_q4_dispatch(
                    &self.layers[li].k_proj,
                    kv_dim,
                    h,
                    &x_norm,
                    &mut k_token,
                    &mut scratch,
                )?;
                self.matmul_q4_dispatch(
                    &self.layers[li].v_proj,
                    kv_dim,
                    h,
                    &x_norm,
                    &mut v_token,
                    &mut scratch,
                )?;
                q_raw.copy_from_slice(&q_full);
                k_raw.copy_from_slice(&k_token);
            }
            // The oracle distinguishes the raw Q/K projections from their
            // RoPE-transformed forms. Capture these before the in-place RoPE
            // call so a first divergence points at the right operation.
            let q_raw_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&q_raw));
            let k_raw_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&k_raw));
            let v_raw_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&v_token));
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.q_raw"),
                    &q_raw,
                );
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.k_raw"),
                    &k_raw,
                );
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.v_raw"),
                    &v_token,
                );
            }

            // RoPE on every Q head and every KV head.  The frozen Llama-3.1
            // authority uses linear scaling (represented as `None` here), so
            // run the matching Metal arithmetic instead of host libm.  The
            // Llama-3 metadata-scaled path retains its established CPU form.
            if !qkv_rope_batched {
                if let Some(ctx) = self.metal_ctx.as_ref().filter(|_| rope_scaling.is_none()) {
                    self.record_dispatch();
                    crate::kernels::rope_norm_llama_b9430(
                        ctx,
                        &mut q_full,
                        head_dim,
                        pos as u32,
                        rope_theta,
                        rope_freq_factors,
                    )?;
                    self.record_dispatch();
                    crate::kernels::rope_norm_llama_b9430(
                        ctx,
                        &mut k_token,
                        head_dim,
                        pos as u32,
                        rope_theta,
                        rope_freq_factors,
                    )?;
                } else {
                    for h_i in 0..n_heads {
                        let off = h_i * head_dim;
                        rope_inplace_normal_with_factors(
                            &mut q_full[off..off + head_dim],
                            pos as u32,
                            rope_theta,
                            rope_scaling,
                            rope_freq_factors,
                        );
                    }
                    for h_i in 0..n_kv_heads {
                        let off = h_i * head_dim;
                        rope_inplace_normal_with_factors(
                            &mut k_token[off..off + head_dim],
                            pos as u32,
                            rope_theta,
                            rope_scaling,
                            rope_freq_factors,
                        );
                    }
                }
            }
            let q_rope_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&q_full));
            let k_rope_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&k_token));
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.q_rope"),
                    &q_full,
                );
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.k_rope"),
                    &k_token,
                );
            }

            // llama.cpp's GGUF path stores K/V in f16 (the independent eval
            // callback exposes `cache_{k,v}` as f16). Keep Hawking's generic
            // cache allocation as f32 for now, but round at the write seam so
            // every later attention read observes the same numerical state.
            // A true f16 resident cache is the throughput follow-up.
            for (stored, value) in self.kv.keys[li][kv_off..kv_off + stride]
                .iter_mut()
                .zip(k_token.iter())
            {
                *stored = f16::from_f32(*value).to_f32();
            }
            for (stored, value) in self.kv.values[li][kv_off..kv_off + stride]
                .iter_mut()
                .zip(v_token.iter())
            {
                *stored = f16::from_f32(*value).to_f32();
            }
            if checkpoint.is_some() {
                // These are the exact f16-rounded values subsequently read
                // by attention. The matching llama.cpp callback surfaces are
                // `cache_{k,v}_l{N} (view)`; keep this trace-only seam
                // distinct from `k_rope` and `v_raw` so a cache conversion
                // mismatch cannot be misattributed to Flash Attention.
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.k_cached"),
                    &self.kv.keys[li][kv_off..kv_off + stride],
                );
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.v_cached"),
                    &self.kv.values[li][kv_off..kv_off + stride],
                );
            }

            let kv_size = mha_seq_len * stride;
            let keys = &self.kv.keys[li][..kv_size];
            let values = &self.kv.values[li][..kv_size];

            if checkpoint.is_some()
                && self
                    .checkpoint_vector_surfaces
                    .iter()
                    .any(|surface| surface == &format!("layer.{li}.q_attn_f16"))
            {
                // b9430 converts Q to half4 at the Flash-Attention entry.
                // Trace that exact boundary only when explicitly requested;
                // normal decode does not allocate this diagnostic vector.
                let q_attn_f16 = q_full
                    .iter()
                    .map(|value| f16::from_f32(*value).to_f32())
                    .collect::<Vec<_>>();
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.q_attn_f16"),
                    &q_attn_f16,
                );
            }

            let mut attn_out = vec![0.0f32; q_dim];
            if let Some(ctx) = self
                .metal_ctx
                .as_ref()
                .filter(|_| !crate::env_on("HAWKING_LLAMA_CPU_ATTN_AUTHORITY"))
            {
                self.record_dispatch();
                if crate::env_on("HAWKING_LLAMA_GGML_FATTN_AUTHORITY") {
                    // Diagnostic-only authority adapter. It launches the
                    // pinned GGML Metal FLASH_ATTN_EXT primitive directly to
                    // determine whether the final Llama K0 discrepancy is
                    // solely Hawking's custom attention reduction. This is
                    // never a normal execution or throughput path.
                    crate::kernels::mha_decode_llama_ggml_fattn_authority(
                        &q_full,
                        keys,
                        values,
                        mha_seq_len,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        &mut attn_out,
                    )?;
                } else if mha_seq_len <= 32 && head_dim == 128 {
                    crate::kernels::mha_decode_llama_b9430_short_metal(
                        ctx,
                        &q_full,
                        keys,
                        values,
                        mha_seq_len,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        &mut attn_out,
                    )?;
                } else {
                    // b9430's f16-KV Flash-Attention specialization stages
                    // f32 Q through half4. Preserve that boundary for the
                    // generic device bridge until its multi-workgroup
                    // authority reduction is installed.
                    let q_half_staged = q_full
                        .iter()
                        .map(|value| f16::from_f32(*value).to_f32())
                        .collect::<Vec<_>>();
                    crate::kernels::mha_decode_f32_metal(
                        ctx,
                        &q_half_staged,
                        keys,
                        values,
                        mha_seq_len,
                        head_dim,
                        n_heads,
                        n_kv_heads,
                        &mut attn_out,
                    )?;
                }
            } else {
                self.record_cpu_fallback();
                mha_decode_step(
                    &q_full,
                    keys,
                    values,
                    n_heads,
                    n_kv_heads,
                    head_dim,
                    mha_seq_len,
                    &mut attn_out,
                )?;
            }
            if checkpoint.is_some() {
                // llama.cpp labels this pre-output-projection context tensor
                // `__fattn__-{li}`.  Keep it vector-selectable so attention
                // arithmetic can be checked directly rather than inferred
                // through the Q4_K output projection.
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.attn_context"),
                    &attn_out,
                );
            }

            let mut o = vec![0.0f32; h];
            self.matmul_q4_dispatch(
                &self.layers[li].o_proj,
                h,
                q_dim,
                &attn_out,
                &mut o,
                &mut scratch,
            )?;
            add_inplace(&mut x, &o);
            let attn_out_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&o));
            let ffn_input_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&x));
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(&mut checkpoint, format!("layer.{li}.attn_out"), &o);
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.ffn_input"),
                    &x,
                );
            }

            let mut x_norm2 = vec![0.0f32; h];
            let mut g = vec![0.0f32; mid];
            let mut u = vec![0.0f32; mid];
            let mut a = vec![0.0f32; mid];
            let mut f = vec![0.0f32; h];
            let ffn_rmsnorm_batched = if let (Some(ctx), Some(model_buf)) =
                (self.metal_ctx.as_ref(), self.weights_mmap_buf.as_ref())
            {
                let layer = &self.layers[li];
                if layer.ffn_gate.dtype == GgmlType::Q4_K
                    && layer.ffn_up.dtype == GgmlType::Q4_K
                    && layer.ffn_down.dtype == GgmlType::Q6_K
                {
                    for _ in 0..5 {
                        self.record_dispatch();
                    }
                    crate::kernels::llama_b9430_rmsnorm_ffn_q4_q6_pinned(
                        ctx,
                        model_buf,
                        &x,
                        &layer.ffn_norm,
                        rms_eps,
                        &mut x_norm2,
                        layer.ffn_gate.offset,
                        layer.ffn_gate.byte_size,
                        layer.ffn_up.offset,
                        layer.ffn_up.byte_size,
                        layer.ffn_down.offset,
                        layer.ffn_down.byte_size,
                        mid,
                        &mut g,
                        &mut u,
                        &mut a,
                        &mut f,
                    )?;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            if !ffn_rmsnorm_batched {
                self.rmsnorm_dispatch(&x, &self.layers[li].ffn_norm, rms_eps, &mut x_norm2)?;
            }
            let ffn_norm_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&x_norm2));
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.ffn_norm"),
                    &x_norm2,
                );
            }
            let ffn_batched = ffn_rmsnorm_batched
                || if let (Some(ctx), Some(model_buf)) =
                    (self.metal_ctx.as_ref(), self.weights_mmap_buf.as_ref())
                {
                    let layer = &self.layers[li];
                    if layer.ffn_gate.dtype == GgmlType::Q4_K
                        && layer.ffn_up.dtype == GgmlType::Q4_K
                        && layer.ffn_down.dtype == GgmlType::Q6_K
                    {
                        for _ in 0..4 {
                            self.record_dispatch();
                        }
                        crate::kernels::llama_b9430_ffn_q4_q6_pinned(
                            ctx,
                            model_buf,
                            layer.ffn_gate.offset,
                            layer.ffn_gate.byte_size,
                            layer.ffn_up.offset,
                            layer.ffn_up.byte_size,
                            layer.ffn_down.offset,
                            layer.ffn_down.byte_size,
                            h,
                            mid,
                            &x_norm2,
                            &mut g,
                            &mut u,
                            &mut a,
                            &mut f,
                        )?;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                };
            if !ffn_batched {
                self.matmul_q4_dispatch(
                    &self.layers[li].ffn_gate,
                    mid,
                    h,
                    &x_norm2,
                    &mut g,
                    &mut scratch,
                )?;
                self.matmul_q4_dispatch(
                    &self.layers[li].ffn_up,
                    mid,
                    h,
                    &x_norm2,
                    &mut u,
                    &mut scratch,
                )?;
                if let Some(ctx) = &self.metal_ctx {
                    self.record_dispatch();
                    crate::kernels::swiglu_llama_b9430(ctx, &g, &u, &mut a)?;
                } else {
                    self.record_cpu_fallback();
                    silu_mul(&g, &u, &mut a);
                }
                self.matmul_q4_dispatch(
                    &self.layers[li].ffn_down,
                    h,
                    mid,
                    &a,
                    &mut f,
                    &mut scratch,
                )?;
            }
            let ffn_gate_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&g));
            let ffn_up_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&u));
            let ffn_swiglu_sum = checkpoint.as_ref().map(|_| checkpoint_sum(&a));
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(&mut checkpoint, format!("layer.{li}.ffn_gate"), &g);
                self.capture_checkpoint_vector(&mut checkpoint, format!("layer.{li}.ffn_up"), &u);
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.ffn_swiglu"),
                    &a,
                );
            }
            add_inplace(&mut x, &f);
            if checkpoint.is_some() {
                self.capture_checkpoint_vector(&mut checkpoint, format!("layer.{li}.ffn_out"), &f);
                self.capture_checkpoint_vector(
                    &mut checkpoint,
                    format!("layer.{li}.layer_out"),
                    &x,
                );
            }
            if let Some(record) = checkpoint.as_mut() {
                record.layers.push(LlamaLayerCheckpoint {
                    layer: li,
                    attn_norm_sum: attn_norm_sum.expect("checkpoint enabled"),
                    q_raw_sum: q_raw_sum.expect("checkpoint enabled"),
                    k_raw_sum: k_raw_sum.expect("checkpoint enabled"),
                    v_raw_sum: v_raw_sum.expect("checkpoint enabled"),
                    q_rope_sum: q_rope_sum.expect("checkpoint enabled"),
                    k_rope_sum: k_rope_sum.expect("checkpoint enabled"),
                    attn_out_sum: attn_out_sum.expect("checkpoint enabled"),
                    ffn_input_sum: ffn_input_sum.expect("checkpoint enabled"),
                    ffn_norm_sum: ffn_norm_sum.expect("checkpoint enabled"),
                    ffn_gate_sum: ffn_gate_sum.expect("checkpoint enabled"),
                    ffn_up_sum: ffn_up_sum.expect("checkpoint enabled"),
                    ffn_swiglu_sum: ffn_swiglu_sum.expect("checkpoint enabled"),
                    ffn_out_sum: checkpoint_sum(&f),
                    layer_out_sum: checkpoint_sum(&x),
                });
            }
        }

        self.kv.seq_len += 1;

        // final_norm and the LM head only need shared borrows of `self`
        // (no intervening &mut), so the dispatchers read them in place —
        // no per-token clone of the multi-hundred-MB weight matrices.
        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, rms_eps, &mut x_norm)?;

        let mut logits = vec![0.0f32; vocab_size];
        self.lm_head_dispatch(vocab_size, h, &x_norm, &mut logits)?;
        if checkpoint.is_some() {
            self.capture_checkpoint_vector(&mut checkpoint, "final_norm".into(), &x_norm);
            self.capture_checkpoint_vector(&mut checkpoint, "logits".into(), &logits);
        }
        if let Some(mut record) = checkpoint {
            record.final_norm_sum = checkpoint_sum(&x_norm);
            record.logits_sum = checkpoint_sum(&logits);
            record.greedy_token_id = logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(index, _)| index as u32)
                .unwrap_or_default();
            self.checkpoint_records.push(record);
        }
        Ok(logits)
    }

    fn flush_checkpoint_records(&mut self, prompt_ids: &[u32]) -> Result<()> {
        let path = if let Some(dir) = self.checkpoint_summary_dir.as_ref() {
            std::fs::create_dir_all(dir)?;
            let path = dir.join(format!(
                "checkpoint-{:05}.json",
                self.checkpoint_capture_index
            ));
            self.checkpoint_capture_index = self.checkpoint_capture_index.saturating_add(1);
            path
        } else if let Some(path) = self.checkpoint_summary_path.as_ref() {
            path.clone()
        } else {
            return Ok(());
        };
        // Resident student capture has exactly two f32 planes per prompt
        // position.  Writing them as decimal JSON costs ~8x the useful bytes
        // and makes calibration host-bound.  Keep the normal checkpoint JSON
        // untouched; this explicitly named diagnostic writes a tiny JSON
        // header followed by token ids and the two native f32 planes.
        let resident_vector_capture = crate::env_on("HAWKING_LLAMA_RESIDENT_COMPACT_CAPTURE")
            && !self.checkpoint_records.is_empty()
            && self.checkpoint_records.len() == prompt_ids.len()
            && self
                .checkpoint_records
                .iter()
                .all(|record| record.capture_kind == Some("resident_final_ffn_vectors_only"));
        if resident_vector_capture {
            let first = &self.checkpoint_records[0];
            let ffn_norm = first
                .debug_vectors
                .iter()
                .find(|vector| vector.surface.ends_with(".ffn_norm"))
                .ok_or_else(|| Error::Model("resident capture lacks ffn_norm vector".into()))?;
            let ffn_out = first
                .debug_vectors
                .iter()
                .find(|vector| vector.surface.ends_with(".ffn_out"))
                .ok_or_else(|| Error::Model("resident capture lacks ffn_out vector".into()))?;
            let width = ffn_norm.values.len();
            if width == 0 || ffn_out.values.len() != width {
                return Err(Error::Model(
                    "resident capture has invalid vector width".into(),
                ));
            }
            for record in &self.checkpoint_records {
                let norm = record
                    .debug_vectors
                    .iter()
                    .find(|vector| vector.surface == ffn_norm.surface)
                    .ok_or_else(|| Error::Model("resident capture row lacks ffn_norm".into()))?;
                let out = record
                    .debug_vectors
                    .iter()
                    .find(|vector| vector.surface == ffn_out.surface)
                    .ok_or_else(|| Error::Model("resident capture row lacks ffn_out".into()))?;
                if norm.values.len() != width || out.values.len() != width {
                    return Err(Error::Model("resident capture row width mismatch".into()));
                }
            }
            let header = serde_json::json!({
                "schema": "hawking.tg.llama_resident_f32_capture.v1",
                "model_id": self.model_id,
                "model_arch": self.config.arch,
                "weights_path": self._weights_path,
                "rows": self.checkpoint_records.len(),
                "width": width,
                "input_surface": ffn_norm.surface,
                "target_surface": ffn_out.surface,
            });
            let header = serde_json::to_vec(&header)
                .map_err(|err| Error::Model(format!("serialize resident capture header: {err}")))?;
            let binary_path = path.with_extension("bin");
            let mut file = std::fs::File::create(&binary_path)?;
            file.write_all(b"HLRFFN1\0")?;
            file.write_all(&(header.len() as u32).to_le_bytes())?;
            file.write_all(&header)?;
            file.write_all(bytemuck::cast_slice(prompt_ids))?;
            for record in &self.checkpoint_records {
                let vector = record
                    .debug_vectors
                    .iter()
                    .find(|vector| vector.surface == ffn_norm.surface)
                    .expect("validated resident ffn_norm vector");
                file.write_all(bytemuck::cast_slice(vector.values.as_slice()))?;
            }
            for record in &self.checkpoint_records {
                let vector = record
                    .debug_vectors
                    .iter()
                    .find(|vector| vector.surface == ffn_out.surface)
                    .expect("validated resident ffn_out vector");
                file.write_all(bytemuck::cast_slice(vector.values.as_slice()))?;
            }
            file.flush()?;
            return Ok(());
        }
        let document = serde_json::json!({
            "schema": "hawking.tg.llama_checkpoint_summary.v1",
            "model_id": self.model_id,
            "model_arch": self.config.arch,
            "weights_path": self._weights_path,
            "prompt_token_ids": prompt_ids,
            "records": &self.checkpoint_records,
            "note": "Debug-only scalar aggregates. Sum prompt-position records before comparing to llama-eval-callback's batched aggregates.",
        });
        let bytes = serde_json::to_vec_pretty(&document)
            .map_err(|err| Error::Model(format!("serialize Llama checkpoint trace: {err}")))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    fn record_execution_stats(&self, stats: &mut GenStats) {
        stats.device_id = self.metal_ctx.as_ref().map(|ctx| ctx.device_name());
        stats.dispatch_samples = self
            .metal_ctx
            .as_ref()
            .map(|ctx| ctx.drain_trace())
            .unwrap_or_default();
        let (buffers_created, bytes_allocated, commits) = self
            .metal_ctx
            .as_ref()
            .map(|ctx| ctx.drain_stats())
            .unwrap_or_default();
        stats.metal_buffers_created = buffers_created;
        stats.metal_bytes_allocated = bytes_allocated;
        stats.metal_commits = commits;
        stats.metal_dispatches = self.last_dispatch_count.load(Ordering::Relaxed);
        stats.dispatches_per_forward = stats.metal_dispatches;
        stats.cpu_reference_fallback_count = self
            .last_cpu_reference_fallback_count
            .load(Ordering::Relaxed);
    }
}

impl Engine for LlamaDense {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = LlamaConfig::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("llama-dense").to_string();

        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        let tokenizer = if let Some(p) = sidecar {
            Tokenizer::from_file(&p)?
        } else {
            Tokenizer::from_gguf(&gguf)?
        };

        let embed = dequant_f32(&gguf, "token_embd.weight")?;
        let embed_f16_for_metal = if gguf.tensor("output.weight").is_some() {
            None
        } else {
            Some(dequant_f16(&gguf, "token_embd.weight")?)
        };
        let final_norm = dequant_f32(&gguf, "output_norm.weight")?;
        let lm_head = if gguf.tensor("output.weight").is_some() {
            Some(dequant_f16(&gguf, "output.weight")?)
        } else {
            None
        };
        let lm_head_raw = if gguf.tensor("output.weight").is_some() {
            Some(tensor_ref(&gguf, "output.weight")?)
        } else {
            None
        };
        let lm_head_f32 = if gguf.tensor("output.weight").is_some() {
            Some(dequant_f32(&gguf, "output.weight")?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");
            layers.push(LlamaLayer {
                attn_norm: dequant_f32(&gguf, &lp("attn_norm.weight"))?,
                ffn_norm: dequant_f32(&gguf, &lp("ffn_norm.weight"))?,
                q_proj: tensor_ref(&gguf, &lp("attn_q.weight"))?,
                k_proj: tensor_ref(&gguf, &lp("attn_k.weight"))?,
                v_proj: tensor_ref(&gguf, &lp("attn_v.weight"))?,
                o_proj: tensor_ref(&gguf, &lp("attn_output.weight"))?,
                ffn_gate: tensor_ref(&gguf, &lp("ffn_gate.weight"))?,
                ffn_up: tensor_ref(&gguf, &lp("ffn_up.weight"))?,
                ffn_down: tensor_ref(&gguf, &lp("ffn_down.weight"))?,
            });
        }

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(cfg.n_layers, max_seq, cfg.n_kv_heads, cfg.head_dim);
        let sampler = Sampler::new(0);

        // Match the other dense engines: a CPU-only parity diagnostic must
        // have a real way to suppress every Metal path, rather than merely
        // infer it from a missing device.
        let metal_ctx = crate::engine::init_optional_metal(&config)?;
        let device_name = metal_ctx.as_ref().map(|ctx| ctx.device_name());
        if let Some(profile) = config.kernel_profile.as_ref() {
            profile.validate_for_gguf(&gguf, device_name.as_deref())?;
        }
        let kernel_profile = config.kernel_profile.clone();
        let weights_mmap_buf = {
            #[cfg(target_os = "macos")]
            {
                metal_ctx
                    .as_ref()
                    .map(|ctx| unsafe { ctx.new_buffer_no_copy(&gguf.mmap) })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = &metal_ctx;
                None
            }
        };
        let checkpoint_summary_path = std::env::var_os("HAWKING_LLAMA_CHECKPOINT_SUMMARY_PATH")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let checkpoint_summary_dir = std::env::var_os("HAWKING_LLAMA_CHECKPOINT_SUMMARY_DIR")
            .filter(|value| !value.is_empty())
            .map(PathBuf::from);
        let checkpoint_vector_surfaces = std::env::var("HAWKING_LLAMA_CHECKPOINT_VECTOR")
            .ok()
            .map(|value| {
                value
                    .split(',')
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();

        Ok(Self {
            config: cfg,
            tokenizer,
            model_id,
            weights_mmap_buf,
            gguf,
            embed,
            embed_f16_for_metal,
            final_norm,
            lm_head,
            lm_head_raw,
            lm_head_f32,
            layers,
            kv,
            sampler,
            kernel_profile,
            _weights_path: weights.to_path_buf(),
            metal_ctx,
            #[cfg(target_os = "macos")]
            resident_arena: None,
            checkpoint_summary_path,
            checkpoint_summary_dir,
            checkpoint_capture_index: 0,
            checkpoint_vector_surfaces,
            checkpoint_records: Vec::new(),
            track_execution: config.trace_dispatch || crate::env_on("HAWKING_TRACE_DISPATCH"),
            last_dispatch_count: AtomicUsize::new(0),
            last_command_buffer_count: AtomicUsize::new(0),
            last_cpu_reference_fallback_count: AtomicUsize::new(0),
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        // Single-token serial prefill + decode through `forward_token`
        // (Metal-hybrid on macOS). Spec-decode, prefix caching, and the
        // full TCB+predec arena are deliberately out of scope here — they
        // layer on later behind the same env-var gates Qwen uses.
        if let Some(seed) = req.sampling.seed {
            self.sampler = Sampler::new(seed);
        }
        let abort_set = |req: &GenerateRequest| -> bool {
            req.abort
                .as_ref()
                .map(|f| f.load(Ordering::Relaxed))
                .unwrap_or(false)
        };
        let stall_limit = std::time::Duration::from_millis(req.max_stall_ms);
        let stall_active = req.max_stall_ms > 0;

        let prompt_ids = self.tokenizer.encode(&req.prompt, true)?;
        if prompt_ids.is_empty() {
            return Err(Error::Model("empty prompt after tokenization".into()));
        }
        let prompt_len = prompt_ids.len();
        let mut stats = GenStats {
            prompt_tokens: prompt_len,
            profile_id: self.kernel_profile.as_ref().map(|p| p.profile_id.clone()),
            ..Default::default()
        };

        self.kv.reset();
        self.checkpoint_records.clear();

        // Prefill: run every prompt token to populate the KV cache.  The
        // logits from the final prompt token are *already* the distribution
        // for completion token zero.  Do not feed that token a second time at
        // `pos = prompt_len`: doing so creates a duplicate KV entry and
        // changes every later position.  That was the common Llama/Mistral
        // state-advance defect found by the TG breadth oracle.
        let prefill_start = Instant::now();
        let mut prefill_aborted = false;
        let mut final_prompt_logits = None;
        // The generic-Q4 packed graph is a correctness scaffold, not a
        // production candidate: its same-model receipts are exact but slower
        // than serial prefill. Keep it behind an explicitly unsafe diagnostic
        // companion flag until source-compatible batched Q4 kernels replace
        // its generic projections.
        let packed_prefill_requested = crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_PACKED_PREFILL")
            && crate::env_on("HAWKING_LLAMA_RESIDENT_B9430_PACKED_PREFILL_UNSAFE_DIAGNOSTIC");
        if packed_prefill_requested {
            // Retain the known-short attention path for the first 32 prompt
            // positions, then process all but the final prompt token in
            // B≤8 causal chunks. The final token stays on the decode graph so
            // its logits are immediately the completion-zero distribution.
            let batch_end = prompt_len.saturating_sub(1);
            let packed_batch = crate::env_usize("HAWKING_LLAMA_PREFILL_BATCH", 8).clamp(2, 8);
            let mut i = 0usize;
            while i < batch_end {
                if abort_set(&req) {
                    prefill_aborted = true;
                    break;
                }
                let step_start = Instant::now();
                let end = if i >= 32 {
                    (i + packed_batch).min(batch_end)
                } else {
                    i + 1
                };
                let used_packed =
                    self.prefill_tokens_resident_b9430_packed(&prompt_ids[i..end], i)?;
                if !used_packed {
                    for (pos, &token) in prompt_ids[i..end].iter().enumerate() {
                        final_prompt_logits = Some(self.forward_token(token, i + pos)?);
                    }
                }
                if stall_active && step_start.elapsed() > stall_limit {
                    prefill_aborted = true;
                    break;
                }
                i = end;
            }
            if !prefill_aborted {
                let final_pos = prompt_len - 1;
                final_prompt_logits = Some(self.forward_token(prompt_ids[final_pos], final_pos)?);
            }
        } else {
            for (i, &t) in prompt_ids.iter().enumerate() {
                if abort_set(&req) {
                    prefill_aborted = true;
                    break;
                }
                let step_start = Instant::now();
                final_prompt_logits = Some(self.forward_token(t, i)?);
                if stall_active && step_start.elapsed() > stall_limit {
                    prefill_aborted = true;
                    break;
                }
            }
        }
        stats.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        if prefill_aborted {
            self.flush_checkpoint_records(&prompt_ids)?;
            self.record_execution_stats(&mut stats);
            sink(StreamEvent::Done {
                reason: StopReason::Aborted,
                stats: stats.clone(),
            });
            return Ok(stats);
        }

        // Decode loop.  `final_prompt_logits` supplies completion token zero;
        // each later iteration forwards the previous *generated* token at
        // the next unused position.  This matches the executable Gravity
        // engine and ordinary autoregressive KV-cache semantics.
        // The matched Llama protocol measures a contiguous suffix after real
        // autoregressive warmup in the *same* KV state.  This is deliberately
        // an opt-in environment seam rather than a product sampling option:
        // normal generate semantics and timings remain unchanged.
        let matched_warmup_tokens = std::env::var("HAWKING_LLAMA_MATCHED_WARMUP_TOKENS")
            .ok()
            .map(|value| value.parse::<usize>().map_err(|_| {
                Error::Model(format!(
                    "HAWKING_LLAMA_MATCHED_WARMUP_TOKENS must be a non-negative integer; got {value:?}"
                ))
            }))
            .transpose()?
            .unwrap_or(0);
        if matched_warmup_tokens >= req.max_new_tokens {
            return Err(Error::Model(format!(
                "matched warmup ({matched_warmup_tokens}) must be smaller than max_new_tokens ({})",
                req.max_new_tokens
            )));
        }
        // Completion token zero is selected from prompt-prefill logits.  Start
        // the decode interval immediately before the first measured forward,
        // never before that sampling-only event.
        let mut decode_start: Option<Instant> = None;
        let mut measured_token_ms =
            Vec::with_capacity(req.max_new_tokens.saturating_sub(matched_warmup_tokens));
        let mut measured_metal_dispatches_total = 0usize;
        let mut completed_decode_forwards = 0usize;
        let mut decode_command_buffers_total = 0usize;
        let mut measured_cpu_fallback_total = 0usize;
        let mut last_id = *prompt_ids.last().unwrap();
        let mut next_logits =
            final_prompt_logits.ok_or_else(|| Error::Model("prefill produced no logits".into()))?;
        let mut produced = 0usize;
        let mut reason = StopReason::MaxTokens;
        let eos = self.tokenizer.eos_id();

        for step in 0..req.max_new_tokens {
            if abort_set(&req) {
                reason = StopReason::Aborted;
                break;
            }
            // Completion token zero samples the final prompt logits; it has no
            // decode forward behind it.  Never include that instantaneous
            // sampling-only event in complete-token decode latency/TPS.
            let measured_step_start =
                (step > 0 && step >= matched_warmup_tokens).then(Instant::now);
            if step > 0 {
                if step >= matched_warmup_tokens && decode_start.is_none() {
                    decode_start = Some(Instant::now());
                }
                let pos = prompt_len + step - 1;
                let step_start = Instant::now();
                next_logits = self.forward_token(last_id, pos)?;
                if stall_active && step_start.elapsed() > stall_limit {
                    reason = StopReason::Aborted;
                    break;
                }
                if step >= matched_warmup_tokens {
                    measured_metal_dispatches_total = measured_metal_dispatches_total
                        .saturating_add(self.last_dispatch_count.load(Ordering::Relaxed));
                    let command_buffers = self.last_command_buffer_count.load(Ordering::Relaxed);
                    decode_command_buffers_total =
                        decode_command_buffers_total.saturating_add(command_buffers);
                    if command_buffers > 0 {
                        completed_decode_forwards = completed_decode_forwards.saturating_add(1);
                    }
                    measured_cpu_fallback_total = measured_cpu_fallback_total.saturating_add(
                        self.last_cpu_reference_fallback_count
                            .load(Ordering::Relaxed),
                    );
                }
            }
            let next_id = self.sampler.sample(&mut next_logits, &req.sampling);
            self.sampler.record(next_id);
            let text = self.tokenizer.decode_one(next_id).unwrap_or_default();
            sink(StreamEvent::Token { id: next_id, text });
            produced += 1;
            if let Some(step_start) = measured_step_start {
                measured_token_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);
            }
            if Some(next_id) == eos {
                reason = StopReason::Eos;
                break;
            }
            last_id = next_id;
        }

        stats.decode_ms = decode_start
            .map(|start| start.elapsed().as_secs_f64() * 1000.0)
            .unwrap_or(0.0);
        stats.completion_tokens = produced.saturating_sub(matched_warmup_tokens);
        stats.decode_token_ms = measured_token_ms;
        stats.decode_metal_dispatches_total = measured_metal_dispatches_total;
        stats.completed_decode_forwards = completed_decode_forwards;
        stats.decode_command_buffers_total = decode_command_buffers_total;
        stats.decode_cpu_reference_fallback_total = measured_cpu_fallback_total;
        self.flush_checkpoint_records(&prompt_ids)?;
        self.record_execution_stats(&mut stats);
        sink(StreamEvent::Done {
            reason,
            stats: stats.clone(),
        });
        Ok(stats)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn model_arch(&self) -> &str {
        &self.config.arch
    }

    fn encode_prompt_for_batch(&self, prompt: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(prompt, true)
    }

    fn decode_token_for_batch(&self, token: u32) -> Result<String> {
        self.tokenizer.decode_one(token)
    }

    fn eos_id_for_batch(&self) -> Option<u32> {
        self.tokenizer.eos_id()
    }

    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != positions.len() {
            return Err(Error::Model(format!(
                "forward_tokens shape: tokens={} positions={}",
                tokens.len(),
                positions.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            out.push(self.forward_token(token, positions[i])?);
        }
        Ok(out)
    }

    fn reset_kv_for_test(&mut self) {
        self.kv.seq_len = 0;
    }

    fn last_forward_dispatch_count(&self) -> usize {
        self.last_dispatch_count.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use crate::kernels::Llama3RopeScaling;

    /// Completion token zero comes directly from the final prompt logits;
    /// only completion tokens 1..N feed a generated token through the model.
    fn decode_input_positions(prompt_len: usize, completion_tokens: usize) -> Vec<Option<usize>> {
        (0..completion_tokens)
            .map(|step| (step > 0).then_some(prompt_len + step - 1))
            .collect()
    }

    #[test]
    fn rope_scaling_params_round_trip_into_struct() {
        let s = Llama3RopeScaling {
            factor: 32.0,
            low_freq_factor: 1.0,
            high_freq_factor: 4.0,
            original_max_position_embeddings: 8192,
        };
        assert_eq!(s.factor, 32.0);
        assert_eq!(s.original_max_position_embeddings, 8192);
    }

    #[test]
    fn decode_does_not_duplicate_the_last_prompt_token() {
        assert_eq!(
            decode_input_positions(5, 4),
            vec![None, Some(5), Some(6), Some(7)],
            "the old schedule started at Some(5) for completion zero, which fed the final prompt token twice"
        );
    }
}
