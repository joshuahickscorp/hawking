//! DeepSeek-V2-Lite forward pass.
//!
//! Architecture: MLA attention + MoE FFN with 2 shared experts and
//! top-6 of 64 routed experts (auxiliary-loss-free routing). First
//! transformer layer is dense (no routing); we still drive it
//! through the MoE kernel with a single-expert config so there's no
//! separate dense code path.
//!
//! The Phase-0 path runs entirely on CPU in fp32; the model layer's
//! job is bookkeeping (dequant on demand, KV cache management,
//! routing, residual stream). Phase 1+ swaps individual ops out for
//! Metal kernels under the same Rust signatures.

use crate::cache::KvCache;
use crate::engine::{
    Engine, EngineConfig, GenStats, GenerateRequest, SpeculateMode, StopReason, StreamEvent,
};
use crate::gguf::{GgmlType, GgufFile, TensorInfo};
use crate::kernels::{add_inplace, embed_lookup, gemv_f32, rmsnorm, rope_inplace, silu_mul};
use crate::metal::{DecodeArena, MetalContext, PinnedBuffer};
use crate::moe::topk_gate;
use crate::profile::KernelProfile;
use crate::quant;
use crate::sample::Sampler;
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use half::f16;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct DeepSeekConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub kv_lora_rank: usize,
    pub q_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// Intermediate width of the leading dense FFN layers.
    /// (`deepseek2.feed_forward_length`)
    pub ffn_intermediate: usize,
    /// Intermediate width of one routed/shared expert.
    /// (`deepseek2.expert_feed_forward_length`)
    pub moe_intermediate: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub top_k_routed: usize,
    pub first_k_dense_layers: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
}

impl DeepSeekConfig {
    fn from_gguf(g: &GgufFile) -> Result<Self> {
        let get_u32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_u32());
        let get_f32 = |k: &str| g.metadata.get(k).and_then(|v| v.as_f32());

        let n_layers = get_u32("deepseek2.block_count")
            .or_else(|| get_u32("llama.block_count"))
            .ok_or_else(|| Error::Model("missing block_count".into()))?
            as usize;
        let hidden = get_u32("deepseek2.embedding_length")
            .ok_or_else(|| Error::Model("missing embedding_length".into()))?
            as usize;
        let n_heads = get_u32("deepseek2.attention.head_count")
            .ok_or_else(|| Error::Model("missing head_count".into()))?
            as usize;
        let n_kv_heads =
            get_u32("deepseek2.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let vocab_size = get_u32("deepseek2.vocab_size")
            .or_else(|| get_u32("llama.vocab_size"))
            .ok_or_else(|| Error::Model("missing vocab_size".into()))?
            as usize;

        Ok(Self {
            n_layers,
            hidden,
            n_heads,
            n_kv_heads,
            kv_lora_rank: get_u32("deepseek2.attention.kv_lora_rank").unwrap_or(512) as usize,
            q_lora_rank: get_u32("deepseek2.attention.q_lora_rank").unwrap_or(0) as usize,
            qk_nope_head_dim: get_u32("deepseek2.attention.qk_nope_head_dim").unwrap_or(128)
                as usize,
            qk_rope_head_dim: get_u32("deepseek2.attention.qk_rope_head_dim").unwrap_or(64)
                as usize,
            v_head_dim: get_u32("deepseek2.attention.v_head_dim").unwrap_or(128) as usize,
            ffn_intermediate: get_u32("deepseek2.feed_forward_length")
                .or_else(|| get_u32("llama.feed_forward_length"))
                .unwrap_or(10944) as usize,
            moe_intermediate: get_u32("deepseek2.expert_feed_forward_length").unwrap_or(1408)
                as usize,
            n_routed_experts: get_u32("deepseek2.expert_count").unwrap_or(64) as usize,
            n_shared_experts: get_u32("deepseek2.expert_shared_count").unwrap_or(2) as usize,
            top_k_routed: get_u32("deepseek2.expert_used_count").unwrap_or(6) as usize,
            first_k_dense_layers: get_u32("deepseek2.leading_dense_block_count").unwrap_or(1)
                as usize,
            vocab_size,
            rope_theta: get_f32("deepseek2.rope.freq_base").unwrap_or(10_000.0),
            rms_norm_eps: get_f32("deepseek2.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
            max_seq_len: get_u32("deepseek2.context_length").unwrap_or(4096) as usize,
        })
    }
}

/// Phase 0 keeps the GGUF mmap'd for the lifetime of the engine and
/// lazily dequantizes expert weights on every forward pass. This
/// trades CPU work (per-call dequant) for resident memory — needed
/// because Q4_K_M weights expand 8× when materialized to fp32, and
/// DeepSeek-V2-Lite Q4_K_M (9.7 GB on disk) would balloon to ~70 GB
/// dequantized, busting any sub-128GB Mac.
///
/// Phase 1's wedge-2 quant-aware Metal kernels read 4-bit weights
/// directly from the mmap *and* dequant inside the FMA loop, removing
/// even the per-call working buffer.
/// Path-to-90 step 10 follow-up — collected GPU-side capture data
/// from the production Wedge C forward when `eagle4_capture_active`
/// is set. Three per-layer hiddens (residual-stream snapshots at
/// the end of layers 2, 13, 25 — i.e. `x_buf + ffn_out_buf` after
/// each layer's commit) plus the pre-MLP post-attn-rmsnorm hidden
/// at layer 26 (consumed by `cpu_shared_expert_forward` to produce
/// `h_shared`). All vectors are length `config.hidden`.
#[derive(Debug, Clone)]
pub struct Eagle4CaptureBuf {
    pub h_low: Vec<f32>,
    pub h_mid: Vec<f32>,
    pub h_high: Vec<f32>,
    /// Layer-26 pre-MoE post-attn-rmsnorm hidden. Kept as a fallback
    /// input for `cpu_shared_expert_forward` when the production MoE
    /// kernel's `moe_shared_out_buf` isn't populated (e.g. dense
    /// layers, or a future path that bypasses the fused MoE).
    pub x_norm_26: Vec<f32>,
    /// Shared-expert contribution at layer 26, read directly from
    /// the production MoE kernel's `moe_shared_out_buf` after layer
    /// 26's per-layer commit. Zero CPU dequant — same value the
    /// fused MoE summed into ffn_out_buf during the same forward.
    /// When non-empty, the Eagle4 decode loop uses this; otherwise
    /// it falls back to `cpu_shared_expert_forward(26, x_norm_26)`.
    pub h_shared_gpu: Vec<f32>,
}

impl Eagle4CaptureBuf {
    pub fn zeros(hidden: usize) -> Self {
        Self {
            h_low: vec![0.0; hidden],
            h_mid: vec![0.0; hidden],
            h_high: vec![0.0; hidden],
            x_norm_26: vec![0.0; hidden],
            h_shared_gpu: vec![0.0; hidden],
        }
    }
}

pub struct DeepSeekV2 {
    pub config: DeepSeekConfig,
    pub tokenizer: Tokenizer,
    pub model_id: String,

    /// mmap keepalive. Dropping this invalidates every TensorRef held
    /// by `layers`, so the field MUST live as long as any expert dispatch.
    pub gguf: GgufFile,

    pub embed: Vec<f16>,
    pub final_norm: Vec<f32>,
    pub lm_head: Option<Vec<f16>>, // None ⇒ tied to embed
    pub layers: Vec<Layer>,

    pub kv: KvCache,
    /// Wedge 1 — compressed MLA KV cache. Only allocated when
    /// `kernel_profile.selected.mla_schedule == "metal-mla"`. Shape:
    /// mla_c_kv[li][t * kv_lora_rank .. (t+1) * kv_lora_rank].
    pub mla_c_kv: Vec<Vec<f32>>,
    pub mla_k_pe: Vec<Vec<f32>>,
    /// GPU-resident KV cache for merged Phase-1+Wedge-N TCB path.
    /// mla_c_kv_gpu[li] holds max_seq × kv_lora_rank f32 entries.
    /// mla_k_pe_gpu[li] holds max_seq × qk_rope_head_dim f32 entries.
    pub mla_c_kv_gpu: Vec<PinnedBuffer>,
    pub mla_k_pe_gpu: Vec<PinnedBuffer>,
    /// Set to true after the first Wedge C layer to indicate GPU KV is live.
    pub mla_kv_gpu_synced: bool,
    /// v2.3.0 A4: when true, attention block routes to the function-constant-
    /// specialized `mla_decode_kernel_fc` instead of `mla_decode_kernel`.
    /// Enabled by setting `mla_schedule = "metal-mla-fc"` in the kernel
    /// profile. The specialized pipeline is compiled once at engine load
    /// with the model's shape constants baked in.
    pub mla_use_fc: bool,
    /// v2.3.0 A1: when true, attention block routes to
    /// `flash_attn_decode_kernel` (Flash v2 online softmax in tiles of
    /// FLASH_TG=128 tokens). Mutually exclusive with `mla_use_fc`; this
    /// path uses the non-fc kernel signature with all shape args as
    /// runtime buffer args, since the flash kernel reads from the same
    /// buffer layout as `mla_decode_kernel`. Enabled by setting
    /// `mla_schedule = "metal-mla-flash"` in the kernel profile.
    pub mla_use_flash: bool,
    /// v2.3.0 A3: when true, the engine routes `add_inplace + rmsnorm_f32`
    /// pairs to the fused `add_rmsnorm_f32` kernel. Enabled by setting
    /// `residual_fusion = "f32"` in the kernel profile.
    pub residual_fusion_f32: bool,
    pub sampler: Sampler,
    pub _weights_path: PathBuf,

    /// Metal device + library + pipeline cache, threaded to forward
    /// path so kernel dispatchers can target the GPU. `Some` on
    /// Metal-capable boxes (macOS); `None` elsewhere. Per-kernel
    /// dispatch helpers fall back to CPU when `None`.
    pub metal_ctx: Option<MetalContext>,

    /// WB — Phase-2 weight pinning. The LM-head fp16 matrix uploaded
    /// once at load time, reused across every decode token. Eliminates
    /// the per-dispatch `new_buffer_with_bytes` memcpy of ~400 MB that
    /// the byte-slice `gemv_f16_metal` path incurred. `Some` only when
    /// `metal_ctx.is_some()` and the LM head exists (or is tied to
    /// embedding — both share this Buffer).
    pub lm_head_buf: Option<PinnedBuffer>,

    /// Whole GGUF mmap exposed to Metal without copying. Indexed MoE
    /// kernels receive tensor byte offsets into this buffer, so routed
    /// experts can be selected on-GPU without packing selected expert
    /// bytes on the host every token.
    pub weights_mmap_buf: Option<PinnedBuffer>,

    pub kernel_profile: Option<KernelProfile>,
    pub speculate_mode: SpeculateMode,
    pub verify_window: usize,

    /// Path-to-90 step 8 — trained EAGLE-4 draft head, lazy-loaded at
    /// engine construction when `speculate_mode == SpeculateMode::Eagle4`.
    /// `None` for every other speculate mode.
    pub eagle4_head: Option<crate::speculate::eagle4_head::Eagle4Head>,
    /// Path-to-90 step 8 — calibration-sigmoid threshold for the
    /// EAGLE-4 draft. Below this the draft is considered low-confidence
    /// and the verifier's argmax is emitted directly (currently logged
    /// as accept/reject only — see step 8 commit).
    pub eagle4_calib_threshold: f32,
    /// Path-to-90 step 10 follow-up (GPU-side eagle4 capture).
    /// When `true`, `forward_token_final_norm_maybe_read` forces the
    /// per-layer-commit branch (instead of single-TCB fold) and
    /// populates `eagle4_capture` with x_buf + ffn_out_buf reads at
    /// layers {2, 13, 25} plus x_norm_buf at layer 26's pre-MoE
    /// boundary. Toggled around forward_token() calls inside the
    /// Eagle4 decode loop. `false` everywhere else.
    pub eagle4_capture_active: bool,
    /// Path-to-90 step 10 follow-up — destination for GPU-side capture.
    /// Populated by `forward_token_final_norm_maybe_read` when
    /// `eagle4_capture_active` is true; consumed by the Eagle4 decode
    /// loop. `None` otherwise.
    pub eagle4_capture: Option<crate::model::deepseek_v2::Eagle4CaptureBuf>,

    /// v1.2.0-9: Per-layer expert access stats + POSIX madvise offloading.
    /// `Some` when `--max-routed-expert-ram-mb` is set. `None` on V2-Lite default
    /// (all experts fit in RAM, no eviction needed).
    pub expert_cache: Option<crate::model::expert_cache::ExpertCache>,

    /// Wedge 4 — Decode-arena: pre-allocated Metal buffers for the MLA
    /// attention hot path. Allocated once at load time; reused across all
    /// decode steps. Eliminates per-dispatch `new_buffer` overhead.
    /// `Some` only when Metal is available and `gpu_buffer_reuse == "decode-arena"`.
    pub decode_arena: Option<DecodeArena>,

    /// v1.0.0-D: embed table as GPU buffer (f16, hidden × vocab). Enables
    /// embed_lookup_metal_f32_tcb to write x_buf directly without CPU round-trip.
    pub embed_buf: Option<PinnedBuffer>,
    /// v1.0.0-D: final output_norm weight as GPU buffer (f32, hidden).
    /// Used by rmsnorm_metal_buf_tcb in the Wedge C/D final norm step.
    pub final_norm_buf: Option<PinnedBuffer>,
    /// v1.0.0-E: LM-head output buffer (vocab × f32). Persistent; reused each
    /// decode step. Eliminates the ~408 KB per-token logits allocation.
    pub logits_buf: Option<PinnedBuffer>,
    /// v1.0.0-E: Greedy argmax output (1 × u32). GPU writes the winning token
    /// index here; only 4 bytes cross the bus instead of 408 KB logits.
    pub token_buf: Option<PinnedBuffer>,
}

/// Pointer into the mmap'd GGUF for one tensor. Cheap to clone; the
/// dequant happens on demand into a caller-owned buffer.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub offset: usize,
    pub byte_size: usize,
    pub dtype: GgmlType,
    pub n_elems: usize,
}

pub struct Layer {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,

    // MLA attention weights — eagerly dequanted (each layer's attn
    // tensors are tens of MB; cumulative footprint is ~1 GB).
    pub q_proj: Vec<f32>, // optional q-lora-rank path
    pub q_a_proj: Option<Vec<f32>>,
    pub q_a_norm: Option<Vec<f32>>,
    pub q_b_proj: Option<Vec<f32>>,
    pub kv_a_proj_with_mqa: Vec<f32>,
    pub kv_a_norm: Vec<f32>,
    pub kv_b_proj: Vec<f32>,
    pub o_proj: Vec<f32>,

    // FFN — either dense (rare; only `first_k_dense_layers`) or MoE.
    pub mode: LayerMode,

    /// WB — Phase-2 weight pinning. One pre-uploaded `metal::Buffer`
    /// per kernel-bound attention weight, populated by `Engine::load`
    /// when `metal_ctx.is_some()`. The dispatchers
    /// (`gemv_f32_attn_dispatch`) prefer the pinned path when these
    /// are populated. Each Buffer references the same bytes as its
    /// `Vec<f32>` companion; the Vecs stay live as the storage owners.
    pub pinned: LayerPinned,
}

/// WB — pre-uploaded `metal::Buffer` handles for kernel-bound weights
/// on a single transformer layer. Populated only on macOS with
/// `metal_ctx.is_some()`; every field is `None` otherwise. Held on
/// `Layer` so the dispatcher can reach them without a separate field
/// per weight.
#[derive(Default)]
pub struct LayerPinned {
    pub q_a_proj: Option<PinnedBuffer>,
    pub q_b_proj: Option<PinnedBuffer>,
    pub kv_a_proj_with_mqa: Option<PinnedBuffer>,
    pub kv_b_proj: Option<PinnedBuffer>,
    pub o_proj: Option<PinnedBuffer>,
    /// Optional fallback q_proj for non-LoRA models. Phase-0 had a
    /// fallback path; DeepSeek-V2-Lite uses LoRA so this is None in
    /// production but kept for shape-compat.
    pub q_proj: Option<PinnedBuffer>,
    /// Wedge B: pre-uploaded f32 norm weights for TCB rmsnorm dispatches.
    pub attn_norm: Option<PinnedBuffer>,
    pub ffn_norm: Option<PinnedBuffer>,
    /// v1.0.0-C: pre-uploaded q_a_norm and kv_a_norm weights for TCB rmsnorm.
    pub q_a_norm: Option<PinnedBuffer>,
    pub kv_a_norm: Option<PinnedBuffer>,
    /// v1.0.0-C: pre-uploaded MoE gate logit weight for uncounted gate dispatch.
    pub gate_logits_w: Option<PinnedBuffer>,
    /// Dense FFN weights for the leading dense block TCB path.
    pub dense_gate_w: Option<PinnedBuffer>,
    pub dense_up_w: Option<PinnedBuffer>,
    pub dense_down_w: Option<PinnedBuffer>,
}

pub enum LayerMode {
    Dense {
        gate_w: Vec<f32>,
        up_w: Vec<f32>,
        down_w: Vec<f32>,
    },
    MoE {
        gate_logits_w: Vec<f32>, // (n_routed, hidden), eager
        routed_fused: MoEFusedTensors,
        routed: Vec<Expert>, // lazy refs into mmap
        shared_fused: Option<MoEFusedTensors>,
        shared: Vec<Expert>, // lazy refs (length 0 or 1)
    },
}

#[derive(Debug, Clone)]
pub struct MoEFusedTensors {
    pub gate_w: TensorRef,
    pub up_w: TensorRef,
    pub down_w: TensorRef,
}

/// One expert's weight references (lazy). Bytes are borrowed from the
/// mmap on each forward; dequanted into reusable scratch buffers in
/// `ffn`.
pub struct Expert {
    pub gate_w: TensorRef,
    pub up_w: TensorRef,
    pub down_w: TensorRef,
}

/// Wedge M C-3: pre-validated MoE gate setup passed between ffn helpers.
/// Avoids re-checking conditions between gate-encode and moe-dispatch.
#[cfg(target_os = "macos")]
struct FfnMoeSetup {
    routed_gate_off: usize,
    routed_up_off: usize,
    routed_down_off: usize,
    routed_down_dtype: GgmlType,
    shared_gate_off: Option<usize>,
    shared_up_off: Option<usize>,
    shared_down_off: Option<usize>,
    shared_down_dtype: Option<GgmlType>,
    shared_mid: usize,
}

#[cfg(target_os = "macos")]
impl FfnMoeSetup {
    fn q4k_indexed_kernel(q4k_schedule: &str) -> &'static str {
        match q4k_schedule {
            "v2" => "moe_batched_gemm_q4_indexed_v2",
            "v2s" => "moe_batched_gemm_q4_indexed_v2s",
            "llama_port" | "per_shape" => "moe_batched_gemm_q4_indexed_v2",
            // v2t_gu / v2t_gu_serial / v2t_gu_v2 fuse gate+up into one kernel;
            // single-matrix GEMVs (down) use v2t
            "v2t" | "v2t_gu" | "v2t_gu_serial" | "v2t_gu_v2" => "moe_batched_gemm_q4_indexed_v2t",
            _ => "moe_batched_gemm_q4_indexed",
        }
    }

    /// v2.1.0-T2.11: schedule-aware routed-down kernel selection.
    ///
    /// `routed_down_schedule` selects the dispatch variant for Q5_0:
    ///   "basic" (default): the historical 1-row-per-TG tree-reduce kernel
    ///   "v2t":             the new 8-rows-per-TG threadgroup-x_cache simdsum
    ///                      kernel mirroring the Q8_0_v2t pattern.
    ///
    /// Q8_0 and Q4_K paths are unchanged; their schedule is still driven by
    /// `q4k_schedule` (existing `gemm_q4_k_schedule` profile field). This
    /// gives Q5_0 its own opt-in lever so the bench-first gate validates
    /// just the Q5_0 change in isolation.
    fn routed_down_kernel_with_schedule(
        &self,
        q4k_schedule: &str,
        routed_down_schedule: &str,
    ) -> &'static str {
        match self.routed_down_dtype {
            GgmlType::Q8_0 => match q4k_schedule {
                "v2t" | "v2t_gu" | "v2t_gu_serial" | "v2t_gu_v2" => "moe_batched_gemm_q8_0_indexed_v2t",
                _ => "moe_batched_gemm_q8_0_indexed",
            },
            GgmlType::Q5_0 => match routed_down_schedule {
                "v2t" => "moe_batched_gemm_q5_0_indexed_v2t",
                _ => "moe_batched_gemm_q5_0_indexed",
            },
            GgmlType::Q4_K => Self::q4k_indexed_kernel(q4k_schedule),
            _ => unreachable!("ffn_moe_check guards routed down dtype"),
        }
    }

    /// v2.1.0-T2.12: schedule-aware shared-down kernel selection.
    ///
    /// `shared_down_schedule` selects the dispatch variant for Q6_K:
    ///   "basic" (default): 1-row-per-TG tree-reduce kernel
    ///   "v2t":             8-rows-per-TG threadgroup-x_cache simdsum
    ///                      kernel (`moe_batched_gemm_q6_k_indexed_v2t`).
    /// Q4_K shared-down still routes through `q4k_schedule`.
    fn shared_down_kernel_with_schedule(
        &self,
        q4k_schedule: &str,
        shared_down_schedule: &str,
    ) -> &'static str {
        match self.shared_down_dtype {
            Some(GgmlType::Q6_K) => match shared_down_schedule {
                "v2t" => "moe_batched_gemm_q6_k_indexed_v2t",
                _ => "moe_batched_gemm_q6_k_indexed",
            },
            Some(GgmlType::Q4_K) => Self::q4k_indexed_kernel(q4k_schedule),
            None => match shared_down_schedule {
                "v2t" => "moe_batched_gemm_q6_k_indexed_v2t",
                _ => "moe_batched_gemm_q6_k_indexed",
            },
            _ => unreachable!("ffn_moe_check guards shared down dtype"),
        }
    }
}

impl DeepSeekV2 {
    fn q4k_schedule_for_shape(&self, rows: usize, cols: usize) -> &str {
        let Some(profile) = self.kernel_profile.as_ref() else {
            return "scalar";
        };
        if profile.selected.gemm_q4_k_schedule == "per_shape" {
            let key = format!("{rows}x{cols}");
            if let Some(schedule) = profile.selected.gemm_q4_k_schedule_per_shape.get(&key) {
                return schedule.as_str();
            }
        }
        profile.selected.gemm_q4_k_schedule.as_str()
    }

    fn dequant(g: &GgufFile, name: &str) -> Result<Vec<f32>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let bytes = g.tensor_bytes(name).unwrap();
        quant::dequant_to_f32(info, bytes)
    }

    fn dequant_opt(g: &GgufFile, name: &str) -> Result<Option<Vec<f32>>> {
        if g.tensor(name).is_some() {
            Ok(Some(Self::dequant(g, name)?))
        } else {
            Ok(None)
        }
    }

    fn dequant_f16(g: &GgufFile, name: &str) -> Result<Vec<f16>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let bytes = g.tensor_bytes(name).unwrap();
        quant::dequant_to_f16(info, bytes)
    }

    /// Build a `TensorRef` for a single (non-fused) tensor — the
    /// returned ref points into the GGUF mmap.
    fn tensor_ref(g: &GgufFile, name: &str) -> Result<TensorRef> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let n_elems: usize = info.dims.iter().product::<u64>() as usize;
        Ok(TensorRef {
            offset: info.data_offset as usize,
            byte_size: info.byte_size as usize,
            dtype: info.dtype,
            n_elems,
        })
    }

    /// Slice a 3D fused-expert tensor (`blk.{li}.ffn_*_exps.weight`)
    /// into per-expert refs without copying. Each expert occupies a
    /// contiguous byte range because `n_experts` is the outer
    /// (slowest) dimension.
    ///
    /// Per-expert slicing in raw bytes only works when each expert
    /// boundary lands on a quant-block boundary; that's true for the
    /// K-quants we care about here (Q4_K block=256, expert size in
    /// elems = intermediate × hidden = 1408 × 2048 = 2_883_584,
    /// divisible by 256).
    fn fused_expert_refs(g: &GgufFile, name: &str, n_experts: usize) -> Result<Vec<TensorRef>> {
        let info = g
            .tensor(name)
            .ok_or_else(|| Error::Model(format!("missing tensor `{name}`")))?;
        let total_elems: usize = info.dims.iter().product::<u64>() as usize;
        if total_elems % n_experts != 0 {
            return Err(Error::Model(format!(
                "tensor {name}: {total_elems} elems not divisible by {n_experts} experts"
            )));
        }
        let per_elems = total_elems / n_experts;
        let total_bytes = info.byte_size as usize;
        if total_bytes % n_experts != 0 {
            return Err(Error::Model(format!(
                "tensor {name}: {total_bytes} bytes not divisible by {n_experts}; \
                 expert boundary not on a quant-block boundary"
            )));
        }
        let per_bytes = total_bytes / n_experts;
        let base = info.data_offset as usize;
        Ok((0..n_experts)
            .map(|e| TensorRef {
                offset: base + e * per_bytes,
                byte_size: per_bytes,
                dtype: info.dtype,
                n_elems: per_elems,
            })
            .collect())
    }

    /// Dequant a `TensorRef`'s bytes from the engine's mmap into
    /// `buf`, resizing the buffer in place. Reused across calls with
    /// the same shape.
    fn dequant_ref_into(&self, t: &TensorRef, buf: &mut Vec<f32>) -> Result<()> {
        if buf.len() != t.n_elems {
            buf.resize(t.n_elems, 0.0);
        }
        let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
        quant::dequant_into(t.dtype, bytes, buf)
    }
}

impl Engine for DeepSeekV2 {
    fn load(weights: &Path, config: EngineConfig) -> Result<Self> {
        let gguf = GgufFile::open(weights)?;
        let cfg = DeepSeekConfig::from_gguf(&gguf)?;
        let model_id = gguf.name().unwrap_or("deepseek-v2-lite").to_string();

        // Tokenizer: prefer sidecar tokenizer.json, fall back to GGUF.
        let sidecar = weights
            .parent()
            .map(|d| d.join("tokenizer.json"))
            .filter(|p| p.exists());
        let tokenizer = if let Some(p) = sidecar {
            Tokenizer::from_file(&p)?
        } else {
            Tokenizer::from_gguf(&gguf)?
        };

        let embed = Self::dequant_f16(&gguf, "token_embd.weight")?;
        let final_norm = Self::dequant(&gguf, "output_norm.weight")?;
        let lm_head = if gguf.tensor("output.weight").is_some() {
            Some(Self::dequant_f16(&gguf, "output.weight")?)
        } else {
            None
        };

        // Per-layer weights.
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for li in 0..cfg.n_layers {
            let lp = |suf: &str| format!("blk.{li}.{suf}");

            let attn_norm = Self::dequant(&gguf, &lp("attn_norm.weight"))?;
            let ffn_norm = Self::dequant(&gguf, &lp("ffn_norm.weight"))?;

            // MLA layout: kv_a_proj_with_mqa, kv_a_norm, kv_b_proj, q
            // (or q_a_proj/q_a_norm/q_b_proj if q-lora is used), o.
            let q_proj = Self::dequant_opt(&gguf, &lp("attn_q.weight"))?.unwrap_or_default();
            let q_a_proj = Self::dequant_opt(&gguf, &lp("attn_q_a.weight"))?;
            let q_a_norm = Self::dequant_opt(&gguf, &lp("attn_q_a_norm.weight"))?;
            let q_b_proj = Self::dequant_opt(&gguf, &lp("attn_q_b.weight"))?;
            let kv_a_proj_with_mqa = Self::dequant(&gguf, &lp("attn_kv_a_mqa.weight"))?;
            let kv_a_norm = Self::dequant(&gguf, &lp("attn_kv_a_norm.weight"))?;
            let kv_b_proj = Self::dequant(&gguf, &lp("attn_kv_b.weight"))?;
            let o_proj = Self::dequant(&gguf, &lp("attn_output.weight"))?;

            let mode = if li < cfg.first_k_dense_layers {
                LayerMode::Dense {
                    gate_w: Self::dequant(&gguf, &lp("ffn_gate.weight"))?,
                    up_w: Self::dequant(&gguf, &lp("ffn_up.weight"))?,
                    down_w: Self::dequant(&gguf, &lp("ffn_down.weight"))?,
                }
            } else {
                // Modern GGUF MoE layout (llama.cpp post-2024-Q3):
                //
                //   blk.{li}.ffn_gate_exps.weight    — single 3D tensor,
                //   blk.{li}.ffn_up_exps.weight        outer dim is expert id
                //   blk.{li}.ffn_down_exps.weight
                //
                //   blk.{li}.ffn_gate_shexp.weight   — fused-shared MLPs
                //   blk.{li}.ffn_up_shexp.weight       packed into one
                //   blk.{li}.ffn_down_shexp.weight     wider FFN
                //                                      (intermediate × n_shared)
                //
                // Older exports stored one tensor per expert; we no longer
                // try those — if a model needs them, it predates dismantle.
                let gate_logits_w = Self::dequant(&gguf, &lp("ffn_gate_inp.weight"))?;

                let routed_fused = MoEFusedTensors {
                    gate_w: Self::tensor_ref(&gguf, &lp("ffn_gate_exps.weight"))?,
                    up_w: Self::tensor_ref(&gguf, &lp("ffn_up_exps.weight"))?,
                    down_w: Self::tensor_ref(&gguf, &lp("ffn_down_exps.weight"))?,
                };
                let gate_exps = Self::fused_expert_refs(
                    &gguf,
                    &lp("ffn_gate_exps.weight"),
                    cfg.n_routed_experts,
                )?;
                let up_exps = Self::fused_expert_refs(
                    &gguf,
                    &lp("ffn_up_exps.weight"),
                    cfg.n_routed_experts,
                )?;
                let down_exps = Self::fused_expert_refs(
                    &gguf,
                    &lp("ffn_down_exps.weight"),
                    cfg.n_routed_experts,
                )?;

                let routed: Vec<Expert> = gate_exps
                    .into_iter()
                    .zip(up_exps.into_iter())
                    .zip(down_exps.into_iter())
                    .map(|((g, u), d)| Expert {
                        gate_w: g,
                        up_w: u,
                        down_w: d,
                    })
                    .collect();

                // Shared experts are stored as ONE fused MLP whose
                // intermediate width is `n_shared_experts * moe_intermediate`.
                // Length-1 vec; same dispatch path as routed.
                let (shared_fused, shared) = if cfg.n_shared_experts > 0 {
                    let fused = MoEFusedTensors {
                        gate_w: Self::tensor_ref(&gguf, &lp("ffn_gate_shexp.weight"))?,
                        up_w: Self::tensor_ref(&gguf, &lp("ffn_up_shexp.weight"))?,
                        down_w: Self::tensor_ref(&gguf, &lp("ffn_down_shexp.weight"))?,
                    };
                    (
                        Some(fused.clone()),
                        vec![Expert {
                            gate_w: fused.gate_w.clone(),
                            up_w: fused.up_w.clone(),
                            down_w: fused.down_w.clone(),
                        }],
                    )
                } else {
                    (None, Vec::new())
                };
                LayerMode::MoE {
                    gate_logits_w,
                    routed_fused,
                    routed,
                    shared_fused,
                    shared,
                }
            };

            layers.push(Layer {
                attn_norm,
                ffn_norm,
                q_proj,
                q_a_proj,
                q_a_norm,
                q_b_proj,
                kv_a_proj_with_mqa,
                kv_a_norm,
                kv_b_proj,
                o_proj,
                mode,
                pinned: LayerPinned::default(),
            });
        }

        let max_seq = config.max_seq_len.min(cfg.max_seq_len);
        let kv = KvCache::new(
            cfg.n_layers,
            max_seq,
            cfg.n_kv_heads,
            cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
        );

        let mla_schedule_str: &str = config
            .kernel_profile
            .as_ref()
            .map(|p| p.selected.mla_schedule.as_str())
            .unwrap_or("");
        let mla_metal = matches!(
            mla_schedule_str,
            "metal-mla" | "metal-mla-fc" | "metal-mla-flash"
        );
        let mla_use_fc = mla_schedule_str == "metal-mla-fc";
        let mla_use_flash = mla_schedule_str == "metal-mla-flash";
        let residual_fusion_f32 = config
            .kernel_profile
            .as_ref()
            .map(|p| p.selected.residual_fusion.as_str() == "f32")
            .unwrap_or(false);
        let (mla_c_kv, mla_k_pe) = if mla_metal {
            let c_kv = (0..cfg.n_layers)
                .map(|_| vec![0.0f32; max_seq * cfg.kv_lora_rank])
                .collect();
            let k_pe = (0..cfg.n_layers)
                .map(|_| vec![0.0f32; max_seq * cfg.qk_rope_head_dim])
                .collect();
            (c_kv, k_pe)
        } else {
            (Vec::new(), Vec::new())
        };

        let sampler = Sampler::new(0);

        // Metal context: built once per model, owned for the model's
        // lifetime. Errors here (no GPU, shader compile failure) are
        // soft — `None` falls back to CPU kernels in every dispatcher.
        let metal_ctx = MetalContext::new_with_trace(config.trace_dispatch).ok();
        let device_name = metal_ctx.as_ref().map(|ctx| ctx.device_name());
        if let Some(profile) = config.kernel_profile.as_ref() {
            profile.validate_for_gguf(&gguf, device_name.as_deref())?;
        }

        // v2.3.0 A4: compile the function-constant-specialized
        // mla_decode_kernel_fc pipeline once with the model's shape
        // constants baked in. After this call, regular
        // `ctx.pipeline("mla_decode_kernel_fc")` returns the specialized
        // pipeline directly — the TCB dispatcher needs no special path.
        #[cfg(target_os = "macos")]
        if mla_use_fc {
            if let Some(ref ctx) = metal_ctx {
                let n_heads = cfg.n_heads as u32;
                let qk_nope = cfg.qk_nope_head_dim as u32;
                let qk_rope = cfg.qk_rope_head_dim as u32;
                let v_head = cfg.v_head_dim as u32;
                let kv_lora = cfg.kv_lora_rank as u32;
                let head_total = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
                let scale: f32 = 1.0 / (head_total as f32).sqrt();
                ctx.register_specialized_pipeline("mla_decode_kernel_fc", || {
                    use metal::{FunctionConstantValues, MTLDataType};
                    let fcv = FunctionConstantValues::new();
                    fcv.set_constant_value_at_index(
                        &n_heads as *const u32 as *const _,
                        MTLDataType::UInt, 0,
                    );
                    fcv.set_constant_value_at_index(
                        &qk_nope as *const u32 as *const _,
                        MTLDataType::UInt, 1,
                    );
                    fcv.set_constant_value_at_index(
                        &qk_rope as *const u32 as *const _,
                        MTLDataType::UInt, 2,
                    );
                    fcv.set_constant_value_at_index(
                        &v_head as *const u32 as *const _,
                        MTLDataType::UInt, 3,
                    );
                    fcv.set_constant_value_at_index(
                        &kv_lora as *const u32 as *const _,
                        MTLDataType::UInt, 4,
                    );
                    fcv.set_constant_value_at_index(
                        &scale as *const f32 as *const _,
                        MTLDataType::Float, 5,
                    );
                    fcv
                })?;
            }
        }

        // v2.3.0 A4.2: register fc-specialized MoE Q4 routed gate+up kernel
        // when the kernel profile asks for it. Constants:
        //   kFcMoeRows = moe_intermediate (1408 for V2-Lite)
        //   kFcMoeCols = hidden            (2048 for V2-Lite)
        #[cfg(target_os = "macos")]
        if config.kernel_profile.as_ref()
            .map(|p| p.selected.gemm_q4_k_schedule.as_str() == "v2t_gu_v2_fc")
            .unwrap_or(false)
        {
            if let Some(ref ctx) = metal_ctx {
                let moe_rows = cfg.moe_intermediate as u32;
                let moe_cols = cfg.hidden as u32;
                ctx.register_specialized_pipeline(
                    "moe_batched_gemm_q4_indexed_v2t_gu_v2_fc",
                    || {
                        use metal::{FunctionConstantValues, MTLDataType};
                        let fcv = FunctionConstantValues::new();
                        fcv.set_constant_value_at_index(
                            &moe_rows as *const u32 as *const _,
                            MTLDataType::UInt, 10,
                        );
                        fcv.set_constant_value_at_index(
                            &moe_cols as *const u32 as *const _,
                            MTLDataType::UInt, 11,
                        );
                        fcv
                    },
                )?;
            }
        }
        let speculate_mode = if config.speculate && config.speculate_mode == SpeculateMode::Off {
            SpeculateMode::ExactShared
        } else {
            config.speculate_mode
        };
        let verify_window = config.verify_window;

        // Path-to-90 step 8 — load the EAGLE-4 draft head when mode demands it.
        // Both the head NPZ and the frozen V2-Lite NPZ are required; a future
        // commit will let frozen weights come from the already-loaded GGUF
        // tensors (token_embd, output_norm, lm_head) and skip the NPZ read.
        let eagle4_head = if speculate_mode == SpeculateMode::Eagle4 {
            use crate::speculate::eagle4_head::{Eagle4FrozenWeights, Eagle4Head};
            let head_path = config.eagle4_head_path.as_ref().ok_or_else(|| {
                Error::Model(
                    "--speculate eagle4 requires --draft-head <path to eagle4 .npz>".into(),
                )
            })?;
            let frozen_path = config.eagle4_frozen_path.as_ref().ok_or_else(|| {
                Error::Model(
                    "--speculate eagle4 requires --eagle4-frozen <path to v2lite_frozen.npz>"
                        .into(),
                )
            })?;
            let mut head = Eagle4Head::from_npz(head_path)?;
            let frozen = Eagle4FrozenWeights::from_npz(frozen_path)?;
            head.set_frozen(frozen);
            Some(head)
        } else {
            None
        };
        let eagle4_calib_threshold = config.eagle4_calib_threshold;

        // WB weight-pinning: when Metal is alive, upload the LM-head
        // fp16 matrix to a single Buffer that lives for the model's
        // lifetime. The byte-slice `gemv_f16_metal` path was memcpying
        // ~400 MB on every decode token; the pinned variant references
        // this Buffer instead. Tied-embedding case (lm_head=None) uses
        // the embed table — shape-compatible since both are
        // (vocab × hidden) fp16.
        let lm_head_buf = {
            #[cfg(target_os = "macos")]
            {
                metal_ctx.as_ref().map(|ctx| {
                    let w_f16: &[f16] = lm_head.as_deref().unwrap_or(&embed);
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f16, u8>(w_f16))
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = &metal_ctx;
                None
            }
        };

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

        // WB per-layer attn-weight pinning: q_a_proj, q_b_proj,
        // kv_a_proj_with_mqa, kv_b_proj, o_proj. Each is a fp32 matrix
        // already eagerly dequanted (Vec<f32> on Layer). We upload
        // each to its own Buffer once; the gemv_f32_attn dispatcher
        // routes through the pinned path when these are populated.
        // The Vec<f32> stays live as the storage owner; Buffer is just
        // a Metal-side handle into the same memory layout.
        #[cfg(target_os = "macos")]
        if let Some(ctx) = metal_ctx.as_ref() {
            for layer in layers.iter_mut() {
                let upload =
                    |w: &[f32]| ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(w));
                // q_a, q_b, kv_a, o_proj are pinned as f16 to halve weight read bandwidth.
                // The Vec<f32> stays live for the CPU fallback path.
                let upload_f16 = |w: &[f32]| {
                    let f16_vec: Vec<half::f16> =
                        w.iter().map(|&v| half::f16::from_f32(v)).collect();
                    ctx.new_buffer_with_bytes(bytemuck::cast_slice::<half::f16, u8>(&f16_vec))
                };
                if let Some(qa) = layer.q_a_proj.as_deref() {
                    layer.pinned.q_a_proj = Some(upload_f16(qa));
                }
                if let Some(qb) = layer.q_b_proj.as_deref() {
                    layer.pinned.q_b_proj = Some(upload_f16(qb));
                }
                layer.pinned.kv_a_proj_with_mqa = Some(upload_f16(&layer.kv_a_proj_with_mqa));
                layer.pinned.kv_b_proj = Some(upload(&layer.kv_b_proj));
                layer.pinned.o_proj = Some(upload_f16(&layer.o_proj));
                if !layer.q_proj.is_empty() {
                    // v2.1.0-T2.13: pin q_proj as f16. V2-Lite uses the
                    // non-LoRA q_proj path (q_lora_rank=0), making this
                    // kernel a ~11% GPU hotspot in the f32-weight form.
                    // f16w variant halves the weight reads → ~4x faster
                    // per-call. The legacy CPU pair dispatch at
                    // gemv_f32_attn_pair_dispatch:1739 already guards
                    // against undersized pinned buffers, so fallback
                    // remains safe.
                    layer.pinned.q_proj = Some(upload_f16(&layer.q_proj));
                }
                // Wedge B: pre-upload small norm weight buffers for TCB rmsnorm.
                layer.pinned.attn_norm = Some(upload(&layer.attn_norm));
                layer.pinned.ffn_norm = Some(upload(&layer.ffn_norm));
                // v1.0.0-C: pre-upload q_a_norm and kv_a_norm for TCB attention path.
                if let Some(qan) = layer.q_a_norm.as_deref() {
                    layer.pinned.q_a_norm = Some(upload(qan));
                }
                layer.pinned.kv_a_norm = Some(upload(&layer.kv_a_norm));
                // v1.0.0-C: pre-upload gate logit weight for uncounted MoE gate dispatch.
                if let LayerMode::MoE { gate_logits_w, .. } = &layer.mode {
                    layer.pinned.gate_logits_w = Some(upload(gate_logits_w));
                }
                if let LayerMode::Dense { gate_w, up_w, down_w } = &layer.mode {
                    layer.pinned.dense_gate_w = Some(upload(gate_w));
                    layer.pinned.dense_up_w = Some(upload(up_w));
                    layer.pinned.dense_down_w = Some(upload(down_w));
                }
            }
        }

        // Wedge 4 — Decode-arena: allocate pre-warmed Metal buffers when
        // the selected profile requests gpu_buffer_reuse == "decode-arena".
        #[cfg(target_os = "macos")]
        let decode_arena = {
            let wants_arena = config
                .kernel_profile
                .as_ref()
                .map(|p| p.selected.gpu_buffer_reuse == "decode-arena")
                .unwrap_or(false);
            if wants_arena {
                metal_ctx.as_ref().map(|ctx| {
                    DecodeArena::new(
                        ctx,
                        cfg.n_heads,
                        cfg.qk_nope_head_dim,
                        cfg.qk_rope_head_dim,
                        cfg.v_head_dim,
                        cfg.kv_lora_rank,
                        cfg.hidden,
                        max_seq,
                        cfg.n_routed_experts,
                        cfg.top_k_routed,
                        cfg.moe_intermediate,
                        cfg.n_shared_experts,
                        cfg.ffn_intermediate,
                        cfg.q_lora_rank,
                        cfg.n_layers.saturating_sub(cfg.first_k_dense_layers),
                        8, // max_batch_size for Phase 5A K-token batched verify
                    )
                })
            } else {
                None
            }
        };
        #[cfg(not(target_os = "macos"))]
        let decode_arena: Option<DecodeArena> = None;

        // GPU-resident KV cache — persistent per-layer Metal buffers, one entry per seq slot.
        // Allocated only when Metal + MLA are both active. Mirrors mla_c_kv / mla_k_pe.
        #[cfg(target_os = "macos")]
        let (mla_c_kv_gpu, mla_k_pe_gpu) = {
            if metal_ctx.is_some() && mla_metal {
                let ctx = metal_ctx.as_ref().unwrap();
                let c_kv: Vec<PinnedBuffer> = (0..cfg.n_layers)
                    .map(|_| ctx.new_buffer(max_seq * cfg.kv_lora_rank * std::mem::size_of::<f32>()))
                    .collect();
                let k_pe: Vec<PinnedBuffer> = (0..cfg.n_layers)
                    .map(|_| ctx.new_buffer(max_seq * cfg.qk_rope_head_dim * std::mem::size_of::<f32>()))
                    .collect();
                (c_kv, k_pe)
            } else {
                (vec![], vec![])
            }
        };
        #[cfg(not(target_os = "macos"))]
        let (mla_c_kv_gpu, mla_k_pe_gpu): (Vec<PinnedBuffer>, Vec<PinnedBuffer>) = (vec![], vec![]);

        // v1.0.0-D: upload embed table + final norm weight to GPU once.
        #[cfg(target_os = "macos")]
        let (embed_buf, final_norm_buf) = {
            if let Some(ctx) = metal_ctx.as_ref() {
                let eb = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<half::f16, u8>(&embed));
                let fnb = ctx.new_buffer_with_bytes(bytemuck::cast_slice::<f32, u8>(&final_norm));
                (Some(eb), Some(fnb))
            } else {
                (None, None)
            }
        };
        #[cfg(not(target_os = "macos"))]
        let (embed_buf, final_norm_buf): (Option<crate::metal::PinnedBuffer>, Option<crate::metal::PinnedBuffer>) = (None, None);

        // v1.0.0-E: logits buffer (vocab × f32) and token buffer (1 × u32).
        // Allocated once; reused every greedy decode step to avoid per-token heap churn.
        #[cfg(target_os = "macos")]
        let (logits_buf, token_buf) = {
            if let Some(ctx) = metal_ctx.as_ref() {
                let lb = ctx.new_buffer(cfg.vocab_size * std::mem::size_of::<f32>());
                let tb = ctx.new_buffer(std::mem::size_of::<u32>());
                (Some(lb), Some(tb))
            } else {
                (None, None)
            }
        };
        #[cfg(not(target_os = "macos"))]
        let (logits_buf, token_buf): (Option<crate::metal::PinnedBuffer>, Option<crate::metal::PinnedBuffer>) = (None, None);

        // v1.2.0-9: save dimensions before cfg is moved into the struct literal.
        let cfg_n_layers = cfg.n_layers;
        let cfg_n_routed_experts = cfg.n_routed_experts;

        Ok(Self {
            config: cfg,
            tokenizer,
            model_id,
            gguf,
            embed,
            final_norm,
            lm_head,
            layers,
            kv,
            mla_c_kv,
            mla_k_pe,
            mla_c_kv_gpu,
            mla_k_pe_gpu,
            mla_kv_gpu_synced: false,
            mla_use_fc,
            mla_use_flash,
            residual_fusion_f32,
            sampler,
            _weights_path: weights.to_owned(),
            metal_ctx,
            lm_head_buf,
            weights_mmap_buf,
            kernel_profile: config.kernel_profile,
            speculate_mode,
            verify_window,
            eagle4_head,
            eagle4_calib_threshold,
            eagle4_capture_active: false,
            eagle4_capture: None,
            decode_arena,
            embed_buf,
            final_norm_buf,
            logits_buf,
            token_buf,
            // v1.2.0-9: allocate ExpertCache when --max-routed-expert-ram-mb is set.
            // ExpertCache is always Some when the flag is present; madvise calls are
            // no-ops until model_base_addr is attached (Mixtral only for v1.2.0).
            expert_cache: config.max_routed_expert_ram_mb.map(|_| {
                crate::model::expert_cache::ExpertCache::new(
                    cfg_n_layers,
                    cfg_n_routed_experts,
                    256, // rolling window: 256 tokens
                )
            }),
        })
    }

    fn generate(
        &mut self,
        req: GenerateRequest,
        sink: &mut dyn FnMut(StreamEvent),
    ) -> Result<GenStats> {
        use std::sync::atomic::Ordering;

        if let Some(seed) = req.sampling.seed {
            self.sampler = Sampler::new(seed);
        }
        if self.speculate_mode == SpeculateMode::ExactShared {
            if req.sampling.temperature > 0.0 {
                return Err(Error::Model(
                    "--speculate exact-shared currently requires temperature=0".into(),
                ));
            }
            if req.sampling.repetition_penalty != 1.0 {
                return Err(Error::Model(
                    "--speculate exact-shared currently requires repetition_penalty=1.0".into(),
                ));
            }
            if !matches!(self.verify_window, 4 | 8 | 16) {
                return Err(Error::Model(format!(
                    "--verify-window must be 4, 8, or 16 for exact-shared; got {}",
                    self.verify_window
                )));
            }
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
        let prompt_len = prompt_ids.len();
        let mut stats = GenStats {
            prompt_tokens: prompt_len,
            profile_id: self.kernel_profile.as_ref().map(|p| p.profile_id.clone()),
            device_id: self
                .kernel_profile
                .as_ref()
                .map(|p| p.device_name.clone())
                .or_else(|| self.metal_ctx.as_ref().map(|ctx| ctx.device_name())),
            ..Default::default()
        };

        // Prefill — process each prompt token sequentially. In Phase 0
        // there is no batched prefill kernel; the win comes in Phase 2.
        // Both the abort flag and the per-step watchdog are checked at
        // each token boundary.
        self.kv.reset();
        for v in &mut self.mla_c_kv {
            v.fill(0.0);
        }
        for v in &mut self.mla_k_pe {
            v.fill(0.0);
        }
        self.mla_kv_gpu_synced = false;
        let prefill_start = Instant::now();
        let mut prefill_aborted = false;
        for (i, &t) in prompt_ids.iter().enumerate() {
            if abort_set(&req) {
                prefill_aborted = true;
                break;
            }
            let step_start = Instant::now();
            let _ = self.forward_token(t, i)?;
            if stall_active && step_start.elapsed() > stall_limit {
                prefill_aborted = true;
                break;
            }
        }
        stats.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        if prefill_aborted {
            sink(StreamEvent::Done {
                reason: StopReason::Aborted,
                stats: stats.clone(),
            });
            return Ok(stats);
        }

        // Decode loop.
        let decode_start = Instant::now();
        let mut last_id = *prompt_ids.last().unwrap();
        let mut produced = 0usize;
        let mut reason = StopReason::MaxTokens;
        let eos = self.tokenizer.eos_id();

        if self.speculate_mode == crate::SpeculateMode::ExactShared {
            // Speculative decode: draft with shared-only model, verify with full model.
            // Temperature must be 0 (greedy) — validated above.
            let spec_k = self.verify_window;
            let spec_log = std::env::var("DISMANTLE_SPEC_LOG").is_ok();
            let mut pos = prompt_len;

            'spec_loop: while produced < req.max_new_tokens {
                if abort_set(&req) {
                    reason = StopReason::Aborted;
                    break;
                }
                let step_start = Instant::now();
                let draft_start_seq = self.kv.seq_len;
                let remaining = req.max_new_tokens - produced;

                // Clamp draft window: always draft ≥ 1 if budget allows.
                let actual_k = if remaining <= 1 { 0 } else { spec_k.min(remaining - 1) };

                if actual_k == 0 {
                    // Too close to budget — single greedy step.
                    let mut logits = self.forward_token(last_id, pos)?;
                    let next_id = self.sampler.sample(&mut logits, &req.sampling);
                    self.sampler.record(next_id);
                    let text = self.tokenizer.decode_one(next_id).unwrap_or_default();
                    sink(StreamEvent::Token { id: next_id, text });
                    produced += 1;
                    if Some(next_id) == eos { reason = StopReason::Eos; }
                    break 'spec_loop;
                }

                // --- DRAFT: actual_k tokens via shared-only model ---
                let mut draft_ids: Vec<u32> = Vec::with_capacity(actual_k);
                let mut tmp_last = last_id;
                let draft_t0 = Instant::now();
                for k in 0..actual_k {
                    let draft_id = self.forward_token_shared_only_argmax(tmp_last, pos + k)?;
                    draft_ids.push(draft_id);
                    tmp_last = draft_id;
                }
                let draft_ms = draft_t0.elapsed().as_secs_f64() * 1000.0;

                // Rollback KV: draft used future slots as scratch; verifier overwrites them.
                self.kv.seq_len = draft_start_seq;

                // --- VERIFY: stop at the first mismatch ---
                let verify_t0 = Instant::now();
                tmp_last = last_id;
                let use_profiled_greedy = self.profiled_greedy_enabled(&req.sampling);
                let verify_result =
                    crate::speculate::shared::verify_draft_ids_until_mismatch(&draft_ids, |k| {
                        let id = self.forward_token_argmax(tmp_last, pos + k, use_profiled_greedy)?;
                        if id == draft_ids[k] {
                            tmp_last = draft_ids[k]; // feed draft tokens through verify context
                        }
                        Ok(id)
                    })?;
                let first_reject = verify_result.accepted_count;
                let bonus_id = match verify_result.first_divergent_token {
                    Some(id) => id,
                    None => self.forward_token_argmax(tmp_last, pos + actual_k, use_profiled_greedy)?,
                };
                let verify_ms = verify_t0.elapsed().as_secs_f64() * 1000.0;

                // Rollback KV to the correct accepted length.
                self.kv.seq_len = draft_start_seq + first_reject + 1;

                stats.draft_accepted += first_reject;
                stats.draft_rejected += actual_k - first_reject;

                // --- EMIT accepted drafts ---
                for k in 0..first_reject {
                    let id = draft_ids[k];
                    let text = self.tokenizer.decode_one(id).unwrap_or_default();
                    sink(StreamEvent::Token { id, text });
                    self.sampler.record(id);
                    produced += 1;
                    if Some(id) == eos {
                        reason = StopReason::Eos;
                        break 'spec_loop;
                    }
                    if produced >= req.max_new_tokens {
                        break 'spec_loop;
                    }
                }

                // --- EMIT correction / bonus token ---
                let text = self.tokenizer.decode_one(bonus_id).unwrap_or_default();
                sink(StreamEvent::Token { id: bonus_id, text });
                self.sampler.record(bonus_id);
                produced += 1;
                last_id = bonus_id;
                pos += first_reject + 1;

                if spec_log {
                    let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[spec] accept={}/{} draft={:.1}ms verify={:.1}ms step={:.1}ms emit={} tps={:.1}",
                        first_reject, actual_k, draft_ms, verify_ms, step_ms,
                        first_reject + 1,
                        (first_reject + 1) as f64 / (step_ms / 1000.0)
                    );
                }

                if Some(bonus_id) == eos {
                    reason = StopReason::Eos;
                    break;
                }
                if stall_active && step_start.elapsed() > stall_limit {
                    reason = StopReason::Aborted;
                    break;
                }
            }
        } else if self.speculate_mode == crate::SpeculateMode::NGram {
            // N-gram speculative decoding: zero-cost draft from token history,
            // serial verification with full model.
            // Requires greedy (temperature=0) for exact parity.
            if req.sampling.temperature > 0.0 {
                return Err(Error::Model(
                    "--speculate ngram currently requires temperature=0".into(),
                ));
            }
            let spec_k = self.verify_window;
            let spec_log = std::env::var("DISMANTLE_SPEC_LOG").is_ok();
            // Phase 5A: batched verify replaces profiled greedy; retained for near-budget single steps.
            let _use_profiled_greedy = self.profiled_greedy_enabled(&req.sampling);

            // Seed n-gram history with prompt tokens.
            let mut ngram = crate::speculate::ngram::NGramDraft::new(3);
            for &t in &prompt_ids {
                ngram.note_token(t);
            }

            let mut pos = prompt_len;
            'ngram_loop: while produced < req.max_new_tokens {
                if abort_set(&req) {
                    reason = StopReason::Aborted;
                    break;
                }
                let step_start = Instant::now();
                let remaining = req.max_new_tokens - produced;

                // Clamp draft window.
                let actual_k = if remaining <= 1 { 0 } else { spec_k.min(remaining - 1) };

                if actual_k == 0 {
                    // Near budget — single greedy step.
                    let next_id = match self.forward_token_greedy(last_id, pos)? {
                        Some(t) => t,
                        None => {
                            let mut logits = self.forward_token(last_id, pos)?;
                            self.sampler.sample(&mut logits, &req.sampling)
                        }
                    };
                    self.sampler.record(next_id);
                    ngram.note_token(next_id);
                    let text = self.tokenizer.decode_one(next_id).unwrap_or_default();
                    sink(StreamEvent::Token { id: next_id, text });
                    produced += 1;
                    if Some(next_id) == eos { reason = StopReason::Eos; }
                    break 'ngram_loop;
                }

                // --- DRAFT: n-gram lookup (zero compute) ---
                let draft_ids = ngram.propose(actual_k);

                if draft_ids.is_empty() {
                    // No draft available — single greedy step.
                    let next_id = match self.forward_token_greedy(last_id, pos)? {
                        Some(t) => t,
                        None => {
                            let mut logits = self.forward_token(last_id, pos)?;
                            self.sampler.sample(&mut logits, &req.sampling)
                        }
                    };
                    self.sampler.record(next_id);
                    ngram.note_token(next_id);
                    let text = self.tokenizer.decode_one(next_id).unwrap_or_default();
                    sink(StreamEvent::Token { id: next_id, text });
                    produced += 1;
                    if Some(next_id) == eos { reason = StopReason::Eos; break 'ngram_loop; }
                    if stall_active && step_start.elapsed() > stall_limit {
                        reason = StopReason::Aborted; break 'ngram_loop;
                    }
                    last_id = next_id;
                    pos += 1;
                    continue;
                }

                let draft_actual_k = draft_ids.len();

                // Save KV state before verify so we can rollback.
                let draft_start_seq = self.kv.seq_len;

                // --- VERIFY (Phase 5A): batched forward — [last_id, draft_ids...] in one TCB.
                // Batch: [last_id, d0, d1, ..., dK-1] at positions [pos, pos+1, ..., pos+K].
                // logits_batch[k] = model output after processing batch[k] at pos+k.
                // Acceptance: argmax(logits_batch[k]) should match draft_ids[k].
                // Bonus: argmax(logits_batch[K]) if all accepted; else argmax(logits_batch[first_reject]).
                let batch_tokens: Vec<u32> = std::iter::once(last_id)
                    .chain(draft_ids.iter().copied())
                    .collect();
                let batch_positions: Vec<usize> = (0..=draft_actual_k)
                    .map(|ki| pos + ki)
                    .collect();
                let logits_batch = self.forward_tokens_batched(&batch_tokens, &batch_positions)?;

                let mut first_reject = 0usize;
                let mut correction_id: Option<u32> = None;
                for k in 0..draft_actual_k {
                    let pred = crate::kernels::argmax_f32(&logits_batch[k]);
                    if pred != draft_ids[k] {
                        correction_id = Some(pred);
                        break;
                    }
                    first_reject += 1;
                }
                let bonus_id = correction_id.unwrap_or_else(|| {
                    crate::kernels::argmax_f32(&logits_batch[draft_actual_k])
                });

                // Rollback KV to the correct accepted prefix length.
                self.kv.seq_len = draft_start_seq + first_reject + 1;

                stats.draft_accepted += first_reject;
                stats.draft_rejected += draft_actual_k - first_reject;

                // --- EMIT accepted draft tokens ---
                for k in 0..first_reject {
                    let id = draft_ids[k];
                    let text = self.tokenizer.decode_one(id).unwrap_or_default();
                    sink(StreamEvent::Token { id, text });
                    self.sampler.record(id);
                    ngram.note_token(id);
                    produced += 1;
                    if Some(id) == eos { reason = StopReason::Eos; break 'ngram_loop; }
                    if produced >= req.max_new_tokens { break 'ngram_loop; }
                }

                // --- EMIT bonus/correction ---
                let text = self.tokenizer.decode_one(bonus_id).unwrap_or_default();
                sink(StreamEvent::Token { id: bonus_id, text });
                self.sampler.record(bonus_id);
                ngram.note_token(bonus_id);
                produced += 1;
                last_id = bonus_id;
                pos += first_reject + 1;

                if spec_log {
                    let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
                    eprintln!(
                        "[ngram-spec] accept={}/{} step={:.1}ms emit={} tps={:.1}",
                        first_reject, draft_actual_k, step_ms,
                        first_reject + 1,
                        (first_reject + 1) as f64 / (step_ms / 1000.0)
                    );
                }

                if Some(bonus_id) == eos { reason = StopReason::Eos; break; }
                if stall_active && step_start.elapsed() > stall_limit {
                    reason = StopReason::Aborted; break;
                }
            }
        } else if self.speculate_mode == crate::SpeculateMode::Eagle4 {
            // Path-to-90 step 9 — EAGLE-4 spec decode, K=1, GPU emission.
            //
            // Per output token:
            //   1. GPU forward_token_argmax (production Wedge C path)
            //      advances GPU KV and gives V2-Lite's canonical argmax.
            //      This is the EMITTED token — bit-identical to
            //      SpeculateMode::Off by construction.
            //   2. CPU walk (forward_token_eagle4_capture_with_argmax)
            //      captures EAGLE-4's 5-input hidden bundle, advances
            //      CPU MLA KV mirror. seq_len is save/restored around the
            //      CPU walk so the two paths don't double-bump the
            //      shared counter. CPU KV diverges from GPU KV over time
            //      (the foundation-halt CPU attention() divergence —
            //      see reports/path_to_90/foundation_halt.md); eagle4
            //      stats are therefore noisy until the chip lands. But
            //      the emitted output is GPU-clean.
            //   3. head.propose(eagle4_inputs, K=1) feeds the (possibly
            //      degraded) hiddens to the head; the draft prediction
            //      is compared to the GPU argmax and accept/reject is
            //      tallied for stats.
            //
            // K-batched verify (forward_tokens_batched_for_test on K
            // candidates with longest-matching-prefix acceptance) is
            // deferred to Stage 2 Path B kernels. At K=1 spec decode
            // degenerates to "emit verifier's argmax" → step 9's
            // bit-identical regression passes by construction.
            //
            // Greedy only — sampling temp > 0 disabled.
            if req.sampling.temperature > 0.0 {
                return Err(Error::Model(
                    "--speculate eagle4 currently requires temperature=0".into(),
                ));
            }

            let calib_threshold = self.eagle4_calib_threshold;
            let spec_log = std::env::var("DISMANTLE_SPEC_LOG").is_ok();

            // Path-to-90 step 7 — lazy Metal-pin of eagle4 head weights.
            // Convert all 10 large gemv weights from f32 to f16 + upload
            // to Metal shared buffers once; subsequent decode steps use
            // gemv_f16_metal_pinned for the head forward (~5× faster
            // than CPU gemv_f32 at the head's matrix shapes).
            #[cfg(target_os = "macos")]
            {
                if let Some(head_mut) = self.eagle4_head.as_mut() {
                    if !head_mut.has_metal_pinned() {
                        if let Some(ctx) = self.metal_ctx.as_ref() {
                            head_mut.pin_metal(ctx);
                        }
                    }
                }
            }
            let use_profiled_greedy = self.profiled_greedy_enabled(&req.sampling);
            let mut head = self
                .eagle4_head
                .take()
                .ok_or_else(|| Error::Model("--speculate eagle4 requires --draft-head".into()))?;

            for step in 0..req.max_new_tokens {
                if abort_set(&req) {
                    reason = StopReason::Aborted;
                    break;
                }
                let pos = prompt_len + step;
                let step_start = Instant::now();

                // Single GPU forward with capture flag — bit-identical to Off
                // (Wedge C path) AND populates self.eagle4_capture at layers
                // {2, 13, 25} + x_norm_buf at layer 26. The flag forces the
                // per-layer-commit branch in forward_token_final_norm_maybe_read
                // (use_single_tcb = false when eagle4_capture_active), which
                // adds ~26 commit+wait overheads (~4 ms total at ~150 µs each)
                // vs the single-TCB fast path. Net Eagle4 mode cost: ~41 ms /
                // token vs Off's ~37 ms.
                self.eagle4_capture_active = true;
                let argmax_result = self.forward_token_argmax(last_id, pos, use_profiled_greedy);
                self.eagle4_capture_active = false;
                let v2_argmax = argmax_result?;

                let capture = self.eagle4_capture.take().ok_or_else(|| {
                    Error::Model(
                        "--speculate eagle4: GPU capture buffer absent after forward_token; \
                         Wedge C path may not have executed"
                            .into(),
                    )
                })?;

                // h_shared: prefer the production MoE kernel's
                // `moe_shared_out_buf` (read in the per-layer-commit
                // capture above) — zero extra dispatch cost, same number
                // the fused kernel computed during the V2-Lite forward.
                // Fall back to cpu_shared_expert_forward when the GPU
                // buffer is all zeros (e.g. layer 26 is dense, or the
                // capture missed the read for some reason).
                let h_shared_norm2: f32 = capture.h_shared_gpu.iter().map(|v| v * v).sum();
                let h_shared = if h_shared_norm2 > 0.0 {
                    capture.h_shared_gpu.clone()
                } else {
                    self.cpu_shared_expert_forward(26, &capture.x_norm_26)?
                };

                // Eagle4 head forward — Metal path when pinning has run,
                // CPU fallback otherwise. Both skip the CPU LM head
                // gemv (replaced by GPU argmax on V2-Lite's lm_head_buf
                // below).
                #[cfg(target_os = "macos")]
                let head_out = if head.has_metal_pinned() {
                    let ctx = self.metal_ctx.as_ref().ok_or_else(|| {
                        Error::Model("eagle4 metal forward: no metal context".into())
                    })?;
                    head.forward_full_metal_no_lm_head(
                        ctx,
                        last_id,
                        &capture.h_low,
                        &capture.h_mid,
                        &capture.h_high,
                        &h_shared,
                    )?
                } else {
                    head.forward_full_no_lm_head(
                        last_id,
                        &capture.h_low,
                        &capture.h_mid,
                        &capture.h_high,
                        &h_shared,
                    )?
                };
                #[cfg(not(target_os = "macos"))]
                let head_out = head.forward_full_no_lm_head(
                    last_id,
                    &capture.h_low,
                    &capture.h_mid,
                    &capture.h_high,
                    &h_shared,
                )?;
                let hidden_size = head_out.draft_hidden.len();
                let vocab_size = self.config.vocab_size;
                let draft_id = self
                    .gemv_f16_argmax_dispatch(vocab_size, hidden_size, &head_out.draft_hidden)?
                    .unwrap_or(v2_argmax);
                let calib_sigmoid = 1.0 / (1.0 + (-head_out.calib_logit).exp());

                if draft_id == v2_argmax {
                    stats.draft_accepted += 1;
                } else {
                    stats.draft_rejected += 1;
                }

                if spec_log {
                    eprintln!(
                        "[spec/eagle4] step={} pos={} draft={} v2={} calib={:.3} thresh={:.3} {}",
                        step,
                        pos,
                        draft_id,
                        v2_argmax,
                        calib_sigmoid,
                        calib_threshold,
                        if draft_id == v2_argmax { "ACCEPT" } else { "REJECT" }
                    );
                }

                if stall_active && step_start.elapsed() > stall_limit {
                    reason = StopReason::Aborted;
                    break;
                }
                self.sampler.record(v2_argmax);
                let text = self.tokenizer.decode_one(v2_argmax).unwrap_or_default();
                sink(StreamEvent::Token { id: v2_argmax, text });
                produced += 1;
                if Some(v2_argmax) == eos {
                    reason = StopReason::Eos;
                    break;
                }
                last_id = v2_argmax;
            }

            self.eagle4_head = Some(head);
        } else {
            for step in 0..req.max_new_tokens {
                if abort_set(&req) {
                    reason = StopReason::Aborted;
                    break;
                }
                let pos = prompt_len + step;
                let step_start = Instant::now();
                let next_id = if self.profiled_greedy_enabled(&req.sampling) {
                    match self.forward_token_greedy(last_id, pos)? {
                        Some(token) => token,
                        None => {
                            let mut logits = self.forward_token(last_id, pos)?;
                            self.sampler.sample(&mut logits, &req.sampling)
                        }
                    }
                } else {
                    let mut logits = self.forward_token(last_id, pos)?;
                    self.sampler.sample(&mut logits, &req.sampling)
                };
                if stall_active && step_start.elapsed() > stall_limit {
                    reason = StopReason::Aborted;
                    break;
                }
                self.sampler.record(next_id);
                let text = self.tokenizer.decode_one(next_id).unwrap_or_default();
                sink(StreamEvent::Token { id: next_id, text });
                produced += 1;
                if Some(next_id) == eos {
                    reason = StopReason::Eos;
                    break;
                }
                last_id = next_id;
            }
        }
        stats.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
        stats.completion_tokens = produced;
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
        "deepseek2"
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

    fn forward_tokens_batched(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        DeepSeekV2::forward_tokens_batched(self, tokens, positions)
    }

    fn forward_tokens_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens(tokens, positions)
    }

    fn forward_token_shared_only_for_test(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>> {
        self.forward_token_shared_only(token, pos)
    }

    /// Path-to-90 C2 — return `(final_norm_hidden, greedy_next_token)`.
    /// Equivalent to `forward_token` but exposes the pre-lm_head hidden
    /// state used by EAGLE-3 / MTP-style draft heads. KV cache advances.
    fn forward_token_with_hidden_for_test(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<(Vec<f32>, u32)> {
        let x_norm = self.forward_token_final_norm(token, pos)?;
        let h = self.config.hidden;
        let mut logits = vec![0.0f32; self.config.vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        self.gemv_f16_dispatch(w_f16, self.config.vocab_size, h, &x_norm, &mut logits)?;
        let argmax = crate::kernels::argmax_f32(&logits);
        Ok((x_norm, argmax))
    }

    /// Path-to-90 C2 — hidden-only fast path. Skips lm_head + argmax.
    /// KV cache advances identically. Use when `next_token` comes from
    /// the source corpus (teacher forcing) rather than from model greedy.
    fn forward_token_hidden_only_for_test(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<Vec<f32>> {
        self.forward_token_final_norm(token, pos)
    }

    /// Path-to-90 step 3 — capture EAGLE-4's 5-input bundle from a
    /// single decode step. Delegates to the inherent
    /// [`Self::forward_token_eagle4_capture`] (a CPU-walk that mirrors
    /// the existing [`Self::forward_token_shared_only`] choreography).
    /// Advances the CPU-side KV cache the same way `forward_token_shared_only`
    /// does — callers stringing multiple `_eagle4_for_test` calls in
    /// sequence get a consistent autoregressive walk.
    fn forward_token_eagle4_for_test(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<crate::engine::Eagle4Inputs> {
        self.forward_token_eagle4_capture(token, pos)
    }

    fn forward_tokens_batched_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        DeepSeekV2::forward_tokens_batched(self, tokens, positions)
    }

    fn reset_kv_for_test(&mut self) {
        self.reset_kv_state();
    }

    fn expert_access_counts(&self) -> Option<Vec<Vec<u64>>> {
        let cache = self.expert_cache.as_ref()?;
        let n_layers = self.config.n_layers;
        let n_experts = self.config.n_routed_experts;
        let first_dense = self.config.first_k_dense_layers;
        let mut result: Vec<Vec<u64>> = (0..n_layers)
            .map(|li| {
                if li < first_dense {
                    // Dense layers have no routed experts.
                    vec![]
                } else {
                    let layer_stats = match cache.stats.get(li) {
                        Some(s) => s,
                        None => return vec![0u64; n_experts],
                    };
                    layer_stats
                        .experts
                        .iter()
                        .map(|e| e.active_count())
                        .collect()
                }
            })
            .collect();
        // Ensure all MoE layers have exactly n_experts entries.
        for li in first_dense..n_layers {
            if result[li].len() != n_experts {
                result[li].resize(n_experts, 0);
            }
        }
        Some(result)
    }
}

impl DeepSeekV2 {
    /// v2.3.0 A3: encode the per-layer `add_inplace(x, addend) + rmsnorm_f32(x → out)`
    /// pair, fused into a single `add_rmsnorm_f32` dispatch when
    /// `residual_fusion = "f32"`, else as the original two-dispatch sequence.
    /// Semantically equivalent in either path.
    #[cfg(target_os = "macos")]
    #[inline]
    fn encode_add_and_rmsnorm_tcb(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        x_buf: &PinnedBuffer,
        addend_buf: &PinnedBuffer,
        weight_buf: &PinnedBuffer,
        eps: f32,
        hidden: usize,
        out_buf: &PinnedBuffer,
    ) -> Result<()> {
        if self.residual_fusion_f32 {
            crate::kernels::add_rmsnorm_metal_buf_tcb(
                tcb, x_buf, addend_buf, weight_buf, eps, hidden, out_buf,
            )
        } else {
            crate::kernels::add_inplace_metal_tcb(tcb, x_buf, addend_buf, hidden)?;
            crate::kernels::rmsnorm_metal_buf_tcb(
                tcb, x_buf, weight_buf, eps, hidden, out_buf,
            )
        }
    }

    /// v2.3.0 A4: route attention phase-3 to either the function-constant-
    /// specialized `mla_decode_kernel_fc` or the runtime-args
    /// `mla_decode_kernel`, based on the engine flag set from the
    /// `mla_schedule` profile field. Centralizes the routing so the 3
    /// per-token call sites stay legible.
    #[cfg(target_os = "macos")]
    #[inline]
    fn dispatch_mla_decode_and_o_proj(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        arena: &DecodeArena,
        kv_b_proj_buf: &PinnedBuffer,
        o_proj_buf: &PinnedBuffer,
        li: usize,
        seq_len: usize,
        scale: f32,
        h: usize,
    ) -> Result<()> {
        if self.mla_use_flash {
            crate::kernels::flash_attn_decode_and_o_proj_arena_tcb(
                tcb, arena, kv_b_proj_buf, o_proj_buf,
                &self.mla_c_kv_gpu[li],
                &self.mla_k_pe_gpu[li],
                self.config.n_heads,
                self.config.qk_nope_head_dim,
                self.config.qk_rope_head_dim,
                self.config.v_head_dim,
                self.config.kv_lora_rank,
                seq_len, scale, h,
            )
        } else if self.mla_use_fc {
            crate::kernels::mla_decode_and_o_proj_arena_fc_tcb(
                tcb, arena, kv_b_proj_buf, o_proj_buf,
                &self.mla_c_kv_gpu[li],
                &self.mla_k_pe_gpu[li],
                self.config.n_heads,
                self.config.v_head_dim,
                self.config.kv_lora_rank,
                seq_len, h,
            )
        } else {
            crate::kernels::mla_decode_and_o_proj_arena_tcb(
                tcb, arena, kv_b_proj_buf, o_proj_buf,
                &self.mla_c_kv_gpu[li],
                &self.mla_k_pe_gpu[li],
                self.config.n_heads,
                self.config.qk_nope_head_dim,
                self.config.qk_rope_head_dim,
                self.config.v_head_dim,
                self.config.kv_lora_rank,
                seq_len, scale, h,
            )
        }
    }

    fn layer_has_tcb_attention(layer: &Layer) -> bool {
        let q_lora_ready = layer.pinned.q_a_proj.is_some()
            && layer.pinned.q_b_proj.is_some()
            && layer.pinned.q_a_norm.is_some();
        let direct_q_ready = layer.pinned.q_proj.is_some();
        layer.pinned.attn_norm.is_some()
            && layer.pinned.ffn_norm.is_some()
            && layer.pinned.kv_a_proj_with_mqa.is_some()
            && layer.pinned.kv_b_proj.is_some()
            && layer.pinned.o_proj.is_some()
            && layer.pinned.kv_a_norm.is_some()
            && (q_lora_ready || direct_q_ready)
    }

    #[cfg(target_os = "macos")]
    fn greedy_gpu_argmax_available(&self) -> bool {
        self.metal_ctx.is_some()
            && self.decode_arena.is_some()
            && !self.mla_c_kv.is_empty()
            && self.weights_mmap_buf.is_some()
            && self.embed_buf.is_some()
            && self.final_norm_buf.is_some()
            && self.lm_head_buf.is_some()
            && self.logits_buf.is_some()
            && self.token_buf.is_some()
            && self.layers.iter().all(Self::layer_has_tcb_attention)
    }

    /// Phase 5C.2: returns true when the kernel profile selects f16 x_norm output.
    /// When true, the final rmsnorm writes to arena.x_norm_f16_buf (f16) and the
    /// LM head GEMV reads f16 activations instead of f32, halving that bandwidth.
    /// Residual stream stays f32 between layers; no accumulation error.
    #[cfg(target_os = "macos")]
    fn use_x_norm_f16(&self) -> bool {
        self.kernel_profile
            .as_ref()
            .map(|p| p.selected.x_norm_dtype == "f16")
            .unwrap_or(false)
    }

    #[cfg(target_os = "macos")]
    fn shared_only_gpu_argmax_available(&self) -> bool {
        self.greedy_gpu_argmax_available()
            && !self.mla_c_kv_gpu.is_empty()
            && !self.mla_k_pe_gpu.is_empty()
            && self.layers.iter().all(|layer| match &layer.mode {
                LayerMode::Dense { .. } => {
                    layer.pinned.dense_gate_w.is_some()
                        && layer.pinned.dense_up_w.is_some()
                        && layer.pinned.dense_down_w.is_some()
                }
                LayerMode::MoE { shared_fused, .. } => shared_fused
                    .as_ref()
                    .map(|s| {
                        s.gate_w.dtype == GgmlType::Q4_K
                            && s.up_w.dtype == GgmlType::Q4_K
                            && matches!(s.down_w.dtype, GgmlType::Q6_K | GgmlType::Q4_K)
                    })
                    .unwrap_or(false),
            })
    }

    #[cfg(target_os = "macos")]
    fn encode_shared_only_ffn_tcb(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        li: usize,
        arena: &DecodeArena,
        model_buf: &PinnedBuffer,
        q4k_schedule: &str,
        shared_down_schedule: &str,
    ) -> Result<bool> {
        if self.encode_dense_ffn_tcb(tcb, li, arena)? {
            return Ok(true);
        }

        let LayerMode::MoE { shared_fused, .. } = &self.layers[li].mode else {
            return Ok(false);
        };
        let Some(shared) = shared_fused else {
            return Ok(false);
        };
        if shared.gate_w.dtype != GgmlType::Q4_K
            || shared.up_w.dtype != GgmlType::Q4_K
            || !matches!(shared.down_w.dtype, GgmlType::Q6_K | GgmlType::Q4_K)
        {
            return Ok(false);
        }

        let shared_mid = self.config.n_shared_experts * self.config.moe_intermediate;
        let shared_down_kernel = match shared.down_w.dtype {
            GgmlType::Q6_K => match shared_down_schedule {
                "v2t" => "moe_batched_gemm_q6_k_indexed_v2t",
                _ => "moe_batched_gemm_q6_k_indexed",
            },
            GgmlType::Q4_K => FfnMoeSetup::q4k_indexed_kernel(q4k_schedule),
            _ => unreachable!("dtype checked above"),
        };

        crate::kernels::encode_moe_shared_only_indexed_tcb_with_scratch(
            tcb,
            model_buf,
            &arena.shared_route_ids_buf,
            shared.gate_w.offset,
            shared.up_w.offset,
            shared.down_w.offset,
            self.config.hidden,
            shared_mid,
            q4k_schedule,
            shared_down_kernel,
            &arena.x_norm_buf,
            &arena.ffn_out_buf,
            &arena.moe_shared_gate_out_buf,
            &arena.moe_shared_up_out_buf,
            &arena.moe_shared_act_buf,
        )?;
        Ok(true)
    }

    #[cfg(target_os = "macos")]
    fn forward_token_shared_only_gpu_argmax(&mut self, token: u32, pos: usize) -> Result<u32> {
        let h = self.config.hidden;
        let eps = self.config.rms_norm_eps;
        let n_layers = self.config.n_layers;
        let kv_lora_rank = self.config.kv_lora_rank;
        let qk_rope_head_dim = self.config.qk_rope_head_dim;

        if self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model("kv cache full".into()));
        }

        if !self.mla_kv_gpu_synced && self.kv.seq_len > 0 {
            for li in 0..n_layers {
                MetalContext::write_buffer_bytes(
                    &self.mla_c_kv_gpu[li],
                    bytemuck::cast_slice(&self.mla_c_kv[li][..self.kv.seq_len * kv_lora_rank]),
                );
                MetalContext::write_buffer_bytes(
                    &self.mla_k_pe_gpu[li],
                    bytemuck::cast_slice(&self.mla_k_pe[li][..self.kv.seq_len * qk_rope_head_dim]),
                );
            }
            self.mla_kv_gpu_synced = true;
        }

        let seq_slot = self.kv.seq_len;
        let seq_len = seq_slot + 1;
        let ctx = self.metal_ctx.as_ref().unwrap();
        let arena = self.decode_arena.as_ref().unwrap();
        let model_buf = self.weights_mmap_buf.as_ref().unwrap();
        let final_norm_buf = self.final_norm_buf.as_ref().unwrap();
        let lm_head_buf = self.lm_head_buf.as_ref().unwrap();
        let logits_buf = self.logits_buf.as_ref().unwrap();
        let tok_buf = self.token_buf.as_ref().unwrap();
        let q4k_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.gemm_q4_k_schedule.as_str())
            .unwrap_or("scalar");
        let shared_down_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.shared_down_schedule.as_str())
            .unwrap_or("basic");
        let use_simdmat = self
            .kernel_profile
            .as_ref()
            .map(|p| p.selected.lm_head_schedule.contains("simdgroup-matrix"))
            .unwrap_or(false);
        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);

        for li in 0..n_layers {
            crate::metal::set_current_layer(Some(li as u32));
            let kv_b_proj_buf = self.layers[li].pinned.kv_b_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("shared-only: l{li} kv_b_proj not pinned")))?;
            let o_proj_buf = self.layers[li].pinned.o_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("shared-only: l{li} o_proj not pinned")))?;
            let ffn_norm_buf = self.layers[li].pinned.ffn_norm.as_ref().unwrap();
            let head_dim_q = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;
            let scale = 1.0f32 / (head_dim_q as f32).sqrt();

            self.encode_attention_phase1_into_tcb(&mut tcb, li, pos, Some(token), seq_slot)?;
            self.encode_attention_phase2_tcb(&mut tcb, li, pos)?;
            self.dispatch_mla_decode_and_o_proj(
                &mut tcb, arena, kv_b_proj_buf, o_proj_buf, li, seq_len, scale, h,
            )?;
            self.encode_add_and_rmsnorm_tcb(
                &mut tcb, &arena.x_buf, &arena.out, ffn_norm_buf, eps, h, &arena.x_norm_buf,
            )?;
            if !self.encode_shared_only_ffn_tcb(
                &mut tcb, li, arena, model_buf, q4k_schedule, shared_down_schedule,
            )? {
                return Err(Error::Model(format!(
                    "shared-only GPU path could not encode layer {li}"
                )));
            }
        }

        crate::metal::set_current_layer(None);
        if n_layers > 0 {
            crate::kernels::add_inplace_metal_tcb(&mut tcb, &arena.x_buf, &arena.ffn_out_buf, h)?;
        }
        // Phase 5C.2: opt-in f16 final norm + LM head.
        let x_norm_f16 = self.use_x_norm_f16();
        if x_norm_f16 {
            crate::kernels::rmsnorm_f32_to_f16_tcb(
                &mut tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_f16_buf,
            )?;
            crate::kernels::gemv_f16_f16in_tcb(
                &mut tcb, lm_head_buf, self.config.vocab_size, h, &arena.x_norm_f16_buf, logits_buf,
            )?;
        } else if use_simdmat {
            crate::kernels::rmsnorm_metal_buf_tcb(
                &mut tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_buf,
            )?;
            crate::kernels::gemv_f16_simdmat_tcb(
                &mut tcb, lm_head_buf, self.config.vocab_size, h, &arena.x_norm_buf, logits_buf,
            )?;
        } else {
            crate::kernels::rmsnorm_metal_buf_tcb(
                &mut tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_buf,
            )?;
            crate::kernels::gemv_f16_metal_buf_tcb(
                &mut tcb, lm_head_buf, self.config.vocab_size, h, &arena.x_norm_buf, logits_buf,
            )?;
        }
        crate::kernels::sample_argmax_f32_tcb(
            &mut tcb, logits_buf, tok_buf, self.config.vocab_size,
        )?;
        tcb.commit_and_wait()?;

        self.kv.seq_len += 1;
        self.mla_kv_gpu_synced = true;

        let tok_ptr = tok_buf.contents() as *const u32;
        Ok(unsafe { *tok_ptr })
    }

    /// rmsnorm dispatcher: Metal when the context is present, CPU
    /// otherwise. Mirrors `kernels::rmsnorm`'s signature so call
    /// sites read the same.
    fn rmsnorm_dispatch(&self, x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            return crate::kernels::rmsnorm_metal(ctx, x, weight, eps, out);
        }
        rmsnorm(x, weight, eps, out);
        Ok(())
    }

    /// LM-head / embedding-tied GEMV dispatcher. `w_f16` is the
    /// `(rows, cols)` row-major fp16 weight matrix. WB: prefers the
    /// pinned variant when `lm_head_buf` is populated (Metal alive,
    /// load-time upload happened); falls back to the byte-slice
    /// `gemv_f16_metal` path otherwise; CPU off-macOS.
    fn gemv_f16_dispatch(
        &self,
        w_f16: &[f16],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            if let Some(buf) = &self.lm_head_buf {
                return crate::kernels::gemv_f16_metal_pinned(ctx, buf, rows, cols, x, out);
            }
            let w_bytes = bytemuck::cast_slice::<f16, u8>(w_f16);
            return crate::kernels::gemv_f16_metal(ctx, w_bytes, rows, cols, x, out);
        }
        crate::kernels::gemv_f16(w_f16, rows, cols, x, out);
        Ok(())
    }

    /// Attention fp32 GEMV dispatcher (used for o_proj + 4 MLA gemvs).
    /// WB: prefers the pinned path when `pinned` is `Some` (caller has
    /// the corresponding `LayerPinned` field populated); falls back to
    /// the byte-slice `gemv_f32_attn_metal` when only Metal is alive
    /// without pinning; CPU off-macOS.
    fn gemv_f32_attn_dispatch(
        &self,
        w: &[f32],
        pinned: Option<&PinnedBuffer>,
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            // Guard: only use pinned when it is f32-sized; f16-uploaded buffers are
            // half the expected size and must not be fed into the f32 dispatch path.
            let f32_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;
            if let Some(buf) = pinned.filter(|b| b.length() >= f32_bytes) {
                return crate::kernels::gemv_f32_attn_metal_pinned(ctx, buf, rows, cols, x, out);
            }
            return crate::kernels::gemv_f32_attn_metal(ctx, w, rows, cols, x, out);
        }
        let _ = pinned;
        gemv_f32(w, rows, cols, x, out);
        Ok(())
    }

    /// v0.3.4 — shared-input pair dispatcher: coalesces two independent fp32 GEMVs
    /// (e.g. q_a_proj + kv_a_proj_with_mqa) that read the same `x` into one CB.
    fn gemv_f32_attn_pair_dispatch(
        &self,
        w_a: &[f32], pinned_a: Option<&PinnedBuffer>, rows_a: usize,
        w_b: &[f32], pinned_b: Option<&PinnedBuffer>, rows_b: usize,
        cols: usize,
        x: &[f32],
        out_a: &mut [f32],
        out_b: &mut [f32],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            // Guard: skip pinned buffers that are f16-sized (see gemv_f32_attn_dispatch).
            let f32_a = (rows_a * cols * std::mem::size_of::<f32>()) as u64;
            let f32_b = (rows_b * cols * std::mem::size_of::<f32>()) as u64;
            let buf_a = pinned_a.filter(|b| b.length() >= f32_a);
            let buf_b = pinned_b.filter(|b| b.length() >= f32_b);
            if let (Some(buf_a), Some(buf_b)) = (buf_a, buf_b) {
                return crate::kernels::dispatch_gemv_f32_attn_pinned_pair_batched(
                    ctx, buf_a, rows_a, buf_b, rows_b, cols, x, out_a, out_b,
                );
            }
        }
        let _ = (pinned_a, pinned_b);
        self.gemv_f32_attn_dispatch(w_a, None, rows_a, cols, x, out_a)?;
        self.gemv_f32_attn_dispatch(w_b, None, rows_b, cols, x, out_b)
    }

    /// MoE gate-logits fp32 GEMV dispatcher (`ffn_gate_inp`). Tiny but
    /// frequent — once per token per MoE layer.
    fn gemv_f32_moe_dispatch(
        &self,
        w: &[f32],
        rows: usize,
        cols: usize,
        x: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            return crate::kernels::gemv_f32_moe_metal(ctx, w, rows, cols, x, out);
        }
        gemv_f32(w, rows, cols, x, out);
        Ok(())
    }

    /// v0.3.3 — pair+silu dispatcher: coalesces gate+up GEMVs and silu_mul into ONE
    /// CommandBatch when the simd schedule is active, writing the silu'd result
    /// directly into `a`. Fallback computes gate+up separately then CPU silu_mul.
    fn moe_expert_pair_matmul_dispatch(
        &self,
        t_gate: &TensorRef,
        t_up: &TensorRef,
        rows: usize,
        cols: usize,
        x: &[f32],
        a: &mut [f32],
        scratch: &mut Vec<f32>,
    ) -> Result<()> {
        #[cfg(target_os = "macos")]
        if let Some(ctx) = &self.metal_ctx {
            if t_gate.dtype == GgmlType::Q4_K && t_up.dtype == GgmlType::Q4_K {
                let use_simd = self
                    .kernel_profile
                    .as_ref()
                    .map(|p| p.selected.gemm_q4_k_schedule == "simdgroup")
                    .unwrap_or(false);
                if use_simd {
                    let bytes_gate = &self.gguf.mmap[t_gate.offset..t_gate.offset + t_gate.byte_size];
                    let bytes_up = &self.gguf.mmap[t_up.offset..t_up.offset + t_up.byte_size];
                    return crate::kernels::dispatch_gemv_q4_k_m_simd_pair_silu_batched(
                        ctx, bytes_gate, bytes_up, rows, cols, x, a,
                    );
                }
            }
        }
        let mut g_tmp = vec![0.0f32; rows];
        let mut u_tmp = vec![0.0f32; rows];
        self.moe_expert_matmul_dispatch(t_gate, rows, cols, x, &mut g_tmp, scratch)?;
        self.moe_expert_matmul_dispatch(t_up,   rows, cols, x, &mut u_tmp, scratch)?;
        crate::kernels::silu_mul(&g_tmp, &u_tmp, a);
        Ok(())
    }

    /// Routed/shared MoE expert matmul dispatcher. Reads the GGUF
    /// quantized bytes directly when the Metal Q4_K_M-fused kernel is
    /// available; otherwise dequants into `scratch` and runs CPU GEMV.
    fn moe_expert_matmul_dispatch(
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
            if t.dtype == GgmlType::Q4_K {
                let bytes = &self.gguf.mmap[t.offset..t.offset + t.byte_size];
                let schedule = self.q4k_schedule_for_shape(rows, cols);
                if schedule == "v2" {
                    if let Some(model_buf) = &self.weights_mmap_buf {
                        return crate::kernels::gemv_q4_k_m_v2_pinned(
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
                    return crate::kernels::gemv_q4_k_m_v2(ctx, bytes, rows, cols, x, out);
                }
                if schedule == "simdmat" {
                    if let Some(model_buf) = &self.weights_mmap_buf {
                        return crate::kernels::gemv_q4_k_m_simdmat_pinned(
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
                if schedule == "v3_8r" {
                    if let Some(model_buf) = &self.weights_mmap_buf {
                        return crate::kernels::gemv_q4_k_m_v3_8r_pinned(
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
                if schedule == "v3_dual" {
                    if let Some(model_buf) = &self.weights_mmap_buf {
                        return crate::kernels::gemv_q4_k_m_v3_dual_pinned(
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
                if schedule == "v3_llama" || schedule == "llama_port" {
                    if let Some(model_buf) = &self.weights_mmap_buf {
                        return crate::kernels::gemv_q4_k_m_llama_port_pinned(
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
                if schedule == "simdgroup" {
                    return crate::kernels::dispatch_gemv_q4_k_m_simd_batched(
                        ctx, bytes, rows, cols, x, out,
                    );
                }
                // v2.1 T2.1: when the top-level schedule is one of the v2t_gu_*
                // family (selected for the MoE TCB path), the standalone Q4_K
                // dispatch (used for non-TCB attention projections / fallback)
                // doesn't have a matching kernel name. Map all v2t_* schedules
                // to the v2 standalone kernel so we never silently regress to
                // the slow scalar `gemv_q4_k_m` here.
                if matches!(
                    schedule,
                    "v2t" | "v2t_gu" | "v2t_gu_v2" | "v2t_gu_serial"
                ) {
                    if let Some(model_buf) = &self.weights_mmap_buf {
                        return crate::kernels::gemv_q4_k_m_v2_pinned(
                            ctx, model_buf, t.offset, t.byte_size, rows, cols, x, out,
                        );
                    }
                    return crate::kernels::gemv_q4_k_m_v2(ctx, bytes, rows, cols, x, out);
                }
                return crate::kernels::gemv_q4_k_m(ctx, bytes, rows, cols, x, out);
            }
        }
        self.dequant_ref_into(t, scratch)?;
        gemv_f32(scratch, rows, cols, x, out);
        Ok(())
    }

    fn moe_block_batched_dispatch(
        &self,
        routed_fused: &MoEFusedTensors,
        routed: &[Expert],
        shared_fused: Option<&MoEFusedTensors>,
        routes: &[(usize, f32)],
        x: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        #[cfg(target_os = "macos")]
        {
            let Some(ctx) = &self.metal_ctx else {
                return Ok(None);
            };
            let Some(model_buf) = &self.weights_mmap_buf else {
                return Ok(None);
            };
            if routed_fused.gate_w.dtype != GgmlType::Q4_K
                || routed_fused.up_w.dtype != GgmlType::Q4_K
                || routed_fused.down_w.dtype != GgmlType::Q8_0
            {
                return Ok(None);
            }
            if let Some(shared) = shared_fused {
                if shared.gate_w.dtype != GgmlType::Q4_K
                    || shared.up_w.dtype != GgmlType::Q4_K
                    || shared.down_w.dtype != GgmlType::Q6_K
                {
                    return Ok(None);
                }
            }

            let mut route_ids = Vec::with_capacity(routes.len());
            let mut route_weights = Vec::with_capacity(routes.len());
            for &(eid, weight) in routes {
                if eid >= routed.len() {
                    return Err(Error::Model(format!(
                        "route selected expert {eid}, but only {} experts are loaded",
                        routed.len()
                    )));
                }
                if eid > u32::MAX as usize {
                    return Err(Error::Model(format!(
                        "route selected expert {eid}, but Metal route ids are u32"
                    )));
                }
                route_ids.push(eid as u32);
                route_weights.push(weight);
            }

            let mut out = vec![0.0f32; self.config.hidden];
            let (shared_gate_offset, shared_up_offset, shared_down_offset, shared_mid) =
                if let Some(shared) = shared_fused {
                    (
                        Some(shared.gate_w.offset),
                        Some(shared.up_w.offset),
                        Some(shared.down_w.offset),
                        self.config.n_shared_experts * self.config.moe_intermediate,
                    )
                } else {
                    (None, None, None, 0)
                };

            let q4k_schedule = self
                .kernel_profile
                .as_ref()
                .map(|p| p.selected.gemm_q4_k_schedule.as_str())
                .unwrap_or("scalar");
            crate::kernels::moe_block_batched_indexed_metal(
                ctx,
                model_buf,
                routed_fused.gate_w.offset,
                routed_fused.up_w.offset,
                routed_fused.down_w.offset,
                routed.len(),
                &route_ids,
                &route_weights,
                shared_gate_offset,
                shared_up_offset,
                shared_down_offset,
                self.config.hidden,
                self.config.moe_intermediate,
                shared_mid,
                q4k_schedule,
                x,
                &mut out,
            )?;
            Ok(Some(out))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (routed_fused, routed, shared_fused, routes, x);
            Ok(None)
        }
    }

    /// One-token forward pass: takes a token id at absolute position
    /// `pos`, advances the KV cache, returns the output logits vector.
    fn forward_token(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let x_norm = self.forward_token_final_norm(token, pos)?;
        let h = self.config.hidden;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed, // tied to embedding
        };
        self.gemv_f16_dispatch(w_f16, self.config.vocab_size, h, &x_norm, &mut logits)?;
        Ok(logits)
    }

    /// Multi-token forward pass. Phase 2 Wedge 2a: initial impl is a loop
    /// over `forward_token` — semantically identical to N sequential single-
    /// token calls. Subsequent wedges (2c-2f) widen the internals.
    fn forward_tokens(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != positions.len() {
            return Err(Error::Model(format!(
                "forward_tokens shape: tokens={} positions={}",
                tokens.len(), positions.len()
            )));
        }
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            out.push(self.forward_token(token, positions[i])?);
        }
        Ok(out)
    }

    /// Phase 4C/5A — multi-token forward pass for n-gram spec decode verify.
    ///
    /// Accepts K tokens at sequential positions and returns K logit vectors.
    ///
    /// **Phase 5A fast path:** When Wedge C conditions are met (Metal + arena +
    /// all weights pinned + GPU KV active), runs all K tokens in a SINGLE
    /// command buffer commit, eliminating K-1 commit+wait round-trips.
    /// Each token processes sequentially within the CB; a GPU blit saves each
    /// token's final x_norm into `arena.batch_x_norm_buf[ki]` before the next
    /// token overwrites the shared arena buffers.
    ///
    /// **Phase 4C fallback:** sequential loop over `forward_token` when fast
    /// path conditions are not met.
    fn forward_tokens_batched(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        if tokens.len() != positions.len() {
            return Err(Error::Model(format!(
                "forward_tokens_batched: tokens={} positions={}",
                tokens.len(),
                positions.len()
            )));
        }
        if tokens.is_empty() {
            return Ok(vec![]);
        }

        // Phase 5A: single-TCB fast path — same conditions as Wedge C.
        #[cfg(target_os = "macos")]
        {
            let tcb_base = self.metal_ctx.is_some()
                && self.decode_arena.is_some()
                && self.layers.iter().all(|l| {
                    l.pinned.attn_norm.is_some() && l.pinned.ffn_norm.is_some()
                });
            let wedge_c_ready = tcb_base
                && !self.mla_c_kv.is_empty()
                && self.weights_mmap_buf.is_some()
                && self.embed_buf.is_some()
                && self.final_norm_buf.is_some()
                && self.layers.iter().all(Self::layer_has_tcb_attention);
            let k = tokens.len();
            let max_bs = self.decode_arena.as_ref().map(|a| a.max_batch_size).unwrap_or(0);
            if wedge_c_ready && !self.mla_c_kv_gpu.is_empty() && k <= max_bs {
                return self.forward_tokens_batched_tcb(tokens, positions);
            }
        }

        // Phase 4C fallback: sequential per-token loop.
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            out.push(self.forward_token(token, positions[i])?);
        }
        Ok(out)
    }

    /// Phase 5A — single-TCB K-token batched forward.
    ///
    /// Processes K tokens sequentially within one Metal command buffer.
    /// Each token reuses the shared arena scratch buffers (x_buf, x_norm_buf,
    /// q, attn_out, etc.) which Metal executes in strict encoder order.
    /// After each token's final rmsnorm, x_norm_buf is blitted into
    /// `arena.batch_x_norm_buf[ki]` to survive until the CB commits.
    ///
    /// Key invariant: Metal's sequential encoder guarantee means token k's
    /// kv_append writes to GPU KV BEFORE token k+1's mla_decode reads it.
    /// KV rollback (kv.seq_len reset) is handled by the caller.
    #[cfg(target_os = "macos")]
    fn forward_tokens_batched_tcb(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        let k = tokens.len();
        let h = self.config.hidden;
        let n_layers = self.config.n_layers;
        let eps = self.config.rms_norm_eps;
        let kv_lora_rank = self.config.kv_lora_rank;
        let qk_rope_head_dim = self.config.qk_rope_head_dim;

        // One-time KV sync: mirror CPU KV into GPU-resident buffers (matches Wedge C).
        if !self.mla_kv_gpu_synced && self.kv.seq_len > 0 {
            let sl = self.kv.seq_len;
            for li in 0..n_layers {
                crate::metal::MetalContext::write_buffer_bytes(
                    &self.mla_c_kv_gpu[li],
                    bytemuck::cast_slice(&self.mla_c_kv[li][..sl * kv_lora_rank]),
                );
                crate::metal::MetalContext::write_buffer_bytes(
                    &self.mla_k_pe_gpu[li],
                    bytemuck::cast_slice(&self.mla_k_pe[li][..sl * qk_rope_head_dim]),
                );
            }
        }

        let seq_slot_base = self.kv.seq_len;
        let q4k_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.gemm_q4_k_schedule.as_str())
            .unwrap_or("scalar");
        // v2.1.0-T2.11: Q5_0 routed-down kernel selector. "basic" routes to
        // the historical kernel; "v2t" opts into the new 8-rows-per-TG
        // simdsum kernel. Default is "basic" — flip via profile JSON.
        let routed_down_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.routed_down_schedule.as_str())
            .unwrap_or("basic");
        // v2.1.0-T2.12: Q6_K shared-down kernel selector (parallel pattern).
        let shared_down_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.shared_down_schedule.as_str())
            .unwrap_or("basic");

        {
            let ctx = self.metal_ctx.as_ref().unwrap();
            let arena = self.decode_arena.as_ref().unwrap();
            let model_buf = self.weights_mmap_buf.as_ref().unwrap();
            let final_norm_buf = self.final_norm_buf.as_ref().unwrap();

            let mut global_tcb = crate::metal::TokenCommandBuffer::new(ctx);

            for (ki, &token) in tokens.iter().enumerate() {
                let seq_slot = seq_slot_base + ki;
                let seq_len = seq_slot + 1;
                let pos = positions[ki];

                for li in 0..n_layers {
                    crate::metal::set_current_layer(Some(li as u32));

                    let moe_setup = self.ffn_moe_check(li)?;
                    let kv_b_proj_buf = self.layers[li].pinned.kv_b_proj.as_ref()
                        .ok_or_else(|| crate::Error::Model(format!("batched5A: l{li} kv_b_proj not pinned")))?;
                    let o_proj_buf = self.layers[li].pinned.o_proj.as_ref()
                        .ok_or_else(|| crate::Error::Model(format!("batched5A: l{li} o_proj not pinned")))?;
                    let ffn_norm_buf = self.layers[li].pinned.ffn_norm.as_ref().unwrap();
                    let head_dim_q = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;
                    let scale = 1.0f32 / (head_dim_q as f32).sqrt();

                    // Phase 1: embed lookup (li=0) or add_inplace residual (li>0),
                    //           then q/kv projections + kv_append into GPU KV at seq_slot.
                    self.encode_attention_phase1_into_tcb(
                        &mut global_tcb, li, pos, Some(token), seq_slot,
                    )?;
                    // Phase 2: q_b_proj + rope_q.
                    self.encode_attention_phase2_tcb(&mut global_tcb, li, pos)?;
                    // Phase 3: mla_decode (reads seq_len GPU KV entries) + o_proj.
                    self.dispatch_mla_decode_and_o_proj(
                        &mut global_tcb, arena, kv_b_proj_buf, o_proj_buf,
                        li, seq_len, scale, h,
                    )?;
                    // Residual: x_buf += attn_out (arena.out).
                    crate::kernels::add_inplace_metal_tcb(
                        &mut global_tcb, &arena.x_buf, &arena.out, h,
                    )?;
                    // FFN norm: x_buf → x_norm_buf.
                    crate::kernels::rmsnorm_metal_buf_tcb(
                        &mut global_tcb, &arena.x_buf, ffn_norm_buf, eps, h, &arena.x_norm_buf,
                    )?;

                    // FFN: MoE or dense.
                    if let Some(ref setup) = moe_setup {
                        let gate_buf = self.layers[li].pinned.gate_logits_w.as_ref().unwrap();
                        crate::kernels::gemv_f32_moe_pinned_buf_tcb(
                            &mut global_tcb, gate_buf,
                            self.config.n_routed_experts, self.config.hidden,
                            &arena.x_norm_buf, &arena.moe_logits_buf,
                        )?;
                        crate::kernels::moe_topk_gate_tcb(
                            &mut global_tcb,
                            &arena.moe_logits_buf,
                            &arena.moe_route_ids_buf,
                            &arena.moe_route_weights_buf,
                            self.config.n_routed_experts,
                            self.config.top_k_routed,
                        )?;
                        crate::kernels::encode_moe_block_batched_indexed_tcb_with_scratch(
                            &mut global_tcb,
                            model_buf,
                            setup.routed_gate_off,
                            setup.routed_up_off,
                            setup.routed_down_off,
                            &arena.moe_route_ids_buf,
                            &arena.moe_route_weights_buf,
                            self.config.top_k_routed,
                            &arena.shared_route_ids_buf,
                            setup.shared_gate_off,
                            setup.shared_up_off,
                            setup.shared_down_off,
                            self.config.hidden,
                            self.config.moe_intermediate,
                            setup.shared_mid,
                            q4k_schedule,
                            setup.routed_down_kernel_with_schedule(q4k_schedule, routed_down_schedule),
                            setup.shared_down_kernel_with_schedule(q4k_schedule, shared_down_schedule),
                            &arena.x_norm_buf,
                            &arena.ffn_out_buf,
                            &arena.moe_routed_gate_out_buf,
                            &arena.moe_routed_up_out_buf,
                            &arena.moe_routed_act_buf,
                            &arena.moe_routed_out_buf,
                            &arena.moe_shared_gate_out_buf,
                            &arena.moe_shared_up_out_buf,
                            &arena.moe_shared_act_buf,
                            &arena.moe_shared_out_buf,
                        )?;
                    } else {
                        let dense_ok = self.encode_dense_ffn_tcb(&mut global_tcb, li, arena)?;
                        if !dense_ok {
                            // Dense weights not pinned — not supported in batched fast path.
                            // Caller should not reach here (wedge_c_ready requires all weights).
                            return Err(Error::Model(format!(
                                "forward_tokens_batched_tcb: l{li} dense weights not pinned"
                            )));
                        }
                    }
                }

                // Final residual accumulation + final norm for token ki.
                crate::kernels::add_inplace_metal_tcb(
                    &mut global_tcb, &arena.x_buf, &arena.ffn_out_buf, h,
                )?;
                crate::kernels::rmsnorm_metal_buf_tcb(
                    &mut global_tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_buf,
                )?;
                // Blit x_norm_buf → batch_x_norm_buf[ki] so it survives the next token
                // overwriting x_norm_buf.
                let sz = (h * std::mem::size_of::<f32>()) as u64;
                global_tcb.copy_buffer_bytes(
                    &arena.x_norm_buf, 0, &arena.batch_x_norm_buf[ki], 0, sz,
                )?;
            }

            // Single commit covering all K tokens' GPU work.
            global_tcb.commit_and_wait()?;
        }

        self.kv.seq_len += k;
        self.mla_kv_gpu_synced = true;
        crate::metal::set_current_layer(None);

        // Post-commit: run LM head for each token's saved x_norm.
        let w_f16: &[half::f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        let mut results = Vec::with_capacity(k);
        for ki in 0..k {
            let arena = self.decode_arena.as_ref().unwrap();
            let mut x_norm = vec![0.0f32; h];
            {
                let ptr = arena.batch_x_norm_buf[ki].contents() as *const f32;
                let src = unsafe { std::slice::from_raw_parts(ptr, h) };
                x_norm.copy_from_slice(src);
            }
            let mut logits = vec![0.0f32; self.config.vocab_size];
            self.gemv_f16_dispatch(w_f16, self.config.vocab_size, h, &x_norm, &mut logits)?;
            results.push(logits);
        }

        Ok(results)
    }

    /// Reset MLA KV cache to empty state for test isolation.
    fn reset_kv_state(&mut self) {
        self.kv.reset();
        for v in &mut self.mla_c_kv {
            v.fill(0.0);
        }
        for v in &mut self.mla_k_pe {
            v.fill(0.0);
        }
        self.mla_kv_gpu_synced = false;
    }

    /// Append a single (c_kv, k_pe) entry to the MLA cache for layer `li`
    /// at sequence slot `seq_slot`. Pure refactor of the inlined writes;
    /// N=1 semantics unchanged. Phase 2 Wedge 2b — wedge 2d will add a
    /// _batch counterpart.
    ///
    /// Takes field references directly so callers holding `&self.config`
    /// can call this without triggering a whole-self reborrow.
    fn mla_kv_append(
        mla_c_kv: &mut Vec<Vec<f32>>,
        mla_k_pe: &mut Vec<Vec<f32>>,
        li: usize,
        seq_slot: usize,
        kv_lora_rank: usize,
        qk_rope_head_dim: usize,
        c_kv: &[f32],
        k_pe: &[f32],
    ) -> Result<()> {
        if c_kv.len() != kv_lora_rank {
            return Err(Error::Model(format!(
                "mla_kv_append c_kv len: got {} expected {}",
                c_kv.len(), kv_lora_rank
            )));
        }
        if k_pe.len() != qk_rope_head_dim {
            return Err(Error::Model(format!(
                "mla_kv_append k_pe len: got {} expected {}",
                k_pe.len(), qk_rope_head_dim
            )));
        }
        let pos_c = seq_slot * kv_lora_rank;
        mla_c_kv[li][pos_c..pos_c + kv_lora_rank].copy_from_slice(c_kv);
        let pos_k = seq_slot * qk_rope_head_dim;
        mla_k_pe[li][pos_k..pos_k + qk_rope_head_dim].copy_from_slice(k_pe);
        Ok(())
    }

    fn forward_token_greedy(&mut self, token: u32, pos: usize) -> Result<Option<u32>> {
        #[cfg(target_os = "macos")]
        let wedge_e_ok = self.greedy_gpu_argmax_available();
        #[cfg(not(target_os = "macos"))]
        let wedge_e_ok = false;

        let (x_norm, maybe_greedy) = self.forward_token_final_norm_maybe_read(token, pos, !wedge_e_ok)?;

        // Phase 5B.1: LM head + argmax were folded into the global TCB — token ready.
        // This is the fast path: one TCB commit covers all 27 layers + final-norm + LM head.
        #[cfg(target_os = "macos")]
        if let Some(tok) = maybe_greedy {
            return Ok(Some(tok));
        }

        // Fallback: LM head not yet folded (e.g. ExactShared without gpu_shared_draft).
        // arena.x_norm_buf was written by the Wedge M C-1 mini-TCB; run LM head + argmax
        // in a separate mini-TCB. This path is kept for correctness but is not the hot path.
        // v1.0.0-E: GPU argmax via TCB (zero counted dispatches) when the full Wedge C
        // stack ran. arena.x_norm_buf holds the final-normed residual on-GPU, so only 4
        // bytes cross the bus instead of 408 KB.
        #[cfg(target_os = "macos")]
        {
            if wedge_e_ok {
                let vocab = self.config.vocab_size;
                let cols = self.config.hidden;
                let result = {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let lm_head_buf = self.lm_head_buf.as_ref().unwrap();
                    let logits_buf = self.logits_buf.as_ref().unwrap();
                    let tok_buf = self.token_buf.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    // Phase 5C.2: if x_norm_f16 path ran in forward_token_final_norm_maybe_read,
                    // the final norm was already written to x_norm_f16_buf — but that path is
                    // lm_head_foldable and returns early, so we never reach here in the f16 path.
                    // This fallback handles the non-fold (ExactShared) case which stays f32.
                    let use_simdmat = self
                        .kernel_profile
                        .as_ref()
                        .map(|p| p.selected.lm_head_schedule.contains("simdgroup-matrix"))
                        .unwrap_or(false);
                    if use_simdmat {
                        crate::kernels::gemv_f16_simdmat_tcb(
                            &mut tcb, lm_head_buf, vocab, cols, &arena.x_norm_buf, logits_buf,
                        )?;
                    } else {
                        crate::kernels::gemv_f16_metal_buf_tcb(
                            &mut tcb, lm_head_buf, vocab, cols, &arena.x_norm_buf, logits_buf,
                        )?;
                    }
                    crate::kernels::sample_argmax_f32_tcb(&mut tcb, logits_buf, tok_buf, vocab)?;
                    tcb.commit_and_wait()?;
                    let tok_ptr = tok_buf.contents() as *const u32;
                    unsafe { *tok_ptr }
                };
                return Ok(Some(result));
            }
        }

        let x_norm = x_norm.ok_or_else(|| {
            Error::Model("forward_token_greedy: missing CPU final norm for fallback argmax".into())
        })?;
        self.gemv_f16_argmax_dispatch(self.config.vocab_size, self.config.hidden, &x_norm)
    }

    fn forward_token_argmax(
        &mut self,
        token: u32,
        pos: usize,
        use_profiled_greedy: bool,
    ) -> Result<u32> {
        if use_profiled_greedy {
            if let Some(next) = self.forward_token_greedy(token, pos)? {
                return Ok(next);
            }
        }
        let logits = self.forward_token(token, pos)?;
        Ok(crate::kernels::argmax_f32(&logits))
    }

    fn forward_token_shared_only_argmax(&mut self, token: u32, pos: usize) -> Result<u32> {
        #[cfg(target_os = "macos")]
        if self.shared_only_gpu_argmax_available() {
            return self.forward_token_shared_only_gpu_argmax(token, pos);
        }

        let logits = self.forward_token_shared_only(token, pos)?;
        Ok(crate::kernels::argmax_f32(&logits))
    }

    fn forward_token_final_norm(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let (x_norm, _) = self.forward_token_final_norm_maybe_read(token, pos, true)?;
        x_norm.ok_or_else(|| Error::Model("forward_token_final_norm: final norm not read back".into()))
    }

    /// Returns `(Option<x_norm>, Option<greedy_token>)`.
    ///
    /// - `read_back=true`  → `(Some(x_norm), None)`.
    /// - `read_back=false` (non-fold) → `(None, None)`, x_norm_buf on GPU.
    /// - `read_back=false` + Phase 5B.1 LM-head fold → `(None, Some(tok))`,
    ///   LM head + argmax were encoded into the global TCB; single commit.
    fn forward_token_final_norm_maybe_read(
        &mut self,
        token: u32,
        pos: usize,
        read_back: bool,
    ) -> Result<(Option<Vec<f32>>, Option<u32>)> {
        let h = self.config.hidden;

        // Path-to-90 step 10 follow-up — Eagle4 GPU capture.
        // Take the capture buffer OUT of self at the top so the per-
        // layer commit branch below can write into it without
        // conflicting with the `arena = &self.decode_arena` borrow
        // held across the layer loop. Restored to self before each
        // return path.
        let mut local_eagle4_capture: Option<Eagle4CaptureBuf> = if self.eagle4_capture_active {
            Some(
                self.eagle4_capture
                    .take()
                    .unwrap_or_else(|| Eagle4CaptureBuf::zeros(h)),
            )
        } else {
            self.eagle4_capture.take();
            None
        };

        // ---- Wedge C: all attention + FFN kernels on TCB (zero counted dispatches).
        // Active when: Metal + arena present, all norm weights pre-uploaded,
        // MLA path active, model_buf present, and all layers have q_a/kv_a pinned.
        #[cfg(target_os = "macos")]
        {
            let tcb_base = self.metal_ctx.is_some()
                && self.decode_arena.is_some()
                && self.layers.iter().all(|l| {
                    l.pinned.attn_norm.is_some() && l.pinned.ffn_norm.is_some()
                });
            let wedge_c_active = tcb_base
                && !self.mla_c_kv.is_empty()
                && self.weights_mmap_buf.is_some()
                && self.embed_buf.is_some()
                && self.final_norm_buf.is_some()
                && self.layers.iter().all(Self::layer_has_tcb_attention);

            if wedge_c_active && !self.mla_c_kv_gpu.is_empty() {
                let eps = self.config.rms_norm_eps;
                let n_layers = self.config.n_layers;
                let kv_lora_rank = self.config.kv_lora_rank;
                let qk_rope_head_dim = self.config.qk_rope_head_dim;
                let decode_timing = std::env::var("DISMANTLE_DECODE_TIMING").is_ok();
                let mut total_us = 0u64;

                // One-time sync: copy CPU KV into GPU-resident buffers on first Wedge C run.
                if !self.mla_kv_gpu_synced && self.kv.seq_len > 0 {
                    for li in 0..n_layers {
                        MetalContext::write_buffer_bytes(
                            &self.mla_c_kv_gpu[li],
                            bytemuck::cast_slice(&self.mla_c_kv[li][..self.kv.seq_len * kv_lora_rank]),
                        );
                        MetalContext::write_buffer_bytes(
                            &self.mla_k_pe_gpu[li],
                            bytemuck::cast_slice(&self.mla_k_pe[li][..self.kv.seq_len * qk_rope_head_dim]),
                        );
                    }
                }

                // seq_slot is the index of the new KV entry; seq_len is what mla_decode sees
                // after kv_append_f32 writes it (same TCB, Metal auto-barriers guarantee order).
                let seq_slot = self.kv.seq_len;
                let seq_len = seq_slot + 1;

                // Pillar 2: encode ALL 27 layers into a SINGLE command buffer.
                // Metal guarantees sequential encoder execution within one command buffer —
                // writes in encoder N are visible to encoder N+1 without explicit barriers.
                // This eliminates 26 commit+wait round-trips (saves ~4ms/token at 162μs/commit).
                // ExactShared originally needed CPU KV mirrors for CPU draft; with
                // GPU shared-only draft, verifier can use the single-TCB fast path.
                let use_gpu_shared_draft =
                    self.speculate_mode == crate::SpeculateMode::ExactShared
                        && self.shared_only_gpu_argmax_available();
                // Path-to-90 step 10 follow-up — Eagle4 GPU capture forces
                // per-layer commits so capture reads at layers 2/13/25 can
                // pick up arena.x_buf + arena.ffn_out_buf mid-flight.
                let use_single_tcb = (self.speculate_mode != crate::SpeculateMode::ExactShared
                    || use_gpu_shared_draft)
                    && !self.eagle4_capture_active;

                // Phase 5B.1: fold final-norm + LM head + argmax into the global TCB when the
                // greedy-GPU path is available. Eliminates Wedge M C-1 mini-TCB and the separate
                // LM head mini-TCB in forward_token_greedy (saves 2 TCB commits per token).
                let lm_head_foldable = use_single_tcb
                    && !read_back
                    && self.lm_head_buf.is_some()
                    && self.logits_buf.is_some()
                    && self.token_buf.is_some();
                let mut lm_head_folded = false;
                let mut folded_greedy_token: Option<u32> = None;

                let t0 = std::time::Instant::now();
                {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let model_buf = self.weights_mmap_buf.as_ref().unwrap();
                    let q4k_schedule = self.kernel_profile.as_ref()
                        .map(|p| p.selected.gemm_q4_k_schedule.as_str())
                        .unwrap_or("scalar");
                    // v2.1.0-T2.11: Q5_0 routed-down schedule (see above).
                    let routed_down_schedule = self.kernel_profile.as_ref()
                        .map(|p| p.selected.routed_down_schedule.as_str())
                        .unwrap_or("basic");
                    // v2.1.0-T2.12: Q6_K shared-down schedule.
                    let shared_down_schedule = self.kernel_profile.as_ref()
                        .map(|p| p.selected.shared_down_schedule.as_str())
                        .unwrap_or("basic");

                    // Create the command buffer before the loop when using single-TCB.
                    let mut global_tcb = if use_single_tcb {
                        Some(crate::metal::TokenCommandBuffer::new(ctx))
                    } else {
                        None
                    };

                    for li in 0..n_layers {
                        crate::metal::set_current_layer(Some(li as u32));

                        let moe_setup = self.ffn_moe_check(li)?;

                        // Borrow the single global TCB or create a per-layer one.
                        let kv_b_proj_buf = self.layers[li].pinned.kv_b_proj.as_ref()
                            .ok_or_else(|| crate::Error::Model(format!("merged: l{li} kv_b_proj not pinned")))?;
                        let o_proj_buf = self.layers[li].pinned.o_proj.as_ref()
                            .ok_or_else(|| crate::Error::Model(format!("merged: l{li} o_proj not pinned")))?;
                        let ffn_norm_buf = self.layers[li].pinned.ffn_norm.as_ref().unwrap();
                        let head_dim_q = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;
                        let scale = 1.0f32 / (head_dim_q as f32).sqrt();

                        // Encode all kernels for this layer into the active TCB.
                        let encode_layer = |tcb: &mut crate::metal::TokenCommandBuffer<'_>| -> Result<bool> {
                            // Phase 1 + kv_append_f32 (writes GPU KV at seq_slot)
                            self.encode_attention_phase1_into_tcb(tcb, li, pos, Some(token), seq_slot)?;
                            // Phase 2: q_b_proj + rope_q
                            self.encode_attention_phase2_tcb(tcb, li, pos)?;
                            // Phase 3: mla_decode reads GPU KV (seq_len entries, including new one)
                            self.dispatch_mla_decode_and_o_proj(
                                tcb, arena, kv_b_proj_buf, o_proj_buf, li, seq_len, scale, h,
                            )?;
                            self.encode_add_and_rmsnorm_tcb(
                                tcb, &arena.x_buf, &arena.out, ffn_norm_buf, eps, h, &arena.x_norm_buf,
                            )?;
                            if let Some(ref setup) = moe_setup {
                                let gate_buf = self.layers[li].pinned.gate_logits_w.as_ref().unwrap();
                                crate::kernels::gemv_f32_moe_pinned_buf_tcb(
                                    tcb, gate_buf,
                                    self.config.n_routed_experts, self.config.hidden,
                                    &arena.x_norm_buf, &arena.moe_logits_buf,
                                )?;
                                crate::kernels::moe_topk_gate_tcb(
                                    tcb,
                                    &arena.moe_logits_buf,
                                    &arena.moe_route_ids_buf,
                                    &arena.moe_route_weights_buf,
                                    self.config.n_routed_experts,
                                    self.config.top_k_routed,
                                )?;
                                // v1.2.0-9: snapshot route IDs into per-layer history
                                // so expert access stats can be updated after the CB
                                // completes. Without this, only the last layer's routes
                                // are visible (the arena buffer is reused each layer).
                                if self.expert_cache.is_some() {
                                    let moe_li = li.saturating_sub(self.config.first_k_dense_layers);
                                    let dst_off = (moe_li * self.config.top_k_routed
                                        * std::mem::size_of::<u32>()) as u64;
                                    let sz = (self.config.top_k_routed
                                        * std::mem::size_of::<u32>()) as u64;
                                    tcb.copy_buffer_bytes(
                                        &arena.moe_route_ids_buf, 0,
                                        &arena.route_history_buf, dst_off,
                                        sz,
                                    )?;
                                }
                                crate::kernels::encode_moe_block_batched_indexed_tcb_with_scratch(
                                    tcb,
                                    model_buf,
                                    setup.routed_gate_off,
                                    setup.routed_up_off,
                                    setup.routed_down_off,
                                    &arena.moe_route_ids_buf,
                                    &arena.moe_route_weights_buf,
                                    self.config.top_k_routed,
                                    &arena.shared_route_ids_buf,
                                    setup.shared_gate_off,
                                    setup.shared_up_off,
                                    setup.shared_down_off,
                                    self.config.hidden,
                                    self.config.moe_intermediate,
                                    setup.shared_mid,
                                    q4k_schedule,
                                    setup.routed_down_kernel_with_schedule(q4k_schedule, routed_down_schedule),
                                    setup.shared_down_kernel_with_schedule(q4k_schedule, shared_down_schedule),
                                    &arena.x_norm_buf,
                                    &arena.ffn_out_buf,
                                    &arena.moe_routed_gate_out_buf,
                                    &arena.moe_routed_up_out_buf,
                                    &arena.moe_routed_act_buf,
                                    &arena.moe_routed_out_buf,
                                    &arena.moe_shared_gate_out_buf,
                                    &arena.moe_shared_up_out_buf,
                                    &arena.moe_shared_act_buf,
                                    &arena.moe_shared_out_buf,
                                )?;
                                Ok(true)
                            } else {
                                let handled = self.encode_dense_ffn_tcb(tcb, li, arena)?;
                                Ok(handled)
                            }
                        };

                        let dense_handled: bool;
                        if use_single_tcb {
                            dense_handled = encode_layer(global_tcb.as_mut().unwrap())?;
                        } else {
                            // Per-layer fallback (ExactShared spec-decode path,
                            // OR Eagle4 GPU capture path — see step 10 follow-up).
                            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                            dense_handled = encode_layer(&mut tcb)?;
                            tcb.commit_and_wait()?;

                            // Keep CPU KV mirrors in sync after each layer commit.
                            let off_c = seq_slot * kv_lora_rank;
                            let off_pe = seq_slot * qk_rope_head_dim;
                            unsafe {
                                let ptr_c = (self.mla_c_kv_gpu[li].contents() as *const f32).add(off_c);
                                let ptr_pe = (self.mla_k_pe_gpu[li].contents() as *const f32).add(off_pe);
                                self.mla_c_kv[li][off_c..off_c + kv_lora_rank]
                                    .copy_from_slice(std::slice::from_raw_parts(ptr_c, kv_lora_rank));
                                self.mla_k_pe[li][off_pe..off_pe + qk_rope_head_dim]
                                    .copy_from_slice(std::slice::from_raw_parts(ptr_pe, qk_rope_head_dim));
                            }

                            // Eagle4 GPU capture — read x_buf + ffn_out_buf at
                            // capture layers (h = x_buf + ffn_out_buf, the full
                            // post-layer-L residual) and read x_norm_buf at
                            // layer 26 (the pre-MoE input that h_shared is
                            // computed against). Cheap CPU reads from Metal
                            // shared buffers; the per-layer commit above
                            // ensures the writes are visible. The capture
                            // buffer lives in `local_eagle4_capture` (taken
                            // out of self at function entry to avoid a borrow
                            // conflict with `arena = &self.decode_arena`);
                            // restored to self.eagle4_capture before return.
                            if let Some(buf) = local_eagle4_capture.as_mut() {
                                if li == 2 || li == 13 || li == 25 {
                                    let mut x_cpu = vec![0.0f32; h];
                                    let mut ffn_cpu = vec![0.0f32; h];
                                    arena.read_x(&mut x_cpu);
                                    arena.read_ffn_out(&mut ffn_cpu);
                                    let target: &mut Vec<f32> = match li {
                                        2 => &mut buf.h_low,
                                        13 => &mut buf.h_mid,
                                        25 => &mut buf.h_high,
                                        _ => unreachable!(),
                                    };
                                    for i in 0..h {
                                        target[i] = x_cpu[i] + ffn_cpu[i];
                                    }
                                }
                                if li == 26 {
                                    arena.read_x_norm(&mut buf.x_norm_26);
                                    // Production MoE kernel writes shared-
                                    // expert contribution to moe_shared_out_buf
                                    // BEFORE summing into ffn_out_buf. Read it
                                    // here to get h_shared GPU-native — same
                                    // numerical value the routed+shared fused
                                    // kernel produces, no separate dispatch.
                                    if moe_setup.is_some() {
                                        arena.read_moe_shared_out(&mut buf.h_shared_gpu);
                                    }
                                }
                            }
                        }

                        let ffn_handled = moe_setup.is_some() || dense_handled;
                        if !ffn_handled {
                            // CPU fallback FFN: only reachable when dense weights are not
                            // yet pinned. Commit current state to GPU, handle on CPU, then
                            // continue encoding (creates a fresh TCB for remaining layers).
                            if use_single_tcb {
                                if let Some(tcb) = global_tcb.take() {
                                    tcb.commit_and_wait()?;
                                }
                            }
                            let mut x_norm = vec![0.0f32; h];
                            arena.read_x_norm(&mut x_norm);
                            let ffn_out = self.ffn(li, &x_norm)?;
                            arena.write_ffn_out(&ffn_out);
                            if use_single_tcb {
                                global_tcb = Some(crate::metal::TokenCommandBuffer::new(ctx));
                            }
                        }
                    }

                    // Phase 5B.1: encode final add_inplace + rmsnorm + (opt) LM head + argmax
                    // into global_tcb so the single commit covers the full forward pass.
                    if let Some(ref mut tcb) = global_tcb {
                        let final_norm_buf = self.final_norm_buf.as_ref().unwrap();
                        if n_layers > 0 {
                            crate::kernels::add_inplace_metal_tcb(
                                tcb, &arena.x_buf, &arena.ffn_out_buf, h,
                            )?;
                        }
                        // Phase 5C.2: final norm writes f16 when x_norm_dtype="f16".
                        let x_norm_f16 = self.use_x_norm_f16();
                        if x_norm_f16 {
                            crate::kernels::rmsnorm_f32_to_f16_tcb(
                                tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_f16_buf,
                            )?;
                        } else {
                            crate::kernels::rmsnorm_metal_buf_tcb(
                                tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_buf,
                            )?;
                        }
                        if lm_head_foldable {
                            let lm_head_buf = self.lm_head_buf.as_ref().unwrap();
                            let logits_buf  = self.logits_buf.as_ref().unwrap();
                            let tok_buf     = self.token_buf.as_ref().unwrap();
                            let vocab = self.config.vocab_size;
                            if x_norm_f16 {
                                // f16 x_norm path: always use scalar f16in kernel (not simdmat).
                                crate::kernels::gemv_f16_f16in_tcb(
                                    tcb, lm_head_buf, vocab, h, &arena.x_norm_f16_buf, logits_buf,
                                )?;
                            } else {
                                let use_simdmat = self.kernel_profile.as_ref()
                                    .map(|p| p.selected.lm_head_schedule.contains("simdgroup-matrix"))
                                    .unwrap_or(false);
                                if use_simdmat {
                                    crate::kernels::gemv_f16_simdmat_tcb(
                                        tcb, lm_head_buf, vocab, h, &arena.x_norm_buf, logits_buf,
                                    )?;
                                } else {
                                    crate::kernels::gemv_f16_metal_buf_tcb(
                                        tcb, lm_head_buf, vocab, h, &arena.x_norm_buf, logits_buf,
                                    )?;
                                }
                            }
                            crate::kernels::sample_argmax_f32_tcb(
                                tcb, logits_buf, tok_buf, vocab,
                            )?;
                            lm_head_folded = true;
                        }
                    }

                    // Single commit: all 27 layers + final-norm + (opt) LM head + argmax.
                    if let Some(tcb) = global_tcb.take() {
                        tcb.commit_and_wait()?;
                    }

                    // Post-commit: read greedy token if LM head was folded in.
                    if lm_head_folded {
                        let tok_buf = self.token_buf.as_ref().unwrap();
                        let tok_ptr = tok_buf.contents() as *const u32;
                        folded_greedy_token = Some(unsafe { *tok_ptr });
                    }

                    // v1.2.0-9: update expert access stats from route_history_buf.
                    // Per-layer route IDs were blit-copied into route_history_buf
                    // inside encode_layer (once per MoE layer). The CB has committed
                    // by this point so the data is CPU-readable via shared memory.
                    if let Some(cache) = self.expert_cache.as_ref() {
                        let top_k = self.config.top_k_routed;
                        let n_moe_li = arena.n_moe_layers;
                        if top_k > 0 && n_moe_li > 0 {
                            let ptr = arena.route_history_buf.contents() as *const u32;
                            let history = unsafe {
                                std::slice::from_raw_parts(ptr, n_moe_li * top_k)
                            };
                            let first_dense = self.config.first_k_dense_layers;
                            for moe_li in 0..n_moe_li {
                                for slot in 0..top_k {
                                    let expert_id = history[moe_li * top_k + slot];
                                    cache.note_access(
                                        first_dense + moe_li,
                                        expert_id,
                                        pos as u64,
                                    );
                                }
                            }
                        }
                    }
                }
                total_us += t0.elapsed().as_micros() as u64;

                self.kv.seq_len += 1;
                self.mla_kv_gpu_synced = true;
                crate::metal::set_current_layer(None);

                if decode_timing {
                    eprintln!("[timing/merged] total={:.1}ms tps_ceil={:.0}",
                        total_us as f64 / 1000.0,
                        1_000_000.0 / total_us as f64);
                }

                // Phase 5B.1: LM head (and final-norm) already folded into global TCB.
                if lm_head_folded {
                    self.eagle4_capture = local_eagle4_capture.take();
                    return Ok((None, folded_greedy_token));
                }

                // Wedge M C-1: merged add_inplace + final-norm into one TCB.
                // (Fallback: only reached when lm_head_foldable was false, e.g. ExactShared
                // without gpu_shared_draft, or when read_back=true needing x_norm readback.)
                {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let final_norm_buf = self.final_norm_buf.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    if n_layers > 0 {
                        crate::kernels::add_inplace_metal_tcb(
                            &mut tcb, &arena.x_buf, &arena.ffn_out_buf, h,
                        )?;
                    }
                    crate::kernels::rmsnorm_metal_buf_tcb(
                        &mut tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_buf,
                    )?;
                    tcb.commit_and_wait()?;
                    if !read_back {
                        self.eagle4_capture = local_eagle4_capture.take();
                        return Ok((None, None));
                    }
                    let mut x_norm = vec![0.0f32; h];
                    arena.read_x_norm(&mut x_norm);
                    self.eagle4_capture = local_eagle4_capture.take();
                    return Ok((Some(x_norm), None));
                }
            }
        }
        // ---- End Wedge C ----

        // Wedge C is the only supported decode path for V2-Lite. If its
        // preconditions don't hold (Metal context + decode arena + pinned
        // norms + MLA cache + GPU-resident KV mirrors + TCB-attention-ready
        // layers), surface a clear error rather than silently routing through
        // a CPU/Wedge-B fallback. The legacy fallbacks were retired in
        // v2.2.0-cleanup-16 after a runtime-panic audit proved them dead
        // under the shipped production profile.
        let _ = token;
        let _ = pos;
        let _ = read_back;
        Err(Error::Model(
            "forward_token: Wedge C preconditions not met; legacy fallback retired"
                .into(),
        ))
    }

    fn profiled_greedy_enabled(&self, sampling: &crate::engine::SamplingParams) -> bool {
        sampling.temperature <= 0.0
            && sampling.repetition_penalty == 1.0
            && self
                .kernel_profile
                .as_ref()
                .map(|p| {
                    let s = &p.selected.lm_head_schedule;
                    s.contains("argmax") || s.contains("simdgroup-matrix")
                })
                .unwrap_or(false)
    }

    fn gemv_f16_argmax_dispatch(&self, rows: usize, cols: usize, x: &[f32]) -> Result<Option<u32>> {
        #[cfg(target_os = "macos")]
        if let (Some(ctx), Some(buf)) = (&self.metal_ctx, &self.lm_head_buf) {
            let token = crate::kernels::gemv_f16_argmax_metal_pinned(ctx, buf, rows, cols, x)?;
            return Ok(Some(token));
        }
        let _ = (rows, cols, x);
        Ok(None)
    }

    /// Encode Phase 2 attention kernels (q_b_proj GEMV + rope_q) into an existing TCB.
    /// Must be called at the START of the β-TCB, before Phase 3 (mla_decode_and_o_proj).
    /// Direct-q layers skip Phase 2 (arena.q already written in Phase 1).
    #[cfg(target_os = "macos")]
    fn encode_attention_phase2_tcb(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        li: usize,
        pos: usize,
    ) -> Result<()> {
        let q_lora_path = self.layers[li].pinned.q_a_proj.is_some()
            && self.layers[li].pinned.q_b_proj.is_some()
            && self.layers[li].pinned.q_a_norm.is_some();
        if !q_lora_path {
            return Ok(());
        }
        let arena = self.decode_arena.as_ref().unwrap();
        let n_heads = self.config.n_heads;
        let head_dim_q = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;
        let q_lora = self.config.q_lora_rank.max(1);
        let qk_nope_head_dim = self.config.qk_nope_head_dim;
        let qk_rope_head_dim = self.config.qk_rope_head_dim;
        let rope_theta = self.config.rope_theta;
        let q_b_proj_buf = self.layers[li].pinned.q_b_proj.as_ref()
            .ok_or_else(|| Error::Model(format!("phase2_tcb: l{li} q_b_proj not pinned")))?;
        let q_out_rows = n_heads * head_dim_q;
        // q_b_proj pinned as f16; cols=1536 rows=3072 both % 8 == 0
        crate::kernels::gemv_f16_simdmat_tcb(
            tcb, q_b_proj_buf, q_out_rows, q_lora,
            &arena.q_lora_normed_buf, &arena.q,
        )?;
        crate::kernels::rope_q_f32_inplace_tcb(
            tcb,
            &arena.q,
            n_heads,
            head_dim_q,
            qk_nope_head_dim,
            qk_rope_head_dim,
            pos as u32,
            rope_theta,
        )
    }

    /// Encode Phase 1 attention kernels (embed/add_inplace + q_a/kv_a + norms + rope_kv)
    /// into an EXISTING TCB without committing. Also encodes `kv_append_f32` to write
    /// the new KV entry directly into the GPU-resident cache at `seq_slot`.
    /// Used by the merged Phase-1+Wedge-N loop to eliminate one commit+wait per layer.
    #[cfg(target_os = "macos")]
    fn encode_attention_phase1_into_tcb(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        li: usize,
        pos: usize,
        pre_phase_token: Option<u32>,
        seq_slot: usize,
    ) -> Result<()> {
        let kv_a_dim = self.config.kv_lora_rank + self.config.qk_rope_head_dim;
        let q_lora = self.config.q_lora_rank.max(1);
        let kv_lora_rank = self.config.kv_lora_rank;
        let qk_rope_head_dim = self.config.qk_rope_head_dim;
        let qk_nope_head_dim = self.config.qk_nope_head_dim;
        let rope_theta = self.config.rope_theta;
        let eps = self.config.rms_norm_eps;
        let h = self.config.hidden;
        let n_heads = self.config.n_heads;
        let head_dim_q = qk_nope_head_dim + qk_rope_head_dim;

        let q_lora_path = self.layers[li].pinned.q_a_proj.is_some()
            && self.layers[li].pinned.q_b_proj.is_some()
            && self.layers[li].pinned.q_a_norm.is_some();

        // v2.2.0-T2.14: opt-in v2t-pattern dispatcher for the fused
        // rmsnorm+attn GEMV. Falls back to the basic kernel when the
        // schedule is not "v2t" or when row/col constraints don't hold.
        let rmsnorm_attn_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.rmsnorm_attn_schedule.as_str())
            .unwrap_or("basic");
        let use_v2t_rmsnorm_attn = rmsnorm_attn_schedule == "v2t";

        let arena = self.decode_arena.as_ref().unwrap();
        let kv_a_proj_buf = self.layers[li].pinned.kv_a_proj_with_mqa.as_ref()
            .ok_or_else(|| Error::Model(format!("p1_into_tcb: l{li} kv_a_proj not pinned")))?;
        let attn_norm_buf = self.layers[li].pinned.attn_norm.as_ref()
            .ok_or_else(|| Error::Model(format!("p1_into_tcb: l{li} attn_norm not pinned")))?;
        let kv_a_norm_buf = self.layers[li].pinned.kv_a_norm.as_ref()
            .ok_or_else(|| Error::Model(format!("p1_into_tcb: l{li} kv_a_norm not pinned")))?;

        let dispatch_rmsnorm_attn = |tcb: &mut crate::metal::TokenCommandBuffer<'_>,
                                     w: &crate::metal::PinnedBuffer,
                                     x: &crate::metal::PinnedBuffer,
                                     out: &crate::metal::PinnedBuffer,
                                     rows: usize,
                                     cols: usize|
         -> Result<()> {
            if use_v2t_rmsnorm_attn && rows % 8 == 0 && cols % 32 == 0 {
                crate::kernels::rmsnorm_gemv_f16w_attn_pinned_v2t_tcb(
                    tcb, w, x, attn_norm_buf, eps, out, rows, cols,
                )
            } else {
                crate::kernels::rmsnorm_gemv_f16w_attn_pinned_tcb(
                    tcb, w, x, attn_norm_buf, eps, out, rows, cols,
                )
            }
        };

        if let Some(tok) = pre_phase_token {
            if li == 0 {
                let embed_buf = self.embed_buf.as_ref().unwrap();
                crate::kernels::embed_lookup_metal_f32_tcb(tcb, embed_buf, tok, h, &arena.x_buf)?;
            } else {
                crate::kernels::add_inplace_metal_tcb(tcb, &arena.x_buf, &arena.ffn_out_buf, h)?;
            }
        }
        if q_lora_path {
            let q_a_proj_buf = self.layers[li].pinned.q_a_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("p1_into_tcb: l{li} q_a_proj not pinned")))?;
            let q_a_norm_buf = self.layers[li].pinned.q_a_norm.as_ref()
                .ok_or_else(|| Error::Model(format!("p1_into_tcb: l{li} q_a_norm not pinned")))?;
            dispatch_rmsnorm_attn(
                tcb, q_a_proj_buf, &arena.x_buf, &arena.q_lora_buf, q_lora, h,
            )?;
            crate::kernels::rmsnorm_metal_buf_tcb(
                tcb, &arena.q_lora_buf, q_a_norm_buf, eps, q_lora, &arena.q_lora_normed_buf,
            )?;
        } else {
            let q_proj_buf = self.layers[li].pinned.q_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("p1_into_tcb: l{li} q_proj not pinned")))?;
            // v2.1.0-T2.13: q_proj pinned as f16; use f16w rmsnorm kernel.
            dispatch_rmsnorm_attn(
                tcb, q_proj_buf, &arena.x_buf, &arena.q, n_heads * head_dim_q, h,
            )?;
        }
        dispatch_rmsnorm_attn(
            tcb, kv_a_proj_buf, &arena.x_buf, &arena.kv_a_out_buf, kv_a_dim, h,
        )?;
        crate::kernels::rmsnorm_metal_buf_tcb(
            tcb, &arena.kv_a_out_buf, kv_a_norm_buf, eps, kv_lora_rank, &arena.c_kv_normed_buf,
        )?;
        crate::kernels::rope_slice_f32_inplace_tcb(
            tcb,
            &arena.kv_a_out_buf,
            kv_lora_rank,
            qk_rope_head_dim,
            pos as u32,
            rope_theta,
        )?;
        if !q_lora_path {
            crate::kernels::rope_q_f32_inplace_tcb(
                tcb,
                &arena.q,
                n_heads,
                head_dim_q,
                qk_nope_head_dim,
                qk_rope_head_dim,
                pos as u32,
                rope_theta,
            )?;
        }
        crate::kernels::kv_append_f32_tcb(
            tcb,
            &arena.c_kv_normed_buf,
            &arena.kv_a_out_buf,
            &self.mla_c_kv_gpu[li],
            &self.mla_k_pe_gpu[li],
            seq_slot,
            kv_lora_rank,
            qk_rope_head_dim,
        )
    }

    /// Encode the leading dense FFN block into an existing TCB.
    /// Returns false for MoE layers or when the dense weights were not pinned.
    #[cfg(target_os = "macos")]
    fn encode_dense_ffn_tcb(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        li: usize,
        arena: &DecodeArena,
    ) -> Result<bool> {
        if !matches!(&self.layers[li].mode, LayerMode::Dense { .. }) {
            return Ok(false);
        }
        let Some(gate_buf) = self.layers[li].pinned.dense_gate_w.as_ref() else {
            return Ok(false);
        };
        let Some(up_buf) = self.layers[li].pinned.dense_up_w.as_ref() else {
            return Ok(false);
        };
        let Some(down_buf) = self.layers[li].pinned.dense_down_w.as_ref() else {
            return Ok(false);
        };
        let h = self.config.hidden;
        let mid = self.config.ffn_intermediate;
        crate::kernels::gemv_f32_attn_pinned_buf_tcb(
            tcb,
            gate_buf,
            mid,
            h,
            &arena.x_norm_buf,
            &arena.dense_gate_out_buf,
        )?;
        crate::kernels::gemv_f32_attn_pinned_buf_tcb(
            tcb,
            up_buf,
            mid,
            h,
            &arena.x_norm_buf,
            &arena.dense_up_out_buf,
        )?;
        crate::kernels::silu_mul_tcb(
            tcb,
            &arena.dense_gate_out_buf,
            &arena.dense_up_out_buf,
            &arena.dense_act_buf,
            mid,
        )?;
        crate::kernels::gemv_f32_attn_pinned_buf_tcb(
            tcb,
            down_buf,
            h,
            mid,
            &arena.dense_act_buf,
            &arena.ffn_out_buf,
        )?;
        Ok(true)
    }

    /// Wedge M C-3: validate MoE TCB conditions and compute weight offsets.
    /// Returns None for Dense layers or when any required weight is absent/wrong-dtype.
    /// Caller must have metal_ctx and decode_arena available (Wedge C precondition).
    #[cfg(target_os = "macos")]
    fn ffn_moe_check(&self, li: usize) -> Result<Option<FfnMoeSetup>> {
        use crate::gguf::GgmlType;
        if self.metal_ctx.is_none() || self.decode_arena.is_none() || self.weights_mmap_buf.is_none() {
            return Ok(None);
        }
        let setup = {
            let layer = &self.layers[li];
            match &layer.mode {
                LayerMode::MoE { routed_fused, shared_fused, .. } => {
                    if routed_fused.gate_w.dtype != GgmlType::Q4_K
                        || routed_fused.up_w.dtype != GgmlType::Q4_K
                        || !matches!(
                            routed_fused.down_w.dtype,
                            GgmlType::Q8_0 | GgmlType::Q5_0 | GgmlType::Q4_K
                        )
                    {
                        return Ok(None);
                    }
                    let (sg, su, sd, sdty, smid) = if let Some(sf) = shared_fused {
                        if sf.gate_w.dtype != GgmlType::Q4_K
                            || sf.up_w.dtype != GgmlType::Q4_K
                            || !matches!(sf.down_w.dtype, GgmlType::Q6_K | GgmlType::Q4_K)
                        {
                            return Ok(None);
                        }
                        let smid = self.config.n_shared_experts * self.config.moe_intermediate;
                        (
                            Some(sf.gate_w.offset),
                            Some(sf.up_w.offset),
                            Some(sf.down_w.offset),
                            Some(sf.down_w.dtype),
                            smid,
                        )
                    } else {
                        (None, None, None, None, 0usize)
                    };
                    if self.layers[li].pinned.gate_logits_w.is_none() {
                        return Ok(None);
                    }
                    FfnMoeSetup {
                        routed_gate_off: routed_fused.gate_w.offset,
                        routed_up_off: routed_fused.up_w.offset,
                        routed_down_off: routed_fused.down_w.offset,
                        routed_down_dtype: routed_fused.down_w.dtype,
                        shared_gate_off: sg,
                        shared_up_off: su,
                        shared_down_off: sd,
                        shared_down_dtype: sdty,
                        shared_mid: smid,
                    }
                }
                LayerMode::Dense { .. } => return Ok(None),
            }
        };
        Ok(Some(setup))
    }

    /// MLA attention for one token. Compresses K/V into the latent
    /// stream, appends to KV cache, then runs softmax-attention against
    /// the cache. The reference path expands KV back to full-head shape
    /// before the attention math so it shares the MHA kernel.
    fn attention(&mut self, li: usize, pos: usize, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let layer = &self.layers[li];
        let head_dim_q = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim;
        let n_heads = cfg.n_heads;
        let h = cfg.hidden;

        // KV allocation hoisted so both q-lora and non-q-lora branches can
        // coalesce their first GEMV (q_a_proj or q_proj) with kv_a_proj into
        // a single dispatch_batch via gemv_f32_attn_pair_dispatch (v0.3.4).
        let kv_a_dim = cfg.kv_lora_rank + cfg.qk_rope_head_dim;
        let mut kv_a = vec![0.0f32; kv_a_dim];

        // Q projection — either direct (q_proj) or via q-lora
        // (q_a_proj → norm → q_b_proj). W1B (Phase 2 super-haul-1):
        // q_a_proj / q_b_proj routed through gemv_f32_attn_dispatch so
        // they hit Metal under cfg(target_os = "macos") + Some(ctx).
        let mut q_full = vec![0.0f32; n_heads * head_dim_q];
        if let (Some(qa), Some(qan), Some(qb)) = (&layer.q_a_proj, &layer.q_a_norm, &layer.q_b_proj)
        {
            let q_lora = cfg.q_lora_rank.max(1);
            let mut t = vec![0.0f32; q_lora];

            // q_a_proj and kv_a_proj share input x.
            self.gemv_f32_attn_pair_dispatch(
                qa,
                layer.pinned.q_a_proj.as_ref(),
                q_lora,
                &layer.kv_a_proj_with_mqa,
                layer.pinned.kv_a_proj_with_mqa.as_ref(),
                kv_a_dim,
                h,
                x,
                &mut t,
                &mut kv_a,
            )?;

            let mut tn = vec![0.0f32; q_lora];
            self.rmsnorm_dispatch(&t, qan, cfg.rms_norm_eps, &mut tn)?;
            self.gemv_f32_attn_dispatch(
                qb,
                layer.pinned.q_b_proj.as_ref(),
                n_heads * head_dim_q,
                q_lora,
                &tn,
                &mut q_full,
            )?;
        } else if !layer.q_proj.is_empty() {
            // q_proj and kv_a_proj share input x — coalesce into one CB.
            self.gemv_f32_attn_pair_dispatch(
                &layer.q_proj, layer.pinned.q_proj.as_ref(), n_heads * head_dim_q,
                &layer.kv_a_proj_with_mqa,
                    layer.pinned.kv_a_proj_with_mqa.as_ref(),
                    kv_a_dim,
                h, x, &mut q_full, &mut kv_a,
            )?;
        } else {
            return Err(Error::Model(format!("layer {li}: no q projection found")));
        }

        let mut c_kv = kv_a[..cfg.kv_lora_rank].to_vec();
        let mut k_pe = kv_a[cfg.kv_lora_rank..].to_vec();

        let mut c_kv_n = vec![0.0f32; cfg.kv_lora_rank];
        self.rmsnorm_dispatch(&c_kv, &layer.kv_a_norm, cfg.rms_norm_eps, &mut c_kv_n)?;
        std::mem::swap(&mut c_kv, &mut c_kv_n);

        // Apply rope to k_pe and to the rope half of each Q head.
        rope_inplace(&mut k_pe, pos as u32, cfg.rope_theta);
        for h_i in 0..n_heads {
            let off = h_i * head_dim_q + cfg.qk_nope_head_dim;
            let rope_part = &mut q_full[off..off + cfg.qk_rope_head_dim];
            rope_inplace(rope_part, pos as u32, cfg.rope_theta);
        }

        // Wedge 1 — Metal MLA decode path.
        // When active, skip the kv_b_proj expand + mha_decode_step and
        // instead append (c_kv, k_pe) to the compressed cache, then
        // dispatch the mla_decode_kernel which operates on the compressed
        // representation for the whole sequence.
        if !self.mla_c_kv.is_empty() {
            if li == 0 && self.kv.seq_len >= self.kv.max_seq {
                return Err(Error::Model("kv cache full".into()));
            }
            let seq_slot = self.kv.seq_len;
            let n_layers = self.config.n_layers;
            Self::mla_kv_append(
                &mut self.mla_c_kv,
                &mut self.mla_k_pe,
                li,
                seq_slot,
                cfg.kv_lora_rank,
                cfg.qk_rope_head_dim,
                &c_kv,
                &k_pe,
            )?;
            if li + 1 == n_layers {
                self.kv.seq_len += 1;
            }

            let seq_len = self.kv.seq_len.max(1);
            let scale = 1.0f32 / (head_dim_q as f32).sqrt();

            #[cfg(target_os = "macos")]
            if let Some(ctx) = self.metal_ctx.as_ref() {
                let kv_b_buf = layer.pinned.kv_b_proj.as_ref().ok_or_else(|| {
                    Error::Model(format!(
                        "layer {li}: kv_b_proj not pinned for MLA Metal path"
                    ))
                })?;

                let layer_cb = self
                    .kernel_profile
                    .as_ref()
                    .map(|p| p.selected.command_buffering == "layer-cb")
                    .unwrap_or(false);

                // Wedge 4 — Decode-arena: when gpu_buffer_reuse == "decode-arena"
                // AND layer_cb AND o_proj pinned, use pre-allocated arena buffers.
                // Saves one allocation+free per attention layer per token.
                if layer_cb {
                    if let (Some(o_proj_buf), Some(arena)) =
                        (layer.pinned.o_proj.as_ref(), self.decode_arena.as_ref())
                    {
                        arena.write_q(&q_full);
                        MetalContext::write_buffer_bytes(
                            &arena.c_kv,
                            bytemuck::cast_slice(&self.mla_c_kv[li][..seq_len * cfg.kv_lora_rank]),
                        );
                        MetalContext::write_buffer_bytes(
                            &arena.k_pe,
                            bytemuck::cast_slice(
                                &self.mla_k_pe[li][..seq_len * cfg.qk_rope_head_dim],
                            ),
                        );
                        let mut out = vec![0.0f32; h];
                        crate::kernels::mla_decode_and_o_proj_arena_metal(
                            ctx,
                            arena,
                            kv_b_buf,
                            o_proj_buf,
                            n_heads,
                            cfg.qk_nope_head_dim,
                            cfg.qk_rope_head_dim,
                            cfg.v_head_dim,
                            cfg.kv_lora_rank,
                            seq_len,
                            scale,
                            h,
                            &mut out,
                        )?;
                        return Ok(out);
                    }
                }

                // Wedge 3 — Layer-CB (no arena): batch mla_decode + o_proj into
                // one command buffer, saving one commit+wait per attention layer.
                if layer_cb {
                    if let Some(o_proj_buf) = layer.pinned.o_proj.as_ref() {
                        let mut out = vec![0.0f32; h];
                        crate::kernels::mla_decode_and_o_proj_metal(
                            ctx,
                            &q_full,
                            &self.mla_c_kv[li][..seq_len * cfg.kv_lora_rank],
                            &self.mla_k_pe[li][..seq_len * cfg.qk_rope_head_dim],
                            kv_b_buf,
                            o_proj_buf,
                            n_heads,
                            cfg.qk_nope_head_dim,
                            cfg.qk_rope_head_dim,
                            cfg.v_head_dim,
                            cfg.kv_lora_rank,
                            seq_len,
                            scale,
                            h,
                            &mut out,
                        )?;
                        return Ok(out);
                    }
                }

                let attn_schedule = self
                    .kernel_profile
                    .as_ref()
                    .map(|p| p.selected.attn_block_schedule.as_str())
                    .unwrap_or("mla");
                if attn_schedule == "flash" {
                    let mut attn_out = vec![0.0f32; n_heads * cfg.v_head_dim];
                    crate::kernels::flash_attn_decode_metal(
                        ctx,
                        &q_full,
                        &self.mla_c_kv[li][..seq_len * cfg.kv_lora_rank],
                        &self.mla_k_pe[li][..seq_len * cfg.qk_rope_head_dim],
                        kv_b_buf,
                        n_heads,
                        cfg.qk_nope_head_dim,
                        cfg.qk_rope_head_dim,
                        cfg.v_head_dim,
                        cfg.kv_lora_rank,
                        seq_len,
                        scale,
                        &mut attn_out,
                    )?;
                    let mut out = vec![0.0f32; h];
                    self.gemv_f32_attn_dispatch(
                        &layer.o_proj,
                        layer.pinned.o_proj.as_ref(),
                        h,
                        n_heads * cfg.v_head_dim,
                        &attn_out,
                        &mut out,
                    )?;
                    return Ok(out);
                }

                let mut attn_out = vec![0.0f32; n_heads * cfg.v_head_dim];
                crate::kernels::mla_decode_metal(
                    ctx,
                    &q_full,
                    &self.mla_c_kv[li][..seq_len * cfg.kv_lora_rank],
                    &self.mla_k_pe[li][..seq_len * cfg.qk_rope_head_dim],
                    kv_b_buf,
                    n_heads,
                    cfg.qk_nope_head_dim,
                    cfg.qk_rope_head_dim,
                    cfg.v_head_dim,
                    cfg.kv_lora_rank,
                    seq_len,
                    scale,
                    &mut attn_out,
                )?;
                let mut out = vec![0.0f32; h];
                self.gemv_f32_attn_dispatch(
                    &layer.o_proj,
                    layer.pinned.o_proj.as_ref(),
                    h,
                    n_heads * cfg.v_head_dim,
                    &attn_out,
                    &mut out,
                )?;
                return Ok(out);
            }
            return Err(Error::Model(
                "mla_decode: Metal context unavailable on this platform".into(),
            ));
        }

        // Reconstruct full K/V via kv_b_proj, which emits
        // (n_heads * (qk_nope_head_dim + v_head_dim)) elements per token.
        // W1B: kv_b_proj onto Metal via the attn dispatcher.
        let kv_b_out_per_head = cfg.qk_nope_head_dim + cfg.v_head_dim;
        let mut kv_b_out = vec![0.0f32; n_heads * kv_b_out_per_head];
        self.gemv_f32_attn_dispatch(
            &layer.kv_b_proj,
            layer.pinned.kv_b_proj.as_ref(),
            n_heads * kv_b_out_per_head,
            cfg.kv_lora_rank,
            &c_kv,
            &mut kv_b_out,
        )?;

        // Append per-head K/V into the cache. Cache slot is sized to
        // (n_kv_heads, head_dim_q) — for MLA we set n_kv_heads=n_heads
        // and head_dim=head_dim_q. v_head_dim ≠ head_dim_q is fine; we
        // pack the v_head_dim values into the V cache slot, padding
        // with zeros if v_head_dim < head_dim_q.
        let stride = cfg.n_kv_heads * head_dim_q;
        let mut k_token = vec![0.0f32; stride];
        let mut v_token = vec![0.0f32; stride];
        for h_i in 0..n_heads {
            let kv_b_head = &kv_b_out[h_i * kv_b_out_per_head..(h_i + 1) * kv_b_out_per_head];
            let kv_h = if cfg.n_kv_heads == n_heads {
                h_i
            } else {
                h_i * cfg.n_kv_heads / n_heads
            };
            // K = nope_part || k_pe (k_pe shared across heads)
            let k_dst = &mut k_token[kv_h * head_dim_q..(kv_h + 1) * head_dim_q];
            for i in 0..cfg.qk_nope_head_dim {
                k_dst[i] = kv_b_head[i];
            }
            for i in 0..cfg.qk_rope_head_dim {
                k_dst[cfg.qk_nope_head_dim + i] = k_pe[i];
            }
            // V = v_head_dim slice, padded.
            let v_dst = &mut v_token[kv_h * head_dim_q..(kv_h + 1) * head_dim_q];
            for i in 0..cfg.v_head_dim.min(head_dim_q) {
                v_dst[i] = kv_b_head[cfg.qk_nope_head_dim + i];
            }
        }
        // We append K/V for every layer at once; here we have only
        // this layer's, so do a per-layer append.
        let off = self.kv.seq_len * stride;
        if li == 0 {
            // First layer's append "claims" the slot for this token.
            if self.kv.seq_len >= self.kv.max_seq {
                return Err(Error::Model("kv cache full".into()));
            }
        }
        self.kv.keys[li][off..off + stride].copy_from_slice(&k_token);
        self.kv.values[li][off..off + stride].copy_from_slice(&v_token);
        if li + 1 == self.config.n_layers {
            self.kv.seq_len += 1;
        }

        // Run MHA against the (this-layer) cache.
        let seq_len = self.kv.seq_len.max(1); // we just appended → at least 1
        let mut attn_out = vec![0.0f32; n_heads * head_dim_q];
        crate::attn::mha_decode_step(
            &q_full,
            &self.kv.keys[li][..seq_len * stride],
            &self.kv.values[li][..seq_len * stride],
            n_heads,
            cfg.n_kv_heads,
            head_dim_q,
            seq_len,
            &mut attn_out,
        )?;

        // Project V dim slice through o_proj. The o_proj takes
        // (n_heads * v_head_dim) → hidden; we slice each head's first
        // v_head_dim entries from attn_out (padded zeros above don't
        // affect anything).
        let mut concat = vec![0.0f32; n_heads * cfg.v_head_dim];
        for h_i in 0..n_heads {
            let src = &attn_out[h_i * head_dim_q..h_i * head_dim_q + cfg.v_head_dim];
            concat[h_i * cfg.v_head_dim..(h_i + 1) * cfg.v_head_dim].copy_from_slice(src);
        }
        let mut out = vec![0.0f32; h];
        self.gemv_f32_attn_dispatch(
            &layer.o_proj,
            layer.pinned.o_proj.as_ref(),
            h,
            n_heads * cfg.v_head_dim,
            &concat,
            &mut out,
        )?;
        Ok(out)
    }

    fn ffn(&self, li: usize, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let layer = &self.layers[li];
        let mut out = vec![0.0f32; cfg.hidden];

        match &layer.mode {
            LayerMode::Dense {
                gate_w,
                up_w,
                down_w,
            } => {
                // Dense FFN intermediate is *not* the MoE intermediate;
                // for DeepSeek-V2-Lite they're 10944 vs 1408.
                let mid = cfg.ffn_intermediate;
                let mut g = vec![0.0f32; mid];
                let mut u = vec![0.0f32; mid];
                let mut a = vec![0.0f32; mid];
                gemv_f32(gate_w, mid, cfg.hidden, x, &mut g);
                gemv_f32(up_w, mid, cfg.hidden, x, &mut u);
                silu_mul(&g, &u, &mut a);
                gemv_f32(down_w, cfg.hidden, mid, &a, &mut out);
            }
            LayerMode::MoE {
                gate_logits_w,
                routed_fused,
                routed,
                shared_fused,
                shared,
            } => {
                let mut logits = vec![0.0f32; cfg.n_routed_experts];
                self.gemv_f32_moe_dispatch(
                    gate_logits_w,
                    cfg.n_routed_experts,
                    cfg.hidden,
                    x,
                    &mut logits,
                )?;
                let routes = topk_gate(&mut logits, cfg.top_k_routed, true);

                // Batched indexed one-command-buffer path (current default).
                if let Some(batched) = self.moe_block_batched_dispatch(
                    routed_fused,
                    routed,
                    shared_fused.as_ref(),
                    &routes,
                    x,
                )? {
                    out = batched;
                    return Ok(out);
                }

                // Reusable scratch buffers — sized for the routed-expert
                // intermediate (1408 in this model). Reallocated below
                // for the shared expert (intermediate = 2816).
                let mid = cfg.moe_intermediate;
                let mut w_buf = Vec::<f32>::with_capacity(mid * cfg.hidden);
                let mut a_buf = vec![0.0f32; mid];
                let mut tmp = vec![0.0f32; cfg.hidden];

                for &(eid, weight) in &routes {
                    let e = &routed[eid];
                    self.moe_expert_pair_matmul_dispatch(
                        &e.gate_w, &e.up_w, mid, cfg.hidden, x,
                        &mut a_buf, &mut w_buf,
                    )?;
                    self.moe_expert_matmul_dispatch(
                        &e.down_w, cfg.hidden, mid, &a_buf, &mut tmp, &mut w_buf,
                    )?;
                    for i in 0..cfg.hidden {
                        out[i] += weight * tmp[i];
                    }
                }

                // Shared expert (fused; intermediate = n_shared * moe_int).
                if let Some(s) = shared.first() {
                    let smid = cfg.n_shared_experts * cfg.moe_intermediate;
                    let mut sa = vec![0.0f32; smid];

                    self.moe_expert_pair_matmul_dispatch(
                        &s.gate_w, &s.up_w, smid, cfg.hidden, x,
                        &mut sa, &mut w_buf,
                    )?;

                    self.moe_expert_matmul_dispatch(
                        &s.down_w, cfg.hidden, smid, &sa, &mut tmp, &mut w_buf,
                    )?;
                    for i in 0..cfg.hidden {
                        out[i] += tmp[i];
                    }
                }
            }
        }
        Ok(out)
    }

    /// Phase 3 prep: like `ffn()` but skips routed-expert contributions.
    /// Routing gate logits and topk_gate are still computed (fair comparison
    /// with `ffn()`), but the resulting contributions are zeroed. Only the
    /// shared experts run. Dense layers run normally (no routed experts exist).
    fn ffn_shared_only(&self, li: usize, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let layer = &self.layers[li];
        let mut out = vec![0.0f32; cfg.hidden];

        match &layer.mode {
            LayerMode::Dense {
                gate_w,
                up_w,
                down_w,
            } => {
                // Dense layers have no routed experts; identical to full ffn.
                let mid = cfg.ffn_intermediate;
                let mut g = vec![0.0f32; mid];
                let mut u = vec![0.0f32; mid];
                let mut a = vec![0.0f32; mid];
                gemv_f32(gate_w, mid, cfg.hidden, x, &mut g);
                gemv_f32(up_w, mid, cfg.hidden, x, &mut u);
                silu_mul(&g, &u, &mut a);
                gemv_f32(down_w, cfg.hidden, mid, &a, &mut out);
            }
            LayerMode::MoE {
                gate_logits_w,
                routed_fused: _,
                routed: _,
                shared_fused,
                shared,
            } => {
                // Routing: compute logits + topk for fair comparison, but skip
                // the routed contributions.
                let mut logits = vec![0.0f32; cfg.n_routed_experts];
                self.gemv_f32_moe_dispatch(
                    gate_logits_w,
                    cfg.n_routed_experts,
                    cfg.hidden,
                    x,
                    &mut logits,
                )?;
                let _routes = topk_gate(&mut logits, cfg.top_k_routed, true);

                // Shared expert only (same code as in ffn()).
                let mut w_buf = Vec::<f32>::new();
                let mut tmp = vec![0.0f32; cfg.hidden];
                if let Some(s) = shared.first() {
                    let smid = cfg.n_shared_experts * cfg.moe_intermediate;
                    let mut sa = vec![0.0f32; smid];
                    self.moe_expert_pair_matmul_dispatch(
                        &s.gate_w, &s.up_w, smid, cfg.hidden, x,
                        &mut sa, &mut w_buf,
                    )?;
                    self.moe_expert_matmul_dispatch(
                        &s.down_w, cfg.hidden, smid, &sa, &mut tmp, &mut w_buf,
                    )?;
                    for i in 0..cfg.hidden {
                        out[i] += tmp[i];
                    }
                } else if shared_fused.is_some() {
                    // Fused shared path — fall back to the non-fused for simplicity
                    // (shared_fused is None in DeepSeek-V2-Lite w/ current schedule).
                }
            }
        }
        Ok(out)
    }

    /// Path-to-90 step 3 — CPU-walk that captures EAGLE-4's 5-input
    /// bundle in one decode step. Same layer choreography as
    /// [`Self::forward_token_shared_only`] (full attention via
    /// `attention()` which advances the CPU-side MLA KV mirror; full
    /// `ffn()` so x_buf state at the capture layers is correct), with
    /// hooks for the EAGLE-4 capture points:
    ///
    /// - After layers 2 / 13 / 25 are fully applied (attn-add + ffn-add):
    ///   capture x as h_low / h_mid / h_high.
    /// - At layer 26 (the last MoE layer), before its MoE runs: capture
    ///   the post-attn-rmsnorm input and feed it through
    ///   `ffn_shared_only(26, _)` to produce h_shared.
    ///
    /// Layer indices are hard-coded to V2-Lite's {2, 13, 25, 26} per
    /// `reports/path_to_90/eagle4_convergence.md`. The CPU-walk is slow
    /// vs the GPU Wedge C path but is correct and self-contained;
    /// production wiring (Eagle4Head::propose calling this from inside
    /// the decode loop) lands in step 5+ once the CPU path is parity-
    /// validated against the Python reference (step 6).
    pub fn forward_token_eagle4_capture(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<crate::engine::Eagle4Inputs> {
        let cfg = &self.config;
        let h = cfg.hidden;
        let n_layers = cfg.n_layers;
        let eps = cfg.rms_norm_eps;

        // V2-Lite layer indices the EAGLE-4 head is trained against.
        // See `reports/path_to_90/eagle4_convergence.md § EAGLE-4
        // forward`. Hard-coded for now; if non-V2-Lite engines later
        // implement this, they'll override with their own indices.
        const LAYER_LOW: usize = 2;
        const LAYER_MID: usize = 13;
        const LAYER_HIGH: usize = 25;
        const LAYER_SHARED: usize = 26;

        if n_layers <= LAYER_SHARED {
            return Err(Error::Model(format!(
                "forward_token_eagle4_capture: needs n_layers > {LAYER_SHARED}, got {n_layers}"
            )));
        }

        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        let mut h_low = vec![0.0f32; h];
        let mut h_mid = vec![0.0f32; h];
        let mut h_high = vec![0.0f32; h];
        let mut x_norm_pre_mlp_shared = vec![0.0f32; h];

        for li in 0..n_layers {
            crate::metal::set_current_layer(Some(li as u32));

            // Attention block: pre-norm → attention → residual add.
            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(
                &x,
                &self.layers[li].attn_norm,
                eps,
                &mut x_norm,
            )?;
            let attn_out = self.attention(li, pos, &x_norm)?;
            add_inplace(&mut x, &attn_out);

            // FFN block: pre-norm → ffn → residual add.
            self.rmsnorm_dispatch(
                &x.clone(),
                &self.layers[li].ffn_norm,
                eps,
                &mut x_norm,
            )?;

            // Capture layer 26's pre-MLP input BEFORE its MoE runs.
            // h_shared is computed after the loop via a CPU dequant path
            // (the existing `ffn_shared_only` helper goes through a GPU
            // dispatch that returns silently-zero output on V2-Lite Q4_K_M
            // for the shared-expert (2816×2048) shape — see followup in
            // commit message). The CPU path is slow but correctness-
            // first, which is the right tradeoff for `_for_test`.
            if li == LAYER_SHARED {
                x_norm_pre_mlp_shared.copy_from_slice(&x_norm);
            }

            let ffn_out = self.ffn(li, &x_norm)?;
            add_inplace(&mut x, &ffn_out);

            // Capture full layer output (post-attn + post-ffn residual).
            if li == LAYER_LOW {
                h_low.copy_from_slice(&x);
            } else if li == LAYER_MID {
                h_mid.copy_from_slice(&x);
            } else if li == LAYER_HIGH {
                h_high.copy_from_slice(&x);
            }
        }
        crate::metal::set_current_layer(None);

        // h_shared = shared-expert forward at layer 26 against the
        // captured pre-MLP hidden. Direct CPU dequant + gemv_f32 path
        // to dodge the latent zero-output bug in ffn_shared_only's GPU
        // dispatch (see method docstring).
        let h_shared = self.cpu_shared_expert_forward(LAYER_SHARED, &x_norm_pre_mlp_shared)?;

        Ok(crate::engine::Eagle4Inputs {
            prev_token: token,
            h_low,
            h_mid,
            h_high,
            h_shared,
        })
    }

    /// Path-to-90 step 8 — CPU-walk forward that returns BOTH the
    /// EAGLE-4 5-input bundle AND V2-Lite's own greedy argmax at this
    /// position. Used by the Eagle4 spec-decode branch to (a) feed the
    /// trained head's `propose`, and (b) compute the verifier's
    /// canonical answer that the draft is judged against.
    ///
    /// Single forward = one CPU walk through all 27 layers + final norm
    /// + LM head + argmax. KV cache (CPU mirror) advances by one slot.
    pub fn forward_token_eagle4_capture_with_argmax(
        &mut self,
        token: u32,
        pos: usize,
    ) -> Result<(crate::engine::Eagle4Inputs, u32)> {
        // Reuse the existing capture for the eagle4 5-input bundle. It
        // walks all 27 layers and advances CPU KV. After it returns,
        // x_buf in arena (and the CPU side) reflect the post-layer-26
        // residual. We then re-run a minimal residual-stream walk to
        // get the final_norm + LM head argmax — but the capture already
        // discards `x` post-loop, so we instead compute the verifier
        // argmax via a separate forward_token call into the production
        // CPU path.
        //
        // forward_token_shared_only would be wrong here (it skips
        // routed experts). The right reference is V2-Lite's full
        // forward. But that would advance KV AGAIN (double-advance,
        // wrong). So instead we reproduce the CPU walk INLINE and
        // capture both the eagle4 hiddens and the final argmax in one
        // pass.

        use crate::engine::Eagle4Inputs;
        let h = self.config.hidden;
        let n_layers = self.config.n_layers;
        let eps = self.config.rms_norm_eps;
        let vocab_size = self.config.vocab_size;

        const LAYER_LOW: usize = 2;
        const LAYER_MID: usize = 13;
        const LAYER_HIGH: usize = 25;
        const LAYER_SHARED: usize = 26;

        if n_layers <= LAYER_SHARED {
            return Err(Error::Model(format!(
                "forward_token_eagle4_capture_with_argmax: needs n_layers > {LAYER_SHARED}, got {n_layers}"
            )));
        }

        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        let mut h_low = vec![0.0f32; h];
        let mut h_mid = vec![0.0f32; h];
        let mut h_high = vec![0.0f32; h];
        let mut x_norm_pre_mlp_shared = vec![0.0f32; h];

        for li in 0..n_layers {
            crate::metal::set_current_layer(Some(li as u32));
            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(&x, &self.layers[li].attn_norm, eps, &mut x_norm)?;
            let attn_out = self.attention(li, pos, &x_norm)?;
            add_inplace(&mut x, &attn_out);

            self.rmsnorm_dispatch(&x.clone(), &self.layers[li].ffn_norm, eps, &mut x_norm)?;
            if li == LAYER_SHARED {
                x_norm_pre_mlp_shared.copy_from_slice(&x_norm);
            }
            let ffn_out = self.ffn(li, &x_norm)?;
            add_inplace(&mut x, &ffn_out);

            if li == LAYER_LOW {
                h_low.copy_from_slice(&x);
            } else if li == LAYER_MID {
                h_mid.copy_from_slice(&x);
            } else if li == LAYER_HIGH {
                h_high.copy_from_slice(&x);
            }
        }
        crate::metal::set_current_layer(None);

        let h_shared = self.cpu_shared_expert_forward(LAYER_SHARED, &x_norm_pre_mlp_shared)?;

        // V2-Lite verifier path: final_norm + lm_head + argmax.
        let mut x_final_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, eps, &mut x_final_norm)?;
        let mut logits = vec![0.0f32; vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        self.gemv_f16_dispatch(w_f16, vocab_size, h, &x_final_norm, &mut logits)?;
        let v2_argmax = crate::kernels::argmax_f32(&logits);

        Ok((
            Eagle4Inputs {
                prev_token: token,
                h_low,
                h_mid,
                h_high,
                h_shared,
            },
            v2_argmax,
        ))
    }

    /// EAGLE-4 capture helper — CPU-only shared-expert forward at the
    /// given MoE layer. Used by [`Self::forward_token_eagle4_capture`]
    /// to produce `h_shared` deterministically.
    ///
    /// Reason for the bypass: `ffn_shared_only` (and the unfused-MoE
    /// fallback in `ffn`) dispatches the shared-expert GEMVs through
    /// `moe_expert_pair_matmul_dispatch` → Q4_K Metal kernels, which
    /// at the shared-expert shape `(smid=2816, hidden=2048)` on
    /// V2-Lite returns silently-zero output. Production decode avoids
    /// this path entirely because `ffn` short-circuits via the fused
    /// `moe_block_batched_dispatch` first, so the bug went unnoticed.
    /// Bypass below dequantizes the three shared-expert tensors with
    /// the same `dequant_ref_into` helper used elsewhere on this
    /// engine and runs three CPU `gemv_f32` calls + `silu_mul`. Slow
    /// (~10–20 ms per call at smid=2816) but correctness-first.
    fn cpu_shared_expert_forward(&self, li: usize, x: &[f32]) -> Result<Vec<f32>> {
        let cfg = &self.config;
        let layer = &self.layers[li];
        match &layer.mode {
            LayerMode::MoE { shared, .. } => {
                let s = shared.first().ok_or_else(|| {
                    Error::Model(format!(
                        "cpu_shared_expert_forward: layer {li} has no shared expert"
                    ))
                })?;
                let smid = cfg.n_shared_experts * cfg.moe_intermediate;
                let h = cfg.hidden;

                let mut gate_w = vec![0.0f32; smid * h];
                let mut up_w   = vec![0.0f32; smid * h];
                let mut down_w = vec![0.0f32; h * smid];
                self.dequant_ref_into(&s.gate_w, &mut gate_w)?;
                self.dequant_ref_into(&s.up_w,   &mut up_w)?;
                self.dequant_ref_into(&s.down_w, &mut down_w)?;

                let mut g = vec![0.0f32; smid];
                let mut u = vec![0.0f32; smid];
                let mut a = vec![0.0f32; smid];
                let mut out = vec![0.0f32; h];
                gemv_f32(&gate_w, smid, h, x, &mut g);
                gemv_f32(&up_w,   smid, h, x, &mut u);
                silu_mul(&g, &u, &mut a);
                gemv_f32(&down_w, h, smid, &a, &mut out);
                Ok(out)
            }
            LayerMode::Dense { .. } => Err(Error::Model(format!(
                "cpu_shared_expert_forward: layer {li} is dense, no shared expert"
            ))),
        }
    }

    /// Phase 3 prep: like `forward_token` but uses `ffn_shared_only` at every layer.
    /// Exposes the shared-only logits for acceptance-rate measurement.
    pub fn forward_token_shared_only(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let h = self.config.hidden;
        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        for li in 0..self.config.n_layers {
            crate::metal::set_current_layer(Some(li as u32));

            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(
                &x,
                &self.layers[li].attn_norm,
                self.config.rms_norm_eps,
                &mut x_norm,
            )?;
            let attn_out = self.attention(li, pos, &x_norm)?;
            add_inplace(&mut x, &attn_out);

            self.rmsnorm_dispatch(
                &x.clone(),
                &self.layers[li].ffn_norm,
                self.config.rms_norm_eps,
                &mut x_norm,
            )?;
            let ffn_out = self.ffn_shared_only(li, &x_norm)?;
            add_inplace(&mut x, &ffn_out);
        }
        crate::metal::set_current_layer(None);

        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, self.config.rms_norm_eps, &mut x_norm)?;

        let mut logits = vec![0.0f32; self.config.vocab_size];
        let w_f16: &[f16] = match &self.lm_head {
            Some(w) => w,
            None => &self.embed,
        };
        self.gemv_f16_dispatch(w_f16, self.config.vocab_size, h, &x_norm, &mut logits)?;
        Ok(logits)
    }
}

/// Returns true if the GGUF metadata's architecture is one of the
/// DeepSeek-V2 family identifiers.
pub fn is_deepseek_arch(t: &TensorInfo) -> bool {
    t.name.contains("kv_lora_rank") || t.name.contains("attn_kv_a_mqa")
}
