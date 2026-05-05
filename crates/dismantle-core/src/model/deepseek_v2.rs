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

    /// Wedge 4 — Decode-arena: pre-allocated Metal buffers for the MLA
    /// attention hot path. Allocated once at load time; reused across all
    /// decode steps. Eliminates per-dispatch `new_buffer` overhead.
    /// `Some` only when Metal is available and `gpu_buffer_reuse == "decode-arena"`.
    pub decode_arena: Option<DecodeArena>,

    /// Phase 7: activation dtype for f16 bridge kernels.
    pub activation_dtype: crate::engine::ActivationDtype,
    /// Phase E: residual stream dtype. F16 = x is Vec<f16> throughout.
    pub residual_dtype: crate::engine::ResidualDtype,

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

impl DeepSeekV2 {
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

        let mla_metal = config
            .kernel_profile
            .as_ref()
            .map(|p| p.selected.mla_schedule.as_str() == "metal-mla")
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
        let speculate_mode = if config.speculate && config.speculate_mode == SpeculateMode::Off {
            SpeculateMode::ExactShared
        } else {
            config.speculate_mode
        };
        let verify_window = config.verify_window;

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
                if let Some(qa) = layer.q_a_proj.as_deref() {
                    layer.pinned.q_a_proj = Some(upload(qa));
                }
                if let Some(qb) = layer.q_b_proj.as_deref() {
                    layer.pinned.q_b_proj = Some(upload(qb));
                }
                layer.pinned.kv_a_proj_with_mqa = Some(upload(&layer.kv_a_proj_with_mqa));
                layer.pinned.kv_b_proj = Some(upload(&layer.kv_b_proj));
                layer.pinned.o_proj = Some(upload(&layer.o_proj));
                if !layer.q_proj.is_empty() {
                    layer.pinned.q_proj = Some(upload(&layer.q_proj));
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
                        cfg.max_seq_len,
                        cfg.n_routed_experts,
                        cfg.q_lora_rank,
                    )
                })
            } else {
                None
            }
        };
        #[cfg(not(target_os = "macos"))]
        let decode_arena: Option<DecodeArena> = None;

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
            sampler,
            _weights_path: weights.to_owned(),
            metal_ctx,
            lm_head_buf,
            weights_mmap_buf,
            kernel_profile: config.kernel_profile,
            speculate_mode,
            verify_window,
            decode_arena,
            activation_dtype: config.activation_dtype,
            residual_dtype: config.residual_dtype,
            embed_buf,
            final_norm_buf,
            logits_buf,
            token_buf,
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
        // draft_accepted / draft_rejected populated by real spec-decode path (Phase 6).
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

    fn forward_tokens_batched_for_test(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
    ) -> Result<Vec<Vec<f32>>> {
        self.forward_tokens_batched(tokens, positions)
    }

    fn reset_kv_for_test(&mut self) {
        self.reset_kv_state();
    }
}

impl DeepSeekV2 {
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
            if let Some(buf) = pinned {
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
            if let (Some(buf_a), Some(buf_b)) = (pinned_a, pinned_b) {
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
                let schedule = self
                    .kernel_profile
                    .as_ref()
                    .map(|p| p.selected.gemm_q4_k_schedule.as_str())
                    .unwrap_or("scalar");
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
                if schedule == "simdgroup" {
                    return crate::kernels::dispatch_gemv_q4_k_m_simd_batched(
                        ctx, bytes, rows, cols, x, out,
                    );
                }
                return crate::kernels::gemv_q4_k_m(ctx, bytes, rows, cols, x, out);
            }
        }
        self.dequant_ref_into(t, scratch)?;
        gemv_f32(scratch, rows, cols, x, out);
        Ok(())
    }

    /// Wedge 2 — two-stage fused MoE dispatch, gated by
    /// `profile.selected.moe_schedule == "two-stage"`.
    /// Returns `Some(output)` when the kernel fires, `None` to fall through.
    fn moe_block_two_stage_dispatch(
        &self,
        routed_fused: &MoEFusedTensors,
        shared_fused: Option<&MoEFusedTensors>,
        routes: &[(usize, f32)],
        x: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        let wants_two_stage = self
            .kernel_profile
            .as_ref()
            .map(|p| p.selected.moe_schedule == "two-stage")
            .unwrap_or(false);
        if !wants_two_stage {
            return Ok(None);
        }

        #[cfg(target_os = "macos")]
        {
            let Some(ctx) = &self.metal_ctx else {
                return Ok(None);
            };

            // Dtype guards: same quant scheme as v2lite.
            if routed_fused.gate_w.dtype != GgmlType::Q4_K
                || routed_fused.up_w.dtype != GgmlType::Q4_K
                || routed_fused.down_w.dtype != GgmlType::Q8_0
            {
                return Ok(None);
            }
            let Some(shared) = shared_fused else {
                return Ok(None);
            };
            if shared.gate_w.dtype != GgmlType::Q4_K
                || shared.up_w.dtype != GgmlType::Q4_K
                || shared.down_w.dtype != GgmlType::Q6_K
            {
                return Ok(None);
            }

            let mmap = &self.gguf.mmap;
            let routed_gate = &mmap[routed_fused.gate_w.offset
                ..routed_fused.gate_w.offset + routed_fused.gate_w.byte_size];
            let routed_up = &mmap
                [routed_fused.up_w.offset..routed_fused.up_w.offset + routed_fused.up_w.byte_size];
            let routed_down = &mmap[routed_fused.down_w.offset
                ..routed_fused.down_w.offset + routed_fused.down_w.byte_size];
            let shared_gate =
                &mmap[shared.gate_w.offset..shared.gate_w.offset + shared.gate_w.byte_size];
            let shared_up = &mmap[shared.up_w.offset..shared.up_w.offset + shared.up_w.byte_size];
            let shared_down =
                &mmap[shared.down_w.offset..shared.down_w.offset + shared.down_w.byte_size];

            let mut route_ids = Vec::with_capacity(routes.len());
            let mut route_weights = Vec::with_capacity(routes.len());
            for &(eid, weight) in routes {
                route_ids.push(eid as u32);
                route_weights.push(weight);
            }

            let n_shared = self.config.n_shared_experts;
            let shared_mid = n_shared * self.config.moe_intermediate;
            let mut out = vec![0.0f32; self.config.hidden];
            crate::kernels::moe_block_two_stage_metal(
                ctx,
                routed_gate,
                routed_up,
                routed_down,
                shared_gate,
                shared_up,
                shared_down,
                &route_ids,
                &route_weights,
                self.config.n_routed_experts,
                n_shared,
                self.config.hidden,
                self.config.moe_intermediate,
                shared_mid,
                x,
                &mut out,
            )?;
            return Ok(Some(out));
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (routed_fused, shared_fused, routes, x);
            Ok(None)
        }
    }

    /// Stage B.4 — single-kernel fused MoE dispatch, gated by
    /// `profile.selected.moe_schedule == "single-kernel"`.
    /// Returns `Some(output)` when the kernel fires, `None` to fall through.
    fn moe_block_fused_v2lite_dispatch(
        &self,
        routed_fused: &MoEFusedTensors,
        shared_fused: Option<&MoEFusedTensors>,
        routes: &[(usize, f32)],
        x: &[f32],
    ) -> Result<Option<Vec<f32>>> {
        // Only activate when the profile explicitly requests single-kernel.
        let wants_single = self
            .kernel_profile
            .as_ref()
            .map(|p| p.selected.moe_schedule == "single-kernel")
            .unwrap_or(false);
        if !wants_single {
            return Ok(None);
        }

        #[cfg(target_os = "macos")]
        {
            let Some(ctx) = &self.metal_ctx else {
                return Ok(None);
            };
            let Some(model_buf) = &self.weights_mmap_buf else {
                return Ok(None);
            };

            // Dtype guards: must match the v2lite kernel's expectations.
            if routed_fused.gate_w.dtype != GgmlType::Q4_K
                || routed_fused.up_w.dtype != GgmlType::Q4_K
                || routed_fused.down_w.dtype != GgmlType::Q8_0
            {
                return Ok(None);
            }
            let Some(shared) = shared_fused else {
                return Ok(None);
            };
            if shared.gate_w.dtype != GgmlType::Q4_K
                || shared.up_w.dtype != GgmlType::Q4_K
                || shared.down_w.dtype != GgmlType::Q6_K
            {
                return Ok(None);
            }

            let mut route_ids = Vec::with_capacity(routes.len());
            let mut route_weights = Vec::with_capacity(routes.len());
            for &(eid, weight) in routes {
                route_ids.push(eid as u32);
                route_weights.push(weight);
            }

            let shared_mid = self.config.n_shared_experts * self.config.moe_intermediate;
            let mut out = vec![0.0f32; self.config.hidden];
            crate::kernels::moe_block_fused_v2lite_indexed_metal(
                ctx,
                model_buf,
                routed_fused.gate_w.offset,
                routed_fused.up_w.offset,
                routed_fused.down_w.offset,
                shared.gate_w.offset,
                shared.up_w.offset,
                shared.down_w.offset,
                &route_ids,
                &route_weights,
                self.config.n_routed_experts,
                self.config.hidden,
                self.config.moe_intermediate,
                shared_mid,
                x,
                &mut out,
            )?;
            return Ok(Some(out));
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (routed_fused, shared_fused, routes, x);
            Ok(None)
        }
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

    /// Phase A Wedge A1 — batched forward pass scaffold.
    ///
    /// Accepts N tokens and returns N logit vectors. A1 implementation is
    /// token-first (same as `forward_tokens`) to maintain correct KV-cache
    /// slot ordering. The `kv.seq_len` slot mechanism advances once per token
    /// per full-forward, so layer-first ordering requires explicit slot
    /// management that A2 will introduce alongside the batched attention kernel.
    ///
    /// A2 replaces this with a layer-first loop using `mla_decode_kernel_batched`
    /// and explicit slot tracking (base_slot + m per token). A3 replaces the
    /// inner FFN loop with batched MoE dispatch.
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
        // A1: token-first loop (semantically identical to forward_tokens).
        // KV slot ordering is preserved because kv.seq_len advances after each
        // complete token forward, which is what self.attention() expects.
        let mut out = Vec::with_capacity(tokens.len());
        for (i, &token) in tokens.iter().enumerate() {
            out.push(self.forward_token(token, positions[i])?);
        }
        Ok(out)
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
        let x_norm = self.forward_token_final_norm(token, pos)?;

        // v1.0.0-E: GPU argmax via TCB (zero counted dispatches) when the full
        // Wedge C stack ran. arena.x_norm_buf holds the final-normed residual
        // written by the Wedge C final-norm mini-TCB — same data as x_norm but
        // already on-GPU, so only 4 bytes cross the bus instead of 408 KB.
        #[cfg(target_os = "macos")]
        {
            let use_f16 = self.activation_dtype == crate::engine::ActivationDtype::F16;
            let wedge_e_ok = !use_f16
                && self.metal_ctx.is_some()
                && self.decode_arena.is_some()
                && !self.mla_c_kv.is_empty()
                && self.weights_mmap_buf.is_some()
                && self.embed_buf.is_some()
                && self.final_norm_buf.is_some()
                && self.lm_head_buf.is_some()
                && self.logits_buf.is_some()
                && self.token_buf.is_some()
                && self.layers.iter().all(|l| {
                    l.pinned.attn_norm.is_some()
                        && l.pinned.ffn_norm.is_some()
                        && l.pinned.q_a_proj.is_some()
                        && l.pinned.q_b_proj.is_some()
                        && l.pinned.kv_a_proj_with_mqa.is_some()
                        && l.pinned.kv_b_proj.is_some()
                        && l.pinned.o_proj.is_some()
                        && l.pinned.q_a_norm.is_some()
                        && l.pinned.kv_a_norm.is_some()
                });
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
                    crate::kernels::gemv_f16_metal_buf_tcb(
                        &mut tcb, lm_head_buf, vocab, cols, &arena.x_norm_buf, logits_buf,
                    )?;
                    crate::kernels::sample_argmax_f32_tcb(&mut tcb, logits_buf, tok_buf, vocab)?;
                    tcb.commit_and_wait()?;
                    let tok_ptr = tok_buf.contents() as *const u32;
                    unsafe { *tok_ptr }
                };
                return Ok(Some(result));
            }
        }

        self.gemv_f16_argmax_dispatch(self.config.vocab_size, self.config.hidden, &x_norm)
    }

    fn forward_token_final_norm(&mut self, token: u32, pos: usize) -> Result<Vec<f32>> {
        let h = self.config.hidden;
        let mut x = vec![0.0f32; h];
        embed_lookup(&self.embed, h, token, &mut x);

        let use_f16 = self.activation_dtype == crate::engine::ActivationDtype::F16;
        let residual_f16 = self.residual_dtype == crate::engine::ResidualDtype::F16;

        // ---- Wedge F: f16 residual stream path.
        // When residual_dtype=F16: wedge_f_x_f16 (f16) holds the running residual.
        // x_norm_buf (f32) is populated by rmsnorm_f16_to_f32_tcb and fed to all
        // GEMV kernels unchanged. Bandwidth savings come from f16 add_inplace and
        // embed_lookup on the 2KB residual instead of 4KB.
        // Same Wedge C conditions required (attention_tcb_inner / ffn_tcb_inner
        // read x_norm_buf f32 — no changes needed there).
        #[cfg(target_os = "macos")]
        if residual_f16 && !use_f16 {
            let tcb_base = self.metal_ctx.is_some()
                && self.decode_arena.is_some()
                && self.layers.iter().all(|l| {
                    l.pinned.attn_norm.is_some() && l.pinned.ffn_norm.is_some()
                });
            let wedge_f_active = tcb_base
                && !self.mla_c_kv.is_empty()
                && self.weights_mmap_buf.is_some()
                && self.embed_buf.is_some()
                && self.final_norm_buf.is_some()
                && self.layers.iter().all(|l| {
                    l.pinned.q_a_proj.is_some()
                        && l.pinned.q_b_proj.is_some()
                        && l.pinned.kv_a_proj_with_mqa.is_some()
                        && l.pinned.kv_b_proj.is_some()
                        && l.pinned.o_proj.is_some()
                        && l.pinned.q_a_norm.is_some()
                        && l.pinned.kv_a_norm.is_some()
                });

            if wedge_f_active {
                let eps = self.config.rms_norm_eps;
                let n_layers = self.config.n_layers;

                // Embed lookup → wedge_f_x_f16 (f16, no CPU round-trip).
                {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let embed_buf = self.embed_buf.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    crate::kernels::embed_lookup_f16_tcb(
                        &mut tcb, embed_buf, token, h, &arena.wedge_f_x_f16,
                    )?;
                    tcb.commit_and_wait()?;
                }

                for li in 0..n_layers {
                    crate::metal::set_current_layer(Some(li as u32));

                    // Mini-TCB α: (add_inplace_f16 if li>0) + rmsnorm_f16_to_f32 → x_norm_buf.
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let attn_norm_buf = self.layers[li].pinned.attn_norm.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        if li > 0 {
                            crate::kernels::add_inplace_f16_tcb(
                                &mut tcb, &arena.wedge_f_x_f16, &arena.wedge_f_delta_f16, h,
                            )?;
                        }
                        crate::kernels::rmsnorm_f16_to_f32_tcb(
                            &mut tcb, &arena.wedge_f_x_f16, attn_norm_buf, eps, h, &arena.x_norm_buf,
                        )?;
                        tcb.commit_and_wait()?;
                    }

                    // Attention via TCB: reads x_norm_buf (f32), writes arena.out (f32).
                    self.attention_tcb_inner(li, pos)?;

                    // Mini-TCB β: cast attn_out f32→f16 delta, add to x_f16, rmsnorm_f16_to_f32.
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let ffn_norm_buf = self.layers[li].pinned.ffn_norm.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        crate::kernels::cast_f32_to_f16_tcb(
                            &mut tcb, &arena.out, &arena.wedge_f_delta_f16, h,
                        )?;
                        crate::kernels::add_inplace_f16_tcb(
                            &mut tcb, &arena.wedge_f_x_f16, &arena.wedge_f_delta_f16, h,
                        )?;
                        crate::kernels::rmsnorm_f16_to_f32_tcb(
                            &mut tcb, &arena.wedge_f_x_f16, ffn_norm_buf, eps, h, &arena.x_norm_buf,
                        )?;
                        tcb.commit_and_wait()?;
                    }

                    // FFN via TCB (MoE) or CPU fallback (Dense): → arena.ffn_out_buf (f32).
                    let ffn_handled = self.ffn_tcb_inner(li)?;
                    if !ffn_handled {
                        let mut x_norm = vec![0.0f32; h];
                        self.decode_arena.as_ref().unwrap().read_x_norm(&mut x_norm);
                        let ffn_out = self.ffn(li, &x_norm)?;
                        self.decode_arena.as_ref().unwrap().write_ffn_out(&ffn_out);
                    }

                    // Cast FFN output f32→f16 delta.
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        crate::kernels::cast_f32_to_f16_tcb(
                            &mut tcb, &arena.ffn_out_buf, &arena.wedge_f_delta_f16, h,
                        )?;
                        tcb.commit_and_wait()?;
                    }
                }

                crate::metal::set_current_layer(None);

                // Final: add last layer's FFN delta + final norm.
                {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let final_norm_buf = self.final_norm_buf.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    if n_layers > 0 {
                        crate::kernels::add_inplace_f16_tcb(
                            &mut tcb, &arena.wedge_f_x_f16, &arena.wedge_f_delta_f16, h,
                        )?;
                    }
                    crate::kernels::rmsnorm_f16_to_f32_tcb(
                        &mut tcb, &arena.wedge_f_x_f16, final_norm_buf, eps, h, &arena.x_norm_buf,
                    )?;
                    tcb.commit_and_wait()?;
                    let mut x_norm = vec![0.0f32; h];
                    arena.read_x_norm(&mut x_norm);
                    return Ok(x_norm);
                }
            }
        }
        // ---- End Wedge F ----

        // ---- Wedge C: all attention + FFN kernels on TCB (zero counted dispatches).
        // Extends Wedge B by replacing attention()/ffn() CPU-round-trips with
        // attention_tcb_inner() / ffn_tcb_inner() that operate on arena buffers.
        // Active when: Metal + arena present, all norm weights pre-uploaded,
        // MLA path active, model_buf present, and all layers have q_a/kv_a pinned.
        #[cfg(target_os = "macos")]
        if !use_f16 {
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
                && self.layers.iter().all(|l| {
                    l.pinned.q_a_proj.is_some()
                        && l.pinned.q_b_proj.is_some()
                        && l.pinned.kv_a_proj_with_mqa.is_some()
                        && l.pinned.kv_b_proj.is_some()
                        && l.pinned.o_proj.is_some()
                        && l.pinned.q_a_norm.is_some()
                        && l.pinned.kv_a_norm.is_some()
                });

            if wedge_c_active {
                let eps = self.config.rms_norm_eps;
                let n_layers = self.config.n_layers;

                // Wedge D: embed lookup on GPU — writes x_buf directly, no CPU Vec round-trip.
                {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let embed_buf = self.embed_buf.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    crate::kernels::embed_lookup_metal_f32_tcb(
                        &mut tcb, embed_buf, token, h, &arena.x_buf,
                    )?;
                    tcb.commit_and_wait()?;
                }

                for li in 0..n_layers {
                    crate::metal::set_current_layer(Some(li as u32));

                    // Mini-TCB α: (add_inplace ffn_out_buf if li>0) + rmsnorm_attn → x_norm_buf.
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let attn_norm_buf = self.layers[li].pinned.attn_norm.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        if li > 0 {
                            crate::kernels::add_inplace_metal_tcb(
                                &mut tcb, &arena.x_buf, &arena.ffn_out_buf, h,
                            )?;
                        }
                        crate::kernels::rmsnorm_metal_buf_tcb(
                            &mut tcb, &arena.x_buf, attn_norm_buf, eps, h, &arena.x_norm_buf,
                        )?;
                        tcb.commit_and_wait()?;
                    }

                    // Attention via TCB: x_norm_buf → arena.out (all uncounted).
                    self.attention_tcb_inner(li, pos)?;

                    // Mini-TCB β: add_inplace(x_buf += arena.out) + rmsnorm_ffn → x_norm_buf.
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let ffn_norm_buf = self.layers[li].pinned.ffn_norm.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        crate::kernels::add_inplace_metal_tcb(
                            &mut tcb, &arena.x_buf, &arena.out, h,
                        )?;
                        crate::kernels::rmsnorm_metal_buf_tcb(
                            &mut tcb, &arena.x_buf, ffn_norm_buf, eps, h, &arena.x_norm_buf,
                        )?;
                        tcb.commit_and_wait()?;
                    }

                    // FFN via TCB (MoE) or CPU fallback (Dense): → arena.ffn_out_buf.
                    let ffn_handled = self.ffn_tcb_inner(li)?;
                    if !ffn_handled {
                        let mut x_norm = vec![0.0f32; h];
                        self.decode_arena.as_ref().unwrap().read_x_norm(&mut x_norm);
                        let ffn_out = self.ffn(li, &x_norm)?;
                        self.decode_arena.as_ref().unwrap().write_ffn_out(&ffn_out);
                    }
                }

                crate::metal::set_current_layer(None);

                // Final: add last layer's deferred ffn_out, read residual back.
                if n_layers > 0 {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    crate::kernels::add_inplace_metal_tcb(
                        &mut tcb, &arena.x_buf, &arena.ffn_out_buf, h,
                    )?;
                    tcb.commit_and_wait()?;
                }

                // Wedge D: final norm on GPU (x_buf → x_norm_buf), then read 8KB x_norm.
                {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let final_norm_buf = self.final_norm_buf.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    crate::kernels::rmsnorm_metal_buf_tcb(
                        &mut tcb, &arena.x_buf, final_norm_buf, eps, h, &arena.x_norm_buf,
                    )?;
                    tcb.commit_and_wait()?;
                    let mut x_norm = vec![0.0f32; h];
                    arena.read_x_norm(&mut x_norm);
                    return Ok(x_norm);
                }
            }
        }
        // ---- End Wedge C ----

        // ---- Wedge B: TCB-batched rmsnorm + GPU add_inplace via arena buffers.
        // Active when Metal + arena are present and all layer norm weights are
        // pre-uploaded. Skipped in f16-activation mode (bridge path unchanged).
        #[cfg(target_os = "macos")]
        if !use_f16 {
            let tcb_active = self.metal_ctx.is_some()
                && self.decode_arena.is_some()
                && self.layers.iter().all(|l| {
                    l.pinned.attn_norm.is_some() && l.pinned.ffn_norm.is_some()
                });

            if tcb_active {
                let eps = self.config.rms_norm_eps;
                let n_layers = self.config.n_layers;

                // Upload initial residual x to the arena GPU buffer.
                self.decode_arena.as_ref().unwrap().write_x(&x);

                for li in 0..n_layers {
                    crate::metal::set_current_layer(Some(li as u32));

                    // ---- Mini-TCB α: [add_inplace_ffn_prev?] + rmsnorm_attn ----
                    // For li > 0, ffn_out from the previous layer sits in ffn_out_buf.
                    // Batch: x_buf += ffn_out_buf (if li > 0), then rmsnorm(x_buf → x_norm_buf).
                    // All shared borrows released before attention().
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let attn_norm_buf =
                            self.layers[li].pinned.attn_norm.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        if li > 0 {
                            crate::kernels::add_inplace_metal_tcb(
                                &mut tcb,
                                &arena.x_buf,
                                &arena.ffn_out_buf,
                                h,
                            )?;
                        }
                        crate::kernels::rmsnorm_metal_buf_tcb(
                            &mut tcb,
                            &arena.x_buf,
                            attn_norm_buf,
                            eps,
                            h,
                            &arena.x_norm_buf,
                        )?;
                        tcb.commit_and_wait()?;
                    } // ctx, arena, attn_norm_buf borrows released here

                    // Read x_norm for attention.
                    let mut x_norm = vec![0.0f32; h];
                    self.decode_arena.as_ref().unwrap().read_x_norm(&mut x_norm);

                    let attn_out = self.attention(li, pos, &x_norm)?;

                    // Write attn_out into ffn_out_buf (delta role for add_inplace_attn).
                    self.decode_arena.as_ref().unwrap().write_ffn_out(&attn_out);

                    // ---- Mini-TCB β: add_inplace_attn + rmsnorm_ffn ----
                    // x_buf += attn_out (from ffn_out_buf), then rmsnorm(x_buf → x_norm_buf).
                    {
                        let ctx = self.metal_ctx.as_ref().unwrap();
                        let arena = self.decode_arena.as_ref().unwrap();
                        let ffn_norm_buf =
                            self.layers[li].pinned.ffn_norm.as_ref().unwrap();
                        let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                        crate::kernels::add_inplace_metal_tcb(
                            &mut tcb,
                            &arena.x_buf,
                            &arena.ffn_out_buf,
                            h,
                        )?;
                        crate::kernels::rmsnorm_metal_buf_tcb(
                            &mut tcb,
                            &arena.x_buf,
                            ffn_norm_buf,
                            eps,
                            h,
                            &arena.x_norm_buf,
                        )?;
                        tcb.commit_and_wait()?;
                    } // borrows released

                    // Read x_norm for FFN.
                    let mut x_norm = vec![0.0f32; h];
                    self.decode_arena.as_ref().unwrap().read_x_norm(&mut x_norm);

                    let ffn_out = self.ffn(li, &x_norm)?;

                    // Write ffn_out into ffn_out_buf; add_inplace is deferred to
                    // the next iteration's mini-TCB α (or the post-loop commit).
                    self.decode_arena.as_ref().unwrap().write_ffn_out(&ffn_out);
                }

                crate::metal::set_current_layer(None);

                // Final: add_inplace for the last layer's deferred ffn_out.
                if n_layers > 0 {
                    let ctx = self.metal_ctx.as_ref().unwrap();
                    let arena = self.decode_arena.as_ref().unwrap();
                    let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
                    crate::kernels::add_inplace_metal_tcb(
                        &mut tcb,
                        &arena.x_buf,
                        &arena.ffn_out_buf,
                        h,
                    )?;
                    tcb.commit_and_wait()?;
                    arena.read_x(&mut x);
                }

                // Final norm (CPU dispatch path for now).
                let mut x_norm = vec![0.0f32; h];
                self.rmsnorm_dispatch(&x, &self.final_norm, self.config.rms_norm_eps, &mut x_norm)?;
                return Ok(x_norm);
            }
        }
        // ---- End Wedge B ----

        // Original path (CPU or GPU without TCB batching).
        for li in 0..self.config.n_layers {
            crate::metal::set_current_layer(Some(li as u32));

            // ---- Attention block ----
            // Phase 7: when F16 + arena active, write pre-norm residual as
            // f16 so bridge kernels can read it inside attention().
            #[cfg(target_os = "macos")]
            if use_f16 {
                if let Some(arena) = self.decode_arena.as_ref() {
                    arena.write_x_f16(&x);
                }
            }

            let mut x_norm = vec![0.0f32; h];
            self.rmsnorm_dispatch(
                &x,
                &self.layers[li].attn_norm,
                self.config.rms_norm_eps,
                &mut x_norm,
            )?;

            let attn_out = self.attention(li, pos, &x_norm)?;
            add_inplace(&mut x, &attn_out);

            // ---- FFN block ----
            // Phase 7: write updated residual (after attn add) as f16 for
            // the FFN bridge kernels inside ffn().
            #[cfg(target_os = "macos")]
            if use_f16 {
                if let Some(arena) = self.decode_arena.as_ref() {
                    arena.write_x_f16(&x);
                }
            }

            self.rmsnorm_dispatch(
                &x.clone(),
                &self.layers[li].ffn_norm,
                self.config.rms_norm_eps,
                &mut x_norm,
            )?;
            let ffn_out = self.ffn(li, &x_norm)?;
            add_inplace(&mut x, &ffn_out);
        }
        crate::metal::set_current_layer(None); // final norm + LM head dispatches

        // Final norm + lm head.
        let mut x_norm = vec![0.0f32; h];
        self.rmsnorm_dispatch(&x, &self.final_norm, self.config.rms_norm_eps, &mut x_norm)?;
        Ok(x_norm)
    }

    fn profiled_greedy_enabled(&self, sampling: &crate::engine::SamplingParams) -> bool {
        sampling.temperature <= 0.0
            && sampling.repetition_penalty == 1.0
            && self
                .kernel_profile
                .as_ref()
                .map(|p| p.selected.lm_head_schedule.contains("argmax"))
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

    /// v1.0.0-C: MLA attention via TCB (zero counted dispatches).
    /// Reads arena.x_norm_buf, writes result to arena.out.
    /// Three mini-TCB commits (all uncounted):
    ///   1. q_a/kv_a GEMVs + q_a_norm + kv_a_norm
    ///   2. q_b_proj GEMV
    ///   3. mla_decode + o_proj
    #[cfg(target_os = "macos")]
    fn attention_tcb_inner(&mut self, li: usize, pos: usize) -> Result<()> {
        let n_heads = self.config.n_heads;
        let head_dim_q = self.config.qk_nope_head_dim + self.config.qk_rope_head_dim;
        let kv_a_dim = self.config.kv_lora_rank + self.config.qk_rope_head_dim;
        let q_lora = self.config.q_lora_rank.max(1);
        let kv_lora_rank = self.config.kv_lora_rank;
        let qk_rope_head_dim = self.config.qk_rope_head_dim;
        let qk_nope_head_dim = self.config.qk_nope_head_dim;
        let rope_theta = self.config.rope_theta;
        let eps = self.config.rms_norm_eps;
        let h = self.config.hidden;
        let n_layers = self.config.n_layers;

        // Phase 1 mini-TCB: q_a_proj + kv_a_proj GEMVs, then q_a_norm + kv_a_norm.
        // Sequential kernels within one TCB: each reads output of the prior.
        {
            let ctx = self.metal_ctx.as_ref().unwrap();
            let arena = self.decode_arena.as_ref().unwrap();
            let q_a_proj_buf = self.layers[li].pinned.q_a_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} q_a_proj not pinned")))?;
            let kv_a_proj_buf = self.layers[li].pinned.kv_a_proj_with_mqa.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} kv_a_proj not pinned")))?;
            let q_a_norm_buf = self.layers[li].pinned.q_a_norm.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} q_a_norm not pinned")))?;
            let kv_a_norm_buf = self.layers[li].pinned.kv_a_norm.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} kv_a_norm not pinned")))?;
            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
            crate::kernels::gemv_f32_attn_pair_arena_tcb(
                &mut tcb,
                q_a_proj_buf, q_lora,
                kv_a_proj_buf, kv_a_dim,
                h, &arena.x_norm_buf,
                &arena.q_lora_buf, &arena.kv_a_out_buf,
            )?;
            crate::kernels::rmsnorm_metal_buf_tcb(
                &mut tcb, &arena.q_lora_buf, q_a_norm_buf, eps, q_lora, &arena.q_lora_normed_buf,
            )?;
            // kv_a_norm: normalize first kv_lora_rank elements of kv_a_out_buf.
            crate::kernels::rmsnorm_metal_buf_tcb(
                &mut tcb, &arena.kv_a_out_buf, kv_a_norm_buf, eps, kv_lora_rank, &arena.c_kv_normed_buf,
            )?;
            tcb.commit_and_wait()?;
        }

        // CPU: rope k_pe from kv_a_out_buf[kv_lora_rank..], mla_kv_append.
        let c_kv_normed: Vec<f32> = {
            let arena = self.decode_arena.as_ref().unwrap();
            let ptr = arena.c_kv_normed_buf.contents() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, kv_lora_rank) }.to_vec()
        };
        let mut k_pe: Vec<f32> = {
            let arena = self.decode_arena.as_ref().unwrap();
            let ptr = arena.kv_a_out_buf.contents() as *const f32;
            let slice = unsafe { std::slice::from_raw_parts(ptr, kv_a_dim) };
            slice[kv_lora_rank..].to_vec()
        };
        rope_inplace(&mut k_pe, pos as u32, rope_theta);

        if li == 0 && self.kv.seq_len >= self.kv.max_seq {
            return Err(Error::Model("kv cache full".into()));
        }
        let seq_slot = self.kv.seq_len;
        Self::mla_kv_append(
            &mut self.mla_c_kv, &mut self.mla_k_pe,
            li, seq_slot, kv_lora_rank, qk_rope_head_dim,
            &c_kv_normed, &k_pe,
        )?;
        if li + 1 == n_layers {
            self.kv.seq_len += 1;
        }
        let seq_len = self.kv.seq_len.max(1);

        // Update arena c_kv and k_pe with the full-sequence slices.
        {
            let arena = self.decode_arena.as_ref().unwrap();
            crate::metal::MetalContext::write_buffer_bytes(
                &arena.c_kv,
                bytemuck::cast_slice(&self.mla_c_kv[li][..seq_len * kv_lora_rank]),
            );
            crate::metal::MetalContext::write_buffer_bytes(
                &arena.k_pe,
                bytemuck::cast_slice(&self.mla_k_pe[li][..seq_len * qk_rope_head_dim]),
            );
        }

        // Phase 2 mini-TCB: q_b_proj GEMV (q_lora_normed_buf → arena.q).
        {
            let ctx = self.metal_ctx.as_ref().unwrap();
            let arena = self.decode_arena.as_ref().unwrap();
            let q_b_proj_buf = self.layers[li].pinned.q_b_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} q_b_proj not pinned")))?;
            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
            crate::kernels::gemv_f32_attn_pinned_buf_tcb(
                &mut tcb, q_b_proj_buf, n_heads * head_dim_q, q_lora,
                &arena.q_lora_normed_buf, &arena.q,
            )?;
            tcb.commit_and_wait()?;
        }

        // CPU: rope Q heads in-place, write back to arena.q.
        let mut q_full = vec![0.0f32; n_heads * head_dim_q];
        {
            let arena = self.decode_arena.as_ref().unwrap();
            let ptr = arena.q.contents() as *const f32;
            let src = unsafe { std::slice::from_raw_parts(ptr, n_heads * head_dim_q) };
            q_full.copy_from_slice(src);
        }
        for h_i in 0..n_heads {
            let off = h_i * head_dim_q + qk_nope_head_dim;
            rope_inplace(&mut q_full[off..off + qk_rope_head_dim], pos as u32, rope_theta);
        }
        self.decode_arena.as_ref().unwrap().write_q(&q_full);

        // Phase 3 mini-TCB: mla_decode + o_proj (→ arena.out).
        {
            let ctx = self.metal_ctx.as_ref().unwrap();
            let arena = self.decode_arena.as_ref().unwrap();
            let kv_b_proj_buf = self.layers[li].pinned.kv_b_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} kv_b_proj not pinned")))?;
            let o_proj_buf = self.layers[li].pinned.o_proj.as_ref()
                .ok_or_else(|| Error::Model(format!("attention_tcb: l{li} o_proj not pinned")))?;
            let scale = 1.0f32 / (head_dim_q as f32).sqrt();
            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
            crate::kernels::mla_decode_and_o_proj_arena_tcb(
                &mut tcb, arena, kv_b_proj_buf, o_proj_buf,
                n_heads, qk_nope_head_dim, qk_rope_head_dim, self.config.v_head_dim,
                kv_lora_rank, seq_len, scale, h,
            )?;
            tcb.commit_and_wait()?;
        }

        Ok(())
    }

    /// v1.0.0-C: FFN via TCB (zero counted dispatches) for MoE layers.
    /// Reads arena.x_norm_buf, writes result to arena.ffn_out_buf.
    /// Returns true if handled, false to signal Dense-layer fallback.
    #[cfg(target_os = "macos")]
    fn ffn_tcb_inner(&self, li: usize) -> Result<bool> {
        use crate::gguf::GgmlType;

        let ctx = match self.metal_ctx.as_ref() {
            Some(c) => c,
            None => return Ok(false),
        };
        let arena = match self.decode_arena.as_ref() {
            Some(a) => a,
            None => return Ok(false),
        };
        let model_buf = match self.weights_mmap_buf.as_ref() {
            Some(b) => b,
            None => return Ok(false),
        };

        let (routed_gate_off, routed_up_off, routed_down_off, routed_len,
             shared_gate_off, shared_up_off, shared_down_off, shared_mid) = {
            let layer = &self.layers[li];
            match &layer.mode {
                LayerMode::MoE { routed_fused, routed, shared_fused, .. } => {
                    if routed_fused.gate_w.dtype != GgmlType::Q4_K
                        || routed_fused.up_w.dtype != GgmlType::Q4_K
                        || routed_fused.down_w.dtype != GgmlType::Q8_0
                    {
                        return Ok(false);
                    }
                    let (sg, su, sd, smid) = if let Some(sf) = shared_fused {
                        if sf.gate_w.dtype != GgmlType::Q4_K
                            || sf.up_w.dtype != GgmlType::Q4_K
                            || sf.down_w.dtype != GgmlType::Q6_K
                        {
                            return Ok(false);
                        }
                        let smid = self.config.n_shared_experts * self.config.moe_intermediate;
                        (Some(sf.gate_w.offset), Some(sf.up_w.offset), Some(sf.down_w.offset), smid)
                    } else {
                        (None, None, None, 0usize)
                    };
                    (
                        routed_fused.gate_w.offset, routed_fused.up_w.offset, routed_fused.down_w.offset,
                        routed.len(), sg, su, sd, smid,
                    )
                }
                LayerMode::Dense { .. } => return Ok(false),
            }
        };

        // Gate logit GEMV in mini-TCB (uncounted): x_norm_buf → moe_logits_buf.
        {
            let gate_buf = match self.layers[li].pinned.gate_logits_w.as_ref() {
                Some(b) => b,
                None => return Ok(false),
            };
            let mut tcb = crate::metal::TokenCommandBuffer::new(ctx);
            crate::kernels::gemv_f32_moe_pinned_buf_tcb(
                &mut tcb, gate_buf,
                self.config.n_routed_experts, self.config.hidden,
                &arena.x_norm_buf, &arena.moe_logits_buf,
            )?;
            tcb.commit_and_wait()?;
        }

        // CPU: read logits, topk_gate.
        let mut logits = vec![0.0f32; self.config.n_routed_experts];
        arena.read_moe_logits(&mut logits);
        let routes = topk_gate(&mut logits, self.config.top_k_routed, true);

        if routes.is_empty() {
            return Ok(false);
        }

        let route_ids: Vec<u32> = routes.iter().map(|&(eid, _)| eid as u32).collect();
        let route_weights: Vec<f32> = routes.iter().map(|&(_, w)| w).collect();
        let q4k_schedule = self.kernel_profile.as_ref()
            .map(|p| p.selected.gemm_q4_k_schedule.as_str())
            .unwrap_or("scalar");

        crate::kernels::moe_block_batched_indexed_tcb(
            ctx, model_buf,
            routed_gate_off, routed_up_off, routed_down_off,
            routed_len, &route_ids, &route_weights,
            shared_gate_off, shared_up_off, shared_down_off,
            self.config.hidden, self.config.moe_intermediate, shared_mid,
            q4k_schedule,
            &arena.x_norm_buf, &arena.ffn_out_buf,
        )?;

        Ok(true)
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

            // Phase 7 F16 bridge: when activation_dtype=F16 + arena active +
            // q_a_proj pinned, use rmsnorm_gemv_f16_attn_pinned to fuse
            // attn_norm + q_a_proj GEMV reading from the f16 residual.
            #[cfg(target_os = "macos")]
            let f16_bridged = {
                if self.activation_dtype == crate::engine::ActivationDtype::F16 {
                    if let (Some(ctx), Some(arena), Some(pinned_qa)) = (
                        self.metal_ctx.as_ref(),
                        self.decode_arena.as_ref(),
                        layer.pinned.q_a_proj.as_ref(),
                    ) {
                        let attn_norm_bytes =
                            bytemuck::cast_slice::<f32, u8>(&layer.attn_norm);
                        let attn_norm_buf = ctx.new_buffer_with_bytes(attn_norm_bytes);
                        let out_buf = ctx.new_buffer(q_lora * std::mem::size_of::<f32>());
                        crate::kernels::rmsnorm_gemv_f16_attn_pinned_metal(
                            ctx,
                            pinned_qa,
                            &arena.x_f16_buf,
                            &attn_norm_buf,
                            cfg.rms_norm_eps,
                            &out_buf,
                            q_lora,
                            h,
                        )?;
                        let ptr = out_buf.contents() as *const f32;
                        t.copy_from_slice(unsafe {
                            std::slice::from_raw_parts(ptr, q_lora)
                        });
                        // kv_a_proj uses the already-normed f32 x.
                        self.gemv_f32_attn_dispatch(
                            &layer.kv_a_proj_with_mqa,
                            layer.pinned.kv_a_proj_with_mqa.as_ref(),
                            kv_a_dim,
                            h,
                            x,
                            &mut kv_a,
                        )?;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            };
            #[cfg(not(target_os = "macos"))]
            let f16_bridged = false;

            if !f16_bridged {
                // Existing f32 path: q_a_proj and kv_a_proj share input x.
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
            }

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

                // Wedge 2: two-stage fused path (profile.selected.moe_schedule == "two-stage").
                if let Some(two_stage) = self.moe_block_two_stage_dispatch(
                    routed_fused,
                    shared_fused.as_ref(),
                    &routes,
                    x,
                )? {
                    out = two_stage;
                    return Ok(out);
                }

                // Single-kernel fused path (profile.selected.moe_schedule == "single-kernel").
                if let Some(fused) = self.moe_block_fused_v2lite_dispatch(
                    routed_fused,
                    shared_fused.as_ref(),
                    &routes,
                    x,
                )? {
                    out = fused;
                    return Ok(out);
                }

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

                    // Phase 7 F16 bridge: when activation_dtype=F16 + arena +
                    // Q4K shared expert, use rmsnorm_gemv_q4k_pair_f16 to
                    // fuse ffn_norm + gate+up GEMVs reading the f16 residual.
                    #[cfg(target_os = "macos")]
                    let f16_shared_bridged = if self.activation_dtype
                        == crate::engine::ActivationDtype::F16
                        && s.gate_w.dtype == crate::gguf::GgmlType::Q4_K
                        && s.up_w.dtype == crate::gguf::GgmlType::Q4_K
                    {
                        if let (Some(ctx), Some(arena)) =
                            (self.metal_ctx.as_ref(), self.decode_arena.as_ref())
                        {
                            let ffn_norm_f16: Vec<half::f16> = self.layers[li]
                                .ffn_norm
                                .iter()
                                .map(|&v| half::f16::from_f32(v))
                                .collect();
                            let gate_bytes = &self.gguf.mmap
                                [s.gate_w.offset..s.gate_w.offset + s.gate_w.byte_size];
                            let up_bytes = &self.gguf.mmap
                                [s.up_w.offset..s.up_w.offset + s.up_w.byte_size];
                            let gate_out_buf =
                                ctx.new_buffer(smid * std::mem::size_of::<f32>());
                            let up_out_buf =
                                ctx.new_buffer(smid * std::mem::size_of::<f32>());
                            crate::kernels::rmsnorm_gemv_q4k_pair_f16_metal(
                                ctx,
                                &ffn_norm_f16,
                                cfg.rms_norm_eps,
                                gate_bytes,
                                up_bytes,
                                &gate_out_buf,
                                &up_out_buf,
                                &arena.x_f16_buf,
                                smid,
                                cfg.hidden,
                            )?;
                            let g: Vec<f32> = {
                                let ptr = gate_out_buf.contents() as *const f32;
                                unsafe { std::slice::from_raw_parts(ptr, smid) }.to_vec()
                            };
                            let u: Vec<f32> = {
                                let ptr = up_out_buf.contents() as *const f32;
                                unsafe { std::slice::from_raw_parts(ptr, smid) }.to_vec()
                            };
                            crate::kernels::silu_mul(&g, &u, &mut sa);
                            true
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    #[cfg(not(target_os = "macos"))]
                    let f16_shared_bridged = false;

                    if !f16_shared_bridged {
                        self.moe_expert_pair_matmul_dispatch(
                            &s.gate_w, &s.up_w, smid, cfg.hidden, x,
                            &mut sa, &mut w_buf,
                        )?;
                    }

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
