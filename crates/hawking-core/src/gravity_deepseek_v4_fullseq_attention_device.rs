//! Multi-token / non-BOS growing-KV attention for DeepSeek-V4 base layers.
//!
//! Extends the BOS window-KV specialization to a real causal sequence:
//! - ratio-0 layers (0, 1): full growing-KV sparse attention (window 128)
//! - ratio-4 / ratio-128: admitted only while compressed topk is still empty
//!   (`(position+1) // ratio == 0`); refuses once the indexer/compressor would
//!   activate. That is the exact source empty-compressed specialization, not a
//!   fake full compressed graph.
//!
//! Input modes:
//! - Layer 0: seed from an embedding row (BF16[4096])
//! - Layers 1..42: continue from a predecessor child HC (BF16[4,4096])
//!
//! Honesty: NumericParityV21Only; no Engine, HCLI, serve, TPS, or exact-storage
//! multi-layer parity claim.

use std::mem::size_of;

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::gravity_deepseek_v4_bos_layer_attention_device::expected_bos_compress_ratio;
use crate::gravity_deepseek_v4_layer0_continuation::{
    yarn_rope_table_for_position, WINDOW_SIZE,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    EMBED_WEIGHT, HC_EPS, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE, RMS_NORM_EPS,
    VOCAB_SIZE,
};
use crate::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
use crate::metal::{CommandBatch, MetalBatchTiming, MetalContext};
use crate::{Error, Result};

const HC_FLAT_WIDTH: usize = HC_MULT * HIDDEN_SIZE;
const ACT_QUANT_BLOCK: usize = 128;
const WQ_A_ROWS: usize = 1024;
const WQ_A_COLS: usize = HIDDEN_SIZE;
const Q_LORA_RANK: usize = 1024;
const NUM_HEADS: usize = 64;
const HEAD_DIM: usize = 512;
const ROPE_HEAD_DIM: usize = 64;
const NON_ROPE_HEAD_DIM: usize = HEAD_DIM - ROPE_HEAD_DIM;
const KV_QAT_BLOCK: usize = 64;
const O_LORA_RANK: usize = 1024;
const WQ_B_ROWS: usize = NUM_HEADS * HEAD_DIM;
const WKV_ROWS: usize = HEAD_DIM;
const WO_A_ROWS: usize = 8 * O_LORA_RANK;
const WO_A_COLS: usize = NUM_HEADS * HEAD_DIM / 8;
const WO_B_ROWS: usize = HIDDEN_SIZE;
const WO_B_COLS: usize = WO_A_ROWS;

/// Position-0 (no rope kernels): same ordered graph as BOS layer attention.
pub const DSV4F_FULLSEQ_ATTENTION_DISPATCHES_POS0: usize = 22;
/// Position >0 adds Q rope, KV rope, and inverse rope on sparse output.
pub const DSV4F_FULLSEQ_ATTENTION_DISPATCHES_POS_N: usize = 25;
pub const DSV4F_FULLSEQ_HC_STATE_BF16_BYTES: usize = HC_FLAT_WIDTH * size_of::<u16>();
pub const DSV4F_FULLSEQ_KV_ROW_BF16_BYTES: usize = HEAD_DIM * size_of::<u16>();
pub const DSV4F_FULLSEQ_KV_CAPACITY: usize = WINDOW_SIZE;

const HC_PRE_PRED_KERNEL: &str = "deepseek_v4_p7_mhc_ffn_pre_authority";
const HC_PRE_EMBED_KERNEL: &str = "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority";
const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
const CACHE_KERNEL: &str = "deepseek_v4_p4b_kv_cache_write_bf16_authority";
const ROPE_KERNEL: &str = "deepseek_v4_p4b_rope_position1_bf16_authority";
const SPARSE_KERNEL: &str = "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority";
const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
const HC_POST_PRED_KERNEL: &str = "deepseek_v4_p7_mhc_ffn_post_authority";
const HC_POST_EMBED_KERNEL: &str = "deepseek_v4_p4a_hc_attn_post_authority";

/// How the step seeds its residual / mHC-pre input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4FullseqInputKind {
    /// Layer 0: BF16[4096] embedding row.
    Embedding,
    /// Layers 1..42: BF16[4,4096] predecessor child HC.
    PredecessorHc,
}

/// Per-layer growing KV cache owned by the caller across positions.
pub struct DeepSeekV4FullseqLayerKvCache {
    pub layer: usize,
    pub capacity: usize,
    pub filled: usize,
    pub buffer: metal::Buffer,
}

impl DeepSeekV4FullseqLayerKvCache {
    pub fn new(metal: &MetalContext, layer: usize, capacity: usize) -> Result<Self> {
        if capacity == 0 || capacity > DSV4F_FULLSEQ_KV_CAPACITY {
            return Err(fullseq_error(format!(
                "KV capacity {capacity} must be in 1..={DSV4F_FULLSEQ_KV_CAPACITY}"
            )));
        }
        Ok(Self {
            layer,
            capacity,
            filled: 0,
            buffer: metal
                .new_buffer_checked(capacity * DSV4F_FULLSEQ_KV_ROW_BF16_BYTES)?,
        })
    }

    pub fn reset(&mut self) {
        self.filled = 0;
    }
}

/// Output of one fullseq attention step (device-only; no host readback API).
pub struct DeepSeekV4FullseqAttentionDeviceOutput {
    pub attention_hc_state_bf16: metal::Buffer,
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
    pub kv_rows: usize,
    pub compress_ratio: usize,
    pub empty_compressed: bool,
    pub actual_gpu_dispatches: usize,
    pub actual_command_buffers: usize,
    pub actual_cpu_visible_waits: usize,
    pub timing: MetalBatchTiming,
    pub sparse_attention_kernel: &'static str,
}

impl DeepSeekV4FullseqAttentionDeviceOutput {
    #[cfg(target_os = "macos")]
    pub fn p7_attention_state<'a>(
        &'a self,
        metal: &'a MetalContext,
        kv_cache: &'a DeepSeekV4FullseqLayerKvCache,
    ) -> Result<crate::gravity_deepseek_v4_p7_composition::DeepSeekV4P7AttentionDeviceState<'a>>
    {
        crate::gravity_deepseek_v4_p7_composition::DeepSeekV4P7AttentionDeviceState::fullseq(
            metal,
            &self.attention_hc_state_bf16,
            &kv_cache.buffer,
            self.layer,
            self.token_id,
            self.token_position,
            self.kv_rows,
        )
    }
}

struct VerifiedTensor {
    name: String,
    bytes: Vec<u8>,
}

struct Fp8Pair {
    weight: VerifiedTensor,
    scale: VerifiedTensor,
}

struct LayerControls {
    hc_fn: VerifiedTensor,
    hc_base: VerifiedTensor,
    hc_scale: VerifiedTensor,
    attn_norm: VerifiedTensor,
    q_norm: VerifiedTensor,
    kv_norm: VerifiedTensor,
    attn_sink: VerifiedTensor,
    wq_a: Fp8Pair,
    wq_b: Fp8Pair,
    wkv: Fp8Pair,
    wo_a: Fp8Pair,
    wo_b: Fp8Pair,
}

struct HcScratch {
    reduced: metal::Buffer,
    rsqrt: metal::Buffer,
    mixes: metal::Buffer,
    pre: metal::Buffer,
    post: metal::Buffer,
    comb: metal::Buffer,
}

struct LinearScratch {
    activation: metal::Buffer,
    scales: metal::Buffer,
    fp32: metal::Buffer,
    bf16: metal::Buffer,
}

struct KvScratch {
    norm: metal::Buffer,
    qat: metal::Buffer,
    activation: metal::Buffer,
    scales: metal::Buffer,
}

/// Prepared static controls + scratch for one layer; reusable across positions.
pub struct DeepSeekV4FullseqAttentionDeviceExecutor {
    layer: usize,
    compress_ratio: usize,
    input_kind: DeepSeekV4FullseqInputKind,
    context_queue_identity: usize,
    hc_fn: metal::Buffer,
    hc_base: metal::Buffer,
    hc_scale: metal::Buffer,
    attn_norm_weight: metal::Buffer,
    q_norm_weight: metal::Buffer,
    kv_norm_weight: metal::Buffer,
    attn_sink: metal::Buffer,
    wq_a_weight: metal::Buffer,
    wq_a_scale: metal::Buffer,
    wq_b_weight: metal::Buffer,
    wq_b_scale: metal::Buffer,
    wkv_weight: metal::Buffer,
    wkv_scale: metal::Buffer,
    wo_a_weight: metal::Buffer,
    wo_a_scale: metal::Buffer,
    wo_b_weight: metal::Buffer,
    wo_b_scale: metal::Buffer,
    hc: HcScratch,
    attn_norm: metal::Buffer,
    wq_a: LinearScratch,
    q_norm: metal::Buffer,
    wq_b: LinearScratch,
    q_head: metal::Buffer,
    q_rope: metal::Buffer,
    wkv: LinearScratch,
    kv: KvScratch,
    kv_rope: metal::Buffer,
    sparse: metal::Buffer,
    derotated: metal::Buffer,
    scores: metal::Buffer,
    denominators: metal::Buffer,
    wo_a: metal::Buffer,
    wo_b: LinearScratch,
    attention_hc_state_bf16: metal::Buffer,
    rope_cos: metal::Buffer,
    rope_sin: metal::Buffer,
}

impl DeepSeekV4FullseqAttentionDeviceExecutor {
    /// Prepare static `layers.{N}.*` attention controls for multi-token steps.
    pub fn prepare(
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        layer: usize,
    ) -> Result<Self> {
        if layer >= 43 {
            return Err(fullseq_error("fullseq attention targets base layers 0..42"));
        }
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(reader)?;
        // Position 0 always admits; deeper positions checked at execute.
        catalog
            .plan(layer)?
            .require_empty_compressed_growing_kv_attention(0)?;
        validate_required_pipelines(metal)?;
        verify_source_identity(reader)?;
        let compress_ratio = expected_bos_compress_ratio(layer);
        let input_kind = if layer == 0 {
            DeepSeekV4FullseqInputKind::Embedding
        } else {
            DeepSeekV4FullseqInputKind::PredecessorHc
        };
        let controls = LayerControls::load(reader, layer)?;
        Ok(Self {
            layer,
            compress_ratio,
            input_kind,
            context_queue_identity: context_queue_identity(metal),
            hc_fn: metal.new_buffer_with_bytes_checked(&controls.hc_fn.bytes)?,
            hc_base: metal.new_buffer_with_bytes_checked(&controls.hc_base.bytes)?,
            hc_scale: metal.new_buffer_with_bytes_checked(&controls.hc_scale.bytes)?,
            attn_norm_weight: metal.new_buffer_with_bytes_checked(&controls.attn_norm.bytes)?,
            q_norm_weight: metal.new_buffer_with_bytes_checked(&controls.q_norm.bytes)?,
            kv_norm_weight: metal.new_buffer_with_bytes_checked(&controls.kv_norm.bytes)?,
            attn_sink: metal.new_buffer_with_bytes_checked(&controls.attn_sink.bytes)?,
            wq_a_weight: metal.new_buffer_with_bytes_checked(&controls.wq_a.weight.bytes)?,
            wq_a_scale: metal.new_buffer_with_bytes_checked(&controls.wq_a.scale.bytes)?,
            wq_b_weight: metal.new_buffer_with_bytes_checked(&controls.wq_b.weight.bytes)?,
            wq_b_scale: metal.new_buffer_with_bytes_checked(&controls.wq_b.scale.bytes)?,
            wkv_weight: metal.new_buffer_with_bytes_checked(&controls.wkv.weight.bytes)?,
            wkv_scale: metal.new_buffer_with_bytes_checked(&controls.wkv.scale.bytes)?,
            wo_a_weight: metal.new_buffer_with_bytes_checked(&controls.wo_a.weight.bytes)?,
            wo_a_scale: metal.new_buffer_with_bytes_checked(&controls.wo_a.scale.bytes)?,
            wo_b_weight: metal.new_buffer_with_bytes_checked(&controls.wo_b.weight.bytes)?,
            wo_b_scale: metal.new_buffer_with_bytes_checked(&controls.wo_b.scale.bytes)?,
            hc: new_hc_scratch(metal)?,
            attn_norm: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<u16>())?,
            wq_a: new_linear_scratch(metal, WQ_A_COLS, WQ_A_ROWS)?,
            q_norm: metal.new_buffer_checked(Q_LORA_RANK * size_of::<u16>())?,
            wq_b: new_linear_scratch(metal, Q_LORA_RANK, WQ_B_ROWS)?,
            q_head: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            q_rope: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            wkv: new_linear_scratch(metal, HIDDEN_SIZE, WKV_ROWS)?,
            kv: new_kv_scratch(metal)?,
            kv_rope: metal.new_buffer_checked(HEAD_DIM * size_of::<u16>())?,
            sparse: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            derotated: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            scores: metal
                .new_buffer_checked(NUM_HEADS * DSV4F_FULLSEQ_KV_CAPACITY * size_of::<f32>())?,
            denominators: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
            wo_a: metal.new_buffer_checked(WO_A_ROWS * size_of::<u16>())?,
            wo_b: new_linear_scratch(metal, WO_B_COLS, WO_B_ROWS)?,
            attention_hc_state_bf16: metal
                .new_buffer_checked(DSV4F_FULLSEQ_HC_STATE_BF16_BYTES)?,
            rope_cos: metal.new_buffer_checked((ROPE_HEAD_DIM / 2) * size_of::<f32>())?,
            rope_sin: metal.new_buffer_checked((ROPE_HEAD_DIM / 2) * size_of::<f32>())?,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn compress_ratio(&self) -> usize {
        self.compress_ratio
    }

    pub fn input_kind(&self) -> DeepSeekV4FullseqInputKind {
        self.input_kind
    }

    /// Execute one decode position. For layer 0, `input` is the embed row buffer
    /// (BF16[4096]); for layers ≥1 it is the predecessor child HC (BF16[4,4096]).
    pub fn execute_position(
        &mut self,
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        input: &metal::Buffer,
        token_id: u32,
        token_position: usize,
        kv_cache: &mut DeepSeekV4FullseqLayerKvCache,
    ) -> Result<DeepSeekV4FullseqAttentionDeviceOutput> {
        if context_queue_identity(metal) != self.context_queue_identity {
            return Err(fullseq_error(
                "fullseq attention requires its preparation MetalContext/queue",
            ));
        }
        if kv_cache.layer != self.layer {
            return Err(fullseq_error("KV cache layer does not match executor layer"));
        }
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(reader)?;
        catalog
            .plan(self.layer)?
            .require_empty_compressed_growing_kv_attention(token_position)?;
        if token_position != kv_cache.filled {
            return Err(fullseq_error(format!(
                "KV cache filled={} but token_position={token_position}; must execute positions in order",
                kv_cache.filled
            )));
        }
        if token_position >= kv_cache.capacity {
            return Err(fullseq_error("token position exceeds KV cache capacity"));
        }
        let valid_kv = token_position + 1;
        match self.input_kind {
            DeepSeekV4FullseqInputKind::Embedding => {
                if input.length() != (HIDDEN_SIZE * size_of::<u16>()) as u64 {
                    return Err(fullseq_error("layer-0 fullseq input must be BF16[4096] embed"));
                }
            }
            DeepSeekV4FullseqInputKind::PredecessorHc => {
                if input.length() != DSV4F_FULLSEQ_HC_STATE_BF16_BYTES as u64 {
                    return Err(fullseq_error(
                        "layer≥1 fullseq input must be BF16[4,4096] predecessor HC",
                    ));
                }
            }
        }
        // Upload YaRN table for this position (static control, not activation).
        let rope = yarn_rope_table_for_position(reader, token_position)?;
        upload_f32(metal, &self.rope_cos, &rope.cos_f32)?;
        upload_f32(metal, &self.rope_sin, &rope.sin_f32)?;

        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        let mix_width = HC_MIX_WIDTH as u32;
        let sinkhorn = HC_SINKHORN_ITERS as u32;
        let heads = NUM_HEADS as u32;
        let head_dim = HEAD_DIM as u32;
        let q_lora = Q_LORA_RANK as u32;
        let rope_dim = ROPE_HEAD_DIM as u32;
        let kv_block = KV_QAT_BLOCK as u32;
        let sparse_scale = (HEAD_DIM as f32).powf(-0.5);
        let cache_capacity = kv_cache.capacity as u32;
        let cache_position = token_position as u32;
        let valid_kv_u32 = valid_kv as u32;
        let max_score_slots = kv_cache.capacity as u32;
        let apply_rope = token_position > 0;
        let expected_dispatches = if apply_rope {
            DSV4F_FULLSEQ_ATTENTION_DISPATCHES_POS_N
        } else {
            DSV4F_FULLSEQ_ATTENTION_DISPATCHES_POS0
        };

        let timing = metal.dispatch_batch_timed(|batch| {
            // --- mHC pre ---
            match self.input_kind {
                DeepSeekV4FullseqInputKind::Embedding => {
                    dispatch_hc_pre_embed(
                        batch,
                        input,
                        &self.hc_fn,
                        &self.hc_scale,
                        &self.hc_base,
                        &self.hc,
                        hidden,
                        hc_mult,
                        mix_width,
                        sinkhorn,
                    )?;
                }
                DeepSeekV4FullseqInputKind::PredecessorHc => {
                    dispatch_hc_pre_pred(
                        batch,
                        input,
                        &self.hc_fn,
                        &self.hc_scale,
                        &self.hc_base,
                        &self.hc,
                        hidden,
                        hc_mult,
                        mix_width,
                        sinkhorn,
                    )?;
                }
            }
            dispatch_rms(
                batch,
                &self.hc.reduced,
                &self.attn_norm_weight,
                &self.attn_norm,
                hidden,
            )?;
            // WQ-A
            dispatch_qat(
                batch,
                &self.attn_norm,
                &self.wq_a.activation,
                &self.wq_a.scales,
                WQ_A_COLS as u32,
            )?;
            dispatch_fp8(
                batch,
                &self.wq_a_weight,
                &self.wq_a_scale,
                &self.wq_a.activation,
                &self.wq_a.scales,
                &self.wq_a.fp32,
                WQ_A_ROWS as u32,
                WQ_A_COLS as u32,
                (WQ_A_COLS / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(batch, &self.wq_a.fp32, &self.wq_a.bf16, WQ_A_ROWS as u32)?;
            dispatch_rms(
                batch,
                &self.wq_a.bf16,
                &self.q_norm_weight,
                &self.q_norm,
                q_lora,
            )?;
            // WQ-B
            dispatch_qat(
                batch,
                &self.q_norm,
                &self.wq_b.activation,
                &self.wq_b.scales,
                q_lora,
            )?;
            dispatch_fp8(
                batch,
                &self.wq_b_weight,
                &self.wq_b_scale,
                &self.wq_b.activation,
                &self.wq_b.scales,
                &self.wq_b.fp32,
                WQ_B_ROWS as u32,
                q_lora,
                (Q_LORA_RANK / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(batch, &self.wq_b.fp32, &self.wq_b.bf16, WQ_B_ROWS as u32)?;
            dispatch_per_head(batch, &self.wq_b.bf16, &self.q_head, heads, head_dim)?;
            let q_for_attn = if apply_rope {
                dispatch_rope(
                    batch,
                    &self.q_head,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.q_rope,
                    heads,
                    head_dim,
                    rope_dim,
                    false,
                )?;
                &self.q_rope
            } else {
                &self.q_head
            };
            // WKV
            dispatch_qat(
                batch,
                &self.attn_norm,
                &self.wkv.activation,
                &self.wkv.scales,
                hidden,
            )?;
            dispatch_fp8(
                batch,
                &self.wkv_weight,
                &self.wkv_scale,
                &self.wkv.activation,
                &self.wkv.scales,
                &self.wkv.fp32,
                WKV_ROWS as u32,
                hidden,
                (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(batch, &self.wkv.fp32, &self.wkv.bf16, WKV_ROWS as u32)?;
            dispatch_rms(
                batch,
                &self.wkv.bf16,
                &self.kv_norm_weight,
                &self.kv.norm,
                head_dim,
            )?;
            dispatch_kv_qat(
                batch,
                &self.kv.norm,
                &self.kv.qat,
                &self.kv.activation,
                &self.kv.scales,
                head_dim,
                rope_dim,
                kv_block,
            )?;
            let kv_for_cache = if apply_rope {
                dispatch_rope(
                    batch,
                    &self.kv.qat,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.kv_rope,
                    1,
                    head_dim,
                    rope_dim,
                    false,
                )?;
                &self.kv_rope
            } else {
                &self.kv.qat
            };
            dispatch_cache_write(
                batch,
                kv_for_cache,
                &kv_cache.buffer,
                cache_position,
                head_dim,
                cache_capacity,
            )?;
            dispatch_sparse_growing_kv(
                batch,
                q_for_attn,
                &kv_cache.buffer,
                &self.attn_sink,
                &self.sparse,
                &self.scores,
                &self.denominators,
                heads,
                head_dim,
                cache_capacity,
                valid_kv_u32,
                max_score_slots,
                sparse_scale,
            )?;
            let sparse_for_wo = if apply_rope {
                dispatch_rope(
                    batch,
                    &self.sparse,
                    &self.rope_cos,
                    &self.rope_sin,
                    &self.derotated,
                    heads,
                    head_dim,
                    rope_dim,
                    true,
                )?;
                &self.derotated
            } else {
                &self.sparse
            };
            dispatch_wo_a(
                batch,
                &self.wo_a_weight,
                &self.wo_a_scale,
                sparse_for_wo,
                &self.wo_a,
                WO_A_ROWS as u32,
                WO_A_COLS as u32,
                (WO_A_COLS / ACT_QUANT_BLOCK) as u32,
                O_LORA_RANK as u32,
            )?;
            dispatch_qat(
                batch,
                &self.wo_a,
                &self.wo_b.activation,
                &self.wo_b.scales,
                WO_B_COLS as u32,
            )?;
            dispatch_fp8(
                batch,
                &self.wo_b_weight,
                &self.wo_b_scale,
                &self.wo_b.activation,
                &self.wo_b.scales,
                &self.wo_b.fp32,
                WO_B_ROWS as u32,
                WO_B_COLS as u32,
                (WO_B_COLS / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(batch, &self.wo_b.fp32, &self.wo_b.bf16, WO_B_ROWS as u32)?;
            match self.input_kind {
                DeepSeekV4FullseqInputKind::Embedding => {
                    // L0 residual is the same embed buffer that seeded mHC-pre.
                    dispatch_hc_post_embed(
                        batch,
                        &self.wo_b.bf16,
                        input,
                        &self.hc.post,
                        &self.hc.comb,
                        &self.attention_hc_state_bf16,
                        hidden,
                        hc_mult,
                    )?;
                }
                DeepSeekV4FullseqInputKind::PredecessorHc => {
                    dispatch_hc_post_pred(
                        batch,
                        &self.wo_b.bf16,
                        input,
                        &self.hc.post,
                        &self.hc.comb,
                        &self.attention_hc_state_bf16,
                        hidden,
                        hc_mult,
                    )?;
                }
            }
            Ok(())
        })?;

        if timing.command_buffers != 1
            || timing.compute_encoders as usize != expected_dispatches
            || timing.compute_dispatches as usize != expected_dispatches
        {
            return Err(fullseq_error(format!(
                "fullseq attention topology mismatch: expected {expected_dispatches} dispatches, got encoders={} dispatches={} cbs={}",
                timing.compute_encoders, timing.compute_dispatches, timing.command_buffers
            )));
        }
        kv_cache.filled = valid_kv;
        let empty_compressed = self.compress_ratio == 0
            || (token_position + 1) / self.compress_ratio == 0;
        Ok(DeepSeekV4FullseqAttentionDeviceOutput {
            attention_hc_state_bf16: self.attention_hc_state_bf16.to_owned(),
            layer: self.layer,
            token_id,
            token_position,
            kv_rows: valid_kv,
            compress_ratio: self.compress_ratio,
            empty_compressed,
            actual_gpu_dispatches: expected_dispatches,
            actual_command_buffers: 1,
            actual_cpu_visible_waits: 1,
            timing,
            sparse_attention_kernel: SPARSE_KERNEL,
        })
    }
}

/// Load one embedding row as a Metal buffer (BF16[4096]).
pub fn load_embedding_row_buffer(
    metal: &MetalContext,
    reader: &DeepSeekV4FullStreamReader,
    token_id: u32,
) -> Result<metal::Buffer> {
    let bytes = load_embedding_row_bytes(reader, token_id)?;
    metal.new_buffer_with_bytes_checked(&bytes)
}

pub fn load_embedding_row_bytes(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u32,
) -> Result<Vec<u8>> {
    let metadata = reader.tensor_metadata(EMBED_WEIGHT)?;
    if metadata.dtype != "BF16"
        || metadata.shape.as_slice() != [VOCAB_SIZE, HIDDEN_SIZE as u64]
        || u64::from(token_id) >= VOCAB_SIZE
    {
        return Err(fullseq_error("embedding geometry/token invalid"));
    }
    let nbytes = HIDDEN_SIZE * size_of::<u16>();
    let start = u64::from(token_id)
        .checked_mul(nbytes as u64)
        .ok_or_else(|| fullseq_error("embedding row overflow"))?;
    reader.read_verified_range(EMBED_WEIGHT, start..start + nbytes as u64, nbytes)
}

// ---- control loading ----

impl LayerControls {
    fn load(reader: &DeepSeekV4FullStreamReader, layer: usize) -> Result<Self> {
        let name = |suffix: &str| format!("layers.{layer}.{suffix}");
        Ok(Self {
            hc_fn: read_tensor(reader, &name("hc_attn_fn"), "F32")?,
            hc_base: read_tensor(reader, &name("hc_attn_base"), "F32")?,
            hc_scale: read_tensor(reader, &name("hc_attn_scale"), "F32")?,
            attn_norm: read_tensor(reader, &name("attn_norm.weight"), "BF16")?,
            q_norm: read_tensor(reader, &name("attn.q_norm.weight"), "BF16")?,
            kv_norm: read_tensor(reader, &name("attn.kv_norm.weight"), "BF16")?,
            attn_sink: read_tensor(reader, &name("attn.attn_sink"), "F32")?,
            wq_a: read_fp8(
                reader,
                &name("attn.wq_a.weight"),
                &name("attn.wq_a.scale"),
                WQ_A_ROWS,
                WQ_A_COLS,
            )?,
            wq_b: read_fp8(
                reader,
                &name("attn.wq_b.weight"),
                &name("attn.wq_b.scale"),
                WQ_B_ROWS,
                Q_LORA_RANK,
            )?,
            wkv: read_fp8(
                reader,
                &name("attn.wkv.weight"),
                &name("attn.wkv.scale"),
                WKV_ROWS,
                HIDDEN_SIZE,
            )?,
            wo_a: read_fp8(
                reader,
                &name("attn.wo_a.weight"),
                &name("attn.wo_a.scale"),
                WO_A_ROWS,
                WO_A_COLS,
            )?,
            wo_b: read_fp8(
                reader,
                &name("attn.wo_b.weight"),
                &name("attn.wo_b.scale"),
                WO_B_ROWS,
                WO_B_COLS,
            )?,
        })
    }
}

fn read_tensor(reader: &DeepSeekV4FullStreamReader, name: &str, dtype: &str) -> Result<VerifiedTensor> {
    let meta = reader.tensor_metadata(name)?;
    if meta.dtype != dtype {
        return Err(fullseq_error(format!(
            "{name} dtype {} != {dtype}",
            meta.dtype
        )));
    }
    let bytes = reader.read_verified_full(name, meta.bytes as usize)?;
    Ok(VerifiedTensor {
        name: name.to_owned(),
        bytes,
    })
}

fn read_fp8(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    rows: usize,
    cols: usize,
) -> Result<Fp8Pair> {
    let pair = reader.native_scale_pair(weight_name)?;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.weight.name != weight_name
        || pair.scale.name != scale_name
        || pair.weight.shape.as_slice() != [rows as u64, cols as u64]
    {
        return Err(fullseq_error(format!(
            "FP8 pair geometry differs for {weight_name}"
        )));
    }
    Ok(Fp8Pair {
        weight: read_tensor(reader, weight_name, "F8_E4M3")?,
        scale: read_tensor(reader, scale_name, "F8_E8M0")?,
    })
}

fn verify_source_identity(reader: &DeepSeekV4FullStreamReader) -> Result<()> {
    if reader.source_identity().repository != PINNED_REPOSITORY
        || reader.source_identity().revision != PINNED_REVISION
    {
        return Err(fullseq_error(
            "fullseq attention requires the pinned DeepSeek-V4-Flash source identity",
        ));
    }
    Ok(())
}

// ---- scratch / dispatch helpers ----

fn new_hc_scratch(metal: &MetalContext) -> Result<HcScratch> {
    Ok(HcScratch {
        reduced: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<u16>())?,
        rsqrt: metal.new_buffer_checked(size_of::<f32>())?,
        mixes: metal.new_buffer_checked(HC_MIX_WIDTH * size_of::<f32>())?,
        pre: metal.new_buffer_checked(HC_MULT * size_of::<f32>())?,
        post: metal.new_buffer_checked(HC_MULT * size_of::<f32>())?,
        comb: metal.new_buffer_checked(HC_MULT * HC_MULT * size_of::<f32>())?,
    })
}

fn new_linear_scratch(metal: &MetalContext, cols: usize, rows: usize) -> Result<LinearScratch> {
    Ok(LinearScratch {
        activation: metal.new_buffer_checked(cols)?,
        scales: metal.new_buffer_checked(cols / ACT_QUANT_BLOCK)?,
        fp32: metal.new_buffer_checked(rows * size_of::<f32>())?,
        bf16: metal.new_buffer_checked(rows * size_of::<u16>())?,
    })
}

fn new_kv_scratch(metal: &MetalContext) -> Result<KvScratch> {
    Ok(KvScratch {
        norm: metal.new_buffer_checked(HEAD_DIM * size_of::<u16>())?,
        qat: metal.new_buffer_checked(HEAD_DIM * size_of::<u16>())?,
        activation: metal.new_buffer_checked(NON_ROPE_HEAD_DIM)?,
        scales: metal.new_buffer_checked(NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?,
    })
}

fn validate_required_pipelines(metal: &MetalContext) -> Result<()> {
    for kernel in [
        HC_PRE_PRED_KERNEL,
        HC_PRE_EMBED_KERNEL,
        RMS_KERNEL,
        QAT_KERNEL,
        FP8_KERNEL,
        CAST_KERNEL,
        PER_HEAD_KERNEL,
        KV_QAT_KERNEL,
        CACHE_KERNEL,
        ROPE_KERNEL,
        SPARSE_KERNEL,
        WO_A_KERNEL,
        HC_POST_PRED_KERNEL,
        HC_POST_EMBED_KERNEL,
    ] {
        let _ = metal.pipeline(kernel)?;
    }
    Ok(())
}

fn upload_f32(metal: &MetalContext, buffer: &metal::Buffer, values: &[f32]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    if buffer.length() < bytes.len() as u64 {
        return Err(fullseq_error("upload buffer too small"));
    }
    // Host write of static control table only (not activation handoff).
    let ptr = buffer.contents() as *mut u8;
    if ptr.is_null() {
        return Err(fullseq_error("buffer contents null for control upload"));
    }
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, bytes.len());
    }
    let _ = metal; // queue identity already checked by caller
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hc_pre_pred(
    batch: &mut CommandBatch<'_>,
    residual: &metal::Buffer,
    hc_fn: &metal::Buffer,
    hc_scale: &metal::Buffer,
    hc_base: &metal::Buffer,
    scratch: &HcScratch,
    hidden: u32,
    hc_mult: u32,
    mix_width: u32,
    sinkhorn: u32,
) -> Result<()> {
    batch.dispatch_threads(HC_PRE_PRED_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
        e.set_buffer(0, Some(residual), 0);
        e.set_buffer(1, Some(hc_fn), 0);
        e.set_buffer(2, Some(hc_scale), 0);
        e.set_buffer(3, Some(hc_base), 0);
        e.set_buffer(4, Some(&scratch.reduced), 0);
        e.set_buffer(5, Some(&scratch.rsqrt), 0);
        e.set_buffer(6, Some(&scratch.mixes), 0);
        e.set_buffer(7, Some(&scratch.pre), 0);
        e.set_buffer(8, Some(&scratch.post), 0);
        e.set_buffer(9, Some(&scratch.comb), 0);
        set_u32(e, 10, &hidden);
        set_u32(e, 11, &hc_mult);
        set_u32(e, 12, &mix_width);
        set_u32(e, 13, &sinkhorn);
        set_f32(e, 14, &RMS_NORM_EPS);
        set_f32(e, 15, &HC_EPS);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hc_pre_embed(
    batch: &mut CommandBatch<'_>,
    embed: &metal::Buffer,
    hc_fn: &metal::Buffer,
    hc_scale: &metal::Buffer,
    hc_base: &metal::Buffer,
    scratch: &HcScratch,
    hidden: u32,
    hc_mult: u32,
    mix_width: u32,
    sinkhorn: u32,
) -> Result<()> {
    batch.dispatch_threads(HC_PRE_EMBED_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
        e.set_buffer(0, Some(embed), 0);
        e.set_buffer(1, Some(hc_fn), 0);
        e.set_buffer(2, Some(hc_scale), 0);
        e.set_buffer(3, Some(hc_base), 0);
        e.set_buffer(4, Some(&scratch.reduced), 0);
        e.set_buffer(5, Some(&scratch.rsqrt), 0);
        e.set_buffer(6, Some(&scratch.mixes), 0);
        e.set_buffer(7, Some(&scratch.pre), 0);
        e.set_buffer(8, Some(&scratch.post), 0);
        e.set_buffer(9, Some(&scratch.comb), 0);
        set_u32(e, 10, &hidden);
        set_u32(e, 11, &hc_mult);
        set_u32(e, 12, &mix_width);
        set_u32(e, 13, &sinkhorn);
        set_f32(e, 14, &RMS_NORM_EPS);
        set_f32(e, 15, &HC_EPS);
    })
}

fn dispatch_rms(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    weight: &metal::Buffer,
    output: &metal::Buffer,
    width: u32,
) -> Result<()> {
    batch.dispatch_threads(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(weight), 0);
        e.set_buffer(2, Some(output), 0);
        set_u32(e, 3, &width);
        set_f32(e, 4, &RMS_NORM_EPS);
    })
}

fn dispatch_qat(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    scales: &metal::Buffer,
    cols: u32,
) -> Result<()> {
    batch.dispatch_threads(
        QAT_KERNEL,
        (cols / ACT_QUANT_BLOCK as u32, 1, 1),
        (32, 1, 1),
        |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(output), 0);
            e.set_buffer(2, Some(scales), 0);
            set_u32(e, 3, &cols);
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_fp8(
    batch: &mut CommandBatch<'_>,
    weight: &metal::Buffer,
    scales: &metal::Buffer,
    activation: &metal::Buffer,
    activation_scales: &metal::Buffer,
    output: &metal::Buffer,
    rows: u32,
    cols: u32,
    scale_cols: u32,
) -> Result<()> {
    batch.dispatch_threads(FP8_KERNEL, (rows, 1, 1), (256, 1, 1), |e| {
        e.set_buffer(0, Some(weight), 0);
        e.set_buffer(1, Some(scales), 0);
        e.set_buffer(2, Some(activation), 0);
        e.set_buffer(3, Some(activation_scales), 0);
        e.set_buffer(4, Some(output), 0);
        set_u32(e, 5, &rows);
        set_u32(e, 6, &cols);
        set_u32(e, 7, &scale_cols);
    })
}

fn dispatch_cast(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    count: u32,
) -> Result<()> {
    batch.dispatch_threads(CAST_KERNEL, (count, 1, 1), (256, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(output), 0);
        set_u32(e, 2, &count);
    })
}

fn dispatch_per_head(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    heads: u32,
    head_dim: u32,
) -> Result<()> {
    batch.dispatch_threads(PER_HEAD_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(output), 0);
        set_u32(e, 2, &heads);
        set_u32(e, 3, &head_dim);
        set_f32(e, 4, &RMS_NORM_EPS);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_kv_qat(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    activation: &metal::Buffer,
    scales: &metal::Buffer,
    head_dim: u32,
    rope_dim: u32,
    block: u32,
) -> Result<()> {
    batch.dispatch_threads(
        KV_QAT_KERNEL,
        (NON_ROPE_HEAD_DIM as u32 / block, 1, 1),
        (32, 1, 1),
        |e| {
            e.set_buffer(0, Some(input), 0);
            e.set_buffer(1, Some(output), 0);
            e.set_buffer(2, Some(activation), 0);
            e.set_buffer(3, Some(scales), 0);
            set_u32(e, 4, &head_dim);
            set_u32(e, 5, &rope_dim);
            set_u32(e, 6, &block);
        },
    )
}

fn dispatch_cache_write(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    cache: &metal::Buffer,
    position: u32,
    head_dim: u32,
    capacity: u32,
) -> Result<()> {
    batch.dispatch_threads(CACHE_KERNEL, (head_dim, 1, 1), (256, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(cache), 0);
        set_u32(e, 2, &position);
        set_u32(e, 3, &head_dim);
        set_u32(e, 4, &capacity);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_rope(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    cos: &metal::Buffer,
    sin: &metal::Buffer,
    output: &metal::Buffer,
    rows: u32,
    head_dim: u32,
    rope_dim: u32,
    inverse: bool,
) -> Result<()> {
    let inv = if inverse { 1u32 } else { 0u32 };
    let pairs = rows * (head_dim / 2);
    batch.dispatch_threads(ROPE_KERNEL, (pairs, 1, 1), (64, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(cos), 0);
        e.set_buffer(2, Some(sin), 0);
        e.set_buffer(3, Some(output), 0);
        set_u32(e, 4, &rows);
        set_u32(e, 5, &head_dim);
        set_u32(e, 6, &rope_dim);
        set_u32(e, 7, &inv);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_sparse_growing_kv(
    batch: &mut CommandBatch<'_>,
    q: &metal::Buffer,
    kv_cache: &metal::Buffer,
    sink: &metal::Buffer,
    output: &metal::Buffer,
    scores: &metal::Buffer,
    denominators: &metal::Buffer,
    heads: u32,
    head_dim: u32,
    cache_capacity: u32,
    valid_kv_count: u32,
    max_score_slots: u32,
    scale: f32,
) -> Result<()> {
    batch.dispatch_threads(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
        e.set_buffer(0, Some(q), 0);
        e.set_buffer(1, Some(kv_cache), 0);
        e.set_buffer(2, Some(sink), 0);
        e.set_buffer(3, Some(output), 0);
        e.set_buffer(4, Some(scores), 0);
        e.set_buffer(5, Some(denominators), 0);
        set_u32(e, 6, &heads);
        set_u32(e, 7, &head_dim);
        set_u32(e, 8, &cache_capacity);
        set_u32(e, 9, &valid_kv_count);
        set_u32(e, 10, &max_score_slots);
        set_f32(e, 11, &scale);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_wo_a(
    batch: &mut CommandBatch<'_>,
    weight: &metal::Buffer,
    scales: &metal::Buffer,
    input: &metal::Buffer,
    output: &metal::Buffer,
    rows: u32,
    cols: u32,
    scale_cols: u32,
    ranks: u32,
) -> Result<()> {
    batch.dispatch_threads(WO_A_KERNEL, (rows, 1, 1), (256, 1, 1), |e| {
        e.set_buffer(0, Some(weight), 0);
        e.set_buffer(1, Some(scales), 0);
        e.set_buffer(2, Some(input), 0);
        e.set_buffer(3, Some(output), 0);
        set_u32(e, 4, &rows);
        set_u32(e, 5, &cols);
        set_u32(e, 6, &scale_cols);
        set_u32(e, 7, &ranks);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hc_post_pred(
    batch: &mut CommandBatch<'_>,
    attention: &metal::Buffer,
    residual: &metal::Buffer,
    post: &metal::Buffer,
    comb: &metal::Buffer,
    output: &metal::Buffer,
    hidden: u32,
    hc_mult: u32,
) -> Result<()> {
    batch.dispatch_threads(
        HC_POST_PRED_KERNEL,
        (hidden * hc_mult, 1, 1),
        (256, 1, 1),
        |e| {
            e.set_buffer(0, Some(attention), 0);
            e.set_buffer(1, Some(residual), 0);
            e.set_buffer(2, Some(post), 0);
            e.set_buffer(3, Some(comb), 0);
            e.set_buffer(4, Some(output), 0);
            set_u32(e, 5, &hidden);
            set_u32(e, 6, &hc_mult);
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hc_post_embed(
    batch: &mut CommandBatch<'_>,
    attention: &metal::Buffer,
    residual_embed: &metal::Buffer,
    post: &metal::Buffer,
    comb: &metal::Buffer,
    output: &metal::Buffer,
    hidden: u32,
    hc_mult: u32,
) -> Result<()> {
    batch.dispatch_threads(
        HC_POST_EMBED_KERNEL,
        (hidden * hc_mult, 1, 1),
        (256, 1, 1),
        |e| {
            e.set_buffer(0, Some(attention), 0);
            e.set_buffer(1, Some(residual_embed), 0);
            e.set_buffer(2, Some(post), 0);
            e.set_buffer(3, Some(comb), 0);
            e.set_buffer(4, Some(output), 0);
            set_u32(e, 5, &hidden);
            set_u32(e, 6, &hc_mult);
        },
    )
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
    encoder.set_bytes(
        index,
        size_of::<u32>() as u64,
        value as *const u32 as *const _,
    );
}

fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
    encoder.set_bytes(
        index,
        size_of::<f32>() as u64,
        value as *const f32 as *const _,
    );
}

fn context_queue_identity(context: &MetalContext) -> usize {
    context.queue() as *const _ as usize
}

fn fullseq_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 fullseq attention device: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_counts_are_explicit() {
        assert_eq!(DSV4F_FULLSEQ_ATTENTION_DISPATCHES_POS0, 22);
        assert_eq!(DSV4F_FULLSEQ_ATTENTION_DISPATCHES_POS_N, 25);
        assert_eq!(DSV4F_FULLSEQ_KV_CAPACITY, 128);
        assert!(SPARSE_KERNEL.contains("growing_kv"));
    }

}
