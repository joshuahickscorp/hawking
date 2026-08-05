//! Reusable, bounded all-device P4B layer-0 position-one attention graph.
//!
//! This is the source-backed handoff body behind the sealed P4B authority
//! probe.  It deliberately remains below a causal runtime: it runs only the
//! fixed `[BOS, Hello]` trace at layer 0 / position 1, has no sampling or
//! endpoint surface, and retains the terminal state as
//! [`DeepSeekV4P4bParityClassification::NumericParityV21Only`].  In
//! particular, it must not be relabelled as exact-storage parity: the sealed
//! authority path measured one terminal mHC-post BF16 word difference after
//! source/device exp-based controls, while all causal cache stores remain
//! exact and V2.1 passes.
//!
//! The executor never creates a Metal context and never exposes a host
//! activation/readback API.  Its caller supplies the context for preparation,
//! execution, and the P7 borrow; a queue identity check rejects cross-context
//! use.  The retained terminal buffers are exactly the P7 boundary:
//! `BF16[4,4096]` attention state and bounded `BF16[2,512]` causal KV rows.

use std::mem::size_of;

use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::gravity_deepseek_v4_act_quant::{
    ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
};
use crate::gravity_deepseek_v4_layer0_attention::{
    verify_layer0_attention_source_anchors, HEAD_DIM, KV_QAT_BLOCK, LAYER0_ATTN_SINK,
    LAYER0_KV_NORM_WEIGHT, LAYER0_Q_NORM_WEIGHT, LAYER0_WKV_SCALE, LAYER0_WKV_WEIGHT,
    LAYER0_WO_A_SCALE, LAYER0_WO_A_WEIGHT, LAYER0_WO_B_SCALE, LAYER0_WO_B_WEIGHT,
    LAYER0_WQ_B_SCALE, LAYER0_WQ_B_WEIGHT, NON_ROPE_HEAD_DIM, NUM_HEADS, O_LORA_RANK, Q_LORA_RANK,
    ROPE_HEAD_DIM, WKV_ROWS, WO_A_COLS, WO_A_ROWS, WO_B_COLS, WO_B_ROWS, WQ_B_ROWS,
};
use crate::gravity_deepseek_v4_layer0_continuation::{
    layer0_position1_rope_table, verify_layer0_position1_continuation_anchors, POSITION1,
    POSITION1_KV_ROWS, POSITION1_TOKEN_ID,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    EMBED_WEIGHT, HC_EPS, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE,
    LAYER0_ATTN_NORM_WEIGHT, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_FN, LAYER0_HC_ATTN_SCALE,
    PREFIX_TOKEN_ID, RMS_NORM_EPS, VOCAB_SIZE,
};
use crate::gravity_deepseek_v4_p7_composition::DeepSeekV4P7AttentionDeviceState;
use crate::metal::{CommandBatch, MetalBatchTiming, MetalContext};
use crate::{Error, Result};

const LAYER0: usize = 0;
const HC_KERNEL: &str = "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority";
const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
const ROPE_KERNEL: &str = "deepseek_v4_p4b_rope_position1_bf16_authority";
const CACHE_KERNEL: &str = "deepseek_v4_p4b_kv_cache_write_bf16_authority";
const SPARSE_KERNEL: &str = "deepseek_v4_p4b_sparse_attention_position1_two_kv_sink_authority";
const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
const HC_POST_KERNEL: &str = "deepseek_v4_p4a_hc_attn_post_authority";

pub const DSV4F_P4B_LAYER: usize = LAYER0;
pub const DSV4F_P4B_POSITION1_TOKEN_ID: u32 = POSITION1_TOKEN_ID as u32;
pub const DSV4F_P4B_POSITION1: usize = POSITION1;
pub const DSV4F_P4B_CAUSAL_KV_ROWS: usize = POSITION1_KV_ROWS;
pub const DSV4F_P4B_CAUSAL_KV_BF16_BYTES: usize = POSITION1_KV_ROWS * HEAD_DIM * size_of::<u16>();
pub const DSV4F_P4B_ATTENTION_HC_POST_BF16_BYTES: usize = HC_MULT * HIDDEN_SIZE * size_of::<u16>();
pub const DSV4F_P4B_DEVICE_DISPATCHES: usize = 33;

/// The strongest honesty label currently available for this reusable P4B
/// state.  It is intentionally not convertible to an exact-parity label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4P4bParityClassification {
    NumericParityV21Only,
}

impl DeepSeekV4P4bParityClassification {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NumericParityV21Only => "NUMERIC_PARITY_V2_1_ONLY",
        }
    }

    pub const fn is_exact_storage(self) -> bool {
        false
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4P4bDevicePhase {
    Prepared,
    Position1Complete,
}

/// Metadata-only source binding for a raw-artifact P4B graph.  No hidden
/// state or source tensor payload is retained in this public surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P4bDeviceSourceBindings {
    pub artifact_manifest_seal_sha256: String,
    pub repository: String,
    pub revision: String,
    pub layer: usize,
    pub token_ids: [u32; 2],
    pub terminal_position: usize,
    pub embedding_row_sha256: [String; 2],
    pub inference_model_py_sha256: String,
    pub inference_kernel_py_sha256: String,
    pub inference_convert_py_sha256: String,
    pub inference_config_json_sha256: String,
    pub source_parent_retained: bool,
    pub host_activation_handoff_permitted: bool,
    pub terminal_parity_classification: DeepSeekV4P4bParityClassification,
}

/// Completion accounting for one caller-owned, all-device P4B invocation.
/// Timing is a completed one-command-buffer aggregate; it is not a decode
/// benchmark and does not promote the topology into a persistent graph.
#[derive(Debug, Clone, Copy)]
pub struct DeepSeekV4P4bDeviceExecution {
    pub phase: DeepSeekV4P4bDevicePhase,
    pub parity_classification: DeepSeekV4P4bParityClassification,
    pub timing: MetalBatchTiming,
    pub actual_command_buffers: usize,
    pub actual_compute_encoders: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
    pub host_intermediate_handoff_bytes: usize,
}

struct HcState {
    embed: metal::Buffer,
    reduced: metal::Buffer,
    rsqrt: metal::Buffer,
    mixes: metal::Buffer,
    pre: metal::Buffer,
    post: metal::Buffer,
    comb: metal::Buffer,
    attn_norm: metal::Buffer,
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

/// Bounded reusable P4B resource graph.  It owns static raw-artifact controls
/// and all device intermediates, but never owns a Metal context.  Its final
/// state can be borrowed directly by P7 only through [`Self::p7_attention_state`].
pub struct DeepSeekV4Layer0P4bDeviceExecutor {
    source_bindings: DeepSeekV4P4bDeviceSourceBindings,
    context_queue_identity: usize,
    phase: DeepSeekV4P4bDevicePhase,
    hc_fn: metal::Buffer,
    hc_scale: metal::Buffer,
    hc_base: metal::Buffer,
    attn_norm_weight: metal::Buffer,
    q_norm_weight: metal::Buffer,
    kv_norm_weight: metal::Buffer,
    attn_sink: metal::Buffer,
    rope_cos: metal::Buffer,
    rope_sin: metal::Buffer,
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
    p0: HcState,
    p1: HcState,
    p0_wkv: LinearScratch,
    p1_wq_a: LinearScratch,
    p1_wq_b: LinearScratch,
    p1_wkv: LinearScratch,
    p1_wo_b: LinearScratch,
    p0_kv: KvScratch,
    p1_kv: KvScratch,
    p1_q_norm: metal::Buffer,
    p1_q_head: metal::Buffer,
    p1_q_rope: metal::Buffer,
    p1_kv_rope: metal::Buffer,
    causal_kv_cache_bf16: metal::Buffer,
    p1_sparse: metal::Buffer,
    p1_scores: metal::Buffer,
    p1_denominators: metal::Buffer,
    p1_derotated: metal::Buffer,
    p1_wo_a: metal::Buffer,
    attention_hc_post_bf16: metal::Buffer,
}

impl DeepSeekV4Layer0P4bDeviceExecutor {
    /// Upload only raw-artifact controls and fixed tokenizer-bound embedding
    /// rows into a caller-owned Metal context.  This performs no model
    /// dispatch and no host-built hidden-state upload.
    pub fn prepare(metal: &MetalContext, reader: &DeepSeekV4FullStreamReader) -> Result<Self> {
        verify_layer0_position1_continuation_anchors(reader)?;
        let anchors = verify_layer0_attention_source_anchors(reader)?;
        if reader.source_identity().repository != PINNED_REPOSITORY
            || reader.source_identity().revision != PINNED_REVISION
        {
            return Err(p4b_error(
                "P4B executor requires the pinned DeepSeek-V4 source",
            ));
        }
        let embed0 = embedding_row(reader, PREFIX_TOKEN_ID)?;
        let embed1 = embedding_row(reader, POSITION1_TOKEN_ID)?;
        let hc_fn = full(reader, LAYER0_HC_ATTN_FN)?;
        let hc_scale = full(reader, LAYER0_HC_ATTN_SCALE)?;
        let hc_base = full(reader, LAYER0_HC_ATTN_BASE)?;
        let attn_norm = full(reader, LAYER0_ATTN_NORM_WEIGHT)?;
        let q_norm = full(reader, LAYER0_Q_NORM_WEIGHT)?;
        let kv_norm = full(reader, LAYER0_KV_NORM_WEIGHT)?;
        let sink = full(reader, LAYER0_ATTN_SINK)?;
        let rope = layer0_position1_rope_table(reader)?;
        validate_geometry(
            &embed0, &embed1, &hc_fn, &hc_scale, &hc_base, &attn_norm, &q_norm, &kv_norm, &sink,
        )?;
        let (wq_a_weight, wq_a_scale) = fp8_pair(
            reader,
            LAYER0_WQ_A_WEIGHT,
            LAYER0_WQ_A_SCALE,
            LAYER0_WQ_A_ROWS,
            LAYER0_WQ_A_COLS,
        )?;
        let (wq_b_weight, wq_b_scale) = fp8_pair(
            reader,
            LAYER0_WQ_B_WEIGHT,
            LAYER0_WQ_B_SCALE,
            WQ_B_ROWS,
            Q_LORA_RANK,
        )?;
        let (wkv_weight, wkv_scale) = fp8_pair(
            reader,
            LAYER0_WKV_WEIGHT,
            LAYER0_WKV_SCALE,
            WKV_ROWS,
            HIDDEN_SIZE,
        )?;
        let (wo_a_weight, wo_a_scale) = fp8_pair(
            reader,
            LAYER0_WO_A_WEIGHT,
            LAYER0_WO_A_SCALE,
            WO_A_ROWS,
            WO_A_COLS,
        )?;
        let (wo_b_weight, wo_b_scale) = fp8_pair(
            reader,
            LAYER0_WO_B_WEIGHT,
            LAYER0_WO_B_SCALE,
            WO_B_ROWS,
            WO_B_COLS,
        )?;
        for kernel in [
            HC_KERNEL,
            RMS_KERNEL,
            QAT_KERNEL,
            FP8_KERNEL,
            CAST_KERNEL,
            PER_HEAD_KERNEL,
            KV_QAT_KERNEL,
            ROPE_KERNEL,
            CACHE_KERNEL,
            SPARSE_KERNEL,
            WO_A_KERNEL,
            HC_POST_KERNEL,
        ] {
            metal.pipeline(kernel)?;
        }
        let source_bindings = DeepSeekV4P4bDeviceSourceBindings {
            artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            repository: reader.source_identity().repository.to_owned(),
            revision: reader.source_identity().revision.to_owned(),
            layer: LAYER0,
            token_ids: [PREFIX_TOKEN_ID as u32, POSITION1_TOKEN_ID as u32],
            terminal_position: POSITION1,
            embedding_row_sha256: [sha256(&embed0), sha256(&embed1)],
            inference_model_py_sha256: anchors.prefix.act_quant.inference_model_py_sha256,
            inference_kernel_py_sha256: anchors.prefix.act_quant.inference_kernel_py_sha256,
            inference_convert_py_sha256: anchors.inference_convert_py_sha256,
            inference_config_json_sha256: anchors.prefix.act_quant.inference_config_json_sha256,
            source_parent_retained: false,
            host_activation_handoff_permitted: false,
            terminal_parity_classification: DeepSeekV4P4bParityClassification::NumericParityV21Only,
        };
        Ok(Self {
            source_bindings,
            context_queue_identity: context_queue_identity(metal),
            phase: DeepSeekV4P4bDevicePhase::Prepared,
            hc_fn: metal.new_buffer_with_bytes_checked(&hc_fn)?,
            hc_scale: metal.new_buffer_with_bytes_checked(&hc_scale)?,
            hc_base: metal.new_buffer_with_bytes_checked(&hc_base)?,
            attn_norm_weight: metal.new_buffer_with_bytes_checked(&attn_norm)?,
            q_norm_weight: metal.new_buffer_with_bytes_checked(&q_norm)?,
            kv_norm_weight: metal.new_buffer_with_bytes_checked(&kv_norm)?,
            attn_sink: metal.new_buffer_with_bytes_checked(&sink)?,
            rope_cos: metal.new_buffer_with_bytes_checked(&f32bytes(&rope.cos_f32))?,
            rope_sin: metal.new_buffer_with_bytes_checked(&f32bytes(&rope.sin_f32))?,
            wq_a_weight: metal.new_buffer_with_bytes_checked(&wq_a_weight)?,
            wq_a_scale: metal.new_buffer_with_bytes_checked(&wq_a_scale)?,
            wq_b_weight: metal.new_buffer_with_bytes_checked(&wq_b_weight)?,
            wq_b_scale: metal.new_buffer_with_bytes_checked(&wq_b_scale)?,
            wkv_weight: metal.new_buffer_with_bytes_checked(&wkv_weight)?,
            wkv_scale: metal.new_buffer_with_bytes_checked(&wkv_scale)?,
            wo_a_weight: metal.new_buffer_with_bytes_checked(&wo_a_weight)?,
            wo_a_scale: metal.new_buffer_with_bytes_checked(&wo_a_scale)?,
            wo_b_weight: metal.new_buffer_with_bytes_checked(&wo_b_weight)?,
            wo_b_scale: metal.new_buffer_with_bytes_checked(&wo_b_scale)?,
            p0: new_hc_state(metal, &embed0)?,
            p1: new_hc_state(metal, &embed1)?,
            p0_wkv: new_linear_scratch(metal, HIDDEN_SIZE, WKV_ROWS)?,
            p1_wq_a: new_linear_scratch(metal, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS)?,
            p1_wq_b: new_linear_scratch(metal, Q_LORA_RANK, WQ_B_ROWS)?,
            p1_wkv: new_linear_scratch(metal, HIDDEN_SIZE, WKV_ROWS)?,
            p1_wo_b: new_linear_scratch(metal, WO_B_COLS, WO_B_ROWS)?,
            p0_kv: new_kv_scratch(metal)?,
            p1_kv: new_kv_scratch(metal)?,
            p1_q_norm: metal.new_buffer_checked(Q_LORA_RANK * size_of::<u16>())?,
            p1_q_head: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            p1_q_rope: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            p1_kv_rope: metal.new_buffer_checked(HEAD_DIM * size_of::<u16>())?,
            causal_kv_cache_bf16: metal.new_buffer_checked(DSV4F_P4B_CAUSAL_KV_BF16_BYTES)?,
            p1_sparse: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            p1_scores: metal
                .new_buffer_checked(NUM_HEADS * POSITION1_KV_ROWS * size_of::<f32>())?,
            p1_denominators: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
            p1_derotated: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            p1_wo_a: metal.new_buffer_checked(WO_A_ROWS * size_of::<u16>())?,
            attention_hc_post_bf16: metal
                .new_buffer_checked(DSV4F_P4B_ATTENTION_HC_POST_BF16_BYTES)?,
        })
    }

    pub fn source_bindings(&self) -> &DeepSeekV4P4bDeviceSourceBindings {
        &self.source_bindings
    }
    pub const fn phase(&self) -> DeepSeekV4P4bDevicePhase {
        self.phase
    }
    pub const fn parity_classification(&self) -> DeepSeekV4P4bParityClassification {
        DeepSeekV4P4bParityClassification::NumericParityV21Only
    }

    /// Execute P0 cache write followed by P1 causal cache read and complete
    /// attention tail.  The one CB contains ordered encoders only; no
    /// concurrency or persistent-decode topology is claimed.
    pub fn execute_position1(
        &mut self,
        metal: &MetalContext,
    ) -> Result<DeepSeekV4P4bDeviceExecution> {
        self.check_context(metal)?;
        let hidden = HIDDEN_SIZE as u32;
        let hc_mult = HC_MULT as u32;
        let mix_width = HC_MIX_WIDTH as u32;
        let sinkhorn = HC_SINKHORN_ITERS as u32;
        let norm_eps = RMS_NORM_EPS;
        let hc_eps = HC_EPS;
        let heads = NUM_HEADS as u32;
        let head_dim = HEAD_DIM as u32;
        let q_lora = Q_LORA_RANK as u32;
        let rope_dim = ROPE_HEAD_DIM as u32;
        let kv_block = KV_QAT_BLOCK as u32;
        let cache_capacity = POSITION1_KV_ROWS as u32;
        let sparse_scale = (HEAD_DIM as f32).powf(-0.5);
        let position0 = 0u32;
        let position1 = POSITION1 as u32;
        let forward = 0u32;
        let inverse = 1u32;
        let timing = metal.dispatch_batch_timed(|batch| {
            dispatch_hc(
                batch,
                &self.p0,
                &self.hc_fn,
                &self.hc_scale,
                &self.hc_base,
                hidden,
                hc_mult,
                mix_width,
                sinkhorn,
                norm_eps,
                hc_eps,
            )?;
            dispatch_rms(
                batch,
                &self.p0.reduced,
                &self.attn_norm_weight,
                &self.p0.attn_norm,
                hidden,
                norm_eps,
            )?;
            dispatch_qat(
                batch,
                &self.p0.attn_norm,
                &self.p0_wkv.activation,
                &self.p0_wkv.scales,
                hidden,
            )?;
            dispatch_fp8(
                batch,
                &self.wkv_weight,
                &self.wkv_scale,
                &self.p0_wkv.activation,
                &self.p0_wkv.scales,
                &self.p0_wkv.fp32,
                WKV_ROWS as u32,
                hidden,
                (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(batch, &self.p0_wkv.fp32, &self.p0_wkv.bf16, WKV_ROWS as u32)?;
            dispatch_rms(
                batch,
                &self.p0_wkv.bf16,
                &self.kv_norm_weight,
                &self.p0_kv.norm,
                head_dim,
                norm_eps,
            )?;
            dispatch_kv_qat(
                batch,
                &self.p0_kv.norm,
                &self.p0_kv.qat,
                &self.p0_kv.activation,
                &self.p0_kv.scales,
                head_dim,
                rope_dim,
                kv_block,
            )?;
            dispatch_cache_write(
                batch,
                &self.p0_kv.qat,
                &self.causal_kv_cache_bf16,
                position0,
                head_dim,
                cache_capacity,
            )?;

            dispatch_hc(
                batch,
                &self.p1,
                &self.hc_fn,
                &self.hc_scale,
                &self.hc_base,
                hidden,
                hc_mult,
                mix_width,
                sinkhorn,
                norm_eps,
                hc_eps,
            )?;
            dispatch_rms(
                batch,
                &self.p1.reduced,
                &self.attn_norm_weight,
                &self.p1.attn_norm,
                hidden,
                norm_eps,
            )?;
            dispatch_qat(
                batch,
                &self.p1.attn_norm,
                &self.p1_wq_a.activation,
                &self.p1_wq_a.scales,
                LAYER0_WQ_A_COLS as u32,
            )?;
            dispatch_fp8(
                batch,
                &self.wq_a_weight,
                &self.wq_a_scale,
                &self.p1_wq_a.activation,
                &self.p1_wq_a.scales,
                &self.p1_wq_a.fp32,
                LAYER0_WQ_A_ROWS as u32,
                LAYER0_WQ_A_COLS as u32,
                (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(
                batch,
                &self.p1_wq_a.fp32,
                &self.p1_wq_a.bf16,
                LAYER0_WQ_A_ROWS as u32,
            )?;
            dispatch_rms(
                batch,
                &self.p1_wq_a.bf16,
                &self.q_norm_weight,
                &self.p1_q_norm,
                q_lora,
                norm_eps,
            )?;
            dispatch_qat(
                batch,
                &self.p1_q_norm,
                &self.p1_wq_b.activation,
                &self.p1_wq_b.scales,
                q_lora,
            )?;
            dispatch_fp8(
                batch,
                &self.wq_b_weight,
                &self.wq_b_scale,
                &self.p1_wq_b.activation,
                &self.p1_wq_b.scales,
                &self.p1_wq_b.fp32,
                WQ_B_ROWS as u32,
                q_lora,
                (Q_LORA_RANK / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(
                batch,
                &self.p1_wq_b.fp32,
                &self.p1_wq_b.bf16,
                WQ_B_ROWS as u32,
            )?;
            dispatch_per_head(
                batch,
                &self.p1_wq_b.bf16,
                &self.p1_q_head,
                heads,
                head_dim,
                norm_eps,
            )?;
            dispatch_rope(
                batch,
                &self.p1_q_head,
                &self.rope_cos,
                &self.rope_sin,
                &self.p1_q_rope,
                heads,
                head_dim,
                rope_dim,
                forward,
            )?;
            dispatch_qat(
                batch,
                &self.p1.attn_norm,
                &self.p1_wkv.activation,
                &self.p1_wkv.scales,
                hidden,
            )?;
            dispatch_fp8(
                batch,
                &self.wkv_weight,
                &self.wkv_scale,
                &self.p1_wkv.activation,
                &self.p1_wkv.scales,
                &self.p1_wkv.fp32,
                WKV_ROWS as u32,
                hidden,
                (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(batch, &self.p1_wkv.fp32, &self.p1_wkv.bf16, WKV_ROWS as u32)?;
            dispatch_rms(
                batch,
                &self.p1_wkv.bf16,
                &self.kv_norm_weight,
                &self.p1_kv.norm,
                head_dim,
                norm_eps,
            )?;
            dispatch_kv_qat(
                batch,
                &self.p1_kv.norm,
                &self.p1_kv.qat,
                &self.p1_kv.activation,
                &self.p1_kv.scales,
                head_dim,
                rope_dim,
                kv_block,
            )?;
            dispatch_rope(
                batch,
                &self.p1_kv.qat,
                &self.rope_cos,
                &self.rope_sin,
                &self.p1_kv_rope,
                1,
                head_dim,
                rope_dim,
                forward,
            )?;
            dispatch_cache_write(
                batch,
                &self.p1_kv_rope,
                &self.causal_kv_cache_bf16,
                position1,
                head_dim,
                cache_capacity,
            )?;
            dispatch_sparse(
                batch,
                &self.p1_q_rope,
                &self.causal_kv_cache_bf16,
                &self.attn_sink,
                &self.p1_sparse,
                &self.p1_scores,
                &self.p1_denominators,
                heads,
                head_dim,
                cache_capacity,
                sparse_scale,
            )?;
            dispatch_rope(
                batch,
                &self.p1_sparse,
                &self.rope_cos,
                &self.rope_sin,
                &self.p1_derotated,
                heads,
                head_dim,
                rope_dim,
                inverse,
            )?;
            dispatch_wo_a(
                batch,
                &self.wo_a_weight,
                &self.wo_a_scale,
                &self.p1_derotated,
                &self.p1_wo_a,
                WO_A_ROWS as u32,
                WO_A_COLS as u32,
                (WO_A_COLS / ACT_QUANT_BLOCK) as u32,
                O_LORA_RANK as u32,
            )?;
            dispatch_qat(
                batch,
                &self.p1_wo_a,
                &self.p1_wo_b.activation,
                &self.p1_wo_b.scales,
                WO_B_COLS as u32,
            )?;
            dispatch_fp8(
                batch,
                &self.wo_b_weight,
                &self.wo_b_scale,
                &self.p1_wo_b.activation,
                &self.p1_wo_b.scales,
                &self.p1_wo_b.fp32,
                WO_B_ROWS as u32,
                WO_B_COLS as u32,
                (WO_B_COLS / ACT_QUANT_BLOCK) as u32,
            )?;
            dispatch_cast(
                batch,
                &self.p1_wo_b.fp32,
                &self.p1_wo_b.bf16,
                WO_B_ROWS as u32,
            )?;
            dispatch_hc_post(
                batch,
                &self.p1_wo_b.bf16,
                &self.p1.embed,
                &self.p1.post,
                &self.p1.comb,
                &self.attention_hc_post_bf16,
                hidden,
                hc_mult,
            )?;
            Ok(())
        })?;
        if timing.command_buffers != 1
            || timing.compute_encoders as usize != DSV4F_P4B_DEVICE_DISPATCHES
            || timing.compute_dispatches as usize != DSV4F_P4B_DEVICE_DISPATCHES
        {
            return Err(p4b_error(
                "P4B executor command topology differs from its declared ordered graph",
            ));
        }
        self.phase = DeepSeekV4P4bDevicePhase::Position1Complete;
        Ok(DeepSeekV4P4bDeviceExecution {
            phase: self.phase,
            parity_classification: self.parity_classification(),
            timing,
            actual_command_buffers: 1,
            actual_compute_encoders: DSV4F_P4B_DEVICE_DISPATCHES,
            actual_gpu_dispatches: DSV4F_P4B_DEVICE_DISPATCHES,
            actual_cpu_visible_waits: 1,
            host_intermediate_handoff_bytes: 0,
        })
    }

    /// Borrow the terminal device state for P7.  The caller must pass the
    /// same context used during `prepare`/`execute_position1`; no buffer is
    /// copied, mapped, or read back here.
    pub fn p7_attention_state<'a>(
        &'a self,
        metal: &'a MetalContext,
    ) -> Result<DeepSeekV4P7AttentionDeviceState<'a>> {
        self.check_context(metal)?;
        if self.phase != DeepSeekV4P4bDevicePhase::Position1Complete {
            return Err(p4b_error(
                "P7 state requested before P4B position-one completion",
            ));
        }
        DeepSeekV4P7AttentionDeviceState::position1(
            metal,
            &self.attention_hc_post_bf16,
            &self.causal_kv_cache_bf16,
            LAYER0,
            POSITION1_TOKEN_ID as u32,
        )
    }

    fn check_context(&self, metal: &MetalContext) -> Result<()> {
        if context_queue_identity(metal) != self.context_queue_identity {
            return Err(p4b_error(
                "P4B executor requires the caller-owned preparation context/queue",
            ));
        }
        Ok(())
    }
}

fn new_hc_state(metal: &MetalContext, embed: &[u8]) -> Result<HcState> {
    Ok(HcState {
        embed: metal.new_buffer_with_bytes_checked(embed)?,
        reduced: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<u16>())?,
        rsqrt: metal.new_buffer_checked(size_of::<f32>())?,
        mixes: metal.new_buffer_checked(HC_MIX_WIDTH * size_of::<f32>())?,
        pre: metal.new_buffer_checked(HC_MULT * size_of::<f32>())?,
        post: metal.new_buffer_checked(HC_MULT * size_of::<f32>())?,
        comb: metal.new_buffer_checked(HC_MULT * HC_MULT * size_of::<f32>())?,
        attn_norm: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<u16>())?,
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

#[allow(clippy::too_many_arguments)]
fn dispatch_hc(
    batch: &mut CommandBatch<'_>,
    state: &HcState,
    hc_fn: &metal::Buffer,
    hc_scale: &metal::Buffer,
    hc_base: &metal::Buffer,
    hidden: u32,
    hc_mult: u32,
    mix_width: u32,
    sinkhorn: u32,
    norm_eps: f32,
    hc_eps: f32,
) -> Result<()> {
    batch.dispatch_threads(HC_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
        e.set_buffer(0, Some(&state.embed), 0);
        e.set_buffer(1, Some(hc_fn), 0);
        e.set_buffer(2, Some(hc_scale), 0);
        e.set_buffer(3, Some(hc_base), 0);
        e.set_buffer(4, Some(&state.reduced), 0);
        e.set_buffer(5, Some(&state.rsqrt), 0);
        e.set_buffer(6, Some(&state.mixes), 0);
        e.set_buffer(7, Some(&state.pre), 0);
        e.set_buffer(8, Some(&state.post), 0);
        e.set_buffer(9, Some(&state.comb), 0);
        set_u32(e, 10, &hidden);
        set_u32(e, 11, &hc_mult);
        set_u32(e, 12, &mix_width);
        set_u32(e, 13, &sinkhorn);
        set_f32(e, 14, &norm_eps);
        set_f32(e, 15, &hc_eps);
    })
}

fn dispatch_rms(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    weight: &metal::Buffer,
    output: &metal::Buffer,
    width: u32,
    eps: f32,
) -> Result<()> {
    batch.dispatch_threads(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(weight), 0);
        e.set_buffer(2, Some(output), 0);
        set_u32(e, 3, &width);
        set_f32(e, 4, &eps);
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
    eps: f32,
) -> Result<()> {
    batch.dispatch_threads(PER_HEAD_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(output), 0);
        set_u32(e, 2, &heads);
        set_u32(e, 3, &head_dim);
        set_f32(e, 4, &eps);
    })
}

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

fn dispatch_rope(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    cos: &metal::Buffer,
    sin: &metal::Buffer,
    output: &metal::Buffer,
    rows: u32,
    head_dim: u32,
    rope_dim: u32,
    inverse: u32,
) -> Result<()> {
    batch.dispatch_threads(ROPE_KERNEL, (rows * head_dim / 2, 1, 1), (256, 1, 1), |e| {
        e.set_buffer(0, Some(input), 0);
        e.set_buffer(1, Some(cos), 0);
        e.set_buffer(2, Some(sin), 0);
        e.set_buffer(3, Some(output), 0);
        set_u32(e, 4, &rows);
        set_u32(e, 5, &head_dim);
        set_u32(e, 6, &rope_dim);
        set_u32(e, 7, &inverse);
    })
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
fn dispatch_sparse(
    batch: &mut CommandBatch<'_>,
    q: &metal::Buffer,
    cache: &metal::Buffer,
    sink: &metal::Buffer,
    output: &metal::Buffer,
    scores: &metal::Buffer,
    denominators: &metal::Buffer,
    heads: u32,
    head_dim: u32,
    capacity: u32,
    scale: f32,
) -> Result<()> {
    batch.dispatch_threads(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |e| {
        e.set_buffer(0, Some(q), 0);
        e.set_buffer(1, Some(cache), 0);
        e.set_buffer(2, Some(sink), 0);
        e.set_buffer(3, Some(output), 0);
        e.set_buffer(4, Some(scores), 0);
        e.set_buffer(5, Some(denominators), 0);
        set_u32(e, 6, &heads);
        set_u32(e, 7, &head_dim);
        set_u32(e, 8, &capacity);
        set_f32(e, 9, &scale);
    })
}

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

fn dispatch_hc_post(
    batch: &mut CommandBatch<'_>,
    attention: &metal::Buffer,
    embed: &metal::Buffer,
    post: &metal::Buffer,
    comb: &metal::Buffer,
    output: &metal::Buffer,
    hidden: u32,
    hc_mult: u32,
) -> Result<()> {
    batch.dispatch_threads(HC_POST_KERNEL, (hidden * hc_mult, 1, 1), (256, 1, 1), |e| {
        e.set_buffer(0, Some(attention), 0);
        e.set_buffer(1, Some(embed), 0);
        e.set_buffer(2, Some(post), 0);
        e.set_buffer(3, Some(comb), 0);
        e.set_buffer(4, Some(output), 0);
        set_u32(e, 5, &hidden);
        set_u32(e, 6, &hc_mult);
    })
}

fn embedding_row(reader: &DeepSeekV4FullStreamReader, token_id: u64) -> Result<Vec<u8>> {
    let metadata = reader.tensor_metadata(EMBED_WEIGHT)?;
    if metadata.dtype != "BF16"
        || metadata.shape.as_slice() != [VOCAB_SIZE, HIDDEN_SIZE as u64]
        || token_id >= VOCAB_SIZE
    {
        return Err(p4b_error("P4B embedding source geometry/token changed"));
    }
    let bytes = HIDDEN_SIZE * size_of::<u16>();
    let start = token_id
        .checked_mul(bytes as u64)
        .ok_or_else(|| p4b_error("P4B embedding row overflow"))?;
    reader.read_verified_range(EMBED_WEIGHT, start..start + bytes as u64, bytes)
}

fn full(reader: &DeepSeekV4FullStreamReader, name: &str) -> Result<Vec<u8>> {
    let metadata = reader.tensor_metadata(name)?;
    reader.read_verified_full(name, metadata.bytes as usize)
}

fn fp8_pair(
    reader: &DeepSeekV4FullStreamReader,
    weight: &str,
    scale: &str,
    rows: usize,
    cols: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let pair = reader.native_scale_pair(weight)?;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.scale.name != scale
        || pair.weight.shape.as_slice() != [rows as u64, cols as u64]
        || pair.scale.shape.as_slice()
            != [
                (rows / ACT_QUANT_BLOCK) as u64,
                (cols / ACT_QUANT_BLOCK) as u64,
            ]
    {
        return Err(p4b_error(format!(
            "P4B FP8 source pair changed for {weight}"
        )));
    }
    Ok((
        reader.read_verified_full(weight, pair.weight.bytes as usize)?,
        reader.read_verified_full(scale, pair.scale.bytes as usize)?,
    ))
}

#[allow(clippy::too_many_arguments)]
fn validate_geometry(
    embed0: &[u8],
    embed1: &[u8],
    hc_fn: &[u8],
    hc_scale: &[u8],
    hc_base: &[u8],
    attn_norm: &[u8],
    q_norm: &[u8],
    kv_norm: &[u8],
    sink: &[u8],
) -> Result<()> {
    if embed0.len() != HIDDEN_SIZE * size_of::<u16>()
        || embed1.len() != HIDDEN_SIZE * size_of::<u16>()
        || hc_fn.len() != HC_MIX_WIDTH * HC_MULT * HIDDEN_SIZE * size_of::<f32>()
        || hc_scale.len() != 3 * size_of::<f32>()
        || hc_base.len() != HC_MIX_WIDTH * size_of::<f32>()
        || attn_norm.len() != HIDDEN_SIZE * size_of::<u16>()
        || q_norm.len() != Q_LORA_RANK * size_of::<u16>()
        || kv_norm.len() != HEAD_DIM * size_of::<u16>()
        || sink.len() != NUM_HEADS * size_of::<f32>()
    {
        return Err(p4b_error("P4B static tensor geometry changed"));
    }
    Ok(())
}

fn f32bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
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
fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn p4b_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 reusable P4B device graph: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p4b_terminal_classification_cannot_claim_exact_storage() {
        let classification = DeepSeekV4P4bParityClassification::NumericParityV21Only;
        assert_eq!(classification.as_str(), "NUMERIC_PARITY_V2_1_ONLY");
        assert!(!classification.is_exact_storage());
    }

    #[test]
    fn p4b_p7_handoff_geometry_is_exactly_bounded() {
        assert_eq!(DSV4F_P4B_ATTENTION_HC_POST_BF16_BYTES, 4 * 4096 * 2);
        assert_eq!(DSV4F_P4B_CAUSAL_KV_BF16_BYTES, 2 * 512 * 2);
        assert_eq!(DSV4F_P4B_DEVICE_DISPATCHES, 33);
    }
}
