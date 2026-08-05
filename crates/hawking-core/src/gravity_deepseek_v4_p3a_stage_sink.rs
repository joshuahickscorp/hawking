//! Reusable artifact-backed Metal sink for the bounded DeepSeek-V4 P3A rung.
//!
//! This consumes scheduler leases for one position-zero, token-zero layer-0
//! path and executes the source-native mHC-attention-pre / RMSNorm / Q chain
//! entirely through device intermediates. It is deliberately *not* an Engine,
//! a full attention implementation, a causal loop, HCLI surface, parity
//! promotion, or TPS measurement. The scalar P3A authority kernels remain a
//! diagnostic foundation until a separately parity-gated parallel replacement
//! is admitted.

use crate::metal::{MetalBatchTiming, MetalDispatchTiming};

/// Exact number of completed device dispatches in the bounded P3A chain:
/// mHC + attention norm + WQ-A QAT/projection/cast + Q norm + WQ-B
/// QAT/projection/cast + per-head Q norm.
pub const DSV4F_P3A_Q_CHAIN_DISPATCHES: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4P3aStageSinkPhase {
    AwaitMhcAttentionControl,
    AwaitWqAControl,
    AwaitWqBControl,
    Complete,
}

impl DeepSeekV4P3aStageSinkPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AwaitMhcAttentionControl => "await_mhc_attention_control",
            Self::AwaitWqAControl => "await_wq_a_control",
            Self::AwaitWqBControl => "await_wq_b_control",
            Self::Complete => "complete_bounded_p3a_q_chain",
        }
    }
}

/// Bounded, completed device-work accounting. These counters are populated
/// only after each command buffer completes with a real GPU timestamp.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeepSeekV4P3aStageSinkCounters {
    pub source_control_leases_consumed: usize,
    pub static_artifact_control_reads: usize,
    pub source_upload_bytes: usize,
    pub actual_command_buffers: usize,
    pub actual_compute_encoders: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
    pub gpu_timestamped_dispatches: usize,
    pub aggregate_gpu_duration_us: u64,
    pub host_intermediate_handoff_bytes: usize,
}

/// Hash-bound source payload which was uploaded directly to the device for a
/// single bounded dispatch. It is metadata only; no source bytes are retained
/// in reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P3aSourcePayload {
    pub label: &'static str,
    pub bytes: usize,
    pub sha256: String,
}

/// One completed, timestamped Metal dispatch in the P3A source chain.
#[derive(Debug, Clone)]
pub struct DeepSeekV4P3aStageDispatch {
    pub stage: &'static str,
    pub kernel: &'static str,
    pub timing: MetalDispatchTiming,
    pub bytes_read: usize,
    pub bytes_written: usize,
    pub source_payloads: Vec<DeepSeekV4P3aSourcePayload>,
}

/// Static artifact and source-code anchors bound before any P3A source upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P3aSourceBindings {
    pub artifact_manifest_seal_sha256: String,
    pub repository: String,
    pub revision: String,
    pub token_id: u32,
    pub position: usize,
    pub embedding_sha256: String,
    pub attn_norm_sha256: String,
    pub q_norm_sha256: String,
    pub inference_model_py_sha256: String,
    pub inference_kernel_py_sha256: String,
    pub inference_config_json_sha256: String,
    pub model_config_json_sha256: String,
    pub inference_convert_py_sha256: String,
}

/// Finalized bounded-stage report. This is an execution receipt surface, not
/// a model-runtime result: device Q output remains device-resident for a
/// future P4 stage and is never read back by this sink.
#[derive(Debug, Clone)]
pub struct DeepSeekV4P3aStageSinkReport {
    pub phase: DeepSeekV4P3aStageSinkPhase,
    pub source_bindings: DeepSeekV4P3aSourceBindings,
    pub counters: DeepSeekV4P3aStageSinkCounters,
    pub dispatches: Vec<DeepSeekV4P3aStageDispatch>,
    pub buffers_created: usize,
    pub device_bytes_allocated: usize,
    pub trace_samples: usize,
    pub source_parent_retained: bool,
    pub q_head_output_device_bytes: usize,
    pub runtime_boundary: &'static str,
}

/// State of the optional exact-P4A continuation.  It is deliberately a
/// continuation of the P3A device buffers rather than a second attention
/// implementation: all source leases are collected first, then the already
/// verified ordered P4A operator chain is issued against those same buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4P4aContinuationPhase {
    AwaitWkvControl,
    AwaitWoAControl,
    AwaitWoBControl,
    Complete,
}

impl DeepSeekV4P4aContinuationPhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AwaitWkvControl => "await_wkv_control",
            Self::AwaitWoAControl => "await_wo_a_control",
            Self::AwaitWoBControl => "await_wo_b_control",
            Self::Complete => "complete_bounded_p4a_attention_continuation",
        }
    }
}

/// Completed accounting for the continuation command graph.  A continuation
/// is not a decode loop: this is one bounded BOS/position-zero attention
/// transaction, with no readback of any intermediate activation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeepSeekV4P4aContinuationCounters {
    pub source_control_leases_consumed: usize,
    pub static_artifact_control_reads: usize,
    pub source_upload_bytes: usize,
    pub actual_command_buffers: usize,
    pub actual_compute_encoders: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
    pub gpu_timestamped_command_buffers: usize,
    pub aggregate_gpu_duration_us: u64,
    pub host_intermediate_handoff_bytes: usize,
}

/// Static source bindings used only by the verified P4A continuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P4aContinuationSourceBindings {
    pub kv_norm_sha256: String,
    pub attention_sink_sha256: String,
    pub p4a_authority_receipt_seal_sha256: &'static str,
    pub p4a_topology_receipt_seal_sha256: &'static str,
}

/// Final bounded P3A-to-P4A continuation report.  `batch_timing` is an
/// aggregate completed-command-buffer timestamp; it deliberately does not
/// pretend to provide per-encoder GPU timing inside the one-CB topology.
#[derive(Debug, Clone)]
pub struct DeepSeekV4P4aContinuationReport {
    pub phase: DeepSeekV4P4aContinuationPhase,
    pub p3a_source_bindings: DeepSeekV4P3aSourceBindings,
    pub p4a_source_bindings: DeepSeekV4P4aContinuationSourceBindings,
    pub p3a_counters: DeepSeekV4P3aStageSinkCounters,
    pub p4a_counters: DeepSeekV4P4aContinuationCounters,
    pub p3a_dispatches: Vec<DeepSeekV4P3aStageDispatch>,
    pub p4a_kernel_order: Vec<&'static str>,
    pub p4a_batch_timing: MetalBatchTiming,
    pub buffers_created: usize,
    pub device_bytes_allocated: usize,
    pub trace_samples: usize,
    pub source_parent_retained: bool,
    pub attention_output_device_bytes: usize,
    pub runtime_boundary: &'static str,
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use std::mem::size_of;

    use sha2::{Digest, Sha256};

    use crate::gravity_deepseek_v4::NativeScalePairKind;
    use crate::gravity_deepseek_v4_act_quant::{
        ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
    };
    use crate::gravity_deepseek_v4_execution_context::{
        DeepSeekV4ControlPayload, DeepSeekV4ExecutionContext, DeepSeekV4MhcBranch,
        DeepSeekV4PreparedDecodeInput,
    };
    use crate::gravity_deepseek_v4_layer0_attention::{
        verify_layer0_attention_source_anchors, HEAD_DIM, KV_QAT_BLOCK, LAYER0_ATTN_SINK,
        LAYER0_KV_NORM_WEIGHT, LAYER0_Q_NORM_WEIGHT, LAYER0_WKV_SCALE, LAYER0_WKV_WEIGHT,
        LAYER0_WO_A_SCALE, LAYER0_WO_A_WEIGHT, LAYER0_WO_B_SCALE, LAYER0_WO_B_WEIGHT,
        LAYER0_WQ_B_SCALE, LAYER0_WQ_B_WEIGHT, NON_ROPE_HEAD_DIM, NUM_HEADS, O_LORA_RANK,
        Q_LORA_RANK, ROPE_HEAD_DIM, WKV_ROWS, WO_A_COLS, WO_A_ROWS, WO_B_COLS, WO_B_ROWS,
        WQ_B_ROWS,
    };
    use crate::gravity_deepseek_v4_layer0_prefix::{
        HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS, HIDDEN_SIZE,
        LAYER0_ATTN_NORM_WEIGHT, LAYER0_HC_ATTN_BASE, LAYER0_HC_ATTN_FN, LAYER0_HC_ATTN_SCALE,
        PREFIX_TOKEN_ID, RMS_NORM_EPS,
    };
    use crate::gravity_deepseek_v4_layer_scheduler::{
        DeepSeekV4LayerPreparationStage, DeepSeekV4NativeStage, DeepSeekV4NativeStageConsumption,
        DeepSeekV4NativeStageSink,
    };
    use crate::gravity_deepseek_v4_runtime_spine::{
        DeepSeekV4StagedNativePair, DeepSeekV4StagedTensor,
    };
    use crate::metal::MetalContext;
    use crate::{Error, Result};

    const HC_KERNEL: &str = "deepseek_v4_p3a_layer0_hc_attn_pre_bos_authority";
    const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
    const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
    const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
    const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
    const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
    const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
    /// Production ratio-0 path: growing-KV supersedes the fixed position-0 specialization.
    const SPARSE_KERNEL: &str = "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority";
    const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
    const HC_POST_KERNEL: &str = "deepseek_v4_p4a_hc_attn_post_authority";
    const P4A_AUTHORITY_RECEIPT_SEAL_SHA256: &str =
        "9ae79af84a7184e4f6e9d7f4bd02639cb6277b6ffc01076476530ca32e12f570";
    const P4A_TOPOLOGY_RECEIPT_SEAL_SHA256: &str =
        "b3e76911d6175dc96534793bfb87aac018ddf33802d676e9c253708b16267aba";

    const P4A_CONTINUATION_KERNEL_ORDER: [&str; 11] = [
        QAT_KERNEL,
        FP8_KERNEL,
        CAST_KERNEL,
        RMS_KERNEL,
        KV_QAT_KERNEL,
        SPARSE_KERNEL,
        WO_A_KERNEL,
        QAT_KERNEL,
        FP8_KERNEL,
        CAST_KERNEL,
        HC_POST_KERNEL,
    ];

    struct DeepSeekV4P4aContinuationState {
        phase: DeepSeekV4P4aContinuationPhase,
        source_bindings: DeepSeekV4P4aContinuationSourceBindings,
        counters: DeepSeekV4P4aContinuationCounters,
        kv_norm_weight_buffer: metal::Buffer,
        attention_sink_buffer: metal::Buffer,
        wkv_weight_buffer: Option<metal::Buffer>,
        wkv_scale_buffer: Option<metal::Buffer>,
        wo_a_weight_buffer: Option<metal::Buffer>,
        wo_a_scale_buffer: Option<metal::Buffer>,
        wo_b_weight_buffer: Option<metal::Buffer>,
        wo_b_scale_buffer: Option<metal::Buffer>,
        wkv_activation_buffer: metal::Buffer,
        wkv_activation_scale_buffer: metal::Buffer,
        wkv_fp32_output_buffer: metal::Buffer,
        wkv_bf16_output_buffer: metal::Buffer,
        kv_norm_output_buffer: metal::Buffer,
        kv_qat_output_buffer: metal::Buffer,
        kv_qat_activation_buffer: metal::Buffer,
        kv_qat_scale_buffer: metal::Buffer,
        sparse_output_buffer: metal::Buffer,
        sparse_scores_buffer: metal::Buffer,
        sparse_denominators_buffer: metal::Buffer,
        wo_a_output_buffer: metal::Buffer,
        wo_b_activation_buffer: metal::Buffer,
        wo_b_activation_scale_buffer: metal::Buffer,
        wo_b_fp32_output_buffer: metal::Buffer,
        wo_b_bf16_output_buffer: metal::Buffer,
        hc_final_output_buffer: metal::Buffer,
        batch_timing: Option<MetalBatchTiming>,
    }

    /// Concrete reusable Metal sink for the layer-0 P3A pre-attention/Q rung.
    /// It owns all device intermediates and never reads one back to host.
    pub struct DeepSeekV4P3aMetalStageSink {
        metal: MetalContext,
        phase: DeepSeekV4P3aStageSinkPhase,
        source_bindings: DeepSeekV4P3aSourceBindings,
        counters: DeepSeekV4P3aStageSinkCounters,
        dispatches: Vec<DeepSeekV4P3aStageDispatch>,
        final_report: Option<DeepSeekV4P3aStageSinkReport>,
        p4a_final_report: Option<DeepSeekV4P4aContinuationReport>,
        p4a: Option<DeepSeekV4P4aContinuationState>,
        embed_buffer: metal::Buffer,
        attn_norm_weight_buffer: metal::Buffer,
        q_norm_weight_buffer: metal::Buffer,
        hc_reduced_buffer: metal::Buffer,
        hc_rsqrt_buffer: metal::Buffer,
        hc_mixes_buffer: metal::Buffer,
        hc_pre_buffer: metal::Buffer,
        hc_post_buffer: metal::Buffer,
        hc_comb_buffer: metal::Buffer,
        attn_norm_output_buffer: metal::Buffer,
        wq_a_activation_buffer: metal::Buffer,
        wq_a_activation_scale_buffer: metal::Buffer,
        wq_a_fp32_output_buffer: metal::Buffer,
        wq_a_bf16_output_buffer: metal::Buffer,
        q_norm_output_buffer: metal::Buffer,
        wq_b_activation_buffer: metal::Buffer,
        wq_b_activation_scale_buffer: metal::Buffer,
        wq_b_fp32_output_buffer: metal::Buffer,
        wq_b_bf16_output_buffer: metal::Buffer,
        q_head_output_buffer: metal::Buffer,
    }

    impl DeepSeekV4P3aMetalStageSink {
        /// Bind one already-prepared, source-authenticated BOS embedding to the
        /// bounded P3A sink. This intentionally admits only layer 0, position
        /// 0, token 0—the scope independently parity-checked by P3A.
        pub fn new(
            context: &DeepSeekV4ExecutionContext,
            prepared: &DeepSeekV4PreparedDecodeInput,
        ) -> Result<Self> {
            Self::new_with_compile_profile(context, prepared, false)
        }

        /// Internal constructor for the normal P3A surface and the explicit
        /// diagnostic-only strict-math candidate.  `false` deliberately
        /// preserves the default `MetalContext::new_with_trace` behavior.
        fn new_with_compile_profile(
            context: &DeepSeekV4ExecutionContext,
            prepared: &DeepSeekV4PreparedDecodeInput,
            strict_math_candidate: bool,
        ) -> Result<Self> {
            if prepared.token_id != PREFIX_TOKEN_ID as u32 || prepared.position != 0 {
                return Err(sink_error(
                    "P3A sink admits only tokenizer-bound BOS token 0 at position 0",
                ));
            }
            let anchors = verify_layer0_attention_source_anchors(context.spine().reader())?;
            let embedding = embedding_bytes(context, prepared)?;
            let attn_norm = stage_static_bf16(context, LAYER0_ATTN_NORM_WEIGHT, HIDDEN_SIZE)?;
            let q_norm = stage_static_bf16(context, LAYER0_Q_NORM_WEIGHT, Q_LORA_RANK)?;
            let reader = context.spine().reader();
            let source_bindings = DeepSeekV4P3aSourceBindings {
                artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
                repository: reader.source_identity().repository.to_owned(),
                revision: reader.source_identity().revision.to_owned(),
                token_id: prepared.token_id,
                position: prepared.position,
                embedding_sha256: sha256(&embedding),
                attn_norm_sha256: sha256(&attn_norm),
                q_norm_sha256: sha256(&q_norm),
                inference_model_py_sha256: anchors.prefix.act_quant.inference_model_py_sha256,
                inference_kernel_py_sha256: anchors.prefix.act_quant.inference_kernel_py_sha256,
                inference_config_json_sha256: anchors.prefix.act_quant.inference_config_json_sha256,
                model_config_json_sha256: anchors.prefix.act_quant.model_config_json_sha256,
                inference_convert_py_sha256: anchors.inference_convert_py_sha256,
            };

            let metal = if strict_math_candidate {
                MetalContext::new_with_trace_strict_math(true)?
            } else {
                MetalContext::new_with_trace(true)?
            };
            for kernel in [
                HC_KERNEL,
                RMS_KERNEL,
                QAT_KERNEL,
                FP8_KERNEL,
                CAST_KERNEL,
                PER_HEAD_KERNEL,
            ] {
                let _ = metal.pipeline(kernel)?;
            }

            let row_bytes = checked_mul(HIDDEN_SIZE, size_of::<u16>(), "hidden BF16 bytes")?;
            let wq_a_scale_cols = LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK;
            let wq_b_scale_cols = Q_LORA_RANK / ACT_QUANT_BLOCK;
            let mut counters = DeepSeekV4P3aStageSinkCounters {
                static_artifact_control_reads: 2,
                source_upload_bytes: checked_add(
                    checked_add(embedding.len(), attn_norm.len(), "static P3A source bytes")?,
                    q_norm.len(),
                    "static P3A source bytes",
                )?,
                ..DeepSeekV4P3aStageSinkCounters::default()
            };
            // The only host writes are authenticated source uploads. Every
            // later activation/intermediate remains in a device buffer.
            counters.host_intermediate_handoff_bytes = 0;

            Ok(Self {
                embed_buffer: metal.new_buffer_with_bytes_checked(&embedding)?,
                attn_norm_weight_buffer: metal.new_buffer_with_bytes_checked(&attn_norm)?,
                q_norm_weight_buffer: metal.new_buffer_with_bytes_checked(&q_norm)?,
                hc_reduced_buffer: metal.new_buffer_checked(row_bytes)?,
                hc_rsqrt_buffer: metal.new_buffer_checked(size_of::<f32>())?,
                hc_mixes_buffer: metal.new_buffer_checked(checked_mul(
                    HC_MIX_WIDTH,
                    size_of::<f32>(),
                    "mHC mixes bytes",
                )?)?,
                hc_pre_buffer: metal.new_buffer_checked(checked_mul(
                    HC_MULT,
                    size_of::<f32>(),
                    "mHC pre bytes",
                )?)?,
                hc_post_buffer: metal.new_buffer_checked(checked_mul(
                    HC_MULT,
                    size_of::<f32>(),
                    "mHC post bytes",
                )?)?,
                hc_comb_buffer: metal.new_buffer_checked(checked_mul(
                    HC_MULT * HC_MULT,
                    size_of::<f32>(),
                    "mHC Sinkhorn bytes",
                )?)?,
                attn_norm_output_buffer: metal.new_buffer_checked(row_bytes)?,
                wq_a_activation_buffer: metal.new_buffer_checked(LAYER0_WQ_A_COLS)?,
                wq_a_activation_scale_buffer: metal.new_buffer_checked(wq_a_scale_cols)?,
                wq_a_fp32_output_buffer: metal.new_buffer_checked(checked_mul(
                    LAYER0_WQ_A_ROWS,
                    size_of::<f32>(),
                    "WQ-A FP32 bytes",
                )?)?,
                wq_a_bf16_output_buffer: metal.new_buffer_checked(checked_mul(
                    LAYER0_WQ_A_ROWS,
                    size_of::<u16>(),
                    "WQ-A BF16 bytes",
                )?)?,
                q_norm_output_buffer: metal.new_buffer_checked(checked_mul(
                    Q_LORA_RANK,
                    size_of::<u16>(),
                    "Q norm BF16 bytes",
                )?)?,
                wq_b_activation_buffer: metal.new_buffer_checked(Q_LORA_RANK)?,
                wq_b_activation_scale_buffer: metal.new_buffer_checked(wq_b_scale_cols)?,
                wq_b_fp32_output_buffer: metal.new_buffer_checked(checked_mul(
                    WQ_B_ROWS,
                    size_of::<f32>(),
                    "WQ-B FP32 bytes",
                )?)?,
                wq_b_bf16_output_buffer: metal.new_buffer_checked(checked_mul(
                    WQ_B_ROWS,
                    size_of::<u16>(),
                    "WQ-B BF16 bytes",
                )?)?,
                q_head_output_buffer: metal.new_buffer_checked(checked_mul(
                    WQ_B_ROWS,
                    size_of::<u16>(),
                    "per-head Q BF16 bytes",
                )?)?,
                metal,
                phase: DeepSeekV4P3aStageSinkPhase::AwaitMhcAttentionControl,
                source_bindings,
                counters,
                dispatches: Vec::with_capacity(DSV4F_P3A_Q_CHAIN_DISPATCHES),
                final_report: None,
                p4a_final_report: None,
                p4a: None,
            })
        }

        /// Construct the P3A sink with the exact, independently verified P4A
        /// continuation enabled.  This preserves the P4A source-kernel
        /// sequence and its one-command-buffer ordered-encoder topology; it
        /// does not create a second attention algorithm or a runtime loop.
        ///
        /// The continuation intentionally starts only after the scheduler has
        /// supplied WKV, WO-A, and WO-B leases.  Their source bytes are bound
        /// directly into device buffers while each lease is live, so no host
        /// activation/intermediate bridge is introduced.
        pub fn new_for_verified_p4a_continuation(
            context: &DeepSeekV4ExecutionContext,
            prepared: &DeepSeekV4PreparedDecodeInput,
        ) -> Result<Self> {
            Self::new_for_verified_p4a_continuation_with_compile_profile(context, prepared, false)
        }

        /// Construct the same bounded P3A/P4A graph with an explicitly
        /// strict-math Metal library. This is a diagnostic comparison surface
        /// only: it compiles all linked shader functions with fast math off,
        /// does not change the default constructor, and carries no promotion
        /// or runtime implication.
        pub fn new_for_verified_p4a_continuation_strict_math(
            context: &DeepSeekV4ExecutionContext,
            prepared: &DeepSeekV4PreparedDecodeInput,
        ) -> Result<Self> {
            Self::new_for_verified_p4a_continuation_with_compile_profile(context, prepared, true)
        }

        fn new_for_verified_p4a_continuation_with_compile_profile(
            context: &DeepSeekV4ExecutionContext,
            prepared: &DeepSeekV4PreparedDecodeInput,
            strict_math_candidate: bool,
        ) -> Result<Self> {
            let mut sink =
                Self::new_with_compile_profile(context, prepared, strict_math_candidate)?;
            let kv_norm = stage_static_bf16(context, LAYER0_KV_NORM_WEIGHT, HEAD_DIM)?;
            let attention_sink = stage_static_f32(context, LAYER0_ATTN_SINK, NUM_HEADS)?;
            for kernel in [KV_QAT_KERNEL, SPARSE_KERNEL, WO_A_KERNEL, HC_POST_KERNEL] {
                let _ = sink.metal.pipeline(kernel)?;
            }
            let p4a_source_bytes = checked_add(
                kv_norm.len(),
                attention_sink.len(),
                "P4A static source upload bytes",
            )?;
            let p4a = DeepSeekV4P4aContinuationState {
                phase: DeepSeekV4P4aContinuationPhase::AwaitWkvControl,
                source_bindings: DeepSeekV4P4aContinuationSourceBindings {
                    kv_norm_sha256: sha256(&kv_norm),
                    attention_sink_sha256: sha256(&attention_sink),
                    p4a_authority_receipt_seal_sha256: P4A_AUTHORITY_RECEIPT_SEAL_SHA256,
                    p4a_topology_receipt_seal_sha256: P4A_TOPOLOGY_RECEIPT_SEAL_SHA256,
                },
                counters: DeepSeekV4P4aContinuationCounters {
                    static_artifact_control_reads: 2,
                    source_upload_bytes: p4a_source_bytes,
                    ..DeepSeekV4P4aContinuationCounters::default()
                },
                kv_norm_weight_buffer: sink.metal.new_buffer_with_bytes_checked(&kv_norm)?,
                attention_sink_buffer: sink.metal.new_buffer_with_bytes_checked(&attention_sink)?,
                wkv_weight_buffer: None,
                wkv_scale_buffer: None,
                wo_a_weight_buffer: None,
                wo_a_scale_buffer: None,
                wo_b_weight_buffer: None,
                wo_b_scale_buffer: None,
                wkv_activation_buffer: sink.metal.new_buffer_checked(HIDDEN_SIZE)?,
                wkv_activation_scale_buffer: sink
                    .metal
                    .new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?,
                wkv_fp32_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    WKV_ROWS,
                    size_of::<f32>(),
                    "WKV FP32 bytes",
                )?)?,
                wkv_bf16_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    WKV_ROWS,
                    size_of::<u16>(),
                    "WKV BF16 bytes",
                )?)?,
                kv_norm_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    HEAD_DIM,
                    size_of::<u16>(),
                    "KV norm BF16 bytes",
                )?)?,
                kv_qat_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    HEAD_DIM,
                    size_of::<u16>(),
                    "KV QAT BF16 bytes",
                )?)?,
                kv_qat_activation_buffer: sink.metal.new_buffer_checked(NON_ROPE_HEAD_DIM)?,
                kv_qat_scale_buffer: sink
                    .metal
                    .new_buffer_checked(NON_ROPE_HEAD_DIM / KV_QAT_BLOCK)?,
                sparse_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    WQ_B_ROWS,
                    size_of::<u16>(),
                    "sparse attention BF16 bytes",
                )?)?,
                sparse_scores_buffer: sink.metal.new_buffer_checked(checked_mul(
                    NUM_HEADS,
                    size_of::<f32>(),
                    "sparse score bytes",
                )?)?,
                sparse_denominators_buffer: sink.metal.new_buffer_checked(checked_mul(
                    NUM_HEADS,
                    size_of::<f32>(),
                    "sparse denominator bytes",
                )?)?,
                wo_a_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    WO_A_ROWS,
                    size_of::<u16>(),
                    "WO-A BF16 bytes",
                )?)?,
                wo_b_activation_buffer: sink.metal.new_buffer_checked(WO_B_COLS)?,
                wo_b_activation_scale_buffer: sink
                    .metal
                    .new_buffer_checked(WO_B_COLS / ACT_QUANT_BLOCK)?,
                wo_b_fp32_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    WO_B_ROWS,
                    size_of::<f32>(),
                    "WO-B FP32 bytes",
                )?)?,
                wo_b_bf16_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    WO_B_ROWS,
                    size_of::<u16>(),
                    "WO-B BF16 bytes",
                )?)?,
                hc_final_output_buffer: sink.metal.new_buffer_checked(checked_mul(
                    checked_mul(HC_MULT, HIDDEN_SIZE, "mHC final BF16 elements")?,
                    size_of::<u16>(),
                    "mHC final BF16 bytes",
                )?)?,
                batch_timing: None,
            };
            sink.p4a = Some(p4a);
            Ok(sink)
        }

        pub const fn phase(&self) -> DeepSeekV4P3aStageSinkPhase {
            self.phase
        }

        pub fn counters(&self) -> &DeepSeekV4P3aStageSinkCounters {
            &self.counters
        }

        pub fn dispatches(&self) -> &[DeepSeekV4P3aStageDispatch] {
            &self.dispatches
        }

        pub fn source_bindings(&self) -> &DeepSeekV4P3aSourceBindings {
            &self.source_bindings
        }

        /// The caller-owned context for the bounded P3A/P4A device chain.
        /// A later bounded continuation may borrow this exact context together
        /// with [`Self::p4a_attention_output_buffer`] so the attention result
        /// never crosses a host activation boundary.  This is intentionally a
        /// borrow only: the P3A sink remains the owner of its trace and state.
        pub fn metal_context(&self) -> &MetalContext {
            &self.metal
        }

        pub fn p4a_continuation_phase(&self) -> Option<DeepSeekV4P4aContinuationPhase> {
            self.p4a.as_ref().map(|p4a| p4a.phase)
        }

        /// Device-resident P3A Q output for the future attention rung. This
        /// sink never reads this buffer back to host.
        pub fn q_head_output_buffer(&self) -> &metal::Buffer {
            &self.q_head_output_buffer
        }

        pub const fn q_head_output_device_bytes(&self) -> usize {
            WQ_B_ROWS * size_of::<u16>()
        }

        /// Final layer-0 attention residual surface from the P4A continuation.
        /// It remains device-resident and must not be represented as a CPU
        /// activation handoff by a later FFN sink.
        pub fn p4a_attention_output_buffer(&self) -> Result<&metal::Buffer> {
            let p4a = self
                .p4a
                .as_ref()
                .ok_or_else(|| sink_error("P4A continuation was not enabled for this P3A sink"))?;
            if p4a.phase != DeepSeekV4P4aContinuationPhase::Complete {
                return Err(sink_error(
                    "P4A attention output requested before its exact continuation completed",
                ));
            }
            Ok(&p4a.hc_final_output_buffer)
        }

        pub const fn p4a_attention_output_device_bytes(&self) -> usize {
            HC_MULT * HIDDEN_SIZE * size_of::<u16>()
        }

        /// Freeze the bounded chain and return its exact completed Metal
        /// accounting. This requires all ten P3A authority dispatches and no
        /// host intermediate handoff; it is still not full-layer parity.
        pub fn finish(&mut self) -> Result<DeepSeekV4P3aStageSinkReport> {
            if let Some(report) = &self.final_report {
                return Ok(report.clone());
            }
            if self.p4a.is_some() {
                return Err(sink_error(
                    "P4A continuation is enabled; finalize the composed path with finish_p4a_continuation instead",
                ));
            }
            if self.phase != DeepSeekV4P3aStageSinkPhase::Complete {
                return Err(sink_error(
                    "cannot finalize P3A sink before mHC/WQ-A/WQ-B chain completes",
                ));
            }
            if self.dispatches.len() != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_gpu_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_command_buffers != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_compute_encoders != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_cpu_visible_waits != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.gpu_timestamped_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.host_intermediate_handoff_bytes != 0
            {
                return Err(sink_error(
                    "P3A sink has incomplete or non-device-resident dispatch accounting",
                ));
            }
            let (buffers_created, device_bytes_allocated, commits) = self.metal.drain_stats();
            let trace_samples = self.metal.drain_trace().len();
            if commits != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || trace_samples != DSV4F_P3A_Q_CHAIN_DISPATCHES
            {
                return Err(sink_error(
                    "P3A sink trace/commit count differs from completed dispatch count",
                ));
            }
            let report = DeepSeekV4P3aStageSinkReport {
                phase: self.phase,
                source_bindings: self.source_bindings.clone(),
                counters: self.counters.clone(),
                dispatches: self.dispatches.clone(),
                buffers_created,
                device_bytes_allocated,
                trace_samples,
                source_parent_retained: false,
                q_head_output_device_bytes: self.q_head_output_device_bytes(),
                runtime_boundary: "bounded layer-0 BOS P3A mHC/norm/Q device chain only; no full attention, KV, routing, MoE, causal loop, Engine, HCLI, parity promotion, or TPS claim",
            };
            self.final_report = Some(report.clone());
            Ok(report)
        }

        /// Freeze the composed P3A -> exact-P4A bounded attention path.  The
        /// P3A portion retains its ten timestamped authority dispatches; the
        /// verified P4A continuation retains its selected one-CB / eleven
        /// ordered-encoder continuation shape.  This is not a claim that the
        /// *combined* graph has promoted the P4A whole-chain 1-CB topology:
        /// P3A's independently timed baseline remains visible here.
        pub fn finish_p4a_continuation(&mut self) -> Result<DeepSeekV4P4aContinuationReport> {
            if let Some(report) = &self.p4a_final_report {
                return Ok(report.clone());
            }
            if self.final_report.is_some() {
                return Err(sink_error(
                    "P3A-only finalization prevents a later P4A continuation",
                ));
            }
            if self.phase != DeepSeekV4P3aStageSinkPhase::Complete {
                return Err(sink_error(
                    "cannot finalize P4A continuation before the P3A Q chain completes",
                ));
            }
            let p4a = self
                .p4a
                .as_ref()
                .ok_or_else(|| sink_error("P4A continuation was not enabled"))?;
            if p4a.phase != DeepSeekV4P4aContinuationPhase::Complete {
                return Err(sink_error(
                    "cannot finalize P4A continuation before WKV/WO-A/WO-B complete",
                ));
            }
            let timing = p4a
                .batch_timing
                .ok_or_else(|| sink_error("P4A continuation has no completed batch timing"))?;
            if self.dispatches.len() != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_gpu_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_command_buffers != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_compute_encoders != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.actual_cpu_visible_waits != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.gpu_timestamped_dispatches != DSV4F_P3A_Q_CHAIN_DISPATCHES
                || self.counters.host_intermediate_handoff_bytes != 0
                || p4a.counters.actual_command_buffers != 1
                || p4a.counters.actual_compute_encoders != P4A_CONTINUATION_KERNEL_ORDER.len()
                || p4a.counters.actual_gpu_dispatches != P4A_CONTINUATION_KERNEL_ORDER.len()
                || p4a.counters.actual_cpu_visible_waits != 1
                || p4a.counters.gpu_timestamped_command_buffers != 1
                || p4a.counters.host_intermediate_handoff_bytes != 0
                || timing.command_buffers != 1
                || timing.compute_encoders as usize != P4A_CONTINUATION_KERNEL_ORDER.len()
                || timing.compute_dispatches as usize != P4A_CONTINUATION_KERNEL_ORDER.len()
                || timing.gpu_duration_us.is_none()
            {
                return Err(sink_error(
                    "P3A/P4A continuation has incomplete, untimestamped, or host-handed-off accounting",
                ));
            }
            let (buffers_created, device_bytes_allocated, commits) = self.metal.drain_stats();
            let trace_samples = self.metal.drain_trace().len();
            let expected_commands = DSV4F_P3A_Q_CHAIN_DISPATCHES + 1;
            if commits != expected_commands || trace_samples != expected_commands {
                return Err(sink_error(
                    "P3A/P4A trace/commit count differs from the measured composed topology",
                ));
            }
            let report = DeepSeekV4P4aContinuationReport {
                phase: p4a.phase,
                p3a_source_bindings: self.source_bindings.clone(),
                p4a_source_bindings: p4a.source_bindings.clone(),
                p3a_counters: self.counters.clone(),
                p4a_counters: p4a.counters.clone(),
                p3a_dispatches: self.dispatches.clone(),
                p4a_kernel_order: P4A_CONTINUATION_KERNEL_ORDER.to_vec(),
                p4a_batch_timing: timing,
                buffers_created,
                device_bytes_allocated,
                trace_samples,
                source_parent_retained: false,
                attention_output_device_bytes: self.p4a_attention_output_device_bytes(),
                runtime_boundary: "bounded layer-0 BOS P3A-to-P4A device path only; no FFN, KV persistence across positions, router, MoE, causal loop, Engine, HCLI, combined-path parity promotion, or TPS claim",
            };
            self.p4a_final_report = Some(report.clone());
            Ok(report)
        }

        fn consume_control(
            &mut self,
            step: &crate::gravity_deepseek_v4_layer_scheduler::DeepSeekV4LayerPreparationStep,
            payload: &DeepSeekV4ControlPayload,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            if step.layer != 0 || step.token_position != 0 {
                return Err(sink_error(
                    "P3A sink rejects a non-layer-0 or non-position-0 scheduler stage",
                ));
            }
            if self.phase == DeepSeekV4P3aStageSinkPhase::Complete && self.p4a.is_some() {
                return self.consume_p4a_continuation_control(step, payload);
            }
            match (self.phase, step.stage, payload) {
                (
                    DeepSeekV4P3aStageSinkPhase::AwaitMhcAttentionControl,
                    DeepSeekV4LayerPreparationStage::MhcAttentionControl,
                    DeepSeekV4ControlPayload::MhcControl {
                        layer,
                        branch: DeepSeekV4MhcBranch::Attention,
                        tensors,
                    },
                ) if *layer == 0 => self.consume_mhc_attention_control(tensors),
                (
                    DeepSeekV4P3aStageSinkPhase::AwaitWqAControl,
                    DeepSeekV4LayerPreparationStage::AttentionControl(
                        crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WqA,
                    ),
                    DeepSeekV4ControlPayload::NativePair(pair),
                ) => self.consume_wq_a_control(pair),
                (
                    DeepSeekV4P3aStageSinkPhase::AwaitWqBControl,
                    DeepSeekV4LayerPreparationStage::AttentionControl(
                        crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WqB,
                    ),
                    DeepSeekV4ControlPayload::NativePair(pair),
                ) => self.consume_wq_b_control(pair),
                _ => Err(sink_error(format!(
                    "P3A sink phase {} cannot consume scheduler stage {}",
                    self.phase.as_str(),
                    step.stage.as_str()
                ))),
            }
        }

        fn consume_p4a_continuation_control(
            &mut self,
            step: &crate::gravity_deepseek_v4_layer_scheduler::DeepSeekV4LayerPreparationStep,
            payload: &DeepSeekV4ControlPayload,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            if self.p4a_final_report.is_some() {
                return Err(sink_error(
                    "cannot consume a P4A control after continuation finalization",
                ));
            }
            let phase = self
                .p4a
                .as_ref()
                .ok_or_else(|| sink_error("P4A continuation was not enabled"))?
                .phase;
            match (phase, step.stage, payload) {
                (
                    DeepSeekV4P4aContinuationPhase::AwaitWkvControl,
                    DeepSeekV4LayerPreparationStage::AttentionControl(
                        crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::Wkv,
                    ),
                    DeepSeekV4ControlPayload::NativePair(pair),
                ) => self.consume_p4a_wkv_control(pair),
                (
                    DeepSeekV4P4aContinuationPhase::AwaitWoAControl,
                    DeepSeekV4LayerPreparationStage::AttentionControl(
                        crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WoA,
                    ),
                    DeepSeekV4ControlPayload::NativePair(pair),
                ) => self.consume_p4a_wo_a_control(pair),
                (
                    DeepSeekV4P4aContinuationPhase::AwaitWoBControl,
                    DeepSeekV4LayerPreparationStage::AttentionControl(
                        crate::gravity_deepseek_v4_runtime_spine::DeepSeekV4ControlProjection::WoB,
                    ),
                    DeepSeekV4ControlPayload::NativePair(pair),
                ) => self.consume_p4a_wo_b_control(pair),
                _ => Err(sink_error(format!(
                    "P4A continuation phase {} cannot consume scheduler stage {}",
                    phase.as_str(),
                    step.stage.as_str()
                ))),
            }
        }

        fn consume_p4a_wkv_control(
            &mut self,
            pair: &DeepSeekV4StagedNativePair,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            validate_fp8_pair(
                pair,
                LAYER0_WKV_WEIGHT,
                LAYER0_WKV_SCALE,
                WKV_ROWS,
                HIDDEN_SIZE,
            )?;
            let payloads = vec![
                source_payload("wkv_weight", &pair.weight.bytes),
                source_payload("wkv_scale", &pair.scale.bytes),
            ];
            let weight = self
                .metal
                .new_buffer_with_bytes_checked(&pair.weight.bytes)?;
            let scale = self
                .metal
                .new_buffer_with_bytes_checked(&pair.scale.bytes)?;
            let p4a = self
                .p4a
                .as_mut()
                .ok_or_else(|| sink_error("P4A continuation was not enabled"))?;
            if p4a.phase != DeepSeekV4P4aContinuationPhase::AwaitWkvControl {
                return Err(sink_error("WKV source control arrived out of P4A order"));
            }
            note_p4a_control_source_uploads(&mut p4a.counters, &payloads)?;
            p4a.wkv_weight_buffer = Some(weight);
            p4a.wkv_scale_buffer = Some(scale);
            p4a.phase = DeepSeekV4P4aContinuationPhase::AwaitWoAControl;
            Ok(DeepSeekV4NativeStageConsumption::default())
        }

        fn consume_p4a_wo_a_control(
            &mut self,
            pair: &DeepSeekV4StagedNativePair,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            validate_fp8_pair(
                pair,
                LAYER0_WO_A_WEIGHT,
                LAYER0_WO_A_SCALE,
                WO_A_ROWS,
                WO_A_COLS,
            )?;
            let payloads = vec![
                source_payload("wo_a_weight", &pair.weight.bytes),
                source_payload("wo_a_scale", &pair.scale.bytes),
            ];
            let weight = self
                .metal
                .new_buffer_with_bytes_checked(&pair.weight.bytes)?;
            let scale = self
                .metal
                .new_buffer_with_bytes_checked(&pair.scale.bytes)?;
            let p4a = self
                .p4a
                .as_mut()
                .ok_or_else(|| sink_error("P4A continuation was not enabled"))?;
            if p4a.phase != DeepSeekV4P4aContinuationPhase::AwaitWoAControl {
                return Err(sink_error("WO-A source control arrived out of P4A order"));
            }
            note_p4a_control_source_uploads(&mut p4a.counters, &payloads)?;
            p4a.wo_a_weight_buffer = Some(weight);
            p4a.wo_a_scale_buffer = Some(scale);
            p4a.phase = DeepSeekV4P4aContinuationPhase::AwaitWoBControl;
            Ok(DeepSeekV4NativeStageConsumption::default())
        }

        fn consume_p4a_wo_b_control(
            &mut self,
            pair: &DeepSeekV4StagedNativePair,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            validate_fp8_pair(
                pair,
                LAYER0_WO_B_WEIGHT,
                LAYER0_WO_B_SCALE,
                WO_B_ROWS,
                WO_B_COLS,
            )?;
            let payloads = vec![
                source_payload("wo_b_weight", &pair.weight.bytes),
                source_payload("wo_b_scale", &pair.scale.bytes),
            ];
            let weight = self
                .metal
                .new_buffer_with_bytes_checked(&pair.weight.bytes)?;
            let scale = self
                .metal
                .new_buffer_with_bytes_checked(&pair.scale.bytes)?;
            {
                let p4a = self
                    .p4a
                    .as_mut()
                    .ok_or_else(|| sink_error("P4A continuation was not enabled"))?;
                if p4a.phase != DeepSeekV4P4aContinuationPhase::AwaitWoBControl {
                    return Err(sink_error("WO-B source control arrived out of P4A order"));
                }
                note_p4a_control_source_uploads(&mut p4a.counters, &payloads)?;
                p4a.wo_b_weight_buffer = Some(weight);
                p4a.wo_b_scale_buffer = Some(scale);
            }
            self.execute_verified_p4a_continuation_batch()
        }

        fn execute_verified_p4a_continuation_batch(
            &mut self,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            let (
                kv_norm_weight,
                attention_sink,
                wkv_weight,
                wkv_scale,
                wo_a_weight,
                wo_a_scale,
                wo_b_weight,
                wo_b_scale,
                wkv_activation,
                wkv_activation_scale,
                wkv_fp32_output,
                wkv_bf16_output,
                kv_norm_output,
                kv_qat_output,
                kv_qat_activation,
                kv_qat_scale,
                sparse_output,
                sparse_scores,
                sparse_denominators,
                wo_a_output,
                wo_b_activation,
                wo_b_activation_scale,
                wo_b_fp32_output,
                wo_b_bf16_output,
                hc_final_output,
            ) = {
                let p4a = self
                    .p4a
                    .as_ref()
                    .ok_or_else(|| sink_error("P4A continuation was not enabled"))?;
                if p4a.phase != DeepSeekV4P4aContinuationPhase::AwaitWoBControl {
                    return Err(sink_error(
                        "cannot issue P4A batch before all three continuation controls are staged",
                    ));
                }
                (
                    p4a.kv_norm_weight_buffer.clone(),
                    p4a.attention_sink_buffer.clone(),
                    p4a.wkv_weight_buffer
                        .as_ref()
                        .ok_or_else(|| sink_error("P4A missing WKV weight buffer"))?
                        .clone(),
                    p4a.wkv_scale_buffer
                        .as_ref()
                        .ok_or_else(|| sink_error("P4A missing WKV scale buffer"))?
                        .clone(),
                    p4a.wo_a_weight_buffer
                        .as_ref()
                        .ok_or_else(|| sink_error("P4A missing WO-A weight buffer"))?
                        .clone(),
                    p4a.wo_a_scale_buffer
                        .as_ref()
                        .ok_or_else(|| sink_error("P4A missing WO-A scale buffer"))?
                        .clone(),
                    p4a.wo_b_weight_buffer
                        .as_ref()
                        .ok_or_else(|| sink_error("P4A missing WO-B weight buffer"))?
                        .clone(),
                    p4a.wo_b_scale_buffer
                        .as_ref()
                        .ok_or_else(|| sink_error("P4A missing WO-B scale buffer"))?
                        .clone(),
                    p4a.wkv_activation_buffer.clone(),
                    p4a.wkv_activation_scale_buffer.clone(),
                    p4a.wkv_fp32_output_buffer.clone(),
                    p4a.wkv_bf16_output_buffer.clone(),
                    p4a.kv_norm_output_buffer.clone(),
                    p4a.kv_qat_output_buffer.clone(),
                    p4a.kv_qat_activation_buffer.clone(),
                    p4a.kv_qat_scale_buffer.clone(),
                    p4a.sparse_output_buffer.clone(),
                    p4a.sparse_scores_buffer.clone(),
                    p4a.sparse_denominators_buffer.clone(),
                    p4a.wo_a_output_buffer.clone(),
                    p4a.wo_b_activation_buffer.clone(),
                    p4a.wo_b_activation_scale_buffer.clone(),
                    p4a.wo_b_fp32_output_buffer.clone(),
                    p4a.wo_b_bf16_output_buffer.clone(),
                    p4a.hc_final_output_buffer.clone(),
                )
            };
            let embed = self.embed_buffer.clone();
            let attn_norm_output = self.attn_norm_output_buffer.clone();
            let q_head_output = self.q_head_output_buffer.clone();
            let hc_post = self.hc_post_buffer.clone();
            let hc_comb = self.hc_comb_buffer.clone();
            let hidden = HIDDEN_SIZE as u32;
            let heads = NUM_HEADS as u32;
            let head_dim = HEAD_DIM as u32;
            let norm_eps = RMS_NORM_EPS;
            let wkv_rows = WKV_ROWS as u32;
            let wkv_scale_cols = (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u32;
            let kv_block = KV_QAT_BLOCK as u32;
            let rope_dim = ROPE_HEAD_DIM as u32;
            let sparse_scale = (HEAD_DIM as f32).powf(-0.5);
            let wo_a_rows = WO_A_ROWS as u32;
            let wo_a_cols = WO_A_COLS as u32;
            let wo_a_scale_cols = (WO_A_COLS / ACT_QUANT_BLOCK) as u32;
            let o_rank = O_LORA_RANK as u32;
            let wo_b_rows = WO_B_ROWS as u32;
            let wo_b_cols = WO_B_COLS as u32;
            let wo_b_scale_cols = (WO_B_COLS / ACT_QUANT_BLOCK) as u32;
            let hc_mult = HC_MULT as u32;

            // This is the exact P4A continuation ordering from the sealed
            // authority/topology receipts.  It deliberately uses 11 ordered
            // compute encoders in one command buffer—not a same-encoder or
            // concurrent-dispatch assertion.
            let timing = self.metal.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(
                    QAT_KERNEL,
                    (wkv_scale_cols, 1, 1),
                    (32, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&attn_norm_output), 0);
                        encoder.set_buffer(1, Some(&wkv_activation), 0);
                        encoder.set_buffer(2, Some(&wkv_activation_scale), 0);
                        set_u32(encoder, 3, &hidden);
                    },
                )?;
                batch.dispatch_threads(FP8_KERNEL, (wkv_rows, 1, 1), (256, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&wkv_weight), 0);
                    encoder.set_buffer(1, Some(&wkv_scale), 0);
                    encoder.set_buffer(2, Some(&wkv_activation), 0);
                    encoder.set_buffer(3, Some(&wkv_activation_scale), 0);
                    encoder.set_buffer(4, Some(&wkv_fp32_output), 0);
                    set_u32(encoder, 5, &wkv_rows);
                    set_u32(encoder, 6, &hidden);
                    set_u32(encoder, 7, &wkv_scale_cols);
                })?;
                batch.dispatch_threads(CAST_KERNEL, (wkv_rows, 1, 1), (256, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&wkv_fp32_output), 0);
                    encoder.set_buffer(1, Some(&wkv_bf16_output), 0);
                    set_u32(encoder, 2, &wkv_rows);
                })?;
                batch.dispatch_threads(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&wkv_bf16_output), 0);
                    encoder.set_buffer(1, Some(&kv_norm_weight), 0);
                    encoder.set_buffer(2, Some(&kv_norm_output), 0);
                    set_u32(encoder, 3, &head_dim);
                    set_f32(encoder, 4, &norm_eps);
                })?;
                batch.dispatch_threads(
                    KV_QAT_KERNEL,
                    (NON_ROPE_HEAD_DIM as u32 / kv_block, 1, 1),
                    (32, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&kv_norm_output), 0);
                        encoder.set_buffer(1, Some(&kv_qat_output), 0);
                        encoder.set_buffer(2, Some(&kv_qat_activation), 0);
                        encoder.set_buffer(3, Some(&kv_qat_scale), 0);
                        set_u32(encoder, 4, &head_dim);
                        set_u32(encoder, 5, &rope_dim);
                        set_u32(encoder, 6, &kv_block);
                    },
                )?;
                // BOS/position-0: treat the single KV QAT row as a 1-slot growing cache.
                let cache_capacity = 1u32;
                let valid_kv_count = 1u32;
                let max_score_slots = 1u32;
                batch.dispatch_threads(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&q_head_output), 0);
                    encoder.set_buffer(1, Some(&kv_qat_output), 0);
                    encoder.set_buffer(2, Some(&attention_sink), 0);
                    encoder.set_buffer(3, Some(&sparse_output), 0);
                    encoder.set_buffer(4, Some(&sparse_scores), 0);
                    encoder.set_buffer(5, Some(&sparse_denominators), 0);
                    set_u32(encoder, 6, &heads);
                    set_u32(encoder, 7, &head_dim);
                    set_u32(encoder, 8, &cache_capacity);
                    set_u32(encoder, 9, &valid_kv_count);
                    set_u32(encoder, 10, &max_score_slots);
                    set_f32(encoder, 11, &sparse_scale);
                })?;
                batch.dispatch_threads(WO_A_KERNEL, (wo_a_rows, 1, 1), (256, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&wo_a_weight), 0);
                    encoder.set_buffer(1, Some(&wo_a_scale), 0);
                    encoder.set_buffer(2, Some(&sparse_output), 0);
                    encoder.set_buffer(3, Some(&wo_a_output), 0);
                    set_u32(encoder, 4, &wo_a_rows);
                    set_u32(encoder, 5, &wo_a_cols);
                    set_u32(encoder, 6, &wo_a_scale_cols);
                    set_u32(encoder, 7, &o_rank);
                })?;
                batch.dispatch_threads(
                    QAT_KERNEL,
                    (wo_b_scale_cols, 1, 1),
                    (32, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&wo_a_output), 0);
                        encoder.set_buffer(1, Some(&wo_b_activation), 0);
                        encoder.set_buffer(2, Some(&wo_b_activation_scale), 0);
                        set_u32(encoder, 3, &wo_b_cols);
                    },
                )?;
                batch.dispatch_threads(FP8_KERNEL, (wo_b_rows, 1, 1), (256, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&wo_b_weight), 0);
                    encoder.set_buffer(1, Some(&wo_b_scale), 0);
                    encoder.set_buffer(2, Some(&wo_b_activation), 0);
                    encoder.set_buffer(3, Some(&wo_b_activation_scale), 0);
                    encoder.set_buffer(4, Some(&wo_b_fp32_output), 0);
                    set_u32(encoder, 5, &wo_b_rows);
                    set_u32(encoder, 6, &wo_b_cols);
                    set_u32(encoder, 7, &wo_b_scale_cols);
                })?;
                batch.dispatch_threads(CAST_KERNEL, (wo_b_rows, 1, 1), (256, 1, 1), |encoder| {
                    encoder.set_buffer(0, Some(&wo_b_fp32_output), 0);
                    encoder.set_buffer(1, Some(&wo_b_bf16_output), 0);
                    set_u32(encoder, 2, &wo_b_rows);
                })?;
                batch.dispatch_threads(
                    HC_POST_KERNEL,
                    (hidden * hc_mult, 1, 1),
                    (256, 1, 1),
                    |encoder| {
                        encoder.set_buffer(0, Some(&wo_b_bf16_output), 0);
                        encoder.set_buffer(1, Some(&embed), 0);
                        encoder.set_buffer(2, Some(&hc_post), 0);
                        encoder.set_buffer(3, Some(&hc_comb), 0);
                        encoder.set_buffer(4, Some(&hc_final_output), 0);
                        set_u32(encoder, 5, &hidden);
                        set_u32(encoder, 6, &hc_mult);
                    },
                )?;
                Ok(())
            })?;
            if timing.command_buffers != 1
                || timing.compute_encoders as usize != P4A_CONTINUATION_KERNEL_ORDER.len()
                || timing.compute_dispatches as usize != P4A_CONTINUATION_KERNEL_ORDER.len()
            {
                return Err(sink_error(
                    "verified P4A continuation did not preserve its one-CB ordered topology",
                ));
            }
            let gpu_duration_us = timing.gpu_duration_us.ok_or_else(|| {
                sink_error("P4A continuation completed without a usable GPU timestamp")
            })?;
            let p4a = self
                .p4a
                .as_mut()
                .ok_or_else(|| sink_error("P4A continuation was not enabled"))?;
            p4a.counters.actual_command_buffers = 1;
            p4a.counters.actual_compute_encoders = timing.compute_encoders as usize;
            p4a.counters.actual_gpu_dispatches = timing.compute_dispatches as usize;
            p4a.counters.actual_cpu_visible_waits = 1;
            p4a.counters.gpu_timestamped_command_buffers = 1;
            p4a.counters.aggregate_gpu_duration_us = gpu_duration_us;
            p4a.counters.host_intermediate_handoff_bytes = 0;
            p4a.batch_timing = Some(timing);
            p4a.phase = DeepSeekV4P4aContinuationPhase::Complete;
            Ok(DeepSeekV4NativeStageConsumption {
                actual_command_buffers: 1,
                actual_compute_encoders: P4A_CONTINUATION_KERNEL_ORDER.len(),
                actual_gpu_dispatches: P4A_CONTINUATION_KERNEL_ORDER.len(),
                actual_cpu_visible_waits: 1,
                host_intermediate_handoff_bytes: 0,
            })
        }

        fn consume_mhc_attention_control(
            &mut self,
            tensors: &[DeepSeekV4StagedTensor; 3],
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            let counter_start = self.counters.clone();
            validate_tensor(
                &tensors[0],
                LAYER0_HC_ATTN_FN,
                "F32",
                &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
            )?;
            validate_tensor(
                &tensors[1],
                LAYER0_HC_ATTN_BASE,
                "F32",
                &[HC_MIX_WIDTH as u64],
            )?;
            validate_tensor(&tensors[2], LAYER0_HC_ATTN_SCALE, "F32", &[3])?;
            let source_payloads = vec![
                source_payload("hc_attn_fn", &tensors[0].bytes),
                source_payload("hc_attn_base", &tensors[1].bytes),
                source_payload("hc_attn_scale", &tensors[2].bytes),
            ];
            self.note_control_lease_source_uploads(&source_payloads)?;
            let hc_fn_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&tensors[0].bytes)?;
            let hc_base_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&tensors[1].bytes)?;
            let hc_scale_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&tensors[2].bytes)?;
            let hidden = HIDDEN_SIZE as u32;
            let hc_mult = HC_MULT as u32;
            let mix_width = HC_MIX_WIDTH as u32;
            let sinkhorn_iters = HC_SINKHORN_ITERS as u32;
            let norm_eps = RMS_NORM_EPS;
            let hc_eps = HC_EPS;
            let timing =
                self.metal
                    .dispatch_threads_timed(HC_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                        encoder.set_buffer(0, Some(&self.embed_buffer), 0);
                        encoder.set_buffer(1, Some(&hc_fn_buffer), 0);
                        encoder.set_buffer(2, Some(&hc_scale_buffer), 0);
                        encoder.set_buffer(3, Some(&hc_base_buffer), 0);
                        encoder.set_buffer(4, Some(&self.hc_reduced_buffer), 0);
                        encoder.set_buffer(5, Some(&self.hc_rsqrt_buffer), 0);
                        encoder.set_buffer(6, Some(&self.hc_mixes_buffer), 0);
                        encoder.set_buffer(7, Some(&self.hc_pre_buffer), 0);
                        encoder.set_buffer(8, Some(&self.hc_post_buffer), 0);
                        encoder.set_buffer(9, Some(&self.hc_comb_buffer), 0);
                        set_u32(encoder, 10, &hidden);
                        set_u32(encoder, 11, &hc_mult);
                        set_u32(encoder, 12, &mix_width);
                        set_u32(encoder, 13, &sinkhorn_iters);
                        set_f32(encoder, 14, &norm_eps);
                        set_f32(encoder, 15, &hc_eps);
                    })?;
            self.record_dispatch(
                "mhc_attn_pre_sinkhorn",
                HC_KERNEL,
                timing,
                tensors[0].bytes.len()
                    + tensors[1].bytes.len()
                    + tensors[2].bytes.len()
                    + HIDDEN_SIZE * HC_MULT * size_of::<u16>(),
                HIDDEN_SIZE * size_of::<u16>()
                    + size_of::<f32>()
                    + HC_MIX_WIDTH * size_of::<f32>()
                    + (HC_MULT * 2 + HC_MULT * HC_MULT) * size_of::<f32>(),
                source_payloads,
            )?;

            let norm_eps = RMS_NORM_EPS;
            let timing =
                self.metal
                    .dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                        encoder.set_buffer(0, Some(&self.hc_reduced_buffer), 0);
                        encoder.set_buffer(1, Some(&self.attn_norm_weight_buffer), 0);
                        encoder.set_buffer(2, Some(&self.attn_norm_output_buffer), 0);
                        set_u32(encoder, 3, &hidden);
                        set_f32(encoder, 4, &norm_eps);
                    })?;
            self.record_dispatch(
                "attn_rmsnorm",
                RMS_KERNEL,
                timing,
                HIDDEN_SIZE * size_of::<u16>() * 2,
                HIDDEN_SIZE * size_of::<u16>(),
                Vec::new(),
            )?;
            self.phase = DeepSeekV4P3aStageSinkPhase::AwaitWqAControl;
            self.consumption_delta(&counter_start)
        }

        fn consume_wq_a_control(
            &mut self,
            pair: &DeepSeekV4StagedNativePair,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            validate_fp8_pair(
                pair,
                LAYER0_WQ_A_WEIGHT,
                LAYER0_WQ_A_SCALE,
                LAYER0_WQ_A_ROWS,
                LAYER0_WQ_A_COLS,
            )?;
            let source_payloads = vec![
                source_payload("wq_a_weight", &pair.weight.bytes),
                source_payload("wq_a_scale", &pair.scale.bytes),
            ];
            let counter_start = self.counters.clone();
            self.note_control_lease_source_uploads(&source_payloads)?;
            let weight_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&pair.weight.bytes)?;
            let scale_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&pair.scale.bytes)?;
            let rows = LAYER0_WQ_A_ROWS as u32;
            let cols = LAYER0_WQ_A_COLS as u32;
            let scale_cols = (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u32;
            let timing = self.metal.dispatch_threads_timed(
                QAT_KERNEL,
                (scale_cols, 1, 1),
                (32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.attn_norm_output_buffer), 0);
                    encoder.set_buffer(1, Some(&self.wq_a_activation_buffer), 0);
                    encoder.set_buffer(2, Some(&self.wq_a_activation_scale_buffer), 0);
                    set_u32(encoder, 3, &cols);
                },
            )?;
            self.record_dispatch(
                "wq_a_act_quant",
                QAT_KERNEL,
                timing,
                HIDDEN_SIZE * size_of::<u16>(),
                LAYER0_WQ_A_COLS + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
                Vec::new(),
            )?;
            let timing = self.metal.dispatch_threads_timed(
                FP8_KERNEL,
                (rows, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weight_buffer), 0);
                    encoder.set_buffer(1, Some(&scale_buffer), 0);
                    encoder.set_buffer(2, Some(&self.wq_a_activation_buffer), 0);
                    encoder.set_buffer(3, Some(&self.wq_a_activation_scale_buffer), 0);
                    encoder.set_buffer(4, Some(&self.wq_a_fp32_output_buffer), 0);
                    set_u32(encoder, 5, &rows);
                    set_u32(encoder, 6, &cols);
                    set_u32(encoder, 7, &scale_cols);
                },
            )?;
            self.record_dispatch(
                "wq_a_fp8_matvec",
                FP8_KERNEL,
                timing,
                pair.weight.bytes.len()
                    + pair.scale.bytes.len()
                    + LAYER0_WQ_A_COLS
                    + LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK,
                LAYER0_WQ_A_ROWS * size_of::<f32>(),
                source_payloads,
            )?;
            let timing = self.metal.dispatch_threads_timed(
                CAST_KERNEL,
                (rows, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.wq_a_fp32_output_buffer), 0);
                    encoder.set_buffer(1, Some(&self.wq_a_bf16_output_buffer), 0);
                    set_u32(encoder, 2, &rows);
                },
            )?;
            self.record_dispatch(
                "wq_a_fp32_to_bf16",
                CAST_KERNEL,
                timing,
                LAYER0_WQ_A_ROWS * size_of::<f32>(),
                LAYER0_WQ_A_ROWS * size_of::<u16>(),
                Vec::new(),
            )?;
            self.phase = DeepSeekV4P3aStageSinkPhase::AwaitWqBControl;
            self.consumption_delta(&counter_start)
        }

        fn consume_wq_b_control(
            &mut self,
            pair: &DeepSeekV4StagedNativePair,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            validate_fp8_pair(
                pair,
                LAYER0_WQ_B_WEIGHT,
                LAYER0_WQ_B_SCALE,
                WQ_B_ROWS,
                Q_LORA_RANK,
            )?;
            let source_payloads = vec![
                source_payload("wq_b_weight", &pair.weight.bytes),
                source_payload("wq_b_scale", &pair.scale.bytes),
            ];
            let counter_start = self.counters.clone();
            let q_rank = Q_LORA_RANK as u32;
            let norm_eps = RMS_NORM_EPS;
            let timing =
                self.metal
                    .dispatch_threads_timed(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
                        encoder.set_buffer(0, Some(&self.wq_a_bf16_output_buffer), 0);
                        encoder.set_buffer(1, Some(&self.q_norm_weight_buffer), 0);
                        encoder.set_buffer(2, Some(&self.q_norm_output_buffer), 0);
                        set_u32(encoder, 3, &q_rank);
                        set_f32(encoder, 4, &norm_eps);
                    })?;
            self.record_dispatch(
                "q_rmsnorm",
                RMS_KERNEL,
                timing,
                Q_LORA_RANK * size_of::<u16>() * 2,
                Q_LORA_RANK * size_of::<u16>(),
                Vec::new(),
            )?;
            self.note_control_lease_source_uploads(&source_payloads)?;
            let weight_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&pair.weight.bytes)?;
            let scale_buffer = self
                .metal
                .new_buffer_with_bytes_checked(&pair.scale.bytes)?;
            let rows = WQ_B_ROWS as u32;
            let cols = Q_LORA_RANK as u32;
            let scale_cols = (Q_LORA_RANK / ACT_QUANT_BLOCK) as u32;
            let timing = self.metal.dispatch_threads_timed(
                QAT_KERNEL,
                (scale_cols, 1, 1),
                (32, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.q_norm_output_buffer), 0);
                    encoder.set_buffer(1, Some(&self.wq_b_activation_buffer), 0);
                    encoder.set_buffer(2, Some(&self.wq_b_activation_scale_buffer), 0);
                    set_u32(encoder, 3, &cols);
                },
            )?;
            self.record_dispatch(
                "wq_b_act_quant",
                QAT_KERNEL,
                timing,
                Q_LORA_RANK * size_of::<u16>(),
                Q_LORA_RANK + Q_LORA_RANK / ACT_QUANT_BLOCK,
                Vec::new(),
            )?;
            let timing = self.metal.dispatch_threads_timed(
                FP8_KERNEL,
                (rows, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&weight_buffer), 0);
                    encoder.set_buffer(1, Some(&scale_buffer), 0);
                    encoder.set_buffer(2, Some(&self.wq_b_activation_buffer), 0);
                    encoder.set_buffer(3, Some(&self.wq_b_activation_scale_buffer), 0);
                    encoder.set_buffer(4, Some(&self.wq_b_fp32_output_buffer), 0);
                    set_u32(encoder, 5, &rows);
                    set_u32(encoder, 6, &cols);
                    set_u32(encoder, 7, &scale_cols);
                },
            )?;
            self.record_dispatch(
                "wq_b_fp8_matvec",
                FP8_KERNEL,
                timing,
                pair.weight.bytes.len()
                    + pair.scale.bytes.len()
                    + Q_LORA_RANK
                    + Q_LORA_RANK / ACT_QUANT_BLOCK,
                WQ_B_ROWS * size_of::<f32>(),
                source_payloads,
            )?;
            let timing = self.metal.dispatch_threads_timed(
                CAST_KERNEL,
                (rows, 1, 1),
                (256, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.wq_b_fp32_output_buffer), 0);
                    encoder.set_buffer(1, Some(&self.wq_b_bf16_output_buffer), 0);
                    set_u32(encoder, 2, &rows);
                },
            )?;
            self.record_dispatch(
                "wq_b_fp32_to_bf16",
                CAST_KERNEL,
                timing,
                WQ_B_ROWS * size_of::<f32>(),
                WQ_B_ROWS * size_of::<u16>(),
                Vec::new(),
            )?;
            let heads = NUM_HEADS as u32;
            let head_dim = (WQ_B_ROWS / NUM_HEADS) as u32;
            let timing = self.metal.dispatch_threads_timed(
                PER_HEAD_KERNEL,
                (heads, 1, 1),
                (64, 1, 1),
                |encoder| {
                    encoder.set_buffer(0, Some(&self.wq_b_bf16_output_buffer), 0);
                    encoder.set_buffer(1, Some(&self.q_head_output_buffer), 0);
                    set_u32(encoder, 2, &heads);
                    set_u32(encoder, 3, &head_dim);
                    set_f32(encoder, 4, &norm_eps);
                },
            )?;
            self.record_dispatch(
                "q_per_head_rmsnorm",
                PER_HEAD_KERNEL,
                timing,
                WQ_B_ROWS * size_of::<u16>(),
                WQ_B_ROWS * size_of::<u16>(),
                Vec::new(),
            )?;
            self.phase = DeepSeekV4P3aStageSinkPhase::Complete;
            self.consumption_delta(&counter_start)
        }

        fn note_control_lease_source_uploads(
            &mut self,
            payloads: &[DeepSeekV4P3aSourcePayload],
        ) -> Result<()> {
            self.counters.source_control_leases_consumed = checked_add(
                self.counters.source_control_leases_consumed,
                1,
                "P3A consumed control lease count",
            )?;
            let bytes = payloads.iter().try_fold(0usize, |total, payload| {
                checked_add(total, payload.bytes, "P3A control source payload bytes")
            })?;
            self.counters.source_upload_bytes = checked_add(
                self.counters.source_upload_bytes,
                bytes,
                "P3A source upload bytes",
            )?;
            Ok(())
        }

        fn record_dispatch(
            &mut self,
            stage: &'static str,
            kernel: &'static str,
            timing: MetalDispatchTiming,
            bytes_read: usize,
            bytes_written: usize,
            source_payloads: Vec<DeepSeekV4P3aSourcePayload>,
        ) -> Result<()> {
            if timing.command_buffers != 1
                || timing.compute_encoders != 1
                || timing.compute_dispatches != 1
            {
                return Err(sink_error(format!(
                    "{stage} did not complete exactly one command buffer/encoder/dispatch"
                )));
            }
            let gpu_duration_us = timing.gpu_duration_us.ok_or_else(|| {
                sink_error(format!("{stage} completed without a usable GPU timestamp"))
            })?;
            self.counters.actual_command_buffers = checked_add(
                self.counters.actual_command_buffers,
                timing.command_buffers as usize,
                "P3A command buffers",
            )?;
            self.counters.actual_compute_encoders = checked_add(
                self.counters.actual_compute_encoders,
                timing.compute_encoders as usize,
                "P3A compute encoders",
            )?;
            self.counters.actual_gpu_dispatches = checked_add(
                self.counters.actual_gpu_dispatches,
                timing.compute_dispatches as usize,
                "P3A GPU dispatches",
            )?;
            self.counters.actual_cpu_visible_waits = checked_add(
                self.counters.actual_cpu_visible_waits,
                timing.command_buffers as usize,
                "P3A CPU-visible waits",
            )?;
            self.counters.gpu_timestamped_dispatches = checked_add(
                self.counters.gpu_timestamped_dispatches,
                1,
                "P3A timestamped dispatches",
            )?;
            self.counters.aggregate_gpu_duration_us = self
                .counters
                .aggregate_gpu_duration_us
                .checked_add(gpu_duration_us)
                .ok_or_else(|| sink_error("P3A aggregate GPU duration overflow"))?;
            self.dispatches.push(DeepSeekV4P3aStageDispatch {
                stage,
                kernel,
                timing,
                bytes_read,
                bytes_written,
                source_payloads,
            });
            Ok(())
        }

        fn consumption_delta(
            &self,
            start: &DeepSeekV4P3aStageSinkCounters,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            if self.counters.actual_command_buffers < start.actual_command_buffers
                || self.counters.actual_compute_encoders < start.actual_compute_encoders
                || self.counters.actual_gpu_dispatches < start.actual_gpu_dispatches
                || self.counters.actual_cpu_visible_waits < start.actual_cpu_visible_waits
            {
                return Err(sink_error("P3A consumption counter regressed"));
            }
            Ok(DeepSeekV4NativeStageConsumption {
                actual_command_buffers: self.counters.actual_command_buffers
                    - start.actual_command_buffers,
                actual_compute_encoders: self.counters.actual_compute_encoders
                    - start.actual_compute_encoders,
                actual_gpu_dispatches: self.counters.actual_gpu_dispatches
                    - start.actual_gpu_dispatches,
                actual_cpu_visible_waits: self.counters.actual_cpu_visible_waits
                    - start.actual_cpu_visible_waits,
                host_intermediate_handoff_bytes: 0,
            })
        }
    }

    impl DeepSeekV4NativeStageSink for DeepSeekV4P3aMetalStageSink {
        fn consume_native_stage(
            &mut self,
            stage: DeepSeekV4NativeStage<'_>,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            if self.final_report.is_some() {
                return Err(sink_error(
                    "cannot consume a stage after P3A sink finalization",
                ));
            }
            match stage {
                DeepSeekV4NativeStage::Control { step, payload } => {
                    self.consume_control(step, payload)
                }
                DeepSeekV4NativeStage::RoutedExpertWave { step, .. } => Err(sink_error(format!(
                    "P3A sink stops before routed experts and cannot consume {}",
                    step.stage.as_str()
                ))),
            }
        }
    }

    fn stage_static_bf16(
        context: &DeepSeekV4ExecutionContext,
        name: &str,
        elements: usize,
    ) -> Result<Vec<u8>> {
        let metadata = context.spine().reader().tensor_metadata(name)?;
        let expected_bytes = checked_mul(elements, size_of::<u16>(), "static BF16 bytes")?;
        if metadata.dtype != "BF16"
            || metadata.shape.as_slice() != [elements as u64]
            || metadata.bytes != expected_bytes as u64
        {
            return Err(sink_error(format!(
                "{name} source BF16 geometry differs from bounded P3A contract"
            )));
        }
        Ok(context
            .spine()
            .stage_base_tensor_range(name, 0..metadata.bytes, expected_bytes)?
            .bytes)
    }

    fn stage_static_f32(
        context: &DeepSeekV4ExecutionContext,
        name: &str,
        elements: usize,
    ) -> Result<Vec<u8>> {
        let metadata = context.spine().reader().tensor_metadata(name)?;
        let expected_bytes = checked_mul(elements, size_of::<f32>(), "static F32 bytes")?;
        if metadata.dtype != "F32"
            || metadata.shape.as_slice() != [elements as u64]
            || metadata.bytes != expected_bytes as u64
        {
            return Err(sink_error(format!(
                "{name} source F32 geometry differs from bounded P4A continuation contract"
            )));
        }
        Ok(context
            .spine()
            .stage_base_tensor_range(name, 0..metadata.bytes, expected_bytes)?
            .bytes)
    }

    fn note_p4a_control_source_uploads(
        counters: &mut DeepSeekV4P4aContinuationCounters,
        payloads: &[DeepSeekV4P3aSourcePayload],
    ) -> Result<()> {
        counters.source_control_leases_consumed = checked_add(
            counters.source_control_leases_consumed,
            1,
            "P4A consumed control lease count",
        )?;
        let bytes = payloads.iter().try_fold(0usize, |total, payload| {
            checked_add(total, payload.bytes, "P4A control source payload bytes")
        })?;
        counters.source_upload_bytes = checked_add(
            counters.source_upload_bytes,
            bytes,
            "P4A source upload bytes",
        )?;
        Ok(())
    }

    fn embedding_bytes(
        context: &DeepSeekV4ExecutionContext,
        prepared: &DeepSeekV4PreparedDecodeInput,
    ) -> Result<Vec<u8>> {
        match context.control_arena().get(prepared.embedding_lease)? {
            DeepSeekV4ControlPayload::EmbeddingRow {
                token_id,
                bf16_bits,
            } if *token_id == prepared.token_id
                && bf16_bits.len() == HIDDEN_SIZE
                && prepared.token_id == PREFIX_TOKEN_ID as u32 =>
            {
                Ok(bf16_le_bytes(bf16_bits))
            }
            DeepSeekV4ControlPayload::EmbeddingRow { .. } => Err(sink_error(
                "prepared embedding lease has unexpected token or hidden width",
            )),
            _ => Err(sink_error(
                "prepared embedding lease does not hold a BF16 embedding row",
            )),
        }
    }

    fn validate_tensor(
        tensor: &DeepSeekV4StagedTensor,
        name: &str,
        dtype: &str,
        shape: &[u64],
    ) -> Result<()> {
        if tensor.name != name
            || tensor.dtype != dtype
            || tensor.shape.as_slice() != shape
            || tensor.bytes.is_empty()
        {
            return Err(sink_error(format!(
                "{name} source control tensor differs from bounded P3A contract"
            )));
        }
        Ok(())
    }

    fn validate_fp8_pair(
        pair: &DeepSeekV4StagedNativePair,
        weight_name: &str,
        scale_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let scale_bytes = checked_mul(
            checked_mul(
                rows / ACT_QUANT_BLOCK,
                cols / ACT_QUANT_BLOCK,
                "FP8 scale elements",
            )?,
            size_of::<u8>(),
            "FP8 scale bytes",
        )?;
        if pair.kind != NativeScalePairKind::Fp8E4M3fn
            || pair.weight.name != weight_name
            || pair.scale.name != scale_name
            || pair.logical_k != cols as u64
            || pair.out_rows != rows as u64
            || pair.weight.bytes.len() != checked_mul(rows, cols, "FP8 weight bytes")?
            || pair.scale.bytes.len() != scale_bytes
        {
            return Err(sink_error(format!(
                "{weight_name} native FP8/E8M0 source pair differs from bounded P3A contract"
            )));
        }
        Ok(())
    }

    fn source_payload(label: &'static str, bytes: &[u8]) -> DeepSeekV4P3aSourcePayload {
        DeepSeekV4P3aSourcePayload {
            label,
            bytes: bytes.len(),
            sha256: sha256(bytes),
        }
    }

    fn bf16_le_bytes(values: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * size_of::<u16>());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
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

    fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize> {
        left.checked_mul(right)
            .ok_or_else(|| sink_error(format!("{label} overflow")))
    }

    fn checked_add(left: usize, right: usize, label: &str) -> Result<usize> {
        left.checked_add(right)
            .ok_or_else(|| sink_error(format!("{label} overflow")))
    }

    fn sink_error(message: impl Into<String>) -> Error {
        Error::Gravity(format!(
            "DeepSeek-V4 P3A Metal stage sink: {}",
            message.into()
        ))
    }
}

#[cfg(target_os = "macos")]
pub use macos::DeepSeekV4P3aMetalStageSink;

#[cfg(not(target_os = "macos"))]
mod unsupported {
    use super::*;
    use crate::gravity_deepseek_v4_execution_context::{
        DeepSeekV4ExecutionContext, DeepSeekV4PreparedDecodeInput,
    };
    use crate::gravity_deepseek_v4_layer_scheduler::{
        DeepSeekV4NativeStage, DeepSeekV4NativeStageConsumption, DeepSeekV4NativeStageSink,
    };
    use crate::{Error, Result};

    pub struct DeepSeekV4P3aMetalStageSink;

    impl DeepSeekV4P3aMetalStageSink {
        pub fn new(
            _context: &DeepSeekV4ExecutionContext,
            _prepared: &DeepSeekV4PreparedDecodeInput,
        ) -> Result<Self> {
            Err(Error::Metal(
                "DeepSeek-V4 P3A Metal stage sink requires macOS Metal".into(),
            ))
        }
    }

    impl DeepSeekV4NativeStageSink for DeepSeekV4P3aMetalStageSink {
        fn consume_native_stage(
            &mut self,
            _stage: DeepSeekV4NativeStage<'_>,
        ) -> Result<DeepSeekV4NativeStageConsumption> {
            Err(Error::Metal(
                "DeepSeek-V4 P3A Metal stage sink requires macOS Metal".into(),
            ))
        }
    }
}

#[cfg(not(target_os = "macos"))]
pub use unsupported::DeepSeekV4P3aMetalStageSink;
