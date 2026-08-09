//! Bounded caller-owned-Metal continuation for DeepSeek-V4 layer 1 BOS
//! attention.
//!
//! This is deliberately the smallest real successor to the existing layer-0
//! child boundary.  It accepts a device-resident `BF16[4,4096]` L0 child state
//! on the caller's existing Metal context, stages only authenticated
//! `layers.1.*` attention controls, and executes this one ordered graph:
//!
//! ```text
//! L0 child HC state
//!   -> L1 hc_attn_pre / attention RMSNorm
//!   -> ratio-zero Q / KV / causal slot-0 write / sparse attention
//!   -> WO-A / WO-B / hc_attn_post
//!   -> L1 attention HC state
//! ```
//!
//! The graph reuses the already-compiled generic mHC pre/post and attention
//! authority kernels.  It makes no Engine, full-forward, HCLI, decoding-loop,
//! parity-receipt, or TPS claim.  In particular, the one-token ratio-zero
//! specialization is not a substitute for the complete causal graph.

use std::mem::size_of;

use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::gravity_deepseek_v4_p7_device::DeepSeekV4P7DeviceOutput;
use crate::metal::{CommandBatch, MetalBatchTiming, MetalContext};
use crate::{Error, Result};

// These are the pinned architecture constants for the all-base-layer source
// grammar, deliberately declared locally so this L1 successor has no runtime
// dependency on a layer-0 oracle or metadata verifier.
const HIDDEN_SIZE: usize = 4096;
const HC_MULT: usize = 4;
const HC_MIX_WIDTH: usize = (2 + HC_MULT) * HC_MULT;
const HC_FLAT_WIDTH: usize = HC_MULT * HIDDEN_SIZE;
const HC_SINKHORN_ITERS: usize = 20;
const HC_EPS: f32 = 1.0e-6;
const RMS_NORM_EPS: f32 = 1.0e-6;
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

const OFFICIAL_INFERENCE_MODEL_PY_SHA256: &str =
    "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71";
const OFFICIAL_INFERENCE_KERNEL_PY_SHA256: &str =
    "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2";
const OFFICIAL_INFERENCE_CONVERT_PY_SHA256: &str =
    "912acfc20bdd9ae4dbd5bde9dc7c8e61f6d27b6826d3ac2d052b2534c0881454";
const OFFICIAL_INFERENCE_CONFIG_JSON_SHA256: &str =
    "6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733";

/// This primitive is intentionally only the first base-layer successor.
pub const DSV4F_L1_BOS_LAYER: usize = 1;
/// The primitive consumes the BOS child state at causal position zero.
pub const DSV4F_L1_BOS_TOKEN_ID: u32 = 0;
pub const DSV4F_L1_BOS_POSITION: usize = 0;
/// Layer-1 has the source's ratio-zero attention geometry at BOS.
pub const DSV4F_L1_BOS_COMPRESS_RATIO: usize = 0;
pub const DSV4F_L1_BOS_HC_STATE_BF16_BYTES: usize = HC_FLAT_WIDTH * size_of::<u16>();
pub const DSV4F_L1_BOS_KV_SLOT0_BF16_BYTES: usize = HEAD_DIM * size_of::<u16>();
/// One ordered command buffer: mHC pre/norm, Q/KV, attention, output, mHC
/// post.  This is topology accounting for this bounded primitive only, not a
/// token-runtime command topology.
pub const DSV4F_L1_BOS_ATTENTION_DISPATCHES: usize = 22;

const HC_PRE_KERNEL: &str = "deepseek_v4_p7_mhc_ffn_pre_authority";
const RMS_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
const QAT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const PER_HEAD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
const CACHE_KERNEL: &str = "deepseek_v4_p4b_kv_cache_write_bf16_authority";
/// Production ratio-0 path: growing-KV supersedes the fixed position-0 specialization.
const SPARSE_KERNEL: &str = "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority";
const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
const HC_POST_KERNEL: &str = "deepseek_v4_p7_mhc_ffn_post_authority";

/// Immutable source-native identity of one static L1 control.  The raw bytes
/// are uploaded directly to Metal and are never retained by this public
/// surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4L1BosAttentionSourceTensorBinding {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: usize,
    pub sha256: String,
}

/// Metadata-only static binding for the primitive.  Every listed control must
/// be an authenticated `layers.1.*` source tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4L1BosAttentionSourceBindings {
    pub artifact_manifest_seal_sha256: String,
    pub repository: String,
    pub revision: String,
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
    pub compress_ratio: usize,
    pub inference_model_py_sha256: String,
    pub inference_kernel_py_sha256: String,
    pub inference_convert_py_sha256: String,
    pub inference_config_json_sha256: String,
    pub controls: Vec<DeepSeekV4L1BosAttentionSourceTensorBinding>,
    pub source_parent_retained: bool,
    pub host_activation_handoff_permitted: bool,
    pub runtime_boundary: &'static str,
}

/// One caller-owned L0 child boundary.  It deliberately fixes the only
/// accepted identity (layer 0/BOS/position 0), so a caller cannot relabel a
/// different layer or token as the L1 predecessor.  Metal buffers are owned
/// by a device rather than a queue; this value records the caller's exact
/// queue identity and rejects execution through another context/queue.
pub struct DeepSeekV4L1BosChildDeviceInput<'a> {
    metal: &'a MetalContext,
    child_hc_state_bf16: &'a metal::Buffer,
    queue_identity: usize,
}

impl<'a> DeepSeekV4L1BosChildDeviceInput<'a> {
    /// Bind a real L0/BOS child buffer to its existing Metal context.  The
    /// layer/token/position are fixed by this type rather than accepted as
    /// caller-controlled labels.  No activation is copied or mapped.
    pub fn from_l0_bos_child(
        metal: &'a MetalContext,
        child_hc_state_bf16: &'a metal::Buffer,
    ) -> Result<Self> {
        validate_buffer_device(metal, child_hc_state_bf16, "L0 child")?;
        if child_hc_state_bf16.length() != DSV4F_L1_BOS_HC_STATE_BF16_BYTES as u64 {
            return Err(l1_error(
                "L0 child input must be an exact BF16[4,4096] Metal buffer",
            ));
        }
        Ok(Self {
            metal,
            child_hc_state_bf16,
            queue_identity: context_queue_identity(metal),
        })
    }

    /// Construct the same fixed L0/BOS boundary from the existing bounded P7
    /// child result.  This is the strongest available provenance-preserving
    /// convenience path; it rejects the position-one P7 diagnostic output.
    pub fn from_p7_position0_child(
        metal: &'a MetalContext,
        output: &'a DeepSeekV4P7DeviceOutput,
    ) -> Result<Self> {
        output.validate()?;
        if output.layer != 0
            || output.token_id != DSV4F_L1_BOS_TOKEN_ID
            || output.token_position != DSV4F_L1_BOS_POSITION
        {
            return Err(l1_error(
                "L1 BOS continuation accepts only the layer-0/BOS/position-0 P7 child",
            ));
        }
        Self::from_l0_bos_child(metal, &output.child_hc_state_bf16)
    }

    pub const fn source_layer(&self) -> usize {
        0
    }

    pub const fn token_id(&self) -> u32 {
        DSV4F_L1_BOS_TOKEN_ID
    }

    pub const fn token_position(&self) -> usize {
        DSV4F_L1_BOS_POSITION
    }

    fn validate_for(&self, metal: &MetalContext) -> Result<()> {
        if !std::ptr::eq(self.metal, metal) || self.queue_identity != context_queue_identity(metal)
        {
            return Err(l1_error(
                "L0 child state must execute through its original caller-owned MetalContext/queue",
            ));
        }
        validate_buffer_device(metal, self.child_hc_state_bf16, "L0 child")?;
        if self.child_hc_state_bf16.length() != DSV4F_L1_BOS_HC_STATE_BF16_BYTES as u64 {
            return Err(l1_error(
                "L0 child buffer geometry changed after its BOS input binding",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4L1BosAttentionDevicePhase {
    Prepared,
    Complete,
}

impl DeepSeekV4L1BosAttentionDevicePhase {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Prepared => "prepared_l1_bos_attention_not_executed",
            Self::Complete => "complete_l1_bos_attention_device_only",
        }
    }
}

/// Device-only retained state after the bounded L1 BOS attention graph.  The
/// buffers remain tied to the caller's Metal device; there is deliberately no
/// host activation/readback method on this type.
pub struct DeepSeekV4L1BosAttentionDeviceOutput {
    pub attention_hc_state_bf16: metal::Buffer,
    pub causal_kv_slot0_bf16: metal::Buffer,
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
    pub kv_rows: usize,
    pub timing: MetalBatchTiming,
    pub actual_command_buffers: usize,
    pub actual_compute_encoders: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
    pub host_intermediate_handoff_bytes: usize,
}

impl DeepSeekV4L1BosAttentionDeviceOutput {
    /// Borrow the complete attention residual as a P7 position-0 input on the
    /// caller's same Metal context. No host readback or buffer copy.
    #[cfg(target_os = "macos")]
    pub fn p7_attention_state<'a>(
        &'a self,
        metal: &'a MetalContext,
    ) -> Result<crate::gravity_deepseek_v4_p7_composition::DeepSeekV4P7AttentionDeviceState<'a>>
    {
        self.validate()?;
        crate::gravity_deepseek_v4_p7_composition::DeepSeekV4P7AttentionDeviceState::position0(
            metal,
            &self.attention_hc_state_bf16,
            self.layer,
            self.token_id,
        )
    }

    pub fn validate(&self) -> Result<()> {
        if self.attention_hc_state_bf16.length() != DSV4F_L1_BOS_HC_STATE_BF16_BYTES as u64
            || self.causal_kv_slot0_bf16.length() != DSV4F_L1_BOS_KV_SLOT0_BF16_BYTES as u64
            || self.layer != DSV4F_L1_BOS_LAYER
            || self.token_id != DSV4F_L1_BOS_TOKEN_ID
            || self.token_position != DSV4F_L1_BOS_POSITION
            || self.kv_rows != 1
            || self.actual_command_buffers != 1
            || self.actual_compute_encoders != DSV4F_L1_BOS_ATTENTION_DISPATCHES
            || self.actual_gpu_dispatches != DSV4F_L1_BOS_ATTENTION_DISPATCHES
            || self.actual_cpu_visible_waits != 1
            || self.host_intermediate_handoff_bytes != 0
        {
            return Err(l1_error(
                "L1 BOS attention output has invalid state identity, geometry, or topology",
            ));
        }
        Ok(())
    }
}

struct VerifiedTensor {
    binding: DeepSeekV4L1BosAttentionSourceTensorBinding,
    bytes: Vec<u8>,
}

struct Fp8Pair {
    weight: VerifiedTensor,
    scale: VerifiedTensor,
}

struct Layer1Controls {
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

impl Layer1Controls {
    fn load(reader: &DeepSeekV4FullStreamReader) -> Result<Self> {
        let layer = DSV4F_L1_BOS_LAYER;
        let name = |suffix: &str| format!("layers.{layer}.{suffix}");
        Ok(Self {
            hc_fn: read_verified_tensor(
                reader,
                &name("hc_attn_fn"),
                "F32",
                &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
                HC_MIX_WIDTH * HC_FLAT_WIDTH * size_of::<f32>(),
            )?,
            hc_base: read_verified_tensor(
                reader,
                &name("hc_attn_base"),
                "F32",
                &[HC_MIX_WIDTH as u64],
                HC_MIX_WIDTH * size_of::<f32>(),
            )?,
            hc_scale: read_verified_tensor(
                reader,
                &name("hc_attn_scale"),
                "F32",
                &[3],
                3 * size_of::<f32>(),
            )?,
            attn_norm: read_verified_tensor(
                reader,
                &name("attn_norm.weight"),
                "BF16",
                &[HIDDEN_SIZE as u64],
                HIDDEN_SIZE * size_of::<u16>(),
            )?,
            q_norm: read_verified_tensor(
                reader,
                &name("attn.q_norm.weight"),
                "BF16",
                &[Q_LORA_RANK as u64],
                Q_LORA_RANK * size_of::<u16>(),
            )?,
            kv_norm: read_verified_tensor(
                reader,
                &name("attn.kv_norm.weight"),
                "BF16",
                &[HEAD_DIM as u64],
                HEAD_DIM * size_of::<u16>(),
            )?,
            attn_sink: read_verified_tensor(
                reader,
                &name("attn.attn_sink"),
                "F32",
                &[NUM_HEADS as u64],
                NUM_HEADS * size_of::<f32>(),
            )?,
            wq_a: read_fp8_pair(
                reader,
                &name("attn.wq_a.weight"),
                &name("attn.wq_a.scale"),
                WQ_A_ROWS,
                WQ_A_COLS,
            )?,
            wq_b: read_fp8_pair(
                reader,
                &name("attn.wq_b.weight"),
                &name("attn.wq_b.scale"),
                WQ_B_ROWS,
                Q_LORA_RANK,
            )?,
            wkv: read_fp8_pair(
                reader,
                &name("attn.wkv.weight"),
                &name("attn.wkv.scale"),
                WKV_ROWS,
                HIDDEN_SIZE,
            )?,
            wo_a: read_fp8_pair(
                reader,
                &name("attn.wo_a.weight"),
                &name("attn.wo_a.scale"),
                WO_A_ROWS,
                WO_A_COLS,
            )?,
            wo_b: read_fp8_pair(
                reader,
                &name("attn.wo_b.weight"),
                &name("attn.wo_b.scale"),
                WO_B_ROWS,
                WO_B_COLS,
            )?,
        })
    }

    fn bindings(&self) -> Vec<DeepSeekV4L1BosAttentionSourceTensorBinding> {
        [
            &self.hc_fn,
            &self.hc_base,
            &self.hc_scale,
            &self.attn_norm,
            &self.q_norm,
            &self.kv_norm,
            &self.attn_sink,
            &self.wq_a.weight,
            &self.wq_a.scale,
            &self.wq_b.weight,
            &self.wq_b.scale,
            &self.wkv.weight,
            &self.wkv.scale,
            &self.wo_a.weight,
            &self.wo_a.scale,
            &self.wo_b.weight,
            &self.wo_b.scale,
        ]
        .into_iter()
        .map(|control| control.binding.clone())
        .collect()
    }
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

/// Prepared static L1 controls and device scratch for exactly one L0/BOS
/// child continuation.  It does not own a Metal context and therefore cannot
/// create an alternate queue or a hidden host activation bridge.
pub struct DeepSeekV4Layer1BosAttentionDeviceExecutor {
    source_bindings: DeepSeekV4L1BosAttentionSourceBindings,
    context_queue_identity: usize,
    phase: DeepSeekV4L1BosAttentionDevicePhase,
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
    wkv: LinearScratch,
    kv: KvScratch,
    causal_kv_slot0_bf16: metal::Buffer,
    sparse: metal::Buffer,
    scores: metal::Buffer,
    denominators: metal::Buffer,
    wo_a: metal::Buffer,
    wo_b: LinearScratch,
    attention_hc_state_bf16: metal::Buffer,
}

impl DeepSeekV4Layer1BosAttentionDeviceExecutor {
    /// Verify and upload only static `layers.1.*` source controls.  The
    /// caller-owned L0 state is deliberately not read, copied, or uploaded
    /// during preparation.
    pub fn prepare(metal: &MetalContext, reader: &DeepSeekV4FullStreamReader) -> Result<Self> {
        verify_l1_bos_source_contract(reader)?;
        validate_required_pipelines(metal)?;
        let controls = Layer1Controls::load(reader)?;
        let bindings = DeepSeekV4L1BosAttentionSourceBindings {
            artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            repository: reader.source_identity().repository.to_owned(),
            revision: reader.source_identity().revision.to_owned(),
            layer: DSV4F_L1_BOS_LAYER,
            token_id: DSV4F_L1_BOS_TOKEN_ID,
            token_position: DSV4F_L1_BOS_POSITION,
            compress_ratio: DSV4F_L1_BOS_COMPRESS_RATIO,
            inference_model_py_sha256: reader
                .source_metadata_asset_sha256("inference/model.py")?
                .to_owned(),
            inference_kernel_py_sha256: reader
                .source_metadata_asset_sha256("inference/kernel.py")?
                .to_owned(),
            inference_convert_py_sha256: reader
                .source_metadata_asset_sha256("inference/convert.py")?
                .to_owned(),
            inference_config_json_sha256: reader
                .source_metadata_asset_sha256("inference/config.json")?
                .to_owned(),
            controls: controls.bindings(),
            source_parent_retained: false,
            host_activation_handoff_permitted: false,
            runtime_boundary: "this is one caller-owned-context layer-1/BOS ratio-zero attention continuation only; it has no FFN/MoE continuation, full causal graph, Engine, HCLI, endpoint, decode loop, parity receipt, or TPS claim",
        };
        if bindings.controls.len() != 17
            || bindings
                .controls
                .iter()
                .any(|control| !control.name.starts_with("layers.1."))
        {
            return Err(l1_error(
                "L1 preparation tried to stage a non-layer-1 or incomplete source-control set",
            ));
        }

        Ok(Self {
            source_bindings: bindings,
            context_queue_identity: context_queue_identity(metal),
            phase: DeepSeekV4L1BosAttentionDevicePhase::Prepared,
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
            wkv: new_linear_scratch(metal, HIDDEN_SIZE, WKV_ROWS)?,
            kv: new_kv_scratch(metal)?,
            causal_kv_slot0_bf16: metal.new_buffer_checked(DSV4F_L1_BOS_KV_SLOT0_BF16_BYTES)?,
            sparse: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
            scores: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
            denominators: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
            wo_a: metal.new_buffer_checked(WO_A_ROWS * size_of::<u16>())?,
            wo_b: new_linear_scratch(metal, WO_B_COLS, WO_B_ROWS)?,
            attention_hc_state_bf16: metal.new_buffer_checked(DSV4F_L1_BOS_HC_STATE_BF16_BYTES)?,
        })
    }

    pub fn source_bindings(&self) -> &DeepSeekV4L1BosAttentionSourceBindings {
        &self.source_bindings
    }

    pub const fn phase(&self) -> DeepSeekV4L1BosAttentionDevicePhase {
        self.phase
    }

    /// Execute the complete bounded L1/BOS attention graph with no host
    /// activation transfer.  A prepared executor is deliberately one-shot so
    /// a caller cannot relabel a repeated write as a distinct causal token.
    pub fn execute(
        &mut self,
        metal: &MetalContext,
        input: DeepSeekV4L1BosChildDeviceInput<'_>,
    ) -> Result<DeepSeekV4L1BosAttentionDeviceOutput> {
        if self.phase != DeepSeekV4L1BosAttentionDevicePhase::Prepared {
            return Err(l1_error(
                "L1 BOS attention executor may execute only once per preparation",
            ));
        }
        if context_queue_identity(metal) != self.context_queue_identity {
            return Err(l1_error(
                "L1 BOS attention executor requires its preparation MetalContext/queue",
            ));
        }
        input.validate_for(metal)?;

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
        let slot0 = 0u32;
        let cache_capacity = 1u32;

        let timing = metal.dispatch_batch_timed(|batch| {
            dispatch_hc_pre(
                batch,
                input.child_hc_state_bf16,
                &self.hc_fn,
                &self.hc_scale,
                &self.hc_base,
                &self.hc,
                hidden,
                hc_mult,
                mix_width,
                sinkhorn,
            )?;
            dispatch_rms(
                batch,
                &self.hc.reduced,
                &self.attn_norm_weight,
                &self.attn_norm,
                hidden,
            )?;
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
            dispatch_cache_write(
                batch,
                &self.kv.qat,
                &self.causal_kv_slot0_bf16,
                slot0,
                head_dim,
                cache_capacity,
            )?;
            // BOS/position-0: one written KV row; growing-KV with valid_kv_count=1.
            dispatch_sparse_growing_kv(
                batch,
                &self.q_head,
                &self.causal_kv_slot0_bf16,
                &self.attn_sink,
                &self.sparse,
                &self.scores,
                &self.denominators,
                heads,
                head_dim,
                cache_capacity,
                /*valid_kv_count=*/ 1,
                /*max_score_slots=*/ 1,
                sparse_scale,
            )?;
            dispatch_wo_a(
                batch,
                &self.wo_a_weight,
                &self.wo_a_scale,
                &self.sparse,
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
            // The P7 mHC-post authority kernel is algebraically generic
            // `Block.hc_post`; this invocation supplies an attention vector
            // and L1 `hc_attn_*` controls, never a P6/MoE buffer.
            dispatch_hc_post(
                batch,
                &self.wo_b.bf16,
                input.child_hc_state_bf16,
                &self.hc.post,
                &self.hc.comb,
                &self.attention_hc_state_bf16,
                hidden,
                hc_mult,
            )?;
            Ok(())
        })?;

        if timing.command_buffers != 1
            || timing.compute_encoders as usize != DSV4F_L1_BOS_ATTENTION_DISPATCHES
            || timing.compute_dispatches as usize != DSV4F_L1_BOS_ATTENTION_DISPATCHES
        {
            return Err(l1_error(
                "L1 BOS attention completion topology differs from its declared ordered graph",
            ));
        }
        self.phase = DeepSeekV4L1BosAttentionDevicePhase::Complete;
        let output = DeepSeekV4L1BosAttentionDeviceOutput {
            attention_hc_state_bf16: self.attention_hc_state_bf16.to_owned(),
            causal_kv_slot0_bf16: self.causal_kv_slot0_bf16.to_owned(),
            layer: DSV4F_L1_BOS_LAYER,
            token_id: DSV4F_L1_BOS_TOKEN_ID,
            token_position: DSV4F_L1_BOS_POSITION,
            kv_rows: 1,
            timing,
            actual_command_buffers: 1,
            actual_compute_encoders: DSV4F_L1_BOS_ATTENTION_DISPATCHES,
            actual_gpu_dispatches: DSV4F_L1_BOS_ATTENTION_DISPATCHES,
            actual_cpu_visible_waits: 1,
            host_intermediate_handoff_bytes: 0,
        };
        output.validate()?;
        Ok(output)
    }
}

fn verify_l1_bos_source_contract(reader: &DeepSeekV4FullStreamReader) -> Result<()> {
    if reader.source_identity().repository != PINNED_REPOSITORY
        || reader.source_identity().revision != PINNED_REVISION
    {
        return Err(l1_error(
            "L1 BOS attention requires the pinned DeepSeek-V4-Flash source identity",
        ));
    }
    let model_sha = reader.source_metadata_asset_sha256("inference/model.py")?;
    let kernel_sha = reader.source_metadata_asset_sha256("inference/kernel.py")?;
    let convert_sha = reader.source_metadata_asset_sha256("inference/convert.py")?;
    let config_sha = reader.source_metadata_asset_sha256("inference/config.json")?;
    if model_sha != OFFICIAL_INFERENCE_MODEL_PY_SHA256
        || kernel_sha != OFFICIAL_INFERENCE_KERNEL_PY_SHA256
        || convert_sha != OFFICIAL_INFERENCE_CONVERT_PY_SHA256
        || config_sha != OFFICIAL_INFERENCE_CONFIG_JSON_SHA256
    {
        return Err(l1_error(
            "L1 source metadata hashes differ from the pinned source grammar",
        ));
    }
    let config: Value = serde_json::from_slice(
        &reader.read_verified_metadata_asset("inference/config.json", 64 * 1024)?,
    )
    .map_err(|error| l1_error(format!("L1 inference config JSON: {error}")))?;
    for (key, expected) in [
        ("dim", HIDDEN_SIZE as u64),
        ("n_layers", 43),
        ("n_heads", NUM_HEADS as u64),
        ("q_lora_rank", Q_LORA_RANK as u64),
        ("head_dim", HEAD_DIM as u64),
        ("rope_head_dim", ROPE_HEAD_DIM as u64),
        ("o_groups", 8),
        ("o_lora_rank", O_LORA_RANK as u64),
        ("window_size", 128),
        ("hc_mult", HC_MULT as u64),
        ("hc_sinkhorn_iters", HC_SINKHORN_ITERS as u64),
    ] {
        if config.get(key).and_then(Value::as_u64) != Some(expected) {
            return Err(l1_error(format!(
                "L1 inference config {key:?} differs from the pinned geometry"
            )));
        }
    }
    if config.get("dtype").and_then(Value::as_str) != Some("fp8")
        || config.get("scale_fmt").and_then(Value::as_str) != Some("ue8m0")
    {
        return Err(l1_error(
            "L1 inference config does not retain the native FP8/E8M0 grammar",
        ));
    }
    let ratios = config
        .get("compress_ratios")
        .and_then(Value::as_array)
        .ok_or_else(|| l1_error("L1 inference config lacks compress_ratios"))?;
    // Source lists 44 schedule entries (43 base + 1 dense head). Admit either
    // 43 or 44 as long as base layers 0 and 1 remain ratio-zero.
    if !(ratios.len() == 43 || ratios.len() == 44)
        || ratios.get(0).and_then(Value::as_u64) != Some(0)
        || ratios.get(DSV4F_L1_BOS_LAYER).and_then(Value::as_u64)
            != Some(DSV4F_L1_BOS_COMPRESS_RATIO as u64)
    {
        return Err(l1_error(
            "layer-1 compression ratio is not the required ratio-zero BOS specialization",
        ));
    }

    let model_config: Value =
        serde_json::from_slice(&reader.read_verified_metadata_asset("config.json", 64 * 1024)?)
            .map_err(|error| l1_error(format!("L1 model config JSON: {error}")))?;
    for (key, expected) in [("rms_norm_eps", RMS_NORM_EPS), ("hc_eps", HC_EPS)] {
        let actual = model_config.get(key).and_then(Value::as_f64);
        if actual.is_none_or(|value| (value as f32).to_bits() != expected.to_bits()) {
            return Err(l1_error(format!(
                "L1 model config {key:?} differs from the pinned F32 control"
            )));
        }
    }

    // Keep this module source-specific without delegating its admission to a
    // layer-0 checkpoint.  These are short grammar guards in addition to the
    // exact metadata hashes above; no source tensor is staged here.
    let model_py = reader.read_verified_metadata_asset("inference/model.py", 128 * 1024)?;
    let kernel_py = reader.read_verified_metadata_asset("inference/kernel.py", 128 * 1024)?;
    let convert_py = reader.read_verified_metadata_asset("inference/convert.py", 128 * 1024)?;
    for (asset, needle) in [
        (
            model_py.as_slice(),
            b"x, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)"
                .as_slice(),
        ),
        (
            model_py.as_slice(),
            b"x = self.hc_post(x, residual, post, comb)".as_slice(),
        ),
        (
            model_py.as_slice(),
            b"act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)".as_slice(),
        ),
        (
            model_py.as_slice(),
            b"o = torch.einsum(\"bsgd,grd->bsgr\", o, wo_a)".as_slice(),
        ),
        (
            kernel_py.as_slice(),
            b"def sparse_attn_kernel(h: int, d: int, scale=None):".as_slice(),
        ),
        (convert_py.as_slice(), b"if name.endswith(\"wo_a.weight\"):".as_slice()),
    ] {
        if !contains_bytes(asset, needle) {
            return Err(l1_error("pinned L1 source grammar anchor is absent"));
        }
    }
    Ok(())
}

fn contains_bytes(haystack: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty()
        && haystack
            .windows(needle.len())
            .any(|window| window == needle)
}

fn read_verified_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    dtype: &str,
    shape: &[u64],
    expected_bytes: usize,
) -> Result<VerifiedTensor> {
    if !name.starts_with("layers.1.") {
        return Err(l1_error(format!(
            "L1 static-control loader refuses non-layer-1 tensor {name:?}"
        )));
    }
    let metadata = reader.tensor_metadata(name)?;
    if metadata.name != name
        || metadata.dtype != dtype
        || metadata.shape.as_slice() != shape
        || metadata.bytes != expected_bytes as u64
    {
        return Err(l1_error(format!(
            "{name} differs from the required L1 source-native geometry"
        )));
    }
    let bytes = reader.read_verified_full(name, expected_bytes)?;
    if bytes.len() != expected_bytes {
        return Err(l1_error(format!(
            "{name} verified read length differs from the source binding"
        )));
    }
    Ok(VerifiedTensor {
        binding: DeepSeekV4L1BosAttentionSourceTensorBinding {
            name: metadata.name.clone(),
            dtype: metadata.dtype.clone(),
            shape: metadata.shape.clone(),
            bytes: expected_bytes,
            sha256: sha256(&bytes),
        },
        bytes,
    })
}

fn read_fp8_pair(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    rows: usize,
    cols: usize,
) -> Result<Fp8Pair> {
    let pair = reader.native_scale_pair(weight_name)?;
    let scale_rows = rows / ACT_QUANT_BLOCK;
    let scale_cols = cols / ACT_QUANT_BLOCK;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.weight.name != weight_name
        || pair.scale.name != scale_name
        || pair.out_rows != rows as u64
        || pair.logical_k != cols as u64
        || pair.packed_k != cols as u64
        || pair.scale_rows != scale_rows as u64
        || pair.scale_cols != scale_cols as u64
        || pair.weight.shape.as_slice() != [rows as u64, cols as u64]
        || pair.scale.shape.as_slice() != [scale_rows as u64, scale_cols as u64]
    {
        return Err(l1_error(format!(
            "L1 native FP8/E8M0 pair geometry differs for {weight_name}"
        )));
    }
    Ok(Fp8Pair {
        weight: read_verified_tensor(
            reader,
            weight_name,
            "F8_E4M3",
            &[rows as u64, cols as u64],
            rows.checked_mul(cols)
                .ok_or_else(|| l1_error("L1 FP8 weight byte geometry overflow"))?,
        )?,
        scale: read_verified_tensor(
            reader,
            scale_name,
            "F8_E8M0",
            &[scale_rows as u64, scale_cols as u64],
            scale_rows
                .checked_mul(scale_cols)
                .ok_or_else(|| l1_error("L1 FP8 scale byte geometry overflow"))?,
        )?,
    })
}

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
    if cols == 0 || rows == 0 || cols % ACT_QUANT_BLOCK != 0 {
        return Err(l1_error("L1 FP8 scratch geometry is not block-128 aligned"));
    }
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
    for (kernel, threads) in [
        (HC_PRE_KERNEL, 1u32),
        (RMS_KERNEL, 1),
        (QAT_KERNEL, 32),
        (FP8_KERNEL, 256),
        (CAST_KERNEL, 256),
        (PER_HEAD_KERNEL, 64),
        (KV_QAT_KERNEL, 32),
        (CACHE_KERNEL, 256),
        (SPARSE_KERNEL, 64),
        (WO_A_KERNEL, 256),
        (HC_POST_KERNEL, 256),
    ] {
        let maximum = metal.pipeline(kernel)?.max_total_threads_per_threadgroup() as u32;
        if maximum < threads {
            return Err(l1_error(format!(
                "{kernel} supports only {maximum} threads, below L1 required {threads}",
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hc_pre(
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
    batch.dispatch_threads(HC_PRE_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(residual), 0);
        encoder.set_buffer(1, Some(hc_fn), 0);
        encoder.set_buffer(2, Some(hc_scale), 0);
        encoder.set_buffer(3, Some(hc_base), 0);
        encoder.set_buffer(4, Some(&scratch.reduced), 0);
        encoder.set_buffer(5, Some(&scratch.rsqrt), 0);
        encoder.set_buffer(6, Some(&scratch.mixes), 0);
        encoder.set_buffer(7, Some(&scratch.pre), 0);
        encoder.set_buffer(8, Some(&scratch.post), 0);
        encoder.set_buffer(9, Some(&scratch.comb), 0);
        set_u32(encoder, 10, &hidden);
        set_u32(encoder, 11, &hc_mult);
        set_u32(encoder, 12, &mix_width);
        set_u32(encoder, 13, &sinkhorn);
        set_f32(encoder, 14, &RMS_NORM_EPS);
        set_f32(encoder, 15, &HC_EPS);
    })
}

fn dispatch_rms(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    weight: &metal::Buffer,
    output: &metal::Buffer,
    width: u32,
) -> Result<()> {
    batch.dispatch_threads(RMS_KERNEL, (1, 1, 1), (1, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        set_u32(encoder, 3, &width);
        set_f32(encoder, 4, &RMS_NORM_EPS);
    })
}

fn dispatch_qat(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    scales: &metal::Buffer,
    cols: u32,
) -> Result<()> {
    if cols == 0 || cols % ACT_QUANT_BLOCK as u32 != 0 {
        return Err(l1_error("L1 activation quantization width is invalid"));
    }
    batch.dispatch_threads(
        QAT_KERNEL,
        (cols / ACT_QUANT_BLOCK as u32, 1, 1),
        (32, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_buffer(2, Some(scales), 0);
            set_u32(encoder, 3, &cols);
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
    if rows == 0 || cols == 0 || scale_cols != cols / ACT_QUANT_BLOCK as u32 {
        return Err(l1_error("L1 FP8 matvec source geometry is invalid"));
    }
    batch.dispatch_threads(FP8_KERNEL, (rows, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(activation), 0);
        encoder.set_buffer(3, Some(activation_scales), 0);
        encoder.set_buffer(4, Some(output), 0);
        set_u32(encoder, 5, &rows);
        set_u32(encoder, 6, &cols);
        set_u32(encoder, 7, &scale_cols);
    })
}

fn dispatch_cast(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    count: u32,
) -> Result<()> {
    batch.dispatch_threads(CAST_KERNEL, (count, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        set_u32(encoder, 2, &count);
    })
}

fn dispatch_per_head(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    heads: u32,
    head_dim: u32,
) -> Result<()> {
    batch.dispatch_threads(PER_HEAD_KERNEL, (heads, 1, 1), (64, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(output), 0);
        set_u32(encoder, 2, &heads);
        set_u32(encoder, 3, &head_dim);
        set_f32(encoder, 4, &RMS_NORM_EPS);
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
    if head_dim != HEAD_DIM as u32
        || rope_dim != ROPE_HEAD_DIM as u32
        || block != KV_QAT_BLOCK as u32
    {
        return Err(l1_error("L1 ratio-zero KV QAT geometry is invalid"));
    }
    batch.dispatch_threads(
        KV_QAT_KERNEL,
        (NON_ROPE_HEAD_DIM as u32 / block, 1, 1),
        (32, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_buffer(2, Some(activation), 0);
            encoder.set_buffer(3, Some(scales), 0);
            set_u32(encoder, 4, &head_dim);
            set_u32(encoder, 5, &rope_dim);
            set_u32(encoder, 6, &block);
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
    if position != 0 || head_dim != HEAD_DIM as u32 || capacity != 1 {
        return Err(l1_error("L1 causal cache write must target only slot zero"));
    }
    batch.dispatch_threads(CACHE_KERNEL, (head_dim, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(cache), 0);
        set_u32(encoder, 2, &position);
        set_u32(encoder, 3, &head_dim);
        set_u32(encoder, 4, &capacity);
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
    if heads != NUM_HEADS as u32
        || head_dim != HEAD_DIM as u32
        || valid_kv_count == 0
        || valid_kv_count > cache_capacity
        || max_score_slots < valid_kv_count
        || !(scale > 0.0)
    {
        return Err(l1_error(
            "L1 ratio-zero growing-KV sparse-attention geometry is invalid",
        ));
    }
    batch.dispatch_threads(SPARSE_KERNEL, (heads, 1, 1), (64, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(q), 0);
        encoder.set_buffer(1, Some(kv_cache), 0);
        encoder.set_buffer(2, Some(sink), 0);
        encoder.set_buffer(3, Some(output), 0);
        encoder.set_buffer(4, Some(scores), 0);
        encoder.set_buffer(5, Some(denominators), 0);
        set_u32(encoder, 6, &heads);
        set_u32(encoder, 7, &head_dim);
        set_u32(encoder, 8, &cache_capacity);
        set_u32(encoder, 9, &valid_kv_count);
        set_u32(encoder, 10, &max_score_slots);
        set_f32(encoder, 11, &scale);
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
    if rows != WO_A_ROWS as u32
        || cols != WO_A_COLS as u32
        || scale_cols != (WO_A_COLS / ACT_QUANT_BLOCK) as u32
        || ranks != O_LORA_RANK as u32
    {
        return Err(l1_error("L1 WO-A converted-einsum geometry is invalid"));
    }
    batch.dispatch_threads(WO_A_KERNEL, (rows, 1, 1), (256, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(weight), 0);
        encoder.set_buffer(1, Some(scales), 0);
        encoder.set_buffer(2, Some(input), 0);
        encoder.set_buffer(3, Some(output), 0);
        set_u32(encoder, 4, &rows);
        set_u32(encoder, 5, &cols);
        set_u32(encoder, 6, &scale_cols);
        set_u32(encoder, 7, &ranks);
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hc_post(
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
        HC_POST_KERNEL,
        (hidden * hc_mult, 1, 1),
        (256, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(attention), 0);
            encoder.set_buffer(1, Some(residual), 0);
            encoder.set_buffer(2, Some(post), 0);
            encoder.set_buffer(3, Some(comb), 0);
            encoder.set_buffer(4, Some(output), 0);
            set_u32(encoder, 5, &hidden);
            set_u32(encoder, 6, &hc_mult);
        },
    )
}

fn validate_buffer_device(metal: &MetalContext, buffer: &metal::Buffer, label: &str) -> Result<()> {
    if buffer.device().registry_id() != metal.device().registry_id() {
        return Err(l1_error(format!(
            "{label} buffer belongs to a different Metal device",
        )));
    }
    Ok(())
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

fn l1_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 layer-1 BOS attention device primitive: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l1_bos_geometry_and_topology_are_explicitly_bounded() {
        assert_eq!(DSV4F_L1_BOS_LAYER, 1);
        assert_eq!(DSV4F_L1_BOS_COMPRESS_RATIO, 0);
        assert_eq!(DSV4F_L1_BOS_HC_STATE_BF16_BYTES, 4 * 4096 * 2);
        assert_eq!(DSV4F_L1_BOS_KV_SLOT0_BF16_BYTES, 512 * 2);
        assert_eq!(DSV4F_L1_BOS_ATTENTION_DISPATCHES, 22);
    }
}
