//! Source-hash-bound data plane for the future DeepSeek-V4 causal executor.
//!
//! This is intentionally **not** an [`crate::Engine`].  The sealed V4
//! artifact is a complete content-addressed source stream, but it still has
//! no source-faithful 43-layer causal forward.  Registering a skeleton with
//! the public serve path would make a false readiness claim, so this module
//! only provides the prerequisite data plane:
//!
//! - fail-closed validation of the exact 43 base decoder layers and their
//!   mHC, attention, router, routed-expert, shared-expert, and head bindings;
//! - exact source/config hashes and cross-config geometry checks;
//! - bounded, verified staging of native FP8/FP4 operator pairs; and
//! - bounded BF16 embedding, final-norm, and LM-head row loading primitives.
//!
//! The MTP auxiliary layer is deliberately excluded from the base topology.
//! It cannot accidentally participate in `BASE_TRUE_TPS`, and no method here
//! allocates Metal resources, maintains causal state, forwards a token,
//! samples, streams HCLI output, or implements `Engine`.

use std::ops::Range;

use half::bf16;
use serde_json::Value;

use crate::gravity_deepseek_v4::{DeepSeekV4FullStreamReader, NativeScalePairKind};
use crate::{Error, Result};

/// Exact base-body geometry; the one extra compression-ratio entry belongs to
/// the excluded MTP auxiliary layer.
pub const DSV4F_BASE_LAYER_COUNT: usize = 43;
pub const DSV4F_HIDDEN_SIZE: usize = 4_096;
pub const DSV4F_VOCAB_SIZE: usize = 129_280;
pub const DSV4F_ROUTED_EXPERT_COUNT: usize = 256;
pub const DSV4F_TOP_K_EXPERTS: usize = 6;
pub const DSV4F_HC_MULT: usize = 4;
pub const DSV4F_HC_SINKHORN_ITERS: usize = 20;
pub const DSV4F_MTP_LAYER_COUNT: usize = 1;

/// A deliberately small, explicit upper bound for a future executor's
/// one-shot host staging request.  It permits every base control/expert
/// operator but rejects accidental full embedding/head downloads.
pub const MAX_STAGED_OPERATOR_BYTES: usize = 64 * 1024 * 1024;
pub const BF16_VECTOR_BYTES: usize = DSV4F_HIDDEN_SIZE * 2;
const GIB: u64 = 1024 * 1024 * 1024;
/// A planning ceiling only, not an allocation performed by the spine.
pub const PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES: u64 = 12 * GIB;
/// Native routed-expert cache caps reserved for the future runtime's explicit
/// hot/cold policy.  They are not a residency or throughput measurement.
pub const PROVISIONAL_ROUTED_EXPERT_HOT_CEILING_BYTES: u64 = 32 * GIB;
pub const PROVISIONAL_ROUTED_EXPERT_COLD_CEILING_BYTES: u64 = 8 * GIB;

const EMBEDDING_WEIGHT: &str = "embed.weight";
const FINAL_NORM_WEIGHT: &str = "norm.weight";
const LM_HEAD_WEIGHT: &str = "head.weight";
const HC_HEAD_FN: &str = "hc_head_fn";
const HC_HEAD_BASE: &str = "hc_head_base";
const HC_HEAD_SCALE: &str = "hc_head_scale";

const OFFICIAL_INFERENCE_MODEL_PY_SHA256: &str =
    "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71";
const OFFICIAL_INFERENCE_KERNEL_PY_SHA256: &str =
    "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2";
const OFFICIAL_INFERENCE_CONFIG_JSON_SHA256: &str =
    "6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733";
const OFFICIAL_MODEL_CONFIG_JSON_SHA256: &str =
    "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818";

/// The compression grammar of one *base* decoder layer.  This is a topology
/// binding, not a causal-state allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4CompressionMode {
    SlidingWindowOnly,
    Ratio4WithIndexer,
    Ratio128,
}

impl DeepSeekV4CompressionMode {
    pub const fn ratio(self) -> usize {
        match self {
            Self::SlidingWindowOnly => 0,
            Self::Ratio4WithIndexer => 4,
            Self::Ratio128 => 128,
        }
    }

    pub const fn has_compressor(self) -> bool {
        !matches!(self, Self::SlidingWindowOnly)
    }

    pub const fn has_indexer(self) -> bool {
        matches!(self, Self::Ratio4WithIndexer)
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::SlidingWindowOnly => "sliding_window_only",
            Self::Ratio4WithIndexer => "ratio_4_with_indexer",
            Self::Ratio128 => "ratio_128",
        }
    }
}

/// The source gate's exact discrete-decision mode for one base layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4RouterMode {
    HashTokenToExpert,
    LearnedScoresWithBias,
}

impl DeepSeekV4RouterMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::HashTokenToExpert => "hash_token_to_expert",
            Self::LearnedScoresWithBias => "learned_scores_with_bias",
        }
    }
}

/// Native projection families that are present in every base attention block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4ControlProjection {
    WqA,
    WqB,
    Wkv,
    WoA,
    WoB,
}

impl DeepSeekV4ControlProjection {
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::WqA => "attn.wq_a.weight",
            Self::WqB => "attn.wq_b.weight",
            Self::Wkv => "attn.wkv.weight",
            Self::WoA => "attn.wo_a.weight",
            Self::WoB => "attn.wo_b.weight",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WqA => "wq_a",
            Self::WqB => "wq_b",
            Self::Wkv => "wkv",
            Self::WoA => "wo_a",
            Self::WoB => "wo_b",
        }
    }

    const ALL: [Self; 5] = [Self::WqA, Self::WqB, Self::Wkv, Self::WoA, Self::WoB];
}

/// One of the three SwiGLU expert operators.  Routed experts are native FP4;
/// the source's single shared expert is native FP8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4ExpertProjection {
    W1,
    W2,
    W3,
}

impl DeepSeekV4ExpertProjection {
    pub const fn suffix(self) -> &'static str {
        match self {
            Self::W1 => "w1.weight",
            Self::W2 => "w2.weight",
            Self::W3 => "w3.weight",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::W1 => "w1",
            Self::W2 => "w2",
            Self::W3 => "w3",
        }
    }

    const ALL: [Self; 3] = [Self::W1, Self::W2, Self::W3];
}

/// Immutable, validated tensor-count and routing/compression binding for one
/// base layer.  Names are derived through the checked helpers below instead of
/// duplicating 67k strings in resident host memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4LayerBinding {
    pub layer: usize,
    pub compression: DeepSeekV4CompressionMode,
    pub router: DeepSeekV4RouterMode,
    /// Includes all scales and the 256×3 routed-expert native pair tensors.
    pub tensor_count: usize,
}

impl DeepSeekV4LayerBinding {
    pub fn control_weight_name(&self, projection: DeepSeekV4ControlProjection) -> String {
        format!("layers.{}.{}", self.layer, projection.suffix())
    }

    pub fn shared_expert_weight_name(&self, projection: DeepSeekV4ExpertProjection) -> String {
        format!(
            "layers.{}.ffn.shared_experts.{}",
            self.layer,
            projection.suffix()
        )
    }

    pub fn routed_expert_weight_name(
        &self,
        expert: usize,
        projection: DeepSeekV4ExpertProjection,
    ) -> Result<String> {
        if expert >= DSV4F_ROUTED_EXPERT_COUNT {
            return Err(spine_error(format!(
                "expert {expert} is outside 0..{} for layer {}",
                DSV4F_ROUTED_EXPERT_COUNT, self.layer
            )));
        }
        Ok(format!(
            "layers.{}.ffn.experts.{expert}.{}",
            self.layer,
            projection.suffix()
        ))
    }
}

/// Full topology of the base 43-layer child body.  The sealed stream has one
/// MTP auxiliary layer too; its tensor count is carried separately so no
/// consumer can mistake it for base causal work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4BaseBodyTopology {
    pub layers: Vec<DeepSeekV4LayerBinding>,
    pub base_tensor_count: usize,
    pub mtp_auxiliary_tensor_count: usize,
    /// Complete admitted-stream inventory, including the excluded MTP layer.
    pub global_tensor_count: usize,
    /// The six non-layer tensors which belong to the base child body:
    /// embedding, final norm, LM head, and three HC-head tensors.
    pub base_global_tensor_count: usize,
    /// Exact sealed bytes in global/layer control tensors, including the full
    /// embedding and LM head (which the spine itself only stages row-wise).
    pub static_control_tensor_bytes: u64,
    /// Exact sealed bytes in all 43×256×3 routed-expert weight/scale pairs.
    pub routed_expert_tensor_bytes: u64,
    /// The two full BF16 vocab matrices.  They are counted above but are not
    /// eligible for unbounded staging through this spine.
    pub vocab_matrix_tensor_bytes: u64,
}

impl DeepSeekV4BaseBodyTopology {
    pub fn layer(&self, layer: usize) -> Result<&DeepSeekV4LayerBinding> {
        self.layers
            .get(layer)
            .ok_or_else(|| spine_error(format!("layer {layer} is outside the 43-layer base body")))
    }
}

/// A bounded residency contract derived from the sealed base-body metadata.
/// It is a planning input for the actual executor, not an allocation, cache
/// measurement, or performance claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ResidencyPlan {
    pub all_model_materialization_allowed: bool,
    pub full_base_body_tensor_bytes: u64,
    pub static_control_tensor_bytes: u64,
    pub non_vocab_control_tensor_bytes: u64,
    pub vocab_matrix_tensor_bytes: u64,
    pub routed_expert_tensor_bytes: u64,
    pub control_resident_ceiling_bytes: u64,
    pub routed_expert_hot_ceiling_bytes: u64,
    pub routed_expert_cold_ceiling_bytes: u64,
    pub maximum_single_stage_bytes: usize,
    pub cache_contract: &'static str,
}

/// Exact source hashes which bind the data plane to the source's model and
/// kernel grammar, rather than merely to a model-name label.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4RuntimeSourceAnchors {
    pub inference_model_py_sha256: String,
    pub inference_kernel_py_sha256: String,
    pub inference_config_json_sha256: String,
    pub model_config_json_sha256: String,
}

/// Geometry checked independently in both official configs before any tensor
/// staging is exposed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4RuntimeGeometry {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub base_layer_count: usize,
    pub mtp_layer_count: usize,
    pub hash_layer_count: usize,
    pub routed_expert_count: usize,
    pub shared_expert_count: usize,
    pub activated_experts: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub q_lora_rank: usize,
    pub o_lora_rank: usize,
    pub attention_head_count: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub compression_ratios: Vec<usize>,
}

/// A verified, bounded source-native tensor payload.  This is an in-memory
/// staging result only; callers own its lifetime and it is never persisted by
/// the spine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4StagedTensor {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<u64>,
    pub source_shard: String,
    pub range: Range<u64>,
    pub bytes: Vec<u8>,
}

/// A verified native weight/scale pair staged together for one future device
/// upload.  The pair geometry has already been checked by the full reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4StagedNativePair {
    pub kind: NativeScalePairKind,
    pub weight: DeepSeekV4StagedTensor,
    pub scale: DeepSeekV4StagedTensor,
    pub logical_k: u64,
    pub out_rows: u64,
}

/// The public capability gate.  Every unavailable field is explicit so the
/// future executor cannot be confused with a registered serving engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4RuntimeCapabilityGate {
    pub full_stream_reader_admitted: bool,
    pub source_hashes_and_configs_validated: bool,
    pub all_base_operator_bindings_validated: bool,
    pub bounded_tensor_staging_available: bool,
    pub embedding_final_norm_head_primitives_available: bool,
    pub registered_43_layer_engine: bool,
    pub causal_forward_available: bool,
    pub continuation_available: bool,
    pub metal_dispatches_available: bool,
    pub hcli_endpoint_available: bool,
    pub numeric_parity_v21_passed: bool,
    pub base_true_tps_eligible: bool,
    pub next_parity_rung: &'static str,
}

impl DeepSeekV4RuntimeCapabilityGate {
    /// Fail closed at the public-runtime boundary.  This exact denial is used
    /// by the receipt producer as a regression guard for the CLI refusal.
    pub fn require_full_causal_runtime(&self) -> Result<()> {
        if self.registered_43_layer_engine
            && self.causal_forward_available
            && self.continuation_available
            && self.metal_dispatches_available
            && self.hcli_endpoint_available
            && self.numeric_parity_v21_passed
        {
            return Ok(());
        }
        Err(spine_error(
            "43-layer causal Engine is not registered: this source-bound spine stages data only; the current next parity rung is P3 (mHC + norm + Q path)",
        ))
    }
}

/// Validated owner of the sealed reader plus the future-executor data-plane
/// contract.  It deliberately does not implement [`crate::Engine`].
pub struct DeepSeekV4RuntimeSpine {
    reader: DeepSeekV4FullStreamReader,
    anchors: DeepSeekV4RuntimeSourceAnchors,
    geometry: DeepSeekV4RuntimeGeometry,
    topology: DeepSeekV4BaseBodyTopology,
    residency_plan: DeepSeekV4ResidencyPlan,
    capabilities: DeepSeekV4RuntimeCapabilityGate,
}

impl DeepSeekV4RuntimeSpine {
    /// Admit the sealed full stream, authenticate its source grammar, and
    /// eagerly validate every base decoder operator binding.  It does not read
    /// any weight payload until a bounded staging method is called.
    pub fn admit(root: impl AsRef<std::path::Path>) -> Result<Self> {
        let reader = DeepSeekV4FullStreamReader::admit(root)?;
        let anchors = validate_source_anchors(&reader)?;
        let geometry = validate_geometry(&reader)?;
        let topology = validate_base_body_topology(&reader, &geometry)?;
        let residency_plan = build_residency_plan(&topology)?;
        let capabilities = DeepSeekV4RuntimeCapabilityGate {
            full_stream_reader_admitted: true,
            source_hashes_and_configs_validated: true,
            all_base_operator_bindings_validated: true,
            bounded_tensor_staging_available: true,
            embedding_final_norm_head_primitives_available: true,
            registered_43_layer_engine: false,
            causal_forward_available: false,
            continuation_available: false,
            metal_dispatches_available: false,
            hcli_endpoint_available: false,
            numeric_parity_v21_passed: false,
            base_true_tps_eligible: false,
            // P0/P1/P2 are owned by existing source-linear evidence; this
            // module's data plane directly unblocks the source mHC/Q path.
            next_parity_rung: "P3_MHC_NORM_Q_PATH",
        };
        Ok(Self {
            reader,
            anchors,
            geometry,
            topology,
            residency_plan,
            capabilities,
        })
    }

    pub fn reader(&self) -> &DeepSeekV4FullStreamReader {
        &self.reader
    }

    pub fn source_anchors(&self) -> &DeepSeekV4RuntimeSourceAnchors {
        &self.anchors
    }

    pub fn geometry(&self) -> &DeepSeekV4RuntimeGeometry {
        &self.geometry
    }

    pub fn topology(&self) -> &DeepSeekV4BaseBodyTopology {
        &self.topology
    }

    pub fn residency_plan(&self) -> &DeepSeekV4ResidencyPlan {
        &self.residency_plan
    }

    pub fn capabilities(&self) -> &DeepSeekV4RuntimeCapabilityGate {
        &self.capabilities
    }

    /// Bound a generic base-body tensor read.  The name must be in the
    /// validated base topology (not MTP) and callers must state a ceiling no
    /// larger than [`MAX_STAGED_OPERATOR_BYTES`].
    pub fn stage_base_tensor_range(
        &self,
        name: &str,
        range: Range<u64>,
        max_output_bytes: usize,
    ) -> Result<DeepSeekV4StagedTensor> {
        if !is_base_body_tensor_name(name) {
            return Err(spine_error(format!(
                "{name:?} is not a base-body tensor eligible for runtime staging"
            )));
        }
        if max_output_bytes == 0 || max_output_bytes > MAX_STAGED_OPERATOR_BYTES {
            return Err(spine_error(format!(
                "runtime staging ceiling must be within 1..={MAX_STAGED_OPERATOR_BYTES} bytes"
            )));
        }
        let metadata = self.reader.tensor_metadata(name)?;
        if range.start >= range.end || range.end > metadata.bytes {
            return Err(spine_error(format!(
                "{name}: invalid staging range {}..{} for {} bytes",
                range.start, range.end, metadata.bytes
            )));
        }
        let requested = usize::try_from(range.end - range.start)
            .map_err(|_| spine_error(format!("{name}: staging range exceeds host usize")))?;
        if requested > max_output_bytes {
            return Err(spine_error(format!(
                "{name}: requested {requested} bytes exceeds explicit staging ceiling {max_output_bytes}"
            )));
        }
        let bytes = self
            .reader
            .read_verified_range(name, range.clone(), max_output_bytes)?;
        Ok(DeepSeekV4StagedTensor {
            name: metadata.name.clone(),
            dtype: metadata.dtype.clone(),
            shape: metadata.shape.clone(),
            source_shard: metadata.source_shard.clone(),
            range,
            bytes,
        })
    }

    /// Stage one native FP8 attention-control pair.  This remains a data
    /// upload primitive; it does not encode a command buffer or projection.
    pub fn stage_control_pair(
        &self,
        layer: usize,
        projection: DeepSeekV4ControlProjection,
    ) -> Result<DeepSeekV4StagedNativePair> {
        let binding = self.topology.layer(layer)?;
        let weight_name = binding.control_weight_name(projection);
        self.stage_native_pair(&weight_name, NativeScalePairKind::Fp8E4M3fn)
    }

    /// Stage one native FP8 shared-expert pair for a validated base layer.
    pub fn stage_shared_expert_pair(
        &self,
        layer: usize,
        projection: DeepSeekV4ExpertProjection,
    ) -> Result<DeepSeekV4StagedNativePair> {
        let binding = self.topology.layer(layer)?;
        let weight_name = binding.shared_expert_weight_name(projection);
        self.stage_native_pair(&weight_name, NativeScalePairKind::Fp8E4M3fn)
    }

    /// Stage one native FP4 routed-expert pair for a validated base layer and
    /// expert id.  Residency/prefetch policy stays outside this primitive.
    pub fn stage_routed_expert_pair(
        &self,
        layer: usize,
        expert: usize,
        projection: DeepSeekV4ExpertProjection,
    ) -> Result<DeepSeekV4StagedNativePair> {
        let binding = self.topology.layer(layer)?;
        let weight_name = binding.routed_expert_weight_name(expert, projection)?;
        self.stage_native_pair(&weight_name, NativeScalePairKind::Fp4E2M1fnX2)
    }

    /// Exact BF16 storage row for one embedding token.  It verifies the
    /// source chunk before returning the 4096 native BF16 elements.
    pub fn load_embedding_row_bf16(&self, token_id: u32) -> Result<Vec<u16>> {
        self.load_bf16_matrix_row(EMBEDDING_WEIGHT, token_id as usize)
    }

    /// Exact BF16 storage row for one LM-head vocabulary output.  This is a
    /// bounded row loader only; it is not a full logits calculation.
    pub fn load_lm_head_row_bf16(&self, token_id: u32) -> Result<Vec<u16>> {
        self.load_bf16_matrix_row(LM_HEAD_WEIGHT, token_id as usize)
    }

    /// Exact BF16 final RMSNorm parameter vector from the sealed source.
    pub fn load_final_norm_bf16(&self) -> Result<Vec<u16>> {
        let staged = self.stage_base_tensor_range(
            FINAL_NORM_WEIGHT,
            0..BF16_VECTOR_BYTES as u64,
            BF16_VECTOR_BYTES,
        )?;
        decode_bf16_bits(&staged.bytes, FINAL_NORM_WEIGHT)
    }

    /// Decode source-native BF16 storage bits without changing their ordering
    /// or precision.  It is exposed separately so a future Metal uploader can
    /// use the raw bits while a CPU parity oracle can use f32 values.
    pub fn bf16_bits_to_f32(bits: &[u16]) -> Vec<f32> {
        bits.iter()
            .map(|value| bf16::from_bits(*value).to_f32())
            .collect()
    }

    /// Explicitly reject the only unsafe storage policy for this source body:
    /// materializing all ~160 GB of source-native weights into unified memory.
    /// The future causal runtime must use the bounded control/cache plan and
    /// measured prefetch/residency policy instead.
    pub fn reject_all_model_materialization(&self) -> Result<()> {
        Err(spine_error(format!(
            "all-model materialization is forbidden: base body has {} bytes and the spine permits at most {} bytes per source staging request",
            self.residency_plan.full_base_body_tensor_bytes,
            self.residency_plan.maximum_single_stage_bytes,
        )))
    }

    fn stage_native_pair(
        &self,
        weight_name: &str,
        expected_kind: NativeScalePairKind,
    ) -> Result<DeepSeekV4StagedNativePair> {
        let pair = self.reader.native_scale_pair(weight_name)?;
        if pair.kind != expected_kind {
            return Err(spine_error(format!(
                "{weight_name}: native pair kind {} differs from required {}",
                pair.kind.as_str(),
                expected_kind.as_str()
            )));
        }
        let weight = self.stage_base_tensor_range(
            weight_name,
            0..pair.weight.bytes,
            MAX_STAGED_OPERATOR_BYTES,
        )?;
        let scale = self.stage_base_tensor_range(
            &pair.scale.name,
            0..pair.scale.bytes,
            MAX_STAGED_OPERATOR_BYTES,
        )?;
        Ok(DeepSeekV4StagedNativePair {
            kind: pair.kind,
            weight,
            scale,
            logical_k: pair.logical_k,
            out_rows: pair.out_rows,
        })
    }

    fn load_bf16_matrix_row(&self, name: &str, row: usize) -> Result<Vec<u16>> {
        if row >= DSV4F_VOCAB_SIZE {
            return Err(spine_error(format!(
                "{name}: token id {row} is outside 0..{DSV4F_VOCAB_SIZE}"
            )));
        }
        let start = row
            .checked_mul(BF16_VECTOR_BYTES)
            .ok_or_else(|| spine_error(format!("{name}: row byte offset overflow")))?;
        let end = start
            .checked_add(BF16_VECTOR_BYTES)
            .ok_or_else(|| spine_error(format!("{name}: row byte end overflow")))?;
        let staged =
            self.stage_base_tensor_range(name, start as u64..end as u64, BF16_VECTOR_BYTES)?;
        decode_bf16_bits(&staged.bytes, name)
    }
}

fn validate_source_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4RuntimeSourceAnchors> {
    let anchors = DeepSeekV4RuntimeSourceAnchors {
        inference_model_py_sha256: reader
            .source_metadata_asset_sha256("inference/model.py")?
            .to_owned(),
        inference_kernel_py_sha256: reader
            .source_metadata_asset_sha256("inference/kernel.py")?
            .to_owned(),
        inference_config_json_sha256: reader
            .source_metadata_asset_sha256("inference/config.json")?
            .to_owned(),
        model_config_json_sha256: reader
            .source_metadata_asset_sha256("config.json")?
            .to_owned(),
    };
    if anchors.inference_model_py_sha256 != OFFICIAL_INFERENCE_MODEL_PY_SHA256
        || anchors.inference_kernel_py_sha256 != OFFICIAL_INFERENCE_KERNEL_PY_SHA256
        || anchors.inference_config_json_sha256 != OFFICIAL_INFERENCE_CONFIG_JSON_SHA256
        || anchors.model_config_json_sha256 != OFFICIAL_MODEL_CONFIG_JSON_SHA256
    {
        return Err(spine_error(
            "official source model/kernel/config hashes differ from the runtime-spine anchors",
        ));
    }
    Ok(anchors)
}

fn validate_geometry(reader: &DeepSeekV4FullStreamReader) -> Result<DeepSeekV4RuntimeGeometry> {
    let model = parse_json_asset(reader, "config.json", 64 * 1024)?;
    let inference = parse_json_asset(reader, "inference/config.json", 64 * 1024)?;
    expect_string(&model, "model_type", "deepseek_v4", "config.json")?;
    expect_string(&model, "torch_dtype", "bfloat16", "config.json")?;
    expect_string(&model, "expert_dtype", "fp4", "config.json")?;
    expect_string(&inference, "dtype", "fp8", "inference/config.json")?;
    expect_string(&inference, "expert_dtype", "fp4", "inference/config.json")?;
    expect_string(&inference, "scale_fmt", "ue8m0", "inference/config.json")?;

    let geometry = DeepSeekV4RuntimeGeometry {
        hidden_size: checked_usize(
            json_u64(&model, "hidden_size", "config.json")?,
            "hidden_size",
        )?,
        vocab_size: checked_usize(json_u64(&model, "vocab_size", "config.json")?, "vocab_size")?,
        base_layer_count: checked_usize(
            json_u64(&model, "num_hidden_layers", "config.json")?,
            "num_hidden_layers",
        )?,
        mtp_layer_count: checked_usize(
            json_u64(&model, "num_nextn_predict_layers", "config.json")?,
            "num_nextn_predict_layers",
        )?,
        hash_layer_count: checked_usize(
            json_u64(&model, "num_hash_layers", "config.json")?,
            "num_hash_layers",
        )?,
        routed_expert_count: checked_usize(
            json_u64(&model, "n_routed_experts", "config.json")?,
            "n_routed_experts",
        )?,
        shared_expert_count: checked_usize(
            json_u64(&model, "n_shared_experts", "config.json")?,
            "n_shared_experts",
        )?,
        activated_experts: checked_usize(
            json_u64(&model, "num_experts_per_tok", "config.json")?,
            "num_experts_per_tok",
        )?,
        hc_mult: checked_usize(json_u64(&model, "hc_mult", "config.json")?, "hc_mult")?,
        hc_sinkhorn_iters: checked_usize(
            json_u64(&model, "hc_sinkhorn_iters", "config.json")?,
            "hc_sinkhorn_iters",
        )?,
        q_lora_rank: checked_usize(
            json_u64(&model, "q_lora_rank", "config.json")?,
            "q_lora_rank",
        )?,
        o_lora_rank: checked_usize(
            json_u64(&model, "o_lora_rank", "config.json")?,
            "o_lora_rank",
        )?,
        attention_head_count: checked_usize(
            json_u64(&model, "num_attention_heads", "config.json")?,
            "num_attention_heads",
        )?,
        head_dim: checked_usize(json_u64(&model, "head_dim", "config.json")?, "head_dim")?,
        rope_head_dim: checked_usize(
            json_u64(&model, "qk_rope_head_dim", "config.json")?,
            "qk_rope_head_dim",
        )?,
        compression_ratios: json_usize_array(
            &inference,
            "compress_ratios",
            "inference/config.json",
        )?,
    };
    if geometry.hidden_size != DSV4F_HIDDEN_SIZE
        || geometry.vocab_size != DSV4F_VOCAB_SIZE
        || geometry.base_layer_count != DSV4F_BASE_LAYER_COUNT
        || geometry.mtp_layer_count != DSV4F_MTP_LAYER_COUNT
        || geometry.hash_layer_count != 3
        || geometry.routed_expert_count != DSV4F_ROUTED_EXPERT_COUNT
        || geometry.shared_expert_count != 1
        || geometry.activated_experts != DSV4F_TOP_K_EXPERTS
        || geometry.hc_mult != DSV4F_HC_MULT
        || geometry.hc_sinkhorn_iters != DSV4F_HC_SINKHORN_ITERS
        || geometry.q_lora_rank != 1024
        || geometry.o_lora_rank != 1024
        || geometry.attention_head_count != 64
        || geometry.head_dim != 512
        || geometry.rope_head_dim != 64
        || geometry.compression_ratios.len() != DSV4F_BASE_LAYER_COUNT + DSV4F_MTP_LAYER_COUNT
    {
        return Err(spine_error(
            "config geometry differs from the pinned 43-layer DeepSeek-V4 child body",
        ));
    }

    // Cross-check every duplicated runtime-function field between the public
    // model config and the official inference configuration.
    for (model_key, inference_key, expected) in [
        ("vocab_size", "vocab_size", geometry.vocab_size),
        ("hidden_size", "dim", geometry.hidden_size),
        ("num_hidden_layers", "n_layers", geometry.base_layer_count),
        (
            "num_hash_layers",
            "n_hash_layers",
            geometry.hash_layer_count,
        ),
        (
            "n_routed_experts",
            "n_routed_experts",
            geometry.routed_expert_count,
        ),
        (
            "n_shared_experts",
            "n_shared_experts",
            geometry.shared_expert_count,
        ),
        (
            "num_experts_per_tok",
            "n_activated_experts",
            geometry.activated_experts,
        ),
        ("hc_mult", "hc_mult", geometry.hc_mult),
        (
            "hc_sinkhorn_iters",
            "hc_sinkhorn_iters",
            geometry.hc_sinkhorn_iters,
        ),
        ("q_lora_rank", "q_lora_rank", geometry.q_lora_rank),
        ("o_lora_rank", "o_lora_rank", geometry.o_lora_rank),
        (
            "num_attention_heads",
            "n_heads",
            geometry.attention_head_count,
        ),
        ("head_dim", "head_dim", geometry.head_dim),
        ("qk_rope_head_dim", "rope_head_dim", geometry.rope_head_dim),
    ] {
        let actual = checked_usize(
            json_u64(&inference, inference_key, "inference/config.json")?,
            inference_key,
        )?;
        if actual != expected {
            return Err(spine_error(format!(
                "config mismatch: config.json.{model_key}={expected} but inference/config.json.{inference_key}={actual}"
            )));
        }
    }
    Ok(geometry)
}

fn validate_base_body_topology(
    reader: &DeepSeekV4FullStreamReader,
    geometry: &DeepSeekV4RuntimeGeometry,
) -> Result<DeepSeekV4BaseBodyTopology> {
    validate_bf16_matrix(
        reader,
        EMBEDDING_WEIGHT,
        DSV4F_VOCAB_SIZE,
        DSV4F_HIDDEN_SIZE,
    )?;
    validate_bf16_vector(reader, FINAL_NORM_WEIGHT, DSV4F_HIDDEN_SIZE)?;
    validate_bf16_matrix(reader, LM_HEAD_WEIGHT, DSV4F_VOCAB_SIZE, DSV4F_HIDDEN_SIZE)?;
    validate_f32_matrix(
        reader,
        HC_HEAD_FN,
        DSV4F_HC_MULT,
        DSV4F_HC_MULT * DSV4F_HIDDEN_SIZE,
    )?;
    validate_f32_vector(reader, HC_HEAD_BASE, DSV4F_HC_MULT)?;
    validate_f32_vector(reader, HC_HEAD_SCALE, 1)?;

    let mut layers = Vec::with_capacity(DSV4F_BASE_LAYER_COUNT);
    let mut base_tensor_count = 6usize;
    for layer in 0..geometry.base_layer_count {
        let ratio = *geometry.compression_ratios.get(layer).ok_or_else(|| {
            spine_error(format!("missing compression ratio for base layer {layer}"))
        })?;
        let compression = match ratio {
            0 => DeepSeekV4CompressionMode::SlidingWindowOnly,
            4 => DeepSeekV4CompressionMode::Ratio4WithIndexer,
            128 => DeepSeekV4CompressionMode::Ratio128,
            other => {
                return Err(spine_error(format!(
                    "base layer {layer} has unsupported source compression ratio {other}"
                )))
            }
        };
        let router = if layer < geometry.hash_layer_count {
            DeepSeekV4RouterMode::HashTokenToExpert
        } else {
            DeepSeekV4RouterMode::LearnedScoresWithBias
        };
        validate_layer_bindings(reader, layer, compression, router)?;
        let tensor_count = base_layer_tensor_count(compression);
        base_tensor_count = base_tensor_count
            .checked_add(tensor_count)
            .ok_or_else(|| spine_error("base tensor count overflow"))?;
        layers.push(DeepSeekV4LayerBinding {
            layer,
            compression,
            router,
            tensor_count,
        });
    }

    // The admission reader contains the full source stream.  Deriving this
    // count means the future base executor can assert that it neither omits a
    // base tensor nor silently includes the auxiliary MTP body.
    let mtp_auxiliary_tensor_count = reader
        .tensor_count()
        .checked_sub(base_tensor_count)
        .ok_or_else(|| spine_error("base topology exceeds admitted tensor inventory"))?;
    if base_tensor_count != 67_612 || mtp_auxiliary_tensor_count != 1_575 {
        return Err(spine_error(format!(
            "unexpected base/MTP tensor partition: base={base_tensor_count}, mtp={mtp_auxiliary_tensor_count}"
        )));
    }
    let (static_control_tensor_bytes, routed_expert_tensor_bytes, vocab_matrix_tensor_bytes) =
        base_body_byte_ledger(reader, geometry)?;
    let base_bytes = static_control_tensor_bytes
        .checked_add(routed_expert_tensor_bytes)
        .ok_or_else(|| spine_error("base byte ledger overflow"))?;
    if base_bytes > reader.tensor_bytes() || vocab_matrix_tensor_bytes > static_control_tensor_bytes
    {
        return Err(spine_error(
            "base byte ledger is inconsistent with admitted stream",
        ));
    }
    Ok(DeepSeekV4BaseBodyTopology {
        layers,
        base_tensor_count,
        mtp_auxiliary_tensor_count,
        global_tensor_count: reader.tensor_count(),
        base_global_tensor_count: 6,
        static_control_tensor_bytes,
        routed_expert_tensor_bytes,
        vocab_matrix_tensor_bytes,
    })
}

fn validate_layer_bindings(
    reader: &DeepSeekV4FullStreamReader,
    layer: usize,
    compression: DeepSeekV4CompressionMode,
    router: DeepSeekV4RouterMode,
) -> Result<()> {
    let prefix = format!("layers.{layer}");
    validate_f32_vector(reader, &format!("{prefix}.attn.attn_sink"), 64)?;
    validate_bf16_vector(reader, &format!("{prefix}.attn.q_norm.weight"), 1024)?;
    validate_bf16_vector(reader, &format!("{prefix}.attn.kv_norm.weight"), 512)?;
    validate_bf16_vector(
        reader,
        &format!("{prefix}.attn_norm.weight"),
        DSV4F_HIDDEN_SIZE,
    )?;
    validate_bf16_vector(
        reader,
        &format!("{prefix}.ffn_norm.weight"),
        DSV4F_HIDDEN_SIZE,
    )?;

    for projection in DeepSeekV4ControlProjection::ALL {
        let weight = format!("{prefix}.{}", projection.suffix());
        validate_native_pair(reader, &weight, NativeScalePairKind::Fp8E4M3fn)?;
    }

    validate_bf16_matrix(
        reader,
        &format!("{prefix}.ffn.gate.weight"),
        256,
        DSV4F_HIDDEN_SIZE,
    )?;
    match router {
        DeepSeekV4RouterMode::HashTokenToExpert => {
            validate_i64_matrix(
                reader,
                &format!("{prefix}.ffn.gate.tid2eid"),
                DSV4F_VOCAB_SIZE,
                DSV4F_TOP_K_EXPERTS,
            )?;
        }
        DeepSeekV4RouterMode::LearnedScoresWithBias => {
            validate_f32_vector(reader, &format!("{prefix}.ffn.gate.bias"), 256)?;
        }
    }

    for projection in DeepSeekV4ExpertProjection::ALL {
        let shared = format!("{prefix}.ffn.shared_experts.{}", projection.suffix());
        validate_native_pair(reader, &shared, NativeScalePairKind::Fp8E4M3fn)?;
    }
    for expert in 0..DSV4F_ROUTED_EXPERT_COUNT {
        for projection in DeepSeekV4ExpertProjection::ALL {
            let routed = format!("{prefix}.ffn.experts.{expert}.{}", projection.suffix());
            validate_native_pair(reader, &routed, NativeScalePairKind::Fp4E2M1fnX2)?;
        }
    }

    let hc_width = DSV4F_HC_MULT * DSV4F_HIDDEN_SIZE;
    let hc_mix = (2 + DSV4F_HC_MULT) * DSV4F_HC_MULT;
    for stem in ["hc_attn", "hc_ffn"] {
        validate_f32_matrix(reader, &format!("{prefix}.{stem}_fn"), hc_mix, hc_width)?;
        validate_f32_vector(reader, &format!("{prefix}.{stem}_base"), hc_mix)?;
        validate_f32_vector(reader, &format!("{prefix}.{stem}_scale"), 3)?;
    }

    if compression.has_compressor() {
        validate_compressor_bindings(reader, &prefix, compression.ratio(), false)?;
    }
    if compression.has_indexer() {
        validate_compressor_bindings(reader, &format!("{prefix}.attn.indexer"), 4, true)?;
        validate_native_pair(
            reader,
            &format!("{prefix}.attn.indexer.wq_b.weight"),
            NativeScalePairKind::Fp8E4M3fn,
        )?;
        validate_bf16_matrix(
            reader,
            &format!("{prefix}.attn.indexer.weights_proj.weight"),
            64,
            DSV4F_HIDDEN_SIZE,
        )?;
    }
    Ok(())
}

fn validate_compressor_bindings(
    reader: &DeepSeekV4FullStreamReader,
    layer_prefix: &str,
    ratio: usize,
    indexer: bool,
) -> Result<()> {
    let head_dim = if indexer { 128 } else { 512 };
    let overlap_multiplier = if ratio == 4 { 2 } else { 1 };
    let compressor = if indexer {
        format!("{layer_prefix}.compressor")
    } else {
        format!("{layer_prefix}.attn.compressor")
    };
    validate_f32_matrix(
        reader,
        &format!("{compressor}.ape"),
        ratio,
        overlap_multiplier * head_dim,
    )?;
    validate_bf16_matrix(
        reader,
        &format!("{compressor}.wkv.weight"),
        overlap_multiplier * head_dim,
        DSV4F_HIDDEN_SIZE,
    )?;
    validate_bf16_matrix(
        reader,
        &format!("{compressor}.wgate.weight"),
        overlap_multiplier * head_dim,
        DSV4F_HIDDEN_SIZE,
    )?;
    validate_bf16_vector(reader, &format!("{compressor}.norm.weight"), head_dim)
}

fn build_residency_plan(topology: &DeepSeekV4BaseBodyTopology) -> Result<DeepSeekV4ResidencyPlan> {
    let full_base_body_tensor_bytes = topology
        .static_control_tensor_bytes
        .checked_add(topology.routed_expert_tensor_bytes)
        .ok_or_else(|| spine_error("residency plan base byte overflow"))?;
    let non_vocab_control_tensor_bytes = topology
        .static_control_tensor_bytes
        .checked_sub(topology.vocab_matrix_tensor_bytes)
        .ok_or_else(|| spine_error("residency plan vocabulary byte underflow"))?;
    if topology.static_control_tensor_bytes > PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES {
        return Err(spine_error(format!(
            "control body has {} bytes, exceeding its provisional {}-byte ceiling",
            topology.static_control_tensor_bytes, PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES
        )));
    }
    Ok(DeepSeekV4ResidencyPlan {
        all_model_materialization_allowed: false,
        full_base_body_tensor_bytes,
        static_control_tensor_bytes: topology.static_control_tensor_bytes,
        non_vocab_control_tensor_bytes,
        vocab_matrix_tensor_bytes: topology.vocab_matrix_tensor_bytes,
        routed_expert_tensor_bytes: topology.routed_expert_tensor_bytes,
        control_resident_ceiling_bytes: PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES,
        routed_expert_hot_ceiling_bytes: PROVISIONAL_ROUTED_EXPERT_HOT_CEILING_BYTES,
        routed_expert_cold_ceiling_bytes: PROVISIONAL_ROUTED_EXPERT_COLD_CEILING_BYTES,
        maximum_single_stage_bytes: MAX_STAGED_OPERATOR_BYTES,
        cache_contract: "control residency is bounded; embedding/head remain row-stage-only in this spine; routed FP4 experts require an explicit bounded hot/cold cache with verified promotion and eviction; all-model materialization is forbidden",
    })
}

fn base_body_byte_ledger(
    reader: &DeepSeekV4FullStreamReader,
    geometry: &DeepSeekV4RuntimeGeometry,
) -> Result<(u64, u64, u64)> {
    let mut control = 0u64;
    let mut routed = 0u64;
    for name in [
        EMBEDDING_WEIGHT,
        LM_HEAD_WEIGHT,
        FINAL_NORM_WEIGHT,
        HC_HEAD_FN,
        HC_HEAD_BASE,
        HC_HEAD_SCALE,
    ] {
        add_tensor_bytes(reader, name, &mut control)?;
    }
    let vocab_matrix_tensor_bytes = reader
        .tensor_metadata(EMBEDDING_WEIGHT)?
        .bytes
        .checked_add(reader.tensor_metadata(LM_HEAD_WEIGHT)?.bytes)
        .ok_or_else(|| spine_error("vocabulary matrix byte overflow"))?;
    for layer in 0..geometry.base_layer_count {
        let prefix = format!("layers.{layer}");
        for suffix in [
            "attn.attn_sink",
            "attn.q_norm.weight",
            "attn.kv_norm.weight",
            "attn_norm.weight",
            "ffn_norm.weight",
        ] {
            add_tensor_bytes(reader, &format!("{prefix}.{suffix}"), &mut control)?;
        }
        for projection in DeepSeekV4ControlProjection::ALL {
            let weight = format!("{prefix}.{}", projection.suffix());
            add_pair_bytes(reader, &weight, &mut control)?;
        }
        add_tensor_bytes(reader, &format!("{prefix}.ffn.gate.weight"), &mut control)?;
        let router_tensor = if layer < geometry.hash_layer_count {
            format!("{prefix}.ffn.gate.tid2eid")
        } else {
            format!("{prefix}.ffn.gate.bias")
        };
        add_tensor_bytes(reader, &router_tensor, &mut control)?;
        for projection in DeepSeekV4ExpertProjection::ALL {
            add_pair_bytes(
                reader,
                &format!("{prefix}.ffn.shared_experts.{}", projection.suffix()),
                &mut control,
            )?;
        }
        for stem in ["hc_attn", "hc_ffn"] {
            for suffix in ["fn", "base", "scale"] {
                add_tensor_bytes(reader, &format!("{prefix}.{stem}_{suffix}"), &mut control)?;
            }
        }
        let ratio = geometry.compression_ratios[layer];
        if ratio != 0 {
            let compressor = format!("{prefix}.attn.compressor");
            for suffix in ["ape", "wkv.weight", "wgate.weight", "norm.weight"] {
                add_tensor_bytes(reader, &format!("{compressor}.{suffix}"), &mut control)?;
            }
        }
        if ratio == 4 {
            let compressor = format!("{prefix}.attn.indexer.compressor");
            for suffix in ["ape", "wkv.weight", "wgate.weight", "norm.weight"] {
                add_tensor_bytes(reader, &format!("{compressor}.{suffix}"), &mut control)?;
            }
            add_pair_bytes(
                reader,
                &format!("{prefix}.attn.indexer.wq_b.weight"),
                &mut control,
            )?;
            add_tensor_bytes(
                reader,
                &format!("{prefix}.attn.indexer.weights_proj.weight"),
                &mut control,
            )?;
        }
        for expert in 0..DSV4F_ROUTED_EXPERT_COUNT {
            for projection in DeepSeekV4ExpertProjection::ALL {
                add_pair_bytes(
                    reader,
                    &format!("{prefix}.ffn.experts.{expert}.{}", projection.suffix()),
                    &mut routed,
                )?;
            }
        }
    }
    Ok((control, routed, vocab_matrix_tensor_bytes))
}

fn add_tensor_bytes(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    total: &mut u64,
) -> Result<()> {
    *total = total
        .checked_add(reader.tensor_metadata(name)?.bytes)
        .ok_or_else(|| spine_error("source byte ledger overflow"))?;
    Ok(())
}

fn add_pair_bytes(
    reader: &DeepSeekV4FullStreamReader,
    weight: &str,
    total: &mut u64,
) -> Result<()> {
    let pair = reader.native_scale_pair(weight)?;
    *total = total
        .checked_add(pair.weight.bytes)
        .and_then(|value| value.checked_add(pair.scale.bytes))
        .ok_or_else(|| spine_error("source native-pair byte ledger overflow"))?;
    Ok(())
}

fn base_layer_tensor_count(compression: DeepSeekV4CompressionMode) -> usize {
    // 14 attention tensors + 2 norms + gate pair + 6 HC tensors + 6 shared
    // tensors + 256 × (weight,scale) × 3 routed-expert tensors = 1565.
    let base = 1_565;
    base + match compression {
        DeepSeekV4CompressionMode::SlidingWindowOnly => 0,
        DeepSeekV4CompressionMode::Ratio128 => 4,
        DeepSeekV4CompressionMode::Ratio4WithIndexer => 11,
    }
}

fn validate_native_pair(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    expected: NativeScalePairKind,
) -> Result<()> {
    let pair = reader.native_scale_pair(name)?;
    if pair.kind != expected || pair.out_rows == 0 || pair.logical_k == 0 {
        return Err(spine_error(format!(
            "{name}: invalid native pair kind/geometry for runtime topology"
        )));
    }
    Ok(())
}

fn validate_bf16_vector(reader: &DeepSeekV4FullStreamReader, name: &str, len: usize) -> Result<()> {
    validate_tensor(reader, name, "BF16", &[len as u64], len * 2)
}

fn validate_bf16_matrix(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_tensor(
        reader,
        name,
        "BF16",
        &[rows as u64, cols as u64],
        rows * cols * 2,
    )
}

fn validate_f32_vector(reader: &DeepSeekV4FullStreamReader, name: &str, len: usize) -> Result<()> {
    validate_tensor(reader, name, "F32", &[len as u64], len * 4)
}

fn validate_f32_matrix(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_tensor(
        reader,
        name,
        "F32",
        &[rows as u64, cols as u64],
        rows * cols * 4,
    )
}

fn validate_i64_matrix(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<()> {
    validate_tensor(
        reader,
        name,
        "I64",
        &[rows as u64, cols as u64],
        rows * cols * 8,
    )
}

fn validate_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    dtype: &str,
    shape: &[u64],
    expected_bytes: usize,
) -> Result<()> {
    let metadata = reader.tensor_metadata(name)?;
    if metadata.dtype != dtype
        || metadata.shape.as_slice() != shape
        || metadata.bytes != expected_bytes as u64
        || metadata.segments.is_empty()
    {
        return Err(spine_error(format!(
            "{name}: source tensor metadata differs from required {dtype} {shape:?} / {expected_bytes} bytes"
        )));
    }
    Ok(())
}

fn is_base_body_tensor_name(name: &str) -> bool {
    if matches!(
        name,
        EMBEDDING_WEIGHT
            | FINAL_NORM_WEIGHT
            | LM_HEAD_WEIGHT
            | HC_HEAD_FN
            | HC_HEAD_BASE
            | HC_HEAD_SCALE
    ) {
        return true;
    }
    let Some(rest) = name.strip_prefix("layers.") else {
        return false;
    };
    let Some((layer, _)) = rest.split_once('.') else {
        return false;
    };
    layer
        .parse::<usize>()
        .map(|index| index < DSV4F_BASE_LAYER_COUNT)
        .unwrap_or(false)
}

fn decode_bf16_bits(bytes: &[u8], label: &str) -> Result<Vec<u16>> {
    if bytes.len() != BF16_VECTOR_BYTES {
        return Err(spine_error(format!(
            "{label}: expected {BF16_VECTOR_BYTES} BF16 bytes, got {}",
            bytes.len()
        )));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect())
}

fn parse_json_asset(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    bound: usize,
) -> Result<Value> {
    let bytes = reader.read_verified_metadata_asset(name, bound)?;
    serde_json::from_slice(&bytes)
        .map_err(|error| spine_error(format!("{name}: invalid authenticated JSON: {error}")))
}

fn json_u64(value: &Value, key: &str, label: &str) -> Result<u64> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| spine_error(format!("{label}: missing unsigned integer {key:?}")))
}

fn json_usize_array(value: &Value, key: &str, label: &str) -> Result<Vec<usize>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| spine_error(format!("{label}: missing integer array {key:?}")))?
        .iter()
        .map(|item| {
            checked_usize(
                item.as_u64().ok_or_else(|| {
                    spine_error(format!("{label}: {key:?} has a non-unsigned-integer item"))
                })?,
                key,
            )
        })
        .collect()
}

fn checked_usize(value: u64, label: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| spine_error(format!("{label}: {value} does not fit host usize")))
}

fn expect_string(value: &Value, key: &str, expected: &str, label: &str) -> Result<()> {
    let actual = value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| spine_error(format!("{label}: missing string {key:?}")))?;
    if actual != expected {
        return Err(spine_error(format!(
            "{label}: {key:?}={actual:?}, expected {expected:?}"
        )));
    }
    Ok(())
}

fn spine_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!("DeepSeek-V4 runtime spine: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_body_tensor_partition_is_exact() {
        let mut total = 6usize;
        total += 2 * base_layer_tensor_count(DeepSeekV4CompressionMode::SlidingWindowOnly);
        for _ in 0..20 {
            total += base_layer_tensor_count(DeepSeekV4CompressionMode::Ratio4WithIndexer);
            total += base_layer_tensor_count(DeepSeekV4CompressionMode::Ratio128);
        }
        total += base_layer_tensor_count(DeepSeekV4CompressionMode::Ratio4WithIndexer);
        assert_eq!(total, 67_612);
    }

    #[test]
    fn bfloat16_storage_decode_is_little_endian_and_exact_width() {
        let mut bytes = vec![0_u8; BF16_VECTOR_BYTES];
        bytes[0..2].copy_from_slice(&0x3f80_u16.to_le_bytes()); // 1.0f32 BF16
        let bits = decode_bf16_bits(&bytes, "test").expect("valid vector");
        assert_eq!(bits.len(), DSV4F_HIDDEN_SIZE);
        assert_eq!(DeepSeekV4RuntimeSpine::bf16_bits_to_f32(&bits)[0], 1.0);
    }

    #[test]
    fn mtp_is_not_a_base_body_staging_name() {
        assert!(is_base_body_tensor_name("layers.42.attn.wq_a.weight"));
        assert!(!is_base_body_tensor_name("mtp.0.attn.wq_a.weight"));
        assert!(!is_base_body_tensor_name("layers.43.attn.wq_a.weight"));
    }

    #[test]
    fn capability_gate_refuses_unregistered_engine() {
        let gate = DeepSeekV4RuntimeCapabilityGate {
            full_stream_reader_admitted: true,
            source_hashes_and_configs_validated: true,
            all_base_operator_bindings_validated: true,
            bounded_tensor_staging_available: true,
            embedding_final_norm_head_primitives_available: true,
            registered_43_layer_engine: false,
            causal_forward_available: false,
            continuation_available: false,
            metal_dispatches_available: false,
            hcli_endpoint_available: false,
            numeric_parity_v21_passed: false,
            base_true_tps_eligible: false,
            next_parity_rung: "P3_MHC_NORM_Q_PATH",
        };
        assert!(gate.require_full_causal_runtime().is_err());
    }
}
