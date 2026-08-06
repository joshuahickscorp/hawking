//! Bounded verifier for the qualified DeepSeek-V4 P0 Torch F32 Gate target.
//!
//! This module accepts only the small, unsealed calibration shard and its
//! immutable P0 Gate-input trace. It binds both to an already admitted full
//! Gravity stream, exposes the finite F32[256] target to an explicit CPU
//! diagnostic caller, and has no Metal, runtime, routing, or TPS surface.
//!
//! The target is deliberately not a default replacement for the serial CPU
//! Gate oracle. A caller must opt in, then prove after graph completion that
//! the observed BF16 Gate input has the trace-bound SHA-256 before it may use
//! these logits to construct a source route.

use std::fs;
use std::path::{Path, PathBuf};

use half::bf16;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use crate::gravity_deepseek_v4_layer0_moe::{
    layer0_moe_body_f32_oracle_from_verified_gate_route_for_token, Layer0MoeBodyF32OracleResult,
    ACTIVATED_EXPERTS, LAYER0_FFN_GATE_TID2EID, LAYER0_FFN_GATE_WEIGHT, ROUTED_EXPERTS,
    ROUTE_SCALE,
};
use crate::gravity_deepseek_v4_layer0_prefix::{HIDDEN_SIZE, PREFIX_TOKEN_ID};
use crate::{Error, Result};

pub const P0_GATE_TORCH_F32_CALIBRATION_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.p0_gate_torch_f32_calibration_shard.v1";
pub const P0_GATE_TORCH_F32_CALIBRATION_STATUS: &str =
    "UNSEALED_QUALIFIED_SOURCE_CPU_TORCH_F32_GATE_TARGET_NON_RECEIPT";
pub const P0_GATE_TORCH_F32_ROUTE_CALIBRATION_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.p0_gate_torch_f32_route_calibration_shard.v2";
pub const P0_GATE_TORCH_F32_ROUTE_CALIBRATION_STATUS: &str =
    "UNSEALED_QUALIFIED_SOURCE_CPU_TORCH_F32_GATE_ROUTE_TARGET_NON_RECEIPT";
pub const P0_GATE_INPUT_TRACE_SCHEMA: &str =
    "hawking.gravity.deepseek_v4.p7_layer0_position0_gate_input_trace.v1";
pub const P0_GATE_INPUT_TRACE_STATUS: &str =
    "UNSEALED_POST_COMPLETION_BOS_FFN_NORM_GATE_INPUT_TRACE_NON_RECEIPT";

const CALIBRATION_MAX_BYTES: usize = 32 * 1024;
const TRACE_MAX_BYTES: usize = 64 * 1024;
const GATE_INPUT_BYTES: usize = HIDDEN_SIZE * std::mem::size_of::<u16>();
const GATE_LOGIT_BYTES: usize = ROUTED_EXPERTS * std::mem::size_of::<f32>();
const GATE_WEIGHT_BYTES: usize = ROUTED_EXPERTS * GATE_INPUT_BYTES;
const TID2EID_BYTES: usize = 129_280 * ACTIVATED_EXPERTS * std::mem::size_of::<i64>();
const ROUTE_WEIGHT_BYTES: usize = ACTIVATED_EXPERTS * std::mem::size_of::<f32>();
const ROUTE_ID_BYTES: usize = ACTIVATED_EXPERTS * std::mem::size_of::<u16>();
const ROUTE_TARGET_RAW_BYTES: usize = GATE_LOGIT_BYTES * 2 + ROUTE_WEIGHT_BYTES + ROUTE_ID_BYTES;

/// Immutable source/trace/artifact facts proved by
/// [`load_verified_p0_gate_torch_f32_calibration`]. The values are safe to
/// retain in a diagnostic receipt because they contain paths, identifiers, and
/// hashes only; the raw Torch target remains in the owning calibration object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P0GateCalibrationBindings {
    pub calibration_path: PathBuf,
    pub calibration_file_sha256: String,
    pub trace_path: PathBuf,
    pub trace_file_sha256: String,
    pub trace_gate_input_bf16_le_sha256: String,
    pub trace_baseline_gate_logits_f32_le_sha256: String,
    pub torch_logits_f32_le_sha256: String,
    pub artifact_manifest_seal_sha256: String,
    pub artifact_manifest_file_sha256: String,
    pub artifact_restart_receipt_seal_sha256: String,
    pub source_repository: String,
    pub source_revision: String,
    pub source_model_py_sha256: String,
    pub gate_weight_name: String,
    pub gate_weight_sha256: String,
    pub tid2eid_name: String,
    pub tid2eid_sha256: String,
    pub layer: u64,
    pub token_id: u64,
    pub token_position: u64,
}

/// A qualified, bounded source target. The logits are intentionally exposed
/// only as a slice; callers must still invoke
/// [`Self::validate_observed_gate_input_bf16`] after graph completion before
/// using them in a calibrated source-reference path.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4P0GateTorchF32Calibration {
    bindings: DeepSeekV4P0GateCalibrationBindings,
    logits_f32: Vec<f32>,
}

/// Immutable source/trace/artifact facts proved by the qualified v2
/// Torch-Gate-route loader. This is safe receipt metadata only: all raw target
/// vectors remain private to the calibration object.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4P0GateRouteCalibrationBindings {
    pub calibration_path: PathBuf,
    pub calibration_file_sha256: String,
    pub calibration_canonical_sha256: String,
    pub trace_path: PathBuf,
    pub trace_file_sha256: String,
    pub trace_gate_input_bf16_le_sha256: String,
    pub trace_baseline_gate_logits_f32_le_sha256: String,
    pub torch_logits_f32_le_sha256: String,
    pub original_scores_f32_le_sha256: String,
    pub selected_weights_f32_le_sha256: String,
    pub selected_expert_ids_u16_le_sha256: String,
    pub artifact_manifest_seal_sha256: String,
    pub artifact_manifest_file_sha256: String,
    pub artifact_restart_receipt_seal_sha256: String,
    pub source_repository: String,
    pub source_revision: String,
    pub source_model_py_sha256: String,
    pub gate_weight_name: String,
    pub gate_weight_sha256: String,
    pub tid2eid_name: String,
    pub tid2eid_sha256: String,
    pub route_scale: f32,
    pub layer: u64,
    pub token_id: u64,
    pub token_position: u64,
}

/// Bounded direct source-CPU Torch Gate-route target. It retains only the
/// four values needed to compare the post-linear route path; no raw input or
/// source weights are exposed by this API.
#[derive(Debug, Clone, PartialEq)]
pub struct DeepSeekV4P0GateTorchF32RouteCalibration {
    bindings: DeepSeekV4P0GateRouteCalibrationBindings,
    logits_f32: Vec<f32>,
    original_scores_f32: Vec<f32>,
    selected_weights_f32: Vec<f32>,
    selected_expert_ids: Vec<u16>,
}

#[derive(Debug)]
struct BoundP0GateTrace {
    path: PathBuf,
    file_sha256: String,
    input_bf16_le_sha256: String,
    baseline_gate_logits_f32_le_sha256: String,
}

impl DeepSeekV4P0GateTorchF32Calibration {
    pub fn bindings(&self) -> &DeepSeekV4P0GateCalibrationBindings {
        &self.bindings
    }

    /// The verified little-endian F32[256] Torch target. This method does not
    /// assert it belongs to a later observed graph input; use
    /// [`Self::validate_observed_gate_input_bf16`] for that fail-closed gate.
    pub fn logits_f32(&self) -> &[f32] {
        &self.logits_f32
    }

    /// Refuse use of the calibration unless the completed graph's exact
    /// BF16[4096] Gate input equals the immutable trace payload binding.
    pub fn validate_observed_gate_input_bf16(&self, observed: &[u16]) -> Result<()> {
        if observed.len() != HIDDEN_SIZE {
            return Err(calibration_error(
                "observed P0 Gate input is not one BF16[4096] row",
            ));
        }
        if observed
            .iter()
            .any(|bits| !bf16::from_bits(*bits).to_f32().is_finite())
        {
            return Err(calibration_error(
                "observed P0 Gate input contains a non-finite BF16 value",
            ));
        }
        let bytes = observed
            .iter()
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<_>>();
        let observed_sha256 = sha256_hex(&bytes);
        if observed_sha256 != self.bindings.trace_gate_input_bf16_le_sha256 {
            return Err(calibration_error(format!(
                "completed P0 BF16 Gate-input SHA-256 {observed_sha256} does not match the qualified calibration trace {}",
                self.bindings.trace_gate_input_bf16_le_sha256,
            )));
        }
        Ok(())
    }
}

impl DeepSeekV4P0GateTorchF32RouteCalibration {
    pub fn bindings(&self) -> &DeepSeekV4P0GateRouteCalibrationBindings {
        &self.bindings
    }

    /// The verified direct Torch F32 Gate logits. These remain usable only
    /// after [`Self::validate_observed_gate_input_bf16`] succeeds.
    pub fn logits_f32(&self) -> &[f32] {
        &self.logits_f32
    }

    pub fn original_scores_f32(&self) -> &[f32] {
        &self.original_scores_f32
    }

    pub fn selected_weights_f32(&self) -> &[f32] {
        &self.selected_weights_f32
    }

    pub fn selected_expert_ids(&self) -> &[u16] {
        &self.selected_expert_ids
    }

    /// Refuse route-target use unless the completed graph's BF16 Gate input
    /// exactly matches the immutable trace bound at production time.
    pub fn validate_observed_gate_input_bf16(&self, observed: &[u16]) -> Result<()> {
        validate_observed_gate_input_bf16_against_hash(
            observed,
            &self.bindings.trace_gate_input_bf16_le_sha256,
            "qualified v2 route calibration",
        )
    }
}

/// Execute the CPU F32 MoE diagnostic body using the only admitted v2 direct
/// source-CPU Torch Gate-route target. The target cannot influence a runtime
/// graph: this wrapper first fails closed on the completed BF16 input hash,
/// then routes through the artifact-bound `tid2eid` row on CPU only.
pub fn layer0_moe_body_f32_oracle_from_qualified_torch_route_calibration_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
    calibration: &DeepSeekV4P0GateTorchF32RouteCalibration,
) -> Result<Layer0MoeBodyF32OracleResult> {
    if token_id != calibration.bindings.token_id {
        return Err(calibration_error(
            "qualified v2 route calibration token ID differs from the requested CPU diagnostic token",
        ));
    }
    calibration.validate_observed_gate_input_bf16(ffn_norm_bf16_bits)?;
    layer0_moe_body_f32_oracle_from_verified_gate_route_for_token(
        reader,
        token_id,
        ffn_norm_bf16_bits,
        &calibration.logits_f32,
        &calibration.original_scores_f32,
        &calibration.selected_expert_ids,
        &calibration.selected_weights_f32,
    )
}

fn validate_observed_gate_input_bf16_against_hash(
    observed: &[u16],
    expected_sha256: &str,
    label: &str,
) -> Result<()> {
    if observed.len() != HIDDEN_SIZE {
        return Err(calibration_error(format!(
            "{label} observed P0 Gate input is not one BF16[4096] row",
        )));
    }
    if observed
        .iter()
        .any(|bits| !bf16::from_bits(*bits).to_f32().is_finite())
    {
        return Err(calibration_error(format!(
            "{label} observed P0 Gate input contains a non-finite BF16 value",
        )));
    }
    let bytes = observed
        .iter()
        .flat_map(|bits| bits.to_le_bytes())
        .collect::<Vec<_>>();
    let observed_sha256 = sha256_hex(&bytes);
    if observed_sha256 != expected_sha256 {
        return Err(calibration_error(format!(
            "{label} completed P0 BF16 Gate-input SHA-256 {observed_sha256} does not match {expected_sha256}",
        )));
    }
    Ok(())
}

/// Load the only admitted qualified P0 source-Gate calibration format.
///
/// The loader validates its bounded F32[256] payload, the referenced immutable
/// trace, the admitted artifact's manifest/restart/model identity, and the
/// full verified BF16[256,4096] Gate tensor hash. It deliberately does not
/// score V2.1, create a Metal resource, or alter any default oracle route.
pub fn load_verified_p0_gate_torch_f32_calibration(
    reader: &DeepSeekV4FullStreamReader,
    calibration_path: impl AsRef<Path>,
) -> Result<DeepSeekV4P0GateTorchF32Calibration> {
    let calibration_path = calibration_path.as_ref();
    let calibration_raw = read_absolute_regular_bounded(
        calibration_path,
        "P0 Torch F32 Gate calibration shard",
        CALIBRATION_MAX_BYTES,
    )?;
    let calibration_file_sha256 = sha256_hex(&calibration_raw);
    let calibration = parse_json(&calibration_raw, "P0 Torch F32 Gate calibration shard")?;

    expect_string(
        &calibration,
        &["schema"],
        "calibration schema",
        P0_GATE_TORCH_F32_CALIBRATION_SCHEMA,
    )?;
    expect_string(
        &calibration,
        &["status"],
        "calibration status",
        P0_GATE_TORCH_F32_CALIBRATION_STATUS,
    )?;
    expect_bool(
        &calibration,
        &["unsealed"],
        "calibration unsealed flag",
        true,
    )?;
    expect_bool(
        &calibration,
        &["receipt_promoted"],
        "calibration receipt-promoted flag",
        false,
    )?;
    expect_bool(
        &calibration,
        &["is_receipt"],
        "calibration receipt flag",
        false,
    )?;
    if count_object_key(&calibration, "data") != 1 {
        return Err(calibration_error(
            "calibration must retain exactly one raw data field: F32[256] Torch Gate logits",
        ));
    }
    validate_calibration_storage_bound(&calibration)?;
    validate_artifact_binding(
        &calibration,
        &["artifact_binding"],
        reader,
        "calibration artifact",
    )?;
    validate_calibration_source_binding(&calibration, reader)?;
    validate_source_cpu_torch_binding(&calibration)?;

    expect_string(
        &calibration,
        &["trace_binding", "schema"],
        "calibration trace schema",
        P0_GATE_INPUT_TRACE_SCHEMA,
    )?;
    expect_string(
        &calibration,
        &["trace_binding", "status"],
        "calibration trace status",
        P0_GATE_INPUT_TRACE_STATUS,
    )?;
    expect_bool(
        &calibration,
        &["trace_binding", "immutable_existing_trace"],
        "calibration immutable trace flag",
        true,
    )?;
    expect_bool(
        &calibration,
        &["trace_binding", "raw_p0_input_copied_into_this_shard"],
        "calibration copied-input flag",
        false,
    )?;
    expect_u64(
        &calibration,
        &["trace_binding", "layer"],
        "calibration trace layer",
        0,
    )?;
    expect_u64(
        &calibration,
        &["trace_binding", "token_id"],
        "calibration trace token ID",
        PREFIX_TOKEN_ID,
    )?;
    expect_u64(
        &calibration,
        &["trace_binding", "token_position"],
        "calibration trace token position",
        0,
    )?;
    expect_string(
        &calibration,
        &["trace_binding", "p0_gate_input_geometry", "dtype"],
        "calibration Gate-input dtype",
        "BF16",
    )?;
    expect_shape(
        &calibration,
        &["trace_binding", "p0_gate_input_geometry", "shape"],
        &[HIDDEN_SIZE as u64],
        "calibration Gate-input shape",
    )?;
    expect_u64(
        &calibration,
        &["trace_binding", "p0_gate_input_geometry", "bytes"],
        "calibration Gate-input bytes",
        GATE_INPUT_BYTES as u64,
    )?;

    let trace_path = PathBuf::from(required_string(
        &calibration,
        &["trace_binding", "path"],
        "calibration trace path",
    )?);
    let calibration_trace_file_sha256 = required_string(
        &calibration,
        &["trace_binding", "file_sha256"],
        "calibration trace file SHA-256",
    )?;
    let calibration_trace_input_sha256 = required_string(
        &calibration,
        &["trace_binding", "raw_gate_input_payload_sha256"],
        "calibration trace Gate-input SHA-256",
    )?;
    expect_string(
        &calibration,
        &["trace_binding", "p0_gate_input_sha256"],
        "calibration duplicate Gate-input SHA-256",
        &calibration_trace_input_sha256,
    )?;

    let trace_raw = read_absolute_regular_bounded(
        &trace_path,
        "immutable P0 Gate-input trace",
        TRACE_MAX_BYTES,
    )?;
    let trace_file_sha256 = sha256_hex(&trace_raw);
    if trace_file_sha256 != calibration_trace_file_sha256 {
        return Err(calibration_error(
            "immutable P0 Gate-input trace file SHA-256 differs from calibration binding",
        ));
    }
    let trace = parse_json(&trace_raw, "immutable P0 Gate-input trace")?;
    validate_trace_binding(&trace, reader)?;

    let trace_raw_payload = required_object(&trace, &["raw_payload"], "trace raw payload")?;
    expect_string(trace_raw_payload, &["dtype"], "trace raw dtype", "BF16")?;
    expect_string(
        trace_raw_payload,
        &["byte_order"],
        "trace raw byte order",
        "little_endian",
    )?;
    expect_string(
        trace_raw_payload,
        &["encoding"],
        "trace raw encoding",
        "lowercase_hex_raw_bf16_le",
    )?;
    expect_shape(
        trace_raw_payload,
        &["shape"],
        &[HIDDEN_SIZE as u64],
        "trace raw shape",
    )?;
    expect_u64(
        trace_raw_payload,
        &["element_count"],
        "trace raw element count",
        HIDDEN_SIZE as u64,
    )?;
    expect_u64(
        trace_raw_payload,
        &["byte_count"],
        "trace raw byte count",
        GATE_INPUT_BYTES as u64,
    )?;
    let trace_payload_sha256 = required_string(
        trace_raw_payload,
        &["sha256"],
        "trace raw Gate-input SHA-256",
    )?;
    let trace_payload = decode_lowercase_hex(
        required_string(trace_raw_payload, &["data"], "trace raw Gate-input data")?,
        GATE_INPUT_BYTES,
        "trace raw Gate-input",
    )?;
    if sha256_hex(&trace_payload) != trace_payload_sha256
        || trace_payload_sha256 != calibration_trace_input_sha256
    {
        return Err(calibration_error(
            "trace raw BF16 Gate-input payload/hash differs from calibration binding",
        ));
    }
    expect_string(
        &trace,
        &["input_output_sha256", "p7_producer_output_ffn_norm_bf16_le"],
        "trace P7 FFn-norm SHA-256",
        &trace_payload_sha256,
    )?;
    expect_string(
        &trace,
        &["input_output_sha256", "p6_gate_input_ffn_norm_bf16_le"],
        "trace P6 Gate-input SHA-256",
        &trace_payload_sha256,
    )?;
    let trace_baseline_gate_logits_f32_le_sha256 = required_string(
        &trace,
        &["input_output_sha256", "p6_gate_output_logits_f32_le"],
        "trace baseline Gate-logit SHA-256",
    )?;
    expect_string(
        &calibration,
        &["trace_binding", "recorded_metal_gate_logits_f32_le_sha256"],
        "calibration trace baseline Gate-logit SHA-256",
        &trace_baseline_gate_logits_f32_le_sha256,
    )?;

    let (gate_weight_name, gate_weight_sha256, tid2eid_name, tid2eid_sha256) =
        validate_gate_tensor_bindings(&calibration, &trace, reader)?;
    let raw_target = required_object(&calibration, &["raw_f32_le"], "calibration raw F32 target")?;
    expect_string(
        raw_target,
        &["name"],
        "calibration raw target name",
        "p0_gate_logits_torch_f32_le",
    )?;
    expect_string(raw_target, &["dtype"], "calibration target dtype", "F32")?;
    expect_string(
        raw_target,
        &["byte_order"],
        "calibration target byte order",
        "little_endian",
    )?;
    expect_string(
        raw_target,
        &["encoding"],
        "calibration target encoding",
        "lowercase_hex_raw_f32_le",
    )?;
    expect_shape(
        raw_target,
        &["shape"],
        &[ROUTED_EXPERTS as u64],
        "calibration target shape",
    )?;
    expect_u64(
        raw_target,
        &["element_count"],
        "calibration target element count",
        ROUTED_EXPERTS as u64,
    )?;
    expect_u64(
        raw_target,
        &["byte_count"],
        "calibration target byte count",
        GATE_LOGIT_BYTES as u64,
    )?;
    let torch_logits_f32_le_sha256 =
        required_string(raw_target, &["sha256"], "calibration target SHA-256")?;
    let target_bytes = decode_lowercase_hex(
        required_string(raw_target, &["data"], "calibration target data")?,
        GATE_LOGIT_BYTES,
        "calibration F32 target",
    )?;
    if sha256_hex(&target_bytes) != torch_logits_f32_le_sha256 {
        return Err(calibration_error(
            "calibration F32 target payload SHA-256 mismatch",
        ));
    }
    let logits_f32 = decode_f32_le(&target_bytes)?;
    if logits_f32.len() != ROUTED_EXPERTS || logits_f32.iter().any(|value| !value.is_finite()) {
        return Err(calibration_error(
            "calibration target is not finite F32[256]",
        ));
    }

    Ok(DeepSeekV4P0GateTorchF32Calibration {
        bindings: DeepSeekV4P0GateCalibrationBindings {
            calibration_path: calibration_path.to_owned(),
            calibration_file_sha256,
            trace_path,
            trace_file_sha256,
            trace_gate_input_bf16_le_sha256: trace_payload_sha256.to_owned(),
            trace_baseline_gate_logits_f32_le_sha256: trace_baseline_gate_logits_f32_le_sha256
                .to_owned(),
            torch_logits_f32_le_sha256: torch_logits_f32_le_sha256.to_owned(),
            artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            artifact_manifest_file_sha256: reader.manifest_file_sha256().to_owned(),
            artifact_restart_receipt_seal_sha256: reader.restart_seal_sha256().to_owned(),
            source_repository: reader.source_identity().repository.clone(),
            source_revision: reader.source_identity().revision.clone(),
            source_model_py_sha256: reader
                .source_metadata_asset_sha256("inference/model.py")?
                .to_owned(),
            gate_weight_name,
            gate_weight_sha256,
            tid2eid_name,
            tid2eid_sha256,
            layer: 0,
            token_id: PREFIX_TOKEN_ID,
            token_position: 0,
        },
        logits_f32,
    })
}

/// Load the opt-in qualified v2 direct source-CPU Torch Gate-route target.
///
/// Unlike the v1 logit-only shard, this validates all bounded post-linear
/// route targets produced by `Gate.forward`: F32 logits and original scores,
/// the fixed `tid2eid` IDs, and normalized F32 route weights. It still has no
/// Metal/runtime surface and cannot itself establish a V2.1, endpoint, token,
/// or TPS claim.
pub fn load_verified_p0_gate_torch_f32_route_calibration(
    reader: &DeepSeekV4FullStreamReader,
    calibration_path: impl AsRef<Path>,
) -> Result<DeepSeekV4P0GateTorchF32RouteCalibration> {
    let calibration_path = calibration_path.as_ref();
    let calibration_raw = read_absolute_regular_bounded(
        calibration_path,
        "P0 Torch F32 Gate-route calibration shard",
        CALIBRATION_MAX_BYTES,
    )?;
    let calibration_file_sha256 = sha256_hex(&calibration_raw);
    let calibration = parse_json(
        &calibration_raw,
        "P0 Torch F32 Gate-route calibration shard",
    )?;
    expect_string(
        &calibration,
        &["schema"],
        "v2 route calibration schema",
        P0_GATE_TORCH_F32_ROUTE_CALIBRATION_SCHEMA,
    )?;
    expect_string(
        &calibration,
        &["status"],
        "v2 route calibration status",
        P0_GATE_TORCH_F32_ROUTE_CALIBRATION_STATUS,
    )?;
    expect_bool(
        &calibration,
        &["unsealed"],
        "v2 route calibration unsealed flag",
        true,
    )?;
    expect_bool(
        &calibration,
        &["is_receipt"],
        "v2 route calibration receipt flag",
        false,
    )?;
    expect_bool(
        &calibration,
        &["receipt_promoted"],
        "v2 route calibration receipt-promoted flag",
        false,
    )?;
    let calibration_canonical_sha256 = required_string(
        &calibration,
        &["canonical_sha256"],
        "v2 route calibration canonical SHA-256",
    )?
    .to_owned();
    if count_object_key(&calibration, "data") != 4 {
        return Err(calibration_error(
            "v2 route calibration must retain exactly four raw target data fields",
        ));
    }
    validate_route_calibration_storage_bound(&calibration)?;
    validate_artifact_binding(
        &calibration,
        &["artifact_binding"],
        reader,
        "v2 route calibration artifact",
    )?;
    validate_calibration_source_binding(&calibration, reader)?;
    validate_route_source_cpu_torch_binding(&calibration)?;
    let trace = load_bound_v2_route_trace(&calibration, reader)?;
    let (gate_weight_name, gate_weight_sha256, tid2eid_name, tid2eid_sha256) =
        validate_v2_route_tensor_bindings(&calibration, reader)?;
    validate_v2_route_target_contract(&calibration)?;

    let raw_targets = required_object(
        &calibration,
        &["raw_targets"],
        "v2 route calibration raw targets",
    )?;
    let (torch_logits_f32_le_sha256, logits_f32) = load_f32_route_target(
        raw_targets,
        "torch_logits_f32_le",
        "p0_gate_logits_torch_f32_le",
        ROUTED_EXPERTS,
        "v2 route calibration Torch logits",
    )?;
    let (original_scores_f32_le_sha256, original_scores_f32) = load_f32_route_target(
        raw_targets,
        "original_scores_f32_le",
        "p0_gate_original_scores_torch_f32_le",
        ROUTED_EXPERTS,
        "v2 route calibration original scores",
    )?;
    if original_scores_f32
        .iter()
        .any(|value| !(value.is_finite() && *value > 0.0))
    {
        return Err(calibration_error(
            "v2 route calibration original scores must be positive finite F32[256]",
        ));
    }
    let (selected_weights_f32_le_sha256, selected_weights_f32) = load_f32_route_target(
        raw_targets,
        "selected_weights_f32_le",
        "p0_gate_selected_weights_torch_f32_le",
        ACTIVATED_EXPERTS,
        "v2 route calibration selected weights",
    )?;
    if selected_weights_f32
        .iter()
        .any(|value| !(value.is_finite() && *value > 0.0))
    {
        return Err(calibration_error(
            "v2 route calibration selected weights must be positive finite F32[6]",
        ));
    }
    let (selected_expert_ids_u16_le_sha256, selected_expert_ids) = load_u16_route_target(
        raw_targets,
        "selected_expert_ids_u16_le",
        "p0_gate_selected_expert_ids_tid2eid_u16_le",
        ACTIVATED_EXPERTS,
        "v2 route calibration selected expert IDs",
    )?;
    if selected_expert_ids
        .iter()
        .any(|&expert| usize::from(expert) >= ROUTED_EXPERTS)
    {
        return Err(calibration_error(
            "v2 route calibration selected expert IDs are outside F32[256] scores",
        ));
    }
    let bound_ids = read_verified_tid2eid_row(reader, PREFIX_TOKEN_ID)?;
    if bound_ids != selected_expert_ids {
        return Err(calibration_error(
            "v2 route calibration selected expert IDs differ from the artifact tid2eid P0 row",
        ));
    }
    let weights_sum = selected_weights_f32.iter().copied().sum::<f32>();
    if !weights_sum.is_finite() || (weights_sum - ROUTE_SCALE).abs() > 1.0e-5 {
        return Err(calibration_error(
            "v2 route calibration selected weights do not normalize to route_scale=1.5",
        ));
    }
    for (index, (&expert, &weight)) in selected_expert_ids
        .iter()
        .zip(&selected_weights_f32)
        .enumerate()
    {
        let score = original_scores_f32[usize::from(expert)];
        if !(score.is_finite() && score > 0.0 && weight.is_finite() && weight > 0.0) {
            return Err(calibration_error(format!(
                "v2 route calibration selected score/weight {index} is invalid",
            )));
        }
    }

    Ok(DeepSeekV4P0GateTorchF32RouteCalibration {
        bindings: DeepSeekV4P0GateRouteCalibrationBindings {
            calibration_path: calibration_path.to_owned(),
            calibration_file_sha256,
            calibration_canonical_sha256,
            trace_path: trace.path,
            trace_file_sha256: trace.file_sha256,
            trace_gate_input_bf16_le_sha256: trace.input_bf16_le_sha256,
            trace_baseline_gate_logits_f32_le_sha256: trace.baseline_gate_logits_f32_le_sha256,
            torch_logits_f32_le_sha256,
            original_scores_f32_le_sha256,
            selected_weights_f32_le_sha256,
            selected_expert_ids_u16_le_sha256,
            artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            artifact_manifest_file_sha256: reader.manifest_file_sha256().to_owned(),
            artifact_restart_receipt_seal_sha256: reader.restart_seal_sha256().to_owned(),
            source_repository: reader.source_identity().repository.clone(),
            source_revision: reader.source_identity().revision.clone(),
            source_model_py_sha256: reader
                .source_metadata_asset_sha256("inference/model.py")?
                .to_owned(),
            gate_weight_name,
            gate_weight_sha256,
            tid2eid_name,
            tid2eid_sha256,
            route_scale: ROUTE_SCALE,
            layer: 0,
            token_id: PREFIX_TOKEN_ID,
            token_position: 0,
        },
        logits_f32,
        original_scores_f32,
        selected_weights_f32,
        selected_expert_ids,
    })
}

fn validate_calibration_storage_bound(calibration: &Value) -> Result<()> {
    expect_u64(
        calibration,
        &["storage_policy", "raw_payload_count"],
        "calibration raw payload count",
        1,
    )?;
    expect_u64(
        calibration,
        &["storage_policy", "raw_payload_actual_bytes"],
        "calibration raw payload bytes",
        GATE_LOGIT_BYTES as u64,
    )?;
    expect_u64(
        calibration,
        &["storage_policy", "raw_payload_hard_max_bytes"],
        "calibration raw payload hard maximum",
        GATE_LOGIT_BYTES as u64,
    )?;
    for field in [
        "raw_fp64_authority_payloads",
        "raw_input_payloads",
        "raw_route_payloads",
        "raw_route_weight_payloads",
        "raw_source_weight_payloads",
    ] {
        expect_u64(
            calibration,
            &["storage_policy", field],
            "calibration forbidden raw-payload count",
            0,
        )?;
    }
    Ok(())
}

fn validate_calibration_source_binding(
    calibration: &Value,
    reader: &DeepSeekV4FullStreamReader,
) -> Result<()> {
    expect_string(
        calibration,
        &["source_binding", "repository"],
        "calibration source repository",
        &reader.source_identity().repository,
    )?;
    expect_string(
        calibration,
        &["source_binding", "revision"],
        "calibration source revision",
        &reader.source_identity().revision,
    )?;
    expect_string(
        calibration,
        &["source_binding", "model_py_sha256"],
        "calibration source model.py SHA-256",
        reader.source_metadata_asset_sha256("inference/model.py")?,
    )?;
    let model_py_bytes = reader.read_verified_metadata_asset("inference/model.py", 128 * 1024)?;
    expect_u64(
        calibration,
        &["source_binding", "model_py_bytes"],
        "calibration source model.py bytes",
        model_py_bytes.len() as u64,
    )?;
    Ok(())
}

fn validate_source_cpu_torch_binding(calibration: &Value) -> Result<()> {
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "execution_device"],
        "calibration Torch execution device",
        "cpu",
    )?;
    expect_bool(
        calibration,
        &["source_cpu_torch_binding", "torch_cuda_invoked"],
        "calibration Torch CUDA-invoked flag",
        false,
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "input_dtype"],
        "calibration Torch input dtype",
        "BF16 decoded to F32",
    )?;
    expect_shape(
        calibration,
        &["source_cpu_torch_binding", "input_shape"],
        &[HIDDEN_SIZE as u64],
        "calibration Torch input shape",
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "weight_dtype"],
        "calibration Torch weight dtype",
        "BF16 decoded to F32",
    )?;
    expect_shape(
        calibration,
        &["source_cpu_torch_binding", "weight_shape"],
        &[ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64],
        "calibration Torch weight shape",
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "output_dtype"],
        "calibration Torch output dtype",
        "F32",
    )?;
    expect_shape(
        calibration,
        &["source_cpu_torch_binding", "output_shape"],
        &[ROUTED_EXPERTS as u64],
        "calibration Torch output shape",
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "torch_f_linear_contract"],
        "calibration Torch F.linear contract",
        "torch.nn.functional.linear(x.float(), weight.float()) on CPU with the bound 1D P0 input",
    )?;
    Ok(())
}

fn validate_route_calibration_storage_bound(calibration: &Value) -> Result<()> {
    expect_u64(
        calibration,
        &["storage_policy", "raw_payload_count"],
        "v2 route calibration raw payload count",
        4,
    )?;
    expect_u64(
        calibration,
        &["storage_policy", "raw_payload_actual_bytes"],
        "v2 route calibration raw payload bytes",
        ROUTE_TARGET_RAW_BYTES as u64,
    )?;
    expect_u64(
        calibration,
        &["storage_policy", "raw_payload_hard_max_bytes"],
        "v2 route calibration raw payload hard maximum",
        ROUTE_TARGET_RAW_BYTES as u64,
    )?;
    expect_u64(
        calibration,
        &["storage_policy", "serialized_shard_hard_max_bytes"],
        "v2 route calibration serialized hard maximum",
        CALIBRATION_MAX_BYTES as u64,
    )?;
    for field in [
        "raw_source_weight_payloads",
        "raw_input_payloads",
        "raw_other_activation_payloads",
        "raw_fp64_authority_payloads",
    ] {
        expect_u64(
            calibration,
            &["storage_policy", field],
            "v2 route calibration forbidden raw-payload count",
            0,
        )?;
    }
    expect_u64(
        calibration,
        &["storage_policy", "raw_route_payloads"],
        "v2 route calibration selected-ID payload count",
        1,
    )?;
    expect_u64(
        calibration,
        &["storage_policy", "raw_route_weight_payloads"],
        "v2 route calibration selected-weight payload count",
        1,
    )?;
    Ok(())
}

fn validate_route_source_cpu_torch_binding(calibration: &Value) -> Result<()> {
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "execution_device"],
        "v2 route calibration Torch execution device",
        "cpu",
    )?;
    expect_bool(
        calibration,
        &["source_cpu_torch_binding", "torch_cuda_invoked"],
        "v2 route calibration Torch CUDA-invoked flag",
        false,
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "input_dtype"],
        "v2 route calibration Torch input dtype",
        "BF16 decoded to F32",
    )?;
    expect_shape(
        calibration,
        &["source_cpu_torch_binding", "input_shape"],
        &[1, HIDDEN_SIZE as u64],
        "v2 route calibration Torch input shape",
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "weight_dtype"],
        "v2 route calibration Torch weight dtype",
        "BF16 decoded to F32",
    )?;
    expect_shape(
        calibration,
        &["source_cpu_torch_binding", "weight_shape"],
        &[ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64],
        "v2 route calibration Torch weight shape",
    )?;
    for (field, shape, label) in [
        ("logits", vec![1, ROUTED_EXPERTS as u64], "logits"),
        (
            "original_scores",
            vec![1, ROUTED_EXPERTS as u64],
            "original-scores",
        ),
        (
            "selected_weights",
            vec![1, ACTIVATED_EXPERTS as u64],
            "selected-weights",
        ),
    ] {
        expect_string(
            calibration,
            &["source_cpu_torch_binding", &format!("{field}_dtype")],
            "v2 route calibration Torch route value dtype",
            "F32",
        )?;
        expect_shape(
            calibration,
            &["source_cpu_torch_binding", &format!("{field}_shape")],
            &shape,
            &format!("v2 route calibration Torch {label} shape"),
        )?;
    }
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "selected_ids_dtype"],
        "v2 route calibration Torch selected-ID dtype",
        "I64",
    )?;
    expect_shape(
        calibration,
        &["source_cpu_torch_binding", "selected_ids_shape"],
        &[1, ACTIVATED_EXPERTS as u64],
        "v2 route calibration Torch selected-ID shape",
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "torch_f_linear_contract"],
        "v2 route calibration Torch F.linear contract",
        "torch.nn.functional.linear(x.float(), weight.float()) on CPU with bound P0 input reshaped to [1, 4096]",
    )?;
    expect_string(
        calibration,
        &["source_cpu_torch_binding", "post_linear_route_contract"],
        "v2 route calibration post-linear route contract",
        "original_scores = torch.nn.functional.softplus(logits).sqrt(); indices = tid2eid[input_ids]; weights = original_scores.gather(1, indices); weights /= weights.sum(dim=-1, keepdim=True); weights *= route_scale",
    )?;
    expect_f64(
        calibration,
        &["source_cpu_torch_binding", "route_scale"],
        "v2 route calibration Torch route scale",
        f64::from(ROUTE_SCALE),
    )?;
    Ok(())
}

fn validate_trace_binding(trace: &Value, reader: &DeepSeekV4FullStreamReader) -> Result<()> {
    expect_string(
        trace,
        &["schema"],
        "trace schema",
        P0_GATE_INPUT_TRACE_SCHEMA,
    )?;
    expect_string(
        trace,
        &["status"],
        "trace status",
        P0_GATE_INPUT_TRACE_STATUS,
    )?;
    expect_bool(trace, &["unsealed"], "trace unsealed flag", true)?;
    expect_bool(
        trace,
        &["receipt_promoted"],
        "trace receipt-promoted flag",
        false,
    )?;
    expect_bool(trace, &["is_receipt"], "trace receipt flag", false)?;
    if count_object_key(trace, "data") != 1 {
        return Err(calibration_error(
            "immutable trace must retain exactly one raw data field",
        ));
    }
    expect_u64(trace, &["trace_binding", "layer"], "trace layer", 0)?;
    expect_u64(
        trace,
        &["trace_binding", "token_id"],
        "trace token ID",
        PREFIX_TOKEN_ID,
    )?;
    expect_u64(
        trace,
        &["trace_binding", "token_position"],
        "trace token position",
        0,
    )?;
    for field in [
        "post_completion_readback_only",
        "real_graph_completed_before_trace_emission",
        "trace_does_not_feed_graph",
        "trace_does_not_modify_graph_counters",
        "p4a_attention_predecessor_exact",
    ] {
        expect_bool(trace, &["trace_binding", field], "trace safety flag", true)?;
    }
    expect_string(
        trace,
        &["trace_binding", "p4a_attention_predecessor_label"],
        "trace P4A predecessor label",
        "P4A_EXACT_ATTENTION_ONLY",
    )?;
    validate_artifact_binding(trace, &["artifact"], reader, "trace artifact")?;
    expect_bool(
        trace,
        &["artifact", "source_parent_retained"],
        "trace source-parent-retained flag",
        false,
    )?;
    expect_string(
        trace,
        &["model_source", "repository"],
        "trace source repository",
        &reader.source_identity().repository,
    )?;
    expect_string(
        trace,
        &["model_source", "revision"],
        "trace source revision",
        &reader.source_identity().revision,
    )?;
    for asset in [
        "inference/model.py",
        "inference/kernel.py",
        "inference/convert.py",
        "inference/config.json",
        "config.json",
    ] {
        expect_string(
            trace,
            &["model_source", "metadata_asset_sha256", asset],
            "trace source metadata SHA-256",
            reader.source_metadata_asset_sha256(asset)?,
        )?;
    }
    expect_u64(
        trace,
        &["privacy_and_storage_bound", "raw_payload_count"],
        "trace raw payload count",
        1,
    )?;
    expect_u64(
        trace,
        &["privacy_and_storage_bound", "raw_payload_actual_bytes"],
        "trace raw payload bytes",
        GATE_INPUT_BYTES as u64,
    )?;
    for field in [
        "raw_source_weight_payloads",
        "raw_other_activation_payloads",
        "raw_gate_output_payloads",
        "raw_route_weight_payloads",
    ] {
        expect_u64(
            trace,
            &["privacy_and_storage_bound", field],
            "trace forbidden raw-payload count",
            0,
        )?;
    }
    Ok(())
}

fn load_bound_v2_route_trace(
    calibration: &Value,
    reader: &DeepSeekV4FullStreamReader,
) -> Result<BoundP0GateTrace> {
    expect_string(
        calibration,
        &["trace_binding", "schema"],
        "v2 route calibration trace schema",
        P0_GATE_INPUT_TRACE_SCHEMA,
    )?;
    expect_string(
        calibration,
        &["trace_binding", "status"],
        "v2 route calibration trace status",
        P0_GATE_INPUT_TRACE_STATUS,
    )?;
    expect_bool(
        calibration,
        &["trace_binding", "immutable_existing_trace"],
        "v2 route calibration immutable trace flag",
        true,
    )?;
    expect_bool(
        calibration,
        &["trace_binding", "raw_p0_input_copied_into_this_shard"],
        "v2 route calibration copied-input flag",
        false,
    )?;
    expect_u64(
        calibration,
        &["trace_binding", "layer"],
        "v2 route calibration trace layer",
        0,
    )?;
    expect_u64(
        calibration,
        &["trace_binding", "token_id"],
        "v2 route calibration trace token ID",
        PREFIX_TOKEN_ID,
    )?;
    expect_u64(
        calibration,
        &["trace_binding", "token_position"],
        "v2 route calibration trace token position",
        0,
    )?;
    expect_string(
        calibration,
        &["trace_binding", "p0_gate_input_geometry", "dtype"],
        "v2 route calibration Gate-input dtype",
        "BF16",
    )?;
    expect_shape(
        calibration,
        &["trace_binding", "p0_gate_input_geometry", "shape"],
        &[HIDDEN_SIZE as u64],
        "v2 route calibration Gate-input shape",
    )?;
    expect_u64(
        calibration,
        &["trace_binding", "p0_gate_input_geometry", "bytes"],
        "v2 route calibration Gate-input bytes",
        GATE_INPUT_BYTES as u64,
    )?;
    let trace_path = PathBuf::from(required_string(
        calibration,
        &["trace_binding", "path"],
        "v2 route calibration trace path",
    )?);
    let expected_file_sha256 = required_string(
        calibration,
        &["trace_binding", "file_sha256"],
        "v2 route calibration trace file SHA-256",
    )?;
    let expected_input_sha256 = required_string(
        calibration,
        &["trace_binding", "raw_gate_input_payload_sha256"],
        "v2 route calibration trace Gate-input SHA-256",
    )?;
    expect_string(
        calibration,
        &["trace_binding", "p0_gate_input_sha256"],
        "v2 route calibration duplicate Gate-input SHA-256",
        expected_input_sha256,
    )?;

    let trace_raw = read_absolute_regular_bounded(
        &trace_path,
        "immutable P0 Gate-input trace for v2 route calibration",
        TRACE_MAX_BYTES,
    )?;
    let trace_file_sha256 = sha256_hex(&trace_raw);
    if trace_file_sha256 != expected_file_sha256 {
        return Err(calibration_error(
            "v2 route calibration trace file SHA-256 differs from its binding",
        ));
    }
    let trace = parse_json(
        &trace_raw,
        "immutable P0 Gate-input trace for v2 route calibration",
    )?;
    validate_trace_binding(&trace, reader)?;
    let raw_payload = required_object(&trace, &["raw_payload"], "v2 route trace raw payload")?;
    expect_string(raw_payload, &["dtype"], "v2 route trace raw dtype", "BF16")?;
    expect_string(
        raw_payload,
        &["byte_order"],
        "v2 route trace raw byte order",
        "little_endian",
    )?;
    expect_string(
        raw_payload,
        &["encoding"],
        "v2 route trace raw encoding",
        "lowercase_hex_raw_bf16_le",
    )?;
    expect_shape(
        raw_payload,
        &["shape"],
        &[HIDDEN_SIZE as u64],
        "v2 route trace raw shape",
    )?;
    expect_u64(
        raw_payload,
        &["element_count"],
        "v2 route trace raw element count",
        HIDDEN_SIZE as u64,
    )?;
    expect_u64(
        raw_payload,
        &["byte_count"],
        "v2 route trace raw byte count",
        GATE_INPUT_BYTES as u64,
    )?;
    let input_bf16_le_sha256 = required_string(
        raw_payload,
        &["sha256"],
        "v2 route trace raw Gate-input SHA-256",
    )?
    .to_owned();
    let payload = decode_lowercase_hex(
        required_string(raw_payload, &["data"], "v2 route trace raw Gate-input data")?,
        GATE_INPUT_BYTES,
        "v2 route trace raw Gate-input",
    )?;
    if sha256_hex(&payload) != input_bf16_le_sha256 || input_bf16_le_sha256 != expected_input_sha256
    {
        return Err(calibration_error(
            "v2 route trace raw Gate-input payload/hash differs from calibration binding",
        ));
    }
    expect_string(
        &trace,
        &["input_output_sha256", "p7_producer_output_ffn_norm_bf16_le"],
        "v2 route trace P7 FFn-norm SHA-256",
        &input_bf16_le_sha256,
    )?;
    expect_string(
        &trace,
        &["input_output_sha256", "p6_gate_input_ffn_norm_bf16_le"],
        "v2 route trace P6 Gate-input SHA-256",
        &input_bf16_le_sha256,
    )?;
    let baseline_gate_logits_f32_le_sha256 = required_string(
        &trace,
        &["input_output_sha256", "p6_gate_output_logits_f32_le"],
        "v2 route trace baseline Gate-logit SHA-256",
    )?
    .to_owned();
    expect_string(
        calibration,
        &["trace_binding", "recorded_metal_gate_logits_f32_le_sha256"],
        "v2 route calibration trace baseline Gate-logit SHA-256",
        &baseline_gate_logits_f32_le_sha256,
    )?;
    Ok(BoundP0GateTrace {
        path: trace_path,
        file_sha256: trace_file_sha256,
        input_bf16_le_sha256,
        baseline_gate_logits_f32_le_sha256,
    })
}

fn validate_artifact_binding(
    value: &Value,
    path: &[&str],
    reader: &DeepSeekV4FullStreamReader,
    label: &str,
) -> Result<()> {
    let binding = required_object(value, path, label)?;
    expect_string(
        binding,
        &["manifest_seal_sha256"],
        "artifact manifest seal SHA-256",
        reader.manifest_seal_sha256(),
    )?;
    expect_string(
        binding,
        &["manifest_file_sha256"],
        "artifact manifest file SHA-256",
        reader.manifest_file_sha256(),
    )?;
    expect_string(
        binding,
        &["restart_receipt_seal_sha256"],
        "artifact restart receipt seal SHA-256",
        reader.restart_seal_sha256(),
    )?;
    Ok(())
}

fn validate_gate_tensor_bindings(
    calibration: &Value,
    trace: &Value,
    reader: &DeepSeekV4FullStreamReader,
) -> Result<(String, String, String, String)> {
    let gate = required_object(
        calibration,
        &["source_tensor_binding"],
        "calibration Gate tensor binding",
    )?;
    expect_string(
        gate,
        &["name"],
        "calibration Gate tensor name",
        LAYER0_FFN_GATE_WEIGHT,
    )?;
    expect_string(gate, &["dtype"], "calibration Gate tensor dtype", "BF16")?;
    expect_shape(
        gate,
        &["shape"],
        &[ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64],
        "calibration Gate tensor shape",
    )?;
    expect_u64(
        gate,
        &["bytes"],
        "calibration Gate tensor bytes",
        GATE_WEIGHT_BYTES as u64,
    )?;
    expect_u64(
        gate,
        &["verified_chunk_count"],
        "calibration Gate verified chunk count",
        1,
    )?;
    expect_u64(
        gate,
        &["verified_chunk_bytes"],
        "calibration Gate verified chunk bytes",
        GATE_WEIGHT_BYTES as u64,
    )?;
    let gate_weight_name =
        required_string(gate, &["name"], "calibration Gate tensor name")?.to_owned();
    let gate_weight_sha256 = required_string(
        gate,
        &["logical_tensor_sha256"],
        "calibration Gate tensor SHA-256",
    )?
    .to_owned();
    expect_single_sha256(
        gate,
        &["verified_chunk_sha256"],
        "calibration Gate verified chunk SHA-256",
        &gate_weight_sha256,
    )?;
    let duplicate_gate = required_object(
        calibration,
        &["source_tensor_bindings", "gate"],
        "calibration source_tensor_bindings Gate descriptor",
    )?;
    expect_string(
        duplicate_gate,
        &["name"],
        "calibration duplicate Gate tensor name",
        &gate_weight_name,
    )?;
    expect_string(
        duplicate_gate,
        &["dtype"],
        "calibration duplicate Gate tensor dtype",
        "BF16",
    )?;
    expect_shape(
        duplicate_gate,
        &["shape"],
        &[ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64],
        "calibration duplicate Gate tensor shape",
    )?;
    expect_u64(
        duplicate_gate,
        &["bytes"],
        "calibration duplicate Gate tensor bytes",
        GATE_WEIGHT_BYTES as u64,
    )?;
    expect_u64(
        duplicate_gate,
        &["verified_chunk_count"],
        "calibration duplicate Gate verified chunk count",
        1,
    )?;
    expect_u64(
        duplicate_gate,
        &["verified_chunk_bytes"],
        "calibration duplicate Gate verified chunk bytes",
        GATE_WEIGHT_BYTES as u64,
    )?;
    expect_string(
        duplicate_gate,
        &["logical_tensor_sha256"],
        "calibration duplicate Gate tensor SHA-256",
        &gate_weight_sha256,
    )?;
    expect_single_sha256(
        duplicate_gate,
        &["verified_chunk_sha256"],
        "calibration duplicate Gate verified chunk SHA-256",
        &gate_weight_sha256,
    )?;

    expect_string(
        trace,
        &["model_source", "p6_gate_route_bindings", "gate_weight_name"],
        "trace Gate tensor name",
        &gate_weight_name,
    )?;
    expect_string(
        trace,
        &[
            "model_source",
            "p6_gate_route_bindings",
            "gate_weight_sha256",
        ],
        "trace Gate tensor SHA-256",
        &gate_weight_sha256,
    )?;
    let metadata = reader.tensor_metadata(&gate_weight_name)?;
    if metadata.dtype != "BF16"
        || metadata.shape.as_slice() != [ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64]
        || metadata.bytes != GATE_WEIGHT_BYTES as u64
    {
        return Err(calibration_error(
            "admitted artifact Gate tensor metadata is not BF16[256,4096]",
        ));
    }
    let gate_bytes = reader.read_verified_full(&gate_weight_name, GATE_WEIGHT_BYTES)?;
    if gate_bytes.len() != GATE_WEIGHT_BYTES || sha256_hex(&gate_bytes) != gate_weight_sha256 {
        return Err(calibration_error(
            "admitted artifact Gate tensor bytes/hash differ from calibration binding",
        ));
    }

    let tid2eid = required_object(
        calibration,
        &["source_tensor_bindings", "tid2eid"],
        "calibration source_tensor_bindings tid2eid descriptor",
    )?;
    expect_string(
        tid2eid,
        &["name"],
        "calibration tid2eid tensor name",
        LAYER0_FFN_GATE_TID2EID,
    )?;
    expect_string(
        tid2eid,
        &["dtype"],
        "calibration tid2eid tensor dtype",
        "I64",
    )?;
    expect_shape(
        tid2eid,
        &["shape"],
        &[129_280, ACTIVATED_EXPERTS as u64],
        "calibration tid2eid tensor shape",
    )?;
    expect_u64(
        tid2eid,
        &["bytes"],
        "calibration tid2eid tensor bytes",
        TID2EID_BYTES as u64,
    )?;
    expect_u64(
        tid2eid,
        &["verified_chunk_count"],
        "calibration tid2eid verified chunk count",
        1,
    )?;
    expect_u64(
        tid2eid,
        &["verified_chunk_bytes"],
        "calibration tid2eid verified chunk bytes",
        TID2EID_BYTES as u64,
    )?;
    let tid2eid_name =
        required_string(tid2eid, &["name"], "calibration tid2eid tensor name")?.to_owned();
    let tid2eid_sha256 = required_string(
        tid2eid,
        &["logical_tensor_sha256"],
        "calibration tid2eid tensor SHA-256",
    )?
    .to_owned();
    expect_single_sha256(
        tid2eid,
        &["verified_chunk_sha256"],
        "calibration tid2eid verified chunk SHA-256",
        &tid2eid_sha256,
    )?;
    expect_string(
        trace,
        &["model_source", "p6_gate_route_bindings", "tid2eid_name"],
        "trace tid2eid tensor name",
        &tid2eid_name,
    )?;
    expect_string(
        trace,
        &["model_source", "p6_gate_route_bindings", "tid2eid_sha256"],
        "trace tid2eid tensor SHA-256",
        &tid2eid_sha256,
    )?;
    let tid2eid_metadata = reader.tensor_metadata(&tid2eid_name)?;
    if tid2eid_metadata.dtype != "I64"
        || tid2eid_metadata.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64]
        || tid2eid_metadata.bytes != TID2EID_BYTES as u64
    {
        return Err(calibration_error(
            "admitted artifact tid2eid metadata is not I64[129280,6]",
        ));
    }
    let tid2eid_bytes = reader.read_verified_full(&tid2eid_name, TID2EID_BYTES)?;
    if tid2eid_bytes.len() != TID2EID_BYTES || sha256_hex(&tid2eid_bytes) != tid2eid_sha256 {
        return Err(calibration_error(
            "admitted artifact tid2eid bytes/hash differ from trace route binding",
        ));
    }
    Ok((
        gate_weight_name,
        gate_weight_sha256,
        tid2eid_name,
        tid2eid_sha256,
    ))
}

fn validate_v2_route_tensor_bindings(
    calibration: &Value,
    reader: &DeepSeekV4FullStreamReader,
) -> Result<(String, String, String, String)> {
    let gate = required_object(
        calibration,
        &["source_tensor_bindings", "gate"],
        "v2 route calibration Gate tensor descriptor",
    )?;
    expect_string(
        gate,
        &["name"],
        "v2 route calibration Gate tensor name",
        LAYER0_FFN_GATE_WEIGHT,
    )?;
    expect_string(
        gate,
        &["dtype"],
        "v2 route calibration Gate tensor dtype",
        "BF16",
    )?;
    expect_shape(
        gate,
        &["shape"],
        &[ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64],
        "v2 route calibration Gate tensor shape",
    )?;
    expect_u64(
        gate,
        &["bytes"],
        "v2 route calibration Gate tensor bytes",
        GATE_WEIGHT_BYTES as u64,
    )?;
    expect_u64(
        gate,
        &["verified_chunk_count"],
        "v2 route calibration Gate verified chunk count",
        1,
    )?;
    expect_u64(
        gate,
        &["verified_chunk_bytes"],
        "v2 route calibration Gate verified chunk bytes",
        GATE_WEIGHT_BYTES as u64,
    )?;
    let gate_weight_name =
        required_string(gate, &["name"], "v2 route calibration Gate tensor name")?.to_owned();
    let gate_weight_sha256 = required_string(
        gate,
        &["logical_tensor_sha256"],
        "v2 route calibration Gate tensor SHA-256",
    )?
    .to_owned();
    expect_single_sha256(
        gate,
        &["verified_chunk_sha256"],
        "v2 route calibration Gate verified chunk SHA-256",
        &gate_weight_sha256,
    )?;
    let gate_metadata = reader.tensor_metadata(&gate_weight_name)?;
    if gate_metadata.dtype != "BF16"
        || gate_metadata.shape.as_slice() != [ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64]
        || gate_metadata.bytes != GATE_WEIGHT_BYTES as u64
    {
        return Err(calibration_error(
            "admitted artifact Gate tensor metadata is not BF16[256,4096] for v2 route calibration",
        ));
    }
    let gate_bytes = reader.read_verified_full(&gate_weight_name, GATE_WEIGHT_BYTES)?;
    if gate_bytes.len() != GATE_WEIGHT_BYTES || sha256_hex(&gate_bytes) != gate_weight_sha256 {
        return Err(calibration_error(
            "admitted artifact Gate bytes/hash differ from v2 route calibration binding",
        ));
    }

    let tid2eid = required_object(
        calibration,
        &["source_tensor_bindings", "tid2eid"],
        "v2 route calibration tid2eid tensor descriptor",
    )?;
    expect_string(
        tid2eid,
        &["name"],
        "v2 route calibration tid2eid tensor name",
        LAYER0_FFN_GATE_TID2EID,
    )?;
    expect_string(
        tid2eid,
        &["dtype"],
        "v2 route calibration tid2eid tensor dtype",
        "I64",
    )?;
    expect_shape(
        tid2eid,
        &["shape"],
        &[129_280, ACTIVATED_EXPERTS as u64],
        "v2 route calibration tid2eid tensor shape",
    )?;
    expect_u64(
        tid2eid,
        &["bytes"],
        "v2 route calibration tid2eid tensor bytes",
        TID2EID_BYTES as u64,
    )?;
    expect_u64(
        tid2eid,
        &["verified_chunk_count"],
        "v2 route calibration tid2eid verified chunk count",
        1,
    )?;
    expect_u64(
        tid2eid,
        &["verified_chunk_bytes"],
        "v2 route calibration tid2eid verified chunk bytes",
        TID2EID_BYTES as u64,
    )?;
    let tid2eid_name = required_string(
        tid2eid,
        &["name"],
        "v2 route calibration tid2eid tensor name",
    )?
    .to_owned();
    let tid2eid_sha256 = required_string(
        tid2eid,
        &["logical_tensor_sha256"],
        "v2 route calibration tid2eid tensor SHA-256",
    )?
    .to_owned();
    expect_single_sha256(
        tid2eid,
        &["verified_chunk_sha256"],
        "v2 route calibration tid2eid verified chunk SHA-256",
        &tid2eid_sha256,
    )?;
    let tid2eid_metadata = reader.tensor_metadata(&tid2eid_name)?;
    if tid2eid_metadata.dtype != "I64"
        || tid2eid_metadata.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64]
        || tid2eid_metadata.bytes != TID2EID_BYTES as u64
    {
        return Err(calibration_error(
            "admitted artifact tid2eid metadata is not I64[129280,6] for v2 route calibration",
        ));
    }
    let tid2eid_bytes = reader.read_verified_full(&tid2eid_name, TID2EID_BYTES)?;
    if tid2eid_bytes.len() != TID2EID_BYTES || sha256_hex(&tid2eid_bytes) != tid2eid_sha256 {
        return Err(calibration_error(
            "admitted artifact tid2eid bytes/hash differ from v2 route calibration binding",
        ));
    }
    Ok((
        gate_weight_name,
        gate_weight_sha256,
        tid2eid_name,
        tid2eid_sha256,
    ))
}

fn validate_v2_route_target_contract(calibration: &Value) -> Result<()> {
    expect_string(
        calibration,
        &["route_target_contract", "source_model_operator"],
        "v2 route calibration source model operator",
        "Gate.forward",
    )?;
    expect_string(
        calibration,
        &["route_target_contract", "score_function"],
        "v2 route calibration score function",
        "sqrtsoftplus",
    )?;
    expect_string(
        calibration,
        &["route_target_contract", "post_linear_operator_order"],
        "v2 route calibration post-linear operator order",
        "F.softplus(scores).sqrt() -> original_scores -> tid2eid[input_ids] -> original_scores.gather(1, indices) -> divide selected sum -> multiply route_scale",
    )?;
    expect_string(
        calibration,
        &["route_target_contract", "selection_method"],
        "v2 route calibration selection method",
        "source tid2eid[token_id] fixed row",
    )?;
    expect_u64(
        calibration,
        &["route_target_contract", "layer"],
        "v2 route calibration route layer",
        0,
    )?;
    expect_u64(
        calibration,
        &["route_target_contract", "token_id"],
        "v2 route calibration route token ID",
        PREFIX_TOKEN_ID,
    )?;
    expect_u64(
        calibration,
        &["route_target_contract", "token_position"],
        "v2 route calibration route token position",
        0,
    )?;
    expect_u64(
        calibration,
        &["route_target_contract", "top_k"],
        "v2 route calibration top-k",
        ACTIVATED_EXPERTS as u64,
    )?;
    expect_f64(
        calibration,
        &["route_target_contract", "route_scale"],
        "v2 route calibration route scale",
        f64::from(ROUTE_SCALE),
    )
}

fn read_absolute_regular_bounded(path: &Path, label: &str, max_bytes: usize) -> Result<Vec<u8>> {
    if !path.is_absolute() {
        return Err(calibration_error(format!("{label} path must be absolute")));
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(calibration_error(format!(
            "{label} must be a regular non-symlink file",
        )));
    }
    let byte_count = usize::try_from(metadata.len())
        .map_err(|_| calibration_error(format!("{label} length does not fit host usize")))?;
    if byte_count > max_bytes {
        return Err(calibration_error(format!(
            "{label} has {byte_count} bytes, exceeding its {max_bytes}-byte bound",
        )));
    }
    let raw = fs::read(path)?;
    if raw.len() != byte_count {
        return Err(calibration_error(format!(
            "{label} changed length while being read",
        )));
    }
    Ok(raw)
}

fn parse_json(raw: &[u8], label: &str) -> Result<Value> {
    serde_json::from_slice(raw)
        .map_err(|error| calibration_error(format!("{label} is not valid JSON: {error}")))
}

fn required_value<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a Value> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| calibration_error(format!("missing {label} at {}", path.join("."))))?;
    }
    Ok(current)
}

fn required_object<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a Value> {
    let object = required_value(value, path, label)?;
    if !object.is_object() {
        return Err(calibration_error(format!("{label} must be a JSON object")));
    }
    Ok(object)
}

fn required_string<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a str> {
    required_value(value, path, label)?
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| calibration_error(format!("{label} must be a non-empty string")))
}

fn expect_string(value: &Value, path: &[&str], label: &str, expected: &str) -> Result<()> {
    let observed = required_string(value, path, label)?;
    if observed != expected {
        return Err(calibration_error(format!(
            "{label} mismatch: expected {expected:?}, got {observed:?}",
        )));
    }
    Ok(())
}

fn expect_bool(value: &Value, path: &[&str], label: &str, expected: bool) -> Result<()> {
    let observed = required_value(value, path, label)?
        .as_bool()
        .ok_or_else(|| calibration_error(format!("{label} must be boolean")))?;
    if observed != expected {
        return Err(calibration_error(format!(
            "{label} mismatch: expected {expected}, got {observed}",
        )));
    }
    Ok(())
}

fn expect_u64(value: &Value, path: &[&str], label: &str, expected: u64) -> Result<()> {
    let observed = required_value(value, path, label)?
        .as_u64()
        .ok_or_else(|| calibration_error(format!("{label} must be an unsigned integer")))?;
    if observed != expected {
        return Err(calibration_error(format!(
            "{label} mismatch: expected {expected}, got {observed}",
        )));
    }
    Ok(())
}

fn expect_f64(value: &Value, path: &[&str], label: &str, expected: f64) -> Result<()> {
    let observed = required_value(value, path, label)?
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| calibration_error(format!("{label} must be a finite JSON number")))?;
    if observed.to_bits() != expected.to_bits() {
        return Err(calibration_error(format!(
            "{label} mismatch: expected {expected}, got {observed}",
        )));
    }
    Ok(())
}

fn expect_shape(value: &Value, path: &[&str], expected: &[u64], label: &str) -> Result<()> {
    let observed = required_value(value, path, label)?
        .as_array()
        .ok_or_else(|| calibration_error(format!("{label} must be an integer array")))?
        .iter()
        .map(|dimension| {
            dimension
                .as_u64()
                .ok_or_else(|| calibration_error(format!("{label} has a non-integer dimension")))
        })
        .collect::<Result<Vec<_>>>()?;
    if observed.as_slice() != expected {
        return Err(calibration_error(format!(
            "{label} mismatch: expected {expected:?}, got {observed:?}",
        )));
    }
    Ok(())
}

fn expect_single_sha256(value: &Value, path: &[&str], label: &str, expected: &str) -> Result<()> {
    let values = required_value(value, path, label)?
        .as_array()
        .ok_or_else(|| calibration_error(format!("{label} must be a one-element string array")))?;
    if values.len() != 1 {
        return Err(calibration_error(format!(
            "{label} must contain exactly one SHA-256 string",
        )));
    }
    let observed = values[0]
        .as_str()
        .filter(|text| !text.is_empty())
        .ok_or_else(|| calibration_error(format!("{label} element must be a non-empty string")))?;
    if observed != expected {
        return Err(calibration_error(format!(
            "{label} mismatch: expected {expected:?}, got {observed:?}",
        )));
    }
    Ok(())
}

fn decode_lowercase_hex(input: &str, expected_bytes: usize, label: &str) -> Result<Vec<u8>> {
    if input.len() != expected_bytes * 2
        || !input
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(calibration_error(format!(
            "{label} must be exactly lowercase hex for {expected_bytes} bytes",
        )));
    }
    input
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = hex_nibble(pair[0])
                .ok_or_else(|| calibration_error(format!("{label} contains invalid hex")))?;
            let low = hex_nibble(pair[1])
                .ok_or_else(|| calibration_error(format!("{label} contains invalid hex")))?;
            Ok((high << 4) | low)
        })
        .collect()
}

fn hex_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        _ => None,
    }
}

fn decode_f32_le(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() != GATE_LOGIT_BYTES {
        return Err(calibration_error(
            "calibration target byte length is not F32[256]",
        ));
    }
    let logits = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect::<Vec<_>>();
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(calibration_error(
            "calibration target includes a non-finite F32 value",
        ));
    }
    Ok(logits)
}

fn load_f32_route_target(
    raw_targets: &Value,
    key: &str,
    expected_name: &str,
    expected_count: usize,
    label: &str,
) -> Result<(String, Vec<f32>)> {
    let target = required_object(raw_targets, &[key], label)?;
    expect_string(target, &["name"], label, expected_name)?;
    expect_string(target, &["dtype"], label, "F32")?;
    expect_shape(target, &["shape"], &[expected_count as u64], label)?;
    expect_u64(target, &["element_count"], label, expected_count as u64)?;
    expect_string(target, &["byte_order"], label, "little_endian")?;
    let expected_bytes = expected_count
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| calibration_error(format!("{label} byte count overflow")))?;
    expect_u64(target, &["byte_count"], label, expected_bytes as u64)?;
    expect_string(target, &["encoding"], label, "lowercase_hex_raw_f32_le")?;
    let sha256 = required_string(target, &["sha256"], label)?.to_owned();
    let bytes = decode_lowercase_hex(
        required_string(target, &["data"], label)?,
        expected_bytes,
        label,
    )?;
    if sha256_hex(&bytes) != sha256 {
        return Err(calibration_error(format!(
            "{label} payload SHA-256 mismatch"
        )));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect::<Vec<_>>();
    if values.len() != expected_count || values.iter().any(|value| !value.is_finite()) {
        return Err(calibration_error(format!(
            "{label} must decode to finite F32[{expected_count}]",
        )));
    }
    Ok((sha256, values))
}

fn load_u16_route_target(
    raw_targets: &Value,
    key: &str,
    expected_name: &str,
    expected_count: usize,
    label: &str,
) -> Result<(String, Vec<u16>)> {
    let target = required_object(raw_targets, &[key], label)?;
    expect_string(target, &["name"], label, expected_name)?;
    expect_string(target, &["dtype"], label, "U16")?;
    expect_shape(target, &["shape"], &[expected_count as u64], label)?;
    expect_u64(target, &["element_count"], label, expected_count as u64)?;
    expect_string(target, &["byte_order"], label, "little_endian")?;
    let expected_bytes = expected_count
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| calibration_error(format!("{label} byte count overflow")))?;
    expect_u64(target, &["byte_count"], label, expected_bytes as u64)?;
    expect_string(target, &["encoding"], label, "lowercase_hex_raw_u16_le")?;
    let sha256 = required_string(target, &["sha256"], label)?.to_owned();
    let bytes = decode_lowercase_hex(
        required_string(target, &["data"], label)?,
        expected_bytes,
        label,
    )?;
    if sha256_hex(&bytes) != sha256 {
        return Err(calibration_error(format!(
            "{label} payload SHA-256 mismatch"
        )));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();
    if values.len() != expected_count {
        return Err(calibration_error(format!(
            "{label} must decode to U16[{expected_count}]",
        )));
    }
    Ok((sha256, values))
}

fn read_verified_tid2eid_row(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
) -> Result<Vec<u16>> {
    let metadata = reader.tensor_metadata(LAYER0_FFN_GATE_TID2EID)?;
    if metadata.dtype != "I64"
        || metadata.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64]
        || metadata.bytes != TID2EID_BYTES as u64
        || token_id >= metadata.shape[0]
    {
        return Err(calibration_error(
            "admitted artifact tid2eid metadata/token range differs from P0 route contract",
        ));
    }
    let row_bytes = ACTIVATED_EXPERTS
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| calibration_error("tid2eid row byte count overflow"))?;
    let start = usize::try_from(token_id)
        .map_err(|_| calibration_error("tid2eid token ID does not fit host usize"))?
        .checked_mul(row_bytes)
        .ok_or_else(|| calibration_error("tid2eid row byte offset overflow"))?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| calibration_error("tid2eid row byte end overflow"))?;
    let raw =
        reader.read_verified_range(LAYER0_FFN_GATE_TID2EID, start as u64..end as u64, row_bytes)?;
    let mut ids = Vec::with_capacity(ACTIVATED_EXPERTS);
    for chunk in raw.chunks_exact(std::mem::size_of::<i64>()) {
        let id = i64::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| calibration_error("tid2eid row contains incomplete I64"))?,
        );
        if id < 0 || id >= ROUTED_EXPERTS as i64 {
            return Err(calibration_error(
                "tid2eid row contains an out-of-range routed-expert ID",
            ));
        }
        ids.push(id as u16);
    }
    if ids.len() != ACTIVATED_EXPERTS {
        return Err(calibration_error(
            "tid2eid row did not decode exactly six routed-expert IDs",
        ));
    }
    Ok(ids)
}

fn count_object_key(value: &Value, key: &str) -> usize {
    match value {
        Value::Array(values) => values
            .iter()
            .map(|value| count_object_key(value, key))
            .sum(),
        Value::Object(values) => {
            usize::from(values.contains_key(key))
                + values
                    .values()
                    .map(|value| count_object_key(value, key))
                    .sum::<usize>()
        }
        _ => 0,
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn calibration_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 P0 Torch F32 Gate calibration: {}",
        message.into()
    ))
}
