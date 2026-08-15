//! CPU-only Qwen3-Coder-Next per-layer MoE device-readiness contract.
//!
//! This standalone ledger defines the exact evidence required at every one of
//! Qwen80's 48 MoE boundaries: post-attention norm, router/top-10 selection,
//! all ten routed bodies, shared expert, combine, and second residual. It
//! consumes metadata only. It never opens artifacts, creates Metal work,
//! contacts a runtime/watcher/server/HCLI endpoint, or measures TPS/TG.
//!
//! The known layer-0 fragments are intentionally preserved as partial facts:
//! all-ten CPU witnesses, a shared-expert component, and a materialized
//! fixture-combine component. They have no same-input physical join and cannot
//! be promoted into an MoE layer, decoder, server, or token result.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_moe_per_layer_device_readiness_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_moe_per_layer_device_readiness_result.v1";
const LAYER_EVIDENCE_SCHEMA: &str =
    "hawking.ascension.qwen80_moe_per_layer_physical_device_ledger.v1";
const ALL_TEN_CPU_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_cpu_oracle.v1";
const ALL_TEN_CPU_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_ORACLE_READY_FOR_SEPARATE_DEVICE_LEASE";
const SHARED_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_shared_expert_wave.v1";
const SHARED_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_STRICT_MATH_METAL_COMPONENT_NOT_ROUTED_MOE_OR_LAYER";
const COMBINE_COMPONENT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_moe_combine.v1";
const COMBINE_COMPONENT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const POSTNORM_ROUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const POSTNORM_ROUTER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const LAYER_COUNT: usize = 48;
const HIDDEN: usize = 2_048;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const INTERMEDIATE: usize = 512;
const GROUP_SIZE: usize = 128;
const RMS_NORM_EPSILON: f64 = 1.0e-6;
const L0_ROUTE_IDS: [u16; TOP_K] = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494];
const L0_ROUTE_WEIGHTS: [f64; TOP_K] = [
    0.245_458_886_027_336_12,
    0.119_394_913_315_773_01,
    0.098_652_511_835_098_27,
    0.098_244_741_559_028_63,
    0.081_222_802_400_588_99,
    0.078_011_848_032_474_52,
    0.073_711_447_417_736_05,
    0.071_626_946_330_070_5,
    0.069_213_777_780_532_84,
    0.064_462_073_147_296_9,
];

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SourceIdentity {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
}

impl SourceIdentity {
    fn exact() -> Self {
        Self {
            model_id: MODEL_ID.into(),
            model_key: MODEL_KEY.into(),
            source_repository: SOURCE_REPOSITORY.into(),
            source_revision: SOURCE_REVISION.into(),
            manifest_seal_sha256: MANIFEST_SEAL.into(),
            admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL.into(),
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self != &Self::exact() {
            return Err(format!("{label} source/artifact identity drifted"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
enum MoeOperation {
    PostAttentionRmsNorm,
    RouterProjection,
    Top10SelectAndNormalize,
    RoutedGateUpWaves,
    RoutedSiluTimesUp,
    RoutedDownAndSourceWeight,
    SharedGateUp,
    SharedSiluTimesUp,
    SharedDown,
    SharedScalarSigmoidGate,
    CombineTenWeightedRouteOutputs,
    AddGatedShared,
    SecondResidualAddFirstResidual,
}

fn expected_operation_order() -> Vec<MoeOperation> {
    vec![
        MoeOperation::PostAttentionRmsNorm,
        MoeOperation::RouterProjection,
        MoeOperation::Top10SelectAndNormalize,
        MoeOperation::RoutedGateUpWaves,
        MoeOperation::RoutedSiluTimesUp,
        MoeOperation::RoutedDownAndSourceWeight,
        MoeOperation::SharedGateUp,
        MoeOperation::SharedSiluTimesUp,
        MoeOperation::SharedDown,
        MoeOperation::SharedScalarSigmoidGate,
        MoeOperation::CombineTenWeightedRouteOutputs,
        MoeOperation::AddGatedShared,
        MoeOperation::SecondResidualAddFirstResidual,
    ]
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct MoeGeometry {
    hidden_size: usize,
    expert_count: usize,
    top_k: usize,
    intermediate_size: usize,
    packed_group_size: usize,
    rms_norm_epsilon: f64,
    post_attention_norm_shape: Vec<usize>,
    post_attention_norm_residual_scale_one_plus_weight: bool,
    router_shape: Vec<usize>,
    routed_gate_shape: Vec<usize>,
    routed_up_shape: Vec<usize>,
    routed_down_shape: Vec<usize>,
    shared_gate_shape: Vec<usize>,
    shared_up_shape: Vec<usize>,
    shared_down_shape: Vec<usize>,
    shared_scalar_gate_shape: Vec<usize>,
    route_output_shape: Vec<usize>,
    shared_output_shape: Vec<usize>,
    second_residual_shape: Vec<usize>,
}

impl MoeGeometry {
    fn exact() -> Self {
        Self {
            hidden_size: HIDDEN,
            expert_count: EXPERTS,
            top_k: TOP_K,
            intermediate_size: INTERMEDIATE,
            packed_group_size: GROUP_SIZE,
            rms_norm_epsilon: RMS_NORM_EPSILON,
            post_attention_norm_shape: vec![HIDDEN],
            post_attention_norm_residual_scale_one_plus_weight: true,
            router_shape: vec![EXPERTS, HIDDEN],
            routed_gate_shape: vec![INTERMEDIATE, HIDDEN],
            routed_up_shape: vec![INTERMEDIATE, HIDDEN],
            routed_down_shape: vec![HIDDEN, INTERMEDIATE],
            shared_gate_shape: vec![INTERMEDIATE, HIDDEN],
            shared_up_shape: vec![INTERMEDIATE, HIDDEN],
            shared_down_shape: vec![HIDDEN, INTERMEDIATE],
            shared_scalar_gate_shape: vec![1, HIDDEN],
            route_output_shape: vec![HIDDEN],
            shared_output_shape: vec![HIDDEN],
            second_residual_shape: vec![HIDDEN],
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self.hidden_size != HIDDEN
            || self.expert_count != EXPERTS
            || self.top_k != TOP_K
            || self.intermediate_size != INTERMEDIATE
            || self.packed_group_size != GROUP_SIZE
            || self.rms_norm_epsilon.to_bits() != RMS_NORM_EPSILON.to_bits()
            || self.post_attention_norm_shape != [HIDDEN]
            || !self.post_attention_norm_residual_scale_one_plus_weight
            || self.router_shape != [EXPERTS, HIDDEN]
            || self.routed_gate_shape != [INTERMEDIATE, HIDDEN]
            || self.routed_up_shape != [INTERMEDIATE, HIDDEN]
            || self.routed_down_shape != [HIDDEN, INTERMEDIATE]
            || self.shared_gate_shape != [INTERMEDIATE, HIDDEN]
            || self.shared_up_shape != [INTERMEDIATE, HIDDEN]
            || self.shared_down_shape != [HIDDEN, INTERMEDIATE]
            || self.shared_scalar_gate_shape != [1, HIDDEN]
            || self.route_output_shape != [HIDDEN]
            || self.shared_output_shape != [HIDDEN]
            || self.second_residual_shape != [HIDDEN]
        {
            return Err(format!(
                "{label} postnorm/router/top10/routed/shared/combine/residual geometry drifted"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CurrentLayer0Fragments {
    source_identity: SourceIdentity,
    layer: usize,
    postnorm_router_schema: String,
    postnorm_router_status: String,
    all_ten_cpu_schema: String,
    all_ten_cpu_status: String,
    shared_component_schema: String,
    shared_component_status: String,
    combine_component_schema: String,
    combine_component_status: String,
    route_ids: Vec<u16>,
    route_weights: Vec<f64>,
    all_ten_cpu_waves_executed: bool,
    all_ten_cpu_device_execution: bool,
    all_ten_cpu_device_route_buffers_retained: bool,
    shared_component_strict_math_metal: bool,
    shared_component_has_same_input_provenance: bool,
    combine_component_strict_math_metal: bool,
    combine_materialized_source_route_shaped_fixture_only: bool,
    combine_has_same_input_hash_join: bool,
}

impl CurrentLayer0Fragments {
    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if let Err(error) = self
            .source_identity
            .validate_exact("current layer-0 MoE fragments")
        {
            errors.push(error);
        }
        if self.layer != 0
            || self.postnorm_router_schema != POSTNORM_ROUTER_SCHEMA
            || self.postnorm_router_status != POSTNORM_ROUTER_STATUS
            || self.all_ten_cpu_schema != ALL_TEN_CPU_SCHEMA
            || self.all_ten_cpu_status != ALL_TEN_CPU_STATUS
            || self.shared_component_schema != SHARED_COMPONENT_SCHEMA
            || self.shared_component_status != SHARED_COMPONENT_STATUS
            || self.combine_component_schema != COMBINE_COMPONENT_SCHEMA
            || self.combine_component_status != COMBINE_COMPONENT_STATUS
        {
            errors.push("current layer-0 component schema/status binding drifted".into());
        }
        if self.route_ids.as_slice() != L0_ROUTE_IDS
            || self.route_weights.len() != TOP_K
            || self
                .route_weights
                .iter()
                .zip(L0_ROUTE_WEIGHTS)
                .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
        {
            errors.push("current layer-0 all-ten source route/order/weights drifted".into());
        }
        if !self.all_ten_cpu_waves_executed
            || self.all_ten_cpu_device_execution
            || self.all_ten_cpu_device_route_buffers_retained
            || !self.shared_component_strict_math_metal
            || self.shared_component_has_same_input_provenance
            || !self.combine_component_strict_math_metal
            || !self.combine_materialized_source_route_shaped_fixture_only
            || self.combine_has_same_input_hash_join
        {
            errors.push("current layer-0 fragments no longer describe the known CPU/shared/fixture-combine boundary".into());
        }
        errors
    }

    fn nonjoinable_reasons(&self) -> Vec<&'static str> {
        vec![
            "all-ten evidence is a CPU oracle only and retains no same-capture device route buffers",
            "the shared-expert component has no first-residual/postnorm same-input provenance",
            "the combine component is a materialized source-route-shaped fixture, not ten physical route outputs",
            "no durable single device capture joins first residual, postnorm, router, all ten bodies, shared output, combine, and second residual",
        ]
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct RouteWaveWitness {
    wave_index: usize,
    expert_id: u16,
    normalized_weight: f64,
    input_postnorm_hidden_sha256: String,
    output_weighted_route_sha256: String,
    gate_up_device_parity_passed: bool,
    activation_device_parity_passed: bool,
    down_device_parity_passed: bool,
    source_weight_applied_after_down: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SameInputMoeWitness {
    capture_id_sha256: String,
    first_residual_sha256: String,
    postnorm_input_sha256: String,
    postnorm_hidden_sha256: String,
    router_input_sha256: String,
    routed_input_sha256: String,
    shared_input_sha256: String,
    combine_first_residual_sha256: String,
    combine_postnorm_hidden_sha256: String,
    combine_route_output_sha256: Vec<String>,
    combine_gated_shared_sha256: String,
    gated_shared_sha256: String,
    second_residual_sha256: String,
    route_ids: Vec<u16>,
    normalized_route_weights: Vec<f64>,
    route_waves: Vec<RouteWaveWitness>,
    device_buffers_retained_until_capture_fence: bool,
    one_same_input_command_graph_capture: bool,
    source_router_top10_parity_passed: bool,
    postnorm_device_parity_passed: bool,
    router_device_parity_passed: bool,
    top10_device_parity_passed: bool,
    all_ten_route_bodies_device_parity_passed: bool,
    shared_expert_device_parity_passed: bool,
    combine_device_parity_passed: bool,
    second_residual_device_parity_passed: bool,
}

impl SameInputMoeWitness {
    fn validation_errors(&self, layer: usize) -> Vec<String> {
        let mut errors = Vec::new();
        for (label, value) in [
            ("capture_id", self.capture_id_sha256.as_str()),
            ("first_residual", self.first_residual_sha256.as_str()),
            ("postnorm_input", self.postnorm_input_sha256.as_str()),
            ("postnorm_hidden", self.postnorm_hidden_sha256.as_str()),
            ("router_input", self.router_input_sha256.as_str()),
            ("routed_input", self.routed_input_sha256.as_str()),
            ("shared_input", self.shared_input_sha256.as_str()),
            (
                "combine_first_residual",
                self.combine_first_residual_sha256.as_str(),
            ),
            (
                "combine_postnorm_hidden",
                self.combine_postnorm_hidden_sha256.as_str(),
            ),
            (
                "combine_gated_shared",
                self.combine_gated_shared_sha256.as_str(),
            ),
            ("gated_shared", self.gated_shared_sha256.as_str()),
            ("second_residual", self.second_residual_sha256.as_str()),
        ] {
            if !is_lower_sha256(value) {
                errors.push(format!(
                    "layer {layer} same-input witness has invalid {label} digest"
                ));
            }
        }
        if self.postnorm_input_sha256 != self.first_residual_sha256
            || self.router_input_sha256 != self.postnorm_hidden_sha256
            || self.routed_input_sha256 != self.postnorm_hidden_sha256
            || self.shared_input_sha256 != self.postnorm_hidden_sha256
            || self.combine_first_residual_sha256 != self.first_residual_sha256
            || self.combine_postnorm_hidden_sha256 != self.postnorm_hidden_sha256
            || self.combine_gated_shared_sha256 != self.gated_shared_sha256
        {
            errors.push(format!(
                "layer {layer} postnorm/router/routed/shared/combine inputs are not one same-input provenance chain"
            ));
        }
        if self.route_ids.len() != TOP_K
            || self.normalized_route_weights.len() != TOP_K
            || self.route_waves.len() != TOP_K
            || self.combine_route_output_sha256.len() != TOP_K
        {
            errors.push(format!(
                "layer {layer} witness does not retain exactly ten routed bodies"
            ));
            return errors;
        }
        let route_set = self.route_ids.iter().copied().collect::<BTreeSet<_>>();
        if route_set.len() != TOP_K
            || self
                .route_ids
                .iter()
                .any(|expert| *expert as usize >= EXPERTS)
        {
            errors.push(format!(
                "layer {layer} top-10 route ids are duplicate or out of range"
            ));
        }
        let mut weight_sum = 0.0f64;
        for weight in &self.normalized_route_weights {
            if !weight.is_finite() || *weight < 0.0 {
                errors.push(format!(
                    "layer {layer} has a non-finite or negative normalized route weight"
                ));
                break;
            }
            weight_sum += weight;
        }
        if (weight_sum - 1.0).abs() > 1.0e-9 {
            errors.push(format!(
                "layer {layer} normalized route weights sum to {weight_sum}, not one"
            ));
        }
        for (index, wave) in self.route_waves.iter().enumerate() {
            if wave.wave_index != index
                || wave.expert_id != self.route_ids[index]
                || wave.normalized_weight.to_bits()
                    != self.normalized_route_weights[index].to_bits()
                || wave.input_postnorm_hidden_sha256 != self.postnorm_hidden_sha256
                || wave.output_weighted_route_sha256 != self.combine_route_output_sha256[index]
                || !is_lower_sha256(&wave.output_weighted_route_sha256)
                || !wave.gate_up_device_parity_passed
                || !wave.activation_device_parity_passed
                || !wave.down_device_parity_passed
                || !wave.source_weight_applied_after_down
            {
                errors.push(format!(
                    "layer {layer} routed wave {index} lacks exact all-ten same-input/projection/weight provenance"
                ));
            }
        }
        if !self.device_buffers_retained_until_capture_fence
            || !self.one_same_input_command_graph_capture
            || !self.source_router_top10_parity_passed
            || !self.postnorm_device_parity_passed
            || !self.router_device_parity_passed
            || !self.top10_device_parity_passed
            || !self.all_ten_route_bodies_device_parity_passed
            || !self.shared_expert_device_parity_passed
            || !self.combine_device_parity_passed
            || !self.second_residual_device_parity_passed
        {
            errors.push(format!(
                "layer {layer} lacks a fenced same-input device witness for postnorm/router/top10/all-ten/shared/combine/second residual"
            ));
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct MoeLayerPhysicalEvidence {
    schema: String,
    layer: usize,
    receipt_seal_sha256: String,
    source_tensor_prefix: String,
    source_identity: SourceIdentity,
    geometry: MoeGeometry,
    ordered_operations: Vec<MoeOperation>,
    backend: String,
    device_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    full_moe_boundary_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
    fallback_used: bool,
    same_input_witness: SameInputMoeWitness,
}

impl MoeLayerPhysicalEvidence {
    fn validation_errors(&self, expected_layer: usize) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != LAYER_EVIDENCE_SCHEMA {
            errors.push(format!(
                "layer {expected_layer} physical ledger schema drifted"
            ));
        }
        if self.layer != expected_layer {
            errors.push(format!(
                "physical evidence maps layer {} instead of source MoE layer {expected_layer}",
                self.layer
            ));
        }
        if !is_lower_sha256(&self.receipt_seal_sha256) {
            errors.push(format!(
                "layer {expected_layer} has no valid sealed physical receipt digest"
            ));
        }
        if self.source_tensor_prefix != format!("model.layers.{expected_layer}") {
            errors.push(format!(
                "layer {expected_layer} source tensor prefix drifted"
            ));
        }
        if let Err(error) = self
            .source_identity
            .validate_exact(&format!("layer {expected_layer}"))
        {
            errors.push(error);
        }
        if let Err(error) = self
            .geometry
            .validate_exact(&format!("layer {expected_layer}"))
        {
            errors.push(error);
        }
        if self.ordered_operations != expected_operation_order() {
            errors.push(format!(
                "layer {expected_layer} postnorm/router/top10/routed/shared/combine/second-residual order drifted"
            ));
        }
        if self.backend != "metal"
            || self.device_dispatches == 0
            || !self.source_bound
            || !self.artifact_bound
            || !self.full_moe_boundary_path
            || !self.device_parity_passed
            || self.fixture_only
            || self.synthetic_input
            || self.component_only
            || self.fallback_used
        {
            errors.push(format!(
                "layer {expected_layer} lacks a non-fixture source/artifact-bound full-MoE-boundary Metal parity record"
            ));
        }
        errors.extend(self.same_input_witness.validation_errors(expected_layer));
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReadinessInput {
    schema: String,
    source_identity: SourceIdentity,
    moe_layers: Vec<usize>,
    current_layer0_fragments: CurrentLayer0Fragments,
    per_layer_physical_evidence: Vec<MoeLayerPhysicalEvidence>,
}

#[derive(Serialize)]
struct LayerAssessment {
    layer: usize,
    supplied_records: usize,
    receipt_seals: Vec<String>,
    satisfied: bool,
    failures: Vec<String>,
}

#[derive(Serialize)]
struct ReadinessReport {
    schema: &'static str,
    status: &'static str,
    moe_per_layer_device_readiness_earned: bool,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    input_schema_valid: bool,
    source_identity_valid: bool,
    exact_48_moe_layers_valid: bool,
    current_layer0_fragments_valid: bool,
    current_layer0_fragments_joinable: bool,
    current_layer0_fragment_validation_errors: Vec<String>,
    current_layer0_nonjoinable_reasons: Vec<&'static str>,
    per_layer_assessments: Vec<LayerAssessment>,
    missing_or_invalid_moe_layers: Vec<usize>,
    contract_errors: Vec<String>,
    expected_moe_layers: Vec<usize>,
    read_only_contract: bool,
    live_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    model_execution_performed: bool,
    hcli_execution_performed: bool,
    tps_or_tg_measurement_performed: bool,
    required_before_ready: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        && value.bytes().any(|byte| byte != b'0')
}

fn expected_moe_layers() -> Vec<usize> {
    (0..LAYER_COUNT).collect()
}

fn validate_moe_layers(layers: &[usize]) -> Result<(), String> {
    if layers != expected_moe_layers() {
        return Err("MoE layer list must be exactly 0..47 in source order".into());
    }
    if layers.iter().collect::<BTreeSet<_>>().len() != LAYER_COUNT {
        return Err("MoE layer list contains duplicates".into());
    }
    Ok(())
}

fn evaluate(input: &ReadinessInput) -> ReadinessReport {
    let mut contract_errors = Vec::new();
    let input_schema_valid = input.schema == INPUT_SCHEMA;
    if !input_schema_valid {
        contract_errors.push("input schema drifted".into());
    }
    let source_identity_valid = input.source_identity.validate_exact("input").is_ok();
    if !source_identity_valid {
        contract_errors.push("input source/artifact identity drifted".into());
    }
    let exact_48_moe_layers_valid = match validate_moe_layers(&input.moe_layers) {
        Ok(()) => true,
        Err(error) => {
            contract_errors.push(error);
            false
        }
    };
    let current_layer0_fragment_validation_errors =
        input.current_layer0_fragments.validation_errors();
    let current_layer0_fragments_valid = current_layer0_fragment_validation_errors.is_empty();
    let current_layer0_fragments_joinable = current_layer0_fragments_valid
        && input.current_layer0_fragments.all_ten_cpu_device_execution
        && input
            .current_layer0_fragments
            .all_ten_cpu_device_route_buffers_retained
        && input
            .current_layer0_fragments
            .shared_component_has_same_input_provenance
        && !input
            .current_layer0_fragments
            .combine_materialized_source_route_shaped_fixture_only
        && input
            .current_layer0_fragments
            .combine_has_same_input_hash_join;

    let expected_layers = expected_moe_layers();
    let mut records_by_layer = BTreeMap::<usize, Vec<&MoeLayerPhysicalEvidence>>::new();
    for record in &input.per_layer_physical_evidence {
        records_by_layer
            .entry(record.layer)
            .or_default()
            .push(record);
    }
    for layer in records_by_layer.keys() {
        if *layer >= LAYER_COUNT {
            contract_errors.push(format!(
                "physical evidence covers non-source MoE layer {layer}"
            ));
        }
    }

    let mut per_layer_assessments = Vec::with_capacity(LAYER_COUNT);
    let mut missing_or_invalid_moe_layers = Vec::new();
    for layer in &expected_layers {
        let records = records_by_layer.get(layer).cloned().unwrap_or_default();
        let mut failures = Vec::new();
        if records.len() != 1 {
            failures.push(format!(
                "expected exactly one sealed physical record for layer {layer}, found {}",
                records.len()
            ));
        }
        if let Some(record) = records.first() {
            failures.extend(record.validation_errors(*layer));
        }
        let satisfied = failures.is_empty();
        if !satisfied {
            missing_or_invalid_moe_layers.push(*layer);
        }
        per_layer_assessments.push(LayerAssessment {
            layer: *layer,
            supplied_records: records.len(),
            receipt_seals: records
                .iter()
                .map(|record| record.receipt_seal_sha256.clone())
                .collect(),
            satisfied,
            failures,
        });
    }

    let moe_per_layer_device_readiness_earned = input_schema_valid
        && source_identity_valid
        && exact_48_moe_layers_valid
        && current_layer0_fragments_valid
        && contract_errors.is_empty()
        && missing_or_invalid_moe_layers.is_empty();
    let status = if moe_per_layer_device_readiness_earned {
        "READY_FOR_QWEN80_MOE_PER_LAYER_DEVICE_INTEGRATION_NOT_A_COMPLETE_DECODER"
    } else {
        "INCOMPLETE_QWEN80_MOE_PER_LAYER_DEVICE_READINESS_CURRENT_L0_CPU_SHARED_FIXTURE_COMBINE_FRAGMENTS_ARE_NOT_JOINABLE"
    };
    let mut report = ReadinessReport {
        schema: RESULT_SCHEMA,
        status,
        moe_per_layer_device_readiness_earned,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        input_schema_valid,
        source_identity_valid,
        exact_48_moe_layers_valid,
        current_layer0_fragments_valid,
        current_layer0_fragments_joinable,
        current_layer0_fragment_validation_errors,
        current_layer0_nonjoinable_reasons: input.current_layer0_fragments.nonjoinable_reasons(),
        per_layer_assessments,
        missing_or_invalid_moe_layers,
        contract_errors,
        expected_moe_layers: expected_layers,
        read_only_contract: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        required_before_ready: vec![
            "Produce one sealed source/artifact-bound non-fixture full-MoE-boundary Metal parity ledger for every layer 0..47, with no fallback.",
            "For every ledger, prove exact postnorm, router, source top-10 order/weights, ten direct-packed gate/up/SiLU/down bodies, shared expert/gate, combine, and second residual order.",
            "Retain every first-residual, postnorm, router, all-ten route input/output, shared, combine, and second-residual hash in one fenced same-input command-graph capture.",
            "Prove each of the ten route vectors is source-weighted after down projection, device-parity checked, unique, ordered, and joined to the exact source router result.",
            "After this narrow MoE frontier is satisfied, independently satisfy DeltaNet, GQA, state, terminal, full-graph, tokenizer, and clean TPS/TG gates before any decoder/server claim.",
        ],
        claim_boundary: vec![
            "Current layer-0 all-ten CPU, shared-component, and materialized fixture-combine records remain valid partial evidence but have no same-input physical join.",
            "This CPU-only contract does not open artifacts or a device, execute an MoE boundary, produce a token, expose HCLI, or measure TPS/TG.",
            "Even a ready MoE-per-layer result is not a complete decoder, resident server, capability, Agent OS, tournament, BASE_TRUE_TPS, or TG qualification.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&report).unwrap_or_default())
    );
    report
}

fn current_layer0_fragments() -> CurrentLayer0Fragments {
    CurrentLayer0Fragments {
        source_identity: SourceIdentity::exact(),
        layer: 0,
        postnorm_router_schema: POSTNORM_ROUTER_SCHEMA.into(),
        postnorm_router_status: POSTNORM_ROUTER_STATUS.into(),
        all_ten_cpu_schema: ALL_TEN_CPU_SCHEMA.into(),
        all_ten_cpu_status: ALL_TEN_CPU_STATUS.into(),
        shared_component_schema: SHARED_COMPONENT_SCHEMA.into(),
        shared_component_status: SHARED_COMPONENT_STATUS.into(),
        combine_component_schema: COMBINE_COMPONENT_SCHEMA.into(),
        combine_component_status: COMBINE_COMPONENT_STATUS.into(),
        route_ids: L0_ROUTE_IDS.to_vec(),
        route_weights: L0_ROUTE_WEIGHTS.to_vec(),
        all_ten_cpu_waves_executed: true,
        all_ten_cpu_device_execution: false,
        all_ten_cpu_device_route_buffers_retained: false,
        shared_component_strict_math_metal: true,
        shared_component_has_same_input_provenance: false,
        combine_component_strict_math_metal: true,
        combine_materialized_source_route_shaped_fixture_only: true,
        combine_has_same_input_hash_join: false,
    }
}

fn current_evidence_input() -> ReadinessInput {
    ReadinessInput {
        schema: INPUT_SCHEMA.into(),
        source_identity: SourceIdentity::exact(),
        moe_layers: expected_moe_layers(),
        current_layer0_fragments: current_layer0_fragments(),
        per_layer_physical_evidence: Vec::new(),
    }
}

fn write_report_atomic(path: &Path, report: &ReadinessReport) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent is missing: {}", parent.display()).into());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

enum InputMode {
    Inventory(PathBuf),
    CurrentEvidence,
}

struct Arguments {
    input_mode: InputMode,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_moe_per_layer_device_readiness_contract \
--current-evidence|--input ABSOLUTE_JSON --out ABSOLUTE_JSON"
}

fn parse_args() -> Result<Arguments, Box<dyn Error>> {
    let mut inventory = None;
    let mut current_evidence = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input" => {
                let value = args.next().ok_or("missing absolute path after --input")?;
                if inventory.replace(PathBuf::from(value)).is_some() {
                    return Err("--input supplied more than once".into());
                }
            }
            "--current-evidence" => {
                if current_evidence {
                    return Err("--current-evidence supplied more than once".into());
                }
                current_evidence = true;
            }
            "--out" => {
                let value = args.next().ok_or("missing absolute path after --out")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out supplied more than once".into());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage()).into()),
        }
    }
    if current_evidence == inventory.is_some() {
        return Err(format!(
            "supply exactly one of --current-evidence or --input; {}",
            usage()
        )
        .into());
    }
    let input_mode = if current_evidence {
        InputMode::CurrentEvidence
    } else {
        let path = inventory.expect("checked inventory presence");
        if !path.is_absolute() {
            return Err("--input must be absolute".into());
        }
        InputMode::Inventory(path)
    };
    let out = out.ok_or("missing --out")?;
    if !out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    Ok(Arguments { input_mode, out })
}

fn main() {
    let result = (|| -> Result<(), Box<dyn Error>> {
        let args = parse_args()?;
        let input = match args.input_mode {
            InputMode::CurrentEvidence => current_evidence_input(),
            InputMode::Inventory(path) => serde_json::from_slice(&fs::read(path)?)?,
        };
        let report = evaluate(&input);
        write_report_atomic(&args.out, &report)?;
        if !report.moe_per_layer_device_readiness_earned {
            return Err(format!(
                "MoE per-layer device readiness is incomplete; report written to {}",
                args.out.display()
            )
            .into());
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ascension_qwen80_moe_per_layer_device_readiness_contract: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sha(seed: usize) -> String {
        format!("{:064x}", seed + 1)
    }

    fn same_input_witness(layer: usize) -> SameInputMoeWitness {
        let first_residual = test_sha(layer * 100 + 1);
        let postnorm_hidden = test_sha(layer * 100 + 2);
        let route_ids = (0..TOP_K)
            .map(|index| ((layer * TOP_K + index) % EXPERTS) as u16)
            .collect::<Vec<_>>();
        let normalized_route_weights = vec![0.1; TOP_K];
        let route_waves = (0..TOP_K)
            .map(|index| RouteWaveWitness {
                wave_index: index,
                expert_id: route_ids[index],
                normalized_weight: normalized_route_weights[index],
                input_postnorm_hidden_sha256: postnorm_hidden.clone(),
                output_weighted_route_sha256: test_sha(layer * 1_000 + index + 1),
                gate_up_device_parity_passed: true,
                activation_device_parity_passed: true,
                down_device_parity_passed: true,
                source_weight_applied_after_down: true,
            })
            .collect::<Vec<_>>();
        SameInputMoeWitness {
            capture_id_sha256: test_sha(layer * 100 + 3),
            first_residual_sha256: first_residual.clone(),
            postnorm_input_sha256: first_residual.clone(),
            postnorm_hidden_sha256: postnorm_hidden.clone(),
            router_input_sha256: postnorm_hidden.clone(),
            routed_input_sha256: postnorm_hidden.clone(),
            shared_input_sha256: postnorm_hidden.clone(),
            combine_first_residual_sha256: first_residual,
            combine_postnorm_hidden_sha256: postnorm_hidden,
            combine_route_output_sha256: route_waves
                .iter()
                .map(|wave| wave.output_weighted_route_sha256.clone())
                .collect(),
            combine_gated_shared_sha256: test_sha(layer * 100 + 4),
            gated_shared_sha256: test_sha(layer * 100 + 4),
            second_residual_sha256: test_sha(layer * 100 + 5),
            route_ids,
            normalized_route_weights,
            route_waves,
            device_buffers_retained_until_capture_fence: true,
            one_same_input_command_graph_capture: true,
            source_router_top10_parity_passed: true,
            postnorm_device_parity_passed: true,
            router_device_parity_passed: true,
            top10_device_parity_passed: true,
            all_ten_route_bodies_device_parity_passed: true,
            shared_expert_device_parity_passed: true,
            combine_device_parity_passed: true,
            second_residual_device_parity_passed: true,
        }
    }

    fn fully_physical_input() -> ReadinessInput {
        let mut input = current_evidence_input();
        input.per_layer_physical_evidence = expected_moe_layers()
            .into_iter()
            .map(|layer| MoeLayerPhysicalEvidence {
                schema: LAYER_EVIDENCE_SCHEMA.into(),
                layer,
                receipt_seal_sha256: test_sha(layer),
                source_tensor_prefix: format!("model.layers.{layer}"),
                source_identity: SourceIdentity::exact(),
                geometry: MoeGeometry::exact(),
                ordered_operations: expected_operation_order(),
                backend: "metal".into(),
                device_dispatches: 42,
                source_bound: true,
                artifact_bound: true,
                full_moe_boundary_path: true,
                device_parity_passed: true,
                fixture_only: false,
                synthetic_input: false,
                component_only: false,
                fallback_used: false,
                same_input_witness: same_input_witness(layer),
            })
            .collect();
        input
    }

    #[test]
    fn source_schedule_has_exactly_all_forty_eight_moe_layers() {
        let layers = expected_moe_layers();
        assert_eq!(layers.len(), 48);
        assert_eq!(layers.first(), Some(&0));
        assert_eq!(layers.last(), Some(&47));
        validate_moe_layers(&layers).unwrap();
    }

    #[test]
    fn current_l0_cpu_shared_fixture_combine_fragments_remain_incomplete_and_nonjoinable() {
        let report = evaluate(&current_evidence_input());
        assert!(!report.moe_per_layer_device_readiness_earned);
        assert_eq!(
            report.status,
            "INCOMPLETE_QWEN80_MOE_PER_LAYER_DEVICE_READINESS_CURRENT_L0_CPU_SHARED_FIXTURE_COMBINE_FRAGMENTS_ARE_NOT_JOINABLE"
        );
        assert!(report.current_layer0_fragments_valid);
        assert!(!report.current_layer0_fragments_joinable);
        assert_eq!(report.missing_or_invalid_moe_layers.len(), 48);
        assert!(!report.current_layer0_nonjoinable_reasons.is_empty());
        assert!(!report.complete_decoder_readiness_earned);
    }

    #[test]
    fn rejects_broken_same_input_join_or_fixture_record() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0]
            .same_input_witness
            .shared_input_sha256 = test_sha(9_999);
        let report = evaluate(&input);
        assert!(!report.moe_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 0)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("one same-input provenance chain")));
    }

    #[test]
    fn rejects_duplicate_or_misordered_top_ten_route_witnesses() {
        let mut input = fully_physical_input();
        let first_route_id = input.per_layer_physical_evidence[2]
            .same_input_witness
            .route_ids[0];
        input.per_layer_physical_evidence[2]
            .same_input_witness
            .route_ids[1] = first_route_id;
        let report = evaluate(&input);
        assert!(!report.moe_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 2)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("duplicate or out of range")));
    }

    #[test]
    fn rejects_geometry_or_operator_order_drift() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0]
            .geometry
            .shared_scalar_gate_shape = vec![2, HIDDEN];
        input.per_layer_physical_evidence[0]
            .ordered_operations
            .swap(0, 1);
        let report = evaluate(&input);
        assert!(!report.moe_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 0)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("geometry drifted")));
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("order drifted")));
    }

    #[test]
    fn rejects_unsealed_physical_receipt_or_missing_layer() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0].receipt_seal_sha256 = "0".repeat(64);
        input
            .per_layer_physical_evidence
            .retain(|record| record.layer != 47);
        let report = evaluate(&input);
        assert!(!report.moe_per_layer_device_readiness_earned);
        assert!(report.missing_or_invalid_moe_layers.contains(&47));
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 0)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("sealed physical receipt")));
    }

    #[test]
    fn exact_hypothetical_all_layer_physical_ledger_satisfies_only_this_narrow_frontier() {
        let report = evaluate(&fully_physical_input());
        assert!(report.moe_per_layer_device_readiness_earned);
        assert!(report
            .per_layer_assessments
            .iter()
            .all(|assessment| assessment.satisfied));
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
    }
}
