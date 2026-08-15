//! CPU-only Qwen3-Coder-Next per-layer GQA device-readiness contract.
//!
//! This standalone evidence ledger binds the known sealed two-token layer-3
//! GQA component to the source-selected twelve GQA layers, their independent
//! KV slots, and the future per-session state-buffer layout. It validates
//! metadata only. It never opens a model artifact, allocates a device buffer,
//! creates Metal work, contacts a watcher/server/HCLI endpoint, or measures
//! TPS/TG.
//!
//! A valid layer-3 component receipt remains a fixture-scoped component. It
//! cannot become a GQA-per-layer result until every source-selected GQA layer
//! has its own sealed source/artifact/device-parity record and a physical,
//! causal KV-cache witness bound to the exact per-session layout.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_gqa_per_layer_device_readiness_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_gqa_per_layer_device_readiness_result.v1";
const LAYER_EVIDENCE_SCHEMA: &str =
    "hawking.ascension.qwen80_gqa_per_layer_physical_device_ledger.v1";
const LAYER3_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_layer3_gqa_attention_stage.v1";
const STATE_LAYOUT_SCHEMA: &str = "hawking.ascension.qwen80_device_state_buffer_layout_contract.v1";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const LAYER3_COMPONENT_RECEIPT_SEAL: &str =
    "b0b16468df4b42ae8b02de076de72850bdd244f9dd488be9432f8e61bbe0a44e";
const LAYER3_COMPONENT_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER3_GQA_TWO_TOKEN_COMPONENT_STAGE_NOT_COMPLETE_LAYER_OR_TOKEN";
const MAX_NATIVE_CONTEXT: usize = 4_096;
const LAYER_COUNT: usize = 48;
const DELTANET_LAYERS: usize = 36;
const GQA_LAYERS: usize = 12;
const HIDDEN: usize = 2_048;
const DELTANET_CONV_CHANNELS: usize = 8_192;
const DELTANET_CONV_HISTORY_TOKENS: usize = 3;
const DELTANET_VALUE_HEADS: usize = 32;
const DELTANET_KEY_HEAD_DIM: usize = 128;
const DELTANET_VALUE_HEAD_DIM: usize = 128;
const QUERY_HEADS: usize = 16;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 256;
const QUERIES_PER_KV_HEAD: usize = QUERY_HEADS / KV_HEADS;
const QUERY_DIM: usize = QUERY_HEADS * HEAD_DIM;
const KV_DIM: usize = KV_HEADS * HEAD_DIM;
const Q_PROJ_ROWS: usize = QUERY_DIM * 2;
const GROUP_SIZE: usize = 128;
const PARTIAL_ROTARY_DIMENSIONS: usize = 64;
const ROPE_THETA: f64 = 5_000_000.0;
const RMS_NORM_EPSILON: f64 = 1.0e-6;
const F32_BYTES: usize = 4;

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
enum GqaOperation {
    QProjection,
    KProjection,
    VProjection,
    ExtractQueryRowsFromInterleavedQProjection,
    QueryRmsNorm,
    QueryPartialRope,
    KeyRmsNorm,
    KeyPartialRope,
    AppendKeyAtCurrentPosition,
    AppendValueAtCurrentPosition,
    CausalReadPositionsZeroThroughCurrent,
    QueryProjectionSigmoidGate,
    OProjection,
}

fn expected_operation_order() -> Vec<GqaOperation> {
    vec![
        GqaOperation::QProjection,
        GqaOperation::KProjection,
        GqaOperation::VProjection,
        GqaOperation::ExtractQueryRowsFromInterleavedQProjection,
        GqaOperation::QueryRmsNorm,
        GqaOperation::QueryPartialRope,
        GqaOperation::KeyRmsNorm,
        GqaOperation::KeyPartialRope,
        GqaOperation::AppendKeyAtCurrentPosition,
        GqaOperation::AppendValueAtCurrentPosition,
        GqaOperation::CausalReadPositionsZeroThroughCurrent,
        GqaOperation::QueryProjectionSigmoidGate,
        GqaOperation::OProjection,
    ]
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct GqaGeometry {
    hidden_size: usize,
    query_heads: usize,
    kv_heads: usize,
    queries_per_kv_head: usize,
    head_dim: usize,
    q_proj_shape: Vec<usize>,
    k_proj_shape: Vec<usize>,
    v_proj_shape: Vec<usize>,
    o_proj_shape: Vec<usize>,
    qk_norm_shape: Vec<usize>,
    packed_group_size: usize,
    partial_rotary_dimensions: usize,
    rope_theta: f64,
    rms_norm_epsilon: f64,
    q_norm_residual_scale_one_plus_weight: bool,
    q_proj_gate_is_head_local_sigmoid_second_256_rows: bool,
}

impl GqaGeometry {
    fn exact() -> Self {
        Self {
            hidden_size: HIDDEN,
            query_heads: QUERY_HEADS,
            kv_heads: KV_HEADS,
            queries_per_kv_head: QUERIES_PER_KV_HEAD,
            head_dim: HEAD_DIM,
            q_proj_shape: vec![Q_PROJ_ROWS, HIDDEN],
            k_proj_shape: vec![KV_DIM, HIDDEN],
            v_proj_shape: vec![KV_DIM, HIDDEN],
            o_proj_shape: vec![HIDDEN, QUERY_DIM],
            qk_norm_shape: vec![HEAD_DIM],
            packed_group_size: GROUP_SIZE,
            partial_rotary_dimensions: PARTIAL_ROTARY_DIMENSIONS,
            rope_theta: ROPE_THETA,
            rms_norm_epsilon: RMS_NORM_EPSILON,
            q_norm_residual_scale_one_plus_weight: true,
            q_proj_gate_is_head_local_sigmoid_second_256_rows: true,
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self.hidden_size != HIDDEN
            || self.query_heads != QUERY_HEADS
            || self.kv_heads != KV_HEADS
            || self.queries_per_kv_head != QUERIES_PER_KV_HEAD
            || self.head_dim != HEAD_DIM
            || self.q_proj_shape != [Q_PROJ_ROWS, HIDDEN]
            || self.k_proj_shape != [KV_DIM, HIDDEN]
            || self.v_proj_shape != [KV_DIM, HIDDEN]
            || self.o_proj_shape != [HIDDEN, QUERY_DIM]
            || self.qk_norm_shape != [HEAD_DIM]
            || self.packed_group_size != GROUP_SIZE
            || self.partial_rotary_dimensions != PARTIAL_ROTARY_DIMENSIONS
            || self.rope_theta.to_bits() != ROPE_THETA.to_bits()
            || self.rms_norm_epsilon.to_bits() != RMS_NORM_EPSILON.to_bits()
            || !self.q_norm_residual_scale_one_plus_weight
            || !self.q_proj_gate_is_head_local_sigmoid_second_256_rows
        {
            return Err(format!(
                "{label} Q/K/V, norm, RoPE, or gate geometry drifted"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct GqaScheduleEntry {
    layer: usize,
    slot: usize,
}

fn expected_gqa_schedule() -> Vec<GqaScheduleEntry> {
    (0..GQA_LAYERS)
        .map(|slot| GqaScheduleEntry {
            layer: slot * 4 + 3,
            slot,
        })
        .collect()
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct StateBufferSummary {
    domain: String,
    slot_shape: Vec<usize>,
    slot_count: usize,
    slot_stride_elements: usize,
    active_capacity_elements: usize,
    active_capacity_bytes: usize,
    rollback_capacity_elements: usize,
    rollback_capacity_bytes: usize,
}

fn expected_state_buffer(
    domain: &str,
    slot_shape: Vec<usize>,
    slot_count: usize,
) -> StateBufferSummary {
    let stride = slot_shape.iter().product::<usize>();
    let capacity = slot_count * stride;
    StateBufferSummary {
        domain: domain.into(),
        slot_shape,
        slot_count,
        slot_stride_elements: stride,
        active_capacity_elements: capacity,
        active_capacity_bytes: capacity * F32_BYTES,
        rollback_capacity_elements: capacity,
        rollback_capacity_bytes: capacity * F32_BYTES,
    }
}

fn expected_state_buffers(max_seq_len: usize) -> Vec<StateBufferSummary> {
    vec![
        expected_state_buffer(
            "deltanet_conv_history",
            vec![DELTANET_CONV_CHANNELS, DELTANET_CONV_HISTORY_TOKENS],
            DELTANET_LAYERS,
        ),
        expected_state_buffer(
            "deltanet_recurrent",
            vec![
                DELTANET_VALUE_HEADS,
                DELTANET_KEY_HEAD_DIM,
                DELTANET_VALUE_HEAD_DIM,
            ],
            DELTANET_LAYERS,
        ),
        expected_state_buffer(
            "gqa_key_cache",
            vec![max_seq_len, KV_HEADS, HEAD_DIM],
            GQA_LAYERS,
        ),
        expected_state_buffer(
            "gqa_value_cache",
            vec![max_seq_len, KV_HEADS, HEAD_DIM],
            GQA_LAYERS,
        ),
    ]
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct StateBufferLayoutEvidence {
    schema: String,
    status: String,
    source_identity: SourceIdentity,
    max_seq_len: usize,
    schedule_layers: usize,
    deltanet_layers: usize,
    gqa_layers: usize,
    buffers: Vec<StateBufferSummary>,
    actual_device_allocation_performed: bool,
    actual_device_state_parity_performed: bool,
    actual_rollback_bytes_captured: bool,
}

impl StateBufferLayoutEvidence {
    fn metadata_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != STATE_LAYOUT_SCHEMA {
            errors.push("state-buffer layout schema is not the Qwen80 layout contract".into());
        }
        if self.status.trim().is_empty() {
            errors.push("state-buffer layout status is empty".into());
        }
        if let Err(error) = self.source_identity.validate_exact("state-buffer layout") {
            errors.push(error);
        }
        let max_seq_len_valid = self.max_seq_len > 0 && self.max_seq_len <= MAX_NATIVE_CONTEXT;
        if !max_seq_len_valid {
            errors.push(format!(
                "state-buffer max_seq_len={} is outside 1..={MAX_NATIVE_CONTEXT}",
                self.max_seq_len
            ));
        }
        if self.schedule_layers != LAYER_COUNT
            || self.deltanet_layers != DELTANET_LAYERS
            || self.gqa_layers != GQA_LAYERS
        {
            errors.push(
                "state-buffer layout does not bind the exact 48/36/12 source schedule".into(),
            );
        }
        if max_seq_len_valid {
            if self.buffers.len() != 4 {
                errors.push(
                    "state-buffer layout must contain all four domain buffer summaries".into(),
                );
            }
            for expected in expected_state_buffers(self.max_seq_len) {
                let domain = expected.domain.as_str();
                let found = self
                    .buffers
                    .iter()
                    .filter(|buffer| buffer.domain == domain)
                    .collect::<Vec<_>>();
                if found.len() != 1 {
                    errors.push(format!(
                        "state-buffer layout must contain exactly one {domain} buffer summary"
                    ));
                } else if *found[0] != expected {
                    errors.push(format!("{domain} geometry/offset stride/capacity drifted"));
                }
            }
        }
        errors
    }

    fn physically_ready(&self) -> bool {
        self.actual_device_allocation_performed
            && self.actual_device_state_parity_performed
            && self.actual_rollback_bytes_captured
    }

    fn gqa_stride_elements(&self) -> usize {
        self.max_seq_len * KV_HEADS * HEAD_DIM
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SealedLayer3GqaComponentEvidence {
    schema: String,
    receipt_seal_sha256: String,
    status: String,
    source_identity: SourceIdentity,
    layer: usize,
    fixture_positions: Vec<usize>,
    geometry: GqaGeometry,
    ordered_operations: Vec<GqaOperation>,
    backend: String,
    device_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
}

impl SealedLayer3GqaComponentEvidence {
    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != LAYER3_COMPONENT_SCHEMA {
            errors.push("layer-3 component schema drifted".into());
        }
        if self.receipt_seal_sha256 != LAYER3_COMPONENT_RECEIPT_SEAL {
            errors
                .push("layer-3 component does not bind the valid sealed two-token receipt".into());
        }
        if self.status != LAYER3_COMPONENT_STATUS {
            errors.push(
                "layer-3 component status is not the admitted two-token GQA component status"
                    .into(),
            );
        }
        if let Err(error) = self.source_identity.validate_exact("layer-3 component") {
            errors.push(error);
        }
        if self.layer != 3 || self.fixture_positions != [0, 1] {
            errors.push(
                "layer-3 component does not bind source layer 3 with fixture positions [0, 1]"
                    .into(),
            );
        }
        if let Err(error) = self.geometry.validate_exact("layer-3 component") {
            errors.push(error);
        }
        if self.ordered_operations != expected_operation_order() {
            errors.push(
                "layer-3 component Q/K/V, norm, RoPE, cache, gate, and O order drifted".into(),
            );
        }
        if self.backend != "metal"
            || self.device_dispatches != 14
            || !self.source_bound
            || !self.artifact_bound
            || !self.device_parity_passed
        {
            errors.push(
                "layer-3 component loses its sealed source/artifact/Metal parity binding".into(),
            );
        }
        if !self.fixture_only || !self.synthetic_input || !self.component_only {
            errors.push("layer-3 component boundary drifted; it must remain a synthetic two-token component".into());
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GqaCacheWitness {
    session_id: String,
    max_seq_len: usize,
    position: usize,
    key_slot: usize,
    value_slot: usize,
    key_allocation_id: String,
    value_allocation_id: String,
    key_offset_elements: usize,
    value_offset_elements: usize,
    key_capacity_elements: usize,
    value_capacity_elements: usize,
    key_shape: Vec<usize>,
    value_shape: Vec<usize>,
    device_key_buffer_allocated: bool,
    device_value_buffer_allocated: bool,
    key_append_before_causal_read: bool,
    value_append_before_causal_read: bool,
    causal_read_first_position: usize,
    causal_read_last_position: usize,
    current_position_included: bool,
    future_positions_excluded: bool,
    slot_isolated_from_other_gqa_layers: bool,
    session_isolated_from_other_sessions: bool,
    rollback_key_and_value_bytes_captured: bool,
    state_bytes_device_parity_passed: bool,
}

impl GqaCacheWitness {
    fn physical_errors(
        &self,
        layer: usize,
        slot: usize,
        layout: &StateBufferLayoutEvidence,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.session_id.trim().is_empty() {
            errors.push(format!(
                "layer {layer} cache witness has an empty logical session id"
            ));
        }
        if layout.max_seq_len == 0 || layout.max_seq_len > MAX_NATIVE_CONTEXT {
            errors.push(format!(
                "layer {layer} cache witness cannot bind an invalid state-layout max_seq_len"
            ));
            return errors;
        }
        if self.max_seq_len != layout.max_seq_len || self.position >= layout.max_seq_len {
            errors.push(format!(
                "layer {layer} cache witness position/capacity is outside the bound layout"
            ));
        }
        if self.key_slot != slot || self.value_slot != slot {
            errors.push(format!(
                "layer {layer} cache witness does not own GQA slot {slot}"
            ));
        }
        let expected_shape = vec![layout.max_seq_len, KV_HEADS, HEAD_DIM];
        if self.key_shape != expected_shape || self.value_shape != expected_shape {
            errors.push(format!(
                "layer {layer} cache witness shape is not [max_seq_len,2,256]"
            ));
        }
        let stride = layout.gqa_stride_elements();
        let expected_offset = slot * stride;
        let expected_capacity = expected_offset + stride;
        if self.key_offset_elements != expected_offset
            || self.value_offset_elements != expected_offset
            || self.key_capacity_elements != expected_capacity
            || self.value_capacity_elements != expected_capacity
        {
            errors.push(format!(
                "layer {layer} cache witness offset/capacity does not match slot {slot}"
            ));
        }
        let key_id = format!(
            "qwen80/session={}/arena=active/domain=gqa_key_cache",
            self.session_id
        );
        let value_id = format!(
            "qwen80/session={}/arena=active/domain=gqa_value_cache",
            self.session_id
        );
        if self.key_allocation_id != key_id || self.value_allocation_id != value_id {
            errors.push(format!(
                "layer {layer} cache witness is not bound to the state-layout active K/V allocation ids"
            ));
        }
        if !self.device_key_buffer_allocated
            || !self.device_value_buffer_allocated
            || !self.key_append_before_causal_read
            || !self.value_append_before_causal_read
            || self.causal_read_first_position != 0
            || self.causal_read_last_position != self.position
            || !self.current_position_included
            || !self.future_positions_excluded
            || !self.slot_isolated_from_other_gqa_layers
            || !self.session_isolated_from_other_sessions
            || !self.rollback_key_and_value_bytes_captured
            || !self.state_bytes_device_parity_passed
        {
            errors.push(format!(
                "layer {layer} cache witness lacks physical allocation, causal append/read, isolation, rollback, or state-parity proof"
            ));
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GqaLayerPhysicalEvidence {
    schema: String,
    layer: usize,
    slot: usize,
    receipt_seal_sha256: String,
    component_ancestor_receipt_seal_sha256: Option<String>,
    source_identity: SourceIdentity,
    geometry: GqaGeometry,
    ordered_operations: Vec<GqaOperation>,
    backend: String,
    device_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    full_layer_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
    fallback_used: bool,
    cache_witness: GqaCacheWitness,
}

impl GqaLayerPhysicalEvidence {
    fn validation_errors(
        &self,
        expected: &GqaScheduleEntry,
        layout: &StateBufferLayoutEvidence,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != LAYER_EVIDENCE_SCHEMA {
            errors.push(format!(
                "layer {} physical ledger schema drifted",
                expected.layer
            ));
        }
        if self.layer != expected.layer || self.slot != expected.slot {
            errors.push(format!(
                "physical evidence maps layer {} / slot {} instead of source GQA layer {} / slot {}",
                self.layer, self.slot, expected.layer, expected.slot
            ));
        }
        if !is_lower_sha256(&self.receipt_seal_sha256) {
            errors.push(format!(
                "layer {} has no valid sealed physical receipt digest",
                expected.layer
            ));
        }
        if expected.layer == 3
            && self.component_ancestor_receipt_seal_sha256.as_deref()
                != Some(LAYER3_COMPONENT_RECEIPT_SEAL)
        {
            errors.push("layer 3 physical evidence does not retain the sealed two-token component as an ancestor".into());
        }
        if let Err(error) = self
            .source_identity
            .validate_exact(&format!("layer {}", expected.layer))
        {
            errors.push(error);
        }
        if let Err(error) = self
            .geometry
            .validate_exact(&format!("layer {}", expected.layer))
        {
            errors.push(error);
        }
        if self.ordered_operations != expected_operation_order() {
            errors.push(format!(
                "layer {} Q/K/V, norm, RoPE, cache, gate, and O operation order drifted",
                expected.layer
            ));
        }
        if self.backend != "metal"
            || self.device_dispatches == 0
            || !self.source_bound
            || !self.artifact_bound
            || !self.full_layer_path
            || !self.device_parity_passed
            || self.fixture_only
            || self.synthetic_input
            || self.component_only
            || self.fallback_used
        {
            errors.push(format!(
                "layer {} lacks a non-fixture, source/artifact-bound, full-layer Metal parity record",
                expected.layer
            ));
        }
        errors.extend(
            self.cache_witness
                .physical_errors(expected.layer, expected.slot, layout),
        );
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReadinessInput {
    schema: String,
    source_identity: SourceIdentity,
    gqa_schedule: Vec<GqaScheduleEntry>,
    sealed_layer3_component: SealedLayer3GqaComponentEvidence,
    state_buffer_layout: StateBufferLayoutEvidence,
    per_layer_physical_evidence: Vec<GqaLayerPhysicalEvidence>,
}

#[derive(Serialize)]
struct LayerAssessment {
    layer: usize,
    slot: usize,
    supplied_records: usize,
    receipt_seals: Vec<String>,
    satisfied: bool,
    failures: Vec<String>,
}

#[derive(Serialize)]
struct ReadinessReport {
    schema: &'static str,
    status: &'static str,
    gqa_per_layer_device_readiness_earned: bool,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    input_schema_valid: bool,
    source_identity_valid: bool,
    exact_12_gqa_schedule_valid: bool,
    valid_sealed_two_token_layer3_component_bound: bool,
    layer3_component_validation_errors: Vec<String>,
    state_buffer_layout_metadata_valid: bool,
    state_buffer_layout_physical_ready: bool,
    state_buffer_layout_errors: Vec<String>,
    per_layer_assessments: Vec<LayerAssessment>,
    missing_or_invalid_gqa_layers: Vec<usize>,
    contract_errors: Vec<String>,
    expected_gqa_schedule: Vec<GqaScheduleEntry>,
    sealed_layer3_component_is_fixture_component_only: bool,
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

fn validate_schedule(schedule: &[GqaScheduleEntry]) -> Result<(), String> {
    if schedule != expected_gqa_schedule() {
        return Err("GQA schedule must be exactly layers 3,7,...,47 with slots 0..11".into());
    }
    let layers = schedule
        .iter()
        .map(|entry| entry.layer)
        .collect::<BTreeSet<_>>();
    let slots = schedule
        .iter()
        .map(|entry| entry.slot)
        .collect::<BTreeSet<_>>();
    if layers.len() != GQA_LAYERS || slots.len() != GQA_LAYERS {
        return Err("GQA schedule contains a duplicate layer or slot".into());
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
    let exact_12_gqa_schedule_valid = match validate_schedule(&input.gqa_schedule) {
        Ok(()) => true,
        Err(error) => {
            contract_errors.push(error);
            false
        }
    };
    let layer3_component_validation_errors = input.sealed_layer3_component.validation_errors();
    let valid_sealed_two_token_layer3_component_bound =
        layer3_component_validation_errors.is_empty();
    let state_buffer_layout_errors = input.state_buffer_layout.metadata_errors();
    let state_buffer_layout_metadata_valid = state_buffer_layout_errors.is_empty();
    let state_buffer_layout_physical_ready =
        state_buffer_layout_metadata_valid && input.state_buffer_layout.physically_ready();

    let expected_schedule = expected_gqa_schedule();
    let mut records_by_layer = BTreeMap::<usize, Vec<&GqaLayerPhysicalEvidence>>::new();
    for record in &input.per_layer_physical_evidence {
        records_by_layer
            .entry(record.layer)
            .or_default()
            .push(record);
    }
    for layer in records_by_layer.keys() {
        if !expected_schedule.iter().any(|entry| entry.layer == *layer) {
            contract_errors.push(format!(
                "physical evidence covers non-GQA source layer {layer}"
            ));
        }
    }

    let mut per_layer_assessments = Vec::with_capacity(GQA_LAYERS);
    let mut missing_or_invalid_gqa_layers = Vec::new();
    for expected in &expected_schedule {
        let records = records_by_layer
            .get(&expected.layer)
            .cloned()
            .unwrap_or_default();
        let mut failures = Vec::new();
        if records.len() != 1 {
            failures.push(format!(
                "expected exactly one sealed physical record for layer {}, found {}",
                expected.layer,
                records.len()
            ));
        }
        if let Some(record) = records.first() {
            failures.extend(record.validation_errors(expected, &input.state_buffer_layout));
        }
        let satisfied = failures.is_empty();
        if !satisfied {
            missing_or_invalid_gqa_layers.push(expected.layer);
        }
        per_layer_assessments.push(LayerAssessment {
            layer: expected.layer,
            slot: expected.slot,
            supplied_records: records.len(),
            receipt_seals: records
                .iter()
                .map(|record| record.receipt_seal_sha256.clone())
                .collect(),
            satisfied,
            failures,
        });
    }

    let gqa_per_layer_device_readiness_earned = input_schema_valid
        && source_identity_valid
        && exact_12_gqa_schedule_valid
        && valid_sealed_two_token_layer3_component_bound
        && state_buffer_layout_physical_ready
        && contract_errors.is_empty()
        && missing_or_invalid_gqa_layers.is_empty();
    let status = if gqa_per_layer_device_readiness_earned {
        "READY_FOR_QWEN80_GQA_PER_LAYER_DEVICE_INTEGRATION_NOT_A_COMPLETE_DECODER"
    } else {
        "INCOMPLETE_QWEN80_GQA_PER_LAYER_DEVICE_READINESS_LAYER3_TWO_TOKEN_COMPONENT_DOES_NOT_COVER_ALL_12_GQA_LAYERS"
    };
    let mut report = ReadinessReport {
        schema: RESULT_SCHEMA,
        status,
        gqa_per_layer_device_readiness_earned,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        input_schema_valid,
        source_identity_valid,
        exact_12_gqa_schedule_valid,
        valid_sealed_two_token_layer3_component_bound,
        layer3_component_validation_errors,
        state_buffer_layout_metadata_valid,
        state_buffer_layout_physical_ready,
        state_buffer_layout_errors,
        per_layer_assessments,
        missing_or_invalid_gqa_layers,
        contract_errors,
        expected_gqa_schedule: expected_schedule,
        sealed_layer3_component_is_fixture_component_only: input.sealed_layer3_component.component_only,
        read_only_contract: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        required_before_ready: vec![
            "Produce one sealed source/artifact-bound, non-fixture full-layer Metal parity ledger for each GQA layer 3,7,...,47, with no fallback.",
            "For each ledger, prove exact Q/K/V/O and q/k norm geometry, partial RoPE, source q_proj gate, and the declared operation order.",
            "Allocate the exact per-session active and rollback GQA K/V state ranges, then prove their physical identities, offsets, capacities, and cross-layer/session isolation.",
            "Capture cache bytes and hidden-output CPU/device parity over more than the two-token fixture, including current-position inclusion, future-position exclusion, and rollback restore.",
            "After this narrow GQA frontier is satisfied, independently satisfy the remaining DeltaNet, MoE, terminal, full-graph, tokenizer, and clean TPS/TG gates before any decoder/server claim.",
        ],
        claim_boundary: vec![
            "The known layer-3 receipt is valid sealed two-token component evidence, but remains fixture/synthetic/component scoped.",
            "This CPU-only contract does not open an artifact or device, and cannot itself allocate K/V state, execute a GQA layer, emit a token, expose HCLI, or measure TPS/TG.",
            "Even a ready GQA-per-layer result is not a complete decoder, resident server, capability, Agent OS, tournament, BASE_TRUE_TPS, or TG qualification.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&report).unwrap_or_default())
    );
    report
}

fn current_state_buffer_layout() -> StateBufferLayoutEvidence {
    let max_seq_len = MAX_NATIVE_CONTEXT;
    StateBufferLayoutEvidence {
        schema: STATE_LAYOUT_SCHEMA.into(),
        status:
            "NOT_READY_NO_DEVICE_ALLOCATION_NO_STATE_PARITY_NO_ROLLBACK_CAPTURE_QWEN80_PER_SESSION_BUFFER_LAYOUT_CONTRACT"
                .into(),
        source_identity: SourceIdentity::exact(),
        max_seq_len,
        schedule_layers: LAYER_COUNT,
        deltanet_layers: DELTANET_LAYERS,
        gqa_layers: GQA_LAYERS,
        buffers: expected_state_buffers(max_seq_len),
        actual_device_allocation_performed: false,
        actual_device_state_parity_performed: false,
        actual_rollback_bytes_captured: false,
    }
}

fn current_layer3_component() -> SealedLayer3GqaComponentEvidence {
    SealedLayer3GqaComponentEvidence {
        schema: LAYER3_COMPONENT_SCHEMA.into(),
        receipt_seal_sha256: LAYER3_COMPONENT_RECEIPT_SEAL.into(),
        status: LAYER3_COMPONENT_STATUS.into(),
        source_identity: SourceIdentity::exact(),
        layer: 3,
        fixture_positions: vec![0, 1],
        geometry: GqaGeometry::exact(),
        ordered_operations: expected_operation_order(),
        backend: "metal".into(),
        device_dispatches: 14,
        source_bound: true,
        artifact_bound: true,
        device_parity_passed: true,
        fixture_only: true,
        synthetic_input: true,
        component_only: true,
    }
}

fn cache_witness(
    session_id: &str,
    max_seq_len: usize,
    position: usize,
    slot: usize,
    physical: bool,
) -> GqaCacheWitness {
    let stride = max_seq_len * KV_HEADS * HEAD_DIM;
    let offset = slot * stride;
    let capacity = offset + stride;
    GqaCacheWitness {
        session_id: session_id.into(),
        max_seq_len,
        position,
        key_slot: slot,
        value_slot: slot,
        key_allocation_id: format!("qwen80/session={session_id}/arena=active/domain=gqa_key_cache"),
        value_allocation_id: format!(
            "qwen80/session={session_id}/arena=active/domain=gqa_value_cache"
        ),
        key_offset_elements: offset,
        value_offset_elements: offset,
        key_capacity_elements: capacity,
        value_capacity_elements: capacity,
        key_shape: vec![max_seq_len, KV_HEADS, HEAD_DIM],
        value_shape: vec![max_seq_len, KV_HEADS, HEAD_DIM],
        device_key_buffer_allocated: physical,
        device_value_buffer_allocated: physical,
        key_append_before_causal_read: true,
        value_append_before_causal_read: true,
        causal_read_first_position: 0,
        causal_read_last_position: position,
        current_position_included: true,
        future_positions_excluded: true,
        slot_isolated_from_other_gqa_layers: physical,
        session_isolated_from_other_sessions: physical,
        rollback_key_and_value_bytes_captured: physical,
        state_bytes_device_parity_passed: physical,
    }
}

fn current_layer3_physical_evidence() -> GqaLayerPhysicalEvidence {
    let layout = current_state_buffer_layout();
    GqaLayerPhysicalEvidence {
        schema: LAYER_EVIDENCE_SCHEMA.into(),
        layer: 3,
        slot: 0,
        receipt_seal_sha256: LAYER3_COMPONENT_RECEIPT_SEAL.into(),
        component_ancestor_receipt_seal_sha256: Some(LAYER3_COMPONENT_RECEIPT_SEAL.into()),
        source_identity: SourceIdentity::exact(),
        geometry: GqaGeometry::exact(),
        ordered_operations: expected_operation_order(),
        backend: "metal".into(),
        device_dispatches: 14,
        source_bound: true,
        artifact_bound: true,
        full_layer_path: false,
        device_parity_passed: true,
        fixture_only: true,
        synthetic_input: true,
        component_only: true,
        fallback_used: false,
        cache_witness: cache_witness(
            "layer3-two-token-component",
            layout.max_seq_len,
            1,
            0,
            false,
        ),
    }
}

fn current_evidence_input() -> ReadinessInput {
    ReadinessInput {
        schema: INPUT_SCHEMA.into(),
        source_identity: SourceIdentity::exact(),
        gqa_schedule: expected_gqa_schedule(),
        sealed_layer3_component: current_layer3_component(),
        state_buffer_layout: current_state_buffer_layout(),
        per_layer_physical_evidence: vec![current_layer3_physical_evidence()],
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
    "usage: ascension_qwen80_gqa_per_layer_device_readiness_contract \
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
        if !report.gqa_per_layer_device_readiness_earned {
            return Err(format!(
                "GQA per-layer device readiness is incomplete; report written to {}",
                args.out.display()
            )
            .into());
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ascension_qwen80_gqa_per_layer_device_readiness_contract: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fully_physical_input() -> ReadinessInput {
        let mut input = current_evidence_input();
        input.state_buffer_layout.status =
            "SEALED_QWEN80_PER_SESSION_STATE_BUFFER_PHYSICAL_CAPTURE".into();
        input.state_buffer_layout.actual_device_allocation_performed = true;
        input
            .state_buffer_layout
            .actual_device_state_parity_performed = true;
        input.state_buffer_layout.actual_rollback_bytes_captured = true;
        input.per_layer_physical_evidence = expected_gqa_schedule()
            .into_iter()
            .map(|entry| GqaLayerPhysicalEvidence {
                schema: LAYER_EVIDENCE_SCHEMA.into(),
                layer: entry.layer,
                slot: entry.slot,
                receipt_seal_sha256: format!("{:064x}", entry.layer + 1),
                component_ancestor_receipt_seal_sha256: (entry.layer == 3)
                    .then(|| LAYER3_COMPONENT_RECEIPT_SEAL.into()),
                source_identity: SourceIdentity::exact(),
                geometry: GqaGeometry::exact(),
                ordered_operations: expected_operation_order(),
                backend: "metal".into(),
                device_dispatches: 21,
                source_bound: true,
                artifact_bound: true,
                full_layer_path: true,
                device_parity_passed: true,
                fixture_only: false,
                synthetic_input: false,
                component_only: false,
                fallback_used: false,
                cache_witness: cache_witness(
                    "physical-session",
                    MAX_NATIVE_CONTEXT,
                    1,
                    entry.slot,
                    true,
                ),
            })
            .collect();
        input
    }

    #[test]
    fn source_schedule_has_exactly_twelve_gqa_layers_and_slots() {
        let schedule = expected_gqa_schedule();
        assert_eq!(schedule.len(), 12);
        assert_eq!(
            schedule.first(),
            Some(&GqaScheduleEntry { layer: 3, slot: 0 })
        );
        assert_eq!(
            schedule.last(),
            Some(&GqaScheduleEntry {
                layer: 47,
                slot: 11
            })
        );
        validate_schedule(&schedule).unwrap();
    }

    #[test]
    fn current_valid_layer3_component_remains_incomplete() {
        let report = evaluate(&current_evidence_input());
        assert!(!report.gqa_per_layer_device_readiness_earned);
        assert_eq!(
            report.status,
            "INCOMPLETE_QWEN80_GQA_PER_LAYER_DEVICE_READINESS_LAYER3_TWO_TOKEN_COMPONENT_DOES_NOT_COVER_ALL_12_GQA_LAYERS"
        );
        assert!(report.valid_sealed_two_token_layer3_component_bound);
        assert!(!report.state_buffer_layout_physical_ready);
        assert_eq!(report.per_layer_assessments.len(), 12);
        assert_eq!(report.missing_or_invalid_gqa_layers.len(), 12);
        assert!(!report.complete_decoder_readiness_earned);
    }

    #[test]
    fn rejects_wrong_gqa_slot_and_cache_append_order() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[1].slot = 7;
        input.per_layer_physical_evidence[1]
            .cache_witness
            .key_append_before_causal_read = false;
        let report = evaluate(&input);
        assert!(!report.gqa_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 7)
            .unwrap();
        assert!(!assessment.satisfied);
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("source GQA layer")));
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("causal append/read")));
    }

    #[test]
    fn rejects_geometry_rope_or_norm_drift() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0].geometry.rope_theta = 10_000.0;
        let report = evaluate(&input);
        assert!(!report.gqa_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 3)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("Q/K/V, norm, RoPE, or gate geometry")));
    }

    #[test]
    fn rejects_an_unsealed_or_placeholder_per_layer_receipt() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0].receipt_seal_sha256 = "0".repeat(64);
        let report = evaluate(&input);
        assert!(!report.gqa_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 3)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("sealed physical receipt")));
    }

    #[test]
    fn rejects_missing_one_of_the_twelve_physical_layer_ledgers() {
        let mut input = fully_physical_input();
        input
            .per_layer_physical_evidence
            .retain(|record| record.layer != 47);
        let report = evaluate(&input);
        assert!(!report.gqa_per_layer_device_readiness_earned);
        assert!(report.missing_or_invalid_gqa_layers.contains(&47));
    }

    #[test]
    fn rejects_unphysical_state_layout_even_with_all_layer_ledgers() {
        let mut input = fully_physical_input();
        input.state_buffer_layout.actual_device_allocation_performed = false;
        let report = evaluate(&input);
        assert!(!report.gqa_per_layer_device_readiness_earned);
        assert!(!report.state_buffer_layout_physical_ready);
    }

    #[test]
    fn rejects_state_layout_beyond_native_context_without_promoting_cache_evidence() {
        let mut input = fully_physical_input();
        input.state_buffer_layout.max_seq_len = MAX_NATIVE_CONTEXT + 1;
        let report = evaluate(&input);
        assert!(!report.gqa_per_layer_device_readiness_earned);
        assert!(!report.state_buffer_layout_metadata_valid);
        assert!(report
            .state_buffer_layout_errors
            .iter()
            .any(|error| error.contains("outside 1..=")));
    }

    #[test]
    fn exact_hypothetical_all_layer_physical_ledger_satisfies_only_this_narrow_frontier() {
        let report = evaluate(&fully_physical_input());
        assert!(report.gqa_per_layer_device_readiness_earned);
        assert!(report
            .per_layer_assessments
            .iter()
            .all(|assessment| assessment.satisfied));
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
    }
}
