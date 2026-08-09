//! CPU-only Qwen3-Coder-Next per-layer DeltaNet device-readiness contract.
//!
//! This standalone ledger binds the existing layer-0 direct-packed DeltaNet
//! mixer evidence to the exact 36 source-selected DeltaNet layers and their
//! session-local convolution/recurrent state slots. It validates evidence
//! metadata only; it does not open artifacts, allocate a device buffer, create
//! Metal work, contact a runtime/watcher/server/HCLI endpoint, or measure
//! TPS/TG.
//!
//! The known layer-0 record proves a bounded synthetic mixer component through
//! first residual. It is retained as an ancestor, not promoted. A per-layer
//! result needs a separate sealed, source/artifact-bound, non-fixture
//! full-DeltaNet-mixer device parity and state witness for every one of the 36
//! source-selected slots.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_deltanet_per_layer_device_readiness_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.qwen80_deltanet_per_layer_device_readiness_result.v1";
const LAYER_EVIDENCE_SCHEMA: &str =
    "hawking.ascension.qwen80_deltanet_per_layer_physical_device_ledger.v1";
const LAYER0_COMPONENT_SCHEMA: &str =
    "hawking.ascension.qwen80_layer0_deltanet_mixer_capture_receipt.v1";
const STATE_LAYOUT_SCHEMA: &str = "hawking.ascension.qwen80_device_state_buffer_layout_contract.v1";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const LAYER0_COMPONENT_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER0_DELTANET_MIXER_THROUGH_FIRST_RESIDUAL_NOT_COMPLETE_LAYER_OR_TOKEN";
const LAYER0_COMPONENT_LEDGER_ID: &str = "qwen80-layer0-deltanet-mixer-through-first-residual";
const MAX_NATIVE_CONTEXT: usize = 4_096;
const LAYER_COUNT: usize = 48;
const DELTANET_LAYERS: usize = 36;
const GQA_LAYERS: usize = 12;
const HIDDEN: usize = 2_048;
const GROUP_SIZE: usize = 128;
const RMS_NORM_EPSILON: f64 = 1.0e-6;
const KEY_HEADS: usize = 16;
const VALUE_HEADS: usize = 32;
const VALUES_PER_KEY_HEAD: usize = VALUE_HEADS / KEY_HEADS;
const KEY_HEAD_DIM: usize = 128;
const VALUE_HEAD_DIM: usize = 128;
const QKVZ_ROWS_PER_KEY_HEAD: usize = 768;
const QKVZ_ROWS: usize = KEY_HEADS * QKVZ_ROWS_PER_KEY_HEAD;
const QKVZ_QUERY_OFFSET_ROWS: usize = 0;
const QKVZ_KEY_OFFSET_ROWS: usize = 128;
const QKVZ_VALUE_OFFSET_ROWS: usize = 256;
const QKVZ_Z_OFFSET_ROWS: usize = 512;
const BA_ROWS_PER_KEY_HEAD: usize = 4;
const BA_ROWS: usize = KEY_HEADS * BA_ROWS_PER_KEY_HEAD;
const BA_BETA_OFFSET_ROWS: usize = 0;
const BA_DECAY_OFFSET_ROWS: usize = 2;
const CONV_CHANNELS: usize = 8_192;
const CONV_KERNEL: usize = 4;
const CONV_STATE_TOKENS: usize = CONV_KERNEL - 1;
const CONV_STATE_ELEMENTS: usize = CONV_CHANNELS * CONV_STATE_TOKENS;
const RECURRENT_STATE_ELEMENTS: usize = VALUE_HEADS * KEY_HEAD_DIM * VALUE_HEAD_DIM;
const VALUE_ELEMENTS: usize = VALUE_HEADS * VALUE_HEAD_DIM;
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
enum DeltaNetOperation {
    InputRmsNorm,
    QkvzProjection,
    BaProjection,
    InterleavedQkvzRearrange,
    CausalDepthwiseSiluConvolutionAndConvStateCommit,
    RepeatQueryAndKeyAndL2Normalize,
    BaALogDtBiasToDecayAndBeta,
    DeltaNetRecurrenceAndRecurrentStateCommit,
    ZGatedRmsNorm,
    OutProjection,
    FirstResidual,
}

fn expected_operation_order() -> Vec<DeltaNetOperation> {
    vec![
        DeltaNetOperation::InputRmsNorm,
        DeltaNetOperation::QkvzProjection,
        DeltaNetOperation::BaProjection,
        DeltaNetOperation::InterleavedQkvzRearrange,
        DeltaNetOperation::CausalDepthwiseSiluConvolutionAndConvStateCommit,
        DeltaNetOperation::RepeatQueryAndKeyAndL2Normalize,
        DeltaNetOperation::BaALogDtBiasToDecayAndBeta,
        DeltaNetOperation::DeltaNetRecurrenceAndRecurrentStateCommit,
        DeltaNetOperation::ZGatedRmsNorm,
        DeltaNetOperation::OutProjection,
        DeltaNetOperation::FirstResidual,
    ]
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct DeltaNetGeometry {
    hidden_size: usize,
    packed_group_size: usize,
    rms_norm_epsilon: f64,
    input_layernorm_shape: Vec<usize>,
    input_rms_norm_residual_scale_one_plus_weight: bool,
    key_heads: usize,
    value_heads: usize,
    values_per_key_head: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    qkvz_shape: Vec<usize>,
    qkvz_rows_per_key_head: usize,
    qkvz_query_offset_rows: usize,
    qkvz_key_offset_rows: usize,
    qkvz_value_offset_rows: usize,
    qkvz_z_offset_rows: usize,
    ba_shape: Vec<usize>,
    ba_rows_per_key_head: usize,
    ba_beta_offset_rows: usize,
    ba_decay_offset_rows: usize,
    causal_conv_shape: Vec<usize>,
    conv_state_shape: Vec<usize>,
    recurrent_state_shape: Vec<usize>,
    a_log_shape: Vec<usize>,
    dt_bias_shape: Vec<usize>,
    gated_rms_norm_shape: Vec<usize>,
    gated_rms_norm_residual_scale_one_plus_weight: bool,
    out_proj_shape: Vec<usize>,
}

impl DeltaNetGeometry {
    fn exact() -> Self {
        Self {
            hidden_size: HIDDEN,
            packed_group_size: GROUP_SIZE,
            rms_norm_epsilon: RMS_NORM_EPSILON,
            input_layernorm_shape: vec![HIDDEN],
            input_rms_norm_residual_scale_one_plus_weight: true,
            key_heads: KEY_HEADS,
            value_heads: VALUE_HEADS,
            values_per_key_head: VALUES_PER_KEY_HEAD,
            key_head_dim: KEY_HEAD_DIM,
            value_head_dim: VALUE_HEAD_DIM,
            qkvz_shape: vec![QKVZ_ROWS, HIDDEN],
            qkvz_rows_per_key_head: QKVZ_ROWS_PER_KEY_HEAD,
            qkvz_query_offset_rows: QKVZ_QUERY_OFFSET_ROWS,
            qkvz_key_offset_rows: QKVZ_KEY_OFFSET_ROWS,
            qkvz_value_offset_rows: QKVZ_VALUE_OFFSET_ROWS,
            qkvz_z_offset_rows: QKVZ_Z_OFFSET_ROWS,
            ba_shape: vec![BA_ROWS, HIDDEN],
            ba_rows_per_key_head: BA_ROWS_PER_KEY_HEAD,
            ba_beta_offset_rows: BA_BETA_OFFSET_ROWS,
            ba_decay_offset_rows: BA_DECAY_OFFSET_ROWS,
            causal_conv_shape: vec![CONV_CHANNELS, 1, CONV_KERNEL],
            conv_state_shape: vec![CONV_CHANNELS, CONV_STATE_TOKENS],
            recurrent_state_shape: vec![VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
            a_log_shape: vec![VALUE_HEADS],
            dt_bias_shape: vec![VALUE_HEADS],
            gated_rms_norm_shape: vec![VALUE_HEAD_DIM],
            gated_rms_norm_residual_scale_one_plus_weight: true,
            out_proj_shape: vec![HIDDEN, VALUE_ELEMENTS],
        }
    }

    fn validate_exact(&self, label: &str) -> Result<(), String> {
        if self.hidden_size != HIDDEN
            || self.packed_group_size != GROUP_SIZE
            || self.rms_norm_epsilon.to_bits() != RMS_NORM_EPSILON.to_bits()
            || self.input_layernorm_shape != [HIDDEN]
            || !self.input_rms_norm_residual_scale_one_plus_weight
            || self.key_heads != KEY_HEADS
            || self.value_heads != VALUE_HEADS
            || self.values_per_key_head != VALUES_PER_KEY_HEAD
            || self.key_head_dim != KEY_HEAD_DIM
            || self.value_head_dim != VALUE_HEAD_DIM
            || self.qkvz_shape != [QKVZ_ROWS, HIDDEN]
            || self.qkvz_rows_per_key_head != QKVZ_ROWS_PER_KEY_HEAD
            || self.qkvz_query_offset_rows != QKVZ_QUERY_OFFSET_ROWS
            || self.qkvz_key_offset_rows != QKVZ_KEY_OFFSET_ROWS
            || self.qkvz_value_offset_rows != QKVZ_VALUE_OFFSET_ROWS
            || self.qkvz_z_offset_rows != QKVZ_Z_OFFSET_ROWS
            || self.ba_shape != [BA_ROWS, HIDDEN]
            || self.ba_rows_per_key_head != BA_ROWS_PER_KEY_HEAD
            || self.ba_beta_offset_rows != BA_BETA_OFFSET_ROWS
            || self.ba_decay_offset_rows != BA_DECAY_OFFSET_ROWS
            || self.causal_conv_shape != [CONV_CHANNELS, 1, CONV_KERNEL]
            || self.conv_state_shape != [CONV_CHANNELS, CONV_STATE_TOKENS]
            || self.recurrent_state_shape != [VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM]
            || self.a_log_shape != [VALUE_HEADS]
            || self.dt_bias_shape != [VALUE_HEADS]
            || self.gated_rms_norm_shape != [VALUE_HEAD_DIM]
            || !self.gated_rms_norm_residual_scale_one_plus_weight
            || self.out_proj_shape != [HIDDEN, VALUE_ELEMENTS]
        {
            return Err(format!(
                "{label} QKVZ/BA/conv/recurrent/gated-norm/out-residual geometry drifted"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct DeltaNetScheduleEntry {
    layer: usize,
    slot: usize,
}

fn expected_deltanet_schedule() -> Vec<DeltaNetScheduleEntry> {
    (0..LAYER_COUNT)
        .filter(|layer| layer % 4 != 3)
        .enumerate()
        .map(|(slot, layer)| DeltaNetScheduleEntry { layer, slot })
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
            vec![CONV_CHANNELS, CONV_STATE_TOKENS],
            DELTANET_LAYERS,
        ),
        expected_state_buffer(
            "deltanet_recurrent",
            vec![VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
            DELTANET_LAYERS,
        ),
        expected_state_buffer("gqa_key_cache", vec![max_seq_len, 2, 256], GQA_LAYERS),
        expected_state_buffer("gqa_value_cache", vec![max_seq_len, 2, 256], GQA_LAYERS),
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
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Layer0MixerComponentEvidence {
    schema: String,
    ledger_id: String,
    status: String,
    source_identity: SourceIdentity,
    layer: usize,
    slot: usize,
    geometry: DeltaNetGeometry,
    ordered_operations: Vec<DeltaNetOperation>,
    backend: String,
    metal_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
}

impl Layer0MixerComponentEvidence {
    fn validation_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();
        if self.schema != LAYER0_COMPONENT_SCHEMA
            || self.ledger_id != LAYER0_COMPONENT_LEDGER_ID
            || self.status != LAYER0_COMPONENT_STATUS
        {
            errors.push("layer-0 mixer evidence schema, identity, or status drifted".into());
        }
        if let Err(error) = self
            .source_identity
            .validate_exact("layer-0 mixer evidence")
        {
            errors.push(error);
        }
        if self.layer != 0 || self.slot != 0 {
            errors.push(
                "layer-0 mixer evidence is not bound to source layer 0 / state slot 0".into(),
            );
        }
        if let Err(error) = self.geometry.validate_exact("layer-0 mixer evidence") {
            errors.push(error);
        }
        if self.ordered_operations != expected_operation_order() {
            errors.push(
                "layer-0 mixer QKVZ/BA/conv/recurrent/gated-norm/out-residual order drifted".into(),
            );
        }
        if self.backend != "metal"
            || self.metal_dispatches != 9
            || !self.source_bound
            || !self.artifact_bound
            || !self.device_parity_passed
        {
            errors.push(
                "layer-0 mixer loses its source/artifact/Metal component parity binding".into(),
            );
        }
        if !self.fixture_only || !self.synthetic_input || !self.component_only {
            errors.push(
                "layer-0 mixer boundary drifted; it must remain a synthetic component".into(),
            );
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DeltaNetStateWitness {
    session_id: String,
    max_seq_len: usize,
    position: usize,
    conv_slot: usize,
    recurrent_slot: usize,
    conv_allocation_id: String,
    recurrent_allocation_id: String,
    conv_offset_elements: usize,
    recurrent_offset_elements: usize,
    conv_capacity_elements: usize,
    recurrent_capacity_elements: usize,
    conv_shape: Vec<usize>,
    recurrent_shape: Vec<usize>,
    device_conv_buffer_allocated: bool,
    device_recurrent_buffer_allocated: bool,
    conv_reads_exactly_three_prior_tokens: bool,
    current_projection_is_the_fourth_causal_conv_tap: bool,
    conv_state_committed_once_after_convolution: bool,
    recurrent_state_read_before_update: bool,
    recurrent_state_committed_once: bool,
    slot_isolated_from_other_deltanet_layers: bool,
    session_isolated_from_other_sessions: bool,
    rollback_conv_and_recurrent_bytes_captured: bool,
    conv_state_bytes_device_parity_passed: bool,
    recurrent_state_bytes_device_parity_passed: bool,
}

impl DeltaNetStateWitness {
    fn physical_errors(
        &self,
        layer: usize,
        slot: usize,
        layout: &StateBufferLayoutEvidence,
    ) -> Vec<String> {
        let mut errors = Vec::new();
        if self.session_id.trim().is_empty() {
            errors.push(format!(
                "layer {layer} state witness has an empty logical session id"
            ));
        }
        if layout.max_seq_len == 0 || layout.max_seq_len > MAX_NATIVE_CONTEXT {
            errors.push(format!(
                "layer {layer} state witness cannot bind an invalid state-layout max_seq_len"
            ));
            return errors;
        }
        if self.max_seq_len != layout.max_seq_len || self.position >= layout.max_seq_len {
            errors.push(format!(
                "layer {layer} state witness position/capacity is outside the bound layout"
            ));
        }
        if self.conv_slot != slot || self.recurrent_slot != slot {
            errors.push(format!(
                "layer {layer} state witness does not own DeltaNet slot {slot}"
            ));
        }
        if self.conv_shape != [CONV_CHANNELS, CONV_STATE_TOKENS]
            || self.recurrent_shape != [VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM]
        {
            errors.push(format!(
                "layer {layer} state witness does not use source conv [8192,3] and recurrent [32,128,128] shapes"
            ));
        }
        let expected_conv_offset = slot * CONV_STATE_ELEMENTS;
        let expected_recurrent_offset = slot * RECURRENT_STATE_ELEMENTS;
        if self.conv_offset_elements != expected_conv_offset
            || self.recurrent_offset_elements != expected_recurrent_offset
            || self.conv_capacity_elements != expected_conv_offset + CONV_STATE_ELEMENTS
            || self.recurrent_capacity_elements
                != expected_recurrent_offset + RECURRENT_STATE_ELEMENTS
        {
            errors.push(format!(
                "layer {layer} state witness offsets/capacities do not match slot {slot}"
            ));
        }
        let conv_id = format!(
            "qwen80/session={}/arena=active/domain=deltanet_conv_history",
            self.session_id
        );
        let recurrent_id = format!(
            "qwen80/session={}/arena=active/domain=deltanet_recurrent",
            self.session_id
        );
        if self.conv_allocation_id != conv_id || self.recurrent_allocation_id != recurrent_id {
            errors.push(format!(
                "layer {layer} state witness is not bound to state-layout active conv/recurrent allocation ids"
            ));
        }
        if !self.device_conv_buffer_allocated
            || !self.device_recurrent_buffer_allocated
            || !self.conv_reads_exactly_three_prior_tokens
            || !self.current_projection_is_the_fourth_causal_conv_tap
            || !self.conv_state_committed_once_after_convolution
            || !self.recurrent_state_read_before_update
            || !self.recurrent_state_committed_once
            || !self.slot_isolated_from_other_deltanet_layers
            || !self.session_isolated_from_other_sessions
            || !self.rollback_conv_and_recurrent_bytes_captured
            || !self.conv_state_bytes_device_parity_passed
            || !self.recurrent_state_bytes_device_parity_passed
        {
            errors.push(format!(
                "layer {layer} state witness lacks physical allocation, source causal state transition, isolation, rollback, or state-byte parity proof"
            ));
        }
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DeltaNetLayerPhysicalEvidence {
    schema: String,
    layer: usize,
    slot: usize,
    receipt_seal_sha256: String,
    component_ancestor_ledger_id: Option<String>,
    source_identity: SourceIdentity,
    geometry: DeltaNetGeometry,
    ordered_operations: Vec<DeltaNetOperation>,
    backend: String,
    device_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    full_deltanet_mixer_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
    fallback_used: bool,
    state_witness: DeltaNetStateWitness,
}

impl DeltaNetLayerPhysicalEvidence {
    fn validation_errors(
        &self,
        expected: &DeltaNetScheduleEntry,
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
                "physical evidence maps layer {} / slot {} instead of source DeltaNet layer {} / slot {}",
                self.layer, self.slot, expected.layer, expected.slot
            ));
        }
        if !is_lower_sha256(&self.receipt_seal_sha256) {
            errors.push(format!(
                "layer {} has no valid sealed physical receipt digest",
                expected.layer
            ));
        }
        if expected.layer == 0
            && self.component_ancestor_ledger_id.as_deref() != Some(LAYER0_COMPONENT_LEDGER_ID)
        {
            errors.push(
                "layer 0 physical evidence does not retain the valid mixer evidence as an ancestor"
                    .into(),
            );
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
                "layer {} QKVZ/BA/conv/recurrent/gated-norm/out-residual operation order drifted",
                expected.layer
            ));
        }
        if self.backend != "metal"
            || self.device_dispatches == 0
            || !self.source_bound
            || !self.artifact_bound
            || !self.full_deltanet_mixer_path
            || !self.device_parity_passed
            || self.fixture_only
            || self.synthetic_input
            || self.component_only
            || self.fallback_used
        {
            errors.push(format!(
                "layer {} lacks a non-fixture source/artifact-bound full-DeltaNet-mixer Metal parity record",
                expected.layer
            ));
        }
        errors.extend(
            self.state_witness
                .physical_errors(expected.layer, expected.slot, layout),
        );
        errors
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReadinessInput {
    schema: String,
    source_identity: SourceIdentity,
    deltanet_schedule: Vec<DeltaNetScheduleEntry>,
    valid_layer0_mixer_component: Layer0MixerComponentEvidence,
    state_buffer_layout: StateBufferLayoutEvidence,
    per_layer_physical_evidence: Vec<DeltaNetLayerPhysicalEvidence>,
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
    deltanet_per_layer_device_readiness_earned: bool,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    input_schema_valid: bool,
    source_identity_valid: bool,
    exact_36_deltanet_schedule_valid: bool,
    valid_layer0_mixer_component_bound: bool,
    layer0_component_validation_errors: Vec<String>,
    state_buffer_layout_metadata_valid: bool,
    state_buffer_layout_physical_ready: bool,
    state_buffer_layout_errors: Vec<String>,
    per_layer_assessments: Vec<LayerAssessment>,
    missing_or_invalid_deltanet_layers: Vec<usize>,
    contract_errors: Vec<String>,
    expected_deltanet_schedule: Vec<DeltaNetScheduleEntry>,
    layer0_component_is_fixture_component_only: bool,
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

fn validate_schedule(schedule: &[DeltaNetScheduleEntry]) -> Result<(), String> {
    if schedule != expected_deltanet_schedule() {
        return Err(
            "DeltaNet schedule must be exactly layers 0,1,2,4,...,46 with slots 0..35".into(),
        );
    }
    let layers = schedule
        .iter()
        .map(|entry| entry.layer)
        .collect::<BTreeSet<_>>();
    let slots = schedule
        .iter()
        .map(|entry| entry.slot)
        .collect::<BTreeSet<_>>();
    if layers.len() != DELTANET_LAYERS || slots.len() != DELTANET_LAYERS {
        return Err("DeltaNet schedule contains a duplicate layer or slot".into());
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
    let exact_36_deltanet_schedule_valid = match validate_schedule(&input.deltanet_schedule) {
        Ok(()) => true,
        Err(error) => {
            contract_errors.push(error);
            false
        }
    };
    let layer0_component_validation_errors = input.valid_layer0_mixer_component.validation_errors();
    let valid_layer0_mixer_component_bound = layer0_component_validation_errors.is_empty();
    let state_buffer_layout_errors = input.state_buffer_layout.metadata_errors();
    let state_buffer_layout_metadata_valid = state_buffer_layout_errors.is_empty();
    let state_buffer_layout_physical_ready =
        state_buffer_layout_metadata_valid && input.state_buffer_layout.physically_ready();

    let expected_schedule = expected_deltanet_schedule();
    let mut records_by_layer = BTreeMap::<usize, Vec<&DeltaNetLayerPhysicalEvidence>>::new();
    for record in &input.per_layer_physical_evidence {
        records_by_layer
            .entry(record.layer)
            .or_default()
            .push(record);
    }
    for layer in records_by_layer.keys() {
        if !expected_schedule.iter().any(|entry| entry.layer == *layer) {
            contract_errors.push(format!(
                "physical evidence covers non-DeltaNet source layer {layer}"
            ));
        }
    }

    let mut per_layer_assessments = Vec::with_capacity(DELTANET_LAYERS);
    let mut missing_or_invalid_deltanet_layers = Vec::new();
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
            missing_or_invalid_deltanet_layers.push(expected.layer);
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

    let deltanet_per_layer_device_readiness_earned = input_schema_valid
        && source_identity_valid
        && exact_36_deltanet_schedule_valid
        && valid_layer0_mixer_component_bound
        && state_buffer_layout_physical_ready
        && contract_errors.is_empty()
        && missing_or_invalid_deltanet_layers.is_empty();
    let status = if deltanet_per_layer_device_readiness_earned {
        "READY_FOR_QWEN80_DELTANET_PER_LAYER_DEVICE_INTEGRATION_NOT_A_COMPLETE_DECODER"
    } else {
        "INCOMPLETE_QWEN80_DELTANET_PER_LAYER_DEVICE_READINESS_LAYER0_COMPONENT_DOES_NOT_COVER_ALL_36_DELTANET_LAYERS"
    };
    let mut report = ReadinessReport {
        schema: RESULT_SCHEMA,
        status,
        deltanet_per_layer_device_readiness_earned,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        input_schema_valid,
        source_identity_valid,
        exact_36_deltanet_schedule_valid,
        valid_layer0_mixer_component_bound,
        layer0_component_validation_errors,
        state_buffer_layout_metadata_valid,
        state_buffer_layout_physical_ready,
        state_buffer_layout_errors,
        per_layer_assessments,
        missing_or_invalid_deltanet_layers,
        contract_errors,
        expected_deltanet_schedule: expected_schedule,
        layer0_component_is_fixture_component_only: input.valid_layer0_mixer_component.component_only,
        read_only_contract: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        required_before_ready: vec![
            "Produce one sealed source/artifact-bound, non-fixture full-DeltaNet-mixer Metal parity ledger for every source layer 0,1,2,4,...,46 with no fallback.",
            "For every ledger, prove exact input norm, QKVZ/BA interleaving, causal depthwise SiLU convolution, repeated Q/K L2, BA/A_log/dt_bias controls, DeltaNet recurrence, Z-gated norm, out projection, and first residual order.",
            "Allocate and prove exact active plus rollback per-session conv [8192,3] and recurrent [32,128,128] slot ranges; prove offsets, capacities, and no cross-layer/session aliases.",
            "Capture multi-position CPU/device parity for both state bytes and hidden boundaries, including the three-prior-token convolution window and exactly-once recurrent-state commit/rollback.",
            "After this narrow DeltaNet frontier is satisfied, independently satisfy GQA, MoE, terminal, full-graph, tokenizer, and clean TPS/TG gates before any decoder/server claim.",
        ],
        claim_boundary: vec![
            "The known layer-0 record is a valid direct-packed synthetic mixer component through first residual, not a complete source transformer layer or decoder token.",
            "This CPU-only contract does not open an artifact or device, allocate state, execute a DeltaNet layer, emit a token, expose HCLI, or measure TPS/TG.",
            "Even a ready DeltaNet-per-layer result is not a complete decoder, resident server, capability, Agent OS, tournament, BASE_TRUE_TPS, or TG qualification.",
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

fn current_layer0_mixer_component() -> Layer0MixerComponentEvidence {
    Layer0MixerComponentEvidence {
        schema: LAYER0_COMPONENT_SCHEMA.into(),
        ledger_id: LAYER0_COMPONENT_LEDGER_ID.into(),
        status: LAYER0_COMPONENT_STATUS.into(),
        source_identity: SourceIdentity::exact(),
        layer: 0,
        slot: 0,
        geometry: DeltaNetGeometry::exact(),
        ordered_operations: expected_operation_order(),
        backend: "metal".into(),
        metal_dispatches: 9,
        source_bound: true,
        artifact_bound: true,
        device_parity_passed: true,
        fixture_only: true,
        synthetic_input: true,
        component_only: true,
    }
}

fn state_witness(
    session_id: &str,
    max_seq_len: usize,
    position: usize,
    slot: usize,
    physical: bool,
) -> DeltaNetStateWitness {
    let conv_offset = slot * CONV_STATE_ELEMENTS;
    let recurrent_offset = slot * RECURRENT_STATE_ELEMENTS;
    DeltaNetStateWitness {
        session_id: session_id.into(),
        max_seq_len,
        position,
        conv_slot: slot,
        recurrent_slot: slot,
        conv_allocation_id: format!(
            "qwen80/session={session_id}/arena=active/domain=deltanet_conv_history"
        ),
        recurrent_allocation_id: format!(
            "qwen80/session={session_id}/arena=active/domain=deltanet_recurrent"
        ),
        conv_offset_elements: conv_offset,
        recurrent_offset_elements: recurrent_offset,
        conv_capacity_elements: conv_offset + CONV_STATE_ELEMENTS,
        recurrent_capacity_elements: recurrent_offset + RECURRENT_STATE_ELEMENTS,
        conv_shape: vec![CONV_CHANNELS, CONV_STATE_TOKENS],
        recurrent_shape: vec![VALUE_HEADS, KEY_HEAD_DIM, VALUE_HEAD_DIM],
        device_conv_buffer_allocated: physical,
        device_recurrent_buffer_allocated: physical,
        conv_reads_exactly_three_prior_tokens: true,
        current_projection_is_the_fourth_causal_conv_tap: true,
        conv_state_committed_once_after_convolution: true,
        recurrent_state_read_before_update: true,
        recurrent_state_committed_once: true,
        slot_isolated_from_other_deltanet_layers: physical,
        session_isolated_from_other_sessions: physical,
        rollback_conv_and_recurrent_bytes_captured: physical,
        conv_state_bytes_device_parity_passed: physical,
        recurrent_state_bytes_device_parity_passed: physical,
    }
}

fn current_layer0_physical_evidence() -> DeltaNetLayerPhysicalEvidence {
    let layout = current_state_buffer_layout();
    DeltaNetLayerPhysicalEvidence {
        schema: LAYER_EVIDENCE_SCHEMA.into(),
        layer: 0,
        slot: 0,
        receipt_seal_sha256: String::new(),
        component_ancestor_ledger_id: Some(LAYER0_COMPONENT_LEDGER_ID.into()),
        source_identity: SourceIdentity::exact(),
        geometry: DeltaNetGeometry::exact(),
        ordered_operations: expected_operation_order(),
        backend: "metal".into(),
        device_dispatches: 9,
        source_bound: true,
        artifact_bound: true,
        full_deltanet_mixer_path: true,
        device_parity_passed: true,
        fixture_only: true,
        synthetic_input: true,
        component_only: true,
        fallback_used: false,
        state_witness: state_witness("layer0-mixer-component", layout.max_seq_len, 0, 0, false),
    }
}

fn current_evidence_input() -> ReadinessInput {
    ReadinessInput {
        schema: INPUT_SCHEMA.into(),
        source_identity: SourceIdentity::exact(),
        deltanet_schedule: expected_deltanet_schedule(),
        valid_layer0_mixer_component: current_layer0_mixer_component(),
        state_buffer_layout: current_state_buffer_layout(),
        per_layer_physical_evidence: vec![current_layer0_physical_evidence()],
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
    "usage: ascension_qwen80_deltanet_per_layer_device_readiness_contract \
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
        if !report.deltanet_per_layer_device_readiness_earned {
            return Err(format!(
                "DeltaNet per-layer device readiness is incomplete; report written to {}",
                args.out.display()
            )
            .into());
        }
        Ok(())
    })();
    if let Err(error) = result {
        eprintln!("ascension_qwen80_deltanet_per_layer_device_readiness_contract: {error}");
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
        input.per_layer_physical_evidence = expected_deltanet_schedule()
            .into_iter()
            .map(|entry| DeltaNetLayerPhysicalEvidence {
                schema: LAYER_EVIDENCE_SCHEMA.into(),
                layer: entry.layer,
                slot: entry.slot,
                receipt_seal_sha256: format!("{:064x}", entry.layer + 1),
                component_ancestor_ledger_id: (entry.layer == 0)
                    .then(|| LAYER0_COMPONENT_LEDGER_ID.into()),
                source_identity: SourceIdentity::exact(),
                geometry: DeltaNetGeometry::exact(),
                ordered_operations: expected_operation_order(),
                backend: "metal".into(),
                device_dispatches: 17,
                source_bound: true,
                artifact_bound: true,
                full_deltanet_mixer_path: true,
                device_parity_passed: true,
                fixture_only: false,
                synthetic_input: false,
                component_only: false,
                fallback_used: false,
                state_witness: state_witness(
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
    fn source_schedule_has_exactly_thirty_six_deltanet_layers_and_slots() {
        let schedule = expected_deltanet_schedule();
        assert_eq!(schedule.len(), 36);
        assert_eq!(
            schedule.first(),
            Some(&DeltaNetScheduleEntry { layer: 0, slot: 0 })
        );
        assert_eq!(schedule[2], DeltaNetScheduleEntry { layer: 2, slot: 2 });
        assert_eq!(schedule[3], DeltaNetScheduleEntry { layer: 4, slot: 3 });
        assert_eq!(
            schedule.last(),
            Some(&DeltaNetScheduleEntry {
                layer: 46,
                slot: 35
            })
        );
        validate_schedule(&schedule).unwrap();
    }

    #[test]
    fn current_valid_layer0_component_remains_incomplete() {
        let report = evaluate(&current_evidence_input());
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        assert_eq!(
            report.status,
            "INCOMPLETE_QWEN80_DELTANET_PER_LAYER_DEVICE_READINESS_LAYER0_COMPONENT_DOES_NOT_COVER_ALL_36_DELTANET_LAYERS"
        );
        assert!(report.valid_layer0_mixer_component_bound);
        assert!(!report.state_buffer_layout_physical_ready);
        assert_eq!(report.per_layer_assessments.len(), 36);
        assert_eq!(report.missing_or_invalid_deltanet_layers.len(), 36);
        assert!(!report.complete_decoder_readiness_earned);
    }

    #[test]
    fn rejects_wrong_slot_and_non_causal_state_transition() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[3].slot = 7;
        input.per_layer_physical_evidence[3]
            .state_witness
            .recurrent_state_committed_once = false;
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 4)
            .unwrap();
        assert!(!assessment.satisfied);
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("source DeltaNet layer")));
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("source causal state transition")));
    }

    #[test]
    fn rejects_qkvz_ba_or_gated_norm_geometry_drift() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0]
            .geometry
            .qkvz_z_offset_rows = 511;
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 0)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure
                .contains("QKVZ/BA/conv/recurrent/gated-norm/out-residual geometry")));
    }

    #[test]
    fn rejects_operator_order_or_state_offset_drift() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0]
            .ordered_operations
            .swap(0, 1);
        input.per_layer_physical_evidence[0]
            .state_witness
            .conv_offset_elements += 1;
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        let assessment = report
            .per_layer_assessments
            .iter()
            .find(|assessment| assessment.layer == 0)
            .unwrap();
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("operation order drifted")));
        assert!(assessment
            .failures
            .iter()
            .any(|failure| failure.contains("offsets/capacities")));
    }

    #[test]
    fn rejects_an_unsealed_or_placeholder_per_layer_receipt() {
        let mut input = fully_physical_input();
        input.per_layer_physical_evidence[0].receipt_seal_sha256 = "0".repeat(64);
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
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
    fn rejects_missing_one_of_the_thirty_six_physical_layer_ledgers() {
        let mut input = fully_physical_input();
        input
            .per_layer_physical_evidence
            .retain(|record| record.layer != 46);
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        assert!(report.missing_or_invalid_deltanet_layers.contains(&46));
    }

    #[test]
    fn rejects_unphysical_state_layout_even_with_all_layer_ledgers() {
        let mut input = fully_physical_input();
        input.state_buffer_layout.actual_device_allocation_performed = false;
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        assert!(!report.state_buffer_layout_physical_ready);
    }

    #[test]
    fn rejects_state_layout_beyond_native_context_without_promoting_state_evidence() {
        let mut input = fully_physical_input();
        input.state_buffer_layout.max_seq_len = MAX_NATIVE_CONTEXT + 1;
        let report = evaluate(&input);
        assert!(!report.deltanet_per_layer_device_readiness_earned);
        assert!(!report.state_buffer_layout_metadata_valid);
        assert!(report
            .state_buffer_layout_errors
            .iter()
            .any(|error| error.contains("outside 1..=")));
    }

    #[test]
    fn exact_hypothetical_all_layer_physical_ledger_satisfies_only_this_narrow_frontier() {
        let report = evaluate(&fully_physical_input());
        assert!(report.deltanet_per_layer_device_readiness_earned);
        assert!(report
            .per_layer_assessments
            .iter()
            .all(|assessment| assessment.satisfied));
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
    }
}
