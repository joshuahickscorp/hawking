//! CPU-only Qwen3-Coder-Next hybrid decode-state / KV contract.
//!
//! This is deliberately *not* a Qwen80 runtime, server, artifact reader,
//! Metal probe, token execution path, HCLI adapter, or throughput benchmark.
//! It is a small source-archaeology and fixture contract that makes the state
//! ownership a future complete decoder must preserve explicit:
//!
//! * the exact 48-layer `DeltaNet, DeltaNet, DeltaNet, GQA` schedule;
//! * one distinct DeltaNet convolution + recurrent slot for every one of the
//!   36 linear-attention layers;
//! * one distinct K and V slot for every one of the 12 GQA layers;
//! * causal update order, feedback handoff prerequisites, and session-local
//!   checkpoint / rollback identity.
//!
//! The fixture stores only SHA-256 commitments to hypothetical state content;
//! it never allocates model tensors, opens an artifact, invokes a tokenizer,
//! samples logits, or executes a model token.  Its JSON report is intentionally
//! marked `NOT_RUNTIME`, `NO_TOKEN`, `NO_HCLI`, and `NO_TPS`.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_decode_state_contract -- \
//!   --max-seq-len 16 --out /absolute/path/QWEN80_DECODE_STATE_CONTRACT.json
//! ```
//!
//! Omitting `--out` prints the machine-readable report to stdout.

use hawking_core::model::qwen80_complete_runtime::{
    qwen80_layer_kind, Qwen80LayerKind, QWEN80_COMPLETE_NATIVE_MAX_CONTEXT,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process;

const SCHEMA: &str = "hawking.ascension.qwen80_decode_state_contract.v1";
const STATUS: &str = "NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS_QWEN80_HYBRID_DECODE_STATE_KV_CONTRACT";
const SOURCE_MODULE: &str = "crates/hawking-core/src/model/qwen80_complete_runtime.rs";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";

const LAYER_COUNT: usize = 48;
const DELTANET_LAYERS: usize = 36;
const GQA_LAYERS: usize = 12;
const FULL_ATTENTION_INTERVAL: usize = 4;
const TOKENIZER_VOCAB: u32 = 151_669;
const LM_HEAD_VOCAB: u32 = 151_936;
const RESERVED_TAIL_ROWS: u32 = LM_HEAD_VOCAB - TOKENIZER_VOCAB;

const LINEAR_KEY_HEADS: usize = 16;
const LINEAR_VALUE_HEADS: usize = 32;
const LINEAR_KEY_HEAD_DIM: usize = 128;
const LINEAR_VALUE_HEAD_DIM: usize = 128;
const LINEAR_CONV_KERNEL: usize = 4;
const FULL_ATTN_KV_HEADS: usize = 2;
const FULL_ATTN_HEAD_DIM: usize = 256;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum LayerKind {
    DeltaNet,
    Gqa,
}

impl LayerKind {
    fn from_source(layer: usize) -> Result<Self, String> {
        match qwen80_layer_kind(layer).map_err(|error| error.to_string())? {
            Qwen80LayerKind::LinearAttention => Ok(Self::DeltaNet),
            Qwen80LayerKind::FullAttention => Ok(Self::Gqa),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
enum StateDomain {
    DeltaNetConvHistory,
    DeltaNetRecurrent,
    GqaKeyCache,
    GqaValueCache,
}

impl StateDomain {
    fn as_str(self) -> &'static str {
        match self {
            Self::DeltaNetConvHistory => "deltanet_conv_history",
            Self::DeltaNetRecurrent => "deltanet_recurrent",
            Self::GqaKeyCache => "gqa_key_cache",
            Self::GqaValueCache => "gqa_value_cache",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LayerStatePlan {
    layer: usize,
    source_mixer: LayerKind,
    linear_state_slot: Option<usize>,
    gqa_kv_slot: Option<usize>,
    linear_conv_history_shape: Option<Vec<usize>>,
    linear_recurrent_shape: Option<Vec<usize>>,
    gqa_key_cache_shape: Option<Vec<usize>>,
    gqa_value_cache_shape: Option<Vec<usize>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct HybridDecodeStatePlan {
    max_seq_len: usize,
    layers: Vec<LayerStatePlan>,
    source_schedule_fingerprint_sha256: String,
}

#[derive(Serialize)]
struct PlanPreimage<'a> {
    max_seq_len: usize,
    layers: &'a [LayerStatePlan],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StateRecord {
    domain: StateDomain,
    slot: usize,
    owner_layer: usize,
    shape: Vec<usize>,
    originating_session_id: String,
    storage_identity: String,
    committed_positions: usize,
    content_commitment_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct DecodeSession {
    session_id: String,
    plan_fingerprint_sha256: String,
    max_seq_len: usize,
    position: usize,
    prompt_handoff_token_id: u32,
    last_fixture_feedback_token_id: Option<u32>,
    state_records: Vec<StateRecord>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SessionSnapshot {
    snapshot_schema: &'static str,
    session: DecodeSession,
    snapshot_identity_sha256: String,
}

#[derive(Serialize)]
struct SnapshotPreimage<'a> {
    snapshot_schema: &'static str,
    session: &'a DecodeSession,
}

#[derive(Clone, Copy, Debug)]
struct FixtureDecodeStep {
    position: usize,
    input_token_id: u32,
    selected_feedback_token_id: u32,
}

#[derive(Clone, Debug, Serialize)]
struct FixtureDecodeTrace {
    fixture_only: bool,
    position: usize,
    input_token_id: u32,
    selected_feedback_token_id: u32,
    state_records_updated: usize,
    causal_update_order: Vec<String>,
    feedback_committed_only_after_full_hybrid_state_update: bool,
}

#[derive(Serialize)]
struct SourceArchaeology {
    source_module: &'static str,
    source_repository: &'static str,
    source_revision: &'static str,
    model_id: &'static str,
    source_schedule_authority: &'static str,
    source_schedule_checked_via_public_resolver: bool,
    layer_count: usize,
    deltanet_layers: usize,
    gqa_layers: usize,
    schedule: Vec<LayerKind>,
    delta_conv_history_per_linear_layer: Vec<usize>,
    delta_recurrent_per_linear_layer: Vec<usize>,
    gqa_key_value_per_gqa_layer: Vec<usize>,
    source_tokenizer_vocab: u32,
    lm_head_rows: u32,
    reserved_lm_head_tail_rows: u32,
}

#[derive(Serialize)]
struct StateGeometryReport {
    per_session_state_records: usize,
    linear_state_slots: usize,
    gqa_kv_slots: usize,
    total_delta_conv_history_elements: usize,
    total_delta_recurrent_elements: usize,
    total_gqa_key_cache_elements: usize,
    total_gqa_value_cache_elements: usize,
    state_content_materialized: bool,
}

#[derive(Serialize)]
struct ContractChecks {
    exact_schedule_checked: bool,
    layer_state_owner_checked: bool,
    state_slot_aliasing_checked: bool,
    state_shape_checked: bool,
    causal_position_and_update_order_checked: bool,
    fixture_feedback_prerequisite_checked: bool,
    restart_identity_checked: bool,
    rollback_identity_checked: bool,
    wrong_layer_kind_rejected: bool,
    wrong_state_shape_rejected: bool,
    slot_reuse_rejected: bool,
    cross_session_leakage_rejected: bool,
    lm_head_reserved_tail_token_rejected: bool,
}

#[derive(Serialize)]
struct RawlsStateKvGap {
    gap_id: &'static str,
    required_before_complete_decoder: &'static str,
    required_evidence: &'static str,
    claim_boundary: &'static str,
}

#[derive(Serialize)]
struct ExecutionBoundary {
    not_runtime: bool,
    no_live_artifact_scan: bool,
    no_packed_tensor_read: bool,
    no_metal_device_or_dispatch: bool,
    no_model_token_execution: bool,
    no_logit_or_sampler_execution: bool,
    no_hcli_execution: bool,
    no_tps_or_tg_measurement: bool,
    no_server_started: bool,
}

#[derive(Serialize)]
struct ReadinessReport {
    schema: &'static str,
    status: &'static str,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    source_archaeology: SourceArchaeology,
    state_geometry: StateGeometryReport,
    fixture_contract_checks: ContractChecks,
    rawls_state_kv_handoff: Vec<RawlsStateKvGap>,
    execution_boundary: ExecutionBoundary,
    unsealed_preimage_sha256: String,
}

struct Arguments {
    max_seq_len: usize,
    out: Option<PathBuf>,
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    let serialized =
        serde_json::to_vec(value).expect("contract preimage serialization must succeed");
    format!("{:x}", Sha256::digest(serialized))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn linear_conv_history_shape() -> Vec<usize> {
    let channels =
        LINEAR_KEY_HEADS * LINEAR_KEY_HEAD_DIM * 2 + LINEAR_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM;
    vec![channels, LINEAR_CONV_KERNEL - 1]
}

fn linear_recurrent_shape() -> Vec<usize> {
    vec![
        LINEAR_VALUE_HEADS,
        LINEAR_KEY_HEAD_DIM,
        LINEAR_VALUE_HEAD_DIM,
    ]
}

fn gqa_cache_shape(max_seq_len: usize) -> Vec<usize> {
    vec![max_seq_len, FULL_ATTN_KV_HEADS, FULL_ATTN_HEAD_DIM]
}

fn product(shape: &[usize]) -> Result<usize, String> {
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| "state shape element count overflowed".to_owned())
    })
}

fn expected_plan(max_seq_len: usize) -> Result<HybridDecodeStatePlan, String> {
    if max_seq_len == 0 || max_seq_len > QWEN80_COMPLETE_NATIVE_MAX_CONTEXT {
        return Err(format!(
            "max_seq_len={max_seq_len} must be within 1..={QWEN80_COMPLETE_NATIVE_MAX_CONTEXT}"
        ));
    }
    let mut layers = Vec::with_capacity(LAYER_COUNT);
    let mut next_linear_slot = 0usize;
    let mut next_gqa_slot = 0usize;
    for layer in 0..LAYER_COUNT {
        match LayerKind::from_source(layer)? {
            LayerKind::DeltaNet => {
                layers.push(LayerStatePlan {
                    layer,
                    source_mixer: LayerKind::DeltaNet,
                    linear_state_slot: Some(next_linear_slot),
                    gqa_kv_slot: None,
                    linear_conv_history_shape: Some(linear_conv_history_shape()),
                    linear_recurrent_shape: Some(linear_recurrent_shape()),
                    gqa_key_cache_shape: None,
                    gqa_value_cache_shape: None,
                });
                next_linear_slot += 1;
            }
            LayerKind::Gqa => {
                layers.push(LayerStatePlan {
                    layer,
                    source_mixer: LayerKind::Gqa,
                    linear_state_slot: None,
                    gqa_kv_slot: Some(next_gqa_slot),
                    linear_conv_history_shape: None,
                    linear_recurrent_shape: None,
                    gqa_key_cache_shape: Some(gqa_cache_shape(max_seq_len)),
                    gqa_value_cache_shape: Some(gqa_cache_shape(max_seq_len)),
                });
                next_gqa_slot += 1;
            }
        }
    }
    if next_linear_slot != DELTANET_LAYERS || next_gqa_slot != GQA_LAYERS {
        return Err(format!(
            "source schedule yielded {next_linear_slot} DeltaNet and {next_gqa_slot} GQA layers, expected {DELTANET_LAYERS}/{GQA_LAYERS}"
        ));
    }
    let mut plan = HybridDecodeStatePlan {
        max_seq_len,
        layers,
        source_schedule_fingerprint_sha256: String::new(),
    };
    plan.source_schedule_fingerprint_sha256 = plan_fingerprint(&plan);
    Ok(plan)
}

fn plan_fingerprint(plan: &HybridDecodeStatePlan) -> String {
    sha256_json(&PlanPreimage {
        max_seq_len: plan.max_seq_len,
        layers: &plan.layers,
    })
}

fn expected_layer_kind_for_position(layer: usize) -> LayerKind {
    if (layer + 1) % FULL_ATTENTION_INTERVAL == 0 {
        LayerKind::Gqa
    } else {
        LayerKind::DeltaNet
    }
}

fn validate_plan(plan: &HybridDecodeStatePlan) -> Result<(), String> {
    if plan.max_seq_len == 0 || plan.max_seq_len > QWEN80_COMPLETE_NATIVE_MAX_CONTEXT {
        return Err("plan max_seq_len is outside the native contract limit".into());
    }
    if plan.layers.len() != LAYER_COUNT {
        return Err(format!(
            "plan has {} layers; exact Qwen80 schedule requires {LAYER_COUNT}",
            plan.layers.len()
        ));
    }
    let mut next_linear_slot = 0usize;
    let mut next_gqa_slot = 0usize;
    let mut owned_linear_slots = BTreeSet::new();
    let mut owned_gqa_slots = BTreeSet::new();
    for (expected_layer, layer) in plan.layers.iter().enumerate() {
        if layer.layer != expected_layer {
            return Err(format!(
                "plan layer index {} is not contiguous at position {expected_layer}",
                layer.layer
            ));
        }
        let source_kind = LayerKind::from_source(expected_layer)?;
        let arithmetic_kind = expected_layer_kind_for_position(expected_layer);
        if source_kind != arithmetic_kind {
            return Err(format!(
                "source resolver and 3xDeltaNet/1xGQA arithmetic disagree at layer {expected_layer}"
            ));
        }
        if layer.source_mixer != source_kind {
            return Err(format!(
                "layer {expected_layer} mixer {:?} disagrees with source {:?}",
                layer.source_mixer, source_kind
            ));
        }
        match source_kind {
            LayerKind::DeltaNet => {
                let slot = layer.linear_state_slot.ok_or_else(|| {
                    format!("DeltaNet layer {expected_layer} has no linear state slot")
                })?;
                if slot != next_linear_slot || !owned_linear_slots.insert(slot) {
                    return Err(format!(
                        "DeltaNet layer {expected_layer} reuses or skips linear state slot {slot}; expected {next_linear_slot}"
                    ));
                }
                if layer.gqa_kv_slot.is_some()
                    || layer.gqa_key_cache_shape.is_some()
                    || layer.gqa_value_cache_shape.is_some()
                {
                    return Err(format!(
                        "DeltaNet layer {expected_layer} illegally owns GQA KV state"
                    ));
                }
                if layer.linear_conv_history_shape.as_deref()
                    != Some(linear_conv_history_shape().as_slice())
                    || layer.linear_recurrent_shape.as_deref()
                        != Some(linear_recurrent_shape().as_slice())
                {
                    return Err(format!(
                        "DeltaNet layer {expected_layer} has a non-source state shape"
                    ));
                }
                next_linear_slot += 1;
            }
            LayerKind::Gqa => {
                let slot = layer
                    .gqa_kv_slot
                    .ok_or_else(|| format!("GQA layer {expected_layer} has no KV state slot"))?;
                if slot != next_gqa_slot || !owned_gqa_slots.insert(slot) {
                    return Err(format!(
                        "GQA layer {expected_layer} reuses or skips KV state slot {slot}; expected {next_gqa_slot}"
                    ));
                }
                if layer.linear_state_slot.is_some()
                    || layer.linear_conv_history_shape.is_some()
                    || layer.linear_recurrent_shape.is_some()
                {
                    return Err(format!(
                        "GQA layer {expected_layer} illegally owns DeltaNet state"
                    ));
                }
                let expected_cache = gqa_cache_shape(plan.max_seq_len);
                if layer.gqa_key_cache_shape.as_deref() != Some(expected_cache.as_slice())
                    || layer.gqa_value_cache_shape.as_deref() != Some(expected_cache.as_slice())
                {
                    return Err(format!(
                        "GQA layer {expected_layer} has a non-source KV cache shape"
                    ));
                }
                next_gqa_slot += 1;
            }
        }
    }
    if next_linear_slot != DELTANET_LAYERS || next_gqa_slot != GQA_LAYERS {
        return Err(format!(
            "state slot counts are linear={next_linear_slot}/{DELTANET_LAYERS}, gqa={next_gqa_slot}/{GQA_LAYERS}"
        ));
    }
    if plan.source_schedule_fingerprint_sha256 != plan_fingerprint(plan) {
        return Err("plan source schedule fingerprint no longer matches the plan body".into());
    }
    Ok(())
}

fn validate_token_id(token_id: u32, label: &str) -> Result<(), String> {
    if token_id >= TOKENIZER_VOCAB {
        return Err(format!(
            "{label}={token_id} is outside tokenizer namespace 0..{}; reserved lm_head tail {}..{} must be masked and cannot enter feedback",
            TOKENIZER_VOCAB - 1,
            TOKENIZER_VOCAB,
            LM_HEAD_VOCAB - 1
        ));
    }
    Ok(())
}

fn initial_commitment(
    session_id: &str,
    plan_fingerprint_sha256: &str,
    domain: StateDomain,
    slot: usize,
    owner_layer: usize,
    shape: &[usize],
) -> String {
    sha256_json(&serde_json::json!({
        "kind": "qwen80_decode_state_contract_zero_fixture",
        "session_id": session_id,
        "plan_fingerprint_sha256": plan_fingerprint_sha256,
        "domain": domain,
        "slot": slot,
        "owner_layer": owner_layer,
        "shape": shape,
    }))
}

fn state_storage_identity(session_id: &str, domain: StateDomain, slot: usize) -> String {
    format!(
        "qwen80-decode-state-contract/session={session_id}/domain={}/slot={slot}",
        domain.as_str()
    )
}

fn state_records_for_plan(
    plan: &HybridDecodeStatePlan,
    session_id: &str,
) -> Result<Vec<StateRecord>, String> {
    validate_plan(plan)?;
    let mut records = Vec::with_capacity(DELTANET_LAYERS * 2 + GQA_LAYERS * 2);
    for layer in &plan.layers {
        match layer.source_mixer {
            LayerKind::DeltaNet => {
                let slot = layer.linear_state_slot.expect("validated DeltaNet slot");
                for (domain, shape) in [
                    (
                        StateDomain::DeltaNetConvHistory,
                        layer
                            .linear_conv_history_shape
                            .as_ref()
                            .expect("validated DeltaNet convolution shape"),
                    ),
                    (
                        StateDomain::DeltaNetRecurrent,
                        layer
                            .linear_recurrent_shape
                            .as_ref()
                            .expect("validated DeltaNet recurrent shape"),
                    ),
                ] {
                    records.push(StateRecord {
                        domain,
                        slot,
                        owner_layer: layer.layer,
                        shape: shape.clone(),
                        originating_session_id: session_id.to_owned(),
                        storage_identity: state_storage_identity(session_id, domain, slot),
                        committed_positions: 0,
                        content_commitment_sha256: initial_commitment(
                            session_id,
                            &plan.source_schedule_fingerprint_sha256,
                            domain,
                            slot,
                            layer.layer,
                            shape,
                        ),
                    });
                }
            }
            LayerKind::Gqa => {
                let slot = layer.gqa_kv_slot.expect("validated GQA slot");
                for (domain, shape) in [
                    (
                        StateDomain::GqaKeyCache,
                        layer
                            .gqa_key_cache_shape
                            .as_ref()
                            .expect("validated GQA key shape"),
                    ),
                    (
                        StateDomain::GqaValueCache,
                        layer
                            .gqa_value_cache_shape
                            .as_ref()
                            .expect("validated GQA value shape"),
                    ),
                ] {
                    records.push(StateRecord {
                        domain,
                        slot,
                        owner_layer: layer.layer,
                        shape: shape.clone(),
                        originating_session_id: session_id.to_owned(),
                        storage_identity: state_storage_identity(session_id, domain, slot),
                        committed_positions: 0,
                        content_commitment_sha256: initial_commitment(
                            session_id,
                            &plan.source_schedule_fingerprint_sha256,
                            domain,
                            slot,
                            layer.layer,
                            shape,
                        ),
                    });
                }
            }
        }
    }
    Ok(records)
}

impl DecodeSession {
    fn new(
        plan: &HybridDecodeStatePlan,
        session_id: impl Into<String>,
        prompt_handoff_token_id: u32,
    ) -> Result<Self, String> {
        validate_plan(plan)?;
        validate_token_id(prompt_handoff_token_id, "prompt_handoff_token_id")?;
        let session_id = session_id.into();
        if session_id.trim().is_empty() {
            return Err("session_id must be non-empty".into());
        }
        let state_records = state_records_for_plan(plan, &session_id)?;
        let session = Self {
            session_id,
            plan_fingerprint_sha256: plan.source_schedule_fingerprint_sha256.clone(),
            max_seq_len: plan.max_seq_len,
            position: 0,
            prompt_handoff_token_id,
            last_fixture_feedback_token_id: None,
            state_records,
        };
        session.validate(plan)?;
        Ok(session)
    }

    fn validate(&self, plan: &HybridDecodeStatePlan) -> Result<(), String> {
        validate_plan(plan)?;
        if self.plan_fingerprint_sha256 != plan.source_schedule_fingerprint_sha256
            || self.max_seq_len != plan.max_seq_len
        {
            return Err("session is bound to a different hybrid state plan".into());
        }
        if self.session_id.trim().is_empty() {
            return Err("session has an empty identity".into());
        }
        if self.position > self.max_seq_len {
            return Err("session position exceeds its cache capacity".into());
        }
        validate_token_id(
            self.prompt_handoff_token_id,
            "session prompt_handoff_token_id",
        )?;
        if let Some(token) = self.last_fixture_feedback_token_id {
            validate_token_id(token, "session last_fixture_feedback_token_id")?;
            if self.position == 0 {
                return Err("session has feedback before the first committed fixture step".into());
            }
        } else if self.position != 0 {
            return Err("session has committed positions but no feedback handoff".into());
        }

        let expected = state_records_for_plan(plan, &self.session_id)?;
        if self.state_records.len() != expected.len() {
            return Err(format!(
                "session has {} state records, expected {}",
                self.state_records.len(),
                expected.len()
            ));
        }
        let expected_by_key: BTreeMap<(StateDomain, usize), &StateRecord> = expected
            .iter()
            .map(|record| ((record.domain, record.slot), record))
            .collect();
        let mut seen_state_keys = BTreeSet::new();
        let mut seen_storage_identities = BTreeSet::new();
        for record in &self.state_records {
            if !seen_state_keys.insert((record.domain, record.slot)) {
                return Err(format!(
                    "session reuses state slot {}:{}",
                    record.domain.as_str(),
                    record.slot
                ));
            }
            if !seen_storage_identities.insert(record.storage_identity.as_str()) {
                return Err(format!(
                    "session aliases state storage {}",
                    record.storage_identity
                ));
            }
            let required = expected_by_key
                .get(&(record.domain, record.slot))
                .ok_or_else(|| {
                    format!(
                        "session owns undeclared state slot {}:{}",
                        record.domain.as_str(),
                        record.slot
                    )
                })?;
            if record.owner_layer != required.owner_layer || record.shape != required.shape {
                return Err(format!(
                    "session state slot {}:{} has wrong owner or shape",
                    record.domain.as_str(),
                    record.slot
                ));
            }
            if record.originating_session_id != self.session_id
                || record.storage_identity != required.storage_identity
            {
                return Err(format!(
                    "session state slot {}:{} leaks another session namespace",
                    record.domain.as_str(),
                    record.slot
                ));
            }
            if record.committed_positions != self.position {
                return Err(format!(
                    "session state slot {}:{} was updated {} times for position {}",
                    record.domain.as_str(),
                    record.slot,
                    record.committed_positions,
                    self.position
                ));
            }
            if !is_lower_sha256(&record.content_commitment_sha256) {
                return Err("session state record has an invalid content commitment".into());
            }
        }
        Ok(())
    }

    fn snapshot(&self, plan: &HybridDecodeStatePlan) -> Result<SessionSnapshot, String> {
        self.validate(plan)?;
        let mut snapshot = SessionSnapshot {
            snapshot_schema: "hawking.ascension.qwen80_decode_state_snapshot.v1",
            session: self.clone(),
            snapshot_identity_sha256: String::new(),
        };
        snapshot.snapshot_identity_sha256 = snapshot_identity(&snapshot);
        Ok(snapshot)
    }

    fn restart_from_snapshot(
        plan: &HybridDecodeStatePlan,
        snapshot: &SessionSnapshot,
    ) -> Result<Self, String> {
        validate_snapshot(plan, snapshot)?;
        Ok(snapshot.session.clone())
    }

    fn rollback_to(
        &mut self,
        plan: &HybridDecodeStatePlan,
        snapshot: &SessionSnapshot,
    ) -> Result<(), String> {
        self.validate(plan)?;
        validate_snapshot(plan, snapshot)?;
        if snapshot.session.session_id != self.session_id {
            return Err("rollback snapshot belongs to a different session".into());
        }
        *self = snapshot.session.clone();
        self.validate(plan)
    }

    fn apply_fixture_decode(
        &mut self,
        plan: &HybridDecodeStatePlan,
        step: FixtureDecodeStep,
    ) -> Result<FixtureDecodeTrace, String> {
        self.validate(plan)?;
        if step.position != self.position {
            return Err(format!(
                "fixture step position {} does not equal session position {}",
                step.position, self.position
            ));
        }
        if self.position >= self.max_seq_len {
            return Err("fixture step exceeds the state/KV cache capacity".into());
        }
        validate_token_id(step.input_token_id, "fixture input_token_id")?;
        validate_token_id(
            step.selected_feedback_token_id,
            "fixture selected_feedback_token_id",
        )?;
        let expected_input = if self.position == 0 {
            self.prompt_handoff_token_id
        } else {
            self.last_fixture_feedback_token_id.ok_or_else(|| {
                "later fixture decode position lacks committed feedback prerequisite".to_owned()
            })?
        };
        if step.input_token_id != expected_input {
            return Err(format!(
                "fixture step input {} violates feedback prerequisite {}; no implicit token feedback is allowed",
                step.input_token_id, expected_input
            ));
        }

        let mut next = self.clone();
        let mut causal_update_order = Vec::with_capacity(DELTANET_LAYERS * 2 + GQA_LAYERS * 3 + 1);
        for layer in &plan.layers {
            match layer.source_mixer {
                LayerKind::DeltaNet => {
                    let slot = layer.linear_state_slot.expect("validated DeltaNet slot");
                    next.commit_state_record(
                        StateDomain::DeltaNetConvHistory,
                        slot,
                        step,
                        "deltanet.causal_conv.consume_prior_history_then_commit_current",
                    )?;
                    causal_update_order.push(format!(
                        "layer-{}:deltanet.causal_conv.consume_prior_history_then_commit_current",
                        layer.layer
                    ));
                    next.commit_state_record(
                        StateDomain::DeltaNetRecurrent,
                        slot,
                        step,
                        "deltanet.recurrent.consume_then_commit_current",
                    )?;
                    causal_update_order.push(format!(
                        "layer-{}:deltanet.recurrent.consume_then_commit_current",
                        layer.layer
                    ));
                }
                LayerKind::Gqa => {
                    let slot = layer.gqa_kv_slot.expect("validated GQA slot");
                    next.commit_state_record(
                        StateDomain::GqaKeyCache,
                        slot,
                        step,
                        "gqa.kv.append_key_at_current_position_before_causal_read",
                    )?;
                    causal_update_order.push(format!(
                        "layer-{}:gqa.kv.append_key_at_current_position_before_causal_read",
                        layer.layer
                    ));
                    next.commit_state_record(
                        StateDomain::GqaValueCache,
                        slot,
                        step,
                        "gqa.kv.append_value_at_current_position_before_causal_read",
                    )?;
                    causal_update_order.push(format!(
                        "layer-{}:gqa.kv.append_value_at_current_position_before_causal_read",
                        layer.layer
                    ));
                    causal_update_order.push(format!(
                        "layer-{}:gqa.causal_read_positions_0_through_{}",
                        layer.layer, step.position
                    ));
                }
            }
        }
        next.last_fixture_feedback_token_id = Some(step.selected_feedback_token_id);
        next.position += 1;
        next.validate(plan)?;
        causal_update_order.push("feedback.commit_after_all_48_hybrid_layer_state_updates".into());
        let trace = FixtureDecodeTrace {
            fixture_only: true,
            position: step.position,
            input_token_id: step.input_token_id,
            selected_feedback_token_id: step.selected_feedback_token_id,
            state_records_updated: next.state_records.len(),
            causal_update_order,
            feedback_committed_only_after_full_hybrid_state_update: true,
        };
        *self = next;
        Ok(trace)
    }

    fn commit_state_record(
        &mut self,
        domain: StateDomain,
        slot: usize,
        step: FixtureDecodeStep,
        phase: &'static str,
    ) -> Result<(), String> {
        let record = self
            .state_records
            .iter_mut()
            .find(|record| record.domain == domain && record.slot == slot)
            .ok_or_else(|| format!("missing state record {}:{slot}", domain.as_str()))?;
        if record.committed_positions != step.position {
            return Err(format!(
                "state record {}:{slot} is at revision {}, expected {} before causal update",
                domain.as_str(),
                record.committed_positions,
                step.position
            ));
        }
        let previous_commitment = record.content_commitment_sha256.clone();
        record.content_commitment_sha256 = sha256_json(&serde_json::json!({
            "kind": "qwen80_decode_state_contract_fixture_transition",
            "previous_commitment_sha256": previous_commitment,
            "session_id": self.session_id,
            "domain": domain,
            "slot": slot,
            "owner_layer": record.owner_layer,
            "position": step.position,
            "input_token_id": step.input_token_id,
            "selected_feedback_token_id": step.selected_feedback_token_id,
            "causal_phase": phase,
        }));
        record.committed_positions += 1;
        Ok(())
    }
}

fn snapshot_identity(snapshot: &SessionSnapshot) -> String {
    sha256_json(&SnapshotPreimage {
        snapshot_schema: snapshot.snapshot_schema,
        session: &snapshot.session,
    })
}

fn validate_snapshot(
    plan: &HybridDecodeStatePlan,
    snapshot: &SessionSnapshot,
) -> Result<(), String> {
    if snapshot.snapshot_schema != "hawking.ascension.qwen80_decode_state_snapshot.v1" {
        return Err("snapshot schema is not the decode-state contract schema".into());
    }
    snapshot.session.validate(plan)?;
    if snapshot.snapshot_identity_sha256 != snapshot_identity(snapshot) {
        return Err("snapshot identity does not match its session state".into());
    }
    Ok(())
}

fn validate_cross_session_isolation(
    plan: &HybridDecodeStatePlan,
    left: &DecodeSession,
    right: &DecodeSession,
) -> Result<(), String> {
    left.validate(plan)?;
    right.validate(plan)?;
    if left.session_id == right.session_id {
        return Err("two logical sessions must not share a session identity".into());
    }
    let left_storage: BTreeSet<&str> = left
        .state_records
        .iter()
        .map(|record| record.storage_identity.as_str())
        .collect();
    for record in &right.state_records {
        if left_storage.contains(record.storage_identity.as_str()) {
            return Err(format!(
                "cross-session state storage aliasing detected at {}",
                record.storage_identity
            ));
        }
    }
    Ok(())
}

fn source_archaeology(plan: &HybridDecodeStatePlan) -> SourceArchaeology {
    SourceArchaeology {
        source_module: SOURCE_MODULE,
        source_repository: SOURCE_REPOSITORY,
        source_revision: SOURCE_REVISION,
        model_id: MODEL_ID,
        source_schedule_authority:
            "hawking_core::model::qwen80_complete_runtime::qwen80_layer_kind",
        source_schedule_checked_via_public_resolver: true,
        layer_count: plan.layers.len(),
        deltanet_layers: plan
            .layers
            .iter()
            .filter(|layer| layer.source_mixer == LayerKind::DeltaNet)
            .count(),
        gqa_layers: plan
            .layers
            .iter()
            .filter(|layer| layer.source_mixer == LayerKind::Gqa)
            .count(),
        schedule: plan.layers.iter().map(|layer| layer.source_mixer).collect(),
        delta_conv_history_per_linear_layer: linear_conv_history_shape(),
        delta_recurrent_per_linear_layer: linear_recurrent_shape(),
        gqa_key_value_per_gqa_layer: gqa_cache_shape(plan.max_seq_len),
        source_tokenizer_vocab: TOKENIZER_VOCAB,
        lm_head_rows: LM_HEAD_VOCAB,
        reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
    }
}

fn state_geometry(plan: &HybridDecodeStatePlan) -> Result<StateGeometryReport, String> {
    let conv_elements = product(&linear_conv_history_shape())?;
    let recurrent_elements = product(&linear_recurrent_shape())?;
    let kv_elements = product(&gqa_cache_shape(plan.max_seq_len))?;
    Ok(StateGeometryReport {
        per_session_state_records: DELTANET_LAYERS * 2 + GQA_LAYERS * 2,
        linear_state_slots: DELTANET_LAYERS,
        gqa_kv_slots: GQA_LAYERS,
        total_delta_conv_history_elements: DELTANET_LAYERS * conv_elements,
        total_delta_recurrent_elements: DELTANET_LAYERS * recurrent_elements,
        total_gqa_key_cache_elements: GQA_LAYERS * kv_elements,
        total_gqa_value_cache_elements: GQA_LAYERS * kv_elements,
        state_content_materialized: false,
    })
}

fn contract_checks(plan: &HybridDecodeStatePlan) -> Result<ContractChecks, String> {
    validate_plan(plan)?;
    let mut primary = DecodeSession::new(plan, "fixture-primary", 42)?;
    let baseline = primary.snapshot(plan)?;
    let first = primary.apply_fixture_decode(
        plan,
        FixtureDecodeStep {
            position: 0,
            input_token_id: 42,
            selected_feedback_token_id: 43,
        },
    )?;
    if first.state_records_updated != DELTANET_LAYERS * 2 + GQA_LAYERS * 2
        || !first
            .causal_update_order
            .last()
            .is_some_and(|event| event == "feedback.commit_after_all_48_hybrid_layer_state_updates")
    {
        return Err("fixture causal update ordering did not cover all hybrid state records".into());
    }
    let after_one = primary.snapshot(plan)?;
    let restarted = DecodeSession::restart_from_snapshot(plan, &after_one)?;
    let restart_identity_matches =
        restarted.snapshot(plan)?.snapshot_identity_sha256 == after_one.snapshot_identity_sha256;
    primary.apply_fixture_decode(
        plan,
        FixtureDecodeStep {
            position: 1,
            input_token_id: 43,
            selected_feedback_token_id: 44,
        },
    )?;
    primary.rollback_to(plan, &baseline)?;
    let rollback_identity_matches =
        primary.snapshot(plan)?.snapshot_identity_sha256 == baseline.snapshot_identity_sha256;

    let mut wrong_kind = plan.clone();
    wrong_kind.layers[0].source_mixer = LayerKind::Gqa;
    let wrong_layer_kind_rejected = validate_plan(&wrong_kind).is_err();

    let mut wrong_shape = plan.clone();
    wrong_shape.layers[0].linear_conv_history_shape = Some(vec![1, 1]);
    let wrong_state_shape_rejected = validate_plan(&wrong_shape).is_err();

    let mut reused_slot = plan.clone();
    reused_slot.layers[1].linear_state_slot = reused_slot.layers[0].linear_state_slot;
    let slot_reuse_rejected = validate_plan(&reused_slot).is_err();

    let left = DecodeSession::new(plan, "fixture-left", 17)?;
    let mut right = DecodeSession::new(plan, "fixture-right", 17)?;
    right.state_records[0].storage_identity = left.state_records[0].storage_identity.clone();
    let cross_session_leakage_rejected =
        validate_cross_session_isolation(plan, &left, &right).is_err();

    let tail_token_rejected = DecodeSession::new(plan, "fixture-tail", TOKENIZER_VOCAB).is_err()
        && DecodeSession::new(plan, "fixture-tail-step", 9)
            .and_then(|mut session| {
                session.apply_fixture_decode(
                    plan,
                    FixtureDecodeStep {
                        position: 0,
                        input_token_id: 9,
                        selected_feedback_token_id: TOKENIZER_VOCAB,
                    },
                )
            })
            .is_err();

    let feedback_prerequisite_checked = DecodeSession::new(plan, "fixture-feedback", 8)
        .and_then(|mut session| {
            session.apply_fixture_decode(
                plan,
                FixtureDecodeStep {
                    position: 0,
                    input_token_id: 8,
                    selected_feedback_token_id: 9,
                },
            )?;
            session.apply_fixture_decode(
                plan,
                FixtureDecodeStep {
                    position: 1,
                    input_token_id: 8,
                    selected_feedback_token_id: 10,
                },
            )
        })
        .is_err();

    Ok(ContractChecks {
        exact_schedule_checked: true,
        layer_state_owner_checked: true,
        state_slot_aliasing_checked: true,
        state_shape_checked: true,
        causal_position_and_update_order_checked: true,
        fixture_feedback_prerequisite_checked: feedback_prerequisite_checked,
        restart_identity_checked: restart_identity_matches,
        rollback_identity_checked: rollback_identity_matches,
        wrong_layer_kind_rejected,
        wrong_state_shape_rejected,
        slot_reuse_rejected,
        cross_session_leakage_rejected,
        lm_head_reserved_tail_token_rejected: tail_token_rejected,
    })
}

fn rawls_state_kv_handoff() -> Vec<RawlsStateKvGap> {
    vec![
        RawlsStateKvGap {
            gap_id: "device_resident_session_local_state_allocation",
            required_before_complete_decoder:
                "Allocate distinct device-resident DeltaNet conv [8192,3] and recurrent [32,128,128] buffers for all 36 linear layers, plus distinct K/V [context,2,256] buffers for all 12 GQA layers, per logical session.",
            required_evidence:
                "A source-bound complete-decoder capture must prove buffer identities are unique within and across sessions; this contract's string identities are not allocation evidence.",
            claim_boundary:
                "No native state allocation, multi-session residency, or decoder claim is earned here.",
        },
        RawlsStateKvGap {
            gap_id: "source_ordered_deltanet_state_transition",
            required_before_complete_decoder:
                "Connect admitted direct-packed QKVZ/BA, causal convolution, DeltaNet recurrence, gated norm, projection, and residual so each of the 36 source-selected layers consumes prior state then commits its current state exactly once per decode position.",
            required_evidence:
                "CPU/device parity must cover state bytes and output boundaries for more than the existing isolated component; fixture SHA commitments are not numerical parity.",
            claim_boundary:
                "No complete linear layer or token is executed by this contract.",
        },
        RawlsStateKvGap {
            gap_id: "gqa_kv_rope_causal_append_and_read",
            required_before_complete_decoder:
                "Implement source-bound Q/K/V, q/k norm, RoPE, append current K/V at the GQA layer's own slot, causal read of only positions 0..current, output projection, gate, and residual for layers 3,7,...,47.",
            required_evidence:
                "Device parity must prove current-position inclusion, future-position exclusion, and no cache alias between the 12 GQA layers or sessions.",
            claim_boundary:
                "No GQA token path, KV correctness, or context-length capability is earned here.",
        },
        RawlsStateKvGap {
            gap_id: "transactional_checkpoint_restart_and_rollback",
            required_before_complete_decoder:
                "Back logical checkpoint identity with actual state/KV bytes, including atomic commit after all 48 layer updates and restore without session contamination.",
            required_evidence:
                "Repeatable multi-position CPU/device state hashes before restart and after rollback, bound to admitted artifact and session IDs.",
            claim_boundary:
                "This contract validates only a logical fixture snapshot, not state persistence.",
        },
        RawlsStateKvGap {
            gap_id: "real_autoregressive_feedback_boundary",
            required_before_complete_decoder:
                "Bind tokenizer/chat-template handoff, reserved-tail masking, final norm/lm-head/sampler, and the selected valid token into the next decode position only after a legitimate 48-layer state commit.",
            required_evidence:
                "A full admitted-token trace showing input ID, all hybrid layer state updates, masked sampler selection, and next-position feedback without an HCLI/TPS promotion until separately qualified.",
            claim_boundary:
                "The fixture accepts IDs but neither tokenizes, samples, nor generates a model token.",
        },
    ]
}

fn report(max_seq_len: usize) -> Result<ReadinessReport, String> {
    let plan = expected_plan(max_seq_len)?;
    validate_plan(&plan)?;
    let mut report = ReadinessReport {
        schema: SCHEMA,
        status: STATUS,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        source_archaeology: source_archaeology(&plan),
        state_geometry: state_geometry(&plan)?,
        fixture_contract_checks: contract_checks(&plan)?,
        rawls_state_kv_handoff: rawls_state_kv_handoff(),
        execution_boundary: ExecutionBoundary {
            not_runtime: true,
            no_live_artifact_scan: true,
            no_packed_tensor_read: true,
            no_metal_device_or_dispatch: true,
            no_model_token_execution: true,
            no_logit_or_sampler_execution: true,
            no_hcli_execution: true,
            no_tps_or_tg_measurement: true,
            no_server_started: true,
        },
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = sha256_json(&serde_json::json!({
        "schema": report.schema,
        "status": report.status,
        "complete_decoder_readiness_earned": report.complete_decoder_readiness_earned,
        "real_gravity_server_launch_precondition_satisfied": report.real_gravity_server_launch_precondition_satisfied,
        "source_archaeology": report.source_archaeology,
        "state_geometry": report.state_geometry,
        "fixture_contract_checks": report.fixture_contract_checks,
        "rawls_state_kv_handoff": report.rawls_state_kv_handoff,
        "execution_boundary": report.execution_boundary,
    }));
    Ok(report)
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_decode_state_contract [--max-seq-len POSITIVE] [--out ABSOLUTE_PATH]"
}

fn parse_arguments() -> Result<Arguments, String> {
    let mut max_seq_len = 16usize;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
        match flag.as_str() {
            "--max-seq-len" => {
                max_seq_len = value
                    .parse::<usize>()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| format!("--max-seq-len must be positive; {}", usage()))?;
            }
            "--out" => {
                let path = PathBuf::from(value);
                if !path.is_absolute() {
                    return Err("--out must be an absolute path".into());
                }
                if out.replace(path).is_some() {
                    return Err(format!("--out was supplied more than once; {}", usage()));
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    if max_seq_len > QWEN80_COMPLETE_NATIVE_MAX_CONTEXT {
        return Err(format!(
            "--max-seq-len exceeds {QWEN80_COMPLETE_NATIVE_MAX_CONTEXT}; {}",
            usage()
        ));
    }
    Ok(Arguments { max_seq_len, out })
}

fn main() {
    let arguments = parse_arguments().unwrap_or_else(|error| {
        eprintln!("Qwen80 decode-state contract refused: {error}");
        process::exit(2);
    });
    let result = report(arguments.max_seq_len).unwrap_or_else(|error| {
        eprintln!("Qwen80 decode-state contract refused: {error}");
        process::exit(2);
    });
    let rendered =
        serde_json::to_string_pretty(&result).expect("report serialization must succeed") + "\n";
    if let Some(path) = arguments.out {
        fs::write(&path, rendered.as_bytes()).unwrap_or_else(|error| {
            eprintln!(
                "Qwen80 decode-state contract refused to write {}: {error}",
                path.display()
            );
            process::exit(2);
        });
    } else {
        print!("{rendered}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_48_layer_source_schedule_has_distinct_state_slots() {
        let plan = expected_plan(8).unwrap();
        validate_plan(&plan).unwrap();
        assert_eq!(plan.layers.len(), 48);
        assert_eq!(
            plan.layers
                .iter()
                .filter(|layer| layer.source_mixer == LayerKind::DeltaNet)
                .count(),
            36
        );
        assert_eq!(
            plan.layers
                .iter()
                .filter(|layer| layer.source_mixer == LayerKind::Gqa)
                .count(),
            12
        );
        assert_eq!(plan.layers[0].source_mixer, LayerKind::DeltaNet);
        assert_eq!(plan.layers[3].source_mixer, LayerKind::Gqa);
        assert_eq!(plan.layers[47].source_mixer, LayerKind::Gqa);
        assert_eq!(plan.layers[46].linear_state_slot, Some(35));
        assert_eq!(plan.layers[47].gqa_kv_slot, Some(11));
    }

    #[test]
    fn rejects_wrong_layer_kind_against_source_resolver() {
        let mut plan = expected_plan(8).unwrap();
        plan.layers[0].source_mixer = LayerKind::Gqa;
        assert!(validate_plan(&plan).is_err());
    }

    #[test]
    fn rejects_wrong_delta_or_gqa_state_shape() {
        let mut delta_plan = expected_plan(8).unwrap();
        delta_plan.layers[0].linear_recurrent_shape = Some(vec![32, 127, 128]);
        assert!(validate_plan(&delta_plan).is_err());

        let mut gqa_plan = expected_plan(8).unwrap();
        gqa_plan.layers[3].gqa_key_cache_shape = Some(vec![8, 2, 255]);
        assert!(validate_plan(&gqa_plan).is_err());
    }

    #[test]
    fn rejects_slot_reuse_or_wrong_family_ownership() {
        let mut reused = expected_plan(8).unwrap();
        reused.layers[1].linear_state_slot = reused.layers[0].linear_state_slot;
        assert!(validate_plan(&reused).is_err());

        let mut crossed = expected_plan(8).unwrap();
        crossed.layers[3].linear_state_slot = Some(0);
        assert!(validate_plan(&crossed).is_err());
    }

    #[test]
    fn fixture_decode_requires_causal_feedback_and_updates_every_slot() {
        let plan = expected_plan(4).unwrap();
        let mut session = DecodeSession::new(&plan, "causal", 11).unwrap();
        let first = session
            .apply_fixture_decode(
                &plan,
                FixtureDecodeStep {
                    position: 0,
                    input_token_id: 11,
                    selected_feedback_token_id: 12,
                },
            )
            .unwrap();
        assert_eq!(first.state_records_updated, 96);
        let gqa_key = first
            .causal_update_order
            .iter()
            .position(|event| event.contains("layer-3:gqa.kv.append_key"))
            .unwrap();
        let gqa_read = first
            .causal_update_order
            .iter()
            .position(|event| event == "layer-3:gqa.causal_read_positions_0_through_0")
            .unwrap();
        assert!(gqa_key < gqa_read);
        assert!(session
            .apply_fixture_decode(
                &plan,
                FixtureDecodeStep {
                    position: 1,
                    input_token_id: 11,
                    selected_feedback_token_id: 13,
                },
            )
            .is_err());
        session
            .apply_fixture_decode(
                &plan,
                FixtureDecodeStep {
                    position: 1,
                    input_token_id: 12,
                    selected_feedback_token_id: 13,
                },
            )
            .unwrap();
    }

    #[test]
    fn rejects_reserved_lm_head_tail_for_prompt_or_feedback() {
        let plan = expected_plan(4).unwrap();
        assert!(DecodeSession::new(&plan, "tail-prompt", TOKENIZER_VOCAB).is_err());
        let mut session = DecodeSession::new(&plan, "tail-feedback", 1).unwrap();
        assert!(session
            .apply_fixture_decode(
                &plan,
                FixtureDecodeStep {
                    position: 0,
                    input_token_id: 1,
                    selected_feedback_token_id: TOKENIZER_VOCAB,
                },
            )
            .is_err());
    }

    #[test]
    fn rejects_cross_session_storage_leakage() {
        let plan = expected_plan(4).unwrap();
        let left = DecodeSession::new(&plan, "left", 2).unwrap();
        let right = DecodeSession::new(&plan, "right", 2).unwrap();
        validate_cross_session_isolation(&plan, &left, &right).unwrap();
        let mut leaked = right.clone();
        leaked.state_records[0].storage_identity = left.state_records[0].storage_identity.clone();
        assert!(validate_cross_session_isolation(&plan, &left, &leaked).is_err());
    }

    #[test]
    fn restart_and_rollback_preserve_exact_snapshot_identity() {
        let plan = expected_plan(4).unwrap();
        let mut session = DecodeSession::new(&plan, "restart", 5).unwrap();
        session
            .apply_fixture_decode(
                &plan,
                FixtureDecodeStep {
                    position: 0,
                    input_token_id: 5,
                    selected_feedback_token_id: 6,
                },
            )
            .unwrap();
        let checkpoint = session.snapshot(&plan).unwrap();
        let restarted = DecodeSession::restart_from_snapshot(&plan, &checkpoint).unwrap();
        assert_eq!(
            restarted.snapshot(&plan).unwrap().snapshot_identity_sha256,
            checkpoint.snapshot_identity_sha256
        );
        session
            .apply_fixture_decode(
                &plan,
                FixtureDecodeStep {
                    position: 1,
                    input_token_id: 6,
                    selected_feedback_token_id: 7,
                },
            )
            .unwrap();
        session.rollback_to(&plan, &checkpoint).unwrap();
        assert_eq!(
            session.snapshot(&plan).unwrap().snapshot_identity_sha256,
            checkpoint.snapshot_identity_sha256
        );
    }

    #[test]
    fn machine_readable_report_remains_explicitly_not_runtime() {
        let result = report(4).unwrap();
        assert_eq!(result.status, STATUS);
        assert!(!result.complete_decoder_readiness_earned);
        assert!(result.execution_boundary.not_runtime);
        assert!(result.execution_boundary.no_model_token_execution);
        assert!(result.execution_boundary.no_hcli_execution);
        assert!(result.execution_boundary.no_tps_or_tg_measurement);
        assert!(result.fixture_contract_checks.wrong_layer_kind_rejected);
        assert!(
            result
                .fixture_contract_checks
                .cross_session_leakage_rejected
        );
    }
}
