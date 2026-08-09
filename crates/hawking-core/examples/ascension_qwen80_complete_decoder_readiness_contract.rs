//! Fail-closed Qwen3-Coder-Next complete-decoder readiness contract.
//!
//! This is deliberately a *ledger contract*, not another runtime.  It reads a
//! caller-supplied, sealed descriptor inventory and checks whether the exact
//! admitted Qwen80 artifact has device/full-path evidence for every operator
//! that a 48-layer hybrid decoder needs.  It never opens the Gravity payload,
//! source weights, a live receipt, a Metal device, a watcher, a server, or an
//! HCLI endpoint.
//!
//! The contract's narrow job is to prevent component receipts from being
//! promoted into a decoder claim.  A descriptor is usable only when it binds
//! the current artifact/source admission, records actual Metal parity over a
//! complete-token path, and is explicitly neither synthetic nor component
//! only.  Partial evidence remains visible in the report but earns nothing.
//!
//! The expected input shape is:
//! ```text
//! {
//!   "schema": "hawking.ascension.qwen80_complete_decoder_readiness_input.v1",
//!   "source_artifact_binding": { ... admitted Qwen80 identity ... },
//!   "layer_schedule": [{"layer":0,"mixer":"deltanet"}, ...],
//!   "sealed_component_ledgers": [{ ... one sealed descriptor per coverage ... }]
//! }
//! ```
//!
//! Run against a prepared descriptor inventory:
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_complete_decoder_readiness_contract -- \
//!   --input /absolute/path/QWEN80_COMPLETE_DECODER_LEDGER_INVENTORY.json \
//!   --out /absolute/path/QWEN80_COMPLETE_DECODER_READINESS.json
//! ```
//!
//! `--current-evidence` emits the deliberately incomplete current-evidence
//! assessment without scanning any live path.  An incomplete inventory writes
//! the report and then exits non-zero, so it cannot be mistaken for readiness.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.qwen80_complete_decoder_readiness_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_complete_decoder_readiness_result.v1";
const DESCRIPTOR_SCHEMA: &str =
    "hawking.ascension.qwen80_complete_decoder_component_ledger_descriptor.v1";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const LAYER_COUNT: usize = 48;
const TOP_K: usize = 10;
const COMPLETE_TENSOR_COUNT: usize = 74_391;
const COMPLETE_PAYLOAD_BYTES: u64 = 11_207_187_116;

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SourceArtifactBinding {
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_schema: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_audit_seal_sha256: String,
    complete_tensor_count: usize,
    complete_payload_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct LayerScheduleEntry {
    layer: usize,
    mixer: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DeviceExecutionDescriptor {
    backend: String,
    device_dispatches: usize,
    source_bound: bool,
    artifact_bound: bool,
    full_path: bool,
    complete_token_path: bool,
    device_parity_passed: bool,
    fixture_only: bool,
    synthetic_input: bool,
    component_only: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SealedComponentLedgerDescriptor {
    descriptor_schema: String,
    ledger_id: String,
    ledger_receipt_seal_sha256: String,
    model_id: String,
    model_key: String,
    source_repository: String,
    source_revision: String,
    manifest_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    operator_class: String,
    scope: String,
    covered_layers: Vec<usize>,
    covered_route_slots: Vec<usize>,
    device_execution: DeviceExecutionDescriptor,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ReadinessInput {
    schema: String,
    source_artifact_binding: SourceArtifactBinding,
    layer_schedule: Vec<LayerScheduleEntry>,
    sealed_component_ledgers: Vec<SealedComponentLedgerDescriptor>,
}

#[derive(Clone, Debug)]
enum CoverageRequirement {
    Global,
    PerLayer {
        layers: Vec<usize>,
        route_slots: Option<Vec<usize>>,
    },
}

#[derive(Clone, Debug)]
struct RequiredOperator {
    operator_class: &'static str,
    coverage: CoverageRequirement,
}

#[derive(Serialize)]
struct RejectedDescriptor {
    index: usize,
    ledger_id: String,
    operator_class: String,
    reasons: Vec<String>,
}

#[derive(Clone, Serialize)]
struct OperatorCoverage {
    operator_class: String,
    scope: String,
    required_global: bool,
    required_layers: Vec<usize>,
    required_route_slots: Vec<usize>,
    covered_global: bool,
    covered_layers: Vec<usize>,
    missing_layers: Vec<usize>,
    missing_route_slots_by_layer: BTreeMap<usize, Vec<usize>>,
    satisfied: bool,
}

#[derive(Serialize)]
struct ReadinessReport {
    schema: &'static str,
    status: &'static str,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    input_schema_valid: bool,
    source_artifact_binding_valid: bool,
    exact_48_layer_schedule_valid: bool,
    contract_errors: Vec<String>,
    source_artifact_binding: SourceArtifactBinding,
    expected_48_layer_schedule: Vec<LayerScheduleEntry>,
    received_sealed_component_ledger_descriptors: usize,
    valid_full_path_device_ledger_descriptors: usize,
    rejected_descriptors: Vec<RejectedDescriptor>,
    operator_coverage: Vec<OperatorCoverage>,
    missing_operator_classes_or_layers: Vec<OperatorCoverage>,
    read_only_contract: bool,
    live_artifact_scan_performed: bool,
    metal_device_or_dispatch_performed: bool,
    model_execution_performed: bool,
    hcli_execution_performed: bool,
    tps_or_tg_measurement_performed: bool,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

enum InputMode {
    Inventory(PathBuf),
    CurrentEvidence,
}

struct Arguments {
    input_mode: InputMode,
    out: PathBuf,
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn has_duplicates(values: &[usize]) -> bool {
    values.iter().collect::<BTreeSet<_>>().len() != values.len()
}

fn expected_schedule() -> Vec<LayerScheduleEntry> {
    (0..LAYER_COUNT)
        .map(|layer| LayerScheduleEntry {
            layer,
            mixer: if layer % 4 == 3 {
                "gqa".into()
            } else {
                "deltanet".into()
            },
        })
        .collect()
}

fn deltanet_layers() -> Vec<usize> {
    (0..LAYER_COUNT).filter(|layer| layer % 4 != 3).collect()
}

fn gqa_layers() -> Vec<usize> {
    (0..LAYER_COUNT).filter(|layer| layer % 4 == 3).collect()
}

fn all_layers() -> Vec<usize> {
    (0..LAYER_COUNT).collect()
}

fn all_route_slots() -> Vec<usize> {
    (0..TOP_K).collect()
}

fn requirements() -> Vec<RequiredOperator> {
    let all_layers = all_layers();
    let all_routes = all_route_slots();
    vec![
        RequiredOperator {
            operator_class: "embedding",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "input_norm",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "deltanet_mixer",
            coverage: CoverageRequirement::PerLayer {
                layers: deltanet_layers(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "gqa_mixer",
            coverage: CoverageRequirement::PerLayer {
                layers: gqa_layers(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "post_norm",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "router_top10",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "expert_gather",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: Some(all_routes.clone()),
            },
        },
        RequiredOperator {
            operator_class: "routed_gate_up",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: Some(all_routes.clone()),
            },
        },
        RequiredOperator {
            operator_class: "routed_activation",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: Some(all_routes.clone()),
            },
        },
        RequiredOperator {
            operator_class: "routed_down",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: Some(all_routes.clone()),
            },
        },
        RequiredOperator {
            operator_class: "shared_expert",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "route_combine",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: Some(all_routes.clone()),
            },
        },
        RequiredOperator {
            operator_class: "residual",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "deltanet_recurrent_state",
            coverage: CoverageRequirement::PerLayer {
                layers: deltanet_layers(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "gqa_kv_state",
            coverage: CoverageRequirement::PerLayer {
                layers: gqa_layers(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "final_norm",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "lm_head",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "tail_mask",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "sampler",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "tokenizer_template",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "full_layer_harness",
            coverage: CoverageRequirement::PerLayer {
                layers: all_layers.clone(),
                route_slots: None,
            },
        },
        RequiredOperator {
            operator_class: "command_graph_48_layer",
            coverage: CoverageRequirement::Global,
        },
        RequiredOperator {
            operator_class: "hcli_adapter",
            coverage: CoverageRequirement::Global,
        },
    ]
}

fn requirement_for(operator_class: &str) -> Option<RequiredOperator> {
    requirements()
        .into_iter()
        .find(|requirement| requirement.operator_class == operator_class)
}

fn scope_name(coverage: &CoverageRequirement) -> &'static str {
    match coverage {
        CoverageRequirement::Global => "global",
        CoverageRequirement::PerLayer {
            route_slots: Some(_),
            ..
        } => "per_layer_all_top10_routes",
        CoverageRequirement::PerLayer {
            route_slots: None, ..
        } => "per_layer",
    }
}

fn validate_source_artifact_binding(binding: &SourceArtifactBinding) -> Vec<String> {
    let mut errors = Vec::new();
    for (field, observed, expected) in [
        ("model_id", binding.model_id.as_str(), MODEL_ID),
        ("model_key", binding.model_key.as_str(), MODEL_KEY),
        (
            "source_repository",
            binding.source_repository.as_str(),
            SOURCE_REPOSITORY,
        ),
        (
            "source_revision",
            binding.source_revision.as_str(),
            SOURCE_REVISION,
        ),
        (
            "manifest_schema",
            binding.manifest_schema.as_str(),
            MANIFEST_SCHEMA,
        ),
    ] {
        if observed != expected {
            errors.push(format!(
                "source_artifact_binding.{field}={observed:?}, expected {expected:?}"
            ));
        }
    }
    for (field, value) in [
        (
            "manifest_seal_sha256",
            binding.manifest_seal_sha256.as_str(),
        ),
        (
            "admission_receipt_seal_sha256",
            binding.admission_receipt_seal_sha256.as_str(),
        ),
        (
            "source_audit_seal_sha256",
            binding.source_audit_seal_sha256.as_str(),
        ),
    ] {
        if !is_lower_sha256(value) {
            errors.push(format!(
                "source_artifact_binding.{field} is not a lowercase SHA-256"
            ));
        }
    }
    if binding.complete_tensor_count != COMPLETE_TENSOR_COUNT {
        errors.push(format!(
            "source_artifact_binding.complete_tensor_count={} does not bind the exact admitted Qwen80 count {COMPLETE_TENSOR_COUNT}",
            binding.complete_tensor_count
        ));
    }
    if binding.complete_payload_bytes != COMPLETE_PAYLOAD_BYTES {
        errors.push(format!(
            "source_artifact_binding.complete_payload_bytes={} does not bind the exact admitted Qwen80 payload {COMPLETE_PAYLOAD_BYTES}",
            binding.complete_payload_bytes
        ));
    }
    errors
}

fn validate_schedule(schedule: &[LayerScheduleEntry]) -> Vec<String> {
    let expected = expected_schedule();
    let mut errors = Vec::new();
    if schedule.len() != expected.len() {
        errors.push(format!(
            "layer_schedule has {} entries, expected exactly {}",
            schedule.len(),
            expected.len()
        ));
    }
    for (index, expected_entry) in expected.iter().enumerate() {
        match schedule.get(index) {
            Some(observed) if observed == expected_entry => {}
            Some(observed) => errors.push(format!(
                "layer_schedule[{index}]={observed:?}, expected {expected_entry:?} for exact 3xDeltaNet/1xGQA cadence"
            )),
            None => errors.push(format!(
                "layer_schedule is missing layer {} ({})",
                expected_entry.layer, expected_entry.mixer
            )),
        }
    }
    errors
}

fn validate_descriptor(
    descriptor: &SealedComponentLedgerDescriptor,
    binding: &SourceArtifactBinding,
) -> Vec<String> {
    let mut errors = Vec::new();
    if descriptor.descriptor_schema != DESCRIPTOR_SCHEMA {
        errors.push(format!(
            "descriptor_schema={:?}, expected {DESCRIPTOR_SCHEMA:?}",
            descriptor.descriptor_schema
        ));
    }
    if descriptor.ledger_id.trim().is_empty() {
        errors.push("ledger_id must be non-empty".into());
    }
    if !is_lower_sha256(&descriptor.ledger_receipt_seal_sha256) {
        errors.push("ledger_receipt_seal_sha256 is not a lowercase SHA-256".into());
    }
    for (field, observed, expected) in [
        (
            "model_id",
            descriptor.model_id.as_str(),
            binding.model_id.as_str(),
        ),
        (
            "model_key",
            descriptor.model_key.as_str(),
            binding.model_key.as_str(),
        ),
        (
            "source_repository",
            descriptor.source_repository.as_str(),
            binding.source_repository.as_str(),
        ),
        (
            "source_revision",
            descriptor.source_revision.as_str(),
            binding.source_revision.as_str(),
        ),
        (
            "manifest_seal_sha256",
            descriptor.manifest_seal_sha256.as_str(),
            binding.manifest_seal_sha256.as_str(),
        ),
        (
            "admission_receipt_seal_sha256",
            descriptor.admission_receipt_seal_sha256.as_str(),
            binding.admission_receipt_seal_sha256.as_str(),
        ),
    ] {
        if observed != expected {
            errors.push(format!(
                "{field}={observed:?} is not bound to current artifact"
            ));
        }
    }
    let Some(requirement) = requirement_for(&descriptor.operator_class) else {
        errors.push(format!(
            "operator_class={:?} is not a required Qwen80 decoder operator",
            descriptor.operator_class
        ));
        return errors;
    };
    let expected_scope = scope_name(&requirement.coverage);
    if descriptor.scope != expected_scope {
        errors.push(format!(
            "scope={:?}, expected {expected_scope:?} for {}",
            descriptor.scope, descriptor.operator_class
        ));
    }
    if has_duplicates(&descriptor.covered_layers) {
        errors.push("covered_layers contains duplicate entries".into());
    }
    if has_duplicates(&descriptor.covered_route_slots) {
        errors.push("covered_route_slots contains duplicate entries".into());
    }
    match &requirement.coverage {
        CoverageRequirement::Global => {
            if !descriptor.covered_layers.is_empty() || !descriptor.covered_route_slots.is_empty() {
                errors.push("global ledger must not claim layer or route coverage".into());
            }
        }
        CoverageRequirement::PerLayer {
            layers,
            route_slots,
        } => {
            if descriptor.covered_layers.is_empty() {
                errors.push("per-layer ledger has no covered_layers".into());
            }
            if let Some(unexpected_layer) = descriptor
                .covered_layers
                .iter()
                .find(|layer| !layers.contains(layer))
            {
                errors.push(format!(
                    "covered layer {unexpected_layer} is outside the source schedule for {}",
                    descriptor.operator_class
                ));
            }
            match route_slots {
                Some(expected_slots) => {
                    if descriptor.covered_route_slots.is_empty() {
                        errors.push("all-top10-route ledger has no covered_route_slots".into());
                    }
                    if let Some(unexpected_slot) = descriptor
                        .covered_route_slots
                        .iter()
                        .find(|slot| !expected_slots.contains(slot))
                    {
                        errors.push(format!(
                            "covered route slot {unexpected_slot} is outside Qwen80 top-{TOP_K}"
                        ));
                    }
                }
                None if !descriptor.covered_route_slots.is_empty() => {
                    errors.push("non-route operator must not claim routed-wave slots".into());
                }
                None => {}
            }
        }
    }
    let execution = &descriptor.device_execution;
    if execution.backend != "metal" {
        errors.push(format!(
            "device_execution.backend={:?}, expected real Metal",
            execution.backend
        ));
    }
    if execution.device_dispatches == 0 {
        errors.push("device_execution.device_dispatches must be non-zero".into());
    }
    for (field, observed) in [
        ("source_bound", execution.source_bound),
        ("artifact_bound", execution.artifact_bound),
        ("full_path", execution.full_path),
        ("complete_token_path", execution.complete_token_path),
        ("device_parity_passed", execution.device_parity_passed),
    ] {
        if !observed {
            errors.push(format!("device_execution.{field} must be true"));
        }
    }
    for (field, observed) in [
        ("fixture_only", execution.fixture_only),
        ("synthetic_input", execution.synthetic_input),
        ("component_only", execution.component_only),
    ] {
        if observed {
            errors.push(format!("device_execution.{field} must be false"));
        }
    }
    errors
}

fn assess_requirement(
    requirement: &RequiredOperator,
    descriptors: &[SealedComponentLedgerDescriptor],
    valid_descriptor_indices: &BTreeSet<usize>,
) -> OperatorCoverage {
    let matching = descriptors
        .iter()
        .enumerate()
        .filter(|(index, descriptor)| {
            valid_descriptor_indices.contains(index)
                && descriptor.operator_class == requirement.operator_class
        })
        .collect::<Vec<_>>();
    match &requirement.coverage {
        CoverageRequirement::Global => {
            let covered_global = matching
                .iter()
                .any(|(_, descriptor)| descriptor.scope == "global");
            OperatorCoverage {
                operator_class: requirement.operator_class.into(),
                scope: scope_name(&requirement.coverage).into(),
                required_global: true,
                required_layers: Vec::new(),
                required_route_slots: Vec::new(),
                covered_global,
                covered_layers: Vec::new(),
                missing_layers: Vec::new(),
                missing_route_slots_by_layer: BTreeMap::new(),
                satisfied: covered_global,
            }
        }
        CoverageRequirement::PerLayer {
            layers,
            route_slots,
        } => {
            let covered_layers = layers
                .iter()
                .copied()
                .filter(|layer| {
                    matching
                        .iter()
                        .any(|(_, descriptor)| descriptor.covered_layers.contains(layer))
                })
                .collect::<Vec<_>>();
            let missing_layers = layers
                .iter()
                .copied()
                .filter(|layer| !covered_layers.contains(layer))
                .collect::<Vec<_>>();
            let mut missing_route_slots_by_layer = BTreeMap::new();
            if let Some(required_slots) = route_slots {
                for &layer in layers {
                    let covered_slots = matching
                        .iter()
                        .filter(|(_, descriptor)| descriptor.covered_layers.contains(&layer))
                        .flat_map(|(_, descriptor)| descriptor.covered_route_slots.iter().copied())
                        .collect::<BTreeSet<_>>();
                    let missing_slots = required_slots
                        .iter()
                        .copied()
                        .filter(|slot| !covered_slots.contains(slot))
                        .collect::<Vec<_>>();
                    if !missing_slots.is_empty() {
                        missing_route_slots_by_layer.insert(layer, missing_slots);
                    }
                }
            }
            let satisfied = missing_layers.is_empty() && missing_route_slots_by_layer.is_empty();
            OperatorCoverage {
                operator_class: requirement.operator_class.into(),
                scope: scope_name(&requirement.coverage).into(),
                required_global: false,
                required_layers: layers.clone(),
                required_route_slots: route_slots.clone().unwrap_or_default(),
                covered_global: false,
                covered_layers,
                missing_layers,
                missing_route_slots_by_layer,
                satisfied,
            }
        }
    }
}

fn evaluate(input: ReadinessInput) -> ReadinessReport {
    let mut contract_errors = Vec::new();
    let input_schema_valid = input.schema == INPUT_SCHEMA;
    if !input_schema_valid {
        contract_errors.push(format!(
            "input schema={:?}, expected {INPUT_SCHEMA:?}",
            input.schema
        ));
    }
    let source_errors = validate_source_artifact_binding(&input.source_artifact_binding);
    let source_artifact_binding_valid = source_errors.is_empty();
    contract_errors.extend(source_errors);
    let schedule_errors = validate_schedule(&input.layer_schedule);
    let exact_48_layer_schedule_valid = schedule_errors.is_empty();
    contract_errors.extend(schedule_errors);

    let mut valid_descriptor_indices = BTreeSet::new();
    let mut rejected_descriptors = Vec::new();
    for (index, descriptor) in input.sealed_component_ledgers.iter().enumerate() {
        let reasons = validate_descriptor(descriptor, &input.source_artifact_binding);
        if reasons.is_empty() {
            valid_descriptor_indices.insert(index);
        } else {
            rejected_descriptors.push(RejectedDescriptor {
                index,
                ledger_id: descriptor.ledger_id.clone(),
                operator_class: descriptor.operator_class.clone(),
                reasons,
            });
        }
    }
    let operator_coverage = requirements()
        .iter()
        .map(|requirement| {
            assess_requirement(
                requirement,
                &input.sealed_component_ledgers,
                &valid_descriptor_indices,
            )
        })
        .collect::<Vec<_>>();
    let missing_operator_classes_or_layers = operator_coverage
        .iter()
        .filter(|coverage| !coverage.satisfied)
        .cloned()
        .collect::<Vec<_>>();
    let complete_decoder_readiness_earned = input_schema_valid
        && source_artifact_binding_valid
        && exact_48_layer_schedule_valid
        && missing_operator_classes_or_layers.is_empty();
    let status = if complete_decoder_readiness_earned {
        "EARNED_QWEN80_COMPLETE_DECODER_READINESS_CONTRACT_ONLY"
    } else {
        "INCOMPLETE_QWEN80_COMPLETE_DECODER_READINESS_NO_DECODER_TOKEN_HCLI_OR_TPS_CLAIM"
    };
    let mut report = ReadinessReport {
        schema: RESULT_SCHEMA,
        status,
        complete_decoder_readiness_earned,
        real_gravity_server_launch_precondition_satisfied: complete_decoder_readiness_earned,
        input_schema_valid,
        source_artifact_binding_valid,
        exact_48_layer_schedule_valid,
        contract_errors,
        source_artifact_binding: input.source_artifact_binding,
        expected_48_layer_schedule: expected_schedule(),
        received_sealed_component_ledger_descriptors: input.sealed_component_ledgers.len(),
        valid_full_path_device_ledger_descriptors: valid_descriptor_indices.len(),
        rejected_descriptors,
        operator_coverage,
        missing_operator_classes_or_layers,
        read_only_contract: true,
        live_artifact_scan_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        claim_boundary: vec![
            "This contract consumes descriptors only. It does not open a packed artifact, source shard, receipt path, watcher, server, HCLI endpoint, or Metal device.",
            "A component, synthetic-input, CPU-only, fixture-only, unsealed, wrong-artifact, or partial-route receipt is deliberately non-promotable and remains incomplete evidence.",
            "Even an earned readiness contract is only a launch precondition for an independently controlled real Gravity-server integration. It does not itself execute a decoder token, HCLI request, TPS/TG measurement, capability test, or tournament action.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 =
        sha256_hex(&serde_json::to_vec(&report).expect("report serializes"));
    report
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn current_source_artifact_binding() -> SourceArtifactBinding {
    SourceArtifactBinding {
        model_id: MODEL_ID.into(),
        model_key: MODEL_KEY.into(),
        source_repository: SOURCE_REPOSITORY.into(),
        source_revision: SOURCE_REVISION.into(),
        manifest_schema: MANIFEST_SCHEMA.into(),
        manifest_seal_sha256: "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b"
            .into(),
        admission_receipt_seal_sha256:
            "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628".into(),
        source_audit_seal_sha256:
            "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4".into(),
        complete_tensor_count: COMPLETE_TENSOR_COUNT,
        complete_payload_bytes: COMPLETE_PAYLOAD_BYTES,
    }
}

fn current_component_descriptor(
    binding: &SourceArtifactBinding,
    ledger_id: &str,
    ledger_receipt_seal_sha256: &str,
    operator_class: &str,
    scope: &str,
    covered_layers: Vec<usize>,
    covered_route_slots: Vec<usize>,
    backend: &str,
    device_dispatches: usize,
    synthetic_input: bool,
    component_only: bool,
) -> SealedComponentLedgerDescriptor {
    SealedComponentLedgerDescriptor {
        descriptor_schema: DESCRIPTOR_SCHEMA.into(),
        ledger_id: ledger_id.into(),
        ledger_receipt_seal_sha256: ledger_receipt_seal_sha256.into(),
        model_id: binding.model_id.clone(),
        model_key: binding.model_key.clone(),
        source_repository: binding.source_repository.clone(),
        source_revision: binding.source_revision.clone(),
        manifest_seal_sha256: binding.manifest_seal_sha256.clone(),
        admission_receipt_seal_sha256: binding.admission_receipt_seal_sha256.clone(),
        operator_class: operator_class.into(),
        scope: scope.into(),
        covered_layers,
        covered_route_slots,
        device_execution: DeviceExecutionDescriptor {
            backend: backend.into(),
            device_dispatches,
            source_bound: true,
            artifact_bound: true,
            full_path: false,
            complete_token_path: false,
            device_parity_passed: backend == "metal",
            fixture_only: synthetic_input,
            synthetic_input,
            component_only,
        },
    }
}

/// A static, descriptor-only representation of the known current component
/// frontier.  It intentionally contains only receipts that remain component
/// or fixture scoped, so the same validator explains precisely why none can
/// activate a Qwen80 decoder/server claim.  It does not inspect those paths.
fn current_evidence_input() -> ReadinessInput {
    let binding = current_source_artifact_binding();
    let postnorm_outer_seal = "1bd9af3b1c38de5a6a63b4f221e0d016a15bdb0e2b15823985d6f6b04c4d32f4";
    let gqa_outer_seal = "b0b16468df4b42ae8b02de076de72850bdd244f9dd488be9432f8e61bbe0a44e";
    let kv_component_seal = "467be55e58bba1d9a5650b209cbd5228256f92e1414a53834afa5d7fe5c8f328";
    // This is the sealed source-token L0 9+14 same-TCB record.  Its route
    // guard/readbacks cover all ten selected bodies plus shared/second
    // residual, but it deliberately has no retained post-L0 state/output
    // handoff.  Every descriptor remains component-only and is therefore
    // visible-but-rejected by the complete-decoder gate.
    let source_token_l0_outer_seal =
        "d28eccae71c10067e6d81039d04b14bb13bce9e3e7eaabf4830386524920568f";
    let source_token_l0 =
        |ledger_id: &str, operator_class: &str, scope: &str, covered_route_slots: Vec<usize>| {
            current_component_descriptor(
                &binding,
                ledger_id,
                source_token_l0_outer_seal,
                operator_class,
                scope,
                vec![0],
                covered_route_slots,
                "metal",
                23,
                false,
                true,
            )
        };
    ReadinessInput {
        schema: INPUT_SCHEMA.into(),
        source_artifact_binding: binding.clone(),
        layer_schedule: expected_schedule(),
        sealed_component_ledgers: vec![
            source_token_l0(
                "current-source-token-l0-input-norm-component",
                "input_norm",
                "per_layer",
                Vec::new(),
            ),
            source_token_l0(
                "current-source-token-l0-deltanet-mixer-component",
                "deltanet_mixer",
                "per_layer",
                Vec::new(),
            ),
            source_token_l0(
                "current-source-token-l0-postnorm-component",
                "post_norm",
                "per_layer",
                Vec::new(),
            ),
            source_token_l0(
                "current-source-token-l0-router-top10-component",
                "router_top10",
                "per_layer",
                Vec::new(),
            ),
            source_token_l0(
                "current-source-token-l0-routed-gate-up-component",
                "routed_gate_up",
                "per_layer_all_top10_routes",
                all_route_slots(),
            ),
            source_token_l0(
                "current-source-token-l0-routed-activation-component",
                "routed_activation",
                "per_layer_all_top10_routes",
                all_route_slots(),
            ),
            source_token_l0(
                "current-source-token-l0-routed-down-component",
                "routed_down",
                "per_layer_all_top10_routes",
                all_route_slots(),
            ),
            source_token_l0(
                "current-source-token-l0-shared-expert-component",
                "shared_expert",
                "per_layer",
                Vec::new(),
            ),
            source_token_l0(
                "current-source-token-l0-route-combine-component",
                "route_combine",
                "per_layer_all_top10_routes",
                all_route_slots(),
            ),
            source_token_l0(
                "current-source-token-l0-second-residual-component",
                "residual",
                "per_layer",
                Vec::new(),
            ),
            current_component_descriptor(
                &binding,
                "current-postnorm-l0-component",
                postnorm_outer_seal,
                "post_norm",
                "per_layer",
                vec![0],
                Vec::new(),
                "metal",
                3,
                true,
                true,
            ),
            current_component_descriptor(
                &binding,
                "current-router-top10-l0-component",
                postnorm_outer_seal,
                "router_top10",
                "per_layer",
                vec![0],
                Vec::new(),
                "metal",
                3,
                true,
                true,
            ),
            current_component_descriptor(
                &binding,
                "current-gqa-l3-two-token-component",
                gqa_outer_seal,
                "gqa_mixer",
                "per_layer",
                vec![3],
                Vec::new(),
                "metal",
                14,
                true,
                true,
            ),
            current_component_descriptor(
                &binding,
                "current-routed-gate-up-l0-route0-cpu-component",
                kv_component_seal,
                "routed_gate_up",
                "per_layer_all_top10_routes",
                vec![0],
                vec![0],
                "cpu",
                0,
                true,
                true,
            ),
            current_component_descriptor(
                &binding,
                "current-gqa-kv-l3-component",
                kv_component_seal,
                "gqa_kv_state",
                "per_layer",
                vec![3],
                Vec::new(),
                "cpu",
                0,
                true,
                true,
            ),
            current_component_descriptor(
                &binding,
                "current-final-head-cpu-fixture",
                kv_component_seal,
                "lm_head",
                "global",
                Vec::new(),
                Vec::new(),
                "cpu",
                0,
                true,
                true,
            ),
            current_component_descriptor(
                &binding,
                "current-tokenizer-template-sampler-cpu-fixture",
                kv_component_seal,
                "tokenizer_template",
                "global",
                Vec::new(),
                Vec::new(),
                "cpu",
                0,
                true,
                true,
            ),
        ],
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

fn parse_args() -> Result<Arguments, Box<dyn Error>> {
    let mut input_path = None;
    let mut current_evidence = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--input" => {
                let value = args.next().ok_or("missing absolute path after --input")?;
                if input_path.replace(PathBuf::from(value)).is_some() {
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
            _ => {
                return Err("usage: ascension_qwen80_complete_decoder_readiness_contract (--input ABSOLUTE_PATH | --current-evidence) --out ABSOLUTE_PATH".into())
            }
        }
    }
    let out = out.ok_or("missing --out")?;
    if !out.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    let input_mode = match (input_path, current_evidence) {
        (Some(_), true) => {
            return Err("--input and --current-evidence are mutually exclusive".into())
        }
        (Some(path), false) if path.is_absolute() => InputMode::Inventory(path),
        (Some(_), false) => return Err("--input must be an absolute path".into()),
        (None, true) => InputMode::CurrentEvidence,
        (None, false) => return Err("supply exactly one of --input or --current-evidence".into()),
    };
    Ok(Arguments { input_mode, out })
}

fn run(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let input = match arguments.input_mode {
        InputMode::Inventory(path) => serde_json::from_slice::<ReadinessInput>(&fs::read(&path)?)?,
        InputMode::CurrentEvidence => current_evidence_input(),
    };
    let report = evaluate(input);
    let earned = report.complete_decoder_readiness_earned;
    write_report_atomic(&arguments.out, &report)?;
    if !earned {
        return Err(format!(
            "Qwen80 complete-decoder readiness refused; incomplete report written to {}",
            arguments.out.display()
        )
        .into());
    }
    Ok(())
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_complete_decoder_readiness_contract: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sha(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn fully_valid_execution() -> DeviceExecutionDescriptor {
        DeviceExecutionDescriptor {
            backend: "metal".into(),
            device_dispatches: 1,
            source_bound: true,
            artifact_bound: true,
            full_path: true,
            complete_token_path: true,
            device_parity_passed: true,
            fixture_only: false,
            synthetic_input: false,
            component_only: false,
        }
    }

    fn complete_descriptor(
        binding: &SourceArtifactBinding,
        requirement: &RequiredOperator,
        index: usize,
    ) -> SealedComponentLedgerDescriptor {
        let (scope, covered_layers, covered_route_slots) = match &requirement.coverage {
            CoverageRequirement::Global => ("global", Vec::new(), Vec::new()),
            CoverageRequirement::PerLayer {
                layers,
                route_slots: None,
            } => ("per_layer", layers.clone(), Vec::new()),
            CoverageRequirement::PerLayer {
                layers,
                route_slots: Some(route_slots),
            } => (
                "per_layer_all_top10_routes",
                layers.clone(),
                route_slots.clone(),
            ),
        };
        SealedComponentLedgerDescriptor {
            descriptor_schema: DESCRIPTOR_SCHEMA.into(),
            ledger_id: format!("complete-{}-{index}", requirement.operator_class),
            ledger_receipt_seal_sha256: test_sha('a'),
            model_id: binding.model_id.clone(),
            model_key: binding.model_key.clone(),
            source_repository: binding.source_repository.clone(),
            source_revision: binding.source_revision.clone(),
            manifest_seal_sha256: binding.manifest_seal_sha256.clone(),
            admission_receipt_seal_sha256: binding.admission_receipt_seal_sha256.clone(),
            operator_class: requirement.operator_class.into(),
            scope: scope.into(),
            covered_layers,
            covered_route_slots,
            device_execution: fully_valid_execution(),
        }
    }

    fn complete_input() -> ReadinessInput {
        let binding = current_source_artifact_binding();
        let sealed_component_ledgers = requirements()
            .iter()
            .enumerate()
            .map(|(index, requirement)| complete_descriptor(&binding, requirement, index))
            .collect();
        ReadinessInput {
            schema: INPUT_SCHEMA.into(),
            source_artifact_binding: binding,
            layer_schedule: expected_schedule(),
            sealed_component_ledgers,
        }
    }

    #[test]
    fn full_exact_inventory_is_the_only_positive_contract_case() {
        let report = evaluate(complete_input());
        assert!(report.complete_decoder_readiness_earned);
        assert!(report.real_gravity_server_launch_precondition_satisfied);
        assert!(report.rejected_descriptors.is_empty());
        assert!(report.missing_operator_classes_or_layers.is_empty());
        assert_eq!(report.expected_48_layer_schedule.len(), LAYER_COUNT);
        assert_eq!(
            report
                .expected_48_layer_schedule
                .iter()
                .filter(|entry| entry.mixer == "deltanet")
                .count(),
            36
        );
        assert_eq!(
            report
                .expected_48_layer_schedule
                .iter()
                .filter(|entry| entry.mixer == "gqa")
                .count(),
            12
        );
    }

    #[test]
    fn current_component_evidence_is_explicitly_incomplete() {
        let report = evaluate(current_evidence_input());
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
        assert!(report.status.contains("INCOMPLETE"));
        assert!(report.status.contains("NO_DECODER_TOKEN_HCLI_OR_TPS_CLAIM"));
        assert!(report.rejected_descriptors.iter().any(|descriptor| {
            descriptor
                .reasons
                .iter()
                .any(|reason| reason.contains("component_only"))
        }));
        assert!(report
            .missing_operator_classes_or_layers
            .iter()
            .any(|coverage| coverage.operator_class == "embedding"));
    }

    #[test]
    fn false_component_substitution_cannot_cover_router() {
        let mut input = complete_input();
        let descriptor = input
            .sealed_component_ledgers
            .iter_mut()
            .find(|descriptor| descriptor.operator_class == "router_top10")
            .unwrap();
        descriptor.operator_class = "post_norm".into();
        descriptor.scope = "per_layer".into();
        let report = evaluate(input);
        assert!(!report.complete_decoder_readiness_earned);
        let router = report
            .missing_operator_classes_or_layers
            .iter()
            .find(|coverage| coverage.operator_class == "router_top10")
            .unwrap();
        assert_eq!(router.missing_layers, all_layers());
    }

    #[test]
    fn wrong_hybrid_schedule_is_rejected_even_with_all_operator_ledgers() {
        let mut input = complete_input();
        input.layer_schedule[3].mixer = "deltanet".into();
        let report = evaluate(input);
        assert!(!report.exact_48_layer_schedule_valid);
        assert!(!report.complete_decoder_readiness_earned);
        assert!(report
            .contract_errors
            .iter()
            .any(|error| error.contains("3xDeltaNet/1xGQA cadence")));
    }

    #[test]
    fn missing_sealed_ledger_is_rejected_deterministically() {
        let mut input = complete_input();
        input
            .sealed_component_ledgers
            .retain(|descriptor| descriptor.operator_class != "lm_head");
        let report = evaluate(input);
        assert!(!report.complete_decoder_readiness_earned);
        let lm_head = report
            .missing_operator_classes_or_layers
            .iter()
            .find(|coverage| coverage.operator_class == "lm_head")
            .unwrap();
        assert!(lm_head.required_global);
        assert!(!lm_head.covered_global);
    }
}
