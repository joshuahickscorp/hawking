//! Sealed final-manager scorecard adjudication preparation contract.
//!
//! This CPU-only contract consumes a fixed final-manager tournament protocol,
//! the prepared TG3/freeze comparison authority, a protected blind-corpus
//! commitment, an equalized resource/asymmetry ledger, four sealed scorecard
//! evidence documents, and a sealed Pareto-frontier receipt.  It can only
//! prepare an adjudication package or refuse it.  It cannot run tasks, touch a
//! model/GPU/server/watcher, measure TPS, launch a tournament, score a winner,
//! or designate a winner.  Protected selection remains a later authority.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.manager_tournament_scorecard_adjudication_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.manager_tournament_scorecard_adjudication_contract.v1";
const PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_MANAGER_TOURNAMENT_SCORECARD_ADJUDICATION_PENDING_PROTECTED_SELECTION";
const REFUSED_STATUS: &str =
    "REFUSED_MANAGER_TOURNAMENT_SCORECARD_ADJUDICATION_INCOMPLETE_UNEQUAL_OR_UNVERIFIED";

const FINAL_MANAGER_PROTOCOL_SCHEMA: &str =
    "hawking.ascension.final_manager_tournament_protocol.v1";
const FINAL_MANAGER_PROTOCOL_STATUS: &str =
    "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED";
const FINAL_MANAGER_PROTOCOL_IDENTITY: &str =
    "8e3684af0b7de53690a9c88ce0d52b0cae019e0d798bc2748b5b1556211facf8";
const TG3_COMPARISON_SCHEMA: &str =
    "hawking.ascension.paired_cognition_tg3_freeze_final_comparison_authority.v1";
const TG3_COMPARISON_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_TG3_FROZEN_FINAL_MANAGER_COMPARISON_RESERVED";
const CORPUS_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_task_corpus_commitment.v1";
const CORPUS_PREPARED_STATUS: &str = "PREPARED_PROTECTED_BLIND_TASK_CORPUS_COMMITMENT_NOT_EXECUTED";
const CORPUS_INCOMPLETE_STATUS: &str = "INCOMPLETE_PROTECTED_BLIND_TASK_CORPUS_COMMITMENT";
const LEDGER_SCHEMA: &str =
    "hawking.ascension.manager_tournament_resource_envelope_asymmetry_ledger.v1";
const LEDGER_EQUALIZED_STATUS: &str = "SEALED_EQUALIZED_RESOURCE_ENVELOPE_PENDING_SELECTION";
const LEDGER_INCOMPLETE_STATUS: &str = "INCOMPLETE_OR_UNEQUAL_RESOURCE_ENVELOPE";
const SCORECARD_SCHEMA: &str = "hawking.ascension.manager_tournament_mode_scorecard_evidence.v1";
const SCORECARD_MEASURED_STATUS: &str = "MEASURED_PROTECTED_PENDING_SELECTION";
const SCORECARD_INCOMPLETE_STATUS: &str = "INCOMPLETE_OR_UNVERIFIED_SCORECARD";
const PARETO_SCHEMA: &str = "hawking.ascension.manager_tournament_pareto_frontier_evidence.v1";
const PARETO_COMPLETE_STATUS: &str = "PARETO_FRONTIER_COMPLETE_PENDING_PROTECTED_SELECTION";
const PARETO_INCOMPLETE_STATUS: &str = "INCOMPLETE_OR_UNVERIFIED_PARETO_FRONTIER";

const CANDIDATES: [&str; 2] = [
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
];
const MODES: [&str; 2] = ["SOLO_MANAGER", "MANAGER_AS_ORCHESTRATOR"];
const HARD_GATES: [&str; 13] = [
    "source_identity",
    "complete_bpw_at_most_1_5",
    "tg3_base_true_tps",
    "fallback_zero",
    "real_metal",
    "coherent_real_hcli",
    "capability_retention",
    "agent_os_production_path",
    "residency",
    "restart_recovery",
    "storage_rollback",
    "security_effect_boundaries",
    "manager_contract",
];
const TASK_FAMILIES: [&str; 15] = [
    "repository_architecture",
    "kernel_metal_diagnosis",
    "gravity_research",
    "complete_token_performance",
    "bug_repair",
    "tool_use",
    "delegation",
    "helper_management",
    "context_memory",
    "interrupt_recovery",
    "adversarial_evidence",
    "benchmark_honesty",
    "storage_resource_campaign",
    "security_effect_boundaries",
    "release_integration",
];
const FAIRNESS_ENVELOPE: [&str; 11] = [
    "gpu_opportunity",
    "cpu_budget",
    "wall_clock_or_experiment_budget",
    "logical_agent_budget",
    "context_budget",
    "tool_access",
    "helper_model_access",
    "retry_policy",
    "storage_budget",
    "hidden_evidence",
    "temperature_sampling_policy",
];
const PRIMARY_METRICS: [&str; 10] = [
    "verified_tasks_per_hour",
    "first_pass_success",
    "repair_cycles",
    "failed_experiments",
    "tool_errors",
    "worker_rejection_accuracy",
    "human_intervention_count",
    "regressions_introduced",
    "rollback_events",
    "resource_use",
];
const PERFORMANCE_METRICS: [&str; 12] = [
    "base_true_tps",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "ttft",
    "hcli_overhead",
    "resident_memory",
    "active_bytes_per_token",
    "state_kv_traffic",
    "energy_resource_efficiency",
    "multi_session_throughput",
    "weight_reuse",
];
const INTELLIGENCE_METRICS: [&str; 14] = [
    "architecture_comprehension",
    "root_cause_diagnosis",
    "planning_quality",
    "experiment_quality",
    "scientific_reasoning",
    "kernel_reasoning",
    "representation_reasoning",
    "tool_competence",
    "delegation",
    "recovery",
    "context_retention",
    "evidence_discipline",
    "self_correction",
    "release_discipline",
];
const ORCHESTRATION_METRICS: [&str; 9] = [
    "worker_utilization",
    "duplicate_work",
    "idle_lanes",
    "bad_delegation",
    "integration_failures",
    "merge_conflicts",
    "review_quality",
    "critical_path_scheduling",
    "verified_tasks_per_hour",
];
const RECOVERY_INJECTIONS: [&str; 9] = [
    "bad_worker_patch",
    "failed_build",
    "false_benchmark",
    "stale_receipt",
    "killed_process",
    "disk_pressure",
    "kv_corruption",
    "failed_metal_parity",
    "incorrect_architectural_assumption",
];

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct SealedDocumentBinding {
    path: String,
    document_sha256: String,
    document: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct EvidenceDigest {
    path: String,
    document_schema: String,
    document_status: String,
    document_sha256: String,
    document_seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct AuthorityBoundary {
    new_physical_model_processes_authorized: usize,
    server_starts_authorized: usize,
    port_binds_authorized: usize,
    gpu_leases_authorized: usize,
    tournament_state_mutations_authorized: usize,
    paired_world_activation_authorized: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct ExecutionBoundary {
    live_artifact_scan_performed: bool,
    model_weights_loaded: bool,
    metal_device_or_dispatch_performed: bool,
    gpu_lease_or_registry_mutated: bool,
    model_or_decoder_token_executed: bool,
    logical_session_created: bool,
    runtime_watcher_or_server_started: bool,
    port_bound_or_listener_created: bool,
    hcli_executed: bool,
    tps_or_tg_measured: bool,
    tournament_state_mutated: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct Tg3FreezeComparisonDocument {
    schema: String,
    status: String,
    prepared: bool,
    tournament_active: bool,
    scored_task_execution_active: bool,
    winner_selected: bool,
    bound_final_manager_protocol: EvidenceDigest,
    final_comparison_reservation_complete: bool,
    blockers: Vec<String>,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize)]
struct ProtectedTaskCorpusDocument {
    schema: String,
    status: String,
    final_manager_protocol_identity_sha256: String,
    final_manager_protocol_seal_sha256: String,
    bound_tg3_freeze_final_comparison_authority: EvidenceDigest,
    bound_resource_envelope_asymmetry_ledger: EvidenceDigest,
    candidate_artifacts: Vec<String>,
    evaluation_modes: Vec<String>,
    protected_catalog_sha256: String,
    blind_tasks_frozen: bool,
    hidden_membership_inaccessible_to_candidates: bool,
    real_hawking_work_only: bool,
    protected_verifier_is_only_task_acceptance_authority: bool,
    required_families: Vec<String>,
    corpus_execution_authorized_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ResourceLedgerDocument {
    schema: String,
    status: String,
    final_manager_protocol_identity_sha256: String,
    final_manager_protocol_seal_sha256: String,
    bound_tg3_freeze_final_comparison_authority: EvidenceDigest,
    candidate_artifacts: Vec<String>,
    evaluation_modes: Vec<String>,
    resource_envelope_sha256: String,
    asymmetry_ledger_sha256: String,
    all_asymmetries_recorded: bool,
    equalized: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct MetricEvidence {
    measured: bool,
    protected_verifier_accepted: bool,
    candidate_self_scored: bool,
    evidence_seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct TaskAcceptance {
    accepted_task_count: usize,
    all_submitted_tasks_protected_verifier_accepted: bool,
    candidate_self_accepted: bool,
    protected_verifier_receipt_seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize)]
struct RedTeamAdjudication {
    opposing_candidate_artifact_id: String,
    opposing_candidate_read_only: bool,
    reviewer_modified_candidate_artifact: bool,
    protected_verifier_adjudicated_challenges: bool,
    candidate_self_adjudicated: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ScorecardDocument {
    schema: String,
    status: String,
    final_manager_protocol_identity_sha256: String,
    final_manager_protocol_seal_sha256: String,
    bound_tg3_freeze_final_comparison_authority: EvidenceDigest,
    bound_protected_task_corpus_commitment: EvidenceDigest,
    bound_resource_envelope_asymmetry_ledger: EvidenceDigest,
    candidate_artifact_id: String,
    evaluation_mode: String,
    resource_envelope_sha256: String,
    asymmetry_ledger_sha256: String,
    task_acceptance: TaskAcceptance,
    red_team_adjudication: RedTeamAdjudication,
    primary_metrics: BTreeMap<String, MetricEvidence>,
    performance_metrics: BTreeMap<String, MetricEvidence>,
    manager_intelligence_metrics: BTreeMap<String, MetricEvidence>,
    orchestration_metrics: BTreeMap<String, MetricEvidence>,
    failure_recovery_injections: BTreeMap<String, MetricEvidence>,
    candidate_self_scored: bool,
    winner_selected: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ParetoFrontierDocument {
    schema: String,
    status: String,
    final_manager_protocol_identity_sha256: String,
    final_manager_protocol_seal_sha256: String,
    bound_tg3_freeze_final_comparison_authority: EvidenceDigest,
    bound_protected_task_corpus_commitment: EvidenceDigest,
    bound_resource_envelope_asymmetry_ledger: EvidenceDigest,
    candidate_artifacts: Vec<String>,
    evaluation_modes: Vec<String>,
    scorecard_bindings: BTreeMap<String, BTreeMap<String, EvidenceDigest>>,
    all_required_scorecards_bound: bool,
    pareto_frontier_complete: bool,
    protected_verifier_accepted: bool,
    do_not_collapse_to_scalar_before_complete_evidence_matrix: bool,
    candidate_self_selection: bool,
    winner_selected: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    final_manager_protocol: SealedDocumentBinding,
    tg3_freeze_final_comparison_authority: SealedDocumentBinding,
    protected_task_corpus_commitment: SealedDocumentBinding,
    resource_envelope_asymmetry_ledger: SealedDocumentBinding,
    scorecard_evidence: BTreeMap<String, BTreeMap<String, SealedDocumentBinding>>,
    pareto_frontier_evidence: SealedDocumentBinding,
    protected_selection_requested: bool,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct ScorecardAudit {
    evaluation_mode: String,
    candidate_artifact_id: String,
    protected_task_acceptance_complete: bool,
    read_only_opposing_red_team_adjudicated: bool,
    all_required_metrics_complete_and_verified: bool,
    same_resource_envelope_and_asymmetry_ledger: bool,
    candidate_self_scoring_absent: bool,
    blockers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    fixed_final_protocol_identity_and_prepared_tg3_freeze_authority_bound: bool,
    protected_blind_corpus_and_equalized_resource_ledger_bound: bool,
    both_candidates_in_both_modes_required: bool,
    protected_verifier_read_only_red_team_and_no_self_scoring_required: bool,
    complete_primary_performance_intelligence_orchestration_and_recovery_metrics_required: bool,
    protected_pareto_frontier_required_before_later_selection: bool,
    no_runtime_server_gpu_watcher_tps_tournament_or_winner_authority: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: String,
    prepared: bool,
    tournament_active: bool,
    scorecards_executed_by_this_contract: bool,
    winner_selected: bool,
    protected_selection_authority_required: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_tg3_freeze_final_comparison_authority: EvidenceDigest,
    bound_protected_task_corpus_commitment: EvidenceDigest,
    bound_resource_envelope_asymmetry_ledger: EvidenceDigest,
    bound_pareto_frontier_evidence: EvidenceDigest,
    scorecard_audits: Vec<ScorecardAudit>,
    pareto_frontier_complete: bool,
    blockers: Vec<String>,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    focused_checks: FocusedChecks,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

struct Args {
    input: PathBuf,
    out: PathBuf,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|error| format!("JSON cannot be canonicalized: {error}"))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn verify_sealed_object(value: &Value, label: &str) -> Result<String, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))?;
    let recorded = object
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.seal_sha256 must be a lowercase SHA-256"))?;
    if !is_lower_sha256(recorded) {
        return Err(format!("{label}.seal_sha256 must be a lowercase SHA-256"));
    }
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_json(&Value::Object(unsigned))?;
    if recorded != expected {
        return Err(format!(
            "{label}.seal_sha256 mismatch: recorded={recorded}, expected={expected}"
        ));
    }
    Ok(recorded.into())
}

fn seal_value(mut value: Value) -> Result<Value, String> {
    let object = value
        .as_object_mut()
        .ok_or("only JSON objects can be sealed")?;
    object.remove("seal_sha256");
    let seal = sha256_json(&Value::Object(object.clone()))?;
    object.insert("seal_sha256".into(), Value::String(seal));
    Ok(value)
}

fn require_absolute_path(path: &str, label: &str) -> Result<(), String> {
    if !Path::new(path).is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    Ok(())
}

fn validate_authority_boundary(boundary: &AuthorityBoundary, label: &str) -> Result<(), String> {
    if boundary.new_physical_model_processes_authorized != 0
        || boundary.server_starts_authorized != 0
        || boundary.port_binds_authorized != 0
        || boundary.gpu_leases_authorized != 0
        || boundary.tournament_state_mutations_authorized != 0
        || boundary.paired_world_activation_authorized
    {
        return Err(format!(
            "{label} may not authorize processes, servers, ports, GPU leases, tournament mutation, or activation"
        ));
    }
    Ok(())
}

fn validate_execution_boundary(boundary: &ExecutionBoundary, label: &str) -> Result<(), String> {
    let observed = [
        (
            "live_artifact_scan_performed",
            boundary.live_artifact_scan_performed,
        ),
        ("model_weights_loaded", boundary.model_weights_loaded),
        (
            "metal_device_or_dispatch_performed",
            boundary.metal_device_or_dispatch_performed,
        ),
        (
            "gpu_lease_or_registry_mutated",
            boundary.gpu_lease_or_registry_mutated,
        ),
        (
            "model_or_decoder_token_executed",
            boundary.model_or_decoder_token_executed,
        ),
        ("logical_session_created", boundary.logical_session_created),
        (
            "runtime_watcher_or_server_started",
            boundary.runtime_watcher_or_server_started,
        ),
        (
            "port_bound_or_listener_created",
            boundary.port_bound_or_listener_created,
        ),
        ("hcli_executed", boundary.hcli_executed),
        ("tps_or_tg_measured", boundary.tps_or_tg_measured),
        (
            "tournament_state_mutated",
            boundary.tournament_state_mutated,
        ),
    ];
    for (field, value) in observed {
        if value {
            return Err(format!(
                "{label}.{field} must be false for this CPU-only contract"
            ));
        }
    }
    Ok(())
}

fn binding_digest(binding: &SealedDocumentBinding, label: &str) -> Result<EvidenceDigest, String> {
    require_absolute_path(&binding.path, &format!("{label}.path"))?;
    if !is_lower_sha256(&binding.document_sha256) {
        return Err(format!(
            "{label}.document_sha256 must be a lowercase SHA-256"
        ));
    }
    if sha256_json(&binding.document)? != binding.document_sha256 {
        return Err(format!(
            "{label}.document_sha256 does not bind the embedded document"
        ));
    }
    let seal = verify_sealed_object(&binding.document, &format!("{label}.document"))?;
    let object = binding
        .document
        .as_object()
        .ok_or_else(|| format!("{label}.document must be a JSON object"))?;
    let schema = object
        .get("schema")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{label}.document.schema must be non-empty"))?;
    let status = object
        .get("status")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| format!("{label}.document.status must be non-empty"))?;
    Ok(EvidenceDigest {
        path: binding.path.clone(),
        document_schema: schema.into(),
        document_status: status.into(),
        document_sha256: binding.document_sha256.clone(),
        document_seal_sha256: seal,
    })
}

fn exact_digest(
    expected: &EvidenceDigest,
    actual: &EvidenceDigest,
    label: &str,
) -> Result<(), String> {
    if actual.document_sha256 != expected.document_sha256
        || actual.document_seal_sha256 != expected.document_seal_sha256
        || actual.document_schema != expected.document_schema
        || actual.document_status != expected.document_status
    {
        return Err(format!("{label} must bind the exact sealed document"));
    }
    Ok(())
}

fn exact_list(observed: &[String], expected: &[&str], label: &str) -> Result<(), String> {
    let expected = expected
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    if observed != expected {
        return Err(format!(
            "{label} drift from the fixed final-manager protocol"
        ));
    }
    Ok(())
}

fn value_at<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a Value, String> {
    let mut cursor = value;
    for key in path {
        cursor = cursor
            .as_object()
            .and_then(|object| object.get(*key))
            .ok_or_else(|| format!("{label} is missing"))?;
    }
    Ok(cursor)
}

fn require_bool_at(
    value: &Value,
    path: &[&str],
    expected: bool,
    label: &str,
) -> Result<(), String> {
    let observed = value_at(value, path, label)?
        .as_bool()
        .ok_or_else(|| format!("{label} must be boolean"))?;
    if observed != expected {
        return Err(format!("{label} must be {expected}"));
    }
    Ok(())
}

fn require_string_list_at(
    value: &Value,
    path: &[&str],
    expected: &[&str],
    label: &str,
) -> Result<(), String> {
    let observed = value_at(value, path, label)?
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .map(str::to_owned)
                .ok_or_else(|| format!("{label} entries must be strings"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    exact_list(&observed, expected, label)
}

fn require_required_rows(
    value: &Value,
    path: &[&str],
    expected: &[&str],
    label: &str,
) -> Result<(), String> {
    let rows = value_at(value, path, label)?
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let observed = rows.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_set = expected.iter().copied().collect::<BTreeSet<_>>();
    if observed != expected_set {
        return Err(format!(
            "{label} must exactly cover the protected protocol keys"
        ));
    }
    if rows.values().any(|row| {
        row.as_object()
            .and_then(|object| object.get("required"))
            .and_then(Value::as_bool)
            != Some(true)
    }) {
        return Err(format!("{label} every row must be required"));
    }
    Ok(())
}

fn validate_final_protocol(binding: &SealedDocumentBinding) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "final_manager_protocol")?;
    if digest.document_schema != FINAL_MANAGER_PROTOCOL_SCHEMA
        || digest.document_status != FINAL_MANAGER_PROTOCOL_STATUS
    {
        return Err("final_manager_protocol has the wrong schema or status".into());
    }
    let document = &binding.document;
    let identity = document
        .get("protocol_identity_sha256")
        .and_then(Value::as_str)
        .ok_or("final_manager_protocol lacks protocol_identity_sha256")?;
    if identity != FINAL_MANAGER_PROTOCOL_IDENTITY || !is_lower_sha256(identity) {
        return Err("final_manager_protocol identity is not the armed fixed protocol".into());
    }
    let object = document
        .as_object()
        .ok_or("final_manager_protocol must be an object")?;
    let mut body = object.clone();
    body.remove("seal_sha256");
    body.remove("recorded_at");
    body.remove("protocol_identity_sha256");
    if sha256_json(&Value::Object(body))? != identity {
        return Err("final_manager_protocol identity does not bind its body".into());
    }
    require_string_list_at(
        document,
        &["candidates", "fixed_order"],
        &CANDIDATES,
        "final_manager_protocol.candidates.fixed_order",
    )?;
    require_string_list_at(
        document,
        &["evaluation_modes", "required"],
        &MODES,
        "final_manager_protocol.evaluation_modes.required",
    )?;
    require_required_rows(
        document,
        &["hard_gates", "required"],
        &HARD_GATES,
        "final_manager_protocol.hard_gates.required",
    )?;
    require_string_list_at(
        document,
        &["protected_task_corpus", "required_families"],
        &TASK_FAMILIES,
        "final_manager_protocol.protected_task_corpus.required_families",
    )?;
    require_required_rows(
        document,
        &["fairness_envelope", "equalize"],
        &FAIRNESS_ENVELOPE,
        "final_manager_protocol.fairness_envelope.equalize",
    )?;
    for (key, expected) in [
        ("primary_metrics", &PRIMARY_METRICS[..]),
        ("performance_metrics", &PERFORMANCE_METRICS[..]),
        ("manager_intelligence_metrics", &INTELLIGENCE_METRICS[..]),
        ("orchestration_metrics", &ORCHESTRATION_METRICS[..]),
        ("failure_recovery_injections", &RECOVERY_INJECTIONS[..]),
    ] {
        require_string_list_at(
            document,
            &["scorecards", key],
            expected,
            &format!("final_manager_protocol.scorecards.{key}"),
        )?;
    }
    for (path, label) in [
        (
            &["scorecards", "pareto_frontier_required_before_selection"][..],
            "final_manager_protocol.scorecards.pareto_frontier_required_before_selection",
        ),
        (
            &["scorecards", "do_not_collapse_to_scalar_before_complete_evidence_matrix"][..],
            "final_manager_protocol.scorecards.do_not_collapse_to_scalar_before_complete_evidence_matrix",
        ),
        (
            &["protected_selection", "candidates_cannot_self_grade"][..],
            "final_manager_protocol.protected_selection.candidates_cannot_self_grade",
        ),
        (
            &["protected_selection", "only_protected_controller_or_human_may_select_manager"][..],
            "final_manager_protocol.protected_selection.only_protected_controller_or_human_may_select_manager",
        ),
    ] {
        require_bool_at(document, path, true, label)?;
    }
    Ok(digest)
}

fn validate_tg3_comparison_authority(
    binding: &SealedDocumentBinding,
    protocol_digest: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "tg3_freeze_final_comparison_authority")?;
    let document: Tg3FreezeComparisonDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| {
            format!("tg3_freeze_final_comparison_authority has wrong grammar: {error}")
        })?;
    if document.schema != TG3_COMPARISON_SCHEMA || document.status != TG3_COMPARISON_STATUS {
        return Err("TG3/freeze authority is not the prepared final-comparison authority".into());
    }
    if !document.prepared
        || document.tournament_active
        || document.scored_task_execution_active
        || document.winner_selected
        || !document.final_comparison_reservation_complete
        || !document.blockers.is_empty()
    {
        return Err(
            "TG3/freeze authority must be complete, prepared, inactive, and blocker-free".into(),
        );
    }
    exact_digest(
        protocol_digest,
        &document.bound_final_manager_protocol,
        "TG3/freeze authority",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "TG3/freeze authority.boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "TG3/freeze authority.execution",
    )?;
    Ok(digest)
}

#[derive(Clone, Debug)]
struct LedgerValidation {
    digest: EvidenceDigest,
    resource_envelope_sha256: String,
    asymmetry_ledger_sha256: String,
    complete: bool,
}

fn validate_resource_ledger(
    binding: &SealedDocumentBinding,
    protocol_digest: &EvidenceDigest,
    tg3_digest: &EvidenceDigest,
) -> Result<LedgerValidation, String> {
    let digest = binding_digest(binding, "resource_envelope_asymmetry_ledger")?;
    let document: ResourceLedgerDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| {
            format!("resource_envelope_asymmetry_ledger has wrong grammar: {error}")
        })?;
    if document.schema != LEDGER_SCHEMA
        || (document.status != LEDGER_EQUALIZED_STATUS
            && document.status != LEDGER_INCOMPLETE_STATUS)
    {
        return Err("resource ledger has the wrong schema or status".into());
    }
    if document.final_manager_protocol_identity_sha256 != FINAL_MANAGER_PROTOCOL_IDENTITY
        || document.final_manager_protocol_seal_sha256 != protocol_digest.document_seal_sha256
    {
        return Err("resource ledger does not bind the fixed final-manager protocol".into());
    }
    exact_digest(
        tg3_digest,
        &document.bound_tg3_freeze_final_comparison_authority,
        "resource ledger",
    )?;
    exact_list(
        &document.candidate_artifacts,
        &CANDIDATES,
        "resource ledger candidates",
    )?;
    exact_list(&document.evaluation_modes, &MODES, "resource ledger modes")?;
    if !is_lower_sha256(&document.resource_envelope_sha256)
        || !is_lower_sha256(&document.asymmetry_ledger_sha256)
    {
        return Err(
            "resource ledger needs sealed resource-envelope and asymmetry-ledger identities".into(),
        );
    }
    let observed = document
        .equalized
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = FAIRNESS_ENVELOPE.iter().copied().collect::<BTreeSet<_>>();
    let complete = document.status == LEDGER_EQUALIZED_STATUS
        && document.all_asymmetries_recorded
        && observed == expected
        && document.equalized.values().all(|value| *value);
    Ok(LedgerValidation {
        digest,
        resource_envelope_sha256: document.resource_envelope_sha256,
        asymmetry_ledger_sha256: document.asymmetry_ledger_sha256,
        complete,
    })
}

fn validate_corpus_commitment(
    binding: &SealedDocumentBinding,
    protocol_digest: &EvidenceDigest,
    tg3_digest: &EvidenceDigest,
    ledger_digest: &EvidenceDigest,
) -> Result<(EvidenceDigest, bool), String> {
    let digest = binding_digest(binding, "protected_task_corpus_commitment")?;
    let document: ProtectedTaskCorpusDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("protected_task_corpus_commitment has wrong grammar: {error}"))?;
    if document.schema != CORPUS_SCHEMA
        || (document.status != CORPUS_PREPARED_STATUS
            && document.status != CORPUS_INCOMPLETE_STATUS)
    {
        return Err("protected task corpus has the wrong schema or status".into());
    }
    if document.final_manager_protocol_identity_sha256 != FINAL_MANAGER_PROTOCOL_IDENTITY
        || document.final_manager_protocol_seal_sha256 != protocol_digest.document_seal_sha256
    {
        return Err("protected task corpus does not bind the fixed final-manager protocol".into());
    }
    exact_digest(
        tg3_digest,
        &document.bound_tg3_freeze_final_comparison_authority,
        "protected task corpus",
    )?;
    exact_digest(
        ledger_digest,
        &document.bound_resource_envelope_asymmetry_ledger,
        "protected task corpus",
    )?;
    exact_list(
        &document.candidate_artifacts,
        &CANDIDATES,
        "protected task corpus candidates",
    )?;
    exact_list(
        &document.evaluation_modes,
        &MODES,
        "protected task corpus modes",
    )?;
    exact_list(
        &document.required_families,
        &TASK_FAMILIES,
        "protected task corpus families",
    )?;
    if !is_lower_sha256(&document.protected_catalog_sha256) {
        return Err("protected task corpus must bind a sealed catalog SHA-256".into());
    }
    let complete = document.status == CORPUS_PREPARED_STATUS
        && document.blind_tasks_frozen
        && document.hidden_membership_inaccessible_to_candidates
        && document.real_hawking_work_only
        && document.protected_verifier_is_only_task_acceptance_authority
        && !document.corpus_execution_authorized_by_this_contract;
    Ok((digest, complete))
}

fn metrics_complete(
    metrics: &BTreeMap<String, MetricEvidence>,
    expected: &[&str],
    label: &str,
) -> Result<bool, String> {
    let observed = metrics.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected_keys = expected.iter().copied().collect::<BTreeSet<_>>();
    if observed != expected_keys {
        return Err(format!(
            "{label} must exactly cover the final-manager metric set"
        ));
    }
    Ok(metrics.values().all(|metric| {
        metric.measured
            && metric.protected_verifier_accepted
            && !metric.candidate_self_scored
            && is_lower_sha256(&metric.evidence_seal_sha256)
    }))
}

fn other_candidate(candidate: &str) -> Result<&'static str, String> {
    if candidate == CANDIDATES[0] {
        Ok(CANDIDATES[1])
    } else if candidate == CANDIDATES[1] {
        Ok(CANDIDATES[0])
    } else {
        Err("scorecard has an unknown candidate".into())
    }
}

fn audit_scorecard(
    binding: &SealedDocumentBinding,
    mode: &str,
    candidate: &str,
    protocol_digest: &EvidenceDigest,
    tg3_digest: &EvidenceDigest,
    corpus_digest: &EvidenceDigest,
    ledger: &LedgerValidation,
) -> Result<(EvidenceDigest, ScorecardAudit), String> {
    let digest = binding_digest(binding, &format!("scorecard_evidence.{mode}.{candidate}"))?;
    let document: ScorecardDocument =
        serde_json::from_value(binding.document.clone()).map_err(|error| {
            format!("scorecard_evidence.{mode}.{candidate} has wrong grammar: {error}")
        })?;
    if document.schema != SCORECARD_SCHEMA
        || (document.status != SCORECARD_MEASURED_STATUS
            && document.status != SCORECARD_INCOMPLETE_STATUS)
    {
        return Err(format!(
            "scorecard_evidence.{mode}.{candidate} has the wrong schema or status"
        ));
    }
    if document.final_manager_protocol_identity_sha256 != FINAL_MANAGER_PROTOCOL_IDENTITY
        || document.final_manager_protocol_seal_sha256 != protocol_digest.document_seal_sha256
    {
        return Err(format!(
            "scorecard_evidence.{mode}.{candidate} does not bind the final protocol"
        ));
    }
    exact_digest(
        tg3_digest,
        &document.bound_tg3_freeze_final_comparison_authority,
        &format!("scorecard_evidence.{mode}.{candidate}"),
    )?;
    exact_digest(
        corpus_digest,
        &document.bound_protected_task_corpus_commitment,
        &format!("scorecard_evidence.{mode}.{candidate}"),
    )?;
    exact_digest(
        &ledger.digest,
        &document.bound_resource_envelope_asymmetry_ledger,
        &format!("scorecard_evidence.{mode}.{candidate}"),
    )?;
    if document.candidate_artifact_id != candidate || document.evaluation_mode != mode {
        return Err(format!(
            "scorecard_evidence.{mode}.{candidate} is bound to the wrong candidate or mode"
        ));
    }
    let mut blockers = Vec::new();
    let same_resource = document.resource_envelope_sha256 == ledger.resource_envelope_sha256
        && document.asymmetry_ledger_sha256 == ledger.asymmetry_ledger_sha256;
    if !same_resource {
        blockers.push("resource_envelope_or_asymmetry_ledger_mismatch".into());
    }
    let task_acceptance = document.task_acceptance.accepted_task_count > 0
        && document
            .task_acceptance
            .all_submitted_tasks_protected_verifier_accepted
        && !document.task_acceptance.candidate_self_accepted
        && is_lower_sha256(
            &document
                .task_acceptance
                .protected_verifier_receipt_seal_sha256,
        );
    if !task_acceptance {
        blockers.push("protected_task_acceptance_incomplete_or_self_accepted".into());
    }
    let red_team = document
        .red_team_adjudication
        .opposing_candidate_artifact_id
        == other_candidate(candidate)?
        && document.red_team_adjudication.opposing_candidate_read_only
        && !document
            .red_team_adjudication
            .reviewer_modified_candidate_artifact
        && document
            .red_team_adjudication
            .protected_verifier_adjudicated_challenges
        && !document.red_team_adjudication.candidate_self_adjudicated;
    if !red_team {
        blockers.push("opposing_red_team_not_read_only_or_not_protected_adjudication".into());
    }
    let all_metrics = metrics_complete(
        &document.primary_metrics,
        &PRIMARY_METRICS,
        &format!("scorecard_evidence.{mode}.{candidate}.primary_metrics"),
    )? && metrics_complete(
        &document.performance_metrics,
        &PERFORMANCE_METRICS,
        &format!("scorecard_evidence.{mode}.{candidate}.performance_metrics"),
    )? && metrics_complete(
        &document.manager_intelligence_metrics,
        &INTELLIGENCE_METRICS,
        &format!("scorecard_evidence.{mode}.{candidate}.manager_intelligence_metrics"),
    )? && metrics_complete(
        &document.orchestration_metrics,
        &ORCHESTRATION_METRICS,
        &format!("scorecard_evidence.{mode}.{candidate}.orchestration_metrics"),
    )? && metrics_complete(
        &document.failure_recovery_injections,
        &RECOVERY_INJECTIONS,
        &format!("scorecard_evidence.{mode}.{candidate}.failure_recovery_injections"),
    )?;
    if !all_metrics {
        blockers.push("required_metrics_incomplete_unverified_or_self_scored".into());
    }
    let self_scoring_absent = !document.candidate_self_scored && !document.winner_selected;
    if !self_scoring_absent {
        blockers.push("candidate_self_scored_or_winner_preselected".into());
    }
    if document.status != SCORECARD_MEASURED_STATUS {
        blockers.push("scorecard_not_measured_or_not_ready_for_protected_selection".into());
    }
    blockers.sort();
    blockers.dedup();
    Ok((
        digest,
        ScorecardAudit {
            evaluation_mode: mode.into(),
            candidate_artifact_id: candidate.into(),
            protected_task_acceptance_complete: task_acceptance,
            read_only_opposing_red_team_adjudicated: red_team,
            all_required_metrics_complete_and_verified: all_metrics,
            same_resource_envelope_and_asymmetry_ledger: same_resource,
            candidate_self_scoring_absent: self_scoring_absent,
            blockers,
        },
    ))
}

fn validate_pareto_frontier(
    binding: &SealedDocumentBinding,
    protocol_digest: &EvidenceDigest,
    tg3_digest: &EvidenceDigest,
    corpus_digest: &EvidenceDigest,
    ledger_digest: &EvidenceDigest,
    scorecard_digests: &BTreeMap<String, BTreeMap<String, EvidenceDigest>>,
) -> Result<(EvidenceDigest, bool), String> {
    let digest = binding_digest(binding, "pareto_frontier_evidence")?;
    let document: ParetoFrontierDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("pareto_frontier_evidence has wrong grammar: {error}"))?;
    if document.schema != PARETO_SCHEMA
        || (document.status != PARETO_COMPLETE_STATUS
            && document.status != PARETO_INCOMPLETE_STATUS)
    {
        return Err("pareto_frontier_evidence has the wrong schema or status".into());
    }
    if document.final_manager_protocol_identity_sha256 != FINAL_MANAGER_PROTOCOL_IDENTITY
        || document.final_manager_protocol_seal_sha256 != protocol_digest.document_seal_sha256
    {
        return Err("pareto_frontier_evidence does not bind the final protocol".into());
    }
    exact_digest(
        tg3_digest,
        &document.bound_tg3_freeze_final_comparison_authority,
        "pareto_frontier_evidence",
    )?;
    exact_digest(
        corpus_digest,
        &document.bound_protected_task_corpus_commitment,
        "pareto_frontier_evidence",
    )?;
    exact_digest(
        ledger_digest,
        &document.bound_resource_envelope_asymmetry_ledger,
        "pareto_frontier_evidence",
    )?;
    exact_list(
        &document.candidate_artifacts,
        &CANDIDATES,
        "pareto candidates",
    )?;
    exact_list(&document.evaluation_modes, &MODES, "pareto modes")?;
    let observed_modes = document
        .scorecard_bindings
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if observed_modes != MODES.iter().copied().collect::<BTreeSet<_>>() {
        return Err("pareto scorecard bindings must exactly cover both modes".into());
    }
    for mode in MODES {
        let observed = document
            .scorecard_bindings
            .get(mode)
            .ok_or("pareto scorecard binding missing mode")?;
        let expected = scorecard_digests
            .get(mode)
            .ok_or("internal scorecard digest missing mode")?;
        let observed_keys = observed.keys().map(String::as_str).collect::<BTreeSet<_>>();
        if observed_keys != CANDIDATES.iter().copied().collect::<BTreeSet<_>>() {
            return Err("pareto scorecard bindings must exactly cover both candidates".into());
        }
        for candidate in CANDIDATES {
            let Some(expected_digest) = expected.get(candidate) else {
                return Err("internal scorecard digest missing candidate".into());
            };
            let Some(observed_digest) = observed.get(candidate) else {
                return Ok((digest, false));
            };
            if exact_digest(expected_digest, observed_digest, "pareto scorecard binding").is_err() {
                return Ok((digest, false));
            }
        }
    }
    let complete = document.status == PARETO_COMPLETE_STATUS
        && document.all_required_scorecards_bound
        && document.pareto_frontier_complete
        && document.protected_verifier_accepted
        && document.do_not_collapse_to_scalar_before_complete_evidence_matrix
        && !document.candidate_self_selection
        && !document.winner_selected;
    Ok((digest, complete))
}

fn build_report(input: Input) -> Result<Report, String> {
    if input.schema != INPUT_SCHEMA {
        return Err(format!("input.schema must be {INPUT_SCHEMA:?}"));
    }
    if !is_lower_sha256(&input.seal_sha256) {
        return Err("input.seal_sha256 must be a lowercase SHA-256".into());
    }
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;
    let protocol_digest = validate_final_protocol(&input.final_manager_protocol)?;
    let tg3_digest = validate_tg3_comparison_authority(
        &input.tg3_freeze_final_comparison_authority,
        &protocol_digest,
    )?;
    let ledger = validate_resource_ledger(
        &input.resource_envelope_asymmetry_ledger,
        &protocol_digest,
        &tg3_digest,
    )?;
    let (corpus_digest, corpus_complete) = validate_corpus_commitment(
        &input.protected_task_corpus_commitment,
        &protocol_digest,
        &tg3_digest,
        &ledger.digest,
    )?;
    let mut blockers = Vec::new();
    if !ledger.complete {
        blockers.push("resource_envelope_asymmetry_ledger_incomplete_or_unequal".into());
    }
    if !corpus_complete {
        blockers.push("protected_blind_task_corpus_commitment_incomplete".into());
    }
    if input.protected_selection_requested {
        blockers.push("protected_selection_is_outside_this_adjudication_contract".into());
    }
    let observed_modes = input
        .scorecard_evidence
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if observed_modes != MODES.iter().copied().collect::<BTreeSet<_>>() {
        blockers.push("scorecard_evidence_does_not_exactly_cover_both_modes".into());
    }
    let mut audits = Vec::new();
    let mut scorecard_digests = BTreeMap::new();
    for mode in MODES {
        let Some(by_candidate) = input.scorecard_evidence.get(mode) else {
            continue;
        };
        let candidate_keys = by_candidate
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        if candidate_keys != CANDIDATES.iter().copied().collect::<BTreeSet<_>>() {
            blockers.push(format!(
                "scorecard_evidence.{mode}_does_not_exactly_cover_both_candidates"
            ));
        }
        let mut digests = BTreeMap::new();
        for candidate in CANDIDATES {
            let Some(binding) = by_candidate.get(candidate) else {
                continue;
            };
            let (digest, audit) = audit_scorecard(
                binding,
                mode,
                candidate,
                &protocol_digest,
                &tg3_digest,
                &corpus_digest,
                &ledger,
            )?;
            blockers.extend(
                audit
                    .blockers
                    .iter()
                    .map(|blocker| format!("{mode}:{candidate}:{blocker}")),
            );
            digests.insert(candidate.into(), digest);
            audits.push(audit);
        }
        scorecard_digests.insert(mode.into(), digests);
    }
    let complete_scorecard_set = scorecard_digests.len() == MODES.len()
        && scorecard_digests
            .values()
            .all(|by_candidate| by_candidate.len() == CANDIDATES.len());
    if !complete_scorecard_set {
        blockers.push("scorecard_evidence_has_missing_mode_or_candidate".into());
    }
    let (pareto_digest, pareto_complete) = if complete_scorecard_set {
        validate_pareto_frontier(
            &input.pareto_frontier_evidence,
            &protocol_digest,
            &tg3_digest,
            &corpus_digest,
            &ledger.digest,
            &scorecard_digests,
        )?
    } else {
        let digest = binding_digest(&input.pareto_frontier_evidence, "pareto_frontier_evidence")?;
        (digest, false)
    };
    if !pareto_complete {
        blockers.push("protected_pareto_frontier_incomplete_or_self_selected".into());
    }
    blockers.sort();
    blockers.dedup();
    let all_scorecards_accepted = audits.len() == MODES.len() * CANDIDATES.len()
        && audits.iter().all(|audit| audit.blockers.is_empty());
    let prepared = ledger.complete
        && corpus_complete
        && !input.protected_selection_requested
        && complete_scorecard_set
        && all_scorecards_accepted
        && pareto_complete
        && blockers.is_empty();
    let status = if prepared {
        PREPARED_STATUS
    } else {
        REFUSED_STATUS
    }
    .to_owned();
    let focused_checks = FocusedChecks {
        fixed_final_protocol_identity_and_prepared_tg3_freeze_authority_bound: true,
        protected_blind_corpus_and_equalized_resource_ledger_bound: ledger.complete
            && corpus_complete,
        both_candidates_in_both_modes_required: complete_scorecard_set,
        protected_verifier_read_only_red_team_and_no_self_scoring_required: all_scorecards_accepted,
        complete_primary_performance_intelligence_orchestration_and_recovery_metrics_required:
            all_scorecards_accepted,
        protected_pareto_frontier_required_before_later_selection: pareto_complete,
        no_runtime_server_gpu_watcher_tps_tournament_or_winner_authority: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        prepared,
        tournament_active: false,
        scorecards_executed_by_this_contract: false,
        winner_selected: false,
        protected_selection_authority_required: true,
        bound_final_manager_protocol: protocol_digest,
        bound_tg3_freeze_final_comparison_authority: tg3_digest,
        bound_protected_task_corpus_commitment: corpus_digest,
        bound_resource_envelope_asymmetry_ledger: ledger.digest,
        bound_pareto_frontier_evidence: pareto_digest,
        scorecard_audits: audits,
        pareto_frontier_complete: pareto_complete,
        blockers,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only scorecard adjudication preparation contract, not a tournament executor.",
            "It requires both fixed Gravity manager candidates in identical SOLO_MANAGER and MANAGER_AS_ORCHESTRATOR evidence matrices.",
            "Every submitted task must be accepted by the protected verifier; opposing candidates are read-only red teams whose challenges the protected verifier adjudicates.",
            "Primary, performance, intelligence, orchestration, and failure-recovery evidence must be complete, verifier-accepted, and free of candidate self-scoring.",
            "The resource envelope and every asymmetry must be identically bound before the protected Pareto frontier can be prepared.",
            "A prepared result neither chooses nor recommends a winner; protected selection authority or a human remains required.",
            "No GPU, model, server, watcher, HCLI, TPS/TG, task execution, tournament mutation, or winner-selection action is performed here.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    let preimage = serde_json::to_value(&report)
        .map_err(|error| format!("report cannot be serialized: {error}"))?;
    report.unsealed_preimage_sha256 = sha256_json(&preimage)?;
    Ok(report)
}

fn load_input(path: &Path) -> Result<Input, Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--input must be an absolute path".into());
    }
    let raw: Value = serde_json::from_slice(&fs::read(path)?)?;
    verify_sealed_object(&raw, "input")
        .map_err(|error| format!("invalid sealed input: {error}"))?;
    let input: Input = serde_json::from_value(raw)?;
    if input.schema != INPUT_SCHEMA {
        return Err(format!("input.schema must be {INPUT_SCHEMA}").into());
    }
    Ok(input)
}

fn write_report_create_new(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let value =
        seal_value(serde_json::to_value(report).map_err(|error| {
            format!("scorecard adjudication report cannot be serialized: {error}")
        })?)
        .map_err(|error| format!("scorecard adjudication report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_manager_tournament_scorecard_adjudication_contract --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut input = None;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--input" => {
                let path = args.next().ok_or("--input requires a path")?;
                if input.replace(PathBuf::from(path)).is_some() {
                    return Err("--input repeated".into());
                }
            }
            "--out" => {
                let path = args.next().ok_or("--out requires a path")?;
                if out.replace(PathBuf::from(path)).is_some() {
                    return Err("--out repeated".into());
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unexpected argument {other:?}; {}", usage()).into()),
        }
    }
    let args = Args {
        input: input.ok_or_else(|| format!("missing --input; {}", usage()))?,
        out: out.ok_or_else(|| format!("missing --out; {}", usage()))?,
    };
    if !args.input.is_absolute() || !args.out.is_absolute() {
        return Err(format!("all paths must be absolute; {}", usage()).into());
    }
    Ok(args)
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let input = load_input(&args.input)?;
    let report = build_report(input)
        .map_err(|error| format!("scorecard adjudication validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_manager_tournament_scorecard_adjudication_contract: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn authority_boundary() -> AuthorityBoundary {
        AuthorityBoundary {
            new_physical_model_processes_authorized: 0,
            server_starts_authorized: 0,
            port_binds_authorized: 0,
            gpu_leases_authorized: 0,
            tournament_state_mutations_authorized: 0,
            paired_world_activation_authorized: false,
        }
    }

    fn execution_boundary() -> ExecutionBoundary {
        ExecutionBoundary {
            live_artifact_scan_performed: false,
            model_weights_loaded: false,
            metal_device_or_dispatch_performed: false,
            gpu_lease_or_registry_mutated: false,
            model_or_decoder_token_executed: false,
            logical_session_created: false,
            runtime_watcher_or_server_started: false,
            port_bound_or_listener_created: false,
            hcli_executed: false,
            tps_or_tg_measured: false,
            tournament_state_mutated: false,
        }
    }

    fn binding(path: &str, document: Value) -> SealedDocumentBinding {
        SealedDocumentBinding {
            path: path.into(),
            document_sha256: sha256_json(&document).unwrap(),
            document,
        }
    }

    fn digest_from(binding: &SealedDocumentBinding, schema: &str, status: &str) -> Value {
        json!({
            "path": binding.path,
            "document_schema": schema,
            "document_status": status,
            "document_sha256": binding.document_sha256,
            "document_seal_sha256": binding.document["seal_sha256"],
        })
    }

    fn required_rows(values: &[&str]) -> Value {
        let mut map = Map::new();
        for value in values {
            map.insert((*value).into(), json!({"required": true}));
        }
        Value::Object(map)
    }

    fn protocol_binding() -> SealedDocumentBinding {
        let body = json!({
            "schema": FINAL_MANAGER_PROTOCOL_SCHEMA,
            "status": FINAL_MANAGER_PROTOCOL_STATUS,
            "purpose": "Select the Hawking operating intelligence that produces the most correct, verified, recoverable engineering progress per unit of machine resource while preserving the Ascension constitution.",
            "ascent_boundary": {
                "development_race_is_not_final_selection": true,
                "candidate_specific_optimization_allowed_only_before_freeze": true,
                "freeze_requires_both_complete_manager_qualification_floors": true,
                "post_freeze_candidate_specific_change_requires_failed_hard_gate_repair": true,
                "repair_requires_replay_of_all_affected_scored_tests": true,
            },
            "candidates": {"fixed_order": CANDIDATES, "both_must_meet_complete_manager_qualification_floor_before_freeze": true},
            "freeze_contract": {"both_candidates_frozen_before_scored_execution": true, "candidate_specific_optimization_disabled_during_scored_execution": true, "hard_gate_repair_requires_affected_test_replay": true},
            "evaluation_modes": {
                "required": MODES,
                "SOLO_MANAGER": {"measures": "raw_model_manager_capability", "no_helper_advantage": true, "each_candidate_measured": true},
                "MANAGER_AS_ORCHESTRATOR": {"measures": "organization_level_verified_engineering_progress", "same_hawking_agent_os": true, "symmetric_helper_model_infrastructure": true, "no_hidden_information_or_worker_advantage": true, "each_candidate_measured": true},
            },
            "hard_gates": {"conjunctive": true, "no_score_can_compensate_for_failure": true, "each_gate_required_for_each_candidate": true, "required": required_rows(&HARD_GATES)},
            "protected_task_corpus": {"real_hawking_work_only": true, "blind_tasks_required": true, "hidden_membership_frozen_before_scored_execution": true, "required_families": TASK_FAMILIES},
            "long_horizon_campaigns": {"required": ["six_stage_kernel_optimization", "multi_subsystem_hcli_repair", "exact_model_family_qualification", "storage_pressure_model_acquisition", "agent_os_concurrency_optimization", "gravity_representation_tournament"], "measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion": true},
            "adversarial_review": {"required_after_each_candidate_task": true, "opposing_candidate_is_read_only_red_team_reviewer": true, "reviewer_may_not_modify_candidate_artifact": true, "protected_verifier_adjudicates_challenges": true, "measure_genuine_defects_and_false_objections": true},
            "fairness_envelope": {"record_every_asymmetry": true, "equalize": required_rows(&FAIRNESS_ENVELOPE)},
            "scorecards": {"primary_metric": "verified_tasks_per_hour", "primary_metrics": PRIMARY_METRICS, "performance_metrics": PERFORMANCE_METRICS, "manager_intelligence_metrics": INTELLIGENCE_METRICS, "orchestration_metrics": ORCHESTRATION_METRICS, "failure_recovery_injections": RECOVERY_INJECTIONS, "pareto_frontier_required_before_selection": true, "do_not_collapse_to_scalar_before_complete_evidence_matrix": true},
            "protected_selection": {"candidate_self_assessments_are_evidence_only": true, "candidates_cannot_self_grade": true, "candidates_cannot_change_weights_or_hidden_tests": true, "candidates_cannot_promote_self_or_invalidate_opponent": true, "only_protected_controller_or_human_may_select_manager": true},
            "winner_freeze_and_alternate": {"seal_winner_source_artifact_runtime_kernel_agent_os_context_kv_benchmarks_capability_tournament_and_rollback": true, "seal_loser_before_any_evictability": true, "cold_store_hash_verify_restore_test_one_command_recovery_required": true, "do_not_delete_loser_before_restore_proof": true},
            "final_report": {"side_by_side_candidates_required": true, "required_fields": ["bpw", "tg_rung", "base_true_tps", "p99", "memory", "verified_tasks_per_hour", "solo_capability", "orchestrated_capability", "repository_architecture_tasks", "kernel_tasks", "gravity_tasks", "tool_reliability", "delegation", "context", "restart", "failure_recovery", "adversarial_review", "resource_efficiency", "long_horizon_completion", "human_interventions", "hard_gate_status"], "must_state_winner_loser_decisive_evidence_tradeoffs_restore_path": true},
            "claim_boundary": {"does_not_execute_tournament": true, "does_not_score_candidates": true, "does_not_choose_winner": true, "does_not_activate_a_server_or_sandbox": true, "does_not_relax_tg3_or_other_hard_gates": true},
        });
        let identity = sha256_json(&body).unwrap();
        assert_eq!(identity, FINAL_MANAGER_PROTOCOL_IDENTITY);
        let mut document = body.as_object().unwrap().clone();
        document.insert("protocol_identity_sha256".into(), Value::String(identity));
        document.insert(
            "recorded_at".into(),
            Value::String("2026-08-09T06:44:09.081687Z".into()),
        );
        binding(
            "/sealed/final-manager-protocol.json",
            seal_value(Value::Object(document)).unwrap(),
        )
    }

    fn tg3_binding(protocol: &SealedDocumentBinding) -> SealedDocumentBinding {
        binding(
            "/sealed/tg3-freeze-comparison.json",
            seal_value(json!({
                "schema": TG3_COMPARISON_SCHEMA,
                "status": TG3_COMPARISON_STATUS,
                "prepared": true,
                "tournament_active": false,
                "scored_task_execution_active": false,
                "winner_selected": false,
                "bound_final_manager_protocol": digest_from(protocol, FINAL_MANAGER_PROTOCOL_SCHEMA, FINAL_MANAGER_PROTOCOL_STATUS),
                "final_comparison_reservation_complete": true,
                "blockers": [],
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn ledger_binding(
        protocol: &SealedDocumentBinding,
        tg3: &SealedDocumentBinding,
        complete: bool,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/resource-ledger.json",
            seal_value(json!({
                "schema": LEDGER_SCHEMA,
                "status": if complete { LEDGER_EQUALIZED_STATUS } else { LEDGER_INCOMPLETE_STATUS },
                "final_manager_protocol_identity_sha256": FINAL_MANAGER_PROTOCOL_IDENTITY,
                "final_manager_protocol_seal_sha256": protocol.document["seal_sha256"],
                "bound_tg3_freeze_final_comparison_authority": digest_from(tg3, TG3_COMPARISON_SCHEMA, TG3_COMPARISON_STATUS),
                "candidate_artifacts": CANDIDATES,
                "evaluation_modes": MODES,
                "resource_envelope_sha256": sha('a'),
                "asymmetry_ledger_sha256": sha('b'),
                "all_asymmetries_recorded": complete,
                "equalized": FAIRNESS_ENVELOPE.iter().map(|key| ((*key).to_owned(), complete)).collect::<BTreeMap<_, _>>(),
            }))
            .unwrap(),
        )
    }

    fn corpus_binding(
        protocol: &SealedDocumentBinding,
        tg3: &SealedDocumentBinding,
        ledger: &SealedDocumentBinding,
        complete: bool,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/protected-corpus.json",
            seal_value(json!({
                "schema": CORPUS_SCHEMA,
                "status": if complete { CORPUS_PREPARED_STATUS } else { CORPUS_INCOMPLETE_STATUS },
                "final_manager_protocol_identity_sha256": FINAL_MANAGER_PROTOCOL_IDENTITY,
                "final_manager_protocol_seal_sha256": protocol.document["seal_sha256"],
                "bound_tg3_freeze_final_comparison_authority": digest_from(tg3, TG3_COMPARISON_SCHEMA, TG3_COMPARISON_STATUS),
                "bound_resource_envelope_asymmetry_ledger": digest_from(ledger, LEDGER_SCHEMA, if complete { LEDGER_EQUALIZED_STATUS } else { LEDGER_INCOMPLETE_STATUS }),
                "candidate_artifacts": CANDIDATES,
                "evaluation_modes": MODES,
                "protected_catalog_sha256": sha('c'),
                "blind_tasks_frozen": complete,
                "hidden_membership_inaccessible_to_candidates": complete,
                "real_hawking_work_only": complete,
                "protected_verifier_is_only_task_acceptance_authority": complete,
                "required_families": TASK_FAMILIES,
                "corpus_execution_authorized_by_this_contract": false,
            }))
            .unwrap(),
        )
    }

    fn metric_map(values: &[&str], complete: bool, seed: char) -> BTreeMap<String, MetricEvidence> {
        values
            .iter()
            .enumerate()
            .map(|(index, key)| {
                let hex = ['a', 'b', 'c', 'd', 'e', 'f'];
                let character = hex[(seed as usize + index) % hex.len()];
                (
                    (*key).into(),
                    MetricEvidence {
                        measured: complete,
                        protected_verifier_accepted: complete,
                        candidate_self_scored: false,
                        evidence_seal_sha256: sha(character),
                    },
                )
            })
            .collect()
    }

    fn scorecard_binding(
        protocol: &SealedDocumentBinding,
        tg3: &SealedDocumentBinding,
        corpus: &SealedDocumentBinding,
        ledger: &SealedDocumentBinding,
        mode: &str,
        candidate: &str,
        complete: bool,
    ) -> SealedDocumentBinding {
        binding(
            &format!("/sealed/scorecards/{mode}/{candidate}.json"),
            seal_value(json!({
                "schema": SCORECARD_SCHEMA,
                "status": if complete { SCORECARD_MEASURED_STATUS } else { SCORECARD_INCOMPLETE_STATUS },
                "final_manager_protocol_identity_sha256": FINAL_MANAGER_PROTOCOL_IDENTITY,
                "final_manager_protocol_seal_sha256": protocol.document["seal_sha256"],
                "bound_tg3_freeze_final_comparison_authority": digest_from(tg3, TG3_COMPARISON_SCHEMA, TG3_COMPARISON_STATUS),
                "bound_protected_task_corpus_commitment": digest_from(corpus, CORPUS_SCHEMA, if complete { CORPUS_PREPARED_STATUS } else { CORPUS_INCOMPLETE_STATUS }),
                "bound_resource_envelope_asymmetry_ledger": digest_from(ledger, LEDGER_SCHEMA, if complete { LEDGER_EQUALIZED_STATUS } else { LEDGER_INCOMPLETE_STATUS }),
                "candidate_artifact_id": candidate,
                "evaluation_mode": mode,
                "resource_envelope_sha256": ledger.document["resource_envelope_sha256"],
                "asymmetry_ledger_sha256": ledger.document["asymmetry_ledger_sha256"],
                "task_acceptance": {"accepted_task_count": if complete { 1 } else { 0 }, "all_submitted_tasks_protected_verifier_accepted": complete, "candidate_self_accepted": false, "protected_verifier_receipt_seal_sha256": sha('d')},
                "red_team_adjudication": {"opposing_candidate_artifact_id": other_candidate(candidate).unwrap(), "opposing_candidate_read_only": complete, "reviewer_modified_candidate_artifact": false, "protected_verifier_adjudicated_challenges": complete, "candidate_self_adjudicated": false},
                "primary_metrics": metric_map(&PRIMARY_METRICS, complete, 'a'),
                "performance_metrics": metric_map(&PERFORMANCE_METRICS, complete, 'b'),
                "manager_intelligence_metrics": metric_map(&INTELLIGENCE_METRICS, complete, 'c'),
                "orchestration_metrics": metric_map(&ORCHESTRATION_METRICS, complete, 'd'),
                "failure_recovery_injections": metric_map(&RECOVERY_INJECTIONS, complete, 'a'),
                "candidate_self_scored": false,
                "winner_selected": false,
            }))
            .unwrap(),
        )
    }

    fn pareto_binding(
        protocol: &SealedDocumentBinding,
        tg3: &SealedDocumentBinding,
        corpus: &SealedDocumentBinding,
        ledger: &SealedDocumentBinding,
        scorecards: &BTreeMap<String, BTreeMap<String, SealedDocumentBinding>>,
        complete: bool,
    ) -> SealedDocumentBinding {
        let mut bindings: BTreeMap<String, BTreeMap<String, EvidenceDigest>> = BTreeMap::new();
        for mode in MODES {
            let mut candidates = BTreeMap::new();
            for candidate in CANDIDATES {
                let scorecard = scorecards.get(mode).unwrap().get(candidate).unwrap();
                candidates.insert(
                    candidate.into(),
                    serde_json::from_value(digest_from(
                        scorecard,
                        SCORECARD_SCHEMA,
                        if complete {
                            SCORECARD_MEASURED_STATUS
                        } else {
                            SCORECARD_INCOMPLETE_STATUS
                        },
                    ))
                    .unwrap(),
                );
            }
            bindings.insert(mode.into(), candidates);
        }
        binding(
            "/sealed/pareto.json",
            seal_value(json!({
                "schema": PARETO_SCHEMA,
                "status": if complete { PARETO_COMPLETE_STATUS } else { PARETO_INCOMPLETE_STATUS },
                "final_manager_protocol_identity_sha256": FINAL_MANAGER_PROTOCOL_IDENTITY,
                "final_manager_protocol_seal_sha256": protocol.document["seal_sha256"],
                "bound_tg3_freeze_final_comparison_authority": digest_from(tg3, TG3_COMPARISON_SCHEMA, TG3_COMPARISON_STATUS),
                "bound_protected_task_corpus_commitment": digest_from(corpus, CORPUS_SCHEMA, if complete { CORPUS_PREPARED_STATUS } else { CORPUS_INCOMPLETE_STATUS }),
                "bound_resource_envelope_asymmetry_ledger": digest_from(ledger, LEDGER_SCHEMA, if complete { LEDGER_EQUALIZED_STATUS } else { LEDGER_INCOMPLETE_STATUS }),
                "candidate_artifacts": CANDIDATES,
                "evaluation_modes": MODES,
                "scorecard_bindings": bindings,
                "all_required_scorecards_bound": complete,
                "pareto_frontier_complete": complete,
                "protected_verifier_accepted": complete,
                "do_not_collapse_to_scalar_before_complete_evidence_matrix": true,
                "candidate_self_selection": false,
                "winner_selected": false,
            }))
            .unwrap(),
        )
    }

    fn input_fixture(complete: bool) -> Input {
        let protocol = protocol_binding();
        let tg3 = tg3_binding(&protocol);
        let ledger = ledger_binding(&protocol, &tg3, complete);
        let corpus = corpus_binding(&protocol, &tg3, &ledger, complete);
        let mut scorecards = BTreeMap::new();
        for mode in MODES {
            let mut candidates = BTreeMap::new();
            for candidate in CANDIDATES {
                candidates.insert(
                    candidate.into(),
                    scorecard_binding(&protocol, &tg3, &corpus, &ledger, mode, candidate, complete),
                );
            }
            scorecards.insert(mode.into(), candidates);
        }
        let pareto = pareto_binding(&protocol, &tg3, &corpus, &ledger, &scorecards, complete);
        Input {
            schema: INPUT_SCHEMA.into(),
            final_manager_protocol: protocol,
            tg3_freeze_final_comparison_authority: tg3,
            protected_task_corpus_commitment: corpus,
            resource_envelope_asymmetry_ledger: ledger,
            scorecard_evidence: scorecards,
            pareto_frontier_evidence: pareto,
            protected_selection_requested: false,
            authority_boundary: authority_boundary(),
            execution_boundary: execution_boundary(),
            seal_sha256: sha('f'),
        }
    }

    fn seal_input(input: &Input) -> Value {
        let mut value = serde_json::to_value(input).unwrap();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal_value(value).unwrap()
    }

    #[test]
    fn incomplete_evidence_refuses_without_selecting_a_winner() {
        let report = build_report(input_fixture(false)).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(!report.tournament_active);
        assert!(!report.winner_selected);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("resource_envelope_asymmetry_ledger")));
    }

    #[test]
    fn complete_both_mode_evidence_prepares_but_never_selects() {
        let report = build_report(input_fixture(true)).unwrap();
        assert_eq!(
            report.status, PREPARED_STATUS,
            "blockers: {:?}",
            report.blockers
        );
        assert!(report.prepared);
        assert_eq!(report.scorecard_audits.len(), 4);
        assert!(report.pareto_frontier_complete);
        assert!(!report.tournament_active);
        assert!(!report.scorecards_executed_by_this_contract);
        assert!(!report.winner_selected);
        assert!(report.protected_selection_authority_required);
    }

    #[test]
    fn unequal_resource_ledger_or_self_scoring_refuses() {
        let mut unequal = input_fixture(true);
        let scorecard = unequal
            .scorecard_evidence
            .get_mut("SOLO_MANAGER")
            .unwrap()
            .get_mut(CANDIDATES[0])
            .unwrap();
        scorecard.document["resource_envelope_sha256"] = Value::String(sha('0'));
        let document = scorecard.document.clone();
        scorecard.document = seal_value(document).unwrap();
        scorecard.document_sha256 = sha256_json(&scorecard.document).unwrap();
        let report = build_report(unequal).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("resource_envelope_or_asymmetry_ledger_mismatch")));

        let mut self_scored = input_fixture(true);
        let scorecard = self_scored
            .scorecard_evidence
            .get_mut("MANAGER_AS_ORCHESTRATOR")
            .unwrap()
            .get_mut(CANDIDATES[1])
            .unwrap();
        scorecard.document["candidate_self_scored"] = Value::Bool(true);
        let document = scorecard.document.clone();
        scorecard.document = seal_value(document).unwrap();
        scorecard.document_sha256 = sha256_json(&scorecard.document).unwrap();
        let report = build_report(self_scored).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("candidate_self_scored_or_winner_preselected")));
    }

    #[test]
    fn protocol_tg3_and_pareto_bindings_are_fail_closed() {
        let mut protocol = input_fixture(true);
        protocol.final_manager_protocol.document["protocol_identity_sha256"] =
            Value::String(sha('0'));
        let document = protocol.final_manager_protocol.document.clone();
        protocol.final_manager_protocol.document = seal_value(document).unwrap();
        protocol.final_manager_protocol.document_sha256 =
            sha256_json(&protocol.final_manager_protocol.document).unwrap();
        assert!(build_report(protocol).is_err());

        let mut pareto = input_fixture(true);
        pareto.pareto_frontier_evidence.document["scorecard_bindings"]["SOLO_MANAGER"]
            [CANDIDATES[0]]["document_sha256"] = Value::String(sha('0'));
        let document = pareto.pareto_frontier_evidence.document.clone();
        pareto.pareto_frontier_evidence.document = seal_value(document).unwrap();
        pareto.pareto_frontier_evidence.document_sha256 =
            sha256_json(&pareto.pareto_frontier_evidence.document).unwrap();
        let report = build_report(pareto).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.pareto_frontier_complete);
    }

    #[test]
    fn protected_selection_request_is_refused_by_this_contract() {
        let mut input = input_fixture(true);
        input.protected_selection_requested = true;
        let report = build_report(input).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(!report.winner_selected);
    }

    #[test]
    fn sealed_input_output_and_create_new_are_enforced_cpu_only() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("input.json");
        let output_path = directory.path().join("out.json");
        fs::write(
            &input_path,
            serde_json::to_vec_pretty(&seal_input(&input_fixture(false))).unwrap(),
        )
        .unwrap();
        run(Args {
            input: input_path,
            out: output_path.clone(),
        })
        .unwrap();
        let output: Value = serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
        verify_sealed_object(&output, "output").unwrap();
        assert_eq!(output["status"], Value::String(REFUSED_STATUS.into()));
        assert_eq!(output["winner_selected"], Value::Bool(false));
        assert!(write_report_create_new(
            &output_path,
            &build_report(input_fixture(false)).unwrap()
        )
        .is_err());
    }
}
