//! Sealed TG3/freeze final-manager comparison preparation authority.
//!
//! This is deliberately a CPU-only, fail-closed preparatory authority.  It
//! binds the already-sealed paired-cognition authorities, the fixed
//! final-manager tournament protocol, and sealed per-artifact qualification
//! evidence.  It can only report `PREPARED_NOT_ACTIVE` or `REFUSED`; it cannot
//! launch a tournament, start a server or watcher, use a GPU, run a model,
//! measure TPS, score a candidate, or select a winner.
//!
//! A prepared report means only that the exact two candidate artifacts have
//! independently passed TG3, have been frozen after TG3, and have supplied all
//! hard-gate provenance/replay contracts needed by the protected later
//! tournament controller.  The protected controller/human remains the only
//! authority that may execute, score, or select.

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_tg3_freeze_final_comparison_authority_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_tg3_freeze_final_comparison_authority.v1";
const PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_TG3_FROZEN_FINAL_MANAGER_COMPARISON_RESERVED";
const REFUSED_STATUS: &str =
    "REFUSED_PAIRED_COGNITION_TG3_FROZEN_FINAL_MANAGER_COMPARISON_HARD_GATES_INCOMPLETE";

const FINAL_MANAGER_PROTOCOL_SCHEMA: &str =
    "hawking.ascension.final_manager_tournament_protocol.v1";
const FINAL_MANAGER_PROTOCOL_STATUS: &str =
    "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED";
// This is the identity currently armed by the Ascension lifecycle.  Pinning it
// prevents a resealed, weaker protocol from being substituted as preparation
// evidence.
const FINAL_MANAGER_PROTOCOL_IDENTITY: &str =
    "8e3684af0b7de53690a9c88ce0d52b0cae019e0d798bc2748b5b1556211facf8";

const LANE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_lane_namespace_mission_authority.v1";
const LANE_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_TWO_SEALED_CANDIDATE_WORLDS_NO_RUNTIME_SERVER_OR_TOURNAMENT";
const MUTATION_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_proposal_review_falsification_primary_acceptance_authority.v1";
const MUTATION_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_PRIMARY_ONLY_CHAMPION_MUTATION_PROMOTION_NO_MANAGER_OR_TOURNAMENT_SELECTION";
const KNOWLEDGE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_knowledge_plane_generic_release_authority.v1";
const KNOWLEDGE_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_GENERIC_ONLY_INDEPENDENTLY_VERIFIED_APPEND_ONLY_KNOWLEDGE_PLANE_NO_RUNTIME_OR_TOURNAMENT";
const SCHEDULER_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_one_body_many_logical_session_role_resource_scheduler_authority.v1";
const SCHEDULER_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_ONE_Q30_ONE_Q80_FOUR_ISOLATED_LOGICAL_ROLES_NO_RUNTIME_OR_WINNER_SELECTION";
const TG10_ACTIVATION_SCHEMA: &str =
    "hawking.ascension.paired_cognition_both_tg10_development_activation_state_machine.v1";
const TG10_ACTIVATION_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_EXACT_TG10_OPERATIONAL_RECEIPTS_BOUND_NO_RUNTIME_OR_TOURNAMENT";

const COMPLETE_MANAGER_QUALIFICATION_SCHEMA: &str =
    "hawking.ascension.paired_cognition_complete_manager_qualification_evidence.v1";
const COMPLETE_MANAGER_QUALIFIED_STATUS: &str = "QUALIFIED_COMPLETE_MANAGER";
const COMPLETE_MANAGER_PENDING_STATUS: &str = "PENDING_OR_UNQUALIFIED_COMPLETE_MANAGER";
const TG3_RECEIPT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_tg3_base_true_tps_operational_receipt.v1";
const TG3_QUALIFIED_STATUS: &str = "QUALIFIED_TG3";
const TG3_PENDING_STATUS: &str = "PENDING_OR_UNQUALIFIED_TG3";
const POST_TG3_FREEZE_SCHEMA: &str =
    "hawking.ascension.paired_cognition_post_tg3_freeze_receipt.v1";
const POST_TG3_FROZEN_STATUS: &str = "FROZEN_POST_TG3";
const POST_TG3_NOT_FROZEN_STATUS: &str = "NOT_FROZEN_POST_TG3";
const HARD_GATE_PROVENANCE_SCHEMA: &str =
    "hawking.ascension.paired_cognition_final_manager_hard_gate_provenance.v1";
const HARD_GATE_PASS_STATUS: &str = "PASS";
const HARD_GATE_PENDING_STATUS: &str = "PENDING_OR_FAILED";
const REPAIR_REPLAY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_post_freeze_repair_replay_receipt.v1";
const REPAIR_REPLAY_COMPLETE_STATUS: &str = "AFFECTED_SCORED_TESTS_REPLAYED";

const QWEN30: &str = "qwen30";
const QWEN80: &str = "qwen80";
const CANDIDATE_ARTIFACTS: [&str; 2] = [
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
];
const EVALUATION_MODES: [&str; 2] = ["SOLO_MANAGER", "MANAGER_AS_ORCHESTRATOR"];
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
const LONG_HORIZON_CAMPAIGNS: [&str; 6] = [
    "six_stage_kernel_optimization",
    "multi_subsystem_hcli_repair",
    "exact_model_family_qualification",
    "storage_pressure_model_acquisition",
    "agent_os_concurrency_optimization",
    "gravity_representation_tournament",
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
const MANAGER_INTELLIGENCE_METRICS: [&str; 14] = [
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
const FAILURE_RECOVERY_INJECTIONS: [&str; 9] = [
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
// The final-manager protocol has exactly 21 required side-by-side report
// fields.  The comparison is exact rather than a permissive prefix match.
const FINAL_REPORT_FIELDS_EXACT: [&str; 21] = [
    "bpw",
    "tg_rung",
    "base_true_tps",
    "p99",
    "memory",
    "verified_tasks_per_hour",
    "solo_capability",
    "orchestrated_capability",
    "repository_architecture_tasks",
    "kernel_tasks",
    "gravity_tasks",
    "tool_reliability",
    "delegation",
    "context",
    "restart",
    "failure_recovery",
    "adversarial_review",
    "resource_efficiency",
    "long_horizon_completion",
    "human_interventions",
    "hard_gate_status",
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct Tg10Authority {
    required_base_true_tps: u16,
    operational_pass: bool,
    coherent_hcli_pass: bool,
    complete_token_path_measured: bool,
    fallback_count: usize,
    median_base_true_tps: Option<f64>,
    receipt_seal_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct LaneModelContractBinding {
    model_key: String,
    tg10: Tg10Authority,
}

#[derive(Clone, Debug, Deserialize)]
struct LaneAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    no_new_physical_model_process_authority: bool,
    model_contract_bindings: Vec<LaneModelContractBinding>,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize)]
struct MutationAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    bound_lane_authority: EvidenceDigest,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize)]
struct KnowledgeAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    knowledge_plane_active: bool,
    external_publication_performed: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalModeGateReport {
    solo_manager_evaluation_authorized_by_this_contract: bool,
    symmetric_orchestrator_evaluation_authorized_by_this_contract: bool,
    winner_selection_authorized_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct SchedulerAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_development_active: bool,
    logical_sessions_created: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    final_mode_gate: FinalModeGateReport,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize)]
struct Tg10ActivationAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_development_active: bool,
    paired_development_activation_authorized_by_this_contract: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    bound_scheduler_authority: EvidenceDigest,
    both_exact_fresh_tg10_operational_receipts_present: bool,
    state_blockers: Vec<String>,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct RepairReplayRequirement {
    post_tg3_repair_occurred: bool,
    replay_required_for_repair: bool,
    affected_scored_tests_replayed: bool,
    replay_receipt: Option<SealedDocumentBinding>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HardGateProvenance {
    passed: bool,
    gate_receipt: SealedDocumentBinding,
    repair_replay: RepairReplayRequirement,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CandidateQualification {
    candidate_artifact_id: String,
    source_identity_seal_sha256: String,
    complete_artifact_admission_seal_sha256: String,
    complete_manager_qualification_receipt: SealedDocumentBinding,
    tg3_operational_receipt: SealedDocumentBinding,
    post_tg3_freeze_receipt: SealedDocumentBinding,
    hard_gate_provenance: BTreeMap<String, HardGateProvenance>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModeReservation {
    reserved_for_both_candidates: bool,
    scored_execution_started: bool,
    candidate_results_recorded: bool,
    same_frozen_task_corpus: bool,
    identical_resource_envelope: bool,
    no_helper_advantage: bool,
    same_hawking_agent_os: bool,
    symmetric_helper_model_infrastructure: bool,
    no_hidden_information_or_worker_advantage: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtectedTaskCorpusReservation {
    protected_corpus_sealed: bool,
    blind_tasks_frozen: bool,
    hidden_membership_inaccessible_to_candidates: bool,
    real_hawking_work_only: bool,
    required_families: Vec<String>,
    tasks_executed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AdversarialReviewReservation {
    opposing_candidate_read_only_red_team: bool,
    reviewer_may_not_modify_candidate_artifact: bool,
    protected_verifier_adjudicates_challenges: bool,
    no_candidate_self_scoring: bool,
    review_executed: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FairnessReservation {
    asymmetries_recorded: bool,
    equalized: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProtectedSelectionReservation {
    pareto_frontier_required_before_selection: bool,
    do_not_collapse_to_scalar_before_complete_evidence_matrix: bool,
    candidates_cannot_self_grade: bool,
    candidates_cannot_change_weights_or_hidden_tests: bool,
    candidates_cannot_promote_self_or_invalidate_opponent: bool,
    only_protected_controller_or_human_may_select_manager: bool,
    winner_selected: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct WinnerFreezeAndLoserRestoreReservation {
    winner_freeze_contract_reserved: bool,
    loser_cold_store_contract_reserved: bool,
    loser_restore_hash_verify_one_command_required: bool,
    no_delete_loser_before_restore_proof: bool,
    alternate_evicted: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FinalComparisonReservation {
    tournament_activation_requested: bool,
    scored_task_execution_requested: bool,
    winner_selection_requested: bool,
    evaluation_modes: BTreeMap<String, ModeReservation>,
    protected_task_corpus: ProtectedTaskCorpusReservation,
    adversarial_review: AdversarialReviewReservation,
    fairness: FairnessReservation,
    protected_selection: ProtectedSelectionReservation,
    winner_freeze_and_loser_restore: WinnerFreezeAndLoserRestoreReservation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    final_manager_protocol: SealedDocumentBinding,
    lane_authority: SealedDocumentBinding,
    mutation_authority: SealedDocumentBinding,
    knowledge_authority: SealedDocumentBinding,
    scheduler_authority: SealedDocumentBinding,
    tg10_activation_authority: SealedDocumentBinding,
    candidate_qualifications: BTreeMap<String, CandidateQualification>,
    final_comparison_reservation: FinalComparisonReservation,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct CandidateAudit {
    candidate_artifact_id: String,
    complete_manager_qualified: bool,
    tg3_qualified: bool,
    post_tg3_frozen: bool,
    every_hard_gate_passed_with_provenance: bool,
    all_repair_replay_contracts_present: bool,
    blockers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    fixed_final_manager_protocol_identity_and_full_contract_bound: bool,
    sealed_lane_mutation_knowledge_scheduler_and_tg10_authority_chain_bound: bool,
    qwen30_and_qwen80_complete_manager_tg3_freeze_and_hard_gates_required: bool,
    both_identical_evaluation_modes_reserved_but_not_executed: bool,
    blind_protected_corpus_red_team_verifier_and_no_self_scoring_reserved: bool,
    pareto_before_selection_and_winner_loser_restore_contract_reserved: bool,
    no_runtime_server_gpu_watcher_tps_or_tournament_authority: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: String,
    prepared: bool,
    tournament_active: bool,
    scored_task_execution_active: bool,
    winner_selected: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    bound_scheduler_authority: EvidenceDigest,
    bound_tg10_activation_authority: EvidenceDigest,
    qwen30_audit: CandidateAudit,
    qwen80_audit: CandidateAudit,
    final_comparison_reservation_complete: bool,
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
                "{label}.{field} must be false for this CPU-only authority"
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
        return Err(format!(
            "{label} must bind the exact sealed source authority"
        ));
    }
    Ok(())
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

fn require_string_at<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a str, String> {
    value_at(value, path, label)?
        .as_str()
        .filter(|item| !item.trim().is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn require_usize_at(
    value: &Value,
    path: &[&str],
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let observed = value_at(value, path, label)?
        .as_u64()
        .ok_or_else(|| format!("{label} must be an unsigned integer"))?;
    if observed != expected as u64 {
        return Err(format!("{label} must be {expected}"));
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
    let expected = expected
        .iter()
        .map(|item| (*item).to_owned())
        .collect::<Vec<_>>();
    if observed != expected {
        return Err(format!("{label} drift from the protected protocol"));
    }
    Ok(())
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
    for key in expected {
        let required = rows
            .get(*key)
            .and_then(Value::as_object)
            .and_then(|row| row.get("required"))
            .and_then(Value::as_bool);
        if required != Some(true) {
            return Err(format!("{label}.{key}.required must be true"));
        }
    }
    Ok(())
}

fn validate_tg10_authority(tg10: &Tg10Authority, model_key: &str) -> Result<(), String> {
    if tg10.required_base_true_tps != 100 {
        return Err(format!("{model_key}.tg10 must require 100 BASE_TRUE_TPS"));
    }
    if !tg10.operational_pass {
        return Ok(());
    }
    if !tg10.coherent_hcli_pass || !tg10.complete_token_path_measured || tg10.fallback_count != 0 {
        return Err(format!(
            "{model_key}.tg10 pass must be coherent, complete, and fallback-free"
        ));
    }
    let measured = tg10
        .median_base_true_tps
        .ok_or_else(|| format!("{model_key}.tg10 pass lacks median BASE_TRUE_TPS"))?;
    if !measured.is_finite() || measured < 100.0 {
        return Err(format!(
            "{model_key}.tg10 pass must measure >=100 BASE_TRUE_TPS"
        ));
    }
    let seal = tg10
        .receipt_seal_sha256
        .as_deref()
        .ok_or_else(|| format!("{model_key}.tg10 pass lacks receipt seal"))?;
    if !is_lower_sha256(seal) {
        return Err(format!(
            "{model_key}.tg10 receipt seal must be lowercase SHA-256"
        ));
    }
    Ok(())
}

fn validate_lane_authority(
    binding: &SealedDocumentBinding,
) -> Result<(EvidenceDigest, BTreeMap<String, Tg10Authority>), String> {
    let digest = binding_digest(binding, "lane_authority")?;
    let document: LaneAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("lane_authority.document has wrong grammar: {error}"))?;
    if document.schema != LANE_AUTHORITY_SCHEMA || document.status != LANE_AUTHORITY_STATUS {
        return Err("lane_authority is not the completed paired lane authority".into());
    }
    if !document.prepared
        || document.paired_candidate_worlds_active
        || !document.no_new_physical_model_process_authority
    {
        return Err("lane_authority must remain prepared, inactive, and process-free".into());
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "lane_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "lane_authority.execution_boundary",
    )?;
    let mut models = BTreeMap::new();
    for contract in document.model_contract_bindings {
        if contract.model_key != QWEN30 && contract.model_key != QWEN80 {
            return Err("lane_authority contains unsupported model TG10 authority".into());
        }
        validate_tg10_authority(&contract.tg10, &contract.model_key)?;
        if models
            .insert(contract.model_key.clone(), contract.tg10)
            .is_some()
        {
            return Err("lane_authority has duplicate model TG10 authorities".into());
        }
    }
    if models.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from([QWEN30, QWEN80])
    {
        return Err("lane_authority must bind exactly Q30 and Q80 TG10 authorities".into());
    }
    Ok((digest, models))
}

fn validate_mutation_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "mutation_authority")?;
    let document: MutationAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("mutation_authority.document has wrong grammar: {error}"))?;
    if document.schema != MUTATION_AUTHORITY_SCHEMA || document.status != MUTATION_AUTHORITY_STATUS
    {
        return Err("mutation_authority is not the completed paired mutation authority".into());
    }
    if !document.prepared || document.paired_candidate_worlds_active {
        return Err("mutation_authority must remain prepared and inactive".into());
    }
    exact_digest(
        lane_digest,
        &document.bound_lane_authority,
        "mutation_authority",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "mutation_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "mutation_authority.execution_boundary",
    )?;
    Ok(digest)
}

fn validate_knowledge_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
    mutation_digest: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "knowledge_authority")?;
    let document: KnowledgeAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("knowledge_authority.document has wrong grammar: {error}"))?;
    if document.schema != KNOWLEDGE_AUTHORITY_SCHEMA
        || document.status != KNOWLEDGE_AUTHORITY_STATUS
    {
        return Err("knowledge_authority is not the completed generic-only authority".into());
    }
    if !document.prepared
        || document.knowledge_plane_active
        || document.external_publication_performed
    {
        return Err(
            "knowledge_authority must remain prepared, inactive, and externally unpublished".into(),
        );
    }
    exact_digest(
        lane_digest,
        &document.bound_lane_authority,
        "knowledge_authority",
    )?;
    exact_digest(
        mutation_digest,
        &document.bound_mutation_authority,
        "knowledge_authority",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "knowledge_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "knowledge_authority.execution_boundary",
    )?;
    Ok(digest)
}

fn validate_scheduler_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
    mutation_digest: &EvidenceDigest,
    knowledge_digest: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "scheduler_authority")?;
    let document: SchedulerAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("scheduler_authority.document has wrong grammar: {error}"))?;
    if document.schema != SCHEDULER_AUTHORITY_SCHEMA
        || document.status != SCHEDULER_AUTHORITY_STATUS
    {
        return Err("scheduler_authority is not the completed one-body role scheduler".into());
    }
    if !document.prepared || document.paired_development_active || document.logical_sessions_created
    {
        return Err(
            "scheduler_authority must remain prepared with no active development/session".into(),
        );
    }
    exact_digest(
        lane_digest,
        &document.bound_lane_authority,
        "scheduler_authority",
    )?;
    exact_digest(
        mutation_digest,
        &document.bound_mutation_authority,
        "scheduler_authority",
    )?;
    exact_digest(
        knowledge_digest,
        &document.bound_knowledge_authority,
        "scheduler_authority",
    )?;
    if document
        .final_mode_gate
        .solo_manager_evaluation_authorized_by_this_contract
        || document
            .final_mode_gate
            .symmetric_orchestrator_evaluation_authorized_by_this_contract
        || document
            .final_mode_gate
            .winner_selection_authorized_by_this_contract
    {
        return Err("scheduler_authority must keep final evaluation and winner selection outside this authority".into());
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "scheduler_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "scheduler_authority.execution_boundary",
    )?;
    Ok(digest)
}

fn validate_tg10_activation_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
    mutation_digest: &EvidenceDigest,
    knowledge_digest: &EvidenceDigest,
    scheduler_digest: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "tg10_activation_authority")?;
    let document: Tg10ActivationAuthorityDocument =
        serde_json::from_value(binding.document.clone()).map_err(|error| {
            format!("tg10_activation_authority.document has wrong grammar: {error}")
        })?;
    if document.schema != TG10_ACTIVATION_SCHEMA || document.status != TG10_ACTIVATION_STATUS {
        return Err(
            "tg10_activation_authority is not the prepared exact-both-TG10 authority".into(),
        );
    }
    if !document.prepared
        || document.paired_development_active
        || document.paired_development_activation_authorized_by_this_contract
        || !document.both_exact_fresh_tg10_operational_receipts_present
        || !document.state_blockers.is_empty()
    {
        return Err(
            "tg10_activation_authority must be prepared, inactive, exact, and blocker-free".into(),
        );
    }
    exact_digest(
        lane_digest,
        &document.bound_lane_authority,
        "tg10_activation_authority",
    )?;
    exact_digest(
        mutation_digest,
        &document.bound_mutation_authority,
        "tg10_activation_authority",
    )?;
    exact_digest(
        knowledge_digest,
        &document.bound_knowledge_authority,
        "tg10_activation_authority",
    )?;
    exact_digest(
        scheduler_digest,
        &document.bound_scheduler_authority,
        "tg10_activation_authority",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "tg10_activation_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "tg10_activation_authority.execution_boundary",
    )?;
    Ok(digest)
}

fn validate_final_manager_protocol(
    binding: &SealedDocumentBinding,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "final_manager_protocol")?;
    let document = &binding.document;
    if digest.document_schema != FINAL_MANAGER_PROTOCOL_SCHEMA
        || digest.document_status != FINAL_MANAGER_PROTOCOL_STATUS
    {
        return Err("final_manager_protocol has the wrong schema or status".into());
    }
    let identity = require_string_at(
        document,
        &["protocol_identity_sha256"],
        "final_manager_protocol.protocol_identity_sha256",
    )?;
    if !is_lower_sha256(identity) || identity != FINAL_MANAGER_PROTOCOL_IDENTITY {
        return Err(
            "final_manager_protocol identity is not the current protected protocol identity".into(),
        );
    }
    let object = document
        .as_object()
        .ok_or("final_manager_protocol must be an object")?;
    let mut body = object.clone();
    body.remove("seal_sha256");
    body.remove("recorded_at");
    body.remove("protocol_identity_sha256");
    if sha256_json(&Value::Object(body))? != identity {
        return Err(
            "final_manager_protocol identity does not match its sealed protocol body".into(),
        );
    }

    require_bool_at(
        document,
        &["ascent_boundary", "development_race_is_not_final_selection"],
        true,
        "protocol ascent development_race_is_not_final_selection",
    )?;
    require_bool_at(
        document,
        &[
            "ascent_boundary",
            "candidate_specific_optimization_allowed_only_before_freeze",
        ],
        true,
        "protocol ascent candidate_specific_optimization_allowed_only_before_freeze",
    )?;
    require_bool_at(
        document,
        &[
            "ascent_boundary",
            "freeze_requires_both_complete_manager_qualification_floors",
        ],
        true,
        "protocol ascent freeze_requires_both_complete_manager_qualification_floors",
    )?;
    require_bool_at(
        document,
        &[
            "ascent_boundary",
            "post_freeze_candidate_specific_change_requires_failed_hard_gate_repair",
        ],
        true,
        "protocol ascent post_freeze_candidate_specific_change_requires_failed_hard_gate_repair",
    )?;
    require_bool_at(
        document,
        &[
            "ascent_boundary",
            "repair_requires_replay_of_all_affected_scored_tests",
        ],
        true,
        "protocol ascent repair_requires_replay_of_all_affected_scored_tests",
    )?;
    require_string_list_at(
        document,
        &["candidates", "fixed_order"],
        &CANDIDATE_ARTIFACTS,
        "protocol candidates.fixed_order",
    )?;
    require_bool_at(
        document,
        &[
            "candidates",
            "both_must_meet_complete_manager_qualification_floor_before_freeze",
        ],
        true,
        "protocol candidates.complete_manager_floor",
    )?;
    for key in [
        "both_candidates_frozen_before_scored_execution",
        "candidate_specific_optimization_disabled_during_scored_execution",
        "hard_gate_repair_requires_affected_test_replay",
    ] {
        require_bool_at(
            document,
            &["freeze_contract", key],
            true,
            &format!("protocol freeze_contract.{key}"),
        )?;
    }
    require_string_list_at(
        document,
        &["evaluation_modes", "required"],
        &EVALUATION_MODES,
        "protocol evaluation_modes.required",
    )?;
    for key in ["no_helper_advantage", "each_candidate_measured"] {
        require_bool_at(
            document,
            &["evaluation_modes", "SOLO_MANAGER", key],
            true,
            &format!("protocol SOLO_MANAGER.{key}"),
        )?;
    }
    for key in [
        "same_hawking_agent_os",
        "symmetric_helper_model_infrastructure",
        "no_hidden_information_or_worker_advantage",
        "each_candidate_measured",
    ] {
        require_bool_at(
            document,
            &["evaluation_modes", "MANAGER_AS_ORCHESTRATOR", key],
            true,
            &format!("protocol MANAGER_AS_ORCHESTRATOR.{key}"),
        )?;
    }
    require_bool_at(
        document,
        &["hard_gates", "conjunctive"],
        true,
        "protocol hard_gates.conjunctive",
    )?;
    require_bool_at(
        document,
        &["hard_gates", "no_score_can_compensate_for_failure"],
        true,
        "protocol hard_gates.no_score_can_compensate_for_failure",
    )?;
    require_bool_at(
        document,
        &["hard_gates", "each_gate_required_for_each_candidate"],
        true,
        "protocol hard_gates.each_gate_required_for_each_candidate",
    )?;
    require_required_rows(
        document,
        &["hard_gates", "required"],
        &HARD_GATES,
        "protocol hard_gates.required",
    )?;
    for key in [
        "real_hawking_work_only",
        "blind_tasks_required",
        "hidden_membership_frozen_before_scored_execution",
    ] {
        require_bool_at(
            document,
            &["protected_task_corpus", key],
            true,
            &format!("protocol protected_task_corpus.{key}"),
        )?;
    }
    require_string_list_at(
        document,
        &["protected_task_corpus", "required_families"],
        &TASK_FAMILIES,
        "protocol protected_task_corpus.required_families",
    )?;
    require_string_list_at(
        document,
        &["long_horizon_campaigns", "required"],
        &LONG_HORIZON_CAMPAIGNS,
        "protocol long_horizon_campaigns.required",
    )?;
    require_bool_at(
        document,
        &["long_horizon_campaigns", "measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion"],
        true,
        "protocol long_horizon_campaigns.measurement",
    )?;
    for key in [
        "required_after_each_candidate_task",
        "opposing_candidate_is_read_only_red_team_reviewer",
        "reviewer_may_not_modify_candidate_artifact",
        "protected_verifier_adjudicates_challenges",
        "measure_genuine_defects_and_false_objections",
    ] {
        require_bool_at(
            document,
            &["adversarial_review", key],
            true,
            &format!("protocol adversarial_review.{key}"),
        )?;
    }
    require_bool_at(
        document,
        &["fairness_envelope", "record_every_asymmetry"],
        true,
        "protocol fairness_envelope.record_every_asymmetry",
    )?;
    require_required_rows(
        document,
        &["fairness_envelope", "equalize"],
        &FAIRNESS_ENVELOPE,
        "protocol fairness_envelope.equalize",
    )?;
    require_string_at(
        document,
        &["scorecards", "primary_metric"],
        "protocol scorecards.primary_metric",
    )
    .and_then(|value| {
        (value == "verified_tasks_per_hour")
            .then_some(())
            .ok_or_else(|| {
                "protocol scorecards.primary_metric must be verified_tasks_per_hour".into()
            })
    })?;
    for (key, expected) in [
        ("primary_metrics", &PRIMARY_METRICS[..]),
        ("performance_metrics", &PERFORMANCE_METRICS[..]),
        (
            "manager_intelligence_metrics",
            &MANAGER_INTELLIGENCE_METRICS[..],
        ),
        ("orchestration_metrics", &ORCHESTRATION_METRICS[..]),
        (
            "failure_recovery_injections",
            &FAILURE_RECOVERY_INJECTIONS[..],
        ),
    ] {
        require_string_list_at(
            document,
            &["scorecards", key],
            expected,
            &format!("protocol scorecards.{key}"),
        )?;
    }
    for key in [
        "pareto_frontier_required_before_selection",
        "do_not_collapse_to_scalar_before_complete_evidence_matrix",
    ] {
        require_bool_at(
            document,
            &["scorecards", key],
            true,
            &format!("protocol scorecards.{key}"),
        )?;
    }
    for key in [
        "candidate_self_assessments_are_evidence_only",
        "candidates_cannot_self_grade",
        "candidates_cannot_change_weights_or_hidden_tests",
        "candidates_cannot_promote_self_or_invalidate_opponent",
        "only_protected_controller_or_human_may_select_manager",
    ] {
        require_bool_at(
            document,
            &["protected_selection", key],
            true,
            &format!("protocol protected_selection.{key}"),
        )?;
    }
    for key in [
        "seal_winner_source_artifact_runtime_kernel_agent_os_context_kv_benchmarks_capability_tournament_and_rollback",
        "seal_loser_before_any_evictability",
        "cold_store_hash_verify_restore_test_one_command_recovery_required",
        "do_not_delete_loser_before_restore_proof",
    ] {
        require_bool_at(
            document,
            &["winner_freeze_and_alternate", key],
            true,
            &format!("protocol winner_freeze_and_alternate.{key}"),
        )?;
    }
    require_bool_at(
        document,
        &["final_report", "side_by_side_candidates_required"],
        true,
        "protocol final_report.side_by_side_candidates_required",
    )?;
    require_string_list_at(
        document,
        &["final_report", "required_fields"],
        &FINAL_REPORT_FIELDS_EXACT,
        "protocol final_report.required_fields",
    )?;
    require_bool_at(
        document,
        &[
            "final_report",
            "must_state_winner_loser_decisive_evidence_tradeoffs_restore_path",
        ],
        true,
        "protocol final_report.must_state_winner_loser_decisive_evidence_tradeoffs_restore_path",
    )?;
    for key in [
        "does_not_execute_tournament",
        "does_not_score_candidates",
        "does_not_choose_winner",
        "does_not_activate_a_server_or_sandbox",
        "does_not_relax_tg3_or_other_hard_gates",
    ] {
        require_bool_at(
            document,
            &["claim_boundary", key],
            true,
            &format!("protocol claim_boundary.{key}"),
        )?;
    }
    Ok(digest)
}

fn validate_candidate_document_identity(
    binding: &SealedDocumentBinding,
    label: &str,
    candidate: &CandidateQualification,
    expected_schema: &str,
    permitted_statuses: &[&str],
) -> Result<(EvidenceDigest, String), String> {
    let digest = binding_digest(binding, label)?;
    if digest.document_schema != expected_schema
        || !permitted_statuses.contains(&digest.document_status.as_str())
    {
        return Err(format!(
            "{label} has an unexpected evidence schema or status"
        ));
    }
    let document = &binding.document;
    if require_string_at(
        document,
        &["candidate_artifact_id"],
        &format!("{label}.candidate_artifact_id"),
    )? != candidate.candidate_artifact_id
    {
        return Err(format!("{label} does not bind the candidate artifact"));
    }
    let source = require_string_at(
        document,
        &["source_identity_seal_sha256"],
        &format!("{label}.source_identity_seal_sha256"),
    )?;
    let admission = require_string_at(
        document,
        &["complete_artifact_admission_seal_sha256"],
        &format!("{label}.complete_artifact_admission_seal_sha256"),
    )?;
    if !is_lower_sha256(source)
        || !is_lower_sha256(admission)
        || source != candidate.source_identity_seal_sha256
        || admission != candidate.complete_artifact_admission_seal_sha256
    {
        return Err(format!(
            "{label} source/admission provenance does not match the candidate"
        ));
    }
    Ok((
        digest,
        binding.document["status"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
    ))
}

fn append_if_false(blockers: &mut Vec<String>, condition: bool, blocker: impl Into<String>) {
    if !condition {
        blockers.push(blocker.into());
    }
}

fn audit_candidate(candidate: &CandidateQualification) -> Result<CandidateAudit, String> {
    if !is_lower_sha256(&candidate.source_identity_seal_sha256)
        || !is_lower_sha256(&candidate.complete_artifact_admission_seal_sha256)
    {
        return Err(format!(
            "{} has invalid source/admission SHA-256 provenance",
            candidate.candidate_artifact_id
        ));
    }
    let mut blockers = Vec::new();
    let (qualification_digest, qualification_status) = validate_candidate_document_identity(
        &candidate.complete_manager_qualification_receipt,
        "complete_manager_qualification_receipt",
        candidate,
        COMPLETE_MANAGER_QUALIFICATION_SCHEMA,
        &[
            COMPLETE_MANAGER_QUALIFIED_STATUS,
            COMPLETE_MANAGER_PENDING_STATUS,
        ],
    )?;
    let qualification_document = &candidate.complete_manager_qualification_receipt.document;
    let complete_manager_qualified = qualification_status == COMPLETE_MANAGER_QUALIFIED_STATUS
        && value_at(
            qualification_document,
            &["complete_manager_qualified"],
            "complete_manager_qualification_receipt.complete_manager_qualified",
        )?
        .as_bool()
            == Some(true);
    append_if_false(
        &mut blockers,
        complete_manager_qualified,
        format!(
            "{}:complete_manager_qualification_missing",
            candidate.candidate_artifact_id
        ),
    );

    let (tg3_digest, tg3_status) = validate_candidate_document_identity(
        &candidate.tg3_operational_receipt,
        "tg3_operational_receipt",
        candidate,
        TG3_RECEIPT_SCHEMA,
        &[TG3_QUALIFIED_STATUS, TG3_PENDING_STATUS],
    )?;
    let tg3_document = &candidate.tg3_operational_receipt.document;
    let tg3_truths = [
        value_at(
            tg3_document,
            &["tg3_qualified"],
            "tg3_operational_receipt.tg3_qualified",
        )?
        .as_bool()
            == Some(true),
        value_at(
            tg3_document,
            &["base_true_tps_pass"],
            "tg3_operational_receipt.base_true_tps_pass",
        )?
        .as_bool()
            == Some(true),
        value_at(
            tg3_document,
            &["real_metal"],
            "tg3_operational_receipt.real_metal",
        )?
        .as_bool()
            == Some(true),
        value_at(
            tg3_document,
            &["coherent_real_hcli"],
            "tg3_operational_receipt.coherent_real_hcli",
        )?
        .as_bool()
            == Some(true),
        value_at(
            tg3_document,
            &["complete_token_path_measured"],
            "tg3_operational_receipt.complete_token_path_measured",
        )?
        .as_bool()
            == Some(true),
    ];
    let tg3_measurement = value_at(
        tg3_document,
        &["median_base_true_tps"],
        "tg3_operational_receipt.median_base_true_tps",
    )?
    .as_f64()
    .is_some_and(|value| value.is_finite() && value >= 333.0);
    let tg3_exact_contract = value_at(
        tg3_document,
        &["tg_level"],
        "tg3_operational_receipt.tg_level",
    )?
    .as_u64()
        == Some(3)
        && value_at(
            tg3_document,
            &["required_base_true_tps"],
            "tg3_operational_receipt.required_base_true_tps",
        )?
        .as_u64()
            == Some(333)
        && value_at(
            tg3_document,
            &["fallback_count"],
            "tg3_operational_receipt.fallback_count",
        )?
        .as_u64()
            == Some(0);
    let tg3_qualified = tg3_status == TG3_QUALIFIED_STATUS
        && tg3_truths.into_iter().all(|value| value)
        && tg3_measurement
        && tg3_exact_contract;
    append_if_false(
        &mut blockers,
        tg3_qualified,
        format!(
            "{}:tg3_operational_qualification_missing",
            candidate.candidate_artifact_id
        ),
    );

    let (freeze_digest, freeze_status) = validate_candidate_document_identity(
        &candidate.post_tg3_freeze_receipt,
        "post_tg3_freeze_receipt",
        candidate,
        POST_TG3_FREEZE_SCHEMA,
        &[POST_TG3_FROZEN_STATUS, POST_TG3_NOT_FROZEN_STATUS],
    )?;
    let freeze_document = &candidate.post_tg3_freeze_receipt.document;
    let freeze_tg3 = require_string_at(
        freeze_document,
        &["tg3_receipt_seal_sha256"],
        "post_tg3_freeze_receipt.tg3_receipt_seal_sha256",
    )?;
    let post_tg3_frozen = freeze_status == POST_TG3_FROZEN_STATUS
        && value_at(
            freeze_document,
            &["post_tg3_frozen"],
            "post_tg3_freeze_receipt.post_tg3_frozen",
        )?
        .as_bool()
            == Some(true)
        && value_at(
            freeze_document,
            &["candidate_specific_optimization_disabled"],
            "post_tg3_freeze_receipt.candidate_specific_optimization_disabled",
        )?
        .as_bool()
            == Some(true)
        && value_at(
            freeze_document,
            &["repair_requires_affected_scored_test_replay"],
            "post_tg3_freeze_receipt.repair_requires_affected_scored_test_replay",
        )?
        .as_bool()
            == Some(true)
        && freeze_tg3 == tg3_digest.document_seal_sha256
        && require_usize_at(
            freeze_document,
            &["post_tg3_mutation_count"],
            0,
            "post_tg3_freeze_receipt.post_tg3_mutation_count",
        )
        .is_ok();
    append_if_false(
        &mut blockers,
        post_tg3_frozen,
        format!(
            "{}:post_tg3_freeze_missing_or_mutable",
            candidate.candidate_artifact_id
        ),
    );

    let observed_gates = candidate
        .hard_gate_provenance
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected_gates = HARD_GATES.iter().copied().collect::<BTreeSet<_>>();
    if observed_gates != expected_gates {
        blockers.push(format!(
            "{}:hard_gate_provenance_does_not_exactly_cover_protected_protocol",
            candidate.candidate_artifact_id
        ));
    }
    let mut every_gate_passed = observed_gates == expected_gates;
    let mut replay_contracts_complete = observed_gates == expected_gates;
    for gate in HARD_GATES {
        let Some(provenance) = candidate.hard_gate_provenance.get(gate) else {
            every_gate_passed = false;
            replay_contracts_complete = false;
            continue;
        };
        let (_gate_digest, gate_status) = validate_candidate_document_identity(
            &provenance.gate_receipt,
            &format!("hard_gate_provenance.{gate}.gate_receipt"),
            candidate,
            HARD_GATE_PROVENANCE_SCHEMA,
            &[HARD_GATE_PASS_STATUS, HARD_GATE_PENDING_STATUS],
        )?;
        let gate_document = &provenance.gate_receipt.document;
        let receipt_gate = require_string_at(
            gate_document,
            &["gate"],
            &format!("hard_gate_provenance.{gate}.gate_receipt.gate"),
        )?;
        let receipt_qualification = require_string_at(
            gate_document,
            &["complete_manager_qualification_seal_sha256"],
            &format!("hard_gate_provenance.{gate}.qualification_binding"),
        )?;
        let receipt_tg3 = require_string_at(
            gate_document,
            &["tg3_receipt_seal_sha256"],
            &format!("hard_gate_provenance.{gate}.tg3_binding"),
        )?;
        let receipt_freeze = require_string_at(
            gate_document,
            &["post_tg3_freeze_seal_sha256"],
            &format!("hard_gate_provenance.{gate}.freeze_binding"),
        )?;
        let gate_passed = provenance.passed
            && gate_status == HARD_GATE_PASS_STATUS
            && value_at(
                gate_document,
                &["passed"],
                &format!("hard_gate_provenance.{gate}.gate_receipt.passed"),
            )?
            .as_bool()
                == Some(true)
            && value_at(
                gate_document,
                &["provenance_complete"],
                &format!("hard_gate_provenance.{gate}.gate_receipt.provenance_complete"),
            )?
            .as_bool()
                == Some(true)
            && receipt_gate == gate
            && receipt_qualification == qualification_digest.document_seal_sha256
            && receipt_tg3 == tg3_digest.document_seal_sha256
            && receipt_freeze == freeze_digest.document_seal_sha256;
        append_if_false(
            &mut blockers,
            gate_passed,
            format!(
                "{}:hard_gate_not_passed_or_unproven:{gate}",
                candidate.candidate_artifact_id
            ),
        );
        every_gate_passed &= gate_passed;

        let replay = &provenance.repair_replay;
        let mut replay_valid = replay.replay_required_for_repair;
        if replay.post_tg3_repair_occurred {
            let Some(receipt) = &replay.replay_receipt else {
                replay_valid = false;
                append_if_false(
                    &mut blockers,
                    false,
                    format!(
                        "{}:repair_replay_receipt_missing:{gate}",
                        candidate.candidate_artifact_id
                    ),
                );
                replay_contracts_complete &= replay_valid;
                continue;
            };
            let (replay_digest, replay_status) = validate_candidate_document_identity(
                receipt,
                &format!("hard_gate_provenance.{gate}.repair_replay.replay_receipt"),
                candidate,
                REPAIR_REPLAY_SCHEMA,
                &[REPAIR_REPLAY_COMPLETE_STATUS],
            )?;
            let replay_document = &receipt.document;
            replay_valid &= replay.affected_scored_tests_replayed
                && value_at(
                    replay_document,
                    &["affected_scored_tests_replayed"],
                    &format!("hard_gate_provenance.{gate}.repair_replay.receipt.replayed"),
                )?
                .as_bool()
                    == Some(true)
                && require_string_at(
                    replay_document,
                    &["gate"],
                    &format!("hard_gate_provenance.{gate}.repair_replay.receipt.gate"),
                )? == gate
                && require_string_at(
                    replay_document,
                    &["post_tg3_freeze_seal_sha256"],
                    &format!("hard_gate_provenance.{gate}.repair_replay.receipt.freeze_binding"),
                )? == freeze_digest.document_seal_sha256
                && replay_status == REPAIR_REPLAY_COMPLETE_STATUS
                && !replay_digest.document_seal_sha256.is_empty();
        } else if replay.affected_scored_tests_replayed || replay.replay_receipt.is_some() {
            replay_valid = false;
        }
        append_if_false(
            &mut blockers,
            replay_valid,
            format!(
                "{}:hard_gate_replay_contract_incomplete:{gate}",
                candidate.candidate_artifact_id
            ),
        );
        replay_contracts_complete &= replay_valid;
    }

    blockers.sort();
    blockers.dedup();
    Ok(CandidateAudit {
        candidate_artifact_id: candidate.candidate_artifact_id.clone(),
        complete_manager_qualified,
        tg3_qualified,
        post_tg3_frozen,
        every_hard_gate_passed_with_provenance: every_gate_passed,
        all_repair_replay_contracts_present: replay_contracts_complete,
        blockers,
    })
}

fn validate_mode_reservation(
    mode: &str,
    reservation: &ModeReservation,
    blockers: &mut Vec<String>,
) -> bool {
    let mut valid = reservation.reserved_for_both_candidates
        && !reservation.scored_execution_started
        && !reservation.candidate_results_recorded
        && reservation.same_frozen_task_corpus
        && reservation.identical_resource_envelope;
    if mode == "SOLO_MANAGER" {
        valid &= reservation.no_helper_advantage;
    } else {
        valid &= reservation.same_hawking_agent_os
            && reservation.symmetric_helper_model_infrastructure
            && reservation.no_hidden_information_or_worker_advantage;
    }
    append_if_false(
        blockers,
        valid,
        format!("final_comparison_reservation:{mode}:not_identical_or_not_inactive"),
    );
    valid
}

fn validate_final_comparison_reservation(
    reservation: &FinalComparisonReservation,
    blockers: &mut Vec<String>,
) -> bool {
    append_if_false(
        blockers,
        !reservation.tournament_activation_requested,
        "final_comparison_reservation:tournament_activation_requested",
    );
    append_if_false(
        blockers,
        !reservation.scored_task_execution_requested,
        "final_comparison_reservation:scored_task_execution_requested",
    );
    append_if_false(
        blockers,
        !reservation.winner_selection_requested,
        "final_comparison_reservation:winner_selection_requested",
    );
    let observed = reservation
        .evaluation_modes
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = EVALUATION_MODES.iter().copied().collect::<BTreeSet<_>>();
    let mut valid = observed == expected;
    append_if_false(
        blockers,
        valid,
        "final_comparison_reservation:evaluation_modes_do_not_exactly_cover_protocol",
    );
    for mode in EVALUATION_MODES {
        let Some(mode_reservation) = reservation.evaluation_modes.get(mode) else {
            valid = false;
            continue;
        };
        valid &= validate_mode_reservation(mode, mode_reservation, blockers);
    }
    let corpus = &reservation.protected_task_corpus;
    let corpus_valid = corpus.protected_corpus_sealed
        && corpus.blind_tasks_frozen
        && corpus.hidden_membership_inaccessible_to_candidates
        && corpus.real_hawking_work_only
        && !corpus.tasks_executed
        && corpus.required_families
            == TASK_FAMILIES
                .iter()
                .map(|item| (*item).to_owned())
                .collect::<Vec<_>>();
    append_if_false(
        blockers,
        corpus_valid,
        "final_comparison_reservation:protected_blind_task_corpus_incomplete",
    );
    valid &= corpus_valid;
    let review = &reservation.adversarial_review;
    let review_valid = review.opposing_candidate_read_only_red_team
        && review.reviewer_may_not_modify_candidate_artifact
        && review.protected_verifier_adjudicates_challenges
        && review.no_candidate_self_scoring
        && !review.review_executed;
    append_if_false(
        blockers,
        review_valid,
        "final_comparison_reservation:adversarial_review_or_protected_verifier_incomplete",
    );
    valid &= review_valid;
    let fairness_keys = reservation
        .fairness
        .equalized
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected_fairness = FAIRNESS_ENVELOPE.iter().copied().collect::<BTreeSet<_>>();
    let fairness_valid = reservation.fairness.asymmetries_recorded
        && fairness_keys == expected_fairness
        && reservation.fairness.equalized.values().all(|value| *value);
    append_if_false(
        blockers,
        fairness_valid,
        "final_comparison_reservation:fairness_envelope_not_exactly_equalized",
    );
    valid &= fairness_valid;
    let selection = &reservation.protected_selection;
    let selection_valid = selection.pareto_frontier_required_before_selection
        && selection.do_not_collapse_to_scalar_before_complete_evidence_matrix
        && selection.candidates_cannot_self_grade
        && selection.candidates_cannot_change_weights_or_hidden_tests
        && selection.candidates_cannot_promote_self_or_invalidate_opponent
        && selection.only_protected_controller_or_human_may_select_manager
        && !selection.winner_selected;
    append_if_false(
        blockers,
        selection_valid,
        "final_comparison_reservation:protected_pareto_selection_incomplete_or_preselected",
    );
    valid &= selection_valid;
    let alternate = &reservation.winner_freeze_and_loser_restore;
    let alternate_valid = alternate.winner_freeze_contract_reserved
        && alternate.loser_cold_store_contract_reserved
        && alternate.loser_restore_hash_verify_one_command_required
        && alternate.no_delete_loser_before_restore_proof
        && !alternate.alternate_evicted;
    append_if_false(
        blockers,
        alternate_valid,
        "final_comparison_reservation:winner_freeze_loser_cold_store_restore_contract_incomplete",
    );
    valid &= alternate_valid;
    valid
}

fn exact_candidates<'a>(
    candidates: &'a BTreeMap<String, CandidateQualification>,
) -> Result<(&'a CandidateQualification, &'a CandidateQualification), String> {
    let keys = candidates
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let expected = CANDIDATE_ARTIFACTS.iter().copied().collect::<BTreeSet<_>>();
    if keys != expected {
        return Err(
            "candidate_qualifications must exactly cover the two protected artifacts".into(),
        );
    }
    let qwen30 = candidates
        .get(CANDIDATE_ARTIFACTS[0])
        .ok_or("candidate_qualifications lacks Qwen30 artifact")?;
    let qwen80 = candidates
        .get(CANDIDATE_ARTIFACTS[1])
        .ok_or("candidate_qualifications lacks Qwen80 artifact")?;
    if qwen30.candidate_artifact_id != CANDIDATE_ARTIFACTS[0]
        || qwen80.candidate_artifact_id != CANDIDATE_ARTIFACTS[1]
    {
        return Err(
            "candidate qualification artifact identifiers do not match their fixed keys".into(),
        );
    }
    Ok((qwen30, qwen80))
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
    let protocol_digest = validate_final_manager_protocol(&input.final_manager_protocol)?;
    let (lane_digest, _lane_tg10) = validate_lane_authority(&input.lane_authority)?;
    let mutation_digest = validate_mutation_authority(&input.mutation_authority, &lane_digest)?;
    let knowledge_digest =
        validate_knowledge_authority(&input.knowledge_authority, &lane_digest, &mutation_digest)?;
    let scheduler_digest = validate_scheduler_authority(
        &input.scheduler_authority,
        &lane_digest,
        &mutation_digest,
        &knowledge_digest,
    )?;
    let tg10_digest = validate_tg10_activation_authority(
        &input.tg10_activation_authority,
        &lane_digest,
        &mutation_digest,
        &knowledge_digest,
        &scheduler_digest,
    )?;
    let (qwen30, qwen80) = exact_candidates(&input.candidate_qualifications)?;
    let qwen30_audit = audit_candidate(qwen30)?;
    let qwen80_audit = audit_candidate(qwen80)?;
    let mut blockers = qwen30_audit
        .blockers
        .iter()
        .chain(qwen80_audit.blockers.iter())
        .cloned()
        .collect::<Vec<_>>();
    let reservation_complete =
        validate_final_comparison_reservation(&input.final_comparison_reservation, &mut blockers);
    blockers.sort();
    blockers.dedup();
    let ready = qwen30_audit.complete_manager_qualified
        && qwen30_audit.tg3_qualified
        && qwen30_audit.post_tg3_frozen
        && qwen30_audit.every_hard_gate_passed_with_provenance
        && qwen30_audit.all_repair_replay_contracts_present
        && qwen80_audit.complete_manager_qualified
        && qwen80_audit.tg3_qualified
        && qwen80_audit.post_tg3_frozen
        && qwen80_audit.every_hard_gate_passed_with_provenance
        && qwen80_audit.all_repair_replay_contracts_present
        && reservation_complete
        && blockers.is_empty();
    let status = if ready {
        PREPARED_STATUS
    } else {
        REFUSED_STATUS
    }
    .to_owned();
    let focused_checks = FocusedChecks {
        fixed_final_manager_protocol_identity_and_full_contract_bound: true,
        sealed_lane_mutation_knowledge_scheduler_and_tg10_authority_chain_bound: true,
        qwen30_and_qwen80_complete_manager_tg3_freeze_and_hard_gates_required: true,
        both_identical_evaluation_modes_reserved_but_not_executed: reservation_complete,
        blind_protected_corpus_red_team_verifier_and_no_self_scoring_reserved: reservation_complete,
        pareto_before_selection_and_winner_loser_restore_contract_reserved: reservation_complete,
        no_runtime_server_gpu_watcher_tps_or_tournament_authority: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        prepared: ready,
        tournament_active: false,
        scored_task_execution_active: false,
        winner_selected: false,
        bound_final_manager_protocol: protocol_digest,
        bound_lane_authority: lane_digest,
        bound_mutation_authority: mutation_digest,
        bound_knowledge_authority: knowledge_digest,
        bound_scheduler_authority: scheduler_digest,
        bound_tg10_activation_authority: tg10_digest,
        qwen30_audit,
        qwen80_audit,
        final_comparison_reservation_complete: reservation_complete,
        blockers,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only TG3/freeze final-comparison preparation authority, not a tournament runner.",
            "It accepts only the fixed final-manager protocol and exact sealed paired lane, mutation, Knowledge Plane, scheduler, and both-TG10 activation authority chain.",
            "Both fixed manager artifacts need a complete-manager qualification, an exact TG3 BASE_TRUE_TPS operational receipt, a post-TG3 freeze, and every conjunctive hard-gate provenance/replay contract.",
            "SOLO_MANAGER and MANAGER_AS_ORCHESTRATOR are reserved under the same frozen corpus and resource envelope; neither mode is measured here.",
            "The protected blind corpus, read-only opposing red team, protected verifier, no-self-scoring rule, Pareto requirement, and winner/loser cold-store restore contract remain mandatory.",
            "A prepared result does not activate a tournament, execute tasks, score candidates, select a winner, start a server/watcher, use a GPU, run HCLI, or measure TPS/TG.",
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
            format!("TG3/freeze comparison report cannot be serialized: {error}")
        })?)
        .map_err(|error| format!("TG3/freeze comparison report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_paired_cognition_tg3_freeze_final_comparison_authority --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        .map_err(|error| format!("TG3/freeze comparison validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_paired_cognition_tg3_freeze_final_comparison_authority: {error}");
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
        let mut rows = Map::new();
        for value in values {
            rows.insert((*value).into(), json!({"required": true}));
        }
        Value::Object(rows)
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
            "candidates": {
                "fixed_order": CANDIDATE_ARTIFACTS,
                "both_must_meet_complete_manager_qualification_floor_before_freeze": true,
            },
            "freeze_contract": {
                "both_candidates_frozen_before_scored_execution": true,
                "candidate_specific_optimization_disabled_during_scored_execution": true,
                "hard_gate_repair_requires_affected_test_replay": true,
            },
            "evaluation_modes": {
                "required": EVALUATION_MODES,
                "SOLO_MANAGER": {
                    "measures": "raw_model_manager_capability",
                    "no_helper_advantage": true,
                    "each_candidate_measured": true,
                },
                "MANAGER_AS_ORCHESTRATOR": {
                    "measures": "organization_level_verified_engineering_progress",
                    "same_hawking_agent_os": true,
                    "symmetric_helper_model_infrastructure": true,
                    "no_hidden_information_or_worker_advantage": true,
                    "each_candidate_measured": true,
                },
            },
            "hard_gates": {
                "conjunctive": true,
                "no_score_can_compensate_for_failure": true,
                "each_gate_required_for_each_candidate": true,
                "required": required_rows(&HARD_GATES),
            },
            "protected_task_corpus": {
                "real_hawking_work_only": true,
                "blind_tasks_required": true,
                "hidden_membership_frozen_before_scored_execution": true,
                "required_families": TASK_FAMILIES,
            },
            "long_horizon_campaigns": {
                "required": LONG_HORIZON_CAMPAIGNS,
                "measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion": true,
            },
            "adversarial_review": {
                "required_after_each_candidate_task": true,
                "opposing_candidate_is_read_only_red_team_reviewer": true,
                "reviewer_may_not_modify_candidate_artifact": true,
                "protected_verifier_adjudicates_challenges": true,
                "measure_genuine_defects_and_false_objections": true,
            },
            "fairness_envelope": {
                "record_every_asymmetry": true,
                "equalize": required_rows(&FAIRNESS_ENVELOPE),
            },
            "scorecards": {
                "primary_metric": "verified_tasks_per_hour",
                "primary_metrics": PRIMARY_METRICS,
                "performance_metrics": PERFORMANCE_METRICS,
                "manager_intelligence_metrics": MANAGER_INTELLIGENCE_METRICS,
                "orchestration_metrics": ORCHESTRATION_METRICS,
                "failure_recovery_injections": FAILURE_RECOVERY_INJECTIONS,
                "pareto_frontier_required_before_selection": true,
                "do_not_collapse_to_scalar_before_complete_evidence_matrix": true,
            },
            "protected_selection": {
                "candidate_self_assessments_are_evidence_only": true,
                "candidates_cannot_self_grade": true,
                "candidates_cannot_change_weights_or_hidden_tests": true,
                "candidates_cannot_promote_self_or_invalidate_opponent": true,
                "only_protected_controller_or_human_may_select_manager": true,
            },
            "winner_freeze_and_alternate": {
                "seal_winner_source_artifact_runtime_kernel_agent_os_context_kv_benchmarks_capability_tournament_and_rollback": true,
                "seal_loser_before_any_evictability": true,
                "cold_store_hash_verify_restore_test_one_command_recovery_required": true,
                "do_not_delete_loser_before_restore_proof": true,
            },
            "final_report": {
                "side_by_side_candidates_required": true,
                "required_fields": FINAL_REPORT_FIELDS_EXACT,
                "must_state_winner_loser_decisive_evidence_tradeoffs_restore_path": true,
            },
            "claim_boundary": {
                "does_not_execute_tournament": true,
                "does_not_score_candidates": true,
                "does_not_choose_winner": true,
                "does_not_activate_a_server_or_sandbox": true,
                "does_not_relax_tg3_or_other_hard_gates": true,
            },
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
            "/sealed/lifecycle/FINAL_MANAGER_TOURNAMENT_PROTOCOL.json",
            seal_value(Value::Object(document)).unwrap(),
        )
    }

    fn tg10(pass: bool) -> Tg10Authority {
        Tg10Authority {
            required_base_true_tps: 100,
            operational_pass: pass,
            coherent_hcli_pass: pass,
            complete_token_path_measured: pass,
            fallback_count: 0,
            median_base_true_tps: pass.then_some(100.0),
            receipt_seal_sha256: pass.then(|| sha('a')),
        }
    }

    fn authority_bindings() -> (
        SealedDocumentBinding,
        SealedDocumentBinding,
        SealedDocumentBinding,
        SealedDocumentBinding,
        SealedDocumentBinding,
    ) {
        let lane = binding(
            "/sealed/paired/lane.json",
            seal_value(json!({
                "schema": LANE_AUTHORITY_SCHEMA,
                "status": LANE_AUTHORITY_STATUS,
                "prepared": true,
                "paired_candidate_worlds_active": false,
                "no_new_physical_model_process_authority": true,
                "model_contract_bindings": [
                    {"model_key": QWEN30, "tg10": tg10(true)},
                    {"model_key": QWEN80, "tg10": tg10(true)},
                ],
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        );
        let mutation = binding(
            "/sealed/paired/mutation.json",
            seal_value(json!({
                "schema": MUTATION_AUTHORITY_SCHEMA,
                "status": MUTATION_AUTHORITY_STATUS,
                "prepared": true,
                "paired_candidate_worlds_active": false,
                "bound_lane_authority": digest_from(&lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        );
        let knowledge = binding(
            "/sealed/paired/knowledge.json",
            seal_value(json!({
                "schema": KNOWLEDGE_AUTHORITY_SCHEMA,
                "status": KNOWLEDGE_AUTHORITY_STATUS,
                "prepared": true,
                "knowledge_plane_active": false,
                "external_publication_performed": false,
                "bound_lane_authority": digest_from(&lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest_from(&mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        );
        let scheduler = binding(
            "/sealed/paired/scheduler.json",
            seal_value(json!({
                "schema": SCHEDULER_AUTHORITY_SCHEMA,
                "status": SCHEDULER_AUTHORITY_STATUS,
                "prepared": true,
                "paired_development_active": false,
                "logical_sessions_created": false,
                "bound_lane_authority": digest_from(&lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest_from(&mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "bound_knowledge_authority": digest_from(&knowledge, KNOWLEDGE_AUTHORITY_SCHEMA, KNOWLEDGE_AUTHORITY_STATUS),
                "final_mode_gate": {
                    "solo_manager_evaluation_authorized_by_this_contract": false,
                    "symmetric_orchestrator_evaluation_authorized_by_this_contract": false,
                    "winner_selection_authorized_by_this_contract": false,
                },
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        );
        let tg10_activation = binding(
            "/sealed/paired/tg10-activation.json",
            seal_value(json!({
                "schema": TG10_ACTIVATION_SCHEMA,
                "status": TG10_ACTIVATION_STATUS,
                "prepared": true,
                "paired_development_active": false,
                "paired_development_activation_authorized_by_this_contract": false,
                "bound_lane_authority": digest_from(&lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest_from(&mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "bound_knowledge_authority": digest_from(&knowledge, KNOWLEDGE_AUTHORITY_SCHEMA, KNOWLEDGE_AUTHORITY_STATUS),
                "bound_scheduler_authority": digest_from(&scheduler, SCHEDULER_AUTHORITY_SCHEMA, SCHEDULER_AUTHORITY_STATUS),
                "both_exact_fresh_tg10_operational_receipts_present": true,
                "state_blockers": [],
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        );
        (lane, mutation, knowledge, scheduler, tg10_activation)
    }

    fn evidence_binding(
        path: String,
        schema: &str,
        status: &str,
        candidate: &str,
        source: &str,
        admission: &str,
        extra: Value,
    ) -> SealedDocumentBinding {
        let mut document = Map::new();
        document.insert("schema".into(), Value::String(schema.into()));
        document.insert("status".into(), Value::String(status.into()));
        document.insert(
            "candidate_artifact_id".into(),
            Value::String(candidate.into()),
        );
        document.insert(
            "source_identity_seal_sha256".into(),
            Value::String(source.into()),
        );
        document.insert(
            "complete_artifact_admission_seal_sha256".into(),
            Value::String(admission.into()),
        );
        for (key, value) in extra.as_object().unwrap() {
            document.insert(key.clone(), value.clone());
        }
        binding(&path, seal_value(Value::Object(document)).unwrap())
    }

    fn candidate_qualification(candidate: &str, qualified: bool) -> CandidateQualification {
        let source = if candidate == CANDIDATE_ARTIFACTS[0] {
            sha('1')
        } else {
            sha('2')
        };
        let admission = if candidate == CANDIDATE_ARTIFACTS[0] {
            sha('3')
        } else {
            sha('4')
        };
        let qualification = evidence_binding(
            format!("/sealed/{candidate}/complete-manager.json"),
            COMPLETE_MANAGER_QUALIFICATION_SCHEMA,
            if qualified {
                COMPLETE_MANAGER_QUALIFIED_STATUS
            } else {
                COMPLETE_MANAGER_PENDING_STATUS
            },
            candidate,
            &source,
            &admission,
            json!({"complete_manager_qualified": qualified}),
        );
        let tg3 = evidence_binding(
            format!("/sealed/{candidate}/tg3.json"),
            TG3_RECEIPT_SCHEMA,
            if qualified {
                TG3_QUALIFIED_STATUS
            } else {
                TG3_PENDING_STATUS
            },
            candidate,
            &source,
            &admission,
            json!({
                "tg3_qualified": qualified,
                "tg_level": 3,
                "required_base_true_tps": 333,
                "median_base_true_tps": if qualified { 333.0 } else { 0.0 },
                "base_true_tps_pass": qualified,
                "fallback_count": 0,
                "real_metal": qualified,
                "coherent_real_hcli": qualified,
                "complete_token_path_measured": qualified,
            }),
        );
        let freeze = evidence_binding(
            format!("/sealed/{candidate}/post-tg3-freeze.json"),
            POST_TG3_FREEZE_SCHEMA,
            if qualified {
                POST_TG3_FROZEN_STATUS
            } else {
                POST_TG3_NOT_FROZEN_STATUS
            },
            candidate,
            &source,
            &admission,
            json!({
                "post_tg3_frozen": qualified,
                "candidate_specific_optimization_disabled": qualified,
                "repair_requires_affected_scored_test_replay": true,
                "post_tg3_mutation_count": 0,
                "tg3_receipt_seal_sha256": tg3.document["seal_sha256"],
            }),
        );
        let mut hard_gate_provenance = BTreeMap::new();
        for gate in HARD_GATES {
            let receipt = evidence_binding(
                format!("/sealed/{candidate}/gates/{gate}.json"),
                HARD_GATE_PROVENANCE_SCHEMA,
                if qualified {
                    HARD_GATE_PASS_STATUS
                } else {
                    HARD_GATE_PENDING_STATUS
                },
                candidate,
                &source,
                &admission,
                json!({
                    "gate": gate,
                    "passed": qualified,
                    "provenance_complete": qualified,
                    "complete_manager_qualification_seal_sha256": qualification.document["seal_sha256"],
                    "tg3_receipt_seal_sha256": tg3.document["seal_sha256"],
                    "post_tg3_freeze_seal_sha256": freeze.document["seal_sha256"],
                }),
            );
            hard_gate_provenance.insert(
                gate.into(),
                HardGateProvenance {
                    passed: qualified,
                    gate_receipt: receipt,
                    repair_replay: RepairReplayRequirement {
                        post_tg3_repair_occurred: false,
                        replay_required_for_repair: true,
                        affected_scored_tests_replayed: false,
                        replay_receipt: None,
                    },
                },
            );
        }
        CandidateQualification {
            candidate_artifact_id: candidate.into(),
            source_identity_seal_sha256: source,
            complete_artifact_admission_seal_sha256: admission,
            complete_manager_qualification_receipt: qualification,
            tg3_operational_receipt: tg3,
            post_tg3_freeze_receipt: freeze,
            hard_gate_provenance,
        }
    }

    fn reservation() -> FinalComparisonReservation {
        let mut modes = BTreeMap::new();
        modes.insert(
            "SOLO_MANAGER".into(),
            ModeReservation {
                reserved_for_both_candidates: true,
                scored_execution_started: false,
                candidate_results_recorded: false,
                same_frozen_task_corpus: true,
                identical_resource_envelope: true,
                no_helper_advantage: true,
                same_hawking_agent_os: false,
                symmetric_helper_model_infrastructure: false,
                no_hidden_information_or_worker_advantage: false,
            },
        );
        modes.insert(
            "MANAGER_AS_ORCHESTRATOR".into(),
            ModeReservation {
                reserved_for_both_candidates: true,
                scored_execution_started: false,
                candidate_results_recorded: false,
                same_frozen_task_corpus: true,
                identical_resource_envelope: true,
                no_helper_advantage: false,
                same_hawking_agent_os: true,
                symmetric_helper_model_infrastructure: true,
                no_hidden_information_or_worker_advantage: true,
            },
        );
        FinalComparisonReservation {
            tournament_activation_requested: false,
            scored_task_execution_requested: false,
            winner_selection_requested: false,
            evaluation_modes: modes,
            protected_task_corpus: ProtectedTaskCorpusReservation {
                protected_corpus_sealed: true,
                blind_tasks_frozen: true,
                hidden_membership_inaccessible_to_candidates: true,
                real_hawking_work_only: true,
                required_families: TASK_FAMILIES
                    .iter()
                    .map(|item| (*item).to_owned())
                    .collect(),
                tasks_executed: false,
            },
            adversarial_review: AdversarialReviewReservation {
                opposing_candidate_read_only_red_team: true,
                reviewer_may_not_modify_candidate_artifact: true,
                protected_verifier_adjudicates_challenges: true,
                no_candidate_self_scoring: true,
                review_executed: false,
            },
            fairness: FairnessReservation {
                asymmetries_recorded: true,
                equalized: FAIRNESS_ENVELOPE
                    .iter()
                    .map(|key| ((*key).to_owned(), true))
                    .collect(),
            },
            protected_selection: ProtectedSelectionReservation {
                pareto_frontier_required_before_selection: true,
                do_not_collapse_to_scalar_before_complete_evidence_matrix: true,
                candidates_cannot_self_grade: true,
                candidates_cannot_change_weights_or_hidden_tests: true,
                candidates_cannot_promote_self_or_invalidate_opponent: true,
                only_protected_controller_or_human_may_select_manager: true,
                winner_selected: false,
            },
            winner_freeze_and_loser_restore: WinnerFreezeAndLoserRestoreReservation {
                winner_freeze_contract_reserved: true,
                loser_cold_store_contract_reserved: true,
                loser_restore_hash_verify_one_command_required: true,
                no_delete_loser_before_restore_proof: true,
                alternate_evicted: false,
            },
        }
    }

    fn input_fixture(qualified: bool) -> Input {
        let (lane, mutation, knowledge, scheduler, tg10_activation) = authority_bindings();
        let mut candidate_qualifications = BTreeMap::new();
        for candidate in CANDIDATE_ARTIFACTS {
            candidate_qualifications.insert(
                candidate.into(),
                candidate_qualification(candidate, qualified),
            );
        }
        Input {
            schema: INPUT_SCHEMA.into(),
            final_manager_protocol: protocol_binding(),
            lane_authority: lane,
            mutation_authority: mutation,
            knowledge_authority: knowledge,
            scheduler_authority: scheduler,
            tg10_activation_authority: tg10_activation,
            candidate_qualifications,
            final_comparison_reservation: reservation(),
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
    fn current_unqualified_models_refuse_without_activation() {
        let report = build_report(input_fixture(false)).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(!report.tournament_active);
        assert!(!report.scored_task_execution_active);
        assert!(!report.winner_selected);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("tg3_operational_qualification_missing")));
    }

    #[test]
    fn both_tg3_frozen_complete_artifacts_can_only_prepare_not_activate() {
        let report = build_report(input_fixture(true)).unwrap();
        assert_eq!(report.status, PREPARED_STATUS);
        assert!(report.prepared);
        assert!(report.qwen30_audit.tg3_qualified);
        assert!(report.qwen80_audit.post_tg3_frozen);
        assert!(report.final_comparison_reservation_complete);
        assert!(!report.tournament_active);
        assert!(!report.scored_task_execution_active);
        assert!(!report.winner_selected);
    }

    #[test]
    fn a_single_failed_hard_gate_refuses_even_after_tg3() {
        let mut input = input_fixture(true);
        let candidate = input
            .candidate_qualifications
            .get_mut(CANDIDATE_ARTIFACTS[1])
            .unwrap();
        let gate = candidate
            .hard_gate_provenance
            .get_mut("real_metal")
            .unwrap();
        gate.passed = false;
        gate.gate_receipt.document["status"] = Value::String(HARD_GATE_PENDING_STATUS.into());
        gate.gate_receipt.document["passed"] = Value::Bool(false);
        let document = gate.gate_receipt.document.clone();
        gate.gate_receipt.document = seal_value(document).unwrap();
        gate.gate_receipt.document_sha256 = sha256_json(&gate.gate_receipt.document).unwrap();
        let report = build_report(input).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(report
            .qwen80_audit
            .blockers
            .iter()
            .any(|blocker| blocker.ends_with("hard_gate_not_passed_or_unproven:real_metal")));
    }

    #[test]
    fn reservation_refuses_any_activation_scoring_or_winner_request() {
        let mut input = input_fixture(true);
        input
            .final_comparison_reservation
            .tournament_activation_requested = true;
        input
            .final_comparison_reservation
            .scored_task_execution_requested = true;
        input
            .final_comparison_reservation
            .winner_selection_requested = true;
        let report = build_report(input).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(report
            .blockers
            .iter()
            .any(|blocker| blocker.contains("tournament_activation_requested")));
    }

    #[test]
    fn protocol_identity_and_authority_chain_are_fail_closed() {
        let mut wrong_protocol = input_fixture(true);
        wrong_protocol.final_manager_protocol.document["protocol_identity_sha256"] =
            Value::String(sha('0'));
        let document = wrong_protocol.final_manager_protocol.document.clone();
        wrong_protocol.final_manager_protocol.document = seal_value(document).unwrap();
        wrong_protocol.final_manager_protocol.document_sha256 =
            sha256_json(&wrong_protocol.final_manager_protocol.document).unwrap();
        assert!(build_report(wrong_protocol).is_err());

        let mut wrong_chain = input_fixture(true);
        wrong_chain.tg10_activation_authority.document["bound_scheduler_authority"]
            ["document_sha256"] = Value::String(sha('0'));
        let document = wrong_chain.tg10_activation_authority.document.clone();
        wrong_chain.tg10_activation_authority.document = seal_value(document).unwrap();
        wrong_chain.tg10_activation_authority.document_sha256 =
            sha256_json(&wrong_chain.tg10_activation_authority.document).unwrap();
        assert!(build_report(wrong_chain).is_err());
    }

    #[test]
    fn sealed_io_and_create_new_are_enforced_without_runtime_activity() {
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
        assert_eq!(output["tournament_active"], Value::Bool(false));
        assert!(write_report_create_new(
            &output_path,
            &build_report(input_fixture(false)).unwrap()
        )
        .is_err());
    }
}
