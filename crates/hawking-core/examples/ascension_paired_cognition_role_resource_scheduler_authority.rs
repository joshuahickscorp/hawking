//! Sealed paired-cognition one-body / many-logical-session role scheduler.
//!
//! This is a CPU-only control-plane authority for the fixed paired topology:
//! one Q80 body and one Q30 body, with two isolated logical roles scheduled
//! through each body.  It consumes the sealed lane namespace authority, the
//! sealed primary-only mutation authority, and the sealed generic-only
//! Knowledge Plane authority.  It never starts either body and does not create
//! a logical session; it only verifies the namespace, queue, resource, and
//! fairness reservation a later protected controller must obey.
//!
//! ```text
//! Q80 resident body: Q80-primary in Q80 lane, Q80-helper in Q30 lane
//! Q30 resident body: Q30-primary in Q30 lane, Q30-helper in Q80 lane
//! ```
//!
//! A successful report remains `PREPARED/NOT_ACTIVE`: paired development is
//! merely reserved until both sealed TG10 >=100 BASE_TRUE_TPS operational
//! receipts exist.  Solo-manager and manager-as-orchestrator final modes are
//! likewise reserved until both TG3 receipts and a post-TG3 freeze exist.
//! This contract has no winner-selection authority.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.paired_cognition_role_resource_scheduler_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_one_body_many_logical_session_role_resource_scheduler_authority.v1";
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
const STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_ONE_Q30_ONE_Q80_FOUR_ISOLATED_LOGICAL_ROLES_NO_RUNTIME_OR_WINNER_SELECTION";

const QWEN30: &str = "qwen30";
const QWEN80: &str = "qwen80";
const QWEN30_HOST: &str = "127.0.0.1";
const QWEN80_HOST: &str = "127.0.0.1";
const QWEN30_PORT: u16 = 18_430;
const QWEN80_PORT: u16 = 18_480;
const TG10_BASE_TRUE_TPS: u16 = 100;
const ROUND_ROBIN_EQUAL_WEIGHT: &str = "round_robin_equal_weight";
const TELEMETRY_FIELDS: [&str; 5] = [
    "eligible_sessions",
    "queued_request_count",
    "scheduled_turn_count",
    "oldest_wait_dispatches",
    "consecutive_dispatch_count",
];
const WORK_ACTIONS: [&str; 4] = [
    "submit_proposal",
    "protected_review",
    "falsify",
    "test_in_isolated_worktree",
];
const PRIMARY_ACTIONS: [&str; 7] = [
    "submit_proposal",
    "protected_review",
    "falsify",
    "test_in_isolated_worktree",
    "accept_verified_proposal_for_own_champion",
    "mutate_own_champion",
    "promote_own_champion",
];

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct SealedDocumentBinding {
    path: String,
    document_sha256: String,
    document: Value,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct Endpoint {
    host: String,
    port: u16,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct TopologyAssertion {
    resident_model_processes: usize,
    immutable_weight_copies: usize,
    logical_session_policy: String,
    endpoint: Endpoint,
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct ModelContractBindingReport {
    model_key: String,
    topology: TopologyAssertion,
    tg10: Tg10Authority,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct PrivateNamespaces {
    mission: String,
    experiments: String,
    receipts: String,
    worktree: String,
    sessions: String,
    frontier: String,
    patches: String,
    scores: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct MissionAuthority {
    primary_candidate_mutation_authority: bool,
    primary_candidate_promotion_authority: bool,
    helper_may_inspect_and_critique: bool,
    helper_may_propose_or_test_in_private_worktree: bool,
    helper_may_mutate_primary_champion: bool,
    helper_may_promote_primary_champion: bool,
    opposite_lane_may_mutate_primary_champion: bool,
    primary_or_helper_may_self_score: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct BoundLane {
    lane_id: String,
    primary_model_key: String,
    helper_model_key: String,
    private_namespaces: PrivateNamespaces,
    mission_authority: MissionAuthority,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct CrossLaneReadPolicy {
    allow_cross_lane_mission_reads: bool,
    allow_cross_lane_experiment_reads: bool,
    allow_cross_lane_receipt_reads: bool,
    allow_cross_lane_worktree_reads: bool,
    allow_cross_lane_session_reads: bool,
    allow_cross_lane_frontier_reads: bool,
    allow_cross_lane_patch_reads: bool,
    allow_cross_lane_score_reads: bool,
    verified_generic_knowledge_plane_publication_only: bool,
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
struct LaneAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    no_new_physical_model_process_authority: bool,
    model_contract_bindings: Vec<ModelContractBindingReport>,
    lane_worlds: Vec<BoundLane>,
    cross_lane_read_policy: CrossLaneReadPolicy,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
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
struct MutationLaneActionPolicy {
    lane_id: String,
    primary_model_key: String,
    helper_model_key: String,
    opponent_model_key: String,
    primary_actor_id: String,
    helper_actor_id: String,
    opponent_actor_id: String,
    primary_allowed_actions: Vec<String>,
    helper_allowed_actions: Vec<String>,
    opponent_allowed_actions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct ProtectedRecordPolicy {
    evidence_receipts_immutable: bool,
    mission_records_immutable: bool,
    tournament_receipts_immutable: bool,
    all_roles_may_rewrite_evidence_receipts: bool,
    all_roles_may_rewrite_mission_records: bool,
    all_roles_may_rewrite_tournament_receipts: bool,
    all_roles_may_delete_protected_records: bool,
    cross_lane_private_record_release_allowed: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct MutationAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    bound_lane_authority: EvidenceDigest,
    lane_action_policies: Vec<MutationLaneActionPolicy>,
    protected_record_policy: ProtectedRecordPolicy,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct KnowledgePlanePolicy {
    knowledge_plane_namespace: String,
    release_registry_namespace: String,
    generic_mechanism_science_only: bool,
    append_only_release_identities: bool,
    release_identity_mutation_authorized: bool,
    external_publication_authorized: bool,
    lane_private_record_access_authorized: bool,
    candidate_world_activation_authorized: bool,
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
    knowledge_plane_policy: KnowledgePlanePolicy,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct BodyResourceBudget {
    model_key: String,
    physical_body_id: String,
    endpoint: Endpoint,
    resident_model_processes: usize,
    immutable_weight_copies: usize,
    max_logical_sessions: u8,
    max_inflight_turns: u8,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct RoleSessionAssignment {
    lane_id: String,
    role: String,
    logical_actor_id: String,
    physical_model_key: String,
    physical_body_id: String,
    session_namespace: String,
    queue_namespace: String,
    scheduling_weight: u8,
    max_queued_requests: u8,
    max_inflight_turns: u8,
    private_lane_visibility_only: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct QueueFairnessTelemetryPolicy {
    physical_model_key: String,
    physical_body_id: String,
    telemetry_namespace: String,
    scheduling_discipline: String,
    max_inflight_turns_per_body: u8,
    max_consecutive_dispatches_to_same_session: u8,
    fairness_lag_bound_dispatches: u8,
    required_counter_fields: Vec<String>,
    telemetry_contains_request_content: bool,
    telemetry_contains_generated_tokens: bool,
    telemetry_contains_lane_private_scores: bool,
    telemetry_contains_private_namespace_identifiers: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct Tg3Receipt {
    model_key: String,
    tg_level: u8,
    operational_pass: bool,
    receipt_seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct FinalModeReservation {
    qwen30_tg3_receipt: Option<Tg3Receipt>,
    qwen80_tg3_receipt: Option<Tg3Receipt>,
    post_tg3_freeze_seal_sha256: Option<String>,
    solo_manager_evaluation_requested: bool,
    symmetric_orchestrator_evaluation_requested: bool,
    winner_selection_requested: bool,
    final_mode_authorized_by_this_contract: bool,
    winner_selection_authorized_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    lane_authority: SealedDocumentBinding,
    mutation_authority: SealedDocumentBinding,
    knowledge_authority: SealedDocumentBinding,
    body_resource_budgets: Vec<BodyResourceBudget>,
    role_session_assignments: Vec<RoleSessionAssignment>,
    queue_fairness_telemetry_policies: Vec<QueueFairnessTelemetryPolicy>,
    paired_development_activation_requested: bool,
    final_mode_reservation: FinalModeReservation,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct Tg10Readiness {
    model_key: String,
    required_base_true_tps: u16,
    operational_pass: bool,
    receipt_seal_sha256: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
struct PairedDevelopmentReservation {
    required_base_true_tps: u16,
    qwen30_tg10: Tg10Readiness,
    qwen80_tg10: Tg10Readiness,
    both_tg10_operational_receipts_present: bool,
    paired_development_ready_after_both_tg10: bool,
    paired_development_activation_authorized_by_this_contract: bool,
}

#[derive(Clone, Debug, Serialize)]
struct FinalModeGateReport {
    both_tg3_operational_receipts_present: bool,
    post_tg3_freeze_present: bool,
    final_modes_ready_after_both_tg3_and_freeze: bool,
    solo_manager_evaluation_authorized_by_this_contract: bool,
    symmetric_orchestrator_evaluation_authorized_by_this_contract: bool,
    winner_selection_authorized_by_this_contract: bool,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    exact_sealed_lane_mutation_and_knowledge_authorities_bound: bool,
    one_q30_body_and_one_q80_body_only: bool,
    one_weight_copy_and_one_endpoint_per_body: bool,
    four_expected_primary_helper_roles_and_no_extra_logical_sessions: bool,
    all_session_and_queue_namespaces_are_lane_private_and_distinct: bool,
    role_body_bindings_are_heterogeneous_and_nonduplicating: bool,
    bounded_queues_and_equal_fairness_telemetry_are_private_and_content_free: bool,
    cross_lane_private_reads_and_self_scoring_denied: bool,
    paired_development_reserved_until_two_valid_tg10_receipts: bool,
    solo_and_orchestrator_final_modes_reserved_until_two_tg3_receipts_and_freeze: bool,
    winner_selection_denied: bool,
    no_runtime_server_gpu_or_tournament_authority: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    prepared: bool,
    paired_development_active: bool,
    logical_sessions_created: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    body_resource_budgets: Vec<BodyResourceBudget>,
    role_session_assignments: Vec<RoleSessionAssignment>,
    queue_fairness_telemetry_policies: Vec<QueueFairnessTelemetryPolicy>,
    paired_development_reservation: PairedDevelopmentReservation,
    final_mode_gate: FinalModeGateReport,
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
    let unsigned = Value::Object(object.clone());
    let seal = sha256_json(&unsigned)?;
    object.insert("seal_sha256".into(), Value::String(seal));
    Ok(value)
}

fn require_absolute_path(path: &str, label: &str) -> Result<(), String> {
    if !Path::new(path).is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    Ok(())
}

fn require_sealed_namespace(value: &str, label: &str) -> Result<(), String> {
    if !value.starts_with("sealed://") || value.len() <= "sealed://".len() {
        return Err(format!("{label} must be a non-empty sealed:// namespace"));
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

fn validate_cross_lane_policy(policy: &CrossLaneReadPolicy, label: &str) -> Result<(), String> {
    let prohibited = [
        ("mission", policy.allow_cross_lane_mission_reads),
        ("experiment", policy.allow_cross_lane_experiment_reads),
        ("receipt", policy.allow_cross_lane_receipt_reads),
        ("worktree", policy.allow_cross_lane_worktree_reads),
        ("session", policy.allow_cross_lane_session_reads),
        ("frontier", policy.allow_cross_lane_frontier_reads),
        ("patch", policy.allow_cross_lane_patch_reads),
        ("score", policy.allow_cross_lane_score_reads),
    ];
    for (kind, allowed) in prohibited {
        if allowed {
            return Err(format!("{label} must deny cross-lane {kind} reads"));
        }
    }
    if !policy.verified_generic_knowledge_plane_publication_only {
        return Err(format!(
            "{label} must permit only verified generic Knowledge Plane material"
        ));
    }
    Ok(())
}

fn exact_actions(actual: &[String], expected: &[&str], label: &str) -> Result<(), String> {
    let actual_set: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let expected_set: BTreeSet<&str> = expected.iter().copied().collect();
    if actual_set.len() != actual.len() || actual_set != expected_set {
        return Err(format!("{label} must grant exactly {:?}", expected_set));
    }
    Ok(())
}

fn expected_endpoint(model_key: &str) -> Result<Endpoint, String> {
    match model_key {
        QWEN30 => Ok(Endpoint {
            host: QWEN30_HOST.into(),
            port: QWEN30_PORT,
        }),
        QWEN80 => Ok(Endpoint {
            host: QWEN80_HOST.into(),
            port: QWEN80_PORT,
        }),
        other => Err(format!("unsupported model key {other:?}")),
    }
}

fn expected_body_id(model_key: &str) -> Result<String, String> {
    match model_key {
        QWEN30 | QWEN80 => Ok(format!("{model_key}-one-resident-body")),
        other => Err(format!("unsupported model key {other:?}")),
    }
}

fn expected_actor_ids(primary: &str, helper: &str) -> (String, String, String) {
    (
        format!("{primary}-primary-in-{primary}-lane"),
        format!("{helper}-helper-in-{primary}-lane"),
        format!("{helper}-opponent-reviewer-in-{primary}-lane"),
    )
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

fn validate_tg10(tg10: &Tg10Authority, model_key: &str) -> Result<(), String> {
    if tg10.required_base_true_tps != TG10_BASE_TRUE_TPS {
        return Err(format!(
            "{model_key}.tg10.required_base_true_tps must be {TG10_BASE_TRUE_TPS}"
        ));
    }
    if !tg10.operational_pass {
        return Ok(());
    }
    if !tg10.coherent_hcli_pass || !tg10.complete_token_path_measured || tg10.fallback_count != 0 {
        return Err(format!(
            "{model_key}.tg10 pass must be coherent, complete-token, and fallback-free"
        ));
    }
    let measured = tg10
        .median_base_true_tps
        .ok_or_else(|| format!("{model_key}.tg10 pass must carry measured BASE_TRUE_TPS"))?;
    if !measured.is_finite() || measured < f64::from(TG10_BASE_TRUE_TPS) {
        return Err(format!(
            "{model_key}.tg10 pass must measure >= {TG10_BASE_TRUE_TPS} BASE_TRUE_TPS"
        ));
    }
    let receipt = tg10
        .receipt_seal_sha256
        .as_deref()
        .ok_or_else(|| format!("{model_key}.tg10 pass requires a sealed receipt"))?;
    if !is_lower_sha256(receipt) {
        return Err(format!(
            "{model_key}.tg10 receipt seal must be lowercase SHA-256"
        ));
    }
    Ok(())
}

fn validate_mission_authority(authority: &MissionAuthority, label: &str) -> Result<(), String> {
    if !authority.primary_candidate_mutation_authority
        || !authority.primary_candidate_promotion_authority
        || !authority.helper_may_inspect_and_critique
        || !authority.helper_may_propose_or_test_in_private_worktree
        || authority.helper_may_mutate_primary_champion
        || authority.helper_may_promote_primary_champion
        || authority.opposite_lane_may_mutate_primary_champion
        || authority.primary_or_helper_may_self_score
    {
        return Err(format!(
            "{label} violates primary-only mutation/promotion or self-scoring denial"
        ));
    }
    Ok(())
}

fn validate_lane_authority(
    binding: &SealedDocumentBinding,
) -> Result<
    (
        EvidenceDigest,
        BTreeMap<String, BoundLane>,
        BTreeMap<String, ModelContractBindingReport>,
        BTreeSet<String>,
    ),
    String,
> {
    let digest = binding_digest(binding, "lane_authority")?;
    let document: LaneAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("lane_authority.document has the wrong grammar: {error}"))?;
    if document.schema != LANE_AUTHORITY_SCHEMA || document.status != LANE_AUTHORITY_STATUS {
        return Err("lane_authority.document is not the completed paired lane authority".into());
    }
    if !document.prepared
        || document.paired_candidate_worlds_active
        || !document.no_new_physical_model_process_authority
    {
        return Err(
            "lane_authority.document must remain prepared, inactive, and process-free".into(),
        );
    }
    validate_cross_lane_policy(
        &document.cross_lane_read_policy,
        "lane_authority.cross_lane_read_policy",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "lane_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "lane_authority.execution_boundary",
    )?;
    if document.model_contract_bindings.len() != 2 || document.lane_worlds.len() != 2 {
        return Err("lane_authority must bind exactly two models and two candidate worlds".into());
    }

    let mut models = BTreeMap::new();
    for model in document.model_contract_bindings {
        let expected_endpoint = expected_endpoint(&model.model_key)?;
        if model.topology.resident_model_processes != 1
            || model.topology.immutable_weight_copies != 1
            || model.topology.logical_session_policy != "many_logical_sessions"
            || model.topology.endpoint != expected_endpoint
        {
            return Err(format!(
                "{} topology must preserve one body/copy/endpoint and many logical sessions",
                model.model_key
            ));
        }
        validate_tg10(&model.tg10, &model.model_key)?;
        if models.insert(model.model_key.clone(), model).is_some() {
            return Err("lane_authority has duplicate model topology bindings".into());
        }
    }
    if models.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from([QWEN30, QWEN80])
    {
        return Err("lane_authority must bind exactly Q30 and Q80 bodies".into());
    }

    let mut lanes = BTreeMap::new();
    let mut private_namespaces = BTreeSet::new();
    for lane in document.lane_worlds {
        let expected_helper = match lane.primary_model_key.as_str() {
            QWEN30 => QWEN80,
            QWEN80 => QWEN30,
            other => return Err(format!("unsupported lane primary {other:?}")),
        };
        if lane.lane_id.trim().is_empty()
            || lane.helper_model_key != expected_helper
            || lanes.contains_key(&lane.lane_id)
        {
            return Err("lane_authority has invalid/duplicate primary-helper lane binding".into());
        }
        validate_mission_authority(&lane.mission_authority, &lane.lane_id)?;
        for namespace in [
            &lane.private_namespaces.mission,
            &lane.private_namespaces.experiments,
            &lane.private_namespaces.receipts,
            &lane.private_namespaces.worktree,
            &lane.private_namespaces.sessions,
            &lane.private_namespaces.frontier,
            &lane.private_namespaces.patches,
            &lane.private_namespaces.scores,
        ] {
            require_sealed_namespace(namespace, &format!("{}.private_namespaces", lane.lane_id))?;
            if !private_namespaces.insert(namespace.clone()) {
                return Err("lane_authority aliases private namespaces across lanes".into());
            }
        }
        lanes.insert(lane.lane_id.clone(), lane);
    }
    let primary_set: BTreeSet<&str> = lanes
        .values()
        .map(|lane| lane.primary_model_key.as_str())
        .collect();
    if primary_set != BTreeSet::from([QWEN30, QWEN80]) {
        return Err("lane_authority must have one Q30-primary and one Q80-primary lane".into());
    }
    Ok((digest, lanes, models, private_namespaces))
}

fn validate_mutation_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
    lanes: &BTreeMap<String, BoundLane>,
) -> Result<(EvidenceDigest, BTreeMap<String, MutationLaneActionPolicy>), String> {
    let digest = binding_digest(binding, "mutation_authority")?;
    let document: MutationAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("mutation_authority.document has the wrong grammar: {error}"))?;
    if document.schema != MUTATION_AUTHORITY_SCHEMA || document.status != MUTATION_AUTHORITY_STATUS
    {
        return Err("mutation_authority.document is not the completed mutation authority".into());
    }
    if !document.prepared || document.paired_candidate_worlds_active {
        return Err("mutation_authority.document must remain prepared and inactive".into());
    }
    if document.bound_lane_authority.document_schema != LANE_AUTHORITY_SCHEMA
        || document.bound_lane_authority.document_status != LANE_AUTHORITY_STATUS
        || document.bound_lane_authority.document_sha256 != lane_digest.document_sha256
        || document.bound_lane_authority.document_seal_sha256 != lane_digest.document_seal_sha256
    {
        return Err("mutation_authority must bind this exact sealed lane authority".into());
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "mutation_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "mutation_authority.execution_boundary",
    )?;
    let records = &document.protected_record_policy;
    if !records.evidence_receipts_immutable
        || !records.mission_records_immutable
        || !records.tournament_receipts_immutable
        || records.all_roles_may_rewrite_evidence_receipts
        || records.all_roles_may_rewrite_mission_records
        || records.all_roles_may_rewrite_tournament_receipts
        || records.all_roles_may_delete_protected_records
        || records.cross_lane_private_record_release_allowed
    {
        return Err("mutation_authority must keep protected records immutable and private".into());
    }
    if document.lane_action_policies.len() != 2 {
        return Err("mutation_authority must carry two lane action policies".into());
    }
    let mut policies = BTreeMap::new();
    for policy in document.lane_action_policies {
        let lane = lanes
            .get(&policy.lane_id)
            .ok_or_else(|| format!("mutation policy references unknown lane {}", policy.lane_id))?;
        if policy.primary_model_key != lane.primary_model_key
            || policy.helper_model_key != lane.helper_model_key
            || policy.opponent_model_key != lane.helper_model_key
        {
            return Err(format!(
                "{} mutation policy does not match lane authority",
                policy.lane_id
            ));
        }
        let actors = expected_actor_ids(&lane.primary_model_key, &lane.helper_model_key);
        if (
            policy.primary_actor_id.as_str(),
            policy.helper_actor_id.as_str(),
            policy.opponent_actor_id.as_str(),
        ) != (actors.0.as_str(), actors.1.as_str(), actors.2.as_str())
        {
            return Err(format!(
                "{} has unbound mutation-policy actors",
                policy.lane_id
            ));
        }
        exact_actions(
            &policy.primary_allowed_actions,
            &PRIMARY_ACTIONS,
            &format!("{}.primary_allowed_actions", policy.lane_id),
        )?;
        exact_actions(
            &policy.helper_allowed_actions,
            &WORK_ACTIONS,
            &format!("{}.helper_allowed_actions", policy.lane_id),
        )?;
        exact_actions(
            &policy.opponent_allowed_actions,
            &WORK_ACTIONS,
            &format!("{}.opponent_allowed_actions", policy.lane_id),
        )?;
        if policies.insert(policy.lane_id.clone(), policy).is_some() {
            return Err("mutation_authority has duplicate policy lanes".into());
        }
    }
    Ok((digest, policies))
}

fn validate_knowledge_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
    mutation_digest: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "knowledge_authority")?;
    let document: KnowledgeAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("knowledge_authority.document has the wrong grammar: {error}"))?;
    if document.schema != KNOWLEDGE_AUTHORITY_SCHEMA
        || document.status != KNOWLEDGE_AUTHORITY_STATUS
    {
        return Err("knowledge_authority.document is not the completed generic-only Knowledge Plane authority".into());
    }
    if !document.prepared
        || document.knowledge_plane_active
        || document.external_publication_performed
    {
        return Err("knowledge_authority.document must remain prepared, inactive, and externally unpublished".into());
    }
    for (label, expected, observed) in [
        ("lane", lane_digest, &document.bound_lane_authority),
        (
            "mutation",
            mutation_digest,
            &document.bound_mutation_authority,
        ),
    ] {
        if observed.document_sha256 != expected.document_sha256
            || observed.document_seal_sha256 != expected.document_seal_sha256
        {
            return Err(format!(
                "knowledge_authority is not bound to the exact {label} authority"
            ));
        }
    }
    let policy = &document.knowledge_plane_policy;
    if !policy.generic_mechanism_science_only
        || !policy.append_only_release_identities
        || policy.release_identity_mutation_authorized
        || policy.external_publication_authorized
        || policy.lane_private_record_access_authorized
        || policy.candidate_world_activation_authorized
    {
        return Err(
            "knowledge_authority policy must stay generic-only, append-only, inactive, and private"
                .into(),
        );
    }
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

fn validate_resource_budgets(
    budgets: &[BodyResourceBudget],
    models: &BTreeMap<String, ModelContractBindingReport>,
) -> Result<BTreeMap<String, BodyResourceBudget>, String> {
    if budgets.len() != 2 {
        return Err("exactly one resource budget for each physical body is required".into());
    }
    let mut result = BTreeMap::new();
    let mut endpoints = BTreeSet::new();
    let mut body_ids = BTreeSet::new();
    for budget in budgets {
        let model = models.get(&budget.model_key).ok_or_else(|| {
            format!(
                "resource budget references unknown body {}",
                budget.model_key
            )
        })?;
        if budget.physical_body_id != expected_body_id(&budget.model_key)?
            || budget.endpoint != model.topology.endpoint
            || budget.resident_model_processes != 1
            || budget.immutable_weight_copies != 1
            || budget.max_logical_sessions != 2
            || budget.max_inflight_turns != 1
        {
            return Err(format!("{} budget must preserve one body/copy, exact endpoint, two roles, and one in-flight turn", budget.model_key));
        }
        if !endpoints.insert((budget.endpoint.host.clone(), budget.endpoint.port))
            || !body_ids.insert(budget.physical_body_id.clone())
            || result
                .insert(budget.model_key.clone(), budget.clone())
                .is_some()
        {
            return Err(
                "resource budgets must not duplicate an endpoint, physical body, or model".into(),
            );
        }
    }
    if result.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from([QWEN30, QWEN80])
    {
        return Err("resource budgets must bind exactly Q30 and Q80".into());
    }
    Ok(result)
}

fn expected_assignment(
    lane: &BoundLane,
    policy: &MutationLaneActionPolicy,
    role: &str,
) -> Result<(String, String), String> {
    match role {
        "primary" => Ok((
            policy.primary_actor_id.clone(),
            lane.primary_model_key.clone(),
        )),
        "helper" => Ok((
            policy.helper_actor_id.clone(),
            lane.helper_model_key.clone(),
        )),
        other => Err(format!("unsupported role {other:?}")),
    }
}

fn validate_assignments(
    assignments: &[RoleSessionAssignment],
    lanes: &BTreeMap<String, BoundLane>,
    policies: &BTreeMap<String, MutationLaneActionPolicy>,
    budgets: &BTreeMap<String, BodyResourceBudget>,
) -> Result<Vec<RoleSessionAssignment>, String> {
    if assignments.len() != 4 {
        return Err("exactly four primary/helper logical role assignments are required".into());
    }
    let mut seen_role_keys = BTreeSet::new();
    let mut actors = BTreeSet::new();
    let mut sessions = BTreeSet::new();
    let mut queues = BTreeSet::new();
    let mut per_body = BTreeMap::<String, usize>::new();
    for assignment in assignments {
        let lane = lanes
            .get(&assignment.lane_id)
            .ok_or_else(|| format!("assignment references unknown lane {}", assignment.lane_id))?;
        let policy = policies.get(&assignment.lane_id).ok_or_else(|| {
            format!(
                "assignment has no bound mutation policy for {}",
                assignment.lane_id
            )
        })?;
        let (expected_actor, expected_body) = expected_assignment(lane, policy, &assignment.role)?;
        let budget = budgets.get(&assignment.physical_model_key).ok_or_else(|| {
            format!(
                "assignment references unbudgeted body {}",
                assignment.physical_model_key
            )
        })?;
        if assignment.logical_actor_id != expected_actor
            || assignment.physical_model_key != expected_body
            || assignment.physical_body_id != budget.physical_body_id
            || assignment.scheduling_weight != 1
            || assignment.max_queued_requests == 0
            || assignment.max_queued_requests > 8
            || assignment.max_inflight_turns != 1
            || !assignment.private_lane_visibility_only
        {
            return Err(format!(
                "{} {} assignment violates a fixed actor/body/resource invariant",
                assignment.lane_id, assignment.role
            ));
        }
        let session_prefix = format!("{}/logical/", lane.private_namespaces.sessions);
        if !assignment.session_namespace.starts_with(&session_prefix)
            || assignment.session_namespace
                != format!("{session_prefix}{}", assignment.logical_actor_id)
            || assignment.queue_namespace != format!("{}/queue", assignment.session_namespace)
        {
            return Err(format!(
                "{} {} must own a unique lane-private logical session and queue",
                assignment.lane_id, assignment.role
            ));
        }
        require_sealed_namespace(
            &assignment.session_namespace,
            "assignment.session_namespace",
        )?;
        require_sealed_namespace(&assignment.queue_namespace, "assignment.queue_namespace")?;
        if !seen_role_keys.insert((assignment.lane_id.clone(), assignment.role.clone()))
            || !actors.insert(assignment.logical_actor_id.clone())
            || !sessions.insert(assignment.session_namespace.clone())
            || !queues.insert(assignment.queue_namespace.clone())
        {
            return Err(
                "role actors, sessions, queues, and lane-role slots must all be unique".into(),
            );
        }
        *per_body
            .entry(assignment.physical_model_key.clone())
            .or_default() += 1;
    }
    if seen_role_keys
        != BTreeSet::from([
            ("qwen30-candidate-world".into(), "primary".into()),
            ("qwen30-candidate-world".into(), "helper".into()),
            ("qwen80-candidate-world".into(), "primary".into()),
            ("qwen80-candidate-world".into(), "helper".into()),
        ])
    {
        return Err("assignments must contain only the two primary and two helper roles".into());
    }
    for model_key in [QWEN30, QWEN80] {
        if per_body.get(model_key) != Some(&2) {
            return Err(format!(
                "{model_key} must host exactly two logical roles, not a cloned body"
            ));
        }
    }
    let mut result = assignments.to_vec();
    result.sort_by(|left, right| {
        (
            left.physical_model_key.as_str(),
            left.logical_actor_id.as_str(),
        )
            .cmp(&(
                right.physical_model_key.as_str(),
                right.logical_actor_id.as_str(),
            ))
    });
    Ok(result)
}

fn validate_telemetry(
    telemetry: &[QueueFairnessTelemetryPolicy],
    budgets: &BTreeMap<String, BodyResourceBudget>,
) -> Result<Vec<QueueFairnessTelemetryPolicy>, String> {
    if telemetry.len() != 2 {
        return Err("exactly one fairness telemetry policy per physical body is required".into());
    }
    let mut seen_models = BTreeSet::new();
    let mut namespaces = BTreeSet::new();
    for policy in telemetry {
        let budget = budgets.get(&policy.physical_model_key).ok_or_else(|| {
            format!(
                "telemetry references unbudgeted body {}",
                policy.physical_model_key
            )
        })?;
        let required: BTreeSet<&str> = TELEMETRY_FIELDS.iter().copied().collect();
        let actual: BTreeSet<&str> = policy
            .required_counter_fields
            .iter()
            .map(String::as_str)
            .collect();
        if policy.physical_body_id != budget.physical_body_id
            || policy.telemetry_namespace
                != format!(
                    "sealed://paired-scheduler/{}/fairness-telemetry",
                    policy.physical_model_key
                )
            || policy.scheduling_discipline != ROUND_ROBIN_EQUAL_WEIGHT
            || policy.max_inflight_turns_per_body != 1
            || policy.max_consecutive_dispatches_to_same_session != 1
            || policy.fairness_lag_bound_dispatches != 1
            || actual.len() != policy.required_counter_fields.len()
            || actual != required
            || policy.telemetry_contains_request_content
            || policy.telemetry_contains_generated_tokens
            || policy.telemetry_contains_lane_private_scores
            || policy.telemetry_contains_private_namespace_identifiers
        {
            return Err(format!(
                "{} telemetry must be bounded, equal-fair, and content-free",
                policy.physical_model_key
            ));
        }
        require_sealed_namespace(&policy.telemetry_namespace, "telemetry_namespace")?;
        if !seen_models.insert(policy.physical_model_key.clone())
            || !namespaces.insert(policy.telemetry_namespace.clone())
        {
            return Err("telemetry model and namespace identities must be unique".into());
        }
    }
    if seen_models
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != BTreeSet::from([QWEN30, QWEN80])
    {
        return Err("telemetry must bind exactly Q30 and Q80 bodies".into());
    }
    let mut result = telemetry.to_vec();
    result.sort_by(|left, right| left.physical_model_key.cmp(&right.physical_model_key));
    Ok(result)
}

fn readiness(model: &ModelContractBindingReport) -> Tg10Readiness {
    Tg10Readiness {
        model_key: model.model_key.clone(),
        required_base_true_tps: model.tg10.required_base_true_tps,
        operational_pass: model.tg10.operational_pass,
        receipt_seal_sha256: model.tg10.receipt_seal_sha256.clone(),
    }
}

fn validate_final_receipt(receipt: &Tg3Receipt, model_key: &str) -> Result<(), String> {
    if receipt.model_key != model_key
        || receipt.tg_level != 3
        || !receipt.operational_pass
        || !is_lower_sha256(&receipt.receipt_seal_sha256)
    {
        return Err(format!(
            "{model_key} TG3 receipt must be a sealed operational TG3 pass"
        ));
    }
    Ok(())
}

fn validate_final_modes(reservation: &FinalModeReservation) -> Result<FinalModeGateReport, String> {
    if reservation.solo_manager_evaluation_requested
        || reservation.symmetric_orchestrator_evaluation_requested
        || reservation.winner_selection_requested
        || reservation.final_mode_authorized_by_this_contract
        || reservation.winner_selection_authorized_by_this_contract
    {
        return Err("this scheduler may not activate final modes or select a winner".into());
    }
    let qwen30 = reservation.qwen30_tg3_receipt.as_ref();
    let qwen80 = reservation.qwen80_tg3_receipt.as_ref();
    if let Some(receipt) = qwen30 {
        validate_final_receipt(receipt, QWEN30)?;
    }
    if let Some(receipt) = qwen80 {
        validate_final_receipt(receipt, QWEN80)?;
    }
    if let Some(seal) = &reservation.post_tg3_freeze_seal_sha256 {
        if !is_lower_sha256(seal) {
            return Err(
                "post_tg3_freeze_seal_sha256 must be lowercase SHA-256 when present".into(),
            );
        }
    }
    let both_tg3 = qwen30.is_some() && qwen80.is_some();
    let freeze = reservation.post_tg3_freeze_seal_sha256.is_some();
    Ok(FinalModeGateReport {
        both_tg3_operational_receipts_present: both_tg3,
        post_tg3_freeze_present: freeze,
        final_modes_ready_after_both_tg3_and_freeze: both_tg3 && freeze,
        solo_manager_evaluation_authorized_by_this_contract: false,
        symmetric_orchestrator_evaluation_authorized_by_this_contract: false,
        winner_selection_authorized_by_this_contract: false,
    })
}

fn build_report(input: Input) -> Result<Report, String> {
    if input.schema != INPUT_SCHEMA {
        return Err(format!(
            "input.schema must be {INPUT_SCHEMA:?}, observed {:?}",
            input.schema
        ));
    }
    if !is_lower_sha256(&input.seal_sha256) {
        return Err("input.seal_sha256 must be a lowercase SHA-256".into());
    }
    if input.paired_development_activation_requested {
        return Err("this scheduler cannot activate paired development; both TG10 receipts remain a later protected-controller gate".into());
    }
    let (lane_digest, lanes, models, _) = validate_lane_authority(&input.lane_authority)?;
    let (mutation_digest, policies) =
        validate_mutation_authority(&input.mutation_authority, &lane_digest, &lanes)?;
    let knowledge_digest =
        validate_knowledge_authority(&input.knowledge_authority, &lane_digest, &mutation_digest)?;
    let budgets = validate_resource_budgets(&input.body_resource_budgets, &models)?;
    let assignments =
        validate_assignments(&input.role_session_assignments, &lanes, &policies, &budgets)?;
    let telemetry = validate_telemetry(&input.queue_fairness_telemetry_policies, &budgets)?;
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;
    let qwen30 = models.get(QWEN30).expect("validated Q30 binding");
    let qwen80 = models.get(QWEN80).expect("validated Q80 binding");
    let qwen30_tg10 = readiness(qwen30);
    let qwen80_tg10 = readiness(qwen80);
    let both_tg10 = qwen30_tg10.operational_pass && qwen80_tg10.operational_pass;
    let development = PairedDevelopmentReservation {
        required_base_true_tps: TG10_BASE_TRUE_TPS,
        qwen30_tg10,
        qwen80_tg10,
        both_tg10_operational_receipts_present: both_tg10,
        paired_development_ready_after_both_tg10: both_tg10,
        paired_development_activation_authorized_by_this_contract: false,
    };
    let final_mode_gate = validate_final_modes(&input.final_mode_reservation)?;
    let mut body_resource_budgets = budgets.into_values().collect::<Vec<_>>();
    body_resource_budgets.sort_by(|left, right| left.model_key.cmp(&right.model_key));
    let focused_checks = FocusedChecks {
        exact_sealed_lane_mutation_and_knowledge_authorities_bound: true,
        one_q30_body_and_one_q80_body_only: true,
        one_weight_copy_and_one_endpoint_per_body: true,
        four_expected_primary_helper_roles_and_no_extra_logical_sessions: true,
        all_session_and_queue_namespaces_are_lane_private_and_distinct: true,
        role_body_bindings_are_heterogeneous_and_nonduplicating: true,
        bounded_queues_and_equal_fairness_telemetry_are_private_and_content_free: true,
        cross_lane_private_reads_and_self_scoring_denied: true,
        paired_development_reserved_until_two_valid_tg10_receipts: true,
        solo_and_orchestrator_final_modes_reserved_until_two_tg3_receipts_and_freeze: true,
        winner_selection_denied: true,
        no_runtime_server_gpu_or_tournament_authority: true,
        execution_boundary_cpu_only: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status: STATUS,
        prepared: true,
        paired_development_active: false,
        logical_sessions_created: false,
        bound_lane_authority: lane_digest,
        bound_mutation_authority: mutation_digest,
        bound_knowledge_authority: knowledge_digest,
        body_resource_budgets,
        role_session_assignments: assignments,
        queue_fairness_telemetry_policies: telemetry,
        paired_development_reservation: development,
        final_mode_gate,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only role/resource scheduler authority, not a process launcher, logical-session creator, or live scheduler.",
            "It binds exactly one Q30 body at 127.0.0.1:18430 and one Q80 body at 127.0.0.1:18480, each with one immutable weight copy and two reserved logical roles.",
            "Q80-primary/Q30-helper occupy the Q80 lane and Q30-primary/Q80-helper occupy the Q30 lane; every role session and queue is unique and remains in its own lane-private session namespace.",
            "Per-body queues are bounded and one-inflight with equal round-robin service; telemetry carries only aggregate counters, never request content, generated tokens, private namespaces, or candidate scores.",
            "Private cross-lane reads, helper/opponent champion control, self-scoring, cloned bodies/endpoints/weights, activation, and winner selection remain prohibited.",
            "Paired development is reserved only after both sealed coherent complete TG10 operational receipts measure at least 100 BASE_TRUE_TPS, but this contract cannot activate it.",
            "Solo manager and symmetric manager-as-orchestrator evaluations remain reserved until both TG3 receipts and a post-TG3 freeze exist; this contract never selects a winner.",
            "No model, GPU, server, watcher, port, HCLI, token, TPS/TG measurement, tournament, or runtime state is touched.",
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
    let value = seal_value(
        serde_json::to_value(report)
            .map_err(|error| format!("scheduler report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("scheduler report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_paired_cognition_role_resource_scheduler_authority --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
    let report =
        build_report(input).map_err(|error| format!("scheduler validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_paired_cognition_role_resource_scheduler_authority: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn sha(character: char) -> String {
        character.to_string().repeat(64)
    }

    fn endpoint(model_key: &str) -> Endpoint {
        expected_endpoint(model_key).unwrap()
    }

    fn topology(model_key: &str) -> TopologyAssertion {
        TopologyAssertion {
            resident_model_processes: 1,
            immutable_weight_copies: 1,
            logical_session_policy: "many_logical_sessions".into(),
            endpoint: endpoint(model_key),
        }
    }

    fn tg10() -> Tg10Authority {
        Tg10Authority {
            required_base_true_tps: TG10_BASE_TRUE_TPS,
            operational_pass: false,
            coherent_hcli_pass: false,
            complete_token_path_measured: false,
            fallback_count: 0,
            median_base_true_tps: None,
            receipt_seal_sha256: None,
        }
    }

    fn model(model_key: &str) -> ModelContractBindingReport {
        ModelContractBindingReport {
            model_key: model_key.into(),
            topology: topology(model_key),
            tg10: tg10(),
        }
    }

    fn private_namespaces(lane: &str) -> PrivateNamespaces {
        PrivateNamespaces {
            mission: format!("sealed://{lane}/mission"),
            experiments: format!("sealed://{lane}/experiments"),
            receipts: format!("sealed://{lane}/receipts"),
            worktree: format!("sealed://{lane}/worktree"),
            sessions: format!("sealed://{lane}/sessions"),
            frontier: format!("sealed://{lane}/frontier"),
            patches: format!("sealed://{lane}/patches"),
            scores: format!("sealed://{lane}/scores"),
        }
    }

    fn mission_authority() -> MissionAuthority {
        MissionAuthority {
            primary_candidate_mutation_authority: true,
            primary_candidate_promotion_authority: true,
            helper_may_inspect_and_critique: true,
            helper_may_propose_or_test_in_private_worktree: true,
            helper_may_mutate_primary_champion: false,
            helper_may_promote_primary_champion: false,
            opposite_lane_may_mutate_primary_champion: false,
            primary_or_helper_may_self_score: false,
        }
    }

    fn lane(primary: &str, helper: &str) -> BoundLane {
        BoundLane {
            lane_id: format!("{primary}-candidate-world"),
            primary_model_key: primary.into(),
            helper_model_key: helper.into(),
            private_namespaces: private_namespaces(&format!("{primary}-lane")),
            mission_authority: mission_authority(),
        }
    }

    fn cross_lane_policy() -> CrossLaneReadPolicy {
        CrossLaneReadPolicy {
            allow_cross_lane_mission_reads: false,
            allow_cross_lane_experiment_reads: false,
            allow_cross_lane_receipt_reads: false,
            allow_cross_lane_worktree_reads: false,
            allow_cross_lane_session_reads: false,
            allow_cross_lane_frontier_reads: false,
            allow_cross_lane_patch_reads: false,
            allow_cross_lane_score_reads: false,
            verified_generic_knowledge_plane_publication_only: true,
        }
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

    fn sealed_lane_document() -> Value {
        seal_value(json!({
            "schema": LANE_AUTHORITY_SCHEMA,
            "status": LANE_AUTHORITY_STATUS,
            "prepared": true,
            "paired_candidate_worlds_active": false,
            "no_new_physical_model_process_authority": true,
            "model_contract_bindings": [model(QWEN30), model(QWEN80)],
            "lane_worlds": [lane(QWEN30, QWEN80), lane(QWEN80, QWEN30)],
            "cross_lane_read_policy": cross_lane_policy(),
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }))
        .unwrap()
    }

    fn lane_binding() -> SealedDocumentBinding {
        let document = sealed_lane_document();
        SealedDocumentBinding {
            path: "/sealed/paired-cognition/PAIRED_COGNITION_LANE_AUTHORITY.json".into(),
            document_sha256: sha256_json(&document).unwrap(),
            document,
        }
    }

    fn action_policy(primary: &str, helper: &str) -> MutationLaneActionPolicy {
        let (primary_actor_id, helper_actor_id, opponent_actor_id) =
            expected_actor_ids(primary, helper);
        MutationLaneActionPolicy {
            lane_id: format!("{primary}-candidate-world"),
            primary_model_key: primary.into(),
            helper_model_key: helper.into(),
            opponent_model_key: helper.into(),
            primary_actor_id,
            helper_actor_id,
            opponent_actor_id,
            primary_allowed_actions: PRIMARY_ACTIONS.iter().map(|item| (*item).into()).collect(),
            helper_allowed_actions: WORK_ACTIONS.iter().map(|item| (*item).into()).collect(),
            opponent_allowed_actions: WORK_ACTIONS.iter().map(|item| (*item).into()).collect(),
        }
    }

    fn protected_records() -> ProtectedRecordPolicy {
        ProtectedRecordPolicy {
            evidence_receipts_immutable: true,
            mission_records_immutable: true,
            tournament_receipts_immutable: true,
            all_roles_may_rewrite_evidence_receipts: false,
            all_roles_may_rewrite_mission_records: false,
            all_roles_may_rewrite_tournament_receipts: false,
            all_roles_may_delete_protected_records: false,
            cross_lane_private_record_release_allowed: false,
        }
    }

    fn sealed_mutation_document(lane: &SealedDocumentBinding) -> Value {
        seal_value(json!({
            "schema": MUTATION_AUTHORITY_SCHEMA,
            "status": MUTATION_AUTHORITY_STATUS,
            "prepared": true,
            "paired_candidate_worlds_active": false,
            "bound_lane_authority": {
                "path": lane.path,
                "document_schema": LANE_AUTHORITY_SCHEMA,
                "document_status": LANE_AUTHORITY_STATUS,
                "document_sha256": lane.document_sha256,
                "document_seal_sha256": lane.document["seal_sha256"],
            },
            "lane_action_policies": [action_policy(QWEN30, QWEN80), action_policy(QWEN80, QWEN30)],
            "protected_record_policy": protected_records(),
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }))
        .unwrap()
    }

    fn mutation_binding(lane: &SealedDocumentBinding) -> SealedDocumentBinding {
        let document = sealed_mutation_document(lane);
        SealedDocumentBinding {
            path: "/sealed/paired-cognition/PAIRED_COGNITION_MUTATION_AUTHORITY.json".into(),
            document_sha256: sha256_json(&document).unwrap(),
            document,
        }
    }

    fn knowledge_policy() -> KnowledgePlanePolicy {
        KnowledgePlanePolicy {
            knowledge_plane_namespace: "sealed://knowledge-plane/generic-mechanism-science".into(),
            release_registry_namespace: "sealed://knowledge-plane/append-only-release-registry"
                .into(),
            generic_mechanism_science_only: true,
            append_only_release_identities: true,
            release_identity_mutation_authorized: false,
            external_publication_authorized: false,
            lane_private_record_access_authorized: false,
            candidate_world_activation_authorized: false,
        }
    }

    fn sealed_knowledge_document(
        lane: &SealedDocumentBinding,
        mutation: &SealedDocumentBinding,
    ) -> Value {
        seal_value(json!({
            "schema": KNOWLEDGE_AUTHORITY_SCHEMA,
            "status": KNOWLEDGE_AUTHORITY_STATUS,
            "prepared": true,
            "knowledge_plane_active": false,
            "external_publication_performed": false,
            "bound_lane_authority": {
                "path": lane.path,
                "document_schema": LANE_AUTHORITY_SCHEMA,
                "document_status": LANE_AUTHORITY_STATUS,
                "document_sha256": lane.document_sha256,
                "document_seal_sha256": lane.document["seal_sha256"],
            },
            "bound_mutation_authority": {
                "path": mutation.path,
                "document_schema": MUTATION_AUTHORITY_SCHEMA,
                "document_status": MUTATION_AUTHORITY_STATUS,
                "document_sha256": mutation.document_sha256,
                "document_seal_sha256": mutation.document["seal_sha256"],
            },
            "knowledge_plane_policy": knowledge_policy(),
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }))
        .unwrap()
    }

    fn knowledge_binding(
        lane: &SealedDocumentBinding,
        mutation: &SealedDocumentBinding,
    ) -> SealedDocumentBinding {
        let document = sealed_knowledge_document(lane, mutation);
        SealedDocumentBinding {
            path: "/sealed/paired-cognition/PAIRED_COGNITION_KNOWLEDGE_AUTHORITY.json".into(),
            document_sha256: sha256_json(&document).unwrap(),
            document,
        }
    }

    fn budget(model_key: &str) -> BodyResourceBudget {
        BodyResourceBudget {
            model_key: model_key.into(),
            physical_body_id: expected_body_id(model_key).unwrap(),
            endpoint: endpoint(model_key),
            resident_model_processes: 1,
            immutable_weight_copies: 1,
            max_logical_sessions: 2,
            max_inflight_turns: 1,
        }
    }

    fn assignment(primary: &str, role: &str) -> RoleSessionAssignment {
        let helper = if primary == QWEN30 { QWEN80 } else { QWEN30 };
        let (primary_actor, helper_actor, _) = expected_actor_ids(primary, helper);
        let (actor, physical_model_key) = if role == "primary" {
            (primary_actor, primary)
        } else {
            (helper_actor, helper)
        };
        let session_namespace = format!("sealed://{primary}-lane/sessions/logical/{actor}");
        RoleSessionAssignment {
            lane_id: format!("{primary}-candidate-world"),
            role: role.into(),
            logical_actor_id: actor,
            physical_model_key: physical_model_key.into(),
            physical_body_id: expected_body_id(physical_model_key).unwrap(),
            queue_namespace: format!("{session_namespace}/queue"),
            session_namespace,
            scheduling_weight: 1,
            max_queued_requests: 8,
            max_inflight_turns: 1,
            private_lane_visibility_only: true,
        }
    }

    fn telemetry(model_key: &str) -> QueueFairnessTelemetryPolicy {
        QueueFairnessTelemetryPolicy {
            physical_model_key: model_key.into(),
            physical_body_id: expected_body_id(model_key).unwrap(),
            telemetry_namespace: format!(
                "sealed://paired-scheduler/{model_key}/fairness-telemetry"
            ),
            scheduling_discipline: ROUND_ROBIN_EQUAL_WEIGHT.into(),
            max_inflight_turns_per_body: 1,
            max_consecutive_dispatches_to_same_session: 1,
            fairness_lag_bound_dispatches: 1,
            required_counter_fields: TELEMETRY_FIELDS
                .iter()
                .map(|field| (*field).into())
                .collect(),
            telemetry_contains_request_content: false,
            telemetry_contains_generated_tokens: false,
            telemetry_contains_lane_private_scores: false,
            telemetry_contains_private_namespace_identifiers: false,
        }
    }

    fn final_reservation() -> FinalModeReservation {
        FinalModeReservation {
            qwen30_tg3_receipt: None,
            qwen80_tg3_receipt: None,
            post_tg3_freeze_seal_sha256: None,
            solo_manager_evaluation_requested: false,
            symmetric_orchestrator_evaluation_requested: false,
            winner_selection_requested: false,
            final_mode_authorized_by_this_contract: false,
            winner_selection_authorized_by_this_contract: false,
        }
    }

    fn input_fixture() -> Input {
        let lane_authority = lane_binding();
        let mutation_authority = mutation_binding(&lane_authority);
        let knowledge_authority = knowledge_binding(&lane_authority, &mutation_authority);
        Input {
            schema: INPUT_SCHEMA.into(),
            lane_authority,
            mutation_authority,
            knowledge_authority,
            body_resource_budgets: vec![budget(QWEN30), budget(QWEN80)],
            role_session_assignments: vec![
                assignment(QWEN30, "primary"),
                assignment(QWEN30, "helper"),
                assignment(QWEN80, "primary"),
                assignment(QWEN80, "helper"),
            ],
            queue_fairness_telemetry_policies: vec![telemetry(QWEN30), telemetry(QWEN80)],
            paired_development_activation_requested: false,
            final_mode_reservation: final_reservation(),
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
    fn correct_four_role_schedule_binds_three_authorities_and_stays_prepared() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, STATUS);
        assert!(report.prepared);
        assert!(!report.paired_development_active);
        assert!(!report.logical_sessions_created);
        assert_eq!(report.body_resource_budgets.len(), 2);
        assert_eq!(report.role_session_assignments.len(), 4);
        assert!(report.focused_checks.one_q30_body_and_one_q80_body_only);
        assert!(
            report
                .focused_checks
                .role_body_bindings_are_heterogeneous_and_nonduplicating
        );
        assert!(report.focused_checks.execution_boundary_cpu_only);
        assert!(
            !report
                .paired_development_reservation
                .both_tg10_operational_receipts_present
        );
    }

    #[test]
    fn exact_sealed_authority_chain_and_private_read_denials_are_required() {
        let mut wrong_knowledge = input_fixture();
        wrong_knowledge.knowledge_authority.document["bound_mutation_authority"]
            ["document_sha256"] = Value::String(sha('0'));
        let document = wrong_knowledge.knowledge_authority.document.clone();
        wrong_knowledge.knowledge_authority.document = seal_value(document).unwrap();
        wrong_knowledge.knowledge_authority.document_sha256 =
            sha256_json(&wrong_knowledge.knowledge_authority.document).unwrap();
        assert!(build_report(wrong_knowledge).is_err());

        let mut cross_lane = input_fixture();
        cross_lane.lane_authority.document["cross_lane_read_policy"]
            ["allow_cross_lane_session_reads"] = Value::Bool(true);
        let document = cross_lane.lane_authority.document.clone();
        cross_lane.lane_authority.document = seal_value(document).unwrap();
        cross_lane.lane_authority.document_sha256 =
            sha256_json(&cross_lane.lane_authority.document).unwrap();
        assert!(build_report(cross_lane).is_err());
    }

    #[test]
    fn cloned_body_weight_endpoint_and_role_body_misbindings_are_rejected() {
        let mut cloned = input_fixture();
        cloned.body_resource_budgets[0].immutable_weight_copies = 2;
        assert!(build_report(cloned).is_err());

        let mut endpoint_clone = input_fixture();
        endpoint_clone.body_resource_budgets[1].endpoint = endpoint(QWEN30);
        assert!(build_report(endpoint_clone).is_err());

        let mut wrong_body = input_fixture();
        wrong_body.role_session_assignments[0].physical_model_key = QWEN80.into();
        assert!(build_report(wrong_body).is_err());
    }

    #[test]
    fn duplicate_or_cross_lane_sessions_and_unbounded_queues_are_rejected() {
        let mut duplicate_session = input_fixture();
        duplicate_session.role_session_assignments[1].session_namespace = duplicate_session
            .role_session_assignments[0]
            .session_namespace
            .clone();
        assert!(build_report(duplicate_session).is_err());

        let mut cross_lane_session = input_fixture();
        cross_lane_session.role_session_assignments[0].session_namespace =
            "sealed://qwen80-lane/sessions/logical/qwen30-primary-in-qwen30-lane".into();
        assert!(build_report(cross_lane_session).is_err());

        let mut unbounded_queue = input_fixture();
        unbounded_queue.role_session_assignments[0].max_queued_requests = 9;
        assert!(build_report(unbounded_queue).is_err());
    }

    #[test]
    fn unfair_or_contentful_telemetry_and_self_scoring_are_rejected() {
        let mut unfair = input_fixture();
        unfair.queue_fairness_telemetry_policies[0].max_consecutive_dispatches_to_same_session = 2;
        assert!(build_report(unfair).is_err());

        let mut private_telemetry = input_fixture();
        private_telemetry.queue_fairness_telemetry_policies[0]
            .telemetry_contains_lane_private_scores = true;
        assert!(build_report(private_telemetry).is_err());

        let mut self_score = input_fixture();
        self_score.lane_authority.document["lane_worlds"][0]["mission_authority"]
            ["primary_or_helper_may_self_score"] = Value::Bool(true);
        let document = self_score.lane_authority.document.clone();
        self_score.lane_authority.document = seal_value(document).unwrap();
        self_score.lane_authority.document_sha256 =
            sha256_json(&self_score.lane_authority.document).unwrap();
        assert!(build_report(self_score).is_err());
    }

    #[test]
    fn early_development_final_modes_and_winner_selection_are_rejected() {
        let mut early = input_fixture();
        early.paired_development_activation_requested = true;
        assert!(build_report(early).is_err());

        let mut final_mode = input_fixture();
        final_mode
            .final_mode_reservation
            .solo_manager_evaluation_requested = true;
        assert!(build_report(final_mode).is_err());

        let mut winner = input_fixture();
        winner.final_mode_reservation.winner_selection_requested = true;
        assert!(build_report(winner).is_err());
    }

    #[test]
    fn valid_tg3_and_freeze_can_only_mark_later_readiness_not_authorize_final_mode() {
        let mut input = input_fixture();
        input.final_mode_reservation.qwen30_tg3_receipt = Some(Tg3Receipt {
            model_key: QWEN30.into(),
            tg_level: 3,
            operational_pass: true,
            receipt_seal_sha256: sha('1'),
        });
        input.final_mode_reservation.qwen80_tg3_receipt = Some(Tg3Receipt {
            model_key: QWEN80.into(),
            tg_level: 3,
            operational_pass: true,
            receipt_seal_sha256: sha('2'),
        });
        input.final_mode_reservation.post_tg3_freeze_seal_sha256 = Some(sha('3'));
        let report = build_report(input).unwrap();
        assert!(
            report
                .final_mode_gate
                .final_modes_ready_after_both_tg3_and_freeze
        );
        assert!(
            !report
                .final_mode_gate
                .solo_manager_evaluation_authorized_by_this_contract
        );
        assert!(
            !report
                .final_mode_gate
                .winner_selection_authorized_by_this_contract
        );
    }

    #[test]
    fn cpu_only_boundaries_and_sealed_create_new_io_are_enforced() {
        let mut runtime = input_fixture();
        runtime.execution_boundary.runtime_watcher_or_server_started = true;
        assert!(build_report(runtime).is_err());

        let directory = tempdir().unwrap();
        let input_path = directory.path().join("input.json");
        let output_path = directory.path().join("out.json");
        let sealed = seal_input(&input_fixture());
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        run(Args {
            input: input_path,
            out: output_path.clone(),
        })
        .unwrap();
        let output: Value = serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
        verify_sealed_object(&output, "output").unwrap();
        assert_eq!(output["status"], Value::String(STATUS.into()));
        assert!(
            write_report_create_new(&output_path, &build_report(input_fixture()).unwrap()).is_err()
        );
    }

    #[test]
    fn tampered_outer_input_seal_is_rejected_before_scheduler_evaluation() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("tampered-input.json");
        let output_path = directory.path().join("out.json");
        let mut sealed = seal_input(&input_fixture());
        sealed["paired_development_activation_requested"] = Value::Bool(true);
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        assert!(run(Args {
            input: input_path,
            out: output_path,
        })
        .is_err());
    }
}
