//! Sealed final-manager tournament side-by-side report contract.
//!
//! This CPU-only contract binds the final-manager protocol, blind protected
//! corpus, complete scorecard adjudication, and protected selection/recovery
//! authority.  It validates a complete protected report *structure* for both
//! candidates, including required decision fields, while intentionally never
//! choosing a winner, emitting a report externally, or touching runtime or
//! artifacts.  A prepared result is an immutable report reservation for a later
//! protected controller/verifier, not a final decision.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.manager_tournament_final_report_contract_input.v1";
const RESULT_SCHEMA: &str = "hawking.ascension.manager_tournament_final_report_contract.v1";
const FINAL_PROTOCOL_SCHEMA: &str = "hawking.ascension.final_manager_tournament_protocol.v1";
const FINAL_PROTOCOL_STATUS: &str = "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED";
const FINAL_PROTOCOL_IDENTITY: &str =
    "8e3684af0b7de53690a9c88ce0d52b0cae019e0d798bc2748b5b1556211facf8";
const CORPUS_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_task_corpus_commitment_authority.v1";
const CORPUS_PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PROTECTED_REAL_HAWKING_TASK_CORPUS_METADATA_COMMITTED_NO_HIDDEN_TASKS_OR_SCORED_EXECUTION";
const SCORECARD_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_scorecard_adjudication.v1";
const SCORECARD_ADJUDICATED_STATUS: &str =
    "ADJUDICATED_COMPLETE_TWO_CANDIDATE_TWO_MODE_PARETO_MATRIX_NO_WINNER";
const SELECTION_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_selection_and_recovery_contract.v1";
const SELECTION_PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PROTECTED_SELECTION_AND_RECOVERY_CONTRACT_COMPLETE_NO_WINNER_SELECTED";
const SELECTION_REFUSED_STATUS: &str =
    "REFUSED_PROTECTED_SELECTION_AND_RECOVERY_EVIDENCE_OR_ACTION_REQUEST_INCOMPLETE";
const PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PROTECTED_FINAL_MANAGER_SIDE_BY_SIDE_REPORT_RESERVED_NO_WINNER_OR_EXTERNAL_EMISSION";
const REFUSED_STATUS: &str =
    "REFUSED_PROTECTED_FINAL_MANAGER_REPORT_PARTIAL_SELF_AUTHORED_UNVERIFIED_OR_ACTION_REQUESTED";

const CANDIDATES: [&str; 2] = [
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
];
const REQUIRED_REPORT_FIELDS: [&str; 21] = [
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
const DECISION_FIELDS: [&str; 5] = [
    "winner",
    "loser",
    "decisive_evidence",
    "tradeoffs",
    "restore_path",
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
struct FinalProtocolFinalReport {
    side_by_side_candidates_required: bool,
    required_fields: Vec<String>,
    must_state_winner_loser_decisive_evidence_tradeoffs_restore_path: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolDocument {
    schema: String,
    status: String,
    protocol_identity_sha256: String,
    final_report: FinalProtocolFinalReport,
}

#[derive(Clone, Debug, Deserialize)]
struct CorpusAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    hidden_tasks_created: bool,
    scored_execution_started: bool,
    candidate_or_red_team_hidden_access_granted: bool,
    bound_final_manager_protocol: EvidenceDigest,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ScorecardSelectionIdentity {
    protected_verifier_identity_sha256: String,
    protected_controller_identity_sha256: String,
    candidate_self_selection_authorized: bool,
    candidate_weight_mutation_authorized: bool,
    candidate_self_scoring_authorized: bool,
    winner_selected: bool,
    selected_winner_artifact_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct ScorecardPareto {
    complete_two_candidate_frontier: bool,
    no_early_scalar_collapse: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct ScorecardDocument {
    schema: String,
    status: String,
    prepared: bool,
    scored_execution_completed: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_protected_corpus_commitment: EvidenceDigest,
    complete_two_candidate_two_mode_matrix: bool,
    pareto_frontier: ScorecardPareto,
    protected_selection_identity: ScorecardSelectionIdentity,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct SelectionIdentity {
    protected_verifier_identity_sha256: String,
    protected_controller_identity_sha256: String,
    protected_selection_namespace: String,
    candidate_self_selection_authorized: bool,
    candidate_weight_mutation_authorized: bool,
    candidate_self_scoring_authorized: bool,
    winner_selected: bool,
    selected_winner_artifact_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct SelectionRecoveryDocument {
    schema: String,
    status: String,
    prepared: bool,
    winner_selected: bool,
    artifact_move_performed: bool,
    artifact_delete_performed: bool,
    evictability_granted: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_protected_corpus_commitment: EvidenceDigest,
    bound_scorecard_adjudication: EvidenceDigest,
    protected_selection_identity: SelectionIdentity,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct SideBySideCandidateRow {
    candidate_artifact_id: String,
    required_field_commitment_sha256: String,
    all_required_fields_present_and_verified: bool,
    candidate_authored: bool,
    candidate_self_verified: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProtectedReportAuthorship {
    protected_report_controller_identity_sha256: String,
    protected_report_verifier_identity_sha256: String,
    report_self_authored_by_candidate: bool,
    report_self_verified_by_candidate: bool,
    scorecard_evidence_verified: bool,
    report_external_emission_requested: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct DecisionNarrativeReservation {
    required_decision_fields: Vec<String>,
    winner_value_present: bool,
    loser_value_present: bool,
    decisive_evidence_value_present: bool,
    tradeoffs_value_present: bool,
    restore_path_value_present: bool,
    winner_or_loser_selected_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    final_manager_protocol: SealedDocumentBinding,
    protected_corpus_commitment: SealedDocumentBinding,
    scorecard_adjudication: SealedDocumentBinding,
    selection_recovery_authority: SealedDocumentBinding,
    side_by_side_report_fields: Vec<String>,
    candidate_rows: Vec<SideBySideCandidateRow>,
    protected_report_authorship: ProtectedReportAuthorship,
    decision_narrative_reservation: DecisionNarrativeReservation,
    final_report_execution_requested: bool,
    winner_selection_requested: bool,
    external_report_emission_requested: bool,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    exact_protocol_corpus_scorecard_selection_recovery_chain_bound: bool,
    complete_two_candidate_side_by_side_report_fields_required: bool,
    protected_authorship_and_verification_required_without_candidate_self_authorship: bool,
    winner_loser_decisive_evidence_tradeoffs_restore_path_reservation_complete: bool,
    current_contract_never_selects_winner_executes_or_emits_report: bool,
    no_runtime_server_gpu_watcher_tps_or_tournament_authority: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: String,
    prepared: bool,
    winner_selected: bool,
    final_report_executed: bool,
    external_report_emitted: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_protected_corpus_commitment: EvidenceDigest,
    bound_scorecard_adjudication: EvidenceDigest,
    bound_selection_recovery_authority: EvidenceDigest,
    side_by_side_report_fields: Vec<String>,
    candidate_rows: Vec<SideBySideCandidateRow>,
    protected_report_authorship: ProtectedReportAuthorship,
    decision_narrative_reservation: DecisionNarrativeReservation,
    state_blockers: Vec<String>,
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
        return Err(format!("{label}.document_sha256 must be lowercase SHA-256"));
    }
    if sha256_json(&binding.document)? != binding.document_sha256 {
        return Err(format!(
            "{label}.document_sha256 does not bind embedded document"
        ));
    }
    let document_seal_sha256 =
        verify_sealed_object(&binding.document, &format!("{label}.document"))?;
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
        document_seal_sha256,
    })
}

fn exact_digest(
    expected: &EvidenceDigest,
    observed: &EvidenceDigest,
    label: &str,
) -> Result<(), String> {
    if expected.document_sha256 != observed.document_sha256
        || expected.document_seal_sha256 != observed.document_seal_sha256
    {
        return Err(format!("{label} must bind exact sealed upstream authority"));
    }
    Ok(())
}

fn exact_set(values: &[String], expected: &[&str], label: &str) -> Result<(), String> {
    let actual: BTreeSet<&str> = values.iter().map(String::as_str).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    if actual.len() != values.len() || actual != expected {
        return Err(format!("{label} must be exactly {:?}", expected));
    }
    Ok(())
}

fn validate_protocol(binding: &SealedDocumentBinding) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "final_manager_protocol")?;
    let document: FinalProtocolDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("final_manager_protocol.document has wrong grammar: {error}"))?;
    if document.schema != FINAL_PROTOCOL_SCHEMA
        || document.status != FINAL_PROTOCOL_STATUS
        || document.protocol_identity_sha256 != FINAL_PROTOCOL_IDENTITY
        || !document.final_report.side_by_side_candidates_required
        || !document
            .final_report
            .must_state_winner_loser_decisive_evidence_tradeoffs_restore_path
    {
        return Err(
            "final manager protocol must be expected prepared side-by-side final-report protocol"
                .into(),
        );
    }
    exact_set(
        &document.final_report.required_fields,
        &REQUIRED_REPORT_FIELDS,
        "final manager protocol final report fields",
    )?;
    Ok(digest)
}

fn validate_corpus(
    binding: &SealedDocumentBinding,
    protocol: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "protected_corpus_commitment")?;
    let document: CorpusAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| {
            format!("protected_corpus_commitment.document has wrong grammar: {error}")
        })?;
    if document.schema != CORPUS_SCHEMA
        || document.status != CORPUS_PREPARED_STATUS
        || !document.prepared
        || document.hidden_tasks_created
        || document.scored_execution_started
        || document.candidate_or_red_team_hidden_access_granted
    {
        return Err("protected corpus must stay prepared metadata-only and blind".into());
    }
    exact_digest(
        protocol,
        &document.bound_final_manager_protocol,
        "protected corpus",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "protected corpus authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "protected corpus execution_boundary",
    )?;
    Ok(digest)
}

fn validate_scorecard(
    binding: &SealedDocumentBinding,
    protocol: &EvidenceDigest,
    corpus: &EvidenceDigest,
) -> Result<(EvidenceDigest, ScorecardSelectionIdentity), String> {
    let digest = binding_digest(binding, "scorecard_adjudication")?;
    let document: ScorecardDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("scorecard_adjudication.document has wrong grammar: {error}"))?;
    if document.schema != SCORECARD_SCHEMA
        || document.status != SCORECARD_ADJUDICATED_STATUS
        || !document.prepared
        || !document.scored_execution_completed
        || !document.complete_two_candidate_two_mode_matrix
        || !document.pareto_frontier.complete_two_candidate_frontier
        || !document.pareto_frontier.no_early_scalar_collapse
    {
        return Err(
            "scorecard adjudication must be completed and preserve the two-candidate Pareto matrix"
                .into(),
        );
    }
    exact_digest(
        protocol,
        &document.bound_final_manager_protocol,
        "scorecard adjudication",
    )?;
    exact_digest(
        corpus,
        &document.bound_protected_corpus_commitment,
        "scorecard adjudication",
    )?;
    let identity = &document.protected_selection_identity;
    if !is_lower_sha256(&identity.protected_verifier_identity_sha256)
        || !is_lower_sha256(&identity.protected_controller_identity_sha256)
        || identity.candidate_self_selection_authorized
        || identity.candidate_weight_mutation_authorized
        || identity.candidate_self_scoring_authorized
        || identity.winner_selected
        || identity.selected_winner_artifact_id.is_some()
    {
        return Err("scorecard must be protected and not self-select a winner".into());
    }
    validate_authority_boundary(&document.authority_boundary, "scorecard authority_boundary")?;
    validate_execution_boundary(&document.execution_boundary, "scorecard execution_boundary")?;
    Ok((digest, identity.clone()))
}

fn validate_selection(
    binding: &SealedDocumentBinding,
    protocol: &EvidenceDigest,
    corpus: &EvidenceDigest,
    scorecard: &EvidenceDigest,
    scorecard_identity: &ScorecardSelectionIdentity,
) -> Result<(EvidenceDigest, SelectionIdentity), String> {
    let digest = binding_digest(binding, "selection_recovery_authority")?;
    let document: SelectionRecoveryDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| {
            format!("selection_recovery_authority.document has wrong grammar: {error}")
        })?;
    if document.schema != SELECTION_SCHEMA
        || (document.status != SELECTION_PREPARED_STATUS
            && document.status != SELECTION_REFUSED_STATUS)
        || document.winner_selected
        || document.artifact_move_performed
        || document.artifact_delete_performed
        || document.evictability_granted
    {
        return Err(
            "selection/recovery authority must not select or move/delete/evict anything".into(),
        );
    }
    exact_digest(
        protocol,
        &document.bound_final_manager_protocol,
        "selection/recovery authority",
    )?;
    exact_digest(
        corpus,
        &document.bound_protected_corpus_commitment,
        "selection/recovery authority",
    )?;
    exact_digest(
        scorecard,
        &document.bound_scorecard_adjudication,
        "selection/recovery authority",
    )?;
    let identity = &document.protected_selection_identity;
    if identity.protected_verifier_identity_sha256
        != scorecard_identity.protected_verifier_identity_sha256
        || identity.protected_controller_identity_sha256
            != scorecard_identity.protected_controller_identity_sha256
        || identity.candidate_self_selection_authorized
        || identity.candidate_weight_mutation_authorized
        || identity.candidate_self_scoring_authorized
        || identity.winner_selected
        || identity.selected_winner_artifact_id.is_some()
    {
        return Err(
            "selection/recovery protected identities must match scorecard and remain non-selecting"
                .into(),
        );
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "selection/recovery authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "selection/recovery execution_boundary",
    )?;
    Ok((digest, identity.clone()))
}

fn validate_candidate_rows(
    rows: &[SideBySideCandidateRow],
) -> Result<Vec<SideBySideCandidateRow>, String> {
    if rows.len() != 2 {
        return Err("final report requires exactly two side-by-side candidate rows".into());
    }
    let mut candidates = BTreeSet::new();
    for row in rows {
        if !is_lower_sha256(&row.required_field_commitment_sha256)
            || !row.all_required_fields_present_and_verified
            || row.candidate_authored
            || row.candidate_self_verified
            || !candidates.insert(row.candidate_artifact_id.as_str())
        {
            return Err(
                "candidate rows must be protected-authored/verified, complete, sealed, and unique"
                    .into(),
            );
        }
    }
    if candidates != CANDIDATES.iter().copied().collect() {
        return Err("candidate rows must cover exactly Q30 and Q80 fixed artifacts".into());
    }
    let mut rows = rows.to_vec();
    rows.sort_by(|left, right| left.candidate_artifact_id.cmp(&right.candidate_artifact_id));
    Ok(rows)
}

fn validate_authorship(
    authorship: &ProtectedReportAuthorship,
    identity: &SelectionIdentity,
) -> Result<(), String> {
    if authorship.protected_report_controller_identity_sha256
        != identity.protected_controller_identity_sha256
        || authorship.protected_report_verifier_identity_sha256
            != identity.protected_verifier_identity_sha256
        || authorship.report_self_authored_by_candidate
        || authorship.report_self_verified_by_candidate
        || !authorship.scorecard_evidence_verified
        || authorship.report_external_emission_requested
    {
        return Err("final report must be protected-authored/verified, scorecard-verified, and not externally emitted".into());
    }
    Ok(())
}

fn validate_decision_reservation(reservation: &DecisionNarrativeReservation) -> Result<(), String> {
    exact_set(
        &reservation.required_decision_fields,
        &DECISION_FIELDS,
        "decision narrative required fields",
    )?;
    if reservation.winner_value_present
        || reservation.loser_value_present
        || reservation.decisive_evidence_value_present
        || reservation.tradeoffs_value_present
        || reservation.restore_path_value_present
        || reservation.winner_or_loser_selected_by_this_contract
    {
        return Err("decision narrative may reserve required fields but cannot author a winner/loser or final values".into());
    }
    Ok(())
}

fn build_report(input: Input) -> Result<Report, String> {
    if input.schema != INPUT_SCHEMA {
        return Err(format!(
            "input.schema must be {INPUT_SCHEMA:?}, observed {:?}",
            input.schema
        ));
    }
    if !is_lower_sha256(&input.seal_sha256) {
        return Err("input.seal_sha256 must be lowercase SHA-256".into());
    }
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;
    let protocol = validate_protocol(&input.final_manager_protocol)?;
    let corpus = validate_corpus(&input.protected_corpus_commitment, &protocol)?;
    let (scorecard, scorecard_identity) =
        validate_scorecard(&input.scorecard_adjudication, &protocol, &corpus)?;
    let (selection, selection_identity) = validate_selection(
        &input.selection_recovery_authority,
        &protocol,
        &corpus,
        &scorecard,
        &scorecard_identity,
    )?;
    exact_set(
        &input.side_by_side_report_fields,
        &REQUIRED_REPORT_FIELDS,
        "side_by_side_report_fields",
    )?;
    let candidate_rows = validate_candidate_rows(&input.candidate_rows)?;
    validate_authorship(&input.protected_report_authorship, &selection_identity)?;
    validate_decision_reservation(&input.decision_narrative_reservation)?;
    let mut state_blockers = Vec::new();
    if !matches!(
        input.selection_recovery_authority.document["status"].as_str(),
        Some(SELECTION_PREPARED_STATUS)
    ) {
        state_blockers.push("selection_recovery_authority_not_prepared".into());
    }
    if input.final_report_execution_requested {
        state_blockers.push("final_report_execution_requested_outside_this_contract".into());
    }
    if input.winner_selection_requested {
        state_blockers.push("winner_selection_requested_outside_this_contract".into());
    }
    if input.external_report_emission_requested {
        state_blockers.push("external_report_emission_requested_outside_this_contract".into());
    }
    state_blockers.sort();
    state_blockers.dedup();
    let prepared = state_blockers.is_empty();
    let status = if prepared {
        PREPARED_STATUS
    } else {
        REFUSED_STATUS
    }
    .into();
    let focused_checks = FocusedChecks {
        exact_protocol_corpus_scorecard_selection_recovery_chain_bound: true,
        complete_two_candidate_side_by_side_report_fields_required: true,
        protected_authorship_and_verification_required_without_candidate_self_authorship: true,
        winner_loser_decisive_evidence_tradeoffs_restore_path_reservation_complete: true,
        current_contract_never_selects_winner_executes_or_emits_report: true,
        no_runtime_server_gpu_watcher_tps_or_tournament_authority: true,
        execution_boundary_cpu_only: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        prepared,
        winner_selected: false,
        final_report_executed: false,
        external_report_emitted: false,
        bound_final_manager_protocol: protocol,
        bound_protected_corpus_commitment: corpus,
        bound_scorecard_adjudication: scorecard,
        bound_selection_recovery_authority: selection,
        side_by_side_report_fields: input.side_by_side_report_fields,
        candidate_rows,
        protected_report_authorship: input.protected_report_authorship,
        decision_narrative_reservation: input.decision_narrative_reservation,
        state_blockers,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only final-report contract, not an external report publisher or winner selector.",
            "It requires the full protected final-protocol, blind corpus, completed scorecard/Pareto adjudication, and non-selecting selection/recovery authority chain.",
            "The side-by-side Q30/Q80 report must reserve all 21 fixed fields and the required winner, loser, decisive-evidence, tradeoffs, and restore-path decision fields.",
            "Rows and narrative are protected-authored/verified against sealed scorecard evidence; candidate-authored, self-verified, partial, or unverified reports are refused.",
            "A prepared contract still carries no winner/loser values, does not select a manager, execute a final report, or emit anything externally.",
            "No runtime, server, watcher, GPU, lease, HCLI/TPS/TG, tournament, scorecard mutation, or artifact action is performed.",
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
            .map_err(|error| format!("final report contract cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("final report contract cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_manager_tournament_final_report_contract --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        build_report(input).map_err(|error| format!("final report validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_manager_tournament_final_report_contract: {error}");
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

    fn digest(binding: &SealedDocumentBinding, schema: &str, status: &str) -> Value {
        json!({
            "path": binding.path,
            "document_schema": schema,
            "document_status": status,
            "document_sha256": binding.document_sha256,
            "document_seal_sha256": binding.document["seal_sha256"],
        })
    }

    fn protocol_binding() -> SealedDocumentBinding {
        binding(
            "/sealed/protocol.json",
            seal_value(json!({
                "schema": FINAL_PROTOCOL_SCHEMA,
                "status": FINAL_PROTOCOL_STATUS,
                "protocol_identity_sha256": FINAL_PROTOCOL_IDENTITY,
                "final_report": {
                    "side_by_side_candidates_required": true,
                    "required_fields": REQUIRED_REPORT_FIELDS,
                    "must_state_winner_loser_decisive_evidence_tradeoffs_restore_path": true,
                },
            }))
            .unwrap(),
        )
    }

    fn corpus_binding(protocol: &SealedDocumentBinding) -> SealedDocumentBinding {
        binding(
            "/sealed/corpus.json",
            seal_value(json!({
                "schema": CORPUS_SCHEMA,
                "status": CORPUS_PREPARED_STATUS,
                "prepared": true,
                "hidden_tasks_created": false,
                "scored_execution_started": false,
                "candidate_or_red_team_hidden_access_granted": false,
                "bound_final_manager_protocol": digest(protocol, FINAL_PROTOCOL_SCHEMA, FINAL_PROTOCOL_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn scorecard_identity() -> ScorecardSelectionIdentity {
        ScorecardSelectionIdentity {
            protected_verifier_identity_sha256: sha('1'),
            protected_controller_identity_sha256: sha('2'),
            candidate_self_selection_authorized: false,
            candidate_weight_mutation_authorized: false,
            candidate_self_scoring_authorized: false,
            winner_selected: false,
            selected_winner_artifact_id: None,
        }
    }

    fn scorecard_binding(
        protocol: &SealedDocumentBinding,
        corpus: &SealedDocumentBinding,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/scorecard.json",
            seal_value(json!({
                "schema": SCORECARD_SCHEMA,
                "status": SCORECARD_ADJUDICATED_STATUS,
                "prepared": true,
                "scored_execution_completed": true,
                "bound_final_manager_protocol": digest(protocol, FINAL_PROTOCOL_SCHEMA, FINAL_PROTOCOL_STATUS),
                "bound_protected_corpus_commitment": digest(corpus, CORPUS_SCHEMA, CORPUS_PREPARED_STATUS),
                "complete_two_candidate_two_mode_matrix": true,
                "pareto_frontier": {"complete_two_candidate_frontier": true, "no_early_scalar_collapse": true},
                "protected_selection_identity": scorecard_identity(),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn selection_identity() -> SelectionIdentity {
        SelectionIdentity {
            protected_verifier_identity_sha256: sha('1'),
            protected_controller_identity_sha256: sha('2'),
            protected_selection_namespace:
                "sealed://protected-tournament/selection/controller-verifier".into(),
            candidate_self_selection_authorized: false,
            candidate_weight_mutation_authorized: false,
            candidate_self_scoring_authorized: false,
            winner_selected: false,
            selected_winner_artifact_id: None,
        }
    }

    fn selection_binding(
        protocol: &SealedDocumentBinding,
        corpus: &SealedDocumentBinding,
        scorecard: &SealedDocumentBinding,
        prepared: bool,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/selection.json",
            seal_value(json!({
                "schema": SELECTION_SCHEMA,
                "status": if prepared { SELECTION_PREPARED_STATUS } else { SELECTION_REFUSED_STATUS },
                "prepared": prepared,
                "winner_selected": false,
                "artifact_move_performed": false,
                "artifact_delete_performed": false,
                "evictability_granted": false,
                "bound_final_manager_protocol": digest(protocol, FINAL_PROTOCOL_SCHEMA, FINAL_PROTOCOL_STATUS),
                "bound_protected_corpus_commitment": digest(corpus, CORPUS_SCHEMA, CORPUS_PREPARED_STATUS),
                "bound_scorecard_adjudication": digest(scorecard, SCORECARD_SCHEMA, SCORECARD_ADJUDICATED_STATUS),
                "protected_selection_identity": selection_identity(),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn rows() -> Vec<SideBySideCandidateRow> {
        CANDIDATES
            .iter()
            .enumerate()
            .map(|(index, candidate)| SideBySideCandidateRow {
                candidate_artifact_id: (*candidate).into(),
                required_field_commitment_sha256: sha(if index == 0 { '3' } else { '4' }),
                all_required_fields_present_and_verified: true,
                candidate_authored: false,
                candidate_self_verified: false,
            })
            .collect()
    }

    fn authorship() -> ProtectedReportAuthorship {
        ProtectedReportAuthorship {
            protected_report_controller_identity_sha256: sha('2'),
            protected_report_verifier_identity_sha256: sha('1'),
            report_self_authored_by_candidate: false,
            report_self_verified_by_candidate: false,
            scorecard_evidence_verified: true,
            report_external_emission_requested: false,
        }
    }

    fn decision_reservation() -> DecisionNarrativeReservation {
        DecisionNarrativeReservation {
            required_decision_fields: DECISION_FIELDS
                .iter()
                .map(|field| (*field).into())
                .collect(),
            winner_value_present: false,
            loser_value_present: false,
            decisive_evidence_value_present: false,
            tradeoffs_value_present: false,
            restore_path_value_present: false,
            winner_or_loser_selected_by_this_contract: false,
        }
    }

    fn input_fixture(selection_prepared: bool) -> Input {
        let final_manager_protocol = protocol_binding();
        let protected_corpus_commitment = corpus_binding(&final_manager_protocol);
        let scorecard_adjudication =
            scorecard_binding(&final_manager_protocol, &protected_corpus_commitment);
        let selection_recovery_authority = selection_binding(
            &final_manager_protocol,
            &protected_corpus_commitment,
            &scorecard_adjudication,
            selection_prepared,
        );
        Input {
            schema: INPUT_SCHEMA.into(),
            final_manager_protocol,
            protected_corpus_commitment,
            scorecard_adjudication,
            selection_recovery_authority,
            side_by_side_report_fields: REQUIRED_REPORT_FIELDS
                .iter()
                .map(|field| (*field).into())
                .collect(),
            candidate_rows: rows(),
            protected_report_authorship: authorship(),
            decision_narrative_reservation: decision_reservation(),
            final_report_execution_requested: false,
            winner_selection_requested: false,
            external_report_emission_requested: false,
            authority_boundary: authority_boundary(),
            execution_boundary: execution_boundary(),
            seal_sha256: sha('a'),
        }
    }

    fn seal_input(input: &Input) -> Value {
        let mut value = serde_json::to_value(input).unwrap();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal_value(value).unwrap()
    }

    #[test]
    fn complete_protected_two_candidate_report_contract_can_only_prepare_not_select_or_emit() {
        let report = build_report(input_fixture(true)).unwrap();
        assert_eq!(report.status, PREPARED_STATUS);
        assert!(report.prepared);
        assert!(!report.winner_selected);
        assert!(!report.final_report_executed);
        assert!(!report.external_report_emitted);
        assert_eq!(report.side_by_side_report_fields.len(), 21);
        assert_eq!(report.candidate_rows.len(), 2);
    }

    #[test]
    fn current_unprepared_selection_authority_refuses() {
        let report = build_report(input_fixture(false)).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(report
            .state_blockers
            .iter()
            .any(|blocker| blocker.contains("selection")));
    }

    #[test]
    fn partial_or_duplicate_side_by_side_report_fields_and_rows_are_rejected() {
        let mut partial = input_fixture(true);
        partial.side_by_side_report_fields.pop();
        assert!(build_report(partial).is_err());

        let mut duplicate = input_fixture(true);
        duplicate.candidate_rows[1].candidate_artifact_id =
            duplicate.candidate_rows[0].candidate_artifact_id.clone();
        assert!(build_report(duplicate).is_err());
    }

    #[test]
    fn self_authored_self_verified_or_unverified_reports_are_rejected() {
        let mut self_authored = input_fixture(true);
        self_authored
            .protected_report_authorship
            .report_self_authored_by_candidate = true;
        assert!(build_report(self_authored).is_err());

        let mut self_verified = input_fixture(true);
        self_verified.candidate_rows[0].candidate_self_verified = true;
        assert!(build_report(self_verified).is_err());

        let mut unverified = input_fixture(true);
        unverified
            .protected_report_authorship
            .scorecard_evidence_verified = false;
        assert!(build_report(unverified).is_err());
    }

    #[test]
    fn winner_loser_decisive_evidence_tradeoffs_and_restore_path_are_required_but_not_authored() {
        let mut missing = input_fixture(true);
        missing
            .decision_narrative_reservation
            .required_decision_fields
            .pop();
        assert!(build_report(missing).is_err());

        let mut winner = input_fixture(true);
        winner.decision_narrative_reservation.winner_value_present = true;
        assert!(build_report(winner).is_err());
    }

    #[test]
    fn report_execution_selection_and_external_emission_requests_are_refused() {
        for request in [
            |input: &mut Input| input.final_report_execution_requested = true,
            |input: &mut Input| input.winner_selection_requested = true,
            |input: &mut Input| input.external_report_emission_requested = true,
        ] {
            let mut input = input_fixture(true);
            request(&mut input);
            let report = build_report(input).unwrap();
            assert_eq!(report.status, REFUSED_STATUS);
            assert!(!report.prepared);
            assert!(!report.winner_selected);
            assert!(!report.final_report_executed);
            assert!(!report.external_report_emitted);
        }
    }

    #[test]
    fn authority_chain_self_selection_and_runtime_boundaries_are_fail_closed() {
        let mut bad_chain = input_fixture(true);
        bad_chain.selection_recovery_authority.document["bound_scorecard_adjudication"]
            ["document_sha256"] = Value::String(sha('0'));
        let document = bad_chain.selection_recovery_authority.document.clone();
        bad_chain.selection_recovery_authority.document = seal_value(document).unwrap();
        bad_chain.selection_recovery_authority.document_sha256 =
            sha256_json(&bad_chain.selection_recovery_authority.document).unwrap();
        assert!(build_report(bad_chain).is_err());

        let mut self_select = input_fixture(true);
        self_select.scorecard_adjudication.document["protected_selection_identity"]
            ["candidate_self_selection_authorized"] = Value::Bool(true);
        let document = self_select.scorecard_adjudication.document.clone();
        self_select.scorecard_adjudication.document = seal_value(document).unwrap();
        self_select.scorecard_adjudication.document_sha256 =
            sha256_json(&self_select.scorecard_adjudication.document).unwrap();
        assert!(build_report(self_select).is_err());

        let mut runtime = input_fixture(false);
        runtime.execution_boundary.runtime_watcher_or_server_started = true;
        assert!(build_report(runtime).is_err());
    }

    #[test]
    fn sealed_input_output_create_new_and_unknown_candidate_output_fields_are_enforced() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("input.json");
        let output_path = directory.path().join("out.json");
        let sealed = seal_input(&input_fixture(false));
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        run(Args {
            input: input_path,
            out: output_path.clone(),
        })
        .unwrap();
        let output: Value = serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
        verify_sealed_object(&output, "output").unwrap();
        assert_eq!(output["status"], Value::String(REFUSED_STATUS.into()));
        assert!(write_report_create_new(
            &output_path,
            &build_report(input_fixture(false)).unwrap()
        )
        .is_err());

        let unknown_path = directory.path().join("unknown.json");
        let unknown_out = directory.path().join("unknown-out.json");
        let mut unknown = seal_input(&input_fixture(true));
        unknown["candidate_rows"][0]["candidate_score"] = Value::String("not allowed".into());
        let unknown = seal_value(unknown).unwrap();
        fs::write(&unknown_path, serde_json::to_vec_pretty(&unknown).unwrap()).unwrap();
        assert!(run(Args {
            input: unknown_path,
            out: unknown_out,
        })
        .is_err());
    }
}
