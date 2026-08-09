//! Sealed protected final-manager selection and alternate-recovery contract.
//!
//! This CPU-only authority accepts only sealed final-protocol, TG3/freeze,
//! protected-corpus, and scorecard-adjudication evidence.  It validates the
//! exact two-candidate/two-mode matrix and the required Pareto/protected
//! selection/recovery reservations, but it never chooses a winner, seals a
//! winner, moves/cold-stores/deletes an artifact, grants evictability, or starts
//! a runtime/tournament.  A prepared result is a later-controller reservation,
//! not a selection.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_selection_and_recovery_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_selection_and_recovery_contract.v1";
const FINAL_PROTOCOL_SCHEMA: &str = "hawking.ascension.final_manager_tournament_protocol.v1";
const FINAL_PROTOCOL_STATUS: &str = "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED";
const FINAL_PROTOCOL_IDENTITY: &str =
    "8e3684af0b7de53690a9c88ce0d52b0cae019e0d798bc2748b5b1556211facf8";
const TG3_FREEZE_SCHEMA: &str =
    "hawking.ascension.paired_cognition_tg3_freeze_final_comparison_authority.v1";
const TG3_FREEZE_PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_TG3_FROZEN_FINAL_MANAGER_COMPARISON_RESERVED";
const TG3_FREEZE_REFUSED_STATUS: &str =
    "REFUSED_PAIRED_COGNITION_TG3_FROZEN_FINAL_MANAGER_COMPARISON_HARD_GATES_INCOMPLETE";
const CORPUS_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_task_corpus_commitment_authority.v1";
const CORPUS_PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PROTECTED_REAL_HAWKING_TASK_CORPUS_METADATA_COMMITTED_NO_HIDDEN_TASKS_OR_SCORED_EXECUTION";
const SCORECARD_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_scorecard_adjudication.v1";
const SCORECARD_PENDING_STATUS: &str = "PENDING_OR_UNADJUDICATED_PROTECTED_SCORECARD_MATRIX";
const SCORECARD_ADJUDICATED_STATUS: &str =
    "ADJUDICATED_COMPLETE_TWO_CANDIDATE_TWO_MODE_PARETO_MATRIX_NO_WINNER";
const PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PROTECTED_SELECTION_AND_RECOVERY_CONTRACT_COMPLETE_NO_WINNER_SELECTED";
const REFUSED_STATUS: &str =
    "REFUSED_PROTECTED_SELECTION_AND_RECOVERY_EVIDENCE_OR_ACTION_REQUEST_INCOMPLETE";

const CANDIDATES: [&str; 2] = [
    "Qwen30-Gravity-Manager-Artifact",
    "Qwen80-Gravity-Manager-Artifact",
];
const MODES: [&str; 2] = ["SOLO_MANAGER", "MANAGER_AS_ORCHESTRATOR"];
const WINNER_SEAL_COMPONENTS: [&str; 10] = [
    "model_source",
    "gravity_artifact",
    "runtime",
    "kernel_set",
    "agent_os_configuration",
    "context_kv_policy",
    "benchmark_matrix",
    "capability_matrix",
    "tournament_evidence",
    "rollback_state",
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
struct FinalProtocolSelection {
    candidates_cannot_self_grade: bool,
    candidates_cannot_change_weights_or_hidden_tests: bool,
    candidates_cannot_promote_self_or_invalidate_opponent: bool,
    only_protected_controller_or_human_may_select_manager: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolDocument {
    schema: String,
    status: String,
    protocol_identity_sha256: String,
    protected_selection: FinalProtocolSelection,
}

#[derive(Clone, Debug, Deserialize)]
struct Tg3FreezeAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    tournament_active: bool,
    scored_task_execution_active: bool,
    winner_selected: bool,
    bound_final_manager_protocol: EvidenceDigest,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
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

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct MatrixCell {
    candidate_artifact_id: String,
    evaluation_mode: String,
    protected_scorecard_seal_sha256: String,
    all_task_evidence_verified: bool,
    all_hard_gates_retained: bool,
    adversarial_review_adjudicated: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct ParetoFrontier {
    complete_two_candidate_frontier: bool,
    no_early_scalar_collapse: bool,
    protected_frontier_commitment_sha256: String,
    candidate_artifact_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct ProtectedSelectionIdentity {
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
struct ScorecardAdjudicationDocument {
    schema: String,
    status: String,
    prepared: bool,
    scored_execution_completed: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_tg3_freeze_authority: EvidenceDigest,
    bound_protected_corpus_commitment: EvidenceDigest,
    matrix_cells: Vec<MatrixCell>,
    complete_two_candidate_two_mode_matrix: bool,
    pareto_frontier: ParetoFrontier,
    protected_selection_identity: ProtectedSelectionIdentity,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct WinnerImmutableSealReservation {
    required_component_labels: Vec<String>,
    immutable_revision_zero_required: bool,
    winner_seal_must_cover_all_components_before_any_evictability: bool,
    winner_immutable_seal_sha256: Option<String>,
    winner_selected_by_this_contract: bool,
    selected_winner_artifact_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct LoserRecoveryReservation {
    loser_seal_before_any_evictability: bool,
    cold_store_or_offload_required: bool,
    hash_verify_required: bool,
    restore_test_required: bool,
    one_command_recovery_required: bool,
    no_delete_before_restore_proof: bool,
    artifact_move_performed_by_this_contract: bool,
    artifact_delete_performed_by_this_contract: bool,
    evictability_granted_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    final_manager_protocol: SealedDocumentBinding,
    tg3_freeze_authority: SealedDocumentBinding,
    protected_corpus_commitment: SealedDocumentBinding,
    scorecard_adjudication: SealedDocumentBinding,
    winner_immutable_seal_reservation: WinnerImmutableSealReservation,
    loser_recovery_reservation: LoserRecoveryReservation,
    selection_requested: bool,
    artifact_move_requested: bool,
    artifact_delete_requested: bool,
    evictability_requested: bool,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    exact_final_protocol_tg3_freeze_corpus_and_scorecard_chain_bound: bool,
    complete_two_candidate_two_mode_verified_evidence_matrix_required: bool,
    pareto_frontier_complete_before_protected_selection: bool,
    protected_verifier_and_controller_identity_required_without_candidate_self_selection: bool,
    winner_immutable_seal_component_contract_reserved: bool,
    loser_seal_cold_store_hash_verify_restore_one_command_recovery_before_evictability_reserved:
        bool,
    current_contract_never_selects_winner_or_moves_deletes_or_evicts_artifacts: bool,
    no_runtime_server_gpu_watcher_tps_or_tournament_authority: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: String,
    prepared: bool,
    winner_selected: bool,
    artifact_move_performed: bool,
    artifact_delete_performed: bool,
    evictability_granted: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_tg3_freeze_authority: EvidenceDigest,
    bound_protected_corpus_commitment: EvidenceDigest,
    bound_scorecard_adjudication: EvidenceDigest,
    protected_selection_identity: ProtectedSelectionIdentity,
    winner_immutable_seal_reservation: WinnerImmutableSealReservation,
    loser_recovery_reservation: LoserRecoveryReservation,
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
        return Err(format!(
            "{label} must bind the exact sealed upstream authority"
        ));
    }
    Ok(())
}

fn exact_set(values: &[String], expected: &[&str], label: &str) -> Result<(), String> {
    let actual: BTreeSet<&str> = values.iter().map(String::as_str).collect();
    let expected: BTreeSet<&str> = expected.iter().copied().collect();
    if actual.len() != values.len() || actual != expected {
        return Err(format!("{label} must contain exactly {:?}", expected));
    }
    Ok(())
}

fn validate_final_protocol(binding: &SealedDocumentBinding) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "final_manager_protocol")?;
    let document: FinalProtocolDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("final_manager_protocol.document has wrong grammar: {error}"))?;
    if document.schema != FINAL_PROTOCOL_SCHEMA
        || document.status != FINAL_PROTOCOL_STATUS
        || document.protocol_identity_sha256 != FINAL_PROTOCOL_IDENTITY
    {
        return Err("final manager protocol identity is not the expected prepared protocol".into());
    }
    let selection = &document.protected_selection;
    if !selection.candidates_cannot_self_grade
        || !selection.candidates_cannot_change_weights_or_hidden_tests
        || !selection.candidates_cannot_promote_self_or_invalidate_opponent
        || !selection.only_protected_controller_or_human_may_select_manager
    {
        return Err("final manager protocol must retain protected non-self-selection".into());
    }
    Ok(digest)
}

fn validate_tg3_freeze(
    binding: &SealedDocumentBinding,
    protocol: &EvidenceDigest,
) -> Result<(EvidenceDigest, bool), String> {
    let digest = binding_digest(binding, "tg3_freeze_authority")?;
    let document: Tg3FreezeAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("tg3_freeze_authority.document has wrong grammar: {error}"))?;
    if document.schema != TG3_FREEZE_SCHEMA
        || (document.status != TG3_FREEZE_PREPARED_STATUS
            && document.status != TG3_FREEZE_REFUSED_STATUS)
        || document.tournament_active
        || document.scored_task_execution_active
        || document.winner_selected
    {
        return Err(
            "TG3/freeze authority must remain a non-running, non-selecting preparation authority"
                .into(),
        );
    }
    exact_digest(
        protocol,
        &document.bound_final_manager_protocol,
        "tg3_freeze_authority",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "tg3_freeze_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "tg3_freeze_authority.execution_boundary",
    )?;
    Ok((
        digest,
        document.prepared && document.status == TG3_FREEZE_PREPARED_STATUS,
    ))
}

fn validate_corpus(
    binding: &SealedDocumentBinding,
    protocol: &EvidenceDigest,
) -> Result<(EvidenceDigest, bool), String> {
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
        return Err(
            "protected corpus must remain a prepared metadata-only blind commitment".into(),
        );
    }
    exact_digest(
        protocol,
        &document.bound_final_manager_protocol,
        "protected_corpus_commitment",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "protected_corpus_commitment.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "protected_corpus_commitment.execution_boundary",
    )?;
    Ok((digest, true))
}

fn validate_matrix(cells: &[MatrixCell]) -> Result<(), String> {
    if cells.len() != 4 {
        return Err("scorecard adjudication must contain all four candidate/mode cells".into());
    }
    let mut keys = BTreeSet::new();
    for cell in cells {
        if !is_lower_sha256(&cell.protected_scorecard_seal_sha256)
            || !cell.all_task_evidence_verified
            || !cell.all_hard_gates_retained
            || !cell.adversarial_review_adjudicated
        {
            return Err("each scorecard cell must have a sealed verified hard-gate/adversarial adjudication".into());
        }
        if !CANDIDATES.contains(&cell.candidate_artifact_id.as_str())
            || !MODES.contains(&cell.evaluation_mode.as_str())
            || !keys.insert((
                cell.candidate_artifact_id.as_str(),
                cell.evaluation_mode.as_str(),
            ))
        {
            return Err("scorecard cells must be unique fixed candidate/mode pairs".into());
        }
    }
    let expected = CANDIDATES
        .iter()
        .flat_map(|candidate| MODES.iter().map(move |mode| (*candidate, *mode)))
        .collect::<BTreeSet<_>>();
    if keys != expected {
        return Err("scorecard cells must exactly cover Q30/Q80 x SOLO/ORCHESTRATOR".into());
    }
    Ok(())
}

fn validate_pareto(frontier: &ParetoFrontier) -> Result<(), String> {
    if !frontier.complete_two_candidate_frontier
        || !frontier.no_early_scalar_collapse
        || !is_lower_sha256(&frontier.protected_frontier_commitment_sha256)
    {
        return Err("Pareto frontier must be complete, non-scalar-collapsed, and sealed".into());
    }
    exact_set(
        &frontier.candidate_artifact_ids,
        &CANDIDATES,
        "Pareto frontier candidate_artifact_ids",
    )
}

fn validate_scorecard(
    binding: &SealedDocumentBinding,
    protocol: &EvidenceDigest,
    freeze: &EvidenceDigest,
    corpus: &EvidenceDigest,
) -> Result<(EvidenceDigest, ProtectedSelectionIdentity, bool), String> {
    let digest = binding_digest(binding, "scorecard_adjudication")?;
    let document: ScorecardAdjudicationDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("scorecard_adjudication.document has wrong grammar: {error}"))?;
    if document.schema != SCORECARD_SCHEMA
        || (document.status != SCORECARD_PENDING_STATUS
            && document.status != SCORECARD_ADJUDICATED_STATUS)
    {
        return Err("scorecard adjudication has an unrecognized schema/status".into());
    }
    exact_digest(
        protocol,
        &document.bound_final_manager_protocol,
        "scorecard_adjudication",
    )?;
    exact_digest(
        freeze,
        &document.bound_tg3_freeze_authority,
        "scorecard_adjudication",
    )?;
    exact_digest(
        corpus,
        &document.bound_protected_corpus_commitment,
        "scorecard_adjudication",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "scorecard_adjudication.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "scorecard_adjudication.execution_boundary",
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
        return Err("scorecard adjudication must bind protected identities without a candidate/winner selection".into());
    }
    require_sealed_namespace(
        &identity.protected_selection_namespace,
        "protected_selection_identity.protected_selection_namespace",
    )?;
    if !identity
        .protected_selection_namespace
        .starts_with("sealed://protected-tournament/selection/")
    {
        return Err("protected selection namespace must be controller/verifier protected".into());
    }
    let complete = document.status == SCORECARD_ADJUDICATED_STATUS
        && document.prepared
        && document.scored_execution_completed
        && document.complete_two_candidate_two_mode_matrix;
    if complete {
        validate_matrix(&document.matrix_cells)?;
        validate_pareto(&document.pareto_frontier)?;
    } else if document.status == SCORECARD_PENDING_STATUS
        && (!document.prepared
            && !document.scored_execution_completed
            && !document.complete_two_candidate_two_mode_matrix)
    {
        // A current pending document is truthful evidence that selection must
        // refuse, not malformed input.
    } else {
        return Err(
            "scorecard adjudication state flags do not match pending/complete status".into(),
        );
    }
    Ok((digest, identity.clone(), complete))
}

fn validate_winner_reservation(reservation: &WinnerImmutableSealReservation) -> Result<(), String> {
    exact_set(
        &reservation.required_component_labels,
        &WINNER_SEAL_COMPONENTS,
        "winner immutable seal required components",
    )?;
    if !reservation.immutable_revision_zero_required
        || !reservation.winner_seal_must_cover_all_components_before_any_evictability
        || reservation.winner_immutable_seal_sha256.is_some()
        || reservation.winner_selected_by_this_contract
        || reservation.selected_winner_artifact_id.is_some()
    {
        return Err(
            "winner seal reservation must be complete but cannot create/select an actual winner"
                .into(),
        );
    }
    Ok(())
}

fn validate_loser_reservation(reservation: &LoserRecoveryReservation) -> Result<(), String> {
    if !reservation.loser_seal_before_any_evictability
        || !reservation.cold_store_or_offload_required
        || !reservation.hash_verify_required
        || !reservation.restore_test_required
        || !reservation.one_command_recovery_required
        || !reservation.no_delete_before_restore_proof
        || reservation.artifact_move_performed_by_this_contract
        || reservation.artifact_delete_performed_by_this_contract
        || reservation.evictability_granted_by_this_contract
    {
        return Err("loser recovery reservation must require seal→cold-store→hash→restore→one-command recovery before any evictability".into());
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
    let protocol = validate_final_protocol(&input.final_manager_protocol)?;
    let (freeze, freeze_ready) = validate_tg3_freeze(&input.tg3_freeze_authority, &protocol)?;
    let (corpus, corpus_ready) = validate_corpus(&input.protected_corpus_commitment, &protocol)?;
    let (scorecard, protected_selection_identity, scorecard_ready) =
        validate_scorecard(&input.scorecard_adjudication, &protocol, &freeze, &corpus)?;
    validate_winner_reservation(&input.winner_immutable_seal_reservation)?;
    validate_loser_reservation(&input.loser_recovery_reservation)?;
    let mut state_blockers = Vec::new();
    if !freeze_ready {
        state_blockers.push("tg3_freeze_authority_not_prepared".into());
    }
    if !corpus_ready {
        state_blockers.push("protected_corpus_not_prepared".into());
    }
    if !scorecard_ready {
        state_blockers.push(
            "complete_two_candidate_two_mode_scorecard_pareto_adjudication_not_present".into(),
        );
    }
    if input.selection_requested {
        state_blockers.push("selection_requested_outside_this_contract".into());
    }
    if input.artifact_move_requested {
        state_blockers.push("artifact_move_requested_outside_this_contract".into());
    }
    if input.artifact_delete_requested {
        state_blockers.push("artifact_delete_requested_outside_this_contract".into());
    }
    if input.evictability_requested {
        state_blockers.push("evictability_requested_outside_this_contract".into());
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
        exact_final_protocol_tg3_freeze_corpus_and_scorecard_chain_bound: true,
        complete_two_candidate_two_mode_verified_evidence_matrix_required: true,
        pareto_frontier_complete_before_protected_selection: true,
        protected_verifier_and_controller_identity_required_without_candidate_self_selection: true,
        winner_immutable_seal_component_contract_reserved: true,
        loser_seal_cold_store_hash_verify_restore_one_command_recovery_before_evictability_reserved: true,
        current_contract_never_selects_winner_or_moves_deletes_or_evicts_artifacts: true,
        no_runtime_server_gpu_watcher_tps_or_tournament_authority: true,
        execution_boundary_cpu_only: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        prepared,
        winner_selected: false,
        artifact_move_performed: false,
        artifact_delete_performed: false,
        evictability_granted: false,
        bound_final_manager_protocol: protocol,
        bound_tg3_freeze_authority: freeze,
        bound_protected_corpus_commitment: corpus,
        bound_scorecard_adjudication: scorecard,
        protected_selection_identity,
        winner_immutable_seal_reservation: input.winner_immutable_seal_reservation,
        loser_recovery_reservation: input.loser_recovery_reservation,
        state_blockers,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only protected selection/recovery contract, not a final-manager selector or artifact mover.",
            "It requires exact final-protocol, TG3/freeze, blind-corpus, and scorecard-adjudication bindings before any later protected selection can be considered.",
            "The complete evidence matrix is exactly Q30/Q80 across SOLO_MANAGER and MANAGER_AS_ORCHESTRATOR, with verified task evidence, retained hard gates, and adjudicated read-only adversarial review in every cell.",
            "A sealed two-candidate Pareto frontier must exist before protected verifier/controller selection; candidates cannot self-grade, self-select, mutate weights, promote themselves, or invalidate an opponent.",
            "A future winner requires an immutable revision-zero seal covering source, artifact, runtime, kernels, Agent OS, context/KV, benchmark, capability, tournament, and rollback evidence before any evictability.",
            "A future loser must be sealed, cold-stored/offloaded, hash-verified, restore-tested, and proven one-command recoverable before any evictability; deletion before proof is prohibited.",
            "This contract performs none of those future actions and never launches runtime, GPU, server/watcher, TPS/TG, or tournament work.",
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
            .map_err(|error| format!("selection/recovery report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("selection/recovery report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_manager_tournament_protected_selection_and_recovery_contract --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        .map_err(|error| format!("selection/recovery validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!(
            "ascension_manager_tournament_protected_selection_and_recovery_contract: {error}"
        );
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
            "/sealed/final-manager-protocol.json",
            seal_value(json!({
                "schema": FINAL_PROTOCOL_SCHEMA,
                "status": FINAL_PROTOCOL_STATUS,
                "protocol_identity_sha256": FINAL_PROTOCOL_IDENTITY,
                "protected_selection": {
                    "candidates_cannot_self_grade": true,
                    "candidates_cannot_change_weights_or_hidden_tests": true,
                    "candidates_cannot_promote_self_or_invalidate_opponent": true,
                    "only_protected_controller_or_human_may_select_manager": true,
                },
            }))
            .unwrap(),
        )
    }

    fn freeze_binding(protocol: &SealedDocumentBinding, prepared: bool) -> SealedDocumentBinding {
        binding(
            "/sealed/tg3-freeze.json",
            seal_value(json!({
                "schema": TG3_FREEZE_SCHEMA,
                "status": if prepared { TG3_FREEZE_PREPARED_STATUS } else { TG3_FREEZE_REFUSED_STATUS },
                "prepared": prepared,
                "tournament_active": false,
                "scored_task_execution_active": false,
                "winner_selected": false,
                "bound_final_manager_protocol": digest(protocol, FINAL_PROTOCOL_SCHEMA, FINAL_PROTOCOL_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
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

    fn matrix_cells() -> Vec<MatrixCell> {
        CANDIDATES
            .iter()
            .flat_map(|candidate| {
                MODES
                    .iter()
                    .enumerate()
                    .map(move |(index, mode)| MatrixCell {
                        candidate_artifact_id: (*candidate).into(),
                        evaluation_mode: (*mode).into(),
                        protected_scorecard_seal_sha256: sha(if index == 0 { '1' } else { '2' }),
                        all_task_evidence_verified: true,
                        all_hard_gates_retained: true,
                        adversarial_review_adjudicated: true,
                    })
            })
            .collect()
    }

    fn selection_identity() -> ProtectedSelectionIdentity {
        ProtectedSelectionIdentity {
            protected_verifier_identity_sha256: sha('3'),
            protected_controller_identity_sha256: sha('4'),
            protected_selection_namespace:
                "sealed://protected-tournament/selection/controller-verifier".into(),
            candidate_self_selection_authorized: false,
            candidate_weight_mutation_authorized: false,
            candidate_self_scoring_authorized: false,
            winner_selected: false,
            selected_winner_artifact_id: None,
        }
    }

    fn scorecard_binding(
        protocol: &SealedDocumentBinding,
        freeze: &SealedDocumentBinding,
        corpus: &SealedDocumentBinding,
        complete: bool,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/scorecard-adjudication.json",
            seal_value(json!({
                "schema": SCORECARD_SCHEMA,
                "status": if complete { SCORECARD_ADJUDICATED_STATUS } else { SCORECARD_PENDING_STATUS },
                "prepared": complete,
                "scored_execution_completed": complete,
                "bound_final_manager_protocol": digest(protocol, FINAL_PROTOCOL_SCHEMA, FINAL_PROTOCOL_STATUS),
                "bound_tg3_freeze_authority": digest(freeze, TG3_FREEZE_SCHEMA, if complete { TG3_FREEZE_PREPARED_STATUS } else { TG3_FREEZE_REFUSED_STATUS }),
                "bound_protected_corpus_commitment": digest(corpus, CORPUS_SCHEMA, CORPUS_PREPARED_STATUS),
                "matrix_cells": if complete { matrix_cells() } else { Vec::<MatrixCell>::new() },
                "complete_two_candidate_two_mode_matrix": complete,
                "pareto_frontier": {
                    "complete_two_candidate_frontier": complete,
                    "no_early_scalar_collapse": complete,
                    "protected_frontier_commitment_sha256": sha('5'),
                    "candidate_artifact_ids": if complete { CANDIDATES.to_vec() } else { Vec::<&str>::new() },
                },
                "protected_selection_identity": selection_identity(),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn winner_reservation() -> WinnerImmutableSealReservation {
        WinnerImmutableSealReservation {
            required_component_labels: WINNER_SEAL_COMPONENTS
                .iter()
                .map(|value| (*value).into())
                .collect(),
            immutable_revision_zero_required: true,
            winner_seal_must_cover_all_components_before_any_evictability: true,
            winner_immutable_seal_sha256: None,
            winner_selected_by_this_contract: false,
            selected_winner_artifact_id: None,
        }
    }

    fn loser_reservation() -> LoserRecoveryReservation {
        LoserRecoveryReservation {
            loser_seal_before_any_evictability: true,
            cold_store_or_offload_required: true,
            hash_verify_required: true,
            restore_test_required: true,
            one_command_recovery_required: true,
            no_delete_before_restore_proof: true,
            artifact_move_performed_by_this_contract: false,
            artifact_delete_performed_by_this_contract: false,
            evictability_granted_by_this_contract: false,
        }
    }

    fn input_fixture(complete: bool) -> Input {
        let final_manager_protocol = protocol_binding();
        let tg3_freeze_authority = freeze_binding(&final_manager_protocol, complete);
        let protected_corpus_commitment = corpus_binding(&final_manager_protocol);
        let scorecard_adjudication = scorecard_binding(
            &final_manager_protocol,
            &tg3_freeze_authority,
            &protected_corpus_commitment,
            complete,
        );
        Input {
            schema: INPUT_SCHEMA.into(),
            final_manager_protocol,
            tg3_freeze_authority,
            protected_corpus_commitment,
            scorecard_adjudication,
            winner_immutable_seal_reservation: winner_reservation(),
            loser_recovery_reservation: loser_reservation(),
            selection_requested: false,
            artifact_move_requested: false,
            artifact_delete_requested: false,
            evictability_requested: false,
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
    fn current_pending_inputs_refuse_without_selecting_or_moving_anything() {
        let report = build_report(input_fixture(false)).unwrap();
        assert_eq!(report.status, REFUSED_STATUS);
        assert!(!report.prepared);
        assert!(!report.winner_selected);
        assert!(!report.artifact_move_performed);
        assert!(!report.artifact_delete_performed);
        assert!(!report.evictability_granted);
        assert!(report
            .state_blockers
            .iter()
            .any(|blocker| blocker.contains("scorecard")));
    }

    #[test]
    fn complete_evidence_matrix_pareto_and_reservations_can_only_prepare_without_winner() {
        let report = build_report(input_fixture(true)).unwrap();
        assert_eq!(report.status, PREPARED_STATUS);
        assert!(report.prepared);
        assert!(!report.winner_selected);
        assert!(!report.artifact_move_performed);
        assert!(!report.artifact_delete_performed);
        assert!(!report.evictability_granted);
        assert!(
            report
                .focused_checks
                .complete_two_candidate_two_mode_verified_evidence_matrix_required
        );
    }

    #[test]
    fn incomplete_duplicate_or_unverified_matrix_and_pareto_are_rejected() {
        let mut missing = input_fixture(true);
        missing.scorecard_adjudication.document["matrix_cells"]
            .as_array_mut()
            .unwrap()
            .pop();
        let document = missing.scorecard_adjudication.document.clone();
        missing.scorecard_adjudication.document = seal_value(document).unwrap();
        missing.scorecard_adjudication.document_sha256 =
            sha256_json(&missing.scorecard_adjudication.document).unwrap();
        assert!(build_report(missing).is_err());

        let mut scalar = input_fixture(true);
        scalar.scorecard_adjudication.document["pareto_frontier"]["no_early_scalar_collapse"] =
            Value::Bool(false);
        let document = scalar.scorecard_adjudication.document.clone();
        scalar.scorecard_adjudication.document = seal_value(document).unwrap();
        scalar.scorecard_adjudication.document_sha256 =
            sha256_json(&scalar.scorecard_adjudication.document).unwrap();
        assert!(build_report(scalar).is_err());
    }

    #[test]
    fn candidate_self_selection_and_unprotected_identities_are_rejected() {
        let mut self_select = input_fixture(true);
        self_select.scorecard_adjudication.document["protected_selection_identity"]
            ["candidate_self_selection_authorized"] = Value::Bool(true);
        let document = self_select.scorecard_adjudication.document.clone();
        self_select.scorecard_adjudication.document = seal_value(document).unwrap();
        self_select.scorecard_adjudication.document_sha256 =
            sha256_json(&self_select.scorecard_adjudication.document).unwrap();
        assert!(build_report(self_select).is_err());

        let mut preselected = input_fixture(true);
        preselected.scorecard_adjudication.document["protected_selection_identity"]
            ["winner_selected"] = Value::Bool(true);
        let document = preselected.scorecard_adjudication.document.clone();
        preselected.scorecard_adjudication.document = seal_value(document).unwrap();
        preselected.scorecard_adjudication.document_sha256 =
            sha256_json(&preselected.scorecard_adjudication.document).unwrap();
        assert!(build_report(preselected).is_err());
    }

    #[test]
    fn winner_seal_and_loser_recovery_contract_must_be_complete_but_not_performed() {
        let mut incomplete_winner = input_fixture(true);
        incomplete_winner
            .winner_immutable_seal_reservation
            .required_component_labels
            .pop();
        assert!(build_report(incomplete_winner).is_err());

        let mut actual_winner = input_fixture(true);
        actual_winner
            .winner_immutable_seal_reservation
            .winner_immutable_seal_sha256 = Some(sha('6'));
        assert!(build_report(actual_winner).is_err());

        let mut no_restore = input_fixture(true);
        no_restore
            .loser_recovery_reservation
            .one_command_recovery_required = false;
        assert!(build_report(no_restore).is_err());
    }

    #[test]
    fn selection_move_delete_and_evictability_requests_are_refused() {
        for request in [
            |input: &mut Input| input.selection_requested = true,
            |input: &mut Input| input.artifact_move_requested = true,
            |input: &mut Input| input.artifact_delete_requested = true,
            |input: &mut Input| input.evictability_requested = true,
        ] {
            let mut input = input_fixture(true);
            request(&mut input);
            let report = build_report(input).unwrap();
            assert_eq!(report.status, REFUSED_STATUS);
            assert!(!report.prepared);
            assert!(!report.winner_selected);
        }
    }

    #[test]
    fn authority_chain_runtime_gpu_server_and_tournament_boundaries_are_fail_closed() {
        let mut bad_chain = input_fixture(true);
        bad_chain.scorecard_adjudication.document["bound_protected_corpus_commitment"]
            ["document_sha256"] = Value::String(sha('0'));
        let document = bad_chain.scorecard_adjudication.document.clone();
        bad_chain.scorecard_adjudication.document = seal_value(document).unwrap();
        bad_chain.scorecard_adjudication.document_sha256 =
            sha256_json(&bad_chain.scorecard_adjudication.document).unwrap();
        assert!(build_report(bad_chain).is_err());

        let mut runtime = input_fixture(false);
        runtime.execution_boundary.runtime_watcher_or_server_started = true;
        assert!(build_report(runtime).is_err());

        let mut gpu = input_fixture(false);
        gpu.authority_boundary.gpu_leases_authorized = 1;
        assert!(build_report(gpu).is_err());
    }

    #[test]
    fn sealed_input_output_create_new_and_unknown_scorecard_fields_are_enforced() {
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
        unknown["scorecard_adjudication"]["document"]["winner_name"] =
            Value::String("not allowed".into());
        let unknown = seal_value(unknown).unwrap();
        fs::write(&unknown_path, serde_json::to_vec_pretty(&unknown).unwrap()).unwrap();
        assert!(run(Args {
            input: unknown_path,
            out: unknown_out,
        })
        .is_err());
    }
}
