//! Sealed protected real-Hawking task-corpus commitment for the final manager tournament.
//!
//! This CPU-only authority exposes only public task-family/campaign counts and
//! cryptographic commitments.  Exact tournament membership and task plaintext
//! remain in controller-only namespaces and are not created, opened, or read
//! here.  It binds the existing final-manager protocol identity plus the
//! paired-cognition lane, mutation, Knowledge Plane, scheduler, and TG10
//! authorities.  It cannot start scored work, model bodies, servers, GPU work,
//! watchers, leases, HCLI/TPS measurements, or a tournament.
//!
//! A valid result is always `PREPARED_NOT_ACTIVE` while the corpus is merely
//! committed/frozen for a future protected controller.  Any request to score,
//! materialize hidden tasks, expose membership/plaintext, mutate tasks/weights,
//! or self-score is refused.

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
    "hawking.ascension.manager_tournament_protected_task_corpus_commitment_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_task_corpus_commitment_authority.v1";
const FINAL_MANAGER_PROTOCOL_SCHEMA: &str =
    "hawking.ascension.final_manager_tournament_protocol.v1";
const FINAL_MANAGER_PROTOCOL_STATUS: &str =
    "PREPARED_FINAL_MANAGER_SELECTION_PROTOCOL_NOT_EXECUTED";
const FINAL_MANAGER_PROTOCOL_IDENTITY_SHA256: &str =
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
const TG10_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_both_tg10_development_activation_state_machine.v1";
const TG10_PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_EXACT_TG10_OPERATIONAL_RECEIPTS_BOUND_NO_RUNTIME_OR_TOURNAMENT";
const TG10_REFUSED_TODAY_STATUS: &str =
    "REFUSED_TODAY_PAIRED_COGNITION_WAITING_FOR_BOTH_EXACT_TG10_OPERATIONAL_RECEIPTS";
const STATUS_PREPARED: &str =
    "PREPARED_NOT_ACTIVE_PROTECTED_REAL_HAWKING_TASK_CORPUS_METADATA_COMMITTED_NO_HIDDEN_TASKS_OR_SCORED_EXECUTION";
const STATUS_REFUSED: &str =
    "REFUSED_PROTECTED_TASK_CORPUS_REQUEST_REQUIRES_CONTROLLER_ONLY_FROZEN_METADATA_NO_SCORING_OR_HIDDEN_TASK_ACCESS";
const CORPUS_IDENTITY_SCHEMA: &str =
    "hawking.ascension.manager_tournament_protected_task_corpus_identity.v1";

const FAMILIES: [&str; 15] = [
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
const CAMPAIGNS: [&str; 6] = [
    "six_stage_kernel_optimization",
    "multi_subsystem_hcli_repair",
    "exact_model_family_qualification",
    "storage_pressure_model_acquisition",
    "agent_os_concurrency_optimization",
    "gravity_representation_tournament",
];
const FAIR_ENVELOPE_DIMENSIONS: [&str; 11] = [
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
struct FinalProtocolTaskCorpus {
    blind_tasks_required: bool,
    hidden_membership_frozen_before_scored_execution: bool,
    real_hawking_work_only: bool,
    required_families: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolCampaigns {
    measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion:
        bool,
    required: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolFairnessEnvelope {
    record_every_asymmetry: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolAdversarialReview {
    measure_genuine_defects_and_false_objections: bool,
    opposing_candidate_is_read_only_red_team_reviewer: bool,
    protected_verifier_adjudicates_challenges: bool,
    required_after_each_candidate_task: bool,
    reviewer_may_not_modify_candidate_artifact: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolProtectedSelection {
    candidates_cannot_change_weights_or_hidden_tests: bool,
    candidates_cannot_promote_self_or_invalidate_opponent: bool,
    candidates_cannot_self_grade: bool,
    only_protected_controller_or_human_may_select_manager: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct FinalProtocolDocument {
    schema: String,
    status: String,
    protocol_identity_sha256: String,
    protected_task_corpus: FinalProtocolTaskCorpus,
    long_horizon_campaigns: FinalProtocolCampaigns,
    fairness_envelope: FinalProtocolFairnessEnvelope,
    adversarial_review: FinalProtocolAdversarialReview,
    protected_selection: FinalProtocolProtectedSelection,
}

#[derive(Clone, Debug, Deserialize)]
struct LaneAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    no_new_physical_model_process_authority: bool,
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
struct SchedulerAuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_development_active: bool,
    logical_sessions_created: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize)]
struct Tg10AuthorityDocument {
    schema: String,
    status: String,
    prepared: bool,
    paired_development_active: bool,
    paired_development_activation_authorized_by_this_contract: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    bound_scheduler_authority: EvidenceDigest,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct FamilyMetadata {
    family: String,
    public_task_count: u16,
    protected_membership_commitment_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CampaignMetadata {
    campaign: String,
    public_stage_count: u8,
    protected_membership_commitment_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProtectedTaskNamespaceReservation {
    controller_only_task_membership_namespace: String,
    controller_only_task_plaintext_namespace: String,
    controller_only_answer_key_namespace: String,
    candidates_may_read_hidden_membership: bool,
    red_teams_may_read_hidden_membership: bool,
    candidates_may_read_hidden_plaintext: bool,
    red_teams_may_read_hidden_plaintext: bool,
    controller_only_membership_mutable: bool,
    controller_only_plaintext_mutable: bool,
    hidden_tasks_created_by_this_contract: bool,
    hidden_plaintext_opened_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CandidateBoundary {
    candidates_may_mutate_tasks: bool,
    candidates_may_mutate_tournament_weights: bool,
    candidates_may_self_score: bool,
    candidates_may_self_promote: bool,
    candidates_may_invalidate_opponent: bool,
    red_team_may_modify_candidate_artifact: bool,
    protected_controller_or_verifier_owns_scoring: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct FairEnvelopeHook {
    public_fair_envelope_commitment_sha256: String,
    equalized_dimensions: Vec<String>,
    record_every_asymmetry: bool,
    candidate_specific_resource_advantage: bool,
    scored_execution_started: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct AdversarialReviewHook {
    enabled: bool,
    opposing_candidate_read_only_reviewer: bool,
    reviewer_may_modify_candidate_artifact: bool,
    protected_verifier_adjudicates_challenges: bool,
    review_may_read_hidden_membership: bool,
    review_may_read_hidden_plaintext: bool,
    protected_review_namespace: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct CorpusCommitmentIdentity {
    schema: String,
    corpus_id: String,
    protocol_identity_sha256: String,
    immutable: bool,
    revision: u32,
    supersedes_corpus_id: Option<String>,
    public_metadata_sha256: String,
    protected_full_membership_commitment_sha256: String,
    seal_sha256: String,
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
    tg10_authority: SealedDocumentBinding,
    family_metadata: Vec<FamilyMetadata>,
    campaign_metadata: Vec<CampaignMetadata>,
    protected_full_membership_commitment_sha256: String,
    corpus_commitment_identity: Value,
    protected_task_namespace_reservation: ProtectedTaskNamespaceReservation,
    candidate_boundary: CandidateBoundary,
    fair_envelope_hook: FairEnvelopeHook,
    adversarial_review_hook: AdversarialReviewHook,
    scored_execution_requested: bool,
    hidden_task_materialization_requested: bool,
    hidden_membership_read_requested_by_candidate: bool,
    hidden_plaintext_read_requested_by_red_team: bool,
    task_or_weight_mutation_requested_by_candidate: bool,
    self_score_requested_by_candidate: bool,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct PublicCorpusManifest {
    corpus_id: String,
    protocol_identity_sha256: String,
    family_metadata: Vec<FamilyMetadata>,
    campaign_metadata: Vec<CampaignMetadata>,
    public_metadata_sha256: String,
    protected_full_membership_commitment_sha256: String,
    membership_frozen_before_scored_execution: bool,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    exact_final_protocol_and_paired_authority_chain_bound: bool,
    all_fifteen_real_hawking_task_families_committed: bool,
    all_six_long_horizon_campaigns_committed: bool,
    only_public_family_count_and_commitment_metadata_exposed: bool,
    protected_membership_plaintext_and_answer_keys_controller_only: bool,
    blind_membership_immutable_and_frozen_before_scored_execution: bool,
    candidates_and_red_teams_cannot_read_hidden_membership_or_plaintext: bool,
    candidates_cannot_mutate_tasks_weights_or_self_score: bool,
    fair_envelope_and_read_only_adversarial_review_hooks_bound: bool,
    final_manager_selection_and_tg3_remain_outside_corpus_commitment: bool,
    no_runtime_server_model_gpu_watcher_lease_tps_or_tournament_action: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: String,
    prepared: bool,
    hidden_tasks_created: bool,
    scored_execution_started: bool,
    candidate_or_red_team_hidden_access_granted: bool,
    bound_final_manager_protocol: EvidenceDigest,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    bound_scheduler_authority: EvidenceDigest,
    bound_tg10_authority: EvidenceDigest,
    public_corpus_manifest: PublicCorpusManifest,
    protected_task_namespace_reservation: ProtectedTaskNamespaceReservation,
    fair_envelope_hook: FairEnvelopeHook,
    adversarial_review_hook: AdversarialReviewHook,
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

fn require_sealed_namespace(path: &str, label: &str) -> Result<(), String> {
    if !path.starts_with("sealed://") || path.len() <= "sealed://".len() {
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
            "{label} may not authorize a process, server, port, GPU lease, tournament mutation, or activation"
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
            "{label}.document_sha256 does not bind its embedded document"
        ));
    }
    let document_seal_sha256 =
        verify_sealed_object(&binding.document, &format!("{label}.document"))?;
    let object = binding
        .document
        .as_object()
        .ok_or_else(|| format!("{label}.document must be an object"))?;
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
    let protocol: FinalProtocolDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("final_manager_protocol.document has wrong grammar: {error}"))?;
    if protocol.schema != FINAL_MANAGER_PROTOCOL_SCHEMA
        || protocol.status != FINAL_MANAGER_PROTOCOL_STATUS
        || protocol.protocol_identity_sha256 != FINAL_MANAGER_PROTOCOL_IDENTITY_SHA256
    {
        return Err(
            "final_manager_protocol must be the existing prepared final-manager protocol identity"
                .into(),
        );
    }
    exact_set(
        &protocol.protected_task_corpus.required_families,
        &FAMILIES,
        "final_manager_protocol.protected_task_corpus.required_families",
    )?;
    exact_set(
        &protocol.long_horizon_campaigns.required,
        &CAMPAIGNS,
        "final_manager_protocol.long_horizon_campaigns.required",
    )?;
    if !protocol.protected_task_corpus.blind_tasks_required
        || !protocol
            .protected_task_corpus
            .hidden_membership_frozen_before_scored_execution
        || !protocol.protected_task_corpus.real_hawking_work_only
        || !protocol
            .long_horizon_campaigns
            .measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion
        || !protocol.fairness_envelope.record_every_asymmetry
    {
        return Err("final_manager_protocol must require real blind frozen corpus, campaigns, and fairness asymmetry records".into());
    }
    let review = &protocol.adversarial_review;
    if !review.measure_genuine_defects_and_false_objections
        || !review.opposing_candidate_is_read_only_red_team_reviewer
        || !review.protected_verifier_adjudicates_challenges
        || !review.required_after_each_candidate_task
        || !review.reviewer_may_not_modify_candidate_artifact
    {
        return Err(
            "final_manager_protocol must retain the protected read-only adversarial-review round"
                .into(),
        );
    }
    let selection = &protocol.protected_selection;
    if !selection.candidates_cannot_change_weights_or_hidden_tests
        || !selection.candidates_cannot_promote_self_or_invalidate_opponent
        || !selection.candidates_cannot_self_grade
        || !selection.only_protected_controller_or_human_may_select_manager
    {
        return Err("final_manager_protocol must retain protected non-self-selection".into());
    }
    Ok(digest)
}

fn validate_lane_authority(binding: &SealedDocumentBinding) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "lane_authority")?;
    let document: LaneAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("lane_authority.document has wrong grammar: {error}"))?;
    if document.schema != LANE_AUTHORITY_SCHEMA
        || document.status != LANE_AUTHORITY_STATUS
        || !document.prepared
        || document.paired_candidate_worlds_active
        || !document.no_new_physical_model_process_authority
    {
        return Err(
            "lane_authority must be the prepared inactive one-body paired authority".into(),
        );
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "lane_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "lane_authority.execution_boundary",
    )?;
    Ok(digest)
}

fn validate_mutation_authority(
    binding: &SealedDocumentBinding,
    lane: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "mutation_authority")?;
    let document: MutationAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("mutation_authority.document has wrong grammar: {error}"))?;
    if document.schema != MUTATION_AUTHORITY_SCHEMA
        || document.status != MUTATION_AUTHORITY_STATUS
        || !document.prepared
        || document.paired_candidate_worlds_active
    {
        return Err("mutation_authority must be prepared and inactive".into());
    }
    exact_digest(lane, &document.bound_lane_authority, "mutation_authority")?;
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
    lane: &EvidenceDigest,
    mutation: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "knowledge_authority")?;
    let document: KnowledgeAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("knowledge_authority.document has wrong grammar: {error}"))?;
    if document.schema != KNOWLEDGE_AUTHORITY_SCHEMA
        || document.status != KNOWLEDGE_AUTHORITY_STATUS
        || !document.prepared
        || document.knowledge_plane_active
        || document.external_publication_performed
    {
        return Err(
            "knowledge_authority must be prepared, inactive, and externally unpublished".into(),
        );
    }
    exact_digest(lane, &document.bound_lane_authority, "knowledge_authority")?;
    exact_digest(
        mutation,
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
    lane: &EvidenceDigest,
    mutation: &EvidenceDigest,
    knowledge: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "scheduler_authority")?;
    let document: SchedulerAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("scheduler_authority.document has wrong grammar: {error}"))?;
    if document.schema != SCHEDULER_AUTHORITY_SCHEMA
        || document.status != SCHEDULER_AUTHORITY_STATUS
        || !document.prepared
        || document.paired_development_active
        || document.logical_sessions_created
    {
        return Err("scheduler_authority must remain prepared without runtime sessions".into());
    }
    exact_digest(lane, &document.bound_lane_authority, "scheduler_authority")?;
    exact_digest(
        mutation,
        &document.bound_mutation_authority,
        "scheduler_authority",
    )?;
    exact_digest(
        knowledge,
        &document.bound_knowledge_authority,
        "scheduler_authority",
    )?;
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

fn validate_tg10_authority(
    binding: &SealedDocumentBinding,
    lane: &EvidenceDigest,
    mutation: &EvidenceDigest,
    knowledge: &EvidenceDigest,
    scheduler: &EvidenceDigest,
) -> Result<EvidenceDigest, String> {
    let digest = binding_digest(binding, "tg10_authority")?;
    let document: Tg10AuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("tg10_authority.document has wrong grammar: {error}"))?;
    if document.schema != TG10_AUTHORITY_SCHEMA
        || (document.status != TG10_PREPARED_STATUS && document.status != TG10_REFUSED_TODAY_STATUS)
        || (document.status == TG10_PREPARED_STATUS && !document.prepared)
        || (document.status == TG10_REFUSED_TODAY_STATUS && document.prepared)
        || document.paired_development_active
        || document.paired_development_activation_authorized_by_this_contract
    {
        return Err(
            "tg10_authority must be an inactive prepared/refused gate, never an activator".into(),
        );
    }
    exact_digest(lane, &document.bound_lane_authority, "tg10_authority")?;
    exact_digest(
        mutation,
        &document.bound_mutation_authority,
        "tg10_authority",
    )?;
    exact_digest(
        knowledge,
        &document.bound_knowledge_authority,
        "tg10_authority",
    )?;
    exact_digest(
        scheduler,
        &document.bound_scheduler_authority,
        "tg10_authority",
    )?;
    validate_authority_boundary(
        &document.authority_boundary,
        "tg10_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "tg10_authority.execution_boundary",
    )?;
    Ok(digest)
}

fn validate_family_metadata(metadata: &[FamilyMetadata]) -> Result<Vec<FamilyMetadata>, String> {
    if metadata.len() != FAMILIES.len() {
        return Err("exactly 15 required real-Hawking family metadata records are required".into());
    }
    let mut families = BTreeSet::new();
    for item in metadata {
        if item.public_task_count == 0
            || !is_lower_sha256(&item.protected_membership_commitment_sha256)
        {
            return Err("each family must publish only a nonzero public count and SHA-256 membership commitment".into());
        }
        if !families.insert(item.family.as_str()) {
            return Err("family metadata contains a duplicate family".into());
        }
    }
    if families != FAMILIES.iter().copied().collect() {
        return Err(
            "family metadata must commit every required real-Hawking family exactly once".into(),
        );
    }
    let mut result = metadata.to_vec();
    result.sort_by(|left, right| left.family.cmp(&right.family));
    Ok(result)
}

fn validate_campaign_metadata(
    metadata: &[CampaignMetadata],
) -> Result<Vec<CampaignMetadata>, String> {
    if metadata.len() != CAMPAIGNS.len() {
        return Err(
            "exactly six required long-horizon campaign metadata records are required".into(),
        );
    }
    let mut campaigns = BTreeSet::new();
    for item in metadata {
        if item.public_stage_count < 2
            || !is_lower_sha256(&item.protected_membership_commitment_sha256)
        {
            return Err("each long-horizon campaign needs >=2 public stages and a SHA-256 membership commitment".into());
        }
        if !campaigns.insert(item.campaign.as_str()) {
            return Err("campaign metadata contains a duplicate campaign".into());
        }
    }
    if campaigns != CAMPAIGNS.iter().copied().collect() {
        return Err(
            "campaign metadata must commit every required long-horizon campaign exactly once"
                .into(),
        );
    }
    let mut result = metadata.to_vec();
    result.sort_by(|left, right| left.campaign.cmp(&right.campaign));
    Ok(result)
}

fn public_metadata_sha256(
    families: &[FamilyMetadata],
    campaigns: &[CampaignMetadata],
) -> Result<String, String> {
    sha256_json(&serde_json::json!({
        "families": families,
        "campaigns": campaigns,
    }))
}

fn expected_corpus_id(identity: &Value) -> Result<String, String> {
    let object = identity
        .as_object()
        .ok_or("corpus_commitment_identity must be an object")?;
    let mut preimage = object.clone();
    preimage.remove("corpus_id");
    preimage.remove("seal_sha256");
    Ok(format!(
        "hawking-protected-corpus-{}",
        sha256_json(&Value::Object(preimage))?
    ))
}

fn validate_corpus_identity(
    value: &Value,
    public_metadata_sha256: &str,
    full_membership_commitment: &str,
) -> Result<CorpusCommitmentIdentity, String> {
    let seal = verify_sealed_object(value, "corpus_commitment_identity")?;
    let identity: CorpusCommitmentIdentity = serde_json::from_value(value.clone())
        .map_err(|error| format!("corpus_commitment_identity has wrong grammar: {error}"))?;
    if identity.schema != CORPUS_IDENTITY_SCHEMA
        || identity.protocol_identity_sha256 != FINAL_MANAGER_PROTOCOL_IDENTITY_SHA256
        || !identity.immutable
        || identity.revision != 0
        || identity.supersedes_corpus_id.is_some()
        || identity.seal_sha256 != seal
        || identity.corpus_id != expected_corpus_id(value)?
        || identity.public_metadata_sha256 != public_metadata_sha256
        || identity.protected_full_membership_commitment_sha256 != full_membership_commitment
    {
        return Err("corpus commitment identity must be an immutable revision-zero exact public/membership binding".into());
    }
    if !is_lower_sha256(&identity.public_metadata_sha256)
        || !is_lower_sha256(&identity.protected_full_membership_commitment_sha256)
    {
        return Err("corpus commitment identity must use lowercase SHA-256 commitments".into());
    }
    Ok(identity)
}

fn validate_namespace_reservation(
    reservation: &ProtectedTaskNamespaceReservation,
) -> Result<(), String> {
    let namespaces = [
        &reservation.controller_only_task_membership_namespace,
        &reservation.controller_only_task_plaintext_namespace,
        &reservation.controller_only_answer_key_namespace,
    ];
    let mut unique = BTreeSet::new();
    for namespace in namespaces {
        require_sealed_namespace(namespace, "protected task namespace")?;
        if !namespace.starts_with("sealed://protected-tournament/controller-only/")
            || !unique.insert(namespace)
        {
            return Err("protected membership/plaintext/answer namespaces must be unique controller-only namespaces".into());
        }
    }
    if reservation.candidates_may_read_hidden_membership
        || reservation.red_teams_may_read_hidden_membership
        || reservation.candidates_may_read_hidden_plaintext
        || reservation.red_teams_may_read_hidden_plaintext
        || reservation.controller_only_membership_mutable
        || reservation.controller_only_plaintext_mutable
        || reservation.hidden_tasks_created_by_this_contract
        || reservation.hidden_plaintext_opened_by_this_contract
    {
        return Err("hidden membership/plaintext must remain immutable, unopened, and unavailable to candidates/red teams".into());
    }
    Ok(())
}

fn validate_candidate_boundary(boundary: &CandidateBoundary) -> Result<(), String> {
    if boundary.candidates_may_mutate_tasks
        || boundary.candidates_may_mutate_tournament_weights
        || boundary.candidates_may_self_score
        || boundary.candidates_may_self_promote
        || boundary.candidates_may_invalidate_opponent
        || boundary.red_team_may_modify_candidate_artifact
        || !boundary.protected_controller_or_verifier_owns_scoring
    {
        return Err("candidates/red teams must not mutate, self-score/promote, invalidate, or modify artifacts".into());
    }
    Ok(())
}

fn validate_fair_envelope(hook: &FairEnvelopeHook) -> Result<(), String> {
    if !is_lower_sha256(&hook.public_fair_envelope_commitment_sha256)
        || !hook.record_every_asymmetry
        || hook.candidate_specific_resource_advantage
        || hook.scored_execution_started
    {
        return Err(
            "fair envelope must be committed, asymmetry-recording, equalized, and unscored".into(),
        );
    }
    exact_set(
        &hook.equalized_dimensions,
        &FAIR_ENVELOPE_DIMENSIONS,
        "fair envelope dimensions",
    )
}

fn validate_adversarial_review(hook: &AdversarialReviewHook) -> Result<(), String> {
    if !hook.enabled
        || !hook.opposing_candidate_read_only_reviewer
        || hook.reviewer_may_modify_candidate_artifact
        || !hook.protected_verifier_adjudicates_challenges
        || hook.review_may_read_hidden_membership
        || hook.review_may_read_hidden_plaintext
    {
        return Err("adversarial review must be read-only, verifier-adjudicated, and blind to hidden corpus content".into());
    }
    require_sealed_namespace(
        &hook.protected_review_namespace,
        "protected_review_namespace",
    )?;
    if !hook
        .protected_review_namespace
        .starts_with("sealed://protected-tournament/adversarial-review/")
    {
        return Err(
            "protected_review_namespace must live in the protected adversarial-review namespace"
                .into(),
        );
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
    let final_protocol = validate_final_protocol(&input.final_manager_protocol)?;
    let lane = validate_lane_authority(&input.lane_authority)?;
    let mutation = validate_mutation_authority(&input.mutation_authority, &lane)?;
    let knowledge = validate_knowledge_authority(&input.knowledge_authority, &lane, &mutation)?;
    let scheduler =
        validate_scheduler_authority(&input.scheduler_authority, &lane, &mutation, &knowledge)?;
    let tg10 = validate_tg10_authority(
        &input.tg10_authority,
        &lane,
        &mutation,
        &knowledge,
        &scheduler,
    )?;
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;
    let families = validate_family_metadata(&input.family_metadata)?;
    let campaigns = validate_campaign_metadata(&input.campaign_metadata)?;
    if !is_lower_sha256(&input.protected_full_membership_commitment_sha256) {
        return Err("protected_full_membership_commitment_sha256 must be lowercase SHA-256".into());
    }
    let public_metadata_sha256 = public_metadata_sha256(&families, &campaigns)?;
    let identity = validate_corpus_identity(
        &input.corpus_commitment_identity,
        &public_metadata_sha256,
        &input.protected_full_membership_commitment_sha256,
    )?;
    validate_namespace_reservation(&input.protected_task_namespace_reservation)?;
    validate_candidate_boundary(&input.candidate_boundary)?;
    validate_fair_envelope(&input.fair_envelope_hook)?;
    validate_adversarial_review(&input.adversarial_review_hook)?;

    let mut state_blockers = Vec::new();
    if input.scored_execution_requested {
        state_blockers.push("scored_execution_requested_outside_this_commitment_authority".into());
    }
    if input.hidden_task_materialization_requested {
        state_blockers
            .push("hidden_task_materialization_requested_outside_this_commitment_authority".into());
    }
    if input.hidden_membership_read_requested_by_candidate {
        state_blockers.push("candidate_hidden_membership_read_requested".into());
    }
    if input.hidden_plaintext_read_requested_by_red_team {
        state_blockers.push("red_team_hidden_plaintext_read_requested".into());
    }
    if input.task_or_weight_mutation_requested_by_candidate {
        state_blockers.push("candidate_task_or_weight_mutation_requested".into());
    }
    if input.self_score_requested_by_candidate {
        state_blockers.push("candidate_self_score_requested".into());
    }
    let prepared = state_blockers.is_empty();
    let status = if prepared {
        STATUS_PREPARED.into()
    } else {
        state_blockers.sort();
        state_blockers.dedup();
        STATUS_REFUSED.into()
    };
    let manifest = PublicCorpusManifest {
        corpus_id: identity.corpus_id,
        protocol_identity_sha256: identity.protocol_identity_sha256,
        family_metadata: families,
        campaign_metadata: campaigns,
        public_metadata_sha256,
        protected_full_membership_commitment_sha256: input
            .protected_full_membership_commitment_sha256,
        membership_frozen_before_scored_execution: true,
    };
    let focused_checks = FocusedChecks {
        exact_final_protocol_and_paired_authority_chain_bound: true,
        all_fifteen_real_hawking_task_families_committed: true,
        all_six_long_horizon_campaigns_committed: true,
        only_public_family_count_and_commitment_metadata_exposed: true,
        protected_membership_plaintext_and_answer_keys_controller_only: true,
        blind_membership_immutable_and_frozen_before_scored_execution: true,
        candidates_and_red_teams_cannot_read_hidden_membership_or_plaintext: true,
        candidates_cannot_mutate_tasks_weights_or_self_score: true,
        fair_envelope_and_read_only_adversarial_review_hooks_bound: true,
        final_manager_selection_and_tg3_remain_outside_corpus_commitment: true,
        no_runtime_server_model_gpu_watcher_lease_tps_or_tournament_action: true,
        execution_boundary_cpu_only: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        prepared,
        hidden_tasks_created: false,
        scored_execution_started: false,
        candidate_or_red_team_hidden_access_granted: false,
        bound_final_manager_protocol: final_protocol,
        bound_lane_authority: lane,
        bound_mutation_authority: mutation,
        bound_knowledge_authority: knowledge,
        bound_scheduler_authority: scheduler,
        bound_tg10_authority: tg10,
        public_corpus_manifest: manifest,
        protected_task_namespace_reservation: input.protected_task_namespace_reservation,
        fair_envelope_hook: input.fair_envelope_hook,
        adversarial_review_hook: input.adversarial_review_hook,
        state_blockers,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only protected task-corpus commitment, not a corpus generator, scored tournament runner, or manager selector.",
            "It exposes only the required real-Hawking family/campaign labels, public counts, and SHA-256 commitments; no task membership, task plaintext, answer key, candidate scorecard, or task result is emitted.",
            "All 15 task families and six long-horizon campaigns are exact bindings of the existing final-manager protocol identity.",
            "Membership/plaintext/answer keys are reserved under immutable controller-only namespaces; candidates and red teams cannot read them, mutate them, change tournament weights, self-score, promote themselves, or invalidate an opponent.",
            "Fair-envelope and read-only adversarial-review hooks are policy commitments only. They allocate no runtime resource and authorize no reviewer modification or hidden-corpus access.",
            "The corpus commitment is frozen before scored execution. TG3, qualification, final scorecards, protected selection, and tournament execution remain outside this authority.",
            "No model body, server, watcher, port, GPU lease, device work, HCLI request, token, TPS/TG measurement, hidden task, or tournament state is created or changed.",
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
            .map_err(|error| format!("corpus commitment report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("corpus commitment report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_manager_tournament_protected_task_corpus_commitment --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        .map_err(|error| format!("corpus commitment validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_manager_tournament_protected_task_corpus_commitment: {error}");
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
                "schema": FINAL_MANAGER_PROTOCOL_SCHEMA,
                "status": FINAL_MANAGER_PROTOCOL_STATUS,
                "protocol_identity_sha256": FINAL_MANAGER_PROTOCOL_IDENTITY_SHA256,
                "protected_task_corpus": {
                    "blind_tasks_required": true,
                    "hidden_membership_frozen_before_scored_execution": true,
                    "real_hawking_work_only": true,
                    "required_families": FAMILIES,
                },
                "long_horizon_campaigns": {
                    "measure_initial_plan_adaptation_resets_repeated_mistakes_branches_goal_fidelity_and_completion": true,
                    "required": CAMPAIGNS,
                },
                "fairness_envelope": {"record_every_asymmetry": true},
                "adversarial_review": {
                    "measure_genuine_defects_and_false_objections": true,
                    "opposing_candidate_is_read_only_red_team_reviewer": true,
                    "protected_verifier_adjudicates_challenges": true,
                    "required_after_each_candidate_task": true,
                    "reviewer_may_not_modify_candidate_artifact": true,
                },
                "protected_selection": {
                    "candidates_cannot_change_weights_or_hidden_tests": true,
                    "candidates_cannot_promote_self_or_invalidate_opponent": true,
                    "candidates_cannot_self_grade": true,
                    "only_protected_controller_or_human_may_select_manager": true,
                },
            }))
            .unwrap(),
        )
    }

    fn lane_binding() -> SealedDocumentBinding {
        binding(
            "/sealed/lane.json",
            seal_value(json!({
                "schema": LANE_AUTHORITY_SCHEMA,
                "status": LANE_AUTHORITY_STATUS,
                "prepared": true,
                "paired_candidate_worlds_active": false,
                "no_new_physical_model_process_authority": true,
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn mutation_binding(lane: &SealedDocumentBinding) -> SealedDocumentBinding {
        binding(
            "/sealed/mutation.json",
            seal_value(json!({
                "schema": MUTATION_AUTHORITY_SCHEMA,
                "status": MUTATION_AUTHORITY_STATUS,
                "prepared": true,
                "paired_candidate_worlds_active": false,
                "bound_lane_authority": digest(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn knowledge_binding(
        lane: &SealedDocumentBinding,
        mutation: &SealedDocumentBinding,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/knowledge.json",
            seal_value(json!({
                "schema": KNOWLEDGE_AUTHORITY_SCHEMA,
                "status": KNOWLEDGE_AUTHORITY_STATUS,
                "prepared": true,
                "knowledge_plane_active": false,
                "external_publication_performed": false,
                "bound_lane_authority": digest(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest(mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn scheduler_binding(
        lane: &SealedDocumentBinding,
        mutation: &SealedDocumentBinding,
        knowledge: &SealedDocumentBinding,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/scheduler.json",
            seal_value(json!({
                "schema": SCHEDULER_AUTHORITY_SCHEMA,
                "status": SCHEDULER_AUTHORITY_STATUS,
                "prepared": true,
                "paired_development_active": false,
                "logical_sessions_created": false,
                "bound_lane_authority": digest(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest(mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "bound_knowledge_authority": digest(knowledge, KNOWLEDGE_AUTHORITY_SCHEMA, KNOWLEDGE_AUTHORITY_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn tg10_binding(
        lane: &SealedDocumentBinding,
        mutation: &SealedDocumentBinding,
        knowledge: &SealedDocumentBinding,
        scheduler: &SealedDocumentBinding,
    ) -> SealedDocumentBinding {
        binding(
            "/sealed/tg10.json",
            seal_value(json!({
                "schema": TG10_AUTHORITY_SCHEMA,
                "status": TG10_REFUSED_TODAY_STATUS,
                "prepared": false,
                "paired_development_active": false,
                "paired_development_activation_authorized_by_this_contract": false,
                "bound_lane_authority": digest(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest(mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "bound_knowledge_authority": digest(knowledge, KNOWLEDGE_AUTHORITY_SCHEMA, KNOWLEDGE_AUTHORITY_STATUS),
                "bound_scheduler_authority": digest(scheduler, SCHEDULER_AUTHORITY_SCHEMA, SCHEDULER_AUTHORITY_STATUS),
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn families() -> Vec<FamilyMetadata> {
        FAMILIES
            .iter()
            .enumerate()
            .map(|(index, family)| FamilyMetadata {
                family: (*family).into(),
                public_task_count: u16::try_from(index + 1).unwrap(),
                protected_membership_commitment_sha256: sha(char::from_digit(
                    u32::try_from(index % 10).unwrap(),
                    10,
                )
                .unwrap()),
            })
            .collect()
    }

    fn campaigns() -> Vec<CampaignMetadata> {
        CAMPAIGNS
            .iter()
            .enumerate()
            .map(|(index, campaign)| CampaignMetadata {
                campaign: (*campaign).into(),
                public_stage_count: u8::try_from(index + 2).unwrap(),
                protected_membership_commitment_sha256: sha(char::from_digit(
                    u32::try_from((index + 4) % 10).unwrap(),
                    10,
                )
                .unwrap()),
            })
            .collect()
    }

    fn namespace_reservation() -> ProtectedTaskNamespaceReservation {
        ProtectedTaskNamespaceReservation {
            controller_only_task_membership_namespace:
                "sealed://protected-tournament/controller-only/task-membership".into(),
            controller_only_task_plaintext_namespace:
                "sealed://protected-tournament/controller-only/task-plaintext".into(),
            controller_only_answer_key_namespace:
                "sealed://protected-tournament/controller-only/answer-keys".into(),
            candidates_may_read_hidden_membership: false,
            red_teams_may_read_hidden_membership: false,
            candidates_may_read_hidden_plaintext: false,
            red_teams_may_read_hidden_plaintext: false,
            controller_only_membership_mutable: false,
            controller_only_plaintext_mutable: false,
            hidden_tasks_created_by_this_contract: false,
            hidden_plaintext_opened_by_this_contract: false,
        }
    }

    fn candidate_boundary() -> CandidateBoundary {
        CandidateBoundary {
            candidates_may_mutate_tasks: false,
            candidates_may_mutate_tournament_weights: false,
            candidates_may_self_score: false,
            candidates_may_self_promote: false,
            candidates_may_invalidate_opponent: false,
            red_team_may_modify_candidate_artifact: false,
            protected_controller_or_verifier_owns_scoring: true,
        }
    }

    fn fair_hook() -> FairEnvelopeHook {
        FairEnvelopeHook {
            public_fair_envelope_commitment_sha256: sha('e'),
            equalized_dimensions: FAIR_ENVELOPE_DIMENSIONS
                .iter()
                .map(|value| (*value).into())
                .collect(),
            record_every_asymmetry: true,
            candidate_specific_resource_advantage: false,
            scored_execution_started: false,
        }
    }

    fn adversarial_hook() -> AdversarialReviewHook {
        AdversarialReviewHook {
            enabled: true,
            opposing_candidate_read_only_reviewer: true,
            reviewer_may_modify_candidate_artifact: false,
            protected_verifier_adjudicates_challenges: true,
            review_may_read_hidden_membership: false,
            review_may_read_hidden_plaintext: false,
            protected_review_namespace:
                "sealed://protected-tournament/adversarial-review/read-only".into(),
        }
    }

    fn identity(
        family_metadata: &[FamilyMetadata],
        campaign_metadata: &[CampaignMetadata],
    ) -> Value {
        let mut canonical_families = family_metadata.to_vec();
        canonical_families.sort_by(|left, right| left.family.cmp(&right.family));
        let mut canonical_campaigns = campaign_metadata.to_vec();
        canonical_campaigns.sort_by(|left, right| left.campaign.cmp(&right.campaign));
        let public_metadata_sha256 =
            public_metadata_sha256(&canonical_families, &canonical_campaigns).unwrap();
        let mut value = json!({
            "schema": CORPUS_IDENTITY_SCHEMA,
            "corpus_id": "",
            "protocol_identity_sha256": FINAL_MANAGER_PROTOCOL_IDENTITY_SHA256,
            "immutable": true,
            "revision": 0,
            "supersedes_corpus_id": Value::Null,
            "public_metadata_sha256": public_metadata_sha256,
            "protected_full_membership_commitment_sha256": sha('f'),
        });
        value["corpus_id"] = Value::String(expected_corpus_id(&value).unwrap());
        seal_value(value).unwrap()
    }

    fn input_fixture() -> Input {
        let final_manager_protocol = protocol_binding();
        let lane_authority = lane_binding();
        let mutation_authority = mutation_binding(&lane_authority);
        let knowledge_authority = knowledge_binding(&lane_authority, &mutation_authority);
        let scheduler_authority =
            scheduler_binding(&lane_authority, &mutation_authority, &knowledge_authority);
        let tg10_authority = tg10_binding(
            &lane_authority,
            &mutation_authority,
            &knowledge_authority,
            &scheduler_authority,
        );
        let family_metadata = families();
        let campaign_metadata = campaigns();
        Input {
            schema: INPUT_SCHEMA.into(),
            final_manager_protocol,
            lane_authority,
            mutation_authority,
            knowledge_authority,
            scheduler_authority,
            tg10_authority,
            corpus_commitment_identity: identity(&family_metadata, &campaign_metadata),
            family_metadata,
            campaign_metadata,
            protected_full_membership_commitment_sha256: sha('f'),
            protected_task_namespace_reservation: namespace_reservation(),
            candidate_boundary: candidate_boundary(),
            fair_envelope_hook: fair_hook(),
            adversarial_review_hook: adversarial_hook(),
            scored_execution_requested: false,
            hidden_task_materialization_requested: false,
            hidden_membership_read_requested_by_candidate: false,
            hidden_plaintext_read_requested_by_red_team: false,
            task_or_weight_mutation_requested_by_candidate: false,
            self_score_requested_by_candidate: false,
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
    fn exact_public_commitment_for_all_families_campaigns_and_authorities_stays_prepared() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, STATUS_PREPARED);
        assert!(report.prepared);
        assert!(!report.hidden_tasks_created);
        assert!(!report.scored_execution_started);
        assert_eq!(report.public_corpus_manifest.family_metadata.len(), 15);
        assert_eq!(report.public_corpus_manifest.campaign_metadata.len(), 6);
        assert!(
            report
                .focused_checks
                .all_fifteen_real_hawking_task_families_committed
        );
        assert!(
            report
                .focused_checks
                .all_six_long_horizon_campaigns_committed
        );
        assert!(report.focused_checks.execution_boundary_cpu_only);
    }

    #[test]
    fn missing_or_duplicate_real_hawking_family_and_campaign_commitments_are_rejected() {
        let mut missing_family = input_fixture();
        missing_family.family_metadata.pop();
        assert!(build_report(missing_family).is_err());

        let mut duplicate_campaign = input_fixture();
        duplicate_campaign.campaign_metadata[1].campaign =
            duplicate_campaign.campaign_metadata[0].campaign.clone();
        assert!(build_report(duplicate_campaign).is_err());
    }

    #[test]
    fn task_membership_plaintext_and_answer_keys_remain_controller_only() {
        let mut candidate_read = input_fixture();
        candidate_read
            .protected_task_namespace_reservation
            .candidates_may_read_hidden_membership = true;
        assert!(build_report(candidate_read).is_err());

        let mut mutable_plaintext = input_fixture();
        mutable_plaintext
            .protected_task_namespace_reservation
            .controller_only_plaintext_mutable = true;
        assert!(build_report(mutable_plaintext).is_err());

        let mut nonprotected_namespace = input_fixture();
        nonprotected_namespace
            .protected_task_namespace_reservation
            .controller_only_task_plaintext_namespace = "sealed://qwen30-lane/tasks".into();
        assert!(build_report(nonprotected_namespace).is_err());
    }

    #[test]
    fn candidates_and_red_teams_cannot_mutate_self_score_or_modify_artifacts() {
        let mut weights = input_fixture();
        weights
            .candidate_boundary
            .candidates_may_mutate_tournament_weights = true;
        assert!(build_report(weights).is_err());

        let mut self_score = input_fixture();
        self_score.candidate_boundary.candidates_may_self_score = true;
        assert!(build_report(self_score).is_err());

        let mut red_team = input_fixture();
        red_team
            .adversarial_review_hook
            .reviewer_may_modify_candidate_artifact = true;
        assert!(build_report(red_team).is_err());
    }

    #[test]
    fn fair_envelope_and_adversarial_review_hooks_are_fail_closed() {
        let mut unfair = input_fixture();
        unfair.fair_envelope_hook.equalized_dimensions.pop();
        assert!(build_report(unfair).is_err());

        let mut hidden_review = input_fixture();
        hidden_review
            .adversarial_review_hook
            .review_may_read_hidden_plaintext = true;
        assert!(build_report(hidden_review).is_err());
    }

    #[test]
    fn scored_execution_hidden_task_access_mutation_and_self_score_requests_are_refused() {
        for mutator in [
            |input: &mut Input| input.scored_execution_requested = true,
            |input: &mut Input| input.hidden_task_materialization_requested = true,
            |input: &mut Input| input.hidden_membership_read_requested_by_candidate = true,
            |input: &mut Input| input.hidden_plaintext_read_requested_by_red_team = true,
            |input: &mut Input| input.task_or_weight_mutation_requested_by_candidate = true,
            |input: &mut Input| input.self_score_requested_by_candidate = true,
        ] {
            let mut input = input_fixture();
            mutator(&mut input);
            let report = build_report(input).unwrap();
            assert_eq!(report.status, STATUS_REFUSED);
            assert!(!report.prepared);
            assert!(!report.scored_execution_started);
        }
    }

    #[test]
    fn identity_and_sealed_authority_chain_must_be_exact() {
        let mut altered_identity = input_fixture();
        altered_identity.corpus_commitment_identity["public_metadata_sha256"] =
            Value::String(sha('0'));
        let identity_value = altered_identity.corpus_commitment_identity.clone();
        altered_identity.corpus_commitment_identity = seal_value(identity_value).unwrap();
        assert!(build_report(altered_identity).is_err());

        let mut bad_chain = input_fixture();
        bad_chain.tg10_authority.document["bound_scheduler_authority"]["document_sha256"] =
            Value::String(sha('0'));
        let document = bad_chain.tg10_authority.document.clone();
        bad_chain.tg10_authority.document = seal_value(document).unwrap();
        bad_chain.tg10_authority.document_sha256 =
            sha256_json(&bad_chain.tg10_authority.document).unwrap();
        assert!(build_report(bad_chain).is_err());
    }

    #[test]
    fn runtime_gpu_server_watcher_and_tournament_boundaries_are_fail_closed() {
        let mut runtime = input_fixture();
        runtime.execution_boundary.runtime_watcher_or_server_started = true;
        assert!(build_report(runtime).is_err());

        let mut lease = input_fixture();
        lease.authority_boundary.gpu_leases_authorized = 1;
        assert!(build_report(lease).is_err());
    }

    #[test]
    fn sealed_input_output_create_new_and_unknown_public_task_fields_are_enforced() {
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
        assert_eq!(output["status"], Value::String(STATUS_PREPARED.into()));
        assert!(
            write_report_create_new(&output_path, &build_report(input_fixture()).unwrap()).is_err()
        );

        let rejected_path = directory.path().join("unknown-public-task-field.json");
        let rejected_out = directory.path().join("unknown-public-task-field-out.json");
        let mut unknown = seal_input(&input_fixture());
        unknown["family_metadata"][0]["hidden_task_plaintext"] =
            Value::String("not permitted".into());
        let unknown = seal_value(unknown).unwrap();
        fs::write(&rejected_path, serde_json::to_vec_pretty(&unknown).unwrap()).unwrap();
        assert!(run(Args {
            input: rejected_path,
            out: rejected_out,
        })
        .is_err());
    }
}
