//! Sealed paired-cognition Knowledge Plane generic-discovery release gate.
//!
//! The paired candidate worlds are intentionally private.  This CPU-only
//! contract is the narrow bridge between them: it can validate an append-only,
//! sealed *generic mechanism/science* release, but it cannot read or release a
//! lane's strategy, frontier, private score, current patch, hidden task,
//! experiment history, or receipt contents.  It consumes both the completed
//! lane authority and the completed proposal/review/mutation authority as
//! embedded sealed inputs, so a caller cannot substitute an authority by
//! changing a path after the fact.
//!
//! A successful report is still `PREPARED/NOT_ACTIVE`.  It does not publish to
//! a network, create a model/session/worktree, start a server or watcher, bind
//! a port, acquire a GPU lease, load weights, execute a token, measure TPS/TG,
//! mutate a candidate, or alter a tournament.  A later protected controller
//! may consume an approved immutable identity; this contract only proves that
//! its public payload is generic and independently verified.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_paired_cognition_knowledge_plane_release_gate -- \
//!   --input /absolute/path/KNOWLEDGE_PLANE_INPUT.json \
//!   --out /absolute/new/path/KNOWLEDGE_PLANE_RELEASE_AUTHORITY.json
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str = "hawking.ascension.paired_cognition_knowledge_plane_release_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_knowledge_plane_generic_release_authority.v1";
const LANE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_lane_namespace_mission_authority.v1";
const LANE_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_TWO_SEALED_CANDIDATE_WORLDS_NO_RUNTIME_SERVER_OR_TOURNAMENT";
const MUTATION_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_proposal_review_falsification_primary_acceptance_authority.v1";
const MUTATION_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_PRIMARY_ONLY_CHAMPION_MUTATION_PROMOTION_NO_MANAGER_OR_TOURNAMENT_SELECTION";
const RELEASE_IDENTITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_knowledge_plane_immutable_release_identity.v1";
const STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_GENERIC_ONLY_INDEPENDENTLY_VERIFIED_APPEND_ONLY_KNOWLEDGE_PLANE_NO_RUNTIME_OR_TOURNAMENT";

const QWEN30: &str = "qwen30";
const QWEN80: &str = "qwen80";
const INDEPENDENT_REDACTOR: &str = "knowledge-plane-redaction-reviewer";
const INDEPENDENT_VERIFIER: &str = "knowledge-plane-independent-verifier";
const RELEASE_STEWARD: &str = "knowledge-plane-release-steward";
const GENERIC_CLASSIFICATION: &str = "generic_mechanism_science";
const REDACTED_SOURCE_SCOPE: &str = "sealed_lane_evidence_redacted_to_generic_claim";
const APPROVAL: &str = "approved_generic_only";
const REGISTRATION_MODE: &str = "append_only_sealed_prepared_no_external_publication";

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
const ALLOWED_TOPICS: [&str; 6] = [
    "mechanism",
    "science",
    "operator_semantics",
    "numerical_stability",
    "measurement_method",
    "reproducible_safety",
];
const FORBIDDEN_PUBLIC_TERMS: [&str; 20] = [
    "candidate",
    "strategy",
    "frontier",
    "private",
    "score",
    "patch",
    "hidden task",
    "current task",
    "current implementation",
    "experiment history",
    "mission",
    "worktree",
    "session",
    "tournament",
    "champion",
    "qwen30",
    "qwen80",
    "lane",
    "primary",
    "helper",
];

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct SealedDocumentBinding {
    path: String,
    document_sha256: String,
    document: Value,
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
struct BoundLane {
    lane_id: String,
    primary_model_key: String,
    helper_model_key: String,
    private_namespaces: PrivateNamespaces,
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
#[serde(deny_unknown_fields)]
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

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PublicPayload {
    classification: String,
    topics: Vec<String>,
    generic_summary: String,
    generic_claim: String,
    public_evidence_abstract: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ProvenanceAttestation {
    origin_lane_id: String,
    claim_author_actor_id: String,
    source_commitment_sha256: String,
    provenance_receipt_sha256: String,
    source_scope: String,
    private_evidence_disclosed: bool,
    cross_lane_private_read_used: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct RedactionAttestation {
    redactor_actor_id: String,
    redaction_receipt_sha256: String,
    redaction_complete: bool,
    candidate_strategy_removed: bool,
    frontier_removed: bool,
    private_score_removed: bool,
    current_patch_removed: bool,
    current_hidden_task_removed: bool,
    private_namespace_identifiers_removed: bool,
    raw_experiment_history_removed: bool,
    private_receipt_content_removed: bool,
    redactor_independent_from_claim_author: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct IndependentVerifierApproval {
    verifier_actor_id: String,
    verifier_receipt_sha256: String,
    approval: String,
    verified_generic_scope: bool,
    reviewed_redaction: bool,
    verified_no_lane_private_leak: bool,
    verifier_independent_from_claim_author: bool,
    verifier_independent_from_publisher: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct PublicationAttestation {
    publisher_actor_id: String,
    publisher_receipt_sha256: String,
    publisher_independent_from_claim_author: bool,
    publisher_independent_from_verifier: bool,
    append_only_identity_registration_requested: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct ReleaseIdentity {
    schema: String,
    release_id: String,
    immutable: bool,
    revision: u32,
    supersedes_release_id: Option<String>,
    public_payload_sha256: String,
    source_commitment_sha256: String,
    provenance_receipt_sha256: String,
    redaction_receipt_sha256: String,
    independent_verifier_receipt_sha256: String,
    publisher_receipt_sha256: String,
    publisher_actor_id: String,
    verifier_actor_id: String,
    seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
struct DiscoveryReleaseRequest {
    release_identity: Value,
    public_payload: PublicPayload,
    provenance: ProvenanceAttestation,
    redaction: RedactionAttestation,
    independent_verification: IndependentVerifierApproval,
    publication: PublicationAttestation,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    lane_authority: SealedDocumentBinding,
    mutation_authority: SealedDocumentBinding,
    knowledge_plane_policy: KnowledgePlanePolicy,
    discovery_release_requests: Vec<DiscoveryReleaseRequest>,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct ApprovedReleaseIdentity {
    release_id: String,
    immutable_release_identity_sha256: String,
    immutable_release_identity_seal_sha256: String,
    public_payload_sha256: String,
    source_commitment_sha256: String,
    independent_verifier_receipt_sha256: String,
    topics: Vec<String>,
    generic_summary: String,
    registration_mode: &'static str,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    sealed_lane_authority_bound: bool,
    sealed_mutation_authority_bound_to_exact_lane_authority: bool,
    exactly_two_private_candidate_worlds_preserved: bool,
    all_cross_lane_private_reads_denied: bool,
    generic_mechanism_science_payload_only: bool,
    candidate_strategy_frontier_score_patch_and_hidden_task_redacted: bool,
    provenance_is_lane_bound_without_private_disclosure: bool,
    independent_redactor_verifier_and_publisher_required: bool,
    self_published_and_self_verified_claims_denied: bool,
    immutable_append_only_release_identity_bound: bool,
    no_private_namespace_or_action_workspace_published: bool,
    no_runtime_server_gpu_or_tournament_authority: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    prepared: bool,
    knowledge_plane_active: bool,
    external_publication_performed: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    knowledge_plane_policy: KnowledgePlanePolicy,
    approved_release_identities: Vec<ApprovedReleaseIdentity>,
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
            "{label} may not authorize model processes, servers, ports, GPU leases, tournament mutation, or activation"
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
            "{label} must permit only independently verifier-released generic Knowledge Plane material"
        ));
    }
    Ok(())
}

fn namespace_entries(namespaces: &PrivateNamespaces) -> [&str; 8] {
    [
        &namespaces.mission,
        &namespaces.experiments,
        &namespaces.receipts,
        &namespaces.worktree,
        &namespaces.sessions,
        &namespaces.frontier,
        &namespaces.patches,
        &namespaces.scores,
    ]
}

fn expected_actor_ids(primary: &str, helper: &str) -> (String, String, String) {
    (
        format!("{primary}-primary-in-{primary}-lane"),
        format!("{helper}-helper-in-{primary}-lane"),
        format!("{helper}-opponent-reviewer-in-{primary}-lane"),
    )
}

fn exact_actions(actual: &[String], expected: &[&str], label: &str) -> Result<(), String> {
    let actual_set: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let expected_set: BTreeSet<&str> = expected.iter().copied().collect();
    if actual_set.len() != actual.len() || actual_set != expected_set {
        return Err(format!("{label} must grant exactly {:?}", expected_set));
    }
    Ok(())
}

fn binding_digest(
    binding: &SealedDocumentBinding,
    label: &str,
) -> Result<(EvidenceDigest, String), String> {
    require_absolute_path(&binding.path, &format!("{label}.path"))?;
    if !is_lower_sha256(&binding.document_sha256) {
        return Err(format!(
            "{label}.document_sha256 must be a lowercase SHA-256"
        ));
    }
    let document_sha256 = sha256_json(&binding.document)?;
    if document_sha256 != binding.document_sha256 {
        return Err(format!(
            "{label}.document_sha256 does not bind the embedded document"
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
    Ok((
        EvidenceDigest {
            path: binding.path.clone(),
            document_schema: schema.into(),
            document_status: status.into(),
            document_sha256: binding.document_sha256.clone(),
            document_seal_sha256,
        },
        binding.document_sha256.clone(),
    ))
}

fn validate_lane_authority(
    binding: &SealedDocumentBinding,
) -> Result<
    (
        EvidenceDigest,
        BTreeMap<String, BoundLane>,
        BTreeSet<String>,
    ),
    String,
> {
    let (digest, _) = binding_digest(binding, "lane_authority")?;
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
    if document.lane_worlds.len() != 2 {
        return Err("lane_authority.document must contain exactly two candidate worlds".into());
    }
    let mut lanes = BTreeMap::new();
    let mut private_namespaces = BTreeSet::new();
    for lane in document.lane_worlds {
        if lane.lane_id.trim().is_empty() || lanes.contains_key(&lane.lane_id) {
            return Err("lane_authority.document has an empty or duplicate lane id".into());
        }
        let expected_helper = match lane.primary_model_key.as_str() {
            QWEN30 => QWEN80,
            QWEN80 => QWEN30,
            other => return Err(format!("lane_authority has unsupported primary {other:?}")),
        };
        if lane.helper_model_key != expected_helper {
            return Err(format!("{} has an invalid helper model", lane.lane_id));
        }
        for namespace in namespace_entries(&lane.private_namespaces) {
            require_sealed_namespace(namespace, &format!("{}.private_namespaces", lane.lane_id))?;
            if !private_namespaces.insert(namespace.to_owned()) {
                return Err("lane_authority aliases a private namespace across worlds".into());
            }
        }
        lanes.insert(lane.lane_id.clone(), lane);
    }
    let primaries: BTreeSet<&str> = lanes
        .values()
        .map(|lane| lane.primary_model_key.as_str())
        .collect();
    if primaries != BTreeSet::from([QWEN30, QWEN80]) {
        return Err("lane_authority must bind one Q30 and one Q80 primary world".into());
    }
    Ok((digest, lanes, private_namespaces))
}

fn validate_mutation_authority(
    binding: &SealedDocumentBinding,
    exact_lane_digest: &EvidenceDigest,
    lanes: &BTreeMap<String, BoundLane>,
) -> Result<(EvidenceDigest, BTreeMap<String, MutationLaneActionPolicy>), String> {
    let (digest, _) = binding_digest(binding, "mutation_authority")?;
    let document: MutationAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("mutation_authority.document has the wrong grammar: {error}"))?;
    if document.schema != MUTATION_AUTHORITY_SCHEMA || document.status != MUTATION_AUTHORITY_STATUS
    {
        return Err(
            "mutation_authority.document is not the completed paired mutation authority".into(),
        );
    }
    if !document.prepared || document.paired_candidate_worlds_active {
        return Err("mutation_authority.document must remain prepared and inactive".into());
    }
    if document.bound_lane_authority.document_schema != LANE_AUTHORITY_SCHEMA
        || document.bound_lane_authority.document_status != LANE_AUTHORITY_STATUS
        || document.bound_lane_authority.document_sha256 != exact_lane_digest.document_sha256
        || document.bound_lane_authority.document_seal_sha256
            != exact_lane_digest.document_seal_sha256
    {
        return Err(
            "mutation_authority.document is not bound to this exact sealed lane authority".into(),
        );
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "mutation_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "mutation_authority.execution_boundary",
    )?;
    let protected = &document.protected_record_policy;
    if !protected.evidence_receipts_immutable
        || !protected.mission_records_immutable
        || !protected.tournament_receipts_immutable
        || protected.all_roles_may_rewrite_evidence_receipts
        || protected.all_roles_may_rewrite_mission_records
        || protected.all_roles_may_rewrite_tournament_receipts
        || protected.all_roles_may_delete_protected_records
        || protected.cross_lane_private_record_release_allowed
    {
        return Err(
            "mutation_authority must keep records immutable and cross-lane private records sealed"
                .into(),
        );
    }
    if document.lane_action_policies.len() != lanes.len() {
        return Err("mutation_authority must have one action policy per candidate world".into());
    }
    let mut policies = BTreeMap::new();
    for policy in document.lane_action_policies {
        let lane = lanes
            .get(&policy.lane_id)
            .ok_or_else(|| format!("mutation policy refers to unknown lane {}", policy.lane_id))?;
        if policy.primary_model_key != lane.primary_model_key
            || policy.helper_model_key != lane.helper_model_key
            || policy.opponent_model_key != lane.helper_model_key
        {
            return Err(format!(
                "{} mutation policy does not match its bound lane",
                policy.lane_id
            ));
        }
        let expected = expected_actor_ids(&lane.primary_model_key, &lane.helper_model_key);
        if (
            policy.primary_actor_id.as_str(),
            policy.helper_actor_id.as_str(),
            policy.opponent_actor_id.as_str(),
        ) != (
            expected.0.as_str(),
            expected.1.as_str(),
            expected.2.as_str(),
        ) {
            return Err(format!(
                "{} mutation policy has unbound logical actors",
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
            return Err("mutation_authority has duplicate action-policy lanes".into());
        }
    }
    Ok((digest, policies))
}

fn validate_policy(
    policy: &KnowledgePlanePolicy,
    private_namespaces: &BTreeSet<String>,
) -> Result<(), String> {
    for (label, namespace) in [
        (
            "knowledge_plane_namespace",
            &policy.knowledge_plane_namespace,
        ),
        (
            "release_registry_namespace",
            &policy.release_registry_namespace,
        ),
    ] {
        require_sealed_namespace(namespace, label)?;
        if !namespace.starts_with("sealed://knowledge-plane/") {
            return Err(format!("{label} must live under sealed://knowledge-plane/"));
        }
        if private_namespaces.contains(namespace) {
            return Err(format!("{label} may not alias a lane-private namespace"));
        }
    }
    if policy.knowledge_plane_namespace == policy.release_registry_namespace {
        return Err(
            "knowledge_plane_namespace and release_registry_namespace must be distinct".into(),
        );
    }
    if !policy.generic_mechanism_science_only
        || !policy.append_only_release_identities
        || policy.release_identity_mutation_authorized
        || policy.external_publication_authorized
        || policy.lane_private_record_access_authorized
        || policy.candidate_world_activation_authorized
    {
        return Err("Knowledge Plane must be generic-only, append-only, and have no mutable, external, private, or activation authority".into());
    }
    Ok(())
}

fn validate_public_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty()
        || value.len() > 4096
        || !value.is_ascii()
        || value.chars().any(char::is_control)
    {
        return Err(format!(
            "{label} must be non-empty, printable ASCII, and at most 4096 bytes"
        ));
    }
    let normalized = value.to_ascii_lowercase();
    for forbidden in FORBIDDEN_PUBLIC_TERMS {
        if normalized.contains(forbidden) {
            return Err(format!(
                "{label} contains prohibited private/candidate disclosure term {forbidden:?}"
            ));
        }
    }
    Ok(())
}

fn validate_public_payload(payload: &PublicPayload) -> Result<(), String> {
    if payload.classification != GENERIC_CLASSIFICATION {
        return Err(format!(
            "public_payload.classification must be {GENERIC_CLASSIFICATION:?}"
        ));
    }
    if payload.topics.is_empty() {
        return Err("public_payload.topics must not be empty".into());
    }
    let mut seen = BTreeSet::new();
    for topic in &payload.topics {
        if !ALLOWED_TOPICS.contains(&topic.as_str()) || !seen.insert(topic) {
            return Err(
                "public_payload.topics must be unique generic mechanism/science topics".into(),
            );
        }
    }
    validate_public_text(&payload.generic_summary, "public_payload.generic_summary")?;
    validate_public_text(&payload.generic_claim, "public_payload.generic_claim")?;
    validate_public_text(
        &payload.public_evidence_abstract,
        "public_payload.public_evidence_abstract",
    )?;
    Ok(())
}

fn validate_provenance(
    provenance: &ProvenanceAttestation,
    lanes: &BTreeMap<String, BoundLane>,
    policies: &BTreeMap<String, MutationLaneActionPolicy>,
) -> Result<(), String> {
    if provenance.source_scope != REDACTED_SOURCE_SCOPE
        || provenance.private_evidence_disclosed
        || provenance.cross_lane_private_read_used
    {
        return Err("provenance must bind redacted lane evidence without disclosure or cross-lane private reads".into());
    }
    if !is_lower_sha256(&provenance.source_commitment_sha256)
        || !is_lower_sha256(&provenance.provenance_receipt_sha256)
    {
        return Err("provenance commitment and receipt must be lowercase SHA-256 values".into());
    }
    let lane = lanes.get(&provenance.origin_lane_id).ok_or_else(|| {
        format!(
            "provenance origin {} is not a sealed candidate world",
            provenance.origin_lane_id
        )
    })?;
    let policy = policies.get(&provenance.origin_lane_id).ok_or_else(|| {
        format!(
            "provenance origin {} has no mutation policy",
            provenance.origin_lane_id
        )
    })?;
    if policy.primary_model_key != lane.primary_model_key
        || policy.helper_model_key != lane.helper_model_key
    {
        return Err("provenance lane and mutation policy disagree".into());
    }
    if provenance.claim_author_actor_id != policy.primary_actor_id
        && provenance.claim_author_actor_id != policy.helper_actor_id
    {
        return Err("claim author must be the originating lane's sealed primary or helper, never an unbound or opposing actor".into());
    }
    Ok(())
}

fn validate_redaction(
    redaction: &RedactionAttestation,
    provenance: &ProvenanceAttestation,
) -> Result<(), String> {
    if redaction.redactor_actor_id != INDEPENDENT_REDACTOR
        || !is_lower_sha256(&redaction.redaction_receipt_sha256)
        || !redaction.redaction_complete
        || !redaction.candidate_strategy_removed
        || !redaction.frontier_removed
        || !redaction.private_score_removed
        || !redaction.current_patch_removed
        || !redaction.current_hidden_task_removed
        || !redaction.private_namespace_identifiers_removed
        || !redaction.raw_experiment_history_removed
        || !redaction.private_receipt_content_removed
        || !redaction.redactor_independent_from_claim_author
        || redaction.redactor_actor_id == provenance.claim_author_actor_id
    {
        return Err("redaction must be independently attested and remove every lane-private/candidate field".into());
    }
    Ok(())
}

fn validate_independent_verification(
    approval: &IndependentVerifierApproval,
    provenance: &ProvenanceAttestation,
    publication: &PublicationAttestation,
) -> Result<(), String> {
    if approval.verifier_actor_id != INDEPENDENT_VERIFIER
        || !is_lower_sha256(&approval.verifier_receipt_sha256)
        || approval.approval != APPROVAL
        || !approval.verified_generic_scope
        || !approval.reviewed_redaction
        || !approval.verified_no_lane_private_leak
        || !approval.verifier_independent_from_claim_author
        || !approval.verifier_independent_from_publisher
        || approval.verifier_actor_id == provenance.claim_author_actor_id
        || approval.verifier_actor_id == publication.publisher_actor_id
    {
        return Err("an independently approved generic-only release is required; self-verification is prohibited".into());
    }
    Ok(())
}

fn validate_publication(
    publication: &PublicationAttestation,
    provenance: &ProvenanceAttestation,
    redaction: &RedactionAttestation,
    approval: &IndependentVerifierApproval,
) -> Result<(), String> {
    if publication.publisher_actor_id != RELEASE_STEWARD
        || !is_lower_sha256(&publication.publisher_receipt_sha256)
        || !publication.publisher_independent_from_claim_author
        || !publication.publisher_independent_from_verifier
        || !publication.append_only_identity_registration_requested
        || publication.publisher_actor_id == provenance.claim_author_actor_id
        || publication.publisher_actor_id == approval.verifier_actor_id
        || publication.publisher_actor_id == redaction.redactor_actor_id
    {
        return Err(
            "publication must use the independent release steward and cannot self-publish a claim"
                .into(),
        );
    }
    let distinct: BTreeSet<&str> = [
        provenance.claim_author_actor_id.as_str(),
        redaction.redactor_actor_id.as_str(),
        approval.verifier_actor_id.as_str(),
        publication.publisher_actor_id.as_str(),
    ]
    .into_iter()
    .collect();
    if distinct.len() != 4 {
        return Err("claim author, redactor, verifier, and publisher must be four distinct logical identities".into());
    }
    Ok(())
}

fn expected_release_id(identity: &Value) -> Result<String, String> {
    let object = identity
        .as_object()
        .ok_or("release_identity must be a JSON object")?;
    let mut immutable_preimage = object.clone();
    immutable_preimage.remove("release_id");
    immutable_preimage.remove("seal_sha256");
    Ok(format!(
        "kp-{}",
        sha256_json(&Value::Object(immutable_preimage))?
    ))
}

fn validate_release_identity(
    identity_value: &Value,
    payload: &PublicPayload,
    provenance: &ProvenanceAttestation,
    redaction: &RedactionAttestation,
    approval: &IndependentVerifierApproval,
    publication: &PublicationAttestation,
) -> Result<(ReleaseIdentity, String, String), String> {
    let identity_seal = verify_sealed_object(identity_value, "release_identity")?;
    let identity_sha256 = sha256_json(identity_value)?;
    let identity: ReleaseIdentity = serde_json::from_value(identity_value.clone())
        .map_err(|error| format!("release_identity has the wrong grammar: {error}"))?;
    if identity.schema != RELEASE_IDENTITY_SCHEMA
        || !identity.immutable
        || identity.revision != 0
        || identity.supersedes_release_id.is_some()
        || identity.seal_sha256 != identity_seal
    {
        return Err(
            "release_identity must be an original immutable revision-zero sealed identity".into(),
        );
    }
    if identity.release_id != expected_release_id(identity_value)? {
        return Err(
            "release_identity.release_id must be the deterministic immutable preimage identity"
                .into(),
        );
    }
    let public_payload_value = serde_json::to_value(payload)
        .map_err(|error| format!("public payload cannot be serialized: {error}"))?;
    let expected_payload_sha256 = sha256_json(&public_payload_value)?;
    let required_hashes = [
        (
            "public_payload_sha256",
            &identity.public_payload_sha256,
            &expected_payload_sha256,
        ),
        (
            "source_commitment_sha256",
            &identity.source_commitment_sha256,
            &provenance.source_commitment_sha256,
        ),
        (
            "provenance_receipt_sha256",
            &identity.provenance_receipt_sha256,
            &provenance.provenance_receipt_sha256,
        ),
        (
            "redaction_receipt_sha256",
            &identity.redaction_receipt_sha256,
            &redaction.redaction_receipt_sha256,
        ),
        (
            "independent_verifier_receipt_sha256",
            &identity.independent_verifier_receipt_sha256,
            &approval.verifier_receipt_sha256,
        ),
        (
            "publisher_receipt_sha256",
            &identity.publisher_receipt_sha256,
            &publication.publisher_receipt_sha256,
        ),
    ];
    for (label, observed, expected) in required_hashes {
        if !is_lower_sha256(observed) || observed != expected {
            return Err(format!(
                "release_identity.{label} does not bind the validated release input"
            ));
        }
    }
    if identity.publisher_actor_id != publication.publisher_actor_id
        || identity.verifier_actor_id != approval.verifier_actor_id
    {
        return Err(
            "release_identity must bind the independent verifier and publisher identities".into(),
        );
    }
    Ok((identity, identity_sha256, identity_seal))
}

fn validate_release_request(
    request: &DiscoveryReleaseRequest,
    lanes: &BTreeMap<String, BoundLane>,
    policies: &BTreeMap<String, MutationLaneActionPolicy>,
) -> Result<ApprovedReleaseIdentity, String> {
    validate_public_payload(&request.public_payload)?;
    validate_provenance(&request.provenance, lanes, policies)?;
    validate_redaction(&request.redaction, &request.provenance)?;
    validate_independent_verification(
        &request.independent_verification,
        &request.provenance,
        &request.publication,
    )?;
    validate_publication(
        &request.publication,
        &request.provenance,
        &request.redaction,
        &request.independent_verification,
    )?;
    let (identity, identity_sha256, identity_seal) = validate_release_identity(
        &request.release_identity,
        &request.public_payload,
        &request.provenance,
        &request.redaction,
        &request.independent_verification,
        &request.publication,
    )?;
    Ok(ApprovedReleaseIdentity {
        release_id: identity.release_id,
        immutable_release_identity_sha256: identity_sha256,
        immutable_release_identity_seal_sha256: identity_seal,
        public_payload_sha256: identity.public_payload_sha256,
        source_commitment_sha256: identity.source_commitment_sha256,
        independent_verifier_receipt_sha256: identity.independent_verifier_receipt_sha256,
        topics: request.public_payload.topics.clone(),
        generic_summary: request.public_payload.generic_summary.clone(),
        registration_mode: REGISTRATION_MODE,
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
    let (lane_digest, lanes, private_namespaces) = validate_lane_authority(&input.lane_authority)?;
    let (mutation_digest, policies) =
        validate_mutation_authority(&input.mutation_authority, &lane_digest, &lanes)?;
    validate_policy(&input.knowledge_plane_policy, &private_namespaces)?;
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;
    if input.discovery_release_requests.is_empty() {
        return Err(
            "at least one independently verified generic release request is required".into(),
        );
    }
    let mut approved_release_identities =
        Vec::with_capacity(input.discovery_release_requests.len());
    let mut release_ids = BTreeSet::new();
    let mut identity_hashes = BTreeSet::new();
    for request in &input.discovery_release_requests {
        let approved = validate_release_request(request, &lanes, &policies)?;
        if !release_ids.insert(approved.release_id.clone()) {
            return Err("duplicate immutable Knowledge Plane release_id".into());
        }
        if !identity_hashes.insert(approved.immutable_release_identity_sha256.clone()) {
            return Err("duplicate immutable Knowledge Plane release identity".into());
        }
        approved_release_identities.push(approved);
    }
    approved_release_identities.sort_by(|left, right| left.release_id.cmp(&right.release_id));

    let focused_checks = FocusedChecks {
        sealed_lane_authority_bound: true,
        sealed_mutation_authority_bound_to_exact_lane_authority: true,
        exactly_two_private_candidate_worlds_preserved: lanes.len() == 2,
        all_cross_lane_private_reads_denied: true,
        generic_mechanism_science_payload_only: true,
        candidate_strategy_frontier_score_patch_and_hidden_task_redacted: true,
        provenance_is_lane_bound_without_private_disclosure: true,
        independent_redactor_verifier_and_publisher_required: true,
        self_published_and_self_verified_claims_denied: true,
        immutable_append_only_release_identity_bound: true,
        no_private_namespace_or_action_workspace_published: true,
        no_runtime_server_gpu_or_tournament_authority: true,
        execution_boundary_cpu_only: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status: STATUS,
        prepared: true,
        knowledge_plane_active: false,
        external_publication_performed: false,
        bound_lane_authority: lane_digest,
        bound_mutation_authority: mutation_digest,
        knowledge_plane_policy: input.knowledge_plane_policy,
        approved_release_identities,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only generic-discovery release gate, not a live Knowledge Plane service or cross-lane read channel.",
            "It consumes the exact sealed lane and mutation authorities and keeps both candidate worlds private and inactive.",
            "Only a redacted generic mechanism/science payload with lane-bound digest provenance, an independent redactor, an independent verifier, and an independent release steward can receive an append-only identity.",
            "Candidate strategy, frontier, private score, current patch, hidden task, experiment history, private namespaces, and private receipt contents are forbidden from public payloads and must be positively redacted.",
            "A claim author cannot self-redact, self-verify, or self-publish; the release identity binds the payload and every protected receipt digest at immutable revision zero.",
            "It starts no model process, logical session, server, watcher, port listener, GPU lease, device dispatch, HCLI run, token, TPS/TG measurement, candidate mutation, activation, or tournament action.",
            "No external publication occurs here; a later protected controller may only consume the sealed prepared identity under its own qualification gates.",
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
            .map_err(|error| format!("Knowledge Plane report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("Knowledge Plane report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_paired_cognition_knowledge_plane_release_gate --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        .map_err(|error| format!("Knowledge Plane validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_paired_cognition_knowledge_plane_release_gate: {error}");
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

    fn lane(primary: &str, helper: &str) -> BoundLane {
        BoundLane {
            lane_id: format!("{primary}-candidate-world"),
            primary_model_key: primary.into(),
            helper_model_key: helper.into(),
            private_namespaces: private_namespaces(&format!("{primary}-lane")),
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

    fn sealed_lane_authority_document() -> Value {
        seal_value(json!({
            "schema": LANE_AUTHORITY_SCHEMA,
            "status": LANE_AUTHORITY_STATUS,
            "prepared": true,
            "paired_candidate_worlds_active": false,
            "no_new_physical_model_process_authority": true,
            "lane_worlds": [lane(QWEN80, QWEN30), lane(QWEN30, QWEN80)],
            "cross_lane_read_policy": cross_lane_policy(),
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }))
        .unwrap()
    }

    fn lane_authority_binding() -> SealedDocumentBinding {
        let document = sealed_lane_authority_document();
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
            primary_allowed_actions: PRIMARY_ACTIONS
                .iter()
                .map(|value| (*value).into())
                .collect(),
            helper_allowed_actions: WORK_ACTIONS.iter().map(|value| (*value).into()).collect(),
            opponent_allowed_actions: WORK_ACTIONS.iter().map(|value| (*value).into()).collect(),
        }
    }

    fn protected_record_policy() -> ProtectedRecordPolicy {
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

    fn sealed_mutation_authority_document(lane_binding: &SealedDocumentBinding) -> Value {
        let lane_document_seal = lane_binding.document["seal_sha256"].as_str().unwrap();
        seal_value(json!({
            "schema": MUTATION_AUTHORITY_SCHEMA,
            "status": MUTATION_AUTHORITY_STATUS,
            "prepared": true,
            "paired_candidate_worlds_active": false,
            "bound_lane_authority": {
                "path": lane_binding.path,
                "document_schema": LANE_AUTHORITY_SCHEMA,
                "document_status": LANE_AUTHORITY_STATUS,
                "document_sha256": lane_binding.document_sha256,
                "document_seal_sha256": lane_document_seal,
            },
            "lane_action_policies": [action_policy(QWEN80, QWEN30), action_policy(QWEN30, QWEN80)],
            "protected_record_policy": protected_record_policy(),
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }))
        .unwrap()
    }

    fn mutation_authority_binding(lane_binding: &SealedDocumentBinding) -> SealedDocumentBinding {
        let document = sealed_mutation_authority_document(lane_binding);
        SealedDocumentBinding {
            path: "/sealed/paired-cognition/PAIRED_COGNITION_MUTATION_AUTHORITY.json".into(),
            document_sha256: sha256_json(&document).unwrap(),
            document,
        }
    }

    fn policy() -> KnowledgePlanePolicy {
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

    fn public_payload() -> PublicPayload {
        PublicPayload {
            classification: GENERIC_CLASSIFICATION.into(),
            topics: vec!["mechanism".into(), "numerical_stability".into()],
            generic_summary: "A fixed reduction order makes a numerical mechanism reproducible.".into(),
            generic_claim: "A documented reduction order preserves repeatable arithmetic behavior.".into(),
            public_evidence_abstract: "Independent arithmetic checks agree on the stated mechanism under a fixed input fixture.".into(),
        }
    }

    fn provenance() -> ProvenanceAttestation {
        ProvenanceAttestation {
            origin_lane_id: "qwen80-candidate-world".into(),
            claim_author_actor_id: "qwen80-primary-in-qwen80-lane".into(),
            source_commitment_sha256: sha('a'),
            provenance_receipt_sha256: sha('b'),
            source_scope: REDACTED_SOURCE_SCOPE.into(),
            private_evidence_disclosed: false,
            cross_lane_private_read_used: false,
        }
    }

    fn redaction() -> RedactionAttestation {
        RedactionAttestation {
            redactor_actor_id: INDEPENDENT_REDACTOR.into(),
            redaction_receipt_sha256: sha('c'),
            redaction_complete: true,
            candidate_strategy_removed: true,
            frontier_removed: true,
            private_score_removed: true,
            current_patch_removed: true,
            current_hidden_task_removed: true,
            private_namespace_identifiers_removed: true,
            raw_experiment_history_removed: true,
            private_receipt_content_removed: true,
            redactor_independent_from_claim_author: true,
        }
    }

    fn verifier() -> IndependentVerifierApproval {
        IndependentVerifierApproval {
            verifier_actor_id: INDEPENDENT_VERIFIER.into(),
            verifier_receipt_sha256: sha('d'),
            approval: APPROVAL.into(),
            verified_generic_scope: true,
            reviewed_redaction: true,
            verified_no_lane_private_leak: true,
            verifier_independent_from_claim_author: true,
            verifier_independent_from_publisher: true,
        }
    }

    fn publication() -> PublicationAttestation {
        PublicationAttestation {
            publisher_actor_id: RELEASE_STEWARD.into(),
            publisher_receipt_sha256: sha('e'),
            publisher_independent_from_claim_author: true,
            publisher_independent_from_verifier: true,
            append_only_identity_registration_requested: true,
        }
    }

    fn release_identity(
        payload: &PublicPayload,
        provenance: &ProvenanceAttestation,
        redaction: &RedactionAttestation,
        verifier: &IndependentVerifierApproval,
        publication: &PublicationAttestation,
    ) -> Value {
        let payload_sha = sha256_json(&serde_json::to_value(payload).unwrap()).unwrap();
        let mut value = json!({
            "schema": RELEASE_IDENTITY_SCHEMA,
            "release_id": "",
            "immutable": true,
            "revision": 0,
            "supersedes_release_id": Value::Null,
            "public_payload_sha256": payload_sha,
            "source_commitment_sha256": provenance.source_commitment_sha256,
            "provenance_receipt_sha256": provenance.provenance_receipt_sha256,
            "redaction_receipt_sha256": redaction.redaction_receipt_sha256,
            "independent_verifier_receipt_sha256": verifier.verifier_receipt_sha256,
            "publisher_receipt_sha256": publication.publisher_receipt_sha256,
            "publisher_actor_id": publication.publisher_actor_id,
            "verifier_actor_id": verifier.verifier_actor_id,
        });
        let release_id = expected_release_id(&value).unwrap();
        value["release_id"] = Value::String(release_id);
        seal_value(value).unwrap()
    }

    fn release_request() -> DiscoveryReleaseRequest {
        let payload = public_payload();
        let provenance = provenance();
        let redaction = redaction();
        let independent_verification = verifier();
        let publication = publication();
        let release_identity = release_identity(
            &payload,
            &provenance,
            &redaction,
            &independent_verification,
            &publication,
        );
        DiscoveryReleaseRequest {
            release_identity,
            public_payload: payload,
            provenance,
            redaction,
            independent_verification,
            publication,
        }
    }

    fn input_fixture() -> Input {
        let lane_authority = lane_authority_binding();
        let mutation_authority = mutation_authority_binding(&lane_authority);
        Input {
            schema: INPUT_SCHEMA.into(),
            lane_authority,
            mutation_authority,
            knowledge_plane_policy: policy(),
            discovery_release_requests: vec![release_request()],
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

    fn reseal_identity(request: &mut DiscoveryReleaseRequest) {
        let mut value = request.release_identity.clone();
        value["release_id"] = Value::String(expected_release_id(&value).unwrap());
        request.release_identity = seal_value(value).unwrap();
    }

    #[test]
    fn correct_generic_release_binds_both_authorities_and_stays_prepared_cpu_only() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, STATUS);
        assert!(report.prepared);
        assert!(!report.knowledge_plane_active);
        assert!(!report.external_publication_performed);
        assert_eq!(report.approved_release_identities.len(), 1);
        assert!(report.focused_checks.sealed_lane_authority_bound);
        assert!(
            report
                .focused_checks
                .sealed_mutation_authority_bound_to_exact_lane_authority
        );
        assert!(
            report
                .focused_checks
                .self_published_and_self_verified_claims_denied
        );
        assert!(
            report
                .focused_checks
                .immutable_append_only_release_identity_bound
        );
        assert!(report.focused_checks.execution_boundary_cpu_only);
    }

    #[test]
    fn exact_completed_lane_and_mutation_authorities_are_required() {
        let mut wrong_lane = input_fixture();
        wrong_lane.mutation_authority.document["bound_lane_authority"]["document_sha256"] =
            Value::String(sha('0'));
        let document = wrong_lane.mutation_authority.document.clone();
        wrong_lane.mutation_authority.document = seal_value(document).unwrap();
        wrong_lane.mutation_authority.document_sha256 =
            sha256_json(&wrong_lane.mutation_authority.document).unwrap();
        assert!(build_report(wrong_lane).is_err());

        let mut cross_read = input_fixture();
        cross_read.lane_authority.document["cross_lane_read_policy"]
            ["allow_cross_lane_frontier_reads"] = Value::Bool(true);
        let document = cross_read.lane_authority.document.clone();
        cross_read.lane_authority.document = seal_value(document).unwrap();
        cross_read.lane_authority.document_sha256 =
            sha256_json(&cross_read.lane_authority.document).unwrap();
        assert!(build_report(cross_read).is_err());
    }

    #[test]
    fn candidate_strategy_frontier_score_patch_hidden_task_and_private_text_are_rejected() {
        for leak in [
            "The candidate strategy is useful.",
            "A frontier observation is useful.",
            "A private score is useful.",
            "The current patch is useful.",
            "The hidden task is useful.",
        ] {
            let mut input = input_fixture();
            input.discovery_release_requests[0]
                .public_payload
                .generic_summary = leak.into();
            assert!(build_report(input).is_err(), "{leak} must fail");
        }
    }

    #[test]
    fn unredacted_or_cross_lane_provenance_is_rejected() {
        let mut unredacted = input_fixture();
        unredacted.discovery_release_requests[0]
            .redaction
            .private_receipt_content_removed = false;
        assert!(build_report(unredacted).is_err());

        let mut cross_lane = input_fixture();
        cross_lane.discovery_release_requests[0]
            .provenance
            .cross_lane_private_read_used = true;
        assert!(build_report(cross_lane).is_err());

        let mut unbound_actor = input_fixture();
        unbound_actor.discovery_release_requests[0]
            .provenance
            .claim_author_actor_id = "unbound-actor".into();
        assert!(build_report(unbound_actor).is_err());
    }

    #[test]
    fn self_redaction_self_verification_and_self_publication_are_rejected() {
        let mut self_redaction = input_fixture();
        self_redaction.discovery_release_requests[0]
            .redaction
            .redactor_actor_id = "qwen80-primary-in-qwen80-lane".into();
        assert!(build_report(self_redaction).is_err());

        let mut self_verify = input_fixture();
        self_verify.discovery_release_requests[0]
            .independent_verification
            .verifier_actor_id = "qwen80-primary-in-qwen80-lane".into();
        assert!(build_report(self_verify).is_err());

        let mut self_publish = input_fixture();
        self_publish.discovery_release_requests[0]
            .publication
            .publisher_actor_id = "qwen80-primary-in-qwen80-lane".into();
        assert!(build_report(self_publish).is_err());
    }

    #[test]
    fn immutable_identity_must_bind_payload_receipts_and_append_only_policy() {
        let mut altered_identity = input_fixture();
        altered_identity.discovery_release_requests[0].release_identity["public_payload_sha256"] =
            Value::String(sha('0'));
        reseal_identity(&mut altered_identity.discovery_release_requests[0]);
        assert!(build_report(altered_identity).is_err());

        let mut mutable_identity = input_fixture();
        mutable_identity.discovery_release_requests[0].release_identity["revision"] =
            Value::from(1);
        reseal_identity(&mut mutable_identity.discovery_release_requests[0]);
        assert!(build_report(mutable_identity).is_err());

        let mut mutable_registry = input_fixture();
        mutable_registry
            .knowledge_plane_policy
            .release_identity_mutation_authorized = true;
        assert!(build_report(mutable_registry).is_err());
    }

    #[test]
    fn runtime_gpu_server_and_tournament_boundaries_are_fail_closed() {
        let mut runtime = input_fixture();
        runtime.execution_boundary.runtime_watcher_or_server_started = true;
        assert!(build_report(runtime).is_err());

        let mut gpu = input_fixture();
        gpu.authority_boundary.gpu_leases_authorized = 1;
        assert!(build_report(gpu).is_err());

        let mut external = input_fixture();
        external
            .knowledge_plane_policy
            .external_publication_authorized = true;
        assert!(build_report(external).is_err());
    }

    #[test]
    fn sealed_input_output_and_create_new_are_enforced() {
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
    fn tampered_outer_seal_is_rejected_before_release_evaluation() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("tampered-input.json");
        let output_path = directory.path().join("out.json");
        let mut sealed = seal_input(&input_fixture());
        sealed["knowledge_plane_policy"]["external_publication_authorized"] = Value::Bool(true);
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        assert!(run(Args {
            input: input_path,
            out: output_path,
        })
        .is_err());
    }
}
