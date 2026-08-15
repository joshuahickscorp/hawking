//! Sealed paired-cognition proposal, review, falsification, and acceptance authority.
//!
//! This CPU-only control-plane contract consumes the already-sealed paired
//! lane namespace/mission authority and produces a narrower authority for
//! two candidate worlds.  It deliberately does *not* create a model body,
//! logical session, worktree, server, watcher, GPU lease, device dispatch,
//! token execution, benchmark, tournament receipt, or manager selection.
//!
//! Each lane has one primary that may accept a fully reviewed proposal and
//! mutate/promote only that lane's champion.  Its helper and the opponent
//! receive only proposal, protected-review, falsification, and isolated-test
//! actions.  The opponent sandbox is separate from either candidate world's
//! private namespaces, so this grant cannot become a cross-lane private read.
//!
//! Example:
//!
//! ```text
//! cargo run -p hawking-core --example ascension_paired_cognition_mutation_authority_contract -- \
//!   --input /absolute/path/PAIRED_COGNITION_MUTATION_INPUT.json \
//!   --out /absolute/new/path/PAIRED_COGNITION_MUTATION_AUTHORITY.json
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

const INPUT_SCHEMA: &str = "hawking.ascension.paired_cognition_mutation_authority_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_proposal_review_falsification_primary_acceptance_authority.v1";
const LANE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.paired_cognition_lane_namespace_mission_authority.v1";
const LANE_AUTHORITY_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_TWO_SEALED_CANDIDATE_WORLDS_NO_RUNTIME_SERVER_OR_TOURNAMENT";
const STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_PRIMARY_ONLY_CHAMPION_MUTATION_PROMOTION_NO_MANAGER_OR_TOURNAMENT_SELECTION";

const QWEN30: &str = "qwen30";
const QWEN80: &str = "qwen80";

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
const ACCEPTANCE_EVIDENCE: [&str; 4] = [
    "sealed_proposal",
    "protected_adversarial_review",
    "independent_falsification",
    "independent_verifier_receipt",
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
    protected_adversarial_reviews: String,
    independent_verification: String,
    hard_gate_conjunction: String,
    post_tg3_freeze: String,
    solo_manager_evaluation: String,
    symmetric_orchestrator_evaluation: String,
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
    protected_adversarial_review_required_before_promotion: bool,
    independent_verifier_required_before_promotion: bool,
    hard_gate_conjunction_required_before_activation: bool,
    post_tg3_freeze_required_before_final_evaluation: bool,
    solo_and_symmetric_orchestrator_evaluations_required: bool,
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
    lane_worlds: Vec<BoundLane>,
    cross_lane_read_policy: CrossLaneReadPolicy,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct LaneActionPolicy {
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
    primary_champion_namespace: String,
    primary_acceptance_request_namespace: String,
    helper_isolated_worktree_namespace: String,
    opponent_isolated_worktree_namespace: String,
    proposal_namespace: String,
    protected_review_namespace: String,
    falsification_namespace: String,
    required_primary_acceptance_evidence: Vec<String>,
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
    protected_tournament_receipt_namespace: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct FinalSelectionReservation {
    post_tg3_freeze_required: bool,
    protected_final_selection_required: bool,
    solo_manager_evaluation_required: bool,
    symmetric_orchestrator_evaluation_required: bool,
    manager_selection_authorized_by_this_contract: bool,
    tournament_selection_authorized_by_this_contract: bool,
    final_selection_authorized_by_this_contract: bool,
    post_tg3_freeze_namespace: String,
    protected_final_selection_namespace: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Input {
    schema: String,
    lane_authority: SealedDocumentBinding,
    lane_action_policies: Vec<LaneActionPolicy>,
    protected_record_policy: ProtectedRecordPolicy,
    final_selection_reservation: FinalSelectionReservation,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct EvidenceDigest {
    path: String,
    document_schema: String,
    document_status: String,
    document_sha256: String,
    document_seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    completed_lane_authority_bound: bool,
    exactly_two_lane_action_policies: bool,
    qwen80_primary_qwen30_helper_and_opponent: bool,
    qwen30_primary_qwen80_helper_and_opponent: bool,
    primary_only_own_champion_acceptance_mutation_and_promotion: bool,
    helper_and_opponent_limited_to_propose_review_falsify_and_isolated_test: bool,
    helper_worktree_is_its_lane_private_worktree: bool,
    opponent_sandbox_is_outside_all_private_lane_namespaces: bool,
    all_cross_lane_private_reads_denied: bool,
    protected_evidence_mission_and_tournament_receipts_immutable: bool,
    self_promotion_and_manager_selection_denied: bool,
    post_tg3_freeze_and_protected_final_selection_reserved: bool,
    no_runtime_or_tournament_authority: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    bound_lane_authority: EvidenceDigest,
    lane_action_policies: Vec<LaneActionPolicy>,
    protected_record_policy: ProtectedRecordPolicy,
    final_selection_reservation: FinalSelectionReservation,
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

/// The policy grammar contains no floats, and `serde_json::Map` is sorted in
/// this workspace.  This is the same compact/sorted JSON seal family used by
/// the paired-lane authority it consumes.
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

fn namespace_entries(namespaces: &PrivateNamespaces) -> Vec<&str> {
    vec![
        &namespaces.mission,
        &namespaces.experiments,
        &namespaces.receipts,
        &namespaces.worktree,
        &namespaces.sessions,
        &namespaces.frontier,
        &namespaces.patches,
        &namespaces.scores,
        &namespaces.protected_adversarial_reviews,
        &namespaces.independent_verification,
        &namespaces.hard_gate_conjunction,
        &namespaces.post_tg3_freeze,
        &namespaces.solo_manager_evaluation,
        &namespaces.symmetric_orchestrator_evaluation,
    ]
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
            "{label} may not authorize model processes, servers, ports, GPU leases, tournament mutations, or activation"
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
            "{label} must permit only verifier-released generic Knowledge Plane material"
        ));
    }
    Ok(())
}

fn validate_mission_authority(authority: &MissionAuthority, label: &str) -> Result<(), String> {
    if !authority.primary_candidate_mutation_authority
        || !authority.primary_candidate_promotion_authority
        || !authority.helper_may_inspect_and_critique
        || !authority.helper_may_propose_or_test_in_private_worktree
    {
        return Err(format!(
            "{label} lacks the required primary/helper role split"
        ));
    }
    if authority.helper_may_mutate_primary_champion
        || authority.helper_may_promote_primary_champion
        || authority.opposite_lane_may_mutate_primary_champion
        || authority.primary_or_helper_may_self_score
    {
        return Err(format!(
            "{label} permits non-primary champion control or self-scoring"
        ));
    }
    if !authority.protected_adversarial_review_required_before_promotion
        || !authority.independent_verifier_required_before_promotion
        || !authority.hard_gate_conjunction_required_before_activation
        || !authority.post_tg3_freeze_required_before_final_evaluation
        || !authority.solo_and_symmetric_orchestrator_evaluations_required
    {
        return Err(format!(
            "{label} must retain review, verifier, hard-gate, freeze, and both final evaluation reservations"
        ));
    }
    Ok(())
}

fn binding_digest(
    binding: &SealedDocumentBinding,
) -> Result<(EvidenceDigest, LaneAuthorityDocument), String> {
    require_absolute_path(&binding.path, "lane_authority.path")?;
    if !is_lower_sha256(&binding.document_sha256) {
        return Err("lane_authority.document_sha256 must be a lowercase SHA-256".into());
    }
    let actual = sha256_json(&binding.document)?;
    if actual != binding.document_sha256 {
        return Err("lane_authority.document_sha256 does not bind its embedded document".into());
    }
    let seal = verify_sealed_object(&binding.document, "lane_authority.document")?;
    let document: LaneAuthorityDocument = serde_json::from_value(binding.document.clone())
        .map_err(|error| format!("lane_authority.document has the wrong grammar: {error}"))?;
    if document.schema != LANE_AUTHORITY_SCHEMA || document.status != LANE_AUTHORITY_STATUS {
        return Err("lane_authority.document is not the completed paired lane authority".into());
    }
    if !document.prepared
        || document.paired_candidate_worlds_active
        || !document.no_new_physical_model_process_authority
    {
        return Err("lane_authority.document must be prepared, inactive, and process-free".into());
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
    Ok((
        EvidenceDigest {
            path: binding.path.clone(),
            document_schema: document.schema.clone(),
            document_status: document.status.clone(),
            document_sha256: binding.document_sha256.clone(),
            document_seal_sha256: seal,
        },
        document,
    ))
}

fn expected_actor_ids(primary: &str, helper: &str) -> (String, String, String) {
    (
        format!("{primary}-primary-in-{primary}-lane"),
        format!("{helper}-helper-in-{primary}-lane"),
        format!("{helper}-opponent-reviewer-in-{primary}-lane"),
    )
}

fn exact_actions(actual: &[String], required: &[&str], label: &str) -> Result<(), String> {
    let observed: BTreeSet<&str> = actual.iter().map(String::as_str).collect();
    let expected: BTreeSet<&str> = required.iter().copied().collect();
    if observed.len() != actual.len() {
        return Err(format!("{label} contains duplicate action grants"));
    }
    if observed != expected {
        return Err(format!(
            "{label} must grant exactly {:?}, observed {:?}",
            expected, observed
        ));
    }
    Ok(())
}

fn validate_lane_document(
    document: &LaneAuthorityDocument,
) -> Result<BTreeMap<String, BoundLane>, String> {
    if document.lane_worlds.len() != 2 {
        return Err("completed lane authority must contain exactly two candidate worlds".into());
    }
    let mut result = BTreeMap::new();
    let mut all_private_namespaces = BTreeSet::new();
    for lane in &document.lane_worlds {
        if lane.primary_model_key != QWEN30 && lane.primary_model_key != QWEN80 {
            return Err(format!(
                "unsupported lane primary {:?}",
                lane.primary_model_key
            ));
        }
        let expected_helper = if lane.primary_model_key == QWEN30 {
            QWEN80
        } else {
            QWEN30
        };
        if lane.helper_model_key != expected_helper {
            return Err(format!("{} has the wrong helper", lane.lane_id));
        }
        if lane.lane_id.trim().is_empty() || result.contains_key(&lane.lane_id) {
            return Err("completed lane authority has an empty or duplicate lane id".into());
        }
        validate_mission_authority(&lane.mission_authority, &lane.lane_id)?;
        for namespace in namespace_entries(&lane.private_namespaces) {
            require_sealed_namespace(namespace, &format!("{}.private_namespaces", lane.lane_id))?;
            if !all_private_namespaces.insert(namespace.to_owned()) {
                return Err(
                    "completed lane authority aliases a private namespace across candidate worlds"
                        .into(),
                );
            }
        }
        result.insert(lane.lane_id.clone(), lane.clone());
    }
    let primary_models: BTreeSet<&str> = result
        .values()
        .map(|lane| lane.primary_model_key.as_str())
        .collect();
    if primary_models != BTreeSet::from([QWEN30, QWEN80]) {
        return Err(
            "completed lane authority must contain exactly one Q30 and one Q80 primary".into(),
        );
    }
    Ok(result)
}

fn all_private_namespaces(lanes: &BTreeMap<String, BoundLane>) -> BTreeSet<String> {
    lanes
        .values()
        .flat_map(|lane| namespace_entries(&lane.private_namespaces))
        .map(ToOwned::to_owned)
        .collect()
}

fn validate_policy_namespaces(
    policy: &LaneActionPolicy,
    lane: &BoundLane,
    all_private: &BTreeSet<String>,
    all_action_namespaces: &mut BTreeSet<String>,
) -> Result<(), String> {
    if policy.primary_champion_namespace != lane.private_namespaces.patches {
        return Err(format!(
            "{} primary_champion_namespace must bind only its own lane patch/champion namespace",
            policy.lane_id
        ));
    }
    if policy.helper_isolated_worktree_namespace != lane.private_namespaces.worktree {
        return Err(format!(
            "{} helper isolated worktree must be that lane's already-isolated private worktree",
            policy.lane_id
        ));
    }
    let novel_namespaces = [
        (
            "primary_acceptance_request_namespace",
            &policy.primary_acceptance_request_namespace,
        ),
        (
            "opponent_isolated_worktree_namespace",
            &policy.opponent_isolated_worktree_namespace,
        ),
        ("proposal_namespace", &policy.proposal_namespace),
        (
            "protected_review_namespace",
            &policy.protected_review_namespace,
        ),
        ("falsification_namespace", &policy.falsification_namespace),
    ];
    for (kind, namespace) in novel_namespaces {
        require_sealed_namespace(namespace, &format!("{}.{}", policy.lane_id, kind))?;
        if all_private.contains(namespace) {
            return Err(format!(
                "{}.{} may not alias a private candidate-world namespace",
                policy.lane_id, kind
            ));
        }
        if !all_action_namespaces.insert(namespace.clone()) {
            return Err(format!(
                "action namespace {:?} is reused across roles or lanes",
                namespace
            ));
        }
    }
    Ok(())
}

fn validate_lane_action_policies(
    policies: &[LaneActionPolicy],
    lanes: &BTreeMap<String, BoundLane>,
) -> Result<Vec<LaneActionPolicy>, String> {
    if policies.len() != 2 {
        return Err("exactly two lane action policies are required".into());
    }
    let all_private = all_private_namespaces(lanes);
    let mut seen_lanes = BTreeSet::new();
    let mut all_actor_ids = BTreeSet::new();
    let mut all_action_namespaces = BTreeSet::new();
    let mut validated = Vec::new();
    for policy in policies {
        let lane = lanes
            .get(&policy.lane_id)
            .ok_or_else(|| format!("{} is not a bound candidate world", policy.lane_id))?;
        if !seen_lanes.insert(policy.lane_id.as_str()) {
            return Err(format!("duplicate action policy for {}", policy.lane_id));
        }
        if policy.primary_model_key != lane.primary_model_key
            || policy.helper_model_key != lane.helper_model_key
            || policy.opponent_model_key != lane.helper_model_key
        {
            return Err(format!(
                "{} action policy does not match its sealed primary/helper/opponent identities",
                policy.lane_id
            ));
        }
        let expected_actors = expected_actor_ids(&lane.primary_model_key, &lane.helper_model_key);
        if (
            policy.primary_actor_id.as_str(),
            policy.helper_actor_id.as_str(),
            policy.opponent_actor_id.as_str(),
        ) != (
            expected_actors.0.as_str(),
            expected_actors.1.as_str(),
            expected_actors.2.as_str(),
        ) {
            return Err(format!(
                "{} action policy has unbound logical actor ids",
                policy.lane_id
            ));
        }
        for actor in [
            &policy.primary_actor_id,
            &policy.helper_actor_id,
            &policy.opponent_actor_id,
        ] {
            if !all_actor_ids.insert(actor.as_str()) {
                return Err(format!("logical actor id {:?} is reused", actor));
            }
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
        exact_actions(
            &policy.required_primary_acceptance_evidence,
            &ACCEPTANCE_EVIDENCE,
            &format!("{}.required_primary_acceptance_evidence", policy.lane_id),
        )?;
        validate_policy_namespaces(policy, lane, &all_private, &mut all_action_namespaces)?;
        validated.push(policy.clone());
    }
    if seen_lanes.len() != lanes.len() {
        return Err("every sealed candidate world must have one action policy".into());
    }
    validated.sort_by(|left, right| left.lane_id.cmp(&right.lane_id));
    Ok(validated)
}

fn validate_protected_record_policy(
    policy: &ProtectedRecordPolicy,
    lanes: &BTreeMap<String, BoundLane>,
    action_policies: &[LaneActionPolicy],
) -> Result<(), String> {
    if !policy.evidence_receipts_immutable
        || !policy.mission_records_immutable
        || !policy.tournament_receipts_immutable
        || policy.all_roles_may_rewrite_evidence_receipts
        || policy.all_roles_may_rewrite_mission_records
        || policy.all_roles_may_rewrite_tournament_receipts
        || policy.all_roles_may_delete_protected_records
        || policy.cross_lane_private_record_release_allowed
    {
        return Err(
            "evidence, mission, and tournament receipts must remain immutable and private".into(),
        );
    }
    require_sealed_namespace(
        &policy.protected_tournament_receipt_namespace,
        "protected_tournament_receipt_namespace",
    )?;
    let private = all_private_namespaces(lanes);
    if private.contains(&policy.protected_tournament_receipt_namespace) {
        return Err(
            "protected_tournament_receipt_namespace may not alias a lane-private namespace".into(),
        );
    }
    if action_policies.iter().any(|action| {
        action.primary_acceptance_request_namespace == policy.protected_tournament_receipt_namespace
            || action.opponent_isolated_worktree_namespace
                == policy.protected_tournament_receipt_namespace
            || action.proposal_namespace == policy.protected_tournament_receipt_namespace
            || action.protected_review_namespace == policy.protected_tournament_receipt_namespace
            || action.falsification_namespace == policy.protected_tournament_receipt_namespace
    }) {
        return Err("an action workspace may not alias protected tournament receipts".into());
    }
    Ok(())
}

fn validate_final_selection_reservation(
    reservation: &FinalSelectionReservation,
    lanes: &BTreeMap<String, BoundLane>,
    action_policies: &[LaneActionPolicy],
    protected: &ProtectedRecordPolicy,
) -> Result<(), String> {
    if !reservation.post_tg3_freeze_required
        || !reservation.protected_final_selection_required
        || !reservation.solo_manager_evaluation_required
        || !reservation.symmetric_orchestrator_evaluation_required
        || reservation.manager_selection_authorized_by_this_contract
        || reservation.tournament_selection_authorized_by_this_contract
        || reservation.final_selection_authorized_by_this_contract
    {
        return Err("post-TG3 freeze and protected final selection must remain reserved outside this contract".into());
    }
    for (label, namespace) in [
        (
            "post_tg3_freeze_namespace",
            &reservation.post_tg3_freeze_namespace,
        ),
        (
            "protected_final_selection_namespace",
            &reservation.protected_final_selection_namespace,
        ),
    ] {
        require_sealed_namespace(namespace, label)?;
        if all_private_namespaces(lanes).contains(namespace) {
            return Err(format!("{label} may not alias a lane-private namespace"));
        }
        if namespace == &protected.protected_tournament_receipt_namespace {
            return Err(format!(
                "{label} may not alias protected tournament receipts"
            ));
        }
        if action_policies.iter().any(|action| {
            action.primary_acceptance_request_namespace == *namespace
                || action.opponent_isolated_worktree_namespace == *namespace
                || action.proposal_namespace == *namespace
                || action.protected_review_namespace == *namespace
                || action.falsification_namespace == *namespace
        }) {
            return Err(format!("{label} may not alias an actionable workspace"));
        }
    }
    if reservation.post_tg3_freeze_namespace == reservation.protected_final_selection_namespace {
        return Err("freeze and protected final-selection namespaces must be distinct".into());
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
        return Err("input.seal_sha256 must be a lowercase SHA-256".into());
    }
    let (bound_lane_authority, lane_document) = binding_digest(&input.lane_authority)?;
    let lanes = validate_lane_document(&lane_document)?;
    let policies = validate_lane_action_policies(&input.lane_action_policies, &lanes)?;
    validate_protected_record_policy(&input.protected_record_policy, &lanes, &policies)?;
    validate_final_selection_reservation(
        &input.final_selection_reservation,
        &lanes,
        &policies,
        &input.protected_record_policy,
    )?;
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;

    let by_primary: BTreeMap<&str, &LaneActionPolicy> = policies
        .iter()
        .map(|policy| (policy.primary_model_key.as_str(), policy))
        .collect();
    let qwen80 = by_primary
        .get(QWEN80)
        .expect("validated Q80 primary policy");
    let qwen30 = by_primary
        .get(QWEN30)
        .expect("validated Q30 primary policy");
    let primary_only = policies.iter().all(|policy| {
        exact_actions(
            &policy.primary_allowed_actions,
            &PRIMARY_ACTIONS,
            "focused-primary",
        )
        .is_ok()
            && policy.primary_champion_namespace
                == lanes
                    .get(&policy.lane_id)
                    .expect("validated lane")
                    .private_namespaces
                    .patches
    });
    let helper_opponent_limited = policies.iter().all(|policy| {
        exact_actions(
            &policy.helper_allowed_actions,
            &WORK_ACTIONS,
            "focused-helper",
        )
        .is_ok()
            && exact_actions(
                &policy.opponent_allowed_actions,
                &WORK_ACTIONS,
                "focused-opponent",
            )
            .is_ok()
    });
    let helper_worktree_bound = policies.iter().all(|policy| {
        policy.helper_isolated_worktree_namespace
            == lanes
                .get(&policy.lane_id)
                .expect("validated lane")
                .private_namespaces
                .worktree
    });
    let private = all_private_namespaces(&lanes);
    let opponent_sandbox_isolated = policies
        .iter()
        .all(|policy| !private.contains(&policy.opponent_isolated_worktree_namespace));
    let cross_reads_denied = validate_cross_lane_policy(
        &lane_document.cross_lane_read_policy,
        "focused-cross-lane-policy",
    )
    .is_ok()
        && !input
            .protected_record_policy
            .cross_lane_private_record_release_allowed;
    let protected_immutable = input.protected_record_policy.evidence_receipts_immutable
        && input.protected_record_policy.mission_records_immutable
        && input.protected_record_policy.tournament_receipts_immutable
        && !input
            .protected_record_policy
            .all_roles_may_rewrite_evidence_receipts
        && !input
            .protected_record_policy
            .all_roles_may_rewrite_mission_records
        && !input
            .protected_record_policy
            .all_roles_may_rewrite_tournament_receipts
        && !input
            .protected_record_policy
            .all_roles_may_delete_protected_records;
    let manager_denied = !input
        .final_selection_reservation
        .manager_selection_authorized_by_this_contract
        && !input
            .final_selection_reservation
            .tournament_selection_authorized_by_this_contract
        && !input
            .final_selection_reservation
            .final_selection_authorized_by_this_contract;
    let final_selection_reserved = input.final_selection_reservation.post_tg3_freeze_required
        && input
            .final_selection_reservation
            .protected_final_selection_required
        && input
            .final_selection_reservation
            .solo_manager_evaluation_required
        && input
            .final_selection_reservation
            .symmetric_orchestrator_evaluation_required
        && manager_denied;
    let no_runtime_or_tournament_authority =
        validate_authority_boundary(&input.authority_boundary, "focused-authority-boundary")
            .is_ok();
    let execution_boundary_cpu_only =
        validate_execution_boundary(&input.execution_boundary, "focused-execution-boundary")
            .is_ok();

    let focused_checks = FocusedChecks {
        completed_lane_authority_bound: true,
        exactly_two_lane_action_policies: policies.len() == 2,
        qwen80_primary_qwen30_helper_and_opponent: qwen80.helper_model_key == QWEN30
            && qwen80.opponent_model_key == QWEN30,
        qwen30_primary_qwen80_helper_and_opponent: qwen30.helper_model_key == QWEN80
            && qwen30.opponent_model_key == QWEN80,
        primary_only_own_champion_acceptance_mutation_and_promotion: primary_only,
        helper_and_opponent_limited_to_propose_review_falsify_and_isolated_test:
            helper_opponent_limited,
        helper_worktree_is_its_lane_private_worktree: helper_worktree_bound,
        opponent_sandbox_is_outside_all_private_lane_namespaces: opponent_sandbox_isolated,
        all_cross_lane_private_reads_denied: cross_reads_denied,
        protected_evidence_mission_and_tournament_receipts_immutable: protected_immutable,
        self_promotion_and_manager_selection_denied: manager_denied,
        post_tg3_freeze_and_protected_final_selection_reserved: final_selection_reserved,
        no_runtime_or_tournament_authority,
        execution_boundary_cpu_only,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status: STATUS,
        prepared: true,
        paired_candidate_worlds_active: false,
        bound_lane_authority,
        lane_action_policies: policies,
        protected_record_policy: input.protected_record_policy,
        final_selection_reservation: input.final_selection_reservation,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only action-policy authority, not a model runtime or live lane controller.",
            "It starts no Q30 or Q80 model process, server, watcher, logical session, port listener, GPU lease, or device dispatch.",
            "It does not execute a model token, HCLI request, TPS/TG measurement, experiment, benchmark, or tournament mutation.",
            "Only a lane primary may accept a proposal and mutate/promote that same lane's champion after all named protected evidence is present.",
            "Helpers and opponents can only propose, protected-review, falsify, or test in their bound isolated worktrees; they cannot self-promote, mutate a champion, or select a manager.",
            "Evidence, mission records, and tournament receipts remain immutable and are never an action workspace.",
            "Post-TG3 freeze and solo plus symmetric manager-as-orchestrator final selection remain reserved for a later protected authority.",
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
            .map_err(|error| format!("mutation-authority report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("mutation-authority report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_paired_cognition_mutation_authority_contract --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        build_report(input).map_err(|error| format!("authority validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_paired_cognition_mutation_authority_contract: {error}");
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
            protected_adversarial_reviews: format!("sealed://{lane}/protected-adversarial-reviews"),
            independent_verification: format!("sealed://{lane}/independent-verification"),
            hard_gate_conjunction: format!("sealed://{lane}/hard-gate-conjunction"),
            post_tg3_freeze: format!("sealed://{lane}/post-tg3-freeze"),
            solo_manager_evaluation: format!("sealed://{lane}/solo-manager-evaluation"),
            symmetric_orchestrator_evaluation: format!(
                "sealed://{lane}/symmetric-orchestrator-evaluation"
            ),
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
            protected_adversarial_review_required_before_promotion: true,
            independent_verifier_required_before_promotion: true,
            hard_gate_conjunction_required_before_activation: true,
            post_tg3_freeze_required_before_final_evaluation: true,
            solo_and_symmetric_orchestrator_evaluations_required: true,
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

    fn action_policy(primary: &str, helper: &str) -> LaneActionPolicy {
        let lane_id = format!("{primary}-candidate-world");
        let (primary_actor_id, helper_actor_id, opponent_actor_id) =
            expected_actor_ids(primary, helper);
        LaneActionPolicy {
            lane_id,
            primary_model_key: primary.into(),
            helper_model_key: helper.into(),
            opponent_model_key: helper.into(),
            primary_actor_id,
            helper_actor_id,
            opponent_actor_id,
            primary_allowed_actions: PRIMARY_ACTIONS.iter().map(|item| (*item).into()).collect(),
            helper_allowed_actions: WORK_ACTIONS.iter().map(|item| (*item).into()).collect(),
            opponent_allowed_actions: WORK_ACTIONS.iter().map(|item| (*item).into()).collect(),
            primary_champion_namespace: format!("sealed://{primary}-lane/patches"),
            primary_acceptance_request_namespace: format!(
                "sealed://{primary}-lane/primary-acceptance-request"
            ),
            helper_isolated_worktree_namespace: format!("sealed://{primary}-lane/worktree"),
            opponent_isolated_worktree_namespace: format!(
                "sealed://{primary}-lane/opponent-isolated-worktree"
            ),
            proposal_namespace: format!("sealed://{primary}-lane/proposals"),
            protected_review_namespace: format!("sealed://{primary}-lane/protected-reviews"),
            falsification_namespace: format!("sealed://{primary}-lane/falsifications"),
            required_primary_acceptance_evidence: ACCEPTANCE_EVIDENCE
                .iter()
                .map(|item| (*item).into())
                .collect(),
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
            protected_tournament_receipt_namespace: "sealed://tournament/protected-receipts".into(),
        }
    }

    fn final_selection_reservation() -> FinalSelectionReservation {
        FinalSelectionReservation {
            post_tg3_freeze_required: true,
            protected_final_selection_required: true,
            solo_manager_evaluation_required: true,
            symmetric_orchestrator_evaluation_required: true,
            manager_selection_authorized_by_this_contract: false,
            tournament_selection_authorized_by_this_contract: false,
            final_selection_authorized_by_this_contract: false,
            post_tg3_freeze_namespace: "sealed://tournament/post-tg3-freeze".into(),
            protected_final_selection_namespace: "sealed://tournament/protected-final-selection"
                .into(),
        }
    }

    fn input_fixture() -> Input {
        Input {
            schema: INPUT_SCHEMA.into(),
            lane_authority: lane_authority_binding(),
            lane_action_policies: vec![
                action_policy(QWEN80, QWEN30),
                action_policy(QWEN30, QWEN80),
            ],
            protected_record_policy: protected_record_policy(),
            final_selection_reservation: final_selection_reservation(),
            authority_boundary: authority_boundary(),
            execution_boundary: execution_boundary(),
            seal_sha256: sha('0'),
        }
    }

    fn seal_input(input: &Input) -> Value {
        let mut value = serde_json::to_value(input).unwrap();
        value.as_object_mut().unwrap().remove("seal_sha256");
        seal_value(value).unwrap()
    }

    #[test]
    fn correct_policy_binds_lane_authority_and_stays_prepared_cpu_only() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, STATUS);
        assert!(report.prepared);
        assert!(!report.paired_candidate_worlds_active);
        assert!(report.focused_checks.completed_lane_authority_bound);
        assert!(
            report
                .focused_checks
                .primary_only_own_champion_acceptance_mutation_and_promotion
        );
        assert!(
            report
                .focused_checks
                .helper_and_opponent_limited_to_propose_review_falsify_and_isolated_test
        );
        assert!(
            report
                .focused_checks
                .protected_evidence_mission_and_tournament_receipts_immutable
        );
        assert!(report.focused_checks.execution_boundary_cpu_only);
        assert!(!report.execution_boundary.metal_device_or_dispatch_performed);
        assert!(!report.execution_boundary.runtime_watcher_or_server_started);
    }

    #[test]
    fn completed_sealed_lane_authority_is_required() {
        let mut tampered = input_fixture();
        tampered.lane_authority.document["prepared"] = Value::Bool(false);
        assert!(build_report(tampered).is_err());

        let mut wrong_schema = input_fixture();
        wrong_schema.lane_authority.document["schema"] = Value::String("other.schema".into());
        let document = wrong_schema.lane_authority.document.clone();
        wrong_schema.lane_authority.document = seal_value(document).unwrap();
        wrong_schema.lane_authority.document_sha256 =
            sha256_json(&wrong_schema.lane_authority.document).unwrap();
        assert!(build_report(wrong_schema).is_err());
    }

    #[test]
    fn helper_or_opponent_self_promotion_mutation_or_acceptance_is_rejected() {
        for forbidden in [
            "accept_verified_proposal_for_own_champion",
            "mutate_own_champion",
            "promote_own_champion",
        ] {
            let mut helper = input_fixture();
            helper.lane_action_policies[0]
                .helper_allowed_actions
                .push(forbidden.into());
            assert!(
                build_report(helper).is_err(),
                "helper {forbidden} must fail"
            );

            let mut opponent = input_fixture();
            opponent.lane_action_policies[1]
                .opponent_allowed_actions
                .push(forbidden.into());
            assert!(
                build_report(opponent).is_err(),
                "opponent {forbidden} must fail"
            );
        }
    }

    #[test]
    fn primary_cannot_target_another_lane_champion_or_accept_without_all_evidence() {
        let mut cross_lane = input_fixture();
        cross_lane.lane_action_policies[0].primary_champion_namespace =
            "sealed://qwen30-lane/patches".into();
        assert!(build_report(cross_lane).is_err());

        let mut insufficient_evidence = input_fixture();
        insufficient_evidence.lane_action_policies[0]
            .required_primary_acceptance_evidence
            .pop();
        assert!(build_report(insufficient_evidence).is_err());
    }

    #[test]
    fn cross_lane_private_reads_and_opponent_private_worktree_aliases_are_rejected() {
        let mut private_read = input_fixture();
        private_read.lane_authority.document["cross_lane_read_policy"]
            ["allow_cross_lane_receipt_reads"] = Value::Bool(true);
        let document = private_read.lane_authority.document.clone();
        private_read.lane_authority.document = seal_value(document).unwrap();
        private_read.lane_authority.document_sha256 =
            sha256_json(&private_read.lane_authority.document).unwrap();
        assert!(build_report(private_read).is_err());

        let mut aliased_sandbox = input_fixture();
        aliased_sandbox.lane_action_policies[0].opponent_isolated_worktree_namespace =
            "sealed://qwen30-lane/worktree".into();
        assert!(build_report(aliased_sandbox).is_err());
    }

    #[test]
    fn protected_records_and_manager_selection_are_rejected_when_unprotected() {
        let mut mutable_records = input_fixture();
        mutable_records
            .protected_record_policy
            .all_roles_may_rewrite_tournament_receipts = true;
        assert!(build_report(mutable_records).is_err());

        let mut manager_selection = input_fixture();
        manager_selection
            .final_selection_reservation
            .manager_selection_authorized_by_this_contract = true;
        assert!(build_report(manager_selection).is_err());
    }

    #[test]
    fn post_tg3_freeze_and_protected_final_selection_are_required() {
        let mut no_freeze = input_fixture();
        no_freeze
            .final_selection_reservation
            .post_tg3_freeze_required = false;
        assert!(build_report(no_freeze).is_err());

        let mut alias = input_fixture();
        alias
            .final_selection_reservation
            .protected_final_selection_namespace = "sealed://tournament/protected-receipts".into();
        assert!(build_report(alias).is_err());
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
    fn tampered_outer_input_seal_is_rejected_before_authority_evaluation() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("tampered-input.json");
        let output_path = directory.path().join("out.json");
        let mut sealed = seal_input(&input_fixture());
        sealed["final_selection_reservation"]["final_selection_authorized_by_this_contract"] =
            Value::Bool(true);
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        assert!(run(Args {
            input: input_path,
            out: output_path,
        })
        .is_err());
    }
}
