//! Sealed-input paired-cognition lane namespace and mission authority.
//!
//! This is a CPU-only control-plane contract for the two *candidate worlds*
//! described by Ascension V3:
//!
//! ```text
//! Q80 lane: Q80 primary + Q30 helper, private Q80 candidate world
//! Q30 lane: Q30 primary + Q80 helper, private Q30 candidate world
//! ```
//!
//! It deliberately creates neither a model body nor a logical session.  It
//! only consumes a caller-supplied sealed input that embeds the already sealed
//! activation, memory, and session authority documents for both models.  The
//! contract binds their identity hashes, checks that every authority agrees on
//! one Q30 body at `127.0.0.1:18430` and one Q80 body at
//! `127.0.0.1:18480`, and publishes the namespace/mission rules a later
//! controller must obey.
//!
//! A successful result is always `PREPARED/NOT_ACTIVE`.  In particular, this
//! program has no authority to start a server, create another physical model
//! process, bind a port, obtain a GPU lease, execute a token, mutate a
//! tournament state, or activate the paired worlds.  An attempted activation
//! below the two independently sealed TG10 (`>=100 BASE_TRUE_TPS`) conditions
//! is rejected before an output is written; an attempted activation above the
//! conditions is also rejected because activation belongs to the existing
//! protected controller, not this namespace contract.
//!
//! The inner evidence documents are embedded instead of reopened by path.  A
//! path plus a canonical JSON digest records the intended external authority,
//! while the embedded sealed object makes this contract deterministic and
//! prevents a later path substitution from changing what was evaluated.
//!
//! Example:
//!
//! ```text
//! cargo run -p hawking-core --example ascension_paired_cognition_lane_authority_contract -- \
//!   --input /absolute/path/PAIRED_COGNITION_LANE_INPUT.json \
//!   --out /absolute/new/path/PAIRED_COGNITION_LANE_AUTHORITY.json
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

const INPUT_SCHEMA: &str = "hawking.ascension.paired_cognition_lane_authority_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_lane_namespace_mission_authority.v1";
const STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_TWO_SEALED_CANDIDATE_WORLDS_NO_RUNTIME_SERVER_OR_TOURNAMENT";

const QWEN30: &str = "qwen30";
const QWEN80: &str = "qwen80";
const QWEN30_HOST: &str = "127.0.0.1";
const QWEN80_HOST: &str = "127.0.0.1";
const QWEN30_PORT: u16 = 18_430;
const QWEN80_PORT: u16 = 18_480;
const TG10_BASE_TRUE_TPS: u16 = 100;

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq, Ord, PartialOrd)]
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ContractBinding {
    path: String,
    document_sha256: String,
    document: Value,
    topology_assertion: TopologyAssertion,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Tg10Authority {
    required_base_true_tps: u16,
    operational_pass: bool,
    coherent_hcli_pass: bool,
    complete_token_path_measured: bool,
    fallback_count: usize,
    median_base_true_tps: Option<f64>,
    receipt_seal_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ActivationBinding {
    contract: ContractBinding,
    tg10: Tg10Authority,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct ModelAuthority {
    model_key: String,
    activation: ActivationBinding,
    memory: ContractBinding,
    session: ContractBinding,
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
struct LaneDefinition {
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

#[derive(Clone, Debug, Deserialize, Serialize)]
struct Input {
    schema: String,
    activation_requested: bool,
    model_authorities: Vec<ModelAuthority>,
    lanes: Vec<LaneDefinition>,
    cross_lane_read_policy: CrossLaneReadPolicy,
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
struct ModelContractBindingReport {
    model_key: String,
    topology: TopologyAssertion,
    activation: EvidenceDigest,
    memory: EvidenceDigest,
    session: EvidenceDigest,
    tg10: Tg10Authority,
}

#[derive(Clone, Debug, Serialize)]
struct PhysicalBodyTopology {
    model_key: String,
    resident_model_processes: usize,
    immutable_weight_copies: usize,
    endpoint: Endpoint,
    logical_session_policy: String,
}

#[derive(Clone, Debug, Serialize)]
struct ActivationGateReport {
    activation_requested: bool,
    required_base_true_tps: u16,
    qwen30_tg10_operational_pass: bool,
    qwen80_tg10_operational_pass: bool,
    both_tg10_operational_passes_present: bool,
    paired_world_activation_authorized_by_this_contract: bool,
    blockers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    exact_two_candidate_worlds: bool,
    qwen80_primary_qwen30_helper: bool,
    qwen30_primary_qwen80_helper: bool,
    all_private_namespaces_distinct: bool,
    all_cross_lane_private_reads_denied: bool,
    primary_only_champion_mutation_and_promotion: bool,
    self_scoring_prohibited: bool,
    deferred_adversarial_verifier_gate_freeze_and_evaluation_namespaces_bound: bool,
    exactly_one_qwen30_body: bool,
    exactly_one_qwen80_body: bool,
    qwen30_qwen80_ports_distinct: bool,
    no_new_physical_process_or_server_authority: bool,
    below_tg10_activation_rejected: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    prepared: bool,
    paired_candidate_worlds_active: bool,
    no_new_physical_model_process_authority: bool,
    model_contract_bindings: Vec<ModelContractBindingReport>,
    physical_body_topology: Vec<PhysicalBodyTopology>,
    lane_worlds: Vec<LaneDefinition>,
    cross_lane_read_policy: CrossLaneReadPolicy,
    authority_boundary: AuthorityBoundary,
    activation_gate: ActivationGateReport,
    focused_checks: FocusedChecks,
    execution_boundary: ExecutionBoundary,
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

/// Verify the same compact/sorted JSON seal family used by the surrounding
/// Ascension contracts.  `serde_json::Map` is key-sorted in this workspace,
/// so the canonical representation agrees with Python's `sort_keys=True`
/// documents for this non-floating policy grammar.
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

fn require_absolute_path(path: &str, label: &str) -> Result<(), String> {
    if !Path::new(path).is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    Ok(())
}

fn nonempty(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

fn binding_digest(binding: &ContractBinding, label: &str) -> Result<EvidenceDigest, String> {
    require_absolute_path(&binding.path, &format!("{label}.path"))?;
    if !is_lower_sha256(&binding.document_sha256) {
        return Err(format!(
            "{label}.document_sha256 must be a lowercase SHA-256"
        ));
    }
    let actual_document_sha256 = sha256_json(&binding.document)?;
    if binding.document_sha256 != actual_document_sha256 {
        return Err(format!(
            "{label}.document_sha256 does not bind the embedded sealed document"
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
        .ok_or_else(|| format!("{label}.document.schema must be a non-empty string"))?;
    let status = object
        .get("status")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.document.status must be a non-empty string"))?;
    nonempty(schema, &format!("{label}.document.schema"))?;
    nonempty(status, &format!("{label}.document.status"))?;
    Ok(EvidenceDigest {
        path: binding.path.clone(),
        document_schema: schema.into(),
        document_status: status.into(),
        document_sha256: binding.document_sha256.clone(),
        document_seal_sha256: seal,
    })
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

fn validate_topology(
    topology: &TopologyAssertion,
    model_key: &str,
    label: &str,
) -> Result<(), String> {
    if topology.resident_model_processes != 1 {
        return Err(format!(
            "{label}.resident_model_processes must be exactly one; duplicate model bodies are prohibited"
        ));
    }
    if topology.immutable_weight_copies != 1 {
        return Err(format!(
            "{label}.immutable_weight_copies must be exactly one; cloned model weights are prohibited"
        ));
    }
    if topology.logical_session_policy != "many_logical_sessions" {
        return Err(format!(
            "{label}.logical_session_policy must be many_logical_sessions"
        ));
    }
    let expected = expected_endpoint(model_key)?;
    if topology.endpoint != expected {
        return Err(format!(
            "{label}.endpoint must be {}:{} for {model_key}, observed {}:{}",
            expected.host, expected.port, topology.endpoint.host, topology.endpoint.port
        ));
    }
    Ok(())
}

fn validate_same_topology(
    first: &TopologyAssertion,
    second: &TopologyAssertion,
    label: &str,
) -> Result<(), String> {
    if first != second {
        return Err(format!(
            "{label} topology assertions disagree; activation, memory, and session contracts must bind one identical body"
        ));
    }
    Ok(())
}

fn validate_tg10(tg10: &Tg10Authority, model_key: &str) -> Result<(), String> {
    if tg10.required_base_true_tps != TG10_BASE_TRUE_TPS {
        return Err(format!(
            "{model_key}.activation.tg10.required_base_true_tps must be exactly {TG10_BASE_TRUE_TPS}"
        ));
    }
    if !tg10.operational_pass {
        return Ok(());
    }
    if !tg10.coherent_hcli_pass {
        return Err(format!(
            "{model_key}.activation.tg10 cannot pass without coherent HCLI"
        ));
    }
    if !tg10.complete_token_path_measured {
        return Err(format!(
            "{model_key}.activation.tg10 cannot pass without a complete-token measurement"
        ));
    }
    if tg10.fallback_count != 0 {
        return Err(format!(
            "{model_key}.activation.tg10 cannot pass with fallback_count != 0"
        ));
    }
    let measured = tg10
        .median_base_true_tps
        .ok_or_else(|| format!("{model_key}.activation.tg10 pass requires median_base_true_tps"))?;
    if !measured.is_finite() || measured < f64::from(TG10_BASE_TRUE_TPS) {
        return Err(format!(
            "{model_key}.activation.tg10 pass requires median BASE_TRUE_TPS >= {TG10_BASE_TRUE_TPS}"
        ));
    }
    let receipt = tg10.receipt_seal_sha256.as_deref().ok_or_else(|| {
        format!("{model_key}.activation.tg10 pass requires a sealed TG10 receipt")
    })?;
    if !is_lower_sha256(receipt) {
        return Err(format!(
            "{model_key}.activation.tg10.receipt_seal_sha256 must be a lowercase SHA-256"
        ));
    }
    Ok(())
}

fn validate_model_authority(
    authority: &ModelAuthority,
) -> Result<(ModelContractBindingReport, PhysicalBodyTopology), String> {
    if authority.model_key != QWEN30 && authority.model_key != QWEN80 {
        return Err(format!(
            "model_authorities contains unsupported model key {:?}",
            authority.model_key
        ));
    }
    let activation = binding_digest(
        &authority.activation.contract,
        &format!("{}.activation", authority.model_key),
    )?;
    let memory = binding_digest(
        &authority.memory,
        &format!("{}.memory", authority.model_key),
    )?;
    let session = binding_digest(
        &authority.session,
        &format!("{}.session", authority.model_key),
    )?;
    validate_topology(
        &authority.activation.contract.topology_assertion,
        &authority.model_key,
        &format!("{}.activation.topology_assertion", authority.model_key),
    )?;
    validate_topology(
        &authority.memory.topology_assertion,
        &authority.model_key,
        &format!("{}.memory.topology_assertion", authority.model_key),
    )?;
    validate_topology(
        &authority.session.topology_assertion,
        &authority.model_key,
        &format!("{}.session.topology_assertion", authority.model_key),
    )?;
    validate_same_topology(
        &authority.activation.contract.topology_assertion,
        &authority.memory.topology_assertion,
        &format!("{} activation/memory", authority.model_key),
    )?;
    validate_same_topology(
        &authority.activation.contract.topology_assertion,
        &authority.session.topology_assertion,
        &format!("{} activation/session", authority.model_key),
    )?;
    validate_tg10(&authority.activation.tg10, &authority.model_key)?;

    let topology = authority.activation.contract.topology_assertion.clone();
    Ok((
        ModelContractBindingReport {
            model_key: authority.model_key.clone(),
            topology: topology.clone(),
            activation,
            memory,
            session,
            tg10: authority.activation.tg10.clone(),
        },
        PhysicalBodyTopology {
            model_key: authority.model_key.clone(),
            resident_model_processes: topology.resident_model_processes,
            immutable_weight_copies: topology.immutable_weight_copies,
            endpoint: topology.endpoint,
            logical_session_policy: topology.logical_session_policy,
        },
    ))
}

fn namespace_entries(namespaces: &PrivateNamespaces) -> Vec<(&'static str, &str)> {
    vec![
        ("mission", &namespaces.mission),
        ("experiments", &namespaces.experiments),
        ("receipts", &namespaces.receipts),
        ("worktree", &namespaces.worktree),
        ("sessions", &namespaces.sessions),
        ("frontier", &namespaces.frontier),
        ("patches", &namespaces.patches),
        ("scores", &namespaces.scores),
        (
            "protected_adversarial_reviews",
            &namespaces.protected_adversarial_reviews,
        ),
        (
            "independent_verification",
            &namespaces.independent_verification,
        ),
        ("hard_gate_conjunction", &namespaces.hard_gate_conjunction),
        ("post_tg3_freeze", &namespaces.post_tg3_freeze),
        (
            "solo_manager_evaluation",
            &namespaces.solo_manager_evaluation,
        ),
        (
            "symmetric_orchestrator_evaluation",
            &namespaces.symmetric_orchestrator_evaluation,
        ),
    ]
}

fn validate_mission_authority(authority: &MissionAuthority, label: &str) -> Result<(), String> {
    if !authority.primary_candidate_mutation_authority
        || !authority.primary_candidate_promotion_authority
        || !authority.helper_may_inspect_and_critique
        || !authority.helper_may_propose_or_test_in_private_worktree
    {
        return Err(format!(
            "{label} must grant the primary mutation/promotion and the helper review/proposal roles"
        ));
    }
    if authority.helper_may_mutate_primary_champion
        || authority.helper_may_promote_primary_champion
        || authority.opposite_lane_may_mutate_primary_champion
    {
        return Err(format!(
            "{label} permits a helper or opposite lane to alter a champion"
        ));
    }
    if authority.primary_or_helper_may_self_score {
        return Err(format!("{label} permits primary/helper self-scoring"));
    }
    if !authority.protected_adversarial_review_required_before_promotion
        || !authority.independent_verifier_required_before_promotion
        || !authority.hard_gate_conjunction_required_before_activation
        || !authority.post_tg3_freeze_required_before_final_evaluation
        || !authority.solo_and_symmetric_orchestrator_evaluations_required
    {
        return Err(format!(
            "{label} must reserve protected adversarial review, independent verification, hard-gate conjunction, post-TG3 freeze, and both final evaluation modes"
        ));
    }
    Ok(())
}

fn validate_lanes(lanes: &[LaneDefinition]) -> Result<Vec<LaneDefinition>, String> {
    if lanes.len() != 2 {
        return Err("exactly two paired candidate worlds are required".into());
    }
    let mut by_primary = BTreeMap::new();
    let mut all_namespaces = BTreeSet::new();
    for lane in lanes {
        nonempty(&lane.lane_id, "lane.lane_id")?;
        if by_primary
            .insert(lane.primary_model_key.as_str(), lane)
            .is_some()
        {
            return Err(format!(
                "duplicate primary model key {:?} across paired candidate worlds",
                lane.primary_model_key
            ));
        }
        validate_mission_authority(&lane.mission_authority, &lane.lane_id)?;
        for (kind, namespace) in namespace_entries(&lane.private_namespaces) {
            nonempty(
                namespace,
                &format!("{}.private_namespaces.{kind}", lane.lane_id),
            )?;
            if !all_namespaces.insert(namespace) {
                return Err(format!(
                    "private namespace collision at {namespace:?}; every lane namespace, including review/verifier/gate/freeze/evaluation bindings, must be distinct"
                ));
            }
        }
    }
    let qwen80_lane = by_primary
        .get(QWEN80)
        .ok_or("missing Q80-primary candidate world")?;
    if qwen80_lane.helper_model_key != QWEN30 {
        return Err("Q80-primary candidate world must use Q30 as its helper".into());
    }
    let qwen30_lane = by_primary
        .get(QWEN30)
        .ok_or("missing Q30-primary candidate world")?;
    if qwen30_lane.helper_model_key != QWEN80 {
        return Err("Q30-primary candidate world must use Q80 as its helper".into());
    }
    Ok(lanes.to_vec())
}

fn validate_cross_lane_policy(policy: &CrossLaneReadPolicy) -> Result<(), String> {
    let denied = [
        ("mission", policy.allow_cross_lane_mission_reads),
        ("experiment", policy.allow_cross_lane_experiment_reads),
        ("receipt", policy.allow_cross_lane_receipt_reads),
        ("worktree", policy.allow_cross_lane_worktree_reads),
        ("session", policy.allow_cross_lane_session_reads),
        ("frontier", policy.allow_cross_lane_frontier_reads),
        ("patch", policy.allow_cross_lane_patch_reads),
        ("score", policy.allow_cross_lane_score_reads),
    ];
    for (kind, allowed) in denied {
        if allowed {
            return Err(format!("cross-lane {kind} reads must be denied"));
        }
    }
    if !policy.verified_generic_knowledge_plane_publication_only {
        return Err(
            "a shared Knowledge Plane may contain only verifier-released generic mechanisms".into(),
        );
    }
    Ok(())
}

fn validate_authority_boundary(boundary: &AuthorityBoundary) -> Result<(), String> {
    if boundary.new_physical_model_processes_authorized != 0
        || boundary.server_starts_authorized != 0
        || boundary.port_binds_authorized != 0
        || boundary.gpu_leases_authorized != 0
        || boundary.tournament_state_mutations_authorized != 0
        || boundary.paired_world_activation_authorized
    {
        return Err(
            "this lane namespace contract may not authorize a process, server, port, lease, tournament mutation, or activation"
                .into(),
        );
    }
    Ok(())
}

fn validate_execution_boundary(boundary: &ExecutionBoundary) -> Result<(), String> {
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
                "execution_boundary.{field} must be false for this CPU-only authority"
            ));
        }
    }
    Ok(())
}

fn below_tg10_blockers(reports: &[ModelContractBindingReport]) -> Vec<String> {
    reports
        .iter()
        .filter(|report| !report.tg10.operational_pass)
        .map(|report| {
            format!(
                "{} has no valid TG10 operational pass at >= {} BASE_TRUE_TPS",
                report.model_key, TG10_BASE_TRUE_TPS
            )
        })
        .collect()
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
    if input.model_authorities.len() != 2 {
        return Err("input requires exactly one Q30 and one Q80 model authority".into());
    }
    let mut seen_models = BTreeSet::new();
    let mut reports = Vec::new();
    let mut bodies = Vec::new();
    let mut bound_paths = BTreeSet::new();
    for authority in &input.model_authorities {
        if !seen_models.insert(authority.model_key.as_str()) {
            return Err(format!(
                "duplicate model authority for {:?}; duplicate bodies are prohibited",
                authority.model_key
            ));
        }
        for binding in [
            &authority.activation.contract,
            &authority.memory,
            &authority.session,
        ] {
            if !bound_paths.insert(binding.path.as_str()) {
                return Err(format!(
                    "contract path {:?} is bound more than once; activation/memory/session evidence must remain distinct",
                    binding.path
                ));
            }
        }
        let (report, body) = validate_model_authority(authority)?;
        reports.push(report);
        bodies.push(body);
    }
    if !seen_models.contains(QWEN30) || !seen_models.contains(QWEN80) {
        return Err("model authorities must contain exactly qwen30 and qwen80".into());
    }
    bodies.sort_by(|left, right| left.model_key.cmp(&right.model_key));
    reports.sort_by(|left, right| left.model_key.cmp(&right.model_key));
    if bodies[0].endpoint == bodies[1].endpoint {
        return Err("Q30 and Q80 may not share a port or endpoint".into());
    }
    let lanes = validate_lanes(&input.lanes)?;
    validate_cross_lane_policy(&input.cross_lane_read_policy)?;
    validate_authority_boundary(&input.authority_boundary)?;
    validate_execution_boundary(&input.execution_boundary)?;

    let qwen30_tg10 = reports
        .iter()
        .find(|report| report.model_key == QWEN30)
        .map(|report| report.tg10.operational_pass)
        .expect("validated qwen30 report exists");
    let qwen80_tg10 = reports
        .iter()
        .find(|report| report.model_key == QWEN80)
        .map(|report| report.tg10.operational_pass)
        .expect("validated qwen80 report exists");
    let both_tg10 = qwen30_tg10 && qwen80_tg10;
    let mut blockers = below_tg10_blockers(&reports);
    if input.activation_requested {
        if !both_tg10 {
            return Err(format!(
                "paired-world activation is refused below TG10: {}",
                blockers.join("; ")
            ));
        }
        return Err(
            "paired-world activation is outside this contract's authority even after both TG10 passes"
                .into(),
        );
    }
    if blockers.is_empty() {
        blockers.push(
            "both TG10 conditions are present, but this namespace contract deliberately has no activation authority"
                .into(),
        );
    }

    let primary_map: BTreeMap<&str, &LaneDefinition> = lanes
        .iter()
        .map(|lane| (lane.primary_model_key.as_str(), lane))
        .collect();
    let qwen80_lane = primary_map.get(QWEN80).expect("validated Q80 lane exists");
    let qwen30_lane = primary_map.get(QWEN30).expect("validated Q30 lane exists");
    let all_private_namespaces_distinct = {
        let mut names = BTreeSet::new();
        lanes.iter().all(|lane| {
            namespace_entries(&lane.private_namespaces)
                .iter()
                .all(|(_, namespace)| names.insert(*namespace))
        })
    };
    let all_cross_lane_private_reads_denied =
        !input.cross_lane_read_policy.allow_cross_lane_mission_reads
            && !input
                .cross_lane_read_policy
                .allow_cross_lane_experiment_reads
            && !input.cross_lane_read_policy.allow_cross_lane_receipt_reads
            && !input.cross_lane_read_policy.allow_cross_lane_worktree_reads
            && !input.cross_lane_read_policy.allow_cross_lane_session_reads
            && !input.cross_lane_read_policy.allow_cross_lane_frontier_reads
            && !input.cross_lane_read_policy.allow_cross_lane_patch_reads
            && !input.cross_lane_read_policy.allow_cross_lane_score_reads;
    let primary_only_champion_mutation_and_promotion = lanes.iter().all(|lane| {
        lane.mission_authority.primary_candidate_mutation_authority
            && lane.mission_authority.primary_candidate_promotion_authority
            && !lane.mission_authority.helper_may_mutate_primary_champion
            && !lane.mission_authority.helper_may_promote_primary_champion
            && !lane
                .mission_authority
                .opposite_lane_may_mutate_primary_champion
    });
    let self_scoring_prohibited = lanes
        .iter()
        .all(|lane| !lane.mission_authority.primary_or_helper_may_self_score);
    let deferred_adversarial_verifier_gate_freeze_and_evaluation_namespaces_bound =
        lanes.iter().all(|lane| {
            !lane
                .private_namespaces
                .protected_adversarial_reviews
                .is_empty()
                && !lane.private_namespaces.independent_verification.is_empty()
                && !lane.private_namespaces.hard_gate_conjunction.is_empty()
                && !lane.private_namespaces.post_tg3_freeze.is_empty()
                && !lane.private_namespaces.solo_manager_evaluation.is_empty()
                && !lane
                    .private_namespaces
                    .symmetric_orchestrator_evaluation
                    .is_empty()
                && lane
                    .mission_authority
                    .protected_adversarial_review_required_before_promotion
                && lane
                    .mission_authority
                    .independent_verifier_required_before_promotion
                && lane
                    .mission_authority
                    .hard_gate_conjunction_required_before_activation
                && lane
                    .mission_authority
                    .post_tg3_freeze_required_before_final_evaluation
                && lane
                    .mission_authority
                    .solo_and_symmetric_orchestrator_evaluations_required
        });
    let no_new_physical_model_process_authority = input
        .authority_boundary
        .new_physical_model_processes_authorized
        == 0
        && input.authority_boundary.server_starts_authorized == 0
        && input.authority_boundary.port_binds_authorized == 0
        && input.authority_boundary.gpu_leases_authorized == 0;
    let focused_checks = FocusedChecks {
        exact_two_candidate_worlds: lanes.len() == 2,
        qwen80_primary_qwen30_helper: qwen80_lane.primary_model_key == QWEN80
            && qwen80_lane.helper_model_key == QWEN30,
        qwen30_primary_qwen80_helper: qwen30_lane.primary_model_key == QWEN30
            && qwen30_lane.helper_model_key == QWEN80,
        all_private_namespaces_distinct,
        all_cross_lane_private_reads_denied,
        primary_only_champion_mutation_and_promotion,
        self_scoring_prohibited,
        deferred_adversarial_verifier_gate_freeze_and_evaluation_namespaces_bound,
        exactly_one_qwen30_body: bodies
            .iter()
            .find(|body| body.model_key == QWEN30)
            .map(|body| body.resident_model_processes == 1 && body.immutable_weight_copies == 1)
            .unwrap_or(false),
        exactly_one_qwen80_body: bodies
            .iter()
            .find(|body| body.model_key == QWEN80)
            .map(|body| body.resident_model_processes == 1 && body.immutable_weight_copies == 1)
            .unwrap_or(false),
        qwen30_qwen80_ports_distinct: bodies[0].endpoint != bodies[1].endpoint,
        no_new_physical_process_or_server_authority: no_new_physical_model_process_authority,
        below_tg10_activation_rejected: !both_tg10,
        execution_boundary_cpu_only: !input.execution_boundary.live_artifact_scan_performed
            && !input.execution_boundary.model_weights_loaded
            && !input.execution_boundary.metal_device_or_dispatch_performed
            && !input.execution_boundary.gpu_lease_or_registry_mutated
            && !input.execution_boundary.model_or_decoder_token_executed
            && !input.execution_boundary.logical_session_created
            && !input.execution_boundary.runtime_watcher_or_server_started
            && !input.execution_boundary.port_bound_or_listener_created
            && !input.execution_boundary.hcli_executed
            && !input.execution_boundary.tps_or_tg_measured
            && !input.execution_boundary.tournament_state_mutated,
    };

    let mut report = Report {
        schema: RESULT_SCHEMA,
        status: STATUS,
        prepared: true,
        paired_candidate_worlds_active: false,
        no_new_physical_model_process_authority,
        model_contract_bindings: reports,
        physical_body_topology: bodies,
        lane_worlds: lanes,
        cross_lane_read_policy: input.cross_lane_read_policy,
        authority_boundary: input.authority_boundary,
        activation_gate: ActivationGateReport {
            activation_requested: false,
            required_base_true_tps: TG10_BASE_TRUE_TPS,
            qwen30_tg10_operational_pass: qwen30_tg10,
            qwen80_tg10_operational_pass: qwen80_tg10,
            both_tg10_operational_passes_present: both_tg10,
            paired_world_activation_authorized_by_this_contract: false,
            blockers,
        },
        focused_checks,
        execution_boundary: input.execution_boundary,
        claim_boundary: vec![
            "This is a sealed CPU-only namespace and mission authority, not a model runtime.",
            "It starts no Q30 or Q80 process, server, watcher, logical session, port listener, GPU lease, or device dispatch.",
            "It does not execute a model token, HCLI request, TPS/TG measurement, experiment, or benchmark.",
            "It does not modify tournament state and cannot activate paired candidate worlds.",
            "It does not release one lane's private mission, experiment, receipt, worktree, session, frontier, patch, or score data to the other lane.",
            "The private namespace surface reserves later bindings for protected adversarial review, independent no-self-scoring verification, hard-gate conjunction, post-TG3 freeze, and solo plus symmetric manager-as-orchestrator evaluation.",
            "A later protected controller must independently verify both TG10 passes and all V3/TG3 final-manager conditions.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    let report_value = serde_json::to_value(&report)
        .map_err(|error| format!("report cannot be serialized: {error}"))?;
    report.unsealed_preimage_sha256 = sha256_json(&report_value)?;
    Ok(report)
}

fn load_input(path: &Path) -> Result<Input, Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--input must be an absolute path".into());
    }
    let bytes = fs::read(path)?;
    let raw: Value = serde_json::from_slice(&bytes)?;
    verify_sealed_object(&raw, "input")
        .map_err(|error| format!("invalid sealed input: {error}"))?;
    let input: Input = serde_json::from_value(raw)?;
    if input.schema != INPUT_SCHEMA {
        return Err(format!("input.schema must be {INPUT_SCHEMA}").into());
    }
    Ok(input)
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

fn write_report_create_new(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--out must be an absolute path".into());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let value = seal_value(
        serde_json::to_value(report)
            .map_err(|error| format!("paired-cognition report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("paired-cognition report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_paired_cognition_lane_authority_contract --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        eprintln!("ascension_paired_cognition_lane_authority_contract: {error}");
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

    fn sealed_document(model_key: &str, role: &str) -> Value {
        seal_value(json!({
            "schema": format!("hawking.ascension.{model_key}.{role}.fixture.v1"),
            "status": format!("PREPARED_{role}_FIXTURE"),
            "model_key": model_key,
        }))
        .unwrap()
    }

    fn topology(model_key: &str) -> TopologyAssertion {
        TopologyAssertion {
            resident_model_processes: 1,
            immutable_weight_copies: 1,
            logical_session_policy: "many_logical_sessions".into(),
            endpoint: expected_endpoint(model_key).unwrap(),
        }
    }

    fn binding(model_key: &str, role: &str) -> ContractBinding {
        let document = sealed_document(model_key, role);
        ContractBinding {
            path: format!("/sealed/{model_key}/{role}.json"),
            document_sha256: sha256_json(&document).unwrap(),
            document,
            topology_assertion: topology(model_key),
        }
    }

    fn tg10_not_earned() -> Tg10Authority {
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

    fn model_authority(model_key: &str) -> ModelAuthority {
        ModelAuthority {
            model_key: model_key.into(),
            activation: ActivationBinding {
                contract: binding(model_key, "activation"),
                tg10: tg10_not_earned(),
            },
            memory: binding(model_key, "memory"),
            session: binding(model_key, "session"),
        }
    }

    fn namespaces(lane: &str) -> PrivateNamespaces {
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

    fn lane(primary: &str, helper: &str) -> LaneDefinition {
        LaneDefinition {
            lane_id: format!("{primary}-candidate-world"),
            primary_model_key: primary.into(),
            helper_model_key: helper.into(),
            private_namespaces: namespaces(&format!("{primary}-lane")),
            mission_authority: mission_authority(),
        }
    }

    fn policy() -> CrossLaneReadPolicy {
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

    fn input_fixture() -> Input {
        Input {
            schema: INPUT_SCHEMA.into(),
            activation_requested: false,
            model_authorities: vec![model_authority(QWEN30), model_authority(QWEN80)],
            lanes: vec![lane(QWEN80, QWEN30), lane(QWEN30, QWEN80)],
            cross_lane_read_policy: policy(),
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
    fn correct_two_worlds_are_prepared_not_active_and_cpu_only() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, STATUS);
        assert!(report.prepared);
        assert!(!report.paired_candidate_worlds_active);
        assert_eq!(report.physical_body_topology.len(), 2);
        assert!(report.focused_checks.qwen80_primary_qwen30_helper);
        assert!(report.focused_checks.qwen30_primary_qwen80_helper);
        assert!(report.focused_checks.all_private_namespaces_distinct);
        assert!(report.focused_checks.all_cross_lane_private_reads_denied);
        assert!(
            report
                .focused_checks
                .primary_only_champion_mutation_and_promotion
        );
        assert!(report.focused_checks.self_scoring_prohibited);
        assert!(
            report
                .focused_checks
                .deferred_adversarial_verifier_gate_freeze_and_evaluation_namespaces_bound
        );
        assert!(
            report
                .focused_checks
                .no_new_physical_process_or_server_authority
        );
        assert!(report.focused_checks.execution_boundary_cpu_only);
        assert!(report.focused_checks.below_tg10_activation_rejected);
        assert!(!report.execution_boundary.runtime_watcher_or_server_started);
        assert!(!report.execution_boundary.metal_device_or_dispatch_performed);
    }

    #[test]
    fn duplicate_body_or_endpoint_is_rejected() {
        let mut duplicate_body = input_fixture();
        duplicate_body.model_authorities[0]
            .activation
            .contract
            .topology_assertion
            .resident_model_processes = 2;
        assert!(build_report(duplicate_body).is_err());

        let mut duplicate_endpoint = input_fixture();
        let qwen30_endpoint = expected_endpoint(QWEN30).unwrap();
        let qwen80 = &mut duplicate_endpoint.model_authorities[1];
        qwen80.activation.contract.topology_assertion.endpoint = qwen30_endpoint.clone();
        qwen80.memory.topology_assertion.endpoint = qwen30_endpoint.clone();
        qwen80.session.topology_assertion.endpoint = qwen30_endpoint;
        assert!(build_report(duplicate_endpoint).is_err());
    }

    #[test]
    fn cross_lane_private_reads_and_namespace_aliases_are_rejected() {
        let mut cross_read = input_fixture();
        cross_read
            .cross_lane_read_policy
            .allow_cross_lane_frontier_reads = true;
        assert!(build_report(cross_read).is_err());

        let mut aliased = input_fixture();
        aliased.lanes[1].private_namespaces.scores =
            aliased.lanes[0].private_namespaces.scores.clone();
        assert!(build_report(aliased).is_err());
    }

    #[test]
    fn helper_champion_mutation_is_rejected() {
        let mut input = input_fixture();
        input.lanes[0]
            .mission_authority
            .helper_may_mutate_primary_champion = true;
        assert!(build_report(input).is_err());
    }

    #[test]
    fn self_scoring_and_missing_post_tg3_freeze_contract_are_rejected() {
        let mut self_scoring = input_fixture();
        self_scoring.lanes[0]
            .mission_authority
            .primary_or_helper_may_self_score = true;
        assert!(build_report(self_scoring).is_err());

        let mut no_freeze = input_fixture();
        no_freeze.lanes[1]
            .mission_authority
            .post_tg3_freeze_required_before_final_evaluation = false;
        assert!(build_report(no_freeze).is_err());
    }

    #[test]
    fn below_tg10_activation_request_is_refused() {
        let mut input = input_fixture();
        input.activation_requested = true;
        let error = build_report(input).unwrap_err();
        assert!(error.contains("refused below TG10"));
    }

    #[test]
    fn sealed_input_and_output_are_verified_and_create_new() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("input.json");
        let output_path = directory.path().join("out.json");
        let sealed = seal_input(&input_fixture());
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        let args = Args {
            input: input_path,
            out: output_path.clone(),
        };
        run(args).unwrap();
        let output: Value = serde_json::from_slice(&fs::read(&output_path).unwrap()).unwrap();
        verify_sealed_object(&output, "output").unwrap();
        assert_eq!(output["status"], Value::String(STATUS.into()));
        assert!(
            write_report_create_new(&output_path, &build_report(input_fixture()).unwrap()).is_err()
        );
    }

    #[test]
    fn tampered_embedded_or_outer_seal_is_rejected() {
        let directory = tempdir().unwrap();
        let input_path = directory.path().join("tampered-input.json");
        let output_path = directory.path().join("out.json");
        let mut sealed = seal_input(&input_fixture());
        sealed["lanes"][0]["primary_model_key"] = Value::String("qwen30".into());
        fs::write(&input_path, serde_json::to_vec_pretty(&sealed).unwrap()).unwrap();
        assert!(run(Args {
            input: input_path,
            out: output_path,
        })
        .is_err());

        let mut input = input_fixture();
        input.model_authorities[0].memory.document["status"] = Value::String("TAMPERED".into());
        assert!(build_report(input).is_err());
    }
}
