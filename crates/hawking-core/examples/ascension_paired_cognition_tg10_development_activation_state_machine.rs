//! Sealed both-TG10 paired-development activation state machine.
//!
//! This CPU-only gate consumes sealed paired-cognition lane, mutation,
//! Knowledge Plane, and one-body scheduler authorities together with an
//! embedded sealed operational-ascent status snapshot.  It can only emit a
//! prepared inactive reservation or a refusal.  It never starts paired
//! development, creates a logical session, launches a model/server/watcher,
//! binds a port, leases a GPU, touches a tournament, scores candidates, or
//! selects a manager.
//!
//! The sole readiness condition is two exact, fresh, model-matched TG10
//! operational receipts: coherent HCLI, complete-token path, no fallback, and
//! median BASE_TRUE_TPS >=100 for both Q30 and Q80.  Each receipt must bind the
//! same operational-ascent snapshot fingerprint and timestamp and agree with
//! every consumed authority.  TG3, final scorecards, and manager selection are
//! intentionally outside this state machine.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const INPUT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_tg10_development_activation_input.v1";
const RESULT_SCHEMA: &str =
    "hawking.ascension.paired_cognition_both_tg10_development_activation_state_machine.v1";
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
const OPERATIONAL_ASCENT_SCHEMA: &str = "hawking.ascension.physical_operational_ascent.v1";
const OPERATIONAL_ASCENT_WAITING_STATUS: &str = "WAITING_FOR_BOTH_VALID_TG10_OPERATIONAL_RECEIPTS";
const OPERATIONAL_ASCENT_READY_STATUS: &str = "BOTH_VALID_TG10_OPERATIONAL_RECEIPTS_BOUND";
const TG10_RECEIPT_SCHEMA: &str = "hawking.ascension.base_true_tps_tg10_operational_receipt.v1";
const PREPARED_STATUS: &str =
    "PREPARED_NOT_ACTIVE_PAIRED_COGNITION_BOTH_EXACT_TG10_OPERATIONAL_RECEIPTS_BOUND_NO_RUNTIME_OR_TOURNAMENT";
const REFUSED_TODAY_STATUS: &str =
    "REFUSED_TODAY_PAIRED_COGNITION_WAITING_FOR_BOTH_EXACT_TG10_OPERATIONAL_RECEIPTS";
const REFUSED_EARLY_STATUS: &str =
    "REFUSED_EARLY_PAIRED_DEVELOPMENT_ACTIVATION_REQUIRES_BOTH_EXACT_TG10_OPERATIONAL_RECEIPTS";
const REFUSED_AUTHORITY_STATUS: &str =
    "REFUSED_PAIRED_DEVELOPMENT_ACTIVATION_REMAINS_OUTSIDE_THIS_CPU_ONLY_STATE_MACHINE";
const OPERATIONAL_STATUS_PATH: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/lifecycle/ASCENSION_OPERATIONAL_ASCENT_STATUS.json";

const QWEN30: &str = "qwen30";
const QWEN80: &str = "qwen80";
const TG10_BASE_TRUE_TPS: u16 = 100;

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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct ModelContractBindingReport {
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
    model_contract_bindings: Vec<ModelContractBindingReport>,
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct Tg10Readiness {
    model_key: String,
    required_base_true_tps: u16,
    operational_pass: bool,
    receipt_seal_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
struct PairedDevelopmentReservation {
    required_base_true_tps: u16,
    qwen30_tg10: Tg10Readiness,
    qwen80_tg10: Tg10Readiness,
    both_tg10_operational_receipts_present: bool,
    paired_development_ready_after_both_tg10: bool,
    paired_development_activation_authorized_by_this_contract: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
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
    paired_development_reservation: PairedDevelopmentReservation,
    final_mode_gate: FinalModeGateReport,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct OperationalModelEvidence {
    complete_artifact_admission_seal_sha256: Option<String>,
    hcli_receipt_seal_sha256: Option<String>,
    kernel_receipt_seal_sha256: Option<String>,
    runtime_receipt_seal_sha256: Option<String>,
    source_identity_seal_sha256: Option<String>,
    source_revalidation_seal_sha256: Option<String>,
    tg10_receipt_seal_sha256: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct OperationalEvidence {
    fixed_candidate_order: Vec<String>,
    models: BTreeMap<String, OperationalModelEvidence>,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct OperationalClaimBoundary {
    operational_ascent_does_not_activate_the_sandbox: bool,
    operational_ascent_does_not_launch_or_score_the_protected_tournament: bool,
    operational_ascent_is_not_a_capability_or_agent_os_pass: bool,
    operational_ascent_is_not_tg3: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
struct ProtectedTournamentBoundary {
    capability_agent_os_and_final_review_remain_required: bool,
    launch_requested: bool,
    qualification_override: bool,
    tg3_remains_required: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct OperationalAscentStatus {
    schema: String,
    status: String,
    both_valid_tg10_receipts: bool,
    evidence_fingerprint: Option<String>,
    next_transition: String,
    recorded_at: String,
    evidence: OperationalEvidence,
    claim_boundary: OperationalClaimBoundary,
    protected_tournament: ProtectedTournamentBoundary,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct Tg10OperationalReceipt {
    schema: String,
    model_key: String,
    tg_level: u8,
    required_base_true_tps: u16,
    operational_pass: bool,
    coherent_hcli_pass: bool,
    complete_token_path_measured: bool,
    fallback_count: usize,
    median_base_true_tps: f64,
    complete_artifact_admission_seal_sha256: String,
    source_identity_seal_sha256: String,
    source_revalidation_seal_sha256: String,
    kernel_receipt_seal_sha256: String,
    hcli_receipt_seal_sha256: String,
    runtime_receipt_seal_sha256: String,
    observed_operational_ascent_recorded_at: String,
    observed_operational_ascent_evidence_fingerprint_sha256: String,
    seal_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Input {
    schema: String,
    lane_authority: SealedDocumentBinding,
    mutation_authority: SealedDocumentBinding,
    knowledge_authority: SealedDocumentBinding,
    scheduler_authority: SealedDocumentBinding,
    operational_ascent_status: SealedDocumentBinding,
    qwen30_tg10_receipt: Option<SealedDocumentBinding>,
    qwen80_tg10_receipt: Option<SealedDocumentBinding>,
    paired_development_activation_requested: bool,
    authority_boundary: AuthorityBoundary,
    execution_boundary: ExecutionBoundary,
    seal_sha256: String,
}

#[derive(Clone, Debug, Serialize)]
struct ReceiptAudit {
    model_key: String,
    receipt_present: bool,
    receipt_document_sha256: Option<String>,
    receipt_seal_sha256: Option<String>,
    valid_exact_fresh_tg10_operational_receipt: bool,
    blockers: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
struct FocusedChecks {
    exact_sealed_lane_mutation_knowledge_and_scheduler_chain_bound: bool,
    sealed_operational_ascent_status_bound_at_canonical_path: bool,
    qwen30_and_qwen80_tg10_receipts_are_exact_fresh_and_model_matched: bool,
    both_receipts_require_coherent_complete_fallback_free_base_true_tps_at_least_100: bool,
    stale_forged_mismatched_and_early_receipts_refused: bool,
    paired_development_never_activated_by_this_contract: bool,
    final_manager_selection_scorecards_and_tg3_outside_activation: bool,
    no_runtime_server_gpu_watcher_or_tournament_authority: bool,
    execution_boundary_cpu_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    schema: &'static str,
    status: String,
    prepared: bool,
    paired_development_active: bool,
    paired_development_activation_authorized_by_this_contract: bool,
    bound_lane_authority: EvidenceDigest,
    bound_mutation_authority: EvidenceDigest,
    bound_knowledge_authority: EvidenceDigest,
    bound_scheduler_authority: EvidenceDigest,
    bound_operational_ascent_status: EvidenceDigest,
    qwen30_tg10_audit: ReceiptAudit,
    qwen80_tg10_audit: ReceiptAudit,
    both_exact_fresh_tg10_operational_receipts_present: bool,
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

fn validate_tg10_authority(tg10: &Tg10Authority, model_key: &str) -> Result<(), String> {
    if tg10.required_base_true_tps != TG10_BASE_TRUE_TPS {
        return Err(format!(
            "{model_key}.tg10 required BASE_TRUE_TPS must be {TG10_BASE_TRUE_TPS}"
        ));
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
    if !measured.is_finite() || measured < f64::from(TG10_BASE_TRUE_TPS) {
        return Err(format!(
            "{model_key}.tg10 pass must measure >= {TG10_BASE_TRUE_TPS} BASE_TRUE_TPS"
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
    if document.model_contract_bindings.len() != 2 {
        return Err("lane_authority must bind exactly two model TG10 authorities".into());
    }
    let mut result = BTreeMap::new();
    for model in document.model_contract_bindings {
        if model.model_key != QWEN30 && model.model_key != QWEN80 {
            return Err("lane_authority contains unsupported model TG10 authority".into());
        }
        validate_tg10_authority(&model.tg10, &model.model_key)?;
        if result.insert(model.model_key.clone(), model.tg10).is_some() {
            return Err("lane_authority has duplicate model TG10 authorities".into());
        }
    }
    if result.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from([QWEN30, QWEN80])
    {
        return Err("lane_authority must bind exactly Q30 and Q80 TG10 authorities".into());
    }
    Ok((digest, result))
}

fn exact_digest(
    expected: &EvidenceDigest,
    actual: &EvidenceDigest,
    label: &str,
) -> Result<(), String> {
    if actual.document_sha256 != expected.document_sha256
        || actual.document_seal_sha256 != expected.document_seal_sha256
    {
        return Err(format!(
            "{label} must bind the exact sealed source authority"
        ));
    }
    Ok(())
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

fn scheduler_tg10_map(
    reservation: &PairedDevelopmentReservation,
) -> Result<BTreeMap<String, Tg10Readiness>, String> {
    if reservation.required_base_true_tps != TG10_BASE_TRUE_TPS
        || reservation.paired_development_activation_authorized_by_this_contract
    {
        return Err(
            "scheduler paired-development reservation has invalid TG10/activation authority".into(),
        );
    }
    let mut result = BTreeMap::new();
    for readiness in [&reservation.qwen30_tg10, &reservation.qwen80_tg10] {
        if readiness.required_base_true_tps != TG10_BASE_TRUE_TPS
            || (readiness.model_key != QWEN30 && readiness.model_key != QWEN80)
        {
            return Err("scheduler TG10 readiness has invalid model or threshold".into());
        }
        if result
            .insert(readiness.model_key.clone(), readiness.clone())
            .is_some()
        {
            return Err("scheduler has duplicate TG10 readiness model".into());
        }
    }
    if result.keys().map(String::as_str).collect::<BTreeSet<_>>()
        != BTreeSet::from([QWEN30, QWEN80])
    {
        return Err("scheduler must carry Q30 and Q80 readiness".into());
    }
    let expected_both = result.values().all(|item| item.operational_pass);
    if reservation.both_tg10_operational_receipts_present != expected_both
        || reservation.paired_development_ready_after_both_tg10 != expected_both
    {
        return Err("scheduler readiness booleans do not agree with the two TG10 records".into());
    }
    Ok(result)
}

fn validate_scheduler_authority(
    binding: &SealedDocumentBinding,
    lane_digest: &EvidenceDigest,
    mutation_digest: &EvidenceDigest,
    knowledge_digest: &EvidenceDigest,
    lane_tg10: &BTreeMap<String, Tg10Authority>,
) -> Result<(EvidenceDigest, BTreeMap<String, Tg10Readiness>), String> {
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
        return Err("scheduler_authority must keep final evaluation and winner selection outside activation".into());
    }
    validate_authority_boundary(
        &document.authority_boundary,
        "scheduler_authority.authority_boundary",
    )?;
    validate_execution_boundary(
        &document.execution_boundary,
        "scheduler_authority.execution_boundary",
    )?;
    let scheduler_tg10 = scheduler_tg10_map(&document.paired_development_reservation)?;
    for model_key in [QWEN30, QWEN80] {
        let lane = lane_tg10.get(model_key).expect("validated lane model");
        let scheduler = scheduler_tg10
            .get(model_key)
            .expect("validated scheduler model");
        if scheduler.operational_pass != lane.operational_pass
            || scheduler.receipt_seal_sha256 != lane.receipt_seal_sha256
        {
            return Err(format!(
                "scheduler {model_key} TG10 readiness does not match lane authority"
            ));
        }
    }
    Ok((digest, scheduler_tg10))
}

fn operational_evidence_fingerprint(status: &OperationalAscentStatus) -> Result<String, String> {
    let qwen30 = status
        .evidence
        .models
        .get(QWEN30)
        .ok_or("operational status lacks Q30 evidence")?;
    let qwen80 = status
        .evidence
        .models
        .get(QWEN80)
        .ok_or("operational status lacks Q80 evidence")?;
    // Deliberately excludes each TG10 receipt seal.  That permits a future
    // status snapshot to record those seals while receipts bind the immutable
    // source/admission/HCLI/kernel/runtime snapshot without a circular seal.
    sha256_json(&json!({
        "schema": status.schema,
        "recorded_at": status.recorded_at,
        "fixed_candidate_order": status.evidence.fixed_candidate_order,
        "qwen30": {
            "complete_artifact_admission_seal_sha256": qwen30.complete_artifact_admission_seal_sha256,
            "hcli_receipt_seal_sha256": qwen30.hcli_receipt_seal_sha256,
            "kernel_receipt_seal_sha256": qwen30.kernel_receipt_seal_sha256,
            "runtime_receipt_seal_sha256": qwen30.runtime_receipt_seal_sha256,
            "source_identity_seal_sha256": qwen30.source_identity_seal_sha256,
            "source_revalidation_seal_sha256": qwen30.source_revalidation_seal_sha256,
        },
        "qwen80": {
            "complete_artifact_admission_seal_sha256": qwen80.complete_artifact_admission_seal_sha256,
            "hcli_receipt_seal_sha256": qwen80.hcli_receipt_seal_sha256,
            "kernel_receipt_seal_sha256": qwen80.kernel_receipt_seal_sha256,
            "runtime_receipt_seal_sha256": qwen80.runtime_receipt_seal_sha256,
            "source_identity_seal_sha256": qwen80.source_identity_seal_sha256,
            "source_revalidation_seal_sha256": qwen80.source_revalidation_seal_sha256,
        },
    }))
}

fn validate_operational_status(
    binding: &SealedDocumentBinding,
) -> Result<(EvidenceDigest, OperationalAscentStatus, Vec<String>), String> {
    if binding.path != OPERATIONAL_STATUS_PATH {
        return Err(
            "operational_ascent_status.path must be the canonical operational-ascent status path"
                .into(),
        );
    }
    let digest = binding_digest(binding, "operational_ascent_status")?;
    let status: OperationalAscentStatus = serde_json::from_value(binding.document.clone())
        .map_err(|error| {
            format!("operational_ascent_status.document has wrong grammar: {error}")
        })?;
    if status.schema != OPERATIONAL_ASCENT_SCHEMA
        || status.recorded_at.trim().is_empty()
        || status.next_transition.trim().is_empty()
    {
        return Err(
            "operational_ascent_status must have the canonical schema, recorded_at, and next_transition".into(),
        );
    }
    if status.status != OPERATIONAL_ASCENT_WAITING_STATUS
        && status.status != OPERATIONAL_ASCENT_READY_STATUS
    {
        return Err("operational_ascent_status has an unrecognized transition state".into());
    }
    if status.evidence.fixed_candidate_order
        != vec![
            String::from("Qwen30-Gravity-Manager-Artifact"),
            String::from("Qwen80-Gravity-Manager-Artifact"),
        ]
        || status.evidence.models.len() != 2
        || !status.evidence.models.contains_key(QWEN30)
        || !status.evidence.models.contains_key(QWEN80)
    {
        return Err(
            "operational_ascent_status must preserve the exact fixed Q30/Q80 evidence set".into(),
        );
    }
    if !status
        .claim_boundary
        .operational_ascent_does_not_activate_the_sandbox
        || !status
            .claim_boundary
            .operational_ascent_does_not_launch_or_score_the_protected_tournament
        || !status
            .claim_boundary
            .operational_ascent_is_not_a_capability_or_agent_os_pass
        || !status.claim_boundary.operational_ascent_is_not_tg3
        || !status
            .protected_tournament
            .capability_agent_os_and_final_review_remain_required
        || status.protected_tournament.launch_requested
        || status.protected_tournament.qualification_override
        || !status.protected_tournament.tg3_remains_required
    {
        return Err(
            "operational_ascent_status must reserve activation, tournament, capability, and TG3"
                .into(),
        );
    }
    let mut blockers = Vec::new();
    if !status.both_valid_tg10_receipts {
        blockers.push("operational_status_both_valid_tg10_receipts_false".into());
    }
    if !status.both_valid_tg10_receipts && status.status != OPERATIONAL_ASCENT_WAITING_STATUS {
        blockers.push("operational_status_false_receipts_requires_waiting_state".into());
    }
    if status.both_valid_tg10_receipts && status.status != OPERATIONAL_ASCENT_READY_STATUS {
        blockers.push("operational_status_true_receipts_requires_ready_state".into());
    }
    let expected_fingerprint = operational_evidence_fingerprint(&status)?;
    match status.evidence_fingerprint.as_deref() {
        Some(value) if is_lower_sha256(value) && value == expected_fingerprint => {}
        Some(_) => blockers.push("operational_status_evidence_fingerprint_mismatch".into()),
        None => blockers.push("operational_status_evidence_fingerprint_missing".into()),
    }
    Ok((digest, status, blockers))
}

fn require_matching_status_seal(
    blockers: &mut Vec<String>,
    label: &str,
    expected: Option<&String>,
    observed: &str,
) {
    match expected {
        Some(value) if is_lower_sha256(value) && value == observed => {}
        _ => blockers.push(format!("{label}_does_not_match_operational_status")),
    }
}

fn audit_receipt(
    model_key: &str,
    binding: &Option<SealedDocumentBinding>,
    status: &OperationalAscentStatus,
    lane_tg10: &Tg10Authority,
    scheduler_tg10: &Tg10Readiness,
) -> ReceiptAudit {
    let mut audit = ReceiptAudit {
        model_key: model_key.into(),
        receipt_present: binding.is_some(),
        receipt_document_sha256: None,
        receipt_seal_sha256: None,
        valid_exact_fresh_tg10_operational_receipt: false,
        blockers: Vec::new(),
    };
    let Some(binding) = binding else {
        audit.blockers.push("tg10_receipt_not_present".into());
        return audit;
    };
    if !Path::new(&binding.path).is_absolute() {
        audit.blockers.push("receipt_path_not_absolute".into());
        return audit;
    }
    if !is_lower_sha256(&binding.document_sha256)
        || sha256_json(&binding.document).ok().as_deref() != Some(&binding.document_sha256)
    {
        audit
            .blockers
            .push("receipt_document_digest_mismatch".into());
        return audit;
    }
    audit.receipt_document_sha256 = Some(binding.document_sha256.clone());
    let seal = match verify_sealed_object(&binding.document, "tg10_receipt") {
        Ok(seal) => seal,
        Err(_) => {
            audit.blockers.push("receipt_seal_invalid_or_forged".into());
            return audit;
        }
    };
    audit.receipt_seal_sha256 = Some(seal.clone());
    let receipt: Tg10OperationalReceipt = match serde_json::from_value(binding.document.clone()) {
        Ok(receipt) => receipt,
        Err(_) => {
            audit.blockers.push("receipt_grammar_invalid".into());
            return audit;
        }
    };
    if receipt.seal_sha256 != seal
        || receipt.schema != TG10_RECEIPT_SCHEMA
        || receipt.model_key != model_key
        || receipt.tg_level != 10
        || receipt.required_base_true_tps != TG10_BASE_TRUE_TPS
        || !receipt.operational_pass
        || !receipt.coherent_hcli_pass
        || !receipt.complete_token_path_measured
        || receipt.fallback_count != 0
        || !receipt.median_base_true_tps.is_finite()
        || receipt.median_base_true_tps < f64::from(TG10_BASE_TRUE_TPS)
    {
        audit
            .blockers
            .push("receipt_not_a_valid_tg10_operational_base_true_tps_pass".into());
    }
    if receipt.observed_operational_ascent_recorded_at != status.recorded_at {
        audit
            .blockers
            .push("receipt_stale_or_wrong_operational_status_timestamp".into());
    }
    match status.evidence_fingerprint.as_deref() {
        Some(fingerprint)
            if is_lower_sha256(fingerprint)
                && receipt.observed_operational_ascent_evidence_fingerprint_sha256
                    == fingerprint => {}
        _ => audit
            .blockers
            .push("receipt_stale_or_wrong_operational_status_fingerprint".into()),
    }
    let evidence = status
        .evidence
        .models
        .get(model_key)
        .expect("validated fixed operational model");
    require_matching_status_seal(
        &mut audit.blockers,
        "complete_artifact_admission",
        evidence.complete_artifact_admission_seal_sha256.as_ref(),
        &receipt.complete_artifact_admission_seal_sha256,
    );
    require_matching_status_seal(
        &mut audit.blockers,
        "source_identity",
        evidence.source_identity_seal_sha256.as_ref(),
        &receipt.source_identity_seal_sha256,
    );
    require_matching_status_seal(
        &mut audit.blockers,
        "source_revalidation",
        evidence.source_revalidation_seal_sha256.as_ref(),
        &receipt.source_revalidation_seal_sha256,
    );
    require_matching_status_seal(
        &mut audit.blockers,
        "kernel",
        evidence.kernel_receipt_seal_sha256.as_ref(),
        &receipt.kernel_receipt_seal_sha256,
    );
    require_matching_status_seal(
        &mut audit.blockers,
        "hcli",
        evidence.hcli_receipt_seal_sha256.as_ref(),
        &receipt.hcli_receipt_seal_sha256,
    );
    require_matching_status_seal(
        &mut audit.blockers,
        "runtime",
        evidence.runtime_receipt_seal_sha256.as_ref(),
        &receipt.runtime_receipt_seal_sha256,
    );
    require_matching_status_seal(
        &mut audit.blockers,
        "tg10_receipt",
        evidence.tg10_receipt_seal_sha256.as_ref(),
        &seal,
    );
    if !lane_tg10.operational_pass
        || lane_tg10.receipt_seal_sha256.as_deref() != Some(seal.as_str())
    {
        audit
            .blockers
            .push("receipt_does_not_match_lane_tg10_authority".into());
    }
    if !scheduler_tg10.operational_pass
        || scheduler_tg10.receipt_seal_sha256.as_deref() != Some(seal.as_str())
    {
        audit
            .blockers
            .push("receipt_does_not_match_scheduler_tg10_authority".into());
    }
    audit.valid_exact_fresh_tg10_operational_receipt = audit.blockers.is_empty();
    audit
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
    let (lane_digest, lane_tg10) = validate_lane_authority(&input.lane_authority)?;
    let mutation_digest = validate_mutation_authority(&input.mutation_authority, &lane_digest)?;
    let knowledge_digest =
        validate_knowledge_authority(&input.knowledge_authority, &lane_digest, &mutation_digest)?;
    let (scheduler_digest, scheduler_tg10) = validate_scheduler_authority(
        &input.scheduler_authority,
        &lane_digest,
        &mutation_digest,
        &knowledge_digest,
        &lane_tg10,
    )?;
    let (operational_digest, operational_status, mut state_blockers) =
        validate_operational_status(&input.operational_ascent_status)?;
    validate_authority_boundary(&input.authority_boundary, "input.authority_boundary")?;
    validate_execution_boundary(&input.execution_boundary, "input.execution_boundary")?;
    let qwen30_audit = audit_receipt(
        QWEN30,
        &input.qwen30_tg10_receipt,
        &operational_status,
        lane_tg10.get(QWEN30).expect("validated lane Q30"),
        scheduler_tg10.get(QWEN30).expect("validated scheduler Q30"),
    );
    let qwen80_audit = audit_receipt(
        QWEN80,
        &input.qwen80_tg10_receipt,
        &operational_status,
        lane_tg10.get(QWEN80).expect("validated lane Q80"),
        scheduler_tg10.get(QWEN80).expect("validated scheduler Q80"),
    );
    state_blockers.extend(
        qwen30_audit
            .blockers
            .iter()
            .map(|value| format!("qwen30:{value}")),
    );
    state_blockers.extend(
        qwen80_audit
            .blockers
            .iter()
            .map(|value| format!("qwen80:{value}")),
    );
    state_blockers.sort();
    state_blockers.dedup();
    let both_receipts = qwen30_audit.valid_exact_fresh_tg10_operational_receipt
        && qwen80_audit.valid_exact_fresh_tg10_operational_receipt
        && state_blockers.is_empty();
    let (status, prepared) = if input.paired_development_activation_requested {
        if both_receipts {
            state_blockers.push("activation_is_reserved_for_a_later_protected_controller".into());
            (REFUSED_AUTHORITY_STATUS.into(), false)
        } else {
            (REFUSED_EARLY_STATUS.into(), false)
        }
    } else if both_receipts {
        (PREPARED_STATUS.into(), true)
    } else {
        (REFUSED_TODAY_STATUS.into(), false)
    };
    let focused_checks = FocusedChecks {
        exact_sealed_lane_mutation_knowledge_and_scheduler_chain_bound: true,
        sealed_operational_ascent_status_bound_at_canonical_path: true,
        qwen30_and_qwen80_tg10_receipts_are_exact_fresh_and_model_matched: both_receipts,
        both_receipts_require_coherent_complete_fallback_free_base_true_tps_at_least_100:
            both_receipts,
        stale_forged_mismatched_and_early_receipts_refused: true,
        paired_development_never_activated_by_this_contract: true,
        final_manager_selection_scorecards_and_tg3_outside_activation: true,
        no_runtime_server_gpu_watcher_or_tournament_authority: true,
        execution_boundary_cpu_only: true,
    };
    let mut report = Report {
        schema: RESULT_SCHEMA,
        status,
        prepared,
        paired_development_active: false,
        paired_development_activation_authorized_by_this_contract: false,
        bound_lane_authority: lane_digest,
        bound_mutation_authority: mutation_digest,
        bound_knowledge_authority: knowledge_digest,
        bound_scheduler_authority: scheduler_digest,
        bound_operational_ascent_status: operational_digest,
        qwen30_tg10_audit: qwen30_audit,
        qwen80_tg10_audit: qwen80_audit,
        both_exact_fresh_tg10_operational_receipts_present: both_receipts,
        state_blockers,
        authority_boundary: input.authority_boundary,
        execution_boundary: input.execution_boundary,
        focused_checks,
        claim_boundary: vec![
            "This is a sealed CPU-only both-TG10 state machine, not a runtime or paired-development activator.",
            "It consumes exact sealed lane, mutation, Knowledge Plane, scheduler, and canonical operational-ascent inputs without reopening paths.",
            "Both Q30 and Q80 must have fresh exact TG10 operational receipts: coherent HCLI, complete-token BASE_TRUE_TPS measurement at least 100, zero fallback, and matching source/admission/kernel/HCLI/runtime status evidence.",
            "Receipt status timestamp and evidence fingerprint mismatches are stale; seal, model, evidence, lane, and scheduler mismatches are refused.",
            "A prepared result reserves a later protected-controller action only; it does not create a logical session, start either server, or activate a paired candidate world.",
            "TG3, final manager evaluation, scorecards, tournament launch, and winner selection remain outside this activation gate.",
            "No model/GPU/server/watcher/port/HCLI/token/TPS/TG/tournament state is touched by this contract.",
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
            .map_err(|error| format!("TG10 activation report cannot be serialized: {error}"))?,
    )
    .map_err(|error| format!("TG10 activation report cannot be sealed: {error}"))?;
    let encoded = serde_json::to_vec_pretty(&value)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&encoded)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_paired_cognition_tg10_development_activation_state_machine --input ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
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
        .map_err(|error| format!("TG10 activation validation failed: {error}"))?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_paired_cognition_tg10_development_activation_state_machine: {error}");
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

    fn tg10(pass: bool, receipt_seal_sha256: Option<String>) -> Tg10Authority {
        Tg10Authority {
            required_base_true_tps: TG10_BASE_TRUE_TPS,
            operational_pass: pass,
            coherent_hcli_pass: pass,
            complete_token_path_measured: pass,
            fallback_count: 0,
            median_base_true_tps: pass.then_some(100.0),
            receipt_seal_sha256,
        }
    }

    fn sealed_lane_document(qwen30: Tg10Authority, qwen80: Tg10Authority) -> Value {
        seal_value(json!({
            "schema": LANE_AUTHORITY_SCHEMA,
            "status": LANE_AUTHORITY_STATUS,
            "prepared": true,
            "paired_candidate_worlds_active": false,
            "no_new_physical_model_process_authority": true,
            "model_contract_bindings": [
                {"model_key": QWEN30, "tg10": qwen30},
                {"model_key": QWEN80, "tg10": qwen80},
            ],
            "authority_boundary": authority_boundary(),
            "execution_boundary": execution_boundary(),
        }))
        .unwrap()
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

    fn mutation_binding(lane: &SealedDocumentBinding) -> SealedDocumentBinding {
        binding(
            "/sealed/paired/PAIRED_COGNITION_MUTATION_AUTHORITY.json",
            seal_value(json!({
                "schema": MUTATION_AUTHORITY_SCHEMA,
                "status": MUTATION_AUTHORITY_STATUS,
                "prepared": true,
                "paired_candidate_worlds_active": false,
                "bound_lane_authority": digest_from(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
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
            "/sealed/paired/PAIRED_COGNITION_KNOWLEDGE_AUTHORITY.json",
            seal_value(json!({
                "schema": KNOWLEDGE_AUTHORITY_SCHEMA,
                "status": KNOWLEDGE_AUTHORITY_STATUS,
                "prepared": true,
                "knowledge_plane_active": false,
                "external_publication_performed": false,
                "bound_lane_authority": digest_from(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest_from(mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
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
        qwen30: &Tg10Authority,
        qwen80: &Tg10Authority,
    ) -> SealedDocumentBinding {
        let both = qwen30.operational_pass && qwen80.operational_pass;
        binding(
            "/sealed/paired/PAIRED_COGNITION_SCHEDULER_AUTHORITY.json",
            seal_value(json!({
                "schema": SCHEDULER_AUTHORITY_SCHEMA,
                "status": SCHEDULER_AUTHORITY_STATUS,
                "prepared": true,
                "paired_development_active": false,
                "logical_sessions_created": false,
                "bound_lane_authority": digest_from(lane, LANE_AUTHORITY_SCHEMA, LANE_AUTHORITY_STATUS),
                "bound_mutation_authority": digest_from(mutation, MUTATION_AUTHORITY_SCHEMA, MUTATION_AUTHORITY_STATUS),
                "bound_knowledge_authority": digest_from(knowledge, KNOWLEDGE_AUTHORITY_SCHEMA, KNOWLEDGE_AUTHORITY_STATUS),
                "paired_development_reservation": {
                    "required_base_true_tps": TG10_BASE_TRUE_TPS,
                    "qwen30_tg10": {"model_key": QWEN30, "required_base_true_tps": TG10_BASE_TRUE_TPS, "operational_pass": qwen30.operational_pass, "receipt_seal_sha256": qwen30.receipt_seal_sha256},
                    "qwen80_tg10": {"model_key": QWEN80, "required_base_true_tps": TG10_BASE_TRUE_TPS, "operational_pass": qwen80.operational_pass, "receipt_seal_sha256": qwen80.receipt_seal_sha256},
                    "both_tg10_operational_receipts_present": both,
                    "paired_development_ready_after_both_tg10": both,
                    "paired_development_activation_authorized_by_this_contract": false,
                },
                "final_mode_gate": {
                    "solo_manager_evaluation_authorized_by_this_contract": false,
                    "symmetric_orchestrator_evaluation_authorized_by_this_contract": false,
                    "winner_selection_authorized_by_this_contract": false,
                },
                "authority_boundary": authority_boundary(),
                "execution_boundary": execution_boundary(),
            }))
            .unwrap(),
        )
    }

    fn operational_model(
        model_key: &str,
        tg10_receipt: Option<String>,
        ready: bool,
    ) -> OperationalModelEvidence {
        let prefix = if model_key == QWEN30 { '1' } else { '2' };
        OperationalModelEvidence {
            complete_artifact_admission_seal_sha256: Some(sha(prefix)),
            hcli_receipt_seal_sha256: ready
                .then(|| sha(if model_key == QWEN30 { '3' } else { '4' })),
            kernel_receipt_seal_sha256: ready
                .then(|| sha(if model_key == QWEN30 { '5' } else { '6' })),
            runtime_receipt_seal_sha256: ready
                .then(|| sha(if model_key == QWEN30 { '7' } else { '8' })),
            source_identity_seal_sha256: Some(sha(if model_key == QWEN30 { '9' } else { 'a' })),
            source_revalidation_seal_sha256: Some(sha(if model_key == QWEN30 { 'b' } else { 'c' })),
            tg10_receipt_seal_sha256: tg10_receipt,
        }
    }

    fn operational_status_value(
        qwen30: OperationalModelEvidence,
        qwen80: OperationalModelEvidence,
        both: bool,
        fingerprint: Option<String>,
    ) -> Value {
        seal_value(json!({
            "schema": OPERATIONAL_ASCENT_SCHEMA,
            "status": if both { OPERATIONAL_ASCENT_READY_STATUS } else { OPERATIONAL_ASCENT_WAITING_STATUS },
            "both_valid_tg10_receipts": both,
            "evidence_fingerprint": fingerprint,
            "next_transition": if both { "paired development remains protected" } else { "await both exact sealed TG10 operational receipts" },
            "recorded_at": "2026-08-09T06:32:15.045674Z",
            "evidence": {
                "fixed_candidate_order": ["Qwen30-Gravity-Manager-Artifact", "Qwen80-Gravity-Manager-Artifact"],
                "models": {QWEN30: qwen30, QWEN80: qwen80},
            },
            "claim_boundary": {
                "operational_ascent_does_not_activate_the_sandbox": true,
                "operational_ascent_does_not_launch_or_score_the_protected_tournament": true,
                "operational_ascent_is_not_a_capability_or_agent_os_pass": true,
                "operational_ascent_is_not_tg3": true,
            },
            "protected_tournament": {
                "capability_agent_os_and_final_review_remain_required": true,
                "launch_requested": false,
                "qualification_override": false,
                "tg3_remains_required": true,
            },
        }))
        .unwrap()
    }

    fn receipt_document(model_key: &str, status: &OperationalAscentStatus) -> Value {
        let evidence = status.evidence.models.get(model_key).unwrap();
        seal_value(json!({
            "schema": TG10_RECEIPT_SCHEMA,
            "model_key": model_key,
            "tg_level": 10,
            "required_base_true_tps": TG10_BASE_TRUE_TPS,
            "operational_pass": true,
            "coherent_hcli_pass": true,
            "complete_token_path_measured": true,
            "fallback_count": 0,
            "median_base_true_tps": 100.0,
            "complete_artifact_admission_seal_sha256": evidence.complete_artifact_admission_seal_sha256,
            "source_identity_seal_sha256": evidence.source_identity_seal_sha256,
            "source_revalidation_seal_sha256": evidence.source_revalidation_seal_sha256,
            "kernel_receipt_seal_sha256": evidence.kernel_receipt_seal_sha256,
            "hcli_receipt_seal_sha256": evidence.hcli_receipt_seal_sha256,
            "runtime_receipt_seal_sha256": evidence.runtime_receipt_seal_sha256,
            "observed_operational_ascent_recorded_at": status.recorded_at,
            "observed_operational_ascent_evidence_fingerprint_sha256": operational_evidence_fingerprint(status).unwrap(),
        }))
        .unwrap()
    }

    fn input_fixture() -> Input {
        let qwen30 = tg10(false, None);
        let qwen80 = tg10(false, None);
        let lane = binding(
            "/sealed/paired/PAIRED_COGNITION_LANE_AUTHORITY.json",
            sealed_lane_document(qwen30.clone(), qwen80.clone()),
        );
        let mutation = mutation_binding(&lane);
        let knowledge = knowledge_binding(&lane, &mutation);
        let scheduler = scheduler_binding(&lane, &mutation, &knowledge, &qwen30, &qwen80);
        let status = binding(
            OPERATIONAL_STATUS_PATH,
            operational_status_value(
                operational_model(QWEN30, None, false),
                operational_model(QWEN80, None, false),
                false,
                None,
            ),
        );
        Input {
            schema: INPUT_SCHEMA.into(),
            lane_authority: lane,
            mutation_authority: mutation,
            knowledge_authority: knowledge,
            scheduler_authority: scheduler,
            operational_ascent_status: status,
            qwen30_tg10_receipt: None,
            qwen80_tg10_receipt: None,
            paired_development_activation_requested: false,
            authority_boundary: authority_boundary(),
            execution_boundary: execution_boundary(),
            seal_sha256: sha('f'),
        }
    }

    fn ready_input() -> Input {
        let status_seed_value = operational_status_value(
            operational_model(QWEN30, None, true),
            operational_model(QWEN80, None, true),
            true,
            Some(sha('0')),
        );
        let status_seed: OperationalAscentStatus =
            serde_json::from_value(status_seed_value).unwrap();
        let fingerprint = operational_evidence_fingerprint(&status_seed).unwrap();
        let qwen30_doc = receipt_document(QWEN30, &status_seed);
        let qwen80_doc = receipt_document(QWEN80, &status_seed);
        let qwen30_seal = qwen30_doc["seal_sha256"].as_str().unwrap().to_owned();
        let qwen80_seal = qwen80_doc["seal_sha256"].as_str().unwrap().to_owned();
        let qwen30 = tg10(true, Some(qwen30_seal));
        let qwen80 = tg10(true, Some(qwen80_seal));
        let lane = binding(
            "/sealed/paired/PAIRED_COGNITION_LANE_AUTHORITY.json",
            sealed_lane_document(qwen30.clone(), qwen80.clone()),
        );
        let mutation = mutation_binding(&lane);
        let knowledge = knowledge_binding(&lane, &mutation);
        let scheduler = scheduler_binding(&lane, &mutation, &knowledge, &qwen30, &qwen80);
        let status = binding(
            OPERATIONAL_STATUS_PATH,
            operational_status_value(
                operational_model(
                    QWEN30,
                    Some(qwen30.receipt_seal_sha256.clone().unwrap()),
                    true,
                ),
                operational_model(
                    QWEN80,
                    Some(qwen80.receipt_seal_sha256.clone().unwrap()),
                    true,
                ),
                true,
                Some(fingerprint),
            ),
        );
        Input {
            schema: INPUT_SCHEMA.into(),
            lane_authority: lane,
            mutation_authority: mutation,
            knowledge_authority: knowledge,
            scheduler_authority: scheduler,
            operational_ascent_status: status,
            qwen30_tg10_receipt: Some(binding("/sealed/tg10/qwen30.json", qwen30_doc)),
            qwen80_tg10_receipt: Some(binding("/sealed/tg10/qwen80.json", qwen80_doc)),
            paired_development_activation_requested: false,
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
    fn current_operational_status_refuses_today_without_tg10_receipts() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.schema, RESULT_SCHEMA);
        assert_eq!(report.status, REFUSED_TODAY_STATUS);
        assert!(!report.prepared);
        assert!(!report.paired_development_active);
        assert!(!report.both_exact_fresh_tg10_operational_receipts_present);
        assert!(report
            .state_blockers
            .iter()
            .any(|blocker| blocker.contains("tg10_receipt_not_present")));
    }

    /// §10 hard gate: both operational passes FALSE (current truth) must refuse
    /// any transition toward MANAGER_ASCENT_TOURNAMENT_ACTIVE and must name the
    /// unmet both-receipt condition.
    #[test]
    fn both_tg10_false_refuses_and_names_unmet_both_receipt_condition() {
        let report = build_report(input_fixture()).unwrap();
        assert_eq!(report.status, REFUSED_TODAY_STATUS);
        assert_ne!(
            report.status.as_str(),
            "MANAGER_ASCENT_TOURNAMENT_ACTIVE",
            "gate must never reach manager-ascent tournament active while both TG10 passes are false"
        );
        assert!(!report.paired_development_active);
        assert!(!report.paired_development_activation_authorized_by_this_contract);
        assert!(!report.both_exact_fresh_tg10_operational_receipts_present);
        assert!(
            report.state_blockers.iter().any(|blocker| {
                blocker.contains("operational_status_both_valid_tg10_receipts_false")
                    || blocker.contains("tg10_receipt_not_present")
            }),
            "refusal must name the unmet both-TG10 condition; blockers={:?}",
            report.state_blockers
        );
        assert!(report
            .state_blockers
            .iter()
            .any(|blocker| blocker.starts_with("qwen30:") && blocker.contains("tg10_receipt_not_present")));
        assert!(report
            .state_blockers
            .iter()
            .any(|blocker| blocker.starts_with("qwen80:") && blocker.contains("tg10_receipt_not_present")));
    }

    /// Exactly one TG10 operational pass is the dangerous partial-success case.
    /// §10 requires BOTH; a single true pass must still refuse.
    fn one_true_input(winner: &str) -> Input {
        let loser = if winner == QWEN30 { QWEN80 } else { QWEN30 };
        let status_seed_value = operational_status_value(
            operational_model(QWEN30, None, winner == QWEN30),
            operational_model(QWEN80, None, winner == QWEN80),
            false,
            None,
        );
        let status_seed: OperationalAscentStatus =
            serde_json::from_value(status_seed_value).unwrap();
        let fingerprint = operational_evidence_fingerprint(&status_seed).unwrap();
        let winner_doc = receipt_document(winner, &status_seed);
        let winner_seal = winner_doc["seal_sha256"].as_str().unwrap().to_owned();
        let qwen30 = if winner == QWEN30 {
            tg10(true, Some(winner_seal.clone()))
        } else {
            tg10(false, None)
        };
        let qwen80 = if winner == QWEN80 {
            tg10(true, Some(winner_seal.clone()))
        } else {
            tg10(false, None)
        };
        let lane = binding(
            "/sealed/paired/PAIRED_COGNITION_LANE_AUTHORITY.json",
            sealed_lane_document(qwen30.clone(), qwen80.clone()),
        );
        let mutation = mutation_binding(&lane);
        let knowledge = knowledge_binding(&lane, &mutation);
        let scheduler = scheduler_binding(&lane, &mutation, &knowledge, &qwen30, &qwen80);
        let status = binding(
            OPERATIONAL_STATUS_PATH,
            operational_status_value(
                operational_model(
                    QWEN30,
                    qwen30.receipt_seal_sha256.clone(),
                    winner == QWEN30,
                ),
                operational_model(
                    QWEN80,
                    qwen80.receipt_seal_sha256.clone(),
                    winner == QWEN80,
                ),
                false,
                Some(fingerprint),
            ),
        );
        let (qwen30_receipt, qwen80_receipt) = if winner == QWEN30 {
            (
                Some(binding("/sealed/tg10/qwen30.json", winner_doc)),
                None,
            )
        } else {
            (
                None,
                Some(binding("/sealed/tg10/qwen80.json", winner_doc)),
            )
        };
        let _ = loser;
        Input {
            schema: INPUT_SCHEMA.into(),
            lane_authority: lane,
            mutation_authority: mutation,
            knowledge_authority: knowledge,
            scheduler_authority: scheduler,
            operational_ascent_status: status,
            qwen30_tg10_receipt: qwen30_receipt,
            qwen80_tg10_receipt: qwen80_receipt,
            paired_development_activation_requested: false,
            authority_boundary: authority_boundary(),
            execution_boundary: execution_boundary(),
            seal_sha256: sha('f'),
        }
    }

    #[test]
    fn exactly_one_tg10_true_still_refuses_because_both_are_required() {
        for winner in [QWEN30, QWEN80] {
            let report = build_report(one_true_input(winner)).unwrap();
            assert_eq!(
                report.status, REFUSED_TODAY_STATUS,
                "exactly one true ({winner}) must still refuse"
            );
            assert_ne!(report.status.as_str(), "MANAGER_ASCENT_TOURNAMENT_ACTIVE");
            assert!(!report.prepared);
            assert!(!report.paired_development_active);
            assert!(!report.both_exact_fresh_tg10_operational_receipts_present);
            assert!(
                report.state_blockers.iter().any(|blocker| {
                    blocker.contains("operational_status_both_valid_tg10_receipts_false")
                        || blocker.contains("tg10_receipt_not_present")
                }),
                "one-true refusal must name unmet both-receipt condition; winner={winner} blockers={:?}",
                report.state_blockers
            );
            let loser_prefix = if winner == QWEN30 { "qwen80:" } else { "qwen30:" };
            assert!(
                report
                    .state_blockers
                    .iter()
                    .any(|blocker| blocker.starts_with(loser_prefix)
                        && blocker.contains("tg10_receipt_not_present")),
                "loser receipt absence must be named; winner={winner} blockers={:?}",
                report.state_blockers
            );
        }
    }

    #[test]
    fn both_exact_fresh_tg10_receipts_can_only_prepare_not_activate() {
        let report = build_report(ready_input()).unwrap();
        assert_eq!(report.status, PREPARED_STATUS);
        assert!(report.prepared);
        assert!(report.both_exact_fresh_tg10_operational_receipts_present);
        assert!(!report.paired_development_active);
        assert!(!report.paired_development_activation_authorized_by_this_contract);
    }

    /// Fixture-only: when both exact TG10 receipts are bound, the state machine
    /// permits the PREPARED reservation transition. It must still never emit
    /// MANAGER_ASCENT_TOURNAMENT_ACTIVE or any activation receipt.
    #[test]
    fn both_tg10_true_permits_prepared_reservation_not_manager_ascent_activation() {
        let report = build_report(ready_input()).unwrap();
        assert_eq!(report.status, PREPARED_STATUS);
        assert!(report.prepared);
        assert!(report.both_exact_fresh_tg10_operational_receipts_present);
        assert_ne!(report.status.as_str(), "MANAGER_ASCENT_TOURNAMENT_ACTIVE");
        assert!(!report.paired_development_active);
        assert!(!report.paired_development_activation_authorized_by_this_contract);
        assert!(report
            .focused_checks
            .paired_development_never_activated_by_this_contract);
    }

    #[test]
    fn tampered_upstream_lane_seal_is_refused() {
        let mut input = input_fixture();
        input.lane_authority.document["seal_sha256"] = Value::String(sha('0'));
        let error = build_report(input).unwrap_err();
        assert!(
            error.contains("seal_sha256") || error.contains("lane_authority"),
            "tampered upstream seal must be refused; error={error}"
        );
    }

    #[test]
    fn tampered_upstream_mutation_document_sha256_is_refused() {
        let mut input = input_fixture();
        input.mutation_authority.document_sha256 = sha('0');
        let error = build_report(input).unwrap_err();
        assert!(
            error.contains("document_sha256") || error.contains("mutation_authority"),
            "tampered upstream document_sha256 must be refused; error={error}"
        );
    }

    #[test]
    fn early_activation_request_is_refused_even_with_current_authority_chain() {
        let mut input = input_fixture();
        input.paired_development_activation_requested = true;
        let report = build_report(input).unwrap();
        assert_eq!(report.status, REFUSED_EARLY_STATUS);
        assert!(!report.prepared);
    }

    #[test]
    fn activation_request_after_valid_receipts_is_still_outside_this_contract() {
        let mut input = ready_input();
        input.paired_development_activation_requested = true;
        let report = build_report(input).unwrap();
        assert_eq!(report.status, REFUSED_AUTHORITY_STATUS);
        assert!(!report.prepared);
    }

    #[test]
    fn forged_stale_and_mismatched_receipts_are_refused() {
        let mut forged = ready_input();
        forged.qwen30_tg10_receipt.as_mut().unwrap().document["seal_sha256"] =
            Value::String(sha('0'));
        let report = build_report(forged).unwrap();
        assert_eq!(report.status, REFUSED_TODAY_STATUS);
        assert!(
            !report
                .qwen30_tg10_audit
                .valid_exact_fresh_tg10_operational_receipt
        );

        let mut stale = ready_input();
        stale.qwen80_tg10_receipt.as_mut().unwrap().document
            ["observed_operational_ascent_recorded_at"] =
            Value::String("1999-01-01T00:00:00Z".into());
        let document = stale.qwen80_tg10_receipt.as_ref().unwrap().document.clone();
        stale.qwen80_tg10_receipt.as_mut().unwrap().document = seal_value(document).unwrap();
        stale.qwen80_tg10_receipt.as_mut().unwrap().document_sha256 =
            sha256_json(&stale.qwen80_tg10_receipt.as_ref().unwrap().document).unwrap();
        let report = build_report(stale).unwrap();
        assert_eq!(report.status, REFUSED_TODAY_STATUS);
        assert!(report
            .qwen80_tg10_audit
            .blockers
            .iter()
            .any(|blocker| blocker.contains("stale")));

        let mut mismatched = ready_input();
        mismatched.qwen30_tg10_receipt = mismatched.qwen80_tg10_receipt.clone();
        let report = build_report(mismatched).unwrap();
        assert_eq!(report.status, REFUSED_TODAY_STATUS);
        assert!(
            !report
                .qwen30_tg10_audit
                .valid_exact_fresh_tg10_operational_receipt
        );
    }

    #[test]
    fn authority_chain_and_final_selection_boundaries_are_fail_closed() {
        let mut wrong_chain = ready_input();
        wrong_chain.scheduler_authority.document["bound_knowledge_authority"]["document_sha256"] =
            Value::String(sha('0'));
        let document = wrong_chain.scheduler_authority.document.clone();
        wrong_chain.scheduler_authority.document = seal_value(document).unwrap();
        wrong_chain.scheduler_authority.document_sha256 =
            sha256_json(&wrong_chain.scheduler_authority.document).unwrap();
        assert!(build_report(wrong_chain).is_err());

        let mut final_selection = ready_input();
        final_selection.scheduler_authority.document["final_mode_gate"]
            ["winner_selection_authorized_by_this_contract"] = Value::Bool(true);
        let document = final_selection.scheduler_authority.document.clone();
        final_selection.scheduler_authority.document = seal_value(document).unwrap();
        final_selection.scheduler_authority.document_sha256 =
            sha256_json(&final_selection.scheduler_authority.document).unwrap();
        assert!(build_report(final_selection).is_err());
    }

    #[test]
    fn runtime_gpu_server_and_tournament_boundaries_are_fail_closed() {
        let mut runtime = input_fixture();
        runtime.execution_boundary.runtime_watcher_or_server_started = true;
        assert!(build_report(runtime).is_err());

        let mut lease = input_fixture();
        lease.authority_boundary.gpu_leases_authorized = 1;
        assert!(build_report(lease).is_err());
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
        assert_eq!(output["status"], Value::String(REFUSED_TODAY_STATUS.into()));
        assert!(
            write_report_create_new(&output_path, &build_report(input_fixture()).unwrap()).is_err()
        );
    }

    #[test]
    fn tampered_outer_input_seal_is_rejected_before_state_evaluation() {
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
