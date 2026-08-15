//! CPU-only Qwen3-Coder-Next one-resident-model logical-session scheduler
//! contract.
//!
//! This target is deliberately not a Qwen80 runtime, server, watcher, model
//! loader, artifact scanner, Metal program, GPU lease, HCLI invocation, token
//! loop, or throughput measurement. It consumes four already-produced
//! preflight documents as external authority and defines the future scheduler
//! boundary for exactly one resident Q80 model instance and 1/2/4/8/16
//! logical sessions:
//!
//! * immutable model weights are represented once, never once per session;
//! * each logical session has a bounded, independently named state/KV arena;
//! * a FIFO decode slot is released while a session waits on a tool;
//! * a resumed tool-waiting session returns at the queue tail;
//! * checkpoint and rollback restore the session identity exactly; and
//! * every session carries the same admitted source-template authority with a
//!   selected source-template prompt binding.
//!
//! The scheduler fixtures below manipulate only SHA-256 commitments to future
//! caller-owned session state. They never allocate state, load a model,
//! execute a layer, generate a token, bind a port, or launch a process.
//! Consequently, a successful report remains PREPARED/INCOMPLETE and is only
//! an input to a future truthful HCLI scheduler implementation.
//!
//! Example:
//!
//! cargo run -p hawking-core --example ascension_qwen80_one_resident_session_scheduler_contract -- \
//!   --activation-contract /absolute/path/QWEN80_ACTIVATION_GATE.json \
//!   --memory-envelope-contract /absolute/path/QWEN80_MEMORY_ENVELOPE.json \
//!   --state-contract /absolute/path/QWEN80_DECODE_STATE_CONTRACT.json \
//!   --source-template-contract /absolute/path/QWEN80_SOURCE_TEMPLATE_AUTHORITY.json \
//!   --out /absolute/new/path/QWEN80_ONE_RESIDENT_SESSION_SCHEDULER.json

use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_one_resident_session_scheduler_contract.v1";
const STATUS: &str =
    "PREPARED_INCOMPLETE_QWEN80_ONE_RESIDENT_MANY_LOGICAL_SESSIONS_SCHEDULER_NO_RUNTIME_SERVER_OR_TPS";

const ACTIVATION_SCHEMA: &str = "hawking.ascension.qwen80_resident_server_activation_result.v1";
const ACTIVATION_REFUSED_STATUS: &str =
    "REFUSED_QWEN80_ONE_RESIDENT_SERVER_ACTIVATION_NOT_READY_NO_SERVER";
const ACTIVATION_ELIGIBLE_STATUS: &str =
    "ELIGIBLE_QWEN80_ONE_RESIDENT_SERVER_AUTOMATIC_LAUNCH_PRECONDITION_ONLY";
const MEMORY_SCHEMA: &str = "hawking.ascension.qwen80_resident_memory_envelope_receipt.v1";
const MEMORY_STATUS: &str =
    "PREPARED_INCOMPLETE_QWEN80_ONE_RESIDENT_MANY_LOGICAL_SESSIONS_MEMORY_ENVELOPE_PREFLIGHT_NO_RUNTIME_SERVER_OR_TPS";
const STATE_SCHEMA: &str = "hawking.ascension.qwen80_decode_state_contract.v1";
const STATE_STATUS: &str =
    "NOT_RUNTIME_NO_TOKEN_NO_HCLI_NO_TPS_QWEN80_HYBRID_DECODE_STATE_KV_CONTRACT";
const TEMPLATE_SCHEMA: &str = "hawking.ascension.qwen80_source_template_ab_authority_preflight.v1";
const TEMPLATE_STATUS: &str = "PREPARED_NOT_EXECUTED_QWEN80_SOURCE_TEMPLATE_AB_AUTHORITY";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const QWEN80_HOST: &str = "127.0.0.1";
const QWEN80_PORT: usize = 18_480;
const QWEN30_PORT: usize = 18_430;
const MAX_SEQUENCE_LENGTH: usize = 4_096;
const LOGICAL_SESSION_COUNTS: [usize; 5] = [1, 2, 4, 8, 16];

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct FileEvidence {
    path: String,
    bytes: usize,
    document_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ExternalContract {
    contract: FileEvidence,
    schema: String,
    status: String,
    unsealed_preimage_sha256: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct StateAuthority {
    contract: ExternalContract,
    max_seq_len: usize,
    per_session_state_records: usize,
    linear_state_slots: usize,
    gqa_kv_slots: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MemoryProfileAuthority {
    logical_sessions: usize,
    q80_planned_resident_bytes: usize,
    state_and_kv_bytes: usize,
    session_control_bytes: usize,
    static_snapshot_envelope_satisfied: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct MemoryAuthority {
    contract: ExternalContract,
    resident_weight_bytes: usize,
    shared_runtime_bytes: usize,
    per_session_state_and_kv_bytes: usize,
    per_session_control_bytes: usize,
    profiles: Vec<MemoryProfileAuthority>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourcePromptBinding {
    label: String,
    rendered_prompt_sha256: String,
    token_ids_sha256: String,
    token_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceTemplateAuthority {
    contract: ExternalContract,
    model_id: String,
    source_repository: String,
    source_revision: String,
    tokenizer_vocab_size: usize,
    lm_head_vocab_size: usize,
    prompt_bindings: Vec<SourcePromptBinding>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ExternalPreconditions {
    activation: ExternalContract,
    memory: MemoryAuthority,
    state: StateAuthority,
    source_template: SourceTemplateAuthority,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct LogicalSessionProfile {
    logical_sessions: usize,
    resident_q80_model_processes: usize,
    immutable_weight_copies: usize,
    resident_weight_bytes: usize,
    shared_runtime_bytes: usize,
    per_session_state_and_kv_bytes: usize,
    per_session_control_bytes: usize,
    per_session_total_bytes: usize,
    aggregate_session_bytes: usize,
    planned_resident_bytes: usize,
    distinct_state_namespace_count: usize,
    source_template_authority_document_sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SourceTemplateHandoff {
    source_template_authority_document_sha256: String,
    source_template_authority_preimage_sha256: String,
    prompt_bindings: Vec<SourcePromptBinding>,
    handoff_rule: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SchedulerSemantics {
    topology: &'static str,
    decode_slot_count: usize,
    queue_discipline: &'static str,
    tool_wait_behavior: &'static str,
    state_ownership: &'static str,
    checkpoint_and_rollback: &'static str,
    source_template_handoff: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct FixtureChecks {
    exact_one_two_four_eight_sixteen_profiles_checked: bool,
    one_immutable_weight_copy_per_profile_checked: bool,
    per_session_state_and_kv_bound_checked: bool,
    state_namespace_aliasing_rejected: bool,
    fifo_fairness_after_tool_wait_checked: bool,
    tool_wait_releases_decode_slot_checked: bool,
    checkpoint_rollback_identity_checked: bool,
    source_template_authority_handoff_checked: bool,
    unsupported_session_count_rejected: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct ExecutionBoundary {
    model_weights_loaded: bool,
    artifact_or_payload_scanned: bool,
    device_or_metal_used: bool,
    gpu_lease_or_registry_mutated: bool,
    runtime_watcher_or_server_started: bool,
    port_bound_or_listener_created: bool,
    decoder_or_model_token_executed: bool,
    hcli_executed: bool,
    tps_or_tg_measured: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    prepared: bool,
    complete_decoder_readiness_earned: bool,
    real_gravity_server_launch_precondition_satisfied: bool,
    external_preconditions: ExternalPreconditions,
    source_template_handoff: SourceTemplateHandoff,
    logical_session_profiles: Vec<LogicalSessionProfile>,
    scheduler_semantics: SchedulerSemantics,
    focused_fixture_checks: FixtureChecks,
    execution_boundary: ExecutionBoundary,
    claim_boundary: Vec<&'static str>,
    unsealed_preimage_sha256: String,
}

struct Args {
    activation_contract: PathBuf,
    memory_envelope_contract: PathBuf,
    state_contract: PathBuf,
    source_template_contract: PathBuf,
    out: PathBuf,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json<T: Serialize>(value: &T) -> String {
    sha256_hex(&serde_json::to_vec(value).expect("scheduler fixture serialization must succeed"))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn checked_add(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("{label} overflowed"))
}

fn checked_mul(left: usize, right: usize, label: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflowed"))
}

fn expected_session_count(value: usize) -> Result<(), String> {
    if LOGICAL_SESSION_COUNTS.contains(&value) {
        Ok(())
    } else {
        Err(format!(
            "logical session count {value} is unsupported; expected one of {:?}",
            LOGICAL_SESSION_COUNTS
        ))
    }
}

fn regular_json(path: &Path, label: &str) -> Result<(FileEvidence, Value), String> {
    if !path.is_absolute() {
        return Err(format!("{label} path must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    let canonical = path
        .canonicalize()
        .map_err(|error| format!("{label} canonicalization failed: {error}"))?;
    let bytes = fs::read(&canonical).map_err(|error| format!("{label} read failed: {error}"))?;
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;
    Ok((
        FileEvidence {
            path: canonical.display().to_string(),
            bytes: bytes.len(),
            document_sha256: sha256_hex(&bytes),
        },
        value,
    ))
}

fn root_object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} root must be a JSON object"))
}

fn object<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} missing object {field:?}"))
}

fn array<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} missing array {field:?}"))
}

fn string<'a>(value: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} missing non-empty string {field:?}"))
}

fn string_eq(
    value: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let observed = string(value, field, label)?;
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "{label}.{field} drifted: expected {expected:?}, observed {observed:?}"
        ))
    }
}

fn usize_value(value: &Map<String, Value>, field: &str, label: &str) -> Result<usize, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| format!("{label} missing usize {field:?}"))
}

fn usize_eq(
    value: &Map<String, Value>,
    field: &str,
    expected: usize,
    label: &str,
) -> Result<(), String> {
    let observed = usize_value(value, field, label)?;
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "{label}.{field} drifted: expected {expected}, observed {observed}"
        ))
    }
}

fn bool_eq(
    value: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    let observed = value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} missing bool {field:?}"))?;
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "{label}.{field} drifted: expected {expected}, observed {observed}"
        ))
    }
}

fn optional_preimage(value: &Map<String, Value>, label: &str) -> Result<Option<String>, String> {
    match value.get("unsealed_preimage_sha256") {
        None => Ok(None),
        Some(Value::String(preimage)) if is_lower_sha256(preimage) => Ok(Some(preimage.clone())),
        Some(_) => Err(format!(
            "{label}.unsealed_preimage_sha256 must be a lowercase SHA-256 when present"
        )),
    }
}

fn contract(
    evidence: FileEvidence,
    root: &Map<String, Value>,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<ExternalContract, String> {
    string_eq(root, "schema", schema, label)?;
    string_eq(root, "status", status, label)?;
    Ok(ExternalContract {
        contract: evidence,
        schema: schema.into(),
        status: status.into(),
        unsealed_preimage_sha256: optional_preimage(root, label)?,
    })
}

fn bools_false(root: &Map<String, Value>, label: &str, fields: &[&str]) -> Result<(), String> {
    for field in fields {
        bool_eq(root, field, false, label)?;
    }
    Ok(())
}

fn load_activation(path: &Path) -> Result<ExternalContract, String> {
    let (evidence, document) = regular_json(path, "activation contract")?;
    let root = root_object(&document, "activation contract")?;
    string_eq(root, "schema", ACTIVATION_SCHEMA, "activation contract")?;
    let status = string(root, "status", "activation contract")?;
    if status != ACTIVATION_REFUSED_STATUS && status != ACTIVATION_ELIGIBLE_STATUS {
        return Err("activation contract has an unrecognized status".into());
    }
    let target = object(root, "target_topology", "activation contract")?;
    usize_eq(
        target,
        "resident_q80_model_processes",
        1,
        "activation contract.target_topology",
    )?;
    string_eq(
        target,
        "logical_sessions",
        "many",
        "activation contract.target_topology",
    )?;
    let endpoint = object(target, "endpoint", "activation contract.target_topology")?;
    string_eq(
        endpoint,
        "host",
        QWEN80_HOST,
        "activation contract.target_topology.endpoint",
    )?;
    usize_eq(
        endpoint,
        "port",
        QWEN80_PORT,
        "activation contract.target_topology.endpoint",
    )?;
    usize_eq(
        target,
        "qwen30_port_reuse_refused",
        QWEN30_PORT,
        "activation contract.target_topology",
    )?;
    let launch = object(root, "automatic_launch_contract", "activation contract")?;
    usize_eq(
        launch,
        "processes_to_start",
        1,
        "activation contract.automatic_launch_contract",
    )?;
    bool_eq(
        launch,
        "duplicate_model_process_start_prohibited",
        true,
        "activation contract.automatic_launch_contract",
    )?;
    bool_eq(
        launch,
        "gate_starts_no_process",
        true,
        "activation contract.automatic_launch_contract",
    )?;
    let boundary = object(root, "claim_boundary", "activation contract")?;
    for field in [
        "gate_started_no_server",
        "gate_bound_no_port",
        "gate_opened_no_model_artifact",
        "gate_executed_no_model_token",
        "gate_executed_no_hcli_request",
        "gate_measured_no_tps_or_tg",
    ] {
        bool_eq(boundary, field, true, "activation contract.claim_boundary")?;
    }
    Ok(ExternalContract {
        contract: evidence,
        schema: ACTIVATION_SCHEMA.into(),
        status: status.into(),
        unsealed_preimage_sha256: optional_preimage(root, "activation contract")?,
    })
}

fn load_memory(path: &Path) -> Result<MemoryAuthority, String> {
    let (evidence, document) = regular_json(path, "memory-envelope contract")?;
    let root = root_object(&document, "memory-envelope contract")?;
    let contract = contract(
        evidence,
        root,
        MEMORY_SCHEMA,
        MEMORY_STATUS,
        "memory-envelope contract",
    )?;
    bool_eq(root, "prepared", true, "memory-envelope contract")?;
    bools_false(
        root,
        "memory-envelope contract",
        &[
            "complete_decoder_readiness_earned",
            "real_gravity_server_launch_precondition_satisfied",
            "memory_envelope_healthy",
            "actual_resident_q80_rss_measured",
            "actual_device_allocation_performed",
            "actual_host_memory_probe_performed_by_this_preflight",
            "actual_runtime_or_server_launch_performed",
            "actual_hcli_or_tps_or_tg_measurement_performed",
        ],
    )?;
    bool_eq(
        root,
        "one_q80_process_envelope",
        true,
        "memory-envelope contract",
    )?;
    let topology = object(
        root,
        "fixed_one_process_topology",
        "memory-envelope contract",
    )?;
    usize_eq(
        topology,
        "resident_q80_model_processes",
        1,
        "memory-envelope contract.fixed_one_process_topology",
    )?;
    usize_eq(
        topology,
        "maximum_logical_sessions",
        LOGICAL_SESSION_COUNTS[LOGICAL_SESSION_COUNTS.len() - 1],
        "memory-envelope contract.fixed_one_process_topology",
    )?;
    usize_eq(
        topology,
        "bounded_max_seq_len",
        MAX_SEQUENCE_LENGTH,
        "memory-envelope contract.fixed_one_process_topology",
    )?;
    let endpoint = object(
        topology,
        "endpoint",
        "memory-envelope contract.fixed_one_process_topology",
    )?;
    string_eq(
        endpoint,
        "host",
        QWEN80_HOST,
        "memory-envelope contract.fixed_one_process_topology.endpoint",
    )?;
    usize_eq(
        endpoint,
        "port",
        QWEN80_PORT,
        "memory-envelope contract.fixed_one_process_topology.endpoint",
    )?;
    let supported = array(
        topology,
        "logical_sessions_supported",
        "memory-envelope contract.fixed_one_process_topology",
    )?;
    let supported_counts = supported
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "logical_sessions_supported must contain usize values".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if supported_counts.as_slice() != LOGICAL_SESSION_COUNTS {
        return Err(format!(
            "memory-envelope contract logical-session support drifted: expected {:?}, observed {:?}",
            LOGICAL_SESSION_COUNTS, supported_counts
        ));
    }
    let source = object(root, "source_artifact_binding", "memory-envelope contract")?;
    string_eq(
        source,
        "model_id",
        MODEL_ID,
        "memory-envelope contract.source_artifact_binding",
    )?;
    string_eq(
        source,
        "model_key",
        MODEL_KEY,
        "memory-envelope contract.source_artifact_binding",
    )?;
    string_eq(
        source,
        "source_repository",
        SOURCE_REPOSITORY,
        "memory-envelope contract.source_artifact_binding",
    )?;
    string_eq(
        source,
        "source_revision",
        SOURCE_REVISION,
        "memory-envelope contract.source_artifact_binding",
    )?;
    let resident_weight_bytes = usize_value(
        source,
        "resident_weight_bytes",
        "memory-envelope contract.source_artifact_binding",
    )?;
    if resident_weight_bytes == 0 {
        return Err("memory-envelope contract has zero resident weight bytes".into());
    }
    let allocations = object(
        root,
        "planned_resident_allocations",
        "memory-envelope contract",
    )?;
    usize_eq(
        allocations,
        "resident_weights_bytes",
        resident_weight_bytes,
        "memory-envelope contract.planned_resident_allocations",
    )?;
    let shared_runtime_bytes = usize_value(
        allocations,
        "shared_runtime_buffers_bytes",
        "memory-envelope contract.planned_resident_allocations",
    )?;
    let per_session = object(
        allocations,
        "per_logical_session_buffers",
        "memory-envelope contract.planned_resident_allocations",
    )?;
    let per_session_state_and_kv_bytes = usize_value(
        per_session,
        "state_and_kv_bytes",
        "memory-envelope contract.planned_resident_allocations.per_logical_session_buffers",
    )?;
    let per_session_control_bytes = usize_value(
        per_session,
        "session_control_bytes",
        "memory-envelope contract.planned_resident_allocations.per_logical_session_buffers",
    )?;
    if per_session_state_and_kv_bytes == 0 || per_session_control_bytes == 0 {
        return Err(
            "memory-envelope contract has an unbounded or zero per-session allocation".into(),
        );
    }
    let profiles = array(
        root,
        "logical_session_memory_profiles",
        "memory-envelope contract",
    )?;
    if profiles.len() != LOGICAL_SESSION_COUNTS.len() {
        return Err(
            "memory-envelope contract must contain exactly five logical-session profiles".into(),
        );
    }
    let mut parsed_profiles = Vec::with_capacity(profiles.len());
    for (&expected_sessions, profile) in LOGICAL_SESSION_COUNTS.iter().zip(profiles) {
        let profile = root_object(profile, "memory-envelope logical-session profile")?;
        usize_eq(
            profile,
            "logical_sessions",
            expected_sessions,
            "memory-envelope logical-session profile",
        )?;
        usize_eq(
            profile,
            "resident_q80_model_processes",
            1,
            "memory-envelope logical-session profile",
        )?;
        let state_and_kv_bytes = checked_add(
            checked_add(
                usize_value(
                    profile,
                    "deltanet_state_bytes",
                    "memory-envelope logical-session profile",
                )?,
                usize_value(
                    profile,
                    "gqa_key_cache_bytes",
                    "memory-envelope logical-session profile",
                )?,
                "memory-envelope profile state bytes",
            )?,
            usize_value(
                profile,
                "gqa_value_cache_bytes",
                "memory-envelope logical-session profile",
            )?,
            "memory-envelope profile state bytes",
        )?;
        let expected_state = checked_mul(
            expected_sessions,
            per_session_state_and_kv_bytes,
            "memory-envelope scaled state/KV bytes",
        )?;
        if state_and_kv_bytes != expected_state {
            return Err(
                "memory-envelope profile state/KV bytes do not scale exactly per session".into(),
            );
        }
        let control_bytes = usize_value(
            profile,
            "session_control_buffers_bytes",
            "memory-envelope logical-session profile",
        )?;
        let expected_control = checked_mul(
            expected_sessions,
            per_session_control_bytes,
            "memory-envelope scaled control bytes",
        )?;
        if control_bytes != expected_control {
            return Err(
                "memory-envelope profile control bytes do not scale exactly per session".into(),
            );
        }
        let q80_planned_resident_bytes = usize_value(
            profile,
            "q80_planned_resident_bytes",
            "memory-envelope logical-session profile",
        )?;
        let expected_total = checked_add(
            checked_add(
                resident_weight_bytes,
                shared_runtime_bytes,
                "memory-envelope resident bytes",
            )?,
            checked_add(
                expected_state,
                expected_control,
                "memory-envelope session bytes",
            )?,
            "memory-envelope resident bytes",
        )?;
        if q80_planned_resident_bytes != expected_total {
            return Err(
                "memory-envelope profile duplicated or omitted resident-model/session bytes".into(),
            );
        }
        let static_snapshot_envelope_satisfied = profile
            .get("static_snapshot_envelope_satisfied")
            .and_then(Value::as_bool)
            .ok_or("memory-envelope profile missing static_snapshot_envelope_satisfied")?;
        parsed_profiles.push(MemoryProfileAuthority {
            logical_sessions: expected_sessions,
            q80_planned_resident_bytes,
            state_and_kv_bytes,
            session_control_bytes: control_bytes,
            static_snapshot_envelope_satisfied,
        });
    }
    let boundary = object(root, "claim_boundary", "memory-envelope contract")?;
    for field in [
        "preflight_opened_no_model_artifact",
        "preflight_scanned_no_artifact_directory",
        "preflight_probed_no_host_memory",
        "preflight_allocated_no_metal_memory",
        "preflight_acquired_no_gpu_lease",
        "preflight_started_no_watcher_or_server",
        "preflight_bound_no_port",
        "preflight_executed_no_token_hcli_tps_or_tg",
    ] {
        bool_eq(
            boundary,
            field,
            true,
            "memory-envelope contract.claim_boundary",
        )?;
    }
    Ok(MemoryAuthority {
        contract,
        resident_weight_bytes,
        shared_runtime_bytes,
        per_session_state_and_kv_bytes,
        per_session_control_bytes,
        profiles: parsed_profiles,
    })
}

fn load_state(path: &Path) -> Result<StateAuthority, String> {
    let (evidence, document) = regular_json(path, "decode-state contract")?;
    let root = root_object(&document, "decode-state contract")?;
    let contract = contract(
        evidence,
        root,
        STATE_SCHEMA,
        STATE_STATUS,
        "decode-state contract",
    )?;
    bool_eq(
        root,
        "complete_decoder_readiness_earned",
        false,
        "decode-state contract",
    )?;
    bool_eq(
        root,
        "real_gravity_server_launch_precondition_satisfied",
        false,
        "decode-state contract",
    )?;
    let source = object(root, "source_archaeology", "decode-state contract")?;
    string_eq(
        source,
        "model_id",
        MODEL_ID,
        "decode-state contract.source_archaeology",
    )?;
    string_eq(
        source,
        "source_repository",
        SOURCE_REPOSITORY,
        "decode-state contract.source_archaeology",
    )?;
    string_eq(
        source,
        "source_revision",
        SOURCE_REVISION,
        "decode-state contract.source_archaeology",
    )?;
    usize_eq(
        source,
        "layer_count",
        48,
        "decode-state contract.source_archaeology",
    )?;
    usize_eq(
        source,
        "deltanet_layers",
        36,
        "decode-state contract.source_archaeology",
    )?;
    usize_eq(
        source,
        "gqa_layers",
        12,
        "decode-state contract.source_archaeology",
    )?;
    usize_eq(
        source,
        "source_tokenizer_vocab",
        151_669,
        "decode-state contract.source_archaeology",
    )?;
    usize_eq(
        source,
        "lm_head_rows",
        151_936,
        "decode-state contract.source_archaeology",
    )?;
    let gqa_shape = array(
        source,
        "gqa_key_value_per_gqa_layer",
        "decode-state contract.source_archaeology",
    )?;
    if gqa_shape.len() != 3 {
        return Err("decode-state contract GQA K/V shape must have rank three".into());
    }
    let max_seq_len = gqa_shape[0]
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or("decode-state contract GQA K/V context must be a usize")?;
    if max_seq_len == 0 || max_seq_len > MAX_SEQUENCE_LENGTH {
        return Err(format!(
            "decode-state contract context {max_seq_len} exceeds the bounded scheduler maximum {MAX_SEQUENCE_LENGTH}"
        ));
    }
    if gqa_shape[1].as_u64() != Some(2) || gqa_shape[2].as_u64() != Some(256) {
        return Err("decode-state contract GQA K/V shape drifted from [context,2,256]".into());
    }
    let geometry = object(root, "state_geometry", "decode-state contract")?;
    usize_eq(
        geometry,
        "per_session_state_records",
        96,
        "decode-state contract.state_geometry",
    )?;
    usize_eq(
        geometry,
        "linear_state_slots",
        36,
        "decode-state contract.state_geometry",
    )?;
    usize_eq(
        geometry,
        "gqa_kv_slots",
        12,
        "decode-state contract.state_geometry",
    )?;
    bool_eq(
        geometry,
        "state_content_materialized",
        false,
        "decode-state contract.state_geometry",
    )?;
    let checks = object(root, "fixture_contract_checks", "decode-state contract")?;
    for field in [
        "exact_schedule_checked",
        "layer_state_owner_checked",
        "state_slot_aliasing_checked",
        "causal_position_and_update_order_checked",
        "restart_identity_checked",
        "rollback_identity_checked",
        "cross_session_leakage_rejected",
    ] {
        bool_eq(
            checks,
            field,
            true,
            "decode-state contract.fixture_contract_checks",
        )?;
    }
    let boundary = object(root, "execution_boundary", "decode-state contract")?;
    for field in [
        "not_runtime",
        "no_live_artifact_scan",
        "no_packed_tensor_read",
        "no_metal_device_or_dispatch",
        "no_model_token_execution",
        "no_logit_or_sampler_execution",
        "no_hcli_execution",
        "no_tps_or_tg_measurement",
        "no_server_started",
    ] {
        bool_eq(
            boundary,
            field,
            true,
            "decode-state contract.execution_boundary",
        )?;
    }
    Ok(StateAuthority {
        contract,
        max_seq_len,
        per_session_state_records: 96,
        linear_state_slots: 36,
        gqa_kv_slots: 12,
    })
}

fn load_source_template(path: &Path) -> Result<SourceTemplateAuthority, String> {
    let (evidence, document) = regular_json(path, "source-template contract")?;
    let root = root_object(&document, "source-template contract")?;
    let contract = contract(
        evidence,
        root,
        TEMPLATE_SCHEMA,
        TEMPLATE_STATUS,
        "source-template contract",
    )?;
    if contract.unsealed_preimage_sha256.is_none() {
        return Err("source-template contract must retain an unsealed preimage SHA-256".into());
    }
    bool_eq(root, "prepared", true, "source-template contract")?;
    bool_eq(root, "executed", false, "source-template contract")?;
    bool_eq(
        root,
        "contracts_are_distinct_by_content_render_and_token_ids",
        true,
        "source-template contract",
    )?;
    let source = object(root, "source_authority", "source-template contract")?;
    string_eq(
        source,
        "model_id",
        MODEL_ID,
        "source-template contract.source_authority",
    )?;
    string_eq(
        source,
        "model_key",
        MODEL_KEY,
        "source-template contract.source_authority",
    )?;
    string_eq(
        source,
        "source_repository",
        SOURCE_REPOSITORY,
        "source-template contract.source_authority",
    )?;
    string_eq(
        source,
        "source_revision",
        SOURCE_REVISION,
        "source-template contract.source_authority",
    )?;
    let tokenizer_vocab_size = usize_value(
        source,
        "tokenizer_addressable_vocab_size",
        "source-template contract.source_authority",
    )?;
    let lm_head_vocab_size = usize_value(
        source,
        "lm_head_vocab_size",
        "source-template contract.source_authority",
    )?;
    if tokenizer_vocab_size != 151_669 || lm_head_vocab_size != 151_936 {
        return Err("source-template contract tokenizer/lm-head geometry drifted".into());
    }
    let prompts = array(root, "prompt_contracts", "source-template contract")?;
    if prompts.len() != 2 {
        return Err("source-template contract must contain exactly A and B prompt bindings".into());
    }
    let mut seen_labels = BTreeSet::new();
    let mut seen_rendered = BTreeSet::new();
    let mut seen_token_ids = BTreeSet::new();
    let mut prompt_bindings = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let prompt = root_object(prompt, "source-template prompt contract")?;
        let label = string(prompt, "label", "source-template prompt contract")?.to_owned();
        let rendered_prompt_sha256 = string(
            prompt,
            "rendered_source_template_prompt_sha256",
            "source-template prompt contract",
        )?
        .to_owned();
        let token_ids_sha256 = string(
            prompt,
            "token_ids_sha256",
            "source-template prompt contract",
        )?
        .to_owned();
        if !is_lower_sha256(&rendered_prompt_sha256) || !is_lower_sha256(&token_ids_sha256) {
            return Err(
                "source-template prompt binding must retain lowercase SHA-256 commitments".into(),
            );
        }
        let token_count = usize_value(prompt, "token_count", "source-template prompt contract")?;
        if token_count == 0 {
            return Err("source-template prompt binding has zero tokens".into());
        }
        if !seen_labels.insert(label.clone())
            || !seen_rendered.insert(rendered_prompt_sha256.clone())
            || !seen_token_ids.insert(token_ids_sha256.clone())
        {
            return Err(
                "source-template prompt bindings must be distinct by label, render, and token IDs"
                    .into(),
            );
        }
        prompt_bindings.push(SourcePromptBinding {
            label,
            rendered_prompt_sha256,
            token_ids_sha256,
            token_count,
        });
    }
    let expected_labels = BTreeSet::from([String::from("A"), String::from("B")]);
    if seen_labels != expected_labels {
        return Err("source-template prompt bindings must be exactly labels A and B".into());
    }
    prompt_bindings.sort_by(|left, right| left.label.cmp(&right.label));
    let boundary = object(root, "execution_boundary", "source-template contract")?;
    for field in [
        "live_packed_artifact_scan_performed",
        "raw_weight_or_gravity_payload_opened",
        "metal_device_or_dispatch_performed",
        "model_or_decoder_execution_performed",
        "server_or_hcli_execution_performed",
        "generation_or_coherence_evaluation_performed",
        "tps_or_tg_measurement_performed",
    ] {
        bool_eq(
            boundary,
            field,
            false,
            "source-template contract.execution_boundary",
        )?;
    }
    Ok(SourceTemplateAuthority {
        contract,
        model_id: MODEL_ID.into(),
        source_repository: SOURCE_REPOSITORY.into(),
        source_revision: SOURCE_REVISION.into(),
        tokenizer_vocab_size,
        lm_head_vocab_size,
        prompt_bindings,
    })
}

fn load_preconditions(args: &Args) -> Result<ExternalPreconditions, String> {
    let activation = load_activation(&args.activation_contract)?;
    let memory = load_memory(&args.memory_envelope_contract)?;
    let state = load_state(&args.state_contract)?;
    let source_template = load_source_template(&args.source_template_contract)?;
    if source_template.model_id != MODEL_ID
        || source_template.source_repository != SOURCE_REPOSITORY
        || source_template.source_revision != SOURCE_REVISION
    {
        return Err(
            "source-template authority does not match the state/memory source identity".into(),
        );
    }
    if state.max_seq_len > MAX_SEQUENCE_LENGTH {
        return Err("state contract exceeds the memory-envelope bounded context".into());
    }
    Ok(ExternalPreconditions {
        activation,
        memory,
        state,
        source_template,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SessionPhase {
    Queued,
    InFlight,
    WaitingOnTool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
struct SessionCheckpoint {
    checkpoint_identity_sha256: String,
    active_state_identity_sha256: String,
    position: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FixtureSession {
    session_id: String,
    state_namespace: String,
    template_label: String,
    template_rendered_prompt_sha256: String,
    template_token_ids_sha256: String,
    template_authority_document_sha256: String,
    per_session_state_and_kv_bytes: usize,
    per_session_control_bytes: usize,
    active_state_identity_sha256: String,
    position: usize,
    checkpoint: Option<SessionCheckpoint>,
    phase: SessionPhase,
}

#[derive(Serialize)]
struct InitialStateIdentity<'a> {
    schema: &'static str,
    session_id: &'a str,
    state_namespace: &'a str,
    template_authority_document_sha256: &'a str,
    template_label: &'a str,
    template_rendered_prompt_sha256: &'a str,
    template_token_ids_sha256: &'a str,
    per_session_state_and_kv_bytes: usize,
    per_session_control_bytes: usize,
    position: usize,
}

#[derive(Serialize)]
struct SchedulerWorkIdentity<'a> {
    schema: &'static str,
    prior_active_state_identity_sha256: &'a str,
    session_id: &'a str,
    state_namespace: &'a str,
    position: usize,
    checkpoint_identity_sha256: &'a str,
}

#[derive(Serialize)]
struct CheckpointIdentity<'a> {
    schema: &'static str,
    session_id: &'a str,
    state_namespace: &'a str,
    active_state_identity_sha256: &'a str,
    position: usize,
}

struct FixtureScheduler {
    maximum_logical_sessions: usize,
    sessions: BTreeMap<String, FixtureSession>,
    ready_queue: VecDeque<String>,
    active_session: Option<String>,
}

impl FixtureScheduler {
    fn new(
        session_count: usize,
        handoff: &SourceTemplateHandoff,
        per_session_state_and_kv_bytes: usize,
        per_session_control_bytes: usize,
    ) -> Result<Self, String> {
        expected_session_count(session_count)?;
        if handoff.prompt_bindings.len() != 2
            || handoff.source_template_authority_document_sha256.is_empty()
            || !is_lower_sha256(&handoff.source_template_authority_document_sha256)
        {
            return Err(
                "fixture scheduler requires exactly two SHA-bound source-template prompts".into(),
            );
        }
        if per_session_state_and_kv_bytes == 0 || per_session_control_bytes == 0 {
            return Err(
                "fixture scheduler requires finite non-zero per-session state/control bounds"
                    .into(),
            );
        }
        let mut sessions = BTreeMap::new();
        let mut ready_queue = VecDeque::new();
        for index in 0..session_count {
            let session_id = format!("logical-session-{index:02}");
            let state_namespace = format!("qwen80/{session_id}/state-kv");
            let prompt = &handoff.prompt_bindings[index % handoff.prompt_bindings.len()];
            let active_state_identity_sha256 = sha256_json(&InitialStateIdentity {
                schema: "hawking.ascension.qwen80_one_resident_scheduler_initial_state.v1",
                session_id: &session_id,
                state_namespace: &state_namespace,
                template_authority_document_sha256: &handoff
                    .source_template_authority_document_sha256,
                template_label: &prompt.label,
                template_rendered_prompt_sha256: &prompt.rendered_prompt_sha256,
                template_token_ids_sha256: &prompt.token_ids_sha256,
                per_session_state_and_kv_bytes,
                per_session_control_bytes,
                position: 0,
            });
            let fixture = FixtureSession {
                session_id: session_id.clone(),
                state_namespace,
                template_label: prompt.label.clone(),
                template_rendered_prompt_sha256: prompt.rendered_prompt_sha256.clone(),
                template_token_ids_sha256: prompt.token_ids_sha256.clone(),
                template_authority_document_sha256: handoff
                    .source_template_authority_document_sha256
                    .clone(),
                per_session_state_and_kv_bytes,
                per_session_control_bytes,
                active_state_identity_sha256,
                position: 0,
                checkpoint: None,
                phase: SessionPhase::Queued,
            };
            if sessions.insert(session_id.clone(), fixture).is_some() {
                return Err("fixture scheduler produced a duplicate session identity".into());
            }
            ready_queue.push_back(session_id);
        }
        let scheduler = Self {
            maximum_logical_sessions: session_count,
            sessions,
            ready_queue,
            active_session: None,
        };
        scheduler.validate()?;
        Ok(scheduler)
    }

    fn validate(&self) -> Result<(), String> {
        if self.sessions.len() != self.maximum_logical_sessions {
            return Err("fixture scheduler session count drifted".into());
        }
        expected_session_count(self.maximum_logical_sessions)?;
        let namespaces = self
            .sessions
            .values()
            .map(|session| session.state_namespace.as_str())
            .collect::<BTreeSet<_>>();
        if namespaces.len() != self.sessions.len() {
            return Err("fixture scheduler detected cross-session state namespace aliasing".into());
        }
        let mut queued = BTreeSet::new();
        for session_id in &self.ready_queue {
            if !queued.insert(session_id.as_str()) {
                return Err("fixture scheduler ready queue has duplicate session IDs".into());
            }
            let session = self
                .sessions
                .get(session_id)
                .ok_or("fixture scheduler queue references an unknown session")?;
            if session.phase != SessionPhase::Queued {
                return Err("only queued sessions may occupy the ready queue".into());
            }
        }
        match &self.active_session {
            Some(session_id) => {
                if queued.contains(session_id.as_str()) {
                    return Err("active session cannot also occupy the ready queue".into());
                }
                let session = self
                    .sessions
                    .get(session_id)
                    .ok_or("fixture scheduler active session is unknown")?;
                if session.phase != SessionPhase::InFlight {
                    return Err("active session must be marked in-flight".into());
                }
            }
            None => {}
        }
        for session in self.sessions.values() {
            let in_queue = queued.contains(session.session_id.as_str());
            let is_active = self.active_session.as_deref() == Some(session.session_id.as_str());
            match session.phase {
                SessionPhase::Queued if !in_queue => {
                    return Err("queued session is missing from the ready queue".into())
                }
                SessionPhase::InFlight if !is_active => {
                    return Err("in-flight session is missing the decode slot".into())
                }
                SessionPhase::WaitingOnTool if in_queue || is_active => {
                    return Err(
                        "tool-waiting session must release the decode slot and ready queue".into(),
                    )
                }
                _ => {}
            }
            if session.per_session_state_and_kv_bytes == 0
                || session.per_session_control_bytes == 0
                || !is_lower_sha256(&session.active_state_identity_sha256)
                || !is_lower_sha256(&session.template_authority_document_sha256)
                || !is_lower_sha256(&session.template_rendered_prompt_sha256)
                || !is_lower_sha256(&session.template_token_ids_sha256)
            {
                return Err(
                    "fixture scheduler session has invalid bounded state/template commitments"
                        .into(),
                );
            }
        }
        Ok(())
    }

    fn dispatch_next(&mut self) -> Result<String, String> {
        if self.active_session.is_some() {
            return Err(
                "fixture scheduler cannot dispatch while the one decode slot is occupied".into(),
            );
        }
        let session_id = self
            .ready_queue
            .pop_front()
            .ok_or("fixture scheduler has no runnable logical session")?;
        let session = self
            .sessions
            .get_mut(&session_id)
            .ok_or("fixture scheduler queue references unknown session")?;
        if session.phase != SessionPhase::Queued {
            return Err("fixture scheduler attempted to dispatch a non-queued session".into());
        }
        session.phase = SessionPhase::InFlight;
        self.active_session = Some(session_id.clone());
        self.validate()?;
        Ok(session_id)
    }

    fn require_active_mut(&mut self, session_id: &str) -> Result<&mut FixtureSession, String> {
        if self.active_session.as_deref() != Some(session_id) {
            return Err("operation requires the currently held one decode slot".into());
        }
        self.sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("unknown logical session {session_id:?}"))
    }

    fn checkpoint(&mut self, session_id: &str) -> Result<String, String> {
        let session = self.require_active_mut(session_id)?;
        let checkpoint_identity_sha256 = sha256_json(&CheckpointIdentity {
            schema: "hawking.ascension.qwen80_one_resident_scheduler_checkpoint.v1",
            session_id: &session.session_id,
            state_namespace: &session.state_namespace,
            active_state_identity_sha256: &session.active_state_identity_sha256,
            position: session.position,
        });
        session.checkpoint = Some(SessionCheckpoint {
            checkpoint_identity_sha256: checkpoint_identity_sha256.clone(),
            active_state_identity_sha256: session.active_state_identity_sha256.clone(),
            position: session.position,
        });
        Ok(checkpoint_identity_sha256)
    }

    fn record_fixture_work_unit(&mut self, session_id: &str) -> Result<(), String> {
        let session = self.require_active_mut(session_id)?;
        let checkpoint = session
            .checkpoint
            .as_ref()
            .ok_or("fixture work unit requires an exact checkpoint first")?;
        session.active_state_identity_sha256 = sha256_json(&SchedulerWorkIdentity {
            schema: "hawking.ascension.qwen80_one_resident_scheduler_fixture_work_unit.v1",
            prior_active_state_identity_sha256: &session.active_state_identity_sha256,
            session_id: &session.session_id,
            state_namespace: &session.state_namespace,
            position: session.position,
            checkpoint_identity_sha256: &checkpoint.checkpoint_identity_sha256,
        });
        session.position = checked_add(session.position, 1, "fixture scheduler position")?;
        Ok(())
    }

    fn rollback_checkpoint_identity(&mut self, session_id: &str) -> Result<(), String> {
        let session = self.require_active_mut(session_id)?;
        let checkpoint = session
            .checkpoint
            .clone()
            .ok_or("fixture rollback requires a checkpoint")?;
        session.active_state_identity_sha256 = checkpoint.active_state_identity_sha256;
        session.position = checkpoint.position;
        let recomputed = sha256_json(&CheckpointIdentity {
            schema: "hawking.ascension.qwen80_one_resident_scheduler_checkpoint.v1",
            session_id: &session.session_id,
            state_namespace: &session.state_namespace,
            active_state_identity_sha256: &session.active_state_identity_sha256,
            position: session.position,
        });
        if recomputed != checkpoint.checkpoint_identity_sha256 {
            return Err("fixture rollback failed to restore checkpoint identity exactly".into());
        }
        Ok(())
    }

    fn yield_for_tool(&mut self, session_id: &str) -> Result<(), String> {
        let session = self.require_active_mut(session_id)?;
        session.phase = SessionPhase::WaitingOnTool;
        self.active_session = None;
        self.validate()
    }

    fn resume_after_tool(&mut self, session_id: &str) -> Result<(), String> {
        if self.active_session.is_some() && self.active_session.as_deref() == Some(session_id) {
            return Err("tool-waiting session unexpectedly still holds the decode slot".into());
        }
        let session = self
            .sessions
            .get_mut(session_id)
            .ok_or_else(|| format!("unknown logical session {session_id:?}"))?;
        if session.phase != SessionPhase::WaitingOnTool {
            return Err("only a tool-waiting session may resume".into());
        }
        session.phase = SessionPhase::Queued;
        self.ready_queue.push_back(session_id.to_owned());
        self.validate()
    }

    fn release_to_queue_tail(&mut self, session_id: &str) -> Result<(), String> {
        let session = self.require_active_mut(session_id)?;
        session.phase = SessionPhase::Queued;
        self.active_session = None;
        self.ready_queue.push_back(session_id.to_owned());
        self.validate()
    }

    fn session(&self, session_id: &str) -> Result<&FixtureSession, String> {
        self.sessions
            .get(session_id)
            .ok_or_else(|| format!("unknown logical session {session_id:?}"))
    }

    fn state_namespace_count(&self) -> usize {
        self.sessions
            .values()
            .map(|session| session.state_namespace.as_str())
            .collect::<BTreeSet<_>>()
            .len()
    }
}

fn source_template_handoff(
    source_template: &SourceTemplateAuthority,
) -> Result<SourceTemplateHandoff, String> {
    let preimage = source_template
        .contract
        .unsealed_preimage_sha256
        .as_ref()
        .ok_or("source-template authority lacks a preimage SHA-256")?;
    Ok(SourceTemplateHandoff {
        source_template_authority_document_sha256: source_template.contract.contract.document_sha256.clone(),
        source_template_authority_preimage_sha256: preimage.clone(),
        prompt_bindings: source_template.prompt_bindings.clone(),
        handoff_rule:
            "Every logical session must retain this exact source-template authority document SHA-256 and select one admitted prompt binding; no ad-hoc prompt/template authority is permitted.",
    })
}

fn logical_session_profiles(
    preconditions: &ExternalPreconditions,
    handoff: &SourceTemplateHandoff,
) -> Result<Vec<LogicalSessionProfile>, String> {
    let memory = &preconditions.memory;
    let per_session_total_bytes = checked_add(
        memory.per_session_state_and_kv_bytes,
        memory.per_session_control_bytes,
        "per-session bounded state/control bytes",
    )?;
    LOGICAL_SESSION_COUNTS
        .iter()
        .map(|&logical_sessions| {
            let authority = memory
                .profiles
                .iter()
                .find(|profile| profile.logical_sessions == logical_sessions)
                .ok_or("memory-envelope authority lacks a required session profile")?;
            let aggregate_session_bytes = checked_mul(
                logical_sessions,
                per_session_total_bytes,
                "aggregate logical-session bytes",
            )?;
            let expected_planned_resident_bytes = checked_add(
                checked_add(
                    memory.resident_weight_bytes,
                    memory.shared_runtime_bytes,
                    "one-resident-model planned bytes",
                )?,
                aggregate_session_bytes,
                "one-resident-model planned bytes",
            )?;
            if authority.q80_planned_resident_bytes != expected_planned_resident_bytes {
                return Err(
                    "memory-envelope authority is not one shared model plus bounded sessions"
                        .into(),
                );
            }
            let scheduler = FixtureScheduler::new(
                logical_sessions,
                handoff,
                memory.per_session_state_and_kv_bytes,
                memory.per_session_control_bytes,
            )?;
            Ok(LogicalSessionProfile {
                logical_sessions,
                resident_q80_model_processes: 1,
                immutable_weight_copies: 1,
                resident_weight_bytes: memory.resident_weight_bytes,
                shared_runtime_bytes: memory.shared_runtime_bytes,
                per_session_state_and_kv_bytes: memory.per_session_state_and_kv_bytes,
                per_session_control_bytes: memory.per_session_control_bytes,
                per_session_total_bytes,
                aggregate_session_bytes,
                planned_resident_bytes: authority.q80_planned_resident_bytes,
                distinct_state_namespace_count: scheduler.state_namespace_count(),
                source_template_authority_document_sha256: handoff
                    .source_template_authority_document_sha256
                    .clone(),
            })
        })
        .collect()
}

fn fixture_checks(
    preconditions: &ExternalPreconditions,
    handoff: &SourceTemplateHandoff,
    profiles: &[LogicalSessionProfile],
) -> Result<FixtureChecks, String> {
    let exact_one_two_four_eight_sixteen_profiles_checked = profiles
        .iter()
        .map(|profile| profile.logical_sessions)
        .eq(LOGICAL_SESSION_COUNTS);
    let one_immutable_weight_copy_per_profile_checked = profiles.iter().all(|profile| {
        profile.resident_q80_model_processes == 1
            && profile.immutable_weight_copies == 1
            && profile.resident_weight_bytes == preconditions.memory.resident_weight_bytes
            && profile.planned_resident_bytes
                == profile.resident_weight_bytes
                    + profile.shared_runtime_bytes
                    + profile.aggregate_session_bytes
    });
    let per_session_state_and_kv_bound_checked = profiles.iter().all(|profile| {
        profile.per_session_state_and_kv_bytes
            == preconditions.memory.per_session_state_and_kv_bytes
            && profile.per_session_control_bytes == preconditions.memory.per_session_control_bytes
            && profile.aggregate_session_bytes
                == profile.logical_sessions * profile.per_session_total_bytes
            && profile.distinct_state_namespace_count == profile.logical_sessions
    });

    let state_namespace_aliasing_rejected = {
        let mut scheduler = FixtureScheduler::new(
            2,
            handoff,
            preconditions.memory.per_session_state_and_kv_bytes,
            preconditions.memory.per_session_control_bytes,
        )?;
        let first = scheduler
            .sessions
            .get("logical-session-00")
            .ok_or("fixture scheduler lacks session zero")?
            .state_namespace
            .clone();
        scheduler
            .sessions
            .get_mut("logical-session-01")
            .ok_or("fixture scheduler lacks session one")?
            .state_namespace = first;
        scheduler.validate().is_err()
    };

    let mut scheduler = FixtureScheduler::new(
        4,
        handoff,
        preconditions.memory.per_session_state_and_kv_bytes,
        preconditions.memory.per_session_control_bytes,
    )?;
    let first = scheduler.dispatch_next()?;
    if first != "logical-session-00" {
        return Err("fixture FIFO queue did not begin with logical-session-00".into());
    }
    let pre_checkpoint_state = scheduler
        .session(&first)?
        .active_state_identity_sha256
        .clone();
    let checkpoint_identity = scheduler.checkpoint(&first)?;
    scheduler.record_fixture_work_unit(&first)?;
    let mutated_state = scheduler
        .session(&first)?
        .active_state_identity_sha256
        .clone();
    if mutated_state == pre_checkpoint_state {
        return Err("fixture work unit did not produce a distinct state commitment".into());
    }
    scheduler.rollback_checkpoint_identity(&first)?;
    let checkpoint_rollback_identity_checked = scheduler
        .session(&first)?
        .checkpoint
        .as_ref()
        .is_some_and(|checkpoint| {
            checkpoint.checkpoint_identity_sha256 == checkpoint_identity
                && scheduler
                    .session(&first)
                    .map(|session| {
                        session.active_state_identity_sha256
                            == checkpoint.active_state_identity_sha256
                            && session.position == checkpoint.position
                    })
                    .unwrap_or(false)
        });
    scheduler.yield_for_tool(&first)?;
    let tool_wait_releases_decode_slot_checked = scheduler.active_session.is_none()
        && scheduler.session(&first)?.phase == SessionPhase::WaitingOnTool;
    scheduler.resume_after_tool(&first)?;
    let mut observed_order = Vec::new();
    for _ in 0..4 {
        let session_id = scheduler.dispatch_next()?;
        observed_order.push(session_id.clone());
        scheduler.release_to_queue_tail(&session_id)?;
    }
    let fifo_fairness_after_tool_wait_checked = observed_order
        == [
            "logical-session-01".to_owned(),
            "logical-session-02".to_owned(),
            "logical-session-03".to_owned(),
            "logical-session-00".to_owned(),
        ];
    let source_template_authority_handoff_checked = scheduler.sessions.values().all(|session| {
        session.template_authority_document_sha256
            == handoff.source_template_authority_document_sha256
            && handoff.prompt_bindings.iter().any(|prompt| {
                prompt.label == session.template_label
                    && prompt.rendered_prompt_sha256 == session.template_rendered_prompt_sha256
                    && prompt.token_ids_sha256 == session.template_token_ids_sha256
            })
    });
    let unsupported_session_count_rejected = FixtureScheduler::new(
        3,
        handoff,
        preconditions.memory.per_session_state_and_kv_bytes,
        preconditions.memory.per_session_control_bytes,
    )
    .is_err();

    Ok(FixtureChecks {
        exact_one_two_four_eight_sixteen_profiles_checked,
        one_immutable_weight_copy_per_profile_checked,
        per_session_state_and_kv_bound_checked,
        state_namespace_aliasing_rejected,
        fifo_fairness_after_tool_wait_checked,
        tool_wait_releases_decode_slot_checked,
        checkpoint_rollback_identity_checked,
        source_template_authority_handoff_checked,
        unsupported_session_count_rejected,
    })
}

fn all_fixture_checks_pass(checks: &FixtureChecks) -> bool {
    checks.exact_one_two_four_eight_sixteen_profiles_checked
        && checks.one_immutable_weight_copy_per_profile_checked
        && checks.per_session_state_and_kv_bound_checked
        && checks.state_namespace_aliasing_rejected
        && checks.fifo_fairness_after_tool_wait_checked
        && checks.tool_wait_releases_decode_slot_checked
        && checks.checkpoint_rollback_identity_checked
        && checks.source_template_authority_handoff_checked
        && checks.unsupported_session_count_rejected
}

fn report(preconditions: ExternalPreconditions) -> Result<Report, String> {
    let handoff = source_template_handoff(&preconditions.source_template)?;
    let profiles = logical_session_profiles(&preconditions, &handoff)?;
    let checks = fixture_checks(&preconditions, &handoff, &profiles)?;
    if !all_fixture_checks_pass(&checks) {
        return Err(
            "one-resident logical-session scheduler fixture checks did not all pass".into(),
        );
    }
    let mut report = Report {
        schema: SCHEMA,
        status: STATUS,
        prepared: true,
        complete_decoder_readiness_earned: false,
        real_gravity_server_launch_precondition_satisfied: false,
        external_preconditions: preconditions,
        source_template_handoff: handoff,
        logical_session_profiles: profiles,
        scheduler_semantics: SchedulerSemantics {
            topology:
                "one resident Q80 model instance; 1/2/4/8/16 logical sessions share immutable weights",
            decode_slot_count: 1,
            queue_discipline:
                "FIFO round-robin: a completed scheduler work unit and a resumed tool-waiting session enter the queue tail",
            tool_wait_behavior:
                "tool wait transitions the active logical session to waiting and releases the sole scheduler decode slot before the tool returns",
            state_ownership:
                "every logical session owns a distinct bounded state/KV namespace sized by the external memory-envelope contract; no state/KV namespace is shared",
            checkpoint_and_rollback:
                "checkpoint stores the session active-state identity before a work unit; rollback must restore both state identity and position exactly",
            source_template_handoff:
                "each logical session retains the same source-template authority document SHA-256 and one exact admitted A/B prompt binding",
        },
        focused_fixture_checks: checks,
        execution_boundary: ExecutionBoundary {
            model_weights_loaded: false,
            artifact_or_payload_scanned: false,
            device_or_metal_used: false,
            gpu_lease_or_registry_mutated: false,
            runtime_watcher_or_server_started: false,
            port_bound_or_listener_created: false,
            decoder_or_model_token_executed: false,
            hcli_executed: false,
            tps_or_tg_measured: false,
        },
        claim_boundary: vec![
            "This scheduler contract reads only four explicitly supplied preflight JSON documents. It does not open an artifact, packed payload, or model weights.",
            "The logical-session state machine uses SHA-256 commitments to future caller-owned state only. It does not allocate state/KV memory, execute a decoder layer, or generate a token.",
            "One resident model and many logical sessions is a required future topology, not evidence that a Q80 server, HCLI path, or multi-session runtime currently exists.",
            "This result cannot qualify Q80 for complete decoder readiness, server launch, HCLI, BASE_TRUE_TPS, TG, capability, Agent OS, or tournament entry.",
        ],
        unsealed_preimage_sha256: String::new(),
    };
    report.unsealed_preimage_sha256 = sha256_json(&report);
    Ok(report)
}

fn write_report_create_new(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    if !path.is_absolute() {
        return Err("--out must be absolute".into());
    }
    let parent = path.parent().ok_or("--out has no parent directory")?;
    if !parent.is_dir() {
        return Err(format!("--out parent directory is missing: {}", parent.display()).into());
    }
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&serde_json::to_vec_pretty(report)?)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_one_resident_session_scheduler_contract \
--activation-contract ABSOLUTE_PATH \
--memory-envelope-contract ABSOLUTE_PATH \
--state-contract ABSOLUTE_PATH \
--source-template-contract ABSOLUTE_PATH \
--out ABSOLUTE_NEW_PATH"
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut activation_contract = None;
    let mut memory_envelope_contract = None;
    let mut state_contract = None;
    let mut source_template_contract = None;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        let path = PathBuf::from(value);
        match flag.as_str() {
            "--activation-contract" => {
                if activation_contract.replace(path).is_some() {
                    return Err("--activation-contract repeated".into());
                }
            }
            "--memory-envelope-contract" => {
                if memory_envelope_contract.replace(path).is_some() {
                    return Err("--memory-envelope-contract repeated".into());
                }
            }
            "--state-contract" => {
                if state_contract.replace(path).is_some() {
                    return Err("--state-contract repeated".into());
                }
            }
            "--source-template-contract" => {
                if source_template_contract.replace(path).is_some() {
                    return Err("--source-template-contract repeated".into());
                }
            }
            "--out" => {
                if out.replace(path).is_some() {
                    return Err("--out repeated".into());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage()).into()),
        }
    }
    let args = Args {
        activation_contract: activation_contract
            .ok_or_else(|| format!("missing --activation-contract; {}", usage()))?,
        memory_envelope_contract: memory_envelope_contract
            .ok_or_else(|| format!("missing --memory-envelope-contract; {}", usage()))?,
        state_contract: state_contract
            .ok_or_else(|| format!("missing --state-contract; {}", usage()))?,
        source_template_contract: source_template_contract
            .ok_or_else(|| format!("missing --source-template-contract; {}", usage()))?,
        out: out.ok_or_else(|| format!("missing --out; {}", usage()))?,
    };
    for (label, path) in [
        ("--activation-contract", &args.activation_contract),
        ("--memory-envelope-contract", &args.memory_envelope_contract),
        ("--state-contract", &args.state_contract),
        ("--source-template-contract", &args.source_template_contract),
        ("--out", &args.out),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute").into());
        }
    }
    Ok(args)
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let preconditions = load_preconditions(&args)?;
    let report = report(preconditions)?;
    write_report_create_new(&args.out, &report)
}

fn main() {
    if let Err(error) = parse_args().and_then(run) {
        eprintln!("ascension_qwen80_one_resident_session_scheduler_contract: {error}");
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

    fn contract_fixture(schema: &str, status: &str, preimage: Option<char>) -> ExternalContract {
        ExternalContract {
            contract: FileEvidence {
                path: format!("/fixture/{schema}.json"),
                bytes: 1,
                document_sha256: sha('a'),
            },
            schema: schema.into(),
            status: status.into(),
            unsealed_preimage_sha256: preimage.map(sha),
        }
    }

    fn fixture_preconditions() -> ExternalPreconditions {
        let resident_weight_bytes = 10_000;
        let shared_runtime_bytes = 1_000;
        let per_session_state_and_kv_bytes = 200;
        let per_session_control_bytes = 20;
        let profiles = LOGICAL_SESSION_COUNTS
            .into_iter()
            .map(|logical_sessions| {
                let session_bytes =
                    logical_sessions * (per_session_state_and_kv_bytes + per_session_control_bytes);
                MemoryProfileAuthority {
                    logical_sessions,
                    q80_planned_resident_bytes: resident_weight_bytes
                        + shared_runtime_bytes
                        + session_bytes,
                    state_and_kv_bytes: logical_sessions * per_session_state_and_kv_bytes,
                    session_control_bytes: logical_sessions * per_session_control_bytes,
                    static_snapshot_envelope_satisfied: false,
                }
            })
            .collect();
        ExternalPreconditions {
            activation: contract_fixture(ACTIVATION_SCHEMA, ACTIVATION_REFUSED_STATUS, None),
            memory: MemoryAuthority {
                contract: contract_fixture(MEMORY_SCHEMA, MEMORY_STATUS, None),
                resident_weight_bytes,
                shared_runtime_bytes,
                per_session_state_and_kv_bytes,
                per_session_control_bytes,
                profiles,
            },
            state: StateAuthority {
                contract: contract_fixture(STATE_SCHEMA, STATE_STATUS, Some('b')),
                max_seq_len: 16,
                per_session_state_records: 96,
                linear_state_slots: 36,
                gqa_kv_slots: 12,
            },
            source_template: SourceTemplateAuthority {
                contract: contract_fixture(TEMPLATE_SCHEMA, TEMPLATE_STATUS, Some('c')),
                model_id: MODEL_ID.into(),
                source_repository: SOURCE_REPOSITORY.into(),
                source_revision: SOURCE_REVISION.into(),
                tokenizer_vocab_size: 151_669,
                lm_head_vocab_size: 151_936,
                prompt_bindings: vec![
                    SourcePromptBinding {
                        label: "A".into(),
                        rendered_prompt_sha256: sha('d'),
                        token_ids_sha256: sha('e'),
                        token_count: 15,
                    },
                    SourcePromptBinding {
                        label: "B".into(),
                        rendered_prompt_sha256: sha('f'),
                        token_ids_sha256: sha('1'),
                        token_count: 25,
                    },
                ],
            },
        }
    }

    fn fixture_handoff() -> SourceTemplateHandoff {
        source_template_handoff(&fixture_preconditions().source_template).unwrap()
    }

    #[test]
    fn exact_profiles_keep_one_weight_copy_and_bound_each_session() {
        let preconditions = fixture_preconditions();
        let handoff = source_template_handoff(&preconditions.source_template).unwrap();
        let profiles = logical_session_profiles(&preconditions, &handoff).unwrap();
        assert_eq!(
            profiles
                .iter()
                .map(|profile| profile.logical_sessions)
                .collect::<Vec<_>>(),
            LOGICAL_SESSION_COUNTS
        );
        for profile in profiles {
            assert_eq!(profile.resident_q80_model_processes, 1);
            assert_eq!(profile.immutable_weight_copies, 1);
            assert_eq!(
                profile.distinct_state_namespace_count,
                profile.logical_sessions
            );
            assert_eq!(
                profile.aggregate_session_bytes,
                profile.logical_sessions * profile.per_session_total_bytes
            );
            assert_eq!(
                profile.planned_resident_bytes,
                profile.resident_weight_bytes
                    + profile.shared_runtime_bytes
                    + profile.aggregate_session_bytes
            );
        }
    }

    #[test]
    fn tool_wait_releases_slot_and_resumes_at_fifo_tail() {
        let preconditions = fixture_preconditions();
        let handoff = fixture_handoff();
        let mut scheduler = FixtureScheduler::new(
            4,
            &handoff,
            preconditions.memory.per_session_state_and_kv_bytes,
            preconditions.memory.per_session_control_bytes,
        )
        .unwrap();
        let first = scheduler.dispatch_next().unwrap();
        assert_eq!(first, "logical-session-00");
        scheduler.yield_for_tool(&first).unwrap();
        assert!(scheduler.active_session.is_none());
        assert_eq!(
            scheduler.session(&first).unwrap().phase,
            SessionPhase::WaitingOnTool
        );
        scheduler.resume_after_tool(&first).unwrap();
        let mut observed = Vec::new();
        for _ in 0..4 {
            let next = scheduler.dispatch_next().unwrap();
            observed.push(next.clone());
            scheduler.release_to_queue_tail(&next).unwrap();
        }
        assert_eq!(
            observed,
            [
                "logical-session-01",
                "logical-session-02",
                "logical-session-03",
                "logical-session-00"
            ]
        );
    }

    #[test]
    fn rollback_restores_checkpoint_identity_and_state_namespaces_are_disjoint() {
        let preconditions = fixture_preconditions();
        let handoff = fixture_handoff();
        let mut scheduler = FixtureScheduler::new(
            2,
            &handoff,
            preconditions.memory.per_session_state_and_kv_bytes,
            preconditions.memory.per_session_control_bytes,
        )
        .unwrap();
        let first = scheduler.dispatch_next().unwrap();
        let before = scheduler
            .session(&first)
            .unwrap()
            .active_state_identity_sha256
            .clone();
        let checkpoint = scheduler.checkpoint(&first).unwrap();
        scheduler.record_fixture_work_unit(&first).unwrap();
        assert_ne!(
            scheduler
                .session(&first)
                .unwrap()
                .active_state_identity_sha256,
            before
        );
        scheduler.rollback_checkpoint_identity(&first).unwrap();
        assert_eq!(
            scheduler
                .session(&first)
                .unwrap()
                .active_state_identity_sha256,
            before
        );
        assert_eq!(
            scheduler
                .session(&first)
                .unwrap()
                .checkpoint
                .as_ref()
                .unwrap()
                .checkpoint_identity_sha256,
            checkpoint
        );

        let namespace = scheduler
            .session("logical-session-00")
            .unwrap()
            .state_namespace
            .clone();
        scheduler
            .sessions
            .get_mut("logical-session-01")
            .unwrap()
            .state_namespace = namespace;
        assert!(scheduler.validate().is_err());
    }

    #[test]
    fn report_is_explicitly_prepared_incomplete_and_not_a_server() {
        let report = report(fixture_preconditions()).unwrap();
        assert_eq!(report.schema, SCHEMA);
        assert_eq!(report.status, STATUS);
        assert!(report.prepared);
        assert!(!report.complete_decoder_readiness_earned);
        assert!(!report.real_gravity_server_launch_precondition_satisfied);
        assert!(!report.execution_boundary.runtime_watcher_or_server_started);
        assert!(!report.execution_boundary.decoder_or_model_token_executed);
        assert!(!report.execution_boundary.hcli_executed);
        assert!(!report.execution_boundary.tps_or_tg_measured);
        assert!(all_fixture_checks_pass(&report.focused_fixture_checks));
        assert!(is_lower_sha256(&report.unsealed_preimage_sha256));
    }

    fn write_json(path: &Path, value: Value) {
        fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
    }

    fn memory_profile(logical_sessions: usize) -> Value {
        let state = logical_sessions * 200usize;
        let control = logical_sessions * 20usize;
        json!({
            "logical_sessions": logical_sessions,
            "resident_q80_model_processes": 1,
            "deltanet_state_bytes": logical_sessions * 100usize,
            "gqa_key_cache_bytes": logical_sessions * 50usize,
            "gqa_value_cache_bytes": logical_sessions * 50usize,
            "session_control_buffers_bytes": control,
            "q80_planned_resident_bytes": 10_000usize + 1_000usize + state + control,
            "static_snapshot_envelope_satisfied": false
        })
    }

    fn write_valid_external_documents(directory: &Path) -> Args {
        let activation = directory.join("activation.json");
        let memory = directory.join("memory.json");
        let state = directory.join("state.json");
        let template = directory.join("template.json");
        write_json(
            &activation,
            json!({
                "schema": ACTIVATION_SCHEMA,
                "status": ACTIVATION_REFUSED_STATUS,
                "target_topology": {
                    "resident_q80_model_processes": 1,
                    "logical_sessions": "many",
                    "endpoint": {"host": QWEN80_HOST, "port": QWEN80_PORT},
                    "qwen30_port_reuse_refused": QWEN30_PORT
                },
                "automatic_launch_contract": {
                    "processes_to_start": 1,
                    "duplicate_model_process_start_prohibited": true,
                    "gate_starts_no_process": true
                },
                "claim_boundary": {
                    "gate_started_no_server": true,
                    "gate_bound_no_port": true,
                    "gate_opened_no_model_artifact": true,
                    "gate_executed_no_model_token": true,
                    "gate_executed_no_hcli_request": true,
                    "gate_measured_no_tps_or_tg": true
                }
            }),
        );
        write_json(
            &memory,
            json!({
                "schema": MEMORY_SCHEMA,
                "status": MEMORY_STATUS,
                "prepared": true,
                "complete_decoder_readiness_earned": false,
                "real_gravity_server_launch_precondition_satisfied": false,
                "memory_envelope_healthy": false,
                "actual_resident_q80_rss_measured": false,
                "actual_device_allocation_performed": false,
                "actual_host_memory_probe_performed_by_this_preflight": false,
                "actual_runtime_or_server_launch_performed": false,
                "actual_hcli_or_tps_or_tg_measurement_performed": false,
                "one_q80_process_envelope": true,
                "fixed_one_process_topology": {
                    "resident_q80_model_processes": 1,
                    "logical_sessions_supported": LOGICAL_SESSION_COUNTS,
                    "maximum_logical_sessions": 16,
                    "bounded_max_seq_len": MAX_SEQUENCE_LENGTH,
                    "endpoint": {"host": QWEN80_HOST, "port": QWEN80_PORT}
                },
                "source_artifact_binding": {
                    "model_id": MODEL_ID,
                    "model_key": MODEL_KEY,
                    "source_repository": SOURCE_REPOSITORY,
                    "source_revision": SOURCE_REVISION,
                    "resident_weight_bytes": 10_000usize
                },
                "planned_resident_allocations": {
                    "resident_weights_bytes": 10_000usize,
                    "shared_runtime_buffers_bytes": 1_000usize,
                    "per_logical_session_buffers": {
                        "state_and_kv_bytes": 200usize,
                        "session_control_bytes": 20usize
                    }
                },
                "logical_session_memory_profiles": LOGICAL_SESSION_COUNTS.map(memory_profile),
                "claim_boundary": {
                    "preflight_opened_no_model_artifact": true,
                    "preflight_scanned_no_artifact_directory": true,
                    "preflight_probed_no_host_memory": true,
                    "preflight_allocated_no_metal_memory": true,
                    "preflight_acquired_no_gpu_lease": true,
                    "preflight_started_no_watcher_or_server": true,
                    "preflight_bound_no_port": true,
                    "preflight_executed_no_token_hcli_tps_or_tg": true
                }
            }),
        );
        write_json(
            &state,
            json!({
                "schema": STATE_SCHEMA,
                "status": STATE_STATUS,
                "complete_decoder_readiness_earned": false,
                "real_gravity_server_launch_precondition_satisfied": false,
                "unsealed_preimage_sha256": sha('9'),
                "source_archaeology": {
                    "model_id": MODEL_ID,
                    "source_repository": SOURCE_REPOSITORY,
                    "source_revision": SOURCE_REVISION,
                    "layer_count": 48,
                    "deltanet_layers": 36,
                    "gqa_layers": 12,
                    "source_tokenizer_vocab": 151_669,
                    "lm_head_rows": 151_936,
                    "gqa_key_value_per_gqa_layer": [16, 2, 256]
                },
                "state_geometry": {
                    "per_session_state_records": 96,
                    "linear_state_slots": 36,
                    "gqa_kv_slots": 12,
                    "state_content_materialized": false
                },
                "fixture_contract_checks": {
                    "exact_schedule_checked": true,
                    "layer_state_owner_checked": true,
                    "state_slot_aliasing_checked": true,
                    "causal_position_and_update_order_checked": true,
                    "restart_identity_checked": true,
                    "rollback_identity_checked": true,
                    "cross_session_leakage_rejected": true
                },
                "execution_boundary": {
                    "not_runtime": true,
                    "no_live_artifact_scan": true,
                    "no_packed_tensor_read": true,
                    "no_metal_device_or_dispatch": true,
                    "no_model_token_execution": true,
                    "no_logit_or_sampler_execution": true,
                    "no_hcli_execution": true,
                    "no_tps_or_tg_measurement": true,
                    "no_server_started": true
                }
            }),
        );
        write_json(
            &template,
            json!({
                "schema": TEMPLATE_SCHEMA,
                "status": TEMPLATE_STATUS,
                "prepared": true,
                "executed": false,
                "contracts_are_distinct_by_content_render_and_token_ids": true,
                "unsealed_preimage_sha256": sha('8'),
                "source_authority": {
                    "model_id": MODEL_ID,
                    "model_key": MODEL_KEY,
                    "source_repository": SOURCE_REPOSITORY,
                    "source_revision": SOURCE_REVISION,
                    "tokenizer_addressable_vocab_size": 151_669,
                    "lm_head_vocab_size": 151_936
                },
                "prompt_contracts": [
                    {
                        "label": "A",
                        "rendered_source_template_prompt_sha256": sha('7'),
                        "token_ids_sha256": sha('6'),
                        "token_count": 15
                    },
                    {
                        "label": "B",
                        "rendered_source_template_prompt_sha256": sha('5'),
                        "token_ids_sha256": sha('4'),
                        "token_count": 25
                    }
                ],
                "execution_boundary": {
                    "live_packed_artifact_scan_performed": false,
                    "raw_weight_or_gravity_payload_opened": false,
                    "metal_device_or_dispatch_performed": false,
                    "model_or_decoder_execution_performed": false,
                    "server_or_hcli_execution_performed": false,
                    "generation_or_coherence_evaluation_performed": false,
                    "tps_or_tg_measurement_performed": false
                }
            }),
        );
        Args {
            activation_contract: activation,
            memory_envelope_contract: memory,
            state_contract: state,
            source_template_contract: template,
            out: directory.join("out.json"),
        }
    }

    #[test]
    fn external_preconditions_are_consumed_and_duplicate_model_topology_is_refused() {
        let directory = tempdir().unwrap();
        let args = write_valid_external_documents(directory.path());
        let preconditions = load_preconditions(&args).unwrap();
        let rendered = report(preconditions).unwrap();
        assert_eq!(rendered.logical_session_profiles.len(), 5);
        assert_eq!(
            rendered
                .source_template_handoff
                .prompt_bindings
                .iter()
                .map(|binding| binding.label.as_str())
                .collect::<Vec<_>>(),
            ["A", "B"]
        );

        let mut activation: Value =
            serde_json::from_slice(&fs::read(&args.activation_contract).unwrap()).unwrap();
        activation["target_topology"]["resident_q80_model_processes"] = json!(2);
        write_json(&args.activation_contract, activation);
        assert!(load_preconditions(&args).is_err());
    }
}
