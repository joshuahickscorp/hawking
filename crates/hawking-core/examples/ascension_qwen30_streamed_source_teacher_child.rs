#![allow(dead_code)] // Real source execution is deliberately unreachable in this CPU/build target.

//! Qwen30 real-streamed-source-teacher child execution interface.
//!
//! This target has two deliberately separated surfaces:
//!
//! * `--mode preflight` validates only sealed future authorities and emits a
//!   prepared interface document; and
//! * `--mode source-teacher` accepts the exact future outer-runner command
//!   grammar, validates those authorities *before touching its source root*,
//!   then refuses because this CPU/build target has no physical execution
//!   enablement.
//!
//! The bounded reader, 48 x 370 traversal, two-vector handoff, cache clear,
//! and receipt-last sequence are exercised only against in-memory synthetic
//! fixtures in tests.  No executable path opens a real source root, payload,
//! model, GPU, server, HCLI endpoint, or lease backend.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;

const INTERFACE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_child_execution_interface.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_EXECUTION_INTERFACE_NOT_EXECUTED";
const CPU_BUILD_REFUSED_STATUS: &str =
    "REFUSED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_CPU_BUILD_EXECUTION_NOT_ENABLED";

const RUNTIME_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1";
const RUNTIME_ADMISSION_STATUS: &str =
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY";
const RUNTIME_RANGE_MAP_SCHEMA: &str = "hawking.ascension.qwen30_source_bf16_range_map.v1";
const DUAL_BRIDGE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1";
const DUAL_BRIDGE_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED";
const SOURCE_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_quiet_lease.v1";
const SOURCE_LEASE_STATUS: &str =
    "GRANTED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_RAW_LOGIT_CAPTURE_ONE_SHOT";
const OPERATOR_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1";
const OPERATOR_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED";
const RANGE_READER_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1";
const RANGE_READER_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED";

const SOURCE_CHILD_EVIDENCE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_child_execution_evidence.v1";
const SOURCE_CHILD_EVIDENCE_STATUS: &str =
    "CAPTURED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_TWO_F32LE_LOGITS_NOT_NATIVE_PHASE";
const SYNTHETIC_EVIDENCE_STATUS: &str =
    "SYNTHETIC_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_TWO_F32LE_LOGITS_FIXTURE_ONLY";

const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const MAX_POSITIONED_READ_BYTES: usize = 1024 * 1024;
const SOURCE_LAYERS: usize = 48;
const SOURCE_FORWARDS: usize = 370;
const SOURCE_TOP_K: usize = 8;
const PREFIX_TOKENS: usize = 369;
const FORCED_TOKEN_ID: usize = 949;
const VOCAB_ROWS: usize = 151_936;
const F32_VECTOR_BYTES: usize = VOCAB_ROWS * 4;
const SOURCE_PAYLOADS: [&str; 2] = [
    "source_bf16_exact_prefix_logits.f32le",
    "source_bf16_forced_shared_continuation_logits.f32le",
];
const SYNTHETIC_SOURCE_SHARD_ID: &str = "fixture-qwen30-source-bf16-00001.safetensors-not-real";
const SYNTHETIC_SOURCE_RECEIPT_NAME: &str = "source-teacher-child.synthetic-fixture.receipt.json";
const SOURCE_OPERATOR_ORDER: [&str; 12] = [
    "embedding",
    "rmsnorm",
    "qkv_serial_k",
    "rope_kv_append_then_causal_read",
    "attention_output_serial_k",
    "residual",
    "router_top8",
    "one_selected_expert_body_at_a_time",
    "source_ordered_route_combine",
    "second_residual",
    "final_rmsnorm",
    "lm_head_serial_k",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Preflight,
    SourceTeacher,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "preflight" => Ok(Self::Preflight),
            "source-teacher" => Ok(Self::SourceTeacher),
            _ => Err("--mode must be preflight or source-teacher".to_owned()),
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    runtime_admission: PathBuf,
    dual_bridge: PathBuf,
    source_lease: PathBuf,
    source_root: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    out: Option<PathBuf>,
}

fn usage() -> &'static str {
    "preflight: ascension_qwen30_streamed_source_teacher_child --mode preflight \\\n+     --runtime-admission ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON \\\n+     --dual-attestation-runtime-admission ABSOLUTE_SEALED_DUAL_BRIDGE_JSON \\\n+     --source-lease ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON \\\n+     --out NEW_ABSOLUTE_PREFLIGHT_JSON\n\
source-teacher (outer-compatible default): ascension_qwen30_streamed_source_teacher_child \\\n+     --source-root ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT \\\n+     --runtime-admission ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON \\\n+     --dual-attestation-runtime-admission ABSOLUTE_SEALED_DUAL_BRIDGE_JSON \\\n+     --source-lease ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON \\\n+     --capture-dir NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut mode = None;
    let mut runtime_admission = None;
    let mut dual_bridge = None;
    let mut source_lease = None;
    let mut source_root = None;
    let mut capture_dir = None;
    let mut out = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            return Err(usage().to_owned());
        }
        let next = values
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        match flag.as_str() {
            "--mode" => {
                if mode.replace(Mode::parse(&next)?).is_some() {
                    return Err("--mode was supplied more than once".to_owned());
                }
            }
            "--runtime-admission" => {
                if runtime_admission.replace(PathBuf::from(next)).is_some() {
                    return Err("--runtime-admission was supplied more than once".to_owned());
                }
            }
            "--dual-attestation-runtime-admission" => {
                if dual_bridge.replace(PathBuf::from(next)).is_some() {
                    return Err(
                        "--dual-attestation-runtime-admission was supplied more than once"
                            .to_owned(),
                    );
                }
            }
            "--source-lease" => {
                if source_lease.replace(PathBuf::from(next)).is_some() {
                    return Err("--source-lease was supplied more than once".to_owned());
                }
            }
            "--source-root" => {
                if source_root.replace(PathBuf::from(next)).is_some() {
                    return Err("--source-root was supplied more than once".to_owned());
                }
            }
            "--capture-dir" => {
                if capture_dir.replace(PathBuf::from(next)).is_some() {
                    return Err("--capture-dir was supplied more than once".to_owned());
                }
            }
            "--out" => {
                if out.replace(PathBuf::from(next)).is_some() {
                    return Err("--out was supplied more than once".to_owned());
                }
            }
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        }
    }
    let required = |value: Option<PathBuf>, flag: &str| {
        value.ok_or_else(|| format!("{flag} is required; {}", usage()))
    };
    let mode = mode.unwrap_or(Mode::SourceTeacher);
    let args = Args {
        mode,
        runtime_admission: required(runtime_admission, "--runtime-admission")?,
        dual_bridge: required(dual_bridge, "--dual-attestation-runtime-admission")?,
        source_lease: required(source_lease, "--source-lease")?,
        source_root,
        capture_dir,
        out,
    };
    for (flag, path) in [
        ("--runtime-admission", &args.runtime_admission),
        ("--dual-attestation-runtime-admission", &args.dual_bridge),
        ("--source-lease", &args.source_lease),
    ] {
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
    }
    match args.mode {
        Mode::Preflight => {
            if args.source_root.is_some() || args.capture_dir.is_some() {
                return Err(
                    "preflight mode must not receive --source-root or --capture-dir".to_owned(),
                );
            }
            let out = args
                .out
                .as_ref()
                .ok_or_else(|| "preflight mode requires --out".to_owned())?;
            if !out.is_absolute() {
                return Err("--out must be absolute".to_owned());
            }
        }
        Mode::SourceTeacher => {
            if args.out.is_some() {
                return Err("source-teacher mode must not receive --out".to_owned());
            }
            for (flag, path) in [
                ("--source-root", args.source_root.as_ref()),
                ("--capture-dir", args.capture_dir.as_ref()),
            ] {
                let path =
                    path.ok_or_else(|| format!("{flag} is required in source-teacher mode"))?;
                if !path.is_absolute() {
                    return Err(format!("{flag} must be absolute"));
                }
            }
        }
    }
    Ok(args)
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(env::args().skip(1))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn canonical_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|error| format!("cannot canonicalize string: {error}")),
        Value::Array(values) => values
            .iter()
            .map(canonical_json)
            .collect::<Result<Vec<_>, _>>()
            .map(|values| format!("[{}]", values.join(","))),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut rendered = Vec::with_capacity(keys.len());
            for key in keys {
                let encoded_key = serde_json::to_string(key)
                    .map_err(|error| format!("cannot canonicalize key: {error}"))?;
                let child = values
                    .get(key)
                    .ok_or_else(|| "canonical object key disappeared".to_owned())?;
                rendered.push(format!("{encoded_key}:{}", canonical_json(child)?));
            }
            Ok(format!("{{{}}}", rendered.join(",")))
        }
    }
}

fn seal_value(mut value: Value) -> Result<Value, String> {
    value
        .as_object_mut()
        .ok_or_else(|| "sealed result must be an object".to_owned())?
        .remove("seal_sha256");
    let seal = sha256_hex(canonical_json(&value)?.as_bytes());
    value
        .as_object_mut()
        .ok_or_else(|| "sealed result became non-object".to_owned())?
        .insert("seal_sha256".to_owned(), Value::String(seal));
    Ok(value)
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let root = object(value, label)?;
    let seal = sha256(
        required(root, "seal_sha256", label)?,
        &format!("{label}.seal_sha256"),
    )?;
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(|| format!("{label} became non-object"))?
        .remove("seal_sha256");
    if sha256_hex(canonical_json(&unsigned)?.as_bytes()) != seal {
        return Err(format!("{label} seal does not bind canonical contents"));
    }
    Ok(seal)
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn array<'a>(value: &'a Value, label: &str) -> Result<&'a [Value], String> {
    value
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} must be an array"))
}

fn required<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Value, String> {
    object
        .get(key)
        .ok_or_else(|| format!("{label} lacks required field {key:?}"))
}

fn string<'a>(value: &'a Value, label: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} must be a non-empty string"))
}

fn sha256(value: &Value, label: &str) -> Result<String, String> {
    let value = string(value, label)?;
    if !is_sha256(value) {
        return Err(format!("{label} must be a lowercase SHA-256"));
    }
    Ok(value.to_owned())
}

fn usize_value(value: &Value, label: &str) -> Result<usize, String> {
    value
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{label} must be a non-negative usize"))
}

fn require_bool(value: &Value, expected: bool, label: &str) -> Result<(), String> {
    if value.as_bool() != Some(expected) {
        return Err(format!("{label} must be {expected}"));
    }
    Ok(())
}

fn require_schema_status(
    object: &Map<String, Value>,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if string(
        required(object, "schema", label)?,
        &format!("{label}.schema"),
    )? != schema
        || string(
            required(object, "status", label)?,
            &format!("{label}.status"),
        )? != status
    {
        return Err(format!("{label} schema/status drifted"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SealedDocument {
    path: PathBuf,
    raw_document_sha256: String,
    seal_sha256: String,
    value: Value,
}

fn read_sealed_metadata(path: &Path, label: &str) -> Result<SealedDocument, String> {
    if !path.is_absolute() || path.extension().and_then(|suffix| suffix.to_str()) != Some("json") {
        return Err(format!("{label} must be an absolute .json receipt"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() == 0 || metadata.len() > MAX_METADATA_BYTES {
        return Err(format!("{label} must be 1..={MAX_METADATA_BYTES} bytes"));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{label} changed while being read"));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} is invalid JSON: {error}"))?;
    let seal_sha256 = verify_seal(&value, label)?;
    Ok(SealedDocument {
        path: fs::canonicalize(path)
            .map_err(|error| format!("cannot canonicalize {label}: {error}"))?,
        raw_document_sha256: sha256_hex(&bytes),
        seal_sha256,
        value,
    })
}

fn fixture_document(path: &str, value: Value) -> SealedDocument {
    let bytes = serde_json::to_vec(&value).expect("fixture serializes");
    SealedDocument {
        path: PathBuf::from(path),
        raw_document_sha256: sha256_hex(&bytes),
        seal_sha256: verify_seal(&value, "fixture").expect("fixture is sealed"),
        value,
    }
}

fn evidence(document: &SealedDocument) -> Value {
    json!({
        "path": document.path,
        "raw_document_sha256": document.raw_document_sha256,
        "seal_sha256": document.seal_sha256,
    })
}

#[derive(Clone, Debug)]
struct AuthorityBundle {
    runtime_admission: SealedDocument,
    dual_bridge: SealedDocument,
    source_lease: SealedDocument,
}

fn validate_runtime_admission(document: &SealedDocument) -> Result<String, String> {
    let root = object(&document.value, "runtime-range admission")?;
    require_schema_status(
        root,
        RUNTIME_ADMISSION_SCHEMA,
        RUNTIME_ADMISSION_STATUS,
        "runtime-range admission",
    )?;
    let map = object(
        required(root, "flat_runtime_range_map", "runtime-range admission")?,
        "runtime-range admission.flat_runtime_range_map",
    )?;
    if string(
        required(map, "schema", "runtime-range admission flat map")?,
        "runtime-range admission flat map.schema",
    )? != RUNTIME_RANGE_MAP_SCHEMA
    {
        return Err("runtime-range admission flat map schema drifted".to_owned());
    }
    sha256(
        required(map, "document_sha256", "runtime-range admission flat map")?,
        "runtime-range admission flat map document hash",
    )?;
    let reader = object(
        required(root, "bounded_positioned_reader", "runtime-range admission")?,
        "runtime-range admission.bounded_positioned_reader",
    )?;
    if usize_value(
        required(
            reader,
            "maximum_positioned_read_bytes",
            "runtime-range admission reader",
        )?,
        "runtime-range admission maximum positioned read",
    )? != MAX_POSITIONED_READ_BYTES
        || usize_value(
            required(
                reader,
                "maximum_live_raw_bf16_windows",
                "runtime-range admission reader",
            )?,
            "runtime-range admission maximum live windows",
        )? != 1
    {
        return Err("runtime-range admission bounded-reader geometry drifted".to_owned());
    }
    for field in [
        "no_mmap_or_full_shard_cache",
        "no_model_residency",
        "payload_open_requires_fresh_source_lease",
    ] {
        require_bool(
            required(reader, field, "runtime-range admission reader")?,
            true,
            &format!("runtime-range admission reader.{field}"),
        )?;
    }
    let bridge_seal = sha256(
        required(root, "dual_bridge_seal_sha256", "runtime-range admission")?,
        "runtime-range admission dual bridge seal",
    )?;
    let boundary = object(
        required(root, "execution_boundary", "runtime-range admission")?,
        "runtime-range admission execution boundary",
    )?;
    for field in [
        "source_tensor_payload_opened",
        "source_model_loaded_or_instantiated",
        "gpu_or_metal_invoked",
        "server_started_or_contacted",
        "hcli_invoked",
        "lease_issued_or_consumed",
    ] {
        require_bool(
            required(boundary, field, "runtime-range admission boundary")?,
            false,
            &format!("runtime-range admission boundary.{field}"),
        )?;
    }
    Ok(bridge_seal)
}

fn validate_dual_bridge(document: &SealedDocument) -> Result<(), String> {
    let root = object(&document.value, "dual-attestation/runtime-admission bridge")?;
    require_schema_status(root, DUAL_BRIDGE_SCHEMA, DUAL_BRIDGE_STATUS, "dual bridge")?;
    let resolution = object(
        required(root, "schema_resolution", "dual bridge")?,
        "dual bridge.schema_resolution",
    )?;
    for (field, expected) in [
        ("runtime_range_map_schema", RUNTIME_RANGE_MAP_SCHEMA),
        ("runtime_admission_schema", RUNTIME_ADMISSION_SCHEMA),
        (
            "runtime_admission_status_only_after_bounded_source_validation",
            RUNTIME_ADMISSION_STATUS,
        ),
    ] {
        if string(
            required(resolution, field, "dual bridge resolution")?,
            &format!("dual bridge resolution.{field}"),
        )? != expected
        {
            return Err(format!("dual bridge resolution {field} drifted"));
        }
    }
    for (field, schema, status) in [
        (
            "operator_accumulation_execution_attestation",
            OPERATOR_ATTESTATION_SCHEMA,
            OPERATOR_ATTESTATION_STATUS,
        ),
        (
            "range_reader_exact_semantics_attestation",
            RANGE_READER_ATTESTATION_SCHEMA,
            RANGE_READER_ATTESTATION_STATUS,
        ),
    ] {
        require_schema_status(
            object(
                required(resolution, field, "dual bridge resolution")?,
                &format!("dual bridge {field}"),
            )?,
            schema,
            status,
            &format!("dual bridge {field}"),
        )?;
    }
    for field in [
        "both_execution_attestations_required_after_source_child",
        "runtime_range_admission_required_before_payload_open",
        "bridge_does_not_authorize_execution",
    ] {
        require_bool(
            required(resolution, field, "dual bridge resolution")?,
            true,
            &format!("dual bridge resolution.{field}"),
        )?;
    }
    let worker = object(
        required(root, "future_source_worker", "dual bridge")?,
        "dual bridge.future_source_worker",
    )?;
    for (field, expected) in [
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_f32le_vectors", 2),
        ("native_f32le_vectors", 4),
    ] {
        if usize_value(
            required(worker, field, "dual bridge worker")?,
            &format!("dual bridge worker.{field}"),
        )? != expected
        {
            return Err(format!("dual bridge worker {field} drifted"));
        }
    }
    for field in [
        "one_bounded_window_only",
        "source_payloads_durable_before_eviction",
        "close_handles_and_clear_cache_before_eviction_receipt",
        "separate_native_four_vector_phase_required",
    ] {
        require_bool(
            required(worker, field, "dual bridge worker")?,
            true,
            &format!("dual bridge worker.{field}"),
        )?;
    }
    Ok(())
}

fn validate_source_lease(document: &SealedDocument) -> Result<(), String> {
    let root = object(&document.value, "source lease")?;
    require_schema_status(
        root,
        SOURCE_LEASE_SCHEMA,
        SOURCE_LEASE_STATUS,
        "source lease",
    )?;
    let lifecycle = object(
        required(root, "one_shot_lifecycle", "source lease")?,
        "source lease.one_shot_lifecycle",
    )?;
    for field in [
        "fresh_for_this_exact_launch",
        "new_capture_root",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
    ] {
        require_bool(
            required(lifecycle, field, "source lease lifecycle")?,
            true,
            &format!("source lease lifecycle.{field}"),
        )?;
    }
    require_bool(
        required(
            lifecycle,
            "automatic_retry_allowed",
            "source lease lifecycle",
        )?,
        false,
        "source lease lifecycle.automatic_retry_allowed",
    )?;
    if required(
        lifecycle,
        "prior_terminal_receipt",
        "source lease lifecycle",
    )?
    .is_null()
        == false
    {
        return Err("source lease must not carry a prior terminal receipt".to_owned());
    }
    sha256(
        required(lifecycle, "exact_launch_nonce", "source lease lifecycle")?,
        "source lease lifecycle launch nonce",
    )?;
    let safety = object(
        required(root, "fresh_pre_child_safety", "source lease")?,
        "source lease.fresh_pre_child_safety",
    )?;
    for field in [
        "observed_immediately_before_child",
        "exclusive_clean_window",
        "no_source_or_native_model_body_resident_before_child",
    ] {
        require_bool(
            required(safety, field, "source lease safety")?,
            true,
            &format!("source lease safety.{field}"),
        )?;
    }
    if usize_value(
        required(safety, "swap_used_bytes", "source lease safety")?,
        "source lease safety.swap_used_bytes",
    )? != 0
        || usize_value(
            required(safety, "swapouts_pages_delta", "source lease safety")?,
            "source lease safety.swapouts_pages_delta",
        )? != 0
    {
        return Err("source lease zero-swap safety drifted".to_owned());
    }
    let reclaimable = usize_value(
        required(safety, "reclaimable_bytes", "source lease safety")?,
        "source lease safety.reclaimable_bytes",
    )?;
    let minimum = usize_value(
        required(
            safety,
            "minimum_reclaimable_bytes_required",
            "source lease safety",
        )?,
        "source lease safety minimum reclaimable bytes",
    )?;
    if minimum == 0 || reclaimable < minimum {
        return Err("source lease reclaimable memory floor is not met".to_owned());
    }
    Ok(())
}

fn validate_authority_bundle(
    runtime_admission: SealedDocument,
    dual_bridge: SealedDocument,
    source_lease: SealedDocument,
) -> Result<AuthorityBundle, String> {
    let runtime_bridge_seal = validate_runtime_admission(&runtime_admission)?;
    validate_dual_bridge(&dual_bridge)?;
    validate_source_lease(&source_lease)?;
    if runtime_bridge_seal != dual_bridge.seal_sha256 {
        return Err("runtime-range admission is not bound to the supplied dual bridge".to_owned());
    }
    Ok(AuthorityBundle {
        runtime_admission,
        dual_bridge,
        source_lease,
    })
}

fn source_child_command_grammar() -> Value {
    json!([
        "ascension_qwen30_streamed_source_teacher_child",
        "--source-root",
        "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
        "--runtime-admission",
        "ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON",
        "--dual-attestation-runtime-admission",
        "ABSOLUTE_SEALED_DUAL_BRIDGE_JSON",
        "--source-lease",
        "ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON",
        "--capture-dir",
        "NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY",
    ])
}

fn source_teacher_input_grammar() -> Value {
    json!({
        "runtime_admission": {
            "schema": RUNTIME_ADMISSION_SCHEMA,
            "status": RUNTIME_ADMISSION_STATUS,
            "sealed_earned_before_source_root_open": true,
        },
        "dual_attestation_runtime_admission": {
            "schema": DUAL_BRIDGE_SCHEMA,
            "status": DUAL_BRIDGE_STATUS,
            "sealed_and_bound_to_runtime_admission": true,
            "non_authorizing_without_real_execution_attestations": true,
        },
        "source_lease": {
            "schema": SOURCE_LEASE_SCHEMA,
            "status": SOURCE_LEASE_STATUS,
            "fresh_one_shot_exact_launch_required": true,
            "must_not_be_issued_or_consumed_by_preflight": true,
        },
        "source_root": {
            "accepted_only_after_all_three_authorities_validate": true,
            "regular_non_symlink_shards_only": true,
            "no_mmap_or_full_shard_cache": true,
        },
    })
}

fn source_teacher_execution_shape() -> Value {
    json!({
        "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
        "maximum_live_raw_bf16_windows": 1,
        "source_layers": SOURCE_LAYERS,
        "source_forwards": SOURCE_FORWARDS,
        "source_top_k": SOURCE_TOP_K,
        "source_layer_traversals": SOURCE_LAYERS * SOURCE_FORWARDS,
        "prefix_tokens": PREFIX_TOKENS,
        "forced_token_id": FORCED_TOKEN_ID,
        "sampling_or_autoregressive_feedback_forbidden": true,
        "source_operator_order": SOURCE_OPERATOR_ORDER,
    })
}

fn source_teacher_worker_output_grammar() -> Value {
    json!({
        "worker_evidence_schema": SOURCE_CHILD_EVIDENCE_SCHEMA,
        "worker_evidence_status_only_after_real_source_execution": SOURCE_CHILD_EVIDENCE_STATUS,
        "source_payloads": SOURCE_PAYLOADS,
        "source_payload_dtype": "f32le",
        "source_payload_vocab_rows": VOCAB_ROWS,
        "source_payload_bytes_each": F32_VECTOR_BYTES,
        "required_worker_evidence_fields": [
            "source_payloads.exact_prefix",
            "source_payloads.forced_shared_continuation",
            "source_payloads_are_create_new_f32le_finite_and_fsynced",
            "bounded_per_read_cache.maximum_allowed_window_bytes",
            "bounded_per_read_cache.maximum_observed_window_bytes",
            "bounded_per_read_cache.maximum_cached_bytes",
            "bounded_per_read_cache.maximum_cached_windows",
            "source_payload_read_accounting.per_shard",
            "runtime_range_admission.seal_sha256",
            "operator_accumulation_execution_attestation.seal_sha256",
            "range_reader_exact_semantics_attestation.seal_sha256",
            "source_handles_closed",
            "streamed_reader_cache_zeroed",
            "source_backend_shutdown",
            "child_exit_after_payload_fsyncs",
        ],
        "must_fsync_two_payloads_before_child_exit": true,
        "must_close_all_source_handles_and_zero_reader_cache_before_child_exit": true,
        "must_emit_runtime_admission_and_both_execution_attestation_identities": true,
        "must_not_write_source_terminal_or_start_native_phase": true,
    })
}

fn source_teacher_outer_handoff_grammar() -> Value {
    json!({
        "worker_must_not_write_source_terminal": true,
        "worker_must_not_start_native_phase": true,
        "outer_must_reap_source_child_before_terminal": true,
        "outer_source_terminal_required_fields": [
            "source_lease.seal_sha256",
            "exact_trace",
            "streamed_execution.mode=layer_streamed_bf16_source_teacher",
            "streamed_execution.outer_reaped_child_before_terminal_receipt=true",
            "streamed_execution.receipt_written_after_payload_fsyncs=true",
            "source_payloads",
            "bounded_per_read_cache",
            "source_payload_read_accounting",
            "dual_execution_attestations.runtime_range_admission",
            "dual_execution_attestations.operator_accumulation",
            "dual_execution_attestations.range_reader_exact_semantics",
        ],
        "source_eviction_before_distinct_native_lease_required": true,
        "six_vector_terminal_requires_all_payload_fsyncs": true,
        "metric_scoring_outside_source_child_and_outer_handoff": true,
    })
}

fn prepared_interface(bundle: &AuthorityBundle) -> Result<Value, String> {
    seal_value(json!({
        "schema": INTERFACE_SCHEMA,
        "status": PREPARED_STATUS,
        "prepared": true,
        "execution_authorized": false,
        "outer_runner_source_child_command": source_child_command_grammar(),
        "input_grammar": source_teacher_input_grammar(),
        "execution_shape": source_teacher_execution_shape(),
        "worker_output_grammar": source_teacher_worker_output_grammar(),
        "exact_outer_handoff_fields": source_teacher_outer_handoff_grammar(),
        "sealed_authorities": {
            "runtime_range_admission": evidence(&bundle.runtime_admission),
            "dual_attestation_runtime_admission_bridge": evidence(&bundle.dual_bridge),
            "source_lease": evidence(&bundle.source_lease),
        },
        "future_source_teacher_shape": {
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "source_layers": SOURCE_LAYERS,
            "source_forwards": SOURCE_FORWARDS,
            "source_top_k": SOURCE_TOP_K,
            "source_layer_traversals": SOURCE_LAYERS * SOURCE_FORWARDS,
            "source_template_token_count": PREFIX_TOKENS,
            "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
            "source_payloads": SOURCE_PAYLOADS,
            "payload_dtype": "f32le",
            "payload_vocab_rows": VOCAB_ROWS,
            "payload_bytes_each": F32_VECTOR_BYTES,
            "source_operator_order": SOURCE_OPERATOR_ORDER,
            "two_vectors_must_be_durable_before_child_exit": true,
            "source_handles_must_close_before_child_exit": true,
            "reader_cache_must_be_zeroed_before_child_exit": true,
            "evidence_receipt_must_be_written_last": true,
            "future_real_evidence": {
                "schema": SOURCE_CHILD_EVIDENCE_SCHEMA,
                "status_only_after_real_source_execution": SOURCE_CHILD_EVIDENCE_STATUS,
            },
        },
        "future_evidence_required": [
            "runtime_range_admission and dual-bridge seal identities",
            "all 16 shard and all authorized BF16 range/hash identities",
            "bounded positioned-read/cache accounting",
            "48 x 370 operator/accumulation and source order evidence",
            "two complete finite F32LE vectors fsynced before child exit",
            "all source handles closed and cache zeroed before receipt-last evidence",
        ],
        "execution_boundary": {
            "source_root_opened_or_statted": false,
            "source_tensor_payload_opened": false,
            "source_model_loaded_or_instantiated": false,
            "whole_source_model_resident": false,
            "gpu_metal_mps_or_other_accelerator_invoked": false,
            "server_started_or_contacted": false,
            "hcli_invoked": false,
            "lease_issued_or_consumed": false,
            "capture_directory_created": false,
            "source_vectors_written": false,
            "real_execution_evidence_written": false,
        },
        "claim_boundary": "Prepared CPU/build interface only. A future physical child must revalidate all three sealed inputs before opening a source root and may not substitute a synthetic fixture, source body, model, server, GPU, HCLI call, or lease action for those checks.",
    }))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RealSourceCommitPhase {
    AuthoritiesValidatedBeforeSourceRootOpen,
    HypotheticalFirstPayloadFsynced,
    HypotheticalTwoPayloadsFsynced,
    HypotheticalSourceClosedAndCacheZeroed,
}

impl RealSourceCommitPhase {
    fn label(self) -> &'static str {
        match self {
            Self::AuthoritiesValidatedBeforeSourceRootOpen => {
                "sealed_authorities_validated_before_source_root_open"
            }
            Self::HypotheticalFirstPayloadFsynced => {
                "hypothetical_first_f32le_payload_fsynced_before_second_payload"
            }
            Self::HypotheticalTwoPayloadsFsynced => {
                "hypothetical_two_f32le_payloads_fsynced_before_source_close"
            }
            Self::HypotheticalSourceClosedAndCacheZeroed => {
                "hypothetical_source_closed_and_cache_zeroed_before_receipt_last"
            }
        }
    }
}

fn cpu_build_refusal(
    bundle: &AuthorityBundle,
    source_root: &Path,
    capture_dir: &Path,
) -> Result<Value, String> {
    cpu_build_refusal_at_phase(
        bundle,
        source_root,
        capture_dir,
        RealSourceCommitPhase::AuthoritiesValidatedBeforeSourceRootOpen,
    )
}

/// Seal a refusal without probing either path.  The non-default phases are
/// testable hypothetical boundaries only: no call in this target may claim
/// that a real-source payload was written or a real commit occurred.
fn cpu_build_refusal_at_phase(
    bundle: &AuthorityBundle,
    source_root: &Path,
    capture_dir: &Path,
    hypothetical_phase: RealSourceCommitPhase,
) -> Result<Value, String> {
    seal_value(json!({
        "schema": INTERFACE_SCHEMA,
        "status": CPU_BUILD_REFUSED_STATUS,
        "prepared": false,
        "execution_authorized": false,
        "source_root_reference": source_root,
        "capture_dir_reference": capture_dir,
        "sealed_authorities": {
            "runtime_range_admission": evidence(&bundle.runtime_admission),
            "dual_attestation_runtime_admission_bridge": evidence(&bundle.dual_bridge),
            "source_lease": evidence(&bundle.source_lease),
        },
        "refusal_reason": "this CPU/build target has no real-source execution enablement; no source-root operation occurs after authority validation",
        "phase_accurate_refusal": {
            "actual_phase": RealSourceCommitPhase::AuthoritiesValidatedBeforeSourceRootOpen.label(),
            "hypothetical_commit_boundary": hypothetical_phase.label(),
            "actual_source_commit_performed": false,
            "actual_payload_fsync_performed": false,
            "actual_source_close_or_cache_clear_performed": false,
            "actual_receipt_last_written": false,
        },
        "execution_boundary": {
            "source_root_opened_or_statted": false,
            "source_tensor_payload_opened": false,
            "source_model_loaded_or_instantiated": false,
            "whole_source_model_resident": false,
            "gpu_metal_mps_or_other_accelerator_invoked": false,
            "server_started_or_contacted": false,
            "hcli_invoked": false,
            "lease_issued_or_consumed": false,
            "capture_directory_created": false,
            "source_vectors_written": false,
            "real_execution_evidence_written": false,
        },
        "claim_boundary": "Refusal occurs before opening or statting the supplied source root/capture directory. No source payload or model operation occurred.",
    }))
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() || !path.parent().is_some_and(Path::is_dir) || path.exists() {
        return Err("--out must be a new absolute path below an existing parent".to_owned());
    }
    let rendered = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize preflight: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(&rendered)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync {}: {error}", path.display()))
}

fn load_bundle(args: &Args) -> Result<AuthorityBundle, String> {
    validate_authority_bundle(
        read_sealed_metadata(&args.runtime_admission, "runtime-range admission")?,
        read_sealed_metadata(
            &args.dual_bridge,
            "dual-attestation/runtime-admission bridge",
        )?,
        read_sealed_metadata(&args.source_lease, "source lease")?,
    )
}

fn run(args: Args) -> Result<Value, String> {
    // This deliberately evaluates every sealed authority before referencing the
    // source-root/capture-dir fields.  The source-teacher branch contains no
    // filesystem operation on either path in this CPU/build target.
    let bundle = load_bundle(&args)?;
    match args.mode {
        Mode::Preflight => {
            let output = prepared_interface(&bundle)?;
            write_new_json(args.out.as_ref().expect("validated --out"), &output)?;
            Ok(output)
        }
        Mode::SourceTeacher => cpu_build_refusal(
            &bundle,
            args.source_root.as_deref().expect("validated source root"),
            args.capture_dir
                .as_deref()
                .expect("validated capture directory"),
        ),
    }
}

#[derive(Debug)]
struct SyntheticBoundedReader {
    maximum_window_bytes: usize,
    cache: Vec<u8>,
    maximum_observed_cache_bytes: usize,
    maximum_cached_bytes: usize,
    read_calls: usize,
    payload_bytes_read: usize,
    cache_zeroed_after_each_read: bool,
    cache_zeroed_at_close: bool,
    closed: bool,
}

impl SyntheticBoundedReader {
    fn new(maximum_window_bytes: usize) -> Result<Self, String> {
        if maximum_window_bytes == 0 || maximum_window_bytes > MAX_POSITIONED_READ_BYTES {
            return Err("synthetic reader window must be 1..=1 MiB".to_owned());
        }
        Ok(Self {
            maximum_window_bytes,
            cache: Vec::with_capacity(maximum_window_bytes),
            maximum_observed_cache_bytes: 0,
            maximum_cached_bytes: 0,
            read_calls: 0,
            payload_bytes_read: 0,
            cache_zeroed_after_each_read: true,
            cache_zeroed_at_close: false,
            closed: false,
        })
    }

    fn clear_cache(&mut self) {
        self.cache.fill(0);
        self.cache.clear();
        self.cache_zeroed_after_each_read = true;
    }

    fn positioned_read(
        &mut self,
        source: &[u8],
        offset: usize,
        length: usize,
    ) -> Result<&[u8], String> {
        if self.closed {
            return Err("synthetic reader is closed".to_owned());
        }
        if length == 0 || length > self.maximum_window_bytes {
            return Err("synthetic reader positioned read exceeds its bounded window".to_owned());
        }
        let end = offset
            .checked_add(length)
            .ok_or_else(|| "synthetic reader offset overflowed".to_owned())?;
        let bytes = source
            .get(offset..end)
            .ok_or_else(|| "synthetic reader positioned read exceeds fixture source".to_owned())?;
        self.clear_cache();
        self.cache.extend_from_slice(bytes);
        self.maximum_observed_cache_bytes = self.maximum_observed_cache_bytes.max(self.cache.len());
        self.maximum_cached_bytes = self.maximum_cached_bytes.max(self.cache.len());
        self.read_calls = self
            .read_calls
            .checked_add(1)
            .ok_or_else(|| "synthetic reader read-call count overflowed".to_owned())?;
        self.payload_bytes_read = self
            .payload_bytes_read
            .checked_add(self.cache.len())
            .ok_or_else(|| "synthetic reader byte accounting overflowed".to_owned())?;
        self.cache_zeroed_after_each_read = false;
        Ok(&self.cache)
    }

    /// The only visitor used by the source-teacher backend.  It never exposes
    /// a raw range outside the callback and zeroes the one bounded window
    /// before another positioned read can begin.
    fn visit_positioned_read(
        &mut self,
        source: &[u8],
        offset: usize,
        length: usize,
        visitor: impl FnOnce(&[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        let result = {
            let bytes = self.positioned_read(source, offset, length)?;
            visitor(bytes)
        };
        self.clear_cache();
        result
    }

    fn close_and_zero_cache(&mut self) {
        self.clear_cache();
        self.cache.shrink_to(0);
        self.cache_zeroed_at_close = true;
        self.closed = true;
    }

    fn cache_is_zeroed_and_closed(&self) -> bool {
        self.closed
            && self.cache.is_empty()
            && self.cache_zeroed_after_each_read
            && self.cache_zeroed_at_close
    }
}

/// These are the only phases a fixture backend can commit.  A real source
/// execution is deliberately not among them; the command-line child refuses
/// at the authority barrier instead of mapping, opening, or probing a root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SyntheticBackendPhase {
    Ready,
    VisitingSourceFixture,
    ExactPayloadFsynced,
    BothPayloadsFsynced,
    SourceClosedAndCacheZeroed,
    ReceiptWrittenLast,
}

impl SyntheticBackendPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Ready => "ready_before_fixture_capture_directory_create",
            Self::VisitingSourceFixture => "fixture_positioned_read_and_operator_visit",
            Self::ExactPayloadFsynced => "exact_prefix_f32le_payload_fsynced",
            Self::BothPayloadsFsynced => "both_source_f32le_payloads_fsynced",
            Self::SourceClosedAndCacheZeroed => {
                "fixture_source_closed_and_single_read_cache_zeroed"
            }
            Self::ReceiptWrittenLast => "fixture_receipt_written_last",
        }
    }
}

/// An in-memory fixture source has no file descriptor, source root, model, or
/// payload path.  Closing it overwrites every synthetic shard before clearing
/// it, so the fixture exercises the same close/cache-zero ordering without
/// granting a real-source I/O surface.
#[derive(Debug)]
struct SyntheticFixtureSource {
    shards: BTreeMap<String, Vec<u8>>,
    closed: bool,
    zeroed_on_close: bool,
}

impl SyntheticFixtureSource {
    fn new() -> Self {
        let mut shards = BTreeMap::new();
        let bytes = (0..512usize)
            .map(|index| ((index.wrapping_mul(37) ^ (index >> 1)) & 0xff) as u8)
            .collect::<Vec<_>>();
        shards.insert(SYNTHETIC_SOURCE_SHARD_ID.to_owned(), bytes);
        Self {
            shards,
            closed: false,
            zeroed_on_close: false,
        }
    }

    fn shard(&self) -> Result<&[u8], String> {
        if self.closed {
            return Err("synthetic fixture source was used after close".to_owned());
        }
        self.shards
            .get(SYNTHETIC_SOURCE_SHARD_ID)
            .map(Vec::as_slice)
            .filter(|bytes| !bytes.is_empty())
            .ok_or_else(|| "synthetic fixture source shard is absent".to_owned())
    }

    fn close_and_zero(&mut self) {
        for bytes in self.shards.values_mut() {
            bytes.fill(0);
            bytes.clear();
            bytes.shrink_to(0);
        }
        self.closed = true;
        self.zeroed_on_close = true;
    }

    fn closed_and_zeroed(&self) -> bool {
        self.closed && self.zeroed_on_close && self.shards.values().all(Vec::is_empty)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceTeacherOperator {
    Embedding,
    RmsNorm,
    QkvSerialK,
    RopeKvAppendThenCausalRead,
    AttentionOutputSerialK,
    Residual,
    RouterTopK,
    OneSelectedExpertBodyAtATime,
    SourceOrderedRouteCombine,
    SecondResidual,
    FinalRmsNorm,
    LmHeadSerialK,
}

impl SourceTeacherOperator {
    fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::RmsNorm => "rmsnorm",
            Self::QkvSerialK => "qkv_serial_k",
            Self::RopeKvAppendThenCausalRead => "rope_kv_append_then_causal_read",
            Self::AttentionOutputSerialK => "attention_output_serial_k",
            Self::Residual => "residual",
            Self::RouterTopK => "router_top8",
            Self::OneSelectedExpertBodyAtATime => "one_selected_expert_body_at_a_time",
            Self::SourceOrderedRouteCombine => "source_ordered_route_combine",
            Self::SecondResidual => "second_residual",
            Self::FinalRmsNorm => "final_rmsnorm",
            Self::LmHeadSerialK => "lm_head_serial_k",
        }
    }

    fn ordinal(self) -> usize {
        SOURCE_OPERATOR_ORDER
            .iter()
            .position(|name| *name == self.name())
            .expect("operator name is fixed by SOURCE_OPERATOR_ORDER")
    }
}

const LAYER_SOURCE_TEACHER_OPERATORS: [SourceTeacherOperator; 11] = [
    SourceTeacherOperator::RmsNorm,
    SourceTeacherOperator::QkvSerialK,
    SourceTeacherOperator::RopeKvAppendThenCausalRead,
    SourceTeacherOperator::AttentionOutputSerialK,
    SourceTeacherOperator::Residual,
    SourceTeacherOperator::RouterTopK,
    SourceTeacherOperator::OneSelectedExpertBodyAtATime,
    SourceTeacherOperator::SourceOrderedRouteCombine,
    SourceTeacherOperator::SecondResidual,
    SourceTeacherOperator::FinalRmsNorm,
    SourceTeacherOperator::LmHeadSerialK,
];

#[derive(Debug)]
struct SyntheticOperatorHooks {
    source_layer_traversals: usize,
    total_operator_hooks: usize,
    per_operator: BTreeMap<&'static str, usize>,
    byte_checksum: u64,
    order_hasher: Sha256,
}

impl SyntheticOperatorHooks {
    fn new() -> Self {
        Self {
            source_layer_traversals: 0,
            total_operator_hooks: 0,
            per_operator: BTreeMap::new(),
            byte_checksum: 0,
            order_hasher: Sha256::new(),
        }
    }

    fn observe(
        &mut self,
        forward: usize,
        layer: Option<usize>,
        route_slot: Option<usize>,
        operator: SourceTeacherOperator,
        window: &[u8],
    ) -> Result<(), String> {
        if window.is_empty() || window.len() > MAX_POSITIONED_READ_BYTES {
            return Err("synthetic operator hook received an invalid bounded window".to_owned());
        }
        self.total_operator_hooks = self
            .total_operator_hooks
            .checked_add(1)
            .ok_or_else(|| "synthetic operator hook count overflowed".to_owned())?;
        let count = self.per_operator.entry(operator.name()).or_default();
        *count = count
            .checked_add(1)
            .ok_or_else(|| "synthetic per-operator count overflowed".to_owned())?;
        self.byte_checksum = self.byte_checksum.wrapping_add(
            window
                .iter()
                .fold(0u64, |sum, byte| sum.wrapping_add(u64::from(*byte))),
        );
        self.order_hasher.update((forward as u64).to_le_bytes());
        self.order_hasher
            .update((layer.unwrap_or(usize::MAX) as u64).to_le_bytes());
        self.order_hasher
            .update((route_slot.unwrap_or(usize::MAX) as u64).to_le_bytes());
        self.order_hasher.update(operator.name().as_bytes());
        self.order_hasher.update(window);
        Ok(())
    }

    fn mark_layer_traversal(&mut self) -> Result<(), String> {
        self.source_layer_traversals = self
            .source_layer_traversals
            .checked_add(1)
            .ok_or_else(|| "synthetic layer traversal count overflowed".to_owned())?;
        Ok(())
    }

    fn verify_complete(&self) -> Result<(), String> {
        if self.source_layer_traversals != SOURCE_LAYERS * SOURCE_FORWARDS {
            return Err("synthetic source traversal count does not equal 48 x 370".to_owned());
        }
        let expected_embedding = SOURCE_FORWARDS;
        if self
            .per_operator
            .get(SourceTeacherOperator::Embedding.name())
            != Some(&expected_embedding)
        {
            return Err("synthetic source embedding hook count drifted".to_owned());
        }
        let expected_layer_hook_count = SOURCE_LAYERS * SOURCE_FORWARDS;
        for operator in LAYER_SOURCE_TEACHER_OPERATORS {
            let expected = if operator == SourceTeacherOperator::OneSelectedExpertBodyAtATime {
                expected_layer_hook_count
                    .checked_mul(SOURCE_TOP_K)
                    .ok_or_else(|| "synthetic expected routed hook count overflowed".to_owned())?
            } else {
                expected_layer_hook_count
            };
            if self.per_operator.get(operator.name()) != Some(&expected) {
                return Err(format!(
                    "synthetic source operator hook count drifted for {}",
                    operator.name()
                ));
            }
        }
        let per_layer_operator_calls = LAYER_SOURCE_TEACHER_OPERATORS
            .len()
            .checked_sub(1)
            .and_then(|ordinary| ordinary.checked_add(SOURCE_TOP_K))
            .ok_or_else(|| "synthetic per-layer operator count overflowed".to_owned())?;
        let expected_total = expected_embedding
            .checked_add(
                expected_layer_hook_count
                    .checked_mul(per_layer_operator_calls)
                    .ok_or_else(|| "synthetic expected operator count overflowed".to_owned())?,
            )
            .ok_or_else(|| "synthetic expected total operator count overflowed".to_owned())?;
        if self.total_operator_hooks != expected_total {
            return Err("synthetic source total operator hook count drifted".to_owned());
        }
        Ok(())
    }

    fn order_sha256(&self) -> String {
        format!("{:x}", self.order_hasher.clone().finalize())
    }
}

fn visit_source_operator(
    source: &SyntheticFixtureSource,
    reader: &mut SyntheticBoundedReader,
    hooks: &mut SyntheticOperatorHooks,
    forward: usize,
    layer: Option<usize>,
    route_slot: Option<usize>,
    operator: SourceTeacherOperator,
) -> Result<(), String> {
    let bytes = source.shard()?;
    let layer_seed = layer.unwrap_or(SOURCE_LAYERS);
    let offset = forward
        .wrapping_mul(131)
        .wrapping_add(layer_seed.wrapping_mul(29))
        .wrapping_add(route_slot.unwrap_or(SOURCE_TOP_K).wrapping_mul(23))
        .wrapping_add(operator.ordinal().wrapping_mul(17))
        % bytes.len();
    let length = (bytes.len() - offset).min(64);
    reader.visit_positioned_read(bytes, offset, length, |window| {
        hooks.observe(forward, layer, route_slot, operator, window)
    })
}

fn synthetic_logits(seed: u64, endpoint_domain: u64) -> Vec<f32> {
    (0..VOCAB_ROWS)
        .map(|index| {
            let bucket = seed
                .wrapping_add(endpoint_domain)
                .wrapping_add((index as u64).wrapping_mul(0x9e37_79b9))
                & 0x0000_ffff;
            (bucket as f32 / 512.0) - 64.0
        })
        .collect()
}

fn execute_synthetic_source_teacher(
    source: &SyntheticFixtureSource,
    reader: &mut SyntheticBoundedReader,
) -> Result<(SyntheticOperatorHooks, Vec<f32>, Vec<f32>), String> {
    let mut hooks = SyntheticOperatorHooks::new();
    let mut exact_prefix_seed = None;
    let mut forced_continuation_seed = None;
    for forward in 0..SOURCE_FORWARDS {
        visit_source_operator(
            source,
            reader,
            &mut hooks,
            forward,
            None,
            None,
            SourceTeacherOperator::Embedding,
        )?;
        for layer in 0..SOURCE_LAYERS {
            hooks.mark_layer_traversal()?;
            for operator in LAYER_SOURCE_TEACHER_OPERATORS {
                if operator == SourceTeacherOperator::OneSelectedExpertBodyAtATime {
                    for route_slot in 0..SOURCE_TOP_K {
                        visit_source_operator(
                            source,
                            reader,
                            &mut hooks,
                            forward,
                            Some(layer),
                            Some(route_slot),
                            operator,
                        )?;
                    }
                } else {
                    visit_source_operator(
                        source,
                        reader,
                        &mut hooks,
                        forward,
                        Some(layer),
                        None,
                        operator,
                    )?;
                }
            }
        }
        if forward + 1 == PREFIX_TOKENS {
            exact_prefix_seed = Some(hooks.byte_checksum);
        }
        if forward + 1 == SOURCE_FORWARDS {
            forced_continuation_seed = Some(hooks.byte_checksum);
        }
    }
    hooks.verify_complete()?;
    if !reader.cache_zeroed_after_each_read || !reader.cache.is_empty() {
        return Err("synthetic source reader retained a raw window after a visitor".to_owned());
    }
    let exact_prefix = synthetic_logits(
        exact_prefix_seed
            .ok_or_else(|| "synthetic exact-prefix endpoint was not reached".to_owned())?,
        0x4558_4143_54,
    );
    let forced_continuation = synthetic_logits(
        forced_continuation_seed
            .ok_or_else(|| "synthetic forced-continuation endpoint was not reached".to_owned())?,
        0x464f_5243_4544,
    );
    Ok((hooks, exact_prefix, forced_continuation))
}

fn f32le_bytes(values: &[f32]) -> Result<Vec<u8>, String> {
    if values.len() != VOCAB_ROWS || values.iter().any(|value| !value.is_finite()) {
        return Err(
            "synthetic source payload must be a complete finite F32LE vocabulary vector".to_owned(),
        );
    }
    let mut bytes = Vec::with_capacity(F32_VECTOR_BYTES);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    if bytes.len() != F32_VECTOR_BYTES {
        return Err("synthetic source F32LE payload byte count drifted".to_owned());
    }
    Ok(bytes)
}

fn sync_directory(path: &Path) -> Result<(), String> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            format!(
                "cannot fsync synthetic fixture directory {}: {error}",
                path.display()
            )
        })
}

fn create_new_synced_file(path: &Path, bytes: &[u8], label: &str) -> Result<(), String> {
    if path.exists() {
        return Err(format!("{label} already exists: {}", path.display()));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {label} {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync {label} {}: {error}", path.display()))
}

fn create_new_synthetic_capture_dir(capture_dir: &Path) -> Result<(), String> {
    if !capture_dir.is_absolute()
        || capture_dir.exists()
        || !capture_dir.parent().is_some_and(Path::is_dir)
    {
        return Err(
            "synthetic capture dir must be new, absolute, and below an existing parent".to_owned(),
        );
    }
    fs::create_dir(capture_dir).map_err(|error| {
        format!(
            "cannot create synthetic fixture capture directory {}: {error}",
            capture_dir.display()
        )
    })?;
    sync_directory(
        capture_dir
            .parent()
            .ok_or_else(|| "synthetic capture dir lacks a parent".to_owned())?,
    )
}

fn write_synthetic_f32le_payload(
    capture_dir: &Path,
    name: &str,
    values: &[f32],
) -> Result<Value, String> {
    if !SOURCE_PAYLOADS.contains(&name) {
        return Err("synthetic backend rejected an unknown source payload name".to_owned());
    }
    let bytes = f32le_bytes(values)?;
    let path = capture_dir.join(name);
    create_new_synced_file(&path, &bytes, "synthetic F32LE source payload")?;
    sync_directory(capture_dir)?;
    Ok(json!({
        "name": name,
        "path": path,
        "dtype": "f32le",
        "vocab_rows": values.len(),
        "bytes": bytes.len(),
        "sha256": sha256_hex(&bytes),
        "all_values_finite": true,
        "create_new": true,
        "fsynced": true,
        "synthetic_fixture_only": true,
    }))
}

fn write_synthetic_receipt_last(capture_dir: &Path, document: Value) -> Result<Value, String> {
    let sealed = seal_value(document)?;
    let mut bytes = serde_json::to_vec_pretty(&sealed)
        .map_err(|error| format!("cannot serialize synthetic fixture receipt: {error}"))?;
    bytes.push(b'\n');
    create_new_synced_file(
        &capture_dir.join(SYNTHETIC_SOURCE_RECEIPT_NAME),
        &bytes,
        "synthetic source-teacher receipt-last evidence",
    )?;
    sync_directory(capture_dir)?;
    Ok(sealed)
}

/// The new backend is explicitly fixture-only.  It uses an in-memory shard
/// and test-only capture directory, never a source-root argument, so a future
/// real child cannot accidentally route through this implementation.
fn run_synthetic_fixture_backend(capture_dir: &Path) -> Result<Value, String> {
    let mut phase = SyntheticBackendPhase::Ready;
    let result = (|| {
        create_new_synthetic_capture_dir(capture_dir)?;
        phase = SyntheticBackendPhase::VisitingSourceFixture;
        let mut source = SyntheticFixtureSource::new();
        let mut reader = SyntheticBoundedReader::new(64)?;
        let (hooks, exact_prefix, forced_continuation) =
            execute_synthetic_source_teacher(&source, &mut reader)?;

        let exact_payload =
            write_synthetic_f32le_payload(capture_dir, SOURCE_PAYLOADS[0], &exact_prefix)?;
        phase = SyntheticBackendPhase::ExactPayloadFsynced;
        let forced_payload =
            write_synthetic_f32le_payload(capture_dir, SOURCE_PAYLOADS[1], &forced_continuation)?;
        phase = SyntheticBackendPhase::BothPayloadsFsynced;

        source.close_and_zero();
        reader.close_and_zero_cache();
        if !source.closed_and_zeroed() || !reader.cache_is_zeroed_and_closed() {
            return Err("synthetic source close/cache-zero precondition failed".to_owned());
        }
        phase = SyntheticBackendPhase::SourceClosedAndCacheZeroed;

        let receipt = write_synthetic_receipt_last(
            capture_dir,
            json!({
                "schema": SOURCE_CHILD_EVIDENCE_SCHEMA,
                "status": SYNTHETIC_EVIDENCE_STATUS,
                "synthetic_fixture": true,
                "not_real_source_execution": true,
                "source_payloads": {
                    "exact_prefix": exact_payload,
                    "forced_shared_continuation": forced_payload,
                    "source_payloads_are_create_new_f32le_finite_and_fsynced": true,
                },
                "traversal": {
                    "source_layers": SOURCE_LAYERS,
                    "source_forwards": SOURCE_FORWARDS,
                    "source_top_k": SOURCE_TOP_K,
                    "observed_layer_traversals": hooks.source_layer_traversals,
                    "total_operator_hooks": hooks.total_operator_hooks,
                    "source_operator_order": SOURCE_OPERATOR_ORDER,
                    "operator_hook_counts": hooks.per_operator,
                    "operator_order_and_bounded_window_sha256": hooks.order_sha256(),
                },
                "bounded_per_read_cache": {
                    "maximum_allowed_window_bytes": MAX_POSITIONED_READ_BYTES,
                    "configured_window_bytes": reader.maximum_window_bytes,
                    "maximum_observed_window_bytes": reader.maximum_observed_cache_bytes,
                    "maximum_cached_bytes": reader.maximum_cached_bytes,
                    "maximum_cached_windows": 1,
                    "read_calls": reader.read_calls,
                    "payload_bytes_read": reader.payload_bytes_read,
                    "cache_zeroed_after_every_positioned_read": reader.cache_zeroed_after_each_read,
                    "cache_zeroed_before_receipt": reader.cache_zeroed_at_close,
                },
                "source_payload_read_accounting": {
                    "per_shard": [{
                        "relative_path": SYNTHETIC_SOURCE_SHARD_ID,
                        "payload_bytes_read": reader.payload_bytes_read,
                        "read_calls": reader.read_calls,
                        "whole_shard_read_as_one_window": false,
                        "whole_shard_cached": false,
                    }],
                },
                "source_handles_closed": true,
                "streamed_reader_cache_zeroed": true,
                "source_backend_shutdown": true,
                "child_exit_after_payload_fsyncs": true,
                "receipt_written_last_after_synthetic_payload_durability": true,
                "artifact_write_order": [
                    SOURCE_PAYLOADS[0],
                    SOURCE_PAYLOADS[1],
                    "fixture_source_close_and_reader_cache_zero",
                    SYNTHETIC_SOURCE_RECEIPT_NAME,
                ],
                "claim_boundary": "Synthetic in-memory fixture only. It exercised the future bounded visitor and receipt-last lifecycle without a source root, source payload file, source model, GPU, server, HCLI, lease action, source terminal, source eviction receipt, native child, benchmark, TPS, TG, capability, or tournament.",
                "execution_boundary": {
                    "real_source_root_opened": false,
                    "real_source_payload_opened": false,
                    "source_model_loaded_or_instantiated": false,
                    "gpu_metal_mps_or_other_accelerator_invoked": false,
                    "server_started_or_contacted": false,
                    "hcli_invoked": false,
                    "lease_issued_or_consumed": false,
                    "source_terminal_written": false,
                    "native_phase_started": false,
                },
            }),
        )?;
        phase = SyntheticBackendPhase::ReceiptWrittenLast;
        Ok(receipt)
    })();
    result.map_err(|error| {
        format!(
            "synthetic source-teacher backend refused at phase {}: {error}",
            phase.label()
        )
    })
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("cannot render Q30 source-teacher interface result: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("Q30 source-teacher child refused before source access: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn hash(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn sealed(value: Value) -> Value {
        seal_value(value).expect("fixture seals")
    }

    fn runtime_admission(bridge_seal: &str) -> SealedDocument {
        fixture_document(
            "/fixtures/runtime-admission.json",
            sealed(json!({
                "schema": RUNTIME_ADMISSION_SCHEMA,
                "status": RUNTIME_ADMISSION_STATUS,
                "flat_runtime_range_map": {
                    "schema": RUNTIME_RANGE_MAP_SCHEMA,
                    "document_sha256": hash("flat-map"),
                },
                "bounded_positioned_reader": {
                    "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
                    "maximum_live_raw_bf16_windows": 1,
                    "no_mmap_or_full_shard_cache": true,
                    "no_model_residency": true,
                    "payload_open_requires_fresh_source_lease": true,
                },
                "dual_bridge_seal_sha256": bridge_seal,
                "execution_boundary": {
                    "source_tensor_payload_opened": false,
                    "source_model_loaded_or_instantiated": false,
                    "gpu_or_metal_invoked": false,
                    "server_started_or_contacted": false,
                    "hcli_invoked": false,
                    "lease_issued_or_consumed": false,
                },
            })),
        )
    }

    fn dual_bridge() -> SealedDocument {
        fixture_document(
            "/fixtures/bridge.json",
            sealed(json!({
                "schema": DUAL_BRIDGE_SCHEMA,
                "status": DUAL_BRIDGE_STATUS,
                "schema_resolution": {
                    "runtime_range_map_schema": RUNTIME_RANGE_MAP_SCHEMA,
                    "runtime_admission_schema": RUNTIME_ADMISSION_SCHEMA,
                    "runtime_admission_status_only_after_bounded_source_validation": RUNTIME_ADMISSION_STATUS,
                    "operator_accumulation_execution_attestation": {"schema": OPERATOR_ATTESTATION_SCHEMA, "status": OPERATOR_ATTESTATION_STATUS},
                    "range_reader_exact_semantics_attestation": {"schema": RANGE_READER_ATTESTATION_SCHEMA, "status": RANGE_READER_ATTESTATION_STATUS},
                    "both_execution_attestations_required_after_source_child": true,
                    "runtime_range_admission_required_before_payload_open": true,
                    "bridge_does_not_authorize_execution": true,
                },
                "future_source_worker": {
                    "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
                    "source_layers": SOURCE_LAYERS,
                    "source_forwards": SOURCE_FORWARDS,
                    "source_f32le_vectors": 2,
                    "native_f32le_vectors": 4,
                    "one_bounded_window_only": true,
                    "source_payloads_durable_before_eviction": true,
                    "close_handles_and_clear_cache_before_eviction_receipt": true,
                    "separate_native_four_vector_phase_required": true,
                },
            })),
        )
    }

    fn source_lease() -> SealedDocument {
        fixture_document(
            "/fixtures/source-lease.json",
            sealed(json!({
                "schema": SOURCE_LEASE_SCHEMA,
                "status": SOURCE_LEASE_STATUS,
                "one_shot_lifecycle": {
                    "fresh_for_this_exact_launch": true,
                    "prior_terminal_receipt": null,
                    "automatic_retry_allowed": false,
                    "new_capture_root": true,
                    "existing_output_reuse_forbidden": true,
                    "replay_or_relaunch_forbidden": true,
                    "exact_launch_nonce": hash("launch-nonce"),
                },
                "fresh_pre_child_safety": {
                    "observed_immediately_before_child": true,
                    "exclusive_clean_window": true,
                    "no_source_or_native_model_body_resident_before_child": true,
                    "swap_used_bytes": 0,
                    "swapouts_pages_delta": 0,
                    "reclaimable_bytes": 2_000_000,
                    "minimum_reclaimable_bytes_required": 1_000_000,
                },
            })),
        )
    }

    fn valid_bundle() -> AuthorityBundle {
        let bridge = dual_bridge();
        let runtime = runtime_admission(&bridge.seal_sha256);
        validate_authority_bundle(runtime, bridge, source_lease()).expect("valid fixture bundle")
    }

    fn write_document(path: &Path, document: &SealedDocument) {
        fs::write(
            path,
            serde_json::to_vec(&document.value).expect("serialize fixture"),
        )
        .expect("write fixture document");
    }

    #[test]
    fn preflight_binds_all_three_authorities_and_matches_outer_command_grammar() {
        let output = prepared_interface(&valid_bundle()).expect("prepared interface");
        assert_eq!(output["schema"], INTERFACE_SCHEMA);
        assert_eq!(output["status"], PREPARED_STATUS);
        assert_eq!(output["execution_authorized"], false);
        assert_eq!(
            output["outer_runner_source_child_command"],
            json!([
                "ascension_qwen30_streamed_source_teacher_child",
                "--source-root",
                "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
                "--runtime-admission",
                "ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON",
                "--dual-attestation-runtime-admission",
                "ABSOLUTE_SEALED_DUAL_BRIDGE_JSON",
                "--source-lease",
                "ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON",
                "--capture-dir",
                "NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY",
            ])
        );
        assert_eq!(
            output["future_source_teacher_shape"]["source_layer_traversals"],
            SOURCE_LAYERS * SOURCE_FORWARDS
        );
        assert_eq!(
            output["execution_shape"]["source_operator_order"],
            json!(SOURCE_OPERATOR_ORDER)
        );
        assert_eq!(
            output["worker_output_grammar"]
                ["must_close_all_source_handles_and_zero_reader_cache_before_child_exit"],
            true
        );
        assert_eq!(
            output["exact_outer_handoff_fields"]["outer_must_reap_source_child_before_terminal"],
            true
        );
        verify_seal(&output, "prepared interface").expect("prepared output seals");
    }

    #[test]
    fn source_teacher_mode_checks_authorities_before_any_source_root_or_capture_access() {
        let temporary = TempDir::new().expect("temporary directory");
        let bridge = dual_bridge();
        let runtime = runtime_admission(&bridge.seal_sha256);
        let lease = source_lease();
        let runtime_path = temporary.path().join("runtime.json");
        let bridge_path = temporary.path().join("bridge.json");
        let lease_path = temporary.path().join("lease.json");
        write_document(&runtime_path, &runtime);
        write_document(&bridge_path, &bridge);
        write_document(&lease_path, &lease);
        let args = parse_args_from(vec![
            "--source-root".to_owned(),
            "/definitely-not-a-real-qwen30-source-root".to_owned(),
            "--runtime-admission".to_owned(),
            runtime_path.to_string_lossy().into_owned(),
            "--dual-attestation-runtime-admission".to_owned(),
            bridge_path.to_string_lossy().into_owned(),
            "--source-lease".to_owned(),
            lease_path.to_string_lossy().into_owned(),
            "--capture-dir".to_owned(),
            "/definitely-not-a-real-qwen30-capture-dir".to_owned(),
        ])
        .expect("outer-compatible source-teacher args parse");
        let refusal = run(args).expect("CPU build returns a sealed refusal after authority checks");
        assert_eq!(refusal["status"], CPU_BUILD_REFUSED_STATUS);
        assert_eq!(
            refusal["execution_boundary"]["source_root_opened_or_statted"],
            false
        );
        assert_eq!(
            refusal["execution_boundary"]["capture_directory_created"],
            false
        );
        verify_seal(&refusal, "CPU build refusal").expect("refusal seals");
    }

    #[test]
    fn missing_or_mismatched_authority_refuses_before_source_teacher_branch() {
        let bridge = dual_bridge();
        let mut runtime = runtime_admission(&bridge.seal_sha256);
        runtime.value["dual_bridge_seal_sha256"] = json!(hash("substituted-bridge"));
        runtime.value = seal_value(runtime.value).expect("re-seal substitution fixture");
        runtime.seal_sha256 = verify_seal(&runtime.value, "re-sealed runtime").expect("seal");
        let error = validate_authority_bundle(runtime, bridge, source_lease())
            .expect_err("mismatched bridge must refuse before source root processing");
        assert!(error.contains("not bound"));
    }

    #[test]
    fn synthetic_backend_uses_one_bounded_window_writes_two_f32le_vectors_and_receipt_last() {
        let temporary = TempDir::new().expect("temporary synthetic fixture root");
        let capture_dir = temporary.path().join("source-teacher-fixture");
        let evidence = run_synthetic_fixture_backend(&capture_dir)
            .expect("synthetic source-teacher backend runs without a source root");
        assert_eq!(evidence["status"], SYNTHETIC_EVIDENCE_STATUS);
        assert_eq!(
            evidence["traversal"]["observed_layer_traversals"],
            SOURCE_LAYERS * SOURCE_FORWARDS
        );
        assert!(
            evidence["bounded_per_read_cache"]["maximum_observed_window_bytes"]
                .as_u64()
                .expect("window measurement")
                <= MAX_POSITIONED_READ_BYTES as u64
        );
        assert_eq!(
            evidence["receipt_written_last_after_synthetic_payload_durability"],
            true
        );
        assert_eq!(evidence["source_handles_closed"], true);
        assert_eq!(evidence["streamed_reader_cache_zeroed"], true);
        assert_eq!(
            evidence["traversal"]["operator_hook_counts"]["embedding"],
            SOURCE_FORWARDS
        );
        assert_eq!(
            evidence["traversal"]["operator_hook_counts"]["lm_head_serial_k"],
            SOURCE_LAYERS * SOURCE_FORWARDS
        );
        assert_eq!(
            evidence["traversal"]["operator_hook_counts"]["one_selected_expert_body_at_a_time"],
            SOURCE_LAYERS * SOURCE_FORWARDS * SOURCE_TOP_K
        );
        assert_eq!(
            evidence["artifact_write_order"],
            json!([
                SOURCE_PAYLOADS[0],
                SOURCE_PAYLOADS[1],
                "fixture_source_close_and_reader_cache_zero",
                SYNTHETIC_SOURCE_RECEIPT_NAME,
            ])
        );
        for payload in SOURCE_PAYLOADS {
            let bytes =
                fs::read(capture_dir.join(payload)).expect("synthetic F32LE payload exists");
            assert_eq!(bytes.len(), F32_VECTOR_BYTES);
            assert!(f32::from_le_bytes(bytes[0..4].try_into().expect("one F32LE")).is_finite());
        }
        assert!(capture_dir.join(SYNTHETIC_SOURCE_RECEIPT_NAME).is_file());
        verify_seal(&evidence, "synthetic evidence").expect("synthetic evidence seals");

        let mut reader = SyntheticBoundedReader::new(MAX_POSITIONED_READ_BYTES).expect("reader");
        assert!(reader
            .positioned_read(&[0; 64], 0, MAX_POSITIONED_READ_BYTES + 1)
            .is_err());
    }

    #[test]
    fn phase_accurate_real_commit_refusal_never_claims_a_hypothetical_commit() {
        for (phase, expected_boundary) in [
            (
                RealSourceCommitPhase::HypotheticalFirstPayloadFsynced,
                "hypothetical_first_f32le_payload_fsynced_before_second_payload",
            ),
            (
                RealSourceCommitPhase::HypotheticalTwoPayloadsFsynced,
                "hypothetical_two_f32le_payloads_fsynced_before_source_close",
            ),
            (
                RealSourceCommitPhase::HypotheticalSourceClosedAndCacheZeroed,
                "hypothetical_source_closed_and_cache_zeroed_before_receipt_last",
            ),
        ] {
            let refusal = cpu_build_refusal_at_phase(
                &valid_bundle(),
                Path::new("/not-a-real-qwen30-source-root"),
                Path::new("/not-a-real-qwen30-capture-dir"),
                phase,
            )
            .expect("phase refusal seals");
            assert_eq!(
                refusal["phase_accurate_refusal"]["actual_phase"],
                "sealed_authorities_validated_before_source_root_open"
            );
            assert_eq!(
                refusal["phase_accurate_refusal"]["hypothetical_commit_boundary"],
                expected_boundary
            );
            assert_eq!(
                refusal["phase_accurate_refusal"]["actual_source_commit_performed"],
                false
            );
            verify_seal(&refusal, "phase refusal").expect("phase refusal is sealed");
        }
    }

    #[test]
    fn parser_keeps_source_root_out_of_preflight_and_out_of_outer_command_extension() {
        let bad = parse_args_from(vec![
            "--mode".to_owned(),
            "preflight".to_owned(),
            "--runtime-admission".to_owned(),
            "/tmp/runtime.json".to_owned(),
            "--dual-attestation-runtime-admission".to_owned(),
            "/tmp/bridge.json".to_owned(),
            "--source-lease".to_owned(),
            "/tmp/lease.json".to_owned(),
            "--source-root".to_owned(),
            "/source".to_owned(),
            "--out".to_owned(),
            "/tmp/out.json".to_owned(),
        ]);
        assert!(bad.is_err());
    }
}
