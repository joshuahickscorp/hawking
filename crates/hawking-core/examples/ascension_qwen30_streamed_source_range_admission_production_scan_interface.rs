#![allow(dead_code)]

//! Strict CPU/build-only interface for a future *production* Q30 hash scan.
//!
//! The synthetic bootstrap backend has a deliberately different namespace.  It
//! is useful for reader mechanics, but it is never production evidence.  This
//! file is the independent, non-fixture admission boundary for a later real
//! 16-shard / 18,867-range hash-map scan.  The production-scan branch contains
//! the bounded backend, but its only exercised invocation is a synthetic-file
//! test.  It cannot touch a source root until all sealed authorities and the
//! fresh lease have validated and a create-new replay reservation exists.
//!
//! A hash-map scan is not a source-teacher execution.  It may later earn a
//! production flat map plus a hash-coverage/capture chain, but it must not
//! manufacture the source-teacher operator/reader attestations or the
//! source-teacher runtime-admission receipt.  Those remain a separately
//! leased, later source-teacher responsibility.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

const INTERFACE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_interface.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_INTERFACE_NOT_EXECUTED";

const RANGE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1";
const RANGE_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED";
const SEMANTICS_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1";
const SEMANTICS_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED";
const RUNTIME_PRODUCER_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_runtime_range_admission_producer_preflight.v1";
const RUNTIME_PRODUCER_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_RUNTIME_RANGE_ADMISSION_PRODUCER_NOT_EXECUTED";

const PRODUCTION_SCAN_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_production_scan_authority.v1";
const PRODUCTION_SCAN_AUTHORITY_STATUS: &str =
    "ADMITTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_ONE_SHOT";
const BOOTSTRAP_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_quiet_lease.v1";
const BOOTSTRAP_LEASE_STATUS: &str =
    "GRANTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_HASH_SCAN_ONE_SHOT";

const FLAT_MAP_SCHEMA: &str = "hawking.ascension.qwen30_source_bf16_range_map.v1";
const HASH_COVERAGE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_production_hash_coverage_attestation.v1";
const HASH_COVERAGE_STATUS: &str =
    "EARNED_QWEN30_STREAMED_SOURCE_PRODUCTION_HASH_COVERAGE_ATTESTED_NOT_SOURCE_TEACHER";
const PRODUCTION_CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_production_capture.v1";
const PRODUCTION_CAPTURE_STATUS: &str =
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_NOT_SOURCE_TEACHER";
const REPLAY_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_production_replay_reservation.v1";
const REPLAY_STATUS: &str =
    "RESERVED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_PRODUCTION_HASH_SCAN_ONE_SHOT_NOT_SPAWNED";

const OPERATOR_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1";
const OPERATOR_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED";
const READER_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1";
const READER_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED";
const SOURCE_TEACHER_RUNTIME_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1";
const SOURCE_TEACHER_RUNTIME_STATUS: &str =
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY";

const MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const SHARDS: u64 = 16;
const TENSORS: u64 = 18_867;
const MAX_POSITIONED_READ_BYTES: u64 = 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const BF16_BYTES: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Preflight,
    ProductionScan,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "preflight" => Ok(Self::Preflight),
            "production-scan" => Ok(Self::ProductionScan),
            _ => Err("--mode must be preflight or production-scan".to_owned()),
        }
    }
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    range_authority: PathBuf,
    semantics_attester: PathBuf,
    runtime_admission_authority: PathBuf,
    interface_authority: Option<PathBuf>,
    production_scan_authority: Option<PathBuf>,
    bootstrap_lease: Option<PathBuf>,
    source_root: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    out: PathBuf,
}

fn usage() -> &'static str {
    "preflight:\n  ascension_qwen30_streamed_source_range_admission_production_scan_interface \\\n+    --mode preflight --range-authority ABS_JSON --semantics-attester ABS_JSON \\\n+    --runtime-admission-authority ABS_SEALED_JSON --out NEW_ABS_JSON\n\
future production scan (only future sealed real authority+lease may reach bounded backend):\n  \\
  ascension_qwen30_streamed_source_range_admission_production_scan_interface \\\n+    --mode production-scan --range-authority ABS_JSON --semantics-attester ABS_JSON \\\n+    --runtime-admission-authority ABS_SEALED_JSON --interface-authority ABS_SEALED_JSON \\\n+    --production-scan-authority ABS_SEALED_JSON --bootstrap-lease ABS_SEALED_JSON \\\n+    --source-root ABS_CANONICAL_QWEN30_SOURCE_ROOT --capture-dir NEW_ABS_DIR \\\n+    --out NEW_ABS_REFUSAL_JSON"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::new();
    let mut iterator = arguments.into_iter();
    while let Some(flag) = iterator.next() {
        if matches!(flag.as_str(), "--help" | "-h") {
            return Err(usage().to_owned());
        }
        let value = iterator
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        if !matches!(
            flag.as_str(),
            "--mode"
                | "--range-authority"
                | "--semantics-attester"
                | "--runtime-admission-authority"
                | "--interface-authority"
                | "--production-scan-authority"
                | "--bootstrap-lease"
                | "--source-root"
                | "--capture-dir"
                | "--out"
        ) {
            return Err(format!("unsupported option {flag:?}; {}", usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("duplicate {flag}; {}", usage()));
        }
    }
    let required = |value: Option<String>, flag: &str| {
        value
            .map(PathBuf::from)
            .ok_or_else(|| format!("{flag} is required; {}", usage()))
    };
    let mode = values
        .remove("--mode")
        .as_deref()
        .map(Mode::parse)
        .transpose()?
        .unwrap_or(Mode::Preflight);
    let args = Args {
        mode,
        range_authority: required(values.remove("--range-authority"), "--range-authority")?,
        semantics_attester: required(
            values.remove("--semantics-attester"),
            "--semantics-attester",
        )?,
        runtime_admission_authority: required(
            values.remove("--runtime-admission-authority"),
            "--runtime-admission-authority",
        )?,
        interface_authority: values.remove("--interface-authority").map(PathBuf::from),
        production_scan_authority: values
            .remove("--production-scan-authority")
            .map(PathBuf::from),
        bootstrap_lease: values.remove("--bootstrap-lease").map(PathBuf::from),
        source_root: values.remove("--source-root").map(PathBuf::from),
        capture_dir: values.remove("--capture-dir").map(PathBuf::from),
        out: required(values.remove("--out"), "--out")?,
    };
    if !values.is_empty() {
        return Err(format!("unexpected arguments; {}", usage()));
    }
    for (flag, path) in [
        ("--range-authority", &args.range_authority),
        ("--semantics-attester", &args.semantics_attester),
        (
            "--runtime-admission-authority",
            &args.runtime_admission_authority,
        ),
        ("--out", &args.out),
    ] {
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
    }
    match args.mode {
        Mode::Preflight => {
            if args.interface_authority.is_some()
                || args.production_scan_authority.is_some()
                || args.bootstrap_lease.is_some()
                || args.source_root.is_some()
                || args.capture_dir.is_some()
            {
                return Err(
                    "preflight accepts only the three authority inputs and --out".to_owned(),
                );
            }
        }
        Mode::ProductionScan => {
            for (flag, path) in [
                ("--interface-authority", args.interface_authority.as_ref()),
                (
                    "--production-scan-authority",
                    args.production_scan_authority.as_ref(),
                ),
                ("--bootstrap-lease", args.bootstrap_lease.as_ref()),
                ("--source-root", args.source_root.as_ref()),
                ("--capture-dir", args.capture_dir.as_ref()),
            ] {
                let path = path.ok_or_else(|| format!("{flag} is required; {}", usage()))?;
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

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|error| format!("cannot canonicalize string: {error}")),
        Value::Array(values) => values
            .iter()
            .map(canonical)
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| format!("[{}]", parts.join(","))),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut parts = Vec::with_capacity(keys.len());
            for key in keys {
                let rendered = serde_json::to_string(key)
                    .map_err(|error| format!("cannot canonicalize key: {error}"))?;
                parts.push(format!(
                    "{rendered}:{}",
                    canonical(values.get(key).expect("canonical key remains present"))?
                ));
            }
            Ok(format!("{{{}}}", parts.join(",")))
        }
    }
}

fn seal(mut value: Value) -> Result<Value, String> {
    value
        .as_object_mut()
        .ok_or_else(|| "sealed value must be an object".to_owned())?
        .remove("seal_sha256");
    let seal_sha256 = sha256(canonical(&value)?.as_bytes());
    value
        .as_object_mut()
        .expect("sealed object remains an object")
        .insert("seal_sha256".to_owned(), Value::String(seal_sha256));
    Ok(value)
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn required<'a>(
    values: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Value, String> {
    values
        .get(key)
        .ok_or_else(|| format!("{label}.{key} is required"))
}

fn object_field<'a>(
    values: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object(required(values, key, label)?, &format!("{label}.{key}"))
}

fn array_field<'a>(
    values: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    required(values, key, label)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label}.{key} must be an array"))
}

fn text<'a>(values: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str, String> {
    required(values, key, label)?
        .as_str()
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{key} must be non-empty text"))
}

fn sha_field(values: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
    let value = text(values, key, label)?;
    if !valid_sha(value) {
        return Err(format!("{label}.{key} must be a lowercase SHA-256"));
    }
    Ok(value.to_owned())
}

fn u64_field(values: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    required(values, key, label)?
        .as_u64()
        .ok_or_else(|| format!("{label}.{key} must be an unsigned integer"))
}

fn exact_bool(
    values: &Map<String, Value>,
    key: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if required(values, key, label)?.as_bool() != Some(expected) {
        return Err(format!("{label}.{key} must be {expected}"));
    }
    Ok(())
}

fn schema_status(
    values: &Map<String, Value>,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if text(values, "schema", label)? != schema || text(values, "status", label)? != status {
        return Err(format!("{label} schema/status drifted"));
    }
    Ok(())
}

fn checked_relative_path(value: &str, label: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("{label} must be a non-empty relative path"));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} may not contain traversal components"));
    }
    Ok(())
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let values = object(value, label)?;
    let actual = sha_field(values, "seal_sha256", label)?;
    let mut unsigned = values.clone();
    unsigned.remove("seal_sha256");
    if sha256(canonical(&Value::Object(unsigned))?.as_bytes()) != actual {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(actual)
}

#[derive(Clone, Debug)]
struct Document {
    path: PathBuf,
    value: Value,
    raw_document_sha256: String,
    canonical_document_sha256: String,
    seal_sha256: Option<String>,
}

fn read_document(path: &Path, label: &str, require_seal: bool) -> Result<Document, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() == 0 || metadata.len() > MAX_METADATA_BYTES {
        return Err(format!(
            "{label} must contain 1..={MAX_METADATA_BYTES} bytes"
        ));
    }
    let bytes = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{label} changed while read"));
    }
    let value = serde_json::from_slice(&bytes).map_err(|error| format!("{label} JSON: {error}"))?;
    object(&value, label)?;
    let seal_sha256 = if object(&value, label)?.contains_key("seal_sha256") {
        Some(verify_seal(&value, label)?)
    } else {
        None
    };
    if require_seal && seal_sha256.is_none() {
        return Err(format!("{label} must be sealed"));
    }
    Ok(Document {
        path: fs::canonicalize(path)
            .map_err(|error| format!("cannot canonicalize {label}: {error}"))?,
        raw_document_sha256: sha256(&bytes),
        canonical_document_sha256: sha256(canonical(&value)?.as_bytes()),
        value,
        seal_sha256,
    })
}

fn fixture_identity_error(value: &Value, label: &str) -> Result<(), String> {
    match value {
        Value::Array(values) => {
            for (index, child) in values.iter().enumerate() {
                fixture_identity_error(child, &format!("{label}[{index}]"))?;
            }
        }
        Value::Object(values) => {
            for (key, child) in values {
                if matches!(key.as_str(), "schema" | "status") {
                    if let Some(identity) = child.as_str() {
                        let lower = identity.to_ascii_lowercase();
                        if lower.contains("fixture") || lower.contains("synthetic") {
                            return Err(format!(
                                "{label}.{key} carries forbidden fixture-only identity {identity:?}"
                            ));
                        }
                    }
                }
                if matches!(
                    key.as_str(),
                    "fixture_only" | "synthetic_fixture_only" | "production_adapter_forbidden"
                ) && child.as_bool() == Some(true)
                {
                    return Err(format!(
                        "{label}.{key} marks fixture-only or production-forbidden evidence"
                    ));
                }
                fixture_identity_error(child, &format!("{label}.{key}"))?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn evidence(document: &Document) -> Value {
    json!({
        "path": document.path,
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": document.canonical_document_sha256,
        "seal_sha256": document.seal_sha256,
    })
}

fn require_pointer(
    value: &Value,
    expected: &Document,
    label: &str,
    require_seal: bool,
) -> Result<(), String> {
    let values = object(value, label)?;
    if sha_field(values, "raw_document_sha256", label)? != expected.raw_document_sha256 {
        return Err(format!("{label} does not bind the supplied document"));
    }
    // The established metadata range and semantics authorities predate this
    // interface and truthfully carry only raw-document SHA-256 references.
    // Keep raw identity mandatory for every input; require canonical identity
    // for sealed inputs, while accepting an absent canonical field only for
    // those legacy unsealed authorities.  If supplied, it must still match.
    match values.get("canonical_document_sha256") {
        Some(Value::String(value)) if valid_sha(value) => {
            if value != &expected.canonical_document_sha256 {
                return Err(format!("{label} canonical binding drifted"));
            }
        }
        Some(_) => {
            return Err(format!(
                "{label}.canonical_document_sha256 must be a SHA-256"
            ))
        }
        None if expected.seal_sha256.is_some() => {
            return Err(format!(
                "{label} must bind canonical identity for a sealed input"
            ));
        }
        None => {}
    }
    match (require_seal, &expected.seal_sha256) {
        (true, Some(expected_seal)) => {
            if sha_field(values, "seal_sha256", label)? != *expected_seal {
                return Err(format!("{label} seal binding drifted"));
            }
        }
        (true, None) => return Err(format!("{label} expected sealed input")),
        (false, _) => {}
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct MetadataBinding {
    document: Document,
    authority_content_sha256: String,
    source_revision: String,
    source_index_relative_path: String,
    source_index_sha256: String,
    maximum_window_bytes: u64,
    shards: Vec<ShardRange>,
    tensors: Vec<TensorRange>,
}

#[derive(Clone, Debug)]
struct ShardRange {
    relative_path: String,
    file_bytes: u64,
    safetensors_header_bytes: u64,
    safetensors_header_sha256: String,
    safetensors_prefix_sha256: String,
}

#[derive(Clone, Debug)]
struct TensorRange {
    tensor_name: String,
    shard_relative_path: String,
    full_shape: Vec<u64>,
    absolute_data_offset: u64,
    data_bytes: u64,
}

fn shape_bytes(value: &Value, label: &str) -> Result<u64, String> {
    let dimensions = value
        .as_array()
        .ok_or_else(|| format!("{label} must be an array"))?;
    if dimensions.is_empty() {
        return Err(format!("{label} may not be empty"));
    }
    dimensions.iter().try_fold(1_u64, |elements, dimension| {
        let dimension = dimension
            .as_u64()
            .filter(|dimension| *dimension > 0)
            .ok_or_else(|| format!("{label} dimensions must be positive"))?;
        elements
            .checked_mul(dimension)
            .ok_or_else(|| format!("{label} overflow"))
    })
}

fn validate_metadata(document: Document) -> Result<MetadataBinding, String> {
    fixture_identity_error(&document.value, "metadata range authority")?;
    let envelope = object(&document.value, "metadata range authority envelope")?;
    let authority_value = required(envelope, "authority", "metadata range authority envelope")?;
    let authority = object(authority_value, "metadata range authority")?;
    let authority_content_sha256 = sha_field(
        envelope,
        "authority_content_sha256",
        "metadata range authority envelope",
    )?;
    if authority_content_sha256 != sha256(canonical(authority_value)?.as_bytes()) {
        return Err("metadata authority content hash does not bind authority".to_owned());
    }
    schema_status(
        authority,
        RANGE_SCHEMA,
        RANGE_STATUS,
        "metadata range authority",
    )?;
    let source = object_field(authority, "source", "metadata range authority")?;
    if text(source, "model_id", "metadata range source")? != MODEL_ID
        || u64_field(source, "source_shard_count", "metadata range source")? != SHARDS
        || u64_field(source, "source_tensor_count", "metadata range source")? != TENSORS
    {
        return Err("metadata range source model/geometry drifted".to_owned());
    }
    let source_revision = text(source, "source_revision", "metadata range source")?.to_owned();
    if !valid_lower_hex(&source_revision, 40) {
        return Err(
            "metadata range source revision must be 40 lowercase hexadecimal characters".to_owned(),
        );
    }
    let index = object_field(source, "source_index", "metadata range source")?;
    let source_index_sha256 = sha_field(index, "sha256", "metadata range source index")?;
    let source_index_relative_path =
        text(index, "relative_path", "metadata range source index")?.to_owned();
    checked_relative_path(
        &source_index_relative_path,
        "metadata range source index path",
    )?;
    if text(index, "format", "metadata range source index")? != "huggingface.safetensors.index.json"
    {
        return Err("metadata range source index format drifted".to_owned());
    }
    if u64_field(
        index,
        "weight_map_tensor_count",
        "metadata range source index",
    )? != TENSORS
    {
        return Err("metadata range index tensor count drifted".to_owned());
    }
    let scope = object_field(
        authority,
        "exact_streamed_oracle_scope",
        "metadata range authority",
    )?;
    for (key, expected) in [
        ("source_template_token_count", 369),
        ("forced_identical_continuation_token_id", 949),
        ("total_forwards_per_replay_arm", 370),
        ("layers", 48),
        ("top_k_routes_per_token", 8),
    ] {
        if u64_field(scope, key, "metadata range exact scope")? != expected {
            return Err(format!("metadata range exact scope {key} drifted"));
        }
    }
    exact_bool(
        scope,
        "sampling_or_autoregressive_feedback_forbidden",
        true,
        "metadata range exact scope",
    )?;
    let boundary = object_field(
        authority,
        "metadata_access_boundary",
        "metadata range authority",
    )?;
    if u64_field(
        boundary,
        "source_tensor_payload_bytes_read",
        "metadata range boundary",
    )? != 0
    {
        return Err("metadata authority must not have opened source payloads".to_owned());
    }
    for key in [
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
        "mmap_or_memory_map_used",
        "source_model_instantiated",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
    ] {
        exact_bool(boundary, key, false, "metadata range boundary")?;
    }
    let mut shard_names = BTreeSet::new();
    let mut shard_lengths = BTreeMap::new();
    let mut shards = Vec::with_capacity(SHARDS as usize);
    for shard in array_field(authority, "shards", "metadata range authority")? {
        let shard = object(shard, "metadata range shard")?;
        let path = text(shard, "relative_path", "metadata range shard")?;
        checked_relative_path(path, "metadata range shard path")?;
        let file_bytes = u64_field(shard, "file_bytes", "metadata range shard")?;
        if file_bytes == 0 || !shard_names.insert(path.to_owned()) {
            return Err("metadata range shard set is invalid".to_owned());
        }
        shard_lengths.insert(path.to_owned(), file_bytes);
        let safetensors_header_bytes =
            u64_field(shard, "safetensors_header_bytes", "metadata range shard")?;
        if safetensors_header_bytes == 0
            || safetensors_header_bytes > MAX_METADATA_BYTES
            || 8_u64
                .checked_add(safetensors_header_bytes)
                .ok_or_else(|| "metadata range header length overflow".to_owned())?
                > file_bytes
        {
            return Err("metadata range safetensors header scope is invalid".to_owned());
        }
        shards.push(ShardRange {
            relative_path: path.to_owned(),
            file_bytes,
            safetensors_header_bytes,
            safetensors_header_sha256: sha_field(
                shard,
                "safetensors_header_sha256",
                "metadata range shard",
            )?,
            safetensors_prefix_sha256: sha_field(
                shard,
                "safetensors_prefix_sha256",
                "metadata range shard",
            )?,
        });
    }
    if shard_names.len() as u64 != SHARDS {
        return Err("metadata range shard array geometry drifted".to_owned());
    }
    let mut tensor_names = BTreeSet::new();
    let mut tensors = Vec::with_capacity(TENSORS as usize);
    let mut intervals_by_shard = BTreeMap::<String, Vec<(u64, u64, String)>>::new();
    let mut maximum_window_bytes = 0_u64;
    for tensor in array_field(authority, "tensors", "metadata range authority")? {
        let tensor = object(tensor, "metadata range tensor")?;
        if text(tensor, "source_dtype", "metadata range tensor")? != "BF16" {
            return Err("metadata range tensor dtype drifted".to_owned());
        }
        let name = text(tensor, "tensor_name", "metadata range tensor")?;
        if !tensor_names.insert(name.to_owned()) {
            return Err("metadata range tensor set duplicates names".to_owned());
        }
        let shard = text(tensor, "shard_relative_path", "metadata range tensor")?;
        if !shard_names.contains(shard) {
            return Err("metadata range tensor references undeclared shard".to_owned());
        }
        let window_bytes = shape_bytes(
            required(tensor, "row_window_shape", "metadata range tensor")?,
            "metadata range tensor row window",
        )?
        .checked_mul(BF16_BYTES)
        .ok_or_else(|| "metadata range tensor window overflows".to_owned())?;
        if window_bytes == 0 || window_bytes > MAX_POSITIONED_READ_BYTES {
            return Err("metadata range tensor exceeds <=1MiB reader ceiling".to_owned());
        }
        let full_shape = required(tensor, "full_shape", "metadata range tensor")?
            .as_array()
            .ok_or_else(|| "metadata range tensor full shape must be an array".to_owned())?
            .iter()
            .map(|dimension| {
                dimension
                    .as_u64()
                    .filter(|dimension| *dimension > 0)
                    .ok_or_else(|| {
                        "metadata range tensor full shape dimensions must be positive".to_owned()
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let _ = shape_bytes(
            required(tensor, "full_shape", "metadata range tensor")?,
            "metadata range tensor full shape",
        )?;
        let data_bytes = u64_field(tensor, "data_bytes", "metadata range tensor")?;
        let expected_data_bytes = shape_bytes(
            required(tensor, "full_shape", "metadata range tensor")?,
            "metadata range tensor full shape",
        )?
        .checked_mul(BF16_BYTES)
        .ok_or_else(|| "metadata range tensor byte count overflows".to_owned())?;
        if data_bytes == 0 || window_bytes > data_bytes || data_bytes != expected_data_bytes {
            return Err("metadata range tensor byte/window scope drifted".to_owned());
        }
        let absolute_data_offset =
            u64_field(tensor, "absolute_data_offset", "metadata range tensor")?;
        let end = absolute_data_offset
            .checked_add(data_bytes)
            .ok_or_else(|| "metadata range tensor range overflows".to_owned())?;
        if end
            > *shard_lengths
                .get(shard)
                .ok_or_else(|| "metadata range tensor shard disappeared".to_owned())?
        {
            return Err("metadata range tensor range exceeds its shard".to_owned());
        }
        intervals_by_shard
            .entry(shard.to_owned())
            .or_default()
            .push((absolute_data_offset, end, name.to_owned()));
        tensors.push(TensorRange {
            tensor_name: name.to_owned(),
            shard_relative_path: shard.to_owned(),
            full_shape,
            absolute_data_offset,
            data_bytes,
        });
        maximum_window_bytes = maximum_window_bytes.max(window_bytes);
    }
    if tensor_names.len() as u64 != TENSORS || maximum_window_bytes == 0 {
        return Err("metadata range tensor array geometry drifted".to_owned());
    }
    for (shard, intervals) in &mut intervals_by_shard {
        intervals.sort_by_key(|(start, _, _)| *start);
        let mut previous_end = 0_u64;
        for (start, end, tensor_name) in intervals {
            if *start < previous_end {
                return Err(format!(
                    "metadata range tensor {tensor_name} overlaps a prior range in shard {shard}"
                ));
            }
            previous_end = *end;
        }
    }
    Ok(MetadataBinding {
        document,
        authority_content_sha256,
        source_revision,
        source_index_relative_path,
        source_index_sha256,
        maximum_window_bytes,
        shards,
        tensors,
    })
}

#[derive(Clone, Debug)]
struct SemanticsBinding {
    document: Document,
}

fn validate_semantics(
    document: Document,
    metadata: &MetadataBinding,
) -> Result<SemanticsBinding, String> {
    fixture_identity_error(&document.value, "semantics attester")?;
    let root = object(&document.value, "semantics attester")?;
    schema_status(
        root,
        SEMANTICS_SCHEMA,
        SEMANTICS_STATUS,
        "semantics attester",
    )?;
    let boundary = object_field(root, "execution_boundary", "semantics attester")?;
    for key in [
        "source_tensor_payload_opened",
        "source_safetensors_or_other_weight_path_accepted",
        "source_model_instantiated",
        "source_inference_executed",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
    ] {
        exact_bool(boundary, key, false, "semantics boundary")?;
    }
    let source = object_field(root, "pinned_source_binding", "semantics attester")?;
    if text(source, "source_model_id", "semantics source")? != MODEL_ID
        || text(source, "source_revision", "semantics source")? != metadata.source_revision
        || sha_field(source, "source_index_sha256", "semantics source")?
            != metadata.source_index_sha256
    {
        return Err("semantics source identity drifted from metadata".to_owned());
    }
    let consumed = object_field(root, "consumed_metadata_contracts", "semantics attester")?;
    let range = object_field(consumed, "range_authority", "semantics consumed contracts")?;
    if sha_field(range, "document_sha256", "semantics range pointer")?
        != metadata.document.raw_document_sha256
        || sha_field(range, "authority_content_sha256", "semantics range pointer")?
            != metadata.authority_content_sha256
    {
        return Err("semantics range binding drifted".to_owned());
    }
    let future = object_field(
        root,
        "future_exact_execution_attestation",
        "semantics attester",
    )?;
    if text(future, "schema", "semantics future execution")? != OPERATOR_ATTESTATION_SCHEMA
        || text(
            future,
            "status_only_after_real_separately_leased_source_execution",
            "semantics future execution",
        )? != OPERATOR_ATTESTATION_STATUS
    {
        return Err("semantics future execution grammar drifted".to_owned());
    }
    Ok(SemanticsBinding { document })
}

fn validate_runtime_authority(
    document: Document,
    metadata: &MetadataBinding,
    semantics: &SemanticsBinding,
) -> Result<Document, String> {
    fixture_identity_error(&document.value, "runtime-admission authority")?;
    let root = object(&document.value, "runtime-admission authority")?;
    schema_status(
        root,
        RUNTIME_PRODUCER_SCHEMA,
        RUNTIME_PRODUCER_STATUS,
        "runtime-admission authority",
    )?;
    exact_bool(root, "prepared", true, "runtime-admission authority")?;
    exact_bool(
        root,
        "runtime_admission_earned",
        false,
        "runtime-admission authority",
    )?;
    exact_bool(
        root,
        "source_payload_validation_executed",
        false,
        "runtime-admission authority",
    )?;
    let metadata_binding = object_field(
        root,
        "sealed_metadata_authority_binding",
        "runtime-admission authority",
    )?;
    let metadata_evidence = required(
        metadata_binding,
        "metadata_range_authority",
        "runtime-admission authority metadata",
    )?;
    require_pointer(
        metadata_evidence,
        &metadata.document,
        "runtime-admission authority metadata pointer",
        false,
    )?;
    if sha_field(
        metadata_binding,
        "authority_content_sha256",
        "runtime-admission authority metadata",
    )? != metadata.authority_content_sha256
    {
        return Err("runtime-admission authority content binding drifted".to_owned());
    }
    let semantics_binding = object_field(
        root,
        "metadata_semantics_binding",
        "runtime-admission authority",
    )?;
    require_pointer(
        required(
            semantics_binding,
            "operator_semantics_attester",
            "runtime-admission authority semantics",
        )?,
        &semantics.document,
        "runtime-admission authority semantics pointer",
        false,
    )?;
    let flat = object_field(
        root,
        "future_flat_runtime_range_map",
        "runtime-admission authority",
    )?;
    if text(flat, "schema", "runtime-admission authority flat map")? != FLAT_MAP_SCHEMA
        || u64_field(
            flat,
            "maximum_positioned_read_bytes",
            "runtime-admission authority flat map",
        )? != MAX_POSITIONED_READ_BYTES
    {
        return Err("runtime-admission authority flat-map grammar drifted".to_owned());
    }
    let runtime = object_field(
        root,
        "future_runtime_admission_receipt",
        "runtime-admission authority",
    )?;
    if text(
        runtime,
        "schema",
        "runtime-admission authority runtime receipt",
    )? != SOURCE_TEACHER_RUNTIME_SCHEMA
        || text(
            runtime,
            "status_only_after_bounded_source_validation",
            "runtime-admission authority runtime receipt",
        )? != SOURCE_TEACHER_RUNTIME_STATUS
    {
        return Err(
            "runtime-admission authority source-teacher receipt grammar drifted".to_owned(),
        );
    }
    Ok(document)
}

#[derive(Clone, Debug)]
struct Inputs {
    metadata: MetadataBinding,
    semantics: SemanticsBinding,
    runtime_authority: Document,
}

fn validate_inputs(args: &Args) -> Result<Inputs, String> {
    // This is deliberately all metadata-only.  In particular, no source-root
    // argument is even inspected by this function.
    let metadata = validate_metadata(read_document(
        &args.range_authority,
        "metadata range authority",
        false,
    )?)?;
    let semantics = validate_semantics(
        read_document(&args.semantics_attester, "semantics attester", false)?,
        &metadata,
    )?;
    let runtime_authority = validate_runtime_authority(
        read_document(
            &args.runtime_admission_authority,
            "runtime-admission authority",
            true,
        )?,
        &metadata,
        &semantics,
    )?;
    Ok(Inputs {
        metadata,
        semantics,
        runtime_authority,
    })
}

fn future_flat_map_grammar(maximum_window_bytes: u64) -> Value {
    json!({
        "schema": FLAT_MAP_SCHEMA,
        "schema_must_not_be_fixture_renamed_or_synthetic": true,
        "source_model_id": MODEL_ID,
        "source_revision": "metadata-bound-40-lowercase-hex",
        "source_tensor_count": TENSORS,
        "source_shards": SHARDS,
        "source_tensors": TENSORS,
        "source_index": {"relative_path": "metadata-bound", "sha256": "metadata-bound", "format": "huggingface.safetensors.index.json"},
        "maximum_window_bytes": maximum_window_bytes,
        "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
        "maximum_live_raw_bf16_windows": 1,
        "shard_fields": ["shard_id", "relative_path", "bytes", "sha256", "safetensors_header_sha256", "safetensors_prefix_sha256"],
        "tensor_fields": ["tensor_name", "shard_id", "dtype=BF16", "shape", "data_offset", "data_bytes", "raw_bf16_sha256"],
        "all_records_must_match_metadata_authority": true,
        "full_shard_or_model_residency_forbidden": true,
    })
}

fn interface_document(inputs: &Inputs) -> Result<Value, String> {
    seal(json!({
        "schema": INTERFACE_SCHEMA,
        "status": PREPARED_STATUS,
        "prepared": true,
        "execution_authorized": false,
        "production_hash_scan_earned": false,
        "input_authorities": {
            "metadata_range_authority": evidence(&inputs.metadata.document),
            "metadata_authority_content_sha256": inputs.metadata.authority_content_sha256,
            "independent_non_fixture_semantics_attester": evidence(&inputs.semantics.document),
            "runtime_admission_producer_authority": evidence(&inputs.runtime_authority),
        },
        "strict_non_fixture_boundary": {
            "before_source_root_access": true,
            "reject_schema_or_status_containing": ["fixture", "synthetic"],
            "reject_true_flags": ["fixture_only", "synthetic_fixture_only", "production_adapter_forbidden"],
            "renamed_or_resealed_fixture_evidence_is_not_accepted_without_full_non_fixture_provenance": true,
        },
        "future_production_scan_authority": {
            "schema": PRODUCTION_SCAN_AUTHORITY_SCHEMA,
            "status": PRODUCTION_SCAN_AUTHORITY_STATUS,
            "must_be_sealed_and_bind_this_interface_and_all_three_inputs": true,
            "fresh_one_shot_non_inference_only": true,
            "fixture_only": false,
            "synthetic_fixture_only": false,
            "production_adapter_forbidden": false,
        },
        "future_fresh_bootstrap_lease": {
            "schema": BOOTSTRAP_LEASE_SCHEMA,
            "status": BOOTSTRAP_LEASE_STATUS,
            "must_be_sealed_fresh_one_shot_and_bind_production_scan_authority": true,
            "no_source_teacher_model_gpu_server_hcli_or_tps_authority": true,
        },
        "future_bounded_hash_scan": {
            "source_root_open_allowed_only_after_all_authorities_validate": true,
            "regular_non_symlink_root_and_files_only": true,
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "source_shards": SHARDS,
            "source_tensors": TENSORS,
            "future_flat_runtime_range_map": future_flat_map_grammar(inputs.metadata.maximum_window_bytes),
            "future_hash_coverage_attestation": {
                "schema": HASH_COVERAGE_SCHEMA,
                "status": HASH_COVERAGE_STATUS,
                "binds_full_shard_and_every_declared_bf16_range_hash": true,
                "not_operator_execution_attestation": true,
            },
            "replay_reservation": {
                "schema": REPLAY_SCHEMA,
                "status": REPLAY_STATUS,
                "create_new_before_any_source_root_open": true,
                "one_child_maximum": true,
            },
            "receipt_last_capture": {
                "schema": PRODUCTION_CAPTURE_SCHEMA,
                "status": PRODUCTION_CAPTURE_STATUS,
                "must_bind_map_hash_coverage_authority_lease_and_replay": true,
                "written_only_after_reader_close_cache_zero_and_hash_coverage": true,
                "not_source_teacher": true,
            },
        },
        "separate_source_teacher_follow_on": {
            "required_later_not_emitted_by_hash_scan": [
                {"schema": OPERATOR_ATTESTATION_SCHEMA, "status": OPERATOR_ATTESTATION_STATUS},
                {"schema": READER_ATTESTATION_SCHEMA, "status": READER_ATTESTATION_STATUS},
                {"schema": SOURCE_TEACHER_RUNTIME_SCHEMA, "status": SOURCE_TEACHER_RUNTIME_STATUS},
            ],
            "must_use_a_distinct_source_teacher_lease": true,
            "hash_scan_must_not_claim_source_semantic_forward_or_logits": true,
        },
        "future_command": [
            "ascension_qwen30_streamed_source_range_admission_production_scan_interface", "--mode", "production-scan",
            "--range-authority", "ABSOLUTE_METADATA_RANGE_AUTHORITY_JSON",
            "--semantics-attester", "ABSOLUTE_NON_FIXTURE_SEMANTICS_ATTESTER_JSON",
            "--runtime-admission-authority", "ABSOLUTE_SEALED_RUNTIME_PRODUCER_AUTHORITY_JSON",
            "--interface-authority", "ABSOLUTE_SEALED_PRODUCTION_INTERFACE_JSON",
            "--production-scan-authority", "ABSOLUTE_SEALED_FRESH_PRODUCTION_SCAN_AUTHORITY_JSON",
            "--bootstrap-lease", "ABSOLUTE_SEALED_FRESH_BOOTSTRAP_LEASE_JSON",
            "--source-root", "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--capture-dir", "NEW_ABSOLUTE_CAPTURE_DIRECTORY",
            "--out", "NEW_ABSOLUTE_PRE_EXECUTION_OR_TERMINAL_RECEIPT_JSON",
        ],
        "execution_boundary": {
            "source_root_opened_or_statted": false,
            "source_payload_opened": false,
            "source_model_loaded": false,
            "source_teacher_or_logits_executed": false,
            "gpu_server_hcli_or_tps_action": false,
            "lease_issued_or_consumed": false,
            "child_started": false,
        },
        "claim_boundary": "Prepared non-fixture production hash-scan interface only. It neither opens a source root nor earns a production map, hash coverage, capture, source-teacher execution attestation, runtime admission, logits, model/GPU/server/HCLI/lease, TPS, or tournament result.",
    }))
}

fn validate_interface(document: Document, inputs: &Inputs) -> Result<Document, String> {
    fixture_identity_error(&document.value, "production scan interface")?;
    let root = object(&document.value, "production scan interface")?;
    schema_status(
        root,
        INTERFACE_SCHEMA,
        PREPARED_STATUS,
        "production scan interface",
    )?;
    exact_bool(root, "prepared", true, "production scan interface")?;
    exact_bool(
        root,
        "execution_authorized",
        false,
        "production scan interface",
    )?;
    let authorities = object_field(root, "input_authorities", "production scan interface")?;
    require_pointer(
        required(authorities, "metadata_range_authority", "interface inputs")?,
        &inputs.metadata.document,
        "interface metadata pointer",
        false,
    )?;
    if sha_field(
        authorities,
        "metadata_authority_content_sha256",
        "interface inputs",
    )? != inputs.metadata.authority_content_sha256
    {
        return Err("interface metadata content binding drifted".to_owned());
    }
    require_pointer(
        required(
            authorities,
            "independent_non_fixture_semantics_attester",
            "interface inputs",
        )?,
        &inputs.semantics.document,
        "interface semantics pointer",
        false,
    )?;
    require_pointer(
        required(
            authorities,
            "runtime_admission_producer_authority",
            "interface inputs",
        )?,
        &inputs.runtime_authority,
        "interface runtime authority pointer",
        true,
    )?;
    let future = object_field(
        root,
        "future_bounded_hash_scan",
        "production scan interface",
    )?;
    if u64_field(future, "source_shards", "interface bounded scan")? != SHARDS
        || u64_field(future, "source_tensors", "interface bounded scan")? != TENSORS
        || u64_field(
            future,
            "maximum_positioned_read_bytes",
            "interface bounded scan",
        )? != MAX_POSITIONED_READ_BYTES
    {
        return Err("interface production geometry drifted".to_owned());
    }
    Ok(document)
}

fn validate_production_scan_authority(
    document: Document,
    inputs: &Inputs,
    interface: &Document,
) -> Result<Document, String> {
    fixture_identity_error(&document.value, "production scan authority")?;
    let root = object(&document.value, "production scan authority")?;
    schema_status(
        root,
        PRODUCTION_SCAN_AUTHORITY_SCHEMA,
        PRODUCTION_SCAN_AUTHORITY_STATUS,
        "production scan authority",
    )?;
    for key in [
        "fresh_for_this_exact_scan",
        "one_shot",
        "non_inference_hash_scan_only",
        "source_root_open_only_after_all_authorities_validate",
    ] {
        exact_bool(root, key, true, "production scan authority")?;
    }
    for key in [
        "fixture_only",
        "synthetic_fixture_only",
        "production_adapter_forbidden",
        "source_teacher_or_logits_authorized",
        "model_gpu_server_hcli_or_tps_authorized",
        "lease_consumed",
    ] {
        exact_bool(root, key, false, "production scan authority")?;
    }
    let bindings = object_field(root, "immutable_bindings", "production scan authority")?;
    require_pointer(
        required(
            bindings,
            "interface_authority",
            "production authority bindings",
        )?,
        interface,
        "production authority interface pointer",
        true,
    )?;
    require_pointer(
        required(
            bindings,
            "metadata_range_authority",
            "production authority bindings",
        )?,
        &inputs.metadata.document,
        "production authority metadata pointer",
        false,
    )?;
    require_pointer(
        required(
            bindings,
            "independent_semantics_attester",
            "production authority bindings",
        )?,
        &inputs.semantics.document,
        "production authority semantics pointer",
        false,
    )?;
    require_pointer(
        required(
            bindings,
            "runtime_admission_producer_authority",
            "production authority bindings",
        )?,
        &inputs.runtime_authority,
        "production authority runtime pointer",
        true,
    )?;
    if sha_field(
        bindings,
        "metadata_authority_content_sha256",
        "production authority bindings",
    )? != inputs.metadata.authority_content_sha256
    {
        return Err("production authority metadata content binding drifted".to_owned());
    }
    let geometry = object_field(root, "geometry", "production scan authority")?;
    if u64_field(geometry, "source_shards", "production authority geometry")? != SHARDS
        || u64_field(geometry, "source_tensors", "production authority geometry")? != TENSORS
        || u64_field(
            geometry,
            "maximum_positioned_read_bytes",
            "production authority geometry",
        )? != MAX_POSITIONED_READ_BYTES
        || u64_field(
            geometry,
            "maximum_live_raw_bf16_windows",
            "production authority geometry",
        )? != 1
    {
        return Err("production authority geometry drifted".to_owned());
    }
    sha_field(root, "exact_scan_nonce_sha256", "production scan authority")?;
    Ok(document)
}

fn validate_bootstrap_lease(document: Document, authority: &Document) -> Result<Document, String> {
    fixture_identity_error(&document.value, "production bootstrap lease")?;
    let root = object(&document.value, "production bootstrap lease")?;
    schema_status(
        root,
        BOOTSTRAP_LEASE_SCHEMA,
        BOOTSTRAP_LEASE_STATUS,
        "production bootstrap lease",
    )?;
    for key in [
        "fresh_for_this_exact_launch",
        "one_shot",
        "non_inference_only",
        "new_capture_root_required",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
        "separate_from_source_teacher_lease",
        "production_source_hash_scan_only",
    ] {
        exact_bool(root, key, true, "production bootstrap lease")?;
    }
    for key in [
        "fixture_only",
        "synthetic_fixture_only",
        "production_adapter_forbidden",
        "source_teacher_or_logits_authorized",
        "model_gpu_server_hcli_or_tps_authorized",
        "lease_consumed_by_this_preflight",
    ] {
        exact_bool(root, key, false, "production bootstrap lease")?;
    }
    require_pointer(
        required(
            root,
            "production_scan_authority",
            "production bootstrap lease",
        )?,
        authority,
        "production bootstrap lease authority pointer",
        true,
    )?;
    sha_field(root, "lease_id", "production bootstrap lease")?;
    Ok(document)
}

/// A single reusable raw-byte window.  It deliberately has no `Read` API, so
/// all source reads remain positioned and bounded.  The backing vector is
/// zeroed after every visit and once more before terminal evidence is made.
struct BoundedPositionedReader {
    scratch: Vec<u8>,
    positioned_read_calls: u64,
    positioned_read_bytes: u64,
}

impl BoundedPositionedReader {
    fn new() -> Self {
        Self {
            scratch: vec![0_u8; MAX_POSITIONED_READ_BYTES as usize],
            positioned_read_calls: 0,
            positioned_read_bytes: 0,
        }
    }

    #[cfg(unix)]
    fn read_at_exact(
        &mut self,
        file: &File,
        offset: u64,
        length: usize,
        label: &str,
    ) -> Result<(), String> {
        if length == 0 || length > self.scratch.len() {
            return Err(format!(
                "{label} positioned read must be 1..={} bytes",
                self.scratch.len()
            ));
        }
        let mut complete = 0_usize;
        while complete < length {
            let bytes = file
                .read_at(
                    &mut self.scratch[complete..length],
                    offset
                        .checked_add(complete as u64)
                        .ok_or_else(|| format!("{label} positioned offset overflow"))?,
                )
                .map_err(|error| format!("cannot positioned-read {label}: {error}"))?;
            if bytes == 0 {
                self.scratch[..length].fill(0);
                return Err(format!("short positioned-read for {label}"));
            }
            complete += bytes;
        }
        self.positioned_read_calls = self
            .positioned_read_calls
            .checked_add(1)
            .ok_or_else(|| "positioned-read call count overflow".to_owned())?;
        self.positioned_read_bytes = self
            .positioned_read_bytes
            .checked_add(length as u64)
            .ok_or_else(|| "positioned-read byte count overflow".to_owned())?;
        Ok(())
    }

    #[cfg(not(unix))]
    fn read_at_exact(
        &mut self,
        _file: &File,
        _offset: u64,
        _length: usize,
        _label: &str,
    ) -> Result<(), String> {
        Err("bounded production positioned reader currently requires unix FileExt".to_owned())
    }

    fn hash_range(
        &mut self,
        file: &File,
        offset: u64,
        bytes: u64,
        label: &str,
    ) -> Result<String, String> {
        if bytes == 0 {
            return Err(format!("{label} hash range must be non-empty"));
        }
        let mut digest = Sha256::new();
        let mut consumed = 0_u64;
        while consumed < bytes {
            let remaining = bytes - consumed;
            let take = remaining.min(self.scratch.len() as u64) as usize;
            self.read_at_exact(
                file,
                offset
                    .checked_add(consumed)
                    .ok_or_else(|| format!("{label} range offset overflow"))?,
                take,
                label,
            )?;
            digest.update(&self.scratch[..take]);
            self.scratch[..take].fill(0);
            consumed += take as u64;
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn read_prefix(
        &mut self,
        file: &File,
        offset: u64,
        bytes: usize,
        label: &str,
    ) -> Result<Vec<u8>, String> {
        self.read_at_exact(file, offset, bytes, label)?;
        let result = self.scratch[..bytes].to_vec();
        self.scratch[..bytes].fill(0);
        Ok(result)
    }

    fn cache_is_zeroed(&self) -> bool {
        self.scratch.iter().all(|byte| *byte == 0)
    }
}

fn checked_source_root(root: &Path) -> Result<(), String> {
    let metadata = fs::symlink_metadata(root).map_err(|error| {
        format!(
            "cannot stat authorized source root {}: {error}",
            root.display()
        )
    })?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err("authorized source root must be a real non-symlink directory".to_owned());
    }
    Ok(())
}

fn checked_regular_file_under_root(
    root: &Path,
    relative: &str,
    label: &str,
) -> Result<PathBuf, String> {
    checked_relative_path(relative, label)?;
    let components = Path::new(relative).components().collect::<Vec<_>>();
    let mut path = root.to_path_buf();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err(format!("{label} path contains an invalid component"));
        };
        path.push(name);
        let metadata = fs::symlink_metadata(&path)
            .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
        if metadata.file_type().is_symlink() {
            return Err(format!("{label} may not traverse a symlink"));
        }
        if index + 1 == components.len() {
            if !metadata.file_type().is_file() {
                return Err(format!("{label} must be a regular file"));
            }
        } else if !metadata.file_type().is_dir() {
            return Err(format!("{label} parent component must be a real directory"));
        }
    }
    Ok(path)
}

fn checked_new_capture_directory(path: &Path, terminal_path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.exists() || !path.parent().is_some_and(Path::is_dir) {
        return Err(
            "--capture-dir must be a new absolute directory below an existing parent".to_owned(),
        );
    }
    if terminal_path.parent() != Some(path) || terminal_path.exists() {
        return Err("production --out must be a new direct child of --capture-dir".to_owned());
    }
    fs::create_dir(path).map_err(|error| {
        format!(
            "cannot create production capture directory {}: {error}",
            path.display()
        )
    })?;
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            format!(
                "cannot fsync production capture directory {}: {error}",
                path.display()
            )
        })
}

fn write_new_document(path: &Path, value: &Value, label: &str) -> Result<Document, String> {
    write_new(path, value)?;
    read_document(path, label, true)
}

fn replay_reservation_document(
    inputs: &Inputs,
    interface: &Document,
    authority: &Document,
    lease: &Document,
) -> Result<Value, String> {
    seal(json!({
        "schema": REPLAY_SCHEMA,
        "status": REPLAY_STATUS,
        "attempt": 1,
        "create_new_before_source_root_open": true,
        "one_child_maximum": true,
        "replay_or_relaunch_forbidden": true,
        "metadata_range_authority": evidence(&inputs.metadata.document),
        "independent_non_fixture_semantics_attester": evidence(&inputs.semantics.document),
        "runtime_admission_producer_authority": evidence(&inputs.runtime_authority),
        "interface_authority": evidence(interface),
        "production_scan_authority": evidence(authority),
        "fresh_bootstrap_lease": evidence(lease),
        "source_root_opened_or_statted": false,
        "source_teacher_or_logits_authorized": false,
        "gpu_server_hcli_or_tps_authorized": false,
    }))
}

#[derive(Clone, Debug)]
struct ScanHashes {
    source_index_sha256: String,
    shard_sha256: BTreeMap<String, String>,
    tensor_sha256: BTreeMap<String, String>,
    positioned_read_calls: u64,
    positioned_read_bytes: u64,
}

fn scan_production_source_root(
    root: &Path,
    metadata: &MetadataBinding,
) -> Result<ScanHashes, String> {
    // This function is intentionally reachable only after the caller created
    // the sealed replay reservation.  It is the sole source-root access path.
    checked_source_root(root)?;
    let mut reader = BoundedPositionedReader::new();
    let index_path = checked_regular_file_under_root(
        root,
        &metadata.source_index_relative_path,
        "source index",
    )?;
    let index_metadata = fs::symlink_metadata(&index_path)
        .map_err(|error| format!("cannot restat source index: {error}"))?;
    let index_file = File::open(&index_path)
        .map_err(|error| format!("cannot open source index {}: {error}", index_path.display()))?;
    let source_index_sha256 =
        reader.hash_range(&index_file, 0, index_metadata.len(), "source index")?;
    drop(index_file);
    if source_index_sha256 != metadata.source_index_sha256 {
        return Err("source index SHA-256 differs from metadata authority".to_owned());
    }

    let mut tensors_by_shard = BTreeMap::<String, Vec<&TensorRange>>::new();
    for tensor in &metadata.tensors {
        tensors_by_shard
            .entry(tensor.shard_relative_path.clone())
            .or_default()
            .push(tensor);
    }
    let mut shard_sha256 = BTreeMap::new();
    let mut tensor_sha256 = BTreeMap::new();
    for shard in &metadata.shards {
        let shard_path = checked_regular_file_under_root(
            root,
            &shard.relative_path,
            "source safetensors shard",
        )?;
        let shard_metadata = fs::symlink_metadata(&shard_path)
            .map_err(|error| format!("cannot restat source shard: {error}"))?;
        if shard_metadata.len() != shard.file_bytes {
            return Err(format!(
                "source shard {} size differs from metadata authority",
                shard.relative_path
            ));
        }
        let file = File::open(&shard_path).map_err(|error| {
            format!("cannot open source shard {}: {error}", shard_path.display())
        })?;
        let prefix = reader.read_prefix(&file, 0, 8, "safetensors header length")?;
        let observed_prefix_sha = sha256(&prefix);
        if observed_prefix_sha != shard.safetensors_prefix_sha256 {
            return Err(format!(
                "source shard {} safetensors prefix SHA-256 differs from metadata authority",
                shard.relative_path
            ));
        }
        let header_bytes = u64::from_le_bytes(
            prefix
                .as_slice()
                .try_into()
                .map_err(|_| "safetensors header prefix must be eight bytes".to_owned())?,
        );
        let data_start = 8_u64
            .checked_add(header_bytes)
            .ok_or_else(|| "safetensors header length overflow".to_owned())?;
        if header_bytes == 0
            || header_bytes != shard.safetensors_header_bytes
            || header_bytes > MAX_METADATA_BYTES
            || data_start > shard.file_bytes
        {
            return Err(format!(
                "source shard {} header bounds are invalid",
                shard.relative_path
            ));
        }
        let observed_header_sha =
            reader.hash_range(&file, 8, header_bytes, "safetensors header")?;
        if observed_header_sha != shard.safetensors_header_sha256 {
            return Err(format!(
                "source shard {} safetensors header SHA-256 differs from metadata authority",
                shard.relative_path
            ));
        }
        let observed_shard_sha =
            reader.hash_range(&file, 0, shard.file_bytes, "full source shard")?;
        for tensor in tensors_by_shard
            .get(&shard.relative_path)
            .into_iter()
            .flatten()
        {
            let end = tensor
                .absolute_data_offset
                .checked_add(tensor.data_bytes)
                .ok_or_else(|| format!("source tensor {} range overflows", tensor.tensor_name))?;
            if tensor.absolute_data_offset < data_start || end > shard.file_bytes {
                return Err(format!(
                    "source tensor {} escapes the safetensors payload boundary",
                    tensor.tensor_name
                ));
            }
            let observed_tensor_sha = reader.hash_range(
                &file,
                tensor.absolute_data_offset,
                tensor.data_bytes,
                &format!("source tensor {}", tensor.tensor_name),
            )?;
            if tensor_sha256
                .insert(tensor.tensor_name.clone(), observed_tensor_sha)
                .is_some()
            {
                return Err("source tensor hash aggregation duplicates a tensor".to_owned());
            }
        }
        drop(file);
        if shard_sha256
            .insert(shard.relative_path.clone(), observed_shard_sha)
            .is_some()
        {
            return Err("source shard hash aggregation duplicates a shard".to_owned());
        }
    }
    if shard_sha256.len() as u64 != SHARDS || tensor_sha256.len() as u64 != TENSORS {
        return Err("production hash aggregation geometry drifted".to_owned());
    }
    if !reader.cache_is_zeroed() {
        return Err("bounded reader cache was not zeroed after source close".to_owned());
    }
    Ok(ScanHashes {
        source_index_sha256,
        shard_sha256,
        tensor_sha256,
        positioned_read_calls: reader.positioned_read_calls,
        positioned_read_bytes: reader.positioned_read_bytes,
    })
}

fn flat_map_document(metadata: &MetadataBinding, scan: &ScanHashes) -> Result<Value, String> {
    let shard_ids = metadata
        .shards
        .iter()
        .enumerate()
        .map(|(index, shard)| {
            (
                shard.relative_path.clone(),
                format!("qwen30-source-shard-{index:02}"),
            )
        })
        .collect::<BTreeMap<_, _>>();
    seal(json!({
        "schema": FLAT_MAP_SCHEMA,
        "source_model_id": MODEL_ID,
        "source_revision": metadata.source_revision,
        "source_tensor_count": TENSORS,
        "source_index": {
            "relative_path": metadata.source_index_relative_path,
            "sha256": scan.source_index_sha256,
            "format": "huggingface.safetensors.index.json",
        },
        "maximum_window_bytes": metadata.maximum_window_bytes,
        "shards": metadata.shards.iter().map(|shard| json!({
            "shard_id": shard_ids.get(&shard.relative_path).expect("shard ID exists"),
            "relative_path": shard.relative_path,
            "bytes": shard.file_bytes,
            "sha256": scan.shard_sha256.get(&shard.relative_path).expect("shard hash exists"),
            "safetensors_header_sha256": shard.safetensors_header_sha256,
            "safetensors_prefix_sha256": shard.safetensors_prefix_sha256,
        })).collect::<Vec<_>>(),
        "tensors": metadata.tensors.iter().map(|tensor| json!({
            "tensor_name": tensor.tensor_name,
            "shard_id": shard_ids.get(&tensor.shard_relative_path).expect("tensor shard ID exists"),
            "dtype": "BF16",
            "shape": tensor.full_shape,
            "data_offset": tensor.absolute_data_offset,
            "data_bytes": tensor.data_bytes,
            "raw_bf16_sha256": scan.tensor_sha256.get(&tensor.tensor_name).expect("tensor hash exists"),
        })).collect::<Vec<_>>(),
        "fixture_only": false,
        "synthetic_fixture_only": false,
        "production_adapter_forbidden": false,
        "claim_boundary": "Production hash-map evidence only; this map does not attest source operator semantics, source teacher forwards, logits, or runtime admission.",
    }))
}

fn hash_coverage_document(
    inputs: &Inputs,
    authority: &Document,
    lease: &Document,
    replay: &Document,
    flat_map: &Document,
    scan: &ScanHashes,
) -> Result<Value, String> {
    seal(json!({
        "schema": HASH_COVERAGE_SCHEMA,
        "status": HASH_COVERAGE_STATUS,
        "production_hash_coverage_earned": true,
        "fixture_only": false,
        "synthetic_fixture_only": false,
        "production_adapter_forbidden": false,
        "metadata_range_authority": evidence(&inputs.metadata.document),
        "production_scan_authority": evidence(authority),
        "fresh_bootstrap_lease": evidence(lease),
        "replay_reservation": evidence(replay),
        "flat_runtime_range_map": evidence(flat_map),
        "coverage": {
            "source_shards": SHARDS,
            "source_tensors": TENSORS,
            "full_shard_sha256_count": scan.shard_sha256.len(),
            "raw_bf16_range_sha256_count": scan.tensor_sha256.len(),
            "source_index_sha256": scan.source_index_sha256,
        },
        "bounded_positioned_reader": {
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "positioned_read_calls": scan.positioned_read_calls,
            "positioned_read_bytes": scan.positioned_read_bytes,
            "cache_zeroed_after_every_visit_and_before_receipt": true,
            "one_shard_handle_at_a_time": true,
            "whole_shard_cache_or_mmap_forbidden": true,
        },
        "source_teacher_execution_or_logits": false,
        "operator_or_reader_execution_attestation_emitted": false,
        "source_teacher_runtime_admission_earned": false,
    }))
}

fn production_capture_document(
    inputs: &Inputs,
    interface: &Document,
    authority: &Document,
    lease: &Document,
    replay: &Document,
    flat_map: &Document,
    coverage: &Document,
    scan: &ScanHashes,
) -> Result<Value, String> {
    seal(json!({
        "schema": PRODUCTION_CAPTURE_SCHEMA,
        "status": PRODUCTION_CAPTURE_STATUS,
        "production_hash_scan_earned": true,
        "receipt_written_last": true,
        "fixture_only": false,
        "synthetic_fixture_only": false,
        "production_adapter_forbidden": false,
        "metadata_range_authority": evidence(&inputs.metadata.document),
        "independent_non_fixture_semantics_attester": evidence(&inputs.semantics.document),
        "runtime_admission_producer_authority": evidence(&inputs.runtime_authority),
        "interface_authority": evidence(interface),
        "production_scan_authority": evidence(authority),
        "fresh_bootstrap_lease": evidence(lease),
        "replay_reservation": evidence(replay),
        "flat_runtime_range_map": evidence(flat_map),
        "hash_coverage_attestation": evidence(coverage),
        "geometry": {
            "source_shards": SHARDS,
            "source_tensors": TENSORS,
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "positioned_read_calls": scan.positioned_read_calls,
        },
        "source_handles_closed": true,
        "reader_cache_zeroed": true,
        "source_teacher_or_logits_executed": false,
        "operator_or_reader_execution_attestation_emitted": false,
        "source_teacher_runtime_admission_earned": false,
        "model_gpu_server_hcli_or_tps_action": false,
        "claim_boundary": "Complete production hash-map component only. It is not source-teacher execution and does not earn source-teacher attestations, runtime admission, logits, model residency, GPU/server/HCLI, TPS, TG, or tournament evidence.",
    }))
}

fn execute_production_hash_scan(
    args: &Args,
    inputs: &Inputs,
    interface: &Document,
    authority: &Document,
    lease: &Document,
) -> Result<Value, String> {
    let capture_dir = args
        .capture_dir
        .as_deref()
        .expect("parser requires capture directory");
    let source_root = args
        .source_root
        .as_deref()
        .expect("parser requires source root");
    checked_new_capture_directory(capture_dir, &args.out)?;
    let replay = write_new_document(
        &capture_dir.join("replay-reservation.json"),
        &replay_reservation_document(inputs, interface, authority, lease)?,
        "production replay reservation",
    )?;
    let scan = scan_production_source_root(source_root, &inputs.metadata)?;
    let flat_map = write_new_document(
        &capture_dir.join("flat-runtime-range-map.json"),
        &flat_map_document(&inputs.metadata, &scan)?,
        "production flat runtime range map",
    )?;
    let coverage = write_new_document(
        &capture_dir.join("hash-coverage-attestation.json"),
        &hash_coverage_document(inputs, authority, lease, &replay, &flat_map, &scan)?,
        "production hash coverage attestation",
    )?;
    let capture = production_capture_document(
        inputs, interface, authority, lease, &replay, &flat_map, &coverage, &scan,
    )?;
    let terminal = write_new_document(&args.out, &capture, "production hash scan capture")?;
    File::open(capture_dir)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| {
            format!("cannot fsync production capture directory after receipt: {error}")
        })?;
    Ok(terminal.value)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() || path.exists() || !path.parent().is_some_and(Path::is_dir) {
        return Err("--out must be a new absolute path below an existing directory".to_owned());
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize interface result: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync {}: {error}", path.display()))
}

fn run(args: Args) -> Result<Value, String> {
    let inputs = validate_inputs(&args)?;
    match args.mode {
        Mode::Preflight => {
            let document = interface_document(&inputs)?;
            write_new(&args.out, &document)?;
            Ok(document)
        }
        Mode::ProductionScan => {
            // Keep this order. Every metadata/fixture/lease validation and
            // the create-new replay reservation in `execute_production_hash_scan`
            // completes before that backend may read a source-root byte.
            let interface = validate_interface(
                read_document(
                    args.interface_authority
                        .as_deref()
                        .expect("parser requires interface authority"),
                    "production scan interface",
                    true,
                )?,
                &inputs,
            )?;
            let authority = validate_production_scan_authority(
                read_document(
                    args.production_scan_authority
                        .as_deref()
                        .expect("parser requires scan authority"),
                    "production scan authority",
                    true,
                )?,
                &inputs,
                &interface,
            )?;
            let lease = validate_bootstrap_lease(
                read_document(
                    args.bootstrap_lease
                        .as_deref()
                        .expect("parser requires bootstrap lease"),
                    "production bootstrap lease",
                    true,
                )?,
                &authority,
            )?;
            execute_production_hash_scan(&args, &inputs, &interface, &authority, &lease)
        }
    }
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("cannot render production-scan interface result: {error}");
                std::process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("Q30 production hash-scan interface refused before source access: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn hash(label: &str) -> String {
        sha256(label.as_bytes())
    }

    fn test_revision() -> String {
        hash("synthetic production scan revision")[..40].to_owned()
    }

    fn write_value(path: &Path, value: &Value) {
        fs::write(
            path,
            serde_json::to_vec_pretty(value).expect("fixture JSON serializes"),
        )
        .expect("fixture JSON writes");
    }

    fn production_range() -> Value {
        let shard_names = (0..SHARDS)
            .map(|index| format!("model-{index:05}-of-00016.safetensors"))
            .collect::<Vec<_>>();
        let mut per_shard_tensors = vec![0_u64; SHARDS as usize];
        for index in 0..TENSORS {
            per_shard_tensors[(index % SHARDS) as usize] += 1;
        }
        let shards = shard_names
            .iter()
            .enumerate()
            .map(|(index, path)| {
                json!({
                    "relative_path": path,
                    "file_bytes": 10 + per_shard_tensors[index] * 2,
                    "safetensors_header_bytes": 2,
                    "safetensors_header_sha256": hash("{}"),
                    "safetensors_prefix_sha256": sha256(&2_u64.to_le_bytes()),
                })
            })
            .collect::<Vec<_>>();
        let mut next_offsets = vec![10_u64; SHARDS as usize];
        let tensors = (0..TENSORS)
            .map(|index| {
                let shard_index = (index % SHARDS) as usize;
                let shard = &shard_names[shard_index];
                let absolute_data_offset = next_offsets[shard_index];
                next_offsets[shard_index] += 2;
                json!({
                    "source_dtype": "BF16",
                    "tensor_name": format!("model.layers.{index}.test.weight"),
                    "shard_relative_path": shard,
                    "full_shape": [1],
                    "row_window_shape": [1],
                    "absolute_data_offset": absolute_data_offset,
                    "data_bytes": 2,
                })
            })
            .collect::<Vec<_>>();
        let authority = json!({
            "schema": RANGE_SCHEMA,
            "status": RANGE_STATUS,
            "source": {
                "model_id": MODEL_ID,
                "source_revision": test_revision(),
                "source_shard_count": SHARDS,
                "source_tensor_count": TENSORS,
                "source_index": {
                    "relative_path": "model.safetensors.index.json",
                    "format": "huggingface.safetensors.index.json",
                    "sha256": hash("index"),
                    "weight_map_tensor_count": TENSORS,
                },
            },
            "exact_streamed_oracle_scope": {
                "source_template_token_count": 369,
                "forced_identical_continuation_token_id": 949,
                "total_forwards_per_replay_arm": 370,
                "layers": 48,
                "top_k_routes_per_token": 8,
                "sampling_or_autoregressive_feedback_forbidden": true,
            },
            "metadata_access_boundary": {
                "source_tensor_payload_bytes_read": 0,
                "tensor_payload_hashes_collected": false,
                "whole_shard_payload_checksum_collected": false,
                "mmap_or_memory_map_used": false,
                "source_model_instantiated": false,
                "gpu_or_metal_invoked": false,
                "server_started": false,
                "hcli_invoked": false,
                "lease_requested": false,
            },
            "shards": shards,
            "tensors": tensors,
        });
        json!({
            "authority_content_sha256": sha256(canonical(&authority).expect("authority canonicalizes").as_bytes()),
            "authority": authority,
        })
    }

    fn write_synthetic_source_root(root: &Path, range: &Value) {
        fs::create_dir(root).expect("synthetic source root creates");
        fs::write(root.join("model.safetensors.index.json"), b"index")
            .expect("synthetic index writes");
        let shards = range["authority"]["shards"]
            .as_array()
            .expect("synthetic shards");
        for (index, shard) in shards.iter().enumerate() {
            let relative_path = shard["relative_path"].as_str().expect("shard path");
            let bytes = shard["file_bytes"].as_u64().expect("shard bytes") as usize;
            let mut contents = Vec::with_capacity(bytes);
            contents.extend_from_slice(&(2_u64).to_le_bytes());
            contents.extend_from_slice(b"{}");
            contents.extend(std::iter::repeat((index as u8).wrapping_add(1)).take(bytes - 10));
            fs::write(root.join(relative_path), contents).expect("synthetic shard writes");
        }
    }

    fn semantics(range_raw_sha: &str, range_content_sha: &str) -> Value {
        json!({
            "schema": SEMANTICS_SCHEMA,
            "status": SEMANTICS_STATUS,
            "execution_boundary": {
                "source_tensor_payload_opened": false,
                "source_safetensors_or_other_weight_path_accepted": false,
                "source_model_instantiated": false,
                "source_inference_executed": false,
                "gpu_or_metal_invoked": false,
                "server_started": false,
                "hcli_invoked": false,
                "lease_requested": false,
            },
            "pinned_source_binding": {
                "source_model_id": MODEL_ID,
                "source_revision": test_revision(),
                "source_index_sha256": hash("index"),
            },
            "consumed_metadata_contracts": {
                "range_authority": {
                    "document_sha256": range_raw_sha,
                    "authority_content_sha256": range_content_sha,
                },
            },
            "future_exact_execution_attestation": {
                "schema": OPERATOR_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": OPERATOR_ATTESTATION_STATUS,
            },
        })
    }

    fn runtime_authority(range: &Document, semantics: &Document, range_content_sha: &str) -> Value {
        seal(json!({
            "schema": RUNTIME_PRODUCER_SCHEMA,
            "status": RUNTIME_PRODUCER_STATUS,
            "prepared": true,
            "runtime_admission_earned": false,
            "source_payload_validation_executed": false,
            "sealed_metadata_authority_binding": {
                "metadata_range_authority": evidence(range),
                "authority_content_sha256": range_content_sha,
            },
            "metadata_semantics_binding": {
                "operator_semantics_attester": evidence(semantics),
            },
            "future_flat_runtime_range_map": {
                "schema": FLAT_MAP_SCHEMA,
                "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            },
            "future_runtime_admission_receipt": {
                "schema": SOURCE_TEACHER_RUNTIME_SCHEMA,
                "status_only_after_bounded_source_validation": SOURCE_TEACHER_RUNTIME_STATUS,
            },
        }))
        .expect("runtime fixture seals")
    }

    fn paths(temp: &TempDir) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
        (
            temp.path().join("range.json"),
            temp.path().join("semantics.json"),
            temp.path().join("runtime.json"),
            temp.path().join("interface.json"),
        )
    }

    fn preflight_args(range: &Path, semantics: &Path, runtime: &Path, out: &Path) -> Args {
        parse_args_from(vec![
            "--mode".into(),
            "preflight".into(),
            "--range-authority".into(),
            range.display().to_string(),
            "--semantics-attester".into(),
            semantics.display().to_string(),
            "--runtime-admission-authority".into(),
            runtime.display().to_string(),
            "--out".into(),
            out.display().to_string(),
        ])
        .expect("preflight args parse")
    }

    fn production_authority(interface: &Document, inputs: &Inputs) -> Value {
        seal(json!({
            "schema": PRODUCTION_SCAN_AUTHORITY_SCHEMA,
            "status": PRODUCTION_SCAN_AUTHORITY_STATUS,
            "fresh_for_this_exact_scan": true,
            "one_shot": true,
            "non_inference_hash_scan_only": true,
            "source_root_open_only_after_all_authorities_validate": true,
            "fixture_only": false,
            "synthetic_fixture_only": false,
            "production_adapter_forbidden": false,
            "source_teacher_or_logits_authorized": false,
            "model_gpu_server_hcli_or_tps_authorized": false,
            "lease_consumed": false,
            "immutable_bindings": {
                "interface_authority": evidence(interface),
                "metadata_range_authority": evidence(&inputs.metadata.document),
                "metadata_authority_content_sha256": inputs.metadata.authority_content_sha256,
                "independent_semantics_attester": evidence(&inputs.semantics.document),
                "runtime_admission_producer_authority": evidence(&inputs.runtime_authority),
            },
            "geometry": {
                "source_shards": SHARDS,
                "source_tensors": TENSORS,
                "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
                "maximum_live_raw_bf16_windows": 1,
            },
            "exact_scan_nonce_sha256": hash("scan nonce"),
        }))
        .expect("authority fixture seals")
    }

    fn production_lease(authority: &Document) -> Value {
        seal(json!({
            "schema": BOOTSTRAP_LEASE_SCHEMA,
            "status": BOOTSTRAP_LEASE_STATUS,
            "fresh_for_this_exact_launch": true,
            "one_shot": true,
            "non_inference_only": true,
            "new_capture_root_required": true,
            "existing_output_reuse_forbidden": true,
            "replay_or_relaunch_forbidden": true,
            "separate_from_source_teacher_lease": true,
            "production_source_hash_scan_only": true,
            "fixture_only": false,
            "synthetic_fixture_only": false,
            "production_adapter_forbidden": false,
            "source_teacher_or_logits_authorized": false,
            "model_gpu_server_hcli_or_tps_authorized": false,
            "lease_consumed_by_this_preflight": false,
            "production_scan_authority": evidence(authority),
            "lease_id": hash("production lease"),
        }))
        .expect("lease fixture seals")
    }

    #[test]
    fn preflight_reserves_real_geometry_and_true_teacher_boundary() {
        let temporary = TempDir::new().expect("temporary directory");
        let (range_path, semantics_path, runtime_path, out_path) = paths(&temporary);
        let range = production_range();
        let range_content_sha = range["authority_content_sha256"]
            .as_str()
            .expect("range content hash")
            .to_owned();
        write_value(&range_path, &range);
        let range_document = read_document(&range_path, "range", false).expect("range reads");
        let semantics_value = semantics(&range_document.raw_document_sha256, &range_content_sha);
        write_value(&semantics_path, &semantics_value);
        let semantics_document =
            read_document(&semantics_path, "semantics", false).expect("semantics reads");
        let mut runtime =
            runtime_authority(&range_document, &semantics_document, &range_content_sha);
        runtime["sealed_metadata_authority_binding"]["metadata_range_authority"]
            .as_object_mut()
            .expect("metadata pointer object")
            .remove("canonical_document_sha256");
        runtime["metadata_semantics_binding"]["operator_semantics_attester"]
            .as_object_mut()
            .expect("semantics pointer object")
            .remove("canonical_document_sha256");
        write_value(
            &runtime_path,
            &seal(runtime).expect("legacy raw-pointer runtime authority reseals"),
        );

        let result = run(preflight_args(
            &range_path,
            &semantics_path,
            &runtime_path,
            &out_path,
        ))
        .expect("production preflight validates");
        assert_eq!(result["schema"], INTERFACE_SCHEMA);
        assert_eq!(result["status"], PREPARED_STATUS);
        assert_eq!(result["future_bounded_hash_scan"]["source_shards"], SHARDS);
        assert_eq!(
            result["future_bounded_hash_scan"]["source_tensors"],
            TENSORS
        );
        assert_eq!(
            result["future_bounded_hash_scan"]["future_hash_coverage_attestation"]["status"],
            HASH_COVERAGE_STATUS
        );
        assert_eq!(
            result["separate_source_teacher_follow_on"]["required_later_not_emitted_by_hash_scan"]
                .as_array()
                .expect("separate source-teacher outputs"),
            &vec![
                json!({"schema": OPERATOR_ATTESTATION_SCHEMA, "status": OPERATOR_ATTESTATION_STATUS}),
                json!({"schema": READER_ATTESTATION_SCHEMA, "status": READER_ATTESTATION_STATUS}),
                json!({"schema": SOURCE_TEACHER_RUNTIME_SCHEMA, "status": SOURCE_TEACHER_RUNTIME_STATUS}),
            ]
        );
        verify_seal(&result, "preflight output").expect("preflight is sealed");
    }

    #[test]
    fn fixture_identity_is_rejected_before_any_source_root_probe() {
        let temporary = TempDir::new().expect("temporary directory");
        let (range_path, semantics_path, runtime_path, interface_path) = paths(&temporary);
        let range = production_range();
        let range_content_sha = range["authority_content_sha256"]
            .as_str()
            .expect("range content hash")
            .to_owned();
        write_value(&range_path, &range);
        let range_document = read_document(&range_path, "range", false).expect("range reads");
        let semantics_value = semantics(&range_document.raw_document_sha256, &range_content_sha);
        write_value(&semantics_path, &semantics_value);
        let semantics_document =
            read_document(&semantics_path, "semantics", false).expect("semantics reads");
        write_value(
            &runtime_path,
            &runtime_authority(&range_document, &semantics_document, &range_content_sha),
        );
        run(preflight_args(
            &range_path,
            &semantics_path,
            &runtime_path,
            &interface_path,
        ))
        .expect("interface preflight writes");
        let args = preflight_args(&range_path, &semantics_path, &runtime_path, &interface_path);
        let inputs = validate_inputs(&args).expect("inputs validate");
        let interface = read_document(&interface_path, "interface", true).expect("interface reads");
        let mut authority = production_authority(&interface, &inputs);
        authority
            .as_object_mut()
            .expect("authority object")
            .insert(
                "attempted_flat_map".to_owned(),
                json!({"schema": "hawking.ascension.qwen30_source_bf16_range_map_fixture.v1", "status": "SYNTHETIC_FIXTURE_ONLY"}),
            );
        let authority = seal(authority).expect("fixture-relabeled authority reseals");
        let authority_path = temporary.path().join("fixture-authority.json");
        write_value(&authority_path, &authority);
        let missing_root = temporary.path().join("must-never-be-probed-source-root");
        let capture_dir = temporary.path().join("must-never-be-created-capture");
        let output = temporary.path().join("refusal.json");
        let attempt = parse_args_from(vec![
            "--mode".into(),
            "production-scan".into(),
            "--range-authority".into(),
            range_path.display().to_string(),
            "--semantics-attester".into(),
            semantics_path.display().to_string(),
            "--runtime-admission-authority".into(),
            runtime_path.display().to_string(),
            "--interface-authority".into(),
            interface_path.display().to_string(),
            "--production-scan-authority".into(),
            authority_path.display().to_string(),
            "--bootstrap-lease".into(),
            temporary
                .path()
                .join("unread-lease.json")
                .display()
                .to_string(),
            "--source-root".into(),
            missing_root.display().to_string(),
            "--capture-dir".into(),
            capture_dir.display().to_string(),
            "--out".into(),
            output.display().to_string(),
        ])
        .expect("attempt args parse");
        let error = run(attempt).expect_err("fixture identity must refuse");
        assert!(error.contains("fixture-only identity"));
        assert!(
            !missing_root.exists(),
            "source root was never probed or created"
        );
        assert!(!capture_dir.exists(), "capture directory was never created");
        assert!(
            !output.exists(),
            "no refusal record after invalid authority"
        );
    }

    #[test]
    fn sealed_lease_authority_pointer_requires_exact_canonical_identity_before_source_probe() {
        let temporary = TempDir::new().expect("temporary directory");
        let (range_path, semantics_path, runtime_path, interface_path) = paths(&temporary);
        let range = production_range();
        let range_content_sha = range["authority_content_sha256"]
            .as_str()
            .expect("range content hash")
            .to_owned();
        write_value(&range_path, &range);
        let range_document = read_document(&range_path, "range", false).expect("range reads");
        let semantics_value = semantics(&range_document.raw_document_sha256, &range_content_sha);
        write_value(&semantics_path, &semantics_value);
        let semantics_document =
            read_document(&semantics_path, "semantics", false).expect("semantics reads");
        write_value(
            &runtime_path,
            &runtime_authority(&range_document, &semantics_document, &range_content_sha),
        );
        run(preflight_args(
            &range_path,
            &semantics_path,
            &runtime_path,
            &interface_path,
        ))
        .expect("interface preflight writes");
        let inputs = validate_inputs(&preflight_args(
            &range_path,
            &semantics_path,
            &runtime_path,
            &interface_path,
        ))
        .expect("inputs validate");
        let interface = read_document(&interface_path, "interface", true).expect("interface reads");
        let authority_value = production_authority(&interface, &inputs);
        let authority_path = temporary.path().join("authority.json");
        write_value(&authority_path, &authority_value);
        let authority = read_document(&authority_path, "authority", true).expect("authority reads");

        for (label, canonical) in [
            ("missing", None),
            (
                "substituted",
                Some(hash("different canonical authority identity")),
            ),
        ] {
            let mut lease = production_lease(&authority);
            let pointer = lease["production_scan_authority"]
                .as_object_mut()
                .expect("lease authority pointer object");
            match canonical {
                None => {
                    pointer.remove("canonical_document_sha256");
                }
                Some(value) => {
                    pointer.insert("canonical_document_sha256".to_owned(), Value::String(value));
                }
            }
            let lease_path = temporary.path().join(format!("{label}-lease.json"));
            write_value(&lease_path, &seal(lease).expect("tampered lease reseals"));
            let missing_root = temporary
                .path()
                .join(format!("{label}-must-not-be-probed-root"));
            let capture_dir = temporary
                .path()
                .join(format!("{label}-must-not-be-created-capture"));
            let output = temporary.path().join(format!("{label}-refusal.json"));
            let attempt = parse_args_from(vec![
                "--mode".into(),
                "production-scan".into(),
                "--range-authority".into(),
                range_path.display().to_string(),
                "--semantics-attester".into(),
                semantics_path.display().to_string(),
                "--runtime-admission-authority".into(),
                runtime_path.display().to_string(),
                "--interface-authority".into(),
                interface_path.display().to_string(),
                "--production-scan-authority".into(),
                authority_path.display().to_string(),
                "--bootstrap-lease".into(),
                lease_path.display().to_string(),
                "--source-root".into(),
                missing_root.display().to_string(),
                "--capture-dir".into(),
                capture_dir.display().to_string(),
                "--out".into(),
                output.display().to_string(),
            ])
            .expect("authority-pointer attempt parses");
            let error = run(attempt).expect_err("canonical authority drift must refuse");
            if label == "missing" {
                assert!(error.contains("must bind canonical identity"));
            } else {
                assert!(error.contains("canonical binding drifted"));
            }
            assert!(!missing_root.exists(), "source root must not be probed");
            assert!(
                !capture_dir.exists(),
                "capture directory must not be created"
            );
            assert!(!output.exists(), "no output follows lease ABI refusal");
        }
    }

    #[test]
    fn sealed_non_fixture_bundle_exercises_bounded_16_by_18867_backend_on_synthetic_files() {
        let temporary = TempDir::new().expect("temporary directory");
        let (range_path, semantics_path, runtime_path, interface_path) = paths(&temporary);
        let range = production_range();
        let range_content_sha = range["authority_content_sha256"]
            .as_str()
            .expect("range content hash")
            .to_owned();
        write_value(&range_path, &range);
        let range_document = read_document(&range_path, "range", false).expect("range reads");
        let semantics_value = semantics(&range_document.raw_document_sha256, &range_content_sha);
        write_value(&semantics_path, &semantics_value);
        let semantics_document =
            read_document(&semantics_path, "semantics", false).expect("semantics reads");
        write_value(
            &runtime_path,
            &runtime_authority(&range_document, &semantics_document, &range_content_sha),
        );
        run(preflight_args(
            &range_path,
            &semantics_path,
            &runtime_path,
            &interface_path,
        ))
        .expect("interface preflight writes");
        let inputs = validate_inputs(&preflight_args(
            &range_path,
            &semantics_path,
            &runtime_path,
            &interface_path,
        ))
        .expect("inputs validate");
        let interface = read_document(&interface_path, "interface", true).expect("interface reads");
        let authority_value = production_authority(&interface, &inputs);
        let authority_path = temporary.path().join("authority.json");
        write_value(&authority_path, &authority_value);
        let authority = read_document(&authority_path, "authority", true).expect("authority reads");
        let lease_path = temporary.path().join("lease.json");
        write_value(&lease_path, &production_lease(&authority));
        let synthetic_root = temporary.path().join("synthetic-production-source-root");
        write_synthetic_source_root(&synthetic_root, &range);
        let capture_dir = temporary.path().join("new-production-capture");
        let terminal_path = capture_dir.join("production-capture.json");
        let result = run(parse_args_from(vec![
            "--mode".into(),
            "production-scan".into(),
            "--range-authority".into(),
            range_path.display().to_string(),
            "--semantics-attester".into(),
            semantics_path.display().to_string(),
            "--runtime-admission-authority".into(),
            runtime_path.display().to_string(),
            "--interface-authority".into(),
            interface_path.display().to_string(),
            "--production-scan-authority".into(),
            authority_path.display().to_string(),
            "--bootstrap-lease".into(),
            lease_path.display().to_string(),
            "--source-root".into(),
            synthetic_root.display().to_string(),
            "--capture-dir".into(),
            capture_dir.display().to_string(),
            "--out".into(),
            terminal_path.display().to_string(),
        ])
        .expect("production attempt args parse"))
        .expect("synthetic production backend completes");
        assert_eq!(result["schema"], PRODUCTION_CAPTURE_SCHEMA);
        assert_eq!(result["status"], PRODUCTION_CAPTURE_STATUS);
        assert_eq!(result["geometry"]["source_shards"], SHARDS);
        assert_eq!(result["geometry"]["source_tensors"], TENSORS);
        assert_eq!(result["source_teacher_or_logits_executed"], false);
        assert_eq!(result["source_teacher_runtime_admission_earned"], false);
        assert!(terminal_path.is_file(), "terminal receipt is written last");
        let replay = read_document(
            &capture_dir.join("replay-reservation.json"),
            "synthetic replay",
            true,
        )
        .expect("replay is sealed");
        let flat_map = read_document(
            &capture_dir.join("flat-runtime-range-map.json"),
            "synthetic flat map",
            true,
        )
        .expect("flat map is sealed");
        let coverage = read_document(
            &capture_dir.join("hash-coverage-attestation.json"),
            "synthetic coverage",
            true,
        )
        .expect("coverage is sealed");
        assert_eq!(replay.value["status"], REPLAY_STATUS);
        assert_eq!(flat_map.value["schema"], FLAT_MAP_SCHEMA);
        assert_eq!(flat_map.value["source_tensor_count"], TENSORS);
        assert_eq!(
            flat_map.value["shards"]
                .as_array()
                .expect("map shards")
                .len() as u64,
            SHARDS
        );
        assert_eq!(
            flat_map.value["tensors"]
                .as_array()
                .expect("map tensors")
                .len() as u64,
            TENSORS
        );
        assert_eq!(coverage.value["status"], HASH_COVERAGE_STATUS);
        assert_eq!(
            coverage.value["coverage"]["full_shard_sha256_count"],
            SHARDS
        );
        assert_eq!(
            coverage.value["coverage"]["raw_bf16_range_sha256_count"],
            TENSORS
        );
        assert!(
            coverage.value["bounded_positioned_reader"]["positioned_read_calls"]
                .as_u64()
                .expect("reader calls")
                > TENSORS,
            "full shard, header, and range reads all use the one bounded visitor"
        );
        verify_seal(&result, "terminal output").expect("terminal is sealed");
    }
}
