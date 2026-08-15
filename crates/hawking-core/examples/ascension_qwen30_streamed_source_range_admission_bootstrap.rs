#![allow(dead_code)]

//! CPU/build-only one-shot Q30 range-admission bootstrap contract.
//!
//! This program reserves a future non-inference hash/range scan. It never
//! opens, stats, maps, or scans a real source root. The file-backed reader is
//! exercised only with synthetic fixture shards in tests.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::FileExt;

const SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_preflight.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_NOT_EXECUTED";
const REFUSED_STATUS: &str =
    "REFUSED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_CPU_BUILD_NOT_ENABLED";

const RANGE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1";
const RANGE_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED";
const SEMANTICS_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1";
const SEMANTICS_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED";
const FLAT_MAP_SCHEMA: &str = "hawking.ascension.qwen30_source_bf16_range_map.v1";
const RUNTIME_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1";
const RUNTIME_ADMISSION_STATUS: &str =
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY";
const OPERATOR_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1";
const OPERATOR_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED";
const READER_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1";
const READER_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED";
const BOOTSTRAP_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_quiet_lease.v1";
const BOOTSTRAP_LEASE_STATUS: &str =
    "GRANTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_HASH_SCAN_ONE_SHOT";
const BOOTSTRAP_CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_capture.v1";
const BOOTSTRAP_CAPTURE_STATUS: &str =
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_HASH_SCAN_NOT_SOURCE_TEACHER";
const FIXTURE_SCAN_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_fixture_authority.v1";
const FIXTURE_SCAN_AUTHORITY_STATUS: &str =
    "ADMITTED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_SYNTHETIC_FIXTURE_ONLY";
const FIXTURE_FLAT_MAP_SCHEMA: &str = "hawking.ascension.qwen30_source_bf16_range_map_fixture.v1";
const FIXTURE_OPERATOR_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_operator_accumulation_fixture_attestation.v1";
const FIXTURE_OPERATOR_ATTESTATION_STATUS: &str =
    "CAPTURED_QWEN30_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SYNTHETIC_FIXTURE_ONLY";
const FIXTURE_READER_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_bf16_range_reader_fixture_attestation.v1";
const FIXTURE_READER_ATTESTATION_STATUS: &str =
    "CAPTURED_QWEN30_STREAMED_SOURCE_BF16_RANGE_READER_SYNTHETIC_FIXTURE_ONLY";
const FIXTURE_RUNTIME_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_runtime_range_admission_fixture.v1";
const FIXTURE_RUNTIME_ADMISSION_STATUS: &str =
    "EARNED_QWEN30_STREAMED_SOURCE_RUNTIME_RANGE_ADMISSION_SYNTHETIC_FIXTURE_ONLY_NOT_SOURCE_TEACHER";
const FIXTURE_CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_fixture_capture.v1";
const FIXTURE_CAPTURE_STATUS: &str =
    "CAPTURED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_SYNTHETIC_FIXTURE_ONLY";

const MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const PRODUCTION_SHARDS: u64 = 16;
const PRODUCTION_TENSORS: u64 = 18_867;
const MAX_WINDOW_BYTES: usize = 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const BF16_BYTES: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Preflight,
    BootstrapScan,
}

#[derive(Debug)]
struct Args {
    mode: Mode,
    range_authority: PathBuf,
    semantics: PathBuf,
    bootstrap_lease: Option<PathBuf>,
    scan_authority: Option<PathBuf>,
    source_root: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct Metadata {
    range_document_sha256: String,
    authority_content_sha256: String,
    revision: String,
    index_sha256: String,
    maximum_window_bytes: usize,
    index_relative_path: String,
    shards: Vec<ShardRange>,
    tensors: Vec<TensorRange>,
}

#[derive(Clone, Debug)]
struct ShardRange {
    relative_path: String,
    bytes: u64,
    safetensors_header_sha256: String,
}

#[derive(Clone, Debug)]
struct TensorRange {
    tensor_name: String,
    shard_relative_path: String,
    full_shape: Vec<u64>,
    absolute_data_offset: u64,
    data_bytes: u64,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_streamed_source_range_admission_bootstrap \\
  --mode preflight --range-authority ABS_JSON --semantics ABS_JSON --out NEW_ABS_JSON
future authority-gated scan (never invoked by this build step):
  --mode bootstrap-scan --range-authority ABS_JSON --semantics ABS_JSON \\
  --bootstrap-lease ABS_FRESH_LEASE --scan-authority ABS_SEALED_SCAN_AUTHORITY \\
  --source-root ABS_AUTHORIZED_SYNTHETIC_OR_FUTURE_ROOT \\
  --capture-dir NEW_ABS_CAPTURE_DIR --out NEW_ABS_JSON"
}

fn parse_mode(value: &str) -> Result<Mode, String> {
    match value {
        "preflight" => Ok(Mode::Preflight),
        "bootstrap-scan" => Ok(Mode::BootstrapScan),
        _ => Err("--mode must be preflight or bootstrap-scan".to_owned()),
    }
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::new();
    let mut it = arguments.into_iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        if !matches!(
            flag.as_str(),
            "--mode"
                | "--range-authority"
                | "--semantics"
                | "--bootstrap-lease"
                | "--scan-authority"
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
    let mode = values
        .remove("--mode")
        .as_deref()
        .map(parse_mode)
        .transpose()?
        .unwrap_or(Mode::Preflight);
    let required_path = |value: Option<String>, flag: &str| {
        value
            .map(PathBuf::from)
            .ok_or_else(|| format!("{flag} is required; {}", usage()))
    };
    let args = Args {
        mode,
        range_authority: required_path(values.remove("--range-authority"), "--range-authority")?,
        semantics: required_path(values.remove("--semantics"), "--semantics")?,
        bootstrap_lease: values.remove("--bootstrap-lease").map(PathBuf::from),
        scan_authority: values.remove("--scan-authority").map(PathBuf::from),
        source_root: values.remove("--source-root").map(PathBuf::from),
        capture_dir: values.remove("--capture-dir").map(PathBuf::from),
        out: required_path(values.remove("--out"), "--out")?,
    };
    if !values.is_empty() {
        return Err(format!("unexpected arguments; {}", usage()));
    }
    for (flag, path) in [
        ("--range-authority", &args.range_authority),
        ("--semantics", &args.semantics),
        ("--out", &args.out),
    ] {
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
    }
    match args.mode {
        Mode::Preflight => {
            if args.bootstrap_lease.is_some()
                || args.scan_authority.is_some()
                || args.source_root.is_some()
                || args.capture_dir.is_some()
            {
                return Err("preflight does not accept lease, source-root, or capture-dir".into());
            }
        }
        Mode::BootstrapScan => {
            for (flag, value) in [
                ("--bootstrap-lease", args.bootstrap_lease.as_ref()),
                ("--scan-authority", args.scan_authority.as_ref()),
                ("--source-root", args.source_root.as_ref()),
                ("--capture-dir", args.capture_dir.as_ref()),
            ] {
                let value = value.ok_or_else(|| format!("{flag} is required; {}", usage()))?;
                if !value.is_absolute() {
                    return Err(format!("{flag} must be absolute"));
                }
            }
        }
    }
    Ok(args)
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

fn canonical(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".into()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value).map_err(|error| error.to_string()),
        Value::Array(values) => values
            .iter()
            .map(canonical)
            .collect::<Result<Vec<_>, _>>()
            .map(|items| format!("[{}]", items.join(","))),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            keys.into_iter()
                .map(|key| {
                    Ok(format!(
                        "{}:{}",
                        serde_json::to_string(key).map_err(|error| error.to_string())?,
                        canonical(values.get(key).expect("key remains present"))?
                    ))
                })
                .collect::<Result<Vec<_>, String>>()
                .map(|items| format!("{{{}}}", items.join(",")))
        }
    }
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{key} must be an object"))
}

fn array_field<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label}.{key} must be an array"))
}

fn text<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{key} must be non-empty text"))
}

fn hash_field(object: &Map<String, Value>, key: &str, label: &str) -> Result<String, String> {
    let value = text(object, key, label)?;
    if !valid_sha(value) {
        return Err(format!("{label}.{key} must be a lowercase SHA-256"));
    }
    Ok(value.into())
}

fn u64_field(object: &Map<String, Value>, key: &str, label: &str) -> Result<u64, String> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{key} must be an unsigned integer"))
}

fn required_bool(
    object: &Map<String, Value>,
    key: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if object.get(key).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label}.{key} must be {expected}"));
    }
    Ok(())
}

fn exact_text(
    object: &Map<String, Value>,
    key: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    if text(object, key, label)? != expected {
        return Err(format!("{label}.{key} drifted"));
    }
    Ok(())
}

fn seal(mut value: Value) -> Result<Value, String> {
    let root = value
        .as_object_mut()
        .ok_or_else(|| "output must be an object".to_owned())?;
    root.remove("seal_sha256");
    let seal = sha256(canonical(&value)?.as_bytes());
    value
        .as_object_mut()
        .expect("object remains object")
        .insert("seal_sha256".into(), Value::String(seal));
    Ok(value)
}

fn verify_seal(value: &Value, label: &str) -> Result<(), String> {
    let root = object(value, label)?;
    let seal = hash_field(root, "seal_sha256", label)?;
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    if sha256(canonical(&Value::Object(unsigned))?.as_bytes()) != seal {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct SealedDocument {
    value: Value,
    raw_document_sha256: String,
    seal_sha256: String,
}

fn read_sealed_document(path: &Path, label: &str) -> Result<SealedDocument, String> {
    let (value, raw_document_sha256) = read_metadata(path, label)?;
    verify_seal(&value, label)?;
    let seal_sha256 = hash_field(object(&value, label)?, "seal_sha256", label)?;
    Ok(SealedDocument {
        value,
        raw_document_sha256,
        seal_sha256,
    })
}

fn evidence(document: &SealedDocument) -> Value {
    json!({
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": sha256(canonical(&document.value).expect("sealed document canonicalizes").as_bytes()),
        "seal_sha256": document.seal_sha256,
    })
}

fn read_metadata(path: &Path, label: &str) -> Result<(Value, String), String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata =
        fs::symlink_metadata(path).map_err(|error| format!("cannot stat {label}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() == 0 || metadata.len() > MAX_METADATA_BYTES {
        return Err(format!("{label} must contain bounded metadata"));
    }
    let bytes = fs::read(path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value =
        serde_json::from_slice(&bytes).map_err(|error| format!("cannot parse {label}: {error}"))?;
    object(&value, label)?;
    Ok((value, sha256(&bytes)))
}

fn window_bytes(tensor: &Map<String, Value>) -> Result<usize, String> {
    let shape = array_field(tensor, "row_window_shape", "range authority tensor")?;
    let mut elements = 1_u64;
    for dimension in shape {
        elements = elements
            .checked_mul(
                dimension
                    .as_u64()
                    .filter(|value| *value > 0)
                    .ok_or("range authority tensor row_window_shape must be positive")?,
            )
            .ok_or("range authority tensor row_window_shape overflow")?;
    }
    usize::try_from(
        elements
            .checked_mul(BF16_BYTES)
            .ok_or("range authority tensor bytes overflow")?,
    )
    .map_err(|_| "range authority tensor bytes exceed platform".to_owned())
}

fn checked_relative_path(value: &str, label: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("{label} must be a non-empty relative path"));
    }
    for component in path.components() {
        if !matches!(component, Component::Normal(_)) {
            return Err(format!("{label} may not contain traversal components"));
        }
    }
    Ok(())
}

fn positive_shape(tensor: &Map<String, Value>, key: &str, label: &str) -> Result<Vec<u64>, String> {
    let values = array_field(tensor, key, label)?;
    if values.is_empty() {
        return Err(format!("{label}.{key} must be non-empty"));
    }
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("{label}.{key} must contain positive dimensions"))
        })
        .collect()
}

fn validate_range(
    document: &Value,
    document_sha256: String,
    expected_shards: u64,
    expected_tensors: u64,
) -> Result<Metadata, String> {
    let envelope = object(document, "range authority envelope")?;
    let authority = object_field(envelope, "authority", "range authority envelope")?;
    let authority_value = Value::Object(authority.clone());
    let content_hash = hash_field(
        envelope,
        "authority_content_sha256",
        "range authority envelope",
    )?;
    if content_hash != sha256(canonical(&authority_value)?.as_bytes()) {
        return Err("range authority content hash mismatch".into());
    }
    exact_text(authority, "schema", RANGE_SCHEMA, "range authority")?;
    exact_text(authority, "status", RANGE_STATUS, "range authority")?;
    let source = object_field(authority, "source", "range authority")?;
    exact_text(source, "model_id", MODEL_ID, "range authority.source")?;
    if u64_field(source, "source_shard_count", "range authority.source")? != expected_shards
        || u64_field(source, "source_tensor_count", "range authority.source")? != expected_tensors
    {
        return Err("range authority source geometry drifted".into());
    }
    let revision = text(source, "source_revision", "range authority.source")?.to_owned();
    let index = object_field(source, "source_index", "range authority.source")?;
    let index_sha256 = hash_field(index, "sha256", "range authority.source.index")?;
    let index_relative_path =
        text(index, "relative_path", "range authority.source.index")?.to_owned();
    checked_relative_path(
        &index_relative_path,
        "range authority.source.index.relative_path",
    )?;
    let boundary = object_field(authority, "metadata_access_boundary", "range authority")?;
    if u64_field(
        boundary,
        "source_tensor_payload_bytes_read",
        "range authority boundary",
    )? != 0
    {
        return Err("range authority must not read tensor payload bytes".into());
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
        required_bool(boundary, key, false, "range authority boundary")?;
    }
    let shards = array_field(authority, "shards", "range authority")?;
    let tensors = array_field(authority, "tensors", "range authority")?;
    if shards.len() as u64 != expected_shards || tensors.len() as u64 != expected_tensors {
        return Err("range authority array geometry drifted".into());
    }
    let mut shard_ranges = Vec::with_capacity(shards.len());
    let mut shard_bytes = BTreeMap::new();
    for shard in shards {
        let shard = object(shard, "range authority shard")?;
        let relative_path = text(shard, "relative_path", "range authority shard")?.to_owned();
        checked_relative_path(&relative_path, "range authority shard.relative_path")?;
        let bytes = u64_field(shard, "file_bytes", "range authority shard")?;
        if bytes == 0 {
            return Err("range authority shard has zero bytes".into());
        }
        let header = hash_field(shard, "safetensors_header_sha256", "range authority shard")?;
        if shard_bytes.insert(relative_path.clone(), bytes).is_some() {
            return Err("range authority names a duplicate shard".into());
        }
        shard_ranges.push(ShardRange {
            relative_path,
            bytes,
            safetensors_header_sha256: header,
        });
    }
    let mut tensor_ranges = Vec::with_capacity(tensors.len());
    let mut names = BTreeMap::new();
    let mut maximum_window_bytes = 0_usize;
    for tensor in tensors {
        let tensor = object(tensor, "range authority tensor")?;
        exact_text(tensor, "source_dtype", "BF16", "range authority tensor")?;
        let tensor_name = text(tensor, "tensor_name", "range authority tensor")?.to_owned();
        let shard_relative_path =
            text(tensor, "shard_relative_path", "range authority tensor")?.to_owned();
        checked_relative_path(
            &shard_relative_path,
            "range authority tensor.shard_relative_path",
        )?;
        let data_bytes = u64_field(tensor, "data_bytes", "range authority tensor")?;
        if data_bytes == 0 {
            return Err("range authority tensor has zero bytes".into());
        }
        let absolute_data_offset =
            u64_field(tensor, "absolute_data_offset", "range authority tensor")?;
        let end = absolute_data_offset
            .checked_add(data_bytes)
            .ok_or("range authority tensor range overflows")?;
        let shard_size = shard_bytes
            .get(&shard_relative_path)
            .ok_or("range authority tensor references undeclared shard")?;
        if end > *shard_size {
            return Err("range authority tensor range exceeds its shard".into());
        }
        let window = window_bytes(tensor)?;
        if window == 0 || window > MAX_WINDOW_BYTES || window as u64 > data_bytes {
            return Err("range authority tensor exceeds the <=1 MiB bootstrap bound".into());
        }
        let full_shape = positive_shape(tensor, "full_shape", "range authority tensor")?;
        if names.insert(tensor_name.clone(), ()).is_some() {
            return Err("range authority names a duplicate tensor".into());
        }
        maximum_window_bytes = maximum_window_bytes.max(window);
        tensor_ranges.push(TensorRange {
            tensor_name,
            shard_relative_path,
            full_shape,
            absolute_data_offset,
            data_bytes,
        });
    }
    Ok(Metadata {
        range_document_sha256: document_sha256,
        authority_content_sha256: content_hash,
        revision,
        index_sha256,
        maximum_window_bytes,
        index_relative_path,
        shards: shard_ranges,
        tensors: tensor_ranges,
    })
}

fn validate_semantics(
    document: &Value,
    document_sha256: String,
    metadata: &Metadata,
) -> Result<(), String> {
    let root = object(document, "semantics attester")?;
    exact_text(root, "schema", SEMANTICS_SCHEMA, "semantics attester")?;
    exact_text(root, "status", SEMANTICS_STATUS, "semantics attester")?;
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
        required_bool(boundary, key, false, "semantics boundary")?;
    }
    let source = object_field(root, "pinned_source_binding", "semantics attester")?;
    exact_text(source, "source_model_id", MODEL_ID, "semantics source")?;
    if text(source, "source_revision", "semantics source")? != metadata.revision
        || hash_field(source, "source_index_sha256", "semantics source")? != metadata.index_sha256
    {
        return Err("semantics source identity drifted".into());
    }
    let future = object_field(
        root,
        "future_exact_execution_attestation",
        "semantics attester",
    )?;
    exact_text(
        future,
        "schema",
        OPERATOR_ATTESTATION_SCHEMA,
        "semantics future",
    )?;
    exact_text(
        future,
        "status_only_after_real_separately_leased_source_execution",
        OPERATOR_ATTESTATION_STATUS,
        "semantics future",
    )?;
    if document_sha256.is_empty() {
        return Err("semantics document hash unexpectedly empty".into());
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct FixtureScanAuthority {
    run_id: String,
    root_marker_raw_sha256: String,
    operator_trace_sha256: String,
    document: SealedDocument,
}

#[derive(Clone, Debug)]
struct FixtureLease {
    lease_id: String,
    document: SealedDocument,
}

fn validate_fixture_scan_authority(
    document: SealedDocument,
    metadata: &Metadata,
    range_document_sha256: &str,
    semantics_document_sha256: &str,
) -> Result<FixtureScanAuthority, String> {
    let root = object(&document.value, "fixture scan authority")?;
    exact_text(
        root,
        "schema",
        FIXTURE_SCAN_AUTHORITY_SCHEMA,
        "fixture scan authority",
    )?;
    exact_text(
        root,
        "status",
        FIXTURE_SCAN_AUTHORITY_STATUS,
        "fixture scan authority",
    )?;
    for key in [
        "fixture_only",
        "production_adapter_forbidden",
        "fresh_one_shot_run",
        "non_inference_hash_scan_only",
    ] {
        required_bool(root, key, true, "fixture scan authority")?;
    }
    exact_text(
        root,
        "source_root_kind",
        "synthetic_fixture",
        "fixture scan authority",
    )?;
    if hash_field(
        root,
        "range_authority_document_sha256",
        "fixture scan authority",
    )? != range_document_sha256
        || hash_field(root, "semantics_document_sha256", "fixture scan authority")?
            != semantics_document_sha256
        || text(root, "source_revision", "fixture scan authority")? != metadata.revision
        || hash_field(root, "source_index_sha256", "fixture scan authority")?
            != metadata.index_sha256
        || u64_field(root, "expected_shards", "fixture scan authority")?
            != metadata.shards.len() as u64
        || u64_field(root, "expected_tensors", "fixture scan authority")?
            != metadata.tensors.len() as u64
        || u64_field(
            root,
            "maximum_positioned_read_bytes",
            "fixture scan authority",
        )? != MAX_WINDOW_BYTES as u64
        || u64_field(
            root,
            "maximum_live_raw_bf16_windows",
            "fixture scan authority",
        )? != 1
    {
        return Err("fixture scan authority binding/geometry drifted".into());
    }
    let lease = object_field(root, "fresh_bootstrap_lease", "fixture scan authority")?;
    exact_text(
        lease,
        "schema",
        BOOTSTRAP_LEASE_SCHEMA,
        "fixture scan authority lease",
    )?;
    exact_text(
        lease,
        "status",
        BOOTSTRAP_LEASE_STATUS,
        "fixture scan authority lease",
    )?;
    Ok(FixtureScanAuthority {
        run_id: hash_field(root, "fixture_run_id_sha256", "fixture scan authority")?,
        root_marker_raw_sha256: hash_field(
            root,
            "fixture_root_marker_raw_sha256",
            "fixture scan authority",
        )?,
        operator_trace_sha256: hash_field(
            root,
            "synthetic_operator_trace_sha256",
            "fixture scan authority",
        )?,
        document,
    })
}

fn validate_fixture_lease(
    document: SealedDocument,
    authority: &FixtureScanAuthority,
) -> Result<FixtureLease, String> {
    let root = object(&document.value, "fixture bootstrap lease")?;
    exact_text(
        root,
        "schema",
        BOOTSTRAP_LEASE_SCHEMA,
        "fixture bootstrap lease",
    )?;
    exact_text(
        root,
        "status",
        BOOTSTRAP_LEASE_STATUS,
        "fixture bootstrap lease",
    )?;
    for key in [
        "fresh_for_this_exact_launch",
        "one_shot",
        "non_inference_only",
        "new_capture_root_required",
        "existing_output_reuse_forbidden",
        "replay_or_relaunch_forbidden",
        "separate_from_source_teacher_lease",
        "fixture_only",
    ] {
        required_bool(root, key, true, "fixture bootstrap lease")?;
    }
    for key in [
        "source_teacher_or_logits_authorized",
        "model_gpu_server_hcli_or_tps_authorized",
        "lease_consumed_by_this_bootstrap",
    ] {
        required_bool(root, key, false, "fixture bootstrap lease")?;
    }
    if hash_field(
        root,
        "scan_authority_seal_sha256",
        "fixture bootstrap lease",
    )? != authority.document.seal_sha256
        || hash_field(root, "fixture_run_id_sha256", "fixture bootstrap lease")? != authority.run_id
    {
        return Err("fixture bootstrap lease does not bind the scan authority".into());
    }
    Ok(FixtureLease {
        lease_id: hash_field(root, "lease_id", "fixture bootstrap lease")?,
        document,
    })
}

fn validate_fixture_root(root: &Path, authority: &FixtureScanAuthority) -> Result<(), String> {
    if !root.is_absolute() {
        return Err("fixture source root must be absolute".into());
    }
    let metadata = fs::symlink_metadata(root)
        .map_err(|error| format!("cannot stat fixture source root: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_dir() {
        return Err("fixture source root must be a regular non-symlink directory".into());
    }
    let marker = root.join(".hawking-q30-bootstrap-fixture.json");
    let marker_metadata = fs::symlink_metadata(&marker)
        .map_err(|error| format!("cannot stat fixture root marker: {error}"))?;
    if marker_metadata.file_type().is_symlink()
        || !marker_metadata.file_type().is_file()
        || marker_metadata.len() == 0
        || marker_metadata.len() > MAX_METADATA_BYTES
    {
        return Err("fixture root marker must be a bounded regular non-symlink file".into());
    }
    let bytes =
        fs::read(&marker).map_err(|error| format!("cannot read fixture root marker: {error}"))?;
    if sha256(&bytes) != authority.root_marker_raw_sha256 {
        return Err("fixture root marker does not bind the fixture authority".into());
    }
    let value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse fixture root marker: {error}"))?;
    verify_seal(&value, "fixture root marker")?;
    let marker = object(&value, "fixture root marker")?;
    exact_text(
        marker,
        "schema",
        "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_fixture_root.v1",
        "fixture root marker",
    )?;
    exact_text(
        marker,
        "status",
        "READY_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_SYNTHETIC_FIXTURE_ONLY",
        "fixture root marker",
    )?;
    required_bool(marker, "fixture_only", true, "fixture root marker")?;
    required_bool(
        marker,
        "production_adapter_forbidden",
        true,
        "fixture root marker",
    )?;
    if hash_field(marker, "fixture_run_id_sha256", "fixture root marker")? != authority.run_id {
        return Err("fixture root marker run ID drifted".into());
    }
    Ok(())
}

fn future_outputs() -> Value {
    json!({
        "bootstrap_capture": {
            "schema": BOOTSTRAP_CAPTURE_SCHEMA,
            "status": BOOTSTRAP_CAPTURE_STATUS,
            "non_inference_only": true,
            "source_teacher_or_logits_allowed": false,
        },
        "flat_runtime_range_map": {
            "schema": FLAT_MAP_SCHEMA,
            "shards": PRODUCTION_SHARDS,
            "tensors": PRODUCTION_TENSORS,
            "per_shard_raw_sha256_required": true,
            "per_tensor_raw_bf16_sha256_required": true,
        },
        "operator_accumulation_attestation": {
            "schema": OPERATOR_ATTESTATION_SCHEMA,
            "status": OPERATOR_ATTESTATION_STATUS,
        },
        "range_reader_attestation": {
            "schema": READER_ATTESTATION_SCHEMA,
            "status": READER_ATTESTATION_STATUS,
        },
        "runtime_admission": {
            "schema": RUNTIME_ADMISSION_SCHEMA,
            "status": RUNTIME_ADMISSION_STATUS,
            "issued_by_distinct_controller_after_capture": true,
        },
        "compiled_synthetic_fixture_backend": {
            "scan_authority_schema": FIXTURE_SCAN_AUTHORITY_SCHEMA,
            "scan_authority_status": FIXTURE_SCAN_AUTHORITY_STATUS,
            "flat_runtime_range_map_schema": FIXTURE_FLAT_MAP_SCHEMA,
            "operator_attestation_schema": FIXTURE_OPERATOR_ATTESTATION_SCHEMA,
            "range_reader_attestation_schema": FIXTURE_READER_ATTESTATION_SCHEMA,
            "runtime_admission_schema": FIXTURE_RUNTIME_ADMISSION_SCHEMA,
            "runtime_admission_status": FIXTURE_RUNTIME_ADMISSION_STATUS,
            "production_adapter_forbidden": true,
            "source_teacher_eligibility": false,
        },
        "remaining_real_source_gate": {
            "requires_distinct_non_fixture_scan_authority_and_fresh_lease": true,
            "requires_independent_real_source_operator_semantics_attestation": true,
            "requires_production_child_to_reject_fixture_schema_statuses": true,
            "not_enabled_by_this_preflight": true,
        },
    })
}

fn command_grammar() -> Value {
    json!([
        "ascension_qwen30_streamed_source_range_admission_bootstrap",
        "--mode",
        "bootstrap-scan",
        "--range-authority",
        "ABSOLUTE_METADATA_RANGE_AUTHORITY_JSON",
        "--semantics",
        "ABSOLUTE_METADATA_SEMANTICS_JSON",
        "--bootstrap-lease",
        "ABSOLUTE_FRESH_ONE_SHOT_BOOTSTRAP_LEASE_JSON",
        "--scan-authority",
        "ABSOLUTE_SEALED_FIXTURE_ONLY_OR_FUTURE_SCAN_AUTHORITY_JSON",
        "--source-root",
        "ABSOLUTE_SEALED_SYNTHETIC_FIXTURE_ROOT_OR_FUTURE_SOURCE_ROOT",
        "--capture-dir",
        "NEW_ABSOLUTE_BOOTSTRAP_CAPTURE_DIR",
        "--out",
        "NEW_ABSOLUTE_BOOTSTRAP_CAPTURE_RECEIPT_JSON",
    ])
}

fn prepared(metadata: &Metadata) -> Result<Value, String> {
    seal(json!({
        "schema": SCHEMA,
        "status": PREPARED_STATUS,
        "prepared": true,
        "execution_authorized": false,
        "metadata_bindings": {
            "range_authority_document_sha256": metadata.range_document_sha256,
            "range_authority_content_sha256": metadata.authority_content_sha256,
            "source_revision": metadata.revision,
            "source_index_sha256": metadata.index_sha256,
            "maximum_declared_bf16_window_bytes": metadata.maximum_window_bytes,
        },
        "future_bootstrap_lease": {
            "schema": BOOTSTRAP_LEASE_SCHEMA,
            "status": BOOTSTRAP_LEASE_STATUS,
            "one_shot": true,
            "separate_from_source_teacher_lease": true,
            "non_inference_only": true,
            "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "model_server_gpu_hcli_or_tps_allowed": false,
        },
        "future_child_command": command_grammar(),
        "future_outputs_required_before_source_teacher": future_outputs(),
        "admission_order": [
            "prepared dual-attestation bridge",
            "fresh distinct bootstrap lease",
            "one bounded non-inference hash/range scan",
            "bootstrap receipt last",
            "two attestation seals and earned runtime admission",
            "unchanged source-teacher admission-before-open validation",
        ],
        "execution_boundary": {
            "source_root_opened_or_statted": false,
            "source_tensor_payload_opened": false,
            "flat_runtime_range_map_emitted": false,
            "two_attestations_emitted": false,
            "runtime_admission_earned": false,
            "source_teacher_started": false,
            "model_gpu_server_hcli_or_tps_action": false,
            "lease_issued_or_consumed": false,
        },
        "claim_boundary": "Prepared CPU/build reservation only; it does not fabricate flat-map/hash/attestation/runtime-admission evidence or execute any source, model, GPU, server, lease, or tournament action.",
    }))
}

fn refusal(
    metadata: &Metadata,
    root: &Path,
    lease: &Path,
    capture: &Path,
) -> Result<Value, String> {
    // These are opaque future references. This function never probes them.
    seal(json!({
        "schema": SCHEMA,
        "status": REFUSED_STATUS,
        "prepared": false,
        "execution_authorized": false,
        "future_references": {
            "source_root": root,
            "bootstrap_lease": lease,
            "capture_dir": capture,
        },
        "metadata_bindings": {
            "range_authority_document_sha256": metadata.range_document_sha256,
            "range_authority_content_sha256": metadata.authority_content_sha256,
        },
        "future_child_command": command_grammar(),
        "future_outputs_required_before_source_teacher": future_outputs(),
        "refusal_reason": "CPU/build target has no real bootstrap scan; refused before any source-root or lease operation",
        "execution_boundary": {
            "source_root_opened_or_statted": false,
            "source_tensor_payload_opened": false,
            "flat_runtime_range_map_emitted": false,
            "two_attestations_emitted": false,
            "runtime_admission_earned": false,
            "source_teacher_started": false,
            "model_gpu_server_hcli_or_tps_action": false,
            "lease_issued_or_consumed": false,
            "capture_directory_created": false,
        },
    }))
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() || !path.parent().is_some_and(Path::is_dir) || path.exists() {
        return Err("--out must be a new absolute path below an existing parent".into());
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create --out: {error}"))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync --out: {error}"))
}

fn run(args: Args) -> Result<Value, String> {
    let (range, range_sha) = read_metadata(&args.range_authority, "range authority")?;
    let metadata = validate_range(
        &range,
        range_sha.clone(),
        PRODUCTION_SHARDS,
        PRODUCTION_TENSORS,
    )?;
    let (semantics, semantics_sha) = read_metadata(&args.semantics, "semantics attester")?;
    validate_semantics(&semantics, semantics_sha.clone(), &metadata)?;
    match args.mode {
        Mode::Preflight => {
            let output = prepared(&metadata)?;
            write_new(&args.out, &output)?;
            Ok(output)
        }
        Mode::BootstrapScan => {
            execute_fixture_bootstrap(&args, &metadata, &range_sha, &semantics_sha)
        }
    }
}

#[derive(Debug)]
struct FileBackedBoundedReader {
    maximum: usize,
    maximum_observed: usize,
    maximum_live_windows: usize,
    reads: usize,
    zeroed_after_visit: bool,
}

impl FileBackedBoundedReader {
    fn new(maximum: usize) -> Result<Self, String> {
        if maximum == 0 || maximum > MAX_WINDOW_BYTES {
            return Err("bounded reader must use 1..=1 MiB windows".into());
        }
        Ok(Self {
            maximum,
            maximum_observed: 0,
            maximum_live_windows: 0,
            reads: 0,
            zeroed_after_visit: true,
        })
    }

    #[cfg(unix)]
    fn visit_at(
        &mut self,
        file: &File,
        offset: u64,
        length: usize,
        visitor: impl FnOnce(&[u8]) -> Result<(), String>,
    ) -> Result<(), String> {
        if length == 0 || length > self.maximum {
            return Err("positioned read exceeds bounded window".into());
        }
        let mut bytes = vec![0_u8; length];
        let mut cursor = 0_usize;
        while cursor < bytes.len() {
            let at = offset
                .checked_add(u64::try_from(cursor).map_err(|_| "offset overflow")?)
                .ok_or("offset overflow")?;
            let count = file
                .read_at(&mut bytes[cursor..], at)
                .map_err(|error| format!("positioned read failed: {error}"))?;
            if count == 0 {
                bytes.fill(0);
                return Err("positioned read hit EOF".into());
            }
            cursor += count;
        }
        self.maximum_observed = self.maximum_observed.max(bytes.len());
        self.maximum_live_windows = self.maximum_live_windows.max(1);
        self.reads += 1;
        self.zeroed_after_visit = false;
        let result = visitor(&bytes);
        bytes.fill(0);
        self.zeroed_after_visit = true;
        result
    }

    #[cfg(unix)]
    fn hash_range(&mut self, file: &File, offset: u64, bytes: u64) -> Result<String, String> {
        if bytes == 0 {
            return Err("positioned hash range must be non-empty".into());
        }
        let mut remaining = bytes;
        let mut cursor = offset;
        let mut digest = Sha256::new();
        while remaining > 0 {
            let length = usize::try_from(remaining.min(self.maximum as u64))
                .map_err(|_| "positioned hash length does not fit usize")?;
            self.visit_at(file, cursor, length, |window| {
                digest.update(window);
                Ok(())
            })?;
            cursor = cursor
                .checked_add(length as u64)
                .ok_or("positioned hash offset overflow")?;
            remaining -= length as u64;
        }
        Ok(format!("{:x}", digest.finalize()))
    }
}

#[derive(Clone, Debug)]
struct ScanHashes {
    shard_raw_sha256: BTreeMap<String, String>,
    tensor_raw_bf16_sha256: BTreeMap<String, String>,
    maximum_observed_window_bytes: usize,
    maximum_live_windows: usize,
    positioned_read_calls: usize,
}

#[derive(Clone, Debug)]
struct WrittenEvidence {
    path: PathBuf,
    raw_document_sha256: String,
    canonical_document_sha256: String,
    seal_sha256: String,
}

fn evidence_of_written(document: &WrittenEvidence) -> Value {
    json!({
        "path": document.path,
        "raw_document_sha256": document.raw_document_sha256,
        "canonical_document_sha256": document.canonical_document_sha256,
        "seal_sha256": document.seal_sha256,
    })
}

fn validate_new_output_path(path: &Path, label: &str) -> Result<(), String> {
    if !path.is_absolute() || path.exists() || !path.parent().is_some_and(Path::is_dir) {
        return Err(format!(
            "{label} must be a new absolute path below an existing parent"
        ));
    }
    Ok(())
}

fn write_new_with_evidence(path: &Path, value: &Value) -> Result<WrittenEvidence, String> {
    validate_new_output_path(path, "output")?;
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|error| error.to_string())?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create output: {error}"))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync output: {error}"))?;
    let root = object(value, "written document")?;
    Ok(WrittenEvidence {
        path: path.to_owned(),
        raw_document_sha256: sha256(&bytes),
        canonical_document_sha256: sha256(canonical(value)?.as_bytes()),
        seal_sha256: hash_field(root, "seal_sha256", "written document")?,
    })
}

fn create_new_capture_dir(path: &Path) -> Result<(), String> {
    if !path.is_absolute() || path.exists() || !path.parent().is_some_and(Path::is_dir) {
        return Err(
            "capture directory must be a new absolute path below an existing parent".into(),
        );
    }
    fs::create_dir(path).map_err(|error| format!("cannot create capture directory: {error}"))
}

fn single_component_shard_path(root: &Path, relative_path: &str) -> Result<PathBuf, String> {
    checked_relative_path(relative_path, "fixture shard relative path")?;
    if Path::new(relative_path).components().count() != 1 {
        return Err("fixture shard paths must be a single regular filename".into());
    }
    let path = root.join(relative_path);
    let metadata = fs::symlink_metadata(&path)
        .map_err(|error| format!("cannot stat fixture shard {relative_path}: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err("fixture shard must be a regular non-symlink file".into());
    }
    Ok(path)
}

#[cfg(unix)]
fn scan_fixture_ranges(root: &Path, metadata: &Metadata) -> Result<ScanHashes, String> {
    let mut reader = FileBackedBoundedReader::new(metadata.maximum_window_bytes)?;
    let mut shard_raw_sha256 = BTreeMap::new();
    let mut tensor_raw_bf16_sha256 = BTreeMap::new();
    for shard in &metadata.shards {
        let path = single_component_shard_path(root, &shard.relative_path)?;
        let file_metadata = fs::metadata(&path)
            .map_err(|error| format!("cannot inspect fixture shard: {error}"))?;
        if file_metadata.len() != shard.bytes {
            return Err("fixture shard byte count differs from the metadata authority".into());
        }
        let file =
            File::open(&path).map_err(|error| format!("cannot open fixture shard: {error}"))?;
        let raw_sha256 = reader.hash_range(&file, 0, shard.bytes)?;
        shard_raw_sha256.insert(shard.relative_path.clone(), raw_sha256);
        for tensor in metadata
            .tensors
            .iter()
            .filter(|tensor| tensor.shard_relative_path == shard.relative_path)
        {
            let end = tensor
                .absolute_data_offset
                .checked_add(tensor.data_bytes)
                .ok_or("fixture tensor range overflows")?;
            if end > shard.bytes {
                return Err("fixture tensor range exceeds its shard".into());
            }
            let hash = reader.hash_range(&file, tensor.absolute_data_offset, tensor.data_bytes)?;
            if tensor_raw_bf16_sha256
                .insert(tensor.tensor_name.clone(), hash)
                .is_some()
            {
                return Err("fixture tensor range was visited more than once".into());
            }
        }
        // `file` drops here before the next shard; no shard cache is retained.
    }
    if shard_raw_sha256.len() != metadata.shards.len()
        || tensor_raw_bf16_sha256.len() != metadata.tensors.len()
        || !reader.zeroed_after_visit
        || reader.maximum_live_windows != 1
        || reader.maximum_observed > MAX_WINDOW_BYTES
    {
        return Err("bounded fixture reader accounting drifted".into());
    }
    Ok(ScanHashes {
        shard_raw_sha256,
        tensor_raw_bf16_sha256,
        maximum_observed_window_bytes: reader.maximum_observed,
        maximum_live_windows: reader.maximum_live_windows,
        positioned_read_calls: reader.reads,
    })
}

fn fixture_flat_map(
    metadata: &Metadata,
    hashes: &ScanHashes,
    authority: &FixtureScanAuthority,
    lease: &FixtureLease,
) -> Result<Value, String> {
    let mut shard_ids = BTreeMap::new();
    let shards = metadata
        .shards
        .iter()
        .enumerate()
        .map(|(index, shard)| {
            let shard_id = format!("fixture-shard-{index:02}");
            shard_ids.insert(shard.relative_path.clone(), shard_id.clone());
            json!({
                "shard_id": shard_id,
                "relative_path": shard.relative_path,
                "bytes": shard.bytes,
                "sha256": hashes.shard_raw_sha256.get(&shard.relative_path).expect("scan covers every shard"),
                "safetensors_header_sha256": shard.safetensors_header_sha256,
            })
        })
        .collect::<Vec<_>>();
    let tensors = metadata
        .tensors
        .iter()
        .map(|tensor| {
            json!({
                "tensor_name": tensor.tensor_name,
                "shard_id": shard_ids.get(&tensor.shard_relative_path).expect("metadata shard exists"),
                "dtype": "BF16",
                "shape": tensor.full_shape,
                "data_offset": tensor.absolute_data_offset,
                "data_bytes": tensor.data_bytes,
                "raw_bf16_sha256": hashes.tensor_raw_bf16_sha256.get(&tensor.tensor_name).expect("scan covers every tensor"),
            })
        })
        .collect::<Vec<_>>();
    seal(json!({
        "schema": FIXTURE_FLAT_MAP_SCHEMA,
        "status": "CAPTURED_QWEN30_SOURCE_BF16_RANGE_MAP_SYNTHETIC_FIXTURE_ONLY",
        "fixture_only": true,
        "production_adapter_forbidden": true,
        "source_model_id": MODEL_ID,
        "source_revision": metadata.revision,
        "source_tensor_count": metadata.tensors.len(),
        "source_index": {
            "relative_path": metadata.index_relative_path,
            "sha256": metadata.index_sha256,
            "format": "huggingface.safetensors.index.json",
        },
        "maximum_window_bytes": metadata.maximum_window_bytes,
        "shards": shards,
        "tensors": tensors,
        "fixture_run_id_sha256": authority.run_id,
        "fixture_scan_authority": evidence(&authority.document),
        "fixture_bootstrap_lease": evidence(&lease.document),
        "bounded_reader": {
            "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
            "maximum_observed_window_bytes": hashes.maximum_observed_window_bytes,
            "maximum_live_raw_bf16_windows": hashes.maximum_live_windows,
            "positioned_read_calls": hashes.positioned_read_calls,
            "mmap_or_memory_map_used": false,
            "whole_shard_cached": false,
        },
    }))
}

fn fixture_operator_attestation(
    authority: &FixtureScanAuthority,
    lease: &FixtureLease,
    flat: &WrittenEvidence,
) -> Result<Value, String> {
    seal(json!({
        "schema": FIXTURE_OPERATOR_ATTESTATION_SCHEMA,
        "status": FIXTURE_OPERATOR_ATTESTATION_STATUS,
        "fixture_only": true,
        "production_adapter_forbidden": true,
        "fixture_run_id_sha256": authority.run_id,
        "synthetic_operator_trace_sha256": authority.operator_trace_sha256,
        "flat_runtime_range_map": evidence_of_written(flat),
        "fixture_scan_authority": evidence(&authority.document),
        "fixture_bootstrap_lease": evidence(&lease.document),
        "non_inference_hash_scan_only": true,
        "source_teacher_or_native_execution": false,
        "does_not_establish_real_source_operator_semantics": true,
    }))
}

fn fixture_reader_attestation(
    authority: &FixtureScanAuthority,
    lease: &FixtureLease,
    flat: &WrittenEvidence,
    hashes: &ScanHashes,
) -> Result<Value, String> {
    seal(json!({
        "schema": FIXTURE_READER_ATTESTATION_SCHEMA,
        "status": FIXTURE_READER_ATTESTATION_STATUS,
        "fixture_only": true,
        "production_adapter_forbidden": true,
        "fixture_run_id_sha256": authority.run_id,
        "flat_runtime_range_map": evidence_of_written(flat),
        "fixture_scan_authority": evidence(&authority.document),
        "fixture_bootstrap_lease": evidence(&lease.document),
        "bounded_positioned_reader": {
            "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
            "maximum_observed_window_bytes": hashes.maximum_observed_window_bytes,
            "maximum_live_raw_bf16_windows": hashes.maximum_live_windows,
            "positioned_read_calls": hashes.positioned_read_calls,
            "all_declared_shards_hashed": true,
            "all_declared_bf16_tensor_ranges_hashed": true,
            "mmap_or_memory_map_used": false,
            "whole_shard_or_model_cached": false,
            "reader_window_zeroed_after_each_visit": true,
        },
    }))
}

fn fixture_runtime_admission(
    authority: &FixtureScanAuthority,
    lease: &FixtureLease,
    flat: &WrittenEvidence,
    operator: &WrittenEvidence,
    reader: &WrittenEvidence,
) -> Result<Value, String> {
    seal(json!({
        "schema": FIXTURE_RUNTIME_ADMISSION_SCHEMA,
        "status": FIXTURE_RUNTIME_ADMISSION_STATUS,
        "fixture_only": true,
        "production_adapter_forbidden": true,
        "eligible_for_production_source_teacher": false,
        "fixture_runtime_admission_earned": true,
        "fixture_run_id_sha256": authority.run_id,
        "flat_runtime_range_map": evidence_of_written(flat),
        "operator_accumulation_fixture_attestation": evidence_of_written(operator),
        "range_reader_fixture_attestation": evidence_of_written(reader),
        "fixture_scan_authority": evidence(&authority.document),
        "fixture_bootstrap_lease": evidence(&lease.document),
        "bounded_positioned_reader": {
            "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "no_mmap_or_full_shard_cache": true,
            "no_model_residency": true,
        },
        "execution_boundary": {
            "source_teacher_started": false,
            "native_phase_started": false,
            "source_model_loaded": false,
            "gpu_server_hcli_or_tps_action": false,
        },
    }))
}

fn fixture_capture(
    authority: &FixtureScanAuthority,
    lease: &FixtureLease,
    reservation: &WrittenEvidence,
    flat: &WrittenEvidence,
    operator: &WrittenEvidence,
    reader: &WrittenEvidence,
    runtime: &WrittenEvidence,
    hashes: &ScanHashes,
) -> Result<Value, String> {
    seal(json!({
        "schema": FIXTURE_CAPTURE_SCHEMA,
        "status": FIXTURE_CAPTURE_STATUS,
        "fixture_only": true,
        "production_adapter_forbidden": true,
        "fixture_run_id_sha256": authority.run_id,
        "fixture_bootstrap_lease": evidence(&lease.document),
        "fixture_scan_authority": evidence(&authority.document),
        "replay_reservation": evidence_of_written(reservation),
        "flat_runtime_range_map": evidence_of_written(flat),
        "operator_accumulation_fixture_attestation": evidence_of_written(operator),
        "range_reader_fixture_attestation": evidence_of_written(reader),
        "fixture_runtime_admission": evidence_of_written(runtime),
        "non_inference_only": true,
        "one_bounded_window": true,
        "flat_runtime_range_map_emitted": true,
        "operator_attestation_emitted": true,
        "range_reader_attestation_emitted": true,
        "fixture_runtime_admission_earned": true,
        "receipt_written_last": true,
        "source_teacher_started": false,
        "native_phase_started": false,
        "logits_or_vectors_written": false,
        "source_model_loaded": false,
        "gpu_server_hcli_or_tps_action": false,
        "bounded_reader": {
            "maximum_observed_window_bytes": hashes.maximum_observed_window_bytes,
            "maximum_live_raw_bf16_windows": hashes.maximum_live_windows,
            "positioned_read_calls": hashes.positioned_read_calls,
            "all_handles_closed_before_capture_receipt": true,
            "cache_zeroed_before_capture_receipt": true,
        },
        "claim_boundary": "Synthetic fixture mechanics only. This capture cannot be adapted into the production Q30 source-teacher, native, GPU/server, HCLI, TPS/TG, or tournament authority chain.",
    }))
}

#[cfg(unix)]
fn execute_fixture_bootstrap(
    args: &Args,
    metadata: &Metadata,
    range_document_sha256: &str,
    semantics_document_sha256: &str,
) -> Result<Value, String> {
    let authority = validate_fixture_scan_authority(
        read_sealed_document(
            args.scan_authority
                .as_deref()
                .expect("validated scan authority"),
            "fixture scan authority",
        )?,
        metadata,
        range_document_sha256,
        semantics_document_sha256,
    )?;
    let lease = validate_fixture_lease(
        read_sealed_document(
            args.bootstrap_lease
                .as_deref()
                .expect("validated bootstrap lease"),
            "fixture bootstrap lease",
        )?,
        &authority,
    )?;
    let root = args.source_root.as_deref().expect("validated source root");
    validate_fixture_root(root, &authority)?;
    let capture_dir = args.capture_dir.as_deref().expect("validated capture dir");
    validate_new_output_path(&args.out, "capture receipt")?;
    create_new_capture_dir(capture_dir)?;
    let reservation = seal(json!({
        "schema": "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_fixture_replay_reservation.v1",
        "status": "RESERVED_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_SYNTHETIC_FIXTURE_ONE_SHOT",
        "fixture_only": true,
        "production_adapter_forbidden": true,
        "create_new_before_payload_reads": true,
        "one_child_maximum": true,
        "replay_or_relaunch_forbidden": true,
        "attempt": 1,
        "fixture_run_id_sha256": authority.run_id,
        "lease_id": lease.lease_id,
        "fixture_scan_authority": evidence(&authority.document),
        "fixture_bootstrap_lease": evidence(&lease.document),
    }))?;
    let reservation =
        write_new_with_evidence(&capture_dir.join("replay-reservation.json"), &reservation)?;
    let hashes = scan_fixture_ranges(root, metadata)?;
    let flat = fixture_flat_map(metadata, &hashes, &authority, &lease)?;
    let flat = write_new_with_evidence(&capture_dir.join("flat-runtime-range-map.json"), &flat)?;
    let operator = fixture_operator_attestation(&authority, &lease, &flat)?;
    let operator = write_new_with_evidence(
        &capture_dir.join("operator-accumulation-fixture-attestation.json"),
        &operator,
    )?;
    let reader = fixture_reader_attestation(&authority, &lease, &flat, &hashes)?;
    let reader = write_new_with_evidence(
        &capture_dir.join("range-reader-fixture-attestation.json"),
        &reader,
    )?;
    let runtime = fixture_runtime_admission(&authority, &lease, &flat, &operator, &reader)?;
    let runtime = write_new_with_evidence(
        &capture_dir.join("fixture-runtime-admission.json"),
        &runtime,
    )?;
    let capture = fixture_capture(
        &authority,
        &lease,
        &reservation,
        &flat,
        &operator,
        &reader,
        &runtime,
        &hashes,
    )?;
    write_new_with_evidence(&args.out, &capture)?;
    Ok(capture)
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(run) {
        Ok(output) => println!(
            "{{\"status\":\"{}\",\"seal_sha256\":\"{}\"}}",
            output["status"].as_str().unwrap_or(""),
            output["seal_sha256"].as_str().unwrap_or("")
        ),
        Err(error) => {
            eprintln!("Q30 range-admission bootstrap refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn hash(value: &str) -> String {
        sha256(value.as_bytes())
    }

    fn authority() -> Value {
        let authority = json!({
            "schema": RANGE_SCHEMA,
            "status": RANGE_STATUS,
            "source": {
                "model_id": MODEL_ID,
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
                "source_shard_count": 1,
                "source_tensor_count": 1,
                "source_index": {
                    "relative_path": "model.safetensors.index.json",
                    "sha256": hash("index"),
                },
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
            "shards": [{
                "relative_path": "one.safetensors",
                "file_bytes": 64,
                "safetensors_header_sha256": hash("header"),
            }],
            "tensors": [{
                "tensor_name": "fixture.weight",
                "source_dtype": "BF16",
                "full_shape": [2, 4],
                "shard_relative_path": "one.safetensors",
                "absolute_data_offset": 32,
                "data_bytes": 16,
                "row_window_shape": [2, 4],
            }],
        });
        json!({
            "authority_content_sha256": sha256(canonical(&authority).unwrap().as_bytes()),
            "authority": authority,
        })
    }

    fn semantics() -> Value {
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
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
                "source_index_sha256": hash("index"),
            },
            "future_exact_execution_attestation": {
                "schema": OPERATOR_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": OPERATOR_ATTESTATION_STATUS,
            },
        })
    }

    fn metadata() -> Metadata {
        validate_range(&authority(), hash("range-doc"), 1, 1).unwrap()
    }

    fn write_value(path: &Path, value: &Value) -> String {
        let bytes = serde_json::to_vec_pretty(value).unwrap();
        let mut file = File::create(path).unwrap();
        file.write_all(&bytes).unwrap();
        file.sync_all().unwrap();
        sha256(&bytes)
    }

    fn production_geometry_fixture() -> Value {
        let mut shards = Vec::new();
        let mut tensors = Vec::new();
        let mut next_tensor = 0_u64;
        for shard_index in 0..PRODUCTION_SHARDS {
            let remaining = PRODUCTION_TENSORS - next_tensor;
            let shard_count = remaining.div_ceil(PRODUCTION_SHARDS - shard_index);
            let relative_path = format!("fixture-{shard_index:02}.safetensors");
            shards.push(json!({
                "relative_path": relative_path,
                "file_bytes": shard_count * BF16_BYTES,
                "safetensors_header_sha256": hash(&format!("header-{shard_index}")),
            }));
            for local_index in 0..shard_count {
                let tensor_name = format!("fixture.tensor.{next_tensor:05}");
                tensors.push(json!({
                    "tensor_name": tensor_name,
                    "source_dtype": "BF16",
                    "full_shape": [1],
                    "row_window_shape": [1],
                    "shard_relative_path": relative_path,
                    "absolute_data_offset": local_index * BF16_BYTES,
                    "data_bytes": BF16_BYTES,
                }));
                next_tensor += 1;
            }
        }
        assert_eq!(next_tensor, PRODUCTION_TENSORS);
        let authority = json!({
            "schema": RANGE_SCHEMA,
            "status": RANGE_STATUS,
            "source": {
                "model_id": MODEL_ID,
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
                "source_shard_count": PRODUCTION_SHARDS,
                "source_tensor_count": PRODUCTION_TENSORS,
                "source_index": {
                    "relative_path": "model.safetensors.index.json",
                    "sha256": hash("fixture-index"),
                },
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
            "authority_content_sha256": sha256(canonical(&authority).unwrap().as_bytes()),
            "authority": authority,
        })
    }

    fn production_geometry_semantics() -> Value {
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
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
                "source_index_sha256": hash("fixture-index"),
            },
            "future_exact_execution_attestation": {
                "schema": OPERATOR_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": OPERATOR_ATTESTATION_STATUS,
            },
        })
    }

    fn sealed_fixture_authority(
        metadata: &Metadata,
        range_document_sha256: &str,
        semantics_document_sha256: &str,
        run_id: &str,
        marker_raw_sha256: &str,
    ) -> Value {
        seal(json!({
            "schema": FIXTURE_SCAN_AUTHORITY_SCHEMA,
            "status": FIXTURE_SCAN_AUTHORITY_STATUS,
            "fixture_only": true,
            "production_adapter_forbidden": true,
            "fresh_one_shot_run": true,
            "non_inference_hash_scan_only": true,
            "source_root_kind": "synthetic_fixture",
            "fixture_run_id_sha256": run_id,
            "fixture_root_marker_raw_sha256": marker_raw_sha256,
            "synthetic_operator_trace_sha256": hash("fixture-operator-trace"),
            "range_authority_document_sha256": range_document_sha256,
            "semantics_document_sha256": semantics_document_sha256,
            "source_revision": metadata.revision,
            "source_index_sha256": metadata.index_sha256,
            "expected_shards": metadata.shards.len(),
            "expected_tensors": metadata.tensors.len(),
            "maximum_positioned_read_bytes": MAX_WINDOW_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "fresh_bootstrap_lease": {
                "schema": BOOTSTRAP_LEASE_SCHEMA,
                "status": BOOTSTRAP_LEASE_STATUS,
            },
        }))
        .unwrap()
    }

    fn sealed_fixture_lease(authority: &Value, run_id: &str) -> Value {
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
            "fixture_only": true,
            "source_teacher_or_logits_authorized": false,
            "model_gpu_server_hcli_or_tps_authorized": false,
            "lease_consumed_by_this_bootstrap": false,
            "scan_authority_seal_sha256": authority["seal_sha256"],
            "fixture_run_id_sha256": run_id,
            "lease_id": hash("fixture-lease"),
        }))
        .unwrap()
    }

    #[test]
    fn prepared_contract_reserves_outputs_but_never_earns_them() {
        let output = prepared(&metadata()).unwrap();
        assert_eq!(output["schema"], SCHEMA);
        assert_eq!(output["status"], PREPARED_STATUS);
        assert_eq!(output["execution_authorized"], false);
        assert_eq!(
            output["future_outputs_required_before_source_teacher"]["runtime_admission"]["status"],
            RUNTIME_ADMISSION_STATUS
        );
        assert_eq!(
            output["execution_boundary"]["runtime_admission_earned"],
            false
        );
        verify_seal(&output, "prepared").unwrap();
        validate_semantics(&semantics(), hash("semantics-doc"), &metadata()).unwrap();
    }

    #[test]
    fn synthetic_file_backed_reader_enforces_one_window_and_zeroes_it() {
        let temp = TempDir::new().unwrap();
        let path = temp.path().join("synthetic.shard");
        let source = (0..4_096)
            .map(|value| (value % 251) as u8)
            .collect::<Vec<_>>();
        let mut writer = File::create(&path).unwrap();
        writer.write_all(&source).unwrap();
        writer.sync_all().unwrap();
        let file = File::open(&path).unwrap();
        let mut reader = FileBackedBoundedReader::new(97).unwrap();
        let mut digest = Sha256::new();
        for offset in (0..source.len()).step_by(97) {
            let length = (source.len() - offset).min(97);
            reader
                .visit_at(&file, offset as u64, length, |window| {
                    digest.update(window);
                    Ok(())
                })
                .unwrap();
            assert!(reader.zeroed_after_visit);
        }
        assert_eq!(format!("{:x}", digest.finalize()), sha256(&source));
        assert!(reader.maximum_observed <= 97);
        assert_eq!(reader.maximum_live_windows, 1);
        assert!(reader.reads > 1);
        assert!(reader
            .visit_at(&file, 0, MAX_WINDOW_BYTES + 1, |_| Ok(()))
            .unwrap_err()
            .contains("bounded window"));
    }

    #[test]
    fn sealed_fixture_authority_scans_all_16_shards_and_18867_ranges_receipt_last() {
        let temporary = TempDir::new().unwrap();
        let root = temporary.path().join("fixture-root");
        fs::create_dir(&root).unwrap();
        let run_id = hash("fixture-run");
        let marker = seal(json!({
            "schema": "hawking.ascension.qwen30_streamed_source_range_admission_bootstrap_fixture_root.v1",
            "status": "READY_QWEN30_STREAMED_SOURCE_RANGE_ADMISSION_BOOTSTRAP_SYNTHETIC_FIXTURE_ONLY",
            "fixture_only": true,
            "production_adapter_forbidden": true,
            "fixture_run_id_sha256": run_id,
        }))
        .unwrap();
        let marker_sha = write_value(&root.join(".hawking-q30-bootstrap-fixture.json"), &marker);

        let range = production_geometry_fixture();
        let range_path = temporary.path().join("range.json");
        let range_sha = write_value(&range_path, &range);
        let metadata = validate_range(
            &range,
            range_sha.clone(),
            PRODUCTION_SHARDS,
            PRODUCTION_TENSORS,
        )
        .expect("production-geometry fixture authority validates");
        let semantics = production_geometry_semantics();
        let semantics_path = temporary.path().join("semantics.json");
        let semantics_sha = write_value(&semantics_path, &semantics);
        validate_semantics(&semantics, semantics_sha.clone(), &metadata).unwrap();

        for (shard_index, shard) in metadata.shards.iter().enumerate() {
            let mut bytes = vec![0_u8; usize::try_from(shard.bytes).unwrap()];
            for (index, byte) in bytes.iter_mut().enumerate() {
                *byte = ((shard_index * 17 + index) % 251) as u8;
            }
            let mut file = File::create(root.join(&shard.relative_path)).unwrap();
            file.write_all(&bytes).unwrap();
            file.sync_all().unwrap();
        }

        let authority =
            sealed_fixture_authority(&metadata, &range_sha, &semantics_sha, &run_id, &marker_sha);
        let authority_path = temporary.path().join("fixture-authority.json");
        write_value(&authority_path, &authority);
        let lease = sealed_fixture_lease(&authority, &run_id);
        let lease_path = temporary.path().join("fixture-lease.json");
        write_value(&lease_path, &lease);

        let args = Args {
            mode: Mode::BootstrapScan,
            range_authority: range_path,
            semantics: semantics_path,
            bootstrap_lease: Some(lease_path),
            scan_authority: Some(authority_path),
            source_root: Some(root),
            capture_dir: Some(temporary.path().join("capture")),
            out: temporary.path().join("capture-receipt.json"),
        };
        let output = execute_fixture_bootstrap(&args, &metadata, &range_sha, &semantics_sha)
            .expect("fixture-only scan succeeds");
        assert_eq!(output["schema"], FIXTURE_CAPTURE_SCHEMA);
        assert_eq!(output["status"], FIXTURE_CAPTURE_STATUS);
        assert_eq!(output["fixture_runtime_admission_earned"], true);
        assert_eq!(output["production_adapter_forbidden"], true);
        assert_eq!(output["bounded_reader"]["maximum_live_raw_bf16_windows"], 1);
        assert_eq!(
            output["bounded_reader"]["positioned_read_calls"],
            PRODUCTION_TENSORS * 2
        );
        verify_seal(&output, "fixture capture").unwrap();

        let flat_path = temporary.path().join("capture/flat-runtime-range-map.json");
        let flat: Value = serde_json::from_slice(&fs::read(&flat_path).unwrap()).unwrap();
        verify_seal(&flat, "fixture flat map").unwrap();
        assert_eq!(flat["schema"], FIXTURE_FLAT_MAP_SCHEMA);
        assert_ne!(flat["schema"], FLAT_MAP_SCHEMA);
        assert_eq!(
            flat["shards"].as_array().unwrap().len(),
            PRODUCTION_SHARDS as usize
        );
        assert_eq!(
            flat["tensors"].as_array().unwrap().len(),
            PRODUCTION_TENSORS as usize
        );

        let runtime_path = temporary
            .path()
            .join("capture/fixture-runtime-admission.json");
        let runtime: Value = serde_json::from_slice(&fs::read(runtime_path).unwrap()).unwrap();
        verify_seal(&runtime, "fixture runtime admission").unwrap();
        assert_eq!(runtime["schema"], FIXTURE_RUNTIME_ADMISSION_SCHEMA);
        assert_ne!(runtime["schema"], RUNTIME_ADMISSION_SCHEMA);
        assert_eq!(runtime["eligible_for_production_source_teacher"], false);
        assert!(temporary.path().join("capture-receipt.json").is_file());
    }

    #[test]
    fn bootstrap_mode_refuses_without_probing_future_source_paths() {
        let output = refusal(
            &metadata(),
            Path::new("/definitely/not/a/source-root"),
            Path::new("/definitely/not/a/bootstrap-lease.json"),
            Path::new("/definitely/not/a/capture-dir"),
        )
        .unwrap();
        assert_eq!(output["status"], REFUSED_STATUS);
        assert_eq!(
            output["execution_boundary"]["source_root_opened_or_statted"],
            false
        );
        assert_eq!(
            output["execution_boundary"]["source_tensor_payload_opened"],
            false
        );
        assert_eq!(
            output["execution_boundary"]["lease_issued_or_consumed"],
            false
        );
        verify_seal(&output, "refusal").unwrap();
    }

    #[test]
    fn fixture_lease_authority_drift_refuses_before_any_fixture_root_probe() {
        let metadata = metadata();
        let run_id = hash("fixture-run");
        let authority_value = sealed_fixture_authority(
            &metadata,
            &hash("range"),
            &hash("semantics"),
            &run_id,
            &hash("marker"),
        );
        let authority = validate_fixture_scan_authority(
            SealedDocument {
                raw_document_sha256: hash("authority-raw"),
                seal_sha256: authority_value["seal_sha256"].as_str().unwrap().to_owned(),
                value: authority_value.clone(),
            },
            &metadata,
            &hash("range"),
            &hash("semantics"),
        )
        .unwrap();
        let mut lease_value = sealed_fixture_lease(&authority_value, &run_id);
        lease_value["scan_authority_seal_sha256"] = json!(hash("substituted-authority"));
        let lease_value = seal(lease_value).unwrap();
        let error = validate_fixture_lease(
            SealedDocument {
                raw_document_sha256: hash("lease-raw"),
                seal_sha256: lease_value["seal_sha256"].as_str().unwrap().to_owned(),
                value: lease_value,
            },
            &authority,
        )
        .unwrap_err();
        assert!(error.contains("does not bind"));
    }

    #[test]
    fn malformed_metadata_fails_before_any_bootstrap_scan() {
        let mut bad = authority();
        bad["authority"]["tensors"][0]["row_window_shape"] = json!([1024, 1025]);
        bad["authority_content_sha256"] =
            json!(sha256(canonical(&bad["authority"]).unwrap().as_bytes()));
        assert!(validate_range(&bad, hash("range-doc"), 1, 1)
            .unwrap_err()
            .contains("<=1 MiB"));

        let error = parse_args([
            "--mode".into(),
            "bootstrap-scan".into(),
            "--range-authority".into(),
            "/tmp/range.json".into(),
            "--semantics".into(),
            "/tmp/semantics.json".into(),
            "--out".into(),
            "/tmp/out.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("--bootstrap-lease"));
    }
}
