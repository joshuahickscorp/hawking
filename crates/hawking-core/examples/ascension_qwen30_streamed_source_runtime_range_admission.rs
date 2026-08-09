#![allow(dead_code)] // The only executable surface is a metadata-only preflight.

//! CPU/build-only Qwen30 runtime range-admission producer/validator.
//!
//! The existing metadata range authority deliberately records header-derived
//! offsets but no source-payload hashes.  The later bounded reader needs a
//! separate flat map with a raw SHA-256 for every shard and BF16 tensor range.
//! This target validates the first document, describes the exact second
//! document, and seals that description.  It never opens a source root or a
//! safetensors shard.  A supplied source root is only an opaque, absolute
//! future-execution reference and is never statted, canonicalized, or read.
//!
//! Consequently every document emitted here is `PREPARED_*_NOT_EXECUTED`.
//! The future source worker must separately earn the runtime-admission and
//! both execution-attestation receipts after bounded <=1 MiB positioned reads.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process;

const SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_runtime_range_admission_producer_preflight.v1";
const STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_RUNTIME_RANGE_ADMISSION_PRODUCER_NOT_EXECUTED";

const RANGE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_oracle_metadata_only_range_map_authority.v1";
const RANGE_AUTHORITY_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_ORACLE_SOURCE_RANGE_MAP_AUTHORITY_NOT_EXECUTED";
const SEMANTICS_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_semantics_attester.v1";
const SEMANTICS_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_NOT_EXECUTED";
const OPERATOR_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_operator_accumulation_execution_attestation.v1";
const OPERATOR_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_OPERATOR_ACCUMULATION_SEMANTICS_ATTESTED";
const RANGE_READER_ATTESTATION_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_exact_semantics_attestation.v1";
const RANGE_READER_ATTESTATION_STATUS: &str =
    "EARNED_QWEN30_LAYER_STREAMED_SOURCE_BF16_EXACT_SEMANTICS_ATTESTED";
const RUNTIME_RANGE_MAP_SCHEMA: &str = "hawking.ascension.qwen30_source_bf16_range_map.v1";
const RUNTIME_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_runtime_range_admission.v1";
const RUNTIME_ADMISSION_STATUS: &str =
    "EARNED_QWEN30_STREAMED_SOURCE_TEACHER_RUNTIME_RANGE_ADMISSION_NO_MODEL_RESIDENCY";

const SOURCE_MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const SOURCE_TENSOR_COUNT: u64 = 18_867;
const SOURCE_SHARD_COUNT: u64 = 16;
const SOURCE_LAYERS: u64 = 48;
const SOURCE_FORWARDS: u64 = 370;
const PREFIX_TOKENS: u64 = 369;
const FORCED_TOKEN_ID: u64 = 949;
const TOP_K: u64 = 8;
const MAX_POSITIONED_READ_BYTES: u64 = 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;
const BF16_BYTES: u64 = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    MetadataPreflight,
    FutureSourceRootExecutionReference,
}

impl Mode {
    fn parse(value: &str) -> Result<Self, String> {
        match value {
            "metadata-preflight" => Ok(Self::MetadataPreflight),
            "future-source-root-execution-reference" => {
                Ok(Self::FutureSourceRootExecutionReference)
            }
            _ => Err(
                "--mode must be metadata-preflight or future-source-root-execution-reference"
                    .to_owned(),
            ),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::MetadataPreflight => "metadata-preflight",
            Self::FutureSourceRootExecutionReference => "future-source-root-execution-reference",
        }
    }
}

#[derive(Debug)]
struct Args {
    range_authority: PathBuf,
    semantics_attester: PathBuf,
    mode: Mode,
    future_source_root: Option<PathBuf>,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_streamed_source_runtime_range_admission \\\n+     --range-authority ABSOLUTE_METADATA_RANGE_AUTHORITY_JSON \\\n+     --semantics-attester ABSOLUTE_METADATA_SEMANTICS_JSON \\\n+     --out NEW_ABSOLUTE_PREFLIGHT_JSON \\\n+     [--mode metadata-preflight|future-source-root-execution-reference] \\\n+     [--future-source-root ABSOLUTE_FUTURE_SOURCE_ROOT]"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut range_authority = None;
    let mut semantics_attester = None;
    let mut mode = None;
    let mut future_source_root = None;
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
            "--range-authority" => {
                if range_authority.replace(PathBuf::from(next)).is_some() {
                    return Err("--range-authority was supplied more than once".to_owned());
                }
            }
            "--semantics-attester" => {
                if semantics_attester.replace(PathBuf::from(next)).is_some() {
                    return Err("--semantics-attester was supplied more than once".to_owned());
                }
            }
            "--mode" => {
                if mode.replace(Mode::parse(&next)?).is_some() {
                    return Err("--mode was supplied more than once".to_owned());
                }
            }
            "--future-source-root" => {
                if future_source_root.replace(PathBuf::from(next)).is_some() {
                    return Err("--future-source-root was supplied more than once".to_owned());
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
    let mode = mode.unwrap_or(Mode::MetadataPreflight);
    let args = Args {
        range_authority: required(range_authority, "--range-authority")?,
        semantics_attester: required(semantics_attester, "--semantics-attester")?,
        mode,
        future_source_root,
        out: required(out, "--out")?,
    };
    for (flag, path) in [
        ("--range-authority", &args.range_authority),
        ("--semantics-attester", &args.semantics_attester),
        ("--out", &args.out),
    ] {
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
    }
    match (&args.mode, &args.future_source_root) {
        (Mode::MetadataPreflight, None) => {}
        (Mode::MetadataPreflight, Some(_)) => {
            return Err(
                "--future-source-root is accepted only in future-source-root-execution-reference mode"
                    .to_owned(),
            );
        }
        (Mode::FutureSourceRootExecutionReference, Some(path)) if path.is_absolute() => {}
        (Mode::FutureSourceRootExecutionReference, Some(_)) => {
            return Err("--future-source-root must be absolute".to_owned());
        }
        (Mode::FutureSourceRootExecutionReference, None) => {
            return Err(
                "future-source-root-execution-reference mode requires --future-source-root"
                    .to_owned(),
            );
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

fn is_lower_hex(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
            .map(|items| format!("[{}]", items.join(","))),
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut items = Vec::with_capacity(keys.len());
            for key in keys {
                let rendered_key = serde_json::to_string(key)
                    .map_err(|error| format!("cannot canonicalize key: {error}"))?;
                let child = values
                    .get(key)
                    .ok_or_else(|| "canonical object key disappeared".to_owned())?;
                items.push(format!("{rendered_key}:{}", canonical_json(child)?));
            }
            Ok(format!("{{{}}}", items.join(",")))
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

fn u64_value(value: &Value, label: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{label} must be a non-negative integer"))
}

fn require_bool(value: &Value, expected: bool, label: &str) -> Result<(), String> {
    if value.as_bool() != Some(expected) {
        return Err(format!("{label} must be {expected}"));
    }
    Ok(())
}

fn require_schema_status(
    value: &Map<String, Value>,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if string(
        required(value, "schema", label)?,
        &format!("{label}.schema"),
    )? != schema
        || string(
            required(value, "status", label)?,
            &format!("{label}.status"),
        )? != status
    {
        return Err(format!("{label} schema/status drifted"));
    }
    Ok(())
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflowed"))
}

fn validate_relative_path(value: &str, label: &str) -> Result<(), String> {
    let path = Path::new(value);
    if value.is_empty() || path.is_absolute() {
        return Err(format!("{label} must be a non-empty relative path"));
    }
    if path
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!("{label} contains unsafe path components"));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct LoadedDocument {
    path: PathBuf,
    raw_sha256: String,
    bytes: u64,
    value: Value,
    seal_sha256: Option<String>,
}

fn read_metadata_document(path: &Path, label: &str) -> Result<LoadedDocument, String> {
    if !path.is_absolute() || path.extension().and_then(|item| item.to_str()) != Some("json") {
        return Err(format!(
            "{label} must be an absolute .json metadata document"
        ));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    if metadata.len() == 0 || metadata.len() > MAX_METADATA_BYTES {
        return Err(format!(
            "{label} must be 1..={MAX_METADATA_BYTES} metadata bytes"
        ));
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    File::open(path)
        .and_then(|mut file| file.read_to_end(&mut bytes))
        .map_err(|error| format!("cannot read {label} {}: {error}", path.display()))?;
    if bytes.len() as u64 != metadata.len() {
        return Err(format!("{label} changed while it was read"));
    }
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;
    let seal_sha256 = if object(&value, label)?.contains_key("seal_sha256") {
        Some(verify_seal(&value, label)?)
    } else {
        None
    };
    Ok(LoadedDocument {
        path: fs::canonicalize(path)
            .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))?,
        raw_sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
        value,
        seal_sha256,
    })
}

fn memory_document(path: &str, value: Value) -> LoadedDocument {
    let bytes = serde_json::to_vec(&value).expect("fixture JSON serializes");
    let seal_sha256 = if value
        .as_object()
        .is_some_and(|object| object.contains_key("seal_sha256"))
    {
        Some(verify_seal(&value, "fixture document").expect("fixture seal validates"))
    } else {
        None
    };
    LoadedDocument {
        path: PathBuf::from(path),
        raw_sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
        value,
        seal_sha256,
    }
}

fn document_evidence(document: &LoadedDocument) -> Value {
    let mut evidence = json!({
        "path": document.path,
        "bytes": document.bytes,
        "raw_document_sha256": document.raw_sha256,
    });
    if let Some(seal) = &document.seal_sha256 {
        evidence
            .as_object_mut()
            .expect("evidence is an object")
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
    }
    evidence
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    tensor_count: u64,
    shard_count: u64,
}

const PRODUCTION_GEOMETRY: Geometry = Geometry {
    tensor_count: SOURCE_TENSOR_COUNT,
    shard_count: SOURCE_SHARD_COUNT,
};

#[derive(Clone, Debug)]
struct MetadataShard {
    relative_path: String,
    file_bytes: u64,
    header_sha256: String,
    prefix_sha256: String,
}

#[derive(Clone, Debug)]
struct MetadataTensor {
    tensor_name: String,
    shard_relative_path: String,
    full_shape: Vec<u64>,
    row_window_shape: Vec<u64>,
    absolute_data_offset: u64,
    data_bytes: u64,
}

#[derive(Clone, Debug)]
struct MetadataBinding {
    evidence: Value,
    authority_content_sha256: String,
    source_revision: String,
    source_index_relative_path: String,
    source_index_sha256: String,
    tensors: BTreeMap<String, MetadataTensor>,
    shards: BTreeMap<String, MetadataShard>,
    maximum_window_bytes: u64,
}

fn parse_shape(value: &Value, label: &str) -> Result<Vec<u64>, String> {
    let shape = array(value, label)?;
    if shape.is_empty() {
        return Err(format!("{label} must be non-empty"));
    }
    shape
        .iter()
        .enumerate()
        .map(|(index, dimension)| {
            let dimension = u64_value(dimension, &format!("{label}[{index}]"))?;
            if dimension == 0 {
                return Err(format!("{label}[{index}] must be positive"));
            }
            Ok(dimension)
        })
        .collect()
}

fn shape_bytes(shape: &[u64], label: &str) -> Result<u64, String> {
    let elements = shape.iter().try_fold(1u64, |count, dimension| {
        checked_mul(count, *dimension, label)
    })?;
    checked_mul(elements, BF16_BYTES, label)
}

fn validate_metadata_authority(
    document: &LoadedDocument,
    geometry: Geometry,
) -> Result<MetadataBinding, String> {
    let root = object(&document.value, "metadata range authority envelope")?;
    let authority = required(root, "authority", "metadata range authority envelope")?;
    let declared_content_sha = sha256(
        required(
            root,
            "authority_content_sha256",
            "metadata range authority envelope",
        )?,
        "metadata range authority envelope.authority_content_sha256",
    )?;
    if sha256_hex(canonical_json(authority)?.as_bytes()) != declared_content_sha {
        return Err(
            "metadata range authority content hash does not bind authority material".to_owned(),
        );
    }
    let authority = object(authority, "metadata range authority")?;
    require_schema_status(
        authority,
        RANGE_AUTHORITY_SCHEMA,
        RANGE_AUTHORITY_STATUS,
        "metadata range authority",
    )?;
    let source = object(
        required(authority, "source", "metadata range authority")?,
        "metadata range authority.source",
    )?;
    if string(
        required(source, "model_id", "metadata range authority.source")?,
        "metadata range authority.source.model_id",
    )? != SOURCE_MODEL_ID
    {
        return Err("metadata range authority source model drifted".to_owned());
    }
    let source_revision = string(
        required(source, "source_revision", "metadata range authority.source")?,
        "metadata range authority.source_revision",
    )?
    .to_owned();
    if !is_lower_hex(&source_revision, 40) {
        return Err(
            "metadata range authority source revision must be exactly 40 lowercase hex characters"
                .to_owned(),
        );
    }
    if u64_value(
        required(
            source,
            "source_tensor_count",
            "metadata range authority.source",
        )?,
        "metadata range authority.source_tensor_count",
    )? != geometry.tensor_count
        || u64_value(
            required(
                source,
                "source_shard_count",
                "metadata range authority.source",
            )?,
            "metadata range authority.source_shard_count",
        )? != geometry.shard_count
    {
        return Err("metadata range authority source geometry drifted".to_owned());
    }
    let source_index = object(
        required(source, "source_index", "metadata range authority.source")?,
        "metadata range authority.source_index",
    )?;
    let source_index_relative_path = string(
        required(
            source_index,
            "relative_path",
            "metadata range authority.source.source_index",
        )?,
        "metadata range authority.source.source_index.relative_path",
    )?
    .to_owned();
    validate_relative_path(
        &source_index_relative_path,
        "metadata range authority source index relative path",
    )?;
    let source_index_sha256 = sha256(
        required(
            source_index,
            "sha256",
            "metadata range authority.source.source_index",
        )?,
        "metadata range authority.source.source_index.sha256",
    )?;
    if u64_value(
        required(
            source_index,
            "weight_map_tensor_count",
            "metadata range authority.source.source_index",
        )?,
        "metadata range authority.source.source_index.weight_map_tensor_count",
    )? != geometry.tensor_count
    {
        return Err("metadata range authority source index tensor count drifted".to_owned());
    }
    let scope = object(
        required(
            authority,
            "exact_streamed_oracle_scope",
            "metadata range authority",
        )?,
        "metadata range authority.exact_streamed_oracle_scope",
    )?;
    for (field, expected) in [
        ("source_template_token_count", PREFIX_TOKENS),
        ("forced_identical_continuation_token_id", FORCED_TOKEN_ID),
        ("total_forwards_per_replay_arm", SOURCE_FORWARDS),
        ("layers", SOURCE_LAYERS),
        ("top_k_routes_per_token", TOP_K),
    ] {
        if u64_value(
            required(scope, field, "metadata range authority scope")?,
            &format!("metadata range authority scope.{field}"),
        )? != expected
        {
            return Err(format!("metadata range authority scope {field} drifted"));
        }
    }
    require_bool(
        required(
            scope,
            "sampling_or_autoregressive_feedback_forbidden",
            "metadata range authority scope",
        )?,
        true,
        "metadata range authority sampling boundary",
    )?;
    let boundary = object(
        required(
            authority,
            "metadata_access_boundary",
            "metadata range authority",
        )?,
        "metadata range authority.metadata_access_boundary",
    )?;
    if u64_value(
        required(
            boundary,
            "source_tensor_payload_bytes_read",
            "metadata range authority boundary",
        )?,
        "metadata range authority source tensor payload bytes read",
    )? != 0
    {
        return Err("metadata range authority already read source payload bytes".to_owned());
    }
    for field in [
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
        "mmap_or_memory_map_used",
        "source_model_instantiated",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
    ] {
        require_bool(
            required(boundary, field, "metadata range authority boundary")?,
            false,
            &format!("metadata range authority boundary.{field}"),
        )?;
    }
    let mut shards = BTreeMap::new();
    for (index, row) in array(
        required(authority, "shards", "metadata range authority")?,
        "metadata range authority.shards",
    )?
    .iter()
    .enumerate()
    {
        let row = object(row, &format!("metadata range authority shard {index}"))?;
        let relative_path = string(
            required(row, "relative_path", "metadata shard")?,
            "metadata shard.relative_path",
        )?
        .to_owned();
        validate_relative_path(&relative_path, "metadata shard relative path")?;
        let shard = MetadataShard {
            relative_path: relative_path.clone(),
            file_bytes: u64_value(
                required(row, "file_bytes", "metadata shard")?,
                "metadata shard.file_bytes",
            )?,
            header_sha256: sha256(
                required(row, "safetensors_header_sha256", "metadata shard")?,
                "metadata shard.safetensors_header_sha256",
            )?,
            prefix_sha256: sha256(
                required(row, "safetensors_prefix_sha256", "metadata shard")?,
                "metadata shard.safetensors_prefix_sha256",
            )?,
        };
        if shard.file_bytes == 0 || shards.insert(relative_path, shard).is_some() {
            return Err(
                "metadata range authority shard set is empty, duplicate, or invalid".to_owned(),
            );
        }
    }
    if shards.len() as u64 != geometry.shard_count {
        return Err("metadata range authority shard count drifted".to_owned());
    }
    let mut tensors = BTreeMap::new();
    let mut maximum_window_bytes = 0u64;
    for (index, row) in array(
        required(authority, "tensors", "metadata range authority")?,
        "metadata range authority.tensors",
    )?
    .iter()
    .enumerate()
    {
        let row = object(row, &format!("metadata range authority tensor {index}"))?;
        if string(
            required(row, "source_dtype", "metadata tensor")?,
            "metadata tensor.source_dtype",
        )? != "BF16"
        {
            return Err("metadata range authority tensor dtype must remain BF16".to_owned());
        }
        let tensor_name = string(
            required(row, "tensor_name", "metadata tensor")?,
            "metadata tensor.tensor_name",
        )?
        .to_owned();
        let shard_relative_path = string(
            required(row, "shard_relative_path", "metadata tensor")?,
            "metadata tensor.shard_relative_path",
        )?
        .to_owned();
        if !shards.contains_key(&shard_relative_path) {
            return Err("metadata tensor references an undeclared shard".to_owned());
        }
        let full_shape = parse_shape(
            required(row, "full_shape", "metadata tensor")?,
            "metadata tensor.full_shape",
        )?;
        let row_window_shape = parse_shape(
            required(row, "row_window_shape", "metadata tensor")?,
            "metadata tensor.row_window_shape",
        )?;
        let window_bytes = shape_bytes(&row_window_shape, "metadata tensor row window")?;
        if window_bytes == 0 || window_bytes > MAX_POSITIONED_READ_BYTES {
            return Err("metadata tensor row window exceeds the <=1 MiB reader ceiling".to_owned());
        }
        maximum_window_bytes = maximum_window_bytes.max(window_bytes);
        let tensor = MetadataTensor {
            tensor_name: tensor_name.clone(),
            shard_relative_path,
            full_shape,
            row_window_shape,
            absolute_data_offset: u64_value(
                required(row, "absolute_data_offset", "metadata tensor")?,
                "metadata tensor.absolute_data_offset",
            )?,
            data_bytes: u64_value(
                required(row, "data_bytes", "metadata tensor")?,
                "metadata tensor.data_bytes",
            )?,
        };
        if tensor.data_bytes == 0 || tensors.insert(tensor_name, tensor).is_some() {
            return Err("metadata range authority tensor set is duplicate or invalid".to_owned());
        }
    }
    if tensors.len() as u64 != geometry.tensor_count || maximum_window_bytes == 0 {
        return Err("metadata range authority tensor geometry drifted".to_owned());
    }
    Ok(MetadataBinding {
        evidence: document_evidence(document),
        authority_content_sha256: declared_content_sha,
        source_revision,
        source_index_relative_path,
        source_index_sha256,
        tensors,
        shards,
        maximum_window_bytes,
    })
}

#[derive(Clone, Debug)]
struct SemanticsBinding {
    evidence: Value,
    operator_execution_schema: String,
    operator_execution_status: String,
}

fn validate_semantics_attester(
    document: &LoadedDocument,
    metadata: &MetadataBinding,
) -> Result<SemanticsBinding, String> {
    let root = object(&document.value, "metadata operator semantics attester")?;
    require_schema_status(
        root,
        SEMANTICS_SCHEMA,
        SEMANTICS_STATUS,
        "metadata semantics attester",
    )?;
    let boundary = object(
        required(root, "execution_boundary", "metadata semantics attester")?,
        "metadata semantics attester.execution_boundary",
    )?;
    for field in [
        "source_tensor_payload_opened",
        "source_safetensors_or_other_weight_path_accepted",
        "source_model_instantiated",
        "source_inference_executed",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
    ] {
        require_bool(
            required(boundary, field, "metadata semantics boundary")?,
            false,
            &format!("metadata semantics boundary.{field}"),
        )?;
    }
    let pinned = object(
        required(root, "pinned_source_binding", "metadata semantics attester")?,
        "metadata semantics attester.pinned_source_binding",
    )?;
    if string(
        required(pinned, "source_model_id", "metadata semantics source")?,
        "metadata semantics source model",
    )? != SOURCE_MODEL_ID
        || string(
            required(pinned, "source_revision", "metadata semantics source")?,
            "metadata semantics source revision",
        )? != metadata.source_revision
        || sha256(
            required(pinned, "source_index_sha256", "metadata semantics source")?,
            "metadata semantics source index hash",
        )? != metadata.source_index_sha256
    {
        return Err("metadata semantics source binding differs from range authority".to_owned());
    }
    let consumed = object(
        required(
            root,
            "consumed_metadata_contracts",
            "metadata semantics attester",
        )?,
        "metadata semantics consumed contracts",
    )?;
    let range = object(
        required(
            consumed,
            "range_authority",
            "metadata semantics consumed contracts",
        )?,
        "metadata semantics consumed range authority",
    )?;
    if sha256(
        required(
            range,
            "document_sha256",
            "metadata semantics range reference",
        )?,
        "metadata semantics range document hash",
    )? != document_sha_from_metadata_evidence(metadata)
        || sha256(
            required(
                range,
                "authority_content_sha256",
                "metadata semantics range reference",
            )?,
            "metadata semantics range authority content hash",
        )? != metadata.authority_content_sha256
    {
        return Err("metadata semantics range-authority binding drifted".to_owned());
    }
    let future = object(
        required(
            root,
            "future_exact_execution_attestation",
            "metadata semantics attester",
        )?,
        "metadata semantics future execution attestation",
    )?;
    let operator_execution_schema = string(
        required(
            future,
            "schema",
            "metadata semantics future execution attestation",
        )?,
        "metadata semantics future execution schema",
    )?
    .to_owned();
    let operator_execution_status = string(
        required(
            future,
            "status_only_after_real_separately_leased_source_execution",
            "metadata semantics future execution attestation",
        )?,
        "metadata semantics future execution status",
    )?
    .to_owned();
    if operator_execution_schema != OPERATOR_ATTESTATION_SCHEMA
        || operator_execution_status != OPERATOR_ATTESTATION_STATUS
    {
        return Err("metadata semantics execution-attestation grammar drifted".to_owned());
    }
    Ok(SemanticsBinding {
        evidence: document_evidence(document),
        operator_execution_schema,
        operator_execution_status,
    })
}

fn document_sha_from_metadata_evidence(metadata: &MetadataBinding) -> String {
    metadata
        .evidence
        .get("raw_document_sha256")
        .and_then(Value::as_str)
        .expect("validated metadata evidence has raw document hash")
        .to_owned()
}

#[derive(Clone, Debug)]
struct FlatMapBinding {
    document_sha256: String,
    shard_raw_hashes: BTreeMap<String, String>,
    tensor_raw_bf16_hashes: BTreeMap<String, String>,
}

fn validate_flat_runtime_range_map(
    value: &Value,
    metadata: &MetadataBinding,
    geometry: Geometry,
) -> Result<FlatMapBinding, String> {
    let root = object(value, "flat runtime range map")?;
    if string(
        required(root, "schema", "flat runtime range map")?,
        "flat runtime range map.schema",
    )? != RUNTIME_RANGE_MAP_SCHEMA
        || string(
            required(root, "source_model_id", "flat runtime range map")?,
            "flat runtime range map.source_model_id",
        )? != SOURCE_MODEL_ID
        || string(
            required(root, "source_revision", "flat runtime range map")?,
            "flat runtime range map.source_revision",
        )? != metadata.source_revision
        || u64_value(
            required(root, "source_tensor_count", "flat runtime range map")?,
            "flat runtime range map.source_tensor_count",
        )? != geometry.tensor_count
    {
        return Err("flat runtime range map source identity/geometry drifted".to_owned());
    }
    let index = object(
        required(root, "source_index", "flat runtime range map")?,
        "flat runtime range map.source_index",
    )?;
    if string(
        required(
            index,
            "relative_path",
            "flat runtime range map source index",
        )?,
        "flat runtime range map source index path",
    )? != metadata.source_index_relative_path
        || sha256(
            required(index, "sha256", "flat runtime range map source index")?,
            "flat runtime range map source index hash",
        )? != metadata.source_index_sha256
        || string(
            required(index, "format", "flat runtime range map source index")?,
            "flat runtime range map source index format",
        )? != "huggingface.safetensors.index.json"
    {
        return Err("flat runtime range map source-index binding drifted".to_owned());
    }
    if u64_value(
        required(root, "maximum_window_bytes", "flat runtime range map")?,
        "flat runtime range map.maximum_window_bytes",
    )? != metadata.maximum_window_bytes
        || metadata.maximum_window_bytes > MAX_POSITIONED_READ_BYTES
    {
        return Err("flat runtime range map <=1 MiB window binding drifted".to_owned());
    }
    let mut shard_by_id = BTreeMap::<String, String>::new();
    let mut shard_raw_hashes = BTreeMap::new();
    for (index, row) in array(
        required(root, "shards", "flat runtime range map")?,
        "flat runtime range map.shards",
    )?
    .iter()
    .enumerate()
    {
        let row = object(row, &format!("flat runtime range map shard {index}"))?;
        let shard_id = string(
            required(row, "shard_id", "flat runtime shard")?,
            "flat runtime shard.id",
        )?
        .to_owned();
        let relative_path = string(
            required(row, "relative_path", "flat runtime shard")?,
            "flat runtime shard.relative_path",
        )?
        .to_owned();
        validate_relative_path(&relative_path, "flat runtime shard relative path")?;
        let metadata_shard = metadata
            .shards
            .get(&relative_path)
            .ok_or_else(|| "flat runtime map contains an undeclared shard".to_owned())?;
        if u64_value(
            required(row, "bytes", "flat runtime shard")?,
            "flat runtime shard.bytes",
        )? != metadata_shard.file_bytes
            || sha256(
                required(row, "safetensors_header_sha256", "flat runtime shard")?,
                "flat runtime shard header hash",
            )? != metadata_shard.header_sha256
        {
            return Err("flat runtime shard metadata differs from authority".to_owned());
        }
        let raw_sha = sha256(
            required(row, "sha256", "flat runtime shard")?,
            "flat runtime shard raw SHA-256",
        )?;
        if shard_by_id
            .insert(shard_id, relative_path.clone())
            .is_some()
            || shard_raw_hashes.insert(relative_path, raw_sha).is_some()
        {
            return Err("flat runtime map duplicates shard identity".to_owned());
        }
    }
    if shard_by_id.len() as u64 != geometry.shard_count
        || shard_raw_hashes.len() as u64 != geometry.shard_count
    {
        return Err("flat runtime map shard count drifted".to_owned());
    }
    let mut tensor_raw_bf16_hashes = BTreeMap::new();
    for (index, row) in array(
        required(root, "tensors", "flat runtime range map")?,
        "flat runtime range map.tensors",
    )?
    .iter()
    .enumerate()
    {
        let row = object(row, &format!("flat runtime range map tensor {index}"))?;
        let tensor_name = string(
            required(row, "tensor_name", "flat runtime tensor")?,
            "flat runtime tensor.tensor_name",
        )?
        .to_owned();
        let metadata_tensor = metadata
            .tensors
            .get(&tensor_name)
            .ok_or_else(|| "flat runtime map contains an undeclared tensor".to_owned())?;
        let shard_id = string(
            required(row, "shard_id", "flat runtime tensor")?,
            "flat runtime tensor.shard_id",
        )?;
        if shard_by_id.get(shard_id) != Some(&metadata_tensor.shard_relative_path)
            || string(
                required(row, "dtype", "flat runtime tensor")?,
                "flat runtime tensor.dtype",
            )? != "BF16"
            || parse_shape(
                required(row, "shape", "flat runtime tensor")?,
                "flat runtime tensor.shape",
            )? != metadata_tensor.full_shape
            || u64_value(
                required(row, "data_offset", "flat runtime tensor")?,
                "flat runtime tensor.data_offset",
            )? != metadata_tensor.absolute_data_offset
            || u64_value(
                required(row, "data_bytes", "flat runtime tensor")?,
                "flat runtime tensor.data_bytes",
            )? != metadata_tensor.data_bytes
        {
            return Err("flat runtime tensor range semantics drifted from authority".to_owned());
        }
        let raw_bf16_sha = sha256(
            required(row, "raw_bf16_sha256", "flat runtime tensor")?,
            "flat runtime tensor raw BF16 SHA-256",
        )?;
        if tensor_raw_bf16_hashes
            .insert(tensor_name, raw_bf16_sha)
            .is_some()
        {
            return Err("flat runtime map duplicates tensor identity".to_owned());
        }
    }
    if tensor_raw_bf16_hashes.len() as u64 != geometry.tensor_count {
        return Err("flat runtime map tensor count drifted".to_owned());
    }
    Ok(FlatMapBinding {
        document_sha256: sha256_hex(canonical_json(value)?.as_bytes()),
        shard_raw_hashes,
        tensor_raw_bf16_hashes,
    })
}

fn preflight_document(
    metadata: MetadataBinding,
    semantics: SemanticsBinding,
    mode: Mode,
    future_source_root: Option<&Path>,
) -> Result<Value, String> {
    let source_root_reference = match (mode, future_source_root) {
        (Mode::MetadataPreflight, None) => json!({
            "mode": mode.as_str(),
            "provided": false,
            "source_root_opened_or_statted_by_this_preflight": false,
        }),
        (Mode::FutureSourceRootExecutionReference, Some(path)) => json!({
            "mode": mode.as_str(),
            "provided": true,
            "absolute_future_source_root": path,
            "source_root_opened_or_statted_by_this_preflight": false,
            "future_child_must_require_a_real_non_symlink_root_before_any_payload_read": true,
            "future_child_must_not_mmap_or_map_a_shard": true,
        }),
        _ => return Err("source-root mode/argument validation drifted".to_owned()),
    };
    let runtime_map_fields = json!({
        "schema": RUNTIME_RANGE_MAP_SCHEMA,
        "required_top_level_fields": [
            "schema", "source_model_id", "source_revision", "source_tensor_count", "source_index",
            "maximum_window_bytes", "shards", "tensors"
        ],
        "source_index_fields": ["relative_path", "sha256", "format"],
        "shard_fields": ["shard_id", "relative_path", "bytes", "sha256", "safetensors_header_sha256"],
        "tensor_fields": ["tensor_name", "shard_id", "dtype", "shape", "data_offset", "data_bytes", "raw_bf16_sha256"],
        "per_shard_raw_sha256_required": true,
        "per_tensor_raw_bf16_sha256_required": true,
        "ranges_must_match_metadata_authority_offsets_shapes_dtype_and_row_windows": true,
        "maximum_window_bytes": metadata.maximum_window_bytes,
        "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
        "full_shard_or_full_model_residency_forbidden": true,
    });
    seal_value(json!({
        "schema": SCHEMA,
        "status": STATUS,
        "prepared": true,
        "runtime_admission_earned": false,
        "source_payload_validation_executed": false,
        "future_source_root_reference": source_root_reference,
        "sealed_metadata_authority_binding": {
            "metadata_range_authority": metadata.evidence,
            "authority_content_sha256": metadata.authority_content_sha256,
            "source_revision": metadata.source_revision,
            "source_index": {
                "relative_path": metadata.source_index_relative_path,
                "sha256": metadata.source_index_sha256,
            },
            "source_tensor_count": metadata.tensors.len(),
            "source_shard_count": metadata.shards.len(),
            "maximum_declared_bf16_row_window_bytes": metadata.maximum_window_bytes,
            "metadata_authority_is_not_a_payload_hash_or_execution_attestation": true,
        },
        "metadata_semantics_binding": {
            "operator_semantics_attester": semantics.evidence,
            "future_operator_accumulation_execution_attestation": {
                "schema": semantics.operator_execution_schema,
                "status_only_after_real_separately_leased_source_execution": semantics.operator_execution_status,
                "must_be_sealed_and_bind_same_runtime_admission": true,
            },
            "future_range_reader_exact_semantics_attestation": {
                "schema": RANGE_READER_ATTESTATION_SCHEMA,
                "status_only_after_real_separately_leased_source_execution": RANGE_READER_ATTESTATION_STATUS,
                "must_be_sealed_and_bind_same_runtime_admission": true,
            },
            "both_execution_attestations_required": true,
        },
        "future_flat_runtime_range_map": runtime_map_fields,
        "future_runtime_admission_receipt": {
            "schema": RUNTIME_ADMISSION_SCHEMA,
            "status_only_after_bounded_source_validation": RUNTIME_ADMISSION_STATUS,
            "must_be_sealed": true,
            "must_bind": [
                "sealed_metadata_authority_binding.metadata_range_authority.raw_document_sha256",
                "sealed_metadata_authority_binding.authority_content_sha256",
                "flat_runtime_range_map.document_sha256",
                "flat_runtime_range_map.per_shard_raw_sha256s",
                "flat_runtime_range_map.per_tensor_raw_bf16_sha256s",
                "operator_accumulation_execution_attestation.seal_sha256",
                "range_reader_exact_semantics_attestation.seal_sha256",
                "same source revision/index/trace and <=1 MiB one-window evidence",
            ],
            "must_not_claim_model_residency_or_server": true,
        },
        "future_reader_validation_order": [
            "verify sealed runtime-admission receipt before source payload open",
            "verify index/header/relative path and flat-range-map identities",
            "read at most one <=1 MiB positioned BF16 window at a time",
            "compute and compare every declared raw shard/tensor hash without retaining full bodies",
            "record row-major BF16 range order and exact operator accumulation semantics",
            "write both execution attestations only after genuine source execution",
        ],
        "execution_boundary": {
            "metadata_documents_read": true,
            "future_source_root_opened_or_statted": false,
            "source_tensor_payload_opened": false,
            "source_model_loaded_or_instantiated": false,
            "whole_shard_mapped_or_cached": false,
            "whole_source_model_resident": false,
            "gpu_metal_mps_or_other_accelerator_invoked": false,
            "server_started_or_contacted": false,
            "hcli_invoked": false,
            "lease_requested_issued_or_consumed": false,
            "child_process_started": false,
            "tps_or_tg_measured": false,
        },
        "claim_boundary": "Prepared CPU/build-only admission grammar. This does not emit a flat runtime map, validate any real source payload hash, earn runtime admission, execute a source teacher, load a model, start a server, acquire a lease, or report HCLI/TPS/TG/tournament evidence.",
    }))
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() || !path.parent().is_some_and(Path::is_dir) || path.exists() {
        return Err("--out must be a new absolute path below an existing parent".to_owned());
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize runtime range-admission preflight: {error}"))?;
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
    let range = read_metadata_document(&args.range_authority, "metadata range authority")?;
    let semantics =
        read_metadata_document(&args.semantics_attester, "metadata semantics attester")?;
    let metadata = validate_metadata_authority(&range, PRODUCTION_GEOMETRY)?;
    let semantics = validate_semantics_attester(&semantics, &metadata)?;
    let document = preflight_document(
        metadata,
        semantics,
        args.mode,
        args.future_source_root.as_deref(),
    )?;
    write_new_json(&args.out, &document)?;
    Ok(document)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("cannot render runtime range-admission preflight: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!("Q30 runtime range-admission preflight refused: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE_GEOMETRY: Geometry = Geometry {
        tensor_count: 1,
        shard_count: 1,
    };

    fn hash(label: &str) -> String {
        sha256_hex(label.as_bytes())
    }

    fn metadata_authority() -> LoadedDocument {
        let authority = json!({
            "schema": RANGE_AUTHORITY_SCHEMA,
            "status": RANGE_AUTHORITY_STATUS,
            "source": {
                "model_id": SOURCE_MODEL_ID,
                "source_revision": "0123456789abcdef0123456789abcdef01234567",
                "source_tensor_count": FIXTURE_GEOMETRY.tensor_count,
                "source_shard_count": FIXTURE_GEOMETRY.shard_count,
                "source_index": {
                    "relative_path": "model.safetensors.index.json",
                    "sha256": hash("source-index"),
                    "weight_map_tensor_count": FIXTURE_GEOMETRY.tensor_count,
                },
            },
            "exact_streamed_oracle_scope": {
                "source_template_token_count": PREFIX_TOKENS,
                "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
                "total_forwards_per_replay_arm": SOURCE_FORWARDS,
                "layers": SOURCE_LAYERS,
                "top_k_routes_per_token": TOP_K,
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
            "shards": [{
                "relative_path": "model-00001-of-00001.safetensors",
                "file_bytes": 128,
                "safetensors_header_sha256": hash("header"),
                "safetensors_prefix_sha256": hash("prefix"),
            }],
            "tensors": [{
                "tensor_name": "fixture.weight",
                "source_dtype": "BF16",
                "full_shape": [2, 2],
                "row_window_shape": [2, 2],
                "shard_relative_path": "model-00001-of-00001.safetensors",
                "absolute_data_offset": 32,
                "data_bytes": 8,
            }],
        });
        let envelope = json!({
            "authority_content_sha256": sha256_hex(canonical_json(&authority).expect("canonical fixture").as_bytes()),
            "authority": authority,
        });
        memory_document("/fixtures/range-authority.json", envelope)
    }

    fn semantics_attester(range: &LoadedDocument) -> LoadedDocument {
        let authority_content_sha256 = range.value["authority_content_sha256"]
            .as_str()
            .expect("authority content hash");
        memory_document(
            "/fixtures/semantics.json",
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
                    "source_model_id": SOURCE_MODEL_ID,
                    "source_revision": "0123456789abcdef0123456789abcdef01234567",
                    "source_index_sha256": hash("source-index"),
                },
                "consumed_metadata_contracts": {
                    "range_authority": {
                        "document_sha256": range.raw_sha256,
                        "authority_content_sha256": authority_content_sha256,
                    }
                },
                "future_exact_execution_attestation": {
                    "schema": OPERATOR_ATTESTATION_SCHEMA,
                    "status_only_after_real_separately_leased_source_execution": OPERATOR_ATTESTATION_STATUS,
                }
            }),
        )
    }

    fn flat_map() -> Value {
        json!({
            "schema": RUNTIME_RANGE_MAP_SCHEMA,
            "source_model_id": SOURCE_MODEL_ID,
            "source_revision": "0123456789abcdef0123456789abcdef01234567",
            "source_tensor_count": FIXTURE_GEOMETRY.tensor_count,
            "source_index": {
                "relative_path": "model.safetensors.index.json",
                "sha256": hash("source-index"),
                "format": "huggingface.safetensors.index.json",
            },
            "maximum_window_bytes": 8,
            "shards": [{
                "shard_id": "shard-0",
                "relative_path": "model-00001-of-00001.safetensors",
                "bytes": 128,
                "sha256": hash("raw-shard"),
                "safetensors_header_sha256": hash("header"),
            }],
            "tensors": [{
                "tensor_name": "fixture.weight",
                "shard_id": "shard-0",
                "dtype": "BF16",
                "shape": [2, 2],
                "data_offset": 32,
                "data_bytes": 8,
                "raw_bf16_sha256": hash("raw-bf16"),
            }],
        })
    }

    #[test]
    fn metadata_preflight_is_sealed_prepared_and_never_opens_a_source_root() {
        let range = metadata_authority();
        let semantics = semantics_attester(&range);
        let binding = validate_metadata_authority(&range, FIXTURE_GEOMETRY)
            .expect("fixture authority validates");
        let semantics =
            validate_semantics_attester(&semantics, &binding).expect("fixture semantics validates");
        let output = preflight_document(binding, semantics, Mode::MetadataPreflight, None)
            .expect("preflight serializes");
        assert_eq!(output["schema"], SCHEMA);
        assert_eq!(output["status"], STATUS);
        assert_eq!(output["runtime_admission_earned"], false);
        assert_eq!(
            output["execution_boundary"]["source_tensor_payload_opened"],
            false
        );
        assert_eq!(
            output["execution_boundary"]["future_source_root_opened_or_statted"],
            false
        );
        assert_eq!(
            output["future_runtime_admission_receipt"]["schema"],
            RUNTIME_ADMISSION_SCHEMA
        );
        verify_seal(&output, "prepared output").expect("output is sealed");
    }

    #[test]
    fn flat_fixture_binds_each_raw_shard_and_tensor_hash_to_metadata_ranges() {
        let range = metadata_authority();
        let metadata = validate_metadata_authority(&range, FIXTURE_GEOMETRY)
            .expect("fixture authority validates");
        let map = flat_map();
        let validated = validate_flat_runtime_range_map(&map, &metadata, FIXTURE_GEOMETRY)
            .expect("flat fixture validates");
        assert_eq!(validated.shard_raw_hashes.len(), 1);
        assert_eq!(validated.tensor_raw_bf16_hashes.len(), 1);
        assert_eq!(
            validated.document_sha256,
            sha256_hex(canonical_json(&map).expect("canonical map").as_bytes())
        );
    }

    #[test]
    fn flat_fixture_rejects_range_or_window_drift_without_reading_payloads() {
        let range = metadata_authority();
        let metadata = validate_metadata_authority(&range, FIXTURE_GEOMETRY)
            .expect("fixture authority validates");
        let mut wrong_offset = flat_map();
        wrong_offset["tensors"][0]["data_offset"] = json!(33);
        let error = validate_flat_runtime_range_map(&wrong_offset, &metadata, FIXTURE_GEOMETRY)
            .expect_err("offset drift must refuse");
        assert!(error.contains("range semantics"));

        let mut wrong_window = flat_map();
        wrong_window["maximum_window_bytes"] = json!(MAX_POSITIONED_READ_BYTES + 1);
        let error = validate_flat_runtime_range_map(&wrong_window, &metadata, FIXTURE_GEOMETRY)
            .expect_err("window drift must refuse");
        assert!(error.contains("<=1 MiB"));
    }

    #[test]
    fn future_source_root_is_syntactic_reference_only_and_requires_its_inert_mode() {
        let base = vec![
            "--range-authority".to_owned(),
            "/tmp/range.json".to_owned(),
            "--semantics-attester".to_owned(),
            "/tmp/semantics.json".to_owned(),
            "--out".to_owned(),
            "/tmp/output.json".to_owned(),
        ];
        let mut bad = base.clone();
        bad.extend(["--future-source-root".to_owned(), "/real/source".to_owned()]);
        assert!(parse_args_from(bad).is_err());

        let mut unsupported = base.clone();
        unsupported.extend(["--source-root".to_owned(), "/real/source".to_owned()]);
        assert!(parse_args_from(unsupported).is_err());

        let mut accepted = base;
        accepted.extend([
            "--mode".to_owned(),
            "future-source-root-execution-reference".to_owned(),
            "--future-source-root".to_owned(),
            "/future/source".to_owned(),
        ]);
        let args =
            parse_args_from(accepted).expect("opaque future root reference is syntactically valid");
        assert_eq!(args.mode, Mode::FutureSourceRootExecutionReference);
        assert_eq!(
            args.future_source_root,
            Some(PathBuf::from("/future/source"))
        );
    }
}
