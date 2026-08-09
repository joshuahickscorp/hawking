#![allow(dead_code)] // This is a deliberately inert, CPU-only future-child contract.

//! CPU/build-only preflight for the future Qwen30 streamed BF16 source-teacher
//! child.
//!
//! This binary is intentionally *not* a source reader or a model runner.  It
//! consumes only bounded JSON authority/contract documents and emits a sealed
//! prepared or refused record describing the only permissible future child
//! surface.  It has no source-root option, no safetensors option, no child
//! command, no accelerator/server/HCLI option, and no lease-issuing surface.
//!
//! Two independently prepared Q30 records use different future schemas:
//!
//! * the metadata semantics attester names an operator/accumulation execution
//!   attestation; and
//! * the range-reader preflight plus feasibility gate name a BF16 exact
//!   semantics attestation and a flat runtime range-map schema.
//!
//! A future source-teacher execution must therefore consume a separately
//! sealed, non-authorizing dual-attestation/runtime-admission bridge.  This
//! preflight refuses a malformed bridge and otherwise remains
//! `PREPARED_*_NOT_EXECUTED`; it can never turn an existing metadata or fixture
//! document into source-teacher evidence.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;

const RESULT_SCHEMA: &str = "hawking.ascension.qwen30_streamed_source_teacher_child_preflight.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_INTERFACE_NOT_EXECUTED";
const REFUSED_STATUS: &str =
    "REFUSED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_PREFLIGHT_INVALID_OR_INCOMPLETE";

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
const DUAL_BRIDGE_SCHEMA: &str =
    "hawking.ascension.qwen30_streamed_source_teacher_dual_attestation_runtime_admission.v1";
const DUAL_BRIDGE_STATUS: &str =
    "PREPARED_QWEN30_STREAMED_SOURCE_TEACHER_DUAL_ATTESTATION_RUNTIME_ADMISSION_NOT_EXECUTED";

const FEASIBILITY_SCHEMA: &str =
    "hawking.ascension.qwen30_layer_streamed_source_bf16_final_logit_oracle_feasibility.v1";
const FEASIBILITY_PREPARED_STATUS: &str =
    "PREPARED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_NOT_EXECUTED";
const FEASIBILITY_REFUSED_STATUS: &str =
    "REFUSED_QWEN30_LAYER_STREAMED_SOURCE_BF16_ORACLE_FEASIBILITY_UNSAFE_OR_UNPROVEN";
const RAW_SIX_VECTOR_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_successor.v1";
const RAW_SIX_VECTOR_STATUS: &str = "PREPARED_RAW_FINAL_LOGIT_RETENTION_SUCCESSOR_NOT_RUN";
const CURRENT_TRACE_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_all_layer_current_trace_comparison.v1";
const CURRENT_TRACE_STATUS: &str =
    "EARNED_CANDIDATE_LOCAL_ALL_LAYER_DIVERGENCE_UNQUALIFIED_NON_PROMOTABLE";
const SOURCE_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_quiet_lease.v1";
const SOURCE_LEASE_STATUS: &str =
    "GRANTED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_RAW_LOGIT_CAPTURE_ONE_SHOT";
const SOURCE_TERMINAL_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_raw_logit_capture.v1";
const SOURCE_TERMINAL_STATUS: &str =
    "CAPTURED_QWEN30_HQ30GR2_SOURCE_BF16_TWO_RAW_FINAL_LOGITS_TEACHER_ONLY";
const SOURCE_EVICTION_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_source_bf16_teacher_eviction.v1";
const SOURCE_EVICTION_STATUS: &str =
    "EARNED_QWEN30_HQ30GR2_SOURCE_BF16_TEACHER_EVICTED_BEFORE_NATIVE_CAPTURE";
const NATIVE_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_quiet_lease.v1";
const NATIVE_LEASE_STATUS: &str = "GRANTED_QWEN30_HQ30GR2_RAW_FINAL_LOGIT_RETENTION_ONE_SHOT";
const NATIVE_TERMINAL_SCHEMA: &str =
    "hawking.ascension.qwen30_hq30gr2_raw_final_logit_retention_capture.v1";
const NATIVE_TERMINAL_STATUS: &str =
    "EARNED_NEW_DIAGNOSTIC_RAW_FINAL_LOGITS_RETAINED_NOT_THREE_WAY_ORACLE";

const SOURCE_MODEL_ID: &str = "Qwen3-Coder-30B-A3B-Instruct";
const SOURCE_TENSORS: u64 = 18_867;
const SOURCE_SHARDS: u64 = 16;
const SOURCE_LAYERS: u64 = 48;
const SOURCE_FORWARDS: u64 = 370;
const PREFIX_TOKENS: u64 = 369;
const FORCED_TOKEN_ID: u64 = 949;
const TOP_K: u64 = 8;
const VOCAB_ROWS: u64 = 151_936;
const F32_VECTOR_BYTES: u64 = VOCAB_ROWS * 4;
const MAX_POSITIONED_READ_BYTES: u64 = 1024 * 1024;
const MAX_METADATA_BYTES: u64 = 64 * 1024 * 1024;

const SOURCE_PAYLOADS: [&str; 2] = [
    "source_bf16_exact_prefix_logits.f32le",
    "source_bf16_forced_shared_continuation_logits.f32le",
];
const NATIVE_PAYLOADS: [&str; 4] = [
    "scalar_control_exact_prefix_logits.f32le",
    "scalar_control_forced_shared_continuation_logits.f32le",
    "hq30gr2_candidate_exact_prefix_logits.f32le",
    "hq30gr2_candidate_forced_shared_continuation_logits.f32le",
];

#[derive(Debug)]
struct Args {
    range_authority: PathBuf,
    semantics: PathBuf,
    feasibility: Option<PathBuf>,
    raw_six_vector: PathBuf,
    current_trace: PathBuf,
    dual_bridge: Option<PathBuf>,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen30_streamed_source_teacher_child_preflight \\\n+     --range-authority ABSOLUTE_METADATA_RANGE_AUTHORITY_JSON \\\n+     --semantics ABSOLUTE_METADATA_SEMANTICS_JSON \\\n+     [--feasibility ABSOLUTE_SEALED_STREAMED_FEASIBILITY_JSON] \\\n+     --raw-six-vector ABSOLUTE_SEALED_RAW_SIX_VECTOR_CONTRACT_JSON \\\n+     --current-trace ABSOLUTE_SEALED_CURRENT_TRACE_JSON \\\n+     [--dual-attestation-runtime-admission ABSOLUTE_SEALED_BRIDGE_JSON] \\\n+     --out NEW_ABSOLUTE_PREFLIGHT_JSON"
}

fn parse_args_from<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut range_authority = None;
    let mut semantics = None;
    let mut feasibility = None;
    let mut raw_six_vector = None;
    let mut current_trace = None;
    let mut dual_bridge = None;
    let mut out = None;
    let mut values = arguments.into_iter();
    while let Some(flag) = values.next() {
        let next = |name: &str, values: &mut I::IntoIter| {
            values
                .next()
                .ok_or_else(|| format!("missing value for {name}; {}", usage()))
        };
        let destination = match flag.as_str() {
            "--range-authority" => &mut range_authority,
            "--semantics" => &mut semantics,
            "--feasibility" => &mut feasibility,
            "--raw-six-vector" => &mut raw_six_vector,
            "--current-trace" => &mut current_trace,
            "--dual-attestation-runtime-admission" => &mut dual_bridge,
            "--out" => &mut out,
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
        };
        if destination
            .replace(PathBuf::from(next(&flag, &mut values)?))
            .is_some()
        {
            return Err(format!("{flag} was supplied more than once; {}", usage()));
        }
    }
    let required = |value: Option<PathBuf>, flag: &str| {
        value.ok_or_else(|| format!("{flag} is required; {}", usage()))
    };
    let args = Args {
        range_authority: required(range_authority, "--range-authority")?,
        semantics: required(semantics, "--semantics")?,
        feasibility,
        raw_six_vector: required(raw_six_vector, "--raw-six-vector")?,
        current_trace: required(current_trace, "--current-trace")?,
        dual_bridge,
        out: required(out, "--out")?,
    };
    for (label, path) in [
        ("--range-authority", &args.range_authority),
        ("--semantics", &args.semantics),
        ("--raw-six-vector", &args.raw_six_vector),
        ("--current-trace", &args.current_trace),
        ("--out", &args.out),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
    }
    for (label, path) in [
        ("--feasibility", args.feasibility.as_ref()),
        (
            "--dual-attestation-runtime-admission",
            args.dual_bridge.as_ref(),
        ),
    ] {
        if path.is_some_and(|path| !path.is_absolute()) {
            return Err(format!("{label} must be absolute"));
        }
    }
    if !args.out.parent().is_some_and(Path::is_dir) {
        return Err("--out parent must already exist".to_owned());
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

/// Canonical JSON compatible with the repository's sealed-receipt boundary.
fn canonical_json(value: &Value) -> Result<String, String> {
    match value {
        Value::Null => Ok("null".to_owned()),
        Value::Bool(value) => Ok(value.to_string()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(value) => serde_json::to_string(value)
            .map_err(|error| format!("cannot canonicalize string: {error}")),
        Value::Array(values) => {
            let mut result = String::from("[");
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    result.push(',');
                }
                result.push_str(&canonical_json(item)?);
            }
            result.push(']');
            Ok(result)
        }
        Value::Object(values) => {
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            let mut result = String::from("{");
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    result.push(',');
                }
                result.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("cannot canonicalize key: {error}"))?,
                );
                result.push(':');
                result.push_str(&canonical_json(
                    values
                        .get(key)
                        .ok_or_else(|| "canonical object key disappeared".to_owned())?,
                )?);
            }
            result.push('}');
            Ok(result)
        }
    }
}

fn seal_value(mut value: Value) -> Result<Value, String> {
    value
        .as_object_mut()
        .ok_or_else(|| "sealed receipt must be an object".to_owned())?
        .remove("seal_sha256");
    let canonical = canonical_json(&value)?;
    value
        .as_object_mut()
        .ok_or_else(|| "sealed receipt became non-object".to_owned())?
        .insert(
            "seal_sha256".to_owned(),
            Value::String(sha256_hex(canonical.as_bytes())),
        );
    Ok(value)
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let seal = string(
        required(object, "seal_sha256", label)?,
        &format!("{label}.seal_sha256"),
    )?;
    if !is_sha256(seal) {
        return Err(format!("{label}.seal_sha256 must be a lowercase SHA-256"));
    }
    let mut unsigned = value.clone();
    unsigned
        .as_object_mut()
        .ok_or_else(|| format!("{label} became non-object"))?
        .remove("seal_sha256");
    if sha256_hex(canonical_json(&unsigned)?.as_bytes()) != seal {
        return Err(format!("{label} seal does not bind canonical contents"));
    }
    Ok(seal.to_owned())
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
    document: &Map<String, Value>,
    schema: &str,
    status: &str,
    label: &str,
) -> Result<(), String> {
    if string(
        required(document, "schema", label)?,
        &format!("{label}.schema"),
    )? != schema
        || string(
            required(document, "status", label)?,
            &format!("{label}.status"),
        )? != status
    {
        return Err(format!("{label} schema/status drifted"));
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

fn read_metadata_document(
    path: &Path,
    label: &str,
    sealed: bool,
) -> Result<LoadedDocument, String> {
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
    let seal_sha256 = if sealed {
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

fn memory_document(path: &str, value: Value, sealed: bool) -> LoadedDocument {
    let bytes = serde_json::to_vec(&value).expect("test JSON serializes");
    LoadedDocument {
        path: PathBuf::from(path),
        raw_sha256: sha256_hex(&bytes),
        bytes: bytes.len() as u64,
        seal_sha256: if sealed {
            Some(verify_seal(&value, "test sealed document").expect("test document seals"))
        } else {
            None
        },
        value,
    }
}

fn document_evidence(document: &LoadedDocument) -> Value {
    let mut value = json!({
        "path": document.path,
        "bytes": document.bytes,
        "raw_document_sha256": document.raw_sha256,
    });
    if let Some(seal) = &document.seal_sha256 {
        value
            .as_object_mut()
            .expect("evidence is an object")
            .insert("seal_sha256".to_owned(), Value::String(seal.clone()));
    }
    value
}

#[derive(Clone, Debug)]
struct TraceBinding {
    token_sha256: String,
}

fn validate_trace(
    trace: &Map<String, Value>,
    label: &str,
    token_field: &str,
) -> Result<TraceBinding, String> {
    if u64_value(
        required(trace, "source_template_token_count", label)?,
        &format!("{label}.source_template_token_count"),
    )? != PREFIX_TOKENS
    {
        return Err(format!("{label} prefix length drifted"));
    }
    if u64_value(
        required(trace, "forced_identical_continuation_token_id", label)?,
        &format!("{label}.forced_identical_continuation_token_id"),
    )? != FORCED_TOKEN_ID
    {
        return Err(format!("{label} forced continuation token drifted"));
    }
    Ok(TraceBinding {
        token_sha256: sha256(
            required(trace, token_field, label)?,
            &format!("{label}.{token_field}"),
        )?,
    })
}

#[derive(Clone, Debug)]
struct RangeBinding {
    evidence: Value,
    authority_content_sha256: String,
    source_revision: String,
    source_index_sha256: String,
    maximum_window_bytes: u64,
}

fn checked_mul(left: u64, right: u64, label: &str) -> Result<u64, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{label} overflows u64"))
}

fn validate_range_authority(document: &LoadedDocument) -> Result<RangeBinding, String> {
    let root = object(&document.value, "range authority document")?;
    let authority_content_sha256 = sha256(
        required(root, "authority_content_sha256", "range authority document")?,
        "range authority document.authority_content_sha256",
    )?;
    let authority = object(
        required(root, "authority", "range authority document")?,
        "range authority document.authority",
    )?;
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
        || u64_value(
            required(
                source,
                "source_tensor_count",
                "metadata range authority.source",
            )?,
            "metadata range authority.source.source_tensor_count",
        )? != SOURCE_TENSORS
        || u64_value(
            required(
                source,
                "source_shard_count",
                "metadata range authority.source",
            )?,
            "metadata range authority.source.source_shard_count",
        )? != SOURCE_SHARDS
    {
        return Err("metadata range authority source identity/geometry drifted".to_owned());
    }
    let source_revision = string(
        required(source, "source_revision", "metadata range authority.source")?,
        "metadata range authority.source.source_revision",
    )?
    .to_owned();
    let source_index = object(
        required(source, "source_index", "metadata range authority.source")?,
        "metadata range authority.source.source_index",
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
    )? != SOURCE_TENSORS
    {
        return Err("metadata range authority source-index tensor count drifted".to_owned());
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
        ("layers", SOURCE_LAYERS),
        ("top_k_routes_per_token", TOP_K),
        ("row_tile_rows", 128),
        ("total_forwards_per_replay_arm", SOURCE_FORWARDS),
    ] {
        if u64_value(
            required(
                scope,
                field,
                "metadata range authority.exact_streamed_oracle_scope",
            )?,
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
            "metadata range authority.exact_streamed_oracle_scope",
        )?,
        true,
        "metadata range authority forbids sampling/autoregressive feedback",
    )?;
    let boundary = object(
        required(
            authority,
            "metadata_access_boundary",
            "metadata range authority",
        )?,
        "metadata range authority.metadata_access_boundary",
    )?;
    for field in [
        "mmap_or_memory_map_used",
        "source_model_instantiated",
        "gpu_or_metal_invoked",
        "server_started",
        "hcli_invoked",
        "lease_requested",
        "tensor_payload_hashes_collected",
        "whole_shard_payload_checksum_collected",
    ] {
        require_bool(
            required(
                boundary,
                field,
                "metadata range authority.metadata_access_boundary",
            )?,
            false,
            &format!("metadata range authority.{field}"),
        )?;
    }
    if u64_value(
        required(
            boundary,
            "source_tensor_payload_bytes_read",
            "metadata range authority.metadata_access_boundary",
        )?,
        "metadata range authority.source_tensor_payload_bytes_read",
    )? != 0
    {
        return Err(
            "metadata range authority must not already contain source payload reads".to_owned(),
        );
    }
    let tensors = array(
        required(authority, "tensors", "metadata range authority")?,
        "metadata range authority.tensors",
    )?;
    if tensors.is_empty() {
        return Err("metadata range authority lacks declared tensor windows".to_owned());
    }
    let mut maximum_window_bytes = 0u64;
    for (index, tensor) in tensors.iter().enumerate() {
        let tensor = object(tensor, &format!("metadata range authority tensor {index}"))?;
        if string(
            required(tensor, "source_dtype", "metadata range authority tensor")?,
            &format!("metadata range authority tensor {index}.source_dtype"),
        )? != "BF16"
        {
            return Err("metadata range authority must remain BF16".to_owned());
        }
        let shape = array(
            required(
                tensor,
                "row_window_shape",
                "metadata range authority tensor",
            )?,
            &format!("metadata range authority tensor {index}.row_window_shape"),
        )?;
        if shape.is_empty() {
            return Err(
                "metadata range authority tensor window shape must be non-empty".to_owned(),
            );
        }
        let elements = shape.iter().try_fold(1u64, |count, dimension| {
            let dimension = u64_value(dimension, "metadata range authority row window dimension")?;
            if dimension == 0 {
                return Err(
                    "metadata range authority row window dimension must be positive".to_owned(),
                );
            }
            checked_mul(
                count,
                dimension,
                "metadata range authority row window elements",
            )
        })?;
        maximum_window_bytes = maximum_window_bytes.max(checked_mul(
            elements,
            2,
            "metadata range authority BF16 row window bytes",
        )?);
    }
    if maximum_window_bytes == 0 || maximum_window_bytes > MAX_POSITIONED_READ_BYTES {
        return Err(format!(
            "metadata range authority requires {maximum_window_bytes} bytes per BF16 window, above the {MAX_POSITIONED_READ_BYTES}-byte source-reader ceiling"
        ));
    }
    Ok(RangeBinding {
        evidence: document_evidence(document),
        authority_content_sha256,
        source_revision,
        source_index_sha256,
        maximum_window_bytes,
    })
}

#[derive(Clone, Debug)]
struct SemanticsBinding {
    evidence: Value,
    source_revision: String,
    source_index_sha256: String,
}

fn validate_semantics(
    document: &LoadedDocument,
    range: &RangeBinding,
) -> Result<SemanticsBinding, String> {
    let root = object(&document.value, "operator semantics attester")?;
    require_schema_status(
        root,
        SEMANTICS_SCHEMA,
        SEMANTICS_STATUS,
        "operator semantics attester",
    )?;
    let boundary = object(
        required(root, "execution_boundary", "operator semantics attester")?,
        "operator semantics attester.execution_boundary",
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
            required(
                boundary,
                field,
                "operator semantics attester.execution_boundary",
            )?,
            false,
            &format!("operator semantics attester.{field}"),
        )?;
    }
    let pinned = object(
        required(root, "pinned_source_binding", "operator semantics attester")?,
        "operator semantics attester.pinned_source_binding",
    )?;
    if string(
        required(
            pinned,
            "source_model_id",
            "operator semantics attester.pinned_source_binding",
        )?,
        "operator semantics attester source model",
    )? != SOURCE_MODEL_ID
    {
        return Err("operator semantics attester source model drifted".to_owned());
    }
    let source_revision = string(
        required(
            pinned,
            "source_revision",
            "operator semantics attester.pinned_source_binding",
        )?,
        "operator semantics attester source revision",
    )?
    .to_owned();
    let source_index_sha256 = sha256(
        required(
            pinned,
            "source_index_sha256",
            "operator semantics attester.pinned_source_binding",
        )?,
        "operator semantics attester source index SHA",
    )?;
    if source_revision != range.source_revision || source_index_sha256 != range.source_index_sha256
    {
        return Err(
            "operator semantics attester source binding differs from metadata range authority"
                .to_owned(),
        );
    }
    let future = object(
        required(
            root,
            "future_exact_execution_attestation",
            "operator semantics attester",
        )?,
        "operator semantics attester.future_exact_execution_attestation",
    )?;
    if string(
        required(
            future,
            "schema",
            "operator semantics future execution attestation",
        )?,
        "operator semantics future execution attestation.schema",
    )? != OPERATOR_ATTESTATION_SCHEMA
        || string(
            required(
                future,
                "status_only_after_real_separately_leased_source_execution",
                "operator semantics future execution attestation",
            )?,
            "operator semantics future execution attestation.status_only_after_real_separately_leased_source_execution",
        )? != OPERATOR_ATTESTATION_STATUS
    {
        return Err("operator semantics future execution-attestation grammar drifted".to_owned());
    }
    Ok(SemanticsBinding {
        evidence: document_evidence(document),
        source_revision,
        source_index_sha256,
    })
}

#[derive(Clone, Debug)]
struct FeasibilityBinding {
    evidence: Value,
    status: String,
    semantic_equivalence_proven: bool,
    streamed_memory_arithmetic_fits: bool,
    zero_swap_condition_met: bool,
}

fn validate_feasibility(
    document: &LoadedDocument,
    trace: &TraceBinding,
) -> Result<FeasibilityBinding, String> {
    let root = object(&document.value, "streamed feasibility")?;
    let seal = document
        .seal_sha256
        .as_ref()
        .ok_or_else(|| "streamed feasibility must be sealed".to_owned())?;
    if seal.is_empty() {
        return Err("streamed feasibility has empty seal".to_owned());
    }
    let status = string(
        required(root, "status", "streamed feasibility")?,
        "streamed feasibility.status",
    )?;
    if string(
        required(root, "schema", "streamed feasibility")?,
        "streamed feasibility.schema",
    )? != FEASIBILITY_SCHEMA
        || !matches!(
            status,
            FEASIBILITY_PREPARED_STATUS | FEASIBILITY_REFUSED_STATUS
        )
    {
        return Err("streamed feasibility schema/status drifted".to_owned());
    }
    let exact = object(
        required(root, "exact_trace", "streamed feasibility")?,
        "streamed feasibility.exact_trace",
    )?;
    if u64_value(
        required(
            exact,
            "prefix_token_count",
            "streamed feasibility.exact_trace",
        )?,
        "streamed feasibility prefix",
    )? != PREFIX_TOKENS
        || u64_value(
            required(exact, "forced_token_id", "streamed feasibility.exact_trace")?,
            "streamed feasibility forced token",
        )? != FORCED_TOKEN_ID
        || sha256(
            required(
                exact,
                "source_template_token_ids_u32le_sha256",
                "streamed feasibility.exact_trace",
            )?,
            "streamed feasibility token SHA",
        )? != trace.token_sha256
    {
        return Err("streamed feasibility exact trace differs from current trace".to_owned());
    }
    let assessment = object(
        required(root, "memory_assessment", "streamed feasibility")?,
        "streamed feasibility.memory_assessment",
    )?;
    let feasibility = object(
        required(root, "feasibility", "streamed feasibility")?,
        "streamed feasibility.feasibility",
    )?;
    require_bool(
        required(
            feasibility,
            "oracle_execution_authorized",
            "streamed feasibility.feasibility",
        )?,
        false,
        "streamed feasibility may not authorize execution",
    )?;
    Ok(FeasibilityBinding {
        evidence: document_evidence(document),
        status: status.to_owned(),
        semantic_equivalence_proven: required(
            feasibility,
            "semantic_equivalence_proven_by_external_sealed_attestation",
            "streamed feasibility.feasibility",
        )?
        .as_bool()
        .ok_or_else(|| {
            "streamed feasibility semantic-equivalence flag must be boolean".to_owned()
        })?,
        streamed_memory_arithmetic_fits: required(
            assessment,
            "streamed_memory_arithmetic_fits",
            "streamed feasibility.memory_assessment",
        )?
        .as_bool()
        .ok_or_else(|| "streamed feasibility memory-fit flag must be boolean".to_owned())?,
        zero_swap_condition_met: required(
            assessment,
            "zero_swap_condition_met",
            "streamed feasibility.memory_assessment",
        )?
        .as_bool()
        .ok_or_else(|| "streamed feasibility zero-swap flag must be boolean".to_owned())?,
    })
}

#[derive(Clone, Debug)]
struct RawSixVectorBinding {
    evidence: Value,
    source_teacher_currently_blocked: bool,
}

fn expected_payloads() -> Vec<Value> {
    SOURCE_PAYLOADS
        .iter()
        .chain(NATIVE_PAYLOADS.iter())
        .map(|name| Value::String((*name).to_owned()))
        .collect()
}

fn validate_raw_six_vector(
    document: &LoadedDocument,
    trace: &TraceBinding,
) -> Result<RawSixVectorBinding, String> {
    let root = object(&document.value, "raw six-vector contract")?;
    if document.seal_sha256.is_none() {
        return Err("raw six-vector contract must be sealed".to_owned());
    }
    require_schema_status(
        root,
        RAW_SIX_VECTOR_SCHEMA,
        RAW_SIX_VECTOR_STATUS,
        "raw six-vector contract",
    )?;
    let replay = object(
        required(root, "replay_binding", "raw six-vector contract")?,
        "raw six-vector contract.replay_binding",
    )?;
    let exact = object(
        required(
            replay,
            "exact_trace",
            "raw six-vector contract.replay_binding",
        )?,
        "raw six-vector contract.replay_binding.exact_trace",
    )?;
    let observed_trace = validate_trace(
        exact,
        "raw six-vector contract.replay_binding.exact_trace",
        "source_template_token_ids_u32le_sha256",
    )?;
    if observed_trace.token_sha256 != trace.token_sha256 {
        return Err("raw six-vector contract token trace differs from current trace".to_owned());
    }
    let plan = object(
        required(
            root,
            "six_vector_retention_contract",
            "raw six-vector contract",
        )?,
        "raw six-vector contract.six_vector_retention_contract",
    )?;
    if string(
        required(
            plan,
            "dtype",
            "raw six-vector contract.six_vector_retention_contract",
        )?,
        "raw six-vector dtype",
    )? != "f32le"
        || u64_value(
            required(
                plan,
                "vocab_rows",
                "raw six-vector contract.six_vector_retention_contract",
            )?,
            "raw six-vector vocab rows",
        )? != VOCAB_ROWS
        || u64_value(
            required(
                plan,
                "bytes_per_vector",
                "raw six-vector contract.six_vector_retention_contract",
            )?,
            "raw six-vector bytes per vector",
        )? != F32_VECTOR_BYTES
        || u64_value(
            required(
                plan,
                "required_payload_count",
                "raw six-vector contract.six_vector_retention_contract",
            )?,
            "raw six-vector count",
        )? != 6
        || u64_value(
            required(
                plan,
                "required_total_payload_bytes",
                "raw six-vector contract.six_vector_retention_contract",
            )?,
            "raw six-vector total bytes",
        )? != F32_VECTOR_BYTES * 6
        || array(
            required(
                plan,
                "required_payloads",
                "raw six-vector contract.six_vector_retention_contract",
            )?,
            "raw six-vector required payloads",
        )? != expected_payloads().as_slice()
    {
        return Err("raw six-vector contract geometry/name plan drifted".to_owned());
    }
    require_bool(
        required(
            plan,
            "receipt_must_be_written_after_all_six_payloads_and_fsyncs",
            "raw six-vector contract.six_vector_retention_contract",
        )?,
        true,
        "raw six-vector requires receipt-last persistence",
    )?;
    let gate = object(
        required(
            root,
            "source_memory_and_eviction_gate",
            "raw six-vector contract",
        )?,
        "raw six-vector contract.source_memory_and_eviction_gate",
    )?;
    Ok(RawSixVectorBinding {
        evidence: document_evidence(document),
        source_teacher_currently_blocked: required(
            gate,
            "source_teacher_capture_is_currently_blocked",
            "raw six-vector contract.source_memory_and_eviction_gate",
        )?
        .as_bool()
        .ok_or_else(|| "raw six-vector source-teacher block flag must be boolean".to_owned())?,
    })
}

#[derive(Clone, Debug)]
struct CurrentTraceBinding {
    evidence: Value,
    trace: TraceBinding,
}

fn validate_current_trace(document: &LoadedDocument) -> Result<CurrentTraceBinding, String> {
    let root = object(&document.value, "current trace")?;
    if document.seal_sha256.is_none() {
        return Err("current trace must be sealed".to_owned());
    }
    require_schema_status(
        root,
        CURRENT_TRACE_SCHEMA,
        CURRENT_TRACE_STATUS,
        "current trace",
    )?;
    let trace = validate_trace(
        object(
            required(root, "binding", "current trace")?,
            "current trace.binding",
        )?,
        "current trace.binding",
        "source_template_token_ids_u32le_sha256",
    )?;
    let boundary = object(
        required(root, "claim_boundary", "current trace")?,
        "current trace.claim_boundary",
    )?;
    require_bool(
        required(
            boundary,
            "does_not_execute_or_modify_candidate_runtime_or_artifact",
            "current trace.claim_boundary",
        )?,
        true,
        "current trace must remain read-only",
    )?;
    Ok(CurrentTraceBinding {
        evidence: document_evidence(document),
        trace,
    })
}

#[derive(Clone, Debug)]
struct BridgeBinding {
    evidence: Value,
}

fn bridge_document_sha(
    document: &LoadedDocument,
    bridge: &Map<String, Value>,
    key: &str,
) -> Result<(), String> {
    let upstream = object(
        required(
            bridge,
            "upstream_metadata",
            "dual-attestation/runtime-admission bridge",
        )?,
        "dual-attestation/runtime-admission bridge.upstream_metadata",
    )?;
    let observed = object(
        required(
            upstream,
            key,
            "dual-attestation/runtime-admission bridge.upstream_metadata",
        )?,
        &format!("dual-attestation/runtime-admission bridge.upstream_metadata.{key}"),
    )?;
    let expected_sha = sha256(
        required(
            observed,
            "raw_document_sha256",
            "dual-attestation/runtime-admission bridge upstream",
        )?,
        "dual-attestation/runtime-admission bridge upstream raw document SHA",
    )?;
    if expected_sha != document.raw_sha256 {
        return Err(format!(
            "dual-attestation/runtime-admission bridge {key} raw document SHA drifted"
        ));
    }
    if let Some(seal) = &document.seal_sha256 {
        let observed_seal = sha256(
            required(
                observed,
                "seal_sha256",
                "dual-attestation/runtime-admission bridge upstream",
            )?,
            "dual-attestation/runtime-admission bridge upstream seal",
        )?;
        if observed_seal != *seal {
            return Err(format!(
                "dual-attestation/runtime-admission bridge {key} seal drifted"
            ));
        }
    }
    Ok(())
}

fn validate_bridge(
    document: &LoadedDocument,
    range_document: &LoadedDocument,
    range: &RangeBinding,
    semantics_document: &LoadedDocument,
    raw_document: &LoadedDocument,
    current_document: &LoadedDocument,
    feasibility_document: Option<&LoadedDocument>,
) -> Result<BridgeBinding, String> {
    let root = object(&document.value, "dual-attestation/runtime-admission bridge")?;
    if document.seal_sha256.is_none() {
        return Err("dual-attestation/runtime-admission bridge must be sealed".to_owned());
    }
    require_schema_status(
        root,
        DUAL_BRIDGE_SCHEMA,
        DUAL_BRIDGE_STATUS,
        "dual-attestation/runtime-admission bridge",
    )?;
    bridge_document_sha(range_document, root, "range_authority")?;
    bridge_document_sha(semantics_document, root, "semantics_attester")?;
    bridge_document_sha(raw_document, root, "raw_six_vector_contract")?;
    bridge_document_sha(current_document, root, "current_trace")?;
    let feasibility = feasibility_document.ok_or_else(|| {
        "dual-attestation/runtime-admission bridge requires a supplied sealed streamed feasibility receipt"
            .to_owned()
    })?;
    bridge_document_sha(feasibility, root, "streamed_feasibility")?;
    let upstream = object(
        required(
            root,
            "upstream_metadata",
            "dual-attestation/runtime-admission bridge",
        )?,
        "dual-attestation/runtime-admission bridge.upstream_metadata",
    )?;
    let range_reference = object(
        required(
            upstream,
            "range_authority",
            "dual-attestation/runtime-admission bridge.upstream_metadata",
        )?,
        "dual-attestation/runtime-admission bridge range authority",
    )?;
    if sha256(
        required(
            range_reference,
            "authority_content_sha256",
            "dual-attestation/runtime-admission bridge range authority",
        )?,
        "dual-attestation/runtime-admission bridge authority content SHA",
    )? != range.authority_content_sha256
    {
        return Err(
            "dual-attestation/runtime-admission bridge range authority content SHA drifted"
                .to_owned(),
        );
    }
    let resolution = object(
        required(
            root,
            "schema_resolution",
            "dual-attestation/runtime-admission bridge",
        )?,
        "dual-attestation/runtime-admission bridge.schema_resolution",
    )?;
    if string(
        required(
            resolution,
            "runtime_range_map_schema",
            "dual-attestation/runtime-admission bridge.schema_resolution",
        )?,
        "dual bridge runtime range map schema",
    )? != RUNTIME_RANGE_MAP_SCHEMA
        || string(
            required(
                resolution,
                "runtime_admission_schema",
                "dual-attestation/runtime-admission bridge.schema_resolution",
            )?,
            "dual bridge runtime admission schema",
        )? != RUNTIME_ADMISSION_SCHEMA
        || string(
            required(
                resolution,
                "runtime_admission_status_only_after_bounded_source_validation",
                "dual-attestation/runtime-admission bridge.schema_resolution",
            )?,
            "dual bridge runtime admission status",
        )? != RUNTIME_ADMISSION_STATUS
    {
        return Err(
            "dual-attestation/runtime-admission bridge runtime range-admission grammar drifted"
                .to_owned(),
        );
    }
    for (key, schema, status) in [
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
        let row = object(
            required(
                resolution,
                key,
                "dual-attestation/runtime-admission bridge.schema_resolution",
            )?,
            &format!("dual-attestation/runtime-admission bridge.schema_resolution.{key}"),
        )?;
        require_schema_status(row, schema, status, &format!("dual bridge {key}"))?;
    }
    for key in [
        "both_execution_attestations_required_after_source_child",
        "runtime_range_admission_required_before_payload_open",
        "bridge_does_not_authorize_execution",
    ] {
        require_bool(
            required(
                resolution,
                key,
                "dual-attestation/runtime-admission bridge.schema_resolution",
            )?,
            true,
            &format!("dual-attestation/runtime-admission bridge.{key}"),
        )?;
    }
    let worker = object(
        required(
            root,
            "future_source_worker",
            "dual-attestation/runtime-admission bridge",
        )?,
        "dual-attestation/runtime-admission bridge.future_source_worker",
    )?;
    for (field, expected) in [
        ("maximum_positioned_read_bytes", MAX_POSITIONED_READ_BYTES),
        ("source_layers", SOURCE_LAYERS),
        ("source_forwards", SOURCE_FORWARDS),
        ("source_f32le_vectors", 2),
        ("native_f32le_vectors", 4),
    ] {
        if u64_value(
            required(
                worker,
                field,
                "dual-attestation/runtime-admission bridge.future_source_worker",
            )?,
            &format!("dual bridge future worker.{field}"),
        )? != expected
        {
            return Err(format!(
                "dual-attestation/runtime-admission bridge future worker {field} drifted"
            ));
        }
    }
    for key in [
        "one_bounded_window_only",
        "source_payloads_durable_before_eviction",
        "close_handles_and_clear_cache_before_eviction_receipt",
        "separate_native_four_vector_phase_required",
    ] {
        require_bool(
            required(
                worker,
                key,
                "dual-attestation/runtime-admission bridge.future_source_worker",
            )?,
            true,
            &format!("dual-attestation/runtime-admission bridge future worker.{key}"),
        )?;
    }
    Ok(BridgeBinding {
        evidence: document_evidence(document),
    })
}

fn future_child_grammar() -> Value {
    json!({
        "binary": "ascension_qwen30_streamed_source_teacher_child",
        "command": [
            "ascension_qwen30_streamed_source_teacher_child",
            "--source-root", "ABSOLUTE_CANONICAL_QWEN30_SOURCE_ROOT",
            "--runtime-admission", "ABSOLUTE_SEALED_RUNTIME_ADMISSION_JSON",
            "--dual-attestation-runtime-admission", "ABSOLUTE_SEALED_DUAL_BRIDGE_JSON",
            "--source-lease", "ABSOLUTE_SEALED_ONE_SHOT_SOURCE_LEASE_JSON",
            "--capture-dir", "NEW_ABSOLUTE_SOURCE_CHILD_CAPTURE_DIRECTORY"
        ],
        "input_grammar": {
            "runtime_admission": {"schema": RUNTIME_ADMISSION_SCHEMA, "status_only_after_bounded_source_validation": RUNTIME_ADMISSION_STATUS},
            "dual_attestation_runtime_admission": {"schema": DUAL_BRIDGE_SCHEMA, "status": DUAL_BRIDGE_STATUS},
            "source_lease": {
                "schema": SOURCE_LEASE_SCHEMA,
                "status": SOURCE_LEASE_STATUS,
                "fresh_one_shot_exact_launch_required": true,
                "lease_must_not_be_issued_or_read_by_this_preflight": true
            },
            "source_root": {
                "accepted_only_by_future_authorized_child": true,
                "regular_non_symlink_shards_only": true,
                "no_mmap_or_full_shard_cache": true
            }
        },
        "execution_shape": {
            "maximum_positioned_read_bytes": MAX_POSITIONED_READ_BYTES,
            "maximum_live_raw_bf16_windows": 1,
            "source_layers": SOURCE_LAYERS,
            "source_forwards": SOURCE_FORWARDS,
            "prefix_tokens": PREFIX_TOKENS,
            "forced_token_id": FORCED_TOKEN_ID,
            "sampling_or_autoregressive_feedback_forbidden": true,
            "source_operator_order": [
                "embedding", "rmsnorm", "qkv_serial_k", "rope_kv_append_then_causal_read",
                "attention_output_serial_k", "residual", "router_top8", "one_selected_expert_body_at_a_time",
                "source_ordered_route_combine", "second_residual", "final_rmsnorm", "lm_head_serial_k"
            ]
        },
        "worker_output_grammar": {
            "worker_evidence_schema": "hawking.ascension.qwen30_streamed_source_teacher_child_execution_evidence.v1",
            "worker_evidence_status_only_after_real_source_execution": "CAPTURED_QWEN30_STREAMED_SOURCE_TEACHER_CHILD_TWO_F32LE_LOGITS_NOT_NATIVE_PHASE",
            "source_payloads": SOURCE_PAYLOADS,
            "source_payload_dtype": "f32le",
            "source_payload_vocab_rows": VOCAB_ROWS,
            "source_payload_bytes_each": F32_VECTOR_BYTES,
            "required_evidence": [
                "all_16_shard and range-admission identities",
                "bounded positioned-read/cache accounting",
                "all_48_layers_x_370_forwards operator/order evidence",
                "both full F32LE source logits fsynced before worker completion",
                "source handles closed and reader cache zeroed before child exit",
                "both named execution-attestation receipts"
            ],
            "worker_must_not_write_outer_source_terminal_or_native_payloads": true
        },
        "exact_outer_handoff_fields": {
            "source_worker_to_outer": {
                "required_worker_evidence_fields": [
                    "source_payloads.exact_prefix", "source_payloads.forced_shared_continuation",
                    "source_payloads_are_create_new_f32le_finite_and_fsynced",
                    "bounded_per_read_cache.maximum_allowed_window_bytes",
                    "bounded_per_read_cache.maximum_observed_window_bytes",
                    "bounded_per_read_cache.maximum_cached_bytes",
                    "bounded_per_read_cache.maximum_cached_windows",
                    "source_payload_read_accounting.per_shard",
                    "runtime_range_admission.seal_sha256",
                    "operator_accumulation_execution_attestation.seal_sha256",
                    "range_reader_exact_semantics_attestation.seal_sha256",
                    "source_handles_closed", "streamed_reader_cache_zeroed", "source_backend_shutdown",
                    "child_exit_after_payload_fsyncs"
                ],
                "worker_must_not_write_source_terminal": true,
                "worker_must_not_start_native_phase": true
            },
            "outer_after_reap_source_terminal": {
                "schema": SOURCE_TERMINAL_SCHEMA,
                "status": SOURCE_TERMINAL_STATUS,
                "required_fields": [
                    "source_lease.seal_sha256", "exact_trace", "streamed_execution.mode=layer_streamed_bf16_source_teacher",
                    "streamed_execution.outer_reaped_child_before_terminal_receipt=true",
                    "streamed_execution.receipt_written_after_payload_fsyncs=true", "source_payloads",
                    "bounded_per_read_cache", "source_payload_read_accounting"
                ]
            },
            "outer_source_eviction_before_native": {
                "schema": SOURCE_EVICTION_SCHEMA,
                "status": SOURCE_EVICTION_STATUS,
                "required_fields": [
                    "source_teacher_terminal.seal_sha256", "eviction.source_weights_evicted=true",
                    "eviction.source_backend_shutdown=true", "eviction.source_model_residency_released=true",
                    "eviction.streamed_reader_cache_cleared=true", "eviction.source_payloads_durable_and_immutable=true",
                    "eviction.swap_remained_zero=true", "eviction.pre_native_lease_process_tree_checked=true"
                ]
            },
            "separate_native_phase": {
                "lease": {"schema": NATIVE_LEASE_SCHEMA, "status": NATIVE_LEASE_STATUS},
                "terminal": {"schema": NATIVE_TERMINAL_SCHEMA, "status": NATIVE_TERMINAL_STATUS},
                "native_lease_must_bind_source_eviction_seal": true,
                "native_phase_may_write_only_the_four_named_native_payloads": NATIVE_PAYLOADS
            },
            "six_vector_terminal_rule": {
                "all_six_payloads_must_be_fsynced_before_terminal": true,
                "metric_scoring_is_outside_the_source_child_and_outer_handoff": true
            }
        },
        "outer_sequence_after_child": [
            "outer reaps source child before source terminal receipt",
            "outer writes source terminal only after both source payload fsyncs",
            "outer proves source backend shutdown, handles closed, cache cleared, and source residency released",
            "outer writes source-eviction receipt",
            "only then may a distinct native lease start the existing four-vector native phase",
            "six-vector terminal receipt is written last after all six payload fsyncs"
        ]
    })
}

fn prepared_document(
    range: RangeBinding,
    semantics: SemanticsBinding,
    feasibility: Option<FeasibilityBinding>,
    raw: RawSixVectorBinding,
    current: CurrentTraceBinding,
    bridge: Option<BridgeBinding>,
) -> Result<Value, String> {
    let mut blockers = Vec::<Value>::new();
    match &feasibility {
        None => blockers.push(Value::String(
            "sealed_streamed_feasibility_receipt_absent".to_owned(),
        )),
        Some(feasibility) => {
            if feasibility.status != FEASIBILITY_PREPARED_STATUS {
                blockers.push(Value::String(
                    "streamed_feasibility_is_refused_or_not_prepared".to_owned(),
                ));
            }
            if !feasibility.semantic_equivalence_proven {
                blockers.push(Value::String(
                    "exact_source_semantics_attestation_not_earned".to_owned(),
                ));
            }
            if !feasibility.streamed_memory_arithmetic_fits || !feasibility.zero_swap_condition_met
            {
                blockers.push(Value::String(
                    "streamed_memory_or_zero_swap_precondition_not_currently_met".to_owned(),
                ));
            }
        }
    }
    if bridge.is_none() {
        blockers.push(Value::String(
            "sealed_dual_attestation_runtime_admission_bridge_absent".to_owned(),
        ));
    }
    if raw.source_teacher_currently_blocked {
        blockers.push(Value::String(
            "current_raw_six_vector_contract_marks_source_teacher_capture_blocked".to_owned(),
        ));
    }
    blockers.extend([
        Value::String("no_fresh_one_shot_source_lease_or_runtime_admission_exists".to_owned()),
        Value::String("no_real_source_teacher_child_or_execution_attestations_exist".to_owned()),
        Value::String("no_source_terminal_eviction_or_distinct_native_lease_exists".to_owned()),
    ]);
    let feasibility_value = feasibility.map(|item| {
        json!({
            "present": true,
            "evidence": item.evidence,
            "status": item.status,
            "semantic_equivalence_proven": item.semantic_equivalence_proven,
            "streamed_memory_arithmetic_fits": item.streamed_memory_arithmetic_fits,
            "zero_swap_condition_met": item.zero_swap_condition_met,
        })
    });
    let bridge_value = bridge.map(|item| json!({"present": true, "evidence": item.evidence}));
    seal_value(json!({
        "schema": RESULT_SCHEMA,
        "status": PREPARED_STATUS,
        "input_bindings": {
            "metadata_range_authority": {
                "evidence": range.evidence,
                "authority_content_sha256": range.authority_content_sha256,
                "source_revision": range.source_revision,
                "source_index_sha256": range.source_index_sha256,
                "maximum_declared_bf16_row_window_bytes": range.maximum_window_bytes,
            },
            "metadata_operator_semantics_attester": {
                "evidence": semantics.evidence,
                "source_revision": semantics.source_revision,
                "source_index_sha256": semantics.source_index_sha256,
            },
            "streamed_feasibility": feasibility_value.unwrap_or_else(|| json!({"present": false})),
            "raw_six_vector_contract": raw.evidence,
            "current_trace": current.evidence,
            "dual_attestation_runtime_admission_bridge": bridge_value.unwrap_or_else(|| json!({"present": false})),
        },
        "trace_binding": {
            "source_template_token_count": PREFIX_TOKENS,
            "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
            "source_template_token_ids_u32le_sha256": current.trace.token_sha256,
            "sampling_or_autoregressive_feedback_forbidden": true,
        },
        "future_child_interface": future_child_grammar(),
        "required_dual_schema_resolution": {
            "metadata_range_authority_is_not_the_flat_runtime_range_map": true,
            "future_runtime_range_map_schema": RUNTIME_RANGE_MAP_SCHEMA,
            "future_runtime_admission_schema": RUNTIME_ADMISSION_SCHEMA,
            "future_operator_accumulation_attestation": {"schema": OPERATOR_ATTESTATION_SCHEMA, "status": OPERATOR_ATTESTATION_STATUS},
            "future_range_reader_exact_semantics_attestation": {"schema": RANGE_READER_ATTESTATION_SCHEMA, "status": RANGE_READER_ATTESTATION_STATUS},
            "both_execution_attestations_must_bind_the_same_runtime_admission_and_source_payloads": true,
            "a_prepared_bridge_is_non_authorizing_and_cannot_substitute_for_either_execution_attestation": true,
        },
        "current_blockers": blockers,
        "execution_authorized": false,
        "execution_boundary": {
            "source_tensor_payload_opened": false,
            "source_model_loaded_or_instantiated": false,
            "whole_source_model_resident": false,
            "gpu_metal_mps_or_other_accelerator_invoked": false,
            "server_started_or_contacted": false,
            "hcli_invoked": false,
            "lease_requested_issued_or_consumed": false,
            "child_process_started": false,
            "source_teacher_or_native_vector_written": false,
            "source_eviction_or_native_phase_performed": false,
        },
        "claim_boundary": "Prepared CPU/build interface only. This record does not authorize or report a source teacher, source payload read, source/native comparison, quality result, coherence, HCLI, TPS, TG, serving, promotion, or tournament result.",
    }))
}

fn refusal_document(reason: &str) -> Result<Value, String> {
    seal_value(json!({
        "schema": RESULT_SCHEMA,
        "status": REFUSED_STATUS,
        "refusal_reason": reason,
        "execution_authorized": false,
        "execution_boundary": {
            "source_tensor_payload_opened": false,
            "source_model_loaded_or_instantiated": false,
            "whole_source_model_resident": false,
            "gpu_metal_mps_or_other_accelerator_invoked": false,
            "server_started_or_contacted": false,
            "hcli_invoked": false,
            "lease_requested_issued_or_consumed": false,
            "child_process_started": false,
            "source_teacher_or_native_vector_written": false,
            "source_eviction_or_native_phase_performed": false,
        },
        "claim_boundary": "Refusal is a CPU-only metadata result; no source payload or runtime operation occurred.",
    }))
}

fn build_preflight(
    range_document: &LoadedDocument,
    semantics_document: &LoadedDocument,
    feasibility_document: Option<&LoadedDocument>,
    raw_document: &LoadedDocument,
    current_document: &LoadedDocument,
    bridge_document: Option<&LoadedDocument>,
) -> Result<Value, String> {
    let current = validate_current_trace(current_document)?;
    let range = validate_range_authority(range_document)?;
    let semantics = validate_semantics(semantics_document, &range)?;
    let feasibility = feasibility_document
        .map(|document| validate_feasibility(document, &current.trace))
        .transpose()?;
    let raw = validate_raw_six_vector(raw_document, &current.trace)?;
    let bridge = bridge_document
        .map(|document| {
            validate_bridge(
                document,
                range_document,
                &range,
                semantics_document,
                raw_document,
                current_document,
                feasibility_document,
            )
        })
        .transpose()?;
    prepared_document(range, semantics, feasibility, raw, current, bridge)
}

fn write_new_json(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() || path.exists() || !path.parent().is_some_and(Path::is_dir) {
        return Err("--out must be a new absolute path below an existing parent".to_owned());
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize preflight receipt: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| {
            format!(
                "cannot create preflight receipt {}: {error}",
                path.display()
            )
        })?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot fsync preflight receipt {}: {error}", path.display()))?;
    Ok(())
}

fn run(args: Args) -> Result<Value, String> {
    let range = read_metadata_document(&args.range_authority, "range authority", false)?;
    let semantics = read_metadata_document(&args.semantics, "operator semantics attester", false)?;
    let feasibility = args
        .feasibility
        .as_deref()
        .map(|path| read_metadata_document(path, "streamed feasibility", true))
        .transpose()?;
    let raw = read_metadata_document(&args.raw_six_vector, "raw six-vector contract", true)?;
    let current = read_metadata_document(&args.current_trace, "current trace", true)?;
    let bridge = args
        .dual_bridge
        .as_deref()
        .map(|path| read_metadata_document(path, "dual-attestation/runtime-admission bridge", true))
        .transpose()?;
    let result = match build_preflight(
        &range,
        &semantics,
        feasibility.as_ref(),
        &raw,
        &current,
        bridge.as_ref(),
    ) {
        Ok(value) => value,
        Err(reason) => refusal_document(&reason)?,
    };
    write_new_json(&args.out, &result)?;
    Ok(result)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(document) => match serde_json::to_string_pretty(&document) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("cannot render preflight receipt: {error}");
                process::exit(1);
            }
        },
        Err(error) => {
            eprintln!(
                "Q30 streamed source-teacher child preflight refused before receipt: {error}"
            );
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sealed(value: Value) -> Value {
        seal_value(value).expect("fixture seals")
    }

    fn hash(value: &str) -> String {
        sha256_hex(value.as_bytes())
    }

    fn trace() -> Value {
        json!({
            "probe_id": "literal_hawking",
            "source_template_token_count": PREFIX_TOKENS,
            "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
            "source_template_token_ids_u32le_sha256": hash("literal-hawking-token-ids"),
        })
    }

    fn range_authority(window_rows: u64) -> LoadedDocument {
        let tensors = (0..SOURCE_TENSORS)
            .map(|index| {
                json!({
                    "tensor_name": format!("tensor.{index}"),
                    "source_dtype": "BF16",
                    "row_window_shape": [window_rows, 2048],
                })
            })
            .collect::<Vec<_>>();
        memory_document(
            "/fixtures/range-authority.json",
            json!({
                "authority_content_sha256": hash("range-authority-content"),
                "authority": {
                    "schema": RANGE_AUTHORITY_SCHEMA,
                    "status": RANGE_AUTHORITY_STATUS,
                    "source": {
                        "model_id": SOURCE_MODEL_ID,
                        "source_revision": "b2cff646eb4bb1d68355c01b18ae02e7cf42d120",
                        "source_tensor_count": SOURCE_TENSORS,
                        "source_shard_count": SOURCE_SHARDS,
                        "source_index": {"sha256": hash("source-index"), "weight_map_tensor_count": SOURCE_TENSORS},
                    },
                    "exact_streamed_oracle_scope": {
                        "source_template_token_count": PREFIX_TOKENS,
                        "forced_identical_continuation_token_id": FORCED_TOKEN_ID,
                        "layers": SOURCE_LAYERS,
                        "top_k_routes_per_token": TOP_K,
                        "row_tile_rows": 128,
                        "total_forwards_per_replay_arm": SOURCE_FORWARDS,
                        "sampling_or_autoregressive_feedback_forbidden": true,
                    },
                    "metadata_access_boundary": {
                        "source_tensor_payload_bytes_read": 0,
                        "mmap_or_memory_map_used": false,
                        "source_model_instantiated": false,
                        "gpu_or_metal_invoked": false,
                        "server_started": false,
                        "hcli_invoked": false,
                        "lease_requested": false,
                        "tensor_payload_hashes_collected": false,
                        "whole_shard_payload_checksum_collected": false,
                    },
                    "tensors": tensors,
                }
            }),
            false,
        )
    }

    fn semantics(range: &LoadedDocument) -> LoadedDocument {
        let authority = range.value["authority"]
            .as_object()
            .expect("authority object");
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
                    "source_revision": authority["source"]["source_revision"],
                    "source_index_sha256": authority["source"]["source_index"]["sha256"],
                },
                "future_exact_execution_attestation": {
                    "schema": OPERATOR_ATTESTATION_SCHEMA,
                    "status_only_after_real_separately_leased_source_execution": OPERATOR_ATTESTATION_STATUS,
                }
            }),
            false,
        )
    }

    fn current_trace() -> LoadedDocument {
        memory_document(
            "/fixtures/current-trace.json",
            sealed(json!({
                "schema": CURRENT_TRACE_SCHEMA,
                "status": CURRENT_TRACE_STATUS,
                "binding": trace(),
                "claim_boundary": {"does_not_execute_or_modify_candidate_runtime_or_artifact": true},
            })),
            true,
        )
    }

    fn raw_six_vector() -> LoadedDocument {
        memory_document(
            "/fixtures/raw-six-vector.json",
            sealed(json!({
                "schema": RAW_SIX_VECTOR_SCHEMA,
                "status": RAW_SIX_VECTOR_STATUS,
                "replay_binding": {"exact_trace": trace()},
                "six_vector_retention_contract": {
                    "dtype": "f32le",
                    "vocab_rows": VOCAB_ROWS,
                    "bytes_per_vector": F32_VECTOR_BYTES,
                    "required_payload_count": 6,
                    "required_total_payload_bytes": F32_VECTOR_BYTES * 6,
                    "required_payloads": expected_payloads(),
                    "receipt_must_be_written_after_all_six_payloads_and_fsyncs": true,
                },
                "source_memory_and_eviction_gate": {"source_teacher_capture_is_currently_blocked": true},
            })),
            true,
        )
    }

    fn feasibility(prepared: bool) -> LoadedDocument {
        memory_document(
            "/fixtures/feasibility.json",
            sealed(json!({
                "schema": FEASIBILITY_SCHEMA,
                "status": if prepared { FEASIBILITY_PREPARED_STATUS } else { FEASIBILITY_REFUSED_STATUS },
                "exact_trace": {
                    "prefix_token_count": PREFIX_TOKENS,
                    "forced_token_id": FORCED_TOKEN_ID,
                    "source_template_token_ids_u32le_sha256": trace()["source_template_token_ids_u32le_sha256"],
                },
                "memory_assessment": {
                    "streamed_memory_arithmetic_fits": prepared,
                    "zero_swap_condition_met": prepared,
                },
                "feasibility": {
                    "oracle_execution_authorized": false,
                    "semantic_equivalence_proven_by_external_sealed_attestation": prepared,
                },
            })),
            true,
        )
    }

    fn bridge(
        range: &LoadedDocument,
        semantics: &LoadedDocument,
        feasibility: &LoadedDocument,
        raw: &LoadedDocument,
        current: &LoadedDocument,
    ) -> LoadedDocument {
        let upstream = |document: &LoadedDocument, authority_content: Option<String>| {
            let mut value = json!({"raw_document_sha256": document.raw_sha256, "seal_sha256": document.seal_sha256});
            if let Some(authority_content) = authority_content {
                value.as_object_mut().expect("upstream object").insert(
                    "authority_content_sha256".to_owned(),
                    Value::String(authority_content),
                );
            }
            value
        };
        memory_document(
            "/fixtures/dual-bridge.json",
            sealed(json!({
                "schema": DUAL_BRIDGE_SCHEMA,
                "status": DUAL_BRIDGE_STATUS,
                "upstream_metadata": {
                    "range_authority": upstream(range, Some(range.value["authority_content_sha256"].as_str().expect("content hash").to_owned())),
                    "semantics_attester": upstream(semantics, None),
                    "streamed_feasibility": upstream(feasibility, None),
                    "raw_six_vector_contract": upstream(raw, None),
                    "current_trace": upstream(current, None),
                },
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
                }
            })),
            true,
        )
    }

    #[test]
    fn current_inputs_without_bridge_are_prepared_but_inert() {
        let range = range_authority(128);
        let semantics = semantics(&range);
        let raw = raw_six_vector();
        let current = current_trace();
        let result = build_preflight(&range, &semantics, None, &raw, &current, None)
            .expect("current static inputs should describe an inert preflight");
        assert_eq!(result["status"], PREPARED_STATUS);
        assert_eq!(result["execution_authorized"], false);
        assert_eq!(
            result["execution_boundary"]["source_tensor_payload_opened"],
            false
        );
        assert_eq!(
            result["future_child_interface"]["execution_shape"]["maximum_positioned_read_bytes"],
            MAX_POSITIONED_READ_BYTES
        );
        assert!(result["current_blockers"]
            .as_array()
            .expect("blockers")
            .iter()
            .any(|item| item == "sealed_dual_attestation_runtime_admission_bridge_absent"));
        verify_seal(&result, "prepared result").expect("prepared result seals");
    }

    #[test]
    fn valid_bridge_resolves_both_schema_routes_but_never_activates() {
        let range = range_authority(128);
        let semantics = semantics(&range);
        let feasibility = feasibility(true);
        let raw = raw_six_vector();
        let current = current_trace();
        let bridge = bridge(&range, &semantics, &feasibility, &raw, &current);
        let result = build_preflight(
            &range,
            &semantics,
            Some(&feasibility),
            &raw,
            &current,
            Some(&bridge),
        )
        .expect("bridge must bind the two future receipt schemas");
        assert_eq!(result["status"], PREPARED_STATUS);
        assert_eq!(result["execution_authorized"], false);
        assert_eq!(
            result["input_bindings"]["dual_attestation_runtime_admission_bridge"]["present"],
            true
        );
        assert_eq!(
            result["future_child_interface"]["worker_output_grammar"]["source_payloads"],
            json!(SOURCE_PAYLOADS)
        );
        assert_eq!(
            result["future_child_interface"]["worker_output_grammar"]["source_payload_bytes_each"],
            F32_VECTOR_BYTES
        );
    }

    #[test]
    fn bridge_with_wrong_authority_identity_refuses_without_execution() {
        let range = range_authority(128);
        let semantics = semantics(&range);
        let feasibility = feasibility(true);
        let raw = raw_six_vector();
        let current = current_trace();
        let mut bridge = bridge(&range, &semantics, &feasibility, &raw, &current);
        bridge.value["upstream_metadata"]["range_authority"]["authority_content_sha256"] =
            Value::String("0".repeat(64));
        bridge.value = seal_value(bridge.value).expect("re-seal malformed bridge");
        bridge.seal_sha256 =
            Some(verify_seal(&bridge.value, "malformed bridge").expect("bridge still seals"));
        let error = build_preflight(
            &range,
            &semantics,
            Some(&feasibility),
            &raw,
            &current,
            Some(&bridge),
        )
        .expect_err("bridge identity drift must fail closed");
        let refusal = refusal_document(&error).expect("refusal seals");
        assert_eq!(refusal["status"], REFUSED_STATUS);
        assert_eq!(
            refusal["execution_boundary"]["child_process_started"],
            false
        );
    }

    #[test]
    fn over_one_mib_metadata_window_refuses_before_any_source_reader_exists() {
        let range = range_authority(257);
        let error =
            validate_range_authority(&range).expect_err("257x2048 BF16 bytes exceed one MiB");
        assert!(error.contains("source-reader ceiling"));
    }

    #[test]
    fn parser_rejects_source_root_or_execution_options() {
        let error = parse_args_from(vec![
            "--range-authority".to_owned(),
            "/tmp/range.json".to_owned(),
            "--semantics".to_owned(),
            "/tmp/semantics.json".to_owned(),
            "--raw-six-vector".to_owned(),
            "/tmp/raw.json".to_owned(),
            "--current-trace".to_owned(),
            "/tmp/current.json".to_owned(),
            "--out".to_owned(),
            "/tmp/out.json".to_owned(),
            "--source-root".to_owned(),
            "/model".to_owned(),
        ])
        .expect_err("preflight must expose no source execution surface");
        assert!(error.contains("unsupported option"));
    }
}
