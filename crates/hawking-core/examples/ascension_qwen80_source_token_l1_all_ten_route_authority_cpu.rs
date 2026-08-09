//! CPU-only source-token Qwen80 Layer-1 router/all-ten authority producer.
//!
//! This program has two deliberately separated modes:
//!
//! * preflight validates only sealed provenance documents and the selected
//!   producer binary. It never opens the admitted artifact or a compact
//!   payload.
//! * cpu-oracle is a future, outer-authorized one-shot CPU child. It
//!   performs exactly one complete-artifact admission, replays source token
//!   one through a fresh L0 full MoE CPU oracle and a fresh L1 full MoE CPU
//!   oracle, and seals the dynamic L1 router/all-ten authority.
//!
//! The latter mode is intentionally not run by tests or this development
//! task. It has no Metal, lease, watcher, server, HCLI, or TPS path. The
//! caller must first obtain the separate receipt-last outer/replay approval.

#![recursion_limit = "256"]

use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CanonicalLinearLayerCpuInput, Qwen80CompleteArtifactCatalog, Qwen80PackedTensorBinding,
};
use hawking_core::model::qwen_complete_binary::{
    CompleteBinaryAdmission, CompleteBinaryHeader, QwenCompleteBinaryModel,
};
use hawking_core::moe::route_tie_epsilon;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process;

const PRODUCER_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority_producer_preflight.v1";
const PRODUCER_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_CPU_PRODUCER_NOT_EXECUTED";
const OUTER_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority_outer_launch_authority.v1";
const OUTER_AUTHORITY_STATUS: &str =
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_CPU_CHILD_ONE_SHOT";
const AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1";
const AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_SAME_RUNTIME_MOE_SUFFIX";

const JOINT_ASSESSMENT_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1";
const JOINT_ASSESSMENT_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER";
const COMPLETION_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_moe_completion_preflight.v1";
const COMPLETION_PREFLIGHT_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_MOE_COMPLETION_ROUTE_AUTHORITY_REQUIRED_NOT_LEASED_OR_EXECUTED";

const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_POINTER_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_POINTER_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL_SHA256: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL_SHA256: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";

const SOURCE_TOKEN_ID: u32 = 1;
const L0_LAYER: usize = 0;
const L1_LAYER: usize = 1;
const L1_LINEAR_STATE_SLOT: usize = 1;
const HIDDEN: usize = 2_048;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const GROUP_SIZE: usize = 128;
const L0_DISPATCHES: usize = 23;
const L1_PREFIX_DISPATCHES: usize = 9;
const MAX_WORKERS: usize = 4;
const MAX_ROUTE_WEIGHT_SUM_ERROR: f32 = 2.0e-6;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Preflight,
    CpuOracle,
}

#[derive(Clone, Debug)]
struct Args {
    mode: Mode,
    manifest: PathBuf,
    admission_current: PathBuf,
    joint_assessment: PathBuf,
    completion_preflight: PathBuf,
    producer_binary: PathBuf,
    producer_preflight: Option<PathBuf>,
    outer_launch_authority: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    workers: Option<usize>,
    out: PathBuf,
}

#[derive(Clone, Debug)]
struct BoundDocument {
    path: PathBuf,
    bytes: u64,
    raw_sha256: String,
    document_sha256: String,
    document_seal_sha256: String,
    value: Value,
}

#[derive(Clone, Debug)]
struct SourceIdentity {
    manifest: BoundDocument,
    admission_current: BoundDocument,
    admission_receipt: BoundDocument,
    manifest_seal_sha256: String,
    admission_pointer_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_audit_seal_sha256: String,
    source_revision: String,
}

#[derive(Clone, Debug)]
struct PreflightInputs {
    source: SourceIdentity,
    joint_assessment: BoundDocument,
    completion_preflight: BoundDocument,
    producer_binary: FileEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct FileEvidence {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl FileEvidence {
    fn json(&self) -> Value {
        json!({
            "path": self.path,
            "present": true,
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_l1_all_ten_route_authority_cpu \\\n+--mode preflight|cpu-oracle \\\n+--manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH \\\n+--joint-assessment ABSOLUTE_PATH --completion-preflight ABSOLUTE_PATH \\\n+--producer-binary ABSOLUTE_EXECUTABLE \\\n+[--producer-preflight ABSOLUTE_SEALED_PREFLIGHT \\\n+  --outer-launch-authority ABSOLUTE_SEALED_AUTHORITY \\\n+  --capture-dir ABSOLUTE_EXISTING_DIRECTORY --workers 1..4] \\\n+--out ABSOLUTE_NEW_FILE"
}

fn parse_mode(value: &str) -> Result<Mode, String> {
    match value {
        "preflight" => Ok(Mode::Preflight),
        "cpu-oracle" => Ok(Mode::CpuOracle),
        _ => Err(format!(
            "--mode must be preflight or cpu-oracle; {}",
            usage()
        )),
    }
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--mode"
            | "--manifest"
            | "--admission-current"
            | "--joint-assessment"
            | "--completion-preflight"
            | "--producer-binary"
            | "--producer-preflight"
            | "--outer-launch-authority"
            | "--capture-dir"
            | "--workers"
            | "--out" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} may not be repeated; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        }
    }
    let mode = parse_mode(
        &values
            .remove("--mode")
            .ok_or_else(|| format!("missing --mode; {}", usage()))?,
    )?;
    let mut required_path = |flag: &str| -> Result<PathBuf, String> {
        let path = PathBuf::from(
            values
                .remove(flag)
                .ok_or_else(|| format!("missing {flag}; {}", usage()))?,
        );
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    let manifest = required_path("--manifest")?;
    let admission_current = required_path("--admission-current")?;
    let joint_assessment = required_path("--joint-assessment")?;
    let completion_preflight = required_path("--completion-preflight")?;
    let producer_binary = required_path("--producer-binary")?;
    let out = required_path("--out")?;
    let optional_path =
        |flag: &str, values: &mut BTreeMap<String, String>| -> Result<Option<PathBuf>, String> {
            values
                .remove(flag)
                .map(PathBuf::from)
                .map(|path| {
                    if path.is_absolute() {
                        Ok(path)
                    } else {
                        Err(format!("{flag} must be absolute"))
                    }
                })
                .transpose()
        };
    let producer_preflight = optional_path("--producer-preflight", &mut values)?;
    let outer_launch_authority = optional_path("--outer-launch-authority", &mut values)?;
    let capture_dir = optional_path("--capture-dir", &mut values)?;
    let workers = values
        .remove("--workers")
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("--workers must be an integer: {error}"))
        })
        .transpose()?;
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    if out.exists() || !out.parent().is_some_and(Path::is_dir) {
        return Err("--out must be a new file beneath an existing parent".into());
    }
    match mode {
        Mode::Preflight => {
            if producer_preflight.is_some()
                || outer_launch_authority.is_some()
                || capture_dir.is_some()
                || workers.is_some()
            {
                return Err(
                    "preflight mode refuses CPU-oracle authority, capture, or worker flags".into(),
                );
            }
        }
        Mode::CpuOracle => {
            let capture_dir = capture_dir
                .as_deref()
                .ok_or("cpu-oracle mode requires --capture-dir")?;
            if !capture_dir.is_dir() {
                return Err(
                    "--capture-dir must be an existing directory owned by the outer".into(),
                );
            }
            if !out.starts_with(capture_dir) {
                return Err("--out must be beneath --capture-dir in cpu-oracle mode".into());
            }
            if producer_preflight.is_none() || outer_launch_authority.is_none() {
                return Err(
                    "cpu-oracle mode requires --producer-preflight and --outer-launch-authority"
                        .into(),
                );
            }
            let workers = workers.ok_or("cpu-oracle mode requires --workers")?;
            if !(1..=MAX_WORKERS).contains(&workers) {
                return Err(format!("--workers must be in 1..={MAX_WORKERS}"));
            }
        }
    }
    Ok(Args {
        mode,
        manifest,
        admission_current,
        joint_assessment,
        completion_preflight,
        producer_binary,
        producer_preflight,
        outer_launch_authority,
        capture_dir,
        workers,
        out,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !is_lower_sha256(value) {
        return Err(format!("{label} must be a lowercase SHA-256"));
    }
    Ok(())
}

/// Match the sorted, compact Python receipt canonicalization for finite
/// numbers. Rust-created documents are consumed by Python outer controllers.
fn python_json_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("canonical JSON float must be finite")?;
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        });
    }
    let raw = number.to_string();
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.as_str()),
    };
    let (mantissa, exponent) = match unsigned.find('e').or_else(|| unsigned.find('E')) {
        Some(index) => (
            &unsigned[..index],
            unsigned[index + 1..]
                .parse::<i32>()
                .map_err(|error| format!("invalid float exponent: {error}"))?,
        ),
        None => (unsigned, 0),
    };
    let mut fractional = 0_i32;
    let mut after_decimal = false;
    let mut digits = String::new();
    for byte in mantissa.bytes() {
        match byte {
            b'.' if !after_decimal => after_decimal = true,
            b'0'..=b'9' => {
                if after_decimal {
                    fractional = fractional
                        .checked_add(1)
                        .ok_or("canonical float fractional length overflow")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err(format!("invalid canonical float mantissa {raw:?}")),
        }
    }
    let first = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("nonzero canonical float has no significant digit")?;
    let mut significant = digits[first..].to_owned();
    let mut decimal_power = exponent
        .checked_sub(fractional)
        .ok_or("canonical decimal exponent overflow")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        decimal_power = decimal_power
            .checked_add(1)
            .ok_or("canonical decimal exponent overflow")?;
    }
    let scientific = decimal_power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("canonical scientific exponent overflow")?;
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&scientific) {
        let mut rendered = significant[..1].to_owned();
        if significant.len() > 1 {
            rendered.push('.');
            rendered.push_str(&significant[1..]);
        }
        let exponent_sign = if scientific < 0 { '-' } else { '+' };
        return Ok(format!(
            "{sign}{rendered}e{exponent_sign}{:02}",
            scientific.unsigned_abs()
        ));
    }
    let position = scientific + 1;
    let rendered = if position <= 0 {
        format!(
            "0.{}{}",
            "0".repeat(usize::try_from(-position).unwrap_or(usize::MAX)),
            significant
        )
    } else if usize::try_from(position).unwrap_or(usize::MAX) >= significant.len() {
        format!(
            "{}{}.0",
            significant,
            "0".repeat(usize::try_from(position).unwrap_or(usize::MAX) - significant.len())
        )
    } else {
        let position = usize::try_from(position).map_err(|_| "decimal position is negative")?;
        format!("{}.{}", &significant[..position], &significant[position..])
    };
    Ok(format!("{sign}{rendered}"))
}

fn canonical_json_into(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                output.push_str(&number.to_string());
            } else {
                output.push_str(&python_json_float(number)?);
            }
        }
        Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("cannot render JSON string: {error}"))?,
        ),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                canonical_json_into(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut first = true;
            for (key, value) in values {
                if !first {
                    output.push(',');
                }
                first = false;
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("cannot render JSON key: {error}"))?,
                );
                output.push(':');
                canonical_json_into(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, String> {
    let mut rendered = String::new();
    canonical_json_into(value, &mut rendered)?;
    Ok(rendered.into_bytes())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn array_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    object
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label}.{field} must be an array"))
}

fn string<'a>(object: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
}

fn require_exact_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let actual = string(object, field, label)?;
    if actual != expected {
        return Err(format!("{label}.{field}={actual:?}, expected {expected:?}"));
    }
    Ok(())
}

fn require_sha_field(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<String, String> {
    let value = string(object, field, label)?;
    require_sha256(value, &format!("{label}.{field}"))?;
    Ok(value.to_owned())
}

fn require_bool(
    object: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if object.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label}.{field} must be {expected}"));
    }
    Ok(())
}

fn require_u64(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn document_sha256(value: &Value, label: &str) -> Result<String, String> {
    let root = object(value, label)?;
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    Ok(sha256_hex(&canonical_json_bytes(&Value::Object(unsigned))?))
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let root = object(value, label)?;
    let observed = string(root, "seal_sha256", label)?;
    require_sha256(observed, &format!("{label}.seal_sha256"))?;
    let expected = document_sha256(value, label)?;
    if observed != expected {
        return Err(format!("{label} seal does not match canonical document"));
    }
    Ok(observed.to_owned())
}

fn seal(value: &mut Value) -> Result<String, String> {
    if object(value, "output")?.contains_key("seal_sha256") {
        return Err("output is already sealed".into());
    }
    let seal = document_sha256(value, "output")?;
    value
        .as_object_mut()
        .expect("validated output object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn canonical_regular(path: &Path, label: &str, executable: bool) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    #[cfg(unix)]
    if executable && metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!("{label} must be executable"));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn file_evidence(path: &Path, label: &str, executable: bool) -> Result<FileEvidence, String> {
    let path = canonical_regular(path, label, executable)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    Ok(FileEvidence {
        bytes: u64::try_from(raw.len()).map_err(|_| format!("{label} is too large"))?,
        path,
        sha256: sha256_hex(&raw),
    })
}

fn read_bound_document(path: &Path, label: &str) -> Result<BoundDocument, String> {
    let path = canonical_regular(path, label, false)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("{label} is not JSON: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    let document_seal_sha256 = verify_seal(&value, label)?;
    let document_sha256 = document_sha256(&value, label)?;
    Ok(BoundDocument {
        path,
        bytes: u64::try_from(raw.len()).map_err(|_| format!("{label} is too large"))?,
        raw_sha256: sha256_hex(&raw),
        document_sha256,
        document_seal_sha256,
        value,
    })
}

fn bound_json(document: &BoundDocument) -> Value {
    json!({
        "path": document.path,
        "present": true,
        "bytes": document.bytes,
        "sha256": document.raw_sha256,
        "raw_sha256": document.raw_sha256,
        "document_sha256": document.document_sha256,
        "document_seal_sha256": document.document_seal_sha256,
    })
}

/// The admission-current document is deliberately the only mutable item in
/// this authority chain.  The watcher may reseal it while retaining the same
/// immutable manifest and admission receipt.  Keep every observed pointer
/// byte/seal pair as history, but never treat that historical pair as the
/// immutable authority itself.
fn versioned_current_acceptance() -> Value {
    json!({
        "canonical_pointer_path_required": true,
        "pointer_reseal_allowed_only_when_immutable_authority_is_exact": true,
        "immutable_manifest_raw_sha_and_seal_must_remain_exact": true,
        "immutable_admission_receipt_raw_sha_and_seal_must_remain_exact": true,
        "manifest_or_receipt_substitution_accepted": false,
    })
}

fn versioned_current_preflight_observation(inputs: &PreflightInputs) -> Value {
    json!({
        "canonical_pointer_path": inputs.source.admission_current.path,
        "preflight_observed": bound_json(&inputs.source.admission_current),
        "immutable_manifest": bound_json(&inputs.source.manifest),
        "immutable_admission_receipt": bound_json(&inputs.source.admission_receipt),
        "acceptance": versioned_current_acceptance(),
    })
}

fn validate_historical_pointer_evidence(
    value: &Value,
    canonical_path: &Path,
    label: &str,
) -> Result<(), String> {
    let observed = object(value, label)?;
    if string(observed, "path", label)? != canonical_path.to_string_lossy()
        || observed.get("present").and_then(Value::as_bool) != Some(true)
        || require_u64(observed, "bytes", label)? == 0
    {
        return Err(format!(
            "{label} is not a nonempty observation of the canonical admission-current path"
        ));
    }
    let sha256 = require_sha_field(observed, "sha256", label)?;
    if observed
        .get("raw_sha256")
        .and_then(Value::as_str)
        .is_some_and(|raw| raw != sha256)
    {
        return Err(format!("{label}.raw_sha256 drifted from sha256"));
    }
    let document_sha256 = require_sha_field(observed, "document_sha256", label)?;
    let document_seal_sha256 = require_sha_field(observed, "document_seal_sha256", label)?;
    if document_sha256 != document_seal_sha256 {
        return Err(format!("{label} document identity/seal drifted"));
    }
    Ok(())
}

fn validate_versioned_current_observation(
    value: &Value,
    canonical_path: &Path,
    manifest: &BoundDocument,
    admission_receipt: &BoundDocument,
    required_observations: &[&str],
    label: &str,
) -> Result<(), String> {
    let observed = object(value, label)?;
    if string(observed, "canonical_pointer_path", label)? != canonical_path.to_string_lossy() {
        return Err(format!("{label}.canonical_pointer_path drifted"));
    }
    for field in required_observations {
        validate_historical_pointer_evidence(
            observed
                .get(*field)
                .ok_or_else(|| format!("{label}.{field} is absent"))?,
            canonical_path,
            &format!("{label}.{field}"),
        )?;
    }
    if observed.get("immutable_manifest") != Some(&bound_json(manifest)) {
        return Err(format!("{label}.immutable_manifest drifted"));
    }
    if observed.get("immutable_admission_receipt") != Some(&bound_json(admission_receipt)) {
        return Err(format!("{label}.immutable_admission_receipt drifted"));
    }
    if observed.get("acceptance") != Some(&versioned_current_acceptance()) {
        return Err(format!("{label}.acceptance drifted"));
    }
    Ok(())
}

fn require_same_immutable_source(
    launch: &SourceIdentity,
    terminal: &SourceIdentity,
    label: &str,
) -> Result<(), String> {
    if bound_json(&launch.manifest) != bound_json(&terminal.manifest)
        || bound_json(&launch.admission_receipt) != bound_json(&terminal.admission_receipt)
        || launch.manifest_seal_sha256 != terminal.manifest_seal_sha256
        || launch.admission_receipt_seal_sha256 != terminal.admission_receipt_seal_sha256
        || launch.source_audit_seal_sha256 != terminal.source_audit_seal_sha256
        || launch.source_revision != terminal.source_revision
    {
        return Err(format!(
            "{label} immutable manifest or admission receipt drifted"
        ));
    }
    Ok(())
}

fn require_evidence_matches(
    object: &Map<String, Value>,
    field: &str,
    expected: &Value,
    label: &str,
) -> Result<(), String> {
    if object.get(field) != Some(expected) {
        return Err(format!("{label}.{field} byte/path identity drifted"));
    }
    Ok(())
}

fn validate_source_identity(args: &Args) -> Result<SourceIdentity, String> {
    let manifest = read_bound_document(&args.manifest, "manifest")?;
    let manifest_root = object(&manifest.value, "manifest")?;
    require_exact_string(manifest_root, "schema", MANIFEST_SCHEMA, "manifest")?;
    if manifest.raw_sha256 != MANIFEST_DOCUMENT_SHA256
        || manifest.document_seal_sha256 != MANIFEST_SEAL_SHA256
    {
        return Err("manifest raw/seal identity drifted from the admitted Qwen80 source".into());
    }

    let admission_current = read_bound_document(&args.admission_current, "admission current")?;
    let admission_root = object(&admission_current.value, "admission current")?;
    require_exact_string(
        admission_root,
        "schema",
        ADMISSION_POINTER_SCHEMA,
        "admission current",
    )?;
    require_exact_string(
        admission_root,
        "status",
        ADMISSION_POINTER_STATUS,
        "admission current",
    )?;
    let selected_manifest = object_field(admission_root, "complete_manifest", "admission current")?;
    if string(
        selected_manifest,
        "document_sha256",
        "admission current.complete_manifest",
    )? != manifest.raw_sha256
        || string(
            selected_manifest,
            "seal_sha256",
            "admission current.complete_manifest",
        )? != manifest.document_seal_sha256
    {
        return Err("admission current manifest identity drifted".into());
    }
    let selected_receipt = object_field(admission_root, "admission_receipt", "admission current")?;
    let receipt_path = PathBuf::from(string(
        selected_receipt,
        "path",
        "admission current.admission_receipt",
    )?);
    let admission_receipt = read_bound_document(&receipt_path, "immutable admission receipt")?;
    let receipt_root = object(&admission_receipt.value, "immutable admission receipt")?;
    require_exact_string(
        receipt_root,
        "schema",
        ADMISSION_RECEIPT_SCHEMA,
        "immutable admission receipt",
    )?;
    require_exact_string(
        receipt_root,
        "status",
        ADMISSION_RECEIPT_STATUS,
        "immutable admission receipt",
    )?;
    if admission_receipt.document_seal_sha256 != ADMISSION_RECEIPT_SEAL_SHA256
        || string(
            selected_receipt,
            "seal_sha256",
            "admission current.admission_receipt",
        )? != admission_receipt.document_seal_sha256
    {
        return Err("admission current immutable receipt identity drifted".into());
    }
    let revalidation = object_field(
        receipt_root,
        "current_source_revalidation",
        "immutable admission receipt",
    )?;
    let source_audit_seal_sha256 = require_sha_field(
        revalidation,
        "source_audit_seal_sha256",
        "immutable admission receipt.current_source_revalidation",
    )?;
    let source_revision = string(
        revalidation,
        "revision",
        "immutable admission receipt.current_source_revalidation",
    )?
    .to_owned();
    if source_revision != SOURCE_REVISION {
        return Err("immutable admission receipt source revision drifted".into());
    }
    Ok(SourceIdentity {
        manifest_seal_sha256: manifest.document_seal_sha256.clone(),
        admission_pointer_seal_sha256: admission_current.document_seal_sha256.clone(),
        admission_receipt_seal_sha256: admission_receipt.document_seal_sha256.clone(),
        source_audit_seal_sha256,
        source_revision,
        manifest,
        admission_current,
        admission_receipt,
    })
}

fn validate_joint_assessment(document: &BoundDocument) -> Result<(), String> {
    let root = object(&document.value, "joint assessment")?;
    require_exact_string(root, "schema", JOINT_ASSESSMENT_SCHEMA, "joint assessment")?;
    require_exact_string(root, "status", JOINT_ASSESSMENT_STATUS, "joint assessment")?;
    require_bool(root, "earned_component_only", true, "joint assessment")?;
    let scope = object_field(root, "component_scope", "joint assessment")?;
    for (field, expected) in [
        ("source_token_id", u64::from(SOURCE_TOKEN_ID)),
        ("fresh_l0_dispatches", L0_DISPATCHES as u64),
        (
            "fresh_l1_slot1_prefix_dispatches",
            L1_PREFIX_DISPATCHES as u64,
        ),
        (
            "fresh_total_dispatches",
            (L0_DISPATCHES + L1_PREFIX_DISPATCHES) as u64,
        ),
    ] {
        if require_u64(scope, field, "joint assessment.component_scope")? != expected {
            return Err(format!("joint assessment.component_scope.{field} drifted"));
        }
    }
    for field in [
        "opaque_same_runtime_continuation_required",
        "single_fence_required",
    ] {
        require_bool(scope, field, true, "joint assessment.component_scope")?;
    }
    require_bool(
        scope,
        "full_layer_or_token_decoder_earned",
        false,
        "joint assessment.component_scope",
    )?;
    let boundary = object_field(root, "claim_boundary", "joint assessment")?;
    for field in [
        "cpu_only_post_capture_assessment",
        "l0_l1_component_not_full_layer_token_decoder",
        "does_not_reuse_historical_l0_receipt_as_execution_input",
        "does_not_accept_raw_pinned_buffer_or_dispatch_count_input",
        "does_not_construct_metal_or_dispatch",
        "does_not_issue_or_release_lease",
        "does_not_start_runtime_server_or_watcher",
        "does_not_measure_tps_or_tg",
        "does_not_claim_decoder_token_or_tournament",
    ] {
        require_bool(boundary, field, true, "joint assessment.claim_boundary")?;
    }
    Ok(())
}

fn validate_completion_preflight(
    document: &BoundDocument,
    assessment: &BoundDocument,
) -> Result<(), String> {
    let root = object(&document.value, "L1 completion preflight")?;
    require_exact_string(
        root,
        "schema",
        COMPLETION_PREFLIGHT_SCHEMA,
        "L1 completion preflight",
    )?;
    require_exact_string(
        root,
        "status",
        COMPLETION_PREFLIGHT_STATUS,
        "L1 completion preflight",
    )?;
    require_bool(
        root,
        "preflight_ready_for_future_outer_authority_only",
        false,
        "L1 completion preflight",
    )?;
    let antecedent = object_field(
        root,
        "antecedent_l0_l1_component",
        "L1 completion preflight",
    )?;
    let expected = bound_json(assessment);
    for field in ["document_sha256", "document_seal_sha256"] {
        if antecedent.get(field) != expected.get(field) {
            return Err(
                "L1 completion preflight does not bind the supplied joint assessment".into(),
            );
        }
    }
    let route_authority = object_field(
        root,
        "l1_source_token_route_authority",
        "L1 completion preflight",
    )?;
    require_bool(
        route_authority,
        "present_and_valid",
        false,
        "L1 completion preflight.l1_source_token_route_authority",
    )?;
    let graph = object_field(
        root,
        "future_joint_command_graph",
        "L1 completion preflight",
    )?;
    for (field, expected) in [
        ("source_token_id", u64::from(SOURCE_TOKEN_ID)),
        ("l0_layer", L0_LAYER as u64),
        ("l1_layer", L1_LAYER as u64),
        ("l1_linear_state_slot", L1_LINEAR_STATE_SLOT as u64),
        ("l0_reencode_dispatches", L0_DISPATCHES as u64),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES as u64),
        ("l1_moe_suffix_dispatches", 14),
        ("total_dispatches", 46),
    ] {
        if require_u64(
            graph,
            field,
            "L1 completion preflight.future_joint_command_graph",
        )? != expected
        {
            return Err(format!(
                "L1 completion preflight.future_joint_command_graph.{field} drifted"
            ));
        }
    }
    for field in [
        "single_runtime_required",
        "single_token_command_buffer_required",
        "single_fence_after_all_dispatches_required",
        "non_timed_trace_required",
    ] {
        require_bool(
            graph,
            field,
            true,
            "L1 completion preflight.future_joint_command_graph",
        )?;
    }
    let boundary = object_field(root, "claim_boundary", "L1 completion preflight")?;
    for field in [
        "cpu_file_only_preflight",
        "future_l1_moe_component_is_not_a_complete_token_or_decoder",
    ] {
        require_bool(
            boundary,
            field,
            true,
            "L1 completion preflight.claim_boundary",
        )?;
    }
    for field in [
        "artifact_scan_or_payload_open_performed",
        "metal_context_or_dispatch_performed",
        "lease_issued_or_consumed",
        "watcher_server_hcli_or_runtime_changed",
        "tps_tg_or_tournament_claim_earned",
    ] {
        require_bool(
            boundary,
            field,
            false,
            "L1 completion preflight.claim_boundary",
        )?;
    }
    Ok(())
}

fn validate_self_binary(binary: &FileEvidence) -> Result<(), String> {
    let current = env::current_exe()
        .map_err(|error| format!("cannot identify current producer executable: {error}"))?;
    let current = canonical_regular(&current, "current producer executable", true)?;
    if current != binary.path {
        return Err(
            "producer binary path does not identify the executable currently running this child"
                .into(),
        );
    }
    let current_evidence = file_evidence(&current, "current producer executable", true)?;
    if current_evidence != *binary {
        return Err("producer binary bytes changed after outer binding".into());
    }
    Ok(())
}

fn load_preflight_inputs(args: &Args, require_self: bool) -> Result<PreflightInputs, String> {
    let source = validate_source_identity(args)?;
    let joint_assessment = read_bound_document(&args.joint_assessment, "joint assessment")?;
    validate_joint_assessment(&joint_assessment)?;
    let completion_preflight =
        read_bound_document(&args.completion_preflight, "L1 completion preflight")?;
    validate_completion_preflight(&completion_preflight, &joint_assessment)?;
    let producer_binary = file_evidence(&args.producer_binary, "producer binary", true)?;
    if require_self {
        validate_self_binary(&producer_binary)?;
    }
    Ok(PreflightInputs {
        source,
        joint_assessment,
        completion_preflight,
        producer_binary,
    })
}

fn fixed_payload_requirements() -> Vec<Value> {
    [
        (
            "post_attention_layernorm",
            "model.layers.1.post_attention_layernorm.weight",
            vec![HIDDEN],
        ),
        (
            "router",
            "model.layers.1.mlp.gate.weight",
            vec![EXPERTS, HIDDEN],
        ),
        (
            "shared_gate_proj",
            "model.layers.1.mlp.shared_expert.gate_proj.weight",
            vec![512, HIDDEN],
        ),
        (
            "shared_up_proj",
            "model.layers.1.mlp.shared_expert.up_proj.weight",
            vec![512, HIDDEN],
        ),
        (
            "shared_down_proj",
            "model.layers.1.mlp.shared_expert.down_proj.weight",
            vec![HIDDEN, 512],
        ),
        (
            "shared_expert_gate",
            "model.layers.1.mlp.shared_expert_gate.weight",
            vec![1, HIDDEN],
        ),
    ]
    .into_iter()
    .map(|(role, tensor_name, shape)| {
        json!({
            "role": role,
            "tensor_name": tensor_name,
            "shape": shape,
            "group_size": GROUP_SIZE,
            "required_layout": {
                "magic": "HQ30G1B1",
                "version": 1,
                "group_size": GROUP_SIZE,
                "scale_dtype": "float16",
                "sign_bit_order": "little",
            },
            "required_descriptor_fields": [
                "artifact_sha256",
                "direct_packed_payload_sha256",
                "header_sha256",
                "payload_bytes",
                "layout",
            ],
        })
    })
    .collect()
}

fn route_descriptor_requirement(role: &str, expert: u16) -> Result<Value, String> {
    let (suffix, shape) = match role {
        "gate" => ("gate_proj.weight", vec![512, HIDDEN]),
        "up" => ("up_proj.weight", vec![512, HIDDEN]),
        "down" => ("down_proj.weight", vec![HIDDEN, 512]),
        _ => return Err(format!("unsupported routed projection role {role:?}")),
    };
    Ok(json!({
        "tensor_name": format!("model.layers.1.mlp.experts.{expert}.{suffix}"),
        "shape": shape,
        "group_size": GROUP_SIZE,
        "required_layout": {
            "magic": "HQ30G1B1",
            "version": 1,
            "group_size": GROUP_SIZE,
            "scale_dtype": "float16",
            "sign_bit_order": "little",
        },
    }))
}

fn producer_preflight_document(inputs: &PreflightInputs) -> Value {
    json!({
        "schema": PRODUCER_PREFLIGHT_SCHEMA,
        "status": PRODUCER_PREFLIGHT_STATUS,
        "source_binding": {
            "manifest": bound_json(&inputs.source.manifest),
            "admission_current": bound_json(&inputs.source.admission_current),
            "admission_receipt": bound_json(&inputs.source.admission_receipt),
            "joint_assessment": bound_json(&inputs.joint_assessment),
            "completion_preflight": bound_json(&inputs.completion_preflight),
            "manifest_seal_sha256": inputs.source.manifest_seal_sha256,
            "admission_current_pointer_seal_sha256": inputs.source.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": inputs.source.admission_receipt_seal_sha256,
            "source_audit_seal_sha256": inputs.source.source_audit_seal_sha256,
            "source_revision": inputs.source.source_revision,
        },
        "producer_binary": inputs.producer_binary.json(),
        "versioned_current_admission": versioned_current_preflight_observation(inputs),
        "dynamic_authority_contract": {
            "schema": AUTHORITY_SCHEMA,
            "status": AUTHORITY_STATUS,
            "outer_launch_authority_binding_required": true,
            "one_current_admitted_cpu_catalog_scan_required": true,
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_reencode_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "l1_moe_suffix_dispatches": 14,
            "l1_layer": L1_LAYER,
            "l1_linear_state_slot": L1_LINEAR_STATE_SLOT,
            "exact_fixed_payload_requirements": fixed_payload_requirements(),
            "exact_route_payloads_required": TOP_K * 3,
            "all_ten_dynamic_router_ids_and_weights_required": true,
            "no_fixture_or_cross_process_buffer_substitution": true,
            "planned_output_must_be_new_under_outer_capture_dir": true,
        },
        "claim_boundary": {
            "preflight_only": true,
            "strict_catalog_admission_scan_performed": false,
            "admitted_payload_snapshot_opened": false,
            "child_started": false,
            "metal_or_gpu_activity_performed": false,
            "lease_issued_or_consumed": false,
            "watcher_or_server_changed": false,
            "model_token_or_tps_claim_earned": false,
            "complete_layer_or_decoder_claim_earned": false,
        },
    })
}

fn validate_producer_preflight(
    document: &BoundDocument,
    inputs: &PreflightInputs,
) -> Result<(), String> {
    let root = object(&document.value, "L1 route authority producer preflight")?;
    require_exact_string(
        root,
        "schema",
        PRODUCER_PREFLIGHT_SCHEMA,
        "L1 route authority producer preflight",
    )?;
    require_exact_string(
        root,
        "status",
        PRODUCER_PREFLIGHT_STATUS,
        "L1 route authority producer preflight",
    )?;
    let source = object_field(
        root,
        "source_binding",
        "L1 route authority producer preflight",
    )?;
    for (field, expected) in [
        ("manifest", bound_json(&inputs.source.manifest)),
        (
            "admission_receipt",
            bound_json(&inputs.source.admission_receipt),
        ),
        ("joint_assessment", bound_json(&inputs.joint_assessment)),
        (
            "completion_preflight",
            bound_json(&inputs.completion_preflight),
        ),
    ] {
        require_evidence_matches(
            source,
            field,
            &expected,
            "L1 route authority producer preflight.source_binding",
        )?;
    }
    validate_historical_pointer_evidence(
        source.get("admission_current").ok_or(
            "L1 route authority producer preflight.source_binding.admission_current is absent",
        )?,
        &inputs.source.admission_current.path,
        "L1 route authority producer preflight.source_binding.admission_current",
    )?;
    require_sha_field(
        source,
        "admission_current_pointer_seal_sha256",
        "L1 route authority producer preflight.source_binding",
    )?;
    let versioned = root
        .get("versioned_current_admission")
        .ok_or("L1 route authority producer preflight.versioned_current_admission is absent")?;
    validate_versioned_current_observation(
        versioned,
        &inputs.source.admission_current.path,
        &inputs.source.manifest,
        &inputs.source.admission_receipt,
        &["preflight_observed"],
        "L1 route authority producer preflight.versioned_current_admission",
    )?;
    let versioned_root = object(
        versioned,
        "L1 route authority producer preflight.versioned_current_admission",
    )?;
    if versioned_root.get("preflight_observed") != source.get("admission_current")
        || versioned_root
            .get("preflight_observed")
            .and_then(Value::as_object)
            .and_then(|pointer| pointer.get("document_seal_sha256"))
            != source.get("admission_current_pointer_seal_sha256")
    {
        return Err(
            "L1 route authority producer preflight historical pointer evidence is internally inconsistent"
                .into(),
        );
    }
    for (field, expected) in [
        (
            "manifest_seal_sha256",
            inputs.source.manifest_seal_sha256.as_str(),
        ),
        (
            "admission_receipt_seal_sha256",
            inputs.source.admission_receipt_seal_sha256.as_str(),
        ),
        (
            "source_audit_seal_sha256",
            inputs.source.source_audit_seal_sha256.as_str(),
        ),
        ("source_revision", inputs.source.source_revision.as_str()),
    ] {
        require_exact_string(
            source,
            field,
            expected,
            "L1 route authority producer preflight.source_binding",
        )?;
    }
    require_evidence_matches(
        root,
        "producer_binary",
        &inputs.producer_binary.json(),
        "L1 route authority producer preflight",
    )?;
    let dynamic = object_field(
        root,
        "dynamic_authority_contract",
        "L1 route authority producer preflight",
    )?;
    require_exact_string(
        dynamic,
        "schema",
        AUTHORITY_SCHEMA,
        "L1 route authority producer preflight.dynamic_authority_contract",
    )?;
    require_exact_string(
        dynamic,
        "status",
        AUTHORITY_STATUS,
        "L1 route authority producer preflight.dynamic_authority_contract",
    )?;
    for field in [
        "outer_launch_authority_binding_required",
        "one_current_admitted_cpu_catalog_scan_required",
        "all_ten_dynamic_router_ids_and_weights_required",
        "no_fixture_or_cross_process_buffer_substitution",
        "planned_output_must_be_new_under_outer_capture_dir",
    ] {
        require_bool(
            dynamic,
            field,
            true,
            "L1 route authority producer preflight.dynamic_authority_contract",
        )?;
    }
    for (field, expected) in [
        ("source_token_id", u64::from(SOURCE_TOKEN_ID)),
        ("l0_reencode_dispatches", L0_DISPATCHES as u64),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES as u64),
        ("l1_moe_suffix_dispatches", 14),
        ("l1_layer", L1_LAYER as u64),
        ("l1_linear_state_slot", L1_LINEAR_STATE_SLOT as u64),
        ("exact_route_payloads_required", (TOP_K * 3) as u64),
    ] {
        if require_u64(
            dynamic,
            field,
            "L1 route authority producer preflight.dynamic_authority_contract",
        )? != expected
        {
            return Err(format!(
                "L1 route authority producer preflight.dynamic_authority_contract.{field} drifted"
            ));
        }
    }
    if dynamic.get("exact_fixed_payload_requirements")
        != Some(&Value::Array(fixed_payload_requirements()))
    {
        return Err(
            "L1 route authority producer preflight fixed Layer-1 payload requirements drifted"
                .into(),
        );
    }
    let boundary = object_field(
        root,
        "claim_boundary",
        "L1 route authority producer preflight",
    )?;
    require_bool(
        boundary,
        "preflight_only",
        true,
        "L1 route authority producer preflight.claim_boundary",
    )?;
    for field in [
        "strict_catalog_admission_scan_performed",
        "admitted_payload_snapshot_opened",
        "child_started",
        "metal_or_gpu_activity_performed",
        "lease_issued_or_consumed",
        "watcher_or_server_changed",
        "model_token_or_tps_claim_earned",
        "complete_layer_or_decoder_claim_earned",
    ] {
        require_bool(
            boundary,
            field,
            false,
            "L1 route authority producer preflight.claim_boundary",
        )?;
    }
    Ok(())
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be an existing non-symlink directory"));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn validate_outer_launch_authority(
    document: &BoundDocument,
    inputs: &PreflightInputs,
    producer_preflight: &BoundDocument,
    args: &Args,
) -> Result<(), String> {
    let root = object(&document.value, "L1 route authority outer launch authority")?;
    require_exact_string(
        root,
        "schema",
        OUTER_AUTHORITY_SCHEMA,
        "L1 route authority outer launch authority",
    )?;
    require_exact_string(
        root,
        "status",
        OUTER_AUTHORITY_STATUS,
        "L1 route authority outer launch authority",
    )?;
    require_evidence_matches(
        root,
        "producer_preflight",
        &bound_json(producer_preflight),
        "L1 route authority outer launch authority",
    )?;
    let source = object_field(
        root,
        "source_binding",
        "L1 route authority outer launch authority",
    )?;
    for (field, expected) in [
        ("manifest", bound_json(&inputs.source.manifest)),
        (
            "admission_receipt",
            bound_json(&inputs.source.admission_receipt),
        ),
        ("joint_assessment", bound_json(&inputs.joint_assessment)),
        (
            "completion_preflight",
            bound_json(&inputs.completion_preflight),
        ),
    ] {
        require_evidence_matches(
            source,
            field,
            &expected,
            "L1 route authority outer launch authority.source_binding",
        )?;
    }
    validate_historical_pointer_evidence(
        source.get("admission_current").ok_or(
            "L1 route authority outer launch authority.source_binding.admission_current is absent",
        )?,
        &inputs.source.admission_current.path,
        "L1 route authority outer launch authority.source_binding.admission_current",
    )?;
    for (field, expected) in [
        (
            "manifest_seal_sha256",
            inputs.source.manifest_seal_sha256.as_str(),
        ),
        (
            "admission_receipt_seal_sha256",
            inputs.source.admission_receipt_seal_sha256.as_str(),
        ),
    ] {
        require_exact_string(
            source,
            field,
            expected,
            "L1 route authority outer launch authority.source_binding",
        )?;
    }
    let versioned = root
        .get("versioned_current_admission")
        .ok_or("L1 route authority outer launch authority.versioned_current_admission is absent")?;
    validate_versioned_current_observation(
        versioned,
        &inputs.source.admission_current.path,
        &inputs.source.manifest,
        &inputs.source.admission_receipt,
        &["preflight_observed", "launch_observed"],
        "L1 route authority outer launch authority.versioned_current_admission",
    )?;
    let versioned_root = object(
        versioned,
        "L1 route authority outer launch authority.versioned_current_admission",
    )?;
    if versioned_root.get("launch_observed") != source.get("admission_current") {
        return Err(
            "L1 route authority outer launch authority source pointer does not match its launch observation"
                .into(),
        );
    }
    let producer_versioned = object_field(
        object(
            &producer_preflight.value,
            "L1 route authority outer launch authority producer preflight",
        )?,
        "versioned_current_admission",
        "L1 route authority outer launch authority producer preflight",
    )?;
    if versioned_root.get("preflight_observed") != producer_versioned.get("preflight_observed") {
        return Err(
            "L1 route authority outer launch authority did not preserve producer preflight pointer history"
                .into(),
        );
    }
    require_evidence_matches(
        root,
        "producer_binary",
        &inputs.producer_binary.json(),
        "L1 route authority outer launch authority",
    )?;
    let capture_dir = canonical_directory(
        args.capture_dir
            .as_deref()
            .ok_or("cpu-oracle mode has no capture directory")?,
        "cpu-oracle capture directory",
    )?;
    if string(
        root,
        "planned_capture_dir",
        "L1 route authority outer launch authority",
    )? != capture_dir.to_string_lossy()
    {
        return Err("outer launch authority capture directory drifted".into());
    }
    if string(
        root,
        "planned_output_authority",
        "L1 route authority outer launch authority",
    )? != args.out.to_string_lossy()
    {
        return Err("outer launch authority output path drifted".into());
    }
    if require_u64(root, "workers", "L1 route authority outer launch authority")?
        != args.workers.ok_or("cpu-oracle mode has no workers")? as u64
    {
        return Err("outer launch authority worker count drifted".into());
    }
    let policy = object_field(
        root,
        "execution_policy",
        "L1 route authority outer launch authority",
    )?;
    if require_u64(
        policy,
        "exact_catalog_admission_scans",
        "L1 route authority outer launch authority.execution_policy",
    )? != 1
    {
        return Err("outer launch authority must permit exactly one catalog admission scan".into());
    }
    for field in [
        "cpu_oracle_only",
        "outer_reaped_required",
        "terminal_receipt_written_last_required",
    ] {
        require_bool(
            policy,
            field,
            true,
            "L1 route authority outer launch authority.execution_policy",
        )?;
    }
    for field in [
        "metal_or_gpu_allowed",
        "lease_allowed",
        "watcher_or_server_allowed",
        "automatic_retry_allowed",
    ] {
        require_bool(
            policy,
            field,
            false,
            "L1 route authority outer launch authority.execution_policy",
        )?;
    }
    let replay = object_field(
        root,
        "replay_guard",
        "L1 route authority outer launch authority",
    )?;
    for field in ["capture_dir_unique", "one_child_maximum"] {
        require_bool(
            replay,
            field,
            true,
            "L1 route authority outer launch authority.replay_guard",
        )?;
    }
    Ok(())
}

fn checked_f32le_sha256(values: &[f32], label: &str) -> Result<String, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} must contain finite f32 values"));
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn header_json(header: &CompleteBinaryHeader) -> Value {
    json!({
        "magic": "HQ30G1B1",
        "version": header.version,
        "group_size": header.group_size,
        "shape": header.shape,
        "elements": header.elements,
        "groups": header.groups,
        "scale_offset": header.scale_offset,
        "sign_offset": header.sign_offset,
        "payload_bytes": header.payload_bytes,
        "scale_dtype": "float16",
        "sign_bit_order": "little",
    })
}

fn descriptor_from_binding(
    catalog: &Qwen80CompleteArtifactCatalog,
    binding: &Qwen80PackedTensorBinding,
    role: &str,
) -> Result<Value, String> {
    let header = catalog
        .direct_tensor_header(&binding.name)
        .map_err(|error| format!("catalog header unavailable for {}: {error}", binding.name))?;
    if header.shape != binding.shape || header.group_size != binding.group_size {
        return Err(format!(
            "catalog header geometry drifted for exact tensor {}",
            binding.name
        ));
    }
    if header.group_size != GROUP_SIZE {
        return Err(format!(
            "catalog tensor {} has non-source group size",
            binding.name
        ));
    }
    let payload = catalog
        .verified_direct_tensor_payload(&binding.name)
        .map_err(|error| format!("catalog payload unavailable for {}: {error}", binding.name))?;
    if payload.len() != header.payload_bytes || header.scale_offset > payload.len() {
        return Err(format!(
            "catalog payload/header length drifted for exact tensor {}",
            binding.name
        ));
    }
    let artifact_sha256 = catalog
        .direct_tensor_artifact_sha256(&binding.name)
        .map_err(|error| {
            format!(
                "catalog artifact SHA unavailable for {}: {error}",
                binding.name
            )
        })?;
    require_sha256(artifact_sha256, "catalog artifact SHA")?;
    Ok(json!({
        "role": role,
        "tensor_name": binding.name,
        "shape": binding.shape,
        "group_size": binding.group_size,
        "artifact_sha256": artifact_sha256,
        "direct_packed_payload_sha256": sha256_hex(payload.as_ref()),
        "header_sha256": sha256_hex(&payload[..header.scale_offset]),
        "header_bytes": header.scale_offset,
        "payload_bytes": payload.len(),
        "layout": header_json(header),
        "payload_from_admitted_catalog_snapshot": true,
        "raw_artifact_reopened_by_this_child": false,
    }))
}

fn route_binding(role: &str, expert: u16) -> Result<Qwen80PackedTensorBinding, String> {
    let expected = route_descriptor_requirement(role, expert)?;
    let expected = object(&expected, "route descriptor requirement")?;
    let name = string(expected, "tensor_name", "route descriptor requirement")?.to_owned();
    let shape = array_field(expected, "shape", "route descriptor requirement")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or("route descriptor requirement shape is invalid")
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Qwen80PackedTensorBinding {
        name,
        shape,
        group_size: GROUP_SIZE,
    })
}

#[derive(Clone, Debug)]
struct CpuAuthorityMaterial {
    source_input_f32le_sha256: String,
    l0_second_residual_cpu_f32le_sha256: String,
    l0_post_conv_state_cpu_f32le_sha256: String,
    l0_post_recurrent_state_cpu_f32le_sha256: String,
    l1_prefix_input_cpu_f32le_sha256: String,
    l1_first_residual_cpu_f32le_sha256: String,
    l1_post_attention_normalized_hidden_cpu_f32le_sha256: String,
    l1_router_logits_cpu_f32le_sha256: String,
    l1_post_conv_state_cpu_f32le_sha256: String,
    l1_post_recurrent_state_cpu_f32le_sha256: String,
    l1_routed_sum_cpu_f32le_sha256: String,
    l1_shared_gated_cpu_f32le_sha256: String,
    l1_moe_output_cpu_f32le_sha256: String,
    l1_second_residual_cpu_f32le_sha256: String,
    l1_shared_gate_value: f32,
    route_tie_epsilon_f32: f32,
    route_ids: Vec<u16>,
    route_weights: Vec<f32>,
    fixed_l1_payloads: Vec<Value>,
    deterministic_waves: Vec<Value>,
    catalog_tensor_count: usize,
}

fn require_unique_descriptor_identities(
    fixed_l1_payloads: &[Value],
    deterministic_waves: &[Value],
) -> Result<(), String> {
    let mut artifacts = BTreeSet::new();
    let mut payloads = BTreeSet::new();
    let mut check = |descriptor: &Value, label: &str| -> Result<(), String> {
        let descriptor = object(descriptor, label)?;
        for (field, seen) in [
            ("artifact_sha256", &mut artifacts),
            ("direct_packed_payload_sha256", &mut payloads),
        ] {
            let digest = require_sha_field(descriptor, field, label)?;
            if !seen.insert(digest) {
                return Err(format!("{label} reuses {field}"));
            }
        }
        require_sha_field(descriptor, "header_sha256", label)?;
        Ok(())
    };
    for (index, descriptor) in fixed_l1_payloads.iter().enumerate() {
        check(descriptor, &format!("fixed Layer-1 descriptor {index}"))?;
    }
    for (wave_index, wave) in deterministic_waves.iter().enumerate() {
        let wave = object(wave, "deterministic wave")?;
        for role in ["gate", "up", "down"] {
            check(
                wave.get(role)
                    .ok_or_else(|| format!("deterministic wave {wave_index} lacks {role}"))?,
                &format!("deterministic wave {wave_index} {role}"),
            )?;
        }
    }
    if artifacts.len() != 36 || payloads.len() != 36 {
        return Err(
            "Layer-1 CPU authority must bind 36 distinct fixed/route artifact and payload identities"
                .into(),
        );
    }
    Ok(())
}

fn material_from_current_catalog(
    catalog: &Qwen80CompleteArtifactCatalog,
    source: &SourceIdentity,
) -> Result<CpuAuthorityMaterial, String> {
    let route_tie_epsilon_f32 = route_tie_epsilon();
    if catalog.manifest_seal() != source.manifest_seal_sha256
        || catalog.config.source_revision != source.source_revision
    {
        return Err("fresh admitted catalog drifted from the sealed source identity".into());
    }
    let embedding = catalog
        .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID)
        .map_err(|error| format!("source-token direct-packed embedding oracle refused: {error}"))?;
    let l0_input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(embedding.hidden);
    let l0 = catalog
        .execute_first_linear_layer_cpu_moe_oracle(&l0_input)
        .map_err(|error| format!("fresh source-token L0 full-MoE CPU oracle refused: {error}"))?;
    if l0.mixer.layer != L0_LAYER || l0.mixer.linear_state_slot != L0_LAYER {
        return Err("fresh L0 CPU oracle did not retain Layer-0 state-slot identity".into());
    }
    let l1_input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(l0.layer_output.clone());
    if l1_input.hidden != l0.layer_output {
        return Err("fresh L1 CPU input was not retained from fresh L0 second residual".into());
    }
    let l1_contract = catalog
        .canonical_linear_moe_operator_contract(L1_LAYER)
        .map_err(|error| {
            format!("Layer-1 canonical direct-packed MoE contract refused: {error}")
        })?;
    if l1_contract.mixer.layer != L1_LAYER
        || l1_contract.mixer.linear_state_slot != L1_LINEAR_STATE_SLOT
    {
        return Err("Layer-1 canonical MoE contract drifted from source slot one".into());
    }
    let l1 = catalog
        .execute_canonical_linear_moe_cpu_oracle(&l1_contract, &l1_input)
        .map_err(|error| format!("fresh source-token L1 full-MoE CPU oracle refused: {error}"))?;
    if l1.mixer.layer != L1_LAYER || l1.mixer.linear_state_slot != L1_LINEAR_STATE_SLOT {
        return Err("fresh L1 CPU oracle did not retain Layer-1 state-slot identity".into());
    }
    if route_tie_epsilon().to_bits() != route_tie_epsilon_f32.to_bits() {
        return Err("source router tie policy drifted during the CPU authority scan".into());
    }
    if l1.route.ids.len() != TOP_K || l1.route.weights.len() != TOP_K {
        return Err("fresh Layer-1 source router did not select exactly ten routes".into());
    }
    let mut route_ids = Vec::with_capacity(TOP_K);
    let mut route_weights = Vec::with_capacity(TOP_K);
    let mut unique_routes = BTreeSet::new();
    for (&id, &weight) in l1.route.ids.iter().zip(&l1.route.weights) {
        if usize::from(id) >= EXPERTS
            || !unique_routes.insert(id)
            || !weight.is_finite()
            || weight < 0.0
        {
            return Err("fresh Layer-1 source router route identity/weight is invalid".into());
        }
        route_ids.push(id);
        route_weights.push(weight);
    }
    let route_weight_sum = route_weights.iter().sum::<f32>();
    if (route_weight_sum - 1.0).abs() > MAX_ROUTE_WEIGHT_SUM_ERROR {
        return Err(format!(
            "fresh Layer-1 source route weights sum to {route_weight_sum}, not one"
        ));
    }

    let fixed_l1_payloads = [
        (
            "post_attention_layernorm",
            &l1_contract.post_attention_layernorm,
        ),
        ("router", &l1_contract.router),
        ("shared_gate_proj", &l1_contract.shared_gate_proj),
        ("shared_up_proj", &l1_contract.shared_up_proj),
        ("shared_down_proj", &l1_contract.shared_down_proj),
        ("shared_expert_gate", &l1_contract.shared_expert_gate),
    ]
    .into_iter()
    .map(|(role, binding)| descriptor_from_binding(catalog, binding, role))
    .collect::<Result<Vec<_>, _>>()?;
    if fixed_l1_payloads.len() != 6 {
        return Err("fresh Layer-1 CPU oracle did not retain six fixed MoE bindings".into());
    }

    if l1.routed_experts.len() != TOP_K {
        return Err("fresh Layer-1 CPU oracle did not retain ten routed witnesses".into());
    }
    let mut deterministic_waves = Vec::with_capacity(TOP_K);
    for (wave_index, ((&expert, &weight), witness)) in l1
        .route
        .ids
        .iter()
        .zip(&l1.route.weights)
        .zip(&l1.routed_experts)
        .enumerate()
    {
        if witness.expert != usize::from(expert)
            || witness.route_weight.to_bits() != weight.to_bits()
        {
            return Err(format!(
                "fresh Layer-1 CPU routed witness {wave_index} drifted from source router order"
            ));
        }
        let gate_binding = route_binding("gate", expert)?;
        let up_binding = route_binding("up", expert)?;
        let down_binding = route_binding("down", expert)?;
        if witness.direct_packed_gate_proj_tensor != gate_binding.name
            || witness.direct_packed_up_proj_tensor != up_binding.name
            || witness.direct_packed_down_proj_tensor != down_binding.name
        {
            return Err(format!(
                "fresh Layer-1 routed witness {wave_index} tensor name drifted from its exact route"
            ));
        }
        deterministic_waves.push(json!({
            "wave_index": wave_index,
            "layer": L1_LAYER,
            "expert_id": expert,
            "normalized_weight": f64::from(weight),
            "normalized_weight_bits_hex": format!("0x{:016x}", f64::from(weight).to_bits()),
            "gate": descriptor_from_binding(catalog, &gate_binding, "gate")?,
            "up": descriptor_from_binding(catalog, &up_binding, "up")?,
            "down": descriptor_from_binding(catalog, &down_binding, "down")?,
            "cpu_weighted_output_f32le_sha256": checked_f32le_sha256(
                &witness.weighted_output,
                "fresh Layer-1 routed weighted output",
            )?,
            "route_execution_status": "CPU_ORACLE_PARITY_ONLY_NOT_NATIVE_COMPONENT_EXECUTION",
        }));
    }
    require_unique_descriptor_identities(&fixed_l1_payloads, &deterministic_waves)?;

    Ok(CpuAuthorityMaterial {
        source_input_f32le_sha256: checked_f32le_sha256(
            &l0_input.hidden,
            "fresh source-token embedding hidden",
        )?,
        l0_second_residual_cpu_f32le_sha256: checked_f32le_sha256(
            &l0.layer_output,
            "fresh L0 second residual",
        )?,
        l0_post_conv_state_cpu_f32le_sha256: checked_f32le_sha256(
            &l0.mixer.next_state.conv_state,
            "fresh L0 post-convolution state",
        )?,
        l0_post_recurrent_state_cpu_f32le_sha256: checked_f32le_sha256(
            &l0.mixer.next_state.recurrent_state,
            "fresh L0 post-recurrent state",
        )?,
        l1_prefix_input_cpu_f32le_sha256: checked_f32le_sha256(
            &l1_input.hidden,
            "fresh L1 prefix input",
        )?,
        l1_first_residual_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.mixer.mixer_residual_output,
            "fresh L1 first residual",
        )?,
        l1_post_attention_normalized_hidden_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.post_attention_rms_norm_output,
            "fresh L1 post-attention normalized hidden",
        )?,
        l1_router_logits_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.router_logits,
            "fresh L1 router logits",
        )?,
        l1_post_conv_state_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.mixer.next_state.conv_state,
            "fresh L1 post-convolution state",
        )?,
        l1_post_recurrent_state_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.mixer.next_state.recurrent_state,
            "fresh L1 post-recurrent state",
        )?,
        l1_routed_sum_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.routed_expert_sum,
            "fresh L1 routed sum",
        )?,
        l1_shared_gated_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.shared_gated_output,
            "fresh L1 shared gated output",
        )?,
        l1_moe_output_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.moe_output,
            "fresh L1 MoE output",
        )?,
        l1_second_residual_cpu_f32le_sha256: checked_f32le_sha256(
            &l1.layer_output,
            "fresh L1 second residual",
        )?,
        l1_shared_gate_value: l1.shared_expert_gate_value,
        route_tie_epsilon_f32,
        route_ids,
        route_weights,
        fixed_l1_payloads,
        deterministic_waves,
        catalog_tensor_count: catalog.tensor_count(),
    })
}

fn build_dynamic_authority(
    inputs: &PreflightInputs,
    producer_preflight: &BoundDocument,
    outer_launch_authority: &BoundDocument,
    args: &Args,
    material: &CpuAuthorityMaterial,
    terminal_source: &SourceIdentity,
) -> Result<Value, String> {
    require_same_immutable_source(
        &inputs.source,
        terminal_source,
        "Layer-1 CPU authority terminal source",
    )?;
    let producer_root = object(
        &producer_preflight.value,
        "Layer-1 CPU authority producer preflight",
    )?;
    let producer_versioned = object_field(
        producer_root,
        "versioned_current_admission",
        "Layer-1 CPU authority producer preflight",
    )?;
    let outer_root = object(
        &outer_launch_authority.value,
        "Layer-1 CPU authority outer launch authority",
    )?;
    let outer_versioned = object_field(
        outer_root,
        "versioned_current_admission",
        "Layer-1 CPU authority outer launch authority",
    )?;
    let preflight_observed = producer_versioned
        .get("preflight_observed")
        .cloned()
        .ok_or("Layer-1 CPU authority producer preflight has no pointer observation")?;
    let launch_observed = outer_versioned
        .get("launch_observed")
        .cloned()
        .ok_or("Layer-1 CPU authority outer launch authority has no launch pointer observation")?;
    if material.route_ids.len() != TOP_K
        || material.route_weights.len() != TOP_K
        || material.fixed_l1_payloads.len() != 6
        || material.deterministic_waves.len() != TOP_K
    {
        return Err("CPU authority material has incomplete fixed or routed evidence".into());
    }
    if material
        .route_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || material
            .route_ids
            .iter()
            .any(|id| usize::from(*id) >= EXPERTS)
        || material.route_ids.iter().collect::<BTreeSet<_>>().len() != TOP_K
    {
        return Err("CPU authority material has invalid dynamic source routes".into());
    }
    require_unique_descriptor_identities(
        &material.fixed_l1_payloads,
        &material.deterministic_waves,
    )?;
    let route_weight_sum = material.route_weights.iter().sum::<f32>();
    if (route_weight_sum - 1.0).abs() > MAX_ROUTE_WEIGHT_SUM_ERROR {
        return Err("CPU authority material route weights are not normalized".into());
    }
    let capture_dir = canonical_directory(
        args.capture_dir
            .as_deref()
            .ok_or("CPU authority output has no capture directory")?,
        "CPU authority capture directory",
    )?;
    let workers = args
        .workers
        .ok_or("CPU authority output has no worker count")?;
    let route_ids = material
        .route_ids
        .iter()
        .map(|id| u64::from(*id))
        .collect::<Vec<_>>();
    let route_weights = material
        .route_weights
        .iter()
        .map(|weight| f64::from(*weight))
        .collect::<Vec<_>>();
    let mut output = json!({
        "schema": AUTHORITY_SCHEMA,
        "status": AUTHORITY_STATUS,
        "fixture_or_synthetic": false,
        "metal_or_gpu_activity_performed": false,
        "versioned_current_admission": {
            "canonical_pointer_path": inputs.source.admission_current.path,
            "preflight_observed": preflight_observed,
            "launch_observed": launch_observed,
            "terminal_observed": bound_json(&terminal_source.admission_current),
            "immutable_manifest": bound_json(&inputs.source.manifest),
            "immutable_admission_receipt": bound_json(&inputs.source.admission_receipt),
            "acceptance": versioned_current_acceptance(),
        },
        "source_binding": {
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": inputs.source.source_revision,
            "manifest_document_sha256": inputs.source.manifest.raw_sha256,
            "manifest_seal_sha256": inputs.source.manifest_seal_sha256,
            "admission_receipt_seal_sha256": inputs.source.admission_receipt_seal_sha256,
            "manifest": bound_json(&inputs.source.manifest),
            "admission_current": bound_json(&inputs.source.admission_current),
            "admission_current_pointer_seal_sha256": inputs.source.admission_pointer_seal_sha256,
            "admission_receipt": bound_json(&inputs.source.admission_receipt),
            "source_audit_seal_sha256": inputs.source.source_audit_seal_sha256,
            "joint_l0_l1_assessment": {
                "document_sha256": inputs.joint_assessment.document_sha256,
                "document_seal_sha256": inputs.joint_assessment.document_seal_sha256,
            },
            "prior_joint_assessment_is_provenance_only": true,
            "cross_process_pinned_buffer_import_allowed": false,
        },
        "producer_preflight": bound_json(producer_preflight),
        "outer_launch_authority_binding": bound_json(outer_launch_authority),
        "producer_binary": inputs.producer_binary.json(),
        "cpu_outer_capture": {
            "capture_dir": capture_dir,
            "output_authority_path": args.out,
            "workers": workers,
            "one_current_admitted_catalog_scan_performed": true,
            "raw_bf16_or_safetensors_reopened": false,
            "outer_terminal_receipt_written_by_parent_last": true,
        },
        "source_token_l1_cpu_oracle": {
            "source_token_id": SOURCE_TOKEN_ID,
            "layer": L1_LAYER,
            "linear_state_slot": L1_LINEAR_STATE_SLOT,
            "fresh_l0_reencode_dispatches": L0_DISPATCHES,
            "fresh_l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "cpu_oracle_reencodes_l0_then_l1_prefix": true,
            "zero_initial_l0_state": true,
            "zero_initial_l1_slot1_state": true,
            "l1_input_derived_directly_from_fresh_l0_second_residual": true,
            "source_input_f32le_sha256": material.source_input_f32le_sha256,
            "l0_second_residual_cpu_f32le_sha256": material.l0_second_residual_cpu_f32le_sha256,
            "l1_prefix_input_cpu_f32le_sha256": material.l1_prefix_input_cpu_f32le_sha256,
            "l1_first_residual_cpu_f32le_sha256": material.l1_first_residual_cpu_f32le_sha256,
            "l1_post_attention_normalized_hidden_cpu_f32le_sha256": material.l1_post_attention_normalized_hidden_cpu_f32le_sha256,
            "l1_router_logits_cpu_f32le_sha256": material.l1_router_logits_cpu_f32le_sha256,
            "l1_post_conv_state_cpu_f32le_sha256": material.l1_post_conv_state_cpu_f32le_sha256,
            "l1_post_recurrent_state_cpu_f32le_sha256": material.l1_post_recurrent_state_cpu_f32le_sha256,
            "fresh_l0_post_conv_state_cpu_f32le_sha256": material.l0_post_conv_state_cpu_f32le_sha256,
            "fresh_l0_post_recurrent_state_cpu_f32le_sha256": material.l0_post_recurrent_state_cpu_f32le_sha256,
            "l1_routed_sum_cpu_f32le_sha256": material.l1_routed_sum_cpu_f32le_sha256,
            "l1_shared_gated_cpu_f32le_sha256": material.l1_shared_gated_cpu_f32le_sha256,
            "l1_moe_output_cpu_f32le_sha256": material.l1_moe_output_cpu_f32le_sha256,
            "l1_second_residual_cpu_f32le_sha256": material.l1_second_residual_cpu_f32le_sha256,
            "l1_shared_expert_gate_f32_bits_hex": format!("0x{:08x}", material.l1_shared_gate_value.to_bits()),
        },
        "source_token_router_evidence": {
            "derived_from_current_admitted_direct_packed_source_token_l0_then_l1_cpu_oracle": true,
            "logit_count": EXPERTS,
            "top_k": TOP_K,
            "selection": "source_qwen80_topk_router",
            "tie_break": "lowest_expert_id_within_route_tie_epsilon",
            "softmax": "subtract_max_exp_f32",
            "route_tie_epsilon_source": "HAWKING_DS_ROUTE_TIE_EPS",
            "route_tie_epsilon": f64::from(material.route_tie_epsilon_f32),
            "route_tie_epsilon_f32_bits_hex": format!(
                "0x{:08x}",
                material.route_tie_epsilon_f32.to_bits()
            ),
            "selected_probabilities_renormalized": true,
            "source_stable_route_ids": route_ids,
            "source_stable_normalized_weights": route_weights,
            "weights_sum": f64::from(route_weight_sum),
        },
        "fixed_l1_payloads": material.fixed_l1_payloads,
        "deterministic_waves": material.deterministic_waves,
        "rawls_real_all_ten_provenance_gate": {
            "schema": "hawking.ascension.qwen80_real_all_ten_routed_expert_provenance_gate_input.v1",
            "all_ten_source_bindings_complete": true,
            "expected_layer": L1_LAYER,
            "execution_receipt_required_for_each_wave": true,
            "direct_packed_execution_required_for_each_wave": true,
            "source_bound_input_required_for_each_wave": true,
            "route_combine_receipt_required_separately": true,
            "shared_expert_receipt_required_separately": true,
            "first_and_second_residual_receipts_required_separately": true,
            "rejects_tensor_substitution": true,
            "rejects_route_reorder": true,
            "rejects_duplicate_experts": true,
            "rejects_missing_tensor_or_weight": true,
        },
        "artifact_scan": {
            "strict_complete_artifact_admission_performed_once": true,
            "admitted_catalog_reused_for_embedding_l0_l1_router_and_all_36_descriptors": true,
            "catalog_tensor_count": material.catalog_tensor_count,
            "raw_bf16_or_safetensors_opened": false,
        },
        "cpu_oracle_execution": {
            "direct_packed_cpu_oracle_route_math_performed": true,
            "direct_packed_cpu_oracle_shared_and_second_residual_math_performed": true,
            "native_component_dispatch_performed": false,
            "route_guard_executed": false,
        },
        "route_execution_performed": false,
        "route_combine_performed": false,
        "shared_expert_performed": false,
        "residual_combine_performed": false,
        "metal_device_or_dispatch_performed": false,
        "model_execution_performed": false,
        "hcli_execution_performed": false,
        "tps_or_tg_measurement_performed": false,
        "complete_layer_or_decoder_claim_earned": false,
        "claim_boundary": {
            "cpu_source_authority_only": true,
            "fresh_l0_l1_cpu_lineage_is_not_a_transferable_device_buffer": true,
            "requires_new_same_runtime_l0_23_l1_prefix_9_moe_suffix_14_component_capture": true,
            "no_complete_layer_token_decoder_server_hcli_tps_tg_or_tournament_claim": true,
        },
    });
    seal(&mut output)?;
    Ok(output)
}

fn validate_descriptor(
    actual: &Value,
    expected: &Value,
    label: &str,
    seen_artifacts: &mut BTreeSet<String>,
    seen_payloads: &mut BTreeSet<String>,
) -> Result<(), String> {
    let actual = object(actual, label)?;
    let expected = object(expected, "descriptor expectation")?;
    for field in ["tensor_name", "shape", "group_size"] {
        if actual.get(field) != expected.get(field) {
            return Err(format!(
                "{label}.{field} drifted from source Layer-1 contract"
            ));
        }
    }
    if let Some(expected_role) = expected.get("role") {
        if actual.get("role") != Some(expected_role) {
            return Err(format!("{label}.role drifted from source Layer-1 contract"));
        }
    }
    let layout = object_field(actual, "layout", label)?;
    let expected_layout = object_field(expected, "required_layout", "descriptor expectation")?;
    for field in [
        "magic",
        "version",
        "group_size",
        "scale_dtype",
        "sign_bit_order",
    ] {
        if layout.get(field) != expected_layout.get(field) {
            return Err(format!("{label}.layout.{field} drifted"));
        }
    }
    if require_u64(actual, "payload_bytes", label)? == 0
        || require_u64(actual, "header_bytes", label)? == 0
    {
        return Err(format!("{label} has an empty compact descriptor"));
    }
    for (field, seen) in [
        ("artifact_sha256", seen_artifacts),
        ("direct_packed_payload_sha256", seen_payloads),
    ] {
        let hash = require_sha_field(actual, field, label)?;
        if !seen.insert(hash) {
            return Err(format!("{label} reuses {field}"));
        }
    }
    require_sha_field(actual, "header_sha256", label)?;
    require_bool(
        actual,
        "payload_from_admitted_catalog_snapshot",
        true,
        label,
    )?;
    require_bool(actual, "raw_artifact_reopened_by_this_child", false, label)?;
    Ok(())
}

fn validate_dynamic_authority(
    authority: &BoundDocument,
    inputs: &PreflightInputs,
    producer_preflight: &BoundDocument,
    outer_launch_authority: &BoundDocument,
    args: &Args,
    terminal_source: &SourceIdentity,
) -> Result<(), String> {
    let root = object(&authority.value, "Layer-1 dynamic route authority")?;
    require_exact_string(
        root,
        "schema",
        AUTHORITY_SCHEMA,
        "Layer-1 dynamic route authority",
    )?;
    require_exact_string(
        root,
        "status",
        AUTHORITY_STATUS,
        "Layer-1 dynamic route authority",
    )?;
    require_bool(
        root,
        "fixture_or_synthetic",
        false,
        "Layer-1 dynamic route authority",
    )?;
    require_bool(
        root,
        "metal_or_gpu_activity_performed",
        false,
        "Layer-1 dynamic route authority",
    )?;

    let source = object_field(root, "source_binding", "Layer-1 dynamic route authority")?;
    for (field, expected) in [
        ("model_id", MODEL_ID),
        ("model_key", MODEL_KEY),
        ("source_repository", SOURCE_REPOSITORY),
        ("source_revision", SOURCE_REVISION),
        ("manifest_document_sha256", MANIFEST_DOCUMENT_SHA256),
        ("manifest_seal_sha256", MANIFEST_SEAL_SHA256),
        (
            "admission_receipt_seal_sha256",
            ADMISSION_RECEIPT_SEAL_SHA256,
        ),
    ] {
        require_exact_string(
            source,
            field,
            expected,
            "Layer-1 dynamic route authority.source_binding",
        )?;
    }
    for (field, expected) in [
        ("manifest", bound_json(&inputs.source.manifest)),
        (
            "admission_receipt",
            bound_json(&inputs.source.admission_receipt),
        ),
    ] {
        require_evidence_matches(
            source,
            field,
            &expected,
            "Layer-1 dynamic route authority.source_binding",
        )?;
    }
    validate_historical_pointer_evidence(
        source
            .get("admission_current")
            .ok_or("Layer-1 dynamic route authority.source_binding.admission_current is absent")?,
        &inputs.source.admission_current.path,
        "Layer-1 dynamic route authority.source_binding.admission_current",
    )?;
    let versioned = root
        .get("versioned_current_admission")
        .ok_or("Layer-1 dynamic route authority.versioned_current_admission is absent")?;
    validate_versioned_current_observation(
        versioned,
        &inputs.source.admission_current.path,
        &inputs.source.manifest,
        &inputs.source.admission_receipt,
        &["preflight_observed", "launch_observed", "terminal_observed"],
        "Layer-1 dynamic route authority.versioned_current_admission",
    )?;
    let versioned_root = object(
        versioned,
        "Layer-1 dynamic route authority.versioned_current_admission",
    )?;
    let producer_versioned = object_field(
        object(
            &producer_preflight.value,
            "Layer-1 dynamic route authority producer preflight",
        )?,
        "versioned_current_admission",
        "Layer-1 dynamic route authority producer preflight",
    )?;
    let outer_versioned = object_field(
        object(
            &outer_launch_authority.value,
            "Layer-1 dynamic route authority outer launch authority",
        )?,
        "versioned_current_admission",
        "Layer-1 dynamic route authority outer launch authority",
    )?;
    if versioned_root.get("preflight_observed") != producer_versioned.get("preflight_observed")
        || versioned_root.get("launch_observed") != outer_versioned.get("launch_observed")
        || versioned_root.get("terminal_observed")
            != Some(&bound_json(&terminal_source.admission_current))
    {
        return Err(
            "Layer-1 dynamic route authority versioned-current observations are inconsistent"
                .into(),
        );
    }
    require_same_immutable_source(
        &inputs.source,
        terminal_source,
        "Layer-1 dynamic route authority terminal source",
    )?;
    let joint = object_field(
        source,
        "joint_l0_l1_assessment",
        "Layer-1 dynamic route authority.source_binding",
    )?;
    if joint.get("document_sha256")
        != Some(&Value::String(
            inputs.joint_assessment.document_sha256.clone(),
        ))
        || joint.get("document_seal_sha256")
            != Some(&Value::String(
                inputs.joint_assessment.document_seal_sha256.clone(),
            ))
    {
        return Err("dynamic authority did not bind the earned L0-L1 assessment".into());
    }
    for field in ["prior_joint_assessment_is_provenance_only"] {
        require_bool(
            source,
            field,
            true,
            "Layer-1 dynamic route authority.source_binding",
        )?;
    }
    require_bool(
        source,
        "cross_process_pinned_buffer_import_allowed",
        false,
        "Layer-1 dynamic route authority.source_binding",
    )?;
    require_evidence_matches(
        root,
        "producer_preflight",
        &bound_json(producer_preflight),
        "Layer-1 dynamic route authority",
    )?;
    require_evidence_matches(
        root,
        "outer_launch_authority_binding",
        &bound_json(outer_launch_authority),
        "Layer-1 dynamic route authority",
    )?;
    require_evidence_matches(
        root,
        "producer_binary",
        &inputs.producer_binary.json(),
        "Layer-1 dynamic route authority",
    )?;

    let oracle = object_field(
        root,
        "source_token_l1_cpu_oracle",
        "Layer-1 dynamic route authority",
    )?;
    for (field, expected) in [
        ("source_token_id", u64::from(SOURCE_TOKEN_ID)),
        ("layer", L1_LAYER as u64),
        ("linear_state_slot", L1_LINEAR_STATE_SLOT as u64),
        ("fresh_l0_reencode_dispatches", L0_DISPATCHES as u64),
        ("fresh_l1_prefix_dispatches", L1_PREFIX_DISPATCHES as u64),
    ] {
        if require_u64(
            oracle,
            field,
            "Layer-1 dynamic route authority.source_token_l1_cpu_oracle",
        )? != expected
        {
            return Err(format!("dynamic authority CPU oracle {field} drifted"));
        }
    }
    for field in [
        "cpu_oracle_reencodes_l0_then_l1_prefix",
        "zero_initial_l0_state",
        "zero_initial_l1_slot1_state",
        "l1_input_derived_directly_from_fresh_l0_second_residual",
    ] {
        require_bool(
            oracle,
            field,
            true,
            "Layer-1 dynamic route authority.source_token_l1_cpu_oracle",
        )?;
    }
    for field in [
        "source_input_f32le_sha256",
        "l0_second_residual_cpu_f32le_sha256",
        "l1_prefix_input_cpu_f32le_sha256",
        "l1_first_residual_cpu_f32le_sha256",
        "l1_post_attention_normalized_hidden_cpu_f32le_sha256",
        "l1_router_logits_cpu_f32le_sha256",
        "l1_post_conv_state_cpu_f32le_sha256",
        "l1_post_recurrent_state_cpu_f32le_sha256",
        "fresh_l0_post_conv_state_cpu_f32le_sha256",
        "fresh_l0_post_recurrent_state_cpu_f32le_sha256",
        "l1_routed_sum_cpu_f32le_sha256",
        "l1_shared_gated_cpu_f32le_sha256",
        "l1_moe_output_cpu_f32le_sha256",
        "l1_second_residual_cpu_f32le_sha256",
    ] {
        require_sha_field(
            oracle,
            field,
            "Layer-1 dynamic route authority.source_token_l1_cpu_oracle",
        )?;
    }
    if require_sha_field(
        oracle,
        "l0_second_residual_cpu_f32le_sha256",
        "Layer-1 dynamic route authority.source_token_l1_cpu_oracle",
    )? != require_sha_field(
        oracle,
        "l1_prefix_input_cpu_f32le_sha256",
        "Layer-1 dynamic route authority.source_token_l1_cpu_oracle",
    )? {
        return Err("dynamic authority L1 input does not equal fresh L0 second residual".into());
    }

    let route = object_field(
        root,
        "source_token_router_evidence",
        "Layer-1 dynamic route authority",
    )?;
    for (field, expected) in [("logit_count", EXPERTS as u64), ("top_k", TOP_K as u64)] {
        if require_u64(
            route,
            field,
            "Layer-1 dynamic route authority.source_token_router_evidence",
        )? != expected
        {
            return Err(format!("dynamic authority router {field} drifted"));
        }
    }
    for (field, expected) in [
        ("selection", "source_qwen80_topk_router"),
        ("tie_break", "lowest_expert_id_within_route_tie_epsilon"),
        ("softmax", "subtract_max_exp_f32"),
        ("route_tie_epsilon_source", "HAWKING_DS_ROUTE_TIE_EPS"),
    ] {
        require_exact_string(
            route,
            field,
            expected,
            "Layer-1 dynamic route authority.source_token_router_evidence",
        )?;
    }
    require_bool(
        route,
        "selected_probabilities_renormalized",
        true,
        "Layer-1 dynamic route authority.source_token_router_evidence",
    )?;
    let epsilon = route
        .get("route_tie_epsilon")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or("dynamic authority router tie epsilon is invalid")?;
    let bits = string(
        route,
        "route_tie_epsilon_f32_bits_hex",
        "Layer-1 dynamic route authority.source_token_router_evidence",
    )?
    .strip_prefix("0x")
    .filter(|value| value.len() == 8 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
    .ok_or("dynamic authority router tie epsilon bits are invalid")?;
    let bits = u32::from_str_radix(bits, 16)
        .map_err(|error| format!("dynamic authority router tie epsilon bits failed: {error}"))?;
    if f64::from(f32::from_bits(bits)).to_bits() != epsilon.to_bits() {
        return Err("dynamic authority router tie epsilon/bits drifted".into());
    }
    let ids = array_field(
        route,
        "source_stable_route_ids",
        "Layer-1 dynamic route authority.source_token_router_evidence",
    )?;
    let weights = array_field(
        route,
        "source_stable_normalized_weights",
        "Layer-1 dynamic route authority.source_token_router_evidence",
    )?;
    if ids.len() != TOP_K || weights.len() != TOP_K {
        return Err("dynamic authority must retain exactly ten source routes".into());
    }
    let mut route_ids = Vec::with_capacity(TOP_K);
    let mut route_weights = Vec::with_capacity(TOP_K);
    let mut seen_ids = BTreeSet::new();
    for (index, (id, weight)) in ids.iter().zip(weights).enumerate() {
        let id = id
            .as_u64()
            .filter(|value| *value < EXPERTS as u64)
            .ok_or_else(|| format!("dynamic authority route id {index} is invalid"))?;
        if !seen_ids.insert(id) {
            return Err("dynamic authority route IDs are not unique".into());
        }
        let weight = weight
            .as_f64()
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| format!("dynamic authority route weight {index} is invalid"))?;
        route_ids.push(id);
        route_weights.push(weight);
    }
    let calculated_weight_sum = route_weights.iter().sum::<f64>();
    let declared_weight_sum = route
        .get("weights_sum")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite())
        .ok_or("dynamic authority router weights_sum is invalid")?;
    if (calculated_weight_sum - 1.0).abs() > f64::from(MAX_ROUTE_WEIGHT_SUM_ERROR)
        || (declared_weight_sum - calculated_weight_sum).abs()
            > f64::from(MAX_ROUTE_WEIGHT_SUM_ERROR)
    {
        return Err("dynamic authority source route weights do not normalize".into());
    }

    let fixed = array_field(root, "fixed_l1_payloads", "Layer-1 dynamic route authority")?;
    let expected_fixed = fixed_payload_requirements();
    if fixed.len() != expected_fixed.len() {
        return Err("dynamic authority must bind six fixed Layer-1 payloads".into());
    }
    let mut seen_artifacts = BTreeSet::new();
    let mut seen_payloads = BTreeSet::new();
    for (index, (actual, expected)) in fixed.iter().zip(&expected_fixed).enumerate() {
        validate_descriptor(
            actual,
            expected,
            &format!("dynamic authority fixed descriptor {index}"),
            &mut seen_artifacts,
            &mut seen_payloads,
        )?;
    }
    let waves = array_field(
        root,
        "deterministic_waves",
        "Layer-1 dynamic route authority",
    )?;
    if waves.len() != TOP_K {
        return Err("dynamic authority must bind ten deterministic waves".into());
    }
    for (index, wave) in waves.iter().enumerate() {
        let wave = object(wave, "dynamic authority deterministic wave")?;
        if require_u64(wave, "wave_index", "dynamic authority deterministic wave")? != index as u64
            || require_u64(wave, "layer", "dynamic authority deterministic wave")?
                != L1_LAYER as u64
            || require_u64(wave, "expert_id", "dynamic authority deterministic wave")?
                != route_ids[index]
        {
            return Err(format!(
                "dynamic authority wave {index} route identity drifted"
            ));
        }
        let weight = wave
            .get("normalized_weight")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite())
            .ok_or_else(|| format!("dynamic authority wave {index} weight is invalid"))?;
        if weight.to_bits() != route_weights[index].to_bits() {
            return Err(format!("dynamic authority wave {index} weight drifted"));
        }
        require_exact_string(
            wave,
            "normalized_weight_bits_hex",
            &format!("0x{:016x}", route_weights[index].to_bits()),
            "dynamic authority deterministic wave",
        )?;
        for role in ["gate", "up", "down"] {
            validate_descriptor(
                wave.get(role)
                    .ok_or_else(|| format!("dynamic authority wave {index} lacks {role}"))?,
                &route_descriptor_requirement(role, route_ids[index] as u16)?,
                &format!("dynamic authority wave {index} {role}"),
                &mut seen_artifacts,
                &mut seen_payloads,
            )?;
        }
    }
    if seen_artifacts.len() != 36 || seen_payloads.len() != 36 {
        return Err("dynamic authority descriptor uniqueness is incomplete".into());
    }

    let provenance = object_field(
        root,
        "rawls_real_all_ten_provenance_gate",
        "Layer-1 dynamic route authority",
    )?;
    require_exact_string(
        provenance,
        "schema",
        "hawking.ascension.qwen80_real_all_ten_routed_expert_provenance_gate_input.v1",
        "Layer-1 dynamic route authority.rawls_real_all_ten_provenance_gate",
    )?;
    if require_u64(
        provenance,
        "expected_layer",
        "Layer-1 dynamic route authority.rawls_real_all_ten_provenance_gate",
    )? != L1_LAYER as u64
    {
        return Err("dynamic authority provenance gate is not locked to Layer-1".into());
    }
    for field in [
        "all_ten_source_bindings_complete",
        "execution_receipt_required_for_each_wave",
        "direct_packed_execution_required_for_each_wave",
        "source_bound_input_required_for_each_wave",
        "route_combine_receipt_required_separately",
        "shared_expert_receipt_required_separately",
        "first_and_second_residual_receipts_required_separately",
        "rejects_tensor_substitution",
        "rejects_route_reorder",
        "rejects_duplicate_experts",
        "rejects_missing_tensor_or_weight",
    ] {
        require_bool(
            provenance,
            field,
            true,
            "Layer-1 dynamic route authority.rawls_real_all_ten_provenance_gate",
        )?;
    }

    let outer = object_field(root, "cpu_outer_capture", "Layer-1 dynamic route authority")?;
    let capture_dir = canonical_directory(
        args.capture_dir
            .as_deref()
            .ok_or("dynamic authority has no CPU capture directory")?,
        "dynamic authority capture directory",
    )?;
    if string(
        outer,
        "capture_dir",
        "Layer-1 dynamic route authority.cpu_outer_capture",
    )? != capture_dir.to_string_lossy()
        || string(
            outer,
            "output_authority_path",
            "Layer-1 dynamic route authority.cpu_outer_capture",
        )? != args.out.to_string_lossy()
        || require_u64(
            outer,
            "workers",
            "Layer-1 dynamic route authority.cpu_outer_capture",
        )? != args
            .workers
            .ok_or("dynamic authority has no worker count")? as u64
    {
        return Err("dynamic authority outer CPU binding drifted".into());
    }
    for field in [
        "one_current_admitted_catalog_scan_performed",
        "outer_terminal_receipt_written_by_parent_last",
    ] {
        require_bool(
            outer,
            field,
            true,
            "Layer-1 dynamic route authority.cpu_outer_capture",
        )?;
    }
    require_bool(
        outer,
        "raw_bf16_or_safetensors_reopened",
        false,
        "Layer-1 dynamic route authority.cpu_outer_capture",
    )?;
    for field in [
        "route_execution_performed",
        "route_combine_performed",
        "shared_expert_performed",
        "residual_combine_performed",
        "metal_device_or_dispatch_performed",
        "model_execution_performed",
        "hcli_execution_performed",
        "tps_or_tg_measurement_performed",
        "complete_layer_or_decoder_claim_earned",
    ] {
        require_bool(root, field, false, "Layer-1 dynamic route authority")?;
    }
    let boundary = object_field(root, "claim_boundary", "Layer-1 dynamic route authority")?;
    for field in [
        "cpu_source_authority_only",
        "fresh_l0_l1_cpu_lineage_is_not_a_transferable_device_buffer",
        "requires_new_same_runtime_l0_23_l1_prefix_9_moe_suffix_14_component_capture",
        "no_complete_layer_token_decoder_server_hcli_tps_tg_or_tournament_claim",
    ] {
        require_bool(
            boundary,
            field,
            true,
            "Layer-1 dynamic route authority.claim_boundary",
        )?;
    }
    Ok(())
}

fn write_new(path: &Path, contents: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(contents)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn rendered_document(value: &Value) -> Result<Vec<u8>, String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot render output: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn temporary_bound_document(path: PathBuf, value: Value) -> Result<BoundDocument, String> {
    let document_seal_sha256 = verify_seal(&value, "generated dynamic authority")?;
    let document_sha256 = document_sha256(&value, "generated dynamic authority")?;
    let raw = rendered_document(&value)?;
    Ok(BoundDocument {
        path,
        bytes: u64::try_from(raw.len()).map_err(|_| "generated output is too large")?,
        raw_sha256: sha256_hex(&raw),
        document_sha256,
        document_seal_sha256,
        value,
    })
}

fn run(args: Args) -> Result<Value, String> {
    match args.mode {
        Mode::Preflight => {
            let inputs = load_preflight_inputs(&args, true)?;
            let mut output = producer_preflight_document(&inputs);
            seal(&mut output)?;
            let bytes = rendered_document(&output)?;
            write_new(&args.out, &bytes)?;
            Ok(output)
        }
        Mode::CpuOracle => {
            let inputs = load_preflight_inputs(&args, true)?;
            let producer_preflight = read_bound_document(
                args.producer_preflight
                    .as_deref()
                    .ok_or("cpu-oracle mode has no producer preflight")?,
                "producer preflight",
            )?;
            validate_producer_preflight(&producer_preflight, &inputs)?;
            let outer_launch_authority = read_bound_document(
                args.outer_launch_authority
                    .as_deref()
                    .ok_or("cpu-oracle mode has no outer launch authority")?,
                "outer launch authority",
            )?;
            validate_outer_launch_authority(
                &outer_launch_authority,
                &inputs,
                &producer_preflight,
                &args,
            )?;
            let admission = CompleteBinaryAdmission {
                model: QwenCompleteBinaryModel::Qwen80CoderNext,
                expected_manifest_seal_sha256: inputs.source.manifest_seal_sha256.clone(),
                expected_source_audit_seal_sha256: inputs.source.source_audit_seal_sha256.clone(),
                expected_source_revision: inputs.source.source_revision.clone(),
            };
            // This is the only mode and line that opens the admitted compact
            // artifact. It runs only after all outer and replay authorities
            // have validated.
            let catalog = Qwen80CompleteArtifactCatalog::load(&args.manifest, &admission)
                .map_err(|error| format!("strict Qwen80 artifact admission refused: {error}"))?;
            let material = material_from_current_catalog(&catalog, &inputs.source)?;
            // The mutable admission-current pointer is revalidated after the
            // one catalog pass.  Only its observed bytes/seal may differ; the
            // immutable manifest and admission receipt must still match the
            // launch authority exactly.
            let terminal_source = validate_source_identity(&args)?;
            require_same_immutable_source(
                &inputs.source,
                &terminal_source,
                "Layer-1 CPU authority terminal source",
            )?;
            let output = build_dynamic_authority(
                &inputs,
                &producer_preflight,
                &outer_launch_authority,
                &args,
                &material,
                &terminal_source,
            )?;
            let generated = temporary_bound_document(args.out.clone(), output.clone())?;
            validate_dynamic_authority(
                &generated,
                &inputs,
                &producer_preflight,
                &outer_launch_authority,
                &args,
                &terminal_source,
            )?;
            let bytes = rendered_document(&output)?;
            write_new(&args.out, &bytes)?;
            Ok(output)
        }
    }
}

fn main() {
    let args = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("Qwen80 source-token L1 route authority argument refusal: {error}");
            process::exit(2);
        }
    };
    match run(args) {
        Ok(output) => {
            println!(
                "{}",
                json!({
                    "schema": output["schema"],
                    "status": output["status"],
                    "seal_sha256": output["seal_sha256"],
                    "metal_or_gpu_activity_performed": false,
                })
            );
        }
        Err(error) => {
            eprintln!("Qwen80 source-token L1 route authority refusal: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered_sha(value: u64) -> String {
        format!("{value:064x}")
    }

    fn sealed(mut value: Value) -> Value {
        seal(&mut value).expect("fixture must seal");
        value
    }

    fn fake_bound(value: Value, path: &str) -> BoundDocument {
        let document_seal_sha256 = verify_seal(&value, "fixture").expect("fixture seal");
        let document_sha256 = document_sha256(&value, "fixture").expect("fixture document hash");
        let raw = rendered_document(&value).expect("fixture render");
        BoundDocument {
            path: PathBuf::from(path),
            bytes: raw.len() as u64,
            raw_sha256: sha256_hex(&raw),
            document_sha256,
            document_seal_sha256,
            value,
        }
    }

    fn assessed_joint_document() -> BoundDocument {
        fake_bound(
            sealed(json!({
                "schema": JOINT_ASSESSMENT_SCHEMA,
                "status": JOINT_ASSESSMENT_STATUS,
                "earned_component_only": true,
                "component_scope": {
                    "source_token_id": SOURCE_TOKEN_ID,
                    "fresh_l0_dispatches": L0_DISPATCHES,
                    "fresh_l1_slot1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                    "fresh_total_dispatches": L0_DISPATCHES + L1_PREFIX_DISPATCHES,
                    "opaque_same_runtime_continuation_required": true,
                    "single_fence_required": true,
                    "full_layer_or_token_decoder_earned": false,
                },
                "claim_boundary": {
                    "cpu_only_post_capture_assessment": true,
                    "l0_l1_component_not_full_layer_token_decoder": true,
                    "does_not_reuse_historical_l0_receipt_as_execution_input": true,
                    "does_not_accept_raw_pinned_buffer_or_dispatch_count_input": true,
                    "does_not_construct_metal_or_dispatch": true,
                    "does_not_issue_or_release_lease": true,
                    "does_not_start_runtime_server_or_watcher": true,
                    "does_not_measure_tps_or_tg": true,
                    "does_not_claim_decoder_token_or_tournament": true,
                },
            })),
            "/tmp/joint-assessment.json",
        )
    }

    fn fake_inputs() -> PreflightInputs {
        let mut manifest = fake_bound(
            sealed(json!({"schema": MANIFEST_SCHEMA})),
            "/tmp/manifest.json",
        );
        manifest.raw_sha256 = MANIFEST_DOCUMENT_SHA256.into();
        manifest.document_seal_sha256 = MANIFEST_SEAL_SHA256.into();
        let admission_current = fake_bound(
            sealed(json!({
                "schema": ADMISSION_POINTER_SCHEMA,
                "status": ADMISSION_POINTER_STATUS,
            })),
            "/tmp/admission-current.json",
        );
        let mut admission_receipt = fake_bound(
            sealed(json!({
                "schema": ADMISSION_RECEIPT_SCHEMA,
                "status": ADMISSION_RECEIPT_STATUS,
            })),
            "/tmp/admission-receipt.json",
        );
        admission_receipt.document_seal_sha256 = ADMISSION_RECEIPT_SEAL_SHA256.into();
        let joint_assessment = assessed_joint_document();
        let completion_preflight = fake_bound(
            sealed(json!({
                "schema": COMPLETION_PREFLIGHT_SCHEMA,
                "status": COMPLETION_PREFLIGHT_STATUS,
                "preflight_ready_for_future_outer_authority_only": false,
                "antecedent_l0_l1_component": {
                    "document_sha256": joint_assessment.document_sha256,
                    "document_seal_sha256": joint_assessment.document_seal_sha256,
                },
                "l1_source_token_route_authority": {"present_and_valid": false},
                "future_joint_command_graph": {
                    "source_token_id": SOURCE_TOKEN_ID,
                    "l0_layer": L0_LAYER,
                    "l1_layer": L1_LAYER,
                    "l1_linear_state_slot": L1_LINEAR_STATE_SLOT,
                    "l0_reencode_dispatches": L0_DISPATCHES,
                    "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                    "l1_moe_suffix_dispatches": 14,
                    "total_dispatches": 46,
                    "single_runtime_required": true,
                    "single_token_command_buffer_required": true,
                    "single_fence_after_all_dispatches_required": true,
                    "non_timed_trace_required": true,
                },
                "claim_boundary": {
                    "cpu_file_only_preflight": true,
                    "artifact_scan_or_payload_open_performed": false,
                    "metal_context_or_dispatch_performed": false,
                    "lease_issued_or_consumed": false,
                    "watcher_server_hcli_or_runtime_changed": false,
                    "future_l1_moe_component_is_not_a_complete_token_or_decoder": true,
                    "tps_tg_or_tournament_claim_earned": false,
                },
            })),
            "/tmp/completion-preflight.json",
        );
        PreflightInputs {
            source: SourceIdentity {
                manifest,
                admission_pointer_seal_sha256: admission_current.document_seal_sha256.clone(),
                admission_current,
                admission_receipt,
                manifest_seal_sha256: MANIFEST_SEAL_SHA256.into(),
                admission_receipt_seal_sha256: ADMISSION_RECEIPT_SEAL_SHA256.into(),
                source_audit_seal_sha256: numbered_sha(802),
                source_revision: SOURCE_REVISION.into(),
            },
            joint_assessment,
            completion_preflight,
            producer_binary: FileEvidence {
                path: PathBuf::from(
                    "/tmp/ascension_qwen80_source_token_l1_all_ten_route_authority_cpu",
                ),
                bytes: 4096,
                sha256: numbered_sha(803),
            },
        }
    }

    fn cpu_args() -> Args {
        let capture_dir = std::env::temp_dir()
            .canonicalize()
            .expect("temporary directory must exist");
        Args {
            mode: Mode::CpuOracle,
            manifest: PathBuf::from("/tmp/manifest.json"),
            admission_current: PathBuf::from("/tmp/admission-current.json"),
            joint_assessment: PathBuf::from("/tmp/joint-assessment.json"),
            completion_preflight: PathBuf::from("/tmp/completion-preflight.json"),
            producer_binary: PathBuf::from(
                "/tmp/ascension_qwen80_source_token_l1_all_ten_route_authority_cpu",
            ),
            producer_preflight: Some(PathBuf::from("/tmp/producer-preflight.json")),
            outer_launch_authority: Some(PathBuf::from("/tmp/outer-authority.json")),
            capture_dir: Some(capture_dir.clone()),
            workers: Some(1),
            out: capture_dir.join("new-l1-route-authority.json"),
        }
    }

    fn producer_preflight_fixture(inputs: &PreflightInputs) -> BoundDocument {
        let mut document = producer_preflight_document(inputs);
        seal(&mut document).expect("producer preflight must seal");
        fake_bound(document, "/tmp/producer-preflight.json")
    }

    fn outer_authority_fixture(
        inputs: &PreflightInputs,
        producer_preflight: &BoundDocument,
        args: &Args,
    ) -> BoundDocument {
        let capture_dir = canonical_directory(
            args.capture_dir.as_deref().expect("capture dir"),
            "fixture capture directory",
        )
        .expect("capture directory must canonicalize");
        fake_bound(
            sealed(json!({
                "schema": OUTER_AUTHORITY_SCHEMA,
                "status": OUTER_AUTHORITY_STATUS,
                "producer_preflight": bound_json(producer_preflight),
                "versioned_current_admission": {
                    "canonical_pointer_path": inputs.source.admission_current.path,
                    "preflight_observed": producer_preflight.value["versioned_current_admission"]["preflight_observed"].clone(),
                    "launch_observed": bound_json(&inputs.source.admission_current),
                    "immutable_manifest": bound_json(&inputs.source.manifest),
                    "immutable_admission_receipt": bound_json(&inputs.source.admission_receipt),
                    "acceptance": versioned_current_acceptance(),
                },
                "source_binding": {
                    "manifest": bound_json(&inputs.source.manifest),
                    "admission_current": bound_json(&inputs.source.admission_current),
                    "admission_receipt": bound_json(&inputs.source.admission_receipt),
                    "joint_assessment": bound_json(&inputs.joint_assessment),
                    "completion_preflight": bound_json(&inputs.completion_preflight),
                    "manifest_seal_sha256": inputs.source.manifest_seal_sha256,
                    "admission_receipt_seal_sha256": inputs.source.admission_receipt_seal_sha256,
                },
                "producer_binary": inputs.producer_binary.json(),
                "planned_capture_dir": capture_dir,
                "planned_output_authority": args.out,
                "workers": args.workers,
                "execution_policy": {
                    "exact_catalog_admission_scans": 1,
                    "cpu_oracle_only": true,
                    "metal_or_gpu_allowed": false,
                    "lease_allowed": false,
                    "watcher_or_server_allowed": false,
                    "automatic_retry_allowed": false,
                    "outer_reaped_required": true,
                    "terminal_receipt_written_last_required": true,
                },
                "replay_guard": {
                    "capture_dir_unique": true,
                    "one_child_maximum": true,
                },
            })),
            "/tmp/outer-authority.json",
        )
    }

    fn descriptor(expected: &Value, nonce: u64, role: Option<&str>) -> Value {
        let expected = object(expected, "descriptor requirement").expect("descriptor requirement");
        let layout =
            object_field(expected, "required_layout", "descriptor requirement").expect("layout");
        json!({
            "role": role.or_else(|| expected.get("role").and_then(Value::as_str)),
            "tensor_name": expected["tensor_name"],
            "shape": expected["shape"],
            "group_size": expected["group_size"],
            "artifact_sha256": numbered_sha(nonce * 3 + 1),
            "direct_packed_payload_sha256": numbered_sha(nonce * 3 + 2),
            "header_sha256": numbered_sha(nonce * 3 + 3),
            "header_bytes": 40,
            "payload_bytes": 128,
            "layout": layout,
            "payload_from_admitted_catalog_snapshot": true,
            "raw_artifact_reopened_by_this_child": false,
        })
    }

    fn material_fixture() -> CpuAuthorityMaterial {
        let fixed_l1_payloads = fixed_payload_requirements()
            .iter()
            .enumerate()
            .map(|(index, expected)| descriptor(expected, index as u64, None))
            .collect::<Vec<_>>();
        let route_ids = (100_u16..110).collect::<Vec<_>>();
        let route_weights = vec![0.1_f32; TOP_K];
        let deterministic_waves = route_ids
            .iter()
            .enumerate()
            .map(|(index, expert)| {
                json!({
                    "wave_index": index,
                    "layer": L1_LAYER,
                    "expert_id": expert,
                    "normalized_weight": f64::from(route_weights[index]),
                    "normalized_weight_bits_hex": format!(
                        "0x{:016x}",
                        f64::from(route_weights[index]).to_bits()
                    ),
                    "gate": descriptor(
                        &route_descriptor_requirement("gate", *expert).unwrap(),
                        100 + (index * 3) as u64,
                        Some("gate"),
                    ),
                    "up": descriptor(
                        &route_descriptor_requirement("up", *expert).unwrap(),
                        101 + (index * 3) as u64,
                        Some("up"),
                    ),
                    "down": descriptor(
                        &route_descriptor_requirement("down", *expert).unwrap(),
                        102 + (index * 3) as u64,
                        Some("down"),
                    ),
                    "cpu_weighted_output_f32le_sha256": numbered_sha(500 + index as u64),
                    "route_execution_status": "CPU_ORACLE_PARITY_ONLY_NOT_NATIVE_COMPONENT_EXECUTION",
                })
            })
            .collect::<Vec<_>>();
        CpuAuthorityMaterial {
            source_input_f32le_sha256: numbered_sha(1),
            l0_second_residual_cpu_f32le_sha256: numbered_sha(2),
            l0_post_conv_state_cpu_f32le_sha256: numbered_sha(3),
            l0_post_recurrent_state_cpu_f32le_sha256: numbered_sha(4),
            l1_prefix_input_cpu_f32le_sha256: numbered_sha(2),
            l1_first_residual_cpu_f32le_sha256: numbered_sha(5),
            l1_post_attention_normalized_hidden_cpu_f32le_sha256: numbered_sha(6),
            l1_router_logits_cpu_f32le_sha256: numbered_sha(7),
            l1_post_conv_state_cpu_f32le_sha256: numbered_sha(8),
            l1_post_recurrent_state_cpu_f32le_sha256: numbered_sha(9),
            l1_routed_sum_cpu_f32le_sha256: numbered_sha(10),
            l1_shared_gated_cpu_f32le_sha256: numbered_sha(11),
            l1_moe_output_cpu_f32le_sha256: numbered_sha(12),
            l1_second_residual_cpu_f32le_sha256: numbered_sha(13),
            l1_shared_gate_value: 0.5,
            route_tie_epsilon_f32: 0.0,
            route_ids,
            route_weights,
            fixed_l1_payloads,
            deterministic_waves,
            catalog_tensor_count: 74_391,
        }
    }

    #[test]
    fn producer_preflight_is_sealed_and_strictly_nonexecuting() {
        let inputs = fake_inputs();
        let document = producer_preflight_fixture(&inputs);
        validate_producer_preflight(&document, &inputs).expect("preflight must validate");
        let boundary = object_field(
            object(&document.value, "producer preflight").unwrap(),
            "claim_boundary",
            "producer preflight",
        )
        .unwrap();
        assert_eq!(
            boundary["strict_catalog_admission_scan_performed"],
            Value::Bool(false)
        );
        assert_eq!(
            boundary["metal_or_gpu_activity_performed"],
            Value::Bool(false)
        );
    }

    #[test]
    fn versioned_current_pointer_reseal_is_accepted_but_immutable_drift_is_refused() {
        let inputs = fake_inputs();
        let producer_preflight = producer_preflight_fixture(&inputs);
        let mut resealed_inputs = inputs.clone();
        resealed_inputs.source.admission_current = fake_bound(
            sealed(json!({
                "schema": ADMISSION_POINTER_SCHEMA,
                "status": ADMISSION_POINTER_STATUS,
                "watcher_housekeeping_reseal": true,
            })),
            "/tmp/admission-current.json",
        );
        resealed_inputs.source.admission_pointer_seal_sha256 = resealed_inputs
            .source
            .admission_current
            .document_seal_sha256
            .clone();

        validate_producer_preflight(&producer_preflight, &resealed_inputs)
            .expect("a canonical pointer reseal must retain preflight eligibility");

        let args = cpu_args();
        let outer = outer_authority_fixture(&inputs, &producer_preflight, &args);
        validate_outer_launch_authority(&outer, &resealed_inputs, &producer_preflight, &args)
            .expect("a canonical pointer reseal must retain outer eligibility");

        let authority = build_dynamic_authority(
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &material_fixture(),
            &resealed_inputs.source,
        )
        .expect("terminal pointer reseal must be representable");
        let authority = temporary_bound_document(args.out.clone(), authority).unwrap();
        validate_dynamic_authority(
            &authority,
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &resealed_inputs.source,
        )
        .expect("terminal pointer reseal must validate");

        let mut immutable_drift = resealed_inputs;
        immutable_drift.source.manifest.raw_sha256 = numbered_sha(9_999);
        assert!(validate_producer_preflight(&producer_preflight, &immutable_drift).is_err());

        let mut receipt_drift = inputs.clone();
        receipt_drift.source.admission_receipt.raw_sha256 = numbered_sha(10_000);
        assert!(validate_producer_preflight(&producer_preflight, &receipt_drift).is_err());

        let mut terminal_immutable_drift = inputs.clone();
        terminal_immutable_drift.source.manifest.raw_sha256 = numbered_sha(10_001);
        assert!(build_dynamic_authority(
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &material_fixture(),
            &terminal_immutable_drift.source,
        )
        .is_err());
    }

    #[test]
    fn dynamic_authority_binds_exact_l1_payloads_and_fresh_cpu_lineage() {
        let inputs = fake_inputs();
        let args = cpu_args();
        let producer_preflight = producer_preflight_fixture(&inputs);
        let outer = outer_authority_fixture(&inputs, &producer_preflight, &args);
        validate_outer_launch_authority(&outer, &inputs, &producer_preflight, &args)
            .expect("outer authority must validate");
        let authority = build_dynamic_authority(
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &material_fixture(),
            &inputs.source,
        )
        .expect("dynamic authority must build");
        let authority = temporary_bound_document(args.out.clone(), authority).unwrap();
        validate_dynamic_authority(
            &authority,
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &inputs.source,
        )
        .expect("dynamic authority must validate");
        assert_eq!(
            authority.value["source_token_l1_cpu_oracle"]["l0_second_residual_cpu_f32le_sha256"],
            authority.value["source_token_l1_cpu_oracle"]["l1_prefix_input_cpu_f32le_sha256"]
        );
    }

    #[test]
    fn dynamic_authority_refuses_duplicate_route_payload_identity() {
        let inputs = fake_inputs();
        let args = cpu_args();
        let producer_preflight = producer_preflight_fixture(&inputs);
        let outer = outer_authority_fixture(&inputs, &producer_preflight, &args);
        let mut material = material_fixture();
        material.deterministic_waves[1]["gate"]["direct_packed_payload_sha256"] =
            material.deterministic_waves[0]["gate"]["direct_packed_payload_sha256"].clone();
        assert!(build_dynamic_authority(
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &material,
            &inputs.source,
        )
        .is_err());
    }

    #[test]
    fn dynamic_authority_refuses_reordered_or_tampered_dynamic_route() {
        let inputs = fake_inputs();
        let args = cpu_args();
        let producer_preflight = producer_preflight_fixture(&inputs);
        let outer = outer_authority_fixture(&inputs, &producer_preflight, &args);
        let authority = build_dynamic_authority(
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &material_fixture(),
            &inputs.source,
        )
        .unwrap();
        let mut reordered = authority.clone();
        reordered["deterministic_waves"]
            .as_array_mut()
            .unwrap()
            .swap(0, 1);
        reordered.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut reordered).unwrap();
        let reordered = temporary_bound_document(args.out.clone(), reordered).unwrap();
        assert!(validate_dynamic_authority(
            &reordered,
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &inputs.source,
        )
        .is_err());

        let mut tampered = authority.clone();
        tampered["source_token_router_evidence"]["route_tie_epsilon_source"] =
            json!("not-the-source-policy");
        tampered.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut tampered).unwrap();
        let tampered = temporary_bound_document(args.out.clone(), tampered).unwrap();
        assert!(validate_dynamic_authority(
            &tampered,
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &inputs.source,
        )
        .is_err());

        let mut weakened = authority;
        weakened["rawls_real_all_ten_provenance_gate"]["rejects_route_reorder"] = json!(false);
        weakened.as_object_mut().unwrap().remove("seal_sha256");
        seal(&mut weakened).unwrap();
        let weakened = temporary_bound_document(args.out.clone(), weakened).unwrap();
        assert!(validate_dynamic_authority(
            &weakened,
            &inputs,
            &producer_preflight,
            &outer,
            &args,
            &inputs.source,
        )
        .is_err());
    }

    #[test]
    fn parser_refuses_cpu_oracle_without_outer_bindings() {
        let arguments = vec![
            "--mode",
            "cpu-oracle",
            "--manifest",
            "/tmp/manifest",
            "--admission-current",
            "/tmp/admission",
            "--joint-assessment",
            "/tmp/joint",
            "--completion-preflight",
            "/tmp/preflight",
            "--producer-binary",
            "/tmp/probe",
            "--capture-dir",
            "/tmp",
            "--workers",
            "1",
            "--out",
            "/tmp/new-output",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
        assert!(parse_args(arguments).is_err());
    }
}
