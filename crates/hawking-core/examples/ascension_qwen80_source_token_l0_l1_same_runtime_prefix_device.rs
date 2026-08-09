//! CPU/build-only concrete host for the future source-token L0→L1 component.
//!
//! The historical L0 handoff receipt is deliberately not an input buffer.
//! When a separately hardened outer/reaper eventually authorizes Metal, this
//! host must construct a new runtime, re-encode source-token L0's exact
//! 9+14 path, turn its retained live resources into the opaque core
//! continuation, append L1/slot-1's exact nine-dispatch DeltaNet prefix, and
//! hand the only command buffer to the consuming finalizer.  The original
//! CLI exposed only a CPU preflight; its `metal` mode is now a strictly gated,
//! outer-reaped child interface.  It accepts only a new joint-specific lease
//! plus an outer-launch authority, never imports a prior receipt buffer, and
//! writes its inner receipt last.  Nothing selects that mode implicitly.

#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_source_token_all_ten_true_moe_graph_device.rs"]
mod source_l0;
#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_first_residual_bridge_device.rs"]
mod source_prefix;

#[cfg(target_os = "macos")]
use hawking_core::metal::TokenCommandBuffer;
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime, Qwen80CompleteRuntimeOptions,
    Qwen80SameRuntimeL0L1PrefixParity,
};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const HOST_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_host_preflight.v1";
const HOST_PREFLIGHT_STATUS: &str =
    "COMPILED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const STATIC_PLAN_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_child_preflight.v1";
const STATIC_PLAN_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_CHILD_NOT_EXECUTED";
const L0_SOURCE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1";
const L0_SOURCE_OUTER_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED";
const JOINT_OUTER_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_outer_preflight.v1";
const JOINT_OUTER_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const JOINT_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_quiet_metal_lease.v1";
const JOINT_LEASE_STATUS: &str =
    "GRANTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_COMPONENT_QUIET_METAL_LEASE";
const EXECUTION_BINDING_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_host_execution_binding.v1";
const EXECUTION_BINDING_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_STRICT_HOST_EXECUTION_INTERFACE";
const JOINT_OUTER_LAUNCH_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_outer_launch_authority.v1";
const JOINT_OUTER_LAUNCH_STATUS: &str =
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_OUTER_REAPED_ONE_SHOT_METAL_CHILD";
const JOINT_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_capture.v1";
const JOINT_INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_COMPONENT_ONLY";
const JOINT_INNER_REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_SLOT1_PREFIX_SAME_RUNTIME_PHASE_ACCURATE_TERMINAL_FAILURE";
const SCHEDULE_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_schedule_sealed_wrapper.v1";
const SCHEDULE_STATUS: &str =
    "SEALED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_BOUND_NOT_EXECUTED";
const CONTINUATION_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_source_token_continuation_readiness_contract.v1";
const CONTINUATION_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L1_SLOT1_DELTANET_PREFIX_CAPTURE_RESERVED_NOT_EXECUTED";
const ASSESSOR_BINDING_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_state_handoff_post_capture_assessor_binding.v1";
const ASSESSOR_BINDING_STATUS: &str =
    "REQUIRED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_POST_CAPTURE_ASSESSMENT_BEFORE_L1_JOINT_CAPTURE";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const SOURCE_TOKEN_ID: u64 = 1;
const L0_DISPATCHES: u64 = 23;
const L1_PREFIX_DISPATCHES: u64 = 9;
const JOINT_DISPATCHES: u64 = L0_DISPATCHES + L1_PREFIX_DISPATCHES;
const MAX_JSON_BYTES: u64 = 100_000_000;

const L0_KERNELS: [&str; 23] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
    "qwen80_postnorm_router_top10_rmsnorm",
    "qwen80_postnorm_router_top10_matvec",
    "qwen80_postnorm_router_top10_select",
    "qwen80_all_ten_routed_wave_route_guard",
    "qwen80_all_ten_routed_wave_gate_up",
    "qwen80_all_ten_routed_wave_swiglu",
    "qwen80_all_ten_routed_wave_down_weighted",
    "qwen80_shared_expert_wave_gate_up",
    "qwen80_shared_expert_wave_swiglu",
    "qwen80_shared_expert_wave_down",
    "qwen80_shared_expert_wave_scalar_gate",
    "qwen80_shared_expert_wave_apply_sigmoid_gate",
    "qwen80_moe_wave_aggregate_second_residual_route_sum",
    "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
];
const L1_KERNELS: [&str; 9] = [
    "qwen_next_direct_packed_input_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_qkvz_rearrange_conv_l2",
    "qwen_next_ba_to_decay_beta",
    "qwen_next_gated_delta_decode_single",
    "qwen_next_deltanet_gated_rmsnorm",
    "qwen_binary_sign_scale_matvec",
    "qwen_next_add_residual",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Preflight,
    SourceAdmissionPreflight,
    Metal,
}

#[derive(Clone, Debug)]
struct Args {
    mode: Mode,
    joint_static_plan: Option<PathBuf>,
    l0_source_outer_preflight: Option<PathBuf>,
    joint_outer_preflight: Option<PathBuf>,
    lease_receipt: Option<PathBuf>,
    outer_launch_authority: Option<PathBuf>,
    outer_capture_dir: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    out: Option<PathBuf>,
    workers: usize,
}

#[derive(Clone, Debug)]
struct FileEvidence {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug)]
struct SealedFile {
    file: FileEvidence,
    document: Value,
    document_sha256: String,
    seal_sha256: String,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct JointCaptureAuthority {
    outer_preflight: SealedFile,
    joint_static_plan: SealedFile,
    l0_source_outer_preflight: SealedFile,
    schedule: SealedFile,
    continuation: SealedFile,
    assessor_binding: SealedFile,
    manifest: SealedFile,
    admission_receipt: SealedFile,
    lease: SealedFile,
    lease_id: String,
    outer_launch: SealedFile,
    host_binary: FileEvidence,
    workers: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default)]
struct MetalExecutionPhase {
    strict_artifact_admission_started: bool,
    strict_artifact_admission_succeeded: bool,
    metal_context_construction_attempted: bool,
    metal_context_constructed: bool,
    structural_kernel_trace_enabled: bool,
    dispatches_encoded: usize,
    encoded_kernel_names: Vec<String>,
    command_commit_may_have_been_attempted: bool,
    command_fence_succeeded: bool,
    readback_started: bool,
}

fn usage() -> &'static str {
    "usage (CPU preflight): ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device \\
--joint-static-plan ABSOLUTE_PATH --l0-source-outer-preflight ABSOLUTE_PATH \\
--out ABSOLUTE_NEW_FILE --workers 1..4\n\
usage (read-only source-admission preflight): ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device \\
--mode source-admission-preflight --joint-outer-preflight ABSOLUTE_PATH --workers 1..4\n\\
usage (outer-reaped Metal child only): ascension_qwen80_source_token_l0_l1_same_runtime_prefix_device \\
--mode metal --joint-outer-preflight ABSOLUTE_PATH --lease-receipt ABSOLUTE_PATH \\
--outer-launch-authority ABSOLUTE_PATH --outer-capture-dir ABSOLUTE_DIRECTORY \\
--capture-dir ABSOLUTE_NEW_DIRECTORY --workers 1..4"
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

/// Render finite JSON floats with CPython's `json.dumps` spelling. Rust's
/// normal scientific exponent spelling differs for values such as `1e-6`, so
/// this is required for receipts also sealed by `lab.receipts.seal`.
fn python_json_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("canonical JSON floating number must be finite")?;
    if value == 0.0 {
        return Ok(if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into());
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
                .map_err(|error| format!("canonical JSON exponent is invalid: {error}"))?,
        ),
        None => (unsigned, 0),
    };
    let mut fractional_digits = 0i32;
    let mut after_decimal = false;
    let mut digits = String::new();
    for byte in mantissa.bytes() {
        match byte {
            b'.' if !after_decimal => after_decimal = true,
            b'0'..=b'9' => {
                if after_decimal {
                    fractional_digits = fractional_digits
                        .checked_add(1)
                        .ok_or("canonical JSON fractional digit count overflows")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err(format!("canonical JSON mantissa is invalid: {raw:?}")),
        }
    }
    let first_significant = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("nonzero canonical JSON float has no significant digit")?;
    let mut significant = digits[first_significant..].to_owned();
    let mut decimal_power = exponent
        .checked_sub(fractional_digits)
        .ok_or("canonical JSON decimal exponent overflows")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        decimal_power = decimal_power
            .checked_add(1)
            .ok_or("canonical JSON decimal exponent overflows")?;
    }
    let scientific_exponent = decimal_power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("canonical JSON decimal exponent overflows")?;
    let sign = if negative { "-" } else { "" };
    if !(-4..16).contains(&scientific_exponent) {
        let mut rendered_mantissa = significant[..1].to_owned();
        if significant.len() > 1 {
            rendered_mantissa.push('.');
            rendered_mantissa.push_str(&significant[1..]);
        }
        let exponent_sign = if scientific_exponent < 0 { '-' } else { '+' };
        return Ok(format!(
            "{sign}{rendered_mantissa}e{exponent_sign}{:02}",
            scientific_exponent.unsigned_abs()
        ));
    }
    let decimal_position = scientific_exponent + 1;
    let rendered = if decimal_position <= 0 {
        format!(
            "0.{}{}",
            "0".repeat(usize::try_from(-decimal_position).unwrap_or(usize::MAX)),
            significant
        )
    } else if usize::try_from(decimal_position).unwrap_or(usize::MAX) >= significant.len() {
        format!(
            "{}{}.0",
            significant,
            "0".repeat(usize::try_from(decimal_position).unwrap_or(usize::MAX) - significant.len())
        )
    } else {
        let position = usize::try_from(decimal_position).unwrap();
        format!("{}.{}", &significant[..position], &significant[position..])
    };
    Ok(format!("{sign}{rendered}"))
}

fn canonical_json_into(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) if number.is_i64() || number.is_u64() => {
            output.push_str(&number.to_string())
        }
        Value::Number(number) => output.push_str(&python_json_float(number)?),
        Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("cannot canonicalize JSON string: {error}"))?,
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
            let mut keys: Vec<&String> = values.keys().collect();
            keys.sort_unstable();
            output.push('{');
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(*key)
                        .map_err(|error| format!("cannot canonicalize JSON key: {error}"))?,
                );
                output.push(':');
                canonical_json_into(&values[*key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_json(value: &Value) -> Result<Vec<u8>, String> {
    let mut output = String::new();
    canonical_json_into(value, &mut output)?;
    Ok(output.into_bytes())
}

fn sha256_json(value: &Value) -> Result<String, String> {
    canonical_json(value).map(|bytes| sha256_hex(&bytes))
}

fn seal(document: &mut Value) -> Result<String, String> {
    if document
        .as_object()
        .ok_or("host preflight must be an object")?
        .contains_key("seal_sha256")
    {
        return Err("host preflight must not already carry a seal".into());
    }
    let seal = sha256_json(document)?;
    document
        .as_object_mut()
        .expect("validated host preflight object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn verify_seal(document: &Value, label: &str) -> Result<String, String> {
    let object = document
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    let observed = object
        .get("seal_sha256")
        .and_then(Value::as_str)
        .filter(|value| is_sha256(value))
        .ok_or_else(|| format!("{label} lacks a lowercase seal_sha256"))?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    if sha256_json(&Value::Object(unsigned))? != observed {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed.to_owned())
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn read_json(path: &Path, label: &str) -> Result<(FileEvidence, Value), String> {
    let path = canonical_regular(path, label)?;
    let bytes = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(format!("{label} exceeds bounded JSON size"));
    }
    let value: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("cannot parse {label}: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((
        FileEvidence {
            path,
            bytes: bytes.len() as u64,
            sha256: sha256_hex(&bytes),
        },
        value,
    ))
}

fn sealed_file(
    path: &Path,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<SealedFile, String> {
    let (file, document) = read_json(path, label)?;
    let seal_sha256 = verify_seal(&document, label)?;
    let root = object(&document, label)?;
    if field_string(root, "schema", label)? != expected_schema
        || field_string(root, "status", label)? != expected_status
    {
        return Err(format!("{label} schema/status drifted"));
    }
    Ok(SealedFile {
        file,
        document_sha256: sha256_json(&document)?,
        document,
        seal_sha256,
    })
}

fn evidence_json(file: &FileEvidence) -> Value {
    json!({"path": file.path, "bytes": file.bytes, "sha256": file.sha256})
}

fn sealed_binding_json(document: &SealedFile) -> Value {
    json!({
        "path": document.file.path,
        "bytes": document.file.bytes,
        "sha256": document.file.sha256,
        "document_sha256": document.document_sha256,
        "document_seal_sha256": document.seal_sha256,
    })
}

fn binding_matches(
    value: &Map<String, Value>,
    expected: &SealedFile,
    label: &str,
) -> Result<(), String> {
    if value.get("present").is_some() && value.get("present").and_then(Value::as_bool) != Some(true)
    {
        return Err(format!("{label}.present must be true"));
    }
    if field_string(value, "path", label)? != expected.file.path.to_string_lossy()
        || value.get("bytes").and_then(Value::as_u64) != Some(expected.file.bytes)
        || field_string(value, "sha256", label)? != expected.file.sha256
        || field_string(value, "document_sha256", label)? != expected.document_sha256
        || field_string(value, "document_seal_sha256", label)? != expected.seal_sha256
    {
        return Err(format!("{label} evidence/document identity drifted"));
    }
    Ok(())
}

/// The admission receipt's `complete_manifest.document_sha256` is the SHA-256
/// of the immutable manifest *file bytes*.  `SealedFile::document_sha256` is
/// our canonical JSON identity, which intentionally differs whenever the
/// persisted formatting differs.  Keep both identities available, but bind
/// this upstream receipt to the raw file SHA mandated by its schema.
fn immutable_admission_manifest_matches(
    admitted_manifest: &Map<String, Value>,
    manifest: &SealedFile,
) -> Result<bool, String> {
    Ok(field_string(
        admitted_manifest,
        "document_sha256",
        "joint capture immutable admission receipt.complete_manifest",
    )? == manifest.file.sha256
        && field_string(
            admitted_manifest,
            "seal_sha256",
            "joint capture immutable admission receipt.complete_manifest",
        )? == manifest.seal_sha256)
}

fn load_bound_from_binding(
    value: &Map<String, Value>,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<SealedFile, String> {
    let path = PathBuf::from(field_string(value, "path", label)?);
    let bound = sealed_file(&path, label, expected_schema, expected_status)?;
    binding_matches(value, &bound, label)?;
    Ok(bound)
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn field_object<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn field_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
}

fn require_u64(
    object: &Map<String, Value>,
    field: &str,
    expected: u64,
    label: &str,
) -> Result<(), String> {
    if object.get(field).and_then(Value::as_u64) != Some(expected) {
        return Err(format!("{label}.{field} must be {expected}"));
    }
    Ok(())
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

fn expected_trace(kernels: &[&str]) -> Vec<Value> {
    kernels
        .iter()
        .enumerate()
        .map(|(index, kernel)| json!({"ordinal": index + 1, "kernel": kernel}))
        .collect()
}

fn exact_joint_trace() -> Vec<Value> {
    L0_KERNELS
        .iter()
        .chain(L1_KERNELS.iter())
        .enumerate()
        .map(|(index, kernel)| json!({"ordinal": index + 1, "kernel": kernel}))
        .collect()
}

fn validate_static_plan(value: &Value, own_binary: &FileEvidence) -> Result<String, String> {
    let seal = verify_seal(value, "joint static plan")?;
    let root = object(value, "joint static plan")?;
    if field_string(root, "schema", "joint static plan")? != STATIC_PLAN_SCHEMA
        || field_string(root, "status", "joint static plan")? != STATIC_PLAN_STATUS
    {
        return Err("joint static plan schema/status drifted".into());
    }
    require_bool(
        root,
        "future_joint_host_binary_bound",
        true,
        "joint static plan",
    )?;
    if field_string(root, "future_joint_host_binary_role", "joint static plan")?
        != "strict_joint_l0_l1_same_runtime_host"
    {
        return Err("joint static plan does not name this concrete strict joint host".into());
    }
    if field_string(root, "future_joint_l0_l1_child_sha256", "joint static plan")?
        != own_binary.sha256
    {
        return Err("joint static plan host SHA does not bind this executable".into());
    }
    require_bool(root, "same_runtime_required", true, "joint static plan")?;
    require_bool(root, "same_session_required", true, "joint static plan")?;
    require_bool(root, "same_tcb_required", true, "joint static plan")?;
    require_bool(
        root,
        "opaque_canonical_l0_continuation_required",
        true,
        "joint static plan",
    )?;
    require_bool(
        root,
        "raw_pinned_buffer_or_dispatch_count_input_allowed",
        false,
        "joint static plan",
    )?;
    let graph = field_object(root, "joint_command_graph", "joint static plan")?;
    require_u64(
        graph,
        "l0_dispatches",
        L0_DISPATCHES,
        "joint static plan.joint_command_graph",
    )?;
    require_u64(
        graph,
        "l1_prefix_dispatches",
        L1_PREFIX_DISPATCHES,
        "joint static plan.joint_command_graph",
    )?;
    require_u64(
        graph,
        "total_dispatches",
        JOINT_DISPATCHES,
        "joint static plan.joint_command_graph",
    )?;
    require_bool(
        graph,
        "single_fence_after_l0_and_l1_prefix_required",
        true,
        "joint static plan.joint_command_graph",
    )?;
    require_bool(
        graph,
        "non_timed_token_command_buffer_required",
        true,
        "joint static plan.joint_command_graph",
    )?;
    if field_string(
        graph,
        "tcb_trace_mode",
        "joint static plan.joint_command_graph",
    )? != "off"
    {
        return Err("joint static plan requires TcbTraceMode::Off".into());
    }
    if graph.get("exact_l0_kernel_trace") != Some(&Value::Array(expected_trace(&L0_KERNELS)))
        || graph.get("exact_joint_kernel_trace") != Some(&Value::Array(exact_joint_trace()))
    {
        return Err("joint static plan exact structural kernel trace drifted".into());
    }
    if field_string(graph, "opaque_capability_factory", "joint static plan.joint_command_graph")?
        != "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation"
        || field_string(graph, "runtime_api", "joint static plan.joint_command_graph")?
            != "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into"
        || field_string(graph, "consuming_finalizer", "joint static plan.joint_command_graph")?
            != "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence"
    {
        return Err("joint static plan opaque continuation/finalizer ABI drifted".into());
    }
    Ok(seal)
}

fn validate_l0_source_outer(value: &Value) -> Result<String, String> {
    let seal = verify_seal(value, "source-token L0 outer preflight")?;
    let root = object(value, "source-token L0 outer preflight")?;
    if field_string(root, "schema", "source-token L0 outer preflight")? != L0_SOURCE_OUTER_SCHEMA
        || field_string(root, "status", "source-token L0 outer preflight")?
            != L0_SOURCE_OUTER_STATUS
    {
        return Err("source-token L0 outer preflight schema/status drifted".into());
    }
    let route = field_object(
        root,
        "source_token_route",
        "source-token L0 outer preflight",
    )?;
    if route.get("token_id").and_then(Value::as_u64) != Some(SOURCE_TOKEN_ID) {
        return Err("source-token L0 outer preflight must bind token one".into());
    }
    require_bool(
        route,
        "same_command_graph_required",
        true,
        "source-token L0 outer preflight.source_token_route",
    )?;
    require_bool(
        route,
        "zero_l0_state_required",
        true,
        "source-token L0 outer preflight.source_token_route",
    )?;
    require_bool(
        route,
        "all_ten_unique",
        true,
        "source-token L0 outer preflight.source_token_route",
    )?;
    let boundary = field_object(root, "claim_boundary", "source-token L0 outer preflight")?;
    require_bool(
        boundary,
        "lease_issued",
        false,
        "source-token L0 outer preflight.claim_boundary",
    )?;
    require_bool(
        boundary,
        "metal_device_or_dispatch_performed",
        false,
        "source-token L0 outer preflight.claim_boundary",
    )?;
    let next = field_object(
        root,
        "next_child_contract",
        "source-token L0 outer preflight",
    )?;
    require_bool(
        next,
        "requires_same_tcb_prefix_lineage",
        true,
        "source-token L0 outer preflight.next_child_contract",
    )?;
    require_bool(
        next,
        "requires_source_token_authority_and_typed_bridge",
        true,
        "source-token L0 outer preflight.next_child_contract",
    )?;
    Ok(seal)
}

fn source_route_from_l0_outer(value: &Value) -> Result<(Vec<u32>, Vec<f64>), String> {
    let root = object(value, "source-token L0 outer preflight")?;
    let route = field_object(
        root,
        "source_token_route",
        "source-token L0 outer preflight",
    )?;
    let ids = route
        .get("route_ids")
        .and_then(Value::as_array)
        .ok_or("source-token L0 outer preflight route IDs must be an array")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or_else(|| "source-token L0 outer preflight route ID is invalid".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = route
        .get("normalized_weights")
        .and_then(Value::as_array)
        .ok_or("source-token L0 outer preflight route weights must be an array")?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .filter(|value| value.is_finite() && *value >= 0.0)
                .ok_or_else(|| "source-token L0 outer preflight route weight is invalid".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != 10
        || weights.len() != 10
        || ids.iter().collect::<std::collections::BTreeSet<_>>().len() != 10
    {
        return Err(
            "source-token L0 outer preflight must retain ten unique route IDs/weights".into(),
        );
    }
    if (weights.iter().sum::<f64>() - 1.0).abs() > 2.0e-6 {
        return Err("source-token L0 outer preflight route weights are not normalized".into());
    }
    Ok((ids, weights))
}

fn canonical_new_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let parent = path
        .parent()
        .filter(|parent| parent.is_absolute())
        .ok_or_else(|| format!("{label} needs an absolute parent"))?;
    let metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat {label} parent {}: {error}", parent.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{label} parent must be a real directory"));
    }
    if path.exists() {
        return Err(format!("{label} must be create-new"));
    }
    Ok(path.to_path_buf())
}

fn canonical_directory(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!("{label} must be a regular non-symlink directory"));
    }
    fs::canonicalize(path).map_err(|error| format!("cannot canonicalize {label}: {error}"))
}

fn absolute_path(path: PathBuf, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    Ok(path)
}

fn write_new(path: &Path, document: &Value) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(document)
        .map_err(|error| format!("cannot encode host preflight: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .map_err(|error| format!("cannot write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync {}: {error}", path.display()))
}

fn parse_args(arguments: &[String]) -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut index = 1usize;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{flag} needs a value; {}", usage()))?;
        match flag.as_str() {
            "--mode"
            | "--joint-static-plan"
            | "--l0-source-outer-preflight"
            | "--joint-outer-preflight"
            | "--lease-receipt"
            | "--outer-launch-authority"
            | "--outer-capture-dir"
            | "--capture-dir"
            | "--out"
            | "--workers" => {
                if values.insert(flag.clone(), value.clone()).is_some() {
                    return Err(format!("{flag} may not be repeated; {}", usage()));
                }
            }
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        }
        index += 2;
    }
    let mode = match values.remove("--mode").as_deref() {
        None | Some("preflight") => Mode::Preflight,
        Some("source-admission-preflight") => Mode::SourceAdmissionPreflight,
        Some("metal") => Mode::Metal,
        _ => {
            return Err(format!(
                "--mode must be preflight, source-admission-preflight, or metal; {}",
                usage()
            ))
        }
    };
    let workers = values
        .remove("--workers")
        .ok_or_else(|| format!("--workers 1..4 is required; {}", usage()))?
        .parse::<usize>()
        .map_err(|_| "--workers must be an integer")?;
    if !(1..=4).contains(&workers) {
        return Err("--workers must be in 1..=4".into());
    }
    if mode == Mode::SourceAdmissionPreflight
        && values.keys().any(|flag| flag != "--joint-outer-preflight")
    {
        return Err(format!(
            "source-admission-preflight refuses capture arguments: {:?}",
            values.keys()
        ));
    }
    let mut required = |flag: &str| -> Result<PathBuf, String> {
        absolute_path(
            PathBuf::from(
                values
                    .remove(flag)
                    .ok_or_else(|| format!("{flag} is required; {}", usage()))?,
            ),
            flag,
        )
    };
    let args = match mode {
        Mode::Preflight => {
            let joint_static_plan =
                canonical_regular(&required("--joint-static-plan")?, "--joint-static-plan")?;
            let l0_source_outer_preflight = canonical_regular(
                &required("--l0-source-outer-preflight")?,
                "--l0-source-outer-preflight",
            )?;
            let out = canonical_new_file(&required("--out")?, "--out")?;
            if !values.is_empty() {
                return Err(format!(
                    "preflight mode refuses execution arguments: {:?}",
                    values.keys()
                ));
            }
            Args {
                mode,
                joint_static_plan: Some(joint_static_plan),
                l0_source_outer_preflight: Some(l0_source_outer_preflight),
                joint_outer_preflight: None,
                lease_receipt: None,
                outer_launch_authority: None,
                outer_capture_dir: None,
                capture_dir: None,
                out: Some(out),
                workers,
            }
        }
        Mode::SourceAdmissionPreflight => {
            let joint_outer_preflight = canonical_regular(
                &required("--joint-outer-preflight")?,
                "--joint-outer-preflight",
            )?;
            if !values.is_empty() {
                return Err(format!(
                    "source-admission-preflight refuses capture arguments: {:?}",
                    values.keys()
                ));
            }
            Args {
                mode,
                joint_static_plan: None,
                l0_source_outer_preflight: None,
                joint_outer_preflight: Some(joint_outer_preflight),
                lease_receipt: None,
                outer_launch_authority: None,
                outer_capture_dir: None,
                capture_dir: None,
                out: None,
                workers,
            }
        }
        Mode::Metal => {
            let joint_outer_preflight = canonical_regular(
                &required("--joint-outer-preflight")?,
                "--joint-outer-preflight",
            )?;
            let lease_receipt =
                canonical_regular(&required("--lease-receipt")?, "--lease-receipt")?;
            let outer_launch_authority = canonical_regular(
                &required("--outer-launch-authority")?,
                "--outer-launch-authority",
            )?;
            let outer_capture_dir =
                canonical_directory(&required("--outer-capture-dir")?, "--outer-capture-dir")?;
            let capture_dir = required("--capture-dir")?;
            if capture_dir.exists() {
                return Err("--capture-dir must be a create-new path".into());
            }
            if !values.is_empty() {
                return Err(format!(
                    "metal mode refuses preflight-only arguments: {:?}",
                    values.keys()
                ));
            }
            Args {
                mode,
                joint_static_plan: None,
                l0_source_outer_preflight: None,
                joint_outer_preflight: Some(joint_outer_preflight),
                lease_receipt: Some(lease_receipt),
                outer_launch_authority: Some(outer_launch_authority),
                outer_capture_dir: Some(outer_capture_dir),
                capture_dir: Some(capture_dir),
                out: None,
                workers,
            }
        }
    };
    Ok(args)
}

fn preflight_document(args: &Args) -> Result<Value, String> {
    if args.mode != Mode::Preflight {
        return Err("host preflight document may only be built in preflight mode".into());
    }
    let own_path = env::current_exe()
        .map_err(|error| format!("cannot resolve current host executable: {error}"))?;
    let own_bytes = fs::read(&own_path)
        .map_err(|error| format!("cannot read current host executable: {error}"))?;
    let own = FileEvidence {
        path: canonical_regular(&own_path, "current host executable")?,
        bytes: own_bytes.len() as u64,
        sha256: sha256_hex(&own_bytes),
    };
    let joint_static_plan = args
        .joint_static_plan
        .as_ref()
        .ok_or("preflight mode lacks joint static plan")?;
    let (static_evidence, static_plan) = read_json(joint_static_plan, "joint static plan")?;
    let static_seal = validate_static_plan(&static_plan, &own)?;
    let static_document_sha256 = sha256_json(&static_plan)?;
    let (l0_evidence, l0_outer) = read_json(
        args.l0_source_outer_preflight
            .as_ref()
            .ok_or("preflight mode lacks source-token L0 outer preflight")?,
        "source-token L0 outer preflight",
    )?;
    let l0_seal = validate_l0_source_outer(&l0_outer)?;
    let l0_outer_document_sha256 = sha256_json(&l0_outer)?;
    let mut document = json!({
        "schema": HOST_PREFLIGHT_SCHEMA,
        "status": HOST_PREFLIGHT_STATUS,
        "child_started": false,
        "metal_or_gpu_activity_performed": false,
        "lease_issued_or_consumed": false,
        "component_only": true,
        "host_binary": {"path": own.path, "bytes": own.bytes, "sha256": own.sha256},
        "joint_static_plan": {
            "path": static_evidence.path,
            "bytes": static_evidence.bytes,
            "sha256": static_evidence.sha256,
            "document_sha256": static_document_sha256,
            "seal_sha256": static_seal,
        },
        "l0_source_outer_preflight": {
            "path": l0_evidence.path,
            "bytes": l0_evidence.bytes,
            "sha256": l0_evidence.sha256,
            "document_sha256": l0_outer_document_sha256,
            "seal_sha256": l0_seal,
        },
        "workers": args.workers,
        "same_runtime_same_session_same_tcb_required": true,
        "joint_command_graph": {
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "total_dispatches": JOINT_DISPATCHES,
            "single_fence_required": true,
            "non_timed_token_command_buffer_required": true,
            "tcb_trace_mode": "off",
            "exact_l0_kernel_trace": expected_trace(&L0_KERNELS),
            "exact_joint_kernel_trace": exact_joint_trace(),
            "opaque_capability_factory": "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation",
            "runtime_api": "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into",
            "consuming_finalizer": "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence",
        },
        "host_body": {
            "strict_joint_entrypoint_compiled": true,
            "strict_joint_capture_interface_compiled": true,
            "metal_entrypoint_available_only_under_new_joint_lease_and_outer_launch_authority": true,
            "writes_assessor_compatible_inner_receipt_last": true,
            "phase_accurate_terminal_refusal_receipt_supported": true,
            "future_outer_reaper_and_fresh_lease_required_before_entrypoint_may_run": true,
            "historical_l0_receipt_or_pinned_buffer_import_allowed": false,
            "l1_suffix_or_moe_authorized": false,
        },
        "claim_boundary": {
            "complete_layer_or_token_decoder_hcli_tps_tg_or_tournament_allowed": false,
            "watcher_server_or_runtime_transition_allowed": false,
            "automatic_retry_allowed": false,
        },
    });
    seal(&mut document)?;
    Ok(document)
}

fn current_binary_evidence() -> Result<FileEvidence, String> {
    let own_path = env::current_exe()
        .map_err(|error| format!("cannot resolve current joint host executable: {error}"))?;
    let path = canonical_regular(&own_path, "current joint host executable")?;
    let bytes = fs::read(&path)
        .map_err(|error| format!("cannot read current joint host executable: {error}"))?;
    Ok(FileEvidence {
        path,
        bytes: bytes.len() as u64,
        sha256: sha256_hex(&bytes),
    })
}

fn binding_matches_with_seal_field(
    value: &Map<String, Value>,
    expected: &SealedFile,
    seal_field: &str,
    label: &str,
) -> Result<(), String> {
    if field_string(value, "path", label)? != expected.file.path.to_string_lossy()
        || value.get("bytes").and_then(Value::as_u64) != Some(expected.file.bytes)
        || field_string(value, "sha256", label)? != expected.file.sha256
        || field_string(value, "document_sha256", label)? != expected.document_sha256
        || field_string(value, seal_field, label)? != expected.seal_sha256
    {
        return Err(format!("{label} evidence/document identity drifted"));
    }
    Ok(())
}

fn load_chain_document(
    chain: &Map<String, Value>,
    field: &str,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<SealedFile, String> {
    let binding = field_object(chain, field, label)?;
    load_bound_from_binding(
        binding,
        &format!("{label}.{field}"),
        expected_schema,
        expected_status,
    )
}

fn require_sha_field(
    object: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<String, String> {
    let value = field_string(object, field, label)?.to_owned();
    if !is_sha256(&value) {
        return Err(format!("{label}.{field} must be a lowercase SHA-256"));
    }
    Ok(value)
}

fn require_exact_bool(
    object: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    require_bool(object, field, expected, label)
}

#[cfg(target_os = "macos")]
fn validate_joint_outer_preflight(
    path: &Path,
    host_binary: &FileEvidence,
) -> Result<
    (
        SealedFile,
        SealedFile,
        SealedFile,
        SealedFile,
        SealedFile,
        SealedFile,
        SealedFile,
        SealedFile,
    ),
    String,
> {
    let outer = sealed_file(
        path,
        "joint L0→L1 outer preflight",
        JOINT_OUTER_PREFLIGHT_SCHEMA,
        JOINT_OUTER_PREFLIGHT_STATUS,
    )?;
    let root = object(&outer.document, "joint L0→L1 outer preflight")?;
    for (field, expected) in [
        ("prepared", true),
        ("child_started", false),
        ("metal_or_gpu_activity_performed", false),
        ("lease_issued_or_consumed", false),
    ] {
        require_exact_bool(root, field, expected, "joint L0→L1 outer preflight")?;
    }
    let scope = field_object(root, "exact_joint_scope", "joint L0→L1 outer preflight")?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("total_dispatches", JOINT_DISPATCHES),
    ] {
        require_u64(
            scope,
            field,
            expected,
            "joint L0→L1 outer preflight.exact_joint_scope",
        )?;
    }
    require_exact_bool(
        scope,
        "single_fence_required",
        true,
        "joint L0→L1 outer preflight.exact_joint_scope",
    )?;
    require_exact_bool(
        scope,
        "non_timed_required",
        true,
        "joint L0→L1 outer preflight.exact_joint_scope",
    )?;
    if field_string(
        scope,
        "tcb_trace_mode",
        "joint L0→L1 outer preflight.exact_joint_scope",
    )? != "off"
        || field_string(
            scope,
            "host_binary_sha256",
            "joint L0→L1 outer preflight.exact_joint_scope",
        )? != host_binary.sha256
    {
        return Err("joint outer preflight host/TCB scope drifted".into());
    }
    let interface = field_object(
        root,
        "host_execution_interface",
        "joint L0→L1 outer preflight",
    )?;
    require_exact_bool(
        interface,
        "compiled_host_preflight_only",
        false,
        "joint L0→L1 outer preflight.host_execution_interface",
    )?;
    require_exact_bool(
        interface,
        "metal_entrypoint_available",
        true,
        "joint L0→L1 outer preflight.host_execution_interface",
    )?;
    require_exact_bool(
        interface,
        "writes_assessor_compatible_inner_receipt_last",
        true,
        "joint L0→L1 outer preflight.host_execution_interface",
    )?;
    let chain = field_object(root, "authority_chain", "joint L0→L1 outer preflight")?;
    let static_plan = load_chain_document(
        chain,
        "joint_l0_l1_child_preflight",
        "joint L0→L1 outer preflight.authority_chain",
        STATIC_PLAN_SCHEMA,
        STATIC_PLAN_STATUS,
    )?;
    let host_preflight = load_chain_document(
        chain,
        "joint_l0_l1_host_preflight",
        "joint L0→L1 outer preflight.authority_chain",
        HOST_PREFLIGHT_SCHEMA,
        HOST_PREFLIGHT_STATUS,
    )?;
    let l0_source_outer_preflight = load_chain_document(
        chain,
        "l0_source_outer_preflight",
        "joint L0→L1 outer preflight.authority_chain",
        L0_SOURCE_OUTER_SCHEMA,
        L0_SOURCE_OUTER_STATUS,
    )?;
    let schedule = load_chain_document(
        chain,
        "schedule",
        "joint L0→L1 outer preflight.authority_chain",
        SCHEDULE_SCHEMA,
        SCHEDULE_STATUS,
    )?;
    let continuation = load_chain_document(
        chain,
        "continuation_readiness",
        "joint L0→L1 outer preflight.authority_chain",
        CONTINUATION_SCHEMA,
        CONTINUATION_STATUS,
    )?;
    let assessor_binding = load_chain_document(
        chain,
        "l0_post_capture_assessor_binding",
        "joint L0→L1 outer preflight.authority_chain",
        ASSESSOR_BINDING_SCHEMA,
        ASSESSOR_BINDING_STATUS,
    )?;
    let manifest = load_chain_document(
        chain,
        "manifest",
        "joint L0→L1 outer preflight.authority_chain",
        MANIFEST_SCHEMA,
        MANIFEST_STATUS,
    )?;
    let admission_receipt = load_chain_document(
        chain,
        "admission_receipt",
        "joint L0→L1 outer preflight.authority_chain",
        ADMISSION_RECEIPT_SCHEMA,
        ADMISSION_RECEIPT_STATUS,
    )?;
    validate_static_plan(&static_plan.document, host_binary)?;
    let host_root = object(&host_preflight.document, "joint host preflight")?;
    let host_file = field_object(host_root, "host_binary", "joint host preflight")?;
    if field_string(host_file, "path", "joint host preflight.host_binary")?
        != host_binary.path.to_string_lossy()
        || host_file.get("bytes").and_then(Value::as_u64) != Some(host_binary.bytes)
        || field_string(host_file, "sha256", "joint host preflight.host_binary")?
            != host_binary.sha256
    {
        return Err("joint host preflight no longer binds the current executable".into());
    }
    binding_matches_with_seal_field(
        field_object(host_root, "joint_static_plan", "joint host preflight")?,
        &static_plan,
        "seal_sha256",
        "joint host preflight.joint_static_plan",
    )?;
    binding_matches_with_seal_field(
        field_object(
            host_root,
            "l0_source_outer_preflight",
            "joint host preflight",
        )?,
        &l0_source_outer_preflight,
        "seal_sha256",
        "joint host preflight.l0_source_outer_preflight",
    )?;
    let host_body = field_object(host_root, "host_body", "joint host preflight")?;
    for field in [
        "strict_joint_entrypoint_compiled",
        "strict_joint_capture_interface_compiled",
        "metal_entrypoint_available_only_under_new_joint_lease_and_outer_launch_authority",
        "writes_assessor_compatible_inner_receipt_last",
        "phase_accurate_terminal_refusal_receipt_supported",
        "future_outer_reaper_and_fresh_lease_required_before_entrypoint_may_run",
    ] {
        require_exact_bool(host_body, field, true, "joint host preflight.host_body")?;
    }
    for field in [
        "historical_l0_receipt_or_pinned_buffer_import_allowed",
        "l1_suffix_or_moe_authorized",
    ] {
        require_exact_bool(host_body, field, false, "joint host preflight.host_body")?;
    }
    let l0_root = object(
        &l0_source_outer_preflight.document,
        "source-token L0 outer preflight",
    )?;
    let l0_binding = field_object(l0_root, "source_binding", "source-token L0 outer preflight")?;
    let l0_manifest = field_object(
        l0_binding,
        "manifest",
        "source-token L0 outer preflight.source_binding",
    )?;
    let l0_admission = field_object(
        l0_binding,
        "admission_receipt",
        "source-token L0 outer preflight.source_binding",
    )?;
    if field_string(
        l0_manifest,
        "path",
        "source-token L0 outer preflight.source_binding.manifest",
    )? != manifest.file.path.to_string_lossy()
        || l0_manifest.get("bytes").and_then(Value::as_u64) != Some(manifest.file.bytes)
        || field_string(
            l0_manifest,
            "sha256",
            "source-token L0 outer preflight.source_binding.manifest",
        )? != manifest.file.sha256
        || field_string(
            l0_binding,
            "manifest_seal_sha256",
            "source-token L0 outer preflight.source_binding",
        )? != manifest.seal_sha256
        || field_string(
            l0_admission,
            "path",
            "source-token L0 outer preflight.source_binding.admission_receipt",
        )? != admission_receipt.file.path.to_string_lossy()
        || field_string(
            l0_admission,
            "sha256",
            "source-token L0 outer preflight.source_binding.admission_receipt",
        )? != admission_receipt.file.sha256
        || field_string(
            l0_binding,
            "admission_receipt_seal_sha256",
            "source-token L0 outer preflight.source_binding",
        )? != admission_receipt.seal_sha256
    {
        return Err("source-token L0 outer artifact/admission authority drifted".into());
    }
    let _ = source_route_from_l0_outer(&l0_source_outer_preflight.document)?;
    Ok((
        outer,
        static_plan,
        l0_source_outer_preflight,
        schedule,
        continuation,
        assessor_binding,
        manifest,
        admission_receipt,
    ))
}

#[cfg(target_os = "macos")]
fn validate_joint_lease_and_launch(
    args: &Args,
    outer_preflight: &SealedFile,
    host_binary: &FileEvidence,
) -> Result<(SealedFile, String, SealedFile), String> {
    let lease_path = args
        .lease_receipt
        .as_ref()
        .ok_or("metal mode lacks joint lease receipt")?;
    let lease = sealed_file(
        lease_path,
        "joint L0→L1 component lease",
        JOINT_LEASE_SCHEMA,
        JOINT_LEASE_STATUS,
    )?;
    let lease_root = object(&lease.document, "joint L0→L1 component lease")?;
    let lease_id = require_sha_field(lease_root, "lease_id", "joint L0→L1 component lease")?;
    binding_matches(
        field_object(lease_root, "outer_preflight", "joint L0→L1 component lease")?,
        outer_preflight,
        "joint L0→L1 component lease.outer_preflight",
    )?;
    let execution_path = PathBuf::from(field_string(
        field_object(
            lease_root,
            "execution_binding",
            "joint L0→L1 component lease",
        )?,
        "path",
        "joint L0→L1 component lease.execution_binding",
    )?);
    let execution = sealed_file(
        &execution_path,
        "joint strict host execution binding",
        EXECUTION_BINDING_SCHEMA,
        EXECUTION_BINDING_STATUS,
    )?;
    binding_matches(
        field_object(
            lease_root,
            "execution_binding",
            "joint L0→L1 component lease",
        )?,
        &execution,
        "joint L0→L1 component lease.execution_binding",
    )?;
    let execution_root = object(&execution.document, "joint strict host execution binding")?;
    for field in [
        "metal_entrypoint_available",
        "writes_assessor_compatible_inner_receipt",
        "outer_reaped_receipt_last_required",
        "non_timed_exact_32_dispatches_required",
    ] {
        require_exact_bool(
            execution_root,
            field,
            true,
            "joint strict host execution binding",
        )?;
    }
    let execution_host = field_object(
        execution_root,
        "host_binary",
        "joint strict host execution binding",
    )?;
    if field_string(
        execution_host,
        "sha256",
        "joint strict host execution binding.host_binary",
    )? != host_binary.sha256
    {
        return Err("joint strict host execution binding host SHA drifted".into());
    }
    let execution_outer = field_object(
        execution_root,
        "outer_preflight",
        "joint strict host execution binding",
    )?;
    if field_string(
        execution_outer,
        "document_sha256",
        "joint strict host execution binding.outer_preflight",
    )? != outer_preflight.document_sha256
        || field_string(
            execution_outer,
            "document_seal_sha256",
            "joint strict host execution binding.outer_preflight",
        )? != outer_preflight.seal_sha256
    {
        return Err("joint strict host execution binding outer preflight drifted".into());
    }
    if field_string(
        lease_root,
        "host_binary_sha256",
        "joint L0→L1 component lease",
    )? != host_binary.sha256
    {
        return Err("joint lease host SHA drifted".into());
    }
    let policy = field_object(
        lease_root,
        "execution_policy",
        "joint L0→L1 component lease",
    )?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("total_dispatches", JOINT_DISPATCHES),
    ] {
        require_u64(
            policy,
            field,
            expected,
            "joint L0→L1 component lease.execution_policy",
        )?;
    }
    for field in ["single_fence_required", "non_timed", "strict_math_required"] {
        require_exact_bool(
            policy,
            field,
            true,
            "joint L0→L1 component lease.execution_policy",
        )?;
    }
    if field_string(
        policy,
        "tcb_trace_mode",
        "joint L0→L1 component lease.execution_policy",
    )? != "off"
    {
        return Err("joint lease does not require TcbTraceMode::Off".into());
    }
    for field in [
        "l1_suffix_or_moe_allowed",
        "complete_layer_or_token_allowed",
        "server_hcli_tps_tg_or_tournament_allowed",
    ] {
        require_exact_bool(
            policy,
            field,
            false,
            "joint L0→L1 component lease.execution_policy",
        )?;
    }
    let lifecycle = field_object(lease_root, "lifecycle", "joint L0→L1 component lease")?;
    for field in [
        "fresh_for_exact_outer_preflight",
        "create_new_replay_reservation_required",
        "outer_reaped_capture_required",
        "terminal_receipt_written_last_required",
        "separate_actual_release_required",
        "automatic_retry_prohibited",
    ] {
        require_exact_bool(
            lifecycle,
            field,
            true,
            "joint L0→L1 component lease.lifecycle",
        )?;
    }
    let launch_path = args
        .outer_launch_authority
        .as_ref()
        .ok_or("metal mode lacks joint outer-launch authority")?;
    let launch = sealed_file(
        launch_path,
        "joint L0→L1 outer-launch authority",
        JOINT_OUTER_LAUNCH_SCHEMA,
        JOINT_OUTER_LAUNCH_STATUS,
    )?;
    let launch_root = object(&launch.document, "joint L0→L1 outer-launch authority")?;
    let launch_lease = field_object(launch_root, "lease", "joint L0→L1 outer-launch authority")?;
    binding_matches(
        launch_lease,
        &lease,
        "joint L0→L1 outer-launch authority.lease",
    )?;
    if field_string(
        launch_lease,
        "lease_id",
        "joint L0→L1 outer-launch authority.lease",
    )? != lease_id
    {
        return Err("joint outer-launch authority lease ID drifted".into());
    }
    binding_matches(
        field_object(
            launch_root,
            "outer_preflight",
            "joint L0→L1 outer-launch authority",
        )?,
        outer_preflight,
        "joint L0→L1 outer-launch authority.outer_preflight",
    )?;
    binding_matches(
        field_object(
            launch_root,
            "execution_binding",
            "joint L0→L1 outer-launch authority",
        )?,
        &execution,
        "joint L0→L1 outer-launch authority.execution_binding",
    )?;
    let outer_capture_dir = args
        .outer_capture_dir
        .as_ref()
        .ok_or("metal mode lacks outer capture directory")?;
    let capture_dir = args
        .capture_dir
        .as_ref()
        .ok_or("metal mode lacks inner capture directory")?;
    if launch_root
        .get("planned_outer_capture_dir")
        .and_then(Value::as_str)
        != Some(outer_capture_dir.to_string_lossy().as_ref())
        || launch_root
            .get("planned_inner_capture_dir")
            .and_then(Value::as_str)
            != Some(capture_dir.to_string_lossy().as_ref())
        || launch_root.get("workers").and_then(Value::as_u64) != Some(args.workers as u64)
    {
        return Err("joint outer-launch authority capture path/worker drifted".into());
    }
    require_sha_field(
        launch_root,
        "launch_identity_sha256",
        "joint L0→L1 outer-launch authority",
    )?;
    let launch_policy = field_object(
        launch_root,
        "execution_policy",
        "joint L0→L1 outer-launch authority",
    )?;
    for field in ["strict_math", "non_timed", "single_fence_required"] {
        require_exact_bool(
            launch_policy,
            field,
            true,
            "joint L0→L1 outer-launch authority.execution_policy",
        )?;
    }
    require_u64(
        launch_policy,
        "total_dispatches",
        JOINT_DISPATCHES,
        "joint L0→L1 outer-launch authority.execution_policy",
    )?;
    let launch_lifecycle = field_object(
        launch_root,
        "lifecycle",
        "joint L0→L1 outer-launch authority",
    )?;
    for field in [
        "outer_reaped_capture_required",
        "terminal_receipt_written_last",
        "automatic_retry_prohibited",
    ] {
        require_exact_bool(
            launch_lifecycle,
            field,
            true,
            "joint L0→L1 outer-launch authority.lifecycle",
        )?;
    }
    let boundary = field_object(
        launch_root,
        "claim_boundary",
        "joint L0→L1 outer-launch authority",
    )?;
    require_exact_bool(
        boundary,
        "component_only",
        true,
        "joint L0→L1 outer-launch authority.claim_boundary",
    )?;
    for field in [
        "l1_suffix_or_moe_authorized",
        "complete_layer_or_token_authorized",
    ] {
        require_exact_bool(
            boundary,
            field,
            false,
            "joint L0→L1 outer-launch authority.claim_boundary",
        )?;
    }
    Ok((lease, lease_id, launch))
}

#[cfg(target_os = "macos")]
fn validate_joint_capture_authority(args: &Args) -> Result<JointCaptureAuthority, String> {
    if args.mode != Mode::Metal {
        return Err("joint capture authority may only be validated in metal mode".into());
    }
    let host_binary = current_binary_evidence()?;
    let outer_path = args
        .joint_outer_preflight
        .as_ref()
        .ok_or("metal mode lacks joint outer preflight")?;
    let (
        outer_preflight,
        joint_static_plan,
        l0_source_outer_preflight,
        schedule,
        continuation,
        assessor_binding,
        manifest,
        admission_receipt,
    ) = validate_joint_outer_preflight(outer_path, &host_binary)?;
    let (lease, lease_id, outer_launch) =
        validate_joint_lease_and_launch(args, &outer_preflight, &host_binary)?;
    Ok(JointCaptureAuthority {
        outer_preflight,
        joint_static_plan,
        l0_source_outer_preflight,
        schedule,
        continuation,
        assessor_binding,
        manifest,
        admission_receipt,
        lease,
        lease_id,
        outer_launch,
        host_binary,
        workers: args.workers,
    })
}

#[cfg(target_os = "macos")]
fn require_non_timed_tcb_trace_off() -> Result<(), String> {
    match env::var("HAWKING_TCB_TRACE") {
        Err(_) => Ok(()),
        Ok(value) if value.is_empty() || value == "0" => Ok(()),
        Ok(value) => Err(format!(
            "joint capture refuses HAWKING_TCB_TRACE={value:?}; exact one-TCB non-timed capture requires it unset or 0"
        )),
    }
}

#[cfg(target_os = "macos")]
fn source_artifact_admission(
    authority: &JointCaptureAuthority,
) -> Result<CompleteBinaryAdmission, String> {
    source_artifact_admission_from_documents(&authority.manifest, &authority.admission_receipt)
}

#[cfg(target_os = "macos")]
fn source_artifact_admission_from_documents(
    manifest_file: &SealedFile,
    admission_file: &SealedFile,
) -> Result<CompleteBinaryAdmission, String> {
    let manifest = object(&manifest_file.document, "joint capture manifest")?;
    let manifest_seal = verify_seal(&manifest_file.document, "joint capture manifest")?;
    if manifest_seal != manifest_file.seal_sha256 {
        return Err("joint capture manifest seal drifted after authority validation".into());
    }
    let admission_root = object(
        &admission_file.document,
        "joint capture immutable admission receipt",
    )?;
    if verify_seal(
        &admission_file.document,
        "joint capture immutable admission receipt",
    )? != admission_file.seal_sha256
    {
        return Err("joint capture immutable admission receipt seal drifted".into());
    }
    let admitted_manifest = field_object(
        admission_root,
        "complete_manifest",
        "joint capture immutable admission receipt",
    )?;
    if !immutable_admission_manifest_matches(admitted_manifest, manifest_file)? {
        return Err("joint capture admission receipt does not bind the exact manifest".into());
    }
    let source_audit_seal = require_sha_field(
        manifest,
        "source_body_audit_seal_sha256",
        "joint capture manifest",
    )?;
    let revalidation_path = PathBuf::from(field_string(
        manifest,
        "source_revalidation_receipt_path",
        "joint capture manifest",
    )?);
    let expected_revalidation_seal = require_sha_field(
        manifest,
        "source_revalidation_receipt_seal_sha256",
        "joint capture manifest",
    )?;
    let (_, revalidation) = read_json(&revalidation_path, "joint capture source revalidation")?;
    if verify_seal(&revalidation, "joint capture source revalidation")?
        != expected_revalidation_seal
    {
        return Err("joint capture source revalidation seal drifted".into());
    }
    let revalidation_root = object(&revalidation, "joint capture source revalidation")?;
    if require_sha_field(
        revalidation_root,
        "source_audit_seal_sha256",
        "joint capture source revalidation",
    )? != source_audit_seal
    {
        return Err("joint capture source audit seal drifted".into());
    }
    let source_revision = field_string(
        revalidation_root,
        "source_revision",
        "joint capture source revalidation",
    )?
    .to_owned();
    if source_revision.len() != 40
        || !source_revision.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
    {
        return Err("joint capture source revision must remain a lowercase git SHA-1".into());
    }
    Ok(CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: manifest_file.seal_sha256.clone(),
        expected_source_audit_seal_sha256: source_audit_seal,
        expected_source_revision: source_revision,
    })
}

/// Validate the complete, immutable source-admission chain without creating a
/// Metal context, a command buffer, a lease, or a capture directory.  This is
/// intentionally separate from the outer-reaped `metal` mode so an authority
/// identity error can be found before a one-shot device lease is issued.
#[cfg(target_os = "macos")]
fn run_source_admission_preflight(args: &Args) -> Result<(), String> {
    if args.mode != Mode::SourceAdmissionPreflight {
        return Err("source-admission preflight may only run in its dedicated mode".into());
    }
    let host_binary = current_binary_evidence()?;
    let outer_path = args
        .joint_outer_preflight
        .as_ref()
        .ok_or("source-admission preflight lacks joint outer preflight")?;
    let (
        _outer,
        _static_plan,
        _l0_source_outer,
        _schedule,
        _continuation,
        _assessor_binding,
        manifest,
        admission,
    ) = validate_joint_outer_preflight(outer_path, &host_binary)?;
    let _validated = source_artifact_admission_from_documents(&manifest, &admission)?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn derived_identity(label: &str, fields: &[&str]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(label.as_bytes());
    for field in fields {
        hasher.update([0]);
        hasher.update(field.as_bytes());
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(target_os = "macos")]
fn bytes_from_elements(elements: usize, label: &str) -> Result<u64, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte count overflows"))?;
    u64::try_from(bytes).map_err(|_| format!("{label} byte count exceeds u64"))
}

#[cfg(target_os = "macos")]
fn finite_component_error(value: f32, label: &str) -> Result<f64, String> {
    if !value.is_finite() || value < 0.0 || value > 1.0e-3 {
        return Err(format!("{label} parity error is invalid: {value}"));
    }
    Ok(f64::from(value))
}

#[cfg(target_os = "macos")]
fn parity_record(cpu: &str, device: &str, error: f32, label: &str) -> Result<Value, String> {
    if !is_sha256(cpu) || !is_sha256(device) {
        return Err(format!("{label} parity hashes are invalid"));
    }
    Ok(json!({
        "passed": true,
        "cpu_f32le_sha256": cpu,
        "device_f32le_sha256": device,
        "max_abs_error": finite_component_error(error, label)?,
    }))
}

#[cfg(target_os = "macos")]
fn state_record(
    slot: u64,
    offset_bytes: u64,
    capacity_bytes: u64,
    identity: &str,
    f32le_sha256: &str,
    error: f32,
    label: &str,
) -> Result<Value, String> {
    if !is_sha256(identity) || !is_sha256(f32le_sha256) {
        return Err(format!("{label} state hashes are invalid"));
    }
    Ok(json!({
        "passed": true,
        "slot": slot,
        "offset_bytes": offset_bytes,
        "capacity_bytes": capacity_bytes,
        "device_buffer_identity_sha256": identity,
        "f32le_sha256": f32le_sha256,
        "max_abs_error": finite_component_error(error, label)?,
    }))
}

#[cfg(target_os = "macos")]
fn phase_document(phase: &MetalExecutionPhase) -> Value {
    let device_dispatch_may_have_occurred = if phase.command_fence_succeeded {
        Value::Bool(true)
    } else if phase.command_commit_may_have_been_attempted {
        // The consuming finalizer owns the commit.  If it returns an error,
        // we cannot honestly assert that no submission reached Metal.
        Value::Null
    } else {
        Value::Bool(false)
    };
    json!({
        "strict_artifact_admission_started": phase.strict_artifact_admission_started,
        "strict_artifact_admission_succeeded": phase.strict_artifact_admission_succeeded,
        "metal_context_construction_attempted": phase.metal_context_construction_attempted,
        "metal_context_constructed": phase.metal_context_constructed,
        "structural_kernel_trace_enabled": phase.structural_kernel_trace_enabled,
        "dispatches_encoded": phase.dispatches_encoded,
        "encoded_kernel_names": phase.encoded_kernel_names,
        "command_commit_may_have_been_attempted": phase.command_commit_may_have_been_attempted,
        "command_fence_succeeded": phase.command_fence_succeeded,
        "readback_started": phase.readback_started,
        "device_dispatch_may_have_occurred": device_dispatch_may_have_occurred,
    })
}

#[cfg(target_os = "macos")]
fn require_exact_joint_parity(parity: &Qwen80SameRuntimeL0L1PrefixParity) -> Result<(), String> {
    if parity.l0_dispatches != L0_DISPATCHES as usize
        || parity.l1_prefix_dispatches != L1_PREFIX_DISPATCHES as usize
        || parity.total_dispatches != JOINT_DISPATCHES as usize
        || !parity.same_runtime_same_command_buffer_required
        || !parity.single_fence_after_l0_and_l1_prefix_required
        || parity.l1_suffix_or_moe_executed
        || parity
            .structural_kernel_names
            .iter()
            .map(String::as_str)
            .ne(L0_KERNELS.iter().chain(L1_KERNELS.iter()).copied())
    {
        return Err("joint host parity drifted from exact non-timed 23+9 scope".into());
    }
    if parity.l0_first_residual.source_token_id != SOURCE_TOKEN_ID as u32
        || parity.l0_first_residual.layer != 0
        || parity.l0_first_residual.linear_state_slot != 0
        || parity.l0_first_residual.first_residual_elements != 2_048
        || parity.l0_first_residual.first_residual_bytes != 8_192
        || parity.l0_first_residual.linear_conv_state_bytes != 98_304
        || parity.l0_first_residual.linear_recurrent_state_bytes != 2_097_152
        || parity.l1_prefix.source_token_id != SOURCE_TOKEN_ID as u32
        || parity.l1_prefix.layer != 1
        || parity.l1_prefix.linear_state_slot != 1
        || parity.l1_prefix.first_residual_elements != 2_048
        || parity.l1_prefix.first_residual_bytes != 8_192
        || bytes_from_elements(
            parity.l1_prefix.conv_state_offset_elements,
            "joint L1 convolution offset",
        )? != 98_304
        || bytes_from_elements(
            parity.l1_prefix.conv_state_capacity_elements,
            "joint L1 convolution capacity",
        )? != 196_608
        || bytes_from_elements(
            parity.l1_prefix.recurrent_state_offset_elements,
            "joint L1 recurrent offset",
        )? != 2_097_152
        || bytes_from_elements(
            parity.l1_prefix.recurrent_state_capacity_elements,
            "joint L1 recurrent capacity",
        )? != 4_194_304
    {
        return Err("joint host parity state/output geometry drifted".into());
    }
    for error in [
        parity.l0_first_residual.first_residual_max_abs_error,
        parity.l0_first_residual.conv_state_max_abs_error,
        parity.l0_first_residual.recurrent_state_max_abs_error,
        parity.l0_true_moe_suffix.postnorm_max_abs_error,
        parity.l0_true_moe_suffix.router_logits_max_abs_error,
        parity.l0_true_moe_suffix.route_weights_max_abs_error,
        parity.l0_true_moe_suffix.shared_max_abs_error,
        parity.l0_true_moe_suffix.routed_sum_max_abs_error,
        parity.l0_true_moe_suffix.second_residual_max_abs_error,
        parity.l0_second_residual_max_abs_error,
        parity.l1_prefix.input_max_abs_error,
        parity.l1_prefix.first_residual_max_abs_error,
        parity.l1_prefix.conv_state_max_abs_error,
        parity.l1_prefix.recurrent_state_max_abs_error,
    ] {
        let _ = finite_component_error(error, "joint host parity")?;
    }
    if parity.l0_true_moe_suffix.route_guard != 1
        || parity.l0_route_ids.len() != 10
        || parity.l0_route_weights.len() != 10
        || parity.l0_true_moe_suffix.all_ten_route_witnesses.len() != 10
    {
        return Err("joint host parity route guard/witness count drifted".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn build_success_inner_receipt(
    authority: &JointCaptureAuthority,
    capture: &Path,
    parity: &Qwen80SameRuntimeL0L1PrefixParity,
    phase: &MetalExecutionPhase,
) -> Result<Value, String> {
    require_exact_joint_parity(parity)?;
    let (outer_ids, outer_weights) =
        source_route_from_l0_outer(&authority.l0_source_outer_preflight.document)?;
    let route_ids = parity
        .l0_route_ids
        .iter()
        .copied()
        .map(u32::from)
        .collect::<Vec<_>>();
    if route_ids != outer_ids {
        return Err(
            "joint host fresh L0 route IDs differ from source-token outer authority".into(),
        );
    }
    let route_weights = parity
        .l0_route_weights
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    if route_weights
        .iter()
        .zip(&outer_weights)
        .any(|(actual, expected)| (actual - expected).abs() > 1.0e-6)
    {
        return Err(
            "joint host fresh L0 route weights differ from source-token outer authority".into(),
        );
    }
    let l0 = &parity.l0_first_residual;
    let suffix = &parity.l0_true_moe_suffix;
    let l1 = &parity.l1_prefix;
    let runtime_identity = derived_identity(
        "qwen80-joint-runtime",
        &[
            &l0.active_conv_state_buffer_identity_sha256,
            &l0.active_recurrent_state_buffer_identity_sha256,
            &l1.active_conv_state_buffer_identity_sha256,
            &l1.active_recurrent_state_buffer_identity_sha256,
        ],
    );
    let arena_identity = derived_identity(
        "qwen80-joint-state-arena",
        &[
            &l0.active_conv_state_buffer_identity_sha256,
            &l0.active_recurrent_state_buffer_identity_sha256,
        ],
    );
    let tcb_identity = derived_identity(
        "qwen80-joint-tcb",
        &[
            &runtime_identity,
            &parity.l0_second_residual_buffer_identity_sha256,
            &l1.first_residual_buffer_identity_sha256,
            &parity.structural_kernel_names.join("|"),
        ],
    );
    let capability_identity = derived_identity(
        "qwen80-joint-opaque-l0-capability",
        &[
            &runtime_identity,
            &arena_identity,
            &tcb_identity,
            &parity.l0_second_residual_buffer_identity_sha256,
        ],
    );
    let session_identity = derived_identity(
        "qwen80-joint-fresh-session",
        &[
            &authority.lease_id,
            &authority.outer_launch.seal_sha256,
            &runtime_identity,
        ],
    );
    let route_witnesses = suffix
        .all_ten_route_witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| -> Result<Value, String> {
            if witness.wave_index != index
                || witness.expert_id != usize::from(parity.l0_route_ids[index])
                || !is_sha256(&witness.cpu_output_f32le_sha256)
                || !is_sha256(&witness.device_output_f32le_sha256)
            {
                return Err(format!("joint host route witness {index} drifted"));
            }
            Ok(json!({
                "wave_index": index,
                "expert_id": witness.expert_id,
                "passed": true,
                "f32le_sha256": witness.device_output_f32le_sha256,
                "cpu_f32le_sha256": witness.cpu_output_f32le_sha256,
                "max_abs_error": finite_component_error(witness.max_abs_error, "joint route witness")?,
            }))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let fresh_readbacks = json!({
        "l0_suffix": {
            "route_guard": {
                "passed": true,
                "value": suffix.route_guard,
                "expected_route_ids": route_ids,
                "observed_route_ids": suffix.observed_route_ids,
                "expected_route_weights": route_weights,
                "observed_route_weights": suffix.observed_route_weights,
                "weights_max_abs_error": finite_component_error(suffix.route_weights_max_abs_error, "joint route weights")?,
            },
            "postnorm": parity_record(&suffix.postnorm_cpu_f32le_sha256, &suffix.postnorm_output_f32le_sha256, suffix.postnorm_max_abs_error, "joint postnorm")?,
            "router_logits": parity_record(&suffix.router_logits_cpu_f32le_sha256, &suffix.router_logits_output_f32le_sha256, suffix.router_logits_max_abs_error, "joint router logits")?,
            "all_ten_weighted_route_witnesses": route_witnesses,
            "shared_output": parity_record(&suffix.shared_cpu_f32le_sha256, &suffix.shared_output_f32le_sha256, suffix.shared_max_abs_error, "joint shared output")?,
            "routed_sum": parity_record(&suffix.routed_sum_cpu_f32le_sha256, &suffix.routed_sum_output_f32le_sha256, suffix.routed_sum_max_abs_error, "joint routed sum")?,
            "second_residual": parity_record(&suffix.second_residual_cpu_f32le_sha256, &suffix.second_residual_output_f32le_sha256, suffix.second_residual_max_abs_error, "joint L0 second residual")?,
        },
        "fresh_l0_state": {
            "active_conv": state_record(0, 0, 98_304, &l0.active_conv_state_buffer_identity_sha256, &l0.device_post_conv_state_f32le_sha256, l0.conv_state_max_abs_error, "joint L0 active convolution state")?,
            "active_recurrent": state_record(0, 0, 2_097_152, &l0.active_recurrent_state_buffer_identity_sha256, &l0.device_post_recurrent_state_f32le_sha256, l0.recurrent_state_max_abs_error, "joint L0 active recurrent state")?,
            "rollback_conv": state_record(0, 0, 98_304, &l0.rollback_conv_state_buffer_identity_sha256, &l0.rollback_conv_state_f32le_sha256, 0.0, "joint L0 rollback convolution state")?,
            "rollback_recurrent": state_record(0, 0, 2_097_152, &l0.rollback_recurrent_state_buffer_identity_sha256, &l0.rollback_recurrent_state_f32le_sha256, 0.0, "joint L0 rollback recurrent state")?,
        },
        "fresh_l1_slot1": {
            "layer": 1,
            "linear_state_slot": 1,
            "output_elements": 2_048,
            "output_bytes": 8_192,
            "input": parity_record(&l1.input_f32le_sha256, &l1.device_input_f32le_sha256, l1.input_max_abs_error, "joint L1 input")?,
            "first_residual_output": parity_record(&l1.cpu_first_residual_f32le_sha256, &l1.device_first_residual_f32le_sha256, l1.first_residual_max_abs_error, "joint L1 first residual")?,
            "active_conv": state_record(1, 98_304, 196_608, &l1.active_conv_state_buffer_identity_sha256, &l1.device_post_conv_state_f32le_sha256, l1.conv_state_max_abs_error, "joint L1 active convolution state")?,
            "active_recurrent": state_record(1, 2_097_152, 4_194_304, &l1.active_recurrent_state_buffer_identity_sha256, &l1.device_post_recurrent_state_f32le_sha256, l1.recurrent_state_max_abs_error, "joint L1 active recurrent state")?,
            "rollback_conv": state_record(1, 98_304, 196_608, &l1.rollback_conv_state_buffer_identity_sha256, &l1.rollback_conv_state_f32le_sha256, 0.0, "joint L1 rollback convolution state")?,
            "rollback_recurrent": state_record(1, 2_097_152, 4_194_304, &l1.rollback_recurrent_state_buffer_identity_sha256, &l1.rollback_recurrent_state_f32le_sha256, 0.0, "joint L1 rollback recurrent state")?,
        },
    });
    let mut receipt = json!({
        "schema": JOINT_INNER_SCHEMA,
        "status": JOINT_INNER_STATUS,
        "fixture_or_synthetic": false,
        "self_asserted": false,
        "issuer": {"role": "joint_component_capture_child", "issuer_identity_sha256": derived_identity("qwen80-joint-child", &[&authority.host_binary.sha256, &authority.lease_id, &authority.outer_launch.seal_sha256])},
        "upstream_authorities": {
            "schedule_wrapper": {"present": true, "document_sha256": authority.schedule.document_sha256, "document_seal_sha256": authority.schedule.seal_sha256},
            "continuation": {"present": true, "document_sha256": authority.continuation.document_sha256, "document_seal_sha256": authority.continuation.seal_sha256},
            "assessor_binding": {"present": true, "document_sha256": authority.assessor_binding.document_sha256, "document_seal_sha256": authority.assessor_binding.seal_sha256},
        },
        "opaque_l0_continuation": {
            "factory": "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation",
            "l1_encoder": "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into",
            "consuming_finalizer": "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence",
            "opaque": true,
            "freshly_derived_from_l0_23_dispatch_graph": true,
            "same_runtime_state_arena_bound": true,
            "same_command_buffer_bound": true,
            "non_transferable_across_processes": true,
            "raw_pinned_buffer_or_dispatch_count_input_accepted": false,
            "capability_identity_sha256": capability_identity,
            "runtime_identity_sha256": runtime_identity,
            "runtime_state_arena_identity_sha256": arena_identity,
            "command_buffer_identity_sha256": tcb_identity,
        },
        "fresh_joint_execution": {
            "fresh_runtime": true,
            "fresh_session": true,
            "same_runtime": true,
            "same_tcb": true,
            "structural_trace_non_timed": true,
            "route_guard_enforced_before_l1": true,
            "runtime_identity_sha256": runtime_identity,
            "session_identity_sha256": session_identity,
            "tcb_identity_sha256": tcb_identity,
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "total_dispatches": JOINT_DISPATCHES,
            "fence_count": 1,
        },
        "structural_kernel_trace": {"non_timed": true, "exact_order": true, "kernel_names": parity.structural_kernel_names},
        "single_fence": {"consuming_finalizer": "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence", "only_command_buffer_consumed": true, "fence_succeeded": true, "readbacks_after_fence": true, "append_after_fence_possible": false, "fence_count": 1},
        "fresh_readbacks": fresh_readbacks,
        "outer_launch_authority_binding": sealed_binding_json(&authority.outer_launch),
        "joint_outer_preflight_binding": sealed_binding_json(&authority.outer_preflight),
        "joint_static_plan_binding": sealed_binding_json(&authority.joint_static_plan),
        "artifact_binding": {
            "manifest": sealed_binding_json(&authority.manifest),
            "admission_receipt": sealed_binding_json(&authority.admission_receipt),
        },
        "joint_lease_binding": {"lease_id": authority.lease_id, "receipt": sealed_binding_json(&authority.lease)},
        "execution_phase": phase_document(phase),
        "durable_capture": {"capture_directory": capture, "receipt_written_last_is_completion_marker": true, "outer_reaped_capture_required": true, "replay_guarded": true},
        "claim_boundary": {"component_only": true, "l1_suffix_or_moe_executed": false, "complete_layer_executed": false, "token_generated": false, "decoder_started": false, "server_or_watcher_started": false},
    });
    validate_success_inner_receipt(&receipt, authority)?;
    seal(&mut receipt)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn require_inner_parity(value: &Value, label: &str) -> Result<(), String> {
    let record = object(value, label)?;
    require_bool(record, "passed", true, label)?;
    for field in ["cpu_f32le_sha256", "device_f32le_sha256"] {
        let _ = require_sha_field(record, field, label)?;
    }
    let error = record
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|error| error.is_finite() && *error >= 0.0 && *error <= 1.0e-3)
        .ok_or_else(|| format!("{label}.max_abs_error must be a finite component tolerance"))?;
    if !error.is_finite() {
        return Err(format!("{label}.max_abs_error is not finite"));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn require_inner_route_witness(
    value: &Value,
    index: usize,
    expected_expert: u32,
) -> Result<(), String> {
    let label = format!("joint inner route witness {index}");
    let record = object(value, &label)?;
    require_u64(record, "wave_index", index as u64, &label)?;
    require_u64(record, "expert_id", u64::from(expected_expert), &label)?;
    require_bool(record, "passed", true, &label)?;
    let _ = require_sha_field(record, "f32le_sha256", &label)?;
    let _ = require_sha_field(record, "cpu_f32le_sha256", &label)?;
    let error = record
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|error| error.is_finite() && *error >= 0.0 && *error <= 1.0e-3)
        .ok_or_else(|| format!("{label}.max_abs_error must be a finite component tolerance"))?;
    if !error.is_finite() {
        return Err(format!("{label}.max_abs_error is not finite"));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn require_inner_state(
    value: &Value,
    label: &str,
    slot: u64,
    offset_bytes: u64,
    capacity_bytes: u64,
) -> Result<(), String> {
    let record = object(value, label)?;
    require_bool(record, "passed", true, label)?;
    require_u64(record, "slot", slot, label)?;
    require_u64(record, "offset_bytes", offset_bytes, label)?;
    require_u64(record, "capacity_bytes", capacity_bytes, label)?;
    let _ = require_sha_field(record, "device_buffer_identity_sha256", label)?;
    let _ = require_sha_field(record, "f32le_sha256", label)?;
    let error = record
        .get("max_abs_error")
        .and_then(Value::as_f64)
        .filter(|error| error.is_finite() && *error >= 0.0 && *error <= 1.0e-3)
        .ok_or_else(|| format!("{label}.max_abs_error must be a finite component tolerance"))?;
    if !error.is_finite() {
        return Err(format!("{label}.max_abs_error is not finite"));
    }
    Ok(())
}

/// Validate the complete inner receipt before it is sealed.  This makes a
/// success status impossible unless the same opaque capability/runtime/TCB,
/// exact trace, fresh route witnesses, and post-fence state/output evidence
/// are all present and internally consistent.  The independent outer reaper
/// repeats this validation after the child exits; the two validators are
/// intentionally redundant across the Rust/Python boundary.
#[cfg(target_os = "macos")]
fn validate_success_inner_receipt(
    value: &Value,
    authority: &JointCaptureAuthority,
) -> Result<(), String> {
    let root = object(value, "joint success inner receipt")?;
    if field_string(root, "schema", "joint success inner receipt")? != JOINT_INNER_SCHEMA
        || field_string(root, "status", "joint success inner receipt")? != JOINT_INNER_STATUS
    {
        return Err("joint success inner receipt schema/status drifted".into());
    }
    require_bool(
        root,
        "fixture_or_synthetic",
        false,
        "joint success inner receipt",
    )?;
    require_bool(root, "self_asserted", false, "joint success inner receipt")?;
    let issuer = field_object(root, "issuer", "joint success inner receipt")?;
    if field_string(issuer, "role", "joint success inner receipt.issuer")?
        != "joint_component_capture_child"
    {
        return Err("joint success inner receipt issuer role drifted".into());
    }
    let _ = require_sha_field(
        issuer,
        "issuer_identity_sha256",
        "joint success inner receipt.issuer",
    )?;

    let upstream = field_object(root, "upstream_authorities", "joint success inner receipt")?;
    for (field, expected) in [
        ("schedule_wrapper", &authority.schedule),
        ("continuation", &authority.continuation),
        ("assessor_binding", &authority.assessor_binding),
    ] {
        let binding = field_object(
            upstream,
            field,
            "joint success inner receipt.upstream_authorities",
        )?;
        require_bool(
            binding,
            "present",
            true,
            "joint success inner receipt upstream",
        )?;
        if field_string(
            binding,
            "document_sha256",
            "joint success inner receipt upstream",
        )? != expected.document_sha256
            || field_string(
                binding,
                "document_seal_sha256",
                "joint success inner receipt upstream",
            )? != expected.seal_sha256
        {
            return Err(format!(
                "joint success inner receipt upstream {field} drifted"
            ));
        }
    }

    let capability = field_object(
        root,
        "opaque_l0_continuation",
        "joint success inner receipt",
    )?;
    if field_string(capability, "factory", "joint opaque continuation")?
        != "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation"
        || field_string(capability, "l1_encoder", "joint opaque continuation")?
            != "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into"
        || field_string(capability, "consuming_finalizer", "joint opaque continuation")?
            != "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence"
    {
        return Err("joint success inner receipt opaque continuation ABI drifted".into());
    }
    for field in [
        "opaque",
        "freshly_derived_from_l0_23_dispatch_graph",
        "same_runtime_state_arena_bound",
        "same_command_buffer_bound",
        "non_transferable_across_processes",
    ] {
        require_bool(capability, field, true, "joint opaque continuation")?;
    }
    require_bool(
        capability,
        "raw_pinned_buffer_or_dispatch_count_input_accepted",
        false,
        "joint opaque continuation",
    )?;
    let capability_runtime = require_sha_field(
        capability,
        "runtime_identity_sha256",
        "joint opaque continuation",
    )?;
    let capability_tcb = require_sha_field(
        capability,
        "command_buffer_identity_sha256",
        "joint opaque continuation",
    )?;
    for field in [
        "capability_identity_sha256",
        "runtime_state_arena_identity_sha256",
    ] {
        let _ = require_sha_field(capability, field, "joint opaque continuation")?;
    }

    let execution = field_object(root, "fresh_joint_execution", "joint success inner receipt")?;
    for field in [
        "fresh_runtime",
        "fresh_session",
        "same_runtime",
        "same_tcb",
        "structural_trace_non_timed",
        "route_guard_enforced_before_l1",
    ] {
        require_bool(execution, field, true, "joint fresh execution")?;
    }
    let execution_runtime = require_sha_field(
        execution,
        "runtime_identity_sha256",
        "joint fresh execution",
    )?;
    let execution_tcb =
        require_sha_field(execution, "tcb_identity_sha256", "joint fresh execution")?;
    let _ = require_sha_field(
        execution,
        "session_identity_sha256",
        "joint fresh execution",
    )?;
    if capability_runtime != execution_runtime || capability_tcb != execution_tcb {
        return Err(
            "joint opaque continuation runtime/TCB identity differs from fresh execution".into(),
        );
    }
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("l0_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("total_dispatches", JOINT_DISPATCHES),
        ("fence_count", 1),
    ] {
        require_u64(execution, field, expected, "joint fresh execution")?;
    }

    let trace = field_object(
        root,
        "structural_kernel_trace",
        "joint success inner receipt",
    )?;
    require_bool(trace, "non_timed", true, "joint structural trace")?;
    require_bool(trace, "exact_order", true, "joint structural trace")?;
    let expected_names = Value::Array(
        L0_KERNELS
            .iter()
            .chain(L1_KERNELS.iter())
            .map(|kernel| Value::String((*kernel).to_owned()))
            .collect(),
    );
    if trace.get("kernel_names") != Some(&expected_names) {
        return Err("joint success inner receipt structural kernel trace drifted".into());
    }
    let fence = field_object(root, "single_fence", "joint success inner receipt")?;
    if field_string(fence, "consuming_finalizer", "joint single fence")?
        != "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_joint_fence"
    {
        return Err("joint success inner receipt finalizer drifted".into());
    }
    for (field, expected) in [
        ("only_command_buffer_consumed", true),
        ("fence_succeeded", true),
        ("readbacks_after_fence", true),
        ("append_after_fence_possible", false),
    ] {
        require_bool(fence, field, expected, "joint single fence")?;
    }
    require_u64(fence, "fence_count", 1, "joint single fence")?;

    let (route_ids, route_weights) =
        source_route_from_l0_outer(&authority.l0_source_outer_preflight.document)?;
    let readbacks = field_object(root, "fresh_readbacks", "joint success inner receipt")?;
    let l0_suffix = field_object(readbacks, "l0_suffix", "joint fresh readbacks")?;
    let guard = field_object(l0_suffix, "route_guard", "joint L0 route guard")?;
    require_bool(guard, "passed", true, "joint L0 route guard")?;
    require_u64(guard, "value", 1, "joint L0 route guard")?;
    let parse_ids = |field: &str| -> Result<Vec<u32>, String> {
        guard
            .get(field)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("joint L0 route guard.{field} must be an array"))?
            .iter()
            .map(|item| {
                item.as_u64()
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| format!("joint L0 route guard.{field} contains an invalid ID"))
            })
            .collect()
    };
    if parse_ids("expected_route_ids")? != route_ids
        || parse_ids("observed_route_ids")? != route_ids
    {
        return Err("joint success inner receipt source route IDs drifted".into());
    }
    for field in ["expected_route_weights", "observed_route_weights"] {
        let observed = guard
            .get(field)
            .and_then(Value::as_array)
            .ok_or_else(|| format!("joint L0 route guard.{field} must be an array"))?;
        if observed.len() != route_weights.len()
            || observed
                .iter()
                .zip(&route_weights)
                .any(|(actual, expected)| {
                    actual
                        .as_f64()
                        .filter(|actual| actual.is_finite() && (actual - expected).abs() <= 1.0e-6)
                        .is_none()
                })
        {
            return Err("joint success inner receipt source route weights drifted".into());
        }
    }
    let weight_error = guard
        .get("weights_max_abs_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0e-3)
        .ok_or("joint L0 route guard weights_max_abs_error is invalid")?;
    if !weight_error.is_finite() {
        return Err("joint L0 route guard weights_max_abs_error is invalid".into());
    }
    for field in [
        "postnorm",
        "router_logits",
        "shared_output",
        "routed_sum",
        "second_residual",
    ] {
        require_inner_parity(
            l0_suffix
                .get(field)
                .ok_or_else(|| format!("joint L0 suffix lacks {field}"))?,
            &format!("joint L0 suffix {field}"),
        )?;
    }
    let witnesses = l0_suffix
        .get("all_ten_weighted_route_witnesses")
        .and_then(Value::as_array)
        .ok_or("joint L0 suffix route witnesses must be an array")?;
    if witnesses.len() != route_ids.len() {
        return Err("joint L0 suffix does not retain exactly ten route witnesses".into());
    }
    for (index, witness) in witnesses.iter().enumerate() {
        require_inner_route_witness(witness, index, route_ids[index])?;
    }
    let l0_state = field_object(readbacks, "fresh_l0_state", "joint fresh readbacks")?;
    for (field, capacity) in [
        ("active_conv", 98_304),
        ("rollback_conv", 98_304),
        ("active_recurrent", 2_097_152),
        ("rollback_recurrent", 2_097_152),
    ] {
        require_inner_state(
            l0_state
                .get(field)
                .ok_or_else(|| format!("joint L0 state lacks {field}"))?,
            &format!("joint L0 state {field}"),
            0,
            0,
            capacity,
        )?;
    }
    let l1 = field_object(readbacks, "fresh_l1_slot1", "joint fresh readbacks")?;
    for (field, expected) in [
        ("layer", 1),
        ("linear_state_slot", 1),
        ("output_elements", 2_048),
        ("output_bytes", 8_192),
    ] {
        require_u64(l1, field, expected, "joint L1 slot-one readback")?;
    }
    require_inner_parity(
        l1.get("input")
            .ok_or("joint L1 slot-one lacks input parity")?,
        "joint L1 slot-one input",
    )?;
    require_inner_parity(
        l1.get("first_residual_output")
            .ok_or("joint L1 slot-one lacks output parity")?,
        "joint L1 slot-one first residual",
    )?;
    for (field, offset, capacity) in [
        ("active_conv", 98_304, 196_608),
        ("rollback_conv", 98_304, 196_608),
        ("active_recurrent", 2_097_152, 4_194_304),
        ("rollback_recurrent", 2_097_152, 4_194_304),
    ] {
        require_inner_state(
            l1.get(field)
                .ok_or_else(|| format!("joint L1 slot-one lacks {field}"))?,
            &format!("joint L1 state {field}"),
            1,
            offset,
            capacity,
        )?;
    }

    binding_matches(
        field_object(
            root,
            "outer_launch_authority_binding",
            "joint success inner receipt",
        )?,
        &authority.outer_launch,
        "joint success inner receipt.outer_launch_authority_binding",
    )?;
    binding_matches(
        field_object(
            root,
            "joint_outer_preflight_binding",
            "joint success inner receipt",
        )?,
        &authority.outer_preflight,
        "joint success inner receipt.joint_outer_preflight_binding",
    )?;
    binding_matches(
        field_object(
            root,
            "joint_static_plan_binding",
            "joint success inner receipt",
        )?,
        &authority.joint_static_plan,
        "joint success inner receipt.joint_static_plan_binding",
    )?;
    let artifact = field_object(root, "artifact_binding", "joint success inner receipt")?;
    binding_matches(
        field_object(artifact, "manifest", "joint success artifact binding")?,
        &authority.manifest,
        "joint success artifact manifest binding",
    )?;
    binding_matches(
        field_object(
            artifact,
            "admission_receipt",
            "joint success artifact binding",
        )?,
        &authority.admission_receipt,
        "joint success artifact admission receipt binding",
    )?;
    let lease_binding = field_object(root, "joint_lease_binding", "joint success inner receipt")?;
    if field_string(
        lease_binding,
        "lease_id",
        "joint success inner receipt.joint_lease_binding",
    )? != authority.lease_id
    {
        return Err("joint success inner receipt lease ID drifted".into());
    }
    binding_matches(
        field_object(
            lease_binding,
            "receipt",
            "joint success inner receipt.joint_lease_binding",
        )?,
        &authority.lease,
        "joint success inner receipt.joint_lease_binding.receipt",
    )?;

    let phase = field_object(root, "execution_phase", "joint success inner receipt")?;
    for field in [
        "strict_artifact_admission_started",
        "strict_artifact_admission_succeeded",
        "metal_context_construction_attempted",
        "metal_context_constructed",
        "structural_kernel_trace_enabled",
        "command_commit_may_have_been_attempted",
        "command_fence_succeeded",
        "readback_started",
    ] {
        require_bool(phase, field, true, "joint success execution phase")?;
    }
    require_u64(
        phase,
        "dispatches_encoded",
        JOINT_DISPATCHES,
        "joint success execution phase",
    )?;
    if phase.get("encoded_kernel_names") != Some(&expected_names)
        || phase.get("device_dispatch_may_have_occurred") != Some(&Value::Bool(true))
    {
        return Err(
            "joint success execution phase does not prove exact post-fence execution".into(),
        );
    }
    let durable = field_object(root, "durable_capture", "joint success inner receipt")?;
    require_bool(
        durable,
        "receipt_written_last_is_completion_marker",
        true,
        "joint success durable capture",
    )?;
    require_bool(
        durable,
        "outer_reaped_capture_required",
        true,
        "joint success durable capture",
    )?;
    require_bool(
        durable,
        "replay_guarded",
        true,
        "joint success durable capture",
    )?;
    let boundary = field_object(root, "claim_boundary", "joint success inner receipt")?;
    require_bool(
        boundary,
        "component_only",
        true,
        "joint success claim boundary",
    )?;
    for field in [
        "l1_suffix_or_moe_executed",
        "complete_layer_executed",
        "token_generated",
        "decoder_started",
        "server_or_watcher_started",
    ] {
        require_bool(boundary, field, false, "joint success claim boundary")?;
    }
    for forbidden in [
        "historical_l0_receipt",
        "old_l0_receipt",
        "input_device_buffer_id",
        "input_f32le_sha256",
        "raw_pinned_buffer",
        "raw_dispatch_count",
    ] {
        if root.contains_key(forbidden)
            || capability.contains_key(forbidden)
            || execution.contains_key(forbidden)
        {
            return Err(format!("joint success receipt may not import {forbidden}"));
        }
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn verify_metal_capture_paths(args: &Args) -> Result<(PathBuf, PathBuf), String> {
    let outer = args
        .outer_capture_dir
        .as_ref()
        .ok_or("metal mode lacks outer capture directory")?;
    let inner = args
        .capture_dir
        .as_ref()
        .ok_or("metal mode lacks inner capture directory")?;
    if inner.exists() {
        return Err("joint inner capture directory must be create-new".into());
    }
    let outer = canonical_directory(outer, "joint outer capture directory")?;
    let parent = inner
        .parent()
        .ok_or("joint inner capture directory has no parent")?;
    if canonical_directory(parent, "joint inner capture parent")? != outer {
        return Err(
            "joint inner capture directory must be a direct child of the outer capture".into(),
        );
    }
    Ok((outer, inner.clone()))
}

#[cfg(target_os = "macos")]
fn write_capture_invocation(
    authority: &JointCaptureAuthority,
    capture: &Path,
) -> Result<(), String> {
    let mut invocation = json!({
        "schema": "hawking.ascension.qwen80_source_token_l0_l1_same_runtime_prefix_inner_invocation.v1",
        "status": "STARTED_QWEN80_SOURCE_TOKEN_L0_L1_SAME_RUNTIME_PREFIX_STRICT_METAL_CHILD_OUTER_REAPED",
        "mode": "metal",
        "host_binary": evidence_json(&authority.host_binary),
        "outer_launch_authority": sealed_binding_json(&authority.outer_launch),
        "joint_outer_preflight": sealed_binding_json(&authority.outer_preflight),
        "joint_lease": {"lease_id": authority.lease_id, "receipt": sealed_binding_json(&authority.lease)},
        "capture_directory": capture.to_string_lossy(),
        "execution_policy": {"source_token_id": SOURCE_TOKEN_ID, "l0_dispatches": L0_DISPATCHES, "l1_prefix_dispatches": L1_PREFIX_DISPATCHES, "total_dispatches": JOINT_DISPATCHES, "single_fence_required": true, "non_timed": true, "tcb_trace_mode": "off"},
        "claim_boundary": {"component_only": true, "l1_suffix_or_moe_authorized": false, "complete_layer_or_token_authorized": false, "server_hcli_tps_or_tournament_authorized": false},
    });
    seal(&mut invocation)?;
    write_new(&capture.join("invocation.json"), &invocation)
}

#[cfg(target_os = "macos")]
fn run_joint_metal(
    authority: &JointCaptureAuthority,
    capture: &Path,
    phase: &mut MetalExecutionPhase,
) -> Result<Value, String> {
    require_non_timed_tcb_trace_off()?;
    let admission = source_artifact_admission(authority)?;
    phase.strict_artifact_admission_started = true;
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.manifest.file.path, &admission)
        .map_err(|error| format!("joint strict artifact admission failed: {error}"))?;
    phase.strict_artifact_admission_succeeded = true;
    phase.metal_context_construction_attempted = true;
    let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
        catalog,
        Qwen80CompleteRuntimeOptions {
            max_seq_len: 1,
            trace_dispatch: false,
        },
    )
    .map_err(|error| format!("joint strict-Math runtime construction failed: {error}"))?;
    phase.metal_context_constructed = true;
    let command = runtime.begin_component_token_command_buffer();
    let parity = encode_exact_joint_l0_l1_prefix(
        &runtime,
        command,
        &authority.l0_source_outer_preflight.file.path,
        authority.workers,
        phase,
    )?;
    require_exact_joint_parity(&parity)?;
    build_success_inner_receipt(authority, capture, &parity, phase)
}

#[cfg(target_os = "macos")]
fn refusal_inner_receipt(
    authority: &JointCaptureAuthority,
    capture: &Path,
    phase: &MetalExecutionPhase,
    error: &str,
) -> Result<Value, String> {
    let mut receipt = json!({
        "schema": JOINT_INNER_SCHEMA,
        "status": JOINT_INNER_REFUSED_STATUS,
        "fixture_or_synthetic": false,
        "self_asserted": false,
        "issuer": {"role": "joint_component_capture_child", "issuer_identity_sha256": derived_identity("qwen80-joint-child-refusal", &[&authority.host_binary.sha256, &authority.lease_id, &authority.outer_launch.seal_sha256])},
        "outer_launch_authority_binding": sealed_binding_json(&authority.outer_launch),
        "joint_outer_preflight_binding": sealed_binding_json(&authority.outer_preflight),
        "joint_static_plan_binding": sealed_binding_json(&authority.joint_static_plan),
        "artifact_binding": {"manifest": sealed_binding_json(&authority.manifest), "admission_receipt": sealed_binding_json(&authority.admission_receipt)},
        "joint_lease_binding": {"lease_id": authority.lease_id, "receipt": sealed_binding_json(&authority.lease)},
        "execution_phase": phase_document(phase),
        "terminal_error": error,
        "durable_capture": {"capture_directory": capture.to_string_lossy(), "receipt_written_last_is_completion_marker": true, "outer_reaped_capture_required": true, "replay_guarded": true},
        "claim_boundary": {"component_only": true, "l1_suffix_or_moe_executed": false, "complete_layer_executed": false, "token_generated": false, "decoder_started": false, "server_or_watcher_started": false},
    });
    seal(&mut receipt)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn finalize_joint_capture(
    authority: &JointCaptureAuthority,
    capture: &Path,
    phase: &MetalExecutionPhase,
    outcome: Result<Value, String>,
) -> Result<(Value, bool), String> {
    match outcome {
        Ok(receipt) => {
            write_new(&capture.join("receipt.json"), &receipt)?;
            Ok((receipt, true))
        }
        Err(error) => {
            let receipt = refusal_inner_receipt(authority, capture, phase, &error)?;
            write_new(&capture.join("receipt.json"), &receipt)?;
            Ok((receipt, false))
        }
    }
}

#[cfg(target_os = "macos")]
fn run_metal_child(args: &Args) -> Result<String, String> {
    let authority = validate_joint_capture_authority(args)?;
    let (_outer, capture) = verify_metal_capture_paths(args)?;
    fs::create_dir(&capture).map_err(|error| {
        format!(
            "cannot create joint inner capture directory {}: {error}",
            capture.display()
        )
    })?;
    let capture = canonical_directory(&capture, "joint inner capture directory")?;
    write_capture_invocation(&authority, &capture)?;
    let mut phase = MetalExecutionPhase::default();
    let outcome = run_joint_metal(&authority, &capture, &mut phase);
    let (receipt, success) = finalize_joint_capture(&authority, &capture, &phase, outcome)?;
    let seal = receipt
        .get("seal_sha256")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if success {
        Ok(seal)
    } else {
        Err(format!(
            "joint strict-Metal child sealed a phase-accurate terminal refusal {seal}: {}",
            receipt
                .get("terminal_error")
                .and_then(Value::as_str)
                .unwrap_or("unknown terminal failure")
        ))
    }
}

/// The concrete, future-only physical body.  It is intentionally unreachable
/// from this preflight-only CLI until a separate outer/reaper validates a
/// fresh lease and receipt schema.  Keeping the real call chain compiled here
/// proves that the host is wired to the opaque continuation and consuming
/// finalizer rather than a receipt-derived buffer substitute.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
fn encode_exact_joint_l0_l1_prefix(
    runtime: &Qwen80CompleteNativeRuntime,
    command: TokenCommandBuffer<'_>,
    l0_source_outer_preflight: &Path,
    workers: usize,
    phase: &mut MetalExecutionPhase,
) -> Result<Qwen80SameRuntimeL0L1PrefixParity, String> {
    let (source_bridge, _) = source_l0::build_source_token_all_ten_bridge_from_outer_preflight(
        runtime,
        l0_source_outer_preflight,
        workers,
    )?;
    let mut command = command;
    command
        .enable_structural_kernel_trace()
        .map_err(|error| format!("joint host requires non-timed structural trace: {error}"))?;
    phase.structural_kernel_trace_enabled = true;
    let l0_resources = source_prefix::encode_source_input_l0_true_moe_capture(
        runtime,
        &mut command,
        SOURCE_TOKEN_ID as u32,
        &source_bridge,
    )?;
    let continuation = l0_resources.into_canonical_l0_true_moe_continuation(runtime, &command)?;
    let l1 = runtime
        .encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into(
            &mut command,
            continuation,
        )
        .map_err(|error| format!("joint host L1 prefix refused: {error}"))?;
    phase.dispatches_encoded = command.dispatch_count();
    phase.encoded_kernel_names = command
        .structural_kernel_names()
        .ok_or("joint host did not retain its structural kernel trace")?
        .to_vec();
    if phase.dispatches_encoded != JOINT_DISPATCHES as usize
        || phase
            .encoded_kernel_names
            .iter()
            .map(String::as_str)
            .ne(L0_KERNELS.iter().chain(L1_KERNELS.iter()).copied())
    {
        return Err("joint host encoded trace drifted before the consuming finalizer".into());
    }
    // The consuming finalizer owns the sole commit.  Set this immediately
    // before invoking it so a terminal refusal cannot falsely claim that no
    // device submission was possible after this point.
    phase.command_commit_may_have_been_attempted = true;
    let parity = l1
        .finalize_after_exact_joint_fence(runtime, command)
        .map_err(|error| format!("joint host exact finalizer refused: {error}"))?;
    phase.command_fence_succeeded = true;
    phase.readback_started = true;
    Ok(parity)
}

fn main() {
    let outcome = parse_args(&env::args().collect::<Vec<_>>()).and_then(|args| match args.mode {
        Mode::Preflight => {
            let document = preflight_document(&args)?;
            let out = args
                .out
                .as_ref()
                .ok_or("preflight mode lacks output path")?;
            write_new(out, &document)?;
            Ok((
                "prepared sealed CPU-only strict joint L0→L1 host preflight",
                document["seal_sha256"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            ))
        }
        Mode::SourceAdmissionPreflight => {
            #[cfg(target_os = "macos")]
            {
                run_source_admission_preflight(&args).map(|()| {
                    (
                        "validated read-only strict source admission for the joint host",
                        String::new(),
                    )
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = args;
                Err("strict same-runtime Metal capture is unavailable on this target".into())
            }
        }
        Mode::Metal => {
            #[cfg(target_os = "macos")]
            {
                run_metal_child(&args).map(|seal| {
                    (
                        "sealed strict-Metal same-runtime L0(23)+L1(9) component receipt",
                        seal,
                    )
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = args;
                Err("strict same-runtime Metal capture is unavailable on this target".into())
            }
        }
    });
    match outcome {
        Ok((label, seal)) => println!("{label}: {seal}"),
        Err(error) => {
            eprintln!("{error}\n{}", usage());
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_joint_kernel_trace_is_23_plus_9_and_non_timed() {
        assert_eq!(L0_KERNELS.len() as u64, L0_DISPATCHES);
        assert_eq!(L1_KERNELS.len() as u64, L1_PREFIX_DISPATCHES);
        assert_eq!(exact_joint_trace().len() as u64, JOINT_DISPATCHES);
        assert_eq!(expected_trace(&L0_KERNELS).len() as u64, L0_DISPATCHES);
    }

    #[test]
    fn parser_requires_complete_mode_specific_authority_and_absolute_paths() {
        let error = parse_args(&["host".into(), "--mode".into(), "metal".into()]).unwrap_err();
        assert!(error.contains("--workers 1..4 is required"));
        let error = parse_args(&[
            "host".into(),
            "--mode".into(),
            "metal".into(),
            "--workers".into(),
            "1".into(),
            "--joint-static-plan".into(),
            "/tmp/static.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("--joint-outer-preflight is required"));
        let error = parse_args(&[
            "host".into(),
            "--joint-static-plan".into(),
            "relative.json".into(),
            "--l0-source-outer-preflight".into(),
            "/tmp/l0.json".into(),
            "--out".into(),
            "/tmp/out.json".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("absolute"));
        let error = parse_args(&[
            "host".into(),
            "--mode".into(),
            "source-admission-preflight".into(),
            "--joint-outer-preflight".into(),
            "/tmp/outer.json".into(),
            "--workers".into(),
            "1".into(),
            "--lease-receipt".into(),
            "/tmp/lease.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("refuses capture arguments"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn phase_receipt_is_honest_after_a_possible_unconfirmed_submit() {
        let phase = MetalExecutionPhase {
            command_commit_may_have_been_attempted: true,
            ..MetalExecutionPhase::default()
        };
        assert_eq!(
            phase_document(&phase)["device_dispatch_may_have_occurred"],
            Value::Null
        );
        let phase = MetalExecutionPhase {
            command_commit_may_have_been_attempted: true,
            command_fence_succeeded: true,
            ..MetalExecutionPhase::default()
        };
        assert_eq!(
            phase_document(&phase)["device_dispatch_may_have_occurred"],
            Value::Bool(true)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn receipt_parity_contract_refuses_missing_device_witness() {
        let value = json!({
            "passed": true,
            "cpu_f32le_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "max_abs_error": 0.0,
        });
        let error = require_inner_parity(&value, "test parity").unwrap_err();
        assert!(error.contains("device_f32le_sha256"));
    }

    #[test]
    fn receipt_seal_uses_python_compact_float_spelling() {
        let document = json!({"z": 1.0, "a": 1.0e-6, "negative_zero": -0.0});
        let canonical = String::from_utf8(canonical_json(&document).unwrap()).unwrap();
        assert_eq!(canonical, r#"{"a":1e-06,"negative_zero":-0.0,"z":1.0}"#);
    }

    #[test]
    fn immutable_admission_manifest_uses_raw_file_sha_not_canonical_json_identity() {
        let manifest = SealedFile {
            file: FileEvidence {
                path: PathBuf::from("/tmp/manifest.json"),
                bytes: 17,
                sha256: "a".repeat(64),
            },
            document: json!({"schema": "fixture"}),
            document_sha256: "b".repeat(64),
            seal_sha256: "c".repeat(64),
        };
        let admitted = json!({
            "document_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "seal_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        });
        assert!(immutable_admission_manifest_matches(
            admitted.as_object().expect("fixture object"),
            &manifest,
        )
        .expect("fixture parses"));

        let substituted = json!({
            "document_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "seal_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        });
        assert!(!immutable_admission_manifest_matches(
            substituted.as_object().expect("fixture object"),
            &manifest,
        )
        .expect("fixture parses"));
    }
}
