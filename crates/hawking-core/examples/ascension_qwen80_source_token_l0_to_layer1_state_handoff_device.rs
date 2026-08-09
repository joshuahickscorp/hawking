//! Build-only source-token Qwen80 L0 state/output handoff child.
//!
//! The earned source-token L0 9+14 component has one strict-Metal second
//! residual *hash*, but its first receipt intentionally did not retain a live
//! output allocation or post-L0 state/checkpoint witnesses. This target is
//! the narrow successor child contract. It consumes only the sealed
//! incomplete L0-to-L1 authority, rejects raw/legacy component receipts, and
//! prepares a fresh source-token L0 re-encode that can later retain:
//!
//! - the exact 2,048-f32 / 8,192-byte second-residual `PinnedBuffer`,
//! - post-L0 active DeltaNet conv/recurrent state bytes,
//! - device-resident source-zero rollback checkpoints, and
//! - a same-runtime Layer-1 active-slot-1 input binding.
//!
//! Its default CLI mode is a file-only preflight.  The separate `metal` mode
//! is deliberately unavailable without a fresh sealed outer preflight, an
//! immutable one-shot lease, and an outer-reaper launch authority.  The
//! device child still stops at a retained L0 result plus a typed *unexecuted*
//! L1 slot-1 binding: it never encodes Layer 1 and cannot promote a decoder
//! or complete-layer claim.

#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "ascension_qwen80_first_residual_bridge_device.rs"]
mod source_prefix;

// Reuse the already hardened source-token/all-ten authority parser only as a
// CPU validation/build bridge. The state-handoff child never invokes that
// module's executable entrypoint.
#[cfg(target_os = "macos")]
#[allow(dead_code)]
#[path = "ascension_qwen80_source_token_all_ten_true_moe_graph_device.rs"]
mod all_ten_source_child;

#[cfg(target_os = "macos")]
use hawking_core::metal::{PinnedBuffer, TokenCommandBuffer};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80AllTenTrueMoeSourceBridge, Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime,
    Qwen80CompleteRuntimeOptions, Qwen80L0TrueMoeFixedDeviceBuffers,
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
use std::process;

const CHILD_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_to_layer1_state_handoff_device.v1";
const PREPARED_STATUS: &str =
    "PREPARED_QWEN80_SOURCE_TOKEN_L0_STATE_COMMIT_ROLLBACK_AND_LAYER1_HANDOFF_CHILD_NOT_EXECUTED";
const BASELINE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_to_layer1_handoff_authority.v1";
const BASELINE_AUTHORITY_STATUS: &str =
    "ASSESSED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_HANDOFF_INCOMPLETE_MISSING_RETAINED_DEVICE_OUTPUT_AND_POST_STATE_WITNESSES";
const NEXT_LAYER_HANDOFF_WITNESS_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_to_layer1_device_handoff_witness.v1";
const NEXT_LAYER_HANDOFF_WITNESS_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_TO_LAYER1_RETAINED_DEVICE_HANDOFF_COMPONENT_ONLY";
const PRE_L1_HANDOFF_CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_capture.v1";
const PRE_L1_HANDOFF_CAPTURE_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_COMPONENT_ONLY";
const OUTER_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_preflight.v1";
const OUTER_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_READY_NOT_LEASED_OR_EXECUTED";
const LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_quiet_metal_lease.v1";
const LEASE_STATUS: &str =
    "GRANTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_NON_TIMED_DEVICE_PARITY_LEASE";
const OUTER_LAUNCH_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_outer_launch_authority.v1";
const OUTER_LAUNCH_STATUS: &str =
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_REAPED_ONE_SHOT_METAL_CHILD";
const OUTER_PREFLIGHT_PROOF_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_state_handoff_preflight_proof.v1";
const OUTER_PREFLIGHT_PROOF_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_OUTER_AND_CHILD_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const SOURCE_ALL_TEN_OUTER_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1";
const SOURCE_ALL_TEN_OUTER_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED";
const ADMISSION_POINTER_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_POINTER_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MAX_WORKERS: usize = 4;
const MODEL_KEY: &str = "qwen80";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_DOCUMENT_SHA: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const HIDDEN: u64 = 2_048;
const HIDDEN_BYTES: u64 = HIDDEN * 4;
const SOURCE_TOKEN_ID: u64 = 1;
const PREFIX_DISPATCHES: u64 = 9;
const SUFFIX_DISPATCHES: u64 = 14;
const TOTAL_DISPATCHES: u64 = PREFIX_DISPATCHES + SUFFIX_DISPATCHES;
const PREFIX_KERNELS: [&str; PREFIX_DISPATCHES as usize] = [
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
const SUFFIX_KERNELS: [&str; SUFFIX_DISPATCHES as usize] = [
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
const L0_CONV_BYTES: u64 = 98_304;
const L0_RECURRENT_BYTES: u64 = 2_097_152;
const L1_CONV_OFFSET_BYTES: u64 = L0_CONV_BYTES;
const L1_RECURRENT_OFFSET_BYTES: u64 = L0_RECURRENT_BYTES;

fn expected_kernel_order() -> Vec<&'static str> {
    PREFIX_KERNELS
        .iter()
        .chain(SUFFIX_KERNELS.iter())
        .copied()
        .collect()
}

#[derive(Clone, Debug)]
struct BoundFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

impl BoundFile {
    fn json(&self) -> Value {
        json!({
            "path": self.path,
            "present": true,
            "bytes": self.bytes,
            "sha256": self.sha256,
        })
    }
}

#[derive(Debug)]
struct Args {
    handoff_authority: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CaptureMode {
    Preflight,
    Metal,
}

#[derive(Clone, Debug)]
struct CaptureArgs {
    outer_preflight: PathBuf,
    mode: CaptureMode,
    lease_receipt: Option<PathBuf>,
    outer_launch_authority: Option<PathBuf>,
    outer_capture_dir: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    workers: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct CaptureAuthority {
    args: CaptureArgs,
    outer_preflight: BoundFile,
    outer_preflight_seal: String,
    manifest: BoundFile,
    manifest_seal: String,
    // The outer preflight preserves the pointer observation it saw.  The
    // pointer is a versioned-current document and may be resealed for
    // housekeeping, so this historical evidence is never substituted for
    // the freshly validated launch observation below.
    admission_current: BoundFile,
    admission_pointer_seal: String,
    launch_admission_current: BoundFile,
    launch_admission_pointer_seal: String,
    admission_receipt: BoundFile,
    admission_receipt_seal: String,
    source_audit_seal: String,
    source_revision: String,
    source_all_ten_outer_preflight: BoundFile,
    source_all_ten_outer_preflight_seal: String,
    child_preflight: BoundFile,
    child_preflight_seal: String,
    baseline_handoff_authority: BoundFile,
    baseline_handoff_authority_seal: String,
    baseline_second_residual_f32le_sha256: String,
    static_binding: Qwen80SourceTokenL0ToL1StaticBinding,
    lease: Option<(BoundFile, String)>,
    lease_id: Option<String>,
    outer_launch_authority: Option<(BoundFile, String)>,
}

/// A live validation of the mutable admission-current pointer.  Only this
/// object is trusted for the current selection; historical outer-preflight
/// evidence remains in receipts for provenance but cannot authorize a
/// different immutable manifest or admission receipt.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct VersionedCurrentPointer {
    file: BoundFile,
    seal: String,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug, Default)]
struct MetalExecutionPhase {
    strict_artifact_admission_started: bool,
    strict_artifact_admission_succeeded: bool,
    metal_context_construction_attempted: bool,
    metal_context_constructed: bool,
    structural_kernel_trace_enabled: bool,
    command_commit_attempted: bool,
    command_fence_succeeded: bool,
    readback_started: bool,
    dispatches_encoded: usize,
    encoded_kernel_names: Vec<String>,
}

#[derive(Clone, Debug)]
struct BaselineAuthority {
    file: BoundFile,
    seal: String,
    second_residual_sha256: String,
    session_id: String,
    source_routes: Vec<u64>,
    l0_active_conv_allocation: String,
    l0_active_recurrent_allocation: String,
    l0_rollback_conv_allocation: String,
    l0_rollback_recurrent_allocation: String,
    l1_active_conv_allocation: String,
    l1_active_recurrent_allocation: String,
}

/// The exact static state-layout facts carried from the sealed CPU assessor
/// into a later same-runtime device child.  These are metadata only; live
/// buffer identities are captured after the shared command buffer fences.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
pub struct Qwen80SourceTokenL0ToL1StaticBinding {
    pub session_id: String,
    pub l0_active_conv_allocation: String,
    pub l0_active_recurrent_allocation: String,
    pub l0_rollback_conv_allocation: String,
    pub l0_rollback_recurrent_allocation: String,
    pub l1_active_conv_allocation: String,
    pub l1_active_recurrent_allocation: String,
}

#[cfg(target_os = "macos")]
impl From<&BaselineAuthority> for Qwen80SourceTokenL0ToL1StaticBinding {
    fn from(authority: &BaselineAuthority) -> Self {
        Self {
            session_id: authority.session_id.clone(),
            l0_active_conv_allocation: authority.l0_active_conv_allocation.clone(),
            l0_active_recurrent_allocation: authority.l0_active_recurrent_allocation.clone(),
            l0_rollback_conv_allocation: authority.l0_rollback_conv_allocation.clone(),
            l0_rollback_recurrent_allocation: authority.l0_rollback_recurrent_allocation.clone(),
            l1_active_conv_allocation: authority.l1_active_conv_allocation.clone(),
            l1_active_recurrent_allocation: authority.l1_active_recurrent_allocation.clone(),
        }
    }
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

fn file_evidence(path: &Path, label: &str) -> Result<BoundFile, String> {
    let path = canonical_regular(path, label)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    Ok(BoundFile {
        path,
        bytes: raw.len() as u64,
        sha256: sha256_hex(&raw),
    })
}

fn read_json(path: &Path, label: &str) -> Result<(BoundFile, Value), String> {
    let evidence = file_evidence(path, label)?;
    let raw =
        fs::read(&evidence.path).map_err(|error| format!("cannot reread {label}: {error}"))?;
    let value = serde_json::from_slice::<Value>(&raw)
        .map_err(|error| format!("cannot parse {label}: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((evidence, value))
}

fn require_evidence(
    object: &Map<String, Value>,
    field: &str,
    expected: &BoundFile,
    label: &str,
) -> Result<(), String> {
    let observed = object_field(object, field, label)?;
    if observed.get("present").and_then(Value::as_bool) != Some(true)
        || string_field(observed, "path", &format!("{label}.{field}"))?
            != expected.path.to_string_lossy()
        || u64_field(observed, "bytes", &format!("{label}.{field}"))? != expected.bytes
        || sha_field(observed, "sha256", &format!("{label}.{field}"))? != expected.sha256
    {
        return Err(format!("{label}.{field} file evidence drifted"));
    }
    Ok(())
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

fn string_field<'a>(
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

fn u64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn bool_field(
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

fn exact_string(
    object: &Map<String, Value>,
    field: &str,
    expected: &str,
    label: &str,
) -> Result<(), String> {
    let observed = string_field(object, field, label)?;
    if observed != expected {
        return Err(format!(
            "{label}.{field}={observed:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn sha_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let value = string_field(object, field, label)?;
    if !is_sha256(value) {
        return Err(format!("{label}.{field} must be a lowercase SHA-256"));
    }
    Ok(value.into())
}

/// Render finite JSON floats with CPython's `json.dumps` spelling.
///
/// `lab.receipts.seal` uses Python's sorted, compact JSON encoding.  Rust's
/// otherwise-valid Ryu rendering differs for scientific exponents (`e-6`
/// versus Python's `e-06`), so receipt sealing must use this shared lexical
/// form rather than `serde_json::to_vec` directly.
fn python_json_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("canonical JSON floating number must be finite")?;
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
        Some(index) => {
            let exponent = unsigned[index + 1..]
                .parse::<i32>()
                .map_err(|error| format!("canonical JSON exponent is invalid: {error}"))?;
            (&unsigned[..index], exponent)
        }
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
        .ok_or("canonical JSON scientific exponent overflows")?;
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
        Value::Number(number) => {
            if number.is_i64() || number.is_u64() {
                output.push_str(&number.to_string());
            } else {
                output.push_str(&python_json_float(number)?);
            }
        }
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

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let object = object(value, label)?;
    let observed = sha_field(object, "seal_sha256", label)?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_hex(&canonical_json(&Value::Object(unsigned))?);
    if observed != expected {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed)
}

fn seal(value: &mut Value) -> Result<String, String> {
    let object = object(value, "child preflight output")?;
    if object.contains_key("seal_sha256") {
        return Err("child preflight output must be unsealed before sealing".into());
    }
    let seal = sha256_hex(&canonical_json(value)?);
    value
        .as_object_mut()
        .expect("child preflight output object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn state_allocation(
    layer: &Map<String, Value>,
    field: &str,
    expected_slot: u64,
    expected_offset: u64,
    expected_capacity: u64,
    label: &str,
) -> Result<String, String> {
    let range = object_field(layer, field, label)?;
    if u64_field(range, "slot", label)? != expected_slot
        || u64_field(range, "offset_bytes", label)? != expected_offset
        || u64_field(range, "capacity_bytes", label)? != expected_capacity
    {
        return Err(format!("{label}.{field} geometry drifted"));
    }
    string_field(range, "allocation_id", label).map(str::to_owned)
}

fn parse_baseline_authority(file: BoundFile, value: &Value) -> Result<BaselineAuthority, String> {
    let root = object(value, "L0-to-L1 baseline authority")?;
    exact_string(
        root,
        "schema",
        BASELINE_AUTHORITY_SCHEMA,
        "L0-to-L1 baseline authority",
    )?;
    exact_string(
        root,
        "status",
        BASELINE_AUTHORITY_STATUS,
        "L0-to-L1 baseline authority",
    )?;
    let seal = verify_seal(value, "L0-to-L1 baseline authority")?;
    bool_field(
        root,
        "ready_for_l1_device_handoff",
        false,
        "L0-to-L1 baseline authority",
    )?;
    bool_field(root, "component_only", true, "L0-to-L1 baseline authority")?;
    let source = object_field(root, "source_binding", "L0-to-L1 baseline authority")?;
    exact_string(
        source,
        "model_key",
        MODEL_KEY,
        "L0-to-L1 baseline authority.source_binding",
    )?;
    exact_string(
        source,
        "source_revision",
        SOURCE_REVISION,
        "L0-to-L1 baseline authority.source_binding",
    )?;
    exact_string(
        source,
        "manifest_document_sha256",
        MANIFEST_DOCUMENT_SHA,
        "L0-to-L1 baseline authority.source_binding",
    )?;
    exact_string(
        source,
        "manifest_seal_sha256",
        MANIFEST_SEAL,
        "L0-to-L1 baseline authority.source_binding",
    )?;
    exact_string(
        source,
        "admission_receipt_seal_sha256",
        ADMISSION_RECEIPT_SEAL,
        "L0-to-L1 baseline authority.source_binding",
    )?;
    if u64_field(
        source,
        "source_token_id",
        "L0-to-L1 baseline authority.source_binding",
    )? != SOURCE_TOKEN_ID
    {
        return Err("L0-to-L1 baseline authority source token drifted".into());
    }
    let capture = object_field(
        root,
        "consumed_component_capture",
        "L0-to-L1 baseline authority",
    )?;
    if u64_field(
        capture,
        "layer",
        "L0-to-L1 baseline authority.consumed_component_capture",
    )? != 0
        || u64_field(
            capture,
            "linear_state_slot",
            "L0-to-L1 baseline authority.consumed_component_capture",
        )? != 0
    {
        return Err("L0-to-L1 baseline authority must bind L0/slot0".into());
    }
    let graph = object_field(
        capture,
        "same_command_graph",
        "L0-to-L1 baseline authority.consumed_component_capture",
    )?;
    for (field, expected) in [
        ("prefix_dispatches", PREFIX_DISPATCHES),
        ("suffix_dispatches", SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ] {
        if u64_field(
            graph,
            field,
            "L0-to-L1 baseline authority.consumed_component_capture.same_command_graph",
        )? != expected
        {
            return Err(format!("baseline component {field} drifted"));
        }
    }
    let second = object_field(
        capture,
        "second_residual",
        "L0-to-L1 baseline authority.consumed_component_capture",
    )?;
    if u64_field(
        second,
        "elements",
        "L0-to-L1 baseline authority.consumed_component_capture.second_residual",
    )? != HIDDEN
        || u64_field(
            second,
            "bytes",
            "L0-to-L1 baseline authority.consumed_component_capture.second_residual",
        )? != HIDDEN_BYTES
    {
        return Err("baseline component second-residual geometry drifted".into());
    }
    let second_residual_sha256 = sha_field(
        second,
        "f32le_sha256",
        "L0-to-L1 baseline authority.consumed_component_capture.second_residual",
    )?;
    let route_guard = object_field(
        capture,
        "route_guard",
        "L0-to-L1 baseline authority.consumed_component_capture",
    )?;
    bool_field(
        route_guard,
        "passed",
        true,
        "L0-to-L1 baseline authority.consumed_component_capture.route_guard",
    )?;
    if u64_field(
        route_guard,
        "route_witness_count",
        "L0-to-L1 baseline authority.consumed_component_capture.route_guard",
    )? != 10
    {
        return Err("baseline component does not retain ten route witnesses".into());
    }
    let source_routes = array_field(
        route_guard,
        "ordered_source_route_ids",
        "L0-to-L1 baseline authority.consumed_component_capture.route_guard",
    )?
    .iter()
    .map(|value| value.as_u64().ok_or("source route id must be u64"))
    .collect::<Result<Vec<_>, _>>()?;
    if source_routes.len() != 10 {
        return Err("baseline component source route count must be ten".into());
    }
    let state = object_field(
        root,
        "static_state_layout_authority",
        "L0-to-L1 baseline authority",
    )?;
    let session_id = string_field(
        state,
        "session_id",
        "L0-to-L1 baseline authority.static_state_layout_authority",
    )?
    .to_owned();
    bool_field(
        state,
        "l0_and_l1_slots_verified_disjoint",
        true,
        "L0-to-L1 baseline authority.static_state_layout_authority",
    )?;
    let l0 = object_field(
        state,
        "l0",
        "L0-to-L1 baseline authority.static_state_layout_authority",
    )?;
    let l1 = object_field(
        state,
        "l1",
        "L0-to-L1 baseline authority.static_state_layout_authority",
    )?;
    if u64_field(
        l0,
        "linear_state_slot",
        "L0-to-L1 baseline authority.static_state_layout_authority.l0",
    )? != 0
        || u64_field(
            l1,
            "linear_state_slot",
            "L0-to-L1 baseline authority.static_state_layout_authority.l1",
        )? != 1
    {
        return Err("baseline state layout slot mapping drifted".into());
    }
    let l0_active_conv_allocation =
        state_allocation(l0, "active_conv", 0, 0, L0_CONV_BYTES, "baseline L0")?;
    let l0_active_recurrent_allocation = state_allocation(
        l0,
        "active_recurrent",
        0,
        0,
        L0_RECURRENT_BYTES,
        "baseline L0",
    )?;
    let l0_rollback_conv_allocation =
        state_allocation(l0, "rollback_conv", 0, 0, L0_CONV_BYTES, "baseline L0")?;
    let l0_rollback_recurrent_allocation = state_allocation(
        l0,
        "rollback_recurrent",
        0,
        0,
        L0_RECURRENT_BYTES,
        "baseline L0",
    )?;
    let l1_active_conv_allocation = state_allocation(
        l1,
        "active_conv",
        1,
        L1_CONV_OFFSET_BYTES,
        L1_CONV_OFFSET_BYTES + L0_CONV_BYTES,
        "baseline L1",
    )?;
    let l1_active_recurrent_allocation = state_allocation(
        l1,
        "active_recurrent",
        1,
        L1_RECURRENT_OFFSET_BYTES,
        L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES,
        "baseline L1",
    )?;
    let required = object_field(
        root,
        "next_required_real_decoder_dependency",
        "L0-to-L1 baseline authority",
    )?;
    exact_string(
        required,
        "schema",
        NEXT_LAYER_HANDOFF_WITNESS_SCHEMA,
        "L0-to-L1 baseline authority.next_required_real_decoder_dependency",
    )?;
    exact_string(
        required,
        "required_status",
        NEXT_LAYER_HANDOFF_WITNESS_STATUS,
        "L0-to-L1 baseline authority.next_required_real_decoder_dependency",
    )?;
    let assessment = object_field(root, "handoff_assessment", "L0-to-L1 baseline authority")?;
    bool_field(
        assessment,
        "historical_output_hash_is_not_a_retained_device_buffer",
        true,
        "L0-to-L1 baseline authority.handoff_assessment",
    )?;
    bool_field(
        assessment,
        "historical_initial_state_hashes_are_not_post_state_commit_witnesses",
        true,
        "L0-to-L1 baseline authority.handoff_assessment",
    )?;
    let missing = array_field(
        assessment,
        "missing_real_evidence",
        "L0-to-L1 baseline authority.handoff_assessment",
    )?;
    if missing.len() != 3 {
        return Err(
            "baseline authority must enumerate exactly three missing handoff witnesses".into(),
        );
    }
    let claim = object_field(root, "claim_boundary", "L0-to-L1 baseline authority")?;
    bool_field(
        claim,
        "cpu_only_assessment",
        true,
        "L0-to-L1 baseline authority.claim_boundary",
    )?;
    bool_field(
        claim,
        "no_metal_context_or_device_dispatch",
        true,
        "L0-to-L1 baseline authority.claim_boundary",
    )?;
    Ok(BaselineAuthority {
        file,
        seal,
        second_residual_sha256,
        session_id,
        source_routes,
        l0_active_conv_allocation,
        l0_active_recurrent_allocation,
        l0_rollback_conv_allocation,
        l0_rollback_recurrent_allocation,
        l1_active_conv_allocation,
        l1_active_recurrent_allocation,
    })
}

fn prepared_document(authority: &BaselineAuthority) -> Value {
    json!({
        "schema": CHILD_SCHEMA,
        "status": PREPARED_STATUS,
        "mode": "cpu_only_preflight",
        "baseline_handoff_authority": authority.file.json(),
        "baseline_handoff_authority_seal_sha256": authority.seal,
        "source_binding": {
            "model_key": MODEL_KEY,
            "source_revision": SOURCE_REVISION,
            "manifest_document_sha256": MANIFEST_DOCUMENT_SHA,
            "manifest_seal_sha256": MANIFEST_SEAL,
            "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
            "source_token_id": SOURCE_TOKEN_ID,
        },
        "fresh_capture_required": {
            "must_reencode_source_token_l0": true,
            "may_not_reuse_historical_component_output_hash_as_a_live_buffer": true,
            "same_token_command_buffer": {
                "prefix_dispatches": PREFIX_DISPATCHES,
                "suffix_dispatches": SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "fence_once_after_prefix_and_suffix": true,
            },
            "source_route_ids": authority.source_routes,
        },
        "planned_pre_l1_handoff_capture": {
            "schema": PRE_L1_HANDOFF_CAPTURE_SCHEMA,
            "status": PRE_L1_HANDOFF_CAPTURE_STATUS,
            "l1_binding_not_executed": true,
            "l1_prefix_dispatches": 0,
            "may_not_satisfy_next_layer_execution_dependency": true,
            "receipt_must_record_l0_post_state_rollback_and_retained_output": true,
        },
        "required_next_layer_handoff_witness": {
            "schema": NEXT_LAYER_HANDOFF_WITNESS_SCHEMA,
            "status": NEXT_LAYER_HANDOFF_WITNESS_STATUS,
            "remains_required_after_planned_pre_l1_capture": true,
            "component_only": true,
            "retained_l0_second_residual": {
                "elements": HIDDEN,
                "bytes": HIDDEN_BYTES,
                "f32le_sha256_must_equal_baseline": authority.second_residual_sha256,
                "device_buffer_id_required": true,
                "future_layer1_execution_retention_required": true,
            },
            "l0_post_state_commit": {
                "layer": 0,
                "linear_state_slot": 0,
                "checkpoint_before_mutation_required": true,
                "active_conv": {"allocation_id": authority.l0_active_conv_allocation, "slot":0, "offset_bytes":0, "capacity_bytes":L0_CONV_BYTES, "post_state_hash_required":true},
                "active_recurrent": {"allocation_id": authority.l0_active_recurrent_allocation, "slot":0, "offset_bytes":0, "capacity_bytes":L0_RECURRENT_BYTES, "post_state_hash_required":true},
                "rollback_conv": {"allocation_id": authority.l0_rollback_conv_allocation, "slot":0, "offset_bytes":0, "capacity_bytes":L0_CONV_BYTES, "checkpoint_hash_required":true},
                "rollback_recurrent": {"allocation_id": authority.l0_rollback_recurrent_allocation, "slot":0, "offset_bytes":0, "capacity_bytes":L0_RECURRENT_BYTES, "checkpoint_hash_required":true},
            },
            "layer1_input_binding": {
                "session_id": authority.session_id,
                "layer": 1,
                "linear_state_slot": 1,
                "input_device_buffer_id_must_equal_retained_l0_output": true,
                "input_f32le_sha256_must_equal_retained_l0_output": true,
                "same_command_graph_retained_required": true,
                "active_conv": {"allocation_id": authority.l1_active_conv_allocation, "slot":1, "offset_bytes":L1_CONV_OFFSET_BYTES, "capacity_bytes":L1_CONV_OFFSET_BYTES + L0_CONV_BYTES, "device_buffer_identity_required":true},
                "active_recurrent": {"allocation_id": authority.l1_active_recurrent_allocation, "slot":1, "offset_bytes":L1_RECURRENT_OFFSET_BYTES, "capacity_bytes":L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES, "device_buffer_identity_required":true},
            },
        },
        "implementation_boundary": {
            "macos_encoder_type_checked": true,
            "device_context_or_dispatch_performed": false,
            "artifact_scan_or_payload_open_performed": false,
            "outer_reaped_receipt_last_replay_guard_required_before_future_metal_mode": true,
        },
        "claim_boundary": {
            "component_only_even_after_a_future_pass": true,
            "planned_pre_l1_capture_is_not_l1_execution": true,
            "planned_pre_l1_capture_may_not_promote_the_next_layer_dependency": true,
            "not_a_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_result": true,
            "watcher_or_server_transition_not_authorized": true,
        },
    })
}

fn write_new_labeled(path: &Path, label: &str, bytes: &[u8]) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let parent = path
        .parent()
        .ok_or_else(|| format!("{label} has no parent"))?;
    let metadata = fs::symlink_metadata(parent)
        .map_err(|error| format!("cannot stat {label} parent {}: {error}", parent.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(format!(
            "{label} parent must be an existing non-symlink directory"
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("refusing to overwrite {label} {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("cannot write {label}: {error}"))?;
    file.sync_all()
        .map_err(|error| format!("cannot sync {label}: {error}"))?;
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    write_new_labeled(path, "--out", bytes)
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_l0_to_layer1_state_handoff_device \\\n--handoff-authority ABSOLUTE_PATH --out ABSOLUTE_NEW_PATH"
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", usage()))?;
        if !matches!(flag.as_str(), "--handoff-authority" | "--out") {
            return Err(format!("unsupported option {flag:?}; {}", usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("{flag} was repeated"));
        }
    }
    let required = |flag: &str| {
        values
            .get(flag)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing {flag}; {}", usage()))
    };
    let args = Args {
        handoff_authority: required("--handoff-authority")?,
        out: required("--out")?,
    };
    if !args.out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    Ok(args)
}

fn capture_usage() -> &'static str {
    "usage: ascension_qwen80_source_token_l0_to_layer1_state_handoff_device \\
--outer-preflight ABSOLUTE_PATH --mode preflight --workers 1..4 | \\
--outer-preflight ABSOLUTE_PATH --mode metal --lease-receipt ABSOLUTE_PATH \\
--outer-launch-authority ABSOLUTE_PATH \\
--outer-capture-dir ABSOLUTE_DIRECTORY --capture-dir NEW_ABSOLUTE_DIRECTORY --workers 1..4"
}

fn parse_capture_args<I>(arguments: I) -> Result<CaptureArgs, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let value = arguments
            .next()
            .ok_or_else(|| format!("missing value after {flag}; {}", capture_usage()))?;
        if !matches!(
            flag.as_str(),
            "--outer-preflight"
                | "--mode"
                | "--lease-receipt"
                | "--outer-launch-authority"
                | "--outer-capture-dir"
                | "--capture-dir"
                | "--workers"
        ) {
            return Err(format!("unsupported option {flag:?}; {}", capture_usage()));
        }
        if values.insert(flag.clone(), value).is_some() {
            return Err(format!("{flag} was repeated"));
        }
    }
    let mut required_path = |flag: &str| {
        values
            .remove(flag)
            .map(PathBuf::from)
            .ok_or_else(|| format!("missing {flag}; {}", capture_usage()))
    };
    let outer_preflight = required_path("--outer-preflight")?;
    let mode = match values.remove("--mode").as_deref() {
        Some("preflight") => CaptureMode::Preflight,
        Some("metal") => CaptureMode::Metal,
        _ => {
            return Err(format!(
                "--mode must be preflight or metal; {}",
                capture_usage()
            ))
        }
    };
    let optional_path =
        |flag: &str, values: &mut BTreeMap<String, String>| values.remove(flag).map(PathBuf::from);
    let lease_receipt = optional_path("--lease-receipt", &mut values);
    let outer_launch_authority = optional_path("--outer-launch-authority", &mut values);
    let outer_capture_dir = optional_path("--outer-capture-dir", &mut values);
    let capture_dir = optional_path("--capture-dir", &mut values);
    let workers = values
        .remove("--workers")
        .ok_or_else(|| format!("missing --workers; {}", capture_usage()))?
        .parse::<usize>()
        .map_err(|_| "--workers must be an integer".to_owned())?;
    if !values.is_empty() {
        return Err(format!(
            "unsupported arguments remain: {:?}; {}",
            values.keys().collect::<Vec<_>>(),
            capture_usage()
        ));
    }
    if !outer_preflight.is_absolute() || !(1..=MAX_WORKERS).contains(&workers) {
        return Err(format!(
            "--outer-preflight must be absolute and --workers must be 1..={MAX_WORKERS}"
        ));
    }
    match mode {
        CaptureMode::Preflight
            if lease_receipt.is_some()
                || outer_launch_authority.is_some()
                || outer_capture_dir.is_some()
                || capture_dir.is_some() =>
        {
            Err("preflight mode refuses lease, outer-authority, and capture paths".into())
        }
        CaptureMode::Metal
            if lease_receipt.is_none()
                || outer_launch_authority.is_none()
                || outer_capture_dir.is_none()
                || capture_dir.is_none() =>
        {
            Err("metal mode requires lease, outer authority, outer capture, and fresh child capture paths".into())
        }
        _ => Ok(CaptureArgs {
            outer_preflight,
            mode,
            lease_receipt,
            outer_launch_authority,
            outer_capture_dir,
            capture_dir,
            workers,
        }),
    }
}

fn run(args: Args) -> Result<(PathBuf, String), String> {
    let (file, value) = read_json(&args.handoff_authority, "--handoff-authority")?;
    let authority = parse_baseline_authority(file, &value)?;
    let mut document = prepared_document(&authority);
    let seal = seal(&mut document)?;
    let bytes = serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?;
    write_new(&args.out, &bytes)?;
    Ok((args.out, seal))
}

/// Load the only static handoff plan accepted by the future strict-Metal
/// child. The plan is re-derived from its sealed incomplete assessor input,
/// so a self-consistent substitute plan cannot redirect the next capture.
#[cfg(target_os = "macos")]
pub fn load_l0_to_l1_static_binding_from_child_preflight(
    path: &Path,
) -> Result<Qwen80SourceTokenL0ToL1StaticBinding, String> {
    let (_preflight_file, preflight) = read_json(path, "L0-to-L1 child preflight")?;
    let root = object(&preflight, "L0-to-L1 child preflight")?;
    exact_string(root, "schema", CHILD_SCHEMA, "L0-to-L1 child preflight")?;
    exact_string(root, "status", PREPARED_STATUS, "L0-to-L1 child preflight")?;
    verify_seal(&preflight, "L0-to-L1 child preflight")?;
    exact_string(
        root,
        "mode",
        "cpu_only_preflight",
        "L0-to-L1 child preflight",
    )?;
    let baseline_evidence = object_field(
        root,
        "baseline_handoff_authority",
        "L0-to-L1 child preflight",
    )?;
    let baseline_path = PathBuf::from(string_field(
        baseline_evidence,
        "path",
        "L0-to-L1 child preflight.baseline_handoff_authority",
    )?);
    let (baseline_file, baseline_value) = read_json(
        &baseline_path,
        "L0-to-L1 child preflight baseline authority",
    )?;
    if baseline_evidence.get("present").and_then(Value::as_bool) != Some(true)
        || u64_field(
            baseline_evidence,
            "bytes",
            "L0-to-L1 child preflight.baseline_handoff_authority",
        )? != baseline_file.bytes
        || sha_field(
            baseline_evidence,
            "sha256",
            "L0-to-L1 child preflight.baseline_handoff_authority",
        )? != baseline_file.sha256
        || string_field(
            root,
            "baseline_handoff_authority_seal_sha256",
            "L0-to-L1 child preflight",
        )? != verify_seal(
            &baseline_value,
            "L0-to-L1 child preflight baseline authority",
        )?
    {
        return Err(
            "L0-to-L1 child preflight baseline evidence/seal no longer matches its raw authority"
                .into(),
        );
    }
    let authority = parse_baseline_authority(baseline_file, &baseline_value)?;
    let mut expected = prepared_document(&authority);
    seal(&mut expected)?;
    if preflight != expected {
        return Err(
            "L0-to-L1 child preflight is not the exact re-derived sealed plan for its baseline authority"
                .into(),
        );
    }
    Ok(Qwen80SourceTokenL0ToL1StaticBinding::from(&authority))
}

#[cfg(target_os = "macos")]
fn read_evidenced_json(
    source: &Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<(BoundFile, Value), String> {
    let evidence = object_field(source, field, label)?;
    let path = PathBuf::from(string_field(evidence, "path", &format!("{label}.{field}"))?);
    let (file, value) = read_json(&path, &format!("{label}.{field}"))?;
    require_evidence(source, field, &file, label)?;
    Ok((file, value))
}

/// Preserve an outer preflight's immutable record of the *versioned current*
/// pointer without pretending that its raw document bytes must survive a
/// later pointer reseal.  The canonical pointer path must still exist now;
/// its live document is separately validated against the immutable chain.
#[cfg(target_os = "macos")]
fn historical_current_pointer_evidence(
    source: &Map<String, Value>,
    label: &str,
) -> Result<(BoundFile, String), String> {
    let evidence = object_field(source, "admission_current", label)?;
    if evidence.get("present").and_then(Value::as_bool) != Some(true) {
        return Err(format!("{label}.admission_current must be present"));
    }
    let historical_path = PathBuf::from(string_field(
        evidence,
        "path",
        &format!("{label}.admission_current"),
    )?);
    let path = canonical_regular(&historical_path, &format!("{label}.admission_current"))?;
    let bytes = u64_field(evidence, "bytes", &format!("{label}.admission_current"))?;
    if bytes == 0 {
        return Err(format!(
            "{label}.admission_current historical bytes must be nonzero"
        ));
    }
    let sha256 = sha_field(evidence, "sha256", &format!("{label}.admission_current"))?;
    let seal = sha_field(source, "admission_pointer_seal_sha256", label)?;
    Ok((
        BoundFile {
            path,
            bytes,
            sha256,
        },
        seal,
    ))
}

/// Validate the current mutable pointer at the moment a preflight or capture
/// begins.  Pointer bytes/seal may differ from historical outer evidence,
/// but the canonical pointer path and its immutable manifest/receipt
/// selection are exact and non-negotiable.
#[cfg(target_os = "macos")]
fn observe_versioned_current_pointer(
    path: &Path,
    manifest: &BoundFile,
    manifest_seal: &str,
    admission_receipt: &BoundFile,
    admission_receipt_seal: &str,
    label: &str,
) -> Result<VersionedCurrentPointer, String> {
    let (file, value) = read_json(path, label)?;
    let seal = verify_seal(&value, label)?;
    let pointer = object(&value, label)?;
    exact_string(pointer, "schema", ADMISSION_POINTER_SCHEMA, label)?;
    exact_string(pointer, "status", ADMISSION_POINTER_STATUS, label)?;
    let pointer_manifest = object_field(pointer, "complete_manifest", label)?;
    if string_field(
        pointer_manifest,
        "path",
        &format!("{label}.complete_manifest"),
    )? != manifest.path.to_string_lossy()
        || string_field(
            pointer_manifest,
            "document_sha256",
            &format!("{label}.complete_manifest"),
        )? != manifest.sha256
        || string_field(
            pointer_manifest,
            "seal_sha256",
            &format!("{label}.complete_manifest"),
        )? != manifest_seal
    {
        return Err(format!(
            "{label} does not select the pinned immutable manifest"
        ));
    }
    let pointer_receipt = object_field(pointer, "admission_receipt", label)?;
    if string_field(
        pointer_receipt,
        "path",
        &format!("{label}.admission_receipt"),
    )? != admission_receipt.path.to_string_lossy()
        || string_field(
            pointer_receipt,
            "document_sha256",
            &format!("{label}.admission_receipt"),
        )? != admission_receipt.sha256
        || string_field(
            pointer_receipt,
            "seal_sha256",
            &format!("{label}.admission_receipt"),
        )? != admission_receipt_seal
    {
        return Err(format!(
            "{label} does not select the pinned immutable admission receipt"
        ));
    }
    Ok(VersionedCurrentPointer { file, seal })
}

#[cfg(target_os = "macos")]
fn validate_capture_outer_preflight(args: &CaptureArgs) -> Result<CaptureAuthority, String> {
    let (outer_preflight, outer_value) =
        read_json(&args.outer_preflight, "L0 state-handoff outer preflight")?;
    let outer_seal = verify_seal(&outer_value, "L0 state-handoff outer preflight")?;
    let outer = object(&outer_value, "L0 state-handoff outer preflight")?;
    exact_string(
        outer,
        "schema",
        OUTER_PREFLIGHT_SCHEMA,
        "L0 state-handoff outer preflight",
    )?;
    exact_string(
        outer,
        "status",
        OUTER_PREFLIGHT_STATUS,
        "L0 state-handoff outer preflight",
    )?;
    let source = object_field(outer, "source_binding", "L0 state-handoff outer preflight")?;

    let (manifest, manifest_value) =
        read_evidenced_json(source, "manifest", "L0 state-handoff outer source binding")?;
    let manifest_seal = verify_seal(&manifest_value, "L0 state-handoff manifest")?;
    let manifest_object = object(&manifest_value, "L0 state-handoff manifest")?;
    exact_string(
        manifest_object,
        "schema",
        MANIFEST_SCHEMA,
        "L0 state-handoff manifest",
    )?;
    if manifest.sha256 != MANIFEST_DOCUMENT_SHA
        || manifest_seal != MANIFEST_SEAL
        || string_field(
            source,
            "manifest_seal_sha256",
            "L0 state-handoff outer source binding",
        )? != manifest_seal
    {
        return Err("L0 state-handoff manifest identity drifted".into());
    }

    let (admission_current, admission_pointer_seal) =
        historical_current_pointer_evidence(source, "L0 state-handoff outer source binding")?;

    let (admission_receipt, receipt_value) = read_evidenced_json(
        source,
        "admission_receipt",
        "L0 state-handoff outer source binding",
    )?;
    let admission_receipt_seal = verify_seal(
        &receipt_value,
        "L0 state-handoff immutable admission receipt",
    )?;
    let receipt = object(
        &receipt_value,
        "L0 state-handoff immutable admission receipt",
    )?;
    exact_string(
        receipt,
        "schema",
        ADMISSION_RECEIPT_SCHEMA,
        "L0 state-handoff immutable admission receipt",
    )?;
    exact_string(
        receipt,
        "status",
        ADMISSION_RECEIPT_STATUS,
        "L0 state-handoff immutable admission receipt",
    )?;
    if admission_receipt_seal != ADMISSION_RECEIPT_SEAL
        || string_field(
            source,
            "admission_receipt_seal_sha256",
            "L0 state-handoff outer source binding",
        )? != admission_receipt_seal
    {
        return Err("L0 state-handoff immutable admission receipt seal drifted".into());
    }
    let launch_pointer = observe_versioned_current_pointer(
        &admission_current.path,
        &manifest,
        &manifest_seal,
        &admission_receipt,
        &admission_receipt_seal,
        "L0 state-handoff current admission pointer",
    )?;
    let receipt_manifest = object_field(
        receipt,
        "complete_manifest",
        "L0 state-handoff immutable admission receipt",
    )?;
    if string_field(
        receipt_manifest,
        "document_sha256",
        "L0 state-handoff immutable admission receipt.complete_manifest",
    )? != manifest.sha256
        || string_field(
            receipt_manifest,
            "seal_sha256",
            "L0 state-handoff immutable admission receipt.complete_manifest",
        )? != manifest_seal
    {
        return Err("L0 state-handoff receipt manifest identity drifted".into());
    }
    let model = object_field(
        receipt,
        "model",
        "L0 state-handoff immutable admission receipt",
    )?;
    exact_string(
        model,
        "key",
        MODEL_KEY,
        "L0 state-handoff immutable admission receipt.model",
    )?;
    let source_revision = string_field(
        model,
        "revision",
        "L0 state-handoff immutable admission receipt.model",
    )?
    .to_owned();
    if source_revision != SOURCE_REVISION
        || string_field(
            source,
            "source_revision",
            "L0 state-handoff outer source binding",
        )? != source_revision
    {
        return Err("L0 state-handoff source revision drifted".into());
    }
    let source_audit_seal = string_field(
        source,
        "source_audit_seal_sha256",
        "L0 state-handoff outer source binding",
    )?
    .to_owned();
    if !is_sha256(&source_audit_seal) {
        return Err("L0 state-handoff source audit seal is malformed".into());
    }
    let revalidation = object_field(
        receipt,
        "current_source_revalidation",
        "L0 state-handoff immutable admission receipt",
    )?;
    if string_field(
        revalidation,
        "source_audit_seal_sha256",
        "L0 state-handoff immutable admission receipt.current_source_revalidation",
    )? != source_audit_seal
    {
        return Err("L0 state-handoff source audit seal differs from immutable admission".into());
    }

    let (source_all_ten_outer_preflight, source_all_ten_outer_value) = read_evidenced_json(
        source,
        "source_all_ten_outer_preflight",
        "L0 state-handoff outer source binding",
    )?;
    let source_all_ten_outer_preflight_seal = verify_seal(
        &source_all_ten_outer_value,
        "source-token all-ten antecedent outer preflight",
    )?;
    let source_all_ten_outer = object(
        &source_all_ten_outer_value,
        "source-token all-ten antecedent outer preflight",
    )?;
    exact_string(
        source_all_ten_outer,
        "schema",
        SOURCE_ALL_TEN_OUTER_PREFLIGHT_SCHEMA,
        "source-token all-ten antecedent outer preflight",
    )?;
    exact_string(
        source_all_ten_outer,
        "status",
        SOURCE_ALL_TEN_OUTER_PREFLIGHT_STATUS,
        "source-token all-ten antecedent outer preflight",
    )?;
    if string_field(
        source,
        "source_all_ten_outer_preflight_seal_sha256",
        "L0 state-handoff outer source binding",
    )? != source_all_ten_outer_preflight_seal
    {
        return Err("L0 state-handoff source all-ten outer preflight seal drifted".into());
    }

    let (child_preflight, _child_value) = read_evidenced_json(
        source,
        "l0_state_handoff_child_preflight",
        "L0 state-handoff outer source binding",
    )?;
    let child_preflight_seal = verify_seal(&_child_value, "L0 state-handoff child preflight")?;
    if string_field(
        source,
        "l0_state_handoff_child_preflight_seal_sha256",
        "L0 state-handoff outer source binding",
    )? != child_preflight_seal
    {
        return Err("L0 state-handoff child preflight seal drifted".into());
    }
    let static_binding = load_l0_to_l1_static_binding_from_child_preflight(&child_preflight.path)?;

    let (baseline_handoff_authority, baseline_value) = read_evidenced_json(
        source,
        "baseline_l0_to_l1_handoff_authority",
        "L0 state-handoff outer source binding",
    )?;
    let baseline_handoff_authority_seal = verify_seal(
        &baseline_value,
        "L0 state-handoff baseline L0-to-L1 authority",
    )?;
    if string_field(
        source,
        "baseline_l0_to_l1_handoff_authority_seal_sha256",
        "L0 state-handoff outer source binding",
    )? != baseline_handoff_authority_seal
    {
        return Err("L0 state-handoff baseline authority seal drifted".into());
    }
    let baseline = parse_baseline_authority(baseline_handoff_authority.clone(), &baseline_value)?;
    if baseline.seal != baseline_handoff_authority_seal
        || static_binding.session_id != baseline.session_id
        || static_binding.l1_active_conv_allocation != baseline.l1_active_conv_allocation
        || static_binding.l1_active_recurrent_allocation != baseline.l1_active_recurrent_allocation
    {
        return Err("L0 state-handoff child preflight/static binding does not derive from its exact baseline authority".into());
    }

    let contract = object_field(
        outer,
        "handoff_contract",
        "L0 state-handoff outer preflight",
    )?;
    if u64_field(
        contract,
        "source_token_id",
        "L0 state-handoff outer preflight.handoff_contract",
    )? != SOURCE_TOKEN_ID
        || u64_field(
            contract,
            "prefix_dispatches",
            "L0 state-handoff outer preflight.handoff_contract",
        )? != PREFIX_DISPATCHES
        || u64_field(
            contract,
            "suffix_dispatches",
            "L0 state-handoff outer preflight.handoff_contract",
        )? != SUFFIX_DISPATCHES
        || u64_field(
            contract,
            "total_dispatches",
            "L0 state-handoff outer preflight.handoff_contract",
        )? != TOTAL_DISPATCHES
        || u64_field(
            contract,
            "l1_prefix_dispatches",
            "L0 state-handoff outer preflight.handoff_contract",
        )? != 0
    {
        return Err("L0 state-handoff outer preflight dispatch geometry drifted".into());
    }
    bool_field(
        contract,
        "l1_binding_not_executed",
        true,
        "L0 state-handoff outer preflight.handoff_contract",
    )?;
    let claim = object_field(outer, "claim_boundary", "L0 state-handoff outer preflight")?;
    for field in [
        "metal_device_or_dispatch_performed",
        "lease_issued",
        "l1_prefix_executed",
        "complete_layer_or_token_performed",
    ] {
        bool_field(
            claim,
            field,
            false,
            "L0 state-handoff outer preflight.claim_boundary",
        )?;
    }

    Ok(CaptureAuthority {
        args: args.clone(),
        outer_preflight,
        outer_preflight_seal: outer_seal,
        manifest,
        manifest_seal,
        admission_current,
        admission_pointer_seal,
        launch_admission_current: launch_pointer.file,
        launch_admission_pointer_seal: launch_pointer.seal,
        admission_receipt,
        admission_receipt_seal,
        source_audit_seal,
        source_revision,
        source_all_ten_outer_preflight,
        source_all_ten_outer_preflight_seal,
        child_preflight,
        child_preflight_seal,
        baseline_handoff_authority,
        baseline_handoff_authority_seal,
        baseline_second_residual_f32le_sha256: baseline.second_residual_sha256,
        static_binding,
        lease: None,
        lease_id: None,
        outer_launch_authority: None,
    })
}

#[cfg(target_os = "macos")]
fn capture_preflight_document(authority: &CaptureAuthority) -> Value {
    json!({
        "schema": PRE_L1_HANDOFF_CAPTURE_SCHEMA,
        "status": "PREPARED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_CHILD_NOT_LEASED_OR_EXECUTED",
        "mode": "preflight",
        "outer_preflight_binding": {
            "path": authority.outer_preflight.path,
            "document_sha256": authority.outer_preflight.sha256,
            "seal_sha256": authority.outer_preflight_seal,
        },
        "admission_current_binding": {
            "path": authority.admission_current.path,
            "document_sha256": authority.admission_current.sha256,
            "pointer_seal_sha256": authority.admission_pointer_seal,
            "immutable_admission_receipt": {
                "path": authority.admission_receipt.path,
                "document_sha256": authority.admission_receipt.sha256,
                "seal_sha256": authority.admission_receipt_seal,
            },
        },
        "versioned_current_pointer_observation": {
            "historical_outer_preflight": {
                "path": authority.admission_current.path,
                "document_sha256": authority.admission_current.sha256,
                "seal_sha256": authority.admission_pointer_seal,
            },
            "validated_at_child_start": {
                "path": authority.launch_admission_current.path,
                "document_sha256": authority.launch_admission_current.sha256,
                "seal_sha256": authority.launch_admission_pointer_seal,
            },
            "canonical_path_required": true,
            "immutable_manifest_and_admission_receipt_must_remain_exact": true,
            "pointer_reseal_accepted_only_as_versioned_current_housekeeping": true,
        },
        "source_all_ten_outer_preflight_binding": {
            "path": authority.source_all_ten_outer_preflight.path,
            "document_sha256": authority.source_all_ten_outer_preflight.sha256,
            "seal_sha256": authority.source_all_ten_outer_preflight_seal,
        },
        "l0_state_handoff_child_preflight_binding": {
            "path": authority.child_preflight.path,
            "document_sha256": authority.child_preflight.sha256,
            "seal_sha256": authority.child_preflight_seal,
        },
        "baseline_l0_to_l1_handoff_authority_binding": {
            "path": authority.baseline_handoff_authority.path,
            "document_sha256": authority.baseline_handoff_authority.sha256,
            "seal_sha256": authority.baseline_handoff_authority_seal,
        },
        "same_command_graph_contract": {
            "source_token_id": SOURCE_TOKEN_ID,
            "prefix_dispatches": PREFIX_DISPATCHES,
            "suffix_dispatches": SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "l1_prefix_dispatches": 0,
            "l1_binding_not_executed": true,
            "retained_l0_second_residual_elements": HIDDEN,
            "retained_l0_second_residual_bytes": HIDDEN_BYTES,
            "l0_slot": 0,
            "l1_slot": 1,
        },
        "claim_boundary": {
            "metal_device_or_dispatch_performed": false,
            "lease_issued": false,
            "l1_prefix_executed": false,
            "complete_layer_or_token_performed": false,
            "cannot_satisfy_next_layer_execution_dependency": true,
            "no_decoder_generation_server_hcli_tps_tg_or_tournament_claim": true,
        },
    })
}

/// Validate a reaped CPU child preflight proof while treating its current
/// pointer observation as historical/versioned evidence.  All stable fields
/// must equal the freshly rebuilt contract; only the mutable pointer raw
/// bytes/seal may differ after a later housekeeping reseal.
#[cfg(target_os = "macos")]
fn validate_reaped_preflight_document(
    parsed: &Value,
    authority: &CaptureAuthority,
) -> Result<(), String> {
    let parsed_root = object(parsed, "L0 state-handoff reaped child preflight")?;
    exact_string(
        parsed_root,
        "schema",
        PRE_L1_HANDOFF_CAPTURE_SCHEMA,
        "L0 state-handoff reaped child preflight",
    )?;
    exact_string(
        parsed_root,
        "status",
        "PREPARED_QWEN80_SOURCE_TOKEN_L0_POST_STATE_ROLLBACK_RETAINED_OUTPUT_L1_BINDING_NOT_EXECUTED_CHILD_NOT_LEASED_OR_EXECUTED",
        "L0 state-handoff reaped child preflight",
    )?;
    exact_string(
        parsed_root,
        "mode",
        "preflight",
        "L0 state-handoff reaped child preflight",
    )?;
    let observation = object_field(
        parsed_root,
        "versioned_current_pointer_observation",
        "L0 state-handoff reaped child preflight",
    )?;
    let historical = object_field(
        observation,
        "historical_outer_preflight",
        "L0 state-handoff reaped child preflight.versioned_current_pointer_observation",
    )?;
    if string_field(
        historical,
        "path",
        "L0 state-handoff reaped child preflight historical pointer",
    )? != authority.admission_current.path.to_string_lossy()
        || sha_field(
            historical,
            "document_sha256",
            "L0 state-handoff reaped child preflight historical pointer",
        )? != authority.admission_current.sha256
        || sha_field(
            historical,
            "seal_sha256",
            "L0 state-handoff reaped child preflight historical pointer",
        )? != authority.admission_pointer_seal
    {
        return Err("L0 state-handoff reaped child historical pointer drifted".into());
    }
    let observed = object_field(
        observation,
        "validated_at_child_start",
        "L0 state-handoff reaped child preflight.versioned_current_pointer_observation",
    )?;
    if string_field(
        observed,
        "path",
        "L0 state-handoff reaped child preflight current pointer",
    )? != authority.admission_current.path.to_string_lossy()
        || !is_sha256(string_field(
            observed,
            "document_sha256",
            "L0 state-handoff reaped child preflight current pointer",
        )?)
        || !is_sha256(string_field(
            observed,
            "seal_sha256",
            "L0 state-handoff reaped child preflight current pointer",
        )?)
    {
        return Err(
            "L0 state-handoff reaped child current pointer observation is malformed".into(),
        );
    }
    for field in [
        "canonical_path_required",
        "immutable_manifest_and_admission_receipt_must_remain_exact",
        "pointer_reseal_accepted_only_as_versioned_current_housekeeping",
    ] {
        bool_field(
            observation,
            field,
            true,
            "L0 state-handoff reaped child preflight.versioned_current_pointer_observation",
        )?;
    }
    let mut stable_parsed = parsed.clone();
    stable_parsed
        .as_object_mut()
        .expect("reaped child preflight object")
        .remove("versioned_current_pointer_observation");
    let mut stable_expected = capture_preflight_document(authority);
    stable_expected
        .as_object_mut()
        .expect("expected child preflight object")
        .remove("versioned_current_pointer_observation");
    if stable_parsed != stable_expected {
        return Err(
            "L0 state-handoff reaped child preflight differs outside permitted versioned-current pointer evidence"
                .into(),
        );
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn validate_component_lease(
    value: &Value,
    authority: &CaptureAuthority,
) -> Result<(String, String), String> {
    let seal = verify_seal(value, "L0 state-handoff component lease")?;
    let lease = object(value, "L0 state-handoff component lease")?;
    exact_string(
        lease,
        "schema",
        LEASE_SCHEMA,
        "L0 state-handoff component lease",
    )?;
    exact_string(
        lease,
        "status",
        LEASE_STATUS,
        "L0 state-handoff component lease",
    )?;
    let lease_id = sha_field(lease, "lease_id", "L0 state-handoff component lease")?;
    require_evidence(
        lease,
        "outer_preflight",
        &authority.outer_preflight,
        "L0 state-handoff component lease",
    )?;
    if string_field(
        lease,
        "outer_preflight_seal_sha256",
        "L0 state-handoff component lease",
    )? != authority.outer_preflight_seal
    {
        return Err("L0 state-handoff lease outer preflight seal drifted".into());
    }
    require_evidence(
        lease,
        "l0_state_handoff_child_preflight",
        &authority.child_preflight,
        "L0 state-handoff component lease",
    )?;
    if string_field(
        lease,
        "l0_state_handoff_child_preflight_seal_sha256",
        "L0 state-handoff component lease",
    )? != authority.child_preflight_seal
    {
        return Err("L0 state-handoff lease child preflight seal drifted".into());
    }
    require_evidence(
        lease,
        "baseline_l0_to_l1_handoff_authority",
        &authority.baseline_handoff_authority,
        "L0 state-handoff component lease",
    )?;
    if string_field(
        lease,
        "baseline_l0_to_l1_handoff_authority_seal_sha256",
        "L0 state-handoff component lease",
    )? != authority.baseline_handoff_authority_seal
    {
        return Err("L0 state-handoff lease baseline authority seal drifted".into());
    }
    let contract = object_field(
        lease,
        "handoff_contract",
        "L0 state-handoff component lease",
    )?;
    if u64_field(
        contract,
        "source_token_id",
        "L0 state-handoff component lease.handoff_contract",
    )? != SOURCE_TOKEN_ID
        || u64_field(
            contract,
            "prefix_dispatches",
            "L0 state-handoff component lease.handoff_contract",
        )? != PREFIX_DISPATCHES
        || u64_field(
            contract,
            "suffix_dispatches",
            "L0 state-handoff component lease.handoff_contract",
        )? != SUFFIX_DISPATCHES
        || u64_field(
            contract,
            "total_dispatches",
            "L0 state-handoff component lease.handoff_contract",
        )? != TOTAL_DISPATCHES
        || u64_field(
            contract,
            "l1_prefix_dispatches",
            "L0 state-handoff component lease.handoff_contract",
        )? != 0
    {
        return Err("L0 state-handoff lease command graph contract drifted".into());
    }
    bool_field(
        contract,
        "l1_binding_not_executed",
        true,
        "L0 state-handoff component lease.handoff_contract",
    )?;
    let policy = object_field(
        lease,
        "execution_policy",
        "L0 state-handoff component lease",
    )?;
    exact_string(
        policy,
        "component",
        "qwen80_source_token_l0_state_handoff",
        "L0 state-handoff component lease.execution_policy",
    )?;
    for field in ["quiet_qwen80_device_lease", "strict_math"] {
        bool_field(
            policy,
            field,
            true,
            "L0 state-handoff component lease.execution_policy",
        )?;
    }
    for field in [
        "timing_or_benchmarking_allowed",
        "l1_prefix_execution_allowed",
        "complete_layer_or_token_allowed",
        "tps_or_tg_claim_allowed",
    ] {
        bool_field(
            policy,
            field,
            false,
            "L0 state-handoff component lease.execution_policy",
        )?;
    }
    let lifecycle = object_field(lease, "lifecycle", "L0 state-handoff component lease")?;
    for field in [
        "fresh_for_this_exact_launch",
        "outer_reaped_capture_required",
        "lease_released_after_first_terminal_child",
        "automatic_retry_prohibited",
        "replay_guarded",
    ] {
        bool_field(
            lifecycle,
            field,
            true,
            "L0 state-handoff component lease.lifecycle",
        )?;
    }
    let watcher = object_field(
        lease,
        "watcher_coordination",
        "L0 state-handoff component lease",
    )?;
    bool_field(
        watcher,
        "watcher_hold_must_remain_active",
        true,
        "L0 state-handoff component lease.watcher_coordination",
    )?;
    bool_field(
        watcher,
        "watcher_restart_or_transition_authorized",
        false,
        "L0 state-handoff component lease.watcher_coordination",
    )?;
    Ok((seal, lease_id))
}

#[cfg(target_os = "macos")]
fn validate_outer_launch_authority(
    value: &Value,
    authority: &CaptureAuthority,
) -> Result<String, String> {
    let seal = verify_seal(value, "L0 state-handoff outer launch authority")?;
    let launch = object(value, "L0 state-handoff outer launch authority")?;
    exact_string(
        launch,
        "schema",
        OUTER_LAUNCH_SCHEMA,
        "L0 state-handoff outer launch authority",
    )?;
    exact_string(
        launch,
        "status",
        OUTER_LAUNCH_STATUS,
        "L0 state-handoff outer launch authority",
    )?;
    let lease = authority
        .lease
        .as_ref()
        .ok_or("L0 state-handoff launch authority requires a validated lease")?;
    let lease_id = authority
        .lease_id
        .as_deref()
        .ok_or("L0 state-handoff launch authority lacks a lease ID")?;
    require_evidence(
        launch,
        "lease_receipt",
        &lease.0,
        "L0 state-handoff outer launch authority",
    )?;
    if string_field(
        launch,
        "lease_receipt_seal_sha256",
        "L0 state-handoff outer launch authority",
    )? != lease.1
        || string_field(
            launch,
            "lease_id",
            "L0 state-handoff outer launch authority",
        )? != lease_id
    {
        return Err("L0 state-handoff launch authority lease lineage drifted".into());
    }
    require_evidence(
        launch,
        "outer_preflight",
        &authority.outer_preflight,
        "L0 state-handoff outer launch authority",
    )?;
    if string_field(
        launch,
        "outer_preflight_seal_sha256",
        "L0 state-handoff outer launch authority",
    )? != authority.outer_preflight_seal
    {
        return Err("L0 state-handoff launch authority outer-preflight seal drifted".into());
    }
    for (field, expected, expected_seal) in [
        (
            "l0_state_handoff_child_preflight",
            &authority.child_preflight,
            &authority.child_preflight_seal,
        ),
        (
            "baseline_l0_to_l1_handoff_authority",
            &authority.baseline_handoff_authority,
            &authority.baseline_handoff_authority_seal,
        ),
    ] {
        require_evidence(
            launch,
            field,
            expected,
            "L0 state-handoff outer launch authority",
        )?;
        if string_field(
            launch,
            &format!("{field}_seal_sha256"),
            "L0 state-handoff outer launch authority",
        )? != expected_seal
        {
            return Err(format!(
                "L0 state-handoff launch authority {field} seal drifted"
            ));
        }
    }
    let executable = std::env::current_exe()
        .map_err(|error| format!("cannot resolve L0 state-handoff child executable: {error}"))?;
    let executable = file_evidence(&executable, "current L0 state-handoff child executable")?;
    require_evidence(
        launch,
        "probe_binary",
        &executable,
        "L0 state-handoff outer launch authority",
    )?;
    let proof_evidence = object_field(
        launch,
        "preflight_proof",
        "L0 state-handoff outer launch authority",
    )?;
    let proof_path = PathBuf::from(string_field(
        proof_evidence,
        "path",
        "L0 state-handoff outer launch authority.preflight_proof",
    )?);
    let (proof_file, proof_value) = read_json(
        &proof_path,
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    require_evidence(
        launch,
        "preflight_proof",
        &proof_file,
        "L0 state-handoff outer launch authority",
    )?;
    let proof_seal = verify_seal(
        &proof_value,
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    if string_field(
        launch,
        "preflight_proof_seal_sha256",
        "L0 state-handoff outer launch authority",
    )? != proof_seal
    {
        return Err("L0 state-handoff launch authority preflight proof seal drifted".into());
    }
    let child_binding = object_field(
        launch,
        "child_preflight_proof_binding",
        "L0 state-handoff outer launch authority",
    )?;
    if string_field(
        child_binding,
        "path",
        "L0 state-handoff outer launch authority.child_preflight_proof_binding",
    )? != proof_file.path.to_string_lossy()
        || sha_field(
            child_binding,
            "document_sha256",
            "L0 state-handoff outer launch authority.child_preflight_proof_binding",
        )? != proof_file.sha256
        || sha_field(
            child_binding,
            "seal_sha256",
            "L0 state-handoff outer launch authority.child_preflight_proof_binding",
        )? != proof_seal
    {
        return Err(
            "L0 state-handoff launch authority child-preflight proof binding drifted".into(),
        );
    }
    let proof = object(
        &proof_value,
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    exact_string(
        proof,
        "schema",
        OUTER_PREFLIGHT_PROOF_SCHEMA,
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    exact_string(
        proof,
        "status",
        OUTER_PREFLIGHT_PROOF_STATUS,
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    let proof_source = object_field(
        proof,
        "source_binding",
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    require_evidence(
        proof_source,
        "probe_binary",
        &executable,
        "L0 state-handoff outer launch authority preflight proof.source_binding",
    )?;
    let proof_outer = object_field(
        proof_source,
        "outer_preflight",
        "L0 state-handoff outer launch authority preflight proof.source_binding",
    )?;
    if string_field(
        proof_outer,
        "path",
        "L0 state-handoff outer launch authority preflight proof.source_binding.outer_preflight",
    )? != authority.outer_preflight.path.to_string_lossy()
        || sha_field(
            proof_outer,
            "sha256",
            "L0 state-handoff outer launch authority preflight proof.source_binding.outer_preflight",
        )? != authority.outer_preflight.sha256
        || sha_field(
            proof_outer,
            "seal_sha256",
            "L0 state-handoff outer launch authority preflight proof.source_binding.outer_preflight",
        )? != authority.outer_preflight_seal
    {
        return Err("L0 state-handoff proof does not bind the current outer preflight".into());
    }
    let child = object_field(
        proof,
        "child_preflight",
        "L0 state-handoff outer launch authority preflight proof",
    )?;
    if child.get("exit_code").and_then(Value::as_u64) != Some(0)
        || child.get("reaped").and_then(Value::as_bool) != Some(true)
        || child.get("stderr_bytes").and_then(Value::as_u64) != Some(0)
    {
        return Err("L0 state-handoff proof has no clean reaped child preflight".into());
    }
    let parsed = child
        .get("parsed")
        .ok_or("L0 state-handoff proof lacks parsed child preflight")?;
    validate_reaped_preflight_document(parsed, authority)?;
    let outer_dir = authority
        .args
        .outer_capture_dir
        .as_ref()
        .ok_or("L0 state-handoff launch authority lacks outer capture dir")?;
    let inner_dir = authority
        .args
        .capture_dir
        .as_ref()
        .ok_or("L0 state-handoff launch authority lacks inner capture dir")?;
    if string_field(
        launch,
        "planned_outer_capture_dir",
        "L0 state-handoff outer launch authority",
    )? != outer_dir.to_string_lossy()
        || string_field(
            launch,
            "planned_inner_capture_dir",
            "L0 state-handoff outer launch authority",
        )? != inner_dir.to_string_lossy()
        || u64_field(launch, "workers", "L0 state-handoff outer launch authority")?
            != authority.args.workers as u64
    {
        return Err("L0 state-handoff launch authority planned path/worker drifted".into());
    }
    let policy = object_field(
        launch,
        "execution_policy",
        "L0 state-handoff outer launch authority",
    )?;
    for field in [
        "quiet_qwen80_device_lease",
        "strict_math",
        "outer_reaped_capture_required",
    ] {
        bool_field(
            policy,
            field,
            true,
            "L0 state-handoff outer launch authority.execution_policy",
        )?;
    }
    for field in [
        "timing_or_benchmarking_allowed",
        "l1_prefix_execution_allowed",
        "complete_layer_or_token_allowed",
        "tps_or_tg_claim_allowed",
        "automatic_retry_allowed",
    ] {
        bool_field(
            policy,
            field,
            false,
            "L0 state-handoff outer launch authority.execution_policy",
        )?;
    }
    let lifecycle = object_field(
        launch,
        "lifecycle",
        "L0 state-handoff outer launch authority",
    )?;
    for field in [
        "one_shot",
        "receipt_last",
        "replay_guarded",
        "lease_release_required_on_every_terminal_outcome",
    ] {
        bool_field(
            lifecycle,
            field,
            true,
            "L0 state-handoff outer launch authority.lifecycle",
        )?;
    }
    let watcher = object_field(
        launch,
        "watcher_coordination",
        "L0 state-handoff outer launch authority",
    )?;
    bool_field(
        watcher,
        "watcher_hold_must_remain_active",
        true,
        "L0 state-handoff outer launch authority.watcher_coordination",
    )?;
    bool_field(
        watcher,
        "watcher_restart_or_transition_authorized",
        false,
        "L0 state-handoff outer launch authority.watcher_coordination",
    )?;
    Ok(seal)
}

#[cfg(target_os = "macos")]
fn validate_capture_authority(args: &CaptureArgs) -> Result<CaptureAuthority, String> {
    let provisional = validate_capture_outer_preflight(args)?;
    let (lease, lease_id) = match args.mode {
        CaptureMode::Preflight => (None, None),
        CaptureMode::Metal => {
            let path = args
                .lease_receipt
                .as_ref()
                .ok_or("L0 state-handoff metal mode lacks lease receipt")?;
            let (evidence, value) = read_json(path, "L0 state-handoff component lease")?;
            let (seal, lease_id) = validate_component_lease(&value, &provisional)?;
            (Some((evidence, seal)), Some(lease_id))
        }
    };
    let leased = CaptureAuthority {
        lease,
        lease_id,
        ..provisional
    };
    let outer_launch_authority = match args.mode {
        CaptureMode::Preflight => None,
        CaptureMode::Metal => {
            let path = args
                .outer_launch_authority
                .as_ref()
                .ok_or("L0 state-handoff metal mode lacks outer launch authority")?;
            let (evidence, value) = read_json(path, "L0 state-handoff outer launch authority")?;
            let seal = validate_outer_launch_authority(&value, &leased)?;
            Some((evidence, seal))
        }
    };
    Ok(CaptureAuthority {
        outer_launch_authority,
        ..leased
    })
}

#[cfg(target_os = "macos")]
fn verify_metal_capture_paths(authority: &CaptureAuthority) -> Result<(PathBuf, PathBuf), String> {
    let outer = authority
        .args
        .outer_capture_dir
        .as_ref()
        .ok_or("L0 state-handoff metal mode lacks outer capture directory")?;
    let inner = authority
        .args
        .capture_dir
        .as_ref()
        .ok_or("L0 state-handoff metal mode lacks inner capture directory")?;
    if !outer.is_absolute() || !inner.is_absolute() || inner.exists() {
        return Err(
            "L0 state-handoff capture paths must be absolute and the inner directory must be new"
                .into(),
        );
    }
    let outer = fs::canonicalize(outer)
        .map_err(|error| format!("cannot canonicalize L0 state-handoff outer capture: {error}"))?;
    let metadata = fs::symlink_metadata(&outer)
        .map_err(|error| format!("cannot stat L0 state-handoff outer capture: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(
            "L0 state-handoff outer capture must be an existing non-symlink directory".into(),
        );
    }
    let parent = inner
        .parent()
        .ok_or("L0 state-handoff inner capture has no parent")?;
    let parent = fs::canonicalize(parent)
        .map_err(|error| format!("cannot canonicalize L0 state-handoff inner parent: {error}"))?;
    if parent != outer {
        return Err(
            "L0 state-handoff inner capture must be a direct child of outer capture".into(),
        );
    }
    Ok((outer, inner.clone()))
}

#[cfg(target_os = "macos")]
fn run_metal(
    authority: &CaptureAuthority,
    capture: &Path,
    phase: &mut MetalExecutionPhase,
) -> Result<Value, String> {
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest_seal.clone(),
        expected_source_audit_seal_sha256: authority.source_audit_seal.clone(),
        expected_source_revision: authority.source_revision.clone(),
    };
    phase.strict_artifact_admission_started = true;
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.manifest.path, &admission)
        .map_err(|error| format!("L0 state-handoff strict artifact admission failed: {error}"))?;
    phase.strict_artifact_admission_succeeded = true;
    phase.metal_context_construction_attempted = true;
    let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
        catalog,
        Qwen80CompleteRuntimeOptions {
            max_seq_len: 1,
            trace_dispatch: true,
        },
    )
    .map_err(|error| {
        format!("L0 state-handoff strict-Math runtime construction failed: {error}")
    })?;
    phase.metal_context_constructed = true;
    let (source_bridge, lineage) =
        all_ten_source_child::build_source_token_all_ten_bridge_from_outer_preflight(
            &runtime,
            &authority.source_all_ten_outer_preflight.path,
            authority.args.workers,
        )?;
    if lineage.manifest_document_sha256 != authority.manifest.sha256
        || lineage.manifest_seal_sha256 != authority.manifest_seal
        || lineage.admission_receipt_seal_sha256 != authority.admission_receipt_seal
        || lineage.source_revision != authority.source_revision
        || lineage.source_token_id != SOURCE_TOKEN_ID as u32
    {
        return Err(
            "L0 state-handoff source-token all-ten lineage drifted from current admission".into(),
        );
    }

    let mut command = runtime.begin_component_token_command_buffer();
    command
        .enable_structural_kernel_trace()
        .map_err(|error| format!("L0 state-handoff non-timed structural trace refusal: {error}"))?;
    phase.structural_kernel_trace_enabled = true;
    let prepared = encode_source_token_l0_state_handoff(
        &runtime,
        &mut command,
        &source_bridge,
        authority.static_binding.clone(),
    )?;
    if command.dispatch_count() != TOTAL_DISPATCHES as usize
        || prepared.next_layer() != 1
        || prepared.next_linear_state_slot() != 1
        || prepared.retained_l0_second_residual().length() != HIDDEN_BYTES
    {
        return Err(
            "L0 state-handoff did not retain the exact 9+14 L0 graph/output boundary".into(),
        );
    }
    phase.dispatches_encoded = command.dispatch_count();
    phase.encoded_kernel_names = command
        .structural_kernel_names()
        .ok_or("L0 state-handoff structural kernel trace was not retained")?
        .to_vec();
    let expected_kernels = expected_kernel_order();
    if phase
        .encoded_kernel_names
        .iter()
        .map(String::as_str)
        .ne(expected_kernels.iter().copied())
    {
        return Err(format!(
            "L0 state-handoff structural 9+14 kernel order drifted: expected {expected_kernels:?}, observed {:?}",
            phase.encoded_kernel_names
        ));
    }
    phase.command_commit_attempted = true;
    command
        .commit_and_wait()
        .map_err(|error| format!("L0 state-handoff common fence failed: {error}"))?;
    phase.command_fence_succeeded = true;
    phase.readback_started = true;

    let handoff_witness = prepared.verify_after_fence(&runtime)?;
    let witness = object(&handoff_witness, "L0 state-handoff device witness")?;
    exact_string(
        witness,
        "schema",
        PRE_L1_HANDOFF_CAPTURE_SCHEMA,
        "L0 state-handoff device witness",
    )?;
    exact_string(
        witness,
        "status",
        PRE_L1_HANDOFF_CAPTURE_STATUS,
        "L0 state-handoff device witness",
    )?;
    bool_field(
        witness,
        "l1_binding_not_executed",
        true,
        "L0 state-handoff device witness",
    )?;
    if witness.get("l1_prefix_dispatches").and_then(Value::as_u64) != Some(0) {
        return Err("L0 state-handoff witness must record zero L1 dispatches".into());
    }
    let retained = object_field(
        witness,
        "retained_l0_second_residual",
        "L0 state-handoff device witness",
    )?;
    if u64_field(
        retained,
        "elements",
        "L0 state-handoff device witness.retained_l0_second_residual",
    )? != HIDDEN
        || u64_field(
            retained,
            "bytes",
            "L0 state-handoff device witness.retained_l0_second_residual",
        )? != HIDDEN_BYTES
        || sha_field(
            retained,
            "f32le_sha256",
            "L0 state-handoff device witness.retained_l0_second_residual",
        )? != authority.baseline_second_residual_f32le_sha256
        || string_field(
            retained,
            "device_buffer_id",
            "L0 state-handoff device witness.retained_l0_second_residual",
        )?
        .trim()
        .is_empty()
    {
        return Err(
            "L0 state-handoff retained output does not match the sealed L0 baseline".into(),
        );
    }
    let l1 = object_field(
        witness,
        "layer1_input_binding",
        "L0 state-handoff device witness",
    )?;
    if u64_field(
        l1,
        "layer",
        "L0 state-handoff device witness.layer1_input_binding",
    )? != 1
        || u64_field(
            l1,
            "linear_state_slot",
            "L0 state-handoff device witness.layer1_input_binding",
        )? != 1
        || l1.get("l1_binding_executed").and_then(Value::as_bool) != Some(false)
        || string_field(
            l1,
            "input_device_buffer_id",
            "L0 state-handoff device witness.layer1_input_binding",
        )? != string_field(
            retained,
            "device_buffer_id",
            "L0 state-handoff device witness.retained_l0_second_residual",
        )?
    {
        return Err("L0 state-handoff L1 binding drifted or implied execution".into());
    }

    let embedding = runtime
        .catalog()
        .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID as u32)
        .map_err(|error| format!("L0 state-handoff source embedding oracle rejected: {error}"))?;
    let cpu_input =
        hawking_core::model::qwen80_complete_runtime::Qwen80CanonicalLinearLayerCpuInput::with_zero_state(
            embedding.hidden,
        );
    let cpu = runtime
        .catalog()
        .execute_first_linear_layer_cpu_moe_oracle(&cpu_input)
        .map_err(|error| format!("L0 state-handoff source CPU oracle rejected: {error}"))?;
    let fixed = prepared.fixed_buffers();
    let postnorm = snapshot_f32(&fixed.postnorm_hidden, HIDDEN as usize, "L0 postnorm")?;
    let logits = snapshot_f32(&fixed.router_logits, 512, "L0 router logits")?;
    let route_ids = snapshot_u32(&fixed.router_route_ids, 10, "L0 route IDs")?;
    let route_weights = snapshot_f32(&fixed.router_route_weights, 10, "L0 route weights")?;
    let route_guard = snapshot_u32(&fixed.route_guard, 1, "L0 route guard")?[0];
    if route_guard != 1 || route_ids.as_slice() != lineage.route_ids {
        return Err(
            "L0 state-handoff route guard does not match its source-token route lineage".into(),
        );
    }
    for index in 0..10 {
        if (route_weights[index] - lineage.route_weights[index]).abs() > 2.0e-5 {
            return Err(format!(
                "L0 state-handoff route weight {index} differs from sealed source authority"
            ));
        }
    }
    let postnorm_error = require_parity(
        &cpu.post_attention_rms_norm_output,
        &postnorm,
        "L0 postnorm",
        2.0e-4,
    )?;
    let logits_error = require_parity(&cpu.router_logits, &logits, "L0 router logits", 5.0e-4)?;
    let route_weight_error = require_parity(
        &cpu.route.weights,
        &route_weights,
        "L0 route weights",
        2.0e-5,
    )?;
    let route_weighted = snapshot_f32(&fixed.route_weighted, 10 * HIDDEN as usize, "L0 routes")?;
    let mut route_witnesses = Vec::with_capacity(10);
    for (index, expected) in cpu.routed_experts.iter().enumerate() {
        let observed = &route_weighted[index * HIDDEN as usize..(index + 1) * HIDDEN as usize];
        let max_abs_error = require_parity(
            &expected.weighted_output,
            observed,
            &format!("L0 route {index}"),
            3.0e-4,
        )?;
        route_witnesses.push(json!({
            "wave_index": index,
            "expert_id": expected.expert,
            "normalized_weight": expected.route_weight,
            "elements": HIDDEN,
            "output_sha256": f32_sha(observed, "L0 route witness")?,
            "max_abs_error": max_abs_error,
        }));
    }
    let shared = snapshot_f32(&fixed.gated_shared, HIDDEN as usize, "L0 shared output")?;
    let routed_sum = snapshot_f32(&fixed.routed_sum, HIDDEN as usize, "L0 routed sum")?;
    let second_residual = snapshot_f32(
        &fixed.second_residual,
        HIDDEN as usize,
        "L0 second residual",
    )?;
    let shared_error = require_parity(
        &cpu.shared_gated_output,
        &shared,
        "L0 shared output",
        3.0e-4,
    )?;
    let routed_sum_error =
        require_parity(&cpu.routed_expert_sum, &routed_sum, "L0 routed sum", 3.0e-5)?;
    let second_residual_error = require_parity(
        &cpu.layer_output,
        &second_residual,
        "L0 second residual",
        3.0e-5,
    )?;
    let second_residual_sha = f32_sha(&second_residual, "L0 second residual")?;
    if second_residual_sha != authority.baseline_second_residual_f32le_sha256 {
        return Err(
            "L0 state-handoff second residual differs from its sealed baseline hash".into(),
        );
    }
    let lease = authority
        .lease
        .as_ref()
        .ok_or("L0 state-handoff success has no validated lease")?;
    let launch = authority
        .outer_launch_authority
        .as_ref()
        .ok_or("L0 state-handoff success has no validated outer launch authority")?;
    let launch_binding = json!({
        "path": launch.0.path,
        "document_sha256": launch.0.sha256,
        "seal_sha256": launch.1,
    });
    Ok(json!({
        "schema": PRE_L1_HANDOFF_CAPTURE_SCHEMA,
        "status": PRE_L1_HANDOFF_CAPTURE_STATUS,
        "mode": "metal",
        "metal_device_or_dispatch_performed": true,
        "component_only": true,
        "l1_binding_not_executed": true,
        "l1_prefix_dispatches": 0,
        "complete_layer_or_token_performed": false,
        "complete_artifact_scan_performed_once": true,
        "raw_bf16_or_safetensors_opened": false,
        "artifact_binding": {
            "manifest_document_sha256": authority.manifest.sha256,
            "manifest_seal_sha256": authority.manifest_seal,
            "admission_pointer_seal_sha256": authority.launch_admission_pointer_seal,
            "admission_receipt_seal_sha256": authority.admission_receipt_seal,
            "source_audit_seal_sha256": authority.source_audit_seal,
            "source_revision": authority.source_revision,
            "layer": 0,
            "linear_state_slot": 0,
            "native_device": runtime.device_name(),
        },
        "versioned_current_pointer_observations": {
            "historical_outer_preflight": {
                "path": authority.admission_current.path,
                "document_sha256": authority.admission_current.sha256,
                "seal_sha256": authority.admission_pointer_seal,
            },
            "validated_at_child_launch": {
                "path": authority.launch_admission_current.path,
                "document_sha256": authority.launch_admission_current.sha256,
                "seal_sha256": authority.launch_admission_pointer_seal,
            },
            "terminal_observation_pending": true,
            "canonical_path_required": true,
            "immutable_manifest_and_admission_receipt_must_remain_exact": true,
        },
        "outer_preflight_binding": {
            "path": authority.outer_preflight.path,
            "document_sha256": authority.outer_preflight.sha256,
            "seal_sha256": authority.outer_preflight_seal,
        },
        "outer_launch_authority_binding": launch_binding.clone(),
        "same_command_graph": {
            "source_token_id": SOURCE_TOKEN_ID,
            "prefix_dispatches": PREFIX_DISPATCHES,
            "suffix_dispatches": SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "same_command_graph_retained": true,
            "fenced_once_after_prefix_and_suffix": true,
            "structural_kernel_trace_non_timed": true,
            "encoded_kernel_names": phase.encoded_kernel_names,
            "expected_kernel_names": expected_kernels,
        },
        "source_all_ten_lineage": lineage,
        "l0_state_handoff": handoff_witness,
        "route_guard_readback": {
            "value": route_guard,
            "passed": true,
            "observed_ids": route_ids,
            "expected_ids": cpu.route.ids.map(u32::from),
            "observed_weights": route_weights,
            "expected_weights": cpu.route.weights,
            "weights_max_abs_error": route_weight_error,
        },
        "readback_parity": {
            "postnorm": {"elements": HIDDEN, "output_sha256": f32_sha(&postnorm, "L0 postnorm")?, "max_abs_error": postnorm_error, "tolerance": 2.0e-4},
            "router_logits": {"elements": 512, "output_sha256": f32_sha(&logits, "L0 router logits")?, "max_abs_error": logits_error, "tolerance": 5.0e-4},
            "all_ten_route_witness_count": route_witnesses.len(),
            "all_ten_route_witnesses": route_witnesses,
            "shared_expert": {"elements": HIDDEN, "output_sha256": f32_sha(&shared, "L0 shared output")?, "max_abs_error": shared_error, "tolerance": 3.0e-4},
            "routed_sum": {"elements": HIDDEN, "output_sha256": f32_sha(&routed_sum, "L0 routed sum")?, "max_abs_error": routed_sum_error, "tolerance": 3.0e-5},
            "second_residual": {"elements": HIDDEN, "output_sha256": second_residual_sha, "max_abs_error": second_residual_error, "tolerance": 3.0e-5},
        },
        "execution_phase": {
            "strict_artifact_admission_started": phase.strict_artifact_admission_started,
            "strict_artifact_admission_succeeded": phase.strict_artifact_admission_succeeded,
            "metal_context_construction_attempted": phase.metal_context_construction_attempted,
            "metal_context_constructed": phase.metal_context_constructed,
            "structural_kernel_trace_enabled": phase.structural_kernel_trace_enabled,
            "dispatches_encoded": phase.dispatches_encoded,
            "encoded_kernel_names": phase.encoded_kernel_names,
            "command_commit_attempted": phase.command_commit_attempted,
            "command_fence_succeeded": phase.command_fence_succeeded,
            "readback_started": phase.readback_started,
            "device_dispatch_may_have_occurred": phase.command_commit_attempted,
        },
        "metal_execution_policy": {
            "strict_math_required": true,
            "timing_or_benchmarking_allowed": false,
            "l1_prefix_execution_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "lease_binding": {"path": lease.0.path, "document_sha256": lease.0.sha256, "seal_sha256": lease.1, "lease_id": authority.lease_id},
            "outer_launch_authority_binding": launch_binding.clone(),
        },
        "outer_reaper_binding": {"lease_id": authority.lease_id, "outer_launch_authority": launch_binding.clone()},
        "durable_capture": {
            "capture_directory": capture,
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_capture_required": true,
            "replay_guarded": true,
            "outer_reaper_binding": {"lease_id": authority.lease_id, "outer_launch_authority": launch_binding.clone()},
        },
        "claim_boundary": {
            "l0_post_state_rollback_retained_output_component_only": true,
            "l1_binding_not_executed": true,
            "may_not_satisfy_next_layer_execution_dependency": true,
            "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": true,
            "no_watcher_or_server_started": true,
        },
    }))
}

#[cfg(target_os = "macos")]
fn f32_sha(values: &[f32], label: &str) -> Result<String, String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} is empty/non-finite"));
    }
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_bits().to_le_bytes());
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(target_os = "macos")]
fn buffer_identity(buffer: &PinnedBuffer, label: &str) -> Result<String, String> {
    let bytes = buffer.length();
    let contents = buffer.contents() as usize;
    if bytes == 0 || contents == 0 {
        return Err(format!("{label} lacks a live shared device allocation"));
    }
    Ok(sha256_hex(
        format!("qwen80-l0-to-l1-live-buffer-v1:{label}:{contents:x}:{bytes}").as_bytes(),
    ))
}

/// A live, typed L1 input reference. It aliases the retained L0
/// second-residual allocation; it is not a CPU copy and it does not encode
/// Layer 1. The caller must retain this holder until its later Layer-1
/// encoder has appended work to the same session command graph.
#[cfg(target_os = "macos")]
pub struct Qwen80SourceTokenLayer1InputBinding {
    retained_input: PinnedBuffer,
    linear_state_slot: usize,
    active_conv_state_buffer_identity_sha256: String,
    active_recurrent_state_buffer_identity_sha256: String,
}

#[cfg(target_os = "macos")]
impl Qwen80SourceTokenLayer1InputBinding {
    pub fn input(&self) -> &PinnedBuffer {
        &self.retained_input
    }

    pub fn linear_state_slot(&self) -> usize {
        self.linear_state_slot
    }

    pub fn active_conv_state_buffer_identity_sha256(&self) -> &str {
        &self.active_conv_state_buffer_identity_sha256
    }

    pub fn active_recurrent_state_buffer_identity_sha256(&self) -> &str {
        &self.active_recurrent_state_buffer_identity_sha256
    }
}

/// A live same-runtime resource holder for one *future* L0 handoff capture.
/// It keeps the exact second-residual allocation alive with every prefix and
/// suffix allocation until the caller has both fenced the TCB and passed the
/// output directly into a Layer-1 encoder. There is intentionally no CPU
/// vector substitution path.
#[cfg(target_os = "macos")]
pub struct Qwen80SourceTokenL0ToL1PreparedHandoff {
    resources: source_prefix::Qwen80SourceInputL0TrueMoeCaptureResources,
    retained_l0_second_residual: PinnedBuffer,
    layer1_input: Qwen80SourceTokenLayer1InputBinding,
    static_binding: Qwen80SourceTokenL0ToL1StaticBinding,
}

#[cfg(target_os = "macos")]
impl Qwen80SourceTokenL0ToL1PreparedHandoff {
    pub fn retained_l0_second_residual(&self) -> &PinnedBuffer {
        &self.retained_l0_second_residual
    }

    /// The suffix buffers are retained by this handoff owner until the one
    /// shared fence. Callers may only snapshot them after that fence for
    /// parity; they may not replace the retained second residual with CPU
    /// bytes.
    pub fn fixed_buffers(&self) -> &Qwen80L0TrueMoeFixedDeviceBuffers {
        &self.resources.fixed
    }

    pub fn next_layer(&self) -> usize {
        1
    }

    pub fn next_linear_state_slot(&self) -> usize {
        1
    }

    /// This typed alias is created before the common L0 command buffer is
    /// fenced and stays alive with every L0 resource. It binds a future
    /// Layer-1 encoder without claiming Layer 1 was dispatched.
    pub fn layer1_input(&self) -> &Qwen80SourceTokenLayer1InputBinding {
        &self.layer1_input
    }

    /// Must be called only after the caller fences the one TCB that contains
    /// all 9+14 L0 dispatches. The returned metadata is intentionally still a
    /// component witness; it does not execute Layer 1 or a full token.
    pub fn verify_after_fence(
        &self,
        runtime: &Qwen80CompleteNativeRuntime,
    ) -> Result<Value, String> {
        let prefix = self
            .resources
            .graph
            .verify_first_residual_after_fence(runtime)
            .map_err(|error| error.to_string())?;
        let output = snapshot_f32(
            &self.retained_l0_second_residual,
            HIDDEN as usize,
            "retained L0 second residual",
        )?;
        let output_sha = f32_sha(&output, "retained L0 second residual")?;
        if prefix.linear_state_slot != 0
            || prefix.linear_conv_state_elements != 24_576
            || prefix.linear_recurrent_state_elements != 524_288
            || prefix.linear_conv_state_bytes != L0_CONV_BYTES as usize
            || prefix.linear_recurrent_state_bytes != L0_RECURRENT_BYTES as usize
        {
            return Err("L0 state witness geometry drifted".into());
        }
        let layer1 = self.layer1_input();
        let output_buffer_id =
            buffer_identity(&self.retained_l0_second_residual, "l0-second-residual")?;
        if buffer_identity(layer1.input(), "l0-second-residual")? != output_buffer_id {
            return Err(
                "Layer-1 input no longer aliases the retained L0 second-residual allocation".into(),
            );
        }
        Ok(json!({
            "schema": PRE_L1_HANDOFF_CAPTURE_SCHEMA,
            "status": PRE_L1_HANDOFF_CAPTURE_STATUS,
            "session_id": self.static_binding.session_id,
            "source_token_id": SOURCE_TOKEN_ID,
            "same_command_graph_retained": true,
            "l1_binding_not_executed": true,
            "l1_prefix_dispatches": 0,
            "retained_l0_second_residual": {
                "elements": HIDDEN,
                "bytes": HIDDEN_BYTES,
                "f32le_sha256": output_sha,
                "device_buffer_id": output_buffer_id,
                "retained_for_future_layer1_encode": true,
            },
            "l0_post_state_commit": {
                "layer": 0,
                "linear_state_slot": 0,
                "checkpoint_before_mutation": true,
                "active_conv": {
                    "allocation_id": self.static_binding.l0_active_conv_allocation,
                    "slot": 0,
                    "offset_bytes": 0,
                    "capacity_bytes": L0_CONV_BYTES,
                    "device_buffer_id": prefix.active_conv_state_buffer_identity_sha256,
                    "post_state_f32le_sha256": prefix.device_post_conv_state_f32le_sha256,
                },
                "active_recurrent": {
                    "allocation_id": self.static_binding.l0_active_recurrent_allocation,
                    "slot": 0,
                    "offset_bytes": 0,
                    "capacity_bytes": L0_RECURRENT_BYTES,
                    "device_buffer_id": prefix.active_recurrent_state_buffer_identity_sha256,
                    "post_state_f32le_sha256": prefix.device_post_recurrent_state_f32le_sha256,
                },
                "rollback_conv": {
                    "allocation_id": self.static_binding.l0_rollback_conv_allocation,
                    "slot": 0,
                    "offset_bytes": 0,
                    "capacity_bytes": L0_CONV_BYTES,
                    "device_buffer_id": prefix.rollback_conv_state_buffer_identity_sha256,
                    "checkpoint_f32le_sha256": prefix.rollback_conv_state_f32le_sha256,
                },
                "rollback_recurrent": {
                    "allocation_id": self.static_binding.l0_rollback_recurrent_allocation,
                    "slot": 0,
                    "offset_bytes": 0,
                    "capacity_bytes": L0_RECURRENT_BYTES,
                    "device_buffer_id": prefix.rollback_recurrent_state_buffer_identity_sha256,
                    "checkpoint_f32le_sha256": prefix.rollback_recurrent_state_f32le_sha256,
                },
            },
            "layer1_input_binding": {
                "session_id": self.static_binding.session_id,
                "layer": self.next_layer(),
                "linear_state_slot": layer1.linear_state_slot(),
                "input_device_buffer_id": output_buffer_id,
                "input_f32le_sha256": output_sha,
                "same_command_graph_retained": true,
                "l1_binding_executed": false,
                "active_conv": {
                    "allocation_id": self.static_binding.l1_active_conv_allocation,
                    "slot": 1,
                    "offset_bytes": L1_CONV_OFFSET_BYTES,
                    "capacity_bytes": L1_CONV_OFFSET_BYTES + L0_CONV_BYTES,
                    "device_buffer_id": layer1.active_conv_state_buffer_identity_sha256(),
                    "device_buffer_identity_sha256": layer1.active_conv_state_buffer_identity_sha256(),
                },
                "active_recurrent": {
                    "allocation_id": self.static_binding.l1_active_recurrent_allocation,
                    "slot": 1,
                    "offset_bytes": L1_RECURRENT_OFFSET_BYTES,
                    "capacity_bytes": L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES,
                    "device_buffer_id": layer1.active_recurrent_state_buffer_identity_sha256(),
                    "device_buffer_identity_sha256": layer1.active_recurrent_state_buffer_identity_sha256(),
                },
            },
            "l0_prefix_parity": prefix,
            "claim_boundary": {
                "component_only": true,
                "layer1_not_encoded": true,
                "retention_binding_is_not_a_layer1_execution_claim": true,
                "may_not_satisfy_next_layer_execution_dependency": true,
            },
        }))
    }
}

/// Construct the typed L1 input alias before the shared L0 token command
/// buffer fences. This validates the exact source schedule and active state
/// arena slice, but deliberately does not encode a Layer-1 operator.
#[cfg(target_os = "macos")]
fn bind_layer1_input_before_fence(
    runtime: &Qwen80CompleteNativeRuntime,
    retained_l0_second_residual: &PinnedBuffer,
) -> Result<Qwen80SourceTokenLayer1InputBinding, String> {
    let state = runtime
        .linear_deltanet_state_slot_device_binding(1)
        .map_err(|error| error.to_string())?;
    if state.layer != 1
        || state.linear_state_slot != 1
        || state.conv_state_offset_elements != (L1_CONV_OFFSET_BYTES / 4) as usize
        || state.conv_state_capacity_elements
            != ((L1_CONV_OFFSET_BYTES + L0_CONV_BYTES) / 4) as usize
        || state.recurrent_state_offset_elements != (L1_RECURRENT_OFFSET_BYTES / 4) as usize
        || state.recurrent_state_capacity_elements
            != ((L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES) / 4) as usize
    {
        return Err(
            "source-token Layer-1 active state binding drifted from its sealed slot-one layout"
                .into(),
        );
    }
    if retained_l0_second_residual.length() != HIDDEN_BYTES {
        return Err("source-token Layer-1 input is not the exact 8192-byte L0 output".into());
    }
    Ok(Qwen80SourceTokenLayer1InputBinding {
        retained_input: retained_l0_second_residual.to_owned(),
        linear_state_slot: state.linear_state_slot,
        active_conv_state_buffer_identity_sha256: state.active_conv_state_buffer_identity_sha256,
        active_recurrent_state_buffer_identity_sha256: state
            .active_recurrent_state_buffer_identity_sha256,
    })
}

#[cfg(target_os = "macos")]
fn snapshot_f32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte count overflowed"))?;
    if buffer.length() < bytes as u64 {
        return Err(format!("{label} buffer is too short"));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() })
}

#[cfg(target_os = "macos")]
fn snapshot_u32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<u32>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("{label} byte count overflowed"))?;
    if buffer.length() < bytes as u64 {
        return Err(format!("{label} buffer is too short"));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec() })
}

#[cfg(target_os = "macos")]
fn require_parity(
    expected: &[f32],
    observed: &[f32],
    label: &str,
    tolerance: f32,
) -> Result<f32, String> {
    if expected.len() != observed.len() {
        return Err(format!(
            "{label} length differs: {} != {}",
            observed.len(),
            expected.len()
        ));
    }
    let mut maximum = 0.0f32;
    for (index, (&left, &right)) in expected.iter().zip(observed).enumerate() {
        if !left.is_finite() || !right.is_finite() {
            return Err(format!("{label} is non-finite at {index}"));
        }
        maximum = maximum.max((left - right).abs());
    }
    if maximum > tolerance {
        return Err(format!("{label} parity {maximum} exceeds {tolerance}"));
    }
    Ok(maximum)
}

/// Type-checked future device body. It has no context construction, admission,
/// fence, or dispatch responsibility; a later receipt-last outer reaper must
/// own those operations and bind the preflight generated above.
#[cfg(target_os = "macos")]
pub fn encode_source_token_l0_state_handoff(
    runtime: &Qwen80CompleteNativeRuntime,
    command: &mut TokenCommandBuffer<'_>,
    source_bridge: &Qwen80AllTenTrueMoeSourceBridge,
    static_binding: Qwen80SourceTokenL0ToL1StaticBinding,
) -> Result<Qwen80SourceTokenL0ToL1PreparedHandoff, String> {
    let resources = source_prefix::encode_source_input_l0_true_moe_capture(
        runtime,
        command,
        SOURCE_TOKEN_ID as u32,
        source_bridge,
    )?;
    if resources.graph.prefix_dispatches != PREFIX_DISPATCHES as usize
        || resources.graph.suffix_dispatches != SUFFIX_DISPATCHES as usize
        || command.dispatch_count() != TOTAL_DISPATCHES as usize
    {
        return Err("source-token L0 state-handoff must retain an exact 9+14 command graph".into());
    }
    let retained_l0_second_residual = resources.fixed.second_residual.to_owned();
    if retained_l0_second_residual.length() != HIDDEN_BYTES {
        return Err(
            "source-token L0 state-handoff retained output is not exactly 8192 bytes".into(),
        );
    }
    // The alias is created before the caller can fence the shared 9+14 TCB.
    // It keeps a live clone of the exact output allocation and pins the
    // source-scheduled L1 active state slot without executing Layer 1.
    let layer1_input = bind_layer1_input_before_fence(runtime, &retained_l0_second_residual)?;
    Ok(Qwen80SourceTokenL0ToL1PreparedHandoff {
        resources,
        retained_l0_second_residual,
        layer1_input,
        static_binding,
    })
}

#[cfg(target_os = "macos")]
fn execution_phase_document(phase: &MetalExecutionPhase) -> Value {
    json!({
        "strict_artifact_admission_started": phase.strict_artifact_admission_started,
        "strict_artifact_admission_succeeded": phase.strict_artifact_admission_succeeded,
        "metal_context_construction_attempted": phase.metal_context_construction_attempted,
        "metal_context_constructed": phase.metal_context_constructed,
        "structural_kernel_trace_enabled": phase.structural_kernel_trace_enabled,
        "dispatches_encoded": phase.dispatches_encoded,
        "encoded_kernel_names": phase.encoded_kernel_names,
        "command_commit_attempted": phase.command_commit_attempted,
        "command_fence_succeeded": phase.command_fence_succeeded,
        "readback_started": phase.readback_started,
        // A submit attempt can leave device execution unknowable if the
        // driver returns an error before the fence.  Never write a false
        // no-device claim in that case.
        "device_dispatch_may_have_occurred": phase.command_commit_attempted,
    })
}

#[cfg(target_os = "macos")]
fn phase_accurate_device_activity(phase: &MetalExecutionPhase) -> Value {
    if phase.command_fence_succeeded {
        Value::Bool(true)
    } else if phase.command_commit_attempted {
        Value::Null
    } else {
        Value::Bool(false)
    }
}

#[cfg(target_os = "macos")]
fn capture_invocation_document(
    authority: &CaptureAuthority,
    capture: &Path,
    outer: &Path,
) -> Result<Value, String> {
    let executable = std::env::current_exe()
        .map_err(|error| format!("cannot resolve L0 state-handoff child executable: {error}"))?;
    let executable = file_evidence(&executable, "L0 state-handoff child executable")?;
    let lease = authority
        .lease
        .as_ref()
        .ok_or("L0 state-handoff invocation lacks a validated lease")?;
    let launch = authority
        .outer_launch_authority
        .as_ref()
        .ok_or("L0 state-handoff invocation lacks an outer launch authority")?;
    Ok(json!({
        "schema": "hawking.ascension.qwen80_source_token_l0_state_handoff_inner_invocation.v1",
        "status": "INVOKED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_STRICT_MATH_ONE_SHOT_COMPONENT_CHILD",
        "probe_binary": executable.json(),
        "outer_capture_dir": outer,
        "inner_capture_dir": capture,
        "outer_preflight": {
            "path": authority.outer_preflight.path,
            "document_sha256": authority.outer_preflight.sha256,
            "seal_sha256": authority.outer_preflight_seal,
        },
        "versioned_current_pointer_observation_at_launch": {
            "historical_outer_preflight": {
                "path": authority.admission_current.path,
                "document_sha256": authority.admission_current.sha256,
                "seal_sha256": authority.admission_pointer_seal,
            },
            "validated_current": {
                "path": authority.launch_admission_current.path,
                "document_sha256": authority.launch_admission_current.sha256,
                "seal_sha256": authority.launch_admission_pointer_seal,
            },
            "canonical_path_required": true,
            "immutable_manifest_and_admission_receipt_must_remain_exact": true,
            "pointer_reseal_accepted_only_as_versioned_current_housekeeping": true,
        },
        "lease": {
            "path": lease.0.path,
            "document_sha256": lease.0.sha256,
            "seal_sha256": lease.1,
            "lease_id": authority.lease_id,
        },
        "outer_launch_authority": {
            "path": launch.0.path,
            "document_sha256": launch.0.sha256,
            "seal_sha256": launch.1,
        },
        "same_command_graph_contract": {
            "source_token_id": SOURCE_TOKEN_ID,
            "prefix_dispatches": PREFIX_DISPATCHES,
            "suffix_dispatches": SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "l1_prefix_dispatches": 0,
            "l1_binding_not_executed": true,
        },
        "execution_policy": {
            "strict_math_required": true,
            "timing_or_benchmarking_allowed": false,
            "l1_prefix_execution_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "automatic_retry_allowed": false,
        },
        "watcher_coordination": {
            "watcher_hold_must_remain_active": true,
            "watcher_restart_or_transition_authorized": false,
        },
    }))
}

#[cfg(target_os = "macos")]
fn validate_success_capture_document(
    value: &Value,
    authority: &CaptureAuthority,
) -> Result<(), String> {
    let root = object(value, "L0 state-handoff success receipt")?;
    exact_string(
        root,
        "schema",
        PRE_L1_HANDOFF_CAPTURE_SCHEMA,
        "L0 state-handoff success receipt",
    )?;
    exact_string(
        root,
        "status",
        PRE_L1_HANDOFF_CAPTURE_STATUS,
        "L0 state-handoff success receipt",
    )?;
    for (field, expected) in [
        ("metal_device_or_dispatch_performed", true),
        ("component_only", true),
        ("l1_binding_not_executed", true),
        ("complete_layer_or_token_performed", false),
    ] {
        bool_field(root, field, expected, "L0 state-handoff success receipt")?;
    }
    if root.get("l1_prefix_dispatches").and_then(Value::as_u64) != Some(0) {
        return Err("L0 state-handoff success receipt claims an L1 prefix dispatch".into());
    }
    let graph = object_field(
        root,
        "same_command_graph",
        "L0 state-handoff success receipt",
    )?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("prefix_dispatches", PREFIX_DISPATCHES),
        ("suffix_dispatches", SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ] {
        if u64_field(
            graph,
            field,
            "L0 state-handoff success receipt.same_command_graph",
        )? != expected
        {
            return Err(format!("L0 state-handoff success receipt {field} drifted"));
        }
    }
    bool_field(
        graph,
        "same_command_graph_retained",
        true,
        "L0 state-handoff success receipt.same_command_graph",
    )?;
    bool_field(
        graph,
        "fenced_once_after_prefix_and_suffix",
        true,
        "L0 state-handoff success receipt.same_command_graph",
    )?;
    let names = array_field(
        graph,
        "encoded_kernel_names",
        "L0 state-handoff success receipt.same_command_graph",
    )?;
    let expected_names = expected_kernel_order();
    if names.len() != expected_names.len()
        || names
            .iter()
            .zip(expected_names.iter())
            .any(|(actual, expected)| actual.as_str() != Some(expected))
    {
        return Err("L0 state-handoff success receipt kernel order drifted".into());
    }
    let route = object_field(
        root,
        "route_guard_readback",
        "L0 state-handoff success receipt",
    )?;
    if route.get("value").and_then(Value::as_u64) != Some(1) {
        return Err("L0 state-handoff success receipt route guard did not pass".into());
    }
    bool_field(
        route,
        "passed",
        true,
        "L0 state-handoff success receipt.route_guard_readback",
    )?;
    let parity = object_field(root, "readback_parity", "L0 state-handoff success receipt")?;
    if parity
        .get("all_ten_route_witness_count")
        .and_then(Value::as_u64)
        != Some(10)
        || array_field(
            parity,
            "all_ten_route_witnesses",
            "L0 state-handoff success receipt.readback_parity",
        )?
        .len()
            != 10
    {
        return Err("L0 state-handoff success receipt lacks all ten route witnesses".into());
    }
    let handoff = object_field(root, "l0_state_handoff", "L0 state-handoff success receipt")?;
    exact_string(
        handoff,
        "schema",
        PRE_L1_HANDOFF_CAPTURE_SCHEMA,
        "L0 state-handoff success receipt.l0_state_handoff",
    )?;
    exact_string(
        handoff,
        "status",
        PRE_L1_HANDOFF_CAPTURE_STATUS,
        "L0 state-handoff success receipt.l0_state_handoff",
    )?;
    bool_field(
        handoff,
        "l1_binding_not_executed",
        true,
        "L0 state-handoff success receipt.l0_state_handoff",
    )?;
    if handoff.get("l1_prefix_dispatches").and_then(Value::as_u64) != Some(0) {
        return Err("L0 state-handoff nested handoff claims L1 work".into());
    }
    let output = object_field(
        handoff,
        "retained_l0_second_residual",
        "L0 state-handoff success receipt.l0_state_handoff",
    )?;
    if u64_field(
        output,
        "elements",
        "L0 state-handoff success receipt.l0_state_handoff.retained_l0_second_residual",
    )? != HIDDEN
        || u64_field(
            output,
            "bytes",
            "L0 state-handoff success receipt.l0_state_handoff.retained_l0_second_residual",
        )? != HIDDEN_BYTES
        || sha_field(
            output,
            "f32le_sha256",
            "L0 state-handoff success receipt.l0_state_handoff.retained_l0_second_residual",
        )? != authority.baseline_second_residual_f32le_sha256
        || string_field(
            output,
            "device_buffer_id",
            "L0 state-handoff success receipt.l0_state_handoff.retained_l0_second_residual",
        )?
        .is_empty()
    {
        return Err("L0 state-handoff success receipt retained-output witness drifted".into());
    }
    let phase = object_field(root, "execution_phase", "L0 state-handoff success receipt")?;
    if phase.get("dispatches_encoded").and_then(Value::as_u64) != Some(TOTAL_DISPATCHES)
        || phase
            .get("command_fence_succeeded")
            .and_then(Value::as_bool)
            != Some(true)
        || phase.get("readback_started").and_then(Value::as_bool) != Some(true)
    {
        return Err("L0 state-handoff success receipt execution phase is incomplete".into());
    }
    let launch = authority
        .outer_launch_authority
        .as_ref()
        .ok_or("L0 state-handoff success receipt lacks launch authority")?;
    let binding = object_field(
        root,
        "outer_launch_authority_binding",
        "L0 state-handoff success receipt",
    )?;
    if string_field(
        binding,
        "path",
        "L0 state-handoff success receipt.outer_launch_authority_binding",
    )? != launch.0.path.to_string_lossy()
        || sha_field(
            binding,
            "document_sha256",
            "L0 state-handoff success receipt.outer_launch_authority_binding",
        )? != launch.0.sha256
        || sha_field(
            binding,
            "seal_sha256",
            "L0 state-handoff success receipt.outer_launch_authority_binding",
        )? != launch.1
    {
        return Err("L0 state-handoff success receipt outer launch binding drifted".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn finalize_capture(
    authority: &CaptureAuthority,
    capture: &Path,
    phase: &MetalExecutionPhase,
    outcome: Result<Value, String>,
) -> Result<Value, String> {
    let mut outcome = outcome;
    if let Ok(value) = outcome.as_ref() {
        if let Err(error) = validate_success_capture_document(value, authority) {
            outcome = Err(format!(
                "L0 state-handoff success receipt validation failed: {error}"
            ));
        }
    }
    // The mutable pointer may be resealed during the one-shot device child.
    // A terminal success remains valid only if the current pointer still
    // selects the exact immutable manifest and admission receipt.  The raw
    // pointer evidence is recorded rather than hidden.
    let terminal_pointer = observe_versioned_current_pointer(
        &authority.admission_current.path,
        &authority.manifest,
        &authority.manifest_seal,
        &authority.admission_receipt,
        &authority.admission_receipt_seal,
        "L0 state-handoff terminal current admission pointer",
    );
    if let Err(error) = terminal_pointer.as_ref() {
        if outcome.is_ok() {
            outcome = Err(format!(
                "L0 state-handoff terminal current-pointer validation failed: {error}"
            ));
        }
    }
    let success = outcome.is_ok();
    let terminal_pointer_record = match &terminal_pointer {
        Ok(pointer) => json!({
            "path": pointer.file.path,
            "document_sha256": pointer.file.sha256,
            "seal_sha256": pointer.seal,
            "valid_current_pointer_selecting_exact_immutable_authority": true,
        }),
        Err(error) => json!({
            "valid_current_pointer_selecting_exact_immutable_authority": false,
            "validation_error": error,
        }),
    };
    let stdout_path = capture.join("stdout.jsonl");
    let stderr_path = capture.join("stderr.log");
    let stdout = match outcome.as_ref() {
        Ok(value) => {
            let mut bytes = serde_json::to_vec(value)
                .map_err(|error| format!("cannot serialize L0 state-handoff stdout: {error}"))?;
            bytes.push(b'\n');
            bytes
        }
        Err(_) => Vec::new(),
    };
    let stderr = match outcome.as_ref() {
        Ok(_) => Vec::new(),
        Err(error) => {
            let mut bytes = error.as_bytes().to_vec();
            bytes.push(b'\n');
            bytes
        }
    };
    write_new_labeled(&stdout_path, "L0 state-handoff stdout", &stdout)?;
    write_new_labeled(&stderr_path, "L0 state-handoff stderr", &stderr)?;
    let stdout_evidence = file_evidence(&stdout_path, "L0 state-handoff stdout")?;
    let stderr_evidence = file_evidence(&stderr_path, "L0 state-handoff stderr")?;
    let lease = authority
        .lease
        .as_ref()
        .ok_or("L0 state-handoff terminal capture lacks lease")?;
    let launch = authority
        .outer_launch_authority
        .as_ref()
        .ok_or("L0 state-handoff terminal capture lacks outer launch authority")?;
    let launch_binding = json!({
        "path": launch.0.path,
        "document_sha256": launch.0.sha256,
        "seal_sha256": launch.1,
    });
    let phase_value = execution_phase_document(phase);
    let mut receipt = match outcome {
        Ok(mut value) => {
            let root = value
                .as_object_mut()
                .ok_or("L0 state-handoff success result must be a JSON object")?;
            root.insert("execution_phase".into(), phase_value.clone());
            root.insert(
                "terminal_child".into(),
                json!({
                    "exit_code": 0,
                    "stdout": stdout_evidence.json(),
                    "stderr": stderr_evidence.json(),
                    "receipt_path": capture.join("receipt.json"),
                    "receipt_written_last_is_completion_marker": true,
                }),
            );
            root.insert(
                "durable_capture".into(),
                json!({
                    "capture_directory": capture,
                    "invocation_path": capture.join("invocation.json"),
                    "stdout": stdout_evidence.json(),
                    "stderr": stderr_evidence.json(),
                    "receipt_written_last_is_completion_marker": true,
                    "outer_reaped_capture_required": true,
                    "replay_guarded": true,
                    "outer_reaper_binding": {
                        "lease_id": authority.lease_id,
                        "outer_launch_authority": launch_binding.clone(),
                    },
                }),
            );
            root.insert(
                "versioned_current_pointer_observations".into(),
                json!({
                    "historical_outer_preflight": {
                        "path": authority.admission_current.path,
                        "document_sha256": authority.admission_current.sha256,
                        "seal_sha256": authority.admission_pointer_seal,
                    },
                    "validated_at_child_launch": {
                        "path": authority.launch_admission_current.path,
                        "document_sha256": authority.launch_admission_current.sha256,
                        "seal_sha256": authority.launch_admission_pointer_seal,
                    },
                    "observed_at_terminal": terminal_pointer_record.clone(),
                    "canonical_path_required": true,
                    "immutable_manifest_and_admission_receipt_must_remain_exact": true,
                    "pointer_reseal_accepted_only_as_versioned_current_housekeeping": true,
                }),
            );
            value
        }
        Err(error) => json!({
            "schema": PRE_L1_HANDOFF_CAPTURE_SCHEMA,
            "status": "REFUSED_QWEN80_SOURCE_TOKEN_L0_STATE_HANDOFF_PHASE_ACCURATE_TERMINAL_FAILURE",
            "mode": "metal",
            "terminal_error": error,
            "metal_device_or_dispatch_performed": phase_accurate_device_activity(phase),
            "component_only": true,
            "l1_binding_not_executed": true,
            "l1_prefix_dispatches": 0,
            "complete_layer_or_token_performed": false,
            "execution_phase": phase_value,
            "outer_preflight_binding": {
                "path": authority.outer_preflight.path,
                "document_sha256": authority.outer_preflight.sha256,
                "seal_sha256": authority.outer_preflight_seal,
            },
            "outer_launch_authority_binding": launch_binding.clone(),
            "metal_execution_policy": {
                "strict_math_required": true,
                "timing_or_benchmarking_allowed": false,
                "l1_prefix_execution_allowed": false,
                "complete_layer_or_token_allowed": false,
                "tps_or_tg_claim_allowed": false,
                "lease_binding": {
                    "path": lease.0.path,
                    "document_sha256": lease.0.sha256,
                    "seal_sha256": lease.1,
                    "lease_id": authority.lease_id,
                },
            },
            "outer_reaper_binding": {
                "lease_id": authority.lease_id,
                "outer_launch_authority": launch_binding.clone(),
            },
            "versioned_current_pointer_observations": {
                "historical_outer_preflight": {
                    "path": authority.admission_current.path,
                    "document_sha256": authority.admission_current.sha256,
                    "seal_sha256": authority.admission_pointer_seal,
                },
                "validated_at_child_launch": {
                    "path": authority.launch_admission_current.path,
                    "document_sha256": authority.launch_admission_current.sha256,
                    "seal_sha256": authority.launch_admission_pointer_seal,
                },
                "observed_at_terminal": terminal_pointer_record.clone(),
                "canonical_path_required": true,
                "immutable_manifest_and_admission_receipt_must_remain_exact": true,
                "pointer_reseal_accepted_only_as_versioned_current_housekeeping": true,
            },
            "terminal_child": {
                "exit_code": 2,
                "stdout": stdout_evidence.json(),
                "stderr": stderr_evidence.json(),
                "receipt_path": capture.join("receipt.json"),
                "receipt_written_last_is_completion_marker": true,
            },
            "durable_capture": {
                "capture_directory": capture,
                "invocation_path": capture.join("invocation.json"),
                "stdout": stdout_evidence.json(),
                "stderr": stderr_evidence.json(),
                "receipt_written_last_is_completion_marker": true,
                "outer_reaped_capture_required": true,
                "replay_guarded": true,
            },
            "claim_boundary": {
                "l0_post_state_rollback_retained_output_component_only": true,
                "l1_binding_not_executed": true,
                "may_not_satisfy_next_layer_execution_dependency": true,
                "no_complete_layer_token_decoder_generation_server_hcli_tps_tg_or_tournament_claim": true,
                "no_watcher_or_server_started": true,
            },
        }),
    };
    let receipt_seal = seal(&mut receipt)?;
    let receipt_path = capture.join("receipt.json");
    let bytes = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("cannot serialize L0 state-handoff terminal receipt: {error}"))?;
    // This must be the last file created by the inner child.  The outer
    // reaper seals its own terminal record only after this child exits.
    write_new_labeled(&receipt_path, "L0 state-handoff terminal receipt", &bytes)?;
    if success {
        Ok(json!({
            "path": receipt_path,
            "seal_sha256": receipt_seal,
            "status": PRE_L1_HANDOFF_CAPTURE_STATUS,
            "component_only": true,
            "l1_binding_not_executed": true,
            "l1_prefix_dispatches": 0,
        }))
    } else {
        Err(format!(
            "L0 state-handoff child sealed phase-accurate refusal at {} (seal {})",
            receipt_path.display(),
            receipt_seal
        ))
    }
}

#[cfg(target_os = "macos")]
fn run_capture_cli(args: CaptureArgs) -> Result<Value, String> {
    let authority = validate_capture_authority(&args)?;
    match args.mode {
        CaptureMode::Preflight => Ok(capture_preflight_document(&authority)),
        CaptureMode::Metal => {
            let (outer, inner) = verify_metal_capture_paths(&authority)?;
            fs::create_dir(&inner).map_err(|error| {
                format!("cannot create L0 state-handoff inner capture: {error}")
            })?;
            let metadata = fs::symlink_metadata(&inner)
                .map_err(|error| format!("cannot stat L0 state-handoff inner capture: {error}"))?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(
                    "L0 state-handoff inner capture must be a new non-symlink directory".into(),
                );
            }
            let inner = fs::canonicalize(&inner).map_err(|error| {
                format!("cannot canonicalize L0 state-handoff inner capture: {error}")
            })?;
            let invocation = capture_invocation_document(&authority, &inner, &outer)?;
            let invocation_bytes = serde_json::to_vec_pretty(&invocation).map_err(|error| {
                format!("cannot serialize L0 state-handoff invocation: {error}")
            })?;
            write_new_labeled(
                &inner.join("invocation.json"),
                "L0 state-handoff invocation",
                &invocation_bytes,
            )?;
            let mut phase = MetalExecutionPhase::default();
            let outcome = run_metal(&authority, &inner, &mut phase);
            finalize_capture(&authority, &inner, &phase, outcome)
        }
    }
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    // Keep the immutable-plan producer as a deliberately separate, narrow
    // entrypoint.  It cannot be combined with the capture grammar because a
    // mixed invocation would otherwise make the plan look device-authorized.
    let legacy_plan_mode = arguments
        .iter()
        .any(|argument| matches!(argument.as_str(), "--handoff-authority" | "--out"));
    let result = if legacy_plan_mode {
        parse_args(arguments).and_then(run).map(|(path, seal)| {
            json!({
                "path": path,
                "seal_sha256": seal,
                "cpu_only_preflight": true,
            })
        })
    } else {
        #[cfg(target_os = "macos")]
        {
            parse_capture_args(arguments).and_then(run_capture_cli)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = arguments;
            Err("source-token L0 state-handoff capture is macOS-only".into())
        }
    };
    match result {
        Ok(value) => println!(
            "{}",
            serde_json::to_string_pretty(&value).expect("result JSON")
        ),
        Err(error) => {
            eprintln!("ascension_qwen80_source_token_l0_to_layer1_state_handoff_device: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sha() -> String {
        "a".repeat(64)
    }

    fn seal_test(mut value: Value) -> Value {
        seal(&mut value).unwrap();
        value
    }

    fn range(allocation: &str, slot: u64, offset: u64, capacity: u64) -> Value {
        json!({
            "allocation_id": allocation,
            "slot": slot,
            "offset_bytes": offset,
            "capacity_bytes": capacity,
        })
    }

    fn authority_fixture() -> Value {
        seal_test(json!({
            "schema": BASELINE_AUTHORITY_SCHEMA,
            "status": BASELINE_AUTHORITY_STATUS,
            "ready_for_l1_device_handoff": false,
            "component_only": true,
            "source_binding": {
                "model_key": MODEL_KEY,
                "source_revision": SOURCE_REVISION,
                "manifest_document_sha256": MANIFEST_DOCUMENT_SHA,
                "manifest_seal_sha256": MANIFEST_SEAL,
                "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
                "source_token_id": SOURCE_TOKEN_ID,
            },
            "consumed_component_capture": {
                "layer": 0,
                "linear_state_slot": 0,
                "same_command_graph": {"prefix_dispatches":PREFIX_DISPATCHES,"suffix_dispatches":SUFFIX_DISPATCHES,"total_dispatches":TOTAL_DISPATCHES},
                "second_residual": {"elements":HIDDEN,"bytes":HIDDEN_BYTES,"f32le_sha256":sha()},
                "route_guard": {"passed":true,"route_witness_count":10,"ordered_source_route_ids":(0..10).collect::<Vec<_>>()},
            },
            "static_state_layout_authority": {
                "session_id":"test-session",
                "l0_and_l1_slots_verified_disjoint":true,
                "l0": {
                    "linear_state_slot":0,
                    "active_conv":range("active-conv",0,0,L0_CONV_BYTES),
                    "active_recurrent":range("active-recurrent",0,0,L0_RECURRENT_BYTES),
                    "rollback_conv":range("rollback-conv",0,0,L0_CONV_BYTES),
                    "rollback_recurrent":range("rollback-recurrent",0,0,L0_RECURRENT_BYTES),
                },
                "l1": {
                    "linear_state_slot":1,
                    "active_conv":range("active-conv",1,L1_CONV_OFFSET_BYTES,L1_CONV_OFFSET_BYTES + L0_CONV_BYTES),
                    "active_recurrent":range("active-recurrent",1,L1_RECURRENT_OFFSET_BYTES,L1_RECURRENT_OFFSET_BYTES + L0_RECURRENT_BYTES),
                },
            },
            "next_required_real_decoder_dependency": {"schema":NEXT_LAYER_HANDOFF_WITNESS_SCHEMA,"required_status":NEXT_LAYER_HANDOFF_WITNESS_STATUS},
            "handoff_assessment": {
                "historical_output_hash_is_not_a_retained_device_buffer":true,
                "historical_initial_state_hashes_are_not_post_state_commit_witnesses":true,
                "missing_real_evidence":["output","state","l1"],
            },
            "claim_boundary":{"cpu_only_assessment":true,"no_metal_context_or_device_dispatch":true},
        }))
    }

    fn parsed_fixture() -> BaselineAuthority {
        parse_baseline_authority(
            BoundFile {
                path: "/tmp/handoff.json".into(),
                bytes: 1,
                sha256: sha(),
            },
            &authority_fixture(),
        )
        .unwrap()
    }

    #[cfg(target_os = "macos")]
    fn write_sealed_child_preflight_fixture(tamper: bool) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "hawking-qwen80-l0-l1-preflight-test-{}-{}",
            std::process::id(),
            if tamper { "tampered" } else { "exact" }
        ));
        let _ = fs::remove_dir_all(&directory);
        fs::create_dir(&directory).unwrap();
        let baseline_path = directory.join("baseline-authority.json");
        let baseline = authority_fixture();
        fs::write(
            &baseline_path,
            serde_json::to_vec_pretty(&baseline).unwrap(),
        )
        .unwrap();
        let (file, value) = read_json(&baseline_path, "test baseline").unwrap();
        let authority = parse_baseline_authority(file, &value).unwrap();
        let mut child = prepared_document(&authority);
        if tamper {
            child["fresh_capture_required"]["must_reencode_source_token_l0"] = json!(false);
        }
        seal(&mut child).unwrap();
        let child_path = directory.join("child-preflight.json");
        fs::write(&child_path, serde_json::to_vec_pretty(&child).unwrap()).unwrap();
        child_path
    }

    #[test]
    fn current_incomplete_authority_is_the_only_valid_baseline_for_the_child() {
        let authority = parsed_fixture();
        let plan = prepared_document(&authority);
        assert_eq!(plan["status"], PREPARED_STATUS);
        assert_eq!(
            plan["fresh_capture_required"]["must_reencode_source_token_l0"],
            true
        );
        assert_eq!(
            plan["required_next_layer_handoff_witness"]["retained_l0_second_residual"]["bytes"],
            HIDDEN_BYTES
        );
        assert_eq!(
            plan["required_next_layer_handoff_witness"]["layer1_input_binding"]
                ["linear_state_slot"],
            1
        );
        assert_eq!(
            plan["planned_pre_l1_handoff_capture"]["l1_binding_not_executed"],
            true
        );
        assert_eq!(
            plan["planned_pre_l1_handoff_capture"]
                ["may_not_satisfy_next_layer_execution_dependency"],
            true
        );
        assert_eq!(
            plan["required_next_layer_handoff_witness"]
                ["remains_required_after_planned_pre_l1_capture"],
            true
        );
    }

    #[test]
    fn sealed_authority_cannot_be_tampered_or_replaced_by_a_ready_claim() {
        let authority = authority_fixture();
        assert!(parse_baseline_authority(
            BoundFile {
                path: "/tmp/a.json".into(),
                bytes: 1,
                sha256: sha()
            },
            &authority,
        )
        .is_ok());
        let mut tampered = authority.clone();
        tampered["status"] = json!("READY");
        assert!(parse_baseline_authority(
            BoundFile {
                path: "/tmp/a.json".into(),
                bytes: 1,
                sha256: sha()
            },
            &tampered,
        )
        .is_err());
    }

    #[test]
    fn missing_output_or_state_gap_cannot_be_silently_dropped() {
        let mut authority = authority_fixture();
        authority["handoff_assessment"]["missing_real_evidence"] = json!(["output", "state"]);
        let authority = seal_test({
            let mut unsigned = authority.as_object().unwrap().clone();
            unsigned.remove("seal_sha256");
            Value::Object(unsigned)
        });
        assert!(parse_baseline_authority(
            BoundFile {
                path: "/tmp/a.json".into(),
                bytes: 1,
                sha256: sha()
            },
            &authority,
        )
        .is_err());
    }

    #[test]
    fn child_preflight_is_sealed_and_never_claims_a_device_run() {
        let authority = parsed_fixture();
        let mut plan = prepared_document(&authority);
        let seal_value = seal(&mut plan).unwrap();
        assert_eq!(verify_seal(&plan, "plan").unwrap(), seal_value);
        assert_eq!(
            plan["implementation_boundary"]["device_context_or_dispatch_performed"],
            false
        );
        assert_eq!(
            plan["claim_boundary"]["component_only_even_after_a_future_pass"],
            true
        );
    }

    #[test]
    fn receipt_seal_uses_python_compact_float_spelling() {
        // This exact compact preimage/hash is produced by:
        // json.dumps(value, sort_keys=True, separators=(",", ":"),
        //            ensure_ascii=False, allow_nan=False)
        // followed by SHA-256.  The e-06/e-05 spelling is the historical
        // cross-language fault line: serde_json renders those as e-6/e-5.
        let mut receipt = json!({
            "a": 5.029141902923584e-8,
            "b": 1e-6,
            "c": 1e16,
            "d": 1.0,
            "e": 0.0001,
            "f": 0.00001,
            "g": -0.0,
            "nested": {"z": 2.0489096641540527e-8},
        });
        assert_eq!(
            String::from_utf8(canonical_json(&receipt).unwrap()).unwrap(),
            "{\"a\":5.029141902923584e-08,\"b\":1e-06,\"c\":1e+16,\"d\":1.0,\"e\":0.0001,\"f\":1e-05,\"g\":-0.0,\"nested\":{\"z\":2.0489096641540527e-08}}"
        );
        let observed = seal(&mut receipt).unwrap();
        assert_eq!(
            observed,
            "7cc976eaf94bc935878ca15fa1291e247d85117136c0ea8708d1ee222d364473"
        );
        assert_eq!(
            verify_seal(&receipt, "python-compatible receipt").unwrap(),
            observed
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn child_preflight_rederives_its_static_state_binding_from_raw_baseline() {
        let child_path = write_sealed_child_preflight_fixture(false);
        let binding = load_l0_to_l1_static_binding_from_child_preflight(&child_path).unwrap();
        assert_eq!(binding.session_id, "test-session");
        assert_eq!(binding.l1_active_conv_allocation, "active-conv");
        assert_eq!(binding.l1_active_recurrent_allocation, "active-recurrent");
        let _ = fs::remove_dir_all(child_path.parent().unwrap());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn child_preflight_refuses_a_resealed_but_semantically_divergent_plan() {
        let child_path = write_sealed_child_preflight_fixture(true);
        let error = load_l0_to_l1_static_binding_from_child_preflight(&child_path).unwrap_err();
        assert!(error.contains("exact re-derived sealed plan"));
        let _ = fs::remove_dir_all(child_path.parent().unwrap());
    }

    #[test]
    fn capture_preflight_parser_is_file_only_and_refuses_lease_inputs() {
        let parsed = parse_capture_args(vec![
            "--outer-preflight".into(),
            "/tmp/outer-preflight.json".into(),
            "--mode".into(),
            "preflight".into(),
            "--workers".into(),
            "2".into(),
        ])
        .unwrap();
        assert_eq!(parsed.mode, CaptureMode::Preflight);
        assert!(parsed.lease_receipt.is_none());
        assert!(parsed.capture_dir.is_none());
        let error = parse_capture_args(vec![
            "--outer-preflight".into(),
            "/tmp/outer-preflight.json".into(),
            "--mode".into(),
            "preflight".into(),
            "--lease-receipt".into(),
            "/tmp/lease.json".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("preflight mode refuses lease"));
    }

    #[test]
    fn capture_metal_parser_requires_full_outer_reaped_path_set() {
        let error = parse_capture_args(vec![
            "--outer-preflight".into(),
            "/tmp/outer-preflight.json".into(),
            "--mode".into(),
            "metal".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("metal mode requires lease"));

        let parsed = parse_capture_args(vec![
            "--outer-preflight".into(),
            "/tmp/outer-preflight.json".into(),
            "--mode".into(),
            "metal".into(),
            "--lease-receipt".into(),
            "/tmp/lease.json".into(),
            "--outer-launch-authority".into(),
            "/tmp/outer-launch.json".into(),
            "--outer-capture-dir".into(),
            "/tmp/outer-capture".into(),
            "--capture-dir".into(),
            "/tmp/outer-capture/inner".into(),
            "--workers".into(),
            "4".into(),
        ])
        .unwrap();
        assert_eq!(parsed.mode, CaptureMode::Metal);
        assert_eq!(parsed.workers, MAX_WORKERS);
        assert_eq!(
            parsed.capture_dir.unwrap(),
            PathBuf::from("/tmp/outer-capture/inner")
        );
    }

    #[test]
    fn combined_structural_kernel_order_is_exactly_nine_plus_fourteen() {
        let order = expected_kernel_order();
        assert_eq!(order.len(), TOTAL_DISPATCHES as usize);
        assert_eq!(&order[..PREFIX_DISPATCHES as usize], &PREFIX_KERNELS);
        assert_eq!(&order[PREFIX_DISPATCHES as usize..], &SUFFIX_KERNELS);
        assert_ne!(order[0], order[order.len() - 1]);
    }

    #[cfg(target_os = "macos")]
    fn pointer_document(
        manifest: &BoundFile,
        manifest_seal: &str,
        receipt: &BoundFile,
        receipt_seal: &str,
        epoch: u64,
    ) -> Value {
        seal_test(json!({
            "schema": ADMISSION_POINTER_SCHEMA,
            "status": ADMISSION_POINTER_STATUS,
            "pointer_epoch": epoch,
            "complete_manifest": {
                "path": manifest.path,
                "document_sha256": manifest.sha256,
                "seal_sha256": manifest_seal,
            },
            "admission_receipt": {
                "path": receipt.path,
                "document_sha256": receipt.sha256,
                "seal_sha256": receipt_seal,
            },
        }))
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn versioned_current_pointer_accepts_reseal_but_refuses_immutable_drift() {
        let directory = std::env::temp_dir().join(format!(
            "hawking-qwen80-versioned-current-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir(&directory).unwrap();
        let manifest_path = directory.join("manifest.json");
        let receipt_path = directory.join("admission-receipt.json");
        let pointer_path = directory.join("current.json");
        let manifest_value = seal_test(json!({"schema":"test-manifest","identity":"immutable"}));
        let receipt_value = seal_test(json!({"schema":"test-receipt","identity":"immutable"}));
        fs::write(&manifest_path, serde_json::to_vec(&manifest_value).unwrap()).unwrap();
        fs::write(&receipt_path, serde_json::to_vec(&receipt_value).unwrap()).unwrap();
        let manifest = file_evidence(&manifest_path, "test manifest").unwrap();
        let receipt = file_evidence(&receipt_path, "test receipt").unwrap();
        let manifest_seal = verify_seal(&manifest_value, "test manifest").unwrap();
        let receipt_seal = verify_seal(&receipt_value, "test receipt").unwrap();
        let first = pointer_document(&manifest, &manifest_seal, &receipt, &receipt_seal, 1);
        fs::write(&pointer_path, serde_json::to_vec(&first).unwrap()).unwrap();
        let historical = file_evidence(&pointer_path, "test pointer").unwrap();
        let historical_seal = verify_seal(&first, "test pointer").unwrap();

        // A pointer housekeeping reseal has different raw evidence but the
        // same canonical path and immutable selections; it is admissible.
        let resealed = pointer_document(&manifest, &manifest_seal, &receipt, &receipt_seal, 2);
        fs::write(&pointer_path, serde_json::to_vec(&resealed).unwrap()).unwrap();
        let source = json!({
            "admission_current": historical.json(),
            "admission_pointer_seal_sha256": historical_seal,
        });
        let source = source.as_object().unwrap();
        let (preserved, preserved_seal) =
            historical_current_pointer_evidence(source, "test source").unwrap();
        assert_eq!(preserved.path, historical.path);
        assert_eq!(preserved.sha256, historical.sha256);
        assert_eq!(preserved_seal, historical_seal);
        let current = observe_versioned_current_pointer(
            &preserved.path,
            &manifest,
            &manifest_seal,
            &receipt,
            &receipt_seal,
            "test current pointer",
        )
        .unwrap();
        assert_ne!(current.file.sha256, historical.sha256);
        assert_ne!(current.seal, historical_seal);

        let wrong_manifest = BoundFile {
            path: manifest.path.clone(),
            bytes: manifest.bytes,
            sha256: "b".repeat(64),
        };
        let manifest_drift =
            pointer_document(&wrong_manifest, &manifest_seal, &receipt, &receipt_seal, 3);
        fs::write(&pointer_path, serde_json::to_vec(&manifest_drift).unwrap()).unwrap();
        assert!(observe_versioned_current_pointer(
            &pointer_path,
            &manifest,
            &manifest_seal,
            &receipt,
            &receipt_seal,
            "test manifest drift",
        )
        .unwrap_err()
        .contains("pinned immutable manifest"));

        let receipt_drift = pointer_document(
            &manifest,
            &manifest_seal,
            &BoundFile {
                path: receipt.path.clone(),
                bytes: receipt.bytes,
                sha256: "c".repeat(64),
            },
            &receipt_seal,
            4,
        );
        fs::write(&pointer_path, serde_json::to_vec(&receipt_drift).unwrap()).unwrap();
        assert!(observe_versioned_current_pointer(
            &pointer_path,
            &manifest,
            &manifest_seal,
            &receipt,
            &receipt_seal,
            "test receipt drift",
        )
        .unwrap_err()
        .contains("pinned immutable admission receipt"));
        let _ = fs::remove_dir_all(directory);
    }
}
