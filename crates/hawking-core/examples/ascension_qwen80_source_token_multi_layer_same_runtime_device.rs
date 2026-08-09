//! CPU/build host for N-layer Qwen80 same-runtime sequential hidden propagation.
//!
//! Extends the proven L0+L1 full-layer same-runtime pattern:
//! one runtime, one command buffer, ONE fence after all dispatches, caller-owned
//! per-layer state slots, structural kernel-name trace, receipt written last.
//!
//! `--layer-count N` selects layers `0..N` (e.g. 3 ⇒ L0+L1+L2 = 69 dispatches;
//! 4 ⇒ L0..L3 = 92 dispatches and crosses the first GQA layer).  GQA full-layer
//! same-runtime encode is wired; physical metal captures for layer-count 3 and
//! 4 are owner-run under resource admission.  Default CLI is preflight-only and
//! never creates a Metal context or loads the 148 GB body.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_source_token_multi_layer_same_runtime_device -- \
//!   --mode preflight --layer-count 4 \
//!   --execution-schedule-authority ABSOLUTE_SEALED_SCHEDULE \
//!   --chain-cpu-oracle ABSOLUTE_SEALED_CHAIN_ORACLE \
//!   --l1-full-layer-assessment ABSOLUTE_SEALED_L1_COMPLETION_ASSESSMENT \
//!   --joint-assessment ABSOLUTE_SEALED_L0_L1_JOINT_POST_CAPTURE_ASSESSMENT \
//!   --host-binary ABSOLUTE_CURRENT_HOST_BINARY \
//!   --out ABSOLUTE_NEW_JSON --workers 1
//!
//! # Owner physical capture (after lease + outer launch authority) — first GQA crossing:
//! cargo run -p hawking-core --example ascension_qwen80_source_token_multi_layer_same_runtime_device -- \
//!   --mode metal --layer-count 4 \
//!   --outer-preflight ABSOLUTE_SEALED_OUTER_PREFLIGHT \
//!   --lease-receipt ABSOLUTE_SEALED_LEASE \
//!   --outer-launch-authority ABSOLUTE_SEALED_OUTER_LAUNCH \
//!   --outer-capture-dir ABSOLUTE_EXISTING_OUTER_DIRECTORY \
//!   --capture-dir ABSOLUTE_NEW_DIRECT_CHILD_DIRECTORY --workers 1
//! ```

#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_all_ten_true_moe_graph_device.rs"]
mod all_ten;
#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_source_token_all_ten_true_moe_graph_device.rs"]
mod source_l0;
#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_first_residual_bridge_device.rs"]
mod source_prefix;
#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_source_token_l1_moe_completion_preflight.rs"]
mod completion_preflight;

#[cfg(target_os = "macos")]
use std::collections::BTreeMap;
use hawking_core::metal::TokenCommandBuffer;
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime, Qwen80CompleteRuntimeOptions,
    Qwen80MultiLayerSuffixCpuOracle, Qwen80MultiLayerSuffixWitness, Qwen80RouteSelection,
    Qwen80SameRuntimeMultiLayerChainParity, Qwen80SameRuntimeSubsequentLayerPrefix,
};
#[cfg(target_os = "macos")]
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};

use hawking_core::model::qwen80_48_layer_execution_schedule::{
    qwen80_layer_execution_schedule, qwen80_multi_layer_structural_kernel_trace,
    qwen80_multi_layer_total_dispatches, Qwen80ExecutionScheduleSourceBinding,
    QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA, QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
    QWEN80_DELTANET_FULL_LAYER_DISPATCHES, QWEN80_GRAVITY_MANIFEST_SEAL_SHA256, QWEN80_LAYERS,
    QWEN80_SOURCE_REVISION,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const HOST_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_host_preflight.v1";
const HOST_PREFLIGHT_STATUS: &str =
    "COMPILED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const FUTURE_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_capture.v1";
const FUTURE_INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_ONLY";
const FUTURE_INNER_REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_PHASE_ACCURATE_TERMINAL_FAILURE";
const FUTURE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_capture.v1";
const FUTURE_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY";
const FUTURE_RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_quiet_metal_lease_release.v1";
const FUTURE_RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";
const OUTER_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_preflight.v1";
const OUTER_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const METAL_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_quiet_metal_lease.v1";
const METAL_LEASE_STATUS: &str =
    "GRANTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_QUIET_METAL_LEASE";
const METAL_OUTER_LAUNCH_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_launch_authority.v1";
const METAL_OUTER_LAUNCH_STATUS: &str =
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_REAPED_ONE_SHOT_METAL_CHILD";
const L1_ASSESSMENT_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_full_layer_completion_assessment.v1";
const L1_EARNED_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L1_COMPLETE_LAYER_COMPONENT_NOT_TOKEN_DECODER";
/// Metal-path provenance for the shared L0+L1 finalizer / route-authority validator.
/// Distinct from [`L1_ASSESSMENT_SCHEMA`]: preflight validates the completion assessment;
/// metal loads this joint post-capture assessment. One flag cannot satisfy both.
const JOINT_ASSESSMENT_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1";
const JOINT_ASSESSMENT_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER";
const L1_ROUTE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1";
const L1_ROUTE_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_SAME_RUNTIME_MOE_SUFFIX";
const L0_SOURCE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1";
const L0_SOURCE_OUTER_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const CHAIN_ORACLE_SCHEMA: &str = "hawking.ascension.qwen80_multi_layer_chain_cpu_oracle.v1";
const MAX_JSON_BYTES: u64 = 100_000_000;
const SOURCE_TOKEN_ID: u64 = 1;
/// Proven DeltaNet-only multi-layer capture: L0..L2 (three full layers).
const RECOMMENDED_DELTANET_LAYER_COUNT: usize = 3;
/// First chain that crosses GQA: L0..L3 (three DeltaNet + one GQA = 92 dispatches).
const RECOMMENDED_FIRST_GQA_LAYER_COUNT: usize = 4;
/// Back-compat alias for preflight docs that still name the DeltaNet-only step.
const RECOMMENDED_FIRST_LAYER_COUNT: usize = RECOMMENDED_DELTANET_LAYER_COUNT;
const L0_DISPATCHES: usize = 23;
const L1_PREFIX_DISPATCHES: usize = 9;
const L1_MOE_SUFFIX_DISPATCHES: usize = 14;
const GRAVITY_MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";

/// Frozen L0..L2 structural kernel order (3 × 23 DeltaNet full layers).
const MULTI_LAYER_L0_L2_KERNELS: [&str; 69] = {
    const ONE: [&str; 23] = [
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
    let mut out = [""; 69];
    let mut i = 0;
    while i < 69 {
        out[i] = ONE[i % 23];
        i += 1;
    }
    out
};

#[derive(Debug)]
struct Args {
    layer_count: usize,
    execution_schedule_authority: PathBuf,
    chain_cpu_oracle: PathBuf,
    l1_full_layer_assessment: PathBuf,
    joint_assessment: PathBuf,
    host_binary: PathBuf,
    out: PathBuf,
    workers: usize,
}

#[derive(Debug)]
struct MetalArgs {
    layer_count: usize,
    outer_preflight: PathBuf,
    lease_receipt: PathBuf,
    outer_launch_authority: PathBuf,
    outer_capture_dir: PathBuf,
    capture_dir: PathBuf,
    workers: usize,
}

#[derive(Debug)]
enum Invocation {
    Preflight(Args),
    Metal(MetalArgs),
}

#[derive(Clone, Debug)]
struct FileEvidence {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug)]
struct SealedDocument {
    file: FileEvidence,
    value: Value,
    document_sha256: String,
    seal_sha256: String,
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
    "usage: ascension_qwen80_source_token_multi_layer_same_runtime_device \\\n  --mode preflight --layer-count N \\\n  --execution-schedule-authority ABSOLUTE_SEALED_SCHEDULE \\\n  --chain-cpu-oracle ABSOLUTE_SEALED_CHAIN_ORACLE \\\n  --l1-full-layer-assessment ABSOLUTE_SEALED_L1_COMPLETION_ASSESSMENT \\\n  --joint-assessment ABSOLUTE_SEALED_L0_L1_JOINT_POST_CAPTURE_ASSESSMENT \\\n  --host-binary ABSOLUTE_CURRENT_HOST_BINARY \\\n  --out ABSOLUTE_NEW_JSON --workers 1..4\n\
or: ... --mode metal --layer-count N \\\n  --outer-preflight ABSOLUTE --lease-receipt ABSOLUTE \\\n  --outer-launch-authority ABSOLUTE --outer-capture-dir ABSOLUTE \\\n  --capture-dir ABSOLUTE_NEW_CHILD --workers 1..4\n\
preflight validates the L1 full-layer completion assessment; metal-path provenance requires the joint post-capture assessment bound through the outer. metal mode is intentionally not default; the owner runs physical capture under resource admission after this preflight."
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

// Campaign-canonical JSON: keys sorted, compact separators, and Python-style float
// repr. serde_json's default number formatting differs from Python's in exponent
// zero-padding (e-7 vs e-07) and integral floats (1 vs 1.0), which made every
// sealed multi-layer child receipt fail the outer's seal check even though the
// capture itself succeeded. Every other sealed document in this campaign is
// produced or verified through the Python form, so the child must match it.
fn python_float(number: &serde_json::Number) -> Result<String, String> {
    let value = number
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or("floating JSON number must be finite")?;
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
                .map_err(|error| format!("bad exponent: {error}"))?,
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
                    fractional = fractional.checked_add(1).ok_or("float length overflow")?;
                }
                digits.push(char::from(byte));
            }
            _ => return Err("bad float mantissa".into()),
        }
    }
    let first = digits
        .bytes()
        .position(|byte| byte != b'0')
        .ok_or("zero float has no significant digit")?;
    let mut significant = digits[first..].to_owned();
    let mut power = exponent
        .checked_sub(fractional)
        .ok_or("float power overflow")?;
    while significant.len() > 1 && significant.ends_with('0') {
        significant.pop();
        power = power.checked_add(1).ok_or("float power overflow")?;
    }
    let scientific = power
        .checked_add(i32::try_from(significant.len() - 1).unwrap_or(i32::MAX))
        .ok_or("float power overflow")?;
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
        let position = usize::try_from(position).map_err(|_| "negative float position")?;
        format!("{}.{}", &significant[..position], &significant[position..])
    };
    Ok(format!("{sign}{rendered}"))
}

fn canonical_into(value: &Value, output: &mut String) -> Result<(), String> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => {
            if value.is_i64() || value.is_u64() {
                output.push_str(&value.to_string());
            } else {
                output.push_str(&python_float(value)?);
            }
        }
        Value::String(value) => output.push_str(
            &serde_json::to_string(value)
                .map_err(|error| format!("string canonicalization: {error}"))?,
        ),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                canonical_into(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in values {
                ordered.insert(key, value);
            }
            output.push('{');
            for (index, (key, value)) in ordered.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(
                    &serde_json::to_string(key)
                        .map_err(|error| format!("key canonicalization: {error}"))?,
                );
                output.push(':');
                canonical_into(value, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn canonical_json_sha(value: &Value) -> Result<String, String> {
    let mut rendered = String::new();
    canonical_into(value, &mut rendered)?;
    Ok(sha256_hex(rendered.as_bytes()))
}

fn seal(value: &mut Value) -> Result<String, String> {
    {
        let object = value
            .as_object_mut()
            .ok_or("document must be a JSON object")?;
        object.remove("seal_sha256");
    }
    let seal = canonical_json_sha(value)?;
    value
        .as_object_mut()
        .ok_or("document must be a JSON object")?
        .insert("seal_sha256".into(), json!(seal.clone()));
    Ok(seal)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("{} must be absolute", path.display()));
    }
    if path.exists() {
        return Err(format!("create-new required; {} exists", path.display()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create parent: {e}"))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| format!("serialize: {e}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("open: {e}"))?;
    file.write_all(&bytes).map_err(|e| format!("write: {e}"))?;
    file.sync_all().map_err(|e| format!("sync: {e}"))?;
    Ok(())
}

fn read_json(path: &Path, label: &str) -> Result<Value, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let meta = fs::metadata(path).map_err(|e| format!("stat {label}: {e}"))?;
    if meta.len() > MAX_JSON_BYTES {
        return Err(format!(
            "{label} size observed={}, exceeds max={MAX_JSON_BYTES}",
            meta.len()
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("read {label}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {label}: {e}"))
}

fn file_sha(path: &Path, label: &str) -> Result<(u64, String), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {label}: {e}"))?;
    Ok((bytes.len() as u64, sha256_hex(&bytes)))
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let mut unsigned = value.clone();
    let object = unsigned
        .as_object_mut()
        .ok_or_else(|| format!("{label} must be object"))?;
    let seal = object
        .remove("seal_sha256")
        .and_then(|v| v.as_str().map(str::to_owned))
        .ok_or_else(|| format!("{label} missing seal_sha256"))?;
    if seal.len() != 64 {
        return Err(format!(
            "{label} seal_sha256 length observed={}, expected=64",
            seal.len()
        ));
    }
    let expected = canonical_json_sha(&unsigned)?;
    if seal != expected {
        return Err(format!(
            "{label} seal mismatch: observed={seal}, expected={expected}"
        ));
    }
    Ok(seal)
}

fn require_absolute(path: PathBuf, flag: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{flag} must be absolute (observed={})", path.display()));
    }
    Ok(path)
}

fn parse_layer_count(raw: &str) -> Result<usize, String> {
    let n: usize = raw
        .parse()
        .map_err(|_| format!("--layer-count must be integer, got {raw}"))?;
    if n < 2 || n > QWEN80_LAYERS {
        return Err(format!(
            "--layer-count observed={n}, expected in 2..={QWEN80_LAYERS} (multi-layer starts at L0+L1)"
        ));
    }
    Ok(n)
}

fn parse_workers(raw: &str) -> Result<usize, String> {
    let n: usize = raw
        .parse()
        .map_err(|_| format!("--workers must be integer, got {raw}"))?;
    if !(1..=4).contains(&n) {
        return Err(format!("--workers observed={n}, expected 1..=4"));
    }
    Ok(n)
}

fn parse_preflight_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut layer_count = None;
    let mut execution_schedule_authority = None;
    let mut chain_cpu_oracle = None;
    let mut l1_full_layer_assessment = None;
    let mut joint_assessment = None;
    let mut host_binary = None;
    let mut out = None;
    let mut workers = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--layer-count" => {
                let raw = args.next().ok_or("--layer-count requires a value")?;
                if layer_count.replace(parse_layer_count(&raw)?).is_some() {
                    return Err("--layer-count may not be repeated".into());
                }
            }
            "--execution-schedule-authority" => {
                let value = args
                    .next()
                    .ok_or("--execution-schedule-authority requires a value")?;
                if execution_schedule_authority
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err("--execution-schedule-authority may not be repeated".into());
                }
            }
            "--chain-cpu-oracle" => {
                let value = args.next().ok_or("--chain-cpu-oracle requires a value")?;
                if chain_cpu_oracle.replace(PathBuf::from(value)).is_some() {
                    return Err("--chain-cpu-oracle may not be repeated".into());
                }
            }
            "--l1-full-layer-assessment" => {
                let value = args
                    .next()
                    .ok_or("--l1-full-layer-assessment requires a value")?;
                if l1_full_layer_assessment
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err("--l1-full-layer-assessment may not be repeated".into());
                }
            }
            "--joint-assessment" => {
                let value = args.next().ok_or("--joint-assessment requires a value")?;
                if joint_assessment.replace(PathBuf::from(value)).is_some() {
                    return Err("--joint-assessment may not be repeated".into());
                }
            }
            "--host-binary" => {
                let value = args.next().ok_or("--host-binary requires a value")?;
                if host_binary.replace(PathBuf::from(value)).is_some() {
                    return Err("--host-binary may not be repeated".into());
                }
            }
            "--out" => {
                let value = args.next().ok_or("--out requires a value")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out may not be repeated".into());
                }
            }
            "--workers" => {
                let raw = args.next().ok_or("--workers requires a value")?;
                if workers.replace(parse_workers(&raw)?).is_some() {
                    return Err("--workers may not be repeated".into());
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => {
                return Err(format!(
                    "preflight mode refuses unsupported argument {other:?}; {}",
                    usage()
                ))
            }
        }
    }
    let layer_count = layer_count.ok_or_else(|| format!("missing --layer-count; {}", usage()))?;
    let workers = workers.ok_or_else(|| format!("missing --workers; {}", usage()))?;
    let require = |path: Option<PathBuf>, flag: &str| -> Result<PathBuf, String> {
        require_absolute(
            path.ok_or_else(|| format!("missing {flag}; {}", usage()))?,
            flag,
        )
    };
    Ok(Args {
        layer_count,
        execution_schedule_authority: require(
            execution_schedule_authority,
            "--execution-schedule-authority",
        )?,
        chain_cpu_oracle: require(chain_cpu_oracle, "--chain-cpu-oracle")?,
        l1_full_layer_assessment: require(l1_full_layer_assessment, "--l1-full-layer-assessment")?,
        joint_assessment: require(joint_assessment, "--joint-assessment")?,
        host_binary: require(host_binary, "--host-binary")?,
        out: require(out, "--out")?,
        workers,
    })
}

fn parse_metal_args(mut args: impl Iterator<Item = String>) -> Result<MetalArgs, String> {
    let mut layer_count = None;
    let mut outer_preflight = None;
    let mut lease_receipt = None;
    let mut outer_launch_authority = None;
    let mut outer_capture_dir = None;
    let mut capture_dir = None;
    let mut workers = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--layer-count" => {
                let raw = args.next().ok_or("--layer-count requires a value")?;
                if layer_count.replace(parse_layer_count(&raw)?).is_some() {
                    return Err("--layer-count may not be repeated".into());
                }
            }
            "--outer-preflight" => {
                let value = args.next().ok_or("--outer-preflight requires a value")?;
                if outer_preflight.replace(PathBuf::from(value)).is_some() {
                    return Err("--outer-preflight may not be repeated".into());
                }
            }
            "--lease-receipt" => {
                let value = args.next().ok_or("--lease-receipt requires a value")?;
                if lease_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("--lease-receipt may not be repeated".into());
                }
            }
            "--outer-launch-authority" => {
                let value = args
                    .next()
                    .ok_or("--outer-launch-authority requires a value")?;
                if outer_launch_authority
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err("--outer-launch-authority may not be repeated".into());
                }
            }
            "--outer-capture-dir" => {
                let value = args.next().ok_or("--outer-capture-dir requires a value")?;
                if outer_capture_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--outer-capture-dir may not be repeated".into());
                }
            }
            "--capture-dir" => {
                let value = args.next().ok_or("--capture-dir requires a value")?;
                if capture_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--capture-dir may not be repeated".into());
                }
            }
            "--workers" => {
                let raw = args.next().ok_or("--workers requires a value")?;
                if workers.replace(parse_workers(&raw)?).is_some() {
                    return Err("--workers may not be repeated".into());
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => {
                return Err(format!(
                    "metal mode refuses unsupported argument {other:?}; {}",
                    usage()
                ))
            }
        }
    }
    let layer_count = layer_count.ok_or("--layer-count is required")?;
    // Physical metal captures admitted for L0..L2 (DeltaNet-only, earned) and
    // L0..L3 (first GQA crossing). Longer chains remain blocked until each
    // intermediate handoff is proven.
    if layer_count != RECOMMENDED_DELTANET_LAYER_COUNT
        && layer_count != RECOMMENDED_FIRST_GQA_LAYER_COUNT
    {
        return Err(format!(
            "metal mode --layer-count observed={layer_count}, expected={RECOMMENDED_DELTANET_LAYER_COUNT} (L0..L2 DeltaNet-only) or {RECOMMENDED_FIRST_GQA_LAYER_COUNT} (L0..L3 first GQA crossing)"
        ));
    }
    let workers = workers.ok_or("--workers is required")?;
    Ok(MetalArgs {
        layer_count,
        outer_preflight: require_absolute(
            outer_preflight.ok_or("--outer-preflight is required")?,
            "--outer-preflight",
        )?,
        lease_receipt: require_absolute(
            lease_receipt.ok_or("--lease-receipt is required")?,
            "--lease-receipt",
        )?,
        outer_launch_authority: require_absolute(
            outer_launch_authority.ok_or("--outer-launch-authority is required")?,
            "--outer-launch-authority",
        )?,
        outer_capture_dir: require_absolute(
            outer_capture_dir.ok_or("--outer-capture-dir is required")?,
            "--outer-capture-dir",
        )?,
        capture_dir: require_absolute(
            capture_dir.ok_or("--capture-dir is required")?,
            "--capture-dir",
        )?,
        workers,
    })
}

fn parse_invocation(mut args: impl Iterator<Item = String>) -> Result<Invocation, String> {
    let mut mode = None;
    let mut remaining = Vec::new();
    while let Some(flag) = args.next() {
        if flag == "--mode" {
            let value = args.next().ok_or("--mode requires a value")?;
            if mode.replace(value).is_some() {
                return Err(format!("--mode may not be repeated; {}", usage()));
            }
        } else {
            remaining.push(flag);
        }
    }
    match mode.as_deref() {
        Some("preflight") => Ok(Invocation::Preflight(parse_preflight_args(remaining.into_iter())?)),
        Some("metal") => Ok(Invocation::Metal(parse_metal_args(remaining.into_iter())?)),
        Some(other) => Err(format!(
            "--mode observed={other}, expected=preflight or metal; {}",
            usage()
        )),
        None => Err(format!("an explicit --mode is required; {}", usage())),
    }
}

fn obj<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be object"))
}

fn text<'a>(map: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    map.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.{field} must be string"))
}

fn number(map: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    map.get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be unsigned integer"))
}

fn boolean(map: &Map<String, Value>, field: &str, expected: bool, label: &str) -> Result<(), String> {
    let observed = map
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{field} must be bool"))?;
    if observed != expected {
        return Err(format!(
            "{label}.{field} observed={observed}, expected={expected}"
        ));
    }
    Ok(())
}

fn validate_schedule_authority(value: &Value, layer_count: usize) -> Result<String, String> {
    let seal = verify_seal(value, "execution schedule authority")?;
    let root = obj(value, "execution schedule authority")?;
    if text(root, "schema", "schedule")? != QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA {
        return Err(format!(
            "schedule schema observed={}, expected={QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA}",
            text(root, "schema", "schedule")?
        ));
    }
    if text(root, "status", "schedule")? != QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS {
        return Err(format!(
            "schedule status observed={}, expected={QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS}",
            text(root, "status", "schedule")?
        ));
    }
    let source = obj(
        root.get("source_authority")
            .ok_or("schedule missing source_authority")?,
        "schedule source_authority",
    )?;
    if text(source, "source_revision", "schedule")? != QWEN80_SOURCE_REVISION {
        return Err(format!(
            "schedule source_revision observed={}, expected={QWEN80_SOURCE_REVISION}",
            text(source, "source_revision", "schedule")?
        ));
    }
    if text(source, "gravity_manifest_seal_sha256", "schedule")?
        != QWEN80_GRAVITY_MANIFEST_SEAL_SHA256
    {
        return Err(format!(
            "schedule gravity seal observed={}, expected={QWEN80_GRAVITY_MANIFEST_SEAL_SHA256}",
            text(source, "gravity_manifest_seal_sha256", "schedule")?
        ));
    }
    let layers = root
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("schedule.layers must be array")?;
    if layers.len() != QWEN80_LAYERS {
        return Err(format!(
            "schedule.layers len observed={}, expected={QWEN80_LAYERS}",
            layers.len()
        ));
    }
    for layer in 0..layer_count {
        let entry = obj(&layers[layer], &format!("schedule.layers[{layer}]"))?;
        let expected = qwen80_layer_execution_schedule(layer)?;
        let mixer = text(entry, "mixer", &format!("layer {layer}"))?;
        if mixer != expected.mixer.as_str() {
            return Err(format!(
                "schedule.layers[{layer}].mixer observed={mixer}, expected={}",
                expected.mixer.as_str()
            ));
        }
        if !expected.same_runtime_full_layer_encode_ready {
            return Err(format!(
                "layer {layer} mixer={} is not same-runtime full-layer encode ready (observed ready=false). Reduce --layer-count to a DeltaNet-only prefix (recommended {RECOMMENDED_FIRST_LAYER_COUNT} for L0..L2).",
                expected.mixer
            ));
        }
        let dispatch = number(
            entry,
            "full_layer_dispatch_count",
            &format!("layer {layer}"),
        )?;
        if dispatch != expected.full_layer_dispatch_count as u64 {
            return Err(format!(
                "schedule.layers[{layer}].full_layer_dispatch_count observed={dispatch}, expected={}",
                expected.full_layer_dispatch_count
            ));
        }
    }
    Ok(seal)
}

fn validate_chain_oracle(value: &Value, layer_count: usize) -> Result<String, String> {
    let seal = verify_seal(value, "chain cpu oracle")?;
    let root = obj(value, "chain cpu oracle")?;
    if text(root, "schema", "oracle")? != CHAIN_ORACLE_SCHEMA {
        return Err(format!(
            "oracle schema observed={}, expected={CHAIN_ORACLE_SCHEMA}",
            text(root, "schema", "oracle")?
        ));
    }
    let observed_count = number(root, "layer_count", "oracle")?;
    if observed_count != layer_count as u64 {
        return Err(format!(
            "oracle.layer_count observed={observed_count}, expected={layer_count}"
        ));
    }
    if root
        .get("includes_unready_gqa")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err(format!(
            "oracle includes_unready_gqa=true for layer_count={layer_count} (observed true, expected false); every mixer in the requested prefix must be same-runtime encode-ready"
        ));
    }
    let total = number(root, "total_dispatches_physical_capture", "oracle").or_else(|_| {
        // composed oracle uses total_dispatches
        number(root, "total_dispatches", "oracle")
    })?;
    let expected = qwen80_multi_layer_total_dispatches(layer_count, false)?;
    if total != expected as u64 {
        return Err(format!(
            "oracle total_dispatches observed={total}, expected={expected}"
        ));
    }
    Ok(seal)
}

fn validate_l1_assessment(value: &Value) -> Result<String, String> {
    let seal = verify_seal(value, "L1 full-layer assessment")?;
    let root = obj(value, "L1 assessment")?;
    if text(root, "schema", "L1 assessment")? != L1_ASSESSMENT_SCHEMA {
        return Err(format!(
            "L1 assessment schema observed={}, expected={L1_ASSESSMENT_SCHEMA}",
            text(root, "schema", "L1 assessment")?
        ));
    }
    if text(root, "status", "L1 assessment")? != L1_EARNED_STATUS {
        return Err(format!(
            "L1 assessment status observed={}, expected={L1_EARNED_STATUS}",
            text(root, "status", "L1 assessment")?
        ));
    }
    boolean(root, "earned_complete_l1_component_only", true, "L1 assessment")?;
    Ok(seal)
}

fn validate_joint_assessment(value: &Value) -> Result<String, String> {
    let seal = verify_seal(value, "joint assessment")?;
    let root = obj(value, "joint assessment")?;
    if text(root, "schema", "joint assessment")? != JOINT_ASSESSMENT_SCHEMA {
        return Err(format!(
            "joint assessment schema observed={}, expected={JOINT_ASSESSMENT_SCHEMA}",
            text(root, "schema", "joint assessment")?
        ));
    }
    if text(root, "status", "joint assessment")? != JOINT_ASSESSMENT_STATUS {
        return Err(format!(
            "joint assessment status observed={}, expected={JOINT_ASSESSMENT_STATUS}",
            text(root, "status", "joint assessment")?
        ));
    }
    boolean(root, "earned_component_only", true, "joint assessment")?;
    Ok(seal)
}

fn build_preflight(args: &Args) -> Result<Value, String> {
    Qwen80ExecutionScheduleSourceBinding::exact().validate_exact()?;
    let schedule = read_json(
        &args.execution_schedule_authority,
        "execution schedule authority",
    )?;
    let schedule_seal = validate_schedule_authority(&schedule, args.layer_count)?;
    let oracle = read_json(&args.chain_cpu_oracle, "chain cpu oracle")?;
    let oracle_seal = validate_chain_oracle(&oracle, args.layer_count)?;
    let l1 = read_json(&args.l1_full_layer_assessment, "L1 full-layer assessment")?;
    let l1_seal = validate_l1_assessment(&l1)?;
    let joint = read_json(&args.joint_assessment, "joint assessment")?;
    let joint_seal = validate_joint_assessment(&joint)?;
    let (host_bytes, host_sha) = file_sha(&args.host_binary, "host binary")?;

    let expected_kernels = qwen80_multi_layer_structural_kernel_trace(args.layer_count, false)?;
    let total_dispatches = expected_kernels.len();
    if total_dispatches != args.layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES {
        return Err(format!(
            "total_dispatches drifted: observed={total_dispatches}, expected={}",
            args.layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES
        ));
    }

    let mut per_layer = Vec::new();
    for layer in 0..args.layer_count {
        let schedule = qwen80_layer_execution_schedule(layer)?;
        per_layer.push(json!({
            "layer": layer,
            "mixer": schedule.mixer.as_str(),
            "state_slot": schedule.state_slot.slot,
            "domain": schedule.state_slot.domain.as_str(),
            "dispatch_count": schedule.full_layer_dispatch_count,
            "kernel_names": schedule.full_layer_kernel_names,
            "exclusive_caller_owned_slot": true,
        }));
    }

    // Cumulative dispatch offsets for diagnostics.
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    for layer in 0..args.layer_count {
        offsets.push(json!({"layer": layer, "dispatch_offset": cursor, "dispatch_count": 23}));
        cursor += 23;
    }

    let mut document = json!({
        "schema": HOST_PREFLIGHT_SCHEMA,
        "status": HOST_PREFLIGHT_STATUS,
        "source_token_id": SOURCE_TOKEN_ID,
        "layer_count": args.layer_count,
        "layers_inclusive_range": {
            "first": 0,
            "last": args.layer_count - 1,
        },
        "source_authority": {
            "source_revision": QWEN80_SOURCE_REVISION,
            "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
        },
        "execution_schedule_authority": {
            "path": args.execution_schedule_authority.to_string_lossy(),
            "document_seal_sha256": schedule_seal,
            "document_sha256": schedule_seal,
        },
        "chain_cpu_oracle": {
            "path": args.chain_cpu_oracle.to_string_lossy(),
            "document_seal_sha256": oracle_seal,
            "document_sha256": oracle_seal,
        },
        "l1_full_layer_assessment_provenance": {
            "path": args.l1_full_layer_assessment.to_string_lossy(),
            "document_seal_sha256": l1_seal,
            "document_sha256": l1_seal,
            "historical_component_only": true,
            "does_not_import_pinned_buffers": true,
        },
        "joint_assessment": {
            "path": args.joint_assessment.to_string_lossy(),
            "document_seal_sha256": joint_seal,
            "document_sha256": joint_seal,
            "metal_path_provenance_only": true,
            "does_not_import_pinned_buffers": true,
            "schema": JOINT_ASSESSMENT_SCHEMA,
        },
        "host_binary": {
            "path": args.host_binary.to_string_lossy(),
            "bytes": host_bytes,
            "sha256": host_sha,
        },
        "workers": args.workers,
        "execution_policy": {
            "one_runtime": true,
            "one_command_buffer": true,
            "single_fence_after_all_dispatches": true,
            "fence_count": 1,
            "non_timed": true,
            "structural_kernel_trace_required": true,
            "receipt_written_last": true,
            "caller_owned_per_layer_state_slots": true,
            "bounded_cpu_device_parity_per_retained_vector": true,
            "retained_max_abs_error_in_receipt": true,
            "not_tolerance_scalar_in_isolation": true,
            "total_dispatches": total_dispatches,
            "per_layer_dispatch_count": QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "dispatch_offsets": offsets,
        },
        "structural_kernel_trace": {
            "exact_order": true,
            "kernel_names": expected_kernels,
        },
        "per_layer_schedule": per_layer,
        "future_capture_schemas": {
            "inner": FUTURE_INNER_SCHEMA,
            "inner_status": FUTURE_INNER_STATUS,
            "outer": FUTURE_OUTER_SCHEMA,
            "outer_status": FUTURE_OUTER_STATUS,
            "release": FUTURE_RELEASE_SCHEMA,
            "release_status": FUTURE_RELEASE_STATUS,
        },
        "recommended_first_physical_capture": {
            "layer_count": RECOMMENDED_FIRST_GQA_LAYER_COUNT,
            "layers": "L0..L3",
            "total_dispatches": RECOMMENDED_FIRST_GQA_LAYER_COUNT
                * QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "gqa_layer": 3,
            "gqa_prefix_dispatches": 9,
            "gqa_full_layer_dispatches": 23,
            "frozen_gqa_mixer_prefix_kernels": [
                "qwen_next_direct_packed_input_rmsnorm",
                "qwen_binary_sign_scale_matvec",
                "qwen_binary_sign_scale_matvec",
                "qwen_binary_sign_scale_matvec",
                "qwen80_attention_qk_norm_rope_cache",
                "mha_decode_f32",
                "qwen80_attention_apply_sigmoid_gate",
                "qwen_binary_sign_scale_matvec",
                "qwen_next_add_residual",
            ],
            "reason": "L0..L3 is the first chain that crosses a GQA layer (layer 3) after the earned L0..L2 DeltaNet-only component; 92 dispatches, one command buffer, one fence",
            "earned_deltanet_prefix": {
                "layer_count": RECOMMENDED_DELTANET_LAYER_COUNT,
                "layers": "L0..L2",
                "total_dispatches": RECOMMENDED_DELTANET_LAYER_COUNT
                    * QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            },
        },
        "metal_path": {
            "preflight_only": true,
            "metal_context_or_dispatch_performed": false,
            "physical_capture_requires_owner_lease_and_admission": true,
            "capture_body_wired": true,
            "mode_metal_available": true,
            "total_dispatches_for_layer_count": total_dispatches,
            "frozen_kernel_order_layer_count_3": MULTI_LAYER_L0_L2_KERNELS.to_vec(),
            "encode_path": {
                "layer_0": "encode_source_input_l0_true_moe_capture (proven)",
                "layer_1": "encode_source_token_l1_deltanet_prefix + MoE suffix (proven)",
                "layer_2_plus_deltanet": "encode_source_token_deltanet_prefix_from_previous_second_residual_into + MoE suffix (proven)",
                "layer_3_gqa": "encode_source_token_gqa_prefix_from_previous_second_residual_into + MoE suffix (wired)",
                "finalizer": "finalize_after_exact_multi_layer_deltanet_chain_fence_with_readbacks",
                "gqa_layers": "encode-ready; physical parity pending owner L0..L3 capture",
            },
            "future_metal_entrypoint": {
                "explicit_mode_required": true,
                "default_execution_disabled": true,
                "requires_new_multi_layer_lease": true,
                "requires_sealed_outer_launch_authority": true,
                "requires_fresh_outer_and_inner_capture_directories": true,
                "capture_body_wired": true,
            },
        },
        "claim_boundary": {
            "host_preflight_only": true,
            "multi_layer_device_parity": false,
            "component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "server_or_watcher_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
            "test_only_fake_child": false,
            "fixture_or_synthetic": false,
        },
        "refusal_diagnostics_contract": {
            "every_refusal_carries_observed_vs_expected_values": true,
            "generic_undifferentiated_errors_are_defects": true,
        },
    });
    seal(&mut document)?;
    Ok(document)
}

/// Expected cumulative kernel list used by unit tests and future assessor.
pub fn expected_multi_layer_kernels(layer_count: usize) -> Result<Vec<&'static str>, String> {
    qwen80_multi_layer_structural_kernel_trace(layer_count, false)
}

/// Producer-convention provenance pointer: both sha fields carry the seal.
fn producer_pointer(path: &Path, seal: &str, bytes: u64, raw_sha: &str) -> Value {
    json!({
        "present": true,
        "path": path.to_string_lossy(),
        "bytes": bytes,
        "sha256": raw_sha,
        "document_sha256": seal,
        "document_seal_sha256": seal,
    })
}

/// Outer terminal template fields required by the multi-layer assessor.
/// Used by tests and documented for the lifecycle operator.
pub fn outer_terminal_producer_fields(
    lease_id: &str,
    inner_seal: &str,
    exit_code: u64,
) -> Value {
    json!({
        "schema": FUTURE_OUTER_SCHEMA,
        "status": FUTURE_OUTER_STATUS,
        "fixture_or_synthetic": false,
        "test_only_fake_child": false,
        "self_asserted": false,
        "lease_id": lease_id,
        "inner_capture": {
            "present": true,
            "document_sha256": inner_seal,
            "document_seal_sha256": inner_seal,
        },
        "child_terminal": {
            "exit_code": exit_code,
            "reaped": true,
            "terminal_receipt_written_last": true,
        },
        "claim_boundary": {
            "multi_layer_component_only": true,
            "test_only_fake_child": false,
            "token_generated": false,
            "decoder_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
        },
    })
}

/// Release receipt fields required by the multi-layer assessor (dual names).
pub fn release_producer_fields(lease_id: &str, outer_seal: &str, capture_succeeded: bool) -> Value {
    json!({
        "schema": FUTURE_RELEASE_SCHEMA,
        "status": FUTURE_RELEASE_STATUS,
        "lease_id": lease_id,
        "actual_release_performed": true,
        "released_after_outer_terminal": true,
        "release_after_outer_terminal": true,
        "lease_released": true,
        "automatic_retry_prohibited": true,
        "fresh_lease_required_for_any_future_gpu_work": true,
        "watcher_restart_or_transition_authorized": false,
        "capture_succeeded": capture_succeeded,
        "outer_terminal": {
            "present": true,
            "document_sha256": outer_seal,
            "document_seal_sha256": outer_seal,
        },
        "claim_boundary": {
            "multi_layer_component_only": true,
            "token_generated": false,
        },
    })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn file_evidence(path: &Path, label: &str) -> Result<FileEvidence, String> {
    let (bytes, sha) = file_sha(path, label)?;
    Ok(FileEvidence {
        path: path.to_path_buf(),
        bytes,
        sha256: sha,
    })
}

fn read_sealed_document(
    path: &Path,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<SealedDocument, String> {
    let file = file_evidence(path, label)?;
    let value = read_json(path, label)?;
    let seal = verify_seal(&value, label)?;
    let root = obj(&value, label)?;
    let schema = text(root, "schema", label)?;
    if schema != expected_schema {
        return Err(format!(
            "{label}.schema observed={schema}, expected={expected_schema}"
        ));
    }
    let status = text(root, "status", label)?;
    if status != expected_status {
        return Err(format!(
            "{label}.status observed={status}, expected={expected_status}"
        ));
    }
    let document_sha256 = canonical_json_sha(&value)?;
    Ok(SealedDocument {
        file,
        value,
        // Assessor wrapper uses json_sha of full sealed document; receipt
        // pointers use seal in document_sha256 (producer convention).
        document_sha256,
        seal_sha256: seal,
    })
}

fn main() {
    let outcome = parse_invocation(env::args().skip(1)).and_then(|invocation| match invocation {
        Invocation::Preflight(args) => {
            let document = build_preflight(&args)?;
            let seal = document["seal_sha256"]
                .as_str()
                .ok_or("missing seal")?
                .to_owned();
            write_new(&args.out, &document)?;
            Ok(json!({
                "schema": HOST_PREFLIGHT_SCHEMA,
                "status": HOST_PREFLIGHT_STATUS,
                "layer_count": args.layer_count,
                "seal_sha256": seal,
                "out": args.out.to_string_lossy(),
                "metal_or_gpu_activity_performed": false,
            }))
        }
        Invocation::Metal(args) => {
            #[cfg(target_os = "macos")]
            {
                run_metal_child(&args).map(|seal| {
                    json!({
                        "schema": FUTURE_INNER_SCHEMA,
                        "status": FUTURE_INNER_STATUS,
                        "layer_count": args.layer_count,
                        "seal_sha256": seal,
                        "metal_or_gpu_activity_performed": true,
                        "catalog_or_payload_scan_performed": true,
                    })
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = args;
                Err("strict multi-layer Metal capture is unavailable on this target".into())
            }
        }
    });
    match outcome {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!(
                "ascension_qwen80_source_token_multi_layer_same_runtime_device refused: {error}"
            );
            process::exit(2);
        }
    }
}

// ---------------------------------------------------------------------------
// Metal encode path (owner-run under resource admission; not exercised by CI)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn phase_document(phase: &MetalExecutionPhase) -> Value {
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
        "device_dispatch_may_have_occurred": phase.command_fence_succeeded,
    })
}

#[cfg(target_os = "macos")]
fn encode_source_token_multi_layer_same_runtime(
    runtime: &Qwen80CompleteNativeRuntime,
    command: TokenCommandBuffer<'_>,
    layer_count: usize,
    l0_source_outer_preflight: &Path,
    l1_route_authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    workers: usize,
    phase: &mut MetalExecutionPhase,
) -> Result<Qwen80SameRuntimeMultiLayerChainParity, String> {
    if layer_count != RECOMMENDED_DELTANET_LAYER_COUNT
        && layer_count != RECOMMENDED_FIRST_GQA_LAYER_COUNT
    {
        return Err(format!(
            "encode host --layer-count observed={layer_count}, expected={RECOMMENDED_DELTANET_LAYER_COUNT} or {RECOMMENDED_FIRST_GQA_LAYER_COUNT}"
        ));
    }
    encode_source_token_multi_layer_l0_l2_same_runtime(
        runtime,
        command,
        layer_count,
        l0_source_outer_preflight,
        l1_route_authority,
        workers,
        phase,
    )
}

#[cfg(target_os = "macos")]
fn encode_source_token_multi_layer_l0_l2_same_runtime(
    runtime: &Qwen80CompleteNativeRuntime,
    command: TokenCommandBuffer<'_>,
    layer_count: usize,
    l0_source_outer_preflight: &Path,
    l1_route_authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    workers: usize,
    phase: &mut MetalExecutionPhase,
) -> Result<Qwen80SameRuntimeMultiLayerChainParity, String> {
    let (l0_source_bridge, _) = source_l0::build_source_token_all_ten_bridge_from_outer_preflight(
        runtime,
        l0_source_outer_preflight,
        workers,
    )?;
    let l1_route = Qwen80RouteSelection {
        ids: l1_route_authority.route_ids,
        weights: l1_route_authority.route_weights,
    };
    let l1_source_bridge = runtime
        .catalog()
        .build_source_token_l1_all_ten_true_moe_source_bridge_from_validated_authority(
            GRAVITY_MANIFEST_DOCUMENT_SHA256,
            &l1_route_authority.route_authority_document_sha256,
            l1_route.clone(),
        )
        .map_err(|error| format!("same-runtime Layer-1 route bridge refused: {error}"))?;

    let mut command = command;
    command
        .enable_structural_kernel_trace()
        .map_err(|error| format!("multi-layer host requires structural trace: {error}"))?;
    phase.structural_kernel_trace_enabled = true;

    // --- L0 full layer (23) ---
    let l0_resources = source_prefix::encode_source_input_l0_true_moe_capture(
        runtime,
        &mut command,
        SOURCE_TOKEN_ID as u32,
        &l0_source_bridge,
    )?;
    let continuation = l0_resources.into_canonical_l0_true_moe_continuation(runtime, &command)?;

    // --- L1 prefix (9) ---
    let l1_prefix = runtime
        .encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into(
            &mut command,
            continuation,
        )
        .map_err(|error| format!("same-runtime Layer-1 prefix refused: {error}"))?;
    let l1_cpu = l1_prefix
        .derive_fresh_l1_full_cpu_oracle(runtime)
        .map_err(|error| format!("same-runtime Layer-1 CPU oracle refused: {error}"))?;
    if l1_cpu.route.ids != l1_route_authority.route_ids
        || l1_cpu
            .route
            .weights
            .iter()
            .zip(l1_route_authority.route_weights.iter())
            .any(|(a, b)| (a - b).abs() > 2.0e-5)
    {
        return Err(format!(
            "L1 sealed route authority drifted from fresh CPU oracle (ids observed={:?}, expected={:?})",
            l1_cpu.route.ids, l1_route_authority.route_ids
        ));
    }

    // --- L1 MoE suffix (14) ---
    let l1_route_bridge = runtime
        .upload_all_ten_true_moe_device_bridge(
            &l1_source_bridge,
            l1_prefix.first_residual().to_owned(),
        )
        .map_err(|error| format!("Layer-1 route upload refused: {error}"))?;
    let l1_fixed = runtime
        .upload_canonical_linear_moe_fixed_device_buffers(1)
        .map_err(|error| format!("Layer-1 fixed upload refused: {error}"))?;
    {
        let fixed_buffers = all_ten::Qwen80AllTenTrueMoeGraphFixedBuffers {
            postnorm_signs: &l1_fixed.postnorm.signs,
            postnorm_scales: &l1_fixed.postnorm.scales,
            postnorm_hidden: &l1_fixed.postnorm_hidden,
            router_signs: &l1_fixed.router.signs,
            router_scales: &l1_fixed.router.scales,
            router_logits: &l1_fixed.router_logits,
            router_probabilities: &l1_fixed.router_probabilities,
            router_route_ids: &l1_fixed.router_route_ids,
            router_route_weights: &l1_fixed.router_route_weights,
            route_guard: &l1_fixed.route_guard,
            route_gate: &l1_fixed.route_gate,
            route_up: &l1_fixed.route_up,
            route_activated: &l1_fixed.route_activated,
            route_weighted: &l1_fixed.route_weighted,
            shared_gate_signs: &l1_fixed.shared_gate_proj.signs,
            shared_gate_scales: &l1_fixed.shared_gate_proj.scales,
            shared_up_signs: &l1_fixed.shared_up_proj.signs,
            shared_up_scales: &l1_fixed.shared_up_proj.scales,
            shared_down_signs: &l1_fixed.shared_down_proj.signs,
            shared_down_scales: &l1_fixed.shared_down_proj.scales,
            shared_scalar_signs: &l1_fixed.shared_expert_gate.signs,
            shared_scalar_scales: &l1_fixed.shared_expert_gate.scales,
            shared_gate: &l1_fixed.shared_gate,
            shared_up: &l1_fixed.shared_up,
            shared_activated: &l1_fixed.shared_activated,
            shared_output: &l1_fixed.shared_output,
            shared_scalar_logit: &l1_fixed.shared_scalar_logit,
            gated_shared: &l1_fixed.gated_shared,
            routed_sum: &l1_fixed.routed_sum,
            second_residual: &l1_fixed.second_residual,
        };
        let graph = all_ten::Qwen80AllTenTrueMoeGraphBuffers::from_admitted_route_bridge(
            &l1_route_bridge,
            fixed_buffers,
        );
        let suffix = all_ten::encode_all_ten_true_moe_from_first_residual(&mut command, &graph)?;
        if suffix != L1_MOE_SUFFIX_DISPATCHES {
            return Err(format!(
                "L1 MoE suffix dispatches observed={suffix}, expected={L1_MOE_SUFFIX_DISPATCHES}"
            ));
        }
    }
    if command.dispatch_count() != L0_DISPATCHES + L1_PREFIX_DISPATCHES + L1_MOE_SUFFIX_DISPATCHES {
        return Err(format!(
            "after L1 full layer total dispatches observed={}, expected=46",
            command.dispatch_count()
        ));
    }

    // --- L2 prefix (9) from L1 second residual ---
    let l2_prefix = runtime
        .encode_source_token_deltanet_prefix_from_previous_second_residual_into(
            &mut command,
            2,
            l1_fixed.second_residual.to_owned(),
            &l1_cpu.layer_output,
            46,
        )
        .map_err(|error| format!("same-runtime Layer-2 prefix refused: {error}"))?;
    let l2_cpu = l2_prefix
        .derive_full_cpu_oracle(runtime)
        .map_err(|error| format!("same-runtime Layer-2 CPU oracle refused: {error}"))?;
    // Live CPU-oracle route for L2 (not from device).
    let l2_plan_sha = {
        let mut hasher = Sha256::new();
        hasher.update(b"qwen80-live-layer-2-cpu-route\0");
        for id in &l2_cpu.route.ids {
            hasher.update(id.to_le_bytes());
        }
        for w in &l2_cpu.route.weights {
            hasher.update(w.to_bits().to_le_bytes());
        }
        format!("{:x}", hasher.finalize())
    };
    let l2_source_bridge = runtime
        .build_source_token_layer_all_ten_true_moe_source_bridge_from_route(
            2,
            GRAVITY_MANIFEST_DOCUMENT_SHA256,
            &l2_plan_sha,
            l2_cpu.route.clone(),
        )
        .map_err(|error| format!("Layer-2 route bridge refused: {error}"))?;
    let l2_route_bridge = runtime
        .upload_all_ten_true_moe_device_bridge(
            &l2_source_bridge,
            l2_prefix.first_residual().to_owned(),
        )
        .map_err(|error| format!("Layer-2 route upload refused: {error}"))?;
    let l2_fixed = runtime
        .upload_canonical_linear_moe_fixed_device_buffers(2)
        .map_err(|error| format!("Layer-2 fixed upload refused: {error}"))?;
    {
        let fixed_buffers = all_ten::Qwen80AllTenTrueMoeGraphFixedBuffers {
            postnorm_signs: &l2_fixed.postnorm.signs,
            postnorm_scales: &l2_fixed.postnorm.scales,
            postnorm_hidden: &l2_fixed.postnorm_hidden,
            router_signs: &l2_fixed.router.signs,
            router_scales: &l2_fixed.router.scales,
            router_logits: &l2_fixed.router_logits,
            router_probabilities: &l2_fixed.router_probabilities,
            router_route_ids: &l2_fixed.router_route_ids,
            router_route_weights: &l2_fixed.router_route_weights,
            route_guard: &l2_fixed.route_guard,
            route_gate: &l2_fixed.route_gate,
            route_up: &l2_fixed.route_up,
            route_activated: &l2_fixed.route_activated,
            route_weighted: &l2_fixed.route_weighted,
            shared_gate_signs: &l2_fixed.shared_gate_proj.signs,
            shared_gate_scales: &l2_fixed.shared_gate_proj.scales,
            shared_up_signs: &l2_fixed.shared_up_proj.signs,
            shared_up_scales: &l2_fixed.shared_up_proj.scales,
            shared_down_signs: &l2_fixed.shared_down_proj.signs,
            shared_down_scales: &l2_fixed.shared_down_proj.scales,
            shared_scalar_signs: &l2_fixed.shared_expert_gate.signs,
            shared_scalar_scales: &l2_fixed.shared_expert_gate.scales,
            shared_gate: &l2_fixed.shared_gate,
            shared_up: &l2_fixed.shared_up,
            shared_activated: &l2_fixed.shared_activated,
            shared_output: &l2_fixed.shared_output,
            shared_scalar_logit: &l2_fixed.shared_scalar_logit,
            gated_shared: &l2_fixed.gated_shared,
            routed_sum: &l2_fixed.routed_sum,
            second_residual: &l2_fixed.second_residual,
        };
        let graph = all_ten::Qwen80AllTenTrueMoeGraphBuffers::from_admitted_route_bridge(
            &l2_route_bridge,
            fixed_buffers,
        );
        let suffix = all_ten::encode_all_ten_true_moe_from_first_residual(&mut command, &graph)?;
        if suffix != L1_MOE_SUFFIX_DISPATCHES {
            return Err(format!(
                "L2 MoE suffix dispatches observed={suffix}, expected={L1_MOE_SUFFIX_DISPATCHES}"
            ));
        }
    }

    // After L2 full layer: 69 dispatches.
    if command.dispatch_count() != 69 {
        return Err(format!(
            "after L2 full layer total dispatches observed={}, expected=69",
            command.dispatch_count()
        ));
    }

    let mut subsequent: Vec<Qwen80SameRuntimeSubsequentLayerPrefix> =
        vec![Qwen80SameRuntimeSubsequentLayerPrefix::DeltaNet(l2_prefix)];
    let mut suffixes = vec![
        Qwen80MultiLayerSuffixWitness {
            layer: 1,
            fixed: l1_fixed,
            cpu: Qwen80MultiLayerSuffixCpuOracle::Linear(l1_cpu),
        },
        Qwen80MultiLayerSuffixWitness {
            layer: 2,
            fixed: l2_fixed,
            cpu: Qwen80MultiLayerSuffixCpuOracle::Linear(l2_cpu),
        },
    ];

    // Optional L3 GQA full layer (23) when layer_count=4.
    if layer_count == RECOMMENDED_FIRST_GQA_LAYER_COUNT {
        let l2_second_residual = suffixes[1].fixed.second_residual.to_owned();
        let l2_layer_output = match &suffixes[1].cpu {
            Qwen80MultiLayerSuffixCpuOracle::Linear(cpu) => cpu.layer_output.clone(),
            Qwen80MultiLayerSuffixCpuOracle::Gqa(cpu) => cpu.layer_output.clone(),
        };
        let l3_prefix = runtime
            .encode_source_token_gqa_prefix_from_previous_second_residual_into(
                &mut command,
                3,
                l2_second_residual,
                &l2_layer_output,
                69,
                0, // source-token first position
            )
            .map_err(|error| format!("same-runtime GQA Layer-3 prefix refused: {error}"))?;
        let l3_cpu = l3_prefix
            .derive_full_cpu_oracle(runtime)
            .map_err(|error| format!("same-runtime GQA Layer-3 CPU oracle refused: {error}"))?;
        let l3_plan_sha = {
            let mut hasher = Sha256::new();
            hasher.update(b"qwen80-live-layer-3-gqa-cpu-route\0");
            for id in &l3_cpu.route.ids {
                hasher.update(id.to_le_bytes());
            }
            for w in &l3_cpu.route.weights {
                hasher.update(w.to_bits().to_le_bytes());
            }
            format!("{:x}", hasher.finalize())
        };
        let l3_source_bridge = runtime
            .build_source_token_layer_all_ten_true_moe_source_bridge_from_route(
                3,
                GRAVITY_MANIFEST_DOCUMENT_SHA256,
                &l3_plan_sha,
                l3_cpu.route.clone(),
            )
            .map_err(|error| format!("Layer-3 GQA route bridge refused: {error}"))?;
        let l3_route_bridge = runtime
            .upload_all_ten_true_moe_device_bridge(
                &l3_source_bridge,
                l3_prefix.first_residual().to_owned(),
            )
            .map_err(|error| format!("Layer-3 GQA route upload refused: {error}"))?;
        let l3_fixed = runtime
            .upload_canonical_gqa_moe_fixed_device_buffers(3)
            .map_err(|error| format!("Layer-3 GQA fixed upload refused: {error}"))?;
        {
            let fixed_buffers = all_ten::Qwen80AllTenTrueMoeGraphFixedBuffers {
                postnorm_signs: &l3_fixed.postnorm.signs,
                postnorm_scales: &l3_fixed.postnorm.scales,
                postnorm_hidden: &l3_fixed.postnorm_hidden,
                router_signs: &l3_fixed.router.signs,
                router_scales: &l3_fixed.router.scales,
                router_logits: &l3_fixed.router_logits,
                router_probabilities: &l3_fixed.router_probabilities,
                router_route_ids: &l3_fixed.router_route_ids,
                router_route_weights: &l3_fixed.router_route_weights,
                route_guard: &l3_fixed.route_guard,
                route_gate: &l3_fixed.route_gate,
                route_up: &l3_fixed.route_up,
                route_activated: &l3_fixed.route_activated,
                route_weighted: &l3_fixed.route_weighted,
                shared_gate_signs: &l3_fixed.shared_gate_proj.signs,
                shared_gate_scales: &l3_fixed.shared_gate_proj.scales,
                shared_up_signs: &l3_fixed.shared_up_proj.signs,
                shared_up_scales: &l3_fixed.shared_up_proj.scales,
                shared_down_signs: &l3_fixed.shared_down_proj.signs,
                shared_down_scales: &l3_fixed.shared_down_proj.scales,
                shared_scalar_signs: &l3_fixed.shared_expert_gate.signs,
                shared_scalar_scales: &l3_fixed.shared_expert_gate.scales,
                shared_gate: &l3_fixed.shared_gate,
                shared_up: &l3_fixed.shared_up,
                shared_activated: &l3_fixed.shared_activated,
                shared_output: &l3_fixed.shared_output,
                shared_scalar_logit: &l3_fixed.shared_scalar_logit,
                gated_shared: &l3_fixed.gated_shared,
                routed_sum: &l3_fixed.routed_sum,
                second_residual: &l3_fixed.second_residual,
            };
            let graph = all_ten::Qwen80AllTenTrueMoeGraphBuffers::from_admitted_route_bridge(
                &l3_route_bridge,
                fixed_buffers,
            );
            let suffix = all_ten::encode_all_ten_true_moe_from_first_residual(&mut command, &graph)?;
            if suffix != L1_MOE_SUFFIX_DISPATCHES {
                return Err(format!(
                    "L3 GQA MoE suffix dispatches observed={suffix}, expected={L1_MOE_SUFFIX_DISPATCHES}"
                ));
            }
        }
        subsequent.push(Qwen80SameRuntimeSubsequentLayerPrefix::Gqa(l3_prefix));
        suffixes.push(Qwen80MultiLayerSuffixWitness {
            layer: 3,
            fixed: l3_fixed,
            cpu: Qwen80MultiLayerSuffixCpuOracle::Gqa(l3_cpu),
        });
    }

    let expected_total = layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES;
    let expected_kernels = expected_multi_layer_kernels(layer_count)?;
    if command.dispatch_count() != expected_total
        || command
            .structural_kernel_names()
            .ok_or("multi-layer host did not retain structural trace")?
            .iter()
            .map(String::as_str)
            .ne(expected_kernels.iter().copied())
    {
        let names = command
            .structural_kernel_names()
            .map(|n| n.len())
            .unwrap_or(0);
        return Err(format!(
            "multi-layer L0..L{} structural trace drifted: dispatch_count observed={}, expected={expected_total}; kernel_names_len observed={names}, expected={}",
            layer_count - 1,
            command.dispatch_count(),
            expected_kernels.len()
        ));
    }
    phase.dispatches_encoded = command.dispatch_count();
    phase.encoded_kernel_names = command
        .structural_kernel_names()
        .ok_or("multi-layer host did not retain structural trace")?
        .to_vec();
    phase.command_commit_may_have_been_attempted = true;

    // Recover L1 prefix owner: still held as the first subsequent's peer is L2.
    // The l1_prefix was moved into... wait, l1_prefix is still in scope above.
    let parity = l1_prefix
        .finalize_after_exact_multi_layer_deltanet_chain_fence_with_readbacks(
            runtime,
            command,
            &suffixes,
            &subsequent,
        )
        .map_err(|error| format!("multi-layer exact finalizer refused: {error}"))?;
    phase.command_fence_succeeded = true;
    phase.readback_started = true;
    Ok(parity)
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct MultiLayerCaptureAuthority {
    outer_preflight: SealedDocument,
    host_preflight: SealedDocument,
    schedule: SealedDocument,
    chain_oracle: SealedDocument,
    l1_assessment: SealedDocument,
    joint_assessment: SealedDocument,
    l0_source_outer_preflight: SealedDocument,
    l1_route_authority: completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    manifest: SealedDocument,
    admission_receipt: SealedDocument,
    lease: SealedDocument,
    lease_id: String,
    outer_launch: SealedDocument,
    host_binary: FileEvidence,
    workers: usize,
    layer_count: usize,
}

#[cfg(target_os = "macos")]
fn validate_metal_gate(args: &MetalArgs) -> Result<(SealedDocument, SealedDocument, SealedDocument, FileEvidence, String), String> {
    // File-only gate: no Metal context, no catalog open.
    let outer = read_sealed_document(
        &args.outer_preflight,
        "multi-layer outer CPU preflight",
        OUTER_PREFLIGHT_SCHEMA,
        OUTER_PREFLIGHT_STATUS,
    )?;
    let own_path = env::current_exe().map_err(|e| format!("current exe: {e}"))?;
    let own = file_evidence(&own_path, "current multi-layer host executable")?;
    let outer_root = obj(&outer.value, "multi-layer outer CPU preflight")?;
    let host_bin = obj(
        outer_root
            .get("host_binary")
            .ok_or("outer preflight missing host_binary")?,
        "outer host_binary",
    )?;
    if text(host_bin, "sha256", "outer host_binary")? != own.sha256 {
        return Err(format!(
            "outer preflight host_binary.sha256 observed={}, expected current host {}",
            text(host_bin, "sha256", "outer host_binary")?,
            own.sha256
        ));
    }
    let scope = obj(
        outer_root
            .get("exact_component_scope")
            .ok_or("outer preflight missing exact_component_scope")?,
        "outer scope",
    )?;
    let total = number(scope, "total_dispatches", "outer scope")?;
    let expected_total =
        (args.layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES) as u64;
    if total != expected_total {
        return Err(format!(
            "outer preflight total_dispatches observed={total}, expected={expected_total}"
        ));
    }
    let lease = read_sealed_document(
        &args.lease_receipt,
        "multi-layer quiet Metal lease",
        METAL_LEASE_SCHEMA,
        METAL_LEASE_STATUS,
    )?;
    let lease_root = obj(&lease.value, "lease")?;
    let lease_id = text(lease_root, "lease_id", "lease")?.to_owned();
    if !is_sha256(&lease_id) {
        return Err(format!(
            "lease.lease_id must be lowercase SHA-256 (len observed={})",
            lease_id.len()
        ));
    }
    let launch = read_sealed_document(
        &args.outer_launch_authority,
        "multi-layer outer launch authority",
        METAL_OUTER_LAUNCH_SCHEMA,
        METAL_OUTER_LAUNCH_STATUS,
    )?;
    let launch_root = obj(&launch.value, "launch")?;
    if text(launch_root, "lease_id", "launch")? != lease_id {
        return Err(format!(
            "outer launch lease_id observed={}, expected={lease_id}",
            text(launch_root, "lease_id", "launch")?
        ));
    }
    if text(launch_root, "planned_outer_capture_dir", "launch")?
        != args.outer_capture_dir.to_string_lossy()
        || text(launch_root, "planned_inner_capture_dir", "launch")?
            != args.capture_dir.to_string_lossy()
    {
        return Err(format!(
            "outer launch capture paths drifted: planned_outer observed={}, expected={}; planned_inner observed={}, expected={}",
            text(launch_root, "planned_outer_capture_dir", "launch")?,
            args.outer_capture_dir.display(),
            text(launch_root, "planned_inner_capture_dir", "launch")?,
            args.capture_dir.display()
        ));
    }
    Ok((outer, lease, launch, own, lease_id))
}

#[cfg(target_os = "macos")]
fn resolve_multi_layer_authority(
    args: &MetalArgs,
    outer: SealedDocument,
    lease: SealedDocument,
    outer_launch: SealedDocument,
    host_binary: FileEvidence,
    lease_id: String,
) -> Result<MultiLayerCaptureAuthority, String> {
    let outer_root = obj(&outer.value, "outer preflight")?;
    let load = |field: &str, schema: &str, status: &str| -> Result<SealedDocument, String> {
        let binding = obj(
            outer_root
                .get(field)
                .ok_or_else(|| format!("outer preflight missing {field}"))?,
            field,
        )?;
        let path = PathBuf::from(text(binding, "path", field)?);
        let doc = read_sealed_document(&path, field, schema, status)?;
        let expected_seal = text(binding, "document_seal_sha256", field)
            .or_else(|_| text(binding, "document_sha256", field))?;
        if doc.seal_sha256 != expected_seal {
            return Err(format!(
                "outer preflight {field} seal observed={}, binding expected={expected_seal}",
                doc.seal_sha256
            ));
        }
        Ok(doc)
    };
    let host_preflight = load(
        "host_preflight",
        HOST_PREFLIGHT_SCHEMA,
        HOST_PREFLIGHT_STATUS,
    )?;
    let schedule = load(
        "execution_schedule_authority",
        QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA,
        QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
    )?;
    let chain_oracle = {
        let binding = obj(
            outer_root
                .get("chain_cpu_oracle")
                .ok_or("outer preflight missing chain_cpu_oracle")?,
            "chain_cpu_oracle",
        )?;
        let path = PathBuf::from(text(binding, "path", "chain_cpu_oracle")?);
        let file = file_evidence(&path, "chain_cpu_oracle")?;
        let value = read_json(&path, "chain_cpu_oracle")?;
        let seal = verify_seal(&value, "chain_cpu_oracle")?;
        let root = obj(&value, "chain_cpu_oracle")?;
        if text(root, "schema", "chain_cpu_oracle")? != CHAIN_ORACLE_SCHEMA {
            return Err(format!(
                "chain_cpu_oracle.schema observed={}, expected={CHAIN_ORACLE_SCHEMA}",
                text(root, "schema", "chain_cpu_oracle")?
            ));
        }
        SealedDocument {
            file,
            document_sha256: canonical_json_sha(&value)?,
            value,
            seal_sha256: seal,
        }
    };
    let l1_assessment = load(
        "l1_full_layer_assessment",
        L1_ASSESSMENT_SCHEMA,
        L1_EARNED_STATUS,
    )?;
    // Metal-path provenance: joint post-capture assessment (not the L1 completion
    // assessment). The shared L0+L1 route-authority validator requires this schema.
    let joint_assessment = load(
        "joint_assessment",
        JOINT_ASSESSMENT_SCHEMA,
        JOINT_ASSESSMENT_STATUS,
    )?;
    let l0_source_outer_preflight = load(
        "l0_source_outer_preflight",
        L0_SOURCE_OUTER_SCHEMA,
        L0_SOURCE_OUTER_STATUS,
    )?;
    let route_document = load(
        "original_l1_route_authority",
        L1_ROUTE_AUTHORITY_SCHEMA,
        L1_ROUTE_AUTHORITY_STATUS,
    )?;
    // Prefer host-preflight joint binding when present so host/outer cannot drift.
    let host_root = obj(&host_preflight.value, "host preflight")?;
    if let Some(host_joint) = host_root.get("joint_assessment") {
        let host_joint = obj(host_joint, "host preflight.joint_assessment")?;
        let host_joint_seal = text(host_joint, "document_seal_sha256", "host joint")
            .or_else(|_| text(host_joint, "document_sha256", "host joint"))?;
        if joint_assessment.seal_sha256 != host_joint_seal {
            return Err(format!(
                "host/outer joint assessment seal observed outer={}, host expected={host_joint_seal}",
                joint_assessment.seal_sha256
            ));
        }
    }
    let l1_route_authority = completion_preflight::validate_source_token_l1_route_authority_files(
        &joint_assessment.file.path,
        &route_document.file.path,
    )?;
    // Source artifact chain from L0 outer.
    let l0_root = obj(&l0_source_outer_preflight.value, "l0 outer")?;
    let source = obj(
        l0_root
            .get("source_binding")
            .ok_or("l0 outer missing source_binding")?,
        "source_binding",
    )?;
    let manifest = {
        let m = obj(
            source
                .get("manifest")
                .ok_or("source_binding missing manifest")?,
            "manifest",
        )?;
        let path = PathBuf::from(text(m, "path", "manifest")?);
        read_sealed_document(&path, "manifest", MANIFEST_SCHEMA, MANIFEST_STATUS)?
    };
    let admission_receipt = {
        let a = obj(
            source
                .get("admission_receipt")
                .ok_or("source_binding missing admission_receipt")?,
            "admission_receipt",
        )?;
        let path = PathBuf::from(text(a, "path", "admission_receipt")?);
        read_sealed_document(
            &path,
            "admission_receipt",
            ADMISSION_RECEIPT_SCHEMA,
            ADMISSION_RECEIPT_STATUS,
        )?
    };
    Ok(MultiLayerCaptureAuthority {
        outer_preflight: outer,
        host_preflight,
        schedule,
        chain_oracle,
        l1_assessment,
        joint_assessment,
        l0_source_outer_preflight,
        l1_route_authority,
        manifest,
        admission_receipt,
        lease,
        lease_id,
        outer_launch,
        host_binary,
        workers: args.workers,
        layer_count: args.layer_count,
    })
}

#[cfg(target_os = "macos")]
fn source_artifact_admission(
    authority: &MultiLayerCaptureAuthority,
) -> Result<CompleteBinaryAdmission, String> {
    let manifest = obj(&authority.manifest.value, "manifest")?;
    let source_audit_seal = text(manifest, "source_body_audit_seal_sha256", "manifest")?.to_owned();
    if !is_sha256(&source_audit_seal) {
        return Err(format!(
            "manifest source_body_audit_seal_sha256 malformed (len={})",
            source_audit_seal.len()
        ));
    }
    let revalidation_path = PathBuf::from(text(
        manifest,
        "source_revalidation_receipt_path",
        "manifest",
    )?);
    let expected_revalidation_seal =
        text(manifest, "source_revalidation_receipt_seal_sha256", "manifest")?;
    if !is_sha256(expected_revalidation_seal) {
        return Err(format!(
            "manifest source_revalidation_receipt_seal_sha256 malformed (len={})",
            expected_revalidation_seal.len()
        ));
    }
    let revalidation = completion_preflight::read_verified_sealed_document(
        &revalidation_path,
        "source revalidation",
    )?;
    if revalidation.document_seal_sha256 != expected_revalidation_seal {
        return Err(format!(
            "source revalidation seal observed={}, expected={expected_revalidation_seal}",
            revalidation.document_seal_sha256
        ));
    }
    let revalidation_root = obj(&revalidation.value, "source revalidation")?;
    if text(revalidation_root, "source_audit_seal_sha256", "revalidation")? != source_audit_seal {
        return Err(format!(
            "source revalidation audit seal observed={}, expected={source_audit_seal}",
            text(revalidation_root, "source_audit_seal_sha256", "revalidation")?
        ));
    }
    let source_revision = text(revalidation_root, "source_revision", "revalidation")?.to_owned();
    if source_revision.len() != 40
        || !source_revision.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
    {
        return Err(format!(
            "source revalidation revision malformed (len={}, expected=40 hex)",
            source_revision.len()
        ));
    }
    let _ = &authority.admission_receipt; // retained for provenance binding
    Ok(CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest.seal_sha256.clone(),
        expected_source_audit_seal_sha256: source_audit_seal,
        expected_source_revision: source_revision,
    })
}

#[cfg(target_os = "macos")]
fn build_success_inner_receipt(
    authority: &MultiLayerCaptureAuthority,
    capture: &Path,
    parity: &Qwen80SameRuntimeMultiLayerChainParity,
    phase: &MetalExecutionPhase,
) -> Result<Value, String> {
    if !phase.command_fence_succeeded
        || phase.dispatches_encoded != parity.total_dispatches
        || parity.layer_count != authority.layer_count
    {
        return Err(format!(
            "success receipt cannot be built without fenced multi-layer evidence (fence={}, dispatches observed={}, expected={}, layer_count observed={}, expected={})",
            phase.command_fence_succeeded,
            phase.dispatches_encoded,
            parity.total_dispatches,
            parity.layer_count,
            authority.layer_count
        ));
    }
    let mut per_layer_readbacks = Vec::new();
    // Layer 0 second residual from fresh_l0.
    per_layer_readbacks.push(json!({
        "layer": 0,
        "output_elements": 2048,
        "second_residual_output": {
            "passed": true,
            "cpu_f32le_sha256": parity.fresh_l0.second_residual_cpu_f32le_sha256,
            "device_f32le_sha256": parity.fresh_l0.second_residual_device_f32le_sha256,
            "max_abs_error": parity.fresh_l0.second_residual_max_abs_error,
        },
    }));
    for suffix in &parity.per_layer_suffix {
        per_layer_readbacks.push(json!({
            "layer": suffix.layer,
            "output_elements": 2048,
            "second_residual_output": {
                "passed": true,
                "cpu_f32le_sha256": suffix.second_residual_cpu_f32le_sha256,
                "device_f32le_sha256": suffix.second_residual_output_f32le_sha256,
                "max_abs_error": suffix.second_residual_max_abs_error,
            },
        }));
    }
    let mut receipt = json!({
        "schema": FUTURE_INNER_SCHEMA,
        "status": FUTURE_INNER_STATUS,
        "fixture_or_synthetic": false,
        "self_asserted": false,
        "capture_body_wired": true,
        "lease_id": authority.lease_id,
        "execution_schedule_provenance": producer_pointer(
            &authority.schedule.file.path,
            &authority.schedule.seal_sha256,
            authority.schedule.file.bytes,
            &authority.schedule.file.sha256,
        ),
        "chain_cpu_oracle_provenance": producer_pointer(
            &authority.chain_oracle.file.path,
            &authority.chain_oracle.seal_sha256,
            authority.chain_oracle.file.bytes,
            &authority.chain_oracle.file.sha256,
        ),
        "host_preflight_provenance": producer_pointer(
            &authority.host_preflight.file.path,
            &authority.host_preflight.seal_sha256,
            authority.host_preflight.file.bytes,
            &authority.host_preflight.file.sha256,
        ),
        "fresh_same_runtime_execution": {
            "fresh_runtime": true,
            "same_runtime": true,
            "same_tcb": true,
            "single_fence_after_all_dispatches": true,
            "layer_count": authority.layer_count,
            "total_dispatches": parity.total_dispatches,
            "fence_count": 1,
            "per_layer_dispatches": QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
        },
        "structural_kernel_trace": {
            "exact_order": true,
            "kernel_names": parity.structural_kernel_names,
        },
        "per_layer_readbacks": per_layer_readbacks,
        "retained_max_abs_error": parity.retained_max_abs_error,
        "execution_phase": phase_document(phase),
        "durable_capture": {
            "capture_directory": capture.to_string_lossy(),
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_capture_required": true,
            "replay_guarded": true,
        },
        "claim_boundary": {
            "multi_layer_component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "server_or_watcher_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
            "test_only_fake_child": false,
            "fixture_or_synthetic": false,
        },
    });
    seal(&mut receipt)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn refusal_inner_receipt(
    authority: &MultiLayerCaptureAuthority,
    capture: &Path,
    phase: &MetalExecutionPhase,
    error: &str,
) -> Result<Value, String> {
    let mut receipt = json!({
        "schema": FUTURE_INNER_SCHEMA,
        "status": FUTURE_INNER_REFUSED_STATUS,
        "fixture_or_synthetic": false,
        "self_asserted": false,
        "capture_body_wired": true,
        "lease_id": authority.lease_id,
        "execution_phase": phase_document(phase),
        "terminal_error": error,
        "durable_capture": {
            "capture_directory": capture.to_string_lossy(),
            "receipt_written_last_is_completion_marker": true,
        },
        "claim_boundary": {
            "multi_layer_component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "tps_or_tg_measured": false,
        },
    });
    seal(&mut receipt)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn run_metal_child(args: &MetalArgs) -> Result<String, String> {
    let (outer, lease, launch, host_binary, lease_id) = validate_metal_gate(args)?;
    let authority = resolve_multi_layer_authority(
        args,
        outer,
        lease,
        launch,
        host_binary,
        lease_id,
    )?;
    if !args.outer_capture_dir.is_dir() {
        return Err(format!(
            "--outer-capture-dir must be an existing directory (observed path={}, is_dir=false)",
            args.outer_capture_dir.display()
        ));
    }
    if args.capture_dir.exists() {
        return Err(format!(
            "--capture-dir must be create-new; {} exists",
            args.capture_dir.display()
        ));
    }
    if args.capture_dir.parent() != Some(args.outer_capture_dir.as_path()) {
        // Soft check: capture should be direct child when parent exists.
        // Allow any absolute path under outer when parent chain matches.
        let capture_parent = args.capture_dir.parent().map(Path::to_path_buf);
        let outer_canon = fs::canonicalize(&args.outer_capture_dir)
            .map_err(|e| format!("canonicalize outer capture dir: {e}"))?;
        if capture_parent.as_ref() != Some(&outer_canon)
            && capture_parent.as_ref() != Some(&args.outer_capture_dir)
        {
            return Err(format!(
                "--capture-dir must be a direct child of --outer-capture-dir (observed parent={:?}, expected={})",
                capture_parent,
                args.outer_capture_dir.display()
            ));
        }
    }
    fs::create_dir(&args.capture_dir).map_err(|e| {
        format!(
            "cannot create multi-layer inner capture directory {}: {e}",
            args.capture_dir.display()
        )
    })?;
    let capture = fs::canonicalize(&args.capture_dir)
        .map_err(|e| format!("canonicalize capture dir: {e}"))?;
    let mut phase = MetalExecutionPhase::default();

    let mut invocation = json!({
        "schema": "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_inner_invocation.v1",
        "status": "STARTED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_STRICT_METAL_CHILD_OUTER_REAPED",
        "mode": "metal",
        "layer_count": authority.layer_count,
        "lease_id": authority.lease_id,
        "capture_body_wired": true,
        "execution_policy": {
            "source_token_id": SOURCE_TOKEN_ID,
            "total_dispatches": authority.layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "single_fence_required": true,
            "non_timed": true,
        },
    });
    seal(&mut invocation)?;
    write_new(&capture.join("invocation.json"), &invocation)?;

    let outcome = (|| -> Result<Value, String> {
        phase.strict_artifact_admission_started = true;
        let admission = source_artifact_admission(&authority)?;
        let catalog =
            Qwen80CompleteArtifactCatalog::load(&authority.manifest.file.path, &admission)
                .map_err(|e| format!("strict artifact admission failed: {e}"))?;
        phase.strict_artifact_admission_succeeded = true;
        phase.metal_context_construction_attempted = true;
        let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
            catalog,
            Qwen80CompleteRuntimeOptions {
                max_seq_len: 1,
                trace_dispatch: false,
            },
        )
        .map_err(|e| format!("strict-Math runtime construction failed: {e}"))?;
        phase.metal_context_constructed = true;
        let command = runtime.begin_component_token_command_buffer();
        let parity = encode_source_token_multi_layer_same_runtime(
            &runtime,
            command,
            authority.layer_count,
            &authority.l0_source_outer_preflight.file.path,
            &authority.l1_route_authority,
            authority.workers,
            &mut phase,
        )?;
        build_success_inner_receipt(&authority, &capture, &parity, &phase)
    })();

    match outcome {
        Ok(receipt) => {
            write_new(&capture.join("receipt.json"), &receipt)?;
            Ok(receipt["seal_sha256"]
                .as_str()
                .unwrap_or_default()
                .to_owned())
        }
        Err(error) => {
            let receipt = refusal_inner_receipt(&authority, &capture, &phase, &error)?;
            write_new(&capture.join("receipt.json"), &receipt)?;
            let seal = receipt["seal_sha256"].as_str().unwrap_or_default();
            Err(format!(
                "multi-layer strict-Metal child sealed a phase-accurate terminal refusal {seal}: {error}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_core::model::qwen80_48_layer_execution_schedule::{
        qwen80_all_48_layer_execution_schedules, validate_full_48_layer_schedule,
    };

    fn sealed_schedule() -> Value {
        let layers = qwen80_all_48_layer_execution_schedules().unwrap();
        validate_full_48_layer_schedule(&layers).unwrap();
        let layer_entries: Vec<Value> = layers
            .iter()
            .map(|layer| {
                json!({
                    "layer": layer.layer,
                    "mixer": layer.mixer.as_str(),
                    "source_layer_type": layer.source_layer_type,
                    "state_slot": {
                        "layer": layer.state_slot.layer,
                        "slot": layer.state_slot.slot,
                        "domain": layer.state_slot.domain.as_str(),
                        "device_buffers_required_before_execution": layer.state_slot.device_buffers_required_before_execution,
                        "rollback_buffers_required_before_execution": layer.state_slot.rollback_buffers_required_before_execution,
                        "exclusive_caller_owned_slot": true,
                    },
                    "mixer_prefix_dispatch_count": layer.mixer_prefix_dispatch_count,
                    "moe_suffix_dispatch_count": layer.moe_suffix_dispatch_count,
                    "full_layer_dispatch_count": layer.full_layer_dispatch_count,
                    "mixer_prefix_kernel_names": layer.mixer_prefix_kernel_names,
                    "moe_suffix_kernel_names": layer.moe_suffix_kernel_names,
                    "full_layer_kernel_names": layer.full_layer_kernel_names,
                    "residency": {
                        "input_hidden_elements": 2048,
                        "output_hidden_elements": 2048,
                        "mixer_compact_payloads_required": true,
                        "moe_fixed_compact_payloads_required": true,
                        "moe_routed_top10_compact_payloads_required": true,
                        "shared_expert_compact_payloads_required": true,
                        "state_slot_zeroed_or_caller_restored_before_encode": true,
                        "second_residual_is_next_layer_input": true,
                    },
                    "same_runtime_full_layer_encode_ready": layer.same_runtime_full_layer_encode_ready,
                })
            })
            .collect();
        let mut document = json!({
            "schema": QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA,
            "status": QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
            "source_authority": {
                "model_id": "Qwen3-Coder-Next-80B",
                "model_key": "qwen80",
                "source_repository": "Qwen/Qwen3-Coder-Next",
                "source_revision": QWEN80_SOURCE_REVISION,
                "source_config_sha256": "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8",
                "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
                "payload_schedule_authority_schema": "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1",
                "full_attention_interval": 4,
                "layer_count": 48,
                "deltanet_layers": 36,
                "gqa_layers": 12,
            },
            "layers": layer_entries,
            "claim_boundary": {"execution_schedule_authority_only": true},
        });
        seal(&mut document).unwrap();
        document
    }

    fn sealed_oracle(layer_count: usize) -> Value {
        let trace = qwen80_multi_layer_structural_kernel_trace(layer_count, false).unwrap();
        let mut document = json!({
            "schema": CHAIN_ORACLE_SCHEMA,
            "status": "PREPARED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_STRUCTURE_NOT_NUMERIC_WITHOUT_LAYER_RECEIPTS",
            "layer_count": layer_count,
            "includes_unready_gqa": false,
            "total_dispatches_physical_capture": trace.len(),
            "structural_kernel_trace_physical_capture": trace,
            "claim_boundary": {"cpu_oracle_structure_only": true},
        });
        seal(&mut document).unwrap();
        document
    }

    fn sealed_l1() -> Value {
        let mut document = json!({
            "schema": L1_ASSESSMENT_SCHEMA,
            "status": L1_EARNED_STATUS,
            "earned_complete_l1_component_only": true,
            "component_scope": {
                "fresh_total_dispatches": 46,
                "full_layer_or_token_decoder_earned": false,
            },
        });
        seal(&mut document).unwrap();
        document
    }

    fn sealed_joint() -> Value {
        let mut document = json!({
            "schema": JOINT_ASSESSMENT_SCHEMA,
            "status": JOINT_ASSESSMENT_STATUS,
            "earned_component_only": true,
            "component_scope": {
                "source_token_id": SOURCE_TOKEN_ID,
                "fresh_l0_dispatches": L0_DISPATCHES,
                "fresh_l1_slot1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "fresh_total_dispatches": L0_DISPATCHES + L1_PREFIX_DISPATCHES,
            },
        });
        seal(&mut document).unwrap();
        document
    }

    fn write_temp(dir: &Path, name: &str, value: &Value) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        fs::canonicalize(&path).unwrap()
    }

    #[test]
    fn preflight_earns_for_layer_count_3() {
        let dir = tempfile::tempdir().unwrap();
        let schedule = write_temp(dir.path(), "schedule.json", &sealed_schedule());
        let oracle = write_temp(dir.path(), "oracle.json", &sealed_oracle(3));
        let l1 = write_temp(dir.path(), "l1.json", &sealed_l1());
        let joint = write_temp(dir.path(), "joint.json", &sealed_joint());
        let host = write_temp(dir.path(), "host.bin", &json!({"binary": true}));
        let out_abs = fs::canonicalize(dir.path()).unwrap().join("out.json");
        let args = Args {
            layer_count: 3,
            execution_schedule_authority: schedule,
            chain_cpu_oracle: oracle,
            l1_full_layer_assessment: l1,
            joint_assessment: joint,
            host_binary: host,
            out: out_abs.clone(),
            workers: 1,
        };
        let document = build_preflight(&args).unwrap();
        assert_eq!(document["status"], HOST_PREFLIGHT_STATUS);
        assert_eq!(document["layer_count"], 3);
        assert_eq!(document["execution_policy"]["total_dispatches"], 69);
        assert_eq!(document["execution_policy"]["fence_count"], 1);
        assert_eq!(
            document["structural_kernel_trace"]["kernel_names"]
                .as_array()
                .unwrap()
                .len(),
            69
        );
        assert_eq!(document["claim_boundary"]["fixture_or_synthetic"], false);
        assert_eq!(document["claim_boundary"]["test_only_fake_child"], false);
        assert_eq!(document["metal_path"]["capture_body_wired"], true);
        assert_eq!(document["metal_path"]["mode_metal_available"], true);
        assert_eq!(
            document["metal_path"]["frozen_kernel_order_layer_count_3"]
                .as_array()
                .unwrap()
                .len(),
            69
        );
        assert_eq!(
            document["joint_assessment"]["schema"],
            JOINT_ASSESSMENT_SCHEMA
        );
        assert_eq!(
            document["joint_assessment"]["metal_path_provenance_only"],
            true
        );
        write_new(&out_abs, &document).unwrap();
        assert!(out_abs.exists());
    }

    #[test]
    fn preflight_refuses_joint_document_on_l1_full_layer_assessment_flag() {
        let dir = tempfile::tempdir().unwrap();
        let schedule = write_temp(dir.path(), "schedule.json", &sealed_schedule());
        let oracle = write_temp(dir.path(), "oracle.json", &sealed_oracle(3));
        // Wrong document on the L1 flag: joint post-capture assessment.
        let joint_as_l1 = write_temp(dir.path(), "wrong-l1.json", &sealed_joint());
        let joint = write_temp(dir.path(), "joint.json", &sealed_joint());
        let host = write_temp(dir.path(), "host.bin", &json!({}));
        let args = Args {
            layer_count: 3,
            execution_schedule_authority: schedule,
            chain_cpu_oracle: oracle,
            l1_full_layer_assessment: joint_as_l1,
            joint_assessment: joint,
            host_binary: host,
            out: fs::canonicalize(dir.path()).unwrap().join("out.json"),
            workers: 1,
        };
        let err = build_preflight(&args).unwrap_err();
        assert!(
            err.contains("L1 assessment schema observed=")
                && err.contains(JOINT_ASSESSMENT_SCHEMA)
                && err.contains(&format!("expected={L1_ASSESSMENT_SCHEMA}")),
            "{err}"
        );
    }

    #[test]
    fn preflight_refuses_l1_completion_on_joint_assessment_flag() {
        let dir = tempfile::tempdir().unwrap();
        let schedule = write_temp(dir.path(), "schedule.json", &sealed_schedule());
        let oracle = write_temp(dir.path(), "oracle.json", &sealed_oracle(3));
        let l1 = write_temp(dir.path(), "l1.json", &sealed_l1());
        // Wrong document on the joint flag: L1 full-layer completion assessment.
        let l1_as_joint = write_temp(dir.path(), "wrong-joint.json", &sealed_l1());
        let host = write_temp(dir.path(), "host.bin", &json!({}));
        let args = Args {
            layer_count: 3,
            execution_schedule_authority: schedule,
            chain_cpu_oracle: oracle,
            l1_full_layer_assessment: l1,
            joint_assessment: l1_as_joint,
            host_binary: host,
            out: fs::canonicalize(dir.path()).unwrap().join("out.json"),
            workers: 1,
        };
        let err = build_preflight(&args).unwrap_err();
        assert!(
            err.contains("joint assessment schema observed=")
                && err.contains(L1_ASSESSMENT_SCHEMA)
                && err.contains(&format!("expected={JOINT_ASSESSMENT_SCHEMA}")),
            "{err}"
        );
    }

    #[test]
    fn metal_path_joint_assessment_load_refuses_l1_completion_schema() {
        // Metal authority load of outer.joint_assessment accepts only the joint schema.
        let dir = tempfile::tempdir().unwrap();
        let wrong = write_temp(dir.path(), "l1-as-joint.json", &sealed_l1());
        let err = read_sealed_document(
            &wrong,
            "joint_assessment",
            JOINT_ASSESSMENT_SCHEMA,
            JOINT_ASSESSMENT_STATUS,
        )
        .unwrap_err();
        assert!(
            err.contains("joint_assessment.schema observed=")
                && err.contains(L1_ASSESSMENT_SCHEMA)
                && err.contains(&format!("expected={JOINT_ASSESSMENT_SCHEMA}")),
            "{err}"
        );
    }

    #[test]
    fn metal_path_l1_assessment_load_refuses_joint_schema() {
        let dir = tempfile::tempdir().unwrap();
        let wrong = write_temp(dir.path(), "joint-as-l1.json", &sealed_joint());
        let err = read_sealed_document(
            &wrong,
            "l1_full_layer_assessment",
            L1_ASSESSMENT_SCHEMA,
            L1_EARNED_STATUS,
        )
        .unwrap_err();
        assert!(
            err.contains("l1_full_layer_assessment.schema observed=")
                && err.contains(JOINT_ASSESSMENT_SCHEMA)
                && err.contains(&format!("expected={L1_ASSESSMENT_SCHEMA}")),
            "{err}"
        );
    }

    #[test]
    fn metal_path_accepts_joint_assessment_schema() {
        let dir = tempfile::tempdir().unwrap();
        let joint = write_temp(dir.path(), "joint.json", &sealed_joint());
        let doc = read_sealed_document(
            &joint,
            "joint_assessment",
            JOINT_ASSESSMENT_SCHEMA,
            JOINT_ASSESSMENT_STATUS,
        )
        .unwrap();
        assert_eq!(doc.seal_sha256.len(), 64);
        assert_eq!(
            doc.value["schema"].as_str().unwrap(),
            JOINT_ASSESSMENT_SCHEMA
        );
    }

    #[test]
    fn preflight_refuses_oracle_that_still_flags_unready_gqa_with_values() {
        let dir = tempfile::tempdir().unwrap();
        let schedule = write_temp(dir.path(), "schedule.json", &sealed_schedule());
        // Adversarial: schedule is encode-ready but the chain oracle still
        // claims includes_unready_gqa=true — host must refuse with values.
        let mut oracle_doc = json!({
            "schema": CHAIN_ORACLE_SCHEMA,
            "status": "PREPARED",
            "layer_count": 4,
            "includes_unready_gqa": true,
            "total_dispatches_physical_capture": null,
            "claim_boundary": {},
        });
        seal(&mut oracle_doc).unwrap();
        let oracle = write_temp(dir.path(), "oracle.json", &oracle_doc);
        let l1 = write_temp(dir.path(), "l1.json", &sealed_l1());
        let joint = write_temp(dir.path(), "joint.json", &sealed_joint());
        let host = write_temp(dir.path(), "host.bin", &json!({}));
        let args = Args {
            layer_count: 4,
            execution_schedule_authority: schedule,
            chain_cpu_oracle: oracle,
            l1_full_layer_assessment: l1,
            joint_assessment: joint,
            host_binary: host,
            out: fs::canonicalize(dir.path()).unwrap().join("out.json"),
            workers: 1,
        };
        let err = build_preflight(&args).unwrap_err();
        assert!(
            err.contains("includes_unready_gqa=true")
                && err.contains("observed true")
                && err.contains("expected false"),
            "{err}"
        );
    }

    #[test]
    fn expected_kernels_l0_l3_is_92_with_gqa_tail() {
        let kernels = expected_multi_layer_kernels(4).unwrap();
        assert_eq!(kernels.len(), 92);
        assert_eq!(kernels[69], "qwen_next_direct_packed_input_rmsnorm");
        assert_eq!(kernels[73], "qwen80_attention_qk_norm_rope_cache");
        assert_eq!(kernels[74], "mha_decode_f32");
        assert_eq!(kernels[75], "qwen80_attention_apply_sigmoid_gate");
        assert_eq!(
            kernels[91],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
    }

    #[test]
    fn expected_kernels_l0_l1_match_proven_46() {
        let kernels = expected_multi_layer_kernels(2).unwrap();
        assert_eq!(kernels.len(), 46);
        assert_eq!(kernels[0], "qwen_next_direct_packed_input_rmsnorm");
        assert_eq!(
            kernels[45],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
    }

    #[test]
    fn frozen_l0_l2_kernel_order_is_exactly_69() {
        assert_eq!(MULTI_LAYER_L0_L2_KERNELS.len(), 69);
        assert_eq!(
            MULTI_LAYER_L0_L2_KERNELS[0],
            "qwen_next_direct_packed_input_rmsnorm"
        );
        assert_eq!(
            MULTI_LAYER_L0_L2_KERNELS[22],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
        assert_eq!(
            MULTI_LAYER_L0_L2_KERNELS[23],
            "qwen_next_direct_packed_input_rmsnorm"
        );
        assert_eq!(
            MULTI_LAYER_L0_L2_KERNELS[46],
            "qwen_next_direct_packed_input_rmsnorm"
        );
        assert_eq!(
            MULTI_LAYER_L0_L2_KERNELS[68],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
        let from_schedule = expected_multi_layer_kernels(3).unwrap();
        assert_eq!(from_schedule.as_slice(), MULTI_LAYER_L0_L2_KERNELS.as_slice());
    }

    #[test]
    fn deltanet_only_layers_have_exclusive_slots() {
        use hawking_core::model::qwen80_48_layer_execution_schedule::Qwen80ExecutionMixerKind;
        for layer in 0..3 {
            let s = qwen80_layer_execution_schedule(layer).unwrap();
            assert!(matches!(s.mixer, Qwen80ExecutionMixerKind::DeltaNet));
            assert_eq!(s.state_slot.slot, layer);
            assert!(s.state_slot.exclusive_caller_owned_slot);
        }
    }

    #[test]
    fn metal_mode_refuses_missing_authority_chain_before_runtime_work() {
        let err = parse_invocation(
            [
                "--mode".into(),
                "metal".into(),
                "--workers".into(),
                "1".into(),
            ]
            .into_iter(),
        )
        .unwrap_err();
        assert!(
            err.contains("--layer-count") || err.contains("--outer-preflight"),
            "{err}"
        );
    }

    #[test]
    fn metal_mode_accepts_layer_count_3_and_4_and_refuses_others_with_values() {
        let ok3 = parse_metal_args(
            [
                "--layer-count",
                "3",
                "--outer-preflight",
                "/tmp/o.json",
                "--lease-receipt",
                "/tmp/l.json",
                "--outer-launch-authority",
                "/tmp/a.json",
                "--outer-capture-dir",
                "/tmp/outer",
                "--capture-dir",
                "/tmp/outer/inner",
                "--workers",
                "1",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(ok3.layer_count, 3);
        let ok4 = parse_metal_args(
            [
                "--layer-count",
                "4",
                "--outer-preflight",
                "/tmp/o.json",
                "--lease-receipt",
                "/tmp/l.json",
                "--outer-launch-authority",
                "/tmp/a.json",
                "--outer-capture-dir",
                "/tmp/outer",
                "--capture-dir",
                "/tmp/outer/inner",
                "--workers",
                "1",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(ok4.layer_count, 4);
        let err = parse_metal_args(
            [
                "--layer-count",
                "5",
                "--outer-preflight",
                "/tmp/o.json",
                "--lease-receipt",
                "/tmp/l.json",
                "--outer-launch-authority",
                "/tmp/a.json",
                "--outer-capture-dir",
                "/tmp/outer",
                "--capture-dir",
                "/tmp/outer/inner",
                "--workers",
                "1",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap_err();
        assert!(err.contains("observed=5"), "{err}");
        assert!(err.contains("expected=3") && err.contains("or 4"), "{err}");
    }

    #[test]
    fn invocation_requires_explicit_mode() {
        let err = parse_invocation(std::iter::empty()).unwrap_err();
        assert!(err.contains("explicit --mode"), "{err}");
    }

    #[test]
    fn producer_outer_and_release_publish_required_consumer_names() {
        let lease = "a".repeat(64);
        let inner = "b".repeat(64);
        let outer = "c".repeat(64);
        let terminal = outer_terminal_producer_fields(&lease, &inner, 0);
        assert_eq!(terminal["fixture_or_synthetic"], false);
        assert_eq!(terminal["test_only_fake_child"], false);
        assert_eq!(terminal["self_asserted"], false);
        assert_eq!(terminal["lease_id"], lease);
        assert_eq!(terminal["claim_boundary"]["test_only_fake_child"], false);
        assert_eq!(
            terminal["inner_capture"]["document_sha256"],
            terminal["inner_capture"]["document_seal_sha256"]
        );
        let release = release_producer_fields(&lease, &outer, true);
        assert_eq!(release["actual_release_performed"], true);
        assert_eq!(release["released_after_outer_terminal"], true);
        assert_eq!(release["release_after_outer_terminal"], true);
        assert_eq!(release["lease_released"], true);
        assert_eq!(release["automatic_retry_prohibited"], true);
        assert_eq!(release["fresh_lease_required_for_any_future_gpu_work"], true);
        assert_eq!(release["watcher_restart_or_transition_authorized"], false);
        // Producer convention: document_sha256 == seal on receipt pointers.
        assert_eq!(
            release["outer_terminal"]["document_sha256"],
            release["outer_terminal"]["document_seal_sha256"]
        );
    }

    #[test]
    fn producer_pointer_carries_seal_in_document_sha256() {
        let seal = "d".repeat(64);
        let raw = "e".repeat(64);
        let ptr = producer_pointer(Path::new("/tmp/doc.json"), &seal, 12, &raw);
        assert_eq!(ptr["document_sha256"], seal);
        assert_eq!(ptr["document_seal_sha256"], seal);
        assert_eq!(ptr["sha256"], raw);
        assert_eq!(ptr["present"], true);
    }

    #[test]
    fn metal_mode_does_not_construct_metal_in_parser() {
        // Parsing metal args must not touch MetalContext / device.
        let args = parse_metal_args(
            [
                "--layer-count",
                "3",
                "--outer-preflight",
                "/tmp/o.json",
                "--lease-receipt",
                "/tmp/l.json",
                "--outer-launch-authority",
                "/tmp/a.json",
                "--outer-capture-dir",
                "/tmp/outer",
                "--capture-dir",
                "/tmp/outer/inner",
                "--workers",
                "1",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        assert_eq!(args.layer_count, 3);
        assert_eq!(args.workers, 1);
    }
}
