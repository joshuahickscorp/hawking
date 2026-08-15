//! CPU/build-only strict host and static preflight for one future Qwen80
//! source-token Layer-1 complete-MoE component.
//!
//! The prior L0(23)+L1-prefix(9) capture is evidence only.  Its retained
//! Metal buffers cannot cross a process boundary, so a later explicitly
//! leased child must re-encode the source-token L0 true-MoE graph, append the
//! L1 slot-one DeltaNet prefix, then append exactly the fourteen L1
//! postnorm/router/top-10/routed/shared/second-residual kernels before one
//! fence.  This program binds that future 46-dispatch graph to the original
//! sealed L1 CPU route authority without creating a Metal context, opening a
//! catalog payload, issuing a lease, or dispatching work in its default CLI.
//!
//! The recovery/canonicalization wrapper for the historical router scan is
//! provenance only.  The original valid inner authority is the sole execution
//! input, and the output here deliberately retains only shallow evidence plus
//! the exact six fixed and thirty routed compact payload identities.

#[path = "ascension_qwen80_source_token_l1_moe_completion_preflight.rs"]
mod completion_preflight;

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
use hawking_core::metal::TokenCommandBuffer;
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime, Qwen80CompleteRuntimeOptions,
    Qwen80RouteSelection, Qwen80SameRuntimeFreshL0Parity, Qwen80SameRuntimeL0L1FullLayerParity,
    Qwen80SameRuntimeL1RoutedWaveParity, Qwen80SameRuntimeL1TrueMoeSuffixParity,
    Qwen80SameRuntimeLayer1DeltaNetPrefixParity, Qwen80SourceInputFirstResidualParity,
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

const HOST_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_host_preflight.v1";
const HOST_PREFLIGHT_STATUS: &str =
    "COMPILED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const COMPLETION_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_moe_completion_preflight.v1";
const COMPLETION_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_QWEN80_SOURCE_TOKEN_L1_MOE_COMPLETION_COMPONENT_NOT_LEASED_OR_EXECUTED";
const JOINT_ASSESSMENT_SCHEMA: &str =
    "hawking.ascension.qwen80_l0_l1_joint_post_capture_assessment.v1";
const JOINT_ASSESSMENT_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L0_L1_COMPONENT_NOT_FULL_LAYER_TOKEN_DECODER";
const L0_SOURCE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1";
const L0_SOURCE_OUTER_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED";
const L1_ROUTE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l1_all_ten_route_authority.v1";
const L1_ROUTE_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L1_ALL_TEN_ROUTE_AUTHORITY_READY_FOR_SAME_RUNTIME_MOE_SUFFIX";
const FUTURE_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1";
const FUTURE_INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_COMPONENT_ONLY";
const FUTURE_INNER_REFUSED_STATUS: &str =
    "REFUSED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_PHASE_ACCURATE_TERMINAL_FAILURE";
const FUTURE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_capture.v1";
const FUTURE_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY";
const FUTURE_RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_quiet_metal_lease_release.v1";
const FUTURE_RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";
const OUTER_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_preflight.v1";
const OUTER_PREFLIGHT_STATUS: &str =
    "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_OUTER_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const METAL_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_quiet_metal_lease.v1";
const METAL_LEASE_STATUS: &str =
    "GRANTED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_LAYER_SAME_RUNTIME_COMPONENT_QUIET_METAL_LEASE";
const METAL_OUTER_LAUNCH_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_outer_launch_authority.v1";
const METAL_OUTER_LAUNCH_STATUS: &str =
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_L0_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_OUTER_REAPED_ONE_SHOT_METAL_CHILD";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const SOURCE_TOKEN_ID: u64 = 1;
const L0_DISPATCHES: u64 = 23;
const L1_PREFIX_DISPATCHES: u64 = 9;
const L1_MOE_SUFFIX_DISPATCHES: u64 = 14;
const TOTAL_DISPATCHES: u64 = L0_DISPATCHES + L1_PREFIX_DISPATCHES + L1_MOE_SUFFIX_DISPATCHES;
// This is the source/device route-weight tolerance enforced by the strict
// Metal route-guard kernel.  Route IDs and the CPU oracle remain exact; only
// the device's f32 normalisation readback is allowed its declared bounded
// numeric divergence.
const L1_ROUTE_WEIGHT_TOLERANCE: f32 = 2.0e-5;
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
const L1_PREFIX_KERNELS: [&str; 9] = [
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
const L1_MOE_SUFFIX_KERNELS: [&str; 14] = [
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

#[derive(Debug)]
struct Args {
    joint_assessment: PathBuf,
    l1_route_authority: PathBuf,
    completion_preflight: PathBuf,
    l0_source_outer_preflight: PathBuf,
    host_binary: PathBuf,
    out: PathBuf,
    workers: usize,
}

#[derive(Debug)]
struct MetalArgs {
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

/// File-only authority validated before the sole strict-Metal entrypoint may
/// construct a runtime.  The earlier L0/L1 assessment remains provenance;
/// this value intentionally contains no transferable Metal buffer or state.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct FullL1CaptureAuthority {
    outer_preflight: SealedDocument,
    host_preflight: SealedDocument,
    joint_assessment: SealedDocument,
    completion_preflight: SealedDocument,
    l0_source_outer_preflight: SealedDocument,
    l1_route_authority: completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    manifest: SealedDocument,
    admission_receipt: SealedDocument,
    lease: SealedDocument,
    lease_id: String,
    outer_launch: SealedDocument,
    host_binary: FileEvidence,
    workers: usize,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct MetalLaunchGate {
    outer_preflight: SealedDocument,
    lease: SealedDocument,
    lease_id: String,
    outer_launch: SealedDocument,
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
    "usage: ascension_qwen80_source_token_l0_l1_full_layer_same_runtime_device \\\n+--mode preflight --joint-assessment ABSOLUTE_SEALED_ASSESSMENT \\\n+--l1-route-authority ABSOLUTE_ORIGINAL_SEALED_INNER_AUTHORITY \\\n+--completion-preflight ABSOLUTE_SEALED_L1_COMPLETION_PREFLIGHT \\\n+--l0-source-outer-preflight ABSOLUTE_SEALED_L0_SOURCE_OUTER_PREFLIGHT \\\n+--host-binary ABSOLUTE_CURRENT_HOST_BINARY --out ABSOLUTE_NEW_JSON --workers 1..4\n\
or: ascension_qwen80_source_token_l0_l1_full_layer_same_runtime_device \\\n+--mode metal --outer-preflight ABSOLUTE_SEALED_CPU_OUTER_PREFLIGHT \\\n+--lease-receipt ABSOLUTE_SEALED_NEW_FULL_L1_LEASE \\\n+--outer-launch-authority ABSOLUTE_SEALED_NEW_FULL_L1_OUTER_LAUNCH \\\n+--outer-capture-dir ABSOLUTE_EXISTING_OUTER_DIRECTORY \\\n+--capture-dir ABSOLUTE_NEW_DIRECT_CHILD_DIRECTORY --workers 1..4"
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
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn absolute_new_directory(path: PathBuf, outer_capture_dir: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err("--capture-dir must be absolute".into());
    }
    if path.exists() {
        return Err("--capture-dir must be a create-new path".into());
    }
    let parent = path
        .parent()
        .ok_or("--capture-dir must have an outer-capture-dir parent")?;
    if canonical_directory(parent, "--capture-dir parent")? != outer_capture_dir {
        return Err("--capture-dir must be a direct child of --outer-capture-dir".into());
    }
    Ok(path)
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut joint_assessment = None;
    let mut l1_route_authority = None;
    let mut completion_preflight = None;
    let mut l0_source_outer_preflight = None;
    let mut host_binary = None;
    let mut out = None;
    let mut workers = None;
    let mut arguments = arguments.into_iter();
    while let Some(flag) = arguments.next() {
        let slot = match flag.as_str() {
            "--joint-assessment" => &mut joint_assessment,
            "--l1-route-authority" => &mut l1_route_authority,
            "--completion-preflight" => &mut completion_preflight,
            "--l0-source-outer-preflight" => &mut l0_source_outer_preflight,
            "--host-binary" => &mut host_binary,
            "--out" => &mut out,
            "--workers" => {
                let raw = arguments
                    .next()
                    .ok_or_else(|| format!("--workers requires a value; {}", usage()))?;
                if workers
                    .replace(
                        raw.parse::<usize>()
                            .map_err(|_| "--workers must be an integer")?,
                    )
                    .is_some()
                {
                    return Err(format!("--workers may not be repeated; {}", usage()));
                }
                continue;
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        };
        let value = arguments
            .next()
            .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(format!("{flag} may not be repeated; {}", usage()));
        }
    }
    let require_absolute = |path: Option<PathBuf>, flag: &str| -> Result<PathBuf, String> {
        let path = path.ok_or_else(|| format!("missing {flag}; {}", usage()))?;
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    let workers = workers.ok_or_else(|| format!("--workers 1..4 is required; {}", usage()))?;
    if !(1..=4).contains(&workers) {
        return Err("--workers must be in 1..=4".into());
    }
    let joint_assessment = require_absolute(joint_assessment, "--joint-assessment")?;
    let l1_route_authority = require_absolute(l1_route_authority, "--l1-route-authority")?;
    let completion_preflight = require_absolute(completion_preflight, "--completion-preflight")?;
    let l0_source_outer_preflight =
        require_absolute(l0_source_outer_preflight, "--l0-source-outer-preflight")?;
    let host_binary = require_absolute(host_binary, "--host-binary")?;
    let out = require_absolute(out, "--out")?;
    if out.exists() || !out.parent().is_some_and(Path::is_dir) {
        return Err("--out must be a new file beneath an existing parent".into());
    }
    Ok(Args {
        joint_assessment,
        l1_route_authority,
        completion_preflight,
        l0_source_outer_preflight,
        host_binary,
        out,
        workers,
    })
}

fn parse_metal_args(arguments: Vec<String>) -> Result<MetalArgs, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut iterator = arguments.into_iter();
    while let Some(flag) = iterator.next() {
        let value = iterator
            .next()
            .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
        match flag.as_str() {
            "--outer-preflight"
            | "--lease-receipt"
            | "--outer-launch-authority"
            | "--outer-capture-dir"
            | "--capture-dir"
            | "--workers" => {
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} may not be repeated; {}", usage()));
                }
            }
            _ => {
                return Err(format!(
                    "metal mode refuses unsupported argument {flag:?}; {}",
                    usage()
                ))
            }
        }
    }
    let workers = values
        .remove("--workers")
        .ok_or_else(|| format!("--workers 1..4 is required; {}", usage()))?
        .parse::<usize>()
        .map_err(|_| "--workers must be an integer")?;
    if !(1..=4).contains(&workers) {
        return Err("--workers must be in 1..=4".into());
    }
    let mut required = |flag: &str| -> Result<PathBuf, String> {
        let path = PathBuf::from(
            values
                .remove(flag)
                .ok_or_else(|| format!("{flag} is required; {}", usage()))?,
        );
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    let outer_preflight = canonical_regular(&required("--outer-preflight")?, "--outer-preflight")?;
    let lease_receipt = canonical_regular(&required("--lease-receipt")?, "--lease-receipt")?;
    let outer_launch_authority = canonical_regular(
        &required("--outer-launch-authority")?,
        "--outer-launch-authority",
    )?;
    let outer_capture_dir =
        canonical_directory(&required("--outer-capture-dir")?, "--outer-capture-dir")?;
    let capture_dir = absolute_new_directory(required("--capture-dir")?, &outer_capture_dir)?;
    if !values.is_empty() {
        return Err(format!(
            "metal mode received unconsumed arguments: {:?}",
            values.keys()
        ));
    }
    Ok(MetalArgs {
        outer_preflight,
        lease_receipt,
        outer_launch_authority,
        outer_capture_dir,
        capture_dir,
        workers,
    })
}

fn parse_invocation<I>(arguments: I) -> Result<Invocation, String>
where
    I: IntoIterator<Item = String>,
{
    let arguments = arguments.into_iter().collect::<Vec<_>>();
    let mut mode = None;
    let mut remaining = Vec::new();
    let mut iterator = arguments.into_iter();
    while let Some(flag) = iterator.next() {
        if flag == "--help" || flag == "-h" {
            return Err(usage().to_owned());
        }
        let value = iterator
            .next()
            .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
        if flag == "--mode" {
            if mode.replace(value).is_some() {
                return Err(format!("--mode may not be repeated; {}", usage()));
            }
        } else {
            remaining.push(flag);
            remaining.push(value);
        }
    }
    match mode.as_deref() {
        Some("preflight") => Ok(Invocation::Preflight(parse_args(remaining)?)),
        Some("metal") => Ok(Invocation::Metal(parse_metal_args(remaining)?)),
        Some(_) => Err(format!("--mode must be preflight or metal; {}", usage())),
        None => Err(format!("an explicit --mode is required; {}", usage())),
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
            let mut keys = values.keys().collect::<Vec<_>>();
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

fn document_sha256(value: &Value) -> Result<String, String> {
    Ok(sha256_hex(&canonical_json(value)?))
}

fn seal(document: &mut Value) -> Result<String, String> {
    let root = document
        .as_object()
        .ok_or("host preflight must be an object")?;
    if root.contains_key("seal_sha256") {
        return Err("host preflight may not already carry a seal".into());
    }
    let seal = document_sha256(document)?;
    document
        .as_object_mut()
        .expect("validated host preflight object")
        .insert("seal_sha256".into(), Value::String(seal.clone()));
    Ok(seal)
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!("{label} must be a regular non-symlink file"));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn file_evidence(path: &Path, label: &str) -> Result<FileEvidence, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let path = canonical_regular(path, label)?;
    let bytes = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    if bytes.len() as u64 > MAX_JSON_BYTES && label != "host binary" {
        return Err(format!("{label} exceeds bounded JSON size"));
    }
    Ok(FileEvidence {
        path,
        bytes: bytes.len() as u64,
        sha256: sha256_hex(&bytes),
    })
}

fn read_sealed_document(
    path: &Path,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<SealedDocument, String> {
    let verified = completion_preflight::read_verified_sealed_document(path, label)?;
    let root = verified
        .value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))?;
    if root.get("schema").and_then(Value::as_str) != Some(expected_schema)
        || root.get("status").and_then(Value::as_str) != Some(expected_status)
    {
        return Err(format!("{label} schema/status drifted"));
    }
    let bytes = fs::metadata(&verified.path)
        .map_err(|error| format!("cannot stat {label}: {error}"))?
        .len();
    Ok(SealedDocument {
        file: FileEvidence {
            path: verified.path,
            bytes,
            sha256: verified.raw_sha256,
        },
        value: verified.value,
        document_sha256: verified.document_sha256,
        seal_sha256: verified.document_seal_sha256,
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn object_field<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn bool_field(
    value: &Map<String, Value>,
    field: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if value.get(field).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label}.{field} must be {expected}"));
    }
    Ok(())
}

fn evidence_json(file: &FileEvidence) -> Value {
    json!({
        "path": file.path,
        "present": true,
        "bytes": file.bytes,
        "sha256": file.sha256,
    })
}

fn sealed_binding_json(document: &SealedDocument) -> Value {
    let mut binding = evidence_json(&document.file);
    let object = binding.as_object_mut().expect("evidence is an object");
    object.insert(
        "document_sha256".into(),
        Value::String(document.document_sha256.clone()),
    );
    object.insert(
        "document_seal_sha256".into(),
        Value::String(document.seal_sha256.clone()),
    );
    binding
}

fn require_binding_matches(
    value: &Map<String, Value>,
    expected: &SealedDocument,
    label: &str,
) -> Result<(), String> {
    if value.get("document_sha256").and_then(Value::as_str) != Some(&expected.document_sha256)
        || value.get("document_seal_sha256").and_then(Value::as_str) != Some(&expected.seal_sha256)
    {
        return Err(format!("{label} document identity drifted"));
    }
    Ok(())
}

fn require_full_binding_matches(
    value: &Map<String, Value>,
    expected: &SealedDocument,
    label: &str,
) -> Result<(), String> {
    if value.get("path").and_then(Value::as_str)
        != Some(expected.file.path.to_string_lossy().as_ref())
        || value.get("present").and_then(Value::as_bool) != Some(true)
        || value.get("bytes").and_then(Value::as_u64) != Some(expected.file.bytes)
        || value.get("sha256").and_then(Value::as_str) != Some(expected.file.sha256.as_str())
    {
        return Err(format!("{label} raw evidence drifted"));
    }
    require_binding_matches(value, expected, label)
}

fn require_file_evidence_matches(
    value: &Map<String, Value>,
    expected: &FileEvidence,
    label: &str,
) -> Result<(), String> {
    if value.get("path").and_then(Value::as_str) != Some(expected.path.to_string_lossy().as_ref())
        || value.get("present").and_then(Value::as_bool) != Some(true)
        || value.get("bytes").and_then(Value::as_u64) != Some(expected.bytes)
        || value.get("sha256").and_then(Value::as_str) != Some(expected.sha256.as_str())
    {
        return Err(format!("{label} evidence drifted"));
    }
    Ok(())
}

fn string_field<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.{field} must be a string"))
}

fn u64_field(value: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be an unsigned integer"))
}

fn current_host_binary_evidence() -> Result<FileEvidence, String> {
    let path = env::current_exe()
        .map_err(|error| format!("cannot resolve current full-L1 host executable: {error}"))?;
    file_evidence(&path, "current full-L1 host executable")
}

#[cfg(target_os = "macos")]
fn validate_metal_gate(args: &MetalArgs) -> Result<MetalLaunchGate, String> {
    // This validation is deliberately file-only.  It must complete before a
    // future implementation can call any runtime or Metal constructor.
    let outer = read_sealed_document(
        &args.outer_preflight,
        "full-L1 outer CPU preflight",
        OUTER_PREFLIGHT_SCHEMA,
        OUTER_PREFLIGHT_STATUS,
    )?;
    let own = current_host_binary_evidence()?;
    let outer_root = object(&outer.value, "full-L1 outer CPU preflight")?;
    require_file_evidence_matches(
        object_field(outer_root, "host_binary", "full-L1 outer CPU preflight")?,
        &own,
        "full-L1 outer CPU preflight.host_binary",
    )?;
    // Validate the emitted host-preflight/file chain before the lease can
    // reach any source/artifact or runtime preparation.
    validate_outer_host_preflight_execution_bindings(outer_root, &own)?;
    let outer_scope = object_field(
        outer_root,
        "exact_component_scope",
        "full-L1 outer CPU preflight",
    )?;
    for (field, expected) in [
        ("source_token_id", SOURCE_TOKEN_ID),
        ("l0_reencode_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ] {
        if u64_field(outer_scope, field, "full-L1 outer CPU preflight scope")? != expected {
            return Err(format!("full-L1 outer CPU preflight {field} drifted"));
        }
    }
    for (field, expected) in [
        ("one_fence_required", true),
        ("non_timed_exact_trace_required", true),
    ] {
        bool_field(
            outer_scope,
            field,
            expected,
            "full-L1 outer CPU preflight scope",
        )?;
    }
    let outer_lifecycle = object_field(outer_root, "lifecycle", "full-L1 outer CPU preflight")?;
    for (field, expected) in [
        ("replay_guard_required", true),
        ("one_child_process_required", true),
        ("outer_reaped_terminal_required", true),
        ("automatic_retry_authorized", false),
        (
            "lease_or_device_execution_authorized_by_this_cpu_preflight",
            false,
        ),
        ("real_host_metal_cli_available", true),
    ] {
        bool_field(
            outer_lifecycle,
            field,
            expected,
            "full-L1 outer CPU preflight lifecycle",
        )?;
    }
    let outer_entrypoint = object_field(
        outer_root,
        "future_metal_entrypoint",
        "full-L1 outer CPU preflight",
    )?;
    bool_field(
        outer_entrypoint,
        "capture_body_wired",
        true,
        "full-L1 outer CPU preflight future Metal entrypoint",
    )?;

    let lease = read_sealed_document(
        &args.lease_receipt,
        "full-L1 quiet Metal lease",
        METAL_LEASE_SCHEMA,
        METAL_LEASE_STATUS,
    )?;
    let lease_root = object(&lease.value, "full-L1 quiet Metal lease")?;
    let lease_id = string_field(lease_root, "lease_id", "full-L1 quiet Metal lease")?.to_owned();
    if !is_sha256(&lease_id) {
        return Err("full-L1 quiet Metal lease.lease_id must be a lowercase SHA-256".into());
    }
    require_full_binding_matches(
        object_field(lease_root, "outer_preflight", "full-L1 quiet Metal lease")?,
        &outer,
        "full-L1 quiet Metal lease.outer_preflight",
    )?;
    require_file_evidence_matches(
        object_field(lease_root, "host_binary", "full-L1 quiet Metal lease")?,
        &own,
        "full-L1 quiet Metal lease.host_binary",
    )?;
    let lease_policy = object_field(lease_root, "execution_policy", "full-L1 quiet Metal lease")?;
    for (field, expected) in [
        ("metal_mode_only", true),
        ("non_timed_exact_46_dispatches_required", true),
        ("one_fence_required", true),
        ("component_only", true),
        ("l1_moe_suffix_allowed", true),
        ("automatic_retry_allowed", false),
    ] {
        bool_field(
            lease_policy,
            field,
            expected,
            "full-L1 quiet Metal lease policy",
        )?;
    }

    let launch = read_sealed_document(
        &args.outer_launch_authority,
        "full-L1 outer launch authority",
        METAL_OUTER_LAUNCH_SCHEMA,
        METAL_OUTER_LAUNCH_STATUS,
    )?;
    let launch_root = object(&launch.value, "full-L1 outer launch authority")?;
    require_full_binding_matches(
        object_field(
            launch_root,
            "outer_preflight",
            "full-L1 outer launch authority",
        )?,
        &outer,
        "full-L1 outer launch authority.outer_preflight",
    )?;
    require_full_binding_matches(
        object_field(
            launch_root,
            "lease_receipt",
            "full-L1 outer launch authority",
        )?,
        &lease,
        "full-L1 outer launch authority.lease_receipt",
    )?;
    require_file_evidence_matches(
        object_field(launch_root, "host_binary", "full-L1 outer launch authority")?,
        &own,
        "full-L1 outer launch authority.host_binary",
    )?;
    if string_field(launch_root, "lease_id", "full-L1 outer launch authority")? != lease_id {
        return Err("full-L1 outer launch authority lease ID drifted".into());
    }
    if string_field(
        launch_root,
        "planned_outer_capture_dir",
        "full-L1 outer launch authority",
    )? != args.outer_capture_dir.to_string_lossy()
        || string_field(
            launch_root,
            "planned_inner_capture_dir",
            "full-L1 outer launch authority",
        )? != args.capture_dir.to_string_lossy()
        || u64_field(launch_root, "workers", "full-L1 outer launch authority")?
            != args.workers as u64
    {
        return Err("full-L1 outer launch authority capture path or worker drifted".into());
    }
    let launch_policy = object_field(
        launch_root,
        "execution_policy",
        "full-L1 outer launch authority",
    )?;
    for (field, expected) in [
        ("metal_mode_only", true),
        ("non_timed_exact_46_dispatches_required", true),
        ("one_fence_required", true),
        ("component_only", true),
        ("l1_moe_suffix_allowed", true),
        ("automatic_retry_allowed", false),
    ] {
        bool_field(
            launch_policy,
            field,
            expected,
            "full-L1 outer launch authority policy",
        )?;
    }
    let launch_lifecycle =
        object_field(launch_root, "lifecycle", "full-L1 outer launch authority")?;
    for (field, expected) in [
        ("replay_guard_required", true),
        ("one_child_process_required", true),
        ("outer_reaped_terminal_required", true),
        ("terminal_receipt_written_last_required", true),
    ] {
        bool_field(
            launch_lifecycle,
            field,
            expected,
            "full-L1 outer launch authority lifecycle",
        )?;
    }
    Ok(MetalLaunchGate {
        outer_preflight: outer,
        lease,
        lease_id,
        outer_launch: launch,
        host_binary: own,
        workers: args.workers,
    })
}

#[cfg(target_os = "macos")]
fn load_bound_document(
    binding: &Map<String, Value>,
    label: &str,
    expected_schema: &str,
    expected_status: &str,
) -> Result<SealedDocument, String> {
    let path = PathBuf::from(string_field(binding, "path", label)?);
    let document = read_sealed_document(&path, label, expected_schema, expected_status)?;
    require_full_binding_matches(binding, &document, label)?;
    Ok(document)
}

/// The CPU oracle validator exposes the original route authority's raw and
/// sealed hashes, but not its byte count.  Re-open the sealed file here so
/// the emitted host preflight carries the complete raw binding consumed by
/// the strict execution gate.
fn exact_l1_route_authority_binding(
    authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
) -> Result<Value, String> {
    let document = read_sealed_document(
        &authority.route_authority_path,
        "validated original L1 route authority",
        L1_ROUTE_AUTHORITY_SCHEMA,
        L1_ROUTE_AUTHORITY_STATUS,
    )?;
    if document.file.sha256 != authority.route_authority_raw_sha256
        || document.document_sha256 != authority.route_authority_document_sha256
        || document.seal_sha256 != authority.route_authority_document_seal_sha256
    {
        return Err("validated original L1 route authority identity drifted before host preflight serialization".into());
    }
    Ok(sealed_binding_json(&document))
}

#[cfg(target_os = "macos")]
fn validate_host_preflight_execution_bindings(
    host_preflight: &SealedDocument,
    route_document: &SealedDocument,
    host_binary: &FileEvidence,
) -> Result<(), String> {
    let host_root = object(&host_preflight.value, "full-L1 host CPU preflight")?;
    require_file_evidence_matches(
        object_field(host_root, "host_binary", "full-L1 host CPU preflight")?,
        host_binary,
        "full-L1 host CPU preflight.host_binary",
    )?;
    let route = object_field(
        host_root,
        "l1_route_payload_authority",
        "full-L1 host CPU preflight",
    )?;
    require_full_binding_matches(
        object_field(
            route,
            "binding",
            "full-L1 host CPU preflight route authority",
        )?,
        route_document,
        "full-L1 host CPU preflight original L1 route authority",
    )
}

#[cfg(target_os = "macos")]
fn validate_outer_host_preflight_execution_bindings(
    outer_root: &Map<String, Value>,
    host_binary: &FileEvidence,
) -> Result<(), String> {
    let host_preflight = load_bound_document(
        object_field(outer_root, "host_preflight", "full-L1 outer CPU preflight")?,
        "full-L1 host CPU preflight",
        HOST_PREFLIGHT_SCHEMA,
        HOST_PREFLIGHT_STATUS,
    )?;
    let route_document = load_bound_document(
        object_field(
            outer_root,
            "original_l1_route_authority",
            "full-L1 outer CPU preflight",
        )?,
        "full-L1 original L1 route authority",
        L1_ROUTE_AUTHORITY_SCHEMA,
        L1_ROUTE_AUTHORITY_STATUS,
    )?;
    validate_host_preflight_execution_bindings(&host_preflight, &route_document, host_binary)
}

/// Recover the immutable source admission documents from the sealed L0 source
/// outer.  The mutable admission-current pointer is deliberately not used as
/// a transferable execution input here: exact manifest and immutable
/// admission-receipt identities are what the strict runtime loader consumes.
#[cfg(target_os = "macos")]
fn load_source_artifact_chain(
    l0_outer: &SealedDocument,
) -> Result<(SealedDocument, SealedDocument), String> {
    let l0_root = object(&l0_outer.value, "source-token L0 outer preflight")?;
    let source = object_field(l0_root, "source_binding", "source-token L0 outer preflight")?;
    let manifest = read_sealed_document(
        &PathBuf::from(string_field(
            object_field(source, "manifest", "source-token L0 outer source binding")?,
            "path",
            "source-token L0 outer source binding.manifest",
        )?),
        "source-token L0 immutable manifest",
        MANIFEST_SCHEMA,
        MANIFEST_STATUS,
    )?;
    require_file_evidence_matches(
        object_field(source, "manifest", "source-token L0 outer source binding")?,
        &manifest.file,
        "source-token L0 outer source binding.manifest",
    )?;
    if string_field(
        source,
        "manifest_seal_sha256",
        "source-token L0 outer source binding",
    )? != manifest.seal_sha256
    {
        return Err("source-token L0 outer immutable manifest seal drifted".into());
    }
    let admission_receipt = read_sealed_document(
        &PathBuf::from(string_field(
            object_field(
                source,
                "admission_receipt",
                "source-token L0 outer source binding",
            )?,
            "path",
            "source-token L0 outer source binding.admission_receipt",
        )?),
        "source-token L0 immutable admission receipt",
        ADMISSION_RECEIPT_SCHEMA,
        ADMISSION_RECEIPT_STATUS,
    )?;
    require_file_evidence_matches(
        object_field(
            source,
            "admission_receipt",
            "source-token L0 outer source binding",
        )?,
        &admission_receipt.file,
        "source-token L0 outer source binding.admission_receipt",
    )?;
    if string_field(
        source,
        "admission_receipt_seal_sha256",
        "source-token L0 outer source binding",
    )? != admission_receipt.seal_sha256
    {
        return Err("source-token L0 outer immutable admission receipt seal drifted".into());
    }
    let admission_root = object(
        &admission_receipt.value,
        "source-token L0 immutable admission receipt",
    )?;
    let admitted_manifest = object_field(
        admission_root,
        "complete_manifest",
        "source-token L0 immutable admission receipt",
    )?;
    if string_field(
        admitted_manifest,
        "document_sha256",
        "source-token L0 immutable admission receipt.complete_manifest",
    )? != manifest.file.sha256
        || string_field(
            admitted_manifest,
            "seal_sha256",
            "source-token L0 immutable admission receipt.complete_manifest",
        )? != manifest.seal_sha256
    {
        return Err("immutable admission receipt does not bind the exact manifest".into());
    }
    Ok((manifest, admission_receipt))
}

#[cfg(target_os = "macos")]
fn resolve_full_l1_capture_authority(
    gate: MetalLaunchGate,
) -> Result<FullL1CaptureAuthority, String> {
    let outer_root = object(&gate.outer_preflight.value, "full-L1 outer CPU preflight")?;
    let host_preflight = load_bound_document(
        object_field(outer_root, "host_preflight", "full-L1 outer CPU preflight")?,
        "full-L1 host CPU preflight",
        HOST_PREFLIGHT_SCHEMA,
        HOST_PREFLIGHT_STATUS,
    )?;
    let host_root = object(&host_preflight.value, "full-L1 host CPU preflight")?;
    let entrypoint = object_field(
        host_root,
        "future_metal_entrypoint",
        "full-L1 host CPU preflight",
    )?;
    for (field, expected) in [
        ("explicit_mode_required", true),
        ("default_execution_disabled", true),
        ("requires_new_full_l1_lease", true),
        ("requires_sealed_outer_launch_authority", true),
        ("requires_fresh_outer_and_inner_capture_directories", true),
        ("self_hashes_current_executable", true),
        ("capture_body_wired", true),
    ] {
        bool_field(
            entrypoint,
            field,
            expected,
            "full-L1 host future Metal entrypoint",
        )?;
    }
    let assessment = load_bound_document(
        object_field(
            outer_root,
            "joint_assessment",
            "full-L1 outer CPU preflight",
        )?,
        "full-L1 joint assessment",
        JOINT_ASSESSMENT_SCHEMA,
        JOINT_ASSESSMENT_STATUS,
    )?;
    let completion = load_bound_document(
        object_field(
            outer_root,
            "completion_preflight",
            "full-L1 outer CPU preflight",
        )?,
        "full-L1 completion preflight",
        COMPLETION_PREFLIGHT_SCHEMA,
        COMPLETION_PREFLIGHT_STATUS,
    )?;
    let l0_outer = load_bound_document(
        object_field(
            outer_root,
            "l0_source_outer_preflight",
            "full-L1 outer CPU preflight",
        )?,
        "full-L1 source-token L0 outer preflight",
        L0_SOURCE_OUTER_SCHEMA,
        L0_SOURCE_OUTER_STATUS,
    )?;
    let route_document = load_bound_document(
        object_field(
            outer_root,
            "original_l1_route_authority",
            "full-L1 outer CPU preflight",
        )?,
        "full-L1 original L1 route authority",
        L1_ROUTE_AUTHORITY_SCHEMA,
        L1_ROUTE_AUTHORITY_STATUS,
    )?;
    for (field, expected) in [
        ("joint_assessment", &assessment),
        ("completion_preflight", &completion),
        ("l0_source_outer_preflight", &l0_outer),
    ] {
        require_full_binding_matches(
            object_field(host_root, field, "full-L1 host CPU preflight")?,
            expected,
            &format!("full-L1 host CPU preflight.{field}"),
        )?;
    }
    validate_host_preflight_execution_bindings(
        &host_preflight,
        &route_document,
        &gate.host_binary,
    )?;
    let l1_route_authority = completion_preflight::validate_source_token_l1_route_authority_files(
        &assessment.file.path,
        &route_document.file.path,
    )?;
    validate_completion_preflight(&completion, &assessment, &l1_route_authority)?;
    validate_l0_source_outer(&l0_outer)?;
    let (manifest, admission_receipt) = load_source_artifact_chain(&l0_outer)?;
    Ok(FullL1CaptureAuthority {
        outer_preflight: gate.outer_preflight,
        host_preflight,
        joint_assessment: assessment,
        completion_preflight: completion,
        l0_source_outer_preflight: l0_outer,
        l1_route_authority,
        manifest,
        admission_receipt,
        lease: gate.lease,
        lease_id: gate.lease_id,
        outer_launch: gate.outer_launch,
        host_binary: gate.host_binary,
        workers: gate.workers,
    })
}

#[cfg(target_os = "macos")]
fn source_artifact_admission(
    authority: &FullL1CaptureAuthority,
) -> Result<CompleteBinaryAdmission, String> {
    let manifest = object(&authority.manifest.value, "full-L1 immutable manifest")?;
    let source_audit_seal = string_field(
        manifest,
        "source_body_audit_seal_sha256",
        "full-L1 immutable manifest",
    )?
    .to_owned();
    if !is_sha256(&source_audit_seal) {
        return Err("full-L1 immutable manifest source audit seal is malformed".into());
    }
    let revalidation_path = PathBuf::from(string_field(
        manifest,
        "source_revalidation_receipt_path",
        "full-L1 immutable manifest",
    )?);
    let expected_revalidation_seal = string_field(
        manifest,
        "source_revalidation_receipt_seal_sha256",
        "full-L1 immutable manifest",
    )?;
    if !is_sha256(expected_revalidation_seal) {
        return Err("full-L1 immutable manifest revalidation seal is malformed".into());
    }
    let revalidation = completion_preflight::read_verified_sealed_document(
        &revalidation_path,
        "full-L1 source revalidation",
    )?;
    if revalidation.document_seal_sha256 != expected_revalidation_seal {
        return Err("full-L1 source revalidation seal drifted".into());
    }
    let revalidation_root = object(&revalidation.value, "full-L1 source revalidation")?;
    if string_field(
        revalidation_root,
        "source_audit_seal_sha256",
        "full-L1 source revalidation",
    )? != source_audit_seal
    {
        return Err("full-L1 source revalidation audit seal drifted".into());
    }
    let source_revision = string_field(
        revalidation_root,
        "source_revision",
        "full-L1 source revalidation",
    )?
    .to_owned();
    if source_revision.len() != 40
        || !source_revision.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
    {
        return Err("full-L1 source revalidation revision is malformed".into());
    }
    Ok(CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest.seal_sha256.clone(),
        expected_source_audit_seal_sha256: source_audit_seal,
        expected_source_revision: source_revision,
    })
}

#[cfg(target_os = "macos")]
fn require_non_timed_tcb_trace_off() -> Result<(), String> {
    match env::var("HAWKING_TCB_TRACE") {
        Err(_) => Ok(()),
        Ok(value) if value.is_empty() || value == "0" => Ok(()),
        Ok(value) => Err(format!(
            "full-L1 capture refuses HAWKING_TCB_TRACE={value:?}; exact one-TCB non-timed capture requires it unset or 0"
        )),
    }
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
fn phase_document(phase: &MetalExecutionPhase) -> Value {
    let device_dispatch_may_have_occurred = if phase.command_fence_succeeded {
        Value::Bool(true)
    } else if phase.command_commit_may_have_been_attempted {
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

fn exact_kernel_trace() -> Vec<Value> {
    L0_KERNELS
        .iter()
        .enumerate()
        .map(|(ordinal, kernel)| json!({"ordinal": ordinal, "phase": "fresh_l0_reencode", "kernel": kernel}))
        .chain(L1_PREFIX_KERNELS.iter().enumerate().map(|(offset, kernel)| {
            json!({"ordinal": L0_KERNELS.len() + offset, "phase": "fresh_l1_deltanet_prefix", "kernel": kernel})
        }))
        .chain(L1_MOE_SUFFIX_KERNELS.iter().enumerate().map(|(offset, kernel)| {
            json!({"ordinal": L0_KERNELS.len() + L1_PREFIX_KERNELS.len() + offset, "phase": "fresh_l1_moe_suffix", "kernel": kernel})
        }))
        .collect()
}

fn shallow_descriptor(value: &Value, label: &str) -> Result<Value, String> {
    let value = object(value, label)?;
    let mut output = Map::new();
    for field in [
        "role",
        "tensor_name",
        "artifact_sha256",
        "direct_packed_payload_sha256",
        "header_sha256",
        "payload_bytes",
        "shape",
        "group_size",
        "layout",
    ] {
        let value = value
            .get(field)
            .cloned()
            .ok_or_else(|| format!("{label}.{field} is required"))?;
        output.insert(field.to_owned(), value);
    }
    Ok(Value::Object(output))
}

fn shallow_waves(values: &[Value]) -> Result<Vec<Value>, String> {
    values
        .iter()
        .enumerate()
        .map(|(index, wave)| {
            let wave = object(wave, &format!("route authority wave {index}"))?;
            let mut output = Map::new();
            for field in [
                "wave_index",
                "layer",
                "expert_id",
                "normalized_weight",
                "normalized_weight_bits_hex",
            ] {
                output.insert(
                    field.to_owned(),
                    wave.get(field).cloned().ok_or_else(|| {
                        format!("route authority wave {index}.{field} is required")
                    })?,
                );
            }
            for field in ["gate", "up", "down"] {
                output.insert(
                    field.to_owned(),
                    shallow_descriptor(
                        wave.get(field).ok_or_else(|| {
                            format!("route authority wave {index}.{field} is required")
                        })?,
                        &format!("route authority wave {index}.{field}"),
                    )?,
                );
            }
            Ok(Value::Object(output))
        })
        .collect()
}

fn validate_completion_preflight(
    completion: &SealedDocument,
    assessment: &SealedDocument,
    route_authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
) -> Result<(), String> {
    let root = object(&completion.value, "Layer-1 completion preflight")?;
    bool_field(
        root,
        "preflight_ready_for_future_outer_authority_only",
        true,
        "Layer-1 completion preflight",
    )?;
    let antecedent = object_field(
        root,
        "antecedent_l0_l1_component",
        "Layer-1 completion preflight",
    )?;
    require_binding_matches(
        antecedent,
        assessment,
        "Layer-1 completion preflight antecedent assessment",
    )?;
    let authority = object_field(
        root,
        "l1_source_token_route_authority",
        "Layer-1 completion preflight",
    )?;
    bool_field(
        authority,
        "present_and_valid",
        true,
        "Layer-1 completion preflight route authority",
    )?;
    let binding = object_field(
        authority,
        "binding",
        "Layer-1 completion preflight route authority",
    )?;
    if binding.get("document_sha256").and_then(Value::as_str)
        != Some(route_authority.route_authority_document_sha256.as_str())
        || binding.get("document_seal_sha256").and_then(Value::as_str)
            != Some(
                route_authority
                    .route_authority_document_seal_sha256
                    .as_str(),
            )
        || binding.get("raw_sha256").and_then(Value::as_str)
            != Some(route_authority.route_authority_raw_sha256.as_str())
    {
        return Err("Layer-1 completion preflight route authority binding drifted".into());
    }
    let graph = object_field(
        root,
        "future_joint_command_graph",
        "Layer-1 completion preflight",
    )?;
    for (field, expected) in [
        ("l0_reencode_dispatches", L0_DISPATCHES),
        ("l1_prefix_dispatches", L1_PREFIX_DISPATCHES),
        ("l1_moe_suffix_dispatches", L1_MOE_SUFFIX_DISPATCHES),
        ("total_dispatches", TOTAL_DISPATCHES),
    ] {
        if graph.get(field).and_then(Value::as_u64) != Some(expected) {
            return Err(format!("Layer-1 completion preflight {field} drifted"));
        }
    }
    if graph.get("exact_kernel_trace") != Some(&Value::Array(exact_kernel_trace())) {
        return Err("Layer-1 completion preflight exact 46-kernel trace drifted".into());
    }
    let receipt = object_field(
        root,
        "future_receipt_contract",
        "Layer-1 completion preflight",
    )?;
    for (field, expected) in [
        ("inner_schema", FUTURE_INNER_SCHEMA),
        ("inner_status", FUTURE_INNER_STATUS),
        ("outer_schema", FUTURE_OUTER_SCHEMA),
        ("outer_status", FUTURE_OUTER_STATUS),
        ("release_schema", FUTURE_RELEASE_SCHEMA),
        ("release_status", FUTURE_RELEASE_STATUS),
    ] {
        if receipt.get(field).and_then(Value::as_str) != Some(expected) {
            return Err(format!(
                "Layer-1 completion preflight future receipt {field} drifted"
            ));
        }
    }
    Ok(())
}

fn validate_l0_source_outer(document: &SealedDocument) -> Result<(), String> {
    let root = object(&document.value, "source-token L0 outer preflight")?;
    let boundary = object_field(root, "claim_boundary", "source-token L0 outer preflight")?;
    for (field, expected) in [
        ("artifact_scan_performed_by_preflight", false),
        ("lease_issued", false),
        ("metal_device_or_dispatch_performed", false),
        ("watcher_server_registry_or_hcli_changed", false),
    ] {
        bool_field(
            boundary,
            field,
            expected,
            "source-token L0 outer preflight.claim_boundary",
        )?;
    }
    Ok(())
}

fn host_preflight_document(args: &Args) -> Result<Value, String> {
    let assessment = read_sealed_document(
        &args.joint_assessment,
        "joint L0-L1 assessment",
        JOINT_ASSESSMENT_SCHEMA,
        JOINT_ASSESSMENT_STATUS,
    )?;
    let route_authority = completion_preflight::validate_source_token_l1_route_authority_files(
        &args.joint_assessment,
        &args.l1_route_authority,
    )?;
    let route_authority_binding = exact_l1_route_authority_binding(&route_authority)?;
    let completion = read_sealed_document(
        &args.completion_preflight,
        "Layer-1 completion preflight",
        COMPLETION_PREFLIGHT_SCHEMA,
        COMPLETION_PREFLIGHT_STATUS,
    )?;
    validate_completion_preflight(&completion, &assessment, &route_authority)?;
    let l0_outer = read_sealed_document(
        &args.l0_source_outer_preflight,
        "source-token L0 outer preflight",
        L0_SOURCE_OUTER_SCHEMA,
        L0_SOURCE_OUTER_STATUS,
    )?;
    validate_l0_source_outer(&l0_outer)?;
    let host_binary = file_evidence(&args.host_binary, "host binary")?;
    if host_binary.bytes == 0 {
        return Err("host binary must not be empty".into());
    }
    let fixed_l1_payloads = route_authority
        .fixed_l1_payloads
        .iter()
        .enumerate()
        .map(|(index, value)| shallow_descriptor(value, &format!("fixed L1 payload {index}")))
        .collect::<Result<Vec<_>, _>>()?;
    if fixed_l1_payloads.len() != 6 {
        return Err("validated Layer-1 authority did not retain six fixed payloads".into());
    }
    let deterministic_waves = shallow_waves(&route_authority.deterministic_waves)?;
    if deterministic_waves.len() != 10 {
        return Err("validated Layer-1 authority did not retain ten routed waves".into());
    }
    let mut output = json!({
        "schema": HOST_PREFLIGHT_SCHEMA,
        "status": HOST_PREFLIGHT_STATUS,
        "host_binary": evidence_json(&host_binary),
        "joint_assessment": sealed_binding_json(&assessment),
        "completion_preflight": sealed_binding_json(&completion),
        "l0_source_outer_preflight": sealed_binding_json(&l0_outer),
        "l1_route_payload_authority": {
            "schema": L1_ROUTE_AUTHORITY_SCHEMA,
            "status": L1_ROUTE_AUTHORITY_STATUS,
            "original_inner_authority_required": true,
            "recovery_wrapper_is_provenance_only": true,
            "binding": route_authority_binding,
            "source_stable_route_ids": route_authority.route_ids,
            "source_stable_normalized_weights": route_authority.route_weights,
            "six_fixed_payloads": fixed_l1_payloads,
            "ten_ordered_waves": deterministic_waves,
            "distinct_payload_bindings": 36,
            "route_guard_required_value": 1,
            "raw_authority_embedded_in_future_device_receipt": false,
        },
        "future_same_runtime_host_interface": {
            "entrypoint": "encode_source_token_l0_l1_full_layer_same_runtime",
            "l0_encoder": "ascension_qwen80_first_residual_bridge_device::encode_source_input_l0_true_moe_capture",
            "opaque_l0_continuation_factory": "Qwen80CompleteNativeRuntime::certify_source_token_l0_true_moe_continuation",
            "l1_prefix_encoder": "Qwen80CompleteNativeRuntime::encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into",
            "l1_route_bridge_builder": "Qwen80CompleteArtifactCatalog::build_source_token_l1_all_ten_true_moe_source_bridge_from_validated_authority",
            "l1_fixed_allocator": "Qwen80CompleteNativeRuntime::upload_canonical_linear_moe_fixed_device_buffers",
            "l1_moe_suffix_encoder": "ascension_qwen80_all_ten_true_moe_graph_device::encode_all_ten_true_moe_from_first_residual",
            "consuming_finalizer": "Qwen80SameRuntimeLayer1DeltaNetPrefixEncoder::finalize_after_exact_l1_moe_completion_fence_with_readbacks",
            "receipt_last_required": true,
            "fresh_runtime_required": true,
            "same_runtime_required": true,
            "same_token_command_buffer_required": true,
            "single_fence_required": true,
            "readbacks_after_fence_required": true,
            "cross_process_pinned_buffer_or_state_import_allowed": false,
        },
        "future_metal_entrypoint": {
            "explicit_mode_required": true,
            "default_execution_disabled": true,
            "requires_new_full_l1_lease": true,
            "requires_sealed_outer_launch_authority": true,
            "requires_fresh_outer_and_inner_capture_directories": true,
            "self_hashes_current_executable": true,
            "no_device_execution_in_this_cpu_preflight": true,
            "capture_body_wired": true,
        },
        "future_joint_command_graph": {
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_reencode_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "single_fence_after_all_dispatches_required": true,
            "non_timed_structural_trace_required": true,
            "exact_kernel_trace": exact_kernel_trace(),
        },
        "future_inner_receipt_contract": {
            "schema": FUTURE_INNER_SCHEMA,
            "status": FUTURE_INNER_STATUS,
            "outer_schema": FUTURE_OUTER_SCHEMA,
            "outer_status": FUTURE_OUTER_STATUS,
            "release_schema": FUTURE_RELEASE_SCHEMA,
            "release_status": FUTURE_RELEASE_STATUS,
            "requires_distinct_cpu_and_device_hashes_with_bounded_numeric_parity": true,
            "requires_l1_route_guard_all_ten_shared_routed_sum_and_second_residual_readbacks": true,
            "requires_l0_and_l1_active_rollback_state_witnesses": true,
        },
        "claim_boundary": {
            "cpu_build_preflight_only": true,
            "catalog_or_payload_scan_performed": false,
            "metal_context_or_dispatch_performed": false,
            "lease_issued_or_consumed": false,
            "watcher_server_hcli_or_runtime_changed": false,
            "complete_layer_or_token_decoder_claim_earned": false,
            "tps_tg_or_tournament_claim_earned": false,
            "future_capture_component_only": true,
            "automatic_retry_authorized": false,
        },
    });
    seal(&mut output)?;
    Ok(output)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot render host preflight: {error}"))?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

/// The raw authority is a CPU-oracle authority.  Before submitting any L1
/// suffix work, prove that the opaque same-runtime L1-prefix owner's fresh
/// CPU oracle selected exactly that authority's ordered routes and f32
/// weights.  This is intentionally bit-exact: CPU lineage is the immutable
/// payload authority, whereas the later Metal readback is evaluated under
/// [`L1_ROUTE_WEIGHT_TOLERANCE`].
#[cfg(target_os = "macos")]
fn require_raw_authority_matches_host_post_prefix_cpu_route(
    authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    cpu_route_ids: &[u16],
    cpu_route_weights: &[f32],
) -> Result<(), String> {
    if cpu_route_ids.len() != authority.route_ids.len()
        || cpu_route_weights.len() != authority.route_weights.len()
        || cpu_route_ids.len() != 10
    {
        return Err("full-L1 host post-prefix CPU router did not retain exactly ten routes".into());
    }
    if cpu_route_ids != authority.route_ids.as_slice() {
        return Err(format!(
            "full-L1 host post-prefix CPU route IDs drifted from raw authority: cpu={cpu_route_ids:?}, authority={:?}",
            authority.route_ids
        ));
    }
    if let Some((index, (actual, expected))) = cpu_route_weights
        .iter()
        .zip(authority.route_weights.iter())
        .enumerate()
        .find(|(_, (actual, expected))| actual.to_bits() != expected.to_bits())
        .map(|(index, pair)| (index, pair))
    {
        return Err(format!(
            "full-L1 host post-prefix CPU route weights drifted from raw authority at index {index}: cpu={actual:?} (bits {:#010x}), authority={expected:?} (bits {:#010x})",
            actual.to_bits(),
            expected.to_bits()
        ));
    }
    Ok(())
}

/// Reapply the device route-guard semantics on the post-fence readback.  The
/// source-selected CPU route order and weights are immutable; observed Metal
/// f32 weights need only be finite, non-negative, and within the exact
/// tolerance used by the encoded guard kernel.  Returning the independently
/// measured maximum prevents a receipt from merely asserting a tolerance
/// scalar detached from its retained readback values.
#[cfg(target_os = "macos")]
fn require_post_fence_device_route_guard_matches_raw_authority(
    authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    route_guard: u32,
    observed_route_ids: &[u32],
    observed_route_weights: &[f32],
) -> Result<f32, String> {
    let expected_route_ids = authority
        .route_ids
        .iter()
        .copied()
        .map(u32::from)
        .collect::<Vec<_>>();
    if route_guard != 1 {
        return Err(format!(
            "full-L1 post-fence route guard flag drifted from raw authority: route_guard={route_guard}, required=1"
        ));
    }
    if observed_route_ids.len() != expected_route_ids.len()
        || observed_route_weights.len() != authority.route_weights.len()
    {
        return Err(format!(
            "full-L1 post-fence route guard readback length drifted from raw authority: observed_ids={}, expected_ids={}, observed_weights={}, expected_weights={}",
            observed_route_ids.len(),
            expected_route_ids.len(),
            observed_route_weights.len(),
            authority.route_weights.len()
        ));
    }
    if observed_route_ids != expected_route_ids {
        return Err(format!(
            "full-L1 post-fence route guard expert IDs drifted from raw authority: observed={observed_route_ids:?}, expected={expected_route_ids:?}"
        ));
    }
    let mut max_abs_error = 0.0f32;
    for (index, (&observed, &expected)) in observed_route_weights
        .iter()
        .zip(authority.route_weights.iter())
        .enumerate()
    {
        if !observed.is_finite() || observed < 0.0 || !expected.is_finite() || expected < 0.0 {
            return Err(format!(
                "full-L1 post-fence route guard observed a non-finite or negative route weight at index {index}"
            ));
        }
        let error = (observed - expected).abs();
        if error > L1_ROUTE_WEIGHT_TOLERANCE {
            return Err(format!(
                "full-L1 post-fence route guard weight drifted from raw authority at index {index}: max_abs_error={error}, tolerance={L1_ROUTE_WEIGHT_TOLERANCE}"
            ));
        }
        max_abs_error = max_abs_error.max(error);
    }
    Ok(max_abs_error)
}

/// Render the only assessor-compatible inner receipt shape after the real
/// 46-dispatch finalizer has fenced.  This helper deliberately accepts parity
/// values rather than any detached `PinnedBuffer` or input vector; the live
/// buffer custody has already been consumed inside the runtime finalizer.
///
/// The historical L0/L1 assessment is bound as provenance only.  Its former
/// process-local buffer identifiers are never embedded or accepted as an
/// execution input here.
#[cfg(target_os = "macos")]
fn build_full_l1_inner_receipt(
    parity: &Qwen80SameRuntimeL0L1FullLayerParity,
    assessment: &SealedDocument,
    authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    completion: &SealedDocument,
    runtime_identity_sha256: &str,
    tcb_identity_sha256: &str,
) -> Result<Value, String> {
    for (label, value) in [
        ("runtime identity", runtime_identity_sha256),
        ("token command buffer identity", tcb_identity_sha256),
    ] {
        if !is_sha256(value) {
            return Err(format!("{label} must be a lowercase SHA-256"));
        }
    }
    if parity.l0_dispatches != L0_DISPATCHES as usize
        || parity.l1_prefix_dispatches != L1_PREFIX_DISPATCHES as usize
        || parity.l1_moe_suffix_dispatches != L1_MOE_SUFFIX_DISPATCHES as usize
        || parity.total_dispatches != TOTAL_DISPATCHES as usize
        || !parity.same_runtime_same_command_buffer_required
        || !parity.single_fence_after_all_dispatches_required
    {
        return Err("full-L1 finalizer parity did not retain exact 23+9+14/one-fence scope".into());
    }
    let expected_kernels = L0_KERNELS
        .iter()
        .chain(L1_PREFIX_KERNELS.iter())
        .chain(L1_MOE_SUFFIX_KERNELS.iter())
        .copied()
        .collect::<Vec<_>>();
    if parity
        .structural_kernel_names
        .iter()
        .map(String::as_str)
        .ne(expected_kernels.iter().copied())
    {
        return Err("full-L1 finalizer parity structural trace drifted".into());
    }
    let suffix = &parity.l1_true_moe_suffix;
    if suffix.layer != 1
        || suffix.linear_state_slot != 1
        || suffix.all_ten_route_witnesses.len() != 10
    {
        return Err(format!(
            "full-L1 post-fence route guard suffix scope drifted from raw authority: layer={}, required=1; linear_state_slot={}, required=1; all_ten_route_witnesses={}, required=10",
            suffix.layer,
            suffix.linear_state_slot,
            suffix.all_ten_route_witnesses.len()
        ));
    }
    require_raw_authority_matches_host_post_prefix_cpu_route(
        authority,
        &suffix.expected_route_ids,
        &suffix.expected_route_weights,
    )?;
    let observed_route_weights_max_abs_error =
        require_post_fence_device_route_guard_matches_raw_authority(
            authority,
            suffix.route_guard,
            &suffix.observed_route_ids,
            &suffix.observed_route_weights,
        )?;
    if !suffix.route_weights_max_abs_error.is_finite()
        || suffix.route_weights_max_abs_error > L1_ROUTE_WEIGHT_TOLERANCE
        || suffix.postnorm_max_abs_error > 1.0e-3
        || suffix.router_logits_max_abs_error > 1.0e-3
        || suffix.shared_max_abs_error > 1.0e-3
        || suffix.routed_sum_max_abs_error > 1.0e-3
        || suffix.second_residual_max_abs_error > 1.0e-3
        || parity.l1_prefix.input_max_abs_error > 1.0e-3
        || parity.l1_prefix.first_residual_max_abs_error > 1.0e-3
        || parity.l1_prefix.conv_state_max_abs_error > 1.0e-3
        || parity.l1_prefix.recurrent_state_max_abs_error > 1.0e-3
    {
        return Err("full-L1 post-fence parity exceeded the component tolerance".into());
    }
    let mut route_payloads = Vec::with_capacity(30);
    for (route_index, wave) in authority.deterministic_waves.iter().enumerate() {
        let wave = object(wave, &format!("validated Layer-1 route wave {route_index}"))?;
        let expert_id = wave
            .get("expert_id")
            .and_then(Value::as_u64)
            .ok_or_else(|| {
                format!("validated Layer-1 route wave {route_index}.expert_id missing")
            })?;
        if expert_id != u64::from(authority.route_ids[route_index]) {
            return Err("validated Layer-1 route authority expert order drifted".into());
        }
        let witness = suffix
            .all_ten_route_witnesses
            .get(route_index)
            .ok_or("full-L1 finalizer omitted a routed witness")?;
        if witness.wave_index != route_index || witness.expert_id != expert_id as usize {
            return Err("full-L1 post-fence routed witness order drifted".into());
        }
        for payload_kind in ["gate", "up", "down"] {
            let descriptor = object_field(
                wave,
                payload_kind,
                &format!("validated Layer-1 route wave {route_index}"),
            )?;
            let payload_identity_sha256 = descriptor
                .get("direct_packed_payload_sha256")
                .and_then(Value::as_str)
                .filter(|value| is_sha256(value))
                .ok_or_else(|| {
                    format!(
                        "validated Layer-1 route wave {route_index}.{payload_kind} payload SHA missing"
                    )
                })?;
            let tensor_sha256 = descriptor
                .get("artifact_sha256")
                .and_then(Value::as_str)
                .filter(|value| is_sha256(value))
                .ok_or_else(|| {
                    format!(
                        "validated Layer-1 route wave {route_index}.{payload_kind} artifact SHA missing"
                    )
                })?;
            route_payloads.push(json!({
                "route_index": route_index,
                "expert_id": expert_id,
                "payload_kind": payload_kind,
                "payload_identity_sha256": payload_identity_sha256,
                "tensor_sha256": tensor_sha256,
            }));
        }
    }
    if route_payloads.len() != 30 {
        return Err("full-L1 receipt did not retain exactly thirty route payloads".into());
    }
    let pair = |cpu: &str, device: &str, max_abs_error: f32| -> Result<Value, String> {
        if !is_sha256(cpu)
            || !is_sha256(device)
            || !max_abs_error.is_finite()
            || max_abs_error < 0.0
        {
            return Err("full-L1 parity evidence is malformed".into());
        }
        Ok(json!({
            "passed": true,
            "cpu_f32le_sha256": cpu,
            "device_f32le_sha256": device,
            "max_abs_error": max_abs_error,
        }))
    };
    let prefix = &parity.l1_prefix;
    let l1_readbacks = json!({
        "layer": 1,
        "slot": 1,
        "output_elements": 2048,
        "output_bytes": 8192,
        "input": pair(&prefix.input_f32le_sha256, &prefix.device_input_f32le_sha256, prefix.input_max_abs_error)?,
        "prefix_first_residual": pair(&prefix.cpu_first_residual_f32le_sha256, &prefix.device_first_residual_f32le_sha256, prefix.first_residual_max_abs_error)?,
        "postnorm": pair(&suffix.postnorm_cpu_f32le_sha256, &suffix.postnorm_output_f32le_sha256, suffix.postnorm_max_abs_error)?,
        "router_logits": pair(&suffix.router_logits_cpu_f32le_sha256, &suffix.router_logits_output_f32le_sha256, suffix.router_logits_max_abs_error)?,
        "shared_output": pair(&suffix.shared_cpu_f32le_sha256, &suffix.shared_output_f32le_sha256, suffix.shared_max_abs_error)?,
        "routed_sum": pair(&suffix.routed_sum_cpu_f32le_sha256, &suffix.routed_sum_output_f32le_sha256, suffix.routed_sum_max_abs_error)?,
        "second_residual_output": pair(&suffix.second_residual_cpu_f32le_sha256, &suffix.second_residual_output_f32le_sha256, suffix.second_residual_max_abs_error)?,
        "active_conv": {
            "passed": true,
            "slot": 1,
            "offset_bytes": prefix.conv_state_offset_elements.checked_mul(4).ok_or("Layer-1 conv byte offset overflowed")?,
            "capacity_bytes": prefix.conv_state_capacity_elements.checked_mul(4).ok_or("Layer-1 conv byte capacity overflowed")?,
            "state_identity_sha256": prefix.device_post_conv_state_f32le_sha256,
            "device_buffer_identity_sha256": prefix.active_conv_state_buffer_identity_sha256,
            "f32le_sha256": prefix.device_post_conv_state_f32le_sha256,
            "rollback_is_exact_zero": false,
            "max_abs_error": prefix.conv_state_max_abs_error,
        },
        "active_recurrent": {
            "passed": true,
            "slot": 1,
            "offset_bytes": prefix.recurrent_state_offset_elements.checked_mul(4).ok_or("Layer-1 recurrent byte offset overflowed")?,
            "capacity_bytes": prefix.recurrent_state_capacity_elements.checked_mul(4).ok_or("Layer-1 recurrent byte capacity overflowed")?,
            "state_identity_sha256": prefix.device_post_recurrent_state_f32le_sha256,
            "device_buffer_identity_sha256": prefix.active_recurrent_state_buffer_identity_sha256,
            "f32le_sha256": prefix.device_post_recurrent_state_f32le_sha256,
            "rollback_is_exact_zero": false,
            "max_abs_error": prefix.recurrent_state_max_abs_error,
        },
        "rollback_conv": {
            "passed": true,
            "slot": 1,
            "offset_bytes": prefix.conv_state_offset_elements.checked_mul(4).ok_or("Layer-1 rollback conv byte offset overflowed")?,
            "capacity_bytes": prefix.conv_state_capacity_elements.checked_mul(4).ok_or("Layer-1 rollback conv byte capacity overflowed")?,
            "state_identity_sha256": prefix.rollback_conv_state_f32le_sha256,
            "device_buffer_identity_sha256": prefix.rollback_conv_state_buffer_identity_sha256,
            "f32le_sha256": prefix.rollback_conv_state_f32le_sha256,
            "rollback_is_exact_zero": true,
            "max_abs_error": 0.0,
        },
        "rollback_recurrent": {
            "passed": true,
            "slot": 1,
            "offset_bytes": prefix.recurrent_state_offset_elements.checked_mul(4).ok_or("Layer-1 rollback recurrent byte offset overflowed")?,
            "capacity_bytes": prefix.recurrent_state_capacity_elements.checked_mul(4).ok_or("Layer-1 rollback recurrent byte capacity overflowed")?,
            "state_identity_sha256": prefix.rollback_recurrent_state_f32le_sha256,
            "device_buffer_identity_sha256": prefix.rollback_recurrent_state_buffer_identity_sha256,
            "f32le_sha256": prefix.rollback_recurrent_state_f32le_sha256,
            "rollback_is_exact_zero": true,
            "max_abs_error": 0.0,
        },
    });
    let fresh_l0 = &parity.fresh_l0;
    let mut receipt = json!({
        "schema": FUTURE_INNER_SCHEMA,
        "status": FUTURE_INNER_STATUS,
        "fixture_or_synthetic": false,
        "self_asserted": false,
        "historical_component_provenance": {
            "present": true,
            "document_sha256": assessment.document_sha256,
            "document_seal_sha256": assessment.seal_sha256,
        },
        "historical_provenance_only": true,
        "prior_process_or_buffer_reuse_accepted": false,
        "source_authority_bindings": {
            "original_l1_route_authority": {
                "path": authority.route_authority_path,
                "present": true,
                "raw_sha256": authority.route_authority_raw_sha256,
                "document_sha256": authority.route_authority_document_sha256,
                "document_seal_sha256": authority.route_authority_document_seal_sha256,
            },
            "completion_preflight": sealed_binding_json(completion),
            "fixed_payloads": authority.fixed_l1_payloads,
            "route_waves": shallow_waves(&authority.deterministic_waves)?,
        },
        "fresh_same_runtime_execution": {
            "fresh_runtime": true,
            "fresh_session": true,
            "same_runtime": true,
            "same_tcb": true,
            "l0_reencoded_in_this_capture": true,
            "l1_prefix_and_moe_suffix_in_this_capture": true,
            "route_guard_enforced_before_l1_moe_suffix": true,
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "fence_count": 1,
            "runtime_identity_sha256": runtime_identity_sha256,
            "tcb_identity_sha256": tcb_identity_sha256,
        },
        "opaque_same_runtime_continuation": {
            "opaque": true,
            "same_runtime_state_arena_bound": true,
            "same_command_buffer_bound": true,
            "non_transferable_across_processes": true,
            "raw_pinned_buffer_or_dispatch_count_input_accepted": false,
            "runtime_identity_sha256": runtime_identity_sha256,
            "tcb_identity_sha256": tcb_identity_sha256,
        },
        "structural_kernel_trace": {
            "non_timed": true,
            "exact_order": true,
            "kernel_names": parity.structural_kernel_names,
        },
        "single_fence": {
            "only_command_buffer_consumed": true,
            "fence_succeeded": true,
            "readbacks_after_fence": true,
            "append_after_fence_possible": false,
            "fence_count": 1,
        },
        "l1_route_payload_authority": {
            "route_guard": {
                "passed": true,
                "value": suffix.route_guard,
                "expected_route_ids": suffix.expected_route_ids,
                "observed_route_ids": suffix.observed_route_ids,
                "expected_route_weights": suffix.expected_route_weights,
                "observed_route_weights": suffix.observed_route_weights,
                "weights_max_abs_error": observed_route_weights_max_abs_error,
                "cpu_device_weights_max_abs_error": suffix.route_weights_max_abs_error,
            },
            "route_payloads": route_payloads,
        },
        "l1_completion_readbacks": l1_readbacks,
        "fresh_l0_reencode_readbacks": {
            "layer": 0,
            "slot": 0,
            "first_residual": pair(
                &fresh_l0.first_residual.cpu_first_residual_f32le_sha256,
                &fresh_l0.first_residual.device_first_residual_f32le_sha256,
                fresh_l0.first_residual.first_residual_max_abs_error,
            )?,
            "second_residual": pair(
                &fresh_l0.second_residual_cpu_f32le_sha256,
                &fresh_l0.second_residual_device_f32le_sha256,
                fresh_l0.second_residual_max_abs_error,
            )?,
            "active_conv_state": {
                "device_buffer_identity_sha256": fresh_l0.first_residual.active_conv_state_buffer_identity_sha256,
                "post_state_f32le_sha256": fresh_l0.first_residual.device_post_conv_state_f32le_sha256,
                "rollback_buffer_identity_sha256": fresh_l0.first_residual.rollback_conv_state_buffer_identity_sha256,
                "rollback_f32le_sha256": fresh_l0.first_residual.rollback_conv_state_f32le_sha256,
                "max_abs_error": fresh_l0.first_residual.conv_state_max_abs_error,
                "rollback_is_exact_zero": true,
            },
            "active_recurrent_state": {
                "device_buffer_identity_sha256": fresh_l0.first_residual.active_recurrent_state_buffer_identity_sha256,
                "post_state_f32le_sha256": fresh_l0.first_residual.device_post_recurrent_state_f32le_sha256,
                "rollback_buffer_identity_sha256": fresh_l0.first_residual.rollback_recurrent_state_buffer_identity_sha256,
                "rollback_f32le_sha256": fresh_l0.first_residual.rollback_recurrent_state_f32le_sha256,
                "max_abs_error": fresh_l0.first_residual.recurrent_state_max_abs_error,
                "rollback_is_exact_zero": true,
            },
            "second_residual_buffer_identity_sha256": fresh_l0.second_residual_buffer_identity_sha256,
        },
        "claim_boundary": {
            "complete_l1_component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "server_or_watcher_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
            "next_layer_executed": false,
        },
    });
    seal(&mut receipt)?;
    Ok(receipt)
}

/// The real same-runtime 46-dispatch capture body.  It has no input path for
/// a detached receipt buffer: the opaque L0 continuation is produced and
/// consumed wholly inside this one runtime and token command buffer.
#[cfg(target_os = "macos")]
fn encode_source_token_l0_l1_full_layer_same_runtime(
    runtime: &Qwen80CompleteNativeRuntime,
    command: TokenCommandBuffer<'_>,
    l0_source_outer_preflight: &Path,
    authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    workers: usize,
    phase: &mut MetalExecutionPhase,
) -> Result<Qwen80SameRuntimeL0L1FullLayerParity, String> {
    let (l0_source_bridge, _) = source_l0::build_source_token_all_ten_bridge_from_outer_preflight(
        runtime,
        l0_source_outer_preflight,
        workers,
    )?;
    let route = Qwen80RouteSelection {
        ids: authority.route_ids,
        weights: authority.route_weights,
    };
    let l1_source_bridge = runtime
        .catalog()
        .build_source_token_l1_all_ten_true_moe_source_bridge_from_validated_authority(
            "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10",
            &authority.route_authority_document_sha256,
            route,
        )
        .map_err(|error| format!("same-runtime Layer-1 route bridge refused: {error}"))?;
    let mut command = command;
    command
        .enable_structural_kernel_trace()
        .map_err(|error| format!("same-runtime full-L1 host requires structural trace: {error}"))?;
    phase.structural_kernel_trace_enabled = true;
    let l0_resources = source_prefix::encode_source_input_l0_true_moe_capture(
        runtime,
        &mut command,
        SOURCE_TOKEN_ID as u32,
        &l0_source_bridge,
    )?;
    let continuation = l0_resources.into_canonical_l0_true_moe_continuation(runtime, &command)?;
    let l1_prefix = runtime
        .encode_source_token_l1_deltanet_prefix_from_canonical_l0_continuation_into(
            &mut command,
            continuation,
        )
        .map_err(|error| format!("same-runtime Layer-1 prefix refused: {error}"))?;
    // The dynamic authority was created from this exact source-token CPU
    // lineage.  Prove that relationship before any L1 suffix kernel can be
    // submitted; a device readback later keeps bounded numeric parity rather
    // than pretending independent f32 normalisation must be bit-identical.
    let l1_cpu = l1_prefix
        .derive_fresh_l1_full_cpu_oracle(runtime)
        .map_err(|error| format!("same-runtime Layer-1 CPU route authority refused: {error}"))?;
    require_raw_authority_matches_host_post_prefix_cpu_route(
        authority,
        &l1_cpu.route.ids,
        &l1_cpu.route.weights,
    )?;
    let l1_route_bridge = runtime
        .upload_all_ten_true_moe_device_bridge(
            &l1_source_bridge,
            l1_prefix.first_residual().to_owned(),
        )
        .map_err(|error| format!("same-runtime Layer-1 route upload refused: {error}"))?;
    let fixed = runtime
        .upload_canonical_linear_moe_fixed_device_buffers(1)
        .map_err(|error| format!("same-runtime Layer-1 fixed upload refused: {error}"))?;
    let fixed_buffers = all_ten::Qwen80AllTenTrueMoeGraphFixedBuffers {
        postnorm_signs: &fixed.postnorm.signs,
        postnorm_scales: &fixed.postnorm.scales,
        postnorm_hidden: &fixed.postnorm_hidden,
        router_signs: &fixed.router.signs,
        router_scales: &fixed.router.scales,
        router_logits: &fixed.router_logits,
        router_probabilities: &fixed.router_probabilities,
        router_route_ids: &fixed.router_route_ids,
        router_route_weights: &fixed.router_route_weights,
        route_guard: &fixed.route_guard,
        route_gate: &fixed.route_gate,
        route_up: &fixed.route_up,
        route_activated: &fixed.route_activated,
        route_weighted: &fixed.route_weighted,
        shared_gate_signs: &fixed.shared_gate_proj.signs,
        shared_gate_scales: &fixed.shared_gate_proj.scales,
        shared_up_signs: &fixed.shared_up_proj.signs,
        shared_up_scales: &fixed.shared_up_proj.scales,
        shared_down_signs: &fixed.shared_down_proj.signs,
        shared_down_scales: &fixed.shared_down_proj.scales,
        shared_scalar_signs: &fixed.shared_expert_gate.signs,
        shared_scalar_scales: &fixed.shared_expert_gate.scales,
        shared_gate: &fixed.shared_gate,
        shared_up: &fixed.shared_up,
        shared_activated: &fixed.shared_activated,
        shared_output: &fixed.shared_output,
        shared_scalar_logit: &fixed.shared_scalar_logit,
        gated_shared: &fixed.gated_shared,
        routed_sum: &fixed.routed_sum,
        second_residual: &fixed.second_residual,
    };
    let graph = all_ten::Qwen80AllTenTrueMoeGraphBuffers::from_admitted_route_bridge(
        &l1_route_bridge,
        fixed_buffers,
    );
    let suffix_dispatches =
        all_ten::encode_all_ten_true_moe_from_first_residual(&mut command, &graph)?;
    if suffix_dispatches != L1_MOE_SUFFIX_DISPATCHES as usize
        || command.dispatch_count() != TOTAL_DISPATCHES as usize
        || command
            .structural_kernel_names()
            .ok_or("same-runtime full-L1 host did not retain structural trace")?
            .iter()
            .map(String::as_str)
            .ne(L0_KERNELS
                .iter()
                .chain(L1_PREFIX_KERNELS.iter())
                .chain(L1_MOE_SUFFIX_KERNELS.iter())
                .copied())
    {
        return Err("same-runtime full-L1 host 23+9+14 structural trace drifted".into());
    }
    phase.dispatches_encoded = command.dispatch_count();
    phase.encoded_kernel_names = command
        .structural_kernel_names()
        .ok_or("same-runtime full-L1 host did not retain structural trace")?
        .to_vec();
    // The runtime consuming finalizer owns the single fence.  The route bridge
    // and fixed buffers remain live through this call, which is the custody
    // required for a later receipt-last implementation to snapshot every L1
    // suffix output only after the common fence succeeds.
    phase.command_commit_may_have_been_attempted = true;
    let parity = l1_prefix
        .finalize_after_exact_l1_moe_completion_fence_with_readbacks(runtime, command, &fixed)
        .map_err(|error| format!("same-runtime full-L1 exact finalizer refused: {error}"))?;
    phase.command_fence_succeeded = true;
    phase.readback_started = true;
    Ok(parity)
}

#[cfg(target_os = "macos")]
fn verify_metal_capture_paths(args: &MetalArgs) -> Result<(PathBuf, PathBuf), String> {
    let outer = canonical_directory(&args.outer_capture_dir, "full-L1 outer capture directory")?;
    let capture = absolute_new_directory(args.capture_dir.clone(), &outer)?;
    Ok((outer, capture))
}

#[cfg(target_os = "macos")]
fn write_capture_invocation(
    authority: &FullL1CaptureAuthority,
    capture: &Path,
) -> Result<(), String> {
    let mut invocation = json!({
        "schema": "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_inner_invocation.v1",
        "status": "STARTED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_STRICT_METAL_CHILD_OUTER_REAPED",
        "mode": "metal",
        "capture_body_wired": true,
        "host_binary": evidence_json(&authority.host_binary),
        "outer_launch_authority": sealed_binding_json(&authority.outer_launch),
        "outer_preflight": sealed_binding_json(&authority.outer_preflight),
        "host_preflight": sealed_binding_json(&authority.host_preflight),
        "full_l1_lease": {"lease_id": authority.lease_id, "receipt": sealed_binding_json(&authority.lease)},
        "capture_directory": capture.to_string_lossy(),
        "execution_policy": {
            "source_token_id": SOURCE_TOKEN_ID,
            "l0_reencode_dispatches": L0_DISPATCHES,
            "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
            "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
            "total_dispatches": TOTAL_DISPATCHES,
            "single_fence_required": true,
            "non_timed": true,
            "tcb_trace_mode": "off",
        },
        "claim_boundary": {
            "component_only": true,
            "complete_layer_or_token_decoder_authorized": false,
            "server_hcli_tps_or_tournament_authorized": false,
            "historical_pinned_buffer_import_authorized": false,
        },
    });
    seal(&mut invocation)?;
    write_new(&capture.join("invocation.json"), &invocation)
}

#[cfg(target_os = "macos")]
fn execution_identities(parity: &Qwen80SameRuntimeL0L1FullLayerParity) -> (String, String) {
    let fresh_l0 = &parity.fresh_l0.first_residual;
    let l1 = &parity.l1_prefix;
    let runtime = derived_identity(
        "qwen80-full-l1-runtime",
        &[
            &fresh_l0.active_conv_state_buffer_identity_sha256,
            &fresh_l0.active_recurrent_state_buffer_identity_sha256,
            &l1.active_conv_state_buffer_identity_sha256,
            &l1.active_recurrent_state_buffer_identity_sha256,
        ],
    );
    let trace = parity.structural_kernel_names.join("|");
    let tcb = derived_identity(
        "qwen80-full-l1-token-command-buffer",
        &[
            &runtime,
            &parity.fresh_l0.second_residual_buffer_identity_sha256,
            &l1.first_residual_buffer_identity_sha256,
            &trace,
        ],
    );
    (runtime, tcb)
}

#[cfg(target_os = "macos")]
fn build_success_inner_receipt(
    authority: &FullL1CaptureAuthority,
    capture: &Path,
    parity: &Qwen80SameRuntimeL0L1FullLayerParity,
    phase: &MetalExecutionPhase,
) -> Result<Value, String> {
    if !phase.strict_artifact_admission_succeeded
        || !phase.metal_context_constructed
        || !phase.structural_kernel_trace_enabled
        || phase.dispatches_encoded != TOTAL_DISPATCHES as usize
        || phase
            .encoded_kernel_names
            .iter()
            .map(String::as_str)
            .ne(L0_KERNELS
                .iter()
                .chain(L1_PREFIX_KERNELS.iter())
                .chain(L1_MOE_SUFFIX_KERNELS.iter())
                .copied())
        || !phase.command_fence_succeeded
        || !phase.readback_started
    {
        return Err(
            "full-L1 success receipt cannot be built without exact fenced 46-dispatch evidence"
                .into(),
        );
    }
    let (runtime_identity_sha256, tcb_identity_sha256) = execution_identities(parity);
    let mut receipt = build_full_l1_inner_receipt(
        parity,
        &authority.joint_assessment,
        &authority.l1_route_authority,
        &authority.completion_preflight,
        &runtime_identity_sha256,
        &tcb_identity_sha256,
    )?;
    let root = receipt
        .as_object_mut()
        .ok_or("full-L1 success receipt must be an object")?;
    root.remove("seal_sha256");
    root.insert("capture_body_wired".into(), Value::Bool(true));
    root.insert(
        "outer_launch_authority_binding".into(),
        sealed_binding_json(&authority.outer_launch),
    );
    root.insert(
        "outer_preflight_binding".into(),
        sealed_binding_json(&authority.outer_preflight),
    );
    root.insert(
        "host_preflight_binding".into(),
        sealed_binding_json(&authority.host_preflight),
    );
    root.insert("host_binary".into(), evidence_json(&authority.host_binary));
    root.insert(
        "full_l1_lease_binding".into(),
        json!({"lease_id": authority.lease_id, "receipt": sealed_binding_json(&authority.lease)}),
    );
    root.insert(
        "artifact_binding".into(),
        json!({
            "manifest": sealed_binding_json(&authority.manifest),
            "admission_receipt": sealed_binding_json(&authority.admission_receipt),
        }),
    );
    root.insert("execution_phase".into(), phase_document(phase));
    root.insert(
        "durable_capture".into(),
        json!({
            "capture_directory": capture.to_string_lossy(),
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_capture_required": true,
            "replay_guarded": true,
        }),
    );
    seal(&mut receipt)?;
    validate_success_inner_receipt(&receipt, authority)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn refusal_inner_receipt(
    authority: &FullL1CaptureAuthority,
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
        "historical_provenance_only": true,
        "prior_process_or_buffer_reuse_accepted": false,
        "outer_launch_authority_binding": sealed_binding_json(&authority.outer_launch),
        "outer_preflight_binding": sealed_binding_json(&authority.outer_preflight),
        "host_preflight_binding": sealed_binding_json(&authority.host_preflight),
        "host_binary": evidence_json(&authority.host_binary),
        "full_l1_lease_binding": {"lease_id": authority.lease_id, "receipt": sealed_binding_json(&authority.lease)},
        "artifact_binding": {
            "manifest": sealed_binding_json(&authority.manifest),
            "admission_receipt": sealed_binding_json(&authority.admission_receipt),
        },
        "upstream_authorities": {
            "joint_assessment": sealed_binding_json(&authority.joint_assessment),
            "completion_preflight": sealed_binding_json(&authority.completion_preflight),
            "l0_source_outer_preflight": sealed_binding_json(&authority.l0_source_outer_preflight),
            "original_l1_route_authority": {
                "path": authority.l1_route_authority.route_authority_path,
                "present": true,
                "sha256": authority.l1_route_authority.route_authority_raw_sha256,
                "document_sha256": authority.l1_route_authority.route_authority_document_sha256,
                "document_seal_sha256": authority.l1_route_authority.route_authority_document_seal_sha256,
            },
        },
        "execution_phase": phase_document(phase),
        "terminal_error": error,
        "durable_capture": {
            "capture_directory": capture.to_string_lossy(),
            "receipt_written_last_is_completion_marker": true,
            "outer_reaped_capture_required": true,
            "replay_guarded": true,
        },
        "claim_boundary": {
            "component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "server_or_watcher_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
            "next_layer_executed": false,
        },
    });
    seal(&mut receipt)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn validate_success_inner_receipt(
    receipt: &Value,
    authority: &FullL1CaptureAuthority,
) -> Result<(), String> {
    let root = object(receipt, "full-L1 success receipt")?;
    if string_field(root, "schema", "full-L1 success receipt")? != FUTURE_INNER_SCHEMA
        || string_field(root, "status", "full-L1 success receipt")? != FUTURE_INNER_STATUS
        || root.get("capture_body_wired").and_then(Value::as_bool) != Some(true)
    {
        return Err("full-L1 success receipt schema/status/capture-body claim drifted".into());
    }
    for (field, expected) in [
        ("outer_launch_authority_binding", &authority.outer_launch),
        ("outer_preflight_binding", &authority.outer_preflight),
        ("host_preflight_binding", &authority.host_preflight),
    ] {
        require_full_binding_matches(
            object_field(root, field, "full-L1 success receipt")?,
            expected,
            &format!("full-L1 success receipt.{field}"),
        )?;
    }
    require_file_evidence_matches(
        object_field(root, "host_binary", "full-L1 success receipt")?,
        &authority.host_binary,
        "full-L1 success receipt.host_binary",
    )?;
    let lease = object_field(root, "full_l1_lease_binding", "full-L1 success receipt")?;
    if string_field(lease, "lease_id", "full-L1 success receipt full lease")? != authority.lease_id
    {
        return Err("full-L1 success receipt lease ID drifted".into());
    }
    require_full_binding_matches(
        object_field(lease, "receipt", "full-L1 success receipt full lease")?,
        &authority.lease,
        "full-L1 success receipt full lease receipt",
    )?;
    let phase = object_field(root, "execution_phase", "full-L1 success receipt")?;
    if phase.get("dispatches_encoded").and_then(Value::as_u64) != Some(TOTAL_DISPATCHES)
        || phase
            .get("command_fence_succeeded")
            .and_then(Value::as_bool)
            != Some(true)
        || phase.get("readback_started").and_then(Value::as_bool) != Some(true)
        || phase
            .get("device_dispatch_may_have_occurred")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("full-L1 success receipt phase is not a fenced device capture".into());
    }
    let trace = object_field(root, "structural_kernel_trace", "full-L1 success receipt")?;
    if trace
        .get("kernel_names")
        .and_then(Value::as_array)
        .map(Vec::len)
        != Some(TOTAL_DISPATCHES as usize)
    {
        return Err("full-L1 success receipt exact 46-kernel trace is absent".into());
    }
    let readbacks = object_field(root, "l1_completion_readbacks", "full-L1 success receipt")?;
    for field in [
        "postnorm",
        "router_logits",
        "shared_output",
        "routed_sum",
        "second_residual_output",
    ] {
        bool_field(
            object_field(readbacks, field, "full-L1 success receipt L1 readbacks")?,
            "passed",
            true,
            "full-L1 success receipt L1 readback",
        )?;
    }
    let durable = object_field(root, "durable_capture", "full-L1 success receipt")?;
    bool_field(
        durable,
        "receipt_written_last_is_completion_marker",
        true,
        "full-L1 success receipt durable capture",
    )?;
    for forbidden in [
        "input_device_buffer_id",
        "input_f32le_sha256",
        "raw_pinned_buffer",
        "raw_dispatch_count",
    ] {
        if root.contains_key(forbidden) {
            return Err(format!(
                "full-L1 success receipt may not import {forbidden}"
            ));
        }
    }
    let expected_seal = string_field(root, "seal_sha256", "full-L1 success receipt")?;
    let mut unsigned = root.clone();
    unsigned.remove("seal_sha256");
    if document_sha256(&Value::Object(unsigned))? != expected_seal {
        return Err("full-L1 success receipt seal mismatch".into());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn run_full_l1_metal(
    authority: &FullL1CaptureAuthority,
    capture: &Path,
    phase: &mut MetalExecutionPhase,
) -> Result<Value, String> {
    require_non_timed_tcb_trace_off()?;
    phase.strict_artifact_admission_started = true;
    let admission = source_artifact_admission(authority)?;
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.manifest.file.path, &admission)
        .map_err(|error| format!("full-L1 strict artifact admission failed: {error}"))?;
    phase.strict_artifact_admission_succeeded = true;
    phase.metal_context_construction_attempted = true;
    let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
        catalog,
        Qwen80CompleteRuntimeOptions {
            max_seq_len: 1,
            trace_dispatch: false,
        },
    )
    .map_err(|error| format!("full-L1 strict-Math runtime construction failed: {error}"))?;
    phase.metal_context_constructed = true;
    let command = runtime.begin_component_token_command_buffer();
    let parity = encode_source_token_l0_l1_full_layer_same_runtime(
        &runtime,
        command,
        &authority.l0_source_outer_preflight.file.path,
        &authority.l1_route_authority,
        authority.workers,
        phase,
    )?;
    build_success_inner_receipt(authority, capture, &parity, phase)
}

#[cfg(target_os = "macos")]
fn finalize_full_l1_capture(
    authority: &FullL1CaptureAuthority,
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
fn run_metal_child(args: &MetalArgs) -> Result<String, String> {
    let gate = validate_metal_gate(args)?;
    // Resolve every source/artifact/route antecedent before creating a capture
    // directory.  A failure here is a prelaunch refusal, not a partially
    // started child whose receipt lifecycle would need terminalizing.
    let authority = resolve_full_l1_capture_authority(gate)?;
    let (_outer, capture) = verify_metal_capture_paths(args)?;
    fs::create_dir(&capture).map_err(|error| {
        format!(
            "cannot create full-L1 inner capture directory {}: {error}",
            capture.display()
        )
    })?;
    let capture = canonical_directory(&capture, "full-L1 inner capture directory")?;
    let mut phase = MetalExecutionPhase::default();
    if let Err(error) = write_capture_invocation(&authority, &capture) {
        let (receipt, _) = finalize_full_l1_capture(&authority, &capture, &phase, Err(error))?;
        let seal = receipt
            .get("seal_sha256")
            .and_then(Value::as_str)
            .unwrap_or_default();
        return Err(format!(
            "full-L1 strict-Metal child sealed a pre-dispatch terminal refusal {seal}: {}",
            receipt
                .get("terminal_error")
                .and_then(Value::as_str)
                .unwrap_or("unable to persist invocation")
        ));
    }
    let outcome = run_full_l1_metal(&authority, &capture, &mut phase);
    let (receipt, success) = finalize_full_l1_capture(&authority, &capture, &phase, outcome)?;
    let seal = receipt
        .get("seal_sha256")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if success {
        Ok(seal)
    } else {
        Err(format!(
            "full-L1 strict-Metal child sealed a phase-accurate terminal refusal {seal}: {}",
            receipt
                .get("terminal_error")
                .and_then(Value::as_str)
                .unwrap_or("unknown terminal failure")
        ))
    }
}

fn cli_summary(output: &Value) -> Value {
    json!({
        "schema": output["schema"],
        "status": output["status"],
        "seal_sha256": output["seal_sha256"],
        "metal_or_gpu_activity_performed": output
            .get("metal_or_gpu_activity_performed")
            .cloned()
            .unwrap_or(Value::Bool(false)),
        "catalog_or_payload_scan_performed": output
            .get("catalog_or_payload_scan_performed")
            .cloned()
            .unwrap_or(Value::Bool(false)),
    })
}

fn main() {
    let outcome = parse_invocation(env::args().skip(1)).and_then(|invocation| match invocation {
        Invocation::Preflight(args) => {
            let output = host_preflight_document(&args)?;
            write_new(&args.out, &output)?;
            Ok(output)
        }
        Invocation::Metal(args) => {
            #[cfg(target_os = "macos")]
            {
                run_metal_child(&args).map(|seal| {
                    json!({
                        "schema": FUTURE_INNER_SCHEMA,
                        "status": FUTURE_INNER_STATUS,
                        "seal_sha256": seal,
                        "metal_or_gpu_activity_performed": true,
                        "catalog_or_payload_scan_performed": true,
                    })
                })
            }
            #[cfg(not(target_os = "macos"))]
            {
                let _ = args;
                Err("strict full-L1 Metal capture is unavailable on this target".into())
            }
        }
    });
    match outcome {
        Ok(output) => println!("{}", cli_summary(&output)),
        Err(error) => {
            eprintln!(
                "Qwen80 same-runtime full-L1 host preflight refusal: {error}\n{}",
                usage()
            );
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "macos")]
    fn sha(marker: char) -> String {
        std::iter::repeat_n(marker, 64).collect()
    }

    #[cfg(target_os = "macos")]
    fn fake_descriptor(role: &str, marker: char) -> Value {
        json!({
            "role": role,
            "tensor_name": format!("model.layers.1.{role}.{marker}"),
            "artifact_sha256": sha(marker),
            "direct_packed_payload_sha256": sha(char::from_u32(marker as u32 + 1).unwrap_or('f')),
            "header_sha256": sha('c'),
            "payload_bytes": 64,
            "shape": [512, 2048],
            "group_size": 128,
            "layout": {"magic": "HQ30G1B1"},
        })
    }

    #[cfg(target_os = "macos")]
    fn fake_authority() -> completion_preflight::ValidatedSourceTokenL1RouteAuthority {
        let route_ids = std::array::from_fn(|index| index as u16);
        let route_weights = std::array::from_fn(|index| 0.01 * (index as f32 + 1.0));
        let deterministic_waves = route_ids
            .iter()
            .enumerate()
            .map(|(index, expert_id)| {
                json!({
                    "wave_index": index,
                    "layer": 1,
                    "expert_id": expert_id,
                    "normalized_weight": route_weights[index],
                    "normalized_weight_bits_hex": format!("{:08x}", route_weights[index].to_bits()),
                    "gate": fake_descriptor("gate", 'a'),
                    "up": fake_descriptor("up", 'b'),
                    "down": fake_descriptor("down", 'c'),
                })
            })
            .collect();
        completion_preflight::ValidatedSourceTokenL1RouteAuthority {
            joint_assessment_path: PathBuf::from("/tmp/joint.json"),
            joint_assessment_raw_sha256: sha('a'),
            joint_assessment_document_sha256: sha('b'),
            joint_assessment_document_seal_sha256: sha('c'),
            route_authority_path: PathBuf::from("/tmp/original-l1-authority.json"),
            route_authority_raw_sha256: sha('d'),
            route_authority_document_sha256: sha('e'),
            route_authority_document_seal_sha256: sha('f'),
            route_ids,
            route_weights,
            fixed_l1_payloads: vec![
                fake_descriptor("postnorm", 'a'),
                fake_descriptor("router", 'b'),
                fake_descriptor("shared_gate", 'c'),
                fake_descriptor("shared_up", 'd'),
                fake_descriptor("shared_down", 'e'),
                fake_descriptor("shared_scalar", 'f'),
            ],
            deterministic_waves,
        }
    }

    #[cfg(target_os = "macos")]
    fn fake_first_residual_parity() -> Qwen80SourceInputFirstResidualParity {
        Qwen80SourceInputFirstResidualParity {
            source_token_id: 1,
            source_embedding_tensor: "model.embed_tokens.weight".into(),
            layer: 0,
            linear_state_slot: 0,
            input_f32le_sha256: sha('a'),
            initial_conv_state_f32le_sha256: sha('b'),
            initial_recurrent_state_f32le_sha256: sha('c'),
            cpu_first_residual_f32le_sha256: sha('d'),
            device_first_residual_f32le_sha256: sha('e'),
            device_post_conv_state_f32le_sha256: sha('f'),
            device_post_recurrent_state_f32le_sha256: sha('a'),
            rollback_conv_state_f32le_sha256: sha('b'),
            rollback_recurrent_state_f32le_sha256: sha('c'),
            active_conv_state_buffer_identity_sha256: sha('d'),
            active_recurrent_state_buffer_identity_sha256: sha('e'),
            rollback_conv_state_buffer_identity_sha256: sha('f'),
            rollback_recurrent_state_buffer_identity_sha256: sha('a'),
            first_residual_max_abs_error: 0.0001,
            conv_state_max_abs_error: 0.0001,
            recurrent_state_max_abs_error: 0.0001,
            first_residual_elements: 2048,
            first_residual_bytes: 8192,
            linear_conv_state_elements: 24_576,
            linear_conv_state_bytes: 98_304,
            linear_recurrent_state_elements: 524_288,
            linear_recurrent_state_bytes: 2_097_152,
            same_command_graph_required: true,
            dispatches_encoded_before_suffix: 9,
        }
    }

    #[cfg(target_os = "macos")]
    fn fake_full_parity(
        authority: &completion_preflight::ValidatedSourceTokenL1RouteAuthority,
    ) -> Qwen80SameRuntimeL0L1FullLayerParity {
        let prefix = Qwen80SameRuntimeLayer1DeltaNetPrefixParity {
            source_token_id: 1,
            layer: 1,
            linear_state_slot: 1,
            input_f32le_sha256: sha('a'),
            device_input_f32le_sha256: sha('b'),
            input_buffer_identity_sha256: sha('c'),
            cpu_first_residual_f32le_sha256: sha('d'),
            device_first_residual_f32le_sha256: sha('e'),
            first_residual_buffer_identity_sha256: sha('f'),
            device_post_conv_state_f32le_sha256: sha('a'),
            device_post_recurrent_state_f32le_sha256: sha('b'),
            rollback_conv_state_f32le_sha256: sha('c'),
            rollback_recurrent_state_f32le_sha256: sha('d'),
            active_conv_state_buffer_identity_sha256: sha('e'),
            active_recurrent_state_buffer_identity_sha256: sha('f'),
            rollback_conv_state_buffer_identity_sha256: sha('a'),
            rollback_recurrent_state_buffer_identity_sha256: sha('b'),
            input_max_abs_error: 0.0001,
            first_residual_max_abs_error: 0.0001,
            conv_state_max_abs_error: 0.0001,
            recurrent_state_max_abs_error: 0.0001,
            first_residual_elements: 2048,
            first_residual_bytes: 8192,
            conv_state_offset_elements: 24_576,
            conv_state_capacity_elements: 49_152,
            recurrent_state_offset_elements: 524_288,
            recurrent_state_capacity_elements: 1_048_576,
            required_l0_dispatches_before_prefix: 23,
            total_dispatches_after_prefix: 32,
            same_runtime_same_command_buffer_required: true,
            dispatches_encoded: 9,
        };
        let all_ten_route_witnesses = authority
            .route_ids
            .iter()
            .enumerate()
            .map(
                |(wave_index, expert_id)| Qwen80SameRuntimeL1RoutedWaveParity {
                    wave_index,
                    expert_id: usize::from(*expert_id),
                    normalized_weight: authority.route_weights[wave_index],
                    cpu_output_f32le_sha256: sha('a'),
                    device_output_f32le_sha256: sha('b'),
                    max_abs_error: 0.0001,
                },
            )
            .collect();
        Qwen80SameRuntimeL0L1FullLayerParity {
            fresh_l0: Qwen80SameRuntimeFreshL0Parity {
                first_residual: fake_first_residual_parity(),
                second_residual_cpu_f32le_sha256: sha('c'),
                second_residual_device_f32le_sha256: sha('d'),
                second_residual_buffer_identity_sha256: sha('e'),
                second_residual_max_abs_error: 0.0001,
            },
            l1_prefix: prefix,
            l1_true_moe_suffix: Qwen80SameRuntimeL1TrueMoeSuffixParity {
                layer: 1,
                linear_state_slot: 1,
                route_guard: 1,
                observed_route_ids: authority.route_ids.iter().copied().map(u32::from).collect(),
                expected_route_ids: authority.route_ids.to_vec(),
                observed_route_weights: authority.route_weights.to_vec(),
                expected_route_weights: authority.route_weights.to_vec(),
                route_weights_max_abs_error: 0.0,
                postnorm_cpu_f32le_sha256: sha('a'),
                postnorm_output_f32le_sha256: sha('b'),
                postnorm_max_abs_error: 0.0001,
                router_logits_cpu_f32le_sha256: sha('c'),
                router_logits_output_f32le_sha256: sha('d'),
                router_logits_max_abs_error: 0.0001,
                all_ten_route_witnesses,
                shared_cpu_f32le_sha256: sha('e'),
                shared_output_f32le_sha256: sha('f'),
                shared_max_abs_error: 0.0001,
                routed_sum_cpu_f32le_sha256: sha('a'),
                routed_sum_output_f32le_sha256: sha('b'),
                routed_sum_max_abs_error: 0.0001,
                second_residual_cpu_f32le_sha256: sha('c'),
                second_residual_output_f32le_sha256: sha('d'),
                second_residual_max_abs_error: 0.0001,
            },
            l0_dispatches: 23,
            l1_prefix_dispatches: 9,
            l1_moe_suffix_dispatches: 14,
            total_dispatches: 46,
            structural_kernel_names: L0_KERNELS
                .iter()
                .chain(L1_PREFIX_KERNELS.iter())
                .chain(L1_MOE_SUFFIX_KERNELS.iter())
                .map(|kernel| (*kernel).to_owned())
                .collect(),
            same_runtime_same_command_buffer_required: true,
            single_fence_after_all_dispatches_required: true,
        }
    }

    #[cfg(target_os = "macos")]
    fn fake_sealed_document(name: &str, marker: char) -> SealedDocument {
        SealedDocument {
            file: FileEvidence {
                path: PathBuf::from(format!("/tmp/{name}.json")),
                bytes: 2,
                sha256: sha(marker),
            },
            value: json!({}),
            document_sha256: sha(char::from_u32(marker as u32 + 1).unwrap_or('a')),
            seal_sha256: sha(char::from_u32(marker as u32 + 2).unwrap_or('b')),
        }
    }

    #[cfg(target_os = "macos")]
    fn fake_capture_authority() -> FullL1CaptureAuthority {
        FullL1CaptureAuthority {
            outer_preflight: fake_sealed_document("outer", 'a'),
            host_preflight: fake_sealed_document("host", 'd'),
            joint_assessment: fake_sealed_document("assessment", 'g'),
            completion_preflight: fake_sealed_document("completion", 'j'),
            l0_source_outer_preflight: fake_sealed_document("l0", 'm'),
            l1_route_authority: fake_authority(),
            manifest: fake_sealed_document("manifest", 'p'),
            admission_receipt: fake_sealed_document("admission", 's'),
            lease: fake_sealed_document("lease", 'v'),
            lease_id: sha('y'),
            outer_launch: fake_sealed_document("launch", 'b'),
            host_binary: FileEvidence {
                path: PathBuf::from("/tmp/full-l1-host"),
                bytes: 64,
                sha256: sha('e'),
            },
            workers: 1,
        }
    }

    #[cfg(target_os = "macos")]
    fn contains_forbidden_input_hash_key(value: &Value) -> bool {
        match value {
            Value::Array(values) => values.iter().any(contains_forbidden_input_hash_key),
            Value::Object(values) => values.iter().any(|(key, value)| {
                key == "input_f32le_sha256" || contains_forbidden_input_hash_key(value)
            }),
            _ => false,
        }
    }

    #[test]
    fn exact_full_layer_trace_is_23_plus_9_plus_14() {
        assert_eq!(L0_KERNELS.len() as u64, L0_DISPATCHES);
        assert_eq!(L1_PREFIX_KERNELS.len() as u64, L1_PREFIX_DISPATCHES);
        assert_eq!(L1_MOE_SUFFIX_KERNELS.len() as u64, L1_MOE_SUFFIX_DISPATCHES);
        assert_eq!(exact_kernel_trace().len() as u64, TOTAL_DISPATCHES);
        assert_eq!(exact_kernel_trace()[31]["kernel"], "qwen_next_add_residual");
        assert_eq!(
            exact_kernel_trace()[32]["kernel"],
            "qwen80_postnorm_router_top10_rmsnorm"
        );
    }

    #[test]
    fn parser_refuses_missing_or_relative_execution_inputs() {
        let error = parse_args(Vec::<String>::new()).unwrap_err();
        assert!(error.contains("workers"));
        let error = parse_args([
            "--joint-assessment".into(),
            "relative.json".into(),
            "--l1-route-authority".into(),
            "/tmp/authority.json".into(),
            "--completion-preflight".into(),
            "/tmp/completion.json".into(),
            "--l0-source-outer-preflight".into(),
            "/tmp/l0.json".into(),
            "--host-binary".into(),
            "/tmp/host".into(),
            "--out".into(),
            "/tmp/new.json".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("--joint-assessment must be absolute"));
    }

    #[test]
    fn invocation_requires_an_explicit_nonexecuting_mode() {
        let error = parse_invocation(Vec::<String>::new()).unwrap_err();
        assert!(error.contains("explicit --mode"));
        let error = parse_invocation([
            "--mode".into(),
            "metal".into(),
            "--joint-assessment".into(),
            "/tmp/not-allowed.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("metal mode refuses unsupported argument"));
    }

    #[test]
    fn metal_mode_refuses_missing_authority_chain_before_runtime_work() {
        let error = parse_invocation([
            "--mode".into(),
            "metal".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("--outer-preflight is required"));
    }

    #[test]
    fn metal_gate_validates_file_chain_without_creating_a_capture_directory() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hawking-qwen80-full-l1-gate-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("create test directory");
        let outer_capture_dir = root.join("outer");
        fs::create_dir(&outer_capture_dir).expect("create outer capture directory");
        let own = current_host_binary_evidence().expect("current test binary evidence");

        let route_path = root.join("original-l1-route-authority.json");
        let mut route = json!({
            "schema": L1_ROUTE_AUTHORITY_SCHEMA,
            "status": L1_ROUTE_AUTHORITY_STATUS,
        });
        seal(&mut route).expect("seal fake original route authority");
        fs::write(
            &route_path,
            serde_json::to_vec(&route).expect("encode fake original route authority"),
        )
        .expect("write fake original route authority");
        let route_bound = read_sealed_document(
            &route_path,
            "test original L1 route authority",
            L1_ROUTE_AUTHORITY_SCHEMA,
            L1_ROUTE_AUTHORITY_STATUS,
        )
        .expect("read fake original route authority");

        let host_path = root.join("host-preflight.json");
        let mut host = json!({
            "schema": HOST_PREFLIGHT_SCHEMA,
            "status": HOST_PREFLIGHT_STATUS,
            "host_binary": evidence_json(&own),
            "l1_route_payload_authority": {
                "binding": sealed_binding_json(&route_bound),
            },
        });
        seal(&mut host).expect("seal fake host preflight");
        fs::write(
            &host_path,
            serde_json::to_vec(&host).expect("encode fake host preflight"),
        )
        .expect("write fake host preflight");
        let host_bound = read_sealed_document(
            &host_path,
            "test host preflight",
            HOST_PREFLIGHT_SCHEMA,
            HOST_PREFLIGHT_STATUS,
        )
        .expect("read fake host preflight");

        let outer_path = root.join("outer-preflight.json");
        let mut outer = json!({
            "schema": OUTER_PREFLIGHT_SCHEMA,
            "status": OUTER_PREFLIGHT_STATUS,
            "host_binary": evidence_json(&own),
            "host_preflight": sealed_binding_json(&host_bound),
            "original_l1_route_authority": sealed_binding_json(&route_bound),
            "exact_component_scope": {
                "source_token_id": SOURCE_TOKEN_ID,
                "l0_reencode_dispatches": L0_DISPATCHES,
                "l1_prefix_dispatches": L1_PREFIX_DISPATCHES,
                "l1_moe_suffix_dispatches": L1_MOE_SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "one_fence_required": true,
                "non_timed_exact_trace_required": true,
            },
            "lifecycle": {
                "replay_guard_required": true,
                "one_child_process_required": true,
                "outer_reaped_terminal_required": true,
                "automatic_retry_authorized": false,
                "lease_or_device_execution_authorized_by_this_cpu_preflight": false,
                "real_host_metal_cli_available": true,
            },
            "future_metal_entrypoint": {"capture_body_wired": true},
        });
        seal(&mut outer).expect("seal fake outer preflight");
        fs::write(
            &outer_path,
            serde_json::to_vec(&outer).expect("encode outer"),
        )
        .expect("write fake outer preflight");
        let outer_bound = read_sealed_document(
            &outer_path,
            "test outer preflight",
            OUTER_PREFLIGHT_SCHEMA,
            OUTER_PREFLIGHT_STATUS,
        )
        .expect("read fake outer preflight");

        let lease_path = root.join("lease.json");
        let mut lease = json!({
            "schema": METAL_LEASE_SCHEMA,
            "status": METAL_LEASE_STATUS,
            "lease_id": sha('a'),
            "outer_preflight": sealed_binding_json(&outer_bound),
            "host_binary": evidence_json(&own),
            "execution_policy": {
                "metal_mode_only": true,
                "non_timed_exact_46_dispatches_required": true,
                "one_fence_required": true,
                "component_only": true,
                "l1_moe_suffix_allowed": true,
                "automatic_retry_allowed": false,
            },
        });
        seal(&mut lease).expect("seal fake lease");
        fs::write(
            &lease_path,
            serde_json::to_vec(&lease).expect("encode lease"),
        )
        .expect("write fake lease");
        let lease_bound = read_sealed_document(
            &lease_path,
            "test full-L1 lease",
            METAL_LEASE_SCHEMA,
            METAL_LEASE_STATUS,
        )
        .expect("read fake lease");

        let launch_path = root.join("launch.json");
        let capture_dir = outer_capture_dir.join("inner");
        let mut launch = json!({
            "schema": METAL_OUTER_LAUNCH_SCHEMA,
            "status": METAL_OUTER_LAUNCH_STATUS,
            "outer_preflight": sealed_binding_json(&outer_bound),
            "lease_receipt": sealed_binding_json(&lease_bound),
            "host_binary": evidence_json(&own),
            "lease_id": sha('a'),
            "planned_outer_capture_dir": outer_capture_dir,
            "planned_inner_capture_dir": capture_dir,
            "workers": 1,
            "execution_policy": {
                "metal_mode_only": true,
                "non_timed_exact_46_dispatches_required": true,
                "one_fence_required": true,
                "component_only": true,
                "l1_moe_suffix_allowed": true,
                "automatic_retry_allowed": false,
            },
            "lifecycle": {
                "replay_guard_required": true,
                "one_child_process_required": true,
                "outer_reaped_terminal_required": true,
                "terminal_receipt_written_last_required": true,
            },
        });
        seal(&mut launch).expect("seal fake launch");
        fs::write(
            &launch_path,
            serde_json::to_vec(&launch).expect("encode launch"),
        )
        .expect("write fake launch");

        let gate = validate_metal_gate(&MetalArgs {
            outer_preflight: outer_path,
            lease_receipt: lease_path,
            outer_launch_authority: launch_path,
            outer_capture_dir: outer_capture_dir.clone(),
            capture_dir: capture_dir.clone(),
            workers: 1,
        })
        .expect("fake file-only lease/outer launch chain validates");
        assert_eq!(gate.lease_id, sha('a'));
        assert_eq!(gate.workers, 1);
        assert!(!capture_dir.exists());

        for (mutation, expected_error) in [
            ("missing", "raw evidence drifted"),
            ("mismatched", "raw evidence drifted"),
            ("missing-host-present", "host_binary evidence drifted"),
        ] {
            let mut invalid_host = host.clone();
            invalid_host
                .as_object_mut()
                .expect("fake host is an object")
                .remove("seal_sha256");
            match mutation {
                "missing" => {
                    invalid_host["l1_route_payload_authority"]["binding"]
                        .as_object_mut()
                        .expect("fake route binding is an object")
                        .remove("bytes");
                }
                "mismatched" => {
                    invalid_host["l1_route_payload_authority"]["binding"]["bytes"] =
                        Value::from(route_bound.file.bytes + 1);
                }
                "missing-host-present" => {
                    invalid_host["host_binary"]
                        .as_object_mut()
                        .expect("fake host binary is an object")
                        .remove("present");
                }
                _ => unreachable!("test cases are exhaustive"),
            }
            seal(&mut invalid_host).expect("seal invalid fake host preflight");
            let invalid_path = root.join(format!("host-preflight-{mutation}.json"));
            fs::write(
                &invalid_path,
                serde_json::to_vec(&invalid_host).expect("encode invalid fake host preflight"),
            )
            .expect("write invalid fake host preflight");
            let invalid_bound = read_sealed_document(
                &invalid_path,
                "invalid test host preflight",
                HOST_PREFLIGHT_SCHEMA,
                HOST_PREFLIGHT_STATUS,
            )
            .expect("read invalid fake host preflight");
            assert!(
                validate_host_preflight_execution_bindings(&invalid_bound, &route_bound, &own,)
                    .expect_err("missing or mismatched full host binding must refuse")
                    .contains(expected_error)
            );
        }

        let unwired_path = root.join("outer-preflight-unwired.json");
        let mut unwired = outer.clone();
        unwired
            .as_object_mut()
            .expect("fake outer is an object")
            .remove("seal_sha256");
        unwired["future_metal_entrypoint"]["capture_body_wired"] = Value::Bool(false);
        seal(&mut unwired).expect("seal fake unwired outer preflight");
        fs::write(
            &unwired_path,
            serde_json::to_vec(&unwired).expect("encode unwired outer"),
        )
        .expect("write fake unwired outer");
        let error = validate_metal_gate(&MetalArgs {
            outer_preflight: unwired_path,
            lease_receipt: root.join("lease.json"),
            outer_launch_authority: root.join("launch.json"),
            outer_capture_dir: outer_capture_dir.clone(),
            capture_dir: capture_dir.clone(),
            workers: 1,
        })
        .expect_err("unwired outer must fail before any capture directory exists");
        assert!(error.contains("capture_body_wired"));
        assert!(!capture_dir.exists());
        fs::remove_dir_all(&root).expect("remove test directory");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn emitted_original_l1_route_binding_retains_full_raw_evidence() {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "hawking-qwen80-full-l1-route-binding-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&root).expect("create test directory");
        let route_path = root.join("original-l1-route-authority.json");
        let mut route = json!({
            "schema": L1_ROUTE_AUTHORITY_SCHEMA,
            "status": L1_ROUTE_AUTHORITY_STATUS,
        });
        seal(&mut route).expect("seal fake original route authority");
        fs::write(
            &route_path,
            serde_json::to_vec(&route).expect("encode fake original route authority"),
        )
        .expect("write fake original route authority");
        let route_bound = read_sealed_document(
            &route_path,
            "test original L1 route authority",
            L1_ROUTE_AUTHORITY_SCHEMA,
            L1_ROUTE_AUTHORITY_STATUS,
        )
        .expect("read fake original route authority");
        let mut authority = fake_authority();
        authority.route_authority_path = route_path;
        authority.route_authority_raw_sha256 = route_bound.file.sha256.clone();
        authority.route_authority_document_sha256 = route_bound.document_sha256.clone();
        authority.route_authority_document_seal_sha256 = route_bound.seal_sha256.clone();

        let binding = exact_l1_route_authority_binding(&authority)
            .expect("emitter reopens the validated original authority");
        let binding = binding
            .as_object()
            .expect("emitted route authority binding is an object");
        assert_eq!(binding.get("present"), Some(&Value::Bool(true)));
        assert_eq!(
            binding.get("bytes"),
            Some(&Value::from(route_bound.file.bytes))
        );
        require_full_binding_matches(
            binding,
            &route_bound,
            "emitted original L1 route authority binding",
        )
        .expect("emitter produces the exact full raw binding required by the capture gate");
        fs::remove_dir_all(&root).expect("remove test directory");
    }

    #[test]
    fn shallow_route_descriptor_drops_unrelated_cpu_input_material() {
        let descriptor = json!({
            "role": "gate",
            "tensor_name": "model.layers.1.mlp.experts.7.gate_proj.weight",
            "artifact_sha256": "a".repeat(64),
            "direct_packed_payload_sha256": "b".repeat(64),
            "header_sha256": "c".repeat(64),
            "payload_bytes": 64,
            "shape": [512, 2048],
            "group_size": 128,
            "layout": {"magic": "HQ30G1B1"},
            "input_f32le_sha256": "d".repeat(64),
        });
        let shallow = shallow_descriptor(&descriptor, "fixture").expect("descriptor parses");
        assert!(shallow.get("input_f32le_sha256").is_none());
        assert_eq!(shallow["role"], "gate");
    }

    #[test]
    fn future_receipt_grammar_matches_full_l1_assessor() {
        assert_eq!(
            FUTURE_INNER_SCHEMA,
            "hawking.ascension.qwen80_source_token_l0_l1_full_layer_same_runtime_capture.v1"
        );
        assert_eq!(
            FUTURE_INNER_STATUS,
            "CAPTURED_QWEN80_SOURCE_TOKEN_L0_REENCODE_AND_L1_COMPLETE_MOE_LAYER_SAME_RUNTIME_COMPONENT_ONLY"
        );
        assert!(FUTURE_OUTER_STATUS.contains("COMPLETE_LAYER"));
        assert!(FUTURE_RELEASE_STATUS.contains("COMPLETE_LAYER"));
    }

    #[test]
    fn cli_summary_never_relabels_a_fake_metal_result_as_cpu_only() {
        let summary = cli_summary(&json!({
            "schema": FUTURE_INNER_SCHEMA,
            "status": FUTURE_INNER_STATUS,
            "seal_sha256": "a".repeat(64),
            "metal_or_gpu_activity_performed": true,
            "catalog_or_payload_scan_performed": true,
        }));
        assert_eq!(summary["metal_or_gpu_activity_performed"], true);
        assert_eq!(summary["catalog_or_payload_scan_performed"], true);

        let cpu_summary = cli_summary(&json!({
            "schema": HOST_PREFLIGHT_SCHEMA,
            "status": HOST_PREFLIGHT_STATUS,
            "seal_sha256": "b".repeat(64),
        }));
        assert_eq!(cpu_summary["metal_or_gpu_activity_performed"], false);
        assert_eq!(cpu_summary["catalog_or_payload_scan_performed"], false);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn phase_receipt_is_accurate_for_fake_unsubmitted_and_post_submit_failures() {
        let pre_submit = phase_document(&MetalExecutionPhase::default());
        assert_eq!(pre_submit["device_dispatch_may_have_occurred"], false);
        assert_eq!(pre_submit["dispatches_encoded"], 0);

        let post_submit = phase_document(&MetalExecutionPhase {
            strict_artifact_admission_started: true,
            strict_artifact_admission_succeeded: true,
            metal_context_construction_attempted: true,
            metal_context_constructed: true,
            structural_kernel_trace_enabled: true,
            dispatches_encoded: TOTAL_DISPATCHES as usize,
            encoded_kernel_names: L0_KERNELS
                .iter()
                .chain(L1_PREFIX_KERNELS.iter())
                .chain(L1_MOE_SUFFIX_KERNELS.iter())
                .map(|name| (*name).to_owned())
                .collect(),
            command_commit_may_have_been_attempted: true,
            command_fence_succeeded: false,
            readback_started: false,
        });
        assert!(post_submit["device_dispatch_may_have_occurred"].is_null());
        assert_eq!(post_submit["dispatches_encoded"], TOTAL_DISPATCHES);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn full_inner_receipt_retains_46_post_fence_witnesses_without_importable_buffer_fields() {
        let authority = fake_authority();
        let parity = fake_full_parity(&authority);
        let assessment = SealedDocument {
            file: FileEvidence {
                path: PathBuf::from("/tmp/assessment.json"),
                bytes: 2,
                sha256: sha('a'),
            },
            value: json!({}),
            document_sha256: sha('b'),
            seal_sha256: sha('c'),
        };
        let completion = SealedDocument {
            file: FileEvidence {
                path: PathBuf::from("/tmp/completion.json"),
                bytes: 2,
                sha256: sha('d'),
            },
            value: json!({}),
            document_sha256: sha('e'),
            seal_sha256: sha('f'),
        };
        let receipt = build_full_l1_inner_receipt(
            &parity,
            &assessment,
            &authority,
            &completion,
            &sha('a'),
            &sha('b'),
        )
        .expect("fixture receipt is assessor-compatible");
        assert_eq!(receipt["schema"], FUTURE_INNER_SCHEMA);
        assert_eq!(
            receipt["l1_route_payload_authority"]["route_payloads"]
                .as_array()
                .unwrap()
                .len(),
            30
        );
        assert_eq!(
            receipt["structural_kernel_trace"]["kernel_names"]
                .as_array()
                .unwrap()
                .len(),
            46
        );
        assert_eq!(receipt["l1_completion_readbacks"]["output_bytes"], 8192);
        assert!(!contains_forbidden_input_hash_key(&receipt));
        let mut unsigned = receipt.as_object().unwrap().clone();
        let seal = unsigned
            .remove("seal_sha256")
            .unwrap()
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(document_sha256(&Value::Object(unsigned)).unwrap(), seal);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn fake_success_receipt_requires_exact_outer_lease_and_phase_bindings() {
        let authority = fake_capture_authority();
        let parity = fake_full_parity(&authority.l1_route_authority);
        let phase = MetalExecutionPhase {
            strict_artifact_admission_started: true,
            strict_artifact_admission_succeeded: true,
            metal_context_construction_attempted: true,
            metal_context_constructed: true,
            structural_kernel_trace_enabled: true,
            dispatches_encoded: TOTAL_DISPATCHES as usize,
            encoded_kernel_names: L0_KERNELS
                .iter()
                .chain(L1_PREFIX_KERNELS.iter())
                .chain(L1_MOE_SUFFIX_KERNELS.iter())
                .map(|name| (*name).to_owned())
                .collect(),
            command_commit_may_have_been_attempted: true,
            command_fence_succeeded: true,
            readback_started: true,
        };
        let receipt = build_success_inner_receipt(
            &authority,
            Path::new("/tmp/fake-full-l1-capture"),
            &parity,
            &phase,
        )
        .expect("fake post-fence parity emits a fully bound receipt");
        assert_eq!(receipt["capture_body_wired"], true);
        assert_eq!(
            receipt["outer_launch_authority_binding"]["document_seal_sha256"],
            authority.outer_launch.seal_sha256
        );
        assert_eq!(
            receipt["full_l1_lease_binding"]["lease_id"],
            authority.lease_id
        );
        validate_success_inner_receipt(&receipt, &authority)
            .expect("fresh fake receipt remains exact after serialization");

        let mut tampered = receipt;
        tampered
            .as_object_mut()
            .expect("receipt object")
            .remove("seal_sha256");
        tampered
            .as_object_mut()
            .expect("receipt object")
            .remove("outer_launch_authority_binding");
        seal(&mut tampered).expect("reseal intentionally tampered fake receipt");
        assert!(validate_success_inner_receipt(&tampered, &authority)
            .unwrap_err()
            .contains("outer_launch_authority_binding"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn full_inner_receipt_refuses_route_or_parity_drift_before_serialization() {
        let authority = fake_authority();
        let mut parity = fake_full_parity(&authority);
        parity.l1_true_moe_suffix.route_guard = 0;
        let assessment = SealedDocument {
            file: FileEvidence {
                path: PathBuf::from("/tmp/a"),
                bytes: 1,
                sha256: sha('a'),
            },
            value: json!({}),
            document_sha256: sha('b'),
            seal_sha256: sha('c'),
        };
        let completion = SealedDocument {
            file: FileEvidence {
                path: PathBuf::from("/tmp/b"),
                bytes: 1,
                sha256: sha('d'),
            },
            value: json!({}),
            document_sha256: sha('e'),
            seal_sha256: sha('f'),
        };
        let error = build_full_l1_inner_receipt(
            &parity,
            &assessment,
            &authority,
            &completion,
            &sha('a'),
            &sha('b'),
        )
        .unwrap_err();
        assert!(error.contains("route guard"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn host_post_prefix_cpu_route_authority_requires_exact_raw_ids_and_weights() {
        let authority = fake_authority();
        require_raw_authority_matches_host_post_prefix_cpu_route(
            &authority,
            &authority.route_ids,
            &authority.route_weights,
        )
        .expect("the exact post-prefix CPU route authority must be accepted");

        let mut reordered_ids = authority.route_ids;
        reordered_ids.swap(0, 1);
        assert!(require_raw_authority_matches_host_post_prefix_cpu_route(
            &authority,
            &reordered_ids,
            &authority.route_weights,
        )
        .unwrap_err()
        .contains("route IDs"));

        let mut changed_weights = authority.route_weights;
        changed_weights[0] = f32::from_bits(changed_weights[0].to_bits() + 1);
        assert!(require_raw_authority_matches_host_post_prefix_cpu_route(
            &authority,
            &authority.route_ids,
            &changed_weights,
        )
        .unwrap_err()
        .contains("route weights"));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn full_inner_receipt_applies_bounded_device_route_guard_not_bit_identity() {
        let authority = fake_authority();
        let assessment = fake_sealed_document("assessment", 'a');
        let completion = fake_sealed_document("completion", 'd');

        let mut tolerated = fake_full_parity(&authority);
        tolerated.l1_true_moe_suffix.observed_route_weights[0] += 1.0e-6;
        tolerated.l1_true_moe_suffix.route_weights_max_abs_error = 1.0e-6;
        build_full_l1_inner_receipt(
            &tolerated,
            &assessment,
            &authority,
            &completion,
            &sha('a'),
            &sha('b'),
        )
        .expect("bounded device f32 route-weight drift must retain a receipt");

        let mut reordered_ids = fake_full_parity(&authority);
        reordered_ids
            .l1_true_moe_suffix
            .observed_route_ids
            .swap(0, 1);
        assert!(build_full_l1_inner_receipt(
            &reordered_ids,
            &assessment,
            &authority,
            &completion,
            &sha('a'),
            &sha('b'),
        )
        .unwrap_err()
        .contains("route guard"));

        let mut excessive_weight_drift = fake_full_parity(&authority);
        excessive_weight_drift
            .l1_true_moe_suffix
            .observed_route_weights[0] += L1_ROUTE_WEIGHT_TOLERANCE * 2.0;
        excessive_weight_drift
            .l1_true_moe_suffix
            .route_weights_max_abs_error = 0.0;
        assert!(build_full_l1_inner_receipt(
            &excessive_weight_drift,
            &assessment,
            &authority,
            &completion,
            &sha('a'),
            &sha('b'),
        )
        .unwrap_err()
        .contains("route guard"));
    }
}
