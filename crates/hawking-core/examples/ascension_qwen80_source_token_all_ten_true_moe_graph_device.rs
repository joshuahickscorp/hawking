//! Source-token-only Qwen80 L0 true-MoE component child.
//!
//! This is deliberately separate from the historical fixture-router child.
//! Its only authority is a sealed source-token outer preflight, which binds
//! token 1, zeroed L0 DeltaNet state, the strict-Metal prefix receipt, the
//! source-token route authority, and the source-token static suffix ABI.
//! `--mode preflight` is CPU-only.  `--mode metal` is present for a future
//! explicitly leased outer-reaped invocation; it is never selected by default.

#[cfg(target_os = "macos")]
#[path = "ascension_qwen80_first_residual_bridge_device.rs"]
mod source_prefix;

#[cfg(target_os = "macos")]
use hawking_core::metal::PinnedBuffer;
#[cfg(target_os = "macos")]
use hawking_core::model::qwen80_complete_runtime::{
    Qwen80AllTenTrueMoeSourceBridge, Qwen80CompleteArtifactCatalog, Qwen80CompleteNativeRuntime,
    Qwen80CompleteRuntimeOptions, Qwen80SourceTokenAllTenRoutedExpertPlanAuthority,
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

const CHILD_SCHEMA: &str = "hawking.ascension.qwen80_source_token_all_ten_true_moe_graph_device.v1";
const CHILD_STATUS: &str = "EARNED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_LAYER0_TRUE_INPUT_ALL_TEN_ROUTE_SHARED_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_outer_preflight.v1";
const PREFLIGHT_STATUS: &str = "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_OUTER_READY_FOR_SOURCE_TOKEN_CHILD_NOT_LEASED_OR_EXECUTED";
const SOURCE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_route_plan_authority.v1";
const SOURCE_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ALL_TEN_ROUTE_PLAN_READY_FOR_NEW_TYPED_BRIDGE";
const SOURCE_PLAN_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_routed_expert_binding_plan.v1";
const SOURCE_PLAN_STATUS: &str =
    "SOURCE_TOKEN_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
const SOURCE_BRIDGE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_source_bridge.v1";
const SOURCE_BRIDGE_STATUS: &str = "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_READY_FOR_OUTER_PREFLIGHT";
const PREFIX_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_outer_capture.v1";
const PREFIX_STATUS: &str = "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY";
const FIXED_SCHEMA: &str = "hawking.ascension.qwen80_l0_true_moe_fixed_payload_contract.v1";
const FIXED_STATUS: &str = "PREPARED_QWEN80_L0_TRUE_MOE_FIXED_SUFFIX_PAYLOAD_PLAN_NOT_EXECUTED";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_quiet_metal_lease.v1";
const LEASE_STATUS: &str =
    "GRANTED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_GRAPH_NON_TIMED_DEVICE_PARITY_LEASE";
const OUTER_LAUNCH_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_outer_launch_authority.v1";
const OUTER_LAUNCH_STATUS: &str =
    "AUTHORIZED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_REAPED_ONE_SHOT_METAL_CHILD";
const PREFLIGHT_PROOF_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_true_input_all_ten_moe_graph_preflight_proof.v1";
const PREFLIGHT_PROOF_STATUS: &str = "PREFLIGHTED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_TRUE_INPUT_ALL_TEN_MOE_OUTER_AND_CHILD_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const HIDDEN: usize = 2_048;
const ROUTES: usize = 10;
const PREFIX_DISPATCHES: usize = 9;
const SUFFIX_DISPATCHES: usize = 14;
const TOTAL_DISPATCHES: usize = PREFIX_DISPATCHES + SUFFIX_DISPATCHES;
const SOURCE_TOKEN_ID: u32 = 1;

// This child is an exact successor for the earned source-token L0 chain, not
// a generic alternate-authority executor. A future representation/admission
// epoch must introduce a new child/outer rather than silently redirect this
// one through a self-consistent but different set of documents.
const EXPECTED_MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const EXPECTED_MANIFEST_SEAL_SHA256: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const EXPECTED_ADMISSION_RECEIPT_SEAL_SHA256: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const EXPECTED_SOURCE_ROUTE_AUTHORITY_SEAL_SHA256: &str =
    "5b83ba2721d2cdfc8cb60b051a79ef9d69e434b5bf7eb1949c0d5117caaad692";
const EXPECTED_TYPED_BRIDGE_SEAL_SHA256: &str =
    "329d3022e4319b9b08e3caada93f5ea6730c4bc77298343656cf4f99e2be5310";
const EXPECTED_FIRST_RESIDUAL_OUTER_SEAL_SHA256: &str =
    "1d29db9d1ef180eed105d4664be26b59facc666c596f4e453098e0f4faf3a3f1";
const EXPECTED_SOURCE_TOKEN_FIXED_SUFFIX_SHA256: &str =
    "12ff910c9c3299d5b82062d4a09ecfde69cfdcbbad5baa7c3367a35da60c1243";

/// The capture is deliberately a single non-timed command buffer.  These are
/// the exact names accepted by the dispatch API, in source order: nine
/// source-input DeltaNet prefix dispatches followed by the fourteen fixed
/// source-token MoE suffix dispatches.  The runtime records this list from
/// the successful encoder calls, rather than deriving it from the static plan
/// after the fact.
const PREFIX_KERNELS: [&str; PREFIX_DISPATCHES] = [
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

const SUFFIX_KERNELS: [&str; SUFFIX_DISPATCHES] = [
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

fn expected_kernel_order() -> Vec<&'static str> {
    PREFIX_KERNELS
        .iter()
        .chain(SUFFIX_KERNELS.iter())
        .copied()
        .collect()
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    Preflight,
    Metal,
}

#[derive(Clone, Debug)]
struct Args {
    outer_preflight: PathBuf,
    mode: Mode,
    lease_receipt: Option<PathBuf>,
    outer_launch_authority: Option<PathBuf>,
    outer_capture_dir: Option<PathBuf>,
    capture_dir: Option<PathBuf>,
    workers: usize,
}

#[derive(Clone, Debug)]
struct BoundFile {
    path: PathBuf,
    bytes: u64,
    sha256: String,
}

#[derive(Clone, Debug)]
struct Authority {
    args: Args,
    outer_preflight: BoundFile,
    outer_preflight_seal: String,
    manifest: BoundFile,
    manifest_seal: String,
    admission_current: BoundFile,
    admission_pointer_seal: String,
    admission_receipt_seal: String,
    source_audit_seal: String,
    source_revision: String,
    source_authority: BoundFile,
    source_authority_seal: String,
    source_plan: Value,
    route_ids: [u32; ROUTES],
    route_weights: [f32; ROUTES],
    first_residual: BoundFile,
    first_residual_seal: String,
    first_residual_output_sha: String,
    source_input_hidden_sha: String,
    source_cpu_first_residual_sha: String,
    source_zero_conv_state_sha: String,
    source_zero_recurrent_state_sha: String,
    typed_bridge: BoundFile,
    typed_bridge_seal: String,
    fixed_suffix: BoundFile,
    lease: Option<(BoundFile, String)>,
    lease_id: Option<String>,
    outer_launch: Option<(BoundFile, String)>,
}

/// Source-token facts revalidated from the sealed all-ten preflight before a
/// successor component reuses its admitted in-process catalog. This is not a
/// device receipt and cannot be constructed from a historical fixture plan.
#[cfg(target_os = "macos")]
#[derive(Clone, Debug, serde::Serialize)]
pub struct Qwen80SourceTokenAllTenValidatedLineage {
    pub manifest_document_sha256: String,
    pub manifest_seal_sha256: String,
    pub admission_receipt_seal_sha256: String,
    pub source_revision: String,
    pub source_token_id: u32,
    pub route_ids: [u32; ROUTES],
    pub route_weights: [f32; ROUTES],
    pub source_input_hidden_f32le_sha256: String,
    pub source_zero_conv_state_f32le_sha256: String,
    pub source_zero_recurrent_state_f32le_sha256: String,
    pub source_cpu_first_residual_f32le_sha256: String,
    pub source_first_residual_outer_seal_sha256: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_all_ten_true_moe_graph_device \\
--outer-preflight ABSOLUTE_PATH --mode preflight --workers 1..4 | \\
--outer-preflight ABSOLUTE_PATH --mode metal --lease-receipt ABSOLUTE_PATH \\
--outer-launch-authority ABSOLUTE_PATH \\
--outer-capture-dir ABSOLUTE_DIRECTORY --capture-dir NEW_ABSOLUTE_DIRECTORY --workers 1..4"
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

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !is_sha256(value) {
        return Err(format!("{label} must be a lowercase SHA-256"));
    }
    Ok(())
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
    let bytes = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    Ok(BoundFile {
        path,
        bytes: bytes.len() as u64,
        sha256: sha256_hex(&bytes),
    })
}

fn read_json(path: &Path, label: &str) -> Result<(BoundFile, Value), String> {
    let evidence = file_evidence(path, label)?;
    let raw =
        fs::read(&evidence.path).map_err(|error| format!("cannot reread {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("cannot parse {label}: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((evidence, value))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn field_object<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn field_array<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label}.{field} must be an array"))
}

fn field_string<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
}

fn field_bool(
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

fn canonical_json(value: &Value) -> Result<Value, String> {
    match value {
        Value::Array(values) => values
            .iter()
            .map(canonical_json)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        Value::Object(values) => {
            let mut ordered = BTreeMap::new();
            for (key, value) in values {
                ordered.insert(key.clone(), canonical_json(value)?);
            }
            Ok(Value::Object(ordered.into_iter().collect()))
        }
        value => Ok(value.clone()),
    }
}

/// Validate the exact canonical seal grammar used by `lab.receipts.seal`.
/// Evidence in this path is ASCII JSON, so sorted compact serde JSON is the
/// same signed representation as the Python issuer.
fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let object = object(value, label)?;
    let observed = field_string(object, "seal_sha256", label)?.to_owned();
    require_sha256(&observed, &format!("{label}.seal_sha256"))?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let canonical = canonical_json(&Value::Object(unsigned))?;
    let expected = sha256_hex(&serde_json::to_vec(&canonical).map_err(|error| error.to_string())?);
    if expected != observed {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed)
}

fn require_evidence(
    value: &Map<String, Value>,
    field: &str,
    expected: &BoundFile,
    label: &str,
) -> Result<(), String> {
    let row = field_object(value, field, label)?;
    if row.get("present").and_then(Value::as_bool) != Some(true)
        || field_string(row, "path", &format!("{label}.{field}"))?
            != expected.path.to_string_lossy()
        || row.get("bytes").and_then(Value::as_u64) != Some(expected.bytes)
        || field_string(row, "sha256", &format!("{label}.{field}"))? != expected.sha256
    {
        return Err(format!("{label}.{field} byte/path identity drifted"));
    }
    Ok(())
}

fn parse_args<I>(arguments: I) -> Result<Args, String>
where
    I: IntoIterator<Item = String>,
{
    let mut values = BTreeMap::<String, String>::new();
    let mut iter = arguments.into_iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--outer-preflight"
            | "--mode"
            | "--lease-receipt"
            | "--outer-launch-authority"
            | "--outer-capture-dir"
            | "--capture-dir"
            | "--workers" => {
                let value = iter
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} was repeated; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        }
    }
    let required_path =
        |values: &mut BTreeMap<String, String>, flag: &str| -> Result<PathBuf, String> {
            let raw = values
                .remove(flag)
                .ok_or_else(|| format!("missing {flag}; {}", usage()))?;
            let path = PathBuf::from(raw);
            if !path.is_absolute() {
                return Err(format!("{flag} must be absolute"));
            }
            Ok(path)
        };
    let outer_preflight = required_path(&mut values, "--outer-preflight")?;
    let mode = match values.remove("--mode").as_deref() {
        Some("preflight") => Mode::Preflight,
        Some("metal") => Mode::Metal,
        _ => return Err(format!("--mode must be preflight or metal; {}", usage())),
    };
    let optional_path =
        |values: &mut BTreeMap<String, String>, flag: &str| -> Result<Option<PathBuf>, String> {
            values
                .remove(flag)
                .map(|raw| {
                    let path = PathBuf::from(raw);
                    if !path.is_absolute() {
                        return Err(format!("{flag} must be absolute"));
                    }
                    Ok(path)
                })
                .transpose()
        };
    let lease_receipt = optional_path(&mut values, "--lease-receipt")?;
    let outer_launch_authority = optional_path(&mut values, "--outer-launch-authority")?;
    let outer_capture_dir = optional_path(&mut values, "--outer-capture-dir")?;
    let capture_dir = optional_path(&mut values, "--capture-dir")?;
    let workers = values
        .remove("--workers")
        .ok_or_else(|| format!("missing --workers; {}", usage()))?
        .parse::<usize>()
        .map_err(|_| "--workers must be an unsigned integer".to_owned())?;
    if !(1..=4).contains(&workers) {
        return Err("--workers must be 1..4".to_owned());
    }
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    match mode {
        Mode::Preflight if lease_receipt.is_some() || outer_launch_authority.is_some() || outer_capture_dir.is_some() || capture_dir.is_some() => {
            Err("preflight mode refuses lease/capture arguments".to_owned())
        }
        Mode::Metal if lease_receipt.is_none() || outer_launch_authority.is_none() || outer_capture_dir.is_none() || capture_dir.is_none() => {
            Err("metal mode requires --lease-receipt, --outer-launch-authority, --outer-capture-dir, and --capture-dir".to_owned())
        }
        _ => Ok(Args { outer_preflight, mode, lease_receipt, outer_launch_authority, outer_capture_dir, capture_dir, workers }),
    }
}

fn route_from_object(
    value: &Map<String, Value>,
    ids_field: &str,
    weights_field: &str,
    label: &str,
) -> Result<([u32; ROUTES], [f32; ROUTES]), String> {
    let ids = field_array(value, ids_field, label)?;
    let weights = field_array(value, weights_field, label)?;
    if ids.len() != ROUTES || weights.len() != ROUTES {
        return Err(format!("{label} must contain ten IDs and weights"));
    }
    let mut route_ids = [0u32; ROUTES];
    let mut route_weights = [0f32; ROUTES];
    for index in 0..ROUTES {
        let id = ids[index]
            .as_u64()
            .ok_or_else(|| format!("{label} ID {index} is invalid"))?;
        if id >= 512 {
            return Err(format!("{label} ID {index} is outside [0,512)"));
        }
        let weight = weights[index]
            .as_f64()
            .ok_or_else(|| format!("{label} weight {index} is invalid"))?;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(format!("{label} weight {index} is non-positive/non-finite"));
        }
        route_ids[index] = id as u32;
        route_weights[index] = weight as f32;
    }
    if route_ids
        .iter()
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != ROUTES
    {
        return Err(format!("{label} contains duplicate experts"));
    }
    if (route_weights.iter().sum::<f32>() - 1.0).abs() > 2.0e-5 {
        return Err(format!("{label} weights are not normalized"));
    }
    Ok((route_ids, route_weights))
}

fn assert_same_route(
    expected: &([u32; ROUTES], [f32; ROUTES]),
    observed: &([u32; ROUTES], [f32; ROUTES]),
    label: &str,
) -> Result<(), String> {
    if expected.0 != observed.0 {
        return Err(format!("{label} route IDs drifted"));
    }
    for index in 0..ROUTES {
        if (expected.1[index] - observed.1[index]).abs() > 1.0e-6 {
            return Err(format!("{label} route weight {index} drifted"));
        }
    }
    Ok(())
}

fn validate_fixed_suffix(
    value: &Value,
    manifest: &BoundFile,
    manifest_seal: &str,
    admission_seal: &str,
) -> Result<(), String> {
    let document = object(value, "source-token fixed suffix")?;
    if field_string(document, "schema", "source-token fixed suffix")? != FIXED_SCHEMA
        || field_string(document, "status", "source-token fixed suffix")? != FIXED_STATUS
        || document.get("seal_sha256").is_some()
    {
        return Err("source-token fixed suffix schema/status/seal drifted".to_owned());
    }
    let source = field_object(document, "source_binding", "source-token fixed suffix")?;
    if field_string(
        source,
        "manifest_document_sha256",
        "source-token fixed suffix",
    )? != manifest.sha256
        || field_string(source, "manifest_seal_sha256", "source-token fixed suffix")?
            != manifest_seal
        || field_string(
            source,
            "admission_receipt_seal_sha256",
            "source-token fixed suffix",
        )? != admission_seal
    {
        return Err("source-token fixed suffix artifact identity drifted".to_owned());
    }
    let external = field_object(document, "external_authority", "source-token fixed suffix")?;
    if field_string(external, "route_plan_schema", "source-token fixed suffix")?
        != SOURCE_AUTHORITY_SCHEMA
        || field_string(external, "route_plan_status", "source-token fixed suffix")?
            != SOURCE_AUTHORITY_STATUS
        || field_string(external, "typed_bridge_schema", "source-token fixed suffix")?
            != SOURCE_BRIDGE_SCHEMA
        || field_string(external, "typed_bridge_status", "source-token fixed suffix")?
            != SOURCE_BRIDGE_STATUS
    {
        return Err("source-token fixed suffix authority family drifted".to_owned());
    }
    for field in [
        "route_payloads_materialized_here",
        "first_residual_materialized_here",
        "expected_topk_witness_materialized_here",
        "route_tensor_sha256s_materialized_here",
    ] {
        field_bool(external, field, false, "source-token fixed suffix")?;
    }
    let dispatches = field_array(
        document,
        "fixed_14_dispatch_abi",
        "source-token fixed suffix",
    )?;
    if dispatches.len() != SUFFIX_DISPATCHES {
        return Err("source-token fixed suffix lacks fourteen dispatches".to_owned());
    }
    for (index, expected_name) in SUFFIX_KERNELS.iter().enumerate() {
        let row = object(&dispatches[index], "source-token fixed suffix dispatch")?;
        if row.get("ordinal").and_then(Value::as_u64) != Some((index + 1) as u64)
            || field_string(row, "kernel", "source-token fixed suffix dispatch")? != *expected_name
        {
            return Err("source-token fixed suffix dispatch ABI drifted".to_owned());
        }
    }
    let boundary = field_object(document, "claim_boundary", "source-token fixed suffix")?;
    for field in [
        "artifact_scan_or_payload_open_performed",
        "metal_context_or_dispatch_performed",
        "runtime_watcher_server_registry_or_hcli_changed",
        "token_or_tps_claim",
    ] {
        field_bool(boundary, field, false, "source-token fixed suffix")?;
    }
    if field_string(boundary, "execution_status", "source-token fixed suffix")?
        != "PREPARED_NOT_EXECUTED"
    {
        return Err("source-token fixed suffix execution status drifted".to_owned());
    }
    Ok(())
}

fn validate_component_lease(
    value: &Value,
    authority: &Authority,
) -> Result<(String, String), String> {
    let lease_seal = verify_seal(value, "source-token component lease")?;
    let lease = object(value, "source-token component lease")?;
    if field_string(lease, "schema", "source-token component lease")? != LEASE_SCHEMA
        || field_string(lease, "status", "source-token component lease")? != LEASE_STATUS
    {
        return Err("source-token component lease schema/status drifted".to_owned());
    }
    let lease_id = field_string(lease, "lease_id", "source-token component lease")?.to_owned();
    require_sha256(&lease_id, "source-token component lease ID")?;

    let artifact = field_object(lease, "artifact_binding", "source-token component lease")?;
    if field_string(
        artifact,
        "manifest_document_sha256",
        "source-token component lease artifact",
    )? != authority.manifest.sha256
        || field_string(
            artifact,
            "manifest_seal_sha256",
            "source-token component lease artifact",
        )? != authority.manifest_seal
        || field_string(
            artifact,
            "admission_receipt_seal_sha256",
            "source-token component lease artifact",
        )? != authority.admission_receipt_seal
    {
        return Err("source-token component lease artifact identity drifted".to_owned());
    }

    let exact_binding =
        |field: &str, path: &Path, raw_sha: &str, seal: &str| -> Result<(), String> {
            let row = field_object(lease, field, "source-token component lease")?;
            if field_string(
                row,
                "path",
                &format!("source-token component lease {field}"),
            )? != path.to_string_lossy()
                || field_string(
                    row,
                    "document_sha256",
                    &format!("source-token component lease {field}"),
                )? != raw_sha
                || field_string(
                    row,
                    "seal_sha256",
                    &format!("source-token component lease {field}"),
                )? != seal
            {
                return Err(format!(
                    "source-token component lease {field} identity drifted"
                ));
            }
            Ok(())
        };
    exact_binding(
        "outer_preflight_binding",
        &authority.outer_preflight.path,
        &authority.outer_preflight.sha256,
        &authority.outer_preflight_seal,
    )?;
    exact_binding(
        "source_token_route_authority_binding",
        &authority.source_authority.path,
        &authority.source_authority.sha256,
        &authority.source_authority_seal,
    )?;
    exact_binding(
        "typed_bridge_binding",
        &authority.typed_bridge.path,
        &authority.typed_bridge.sha256,
        &authority.typed_bridge_seal,
    )?;
    exact_binding(
        "first_residual_antecedent",
        &authority.first_residual.path,
        &authority.first_residual.sha256,
        &authority.first_residual_seal,
    )?;
    let first_residual = field_object(
        lease,
        "first_residual_antecedent",
        "source-token component lease",
    )?;
    if field_string(
        first_residual,
        "output_sha256",
        "source-token component lease first residual",
    )? != authority.first_residual_output_sha
    {
        return Err("source-token component lease first-residual output drifted".to_owned());
    }
    let fixed = field_object(
        lease,
        "fixed_suffix_contract_binding",
        "source-token component lease",
    )?;
    if field_string(fixed, "path", "source-token component lease fixed suffix")?
        != authority.fixed_suffix.path.to_string_lossy()
        || field_string(
            fixed,
            "document_sha256",
            "source-token component lease fixed suffix",
        )? != authority.fixed_suffix.sha256
        || field_string(fixed, "schema", "source-token component lease fixed suffix")?
            != FIXED_SCHEMA
        || field_string(fixed, "status", "source-token component lease fixed suffix")?
            != FIXED_STATUS
    {
        return Err("source-token component lease fixed suffix identity drifted".to_owned());
    }

    let implementation = field_object(
        lease,
        "implementation_binding",
        "source-token component lease",
    )?;
    if implementation
        .get("source_token_id")
        .and_then(Value::as_u64)
        != Some(SOURCE_TOKEN_ID as u64)
        || implementation
            .get("prefix_dispatches")
            .and_then(Value::as_u64)
            != Some(PREFIX_DISPATCHES as u64)
        || implementation
            .get("suffix_dispatches")
            .and_then(Value::as_u64)
            != Some(SUFFIX_DISPATCHES as u64)
        || implementation
            .get("total_dispatches")
            .and_then(Value::as_u64)
            != Some(TOTAL_DISPATCHES as u64)
        || implementation
            .get("same_command_buffer_fence_required")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("source-token component lease 9+14 implementation binding drifted".to_owned());
    }
    let policy = field_object(lease, "execution_policy", "source-token component lease")?;
    if field_string(policy, "component", "source-token component lease policy")?
        != "qwen80_source_token_true_input_all_ten_moe_graph"
    {
        return Err("source-token component lease component scope drifted".to_owned());
    }
    for field in ["quiet_qwen80_device_lease", "strict_math"] {
        field_bool(policy, field, true, "source-token component lease policy")?;
    }
    for field in [
        "timing_or_benchmarking_allowed",
        "complete_layer_or_token_allowed",
        "tps_or_tg_claim_allowed",
    ] {
        field_bool(policy, field, false, "source-token component lease policy")?;
    }
    let lifecycle = field_object(lease, "lifecycle", "source-token component lease")?;
    for field in [
        "fresh_for_this_exact_launch",
        "outer_reaped_capture_required",
        "lease_released_after_first_terminal_child",
        "automatic_retry_prohibited",
    ] {
        field_bool(
            lifecycle,
            field,
            true,
            "source-token component lease lifecycle",
        )?;
    }
    let watcher = field_object(
        lease,
        "watcher_coordination",
        "source-token component lease",
    )?;
    if watcher
        .get("watcher_hold_must_remain_active")
        .and_then(Value::as_bool)
        != Some(true)
        || watcher
            .get("watcher_restart_or_transition_authorized")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("source-token component lease watcher coordination drifted".to_owned());
    }
    Ok((lease_seal, lease_id))
}

fn validate_outer_launch_authority(value: &Value, authority: &Authority) -> Result<String, String> {
    let outer_seal = verify_seal(value, "source-token outer launch authority")?;
    let outer = object(value, "source-token outer launch authority")?;
    if field_string(outer, "schema", "source-token outer launch authority")? != OUTER_LAUNCH_SCHEMA
        || field_string(outer, "status", "source-token outer launch authority")?
            != OUTER_LAUNCH_STATUS
    {
        return Err("source-token outer launch authority schema/status drifted".to_owned());
    }
    let lease = authority
        .lease
        .as_ref()
        .ok_or("outer launch authority requires a validated lease")?;
    let lease_id = authority
        .lease_id
        .as_deref()
        .ok_or("outer launch authority lacks validated lease ID")?;
    require_evidence(
        outer,
        "lease_receipt",
        &lease.0,
        "source-token outer launch authority",
    )?;
    if field_string(
        outer,
        "lease_receipt_seal_sha256",
        "source-token outer launch authority",
    )? != lease.1
        || field_string(outer, "lease_id", "source-token outer launch authority")? != lease_id
    {
        return Err("source-token outer launch authority lease lineage drifted".to_owned());
    }
    require_evidence(
        outer,
        "outer_preflight",
        &authority.outer_preflight,
        "source-token outer launch authority",
    )?;
    if field_string(
        outer,
        "outer_preflight_seal_sha256",
        "source-token outer launch authority",
    )? != authority.outer_preflight_seal
    {
        return Err("source-token outer launch authority outer-preflight seal drifted".to_owned());
    }

    let proof_row = field_object(
        outer,
        "preflight_proof",
        "source-token outer launch authority",
    )?;
    let proof_path = PathBuf::from(field_string(
        proof_row,
        "path",
        "source-token outer launch authority preflight proof",
    )?);
    let proof = file_evidence(
        &proof_path,
        "source-token outer launch authority preflight proof",
    )?;
    require_evidence(
        outer,
        "preflight_proof",
        &proof,
        "source-token outer launch authority",
    )?;
    let (_, proof_value) = read_json(
        &proof.path,
        "source-token outer launch authority preflight proof",
    )?;
    let proof_seal = verify_seal(
        &proof_value,
        "source-token outer launch authority preflight proof",
    )?;
    let proof_object = object(
        &proof_value,
        "source-token outer launch authority preflight proof",
    )?;
    if field_string(
        proof_object,
        "schema",
        "source-token outer launch authority preflight proof",
    )? != PREFLIGHT_PROOF_SCHEMA
        || field_string(
            proof_object,
            "status",
            "source-token outer launch authority preflight proof",
        )? != PREFLIGHT_PROOF_STATUS
    {
        return Err(
            "source-token outer launch authority preflight proof schema/status drifted".to_owned(),
        );
    }
    let executable = std::env::current_exe().map_err(|error| {
        format!("cannot resolve current source-token child executable: {error}")
    })?;
    let executable = file_evidence(&executable, "current source-token child executable")?;
    let proof_source = field_object(
        proof_object,
        "source_binding",
        "source-token outer launch authority preflight proof",
    )?;
    require_evidence(
        proof_source,
        "probe_binary",
        &executable,
        "source-token outer launch authority preflight proof",
    )?;
    let proof_outer = field_object(
        field_object(
            proof_object,
            "source_binding",
            "source-token outer launch authority preflight proof",
        )?,
        "outer_preflight",
        "source-token outer launch authority preflight proof",
    )?;
    if field_string(
        proof_outer,
        "path",
        "source-token outer launch authority preflight proof",
    )? != authority.outer_preflight.path.to_string_lossy()
        || field_string(
            proof_outer,
            "sha256",
            "source-token outer launch authority preflight proof",
        )? != authority.outer_preflight.sha256
        || field_string(
            proof_outer,
            "seal_sha256",
            "source-token outer launch authority preflight proof",
        )? != authority.outer_preflight_seal
    {
        return Err(
            "source-token outer launch authority proof does not bind current outer preflight"
                .to_owned(),
        );
    }
    let child_proof = field_object(
        outer,
        "child_preflight_proof_binding",
        "source-token outer launch authority",
    )?;
    if field_string(
        child_proof,
        "path",
        "source-token outer launch authority child proof",
    )? != proof.path.to_string_lossy()
        || field_string(
            child_proof,
            "document_sha256",
            "source-token outer launch authority child proof",
        )? != proof.sha256
        || field_string(
            child_proof,
            "seal_sha256",
            "source-token outer launch authority child proof",
        )? != proof_seal
    {
        return Err(
            "source-token outer launch authority child-preflight proof identity drifted".to_owned(),
        );
    }
    let child = field_object(
        proof_object,
        "child_preflight",
        "source-token outer launch authority preflight proof",
    )?;
    let parsed = field_object(
        child,
        "parsed",
        "source-token outer launch authority preflight proof",
    )?;
    if child
        .get("terminal")
        .and_then(Value::as_object)
        .and_then(|terminal| terminal.get("exit_code"))
        .and_then(Value::as_i64)
        != Some(0)
        || field_string(
            parsed,
            "schema",
            "source-token outer launch authority child proof",
        )? != CHILD_SCHEMA
        || field_string(
            parsed,
            "status",
            "source-token outer launch authority child proof",
        )? != "PREPARED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_DEVICE_CHILD_NOT_LEASED_OR_EXECUTED"
        || field_string(
            parsed,
            "mode",
            "source-token outer launch authority child proof",
        )? != "preflight"
    {
        return Err("source-token outer launch authority child preflight did not earn CPU-only prepared state".to_owned());
    }
    let graph = field_object(
        parsed,
        "same_command_graph_contract",
        "source-token outer launch authority child proof",
    )?;
    if graph.get("source_token_id").and_then(Value::as_u64) != Some(SOURCE_TOKEN_ID as u64)
        || graph.get("prefix_dispatches").and_then(Value::as_u64) != Some(PREFIX_DISPATCHES as u64)
        || graph.get("suffix_dispatches").and_then(Value::as_u64) != Some(SUFFIX_DISPATCHES as u64)
        || graph.get("total_dispatches").and_then(Value::as_u64) != Some(TOTAL_DISPATCHES as u64)
        || graph.get("route_guard_required").and_then(Value::as_bool) != Some(true)
    {
        return Err(
            "source-token outer launch authority child preflight lost 9+14 route-guard contract"
                .to_owned(),
        );
    }
    let boundary = field_object(
        parsed,
        "claim_boundary",
        "source-token outer launch authority child proof",
    )?;
    if boundary
        .get("metal_device_or_dispatch_performed")
        .and_then(Value::as_bool)
        != Some(false)
        || boundary.get("lease_issued").and_then(Value::as_bool) != Some(false)
        || boundary
            .get("legacy_fixture_router_or_plan_accepted")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err(
            "source-token outer launch authority child proof is not CPU-only/fail-closed"
                .to_owned(),
        );
    }

    require_evidence(
        outer,
        "probe_binary",
        &executable,
        "source-token outer launch authority",
    )?;
    let outer_capture = authority
        .args
        .outer_capture_dir
        .as_ref()
        .ok_or("outer launch authority lacks outer capture dir")?;
    let inner_capture = authority
        .args
        .capture_dir
        .as_ref()
        .ok_or("outer launch authority lacks inner capture dir")?;
    if field_string(
        outer,
        "planned_outer_capture_dir",
        "source-token outer launch authority",
    )? != outer_capture.to_string_lossy()
        || field_string(
            outer,
            "planned_inner_capture_dir",
            "source-token outer launch authority",
        )? != inner_capture.to_string_lossy()
        || outer.get("workers").and_then(Value::as_u64) != Some(authority.args.workers as u64)
    {
        return Err(
            "source-token outer launch authority planned capture/worker identity drifted"
                .to_owned(),
        );
    }
    let expected_authority_path = outer_capture.join("outer-launch-authority.json");
    if authority
        .args
        .outer_launch_authority
        .as_ref()
        .map(PathBuf::as_path)
        != Some(expected_authority_path.as_path())
    {
        return Err("source-token outer launch authority must live at the planned outer capture authority path".to_owned());
    }
    let policy = field_object(
        outer,
        "execution_policy",
        "source-token outer launch authority",
    )?;
    for field in ["quiet_qwen80_device_lease", "strict_math"] {
        field_bool(
            policy,
            field,
            true,
            "source-token outer launch authority policy",
        )?;
    }
    for field in [
        "timing_or_benchmarking_allowed",
        "complete_layer_or_token_allowed",
        "tps_or_tg_claim_allowed",
    ] {
        field_bool(
            policy,
            field,
            false,
            "source-token outer launch authority policy",
        )?;
    }
    let lifecycle = field_object(outer, "lifecycle", "source-token outer launch authority")?;
    for field in [
        "fresh_for_this_exact_launch",
        "outer_reaped_capture_required",
        "lease_released_after_first_terminal_child",
        "automatic_retry_prohibited",
    ] {
        field_bool(
            lifecycle,
            field,
            true,
            "source-token outer launch authority lifecycle",
        )?;
    }
    let watcher = field_object(
        outer,
        "watcher_coordination",
        "source-token outer launch authority",
    )?;
    if watcher
        .get("watcher_hold_must_remain_active")
        .and_then(Value::as_bool)
        != Some(true)
        || watcher
            .get("watcher_restart_or_transition_authorized")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("source-token outer launch authority watcher coordination drifted".to_owned());
    }
    Ok(outer_seal)
}

fn validate_authority(args: &Args) -> Result<Authority, String> {
    let (outer_preflight, outer) = read_json(&args.outer_preflight, "--outer-preflight")?;
    let outer_seal = verify_seal(&outer, "source-token outer preflight")?;
    let outer_object = object(&outer, "source-token outer preflight")?;
    if field_string(outer_object, "schema", "source-token outer preflight")? != PREFLIGHT_SCHEMA
        || field_string(outer_object, "status", "source-token outer preflight")? != PREFLIGHT_STATUS
    {
        return Err("source-token outer preflight schema/status drifted".to_owned());
    }
    let boundary = field_object(
        outer_object,
        "claim_boundary",
        "source-token outer preflight",
    )?;
    field_bool(
        boundary,
        "metal_device_or_dispatch_performed",
        false,
        "source-token outer preflight",
    )?;
    field_bool(
        boundary,
        "lease_issued",
        false,
        "source-token outer preflight",
    )?;
    let source = field_object(
        outer_object,
        "source_binding",
        "source-token outer preflight",
    )?;

    let manifest_path = PathBuf::from(field_string(
        field_object(source, "manifest", "source-token outer preflight")?,
        "path",
        "source-token outer preflight manifest",
    )?);
    let admission_path = PathBuf::from(field_string(
        field_object(source, "admission_current", "source-token outer preflight")?,
        "path",
        "source-token outer preflight admission",
    )?);
    let source_authority_path = PathBuf::from(field_string(
        field_object(
            source,
            "source_token_route_authority",
            "source-token outer preflight",
        )?,
        "path",
        "source-token outer preflight authority",
    )?);
    let first_residual_path = PathBuf::from(field_string(
        field_object(
            source,
            "first_residual_receipt",
            "source-token outer preflight",
        )?,
        "path",
        "source-token outer preflight prefix",
    )?);
    let typed_bridge_path = PathBuf::from(field_string(
        field_object(
            source,
            "typed_bridge_receipt",
            "source-token outer preflight",
        )?,
        "path",
        "source-token outer preflight typed bridge",
    )?);
    let fixed_suffix_path = PathBuf::from(field_string(
        field_object(
            source,
            "fixed_suffix_contract",
            "source-token outer preflight",
        )?,
        "path",
        "source-token outer preflight fixed suffix",
    )?);
    let manifest = file_evidence(&manifest_path, "outer preflight manifest")?;
    let admission_current = file_evidence(&admission_path, "outer preflight admission")?;
    let source_authority =
        file_evidence(&source_authority_path, "outer preflight source authority")?;
    let first_residual = file_evidence(&first_residual_path, "outer preflight first residual")?;
    let typed_bridge = file_evidence(&typed_bridge_path, "outer preflight typed bridge")?;
    let fixed_suffix = file_evidence(&fixed_suffix_path, "outer preflight fixed suffix")?;
    require_evidence(
        source,
        "manifest",
        &manifest,
        "source-token outer preflight",
    )?;
    // Current pointers may be safely resealed; the immutable selected manifest
    // and admission receipt remain the effective authority below.
    let historical_admission =
        field_object(source, "admission_current", "source-token outer preflight")?;
    if historical_admission.get("present").and_then(Value::as_bool) != Some(true)
        || field_string(
            historical_admission,
            "path",
            "source-token outer preflight admission",
        )? != admission_current.path.to_string_lossy()
    {
        return Err("source-token outer preflight admission path drifted".to_owned());
    }
    require_evidence(
        source,
        "source_token_route_authority",
        &source_authority,
        "source-token outer preflight",
    )?;
    require_evidence(
        source,
        "first_residual_receipt",
        &first_residual,
        "source-token outer preflight",
    )?;
    require_evidence(
        source,
        "typed_bridge_receipt",
        &typed_bridge,
        "source-token outer preflight",
    )?;
    require_evidence(
        source,
        "fixed_suffix_contract",
        &fixed_suffix,
        "source-token outer preflight",
    )?;

    let (_, manifest_value) = read_json(&manifest.path, "manifest")?;
    let manifest_object = object(&manifest_value, "manifest")?;
    if field_string(manifest_object, "schema", "manifest")? != MANIFEST_SCHEMA {
        return Err("manifest schema drifted".to_owned());
    }
    let manifest_seal = verify_seal(&manifest_value, "manifest")?;
    if manifest.sha256 != EXPECTED_MANIFEST_DOCUMENT_SHA256
        || manifest_seal != EXPECTED_MANIFEST_SEAL_SHA256
        || field_string(
            source,
            "manifest_seal_sha256",
            "source-token outer preflight",
        )? != manifest_seal
    {
        return Err(
            "source-token successor manifest identity is not the pinned current authority"
                .to_owned(),
        );
    }

    let (_, admission_value) = read_json(&admission_current.path, "admission current")?;
    let admission_object = object(&admission_value, "admission current")?;
    if field_string(admission_object, "schema", "admission current")? != ADMISSION_SCHEMA
        || field_string(admission_object, "status", "admission current")? != ADMISSION_STATUS
    {
        return Err("admission current schema/status drifted".to_owned());
    }
    let admission_pointer_seal = verify_seal(&admission_value, "admission current")?;
    let selected_manifest =
        field_object(admission_object, "complete_manifest", "admission current")?;
    if field_string(
        selected_manifest,
        "document_sha256",
        "admission current manifest",
    )? != manifest.sha256
        || field_string(
            selected_manifest,
            "seal_sha256",
            "admission current manifest",
        )? != manifest_seal
    {
        return Err("admission current manifest identity drifted".to_owned());
    }
    let selected_receipt =
        field_object(admission_object, "admission_receipt", "admission current")?;
    let receipt_path = PathBuf::from(field_string(
        selected_receipt,
        "path",
        "admission current receipt",
    )?);
    let receipt_evidence = file_evidence(&receipt_path, "immutable admission receipt")?;
    require_evidence(
        source,
        "admission_receipt",
        &receipt_evidence,
        "source-token outer preflight",
    )?;
    let (_, receipt_value) = read_json(&receipt_path, "immutable admission receipt")?;
    let receipt_object = object(&receipt_value, "immutable admission receipt")?;
    let admission_receipt_seal = verify_seal(&receipt_value, "immutable admission receipt")?;
    if field_string(selected_receipt, "seal_sha256", "admission current receipt")?
        != admission_receipt_seal
        || field_string(
            source,
            "admission_receipt_seal_sha256",
            "source-token outer preflight",
        )? != admission_receipt_seal
        || admission_receipt_seal != EXPECTED_ADMISSION_RECEIPT_SEAL_SHA256
    {
        return Err("immutable admission receipt seal drifted".to_owned());
    }
    let revalidation = field_object(
        receipt_object,
        "current_source_revalidation",
        "immutable admission receipt",
    )?;
    let source_audit_seal = field_string(
        revalidation,
        "source_audit_seal_sha256",
        "admission revalidation",
    )?
    .to_owned();
    let source_revision =
        field_string(revalidation, "revision", "admission revalidation")?.to_owned();
    require_sha256(&source_audit_seal, "source audit seal")?;

    let (_, authority_value) = read_json(&source_authority.path, "source-token route authority")?;
    let source_authority_seal = verify_seal(&authority_value, "source-token route authority")?;
    let authority_object = object(&authority_value, "source-token route authority")?;
    if field_string(authority_object, "schema", "source-token route authority")?
        != SOURCE_AUTHORITY_SCHEMA
        || field_string(authority_object, "status", "source-token route authority")?
            != SOURCE_AUTHORITY_STATUS
        || field_string(
            source,
            "source_token_route_authority_seal_sha256",
            "source-token outer preflight",
        )? != source_authority_seal
        || source_authority_seal != EXPECTED_SOURCE_ROUTE_AUTHORITY_SEAL_SHA256
    {
        return Err("source-token route authority schema/status/seal drifted".to_owned());
    }
    let authority_source = field_object(
        authority_object,
        "source_binding",
        "source-token route authority",
    )?;
    require_evidence(
        authority_source,
        "manifest",
        &manifest,
        "source-token route authority",
    )?;
    require_evidence(
        authority_source,
        "first_residual_outer_receipt",
        &first_residual,
        "source-token route authority",
    )?;
    if field_string(
        authority_source,
        "manifest_seal_sha256",
        "source-token route authority",
    )? != manifest_seal
        || field_string(
            authority_source,
            "admission_receipt_seal_sha256",
            "source-token route authority",
        )? != admission_receipt_seal
    {
        return Err("source-token route authority immutable artifact identity drifted".to_owned());
    }
    let source_plan = authority_object
        .get("source_token_plan")
        .cloned()
        .ok_or("source-token route authority lacks source_token_plan")?;
    let source_plan_object = object(&source_plan, "source-token plan")?;
    if field_string(source_plan_object, "schema", "source-token plan")? != SOURCE_PLAN_SCHEMA
        || field_string(source_plan_object, "status", "source-token plan")? != SOURCE_PLAN_STATUS
    {
        return Err("source-token plan schema/status drifted".to_owned());
    }
    let provenance = field_object(
        source_plan_object,
        "source_input_provenance",
        "source-token plan",
    )?;
    if provenance.get("source_token_id").and_then(Value::as_u64) != Some(SOURCE_TOKEN_ID as u64)
        || provenance
            .get("same_input_state_identity_required")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("source-token plan lost token-1/zero-state identity".to_owned());
    }
    let router = field_object(
        source_plan_object,
        "source_token_router_evidence",
        "source-token plan",
    )?;
    let route = route_from_object(
        router,
        "source_stable_route_ids",
        "source_stable_normalized_weights",
        "source-token plan",
    )?;
    if field_array(
        source_plan_object,
        "deterministic_waves",
        "source-token plan",
    )?
    .len()
        != ROUTES
    {
        return Err("source-token plan does not retain ten deterministic waves".to_owned());
    }

    let (_, prefix_value) = read_json(&first_residual.path, "first-residual antecedent")?;
    let first_residual_seal = verify_seal(&prefix_value, "first-residual antecedent")?;
    let prefix_object = object(&prefix_value, "first-residual antecedent")?;
    if field_string(prefix_object, "schema", "first-residual antecedent")? != PREFIX_SCHEMA
        || field_string(prefix_object, "status", "first-residual antecedent")? != PREFIX_STATUS
        || field_string(
            source,
            "first_residual_receipt_seal_sha256",
            "source-token outer preflight",
        )? != first_residual_seal
        || first_residual_seal != EXPECTED_FIRST_RESIDUAL_OUTER_SEAL_SHA256
    {
        return Err("first-residual antecedent schema/status/seal drifted".to_owned());
    }
    let prefix_source = field_object(prefix_object, "source_binding", "first-residual antecedent")?;
    require_evidence(
        prefix_source,
        "manifest",
        &manifest,
        "first-residual antecedent",
    )?;
    if field_string(
        prefix_source,
        "manifest_seal_sha256",
        "first-residual antecedent",
    )? != manifest_seal
        || field_string(
            prefix_source,
            "admission_receipt_seal_sha256",
            "first-residual antecedent",
        )? != admission_receipt_seal
    {
        return Err("first-residual antecedent artifact identity drifted".to_owned());
    }
    let output = field_object(
        prefix_object,
        "first_residual_output",
        "first-residual antecedent",
    )?;
    if output.get("layer").and_then(Value::as_u64) != Some(0)
        || output.get("linear_state_slot").and_then(Value::as_u64) != Some(0)
        || output.get("elements").and_then(Value::as_u64) != Some(HIDDEN as u64)
        || output
            .get("same_command_graph_required")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("first-residual antecedent geometry/state drifted".to_owned());
    }
    let first_residual_output_sha =
        field_string(output, "sha256", "first-residual antecedent")?.to_owned();
    require_sha256(&first_residual_output_sha, "first-residual output SHA")?;
    require_evidence(
        provenance,
        "prefix_outer_receipt",
        &first_residual,
        "source-token plan",
    )?;
    if field_string(
        provenance,
        "prefix_outer_receipt_seal_sha256",
        "source-token plan",
    )? != first_residual_seal
        || field_string(
            provenance,
            "strict_metal_prefix_first_residual_sha256",
            "source-token plan",
        )? != first_residual_output_sha
    {
        return Err("source-token plan prefix output/outer seal lineage drifted".to_owned());
    }
    let source_input_hidden_sha =
        field_string(provenance, "input_hidden_f32le_sha256", "source-token plan")?.to_owned();
    let source_cpu_first_residual_sha = field_string(
        provenance,
        "cpu_first_residual_f32le_sha256",
        "source-token plan",
    )?
    .to_owned();
    let source_zero_conv_state_sha = field_string(
        provenance,
        "zero_conv_state_f32le_sha256",
        "source-token plan",
    )?
    .to_owned();
    let source_zero_recurrent_state_sha = field_string(
        provenance,
        "zero_recurrent_state_f32le_sha256",
        "source-token plan",
    )?
    .to_owned();
    for (value, label) in [
        (&source_input_hidden_sha, "source-token input hidden SHA"),
        (
            &source_cpu_first_residual_sha,
            "source-token CPU first-residual SHA",
        ),
        (
            &source_zero_conv_state_sha,
            "source-token zero conv-state SHA",
        ),
        (
            &source_zero_recurrent_state_sha,
            "source-token zero recurrent-state SHA",
        ),
    ] {
        require_sha256(value, label)?;
    }
    let inner_evidence_row = field_object(
        prefix_object,
        "inner_probe_capture",
        "first-residual antecedent",
    )?;
    let inner_path = PathBuf::from(field_string(
        inner_evidence_row,
        "path",
        "first-residual inner capture",
    )?);
    let prefix_inner = file_evidence(&inner_path, "first-residual inner capture")?;
    if field_string(inner_evidence_row, "path", "first-residual inner capture")?
        != prefix_inner.path.to_string_lossy()
        || inner_evidence_row.get("bytes").and_then(Value::as_u64) != Some(prefix_inner.bytes)
        || field_string(inner_evidence_row, "sha256", "first-residual inner capture")?
            != prefix_inner.sha256
    {
        return Err("first-residual outer does not bind its immutable inner receipt".to_owned());
    }
    let (_, prefix_inner_value) = read_json(&prefix_inner.path, "first-residual inner capture")?;
    let prefix_inner_object = object(&prefix_inner_value, "first-residual inner capture")?;
    if field_string(prefix_inner_object, "schema", "first-residual inner capture")?
        != "hawking.ascension.qwen80_first_residual_bridge_device.v1"
        || field_string(prefix_inner_object, "status", "first-residual inner capture")?
            != "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_MIXER_FIRST_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
        || prefix_inner_object.get("metal_device_or_dispatch_performed").and_then(Value::as_bool) != Some(true)
        || prefix_inner_object.get("synthetic_input").and_then(Value::as_bool) != Some(false)
        || prefix_inner_object.get("fixture_only").and_then(Value::as_bool) != Some(false)
    {
        return Err("first-residual inner receipt is not the sealed source-input strict-Metal prefix".to_owned());
    }
    let prefix_inner_input = field_object(
        prefix_inner_object,
        "same_input_provenance",
        "first-residual inner capture",
    )?;
    let prefix_inner_output = field_object(
        prefix_inner_object,
        "first_residual_output",
        "first-residual inner capture",
    )?;
    let prefix_inner_state = field_object(
        prefix_inner_object,
        "state_witness",
        "first-residual inner capture",
    )?;
    if prefix_inner_input.get("token_id").and_then(Value::as_u64) != Some(SOURCE_TOKEN_ID as u64)
        || field_string(
            prefix_inner_input,
            "input_hidden_f32le_sha256",
            "first-residual inner capture",
        )? != source_input_hidden_sha
        || field_string(
            field_object(
                prefix_inner_input,
                "initial_conv_state",
                "first-residual inner capture",
            )?,
            "f32le_sha256",
            "first-residual inner capture",
        )? != source_zero_conv_state_sha
        || field_string(
            field_object(
                prefix_inner_input,
                "initial_recurrent_state",
                "first-residual inner capture",
            )?,
            "f32le_sha256",
            "first-residual inner capture",
        )? != source_zero_recurrent_state_sha
        || field_string(
            prefix_inner_output,
            "sha256",
            "first-residual inner capture",
        )? != first_residual_output_sha
        || field_string(
            prefix_inner_output,
            "cpu_reference_sha256",
            "first-residual inner capture",
        )? != source_cpu_first_residual_sha
        || prefix_inner_state
            .get("initial_conv_state_identity_matches_cpu_baseline")
            .and_then(Value::as_bool)
            != Some(true)
        || prefix_inner_state
            .get("initial_recurrent_state_identity_matches_cpu_baseline")
            .and_then(Value::as_bool)
            != Some(true)
        || prefix_inner_state
            .get("state_commit_after_parity_fence")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("source-token first-residual input/state lineage drifted".to_owned());
    }

    let (_, typed_value) = read_json(&typed_bridge.path, "source-token typed bridge")?;
    let typed_bridge_seal = verify_seal(&typed_value, "source-token typed bridge")?;
    let typed_object = object(&typed_value, "source-token typed bridge")?;
    if field_string(typed_object, "schema", "source-token typed bridge")? != SOURCE_BRIDGE_SCHEMA
        || field_string(typed_object, "status", "source-token typed bridge")?
            != SOURCE_BRIDGE_STATUS
        || field_string(
            source,
            "typed_bridge_receipt_seal_sha256",
            "source-token outer preflight",
        )? != typed_bridge_seal
        || typed_bridge_seal != EXPECTED_TYPED_BRIDGE_SEAL_SHA256
    {
        return Err("source-token typed bridge schema/status/seal drifted".to_owned());
    }
    let typed_source = field_object(typed_object, "source_binding", "source-token typed bridge")?;
    require_evidence(
        typed_source,
        "source_token_route_authority",
        &source_authority,
        "source-token typed bridge",
    )?;
    require_evidence(
        typed_source,
        "first_residual_receipt",
        &first_residual,
        "source-token typed bridge",
    )?;
    if field_string(
        typed_source,
        "source_token_route_authority_seal_sha256",
        "source-token typed bridge",
    )? != source_authority_seal
        || field_string(
            typed_source,
            "first_residual_receipt_seal_sha256",
            "source-token typed bridge",
        )? != first_residual_seal
    {
        return Err("source-token typed bridge lineage drifted".to_owned());
    }
    let typed_payload = field_object(typed_object, "typed_bridge", "source-token typed bridge")?;
    if typed_payload.get("source_token_id").and_then(Value::as_u64) != Some(SOURCE_TOKEN_ID as u64)
        || typed_payload
            .get("same_command_graph_required")
            .and_then(Value::as_bool)
            != Some(true)
        || field_string(
            typed_payload,
            "first_residual_output_sha256",
            "source-token typed bridge",
        )? != first_residual_output_sha
    {
        return Err("source-token typed bridge payload lineage drifted".to_owned());
    }
    let typed_route = field_object(typed_object, "route_authority", "source-token typed bridge")?;
    let typed_route = route_from_object(
        typed_route,
        "ids",
        "normalized_weights",
        "source-token typed bridge",
    )?;
    assert_same_route(&route, &typed_route, "source-token typed bridge")?;

    let (_, fixed_value) = read_json(&fixed_suffix.path, "source-token fixed suffix")?;
    if fixed_suffix.sha256 != EXPECTED_SOURCE_TOKEN_FIXED_SUFFIX_SHA256 {
        return Err(
            "source-token fixed suffix raw SHA is not the pinned static authority".to_owned(),
        );
    }
    validate_fixed_suffix(
        &fixed_value,
        &manifest,
        &manifest_seal,
        &admission_receipt_seal,
    )?;

    let provisional_authority = Authority {
        args: args.clone(),
        outer_preflight: outer_preflight.clone(),
        outer_preflight_seal: outer_seal.clone(),
        manifest: manifest.clone(),
        manifest_seal: manifest_seal.clone(),
        admission_current: admission_current.clone(),
        admission_pointer_seal: admission_pointer_seal.clone(),
        admission_receipt_seal: admission_receipt_seal.clone(),
        source_audit_seal: source_audit_seal.clone(),
        source_revision: source_revision.clone(),
        source_authority: source_authority.clone(),
        source_authority_seal: source_authority_seal.clone(),
        source_plan: source_plan.clone(),
        route_ids: route.0,
        route_weights: route.1,
        first_residual: first_residual.clone(),
        first_residual_seal: first_residual_seal.clone(),
        first_residual_output_sha: first_residual_output_sha.clone(),
        source_input_hidden_sha: source_input_hidden_sha.clone(),
        source_cpu_first_residual_sha: source_cpu_first_residual_sha.clone(),
        source_zero_conv_state_sha: source_zero_conv_state_sha.clone(),
        source_zero_recurrent_state_sha: source_zero_recurrent_state_sha.clone(),
        typed_bridge: typed_bridge.clone(),
        typed_bridge_seal: typed_bridge_seal.clone(),
        fixed_suffix: fixed_suffix.clone(),
        lease: None,
        lease_id: None,
        outer_launch: None,
    };
    let (lease, lease_id) = if args.mode == Mode::Metal {
        let path = args
            .lease_receipt
            .as_ref()
            .ok_or("metal mode lacks lease")?;
        let (evidence, value) = read_json(path, "source-token component lease")?;
        let (lease_seal, lease_id) = validate_component_lease(&value, &provisional_authority)?;
        (Some((evidence, lease_seal)), Some(lease_id))
    } else {
        (None, None)
    };
    let leased_authority = Authority {
        lease: lease.clone(),
        lease_id: lease_id.clone(),
        ..provisional_authority
    };
    let outer_launch = if args.mode == Mode::Metal {
        let path = args
            .outer_launch_authority
            .as_ref()
            .ok_or("metal mode lacks outer launch authority")?;
        let (evidence, value) = read_json(path, "source-token outer launch authority")?;
        let outer_launch_seal = validate_outer_launch_authority(&value, &leased_authority)?;
        Some((evidence, outer_launch_seal))
    } else {
        None
    };
    Ok(Authority {
        args: args.clone(),
        outer_preflight,
        outer_preflight_seal: outer_seal,
        manifest,
        manifest_seal,
        admission_current,
        admission_pointer_seal,
        admission_receipt_seal,
        source_audit_seal,
        source_revision,
        source_authority,
        source_authority_seal,
        source_plan,
        route_ids: route.0,
        route_weights: route.1,
        first_residual,
        first_residual_seal,
        first_residual_output_sha,
        source_input_hidden_sha,
        source_cpu_first_residual_sha,
        source_zero_conv_state_sha,
        source_zero_recurrent_state_sha,
        typed_bridge,
        typed_bridge_seal,
        fixed_suffix,
        lease,
        lease_id,
        outer_launch,
    })
}

/// Revalidate the sealed source-token/all-ten chain and build only the
/// direct-packed bridge against a runtime that was already admitted by the
/// caller. This is intentionally the narrow reusable boundary for a future
/// state-handoff successor: it does not create a Metal context, issue a
/// command buffer, dispatch, or accept a lease.
#[cfg(target_os = "macos")]
pub fn build_source_token_all_ten_bridge_from_outer_preflight(
    runtime: &Qwen80CompleteNativeRuntime,
    outer_preflight: &Path,
    workers: usize,
) -> Result<
    (
        Qwen80AllTenTrueMoeSourceBridge,
        Qwen80SourceTokenAllTenValidatedLineage,
    ),
    String,
> {
    if !(1..=4).contains(&workers) {
        return Err("source-token all-ten bridge workers must be in 1..=4".into());
    }
    let authority = validate_authority(&Args {
        outer_preflight: outer_preflight.to_path_buf(),
        mode: Mode::Preflight,
        lease_receipt: None,
        outer_launch_authority: None,
        outer_capture_dir: None,
        capture_dir: None,
        workers,
    })?;
    let catalog = runtime.catalog();
    let hybrid = catalog
        .complete_hybrid_decoder_plan(1)
        .map_err(|error| format!("source-token successor hybrid plan failed: {error}"))?;
    if hybrid.manifest_seal_sha256 != authority.manifest_seal
        || hybrid.source_revision != authority.source_revision
    {
        return Err(
            "source-token successor runtime catalog differs from its sealed all-ten preflight"
                .into(),
        );
    }
    let route_authority = Qwen80SourceTokenAllTenRoutedExpertPlanAuthority {
        manifest_document_sha256: &authority.manifest.sha256,
        plan_authority_document_sha256: &authority.source_authority.sha256,
        admission_receipt_seal_sha256: &authority.admission_receipt_seal,
        first_residual_outer_receipt_seal_sha256: &authority.first_residual_seal,
    };
    let route_plan = hybrid
        .bind_source_token_all_ten_routed_expert_plan(0, &route_authority, &authority.source_plan)
        .map_err(|error| format!("source-token successor all-ten plan rejected: {error}"))?;
    let bridge = catalog
        .build_all_ten_true_moe_source_bridge(
            &route_plan,
            catalog.first_residual_device_binding(0).map_err(|error| {
                format!("source-token successor first residual rejected: {error}")
            })?,
        )
        .map_err(|error| format!("source-token successor bridge rejected: {error}"))?;
    let embedding = catalog
        .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID)
        .map_err(|error| format!("source-token successor embedding oracle rejected: {error}"))?;
    let cpu_input =
        hawking_core::model::qwen80_complete_runtime::Qwen80CanonicalLinearLayerCpuInput::with_zero_state(
            embedding.hidden,
        );
    let input_hidden_sha = f32_sha(&cpu_input.hidden, "source-token successor input hidden")?;
    let zero_conv_state_sha = f32_sha(
        &cpu_input.state.conv_state,
        "source-token successor zero convolution state",
    )?;
    let zero_recurrent_state_sha = f32_sha(
        &cpu_input.state.recurrent_state,
        "source-token successor zero recurrent state",
    )?;
    if input_hidden_sha != authority.source_input_hidden_sha
        || zero_conv_state_sha != authority.source_zero_conv_state_sha
        || zero_recurrent_state_sha != authority.source_zero_recurrent_state_sha
    {
        return Err(
            "source-token successor CPU input/zero state differs from sealed all-ten lineage"
                .into(),
        );
    }
    let cpu = catalog
        .execute_first_linear_layer_cpu_moe_oracle(&cpu_input)
        .map_err(|error| format!("source-token successor L0 CPU oracle rejected: {error}"))?;
    let cpu_first_residual_sha = f32_sha(
        &cpu.mixer.mixer_residual_output,
        "source-token successor CPU first residual",
    )?;
    if cpu_first_residual_sha != authority.source_cpu_first_residual_sha
        || cpu.route.ids.map(u32::from) != authority.route_ids
    {
        return Err(
            "source-token successor first-residual/router lineage differs from sealed all-ten authority"
                .into(),
        );
    }
    for index in 0..ROUTES {
        if (cpu.route.weights[index] - authority.route_weights[index]).abs() > 1.0e-6 {
            return Err(format!(
                "source-token successor route weight {index} differs from sealed authority"
            ));
        }
    }
    Ok((
        bridge,
        Qwen80SourceTokenAllTenValidatedLineage {
            manifest_document_sha256: authority.manifest.sha256,
            manifest_seal_sha256: authority.manifest_seal,
            admission_receipt_seal_sha256: authority.admission_receipt_seal,
            source_revision: authority.source_revision,
            source_token_id: SOURCE_TOKEN_ID,
            route_ids: authority.route_ids,
            route_weights: authority.route_weights,
            source_input_hidden_f32le_sha256: authority.source_input_hidden_sha,
            source_zero_conv_state_f32le_sha256: authority.source_zero_conv_state_sha,
            source_zero_recurrent_state_f32le_sha256: authority.source_zero_recurrent_state_sha,
            source_cpu_first_residual_f32le_sha256: authority.source_cpu_first_residual_sha,
            source_first_residual_outer_seal_sha256: authority.first_residual_seal,
        },
    ))
}

fn preflight_document(authority: &Authority) -> Value {
    json!({
        "schema": CHILD_SCHEMA,
        "status": "PREPARED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_DEVICE_CHILD_NOT_LEASED_OR_EXECUTED",
        "mode": "preflight",
        "outer_preflight_binding": {"path": authority.outer_preflight.path, "document_sha256": authority.outer_preflight.sha256, "seal_sha256": authority.outer_preflight_seal},
        "admission_current_binding": {"path": authority.admission_current.path, "document_sha256": authority.admission_current.sha256, "pointer_seal_sha256": authority.admission_pointer_seal, "immutable_admission_receipt_seal_sha256": authority.admission_receipt_seal},
        "source_token_route_authority_binding": {"path": authority.source_authority.path, "document_sha256": authority.source_authority.sha256, "seal_sha256": authority.source_authority_seal, "route_ids": authority.route_ids, "normalized_weights": authority.route_weights},
        "typed_bridge_binding": {"path": authority.typed_bridge.path, "document_sha256": authority.typed_bridge.sha256, "seal_sha256": authority.typed_bridge_seal},
        "first_residual_antecedent": {"path": authority.first_residual.path, "document_sha256": authority.first_residual.sha256, "seal_sha256": authority.first_residual_seal, "output_sha256": authority.first_residual_output_sha},
        "fixed_suffix_contract_binding": {"path": authority.fixed_suffix.path, "document_sha256": authority.fixed_suffix.sha256, "schema": FIXED_SCHEMA, "status": FIXED_STATUS},
        "same_command_graph_contract": {"source_token_id": SOURCE_TOKEN_ID, "zero_l0_state_required": true, "prefix_dispatches": PREFIX_DISPATCHES, "suffix_dispatches": SUFFIX_DISPATCHES, "total_dispatches": TOTAL_DISPATCHES, "route_guard_required": true, "all_ten_route_shared_routed_sum_second_residual_readbacks_required": true},
        "claim_boundary": {"metal_device_or_dispatch_performed": false, "lease_issued": false, "legacy_fixture_router_or_plan_accepted": false, "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true}
    })
}

/// Refuse a nominally-successful device receipt unless it carries every
/// scalar/binding which the outer reaper validates.  This keeps the child
/// from returning zero with a receipt that cannot be promoted by its only
/// authorized parent.  The outer still checks live artifact identity and
/// exact source routes; this internal gate checks the child-side duplicate
/// fields cannot silently diverge.
fn validate_success_receipt_contract(receipt: &Value) -> Result<(), String> {
    let root = object(receipt, "source-token successful receipt")?;
    if root
        .get("metal_device_or_dispatch_performed")
        .and_then(Value::as_bool)
        != Some(true)
        || root.get("component_only").and_then(Value::as_bool) != Some(true)
        || root
            .get("complete_layer_or_token_performed")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("successful receipt lost its strict component-only boundary".to_owned());
    }

    let outer = field_object(
        root,
        "outer_launch_authority_binding",
        "source-token successful receipt",
    )?;
    for field in ["path", "document_sha256", "seal_sha256"] {
        let value = field_string(outer, field, "source-token successful outer launch binding")?;
        if field != "path" {
            require_sha256(
                value,
                &format!("source-token successful outer launch binding.{field}"),
            )?;
        }
    }
    let policy = field_object(
        root,
        "metal_execution_policy",
        "source-token successful receipt",
    )?;
    if policy.get("outer_launch_authority_binding") != Some(&Value::Object(outer.clone())) {
        return Err(
            "successful receipt outer launch authority differs between top-level and policy"
                .to_owned(),
        );
    }
    let durable = field_object(root, "durable_capture", "source-token successful receipt")?;
    if durable
        .get("receipt_written_last_is_completion_marker")
        .and_then(Value::as_bool)
        != Some(true)
        || durable
            .get("outer_reaped_capture_required")
            .and_then(Value::as_bool)
            != Some(true)
        || durable.get("replay_guarded").and_then(Value::as_bool) != Some(true)
    {
        return Err("successful receipt durable capture policy drifted".to_owned());
    }
    let reaper = field_object(
        durable,
        "outer_reaper_binding",
        "source-token successful receipt",
    )?;
    require_sha256(
        field_string(reaper, "lease_id", "source-token successful outer reaper")?,
        "source-token successful outer reaper lease ID",
    )?;
    if reaper.get("outer_launch_authority") != Some(&Value::Object(outer.clone())) {
        return Err(
            "successful receipt outer reaper authority differs from top-level authority".to_owned(),
        );
    }

    let graph = field_object(
        root,
        "same_command_graph",
        "source-token successful receipt",
    )?;
    if graph.get("source_token_id").and_then(Value::as_u64) != Some(SOURCE_TOKEN_ID as u64)
        || graph.get("prefix_dispatches").and_then(Value::as_u64) != Some(PREFIX_DISPATCHES as u64)
        || graph.get("suffix_dispatches").and_then(Value::as_u64) != Some(SUFFIX_DISPATCHES as u64)
        || graph.get("total_dispatches").and_then(Value::as_u64) != Some(TOTAL_DISPATCHES as u64)
        || graph
            .get("same_command_graph_required")
            .and_then(Value::as_bool)
            != Some(true)
        || graph
            .get("same_command_graph_retained")
            .and_then(Value::as_bool)
            != Some(true)
        || graph
            .get("command_buffer_fenced_once_after_prefix_and_suffix")
            .and_then(Value::as_bool)
            != Some(true)
        || graph
            .get("first_residual_matches_sealed_prefix_antecedent")
            .and_then(Value::as_bool)
            != Some(true)
        || graph
            .get("encoded_kernel_order_matches_expected")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "successful receipt 9+14 same-command-buffer scalar lineage drifted".to_owned(),
        );
    }
    let encoded = field_array(
        graph,
        "encoded_kernel_names",
        "source-token successful receipt",
    )?;
    let expected = field_array(
        graph,
        "expected_kernel_names",
        "source-token successful receipt",
    )?;
    if encoded.len() != TOTAL_DISPATCHES
        || expected.len() != TOTAL_DISPATCHES
        || encoded != expected
    {
        return Err("successful receipt structural kernel trace is absent or differs from the expected 9+14 order".to_owned());
    }
    if encoded
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .as_deref()
        != Some(expected_kernel_order().as_slice())
    {
        return Err(
            "successful receipt structural kernel trace does not match the source-pinned 9+14 ABI"
                .to_owned(),
        );
    }

    let phase = field_object(root, "execution_phase", "source-token successful receipt")?;
    for field in [
        "strict_artifact_admission_started",
        "strict_artifact_admission_succeeded",
        "metal_context_construction_attempted",
        "metal_context_constructed",
        "structural_kernel_trace_enabled",
        "command_commit_attempted",
        "command_fence_succeeded",
        "readback_started",
        "device_dispatch_may_have_occurred",
    ] {
        field_bool(
            phase,
            field,
            true,
            "source-token successful execution phase",
        )?;
    }
    if phase.get("dispatches_encoded").and_then(Value::as_u64) != Some(TOTAL_DISPATCHES as u64) {
        return Err("successful receipt execution phase dispatch count drifted".to_owned());
    }

    let guard = field_object(
        root,
        "route_guard_readback",
        "source-token successful receipt",
    )?;
    if guard.get("value").and_then(Value::as_u64) != Some(1)
        || guard.get("passed").and_then(Value::as_bool) != Some(true)
        || field_array(guard, "observed_ids", "source-token successful route guard")?.len()
            != ROUTES
        || field_array(guard, "expected_ids", "source-token successful route guard")?.len()
            != ROUTES
        || field_array(
            guard,
            "observed_weights",
            "source-token successful route guard",
        )?
        .len()
            != ROUTES
        || field_array(
            guard,
            "expected_weights",
            "source-token successful route guard",
        )?
        .len()
            != ROUTES
    {
        return Err("successful receipt route guard scalar/readback contract drifted".to_owned());
    }
    let guard_error = guard
        .get("weights_max_abs_error")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .ok_or("successful receipt route guard weight error is absent/non-finite")?;
    if !guard_error.is_finite() {
        return Err("successful receipt route guard weight error is non-finite".to_owned());
    }

    let parity = field_object(root, "readback_parity", "source-token successful receipt")?;
    let pairs = [
        ("postnorm_max_abs_error", "postnorm"),
        ("router_logits_max_abs_error", "router_logits"),
        ("shared_expert_max_abs_error", "shared_expert"),
        ("routed_sum_max_abs_error", "routed_sum"),
        ("second_residual_max_abs_error", "second_residual"),
    ];
    for (scalar, nested) in pairs {
        let scalar_value = parity
            .get(scalar)
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| format!("successful receipt {scalar} is absent/non-finite"))?;
        let nested_value = field_object(parity, nested, "source-token successful parity")?
            .get("max_abs_error")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| {
                format!("successful receipt {nested}.max_abs_error is absent/non-finite")
            })?;
        if scalar_value != nested_value {
            return Err(format!(
                "successful receipt {scalar} differs from {nested}.max_abs_error"
            ));
        }
    }
    let witnesses = field_array(
        parity,
        "all_ten_route_witnesses",
        "source-token successful receipt",
    )?;
    if parity
        .get("all_ten_route_witness_count")
        .and_then(Value::as_u64)
        != Some(ROUTES as u64)
        || witnesses.len() != ROUTES
    {
        return Err("successful receipt all-ten route witness count drifted".to_owned());
    }
    for (index, witness) in witnesses.iter().enumerate() {
        let witness = object(witness, "source-token successful route witness")?;
        if witness.get("wave_index").and_then(Value::as_u64) != Some(index as u64)
            || witness.get("elements").and_then(Value::as_u64) != Some(HIDDEN as u64)
        {
            return Err(format!(
                "successful receipt route witness {index} geometry drifted"
            ));
        }
        require_sha256(
            field_string(
                witness,
                "output_sha256",
                "source-token successful route witness",
            )?,
            "source-token successful route witness output SHA",
        )?;
        witness
            .get("max_abs_error")
            .and_then(Value::as_f64)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .ok_or_else(|| {
                format!("successful receipt route witness {index} parity is absent/non-finite")
            })?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn snapshot_f32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("{label} byte count overflow"))?;
    if buffer.length() < bytes as u64 {
        return Err(format!("{label} device buffer is too short"));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() })
}

#[cfg(target_os = "macos")]
fn snapshot_u32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<u32>, String> {
    let bytes = elements
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("{label} byte count overflow"))?;
    if buffer.length() < bytes as u64 {
        return Err(format!("{label} device buffer is too short"));
    }
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec() })
}

#[cfg(target_os = "macos")]
fn parity(expected: &[f32], observed: &[f32], label: &str, tolerance: f32) -> Result<f32, String> {
    if expected.len() != observed.len() {
        return Err(format!("{label} length drifted"));
    }
    let mut maximum = 0.0f32;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(format!("{label} is non-finite at {index}"));
        }
        maximum = maximum.max((expected - observed).abs());
    }
    if maximum > tolerance {
        return Err(format!("{label} parity failed: {maximum} > {tolerance}"));
    }
    Ok(maximum)
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
fn verify_metal_capture_paths(authority: &Authority) -> Result<(PathBuf, PathBuf), String> {
    let outer = authority
        .args
        .outer_capture_dir
        .as_ref()
        .ok_or("metal mode lacks outer capture dir")?;
    let capture = authority
        .args
        .capture_dir
        .as_ref()
        .ok_or("metal mode lacks capture dir")?;
    let metadata = fs::symlink_metadata(outer)
        .map_err(|error| format!("cannot stat outer capture dir: {error}"))?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err("outer capture dir must be a non-symlink directory".to_owned());
    }
    let outer = fs::canonicalize(outer)
        .map_err(|error| format!("cannot canonicalize outer capture dir: {error}"))?;
    if capture.exists()
        || capture
            .parent()
            .and_then(|parent| fs::canonicalize(parent).ok())
            .as_deref()
            != Some(outer.as_path())
    {
        return Err("capture dir must be a new direct child of outer capture dir".to_owned());
    }
    Ok((outer, capture.clone()))
}

#[cfg(target_os = "macos")]
fn write_new(capture: &Path, name: &str, bytes: &[u8]) -> Result<(), String> {
    let path = capture.join(name);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
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
    encoded_kernel_order_matches_expected: bool,
    command_commit_attempted: bool,
    command_fence_succeeded: bool,
    readback_started: bool,
}

#[cfg(target_os = "macos")]
fn run_metal(
    authority: &Authority,
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
        .map_err(|error| format!("strict artifact admission failed: {error}"))?;
    phase.strict_artifact_admission_succeeded = true;
    let hybrid = catalog
        .complete_hybrid_decoder_plan(1)
        .map_err(|error| format!("hybrid plan failed: {error}"))?;
    let route_authority = Qwen80SourceTokenAllTenRoutedExpertPlanAuthority {
        manifest_document_sha256: &authority.manifest.sha256,
        plan_authority_document_sha256: &authority.source_authority.sha256,
        admission_receipt_seal_sha256: &authority.admission_receipt_seal,
        first_residual_outer_receipt_seal_sha256: &authority.first_residual_seal,
    };
    let route_plan = hybrid
        .bind_source_token_all_ten_routed_expert_plan(0, &route_authority, &authority.source_plan)
        .map_err(|error| format!("source-token all-ten plan rejected: {error}"))?;
    let source_bridge = catalog
        .build_all_ten_true_moe_source_bridge(
            &route_plan,
            catalog
                .first_residual_device_binding(0)
                .map_err(|error| format!("first residual binding rejected: {error}"))?,
        )
        .map_err(|error| format!("source-token bridge rejected: {error}"))?;
    let embedding = catalog
        .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID)
        .map_err(|error| format!("source embedding oracle rejected: {error}"))?;
    let cpu_input = hawking_core::model::qwen80_complete_runtime::Qwen80CanonicalLinearLayerCpuInput::with_zero_state(embedding.hidden);
    let input_hidden_sha = f32_sha(&cpu_input.hidden, "source-token input hidden")?;
    let zero_conv_state_sha = f32_sha(
        &cpu_input.state.conv_state,
        "source-token zero convolution state",
    )?;
    let zero_recurrent_state_sha = f32_sha(
        &cpu_input.state.recurrent_state,
        "source-token zero recurrent state",
    )?;
    if input_hidden_sha != authority.source_input_hidden_sha
        || zero_conv_state_sha != authority.source_zero_conv_state_sha
        || zero_recurrent_state_sha != authority.source_zero_recurrent_state_sha
    {
        return Err(
            "source-token CPU input/zero-state hashes differ from the sealed prefix lineage"
                .to_owned(),
        );
    }
    let cpu = catalog
        .execute_first_linear_layer_cpu_moe_oracle(&cpu_input)
        .map_err(|error| format!("source-token L0 CPU oracle rejected: {error}"))?;
    let cpu_first_residual_sha = f32_sha(
        &cpu.mixer.mixer_residual_output,
        "source-token CPU first residual",
    )?;
    if cpu_first_residual_sha != authority.source_cpu_first_residual_sha {
        return Err(
            "source-token CPU first-residual hash differs from the sealed prefix lineage"
                .to_owned(),
        );
    }
    if cpu.route.ids.map(u32::from) != authority.route_ids {
        return Err("CPU router IDs differ from source-token authority".to_owned());
    }
    for index in 0..ROUTES {
        if (cpu.route.weights[index] - authority.route_weights[index]).abs() > 1.0e-6 {
            return Err(format!(
                "CPU router weight {index} differs from source-token authority"
            ));
        }
    }
    phase.metal_context_construction_attempted = true;
    let runtime = Qwen80CompleteNativeRuntime::from_admitted_catalog_strict_math(
        catalog,
        Qwen80CompleteRuntimeOptions {
            max_seq_len: 1,
            trace_dispatch: true,
        },
    )
    .map_err(|error| format!("strict-Math runtime construction failed: {error}"))?;
    phase.metal_context_constructed = true;
    let mut command = runtime.begin_component_token_command_buffer();
    command
        .enable_structural_kernel_trace()
        .map_err(|error| format!("non-timed structural trace refusal: {error}"))?;
    phase.structural_kernel_trace_enabled = true;
    let resources = source_prefix::encode_source_input_l0_true_moe_capture(
        &runtime,
        &mut command,
        SOURCE_TOKEN_ID,
        &source_bridge,
    )?;
    if resources.graph.source_token_id() != SOURCE_TOKEN_ID
        || resources.graph.prefix_dispatches != PREFIX_DISPATCHES
        || resources.graph.suffix_dispatches != SUFFIX_DISPATCHES
        || command.dispatch_count() != TOTAL_DISPATCHES
    {
        return Err("source-token 9+14 same-TCB dispatch lineage drifted".to_owned());
    }
    phase.dispatches_encoded = command.dispatch_count();
    phase.encoded_kernel_names = command
        .structural_kernel_names()
        .ok_or("source-token structural trace was not retained")?
        .to_vec();
    let expected_kernel_order = expected_kernel_order();
    phase.encoded_kernel_order_matches_expected = phase
        .encoded_kernel_names
        .iter()
        .map(String::as_str)
        .eq(expected_kernel_order.iter().copied());
    if !phase.encoded_kernel_order_matches_expected {
        return Err(format!(
            "source-token structural kernel order drifted: expected {:?}, observed {:?}",
            expected_kernel_order, phase.encoded_kernel_names
        ));
    }
    phase.command_commit_attempted = true;
    command
        .commit_and_wait()
        .map_err(|error| format!("source-token common fence failed: {error}"))?;
    phase.command_fence_succeeded = true;
    phase.readback_started = true;
    let prefix = resources
        .graph
        .verify_first_residual_after_fence(&runtime)
        .map_err(|error| format!("prefix parity failed: {error}"))?;
    if prefix.source_token_id != SOURCE_TOKEN_ID
        || prefix.input_f32le_sha256 != authority.source_input_hidden_sha
        || prefix.initial_conv_state_f32le_sha256 != authority.source_zero_conv_state_sha
        || prefix.initial_recurrent_state_f32le_sha256 != authority.source_zero_recurrent_state_sha
        || prefix.cpu_first_residual_f32le_sha256 != authority.source_cpu_first_residual_sha
        || prefix.device_first_residual_f32le_sha256 != authority.first_residual_output_sha
        || prefix.dispatches_encoded_before_suffix != PREFIX_DISPATCHES
        || !prefix.same_command_graph_required
    {
        return Err(
            "sealed prefix input/state/first-residual lineage differs after same-TCB suffix"
                .to_owned(),
        );
    }
    let fixed = &resources.fixed;
    let postnorm = snapshot_f32(&fixed.postnorm_hidden, HIDDEN, "postnorm")?;
    let logits = snapshot_f32(&fixed.router_logits, 512, "router logits")?;
    let ids = snapshot_u32(&fixed.router_route_ids, ROUTES, "router IDs")?;
    let weights = snapshot_f32(&fixed.router_route_weights, ROUTES, "router weights")?;
    let guard = snapshot_u32(&fixed.route_guard, 1, "route guard")?[0];
    if ids.as_slice() != authority.route_ids || guard != 1 {
        return Err("route guard/readback refused source-token route".to_owned());
    }
    let postnorm_error = parity(
        &cpu.post_attention_rms_norm_output,
        &postnorm,
        "postnorm",
        2.0e-4,
    )?;
    let logits_error = parity(&cpu.router_logits, &logits, "router logits", 5.0e-4)?;
    let weights_error = parity(&cpu.route.weights, &weights, "router weights", 2.0e-5)?;
    let weighted = snapshot_f32(&fixed.route_weighted, ROUTES * HIDDEN, "route weighted")?;
    let mut witnesses = Vec::with_capacity(ROUTES);
    for (index, expected) in cpu.routed_experts.iter().enumerate() {
        let observed = &weighted[index * HIDDEN..(index + 1) * HIDDEN];
        let error = parity(
            &expected.weighted_output,
            observed,
            &format!("route {index}"),
            3.0e-4,
        )?;
        witnesses.push(json!({"wave_index":index,"expert_id":expected.expert,"normalized_weight":expected.route_weight,"elements":HIDDEN,"max_abs_error":error,"output_sha256":f32_sha(observed, "route witness")?}));
    }
    let shared = snapshot_f32(&fixed.gated_shared, HIDDEN, "shared")?;
    let sum = snapshot_f32(&fixed.routed_sum, HIDDEN, "routed sum")?;
    let second = snapshot_f32(&fixed.second_residual, HIDDEN, "second residual")?;
    let shared_error = parity(&cpu.shared_gated_output, &shared, "shared", 3.0e-4)?;
    let sum_error = parity(&cpu.routed_expert_sum, &sum, "routed sum", 3.0e-5)?;
    let second_error = parity(&cpu.layer_output, &second, "second residual", 3.0e-5)?;
    let lease = authority
        .lease
        .as_ref()
        .ok_or("metal mode lacks validated lease")?;
    let outer_launch = authority
        .outer_launch
        .as_ref()
        .ok_or("metal mode lacks validated outer launch authority")?;
    let postnorm_sha = f32_sha(&postnorm, "postnorm output")?;
    let logits_sha = f32_sha(&logits, "router logits output")?;
    let shared_sha = f32_sha(&shared, "shared output")?;
    let sum_sha = f32_sha(&sum, "routed sum output")?;
    let second_sha = f32_sha(&second, "second residual output")?;
    let outer_launch_binding = json!({
        "path": outer_launch.0.path,
        "document_sha256": outer_launch.0.sha256,
        "seal_sha256": outer_launch.1,
    });
    let receipt = json!({
        "schema":CHILD_SCHEMA,"status":CHILD_STATUS,"mode":"metal","metal_device_or_dispatch_performed":true,"component_only":true,"complete_layer_or_token_performed":false,"complete_artifact_scan_performed_once":true,"raw_bf16_or_safetensors_opened":false,
        "artifact_binding":{"manifest_document_sha256":authority.manifest.sha256,"manifest_seal_sha256":authority.manifest_seal,"admission_pointer_seal_sha256":authority.admission_pointer_seal,"admission_receipt_seal_sha256":authority.admission_receipt_seal,"source_audit_seal_sha256":authority.source_audit_seal,"source_revision":authority.source_revision,"layer":0,"linear_state_slot":0,"native_device":runtime.device_name()},
        "outer_preflight_binding":{"path":authority.outer_preflight.path,"document_sha256":authority.outer_preflight.sha256,"seal_sha256":authority.outer_preflight_seal},
        "outer_launch_authority_binding":outer_launch_binding.clone(),
        "source_token_route_authority_binding":{"path":authority.source_authority.path,"document_sha256":authority.source_authority.sha256,"seal_sha256":authority.source_authority_seal,"route_ids":authority.route_ids,"normalized_weights":authority.route_weights},
        "typed_bridge_binding":{"path":authority.typed_bridge.path,"document_sha256":authority.typed_bridge.sha256,"seal_sha256":authority.typed_bridge_seal,"schema":SOURCE_BRIDGE_SCHEMA,"status":SOURCE_BRIDGE_STATUS},
        "first_residual_antecedent":{"path":authority.first_residual.path,"document_sha256":authority.first_residual.sha256,"seal_sha256":authority.first_residual_seal,"output_sha256":authority.first_residual_output_sha},
        "fixed_suffix_contract_binding":{"path":authority.fixed_suffix.path,"document_sha256":authority.fixed_suffix.sha256,"schema":FIXED_SCHEMA,"status":FIXED_STATUS},
        "same_command_graph":{"source_token_id":SOURCE_TOKEN_ID,"same_command_graph_required":true,"same_command_graph_retained":true,"prefix_dispatches":PREFIX_DISPATCHES,"suffix_dispatches":SUFFIX_DISPATCHES,"total_dispatches":TOTAL_DISPATCHES,"command_buffer_fenced_once_after_prefix_and_suffix":true,"first_residual_matches_sealed_prefix_antecedent":true,"structural_kernel_trace_non_timed":true,"encoded_kernel_names":phase.encoded_kernel_names,"expected_kernel_names":expected_kernel_order,"encoded_kernel_order_matches_expected":phase.encoded_kernel_order_matches_expected},
        "execution_phase":{"strict_artifact_admission_started":phase.strict_artifact_admission_started,"strict_artifact_admission_succeeded":phase.strict_artifact_admission_succeeded,"metal_context_construction_attempted":phase.metal_context_construction_attempted,"metal_context_constructed":phase.metal_context_constructed,"structural_kernel_trace_enabled":phase.structural_kernel_trace_enabled,"dispatches_encoded":phase.dispatches_encoded,"encoded_kernel_names":phase.encoded_kernel_names,"encoded_kernel_order_matches_expected":phase.encoded_kernel_order_matches_expected,"command_commit_attempted":phase.command_commit_attempted,"command_fence_succeeded":phase.command_fence_succeeded,"readback_started":phase.readback_started,"device_dispatch_may_have_occurred":phase.command_commit_attempted},
        "prefix_parity":{"source_token_id":prefix.source_token_id,"input_hidden_f32le_sha256":prefix.input_f32le_sha256,"initial_conv_state_f32le_sha256":prefix.initial_conv_state_f32le_sha256,"initial_recurrent_state_f32le_sha256":prefix.initial_recurrent_state_f32le_sha256,"cpu_first_residual_f32le_sha256":prefix.cpu_first_residual_f32le_sha256,"device_first_residual_f32le_sha256":prefix.device_first_residual_f32le_sha256,"first_residual_max_abs_error":prefix.first_residual_max_abs_error,"conv_state_max_abs_error":prefix.conv_state_max_abs_error,"recurrent_state_max_abs_error":prefix.recurrent_state_max_abs_error,"elements":prefix.first_residual_elements,"bytes":prefix.first_residual_bytes},
        "route_guard_readback":{"value":guard,"passed":true,"observed_ids":ids,"expected_ids":authority.route_ids,"observed_weights":weights,"expected_weights":authority.route_weights,"weights_max_abs_error":weights_error},
        "readback_parity":{"postnorm_max_abs_error":postnorm_error,"postnorm":{"elements":HIDDEN,"output_sha256":postnorm_sha,"max_abs_error":postnorm_error,"tolerance":2.0e-4},"router_logits_max_abs_error":logits_error,"router_logits":{"elements":512,"output_sha256":logits_sha,"max_abs_error":logits_error,"tolerance":5.0e-4},"all_ten_route_witness_count":witnesses.len(),"all_ten_route_witnesses":witnesses,"shared_expert_max_abs_error":shared_error,"shared_expert":{"elements":HIDDEN,"output_sha256":shared_sha,"max_abs_error":shared_error,"tolerance":3.0e-4},"routed_sum_max_abs_error":sum_error,"routed_sum":{"elements":HIDDEN,"output_sha256":sum_sha,"max_abs_error":sum_error,"tolerance":3.0e-5},"second_residual_max_abs_error":second_error,"second_residual":{"elements":HIDDEN,"output_sha256":second_sha,"max_abs_error":second_error,"tolerance":3.0e-5}},
        "metal_execution_policy":{"strict_math_required":true,"timing_or_benchmarking_allowed":false,"complete_layer_or_token_allowed":false,"tps_or_tg_claim_allowed":false,"lease_binding":{"path":lease.0.path,"document_sha256":lease.0.sha256,"seal_sha256":lease.1,"lease_id":authority.lease_id},"outer_launch_authority_binding":outer_launch_binding.clone()},
        "outer_reaper_binding":{"lease_id":authority.lease_id,"outer_launch_authority":outer_launch_binding.clone()},
        "durable_capture":{"capture_directory":capture,"receipt_written_last_is_completion_marker":true,"outer_reaped_capture_required":true,"outer_reaper_binding":{"lease_id":authority.lease_id,"outer_launch_authority":outer_launch_binding.clone()},"replay_guarded":true},
        "claim_boundary":{"source_token_l0_true_moe_component_only":true,"no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim":true,"no_watcher_or_server_started":true}
    });
    validate_success_receipt_contract(&receipt)?;
    Ok(receipt)
}

#[cfg(target_os = "macos")]
fn finalize_capture(
    authority: &Authority,
    capture: &Path,
    phase: &MetalExecutionPhase,
    outcome: Result<Value, String>,
) -> Result<(Value, Option<String>), String> {
    let (receipt, failure) = match outcome {
        Ok(value) => (value, None),
        Err(error) => {
            // A Metal command buffer can be committed and then fail before a
            // fence/readback receipt exists.  Do not turn that ambiguous
            // post-submit state into a false `false` device claim: `true`
            // means the fence confirmed the command buffer, while `null`
            // means the phase record below is the only safe statement.
            let confirmed_device_execution = phase.command_fence_succeeded.then_some(true);
            (
                json!({
                    "schema":CHILD_SCHEMA,
                    "status":"REFUSED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_STRICT_MATH_COMPONENT",
                    "mode":"metal",
                    "metal_device_or_dispatch_performed":confirmed_device_execution,
                    "component_only":true,
                    "complete_layer_or_token_performed":false,
                    "error":error,
                    "execution_phase":{
                        "strict_artifact_admission_started":phase.strict_artifact_admission_started,
                        "strict_artifact_admission_succeeded":phase.strict_artifact_admission_succeeded,
                        "metal_context_construction_attempted":phase.metal_context_construction_attempted,
                        "metal_context_constructed":phase.metal_context_constructed,
                        "structural_kernel_trace_enabled":phase.structural_kernel_trace_enabled,
                        "dispatches_encoded":phase.dispatches_encoded,
                        "encoded_kernel_names":phase.encoded_kernel_names,
                        "encoded_kernel_order_matches_expected":phase.encoded_kernel_order_matches_expected,
                        "command_commit_attempted":phase.command_commit_attempted,
                        "command_fence_succeeded":phase.command_fence_succeeded,
                        "readback_started":phase.readback_started,
                        "device_dispatch_may_have_occurred":phase.command_commit_attempted
                    },
                    "outer_reaper_binding":{
                        "lease_id":authority.lease_id,
                        "outer_launch_authority":authority.outer_launch.as_ref().map(|binding| json!({"path":binding.0.path,"document_sha256":binding.0.sha256,"seal_sha256":binding.1}))
                    },
                    "claim_boundary":{"no_successful_device_or_runtime_claim":true,"failure_receipt_is_phase_accurate_not_a_no_device_claim":true}
                }),
                Some(error),
            )
        }
    };
    let rendered = serde_json::to_vec_pretty(&receipt).map_err(|error| error.to_string())?;
    let mut stdout = rendered.clone();
    stdout.push(b'\n');
    write_new(capture, "stdout.jsonl", &stdout)?;
    write_new(
        capture,
        "stderr.log",
        failure
            .as_ref()
            .map_or_else(|| b"\n".to_vec(), |error| format!("{error}\n").into_bytes())
            .as_slice(),
    )?;
    write_new(capture, "receipt.json", &rendered)?;
    Ok((receipt, failure))
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    if arguments.is_empty() || arguments == ["--print-plan"] {
        println!("{}", serde_json::to_string_pretty(&json!({"schema":"hawking.ascension.qwen80_source_token_all_ten_true_moe_graph_device_plan.v1","status":"DESCRIBED_SOURCE_TOKEN_CHILD_INTERFACE_NOT_AUTHORITY_NOT_LEASED_OR_EXECUTED","source_token_only":true,"prefix_dispatches":PREFIX_DISPATCHES,"suffix_dispatches":SUFFIX_DISPATCHES,"total_dispatches":TOTAL_DISPATCHES,"claim_boundary":{"metal_device_or_dispatch_performed":false,"legacy_fixture_router_or_plan_accepted":false,"cannot_substitute_for_authority_preflight_or_device_receipt":true}})).unwrap());
        return;
    }
    let args = match parse_args(arguments) {
        Ok(args) => args,
        Err(error) => {
            eprintln!("source-token child argument refusal: {error}");
            process::exit(2);
        }
    };
    let authority = match validate_authority(&args) {
        Ok(authority) => authority,
        Err(error) => {
            eprintln!("source-token child authority refusal: {error}");
            process::exit(2);
        }
    };
    match authority.args.mode {
        Mode::Preflight => println!(
            "{}",
            serde_json::to_string_pretty(&preflight_document(&authority)).unwrap()
        ),
        Mode::Metal => {
            #[cfg(target_os = "macos")]
            {
                let (_, capture) = match verify_metal_capture_paths(&authority) {
                    Ok(paths) => paths,
                    Err(error) => {
                        eprintln!("source-token child capture refusal: {error}");
                        process::exit(2);
                    }
                };
                if let Err(error) = fs::create_dir(&capture) {
                    eprintln!(
                        "source-token child refuses non-exclusive capture directory: {error}"
                    );
                    process::exit(2);
                }
                let invocation = json!({"schema":CHILD_SCHEMA,"status":"STARTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_STRICT_MATH_COMPONENT","mode":"metal","outer_preflight":authority.outer_preflight.path,"workers":authority.args.workers,"metal_execution_policy":{"strict_math":true,"timing_or_benchmarking_allowed":false,"complete_layer_or_token_allowed":false,"tps_or_tg_claim_allowed":false}});
                if let Err(error) = write_new(
                    &capture,
                    "invocation.json",
                    &serde_json::to_vec_pretty(&invocation).unwrap(),
                ) {
                    eprintln!("source-token child invocation refusal: {error}");
                    process::exit(2);
                }
                let mut phase = MetalExecutionPhase::default();
                let outcome = run_metal(&authority, &capture, &mut phase);
                match finalize_capture(&authority, &capture, &phase, outcome) {
                    Ok((receipt, None)) => {
                        println!("{}", serde_json::to_string_pretty(&receipt).unwrap())
                    }
                    Ok((_receipt, Some(error))) => {
                        eprintln!("source-token child terminal refusal: {error}");
                        process::exit(2);
                    }
                    Err(error) => {
                        eprintln!("source-token child durable capture failure: {error}");
                        process::exit(2);
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                eprintln!("Qwen80 source-token Metal child requires macOS");
                process::exit(2);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authority_fixture() -> Authority {
        Authority {
            args: Args {
                outer_preflight: "/tmp/outer.json".into(),
                mode: Mode::Preflight,
                lease_receipt: None,
                outer_launch_authority: None,
                outer_capture_dir: None,
                capture_dir: None,
                workers: 1,
            },
            outer_preflight: BoundFile {
                path: "/tmp/outer.json".into(),
                bytes: 1,
                sha256: "a".repeat(64),
            },
            outer_preflight_seal: "b".repeat(64),
            manifest: BoundFile {
                path: "/tmp/manifest.json".into(),
                bytes: 1,
                sha256: "c".repeat(64),
            },
            manifest_seal: "d".repeat(64),
            admission_current: BoundFile {
                path: "/tmp/admission.json".into(),
                bytes: 1,
                sha256: "e".repeat(64),
            },
            admission_pointer_seal: "f".repeat(64),
            admission_receipt_seal: "1".repeat(64),
            source_audit_seal: "2".repeat(64),
            source_revision: "revision".into(),
            source_authority: BoundFile {
                path: "/tmp/authority.json".into(),
                bytes: 1,
                sha256: "3".repeat(64),
            },
            source_authority_seal: "4".repeat(64),
            source_plan: json!({}),
            route_ids: [423, 463, 367, 451, 379, 2, 444, 237, 198, 328],
            route_weights: [0.1; ROUTES],
            first_residual: BoundFile {
                path: "/tmp/prefix.json".into(),
                bytes: 1,
                sha256: "5".repeat(64),
            },
            first_residual_seal: "6".repeat(64),
            first_residual_output_sha: "7".repeat(64),
            source_input_hidden_sha: "b".repeat(64),
            source_cpu_first_residual_sha: "c".repeat(64),
            source_zero_conv_state_sha: "d".repeat(64),
            source_zero_recurrent_state_sha: "e".repeat(64),
            typed_bridge: BoundFile {
                path: "/tmp/bridge.json".into(),
                bytes: 1,
                sha256: "8".repeat(64),
            },
            typed_bridge_seal: "9".repeat(64),
            fixed_suffix: BoundFile {
                path: "/tmp/fixed.json".into(),
                bytes: 1,
                sha256: "a".repeat(64),
            },
            lease: None,
            lease_id: None,
            outer_launch: None,
        }
    }

    fn sealed_test_document(mut value: Value) -> Value {
        let object = value.as_object_mut().expect("test document must be object");
        let canonical =
            canonical_json(&Value::Object(object.clone())).expect("test canonical JSON");
        object.insert(
            "seal_sha256".to_owned(),
            Value::String(sha256_hex(
                &serde_json::to_vec(&canonical).expect("test JSON"),
            )),
        );
        value
    }

    fn successful_receipt_fixture() -> Value {
        let outer = json!({
            "path": "/tmp/outer-launch-authority.json",
            "document_sha256": "a".repeat(64),
            "seal_sha256": "b".repeat(64),
        });
        let witness = |index: usize| {
            json!({
                "wave_index": index,
                "expert_id": index,
                "normalized_weight": 0.1,
                "elements": HIDDEN,
                "max_abs_error": 0.0,
                "output_sha256": "c".repeat(64),
            })
        };
        let witnesses = (0..ROUTES).map(witness).collect::<Vec<_>>();
        json!({
            "metal_device_or_dispatch_performed": true,
            "component_only": true,
            "complete_layer_or_token_performed": false,
            "outer_launch_authority_binding": outer.clone(),
            "same_command_graph": {
                "source_token_id": SOURCE_TOKEN_ID,
                "same_command_graph_required": true,
                "same_command_graph_retained": true,
                "prefix_dispatches": PREFIX_DISPATCHES,
                "suffix_dispatches": SUFFIX_DISPATCHES,
                "total_dispatches": TOTAL_DISPATCHES,
                "command_buffer_fenced_once_after_prefix_and_suffix": true,
                "first_residual_matches_sealed_prefix_antecedent": true,
                "encoded_kernel_order_matches_expected": true,
                "encoded_kernel_names": expected_kernel_order(),
                "expected_kernel_names": expected_kernel_order(),
            },
            "execution_phase": {
                "strict_artifact_admission_started": true,
                "strict_artifact_admission_succeeded": true,
                "metal_context_construction_attempted": true,
                "metal_context_constructed": true,
                "structural_kernel_trace_enabled": true,
                "dispatches_encoded": TOTAL_DISPATCHES,
                "command_commit_attempted": true,
                "command_fence_succeeded": true,
                "readback_started": true,
                "device_dispatch_may_have_occurred": true,
            },
            "route_guard_readback": {
                "value": 1,
                "passed": true,
                "observed_ids": (0..ROUTES).collect::<Vec<_>>(),
                "expected_ids": (0..ROUTES).collect::<Vec<_>>(),
                "observed_weights": vec![0.1; ROUTES],
                "expected_weights": vec![0.1; ROUTES],
                "weights_max_abs_error": 0.0,
            },
            "readback_parity": {
                "postnorm_max_abs_error": 0.0,
                "postnorm": {"max_abs_error": 0.0},
                "router_logits_max_abs_error": 0.0,
                "router_logits": {"max_abs_error": 0.0},
                "all_ten_route_witness_count": ROUTES,
                "all_ten_route_witnesses": witnesses,
                "shared_expert_max_abs_error": 0.0,
                "shared_expert": {"max_abs_error": 0.0},
                "routed_sum_max_abs_error": 0.0,
                "routed_sum": {"max_abs_error": 0.0},
                "second_residual_max_abs_error": 0.0,
                "second_residual": {"max_abs_error": 0.0},
            },
            "metal_execution_policy": {"outer_launch_authority_binding": outer.clone()},
            "outer_reaper_binding": {"lease_id": "d".repeat(64), "outer_launch_authority": outer.clone()},
            "durable_capture": {
                "receipt_written_last_is_completion_marker": true,
                "outer_reaped_capture_required": true,
                "replay_guarded": true,
                "outer_reaper_binding": {"lease_id": "d".repeat(64), "outer_launch_authority": outer},
            },
        })
    }

    #[test]
    fn parser_rejects_legacy_router_and_plan_flags() {
        let error = parse_args(vec![
            "--outer-preflight".into(),
            "/tmp/preflight.json".into(),
            "--mode".into(),
            "preflight".into(),
            "--workers".into(),
            "1".into(),
            "--router-receipt".into(),
            "/tmp/router.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("unsupported argument"));
        let error = parse_args(vec![
            "--outer-preflight".into(),
            "/tmp/preflight.json".into(),
            "--mode".into(),
            "preflight".into(),
            "--workers".into(),
            "1".into(),
            "--route-plan".into(),
            "/tmp/plan.json".into(),
        ])
        .unwrap_err();
        assert!(error.contains("unsupported argument"));
    }

    #[test]
    fn parser_requires_lease_only_for_metal_mode() {
        assert!(parse_args(vec![
            "--outer-preflight".into(),
            "/tmp/preflight.json".into(),
            "--mode".into(),
            "preflight".into(),
            "--workers".into(),
            "1".into()
        ])
        .is_ok());
        let error = parse_args(vec![
            "--outer-preflight".into(),
            "/tmp/preflight.json".into(),
            "--mode".into(),
            "metal".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("requires --lease-receipt"));
        let error = parse_args(vec![
            "--outer-preflight".into(),
            "/tmp/preflight.json".into(),
            "--mode".into(),
            "metal".into(),
            "--lease-receipt".into(),
            "/tmp/lease.json".into(),
            "--outer-capture-dir".into(),
            "/tmp/outer".into(),
            "--capture-dir".into(),
            "/tmp/outer/inner".into(),
            "--workers".into(),
            "1".into(),
        ])
        .unwrap_err();
        assert!(error.contains("--outer-launch-authority"));
    }

    #[test]
    fn exact_structural_trace_contract_is_nine_prefix_plus_fourteen_suffix() {
        let expected = expected_kernel_order();
        assert_eq!(expected.len(), TOTAL_DISPATCHES);
        assert_eq!(&expected[..PREFIX_DISPATCHES], PREFIX_KERNELS);
        assert_eq!(&expected[PREFIX_DISPATCHES..], SUFFIX_KERNELS);
        assert_eq!(expected[0], "qwen_next_direct_packed_input_rmsnorm");
        assert_eq!(expected[PREFIX_DISPATCHES - 1], "qwen_next_add_residual");
        assert_eq!(
            expected[PREFIX_DISPATCHES],
            "qwen80_postnorm_router_top10_rmsnorm"
        );
        assert_eq!(
            expected[TOTAL_DISPATCHES - 1],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
    }

    #[test]
    fn route_identity_rejects_fixture_or_weight_substitution() {
        let source = (
            [423, 463, 367, 451, 379, 2, 444, 237, 198, 328],
            [0.1; ROUTES],
        );
        let fixture = (
            [65, 245, 227, 35, 189, 440, 298, 405, 109, 494],
            [0.1; ROUTES],
        );
        assert!(assert_same_route(&source, &fixture, "fixture").is_err());
        let mut drift = source;
        drift.1[0] += 1.0e-4;
        assert!(assert_same_route(&source, &drift, "weight").is_err());
    }

    #[test]
    fn staged_contract_retains_exact_nine_plus_fourteen_boundary() {
        let document = preflight_document(&authority_fixture());
        assert_eq!(
            document["same_command_graph_contract"]["prefix_dispatches"],
            PREFIX_DISPATCHES
        );
        assert_eq!(
            document["same_command_graph_contract"]["suffix_dispatches"],
            SUFFIX_DISPATCHES
        );
        assert_eq!(
            document["same_command_graph_contract"]["total_dispatches"],
            TOTAL_DISPATCHES
        );
        assert_eq!(
            document["claim_boundary"]["legacy_fixture_router_or_plan_accepted"],
            false
        );
    }

    #[test]
    fn generic_or_legacy_lease_cannot_authorize_source_token_child() {
        let authority = authority_fixture();
        let lease = sealed_test_document(json!({
            "schema": LEASE_SCHEMA,
            "status": LEASE_STATUS,
            "lease_id": "0".repeat(64),
            "artifact_binding": {
                "manifest_document_sha256": authority.manifest.sha256,
                "manifest_seal_sha256": authority.manifest_seal,
                "admission_receipt_seal_sha256": authority.admission_receipt_seal
            }
        }));
        let error = validate_component_lease(&lease, &authority).unwrap_err();
        assert!(error.contains("outer_preflight_binding"), "{error}");
    }

    #[test]
    fn success_receipt_contract_requires_outer_authority_and_outer_scalar_mirrors() {
        let receipt = successful_receipt_fixture();
        validate_success_receipt_contract(&receipt)
            .expect("fixture must satisfy outer-facing receipt contract");

        let mut missing_authority = receipt.clone();
        missing_authority
            .as_object_mut()
            .expect("receipt object")
            .remove("outer_launch_authority_binding");
        assert!(validate_success_receipt_contract(&missing_authority)
            .unwrap_err()
            .contains("outer_launch_authority_binding"));

        let mut mismatched_scalar = receipt;
        mismatched_scalar["readback_parity"]["routed_sum_max_abs_error"] = json!(0.25);
        assert!(validate_success_receipt_contract(&mismatched_scalar)
            .unwrap_err()
            .contains("routed_sum_max_abs_error differs"));
    }
}
