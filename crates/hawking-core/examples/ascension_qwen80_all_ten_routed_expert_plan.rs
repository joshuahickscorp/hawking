//! Fail-closed Qwen3-Coder-Next layer-0 all-ten routed-expert binding plan.
//!
//! This is intentionally a CPU-only provenance compiler, not an MoE executor.
//! It reads three already-sealed / descriptor-only inputs:
//!
//! 1. the final outer terminal router receipt;
//! 2. the corresponding inner router receipt; and
//! 3. the admitted manifest's descriptor inventory.
//!
//! It then binds the exact source-stable route order
//! `[65,245,227,35,189,440,298,405,109,494]`, its normalized weights, and
//! one direct-packed gate/up/down triplet for every route.  The resulting JSON
//! is the machine-readable input to a future *real* all-ten provenance gate.
//!
//! The plan never opens an `.hq30g` payload, a source weight, a Metal context,
//! a server, a watcher, or an HCLI endpoint.  It does not execute an expert,
//! materialize a route delta, apply a route weight, aggregate routes, invoke a
//! shared expert, or combine a residual.  It is therefore not a layer, token,
//! decoder, generation, HCLI, TPS, TG, or tournament claim.
//!
//! Example:
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_all_ten_routed_expert_plan -- \
//!   --router-outer /absolute/path/outer-terminal-receipt.json \
//!   --router-inner /absolute/path/inner/receipt.json \
//!   --manifest-inventory /absolute/path/QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json \
//!   --out /absolute/path/QWEN80_ALL_TEN_ROUTED_EXPERT_BINDING_PLAN.json
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1";
const PROVENANCE_GATE_SCHEMA: &str =
    "hawking.ascension.qwen80_real_all_ten_routed_expert_provenance_gate_input.v1";
const OUTER_ROUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1";
const OUTER_ROUTER_STATUS: &str =
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY";
const INNER_ROUTER_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const INNER_ROUTER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const ROUTER_OUTER_SEAL: &str = "1bd9af3b6baf4d583f77bbfc02c25f0f1740cd8110c74fb6e55520758256c835";
const ROUTER_INNER_SHA256: &str =
    "31d2e06aae10695d25f445ff50661f6c63ec42c74f617e637b62e56c4f7ac343";
const LAYER: usize = 0;
const EXPERT_COUNT: u16 = 512;
const TOP_K: usize = 10;
const HIDDEN: usize = 2_048;
const INTERMEDIATE: usize = 512;
const GROUP_SIZE: usize = 128;
const COMPLETE_TENSOR_COUNT: usize = 74_391;
const SOURCE_SHARD: &str = "model-00001-of-00040.safetensors";
const SOURCE_SHARD_SHA256: &str =
    "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a";
const EXPECTED_ROUTE_IDS: [u16; TOP_K] = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494];
const EXPECTED_ROUTE_WEIGHTS: [f64; TOP_K] = [
    0.245_458_886_027_336_12,
    0.119_394_913_315_773_01,
    0.098_652_511_835_098_27,
    0.098_244_741_559_028_63,
    0.081_222_802_400_588_99,
    0.078_011_848_032_474_52,
    0.073_711_447_417_736_05,
    0.071_626_946_330_070_5,
    0.069_213_777_780_532_84,
    0.064_462_073_147_296_9,
];

#[derive(Clone, Debug)]
struct Arguments {
    router_outer: PathBuf,
    router_inner: PathBuf,
    manifest_inventory: PathBuf,
    out: PathBuf,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestSource {
    repository: String,
    tensor_count: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestInventory {
    schema: String,
    status: String,
    seal_sha256: String,
    source: ManifestSource,
    tensors: Vec<TensorDescriptor>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct PackedLayout {
    magic: String,
    group_size: usize,
    scale_dtype: String,
    sign_bit_order: String,
    version: u32,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct TensorDescriptor {
    tensor_name: String,
    shape: Vec<usize>,
    elements: usize,
    artifact_path: String,
    artifact_bytes: u64,
    artifact_sha256: String,
    source_dtype: String,
    source_shard: String,
    source_shard_sha256: String,
    layout: PackedLayout,
}

#[derive(Clone, Debug)]
struct RouterEvidence {
    outer_document_sha256: String,
    outer_seal_sha256: String,
    inner_document_sha256: String,
    ids: [u16; TOP_K],
    normalized_weights: [f64; TOP_K],
}

#[derive(Clone, Copy, Debug)]
enum Projection {
    Gate,
    Up,
    Down,
}

impl Projection {
    fn suffix(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    fn expected_shape(self) -> [usize; 2] {
        match self {
            Self::Gate | Self::Up => [INTERMEDIATE, HIDDEN],
            Self::Down => [HIDDEN, INTERMEDIATE],
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct ProjectionBinding {
    tensor_name: String,
    shape: [usize; 2],
    elements: usize,
    artifact_path: String,
    artifact_bytes: u64,
    artifact_sha256: String,
    source_dtype: String,
    source_shard: String,
    source_shard_sha256: String,
    layout: PackedLayout,
    payload_opened_by_this_plan: bool,
}

#[derive(Clone, Debug, Serialize)]
struct WavePlan {
    wave_index: usize,
    layer: usize,
    expert_id: u16,
    normalized_weight: f64,
    normalized_weight_bits_hex: String,
    gate: ProjectionBinding,
    up: ProjectionBinding,
    down: ProjectionBinding,
    fixed_operation_order: [&'static str; 5],
    route_execution_status: &'static str,
    route_delta_materialized: bool,
    route_weight_applied: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RouterEvidenceBinding {
    outer_receipt_document_sha256: String,
    outer_receipt_seal_sha256: String,
    inner_receipt_document_sha256: String,
    source_stable_route_ids: [u16; TOP_K],
    source_stable_normalized_weights: [f64; TOP_K],
    source_router_component_only: bool,
}

#[derive(Clone, Debug, Serialize)]
struct ManifestDescriptorBinding {
    inventory_document_sha256: String,
    manifest_schema: String,
    manifest_status: String,
    manifest_seal_sha256: String,
    source_repository: String,
    declared_tensor_count: usize,
    received_descriptor_count: usize,
    resolved_route_tensor_count: usize,
    payload_opened_by_this_plan: bool,
}

#[derive(Clone, Debug, Serialize)]
struct RealAllTenProvenanceGate {
    schema: &'static str,
    all_ten_source_bindings_complete: bool,
    expected_layer: usize,
    deterministic_wave_indices: [usize; TOP_K],
    route_order: [u16; TOP_K],
    normalized_weights: [f64; TOP_K],
    execution_receipt_required_for_each_wave: bool,
    direct_packed_execution_required_for_each_wave: bool,
    source_bound_input_required_for_each_wave: bool,
    route_combine_receipt_required_separately: bool,
    shared_expert_receipt_required_separately: bool,
    first_and_second_residual_receipts_required_separately: bool,
    rejects_tensor_substitution: bool,
    rejects_route_reorder: bool,
    rejects_duplicate_experts: bool,
    rejects_missing_tensor_or_weight: bool,
}

#[derive(Clone, Debug, Serialize)]
struct PlanReport {
    schema: &'static str,
    status: &'static str,
    model_id: &'static str,
    model_key: &'static str,
    source_repository: &'static str,
    source_revision: &'static str,
    layer: usize,
    router_evidence: RouterEvidenceBinding,
    manifest_descriptor_inventory: ManifestDescriptorBinding,
    deterministic_waves: Vec<WavePlan>,
    rawls_real_all_ten_provenance_gate: RealAllTenProvenanceGate,
    route_execution_performed: bool,
    route_combine_performed: bool,
    shared_expert_performed: bool,
    residual_combine_performed: bool,
    metal_device_or_dispatch_performed: bool,
    model_execution_performed: bool,
    hcli_execution_performed: bool,
    tps_or_tg_measurement_performed: bool,
    complete_layer_or_decoder_claim_earned: bool,
    claim_boundary: [&'static str; 4],
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

fn regular_file_bytes(path: &Path, label: &str) -> Result<Vec<u8>, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    fs::read(path).map_err(|error| format!("{label} read failed at {}: {error}", path.display()))
}

fn json_value(bytes: &[u8], label: &str) -> Result<Value, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("{label} invalid JSON: {error}"))
}

fn object_field<'a>(value: &'a Value, name: &str, label: &str) -> Result<&'a Value, String> {
    value
        .as_object()
        .and_then(|object| object.get(name))
        .ok_or_else(|| format!("{label} missing field {name:?}"))
}

fn string_field<'a>(value: &'a Value, name: &str, label: &str) -> Result<&'a str, String> {
    object_field(value, name, label)?
        .as_str()
        .ok_or_else(|| format!("{label} field {name:?} must be a string"))
}

fn bool_field(value: &Value, name: &str, label: &str) -> Result<bool, String> {
    object_field(value, name, label)?
        .as_bool()
        .ok_or_else(|| format!("{label} field {name:?} must be a boolean"))
}

fn array_field<'a>(value: &'a Value, name: &str, label: &str) -> Result<&'a [Value], String> {
    object_field(value, name, label)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} field {name:?} must be an array"))
}

fn nested<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a Value, String> {
    let mut current = value;
    for field in path {
        current = object_field(current, field, label)?;
    }
    Ok(current)
}

fn nested_string<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a str, String> {
    nested(value, path, label)?
        .as_str()
        .ok_or_else(|| format!("{label} field {:?} must be a string", path.join(".")))
}

fn nested_bool(value: &Value, path: &[&str], label: &str) -> Result<bool, String> {
    nested(value, path, label)?
        .as_bool()
        .ok_or_else(|| format!("{label} field {:?} must be a boolean", path.join(".")))
}

fn nested_u64(value: &Value, path: &[&str], label: &str) -> Result<u64, String> {
    nested(value, path, label)?.as_u64().ok_or_else(|| {
        format!(
            "{label} field {:?} must be an unsigned integer",
            path.join(".")
        )
    })
}

fn require_exact(observed: &str, expected: &str, label: &str) -> Result<(), String> {
    if observed == expected {
        Ok(())
    } else {
        Err(format!(
            "{label} drifted: expected {expected:?}, observed {observed:?}"
        ))
    }
}

fn parse_route_ids(values: &[Value], label: &str) -> Result<[u16; TOP_K], String> {
    if values.len() != TOP_K {
        return Err(format!(
            "{label} must contain exactly {TOP_K} route IDs, found {}",
            values.len()
        ));
    }
    let ids = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let id = value
                .as_u64()
                .ok_or_else(|| format!("{label}[{index}] must be an unsigned route ID"))?;
            if id >= u64::from(EXPERT_COUNT) {
                return Err(format!(
                    "{label}[{index}]={id} is outside Qwen80's {EXPERT_COUNT} experts"
                ));
            }
            u16::try_from(id).map_err(|_| format!("{label}[{index}] does not fit u16"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    ids.try_into()
        .map_err(|_| format!("{label} did not preserve {TOP_K} entries"))
}

fn parse_normalized_weights(values: &[Value], label: &str) -> Result<[f64; TOP_K], String> {
    if values.len() != TOP_K {
        return Err(format!(
            "{label} must contain exactly {TOP_K} normalized weights, found {}",
            values.len()
        ));
    }
    let weights = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let weight = value
                .as_f64()
                .ok_or_else(|| format!("{label}[{index}] must be a finite number"))?;
            if !weight.is_finite() || weight <= 0.0 {
                return Err(format!(
                    "{label}[{index}] must be finite and positive, observed {weight:?}"
                ));
            }
            Ok(weight)
        })
        .collect::<Result<Vec<_>, String>>()?;
    weights
        .try_into()
        .map_err(|_| format!("{label} did not preserve {TOP_K} entries"))
}

fn validate_route(ids: [u16; TOP_K], weights: [f64; TOP_K]) -> Result<(), String> {
    if ids != EXPECTED_ROUTE_IDS {
        return Err(format!(
            "source-stable route order drifted: expected {:?}, observed {:?}",
            EXPECTED_ROUTE_IDS, ids
        ));
    }
    if ids.iter().collect::<BTreeSet<_>>().len() != TOP_K {
        return Err("source-stable route contains duplicate expert IDs".into());
    }
    for (index, (observed, expected)) in weights
        .iter()
        .zip(EXPECTED_ROUTE_WEIGHTS.iter())
        .enumerate()
    {
        if observed.to_bits() != expected.to_bits() {
            return Err(format!(
                "normalized weight at route index {index} drifted: expected {expected:.17}, observed {observed:.17}"
            ));
        }
    }
    let sum: f64 = weights.iter().sum();
    // The sealed source receipt records normalized `f32` values as JSON
    // binary64 decimals.  Preserve every route slot exactly, then allow only
    // the expected f32 accumulation roundoff when checking their sum.
    if (sum - 1.0).abs() > 1.0e-6 {
        return Err(format!(
            "normalized source route weights must sum to one, observed {sum:.17}"
        ));
    }
    Ok(())
}

fn validate_outer_receipt(outer: &Value, inner_sha256: &str) -> Result<String, String> {
    let label = "sealed Qwen80 router outer receipt";
    require_exact(
        string_field(outer, "schema", label)?,
        OUTER_ROUTER_SCHEMA,
        label,
    )?;
    require_exact(
        string_field(outer, "status", label)?,
        OUTER_ROUTER_STATUS,
        label,
    )?;
    let seal = string_field(outer, "seal_sha256", label)?;
    if !is_lower_sha256(seal) {
        return Err(format!("{label} has malformed seal_sha256"));
    }
    require_exact(seal, ROUTER_OUTER_SEAL, "router outer receipt seal")?;
    require_exact(
        nested_string(outer, &["inner_probe_capture", "schema"], label)?,
        INNER_ROUTER_SCHEMA,
        "outer->inner router schema",
    )?;
    require_exact(
        nested_string(outer, &["inner_probe_capture", "status"], label)?,
        INNER_ROUTER_STATUS,
        "outer->inner router status",
    )?;
    require_exact(
        nested_string(outer, &["inner_probe_capture", "sha256"], label)?,
        inner_sha256,
        "outer->inner receipt SHA-256",
    )?;
    if nested_u64(outer, &["inner_probe_capture", "bytes"], label)? == 0 {
        return Err("outer receipt records an empty inner router receipt".into());
    }
    if !nested_bool(outer, &["inner_probe_capture", "metal_performed"], label)? {
        return Err("outer receipt does not bind the sealed source router Metal component".into());
    }
    if nested_u64(outer, &["child", "terminal", "exit_code"], label)? != 0
        || !nested_bool(outer, &["child", "terminal", "reaped"], label)?
        || !nested_bool(outer, &["one_shot", "automatic_retry_disabled"], label)?
    {
        return Err("outer router receipt is not a reaped one-shot terminal capture".into());
    }
    if !nested_bool(
        outer,
        &["claim_boundary", "outer_terminal_capture_only"],
        label,
    )? || !nested_bool(
        outer,
        &[
            "claim_boundary",
            "does_not_execute_a_complete_layer_or_decoder",
        ],
        label,
    )? {
        return Err("outer router receipt claim boundary drifted".into());
    }
    require_exact(
        nested_string(outer, &["source_binding", "manifest", "sha256"], label)?,
        MANIFEST_DOCUMENT_SHA256,
        "outer router manifest document SHA-256",
    )?;
    Ok(seal.to_owned())
}

fn validate_inner_receipt(
    inner: &Value,
    inner_sha256: &str,
) -> Result<([u16; TOP_K], [f64; TOP_K]), String> {
    let label = "sealed Qwen80 router inner receipt";
    require_exact(
        string_field(inner, "schema", label)?,
        INNER_ROUTER_SCHEMA,
        label,
    )?;
    require_exact(
        string_field(inner, "status", label)?,
        INNER_ROUTER_STATUS,
        label,
    )?;
    require_exact(
        inner_sha256,
        ROUTER_INNER_SHA256,
        "router inner receipt document SHA-256",
    )?;
    if !bool_field(inner, "component_only", label)?
        || !bool_field(inner, "metal_device_or_dispatch_performed", label)?
        || !bool_field(inner, "opened_exact_layer0_payloads_only", label)?
        || bool_field(inner, "raw_bf16_or_safetensors_opened", label)?
    {
        return Err("inner router receipt no longer has the sealed component-only boundary".into());
    }
    require_exact(
        nested_string(inner, &["artifact_binding", "manifest_seal_sha256"], label)?,
        MANIFEST_SEAL,
        "inner router manifest seal",
    )?;
    require_exact(
        nested_string(
            inner,
            &["artifact_binding", "manifest_document_sha256"],
            label,
        )?,
        MANIFEST_DOCUMENT_SHA256,
        "inner router manifest document SHA-256",
    )?;
    require_exact(
        nested_string(
            inner,
            &["artifact_binding", "admission_receipt_seal_sha256"],
            label,
        )?,
        ADMISSION_RECEIPT_SEAL,
        "inner router admission receipt seal",
    )?;
    require_exact(
        nested_string(inner, &["artifact_binding", "source_repository"], label)?,
        SOURCE_REPOSITORY,
        "inner router source repository",
    )?;
    require_exact(
        nested_string(inner, &["artifact_binding", "source_revision"], label)?,
        SOURCE_REVISION,
        "inner router source revision",
    )?;
    if nested_u64(inner, &["artifact_binding", "layer"], label)? != LAYER as u64
        || nested_u64(inner, &["artifact_binding", "hidden"], label)? != HIDDEN as u64
        || nested_u64(inner, &["artifact_binding", "experts_per_token"], label)? != TOP_K as u64
    {
        return Err("inner router receipt Qwen80 layer/hidden/top-k binding drifted".into());
    }
    let route = nested(inner, &["source_stable_top10_router"], label)?;
    if !bool_field(
        route,
        "ids_unique_and_in_range",
        "inner source-stable top-10 route",
    )? || !bool_field(
        route,
        "device_ids_exact_match",
        "inner source-stable top-10 route",
    )? {
        return Err("inner router route uniqueness/device-ID parity drifted".into());
    }
    let ids = parse_route_ids(
        array_field(route, "ids", "inner source-stable top-10 route")?,
        "inner source-stable top-10 route ids",
    )?;
    let device_ids = parse_route_ids(
        array_field(route, "device_ids", "inner source-stable top-10 route")?,
        "inner device top-10 route ids",
    )?;
    if ids != device_ids {
        return Err("inner router source and device route ID order disagree".into());
    }
    let weights = parse_normalized_weights(
        array_field(
            route,
            "renormalized_weights",
            "inner source-stable top-10 route",
        )?,
        "inner source-stable top-10 normalized weights",
    )?;
    validate_route(ids, weights)?;
    Ok((ids, weights))
}

fn validate_router_evidence(
    outer_bytes: &[u8],
    inner_bytes: &[u8],
) -> Result<RouterEvidence, String> {
    let outer = json_value(outer_bytes, "sealed Qwen80 router outer receipt")?;
    let inner = json_value(inner_bytes, "sealed Qwen80 router inner receipt")?;
    let outer_document_sha256 = sha256_hex(outer_bytes);
    let inner_document_sha256 = sha256_hex(inner_bytes);
    let outer_seal_sha256 = validate_outer_receipt(&outer, &inner_document_sha256)?;
    let (ids, normalized_weights) = validate_inner_receipt(&inner, &inner_document_sha256)?;
    Ok(RouterEvidence {
        outer_document_sha256,
        outer_seal_sha256,
        inner_document_sha256,
        ids,
        normalized_weights,
    })
}

fn expected_tensor_name(expert_id: u16, projection: Projection) -> String {
    format!(
        "model.layers.{LAYER}.mlp.experts.{expert_id}.{}.weight",
        projection.suffix()
    )
}

fn projection_binding(
    descriptor: &TensorDescriptor,
    expert_id: u16,
    projection: Projection,
) -> Result<ProjectionBinding, String> {
    let expected_name = expected_tensor_name(expert_id, projection);
    if descriptor.tensor_name != expected_name {
        return Err(format!(
            "tensor substitution: expected {expected_name:?}, observed {:?}",
            descriptor.tensor_name
        ));
    }
    let expected_shape = projection.expected_shape();
    if descriptor.shape.as_slice() != expected_shape {
        return Err(format!(
            "{} shape drifted: expected {:?}, observed {:?}",
            descriptor.tensor_name, expected_shape, descriptor.shape
        ));
    }
    if descriptor.elements != expected_shape[0] * expected_shape[1] {
        return Err(format!(
            "{} elements drifted: expected {}, observed {}",
            descriptor.tensor_name,
            expected_shape[0] * expected_shape[1],
            descriptor.elements
        ));
    }
    if descriptor.artifact_bytes == 0 || !is_lower_sha256(&descriptor.artifact_sha256) {
        return Err(format!(
            "{} has no valid direct-packed artifact identity",
            descriptor.tensor_name
        ));
    }
    if !descriptor.artifact_path.starts_with('/') || !descriptor.artifact_path.ends_with(".hq30g") {
        return Err(format!(
            "{} must retain an absolute direct-packed .hq30g descriptor path",
            descriptor.tensor_name
        ));
    }
    require_exact(
        &descriptor.source_dtype,
        "BF16",
        &format!("{} source dtype", descriptor.tensor_name),
    )?;
    require_exact(
        &descriptor.source_shard,
        SOURCE_SHARD,
        &format!("{} source shard", descriptor.tensor_name),
    )?;
    require_exact(
        &descriptor.source_shard_sha256,
        SOURCE_SHARD_SHA256,
        &format!("{} source shard SHA-256", descriptor.tensor_name),
    )?;
    if !is_lower_sha256(&descriptor.source_shard_sha256) {
        return Err(format!(
            "{} has malformed source shard SHA-256",
            descriptor.tensor_name
        ));
    }
    if descriptor.layout.magic != "HQ30G1B1"
        || descriptor.layout.group_size != GROUP_SIZE
        || descriptor.layout.scale_dtype != "float16"
        || descriptor.layout.sign_bit_order != "little"
        || descriptor.layout.version != 1
    {
        return Err(format!(
            "{} is not the expected direct-packed HQ30G1B1 layout",
            descriptor.tensor_name
        ));
    }
    Ok(ProjectionBinding {
        tensor_name: descriptor.tensor_name.clone(),
        shape: expected_shape,
        elements: descriptor.elements,
        artifact_path: descriptor.artifact_path.clone(),
        artifact_bytes: descriptor.artifact_bytes,
        artifact_sha256: descriptor.artifact_sha256.clone(),
        source_dtype: descriptor.source_dtype.clone(),
        source_shard: descriptor.source_shard.clone(),
        source_shard_sha256: descriptor.source_shard_sha256.clone(),
        layout: descriptor.layout.clone(),
        payload_opened_by_this_plan: false,
    })
}

fn resolve_exact_descriptor<'a>(
    descriptors: &'a [TensorDescriptor],
    expert_id: u16,
    projection: Projection,
) -> Result<&'a TensorDescriptor, String> {
    let expected_name = expected_tensor_name(expert_id, projection);
    let matches = descriptors
        .iter()
        .filter(|descriptor| descriptor.tensor_name == expected_name)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [descriptor] => Ok(*descriptor),
        [] => Err(format!(
            "missing required direct-packed tensor {expected_name}"
        )),
        _ => Err(format!(
            "duplicate descriptor entries for required tensor {expected_name}"
        )),
    }
}

fn resolve_waves(
    route: &RouterEvidence,
    descriptors: &[TensorDescriptor],
) -> Result<Vec<WavePlan>, String> {
    validate_route(route.ids, route.normalized_weights)?;
    let mut used_payloads = BTreeSet::new();
    let mut waves = Vec::with_capacity(TOP_K);
    for (wave_index, (&expert_id, &normalized_weight)) in route
        .ids
        .iter()
        .zip(route.normalized_weights.iter())
        .enumerate()
    {
        let gate = projection_binding(
            resolve_exact_descriptor(descriptors, expert_id, Projection::Gate)?,
            expert_id,
            Projection::Gate,
        )?;
        let up = projection_binding(
            resolve_exact_descriptor(descriptors, expert_id, Projection::Up)?,
            expert_id,
            Projection::Up,
        )?;
        let down = projection_binding(
            resolve_exact_descriptor(descriptors, expert_id, Projection::Down)?,
            expert_id,
            Projection::Down,
        )?;
        for tensor in [&gate, &up, &down] {
            if !used_payloads.insert(tensor.artifact_sha256.clone()) {
                return Err(format!(
                    "direct-packed payload substitution/duplication detected at {}",
                    tensor.tensor_name
                ));
            }
        }
        waves.push(WavePlan {
            wave_index,
            layer: LAYER,
            expert_id,
            normalized_weight,
            normalized_weight_bits_hex: format!("0x{:016x}", normalized_weight.to_bits()),
            gate,
            up,
            down,
            fixed_operation_order: [
                "gate_proj [512,2048]",
                "up_proj [512,2048]",
                "SiLU(gate) * up [512]",
                "down_proj [2048,512]",
                "apply this route's source-normalized weight [2048]",
            ],
            route_execution_status: "NOT_EXECUTED_SOURCE_BOUND_PLAN_ONLY",
            route_delta_materialized: false,
            route_weight_applied: false,
        });
    }
    if waves.len() != TOP_K || used_payloads.len() != TOP_K * 3 {
        return Err(
            "all-ten plan did not resolve exactly thirty unique direct-packed tensors".into(),
        );
    }
    Ok(waves)
}

fn validate_manifest_inventory(bytes: &[u8]) -> Result<(ManifestInventory, String), String> {
    let inventory_document_sha256 = sha256_hex(bytes);
    require_exact(
        &inventory_document_sha256,
        MANIFEST_DOCUMENT_SHA256,
        "manifest descriptor inventory document SHA-256",
    )?;
    let inventory = serde_json::from_slice::<ManifestInventory>(bytes)
        .map_err(|error| format!("manifest descriptor inventory invalid JSON: {error}"))?;
    require_exact(
        &inventory.schema,
        MANIFEST_SCHEMA,
        "manifest descriptor inventory schema",
    )?;
    require_exact(
        &inventory.status,
        MANIFEST_STATUS,
        "manifest descriptor inventory status",
    )?;
    require_exact(
        &inventory.seal_sha256,
        MANIFEST_SEAL,
        "manifest descriptor inventory seal",
    )?;
    require_exact(
        &inventory.source.repository,
        SOURCE_REPOSITORY,
        "manifest descriptor inventory source repository",
    )?;
    if inventory.source.tensor_count != COMPLETE_TENSOR_COUNT
        || inventory.tensors.len() != COMPLETE_TENSOR_COUNT
    {
        return Err(format!(
            "manifest descriptor inventory must enumerate {COMPLETE_TENSOR_COUNT} tensors; declared {}, received {}",
            inventory.source.tensor_count,
            inventory.tensors.len()
        ));
    }
    Ok((inventory, inventory_document_sha256))
}

fn build_report(
    router: RouterEvidence,
    inventory: ManifestInventory,
    inventory_document_sha256: String,
) -> Result<PlanReport, String> {
    let waves = resolve_waves(&router, &inventory.tensors)?;
    Ok(PlanReport {
        schema: RESULT_SCHEMA,
        status: "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED",
        model_id: MODEL_ID,
        model_key: MODEL_KEY,
        source_repository: SOURCE_REPOSITORY,
        source_revision: SOURCE_REVISION,
        layer: LAYER,
        router_evidence: RouterEvidenceBinding {
            outer_receipt_document_sha256: router.outer_document_sha256,
            outer_receipt_seal_sha256: router.outer_seal_sha256,
            inner_receipt_document_sha256: router.inner_document_sha256,
            source_stable_route_ids: router.ids,
            source_stable_normalized_weights: router.normalized_weights,
            source_router_component_only: true,
        },
        manifest_descriptor_inventory: ManifestDescriptorBinding {
            inventory_document_sha256,
            manifest_schema: inventory.schema,
            manifest_status: inventory.status,
            manifest_seal_sha256: inventory.seal_sha256,
            source_repository: inventory.source.repository,
            declared_tensor_count: inventory.source.tensor_count,
            received_descriptor_count: inventory.tensors.len(),
            resolved_route_tensor_count: TOP_K * 3,
            payload_opened_by_this_plan: false,
        },
        deterministic_waves: waves,
        rawls_real_all_ten_provenance_gate: RealAllTenProvenanceGate {
            schema: PROVENANCE_GATE_SCHEMA,
            all_ten_source_bindings_complete: true,
            expected_layer: LAYER,
            deterministic_wave_indices: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            route_order: EXPECTED_ROUTE_IDS,
            normalized_weights: EXPECTED_ROUTE_WEIGHTS,
            execution_receipt_required_for_each_wave: true,
            direct_packed_execution_required_for_each_wave: true,
            source_bound_input_required_for_each_wave: true,
            route_combine_receipt_required_separately: true,
            shared_expert_receipt_required_separately: true,
            first_and_second_residual_receipts_required_separately: true,
            rejects_tensor_substitution: true,
            rejects_route_reorder: true,
            rejects_duplicate_experts: true,
            rejects_missing_tensor_or_weight: true,
        },
        route_execution_performed: false,
        route_combine_performed: false,
        shared_expert_performed: false,
        residual_combine_performed: false,
        metal_device_or_dispatch_performed: false,
        model_execution_performed: false,
        hcli_execution_performed: false,
        tps_or_tg_measurement_performed: false,
        complete_layer_or_decoder_claim_earned: false,
        claim_boundary: [
            "This is a source-bound CPU plan; it does not execute any routed expert.",
            "This plan does not aggregate routes, run the shared expert, or combine either residual.",
            "The source router remains a synthetic-input component receipt, not a Qwen80 layer or token.",
            "No decoder, server, HCLI, TPS, TG, capability, or tournament claim follows.",
        ],
    })
}

fn parse_args() -> Result<Arguments, Box<dyn Error>> {
    let mut router_outer = None;
    let mut router_inner = None;
    let mut manifest_inventory = None;
    let mut out = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing absolute path after {flag}"))?;
        let slot = match flag.as_str() {
            "--router-outer" => &mut router_outer,
            "--router-inner" => &mut router_inner,
            "--manifest-inventory" => &mut manifest_inventory,
            "--out" => &mut out,
            _ => return Err("usage: ascension_qwen80_all_ten_routed_expert_plan --router-outer ABSOLUTE_PATH --router-inner ABSOLUTE_PATH --manifest-inventory ABSOLUTE_PATH --out NEW_ABSOLUTE_PATH".into()),
        };
        if slot.replace(PathBuf::from(value)).is_some() {
            return Err(format!("{flag} supplied more than once").into());
        }
    }
    let router_outer = router_outer.ok_or("missing --router-outer")?;
    let router_inner = router_inner.ok_or("missing --router-inner")?;
    let manifest_inventory = manifest_inventory.ok_or("missing --manifest-inventory")?;
    let out = out.ok_or("missing --out")?;
    for (label, path) in [
        ("--router-outer", &router_outer),
        ("--router-inner", &router_inner),
        ("--manifest-inventory", &manifest_inventory),
        ("--out", &out),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be an absolute path").into());
        }
    }
    Ok(Arguments {
        router_outer,
        router_inner,
        manifest_inventory,
        out,
    })
}

fn write_new_report(path: &Path, report: &PlanReport) -> Result<(), Box<dyn Error>> {
    if path.exists() {
        return Err(format!("refusing to overwrite existing plan: {}", path.display()).into());
    }
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent does not exist: {}", parent.display()).into());
    }
    let bytes = serde_json::to_vec_pretty(report)?;
    let mut output = OpenOptions::new().write(true).create_new(true).open(path)?;
    output.write_all(&bytes)?;
    output.write_all(b"\n")?;
    output.sync_all()?;
    Ok(())
}

fn run(arguments: Arguments) -> Result<(), Box<dyn Error>> {
    let outer_bytes = regular_file_bytes(&arguments.router_outer, "router outer receipt")?;
    let inner_bytes = regular_file_bytes(&arguments.router_inner, "router inner receipt")?;
    let manifest_bytes = regular_file_bytes(
        &arguments.manifest_inventory,
        "manifest descriptor inventory",
    )?;
    let router = validate_router_evidence(&outer_bytes, &inner_bytes)?;
    let (inventory, inventory_document_sha256) = validate_manifest_inventory(&manifest_bytes)?;
    let report = build_report(router, inventory, inventory_document_sha256)?;
    write_new_report(&arguments.out, &report)?;
    Ok(())
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_all_ten_routed_expert_plan: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_sha(index: usize) -> String {
        format!("{index:064x}")
    }

    fn direct_packed_descriptor(
        expert_id: u16,
        projection: Projection,
        index: usize,
    ) -> TensorDescriptor {
        let shape = projection.expected_shape();
        TensorDescriptor {
            tensor_name: expected_tensor_name(expert_id, projection),
            shape: shape.to_vec(),
            elements: shape[0] * shape[1],
            artifact_path: format!("/catalog/{expert_id}/{}.hq30g", projection.suffix()),
            artifact_bytes: 1 + index as u64,
            artifact_sha256: fake_sha(index + 1),
            source_dtype: "BF16".into(),
            source_shard: SOURCE_SHARD.into(),
            source_shard_sha256: SOURCE_SHARD_SHA256.into(),
            layout: PackedLayout {
                magic: "HQ30G1B1".into(),
                group_size: GROUP_SIZE,
                scale_dtype: "float16".into(),
                sign_bit_order: "little".into(),
                version: 1,
            },
        }
    }

    fn all_direct_packed_descriptors() -> Vec<TensorDescriptor> {
        EXPECTED_ROUTE_IDS
            .iter()
            .enumerate()
            .flat_map(|(route_index, expert_id)| {
                [Projection::Gate, Projection::Up, Projection::Down]
                    .into_iter()
                    .enumerate()
                    .map(move |(projection_index, projection)| {
                        direct_packed_descriptor(
                            *expert_id,
                            projection,
                            route_index * 3 + projection_index,
                        )
                    })
            })
            .collect()
    }

    fn valid_router() -> RouterEvidence {
        RouterEvidence {
            outer_document_sha256: fake_sha(100),
            outer_seal_sha256: ROUTER_OUTER_SEAL.into(),
            inner_document_sha256: ROUTER_INNER_SHA256.into(),
            ids: EXPECTED_ROUTE_IDS,
            normalized_weights: EXPECTED_ROUTE_WEIGHTS,
        }
    }

    #[test]
    fn complete_inventory_compiles_exact_deterministic_ten_wave_plan() {
        let waves = resolve_waves(&valid_router(), &all_direct_packed_descriptors()).unwrap();
        assert_eq!(waves.len(), TOP_K);
        assert_eq!(
            waves.iter().map(|wave| wave.wave_index).collect::<Vec<_>>(),
            (0..TOP_K).collect::<Vec<_>>()
        );
        assert_eq!(
            waves.iter().map(|wave| wave.expert_id).collect::<Vec<_>>(),
            EXPECTED_ROUTE_IDS
        );
        assert_eq!(
            waves[0].normalized_weight.to_bits(),
            EXPECTED_ROUTE_WEIGHTS[0].to_bits()
        );
        assert!(waves
            .iter()
            .all(|wave| wave.route_execution_status == "NOT_EXECUTED_SOURCE_BOUND_PLAN_ONLY"));
        assert!(waves.iter().all(|wave| !wave.route_weight_applied));
    }

    #[test]
    fn reordered_source_route_is_rejected_before_any_tensor_resolution() {
        let mut router = valid_router();
        router.ids.swap(0, 1);
        let error = resolve_waves(&router, &all_direct_packed_descriptors()).unwrap_err();
        assert!(error.contains("route order drifted"));
    }

    #[test]
    fn duplicate_source_expert_is_rejected() {
        let mut router = valid_router();
        router.ids[9] = router.ids[0];
        let error = resolve_waves(&router, &all_direct_packed_descriptors()).unwrap_err();
        assert!(error.contains("route order drifted") || error.contains("duplicate"));
    }

    #[test]
    fn missing_direct_packed_projection_is_rejected() {
        let mut descriptors = all_direct_packed_descriptors();
        descriptors.retain(|descriptor| {
            descriptor.tensor_name != expected_tensor_name(245, Projection::Down)
        });
        let error = resolve_waves(&valid_router(), &descriptors).unwrap_err();
        assert!(error.contains("missing required direct-packed tensor"));
    }

    #[test]
    fn wrong_projection_shape_cannot_substitute_for_gate() {
        let mut descriptors = all_direct_packed_descriptors();
        let gate = descriptors
            .iter_mut()
            .find(|descriptor| descriptor.tensor_name == expected_tensor_name(65, Projection::Gate))
            .unwrap();
        gate.shape = vec![HIDDEN, INTERMEDIATE];
        let error = resolve_waves(&valid_router(), &descriptors).unwrap_err();
        assert!(error.contains("shape drifted"));
    }

    #[test]
    fn duplicate_payload_identity_is_rejected_as_substitution() {
        let mut descriptors = all_direct_packed_descriptors();
        let first_sha = descriptors[0].artifact_sha256.clone();
        let replacement = descriptors
            .iter_mut()
            .find(|descriptor| descriptor.tensor_name == expected_tensor_name(65, Projection::Up))
            .unwrap();
        replacement.artifact_sha256 = first_sha;
        let error = resolve_waves(&valid_router(), &descriptors).unwrap_err();
        assert!(error.contains("payload substitution/duplication"));
    }

    #[test]
    fn normalized_weight_drift_is_rejected_at_its_exact_route_slot() {
        let mut router = valid_router();
        router.normalized_weights[3] = 0.098_244_741_559_028_64;
        let error = resolve_waves(&router, &all_direct_packed_descriptors()).unwrap_err();
        assert!(error.contains("normalized weight at route index 3 drifted"));
    }

    #[test]
    fn non_direct_packed_layout_is_rejected() {
        let mut descriptors = all_direct_packed_descriptors();
        let up = descriptors
            .iter_mut()
            .find(|descriptor| descriptor.tensor_name == expected_tensor_name(494, Projection::Up))
            .unwrap();
        up.layout.magic = "NOTPACKED".into();
        let error = resolve_waves(&valid_router(), &descriptors).unwrap_err();
        assert!(error.contains("expected direct-packed HQ30G1B1 layout"));
    }

    #[test]
    fn machine_descriptor_never_promotes_execution_or_combine() {
        let inventory = ManifestInventory {
            schema: MANIFEST_SCHEMA.into(),
            status: MANIFEST_STATUS.into(),
            seal_sha256: MANIFEST_SEAL.into(),
            source: ManifestSource {
                repository: SOURCE_REPOSITORY.into(),
                tensor_count: COMPLETE_TENSOR_COUNT,
            },
            tensors: all_direct_packed_descriptors(),
        };
        let report =
            build_report(valid_router(), inventory, MANIFEST_DOCUMENT_SHA256.into()).unwrap();
        assert!(!report.route_execution_performed);
        assert!(!report.route_combine_performed);
        assert!(!report.shared_expert_performed);
        assert!(!report.residual_combine_performed);
        assert!(!report.complete_layer_or_decoder_claim_earned);
        assert!(
            report
                .rawls_real_all_ten_provenance_gate
                .execution_receipt_required_for_each_wave
        );
    }
}
