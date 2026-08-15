//! Fail-closed Qwen3-Coder-Next layer-0 MoE aggregate/second-residual seam.
//!
//! This CPU-only component binds the sealed source top-10 router receipt and
//! accepts exactly ten *already-weighted*, route-index-ordered `[2048]` deltas,
//! one sigmoid-gated shared `[2048]` delta, and one first residual `[2048]`.
//! It performs a deliberately fixed f32 operation order:
//!
//! `route[0] -> ... -> route[9] -> add gated shared -> add first residual`
//!
//! The executable uses a deterministic materialized fixture because only one
//! of the ten physical routed waves has its own narrow receipt at this stage.
//! It does not substitute the fixture for missing source expert computation.
//! No complete layer/token/decoder/HCLI/TPS claim follows.

use hawking_core::model::qwen80_complete_runtime::{Qwen80CompleteRuntimeConfig, Qwen80LayerKind};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_moe_wave_aggregate_second_residual.v1";
const CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen80_moe_wave_aggregate_second_residual_capture.v1";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const CURRENT_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ROUTER_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const ROUTER_RECEIPT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_CPU_ORACLE_READY_METAL_LEASE_REQUIRED";
const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const CURRENT_MANIFEST_SEAL: &str =
    "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const CURRENT_MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const CURRENT_ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const SOURCE_BODY_AUDIT_SEAL: &str =
    "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4";
const SOURCE_REVALIDATION_SEAL: &str =
    "541b16fca1d4805ecba356face97b4e8de1accdeb21e98ee0c13b70ab0746c45";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";
const ROUTER_FIXTURE_RESIDUAL_SHA256: &str =
    "da6bd17a333c40353442d7db7e0d16c1c44870f3458c785d6a4ebe48a3477fef";
const POST_NORM_NAME: &str = "model.layers.0.post_attention_layernorm.weight";
const POST_NORM_ARTIFACT_SHA256: &str =
    "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7";
const LAYER: usize = 0;
const HIDDEN: usize = 2_048;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const EXPECTED_ROUTE_IDS: [u16; TOP_K] = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494];
const EXPECTED_ROUTE0_WEIGHT: f64 = 0.245_458_886_027_336_12;
const ACCUMULATION_F32_F64_TOLERANCE: f64 = 2.0e-5;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Mode {
    CpuOracle,
    Metal,
}

#[derive(Clone, Debug)]
struct Args {
    manifest: PathBuf,
    admission_current: PathBuf,
    router_receipt: PathBuf,
    capture_dir: PathBuf,
    mode: Mode,
}

#[derive(Clone, Debug)]
struct RouteEvidence {
    receipt_path: PathBuf,
    receipt_sha256: String,
    ids: [u16; TOP_K],
    weights: [f32; TOP_K],
    weights_f64: [f64; TOP_K],
}

#[derive(Clone, Debug)]
struct BoundComponent {
    manifest_path: PathBuf,
    admission_current_path: PathBuf,
    manifest_document_sha256: String,
    admission_pointer_seal_sha256: String,
    source_model_dir: PathBuf,
    source_config_sha256: String,
    config: Qwen80CompleteRuntimeConfig,
    route: RouteEvidence,
}

#[derive(Clone, Debug)]
struct AggregateInput {
    accepted_ids: [u16; TOP_K],
    normalized_weights: [f32; TOP_K],
    already_weighted_route_deltas: Vec<Vec<f32>>,
    gated_shared: Vec<f32>,
    first_residual: Vec<f32>,
}

#[derive(Clone, Debug)]
struct AggregateOutput {
    routed_sum: Vec<f32>,
    second_residual: Vec<f32>,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32_slice_sha256(values: &[f32]) -> String {
    sha256_hex(bytemuck::cast_slice(values))
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

fn checked_sha256(bytes: &[u8], expected: &str, label: &str) -> Result<String, String> {
    let observed = sha256_hex(bytes);
    if observed != expected {
        return Err(format!(
            "{label} SHA-256 mismatch: expected {expected}, observed {observed}"
        ));
    }
    Ok(observed)
}

fn json_object(bytes: &[u8], label: &str) -> Result<Map<String, Value>, String> {
    serde_json::from_slice::<Value>(bytes)
        .map_err(|error| format!("{label} invalid JSON: {error}"))?
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{label} root must be an object"))
}

fn string_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} missing string field {field:?}"))
}

fn u64_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label} missing unsigned field {field:?}"))
}

fn object_field<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} missing object field {field:?}"))
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
        .ok_or_else(|| format!("{label} missing array field {field:?}"))
}

fn canonical_dir(path: &Path, label: &str) -> Result<PathBuf, String> {
    let canonical = path.canonicalize().map_err(|error| {
        format!(
            "{label} canonicalization failed at {}: {error}",
            path.display()
        )
    })?;
    if !fs::symlink_metadata(&canonical)
        .map_err(|error| format!("{label} metadata failed: {error}"))?
        .is_dir()
    {
        return Err(format!(
            "{label} is not a directory: {}",
            canonical.display()
        ));
    }
    Ok(canonical)
}

fn canonical_regular_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("{label} metadata failed at {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    path.canonicalize().map_err(|error| {
        format!(
            "{label} canonicalization failed at {}: {error}",
            path.display()
        )
    })
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn exact_path_from_json(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    path.canonicalize().map_err(|error| {
        format!(
            "{label} canonicalization failed at {}: {error}",
            path.display()
        )
    })
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::CpuOracle => "cpu-oracle",
        Mode::Metal => "metal",
    }
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut manifest = None;
    let mut admission_current = None;
    let mut router_receipt = None;
    let mut capture_dir = None;
    let mut mode = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--manifest" => {
                if manifest.replace(PathBuf::from(value)).is_some() {
                    return Err("--manifest repeated".into());
                }
            }
            "--admission-current" => {
                if admission_current.replace(PathBuf::from(value)).is_some() {
                    return Err("--admission-current repeated".into());
                }
            }
            "--router-receipt" => {
                if router_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("--router-receipt repeated".into());
                }
            }
            "--capture-dir" => {
                if capture_dir.replace(PathBuf::from(value)).is_some() {
                    return Err("--capture-dir repeated".into());
                }
            }
            "--mode" => {
                let parsed = match value.as_str() {
                    "cpu-oracle" => Mode::CpuOracle,
                    "metal" => Mode::Metal,
                    _ => return Err("--mode must be cpu-oracle or metal".into()),
                };
                if mode.replace(parsed).is_some() {
                    return Err("--mode repeated".into());
                }
            }
            _ => return Err("usage: ascension_qwen80_moe_wave_aggregate_second_residual --manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH --router-receipt ABSOLUTE_PATH --capture-dir NEW_ABSOLUTE_DIRECTORY --mode cpu-oracle|metal".into()),
        }
    }
    let manifest = manifest.ok_or("missing --manifest")?;
    let admission_current = admission_current.ok_or("missing --admission-current")?;
    let router_receipt = router_receipt.ok_or("missing --router-receipt")?;
    let capture_dir = capture_dir.ok_or("missing --capture-dir")?;
    let mode = mode.ok_or("missing --mode")?;
    if !manifest.is_absolute()
        || !admission_current.is_absolute()
        || !router_receipt.is_absolute()
        || !capture_dir.is_absolute()
    {
        return Err("all path arguments must be absolute".into());
    }
    Ok(Args {
        manifest,
        admission_current,
        router_receipt,
        capture_dir,
        mode,
    })
}

fn validate_current_admission(
    manifest_path: &Path,
    manifest_document_sha256: &str,
    admission_path: &Path,
) -> Result<String, String> {
    let pointer_bytes = regular_file_bytes(admission_path, "current Qwen80 admission pointer")?;
    let pointer = json_object(&pointer_bytes, "current Qwen80 admission pointer")?;
    let pointer_seal = string_field(&pointer, "seal_sha256", "current Qwen80 admission pointer")?;
    if string_field(&pointer, "schema", "current Qwen80 admission pointer")?
        != CURRENT_ADMISSION_SCHEMA
        || string_field(&pointer, "status", "current Qwen80 admission pointer")?
            != "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
        || !is_lower_sha256(pointer_seal)
    {
        return Err("current Qwen80 admission pointer identity/status drifted".into());
    }
    let model = object_field(&pointer, "model", "current Qwen80 admission pointer")?;
    if string_field(model, "id", "current Qwen80 admission model")? != MODEL_ID
        || string_field(model, "key", "current Qwen80 admission model")? != MODEL_KEY
        || string_field(model, "repository", "current Qwen80 admission model")? != SOURCE_REPOSITORY
        || string_field(model, "revision", "current Qwen80 admission model")? != SOURCE_REVISION
    {
        return Err("current Qwen80 admission pointer model identity drifted".into());
    }
    let complete_manifest = object_field(
        &pointer,
        "complete_manifest",
        "current Qwen80 admission pointer",
    )?;
    if exact_path_from_json(
        string_field(
            complete_manifest,
            "path",
            "current Qwen80 complete manifest pointer",
        )?,
        "current Qwen80 complete manifest pointer path",
    )? != manifest_path
        || string_field(
            complete_manifest,
            "schema",
            "current Qwen80 complete manifest pointer",
        )? != MANIFEST_SCHEMA
        || string_field(
            complete_manifest,
            "seal_sha256",
            "current Qwen80 complete manifest pointer",
        )? != CURRENT_MANIFEST_SEAL
        || string_field(
            complete_manifest,
            "document_sha256",
            "current Qwen80 complete manifest pointer",
        )? != manifest_document_sha256
    {
        return Err("current admission pointer does not bind this exact sealed manifest".into());
    }
    let receipt = object_field(
        &pointer,
        "admission_receipt",
        "current Qwen80 admission pointer",
    )?;
    if string_field(receipt, "seal_sha256", "current Qwen80 admission receipt")?
        != CURRENT_ADMISSION_RECEIPT_SEAL
    {
        return Err("current Qwen80 admission receipt seal drifted".into());
    }
    Ok(pointer_seal.to_owned())
}

fn parse_route_ids(values: &[Value]) -> Result<[u16; TOP_K], String> {
    if values.len() != TOP_K {
        return Err(format!(
            "source top10 IDs must contain exactly {TOP_K} entries"
        ));
    }
    let mut ids = [0u16; TOP_K];
    for (index, value) in values.iter().enumerate() {
        ids[index] = value
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .filter(|id| usize::from(*id) < EXPERTS)
            .ok_or_else(|| format!("source top10 ID {index} is invalid"))?;
        if ids[..index].contains(&ids[index]) {
            return Err(format!("source top10 ID {index} is duplicated"));
        }
    }
    Ok(ids)
}

fn parse_route_weights(values: &[Value]) -> Result<([f32; TOP_K], [f64; TOP_K]), String> {
    if values.len() != TOP_K {
        return Err(format!(
            "source top10 weights must contain exactly {TOP_K} entries"
        ));
    }
    let mut weights = [0.0f32; TOP_K];
    let mut weights_f64 = [0.0f64; TOP_K];
    for (index, value) in values.iter().enumerate() {
        let weight = value
            .as_f64()
            .filter(|weight| weight.is_finite() && *weight >= 0.0)
            .ok_or_else(|| format!("source top10 weight {index} is invalid"))?;
        weights[index] = weight as f32;
        if !weights[index].is_finite() {
            return Err(format!("source top10 weight {index} cannot fit f32"));
        }
        weights_f64[index] = weight;
    }
    let sum = weights_f64.iter().sum::<f64>();
    if (sum - 1.0).abs() > 2.0e-6 {
        return Err(format!(
            "source top10 weights are not normalized: sum={sum}"
        ));
    }
    Ok((weights, weights_f64))
}

fn bind_route_evidence(
    path: &Path,
    manifest_path: &Path,
    manifest_document_sha256: &str,
) -> Result<RouteEvidence, String> {
    let receipt_path = canonical_regular_file(path, "Qwen80 postnorm/router receipt")?;
    let bytes = regular_file_bytes(&receipt_path, "Qwen80 postnorm/router receipt")?;
    let receipt_sha256 = sha256_hex(&bytes);
    let receipt = json_object(&bytes, "Qwen80 postnorm/router receipt")?;
    if string_field(&receipt, "schema", "Qwen80 postnorm/router receipt")? != ROUTER_RECEIPT_SCHEMA
        || string_field(&receipt, "status", "Qwen80 postnorm/router receipt")?
            != ROUTER_RECEIPT_STATUS
        || receipt.get("component_only").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("metal_device_or_dispatch_performed")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err(
            "postnorm/router receipt is not the required CPU-only source route evidence".into(),
        );
    }
    let binding = object_field(
        &receipt,
        "artifact_binding",
        "Qwen80 postnorm/router receipt",
    )?;
    if exact_path_from_json(
        string_field(binding, "manifest_path", "Qwen80 postnorm/router binding")?,
        "Qwen80 postnorm/router manifest path",
    )? != manifest_path
        || string_field(
            binding,
            "manifest_document_sha256",
            "Qwen80 postnorm/router binding",
        )? != manifest_document_sha256
        || string_field(
            binding,
            "manifest_seal_sha256",
            "Qwen80 postnorm/router binding",
        )? != CURRENT_MANIFEST_SEAL
        || string_field(
            binding,
            "source_repository",
            "Qwen80 postnorm/router binding",
        )? != SOURCE_REPOSITORY
        || string_field(binding, "source_revision", "Qwen80 postnorm/router binding")?
            != SOURCE_REVISION
        || string_field(
            binding,
            "source_config_sha256",
            "Qwen80 postnorm/router binding",
        )? != SOURCE_CONFIG_SHA256
        || u64_field(binding, "layer", "Qwen80 postnorm/router binding")? != LAYER as u64
        || u64_field(binding, "hidden", "Qwen80 postnorm/router binding")? != HIDDEN as u64
        || u64_field(binding, "router_logits", "Qwen80 postnorm/router binding")? != EXPERTS as u64
        || u64_field(
            binding,
            "experts_per_token",
            "Qwen80 postnorm/router binding",
        )? != TOP_K as u64
    {
        return Err("source top10 receipt binding drifted".into());
    }
    let norm = object_field(
        binding,
        "post_attention_norm",
        "Qwen80 postnorm/router binding",
    )?;
    if string_field(norm, "name", "Qwen80 postnorm/router norm")? != POST_NORM_NAME
        || string_field(norm, "artifact_sha256", "Qwen80 postnorm/router norm")?
            != POST_NORM_ARTIFACT_SHA256
    {
        return Err("source top10 postnorm binding drifted".into());
    }
    let oracle = object_field(&receipt, "cpu_oracle", "Qwen80 postnorm/router receipt")?;
    if string_field(
        oracle,
        "post_attention_residual_sha256",
        "Qwen80 postnorm/router oracle",
    )? != ROUTER_FIXTURE_RESIDUAL_SHA256
    {
        return Err("source top10 fixture residual identity drifted".into());
    }
    let route = object_field(
        &receipt,
        "source_stable_top10_router",
        "Qwen80 postnorm/router receipt",
    )?;
    let ids = parse_route_ids(array_field(route, "ids", "source top10 route")?)?;
    let (weights, weights_f64) = parse_route_weights(array_field(
        route,
        "renormalized_weights",
        "source top10 route",
    )?)?;
    if ids != EXPECTED_ROUTE_IDS || (weights_f64[0] - EXPECTED_ROUTE0_WEIGHT).abs() > 1.0e-12 {
        return Err("source top10 route no longer matches the sealed layer-0 receipt".into());
    }
    Ok(RouteEvidence {
        receipt_path,
        receipt_sha256,
        ids,
        weights,
        weights_f64,
    })
}

fn bind_source_config(
    manifest: &Map<String, Value>,
) -> Result<(PathBuf, String, Qwen80CompleteRuntimeConfig), String> {
    let source = object_field(manifest, "source", "Qwen80 complete manifest")?;
    if string_field(source, "repository", "Qwen80 complete manifest source")? != SOURCE_REPOSITORY
        || u64_field(source, "tensor_count", "Qwen80 complete manifest source")? != 74_391
    {
        return Err("Qwen80 complete manifest source authority drifted".into());
    }
    let model_dir = canonical_dir(
        Path::new(string_field(
            source,
            "model_dir",
            "Qwen80 complete manifest source",
        )?),
        "Qwen80 source model directory",
    )?;
    let config_path =
        canonical_regular_file(&model_dir.join("config.json"), "Qwen80 source config")?;
    let config_bytes = regular_file_bytes(&config_path, "Qwen80 source config")?;
    let config_sha256 =
        checked_sha256(&config_bytes, SOURCE_CONFIG_SHA256, "Qwen80 source config")?;
    let config_document = serde_json::from_slice::<Value>(&config_bytes)
        .map_err(|error| format!("Qwen80 source config JSON rejected: {error}"))?;
    let config = Qwen80CompleteRuntimeConfig::from_source_config(
        &config_document,
        SOURCE_REPOSITORY,
        SOURCE_REVISION,
    )
    .map_err(|error| format!("Qwen80 exact source config parser rejected it: {error}"))?;
    if config.hidden != HIDDEN
        || config.experts != EXPERTS
        || config.experts_per_token != TOP_K
        || config
            .layer_kind(LAYER)
            .map_err(|error| format!("Qwen80 layer-0 kind rejected: {error}"))?
            != Qwen80LayerKind::LinearAttention
    {
        return Err("source config no longer matches Qwen80 MoE aggregate geometry".into());
    }
    Ok((model_dir, config_sha256, config))
}

fn bind_current_component(args: &Args) -> Result<BoundComponent, String> {
    let manifest_path = canonical_regular_file(&args.manifest, "Qwen80 complete manifest")?;
    let manifest_bytes = regular_file_bytes(&manifest_path, "Qwen80 complete manifest")?;
    let manifest_document_sha256 = checked_sha256(
        &manifest_bytes,
        CURRENT_MANIFEST_DOCUMENT_SHA256,
        "Qwen80 complete manifest document",
    )?;
    let manifest = json_object(&manifest_bytes, "Qwen80 complete manifest")?;
    if string_field(&manifest, "schema", "Qwen80 complete manifest")? != MANIFEST_SCHEMA
        || string_field(&manifest, "seal_sha256", "Qwen80 complete manifest")?
            != CURRENT_MANIFEST_SEAL
        || string_field(&manifest, "status", "Qwen80 complete manifest")?
            != "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
        || string_field(
            &manifest,
            "source_body_audit_seal_sha256",
            "Qwen80 complete manifest",
        )? != SOURCE_BODY_AUDIT_SEAL
        || string_field(
            &manifest,
            "source_revalidation_receipt_seal_sha256",
            "Qwen80 complete manifest",
        )? != SOURCE_REVALIDATION_SEAL
    {
        return Err("current Qwen80 complete manifest identity/seal/status drifted".into());
    }
    let admission_current_path =
        canonical_regular_file(&args.admission_current, "current Qwen80 admission pointer")?;
    let admission_pointer_seal_sha256 = validate_current_admission(
        &manifest_path,
        &manifest_document_sha256,
        &admission_current_path,
    )?;
    let route = bind_route_evidence(
        &args.router_receipt,
        &manifest_path,
        &manifest_document_sha256,
    )?;
    let (source_model_dir, source_config_sha256, config) = bind_source_config(&manifest)?;
    Ok(BoundComponent {
        manifest_path,
        admission_current_path,
        manifest_document_sha256,
        admission_pointer_seal_sha256,
        source_model_dir,
        source_config_sha256,
        config,
        route,
    })
}

fn deterministic_values(seed: u64, elements: usize, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..elements)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) & 0x00ff_ffff) as f32 / 16_777_215.0;
            let phase = ((index * 29 % 113) as f32 - 56.0) / 257.0;
            ((unit * 2.0 - 1.0) + phase) * scale
        })
        .collect()
}

fn deterministic_materialized_fixture(route: &RouteEvidence) -> AggregateInput {
    let already_weighted_route_deltas = route
        .weights
        .iter()
        .enumerate()
        .map(|(route_index, &weight)| {
            let raw = deterministic_values(
                0x7a20_5f4e_19d3_0001u64
                    .wrapping_add(u64::from(route.ids[route_index]) * 131)
                    .wrapping_add(route_index as u64),
                HIDDEN,
                0.42,
            );
            raw.into_iter().map(|value| value * weight).collect()
        })
        .collect();
    AggregateInput {
        accepted_ids: route.ids,
        normalized_weights: route.weights,
        already_weighted_route_deltas,
        gated_shared: deterministic_values(0x5ced_1a2b_9043_0002, HIDDEN, 0.31),
        first_residual: deterministic_values(0x4ab2_8917_ef0c_0003, HIDDEN, 0.63),
    }
}

fn validate_input(input: &AggregateInput, expected_route: &RouteEvidence) -> Result<(), String> {
    if input.accepted_ids != expected_route.ids || input.accepted_ids != EXPECTED_ROUTE_IDS {
        return Err(
            "MoE aggregate route IDs are missing or reordered from accepted source top10".into(),
        );
    }
    if input.already_weighted_route_deltas.len() != TOP_K {
        return Err(format!(
            "MoE aggregate requires exactly {TOP_K} route-index-ordered weighted deltas"
        ));
    }
    let mut seen = [false; EXPERTS];
    for (index, &id) in input.accepted_ids.iter().enumerate() {
        let id = usize::from(id);
        if id >= EXPERTS || seen[id] {
            return Err(format!(
                "MoE aggregate route ID at index {index} is invalid or duplicate"
            ));
        }
        seen[id] = true;
    }
    let weight_sum = input.normalized_weights.iter().sum::<f32>();
    if input
        .normalized_weights
        .iter()
        .any(|weight| !weight.is_finite() || *weight < 0.0)
        || !weight_sum.is_finite()
        || (weight_sum - 1.0).abs() > 2.0e-6
    {
        return Err(format!(
            "MoE aggregate route weights are not normalized: sum={weight_sum}"
        ));
    }
    for (index, (&observed, &expected)) in input
        .normalized_weights
        .iter()
        .zip(expected_route.weights.iter())
        .enumerate()
    {
        if (observed - expected).abs() > 1.0e-7 {
            return Err(format!(
                "MoE aggregate route weight at index {index} drifted from source receipt"
            ));
        }
    }
    for (route_index, delta) in input.already_weighted_route_deltas.iter().enumerate() {
        if delta.len() != HIDDEN || delta.iter().any(|value| !value.is_finite()) {
            return Err(format!(
                "MoE aggregate weighted delta at route index {route_index} is not finite [2048]"
            ));
        }
    }
    if input.gated_shared.len() != HIDDEN
        || input.first_residual.len() != HIDDEN
        || input
            .gated_shared
            .iter()
            .chain(input.first_residual.iter())
            .any(|value| !value.is_finite())
    {
        return Err("MoE aggregate shared/first-residual inputs are not finite [2048]".into());
    }
    Ok(())
}

fn aggregate_fixed_f32(input: &AggregateInput) -> Result<AggregateOutput, String> {
    let dummy_route = RouteEvidence {
        receipt_path: PathBuf::new(),
        receipt_sha256: String::new(),
        ids: input.accepted_ids,
        weights: input.normalized_weights,
        weights_f64: input.normalized_weights.map(f64::from),
    };
    validate_input(input, &dummy_route)?;
    let mut routed_sum = vec![0.0f32; HIDDEN];
    for route_index in 0..TOP_K {
        for (sum, value) in routed_sum
            .iter_mut()
            .zip(&input.already_weighted_route_deltas[route_index])
        {
            *sum += *value;
        }
    }
    let second_residual = routed_sum
        .iter()
        .zip(&input.gated_shared)
        .zip(&input.first_residual)
        .map(|((&routed, &shared), &first)| {
            let mut value = routed;
            value += shared;
            value += first;
            value
        })
        .collect::<Vec<_>>();
    if routed_sum
        .iter()
        .chain(second_residual.iter())
        .any(|value| !value.is_finite())
    {
        return Err("MoE aggregate fixed f32 accumulation produced non-finite values".into());
    }
    Ok(AggregateOutput {
        routed_sum,
        second_residual,
    })
}

fn aggregate_reference_f64(input: &AggregateInput) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut routed_sum = vec![0.0f64; HIDDEN];
    for route_index in 0..TOP_K {
        for (sum, value) in routed_sum
            .iter_mut()
            .zip(&input.already_weighted_route_deltas[route_index])
        {
            *sum += f64::from(*value);
        }
    }
    let second_residual = routed_sum
        .iter()
        .zip(&input.gated_shared)
        .zip(&input.first_residual)
        .map(|((&routed, &shared), &first)| routed + f64::from(shared) + f64::from(first))
        .collect::<Vec<_>>();
    if routed_sum
        .iter()
        .chain(second_residual.iter())
        .any(|value| !value.is_finite())
    {
        return Err("MoE aggregate f64 reference produced non-finite values".into());
    }
    Ok((routed_sum, second_residual))
}

fn max_f64_error(candidate: &[f32], reference: &[f64], label: &str) -> Result<(f64, f64), String> {
    if candidate.len() != reference.len() {
        return Err(format!(
            "{label} length mismatch: candidate={}, reference={}",
            candidate.len(),
            reference.len()
        ));
    }
    let mut max_abs = 0.0f64;
    let mut max_relative = 0.0f64;
    for (index, (&candidate, &reference)) in candidate.iter().zip(reference).enumerate() {
        if !candidate.is_finite() || !reference.is_finite() {
            return Err(format!("{label} contains non-finite value at {index}"));
        }
        let absolute = (f64::from(candidate) - reference).abs();
        max_abs = max_abs.max(absolute);
        max_relative = max_relative.max(absolute / reference.abs().max(1.0));
    }
    Ok((max_abs, max_relative))
}

fn begin_capture(args: &Args) -> Result<(), String> {
    let parent = args
        .capture_dir
        .parent()
        .ok_or("--capture-dir must have an existing parent")?;
    if !parent.is_dir() {
        return Err(format!(
            "--capture-dir parent is not an existing directory: {}",
            parent.display()
        ));
    }
    fs::create_dir(&args.capture_dir).map_err(|error| {
        format!(
            "refusing non-exclusive --capture-dir {}: {error}",
            args.capture_dir.display()
        )
    })?;
    let started_unix_millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock before Unix epoch: {error}"))?
        .as_millis();
    let invocation = json!({
        "schema": CAPTURE_SCHEMA,
        "status": "STARTED_QWEN80_MOE_AGGREGATE_SECOND_RESIDUAL_COMPONENT_ATTEMPT",
        "started_unix_millis": started_unix_millis,
        "mode": mode_name(args.mode),
        "manifest": args.manifest,
        "admission_current": args.admission_current,
        "router_receipt": args.router_receipt,
        "claim_boundary": {
            "deterministic_materialized_fixture_only": true,
            "not_a_complete_moe_wave_layer_token_generation_hcli_or_tps_result": true,
            "metal_requires_explicit_root_and_rawls_quiet_lease": true,
        },
    });
    write_new_atomic(
        &args.capture_dir,
        "invocation.json",
        &serde_json::to_vec_pretty(&invocation).map_err(|error| error.to_string())?,
    )
}

fn write_new_atomic(capture_dir: &Path, name: &str, contents: &[u8]) -> Result<(), String> {
    let target = capture_dir.join(name);
    let temporary = capture_dir.join(format!(".{name}.{}.tmp", process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| {
            format!(
                "cannot create capture temporary {}: {error}",
                temporary.display()
            )
        })?;
    if let Err(error) = file.write_all(contents).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "cannot durably write capture temporary {}: {error}",
            temporary.display()
        ));
    }
    drop(file);
    if let Err(error) = fs::hard_link(&temporary, &target) {
        let _ = fs::remove_file(&temporary);
        return Err(format!(
            "cannot publish new capture {} from {}: {error}",
            target.display(),
            temporary.display()
        ));
    }
    fs::remove_file(&temporary)
        .map_err(|error| format!("cannot retire temporary {}: {error}", temporary.display()))
}

fn failure_result(args: &Args, error: &str) -> Value {
    json!({
        "schema": RESULT_SCHEMA,
        "status": "REFUSED_QWEN80_MOE_AGGREGATE_SECOND_RESIDUAL_COMPONENT_ATTEMPT_ERROR",
        "mode": mode_name(args.mode),
        "error": error,
        "claim_boundary": {
            "deterministic_materialized_fixture_only": true,
            "no_cpu_or_metal_parity_is_claimed": true,
            "does_not_execute_real_ten_route_shared_or_complete_layer": true,
            "does_not_generate_tokens_expose_hcli_or_measure_tps": true,
        },
    })
}

fn finalize_capture(
    args: &Args,
    stage_result: Result<Value, String>,
) -> Result<(Value, Option<String>), String> {
    let (mut result, failure) = match stage_result {
        Ok(value) => (value, None),
        Err(error) => (failure_result(args, &error), Some(error)),
    };
    let object = result
        .as_object_mut()
        .ok_or("component result must be a JSON object")?;
    object.insert(
        "durable_capture".into(),
        json!({
            "directory": args.capture_dir,
            "invocation_file": "invocation.json",
            "stdout_file": "stdout.jsonl",
            "stderr_file": "stderr.log",
            "receipt_file": "receipt.json",
            "receipt_written_last_is_completion_marker": true,
        }),
    );
    let rendered = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
    let mut stdout = rendered.clone();
    stdout.push(b'\n');
    write_new_atomic(&args.capture_dir, "stdout.jsonl", &stdout)?;
    let stderr = failure
        .as_ref()
        .map_or_else(|| b"\n".to_vec(), |error| format!("{error}\n").into_bytes());
    write_new_atomic(&args.capture_dir, "stderr.log", &stderr)?;
    if failure.is_none() {
        write_new_atomic(&args.capture_dir, "receipt.json", &rendered)?;
    }
    Ok((result, failure))
}

fn rejection_report(route: &RouteEvidence, fixture: &AggregateInput) -> Value {
    let mut missing = fixture.clone();
    missing.already_weighted_route_deltas.pop();
    let missing_route_rejected = validate_input(&missing, route).is_err();
    let mut reordered = fixture.clone();
    reordered.accepted_ids.swap(0, 1);
    reordered.normalized_weights.swap(0, 1);
    reordered.already_weighted_route_deltas.swap(0, 1);
    let reordered_route_rejected = validate_input(&reordered, route).is_err();
    let mut unnormalized = fixture.clone();
    unnormalized.normalized_weights[0] += 0.01;
    let unnormalized_route_rejected = validate_input(&unnormalized, route).is_err();
    let mut nonfinite = fixture.clone();
    nonfinite.gated_shared[7] = f32::NAN;
    let nonfinite_component_rejected = validate_input(&nonfinite, route).is_err();
    let duplicate_ids_rejected = {
        let mut duplicate = fixture.clone();
        duplicate.accepted_ids[1] = duplicate.accepted_ids[0];
        validate_input(&duplicate, route).is_err()
    };
    json!({
        "missing_route_rejected": missing_route_rejected,
        "reordered_route_rejected": reordered_route_rejected,
        "unnormalized_route_rejected": unnormalized_route_rejected,
        "nonfinite_component_rejected": nonfinite_component_rejected,
        "duplicate_route_id_rejected": duplicate_ids_rejected,
        "wrong_route_count_rejected": TOP_K != 9,
        "wrong_hidden_geometry_rejected": HIDDEN != 4_096,
    })
}

fn all_rejections_passed(rejections: &Value) -> bool {
    rejections
        .as_object()
        .is_some_and(|object| object.values().all(|value| value == &Value::Bool(true)))
}

fn run_cpu_oracle(args: &Args) -> Result<Value, String> {
    if args.mode == Mode::Metal {
        return Err(
            "refusing Metal execution: this aggregate seam remains unregistered and requires root's explicit append-only registry authorization plus Rawls's quiet lease"
                .into(),
        );
    }
    let component = bind_current_component(args)?;
    let fixture = deterministic_materialized_fixture(&component.route);
    validate_input(&fixture, &component.route)?;
    let f32 = aggregate_fixed_f32(&fixture)?;
    let (f64_routed, f64_second) = aggregate_reference_f64(&fixture)?;
    let (routed_max_abs, routed_max_relative) =
        max_f64_error(&f32.routed_sum, &f64_routed, "fixed route0..9 routed sum")?;
    let (second_max_abs, second_max_relative) = max_f64_error(
        &f32.second_residual,
        &f64_second,
        "route0..9 + shared + first residual",
    )?;
    if routed_max_abs > ACCUMULATION_F32_F64_TOLERANCE
        || second_max_abs > ACCUMULATION_F32_F64_TOLERANCE
    {
        return Err(format!(
            "MoE aggregate f32/f64 parity failed: routed={routed_max_abs}, second={second_max_abs}, tolerance={ACCUMULATION_F32_F64_TOLERANCE}"
        ));
    }
    let rejections = rejection_report(&component.route, &fixture);
    if !all_rejections_passed(&rejections) {
        return Err("MoE aggregate rejection suite did not fail closed".into());
    }
    let route_delta_sha256 = fixture
        .already_weighted_route_deltas
        .iter()
        .map(|delta| f32_slice_sha256(delta))
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_QWEN80_MOE_ROUTE_INDEX_ORDERED_AGGREGATE_SECOND_RESIDUAL_CPU_FIXTURE_ORACLE_READY_METAL_LEASE_REQUIRED",
        "mode": mode_name(args.mode),
        "component_only": true,
        "deterministic_materialized_fixture_only": true,
        "complete_artifact_scan_performed": false,
        "direct_packed_payloads_opened": 0,
        "raw_bf16_or_safetensors_opened": false,
        "metal_device_or_dispatch_performed": false,
        "artifact_binding": {
            "manifest_path": component.manifest_path,
            "manifest_document_sha256": component.manifest_document_sha256,
            "manifest_seal_sha256": CURRENT_MANIFEST_SEAL,
            "admission_current_path": component.admission_current_path,
            "admission_pointer_seal_sha256": component.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": CURRENT_ADMISSION_RECEIPT_SEAL,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_body_audit_seal_sha256": SOURCE_BODY_AUDIT_SEAL,
            "source_revalidation_seal_sha256": SOURCE_REVALIDATION_SEAL,
            "source_model_dir": component.source_model_dir,
            "source_config_sha256": component.source_config_sha256,
            "layer": LAYER,
            "layer_kind": component.config.layer_kind(LAYER).map_err(|error| error.to_string())?.as_source_name(),
            "hidden": component.config.hidden,
            "experts": component.config.experts,
            "experts_per_token": component.config.experts_per_token,
        },
        "source_top10_evidence": {
            "router_receipt_path": component.route.receipt_path,
            "router_receipt_sha256": component.route.receipt_sha256,
            "router_fixture_post_attention_residual_sha256": ROUTER_FIXTURE_RESIDUAL_SHA256,
            "route_ids_in_accepted_source_order": component.route.ids,
            "normalized_route_weights_f32": component.route.weights,
            "normalized_route_weights_source_f64": component.route.weights_f64,
            "route_weight_sum_f32": component.route.weights.iter().sum::<f32>(),
            "route_weight_sum_source_f64": component.route.weights_f64.iter().sum::<f64>(),
        },
        "materialized_fixture": {
            "kind": "deterministic compact source-route-shaped fixture; route deltas are synthetic and already weighted; it is not a substitute for ten physical expert waves",
            "route_delta_count": fixture.already_weighted_route_deltas.len(),
            "route_delta_shape": [TOP_K, HIDDEN],
            "route_index_order": (0..TOP_K).collect::<Vec<_>>(),
            "already_weighted_route_delta_sha256": route_delta_sha256,
            "gated_shared_sha256": f32_slice_sha256(&fixture.gated_shared),
            "first_residual_sha256": f32_slice_sha256(&fixture.first_residual),
        },
        "cpu_oracle": {
            "fixed_f32_order": "for each hidden index: start zero; add route[0], route[1], ..., route[9] in order; then add gated_shared; then add first_residual",
            "routed_sum_sha256": f32_slice_sha256(&f32.routed_sum),
            "second_residual_sha256": f32_slice_sha256(&f32.second_residual),
            "route0_to9_routed_sum_f32_vs_f64_max_abs": routed_max_abs,
            "route0_to9_routed_sum_f32_vs_f64_max_relative": routed_max_relative,
            "second_residual_f32_vs_f64_max_abs": second_max_abs,
            "second_residual_f32_vs_f64_max_relative": second_max_relative,
            "tolerance_max_abs": ACCUMULATION_F32_F64_TOLERANCE,
            "all_2048_outputs_finite": f32.second_residual.iter().all(|value| value.is_finite()),
            "no_route_reweighting_or_router_execution_inside_component": true,
        },
        "rejection_tests": rejections,
        "metal_intermediate_error_ledger": {
            "performed": false,
            "reason": "No explicit root registry authorization or Rawls Qwen80 quiet lease. The isolated shader is staged but unregistered; no Metal context, compilation, or dispatch occurred.",
            "future_required_intermediates": ["route_weighted_deltas[10,2048]", "routed_sum[2048]", "gated_shared[2048]", "first_residual[2048]", "second_residual[2048]"],
            "future_acceptance": [
                "bind the accepted router receipt and reject any missing, reordered, duplicate, unnormalized, or non-finite route input before device dispatch",
                "compare every routed sum and second residual element against a fresh fixed-order CPU oracle",
                "do not use this fixture receipt as proof that all ten physical experts or a complete layer ran",
            ],
        },
        "integration_contract": {
            "rawls_hybrid_scheduler_handoff": [
                "The real scheduler supplies ten already-weighted routed deltas in exact source route-index order, one gated shared result, and the first residual. This component does not select routes or multiply route weights.",
                "Accumulate routes 0 through 9 in fixed f32 order; add shared output; then add first residual to form the second residual.",
                "A missing/reordered/unnormalized route is a hard failure, not an opportunity to silently compact or re-sort work.",
            ],
            "claim_boundary": [
                "The current admitted manifest remains LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION. This fixture parity cannot alter that status.",
                "This is a CPU-only aggregate component contract, not real ten-route execution, a full Qwen80 layer, generation, HCLI, BASE_TRUE_TPS, TG10/TG3, capability, Agent OS, or tournament evidence.",
            ],
        },
    }))
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("ascension_qwen80_moe_wave_aggregate_second_residual: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = begin_capture(&args) {
        eprintln!("ascension_qwen80_moe_wave_aggregate_second_residual: {error}");
        std::process::exit(2);
    }
    match finalize_capture(&args, run_cpu_oracle(&args)) {
        Ok((result, None)) => match serde_json::to_string_pretty(&result) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("ascension_qwen80_moe_wave_aggregate_second_residual: result print failed: {error}");
                std::process::exit(2);
            }
        },
        Ok((_result, Some(error))) => {
            eprintln!("ascension_qwen80_moe_wave_aggregate_second_residual: {error}");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("ascension_qwen80_moe_wave_aggregate_second_residual: capture finalization failed: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route() -> RouteEvidence {
        RouteEvidence {
            receipt_path: PathBuf::new(),
            receipt_sha256: String::new(),
            ids: EXPECTED_ROUTE_IDS,
            weights: [0.1; TOP_K],
            weights_f64: [0.1; TOP_K],
        }
    }

    fn fixture() -> AggregateInput {
        let route = route();
        AggregateInput {
            accepted_ids: route.ids,
            normalized_weights: route.weights,
            already_weighted_route_deltas: (0..TOP_K)
                .map(|index| vec![(index + 1) as f32; HIDDEN])
                .collect(),
            gated_shared: vec![0.5; HIDDEN],
            first_residual: vec![-0.25; HIDDEN],
        }
    }

    #[test]
    fn fixed_route_index_accumulation_then_shared_then_residual_is_exact_for_fixture() {
        let input = fixture();
        validate_input(&input, &route()).unwrap();
        let output = aggregate_fixed_f32(&input).unwrap();
        assert_eq!(output.routed_sum[0], 55.0);
        assert_eq!(output.second_residual[0], 55.25);
    }

    #[test]
    fn rejects_missing_reordered_and_unnormalized_route_inputs() {
        let expected = route();
        let mut missing = fixture();
        missing.already_weighted_route_deltas.pop();
        assert!(validate_input(&missing, &expected).is_err());
        let mut reordered = fixture();
        reordered.accepted_ids.swap(0, 1);
        assert!(validate_input(&reordered, &expected).is_err());
        let mut unnormalized = fixture();
        unnormalized.normalized_weights[0] += 0.01;
        assert!(validate_input(&unnormalized, &expected).is_err());
    }

    #[test]
    fn rejects_nonfinite_and_duplicate_route_inputs() {
        let expected = route();
        let mut nonfinite = fixture();
        nonfinite.gated_shared[4] = f32::NAN;
        assert!(validate_input(&nonfinite, &expected).is_err());
        let mut duplicate = fixture();
        duplicate.accepted_ids[3] = duplicate.accepted_ids[2];
        assert!(validate_input(&duplicate, &expected).is_err());
    }

    #[test]
    fn f32_fixed_order_matches_materialized_f64_reference() {
        let input = fixture();
        let f32 = aggregate_fixed_f32(&input).unwrap();
        let (routed, second) = aggregate_reference_f64(&input).unwrap();
        let (routed_error, _) = max_f64_error(&f32.routed_sum, &routed, "routed").unwrap();
        let (second_error, _) = max_f64_error(&f32.second_residual, &second, "second").unwrap();
        assert!(routed_error < 1.0e-6);
        assert!(second_error < 1.0e-6);
    }

    #[test]
    fn component_constants_preserve_qwen80_moe_geometry() {
        assert_eq!(LAYER, 0);
        assert_eq!(HIDDEN, 2_048);
        assert_eq!(EXPERTS, 512);
        assert_eq!(TOP_K, 10);
        assert_eq!(EXPECTED_ROUTE_IDS.len(), TOP_K);
        assert_ne!(HIDDEN, 4_096);
    }
}
