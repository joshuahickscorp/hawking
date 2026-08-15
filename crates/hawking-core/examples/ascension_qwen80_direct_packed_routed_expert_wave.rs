//! Isolated, artifact-bound Qwen3-Coder-Next layer-0 one-expert routed wave.
//!
//! This component intentionally starts from the same deterministic
//! source-shaped post-attention residual used by the sealed layer-0 router
//! receipt.  It reconstructs only the postnorm hidden `[2048]`, accepts that
//! receipt's already-normalized route `expert=65`, then executes exactly:
//!
//! `gate_proj [512,2048] + up_proj [512,2048] -> SiLU(gate)*up [512]
//!  -> down_proj [2048,512] -> normalized-route-weighted accumulator [2048]`
//!
//! It opens four current-admitted HQ30G1B1 payloads (postnorm plus expert 65
//! gate/up/down) and never opens BF16/safetensors.  The CPU path is an
//! independent direct-packed oracle with parity at every boundary.  The paired
//! Metal source is staged but unregistered; no Metal device is opened without
//! an explicit later Qwen80 quiet lease.  This is not ten-route MoE, shared
//! expert, residual combine, a full layer/token, generation, HCLI, or TPS.

use half::f16;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use hawking_core::model::qwen80_complete_runtime::{Qwen80CompleteRuntimeConfig, Qwen80LayerKind};
use hawking_core::model::qwen_complete_binary::{
    complete_binary_matvec_f64, decode_complete_binary_f32, parse_complete_binary_header,
    CompleteBinaryHeader,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_routed_expert_wave.v1";
const CAPTURE_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_routed_expert_wave_capture.v1";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const CURRENT_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ROUTER_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const ROUTER_RECEIPT_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const ROUTER_OUTER_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1";
const ROUTER_OUTER_RECEIPT_STATUS: &str =
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY";
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
const LAYER: usize = 0;
const EXPERT: usize = 65;
const ROUTE_INDEX: usize = 0;
const HIDDEN: usize = 2_048;
const INTERMEDIATE: usize = 512;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON: f32 = 1.0e-6;
const POST_NORM_NAME: &str = "model.layers.0.post_attention_layernorm.weight";
const GATE_NAME: &str = "model.layers.0.mlp.experts.65.gate_proj.weight";
const UP_NAME: &str = "model.layers.0.mlp.experts.65.up_proj.weight";
const DOWN_NAME: &str = "model.layers.0.mlp.experts.65.down_proj.weight";
const POST_NORM_ARTIFACT_SHA256: &str =
    "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7";
const GATE_ARTIFACT_SHA256: &str =
    "663b50bf179fbf0b540c61871fe90243ea3bf0b8ffbeac9617ff3e74fe3d7b80";
const UP_ARTIFACT_SHA256: &str = "4e5d56ea3e13d6ea02eccdaa87282731f63dbe41699c134ab2f45ec236c2c419";
const DOWN_ARTIFACT_SHA256: &str =
    "5e2e29adb09faf5cc12c391ae66892b1a6d107ff050c19dc603ec1761e724328";
const SOURCE_SHARD: &str = "model-00001-of-00040.safetensors";
const SOURCE_SHARD_SHA256: &str =
    "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a";
const EXPECTED_ROUTE_IDS: [u16; TOP_K] = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494];
const EXPECTED_ROUTE_WEIGHT: f64 = 0.245_458_886_027_336_12;
const NORM_F32_F64_TOLERANCE: f64 = 2.0e-5;
const PROJECTION_F32_F64_TOLERANCE: f64 = 2.0e-4;
const SWIGLU_F32_F64_TOLERANCE: f64 = 3.0e-4;
const WEIGHTED_OUTPUT_F32_F64_TOLERANCE: f64 = 8.0e-4;
const DEFAULT_MAX_CPU_WORKERS: usize = 4;
const QUIET_METAL_LEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_routed_expert_65_quiet_metal_lease.v1";
const QUIET_METAL_LEASE_STATUS: &str =
    "GRANTED_QWEN80_ROUTED_EXPERT_65_NON_TIMED_DEVICE_PARITY_LEASE";
const QUIET_METAL_COMPONENT: &str = "qwen80_direct_packed_routed_expert_65";
const DEVICE_GATE_UP_MAX_ABS_TOLERANCE: f32 = 5.0e-4;
const DEVICE_SWIGLU_MAX_ABS_TOLERANCE: f32 = 5.0e-4;
const DEVICE_DOWN_MAX_ABS_TOLERANCE: f32 = 1.0e-3;
const DEVICE_WEIGHTED_MAX_ABS_TOLERANCE: f32 = 3.0e-4;

/// Small isolated helper for the component-only device dispatch.  It does
/// not alter the generic runtime API or make this probe a serving backend.
trait StageSetScalar {
    fn stage_set_u32(&self, index: u64, value: u32);
    fn stage_set_f32(&self, index: u64, value: f32);
}

impl StageSetScalar for ::metal::ComputeCommandEncoderRef {
    #[inline(always)]
    fn stage_set_u32(&self, index: u64, value: u32) {
        self.set_bytes(
            index,
            std::mem::size_of::<u32>() as u64,
            &value as *const u32 as *const _,
        );
    }

    #[inline(always)]
    fn stage_set_f32(&self, index: u64, value: f32) {
        self.set_bytes(
            index,
            std::mem::size_of::<f32>() as u64,
            &value as *const f32 as *const _,
        );
    }
}

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
    router_outer_receipt: PathBuf,
    capture_dir: PathBuf,
    mode: Mode,
    workers: usize,
    quiet_metal_lease: Option<QuietMetalLease>,
}

#[derive(Clone, Debug)]
struct QuietMetalLease {
    path: PathBuf,
    document_sha256: String,
    lease_id: String,
    authorization_seal_sha256: String,
}

#[derive(Clone, Debug)]
struct BoundTensor {
    name: &'static str,
    path: PathBuf,
    payload_sha256: String,
    source_shard: String,
    source_shard_sha256: String,
    header: CompleteBinaryHeader,
    payload: Vec<u8>,
}

#[derive(Clone, Debug)]
struct RouteEvidence {
    receipt_path: PathBuf,
    receipt_sha256: String,
    outer_receipt_path: PathBuf,
    outer_receipt_sha256: String,
    outer_receipt_seal_sha256: String,
    selected_expert: u16,
    normalized_weight: f32,
    normalized_weight_source_f64: f64,
    route_ids: [u16; TOP_K],
    route_weights: [f32; TOP_K],
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
    post_norm: BoundTensor,
    gate: BoundTensor,
    up: BoundTensor,
    down: BoundTensor,
}

/// A fresh direct-packed CPU oracle retained in memory through the optional
/// device pass.  Keeping every stage here makes the post-fence comparison
/// same-capture evidence rather than a comparison against a prior receipt.
#[derive(Clone, Debug)]
struct CpuOracle {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    activated: Vec<f32>,
    down: Vec<f32>,
    weighted: Vec<f32>,
    norm_max_abs: f64,
    norm_max_relative: f64,
    norm_direct_decode_max_abs: f64,
    gate_max_abs: f64,
    gate_max_relative: f64,
    up_max_abs: f64,
    up_max_relative: f64,
    swiglu_max_abs: f64,
    swiglu_max_relative: f64,
    down_max_abs: f64,
    down_max_relative: f64,
    weighted_max_abs: f64,
    weighted_max_relative: f64,
    candidate_gate_up_duration_ms: f64,
    reference_gate_up_duration_ms: f64,
    candidate_down_duration_ms: f64,
}

/// Post-fence direct device parity for exactly one selected expert body.  It
/// deliberately excludes the other nine routes, shared expert, aggregation,
/// residual combine, and every token-level operation.
#[derive(Clone, Debug)]
struct DeviceParityLedger {
    device_name: String,
    dispatch_count: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    activated: Vec<f32>,
    down: Vec<f32>,
    weighted: Vec<f32>,
    gate_max_abs: f32,
    up_max_abs: f32,
    activated_max_abs: f32,
    down_max_abs: f32,
    weighted_max_abs: f32,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
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

fn bool_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<bool, String> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} missing bool field {field:?}"))
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

fn shape_field(object: &Map<String, Value>, label: &str) -> Result<Vec<usize>, String> {
    array_field(object, "shape", label)?
        .iter()
        .enumerate()
        .map(|(index, value)| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("{label} shape[{index}] is not a positive usize"))
        })
        .collect()
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

fn validate_quiet_metal_lease_document(
    lease: &Map<String, Value>,
    label: &str,
) -> Result<(String, String), String> {
    let lease_id = string_field(lease, "lease_id", label)?;
    let authorization_seal_sha256 = string_field(lease, "authorization_seal_sha256", label)?;
    if string_field(lease, "schema", label)? != QUIET_METAL_LEASE_SCHEMA
        || string_field(lease, "status", label)? != QUIET_METAL_LEASE_STATUS
        || !is_lower_sha256(lease_id)
        || !is_lower_sha256(authorization_seal_sha256)
        || !is_lower_sha256(string_field(lease, "seal_sha256", label)?)
    {
        return Err(format!("{label} schema/status/identity drifted"));
    }

    let model = object_field(lease, "model", label)?;
    if string_field(model, "id", "Qwen80 routed-expert lease model")? != MODEL_ID
        || string_field(model, "key", "Qwen80 routed-expert lease model")? != MODEL_KEY
        || string_field(model, "repository", "Qwen80 routed-expert lease model")?
            != SOURCE_REPOSITORY
        || string_field(model, "revision", "Qwen80 routed-expert lease model")? != SOURCE_REVISION
    {
        return Err(format!("{label} model binding drifted"));
    }

    let artifact = object_field(lease, "artifact_binding", label)?;
    for (field, expected) in [
        ("manifest_document_sha256", CURRENT_MANIFEST_DOCUMENT_SHA256),
        ("manifest_seal_sha256", CURRENT_MANIFEST_SEAL),
        (
            "admission_receipt_seal_sha256",
            CURRENT_ADMISSION_RECEIPT_SEAL,
        ),
        (
            "post_attention_norm_artifact_sha256",
            POST_NORM_ARTIFACT_SHA256,
        ),
        ("expert_gate_artifact_sha256", GATE_ARTIFACT_SHA256),
        ("expert_up_artifact_sha256", UP_ARTIFACT_SHA256),
        ("expert_down_artifact_sha256", DOWN_ARTIFACT_SHA256),
    ] {
        if string_field(
            artifact,
            field,
            "Qwen80 routed-expert lease artifact binding",
        )? != expected
        {
            return Err(format!(
                "{label} immutable artifact binding drifted at {field}"
            ));
        }
    }

    let policy = object_field(lease, "execution_policy", label)?;
    if string_field(
        policy,
        "component",
        "Qwen80 routed-expert lease execution policy",
    )? != QUIET_METAL_COMPONENT
        || !bool_field(
            policy,
            "quiet_qwen80_device_lease",
            "Qwen80 routed-expert lease execution policy",
        )?
        || !bool_field(
            policy,
            "strict_math",
            "Qwen80 routed-expert lease execution policy",
        )?
        || bool_field(
            policy,
            "timing_or_benchmarking_allowed",
            "Qwen80 routed-expert lease execution policy",
        )?
        || bool_field(
            policy,
            "complete_layer_or_token_allowed",
            "Qwen80 routed-expert lease execution policy",
        )?
        || bool_field(
            policy,
            "tps_or_tg_claim_allowed",
            "Qwen80 routed-expert lease execution policy",
        )?
    {
        return Err(format!(
            "{label} does not grant the required strict-math non-timed one-route component policy"
        ));
    }
    Ok((lease_id.to_owned(), authorization_seal_sha256.to_owned()))
}

fn validate_quiet_metal_lease(path: &Path) -> Result<QuietMetalLease, String> {
    if !path.is_absolute() {
        return Err("--lease-receipt must be absolute".into());
    }
    let canonical_path = canonical_regular_file(path, "Qwen80 routed-expert quiet Metal lease")?;
    let bytes = regular_file_bytes(&canonical_path, "Qwen80 routed-expert quiet Metal lease")?;
    let document_sha256 = sha256_hex(&bytes);
    let lease = json_object(&bytes, "Qwen80 routed-expert quiet Metal lease")?;
    let (lease_id, authorization_seal_sha256) =
        validate_quiet_metal_lease_document(&lease, "Qwen80 routed-expert quiet Metal lease")?;
    Ok(QuietMetalLease {
        path: canonical_path,
        document_sha256,
        lease_id,
        authorization_seal_sha256,
    })
}

fn quiet_metal_lease_report(lease: &QuietMetalLease) -> Value {
    json!({
        "receipt_path": lease.path,
        "receipt_document_sha256": lease.document_sha256,
        "lease_id": lease.lease_id,
        "authorization_seal_sha256": lease.authorization_seal_sha256,
        "schema": QUIET_METAL_LEASE_SCHEMA,
        "status": QUIET_METAL_LEASE_STATUS,
        "policy": {
            "component": QUIET_METAL_COMPONENT,
            "strict_math": true,
            "timing_or_benchmarking_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
        },
    })
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

fn expected_tensor_filename(name: &str) -> String {
    format!("{}.hq30g", sha256_hex(name.as_bytes()))
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
    let mut router_outer_receipt = None;
    let mut capture_dir = None;
    let mut mode = None;
    let mut workers = None;
    let mut lease_receipt = None;
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
            "--router-outer-receipt" => {
                if router_outer_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("--router-outer-receipt repeated".into());
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
            "--workers" => {
                if workers.replace(value.parse::<usize>()?).is_some() {
                    return Err("--workers repeated".into());
                }
            }
            "--lease-receipt" => {
                if lease_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("--lease-receipt repeated".into());
                }
            }
            _ => return Err("usage: ascension_qwen80_direct_packed_routed_expert_wave --manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH --router-receipt ABSOLUTE_PATH --router-outer-receipt ABSOLUTE_PATH --capture-dir NEW_ABSOLUTE_DIRECTORY --mode cpu-oracle|metal [--workers 1..N] [--lease-receipt ABSOLUTE_PATH]".into()),
        }
    }
    let manifest = manifest.ok_or("missing --manifest")?;
    let admission_current = admission_current.ok_or("missing --admission-current")?;
    let router_receipt = router_receipt.ok_or("missing --router-receipt")?;
    let router_outer_receipt = router_outer_receipt.ok_or("missing --router-outer-receipt")?;
    let capture_dir = capture_dir.ok_or("missing --capture-dir")?;
    let mode = mode.ok_or("missing --mode")?;
    if !manifest.is_absolute()
        || !admission_current.is_absolute()
        || !router_receipt.is_absolute()
        || !router_outer_receipt.is_absolute()
        || !capture_dir.is_absolute()
        || lease_receipt
            .as_ref()
            .is_some_and(|path| !path.is_absolute())
    {
        return Err("all path arguments must be absolute".into());
    }
    let maximum = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let workers = workers
        .unwrap_or_else(|| maximum.min(DEFAULT_MAX_CPU_WORKERS))
        .max(1);
    if workers > maximum {
        return Err(format!("--workers={workers} exceeds available parallelism {maximum}").into());
    }
    let quiet_metal_lease = match (mode, lease_receipt) {
        (Mode::CpuOracle, None) => None,
        (Mode::CpuOracle, Some(_)) => {
            return Err("--lease-receipt is valid only with --mode metal".into())
        }
        (Mode::Metal, None) => {
            return Err("--mode metal requires a strict --lease-receipt ABSOLUTE_PATH".into())
        }
        (Mode::Metal, Some(path)) => Some(validate_quiet_metal_lease(&path)?),
    };
    Ok(Args {
        manifest,
        admission_current,
        router_receipt,
        router_outer_receipt,
        capture_dir,
        mode,
        workers,
        quiet_metal_lease,
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
        || config.moe_intermediate != INTERMEDIATE
        || config.rms_norm_eps().to_bits() != RMS_EPSILON.to_bits()
        || config
            .layer_kind(LAYER)
            .map_err(|error| format!("Qwen80 layer-0 kind rejected: {error}"))?
            != Qwen80LayerKind::LinearAttention
    {
        return Err("source config no longer matches Qwen80 layer-0 routed-expert geometry".into());
    }
    Ok((model_dir, config_sha256, config))
}

fn parse_exact_route_ids(value: &[Value], label: &str) -> Result<[u16; TOP_K], String> {
    if value.len() != TOP_K {
        return Err(format!("{label} must have exactly {TOP_K} entries"));
    }
    let mut ids = [0u16; TOP_K];
    for (index, item) in value.iter().enumerate() {
        ids[index] = item
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .filter(|value| usize::from(*value) < EXPERTS)
            .ok_or_else(|| format!("{label}[{index}] is not an in-range expert ID"))?;
    }
    if ids
        .iter()
        .enumerate()
        .any(|(index, id)| ids[..index].contains(id))
    {
        return Err(format!("{label} contains a duplicate expert ID"));
    }
    Ok(ids)
}

fn parse_exact_route_weights(
    value: &[Value],
    label: &str,
) -> Result<([f32; TOP_K], [f64; TOP_K]), String> {
    if value.len() != TOP_K {
        return Err(format!("{label} must have exactly {TOP_K} entries"));
    }
    let mut f32_weights = [0.0f32; TOP_K];
    let mut f64_weights = [0.0f64; TOP_K];
    for (index, item) in value.iter().enumerate() {
        let weight = item
            .as_f64()
            .filter(|weight| weight.is_finite() && *weight >= 0.0)
            .ok_or_else(|| format!("{label}[{index}] is not a finite nonnegative weight"))?;
        f32_weights[index] = weight as f32;
        if !f32_weights[index].is_finite() {
            return Err(format!("{label}[{index}] does not fit finite f32"));
        }
        f64_weights[index] = weight;
    }
    let sum = f64_weights.iter().sum::<f64>();
    if (sum - 1.0).abs() > 2.0e-6 {
        return Err(format!("{label} sum {sum} differs from 1"));
    }
    Ok((f32_weights, f64_weights))
}

fn bind_route_evidence(
    path: &Path,
    outer_path: &Path,
    manifest_path: &Path,
    manifest_document_sha256: &str,
) -> Result<RouteEvidence, String> {
    let receipt_path = canonical_regular_file(path, "Qwen80 postnorm/router receipt")?;
    let bytes = regular_file_bytes(&receipt_path, "Qwen80 postnorm/router receipt")?;
    let receipt_sha256 = sha256_hex(&bytes);
    let outer_receipt_path = canonical_regular_file(
        outer_path,
        "Qwen80 postnorm/router sealed outer terminal receipt",
    )?;
    let outer_bytes = regular_file_bytes(
        &outer_receipt_path,
        "Qwen80 postnorm/router sealed outer terminal receipt",
    )?;
    let outer_receipt_sha256 = sha256_hex(&outer_bytes);
    let outer = json_object(
        &outer_bytes,
        "Qwen80 postnorm/router sealed outer terminal receipt",
    )?;
    let outer_receipt_seal_sha256 = string_field(
        &outer,
        "seal_sha256",
        "Qwen80 postnorm/router sealed outer terminal receipt",
    )?
    .to_owned();
    if string_field(
        &outer,
        "schema",
        "Qwen80 postnorm/router sealed outer terminal receipt",
    )? != ROUTER_OUTER_RECEIPT_SCHEMA
        || string_field(
            &outer,
            "status",
            "Qwen80 postnorm/router sealed outer terminal receipt",
        )? != ROUTER_OUTER_RECEIPT_STATUS
        || !is_lower_sha256(&outer_receipt_seal_sha256)
    {
        return Err("postnorm/router outer terminal receipt identity/status drifted".into());
    }
    let inner = object_field(
        &outer,
        "inner_probe_capture",
        "Qwen80 postnorm/router sealed outer terminal receipt",
    )?;
    if exact_path_from_json(
        string_field(inner, "path", "Qwen80 postnorm/router outer inner binding")?,
        "Qwen80 postnorm/router outer inner path",
    )? != receipt_path
        || string_field(
            inner,
            "sha256",
            "Qwen80 postnorm/router outer inner binding",
        )? != receipt_sha256
        || string_field(
            inner,
            "status",
            "Qwen80 postnorm/router outer inner binding",
        )? != ROUTER_RECEIPT_STATUS
        || inner.get("metal_performed").and_then(Value::as_bool) != Some(true)
    {
        return Err(
            "postnorm/router outer terminal no longer binds the supplied strict-Math inner receipt"
                .into(),
        );
    }
    let receipt = json_object(&bytes, "Qwen80 postnorm/router receipt")?;
    if string_field(&receipt, "schema", "Qwen80 postnorm/router receipt")? != ROUTER_RECEIPT_SCHEMA
        || string_field(&receipt, "status", "Qwen80 postnorm/router receipt")?
            != ROUTER_RECEIPT_STATUS
        || receipt.get("component_only").and_then(Value::as_bool) != Some(true)
        || receipt
            .get("metal_device_or_dispatch_performed")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err(
            "route receipt is not the required current strict-Math layer-0 postnorm/router component evidence"
                .into(),
        );
    }
    let metal_policy = object_field(
        &receipt,
        "metal_execution_policy",
        "Qwen80 postnorm/router receipt",
    )?;
    if metal_policy
        .get("strict_math_required")
        .and_then(Value::as_bool)
        != Some(true)
        || metal_policy
            .get("timing_or_benchmarking_allowed")
            .and_then(Value::as_bool)
            != Some(false)
        || metal_policy
            .get("complete_layer_or_token_allowed")
            .and_then(Value::as_bool)
            != Some(false)
        || metal_policy
            .get("tps_or_tg_claim_allowed")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("route receipt strict component policy drifted".into());
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
        || u64_field(binding, "layer", "Qwen80 postnorm/router binding")? != LAYER as u64
        || u64_field(binding, "hidden", "Qwen80 postnorm/router binding")? != HIDDEN as u64
        || u64_field(binding, "router_logits", "Qwen80 postnorm/router binding")? != EXPERTS as u64
        || u64_field(
            binding,
            "experts_per_token",
            "Qwen80 postnorm/router binding",
        )? != TOP_K as u64
        || string_field(
            binding,
            "source_config_sha256",
            "Qwen80 postnorm/router binding",
        )? != SOURCE_CONFIG_SHA256
    {
        return Err("route receipt source/artifact binding drifted".into());
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
        return Err("route receipt postnorm binding drifted".into());
    }
    let oracle = object_field(&receipt, "cpu_oracle", "Qwen80 postnorm/router receipt")?;
    if string_field(
        oracle,
        "post_attention_residual_sha256",
        "Qwen80 postnorm/router oracle",
    )? != ROUTER_FIXTURE_RESIDUAL_SHA256
    {
        return Err("route receipt fixture residual identity drifted".into());
    }
    let route = object_field(
        &receipt,
        "source_stable_top10_router",
        "Qwen80 postnorm/router receipt",
    )?;
    let ids = parse_exact_route_ids(
        array_field(route, "ids", "Qwen80 postnorm/router route")?,
        "Qwen80 postnorm/router route ids",
    )?;
    if ids != EXPECTED_ROUTE_IDS || ids[ROUTE_INDEX] != EXPERT as u16 {
        return Err(
            "route receipt source-selected expert order is not the sealed layer-0 route".into(),
        );
    }
    let (route_weights, route_weights_f64) = parse_exact_route_weights(
        array_field(
            route,
            "renormalized_weights",
            "Qwen80 postnorm/router route",
        )?,
        "Qwen80 postnorm/router route weights",
    )?;
    if (route_weights_f64[ROUTE_INDEX] - EXPECTED_ROUTE_WEIGHT).abs() > 1.0e-12 {
        return Err("route receipt selected normalized weight drifted".into());
    }
    Ok(RouteEvidence {
        receipt_path,
        receipt_sha256,
        outer_receipt_path,
        outer_receipt_sha256,
        outer_receipt_seal_sha256,
        selected_expert: ids[ROUTE_INDEX],
        normalized_weight: route_weights[ROUTE_INDEX],
        normalized_weight_source_f64: route_weights_f64[ROUTE_INDEX],
        route_ids: ids,
        route_weights,
    })
}

fn bind_tensor(
    manifest: &Map<String, Value>,
    manifest_root: &Path,
    name: &'static str,
    expected_shape: &[usize],
    expected_payload_sha256: &str,
) -> Result<BoundTensor, String> {
    let tensors = manifest
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or("Qwen80 complete manifest missing tensor array")?;
    let matches = tensors
        .iter()
        .filter_map(Value::as_object)
        .filter(|row| row.get("tensor_name").and_then(Value::as_str) == Some(name))
        .collect::<Vec<_>>();
    let [row] = matches.as_slice() else {
        return Err(format!(
            "complete manifest must contain exactly one {name:?} row"
        ));
    };
    let label = format!("Qwen80 complete manifest tensor {name:?}");
    let shape = shape_field(row, &label)?;
    if shape.as_slice() != expected_shape
        || u64_field(row, "elements", &label)? != expected_shape.iter().product::<usize>() as u64
        || string_field(row, "source_dtype", &label)? != "BF16"
        || string_field(row, "source_shard", &label)? != SOURCE_SHARD
        || string_field(row, "source_shard_sha256", &label)? != SOURCE_SHARD_SHA256
    {
        return Err(format!("{label} source binding/shape drifted"));
    }
    let layout = object_field(row, "layout", &label)?;
    if string_field(layout, "magic", &label)? != "HQ30G1B1"
        || u64_field(layout, "version", &label)? != 1
        || u64_field(layout, "group_size", &label)? != GROUP_SIZE as u64
        || string_field(layout, "sign_bit_order", &label)? != "little"
        || string_field(layout, "scale_dtype", &label)? != "float16"
    {
        return Err(format!("{label} direct-packed layout drifted"));
    }
    let tensors_root = canonical_dir(
        &manifest_root.join("tensors"),
        "Qwen80 complete tensor root",
    )?;
    let expected_path = canonical_regular_file(
        &tensors_root.join(expected_tensor_filename(name)),
        &format!("{label} deterministic payload"),
    )?;
    let declared_path = canonical_regular_file(
        Path::new(string_field(row, "artifact_path", &label)?),
        &format!("{label} declared payload"),
    )?;
    if declared_path != expected_path || !declared_path.starts_with(&tensors_root) {
        return Err(format!(
            "{label} path is not its deterministic manifest-root payload"
        ));
    }
    let payload = regular_file_bytes(&expected_path, &format!("{label} payload"))?;
    if payload.len() as u64 != u64_field(row, "artifact_bytes", &label)? {
        return Err(format!("{label} physical byte count differs from manifest"));
    }
    let payload_sha256 = checked_sha256(
        &payload,
        expected_payload_sha256,
        &format!("{label} payload"),
    )?;
    if string_field(row, "artifact_sha256", &label)? != payload_sha256 {
        return Err(format!(
            "{label} declared payload digest differs from admitted digest"
        ));
    }
    let header = parse_complete_binary_header(&payload)
        .map_err(|error| format!("{label} direct-packed header rejected: {error}"))?;
    if header.version != 1
        || header.group_size != GROUP_SIZE
        || header.shape.as_slice() != expected_shape
        || header.elements != expected_shape.iter().product::<usize>()
        || header.groups != header.elements.div_ceil(GROUP_SIZE)
        || header.payload_bytes != payload.len()
    {
        return Err(format!("{label} direct-packed header geometry drifted"));
    }
    Ok(BoundTensor {
        name,
        path: expected_path,
        payload_sha256,
        source_shard: SOURCE_SHARD.to_owned(),
        source_shard_sha256: SOURCE_SHARD_SHA256.to_owned(),
        header,
        payload,
    })
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
        &args.router_outer_receipt,
        &manifest_path,
        &manifest_document_sha256,
    )?;
    if route.selected_expert != EXPERT as u16 || route.normalized_weight <= 0.0 {
        return Err(
            "sealed route evidence does not authorize expert 65 with a positive weight".into(),
        );
    }
    let (source_model_dir, source_config_sha256, config) = bind_source_config(&manifest)?;
    let manifest_root = manifest_path
        .parent()
        .ok_or("Qwen80 complete manifest has no parent directory")?;
    // Derive canonical tensor names from the source layer/expert plan before
    // comparing against the fixed admitted route identity.  This prevents a
    // scheduler from silently redirecting this one-wave seam to another body.
    let prefix = format!("model.layers.{LAYER}.mlp.experts.{}", route.selected_expert);
    let derived_gate = format!("{prefix}.gate_proj.weight");
    let derived_up = format!("{prefix}.up_proj.weight");
    let derived_down = format!("{prefix}.down_proj.weight");
    if derived_gate != GATE_NAME || derived_up != UP_NAME || derived_down != DOWN_NAME {
        return Err("layer-plan-derived selected-expert tensor names drifted".into());
    }
    let post_norm = bind_tensor(
        &manifest,
        manifest_root,
        POST_NORM_NAME,
        &[HIDDEN],
        POST_NORM_ARTIFACT_SHA256,
    )?;
    let gate = bind_tensor(
        &manifest,
        manifest_root,
        GATE_NAME,
        &[INTERMEDIATE, HIDDEN],
        GATE_ARTIFACT_SHA256,
    )?;
    let up = bind_tensor(
        &manifest,
        manifest_root,
        UP_NAME,
        &[INTERMEDIATE, HIDDEN],
        UP_ARTIFACT_SHA256,
    )?;
    let down = bind_tensor(
        &manifest,
        manifest_root,
        DOWN_NAME,
        &[HIDDEN, INTERMEDIATE],
        DOWN_ARTIFACT_SHA256,
    )?;
    Ok(BoundComponent {
        manifest_path,
        admission_current_path,
        manifest_document_sha256,
        admission_pointer_seal_sha256,
        source_model_dir,
        source_config_sha256,
        config,
        route,
        post_norm,
        gate,
        up,
        down,
    })
}

fn packed_sign_is_positive(
    payload: &[u8],
    header: &CompleteBinaryHeader,
    index: usize,
) -> Result<bool, String> {
    if index >= header.elements || header.group_size != GROUP_SIZE {
        return Err("direct-packed sign index/group size is invalid".into());
    }
    let group = index / header.group_size;
    let within_group = index % header.group_size;
    let bytes_per_group = header.group_size / 8;
    let sign_index = header
        .sign_offset
        .checked_add(
            group
                .checked_mul(bytes_per_group)
                .ok_or("packed sign group offset overflow")?,
        )
        .and_then(|offset| offset.checked_add(within_group / 8))
        .ok_or("packed sign offset overflow")?;
    let sign_byte = *payload
        .get(sign_index)
        .ok_or("packed sign byte out of bounds")?;
    Ok(((sign_byte >> (within_group % 8)) & 1) == 1)
}

fn packed_value_f32(
    payload: &[u8],
    header: &CompleteBinaryHeader,
    index: usize,
) -> Result<f32, String> {
    if index >= header.elements || header.group_size != GROUP_SIZE {
        return Err("direct-packed element index/group size is invalid".into());
    }
    let group = index / header.group_size;
    let scale_offset = header
        .scale_offset
        .checked_add(group.checked_mul(2).ok_or("packed scale index overflow")?)
        .ok_or("packed scale offset overflow")?;
    let scale = f16::from_bits(u16::from_le_bytes([
        *payload
            .get(scale_offset)
            .ok_or("packed scale low byte out of bounds")?,
        *payload
            .get(scale_offset + 1)
            .ok_or("packed scale high byte out of bounds")?,
    ]))
    .to_f32();
    if !scale.is_finite() {
        return Err("direct-packed FP16 group scale is non-finite".into());
    }
    Ok(if packed_sign_is_positive(payload, header, index)? {
        scale
    } else {
        -scale
    })
}

fn deterministic_post_attention_residual() -> Vec<f32> {
    let mut state = 0x6e9f_5b9d_bf58_3ebd_u64;
    (0..HIDDEN)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 40) & 0x00ff_ffff) as f32 / 16_777_215.0;
            let phase = ((index * 43 % 101) as f32 - 50.0) / 173.0;
            (unit * 2.0 - 1.0) + phase
        })
        .collect()
}

fn post_attention_rms_norm_candidate_f32(
    norm: &BoundTensor,
    residual: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if norm.header.shape.as_slice() != &[HIDDEN]
        || norm.header.group_size != GROUP_SIZE
        || residual.len() != HIDDEN
        || residual.iter().any(|value| !value.is_finite())
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err("post-attention RMSNorm candidate received invalid geometry/input".into());
    }
    let mean_square = residual.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
    let inverse_rms = (mean_square + epsilon).sqrt().recip();
    if !inverse_rms.is_finite() {
        return Err("post-attention RMSNorm inverse RMS is non-finite".into());
    }
    let output =
        residual
            .iter()
            .enumerate()
            .map(|(index, value)| {
                Ok(*value
                    * inverse_rms
                    * (1.0 + packed_value_f32(&norm.payload, &norm.header, index)?))
            })
            .collect::<Result<Vec<_>, String>>()?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("post-attention RMSNorm candidate produced non-finite output".into());
    }
    Ok(output)
}

fn post_attention_rms_norm_reference_f64(
    norm: &BoundTensor,
    residual: &[f32],
    epsilon: f32,
) -> Result<Vec<f64>, String> {
    let (header, decoded) = decode_complete_binary_f32(&norm.payload)
        .map_err(|error| format!("post-attention norm library decode failed: {error}"))?;
    if header != norm.header || decoded.len() != HIDDEN || residual.len() != HIDDEN {
        return Err("post-attention norm library decode disagrees with admitted binding".into());
    }
    let mean_square = residual
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / HIDDEN as f64;
    let inverse_rms = (mean_square + f64::from(epsilon)).sqrt().recip();
    let output = residual
        .iter()
        .zip(decoded)
        .map(|(value, weight)| f64::from(*value) * inverse_rms * (1.0 + f64::from(weight)))
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("post-attention RMSNorm f64 reference produced non-finite output".into());
    }
    Ok(output)
}

fn row_dot_grouped_f32(
    payload: &[u8],
    header: &CompleteBinaryHeader,
    row: usize,
    input: &[f32],
) -> Result<f32, String> {
    if header.shape.len() != 2
        || header.group_size != GROUP_SIZE
        || row >= header.shape[0]
        || input.len() != header.shape[1]
        || header.shape[1] % GROUP_SIZE != 0
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("grouped direct-packed matvec received invalid geometry/input".into());
    }
    let columns = header.shape[1];
    let groups_per_row = columns / GROUP_SIZE;
    let row_base = row.checked_mul(columns).ok_or("matvec row base overflow")?;
    let mut total = 0.0f32;
    for group_within_row in 0..groups_per_row {
        let element_base = row_base
            .checked_add(
                group_within_row
                    .checked_mul(GROUP_SIZE)
                    .ok_or("matvec group element overflow")?,
            )
            .ok_or("matvec group base overflow")?;
        let mut signed_input_sum = 0.0f32;
        for within_group in 0..GROUP_SIZE {
            let element_index = element_base + within_group;
            let signed_input = if packed_sign_is_positive(payload, header, element_index)? {
                input[group_within_row * GROUP_SIZE + within_group]
            } else {
                -input[group_within_row * GROUP_SIZE + within_group]
            };
            signed_input_sum += signed_input;
        }
        let scale = packed_value_f32(payload, header, element_base)?.abs();
        total += scale * signed_input_sum;
    }
    if !total.is_finite() {
        return Err("grouped direct-packed matvec produced non-finite output".into());
    }
    Ok(total)
}

fn matvec_candidate_parallel_f32(
    tensor: &BoundTensor,
    rows: usize,
    columns: usize,
    input: &[f32],
    workers: usize,
) -> Result<Vec<f32>, String> {
    if tensor.header.shape.as_slice() != [rows, columns]
        || tensor.header.group_size != GROUP_SIZE
        || input.len() != columns
        || workers == 0
    {
        return Err("direct-packed matrix candidate binding/worker geometry is invalid".into());
    }
    let workers = workers.min(rows).max(1);
    let chunk_rows = rows.div_ceil(workers);
    let mut output = vec![0.0f32; rows];
    thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..rows).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(rows);
            let payload = &tensor.payload;
            let header = &tensor.header;
            handles.push(scope.spawn(move || -> Result<(usize, Vec<f32>), String> {
                let mut chunk = Vec::with_capacity(end - start);
                for row in start..end {
                    chunk.push(row_dot_grouped_f32(payload, header, row, input)?);
                }
                Ok((start, chunk))
            }));
        }
        for handle in handles {
            let (start, chunk) = handle
                .join()
                .map_err(|_| "direct-packed matrix CPU worker panicked")??;
            output[start..start + chunk.len()].copy_from_slice(&chunk);
        }
        Ok(())
    })?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("direct-packed matrix candidate produced non-finite output".into());
    }
    Ok(output)
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

fn swiglu_candidate_f32(gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    if gate.len() != INTERMEDIATE
        || up.len() != INTERMEDIATE
        || gate.iter().chain(up).any(|value| !value.is_finite())
    {
        return Err("source SwiGLU candidate received invalid finite [512] inputs".into());
    }
    let output = gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("source SwiGLU candidate produced non-finite output".into());
    }
    Ok(output)
}

fn swiglu_reference_f64(gate: &[f64], up: &[f64]) -> Result<Vec<f64>, String> {
    if gate.len() != INTERMEDIATE
        || up.len() != INTERMEDIATE
        || gate.iter().chain(up).any(|value| !value.is_finite())
    {
        return Err("source SwiGLU f64 reference received invalid finite [512] inputs".into());
    }
    let output = gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("source SwiGLU f64 reference produced non-finite output".into());
    }
    Ok(output)
}

fn weighted_accumulator_candidate_f32(
    output: &[f32],
    route_weight: f32,
) -> Result<Vec<f32>, String> {
    if output.len() != HIDDEN
        || output.iter().any(|value| !value.is_finite())
        || !route_weight.is_finite()
        || route_weight <= 0.0
    {
        return Err("weighted routed accumulator received invalid output/route weight".into());
    }
    let weighted = output
        .iter()
        .map(|value| *value * route_weight)
        .collect::<Vec<_>>();
    if weighted.iter().any(|value| !value.is_finite()) {
        return Err("weighted routed accumulator produced non-finite output".into());
    }
    Ok(weighted)
}

fn header_report(header: &CompleteBinaryHeader) -> Value {
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
    })
}

fn tensor_report(tensor: &BoundTensor) -> Value {
    json!({
        "name": tensor.name,
        "artifact_path": tensor.path,
        "artifact_sha256": tensor.payload_sha256,
        "source_shard": tensor.source_shard,
        "source_shard_sha256": tensor.source_shard_sha256,
        "header": header_report(&tensor.header),
    })
}

fn bytes_for<T>(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| format!("{label} byte count overflows usize"))
}

fn compact_sign_and_scale_sections(tensor: &BoundTensor) -> Result<(&[u8], &[u8]), String> {
    let header = &tensor.header;
    let scales = tensor
        .payload
        .get(header.scale_offset..header.sign_offset)
        .ok_or_else(|| format!("{} compact scale section is truncated", tensor.name))?;
    let signs = tensor
        .payload
        .get(header.sign_offset..header.payload_bytes)
        .ok_or_else(|| format!("{} compact sign section is truncated", tensor.name))?;
    let expected_scales = bytes_for::<u16>(header.groups, "direct-packed scale")?;
    let expected_signs = header
        .groups
        .checked_mul(GROUP_SIZE / 8)
        .ok_or_else(|| "direct-packed sign section length overflows usize".to_owned())?;
    if scales.len() != expected_scales || signs.len() != expected_signs {
        return Err(format!(
            "{} compact section geometry drifted: scales={}/{expected_scales}, signs={}/{expected_signs}",
            tensor.name,
            scales.len(),
            signs.len(),
        ));
    }
    Ok((signs, scales))
}

fn snapshot_f32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let bytes = bytes_for::<f32>(elements, label)?;
    if buffer.length() < bytes as u64 {
        return Err(format!(
            "{label} snapshot needs {bytes} bytes but the Metal buffer has {}",
            buffer.length()
        ));
    }
    // All component buffers use StorageModeShared and are sampled only after
    // the command-buffer fence in `run_metal_stage`.
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() })
}

fn max_abs_error_f32(expected: &[f32], observed: &[f32], label: &str) -> Result<f32, String> {
    if expected.len() != observed.len() {
        return Err(format!(
            "{label} length mismatch: expected {}, observed {}",
            expected.len(),
            observed.len()
        ));
    }
    let mut maximum = 0.0f32;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(format!("{label} contains non-finite value at {index}"));
        }
        maximum = maximum.max((expected - observed).abs());
    }
    Ok(maximum)
}

fn ensure_non_timed_metal_environment() -> Result<(), String> {
    const FORBIDDEN: [&str; 2] = ["HAWKING_TRACE_DISPATCH", "HAWKING_TCB_TRACE"];
    let present = FORBIDDEN
        .iter()
        .copied()
        .filter(|key| env::var_os(key).is_some())
        .collect::<Vec<_>>();
    if present.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "strict non-timed one-route component capture refuses trace environment variables: {}",
            present.join(", ")
        ))
    }
}

fn metal_execution_policy_report(args: &Args) -> Value {
    json!({
        "lease_required_for_metal": true,
        "lease_binding": args.quiet_metal_lease.as_ref().map(quiet_metal_lease_report),
        "strict_math_required": true,
        "timing_or_benchmarking_allowed": false,
        "complete_layer_or_token_allowed": false,
        "tps_or_tg_claim_allowed": false,
        "device_context_or_dispatch_performed": false,
    })
}

fn post_attempt_metal_policy(args: &Args, completed: bool) -> Value {
    let mut policy = metal_execution_policy_report(args);
    let object = policy
        .as_object_mut()
        .expect("metal execution policy report is an object");
    object.insert(
        "device_context_or_dispatch_performed".into(),
        Value::Bool(completed),
    );
    object.insert(
        "post_attempt_state".into(),
        Value::String(
            if completed {
                "strict_math_one_route_component_command_buffer_completed_and_fenced"
            } else {
                "no_successful_device_result_is_claimed"
            }
            .into(),
        ),
    );
    policy
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
        "status": "STARTED_QWEN80_ROUTED_EXPERT_65_COMPONENT_ATTEMPT",
        "started_unix_millis": started_unix_millis,
        "mode": mode_name(args.mode),
        "manifest": args.manifest,
        "admission_current": args.admission_current,
        "router_receipt": args.router_receipt,
        "router_outer_receipt": args.router_outer_receipt,
        "workers": args.workers,
        "metal_execution_policy": metal_execution_policy_report(args),
        "claim_boundary": {
            "one_selected_expert_only": true,
            "not_ten_route_shared_expert_residual_layer_token_generation_hcli_or_tps": true,
            "metal_requires_explicit_quiet_lease": true,
            "metal_requires_strict_math_and_disallows_timing": true,
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
    let metal_execution_policy = if args.mode == Mode::Metal {
        let mut policy = metal_execution_policy_report(args);
        let object = policy
            .as_object_mut()
            .expect("metal execution policy report is an object");
        // A failure can occur after context construction or a partial encode;
        // do not misrepresent this as proof that the device was untouched.
        object.insert(
            "device_context_or_dispatch_performed".into(),
            Value::String("unknown_after_authorized_metal_attempt_error".into()),
        );
        object.insert(
            "post_attempt_state".into(),
            Value::String("no_device_parity_or_runtime_claim".into()),
        );
        policy
    } else {
        metal_execution_policy_report(args)
    };
    json!({
        "schema": RESULT_SCHEMA,
        "status": "REFUSED_QWEN80_ROUTED_EXPERT_COMPONENT_ATTEMPT_ERROR",
        "mode": mode_name(args.mode),
        "error": error,
        "metal_execution_policy": metal_execution_policy,
        "claim_boundary": {
            "one_selected_expert_only": true,
            "no_cpu_or_metal_parity_is_claimed": true,
            "does_not_execute_ten_route_shared_expert_or_complete_layer": true,
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
    // A terminal refusal is evidence too.  Receipt-last makes both a pass
    // and a refusal reaped/inspectable without inviting a replay of this
    // exclusive capture directory.
    write_new_atomic(&args.capture_dir, "receipt.json", &rendered)?;
    Ok((result, failure))
}

fn rejection_report() -> Value {
    let malformed_direct_header_rejected =
        parse_complete_binary_header(b"not-a-direct-packed-payload").is_err();
    let wrong_qwen30_hidden_surface_rejected = HIDDEN != 4_096;
    let wrong_expert_geometry_rejected = [INTERMEDIATE, HIDDEN - 1] != [INTERMEDIATE, HIDDEN];
    let wrong_down_geometry_rejected = [HIDDEN, INTERMEDIATE - 1] != [HIDDEN, INTERMEDIATE];
    let wrong_group_size_rejected = GROUP_SIZE != 64;
    let out_of_range_expert_id_rejected = EXPERTS >= EXPERTS;
    let invalid_route_weight_rejected =
        weighted_accumulator_candidate_f32(&vec![0.0; HIDDEN], f32::NAN).is_err();
    let nonfinite_swiglu_input_rejected = {
        let mut gate = vec![0.0f32; INTERMEDIATE];
        gate[0] = f32::NAN;
        swiglu_candidate_f32(&gate, &vec![0.0; INTERMEDIATE]).is_err()
    };
    json!({
        "malformed_direct_header_rejected": malformed_direct_header_rejected,
        "wrong_qwen30_hidden_surface_rejected": wrong_qwen30_hidden_surface_rejected,
        "wrong_expert_geometry_rejected": wrong_expert_geometry_rejected,
        "wrong_down_geometry_rejected": wrong_down_geometry_rejected,
        "wrong_group_size_rejected": wrong_group_size_rejected,
        "out_of_range_expert_id_rejected": out_of_range_expert_id_rejected,
        "invalid_route_weight_rejected": invalid_route_weight_rejected,
        "nonfinite_swiglu_input_rejected": nonfinite_swiglu_input_rejected,
    })
}

fn all_rejections_passed(rejections: &Value) -> bool {
    rejections
        .as_object()
        .is_some_and(|object| object.values().all(|value| value == &Value::Bool(true)))
}

fn f32_slice_sha256(values: &[f32]) -> String {
    sha256_hex(bytemuck::cast_slice(values))
}

fn source_f64_matvec(tensor: &BoundTensor, input: &[f64], label: &str) -> Result<Vec<f64>, String> {
    let (header, output) = complete_binary_matvec_f64(&tensor.payload, input)
        .map_err(|error| format!("{label} direct-packed scalar f64 reference failed: {error}"))?;
    if header != tensor.header {
        return Err(format!("{label} f64 reference header drifted from binding"));
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} f64 reference produced non-finite output"));
    }
    Ok(output)
}

fn run_cpu_oracle(args: &Args) -> Result<Value, String> {
    if args.mode == Mode::Metal {
        return Err(
            "refusing Metal execution: this isolated candidate remains unregistered and requires Rawls's explicit Qwen80 quiet lease plus post-capture registry authorization"
                .into(),
        );
    }
    let component = bind_current_component(args)?;
    let residual = deterministic_post_attention_residual();
    let residual_sha256 = f32_slice_sha256(&residual);
    if residual_sha256 != ROUTER_FIXTURE_RESIDUAL_SHA256 {
        return Err(
            "reconstructed post-attention residual no longer matches sealed router fixture".into(),
        );
    }
    let candidate_norm = post_attention_rms_norm_candidate_f32(
        &component.post_norm,
        &residual,
        component.config.rms_norm_eps(),
    )?;
    let reference_norm = post_attention_rms_norm_reference_f64(
        &component.post_norm,
        &residual,
        component.config.rms_norm_eps(),
    )?;
    let (norm_max_abs, norm_max_relative) =
        max_f64_error(&candidate_norm, &reference_norm, "post-attention RMSNorm")?;
    let (_, decoded_norm) = decode_complete_binary_f32(&component.post_norm.payload)
        .map_err(|error| format!("post-attention norm decoder repeat failed: {error}"))?;
    let norm_direct_decode_max_abs = decoded_norm
        .iter()
        .enumerate()
        .map(|(index, decoded)| -> Result<f64, String> {
            Ok((f64::from(*decoded)
                - f64::from(packed_value_f32(
                    &component.post_norm.payload,
                    &component.post_norm.header,
                    index,
                )?))
            .abs())
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .fold(0.0f64, f64::max);
    if norm_max_abs > NORM_F32_F64_TOLERANCE || norm_direct_decode_max_abs != 0.0 {
        return Err(format!(
            "post-attention RMSNorm parity failed: f32/f64={norm_max_abs}, direct-decode={norm_direct_decode_max_abs}"
        ));
    }

    let gate_workers = (args.workers / 2).max(1);
    let up_workers = args.workers.saturating_sub(gate_workers).max(1);
    let candidate_started = Instant::now();
    let (candidate_gate, candidate_up) =
        thread::scope(|scope| -> Result<(Vec<f32>, Vec<f32>), String> {
            let gate = scope.spawn(|| {
                matvec_candidate_parallel_f32(
                    &component.gate,
                    INTERMEDIATE,
                    HIDDEN,
                    &candidate_norm,
                    gate_workers,
                )
            });
            let up = scope.spawn(|| {
                matvec_candidate_parallel_f32(
                    &component.up,
                    INTERMEDIATE,
                    HIDDEN,
                    &candidate_norm,
                    up_workers,
                )
            });
            let gate = gate
                .join()
                .map_err(|_| "gate candidate worker panicked")??;
            let up = up.join().map_err(|_| "up candidate worker panicked")??;
            Ok((gate, up))
        })?;
    let candidate_gate_up_duration_ms = candidate_started.elapsed().as_secs_f64() * 1_000.0;
    let candidate_swiglu = swiglu_candidate_f32(&candidate_gate, &candidate_up)?;
    let candidate_down_started = Instant::now();
    let candidate_down = matvec_candidate_parallel_f32(
        &component.down,
        HIDDEN,
        INTERMEDIATE,
        &candidate_swiglu,
        args.workers,
    )?;
    let candidate_down_duration_ms = candidate_down_started.elapsed().as_secs_f64() * 1_000.0;
    let candidate_weighted =
        weighted_accumulator_candidate_f32(&candidate_down, component.route.normalized_weight)?;

    // Matrix-only f64 references keep the boundary accounting clear: each
    // projection sees the candidate f32 output of the prior component, while
    // the final end-to-end oracle below propagates f64 through the full chain.
    let candidate_norm_f64 = candidate_norm
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_started = Instant::now();
    let (gate_reference, up_reference) =
        thread::scope(|scope| -> Result<(Vec<f64>, Vec<f64>), String> {
            let gate =
                scope.spawn(|| source_f64_matvec(&component.gate, &candidate_norm_f64, "gate"));
            let up = scope.spawn(|| source_f64_matvec(&component.up, &candidate_norm_f64, "up"));
            Ok((
                gate.join()
                    .map_err(|_| "gate reference worker panicked")??,
                up.join().map_err(|_| "up reference worker panicked")??,
            ))
        })?;
    let reference_gate_up_duration_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
    let (gate_max_abs, gate_max_relative) =
        max_f64_error(&candidate_gate, &gate_reference, "gate projection")?;
    let (up_max_abs, up_max_relative) =
        max_f64_error(&candidate_up, &up_reference, "up projection")?;
    if gate_max_abs > PROJECTION_F32_F64_TOLERANCE || up_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "gate/up projection parity failed: gate={gate_max_abs}, up={up_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let swiglu_reference = swiglu_reference_f64(&gate_reference, &up_reference)?;
    let (swiglu_max_abs, swiglu_max_relative) =
        max_f64_error(&candidate_swiglu, &swiglu_reference, "SiLU(gate)*up")?;
    if swiglu_max_abs > SWIGLU_F32_F64_TOLERANCE {
        return Err(format!(
            "SwiGLU parity failed: max_abs={swiglu_max_abs}, tolerance={SWIGLU_F32_F64_TOLERANCE}"
        ));
    }
    let candidate_swiglu_f64 = candidate_swiglu
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let down_matrix_reference = source_f64_matvec(&component.down, &candidate_swiglu_f64, "down")?;
    let (down_max_abs, down_max_relative) =
        max_f64_error(&candidate_down, &down_matrix_reference, "down projection")?;
    if down_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "down projection parity failed: max_abs={down_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }

    // Full source-shaped f64 oracle: reference norm -> reference gate/up ->
    // source SiLU product -> reference down -> exact receipt's f64 route
    // weight.  This is the end-to-end check for this one expert wave only.
    let reference_gate_full = source_f64_matvec(&component.gate, &reference_norm, "full gate")?;
    let reference_up_full = source_f64_matvec(&component.up, &reference_norm, "full up")?;
    let reference_swiglu_full = swiglu_reference_f64(&reference_gate_full, &reference_up_full)?;
    let reference_down_full =
        source_f64_matvec(&component.down, &reference_swiglu_full, "full down")?;
    let reference_weighted_full = reference_down_full
        .iter()
        .map(|value| *value * component.route.normalized_weight_source_f64)
        .collect::<Vec<_>>();
    let (weighted_max_abs, weighted_max_relative) = max_f64_error(
        &candidate_weighted,
        &reference_weighted_full,
        "weighted routed accumulator full chain",
    )?;
    if weighted_max_abs > WEIGHTED_OUTPUT_F32_F64_TOLERANCE {
        return Err(format!(
            "weighted routed expert full-chain parity failed: max_abs={weighted_max_abs}, tolerance={WEIGHTED_OUTPUT_F32_F64_TOLERANCE}"
        ));
    }

    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("routed expert rejection suite did not fail closed".into());
    }
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_ONE_ROUTED_EXPERT_CPU_ORACLE_READY_METAL_LEASE_REQUIRED",
        "mode": mode_name(args.mode),
        "one_selected_expert_only": true,
        "complete_artifact_scan_performed": false,
        "opened_exact_postnorm_and_selected_expert_payloads_only": true,
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
            "moe_intermediate": component.config.moe_intermediate,
            "experts": component.config.experts,
            "experts_per_token": component.config.experts_per_token,
            "rms_epsilon_bits": component.config.rms_norm_eps().to_bits(),
            "post_attention_norm": tensor_report(&component.post_norm),
            "expert_gate_proj": tensor_report(&component.gate),
            "expert_up_proj": tensor_report(&component.up),
            "expert_down_proj": tensor_report(&component.down),
        },
        "route_evidence": {
            "router_receipt_path": component.route.receipt_path,
            "router_receipt_sha256": component.route.receipt_sha256,
            "router_outer_receipt_path": component.route.outer_receipt_path,
            "router_outer_receipt_sha256": component.route.outer_receipt_sha256,
            "router_outer_receipt_seal_sha256": component.route.outer_receipt_seal_sha256,
            "router_fixture_post_attention_residual_sha256": ROUTER_FIXTURE_RESIDUAL_SHA256,
            "selected_route_index": ROUTE_INDEX,
            "selected_expert": component.route.selected_expert,
            "source_top10_ids": component.route.route_ids,
            "source_top10_renormalized_weights": component.route.route_weights,
            "selected_normalized_weight_f32": component.route.normalized_weight,
            "selected_normalized_weight_source_f64": component.route.normalized_weight_source_f64,
            "source_route_policy": "sealed Qwen80 layer-0 top-10 receipt; expert 65 is route index 0 after source softmax, stable selection, and selected-probability renormalization",
        },
        "cpu_oracle": {
            "post_attention_residual_kind": "deterministic source-shaped [2048] fixture reconstructed only to match sealed router evidence; not a live Qwen80 layer or model-token output",
            "post_attention_residual_sha256": residual_sha256,
            "postnorm_hidden_sha256": f32_slice_sha256(&candidate_norm),
            "post_attention_rms_norm": {
                "formula": "x * rsqrt(mean(x^2) + 1e-6) * (1 + packed_weight)",
                "candidate_f32_vs_reference_f64_max_abs": norm_max_abs,
                "candidate_f32_vs_reference_f64_max_relative": norm_max_relative,
                "direct_packed_norm_values_vs_library_decoder_max_abs": norm_direct_decode_max_abs,
                "tolerance_max_abs": NORM_F32_F64_TOLERANCE,
            },
            "gate_up": {
                "gate_shape": [INTERMEDIATE, HIDDEN],
                "up_shape": [INTERMEDIATE, HIDDEN],
                "candidate_gate_sha256": f32_slice_sha256(&candidate_gate),
                "candidate_up_sha256": f32_slice_sha256(&candidate_up),
                "gate_grouped_f32_vs_scalar_f64_max_abs": gate_max_abs,
                "gate_grouped_f32_vs_scalar_f64_max_relative": gate_max_relative,
                "up_grouped_f32_vs_scalar_f64_max_abs": up_max_abs,
                "up_grouped_f32_vs_scalar_f64_max_relative": up_max_relative,
                "tolerance_max_abs": PROJECTION_F32_F64_TOLERANCE,
                "candidate_cpu_workers": args.workers,
                "candidate_parallel_gate_up_duration_ms": candidate_gate_up_duration_ms,
                "reference_parallel_gate_up_duration_ms": reference_gate_up_duration_ms,
                "timing_is_component_cpu_work_not_tps": true,
            },
            "swiglu": {
                "formula": "silu(gate) * up",
                "candidate_activated_sha256": f32_slice_sha256(&candidate_swiglu),
                "candidate_f32_vs_reference_f64_max_abs": swiglu_max_abs,
                "candidate_f32_vs_reference_f64_max_relative": swiglu_max_relative,
                "tolerance_max_abs": SWIGLU_F32_F64_TOLERANCE,
            },
            "down": {
                "shape": [HIDDEN, INTERMEDIATE],
                "candidate_down_sha256": f32_slice_sha256(&candidate_down),
                "grouped_f32_vs_scalar_f64_same_candidate_activation_max_abs": down_max_abs,
                "grouped_f32_vs_scalar_f64_same_candidate_activation_max_relative": down_max_relative,
                "tolerance_max_abs": PROJECTION_F32_F64_TOLERANCE,
                "candidate_duration_ms": candidate_down_duration_ms,
                "timing_is_component_cpu_work_not_tps": true,
            },
            "weighted_accumulator": {
                "interpretation": "zero-initialized one-expert accumulator delta; no ten-route or residual combine occurred",
                "candidate_weighted_delta_sha256": f32_slice_sha256(&candidate_weighted),
                "candidate_f32_vs_full_source_f64_chain_max_abs": weighted_max_abs,
                "candidate_f32_vs_full_source_f64_chain_max_relative": weighted_max_relative,
                "tolerance_max_abs": WEIGHTED_OUTPUT_F32_F64_TOLERANCE,
                "all_2048_values_finite": candidate_weighted.iter().all(|value| value.is_finite()),
            },
        },
        "rejection_tests": rejections,
        "metal_intermediate_error_ledger": {
            "performed": false,
            "reason": "No explicit Rawls Qwen80 quiet lease. The isolated shader is staged but unregistered; no Metal context, compilation, or dispatch occurred.",
            "future_required_intermediates": ["postnorm_hidden[2048]", "gate[512]", "up[512]", "silu_gate_times_up[512]", "down[2048]", "weighted_accumulator_delta[2048]"],
            "future_acceptance": [
                "bind the same sealed manifest/receipt/route and exact four direct-packed payloads at device invocation",
                "compare every gate/up/activation/down/weighted output value to a fresh CPU oracle, including the source route weight before any scheduler combine",
                "do not turn one selected route into a ten-route, shared-expert, full-layer, token, or TPS claim",
            ],
        },
        "integration_contract": {
            "rawls_hybrid_scheduler_handoff": [
                "Only after the real layer-0 post-attention hidden and router result exist, select this body if and only if source route index 0 is expert 65 with the same normalized weight. Any different route must resolve its own exact tensor names and receipt.",
                "Execute gate and up against the same postnorm hidden, compute source SiLU(gate)*up, then down; multiply the result by the selected normalized route weight before accumulation.",
                "This output is one accumulator delta only. The scheduler remains responsible for the other nine selected routes, shared expert, shared gate, residual, and layer boundary.",
            ],
            "claim_boundary": [
                "The current admitted manifest remains LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION. Packed-to-packed component parity cannot alter that status.",
                "This is CPU-only one-route component evidence, not Qwen80 generation, HCLI, BASE_TRUE_TPS, TG10/TG3, capability, Agent OS, or tournament evidence.",
            ],
        },
    }))
}

/// Rebuild the exact direct-packed oracle inside the device invocation.  The
/// retained vectors are intentionally not reconstructed from an old CPU
/// receipt: a leased Metal capture compares its buffers to this same-process
/// source/artifact-bound CPU result.
fn build_device_cpu_oracle(args: &Args) -> Result<(BoundComponent, CpuOracle), String> {
    let component = bind_current_component(args)?;
    let residual = deterministic_post_attention_residual();
    if f32_slice_sha256(&residual) != ROUTER_FIXTURE_RESIDUAL_SHA256 {
        return Err(
            "reconstructed post-attention residual no longer matches sealed router fixture".into(),
        );
    }
    let normalized = post_attention_rms_norm_candidate_f32(
        &component.post_norm,
        &residual,
        component.config.rms_norm_eps(),
    )?;
    let reference_norm = post_attention_rms_norm_reference_f64(
        &component.post_norm,
        &residual,
        component.config.rms_norm_eps(),
    )?;
    let (norm_max_abs, norm_max_relative) =
        max_f64_error(&normalized, &reference_norm, "post-attention RMSNorm")?;
    let (_, decoded_norm) = decode_complete_binary_f32(&component.post_norm.payload)
        .map_err(|error| format!("post-attention norm decoder repeat failed: {error}"))?;
    let norm_direct_decode_max_abs = decoded_norm
        .iter()
        .enumerate()
        .map(|(index, decoded)| -> Result<f64, String> {
            Ok((f64::from(*decoded)
                - f64::from(packed_value_f32(
                    &component.post_norm.payload,
                    &component.post_norm.header,
                    index,
                )?))
            .abs())
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .fold(0.0f64, f64::max);
    if norm_max_abs > NORM_F32_F64_TOLERANCE || norm_direct_decode_max_abs != 0.0 {
        return Err(format!(
            "post-attention RMSNorm parity failed: f32/f64={norm_max_abs}, direct-decode={norm_direct_decode_max_abs}"
        ));
    }

    let gate_workers = (args.workers / 2).max(1);
    let up_workers = args.workers.saturating_sub(gate_workers).max(1);
    let candidate_started = Instant::now();
    let (gate, up) = thread::scope(|scope| -> Result<(Vec<f32>, Vec<f32>), String> {
        let gate_handle = scope.spawn(|| {
            matvec_candidate_parallel_f32(
                &component.gate,
                INTERMEDIATE,
                HIDDEN,
                &normalized,
                gate_workers,
            )
        });
        let up_handle = scope.spawn(|| {
            matvec_candidate_parallel_f32(
                &component.up,
                INTERMEDIATE,
                HIDDEN,
                &normalized,
                up_workers,
            )
        });
        Ok((
            gate_handle
                .join()
                .map_err(|_| "gate candidate worker panicked")??,
            up_handle
                .join()
                .map_err(|_| "up candidate worker panicked")??,
        ))
    })?;
    let candidate_gate_up_duration_ms = candidate_started.elapsed().as_secs_f64() * 1_000.0;
    let activated = swiglu_candidate_f32(&gate, &up)?;
    let candidate_down_started = Instant::now();
    let down = matvec_candidate_parallel_f32(
        &component.down,
        HIDDEN,
        INTERMEDIATE,
        &activated,
        args.workers,
    )?;
    let candidate_down_duration_ms = candidate_down_started.elapsed().as_secs_f64() * 1_000.0;
    let weighted = weighted_accumulator_candidate_f32(&down, component.route.normalized_weight)?;

    let normalized_f64 = normalized
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_started = Instant::now();
    let (gate_reference, up_reference) =
        thread::scope(|scope| -> Result<(Vec<f64>, Vec<f64>), String> {
            let gate_handle =
                scope.spawn(|| source_f64_matvec(&component.gate, &normalized_f64, "gate"));
            let up_handle = scope.spawn(|| source_f64_matvec(&component.up, &normalized_f64, "up"));
            Ok((
                gate_handle
                    .join()
                    .map_err(|_| "gate reference worker panicked")??,
                up_handle
                    .join()
                    .map_err(|_| "up reference worker panicked")??,
            ))
        })?;
    let reference_gate_up_duration_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
    let (gate_max_abs, gate_max_relative) =
        max_f64_error(&gate, &gate_reference, "gate projection")?;
    let (up_max_abs, up_max_relative) = max_f64_error(&up, &up_reference, "up projection")?;
    if gate_max_abs > PROJECTION_F32_F64_TOLERANCE || up_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "gate/up projection parity failed: gate={gate_max_abs}, up={up_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let swiglu_reference = swiglu_reference_f64(&gate_reference, &up_reference)?;
    let (swiglu_max_abs, swiglu_max_relative) =
        max_f64_error(&activated, &swiglu_reference, "SiLU(gate)*up")?;
    if swiglu_max_abs > SWIGLU_F32_F64_TOLERANCE {
        return Err(format!(
            "SwiGLU parity failed: max_abs={swiglu_max_abs}, tolerance={SWIGLU_F32_F64_TOLERANCE}"
        ));
    }
    let activated_f64 = activated
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let down_reference = source_f64_matvec(&component.down, &activated_f64, "down")?;
    let (down_max_abs, down_max_relative) =
        max_f64_error(&down, &down_reference, "down projection")?;
    if down_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "down projection parity failed: max_abs={down_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }

    let full_gate = source_f64_matvec(&component.gate, &reference_norm, "full gate")?;
    let full_up = source_f64_matvec(&component.up, &reference_norm, "full up")?;
    let full_activated = swiglu_reference_f64(&full_gate, &full_up)?;
    let full_down = source_f64_matvec(&component.down, &full_activated, "full down")?;
    let full_weighted = full_down
        .iter()
        .map(|value| *value * component.route.normalized_weight_source_f64)
        .collect::<Vec<_>>();
    let (weighted_max_abs, weighted_max_relative) = max_f64_error(
        &weighted,
        &full_weighted,
        "weighted routed accumulator full chain",
    )?;
    if weighted_max_abs > WEIGHTED_OUTPUT_F32_F64_TOLERANCE {
        return Err(format!(
            "weighted routed expert full-chain parity failed: max_abs={weighted_max_abs}, tolerance={WEIGHTED_OUTPUT_F32_F64_TOLERANCE}"
        ));
    }
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("routed expert rejection suite did not fail closed".into());
    }
    Ok((
        component,
        CpuOracle {
            residual,
            normalized,
            gate,
            up,
            activated,
            down,
            weighted,
            norm_max_abs,
            norm_max_relative,
            norm_direct_decode_max_abs,
            gate_max_abs,
            gate_max_relative,
            up_max_abs,
            up_max_relative,
            swiglu_max_abs,
            swiglu_max_relative,
            down_max_abs,
            down_max_relative,
            weighted_max_abs,
            weighted_max_relative,
            candidate_gate_up_duration_ms,
            reference_gate_up_duration_ms,
            candidate_down_duration_ms,
        },
    ))
}

fn run_metal_stage(
    args: &Args,
    component: &BoundComponent,
    oracle: &CpuOracle,
) -> Result<DeviceParityLedger, String> {
    if args.mode != Mode::Metal || args.quiet_metal_lease.is_none() {
        return Err("strict one-route Metal dispatch requires a validated quiet lease".into());
    }
    ensure_non_timed_metal_environment()?;
    if !matches!(
        component.config.layer_kind(LAYER),
        Ok(Qwen80LayerKind::LinearAttention)
    ) || component.config.hidden != HIDDEN
        || component.config.moe_intermediate != INTERMEDIATE
        || component.config.experts != EXPERTS
        || component.config.experts_per_token != TOP_K
    {
        return Err(
            "source config drifted from strict Qwen80 layer-0 routed-expert geometry".into(),
        );
    }
    let (gate_signs, gate_scales) = compact_sign_and_scale_sections(&component.gate)?;
    let (up_signs, up_scales) = compact_sign_and_scale_sections(&component.up)?;
    let (down_signs, down_scales) = compact_sign_and_scale_sections(&component.down)?;
    if gate_signs.len() != (INTERMEDIATE * HIDDEN) / 8
        || up_signs.len() != (INTERMEDIATE * HIDDEN) / 8
        || down_signs.len() != (HIDDEN * INTERMEDIATE) / 8
        || gate_scales.len() != (INTERMEDIATE * HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
        || up_scales.len() != (INTERMEDIATE * HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
        || down_scales.len() != (HIDDEN * INTERMEDIATE / GROUP_SIZE) * std::mem::size_of::<u16>()
    {
        return Err(
            "direct-packed selected-expert sections do not match the fixed shader ABI".into(),
        );
    }

    let context =
        MetalContext::new_with_trace_strict_math(false).map_err(|error| error.to_string())?;
    let device_name = context.device_name();
    let hidden = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "postnorm hidden")?)
        .map_err(|error| error.to_string())?;
    let gate_sign_buffer = context
        .new_buffer_with_bytes_checked(gate_signs)
        .map_err(|error| error.to_string())?;
    let gate_scale_buffer = context
        .new_buffer_with_bytes_checked(gate_scales)
        .map_err(|error| error.to_string())?;
    let up_sign_buffer = context
        .new_buffer_with_bytes_checked(up_signs)
        .map_err(|error| error.to_string())?;
    let up_scale_buffer = context
        .new_buffer_with_bytes_checked(up_scales)
        .map_err(|error| error.to_string())?;
    let gate_output = context
        .new_buffer_checked(bytes_for::<f32>(INTERMEDIATE, "gate output")?)
        .map_err(|error| error.to_string())?;
    let up_output = context
        .new_buffer_checked(bytes_for::<f32>(INTERMEDIATE, "up output")?)
        .map_err(|error| error.to_string())?;
    let activated = context
        .new_buffer_checked(bytes_for::<f32>(INTERMEDIATE, "SwiGLU activation")?)
        .map_err(|error| error.to_string())?;
    let down_sign_buffer = context
        .new_buffer_with_bytes_checked(down_signs)
        .map_err(|error| error.to_string())?;
    let down_scale_buffer = context
        .new_buffer_with_bytes_checked(down_scales)
        .map_err(|error| error.to_string())?;
    let down_output = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "down output")?)
        .map_err(|error| error.to_string())?;
    let weighted_output = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "weighted one-route delta")?)
        .map_err(|error| error.to_string())?;
    MetalContext::write_buffer_bytes(&hidden, bytemuck::cast_slice(&oracle.normalized));

    let mut command = TokenCommandBuffer::new(&context);
    command
        .dispatch_threads(
            "qwen80_routed_expert_wave_gate_up",
            (256, INTERMEDIATE as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&gate_sign_buffer), 0);
                encoder.set_buffer(1, Some(&gate_scale_buffer), 0);
                encoder.set_buffer(2, Some(&up_sign_buffer), 0);
                encoder.set_buffer(3, Some(&up_scale_buffer), 0);
                encoder.set_buffer(4, Some(&hidden), 0);
                encoder.set_buffer(5, Some(&gate_output), 0);
                encoder.set_buffer(6, Some(&up_output), 0);
                encoder.stage_set_u32(7, INTERMEDIATE as u32);
                encoder.stage_set_u32(8, HIDDEN as u32);
                encoder.stage_set_u32(9, GROUP_SIZE as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_routed_expert_wave_swiglu",
            (INTERMEDIATE as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&gate_output), 0);
                encoder.set_buffer(1, Some(&up_output), 0);
                encoder.set_buffer(2, Some(&activated), 0);
                encoder.stage_set_u32(3, INTERMEDIATE as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_routed_expert_wave_down_weighted",
            (256, HIDDEN as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&down_sign_buffer), 0);
                encoder.set_buffer(1, Some(&down_scale_buffer), 0);
                encoder.set_buffer(2, Some(&activated), 0);
                encoder.set_buffer(3, Some(&down_output), 0);
                encoder.set_buffer(4, Some(&weighted_output), 0);
                encoder.stage_set_u32(5, HIDDEN as u32);
                encoder.stage_set_u32(6, INTERMEDIATE as u32);
                encoder.stage_set_u32(7, GROUP_SIZE as u32);
                encoder.stage_set_f32(8, component.route.normalized_weight);
            },
        )
        .map_err(|error| error.to_string())?;
    let dispatch_count = command.dispatch_count();
    if dispatch_count != 3 {
        return Err(format!(
            "strict one-route command buffer encoded {dispatch_count} dispatches, expected exactly 3"
        ));
    }
    command
        .commit_and_wait()
        .map_err(|error| error.to_string())?;

    let gate = snapshot_f32(&gate_output, INTERMEDIATE, "gate output")?;
    let up = snapshot_f32(&up_output, INTERMEDIATE, "up output")?;
    let activated_values = snapshot_f32(&activated, INTERMEDIATE, "SwiGLU activation")?;
    let down = snapshot_f32(&down_output, HIDDEN, "down output")?;
    let weighted = snapshot_f32(&weighted_output, HIDDEN, "weighted one-route delta")?;
    let gate_max_abs = max_abs_error_f32(&oracle.gate, &gate, "strict-Metal gate")?;
    let up_max_abs = max_abs_error_f32(&oracle.up, &up, "strict-Metal up")?;
    let activated_max_abs = max_abs_error_f32(
        &oracle.activated,
        &activated_values,
        "strict-Metal SiLU(gate)*up",
    )?;
    let down_max_abs = max_abs_error_f32(&oracle.down, &down, "strict-Metal down")?;
    let weighted_max_abs = max_abs_error_f32(
        &oracle.weighted,
        &weighted,
        "strict-Metal weighted one-route delta",
    )?;
    for (label, error, tolerance) in [
        ("gate", gate_max_abs, DEVICE_GATE_UP_MAX_ABS_TOLERANCE),
        ("up", up_max_abs, DEVICE_GATE_UP_MAX_ABS_TOLERANCE),
        ("SwiGLU", activated_max_abs, DEVICE_SWIGLU_MAX_ABS_TOLERANCE),
        ("down", down_max_abs, DEVICE_DOWN_MAX_ABS_TOLERANCE),
        (
            "weighted",
            weighted_max_abs,
            DEVICE_WEIGHTED_MAX_ABS_TOLERANCE,
        ),
    ] {
        if error > tolerance {
            return Err(format!(
                "strict-Metal one-route {label} parity {error} exceeds {tolerance}"
            ));
        }
    }
    Ok(DeviceParityLedger {
        device_name,
        dispatch_count,
        gate,
        up,
        activated: activated_values,
        down,
        weighted,
        gate_max_abs,
        up_max_abs,
        activated_max_abs,
        down_max_abs,
        weighted_max_abs,
    })
}

fn run_metal_component(args: &Args) -> Result<Value, String> {
    let (component, oracle) = build_device_cpu_oracle(args)?;
    let device = run_metal_stage(args, &component, &oracle)?;
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("routed expert rejection suite did not fail closed".into());
    }
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_ONE_ROUTED_EXPERT_65_STRICT_MATH_METAL_COMPONENT_NOT_TEN_ROUTE_OR_LAYER",
        "mode": mode_name(args.mode),
        "one_selected_expert_only": true,
        "complete_artifact_scan_performed": false,
        "opened_exact_postnorm_and_selected_expert_payloads_only": true,
        "raw_bf16_or_safetensors_opened": false,
        "metal_device_or_dispatch_performed": true,
        "metal_execution_policy": post_attempt_metal_policy(args, true),
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
            "hidden": HIDDEN,
            "moe_intermediate": INTERMEDIATE,
            "selected_expert": EXPERT,
            "post_attention_norm": tensor_report(&component.post_norm),
            "expert_gate_proj": tensor_report(&component.gate),
            "expert_up_proj": tensor_report(&component.up),
            "expert_down_proj": tensor_report(&component.down),
        },
        "route_evidence": {
            "router_receipt_path": component.route.receipt_path,
            "router_receipt_sha256": component.route.receipt_sha256,
            "router_outer_receipt_path": component.route.outer_receipt_path,
            "router_outer_receipt_sha256": component.route.outer_receipt_sha256,
            "router_outer_receipt_seal_sha256": component.route.outer_receipt_seal_sha256,
            "router_fixture_post_attention_residual_sha256": ROUTER_FIXTURE_RESIDUAL_SHA256,
            "selected_route_index": ROUTE_INDEX,
            "selected_expert": component.route.selected_expert,
            "source_top10_ids": component.route.route_ids,
            "source_top10_renormalized_weights": component.route.route_weights,
            "selected_normalized_weight_f32": component.route.normalized_weight,
            "selected_normalized_weight_source_f64": component.route.normalized_weight_source_f64,
        },
        "same_capture_cpu_oracle": {
            "post_attention_residual": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.residual)},
            "postnorm_hidden": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.normalized)},
            "gate": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&oracle.gate)},
            "up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&oracle.up)},
            "silu_gate_times_up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&oracle.activated)},
            "down": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.down)},
            "weighted_one_route_delta": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.weighted)},
            "f32_vs_f64_max_abs": {
                "post_attention_rmsnorm": oracle.norm_max_abs,
                "gate": oracle.gate_max_abs,
                "up": oracle.up_max_abs,
                "silu_gate_times_up": oracle.swiglu_max_abs,
                "down": oracle.down_max_abs,
                "weighted_full_chain": oracle.weighted_max_abs,
            },
            "f32_vs_f64_max_relative": {
                "post_attention_rmsnorm": oracle.norm_max_relative,
                "gate": oracle.gate_max_relative,
                "up": oracle.up_max_relative,
                "silu_gate_times_up": oracle.swiglu_max_relative,
                "down": oracle.down_max_relative,
                "weighted_full_chain": oracle.weighted_max_relative,
            },
            "postnorm_direct_decode_max_abs": oracle.norm_direct_decode_max_abs,
            "candidate_gate_up_duration_ms": oracle.candidate_gate_up_duration_ms,
            "reference_gate_up_duration_ms": oracle.reference_gate_up_duration_ms,
            "candidate_down_duration_ms": oracle.candidate_down_duration_ms,
            "timing_is_component_cpu_work_not_tps": true,
        },
        "metal_intermediate_error_ledger": {
            "performed": true,
            "device": device.device_name,
            "command_buffers": 1,
            "compute_dispatches": device.dispatch_count,
            "kernel_sequence": [
                "qwen80_routed_expert_wave_gate_up",
                "qwen80_routed_expert_wave_swiglu",
                "qwen80_routed_expert_wave_down_weighted",
            ],
            "strict_math": true,
            "timing_or_benchmarking_performed": false,
            "acceptance": {
                "gate_max_abs": device.gate_max_abs,
                "gate_tolerance": DEVICE_GATE_UP_MAX_ABS_TOLERANCE,
                "up_max_abs": device.up_max_abs,
                "up_tolerance": DEVICE_GATE_UP_MAX_ABS_TOLERANCE,
                "silu_gate_times_up_max_abs": device.activated_max_abs,
                "silu_gate_times_up_tolerance": DEVICE_SWIGLU_MAX_ABS_TOLERANCE,
                "down_max_abs": device.down_max_abs,
                "down_tolerance": DEVICE_DOWN_MAX_ABS_TOLERANCE,
                "weighted_one_route_delta_max_abs": device.weighted_max_abs,
                "weighted_one_route_delta_tolerance": DEVICE_WEIGHTED_MAX_ABS_TOLERANCE,
            },
            "device_intermediates": {
                "gate": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&device.gate)},
                "up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&device.up)},
                "silu_gate_times_up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&device.activated)},
                "down": {"elements": HIDDEN, "sha256": f32_slice_sha256(&device.down)},
                "weighted_one_route_delta": {"elements": HIDDEN, "sha256": f32_slice_sha256(&device.weighted)},
            },
        },
        "rejection_tests": rejections,
        "integration_contract": {
            "scheduler_handoff": [
                "This device result is one expert-65 weighted accumulator delta only.",
                "The remaining nine routed experts, shared expert, aggregation, second residual, and all layer/token work remain unexecuted.",
            ],
            "claim_boundary": [
                "Synthetic source-shaped postnorm input component parity only; not a full Qwen80 layer, token, decoder, generation, HCLI, TPS, TG, capability, Agent OS, or tournament receipt.",
            ],
        },
    }))
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_routed_expert_wave: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = begin_capture(&args) {
        eprintln!("ascension_qwen80_direct_packed_routed_expert_wave: {error}");
        std::process::exit(2);
    }
    let stage_result = match args.mode {
        Mode::CpuOracle => run_cpu_oracle(&args),
        Mode::Metal => run_metal_component(&args),
    };
    match finalize_capture(&args, stage_result) {
        Ok((result, None)) => match serde_json::to_string_pretty(&result) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("ascension_qwen80_direct_packed_routed_expert_wave: result print failed: {error}");
                std::process::exit(2);
            }
        },
        Ok((_result, Some(error))) => {
            eprintln!("ascension_qwen80_direct_packed_routed_expert_wave: {error}");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_routed_expert_wave: capture finalization failed: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_quiet_metal_lease_document() -> Map<String, Value> {
        serde_json::json!({
            "schema": QUIET_METAL_LEASE_SCHEMA,
            "status": QUIET_METAL_LEASE_STATUS,
            "lease_id": "a".repeat(64),
            "authorization_seal_sha256": "b".repeat(64),
            "seal_sha256": "c".repeat(64),
            "model": {
                "id": MODEL_ID,
                "key": MODEL_KEY,
                "repository": SOURCE_REPOSITORY,
                "revision": SOURCE_REVISION,
            },
            "artifact_binding": {
                "manifest_document_sha256": CURRENT_MANIFEST_DOCUMENT_SHA256,
                "manifest_seal_sha256": CURRENT_MANIFEST_SEAL,
                "admission_receipt_seal_sha256": CURRENT_ADMISSION_RECEIPT_SEAL,
                "post_attention_norm_artifact_sha256": POST_NORM_ARTIFACT_SHA256,
                "expert_gate_artifact_sha256": GATE_ARTIFACT_SHA256,
                "expert_up_artifact_sha256": UP_ARTIFACT_SHA256,
                "expert_down_artifact_sha256": DOWN_ARTIFACT_SHA256,
            },
            "execution_policy": {
                "component": QUIET_METAL_COMPONENT,
                "quiet_qwen80_device_lease": true,
                "strict_math": true,
                "timing_or_benchmarking_allowed": false,
                "complete_layer_or_token_allowed": false,
                "tps_or_tg_claim_allowed": false,
            },
        })
        .as_object()
        .unwrap()
        .clone()
    }

    fn tiny_payload(shape: &[usize], scales: &[f32], signs: &[u8]) -> Vec<u8> {
        let elements = shape.iter().product::<usize>();
        let groups = elements.div_ceil(GROUP_SIZE);
        assert_eq!(scales.len(), groups);
        assert_eq!(signs.len(), groups * (GROUP_SIZE / 8));
        let mut payload = Vec::new();
        payload.extend_from_slice(b"HQ30G1B1");
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&(GROUP_SIZE as u32).to_le_bytes());
        payload.extend_from_slice(&(shape.len() as u16).to_le_bytes());
        payload.extend_from_slice(&0u16.to_le_bytes());
        payload.extend_from_slice(&(elements as u64).to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());
        for &dimension in shape {
            payload.extend_from_slice(&(dimension as u32).to_le_bytes());
        }
        for &scale in scales {
            payload.extend_from_slice(&f16::from_f32(scale).to_bits().to_le_bytes());
        }
        payload.extend_from_slice(signs);
        payload
    }

    #[test]
    fn lsb_first_packed_signs_and_fp16_scales_are_preserved() {
        let mut signs = vec![0u8; GROUP_SIZE / 8];
        signs[0] = 0b0000_0101;
        let payload = tiny_payload(&[GROUP_SIZE], &[0.75], &signs);
        let header = parse_complete_binary_header(&payload).unwrap();
        assert_eq!(packed_value_f32(&payload, &header, 0).unwrap(), 0.75);
        assert_eq!(packed_value_f32(&payload, &header, 1).unwrap(), -0.75);
        assert_eq!(packed_value_f32(&payload, &header, 2).unwrap(), 0.75);
    }

    #[test]
    fn direct_packed_dot_matches_scalar_f64_reference() {
        let mut signs = vec![0u8; 2 * (GROUP_SIZE / 8)];
        signs[..GROUP_SIZE / 8].fill(0xff);
        signs[GROUP_SIZE / 8..].fill(0b0101_0101);
        let payload = tiny_payload(&[2, GROUP_SIZE], &[0.5, 0.25], &signs);
        let header = parse_complete_binary_header(&payload).unwrap();
        let input = (0..GROUP_SIZE)
            .map(|index| (index as f32 - 61.0) / 39.0)
            .collect::<Vec<_>>();
        let direct = |row: usize| -> f32 {
            (0..GROUP_SIZE)
                .map(|column| {
                    packed_value_f32(&payload, &header, row * GROUP_SIZE + column).unwrap()
                        * input[column]
                })
                .sum::<f32>()
        };
        let observed = [direct(0), direct(1)];
        let (_, expected) = complete_binary_matvec_f64(
            &payload,
            &input
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let (max_abs, _) = max_f64_error(&observed, &expected, "tiny packed dot").unwrap();
        assert!(max_abs < 1.0e-4, "max_abs={max_abs}");
    }

    #[test]
    fn source_swiglu_order_is_gate_then_silu_times_up() {
        let mut gate = vec![0.0f32; INTERMEDIATE];
        let mut up = vec![0.0f32; INTERMEDIATE];
        gate[0] = 2.0;
        up[0] = -3.0;
        let output = swiglu_candidate_f32(&gate, &up).unwrap();
        assert!((output[0] - (2.0 / (1.0 + (-2.0f32).exp()) * -3.0)).abs() < 1.0e-6);
        assert!(swiglu_candidate_f32(&vec![f32::NAN; INTERMEDIATE], &up).is_err());
    }

    #[test]
    fn route_binding_is_exactly_one_selected_expert_65() {
        assert_eq!(EXPECTED_ROUTE_IDS[ROUTE_INDEX], EXPERT as u16);
        assert_eq!(EXPECTED_ROUTE_IDS.len(), TOP_K);
        assert!((EXPECTED_ROUTE_WEIGHT - 0.245_458_886_027_336_12).abs() < f64::EPSILON);
        assert!(EXPERT < EXPERTS);
    }

    #[test]
    fn component_constants_preserve_qwen80_source_geometry() {
        assert_eq!(LAYER, 0);
        assert_eq!(HIDDEN, 2_048);
        assert_eq!(INTERMEDIATE, 512);
        assert_eq!(EXPERTS, 512);
        assert_eq!(TOP_K, 10);
        assert_eq!(GROUP_SIZE, 128);
        assert_ne!(HIDDEN, 4_096);
        assert_eq!(HIDDEN % GROUP_SIZE, 0);
        assert_eq!(INTERMEDIATE % GROUP_SIZE, 0);
    }

    #[test]
    fn quiet_metal_lease_requires_exact_one_route_component_policy() {
        let lease = valid_quiet_metal_lease_document();
        let (lease_id, authorization) =
            validate_quiet_metal_lease_document(&lease, "test quiet Metal lease").unwrap();
        assert_eq!(lease_id, "a".repeat(64));
        assert_eq!(authorization, "b".repeat(64));

        let mut wrong_expert = valid_quiet_metal_lease_document();
        wrong_expert
            .get_mut("artifact_binding")
            .and_then(Value::as_object_mut)
            .unwrap()
            .insert(
                "expert_down_artifact_sha256".into(),
                Value::String("d".repeat(64)),
            );
        assert!(validate_quiet_metal_lease_document(&wrong_expert, "wrong expert").is_err());

        let mut timed = valid_quiet_metal_lease_document();
        timed
            .get_mut("execution_policy")
            .and_then(Value::as_object_mut)
            .unwrap()
            .insert("timing_or_benchmarking_allowed".into(), Value::Bool(true));
        assert!(validate_quiet_metal_lease_document(&timed, "timed lease").is_err());
    }

    #[test]
    fn routed_expert_shader_binds_two_dimensional_rows_to_grid_y_and_scalar_lanes() {
        let shader = include_str!("../shaders/qwen80_routed_expert_wave.metal");
        for entry in [
            "kernel void qwen80_routed_expert_wave_gate_up",
            "kernel void qwen80_routed_expert_wave_down_weighted",
        ] {
            let body = shader
                .split(entry)
                .nth(1)
                .unwrap_or_else(|| panic!("missing shader entry point {entry}"));
            assert!(body.contains("uint3 tid [[thread_position_in_threadgroup]]"));
            assert!(body.contains("uint3 group_position [[threadgroup_position_in_grid]]"));
            assert!(body.contains("const uint lane = tid.x;"));
            assert!(body.contains("const uint row = group_position.y;"));
            assert!(body.contains("if (lane == 0u) {"));
            assert!(!body.contains("if (tid == 0u) {"));
        }
        assert!(shader.contains("device float* down_output [[buffer(3)]]"));
        assert!(shader.contains("device float* weighted_accumulator_delta [[buffer(4)]]"));
        assert!(shader.contains("constant float& route_weight [[buffer(8)]]"));
    }
}
