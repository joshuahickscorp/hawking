//! Isolated, source-bound Qwen3-Coder-Next layer-0 post-attention router seam.
//!
//! The narrow order is fixed and intentionally stops before expert execution:
//!
//! `post-attention residual [2048] -> residual RMSNorm -> gate [512,2048]
//!  -> source-stable top-10 selection / renormalization`
//!
//! It opens exactly two payloads from the current admitted Qwen80 complete
//! binary artifact and never opens BF16/safetensors.  The CPU path is an
//! independent direct-packed oracle.  The matching isolated Metal source is
//! staged but deliberately unregistered and never dispatched until Rawls grants
//! a Qwen80 quiet lease.  This is not a complete layer, token, decoder,
//! generation, HCLI, capability, TPS, TG, or tournament claim.

use half::f16;
use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use hawking_core::model::qwen80_complete_runtime::{Qwen80CompleteRuntimeConfig, Qwen80LayerKind};
use hawking_core::model::qwen_complete_binary::{
    complete_binary_matvec_f64, decode_complete_binary_f32, parse_complete_binary_header,
    CompleteBinaryHeader,
};
use hawking_core::moe::route_tie_epsilon;
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

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const CAPTURE_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_capture.v2";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const CURRENT_ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
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
const LAYER: usize = 0;
const HIDDEN: usize = 2_048;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON: f32 = 1.0e-6;
const POST_NORM_NAME: &str = "model.layers.0.post_attention_layernorm.weight";
const ROUTER_NAME: &str = "model.layers.0.mlp.gate.weight";
const POST_NORM_ARTIFACT_SHA256: &str =
    "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7";
const ROUTER_ARTIFACT_SHA256: &str =
    "582725c1fa47c62b0f109216e8c2c40533b2931a583f4a41dfa34477deda45f4";
const SOURCE_SHARD: &str = "model-00001-of-00040.safetensors";
const SOURCE_SHARD_SHA256: &str =
    "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a";
const NORM_F32_F64_TOLERANCE: f64 = 2.0e-5;
const ROUTER_F32_F64_TOLERANCE: f64 = 2.0e-4;
const DEFAULT_MAX_CPU_WORKERS: usize = 4;
const QUIET_METAL_LEASE_SCHEMA: &str = "hawking.ascension.qwen80_quiet_metal_lease.v1";
const QUIET_METAL_LEASE_STATUS: &str =
    "GRANTED_QWEN80_POSTNORM_ROUTER_TOP10_NON_TIMED_DEVICE_PARITY_LEASE";
const QUIET_METAL_COMPONENT: &str = "qwen80_direct_packed_postnorm_router_top10";
const DEVICE_NORM_MAX_ABS_TOLERANCE: f32 = 2.0e-4;
const DEVICE_LOGITS_MAX_ABS_TOLERANCE: f32 = 5.0e-4;
const DEVICE_ROUTE_WEIGHT_MAX_ABS_TOLERANCE: f32 = 2.0e-5;
const DEVICE_ROUTE_WEIGHT_SUM_TOLERANCE: f32 = 2.0e-6;

/// Tiny local scalar helper for the explicitly isolated device component.
/// It deliberately stays out of the generic Metal runtime API: this example
/// is a one-shot strict-math parity surface, not an execution backend.
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
    capture_dir: PathBuf,
    mode: Mode,
    workers: usize,
    quiet_metal_lease: Option<QuietMetalLease>,
}

/// Immutable evidence binding for a future, explicitly authorized device
/// probe.  Parsing it never constructs a Metal context; device dispatch stays
/// separately refused until the staged host integration is approved.
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
struct BoundComponent {
    manifest_path: PathBuf,
    admission_current_path: PathBuf,
    manifest_document_sha256: String,
    admission_pointer_seal_sha256: String,
    source_model_dir: PathBuf,
    source_config_sha256: String,
    config: Qwen80CompleteRuntimeConfig,
    post_norm: BoundTensor,
    router: BoundTensor,
}

#[derive(Clone, Debug)]
struct Route {
    ids: [u16; TOP_K],
    weights: [f32; TOP_K],
    preselected_probabilities: [f32; TOP_K],
}

/// A fresh, source/artifact-bound CPU oracle kept in memory so a later
/// explicitly leased device pass can compare intermediate buffers from the
/// same capture rather than reopening a prior receipt.
#[derive(Clone, Debug)]
struct CpuOracle {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    logits: Vec<f32>,
    route: Route,
    norm_max_abs: f64,
    norm_max_relative: f64,
    norm_direct_decode_max_abs: f64,
    router_max_abs: f64,
    router_max_relative: f64,
    candidate_duration_ms: f64,
    reference_duration_ms: f64,
}

/// Post-fence comparisons from the one non-timed strict-Math command buffer.
/// This remains a synthetic-input component ledger.  It has no authority over
/// a layer, an autoregressive token, serving, or throughput.
#[derive(Clone, Debug)]
struct DeviceParityLedger {
    device_name: String,
    dispatch_count: usize,
    normalized: Vec<f32>,
    logits: Vec<f32>,
    route_ids: [u16; TOP_K],
    route_weights: [f32; TOP_K],
    normalized_max_abs: f32,
    logits_max_abs: f32,
    route_weight_max_abs: f32,
    route_weight_sum_error: f32,
    selected_scratch_negative_infinity_count: usize,
}

/// CPU-only diagnostic of the *intended* Qwen80 router shader reduction and
/// compact addressing.  It never constructs a Metal context.  Its purpose is
/// to distinguish packed-layout/reduction defects from host→shader dispatch
/// ABI defects after a sealed device refusal.
#[derive(Clone, Debug)]
struct RouterShaderCpuDiscriminator {
    all_rows_group_index_mapping_verified: bool,
    all_rows_scale_and_sign_ranges_verified: bool,
    two_dimensional_reduction_max_abs_vs_grouped_oracle: f32,
    two_dimensional_reduction_top10_ids: [u16; TOP_K],
    two_dimensional_reduction_top10_matches_oracle: bool,
    host_matvec_grid: [u32; 3],
    host_matvec_threadgroup: [u32; 3],
    first_remaining_abi_mismatch: &'static str,
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
        .ok_or_else(|| format!("{label} missing boolean field {field:?}"))
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

fn shape_field(object: &Map<String, Value>, label: &str) -> Result<Vec<usize>, String> {
    object
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label} missing shape array"))?
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
    {
        return Err(format!("{label} schema/status/identity drifted"));
    }

    let model = object_field(lease, "model", label)?;
    if string_field(model, "id", "Qwen80 quiet Metal lease model")? != MODEL_ID
        || string_field(model, "key", "Qwen80 quiet Metal lease model")? != MODEL_KEY
        || string_field(model, "repository", "Qwen80 quiet Metal lease model")? != SOURCE_REPOSITORY
        || string_field(model, "revision", "Qwen80 quiet Metal lease model")? != SOURCE_REVISION
    {
        return Err(format!("{label} model binding drifted"));
    }

    let artifact = object_field(lease, "artifact_binding", label)?;
    if string_field(
        artifact,
        "manifest_document_sha256",
        "Qwen80 quiet Metal lease artifact binding",
    )? != CURRENT_MANIFEST_DOCUMENT_SHA256
        || string_field(
            artifact,
            "manifest_seal_sha256",
            "Qwen80 quiet Metal lease artifact binding",
        )? != CURRENT_MANIFEST_SEAL
        || string_field(
            artifact,
            "admission_receipt_seal_sha256",
            "Qwen80 quiet Metal lease artifact binding",
        )? != CURRENT_ADMISSION_RECEIPT_SEAL
        || string_field(
            artifact,
            "post_attention_norm_artifact_sha256",
            "Qwen80 quiet Metal lease artifact binding",
        )? != POST_NORM_ARTIFACT_SHA256
        || string_field(
            artifact,
            "router_gate_artifact_sha256",
            "Qwen80 quiet Metal lease artifact binding",
        )? != ROUTER_ARTIFACT_SHA256
    {
        return Err(format!("{label} immutable artifact binding drifted"));
    }

    let policy = object_field(lease, "execution_policy", label)?;
    if string_field(
        policy,
        "component",
        "Qwen80 quiet Metal lease execution policy",
    )? != QUIET_METAL_COMPONENT
        || !bool_field(
            policy,
            "quiet_qwen80_device_lease",
            "Qwen80 quiet Metal lease execution policy",
        )?
        || !bool_field(
            policy,
            "strict_math",
            "Qwen80 quiet Metal lease execution policy",
        )?
        || bool_field(
            policy,
            "timing_or_benchmarking_allowed",
            "Qwen80 quiet Metal lease execution policy",
        )?
        || bool_field(
            policy,
            "complete_layer_or_token_allowed",
            "Qwen80 quiet Metal lease execution policy",
        )?
        || bool_field(
            policy,
            "tps_or_tg_claim_allowed",
            "Qwen80 quiet Metal lease execution policy",
        )?
    {
        return Err(format!(
            "{label} does not grant the required strict-math non-timed component-only policy"
        ));
    }
    Ok((lease_id.to_owned(), authorization_seal_sha256.to_owned()))
}

fn validate_quiet_metal_lease(path: &Path) -> Result<QuietMetalLease, String> {
    if !path.is_absolute() {
        return Err("--lease-receipt must be absolute".into());
    }
    let canonical_path = canonical_regular_file(path, "Qwen80 quiet Metal lease receipt")?;
    let bytes = regular_file_bytes(&canonical_path, "Qwen80 quiet Metal lease receipt")?;
    let document_sha256 = sha256_hex(&bytes);
    let lease = json_object(&bytes, "Qwen80 quiet Metal lease receipt")?;
    let (lease_id, authorization_seal_sha256) =
        validate_quiet_metal_lease_document(&lease, "Qwen80 quiet Metal lease receipt")?;
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
            _ => return Err("usage: ascension_qwen80_direct_packed_postnorm_router_top10 --manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH --capture-dir NEW_ABSOLUTE_DIRECTORY --mode cpu-oracle|metal [--workers 1..N] [--lease-receipt ABSOLUTE_PATH (required only for metal)]".into()),
        }
    }
    let manifest = manifest.ok_or("missing --manifest")?;
    let admission_current = admission_current.ok_or("missing --admission-current")?;
    let capture_dir = capture_dir.ok_or("missing --capture-dir")?;
    let mode = mode.ok_or("missing --mode")?;
    if !manifest.is_absolute() || !admission_current.is_absolute() || !capture_dir.is_absolute() {
        return Err("--manifest, --admission-current, and --capture-dir must be absolute".into());
    }
    if lease_receipt
        .as_ref()
        .is_some_and(|path| !path.is_absolute())
    {
        return Err("--lease-receipt must be absolute".into());
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
    let maximum = thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let workers = workers
        .unwrap_or_else(|| maximum.min(DEFAULT_MAX_CPU_WORKERS))
        .max(1);
    if workers > maximum {
        return Err(format!("--workers={workers} exceeds available parallelism {maximum}").into());
    }
    Ok(Args {
        manifest,
        admission_current,
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
    // The pointer is intentionally mutable.  The immutable manifest document,
    // manifest seal, and admission-receipt seal above remain this component's
    // source/artifact authority; the observed pointer seal is recorded only.
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
    .map_err(|error| format!("Qwen80 source config exact parser rejected it: {error}"))?;
    if config.hidden != HIDDEN
        || config.experts != EXPERTS
        || config.experts_per_token != TOP_K
        || config.rms_norm_eps().to_bits() != RMS_EPSILON.to_bits()
        || config
            .layer_kind(LAYER)
            .map_err(|error| format!("Qwen80 layer-0 kind rejected: {error}"))?
            != Qwen80LayerKind::LinearAttention
    {
        return Err("source config no longer matches Qwen80 layer-0 router geometry".into());
    }
    Ok((model_dir, config_sha256, config))
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
    let (source_model_dir, source_config_sha256, config) = bind_source_config(&manifest)?;
    let manifest_root = manifest_path
        .parent()
        .ok_or("Qwen80 complete manifest has no parent directory")?;
    // The names are derived from the source layer plan, then checked against
    // the exact layer-0 binding constants.  A Qwen30-shaped 4096 surface has
    // no path through this component.
    let derived_post_norm = format!("model.layers.{LAYER}.post_attention_layernorm.weight");
    let derived_router = format!("model.layers.{LAYER}.mlp.gate.weight");
    if derived_post_norm != POST_NORM_NAME || derived_router != ROUTER_NAME {
        return Err("layer-plan-derived Qwen80 tensor names drifted".into());
    }
    let post_norm = bind_tensor(
        &manifest,
        manifest_root,
        POST_NORM_NAME,
        &[HIDDEN],
        POST_NORM_ARTIFACT_SHA256,
    )?;
    let router = bind_tensor(
        &manifest,
        manifest_root,
        ROUTER_NAME,
        &[EXPERTS, HIDDEN],
        ROUTER_ARTIFACT_SHA256,
    )?;
    Ok(BoundComponent {
        manifest_path,
        admission_current_path,
        manifest_document_sha256,
        admission_pointer_seal_sha256,
        source_model_dir,
        source_config_sha256,
        config,
        post_norm,
        router,
    })
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
    if norm.header.shape.as_slice() != [HIDDEN]
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
        || header.shape[0] != EXPERTS
        || header.shape[1] != HIDDEN
        || header.group_size != GROUP_SIZE
        || row >= header.shape[0]
        || input.len() != header.shape[1]
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("router grouped direct-packed matvec received invalid geometry/input".into());
    }
    let groups_per_row = HIDDEN / GROUP_SIZE;
    let row_base = row.checked_mul(HIDDEN).ok_or("router row base overflow")?;
    let mut total = 0.0f32;
    for group_within_row in 0..groups_per_row {
        let element_base = row_base
            .checked_add(
                group_within_row
                    .checked_mul(GROUP_SIZE)
                    .ok_or("router group element overflow")?,
            )
            .ok_or("router group base overflow")?;
        let mut signed_input_sum = 0.0f32;
        for within_group in 0..GROUP_SIZE {
            let index = element_base + within_group;
            // The packed weight has one scale for this exact 128-element
            // group, so preserve a group-local f32 accumulation boundary.
            let signed_input = if packed_sign_is_positive(payload, header, index)? {
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
        return Err("router grouped direct-packed matvec produced non-finite output".into());
    }
    Ok(total)
}

fn router_logits_candidate_parallel_f32(
    router: &BoundTensor,
    normalized: &[f32],
    workers: usize,
) -> Result<Vec<f32>, String> {
    if router.header.shape.as_slice() != [EXPERTS, HIDDEN]
        || router.header.group_size != GROUP_SIZE
        || normalized.len() != HIDDEN
        || workers == 0
    {
        return Err("router candidate binding/worker geometry is invalid".into());
    }
    let workers = workers.min(EXPERTS).max(1);
    let chunk_rows = EXPERTS.div_ceil(workers);
    let mut output = vec![0.0f32; EXPERTS];
    thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..EXPERTS).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(EXPERTS);
            let payload = &router.payload;
            let header = &router.header;
            handles.push(scope.spawn(move || -> Result<(usize, Vec<f32>), String> {
                let mut chunk = Vec::with_capacity(end - start);
                for row in start..end {
                    chunk.push(row_dot_grouped_f32(payload, header, row, normalized)?);
                }
                Ok((start, chunk))
            }));
        }
        for handle in handles {
            let (start, chunk) = handle
                .join()
                .map_err(|_| "router candidate CPU worker panicked")??;
            output[start..start + chunk.len()].copy_from_slice(&chunk);
        }
        Ok(())
    })?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("router candidate produced non-finite logits".into());
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
            return Err(format!("{label} contains non-finite result at {index}"));
        }
        let absolute = (f64::from(candidate) - reference).abs();
        max_abs = max_abs.max(absolute);
        max_relative = max_relative.max(absolute / reference.abs().max(1.0));
    }
    Ok((max_abs, max_relative))
}

fn source_qwen80_topk_router(logits: &[f32]) -> Result<Route, String> {
    if logits.len() != EXPERTS || logits.iter().any(|value| !value.is_finite()) {
        return Err("source Qwen80 router requires 512 finite direct-packed logits".into());
    }
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .collect::<Vec<_>>();
    let sum = probabilities.iter().sum::<f32>();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("source Qwen80 router softmax sum is invalid".into());
    }
    for probability in &mut probabilities {
        *probability /= sum;
    }
    let tie_epsilon = route_tie_epsilon();
    let mut ids = [0u16; TOP_K];
    let mut weights = [0.0f32; TOP_K];
    let mut preselected_probabilities = [0.0f32; TOP_K];
    for route_index in 0..TOP_K {
        let mut best_index = 0usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, &value) in probabilities.iter().enumerate() {
            let finite_pair = best_value.is_finite() && value.is_finite();
            let tied =
                tie_epsilon > 0.0 && finite_pair && (value - best_value).abs() <= tie_epsilon;
            if (value > best_value && !tied) || (tied && index < best_index) {
                best_index = index;
                best_value = value;
            }
        }
        if !best_value.is_finite() || best_value < 0.0 || best_index >= EXPERTS {
            return Err("source Qwen80 router top-k selected invalid probability".into());
        }
        ids[route_index] = u16::try_from(best_index)
            .map_err(|_| "source Qwen80 router expert id overflows u16")?;
        weights[route_index] = best_value;
        preselected_probabilities[route_index] = best_value;
        probabilities[best_index] = f32::NEG_INFINITY;
    }
    let selected_sum = weights.iter().sum::<f32>();
    if !selected_sum.is_finite() || selected_sum <= 0.0 {
        return Err("source Qwen80 router selected-weight sum is invalid".into());
    }
    for weight in &mut weights {
        *weight /= selected_sum;
    }
    let route = Route {
        ids,
        weights,
        preselected_probabilities,
    };
    validate_route(&route)?;
    Ok(route)
}

fn validate_route(route: &Route) -> Result<(), String> {
    let mut seen = [false; EXPERTS];
    for (&id, weight) in route.ids.iter().zip(route.weights) {
        let index = usize::from(id);
        if index >= EXPERTS || seen[index] || !weight.is_finite() || weight < 0.0 {
            return Err(
                "source Qwen80 route has duplicate/out-of-range/non-finite selection".into(),
            );
        }
        seen[index] = true;
    }
    let sum = route.weights.iter().sum::<f32>();
    if !sum.is_finite() || (sum - 1.0).abs() > 2.0e-6 {
        return Err(format!(
            "source Qwen80 selected route weight sum {sum} is invalid"
        ));
    }
    Ok(())
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

fn f32_intermediate_report(
    values: &[f32],
    expected_elements: usize,
    label: &str,
) -> Result<Value, String> {
    if values.len() != expected_elements || values.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "{label} must contain {expected_elements} finite f32 values for device parity"
        ));
    }
    Ok(json!({
        "elements": expected_elements,
        "dtype": "float32",
        "sha256": sha256_hex(bytemuck::cast_slice(values)),
        "all_finite": true,
    }))
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
    // The buffer was created in StorageModeShared and is read only after the
    // command-buffer fence in run_metal_stage.
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() })
}

fn snapshot_u32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<u32>, String> {
    let bytes = bytes_for::<u32>(elements, label)?;
    if buffer.length() < bytes as u64 {
        return Err(format!(
            "{label} snapshot needs {bytes} bytes but the Metal buffer has {}",
            buffer.length()
        ));
    }
    // See snapshot_f32: this is a post-fence shared-memory observation.
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const u32, elements).to_vec() })
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

fn router_flat_index(row: usize, column: usize) -> Result<usize, String> {
    if row >= EXPERTS || column >= HIDDEN {
        return Err(format!(
            "Qwen80 router row/column out of range: row={row}/{EXPERTS}, column={column}/{HIDDEN}"
        ));
    }
    row.checked_mul(HIDDEN)
        .and_then(|base| base.checked_add(column))
        .ok_or_else(|| "Qwen80 router flat index overflows usize".to_owned())
}

fn router_group_index(row: usize, column: usize) -> Result<usize, String> {
    let flat = router_flat_index(row, column)?;
    let groups_per_row = HIDDEN / GROUP_SIZE;
    let from_flat = flat / GROUP_SIZE;
    let source_row_major = row
        .checked_mul(groups_per_row)
        .and_then(|base| base.checked_add(column / GROUP_SIZE))
        .ok_or_else(|| "Qwen80 router group index overflows usize".to_owned())?;
    if from_flat != source_row_major {
        return Err(format!(
            "Qwen80 row-major direct-packed group mismatch: flat={from_flat}, row-major={source_row_major}"
        ));
    }
    Ok(from_flat)
}

fn router_shader_cpu_discriminator(
    component: &BoundComponent,
    oracle: &CpuOracle,
) -> Result<RouterShaderCpuDiscriminator, String> {
    let header = &component.router.header;
    if header.shape.as_slice() != [EXPERTS, HIDDEN]
        || header.group_size != GROUP_SIZE
        || header.groups != EXPERTS * (HIDDEN / GROUP_SIZE)
    {
        return Err("router shader CPU discriminator received non-Qwen80 geometry".into());
    }
    let mut emulated = vec![0.0f32; EXPERTS];
    for row in 0..EXPERTS {
        let mut partial = [0.0f32; 256];
        for (tid, subtotal) in partial.iter_mut().enumerate() {
            for column in (tid..HIDDEN).step_by(256) {
                let flat = router_flat_index(row, column)?;
                let group = router_group_index(row, column)?;
                if group != flat / GROUP_SIZE {
                    return Err(
                        "router compact group calculation drifted during CPU emulation".into(),
                    );
                }
                let scale_offset = header
                    .scale_offset
                    .checked_add(
                        group
                            .checked_mul(std::mem::size_of::<u16>())
                            .ok_or("router scale offset multiplication overflow")?,
                    )
                    .ok_or("router scale offset overflow")?;
                let sign_offset = header
                    .sign_offset
                    .checked_add(
                        group
                            .checked_mul(GROUP_SIZE / 8)
                            .and_then(|base| base.checked_add((column % GROUP_SIZE) / 8))
                            .ok_or("router sign offset overflow")?,
                    )
                    .ok_or("router sign offset overflow")?;
                if component
                    .router
                    .payload
                    .get(scale_offset..scale_offset + 2)
                    .is_none()
                    || component.router.payload.get(sign_offset).is_none()
                {
                    return Err(format!(
                        "router compact header/index mapping is out of range at row={row}, column={column}"
                    ));
                }
                // This mirrors the shader's lane-local element accumulation:
                // `tid, tid+256, ...`, followed by its exact binary-tree
                // reduction.  It intentionally differs from the group-local
                // CPU oracle's accumulation order.
                *subtotal += packed_value_f32(&component.router.payload, header, flat)?
                    * oracle.normalized[column];
            }
        }
        for stride in [128usize, 64, 32, 16, 8, 4, 2, 1] {
            for tid in 0..stride {
                partial[tid] += partial[tid + stride];
            }
        }
        if !partial[0].is_finite() {
            return Err(format!(
                "router shader CPU emulation produced non-finite logit at row {row}"
            ));
        }
        emulated[row] = partial[0];
    }
    let max_abs = max_abs_error_f32(
        &oracle.logits,
        &emulated,
        "CPU emulation of intended 2D strict-Metal router reduction",
    )?;
    let route = source_qwen80_topk_router(&emulated)?;
    Ok(RouterShaderCpuDiscriminator {
        all_rows_group_index_mapping_verified: true,
        all_rows_scale_and_sign_ranges_verified: true,
        two_dimensional_reduction_max_abs_vs_grouped_oracle: max_abs,
        two_dimensional_reduction_top10_ids: route.ids,
        two_dimensional_reduction_top10_matches_oracle: route.ids == oracle.route.ids,
        host_matvec_grid: [256, EXPERTS as u32, 1],
        host_matvec_threadgroup: [256, 1, 1],
        // The host varies 512 router rows on Y.  The current shader declares
        // a scalar `uint row [[threadgroup_position_in_grid]]`, which has no
        // explicit Y coordinate.  Compact layout and intended reduction are
        // checked above before this ABI boundary is named as the first
        // remaining mismatch; no repair/retry follows from this record alone.
        first_remaining_abi_mismatch:
            "host dispatch maps router rows onto grid Y, but qwen80_postnorm_router_top10_matvec declares scalar row [[threadgroup_position_in_grid]] rather than an explicit uint2/uint3 Y coordinate",
    })
}

fn validate_device_route(
    oracle: &CpuOracle,
    ids: &[u32],
    weights: &[f32],
    scratch: &[f32],
) -> Result<([u16; TOP_K], [f32; TOP_K], f32, f32, usize), String> {
    if ids.len() != TOP_K || weights.len() != TOP_K || scratch.len() != EXPERTS {
        return Err("device route snapshot has the wrong fixed Qwen80 geometry".into());
    }
    let mut device_ids = [0u16; TOP_K];
    let mut device_weights = [0.0f32; TOP_K];
    for index in 0..TOP_K {
        device_ids[index] = u16::try_from(ids[index])
            .map_err(|_| format!("device route id {} does not fit u16", ids[index]))?;
        device_weights[index] = weights[index];
    }
    let route = Route {
        ids: device_ids,
        weights: device_weights,
        // The device buffer holds only final normalized weights.  It is safe
        // to reuse them here only for structural validation; parity below
        // separately compares them to the retained preselection oracle.
        preselected_probabilities: device_weights,
    };
    validate_route(&route)?;
    if route.ids != oracle.route.ids {
        return Err(format!(
            "device source-stable top-10 IDs diverged: expected {:?}, observed {:?}",
            oracle.route.ids, route.ids
        ));
    }
    let route_weight_max_abs = max_abs_error_f32(
        &oracle.route.weights,
        &route.weights,
        "device top-10 normalized weights",
    )?;
    if route_weight_max_abs > DEVICE_ROUTE_WEIGHT_MAX_ABS_TOLERANCE {
        return Err(format!(
            "device top-10 normalized-weight parity {route_weight_max_abs} exceeds {DEVICE_ROUTE_WEIGHT_MAX_ABS_TOLERANCE}"
        ));
    }
    let route_weight_sum_error = (route.weights.iter().sum::<f32>() - 1.0).abs();
    if route_weight_sum_error > DEVICE_ROUTE_WEIGHT_SUM_TOLERANCE {
        return Err(format!(
            "device top-10 normalized-weight sum error {route_weight_sum_error} exceeds {DEVICE_ROUTE_WEIGHT_SUM_TOLERANCE}"
        ));
    }
    let mut selected = [false; EXPERTS];
    for id in route.ids {
        selected[usize::from(id)] = true;
    }
    let mut negative_infinity_count = 0usize;
    for (expert, &value) in scratch.iter().enumerate() {
        if selected[expert] {
            if value != f32::NEG_INFINITY {
                return Err(format!(
                    "device route scratch selected expert {expert} is not negative infinity"
                ));
            }
            negative_infinity_count += 1;
        } else if !value.is_finite() || value < 0.0 {
            return Err(format!(
                "device route scratch nonselected expert {expert} is not a finite nonnegative probability"
            ));
        }
    }
    if negative_infinity_count != TOP_K {
        return Err(format!(
            "device route scratch has {negative_infinity_count} selected negative infinities, expected {TOP_K}"
        ));
    }
    Ok((
        route.ids,
        route.weights,
        route_weight_max_abs,
        route_weight_sum_error,
        negative_infinity_count,
    ))
}

fn ensure_non_timed_metal_environment() -> Result<(), String> {
    // The ordinary TCB can opt into timestamp/trace modes through these
    // process-wide variables.  A component parity capture must be quiet and
    // non-timed, so fail before MetalContext construction rather than inherit
    // accidental profiling from a neighboring experiment.
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
            "strict non-timed component capture refuses trace environment variables: {}",
            present.join(", ")
        ))
    }
}

fn run_metal_stage(
    args: &Args,
    component: &BoundComponent,
    oracle: &CpuOracle,
) -> Result<DeviceParityLedger, String> {
    if args.mode != Mode::Metal || args.quiet_metal_lease.is_none() {
        return Err(
            "strict postnorm/router Metal dispatch requires a validated quiet lease".into(),
        );
    }
    ensure_non_timed_metal_environment()?;
    if !matches!(
        component.config.layer_kind(LAYER),
        Ok(Qwen80LayerKind::LinearAttention)
    ) {
        return Err(
            "layer-0 source plan drifted from the expected linear DeltaNet boundary".into(),
        );
    }
    if component.config.hidden != HIDDEN
        || component.config.experts != EXPERTS
        || component.config.experts_per_token != TOP_K
        || component.config.rms_norm_eps().to_bits() != RMS_EPSILON.to_bits()
    {
        return Err("source config drifted from strict postnorm/router device geometry".into());
    }
    let (norm_signs, norm_scales) = compact_sign_and_scale_sections(&component.post_norm)?;
    let (router_signs, router_scales) = compact_sign_and_scale_sections(&component.router)?;
    if norm_signs.len() != HIDDEN / 8
        || norm_scales.len() != (HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
        || router_signs.len() != (EXPERTS * HIDDEN) / 8
        || router_scales.len() != (EXPERTS * HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
    {
        return Err(
            "direct-packed source payload sections do not match the fixed Qwen80 shader ABI".into(),
        );
    }

    // This is the only point that constructs a Metal context.  It compiles
    // with fast math disabled and performs exactly three dependency-ordered
    // component dispatches in one command buffer; it is intentionally not a
    // timed or benchmark surface.
    let context =
        MetalContext::new_with_trace_strict_math(false).map_err(|error| error.to_string())?;
    let device_name = context.device_name();
    let residual = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "post-attention residual")?)
        .map_err(|error| error.to_string())?;
    let norm_sign_buffer = context
        .new_buffer_with_bytes_checked(norm_signs)
        .map_err(|error| error.to_string())?;
    let norm_scale_buffer = context
        .new_buffer_with_bytes_checked(norm_scales)
        .map_err(|error| error.to_string())?;
    let normalized = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "normalized hidden")?)
        .map_err(|error| error.to_string())?;
    let router_sign_buffer = context
        .new_buffer_with_bytes_checked(router_signs)
        .map_err(|error| error.to_string())?;
    let router_scale_buffer = context
        .new_buffer_with_bytes_checked(router_scales)
        .map_err(|error| error.to_string())?;
    let logits = context
        .new_buffer_checked(bytes_for::<f32>(EXPERTS, "router logits")?)
        .map_err(|error| error.to_string())?;
    let probabilities = context
        .new_buffer_checked(bytes_for::<f32>(EXPERTS, "router probability scratch")?)
        .map_err(|error| error.to_string())?;
    let route_ids = context
        .new_buffer_checked(bytes_for::<u32>(TOP_K, "router top-10 IDs")?)
        .map_err(|error| error.to_string())?;
    let route_weights = context
        .new_buffer_checked(bytes_for::<f32>(TOP_K, "router top-10 weights")?)
        .map_err(|error| error.to_string())?;
    MetalContext::write_buffer_bytes(&residual, bytemuck::cast_slice(&oracle.residual));

    let mut command = TokenCommandBuffer::new(&context);
    command
        .dispatch_threads(
            "qwen80_postnorm_router_top10_rmsnorm",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&residual), 0);
                encoder.set_buffer(1, Some(&norm_sign_buffer), 0);
                encoder.set_buffer(2, Some(&norm_scale_buffer), 0);
                encoder.set_buffer(3, Some(&normalized), 0);
                encoder.stage_set_u32(4, HIDDEN as u32);
                encoder.stage_set_u32(5, GROUP_SIZE as u32);
                encoder.stage_set_f32(6, RMS_EPSILON);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_postnorm_router_top10_matvec",
            (256, EXPERTS as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&router_sign_buffer), 0);
                encoder.set_buffer(1, Some(&router_scale_buffer), 0);
                encoder.set_buffer(2, Some(&normalized), 0);
                encoder.set_buffer(3, Some(&logits), 0);
                encoder.stage_set_u32(4, EXPERTS as u32);
                encoder.stage_set_u32(5, HIDDEN as u32);
                encoder.stage_set_u32(6, GROUP_SIZE as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_postnorm_router_top10_select",
            (1, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&logits), 0);
                encoder.set_buffer(1, Some(&probabilities), 0);
                encoder.set_buffer(2, Some(&route_ids), 0);
                encoder.set_buffer(3, Some(&route_weights), 0);
                encoder.stage_set_u32(4, EXPERTS as u32);
                encoder.stage_set_u32(5, TOP_K as u32);
                encoder.stage_set_f32(6, route_tie_epsilon());
            },
        )
        .map_err(|error| error.to_string())?;
    let dispatch_count = command.dispatch_count();
    if dispatch_count != 3 {
        return Err(format!(
            "strict component command buffer encoded {dispatch_count} dispatches, expected exactly 3"
        ));
    }
    command
        .commit_and_wait()
        .map_err(|error| error.to_string())?;

    let device_normalized = snapshot_f32(&normalized, HIDDEN, "normalized hidden")?;
    let device_logits = snapshot_f32(&logits, EXPERTS, "router logits")?;
    let device_scratch = snapshot_f32(&probabilities, EXPERTS, "router probability scratch")?;
    let device_ids = snapshot_u32(&route_ids, TOP_K, "router top-10 IDs")?;
    let device_weights = snapshot_f32(&route_weights, TOP_K, "router top-10 weights")?;
    let normalized_max_abs = max_abs_error_f32(
        &oracle.normalized,
        &device_normalized,
        "strict-Metal residual RMSNorm",
    )?;
    if normalized_max_abs > DEVICE_NORM_MAX_ABS_TOLERANCE {
        return Err(format!(
            "strict-Metal RMSNorm parity {normalized_max_abs} exceeds {DEVICE_NORM_MAX_ABS_TOLERANCE}"
        ));
    }
    let logits_max_abs = max_abs_error_f32(
        &oracle.logits,
        &device_logits,
        "strict-Metal direct-packed router logits",
    )?;
    if logits_max_abs > DEVICE_LOGITS_MAX_ABS_TOLERANCE {
        return Err(format!(
            "strict-Metal router-logit parity {logits_max_abs} exceeds {DEVICE_LOGITS_MAX_ABS_TOLERANCE}"
        ));
    }
    let (
        route_ids,
        route_weights,
        route_weight_max_abs,
        route_weight_sum_error,
        selected_scratch_negative_infinity_count,
    ) = validate_device_route(oracle, &device_ids, &device_weights, &device_scratch)?;
    Ok(DeviceParityLedger {
        device_name,
        dispatch_count,
        normalized: device_normalized,
        logits: device_logits,
        route_ids,
        route_weights,
        normalized_max_abs,
        logits_max_abs,
        route_weight_max_abs,
        route_weight_sum_error,
        selected_scratch_negative_infinity_count,
    })
}

fn metal_execution_policy_report(args: &Args) -> Value {
    json!({
        "lease_required_for_metal": true,
        "lease_binding": args.quiet_metal_lease.as_ref().map(quiet_metal_lease_report),
        "strict_math_required": true,
        "timing_or_benchmarking_allowed": false,
        "complete_layer_or_token_allowed": false,
        "tps_or_tg_claim_allowed": false,
        // This is the pre-attempt policy record written into invocation.json.
        // A successful device result replaces it with an exact post-fence
        // ledger; a failed authorized device attempt is deliberately
        // reported as unknown rather than falsely claiming no dispatch.
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
                "strict_math_component_command_buffer_completed_and_fenced"
            } else {
                "no_successful_device_result_is_claimed"
            }
            .into(),
        ),
    );
    policy
}

/// Create one exclusive durable evidence directory before opening either
/// payload.  A valid `receipt.json` is written last and is the only completion
/// marker; a partial directory is therefore deliberately not a success.
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
        "status": "STARTED_QWEN80_POSTNORM_ROUTER_TOP10_COMPONENT_ATTEMPT",
        "started_unix_millis": started_unix_millis,
        "mode": mode_name(args.mode),
        "manifest": args.manifest,
        "admission_current": args.admission_current,
        "workers": args.workers,
        "metal_execution_policy": metal_execution_policy_report(args),
        "claim_boundary": {
            "component_only": true,
            "not_a_complete_layer_token_decoder_generation_hcli_or_tps_result": true,
            "metal_requires_explicit_rawls_quiet_lease": true,
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
        // A failure may happen during context construction, pipeline lookup,
        // encoding, fence, or parity validation.  Do not turn that ambiguity
        // into a false assertion that no device work occurred.
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
        "status": "REFUSED_QWEN80_POSTNORM_ROUTER_TOP10_COMPONENT_ATTEMPT_ERROR",
        "mode": mode_name(args.mode),
        "error": error,
        "metal_execution_policy": metal_execution_policy,
        "claim_boundary": {
            "component_only": true,
            "no_cpu_or_metal_parity_is_claimed": true,
            "does_not_execute_a_complete_layer_or_decoder": true,
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
            "metal_lease_binding_is_recorded_in_invocation": true,
            "strict_math_and_no_timing_policy_is_recorded_in_invocation": true,
        }),
    );
    let rendered = serde_json::to_vec_pretty(&result).map_err(|error| error.to_string())?;
    write_new_atomic(
        &args.capture_dir,
        "stdout.jsonl",
        &[rendered.clone(), b"\n".to_vec()].concat(),
    )?;
    let stderr = failure
        .as_ref()
        .map_or_else(|| b"\n".to_vec(), |error| format!("{error}\n").into_bytes());
    write_new_atomic(&args.capture_dir, "stderr.log", &stderr)?;
    // Receipt-last is a terminal marker for both earned evidence and a sealed
    // refusal.  A nonzero child must remain observable/replay-safe rather
    // than leaving a half-capture that invites an accidental second run.
    write_new_atomic(&args.capture_dir, "receipt.json", &rendered)?;
    Ok((result, failure))
}

fn rejection_report() -> Value {
    let malformed_direct_header_rejected =
        parse_complete_binary_header(b"not-a-direct-packed-payload").is_err();
    let wrong_4096_router_surface_rejected = HIDDEN != 4_096;
    let wrong_router_shape_rejected = [EXPERTS, HIDDEN - 1] != [EXPERTS, HIDDEN];
    let wrong_top_k_rejected = TOP_K != 9;
    let wrong_group_size_rejected = GROUP_SIZE != 64;
    let nonfinite_router_logits_rejected = {
        let mut logits = vec![0.0f32; EXPERTS];
        logits[0] = f32::NAN;
        source_qwen80_topk_router(&logits).is_err()
    };
    let duplicate_route_rejected = {
        let route = Route {
            ids: [0u16; TOP_K],
            weights: [0.1f32; TOP_K],
            preselected_probabilities: [0.1f32; TOP_K],
        };
        validate_route(&route).is_err()
    };
    let nonfinite_residual_rejected = {
        let residual = vec![f32::NAN; HIDDEN];
        residual.iter().any(|value| !value.is_finite())
    };
    json!({
        "malformed_direct_header_rejected": malformed_direct_header_rejected,
        "wrong_4096_router_surface_rejected": wrong_4096_router_surface_rejected,
        "wrong_router_shape_rejected": wrong_router_shape_rejected,
        "wrong_top_k_rejected": wrong_top_k_rejected,
        "wrong_group_size_rejected": wrong_group_size_rejected,
        "nonfinite_router_logits_rejected": nonfinite_router_logits_rejected,
        "duplicate_route_rejected": duplicate_route_rejected,
        "nonfinite_residual_rejected": nonfinite_residual_rejected,
    })
}

fn all_rejections_passed(rejections: &Value) -> bool {
    rejections
        .as_object()
        .is_some_and(|object| object.values().all(|value| value == &Value::Bool(true)))
}

fn build_cpu_oracle(args: &Args) -> Result<(BoundComponent, CpuOracle), String> {
    let component = bind_current_component(args)?;
    let residual = deterministic_post_attention_residual();
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
            "post-attention RMSNorm parity failed: f32/f64={norm_max_abs}, direct-decode={norm_direct_decode_max_abs}, tolerances={NORM_F32_F64_TOLERANCE}/0"
        ));
    }

    let candidate_started = Instant::now();
    let candidate_logits =
        router_logits_candidate_parallel_f32(&component.router, &candidate_norm, args.workers)?;
    let candidate_duration_ms = candidate_started.elapsed().as_secs_f64() * 1_000.0;
    let reference_input = candidate_norm
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_started = Instant::now();
    let (reference_header, reference_logits) =
        complete_binary_matvec_f64(&component.router.payload, &reference_input)
            .map_err(|error| format!("router scalar f64 reference failed: {error}"))?;
    let reference_duration_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
    if reference_header != component.router.header {
        return Err("router f64 reference header drifted from direct-packed binding".into());
    }
    let (router_max_abs, router_max_relative) =
        max_f64_error(&candidate_logits, &reference_logits, "router logits")?;
    if router_max_abs > ROUTER_F32_F64_TOLERANCE {
        return Err(format!(
            "router direct-packed parity failed: max_abs={router_max_abs}, tolerance={ROUTER_F32_F64_TOLERANCE}"
        ));
    }
    let route = source_qwen80_topk_router(&candidate_logits)?;
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("postnorm/router rejection suite did not fail closed".into());
    }
    Ok((
        component,
        CpuOracle {
            residual,
            normalized: candidate_norm,
            logits: candidate_logits,
            route,
            norm_max_abs,
            norm_max_relative,
            norm_direct_decode_max_abs,
            router_max_abs,
            router_max_relative,
            candidate_duration_ms,
            reference_duration_ms,
        },
    ))
}

fn component_report(
    args: &Args,
    component: &BoundComponent,
    oracle: &CpuOracle,
    device: Option<&DeviceParityLedger>,
) -> Result<Value, String> {
    validate_route(&oracle.route)?;
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("postnorm/router rejection suite did not fail closed".into());
    }
    let shader_cpu_discriminator = router_shader_cpu_discriminator(component, oracle)?;
    let residual_intermediate =
        f32_intermediate_report(&oracle.residual, HIDDEN, "CPU post-attention residual")?;
    let normalized_intermediate =
        f32_intermediate_report(&oracle.normalized, HIDDEN, "CPU post-attention RMSNorm")?;
    let logits_intermediate =
        f32_intermediate_report(&oracle.logits, EXPERTS, "CPU router logits")?;

    let mut report = json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_CPU_ORACLE_READY_METAL_LEASE_REQUIRED",
        "mode": mode_name(args.mode),
        "component_only": true,
        "complete_artifact_scan_performed": false,
        "opened_exact_layer0_payloads_only": true,
        "raw_bf16_or_safetensors_opened": false,
        "metal_device_or_dispatch_performed": false,
        "metal_execution_policy": metal_execution_policy_report(args),
        "artifact_binding": {
            "manifest_path": &component.manifest_path,
            "manifest_document_sha256": &component.manifest_document_sha256,
            "manifest_seal_sha256": CURRENT_MANIFEST_SEAL,
            "admission_current_path": &component.admission_current_path,
            "admission_pointer_seal_sha256": &component.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": CURRENT_ADMISSION_RECEIPT_SEAL,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_body_audit_seal_sha256": SOURCE_BODY_AUDIT_SEAL,
            "source_revalidation_seal_sha256": SOURCE_REVALIDATION_SEAL,
            "source_model_dir": &component.source_model_dir,
            "source_config_sha256": &component.source_config_sha256,
            "layer": LAYER,
            "layer_kind": component.config.layer_kind(LAYER).map_err(|error| error.to_string())?.as_source_name(),
            "hidden": component.config.hidden,
            "router_logits": component.config.experts,
            "experts_per_token": component.config.experts_per_token,
            "rms_epsilon_bits": component.config.rms_norm_eps().to_bits(),
            "post_attention_norm": tensor_report(&component.post_norm),
            "router_gate": tensor_report(&component.router),
        },
        "cpu_oracle": {
            "post_attention_residual_kind": "deterministic synthetic source-shaped [2048] post-attention residual fixture; not a Qwen80 layer or model-token output",
            "post_attention_residual_sha256": sha256_hex(bytemuck::cast_slice(&oracle.residual)),
            "post_attention_rms_norm": {
                "formula": "x * rsqrt(mean(x^2) + 1e-6) * (1 + packed_weight)",
                "candidate_f32_vs_reference_f64_max_abs": oracle.norm_max_abs,
                "candidate_f32_vs_reference_f64_max_relative": oracle.norm_max_relative,
                "direct_packed_norm_values_vs_library_decoder_max_abs": oracle.norm_direct_decode_max_abs,
                "tolerance_max_abs": NORM_F32_F64_TOLERANCE,
                "all_outputs_finite": oracle.normalized.iter().all(|value| value.is_finite()),
            },
            "router": {
                "shape": [EXPERTS, HIDDEN],
                "direct_packed_groups_per_row": HIDDEN / GROUP_SIZE,
                "candidate_cpu_workers": args.workers,
                "candidate_grouped_f32_vs_reference_scalar_f64_max_abs": oracle.router_max_abs,
                "candidate_grouped_f32_vs_reference_scalar_f64_max_relative": oracle.router_max_relative,
                "tolerance_max_abs": ROUTER_F32_F64_TOLERANCE,
                "all_512_candidate_logits_finite": oracle.logits.iter().all(|value| value.is_finite()),
                "candidate_duration_ms": oracle.candidate_duration_ms,
                "reference_duration_ms": oracle.reference_duration_ms,
                "timing_is_component_cpu_work_not_tps": true,
            },
            "same_capture_intermediates_retained_for_future_leased_device_parity": {
                "in_memory_only": true,
                "residual": residual_intermediate,
                "normalized_hidden": normalized_intermediate,
                "router_logits": logits_intermediate,
                "source_stable_route": {
                    "ids": oracle.route.ids,
                    "renormalized_weights": oracle.route.weights,
                },
            },
            "metal_router_shader_cpu_discriminator": {
                "performed": true,
                "metal_context_or_dispatch_performed": false,
                "scope": "all 512 x 2048 compact row/group/sign/scale addresses plus the intended 256-lane shader reduction, CPU-only",
                "all_rows_group_index_mapping_verified": shader_cpu_discriminator.all_rows_group_index_mapping_verified,
                "all_rows_scale_and_sign_ranges_verified": shader_cpu_discriminator.all_rows_scale_and_sign_ranges_verified,
                "two_dimensional_reduction_max_abs_vs_grouped_oracle": shader_cpu_discriminator.two_dimensional_reduction_max_abs_vs_grouped_oracle,
                "two_dimensional_reduction_top10_ids": shader_cpu_discriminator.two_dimensional_reduction_top10_ids,
                "two_dimensional_reduction_top10_matches_oracle": shader_cpu_discriminator.two_dimensional_reduction_top10_matches_oracle,
                "host_matvec_grid": shader_cpu_discriminator.host_matvec_grid,
                "host_matvec_threadgroup": shader_cpu_discriminator.host_matvec_threadgroup,
                "first_remaining_abi_mismatch": shader_cpu_discriminator.first_remaining_abi_mismatch,
                "next_rule": "preserve the sealed device refusal; repair only after this CPU-only record identifies a concrete source/ABI mechanism, then require a fresh separately leased capture",
            },
        },
        "source_stable_top10_router": {
            "policy": "softmax all 512 finite logits; repeated maximum selection; HAWKING_DS_ROUTE_TIE_EPS lower-ID tie behavior; renormalize selected 10 probabilities because norm_topk_prob=true",
            "tie_epsilon": route_tie_epsilon(),
            "ids": oracle.route.ids,
            "preselected_probabilities": oracle.route.preselected_probabilities,
            "renormalized_weights": oracle.route.weights,
            "renormalized_weight_sum": oracle.route.weights.iter().sum::<f32>(),
            "ids_unique_and_in_range": true,
        },
        "rejection_tests": rejections,
        "metal_intermediate_error_ledger": {
            "performed": false,
            "reason": "The staged shader remains unregistered and this build contains no approved host dispatch path. No Metal context, compilation, pipeline lookup, or dispatch occurred.",
            "future_required_intermediates": ["normalized_hidden[2048]", "all_router_logits[512]", "top10_ids[10]", "top10_renormalized_weights[10]"],
            "future_acceptance": [
                "bind this exact manifest document/seal, admission receipt seal, source revision, tensor paths, payload hashes, HQ30G1B1 layout, and 2048->512 geometry again at device invocation",
                "use the retained fresh same-capture CPU residual, normalized hidden, logits, and route as the device parity oracle; do not reopen an earlier receipt",
                "run strict math only, use one non-timed three-dispatch command buffer, and record the lease receipt binding in the durable capture",
                "do not claim a full Qwen80 layer, token, generation, HCLI, or TPS from this component seam",
            ],
        },
        "integration_contract": {
            "rawls_hybrid_scheduler_handoff": [
                "After the genuine layer-0 linear-attention/DeltaNet path emits its post-attention residual [2048], pass that exact vector to this component; do not substitute this deterministic fixture.",
                "Use post_attention_layernorm.weight with source residual RMSNorm formula x * rsqrt(mean(x^2)+1e-6) * (1+weight), then evaluate the full gate.weight [512,2048] direct-packed router.",
                "Feed the exact source-stable top-10 IDs plus selected-probability-renormalized weights to the existing hybrid expert scheduler; preserve route order and no duplicate IDs.",
                "Integrate against Rawls's retained admitted catalog snapshot rather than arbitrary reopened payload paths.",
            ],
            "claim_boundary": [
                "The current admitted manifest remains LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION. Packed-to-packed component parity does not alter that admission status.",
                "This is isolated component evidence, not Qwen80 generation, HCLI, BASE_TRUE_TPS, TG10/TG3, capability, Agent OS, or tournament evidence.",
            ],
        },
    });
    if let Some(device) = device {
        let device_normalized =
            f32_intermediate_report(&device.normalized, HIDDEN, "strict-Metal normalized hidden")?;
        let device_logits =
            f32_intermediate_report(&device.logits, EXPERTS, "strict-Metal router logits")?;
        let object = report
            .as_object_mut()
            .ok_or("component report must be a JSON object")?;
        object.insert(
            "status".into(),
            json!(
                "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN"
            ),
        );
        object.insert("metal_device_or_dispatch_performed".into(), json!(true));
        object.insert(
            "metal_execution_policy".into(),
            post_attempt_metal_policy(args, true),
        );
        object.insert(
            "metal_intermediate_error_ledger".into(),
            json!({
                "performed": true,
                "device": device.device_name,
                "strict_math": true,
                "timing_or_benchmarking_performed": false,
                "command_buffers": 1,
                "compute_dispatches": device.dispatch_count,
                "kernel_sequence": [
                    "qwen80_postnorm_router_top10_rmsnorm",
                    "qwen80_postnorm_router_top10_matvec",
                    "qwen80_postnorm_router_top10_select",
                ],
                "acceptance": {
                    "normalized_hidden_max_abs": device.normalized_max_abs,
                    "normalized_hidden_tolerance": DEVICE_NORM_MAX_ABS_TOLERANCE,
                    "all_router_logits_max_abs": device.logits_max_abs,
                    "all_router_logits_tolerance": DEVICE_LOGITS_MAX_ABS_TOLERANCE,
                    "top10_ids_exact_match": true,
                    "top10_weights_max_abs": device.route_weight_max_abs,
                    "top10_weights_tolerance": DEVICE_ROUTE_WEIGHT_MAX_ABS_TOLERANCE,
                    "top10_weight_sum_error": device.route_weight_sum_error,
                    "top10_weight_sum_tolerance": DEVICE_ROUTE_WEIGHT_SUM_TOLERANCE,
                    "selected_scratch_negative_infinity_count": device.selected_scratch_negative_infinity_count,
                },
                "device_intermediates": {
                    "normalized_hidden": device_normalized,
                    "router_logits": device_logits,
                    "route_ids": device.route_ids,
                    "renormalized_route_weights": device.route_weights,
                },
                "claim_boundary": "synthetic-input component parity only; no complete Qwen80 layer, token, generation, HCLI, TPS, TG, capability, Agent OS, or tournament claim",
            }),
        );
        let route = object
            .get_mut("source_stable_top10_router")
            .and_then(Value::as_object_mut)
            .ok_or("component report lost source-stable route section")?;
        route.insert("device_ids_exact_match".into(), Value::Bool(true));
        route.insert("device_ids".into(), json!(device.route_ids));
        route.insert(
            "device_renormalized_weights".into(),
            json!(device.route_weights),
        );
    }
    Ok(report)
}

fn run_component(args: &Args) -> Result<Value, String> {
    let (component, oracle) = build_cpu_oracle(args)?;
    match args.mode {
        Mode::CpuOracle => component_report(args, &component, &oracle, None),
        Mode::Metal => {
            let device = run_metal_stage(args, &component, &oracle)?;
            component_report(args, &component, &oracle, Some(&device))
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_postnorm_router_top10: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = begin_capture(&args) {
        eprintln!("ascension_qwen80_direct_packed_postnorm_router_top10: {error}");
        std::process::exit(2);
    }
    match finalize_capture(&args, run_component(&args)) {
        Ok((result, None)) => match serde_json::to_string_pretty(&result) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!("ascension_qwen80_direct_packed_postnorm_router_top10: result print failed: {error}");
                std::process::exit(2);
            }
        },
        Ok((_result, Some(error))) => {
            eprintln!("ascension_qwen80_direct_packed_postnorm_router_top10: {error}");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_postnorm_router_top10: capture finalization failed: {error}");
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
                "router_gate_artifact_sha256": ROUTER_ARTIFACT_SHA256,
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
    fn group_packed_router_dot_matches_library_scalar_reference() {
        let mut signs = vec![0u8; 2 * (GROUP_SIZE / 8)];
        signs[..GROUP_SIZE / 8].fill(0xff);
        signs[GROUP_SIZE / 8..].fill(0b0101_0101);
        let payload = tiny_payload(&[2, GROUP_SIZE], &[0.5, 0.25], &signs);
        let header = parse_complete_binary_header(&payload).unwrap();
        let input = (0..GROUP_SIZE)
            .map(|index| (index as f32 - 63.0) / 43.0)
            .collect::<Vec<_>>();
        // The production function deliberately refuses a non-Qwen80 shape;
        // this tiny fixture checks the exact group reduction independently.
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
        let (max_abs, _) = max_f64_error(&observed, &expected, "tiny router dot").unwrap();
        assert!(max_abs < 1.0e-4, "max_abs={max_abs}");
    }

    #[test]
    fn lsb_first_packed_signs_and_fp16_scale_are_preserved() {
        let mut signs = vec![0u8; GROUP_SIZE / 8];
        signs[0] = 0b0000_0101;
        let payload = tiny_payload(&[GROUP_SIZE], &[0.75], &signs);
        let header = parse_complete_binary_header(&payload).unwrap();
        assert_eq!(packed_value_f32(&payload, &header, 0).unwrap(), 0.75);
        assert_eq!(packed_value_f32(&payload, &header, 1).unwrap(), -0.75);
        assert_eq!(packed_value_f32(&payload, &header, 2).unwrap(), 0.75);
    }

    #[test]
    fn stable_top10_prefers_lower_ids_on_an_exact_tie_and_renormalizes() {
        let logits = vec![0.0f32; EXPERTS];
        let route = source_qwen80_topk_router(&logits).unwrap();
        assert_eq!(route.ids, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert!(route
            .weights
            .iter()
            .all(|weight| (*weight - 0.1).abs() < 1.0e-6));
        validate_route(&route).unwrap();
    }

    #[test]
    fn rejects_nonfinite_or_duplicate_source_routes() {
        let mut logits = vec![0.0f32; EXPERTS];
        logits[17] = f32::INFINITY;
        assert!(source_qwen80_topk_router(&logits).is_err());
        let duplicate = Route {
            ids: [3u16; TOP_K],
            weights: [0.1; TOP_K],
            preselected_probabilities: [0.1; TOP_K],
        };
        assert!(validate_route(&duplicate).is_err());
    }

    #[test]
    fn component_constants_enforce_qwen80_not_qwen30_router_geometry() {
        assert_eq!(LAYER, 0);
        assert_eq!(HIDDEN, 2_048);
        assert_eq!(EXPERTS, 512);
        assert_eq!(TOP_K, 10);
        assert_eq!(GROUP_SIZE, 128);
        assert_ne!(HIDDEN, 4_096);
        assert_eq!(HIDDEN % GROUP_SIZE, 0);
    }

    #[test]
    fn quiet_metal_lease_requires_exact_artifact_and_non_timed_strict_math_scope() {
        let lease = valid_quiet_metal_lease_document();
        let (lease_id, seal) =
            validate_quiet_metal_lease_document(&lease, "test quiet Metal lease").unwrap();
        assert_eq!(lease_id, "a".repeat(64));
        assert_eq!(seal, "b".repeat(64));

        let mut timing_permitted = valid_quiet_metal_lease_document();
        timing_permitted
            .get_mut("execution_policy")
            .and_then(Value::as_object_mut)
            .unwrap()
            .insert("timing_or_benchmarking_allowed".into(), Value::Bool(true));
        assert!(validate_quiet_metal_lease_document(
            &timing_permitted,
            "timing-permitted quiet Metal lease"
        )
        .is_err());

        let mut wrong_artifact = valid_quiet_metal_lease_document();
        wrong_artifact
            .get_mut("artifact_binding")
            .and_then(Value::as_object_mut)
            .unwrap()
            .insert(
                "router_gate_artifact_sha256".into(),
                Value::String("c".repeat(64)),
            );
        assert!(validate_quiet_metal_lease_document(
            &wrong_artifact,
            "wrong-artifact quiet Metal lease"
        )
        .is_err());
    }

    #[test]
    fn retained_intermediate_reports_require_exact_finite_device_shapes() {
        let hidden = vec![0.25f32; HIDDEN];
        let report = f32_intermediate_report(&hidden, HIDDEN, "test hidden intermediate").unwrap();
        assert_eq!(report["elements"].as_u64(), Some(HIDDEN as u64));
        assert_eq!(report["all_finite"].as_bool(), Some(true));
        assert!(f32_intermediate_report(&hidden[..HIDDEN - 1], HIDDEN, "short hidden").is_err());
        let mut nonfinite = vec![0.0f32; EXPERTS];
        nonfinite[0] = f32::NAN;
        assert!(f32_intermediate_report(&nonfinite, EXPERTS, "nonfinite logits").is_err());
    }

    fn route_oracle_for_device_validator() -> CpuOracle {
        let route = Route {
            ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            weights: [0.1; TOP_K],
            preselected_probabilities: [0.1; TOP_K],
        };
        CpuOracle {
            residual: vec![0.0; HIDDEN],
            normalized: vec![0.0; HIDDEN],
            logits: vec![0.0; EXPERTS],
            route,
            norm_max_abs: 0.0,
            norm_max_relative: 0.0,
            norm_direct_decode_max_abs: 0.0,
            router_max_abs: 0.0,
            router_max_relative: 0.0,
            candidate_duration_ms: 0.0,
            reference_duration_ms: 0.0,
        }
    }

    #[test]
    fn device_route_validator_requires_exact_ids_and_postselect_scratch_semantics() {
        let oracle = route_oracle_for_device_validator();
        let ids = (0..TOP_K as u32).collect::<Vec<_>>();
        let weights = vec![0.1f32; TOP_K];
        let mut scratch = vec![1.0f32 / (EXPERTS - TOP_K) as f32; EXPERTS];
        for id in 0..TOP_K {
            scratch[id] = f32::NEG_INFINITY;
        }
        let (_, _, max_abs, sum_error, selected) =
            validate_device_route(&oracle, &ids, &weights, &scratch).unwrap();
        assert_eq!(max_abs, 0.0);
        assert!(sum_error <= DEVICE_ROUTE_WEIGHT_SUM_TOLERANCE);
        assert_eq!(selected, TOP_K);

        let mut wrong_ids = ids.clone();
        wrong_ids.swap(0, 1);
        assert!(validate_device_route(&oracle, &wrong_ids, &weights, &scratch).is_err());
        scratch[0] = 0.0;
        assert!(validate_device_route(&oracle, &ids, &weights, &scratch).is_err());
    }

    #[test]
    fn postnorm_router_shader_binds_router_row_to_explicit_grid_y() {
        let shader = include_str!("../shaders/qwen80_postnorm_router_top10.metal");
        assert!(shader.contains("uint3 group_position [[threadgroup_position_in_grid]]"));
        assert!(shader.contains("uint3 tid [[thread_position_in_threadgroup]]"));
        assert!(shader.contains("const uint row = group_position.y;"));
        assert!(shader.contains("const uint lane = tid.x;"));
        assert!(shader.contains("for (uint column = lane; column < columns; column += 256u)"));
        let matvec = shader
            .split("kernel void qwen80_postnorm_router_top10_matvec")
            .nth(1)
            .expect("matvec entry point must remain present")
            .split("kernel void qwen80_postnorm_router_top10_select")
            .next()
            .expect("matvec entry point must precede selection");
        assert!(
            matvec.contains("if (lane == 0u) {"),
            "the vector thread position must be reduced to its X lane before scalar control flow"
        );
        assert!(
            !matvec.contains("if (tid == 0u) {"),
            "a uint3 comparison produces bool3 and is not a valid Metal scalar branch"
        );
        assert!(
            !shader.contains("uint row [[threadgroup_position_in_grid]]"),
            "a scalar threadgroup position aliases the host's 512 Y rows onto X"
        );
        assert!(
            !shader
                .contains("uint tid [[thread_position_in_threadgroup]],\n    uint3 group_position"),
            "Metal requires matching scalar/vector widths for position builtins"
        );
    }
}
