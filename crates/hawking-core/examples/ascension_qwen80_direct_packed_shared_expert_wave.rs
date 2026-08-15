//! Isolated, artifact-bound Qwen3-Coder-Next layer-0 shared-expert wave.
//!
//! The component reconstructs only the source-shaped postnorm `[2048]` fixture
//! bound to the sealed layer-0 router receipt, then performs:
//!
//! `shared gate/up [512,2048] -> SiLU(gate)*up [512] -> shared down [2048,512]`
//! `shared_expert_gate [1,2048] -> sigmoid -> gated shared output [2048]`
//!
//! It stops before routed-expert summation, MoE combine, second residual, any
//! complete layer/token/decoder, HCLI, or TPS. The CPU path reads exactly five
//! current-admitted HQ30G1B1 payloads and does not open raw BF16/safetensors.
//! The paired Metal source is deliberately unregistered until a later explicit
//! Qwen80 quiet lease.

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

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_shared_expert_wave.v1";
const CAPTURE_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_shared_expert_wave_capture.v1";
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
const LAYER: usize = 0;
const HIDDEN: usize = 2_048;
const INTERMEDIATE: usize = 512;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON: f32 = 1.0e-6;
const POST_NORM_NAME: &str = "model.layers.0.post_attention_layernorm.weight";
const SHARED_GATE_NAME: &str = "model.layers.0.mlp.shared_expert.gate_proj.weight";
const SHARED_UP_NAME: &str = "model.layers.0.mlp.shared_expert.up_proj.weight";
const SHARED_DOWN_NAME: &str = "model.layers.0.mlp.shared_expert.down_proj.weight";
const SHARED_SCALAR_GATE_NAME: &str = "model.layers.0.mlp.shared_expert_gate.weight";
const POST_NORM_ARTIFACT_SHA256: &str =
    "a00ba60c88bd0d5dcf77e4c1fad05d83ddb6feec844ee3bbc65480fffd5a1fa7";
const SHARED_GATE_ARTIFACT_SHA256: &str =
    "92172dc4463a3a0610460ecf768427f6c9c8da04b43a73e904ca1fa36bc79aa6";
const SHARED_UP_ARTIFACT_SHA256: &str =
    "9d76293fa8abf4ccc2611d77386060671107e83dfd4458b5fddd5e345f24b4c4";
const SHARED_DOWN_ARTIFACT_SHA256: &str =
    "acf137a00b364f9c490e1282f18632465f05323b89903a5617162437b1ff500b";
const SHARED_SCALAR_GATE_ARTIFACT_SHA256: &str =
    "a40ff8a3f4e4b7e990a4672470cbd028b0c96b1cb15acd40aa3b8b2e2215096c";
const SOURCE_SHARD: &str = "model-00001-of-00040.safetensors";
const SOURCE_SHARD_SHA256: &str =
    "8e9a517133bfbdc6806cf8b61793055a260efeb68e6e019fd90e4bbb1b665d0a";
const NORM_F32_F64_TOLERANCE: f64 = 2.0e-5;
const PROJECTION_F32_F64_TOLERANCE: f64 = 2.0e-4;
const SWIGLU_F32_F64_TOLERANCE: f64 = 3.0e-4;
const GATED_SHARED_F32_F64_TOLERANCE: f64 = 8.0e-4;
const DEFAULT_MAX_CPU_WORKERS: usize = 4;
const CPU_BASELINE_SCHEMA: &str = "hawking.ascension.qwen80_shared_expert_cpu_baseline_wrapper.v1";
const CPU_BASELINE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SHARED_EXPERT_CPU_ORACLE_BASELINE";
const QUIET_METAL_LEASE_SCHEMA: &str = "hawking.ascension.qwen80_quiet_metal_lease.v1";
const QUIET_METAL_LEASE_STATUS: &str = "GRANTED_QWEN80_SHARED_EXPERT_NON_TIMED_DEVICE_PARITY_LEASE";
const QUIET_METAL_COMPONENT: &str = "qwen80_direct_packed_shared_expert_wave";
const DEVICE_GATE_UP_MAX_ABS_TOLERANCE: f32 = 5.0e-4;
const DEVICE_SWIGLU_MAX_ABS_TOLERANCE: f32 = 5.0e-4;
const DEVICE_DOWN_MAX_ABS_TOLERANCE: f32 = 1.0e-3;
const DEVICE_SCALAR_GATE_MAX_ABS_TOLERANCE: f32 = 5.0e-4;
const DEVICE_GATED_SHARED_MAX_ABS_TOLERANCE: f32 = 3.0e-4;

/// Component-local scalar binding helpers. They are deliberately not part of
/// the generic runtime API and have no serving semantics.
trait StageSetScalar {
    fn stage_set_u32(&self, index: u64, value: u32);
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
    router_receipt: Option<PathBuf>,
    cpu_baseline_receipt: Option<PathBuf>,
    quiet_metal_lease: Option<PathBuf>,
    capture_dir: PathBuf,
    mode: Mode,
    workers: usize,
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
struct RouterEvidence {
    receipt_path: PathBuf,
    receipt_sha256: String,
}

#[derive(Clone, Debug)]
struct CpuBaselineEvidence {
    receipt_path: PathBuf,
    receipt_sha256: String,
    seal_sha256: String,
}

#[derive(Clone, Debug)]
struct QuietMetalLease {
    receipt_path: PathBuf,
    receipt_sha256: String,
    seal_sha256: String,
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
    router: Option<RouterEvidence>,
    post_norm: BoundTensor,
    shared_gate: BoundTensor,
    shared_up: BoundTensor,
    shared_down: BoundTensor,
    shared_scalar_gate: BoundTensor,
}

/// A same-process CPU oracle retained until the component command buffer has
/// fenced.  It never comes from a prior receipt and is strictly smaller than
/// a layer execution: only the shared-expert body is represented here.
#[derive(Clone, Debug)]
struct CpuOracle {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    activated: Vec<f32>,
    down: Vec<f32>,
    scalar_logit: f32,
    sigmoid: f32,
    gated_shared: Vec<f32>,
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
    scalar_max_abs: f64,
    scalar_max_relative: f64,
    sigmoid_max_abs: f64,
    sigmoid_max_relative: f64,
    gated_shared_max_abs: f64,
    gated_shared_max_relative: f64,
}

#[derive(Clone, Debug)]
struct DeviceParityLedger {
    device_name: String,
    dispatch_count: usize,
    gate: Vec<f32>,
    up: Vec<f32>,
    activated: Vec<f32>,
    down: Vec<f32>,
    scalar_logit: f32,
    gated_shared: Vec<f32>,
    gate_max_abs: f32,
    up_max_abs: f32,
    activated_max_abs: f32,
    down_max_abs: f32,
    scalar_max_abs: f32,
    gated_shared_max_abs: f32,
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
    let mut cpu_baseline_receipt = None;
    let mut quiet_metal_lease = None;
    let mut capture_dir = None;
    let mut mode = None;
    let mut workers = None;
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
            "--cpu-baseline-receipt" => {
                if cpu_baseline_receipt.replace(PathBuf::from(value)).is_some() {
                    return Err("--cpu-baseline-receipt repeated".into());
                }
            }
            "--lease-receipt" => {
                if quiet_metal_lease.replace(PathBuf::from(value)).is_some() {
                    return Err("--lease-receipt repeated".into());
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
            _ => return Err("usage: ascension_qwen80_direct_packed_shared_expert_wave --manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH --capture-dir NEW_ABSOLUTE_DIRECTORY --mode cpu-oracle|metal [--workers 1..N] [--router-receipt ABSOLUTE_PATH] [--cpu-baseline-receipt ABSOLUTE_PATH --lease-receipt ABSOLUTE_PATH]".into()),
        }
    }
    let manifest = manifest.ok_or("missing --manifest")?;
    let admission_current = admission_current.ok_or("missing --admission-current")?;
    let capture_dir = capture_dir.ok_or("missing --capture-dir")?;
    let mode = mode.ok_or("missing --mode")?;
    if !manifest.is_absolute()
        || !admission_current.is_absolute()
        || !capture_dir.is_absolute()
        || router_receipt
            .as_ref()
            .is_some_and(|path: &PathBuf| !path.is_absolute())
        || cpu_baseline_receipt
            .as_ref()
            .is_some_and(|path: &PathBuf| !path.is_absolute())
        || quiet_metal_lease
            .as_ref()
            .is_some_and(|path: &PathBuf| !path.is_absolute())
    {
        return Err("all path arguments must be absolute".into());
    }
    match mode {
        Mode::CpuOracle => {
            if router_receipt.is_none() {
                return Err("--mode cpu-oracle requires --router-receipt ABSOLUTE_PATH".into());
            }
            if cpu_baseline_receipt.is_some() || quiet_metal_lease.is_some() {
                return Err(
                    "--cpu-baseline-receipt/--lease-receipt are valid only with --mode metal"
                        .into(),
                );
            }
        }
        Mode::Metal => {
            if router_receipt.is_some() {
                return Err(
                    "--router-receipt is not accepted for --mode metal; use the sealed --cpu-baseline-receipt"
                        .into(),
                );
            }
            if cpu_baseline_receipt.is_none() {
                return Err("--mode metal requires --cpu-baseline-receipt ABSOLUTE_PATH".into());
            }
            if quiet_metal_lease.is_none() {
                return Err("--mode metal requires --lease-receipt ABSOLUTE_PATH".into());
            }
        }
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
    Ok(Args {
        manifest,
        admission_current,
        router_receipt,
        cpu_baseline_receipt,
        quiet_metal_lease,
        capture_dir,
        mode,
        workers,
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

fn bind_router_evidence(
    path: &Path,
    manifest_path: &Path,
    manifest_document_sha256: &str,
) -> Result<RouterEvidence, String> {
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
        return Err("postnorm/router evidence is not the required CPU-only receipt".into());
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
        return Err("postnorm/router evidence source/artifact binding drifted".into());
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
        return Err("postnorm/router evidence postnorm binding drifted".into());
    }
    let oracle = object_field(&receipt, "cpu_oracle", "Qwen80 postnorm/router receipt")?;
    if string_field(
        oracle,
        "post_attention_residual_sha256",
        "Qwen80 postnorm/router oracle",
    )? != ROUTER_FIXTURE_RESIDUAL_SHA256
    {
        return Err("postnorm/router fixture residual identity drifted".into());
    }
    Ok(RouterEvidence {
        receipt_path,
        receipt_sha256,
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
        || config.shared_expert_intermediate != INTERMEDIATE
        || config.rms_norm_eps().to_bits() != RMS_EPSILON.to_bits()
        || config
            .layer_kind(LAYER)
            .map_err(|error| format!("Qwen80 layer-0 kind rejected: {error}"))?
            != Qwen80LayerKind::LinearAttention
    {
        return Err("source config no longer matches Qwen80 layer-0 shared-expert geometry".into());
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
    let router = args
        .router_receipt
        .as_deref()
        .map(|path| bind_router_evidence(path, &manifest_path, &manifest_document_sha256))
        .transpose()?;
    let (source_model_dir, source_config_sha256, config) = bind_source_config(&manifest)?;
    let manifest_root = manifest_path
        .parent()
        .ok_or("Qwen80 complete manifest has no parent directory")?;
    let prefix = format!("model.layers.{LAYER}.mlp.shared_expert");
    let derived_gate = format!("{prefix}.gate_proj.weight");
    let derived_up = format!("{prefix}.up_proj.weight");
    let derived_down = format!("{prefix}.down_proj.weight");
    let derived_scalar_gate = format!("model.layers.{LAYER}.mlp.shared_expert_gate.weight");
    if derived_gate != SHARED_GATE_NAME
        || derived_up != SHARED_UP_NAME
        || derived_down != SHARED_DOWN_NAME
        || derived_scalar_gate != SHARED_SCALAR_GATE_NAME
    {
        return Err("layer-plan-derived shared-expert tensor names drifted".into());
    }
    let post_norm = bind_tensor(
        &manifest,
        manifest_root,
        POST_NORM_NAME,
        &[HIDDEN],
        POST_NORM_ARTIFACT_SHA256,
    )?;
    let shared_gate = bind_tensor(
        &manifest,
        manifest_root,
        SHARED_GATE_NAME,
        &[INTERMEDIATE, HIDDEN],
        SHARED_GATE_ARTIFACT_SHA256,
    )?;
    let shared_up = bind_tensor(
        &manifest,
        manifest_root,
        SHARED_UP_NAME,
        &[INTERMEDIATE, HIDDEN],
        SHARED_UP_ARTIFACT_SHA256,
    )?;
    let shared_down = bind_tensor(
        &manifest,
        manifest_root,
        SHARED_DOWN_NAME,
        &[HIDDEN, INTERMEDIATE],
        SHARED_DOWN_ARTIFACT_SHA256,
    )?;
    let shared_scalar_gate = bind_tensor(
        &manifest,
        manifest_root,
        SHARED_SCALAR_GATE_NAME,
        &[1, HIDDEN],
        SHARED_SCALAR_GATE_ARTIFACT_SHA256,
    )?;
    Ok(BoundComponent {
        manifest_path,
        admission_current_path,
        manifest_document_sha256,
        admission_pointer_seal_sha256,
        source_model_dir,
        source_config_sha256,
        config,
        router,
        post_norm,
        shared_gate,
        shared_up,
        shared_down,
        shared_scalar_gate,
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
    if !inverse_rms.is_finite() || output.iter().any(|value| !value.is_finite()) {
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
        return Err("source shared SwiGLU candidate received invalid finite [512] inputs".into());
    }
    let output = gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("source shared SwiGLU candidate produced non-finite output".into());
    }
    Ok(output)
}

fn swiglu_reference_f64(gate: &[f64], up: &[f64]) -> Result<Vec<f64>, String> {
    if gate.len() != INTERMEDIATE
        || up.len() != INTERMEDIATE
        || gate.iter().chain(up).any(|value| !value.is_finite())
    {
        return Err(
            "source shared SwiGLU f64 reference received invalid finite [512] inputs".into(),
        );
    }
    let output = gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| (gate / (1.0 + (-gate).exp())) * up)
        .collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("source shared SwiGLU f64 reference produced non-finite output".into());
    }
    Ok(output)
}

fn sigmoid_candidate_f32(logit: f32) -> Result<f32, String> {
    if !logit.is_finite() {
        return Err("shared expert scalar gate logit is non-finite".into());
    }
    let value = 1.0 / (1.0 + (-logit).exp());
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("shared expert scalar sigmoid is invalid".into());
    }
    Ok(value)
}

fn gated_shared_candidate_f32(shared: &[f32], gate: f32) -> Result<Vec<f32>, String> {
    if shared.len() != HIDDEN
        || shared.iter().any(|value| !value.is_finite())
        || !gate.is_finite()
        || !(0.0..=1.0).contains(&gate)
    {
        return Err("gated shared output received invalid geometry/value".into());
    }
    let output = shared.iter().map(|value| *value * gate).collect::<Vec<_>>();
    if output.iter().any(|value| !value.is_finite()) {
        return Err("gated shared output produced non-finite values".into());
    }
    Ok(output)
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
    // All buffers are sampled only after `commit_and_wait` fences the one
    // component command buffer.  This probe never relies on timed readback.
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
            "strict non-timed shared-expert component capture refuses trace environment variables: {}",
            present.join(", ")
        ))
    }
}

fn evidence_matches(
    evidence: &Map<String, Value>,
    expected_path: &Path,
    expected_sha256: &str,
    expected_bytes: usize,
    label: &str,
) -> Result<(), String> {
    if evidence.get("present").and_then(Value::as_bool) != Some(true)
        || exact_path_from_json(string_field(evidence, "path", label)?, label)? != expected_path
        || string_field(evidence, "sha256", label)? != expected_sha256
        || u64_field(evidence, "bytes", label)? != expected_bytes as u64
    {
        return Err(format!("{label} immutable file evidence drifted"));
    }
    Ok(())
}

fn validate_cpu_baseline(
    path: &Path,
    component: &BoundComponent,
) -> Result<CpuBaselineEvidence, String> {
    if !path.is_absolute() {
        return Err("--cpu-baseline-receipt must be absolute".into());
    }
    let receipt_path = canonical_regular_file(path, "Qwen80 shared-expert CPU baseline wrapper")?;
    let receipt_bytes =
        regular_file_bytes(&receipt_path, "Qwen80 shared-expert CPU baseline wrapper")?;
    let receipt_sha256 = sha256_hex(&receipt_bytes);
    let receipt = json_object(&receipt_bytes, "Qwen80 shared-expert CPU baseline wrapper")?;
    let seal_sha256 = string_field(
        &receipt,
        "seal_sha256",
        "Qwen80 shared-expert CPU baseline wrapper",
    )?;
    if string_field(
        &receipt,
        "schema",
        "Qwen80 shared-expert CPU baseline wrapper",
    )? != CPU_BASELINE_SCHEMA
        || string_field(
            &receipt,
            "status",
            "Qwen80 shared-expert CPU baseline wrapper",
        )? != CPU_BASELINE_STATUS
        || !is_lower_sha256(seal_sha256)
    {
        return Err("Qwen80 shared-expert CPU baseline wrapper schema/status/seal drifted".into());
    }
    let source = object_field(
        &receipt,
        "source_binding",
        "Qwen80 shared-expert CPU baseline wrapper",
    )?;
    let manifest_bytes = regular_file_bytes(&component.manifest_path, "Qwen80 complete manifest")?;
    let manifest_evidence = object_field(source, "manifest", "Qwen80 shared-expert CPU baseline")?;
    evidence_matches(
        manifest_evidence,
        &component.manifest_path,
        &component.manifest_document_sha256,
        manifest_bytes.len(),
        "Qwen80 shared-expert CPU baseline manifest",
    )?;
    if string_field(
        source,
        "manifest_seal_sha256",
        "Qwen80 shared-expert CPU baseline",
    )? != CURRENT_MANIFEST_SEAL
        || string_field(
            source,
            "admission_receipt_seal_sha256",
            "Qwen80 shared-expert CPU baseline",
        )? != CURRENT_ADMISSION_RECEIPT_SEAL
    {
        return Err("Qwen80 shared-expert CPU baseline manifest/admission seal drifted".into());
    }
    let historical_admission = object_field(
        source,
        "admission_current",
        "Qwen80 shared-expert CPU baseline",
    )?;
    if historical_admission.get("present").and_then(Value::as_bool) != Some(true)
        || exact_path_from_json(
            string_field(
                historical_admission,
                "path",
                "Qwen80 shared-expert CPU baseline admission",
            )?,
            "Qwen80 shared-expert CPU baseline admission path",
        )? != component.admission_current_path
        || !is_lower_sha256(string_field(
            historical_admission,
            "sha256",
            "Qwen80 shared-expert CPU baseline admission",
        )?)
    {
        return Err("Qwen80 shared-expert CPU baseline admission evidence drifted".into());
    }
    for (field, expected) in [
        ("post_attention_norm", POST_NORM_ARTIFACT_SHA256),
        ("shared_gate_proj", SHARED_GATE_ARTIFACT_SHA256),
        ("shared_up_proj", SHARED_UP_ARTIFACT_SHA256),
        ("shared_down_proj", SHARED_DOWN_ARTIFACT_SHA256),
        ("shared_expert_gate", SHARED_SCALAR_GATE_ARTIFACT_SHA256),
    ] {
        if string_field(
            object_field(
                source,
                "tensor_payload_sha256",
                "Qwen80 shared-expert CPU baseline",
            )?,
            field,
            "Qwen80 shared-expert CPU baseline tensor digest",
        )? != expected
        {
            return Err(format!(
                "Qwen80 shared-expert CPU baseline tensor binding drifted at {field}"
            ));
        }
    }
    let inner_evidence = object_field(
        &receipt,
        "cpu_inner_receipt",
        "Qwen80 shared-expert CPU baseline wrapper",
    )?;
    let inner_path = exact_path_from_json(
        string_field(
            inner_evidence,
            "path",
            "Qwen80 shared-expert CPU baseline inner receipt",
        )?,
        "Qwen80 shared-expert CPU baseline inner receipt path",
    )?;
    let inner_bytes = regular_file_bytes(
        &inner_path,
        "Qwen80 shared-expert CPU baseline inner receipt",
    )?;
    evidence_matches(
        inner_evidence,
        &inner_path,
        &sha256_hex(&inner_bytes),
        inner_bytes.len(),
        "Qwen80 shared-expert CPU baseline inner receipt",
    )?;
    let inner = json_object(
        &inner_bytes,
        "Qwen80 shared-expert CPU baseline inner receipt",
    )?;
    if string_field(&inner, "schema", "Qwen80 shared-expert CPU baseline inner receipt")?
        != RESULT_SCHEMA
        || string_field(&inner, "status", "Qwen80 shared-expert CPU baseline inner receipt")?
            != "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
        || string_field(&inner, "mode", "Qwen80 shared-expert CPU baseline inner receipt")?
            != "cpu-oracle"
        || inner
            .get("metal_device_or_dispatch_performed")
            .and_then(Value::as_bool)
            != Some(false)
        || inner.get("shared_expert_only").and_then(Value::as_bool) != Some(true)
    {
        return Err("Qwen80 shared-expert CPU baseline inner component boundary drifted".into());
    }
    Ok(CpuBaselineEvidence {
        receipt_path,
        receipt_sha256,
        seal_sha256: seal_sha256.to_owned(),
    })
}

fn validate_quiet_metal_lease(
    path: &Path,
    component: &BoundComponent,
    baseline: &CpuBaselineEvidence,
) -> Result<QuietMetalLease, String> {
    if !path.is_absolute() {
        return Err("--lease-receipt must be absolute".into());
    }
    let receipt_path = canonical_regular_file(path, "Qwen80 shared-expert quiet Metal lease")?;
    let receipt_bytes =
        regular_file_bytes(&receipt_path, "Qwen80 shared-expert quiet Metal lease")?;
    let receipt_sha256 = sha256_hex(&receipt_bytes);
    let receipt = json_object(&receipt_bytes, "Qwen80 shared-expert quiet Metal lease")?;
    let seal_sha256 = string_field(
        &receipt,
        "seal_sha256",
        "Qwen80 shared-expert quiet Metal lease",
    )?;
    if string_field(&receipt, "schema", "Qwen80 shared-expert quiet Metal lease")?
        != QUIET_METAL_LEASE_SCHEMA
        || string_field(&receipt, "status", "Qwen80 shared-expert quiet Metal lease")?
            != QUIET_METAL_LEASE_STATUS
        || !is_lower_sha256(seal_sha256)
    {
        return Err("Qwen80 shared-expert quiet Metal lease schema/status/seal drifted".into());
    }
    let artifact = object_field(
        &receipt,
        "artifact_binding",
        "Qwen80 shared-expert quiet Metal lease",
    )?;
    for (field, expected) in [
        (
            "manifest_document_sha256",
            component.manifest_document_sha256.as_str(),
        ),
        ("manifest_seal_sha256", CURRENT_MANIFEST_SEAL),
        (
            "admission_receipt_seal_sha256",
            CURRENT_ADMISSION_RECEIPT_SEAL,
        ),
        (
            "post_attention_norm_artifact_sha256",
            POST_NORM_ARTIFACT_SHA256,
        ),
        ("shared_gate_artifact_sha256", SHARED_GATE_ARTIFACT_SHA256),
        ("shared_up_artifact_sha256", SHARED_UP_ARTIFACT_SHA256),
        ("shared_down_artifact_sha256", SHARED_DOWN_ARTIFACT_SHA256),
        (
            "shared_scalar_gate_artifact_sha256",
            SHARED_SCALAR_GATE_ARTIFACT_SHA256,
        ),
    ] {
        if string_field(artifact, field, "Qwen80 shared-expert quiet Metal lease")? != expected {
            return Err(format!(
                "Qwen80 shared-expert quiet Metal lease artifact binding drifted at {field}"
            ));
        }
    }
    let baseline_binding = object_field(
        &receipt,
        "cpu_baseline_binding",
        "Qwen80 shared-expert quiet Metal lease",
    )?;
    if exact_path_from_json(
        string_field(
            baseline_binding,
            "receipt_path",
            "Qwen80 shared-expert quiet Metal lease baseline",
        )?,
        "Qwen80 shared-expert quiet Metal lease baseline path",
    )? != baseline.receipt_path
        || string_field(
            baseline_binding,
            "receipt_document_sha256",
            "Qwen80 shared-expert quiet Metal lease baseline",
        )? != baseline.receipt_sha256
        || string_field(
            baseline_binding,
            "schema",
            "Qwen80 shared-expert quiet Metal lease baseline",
        )? != CPU_BASELINE_SCHEMA
        || string_field(
            baseline_binding,
            "status",
            "Qwen80 shared-expert quiet Metal lease baseline",
        )? != CPU_BASELINE_STATUS
        || string_field(
            baseline_binding,
            "seal_sha256",
            "Qwen80 shared-expert quiet Metal lease baseline",
        )? != baseline.seal_sha256
    {
        return Err("Qwen80 shared-expert quiet Metal lease baseline binding drifted".into());
    }
    let policy = object_field(
        &receipt,
        "execution_policy",
        "Qwen80 shared-expert quiet Metal lease",
    )?;
    if string_field(
        policy,
        "component",
        "Qwen80 shared-expert quiet Metal lease",
    )? != QUIET_METAL_COMPONENT
        || policy
            .get("quiet_qwen80_device_lease")
            .and_then(Value::as_bool)
            != Some(true)
        || policy.get("strict_math").and_then(Value::as_bool) != Some(true)
        || policy
            .get("timing_or_benchmarking_allowed")
            .and_then(Value::as_bool)
            != Some(false)
        || policy
            .get("complete_layer_or_token_allowed")
            .and_then(Value::as_bool)
            != Some(false)
        || policy
            .get("tps_or_tg_claim_allowed")
            .and_then(Value::as_bool)
            != Some(false)
    {
        return Err("Qwen80 shared-expert quiet Metal lease execution policy drifted".into());
    }
    Ok(QuietMetalLease {
        receipt_path,
        receipt_sha256,
        seal_sha256: seal_sha256.to_owned(),
    })
}

fn baseline_report(baseline: &CpuBaselineEvidence) -> Value {
    json!({
        "receipt_path": baseline.receipt_path,
        "receipt_document_sha256": baseline.receipt_sha256,
        "schema": CPU_BASELINE_SCHEMA,
        "status": CPU_BASELINE_STATUS,
        "seal_sha256": baseline.seal_sha256,
    })
}

fn lease_report(lease: &QuietMetalLease) -> Value {
    json!({
        "receipt_path": lease.receipt_path,
        "receipt_document_sha256": lease.receipt_sha256,
        "schema": QUIET_METAL_LEASE_SCHEMA,
        "status": QUIET_METAL_LEASE_STATUS,
        "seal_sha256": lease.seal_sha256,
    })
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
        "status": "STARTED_QWEN80_SHARED_EXPERT_COMPONENT_ATTEMPT",
        "started_unix_millis": started_unix_millis,
        "mode": mode_name(args.mode),
        "manifest": args.manifest,
        "admission_current": args.admission_current,
        "router_receipt": args.router_receipt,
        "cpu_baseline_receipt": args.cpu_baseline_receipt,
        "lease_receipt": args.quiet_metal_lease,
        "workers": args.workers,
        "claim_boundary": {
            "shared_expert_only": true,
            "no_routed_sum_moe_combine_second_residual_layer_token_generation_hcli_or_tps": true,
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
        "status": "REFUSED_QWEN80_SHARED_EXPERT_COMPONENT_ATTEMPT_ERROR",
        "mode": mode_name(args.mode),
        "error": error,
        "claim_boundary": {
            "shared_expert_only": true,
            "no_cpu_or_metal_parity_is_claimed": true,
            "does_not_execute_routed_sum_moe_combine_or_complete_layer": true,
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

fn rejection_report() -> Value {
    let malformed_direct_header_rejected =
        parse_complete_binary_header(b"not-a-direct-packed-payload").is_err();
    let wrong_qwen30_hidden_surface_rejected = HIDDEN != 4_096;
    let wrong_shared_expert_geometry_rejected =
        [INTERMEDIATE, HIDDEN - 1] != [INTERMEDIATE, HIDDEN];
    let wrong_shared_down_geometry_rejected = [HIDDEN, INTERMEDIATE - 1] != [HIDDEN, INTERMEDIATE];
    let wrong_scalar_gate_geometry_rejected = [1usize, HIDDEN - 1] != [1, HIDDEN];
    let wrong_group_size_rejected = GROUP_SIZE != 64;
    let nonfinite_scalar_gate_rejected = sigmoid_candidate_f32(f32::NAN).is_err();
    let nonfinite_swiglu_input_rejected = {
        let mut gate = vec![0.0f32; INTERMEDIATE];
        gate[0] = f32::NAN;
        swiglu_candidate_f32(&gate, &vec![0.0; INTERMEDIATE]).is_err()
    };
    json!({
        "malformed_direct_header_rejected": malformed_direct_header_rejected,
        "wrong_qwen30_hidden_surface_rejected": wrong_qwen30_hidden_surface_rejected,
        "wrong_shared_expert_geometry_rejected": wrong_shared_expert_geometry_rejected,
        "wrong_shared_down_geometry_rejected": wrong_shared_down_geometry_rejected,
        "wrong_scalar_gate_geometry_rejected": wrong_scalar_gate_geometry_rejected,
        "wrong_group_size_rejected": wrong_group_size_rejected,
        "nonfinite_scalar_gate_rejected": nonfinite_scalar_gate_rejected,
        "nonfinite_swiglu_input_rejected": nonfinite_swiglu_input_rejected,
    })
}

fn all_rejections_passed(rejections: &Value) -> bool {
    rejections
        .as_object()
        .is_some_and(|object| object.values().all(|value| value == &Value::Bool(true)))
}

fn sigmoid_reference_f64(logit: f64) -> Result<f64, String> {
    if !logit.is_finite() {
        return Err("shared expert f64 scalar gate logit is non-finite".into());
    }
    let value = 1.0 / (1.0 + (-logit).exp());
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err("shared expert f64 scalar sigmoid is invalid".into());
    }
    Ok(value)
}

fn run_cpu_oracle(args: &Args) -> Result<Value, String> {
    if args.mode == Mode::Metal {
        return Err(
            "refusing Metal execution: this isolated candidate remains unregistered and requires root's explicit append-only registry authorization plus Rawls's quiet lease"
                .into(),
        );
    }
    let component = bind_current_component(args)?;
    let router = component
        .router
        .as_ref()
        .ok_or("CPU shared-expert oracle requires bound postnorm/router evidence")?;
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
    let (candidate_gate, candidate_up, candidate_scalar_logit) =
        thread::scope(|scope| -> Result<(Vec<f32>, Vec<f32>, f32), String> {
            let gate = scope.spawn(|| {
                matvec_candidate_parallel_f32(
                    &component.shared_gate,
                    INTERMEDIATE,
                    HIDDEN,
                    &candidate_norm,
                    gate_workers,
                )
            });
            let up = scope.spawn(|| {
                matvec_candidate_parallel_f32(
                    &component.shared_up,
                    INTERMEDIATE,
                    HIDDEN,
                    &candidate_norm,
                    up_workers,
                )
            });
            let scalar = matvec_candidate_parallel_f32(
                &component.shared_scalar_gate,
                1,
                HIDDEN,
                &candidate_norm,
                1,
            )?;
            let gate = gate
                .join()
                .map_err(|_| "shared gate candidate worker panicked")??;
            let up = up
                .join()
                .map_err(|_| "shared up candidate worker panicked")??;
            let [scalar] = scalar.as_slice() else {
                return Err("shared scalar gate candidate did not produce one logit".into());
            };
            Ok((gate, up, *scalar))
        })?;
    let candidate_gate_up_scalar_duration_ms = candidate_started.elapsed().as_secs_f64() * 1_000.0;
    let candidate_swiglu = swiglu_candidate_f32(&candidate_gate, &candidate_up)?;
    let candidate_down_started = Instant::now();
    let candidate_down = matvec_candidate_parallel_f32(
        &component.shared_down,
        HIDDEN,
        INTERMEDIATE,
        &candidate_swiglu,
        args.workers,
    )?;
    let candidate_down_duration_ms = candidate_down_started.elapsed().as_secs_f64() * 1_000.0;
    let candidate_sigmoid = sigmoid_candidate_f32(candidate_scalar_logit)?;
    let candidate_gated_shared = gated_shared_candidate_f32(&candidate_down, candidate_sigmoid)?;

    // Boundary-only references hold the f32 input of the previous candidate
    // stage fixed. The full-chain reference below propagates pure f64 through
    // all stages, making both local and accumulated error visible.
    let candidate_norm_f64 = candidate_norm
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_started = Instant::now();
    let (gate_reference, up_reference, scalar_reference) =
        thread::scope(|scope| -> Result<(Vec<f64>, Vec<f64>, f64), String> {
            let gate = scope.spawn(|| {
                source_f64_matvec(&component.shared_gate, &candidate_norm_f64, "shared gate")
            });
            let up = scope.spawn(|| {
                source_f64_matvec(&component.shared_up, &candidate_norm_f64, "shared up")
            });
            let scalar = source_f64_matvec(
                &component.shared_scalar_gate,
                &candidate_norm_f64,
                "shared scalar gate",
            )?;
            let [scalar] = scalar.as_slice() else {
                return Err("shared scalar f64 reference did not produce one logit".into());
            };
            Ok((
                gate.join()
                    .map_err(|_| "shared gate reference worker panicked")??,
                up.join()
                    .map_err(|_| "shared up reference worker panicked")??,
                *scalar,
            ))
        })?;
    let reference_gate_up_scalar_duration_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
    let (gate_max_abs, gate_max_relative) =
        max_f64_error(&candidate_gate, &gate_reference, "shared gate projection")?;
    let (up_max_abs, up_max_relative) =
        max_f64_error(&candidate_up, &up_reference, "shared up projection")?;
    if gate_max_abs > PROJECTION_F32_F64_TOLERANCE || up_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared gate/up projection parity failed: gate={gate_max_abs}, up={up_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let swiglu_reference = swiglu_reference_f64(&gate_reference, &up_reference)?;
    let (swiglu_max_abs, swiglu_max_relative) =
        max_f64_error(&candidate_swiglu, &swiglu_reference, "shared SiLU(gate)*up")?;
    if swiglu_max_abs > SWIGLU_F32_F64_TOLERANCE {
        return Err(format!(
            "shared SwiGLU parity failed: max_abs={swiglu_max_abs}, tolerance={SWIGLU_F32_F64_TOLERANCE}"
        ));
    }
    let candidate_swiglu_f64 = candidate_swiglu
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let down_matrix_reference =
        source_f64_matvec(&component.shared_down, &candidate_swiglu_f64, "shared down")?;
    let (down_max_abs, down_max_relative) = max_f64_error(
        &candidate_down,
        &down_matrix_reference,
        "shared down projection",
    )?;
    if down_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared down projection parity failed: max_abs={down_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let scalar_max_abs = (f64::from(candidate_scalar_logit) - scalar_reference).abs();
    let scalar_max_relative = scalar_max_abs / scalar_reference.abs().max(1.0);
    if scalar_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared scalar gate parity failed: max_abs={scalar_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let sigmoid_reference = sigmoid_reference_f64(scalar_reference)?;
    let sigmoid_max_abs = (f64::from(candidate_sigmoid) - sigmoid_reference).abs();
    let sigmoid_max_relative = sigmoid_max_abs / sigmoid_reference.abs().max(1.0);
    if sigmoid_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared scalar sigmoid parity failed: max_abs={sigmoid_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }

    let reference_gate_full =
        source_f64_matvec(&component.shared_gate, &reference_norm, "full shared gate")?;
    let reference_up_full =
        source_f64_matvec(&component.shared_up, &reference_norm, "full shared up")?;
    let reference_swiglu_full = swiglu_reference_f64(&reference_gate_full, &reference_up_full)?;
    let reference_down_full = source_f64_matvec(
        &component.shared_down,
        &reference_swiglu_full,
        "full shared down",
    )?;
    let reference_scalar_full = source_f64_matvec(
        &component.shared_scalar_gate,
        &reference_norm,
        "full shared scalar gate",
    )?;
    let [reference_scalar_full] = reference_scalar_full.as_slice() else {
        return Err("full shared scalar gate reference did not produce one logit".into());
    };
    let reference_sigmoid_full = sigmoid_reference_f64(*reference_scalar_full)?;
    let reference_gated_shared_full = reference_down_full
        .iter()
        .map(|value| *value * reference_sigmoid_full)
        .collect::<Vec<_>>();
    let (gated_shared_max_abs, gated_shared_max_relative) = max_f64_error(
        &candidate_gated_shared,
        &reference_gated_shared_full,
        "gated shared full chain",
    )?;
    if gated_shared_max_abs > GATED_SHARED_F32_F64_TOLERANCE {
        return Err(format!(
            "gated shared full-chain parity failed: max_abs={gated_shared_max_abs}, tolerance={GATED_SHARED_F32_F64_TOLERANCE}"
        ));
    }
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("shared-expert rejection suite did not fail closed".into());
    }

    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_CPU_ORACLE_READY_METAL_LEASE_REQUIRED",
        "mode": mode_name(args.mode),
        "shared_expert_only": true,
        "complete_artifact_scan_performed": false,
        "opened_exact_postnorm_and_shared_expert_payloads_only": true,
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
            "shared_expert_intermediate": component.config.shared_expert_intermediate,
            "experts": component.config.experts,
            "experts_per_token": component.config.experts_per_token,
            "rms_epsilon_bits": component.config.rms_norm_eps().to_bits(),
            "post_attention_norm": tensor_report(&component.post_norm),
            "shared_gate_proj": tensor_report(&component.shared_gate),
            "shared_up_proj": tensor_report(&component.shared_up),
            "shared_down_proj": tensor_report(&component.shared_down),
            "shared_expert_gate": tensor_report(&component.shared_scalar_gate),
        },
        "postnorm_evidence": {
            "router_receipt_path": router.receipt_path,
            "router_receipt_sha256": router.receipt_sha256,
            "router_fixture_post_attention_residual_sha256": ROUTER_FIXTURE_RESIDUAL_SHA256,
            "postnorm_source": "same sealed layer-0 postnorm/router receipt; no new router selection or routed-expert work occurred",
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
            "shared_gate_up": {
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
                "candidate_parallel_gate_up_scalar_duration_ms": candidate_gate_up_scalar_duration_ms,
                "reference_parallel_gate_up_scalar_duration_ms": reference_gate_up_scalar_duration_ms,
                "timing_is_component_cpu_work_not_tps": true,
            },
            "swiglu": {
                "formula": "silu(gate) * up",
                "candidate_activated_sha256": f32_slice_sha256(&candidate_swiglu),
                "candidate_f32_vs_reference_f64_max_abs": swiglu_max_abs,
                "candidate_f32_vs_reference_f64_max_relative": swiglu_max_relative,
                "tolerance_max_abs": SWIGLU_F32_F64_TOLERANCE,
            },
            "shared_down": {
                "shape": [HIDDEN, INTERMEDIATE],
                "candidate_down_sha256": f32_slice_sha256(&candidate_down),
                "grouped_f32_vs_scalar_f64_same_candidate_activation_max_abs": down_max_abs,
                "grouped_f32_vs_scalar_f64_same_candidate_activation_max_relative": down_max_relative,
                "tolerance_max_abs": PROJECTION_F32_F64_TOLERANCE,
                "candidate_duration_ms": candidate_down_duration_ms,
                "timing_is_component_cpu_work_not_tps": true,
            },
            "shared_scalar_gate": {
                "shape": [1, HIDDEN],
                "candidate_logit": candidate_scalar_logit,
                "candidate_logit_vs_scalar_f64_max_abs": scalar_max_abs,
                "candidate_logit_vs_scalar_f64_max_relative": scalar_max_relative,
                "candidate_sigmoid": candidate_sigmoid,
                "candidate_sigmoid_vs_f64_max_abs": sigmoid_max_abs,
                "candidate_sigmoid_vs_f64_max_relative": sigmoid_max_relative,
                "tolerance_max_abs": PROJECTION_F32_F64_TOLERANCE,
            },
            "gated_shared_output": {
                "interpretation": "shared-only output gated by its source scalar sigmoid; routed sum, MoE combine, and residual are explicitly absent",
                "candidate_gated_shared_sha256": f32_slice_sha256(&candidate_gated_shared),
                "candidate_f32_vs_full_source_f64_chain_max_abs": gated_shared_max_abs,
                "candidate_f32_vs_full_source_f64_chain_max_relative": gated_shared_max_relative,
                "tolerance_max_abs": GATED_SHARED_F32_F64_TOLERANCE,
                "all_2048_values_finite": candidate_gated_shared.iter().all(|value| value.is_finite()),
            },
        },
        "rejection_tests": rejections,
        "metal_intermediate_error_ledger": {
            "performed": false,
            "reason": "No explicit root registry authorization or Rawls Qwen80 quiet lease. The isolated shader is staged but unregistered; no Metal context, compilation, or dispatch occurred.",
            "future_required_intermediates": ["postnorm_hidden[2048]", "shared_gate[512]", "shared_up[512]", "shared_silu_gate_times_up[512]", "shared_down[2048]", "shared_scalar_gate_logit[1]", "shared_sigmoid[1]", "gated_shared[2048]"],
            "future_acceptance": [
                "bind the same sealed manifest/receipt and exact five direct-packed payloads at device invocation",
                "compare every gate/up/activation/down/scalar-logit/sigmoid/gated-shared value to a fresh CPU oracle",
                "do not combine this output with routed experts or residuals inside this component proof",
            ],
        },
        "integration_contract": {
            "rawls_hybrid_scheduler_handoff": [
                "Feed the real layer-0 postnorm hidden [2048] into the shared gate/up/down and shared scalar gate using one retained admitted catalog snapshot.",
                "Compute source SiLU(gate)*up then down, compute sigmoid(shared_expert_gate(postnorm_hidden)), and multiply only the shared output by that scalar.",
                "The scheduler, not this component, owns adding the ten routed deltas, MoE combine, second residual, and all subsequent work.",
            ],
            "claim_boundary": [
                "The current admitted manifest remains LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION. Packed-to-packed component parity cannot alter that status.",
                "This is CPU-only shared-expert component evidence, not Qwen80 generation, HCLI, BASE_TRUE_TPS, TG10/TG3, capability, Agent OS, or tournament evidence.",
            ],
        },
    }))
}

/// Rebuild a fresh direct-packed CPU oracle inside a future leased device
/// invocation.  The sealed CPU baseline is evidence of the previous control;
/// it is never used as a substitute for same-capture vector parity.
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
    let (gate, up, scalar_logit) =
        thread::scope(|scope| -> Result<(Vec<f32>, Vec<f32>, f32), String> {
            let gate = scope.spawn(|| {
                matvec_candidate_parallel_f32(
                    &component.shared_gate,
                    INTERMEDIATE,
                    HIDDEN,
                    &normalized,
                    gate_workers,
                )
            });
            let up = scope.spawn(|| {
                matvec_candidate_parallel_f32(
                    &component.shared_up,
                    INTERMEDIATE,
                    HIDDEN,
                    &normalized,
                    up_workers,
                )
            });
            let scalar = matvec_candidate_parallel_f32(
                &component.shared_scalar_gate,
                1,
                HIDDEN,
                &normalized,
                1,
            )?;
            let gate = gate
                .join()
                .map_err(|_| "shared gate candidate worker panicked")??;
            let up = up
                .join()
                .map_err(|_| "shared up candidate worker panicked")??;
            let [scalar] = scalar.as_slice() else {
                return Err("shared scalar gate candidate did not produce one logit".into());
            };
            Ok((gate, up, *scalar))
        })?;
    let activated = swiglu_candidate_f32(&gate, &up)?;
    let down = matvec_candidate_parallel_f32(
        &component.shared_down,
        HIDDEN,
        INTERMEDIATE,
        &activated,
        args.workers,
    )?;
    let sigmoid = sigmoid_candidate_f32(scalar_logit)?;
    let gated_shared = gated_shared_candidate_f32(&down, sigmoid)?;

    let normalized_f64 = normalized
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let (gate_reference, up_reference, scalar_reference) =
        thread::scope(|scope| -> Result<(Vec<f64>, Vec<f64>, f64), String> {
            let gate = scope.spawn(|| {
                source_f64_matvec(&component.shared_gate, &normalized_f64, "shared gate")
            });
            let up = scope
                .spawn(|| source_f64_matvec(&component.shared_up, &normalized_f64, "shared up"));
            let scalar = source_f64_matvec(
                &component.shared_scalar_gate,
                &normalized_f64,
                "shared scalar gate",
            )?;
            let [scalar] = scalar.as_slice() else {
                return Err("shared scalar f64 reference did not produce one logit".into());
            };
            Ok((
                gate.join()
                    .map_err(|_| "shared gate reference worker panicked")??,
                up.join()
                    .map_err(|_| "shared up reference worker panicked")??,
                *scalar,
            ))
        })?;
    let (gate_max_abs, gate_max_relative) =
        max_f64_error(&gate, &gate_reference, "shared gate projection")?;
    let (up_max_abs, up_max_relative) = max_f64_error(&up, &up_reference, "shared up projection")?;
    if gate_max_abs > PROJECTION_F32_F64_TOLERANCE || up_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared gate/up projection parity failed: gate={gate_max_abs}, up={up_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let swiglu_reference = swiglu_reference_f64(&gate_reference, &up_reference)?;
    let (swiglu_max_abs, swiglu_max_relative) =
        max_f64_error(&activated, &swiglu_reference, "shared SiLU(gate)*up")?;
    if swiglu_max_abs > SWIGLU_F32_F64_TOLERANCE {
        return Err(format!(
            "shared SwiGLU parity failed: max_abs={swiglu_max_abs}, tolerance={SWIGLU_F32_F64_TOLERANCE}"
        ));
    }
    let activated_f64 = activated
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let down_reference = source_f64_matvec(&component.shared_down, &activated_f64, "shared down")?;
    let (down_max_abs, down_max_relative) =
        max_f64_error(&down, &down_reference, "shared down projection")?;
    if down_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared down projection parity failed: max_abs={down_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let scalar_max_abs = (f64::from(scalar_logit) - scalar_reference).abs();
    let scalar_max_relative = scalar_max_abs / scalar_reference.abs().max(1.0);
    if scalar_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared scalar gate parity failed: max_abs={scalar_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let sigmoid_reference = sigmoid_reference_f64(scalar_reference)?;
    let sigmoid_max_abs = (f64::from(sigmoid) - sigmoid_reference).abs();
    let sigmoid_max_relative = sigmoid_max_abs / sigmoid_reference.abs().max(1.0);
    if sigmoid_max_abs > PROJECTION_F32_F64_TOLERANCE {
        return Err(format!(
            "shared scalar sigmoid parity failed: max_abs={sigmoid_max_abs}, tolerance={PROJECTION_F32_F64_TOLERANCE}"
        ));
    }
    let full_gate = source_f64_matvec(&component.shared_gate, &reference_norm, "full shared gate")?;
    let full_up = source_f64_matvec(&component.shared_up, &reference_norm, "full shared up")?;
    let full_activated = swiglu_reference_f64(&full_gate, &full_up)?;
    let full_down = source_f64_matvec(&component.shared_down, &full_activated, "full shared down")?;
    let full_scalar = source_f64_matvec(
        &component.shared_scalar_gate,
        &reference_norm,
        "full shared scalar gate",
    )?;
    let [full_scalar] = full_scalar.as_slice() else {
        return Err("full shared scalar gate reference did not produce one logit".into());
    };
    let full_sigmoid = sigmoid_reference_f64(*full_scalar)?;
    let full_gated_shared = full_down
        .iter()
        .map(|value| *value * full_sigmoid)
        .collect::<Vec<_>>();
    let (gated_shared_max_abs, gated_shared_max_relative) =
        max_f64_error(&gated_shared, &full_gated_shared, "gated shared full chain")?;
    if gated_shared_max_abs > GATED_SHARED_F32_F64_TOLERANCE {
        return Err(format!(
            "gated shared full-chain parity failed: max_abs={gated_shared_max_abs}, tolerance={GATED_SHARED_F32_F64_TOLERANCE}"
        ));
    }
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("shared-expert rejection suite did not fail closed".into());
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
            scalar_logit,
            sigmoid,
            gated_shared,
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
            scalar_max_abs,
            scalar_max_relative,
            sigmoid_max_abs,
            sigmoid_max_relative,
            gated_shared_max_abs,
            gated_shared_max_relative,
        },
    ))
}

fn run_metal_stage(
    component: &BoundComponent,
    oracle: &CpuOracle,
) -> Result<DeviceParityLedger, String> {
    ensure_non_timed_metal_environment()?;
    if !matches!(
        component.config.layer_kind(LAYER),
        Ok(Qwen80LayerKind::LinearAttention)
    ) || component.config.hidden != HIDDEN
        || component.config.shared_expert_intermediate != INTERMEDIATE
        || component.config.experts != EXPERTS
        || component.config.experts_per_token != TOP_K
    {
        return Err(
            "source config drifted from strict Qwen80 layer-0 shared-expert geometry".into(),
        );
    }
    let (gate_signs, gate_scales) = compact_sign_and_scale_sections(&component.shared_gate)?;
    let (up_signs, up_scales) = compact_sign_and_scale_sections(&component.shared_up)?;
    let (down_signs, down_scales) = compact_sign_and_scale_sections(&component.shared_down)?;
    let (scalar_signs, scalar_scales) =
        compact_sign_and_scale_sections(&component.shared_scalar_gate)?;
    if gate_signs.len() != (INTERMEDIATE * HIDDEN) / 8
        || up_signs.len() != (INTERMEDIATE * HIDDEN) / 8
        || down_signs.len() != (HIDDEN * INTERMEDIATE) / 8
        || scalar_signs.len() != HIDDEN / 8
        || gate_scales.len() != (INTERMEDIATE * HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
        || up_scales.len() != (INTERMEDIATE * HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
        || down_scales.len() != (HIDDEN * INTERMEDIATE / GROUP_SIZE) * std::mem::size_of::<u16>()
        || scalar_scales.len() != (HIDDEN / GROUP_SIZE) * std::mem::size_of::<u16>()
    {
        return Err(
            "direct-packed shared-expert sections do not match the fixed shader ABI".into(),
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
    let down_sign_buffer = context
        .new_buffer_with_bytes_checked(down_signs)
        .map_err(|error| error.to_string())?;
    let down_scale_buffer = context
        .new_buffer_with_bytes_checked(down_scales)
        .map_err(|error| error.to_string())?;
    let scalar_sign_buffer = context
        .new_buffer_with_bytes_checked(scalar_signs)
        .map_err(|error| error.to_string())?;
    let scalar_scale_buffer = context
        .new_buffer_with_bytes_checked(scalar_scales)
        .map_err(|error| error.to_string())?;
    let gate_output = context
        .new_buffer_checked(bytes_for::<f32>(INTERMEDIATE, "shared gate output")?)
        .map_err(|error| error.to_string())?;
    let up_output = context
        .new_buffer_checked(bytes_for::<f32>(INTERMEDIATE, "shared up output")?)
        .map_err(|error| error.to_string())?;
    let activated = context
        .new_buffer_checked(bytes_for::<f32>(INTERMEDIATE, "shared SwiGLU activation")?)
        .map_err(|error| error.to_string())?;
    let down_output = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "shared down output")?)
        .map_err(|error| error.to_string())?;
    let scalar_logit = context
        .new_buffer_checked(bytes_for::<f32>(1, "shared scalar logit")?)
        .map_err(|error| error.to_string())?;
    let gated_shared = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "gated shared output")?)
        .map_err(|error| error.to_string())?;
    MetalContext::write_buffer_bytes(&hidden, bytemuck::cast_slice(&oracle.normalized));

    let mut command = TokenCommandBuffer::new(&context);
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_gate_up",
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
            "qwen80_shared_expert_wave_swiglu",
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
            "qwen80_shared_expert_wave_down",
            (256, HIDDEN as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&down_sign_buffer), 0);
                encoder.set_buffer(1, Some(&down_scale_buffer), 0);
                encoder.set_buffer(2, Some(&activated), 0);
                encoder.set_buffer(3, Some(&down_output), 0);
                encoder.stage_set_u32(4, HIDDEN as u32);
                encoder.stage_set_u32(5, INTERMEDIATE as u32);
                encoder.stage_set_u32(6, GROUP_SIZE as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_scalar_gate",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&scalar_sign_buffer), 0);
                encoder.set_buffer(1, Some(&scalar_scale_buffer), 0);
                encoder.set_buffer(2, Some(&hidden), 0);
                encoder.set_buffer(3, Some(&scalar_logit), 0);
                encoder.stage_set_u32(4, HIDDEN as u32);
                encoder.stage_set_u32(5, GROUP_SIZE as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
            (HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&down_output), 0);
                encoder.set_buffer(1, Some(&scalar_logit), 0);
                encoder.set_buffer(2, Some(&gated_shared), 0);
                encoder.stage_set_u32(3, HIDDEN as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    let dispatch_count = command.dispatch_count();
    if dispatch_count != 5 {
        return Err(format!(
            "strict shared-expert command buffer encoded {dispatch_count} dispatches, expected exactly 5"
        ));
    }
    command
        .commit_and_wait()
        .map_err(|error| error.to_string())?;

    let gate = snapshot_f32(&gate_output, INTERMEDIATE, "shared gate output")?;
    let up = snapshot_f32(&up_output, INTERMEDIATE, "shared up output")?;
    let activated_values = snapshot_f32(&activated, INTERMEDIATE, "shared SwiGLU activation")?;
    let down = snapshot_f32(&down_output, HIDDEN, "shared down output")?;
    let scalar = snapshot_f32(&scalar_logit, 1, "shared scalar logit")?;
    let [scalar_logit] = scalar.as_slice() else {
        return Err("strict-Metal shared scalar gate did not produce one logit".into());
    };
    let gated = snapshot_f32(&gated_shared, HIDDEN, "gated shared output")?;
    let gate_max_abs = max_abs_error_f32(&oracle.gate, &gate, "strict-Metal shared gate")?;
    let up_max_abs = max_abs_error_f32(&oracle.up, &up, "strict-Metal shared up")?;
    let activated_max_abs = max_abs_error_f32(
        &oracle.activated,
        &activated_values,
        "strict-Metal shared SiLU(gate)*up",
    )?;
    let down_max_abs = max_abs_error_f32(&oracle.down, &down, "strict-Metal shared down")?;
    let scalar_max_abs = max_abs_error_f32(
        std::slice::from_ref(&oracle.scalar_logit),
        std::slice::from_ref(scalar_logit),
        "strict-Metal shared scalar gate",
    )?;
    let gated_shared_max_abs = max_abs_error_f32(
        &oracle.gated_shared,
        &gated,
        "strict-Metal gated shared output",
    )?;
    for (label, error, tolerance) in [
        ("gate", gate_max_abs, DEVICE_GATE_UP_MAX_ABS_TOLERANCE),
        ("up", up_max_abs, DEVICE_GATE_UP_MAX_ABS_TOLERANCE),
        ("SwiGLU", activated_max_abs, DEVICE_SWIGLU_MAX_ABS_TOLERANCE),
        ("down", down_max_abs, DEVICE_DOWN_MAX_ABS_TOLERANCE),
        (
            "scalar gate",
            scalar_max_abs,
            DEVICE_SCALAR_GATE_MAX_ABS_TOLERANCE,
        ),
        (
            "gated shared",
            gated_shared_max_abs,
            DEVICE_GATED_SHARED_MAX_ABS_TOLERANCE,
        ),
    ] {
        if error > tolerance {
            return Err(format!(
                "strict-Metal shared-expert {label} parity {error} exceeds {tolerance}"
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
        scalar_logit: *scalar_logit,
        gated_shared: gated,
        gate_max_abs,
        up_max_abs,
        activated_max_abs,
        down_max_abs,
        scalar_max_abs,
        gated_shared_max_abs,
    })
}

fn run_metal_component(args: &Args) -> Result<Value, String> {
    if args.mode != Mode::Metal {
        return Err("strict shared-expert Metal component requires --mode metal".into());
    }
    let (component, oracle) = build_device_cpu_oracle(args)?;
    let baseline_path = args
        .cpu_baseline_receipt
        .as_deref()
        .ok_or("strict shared-expert Metal component requires --cpu-baseline-receipt")?;
    let baseline = validate_cpu_baseline(baseline_path, &component)?;
    let lease_path = args
        .quiet_metal_lease
        .as_deref()
        .ok_or("strict shared-expert Metal component requires --lease-receipt")?;
    let lease = validate_quiet_metal_lease(lease_path, &component, &baseline)?;
    let device = run_metal_stage(&component, &oracle)?;
    let rejections = rejection_report();
    if !all_rejections_passed(&rejections) {
        return Err("shared-expert rejection suite did not fail closed".into());
    }
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_STRICT_MATH_METAL_COMPONENT_NOT_ROUTED_MOE_OR_LAYER",
        "mode": "metal",
        "shared_expert_only": true,
        "routed_expert_sum_performed": false,
        "moe_combine_performed": false,
        "second_residual_performed": false,
        "complete_artifact_scan_performed": false,
        "opened_exact_postnorm_and_shared_expert_payloads_only": true,
        "raw_bf16_or_safetensors_opened": false,
        "metal_device_or_dispatch_performed": true,
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
            "shared_expert_intermediate": INTERMEDIATE,
            "experts": EXPERTS,
            "experts_per_token": TOP_K,
            "post_attention_norm": tensor_report(&component.post_norm),
            "shared_gate_proj": tensor_report(&component.shared_gate),
            "shared_up_proj": tensor_report(&component.shared_up),
            "shared_down_proj": tensor_report(&component.shared_down),
            "shared_expert_gate": tensor_report(&component.shared_scalar_gate),
        },
        "cpu_baseline_binding": baseline_report(&baseline),
        "same_capture_cpu_oracle": {
            "post_attention_residual": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.residual)},
            "postnorm_hidden": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.normalized)},
            "shared_gate": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&oracle.gate)},
            "shared_up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&oracle.up)},
            "shared_silu_gate_times_up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&oracle.activated)},
            "shared_down": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.down)},
            "shared_scalar_gate_logit": oracle.scalar_logit,
            "shared_sigmoid": oracle.sigmoid,
            "gated_shared": {"elements": HIDDEN, "sha256": f32_slice_sha256(&oracle.gated_shared)},
            "f32_vs_source_f64_max_abs": {
                "post_attention_rmsnorm": oracle.norm_max_abs,
                "gate": oracle.gate_max_abs,
                "up": oracle.up_max_abs,
                "silu_gate_times_up": oracle.swiglu_max_abs,
                "down": oracle.down_max_abs,
                "scalar_gate": oracle.scalar_max_abs,
                "sigmoid": oracle.sigmoid_max_abs,
                "gated_shared_full_chain": oracle.gated_shared_max_abs,
            },
            "f32_vs_source_f64_max_relative": {
                "post_attention_rmsnorm": oracle.norm_max_relative,
                "gate": oracle.gate_max_relative,
                "up": oracle.up_max_relative,
                "silu_gate_times_up": oracle.swiglu_max_relative,
                "down": oracle.down_max_relative,
                "scalar_gate": oracle.scalar_max_relative,
                "sigmoid": oracle.sigmoid_max_relative,
                "gated_shared_full_chain": oracle.gated_shared_max_relative,
            },
            "postnorm_direct_decode_max_abs": oracle.norm_direct_decode_max_abs,
        },
        "metal_execution_policy": {
            "strict_math_required": true,
            "timing_or_benchmarking_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "lease_binding": lease_report(&lease),
        },
        "metal_intermediate_error_ledger": {
            "performed": true,
            "device": device.device_name,
            "command_buffers": 1,
            "compute_dispatches": device.dispatch_count,
            "kernel_sequence": [
                "qwen80_shared_expert_wave_gate_up",
                "qwen80_shared_expert_wave_swiglu",
                "qwen80_shared_expert_wave_down",
                "qwen80_shared_expert_wave_scalar_gate",
                "qwen80_shared_expert_wave_apply_sigmoid_gate",
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
                "scalar_gate_max_abs": device.scalar_max_abs,
                "scalar_gate_tolerance": DEVICE_SCALAR_GATE_MAX_ABS_TOLERANCE,
                "gated_shared_max_abs": device.gated_shared_max_abs,
                "gated_shared_tolerance": DEVICE_GATED_SHARED_MAX_ABS_TOLERANCE,
            },
            "device_intermediates": {
                "shared_gate": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&device.gate)},
                "shared_up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&device.up)},
                "shared_silu_gate_times_up": {"elements": INTERMEDIATE, "sha256": f32_slice_sha256(&device.activated)},
                "shared_down": {"elements": HIDDEN, "sha256": f32_slice_sha256(&device.down)},
                "shared_scalar_gate_logit": device.scalar_logit,
                "gated_shared": {"elements": HIDDEN, "sha256": f32_slice_sha256(&device.gated_shared)},
            },
        },
        "rejection_tests": rejections,
        "integration_contract": {
            "scheduler_handoff": [
                "This result is only the shared-expert body gated by its source scalar sigmoid.",
                "The ten routed waves, their aggregation, MoE combination, second residual, and all later layer/token work remain unexecuted.",
            ],
            "claim_boundary": [
                "Synthetic source-shaped postnorm input component parity only; not a complete Qwen80 layer, token, decoder, generation, HCLI, TPS, TG, capability, Agent OS, or tournament receipt.",
            ],
        },
    }))
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_shared_expert_wave: {error}");
            std::process::exit(2);
        }
    };
    if let Err(error) = begin_capture(&args) {
        eprintln!("ascension_qwen80_direct_packed_shared_expert_wave: {error}");
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
                eprintln!("ascension_qwen80_direct_packed_shared_expert_wave: result print failed: {error}");
                std::process::exit(2);
            }
        },
        Ok((_result, Some(error))) => {
            eprintln!("ascension_qwen80_direct_packed_shared_expert_wave: {error}");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_shared_expert_wave: capture finalization failed: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn source_swiglu_and_scalar_sigmoid_are_ordered_and_bounded() {
        let mut gate = vec![0.0f32; INTERMEDIATE];
        let mut up = vec![0.0f32; INTERMEDIATE];
        gate[0] = 2.0;
        up[0] = -3.0;
        let output = swiglu_candidate_f32(&gate, &up).unwrap();
        assert!((output[0] - (2.0 / (1.0 + (-2.0f32).exp()) * -3.0)).abs() < 1.0e-6);
        let sigmoid = sigmoid_candidate_f32(0.0).unwrap();
        assert!((sigmoid - 0.5).abs() < f32::EPSILON);
        assert!(sigmoid_candidate_f32(f32::NAN).is_err());
    }

    #[test]
    fn gated_shared_output_rejects_invalid_gate_and_preserves_length() {
        let output = gated_shared_candidate_f32(&vec![2.0; HIDDEN], 0.25).unwrap();
        assert_eq!(output.len(), HIDDEN);
        assert!(output
            .iter()
            .all(|value| (*value - 0.5).abs() < f32::EPSILON));
        assert!(gated_shared_candidate_f32(&vec![0.0; HIDDEN], 1.1).is_err());
    }

    #[test]
    fn component_constants_preserve_qwen80_shared_geometry() {
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
    fn shared_metal_matrix_kernels_use_explicit_y_row_abi() {
        let shader = include_str!("../shaders/qwen80_shared_expert_wave.metal");
        for kernel in [
            "qwen80_shared_expert_wave_gate_up",
            "qwen80_shared_expert_wave_down",
        ] {
            let start = shader
                .find(kernel)
                .expect("shared matrix kernel must exist");
            let body = &shader[start..];
            assert!(body.contains("uint3 tid [[thread_position_in_threadgroup]]"));
            assert!(body.contains("uint3 group_position [[threadgroup_position_in_grid]]"));
            assert!(body.contains("const uint lane = tid.x;"));
            assert!(body.contains("const uint row = group_position.y;"));
            assert!(!body.contains("uint row [[threadgroup_position_in_grid]]"));
        }
        assert!(shader.contains("if (lane == 0u)"));
    }

    #[test]
    fn shared_metal_dispatch_contract_has_five_non_timed_component_stages() {
        assert_eq!(
            QUIET_METAL_COMPONENT,
            "qwen80_direct_packed_shared_expert_wave"
        );
        assert!(DEVICE_GATE_UP_MAX_ABS_TOLERANCE > 0.0);
        assert!(DEVICE_GATED_SHARED_MAX_ABS_TOLERANCE > 0.0);
        let shader = include_str!("../shaders/qwen80_shared_expert_wave.metal");
        for kernel in [
            "qwen80_shared_expert_wave_gate_up",
            "qwen80_shared_expert_wave_swiglu",
            "qwen80_shared_expert_wave_down",
            "qwen80_shared_expert_wave_scalar_gate",
            "qwen80_shared_expert_wave_apply_sigmoid_gate",
        ] {
            assert!(
                shader.contains(kernel),
                "missing shared component kernel {kernel}"
            );
        }
    }
}
