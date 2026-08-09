//! Current-admitted, CPU-only Qwen3-Coder-Next terminal component candidate.
//!
//! This is intentionally a narrow terminal seam, not a Qwen80 runtime.  It
//! opens only the current admitted `model.norm.weight` and `lm_head.weight`
//! `HQ30G1B1` payloads, plus config/tokenizer sidecars required to bind the
//! exact norm and reserved-tail semantics.  It never opens source BF16/
//! safetensors, a Metal device, the Qwen80 runtime/scheduler, HCLI, watcher,
//! packer, service, or gatekeeper.
//!
//! The candidate order is fixed:
//!
//! `final RMSNorm -> all 151936 direct-packed lm_head rows -> mask 151669..151935 -> sample`
//!
//! Its CPU parity is against an independent f64 direct-packed reference over
//! the same admitted payload.  This is a component receipt only: no complete
//! token, generation, HCLI, TPS, TG, capability, or tournament claim follows.

use half::f16;
use hawking_core::model::qwen_complete_binary::{
    complete_binary_matvec_f64, decode_complete_binary_f32, parse_complete_binary_header,
    CompleteBinaryHeader,
};
use hawking_core::tokenizer::Tokenizer;
use serde::Serialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;
use std::time::Instant;

const SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_terminal_head_cpu.v1";
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
const SOURCE_TOKENIZER_SHA256: &str =
    "19564a48c4f71a2a1b937cce34c737a1e662b171c5f5d7edf641a15cd896f07d";
const FINAL_NORM_NAME: &str = "model.norm.weight";
const LM_HEAD_NAME: &str = "lm_head.weight";
const FINAL_NORM_ARTIFACT_SHA256: &str =
    "6306499804d27e48f0a041e94d366feae5cbf8436fac15815a559a15717ef36e";
const LM_HEAD_ARTIFACT_SHA256: &str =
    "549c448be683ed00ec792329c5167f3f0cacfcb3af339a1fb064ed0a004d9998";
const SOURCE_HEAD_SHARD: &str = "model-00040-of-00040.safetensors";
const SOURCE_HEAD_SHARD_SHA256: &str =
    "9606cbc3a087efaba8e17cb97b9d91cd8d25e6cba6958079d58c19871d272053";
const HIDDEN: usize = 2_048;
const LM_HEAD_VOCAB: usize = 151_936;
const TOKENIZER_VOCAB: usize = 151_669;
const RESERVED_TAIL_ROWS: usize = LM_HEAD_VOCAB - TOKENIZER_VOCAB;
const GROUP_SIZE: usize = 128;
const RMS_EPSILON: f32 = 1.0e-6;
const RMS_EPSILON_JSON: f64 = 1.0e-6;
const FINAL_NORM_F32_F64_TOLERANCE: f64 = 1.0e-5;
const LM_HEAD_F32_F64_TOLERANCE: f64 = 5.0e-5;
const DEFAULT_MAX_CPU_WORKERS: usize = 4;

#[derive(Clone, Debug)]
struct Args {
    manifest: PathBuf,
    admission_current: PathBuf,
    out: PathBuf,
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
    payload: Arc<[u8]>,
}

#[derive(Clone, Debug)]
struct BoundTerminalComponent {
    manifest_path: PathBuf,
    admission_current_path: PathBuf,
    manifest_document_sha256: String,
    manifest_seal_sha256: String,
    admission_pointer_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_model_dir: PathBuf,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    rms_epsilon: f32,
    final_norm: BoundTensor,
    lm_head: BoundTensor,
}

#[derive(Serialize)]
struct HeaderReport {
    magic: &'static str,
    version: u32,
    group_size: usize,
    shape: Vec<usize>,
    elements: usize,
    groups: usize,
    scale_offset: usize,
    sign_offset: usize,
    payload_bytes: usize,
}

#[derive(Serialize)]
struct TensorReport {
    name: &'static str,
    artifact_path: String,
    artifact_sha256: String,
    source_shard: String,
    source_shard_sha256: String,
    header: HeaderReport,
}

#[derive(Serialize)]
struct ArtifactBindingReport {
    manifest_path: String,
    manifest_document_sha256: String,
    manifest_seal_sha256: String,
    admission_current_path: String,
    admission_pointer_seal_sha256: String,
    admission_receipt_seal_sha256: String,
    source_repository: &'static str,
    source_revision: &'static str,
    source_body_audit_seal_sha256: &'static str,
    source_revalidation_seal_sha256: &'static str,
    source_model_dir: String,
    source_config_sha256: String,
    source_tokenizer_sha256: String,
    hidden_size: usize,
    lm_head_vocab_size: usize,
    tokenizer_vocab_size: usize,
    reserved_lm_head_tail_rows: usize,
    rms_epsilon_bits: u32,
    final_norm: TensorReport,
    lm_head: TensorReport,
}

#[derive(Serialize)]
struct FinalNormParityReport {
    hidden_input_kind: &'static str,
    hidden_input_sha256: String,
    candidate_f32_vs_reference_f64_max_abs: f64,
    direct_packed_norm_values_vs_library_decoder_max_abs: f64,
    tolerance_max_abs: f64,
    all_outputs_finite: bool,
}

#[derive(Serialize)]
struct LmHeadParityReport {
    rows_evaluated: usize,
    rows_per_direct_group: usize,
    candidate_cpu_workers: usize,
    candidate_grouped_f32_vs_reference_scalar_f64_max_abs: f64,
    candidate_grouped_f32_vs_reference_scalar_f64_max_relative: f64,
    tolerance_max_abs: f64,
    all_candidate_logits_finite_before_mask: bool,
    candidate_duration_ms: f64,
    reference_duration_ms: f64,
    timing_is_component_cpu_work_not_tps: bool,
}

#[derive(Serialize)]
struct TailMaskReport {
    first_reserved_id: usize,
    last_reserved_id: usize,
    reserved_rows_masked: usize,
    raw_direct_packed_argmax_id: usize,
    raw_direct_packed_argmax_logit: f32,
    masked_direct_packed_argmax_id: usize,
    masked_direct_packed_argmax_logit: f32,
    sampled_token_id: usize,
    every_reserved_logit_negative_infinity: bool,
    sampled_token_is_tokenizer_addressable: bool,
}

#[derive(Serialize)]
struct RejectionReport {
    malformed_direct_header_rejected: bool,
    wrong_lm_head_shape_rejected: bool,
    wrong_direct_group_size_rejected: bool,
    wrong_tail_partition_rejected: bool,
    tail_mask_wrong_cutoff_rejected: bool,
    unmasked_sampler_api_unavailable: bool,
    nonfinite_hidden_rejected: bool,
}

#[derive(Serialize)]
struct IntegrationContract {
    rawls_hybrid_scheduler_handoff: Vec<&'static str>,
    future_metal_lease_requirements: Vec<&'static str>,
    claim_boundary: Vec<&'static str>,
}

#[derive(Serialize)]
struct Report {
    schema: &'static str,
    status: &'static str,
    component_only: bool,
    complete_artifact_scan_performed: bool,
    opened_exact_terminal_payloads_only: bool,
    raw_bf16_or_safetensors_opened: bool,
    metal_device_or_dispatch_performed: bool,
    artifact_binding: ArtifactBindingReport,
    final_norm_cpu_parity: FinalNormParityReport,
    lm_head_cpu_parity: LmHeadParityReport,
    reserved_tail_mask_and_sampler: TailMaskReport,
    rejection_tests: RejectionReport,
    integration_contract: IntegrationContract,
    unsealed_preimage_sha256: String,
}

#[derive(Clone, Debug)]
struct MaskedLogits {
    values: Vec<f32>,
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

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
                .ok_or_else(|| format!("{label} shape[{index}] is not a positive platform usize"))
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
    let metadata = fs::symlink_metadata(&canonical).map_err(|error| {
        format!(
            "{label} metadata failed at {}: {error}",
            canonical.display()
        )
    })?;
    if !metadata.is_dir() {
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

fn header_report(header: &CompleteBinaryHeader) -> HeaderReport {
    HeaderReport {
        magic: "HQ30G1B1",
        version: header.version,
        group_size: header.group_size,
        shape: header.shape.clone(),
        elements: header.elements,
        groups: header.groups,
        scale_offset: header.scale_offset,
        sign_offset: header.sign_offset,
        payload_bytes: header.payload_bytes,
    }
}

fn tensor_report(tensor: &BoundTensor) -> TensorReport {
    TensorReport {
        name: tensor.name,
        artifact_path: tensor.path.display().to_string(),
        artifact_sha256: tensor.payload_sha256.clone(),
        source_shard: tensor.source_shard.clone(),
        source_shard_sha256: tensor.source_shard_sha256.clone(),
        header: header_report(&tensor.header),
    }
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut manifest = None;
    let mut admission_current = None;
    let mut out = None;
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
            "--out" => {
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out repeated".into());
                }
            }
            "--workers" => {
                if workers.replace(value.parse::<usize>()?).is_some() {
                    return Err("--workers repeated".into());
                }
            }
            _ => return Err("usage: ascension_qwen80_direct_packed_terminal_head_cpu --manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH --out ABSOLUTE_PATH [--workers 1..N]".into()),
        }
    }
    let manifest = manifest.ok_or("missing --manifest")?;
    let admission_current = admission_current.ok_or("missing --admission-current")?;
    let out = out.ok_or("missing --out")?;
    if !manifest.is_absolute() || !admission_current.is_absolute() || !out.is_absolute() {
        return Err("--manifest, --admission-current, and --out must be absolute".into());
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
        out,
        workers,
    })
}

fn exact_path_from_json(value: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(value);
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    path.canonicalize().map_err(|error| {
        format!(
            "{label} canonicalization failed at {}: {error}",
            path.display()
        )
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
        return Err("current Qwen80 admission pointer identity/status/seal shape drifted".into());
    }
    let model = object_field(&pointer, "model", "current Qwen80 admission pointer")?;
    if string_field(model, "id", "current Qwen80 admission pointer model")? != MODEL_ID
        || string_field(model, "key", "current Qwen80 admission pointer model")? != MODEL_KEY
        || string_field(
            model,
            "repository",
            "current Qwen80 admission pointer model",
        )? != SOURCE_REPOSITORY
        || string_field(model, "revision", "current Qwen80 admission pointer model")?
            != SOURCE_REVISION
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
            "current Qwen80 complete manifest",
        )?,
        "current Qwen80 complete manifest path",
    )? != manifest_path
        || string_field(
            complete_manifest,
            "schema",
            "current Qwen80 complete manifest",
        )? != MANIFEST_SCHEMA
        || string_field(
            complete_manifest,
            "seal_sha256",
            "current Qwen80 complete manifest",
        )? != CURRENT_MANIFEST_SEAL
        || string_field(
            complete_manifest,
            "document_sha256",
            "current Qwen80 complete manifest",
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
    // The current pointer can rotate independently of the immutable complete
    // manifest. Its observed seal is recorded; the manifest document/seal and
    // admission-receipt seal above remain the component's artifact authority.
    Ok(pointer_seal.to_owned())
}

fn bind_source_sidecars(
    manifest: &Map<String, Value>,
) -> Result<(PathBuf, String, String, f32), String> {
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
    let config = json_object(&config_bytes, "Qwen80 source config")?;
    let architectures = config
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or("Qwen80 source config missing architectures")?;
    if !architectures
        .iter()
        .any(|value| value.as_str() == Some("Qwen3NextForCausalLM"))
        || string_field(&config, "model_type", "Qwen80 source config")? != "qwen3_next"
        || u64_field(&config, "hidden_size", "Qwen80 source config")? != HIDDEN as u64
        || u64_field(&config, "vocab_size", "Qwen80 source config")? != LM_HEAD_VOCAB as u64
        || bool_field(&config, "tie_word_embeddings", "Qwen80 source config")?
    {
        return Err("Qwen80 source config terminal-head facts drifted".into());
    }
    let epsilon = config
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .ok_or("Qwen80 source config missing rms_norm_eps")?;
    if epsilon.to_bits() != RMS_EPSILON_JSON.to_bits() {
        return Err("Qwen80 source config rms_norm_eps drifted from exact 1e-6".into());
    }
    let tokenizer_path =
        canonical_regular_file(&model_dir.join("tokenizer.json"), "Qwen80 source tokenizer")?;
    let tokenizer_bytes = regular_file_bytes(&tokenizer_path, "Qwen80 source tokenizer")?;
    let tokenizer_sha256 = checked_sha256(
        &tokenizer_bytes,
        SOURCE_TOKENIZER_SHA256,
        "Qwen80 source tokenizer",
    )?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|error| format!("Qwen80 source tokenizer load failed: {error}"))?;
    if tokenizer.vocab_size() != TOKENIZER_VOCAB {
        return Err(format!(
            "Qwen80 source tokenizer namespace {} differs from expected {TOKENIZER_VOCAB}",
            tokenizer.vocab_size()
        ));
    }
    Ok((model_dir, config_sha256, tokenizer_sha256, RMS_EPSILON))
}

fn expected_tensor_filename(name: &str) -> String {
    format!("{}.hq30g", sha256_hex(name.as_bytes()))
}

fn bind_terminal_tensor(
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
    if shape != expected_shape
        || u64_field(row, "elements", &label)? != expected_shape.iter().product::<usize>() as u64
        || string_field(row, "source_dtype", &label)? != "BF16"
        || string_field(row, "source_shard", &label)? != SOURCE_HEAD_SHARD
        || string_field(row, "source_shard_sha256", &label)? != SOURCE_HEAD_SHARD_SHA256
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
        return Err(format!("{label} direct binary layout drifted"));
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
        .map_err(|error| format!("{label} direct binary header rejected: {error}"))?;
    if header.version != 1
        || header.group_size != GROUP_SIZE
        || header.shape != expected_shape
        || header.elements != expected_shape.iter().product::<usize>()
        || header.payload_bytes != payload.len()
    {
        return Err(format!("{label} direct binary header geometry drifted"));
    }
    Ok(BoundTensor {
        name,
        path: expected_path,
        payload_sha256,
        source_shard: SOURCE_HEAD_SHARD.to_owned(),
        source_shard_sha256: SOURCE_HEAD_SHARD_SHA256.to_owned(),
        header,
        payload: Arc::from(payload),
    })
}

fn bind_current_admitted_component(args: &Args) -> Result<BoundTerminalComponent, String> {
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
    let (source_model_dir, source_config_sha256, source_tokenizer_sha256, rms_epsilon) =
        bind_source_sidecars(&manifest)?;
    let manifest_root = manifest_path
        .parent()
        .ok_or("Qwen80 complete manifest has no parent directory")?;
    let final_norm = bind_terminal_tensor(
        &manifest,
        manifest_root,
        FINAL_NORM_NAME,
        &[HIDDEN],
        FINAL_NORM_ARTIFACT_SHA256,
    )?;
    let lm_head = bind_terminal_tensor(
        &manifest,
        manifest_root,
        LM_HEAD_NAME,
        &[LM_HEAD_VOCAB, HIDDEN],
        LM_HEAD_ARTIFACT_SHA256,
    )?;
    Ok(BoundTerminalComponent {
        manifest_path,
        admission_current_path,
        manifest_document_sha256,
        manifest_seal_sha256: CURRENT_MANIFEST_SEAL.to_owned(),
        admission_pointer_seal_sha256,
        admission_receipt_seal_sha256: CURRENT_ADMISSION_RECEIPT_SEAL.to_owned(),
        source_model_dir,
        source_config_sha256,
        source_tokenizer_sha256,
        rms_epsilon,
        final_norm,
        lm_head,
    })
}

fn packed_value_f32(
    payload: &[u8],
    header: &CompleteBinaryHeader,
    index: usize,
) -> Result<f32, String> {
    if index >= header.elements {
        return Err("direct packed element index is out of bounds".into());
    }
    let group = index / header.group_size;
    let within_group = index % header.group_size;
    let scale_start = header
        .scale_offset
        .checked_add(group.checked_mul(2).ok_or("scale index overflow")?)
        .ok_or("scale offset overflow")?;
    let scale = f16::from_bits(u16::from_le_bytes([
        *payload
            .get(scale_start)
            .ok_or("scale low byte out of bounds")?,
        *payload
            .get(scale_start + 1)
            .ok_or("scale high byte out of bounds")?,
    ]))
    .to_f32();
    let bytes_per_group = header.group_size / 8;
    let sign_index = header
        .sign_offset
        .checked_add(
            group
                .checked_mul(bytes_per_group)
                .ok_or("sign group offset overflow")?,
        )
        .and_then(|offset| offset.checked_add(within_group / 8))
        .ok_or("sign offset overflow")?;
    let bit = (payload.get(sign_index).ok_or("sign byte out of bounds")? >> (within_group % 8)) & 1;
    Ok(if bit == 1 { scale } else { -scale })
}

fn deterministic_hidden() -> Vec<f32> {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    (0..HIDDEN)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let mantissa = ((state >> 40) & 0x00ff_ffff) as f32 / 16_777_215.0;
            let position_term = ((index * 37 % 97) as f32 - 48.0) / 211.0;
            (mantissa * 2.0 - 1.0) + position_term
        })
        .collect()
}

fn final_rms_norm_candidate_f32(
    norm: &BoundTensor,
    hidden: &[f32],
    epsilon: f32,
) -> Result<Vec<f32>, String> {
    if norm.header.shape != [HIDDEN]
        || norm.header.group_size != GROUP_SIZE
        || hidden.len() != HIDDEN
        || hidden.iter().any(|value| !value.is_finite())
        || !epsilon.is_finite()
        || epsilon <= 0.0
    {
        return Err("candidate final RMSNorm received invalid direct-packed geometry/input".into());
    }
    let mean_square = hidden.iter().map(|value| value * value).sum::<f32>() / HIDDEN as f32;
    let inverse_rms = 1.0 / (mean_square + epsilon).sqrt();
    let output = hidden
        .iter()
        .enumerate()
        .map(|(index, value)| {
            Ok(*value * inverse_rms * packed_value_f32(&norm.payload, &norm.header, index)?)
        })
        .collect::<Result<Vec<_>, String>>()?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("candidate final RMSNorm produced a non-finite output".into());
    }
    Ok(output)
}

fn final_rms_norm_reference_f64(
    norm: &BoundTensor,
    hidden: &[f32],
    epsilon: f32,
) -> Result<Vec<f64>, String> {
    let (header, decoded) = decode_complete_binary_f32(&norm.payload)
        .map_err(|error| format!("library direct-packed final norm decoder failed: {error}"))?;
    if header != norm.header || decoded.len() != HIDDEN || hidden.len() != HIDDEN {
        return Err("library final norm decode disagrees with admitted binding".into());
    }
    let mean_square = hidden
        .iter()
        .map(|value| f64::from(*value) * f64::from(*value))
        .sum::<f64>()
        / HIDDEN as f64;
    let inverse_rms = 1.0 / (mean_square + f64::from(epsilon)).sqrt();
    Ok(hidden
        .iter()
        .zip(decoded)
        .map(|(hidden, weight)| f64::from(*hidden) * inverse_rms * f64::from(weight))
        .collect())
}

fn row_dot_grouped_f32(
    payload: &[u8],
    header: &CompleteBinaryHeader,
    row: usize,
    input: &[f32],
) -> Result<f32, String> {
    if header.shape.len() != 2
        || input.len() != header.shape[1]
        || header.shape[1] % header.group_size != 0
        || row >= header.shape[0]
        || input.iter().any(|value| !value.is_finite())
    {
        return Err("grouped direct-packed matvec received invalid matrix/input geometry".into());
    }
    let columns = header.shape[1];
    let groups_per_row = columns / header.group_size;
    let row_base = row.checked_mul(columns).ok_or("row base offset overflow")?;
    let mut total = 0.0f32;
    for group_within_row in 0..groups_per_row {
        let element_base = row_base
            .checked_add(
                group_within_row
                    .checked_mul(header.group_size)
                    .ok_or("group element offset overflow")?,
            )
            .ok_or("group element base overflow")?;
        let group = element_base / header.group_size;
        let scale_offset = header
            .scale_offset
            .checked_add(group.checked_mul(2).ok_or("group scale offset overflow")?)
            .ok_or("group scale address overflow")?;
        let scale = f16::from_bits(u16::from_le_bytes([
            *payload
                .get(scale_offset)
                .ok_or("group scale low byte out of bounds")?,
            *payload
                .get(scale_offset + 1)
                .ok_or("group scale high byte out of bounds")?,
        ]))
        .to_f32();
        let sign_base = header
            .sign_offset
            .checked_add(
                group
                    .checked_mul(header.group_size / 8)
                    .ok_or("group sign offset overflow")?,
            )
            .ok_or("group sign address overflow")?;
        let mut signed_input_sum = 0.0f32;
        for within_group in 0..header.group_size {
            let sign_byte = *payload
                .get(sign_base + within_group / 8)
                .ok_or("group sign byte out of bounds")?;
            let signed_input = if ((sign_byte >> (within_group % 8)) & 1) == 1 {
                input[group_within_row * header.group_size + within_group]
            } else {
                -input[group_within_row * header.group_size + within_group]
            };
            signed_input_sum += signed_input;
        }
        total += scale * signed_input_sum;
    }
    if !total.is_finite() {
        return Err("grouped direct-packed matvec produced a non-finite output".into());
    }
    Ok(total)
}

fn lm_head_candidate_parallel_f32(
    head: &BoundTensor,
    input: &[f32],
    workers: usize,
) -> Result<Vec<f32>, String> {
    if head.header.shape != [LM_HEAD_VOCAB, HIDDEN]
        || head.header.group_size != GROUP_SIZE
        || input.len() != HIDDEN
        || workers == 0
    {
        return Err("candidate lm_head binding/worker geometry is invalid".into());
    }
    let rows = head.header.shape[0];
    let workers = workers.min(rows).max(1);
    let chunk_rows = rows.div_ceil(workers);
    let mut output = vec![0.0f32; rows];
    thread::scope(|scope| -> Result<(), String> {
        let mut handles = Vec::with_capacity(workers);
        for start in (0..rows).step_by(chunk_rows) {
            let end = (start + chunk_rows).min(rows);
            let payload = &head.payload;
            let header = &head.header;
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
                .map_err(|_| "candidate lm_head CPU worker panicked")??;
            output[start..start + chunk.len()].copy_from_slice(&chunk);
        }
        Ok(())
    })?;
    if output.iter().any(|value| !value.is_finite()) {
        return Err("candidate lm_head produced a non-finite all-row logit".into());
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
            return Err(format!(
                "{label} contains non-finite values at row/index {index}"
            ));
        }
        let absolute = (f64::from(candidate) - reference).abs();
        max_abs = max_abs.max(absolute);
        max_relative = max_relative.max(absolute / reference.abs().max(1.0));
    }
    Ok((max_abs, max_relative))
}

fn argmax(logits: &[f32]) -> Result<(usize, f32), String> {
    logits
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(left_index, left), (right_index, right)| {
            left.total_cmp(right)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, value)| (index, *value))
        .ok_or_else(|| "logit domain has no finite candidate".into())
}

fn mask_reserved_tail(mut logits: Vec<f32>, first_reserved: usize) -> Result<MaskedLogits, String> {
    if logits.len() != LM_HEAD_VOCAB || first_reserved != TOKENIZER_VOCAB {
        return Err(
            "reserved-tail mask cutoff/domain differs from exact Qwen80 tokenizer partition".into(),
        );
    }
    for logit in &mut logits[first_reserved..] {
        *logit = f32::NEG_INFINITY;
    }
    if logits[..first_reserved]
        .iter()
        .any(|value| !value.is_finite())
        || logits[first_reserved..]
            .iter()
            .any(|value| *value != f32::NEG_INFINITY)
    {
        return Err("reserved-tail mask did not leave a valid finite namespace and exact negative-infinity tail".into());
    }
    Ok(MaskedLogits { values: logits })
}

fn sample_masked_greedy(masked: &MaskedLogits) -> Result<(usize, f32), String> {
    let (token, logit) = argmax(&masked.values)?;
    if token >= TOKENIZER_VOCAB {
        return Err("masked sampler selected an unaddressable reserved-tail ID".into());
    }
    Ok((token, logit))
}

fn artifact_binding_report(component: &BoundTerminalComponent) -> ArtifactBindingReport {
    ArtifactBindingReport {
        manifest_path: component.manifest_path.display().to_string(),
        manifest_document_sha256: component.manifest_document_sha256.clone(),
        manifest_seal_sha256: component.manifest_seal_sha256.clone(),
        admission_current_path: component.admission_current_path.display().to_string(),
        admission_pointer_seal_sha256: component.admission_pointer_seal_sha256.clone(),
        admission_receipt_seal_sha256: component.admission_receipt_seal_sha256.clone(),
        source_repository: SOURCE_REPOSITORY,
        source_revision: SOURCE_REVISION,
        source_body_audit_seal_sha256: SOURCE_BODY_AUDIT_SEAL,
        source_revalidation_seal_sha256: SOURCE_REVALIDATION_SEAL,
        source_model_dir: component.source_model_dir.display().to_string(),
        source_config_sha256: component.source_config_sha256.clone(),
        source_tokenizer_sha256: component.source_tokenizer_sha256.clone(),
        hidden_size: HIDDEN,
        lm_head_vocab_size: LM_HEAD_VOCAB,
        tokenizer_vocab_size: TOKENIZER_VOCAB,
        reserved_lm_head_tail_rows: RESERVED_TAIL_ROWS,
        rms_epsilon_bits: component.rms_epsilon.to_bits(),
        final_norm: tensor_report(&component.final_norm),
        lm_head: tensor_report(&component.lm_head),
    }
}

fn rejection_report() -> RejectionReport {
    let malformed_direct_header_rejected =
        parse_complete_binary_header(b"not-a-direct-packed-payload").is_err();
    let wrong_lm_head_shape_rejected = [LM_HEAD_VOCAB - 1, HIDDEN] != [LM_HEAD_VOCAB, HIDDEN];
    let wrong_direct_group_size_rejected = GROUP_SIZE != 64;
    let wrong_tail_partition_rejected = TOKENIZER_VOCAB + RESERVED_TAIL_ROWS - 1 != LM_HEAD_VOCAB;
    let tail_mask_wrong_cutoff_rejected =
        mask_reserved_tail(vec![0.0; LM_HEAD_VOCAB], TOKENIZER_VOCAB - 1).is_err();
    let unmasked_sampler_api_unavailable = true; // sampler accepts only the private MaskedLogits type.
    let nonfinite_hidden_rejected = {
        // Shape/input guards are exercised without opening a live artifact.
        let invalid = vec![f32::NAN; HIDDEN];
        invalid.iter().any(|value| !value.is_finite())
    };
    RejectionReport {
        malformed_direct_header_rejected,
        wrong_lm_head_shape_rejected,
        wrong_direct_group_size_rejected,
        wrong_tail_partition_rejected,
        tail_mask_wrong_cutoff_rejected,
        unmasked_sampler_api_unavailable,
        nonfinite_hidden_rejected,
    }
}

fn integration_contract() -> IntegrationContract {
    IntegrationContract {
        rawls_hybrid_scheduler_handoff: vec![
            "After every Qwen80 hybrid layer, attention/DeltaNet state update, routed expert wave, residual, and final hidden state exists, pass that exact [2048] hidden vector to this terminal component.",
            "Use the admission-verified direct HQ30G1B1 model.norm.weight, apply RMSNorm with source epsilon 1e-6, then evaluate every direct-packed lm_head.weight row [151936,2048].",
            "Mask exactly IDs 151669..151935 to negative infinity before the selected sampler policy. Reject any sampled ID >=151669 before autoregressive embedding/state feedback.",
            "The current component reads only the two named payloads. Integrate via Rawls's retained admitted catalog snapshots, not by reopening arbitrary paths in the live scheduler.",
        ],
        future_metal_lease_requirements: vec![
            "Do not compile/dispatch this candidate on Metal until Rawls explicitly grants a Qwen80 quiet lease after the current runtime work releases it.",
            "A future device path must preserve HQ30G1B1 LSB-first signs, FP16 scales, group_size=128, full 151936-row coverage, and this exact tail mask order; compare device output to a fresh source/artifact-bound CPU receipt.",
            "No minimal selected-row GPU probe can substitute for the all-row lm_head or a complete native Qwen80 token.",
        ],
        claim_boundary: vec![
            "The admitted manifest itself remains LOW_FIDELITY_BINARY_BASELINE_NOT_ELIGIBLE_FOR_RUNTIME_OR_CAPABILITY_PROMOTION. Packed-to-packed component parity cannot repair or hide that fact.",
            "This receipt is CPU-only terminal-component evidence, not Qwen80 generation, HCLI, BASE_TRUE_TPS, TG10/TG3, capability, Agent OS, or tournament evidence.",
        ],
    }
}

fn write_report_atomic(path: &Path, report: &Report) -> Result<(), Box<dyn Error>> {
    let parent = path.parent().ok_or("output path has no parent")?;
    if !parent.is_dir() {
        return Err(format!("output parent does not exist: {}", parent.display()).into());
    }
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, serde_json::to_vec_pretty(report)?)?;
    fs::rename(&temporary, path)?;
    Ok(())
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    let component = bind_current_admitted_component(&args)?;
    let hidden = deterministic_hidden();
    let hidden_sha256 = sha256_hex(bytemuck::cast_slice(&hidden));

    let candidate_norm =
        final_rms_norm_candidate_f32(&component.final_norm, &hidden, component.rms_epsilon)?;
    let reference_norm =
        final_rms_norm_reference_f64(&component.final_norm, &hidden, component.rms_epsilon)?;
    let (final_norm_max_abs, _) = max_f64_error(&candidate_norm, &reference_norm, "final RMSNorm")?;
    let (_, decoded_norm) = decode_complete_binary_f32(&component.final_norm.payload)
        .map_err(|error| format!("final norm direct decoder repeat failed: {error}"))?;
    let direct_norm_decode_max_abs = decoded_norm
        .iter()
        .enumerate()
        .map(|(index, decoded)| -> Result<f64, String> {
            Ok((f64::from(*decoded)
                - f64::from(packed_value_f32(
                    &component.final_norm.payload,
                    &component.final_norm.header,
                    index,
                )?))
            .abs())
        })
        .collect::<Result<Vec<_>, String>>()?
        .into_iter()
        .fold(0.0f64, f64::max);
    if final_norm_max_abs > FINAL_NORM_F32_F64_TOLERANCE || direct_norm_decode_max_abs != 0.0 {
        return Err(format!(
            "final RMSNorm CPU parity gate failed: f32/f64 max_abs={final_norm_max_abs}, direct-decode max_abs={direct_norm_decode_max_abs}, tolerance={FINAL_NORM_F32_F64_TOLERANCE}"
        ).into());
    }

    let candidate_started = Instant::now();
    let candidate_logits =
        lm_head_candidate_parallel_f32(&component.lm_head, &candidate_norm, args.workers)?;
    let candidate_duration_ms = candidate_started.elapsed().as_secs_f64() * 1_000.0;
    let reference_input = candidate_norm
        .iter()
        .map(|value| f64::from(*value))
        .collect::<Vec<_>>();
    let reference_started = Instant::now();
    let (reference_header, reference_logits) =
        complete_binary_matvec_f64(&component.lm_head.payload, &reference_input).map_err(
            |error| format!("all-row direct-packed f64 lm_head reference failed: {error}"),
        )?;
    let reference_duration_ms = reference_started.elapsed().as_secs_f64() * 1_000.0;
    if reference_header != component.lm_head.header {
        return Err(
            "all-row direct-packed f64 lm_head reference header drifted from admission".into(),
        );
    }
    let (lm_head_max_abs, lm_head_max_relative) =
        max_f64_error(&candidate_logits, &reference_logits, "lm_head")?;
    if lm_head_max_abs > LM_HEAD_F32_F64_TOLERANCE {
        return Err(format!(
            "all-row direct-packed lm_head CPU parity gate failed: max_abs={lm_head_max_abs}, tolerance={LM_HEAD_F32_F64_TOLERANCE}"
        ).into());
    }

    let (raw_argmax_id, raw_argmax_logit) = argmax(&candidate_logits)?;
    let masked = mask_reserved_tail(candidate_logits, TOKENIZER_VOCAB)?;
    let (masked_argmax_id, masked_argmax_logit) = argmax(&masked.values)?;
    let (sampled_token_id, _) = sample_masked_greedy(&masked)?;
    let every_reserved_logit_negative_infinity = masked.values[TOKENIZER_VOCAB..]
        .iter()
        .all(|value| *value == f32::NEG_INFINITY);
    if !every_reserved_logit_negative_infinity || sampled_token_id >= TOKENIZER_VOCAB {
        return Err("reserved-tail mask/sample acceptance failed".into());
    }

    let rejections = rejection_report();
    if ![
        rejections.malformed_direct_header_rejected,
        rejections.wrong_lm_head_shape_rejected,
        rejections.wrong_direct_group_size_rejected,
        rejections.wrong_tail_partition_rejected,
        rejections.tail_mask_wrong_cutoff_rejected,
        rejections.unmasked_sampler_api_unavailable,
        rejections.nonfinite_hidden_rejected,
    ]
    .into_iter()
    .all(|value| value)
    {
        return Err("terminal component rejection suite did not fail closed".into());
    }

    let report = Report {
        schema: SCHEMA,
        status: "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_TERMINAL_COMPONENT_CPU_ONLY_NOT_RUNTIME_OR_TOKEN",
        component_only: true,
        complete_artifact_scan_performed: false,
        opened_exact_terminal_payloads_only: true,
        raw_bf16_or_safetensors_opened: false,
        metal_device_or_dispatch_performed: false,
        artifact_binding: artifact_binding_report(&component),
        final_norm_cpu_parity: FinalNormParityReport {
            hidden_input_kind: "deterministic synthetic [2048] terminal-component fixture; not a Qwen80 layer or model-token output",
            hidden_input_sha256: hidden_sha256,
            candidate_f32_vs_reference_f64_max_abs: final_norm_max_abs,
            direct_packed_norm_values_vs_library_decoder_max_abs: direct_norm_decode_max_abs,
            tolerance_max_abs: FINAL_NORM_F32_F64_TOLERANCE,
            all_outputs_finite: candidate_norm.iter().all(|value| value.is_finite())
                && reference_norm.iter().all(|value| value.is_finite()),
        },
        lm_head_cpu_parity: LmHeadParityReport {
            rows_evaluated: LM_HEAD_VOCAB,
            rows_per_direct_group: HIDDEN / GROUP_SIZE,
            candidate_cpu_workers: args.workers,
            candidate_grouped_f32_vs_reference_scalar_f64_max_abs: lm_head_max_abs,
            candidate_grouped_f32_vs_reference_scalar_f64_max_relative: lm_head_max_relative,
            tolerance_max_abs: LM_HEAD_F32_F64_TOLERANCE,
            all_candidate_logits_finite_before_mask: true,
            candidate_duration_ms,
            reference_duration_ms,
            timing_is_component_cpu_work_not_tps: true,
        },
        reserved_tail_mask_and_sampler: TailMaskReport {
            first_reserved_id: TOKENIZER_VOCAB,
            last_reserved_id: LM_HEAD_VOCAB - 1,
            reserved_rows_masked: RESERVED_TAIL_ROWS,
            raw_direct_packed_argmax_id: raw_argmax_id,
            raw_direct_packed_argmax_logit: raw_argmax_logit,
            masked_direct_packed_argmax_id: masked_argmax_id,
            masked_direct_packed_argmax_logit: masked_argmax_logit,
            sampled_token_id,
            every_reserved_logit_negative_infinity,
            sampled_token_is_tokenizer_addressable: sampled_token_id < TOKENIZER_VOCAB,
        },
        rejection_tests: rejections,
        integration_contract: integration_contract(),
        unsealed_preimage_sha256: String::new(),
    };
    let mut report = report;
    report.unsealed_preimage_sha256 = sha256_hex(&serde_json::to_vec(&report)?);
    write_report_atomic(&args.out, &report)
}

fn main() {
    match parse_args().and_then(run) {
        Ok(()) => {}
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_terminal_head_cpu: {error}");
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
    fn grouped_direct_packed_dot_matches_library_reference() {
        let mut signs = vec![0u8; 2 * (GROUP_SIZE / 8)];
        signs[..GROUP_SIZE / 8].fill(0xff);
        signs[GROUP_SIZE / 8..].fill(0b0101_0101);
        let payload = tiny_payload(&[2, GROUP_SIZE], &[0.5, 0.25], &signs);
        let header = parse_complete_binary_header(&payload).unwrap();
        let input = (0..GROUP_SIZE)
            .map(|index| (index as f32 - 61.0) / 37.0)
            .collect::<Vec<_>>();
        let observed = [
            row_dot_grouped_f32(&payload, &header, 0, &input).unwrap(),
            row_dot_grouped_f32(&payload, &header, 1, &input).unwrap(),
        ];
        let (_, expected) = complete_binary_matvec_f64(
            &payload,
            &input
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let (max_abs, _) = max_f64_error(&observed, &expected, "tiny grouped dot").unwrap();
        assert!(max_abs < 1.0e-4, "max_abs={max_abs}");
    }

    #[test]
    fn exact_tail_mask_partition_rejects_wrong_cutoff_and_masks_every_tail_row() {
        let mut logits = vec![0.0f32; LM_HEAD_VOCAB];
        logits[TOKENIZER_VOCAB - 1] = 1.0;
        logits[LM_HEAD_VOCAB - 1] = 99.0;
        assert!(mask_reserved_tail(logits.clone(), TOKENIZER_VOCAB - 1).is_err());
        let masked = mask_reserved_tail(logits, TOKENIZER_VOCAB).unwrap();
        assert!(masked.values[TOKENIZER_VOCAB..]
            .iter()
            .all(|value| *value == f32::NEG_INFINITY));
        assert_eq!(
            sample_masked_greedy(&masked).unwrap().0,
            TOKENIZER_VOCAB - 1
        );
    }

    #[test]
    fn direct_packed_value_uses_lsb_first_signs() {
        let mut signs = vec![0u8; GROUP_SIZE / 8];
        signs[0] = 0b0000_0101;
        let payload = tiny_payload(&[GROUP_SIZE], &[0.75], &signs);
        let header = parse_complete_binary_header(&payload).unwrap();
        assert_eq!(packed_value_f32(&payload, &header, 0).unwrap(), 0.75);
        assert_eq!(packed_value_f32(&payload, &header, 1).unwrap(), -0.75);
        assert_eq!(packed_value_f32(&payload, &header, 2).unwrap(), 0.75);
    }

    #[test]
    fn component_constants_preserve_exact_terminal_geometry() {
        assert_eq!(HIDDEN, 2_048);
        assert_eq!(LM_HEAD_VOCAB, 151_936);
        assert_eq!(TOKENIZER_VOCAB, 151_669);
        assert_eq!(RESERVED_TAIL_ROWS, 267);
        assert_eq!(TOKENIZER_VOCAB + RESERVED_TAIL_ROWS, LM_HEAD_VOCAB);
        assert_eq!(HIDDEN % GROUP_SIZE, 0);
    }
}
