//! Strict Qwen80 layer-0 MoE-combine / second-residual component.
//!
//! This is deliberately narrower than a Qwen3Next layer.  It binds the
//! source-selected layer-0 router result, then verifies the fixed operation
//! order used to combine ten *materialized* weighted-route vectors with a
//! materialized shared-expert vector and first residual.  The fixture is not
//! a substitute for ten physical expert waves: the component records that
//! fact in every success receipt.  It never advances the native runtime,
//! watcher, token, generation, HCLI, TPS, TG, or tournament state.

use hawking_core::metal::{MetalContext, PinnedBuffer, TokenCommandBuffer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;
use std::time::{SystemTime, UNIX_EPOCH};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_moe_combine.v1";
const CAPTURE_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_moe_combine_capture.v1";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ROUTER_INNER_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const ROUTER_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const ROUTER_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1";
const ROUTER_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY";
const CPU_BASELINE_SCHEMA: &str = "hawking.ascension.qwen80_moe_combine_cpu_baseline_wrapper.v1";
const CPU_BASELINE_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_CPU_ORACLE_BASELINE";
const QUIET_LEASE_SCHEMA: &str = "hawking.ascension.qwen80_moe_combine_quiet_metal_lease.v1";
const QUIET_LEASE_STATUS: &str =
    "GRANTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_NON_TIMED_DEVICE_PARITY_LEASE";
const QUIET_LEASE_COMPONENT: &str = "qwen80_direct_packed_moe_combine";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const MANIFEST_DOCUMENT_SHA256: &str =
    "a0fcac0401a7962402bb8cb87d5055c83667b39575f9e0f4c7470d080758aa10";
const MANIFEST_SEAL: &str = "14cf6c4d17086dabc54b53b4dd28b9f6551ef06c6d8bf4ee8453d775d0f6817b";
const ADMISSION_RECEIPT_SEAL: &str =
    "939b41322363da3db774a2530b207bf380ed641d23cae671fc6438c0eecbf628";
const SOURCE_BODY_AUDIT_SEAL: &str =
    "c572b2270b623b8677c374b43c89ddd729de135c25721488bb874b184ff8c3d4";
const SOURCE_REVALIDATION_SEAL: &str =
    "541b16fca1d4805ecba356face97b4e8de1accdeb21e98ee0c13b70ab0746c45";
const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;
const EXPERTS: usize = 512;
const EXPECTED_IDS: [u16; TOP_K] = [65, 245, 227, 35, 189, 440, 298, 405, 109, 494];
const F32_F64_TOLERANCE: f64 = 2.0e-5;
const DEVICE_ROUTED_SUM_TOLERANCE: f32 = 3.0e-5;
const DEVICE_SECOND_RESIDUAL_TOLERANCE: f32 = 3.0e-5;

/// Component-local scalar bindings.  They deliberately do not enter the
/// generic runtime interface.
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
    router_receipt: PathBuf,
    router_outer_receipt: PathBuf,
    cpu_baseline_receipt: Option<PathBuf>,
    lease_receipt: Option<PathBuf>,
    capture_dir: PathBuf,
    mode: Mode,
    workers: usize,
}

#[derive(Clone, Debug)]
struct FileEvidence {
    path: PathBuf,
    bytes: usize,
    sha256: String,
}

#[derive(Clone, Debug)]
struct RouterEvidence {
    inner: FileEvidence,
    outer: FileEvidence,
    outer_seal_sha256: String,
    ids: [u16; TOP_K],
    weights: [f32; TOP_K],
    weights_f64: [f64; TOP_K],
}

#[derive(Clone, Debug)]
struct BoundComponent {
    manifest: FileEvidence,
    admission: FileEvidence,
    admission_pointer_seal_sha256: String,
    router: RouterEvidence,
}

#[derive(Clone, Debug)]
struct AggregateInput {
    ids: [u16; TOP_K],
    weights: [f32; TOP_K],
    routes: Vec<Vec<f32>>,
    gated_shared: Vec<f32>,
    first_residual: Vec<f32>,
}

#[derive(Clone, Debug)]
struct CpuOracle {
    input: AggregateInput,
    routed_sum: Vec<f32>,
    second_residual: Vec<f32>,
    routed_f32_f64_max_abs: f64,
    second_f32_f64_max_abs: f64,
}

#[derive(Clone, Debug)]
struct BaselineEvidence {
    receipt: FileEvidence,
    seal_sha256: String,
}

#[derive(Clone, Debug)]
struct LeaseEvidence {
    receipt: FileEvidence,
    seal_sha256: String,
}

#[derive(Clone, Debug)]
struct DeviceLedger {
    device_name: String,
    dispatches: usize,
    routed_sum: Vec<f32>,
    second_residual: Vec<f32>,
    routed_sum_max_abs: f32,
    second_residual_max_abs: f32,
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32_sha256(values: &[f32]) -> String {
    sha256_hex(bytemuck::cast_slice(values))
}

fn mode_name(mode: Mode) -> &'static str {
    match mode {
        Mode::CpuOracle => "cpu-oracle",
        Mode::Metal => "metal",
    }
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

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let _ = regular_file_bytes(path, label)?;
    path.canonicalize().map_err(|error| {
        format!(
            "{label} canonicalization failed at {}: {error}",
            path.display()
        )
    })
}

fn evidence(path: &Path, label: &str) -> Result<FileEvidence, String> {
    let path = canonical_regular(path, label)?;
    let bytes = regular_file_bytes(&path, label)?;
    Ok(FileEvidence {
        path,
        bytes: bytes.len(),
        sha256: sha256_hex(&bytes),
    })
}

fn json_object(path: &Path, label: &str) -> Result<(FileEvidence, Map<String, Value>), String> {
    let file = evidence(path, label)?;
    let bytes = regular_file_bytes(&file.path, label)?;
    let document = serde_json::from_slice::<Value>(&bytes)
        .map_err(|error| format!("{label} invalid JSON: {error}"))?
        .as_object()
        .cloned()
        .ok_or_else(|| format!("{label} root must be an object"))?;
    Ok((file, document))
}

fn string<'a>(object: &'a Map<String, Value>, key: &str, label: &str) -> Result<&'a str, String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} missing string {key:?}"))
}

fn object<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} missing object {key:?}"))
}

fn array<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    object
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} missing array {key:?}"))
}

fn bool_field(
    object: &Map<String, Value>,
    key: &str,
    expected: bool,
    label: &str,
) -> Result<(), String> {
    if object.get(key).and_then(Value::as_bool) != Some(expected) {
        return Err(format!("{label} {key:?} is not {expected}"));
    }
    Ok(())
}

fn lower_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} is not a lowercase SHA-256"));
    }
    Ok(())
}

fn json_path(value: &str, label: &str) -> Result<PathBuf, String> {
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

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut manifest = None;
    let mut admission_current = None;
    let mut router_receipt = None;
    let mut router_outer_receipt = None;
    let mut cpu_baseline_receipt = None;
    let mut lease_receipt = None;
    let mut capture_dir = None;
    let mut mode = None;
    let mut workers = None;
    let mut values = env::args().skip(1);
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        match flag.as_str() {
            "--manifest" => set_once(&mut manifest, PathBuf::from(value), "--manifest")?,
            "--admission-current" => set_once(
                &mut admission_current,
                PathBuf::from(value),
                "--admission-current",
            )?,
            "--router-receipt" => set_once(
                &mut router_receipt,
                PathBuf::from(value),
                "--router-receipt",
            )?,
            "--router-outer-receipt" => set_once(
                &mut router_outer_receipt,
                PathBuf::from(value),
                "--router-outer-receipt",
            )?,
            "--cpu-baseline-receipt" => set_once(
                &mut cpu_baseline_receipt,
                PathBuf::from(value),
                "--cpu-baseline-receipt",
            )?,
            "--lease-receipt" => {
                set_once(&mut lease_receipt, PathBuf::from(value), "--lease-receipt")?
            }
            "--capture-dir" => set_once(&mut capture_dir, PathBuf::from(value), "--capture-dir")?,
            "--mode" => {
                let parsed = match value.as_str() {
                    "cpu-oracle" => Mode::CpuOracle,
                    "metal" => Mode::Metal,
                    _ => return Err("--mode must be cpu-oracle or metal".into()),
                };
                set_once(&mut mode, parsed, "--mode")?;
            }
            "--workers" => {
                let parsed = value
                    .parse::<usize>()
                    .map_err(|_| "--workers must be an integer")?;
                if parsed == 0 {
                    return Err("--workers must be positive".into());
                }
                set_once(&mut workers, parsed, "--workers")?;
            }
            _ => {
                return Err("unknown argument; expected strict Qwen80 MoE-combine arguments".into())
            }
        }
    }
    let args = Args {
        manifest: manifest.ok_or("missing --manifest")?,
        admission_current: admission_current.ok_or("missing --admission-current")?,
        router_receipt: router_receipt.ok_or("missing --router-receipt")?,
        router_outer_receipt: router_outer_receipt.ok_or("missing --router-outer-receipt")?,
        cpu_baseline_receipt,
        lease_receipt,
        capture_dir: capture_dir.ok_or("missing --capture-dir")?,
        mode: mode.ok_or("missing --mode")?,
        workers: workers.ok_or("missing --workers")?,
    };
    for (path, label) in [
        (&args.manifest, "--manifest"),
        (&args.admission_current, "--admission-current"),
        (&args.router_receipt, "--router-receipt"),
        (&args.router_outer_receipt, "--router-outer-receipt"),
    ] {
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute").into());
        }
    }
    if !args.capture_dir.is_absolute() {
        return Err("--capture-dir must be absolute".into());
    }
    match args.mode {
        Mode::CpuOracle if args.cpu_baseline_receipt.is_some() || args.lease_receipt.is_some() => {
            return Err("CPU oracle refuses baseline/lease inputs so it cannot mistake a prior result for an oracle".into())
        }
        Mode::Metal if args.cpu_baseline_receipt.is_none() || args.lease_receipt.is_none() => {
            return Err("--mode metal requires --cpu-baseline-receipt and --lease-receipt".into())
        }
        _ => {}
    }
    Ok(args)
}

fn set_once<T>(slot: &mut Option<T>, value: T, label: &str) -> Result<(), Box<dyn Error>> {
    if slot.replace(value).is_some() {
        return Err(format!("{label} repeated").into());
    }
    Ok(())
}

fn validate_manifest(path: &Path) -> Result<FileEvidence, String> {
    let (file, document) = json_object(path, "Qwen80 complete manifest")?;
    if file.sha256 != MANIFEST_DOCUMENT_SHA256
        || string(&document, "schema", "Qwen80 complete manifest")? != MANIFEST_SCHEMA
        || string(&document, "status", "Qwen80 complete manifest")?
            != "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED"
        || string(&document, "seal_sha256", "Qwen80 complete manifest")? != MANIFEST_SEAL
        || string(
            &document,
            "source_body_audit_seal_sha256",
            "Qwen80 complete manifest",
        )? != SOURCE_BODY_AUDIT_SEAL
        || string(
            &document,
            "source_revalidation_receipt_seal_sha256",
            "Qwen80 complete manifest",
        )? != SOURCE_REVALIDATION_SEAL
    {
        return Err("Qwen80 complete manifest identity/status drifted".into());
    }
    let source = object(&document, "source", "Qwen80 complete manifest")?;
    if string(source, "repository", "Qwen80 complete manifest source")? != SOURCE_REPOSITORY
        || source.get("tensor_count").and_then(Value::as_u64) != Some(74_391)
    {
        return Err("Qwen80 complete manifest source authority drifted".into());
    }
    Ok(file)
}

fn validate_admission(
    path: &Path,
    manifest: &FileEvidence,
) -> Result<(FileEvidence, String), String> {
    let (file, document) = json_object(path, "Qwen80 admission current pointer")?;
    let pointer_seal = string(&document, "seal_sha256", "Qwen80 admission current pointer")?;
    lower_sha256(pointer_seal, "Qwen80 admission pointer seal")?;
    if string(&document, "schema", "Qwen80 admission current pointer")? != ADMISSION_SCHEMA
        || string(&document, "status", "Qwen80 admission current pointer")?
            != "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED"
    {
        return Err("Qwen80 current admission schema/status drifted".into());
    }
    let model = object(&document, "model", "Qwen80 admission current pointer")?;
    if string(model, "id", "Qwen80 admission model")? != MODEL_ID
        || string(model, "key", "Qwen80 admission model")? != MODEL_KEY
        || string(model, "repository", "Qwen80 admission model")? != SOURCE_REPOSITORY
        || string(model, "revision", "Qwen80 admission model")? != SOURCE_REVISION
    {
        return Err("Qwen80 admission model identity drifted".into());
    }
    let complete = object(
        &document,
        "complete_manifest",
        "Qwen80 admission current pointer",
    )?;
    if json_path(
        string(complete, "path", "Qwen80 admission manifest")?,
        "Qwen80 admission manifest path",
    )? != manifest.path
        || string(complete, "document_sha256", "Qwen80 admission manifest")? != manifest.sha256
        || string(complete, "seal_sha256", "Qwen80 admission manifest")? != MANIFEST_SEAL
    {
        return Err("Qwen80 admission no longer selects the current manifest".into());
    }
    let receipt = object(
        &document,
        "admission_receipt",
        "Qwen80 admission current pointer",
    )?;
    if string(receipt, "seal_sha256", "Qwen80 admission receipt")? != ADMISSION_RECEIPT_SEAL {
        return Err("Qwen80 admission receipt seal drifted".into());
    }
    Ok((file, pointer_seal.to_owned()))
}

fn parse_ids(values: &[Value]) -> Result<[u16; TOP_K], String> {
    if values.len() != TOP_K {
        return Err("router top-k must contain exactly ten IDs".into());
    }
    let mut ids = [0u16; TOP_K];
    for (index, value) in values.iter().enumerate() {
        ids[index] = value
            .as_u64()
            .and_then(|value| u16::try_from(value).ok())
            .filter(|value| usize::from(*value) < EXPERTS)
            .ok_or_else(|| format!("router ID {index} is invalid"))?;
        if ids[..index].contains(&ids[index]) {
            return Err(format!("router ID {index} is duplicated"));
        }
    }
    if ids != EXPECTED_IDS {
        return Err("router IDs drifted from the exact source-selected layer-0 top-10".into());
    }
    Ok(ids)
}

fn parse_weights(values: &[Value]) -> Result<([f32; TOP_K], [f64; TOP_K]), String> {
    if values.len() != TOP_K {
        return Err("router top-k must contain exactly ten weights".into());
    }
    let mut f32_values = [0.0; TOP_K];
    let mut f64_values = [0.0; TOP_K];
    for (index, value) in values.iter().enumerate() {
        let parsed = value
            .as_f64()
            .filter(|value| value.is_finite() && *value > 0.0)
            .ok_or_else(|| format!("router weight {index} is invalid"))?;
        f32_values[index] = parsed as f32;
        if !f32_values[index].is_finite() {
            return Err(format!("router weight {index} cannot fit f32"));
        }
        f64_values[index] = parsed;
    }
    if (f64_values.iter().sum::<f64>() - 1.0).abs() > 2.0e-6 {
        return Err("router weights do not sum to one".into());
    }
    Ok((f32_values, f64_values))
}

fn validate_router(
    inner_path: &Path,
    outer_path: &Path,
    manifest: &FileEvidence,
    admission: &FileEvidence,
) -> Result<RouterEvidence, String> {
    let (inner_file, inner) = json_object(inner_path, "Qwen80 router inner receipt")?;
    if string(&inner, "schema", "Qwen80 router inner receipt")? != ROUTER_INNER_SCHEMA
        || string(&inner, "status", "Qwen80 router inner receipt")? != ROUTER_INNER_STATUS
        || string(&inner, "mode", "Qwen80 router inner receipt")? != "metal"
    {
        return Err("Qwen80 router inner does not describe the strict-Math Metal component".into());
    }
    bool_field(
        &inner,
        "component_only",
        true,
        "Qwen80 router inner receipt",
    )?;
    bool_field(
        &inner,
        "metal_device_or_dispatch_performed",
        true,
        "Qwen80 router inner receipt",
    )?;
    let artifact = object(&inner, "artifact_binding", "Qwen80 router inner receipt")?;
    if json_path(
        string(artifact, "manifest_path", "Qwen80 router artifact")?,
        "Qwen80 router manifest path",
    )? != manifest.path
        || string(
            artifact,
            "manifest_document_sha256",
            "Qwen80 router artifact",
        )? != manifest.sha256
        || string(artifact, "manifest_seal_sha256", "Qwen80 router artifact")? != MANIFEST_SEAL
        || json_path(
            string(artifact, "admission_current_path", "Qwen80 router artifact")?,
            "Qwen80 router admission path",
        )? != admission.path
        || string(
            artifact,
            "admission_receipt_seal_sha256",
            "Qwen80 router artifact",
        )? != ADMISSION_RECEIPT_SEAL
        || artifact.get("layer").and_then(Value::as_u64) != Some(0)
        || artifact.get("hidden").and_then(Value::as_u64) != Some(HIDDEN as u64)
        || artifact.get("experts_per_token").and_then(Value::as_u64) != Some(TOP_K as u64)
    {
        return Err("Qwen80 router artifact binding drifted".into());
    }
    let top10 = object(
        &inner,
        "source_stable_top10_router",
        "Qwen80 router inner receipt",
    )?;
    bool_field(
        top10,
        "device_ids_exact_match",
        true,
        "Qwen80 router top-10",
    )?;
    bool_field(
        top10,
        "ids_unique_and_in_range",
        true,
        "Qwen80 router top-10",
    )?;
    let ids = parse_ids(array(top10, "ids", "Qwen80 router top-10")?)?;
    if parse_ids(array(top10, "device_ids", "Qwen80 router top-10")?)? != ids {
        return Err("Qwen80 router device IDs differ from accepted top-10".into());
    }
    let (weights, weights_f64) = parse_weights(array(
        top10,
        "renormalized_weights",
        "Qwen80 router top-10",
    )?)?;

    let (outer_file, outer) = json_object(outer_path, "Qwen80 router outer terminal")?;
    let outer_seal = string(&outer, "seal_sha256", "Qwen80 router outer terminal")?;
    lower_sha256(outer_seal, "Qwen80 router outer seal")?;
    if string(&outer, "schema", "Qwen80 router outer terminal")? != ROUTER_OUTER_SCHEMA
        || string(&outer, "status", "Qwen80 router outer terminal")? != ROUTER_OUTER_STATUS
    {
        return Err("Qwen80 router outer terminal schema/status drifted".into());
    }
    let source = object(&outer, "source_binding", "Qwen80 router outer terminal")?;
    let outer_manifest = object(source, "manifest", "Qwen80 router outer source binding")?;
    if string(outer_manifest, "sha256", "Qwen80 router outer manifest")? != manifest.sha256
        || json_path(
            string(outer_manifest, "path", "Qwen80 router outer manifest")?,
            "Qwen80 router outer manifest path",
        )? != manifest.path
    {
        return Err("Qwen80 router outer manifest evidence drifted".into());
    }
    let outer_admission = object(
        source,
        "admission_current",
        "Qwen80 router outer source binding",
    )?;
    if json_path(
        string(outer_admission, "path", "Qwen80 router outer admission")?,
        "Qwen80 router outer admission path",
    )? != admission.path
    {
        return Err("Qwen80 router outer admission path drifted".into());
    }
    let captured = object(
        &outer,
        "inner_probe_capture",
        "Qwen80 router outer terminal",
    )?;
    if json_path(
        string(captured, "path", "Qwen80 router outer inner")?,
        "Qwen80 router outer inner path",
    )? != inner_file.path
        || string(captured, "sha256", "Qwen80 router outer inner")? != inner_file.sha256
        || string(captured, "schema", "Qwen80 router outer inner")? != ROUTER_INNER_SCHEMA
        || string(captured, "status", "Qwen80 router outer inner")? != ROUTER_INNER_STATUS
        || string(captured, "mode", "Qwen80 router outer inner")? != "metal"
    {
        return Err("Qwen80 router outer does not bind the exact inner component".into());
    }
    bool_field(
        captured,
        "metal_performed",
        true,
        "Qwen80 router outer inner",
    )?;
    Ok(RouterEvidence {
        inner: inner_file,
        outer: outer_file,
        outer_seal_sha256: outer_seal.to_owned(),
        ids,
        weights,
        weights_f64,
    })
}

fn bind_current(args: &Args) -> Result<BoundComponent, String> {
    let manifest = validate_manifest(&args.manifest)?;
    let (admission, admission_pointer_seal_sha256) =
        validate_admission(&args.admission_current, &manifest)?;
    let router = validate_router(
        &args.router_receipt,
        &args.router_outer_receipt,
        &manifest,
        &admission,
    )?;
    Ok(BoundComponent {
        manifest,
        admission,
        admission_pointer_seal_sha256,
        router,
    })
}

fn deterministic_values(mut seed: u64, elements: usize, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|index| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((seed >> 40) & 0x00ff_ffff) as f32 / 16_777_215.0;
            let phase = ((index * 29 % 113) as f32 - 56.0) / 257.0;
            ((unit * 2.0 - 1.0) + phase) * scale
        })
        .collect()
}

fn materialized_fixture(router: &RouterEvidence) -> AggregateInput {
    let routes = router
        .weights
        .iter()
        .enumerate()
        .map(|(index, weight)| {
            deterministic_values(
                0x7a20_5f4e_19d3_0001u64
                    .wrapping_add(u64::from(router.ids[index]) * 131)
                    .wrapping_add(index as u64),
                HIDDEN,
                0.42,
            )
            .into_iter()
            .map(|value| value * weight)
            .collect()
        })
        .collect();
    AggregateInput {
        ids: router.ids,
        weights: router.weights,
        routes,
        gated_shared: deterministic_values(0x5ced_1a2b_9043_0002, HIDDEN, 0.31),
        first_residual: deterministic_values(0x4ab2_8917_ef0c_0003, HIDDEN, 0.63),
    }
}

fn validate_input(input: &AggregateInput, router: &RouterEvidence) -> Result<(), String> {
    if input.ids != router.ids || input.ids != EXPECTED_IDS || input.routes.len() != TOP_K {
        return Err(
            "MoE combine requires the exact ten source-selected route-index ordered inputs".into(),
        );
    }
    let mut seen = [false; EXPERTS];
    for (index, id) in input.ids.iter().enumerate() {
        let index_id = usize::from(*id);
        if index_id >= EXPERTS || seen[index_id] {
            return Err(format!("route {index} is invalid or duplicate"));
        }
        seen[index_id] = true;
        if (input.weights[index] - router.weights[index]).abs() > 1.0e-7 {
            return Err(format!(
                "route weight {index} drifted from the bound router receipt"
            ));
        }
    }
    let sum = input.weights.iter().sum::<f32>();
    if !sum.is_finite() || (sum - 1.0).abs() > 2.0e-6 {
        return Err("route weights are not normalized".into());
    }
    for (index, route) in input.routes.iter().enumerate() {
        if route.len() != HIDDEN || route.iter().any(|value| !value.is_finite()) {
            return Err(format!("route {index} is not a finite [2048] vector"));
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
        return Err("shared/first residual vectors are not finite [2048]".into());
    }
    Ok(())
}

fn aggregate_f32(input: &AggregateInput) -> Result<(Vec<f32>, Vec<f32>), String> {
    let mut routed = vec![0.0f32; HIDDEN];
    for route in &input.routes {
        for (sum, value) in routed.iter_mut().zip(route) {
            *sum += *value;
        }
    }
    let second = routed
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
    if routed
        .iter()
        .chain(second.iter())
        .any(|value| !value.is_finite())
    {
        return Err("fixed f32 MoE combine generated a non-finite value".into());
    }
    Ok((routed, second))
}

fn aggregate_f64(input: &AggregateInput) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut routed = vec![0.0f64; HIDDEN];
    for route in &input.routes {
        for (sum, value) in routed.iter_mut().zip(route) {
            *sum += f64::from(*value);
        }
    }
    let second = routed
        .iter()
        .zip(&input.gated_shared)
        .zip(&input.first_residual)
        .map(|((&routed, &shared), &first)| routed + f64::from(shared) + f64::from(first))
        .collect::<Vec<_>>();
    if routed
        .iter()
        .chain(second.iter())
        .any(|value| !value.is_finite())
    {
        return Err("f64 MoE combine reference generated a non-finite value".into());
    }
    Ok((routed, second))
}

fn max_f64_error(candidate: &[f32], reference: &[f64], label: &str) -> Result<f64, String> {
    if candidate.len() != reference.len() {
        return Err(format!("{label} geometry drifted"));
    }
    let mut maximum = 0.0f64;
    for (index, (&candidate, &reference)) in candidate.iter().zip(reference).enumerate() {
        if !candidate.is_finite() || !reference.is_finite() {
            return Err(format!("{label} non-finite value at {index}"));
        }
        maximum = maximum.max((f64::from(candidate) - reference).abs());
    }
    Ok(maximum)
}

fn build_cpu_oracle(component: &BoundComponent) -> Result<CpuOracle, String> {
    let input = materialized_fixture(&component.router);
    validate_input(&input, &component.router)?;
    let (routed_sum, second_residual) = aggregate_f32(&input)?;
    let (routed_f64, second_f64) = aggregate_f64(&input)?;
    let routed_f32_f64_max_abs = max_f64_error(&routed_sum, &routed_f64, "routed sum")?;
    let second_f32_f64_max_abs = max_f64_error(&second_residual, &second_f64, "second residual")?;
    if routed_f32_f64_max_abs > F32_F64_TOLERANCE || second_f32_f64_max_abs > F32_F64_TOLERANCE {
        return Err(format!(
            "fixed order f32/f64 MoE combine parity exceeded {F32_F64_TOLERANCE}: routed={routed_f32_f64_max_abs}, second={second_f32_f64_max_abs}"
        ));
    }
    Ok(CpuOracle {
        input,
        routed_sum,
        second_residual,
        routed_f32_f64_max_abs,
        second_f32_f64_max_abs,
    })
}

fn bytes_for<T>(elements: usize, label: &str) -> Result<usize, String> {
    elements
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| format!("{label} byte count overflow"))
}

fn max_f32_error(expected: &[f32], observed: &[f32], label: &str) -> Result<f32, String> {
    if expected.len() != observed.len() {
        return Err(format!("{label} geometry drifted"));
    }
    let mut maximum = 0.0f32;
    for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
        if !expected.is_finite() || !observed.is_finite() {
            return Err(format!("{label} non-finite value at {index}"));
        }
        maximum = maximum.max((expected - observed).abs());
    }
    Ok(maximum)
}

fn snapshot_f32(buffer: &PinnedBuffer, elements: usize, label: &str) -> Result<Vec<f32>, String> {
    let bytes = bytes_for::<f32>(elements, label)?;
    if buffer.length() < bytes as u64 {
        return Err(format!("{label} buffer is shorter than its fixed geometry"));
    }
    // This only happens after the single command buffer fence below.
    Ok(unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, elements).to_vec() })
}

fn ensure_non_timed_environment() -> Result<(), String> {
    let disallowed = ["HAWKING_TRACE_DISPATCH", "HAWKING_TCB_TRACE"]
        .into_iter()
        .filter(|key| env::var_os(key).is_some())
        .collect::<Vec<_>>();
    if disallowed.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "strict MoE-combine component refuses timing/trace environment: {}",
            disallowed.join(", ")
        ))
    }
}

fn evidence_matches(
    value: &Map<String, Value>,
    expected: &FileEvidence,
    label: &str,
) -> Result<(), String> {
    if value.get("present").and_then(Value::as_bool) != Some(true)
        || json_path(string(value, "path", label)?, &format!("{label}.path"))? != expected.path
        || value.get("bytes").and_then(Value::as_u64) != Some(expected.bytes as u64)
        || string(value, "sha256", label)? != expected.sha256
    {
        return Err(format!("{label} immutable file evidence drifted"));
    }
    Ok(())
}

fn source_top10_binding(component: &BoundComponent) -> Value {
    json!({
        "router_receipt_path": component.router.inner.path,
        "router_receipt_sha256": component.router.inner.sha256,
        "router_outer_receipt_path": component.router.outer.path,
        "router_outer_receipt_sha256": component.router.outer.sha256,
        "router_outer_receipt_seal_sha256": component.router.outer_seal_sha256,
        "ids": component.router.ids,
    })
}

fn validate_source_top10_binding(
    value: &Map<String, Value>,
    component: &BoundComponent,
    label: &str,
) -> Result<(), String> {
    if json_path(
        string(value, "router_receipt_path", label)?,
        &format!("{label}.router_receipt_path"),
    )? != component.router.inner.path
        || string(value, "router_receipt_sha256", label)? != component.router.inner.sha256
        || json_path(
            string(value, "router_outer_receipt_path", label)?,
            &format!("{label}.router_outer_receipt_path"),
        )? != component.router.outer.path
        || string(value, "router_outer_receipt_sha256", label)? != component.router.outer.sha256
        || string(value, "router_outer_receipt_seal_sha256", label)?
            != component.router.outer_seal_sha256
    {
        return Err(format!("{label} router evidence drifted"));
    }
    let ids = parse_ids(array(value, "ids", label)?)?;
    if ids != component.router.ids {
        return Err(format!("{label} source top-10 IDs drifted"));
    }
    Ok(())
}

fn validate_artifact_binding(
    binding: &Map<String, Value>,
    component: &BoundComponent,
    label: &str,
    require_current_pointer: bool,
) -> Result<(), String> {
    if json_path(
        string(binding, "manifest_path", label)?,
        &format!("{label}.manifest_path"),
    )? != component.manifest.path
        || string(binding, "manifest_document_sha256", label)? != component.manifest.sha256
        || string(binding, "manifest_seal_sha256", label)? != MANIFEST_SEAL
        || json_path(
            string(binding, "admission_current_path", label)?,
            &format!("{label}.admission_current_path"),
        )? != component.admission.path
        || string(binding, "admission_receipt_seal_sha256", label)? != ADMISSION_RECEIPT_SEAL
    {
        return Err(format!("{label} complete-artifact binding drifted"));
    }
    if require_current_pointer
        && string(binding, "admission_pointer_seal_sha256", label)?
            != component.admission_pointer_seal_sha256
    {
        return Err(format!(
            "{label} admission pointer resealed during same capture"
        ));
    }
    Ok(())
}

fn validate_cpu_baseline(
    path: &Path,
    component: &BoundComponent,
) -> Result<BaselineEvidence, String> {
    let (receipt, wrapper) = json_object(path, "MoE-combine CPU baseline wrapper")?;
    let seal = string(&wrapper, "seal_sha256", "MoE-combine CPU baseline wrapper")?;
    lower_sha256(seal, "MoE-combine CPU baseline wrapper seal")?;
    if string(&wrapper, "schema", "MoE-combine CPU baseline wrapper")? != CPU_BASELINE_SCHEMA
        || string(&wrapper, "status", "MoE-combine CPU baseline wrapper")? != CPU_BASELINE_STATUS
    {
        return Err("MoE-combine CPU baseline wrapper schema/status drifted".into());
    }
    let source = object(
        &wrapper,
        "source_binding",
        "MoE-combine CPU baseline wrapper",
    )?;
    evidence_matches(
        object(source, "manifest", "MoE-combine CPU baseline wrapper")?,
        &component.manifest,
        "MoE-combine CPU baseline manifest",
    )?;
    if string(
        source,
        "manifest_seal_sha256",
        "MoE-combine CPU baseline wrapper",
    )? != MANIFEST_SEAL
        || string(
            source,
            "admission_receipt_seal_sha256",
            "MoE-combine CPU baseline wrapper",
        )? != ADMISSION_RECEIPT_SEAL
    {
        return Err("MoE-combine CPU baseline source identity drifted".into());
    }
    let baseline_admission = object(
        source,
        "admission_current",
        "MoE-combine CPU baseline wrapper",
    )?;
    if json_path(
        string(
            baseline_admission,
            "path",
            "MoE-combine CPU baseline admission",
        )?,
        "MoE-combine CPU baseline admission path",
    )? != component.admission.path
    {
        return Err("MoE-combine CPU baseline admission path drifted".into());
    }
    lower_sha256(
        string(
            baseline_admission,
            "sha256",
            "MoE-combine CPU baseline admission",
        )?,
        "MoE-combine CPU baseline historical admission digest",
    )?;
    validate_source_top10_binding(
        object(
            source,
            "source_top10_binding",
            "MoE-combine CPU baseline wrapper",
        )?,
        component,
        "MoE-combine CPU baseline source top-10",
    )?;
    let inner_binding = object(
        &wrapper,
        "cpu_inner_receipt",
        "MoE-combine CPU baseline wrapper",
    )?;
    let inner_path = json_path(
        string(inner_binding, "path", "MoE-combine CPU baseline inner")?,
        "MoE-combine CPU baseline inner path",
    )?;
    let (inner_file, inner) = json_object(&inner_path, "MoE-combine CPU baseline inner")?;
    evidence_matches(
        inner_binding,
        &inner_file,
        "MoE-combine CPU baseline inner immutable evidence",
    )?;
    if string(&inner, "schema", "MoE-combine CPU baseline inner")? != RESULT_SCHEMA
        || string(&inner, "status", "MoE-combine CPU baseline inner")?
            != "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_CPU_ORACLE_READY_METAL_LEASE_REQUIRED"
        || string(&inner, "mode", "MoE-combine CPU baseline inner")? != "cpu-oracle"
    {
        return Err("MoE-combine CPU baseline inner schema/status/mode drifted".into());
    }
    for (field, expected) in [
        ("metal_device_or_dispatch_performed", false),
        ("component_only", true),
        ("routed_expert_aggregation_performed", true),
        ("shared_expert_add_performed", true),
        ("second_residual_performed", true),
        ("complete_layer_or_token_performed", false),
    ] {
        bool_field(&inner, field, expected, "MoE-combine CPU baseline inner")?;
    }
    if object(&inner, "durable_capture", "MoE-combine CPU baseline inner")?
        .get("receipt_written_last_is_completion_marker")
        .and_then(Value::as_bool)
        != Some(true)
    {
        return Err("MoE-combine CPU baseline inner lacks receipt-last durability".into());
    }
    validate_artifact_binding(
        object(&inner, "artifact_binding", "MoE-combine CPU baseline inner")?,
        component,
        "MoE-combine CPU baseline inner",
        false,
    )?;
    validate_source_top10_binding(
        object(
            &inner,
            "source_top10_binding",
            "MoE-combine CPU baseline inner",
        )?,
        component,
        "MoE-combine CPU baseline inner source top-10",
    )?;
    Ok(BaselineEvidence {
        receipt,
        seal_sha256: seal.to_owned(),
    })
}

fn validate_lease(
    path: &Path,
    component: &BoundComponent,
    baseline: &BaselineEvidence,
) -> Result<LeaseEvidence, String> {
    let (receipt, document) = json_object(path, "MoE-combine quiet Metal lease")?;
    let seal = string(&document, "seal_sha256", "MoE-combine quiet Metal lease")?;
    lower_sha256(seal, "MoE-combine quiet Metal lease seal")?;
    if string(&document, "schema", "MoE-combine quiet Metal lease")? != QUIET_LEASE_SCHEMA
        || string(&document, "status", "MoE-combine quiet Metal lease")? != QUIET_LEASE_STATUS
    {
        return Err("MoE-combine quiet Metal lease schema/status drifted".into());
    }
    let policy = object(
        &document,
        "execution_policy",
        "MoE-combine quiet Metal lease",
    )?;
    if string(policy, "component", "MoE-combine quiet Metal lease")? != QUIET_LEASE_COMPONENT
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
        return Err(
            "MoE-combine quiet Metal lease policy is not component-only strict Math".into(),
        );
    }
    let artifact = object(
        &document,
        "artifact_binding",
        "MoE-combine quiet Metal lease",
    )?;
    if string(
        artifact,
        "manifest_document_sha256",
        "MoE-combine quiet Metal lease",
    )? != component.manifest.sha256
        || string(
            artifact,
            "manifest_seal_sha256",
            "MoE-combine quiet Metal lease",
        )? != MANIFEST_SEAL
        || string(
            artifact,
            "admission_receipt_seal_sha256",
            "MoE-combine quiet Metal lease",
        )? != ADMISSION_RECEIPT_SEAL
    {
        return Err("MoE-combine quiet Metal lease artifact identity drifted".into());
    }
    validate_source_top10_binding(
        object(
            &document,
            "source_top10_binding",
            "MoE-combine quiet Metal lease",
        )?,
        component,
        "MoE-combine quiet Metal lease source top-10",
    )?;
    let baseline_binding = object(
        &document,
        "cpu_baseline_binding",
        "MoE-combine quiet Metal lease",
    )?;
    if json_path(
        string(
            baseline_binding,
            "receipt_path",
            "MoE-combine quiet Metal lease baseline",
        )?,
        "MoE-combine quiet Metal lease baseline path",
    )? != baseline.receipt.path
        || string(
            baseline_binding,
            "receipt_document_sha256",
            "MoE-combine quiet Metal lease baseline",
        )? != baseline.receipt.sha256
        || string(
            baseline_binding,
            "schema",
            "MoE-combine quiet Metal lease baseline",
        )? != CPU_BASELINE_SCHEMA
        || string(
            baseline_binding,
            "status",
            "MoE-combine quiet Metal lease baseline",
        )? != CPU_BASELINE_STATUS
        || string(
            baseline_binding,
            "seal_sha256",
            "MoE-combine quiet Metal lease baseline",
        )? != baseline.seal_sha256
    {
        return Err("MoE-combine quiet Metal lease baseline binding drifted".into());
    }
    Ok(LeaseEvidence {
        receipt,
        seal_sha256: seal.to_owned(),
    })
}

fn run_metal_stage(oracle: &CpuOracle) -> Result<DeviceLedger, String> {
    ensure_non_timed_environment()?;
    let flattened = oracle
        .input
        .routes
        .iter()
        .flat_map(|route| route.iter().copied())
        .collect::<Vec<_>>();
    if flattened.len() != TOP_K * HIDDEN {
        return Err("MoE-combine flattened route geometry drifted".into());
    }
    let context =
        MetalContext::new_with_trace_strict_math(false).map_err(|error| error.to_string())?;
    let device_name = context.device_name();
    let routes = context
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&flattened))
        .map_err(|error| error.to_string())?;
    let routed_sum = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "routed sum")?)
        .map_err(|error| error.to_string())?;
    let shared = context
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&oracle.input.gated_shared))
        .map_err(|error| error.to_string())?;
    let first_residual = context
        .new_buffer_with_bytes_checked(bytemuck::cast_slice(&oracle.input.first_residual))
        .map_err(|error| error.to_string())?;
    let second_residual = context
        .new_buffer_checked(bytes_for::<f32>(HIDDEN, "second residual")?)
        .map_err(|error| error.to_string())?;
    let mut command = TokenCommandBuffer::new(&context);
    command
        .dispatch_threads(
            "qwen80_moe_wave_aggregate_second_residual_route_sum",
            (HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&routes), 0);
                encoder.set_buffer(1, Some(&routed_sum), 0);
                encoder.stage_set_u32(2, TOP_K as u32);
                encoder.stage_set_u32(3, HIDDEN as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    command
        .dispatch_threads(
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            (HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&routed_sum), 0);
                encoder.set_buffer(1, Some(&shared), 0);
                encoder.set_buffer(2, Some(&first_residual), 0);
                encoder.set_buffer(3, Some(&second_residual), 0);
                encoder.stage_set_u32(4, HIDDEN as u32);
            },
        )
        .map_err(|error| error.to_string())?;
    let dispatches = command.dispatch_count();
    if dispatches != 2 {
        return Err(format!(
            "MoE-combine command buffer encoded {dispatches}, expected exactly 2 dispatches"
        ));
    }
    command
        .commit_and_wait()
        .map_err(|error| error.to_string())?;
    let observed_routed = snapshot_f32(&routed_sum, HIDDEN, "device routed sum")?;
    let observed_second = snapshot_f32(&second_residual, HIDDEN, "device second residual")?;
    let routed_sum_max_abs =
        max_f32_error(&oracle.routed_sum, &observed_routed, "device routed sum")?;
    let second_residual_max_abs = max_f32_error(
        &oracle.second_residual,
        &observed_second,
        "device second residual",
    )?;
    if routed_sum_max_abs > DEVICE_ROUTED_SUM_TOLERANCE
        || second_residual_max_abs > DEVICE_SECOND_RESIDUAL_TOLERANCE
    {
        return Err(format!(
            "MoE-combine strict-Math device parity exceeded tolerances: routed={routed_sum_max_abs}/{DEVICE_ROUTED_SUM_TOLERANCE}, second={second_residual_max_abs}/{DEVICE_SECOND_RESIDUAL_TOLERANCE}"
        ));
    }
    Ok(DeviceLedger {
        device_name,
        dispatches,
        routed_sum: observed_routed,
        second_residual: observed_second,
        routed_sum_max_abs,
        second_residual_max_abs,
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
        "status": "STARTED_QWEN80_MOE_COMBINE_COMPONENT_ATTEMPT",
        "started_unix_millis": started_unix_millis,
        "mode": mode_name(args.mode),
        "manifest": args.manifest,
        "admission_current": args.admission_current,
        "router_receipt": args.router_receipt,
        "router_outer_receipt": args.router_outer_receipt,
        "claim_boundary": {
            "materialized_source_route_shaped_fixture_only": true,
            "does_not_execute_ten_physical_expert_projections": true,
            "does_not_execute_a_complete_layer_token_decoder_hcli_or_tps": true,
            "metal_requires_a_sealed_cpu_baseline_and_component_quiet_lease": true,
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
    if target.exists() {
        return Err(format!(
            "refusing to overwrite capture evidence {}",
            target.display()
        ));
    }
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
            "cannot publish capture {} from {}: {error}",
            target.display(),
            temporary.display()
        ));
    }
    fs::remove_file(&temporary).map_err(|error| {
        format!(
            "cannot retire capture temporary {}: {error}",
            temporary.display()
        )
    })?;
    let parent = fs::File::open(capture_dir).map_err(|error| {
        format!(
            "cannot open capture directory {}: {error}",
            capture_dir.display()
        )
    })?;
    parent.sync_all().map_err(|error| {
        format!(
            "cannot fsync capture directory {}: {error}",
            capture_dir.display()
        )
    })
}

fn failure_result(args: &Args, error: &str) -> Value {
    json!({
        "schema": RESULT_SCHEMA,
        "status": "REFUSED_QWEN80_MOE_COMBINE_COMPONENT_ATTEMPT_ERROR",
        "mode": mode_name(args.mode),
        "error": error,
        "claim_boundary": {
            "no_cpu_or_metal_parity_is_claimed": true,
            "does_not_execute_ten_physical_expert_projections": true,
            "does_not_execute_a_complete_layer_token_decoder_hcli_or_tps": true,
        },
    })
}

fn finalize_capture(
    args: &Args,
    result: Result<Value, String>,
) -> Result<(Value, Option<String>), String> {
    let (mut document, failure) = match result {
        Ok(document) => (document, None),
        Err(error) => (failure_result(args, &error), Some(error)),
    };
    let object = document
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
    let rendered = serde_json::to_vec_pretty(&document).map_err(|error| error.to_string())?;
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
    Ok((document, failure))
}

fn input_rejections(router: &RouterEvidence) -> Value {
    let mut missing = materialized_fixture(router);
    missing.routes.pop();
    let missing_route_rejected = validate_input(&missing, router).is_err();
    let mut reordered = materialized_fixture(router);
    reordered.ids.swap(0, 1);
    reordered.weights.swap(0, 1);
    reordered.routes.swap(0, 1);
    let reordered_route_rejected = validate_input(&reordered, router).is_err();
    let mut unnormalized = materialized_fixture(router);
    unnormalized.weights[0] += 0.01;
    let unnormalized_route_rejected = validate_input(&unnormalized, router).is_err();
    let mut nonfinite = materialized_fixture(router);
    nonfinite.gated_shared[5] = f32::NAN;
    let nonfinite_rejected = validate_input(&nonfinite, router).is_err();
    let mut duplicate = materialized_fixture(router);
    duplicate.ids[1] = duplicate.ids[0];
    let duplicate_rejected = validate_input(&duplicate, router).is_err();
    json!({
        "missing_route_rejected": missing_route_rejected,
        "reordered_route_rejected": reordered_route_rejected,
        "unnormalized_route_rejected": unnormalized_route_rejected,
        "nonfinite_input_rejected": nonfinite_rejected,
        "duplicate_id_rejected": duplicate_rejected,
        "wrong_hidden_geometry_rejected": HIDDEN != 4_096,
    })
}

fn all_rejections_passed(rejections: &Value) -> bool {
    rejections
        .as_object()
        .is_some_and(|values| values.values().all(|value| value == &Value::Bool(true)))
}

fn artifact_binding(component: &BoundComponent) -> Value {
    json!({
        "manifest_path": component.manifest.path,
        "manifest_document_sha256": component.manifest.sha256,
        "manifest_seal_sha256": MANIFEST_SEAL,
        "admission_current_path": component.admission.path,
        "admission_pointer_seal_sha256": component.admission_pointer_seal_sha256,
        "admission_receipt_seal_sha256": ADMISSION_RECEIPT_SEAL,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_body_audit_seal_sha256": SOURCE_BODY_AUDIT_SEAL,
        "source_revalidation_seal_sha256": SOURCE_REVALIDATION_SEAL,
        "layer": 0,
        "layer_kind": "linear_attention",
        "hidden": HIDDEN,
        "experts": EXPERTS,
        "experts_per_token": TOP_K,
    })
}

fn cpu_result(
    args: &Args,
    component: &BoundComponent,
    oracle: &CpuOracle,
) -> Result<Value, String> {
    let rejections = input_rejections(&component.router);
    if !all_rejections_passed(&rejections) {
        return Err("MoE-combine CPU rejection suite did not fail closed".into());
    }
    let route_hashes = oracle
        .input
        .routes
        .iter()
        .map(|values| f32_sha256(values))
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_CPU_ORACLE_READY_METAL_LEASE_REQUIRED",
        "mode": "cpu-oracle",
        "component_only": true,
        "routed_expert_aggregation_performed": true,
        "shared_expert_add_performed": true,
        "second_residual_performed": true,
        "complete_layer_or_token_performed": false,
        "materialized_source_route_shaped_fixture_only": true,
        "complete_artifact_scan_performed": false,
        "direct_packed_payloads_opened": 0,
        "raw_bf16_or_safetensors_opened": false,
        "metal_device_or_dispatch_performed": false,
        "artifact_binding": artifact_binding(component),
        "source_top10_binding": source_top10_binding(component),
        "materialized_fixture": {
            "kind": "deterministic source-top10-shaped fixture; ten route vectors and shared vector are materialized boundary inputs, not outputs of ten physical experts",
            "route_index_order": (0..TOP_K).collect::<Vec<_>>(),
            "route_weights_f32": oracle.input.weights,
            "route_weights_source_f64": component.router.weights_f64,
            "route_weight_sum_f32": oracle.input.weights.iter().sum::<f32>(),
            "weighted_route_delta_shape": [TOP_K, HIDDEN],
            "weighted_route_delta_sha256": route_hashes,
            "gated_shared_sha256": f32_sha256(&oracle.input.gated_shared),
            "first_residual_sha256": f32_sha256(&oracle.input.first_residual),
        },
        "cpu_oracle": {
            "fixed_f32_order": "for each hidden index: start zero; add route[0] through route[9] in source selected order; add the supplied gated_shared; add the supplied first_residual",
            "routed_sum": {"elements": HIDDEN, "sha256": f32_sha256(&oracle.routed_sum)},
            "second_residual": {"elements": HIDDEN, "sha256": f32_sha256(&oracle.second_residual)},
            "routed_sum_f32_vs_f64_max_abs": oracle.routed_f32_f64_max_abs,
            "second_residual_f32_vs_f64_max_abs": oracle.second_f32_f64_max_abs,
            "f32_f64_tolerance": F32_F64_TOLERANCE,
            "all_outputs_finite": oracle.second_residual.iter().all(|value| value.is_finite()),
            "cpu_workers_requested": args.workers,
        },
        "rejection_tests": rejections,
        "metal_intermediate_error_ledger": {
            "performed": false,
            "reason": "CPU-only baseline; the staged direct host dispatcher requires a separate sealed component lease before creating a Metal context.",
            "future_kernel_sequence": [
                "qwen80_moe_wave_aggregate_second_residual_route_sum",
                "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            ],
            "future_acceptance": "same-capture fixed-order CPU vectors vs both fenced device outputs",
        },
        "integration_contract": {
            "scheduler_handoff": "The future layer scheduler must provide real route-index ordered weighted deltas and a real gated shared vector. This component checks only the combine boundary and cannot establish that ten physical expert projections ran.",
            "claim_boundary": "CPU fixture parity only; not complete routed MoE, a complete Qwen80 layer, token, decoder, generation, HCLI, TPS, TG, capability, Agent OS, or tournament evidence.",
        },
    }))
}

fn baseline_binding(baseline: &BaselineEvidence) -> Value {
    json!({
        "receipt_path": baseline.receipt.path,
        "receipt_document_sha256": baseline.receipt.sha256,
        "schema": CPU_BASELINE_SCHEMA,
        "status": CPU_BASELINE_STATUS,
        "seal_sha256": baseline.seal_sha256,
    })
}

fn lease_binding(lease: &LeaseEvidence) -> Value {
    json!({
        "receipt_path": lease.receipt.path,
        "receipt_document_sha256": lease.receipt.sha256,
        "schema": QUIET_LEASE_SCHEMA,
        "status": QUIET_LEASE_STATUS,
        "seal_sha256": lease.seal_sha256,
    })
}

fn metal_result(
    component: &BoundComponent,
    oracle: &CpuOracle,
    baseline: &BaselineEvidence,
    lease: &LeaseEvidence,
    device: &DeviceLedger,
) -> Result<Value, String> {
    let rejections = input_rejections(&component.router);
    if !all_rejections_passed(&rejections) {
        return Err("MoE-combine device rejection suite did not fail closed".into());
    }
    Ok(json!({
        "schema": RESULT_SCHEMA,
        "status": "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN",
        "mode": "metal",
        "component_only": true,
        "routed_expert_aggregation_performed": true,
        "shared_expert_add_performed": true,
        "second_residual_performed": true,
        "complete_layer_or_token_performed": false,
        "materialized_source_route_shaped_fixture_only": true,
        "complete_artifact_scan_performed": false,
        "direct_packed_payloads_opened": 0,
        "raw_bf16_or_safetensors_opened": false,
        "metal_device_or_dispatch_performed": true,
        "artifact_binding": artifact_binding(component),
        "source_top10_binding": source_top10_binding(component),
        "cpu_baseline_binding": baseline_binding(baseline),
        "same_capture_cpu_oracle": {
            "routed_sum": {"elements": HIDDEN, "sha256": f32_sha256(&oracle.routed_sum)},
            "second_residual": {"elements": HIDDEN, "sha256": f32_sha256(&oracle.second_residual)},
            "fixed_order": "route[0]..route[9], then gated_shared, then first_residual",
        },
        "metal_execution_policy": {
            "strict_math_required": true,
            "timing_or_benchmarking_allowed": false,
            "complete_layer_or_token_allowed": false,
            "tps_or_tg_claim_allowed": false,
            "lease_binding": lease_binding(lease),
        },
        "metal_intermediate_error_ledger": {
            "performed": true,
            "device": device.device_name,
            "command_buffers": 1,
            "compute_dispatches": device.dispatches,
            "kernel_sequence": [
                "qwen80_moe_wave_aggregate_second_residual_route_sum",
                "qwen80_moe_wave_aggregate_second_residual_add_shared_residual",
            ],
            "strict_math": true,
            "timing_or_benchmarking_performed": false,
            "acceptance": {
                "routed_sum_max_abs": device.routed_sum_max_abs,
                "routed_sum_tolerance": DEVICE_ROUTED_SUM_TOLERANCE,
                "second_residual_max_abs": device.second_residual_max_abs,
                "second_residual_tolerance": DEVICE_SECOND_RESIDUAL_TOLERANCE,
            },
            "device_intermediates": {
                "routed_sum": {"elements": HIDDEN, "sha256": f32_sha256(&device.routed_sum)},
                "second_residual": {"elements": HIDDEN, "sha256": f32_sha256(&device.second_residual)},
            },
        },
        "rejection_tests": rejections,
        "integration_contract": {
            "scheduler_handoff": "This is a materialized ten-route combine/second-residual boundary only. It does not prove any route input was produced by its physical Qwen80 expert or that this output came from a true preceding attention/DeltaNet layer state.",
            "claim_boundary": "Strict-Math device component parity only; not a complete Qwen80 layer, token, decoder, generation, HCLI, TPS, TG, capability, Agent OS, or tournament receipt.",
        },
    }))
}

fn run_component(args: &Args) -> Result<Value, String> {
    let component = bind_current(args)?;
    let oracle = build_cpu_oracle(&component)?;
    match args.mode {
        Mode::CpuOracle => cpu_result(args, &component, &oracle),
        Mode::Metal => {
            let baseline = validate_cpu_baseline(
                args.cpu_baseline_receipt
                    .as_deref()
                    .ok_or("Metal component requires a CPU baseline")?,
                &component,
            )?;
            let lease = validate_lease(
                args.lease_receipt
                    .as_deref()
                    .ok_or("Metal component requires a quiet lease")?,
                &component,
                &baseline,
            )?;
            let device = run_metal_stage(&oracle)?;
            metal_result(&component, &oracle, &baseline, &lease, &device)
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("ascension_qwen80_direct_packed_moe_combine: {error}");
            process::exit(2);
        }
    };
    if let Err(error) = begin_capture(&args) {
        eprintln!("ascension_qwen80_direct_packed_moe_combine: {error}");
        process::exit(2);
    }
    match finalize_capture(&args, run_component(&args)) {
        Ok((document, None)) => match serde_json::to_string_pretty(&document) {
            Ok(rendered) => println!("{rendered}"),
            Err(error) => {
                eprintln!(
                    "ascension_qwen80_direct_packed_moe_combine: output render failed: {error}"
                );
                process::exit(2);
            }
        },
        Ok((_document, Some(error))) => {
            eprintln!("ascension_qwen80_direct_packed_moe_combine: {error}");
            process::exit(2);
        }
        Err(error) => {
            eprintln!(
                "ascension_qwen80_direct_packed_moe_combine: capture finalization failed: {error}"
            );
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> RouterEvidence {
        RouterEvidence {
            inner: FileEvidence {
                path: PathBuf::from("/tmp/router-inner.json"),
                bytes: 1,
                sha256: "a".repeat(64),
            },
            outer: FileEvidence {
                path: PathBuf::from("/tmp/router-outer.json"),
                bytes: 1,
                sha256: "b".repeat(64),
            },
            outer_seal_sha256: "c".repeat(64),
            ids: EXPECTED_IDS,
            weights: [0.1; TOP_K],
            weights_f64: [0.1; TOP_K],
        }
    }

    #[test]
    fn fixed_source_order_then_shared_then_residual_is_exact_on_simple_fixture() {
        let router = router();
        let input = AggregateInput {
            ids: EXPECTED_IDS,
            weights: [0.1; TOP_K],
            routes: (0..TOP_K)
                .map(|index| vec![(index + 1) as f32; HIDDEN])
                .collect(),
            gated_shared: vec![0.5; HIDDEN],
            first_residual: vec![-0.25; HIDDEN],
        };
        validate_input(&input, &router).unwrap();
        let (routed, second) = aggregate_f32(&input).unwrap();
        assert_eq!(routed[0], 55.0);
        assert_eq!(second[0], 55.25);
    }

    #[test]
    fn rejects_missing_reordered_duplicate_unnormalized_and_nonfinite_inputs() {
        let router = router();
        let mut missing = materialized_fixture(&router);
        missing.routes.pop();
        assert!(validate_input(&missing, &router).is_err());
        let mut reordered = materialized_fixture(&router);
        reordered.ids.swap(0, 1);
        assert!(validate_input(&reordered, &router).is_err());
        let mut duplicate = materialized_fixture(&router);
        duplicate.ids[1] = duplicate.ids[0];
        assert!(validate_input(&duplicate, &router).is_err());
        let mut unnormalized = materialized_fixture(&router);
        unnormalized.weights[0] += 0.01;
        assert!(validate_input(&unnormalized, &router).is_err());
        let mut nonfinite = materialized_fixture(&router);
        nonfinite.first_residual[1] = f32::NAN;
        assert!(validate_input(&nonfinite, &router).is_err());
    }

    #[test]
    fn materialized_fixture_is_source_top10_shaped_and_fixed_order_matches_f64() {
        let component = BoundComponent {
            manifest: FileEvidence {
                path: PathBuf::from("/tmp/manifest.json"),
                bytes: 1,
                sha256: MANIFEST_DOCUMENT_SHA256.to_owned(),
            },
            admission: FileEvidence {
                path: PathBuf::from("/tmp/admission.json"),
                bytes: 1,
                sha256: "d".repeat(64),
            },
            admission_pointer_seal_sha256: "e".repeat(64),
            router: router(),
        };
        let oracle = build_cpu_oracle(&component).unwrap();
        assert_eq!(oracle.input.routes.len(), TOP_K);
        assert_eq!(oracle.routed_sum.len(), HIDDEN);
        assert_eq!(oracle.second_residual.len(), HIDDEN);
        assert!(oracle.routed_f32_f64_max_abs <= F32_F64_TOLERANCE);
        assert!(oracle.second_f32_f64_max_abs <= F32_F64_TOLERANCE);
    }

    #[test]
    fn host_shader_contract_is_exactly_two_component_dispatches() {
        let source = include_str!("../shaders/qwen80_moe_wave_aggregate_second_residual.metal");
        assert!(source.contains("kernel void qwen80_moe_wave_aggregate_second_residual_route_sum("));
        assert!(source.contains(
            "kernel void qwen80_moe_wave_aggregate_second_residual_add_shared_residual("
        ));
        assert!(source.contains("route_weighted_deltas[route * hidden + index]"));
        assert!(source.contains("value += gated_shared[index]"));
        assert!(source.contains("value += first_residual[index]"));
    }
}
