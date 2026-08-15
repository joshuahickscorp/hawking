//! CPU-only source-token L0 router discriminator and all-ten plan producer.
//!
//! The historical Qwen80 all-ten plan is bound to a deliberately synthetic
//! post-attention residual fixture.  This program never treats that route as
//! authority for the real source-token DeltaNet prefix.  It performs one
//! strict complete-binary admission scan, reuses the admitted catalog to
//! replay exactly token `1` with zeroed L0 DeltaNet state, checks the retained
//! prefix CPU baseline, derives the actual postnorm/router top-10, and emits
//! a new descriptor-only plan with all thirty exact compact payload bindings.
//!
//! It opens no source BF16/safetensors payload, creates no Metal context or
//! command buffer, starts no watcher/server, and makes no layer/token/HCLI/TPS
//! claim.  A Python outer wrapper must independently seal the create-new raw
//! material before any future device lease can use it.

use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CanonicalLinearLayerCpuInput, Qwen80CompleteArtifactCatalog,
};
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const MATERIAL_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_router_discriminator_material.v1";
const MATERIAL_STATUS: &str =
    "CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ROUTER_DISCRIMINATOR_MATERIAL_READY_FOR_SEAL";
const SOURCE_PLAN_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_routed_expert_binding_plan.v1";
const SOURCE_PLAN_STATUS: &str =
    "SOURCE_TOKEN_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_POINTER_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_POINTER_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const PREFIX_SCHEMA: &str = "hawking.ascension.qwen80_first_residual_outer_capture.v1";
const PREFIX_STATUS: &str = "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY";
const OLD_PLAN_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1";
const OLD_PLAN_STATUS: &str =
    "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
const SOURCE_TOKEN_ID: u32 = 1;
const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;
const GROUP_SIZE: usize = 128;

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    admission_current: PathBuf,
    first_residual_receipt: PathBuf,
    old_route_plan: PathBuf,
    out: PathBuf,
}

#[derive(Debug)]
struct PrefixAuthority {
    outer_evidence: Value,
    outer_seal_sha256: String,
    cpu_baseline_path: PathBuf,
    cpu_baseline_evidence: Value,
    source_input_sha256: String,
    cpu_first_residual_sha256: String,
    device_first_residual_sha256: String,
    zero_conv_state_sha256: String,
    zero_recurrent_state_sha256: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_l0_router_discriminator \\\n+--manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH \\\n+--first-residual-receipt ABSOLUTE_PATH --old-route-plan ABSOLUTE_PATH \\\n+--out ABSOLUTE_NEW_FILE"
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--manifest"
            | "--admission-current"
            | "--first-residual-receipt"
            | "--old-route-plan"
            | "--out" => {
                let value = iter
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} may not be repeated; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().to_owned()),
            _ => return Err(format!("unsupported argument {flag:?}; {}", usage())),
        }
    }
    let mut required = |flag: &str| -> Result<PathBuf, String> {
        let path = PathBuf::from(
            values
                .remove(flag)
                .ok_or_else(|| format!("missing {flag}; {}", usage()))?,
        );
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    let args = Args {
        manifest: required("--manifest")?,
        admission_current: required("--admission-current")?,
        first_residual_receipt: required("--first-residual-receipt")?,
        old_route_plan: required("--old-route-plan")?,
        out: required("--out")?,
    };
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    if args.out.exists() || !args.out.parent().is_some_and(Path::is_dir) {
        return Err("--out must be a new file beneath an existing parent".to_owned());
    }
    Ok(args)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
    {
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

fn read_json(path: &Path, label: &str) -> Result<(PathBuf, Vec<u8>, Value), String> {
    let path = canonical_regular(path, label)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value =
        serde_json::from_slice(&raw).map_err(|error| format!("cannot parse {label}: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} must be a JSON object"));
    }
    Ok((path, raw, value))
}

fn evidence(path: &Path, raw: &[u8]) -> Value {
    json!({
        "path": path,
        "present": true,
        "bytes": raw.len(),
        "sha256": sha256(raw),
    })
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn field_object<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    object
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn field_string(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{label}.{field} must be a string"))
}

fn field_bool(object: &Map<String, Value>, field: &str, label: &str) -> Result<bool, String> {
    object
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{field} must be a boolean"))
}

fn path_field(object: &Map<String, Value>, field: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(field_string(object, field, label)?);
    canonical_regular(&path, &format!("{label}.{field}"))
}

fn nested_object<'a>(
    value: &'a Value,
    fields: &[&str],
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    let mut current = object(value, label)?;
    for field in fields {
        current = field_object(current, field, label)?;
    }
    Ok(current)
}

fn nested_string(value: &Value, fields: &[&str], label: &str) -> Result<String, String> {
    let (last, parents) = fields
        .split_last()
        .ok_or_else(|| "empty nested field path".to_owned())?;
    field_string(nested_object(value, parents, label)?, last, label)
}

fn evidence_sha(object: &Map<String, Value>, field: &str, label: &str) -> Result<String, String> {
    let evidence = field_object(object, field, label)?;
    if evidence.get("present") != Some(&Value::Bool(true)) {
        return Err(format!("{label}.{field} must attest a present file"));
    }
    let digest = field_string(evidence, "sha256", &format!("{label}.{field}"))?;
    require_sha256(&digest, &format!("{label}.{field}.sha256"))?;
    Ok(digest)
}

fn bytes_to_f32le(bytes: &[u8], label: &str) -> Result<Vec<f32>, String> {
    if bytes.len() != HIDDEN * 4 {
        return Err(format!(
            "{label} has {} bytes, expected {}",
            bytes.len(),
            HIDDEN * 4
        ));
    }
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("{label} contains a non-finite f32"));
    }
    Ok(values)
}

fn f32le_sha256(values: &[f32]) -> Result<String, String> {
    if values.len() != HIDDEN || values.iter().any(|value| !value.is_finite()) {
        return Err("expected exactly 2048 finite f32 values".to_owned());
    }
    let mut bytes = Vec::with_capacity(HIDDEN * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(sha256(&bytes))
}

fn parse_prefix_authority(
    prefix_path: &Path,
    manifest_document_sha256: &str,
    manifest_seal_sha256: &str,
    admission_receipt_seal_sha256: &str,
) -> Result<PrefixAuthority, String> {
    let (prefix_path, prefix_raw, prefix) =
        read_json(prefix_path, "first-residual prefix receipt")?;
    let prefix_object = object(&prefix, "first-residual prefix receipt")?;
    if field_string(prefix_object, "schema", "first-residual prefix receipt")? != PREFIX_SCHEMA
        || field_string(prefix_object, "status", "first-residual prefix receipt")? != PREFIX_STATUS
    {
        return Err("first-residual prefix schema/status drifted".to_owned());
    }
    let outer_seal_sha256 = field_string(
        prefix_object,
        "seal_sha256",
        "first-residual prefix receipt",
    )?;
    require_sha256(&outer_seal_sha256, "first-residual prefix seal")?;
    let source = field_object(
        prefix_object,
        "source_binding",
        "first-residual prefix receipt",
    )?;
    if evidence_sha(source, "manifest", "first-residual prefix receipt")?
        != manifest_document_sha256
        || field_string(
            source,
            "manifest_seal_sha256",
            "first-residual prefix receipt",
        )? != manifest_seal_sha256
        || field_string(
            source,
            "admission_receipt_seal_sha256",
            "first-residual prefix receipt",
        )? != admission_receipt_seal_sha256
    {
        return Err("first-residual prefix artifact identity drifted".to_owned());
    }
    let output = field_object(
        prefix_object,
        "first_residual_output",
        "first-residual prefix receipt",
    )?;
    if output.get("layer") != Some(&json!(0))
        || output.get("linear_state_slot") != Some(&json!(0))
        || output.get("elements") != Some(&json!(HIDDEN))
        || output.get("same_command_graph_required") != Some(&Value::Bool(true))
    {
        return Err("first-residual prefix geometry/state identity drifted".to_owned());
    }
    let device_first_residual_sha256 = field_string(output, "sha256", "first-residual output")?;
    require_sha256(
        &device_first_residual_sha256,
        "first-residual strict-Metal output",
    )?;
    let baseline_evidence = field_object(
        source,
        "cpu_baseline_receipt",
        "first-residual prefix source_binding",
    )?;
    if baseline_evidence.get("present") != Some(&Value::Bool(true)) {
        return Err("first-residual prefix CPU baseline evidence is not present".to_owned());
    }
    let baseline_path = canonical_regular(
        Path::new(&field_string(
            baseline_evidence,
            "path",
            "first-residual prefix CPU baseline evidence",
        )?),
        "first-residual prefix CPU baseline",
    )?;
    let baseline_evidence_sha = field_string(
        baseline_evidence,
        "sha256",
        "first-residual prefix CPU baseline evidence",
    )?;
    require_sha256(
        &baseline_evidence_sha,
        "first-residual prefix CPU baseline evidence SHA",
    )?;
    let (baseline_path, baseline_raw, baseline) =
        read_json(&baseline_path, "first-residual CPU baseline")?;
    if sha256(&baseline_raw) != baseline_evidence_sha {
        return Err("first-residual prefix CPU baseline raw identity drifted".to_owned());
    }
    let baseline_object = object(&baseline, "first-residual CPU baseline")?;
    if field_string(baseline_object, "schema", "first-residual CPU baseline")?
        != "hawking.ascension.qwen80_first_residual_bridge_inner.v1"
        || field_string(baseline_object, "status", "first-residual CPU baseline")?
            != "EARNED_CURRENT_ADMITTED_QWEN80_LAYER0_FIRST_RESIDUAL_CPU_ORACLE_BASELINE_METAL_LEASE_REQUIRED"
    {
        return Err("first-residual CPU baseline schema/status drifted".to_owned());
    }
    let provenance = field_object(
        baseline_object,
        "same_input_provenance",
        "first-residual CPU baseline",
    )?;
    if field_string(provenance, "kind", "first-residual CPU baseline")?
        != "source_direct_packed_embedding_with_zeroed_layer0_deltanet_state"
        || provenance.get("token_id") != Some(&json!(SOURCE_TOKEN_ID))
    {
        return Err(
            "first-residual CPU baseline is not source token 1 with zeroed L0 state".to_owned(),
        );
    }
    let input = field_object(provenance, "input_hidden", "first-residual CPU baseline")?;
    let input_path = path_field(input, "path", "first-residual CPU baseline input hidden")?;
    let input_raw = fs::read(&input_path).map_err(|error| {
        format!(
            "cannot read retained source input {}: {error}",
            input_path.display()
        )
    })?;
    let source_input_sha256 = sha256(&input_raw);
    if source_input_sha256
        != field_string(input, "sha256", "first-residual CPU baseline input hidden")?
        || source_input_sha256
            != field_string(
                provenance,
                "input_hidden_f32le_sha256",
                "first-residual CPU baseline",
            )?
    {
        return Err("retained source input raw identity drifted".to_owned());
    }
    let _ = bytes_to_f32le(&input_raw, "retained source input")?;
    let baseline_output = field_object(
        baseline_object,
        "first_residual_output",
        "first-residual CPU baseline",
    )?;
    let cpu_first_residual_sha256 = field_string(
        baseline_output,
        "sha256",
        "first-residual CPU baseline output",
    )?;
    require_sha256(&cpu_first_residual_sha256, "first-residual CPU output")?;
    let conv = field_object(
        provenance,
        "initial_conv_state",
        "first-residual CPU baseline",
    )?;
    let recurrent = field_object(
        provenance,
        "initial_recurrent_state",
        "first-residual CPU baseline",
    )?;
    if conv.get("zero_initialized") != Some(&Value::Bool(true))
        || recurrent.get("zero_initialized") != Some(&Value::Bool(true))
    {
        return Err("first-residual CPU baseline state is not zero initialized".to_owned());
    }
    let zero_conv_state_sha256 =
        field_string(conv, "f32le_sha256", "first-residual CPU baseline conv")?;
    let zero_recurrent_state_sha256 = field_string(
        recurrent,
        "f32le_sha256",
        "first-residual CPU baseline recurrent",
    )?;
    require_sha256(&zero_conv_state_sha256, "zero conv state SHA")?;
    require_sha256(&zero_recurrent_state_sha256, "zero recurrent state SHA")?;
    Ok(PrefixAuthority {
        outer_evidence: evidence(&prefix_path, &prefix_raw),
        outer_seal_sha256,
        cpu_baseline_path: baseline_path.clone(),
        cpu_baseline_evidence: evidence(&baseline_path, &baseline_raw),
        source_input_sha256,
        cpu_first_residual_sha256,
        device_first_residual_sha256,
        zero_conv_state_sha256,
        zero_recurrent_state_sha256,
    })
}

fn old_route(plan: &Value) -> Result<(Vec<u16>, Vec<f64>, String), String> {
    let object = object(plan, "old route plan")?;
    if field_string(object, "schema", "old route plan")? != OLD_PLAN_SCHEMA
        || field_string(object, "status", "old route plan")? != OLD_PLAN_STATUS
    {
        return Err("old route plan schema/status drifted".to_owned());
    }
    let router = field_object(object, "router_evidence", "old route plan")?;
    let ids = router
        .get("source_stable_route_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "old route plan IDs missing".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u16::try_from(value).ok())
                .ok_or_else(|| "old route plan ID is not u16".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let weights = router
        .get("source_stable_normalized_weights")
        .and_then(Value::as_array)
        .ok_or_else(|| "old route plan weights missing".to_owned())?
        .iter()
        .map(|value| {
            value
                .as_f64()
                .ok_or_else(|| "old route plan weight is not f64".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if ids.len() != TOP_K
        || weights.len() != TOP_K
        || weights.iter().any(|weight| !weight.is_finite())
    {
        return Err("old route plan does not contain exactly ten finite route controls".to_owned());
    }
    let inner_path = PathBuf::from(field_string(
        router,
        "inner_receipt_document_sha256",
        "old route plan",
    )?);
    let _ = inner_path;
    Ok((ids, weights, "historical plan binds a post-attention synthetic fixture; it is not source-token authority".to_owned()))
}

fn tensor_descriptor(
    manifest: &Value,
    catalog: &Qwen80CompleteArtifactCatalog,
    name: &str,
    expected_shape: [usize; 2],
) -> Result<Value, String> {
    let tensors = object(manifest, "manifest")?
        .get("tensors")
        .and_then(Value::as_array)
        .ok_or_else(|| "manifest.tensors must be an array".to_owned())?;
    let descriptor = tensors
        .iter()
        .find(|row| row.get("tensor_name").and_then(Value::as_str) == Some(name))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("manifest lacks exact route tensor {name}"))?;
    let shape = descriptor
        .get("shape")
        .and_then(Value::as_array)
        .ok_or_else(|| format!("manifest route tensor {name} has no shape"))?
        .iter()
        .map(|value| value.as_u64().and_then(|value| usize::try_from(value).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| format!("manifest route tensor {name} shape is invalid"))?;
    if shape.as_slice() != expected_shape {
        return Err(format!(
            "manifest route tensor {name} shape {shape:?} drifted"
        ));
    }
    let header = catalog
        .direct_tensor_header(name)
        .map_err(|error| format!("admitted catalog lacks route tensor {name}: {error}"))?;
    if header.shape.as_slice() != expected_shape || header.group_size != GROUP_SIZE {
        return Err(format!(
            "admitted route tensor {name} header geometry drifted"
        ));
    }
    let descriptor_sha = field_string(descriptor, "artifact_sha256", "manifest route tensor")?;
    let catalog_sha = catalog
        .direct_tensor_artifact_sha256(name)
        .map_err(|error| format!("admitted route tensor {name} SHA unavailable: {error}"))?;
    if descriptor_sha != catalog_sha {
        return Err(format!("admitted route tensor {name} artifact SHA drifted"));
    }
    let layout = field_object(descriptor, "layout", "manifest route tensor")?;
    if field_string(layout, "magic", "manifest route layout")? != "HQ30G1B1"
        || layout.get("group_size") != Some(&json!(GROUP_SIZE))
        || field_string(layout, "scale_dtype", "manifest route layout")? != "float16"
        || field_string(layout, "sign_bit_order", "manifest route layout")? != "little"
        || layout.get("version") != Some(&json!(1))
    {
        return Err(format!(
            "admitted route tensor {name} packed layout drifted"
        ));
    }
    let required = [
        "tensor_name",
        "shape",
        "elements",
        "artifact_path",
        "artifact_bytes",
        "artifact_sha256",
        "source_dtype",
        "source_shard",
        "source_shard_sha256",
        "layout",
    ];
    let mut output = Map::new();
    for field in required {
        output.insert(
            field.to_owned(),
            descriptor
                .get(field)
                .ok_or_else(|| format!("manifest route tensor {name} lacks {field}"))?
                .clone(),
        );
    }
    output.insert("payload_opened_by_this_plan".to_owned(), Value::Bool(false));
    Ok(Value::Object(output))
}

fn write_new(path: &Path, contents: &[u8]) -> Result<(), String> {
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
    if let Err(error) = file.write_all(contents).and_then(|_| file.sync_all()) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("cannot write {}: {error}", temporary.display()));
    }
    drop(file);
    fs::hard_link(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        format!("cannot publish {}: {error}", path.display())
    })?;
    fs::remove_file(&temporary)
        .map_err(|error| format!("cannot retire {}: {error}", temporary.display()))
}

fn main_result(args: &Args) -> Result<Value, String> {
    let (manifest_path, manifest_raw, manifest) = read_json(&args.manifest, "manifest")?;
    let manifest_object = object(&manifest, "manifest")?;
    if field_string(manifest_object, "schema", "manifest")? != MANIFEST_SCHEMA {
        return Err("manifest schema drifted".to_owned());
    }
    let manifest_document_sha256 = sha256(&manifest_raw);
    let manifest_seal_sha256 = field_string(manifest_object, "seal_sha256", "manifest")?;
    require_sha256(&manifest_seal_sha256, "manifest seal")?;

    let (admission_path, admission_raw, admission) =
        read_json(&args.admission_current, "admission current")?;
    let admission_object = object(&admission, "admission current")?;
    if field_string(admission_object, "schema", "admission current")? != ADMISSION_POINTER_SCHEMA
        || field_string(admission_object, "status", "admission current")?
            != ADMISSION_POINTER_STATUS
    {
        return Err("admission-current schema/status drifted".to_owned());
    }
    let selected_manifest =
        field_object(admission_object, "complete_manifest", "admission current")?;
    if field_string(
        selected_manifest,
        "document_sha256",
        "admission current manifest",
    )? != manifest_document_sha256
        || field_string(
            selected_manifest,
            "seal_sha256",
            "admission current manifest",
        )? != manifest_seal_sha256
    {
        return Err("admission-current manifest identity drifted".to_owned());
    }
    let selected_receipt =
        field_object(admission_object, "admission_receipt", "admission current")?;
    let receipt_path = canonical_regular(
        Path::new(&field_string(
            selected_receipt,
            "path",
            "admission receipt",
        )?),
        "admission receipt",
    )?;
    let (receipt_path, receipt_raw, receipt) = read_json(&receipt_path, "admission receipt")?;
    let receipt_object = object(&receipt, "admission receipt")?;
    if field_string(receipt_object, "schema", "admission receipt")? != ADMISSION_RECEIPT_SCHEMA
        || field_string(receipt_object, "status", "admission receipt")? != ADMISSION_RECEIPT_STATUS
    {
        return Err("immutable admission receipt schema/status drifted".to_owned());
    }
    let admission_receipt_seal_sha256 =
        field_string(receipt_object, "seal_sha256", "admission receipt")?;
    if field_string(selected_receipt, "seal_sha256", "admission current receipt")?
        != admission_receipt_seal_sha256
    {
        return Err("admission-current immutable receipt seal drifted".to_owned());
    }
    let revalidation = field_object(
        receipt_object,
        "current_source_revalidation",
        "admission receipt",
    )?;
    let source_audit_seal_sha256 = field_string(
        revalidation,
        "source_audit_seal_sha256",
        "source revalidation",
    )?;
    let source_revision = field_string(revalidation, "revision", "source revalidation")?;
    require_sha256(&source_audit_seal_sha256, "source audit seal")?;

    let prefix = parse_prefix_authority(
        &args.first_residual_receipt,
        &manifest_document_sha256,
        &manifest_seal_sha256,
        &admission_receipt_seal_sha256,
    )?;
    let (old_plan_path, old_plan_raw, old_plan) =
        read_json(&args.old_route_plan, "old route plan")?;
    let (old_ids, old_weights, old_explanation) = old_route(&old_plan)?;

    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: source_audit_seal_sha256.clone(),
        expected_source_revision: source_revision.clone(),
    };
    // This is the one strict compact-artifact scan in the CPU discriminator.
    let catalog = Qwen80CompleteArtifactCatalog::load(&manifest_path, &admission)
        .map_err(|error| format!("strict Qwen80 artifact admission failed: {error}"))?;
    let input_bytes = fs::read(nested_string(
        &read_json(&prefix.cpu_baseline_path, "first-residual CPU baseline")?.2,
        &["same_input_provenance", "input_hidden", "path"],
        "first-residual CPU baseline",
    )?)
    .map_err(|error| format!("cannot reread retained source input: {error}"))?;
    let input_hidden = bytes_to_f32le(&input_bytes, "retained source input")?;
    if sha256(&input_bytes) != prefix.source_input_sha256 {
        return Err("retained source input changed after prefix authority validation".to_owned());
    }
    let embedding = catalog
        .execute_embedding_lookup_cpu_oracle(SOURCE_TOKEN_ID)
        .map_err(|error| format!("direct-packed source token embedding lookup refused: {error}"))?;
    if f32le_sha256(&embedding.hidden)? != prefix.source_input_sha256
        || embedding.hidden != input_hidden
    {
        return Err(
            "retained prefix input does not equal admitted source token 1 embedding".to_owned(),
        );
    }
    let input = Qwen80CanonicalLinearLayerCpuInput::with_zero_state(input_hidden);
    let cpu = catalog
        .execute_first_linear_layer_cpu_moe_oracle(&input)
        .map_err(|error| {
            format!("same-source-token L0 direct-packed CPU oracle refused: {error}")
        })?;
    let cpu_first_residual_sha256 = f32le_sha256(&cpu.mixer.mixer_residual_output)?;
    if cpu_first_residual_sha256 != prefix.cpu_first_residual_sha256 {
        return Err(
            "same-source-token CPU DeltaNet first residual does not match retained prefix baseline"
                .to_owned(),
        );
    }
    let normalized_hidden_sha256 = f32le_sha256(&cpu.post_attention_rms_norm_output)?;
    let router_logits_sha256 = {
        if cpu.router_logits.len() != 512
            || cpu.router_logits.iter().any(|value| !value.is_finite())
        {
            return Err("same-source-token router logits are not 512 finite f32 values".to_owned());
        }
        let mut raw = Vec::with_capacity(cpu.router_logits.len() * 4);
        for value in &cpu.router_logits {
            raw.extend_from_slice(&value.to_le_bytes());
        }
        sha256(&raw)
    };
    if cpu.route.ids.len() != TOP_K || cpu.route.weights.len() != TOP_K {
        return Err("same-source-token router does not contain top-10".to_owned());
    }
    let mut seen = BTreeSet::new();
    if cpu.route.ids.iter().any(|id| !seen.insert(*id))
        || cpu
            .route
            .weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("same-source-token router has duplicate IDs or invalid weights".to_owned());
    }
    let weight_sum = cpu.route.weights.iter().sum::<f32>();
    if (weight_sum - 1.0).abs() > 2.0e-6 {
        return Err(format!(
            "same-source-token router weights sum to {weight_sum}, not one"
        ));
    }

    let mut unique_artifacts = BTreeSet::new();
    let mut waves = Vec::with_capacity(TOP_K);
    for (index, (&expert, &weight)) in cpu
        .route
        .ids
        .iter()
        .zip(cpu.route.weights.iter())
        .enumerate()
    {
        let prefix_name = format!("model.layers.0.mlp.experts.{expert}");
        let gate = tensor_descriptor(
            &manifest,
            &catalog,
            &format!("{prefix_name}.gate_proj.weight"),
            [512, 2048],
        )?;
        let up = tensor_descriptor(
            &manifest,
            &catalog,
            &format!("{prefix_name}.up_proj.weight"),
            [512, 2048],
        )?;
        let down = tensor_descriptor(
            &manifest,
            &catalog,
            &format!("{prefix_name}.down_proj.weight"),
            [2048, 512],
        )?;
        for projection in [&gate, &up, &down] {
            let artifact = projection
                .get("artifact_sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| "route tensor has no artifact SHA".to_owned())?;
            if !unique_artifacts.insert(artifact.to_owned()) {
                return Err(
                    "same-source-token plan reuses an artifact across 30 route projections"
                        .to_owned(),
                );
            }
        }
        waves.push(json!({
            "wave_index": index,
            "layer": 0,
            "expert_id": expert,
            "normalized_weight": weight,
            "normalized_weight_bits_hex": format!("0x{:016x}", f64::from(weight).to_bits()),
            "gate": gate,
            "up": up,
            "down": down,
            "fixed_operation_order": [
                "gate_proj [512,2048]",
                "up_proj [512,2048]",
                "SiLU(gate) * up [512]",
                "down_proj [2048,512]",
                "apply this route's source-normalized weight [2048]",
            ],
            "route_execution_status": "NOT_EXECUTED_SOURCE_TOKEN_BOUND_PLAN_ONLY",
            "route_delta_materialized": false,
            "route_weight_applied": false,
        }));
    }
    if waves.len() != TOP_K || unique_artifacts.len() != TOP_K * 3 {
        return Err(
            "same-source-token plan does not bind exactly 30 unique route payloads".to_owned(),
        );
    }
    let new_ids = cpu.route.ids.iter().copied().collect::<Vec<_>>();
    let new_weights = cpu
        .route
        .weights
        .iter()
        .map(|weight| f64::from(*weight))
        .collect::<Vec<_>>();
    let differences = old_ids
        .iter()
        .zip(&new_ids)
        .enumerate()
        .filter_map(|(index, (&old, &new))| {
            (old != new).then_some(
                json!({"wave_index": index, "old_expert_id": old, "source_token_expert_id": new}),
            )
        })
        .collect::<Vec<_>>();
    let weight_max_abs_difference = old_weights
        .iter()
        .zip(&new_weights)
        .map(|(old, new)| (old - new).abs())
        .fold(0.0_f64, f64::max);

    Ok(json!({
        "schema": MATERIAL_SCHEMA,
        "status": MATERIAL_STATUS,
        "source_binding": {
            "manifest": evidence(&manifest_path, &manifest_raw),
            "admission_current": evidence(&admission_path, &admission_raw),
            "admission_receipt": evidence(&receipt_path, &receipt_raw),
            "first_residual_outer_receipt": prefix.outer_evidence.clone(),
            "historical_fixture_route_plan": evidence(&old_plan_path, &old_plan_raw),
            "manifest_seal_sha256": manifest_seal_sha256.clone(),
            "admission_receipt_seal_sha256": admission_receipt_seal_sha256.clone(),
            "source_audit_seal_sha256": source_audit_seal_sha256.clone(),
            "source_revision": source_revision.clone(),
        },
        "source_token_plan": {
            "schema": SOURCE_PLAN_SCHEMA,
            "status": SOURCE_PLAN_STATUS,
            "model_id": "Qwen3-Coder-Next-80B",
            "model_key": "qwen80",
            "source_repository": "Qwen/Qwen3-Coder-Next",
            "source_revision": source_revision,
            "layer": 0,
            "source_input_provenance": {
                "source_token_id": SOURCE_TOKEN_ID,
                "embedding_tensor": embedding.direct_packed_embedding_tensor,
                "input_hidden_f32le_sha256": prefix.source_input_sha256,
                "cpu_first_residual_f32le_sha256": prefix.cpu_first_residual_sha256,
                "strict_metal_prefix_first_residual_sha256": prefix.device_first_residual_sha256,
                "zero_conv_state_f32le_sha256": prefix.zero_conv_state_sha256,
                "zero_recurrent_state_f32le_sha256": prefix.zero_recurrent_state_sha256,
                "cpu_baseline_receipt": prefix.cpu_baseline_evidence,
                "prefix_outer_receipt": prefix.outer_evidence,
                "prefix_outer_receipt_seal_sha256": prefix.outer_seal_sha256,
                "same_input_state_identity_required": true,
            },
            "source_token_router_evidence": {
                "derived_from_direct_packed_source_token_l0_cpu_oracle": true,
                "post_attention_normalized_hidden_f32le_sha256": normalized_hidden_sha256,
                "router_logits_f32le_sha256": router_logits_sha256,
                "source_stable_route_ids": new_ids,
                "source_stable_normalized_weights": new_weights,
                "weights_sum": weight_sum,
                "router_component_only": true,
            },
            "manifest_descriptor_inventory": {
                "inventory_document_sha256": manifest_document_sha256,
                "manifest_schema": MANIFEST_SCHEMA,
                "manifest_seal_sha256": manifest_seal_sha256,
                "declared_tensor_count": catalog.tensor_count(),
                "resolved_route_tensor_count": unique_artifacts.len(),
                "payload_opened_by_this_plan": false,
            },
            "deterministic_waves": waves,
            "rawls_real_all_ten_provenance_gate": {
                "schema": "hawking.ascension.qwen80_real_all_ten_routed_expert_provenance_gate_input.v1",
                "all_ten_source_bindings_complete": true,
                "expected_layer": 0,
                "deterministic_wave_indices": (0..TOP_K).collect::<Vec<_>>(),
                "route_order": cpu.route.ids,
                "normalized_weights": cpu.route.weights,
                "execution_receipt_required_for_each_wave": true,
                "direct_packed_execution_required_for_each_wave": true,
                "source_bound_input_required_for_each_wave": true,
                "route_combine_receipt_required_separately": true,
                "shared_expert_receipt_required_separately": true,
                "first_and_second_residual_receipts_required_separately": true,
                "rejects_tensor_substitution": true,
                "rejects_route_reorder": true,
                "rejects_duplicate_experts": true,
                "rejects_missing_tensor_or_weight": true,
            },
            "route_execution_performed": false,
            "route_combine_performed": false,
            "shared_expert_performed": false,
            "residual_combine_performed": false,
            "metal_device_or_dispatch_performed": false,
            "model_execution_performed": false,
            "hcli_execution_performed": false,
            "tps_or_tg_measurement_performed": false,
            "complete_layer_or_decoder_claim_earned": false,
        },
        "fixture_divergence": {
            "old_route_plan": evidence(&old_plan_path, &old_plan_raw),
            "old_route_ids": old_ids,
            "old_route_weights": old_weights,
            "old_plan_explanation": old_explanation,
            "same_route_ids": differences.is_empty(),
            "different_route_slots": differences,
            "maximum_normalized_weight_abs_difference": weight_max_abs_difference,
            "conclusion": "the fixture-derived route plan is prohibited from driving the source-token true-MoE graph",
        },
        "artifact_scan": {
            "complete_artifact_admission_performed_once": true,
            "catalog_reused_for_embedding_mixer_router_and_all_thirty_descriptors": true,
            "raw_bf16_or_safetensors_opened": false,
        },
        "claim_boundary": {
            "cpu_discriminator_and_descriptor_plan_only": true,
            "metal_device_or_dispatch_performed": false,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
            "requires_a_new_sealed_source_token_plan_bridge_outer_preflight_and_fresh_component_lease_before_device_work": true,
        },
    }))
}

fn main() {
    let args = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("Qwen80 source-token route discriminator argument refusal: {error}");
            std::process::exit(2);
        }
    };
    match main_result(&args) {
        Ok(material) => match serde_json::to_vec_pretty(&material) {
            Ok(mut raw) => {
                raw.push(b'\n');
                if let Err(error) = write_new(&args.out, &raw) {
                    eprintln!("Qwen80 source-token route discriminator write refusal: {error}");
                    std::process::exit(2);
                }
                println!(
                    "{}",
                    serde_json::to_string(&json!({"out": args.out, "sha256": sha256(&raw)}))
                        .unwrap()
                );
            }
            Err(error) => {
                eprintln!("Qwen80 source-token route discriminator rendering refusal: {error}");
                std::process::exit(2);
            }
        },
        Err(error) => {
            eprintln!("Qwen80 source-token route discriminator refusal: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_token_plan_uses_a_distinct_schema_from_the_fixture_plan() {
        assert_ne!(SOURCE_PLAN_SCHEMA, OLD_PLAN_SCHEMA);
        assert!(SOURCE_PLAN_STATUS.contains("SOURCE_TOKEN_BOUND"));
        assert_eq!(SOURCE_TOKEN_ID, 1);
    }

    #[test]
    fn f32le_parser_refuses_wrong_geometry_and_nonfinite_values() {
        assert!(bytes_to_f32le(&[0; 12], "wrong").is_err());
        let mut bytes = vec![0u8; HIDDEN * 4];
        bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(bytes_to_f32le(&bytes, "nan").is_err());
    }

    #[test]
    fn f32le_hash_is_stable_for_source_shape() {
        let values = vec![0.0f32; HIDDEN];
        assert_eq!(
            f32le_sha256(&values).unwrap(),
            "9f1dcbc35c350d6027f98be0f5c8b43b42ca52b7604459c0c42be3aa88913d47"
        );
    }
}
