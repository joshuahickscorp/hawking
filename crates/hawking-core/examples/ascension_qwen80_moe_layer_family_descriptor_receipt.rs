//! Source-bound, CPU-only Qwen3-Coder-Next MoE layer-family descriptor receipt.
//!
//! This program joins the permanent 48-layer payload/schedule authority to the
//! sealed descriptor inventory and source-config authority.  It resolves, for
//! every released layer, the post-attention norm, router, all 512 routed
//! gate/up/down projections, and the four shared-expert projections.  The
//! result is a compact descriptor receipt for a future *generic* same-input
//! MoE capture; it does not open any compact payload, source shard, or BF16
//! file.
//!
//! The receipt deliberately contains no selected route IDs or route payloads.
//! Those are future same-input results.  Instead it fixes the source-stable
//! top-10 policy and the only permitted f32 combine order:
//!
//! ```text
//! route[0] .. route[9] (source-selected order) -> gated shared -> first residual
//! ```
//!
//! It rejects a missing descriptor, a schedule whose canonical manifest/config
//! bindings disagree with the supplied authorities, a fixture/synthetic
//! authority, or any non-Qwen80 MoE geometry.  It creates no Metal context,
//! lease, registry entry, runtime, watcher, server, HCLI request, model token,
//! or TPS/TG measurement.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_moe_layer_family_descriptor_receipt -- \
//!   --schedule-authority /absolute/path/QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY.json \
//!   --descriptor-inventory /absolute/path/QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json \
//!   --source-config-authority /absolute/path/QWEN80_SOURCE_METADATA_CANDIDATE.json \
//!   --out /absolute/new/QWEN80_MOE_LAYER_FAMILY_DESCRIPTOR_RECEIPT.json
//! ```

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const RESULT_SCHEMA: &str = "hawking.ascension.qwen80_moe_layer_family_descriptor_receipt.v1";
const RESULT_STATUS: &str = "PREPARED_QWEN80_MOE_LAYER_FAMILY_DESCRIPTOR_RECEIPT_NOT_EXECUTED";
const EXECUTION_STATUS: &str = "PREPARED_NOT_EXECUTED";

const SCHEDULE_SCHEMA: &str = "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1";
const SCHEDULE_STATUS: &str = "PREPARED_QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY_NOT_EXECUTED";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const MANIFEST_STATUS: &str = "CANDIDATE_COMPLETE_BINARY_ARTIFACT_LOW_FIDELITY_UNQUALIFIED";
const CONFIG_SCHEMA: &str = "hawking.ascension.source_admission_candidate.v1";
const CONFIG_STATUS: &str = "CANDIDATE_METADATA_CAPTURED";

const MODEL_ID: &str = "Qwen3-Coder-Next-80B";
const MODEL_KEY: &str = "qwen80";
const SOURCE_REPOSITORY: &str = "Qwen/Qwen3-Coder-Next";
const SOURCE_REVISION: &str = "a7fbcb5c0e12d62a448eaa0e260346bf5dcc0feb";
const SOURCE_CONFIG_SHA256: &str =
    "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8";

const LAYERS: usize = 48;
const HIDDEN: usize = 2_048;
const EXPERTS: usize = 512;
const TOP_K: usize = 10;
const INTERMEDIATE: usize = 512;
const GROUP_SIZE: usize = 128;
const RMS_NORM_EPSILON: f64 = 1.0e-6;
const COMPLETE_TENSOR_COUNT: usize = 74_391;
const MOE_DESCRIPTOR_COUNT_PER_LAYER: usize = 2 + EXPERTS * 3 + 4;
const MOE_DESCRIPTOR_COUNT: usize = LAYERS * MOE_DESCRIPTOR_COUNT_PER_LAYER;

#[derive(Debug)]
struct Args {
    schedule_authority: PathBuf,
    descriptor_inventory: PathBuf,
    source_config_authority: PathBuf,
    out: PathBuf,
}

#[derive(Debug)]
struct Document {
    canonical_path: PathBuf,
    raw_sha256: String,
    value: Value,
    seal_sha256: Option<String>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_moe_layer_family_descriptor_receipt \\
--schedule-authority ABSOLUTE_RAW_SCHEDULE_JSON \\
--descriptor-inventory ABSOLUTE_SEALED_MANIFEST_JSON \\
--source-config-authority ABSOLUTE_SEALED_SOURCE_CONFIG_JSON \\
--out ABSOLUTE_NEW_RECEIPT_JSON"
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_json(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|error| format!("cannot serialize canonical JSON: {error}"))
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !is_lower_sha256(value) {
        return Err(format!("{label} is not a lowercase SHA-256"));
    }
    Ok(())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be a JSON object"))
}

fn field_object<'a>(
    value: &'a Value,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} lacks object {field:?}"))
}

fn field_array<'a>(value: &'a Value, field: &str, label: &str) -> Result<&'a [Value], String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} lacks array {field:?}"))
}

fn field_string<'a>(value: &'a Value, field: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} lacks non-empty string {field:?}"))
}

fn field_bool(value: &Value, field: &str, label: &str) -> Result<bool, String> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} lacks boolean {field:?}"))
}

fn field_usize(value: &Value, field: &str, label: &str) -> Result<usize, String> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| format!("{label} lacks usize {field:?}"))
}

fn field_shape(value: &Value, field: &str, label: &str) -> Result<Vec<usize>, String> {
    field_array(value, field, label)?
        .iter()
        .enumerate()
        .map(|(index, dimension)| {
            dimension
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| format!("{label} has invalid {field}[{index}]"))
        })
        .collect()
}

fn require_string(value: &Value, field: &str, expected: &str, label: &str) -> Result<(), String> {
    let actual = field_string(value, field, label)?;
    if actual != expected {
        return Err(format!(
            "{label} {field:?}={actual:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn require_bool(value: &Value, field: &str, expected: bool, label: &str) -> Result<(), String> {
    let actual = field_bool(value, field, label)?;
    if actual != expected {
        return Err(format!(
            "{label} {field:?}={actual:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn checked_elements(shape: &[usize], label: &str) -> Result<usize, String> {
    if shape.is_empty() {
        return Err(format!("{label} cannot have an empty shape"));
    }
    shape.iter().try_fold(1usize, |total, dimension| {
        total
            .checked_mul(*dimension)
            .ok_or_else(|| format!("{label} element count overflow"))
    })
}

fn canonical_regular_file(path: &Path, label: &str) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be an absolute path"));
    }
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| format!("cannot stat {label} {}: {error}", path.display()))?;
    if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
        return Err(format!(
            "{label} must be a regular non-symlink file: {}",
            path.display()
        ));
    }
    path.canonicalize()
        .map_err(|error| format!("cannot canonicalize {label} {}: {error}", path.display()))
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let object = object(value, label)?;
    let observed = object
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} lacks seal_sha256"))?;
    require_sha256(observed, &format!("{label} seal"))?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_json(&Value::Object(unsigned))?;
    if expected != observed {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed.to_owned())
}

fn read_document(path: &Path, label: &str, must_be_sealed: bool) -> Result<Document, String> {
    let canonical_path = canonical_regular_file(path, label)?;
    let bytes = fs::read(&canonical_path)
        .map_err(|error| format!("cannot read {label} {}: {error}", canonical_path.display()))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {label} JSON: {error}"))?;
    object(&value, label)?;
    let seal_sha256 = if must_be_sealed {
        Some(verify_seal(&value, label)?)
    } else {
        if value.get("seal_sha256").is_some() {
            return Err(format!(
                "{label} must be the raw, unsealed permanent schedule authority"
            ));
        }
        None
    };
    Ok(Document {
        canonical_path,
        raw_sha256: sha256_hex(&bytes),
        value,
        seal_sha256,
    })
}

fn parse_args(arguments: &[String]) -> Result<Args, String> {
    let mut schedule_authority = None;
    let mut descriptor_inventory = None;
    let mut source_config_authority = None;
    let mut out = None;
    let mut index = 0usize;
    while index < arguments.len() {
        let flag = &arguments[index];
        let value = arguments
            .get(index + 1)
            .ok_or_else(|| format!("{flag} requires an absolute path; {}", usage()))?;
        let path = PathBuf::from(value);
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute; {}", usage()));
        }
        let slot = match flag.as_str() {
            "--schedule-authority" => &mut schedule_authority,
            "--descriptor-inventory" => &mut descriptor_inventory,
            "--source-config-authority" => &mut source_config_authority,
            "--out" => &mut out,
            _ => return Err(format!("unknown argument {flag}; {}", usage())),
        };
        if slot.replace(path).is_some() {
            return Err(format!("argument {flag} was repeated; {}", usage()));
        }
        index += 2;
    }
    let required = |value: Option<PathBuf>, label: &str| {
        value.ok_or_else(|| format!("missing {label}; {}", usage()))
    };
    let out = required(out, "--out")?;
    let parent = out
        .parent()
        .filter(|parent| parent.is_dir())
        .ok_or("--out parent directory must already exist")?;
    if !parent.is_absolute() || out.exists() {
        return Err(
            "--out must be a new absolute path below an existing absolute directory".into(),
        );
    }
    Ok(Args {
        schedule_authority: required(schedule_authority, "--schedule-authority")?,
        descriptor_inventory: required(descriptor_inventory, "--descriptor-inventory")?,
        source_config_authority: required(source_config_authority, "--source-config-authority")?,
        out,
    })
}

fn validate_source_config(config: &Document) -> Result<(), String> {
    let value = &config.value;
    require_string(value, "schema", CONFIG_SCHEMA, "source-config authority")?;
    require_string(value, "status", CONFIG_STATUS, "source-config authority")?;
    let source = Value::Object(field_object(value, "source", "source-config authority")?.clone());
    require_string(
        &source,
        "repository",
        SOURCE_REPOSITORY,
        "source-config authority source",
    )?;
    require_string(
        &source,
        "revision",
        SOURCE_REVISION,
        "source-config authority source",
    )?;
    let architecture =
        Value::Object(field_object(value, "architecture", "source-config authority")?.clone());
    require_bool(
        &architecture,
        "config_captured",
        true,
        "source-config authority architecture",
    )?;
    require_string(
        &architecture,
        "config_sha256",
        SOURCE_CONFIG_SHA256,
        "source-config authority architecture",
    )?;
    require_string(
        &architecture,
        "model_type",
        "qwen3_next",
        "source-config authority architecture",
    )?;
    if field_usize(
        &architecture,
        "hidden_size",
        "source-config authority architecture",
    )? != HIDDEN
        || field_usize(
            &architecture,
            "num_experts",
            "source-config authority architecture",
        )? != EXPERTS
        || field_usize(
            &architecture,
            "num_experts_per_tok",
            "source-config authority architecture",
        )? != TOP_K
        || field_usize(
            &architecture,
            "num_hidden_layers",
            "source-config authority architecture",
        )? != LAYERS
        || field_usize(
            &architecture,
            "vocab_size",
            "source-config authority architecture",
        )? != 151_936
    {
        return Err("source-config authority Qwen80 MoE geometry drifted".into());
    }
    let architectures = field_array(
        &architecture,
        "architectures",
        "source-config authority architecture",
    )?;
    if !architectures
        .iter()
        .any(|value| value.as_str() == Some("Qwen3NextForCausalLM"))
    {
        return Err("source-config authority lacks Qwen3NextForCausalLM architecture".into());
    }
    Ok(())
}

fn build_manifest_index<'a>(
    manifest: &'a Document,
) -> Result<BTreeMap<&'a str, (usize, &'a Value)>, String> {
    let value = &manifest.value;
    require_string(value, "schema", MANIFEST_SCHEMA, "descriptor inventory")?;
    require_string(value, "status", MANIFEST_STATUS, "descriptor inventory")?;
    let source = Value::Object(field_object(value, "source", "descriptor inventory")?.clone());
    require_string(
        &source,
        "repository",
        SOURCE_REPOSITORY,
        "descriptor inventory source",
    )?;
    if field_usize(&source, "tensor_count", "descriptor inventory source")? != COMPLETE_TENSOR_COUNT
    {
        return Err("descriptor inventory source tensor count drifted".into());
    }
    let tensors = field_array(value, "tensors", "descriptor inventory")?;
    if tensors.len() != COMPLETE_TENSOR_COUNT {
        return Err(format!(
            "descriptor inventory has {} tensors, expected {COMPLETE_TENSOR_COUNT}",
            tensors.len()
        ));
    }
    let mut index = BTreeMap::new();
    for (ordinal, tensor) in tensors.iter().enumerate() {
        let name = field_string(tensor, "tensor_name", "descriptor inventory tensor")?;
        if index.insert(name, (ordinal, tensor)).is_some() {
            return Err(format!("descriptor inventory duplicates tensor {name:?}"));
        }
    }
    if index.len() != tensors.len() {
        return Err("descriptor inventory name index is incomplete".into());
    }
    Ok(index)
}

fn no_fixture_or_synthetic_authority(value: &Value, label: &str) -> Result<(), String> {
    match value {
        Value::Object(fields) => {
            for (key, child) in fields {
                let lower = key.to_ascii_lowercase();
                if (lower.contains("fixture") || lower.contains("synthetic"))
                    && child.as_bool() == Some(true)
                {
                    return Err(format!(
                        "{label} accepts fixture or synthetic authority via {key:?}"
                    ));
                }
                if (lower.contains("fixture") || lower.contains("synthetic"))
                    && child
                        .as_str()
                        .is_some_and(|value| value.to_ascii_lowercase().contains("fixture"))
                {
                    return Err(format!("{label} accepts fixture authority via {key:?}"));
                }
                no_fixture_or_synthetic_authority(child, label)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                no_fixture_or_synthetic_authority(child, label)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn expected_mixer(layer: usize) -> &'static str {
    if layer % 4 == 3 {
        "gqa"
    } else {
        "delta_net"
    }
}

fn expected_moe_suffix() -> [&'static str; 9] {
    [
        "post_attention_rmsnorm",
        "router_gate",
        "source_top10_route",
        "all_ten_routed_expert_gate_up_down",
        "routed_expert_weighted_combine",
        "shared_expert_gate_up_down",
        "shared_expert_scalar_gate",
        "moe_combine",
        "second_residual",
    ]
}

fn validate_schedule_header(
    schedule: &Document,
    manifest: &Document,
    config: &Document,
) -> Result<(), String> {
    let value = &schedule.value;
    require_string(
        value,
        "schema",
        SCHEDULE_SCHEMA,
        "permanent schedule authority",
    )?;
    require_string(
        value,
        "status",
        SCHEDULE_STATUS,
        "permanent schedule authority",
    )?;
    no_fixture_or_synthetic_authority(value, "permanent schedule authority")?;
    let source = Value::Object(
        field_object(value, "source_authority", "permanent schedule authority")?.clone(),
    );
    require_string(&source, "model_id", MODEL_ID, "schedule source authority")?;
    require_string(&source, "model_key", MODEL_KEY, "schedule source authority")?;
    require_string(
        &source,
        "source_repository",
        SOURCE_REPOSITORY,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "source_revision",
        SOURCE_REVISION,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "source_config_sha256",
        SOURCE_CONFIG_SHA256,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "descriptor_inventory_canonical_path",
        &manifest.canonical_path.display().to_string(),
        "schedule source authority",
    )?;
    require_string(
        &source,
        "descriptor_inventory_document_sha256",
        &manifest.raw_sha256,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "descriptor_inventory_schema",
        MANIFEST_SCHEMA,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "descriptor_inventory_status",
        MANIFEST_STATUS,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "descriptor_inventory_seal_sha256",
        manifest
            .seal_sha256
            .as_deref()
            .ok_or("sealed descriptor inventory has no parsed seal")?,
        "schedule source authority",
    )?;
    if field_usize(
        &source,
        "descriptor_inventory_tensor_count",
        "schedule source authority",
    )? != COMPLETE_TENSOR_COUNT
    {
        return Err("schedule source authority descriptor count drifted".into());
    }
    require_string(
        &source,
        "source_config_authority_canonical_path",
        &config.canonical_path.display().to_string(),
        "schedule source authority",
    )?;
    require_string(
        &source,
        "source_config_authority_document_sha256",
        &config.raw_sha256,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "source_config_authority_schema",
        CONFIG_SCHEMA,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "source_config_authority_status",
        CONFIG_STATUS,
        "schedule source authority",
    )?;
    require_string(
        &source,
        "source_config_authority_seal_sha256",
        config
            .seal_sha256
            .as_deref()
            .ok_or("sealed source config has no parsed seal")?,
        "schedule source authority",
    )?;

    let geometry =
        Value::Object(field_object(value, "geometry", "permanent schedule authority")?.clone());
    for (field, expected) in [
        ("layer_count", LAYERS),
        ("hidden_size", HIDDEN),
        ("experts", EXPERTS),
        ("top_k", TOP_K),
        ("moe_intermediate", INTERMEDIATE),
        ("shared_expert_intermediate", INTERMEDIATE),
        ("direct_pack_group_size", GROUP_SIZE),
    ] {
        if field_usize(&geometry, field, "permanent schedule geometry")? != expected {
            return Err(format!(
                "permanent schedule {field} is not Qwen80 MoE geometry"
            ));
        }
    }
    require_bool(
        value,
        "all_48_layers_scheduled",
        true,
        "permanent schedule authority",
    )?;
    require_bool(
        value,
        "all_descriptors_source_artifact_bound",
        true,
        "permanent schedule authority",
    )?;
    let boundary = Value::Object(
        field_object(value, "claim_boundary", "permanent schedule authority")?.clone(),
    );
    require_bool(
        &boundary,
        "assembly_authority_only",
        true,
        "permanent schedule claim boundary",
    )?;
    for field in [
        "decoder_readiness_report",
        "artifact_payload_open_or_scan_performed",
        "metal_device_or_dispatch_performed",
        "runtime_watcher_registry_server_or_hcli_changed",
        "model_execution_performed",
        "token_generation_or_feedback_performed",
        "tps_or_tg_measured",
    ] {
        require_bool(&boundary, field, false, "permanent schedule claim boundary")?;
    }
    require_string(
        &boundary,
        "execution_status",
        EXECUTION_STATUS,
        "permanent schedule claim boundary",
    )?;
    Ok(())
}

fn validate_layout(value: &Value, label: &str) -> Result<Value, String> {
    let layout = Value::Object(field_object(value, "layout", label)?.clone());
    require_string(&layout, "magic", "HQ30G1B1", label)?;
    require_string(&layout, "scale_dtype", "float16", label)?;
    require_string(&layout, "sign_bit_order", "little", label)?;
    if field_usize(&layout, "group_size", label)? != GROUP_SIZE
        || field_usize(&layout, "version", label)? != 1
    {
        return Err(format!("{label} direct-packed group-128 layout drifted"));
    }
    Ok(layout)
}

fn descriptor_reference(
    scheduled: &Value,
    expected_role: &str,
    expected_name: &str,
    expected_shape: &[usize],
    inventory: &BTreeMap<&str, (usize, &Value)>,
    used: &mut BTreeSet<String>,
) -> Result<Value, String> {
    let label = format!("Qwen80 MoE descriptor {expected_name}");
    require_string(scheduled, "role", expected_role, &label)?;
    require_string(scheduled, "tensor_name", expected_name, &label)?;
    let scheduled_shape = field_shape(scheduled, "shape", &label)?;
    if scheduled_shape != expected_shape {
        return Err(format!(
            "{label} shape {:?}, expected {:?}",
            scheduled_shape, expected_shape
        ));
    }
    let expected_elements = checked_elements(expected_shape, &label)?;
    if field_usize(scheduled, "elements", &label)? != expected_elements {
        return Err(format!("{label} elements do not match shape"));
    }
    let (ordinal, manifest) = inventory
        .get(expected_name)
        .copied()
        .ok_or_else(|| format!("{label} is absent from the sealed descriptor inventory"))?;
    if field_usize(scheduled, "inventory_ordinal", &label)? != ordinal {
        return Err(format!("{label} schedule inventory ordinal drifted"));
    }
    for field in [
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
    ] {
        if scheduled.get(field) != manifest.get(field) {
            return Err(format!(
                "{label} schedule binding differs from sealed inventory {field:?}"
            ));
        }
    }
    require_string(scheduled, "source_dtype", "BF16", &label)?;
    require_sha256(
        field_string(scheduled, "artifact_sha256", &label)?,
        &format!("{label} artifact"),
    )?;
    require_sha256(
        field_string(scheduled, "source_shard_sha256", &label)?,
        &format!("{label} source shard"),
    )?;
    let layout = validate_layout(scheduled, &label)?;
    if !used.insert(expected_name.to_owned()) {
        return Err(format!("{label} is scheduled more than once"));
    }
    Ok(json!({
        "inventory_ordinal": ordinal,
        "tensor_name": expected_name,
        "shape": expected_shape,
        "elements": expected_elements,
        "artifact_path": field_string(scheduled, "artifact_path", &label)?,
        "artifact_bytes": field_usize(scheduled, "artifact_bytes", &label)?,
        "artifact_sha256": field_string(scheduled, "artifact_sha256", &label)?,
        "source_dtype": "BF16",
        "source_shard": field_string(scheduled, "source_shard", &label)?,
        "source_shard_sha256": field_string(scheduled, "source_shard_sha256", &label)?,
        "layout": layout,
        "payload_opened_or_uploaded_by_this_receipt": false,
    }))
}

fn expected_expert_shape(projection: &str) -> Result<Vec<usize>, String> {
    match projection {
        "gate_proj" | "up_proj" => Ok(vec![INTERMEDIATE, HIDDEN]),
        "down_proj" => Ok(vec![HIDDEN, INTERMEDIATE]),
        _ => Err(format!(
            "unsupported routed-expert projection {projection:?}"
        )),
    }
}

fn route_descriptor_digest(references: Vec<Value>) -> Result<String, String> {
    sha256_json(&Value::Array(references))
}

fn resolve_routed_projection(
    layer: usize,
    table: &Value,
    projection: &str,
    inventory: &BTreeMap<&str, (usize, &Value)>,
    used: &mut BTreeSet<String>,
) -> Result<Value, String> {
    let label = format!("Qwen80 MoE layer {layer} {projection} table");
    require_string(table, "projection", projection, &label)?;
    let expected_shape = expected_expert_shape(projection)?;
    if field_shape(table, "expected_shape", &label)? != expected_shape {
        return Err(format!("{label} expected geometry drifted"));
    }
    let order = field_array(table, "source_expert_order", &label)?;
    if order.len() != EXPERTS
        || order
            .iter()
            .enumerate()
            .any(|(expert, value)| value.as_u64() != Some(expert as u64))
    {
        return Err(format!("{label} source expert order is not 0 through 511"));
    }
    let descriptors = field_array(table, "descriptors", &label)?;
    if descriptors.len() != EXPERTS {
        return Err(format!(
            "{label} has {} descriptors, expected {EXPERTS}",
            descriptors.len()
        ));
    }
    let mut inventory_ordinals = Vec::with_capacity(EXPERTS);
    let mut identities = Vec::with_capacity(EXPERTS);
    for (expert, scheduled) in descriptors.iter().enumerate() {
        let name = format!("model.layers.{layer}.mlp.experts.{expert}.{projection}.weight");
        let reference = descriptor_reference(
            scheduled,
            &format!("routed_expert_{projection}"),
            &name,
            &expected_shape,
            inventory,
            used,
        )?;
        let ordinal = field_usize(&reference, "inventory_ordinal", &label)?;
        inventory_ordinals.push(ordinal);
        identities.push(json!({
            "expert": expert,
            "inventory_ordinal": ordinal,
            "tensor_name": name,
            "artifact_sha256": field_string(&reference, "artifact_sha256", &label)?,
            "source_shard_sha256": field_string(&reference, "source_shard_sha256", &label)?,
            "shape": expected_shape,
            "layout": reference.get("layout").cloned().ok_or("descriptor reference lacks layout")?,
        }));
    }
    Ok(json!({
        "projection": projection,
        "tensor_name_template": format!("model.layers.{layer}.mlp.experts.{{expert}}.{projection}.weight"),
        "expected_shape": expected_shape,
        "expert_count": EXPERTS,
        "source_expert_order": (0..EXPERTS).collect::<Vec<_>>(),
        "descriptor_inventory_ordinals": inventory_ordinals,
        "descriptor_identity_sha256": route_descriptor_digest(identities)?,
        "all_512_descriptors_resolved_against_sealed_inventory": true,
        "payloads_opened_or_uploaded_by_this_receipt": false,
    }))
}

fn same_input_fixed_order_abi(layer: usize) -> Value {
    let mut execution = vec![
        "post_attention_rmsnorm(first_residual -> normalized_hidden)".to_owned(),
        "router_gate(normalized_hidden -> router_logits[512])".to_owned(),
        "source_stable_top10_select_softmax_renormalize(router_logits -> route_ids[10], route_weights[10])".to_owned(),
    ];
    execution.extend((0..TOP_K).map(|route| {
        format!(
            "route[{route}].gate_up_swiglu_down_then_source_weight(normalized_hidden, selected_expert -> weighted_route[{route}][2048])"
        )
    }));
    execution.extend([
        "shared_expert_gate_up_swiglu_down(normalized_hidden -> shared_output[2048])".to_owned(),
        "shared_scalar_sigmoid_gate(normalized_hidden, shared_output -> gated_shared[2048])"
            .to_owned(),
        "fixed_f32_routed_sum(route[0] + route[1] + ... + route[9] in source-selected order)"
            .to_owned(),
        "fixed_f32_add_gated_shared(routed_sum + gated_shared)".to_owned(),
        "second_residual_add(first_residual + moe_delta -> next_hidden[2048])".to_owned(),
    ]);
    json!({
        "layer": layer,
        "same_input_provenance_required": true,
        "one_token_command_buffer_required": true,
        "retain_buffers_until_capture_fence": true,
        "first_residual_elements": HIDDEN,
        "normalized_hidden_elements": HIDDEN,
        "router_logits_elements": EXPERTS,
        "route_ids_elements": TOP_K,
        "route_weights_elements": TOP_K,
        "weighted_route_shape": [TOP_K, HIDDEN],
        "shared_output_elements": HIDDEN,
        "second_residual_elements": HIDDEN,
        "route_slot_order": (0..TOP_K).collect::<Vec<_>>(),
        "route_policy": {
            "source_stable_top10_required": true,
            "softmax_before_top10_required": true,
            "selected_probability_renormalization_required": true,
            "selected_weights_must_sum_to_one": true,
            "source_selected_order_preserved": true,
            "lower_expert_id_wins_source_tolerance_ties": true,
            "expert_id_domain": [0, EXPERTS - 1],
            "duplicate_selected_expert_ids_rejected": true,
            "route_weight_applied_after_down_projection": true,
            "actual_route_ids_or_weights_materialized_by_this_receipt": false,
        },
        "fixed_f32_combine_order": "route[0] through route[9] in source-selected order; then gated_shared; then first_residual",
        "future_capture_execution_order": execution,
        "fixture_or_synthetic_input_permitted": false,
        "complete_layer_or_token_claim_permitted": false,
    })
}

fn expected_suffix_matches(layer: &Value, layer_index: usize) -> Result<(), String> {
    let order = field_array(layer, "layer_command_boundary_order", "schedule layer")?;
    let actual = order
        .iter()
        .map(|value| {
            value
                .as_str()
                .ok_or("schedule layer command boundary must be string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let expected = expected_moe_suffix();
    if actual.len() < expected.len() || actual[actual.len() - expected.len()..] != expected {
        return Err(format!(
            "schedule layer {layer_index} does not end in the released MoE boundary order"
        ));
    }
    Ok(())
}

fn resolve_moe_layer(
    layer: &Value,
    layer_index: usize,
    inventory: &BTreeMap<&str, (usize, &Value)>,
    used: &mut BTreeSet<String>,
) -> Result<Value, String> {
    let label = format!("schedule layer {layer_index}");
    if field_usize(layer, "layer", &label)? != layer_index {
        return Err(format!("{label} array order/layer number drifted"));
    }
    require_string(layer, "mixer", expected_mixer(layer_index), &label)?;
    expected_suffix_matches(layer, layer_index)?;
    let prefix = format!("model.layers.{layer_index}");
    let postnorm = descriptor_reference(
        layer
            .get("post_attention_layernorm")
            .ok_or_else(|| format!("{label} lacks post_attention_layernorm"))?,
        "post_attention_layernorm",
        &format!("{prefix}.post_attention_layernorm.weight"),
        &[HIDDEN],
        inventory,
        used,
    )?;
    let router = descriptor_reference(
        layer
            .get("router_gate")
            .ok_or_else(|| format!("{label} lacks router_gate"))?,
        "router_gate",
        &format!("{prefix}.mlp.gate.weight"),
        &[EXPERTS, HIDDEN],
        inventory,
        used,
    )?;
    let routed = layer
        .get("routed_experts")
        .ok_or_else(|| format!("{label} lacks routed_experts"))?;
    if field_usize(routed, "expert_count", &label)? != EXPERTS
        || field_usize(routed, "top_k", &label)? != TOP_K
        || field_bool(routed, "route_selection_materialized_by_this_plan", &label)?
        || field_bool(routed, "expert_payload_opened_by_this_plan", &label)?
    {
        return Err(format!(
            "{label} routed-expert policy is not prepared-only Qwen80 top-10"
        ));
    }
    let projection_order = field_array(routed, "projection_execution_order", &label)?;
    let expected_projection_order = ["gate_proj", "up_proj", "down_proj"];
    if projection_order
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .as_deref()
        != Some(&expected_projection_order)
    {
        return Err(format!("{label} routed projection execution order drifted"));
    }
    let tables = field_array(routed, "tables", &label)?;
    if tables.len() != expected_projection_order.len() {
        return Err(format!("{label} routed projection table count drifted"));
    }
    let routed_tables = expected_projection_order
        .iter()
        .zip(tables)
        .map(|(projection, table)| {
            resolve_routed_projection(layer_index, table, projection, inventory, used)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let shared = layer
        .get("shared_expert")
        .ok_or_else(|| format!("{label} lacks shared_expert"))?;
    let shared_order = field_array(shared, "execution_order", &label)?;
    let expected_shared_order = ["gate_proj", "up_proj", "down_proj", "scalar_gate"];
    if shared_order
        .iter()
        .map(Value::as_str)
        .collect::<Option<Vec<_>>>()
        .as_deref()
        != Some(&expected_shared_order)
    {
        return Err(format!("{label} shared-expert execution order drifted"));
    }
    let shared_bindings = [
        (
            "gate_proj",
            "shared_expert_gate_proj",
            format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        ),
        (
            "up_proj",
            "shared_expert_up_proj",
            format!("{prefix}.mlp.shared_expert.up_proj.weight"),
            vec![INTERMEDIATE, HIDDEN],
        ),
        (
            "down_proj",
            "shared_expert_down_proj",
            format!("{prefix}.mlp.shared_expert.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
        ),
        (
            "scalar_gate",
            "shared_expert_scalar_gate",
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            vec![1, HIDDEN],
        ),
    ];
    let mut shared_receipt = Map::new();
    for (field, role, name, shape) in shared_bindings {
        shared_receipt.insert(
            field.to_owned(),
            descriptor_reference(
                shared
                    .get(field)
                    .ok_or_else(|| format!("{label} shared-expert lacks {field}"))?,
                role,
                &name,
                &shape,
                inventory,
                used,
            )?,
        );
    }
    Ok(json!({
        "layer": layer_index,
        "mixer": expected_mixer(layer_index),
        "geometry": {
            "hidden_size": HIDDEN,
            "experts": EXPERTS,
            "top_k": TOP_K,
            "intermediate_size": INTERMEDIATE,
            "direct_pack_group_size": GROUP_SIZE,
            "post_attention_rmsnorm_epsilon": RMS_NORM_EPSILON,
        },
        "post_attention_layernorm": postnorm,
        "router_gate": router,
        "routed_expert_projections": routed_tables,
        "shared_expert": Value::Object(shared_receipt),
        "same_input_fixed_order_abi": same_input_fixed_order_abi(layer_index),
        "descriptor_only": true,
        "route_selection_or_payloads_materialized_by_this_receipt": false,
    }))
}

fn build_receipt(
    schedule: &Document,
    manifest: &Document,
    config: &Document,
) -> Result<Value, String> {
    validate_source_config(config)?;
    let inventory = build_manifest_index(manifest)?;
    validate_schedule_header(schedule, manifest, config)?;
    let layers = field_array(&schedule.value, "layers", "permanent schedule authority")?;
    if layers.len() != LAYERS {
        return Err(format!(
            "permanent schedule has {} layers, expected {LAYERS}",
            layers.len()
        ));
    }
    let mut used = BTreeSet::new();
    let family = layers
        .iter()
        .enumerate()
        .map(|(layer, schedule_layer)| {
            resolve_moe_layer(schedule_layer, layer, &inventory, &mut used)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if used.len() != MOE_DESCRIPTOR_COUNT {
        return Err(format!(
            "resolved {} unique MoE descriptors, expected {MOE_DESCRIPTOR_COUNT}",
            used.len()
        ));
    }
    let mut receipt = json!({
        "schema": RESULT_SCHEMA,
        "status": RESULT_STATUS,
        "source_authority": {
            "model_id": MODEL_ID,
            "model_key": MODEL_KEY,
            "source_repository": SOURCE_REPOSITORY,
            "source_revision": SOURCE_REVISION,
            "source_config_sha256": SOURCE_CONFIG_SHA256,
            "permanent_schedule_authority": {
                "canonical_path": schedule.canonical_path,
                "raw_document_sha256": schedule.raw_sha256,
                "schema": SCHEDULE_SCHEMA,
                "status": SCHEDULE_STATUS,
                "raw_unsealed_static_authority": true,
            },
            "descriptor_inventory": {
                "canonical_path": manifest.canonical_path,
                "raw_document_sha256": manifest.raw_sha256,
                "schema": MANIFEST_SCHEMA,
                "status": MANIFEST_STATUS,
                "seal_sha256": manifest.seal_sha256,
                "tensor_count": COMPLETE_TENSOR_COUNT,
            },
            "source_config_authority": {
                "canonical_path": config.canonical_path,
                "raw_document_sha256": config.raw_sha256,
                "schema": CONFIG_SCHEMA,
                "status": CONFIG_STATUS,
                "seal_sha256": config.seal_sha256,
            },
        },
        "family_geometry": {
            "layer_count": LAYERS,
            "hidden_size": HIDDEN,
            "expert_count": EXPERTS,
            "top_k": TOP_K,
            "intermediate_size": INTERMEDIATE,
            "direct_pack_group_size": GROUP_SIZE,
            "post_attention_rmsnorm_epsilon": RMS_NORM_EPSILON,
        },
        "all_48_moe_layers": family,
        "resolved_moe_descriptor_count": used.len(),
        "all_48_layers_source_bound": true,
        "all_512_expert_projections_resolved_per_projection_per_layer": true,
        "authority_rejections": {
            "external_fixture_authority_rejected": true,
            "synthetic_route_or_payload_authority_rejected": true,
            "missing_or_duplicate_descriptor_rejected": true,
            "non_qwen80_moe_geometry_rejected": true,
            "schedule_manifest_config_hash_or_path_drift_rejected": true,
        },
        "claim_boundary": {
            "descriptor_metadata_read_only": true,
            "artifact_payload_open_or_scan_performed": false,
            "raw_bf16_or_source_shard_opened": false,
            "metal_device_or_dispatch_performed": false,
            "lease_registry_runtime_watcher_server_or_hcli_changed": false,
            "model_execution_or_token_generation_performed": false,
            "complete_layer_or_decoder_earned": false,
            "tps_or_tg_measured": false,
            "execution_status": EXECUTION_STATUS,
        },
    });
    let seal = sha256_json(&receipt)?;
    receipt
        .as_object_mut()
        .ok_or("new MoE family descriptor receipt is not an object")?
        .insert("seal_sha256".into(), Value::String(seal));
    Ok(receipt)
}

fn write_new_receipt(path: &Path, receipt: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("receipt output path must be absolute".into());
    }
    if path.exists() {
        return Err(format!("receipt output already exists: {}", path.display()));
    }
    let mut bytes = serde_json::to_vec_pretty(receipt)
        .map_err(|error| format!("cannot serialize receipt: {error}"))?;
    bytes.push(b'\n');
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create receipt {}: {error}", path.display()))?;
    output
        .write_all(&bytes)
        .map_err(|error| format!("cannot write receipt {}: {error}", path.display()))?;
    output
        .sync_all()
        .map_err(|error| format!("cannot sync receipt {}: {error}", path.display()))?;
    Ok(())
}

fn main() {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let args = parse_args(&arguments).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let schedule = read_document(
        &args.schedule_authority,
        "permanent schedule authority",
        false,
    )
    .unwrap_or_else(|error| panic!("cannot read permanent schedule authority: {error}"));
    let manifest = read_document(&args.descriptor_inventory, "descriptor inventory", true)
        .unwrap_or_else(|error| panic!("cannot read descriptor inventory: {error}"));
    let config = read_document(
        &args.source_config_authority,
        "source-config authority",
        true,
    )
    .unwrap_or_else(|error| panic!("cannot read source-config authority: {error}"));
    let receipt = build_receipt(&schedule, &manifest, &config).unwrap_or_else(|error| {
        panic!("cannot build Qwen80 MoE family descriptor receipt: {error}")
    });
    write_new_receipt(&args.out, &receipt).unwrap_or_else(|error| {
        panic!("cannot emit Qwen80 MoE family descriptor receipt: {error}")
    });
    println!("{}", args.out.display());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seal(value: &mut Value) -> String {
        let seal = sha256_json(value).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("seal_sha256".into(), Value::String(seal.clone()));
        seal
    }

    fn virtual_document(path: &str, value: Value, sealed: bool) -> Document {
        let mut value = value;
        let seal_sha256 = sealed.then(|| seal(&mut value));
        let raw_sha256 = sha256_json(&value).unwrap();
        Document {
            canonical_path: PathBuf::from(path),
            raw_sha256,
            value,
            seal_sha256,
        }
    }

    fn descriptor(name: String, shape: Vec<usize>, ordinal: usize) -> Value {
        let elements = checked_elements(&shape, "fixture descriptor").unwrap();
        json!({
            "tensor_name": name,
            "shape": shape,
            "elements": elements,
            "artifact_path": format!("/fixture/tensors/{ordinal:05}.hq30g"),
            "artifact_bytes": elements / 8 + elements / GROUP_SIZE * 2 + 64,
            "artifact_sha256": sha256_hex(format!("artifact-{ordinal}").as_bytes()),
            "source_dtype": "BF16",
            "source_shard": format!("model-{:05}-of-00040.safetensors", ordinal % 40 + 1),
            "source_shard_sha256": sha256_hex(format!("shard-{}", ordinal % 40).as_bytes()),
            "layout": {"magic": "HQ30G1B1", "group_size": GROUP_SIZE, "scale_dtype": "float16", "sign_bit_order": "little", "version": 1},
        })
    }

    fn scheduled_binding(manifest: &[Value], ordinal: usize, role: &str) -> Value {
        let mut value = manifest[ordinal].clone();
        value
            .as_object_mut()
            .unwrap()
            .insert("inventory_ordinal".into(), json!(ordinal));
        value
            .as_object_mut()
            .unwrap()
            .insert("role".into(), json!(role));
        value
    }

    fn fixture_documents() -> (Document, Document, Document) {
        let config = virtual_document(
            "/fixture/QWEN80_SOURCE_METADATA_CANDIDATE.json",
            json!({
                "schema": CONFIG_SCHEMA,
                "status": CONFIG_STATUS,
                "source": {"repository": SOURCE_REPOSITORY, "revision": SOURCE_REVISION},
                "architecture": {
                    "architectures": ["Qwen3NextForCausalLM"],
                    "config_captured": true,
                    "config_sha256": SOURCE_CONFIG_SHA256,
                    "model_type": "qwen3_next",
                    "hidden_size": HIDDEN,
                    "num_experts": EXPERTS,
                    "num_experts_per_tok": TOP_K,
                    "num_hidden_layers": LAYERS,
                    "vocab_size": 151_936,
                },
            }),
            true,
        );
        let mut manifest_tensors = Vec::with_capacity(COMPLETE_TENSOR_COUNT);
        let mut names = BTreeMap::new();
        let mut add = |name: String, shape: Vec<usize>| {
            let ordinal = manifest_tensors.len();
            names.insert(name.clone(), ordinal);
            manifest_tensors.push(descriptor(name, shape, ordinal));
        };
        for layer in 0..LAYERS {
            let prefix = format!("model.layers.{layer}");
            add(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![HIDDEN],
            );
            add(format!("{prefix}.mlp.gate.weight"), vec![EXPERTS, HIDDEN]);
            for expert in 0..EXPERTS {
                add(
                    format!("{prefix}.mlp.experts.{expert}.gate_proj.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                );
                add(
                    format!("{prefix}.mlp.experts.{expert}.up_proj.weight"),
                    vec![INTERMEDIATE, HIDDEN],
                );
                add(
                    format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                    vec![HIDDEN, INTERMEDIATE],
                );
            }
            add(
                format!("{prefix}.mlp.shared_expert.gate_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
            );
            add(
                format!("{prefix}.mlp.shared_expert.up_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
            );
            add(
                format!("{prefix}.mlp.shared_expert.down_proj.weight"),
                vec![HIDDEN, INTERMEDIATE],
            );
            add(
                format!("{prefix}.mlp.shared_expert_gate.weight"),
                vec![1, HIDDEN],
            );
        }
        drop(add);
        while manifest_tensors.len() < COMPLETE_TENSOR_COUNT {
            let ordinal = manifest_tensors.len();
            let name = format!("unused.fixture_safe.{ordinal}");
            names.insert(name.clone(), ordinal);
            manifest_tensors.push(descriptor(name, vec![GROUP_SIZE], ordinal));
        }
        let manifest = virtual_document(
            "/fixture/QWEN80_COMPLETE_BINARY_GRAVITY_CANDIDATE.json",
            json!({
                "schema": MANIFEST_SCHEMA,
                "status": MANIFEST_STATUS,
                "source": {"repository": SOURCE_REPOSITORY, "tensor_count": COMPLETE_TENSOR_COUNT},
                "tensors": manifest_tensors,
            }),
            true,
        );
        let tensors = field_array(&manifest.value, "tensors", "fixture manifest").unwrap();
        let ordinal = |name: String| *names.get(&name).unwrap();
        let layers = (0..LAYERS)
            .map(|layer| {
                let prefix = format!("model.layers.{layer}");
                let table = |projection: &str, shape: Vec<usize>| {
                    json!({
                        "projection": projection,
                        "expected_shape": shape,
                        "source_expert_order": (0..EXPERTS).collect::<Vec<_>>(),
                        "descriptors": (0..EXPERTS).map(|expert| {
                            let name = format!("{prefix}.mlp.experts.{expert}.{projection}.weight");
                            scheduled_binding(tensors, ordinal(name), &format!("routed_expert_{projection}"))
                        }).collect::<Vec<_>>(),
                    })
                };
                json!({
                    "layer": layer,
                    "mixer": expected_mixer(layer),
                    "post_attention_layernorm": scheduled_binding(tensors, ordinal(format!("{prefix}.post_attention_layernorm.weight")), "post_attention_layernorm"),
                    "router_gate": scheduled_binding(tensors, ordinal(format!("{prefix}.mlp.gate.weight")), "router_gate"),
                    "routed_experts": {
                        "expert_count": EXPERTS,
                        "top_k": TOP_K,
                        "projection_execution_order": ["gate_proj", "up_proj", "down_proj"],
                        "route_selection_materialized_by_this_plan": false,
                        "expert_payload_opened_by_this_plan": false,
                        "tables": [
                            table("gate_proj", vec![INTERMEDIATE, HIDDEN]),
                            table("up_proj", vec![INTERMEDIATE, HIDDEN]),
                            table("down_proj", vec![HIDDEN, INTERMEDIATE]),
                        ],
                    },
                    "shared_expert": {
                        "execution_order": ["gate_proj", "up_proj", "down_proj", "scalar_gate"],
                        "gate_proj": scheduled_binding(tensors, ordinal(format!("{prefix}.mlp.shared_expert.gate_proj.weight")), "shared_expert_gate_proj"),
                        "up_proj": scheduled_binding(tensors, ordinal(format!("{prefix}.mlp.shared_expert.up_proj.weight")), "shared_expert_up_proj"),
                        "down_proj": scheduled_binding(tensors, ordinal(format!("{prefix}.mlp.shared_expert.down_proj.weight")), "shared_expert_down_proj"),
                        "scalar_gate": scheduled_binding(tensors, ordinal(format!("{prefix}.mlp.shared_expert_gate.weight")), "shared_expert_scalar_gate"),
                    },
                    "layer_command_boundary_order": expected_moe_suffix(),
                })
            })
            .collect::<Vec<_>>();
        let schedule = virtual_document(
            "/fixture/QWEN80_48_LAYER_PAYLOAD_SCHEDULE_AUTHORITY.json",
            json!({
                "schema": SCHEDULE_SCHEMA,
                "status": SCHEDULE_STATUS,
                "source_authority": {
                    "model_id": MODEL_ID,
                    "model_key": MODEL_KEY,
                    "source_repository": SOURCE_REPOSITORY,
                    "source_revision": SOURCE_REVISION,
                    "source_config_sha256": SOURCE_CONFIG_SHA256,
                    "descriptor_inventory_canonical_path": manifest.canonical_path,
                    "descriptor_inventory_document_sha256": manifest.raw_sha256,
                    "descriptor_inventory_schema": MANIFEST_SCHEMA,
                    "descriptor_inventory_status": MANIFEST_STATUS,
                    "descriptor_inventory_seal_sha256": manifest.seal_sha256,
                    "descriptor_inventory_tensor_count": COMPLETE_TENSOR_COUNT,
                    "source_config_authority_canonical_path": config.canonical_path,
                    "source_config_authority_document_sha256": config.raw_sha256,
                    "source_config_authority_schema": CONFIG_SCHEMA,
                    "source_config_authority_status": CONFIG_STATUS,
                    "source_config_authority_seal_sha256": config.seal_sha256,
                },
                "geometry": {
                    "layer_count": LAYERS,
                    "hidden_size": HIDDEN,
                    "experts": EXPERTS,
                    "top_k": TOP_K,
                    "moe_intermediate": INTERMEDIATE,
                    "shared_expert_intermediate": INTERMEDIATE,
                    "direct_pack_group_size": GROUP_SIZE,
                },
                "all_48_layers_scheduled": true,
                "all_descriptors_source_artifact_bound": true,
                "claim_boundary": {
                    "assembly_authority_only": true,
                    "decoder_readiness_report": false,
                    "artifact_payload_open_or_scan_performed": false,
                    "metal_device_or_dispatch_performed": false,
                    "runtime_watcher_registry_server_or_hcli_changed": false,
                    "model_execution_performed": false,
                    "token_generation_or_feedback_performed": false,
                    "tps_or_tg_measured": false,
                    "execution_status": EXECUTION_STATUS,
                },
                "layers": layers,
            }),
            false,
        );
        (schedule, manifest, config)
    }

    #[test]
    fn same_input_abi_is_fixed_route_then_shared_then_residual() {
        let abi = same_input_fixed_order_abi(17);
        assert_eq!(
            abi["route_slot_order"],
            json!((0..TOP_K).collect::<Vec<_>>())
        );
        let order = abi["future_capture_execution_order"].as_array().unwrap();
        assert!(order[3].as_str().unwrap().contains("route[0]"));
        assert!(order[12].as_str().unwrap().contains("route[9]"));
        assert!(order[13].as_str().unwrap().contains("shared_expert"));
        assert!(order[17].as_str().unwrap().contains("second_residual"));
        assert_eq!(
            abi["fixed_f32_combine_order"],
            "route[0] through route[9] in source-selected order; then gated_shared; then first_residual"
        );
    }

    #[test]
    fn header_rejects_non_qwen80_geometry_and_fixture_authority() {
        let (mut schedule, manifest, config) = fixture_documents();
        schedule.value["geometry"]["top_k"] = json!(8);
        let error = validate_schedule_header(&schedule, &manifest, &config).unwrap_err();
        assert!(error.contains("top_k"));

        let (mut schedule, manifest, config) = fixture_documents();
        schedule.value["fixture_or_synthetic_authority"] = json!(true);
        let error = validate_schedule_header(&schedule, &manifest, &config).unwrap_err();
        assert!(error.contains("fixture"));
    }

    #[test]
    fn resolves_every_moe_descriptor_and_rejects_descriptor_drift() {
        let (mut schedule, manifest, config) = fixture_documents();
        let receipt = build_receipt(&schedule, &manifest, &config).unwrap();
        assert_eq!(receipt["schema"], RESULT_SCHEMA);
        assert_eq!(receipt["status"], RESULT_STATUS);
        assert_eq!(
            receipt["resolved_moe_descriptor_count"],
            MOE_DESCRIPTOR_COUNT
        );
        assert_eq!(
            receipt["all_48_moe_layers"].as_array().unwrap().len(),
            LAYERS
        );
        assert!(receipt["seal_sha256"].as_str().is_some_and(is_lower_sha256));
        assert_eq!(
            receipt["all_48_moe_layers"][47]["routed_expert_projections"][2]
                ["descriptor_inventory_ordinals"]
                .as_array()
                .unwrap()
                .len(),
            EXPERTS
        );
        schedule.value["layers"][0]["routed_experts"]["tables"][0]["descriptors"][0]["shape"] =
            json!([1, HIDDEN]);
        let error = build_receipt(&schedule, &manifest, &config).unwrap_err();
        assert!(error.contains("shape"));
    }
}
