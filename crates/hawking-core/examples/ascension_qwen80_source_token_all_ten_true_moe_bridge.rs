//! CPU-only source-token Qwen80 L0 all-ten true-MoE bridge material.
//!
//! This is a deliberately distinct path from the historical synthetic-router
//! bridge.  It consumes only a sealed token-1/zero-L0-state route authority,
//! a sealed strict-Metal first-residual prefix, and one current admitted
//! compact artifact.  It materializes direct-packed scale/sign section hashes
//! for the ten exact source-token experts, but it neither creates a Metal
//! context nor executes a layer/token.

use hawking_core::model::qwen80_complete_runtime::{
    Qwen80CompleteArtifactCatalog, Qwen80SourceTokenAllTenRoutedExpertPlanAuthority,
};
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const MATERIAL_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_all_ten_true_moe_source_bridge_material.v1";
const MATERIAL_STATUS: &str =
    "CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_MATERIAL_READY_FOR_SEAL";
const SOURCE_AUTHORITY_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_l0_all_ten_route_plan_authority.v1";
const SOURCE_AUTHORITY_STATUS: &str =
    "SEALED_CURRENT_ADMITTED_QWEN80_SOURCE_TOKEN_L0_ALL_TEN_ROUTE_PLAN_READY_FOR_NEW_TYPED_BRIDGE";
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
const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    admission_current: PathBuf,
    source_token_route_authority: PathBuf,
    first_residual_receipt: PathBuf,
    out: PathBuf,
}

#[derive(Debug)]
struct Authority {
    manifest: PathBuf,
    manifest_raw: Vec<u8>,
    manifest_document_sha256: String,
    manifest_seal_sha256: String,
    admission_current: PathBuf,
    admission_raw: Vec<u8>,
    admission_pointer_seal_sha256: String,
    admission_receipt: PathBuf,
    admission_receipt_raw: Vec<u8>,
    admission_receipt_seal_sha256: String,
    source_audit_seal_sha256: String,
    source_revision: String,
    route_authority: PathBuf,
    route_authority_raw: Vec<u8>,
    route_authority_seal_sha256: String,
    source_token_plan: Value,
    first_residual_receipt: PathBuf,
    first_residual_raw: Vec<u8>,
    first_residual_seal_sha256: String,
    first_residual_output_sha256: String,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_all_ten_true_moe_bridge \\\n+--manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH \\\n+--source-token-route-authority ABSOLUTE_PATH --first-residual-receipt ABSOLUTE_PATH \\\n+--out ABSOLUTE_NEW_FILE"
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--manifest"
            | "--admission-current"
            | "--source-token-route-authority"
            | "--first-residual-receipt"
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
        source_token_route_authority: required("--source-token-route-authority")?,
        first_residual_receipt: required("--first-residual-receipt")?,
        out: required("--out")?,
    };
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    if args.out.exists() || !args.out.parent().is_some_and(Path::is_dir) {
        return Err("--out must be new with an existing parent".to_owned());
    }
    Ok(args)
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || value
            .bytes()
            .any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
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
        return Err(format!("{label} must be an object"));
    }
    Ok((path, raw, value))
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

fn field_string<'a>(
    object: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
}

fn evidence(path: &Path, raw: &[u8]) -> Value {
    json!({"path": path, "present": true, "bytes": raw.len(), "sha256": sha256(raw)})
}

fn evidence_matches(
    observed: &Map<String, Value>,
    expected: &Value,
    label: &str,
) -> Result<(), String> {
    let expected = object(expected, label)?;
    if observed.get("present") != Some(&Value::Bool(true))
        || observed.get("path") != expected.get("path")
        || observed.get("bytes") != expected.get("bytes")
        || observed.get("sha256") != expected.get("sha256")
    {
        return Err(format!("{label} byte/path identity drifted"));
    }
    Ok(())
}

fn validate_authority(args: &Args) -> Result<Authority, String> {
    let (manifest, manifest_raw, manifest_document) = read_json(&args.manifest, "manifest")?;
    let manifest_object = object(&manifest_document, "manifest")?;
    if field_string(manifest_object, "schema", "manifest")? != MANIFEST_SCHEMA {
        return Err("manifest schema drifted".to_owned());
    }
    let manifest_document_sha256 = sha256(&manifest_raw);
    let manifest_seal_sha256 = field_string(manifest_object, "seal_sha256", "manifest")?.to_owned();
    require_sha256(&manifest_seal_sha256, "manifest seal")?;

    let (admission_current, admission_raw, admission_document) =
        read_json(&args.admission_current, "admission current")?;
    let admission_object = object(&admission_document, "admission current")?;
    if field_string(admission_object, "schema", "admission current")? != ADMISSION_POINTER_SCHEMA
        || field_string(admission_object, "status", "admission current")?
            != ADMISSION_POINTER_STATUS
    {
        return Err("admission-current schema/status drifted".to_owned());
    }
    let admission_pointer_seal_sha256 =
        field_string(admission_object, "seal_sha256", "admission current")?.to_owned();
    require_sha256(&admission_pointer_seal_sha256, "admission-current seal")?;
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
    let selection = field_object(admission_object, "admission_receipt", "admission current")?;
    let receipt_path = canonical_regular(
        Path::new(field_string(
            selection,
            "path",
            "admission current receipt",
        )?),
        "immutable admission receipt",
    )?;
    let (admission_receipt, admission_receipt_raw, receipt_document) =
        read_json(&receipt_path, "immutable admission receipt")?;
    let receipt = object(&receipt_document, "immutable admission receipt")?;
    if field_string(receipt, "schema", "immutable admission receipt")? != ADMISSION_RECEIPT_SCHEMA
        || field_string(receipt, "status", "immutable admission receipt")?
            != ADMISSION_RECEIPT_STATUS
    {
        return Err("immutable admission receipt schema/status drifted".to_owned());
    }
    let admission_receipt_seal_sha256 =
        field_string(receipt, "seal_sha256", "immutable admission receipt")?.to_owned();
    require_sha256(
        &admission_receipt_seal_sha256,
        "immutable admission receipt seal",
    )?;
    if field_string(selection, "seal_sha256", "admission current receipt")?
        != admission_receipt_seal_sha256
    {
        return Err("admission-current immutable receipt seal drifted".to_owned());
    }
    let revalidation = field_object(
        receipt,
        "current_source_revalidation",
        "immutable admission receipt",
    )?;
    let source_audit_seal_sha256 = field_string(
        revalidation,
        "source_audit_seal_sha256",
        "source revalidation",
    )?
    .to_owned();
    require_sha256(&source_audit_seal_sha256, "source audit seal")?;
    let source_revision = field_string(revalidation, "revision", "source revalidation")?.to_owned();

    let (first_residual_receipt, first_residual_raw, prefix_document) =
        read_json(&args.first_residual_receipt, "first-residual outer receipt")?;
    let prefix = object(&prefix_document, "first-residual outer receipt")?;
    if field_string(prefix, "schema", "first-residual outer receipt")? != PREFIX_SCHEMA
        || field_string(prefix, "status", "first-residual outer receipt")? != PREFIX_STATUS
    {
        return Err("first-residual outer schema/status drifted".to_owned());
    }
    let first_residual_seal_sha256 =
        field_string(prefix, "seal_sha256", "first-residual outer receipt")?.to_owned();
    require_sha256(&first_residual_seal_sha256, "first-residual outer seal")?;
    let prefix_source = field_object(prefix, "source_binding", "first-residual outer receipt")?;
    let prefix_manifest = field_object(prefix_source, "manifest", "first-residual source")?;
    if field_string(prefix_manifest, "sha256", "first-residual manifest")?
        != manifest_document_sha256
        || field_string(
            prefix_source,
            "manifest_seal_sha256",
            "first-residual source",
        )? != manifest_seal_sha256
        || field_string(
            prefix_source,
            "admission_receipt_seal_sha256",
            "first-residual source",
        )? != admission_receipt_seal_sha256
    {
        return Err("first-residual outer artifact identity drifted".to_owned());
    }
    let prefix_admission =
        field_object(prefix_source, "admission_current", "first-residual source")?;
    let prefix_admission_path = canonical_regular(
        Path::new(field_string(
            prefix_admission,
            "path",
            "first-residual admission",
        )?),
        "first-residual admission pointer",
    )?;
    if prefix_admission_path != admission_current {
        return Err("first-residual outer admission pointer path drifted".to_owned());
    }
    let first_output = field_object(
        prefix,
        "first_residual_output",
        "first-residual outer receipt",
    )?;
    if first_output.get("layer") != Some(&json!(0))
        || first_output.get("linear_state_slot") != Some(&json!(0))
        || first_output.get("elements") != Some(&json!(HIDDEN))
        || first_output.get("same_command_graph_required") != Some(&Value::Bool(true))
    {
        return Err("first-residual outer geometry/state identity drifted".to_owned());
    }
    let first_residual_output_sha256 =
        field_string(first_output, "sha256", "first-residual output")?.to_owned();
    require_sha256(&first_residual_output_sha256, "first-residual output SHA")?;

    let (route_authority, route_authority_raw, route_authority_document) = read_json(
        &args.source_token_route_authority,
        "source-token route authority",
    )?;
    let route_authority_object = object(&route_authority_document, "source-token route authority")?;
    if field_string(
        route_authority_object,
        "schema",
        "source-token route authority",
    )? != SOURCE_AUTHORITY_SCHEMA
        || field_string(
            route_authority_object,
            "status",
            "source-token route authority",
        )? != SOURCE_AUTHORITY_STATUS
    {
        return Err("source-token route authority schema/status drifted".to_owned());
    }
    let route_authority_seal_sha256 = field_string(
        route_authority_object,
        "seal_sha256",
        "source-token route authority",
    )?
    .to_owned();
    require_sha256(
        &route_authority_seal_sha256,
        "source-token route authority seal",
    )?;
    let route_source = field_object(
        route_authority_object,
        "source_binding",
        "source-token route authority",
    )?;
    evidence_matches(
        field_object(route_source, "manifest", "source-token route authority")?,
        &evidence(&manifest, &manifest_raw),
        "source-token route authority manifest",
    )?;
    if field_string(
        route_source,
        "manifest_seal_sha256",
        "source-token route authority",
    )? != manifest_seal_sha256
        || field_string(
            route_source,
            "admission_receipt_seal_sha256",
            "source-token route authority",
        )? != admission_receipt_seal_sha256
        || field_string(
            route_source,
            "first_residual_outer_seal_sha256",
            "source-token route authority",
        )? != first_residual_seal_sha256
    {
        return Err("source-token route authority immutable identity drifted".to_owned());
    }
    let route_admission = field_object(
        route_source,
        "admission_current",
        "source-token route authority",
    )?;
    if canonical_regular(
        Path::new(field_string(
            route_admission,
            "path",
            "source-token route authority",
        )?),
        "source-token route authority admission pointer",
    )? != admission_current
    {
        return Err("source-token route authority admission pointer path drifted".to_owned());
    }
    let route_prefix = field_object(
        route_source,
        "first_residual_outer_receipt",
        "source-token route authority",
    )?;
    evidence_matches(
        route_prefix,
        &evidence(&first_residual_receipt, &first_residual_raw),
        "source-token route authority prefix receipt",
    )?;
    let source_token_plan = route_authority_object
        .get("source_token_plan")
        .cloned()
        .ok_or_else(|| "source-token route authority lacks source_token_plan".to_owned())?;

    Ok(Authority {
        manifest,
        manifest_raw,
        manifest_document_sha256,
        manifest_seal_sha256,
        admission_current,
        admission_raw,
        admission_pointer_seal_sha256,
        admission_receipt,
        admission_receipt_raw,
        admission_receipt_seal_sha256,
        source_audit_seal_sha256,
        source_revision,
        route_authority,
        route_authority_raw,
        route_authority_seal_sha256,
        source_token_plan,
        first_residual_receipt,
        first_residual_raw,
        first_residual_seal_sha256,
        first_residual_output_sha256,
    })
}

fn write_new(path: &Path, document: &Value) -> Result<(), String> {
    let mut raw = serde_json::to_vec_pretty(document).map_err(|error| error.to_string())?;
    raw.push(b'\n');
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;
    if let Err(error) = file.write_all(&raw).and_then(|_| file.sync_all()) {
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

fn run(args: &Args) -> Result<Value, String> {
    let authority = validate_authority(args)?;
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: authority.source_audit_seal_sha256.clone(),
        expected_source_revision: authority.source_revision.clone(),
    };
    // Exactly one strict compact artifact scan.  All direct payload access
    // below is through the admitted in-process catalog; no BF16 source or
    // reopened artifact path is legal here.
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.manifest, &admission)
        .map_err(|error| format!("strict Qwen80 artifact admission failed: {error}"))?;
    let hybrid = catalog
        .complete_hybrid_decoder_plan(2)
        .map_err(|error| format!("complete hybrid decoder-plan bind failed: {error}"))?;
    let route_authority = Qwen80SourceTokenAllTenRoutedExpertPlanAuthority {
        manifest_document_sha256: &authority.manifest_document_sha256,
        plan_authority_document_sha256: &sha256(&authority.route_authority_raw),
        admission_receipt_seal_sha256: &authority.admission_receipt_seal_sha256,
        first_residual_outer_receipt_seal_sha256: &authority.first_residual_seal_sha256,
    };
    let route_plan = hybrid
        .bind_source_token_all_ten_routed_expert_plan(
            0,
            &route_authority,
            &authority.source_token_plan,
        )
        .map_err(|error| format!("source-token all-ten route-plan bind failed: {error}"))?;
    let first_residual = catalog
        .first_residual_device_binding(0)
        .map_err(|error| format!("first-residual device binding failed: {error}"))?;
    let source_bridge = catalog
        .build_all_ten_true_moe_source_bridge(&route_plan, first_residual)
        .map_err(|error| {
            format!("source-token direct-packed all-ten bridge build failed: {error}")
        })?;
    let bundle = source_bridge.route_payloads();
    let route = bundle.route();
    if bundle.layer() != 0
        || bundle.waves().len() != TOP_K
        || source_bridge.first_residual().layer() != 0
        || source_bridge.first_residual().linear_state_slot() != 0
        || source_bridge.first_residual().elements() != HIDDEN
        || !source_bridge.first_residual().same_command_graph_required()
    {
        return Err("source-token bridge geometry/order drifted".to_owned());
    }
    Ok(json!({
        "schema": MATERIAL_SCHEMA,
        "status": MATERIAL_STATUS,
        "source_binding": {
            "manifest": evidence(&authority.manifest, &authority.manifest_raw),
            "admission_current": evidence(&authority.admission_current, &authority.admission_raw),
            "admission_receipt": evidence(&authority.admission_receipt, &authority.admission_receipt_raw),
            "source_token_route_authority": evidence(&authority.route_authority, &authority.route_authority_raw),
            "first_residual_receipt": evidence(&authority.first_residual_receipt, &authority.first_residual_raw),
            "manifest_seal_sha256": authority.manifest_seal_sha256,
            "admission_pointer_seal_sha256": authority.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": authority.admission_receipt_seal_sha256,
            "source_audit_seal_sha256": authority.source_audit_seal_sha256,
            "source_revision": authority.source_revision,
            "source_token_route_authority_seal_sha256": authority.route_authority_seal_sha256,
            "first_residual_receipt_seal_sha256": authority.first_residual_seal_sha256,
        },
        "typed_bridge": {
            "layer": 0,
            "source_token_id": 1,
            "route_count": TOP_K,
            "first_residual_elements": HIDDEN,
            "same_command_graph_required": true,
            "first_residual_output_sha256": authority.first_residual_output_sha256,
            "first_residual_receipt_seal_sha256": authority.first_residual_seal_sha256,
            "source_token_route_authority_seal_sha256": authority.route_authority_seal_sha256,
            "compact_section_sha256": {
                "gate_scales": bundle.gate_scales_sha256(),
                "gate_signs": bundle.gate_signs_sha256(),
                "up_scales": bundle.up_scales_sha256(),
                "up_signs": bundle.up_signs_sha256(),
                "down_scales": bundle.down_scales_sha256(),
                "down_signs": bundle.down_signs_sha256(),
            },
        },
        "route_authority": {
            "ids": route.ids,
            "normalized_weights": route.weights,
            "wave_count": bundle.waves().len(),
            "all_thirty_wave_payloads_use_admission_verified_immutable_snapshots": true,
        },
        "artifact_scan": {
            "complete_artifact_admission_performed_once": true,
            "catalog_reused_for_source_token_all_ten_bridge": true,
            "raw_bf16_or_safetensors_opened": false,
        },
        "claim_boundary": {
            "cpu_source_token_bridge_material_only": true,
            "metal_device_or_dispatch_performed": false,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
            "requires_separate_sealed_source_token_typed_bridge_outer_preflight_and_fresh_component_lease": true,
        },
    }))
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("Qwen80 source-token all-ten bridge material refused: {error}");
        std::process::exit(2);
    });
    match run(&args) {
        Ok(document) => {
            if let Err(error) = write_new(&args.out, &document) {
                eprintln!("Qwen80 source-token all-ten bridge material refused: {error}");
                std::process::exit(2);
            }
            println!("{}", args.out.display());
        }
        Err(error) => {
            eprintln!("Qwen80 source-token all-ten bridge material refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_token_bridge_is_distinct_and_cpu_only() {
        assert!(MATERIAL_SCHEMA.contains("source_token"));
        assert!(MATERIAL_STATUS.contains("READY_FOR_SEAL"));
        assert_eq!(TOP_K, 10);
        assert_eq!(HIDDEN, 2048);
    }

    #[test]
    fn sha256_guard_rejects_noncanonical_values() {
        assert!(require_sha256(&"a".repeat(64), "test").is_ok());
        assert!(require_sha256(&"A".repeat(64), "test").is_err());
        assert!(require_sha256("short", "test").is_err());
    }
}
