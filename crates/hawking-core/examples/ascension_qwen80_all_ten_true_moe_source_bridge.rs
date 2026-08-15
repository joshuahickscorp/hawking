//! Emit the exact direct-packed L0 all-ten source bridge material for a later
//! sealed component lease.
//!
//! This is deliberately CPU/build-only.  It performs one strict complete
//! artifact admission scan, reuses the admitted catalog to bind the existing
//! router-selected all-ten plan, and emits only hashes/geometry for the six
//! compact route sections.  It neither creates a Metal context nor opens a
//! command buffer.  A small Python receipt wrapper seals the immutable output
//! after independently binding the current prefix outer receipt.

use hawking_core::model::qwen80_complete_runtime::{
    Qwen80AllTenRoutedExpertPlanAuthority, Qwen80CompleteArtifactCatalog,
};
use hawking_core::model::qwen_complete_binary::{CompleteBinaryAdmission, QwenCompleteBinaryModel};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const MATERIAL_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_true_moe_source_bridge_material.v1";
const MATERIAL_STATUS: &str =
    "CURRENT_ADMITTED_QWEN80_ALL_TEN_TRUE_MOE_SOURCE_BRIDGE_MATERIAL_READY_FOR_SEAL";
const MANIFEST_SCHEMA: &str = "hawking.ascension.qwen80_complete_binary_gravity.v1";
const ADMISSION_POINTER_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_current_pointer.v1";
const ADMISSION_POINTER_STATUS: &str = "CURRENT_COMPLETE_BINARY_ADMISSION_RECEIPT_SELECTED";
const ADMISSION_RECEIPT_SCHEMA: &str =
    "hawking.ascension.qwen_complete_binary_gravity_admission_receipt.v1";
const ADMISSION_RECEIPT_STATUS: &str =
    "EARNED_COMPLETE_BINARY_ARTIFACT_ADMITTED_NOT_RUNTIME_OR_CAPABILITY_QUALIFIED";
const ROUTER_INNER_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_postnorm_router_top10.v1";
const ROUTER_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_POSTNORM_ROUTER_TOP10_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const ROUTER_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_direct_packed_postnorm_router_top10_outer_launcher.v1";
const ROUTER_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_POSTNORM_ROUTER_TOP10_OUTER_TERMINAL_COMPONENT_ONLY";
const ROUTE_PLAN_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_binding_plan.v1";
const ROUTE_PLAN_STATUS: &str =
    "SOURCE_BOUND_ALL_TEN_ROUTED_EXPERT_PLAN_READY_NOT_EXECUTED_NOT_COMBINED";
const FIRST_RESIDUAL_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_first_residual_outer_capture.v1";
const FIRST_RESIDUAL_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_FIRST_RESIDUAL_STRICT_MATH_COMPONENT_ONLY";
const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;

#[derive(Debug)]
struct Args {
    manifest: PathBuf,
    admission_current: PathBuf,
    router_receipt: PathBuf,
    router_outer_receipt: PathBuf,
    route_plan: PathBuf,
    first_residual_receipt: PathBuf,
    out: PathBuf,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_all_ten_true_moe_source_bridge \\\n+--manifest ABSOLUTE_PATH --admission-current ABSOLUTE_PATH \\\n+--router-receipt ABSOLUTE_PATH --router-outer-receipt ABSOLUTE_PATH \\\n+--route-plan ABSOLUTE_PATH --first-residual-receipt ABSOLUTE_PATH \\\n+--out ABSOLUTE_NEW_FILE"
}

fn parse_args() -> Result<Args, String> {
    let mut values = BTreeMap::<String, String>::new();
    let mut arguments = env::args().skip(1);
    while let Some(flag) = arguments.next() {
        match flag.as_str() {
            "--manifest"
            | "--admission-current"
            | "--router-receipt"
            | "--router-outer-receipt"
            | "--route-plan"
            | "--first-residual-receipt"
            | "--out" => {
                let value = arguments
                    .next()
                    .ok_or_else(|| format!("{flag} requires a value; {}", usage()))?;
                if values.insert(flag.clone(), value).is_some() {
                    return Err(format!("{flag} repeated; {}", usage()));
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
        router_receipt: required("--router-receipt")?,
        router_outer_receipt: required("--router-outer-receipt")?,
        route_plan: required("--route-plan")?,
        first_residual_receipt: required("--first-residual-receipt")?,
        out: required("--out")?,
    };
    if !values.is_empty() {
        return Err(format!("unconsumed arguments: {values:?}"));
    }
    if args.out.exists() {
        return Err(format!(
            "refusing to overwrite immutable --out {}",
            args.out.display()
        ));
    }
    if !args.out.parent().is_some_and(Path::is_dir) {
        return Err("--out parent must already exist".to_owned());
    }
    Ok(args)
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

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn read_json(path: &Path, label: &str) -> Result<(PathBuf, Vec<u8>, Value), String> {
    let path = canonical_regular(path, label)?;
    let raw = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let document =
        serde_json::from_slice(&raw).map_err(|error| format!("cannot parse {label}: {error}"))?;
    Ok((path, raw, document))
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be an object"))
}

fn field_object<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label}.{field} must be an object"))
}

fn field_string<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| format!("{label}.{field} must be a non-empty string"))
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

fn evidence(path: &Path, raw: &[u8]) -> Value {
    json!({
        "path": path,
        "present": true,
        "bytes": raw.len(),
        "sha256": sha256(raw),
    })
}

fn string_array(value: &Map<String, Value>, field: &str, label: &str) -> Result<Vec<u16>, String> {
    let items = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{field} must be an array"))?;
    if items.len() != TOP_K {
        return Err(format!(
            "{label}.{field} must contain exactly {TOP_K} entries"
        ));
    }
    items
        .iter()
        .map(|item| {
            item.as_u64()
                .and_then(|id| u16::try_from(id).ok())
                .ok_or_else(|| format!("{label}.{field} must contain u16 values"))
        })
        .collect()
}

fn number_array(value: &Map<String, Value>, field: &str, label: &str) -> Result<Vec<f64>, String> {
    let items = value
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("{label}.{field} must be an array"))?;
    if items.len() != TOP_K {
        return Err(format!(
            "{label}.{field} must contain exactly {TOP_K} entries"
        ));
    }
    items
        .iter()
        .map(|item| {
            item.as_f64()
                .filter(|value| value.is_finite() && *value >= 0.0)
                .ok_or_else(|| format!("{label}.{field} must contain finite non-negative numbers"))
        })
        .collect()
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
    admission_receipt_seal_sha256: String,
    source_audit_seal_sha256: String,
    source_revision: String,
    router_receipt: PathBuf,
    router_receipt_raw: Vec<u8>,
    router_outer_receipt: PathBuf,
    router_outer_receipt_raw: Vec<u8>,
    router_outer_receipt_seal_sha256: String,
    route_plan: PathBuf,
    route_plan_raw: Vec<u8>,
    first_residual_receipt: PathBuf,
    first_residual_raw: Vec<u8>,
    first_residual_seal_sha256: String,
    first_residual_output_sha256: String,
    first_residual_cpu_baseline: Value,
}

fn validate_authority(args: &Args) -> Result<Authority, String> {
    let (manifest, manifest_raw, manifest_document) = read_json(&args.manifest, "manifest")?;
    let manifest_object = object(&manifest_document, "manifest")?;
    if field_string(manifest_object, "schema", "manifest")? != MANIFEST_SCHEMA {
        return Err("manifest schema drifted".to_owned());
    }
    let manifest_seal_sha256 = field_string(manifest_object, "seal_sha256", "manifest")?.to_owned();
    require_sha256(&manifest_seal_sha256, "manifest seal")?;
    let manifest_document_sha256 = sha256(&manifest_raw);

    let (admission_current, admission_raw, admission_document) =
        read_json(&args.admission_current, "admission current")?;
    let admission_object = object(&admission_document, "admission current")?;
    if field_string(admission_object, "schema", "admission current")? != ADMISSION_POINTER_SCHEMA
        || field_string(admission_object, "status", "admission current")?
            != ADMISSION_POINTER_STATUS
    {
        return Err("admission current schema/status drifted".to_owned());
    }
    let admission_pointer_seal_sha256 =
        field_string(admission_object, "seal_sha256", "admission current")?.to_owned();
    require_sha256(&admission_pointer_seal_sha256, "admission pointer seal")?;
    let selected_manifest =
        field_object(admission_object, "complete_manifest", "admission current")?;
    if field_string(
        selected_manifest,
        "document_sha256",
        "admission current complete_manifest",
    )? != manifest_document_sha256
        || field_string(
            selected_manifest,
            "seal_sha256",
            "admission current complete_manifest",
        )? != manifest_seal_sha256
    {
        return Err("admission current manifest identity drifted".to_owned());
    }
    let selection = field_object(admission_object, "admission_receipt", "admission current")?;
    let receipt_path = canonical_regular(
        Path::new(field_string(
            selection,
            "path",
            "admission current admission_receipt",
        )?),
        "admission receipt",
    )?;
    let (_, _, receipt_document) = read_json(&receipt_path, "admission receipt")?;
    let receipt_object = object(&receipt_document, "admission receipt")?;
    if field_string(receipt_object, "schema", "admission receipt")? != ADMISSION_RECEIPT_SCHEMA
        || field_string(receipt_object, "status", "admission receipt")? != ADMISSION_RECEIPT_STATUS
    {
        return Err("admission receipt schema/status drifted".to_owned());
    }
    let admission_receipt_seal_sha256 =
        field_string(receipt_object, "seal_sha256", "admission receipt")?.to_owned();
    require_sha256(&admission_receipt_seal_sha256, "admission receipt seal")?;
    if field_string(
        selection,
        "seal_sha256",
        "admission current admission_receipt",
    )? != admission_receipt_seal_sha256
    {
        return Err("admission current receipt seal drifted".to_owned());
    }
    let revalidation = field_object(
        receipt_object,
        "current_source_revalidation",
        "admission receipt",
    )?;
    let source_audit_seal_sha256 = field_string(
        revalidation,
        "source_audit_seal_sha256",
        "admission revalidation",
    )?
    .to_owned();
    require_sha256(&source_audit_seal_sha256, "source audit seal")?;
    let source_revision =
        field_string(revalidation, "revision", "admission revalidation")?.to_owned();

    let (router_receipt, router_receipt_raw, router_document) =
        read_json(&args.router_receipt, "router receipt")?;
    let router = object(&router_document, "router receipt")?;
    if field_string(router, "schema", "router receipt")? != ROUTER_INNER_SCHEMA
        || field_string(router, "status", "router receipt")? != ROUTER_INNER_STATUS
        || field_string(router, "mode", "router receipt")? != "metal"
        || router
            .get("metal_device_or_dispatch_performed")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("router receipt is not current strict-Metal component evidence".to_owned());
    }
    let router_binding = field_object(router, "artifact_binding", "router receipt")?;
    if field_string(
        router_binding,
        "manifest_document_sha256",
        "router artifact",
    )? != manifest_document_sha256
        || field_string(router_binding, "manifest_seal_sha256", "router artifact")?
            != manifest_seal_sha256
        || field_string(
            router_binding,
            "admission_receipt_seal_sha256",
            "router artifact",
        )? != admission_receipt_seal_sha256
        || field_string(router_binding, "source_revision", "router artifact")? != source_revision
    {
        return Err("router receipt artifact identity drifted".to_owned());
    }

    let (router_outer_receipt, router_outer_receipt_raw, router_outer_document) =
        read_json(&args.router_outer_receipt, "router outer receipt")?;
    let router_outer = object(&router_outer_document, "router outer receipt")?;
    if field_string(router_outer, "schema", "router outer receipt")? != ROUTER_OUTER_SCHEMA
        || field_string(router_outer, "status", "router outer receipt")? != ROUTER_OUTER_STATUS
    {
        return Err("router outer schema/status drifted".to_owned());
    }
    let router_outer_receipt_seal_sha256 =
        field_string(router_outer, "seal_sha256", "router outer receipt")?.to_owned();
    require_sha256(&router_outer_receipt_seal_sha256, "router outer seal")?;
    let router_inner = field_object(router_outer, "inner_probe_capture", "router outer receipt")?;
    if field_string(router_inner, "path", "router outer inner")? != router_receipt.to_string_lossy()
        || field_string(router_inner, "sha256", "router outer inner")?
            != sha256(&router_receipt_raw)
        || router_inner.get("metal_performed").and_then(Value::as_bool) != Some(true)
    {
        return Err("router outer does not bind supplied strict-Metal router receipt".to_owned());
    }

    let (route_plan, route_plan_raw, route_plan_document) =
        read_json(&args.route_plan, "route plan")?;
    let plan = object(&route_plan_document, "route plan")?;
    if field_string(plan, "schema", "route plan")? != ROUTE_PLAN_SCHEMA
        || field_string(plan, "status", "route plan")? != ROUTE_PLAN_STATUS
    {
        return Err("route plan schema/status drifted".to_owned());
    }
    let plan_manifest = field_object(plan, "manifest_descriptor_inventory", "route plan")?;
    if field_string(
        plan_manifest,
        "inventory_document_sha256",
        "route plan manifest",
    )? != manifest_document_sha256
        || field_string(plan_manifest, "manifest_seal_sha256", "route plan manifest")?
            != manifest_seal_sha256
    {
        return Err("route plan manifest identity drifted".to_owned());
    }
    let plan_router = field_object(plan, "router_evidence", "route plan")?;
    if field_string(
        plan_router,
        "inner_receipt_document_sha256",
        "route plan router",
    )? != sha256(&router_receipt_raw)
        || field_string(
            plan_router,
            "outer_receipt_document_sha256",
            "route plan router",
        )? != sha256(&router_outer_receipt_raw)
        || field_string(
            plan_router,
            "outer_receipt_seal_sha256",
            "route plan router",
        )? != router_outer_receipt_seal_sha256
    {
        return Err("route plan router identity drifted".to_owned());
    }
    let _ = string_array(plan_router, "source_stable_route_ids", "route plan router")?;
    let _ = number_array(
        plan_router,
        "source_stable_normalized_weights",
        "route plan router",
    )?;

    let (first_residual_receipt, first_residual_raw, first_residual_document) =
        read_json(&args.first_residual_receipt, "first-residual outer receipt")?;
    let first_residual = object(&first_residual_document, "first-residual outer receipt")?;
    if field_string(first_residual, "schema", "first-residual outer receipt")?
        != FIRST_RESIDUAL_OUTER_SCHEMA
        || field_string(first_residual, "status", "first-residual outer receipt")?
            != FIRST_RESIDUAL_OUTER_STATUS
    {
        return Err("first-residual outer schema/status drifted".to_owned());
    }
    let first_residual_seal_sha256 = field_string(
        first_residual,
        "seal_sha256",
        "first-residual outer receipt",
    )?
    .to_owned();
    require_sha256(&first_residual_seal_sha256, "first-residual outer seal")?;
    let first_source = field_object(
        first_residual,
        "source_binding",
        "first-residual outer receipt",
    )?;
    let first_manifest = field_object(first_source, "manifest", "first-residual source")?;
    let first_admission = field_object(first_source, "admission_current", "first-residual source")?;
    // Admission-current is a versioned mutable pointer.  The prefix receipt
    // retains its historical raw pointer facts, but a later bridge binds the
    // current pointer path plus the immutable selected manifest and admitted
    // receipt seal.  Do not require byte identity across harmless pointer
    // reseals; a changed path, manifest, or receipt seal remains fatal.
    let first_admission_path = canonical_regular(
        Path::new(field_string(
            first_admission,
            "path",
            "first-residual admission",
        )?),
        "first-residual historical admission path",
    )?;
    if first_admission_path != admission_current {
        return Err("first-residual outer admission pointer path drifted".to_owned());
    }
    require_sha256(
        field_string(
            first_source,
            "admission_pointer_seal_sha256",
            "first-residual source",
        )?,
        "first-residual historical admission pointer seal",
    )?;
    if field_string(first_manifest, "sha256", "first-residual manifest")?
        != manifest_document_sha256
        || field_string(
            first_source,
            "manifest_seal_sha256",
            "first-residual source",
        )? != manifest_seal_sha256
        || field_string(
            first_source,
            "admission_receipt_seal_sha256",
            "first-residual source",
        )? != admission_receipt_seal_sha256
    {
        return Err("first-residual outer artifact authority drifted".to_owned());
    }
    let first_output = field_object(
        first_residual,
        "first_residual_output",
        "first-residual outer receipt",
    )?;
    if first_output.get("layer").and_then(Value::as_u64) != Some(0)
        || first_output
            .get("linear_state_slot")
            .and_then(Value::as_u64)
            != Some(0)
        || first_output.get("elements").and_then(Value::as_u64) != Some(HIDDEN as u64)
        || first_output
            .get("same_command_graph_required")
            .and_then(Value::as_bool)
            != Some(true)
    {
        return Err("first-residual outer geometry/same-command-graph drifted".to_owned());
    }
    let first_residual_output_sha256 =
        field_string(first_output, "sha256", "first-residual output")?.to_owned();
    require_sha256(&first_residual_output_sha256, "first-residual output SHA")?;
    let first_residual_cpu_baseline = first_source
        .get("cpu_baseline_receipt")
        .cloned()
        .ok_or_else(|| "first-residual outer CPU baseline evidence missing".to_owned())?;

    Ok(Authority {
        manifest,
        manifest_raw,
        manifest_document_sha256,
        manifest_seal_sha256,
        admission_current,
        admission_raw,
        admission_pointer_seal_sha256,
        admission_receipt_seal_sha256,
        source_audit_seal_sha256,
        source_revision,
        router_receipt,
        router_receipt_raw,
        router_outer_receipt,
        router_outer_receipt_raw,
        router_outer_receipt_seal_sha256,
        route_plan,
        route_plan_raw,
        first_residual_receipt,
        first_residual_raw,
        first_residual_seal_sha256,
        first_residual_output_sha256,
        first_residual_cpu_baseline,
    })
}

fn write_new(path: &Path, document: &Value) -> Result<(), String> {
    let raw = serde_json::to_vec_pretty(document).map_err(|error| error.to_string())?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create --out {}: {error}", path.display()))?;
    file.write_all(&raw)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write --out {}: {error}", path.display()))
}

fn run(args: &Args) -> Result<Value, String> {
    let authority = validate_authority(args)?;
    let admission = CompleteBinaryAdmission {
        model: QwenCompleteBinaryModel::Qwen80CoderNext,
        expected_manifest_seal_sha256: authority.manifest_seal_sha256.clone(),
        expected_source_audit_seal_sha256: authority.source_audit_seal_sha256.clone(),
        expected_source_revision: authority.source_revision.clone(),
    };
    // One strict scan and then in-process catalog reuse.  This is the only
    // point at which the direct packed artifact is opened for this material.
    let catalog = Qwen80CompleteArtifactCatalog::load(&authority.manifest, &admission)
        .map_err(|error| format!("strict Qwen80 artifact admission failed: {error}"))?;
    let hybrid = catalog
        .complete_hybrid_decoder_plan(2)
        .map_err(|error| format!("complete hybrid decoder-plan bind failed: {error}"))?;
    let route_authority = Qwen80AllTenRoutedExpertPlanAuthority {
        manifest_document_sha256: &authority.manifest_document_sha256,
        plan_document_sha256: &sha256(&authority.route_plan_raw),
        router_receipt_sha256: &sha256(&authority.router_receipt_raw),
        router_outer_receipt_sha256: &sha256(&authority.router_outer_receipt_raw),
        router_outer_receipt_seal_sha256: &authority.router_outer_receipt_seal_sha256,
    };
    let route_document: Value =
        serde_json::from_slice(&authority.route_plan_raw).map_err(|error| {
            format!("cannot reparse route plan after authority validation: {error}")
        })?;
    let route_plan = hybrid
        .bind_all_ten_routed_expert_plan(0, &route_authority, &route_document)
        .map_err(|error| format!("strict all-ten route-plan bind failed: {error}"))?;
    let first_residual = catalog
        .first_residual_device_binding(0)
        .map_err(|error| format!("first-residual device binding failed: {error}"))?;
    let source_bridge = catalog
        .build_all_ten_true_moe_source_bridge(&route_plan, first_residual)
        .map_err(|error| format!("direct-packed all-ten source bridge build failed: {error}"))?;
    let bundle = source_bridge.route_payloads();
    let route = bundle.route();
    if bundle.layer() != 0
        || bundle.waves().len() != TOP_K
        || source_bridge.first_residual().layer() != 0
        || source_bridge.first_residual().linear_state_slot() != 0
        || source_bridge.first_residual().elements() != HIDDEN
        || !source_bridge.first_residual().same_command_graph_required()
    {
        return Err("source bridge geometry/order drifted".to_owned());
    }
    Ok(json!({
        "schema": MATERIAL_SCHEMA,
        "status": MATERIAL_STATUS,
        "source_binding": {
            "manifest": evidence(&authority.manifest, &authority.manifest_raw),
            "admission_current": evidence(&authority.admission_current, &authority.admission_raw),
            "router_receipt": evidence(&authority.router_receipt, &authority.router_receipt_raw),
            "router_outer_receipt": evidence(&authority.router_outer_receipt, &authority.router_outer_receipt_raw),
            "route_plan": evidence(&authority.route_plan, &authority.route_plan_raw),
            "first_residual_receipt": evidence(&authority.first_residual_receipt, &authority.first_residual_raw),
            "first_residual_cpu_baseline": authority.first_residual_cpu_baseline,
            "manifest_seal_sha256": authority.manifest_seal_sha256,
            "admission_pointer_seal_sha256": authority.admission_pointer_seal_sha256,
            "admission_receipt_seal_sha256": authority.admission_receipt_seal_sha256,
            "source_audit_seal_sha256": authority.source_audit_seal_sha256,
            "source_revision": authority.source_revision,
            "router_outer_receipt_seal_sha256": authority.router_outer_receipt_seal_sha256,
        },
        "typed_bridge": {
            "layer": 0,
            "route_count": TOP_K,
            "first_residual_elements": HIDDEN,
            "same_command_graph_required": true,
            "first_residual_output_sha256": authority.first_residual_output_sha256,
            "first_residual_receipt_seal_sha256": authority.first_residual_seal_sha256,
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
            "catalog_reused_for_all_ten_source_bridge": true,
            "raw_bf16_or_safetensors_opened": false,
        },
        "claim_boundary": {
            "metal_device_or_dispatch_performed": false,
            "no_complete_layer_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
            "material_requires_separate_sealed_component_lease_and_outer_reaped_capture": true,
        },
    }))
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("Qwen80 all-ten source bridge material refused: {error}");
        std::process::exit(2);
    });
    match run(&args) {
        Ok(document) => {
            if let Err(error) = write_new(&args.out, &document) {
                eprintln!("Qwen80 all-ten source bridge material refused: {error}");
                std::process::exit(2);
            }
            println!("{}", args.out.display());
        }
        Err(error) => {
            eprintln!("Qwen80 all-ten source bridge material refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_validator_rejects_noncanonical_values() {
        assert!(require_sha256(&"a".repeat(64), "test").is_ok());
        assert!(require_sha256(&"A".repeat(64), "test").is_err());
        assert!(require_sha256("abc", "test").is_err());
    }

    #[test]
    fn material_schema_is_explicitly_cpu_only() {
        assert!(MATERIAL_SCHEMA.contains("source_bridge_material"));
        assert!(MATERIAL_STATUS.contains("READY_FOR_SEAL"));
        assert_eq!(TOP_K, 10);
        assert_eq!(HIDDEN, 2048);
    }
}
