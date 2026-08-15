//! Fail-closed readiness contract for the first real Qwen80 all-ten MoE graph.
//!
//! This is deliberately a CPU-only evidence/graph contract.  It consumes the
//! already sealed component records and says exactly why they cannot yet be
//! spliced into a production layer: the all-ten CPU oracle has ten source
//! witnesses but no device buffers, the shared component lacks a same-input
//! lineage, the old mixer record has no strict first-residual output witness,
//! and the combine receipt is explicitly a materialized fixture.  It defines
//! the only acceptable successor: one same-input graph ending in a real second
//! residual, not a re-labeling of those components as a full layer or token.

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

const SCHEMA: &str = "hawking.ascension.qwen80_true_input_all_ten_moe_graph_contract.v1";
const STATUS: &str = "REFUSED_QWEN80_TRUE_INPUT_ALL_TEN_MOE_GRAPH_NOT_READY";
const HIDDEN: usize = 2_048;
const TOP_K: usize = 10;

const ALL_TEN_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_all_ten_routed_expert_cpu_outer_launcher.v1";
const ALL_TEN_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_OUTER_TERMINAL_PRE_SHARED_PRE_COMBINE_PRE_RESIDUAL";
const ALL_TEN_INNER_SCHEMA: &str = "hawking.ascension.qwen80_all_ten_routed_expert_cpu_oracle.v1";
const ALL_TEN_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_ALL_TEN_ROUTED_EXPERT_CPU_ORACLE_READY_FOR_SEPARATE_DEVICE_LEASE";
const SHARED_OUTER_SCHEMA: &str = "hawking.ascension.qwen80_shared_expert_outer_launcher.v1";
const SHARED_OUTER_STATUS: &str = "CAPTURED_QWEN80_SHARED_EXPERT_OUTER_TERMINAL_COMPONENT_ONLY";
const SHARED_INNER_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_shared_expert_wave.v1";
const SHARED_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_DIRECT_PACKED_SHARED_EXPERT_STRICT_MATH_METAL_COMPONENT_NOT_ROUTED_MOE_OR_LAYER";
const COMBINE_OUTER_SCHEMA: &str = "hawking.ascension.qwen80_moe_combine_outer_launcher.v1";
const COMBINE_OUTER_STATUS: &str = "CAPTURED_QWEN80_MOE_COMBINE_OUTER_TERMINAL_COMPONENT_ONLY";
const COMBINE_INNER_SCHEMA: &str = "hawking.ascension.qwen80_direct_packed_moe_combine.v1";
// Historical fixture status. The real successor remains subject to the
// stricter `DIRECT_PACKED` provenance contract in the runtime; accepting this
// one here only lets the readiness report explain why it cannot be promoted.
const COMBINE_INNER_STATUS: &str =
    "EARNED_CURRENT_ADMITTED_QWEN80_MOE_COMBINE_SHARED_ADD_SECOND_RESIDUAL_STRICT_MATH_METAL_COMPONENT_NOT_COMPLETE_LAYER_OR_TOKEN";
const MIXER_SCHEMA: &str = "hawking.ascension.qwen80_layer0_deltanet_mixer_capture_receipt.v1";
const MIXER_STATUS: &str =
    "EARNED_QWEN80_ADMITTED_DIRECT_PACKED_LAYER0_DELTANET_MIXER_THROUGH_FIRST_RESIDUAL_NOT_COMPLETE_LAYER_OR_TOKEN";

#[derive(Debug)]
struct Args {
    all_ten_outer: PathBuf,
    all_ten_inner: PathBuf,
    shared_outer: PathBuf,
    shared_inner: PathBuf,
    combine_outer: PathBuf,
    combine_inner: PathBuf,
    first_residual_receipt: PathBuf,
    out: PathBuf,
}

#[derive(Debug)]
struct ReadDocument {
    path: PathBuf,
    raw_sha256: String,
    value: Value,
}

#[derive(Debug)]
struct AllTenFacts {
    manifest_document_sha256: String,
    route_plan_document_sha256: String,
    normalized_hidden_sha256: String,
    routed_sum_sha256: String,
    route_ids: Vec<u64>,
    route_weights: Vec<f64>,
    wave_output_sha256: Vec<String>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_true_input_moe_graph_contract \\
        --all-ten-outer ABS --all-ten-inner ABS \\
        --shared-outer ABS --shared-inner ABS \\
        --combine-outer ABS --combine-inner ABS \\
        --first-residual-receipt ABS --out NEW_ABSOLUTE_JSON"
}

fn parse_args() -> Result<Args, String> {
    let mut all_ten_outer = None;
    let mut all_ten_inner = None;
    let mut shared_outer = None;
    let mut shared_inner = None;
    let mut combine_outer = None;
    let mut combine_inner = None;
    let mut first_residual_receipt = None;
    let mut out = None;
    let mut iter = env::args().skip(1);
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}; {}", usage()))?;
        let slot = match flag.as_str() {
            "--all-ten-outer" => &mut all_ten_outer,
            "--all-ten-inner" => &mut all_ten_inner,
            "--shared-outer" => &mut shared_outer,
            "--shared-inner" => &mut shared_inner,
            "--combine-outer" => &mut combine_outer,
            "--combine-inner" => &mut combine_inner,
            "--first-residual-receipt" => &mut first_residual_receipt,
            "--out" => &mut out,
            _ => return Err(format!("unsupported argument {flag}; {}", usage())),
        };
        if slot.replace(value).is_some() {
            return Err(format!("argument {flag} was repeated; {}", usage()));
        }
    }
    let absolute = |value: Option<String>, label: &str| -> Result<PathBuf, String> {
        let path = PathBuf::from(value.ok_or_else(|| format!("missing {label}; {}", usage()))?);
        if !path.is_absolute() {
            return Err(format!("{label} must be absolute"));
        }
        Ok(path)
    };
    let out = absolute(out, "--out")?;
    if !out.parent().is_some_and(Path::is_dir) {
        return Err("--out parent must already exist".into());
    }
    Ok(Args {
        all_ten_outer: absolute(all_ten_outer, "--all-ten-outer")?,
        all_ten_inner: absolute(all_ten_inner, "--all-ten-inner")?,
        shared_outer: absolute(shared_outer, "--shared-outer")?,
        shared_inner: absolute(shared_inner, "--shared-inner")?,
        combine_outer: absolute(combine_outer, "--combine-outer")?,
        combine_inner: absolute(combine_inner, "--combine-inner")?,
        first_residual_receipt: absolute(first_residual_receipt, "--first-residual-receipt")?,
        out,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_value(value: &Value) -> Result<String, String> {
    serde_json::to_vec(value)
        .map(|bytes| sha256_hex(&bytes))
        .map_err(|error| format!("cannot serialize canonical JSON: {error}"))
}

fn canonical_regular(path: &Path, label: &str) -> Result<PathBuf, String> {
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

fn read_json(path: &Path, label: &str) -> Result<ReadDocument, String> {
    let path = canonical_regular(path, label)?;
    let bytes = fs::read(&path).map_err(|error| format!("cannot read {label}: {error}"))?;
    let value: Value = serde_json::from_slice(&bytes)
        .map_err(|error| format!("{label} is not valid JSON: {error}"))?;
    if !value.is_object() {
        return Err(format!("{label} is not a JSON object"));
    }
    Ok(ReadDocument {
        path,
        raw_sha256: sha256_hex(&bytes),
        value,
    })
}

fn object<'a>(
    value: &'a Value,
    field: &str,
    label: &str,
) -> Result<&'a Map<String, Value>, String> {
    value
        .get(field)
        .and_then(Value::as_object)
        .ok_or_else(|| format!("{label} lacks object {field:?}"))
}

fn object_array<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    label: &str,
) -> Result<&'a [Value], String> {
    value
        .get(field)
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{label} lacks array {field:?}"))
}

fn string<'a>(value: &'a Value, field: &str, label: &str) -> Result<&'a str, String> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| format!("{label} lacks non-empty string {field:?}"))
}

fn boolean(value: &Value, field: &str, label: &str) -> Result<bool, String> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} lacks boolean {field:?}"))
}

fn object_boolean(value: &Map<String, Value>, field: &str, label: &str) -> Result<bool, String> {
    value
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label} lacks boolean {field:?}"))
}

fn require(value: &Value, field: &str, expected: &str, label: &str) -> Result<(), String> {
    let observed = string(value, field, label)?;
    if observed != expected {
        return Err(format!(
            "{label} {field:?}={observed:?}, expected {expected:?}"
        ));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value.bytes().all(|byte| {
            byte.is_ascii_digit() || (byte.is_ascii_lowercase() && byte.is_ascii_hexdigit())
        })
}

fn require_sha256(value: &str, label: &str) -> Result<(), String> {
    if !is_sha256(value) {
        return Err(format!("{label} is not a lowercase SHA-256"));
    }
    Ok(())
}

/// Verify the exact canonical seal family used by `lab.receipts.seal`.  All
/// current evidence is ASCII JSON, so Python's sorted compact JSON and
/// serde_json's ordered compact serialization are byte-identical here.
fn verify_seal(document: &Value, label: &str) -> Result<String, String> {
    let object = document
        .as_object()
        .ok_or_else(|| format!("{label} is not an object"))?;
    let observed = object
        .get("seal_sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} lacks seal_sha256"))?;
    require_sha256(observed, &format!("{label} seal"))?;
    let mut unsigned = object.clone();
    unsigned.remove("seal_sha256");
    let expected = sha256_value(&Value::Object(unsigned))?;
    if expected != observed {
        return Err(format!("{label} seal mismatch"));
    }
    Ok(observed.to_owned())
}

fn file_evidence(document: &ReadDocument) -> Value {
    json!({
        "path": document.path,
        "present": true,
        "sha256": document.raw_sha256,
        "bytes": fs::metadata(&document.path).map(|metadata| metadata.len()).unwrap_or(0),
    })
}

fn outer_inner_evidence<'a>(
    binding: &'a Map<String, Value>,
    label: &str,
) -> Result<(&'a str, &'a str), String> {
    if binding.get("binding_valid") != Some(&Value::Bool(true)) {
        return Err(format!(
            "{label} does not attest a valid inner capture binding"
        ));
    }
    // The early all-ten launcher records path/SHA directly, while the shared
    // and combine launchers nest the same immutable evidence under `receipt`.
    // Accept both durable schemas, but never a path-only capture directory.
    let receipt = binding
        .get("receipt")
        .and_then(Value::as_object)
        .unwrap_or(binding);
    let path = receipt
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} lacks inner probe path"))?;
    let sha = receipt
        .get("sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label} lacks inner probe SHA"))?;
    require_sha256(sha, &format!("{label} inner probe SHA"))?;
    Ok((path, sha))
}

fn validate_outer(
    outer: &ReadDocument,
    inner: &ReadDocument,
    outer_schema: &str,
    outer_status: &str,
    label: &str,
) -> Result<String, String> {
    require(&outer.value, "schema", outer_schema, label)?;
    require(&outer.value, "status", outer_status, label)?;
    let seal = verify_seal(&outer.value, label)?;
    let binding = object(&outer.value, "inner_probe_capture", label)?;
    let (path, sha) = outer_inner_evidence(binding, label)?;
    if Path::new(path) != inner.path {
        return Err(format!(
            "{label} inner path does not bind supplied document"
        ));
    }
    if sha != inner.raw_sha256 {
        return Err(format!("{label} inner SHA does not bind supplied document"));
    }
    Ok(seal)
}

fn all_ten_facts(inner: &Value) -> Result<AllTenFacts, String> {
    require(inner, "schema", ALL_TEN_INNER_SCHEMA, "all-ten inner")?;
    require(inner, "status", ALL_TEN_INNER_STATUS, "all-ten inner")?;
    require(inner, "mode", "cpu-oracle", "all-ten inner")?;
    let artifact = object(inner, "artifact_binding", "all-ten inner")?;
    let manifest_document_sha256 = artifact
        .get("manifest_document_sha256")
        .and_then(Value::as_str)
        .ok_or("all-ten inner lacks manifest document SHA")?
        .to_owned();
    require_sha256(&manifest_document_sha256, "all-ten manifest SHA")?;
    let plan = object(inner, "route_plan_binding", "all-ten inner")?;
    let route_plan_document_sha256 = plan
        .get("document_sha256")
        .and_then(Value::as_str)
        .ok_or("all-ten inner lacks route plan SHA")?
        .to_owned();
    require_sha256(&route_plan_document_sha256, "all-ten plan SHA")?;
    let cpu = object(inner, "cpu_oracle", "all-ten inner")?;
    if cpu.get("all_ten_waves_executed") != Some(&Value::Bool(true)) {
        return Err("all-ten inner did not execute all ten CPU oracle waves".into());
    }
    let normalized_hidden_sha256 = cpu
        .get("normalized_hidden_sha256")
        .and_then(Value::as_str)
        .ok_or("all-ten inner lacks normalized hidden SHA")?
        .to_owned();
    let routed_sum_sha256 = cpu
        .get("routed_expert_sum_sha256")
        .and_then(Value::as_str)
        .ok_or("all-ten inner lacks routed sum SHA")?
        .to_owned();
    require_sha256(&normalized_hidden_sha256, "all-ten normalized hidden SHA")?;
    require_sha256(&routed_sum_sha256, "all-ten routed sum SHA")?;
    let ids = object_array(
        plan,
        "stable_source_route_ids",
        "all-ten route plan binding",
    )?;
    let weights = object_array(
        plan,
        "stable_source_route_weights",
        "all-ten route plan binding",
    )?;
    let waves = object_array(cpu, "waves", "all-ten CPU oracle")?;
    if ids.len() != TOP_K || weights.len() != TOP_K || waves.len() != TOP_K {
        return Err("all-ten source route/weights/witnesses do not each have ten entries".into());
    }
    let mut route_ids = Vec::with_capacity(TOP_K);
    let mut route_weights = Vec::with_capacity(TOP_K);
    let mut wave_output_sha256 = Vec::with_capacity(TOP_K);
    for index in 0..TOP_K {
        let id = ids[index]
            .as_u64()
            .ok_or_else(|| format!("all-ten route id {index} is invalid"))?;
        let weight = weights[index]
            .as_f64()
            .filter(|weight| weight.is_finite())
            .ok_or_else(|| format!("all-ten route weight {index} is invalid"))?;
        let wave = waves[index]
            .as_object()
            .ok_or_else(|| format!("all-ten witness {index} is not an object"))?;
        if wave.get("wave_index").and_then(Value::as_u64) != Some(index as u64)
            || wave.get("expert").and_then(Value::as_u64) != Some(id)
            || wave.get("down_output_elements").and_then(Value::as_u64) != Some(HIDDEN as u64)
        {
            return Err(format!(
                "all-ten witness {index} does not bind its exact source route"
            ));
        }
        let output_hash = wave
            .get("weighted_output_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("all-ten witness {index} lacks weighted output SHA"))?;
        require_sha256(output_hash, &format!("all-ten witness {index} output SHA"))?;
        route_ids.push(id);
        route_weights.push(weight);
        wave_output_sha256.push(output_hash.to_owned());
    }
    Ok(AllTenFacts {
        manifest_document_sha256,
        route_plan_document_sha256,
        normalized_hidden_sha256,
        routed_sum_sha256,
        route_ids,
        route_weights,
        wave_output_sha256,
    })
}

fn inner_metal_component(
    inner: &Value,
    schema: &str,
    status: &str,
    label: &str,
    expected_manifest: &str,
) -> Result<(), String> {
    require(inner, "schema", schema, label)?;
    require(inner, "status", status, label)?;
    require(inner, "mode", "metal", label)?;
    if !boolean(inner, "metal_device_or_dispatch_performed", label)? {
        return Err(format!("{label} lacks a real device dispatch"));
    }
    let artifact = object(inner, "artifact_binding", label)?;
    if artifact
        .get("manifest_document_sha256")
        .and_then(Value::as_str)
        != Some(expected_manifest)
    {
        return Err(format!(
            "{label} manifest identity differs from all-ten CPU oracle"
        ));
    }
    let metal = object(inner, "metal_intermediate_error_ledger", label)?;
    if !object_boolean(metal, "performed", label)? || !object_boolean(metal, "strict_math", label)?
    {
        return Err(format!("{label} lacks strict-Math component ledger"));
    }
    Ok(())
}

fn missing_true_input_gaps(
    all_ten: &AllTenFacts,
    shared_inner: &Value,
    combine_inner: &Value,
    mixer: &Value,
) -> Vec<&'static str> {
    let mut gaps = Vec::new();
    // The old mixer proves a bounded component but is intentionally not the
    // strict reusable first-residual witness required by the future graph.
    if mixer.get("schema").and_then(Value::as_str) != Some(MIXER_SCHEMA)
        || mixer.get("status").and_then(Value::as_str) != Some(MIXER_STATUS)
        || mixer.get("first_residual_output").is_some()
    {
        gaps.push("first_residual_needs_a_source_bound_2048_element_device_output_hash_and_strict_component_schema");
    } else {
        gaps.push("first_residual_receipt_has_no_machine_checkable_first_residual_output_hash_or_current_graph_input_link");
    }
    if shared_inner.get("input_provenance").is_none() {
        gaps.push(
            "shared_expert_component_has_no_first_residual_and_normalized_hidden_input_provenance",
        );
    }
    if combine_inner.get("materialized_source_route_shaped_fixture_only")
        != Some(&Value::Bool(false))
    {
        gaps.push(
            "current_combine_is_a_materialized_fixture_not_outputs_of_the_ten_real_route_bodies",
        );
    }
    if combine_inner.get("combine_inputs").is_none() {
        gaps.push("current_combine_has_no_all_ten_plan_first_residual_normalized_hidden_shared_and_ten_output_hash_join");
    }
    gaps.push("future_combine_must_use_the_direct_packed_real_moe_boundary_status_not_the_historical_fixture_component_status");
    // The all-ten CPU oracle deliberately emits only hashes.  That is enough
    // for a CPU witness but not for device parity on a single live body.
    if all_ten.wave_output_sha256.len() == TOP_K {
        gaps.push("all_ten_cpu_oracle_does_not_retain_same_capture_2048_element_route_buffers_for_device_parity");
    }
    gaps.push("no_generic_all_ten_direct_packed_metal_route_executor_has_produced_ten_same_input_device_witnesses");
    gaps.push("no_single_source_bound_command_graph_has_joined_first_residual_router_all_ten_shared_and_second_residual");
    gaps
}

fn graph_plan(all_ten: &AllTenFacts) -> Value {
    json!({
        "layer": 0,
        "identity": {
            "manifest_document_sha256": all_ten.manifest_document_sha256,
            "all_ten_route_plan_document_sha256": all_ten.route_plan_document_sha256,
            "source_route_ids": all_ten.route_ids,
            "source_normalized_weights": all_ten.route_weights,
        },
        "single_same_input_capture_required": true,
        "device_buffers": [
            {"name": "first_residual", "elements": HIDDEN, "producer": "direct-packed DeltaNet mixer", "must_hash_bind_router_shared_combine": true},
            {"name": "postnorm_hidden", "elements": HIDDEN, "producer": "post-attention RMSNorm(first_residual)", "must_hash_bind_all_route_and_shared_bodies": true},
            {"name": "route_ids", "elements": TOP_K, "producer": "router top-10", "must_equal_source_route": true},
            {"name": "route_weights", "elements": TOP_K, "producer": "router normalized top-10", "must_equal_source_route": true},
            {"name": "weighted_route_outputs", "shape": [TOP_K, HIDDEN], "producer": "generic direct-packed gate/up/SiLU/down for each selected route", "must_retain_each_vector_and_hash": true},
            {"name": "gated_shared", "elements": HIDDEN, "producer": "direct-packed shared expert plus sigmoid shared gate", "must_hash_bind_postnorm_hidden": true},
            {"name": "second_residual", "elements": HIDDEN, "producer": "fixed f32 r0..r9 then gated_shared then first_residual", "must_be_fenced_and_parity_checked": true},
        ],
        "fixed_order": [
            "source-bound DeltaNet mixer -> first_residual[2048]",
            "post-attention RMSNorm(first_residual) -> postnorm_hidden[2048]",
            "router(postnorm_hidden) -> exact ordered top10 ids/weights",
            "for route index 0..9: gate/up -> SiLU(gate)*up -> down -> source-normalized weight",
            "shared gate/up/down + sigmoid(shared_expert_gate(postnorm_hidden)) -> gated_shared[2048]",
            "for each hidden element: r0 + r1 + ... + r9 + gated_shared + first_residual -> second_residual",
        ],
        "future_receipt_requirements": {
            "one_admission_scan_then_catalog_reuse": true,
            "strict_math_non_timed_device_dispatch": true,
            "ten_route_witnesses_exactly_ordered_and_bound_to_the_all_ten_plan": true,
            "all_device_input_output_hashes_retained_in_one_durable_capture": true,
            "outer_reaped_stdout_stderr_exit_and_receipt_last": true,
            "no_full_layer_token_decoder_hcli_tps_or_tg_promotion": true,
        },
    })
}

fn seal_document(mut document: Value) -> Result<Value, String> {
    let object = document
        .as_object_mut()
        .ok_or("contract document must be an object")?;
    object.remove("seal_sha256");
    let seal = sha256_value(&Value::Object(object.clone()))?;
    object.insert("seal_sha256".into(), Value::String(seal));
    Ok(document)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if path.exists() {
        return Err(format!("refusing to overwrite {}", path.display()));
    }
    let bytes = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("cannot serialize contract: {error}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    file.write_all(&bytes)
        .and_then(|_| file.write_all(b"\n"))
        .and_then(|_| file.sync_all())
        .map_err(|error| format!("cannot write {}: {error}", path.display()))
}

fn run(args: &Args) -> Result<Value, String> {
    let all_ten_outer = read_json(&args.all_ten_outer, "all-ten outer receipt")?;
    let all_ten_inner = read_json(&args.all_ten_inner, "all-ten inner receipt")?;
    let shared_outer = read_json(&args.shared_outer, "shared outer receipt")?;
    let shared_inner = read_json(&args.shared_inner, "shared inner receipt")?;
    let combine_outer = read_json(&args.combine_outer, "combine outer receipt")?;
    let combine_inner = read_json(&args.combine_inner, "combine inner receipt")?;
    let mixer = read_json(&args.first_residual_receipt, "first-residual receipt")?;

    let all_ten_outer_seal = validate_outer(
        &all_ten_outer,
        &all_ten_inner,
        ALL_TEN_OUTER_SCHEMA,
        ALL_TEN_OUTER_STATUS,
        "all-ten outer receipt",
    )?;
    let shared_outer_seal = validate_outer(
        &shared_outer,
        &shared_inner,
        SHARED_OUTER_SCHEMA,
        SHARED_OUTER_STATUS,
        "shared outer receipt",
    )?;
    let combine_outer_seal = validate_outer(
        &combine_outer,
        &combine_inner,
        COMBINE_OUTER_SCHEMA,
        COMBINE_OUTER_STATUS,
        "combine outer receipt",
    )?;
    let all_ten = all_ten_facts(&all_ten_inner.value)?;
    inner_metal_component(
        &shared_inner.value,
        SHARED_INNER_SCHEMA,
        SHARED_INNER_STATUS,
        "shared inner receipt",
        &all_ten.manifest_document_sha256,
    )?;
    inner_metal_component(
        &combine_inner.value,
        COMBINE_INNER_SCHEMA,
        COMBINE_INNER_STATUS,
        "combine inner receipt",
        &all_ten.manifest_document_sha256,
    )?;
    let gaps = missing_true_input_gaps(
        &all_ten,
        &shared_inner.value,
        &combine_inner.value,
        &mixer.value,
    );
    Ok(seal_document(json!({
        "schema": SCHEMA,
        "status": STATUS,
        "mode": "cpu-only-readiness-contract",
        "recorded_by": "no_artifact_scan_no_metal_no_watcher_no_server",
        "evidence": {
            "all_ten_outer": {"file": file_evidence(&all_ten_outer), "seal_sha256": all_ten_outer_seal},
            "all_ten_inner": file_evidence(&all_ten_inner),
            "shared_outer": {"file": file_evidence(&shared_outer), "seal_sha256": shared_outer_seal},
            "shared_inner": file_evidence(&shared_inner),
            "combine_outer": {"file": file_evidence(&combine_outer), "seal_sha256": combine_outer_seal},
            "combine_inner": file_evidence(&combine_inner),
            "first_residual_receipt": file_evidence(&mixer),
        },
        "validated_current_facts": {
            "all_ten_cpu_source_bound_witnesses": true,
            "all_ten_route_ids": all_ten.route_ids.clone(),
            "all_ten_weighted_output_sha256": all_ten.wave_output_sha256.clone(),
            "all_ten_routed_sum_sha256": all_ten.routed_sum_sha256.clone(),
            "normalized_hidden_sha256": all_ten.normalized_hidden_sha256.clone(),
            "shared_component_has_real_strict_math_metal_dispatch": true,
            "combine_component_has_real_strict_math_metal_dispatch": true,
            "combine_is_explicit_materialized_fixture": combine_inner.value.get("materialized_source_route_shaped_fixture_only"),
            "current_fixture_combine_reusable_as_true_input_production_boundary": false,
            "true_input_graph_plan_defined_but_not_executed": true,
        },
        "true_input_graph_plan": graph_plan(&all_ten),
        "executor_gaps": gaps,
        "claim_boundary": {
            "current_records_do_not_prove_a_real_complete_moe_or_layer": true,
            "no_existing_fixture_or_component_is_promoted": true,
            "no_full_token_decoder_generation_hcli_tps_tg_or_tournament_claim": true,
        },
    }))?)
}

fn main() {
    let args = parse_args().unwrap_or_else(|error| {
        eprintln!("Qwen80 true-input MoE graph contract refused: {error}");
        process::exit(2);
    });
    match run(&args).and_then(|document| {
        write_new(&args.out, &document)?;
        Ok(document)
    }) {
        Ok(document) => println!("{}", serde_json::to_string(&document).unwrap()),
        Err(error) => {
            eprintln!("Qwen80 true-input MoE graph contract refused: {error}");
            process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seal(value: Value) -> Value {
        seal_document(value).unwrap()
    }

    #[test]
    fn seal_round_trip_matches_contract_verifier() {
        let value = seal(json!({"schema": "fixture", "status": "PASS", "ordered": [1, 2, 3]}));
        assert_eq!(verify_seal(&value, "fixture").unwrap().len(), 64);
    }

    #[test]
    fn outer_binding_accepts_direct_and_nested_receipt_evidence() {
        let sha = format!("{:064x}", 42);
        let direct = json!({
            "binding_valid": true,
            "path": "/tmp/inner.json",
            "sha256": sha.clone(),
        });
        let nested = json!({
            "binding_valid": true,
            "receipt": {"path": "/tmp/inner.json", "sha256": sha},
        });
        for value in [&direct, &nested] {
            let binding = value.as_object().unwrap();
            let (path, observed_sha) = outer_inner_evidence(binding, "fixture outer").unwrap();
            assert_eq!(path, "/tmp/inner.json");
            assert_eq!(observed_sha, format!("{:064x}", 42));
        }
    }

    #[test]
    fn all_ten_facts_refuse_a_missing_or_reordered_wave() {
        let mut waves = Vec::new();
        for index in 0..TOP_K {
            waves.push(json!({
                "wave_index": index,
                "expert": index + 10,
                "down_output_elements": HIDDEN,
                "weighted_output_sha256": format!("{:064x}", index + 1),
            }));
        }
        let mut value = json!({
            "schema": ALL_TEN_INNER_SCHEMA,
            "status": ALL_TEN_INNER_STATUS,
            "mode": "cpu-oracle",
            "artifact_binding": {"manifest_document_sha256": format!("{:064x}", 11)},
            "route_plan_binding": {
                "document_sha256": format!("{:064x}", 12),
                "stable_source_route_ids": (10..20).collect::<Vec<_>>(),
                "stable_source_route_weights": vec![0.1; TOP_K],
            },
            "cpu_oracle": {
                "all_ten_waves_executed": true,
                "normalized_hidden_sha256": format!("{:064x}", 13),
                "routed_expert_sum_sha256": format!("{:064x}", 14),
                "waves": waves,
            },
        });
        assert!(all_ten_facts(&value).is_ok());
        value["cpu_oracle"]["waves"][3]["wave_index"] = json!(9);
        assert!(all_ten_facts(&value).is_err());
    }

    #[test]
    fn current_style_fixture_is_explicitly_not_a_true_input_boundary() {
        let facts = AllTenFacts {
            manifest_document_sha256: format!("{:064x}", 1),
            route_plan_document_sha256: format!("{:064x}", 2),
            normalized_hidden_sha256: format!("{:064x}", 3),
            routed_sum_sha256: format!("{:064x}", 4),
            route_ids: (0..TOP_K as u64).collect(),
            route_weights: vec![0.1; TOP_K],
            wave_output_sha256: (0..TOP_K).map(|n| format!("{:064x}", n + 5)).collect(),
        };
        let gaps = missing_true_input_gaps(
            &facts,
            &json!({}),
            &json!({"materialized_source_route_shaped_fixture_only": true}),
            &json!({"schema": MIXER_SCHEMA, "status": MIXER_STATUS}),
        );
        assert!(gaps.iter().any(|gap| gap.contains("materialized_fixture")));
        assert!(gaps.iter().any(|gap| gap.contains("same_capture_2048")));
        assert!(gaps.iter().any(|gap| gap.contains("first_residual")));
    }
}
