//! CPU/build host for N-layer Qwen80 same-runtime sequential hidden propagation.
//!
//! Extends the proven L0+L1 full-layer same-runtime pattern:
//! one runtime, one command buffer, ONE fence after all dispatches, caller-owned
//! per-layer state slots, structural kernel-name trace, receipt written last.
//!
//! `--layer-count N` selects layers `0..N` (e.g. 3 ⇒ L0+L1+L2 = 69 dispatches).
//! Physical capture is currently ready only for DeltaNet-only prefixes (no GQA).
//! Default CLI is preflight-only and never creates a Metal context or loads the
//! 148 GB body.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_source_token_multi_layer_same_runtime_device -- \
//!   --mode preflight --layer-count 3 \
//!   --execution-schedule-authority ABSOLUTE_SEALED_SCHEDULE \
//!   --chain-cpu-oracle ABSOLUTE_SEALED_CHAIN_ORACLE \
//!   --l1-full-layer-assessment ABSOLUTE_SEALED_L1_ASSESSMENT \
//!   --host-binary ABSOLUTE_CURRENT_HOST_BINARY \
//!   --out ABSOLUTE_NEW_JSON --workers 1
//! ```

use hawking_core::model::qwen80_48_layer_execution_schedule::{
    qwen80_layer_execution_schedule, qwen80_multi_layer_structural_kernel_trace,
    qwen80_multi_layer_total_dispatches, Qwen80ExecutionScheduleSourceBinding,
    QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA, QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
    QWEN80_DELTANET_FULL_LAYER_DISPATCHES, QWEN80_GRAVITY_MANIFEST_SEAL_SHA256, QWEN80_LAYERS,
    QWEN80_SOURCE_REVISION,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const HOST_PREFLIGHT_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_host_preflight.v1";
const HOST_PREFLIGHT_STATUS: &str =
    "COMPILED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_HOST_CPU_ONLY_NOT_LEASED_OR_EXECUTED";
const FUTURE_INNER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_capture.v1";
const FUTURE_INNER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_COMPONENT_ONLY";
const FUTURE_OUTER_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_outer_capture.v1";
const FUTURE_OUTER_STATUS: &str =
    "CAPTURED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_SAME_RUNTIME_OUTER_TERMINAL_COMPONENT_ONLY";
const FUTURE_RELEASE_SCHEMA: &str =
    "hawking.ascension.qwen80_source_token_multi_layer_same_runtime_quiet_metal_lease_release.v1";
const FUTURE_RELEASE_STATUS: &str =
    "RELEASED_QWEN80_SOURCE_TOKEN_MULTI_LAYER_COMPONENT_QUIET_METAL_LEASE_AFTER_TERMINAL_CAPTURE";
const L1_ASSESSMENT_SCHEMA: &str =
    "hawking.ascension.qwen80_l1_full_layer_completion_assessment.v1";
const L1_EARNED_STATUS: &str =
    "EARNED_QWEN80_SOURCE_TOKEN_L1_COMPLETE_LAYER_COMPONENT_NOT_TOKEN_DECODER";
const CHAIN_ORACLE_SCHEMA: &str = "hawking.ascension.qwen80_multi_layer_chain_cpu_oracle.v1";
const MAX_JSON_BYTES: u64 = 100_000_000;
const SOURCE_TOKEN_ID: u64 = 1;
/// First physical multi-layer capture: L0..L2 (three DeltaNet full layers).
const RECOMMENDED_FIRST_LAYER_COUNT: usize = 3;

#[derive(Debug)]
struct Args {
    layer_count: usize,
    execution_schedule_authority: PathBuf,
    chain_cpu_oracle: PathBuf,
    l1_full_layer_assessment: PathBuf,
    host_binary: PathBuf,
    out: PathBuf,
    workers: usize,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_source_token_multi_layer_same_runtime_device \\\n  --mode preflight --layer-count N \\\n  --execution-schedule-authority ABSOLUTE_SEALED_SCHEDULE \\\n  --chain-cpu-oracle ABSOLUTE_SEALED_CHAIN_ORACLE \\\n  --l1-full-layer-assessment ABSOLUTE_SEALED_L1_ASSESSMENT \\\n  --host-binary ABSOLUTE_CURRENT_HOST_BINARY \\\n  --out ABSOLUTE_NEW_JSON --workers 1..4\n\
metal mode is intentionally not default; the owner runs physical capture under resource admission after this preflight."
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_sha(value: &Value) -> Result<String, String> {
    Ok(sha256_hex(
        &serde_json::to_vec(value).map_err(|e| format!("canonicalize: {e}"))?,
    ))
}

fn seal(value: &mut Value) -> Result<String, String> {
    {
        let object = value
            .as_object_mut()
            .ok_or("document must be a JSON object")?;
        object.remove("seal_sha256");
    }
    let seal = canonical_json_sha(value)?;
    value
        .as_object_mut()
        .ok_or("document must be a JSON object")?
        .insert("seal_sha256".into(), json!(seal.clone()));
    Ok(seal)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err(format!("{} must be absolute", path.display()));
    }
    if path.exists() {
        return Err(format!("create-new required; {} exists", path.display()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create parent: {e}"))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| format!("serialize: {e}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("open: {e}"))?;
    file.write_all(&bytes).map_err(|e| format!("write: {e}"))?;
    file.sync_all().map_err(|e| format!("sync: {e}"))?;
    Ok(())
}

fn read_json(path: &Path, label: &str) -> Result<Value, String> {
    if !path.is_absolute() {
        return Err(format!("{label} must be absolute"));
    }
    let meta = fs::metadata(path).map_err(|e| format!("stat {label}: {e}"))?;
    if meta.len() > MAX_JSON_BYTES {
        return Err(format!(
            "{label} size observed={}, exceeds max={MAX_JSON_BYTES}",
            meta.len()
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("read {label}: {e}"))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {label}: {e}"))
}

fn file_sha(path: &Path, label: &str) -> Result<(u64, String), String> {
    let bytes = fs::read(path).map_err(|e| format!("read {label}: {e}"))?;
    Ok((bytes.len() as u64, sha256_hex(&bytes)))
}

fn verify_seal(value: &Value, label: &str) -> Result<String, String> {
    let mut unsigned = value.clone();
    let object = unsigned
        .as_object_mut()
        .ok_or_else(|| format!("{label} must be object"))?;
    let seal = object
        .remove("seal_sha256")
        .and_then(|v| v.as_str().map(str::to_owned))
        .ok_or_else(|| format!("{label} missing seal_sha256"))?;
    if seal.len() != 64 {
        return Err(format!(
            "{label} seal_sha256 length observed={}, expected=64",
            seal.len()
        ));
    }
    let expected = canonical_json_sha(&unsigned)?;
    if seal != expected {
        return Err(format!(
            "{label} seal mismatch: observed={seal}, expected={expected}"
        ));
    }
    Ok(seal)
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut mode = None;
    let mut layer_count = None;
    let mut execution_schedule_authority = None;
    let mut chain_cpu_oracle = None;
    let mut l1_full_layer_assessment = None;
    let mut host_binary = None;
    let mut out = None;
    let mut workers = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--mode" => {
                let value = args.next().ok_or("--mode requires a value")?;
                if mode.replace(value).is_some() {
                    return Err("--mode may not be repeated".into());
                }
            }
            "--layer-count" => {
                let raw = args.next().ok_or("--layer-count requires a value")?;
                let n: usize = raw
                    .parse()
                    .map_err(|_| format!("--layer-count must be integer, got {raw}"))?;
                if layer_count.replace(n).is_some() {
                    return Err("--layer-count may not be repeated".into());
                }
            }
            "--execution-schedule-authority" => {
                let value = args
                    .next()
                    .ok_or("--execution-schedule-authority requires a value")?;
                if execution_schedule_authority
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err("--execution-schedule-authority may not be repeated".into());
                }
            }
            "--chain-cpu-oracle" => {
                let value = args.next().ok_or("--chain-cpu-oracle requires a value")?;
                if chain_cpu_oracle.replace(PathBuf::from(value)).is_some() {
                    return Err("--chain-cpu-oracle may not be repeated".into());
                }
            }
            "--l1-full-layer-assessment" => {
                let value = args
                    .next()
                    .ok_or("--l1-full-layer-assessment requires a value")?;
                if l1_full_layer_assessment
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err("--l1-full-layer-assessment may not be repeated".into());
                }
            }
            "--host-binary" => {
                let value = args.next().ok_or("--host-binary requires a value")?;
                if host_binary.replace(PathBuf::from(value)).is_some() {
                    return Err("--host-binary may not be repeated".into());
                }
            }
            "--out" => {
                let value = args.next().ok_or("--out requires a value")?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out may not be repeated".into());
                }
            }
            "--workers" => {
                let raw = args.next().ok_or("--workers requires a value")?;
                let n: usize = raw
                    .parse()
                    .map_err(|_| format!("--workers must be integer, got {raw}"))?;
                if workers.replace(n).is_some() {
                    return Err("--workers may not be repeated".into());
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unsupported {other:?}; {}", usage())),
        }
    }
    let mode = mode.ok_or_else(|| format!("missing --mode; {}", usage()))?;
    if mode != "preflight" {
        return Err(format!(
            "--mode observed={mode}, expected=preflight (metal capture is owner-run under resource admission after this preflight earns)"
        ));
    }
    let layer_count = layer_count.ok_or_else(|| format!("missing --layer-count; {}", usage()))?;
    if layer_count < 2 || layer_count > QWEN80_LAYERS {
        return Err(format!(
            "--layer-count observed={layer_count}, expected in 2..={QWEN80_LAYERS} (multi-layer starts at L0+L1)"
        ));
    }
    let workers = workers.ok_or_else(|| format!("missing --workers; {}", usage()))?;
    if !(1..=4).contains(&workers) {
        return Err(format!("--workers observed={workers}, expected 1..=4"));
    }
    let require = |path: Option<PathBuf>, flag: &str| -> Result<PathBuf, String> {
        let path = path.ok_or_else(|| format!("missing {flag}; {}", usage()))?;
        if !path.is_absolute() {
            return Err(format!("{flag} must be absolute"));
        }
        Ok(path)
    };
    Ok(Args {
        layer_count,
        execution_schedule_authority: require(
            execution_schedule_authority,
            "--execution-schedule-authority",
        )?,
        chain_cpu_oracle: require(chain_cpu_oracle, "--chain-cpu-oracle")?,
        l1_full_layer_assessment: require(l1_full_layer_assessment, "--l1-full-layer-assessment")?,
        host_binary: require(host_binary, "--host-binary")?,
        out: require(out, "--out")?,
        workers,
    })
}

fn obj<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| format!("{label} must be object"))
}

fn text<'a>(map: &'a Map<String, Value>, field: &str, label: &str) -> Result<&'a str, String> {
    map.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{label}.{field} must be string"))
}

fn number(map: &Map<String, Value>, field: &str, label: &str) -> Result<u64, String> {
    map.get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.{field} must be unsigned integer"))
}

fn boolean(map: &Map<String, Value>, field: &str, expected: bool, label: &str) -> Result<(), String> {
    let observed = map
        .get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| format!("{label}.{field} must be bool"))?;
    if observed != expected {
        return Err(format!(
            "{label}.{field} observed={observed}, expected={expected}"
        ));
    }
    Ok(())
}

fn validate_schedule_authority(value: &Value, layer_count: usize) -> Result<String, String> {
    let seal = verify_seal(value, "execution schedule authority")?;
    let root = obj(value, "execution schedule authority")?;
    if text(root, "schema", "schedule")? != QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA {
        return Err(format!(
            "schedule schema observed={}, expected={QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA}",
            text(root, "schema", "schedule")?
        ));
    }
    if text(root, "status", "schedule")? != QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS {
        return Err(format!(
            "schedule status observed={}, expected={QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS}",
            text(root, "status", "schedule")?
        ));
    }
    let source = obj(
        root.get("source_authority")
            .ok_or("schedule missing source_authority")?,
        "schedule source_authority",
    )?;
    if text(source, "source_revision", "schedule")? != QWEN80_SOURCE_REVISION {
        return Err(format!(
            "schedule source_revision observed={}, expected={QWEN80_SOURCE_REVISION}",
            text(source, "source_revision", "schedule")?
        ));
    }
    if text(source, "gravity_manifest_seal_sha256", "schedule")?
        != QWEN80_GRAVITY_MANIFEST_SEAL_SHA256
    {
        return Err(format!(
            "schedule gravity seal observed={}, expected={QWEN80_GRAVITY_MANIFEST_SEAL_SHA256}",
            text(source, "gravity_manifest_seal_sha256", "schedule")?
        ));
    }
    let layers = root
        .get("layers")
        .and_then(Value::as_array)
        .ok_or("schedule.layers must be array")?;
    if layers.len() != QWEN80_LAYERS {
        return Err(format!(
            "schedule.layers len observed={}, expected={QWEN80_LAYERS}",
            layers.len()
        ));
    }
    for layer in 0..layer_count {
        let entry = obj(&layers[layer], &format!("schedule.layers[{layer}]"))?;
        let expected = qwen80_layer_execution_schedule(layer)?;
        let mixer = text(entry, "mixer", &format!("layer {layer}"))?;
        if mixer != expected.mixer.as_str() {
            return Err(format!(
                "schedule.layers[{layer}].mixer observed={mixer}, expected={}",
                expected.mixer.as_str()
            ));
        }
        if !expected.same_runtime_full_layer_encode_ready {
            return Err(format!(
                "layer {layer} mixer={} is not same-runtime full-layer encode ready (observed ready=false). Reduce --layer-count to a DeltaNet-only prefix (recommended {RECOMMENDED_FIRST_LAYER_COUNT} for L0..L2).",
                expected.mixer
            ));
        }
        let dispatch = number(
            entry,
            "full_layer_dispatch_count",
            &format!("layer {layer}"),
        )?;
        if dispatch != expected.full_layer_dispatch_count as u64 {
            return Err(format!(
                "schedule.layers[{layer}].full_layer_dispatch_count observed={dispatch}, expected={}",
                expected.full_layer_dispatch_count
            ));
        }
    }
    Ok(seal)
}

fn validate_chain_oracle(value: &Value, layer_count: usize) -> Result<String, String> {
    let seal = verify_seal(value, "chain cpu oracle")?;
    let root = obj(value, "chain cpu oracle")?;
    if text(root, "schema", "oracle")? != CHAIN_ORACLE_SCHEMA {
        return Err(format!(
            "oracle schema observed={}, expected={CHAIN_ORACLE_SCHEMA}",
            text(root, "schema", "oracle")?
        ));
    }
    let observed_count = number(root, "layer_count", "oracle")?;
    if observed_count != layer_count as u64 {
        return Err(format!(
            "oracle.layer_count observed={observed_count}, expected={layer_count}"
        ));
    }
    if root
        .get("includes_unready_gqa")
        .and_then(Value::as_bool)
        == Some(true)
    {
        return Err(format!(
            "oracle includes_unready_gqa=true for layer_count={layer_count}; physical multi-layer preflight requires a DeltaNet-only chain"
        ));
    }
    let total = number(root, "total_dispatches_physical_capture", "oracle").or_else(|_| {
        // composed oracle uses total_dispatches
        number(root, "total_dispatches", "oracle")
    })?;
    let expected = qwen80_multi_layer_total_dispatches(layer_count, false)?;
    if total != expected as u64 {
        return Err(format!(
            "oracle total_dispatches observed={total}, expected={expected}"
        ));
    }
    Ok(seal)
}

fn validate_l1_assessment(value: &Value) -> Result<String, String> {
    let seal = verify_seal(value, "L1 full-layer assessment")?;
    let root = obj(value, "L1 assessment")?;
    if text(root, "schema", "L1 assessment")? != L1_ASSESSMENT_SCHEMA {
        return Err(format!(
            "L1 assessment schema observed={}, expected={L1_ASSESSMENT_SCHEMA}",
            text(root, "schema", "L1 assessment")?
        ));
    }
    if text(root, "status", "L1 assessment")? != L1_EARNED_STATUS {
        return Err(format!(
            "L1 assessment status observed={}, expected={L1_EARNED_STATUS}",
            text(root, "status", "L1 assessment")?
        ));
    }
    boolean(root, "earned_complete_l1_component_only", true, "L1 assessment")?;
    Ok(seal)
}

fn build_preflight(args: &Args) -> Result<Value, String> {
    Qwen80ExecutionScheduleSourceBinding::exact().validate_exact()?;
    let schedule = read_json(
        &args.execution_schedule_authority,
        "execution schedule authority",
    )?;
    let schedule_seal = validate_schedule_authority(&schedule, args.layer_count)?;
    let oracle = read_json(&args.chain_cpu_oracle, "chain cpu oracle")?;
    let oracle_seal = validate_chain_oracle(&oracle, args.layer_count)?;
    let l1 = read_json(&args.l1_full_layer_assessment, "L1 full-layer assessment")?;
    let l1_seal = validate_l1_assessment(&l1)?;
    let (host_bytes, host_sha) = file_sha(&args.host_binary, "host binary")?;

    let expected_kernels = qwen80_multi_layer_structural_kernel_trace(args.layer_count, false)?;
    let total_dispatches = expected_kernels.len();
    if total_dispatches != args.layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES {
        return Err(format!(
            "total_dispatches drifted: observed={total_dispatches}, expected={}",
            args.layer_count * QWEN80_DELTANET_FULL_LAYER_DISPATCHES
        ));
    }

    let mut per_layer = Vec::new();
    for layer in 0..args.layer_count {
        let schedule = qwen80_layer_execution_schedule(layer)?;
        per_layer.push(json!({
            "layer": layer,
            "mixer": schedule.mixer.as_str(),
            "state_slot": schedule.state_slot.slot,
            "domain": schedule.state_slot.domain.as_str(),
            "dispatch_count": schedule.full_layer_dispatch_count,
            "kernel_names": schedule.full_layer_kernel_names,
            "exclusive_caller_owned_slot": true,
        }));
    }

    // Cumulative dispatch offsets for diagnostics.
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    for layer in 0..args.layer_count {
        offsets.push(json!({"layer": layer, "dispatch_offset": cursor, "dispatch_count": 23}));
        cursor += 23;
    }

    let mut document = json!({
        "schema": HOST_PREFLIGHT_SCHEMA,
        "status": HOST_PREFLIGHT_STATUS,
        "source_token_id": SOURCE_TOKEN_ID,
        "layer_count": args.layer_count,
        "layers_inclusive_range": {
            "first": 0,
            "last": args.layer_count - 1,
        },
        "source_authority": {
            "source_revision": QWEN80_SOURCE_REVISION,
            "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
        },
        "execution_schedule_authority": {
            "path": args.execution_schedule_authority.to_string_lossy(),
            "document_seal_sha256": schedule_seal,
            "document_sha256": schedule_seal,
        },
        "chain_cpu_oracle": {
            "path": args.chain_cpu_oracle.to_string_lossy(),
            "document_seal_sha256": oracle_seal,
            "document_sha256": oracle_seal,
        },
        "l1_full_layer_assessment_provenance": {
            "path": args.l1_full_layer_assessment.to_string_lossy(),
            "document_seal_sha256": l1_seal,
            "document_sha256": l1_seal,
            "historical_component_only": true,
            "does_not_import_pinned_buffers": true,
        },
        "host_binary": {
            "path": args.host_binary.to_string_lossy(),
            "bytes": host_bytes,
            "sha256": host_sha,
        },
        "workers": args.workers,
        "execution_policy": {
            "one_runtime": true,
            "one_command_buffer": true,
            "single_fence_after_all_dispatches": true,
            "fence_count": 1,
            "non_timed": true,
            "structural_kernel_trace_required": true,
            "receipt_written_last": true,
            "caller_owned_per_layer_state_slots": true,
            "bounded_cpu_device_parity_per_retained_vector": true,
            "retained_max_abs_error_in_receipt": true,
            "not_tolerance_scalar_in_isolation": true,
            "total_dispatches": total_dispatches,
            "per_layer_dispatch_count": QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "dispatch_offsets": offsets,
        },
        "structural_kernel_trace": {
            "exact_order": true,
            "kernel_names": expected_kernels,
        },
        "per_layer_schedule": per_layer,
        "future_capture_schemas": {
            "inner": FUTURE_INNER_SCHEMA,
            "inner_status": FUTURE_INNER_STATUS,
            "outer": FUTURE_OUTER_SCHEMA,
            "outer_status": FUTURE_OUTER_STATUS,
            "release": FUTURE_RELEASE_SCHEMA,
            "release_status": FUTURE_RELEASE_STATUS,
        },
        "recommended_first_physical_capture": {
            "layer_count": RECOMMENDED_FIRST_LAYER_COUNT,
            "layers": "L0..L2",
            "total_dispatches": RECOMMENDED_FIRST_LAYER_COUNT * QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "reason": "three sequential DeltaNet+MoE full layers scales the earned L0+L1 path by one layer without entering GQA L3, whose same-runtime full-layer encode is scheduled but not encode-ready",
        },
        "metal_path": {
            "preflight_only": true,
            "metal_context_or_dispatch_performed": false,
            "physical_capture_requires_owner_lease_and_admission": true,
            "encode_path": {
                "layer_0": "encode_source_input_l0_true_moe_capture (proven)",
                "layer_1": "encode_source_token_l1_deltanet_prefix + MoE suffix (proven)",
                "layer_2_plus_deltanet": "generalized next-layer DeltaNet full layer from previous second residual (wired at capture; one capture settles device parity)",
                "gqa_layers": "BLOCKED until same-runtime GQA full-layer encode is proven",
            },
        },
        "claim_boundary": {
            "host_preflight_only": true,
            "multi_layer_device_parity": false,
            "component_only": true,
            "token_generated": false,
            "decoder_started": false,
            "server_or_watcher_started": false,
            "tps_or_tg_measured": false,
            "tournament_started": false,
            "test_only_fake_child": false,
            "fixture_or_synthetic": false,
        },
        "refusal_diagnostics_contract": {
            "every_refusal_carries_observed_vs_expected_values": true,
            "generic_undifferentiated_errors_are_defects": true,
        },
    });
    seal(&mut document)?;
    Ok(document)
}

/// Expected cumulative kernel list used by unit tests and future assessor.
pub fn expected_multi_layer_kernels(layer_count: usize) -> Result<Vec<&'static str>, String> {
    qwen80_multi_layer_structural_kernel_trace(layer_count, false)
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(|args| {
        let document = build_preflight(&args)?;
        let seal = document["seal_sha256"]
            .as_str()
            .ok_or("missing seal")?
            .to_owned();
        write_new(&args.out, &document)?;
        Ok((args.layer_count, seal, args.out))
    }) {
        Ok((layer_count, seal, out)) => {
            println!(
                "{{\"status\":\"{HOST_PREFLIGHT_STATUS}\",\"layer_count\":{layer_count},\"seal_sha256\":\"{seal}\",\"out\":\"{}\"}}",
                out.display()
            );
        }
        Err(error) => {
            eprintln!(
                "ascension_qwen80_source_token_multi_layer_same_runtime_device refused: {error}"
            );
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_core::model::qwen80_48_layer_execution_schedule::{
        qwen80_all_48_layer_execution_schedules, validate_full_48_layer_schedule,
    };

    fn sealed_schedule() -> Value {
        let layers = qwen80_all_48_layer_execution_schedules().unwrap();
        validate_full_48_layer_schedule(&layers).unwrap();
        let layer_entries: Vec<Value> = layers
            .iter()
            .map(|layer| {
                json!({
                    "layer": layer.layer,
                    "mixer": layer.mixer.as_str(),
                    "source_layer_type": layer.source_layer_type,
                    "state_slot": {
                        "layer": layer.state_slot.layer,
                        "slot": layer.state_slot.slot,
                        "domain": layer.state_slot.domain.as_str(),
                        "device_buffers_required_before_execution": layer.state_slot.device_buffers_required_before_execution,
                        "rollback_buffers_required_before_execution": layer.state_slot.rollback_buffers_required_before_execution,
                        "exclusive_caller_owned_slot": true,
                    },
                    "mixer_prefix_dispatch_count": layer.mixer_prefix_dispatch_count,
                    "moe_suffix_dispatch_count": layer.moe_suffix_dispatch_count,
                    "full_layer_dispatch_count": layer.full_layer_dispatch_count,
                    "mixer_prefix_kernel_names": layer.mixer_prefix_kernel_names,
                    "moe_suffix_kernel_names": layer.moe_suffix_kernel_names,
                    "full_layer_kernel_names": layer.full_layer_kernel_names,
                    "residency": {
                        "input_hidden_elements": 2048,
                        "output_hidden_elements": 2048,
                        "mixer_compact_payloads_required": true,
                        "moe_fixed_compact_payloads_required": true,
                        "moe_routed_top10_compact_payloads_required": true,
                        "shared_expert_compact_payloads_required": true,
                        "state_slot_zeroed_or_caller_restored_before_encode": true,
                        "second_residual_is_next_layer_input": true,
                    },
                    "same_runtime_full_layer_encode_ready": layer.same_runtime_full_layer_encode_ready,
                })
            })
            .collect();
        let mut document = json!({
            "schema": QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA,
            "status": QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
            "source_authority": {
                "model_id": "Qwen3-Coder-Next-80B",
                "model_key": "qwen80",
                "source_repository": "Qwen/Qwen3-Coder-Next",
                "source_revision": QWEN80_SOURCE_REVISION,
                "source_config_sha256": "a7b8098d3b05777f12bb5677a26bf1240a1bb09def1b06b29e6be86cae2e84f8",
                "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
                "payload_schedule_authority_schema": "hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1",
                "full_attention_interval": 4,
                "layer_count": 48,
                "deltanet_layers": 36,
                "gqa_layers": 12,
            },
            "layers": layer_entries,
            "claim_boundary": {"execution_schedule_authority_only": true},
        });
        seal(&mut document).unwrap();
        document
    }

    fn sealed_oracle(layer_count: usize) -> Value {
        let trace = qwen80_multi_layer_structural_kernel_trace(layer_count, false).unwrap();
        let mut document = json!({
            "schema": CHAIN_ORACLE_SCHEMA,
            "status": "PREPARED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_STRUCTURE_NOT_NUMERIC_WITHOUT_LAYER_RECEIPTS",
            "layer_count": layer_count,
            "includes_unready_gqa": false,
            "total_dispatches_physical_capture": trace.len(),
            "structural_kernel_trace_physical_capture": trace,
            "claim_boundary": {"cpu_oracle_structure_only": true},
        });
        seal(&mut document).unwrap();
        document
    }

    fn sealed_l1() -> Value {
        let mut document = json!({
            "schema": L1_ASSESSMENT_SCHEMA,
            "status": L1_EARNED_STATUS,
            "earned_complete_l1_component_only": true,
            "component_scope": {
                "fresh_total_dispatches": 46,
                "full_layer_or_token_decoder_earned": false,
            },
        });
        seal(&mut document).unwrap();
        document
    }

    fn write_temp(dir: &Path, name: &str, value: &Value) -> PathBuf {
        let path = dir.join(name);
        fs::write(&path, serde_json::to_vec(value).unwrap()).unwrap();
        fs::canonicalize(&path).unwrap()
    }

    #[test]
    fn preflight_earns_for_layer_count_3() {
        let dir = tempfile::tempdir().unwrap();
        let schedule = write_temp(dir.path(), "schedule.json", &sealed_schedule());
        let oracle = write_temp(dir.path(), "oracle.json", &sealed_oracle(3));
        let l1 = write_temp(dir.path(), "l1.json", &sealed_l1());
        let host = write_temp(dir.path(), "host.bin", &json!({"binary": true}));
        let out_abs = fs::canonicalize(dir.path()).unwrap().join("out.json");
        let args = Args {
            layer_count: 3,
            execution_schedule_authority: schedule,
            chain_cpu_oracle: oracle,
            l1_full_layer_assessment: l1,
            host_binary: host,
            out: out_abs.clone(),
            workers: 1,
        };
        let document = build_preflight(&args).unwrap();
        assert_eq!(document["status"], HOST_PREFLIGHT_STATUS);
        assert_eq!(document["layer_count"], 3);
        assert_eq!(document["execution_policy"]["total_dispatches"], 69);
        assert_eq!(document["execution_policy"]["fence_count"], 1);
        assert_eq!(
            document["structural_kernel_trace"]["kernel_names"]
                .as_array()
                .unwrap()
                .len(),
            69
        );
        assert_eq!(document["claim_boundary"]["fixture_or_synthetic"], false);
        assert_eq!(document["claim_boundary"]["test_only_fake_child"], false);
        write_new(&out_abs, &document).unwrap();
        assert!(out_abs.exists());
    }

    #[test]
    fn preflight_refuses_layer_count_including_gqa_with_values() {
        let dir = tempfile::tempdir().unwrap();
        let schedule = write_temp(dir.path(), "schedule.json", &sealed_schedule());
        // Oracle that claims physical total for L0..L3 would be inconsistent;
        // build oracle that includes gqa flag.
        let mut oracle_doc = json!({
            "schema": CHAIN_ORACLE_SCHEMA,
            "status": "PREPARED",
            "layer_count": 4,
            "includes_unready_gqa": true,
            "total_dispatches_physical_capture": null,
            "claim_boundary": {},
        });
        seal(&mut oracle_doc).unwrap();
        let oracle = write_temp(dir.path(), "oracle.json", &oracle_doc);
        let l1 = write_temp(dir.path(), "l1.json", &sealed_l1());
        let host = write_temp(dir.path(), "host.bin", &json!({}));
        let args = Args {
            layer_count: 4,
            execution_schedule_authority: schedule,
            chain_cpu_oracle: oracle,
            l1_full_layer_assessment: l1,
            host_binary: host,
            out: fs::canonicalize(dir.path()).unwrap().join("out.json"),
            workers: 1,
        };
        let err = build_preflight(&args).unwrap_err();
        assert!(
            err.contains("layer 3") || err.contains("includes_unready_gqa") || err.contains("GQA") || err.contains("gqa") || err.contains("not same-runtime"),
            "{err}"
        );
    }

    #[test]
    fn expected_kernels_l0_l1_match_proven_46() {
        let kernels = expected_multi_layer_kernels(2).unwrap();
        assert_eq!(kernels.len(), 46);
        assert_eq!(kernels[0], "qwen_next_direct_packed_input_rmsnorm");
        assert_eq!(
            kernels[45],
            "qwen80_moe_wave_aggregate_second_residual_add_shared_residual"
        );
    }

    #[test]
    fn deltanet_only_layers_have_exclusive_slots() {
        use hawking_core::model::qwen80_48_layer_execution_schedule::Qwen80ExecutionMixerKind;
        for layer in 0..3 {
            let s = qwen80_layer_execution_schedule(layer).unwrap();
            assert!(matches!(s.mixer, Qwen80ExecutionMixerKind::DeltaNet));
            assert_eq!(s.state_slot.slot, layer);
            assert!(s.state_slot.exclusive_caller_owned_slot);
        }
    }
}
