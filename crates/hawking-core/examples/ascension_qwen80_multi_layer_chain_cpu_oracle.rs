//! CPU oracle for an N-layer Qwen80 same-runtime hidden chain.
//!
//! Chains per-layer full-layer oracle results so device parity is checked
//! against a real sequential reference, not against itself.  Default CLI is
//! schedule/structure only and never opens the 148 GB body.  Optional
//! `--layer-oracle-receipts` mode composes already-sealed per-layer CPU
//! oracle vectors into a chain receipt.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_multi_layer_chain_cpu_oracle -- \
//!   --layer-count 3 --out /absolute/new/QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE.json
//! ```

use hawking_core::model::qwen80_48_layer_execution_schedule::{
    qwen80_layer_execution_schedule, qwen80_multi_layer_structural_kernel_trace,
    qwen80_multi_layer_total_dispatches, Qwen80ExecutionMixerKind,
    Qwen80ExecutionScheduleSourceBinding, QWEN80_GRAVITY_MANIFEST_SEAL_SHA256, QWEN80_HIDDEN,
    QWEN80_LAYERS, QWEN80_SOURCE_REVISION,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const SCHEMA: &str = "hawking.ascension.qwen80_multi_layer_chain_cpu_oracle.v1";
const STATUS_STRUCTURE: &str =
    "PREPARED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_STRUCTURE_NOT_NUMERIC_WITHOUT_LAYER_RECEIPTS";
const STATUS_COMPOSED: &str =
    "COMPOSED_QWEN80_MULTI_LAYER_CHAIN_CPU_ORACLE_FROM_PER_LAYER_RECEIPTS_NOT_DEVICE";
const MAX_JSON_BYTES: u64 = 100_000_000;

#[derive(Debug)]
struct Args {
    layer_count: usize,
    out: PathBuf,
    layer_oracle_receipts: Option<PathBuf>,
}

fn usage() -> &'static str {
    "usage: ascension_qwen80_multi_layer_chain_cpu_oracle --layer-count N --out ABSOLUTE_NEW_JSON [--layer-oracle-receipts ABSOLUTE_JSON_ARRAY]"
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
            .ok_or("oracle document must be a JSON object")?;
        object.remove("seal_sha256");
    }
    let seal = canonical_json_sha(value)?;
    value
        .as_object_mut()
        .ok_or("oracle document must be a JSON object")?
        .insert("seal_sha256".into(), json!(seal.clone()));
    Ok(seal)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be absolute".into());
    }
    if path.exists() {
        return Err(format!("--out must be create-new; {} exists", path.display()));
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

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut layer_count = None;
    let mut out = None;
    let mut layer_oracle_receipts = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--layer-count" => {
                let raw = args
                    .next()
                    .ok_or_else(|| format!("--layer-count requires a value; {}", usage()))?;
                let n: usize = raw
                    .parse()
                    .map_err(|_| format!("--layer-count must be integer; got {raw}"))?;
                if layer_count.replace(n).is_some() {
                    return Err("--layer-count may not be repeated".into());
                }
            }
            "--out" => {
                let value = args
                    .next()
                    .ok_or_else(|| format!("--out requires a value; {}", usage()))?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err("--out may not be repeated".into());
                }
            }
            "--layer-oracle-receipts" => {
                let value = args.next().ok_or("--layer-oracle-receipts requires a value")?;
                if layer_oracle_receipts
                    .replace(PathBuf::from(value))
                    .is_some()
                {
                    return Err("--layer-oracle-receipts may not be repeated".into());
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unsupported {other:?}; {}", usage())),
        }
    }
    let layer_count = layer_count.ok_or_else(|| format!("missing --layer-count; {}", usage()))?;
    if layer_count == 0 || layer_count > QWEN80_LAYERS {
        return Err(format!(
            "--layer-count={layer_count} outside 1..={QWEN80_LAYERS}"
        ));
    }
    let out = out.ok_or_else(|| format!("missing --out; {}", usage()))?;
    if !out.is_absolute() {
        return Err("--out must be absolute".into());
    }
    Ok(Args {
        layer_count,
        out,
        layer_oracle_receipts,
    })
}

/// Structural chain plan: per-layer mixer, slot, kernels, and propagation rule.
fn build_structure_oracle(layer_count: usize) -> Result<Value, String> {
    let source = Qwen80ExecutionScheduleSourceBinding::exact();
    source.validate_exact()?;
    // Physical multi-layer capture refuses GQA until encode-ready; structure
    // oracle may describe scheduled GQA for documentation when layer_count
    // includes it, but marks numeric composition blocked.
    let mut layers = Vec::new();
    let mut includes_unready_gqa = false;
    for layer in 0..layer_count {
        let schedule = qwen80_layer_execution_schedule(layer)?;
        if !schedule.same_runtime_full_layer_encode_ready {
            includes_unready_gqa = true;
        }
        layers.push(json!({
            "layer": layer,
            "mixer": schedule.mixer.as_str(),
            "state_slot": schedule.state_slot.slot,
            "domain": schedule.state_slot.domain.as_str(),
            "full_layer_dispatch_count": schedule.full_layer_dispatch_count,
            "full_layer_kernel_names": schedule.full_layer_kernel_names,
            "input": if layer == 0 {
                "source_token_embedding_hidden"
            } else {
                "previous_layer_second_residual"
            },
            "output": "second_residual_hidden",
            "cpu_oracle_operator": match schedule.mixer {
                Qwen80ExecutionMixerKind::DeltaNet => {
                    "execute_canonical_linear_moe_cpu_oracle"
                }
                Qwen80ExecutionMixerKind::Gqa => {
                    "BLOCKED_UNTIL_SAME_RUNTIME_GQA_FULL_LAYER_CPU_ORACLE"
                }
            },
            "same_runtime_full_layer_encode_ready": schedule.same_runtime_full_layer_encode_ready,
            "propagation": {
                "previous_output_is_next_input": true,
                "hidden_elements": QWEN80_HIDDEN,
                "state_slots_never_shared": true,
            },
        }));
    }

    let physical_trace = if includes_unready_gqa {
        None
    } else {
        Some(qwen80_multi_layer_structural_kernel_trace(layer_count, false)?)
    };
    let physical_total = if includes_unready_gqa {
        None
    } else {
        Some(qwen80_multi_layer_total_dispatches(layer_count, false)?)
    };
    let scheduled_trace = qwen80_multi_layer_structural_kernel_trace(layer_count, true)?;
    let scheduled_total = scheduled_trace.len();

    let mut document = json!({
        "schema": SCHEMA,
        "status": STATUS_STRUCTURE,
        "source_authority": {
            "source_revision": QWEN80_SOURCE_REVISION,
            "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
        },
        "layer_count": layer_count,
        "layers": layers,
        "chain_rule": {
            "sequential_hidden_propagation": true,
            "layer_output_is_next_input": true,
            "embedding_seeds_layer_0_only": true,
            "independent_per_layer_state_slots": true,
            "device_parity_checked_against_this_cpu_oracle": true,
            "not_against_device_self": true,
        },
        "structural_kernel_trace_physical_capture": physical_trace,
        "total_dispatches_physical_capture": physical_total,
        "structural_kernel_trace_scheduled": scheduled_trace,
        "total_dispatches_scheduled": scheduled_total,
        "includes_unready_gqa": includes_unready_gqa,
        "numeric_layer_outputs_composed": false,
        "claim_boundary": {
            "cpu_oracle_structure_only": !includes_unready_gqa || true,
            "numeric_chain_composed": false,
            "device_parity": false,
            "metal_device_or_dispatch_performed": false,
            "artifact_payload_open_or_scan_performed": false,
            "decoder_or_token": false,
        },
    });
    seal(&mut document)?;
    Ok(document)
}

fn read_json(path: &Path) -> Result<Value, String> {
    if !path.is_absolute() {
        return Err(format!("{} must be absolute", path.display()));
    }
    let meta = fs::metadata(path).map_err(|e| format!("stat {}: {e}", path.display()))?;
    if meta.len() > MAX_JSON_BYTES {
        return Err(format!(
            "{} exceeds {MAX_JSON_BYTES} bytes (observed {})",
            path.display(),
            meta.len()
        ));
    }
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

/// Compose sealed per-layer oracle receipts into a chain.
///
/// Each receipt must carry:
/// - `layer`, `mixer`, `state_slot`
/// - `input_f32le_sha256`, `second_residual_f32le_sha256`
/// - optional `second_residual_f32` array for exact handoff check
fn compose_from_receipts(layer_count: usize, path: &Path) -> Result<Value, String> {
    let value = read_json(path)?;
    let receipts = value
        .as_array()
        .ok_or_else(|| {
            format!(
                "--layer-oracle-receipts must be a JSON array (observed type={})",
                match value {
                    Value::Object(_) => "object",
                    Value::String(_) => "string",
                    Value::Number(_) => "number",
                    Value::Bool(_) => "bool",
                    Value::Null => "null",
                    Value::Array(_) => "array",
                }
            )
        })?;
    if receipts.len() != layer_count {
        return Err(format!(
            "layer oracle receipt count observed={}, expected layer_count={layer_count}",
            receipts.len()
        ));
    }

    let mut composed_layers = Vec::new();
    let mut previous_output_sha: Option<String> = None;
    let mut previous_output_vector: Option<Vec<f32>> = None;
    let mut max_abs_handoff_error: f64 = 0.0;

    for (index, receipt) in receipts.iter().enumerate() {
        let object = receipt
            .as_object()
            .ok_or_else(|| format!("receipt[{index}] must be an object"))?;
        let layer = number(object, "layer", index)?;
        if layer != index as u64 {
            return Err(format!(
                "receipt[{index}].layer observed={layer}, expected={index}"
            ));
        }
        let schedule = qwen80_layer_execution_schedule(index)?;
        let mixer = text(object, "mixer", index)?;
        if mixer != schedule.mixer.as_str() {
            return Err(format!(
                "receipt[{index}].mixer observed={mixer}, expected={}",
                schedule.mixer.as_str()
            ));
        }
        let slot = number(object, "state_slot", index)?;
        if slot != schedule.state_slot.slot as u64 {
            return Err(format!(
                "receipt[{index}].state_slot observed={slot}, expected={}",
                schedule.state_slot.slot
            ));
        }
        let input_sha = text(object, "input_f32le_sha256", index)?.to_owned();
        let output_sha = text(object, "second_residual_f32le_sha256", index)?.to_owned();
        if input_sha.len() != 64 || output_sha.len() != 64 {
            return Err(format!(
                "receipt[{index}] sha fields must be 64 hex chars (input_len={}, output_len={})",
                input_sha.len(),
                output_sha.len()
            ));
        }
        if let Some(prev) = &previous_output_sha {
            if &input_sha != prev {
                return Err(format!(
                    "chain handoff broken at layer {index}: input_f32le_sha256 observed={input_sha}, expected previous second_residual_f32le_sha256={prev}"
                ));
            }
        }
        if let Some(vec) = object.get("second_residual_f32").and_then(Value::as_array) {
            let values: Result<Vec<f32>, String> = vec
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    v.as_f64()
                        .filter(|x| x.is_finite())
                        .map(|x| x as f32)
                        .ok_or_else(|| {
                            format!("receipt[{index}].second_residual_f32[{i}] not finite f32")
                        })
                })
                .collect();
            let values = values?;
            if values.len() != QWEN80_HIDDEN {
                return Err(format!(
                    "receipt[{index}].second_residual_f32 len observed={}, expected={QWEN80_HIDDEN}",
                    values.len()
                ));
            }
            if let Some(prev) = &previous_output_vector {
                let mut local_max = 0.0f32;
                for (a, b) in prev.iter().zip(values.iter()) {
                    local_max = local_max.max((a - b).abs());
                }
                // Handoff must be exact when vectors are present.
                if local_max != 0.0 {
                    return Err(format!(
                        "receipt[{index}] vector handoff not exact: max_abs_error observed={local_max}, expected=0.0"
                    ));
                }
                max_abs_handoff_error = max_abs_handoff_error.max(local_max as f64);
            }
            // Next layer's input should match this output — checked via sha;
            // keep vector for optional next comparison if next receipt has input vector.
            previous_output_vector = Some(values);
        }
        previous_output_sha = Some(output_sha.clone());
        composed_layers.push(json!({
            "layer": index,
            "mixer": mixer,
            "state_slot": slot,
            "input_f32le_sha256": input_sha,
            "second_residual_f32le_sha256": output_sha,
            "handoff_from_previous_exact": index == 0 || true,
        }));
    }

    let trace = qwen80_multi_layer_structural_kernel_trace(layer_count, false).map_err(|e| {
        format!(
            "cannot compose numeric chain for layer_count={layer_count} with unready GQA: {e}"
        )
    })?;

    let mut document = json!({
        "schema": SCHEMA,
        "status": STATUS_COMPOSED,
        "source_authority": {
            "source_revision": QWEN80_SOURCE_REVISION,
            "gravity_manifest_seal_sha256": QWEN80_GRAVITY_MANIFEST_SEAL_SHA256,
        },
        "layer_count": layer_count,
        "layers": composed_layers,
        "chain_final_second_residual_f32le_sha256": previous_output_sha,
        "max_abs_handoff_error_retained": max_abs_handoff_error,
        "structural_kernel_trace": trace,
        "total_dispatches": qwen80_multi_layer_total_dispatches(layer_count, false)?,
        "numeric_layer_outputs_composed": true,
        "claim_boundary": {
            "cpu_oracle_structure_only": false,
            "numeric_chain_composed": true,
            "device_parity": false,
            "metal_device_or_dispatch_performed": false,
            "artifact_payload_open_or_scan_performed": false,
            "decoder_or_token": false,
        },
    });
    seal(&mut document)?;
    Ok(document)
}

fn number(object: &Map<String, Value>, field: &str, index: usize) -> Result<u64, String> {
    object
        .get(field)
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("receipt[{index}].{field} must be unsigned integer"))
}

fn text<'a>(object: &'a Map<String, Value>, field: &str, index: usize) -> Result<&'a str, String> {
    object
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("receipt[{index}].{field} must be string"))
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(|args| {
        let document = if let Some(receipts) = &args.layer_oracle_receipts {
            compose_from_receipts(args.layer_count, receipts)?
        } else {
            build_structure_oracle(args.layer_count)?
        };
        let status = document["status"].as_str().unwrap_or("").to_owned();
        let seal = document["seal_sha256"].as_str().unwrap_or("").to_owned();
        write_new(&args.out, &document)?;
        Ok((status, seal, args.out))
    }) {
        Ok((status, seal, out)) => {
            println!(
                "{{\"status\":\"{status}\",\"seal_sha256\":\"{seal}\",\"out\":\"{}\"}}",
                out.display()
            );
        }
        Err(error) => {
            eprintln!("ascension_qwen80_multi_layer_chain_cpu_oracle refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structure_oracle_l0_l2() {
        let doc = build_structure_oracle(3).unwrap();
        assert_eq!(doc["layer_count"], 3);
        assert_eq!(doc["total_dispatches_physical_capture"], 69);
        assert_eq!(doc["includes_unready_gqa"], false);
        assert_eq!(doc["layers"][0]["mixer"], "delta_net");
        assert_eq!(doc["layers"][2]["state_slot"], 2);
        assert_eq!(
            doc["layers"][1]["input"],
            "previous_layer_second_residual"
        );
    }

    #[test]
    fn structure_oracle_marks_gqa_unready() {
        let doc = build_structure_oracle(4).unwrap();
        assert_eq!(doc["includes_unready_gqa"], true);
        assert!(doc["total_dispatches_physical_capture"].is_null());
        assert_eq!(doc["layers"][3]["mixer"], "gqa");
        assert_eq!(
            doc["layers"][3]["cpu_oracle_operator"],
            "BLOCKED_UNTIL_SAME_RUNTIME_GQA_FULL_LAYER_CPU_ORACLE"
        );
    }

    #[test]
    fn compose_detects_handoff_break_with_values() {
        let receipts = json!([
            {
                "layer": 0,
                "mixer": "delta_net",
                "state_slot": 0,
                "input_f32le_sha256": "a".repeat(64),
                "second_residual_f32le_sha256": "b".repeat(64),
            },
            {
                "layer": 1,
                "mixer": "delta_net",
                "state_slot": 1,
                "input_f32le_sha256": "c".repeat(64), // wrong: not previous b
                "second_residual_f32le_sha256": "d".repeat(64),
            },
        ]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("receipts.json");
        // path must be absolute for read_json
        let path = fs::canonicalize(&path).unwrap_or_else(|_| {
            fs::write(&path, serde_json::to_vec(&receipts).unwrap()).unwrap();
            fs::canonicalize(&path).unwrap()
        });
        if !path.exists() {
            fs::write(&path, serde_json::to_vec(&receipts).unwrap()).unwrap();
        }
        let abs = path;
        let err = compose_from_receipts(2, &abs).unwrap_err();
        assert!(err.contains("handoff broken"), "{err}");
        assert!(err.contains("expected previous"), "{err}");
    }

    #[test]
    fn compose_accepts_exact_chain() {
        let sha_a = "a".repeat(64);
        let sha_b = "b".repeat(64);
        let sha_c = "c".repeat(64);
        let receipts = json!([
            {
                "layer": 0,
                "mixer": "delta_net",
                "state_slot": 0,
                "input_f32le_sha256": sha_a,
                "second_residual_f32le_sha256": sha_b,
            },
            {
                "layer": 1,
                "mixer": "delta_net",
                "state_slot": 1,
                "input_f32le_sha256": sha_b,
                "second_residual_f32le_sha256": sha_c,
            },
        ]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.json");
        fs::write(&path, serde_json::to_vec(&receipts).unwrap()).unwrap();
        let abs = fs::canonicalize(&path).unwrap();
        let doc = compose_from_receipts(2, &abs).unwrap();
        assert_eq!(doc["status"], STATUS_COMPOSED);
        assert_eq!(doc["total_dispatches"], 46);
        assert_eq!(doc["chain_final_second_residual_f32le_sha256"], sha_c);
    }
}
