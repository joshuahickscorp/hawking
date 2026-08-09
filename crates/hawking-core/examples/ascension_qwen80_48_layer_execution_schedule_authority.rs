//! CPU-only Qwen80 48-layer *execution* schedule authority producer.
//!
//! Respects the sealed payload schedule authority
//! (`hawking.ascension.qwen80_48_layer_payload_schedule_authority.v1`) and
//! freezes the complementary per-layer runtime schedule: mixer kind, state
//! slot, kernel sequence, dispatch count, and residency requirements for all
//! 48 layers.  Never opens artifacts, creates Metal, or executes kernels.
//!
//! ```text
//! cargo run -p hawking-core --example ascension_qwen80_48_layer_execution_schedule_authority -- \
//!   --out /absolute/new/QWEN80_48_LAYER_EXECUTION_SCHEDULE_AUTHORITY.json
//! ```

use hawking_core::model::qwen80_48_layer_execution_schedule::{
    qwen80_all_48_layer_execution_schedules, validate_full_48_layer_schedule,
    Qwen80ExecutionMixerKind, Qwen80ExecutionScheduleSourceBinding,
    QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA, QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
    QWEN80_DELTANET_FULL_LAYER_DISPATCHES, QWEN80_DELTANET_LAYERS, QWEN80_GQA_LAYERS,
    QWEN80_GRAVITY_MANIFEST_SEAL_SHA256, QWEN80_LAYERS, QWEN80_SOURCE_REVISION,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

fn usage() -> &'static str {
    "usage: ascension_qwen80_48_layer_execution_schedule_authority --out ABSOLUTE_NEW_JSON"
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn canonical_json_sha(value: &Value) -> Result<String, String> {
    let bytes = serde_json::to_vec(value).map_err(|e| format!("canonicalize: {e}"))?;
    Ok(sha256_hex(&bytes))
}

fn seal(value: &mut Value) -> Result<String, String> {
    {
        let object = value
            .as_object_mut()
            .ok_or("execution schedule authority must be a JSON object")?;
        object.remove("seal_sha256");
    }
    let seal = canonical_json_sha(value)?;
    value
        .as_object_mut()
        .ok_or("execution schedule authority must be a JSON object")?
        .insert("seal_sha256".into(), json!(seal.clone()));
    Ok(seal)
}

fn write_new(path: &Path, value: &Value) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("--out must be absolute".into());
    }
    if path.exists() {
        return Err(format!("--out must be create-new; {} already exists", path.display()));
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("create parent: {e}"))?;
    }
    let bytes = serde_json::to_vec_pretty(value).map_err(|e| format!("serialize: {e}"))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("open --out: {e}"))?;
    file.write_all(&bytes).map_err(|e| format!("write --out: {e}"))?;
    file.sync_all().map_err(|e| format!("sync --out: {e}"))?;
    Ok(())
}

fn parse_args(mut args: impl Iterator<Item = String>) -> Result<PathBuf, String> {
    let mut out = None;
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--out" => {
                let value = args.next().ok_or_else(|| format!("--out requires a value; {}", usage()))?;
                if out.replace(PathBuf::from(value)).is_some() {
                    return Err(format!("--out may not be repeated; {}", usage()));
                }
            }
            "--help" | "-h" => return Err(usage().into()),
            other => return Err(format!("unsupported argument {other:?}; {}", usage())),
        }
    }
    out.ok_or_else(|| format!("missing --out; {}", usage()))
}

fn build_authority() -> Result<Value, String> {
    let source = Qwen80ExecutionScheduleSourceBinding::exact();
    source.validate_exact()?;
    let layers = qwen80_all_48_layer_execution_schedules()?;
    validate_full_48_layer_schedule(&layers)?;

    let deltanet_slots: Vec<Value> = layers
        .iter()
        .filter(|layer| {
            matches!(
                layer.mixer,
                hawking_core::model::qwen80_48_layer_execution_schedule::Qwen80ExecutionMixerKind::DeltaNet
            )
        })
        .map(|layer| {
            json!({
                "layer": layer.layer,
                "slot": layer.state_slot.slot,
                "domain": layer.state_slot.domain.as_str(),
                "exclusive_caller_owned_slot": true,
            })
        })
        .collect();
    let gqa_slots: Vec<Value> = layers
        .iter()
        .filter(|layer| {
            matches!(
                layer.mixer,
                hawking_core::model::qwen80_48_layer_execution_schedule::Qwen80ExecutionMixerKind::Gqa
            )
        })
        .map(|layer| {
            json!({
                "layer": layer.layer,
                "slot": layer.state_slot.slot,
                "domain": layer.state_slot.domain.as_str(),
                "exclusive_caller_owned_slot": true,
            })
        })
        .collect();

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
                    "exclusive_caller_owned_slot": layer.state_slot.exclusive_caller_owned_slot,
                },
                "mixer_prefix_dispatch_count": layer.mixer_prefix_dispatch_count,
                "moe_suffix_dispatch_count": layer.moe_suffix_dispatch_count,
                "full_layer_dispatch_count": layer.full_layer_dispatch_count,
                "mixer_prefix_kernel_names": layer.mixer_prefix_kernel_names,
                "moe_suffix_kernel_names": layer.moe_suffix_kernel_names,
                "full_layer_kernel_names": layer.full_layer_kernel_names,
                "residency": {
                    "input_hidden_elements": layer.residency.input_hidden_elements,
                    "output_hidden_elements": layer.residency.output_hidden_elements,
                    "mixer_compact_payloads_required": layer.residency.mixer_compact_payloads_required,
                    "moe_fixed_compact_payloads_required": layer.residency.moe_fixed_compact_payloads_required,
                    "moe_routed_top10_compact_payloads_required": layer.residency.moe_routed_top10_compact_payloads_required,
                    "shared_expert_compact_payloads_required": layer.residency.shared_expert_compact_payloads_required,
                    "state_slot_zeroed_or_caller_restored_before_encode": layer.residency.state_slot_zeroed_or_caller_restored_before_encode,
                    "second_residual_is_next_layer_input": layer.residency.second_residual_is_next_layer_input,
                },
                "same_runtime_full_layer_encode_ready": layer.same_runtime_full_layer_encode_ready,
            })
        })
        .collect();

    let total_all_48 = layers
        .iter()
        .map(|layer| layer.full_layer_dispatch_count)
        .sum::<usize>();
    if total_all_48 != QWEN80_LAYERS * QWEN80_DELTANET_FULL_LAYER_DISPATCHES {
        return Err(format!(
            "all-48 dispatch total drifted: observed={total_all_48}, expected={}",
            QWEN80_LAYERS * QWEN80_DELTANET_FULL_LAYER_DISPATCHES
        ));
    }

    let mut document = json!({
        "schema": QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA,
        "status": QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS,
        "source_authority": {
            "model_id": source.model_id,
            "model_key": source.model_key,
            "source_repository": source.source_repository,
            "source_revision": source.source_revision,
            "source_config_sha256": source.source_config_sha256,
            "gravity_manifest_seal_sha256": source.gravity_manifest_seal_sha256,
            "payload_schedule_authority_schema": source.payload_schedule_authority_schema,
            "full_attention_interval": source.full_attention_interval,
            "layer_count": source.layer_count,
            "deltanet_layers": source.deltanet_layers,
            "gqa_layers": source.gqa_layers,
        },
        "mixer_assignment_rule": {
            "description": "three Gated DeltaNet layers followed by one gated GQA; GQA when (layer+1) % 4 == 0",
            "full_attention_interval": 4,
            "deltanet_count": QWEN80_DELTANET_LAYERS,
            "gqa_count": QWEN80_GQA_LAYERS,
            "source": "Qwen3-Next config full_attention_interval + qwen80_layer_kind / payload schedule Mixer::expected",
        },
        "layers": layer_entries,
        "deltanet_state_slots": deltanet_slots,
        "gqa_state_slots": gqa_slots,
        "aggregate": {
            "layer_count": QWEN80_LAYERS,
            "deltanet_layers": QWEN80_DELTANET_LAYERS,
            "gqa_layers": QWEN80_GQA_LAYERS,
            "full_layer_dispatch_count_each": QWEN80_DELTANET_FULL_LAYER_DISPATCHES,
            "total_dispatches_all_48_layers": total_all_48,
            "same_runtime_deltanet_encode_ready_layer_count": layers
                .iter()
                .filter(|l| {
                    l.same_runtime_full_layer_encode_ready
                        && matches!(l.mixer, Qwen80ExecutionMixerKind::DeltaNet)
                })
                .count(),
            "same_runtime_gqa_encode_ready_layer_count": layers
                .iter()
                .filter(|l| {
                    l.same_runtime_full_layer_encode_ready
                        && matches!(l.mixer, Qwen80ExecutionMixerKind::Gqa)
                })
                .count(),
        },
        "multi_layer_host_parameterization": {
            "layer_count_parameter": "number of sequential layers starting at 0 (e.g. 3 => L0..L2)",
            "recommended_first_physical_capture_layer_count": 4,
            "recommended_first_physical_capture_reason": "L0..L3 is the first chain that crosses a GQA layer (layer 3): 3×DeltaNet + 1×GQA = 92 dispatches, one command buffer, one fence; GQA full-layer same-runtime encode is wired (physical parity is owner-run)",
            "single_fence_after_all_dispatches_required": true,
            "one_runtime_one_command_buffer_required": true,
            "caller_owned_per_layer_state_slots_required": true,
        },
        "respects_payload_schedule_authority": true,
        "does_not_duplicate_payload_tensor_inventory": true,
        "claim_boundary": {
            "execution_schedule_authority_only": true,
            "payload_schedule_authority": false,
            "multi_layer_device_parity": false,
            "decoder_readiness": false,
            "metal_device_or_dispatch_performed": false,
            "artifact_payload_open_or_scan_performed": false,
            "token_generation_or_feedback_performed": false,
            "tps_or_tg_measured": false,
            "execution_status": "PREPARED_NOT_EXECUTED",
        },
    });
    let seal_sha = seal(&mut document)?;
    // Hard identity pins for consumers that do not re-parse nested objects.
    if document["source_authority"]["source_revision"] != QWEN80_SOURCE_REVISION
        || document["source_authority"]["gravity_manifest_seal_sha256"]
            != QWEN80_GRAVITY_MANIFEST_SEAL_SHA256
    {
        return Err(format!(
            "sealed document lost source identity pins (revision/gravity); seal={seal_sha}"
        ));
    }
    Ok(document)
}

fn main() {
    match parse_args(env::args().skip(1)).and_then(|out| {
        let document = build_authority()?;
        let seal = document["seal_sha256"]
            .as_str()
            .ok_or("missing seal after build")?
            .to_owned();
        write_new(&out, &document)?;
        Ok((out, seal))
    }) {
        Ok((out, seal)) => {
            println!(
                "{{\"status\":\"{QWEN80_48_LAYER_EXECUTION_SCHEDULE_STATUS}\",\"seal_sha256\":\"{seal}\",\"out\":\"{}\"}}",
                out.display()
            );
        }
        Err(error) => {
            eprintln!("ascension_qwen80_48_layer_execution_schedule_authority refused: {error}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_core::model::qwen80_48_layer_execution_schedule::{
        qwen80_execution_mixer_kind, Qwen80ExecutionMixerKind,
    };

    #[test]
    fn authority_document_binds_source_and_all_48_layers() {
        let document = build_authority().unwrap();
        assert_eq!(
            document["schema"],
            QWEN80_48_LAYER_EXECUTION_SCHEDULE_SCHEMA
        );
        assert_eq!(
            document["source_authority"]["source_revision"],
            QWEN80_SOURCE_REVISION
        );
        assert_eq!(
            document["source_authority"]["gravity_manifest_seal_sha256"],
            QWEN80_GRAVITY_MANIFEST_SEAL_SHA256
        );
        let layers = document["layers"].as_array().unwrap();
        assert_eq!(layers.len(), 48);
        assert_eq!(layers[0]["mixer"], "delta_net");
        assert_eq!(layers[0]["state_slot"]["slot"], 0);
        assert_eq!(layers[1]["state_slot"]["slot"], 1);
        assert_eq!(layers[3]["mixer"], "gqa");
        assert_eq!(layers[3]["state_slot"]["slot"], 0);
        assert_eq!(layers[3]["same_runtime_full_layer_encode_ready"], true);
        assert_eq!(layers[0]["full_layer_dispatch_count"], 23);
        assert_eq!(
            document["aggregate"]["total_dispatches_all_48_layers"],
            48 * 23
        );
        assert_eq!(
            document["multi_layer_host_parameterization"]
                ["recommended_first_physical_capture_layer_count"],
            4
        );
        assert_eq!(
            document["aggregate"]["same_runtime_gqa_encode_ready_layer_count"],
            12
        );
        assert_eq!(
            document["aggregate"]["same_runtime_deltanet_encode_ready_layer_count"],
            36
        );
        // Seal integrity.
        let mut unsigned = document.clone();
        unsigned.as_object_mut().unwrap().remove("seal_sha256");
        let expected = canonical_json_sha(&unsigned).unwrap();
        assert_eq!(document["seal_sha256"], expected);
    }

    #[test]
    fn mixer_counts_match_source_rule() {
        let mut dn = 0;
        let mut gqa = 0;
        for layer in 0..48 {
            match qwen80_execution_mixer_kind(layer).unwrap() {
                Qwen80ExecutionMixerKind::DeltaNet => dn += 1,
                Qwen80ExecutionMixerKind::Gqa => gqa += 1,
            }
        }
        assert_eq!((dn, gqa), (36, 12));
    }
}
