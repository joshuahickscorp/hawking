//! Bounded two-token bridge: layers 0..2 linear -> layer 3 full organ -> layers 4..6 linear -> layer 7 full organ.
//!
//! This is intentionally not the complete model. It proves that the stateful
//! full-attention organ's final state can feed the next linear species while
//! recurrence remains alive across both token steps and a second attention
//! boundary can consume the resulting state.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash bridge requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;
#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod attention;
#[cfg(target_os = "macos")]
use sha2::{Digest, Sha256};

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use hawking_core::metal::{MetalContext, PinnedBuffer};
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde_json::json;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let argv: Vec<String> = env::args().collect();
    let value = |flag: &str| argv.windows(2).find(|p| p[0] == flag).map(|p| p[1].clone());
    let root = PathBuf::from(value("--root").unwrap_or_else(|| "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc".to_owned()));
    let out = PathBuf::from(value("--out").unwrap_or_else(|| "receipts/headless/FLASH_STATEFUL_LAYER3_LAYER7_BRIDGE.json".to_owned()));
    let token_ids = value("--tokens").unwrap_or_else(|| "248044,248044".to_owned())
        .split(',').map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()?;
    if token_ids.len() < 2 { return Err("bridge requires at least two tokens".into()); }

    let prefix_states = linear::capture_stateful_linear_prefix_outputs(root.clone(), &token_ids)?;
    let stem = out.file_stem().and_then(|s| s.to_str()).unwrap_or("FLASH_STATEFUL_LAYER3_LAYER11_BRIDGE");
    let attention_receipt = out.with_file_name(format!("{}_L3_ATTN.json", stem));
    let layer3_states = attention::run_stateful_attention_probe_from_states_with_outputs(
        root.clone(), 3, &token_ids, &prefix_states, attention_receipt.clone(),
    )?;
    if layer3_states.len() != token_ids.len() { return Err("layer-3 state count mismatch".into()); }

    let root = root.canonicalize()?;
    let index = SourceBf16Index::open(&root)?;
    let context = MetalContext::new_with_trace(true)?;
    let mut layers = (4..=6).map(|layer| linear::StatefulLinearLayer::new(&index, &context, layer, &layer3_states[0]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows = Vec::with_capacity(token_ids.len() * layers.len());
    let mut layer6_states = Vec::with_capacity(token_ids.len());
    for (step, layer3_state) in layer3_states.iter().enumerate() {
        let mut prior_device: Option<PinnedBuffer> = None;
        let mut layer6_state = None;
        for (offset, layer) in layers.iter_mut().enumerate() {
            let layer_index = 4 + offset;
            let input_state_sha256 = if offset == 0 {
                sha256(layer3_state)
            } else {
                let prior = prior_device.as_ref().ok_or("missing device linear handoff")?;
                let values = unsafe { std::slice::from_raw_parts(prior.contents() as *const f32, 2560 * 4).to_vec() };
                sha256(&values)
            };
            let (output, gpu_ns, wall_ns, dispatches, final_state) = if offset == 0 {
                layer.step(&context, Some(layer3_state), None, step == 0)?
            } else {
                let prior = prior_device.as_ref().ok_or("missing device linear handoff")?;
                layer.step(&context, None, Some(prior), step == 0)?
            };
            if offset == 2 { layer6_state = Some(final_state.clone()); }
            let digest = sha256(&final_state);
            rows.push(json!({"step": step, "token_id": token_ids[step], "layer": layer_index,
                "input_state_sha256": input_state_sha256,
                "final_state_sha256": digest, "gpu_ns": gpu_ns, "wall_ns": wall_ns,
                "dispatches": dispatches, "device_inter_layer_handoff": offset > 0,
                "finite": final_state.iter().all(|v| v.is_finite())}));
            prior_device = Some(output);
        }
        layer6_states.push(layer6_state.ok_or("layer-6 state was not produced")?);
    }
    let layer7_receipt = out.with_file_name(format!("{}_L7_ATTN.json", stem));
    let layer7_states = attention::run_stateful_attention_probe_from_states_with_outputs(
        root.clone(), 7, &token_ids, &layer6_states, layer7_receipt.clone(),
    )?;
    if layer7_states.len() != token_ids.len() { return Err("layer-7 state count mismatch".into()); }
    let mut layers_8_10 = (8..=10).map(|layer| linear::StatefulLinearLayer::new(&index, &context, layer, &layer7_states[0]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut rows_8_10 = Vec::with_capacity(token_ids.len() * layers_8_10.len());
    let mut layer10_states = Vec::with_capacity(token_ids.len());
    for (step, layer7_state) in layer7_states.iter().enumerate() {
        let mut prior_device: Option<PinnedBuffer> = None;
        let mut layer10_state = None;
        for (offset, layer) in layers_8_10.iter_mut().enumerate() {
            let layer_index = 8 + offset;
            let input_state_sha256 = if offset == 0 {
                sha256(layer7_state)
            } else {
                let prior = prior_device.as_ref().ok_or("missing device linear handoff")?;
                let values = unsafe { std::slice::from_raw_parts(prior.contents() as *const f32, 2560 * 4).to_vec() };
                sha256(&values)
            };
            let (output, gpu_ns, wall_ns, dispatches, final_state) = if offset == 0 {
                layer.step(&context, Some(layer7_state), None, step == 0)?
            } else {
                let prior = prior_device.as_ref().ok_or("missing device linear handoff")?;
                layer.step(&context, None, Some(prior), step == 0)?
            };
            if offset == 2 { layer10_state = Some(final_state.clone()); }
            rows_8_10.push(json!({"step": step, "token_id": token_ids[step], "layer": layer_index,
                "input_state_sha256": input_state_sha256, "final_state_sha256": sha256(&final_state),
                "gpu_ns": gpu_ns, "wall_ns": wall_ns, "dispatches": dispatches,
                "device_inter_layer_handoff": offset > 0, "finite": final_state.iter().all(|v| v.is_finite())}));
            prior_device = Some(output);
        }
        layer10_states.push(layer10_state.ok_or("layer-10 state was not produced")?);
    }
    let layer11_receipt = out.with_file_name(format!("{}_L11_ATTN.json", stem));
    let layer11_states = attention::run_stateful_attention_probe_from_states_with_outputs(
        root.clone(), 11, &token_ids, &layer10_states, layer11_receipt.clone(),
    )?;
    if layer11_states.len() != token_ids.len() { return Err("layer-11 state count mismatch".into()); }
    let mut receipt = json!({
        "schema": "hawking.flash.stateful_layer3_layer11_bridge.v3",
        "status": "PASSED_STATEFUL_CROSS_SPECIES_CHAIN",
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "token_ids": token_ids,
        "layer3": {"receipt": attention_receipt, "final_state_count": layer3_states.len(), "final_state_sha256": layer3_states.iter().map(|s| sha256(s)).collect::<Vec<_>>()},
        "linear_4_6": {"layers": [4, 5, 6], "species": "linear_attention", "stateful_recurrence": true, "layer6_final_state_sha256": layer6_states.iter().map(|s| sha256(s)).collect::<Vec<_>>(), "steps": rows},
        "layer7": {"receipt": layer7_receipt, "final_state_count": layer7_states.len(), "final_state_sha256": layer7_states.iter().map(|s| sha256(s)).collect::<Vec<_>>()},
        "linear_8_10": {"layers": [8, 9, 10], "species": "linear_attention", "stateful_recurrence": true, "layer10_final_state_sha256": layer10_states.iter().map(|s| sha256(s)).collect::<Vec<_>>(), "steps": rows_8_10},
        "layer11": {"receipt": layer11_receipt, "final_state_count": layer11_states.len(), "final_state_sha256": layer11_states.iter().map(|s| sha256(s)).collect::<Vec<_>>()},
        "execution": {"process_boundary": "one native process", "layer3_to_layer4_handoff": "host diagnostic copy", "layer4_to_layer6_handoffs": "device resident", "layer6_to_layer7_handoff": "host diagnostic copy", "layer7_to_layer8_handoff": "host diagnostic copy", "layer8_to_layer10_handoffs": "device resident", "layer10_to_layer11_handoff": "host diagnostic copy", "linear_context_reused": true, "linear_weights_reused": true},
        "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_layer3_layer11_bridge", "machine": context.device_name(), "rule": "S032 §3 -- bounded stateful bridge; no complete-token performance claim"},
        "accepted_generation_tokens": 0,
        "accepted_tps": null,
        "complete_system_ebpw": null,
        "promotion_allowed": false,
        "claim_boundary": "Two token steps traverse layers 0..2 stateful DeltaNet, layer 3 stateful full-attention KV plus MLP/MoE, layers 4..6 stateful DeltaNet with device inter-layer handoffs, layer 7 stateful full-attention KV plus MLP/MoE, layers 8..10 stateful DeltaNet with device inter-layer handoffs, and layer 11 stateful full-attention KV plus MLP/MoE. Cross-species seams remain host diagnostic copies; this does not prove device-only 48-layer integration, tokenizer acceptance, TPS, EBPW, or residency.",
        "next": "Continue the same state arena through the remaining structural boundaries, then replace the diagnostic species seams with device-resident handoff."
    });
    receipt["seal_sha256"] = serde_json::Value::String(sha256_bytes(&serde_json::to_vec(&receipt)?));
    if let Some(parent) = out.parent() { fs::create_dir_all(parent)?; }
    fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

#[cfg(target_os = "macos")]
fn sha256(values: &[f32]) -> String { sha256_bytes(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()) }

#[cfg(target_os = "macos")]
fn sha256_bytes(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }
