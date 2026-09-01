//! Cross-species stateful Flash seam: linear prefix -> full-attention KV organ.
//!
//! Two token steps run in one process. Layers 0..2 retain their DeltaNet state;
//! their exact layer-2 outputs are fed into layer 3, whose KV cache retains two
//! positions. This is an integration receipt, not a complete-token runtime.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash seam requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;
#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod attention;

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    let argv: Vec<String> = env::args().collect();
    let value = |flag: &str| argv.windows(2).find(|p| p[0] == flag).map(|p| p[1].clone());
    let root = PathBuf::from(value("--root").unwrap_or_else(|| {
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc".to_owned()
    }));
    let out = PathBuf::from(value("--out").unwrap_or_else(|| {
        "receipts/headless/FLASH_STATEFUL_CROSS_SPECIES_SEAM.json".to_owned()
    }));
    let token_ids = value("--tokens").unwrap_or_else(|| "248044,248044".to_owned())
        .split(',').map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()?;
    if token_ids.len() < 2 { return Err("cross-species seam requires at least two tokens".into()); }

    let states = linear::capture_stateful_linear_prefix_outputs(root.clone(), &token_ids)?;
    let state_hashes: Vec<String> = states.iter().map(|state| {
        let mut hasher = Sha256::new();
        for value in state { hasher.update(value.to_le_bytes()); }
        format!("{:x}", hasher.finalize())
    }).collect();
    let attention_receipt = out.with_file_name(format!(
        "{}_ATTN.json",
        out.file_stem().and_then(|s| s.to_str()).unwrap_or("FLASH_STATEFUL_CROSS_SPECIES_SEAM")
    ));
    let final_states = attention::run_stateful_attention_probe_from_states_with_outputs(
        root.clone(), 3, &token_ids, &states,
        attention_receipt.clone(),
    )?;
    let final_state_hashes: Vec<String> = final_states.iter().map(|state| {
        let mut hasher = Sha256::new();
        for value in state { hasher.update(value.to_le_bytes()); }
        format!("{:x}", hasher.finalize())
    }).collect();
    let attention_doc: serde_json::Value = serde_json::from_slice(&fs::read(&attention_receipt)?)?;
    let mut receipt = json!({
        "schema": "hawking.flash.stateful_cross_species_seam.v1",
        "status": if attention_doc.get("status").and_then(|v| v.as_str()) == Some("PASSED_STATEFUL_KV_ORGAN") { "PASSED_CROSS_SPECIES_STATEFUL_SEAM" } else { "BLOCKED_ATTENTION_SEAM" },
        "model": "Qwen/Qwen3.8-Flash-Next",
        "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
        "token_ids": token_ids,
        "linear_prefix": {"layer_range": [0, 2], "state_count": states.len(), "layer2_state_sha256": state_hashes},
        "attention": {"layer": 3, "receipt": attention_receipt, "status": attention_doc.get("status"), "distinct_kv_slots": attention_doc.get("distinct_kv_slots"), "stateful_final_state": attention_doc.get("stateful_final_state"), "final_state_sha256": final_state_hashes, "full_attention_mlp_epilogue": attention_doc.get("execution").and_then(|v| v.get("full_attention_mlp_epilogue")), "first_step_parity": attention_doc.get("steps").and_then(|v| v.get(0)).and_then(|v| v.get("first_step_parity"))},
        "execution": {"process_boundary": "one native process", "device_inter_layer_handoff": "host diagnostic copy between the independently exposed module seams", "stateful_linear_recurrence": true, "stateful_attention_kv": true, "full_attention_mlp_epilogue": attention_doc.get("execution").and_then(|v| v.get("full_attention_mlp_epilogue"))},
        "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_cross_species_seam", "machine": "Apple M3 Ultra", "rule": "S032 §3 -- seam timing is diagnostic; quiescence unknown"},
        "accepted_generation_tokens": 0,
        "accepted_tps": null,
        "complete_system_ebpw": null,
        "promotion_allowed": false,
        "claim_boundary": "This proves a two-step cross-species seam: layers 0..2 DeltaNet state feeds layer 3 full attention with persistent KV slots and its HyperConnection/routed/shared-MoE MLP epilogue. The inter-module seam is still a host diagnostic copy; this does not prove device-only 48-layer integration, accepted generation, TPS, EBPW, or residency.",
        "next": "Feed the full-organ final state through the next qualified Flash layer in the persistent 48-layer session, then remove the diagnostic host seam."
    });
    let bytes = serde_json::to_vec_pretty(&receipt)?;
    let mut hasher = Sha256::new(); hasher.update(&bytes);
    receipt["seal_sha256"] = serde_json::Value::String(format!("{:x}", hasher.finalize()));
    if let Some(parent) = out.parent() { fs::create_dir_all(parent)?; }
    fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}
