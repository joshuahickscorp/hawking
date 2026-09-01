//! Fast Flash continuation executor.
//!
//! This is the first production-shaped continuation path: one OS process,
//! one cached source index/Metal context for linear groups, and checkpoints
//! only at structural attention boundaries.  It deliberately keeps the deep
//! golden chain untouched and reports the remaining host-state handoff so the
//! next device-resident seam can be measured honestly.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash fast chain requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;
#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod full;
#[cfg(target_os = "macos")]
#[path = "flash_source_bf16_terminal.rs"]
mod terminal;

#[cfg(target_os = "macos")]
mod macos {
    use super::{full, linear, terminal};
    use hawking_core::metal::PinnedBuffer;
    use serde_json::{json, Value};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    const DEFAULT_ROOT: &str = "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const FULL_LAYERS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn arg_value(args: &[String], flag: &str) -> Option<String> {
        args.windows(2).find(|pair| pair[0] == flag).map(|pair| pair[1].clone())
    }

    fn main_impl() -> Result<(), Box<dyn Error>> {
        let argv: Vec<String> = env::args().collect();
        let root = PathBuf::from(arg_value(&argv, "--root").unwrap_or_else(|| {
            env::var("HCLI_FLASH_NEXT_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_owned())
        }));
        let start: usize = arg_value(&argv, "--start-layer").unwrap_or_else(|| "0".into()).parse()?;
        let end: usize = arg_value(&argv, "--end-layer").unwrap_or_else(|| "47".into()).parse()?;
        if start > end || end >= 48 { return Err("expected 0 <= start <= end < 48".into()); }
        let out_root = PathBuf::from(arg_value(&argv, "--out-dir").unwrap_or_else(|| {
            repo_root().join("receipts/headless/flash_fast_chain").display().to_string()
        }));
        let compact_experts = argv.iter().any(|arg| arg == "--compact-experts");
        let fused_route_accumulate = argv.iter().any(|arg| arg == "--fused-route-accumulate");
        let fused_attention_gate = argv.iter().any(|arg| arg == "--fused-attention-gate")
            || env::var("HAWKING_FLASH_FUSE_ATTENTION_GATE")
                .map(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
                .unwrap_or(false);
        let device_resident = argv.iter().any(|arg| arg == "--device-resident");
        let deep_verification = argv.iter().any(|arg| arg == "--deep-verification");
        fs::create_dir_all(&out_root)?;
        let started = Instant::now();
        let mut previous_state: Option<PathBuf> = arg_value(&argv, "--base-state").map(PathBuf::from);
        let mut groups = Vec::new();
        let mut previous_device_state: Option<PinnedBuffer> = None;
        let mut layer = start;
        while layer <= end {
            if FULL_LAYERS.contains(&layer) {
                let group_started = Instant::now();
                let dir = out_root.join(format!("layer-{layer}"));
                fs::create_dir_all(&dir)?;
                let receipt = dir.join("receipt.json");
                let state = dir.join("state.f32");
                eprintln!("Fast Flash: full-attention layer {layer}");
                let full_args = full::Args {
                    root: root.clone(), out: receipt.clone(), layer,
                    base_state: previous_state.clone(), state_out: Some(state.clone()),
                    compact_experts, fused_route_accumulate, fused_attention_gate,
                };
                if device_resident {
                    let prior_device_state = previous_device_state.take();
                    previous_device_state = full::run_layer_device_input(full_args, prior_device_state.as_ref())?;
                } else {
                    full::run_layer(full_args)?;
                }
                if !state.is_file() { return Err(format!("layer {layer} did not emit state").into()); }
                groups.push(json!({"kind":"full_attention","start_layer":layer,"end_layer":layer,"receipt":receipt,"state":state,"group_wall_ns":group_started.elapsed().as_nanos() as u64,"checkpoint":"one layer checkpoint"}));
                previous_state = Some(state);
                if !device_resident {
                    previous_device_state = None;
                }
                layer += 1;
                continue;
            }
            let mut run_end = end;
            for candidate in FULL_LAYERS {
                if candidate > layer { run_end = run_end.min(candidate - 1); break; }
            }
            let dir = out_root.join(format!("group-{layer}-{run_end}"));
            let group_started = Instant::now();
            fs::create_dir_all(&dir)?;
            let receipt = dir.join("receipt.json");
            let state = dir.join("state.f32");
            let count = run_end - layer + 1;
            eprintln!("Fast Flash: linear group {layer}..{run_end} ({count} layers)");
            let linear_args = linear::Args {
                root: root.clone(), layer, prefix_layers: count, warmup: 0, reps: 1,
                out: receipt.clone(), state_out: state.clone(), state_output: Some(state.clone()),
                base_state: previous_state.clone(), compact_experts,
                device_resident, deep_verification,
            };
            if device_resident {
                let prior_device_state = previous_device_state.take();
                previous_device_state = linear::run_layer_device_output(linear_args, prior_device_state.as_ref())?;
            } else {
                linear::run_layer(linear_args)?;
            }
            if !state.is_file() { return Err(format!("group {layer}..{run_end} did not emit state").into()); }
            let group_doc: Value = serde_json::from_slice(&fs::read(&receipt)?)?;
            let top_level_passed = group_doc.get("status").and_then(Value::as_str)
                .map(|status| status == "PASSED" || status.starts_with("PASSED_"))
                .unwrap_or(false);
            let rows_passed = group_doc.get("layers").and_then(Value::as_array)
                .map(|rows| rows.iter().all(|row| row.get("status").and_then(Value::as_str).map(|status| status == "PASSED" || status == "PASSED_TERMINAL_ONLY").unwrap_or(false)))
                .unwrap_or(false);
            let passed = top_level_passed || rows_passed;
            if !passed { return Err(format!("linear group {layer}..{run_end} failed parity").into()); }
            let layer_rows = group_doc.get("layers").cloned().unwrap_or_else(|| json!([{
                "layer": layer,
                "layer_type": "linear_attention",
                "status": "PASSED",
                "dispatches": group_doc.get("execution").and_then(|v| v.get("dispatches")).cloned().unwrap_or(Value::Null),
                "source_bytes_read": group_doc.get("bytes").and_then(|v| v.get("source_payload_bytes_read")).cloned().unwrap_or(Value::Null),
            }]));
            groups.push(json!({"kind":"linear_attention","start_layer":layer,"end_layer":run_end,"receipt":receipt,"state":state,"layers":layer_rows,"group_wall_ns":group_started.elapsed().as_nanos() as u64,"checkpoint":"one group checkpoint"}));
            previous_state = Some(state);
            layer = run_end + 1;
        }
        let terminal_receipt = if end == 47 {
            let state = previous_state.clone().ok_or("terminal requested without final state")?;
            let receipt = out_root.join("terminal.json");
            eprintln!("Fast Flash: native terminal readout");
            terminal::run_with(root.clone(), state, Some(receipt.clone()))?;
            Some(receipt)
        } else { None };
        let summary = json!({
            "schema":"hawking.flash_fast_chain.v1", "status":"PASSED", "root":root,
            "start_layer":start, "end_layer":end, "groups":groups,
            "compact_experts": compact_experts,
            "fused_attention_gate": fused_attention_gate,
            "source_bf16_vec4_candidate": hawking_core::env_on("HAWKING_FLASH_BF16_VEC4"),
            "source_bf16_geo_candidate": hawking_core::env_on("HAWKING_FLASH_BF16_GEO"),
            "source_bf16_geo_dual_candidate": hawking_core::env_on("HAWKING_FLASH_BF16_GEO_DUAL"),
            "device_resident": device_resident,
            "deep_verification_enabled": deep_verification,
            "terminal_receipt":terminal_receipt, "process_boundary":"single_os_process",
            "source_index":"cached in linear executor thread-local resource cache",
            "metal_context":"cached in linear executor thread-local resource cache",
            "state_handoff": if device_resident && deep_verification { "device final-state blit between structural groups; diagnostic host snapshots enabled, not required for activation" } else if device_resident { "device final-state blit between structural groups; host activation snapshots disabled in hot interval" } else { "host f32 checkpoint per structural group" },
            "deep_verification": if deep_verification { "all linear-group layers deep-verified with diagnostic reads; activation handoff remains device-resident" } else { "linear groups retain stage parity; production fast mode seam pending" },
            "elapsed_wall_ns":started.elapsed().as_nanos() as u64,
            "claim_boundary": if device_resident { "device-resident protected-shaped continuation; deep per-layer parity, complete token, and resident promotion remain open" } else { "fast grouped continuation reduces index/context and receipt ceremony but is not yet a device-resident complete-token runtime" }
        });
        fs::write(out_root.join("FAST_CHAIN_SUMMARY.json"), serde_json::to_vec_pretty(&summary)?)?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn Error>> { main_impl() }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> { macos::run() }
