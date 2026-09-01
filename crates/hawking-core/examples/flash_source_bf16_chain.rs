//! Single-process Flash source-BF16 chain seam.
//!
//! This deliberately keeps layer state explicit on disk while removing the
//! subprocess boundary. It is a bridge toward a streamed resident executor,
//! not yet a claim of device-resident weights or complete-token TPS.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash chain requires macOS Metal").into())
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
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Instant;

    use super::{full, linear, terminal};

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
        let end: usize = arg_value(&argv, "--end-layer").unwrap_or_else(|| "3".into()).parse()?;
        if start > end || end >= 48 {
            return Err("expected 0 <= --start-layer <= --end-layer < 48".into());
        }
        let out_root = PathBuf::from(arg_value(&argv, "--out-dir").unwrap_or_else(|| {
            repo_root().join("receipts/headless/flash_chain").display().to_string()
        }));
        let run_terminal = argv.iter().any(|value| value == "--terminal") || end == 47;
        fs::create_dir_all(&out_root)?;
        let started = Instant::now();
        let base_state = arg_value(&argv, "--base-state").map(PathBuf::from);
        let mut previous_state: Option<PathBuf> = base_state.clone();
        let mut rows = Vec::new();

        for layer in start..=end {
            let layer_dir = out_root.join(format!("layer-{layer}"));
            fs::create_dir_all(&layer_dir)?;
            let receipt = layer_dir.join("receipt.json");
            let state = layer_dir.join("state.f32");
            eprintln!("Flash chain: layer {layer} ({}/{})", layer - start + 1, end - start + 1);
            if FULL_LAYERS.contains(&layer) {
                let args = full::Args {
                    root: root.clone(),
                    out: receipt.clone(),
                    layer,
                    base_state: previous_state.clone(),
                    state_out: Some(state.clone()),
                };
                full::run_layer(args)?;
            } else {
                let args = linear::Args {
                    root: root.clone(),
                    layer,
                    prefix_layers: 1,
                    warmup: 0,
                    reps: 1,
                    out: receipt.clone(),
                    state_out: state.clone(),
                    state_output: Some(state.clone()),
                    base_state: previous_state.clone(),
                };
                linear::run_layer(args)?;
            }
            if !state.is_file() {
                return Err(format!("layer {layer} did not emit {}", state.display()).into());
            }
            previous_state = Some(state.clone());
            rows.push(serde_json::json!({
                "layer": layer,
                "layer_type": if FULL_LAYERS.contains(&layer) { "full_attention" } else { "linear_attention" },
                "receipt": receipt,
                "state": state,
            }));
        }
        let terminal_receipt = if run_terminal && end == 47 {
            let state = previous_state.clone().ok_or("terminal requested without a final state")?;
            let receipt = out_root.join("terminal.json");
            eprintln!("Flash chain: native terminal readout");
            terminal::run_with(root.clone(), state, Some(receipt.clone()))?;
            Some(receipt)
        } else {
            None
        };
        let summary = serde_json::json!({
            "schema": "hawking.flash_source_bf16_single_process_chain.v1",
            "status": "PASSED",
            "root": root,
            "start_layer": start,
            "end_layer": end,
            "base_state": base_state.as_ref().map(|p| p.display().to_string()),
            "layers": rows,
            "terminal_receipt": terminal_receipt,
            "process_boundary": "single_os_process",
            "state_handoff": "explicit_f32_host_snapshot_between_layer_command_buffers",
            "device_residency": "UNQUALIFIED",
            "complete_token": "UNQUALIFIED",
            "elapsed_wall_ns": started.elapsed().as_nanos() as u64,
            "claim_boundary": "This proves only a single-process contiguous source-BF16 layer seam with explicit state files. It does not prove a streamed whole-token runtime, device-resident chain, TPS, EBPW, or HCLI residence."
        });
        let summary_path = out_root.join("CHAIN_SUMMARY.json");
        fs::write(&summary_path, serde_json::to_vec_pretty(&summary)?)?;
        println!("{}", serde_json::to_string_pretty(&summary)?);
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn Error>> {
        main_impl()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
