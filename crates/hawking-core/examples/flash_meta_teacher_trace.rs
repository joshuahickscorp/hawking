//! Capture a semantically aligned Flash meta-representation teacher surface.
//!
//! This is a source-authority diagnostic capture, not a student, packer, or
//! benchmark.  It runs a dense source-BF16 stateful prefix through layer 3,
//! then runs dense source-BF16 layer 4 and writes the exact post-HyperConnection
//! `mlp_input` vectors consumed by layer-4 routing and `gate_up_proj`.  The
//! resulting rows are suitable input to `tools/flash_meta_coherence_screen.py`.
//! Candidate geometry/fusion environment variables are rejected so the teacher
//! corpus cannot silently mix an accelerator candidate with the authority path.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash meta teacher capture requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod attention;
#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;

#[cfg(target_os = "macos")]
mod macos {
    use super::{attention, linear};
    use hawking_core::metal::MetalContext;
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;
    use std::env;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const DEFAULT_ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const DEFAULT_OUT: &str = "receipts/headless/FLASH_META_TEACHER_L4.json";
    const DEFAULT_STATE_OUT: &str = "receipts/headless/FLASH_META_TEACHER_L4_MLP_INPUT.f32";
    const MIN_CORPUS_ROWS: usize = 256;
    const MAX_ROWS: usize = 4096;

    fn repository_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn arg_value(args: &[String], flag: &str) -> Option<String> {
        args.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].clone())
    }

    fn sha256_bytes(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn flatten_rows(rows: &[Vec<f32>]) -> Vec<u8> {
        let elements: usize = rows.iter().map(Vec::len).sum();
        let mut bytes = Vec::with_capacity(elements * std::mem::size_of::<f32>());
        for row in rows {
            bytes.extend_from_slice(&f32_bytes(row));
        }
        bytes
    }

    fn parse_tokens(argv: &[String]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
        if let Some(raw) = arg_value(argv, "--tokens") {
            let tokens = raw
                .split(',')
                .filter(|value| !value.trim().is_empty())
                .map(|value| value.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?;
            if tokens.len() < 2 {
                return Err("--tokens requires at least two comma-separated IDs".into());
            }
            if tokens.iter().collect::<BTreeSet<_>>().len() != tokens.len() {
                return Err(
                    "--tokens must contain unique token IDs; duplicate rows are not a corpus"
                        .into(),
                );
            }
            return Ok(tokens);
        }
        let start = arg_value(argv, "--token-start")
            .unwrap_or_else(|| "0".to_owned())
            .parse::<usize>()?;
        let count = arg_value(argv, "--count")
            .unwrap_or_else(|| "256".to_owned())
            .parse::<usize>()?;
        if count < 2 || count > MAX_ROWS {
            return Err(format!("--count must be in 2..={MAX_ROWS}").into());
        }
        let end = start
            .checked_add(count)
            .ok_or("token range overflowed usize")?;
        Ok((start..end).collect())
    }

    fn authority_environment_is_clean() -> Result<(), Box<dyn std::error::Error>> {
        const CANDIDATES: &[&str] = &[
            "HAWKING_FLASH_BF16_GEO",
            "HAWKING_FLASH_BF16_VEC4",
            "HAWKING_FLASH_BF16_GEO_DUAL",
            "HAWKING_FLASH_MOE_GEO",
            "HAWKING_FLASH_MOE_VEC4",
            "HAWKING_FLASH_QKV_GQA_FUSED",
            "HAWKING_FLASH_FUSE_ATTENTION_GATE",
            "HAWKING_FLASH_ROUTER_TOPK_FUSED",
            "HAWKING_FLASH_HC_ROUTER_FUSED",
        ];
        let enabled: Vec<&str> = CANDIDATES
            .iter()
            .copied()
            .filter(|key| {
                env::var(key)
                    .map(|value| {
                        matches!(
                            value.trim().to_ascii_lowercase().as_str(),
                            "1" | "true" | "on" | "yes"
                        )
                    })
                    .unwrap_or(false)
            })
            .collect();
        if !enabled.is_empty() {
            return Err(format!(
                "teacher capture requires source-authority controls off; enabled candidates: {enabled:?}"
            )
            .into());
        }
        Ok(())
    }

    fn write_capture_boundary(
        out: &PathBuf,
        root: &PathBuf,
        requested_rows: usize,
        started: &Instant,
        error: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let boundary = out.with_file_name("FLASH_META_TEACHER_L4_CAPTURE_BOUNDARY.json");
        let mut receipt = json!({
            "schema": "hawking.flash.meta_teacher_trace_boundary.v1",
            "status": "BLOCKED_NO_METAL_GPU",
            "model": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "source_root_requested": root,
            "requested_rows": requested_rows,
            "minimum_rows": MIN_CORPUS_ROWS,
            "failure": {
                "stage": "dense_source_bf16_prefix_initialization",
                "error": error,
            },
            "teacher_rows_written": 0,
            "source_authority_capture": false,
            "promotion_allowed": false,
            "physical_ebpw": null,
            "claim_boundary": "The required dense source-BF16 teacher capture could not start because this host has no Metal-capable GPU. No synthetic rows, coherence result, meta artifact, physical EBPW, or latency result is claimed.",
            "bench": {
                "state": "UNKNOWN",
                "recorded_by": "flash_meta_teacher_trace",
                "elapsed_ns": started.elapsed().as_nanos() as u64,
            },
        });
        receipt["seal_sha256"] = json!(sha256_bytes(&serde_json::to_vec(&receipt)?));
        if let Some(parent) = boundary
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(&boundary, serde_json::to_vec_pretty(&receipt)?)?;
        Ok(boundary)
    }

    fn main_impl() -> Result<(), Box<dyn std::error::Error>> {
        let argv: Vec<String> = env::args().collect();
        if argv.iter().any(|arg| arg == "--help" || arg == "-h") {
            println!(
                "usage: flash_meta_teacher_trace [--root DIR] [--tokens ID,ID,... | --token-start N --count N] [--out FILE] [--state-out F32]"
            );
            return Ok(());
        }
        authority_environment_is_clean()?;
        let root =
            PathBuf::from(arg_value(&argv, "--root").unwrap_or_else(|| DEFAULT_ROOT.to_owned()));
        let tokens = parse_tokens(&argv)?;
        if tokens.len() < MIN_CORPUS_ROWS {
            return Err(format!(
                "teacher corpus requires at least {MIN_CORPUS_ROWS} unique token rows; refusing a small probe"
            ).into());
        }
        let repo = repository_root();
        let out = PathBuf::from(
            arg_value(&argv, "--out")
                .unwrap_or_else(|| repo.join(DEFAULT_OUT).display().to_string()),
        );
        let state_out = PathBuf::from(
            arg_value(&argv, "--state-out")
                .unwrap_or_else(|| repo.join(DEFAULT_STATE_OUT).display().to_string()),
        );
        let started = Instant::now();

        // Dense prefix + dense full-attention organ are mandatory here.  A
        // compact first-token route bank would make later rows non-teacher
        // whenever the router changes its selected expert set.
        let prefix_states =
            match linear::capture_stateful_linear_prefix_outputs_dense(root.clone(), &tokens) {
                Ok(states) => states,
                Err(error) => {
                    let boundary = write_capture_boundary(
                        &out,
                        &root,
                        tokens.len(),
                        &started,
                        &error.to_string(),
                    )?;
                    eprintln!(
                        "Flash meta teacher capture blocked: {} (boundary receipt: {})",
                        error,
                        boundary.display()
                    );
                    return Err(error);
                }
            };
        let attention_receipt = out.with_file_name("FLASH_META_TEACHER_L4_LAYER3.json");
        let layer3_states = attention::run_stateful_attention_probe_from_states_with_outputs(
            root.clone(),
            3,
            &tokens,
            &prefix_states,
            attention_receipt.clone(),
        )?;
        if layer3_states.len() != tokens.len() {
            return Err("layer-3 teacher state count does not match token count".into());
        }

        let root = root.canonicalize()?;
        let index = SourceBf16Index::open(&root)?;
        let index_path = root.join("model.safetensors.index.json");
        let index_bytes = fs::read(&index_path)?;
        let config_path = root.join("config.json");
        let config_bytes = fs::read(&config_path)?;
        let source_identity = json!({
            "model": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "artifact_root": root,
            "safetensors_index": {
                "path": index_path,
                "sha256": sha256_bytes(&index_bytes),
                "bf16_tensor_count": index.tensor_count(),
            },
            "config": {
                "path": config_path,
                "sha256": sha256_bytes(&config_bytes),
            },
        });
        let context = MetalContext::new_with_trace(true)?;
        let mut layer4 =
            linear::StatefulLinearLayer::new_dense(&index, &context, 4, &layer3_states[0])?;
        let mut teacher_rows: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
        let mut rows = Vec::with_capacity(tokens.len());
        let mut route_sets = BTreeSet::new();
        let mut route_union = BTreeSet::new();
        for (step, (&token_id, layer3_state)) in tokens.iter().zip(&layer3_states).enumerate() {
            let started_step = Instant::now();
            let (_output, gpu_ns, _wall_ns, dispatches, final_state) =
                layer4.step(&context, Some(layer3_state), None, step == 0)?;
            let mlp_input = layer4.mlp_input();
            if mlp_input.len() != 2560 || mlp_input.iter().any(|value| !value.is_finite()) {
                return Err(
                    format!("layer-4 teacher row {step} is not finite HIDDEN geometry").into(),
                );
            }
            let route_ids = layer4.route_ids();
            let first_source_parity = if step == 0 {
                let source_route_ids = layer4.source_route_ids(layer3_state);
                if source_route_ids != route_ids {
                    return Err(
                        "first layer-4 teacher route IDs disagree with source oracle".into(),
                    );
                }
                Some(layer4.source_mlp_input_parity(layer3_state))
            } else {
                None
            };
            route_sets.insert(
                route_ids
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            );
            let row_sha256 = sha256_bytes(&f32_bytes(&mlp_input));
            let output_sha256 = sha256_bytes(&f32_bytes(&final_state));
            teacher_rows.push(mlp_input);
            route_union.extend(route_ids.iter().copied());
            rows.push(json!({
                "row": step,
                "token_id": token_id,
                "layer3_state_sha256": sha256_bytes(&f32_bytes(layer3_state)),
                "layer4_mlp_input_sha256": row_sha256,
                "layer4_output_sha256": output_sha256,
                "layer4_output_surface": "layer_4.final_state",
                "route_ids": route_ids,
                "first_source_mlp_input_parity": first_source_parity,
                "dispatches": dispatches,
                "gpu_ns": gpu_ns,
                "wall_ns": started_step.elapsed().as_nanos() as u64,
                "final_state_finite": final_state.iter().all(|value| value.is_finite()),
            }));
        }

        let state_bytes = flatten_rows(&teacher_rows);
        let mut unique_rows = BTreeSet::new();
        for row in &teacher_rows {
            unique_rows.insert(sha256_bytes(&f32_bytes(row)));
        }
        if unique_rows.len() < MIN_CORPUS_ROWS {
            return Err(format!(
                "teacher capture produced only {} unique mlp_input rows; refusing synthetic coverage",
                unique_rows.len()
            ).into());
        }
        if route_sets.len() < 2 || route_union.len() < 2 {
            return Err(
                "teacher corpus has no route diversity; refusing a degenerate capture".into(),
            );
        }
        if let Some(parent) = state_out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&state_out, &state_bytes)?;
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
        let mut receipt = json!({
            "schema": "hawking.flash.meta_teacher_trace.v1",
            "status": "CAPTURED_SOURCE_MLP_INPUT_NOT_CAPABILITY_PROVEN",
            "model": "Qwen/Qwen3.8-Flash-Next",
            "pinned_revision": "34567a4712bc9766c4449e2e98e4468bfa24d915",
            "source_identity": source_identity,
            "teacher_trace": {
                "layer": 4,
                "surface": "model.language_model.layers.4.mlp_input",
                "organ": "layer_4.routed_experts.gate_up_proj",
                "rows": teacher_rows.len(),
                "raw_rows": teacher_rows.len(),
                "unique_rows": unique_rows.len(),
                "width": 2560,
                "dtype": "F32_LE",
                "state_path": state_out,
                "state_bytes": state_bytes.len(),
                "state_sha256": sha256_bytes(&state_bytes),
                "row_semantics": "one exact source-BF16 layer-4 post-HyperConnection MLP input per token position",
                "source_pipeline": "dense source-BF16 layers 0..2 -> stateful source-BF16 layer 3 -> dense source-BF16 layer 4",
                "attention_receipt": attention_receipt,
                "expert_bank": "dense; every token chooses from the complete 512-expert bank",
                "candidate_controls": "all listed Flash accelerator controls were required off",
            },
            "route_audit": {
                "rows": rows.len(),
                "unique_ordered_topk_sets": route_sets.len(),
                "route_union": route_union,
                "topk_membership_and_order": "captured per row; this teacher artifact does not alter routing",
            },
            "execution": {
                "device": context.device_name(),
                "provider": "apple_metal",
                "source_bf16_authority": true,
                "dense_prefix": true,
                "dense_layer4": true,
                "source_index_sha256": sha256_bytes(&index_bytes),
                "accepted_generation_tokens": 0,
                "accepted_tps": null,
                "physical_ebpw": null,
            },
            "rows": rows,
            "bench": {
                "state": "UNKNOWN",
                "recorded_at": format!("unix-ms:{timestamp}"),
                "recorded_by": "flash_meta_teacher_trace",
                "rule": "S032 §3 -- teacher surface capture is not a protected model benchmark",
                "elapsed_ns": started.elapsed().as_nanos() as u64,
            },
            "promotion_allowed": false,
            "claim_boundary": "This is aligned source teacher activation evidence for one layer-4 expert input surface. It is not a serialized meta representation, coherence proof, physical EBPW, complete-token capability result, TPS result, or accelerator promotion.",
            "next_gate": "feed the state artifact to flash_meta_coherence_screen.py with at least 256 unique rows, then add router/hidden/routed-output/terminal-logit surfaces before any meta runtime is built",
            "token_ids": tokens,
        });
        receipt["seal_sha256"] = json!(sha256_bytes(&serde_json::to_vec(&receipt)?));
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "status": receipt["status"],
                "rows": teacher_rows.len(),
                "unique_rows": unique_rows.len(),
                "state_out": state_out,
                "receipt": out,
            }))?
        );
        Ok(())
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        main_impl()
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
