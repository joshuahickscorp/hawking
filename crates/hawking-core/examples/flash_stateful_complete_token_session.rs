//! Complete Flash stateful-session attempt.
//!
//! This is the smallest end-to-end session-shaped executor built from the
//! already-qualified linear and full-attention organs.  It deliberately keeps
//! cross-species seams explicit (host vectors) while preserving recurrence in
//! every linear segment and KV state across every full-attention token list.
//! A receipt is emitted for either accepted-candidate success or the first
//! physical boundary; no promotion or performance claim is implied.

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other("Flash stateful session requires macOS Metal").into())
}

#[cfg(target_os = "macos")]
#[path = "flash_full_attention_layer3.rs"]
mod full;
#[cfg(target_os = "macos")]
#[path = "flash_noetic_complete_layer0.rs"]
mod linear;
#[cfg(target_os = "macos")]
#[path = "flash_source_bf16_terminal.rs"]
mod terminal;

#[cfg(target_os = "macos")]
mod macos {
    use super::{full, linear, terminal};
    use hawking_core::metal::MetalContext;
    use hawking_core::model::qwen80_source_bf16_layer_major::SourceBf16Index;
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::env;
    use std::error::Error;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    const DEFAULT_ROOT: &str =
        "/Volumes/corpdrive/hawking-modellake/specimens/Qwen--Qwen3.8-Flash-Next@34567a4712bc";
    const REPO_ID: &str = "Qwen/Qwen3.8-Flash-Next";
    const PINNED_REVISION: &str = "34567a4712bc9766c4449e2e98e4468bfa24d915";
    const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
    const HIDDEN: usize = 2560;
    const STREAMS: usize = 4;
    const HC: usize = HIDDEN * STREAMS;
    const FULL_LAYERS: [usize; 12] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47];
    const PROMPT_IDS: [usize; 5] = [5423, 799, 4581, 3817, 13];
    const DEFAULT_CANDIDATE: usize = 17;

    fn sha256(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    fn arg_value(args: &[String], flag: &str) -> Option<String> {
        args.windows(2)
            .find(|pair| pair[0] == flag)
            .map(|pair| pair[1].clone())
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
    }

    fn parse_token_ids(argv: &[String]) -> Result<Vec<usize>, Box<dyn Error>> {
        if let Some(raw) = arg_value(argv, "--token-ids") {
            let ids = raw
                .split(',')
                .map(|part| part.trim().parse::<usize>())
                .collect::<Result<Vec<_>, _>>()?;
            if ids.is_empty() {
                return Err("--token-ids must not be empty".into());
            }
            return Ok(ids);
        }
        let mut ids = PROMPT_IDS.to_vec();
        ids.push(
            arg_value(argv, "--candidate")
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(DEFAULT_CANDIDATE),
        );
        Ok(ids)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bf16(bytes: &[u8], i: usize) -> f32 {
        f32::from_bits((u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]) as u32) << 16)
    }

    fn embedding_row(
        index: &SourceBf16Index,
        token_id: usize,
        vocab: usize,
    ) -> Result<Vec<f32>, Box<dyn Error>> {
        if token_id >= vocab {
            return Err(format!("token {token_id} outside vocab {vocab}").into());
        }
        let bytes = index.read_raw_range(EMBEDDING, token_id * HIDDEN * 2, HIDDEN * 2)?;
        Ok((0..HIDDEN).map(|i| bf16(&bytes, i)).collect())
    }

    fn repeated_streams(row: &[f32]) -> Vec<f32> {
        (0..STREAMS).flat_map(|_| row.iter().copied()).collect()
    }

    fn vocab(root: &Path) -> Result<usize, Box<dyn Error>> {
        let config: Value = serde_json::from_slice(&fs::read(root.join("config.json"))?)?;
        config
            .get("text_config")
            .and_then(|v| v.get("vocab_size"))
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| "text_config.vocab_size missing".into())
    }

    fn write_state(path: &Path, state: &[f32]) -> Result<(), Box<dyn Error>> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, f32_bytes(state))?;
        Ok(())
    }

    fn run_linear_segment(
        index: &SourceBf16Index,
        context: &MetalContext,
        layer_start: usize,
        layer_end: usize,
        token_ids: &[usize],
        vocab: usize,
    ) -> Result<(Vec<Vec<f32>>, Value), Box<dyn Error>> {
        let first_row = embedding_row(&index, token_ids[0], vocab)?;
        let first_base = repeated_streams(&first_row);
        let mut layers = Vec::new();
        let load_started = Instant::now();
        let bytes_before_layers = index.bytes_read_total();
        for layer in layer_start..=layer_end {
            layers.push(linear::StatefulLinearLayer::new_dense(
                index,
                context,
                layer,
                &first_base,
            )?);
        }
        let layer_prepare_ns = load_started.elapsed().as_nanos() as u64;
        let layer_source_bytes = index.bytes_read_total().saturating_sub(bytes_before_layers);
        let started = Instant::now();
        let mut outputs = Vec::with_capacity(token_ids.len());
        let mut rows = Vec::with_capacity(token_ids.len() * layers.len());
        for (step, &token_id) in token_ids.iter().enumerate() {
            let row = embedding_row(&index, token_id, vocab)?;
            let host_base = repeated_streams(&row);
            let mut prior_device = None;
            let mut final_state = None;
            for (offset, layer) in layers.iter_mut().enumerate() {
                let (device_output, gpu_ns, wall_ns, dispatches, observed) = if offset == 0 {
                    layer.step(context, Some(&host_base), None, step == 0)?
                } else {
                    let input = prior_device
                        .as_ref()
                        .ok_or("missing device inter-layer state")?;
                    layer.step(context, None, Some(input), step == 0)?
                };
                let layer_index = layer_start + offset;
                if observed.iter().any(|v| !v.is_finite()) {
                    return Err(format!(
                        "non-finite linear state at layer {layer_index} step {step}"
                    )
                    .into());
                }
                rows.push(json!({
                    "step": step,
                    "token_id": token_id,
                    "layer": layer_index,
                    "dispatches": dispatches,
                    "gpu_ns": gpu_ns,
                    "wall_ns": wall_ns,
                    "device_inter_layer_handoff": offset > 0,
                    "finite": true,
                    "final_state_sha256": sha256(&f32_bytes(&observed)),
                }));
                prior_device = Some(device_output);
                final_state = Some(observed);
            }
            outputs.push(final_state.ok_or("linear segment produced no output")?);
        }
        let execution_wall_ns = started.elapsed().as_nanos() as u64;
        let execution_gpu_ns = rows
            .iter()
            .map(|r| r.get("gpu_ns").and_then(Value::as_u64).unwrap_or(0))
            .sum::<u64>();
        let execution_dispatches = rows
            .iter()
            .map(|r| r.get("dispatches").and_then(Value::as_u64).unwrap_or(0))
            .sum::<u64>();
        let host_and_unattributed_ns = rows
            .iter()
            .map(|r| {
                r.get("wall_ns")
                    .and_then(Value::as_u64)
                    .unwrap_or(0)
                    .saturating_sub(r.get("gpu_ns").and_then(Value::as_u64).unwrap_or(0))
            })
            .sum::<u64>();
        Ok((
            outputs,
            json!({
                "layers": [layer_start, layer_end],
                "species": "linear_attention",
                "stateful_recurrence": true,
                "device_inter_layer_handoffs": rows.iter().filter(|r| r.get("device_inter_layer_handoff") == Some(&Value::Bool(true))).count(),
                "steps": rows,
                "wall_ns": execution_wall_ns,
                "source_payload_bytes_read": index.bytes_read_total(),
                "timing": {
                    "layer_source_and_device_prepare_ns": layer_prepare_ns,
                    "layer_source_bytes_read": layer_source_bytes,
                    "execution_wall_ns": execution_wall_ns,
                    "execution_gpu_ns": execution_gpu_ns,
                    "execution_dispatches": execution_dispatches,
                    "host_and_unattributed_ns": host_and_unattributed_ns
                },
            }),
        ))
    }

    fn run_full_layer(
        root: &Path,
        layer: usize,
        token_ids: &[usize],
        input_states: &[Vec<f32>],
        out_dir: &Path,
    ) -> Result<(Vec<Vec<f32>>, Value), Box<dyn Error>> {
        let receipt = out_dir.join(format!("layer-{layer}-attention.json"));
        let states = full::run_stateful_attention_probe_from_states_with_outputs(
            root.to_path_buf(),
            layer,
            token_ids,
            input_states,
            receipt.clone(),
        )?;
        if states.len() != token_ids.len()
            || states
                .iter()
                .any(|s| s.len() != HC || s.iter().any(|v| !v.is_finite()))
        {
            return Err(format!(
                "full-attention layer {layer} returned invalid state count or non-finite values"
            )
            .into());
        }
        Ok((
            states,
            json!({
                "layer": layer,
                "species": "full_attention",
                "stateful_kv": true,
                "receipt": receipt,
            }),
        ))
    }

    fn terminal_token(
        executor: &terminal::TerminalExecutor,
        state: &[f32],
        path: &Path,
    ) -> Result<(usize, Value), Box<dyn Error>> {
        write_state(path, state)?;
        let receipt = path.with_extension("terminal.json");
        let (token, _doc) = executor
            .run_state(&path.to_path_buf(), Some(receipt.clone()))
            .map_err(|e| -> Box<dyn Error> { e.into() })?;
        Ok((
            token,
            json!({"token_id": token, "receipt": receipt, "state": path}),
        ))
    }

    fn main_impl() -> Result<(), Box<dyn Error>> {
        let argv: Vec<String> = env::args().collect();
        let root = PathBuf::from(arg_value(&argv, "--root").unwrap_or_else(|| {
            env::var("HCLI_FLASH_NEXT_ROOT").unwrap_or_else(|_| DEFAULT_ROOT.to_owned())
        }))
        .canonicalize()?;
        let out = PathBuf::from(arg_value(&argv, "--out").unwrap_or_else(|| {
            repo_root()
                .join("receipts/headless/FLASH_STATEFUL_COMPLETE_TOKEN_SESSION.json")
                .display()
                .to_string()
        }));
        let out_dir = out
            .parent()
            .unwrap_or_else(|| Path::new("receipts/headless"))
            .join("flash_stateful_complete_token_session");
        fs::create_dir_all(&out_dir)?;
        let token_ids = parse_token_ids(&argv)?;
        if token_ids.len() < 2 {
            return Err("complete session requires prompt plus candidate token".into());
        }
        let vocab = vocab(&root)?;
        // Keep the immutable source index and Metal context alive for the
        // entire native session. Linear segments still release their layer
        // weights at species seams, but no longer reopen the source or create
        // a fresh Metal device/queue for every segment.
        let linear_index = SourceBf16Index::open(&root)?;
        let linear_context = MetalContext::new_with_trace(true)?;
        let started = Instant::now();
        let mut current_states: Option<Vec<Vec<f32>>> = None;
        let mut segments = Vec::new();
        let mut cursor = 0usize;
        for &full_layer in FULL_LAYERS.iter() {
            if cursor < full_layer {
                let end = full_layer - 1;
                let (states, receipt) = run_linear_segment(
                    &linear_index,
                    &linear_context,
                    cursor,
                    end,
                    &token_ids,
                    vocab,
                )?;
                current_states = Some(states);
                segments.push(receipt);
            }
            let input = current_states.take().ok_or(format!(
                "missing input state before full-attention layer {full_layer}"
            ))?;
            let (states, receipt) =
                run_full_layer(&root, full_layer, &token_ids, &input, &out_dir)?;
            current_states = Some(states);
            segments.push(receipt);
            cursor = full_layer + 1;
        }
        if cursor < 48 {
            let (states, receipt) = run_linear_segment(
                &linear_index,
                &linear_context,
                cursor,
                47,
                &token_ids,
                vocab,
            )?;
            current_states = Some(states);
            segments.push(receipt);
        }
        let final_states = current_states.ok_or("complete session produced no final states")?;
        if final_states.len() != token_ids.len() {
            return Err("complete session final state/token count mismatch".into());
        }
        let prompt_len = token_ids.len() - 1;
        let terminal_dir = out_dir.join("terminal");
        fs::create_dir_all(&terminal_dir)?;
        let terminal_executor = terminal::TerminalExecutor::new(root.clone())?;
        let (predicted_candidate, prompt_terminal) = terminal_token(
            &terminal_executor,
            &final_states[prompt_len - 1],
            &terminal_dir.join("prompt-final.state.f32"),
        )?;
        let (next_after_candidate, candidate_terminal) = terminal_token(
            &terminal_executor,
            &final_states[prompt_len],
            &terminal_dir.join("candidate-final.state.f32"),
        )?;
        let candidate = token_ids[prompt_len];
        let accepted = predicted_candidate == candidate;
        let status = if accepted {
            "PASSED_STATEFUL_COMPLETE_TOKEN_SESSION"
        } else {
            "PASSED_COMPLETE_FORWARD_CANDIDATE_REJECTED"
        };
        let mut receipt = json!({
            "schema": "hawking.flash.stateful_complete_token_session.v1",
            "status": status,
            "model": REPO_ID,
            "pinned_revision": PINNED_REVISION,
            "root": root,
            "token_ids": token_ids,
            "prompt_token_ids": &token_ids[..prompt_len],
            "candidate_token_id": candidate,
            "vocab_size": vocab,
            "segments": segments,
            "terminal": {
                "prompt_final": prompt_terminal,
                "candidate_final": candidate_terminal,
                "predicted_candidate": predicted_candidate,
                "next_after_candidate": next_after_candidate,
                "candidate_accepted": accepted,
            },
            "execution": {
                "process_boundary": "one native process",
                "terminal_executor": {
                    "lifetime": "one session",
                    "source_index_reused": true,
                    "readout_weights_reused": true,
                    "lm_head_reused": true,
                    "source_payload_bytes_read": terminal_executor.source_bytes_read(),
                    "device": terminal_executor.device_name(),
                },
                "linear_recurrence_segments": true,
                "full_attention_kv_segments": true,
                "cross_species_activation_handoff": "host diagnostic vectors",
                "source_independent": false,
                "fallback_count": 0,
                "elapsed_wall_ns": started.elapsed().as_nanos() as u64,
            },
            "accepted_generation_tokens": if accepted { 1 } else { 0 },
            "accepted_tps": Value::Null,
            "complete_system_ebpw": Value::Null,
            "promotion_allowed": false,
            "first_physical_failure_boundary": if accepted { Value::Null } else { json!({"stage": "candidate_acceptance", "candidate_token_id": candidate, "predicted_token_id": predicted_candidate, "reason": "complete forward succeeded but candidate did not match terminal argmax"}) },
            "claim_boundary": if accepted {
                "A complete 48-layer stateful Flash forward accepted one tokenizer-bound candidate token in one native process. Cross-species seams still use host diagnostic vectors; this does not prove source-independent NX, TPS, EBPW, capability, or HCLI residency."
            } else {
                "A complete 48-layer stateful Flash forward executed in one native process, but the supplied candidate was rejected by terminal argmax. This is a physical complete-forward result and an exact acceptance boundary, not a TPS, EBPW, capability, or residency claim."
            },
            "next": if accepted { "Run repeated accepted decode steps, then qualify capability/TPS/EBPW and replace host species seams." } else { "Rerun with the terminal-predicted candidate, then measure repeated accepted decode steps." },
            "bench": {"state": "UNKNOWN", "recorded_at": format!("unix-ms:{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis()), "recorded_by": "flash_stateful_complete_token_session", "machine": "Apple Metal", "rule": "S032 §3 -- complete-session timing is not a promotion benchmark"},
        });
        receipt["seal_sha256"] = Value::String(sha256(&serde_json::to_vec(&receipt)?));
        if let Some(parent) = out.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&out, serde_json::to_vec_pretty(&receipt)?)?;
        println!("{}", serde_json::to_string_pretty(&receipt)?);
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
