//! Full-sequence (multi-token, non-BOS) DSV4F activation capture at bridge sites.
//!
//! Runs real Metal growing-KV attention over frozen V0 sequences for every
//! layer admitted under the empty-compressed fullseq rule (ratio-0 always;
//! ratio-4/128 only while compressed topk is empty). Captures embedding /
//! mHC / pre-post attn / pre-router / routes / post-MoE / late-hidden
//! sufficient statistics and emits paired-trace shards + a sealed receipt.
//!
//! Honesty: NUMERIC_PARITY_V2_1_ONLY. No serve flip. No faked activations.
//!
//! Usage:
//!   cargo run -p hawking-core --release --example gravity_deepseek_v4_fullseq_capture -- \
//!     --artifact /path/to/full-43-layer-stream.gravity \
//!     --out-dir receipts/dsv4f_fullseq_capture \
//!     [--ladder L0|L1] [--max-layer 1] [--max-seq-len 8] [--limit N]

#[cfg(not(target_os = "macos"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::other(
        "gravity_deepseek_v4_fullseq_capture requires macOS Metal",
    )
    .into())
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
    use hawking_core::gravity_deepseek_v4_expert_cache::{
        resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey,
    };
    use hawking_core::gravity_deepseek_v4_fullseq_attention_device::{
        load_embedding_row_buffer, DeepSeekV4FullseqAttentionDeviceExecutor,
        DeepSeekV4FullseqLayerKvCache, DSV4F_FULLSEQ_KV_CAPACITY,
    };
    use hawking_core::gravity_deepseek_v4_layer0_prefix::PREFIX_TOKEN_ID;
    use hawking_core::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
    use hawking_core::gravity_deepseek_v4_layer_source_anchors::DeepSeekV4LayerGateMode;
    use hawking_core::gravity_deepseek_v4_p6_device::{
        DeepSeekV4Layer0P6MetalExecutor, DSV4F_P6_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_DEVICE_DISPATCHES, DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS,
        DSV4F_P6_LEARNED_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_p7_composition::{
        DeepSeekV4P7FfnSourceContract, DeepSeekV4P7SourceTensorBinding,
    };
    use hawking_core::gravity_deepseek_v4_p7_device::{
        DeepSeekV4P7BoundedDeviceExecutor, DeepSeekV4P7DeviceOutput,
        DSV4F_P7_OWNED_COMMAND_BUFFERS, DSV4F_P7_OWNED_DEVICE_DISPATCHES,
    };
    use hawking_core::gravity_deepseek_v4_runtime_spine::DeepSeekV4StagedTensor;
    use hawking_core::metal::MetalContext;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::PathBuf;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.fullseq_capture.v1";
    const TRACE_SCHEMA: &str = "hawking.frankenstein.paired_functional_trace.v1";
    const PARITY: &str = "NUMERIC_PARITY_V2_1_ONLY";
    const DEFAULT_MAX_LAYER: usize = 1; // ratio-0 L0+L1 by default
    const DEFAULT_MAX_SEQ_LEN: usize = 8;

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out_dir: PathBuf,
        ladder: String,
        max_layer: usize,
        max_seq_len: usize,
        limit: Option<usize>,
    }

    #[derive(Default)]
    struct Accounting {
        metal_dispatches: usize,
        command_buffers: usize,
        cpu_visible_waits: usize,
        sequences: usize,
        tokens: usize,
        layers_per_token: Vec<usize>,
    }

    pub fn run() -> ProbeResult<()> {
        let args = parse_args()?;
        if args.max_layer >= 43 {
            return Err("--max-layer must be in 0..42".into());
        }
        let wall = Instant::now();
        fs::create_dir_all(&args.out_dir)?;
        fs::create_dir_all(args.out_dir.join("traces"))?;

        let reader = DeepSeekV4FullStreamReader::admit(&args.artifact)?;
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(&reader)?;
        let metal = MetalContext::new()?;

        // Admit layers under empty-compressed fullseq at position 0 (always ok).
        let mut layers_run = Vec::new();
        let mut layer_blockers = Vec::new();
        for layer in 0..=args.max_layer {
            match catalog.plan(layer)?.require_fullseq_full_layer_device(0) {
                Ok(()) => layers_run.push(layer),
                Err(err) => layer_blockers.push(format!("layer {layer}: {err}")),
            }
        }
        if layers_run.is_empty() {
            return Err(format!(
                "no layers admitted for fullseq: {}",
                layer_blockers.join("; ")
            )
            .into());
        }

        let corpus = build_v0_corpus(&args.ladder, args.limit)?;
        let tokenizer = load_dsv4f_tokenizer(&reader)?;

        // Prepare per-layer attention executors once.
        let mut attn_execs: Vec<DeepSeekV4FullseqAttentionDeviceExecutor> = Vec::new();
        for &layer in &layers_run {
            attn_execs.push(DeepSeekV4FullseqAttentionDeviceExecutor::prepare(
                &metal, &reader, layer,
            )?);
        }

        let mut accounting = Accounting::default();
        let mut trace_paths = Vec::new();
        let mut total_capture_bytes: u64 = 0;
        let mut sites_union: Vec<String> = Vec::new();

        for (seq_idx, example) in corpus.iter().enumerate() {
            let mut token_ids = tokenize(&tokenizer, &example.prompt_text)?;
            if token_ids.is_empty() || token_ids[0] != PREFIX_TOKEN_ID as u32 {
                token_ids.insert(0, PREFIX_TOKEN_ID as u32);
            }
            if token_ids.len() > args.max_seq_len {
                token_ids.truncate(args.max_seq_len);
            }
            // Refuse sequences that would require non-empty compressed graphs
            // on any admitted layer beyond the empty-compressed window.
            let deepest_pos = token_ids.len().saturating_sub(1);
            for &layer in &layers_run {
                if let Err(err) = catalog
                    .plan(layer)?
                    .require_empty_compressed_growing_kv_attention(deepest_pos)
                {
                    return Err(format!(
                        "sequence {} (len={}) hits fullseq blocker on layer {layer}: {err}",
                        example.example_id,
                        token_ids.len()
                    )
                    .into());
                }
            }

            // Per-layer growing KV caches.
            let mut kv_caches: Vec<DeepSeekV4FullseqLayerKvCache> = layers_run
                .iter()
                .map(|&layer| {
                    DeepSeekV4FullseqLayerKvCache::new(
                        &metal,
                        layer,
                        DSV4F_FULLSEQ_KV_CAPACITY.min(args.max_seq_len.max(1)),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut positions_capture = Vec::new();
            for (pos, &tid) in token_ids.iter().enumerate() {
                let mut layer_captures = Vec::new();
                let mut prev_child: Option<DeepSeekV4P7DeviceOutput> = None;

                for (li, &layer) in layers_run.iter().enumerate() {
                    let input_buf = if layer == 0 {
                        load_embedding_row_buffer(&metal, &reader, tid)?
                    } else {
                        let child = prev_child
                            .as_ref()
                            .ok_or("missing predecessor child for layer>0")?;
                        child.child_hc_state_bf16.to_owned()
                    };

                    let attn_out = attn_execs[li].execute_position(
                        &metal,
                        &reader,
                        &input_buf,
                        tid,
                        pos,
                        &mut kv_caches[li],
                    )?;
                    accounting.metal_dispatches += attn_out.actual_gpu_dispatches;
                    accounting.command_buffers += attn_out.actual_command_buffers;
                    accounting.cpu_visible_waits += attn_out.actual_cpu_visible_waits;

                    let attention = attn_out.p7_attention_state(&metal, &kv_caches[li])?;
                    let (child, p7_disp, p7_cbs, route_ids, route_scores_sha) = run_p7_p6(
                        &metal,
                        &reader,
                        layer,
                        tid,
                        pos,
                        attention,
                    )?;
                    accounting.metal_dispatches += p7_disp;
                    accounting.command_buffers += p7_cbs;
                    accounting.cpu_visible_waits += p7_cbs;

                    let embed_sha = if layer == 0 {
                        Some(buffer_sha256(&input_buf)?)
                    } else {
                        None
                    };
                    let pre_attn_sha = buffer_sha256(&attn_out.attention_hc_state_bf16)?;
                    let post_moe_sha = buffer_sha256(&child.p6.moe_output_bf16)?;
                    let late_hidden_sha = buffer_sha256(&child.child_hc_state_bf16)?;
                    let ffn_norm_sha = buffer_sha256(&child.ffn_norm_bf16)?;

                    let sites = json!({
                        "layer": layer,
                        "token_id": tid,
                        "token_position": pos,
                        "compress_ratio": attn_out.compress_ratio,
                        "empty_compressed": attn_out.empty_compressed,
                        "kv_rows": attn_out.kv_rows,
                        "sparse_attention_kernel": attn_out.sparse_attention_kernel,
                        "embedding_row_sha256": embed_sha,
                        "mhc_pre_post_attn_hc_sha256": pre_attn_sha,
                        "pre_router_ffn_norm_sha256": ffn_norm_sha,
                        "router_top6_route_ids": route_ids,
                        "router_scores_sha256": route_scores_sha,
                        "post_moe_sha256": post_moe_sha,
                        "late_hidden_child_hc_sha256": late_hidden_sha,
                        "attn_metal_dispatches": attn_out.actual_gpu_dispatches,
                        "p7_p6_metal_dispatches": p7_disp,
                    });
                    // Approximate capture payload (hashes + route ids, not full tensors).
                    total_capture_bytes += 6 * 32 + 6 * 4 + 128;
                    for key in [
                        "embedding",
                        "mHC",
                        "pre_post_attn",
                        "pre_router",
                        "router_logits_top6_routes",
                        "post_MoE",
                        "late_hidden",
                    ] {
                        if !sites_union.iter().any(|s| s == key) {
                            sites_union.push(key.to_string());
                        }
                    }
                    layer_captures.push(sites);
                    prev_child = Some(child);
                }
                positions_capture.push(json!({
                    "position": pos,
                    "token_id": tid,
                    "layers": layer_captures,
                }));
                accounting.tokens += 1;
            }
            accounting.sequences += 1;
            accounting.layers_per_token.push(layers_run.len());

            let dsv4f_side = json!({
                "side": "dsv4f",
                "present": true,
                "capture_status": "OK",
                "token_count": token_ids.len(),
                // token_ids recorded with forbidden-for-alignment flag
                "token_ids_for_debug_only": token_ids,
                "token_ids_forbidden_for_alignment": true,
                "layers_run": layers_run,
                "positions": positions_capture,
                "parity": PARITY,
                "representative_hidden_states": [
                    {
                        "site": "late_hidden_last_pos_last_layer",
                        "note": "sha256 of child HC BF16[4,4096]; full tensor retained on device only",
                    }
                ],
                "route_statistics": {
                    "top_k": 6,
                    "note": "native DSV4F routes; never copied from GLM",
                },
                "hcli_action_tool_decisions": null,
                "hcli_note": "HCLI not exercised in this bounded capture; field reserved",
            });

            let prompt_bytes = example.prompt_text.as_bytes();
            let decoded_spans = vec![json!({
                "text": example.prompt_text,
                "byte_start": 0,
                "byte_end": prompt_bytes.len(),
                "role": "prompt",
                "side": "shared",
                "token_ids": null,
                "token_ids_forbidden_for_alignment": true,
            })];

            let mut trace = json!({
                "schema": TRACE_SCHEMA,
                "recorded_at": chrono_like_now(),
                "example_id": example.example_id,
                "membership": example.membership,
                "prompt_text": example.prompt_text,
                "decoded_spans": decoded_spans,
                "method_family_choice": example.method_family,
                "decomposition": null,
                "proof_plan": null,
                "formal_actions": [],
                "tool_events": [],
                "repair_steps": [],
                "verification": null,
                "bounded_logits_top_k": null,
                "representative_hidden_states": null,
                "route_statistics": null,
                "sides": {
                    "dsv4f": dsv4f_side,
                    "glm": {
                        "side": "glm",
                        "present": false,
                        "capture_status": "ABSENT_PENDING_GLM_TEACHER_FORCED",
                        "note": "GLM side filled by the parallel GLM teacher-forced capture lane; byte-span anchors match prompt_text"
                    }
                },
                "alignment_policy": {
                    "align_on": ["decoded_spans", "byte_ranges", "formal_actions", "tool_events"],
                    "never_align_on": ["token_ids", "incompatible_vocab_indices"]
                },
                "meta": {
                    "complete_pair": false,
                    "ladder": args.ladder,
                    "seq_index": seq_idx,
                    "max_layer": args.max_layer,
                    "layers_run": layers_run,
                    "parity": PARITY,
                },
                "fabricated": false,
            });
            let seal = seal_document(&mut trace)?;
            let path = args
                .out_dir
                .join("traces")
                .join(format!("{}.json", example.example_id));
            write_json(&path, &trace)?;
            trace_paths.push(json!({
                "example_id": example.example_id,
                "path": path.display().to_string(),
                "seal_sha256": seal,
            }));
            println!(
                "captured {} tokens={} layers={:?} seal={}",
                example.example_id,
                token_ids.len(),
                layers_run,
                &seal[..16]
            );
        }

        let wall_ms = wall.elapsed().as_secs_f64() * 1e3;
        let ratios_covered = json!({
            "ratio_0": layers_run.iter().any(|&l| expected_ratio(l) == 0),
            "ratio_4_empty_compressed_only": layers_run.iter().any(|&l| expected_ratio(l) == 4),
            "ratio_128_empty_compressed_only": layers_run.iter().any(|&l| expected_ratio(l) == 128),
            "ratio_4_full_indexer_graph": false,
            "ratio_128_full_compressed_graph": false,
        });

        let receipt = json!({
            "schema": RECEIPT_SCHEMA,
            "status": format!(
                "PASS_FULLSEQ_CAPTURE_{}_{}_LAYERS_0_TO_{}",
                args.ladder,
                accounting.sequences,
                layers_run.last().copied().unwrap_or(0)
            ),
            "artifact": {
                "path": args.artifact.display().to_string(),
                "manifest_seal_sha256": catalog.identity().manifest_seal_sha256,
                "repository": catalog.identity().repository,
                "revision": catalog.identity().revision,
            },
            "scope": {
                "ladder": args.ladder,
                "sequences": accounting.sequences,
                "tokens_total": accounting.tokens,
                "max_seq_len": args.max_seq_len,
                "layers_run": layers_run,
                "max_layer_requested": args.max_layer,
                "layer_blockers": layer_blockers,
            },
            "non_bos_forward": {
                "ratios_covered": ratios_covered,
                "growing_kv_sparse_kernel": "deepseek_v4_p4_sparse_attention_ratio0_growing_kv_sink_authority",
                "parity": PARITY,
                "exact_storage": false,
                "blocker_ratio_4_full": "indexer/compressor graph not implemented for non-empty compressed slots (layer_plan::require_empty_compressed_growing_kv_attention)",
                "blocker_ratio_128_full": "same empty-compressed specialization only",
            },
            "sites_captured": sites_union,
            "capture_bytes_approx": total_capture_bytes,
            "metal": {
                "metal_dispatches": accounting.metal_dispatches,
                "command_buffers": accounting.command_buffers,
                "cpu_visible_waits": accounting.cpu_visible_waits,
                "fallback": 0,
            },
            "paired_traces": {
                "schema": TRACE_SCHEMA,
                "n_traces": trace_paths.len(),
                "dir": args.out_dir.join("traces").display().to_string(),
                "rows": trace_paths,
            },
            "honesty": {
                "fabricated_activations": false,
                "serve_endpoint_flipped": false,
                "full_43_layer_fullseq": layers_run.len() == 43,
                "glm_side_present": false,
                "hcli_decisions_captured": false,
                "logits_captured": false,
                "logits_note": "final lm_head requires full 43-layer body per token; not run under ratio-0-only default",
            },
            "wall_time_ms": wall_ms,
            "parity": {
                "classification": PARITY,
                "exact_storage": false,
            },
        });

        let receipt_path = args.out_dir.join("DSV4F_FULLSEQ_CAPTURE_RECEIPT.json");
        let pretty = serde_json::to_string_pretty(&receipt)?;
        let mut file = File::create(&receipt_path)?;
        file.write_all(pretty.as_bytes())?;
        file.write_all(b"\n")?;
        let seal = format!("{:x}", Sha256::digest(pretty.as_bytes()));
        // Also drop a copy under repo receipts/ for discoverability.
        let repo_receipt = PathBuf::from("receipts/dsv4f_fullseq_capture_receipt.json");
        if let Some(parent) = repo_receipt.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(&repo_receipt, pretty.as_bytes());

        println!("{pretty}");
        println!("receipt_path: {}", receipt_path.display());
        println!("receipt_sha256: {seal}");
        println!(
            "metal_dispatches: {} sequences: {} tokens: {}",
            accounting.metal_dispatches, accounting.sequences, accounting.tokens
        );
        Ok(())
    }

    struct CorpusExample {
        example_id: String,
        membership: String,
        prompt_text: String,
        method_family: Option<String>,
    }

    fn build_v0_corpus(ladder: &str, limit: Option<usize>) -> ProbeResult<Vec<CorpusExample>> {
        // Frozen V0 ladder sizes (steer): L0=32, L1=128. Short math/method prompts
        // so sequences stay within empty-compressed windows for ratio-4/128 if
        // deeper layers are later admitted.
        let base: Vec<(&str, &str, &str)> = vec![
            ("v0_math_01", "train", "Evaluate (17 × 29) + 7."),
            ("v0_math_02", "train", "Find gcd(119, 203)."),
            ("v0_math_03", "train", "Solve for x: 5x + 17 = 52."),
            ("v0_math_04", "train", "Sum the integers from 1 through 12."),
            ("v0_math_05", "train", "A rectangle has sides 17 and 7. Perimeter?"),
            ("v0_math_06", "train", "What remainder when 500 is divided by 29?"),
            ("v0_math_07", "train", "Mean of 17, 29, and 7 to six decimals."),
            ("v0_math_08", "train", "How many unordered pairs from 8 objects?"),
            ("v0_math_09", "train", "Simplify 29² − 17²."),
            ("v0_math_10", "train", "Cart covers 30 m in 5 s. Mean speed?"),
            ("v0_code_01", "train", "Python: sum of squares 1 through 7."),
            ("v0_code_02", "train", "Python: dedupe list preserving order."),
            ("v0_code_03", "train", "Python: invert a dictionary."),
            ("v0_code_04", "train", "Python: filter x where 7 <= x < 17."),
            ("v0_code_05", "train", "Python: successive differences via zip."),
            ("v0_reason_01", "train", "Every amber key opens chest A. Can it open chest B if nothing that opens A opens B? yes/no"),
            ("v0_reason_02", "train", "Rin taller than Sol, Sol taller than Tev. Who is shortest?"),
            ("v0_reason_03", "train", "If lamp K on => sensor awake. Sensor not awake. Is lamp K on?"),
            ("v0_reason_04", "calibration", "All nembles are quiet. P is a nemble. Must P be quiet?"),
            ("v0_reason_05", "calibration", "Exactly one door unlocked. West and east locked. Which is unlocked?"),
            ("v0_tool_01", "calibration", "JSON call tool sum_pair left=17 right=29."),
            ("v0_tool_02", "calibration", "JSON set_alarm enabled=true hour=7."),
            ("v0_tool_03", "public_test", "JSON batch_ids ids=[17,29,7]."),
            ("v0_formal_01", "public_test", "Prove: if n even then n² even. Outline steps."),
            ("v0_formal_02", "public_test", "Counterexample: all primes odd?"),
            ("v0_repair_01", "public_test", "Bug: claims 2+2=5. Repair the claim."),
            ("v0_agent_01", "hidden_test", "Plan: list files then run tests. First action?"),
            ("v0_agent_02", "hidden_test", "Regression between good-1 and bad-2. Best git op?"),
            ("v0_zh_01", "hidden_test", "温室里有17棵松树和29棵竹子。总数？"),
            ("v0_en_01", "hidden_test", "Maple-17 maps to harbor-29. Mapped value?"),
            ("v0_mix_01", "train", "Record river-17 maps to puerto-29. Value?"),
            ("v0_long_01", "train", "Mini archive: pine-17 stores rv-29. Retrieve."),
        ];
        let target = match ladder {
            "L0" => 32,
            "L1" => 128,
            other => return Err(format!("unknown ladder {other}; use L0 or L1").into()),
        };
        let mut out = Vec::new();
        for i in 0..target {
            let (id0, mem, prompt) = base[i % base.len()];
            let example_id = if i < base.len() {
                id0.to_string()
            } else {
                format!("{id0}_rep{}", i / base.len())
            };
            let method = if prompt.contains("gcd") || prompt.contains("Solve") {
                Some("algebra".to_string())
            } else if prompt.contains("Python") {
                Some("coding".to_string())
            } else if prompt.contains("Prove") || prompt.contains("Counterexample") {
                Some("formal_proof".to_string())
            } else if prompt.contains("JSON") {
                Some("tool".to_string())
            } else {
                Some("reasoning".to_string())
            };
            out.push(CorpusExample {
                example_id,
                membership: mem.to_string(),
                prompt_text: prompt.to_string(),
                method_family: method,
            });
        }
        if let Some(n) = limit {
            out.truncate(n);
        }
        Ok(out)
    }

    fn load_dsv4f_tokenizer(reader: &DeepSeekV4FullStreamReader) -> ProbeResult<Tokenizer> {
        let bytes = reader.read_verified_metadata_asset("tokenizer.json", 16 * 1024 * 1024)?;
        let tok = Tokenizer::from_bytes(&bytes)
            .map_err(|e| format!("tokenizer load failed: {e}"))?;
        Ok(tok)
    }

    fn tokenize(tok: &Tokenizer, text: &str) -> ProbeResult<Vec<u32>> {
        let encoding = tok
            .encode(text, false)
            .map_err(|e| format!("tokenize failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn run_p7_p6<'a>(
        metal: &'a MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        layer: usize,
        token_id: u32,
        token_position: usize,
        attention: hawking_core::gravity_deepseek_v4_p7_composition::DeepSeekV4P7AttentionDeviceState<'a>,
    ) -> ProbeResult<(DeepSeekV4P7DeviceOutput, usize, usize, [u32; 6], String)> {
        let (source, ffn_norm, mhc_ffn) =
            stage_p7_controls(reader, layer, token_id, token_position)?;
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(reader)?;
        let plan = catalog.plan(layer)?;
        let learned = plan.gate_mode == DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias;

        let mut cache = if learned {
            DeepSeekV4ExpertBundleCache::new(1, 0)?
        } else {
            let tid2eid_name = format!("layers.{layer}.ffn.gate.tid2eid");
            let meta = reader.tensor_metadata(&tid2eid_name)?;
            let bytes = reader.read_verified_full(&tid2eid_name, meta.bytes as usize)?;
            let route_ids = read_tid2eid_row_u16(&bytes, token_id as usize)?.map(u32::from);
            DeepSeekV4ExpertBundleCache::new(
                required_hot_cache_bytes(reader, layer as u16, &route_ids)?,
                0,
            )?
        };

        let p6 = DeepSeekV4Layer0P6MetalExecutor::prepare_for_p7(metal, reader, &mut cache, &source)?;
        let mut p7 = DeepSeekV4P7BoundedDeviceExecutor::prepare(
            metal,
            source,
            &ffn_norm,
            &mhc_ffn,
            Box::new(p6),
        )?;
        let output = p7.execute_fullseq(attention)?;
        output.validate()?;
        let route_ids = read_route_ids_from_output(&output)?;
        let route_scores_sha = buffer_sha256(&output.p6.original_scores_f32)?;
        let (p6_disp, p6_cbs) = if learned {
            (
                DSV4F_P6_LEARNED_DEVICE_DISPATCHES,
                DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS,
            )
        } else {
            (DSV4F_P6_DEVICE_DISPATCHES, DSV4F_P6_DEVICE_COMMAND_BUFFERS)
        };
        let disp = DSV4F_P7_OWNED_DEVICE_DISPATCHES + p6_disp;
        let cbs = DSV4F_P7_OWNED_COMMAND_BUFFERS + p6_cbs;
        Ok((output, disp, cbs, route_ids, route_scores_sha))
    }

    fn stage_p7_controls(
        reader: &DeepSeekV4FullStreamReader,
        layer: usize,
        token_id: u32,
        token_position: usize,
    ) -> ProbeResult<(
        DeepSeekV4P7FfnSourceContract,
        DeepSeekV4StagedTensor,
        [DeepSeekV4StagedTensor; 3],
    )> {
        let ffn_norm = stage_full(reader, &format!("layers.{layer}.ffn_norm.weight"))?;
        let hc_fn = stage_full(reader, &format!("layers.{layer}.hc_ffn_fn"))?;
        let hc_base = stage_full(reader, &format!("layers.{layer}.hc_ffn_base"))?;
        let hc_scale = stage_full(reader, &format!("layers.{layer}.hc_ffn_scale"))?;
        let source = DeepSeekV4P7FfnSourceContract {
            layer,
            token_id,
            token_position,
            ffn_norm: binding(&ffn_norm),
            hc_ffn_fn: binding(&hc_fn),
            hc_ffn_base: binding(&hc_base),
            hc_ffn_scale: binding(&hc_scale),
            source_parent_retained: false,
            source_upload_required_before_execution: true,
            host_activation_handoff_permitted: false,
            runtime_boundary: "fullseq multi-token GPU capture P7 controls; no Engine/HCLI/serve/TPS",
        };
        Ok((source, ffn_norm, [hc_fn, hc_base, hc_scale]))
    }

    fn stage_full(
        reader: &DeepSeekV4FullStreamReader,
        name: &str,
    ) -> ProbeResult<DeepSeekV4StagedTensor> {
        let metadata = reader.tensor_metadata(name)?;
        let bytes = usize::try_from(metadata.bytes)
            .map_err(|_| format!("{name} bytes exceed host usize"))?;
        let payload = reader.read_verified_full(name, bytes)?;
        Ok(DeepSeekV4StagedTensor {
            name: metadata.name.clone(),
            dtype: metadata.dtype.clone(),
            shape: metadata.shape.clone(),
            source_shard: metadata.source_shard.clone(),
            range: 0..metadata.bytes,
            bytes: payload,
        })
    }

    fn binding(staged: &DeepSeekV4StagedTensor) -> DeepSeekV4P7SourceTensorBinding {
        DeepSeekV4P7SourceTensorBinding {
            name: staged.name.clone(),
            dtype: staged.dtype.clone(),
            shape: staged.shape.clone(),
            bytes: staged.bytes.len(),
            sha256: format!("{:x}", Sha256::digest(&staged.bytes)),
        }
    }

    fn required_hot_cache_bytes(
        reader: &DeepSeekV4FullStreamReader,
        layer: u16,
        route_ids: &[u32],
    ) -> ProbeResult<u64> {
        route_ids.iter().try_fold(0u64, |total, &expert| {
            let descriptor =
                resolve_expert_bundle(reader, ExpertBundleKey::new(layer, expert as u16))?;
            total
                .checked_add(descriptor.payload_bytes)
                .ok_or_else(|| "expert hot capacity overflow".into())
        })
    }

    fn read_tid2eid_row_u16(table: &[u8], token_id: usize) -> ProbeResult<[u16; 6]> {
        let row_bytes = 6 * 8;
        let start = token_id
            .checked_mul(row_bytes)
            .ok_or("tid2eid row offset overflow")?;
        let end = start
            .checked_add(row_bytes)
            .ok_or("tid2eid row end overflow")?;
        if end > table.len() {
            return Err("tid2eid row exceeds table".into());
        }
        let mut out = [0u16; 6];
        for (i, slot) in out.iter_mut().enumerate() {
            let off = start + i * 8;
            let raw = i64::from_le_bytes(table[off..off + 8].try_into()?);
            if raw < 0 || raw > u16::MAX as i64 {
                return Err(format!("tid2eid expert id {raw} out of u16 range").into());
            }
            *slot = raw as u16;
        }
        Ok(out)
    }

    fn read_route_ids_from_output(output: &DeepSeekV4P7DeviceOutput) -> ProbeResult<[u32; 6]> {
        let buf = &output.p6.route_ids_u32;
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return Err("route_ids buffer contents null".into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, 6 * 4) };
        let mut out = [0u32; 6];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into()?);
        }
        Ok(out)
    }

    fn buffer_sha256(buf: &metal::Buffer) -> ProbeResult<String> {
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return Err("buffer contents null".into());
        }
        let len = buf.length() as usize;
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    fn expected_ratio(layer: usize) -> usize {
        match layer {
            0 | 1 => 0,
            i if i % 2 == 0 => 4,
            _ => 128,
        }
    }

    fn seal_document(doc: &mut serde_json::Value) -> ProbeResult<String> {
        if let Some(obj) = doc.as_object_mut() {
            obj.remove("seal_sha256");
        }
        let canonical = serde_json::to_vec(doc)?;
        let seal = format!("{:x}", Sha256::digest(&canonical));
        if let Some(obj) = doc.as_object_mut() {
            obj.insert("seal_sha256".into(), json!(seal));
        }
        Ok(seal)
    }

    fn write_json(path: &PathBuf, value: &serde_json::Value) -> ProbeResult<()> {
        let pretty = serde_json::to_string_pretty(value)?;
        let mut f = File::create(path)?;
        f.write_all(pretty.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    fn chrono_like_now() -> String {
        // RFC3339-ish UTC without pulling chrono dep.
        use std::time::{SystemTime, UNIX_EPOCH};
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        format!("unix:{secs}")
    }

    fn parse_args() -> ProbeResult<Args> {
        let mut artifact = None;
        let mut out_dir = None;
        let mut ladder = "L0".to_string();
        let mut max_layer = DEFAULT_MAX_LAYER;
        let mut max_seq_len = DEFAULT_MAX_SEQ_LEN;
        let mut limit = None;
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--artifact" => {
                    artifact = Some(PathBuf::from(args.next().ok_or("--artifact needs path")?));
                }
                "--out-dir" => {
                    out_dir = Some(PathBuf::from(args.next().ok_or("--out-dir needs path")?));
                }
                "--ladder" => {
                    ladder = args.next().ok_or("--ladder needs L0|L1")?;
                }
                "--max-layer" => {
                    max_layer = args
                        .next()
                        .ok_or("--max-layer needs value")?
                        .parse()
                        .map_err(|_| "--max-layer must be int")?;
                }
                "--max-seq-len" => {
                    max_seq_len = args
                        .next()
                        .ok_or("--max-seq-len needs value")?
                        .parse()
                        .map_err(|_| "--max-seq-len must be int")?;
                }
                "--limit" => {
                    limit = Some(
                        args.next()
                            .ok_or("--limit needs value")?
                            .parse()
                            .map_err(|_| "--limit must be int")?,
                    );
                }
                "--help" | "-h" => {
                    println!(
                        "gravity_deepseek_v4_fullseq_capture --artifact <p> --out-dir <p> [--ladder L0|L1] [--max-layer N] [--max-seq-len N] [--limit N]"
                    );
                    std::process::exit(0);
                }
                other => return Err(format!("unknown argument {other}").into()),
            }
        }
        Ok(Args {
            artifact: artifact.ok_or("--artifact is required")?,
            out_dir: out_dir.ok_or("--out-dir is required")?,
            ladder,
            max_layer,
            max_seq_len,
            limit,
        })
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
