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
//!     [--ladder L0|L1] [--max-layer 1] [--max-seq-len 8] [--limit N] \
//!     [--corpus-mode frozen|synthetic] [--corpus PATH.jsonl] \
//!     [--export-host-activations]
//!
//! Corpus: default `--corpus-mode frozen` loads PROTO_FRANKENSTEIN_V0
//! (pfv0:* example_ids, same rows as GLM teacher-forced). The legacy
//! hard-coded v0_math_*/v0_code_* table is `--corpus-mode synthetic` only
//! (smoke / empty-compressed window tests).
//!
//! Host export: `--export-host-activations` is an **offline-analysis diagnostic**.
//! It copies bounded activation tensors from Metal buffers to
//! `out_dir/activations/` AFTER device execution. It does **not** set
//! `host_activation_handoff_permitted=true` on the P6/P7 source contract
//! (that flag remains false; the device graph still rejects host-handoff
//! contracts). Default runtime path is unchanged (hashes only).

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
    use half::bf16;
    use hawking_core::metal::MetalContext;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;
    use std::error::Error;
    use std::fs::{self, File};
    use std::io::{BufRead, BufReader, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;
    use tokenizers::Tokenizer;

    const RECEIPT_SCHEMA: &str = "hawking.gravity.deepseek_v4.fullseq_capture.v1";
    const TRACE_SCHEMA: &str = "hawking.frankenstein.paired_functional_trace.v1";
    const PARITY: &str = "NUMERIC_PARITY_V2_1_ONLY";
    const DEFAULT_MAX_LAYER: usize = 1; // ratio-0 L0+L1 by default
    const DEFAULT_MAX_SEQ_LEN: usize = 8;
    /// Canonical frozen V0 corpora (same rows GLM teacher-forced freezes).
    const DEFAULT_FROZEN_L0: &str =
        "workspace/campaign/evidence/models/frankenstein/corpus/PROTO_FRANKENSTEIN_V0_L0_CORPUS.jsonl";
    const DEFAULT_FROZEN_L1: &str =
        "workspace/campaign/evidence/models/frankenstein/corpus/PROTO_FRANKENSTEIN_V0_L1_CORPUS.jsonl";
    /// Official GLM L0 freeze used as the correspondence identity oracle.
    const CANONICAL_FROZEN_L0_JSON: &str =
        "workspace/campaign/evidence/models/frankenstein/teacher_forced/official_L0_stream_full_20260805T200728Z/FROZEN_CORPUS_L0.json";

    type ProbeResult<T> = Result<T, Box<dyn Error>>;

    struct Args {
        artifact: PathBuf,
        out_dir: PathBuf,
        ladder: String,
        max_layer: usize,
        max_seq_len: usize,
        limit: Option<usize>,
        /// "frozen" (default, PROTO_FRANKENSTEIN_V0) or "synthetic" (legacy short prompts).
        corpus_mode: String,
        corpus_path: Option<PathBuf>,
        /// Offline-analysis diagnostic: copy bounded activations to host files.
        /// Does NOT flip host_activation_handoff_permitted on the device contract.
        export_host_activations: bool,
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
        if args.export_host_activations {
            fs::create_dir_all(args.out_dir.join("activations"))?;
        }

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

        let (corpus, corpus_provenance) = build_v0_corpus(
            &args.ladder,
            args.limit,
            &args.corpus_mode,
            args.corpus_path.as_deref(),
        )?;
        let tokenizer = load_dsv4f_tokenizer(&reader)?;
        // Per-layer host export accumulators: example order → late_hidden f32[H].
        // Only populated when --export-host-activations is set.
        let mut export_late_hidden: BTreeMap<usize, Vec<Vec<f32>>> = BTreeMap::new();
        let mut export_example_ids: Vec<String> = Vec::new();
        if args.export_host_activations {
            for &layer in &layers_run {
                export_late_hidden.insert(layer, Vec::new());
            }
        }

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

                    // Optional offline host export: copy late_hidden AFTER device
                    // work. Does not change host_activation_handoff_permitted
                    // (still false on the P7 source contract above).
                    let mut late_hidden_export: Option<Vec<f32>> = None;
                    if args.export_host_activations {
                        let f32s = buffer_to_f32_mean_pool(&child.child_hc_state_bf16)?;
                        late_hidden_export = Some(f32s);
                    }

                    let mut sites = json!({
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
                    if let Some(ref vec) = late_hidden_export {
                        // Embed last-position vector only when this is the final
                        // token of the sequence (filled below after the pos loop
                        // for the sidecar; inline for last pos to keep traces useful).
                        sites.as_object_mut().unwrap().insert(
                            "late_hidden_export_mode".into(),
                            json!("offline_analysis_diagnostic"),
                        );
                        sites.as_object_mut().unwrap().insert(
                            "late_hidden_dtype".into(),
                            json!("f32"),
                        );
                        sites.as_object_mut().unwrap().insert(
                            "late_hidden_shape".into(),
                            json!([vec.len()]),
                        );
                        // Stash on sites for last-pos promotion after the token loop.
                        sites.as_object_mut().unwrap().insert(
                            "_late_hidden_f32_tmp".into(),
                            json!(vec),
                        );
                    }
                    // Approximate capture payload (hashes + route ids, not full tensors).
                    total_capture_bytes += 6 * 32 + 6 * 4 + 128;
                    if late_hidden_export.is_some() {
                        total_capture_bytes += late_hidden_export.as_ref().map(|v| v.len() * 4).unwrap_or(0) as u64;
                    }
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

            // Promote last-position late_hidden f32 into the trace for the loader,
            // strip temporary vectors from earlier positions (keep hashes only).
            if args.export_host_activations {
                if let Some(last_pos) = positions_capture.last_mut() {
                    if let Some(layers) = last_pos.get_mut("layers").and_then(|v| v.as_array_mut())
                    {
                        for layer_row in layers.iter_mut() {
                            let obj = layer_row
                                .as_object_mut()
                                .ok_or("layer row not object during export promote")?;
                            if let Some(tmp) = obj.remove("_late_hidden_f32_tmp") {
                                let layer_idx = obj
                                    .get("layer")
                                    .and_then(|v| v.as_u64())
                                    .ok_or("missing layer in export row")?
                                    as usize;
                                let f32s: Vec<f32> = tmp
                                    .as_array()
                                    .ok_or("late_hidden tmp not array")?
                                    .iter()
                                    .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                                    .collect();
                                obj.insert("late_hidden".into(), tmp);
                                if let Some(bucket) = export_late_hidden.get_mut(&layer_idx) {
                                    bucket.push(f32s);
                                }
                            }
                        }
                    }
                }
                // Drop temp vectors from non-last positions to keep traces bounded.
                let npos = positions_capture.len();
                for (i, pos_row) in positions_capture.iter_mut().enumerate() {
                    if i + 1 == npos {
                        continue;
                    }
                    if let Some(layers) = pos_row.get_mut("layers").and_then(|v| v.as_array_mut()) {
                        for layer_row in layers.iter_mut() {
                            if let Some(obj) = layer_row.as_object_mut() {
                                obj.remove("_late_hidden_f32_tmp");
                            }
                        }
                    }
                }
                export_example_ids.push(example.example_id.clone());
            }

            let hidden_note = if args.export_host_activations {
                "offline_analysis_diagnostic: late_hidden f32 exported on last position + activations/LXX.npy sidecars; device contract still host_activation_handoff_permitted=false"
            } else {
                "sha256 of child HC BF16; full tensor retained on device only (set --export-host-activations for offline correspondence export)"
            };

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
                "host_activation_export": args.export_host_activations,
                "host_activation_handoff_permitted": false,
                "representative_hidden_states": [
                    {
                        "site": "late_hidden_last_pos_last_layer",
                        "note": hidden_note,
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

        // Write activation sidecars for the correspondence loader
        // (activations/L{nn:02d}.npy with shape [N, D] float32 C-order).
        let mut export_sidecar_paths: Vec<serde_json::Value> = Vec::new();
        if args.export_host_activations {
            let act_dir = args.out_dir.join("activations");
            let ids_path = act_dir.join("example_ids.json");
            write_json(
                &ids_path,
                &json!({
                    "schema": "hawking.gravity.deepseek_v4.fullseq_activation_export_ids.v1",
                    "example_ids": export_example_ids,
                    "n": export_example_ids.len(),
                    "mode": "offline_analysis_diagnostic",
                    "host_activation_handoff_permitted": false,
                }),
            )?;
            for (&layer, rows) in &export_late_hidden {
                if rows.is_empty() {
                    continue;
                }
                let n = rows.len();
                let d = rows[0].len();
                if rows.iter().any(|r| r.len() != d) {
                    return Err(format!(
                        "export layer {layer}: inconsistent hidden widths"
                    )
                    .into());
                }
                let mut flat: Vec<f32> = Vec::with_capacity(n * d);
                for r in rows {
                    flat.extend_from_slice(r);
                }
                let npy_path = act_dir.join(format!("L{layer:02}.npy"));
                write_npy_f32_2d(&npy_path, &flat, n, d)?;
                export_sidecar_paths.push(json!({
                    "layer": layer,
                    "path": npy_path.display().to_string(),
                    "shape": [n, d],
                    "dtype": "float32",
                    "site": "late_hidden",
                }));
                // Also write a one-key npz-equivalent via a tiny companion json
                // the Python loader understands alongside .npy.
                let meta_path = act_dir.join(format!("L{layer:02}.export.json"));
                write_json(
                    &meta_path,
                    &json!({
                        "schema": "hawking.gravity.deepseek_v4.fullseq_activation_export.v1",
                        "layer": layer,
                        "site": "late_hidden",
                        "npy": npy_path.display().to_string(),
                        "shape": [n, d],
                        "dtype": "float32",
                        "example_ids_path": ids_path.display().to_string(),
                        "mode": "offline_analysis_diagnostic",
                        "host_activation_handoff_permitted": false,
                        "note": "Post-execute host copy for correspondence only; device graph still no-host.",
                    }),
                )?;
            }
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
                "corpus_mode": args.corpus_mode,
                "corpus_path": args.corpus_path.as_ref().map(|p| p.display().to_string()),
                "export_host_activations": args.export_host_activations,
                "empty_compressed_fullseq_note": "ratio-4 layers only admit positions 0..2 (end_pos//4==0); ratio-128 admit 0..126; ratio-0 admit 0..127. All-43-layer capture requires max_seq_len<=3.",
            },
            // Provenance gap fix: prior receipts omitted corpus identity, so the
            // synthetic v0_math_* table went undetected against GLM's pfv0:* freeze.
            "corpus_provenance": corpus_provenance,
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
                "host_activation_handoff_permitted": false,
                "host_activation_export_is_diagnostic_only": true,
            },
            "host_activation_export": {
                "enabled": args.export_host_activations,
                "mode": if args.export_host_activations {
                    "offline_analysis_diagnostic"
                } else {
                    "disabled_default_hashes_only"
                },
                "host_activation_handoff_permitted": false,
                "note": "Export copies bounded Metal buffers to host AFTER execute for correspondence (CKA/CCA/Procrustes). The P6/P7 source contract keeps host_activation_handoff_permitted=false; the device graph is unchanged.",
                "sidecars": export_sidecar_paths,
                "dir": if args.export_host_activations {
                    Some(args.out_dir.join("activations").display().to_string())
                } else {
                    None
                },
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

    fn build_v0_corpus(
        ladder: &str,
        limit: Option<usize>,
        corpus_mode: &str,
        corpus_path: Option<&Path>,
    ) -> ProbeResult<(Vec<CorpusExample>, serde_json::Value)> {
        match corpus_mode {
            "frozen" | "proto" | "pfv0" => {
                let path = match corpus_path {
                    Some(p) => p.to_path_buf(),
                    None => PathBuf::from(match ladder {
                        "L0" => DEFAULT_FROZEN_L0,
                        "L1" => DEFAULT_FROZEN_L1,
                        other => {
                            return Err(format!("unknown ladder {other}; use L0 or L1").into())
                        }
                    }),
                };
                let mut out = load_frozen_proto_corpus(&path)?;
                if let Some(n) = limit {
                    out.truncate(n);
                }
                if out.is_empty() {
                    return Err(format!("frozen corpus empty: {}", path.display()).into());
                }
                let provenance = seal_corpus_provenance("frozen", &path, ladder, &out)?;
                println!(
                    "corpus_mode=frozen path={} n={} id0={} file_sha256={} identical_to_canonical={}",
                    path.display(),
                    out.len(),
                    out[0].example_id,
                    provenance
                        .get("file_sha256")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?"),
                    provenance
                        .get("identical_to_canonical")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                );
                Ok((out, provenance))
            }
            "synthetic" | "legacy_short" => {
                // Legacy short prompts (v0_math_*/v0_code_*). NOT the frozen V0
                // corpus — zero content-hash overlap with GLM/PROTO. Kept only
                // for empty-compressed window smokes.
                let out = build_synthetic_short_corpus(ladder, limit)?;
                println!(
                    "corpus_mode=synthetic n={} WARNING: not PROTO_FRANKENSTEIN_V0; correspondence ID intersection with GLM will be empty",
                    out.len()
                );
                let provenance = json!({
                    "schema": "hawking.gravity.deepseek_v4.fullseq_corpus_provenance.v1",
                    "mode": "synthetic",
                    "path": null,
                    "file_sha256": null,
                    "n_sequences": out.len(),
                    "example_ids": out.iter().map(|e| e.example_id.clone()).collect::<Vec<_>>(),
                    "identical_to_canonical": false,
                    "warning": "legacy v0_math_*/v0_code_* table; zero content-hash overlap with PROTO_FRANKENSTEIN_V0 / GLM freeze",
                });
                Ok((out, provenance))
            }
            other => Err(format!(
                "unknown --corpus-mode {other}; use frozen (default) or synthetic"
            )
            .into()),
        }
    }

    /// Hash the corpus source file and prove identity vs the official GLM freeze
    /// (example_id set + per-row prompt_text_sha256). Prior receipts lacked this
    /// field, which is how the synthetic-prompt bug went undetected.
    fn seal_corpus_provenance(
        mode: &str,
        path: &Path,
        ladder: &str,
        examples: &[CorpusExample],
    ) -> ProbeResult<serde_json::Value> {
        let bytes = fs::read(path)
            .map_err(|e| format!("read corpus for provenance {}: {e}", path.display()))?;
        let file_sha256 = format!("{:x}", Sha256::digest(&bytes));
        let example_ids: Vec<String> = examples.iter().map(|e| e.example_id.clone()).collect();
        let prompt_text_sha256: BTreeMap<String, String> = examples
            .iter()
            .map(|e| {
                (
                    e.example_id.clone(),
                    format!("{:x}", Sha256::digest(e.prompt_text.as_bytes())),
                )
            })
            .collect();

        let mut identical_to_canonical = false;
        let mut canonical_path: Option<String> = None;
        let mut canonical_seal_sha256: Option<String> = None;
        let mut canonical_membership_sha256: Option<String> = None;
        let mut identity_detail = json!({"checked": false});

        if ladder == "L0" {
            let frozen_path = PathBuf::from(CANONICAL_FROZEN_L0_JSON);
            if frozen_path.is_file() {
                let frozen_raw = fs::read_to_string(&frozen_path).map_err(|e| {
                    format!("read canonical freeze {}: {e}", frozen_path.display())
                })?;
                let frozen: serde_json::Value = serde_json::from_str(&frozen_raw).map_err(|e| {
                    format!("parse canonical freeze {}: {e}", frozen_path.display())
                })?;
                canonical_path = Some(frozen_path.display().to_string());
                canonical_seal_sha256 = frozen
                    .get("seal_sha256")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());
                canonical_membership_sha256 = frozen
                    .get("membership_sha256")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                let mut frozen_ids: Vec<String> = Vec::new();
                let mut frozen_hashes: BTreeMap<String, String> = BTreeMap::new();
                if let Some(seqs) = frozen.get("sequences").and_then(|v| v.as_array()) {
                    for s in seqs {
                        if let Some(id) = s.get("example_id").and_then(|v| v.as_str()) {
                            frozen_ids.push(id.to_string());
                            if let Some(h) = s.get("prompt_text_sha256").and_then(|v| v.as_str()) {
                                frozen_hashes.insert(id.to_string(), h.to_string());
                            }
                        }
                    }
                }
                let id_set_equal = {
                    let a: std::collections::BTreeSet<_> = example_ids.iter().collect();
                    let b: std::collections::BTreeSet<_> = frozen_ids.iter().collect();
                    a == b
                };
                let order_equal = example_ids == frozen_ids;
                let mut hash_mismatches: Vec<String> = Vec::new();
                for (id, h) in &prompt_text_sha256 {
                    match frozen_hashes.get(id) {
                        Some(fh) if fh == h => {}
                        Some(_) => hash_mismatches.push(id.clone()),
                        None => hash_mismatches.push(format!("{id}:missing_in_freeze")),
                    }
                }
                identical_to_canonical = id_set_equal && hash_mismatches.is_empty();
                identity_detail = json!({
                    "checked": true,
                    "id_set_equal": id_set_equal,
                    "order_equal": order_equal,
                    "prompt_text_sha256_mismatches": hash_mismatches.len(),
                    "mismatch_example_ids_head": hash_mismatches.iter().take(8).cloned().collect::<Vec<_>>(),
                    "n_local": example_ids.len(),
                    "n_canonical": frozen_ids.len(),
                });
            } else {
                identity_detail = json!({
                    "checked": false,
                    "reason": format!("canonical freeze missing: {}", frozen_path.display()),
                });
            }
        }

        Ok(json!({
            "schema": "hawking.gravity.deepseek_v4.fullseq_corpus_provenance.v1",
            "mode": mode,
            "path": path.display().to_string(),
            "file_sha256": file_sha256,
            "file_bytes": bytes.len(),
            "n_sequences": examples.len(),
            "example_ids": example_ids,
            "prompt_text_sha256": prompt_text_sha256,
            "canonical_frozen_corpus_path": canonical_path,
            "canonical_seal_sha256": canonical_seal_sha256,
            "canonical_membership_sha256": canonical_membership_sha256,
            "identical_to_canonical": identical_to_canonical,
            "identity_basis": "example_id set equality + per-row prompt_text_sha256 match against official GLM FROZEN_CORPUS_L0",
            "identity_detail": identity_detail,
        }))
    }

    fn load_frozen_proto_corpus(path: &Path) -> ProbeResult<Vec<CorpusExample>> {
        let file = File::open(path)
            .map_err(|e| format!("open frozen corpus {}: {e}", path.display()))?;
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for (lineno, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| format!("{}:{}: {e}", path.display(), lineno + 1))?;
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| format!("{}:{}: JSON {e}", path.display(), lineno + 1))?;
            let example_id = v
                .get("example_id")
                .and_then(|x| x.as_str())
                .ok_or_else(|| format!("{}:{}: missing example_id", path.display(), lineno + 1))?
                .to_string();
            let prompt_text = v
                .get("surface_text")
                .or_else(|| v.get("prompt_text"))
                .and_then(|x| x.as_str())
                .ok_or_else(|| {
                    format!(
                        "{}:{}: missing surface_text/prompt_text",
                        path.display(),
                        lineno + 1
                    )
                })?
                .to_string();
            let membership = v
                .get("membership")
                .and_then(|x| x.as_str())
                .unwrap_or("train")
                .to_string();
            let method_family = v
                .get("family")
                .or_else(|| v.get("method_family"))
                .and_then(|x| x.as_str())
                .map(|s| s.to_string());
            out.push(CorpusExample {
                example_id,
                membership,
                prompt_text,
                method_family,
            });
        }
        Ok(out)
    }

    fn build_synthetic_short_corpus(
        ladder: &str,
        limit: Option<usize>,
    ) -> ProbeResult<Vec<CorpusExample>> {
        // Short math/method prompts so sequences stay within empty-compressed
        // windows for ratio-4/128 smoke tests. NOT for real correspondence.
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

    /// Offline-analysis diagnostic: copy a BF16 Metal buffer to host f32.
    ///
    /// Child HC state is BF16 with leading layout rows (typically HC mult × hidden).
    /// We mean-pool over complete 4096-wide rows so the correspondence vector is
    /// width=hidden (4096). This is a post-execute host read for diagnostics only
    /// — it does not set host_activation_handoff_permitted on the device contract.
    fn buffer_to_f32_mean_pool(buf: &metal::Buffer) -> ProbeResult<Vec<f32>> {
        const HIDDEN: usize = 4096;
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return Err("buffer contents null for host export".into());
        }
        let nbytes = buf.length() as usize;
        if nbytes % 2 != 0 {
            return Err(format!("BF16 buffer length {nbytes} not even").into());
        }
        let n_bf16 = nbytes / 2;
        let bits = unsafe { std::slice::from_raw_parts(ptr as *const u16, n_bf16) };
        if n_bf16 == 0 {
            return Err("empty BF16 buffer for host export".into());
        }
        // Prefer mean-pool over full hidden rows; fall back to flat convert.
        if n_bf16 % HIDDEN == 0 {
            let rows = n_bf16 / HIDDEN;
            let mut acc = vec![0.0f32; HIDDEN];
            for r in 0..rows {
                for c in 0..HIDDEN {
                    let v = bf16::from_bits(bits[r * HIDDEN + c]).to_f32();
                    acc[c] += v;
                }
            }
            if rows > 1 {
                let inv = 1.0 / rows as f32;
                for x in acc.iter_mut() {
                    *x *= inv;
                }
            }
            Ok(acc)
        } else {
            Ok(bits
                .iter()
                .map(|b| bf16::from_bits(*b).to_f32())
                .collect())
        }
    }

    /// Minimal NumPy `.npy` writer for C-order float32 2-D arrays (no extra deps).
    fn write_npy_f32_2d(path: &Path, data: &[f32], rows: usize, cols: usize) -> ProbeResult<()> {
        if data.len() != rows * cols {
            return Err(format!(
                "npy size mismatch: data={} expected {}",
                data.len(),
                rows * cols
            )
            .into());
        }
        // NPY v1.0 header.
        let descr = "{'descr': '<f4', 'fortran_order': False, 'shape': (ROWS, COLS), }"
            .replace("ROWS", &rows.to_string())
            .replace("COLS", &cols.to_string());
        // Header must be padded so magic(6)+ver(2)+hlen(2)+header is 64-byte aligned.
        let prefix = 10usize; // 6 magic + 2 ver + 2 header_len
        let mut header = descr;
        // numpy pads with spaces and terminates with '\n'
        let mut total = prefix + header.len() + 1; // +1 for newline
        let pad = (64 - (total % 64)) % 64;
        header.push_str(&" ".repeat(pad));
        header.push('\n');
        total = prefix + header.len();
        if total % 64 != 0 {
            return Err("npy header alignment bug".into());
        }
        let header_len = u16::try_from(header.len())
            .map_err(|_| "npy header exceeds u16")?;
        let mut f = File::create(path)?;
        f.write_all(b"\x93NUMPY")?;
        f.write_all(&[1u8, 0u8])?; // v1.0
        f.write_all(&header_len.to_le_bytes())?;
        f.write_all(header.as_bytes())?;
        for &v in data {
            f.write_all(&v.to_le_bytes())?;
        }
        Ok(())
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
        let mut corpus_mode = "frozen".to_string();
        let mut corpus_path = None;
        let mut export_host_activations = false;
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
                "--corpus-mode" => {
                    corpus_mode = args
                        .next()
                        .ok_or("--corpus-mode needs frozen|synthetic")?;
                }
                "--corpus" => {
                    corpus_path =
                        Some(PathBuf::from(args.next().ok_or("--corpus needs path")?));
                }
                "--export-host-activations" => {
                    export_host_activations = true;
                }
                "--help" | "-h" => {
                    println!(
                        "gravity_deepseek_v4_fullseq_capture --artifact <p> --out-dir <p> \
                         [--ladder L0|L1] [--max-layer N] [--max-seq-len N] [--limit N] \
                         [--corpus-mode frozen|synthetic] [--corpus PATH.jsonl] \
                         [--export-host-activations]"
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
            corpus_mode,
            corpus_path,
            export_host_activations,
        })
    }
}

#[cfg(target_os = "macos")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    macos::run()
}
