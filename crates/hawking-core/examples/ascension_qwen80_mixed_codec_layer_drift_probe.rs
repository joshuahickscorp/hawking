//! Teacher-forced mixed-codec layer-drift probe for Q80.
//!
//! Runs the source BF16 layer-major forward on a real prompt, then re-runs a
//! contiguous span with routed-expert weights replaced by reconstructed mixed
//! (or matched-magnitude shuffled-null) packs. Remaining layers stay true BF16
//! so span-end hidden can be read out as real next-token logits.
//!
//! Does not pack a full artifact. Does not emit a Metal kernel.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen80 mixed-codec layer-drift probe requires macOS");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen80_source_bf16_layer_major::{
        embed_probes, forward_layer_probe, logits_from_final_hidden, peak_rss_bytes, LoadedLayer,
        SourceBf16Index, QWEN80_EXPERTS, QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE,
        STREAMED_PEAK_RSS_HARD_CAP_BYTES,
    };
    use serde_json::{json, Value};
    use std::env;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::time::Instant;
    use tokenizers::Tokenizer;

    const GATE_UP_BYTES: usize = QWEN80_MOE_INTERMEDIATE * QWEN80_HIDDEN * 2;
    const DOWN_BYTES: usize = QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE * 2;
    const PACK_ROLE_BYTES: usize = QWEN80_EXPERTS * GATE_UP_BYTES;

    struct Arguments {
        source_model_dir: PathBuf,
        tokenizer_path: Option<PathBuf>,
        prompt: String,
        span_start: usize,
        span_end: usize,
        mixed_override_dir: PathBuf,
        null_override_dir: PathBuf,
        output_dir: PathBuf,
        continue_remaining: bool,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen80_mixed_codec_layer_drift_probe \
         --source-model-dir ABS \
         --mixed-override-dir ABS --null-override-dir ABS --output-dir ABS \
         [--tokenizer-path ABS] [--prompt TEXT] \
         [--span-start N] [--span-end N] [--no-continue-remaining]"
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        eprintln!("q80 mixed-codec drift probe refused: {}", detail.as_ref());
        process::exit(2);
    }

    fn absolute(path: PathBuf, flag: &str) -> PathBuf {
        if !path.is_absolute() {
            fail(format!("{flag} must be an absolute path"));
        }
        path
    }

    fn parse_usize(value: &str, flag: &str) -> usize {
        value
            .parse::<usize>()
            .unwrap_or_else(|_| fail(format!("{flag} must be an unsigned integer")))
    }

    fn parse_arguments() -> Arguments {
        let mut source_model_dir = None;
        let mut tokenizer_path = None;
        let mut prompt = "Write a function that reverses a string.".to_string();
        let mut span_start = 0usize;
        let mut span_end = 4usize;
        let mut mixed_override_dir = None;
        let mut null_override_dir = None;
        let mut output_dir = None;
        let mut continue_remaining = true;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--source-model-dir" => {
                    source_model_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fail("missing value for --source-model-dir")
                    })));
                }
                "--tokenizer-path" => {
                    tokenizer_path = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fail("missing value for --tokenizer-path")
                    })));
                }
                "--prompt" => {
                    prompt = args
                        .next()
                        .unwrap_or_else(|| fail("missing value for --prompt"));
                }
                "--span-start" => {
                    span_start = parse_usize(
                        &args
                            .next()
                            .unwrap_or_else(|| fail("missing value for --span-start")),
                        "--span-start",
                    );
                }
                "--span-end" => {
                    span_end = parse_usize(
                        &args
                            .next()
                            .unwrap_or_else(|| fail("missing value for --span-end")),
                        "--span-end",
                    );
                }
                "--mixed-override-dir" => {
                    mixed_override_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fail("missing value for --mixed-override-dir")
                    })));
                }
                "--null-override-dir" => {
                    null_override_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fail("missing value for --null-override-dir")
                    })));
                }
                "--output-dir" => {
                    output_dir = Some(PathBuf::from(
                        args.next()
                            .unwrap_or_else(|| fail("missing value for --output-dir")),
                    ));
                }
                "--no-continue-remaining" => continue_remaining = false,
                other => fail(format!("unsupported flag {other}; {}", usage())),
            }
        }
        let source_model_dir = absolute(
            source_model_dir.unwrap_or_else(|| fail("missing --source-model-dir")),
            "--source-model-dir",
        );
        let mixed_override_dir = absolute(
            mixed_override_dir.unwrap_or_else(|| fail("missing --mixed-override-dir")),
            "--mixed-override-dir",
        );
        let null_override_dir = absolute(
            null_override_dir.unwrap_or_else(|| fail("missing --null-override-dir")),
            "--null-override-dir",
        );
        let output_dir = absolute(
            output_dir.unwrap_or_else(|| fail("missing --output-dir")),
            "--output-dir",
        );
        if span_end <= span_start || span_end > QWEN80_LAYERS {
            fail(format!(
                "span [{span_start}, {span_end}) is empty or past {QWEN80_LAYERS} layers"
            ));
        }
        Arguments {
            source_model_dir,
            tokenizer_path,
            prompt,
            span_start,
            span_end,
            mixed_override_dir,
            null_override_dir,
            output_dir,
            continue_remaining,
        }
    }

    fn last_token<'a>(hidden: &'a [f32], seq_len: usize) -> &'a [f32] {
        &hidden[(seq_len - 1) * QWEN80_HIDDEN..seq_len * QWEN80_HIDDEN]
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let xf = f64::from(*x);
            let yf = f64::from(*y);
            dot += xf * yf;
            na += xf * xf;
            nb += yf * yf;
        }
        dot / (na.sqrt() * nb.sqrt()).max(1e-30)
    }

    fn rel_l2(pred: &[f32], truth: &[f32]) -> f64 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (p, t) in pred.iter().zip(truth.iter()) {
            let d = f64::from(*p) - f64::from(*t);
            num += d * d;
            den += f64::from(*t) * f64::from(*t);
        }
        num.sqrt() / den.sqrt().max(1e-30)
    }

    fn mean_token_cosine_rel_l2(pred: &[f32], truth: &[f32], seq_len: usize) -> (f64, f64) {
        let mut c = 0.0;
        let mut e = 0.0;
        for pos in 0..seq_len {
            let a = &pred[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN];
            let b = &truth[pos * QWEN80_HIDDEN..(pos + 1) * QWEN80_HIDDEN];
            c += cosine(a, b);
            e += rel_l2(a, b);
        }
        (c / seq_len as f64, e / seq_len as f64)
    }

    fn l2_norm(v: &[f32]) -> f64 {
        v.iter()
            .map(|x| {
                let f = f64::from(*x);
                f * f
            })
            .sum::<f64>()
            .sqrt()
    }

    fn topk_ids(logits: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_by(|a, b| {
            logits[*b]
                .partial_cmp(&logits[*a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        idx
    }

    fn log_softmax(logits: &[f32]) -> Vec<f64> {
        let mut max = f64::NEG_INFINITY;
        for &v in logits {
            if v.is_finite() {
                max = max.max(f64::from(v));
            }
        }
        let mut out = vec![0.0f64; logits.len()];
        let mut acc = 0.0f64;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_finite() {
                let e = (f64::from(v) - max).exp();
                out[i] = e;
                acc += e;
            } else {
                out[i] = 0.0;
            }
        }
        let ln_acc = acc.max(1e-300).ln();
        for (i, &v) in logits.iter().enumerate() {
            if v.is_finite() {
                out[i] = f64::from(v) - max - ln_acc;
            } else {
                out[i] = f64::NEG_INFINITY;
            }
        }
        out
    }

    fn kl_true_to_other(true_logits: &[f32], other_logits: &[f32]) -> f64 {
        let lt = log_softmax(true_logits);
        let lo = log_softmax(other_logits);
        let mut kl = 0.0f64;
        for (a, b) in lt.iter().zip(lo.iter()) {
            if a.is_finite() {
                let p = a.exp();
                if p > 0.0 {
                    let log_q = if b.is_finite() { *b } else { -1.0e30 };
                    kl += p * (a - log_q);
                }
            }
        }
        kl
    }

    fn apply_pack(layer: &mut LoadedLayer, dir: &Path) -> Result<(), String> {
        let prefix = dir.join(format!("L{:02}", layer.layer));
        let gate = fs::read(prefix.join("gate.bf16"))
            .map_err(|e| format!("read {}: {e}", prefix.join("gate.bf16").display()))?;
        let up = fs::read(prefix.join("up.bf16"))
            .map_err(|e| format!("read {}: {e}", prefix.join("up.bf16").display()))?;
        let down = fs::read(prefix.join("down.bf16"))
            .map_err(|e| format!("read {}: {e}", prefix.join("down.bf16").display()))?;
        if gate.len() != PACK_ROLE_BYTES || up.len() != PACK_ROLE_BYTES || down.len() != PACK_ROLE_BYTES
        {
            return Err(format!(
                "override L{} size mismatch gate={} up={} down={} expected {PACK_ROLE_BYTES}",
                layer.layer,
                gate.len(),
                up.len(),
                down.len()
            ));
        }
        if layer.experts.len() != QWEN80_EXPERTS {
            return Err(format!(
                "loaded expert count {} != {QWEN80_EXPERTS}",
                layer.experts.len()
            ));
        }
        for expert in 0..QWEN80_EXPERTS {
            let g0 = expert * GATE_UP_BYTES;
            let d0 = expert * DOWN_BYTES;
            if layer.experts[expert].gate.len() != GATE_UP_BYTES
                || layer.experts[expert].up.len() != GATE_UP_BYTES
                || layer.experts[expert].down.len() != DOWN_BYTES
            {
                return Err(format!(
                    "source expert {expert} byte geometry mismatch on L{}",
                    layer.layer
                ));
            }
            layer.experts[expert]
                .gate
                .copy_from_slice(&gate[g0..g0 + GATE_UP_BYTES]);
            layer.experts[expert]
                .up
                .copy_from_slice(&up[g0..g0 + GATE_UP_BYTES]);
            layer.experts[expert]
                .down
                .copy_from_slice(&down[d0..d0 + DOWN_BYTES]);
        }
        Ok(())
    }

    fn write_f32le(path: &Path, values: &[f32]) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut bytes = Vec::with_capacity(values.len() * 4);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(path, bytes).map_err(|e| e.to_string())
    }

    fn refuse_if_over_cap(peak: u64) {
        if peak > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            fail(format!(
                "peak RSS {peak} exceeds streamed hard cap {STREAMED_PEAK_RSS_HARD_CAP_BYTES}"
            ));
        }
    }

    pub fn main() {
        let arguments = parse_arguments();
        fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|e| fail(e.to_string()));
        let tokenizer_path = arguments
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| arguments.source_model_dir.join("tokenizer.json"));
        if !tokenizer_path.is_file() {
            fail(format!("tokenizer missing at {}", tokenizer_path.display()));
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .unwrap_or_else(|e| fail(format!("cannot load tokenizer: {e}")));
        let rendered = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            arguments.prompt
        );
        let encoding = tokenizer
            .encode(rendered.as_str(), false)
            .unwrap_or_else(|e| fail(format!("tokenizer encode failed: {e}")));
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            fail("chat-template encoding produced no tokens");
        }
        let seq_len = token_ids.len();
        eprintln!(
            "drift-probe: opening source index; prompt_tokens={seq_len} span=[{}, {})",
            arguments.span_start, arguments.span_end
        );
        let index = SourceBf16Index::open(&arguments.source_model_dir)
            .unwrap_or_else(|e| fail(e.to_string()));
        let probes = vec![("probe".to_string(), token_ids.clone())];
        let mut h_true = embed_probes(&index, &probes)
            .unwrap_or_else(|e| fail(e.to_string()))
            .remove(0);
        let mut h_mixed: Option<Vec<f32>> = None;
        let mut h_null: Option<Vec<f32>> = None;

        let mut per_layer: Vec<Value> = Vec::new();
        let mut true_norms: Vec<f64> = Vec::new();
        let started = Instant::now();

        for layer_idx in 0..QWEN80_LAYERS {
            let t_load = Instant::now();
            let mut layer =
                LoadedLayer::load(&index, layer_idx).unwrap_or_else(|e| fail(e.to_string()));
            let load_secs = t_load.elapsed().as_secs_f64();
            if layer_idx == arguments.span_start {
                h_mixed = Some(h_true.clone());
                h_null = Some(h_true.clone());
            }

            let true_in_last = last_token(&h_true, seq_len).to_vec();
            let t_fwd = Instant::now();
            let true_caps = forward_layer_probe(&layer, &mut h_true, seq_len, 0, 1)
                .unwrap_or_else(|e| fail(e.to_string()));
            let true_out_last = last_token(&h_true, seq_len).to_vec();
            let true_norm = l2_norm(&true_out_last);
            true_norms.push(true_norm);
            let growth_true = if layer_idx == 0 {
                1.0
            } else {
                true_norm / true_norms[layer_idx - 1].max(1e-30)
            };

            let mut mixed_row = json!(null);
            let mut null_row = json!(null);
            let mut mixed_routes: Vec<Value> = Vec::new();
            let mut null_routes: Vec<Value> = Vec::new();
            let in_span = layer_idx >= arguments.span_start && layer_idx < arguments.span_end;
            let after_span = layer_idx >= arguments.span_end && arguments.continue_remaining;

            if in_span {
                apply_pack(&mut layer, &arguments.mixed_override_dir)
                    .unwrap_or_else(|e| fail(e));
                let mixed = h_mixed.as_mut().expect("mixed hidden");
                let mixed_caps = forward_layer_probe(&layer, mixed, seq_len, 0, 1)
                    .unwrap_or_else(|e| fail(e.to_string()));
                let (mean_c, mean_e) = mean_token_cosine_rel_l2(mixed, &h_true, seq_len);
                mixed_row = json!({
                    "last_token_cosine": cosine(last_token(mixed, seq_len), &true_out_last),
                    "last_token_rel_l2": rel_l2(last_token(mixed, seq_len), &true_out_last),
                    "mean_token_cosine": mean_c,
                    "mean_token_rel_l2": mean_e,
                    "last_token_l2": l2_norm(last_token(mixed, seq_len)),
                });
                for (pos, cap) in mixed_caps.iter().enumerate() {
                    mixed_routes.push(json!({
                        "pos": pos,
                        "experts": cap.selected_expert_ids,
                    }));
                }
                apply_pack(&mut layer, &arguments.null_override_dir)
                    .unwrap_or_else(|e| fail(e));
                let null = h_null.as_mut().expect("null hidden");
                let null_caps = forward_layer_probe(&layer, null, seq_len, 0, 1)
                    .unwrap_or_else(|e| fail(e.to_string()));
                let (nmean_c, nmean_e) = mean_token_cosine_rel_l2(null, &h_true, seq_len);
                null_row = json!({
                    "last_token_cosine": cosine(last_token(null, seq_len), &true_out_last),
                    "last_token_rel_l2": rel_l2(last_token(null, seq_len), &true_out_last),
                    "mean_token_cosine": nmean_c,
                    "mean_token_rel_l2": nmean_e,
                    "last_token_l2": l2_norm(last_token(null, seq_len)),
                });
                for (pos, cap) in null_caps.iter().enumerate() {
                    null_routes.push(json!({
                        "pos": pos,
                        "experts": cap.selected_expert_ids,
                    }));
                }
            } else if after_span {
                if let Some(mixed) = h_mixed.as_mut() {
                    forward_layer_probe(&layer, mixed, seq_len, 0, 1)
                        .unwrap_or_else(|e| fail(e.to_string()));
                }
                if let Some(null) = h_null.as_mut() {
                    forward_layer_probe(&layer, null, seq_len, 0, 1)
                        .unwrap_or_else(|e| fail(e.to_string()));
                }
            }

            let fwd_secs = t_fwd.elapsed().as_secs_f64();
            let peak = peak_rss_bytes();
            refuse_if_over_cap(peak);
            let true_routes: Vec<Value> = true_caps
                .iter()
                .enumerate()
                .map(|(pos, cap)| {
                    json!({
                        "pos": pos,
                        "experts": cap.selected_expert_ids,
                    })
                })
                .collect();
            per_layer.push(json!({
                "layer": layer_idx,
                "kind": format!("{:?}", layer.kind),
                "in_span": in_span,
                "load_secs": load_secs,
                "forward_secs": fwd_secs,
                "true_last_token_l2": true_norm,
                "true_residual_growth": growth_true,
                "true_entry_last_token_l2": l2_norm(&true_in_last),
                "mixed": mixed_row,
                "null": null_row,
                "true_routes": true_routes,
                "mixed_routes": mixed_routes,
                "null_routes": null_routes,
                "peak_rss_bytes": peak,
            }));
            eprintln!(
                "drift-probe: L{layer_idx:02} load={load_secs:.2}s fwd={fwd_secs:.2}s rss={:.2} GiB in_span={in_span}",
                peak as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            drop(layer);
        }

        let logits_true = logits_from_final_hidden(&index, last_token(&h_true, seq_len))
            .unwrap_or_else(|e| fail(e.to_string()));
        let mut logits_block = json!({
            "true_top1": topk_ids(&logits_true, 1)[0],
            "true_top5": topk_ids(&logits_true, 5),
        });
        if arguments.continue_remaining {
            if let Some(mixed) = h_mixed.as_ref() {
                let logits_mixed = logits_from_final_hidden(&index, last_token(mixed, seq_len))
                    .unwrap_or_else(|e| fail(e.to_string()));
                let t1 = topk_ids(&logits_true, 1);
                let m1 = topk_ids(&logits_mixed, 1);
                let t5 = topk_ids(&logits_true, 5);
                let m5 = topk_ids(&logits_mixed, 5);
                let overlap = t5.iter().filter(|id| m5.contains(id)).count();
                logits_block["mixed_top1"] = json!(m1[0]);
                logits_block["mixed_top5"] = json!(m5);
                logits_block["mixed_top1_agree"] = json!(t1[0] == m1[0]);
                logits_block["mixed_top5_overlap"] = json!(overlap as f64 / 5.0);
                logits_block["mixed_kl_true_to_mixed"] = json!(kl_true_to_other(&logits_true, &logits_mixed));
                write_f32le(&arguments.output_dir.join("logits_mixed.f32le"), &logits_mixed)
                    .unwrap_or_else(|e| fail(e));
            }
            if let Some(null) = h_null.as_ref() {
                let logits_null = logits_from_final_hidden(&index, last_token(null, seq_len))
                    .unwrap_or_else(|e| fail(e.to_string()));
                let t1 = topk_ids(&logits_true, 1);
                let n1 = topk_ids(&logits_null, 1);
                let t5 = topk_ids(&logits_true, 5);
                let n5 = topk_ids(&logits_null, 5);
                let overlap = t5.iter().filter(|id| n5.contains(id)).count();
                logits_block["null_top1"] = json!(n1[0]);
                logits_block["null_top5"] = json!(n5);
                logits_block["null_top1_agree"] = json!(t1[0] == n1[0]);
                logits_block["null_top5_overlap"] = json!(overlap as f64 / 5.0);
                logits_block["null_kl_true_to_null"] = json!(kl_true_to_other(&logits_true, &logits_null));
                write_f32le(&arguments.output_dir.join("logits_null.f32le"), &logits_null)
                    .unwrap_or_else(|e| fail(e));
            }
        }
        write_f32le(&arguments.output_dir.join("logits_true.f32le"), &logits_true)
            .unwrap_or_else(|e| fail(e));

        let wall = started.elapsed().as_secs_f64();
        let peak = peak_rss_bytes();
        let summary = json!({
            "schema": "hawking.ascension.qwen80_mixed_codec_layer_drift_probe.v1",
            "prompt": arguments.prompt,
            "rendered_prompt": rendered,
            "prompt_token_ids": token_ids,
            "prompt_token_count": seq_len,
            "span_start": arguments.span_start,
            "span_end": arguments.span_end,
            "continue_remaining": arguments.continue_remaining,
            "wall_secs": wall,
            "peak_rss_bytes": peak,
            "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
            "weight_bytes_read": index.bytes_read_total(),
            "layers": per_layer,
            "logits": logits_block,
            "timing_label": "DIRTY_ENGINEERING",
            "timing_note": "host Instant around streamed BF16 layer-major forward; not MTLCommandBuffer GPU time. Other lanes may be running.",
        });
        let out_path = arguments.output_dir.join("probe-result.json");
        fs::write(
            &out_path,
            serde_json::to_string_pretty(&summary).expect("serialize") + "\n",
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        eprintln!(
            "drift-probe: wrote {} ({:.1}s, peak {:.2} GiB)",
            out_path.display(),
            wall,
            peak as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        println!("{}", serde_json::to_string_pretty(&summary).expect("serialize"));
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::main();
}
