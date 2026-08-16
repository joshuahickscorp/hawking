//! Full-depth Q80 mixed-codec teacher-forced drift + optional greedy generate.
//!
//! Closes the three defects of the 4-layer probe:
//! 1. Measures all 48 layers (no geo^44 extrapolation).
//! 2. Builds a matched-magnitude null *distribution* (several seeds) in-process
//!    from (mixed − source) so disk stays one mixed pack per layer.
//! 3. Optional READY/DONE handshake so Python can stream one layer's hats.
//!
//! Generation is mixed-only greedy decode through the same streamed BF16
//! layer-major path with reconstructed hats applied. That is a representation
//! gate, not a packed-runtime gate.
//!
//! Does not pack a full artifact. Does not emit a Metal kernel.
//! Does not raise STREAMED_PEAK_RSS_HARD_CAP_BYTES.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen80 mixed-codec coherence-deep requires macOS");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen80_source_bf16_layer_major::{
        embed_probes, forward_layer_probe, logits_from_final_hidden, peak_rss_bytes, LoadedLayer,
        SourceBf16Index, QWEN80_EXPERTS, QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE,
        STREAMED_PEAK_RSS_HARD_CAP_BYTES,
    };
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use serde_json::{json, Value};
    use std::env;
    use std::fs::{self, File};
    use std::io::Write;
    use std::os::unix::fs::FileExt;
    use std::path::{Path, PathBuf};
    use std::process;
    use std::thread;
    use std::time::{Duration, Instant};
    use tokenizers::Tokenizer;

    const GATE_UP_BYTES: usize = QWEN80_MOE_INTERMEDIATE * QWEN80_HIDDEN * 2;
    const DOWN_BYTES: usize = QWEN80_HIDDEN * QWEN80_MOE_INTERMEDIATE * 2;
    const PACK_ROLE_BYTES: usize = QWEN80_EXPERTS * GATE_UP_BYTES;
    const DEFAULT_NULL_SEEDS: [u64; 7] = [
        20260816, 20260817, 20260818, 20260819, 20260820, 20260821, 20260822,
    ];

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Drift,
        Generate,
    }

    struct Arguments {
        source_model_dir: PathBuf,
        tokenizer_path: Option<PathBuf>,
        prompt: String,
        mixed_override_dir: PathBuf,
        output_dir: PathBuf,
        mode: Mode,
        generate_tokens: usize,
        null_seeds: Vec<u64>,
        wait_for_ready: bool,
        write_done: bool,
        ready_timeout_secs: u64,
        span_start: usize,
        span_end: usize,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen80_mixed_codec_coherence_deep \
         --source-model-dir ABS --mixed-override-dir ABS --output-dir ABS \
         [--mode drift|generate] [--prompt TEXT] [--tokenizer-path ABS] \
         [--generate-tokens N] [--null-seeds CSV] [--no-nulls] \
         [--wait-for-ready] [--write-done] [--ready-timeout-secs N] \
         [--span-start N] [--span-end N]"
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        eprintln!("q80 coherence-deep refused: {}", detail.as_ref());
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
        let mut mixed_override_dir = None;
        let mut output_dir = None;
        let mut mode = Mode::Drift;
        let mut generate_tokens = 0usize;
        let mut null_seeds = DEFAULT_NULL_SEEDS.to_vec();
        let mut wait_for_ready = false;
        let mut write_done = false;
        let mut ready_timeout_secs = 2400u64;
        let mut span_start = 0usize;
        let mut span_end = QWEN80_LAYERS;
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
                "--mixed-override-dir" => {
                    mixed_override_dir = Some(PathBuf::from(args.next().unwrap_or_else(|| {
                        fail("missing value for --mixed-override-dir")
                    })));
                }
                "--output-dir" => {
                    output_dir = Some(PathBuf::from(
                        args.next()
                            .unwrap_or_else(|| fail("missing value for --output-dir")),
                    ));
                }
                "--mode" => {
                    let value = args.next().unwrap_or_else(|| fail("missing value for --mode"));
                    mode = match value.as_str() {
                        "drift" => Mode::Drift,
                        "generate" => Mode::Generate,
                        other => fail(format!("unknown --mode {other}")),
                    };
                }
                "--generate-tokens" => {
                    generate_tokens = parse_usize(
                        &args
                            .next()
                            .unwrap_or_else(|| fail("missing value for --generate-tokens")),
                        "--generate-tokens",
                    );
                }
                "--null-seeds" => {
                    let raw = args
                        .next()
                        .unwrap_or_else(|| fail("missing value for --null-seeds"));
                    null_seeds = raw
                        .split(',')
                        .filter(|s| !s.is_empty())
                        .map(|s| {
                            s.parse::<u64>()
                                .unwrap_or_else(|_| fail(format!("bad null seed {s}")))
                        })
                        .collect();
                }
                "--no-nulls" => null_seeds.clear(),
                "--wait-for-ready" => wait_for_ready = true,
                "--write-done" => write_done = true,
                "--ready-timeout-secs" => {
                    ready_timeout_secs = args
                        .next()
                        .unwrap_or_else(|| fail("missing value for --ready-timeout-secs"))
                        .parse()
                        .unwrap_or_else(|_| fail("--ready-timeout-secs must be u64"));
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
                other => fail(format!("unsupported flag {other}; {}", usage())),
            }
        }
        if mode == Mode::Generate && generate_tokens == 0 {
            generate_tokens = 16;
        }
        if span_end <= span_start || span_end > QWEN80_LAYERS {
            fail(format!(
                "span [{span_start}, {span_end}) is empty or past {QWEN80_LAYERS} layers"
            ));
        }
        Arguments {
            source_model_dir: absolute(
                source_model_dir.unwrap_or_else(|| fail("missing --source-model-dir")),
                "--source-model-dir",
            ),
            tokenizer_path,
            prompt,
            mixed_override_dir: absolute(
                mixed_override_dir.unwrap_or_else(|| fail("missing --mixed-override-dir")),
                "--mixed-override-dir",
            ),
            output_dir: absolute(
                output_dir.unwrap_or_else(|| fail("missing --output-dir")),
                "--output-dir",
            ),
            mode,
            generate_tokens,
            null_seeds,
            wait_for_ready,
            write_done,
            ready_timeout_secs,
            span_start,
            span_end,
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

    fn refuse_if_over_cap(peak: u64) {
        if peak > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            fail(format!(
                "peak RSS {peak} exceeds streamed hard cap {STREAMED_PEAK_RSS_HARD_CAP_BYTES}"
            ));
        }
    }

    fn layer_prefix(dir: &Path, layer: usize) -> PathBuf {
        dir.join(format!("L{layer:02}"))
    }

    fn wait_ready(dir: &Path, layer: usize, timeout_secs: u64) {
        let path = layer_prefix(dir, layer).join("READY");
        let started = Instant::now();
        while !path.is_file() {
            if started.elapsed().as_secs() >= timeout_secs {
                fail(format!(
                    "timeout waiting for {} after {timeout_secs}s (0 fallbacks)",
                    path.display()
                ));
            }
            thread::sleep(Duration::from_millis(100));
        }
    }

    fn write_done(dir: &Path, layer: usize) {
        let prefix = layer_prefix(dir, layer);
        fs::create_dir_all(&prefix).unwrap_or_else(|e| fail(e.to_string()));
        let path = prefix.join("DONE");
        fs::write(&path, b"ok\n").unwrap_or_else(|e| fail(format!("write {}: {e}", path.display())));
    }

    fn open_role(dir: &Path, layer: usize, role: &str, expected: usize) -> File {
        let path = layer_prefix(dir, layer).join(format!("{role}.bf16"));
        let file = File::open(&path)
            .unwrap_or_else(|e| fail(format!("open {}: {e} (0 fallbacks)", path.display())));
        let meta = file
            .metadata()
            .unwrap_or_else(|e| fail(format!("stat {}: {e}", path.display())));
        if meta.len() as usize != expected {
            fail(format!(
                "{} size {} != expected {expected} (0 fallbacks)",
                path.display(),
                meta.len()
            ));
        }
        file
    }

    fn pread_exact(file: &File, buf: &mut [u8], offset: u64, what: &str) {
        file.read_exact_at(buf, offset)
            .unwrap_or_else(|e| fail(format!("pread {what} @{offset}: {e}")));
    }

    fn apply_mixed(layer: &mut LoadedLayer, dir: &Path) {
        let gate_f = open_role(dir, layer.layer, "gate", PACK_ROLE_BYTES);
        let up_f = open_role(dir, layer.layer, "up", PACK_ROLE_BYTES);
        let down_f = open_role(dir, layer.layer, "down", PACK_ROLE_BYTES);
        if layer.experts.len() != QWEN80_EXPERTS {
            fail(format!(
                "loaded expert count {} != {QWEN80_EXPERTS}",
                layer.experts.len()
            ));
        }
        for expert in 0..QWEN80_EXPERTS {
            let g0 = (expert * GATE_UP_BYTES) as u64;
            let d0 = (expert * DOWN_BYTES) as u64;
            if layer.experts[expert].gate.len() != GATE_UP_BYTES
                || layer.experts[expert].up.len() != GATE_UP_BYTES
                || layer.experts[expert].down.len() != DOWN_BYTES
            {
                fail(format!(
                    "source expert {expert} byte geometry mismatch on L{}",
                    layer.layer
                ));
            }
            pread_exact(
                &gate_f,
                &mut layer.experts[expert].gate,
                g0,
                "mixed gate",
            );
            pread_exact(&up_f, &mut layer.experts[expert].up, g0, "mixed up");
            pread_exact(
                &down_f,
                &mut layer.experts[expert].down,
                d0,
                "mixed down",
            );
        }
    }

    fn bf16_to_f32(bytes: &[u8], out: &mut [f32]) {
        if bytes.len() != out.len() * 2 {
            fail("bf16/f32 geometry mismatch");
        }
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            out[i] = f32::from_bits((bits as u32) << 16);
        }
    }

    fn f32_to_bf16(vals: &[f32], out: &mut [u8]) {
        if out.len() != vals.len() * 2 {
            fail("f32/bf16 geometry mismatch");
        }
        for (i, &v) in vals.iter().enumerate() {
            let bits = v.to_bits();
            let rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
            let u = ((bits.wrapping_add(rounding_bias)) >> 16) as u16;
            let b = u.to_le_bytes();
            out[2 * i] = b[0];
            out[2 * i + 1] = b[1];
        }
    }

    fn matched_magnitude_null(src_bf16: &[u8], mixed_bf16: &[u8], dst_bf16: &mut [u8], seed: u64) {
        let n = src_bf16.len() / 2;
        let mut src = vec![0.0f32; n];
        let mut mixed = vec![0.0f32; n];
        let mut err = vec![0.0f32; n];
        bf16_to_f32(src_bf16, &mut src);
        bf16_to_f32(mixed_bf16, &mut mixed);
        for i in 0..n {
            err[i] = mixed[i] - src[i];
        }
        let mut rng = StdRng::seed_from_u64(seed);
        err.shuffle(&mut rng);
        for i in 0..n {
            src[i] += err[i];
        }
        f32_to_bf16(&src, dst_bf16);
    }

    fn apply_null_from_mixed(layer: &mut LoadedLayer, mixed_dir: &Path, seed: u64) {
        let gate_f = open_role(mixed_dir, layer.layer, "gate", PACK_ROLE_BYTES);
        let up_f = open_role(mixed_dir, layer.layer, "up", PACK_ROLE_BYTES);
        let down_f = open_role(mixed_dir, layer.layer, "down", PACK_ROLE_BYTES);
        let mut mixed_gate = vec![0u8; GATE_UP_BYTES];
        let mut mixed_up = vec![0u8; GATE_UP_BYTES];
        let mut mixed_down = vec![0u8; DOWN_BYTES];
        for expert in 0..QWEN80_EXPERTS {
            let g0 = (expert * GATE_UP_BYTES) as u64;
            let d0 = (expert * DOWN_BYTES) as u64;
            pread_exact(&gate_f, &mut mixed_gate, g0, "null gate");
            pread_exact(&up_f, &mut mixed_up, g0, "null up");
            pread_exact(&down_f, &mut mixed_down, d0, "null down");
            let expert_seed = seed
                .wrapping_add((layer.layer as u64).wrapping_mul(10_000))
                .wrapping_add(expert as u64);
            let mut gate_null = vec![0u8; GATE_UP_BYTES];
            let mut up_null = vec![0u8; GATE_UP_BYTES];
            let mut down_null = vec![0u8; DOWN_BYTES];
            matched_magnitude_null(
                &layer.experts[expert].gate,
                &mixed_gate,
                &mut gate_null,
                expert_seed,
            );
            matched_magnitude_null(
                &layer.experts[expert].up,
                &mixed_up,
                &mut up_null,
                expert_seed.wrapping_add(1_000_000),
            );
            matched_magnitude_null(
                &layer.experts[expert].down,
                &mixed_down,
                &mut down_null,
                expert_seed.wrapping_add(2_000_000),
            );
            layer.experts[expert].gate.copy_from_slice(&gate_null);
            layer.experts[expert].up.copy_from_slice(&up_null);
            layer.experts[expert].down.copy_from_slice(&down_null);
        }
    }

    fn stream_metrics(pred: &[f32], truth: &[f32], seq_len: usize) -> Value {
        let (mean_c, mean_e) = mean_token_cosine_rel_l2(pred, truth, seq_len);
        json!({
            "last_token_cosine": cosine(last_token(pred, seq_len), last_token(truth, seq_len)),
            "last_token_rel_l2": rel_l2(last_token(pred, seq_len), last_token(truth, seq_len)),
            "mean_token_cosine": mean_c,
            "mean_token_rel_l2": mean_e,
            "last_token_l2": l2_norm(last_token(pred, seq_len)),
        })
    }

    fn logit_compare(true_logits: &[f32], other: &[f32], prefix: &str) -> Value {
        let t1 = topk_ids(true_logits, 1);
        let o1 = topk_ids(other, 1);
        let t5 = topk_ids(true_logits, 5);
        let o5 = topk_ids(other, 5);
        let overlap = t5.iter().filter(|id| o5.contains(id)).count();
        json!({
            format!("{prefix}_top1"): o1[0],
            format!("{prefix}_top5"): o5,
            format!("{prefix}_top1_agree"): t1[0] == o1[0],
            format!("{prefix}_top5_overlap"): overlap as f64 / 5.0,
            format!("{prefix}_kl_true_to_other"): kl_true_to_other(true_logits, other),
        })
    }

    fn tokenize(arguments: &Arguments) -> (String, Vec<u32>) {
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
        (rendered, token_ids)
    }

    fn append_jsonl(path: &Path, value: &Value) {
        let mut file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .unwrap_or_else(|e| fail(format!("open {}: {e}", path.display())));
        writeln!(file, "{value}").unwrap_or_else(|e| fail(e.to_string()));
    }

    fn run_drift(arguments: &Arguments, index: &SourceBf16Index, token_ids: &[u32]) -> Value {
        let seq_len = token_ids.len();
        let probes = vec![("probe".to_string(), token_ids.to_vec())];
        let mut h_true = embed_probes(index, &probes)
            .unwrap_or_else(|e| fail(e.to_string()))
            .remove(0);
        let mut h_mixed = h_true.clone();
        let mut h_nulls: Vec<Vec<f32>> = arguments
            .null_seeds
            .iter()
            .map(|_| h_true.clone())
            .collect();

        let mut per_layer: Vec<Value> = Vec::new();
        let mut true_norms: Vec<f64> = Vec::new();
        let jsonl = arguments.output_dir.join("drift-layers.jsonl");
        if jsonl.exists() {
            fs::remove_file(&jsonl).ok();
        }

        for layer_idx in 0..QWEN80_LAYERS {
            let t_load = Instant::now();
            let mut layer =
                LoadedLayer::load(index, layer_idx).unwrap_or_else(|e| fail(e.to_string()));
            let load_secs = t_load.elapsed().as_secs_f64();
            let in_span = layer_idx >= arguments.span_start && layer_idx < arguments.span_end;
            // Tile boundary: mixed/null start from the TRUE residual entering
            // span_start, not from the embedding. Without this, a span that
            // begins after layer 0 applies mixed hats to the wrong hidden.
            if layer_idx == arguments.span_start {
                h_mixed = h_true.clone();
                for hidden in h_nulls.iter_mut() {
                    *hidden = h_true.clone();
                }
            }
            if in_span && arguments.wait_for_ready {
                wait_ready(
                    &arguments.mixed_override_dir,
                    layer_idx,
                    arguments.ready_timeout_secs,
                );
            }

            let t_fwd = Instant::now();
            forward_layer_probe(&layer, &mut h_true, seq_len, 0, 1)
                .unwrap_or_else(|e| fail(e.to_string()));
            let true_norm = l2_norm(last_token(&h_true, seq_len));
            true_norms.push(true_norm);
            let growth_true = if layer_idx == 0 {
                1.0
            } else {
                true_norm / true_norms[layer_idx - 1].max(1e-30)
            };

            let mut mixed_row = json!(null);
            let mut null_rows: Vec<Value> = Vec::new();
            if in_span {
                let snap_gate: Vec<Vec<u8>> =
                    layer.experts.iter().map(|e| e.gate.clone()).collect();
                let snap_up: Vec<Vec<u8>> = layer.experts.iter().map(|e| e.up.clone()).collect();
                let snap_down: Vec<Vec<u8>> =
                    layer.experts.iter().map(|e| e.down.clone()).collect();
                apply_mixed(&mut layer, &arguments.mixed_override_dir);
                forward_layer_probe(&layer, &mut h_mixed, seq_len, 0, 1)
                    .unwrap_or_else(|e| fail(e.to_string()));
                mixed_row = stream_metrics(&h_mixed, &h_true, seq_len);
                for (null_i, seed) in arguments.null_seeds.iter().copied().enumerate() {
                    for expert in 0..QWEN80_EXPERTS {
                        layer.experts[expert]
                            .gate
                            .copy_from_slice(&snap_gate[expert]);
                        layer.experts[expert].up.copy_from_slice(&snap_up[expert]);
                        layer.experts[expert]
                            .down
                            .copy_from_slice(&snap_down[expert]);
                    }
                    apply_null_from_mixed(&mut layer, &arguments.mixed_override_dir, seed);
                    forward_layer_probe(&layer, &mut h_nulls[null_i], seq_len, 0, 1)
                        .unwrap_or_else(|e| fail(e.to_string()));
                    null_rows.push(json!({
                        "seed": seed,
                        "metrics": stream_metrics(&h_nulls[null_i], &h_true, seq_len),
                    }));
                }
                drop(snap_gate);
                drop(snap_up);
                drop(snap_down);
                drop(layer);
            } else {
                drop(layer);
            }

            if in_span && arguments.write_done {
                write_done(&arguments.mixed_override_dir, layer_idx);
            }

            let fwd_secs = t_fwd.elapsed().as_secs_f64();
            let peak = peak_rss_bytes();
            refuse_if_over_cap(peak);
            let row = json!({
                "layer": layer_idx,
                "in_span": in_span,
                "load_secs": load_secs,
                "forward_secs": fwd_secs,
                "true_last_token_l2": true_norm,
                "true_residual_growth": growth_true,
                "mixed": mixed_row,
                "nulls": null_rows,
                "peak_rss_bytes": peak,
            });
            append_jsonl(&jsonl, &row);
            per_layer.push(row);
            eprintln!(
                "coherence-deep: L{layer_idx:02} load={load_secs:.2}s fwd={fwd_secs:.2}s rss={:.2} GiB in_span={in_span}",
                peak as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }

        let logits_true = logits_from_final_hidden(index, last_token(&h_true, seq_len))
            .unwrap_or_else(|e| fail(e.to_string()));
        let mut logits_block = json!({
            "true_top1": topk_ids(&logits_true, 1)[0],
            "true_top5": topk_ids(&logits_true, 5),
        });
        let logits_mixed = logits_from_final_hidden(index, last_token(&h_mixed, seq_len))
            .unwrap_or_else(|e| fail(e.to_string()));
        if let Value::Object(map) = logit_compare(&logits_true, &logits_mixed, "mixed") {
            if let Value::Object(dest) = &mut logits_block {
                dest.extend(map);
            }
        }
        let mut null_logits = Vec::new();
        for (i, hidden) in h_nulls.iter().enumerate() {
            let logits = logits_from_final_hidden(index, last_token(hidden, seq_len))
                .unwrap_or_else(|e| fail(e.to_string()));
            let mut block = logit_compare(&logits_true, &logits, "null");
            block["seed"] = json!(arguments.null_seeds[i]);
            null_logits.push(block);
        }
        logits_block["nulls"] = json!(null_logits);
        json!({
            "layers": per_layer,
            "logits": logits_block,
        })
    }

    fn decode_token(tokenizer: &Tokenizer, id: u32) -> String {
        tokenizer
            .decode(&[id], false)
            .unwrap_or_else(|_| format!("<{id}>"))
    }

    fn run_generate(arguments: &Arguments, index: &SourceBf16Index, token_ids: &[u32]) -> Value {
        let tokenizer_path = arguments
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| arguments.source_model_dir.join("tokenizer.json"));
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .unwrap_or_else(|e| fail(format!("cannot load tokenizer: {e}")));
        let mut tokens = token_ids.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut pieces: Vec<String> = Vec::new();
        let mut per_token: Vec<Value> = Vec::new();

        for step in 0..arguments.generate_tokens {
            let seq_len = tokens.len();
            let probes = vec![("probe".to_string(), tokens.clone())];
            let mut hidden = embed_probes(index, &probes)
                .unwrap_or_else(|e| fail(e.to_string()))
                .remove(0);
            for layer_idx in 0..QWEN80_LAYERS {
                if arguments.wait_for_ready {
                    wait_ready(
                        &arguments.mixed_override_dir,
                        layer_idx,
                        arguments.ready_timeout_secs,
                    );
                }
                let mut layer =
                    LoadedLayer::load(index, layer_idx).unwrap_or_else(|e| fail(e.to_string()));
                apply_mixed(&mut layer, &arguments.mixed_override_dir);
                forward_layer_probe(&layer, &mut hidden, seq_len, 0, 1)
                    .unwrap_or_else(|e| fail(e.to_string()));
                if arguments.write_done && step == 0 {
                    write_done(&arguments.mixed_override_dir, layer_idx);
                }
                refuse_if_over_cap(peak_rss_bytes());
                drop(layer);
            }
            let logits = logits_from_final_hidden(index, last_token(&hidden, seq_len))
                .unwrap_or_else(|e| fail(e.to_string()));
            let next = topk_ids(&logits, 1)[0] as u32;
            let piece = decode_token(&tokenizer, next);
            generated.push(next);
            pieces.push(piece.clone());
            tokens.push(next);
            let row = json!({
                "step": step,
                "token_id": next,
                "piece": piece,
                "top5": topk_ids(&logits, 5),
                "logit_top1": logits[next as usize],
                "seq_len_after": tokens.len(),
                "peak_rss_bytes": peak_rss_bytes(),
            });
            append_jsonl(&arguments.output_dir.join("generate-tokens.jsonl"), &row);
            per_token.push(row);
            eprintln!(
                "coherence-deep generate step={step} id={next} piece={piece:?} rss={:.2} GiB",
                peak_rss_bytes() as f64 / (1024.0 * 1024.0 * 1024.0)
            );
        }
        json!({
            "generated_token_ids": generated,
            "generated_text": pieces.join(""),
            "pieces": pieces,
            "tokens": per_token,
        })
    }

    pub fn main() {
        let arguments = parse_arguments();
        fs::create_dir_all(&arguments.output_dir).unwrap_or_else(|e| fail(e.to_string()));
        let (rendered, token_ids) = tokenize(&arguments);
        eprintln!(
            "coherence-deep: mode={:?} prompt_tokens={} span=[{}, {}) nulls={} wait={} generate={}",
            match arguments.mode {
                Mode::Drift => "drift",
                Mode::Generate => "generate",
            },
            token_ids.len(),
            arguments.span_start,
            arguments.span_end,
            arguments.null_seeds.len(),
            arguments.wait_for_ready,
            arguments.generate_tokens
        );
        let index = SourceBf16Index::open(&arguments.source_model_dir)
            .unwrap_or_else(|e| fail(e.to_string()));
        let started = Instant::now();
        let body = match arguments.mode {
            Mode::Drift => run_drift(&arguments, &index, &token_ids),
            Mode::Generate => run_generate(&arguments, &index, &token_ids),
        };
        let wall = started.elapsed().as_secs_f64();
        let peak = peak_rss_bytes();
        let summary = json!({
            "schema": "hawking.ascension.qwen80_mixed_codec_coherence_deep.v1",
            "mode": match arguments.mode { Mode::Drift => "drift", Mode::Generate => "generate" },
            "prompt": arguments.prompt,
            "rendered_prompt": rendered,
            "prompt_token_ids": token_ids,
            "prompt_token_count": token_ids.len(),
            "span_start": arguments.span_start,
            "span_end": arguments.span_end,
            "null_seeds": arguments.null_seeds,
            "generate_tokens": arguments.generate_tokens,
            "wall_secs": wall,
            "peak_rss_bytes": peak,
            "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
            "weight_bytes_read": index.bytes_read_total(),
            "timing_label": "DIRTY_ENGINEERING",
            "timing_note": "host Instant around streamed BF16 layer-major forward; not MTLCommandBuffer GPU time. Other lanes may be running.",
            "result": body,
        });
        let out_path = arguments.output_dir.join("probe-result.json");
        fs::write(
            &out_path,
            serde_json::to_string_pretty(&summary).expect("serialize") + "\n",
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        eprintln!(
            "coherence-deep: wrote {} ({:.1}s, peak {:.2} GiB)",
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
