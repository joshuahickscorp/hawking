//! Layer-major BF16 SOURCE activation route+hidden capture for Q80.
//!
//! Establishes a coherent BF16 source forward (generation) and a layer-major
//! activation capture so activation-aware fit is calibrated on real trajectories.
//!
//! Resource contract (does NOT touch the co-resident memory gate):
//!
//! ```text
//! for layer in 0..48:
//!     range-read layer weights from safetensors shards (~3 GiB experts)
//!     push ALL probe tokens through that layer (probe-local causal state)
//!     retain router-input hiddens under per-expert first-N (after top-k known)
//!     write retained hidden rows + full route membership
//!     free this layer's retained hidden payloads
//!     free the layer weights
//! ```
//!
//! Hidden retention is **per-expert first-N** (default N=64), not a global
//! stratified position set. A shared per-layer budget starves experts when a
//! larger corpus spreads routing; first-N after routing is known guarantees up
//! to N retained rows per (layer, expert) and is deterministic. Q80 has 512
//! experts and top-10 routing — far worse under the retired per-layer scheme.
//!
//! Output layout matches the Q80 broad all-layer capture schema so
//! `lab/operators/ascension_qwen80_activation_weighted_svd_repack.py` can
//! consume the run directory with no repack changes.
//!
//! Modes:
//! * `coherence` — greedy top-1 on the source chat template
//!   ("What is the capital of France?" → must start with Paris)
//! * `capture` — full (or max-probes-bounded) activation capture

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen80 source BF16 layer-major capture requires macOS");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen80_source_bf16_layer_major::{
        capture_all_layers, embed_probes, format_capture_progress, greedy_decode_user_prompt,
        is_coherent_paris_continuation, max_hidden_tokens_per_expert_within_streamed_cap,
        peak_rss_bytes, retained_hidden_relative_path, worst_case_retained_hidden_bytes_per_layer,
        worst_case_unique_rows_per_layer, write_retained_hidden_f32le, LayerTokenCapture,
        SourceBf16Index, DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, QWEN80_EXPERTS, QWEN80_HIDDEN,
        QWEN80_LAYERS, QWEN80_TOP_K, STREAMED_PEAK_RSS_HARD_CAP_BYTES,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::HashSet;
    use std::env;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process;
    use std::time::Instant;

    const ALL_LAYER_INPUT_SCHEMA: &str =
        "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_input.v1";
    const BROAD_INPUT_SCHEMA_Q30_COMPAT: &str =
        "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_input.v1";
    const RESULT_SCHEMA: &str =
        "hawking.ascension.qwen80_broad_activation_all_layer_route_capture_result.v1";
    const CAPTURE_PROTOCOL_REVISION: &str =
        "q80-source-bf16-layer-major-route-hidden-capture-per-expert-first-n-v1";
    const TRACE_STATUS: &str = "NEW_DIAGNOSTIC_NOT_HISTORICAL";
    const PARIS_PROMPT: &str = "What is the capital of France?";

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum Mode {
        Capture,
        Coherence,
    }

    struct Arguments {
        mode: Mode,
        source_model_dir: PathBuf,
        input_json: Option<PathBuf>,
        output_dir: Option<PathBuf>,
        max_hidden_tokens_per_expert: usize,
        max_probes: Option<usize>,
        max_new_tokens: usize,
        tokenizer_path: Option<PathBuf>,
    }

    fn usage() -> &'static str {
        "usage:\n  coherence: ascension_qwen80_source_bf16_layer_major_activation_capture \\\n\
         \x20   --mode coherence --source-model-dir ABSOLUTE_PATH \\\n\
         \x20   [--tokenizer-path ABSOLUTE_PATH] [--max-new-tokens N]\n\
         capture: ... --mode capture --source-model-dir ABSOLUTE_PATH \\\n\
         \x20   --input-json ABSOLUTE_PATH --output-dir ABSOLUTE_PATH \\\n\
         \x20   [--max-hidden-tokens-per-expert N] [--max-probes N]\n\
         note: --max-hidden-tokens-per-layer is retired; use --max-hidden-tokens-per-expert"
    }

    fn required<T>(value: Option<T>, flag: &str) -> Result<T, String> {
        value.ok_or_else(|| format!("missing {flag}; {}", usage()))
    }

    fn parse_usize(value: &str, flag: &str) -> Result<usize, String> {
        value
            .parse::<usize>()
            .map_err(|_| format!("{flag} must be an unsigned decimal integer; {}", usage()))
    }

    fn absolute(path: PathBuf, flag: &str) -> Result<PathBuf, String> {
        if !path.is_absolute() {
            return Err(format!("{flag} must be an absolute path; {}", usage()));
        }
        Ok(path)
    }

    fn parse_arguments() -> Result<Arguments, String> {
        let mut mode = Mode::Coherence;
        let mut source_model_dir = None;
        let mut input_json = None;
        let mut output_dir = None;
        let mut max_hidden_tokens_per_expert = DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT;
        let mut max_probes = None;
        let mut max_new_tokens = 16usize;
        let mut tokenizer_path = None;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--mode" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for --mode; {}", usage()))?;
                    mode = match value.as_str() {
                        "capture" => Mode::Capture,
                        "coherence" => Mode::Coherence,
                        other => {
                            return Err(format!(
                                "--mode must be capture or coherence (got {other:?})"
                            ))
                        }
                    };
                }
                "--source-model-dir" => {
                    let value = args.next().ok_or_else(|| {
                        format!("missing value for --source-model-dir; {}", usage())
                    })?;
                    if source_model_dir.replace(PathBuf::from(value)).is_some() {
                        return Err("--source-model-dir supplied more than once".into());
                    }
                }
                "--input-json" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for --input-json; {}", usage()))?;
                    if input_json.replace(PathBuf::from(value)).is_some() {
                        return Err("--input-json supplied more than once".into());
                    }
                }
                "--output-dir" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for --output-dir; {}", usage()))?;
                    if output_dir.replace(PathBuf::from(value)).is_some() {
                        return Err("--output-dir supplied more than once".into());
                    }
                }
                "--max-hidden-tokens-per-layer" => {
                    // Retired: a per-layer cap shared across 512 experts was the
                    // root cause of unusable organs on Q80 (well under 1 row/organ).
                    // Refuse rather than silently reinterpreting the number as per-expert.
                    return Err(
                        "--max-hidden-tokens-per-layer is retired: retention is now \
                         per-expert first-N (deterministic). Use \
                         --max-hidden-tokens-per-expert N (default 64). A shared \
                         per-layer budget cannot guarantee rows-per-expert."
                            .into(),
                    );
                }
                "--max-hidden-tokens-per-expert" => {
                    let value = args.next().ok_or_else(|| {
                        format!(
                            "missing value for --max-hidden-tokens-per-expert; {}",
                            usage()
                        )
                    })?;
                    max_hidden_tokens_per_expert =
                        parse_usize(&value, "--max-hidden-tokens-per-expert")?;
                }
                "--max-probes" => {
                    let value = args
                        .next()
                        .ok_or_else(|| format!("missing value for --max-probes; {}", usage()))?;
                    max_probes = Some(parse_usize(&value, "--max-probes")?);
                }
                "--max-new-tokens" => {
                    let value = args.next().ok_or_else(|| {
                        format!("missing value for --max-new-tokens; {}", usage())
                    })?;
                    max_new_tokens = parse_usize(&value, "--max-new-tokens")?;
                }
                "--tokenizer-path" => {
                    let value = args.next().ok_or_else(|| {
                        format!("missing value for --tokenizer-path; {}", usage())
                    })?;
                    if tokenizer_path.replace(PathBuf::from(value)).is_some() {
                        return Err("--tokenizer-path supplied more than once".into());
                    }
                }
                "--help" | "-h" => return Err(usage().into()),
                other => return Err(format!("unsupported option {other:?}; {}", usage())),
            }
        }
        if max_hidden_tokens_per_expert == 0 {
            return Err("--max-hidden-tokens-per-expert must be positive".into());
        }
        if max_new_tokens == 0 {
            return Err("--max-new-tokens must be positive".into());
        }
        // Worst-case unique retained rows/layer = N * experts (no multi-route
        // credit). Co-routing typically keeps realised rows well below this.
        // Per-layer flush: only one layer's retained rows are resident.
        // At N=64 × 512 × 2048 × 4 ≈ 0.25 GiB; STREAMED_PEAK_RSS is 16 GiB.
        // Do not raise STREAMED_PEAK_RSS_HARD_CAP_BYTES to fit a larger N.
        if !max_hidden_tokens_per_expert_within_streamed_cap(max_hidden_tokens_per_expert) {
            let budget_bytes =
                worst_case_retained_hidden_bytes_per_layer(max_hidden_tokens_per_expert);
            let max_hidden_budget_bytes = STREAMED_PEAK_RSS_HARD_CAP_BYTES as usize;
            return Err(format!(
                "--max-hidden-tokens-per-expert {max_hidden_tokens_per_expert} implies \
                 worst-case ~{budget_bytes} retained hidden bytes \
                 (> {max_hidden_budget_bytes} streamed RSS hard cap); lower N"
            ));
        }
        Ok(Arguments {
            mode,
            source_model_dir: absolute(
                required(source_model_dir, "--source-model-dir")?,
                "--source-model-dir",
            )?,
            input_json: match input_json {
                Some(p) => Some(absolute(p, "--input-json")?),
                None => None,
            },
            output_dir: match output_dir {
                Some(p) => Some(absolute(p, "--output-dir")?),
                None => None,
            },
            max_hidden_tokens_per_expert,
            max_probes,
            max_new_tokens,
            tokenizer_path: match tokenizer_path {
                Some(p) => Some(absolute(p, "--tokenizer-path")?),
                None => None,
            },
        })
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let mut file =
            File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
        let mut digest = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let n = file
                .read(&mut chunk)
                .map_err(|e| format!("cannot hash {}: {e}", path.display()))?;
            if n == 0 {
                break;
            }
            digest.update(&chunk[..n]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn current_executable_sha256() -> Result<String, String> {
        let path = env::current_exe().map_err(|e| format!("cannot resolve executable: {e}"))?;
        sha256_file(&path)
    }

    fn fail(msg: impl AsRef<str>) -> ! {
        eprintln!("qwen80 source BF16 layer-major: {}", msg.as_ref());
        process::exit(1);
    }

    fn refuse_if_resident_load(peak: u64) {
        if peak > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            fail(format!(
                "peak RSS {peak} exceeds streamed hard cap {STREAMED_PEAK_RSS_HARD_CAP_BYTES}; \
                 looks like a resident load — refusing"
            ));
        }
    }

    fn parse_token_ids(value: &Value, probe_id: &str) -> Result<Vec<u32>, String> {
        let ids = value
            .as_array()
            .ok_or_else(|| format!("{probe_id} source template token IDs are not an array"))?;
        if ids.is_empty() {
            return Err(format!("{probe_id} source template token IDs are empty"));
        }
        ids.iter()
            .map(|id| {
                id.as_u64()
                    .and_then(|v| u32::try_from(v).ok())
                    .ok_or_else(|| format!("{probe_id} contains an invalid token ID"))
            })
            .collect()
    }

    fn parse_input(
        path: &Path,
        max_probes: Option<usize>,
    ) -> Result<(Value, Vec<(String, Vec<u32>)>), String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("cannot read capture input {}: {e}", path.display()))?;
        let document: Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("capture input is not JSON: {e}"))?;
        let schema = document
            .get("schema")
            .and_then(Value::as_str)
            .ok_or_else(|| "capture input lacks schema".to_string())?;
        if schema != ALL_LAYER_INPUT_SCHEMA && schema != BROAD_INPUT_SCHEMA_Q30_COMPAT {
            return Err(format!(
                "capture input schema is not a known all-layer route-capture input schema (got {schema})"
            ));
        }
        // Soft claim_boundary checks when present (Q80 prepare may set them).
        if let Some(status) = document.get("status").and_then(Value::as_str) {
            if status != TRACE_STATUS && !status.contains("DIAGNOSTIC") && !status.contains("READY")
            {
                // accept READY_* from prepare scripts
            }
        }
        let probes = document
            .get("probes")
            .and_then(Value::as_array)
            .ok_or_else(|| "capture input lacks probes".to_string())?;
        if probes.is_empty() {
            return Err("capture input has zero probes".into());
        }
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(probes.len());
        for probe in probes {
            let probe_id = probe
                .get("probe_id")
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
                .ok_or_else(|| "capture input probe lacks a non-empty probe_id".to_string())?
                .to_string();
            if !seen.insert(probe_id.clone()) {
                return Err(format!("capture input repeats probe {probe_id:?}"));
            }
            let token_ids = parse_token_ids(
                probe
                    .pointer("/source_one_user_native_prompt/token_ids")
                    .or_else(|| probe.get("token_ids"))
                    .ok_or_else(|| format!("{probe_id} lacks source one-user native token IDs"))?,
                &probe_id,
            )?;
            if token_ids.len() < 2 {
                return Err(format!(
                    "{probe_id} is too short for activation capture ({} tokens)",
                    token_ids.len()
                ));
            }
            result.push((probe_id, token_ids));
            if let Some(max) = max_probes {
                if result.len() >= max {
                    break;
                }
            }
        }
        Ok((document, result))
    }

    /// Percentile of a non-empty ascending-sorted slice (nearest-rank).
    fn percentile_sorted(sorted: &[usize], p: f64) -> usize {
        if sorted.is_empty() {
            return 0;
        }
        let rank = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        sorted[rank.min(sorted.len() - 1)]
    }

    /// Per-(layer, expert) retained-hidden hit counts — the quantity organs fit on.
    ///
    /// Matches the repack's collect path: a retained token contributes one row to
    /// every expert in its top-k for that layer. Sized from QWEN80_EXPERTS (512).
    fn n_fit_distribution(captures: &[Vec<Vec<LayerTokenCapture>>]) -> Value {
        let n_layers = captures
            .first()
            .and_then(|p| p.first())
            .map(|t| t.len())
            .unwrap_or(0);
        let mut counts: Vec<usize> = Vec::with_capacity(n_layers * QWEN80_EXPERTS);
        for layer in 0..n_layers {
            let mut per_expert = vec![0usize; QWEN80_EXPERTS];
            for probe_caps in captures {
                for token_caps in probe_caps {
                    let layer_cap = &token_caps[layer];
                    if !layer_cap.hidden_retained {
                        continue;
                    }
                    for &expert in &layer_cap.selected_expert_ids {
                        let e = expert as usize;
                        if e < QWEN80_EXPERTS {
                            per_expert[e] += 1;
                        }
                    }
                }
            }
            counts.extend(per_expert);
        }
        let mut sorted = counts.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        let below_8 = sorted.iter().filter(|&&c| c < 8).count();
        let below_16 = sorted.iter().filter(|&&c| c < 16).count();
        let below_32 = sorted.iter().filter(|&&c| c < 32).count();
        let at_or_above_64 = sorted.iter().filter(|&&c| c >= 64).count();
        let zero = sorted.iter().filter(|&&c| c == 0).count();
        json!({
            "unit": "retained_hidden_rows_per_layer_expert",
            "n_layer_expert_pairs": n,
            "experts": QWEN80_EXPERTS,
            "layers": n_layers,
            "p10": percentile_sorted(&sorted, 10.0),
            "p50": percentile_sorted(&sorted, 50.0),
            "p90": percentile_sorted(&sorted, 90.0),
            "max": sorted.last().copied().unwrap_or(0),
            "min": sorted.first().copied().unwrap_or(0),
            "mean": if n == 0 {
                0.0
            } else {
                counts.iter().sum::<usize>() as f64 / n as f64
            },
            "frac_below_8": if n == 0 { 0.0 } else { below_8 as f64 / n as f64 },
            "frac_below_16": if n == 0 { 0.0 } else { below_16 as f64 / n as f64 },
            "frac_below_32": if n == 0 { 0.0 } else { below_32 as f64 / n as f64 },
            "frac_at_or_above_64": if n == 0 {
                0.0
            } else {
                at_or_above_64 as f64 / n as f64
            },
            "pct_zero": if n == 0 { 0.0 } else { 100.0 * zero as f64 / n as f64 },
            "count_below_8": below_8,
            "count_below_16": below_16,
            "count_below_32": below_32,
            "count_at_or_above_64": at_or_above_64,
            "count_zero": zero,
        })
    }

    fn write_json_new(path: &Path, value: &Value) -> Result<(), String> {
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
        file.write_all(text.as_bytes())
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        file.write_all(b"\n")
            .map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        Ok(())
    }

    pub fn main() {
        let arguments = parse_arguments().unwrap_or_else(|e| fail(e));
        match arguments.mode {
            Mode::Coherence => run_coherence(&arguments),
            Mode::Capture => run_capture(&arguments),
        }
    }

    fn run_coherence(arguments: &Arguments) {
        let tokenizer = arguments
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| arguments.source_model_dir.join("tokenizer.json"));
        if !tokenizer.is_file() {
            fail(format!("tokenizer missing at {}", tokenizer.display()));
        }
        eprintln!(
            "coherence: opening source index at {}",
            arguments.source_model_dir.display()
        );
        let index = SourceBf16Index::open(&arguments.source_model_dir)
            .unwrap_or_else(|e| fail(e.to_string()));
        eprintln!(
            "coherence: indexed {} tensors; greedy decode max_new_tokens={}",
            index.tensor_count(),
            arguments.max_new_tokens
        );
        let started = Instant::now();
        let result =
            greedy_decode_user_prompt(&index, &tokenizer, PARIS_PROMPT, arguments.max_new_tokens)
                .unwrap_or_else(|e| fail(e.to_string()));
        let wall = started.elapsed();
        let peak = peak_rss_bytes().max(result.peak_rss_bytes);
        refuse_if_resident_load(peak);
        let ok = is_coherent_paris_continuation(&result.continuation_text);
        let tokens_generated = result.generated_token_ids.len().max(1);
        let tokens_per_sec = tokens_generated as f64 / wall.as_secs_f64().max(1e-9);
        let stream_gib = result.weight_bytes_read as f64 / (1024.0 * 1024.0 * 1024.0);
        let stream_gib_s = stream_gib / wall.as_secs_f64().max(1e-9);
        let summary = json!({
            "mode": "coherence",
            "prompt": PARIS_PROMPT,
            "rendered_prompt": result.rendered_prompt,
            "prompt_token_count": result.prompt_token_count,
            "generated_token_ids": result.generated_token_ids,
            "top1_token_id": result.generated_token_ids.first().copied(),
            "continuation_text": result.continuation_text,
            "coherent_paris_top1": ok,
            "wall_clock_secs": wall.as_secs_f64(),
            "tokens_per_sec": tokens_per_sec,
            "weight_bytes_read": result.weight_bytes_read,
            "weight_gib_read": stream_gib,
            "stream_gib_per_s_wall": stream_gib_s,
            "peak_rss_bytes": peak,
            "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
            "streamed_resource_contract": {
                "co_resident_full_source_load": false,
                "layer_major_range_read": true,
                "peak_rss_hard_cap_bytes": STREAMED_PEAK_RSS_HARD_CAP_BYTES,
            },
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&summary).expect("summary serializes")
        );
        if !ok {
            eprintln!(
                "COHERENCE FAILED: top-1 continuation was {:?}; expected Paris (or obvious variant). \
                 Forward is WRONG — do not proceed to full capture.",
                result.continuation_text
            );
            process::exit(3);
        }
        eprintln!(
            "COHERENCE PASSED: continuation {:?} (peak RSS {:.2} GiB, {:.1}s, {:.3} tok/s)",
            result.continuation_text,
            peak as f64 / (1024.0 * 1024.0 * 1024.0),
            wall.as_secs_f64(),
            tokens_per_sec
        );
    }

    fn run_capture(arguments: &Arguments) {
        let input_json =
            required(arguments.input_json.clone(), "--input-json").unwrap_or_else(|e| fail(e));
        let output_dir =
            required(arguments.output_dir.clone(), "--output-dir").unwrap_or_else(|e| fail(e));
        if output_dir.exists() {
            fail(format!(
                "refusing to reuse or overwrite capture output directory {}",
                output_dir.display()
            ));
        }
        if !output_dir.parent().is_some_and(|parent| parent.is_dir()) {
            fail("capture output parent must already exist");
        }

        let (input, probes) =
            parse_input(&input_json, arguments.max_probes).unwrap_or_else(|e| fail(e));
        let input_sha256 = sha256_file(&input_json).unwrap_or_else(|e| fail(e));
        let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
        let naive_hidden_bytes = total_tokens
            .saturating_mul(QWEN80_LAYERS)
            .saturating_mul(QWEN80_HIDDEN)
            .saturating_mul(4);
        // Worst-case unique rows/layer under first-N (no multi-route credit).
        // Per-layer flush: the RAM budget is one layer, not ×48.
        let worst_case_rows_per_layer =
            worst_case_unique_rows_per_layer(arguments.max_hidden_tokens_per_expert)
                .min(total_tokens);
        let retained_hidden_budget_bytes =
            worst_case_retained_hidden_bytes_per_layer(arguments.max_hidden_tokens_per_expert);

        fs::create_dir(&output_dir).unwrap_or_else(|e| {
            fail(format!(
                "cannot create capture output directory {}: {e}",
                output_dir.display()
            ))
        });
        let executable_sha256 = current_executable_sha256().unwrap_or_else(|e| fail(e));
        let index = SourceBf16Index::open(&arguments.source_model_dir)
            .unwrap_or_else(|e| fail(e.to_string()));
        if index.tensor_count() < 70_000 {
            fail(format!(
                "source index has only {} tensors; expected full Q80 catalog (~74391)",
                index.tensor_count()
            ));
        }

        eprintln!(
            "{}",
            format_capture_progress(
                probes.len(),
                total_tokens,
                arguments.max_hidden_tokens_per_expert,
                index.tensor_count(),
            )
        );

        let started = Instant::now();
        let mut hiddens = embed_probes(&index, &probes).unwrap_or_else(|e| fail(e.to_string()));
        let mut max_layer_resident = 0u64;
        let mut on_layer = |layer: usize, loaded: &hawking_core::model::qwen80_source_bf16_layer_major::LoadedLayer, _telem: &hawking_core::model::qwen80_source_bf16_layer_major::StreamTelemetry| {
            max_layer_resident = max_layer_resident.max(loaded.resident_bytes);
            if layer == 0 || layer == 23 || layer == 47 || layer % 8 == 0 {
                eprintln!(
                    "  layer {layer:02}/47 resident={:.2} GiB load={:.1}s peak_rss={:.2} GiB",
                    loaded.resident_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    loaded.load_secs,
                    peak_rss_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                );
            }
            refuse_if_resident_load(peak_rss_bytes());
        };
        // hidden_writes[probe][position][layer] = metadata written during flush.
        let mut hidden_writes: Vec<Vec<Vec<Option<(String, String, usize, usize)>>>> = probes
            .iter()
            .map(|(_, toks)| {
                (0..toks.len())
                    .map(|_| Vec::with_capacity(QWEN80_LAYERS))
                    .collect()
            })
            .collect();
        let mut hidden_bytes_written = 0usize;
        let mut hidden_rows_retained_total = 0usize;
        let mut hidden_rows_per_layer = vec![0usize; QWEN80_LAYERS];
        let mut on_layer_flush = |layer_idx: usize,
                                  captures: &mut [Vec<Vec<LayerTokenCapture>>]|
         -> hawking_core::Result<()> {
            // Placeholder slot per token so hidden_writes[pi][pos][layer] is dense.
            for (pi, (_, token_ids)) in probes.iter().enumerate() {
                for slot in hidden_writes[pi].iter_mut().take(token_ids.len()) {
                    slot.push(None);
                }
            }
            for (probe_id, _) in probes.iter() {
                let dir = output_dir.join(format!("hidden/L{layer_idx:02}/{probe_id}"));
                std::fs::create_dir_all(&dir).map_err(|e| {
                    hawking_core::Error::Model(format!(
                        "cannot create hidden dir {}: {e}",
                        dir.display()
                    ))
                })?;
            }
            let mut jobs: Vec<(usize, usize, String)> = Vec::new();
            for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
                for pos in 0..token_ids.len() {
                    let cap = captures
                        .get(pi)
                        .and_then(|p| p.get(pos))
                        .and_then(|t| t.get(layer_idx))
                        .ok_or_else(|| {
                            hawking_core::Error::Model(format!(
                                "flush missing capture {probe_id}@{pos} L{layer_idx}"
                            ))
                        })?;
                    if !cap.hidden_retained {
                        continue;
                    }
                    if cap.router_input_hidden.len() != QWEN80_HIDDEN {
                        return Err(hawking_core::Error::Model(format!(
                            "{probe_id}@{pos} L{layer_idx}: retained hidden width {} != {QWEN80_HIDDEN}",
                            cap.router_input_hidden.len()
                        )));
                    }
                    jobs.push((
                        pi,
                        pos,
                        retained_hidden_relative_path(layer_idx, probe_id, pos),
                    ));
                }
            }
            let n_jobs = jobs.len();
            if n_jobs > 0 {
                let n_workers = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .clamp(1, 16)
                    .min(n_jobs);
                let chunk = n_jobs.div_ceil(n_workers);
                let mut results: Vec<Option<(String, usize, usize)>> = vec![None; n_jobs];
                let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
                let caps: &[Vec<Vec<LayerTokenCapture>>] = captures;
                std::thread::scope(|scope| {
                    for (wi, result_chunk) in results.chunks_mut(chunk).enumerate() {
                        let start = wi * chunk;
                        let my_jobs = &jobs[start..start + result_chunk.len()];
                        let err = &err;
                        let output_dir = &output_dir;
                        scope.spawn(move || {
                            for (local, (pi, pos, rel)) in my_jobs.iter().enumerate() {
                                let hidden = &caps[*pi][*pos][layer_idx].router_input_hidden;
                                match write_retained_hidden_f32le(&output_dir.join(rel), hidden) {
                                    Ok((sha, bytes)) => {
                                        result_chunk[local] = Some((sha, bytes, hidden.len()));
                                    }
                                    Err(e) => {
                                        if let Ok(mut g) = err.lock() {
                                            *g = Some(e.to_string());
                                        }
                                        return;
                                    }
                                }
                            }
                        });
                    }
                });
                if let Some(msg) = err.into_inner().unwrap_or(None) {
                    return Err(hawking_core::Error::Model(msg));
                }
                for (job, res) in jobs.iter().zip(results.into_iter()) {
                    let (sha, bytes, elems) = res.ok_or_else(|| {
                        hawking_core::Error::Model("parallel hidden write missing result".into())
                    })?;
                    hidden_bytes_written = hidden_bytes_written.saturating_add(bytes);
                    hidden_rows_retained_total = hidden_rows_retained_total.saturating_add(1);
                    if layer_idx < QWEN80_LAYERS {
                        hidden_rows_per_layer[layer_idx] =
                            hidden_rows_per_layer[layer_idx].saturating_add(1);
                    }
                    hidden_writes[job.0][job.1][layer_idx] =
                        Some((job.2.clone(), sha, bytes, elems));
                }
            }
            refuse_if_resident_load(peak_rss_bytes());
            Ok(())
        };
        let (captures, telem) = capture_all_layers(
            &index,
            &probes,
            &mut hiddens,
            arguments.max_hidden_tokens_per_expert,
            Some(&mut on_layer),
            Some(&mut on_layer_flush),
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        drop(hiddens);

        let mut probe_rows = Vec::with_capacity(probes.len());
        let mut tokens_executed = 0usize;
        let mut route_membership_total = 0usize;

        let layers_executed = telem.layers;
        for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
            let mut steps = Vec::with_capacity(token_ids.len());
            for (pos, &token_id) in token_ids.iter().enumerate() {
                let layer_caps = &captures[pi][pos];
                if layer_caps.len() != layers_executed {
                    fail(format!(
                        "{probe_id}@{pos}: captured {} layers, expected {layers_executed}",
                        layer_caps.len()
                    ));
                }
                let mut layer_rows = Vec::with_capacity(QWEN80_LAYERS);
                let mut any_layer_retained = false;
                for layer_cap in layer_caps {
                    if layer_cap.selected_expert_ids.len() != QWEN80_TOP_K
                        || layer_cap.normalized_route_weights.len() != QWEN80_TOP_K
                    {
                        fail(format!(
                            "{probe_id}@{pos} L{}: route membership is not top-{QWEN80_TOP_K}",
                            layer_cap.layer
                        ));
                    }
                    route_membership_total =
                        route_membership_total.saturating_add(layer_cap.selected_expert_ids.len());
                    // Per-expert first-N is decided inside capture_all_layers after
                    // routing. Rows are written during the per-layer flush; the
                    // retain flag (not the now-empty payload) is the authority.
                    let store_hidden = layer_cap.hidden_retained;
                    if store_hidden {
                        any_layer_retained = true;
                    }
                    let hidden_meta = if store_hidden {
                        let written = hidden_writes
                            .get(pi)
                            .and_then(|p| p.get(pos))
                            .and_then(|t| t.get(layer_cap.layer))
                            .and_then(|slot| slot.as_ref())
                            .unwrap_or_else(|| {
                                fail(format!(
                                    "{probe_id}@{pos} L{}: retained but not written during flush",
                                    layer_cap.layer
                                ))
                            });
                        Some(json!({
                            "relative_path": written.0,
                            "sha256": written.1,
                            "bytes": written.2,
                            "elements": written.3,
                            "source": "BF16-source layer-streamed post-attention RMSNorm buffer at this layer, copied after router top-k and before expert wave",
                        }))
                    } else {
                        None
                    };
                    layer_rows.push(json!({
                        "layer": layer_cap.layer,
                        "selected_expert_ids": layer_cap.selected_expert_ids,
                        "normalized_route_weights": layer_cap.normalized_route_weights,
                        "router_input_hidden_f32le": hidden_meta,
                        "hidden_retained": store_hidden,
                    }));
                }
                steps.push(json!({
                    "position": pos,
                    "input_token_id": token_id,
                    "layers": layer_rows,
                    "all_48_layers_executed": layers_executed == QWEN80_LAYERS,
                    "final_norm_lm_head_sampler_executed": false,
                    "autoregressive_feedback_or_generation_not_executed": true,
                    "hidden_retained_for_this_token": any_layer_retained,
                }));
                tokens_executed += 1;
            }
            probe_rows.push(json!({
                "probe_id": probe_id,
                "source_one_user_native_prompt_token_count": steps.len(),
                "steps": steps,
            }));
        }

        let wall = started.elapsed();
        let peak = peak_rss_bytes().max(telem.peak_rss_bytes);
        refuse_if_resident_load(peak);

        let expected_route_slots = tokens_executed
            .saturating_mul(layers_executed)
            .saturating_mul(QWEN80_TOP_K);
        if route_membership_total != expected_route_slots {
            fail(format!(
                "route membership total {route_membership_total} != expected {expected_route_slots}"
            ));
        }
        let expected_hidden_bytes = hidden_rows_retained_total
            .saturating_mul(QWEN80_HIDDEN)
            .saturating_mul(4);
        if hidden_bytes_written != expected_hidden_bytes {
            fail(format!(
                "hidden bytes written {hidden_bytes_written} != expected {expected_hidden_bytes}"
            ));
        }
        let n_fit_dist = n_fit_distribution(&captures);
        eprintln!(
            "n_fit distribution (retained rows per layer×expert): {}",
            serde_json::to_string(&n_fit_dist).unwrap_or_else(|_| "{}".into())
        );

        let free_crossover = telem.free_corpus_crossover_tokens();
        let claim_boundary = json!({
            "new_diagnostic_not_historical": true,
            "all_48_layers_embedding_mixer_postnorm_router_and_expert_wave_executed": true,
            "source_bf16_layer_major_stream": true,
            "co_resident_full_source_load": false,
            "final_norm_lm_head_sampler_not_executed_during_capture": true,
            "autoregressive_feedback_generation_hcli_and_tps_not_executed": true,
            "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": true,
            "diagnostic_activation_pricing_only": true,
            "broad_activation_diversity_capture": true,
            "bounded_hidden_storage_not_unbounded_raw_dump": true,
            "source_tokenizer_one_user_native_prompts": true,
            "per_expert_first_n_retention": true,
        });

        let result = json!({
            "schema": RESULT_SCHEMA,
            "status": "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_SOURCE_BF16_LAYER_MAJOR_ALL_LAYER_ROUTE_AND_HIDDEN_CAPTURE",
            "capture_protocol_revision": CAPTURE_PROTOCOL_REVISION,
            "input": {
                "path": input_json,
                "sha256": input_sha256,
                "schema": input.get("schema"),
                "status": input.get("status"),
                "max_probes_applied": arguments.max_probes,
            },
            "runtime_binding": {
                "source_model_dir": arguments.source_model_dir,
                "runtime_executable_sha256": executable_sha256,
                "architecture": "Qwen3NextForCausalLM",
                "weight_backend": "source_bf16_safetensors_range_read",
                "metal_not_used": true,
                "packed_complete_binary_not_opened": true,
                "layers": QWEN80_LAYERS,
                "hidden": QWEN80_HIDDEN,
                "experts": QWEN80_EXPERTS,
                "top_k": QWEN80_TOP_K,
                "source_tensor_count": index.tensor_count(),
            },
            "stream_telemetry": {
                "weight_bytes_read": telem.weight_bytes_read,
                "weight_gib_read": telem.weight_bytes_read as f64 / (1024.0 * 1024.0 * 1024.0),
                "load_secs": telem.load_secs,
                "compute_secs": telem.compute_secs,
                "wall_secs": telem.wall_secs,
                "stream_gib_per_s": telem.stream_gib_per_s(),
                "free_corpus_crossover_tokens": free_crossover,
                "max_layer_resident_bytes": telem.max_layer_resident_bytes.max(max_layer_resident),
                "tokens_executed": tokens_executed,
                "layers_executed": layers_executed,
                "moe_workers": telem.phase.moe_workers,
                "phase_secs": {
                    "mixer": telem.phase.mixer_secs,
                    "router_widen": telem.phase.router_widen_secs,
                    "router_gemm": telem.phase.router_gemm_secs,
                    "routing": telem.phase.routing_secs,
                    "shared_expert": telem.phase.shared_expert_secs,
                    "routed_expert_wave": telem.phase.routed_expert_secs,
                    "combine": telem.phase.combine_secs,
                    "residual": telem.phase.residual_secs,
                    "retention_flush": telem.phase.retention_flush_secs,
                },
                "peak_rss_after_routed_bytes": telem.phase.peak_rss_after_routed_bytes,
            },
            "bounded_storage": {
                "strategy": "per_expert_first_n_router_input_hiddens_plus_full_route_membership",
                "why": "Per-layer stratified subsample shared across 512 experts left well under 1 row/organ on Q80 (top-10). First-N per expert after routing is known guarantees up to N retained rows per (layer, expert) while keeping full route membership. Retained hidden rows are flushed per layer and freed before the next layer loads, so the streamed RSS cap bounds one layer rather than ×48.",
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "retention_policy": "first_N_tokens_that_route_to_expert_in_global_token_order",
                "deterministic": true,
                "experts": QWEN80_EXPERTS,
                "worst_case_unique_rows_per_layer": worst_case_rows_per_layer,
                "hidden_rows_retained_total": hidden_rows_retained_total,
                "hidden_rows_retained_per_layer": hidden_rows_per_layer,
                "layers": QWEN80_LAYERS,
                "hidden_width": QWEN80_HIDDEN,
                "total_tokens_executed": tokens_executed,
                "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
                "retained_hidden_budget_bytes": retained_hidden_budget_bytes,
                "retained_hidden_bytes_written": hidden_bytes_written,
                "full_route_membership_for_every_token_every_layer": true,
                "n_fit_distribution": n_fit_dist.clone(),
                "rejected_alternatives": {
                    "full_raw_all_tokens": "unbounded; not acceptable",
                    "per_layer_stratified_subsample": "shared budget across 512 experts; larger corpus spreads routing and starves each expert",
                    "random_reservoir": "not deterministic unless seeded and documented; first-N is byte-identical across runs",
                },
            },
            "capture_summary": {
                "probe_count": probe_rows.len(),
                "total_tokens": tokens_executed,
                "layers_executed": layers_executed,
                "broad_activation_diversity": true,
                "all_layer_activation_capture": true,
                "hidden_rows_retained_total": hidden_rows_retained_total,
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "n_fit_distribution": n_fit_dist.clone(),
            },
            "probes": probe_rows,
            "peak_rss_bytes": peak,
            "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
            "wall_clock_secs": wall.as_secs_f64(),
            "claim_boundary": claim_boundary,
        });
        let result_path = output_dir.join("capture-result.json");
        write_json_new(&result_path, &result).unwrap_or_else(|e| fail(e));
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "status": result.get("status"),
                "schema": RESULT_SCHEMA,
                "output_dir": output_dir,
                "capture_summary": result.get("capture_summary"),
                "stream_telemetry": result.get("stream_telemetry"),
                "bounded_storage": {
                    "strategy": "per_expert_first_n_router_input_hiddens_plus_full_route_membership",
                    "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                    "hidden_rows_retained_total": hidden_rows_retained_total,
                    "retained_hidden_bytes_written": hidden_bytes_written,
                    "n_fit_distribution": n_fit_dist,
                },
                "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
                "wall_clock_secs": wall.as_secs_f64(),
            }))
            .expect("summary serializes")
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::main();
}
