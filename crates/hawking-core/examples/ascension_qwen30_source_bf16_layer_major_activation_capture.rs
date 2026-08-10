//! Layer-major BF16 SOURCE activation route+hidden capture for Q30.
//!
//! Fixes the broken activation calibration path that was feeding the repack
//! activations from the low-fidelity baseline binary instead of the BF16 source.
//!
//! Resource contract (does NOT touch the co-resident memory gate):
//!
//! ```text
//! for layer in 0..48:
//!     range-read layer weights from safetensors shards
//!     push ALL probe tokens through that layer (probe-local causal attention)
//!     retain router-input hiddens under per-expert first-N (after top-k known)
//!     write retained hidden rows + full route membership
//!     free the layer weights
//! ```
//!
//! Working set is ~one layer (~1.2 GiB BF16) + residual streams — single-digit
//! GiB — not the 56.9 GiB full-source residency the co-resident gate correctly
//! refuses. Weight bytes are read once total, not once per token.
//!
//! Hidden retention is **per-expert first-N** (default N=64), not a global
//! stratified position set. A shared per-layer budget starves experts when a
//! larger corpus spreads routing; first-N after routing is known guarantees up
//! to N retained rows per (layer, expert) and is deterministic.
//!
//! Output layout matches
//! `ascension_qwen30_broad_activation_all_layer_route_capture` so
//! `lab/operators/ascension_qwen30_activation_weighted_svd_repack.py` can
//! consume the run directory with no repack changes.
//!
//! Modes:
//! * `capture` — full (or max-probes-bounded) activation capture
//! * `coherence` — greedy top-1 on the source chat template
//!   ("What is the capital of France?" → must start with Paris)

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen30 source BF16 layer-major capture requires macOS");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_source_bf16_layer_major::{
        capture_all_layers, embed_probes, greedy_decode_user_prompt, is_coherent_paris_continuation,
        peak_rss_bytes, CaptureTiming, LayerTokenCapture, SourceBf16Index,
        DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, STREAMED_PEAK_RSS_HARD_CAP_BYTES, QWEN30_EXPERTS,
        QWEN30_HIDDEN, QWEN30_LAYERS,
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

    const BROAD_INPUT_SCHEMA: &str =
        "hawking.ascension.qwen30_broad_activation_layer0_route_capture_input.v1";
    const ALL_LAYER_INPUT_SCHEMA: &str =
        "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_input.v1";
    /// Same result schema as the complete-binary all-layer capture so the
    /// repack's structural path (`capture_is_all_layer` + hidden layout) binds.
    const RESULT_SCHEMA: &str =
        "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_result.v1";
    const CAPTURE_PROTOCOL_REVISION: &str =
        "source-bf16-layer-major-route-hidden-capture-per-expert-first-n-v1";
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
        "usage:\n  capture: ascension_qwen30_source_bf16_layer_major_activation_capture \\\n\
         \x20   --mode capture --source-model-dir ABSOLUTE_PATH \\\n\
         \x20   --input-json ABSOLUTE_PATH --output-dir ABSOLUTE_PATH \\\n\
         \x20   [--max-hidden-tokens-per-expert N] [--max-probes N]\n\
         coherence: ... --mode coherence --source-model-dir ABSOLUTE_PATH \\\n\
         \x20   [--tokenizer-path ABSOLUTE_PATH] [--max-new-tokens N]\n\
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
        let mut mode = Mode::Capture;
        let mut source_model_dir = None;
        let mut input_json = None;
        let mut output_dir = None;
        let mut max_hidden_tokens_per_expert = DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT;
        let mut max_probes = None;
        let mut max_new_tokens = 12usize;
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
                    // Retired: a per-layer cap shared across 128 experts was the
                    // root cause of under-fitted organs on large corpora. Refuse
                    // rather than silently reinterpreting the number as per-expert.
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
        let budget_bytes = max_hidden_tokens_per_expert
            .saturating_mul(QWEN30_EXPERTS)
            .saturating_mul(QWEN30_LAYERS)
            .saturating_mul(QWEN30_HIDDEN)
            .saturating_mul(4);
        // 4 GiB worst-case retained capture budget: default N=64 → ~3.0 GiB.
        // Peak RSS hard cap remains STREAMED_PEAK_RSS_HARD_CAP_BYTES (12 GiB).
        const MAX_HIDDEN_BUDGET_BYTES: usize = 4 * 1024 * 1024 * 1024;
        if budget_bytes > MAX_HIDDEN_BUDGET_BYTES {
            return Err(format!(
                "--max-hidden-tokens-per-expert {max_hidden_tokens_per_expert} implies \
                 worst-case ~{budget_bytes} retained hidden bytes \
                 (> {MAX_HIDDEN_BUDGET_BYTES} hard cap); lower N"
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
        if schema != BROAD_INPUT_SCHEMA && schema != ALL_LAYER_INPUT_SCHEMA {
            return Err(format!(
                "capture input schema is not a known broad/all-layer route-capture input schema (got {schema})"
            ));
        }
        if document.get("status").and_then(Value::as_str) != Some(TRACE_STATUS) {
            return Err("capture input is not marked NEW_DIAGNOSTIC_NOT_HISTORICAL".into());
        }
        if document
            .pointer("/claim_boundary/model_execution_started")
            .and_then(Value::as_bool)
            != Some(false)
        {
            return Err(
                "capture input does not prove preparation stopped before model execution".into(),
            );
        }
        if document
            .pointer("/claim_boundary/diagnostic_activation_pricing_only")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(
                "broad/all-layer capture input must set claim_boundary.diagnostic_activation_pricing_only=true"
                    .into(),
            );
        }
        if document
            .pointer("/claim_boundary/does_not_claim_coherence_hcli_tps_or_capability")
            .and_then(Value::as_bool)
            != Some(true)
        {
            return Err(
                "broad/all-layer capture input must refuse coherence/HCLI/TPS/capability claims"
                    .into(),
            );
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
                    .ok_or_else(|| format!("{probe_id} lacks source one-user native token IDs"))?,
                &probe_id,
            )?;
            if token_ids.len() < 4 {
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

    /// Summarise a vector of non-negative counts into the acceptance distribution.
    fn dist_summary(counts: &[usize]) -> Value {
        let mut sorted = counts.to_vec();
        sorted.sort_unstable();
        let n = sorted.len();
        let frac = |pred: &dyn Fn(usize) -> bool| -> f64 {
            if n == 0 {
                0.0
            } else {
                sorted.iter().filter(|&&c| pred(c)).count() as f64 / n as f64
            }
        };
        json!({
            "n": n,
            "min": sorted.first().copied().unwrap_or(0),
            "p10": percentile_sorted(&sorted, 10.0),
            "p25": percentile_sorted(&sorted, 25.0),
            "p50": percentile_sorted(&sorted, 50.0),
            "p75": percentile_sorted(&sorted, 75.0),
            "p90": percentile_sorted(&sorted, 90.0),
            "max": sorted.last().copied().unwrap_or(0),
            "mean": if n == 0 {
                0.0
            } else {
                counts.iter().sum::<usize>() as f64 / n as f64
            },
            "frac_lt_8": frac(&|c| c < 8),
            "frac_lt_16": frac(&|c| c < 16),
            "frac_lt_32": frac(&|c| c < 32),
            "frac_ge_64": frac(&|c| c >= 64),
            "count_lt_8": sorted.iter().filter(|&&c| c < 8).count(),
            "count_lt_16": sorted.iter().filter(|&&c| c < 16).count(),
            "count_lt_32": sorted.iter().filter(|&&c| c < 32).count(),
            "count_ge_64": sorted.iter().filter(|&&c| c >= 64).count(),
            "count_zero": sorted.iter().filter(|&&c| c == 0).count(),
        })
    }

    /// Per-(layer, expert) retained-hidden hit counts — the quantity organs fit on.
    ///
    /// Matches the repack's collect path: a retained token contributes one row to
    /// every expert in its top-k for that layer. Also reports the same distribution
    /// broken out by layer and by expert (marginals). Components (gate/up/down)
    /// share the router-input row, so component n_fit equals expert n_fit.
    fn n_fit_distribution(captures: &[Vec<Vec<LayerTokenCapture>>]) -> Value {
        let mut by_layer_expert: Vec<usize> =
            Vec::with_capacity(QWEN30_LAYERS * QWEN30_EXPERTS);
        let mut by_layer: Vec<usize> = Vec::with_capacity(QWEN30_LAYERS);
        let mut by_expert: Vec<usize> = vec![0usize; QWEN30_EXPERTS];
        for layer in 0..QWEN30_LAYERS {
            let mut per_expert = vec![0usize; QWEN30_EXPERTS];
            for probe_caps in captures {
                for token_caps in probe_caps {
                    let layer_cap = &token_caps[layer];
                    if layer_cap.router_input_hidden.is_empty() {
                        continue;
                    }
                    for &expert in &layer_cap.selected_expert_ids {
                        let e = expert as usize;
                        if e < QWEN30_EXPERTS {
                            per_expert[e] += 1;
                        }
                    }
                }
            }
            let layer_total: usize = per_expert.iter().sum();
            by_layer.push(layer_total);
            for e in 0..QWEN30_EXPERTS {
                by_expert[e] += per_expert[e];
            }
            by_layer_expert.extend(per_expert);
        }
        json!({
            "unit": "retained_hidden_rows_per_layer_expert",
            "overall": dist_summary(&by_layer_expert),
            "by_layer_totals": dist_summary(&by_layer),
            "by_expert_totals_across_layers": dist_summary(&by_expert),
            "by_component": {
                "note": "gate/up/down organs share the same router-input retained rows; n_fit is identical across components",
                "gate": "same_as_overall",
                "up": "same_as_overall",
                "down": "same_as_overall",
            },
            "comparison_note": "Acceptance: with per-expert first-N=64, frac_lt_16 must be far below the starved per-layer-subsample bands (74%/30%).",
        })
    }

    fn timing_to_json(t: &CaptureTiming) -> Value {
        let total = t.total_secs.max(1e-9);
        let row = |secs: f64| {
            json!({
                "seconds": secs,
                "pct": 100.0 * secs / total,
            })
        };
        json!({
            "load_io": row(t.load_secs),
            "bf16_widen_dense": row(t.bf16_widen_dense_secs),
            "attention": row(t.attention_secs),
            "router": row(t.router_secs),
            "expert_gemm": row(t.expert_gemm_secs),
            "retention_write": row(t.retention_write_secs),
            "total_capture_secs": t.total_secs,
            "unaccounted_secs": (t.total_secs
                - t.load_secs
                - t.bf16_widen_dense_secs
                - t.attention_secs
                - t.router_secs
                - t.expert_gemm_secs
                - t.retention_write_secs)
            .max(0.0),
        })
    }

    fn write_hidden(path: &Path, values: &[f32]) -> Result<(String, usize), String> {
        let parent = path
            .parent()
            .ok_or_else(|| format!("hidden capture path has no parent: {}", path.display()))?;
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "cannot create hidden capture directory {}: {e}",
                parent.display()
            )
        })?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("cannot create hidden capture {}: {e}", path.display()))?;
        // One contiguous write of native LE f32 bytes. Avoid per-element write_all
        // and per-file fsync — those dominated wall clock once retention grew to
        // per-expert N=64 (tens of thousands of files). Integrity is still sealed
        // via the sha256 of the written bytes recorded in capture-result.json;
        // a single fsync of the result file is enough for the receipt.
        let nbytes = values.len() * std::mem::size_of::<f32>();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(values.as_ptr() as *const u8, nbytes)
        };
        let mut digest = Sha256::new();
        digest.update(bytes);
        file.write_all(bytes)
            .map_err(|e| format!("cannot write hidden capture {}: {e}", path.display()))?;
        Ok((format!("{:x}", digest.finalize()), nbytes))
    }

    fn write_json_new(path: &Path, value: &Value) -> Result<(), String> {
        // Compact JSON: pretty-print of the full route census is multi-GB and
        // dominated wall time on the sealed corpus. Compact is still valid JSON
        // for the repack consumer.
        let text = serde_json::to_string(value)
            .map_err(|e| format!("cannot serialize capture result: {e}"))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|e| format!("cannot create result {}: {e}", path.display()))?;
        file.write_all(text.as_bytes())
            .map_err(|e| format!("cannot write result {}: {e}", path.display()))?;
        file.write_all(b"\n")
            .map_err(|e| format!("cannot finish result {}: {e}", path.display()))?;
        file.flush()
            .map_err(|e| format!("cannot flush result {}: {e}", path.display()))?;
        // One fsync for the receipt, not per hidden file.
        file.sync_all()
            .map_err(|e| format!("cannot sync result {}: {e}", path.display()))?;
        Ok(())
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        eprintln!(
            "qwen30 source BF16 layer-major capture refused: {}",
            detail.as_ref()
        );
        process::exit(2);
    }

    fn refuse_if_resident_load(peak: u64) {
        if peak >= STREAMED_PEAK_RSS_HARD_CAP_BYTES {
            fail(format!(
                "peak RSS {peak} bytes approaches or exceeds the streamed hard cap \
                 ({STREAMED_PEAK_RSS_HARD_CAP_BYTES}); this is a co-resident / full-source \
                 load contract violation — stop"
            ));
        }
        // Soft tripwire near the sealed full-source size.
        const FULL_SOURCE_BYTES: u64 = 61_066_575_656;
        if peak >= FULL_SOURCE_BYTES * 3 / 4 {
            fail(format!(
                "peak RSS {peak} bytes is within 75% of full source weight bytes \
                 ({FULL_SOURCE_BYTES}); treat as resident-load, refuse"
            ));
        }
    }

    fn run_coherence(arguments: &Arguments) {
        let tokenizer = arguments
            .tokenizer_path
            .clone()
            .unwrap_or_else(|| arguments.source_model_dir.join("tokenizer.json"));
        if !tokenizer.is_file() {
            fail(format!("tokenizer not found at {}", tokenizer.display()));
        }
        let index = SourceBf16Index::open(&arguments.source_model_dir)
            .unwrap_or_else(|e| fail(e.to_string()));
        eprintln!(
            "coherence: source index has {} tensors; prompt={PARIS_PROMPT:?}",
            index.tensor_count()
        );
        let started = Instant::now();
        let result = greedy_decode_user_prompt(
            &index,
            &tokenizer,
            PARIS_PROMPT,
            arguments.max_new_tokens,
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        let wall = started.elapsed();
        let peak = peak_rss_bytes();
        refuse_if_resident_load(peak);
        let ok = is_coherent_paris_continuation(&result.continuation_text);
        let top1 = result.generated_token_ids.first().copied();
        let summary = json!({
            "mode": "coherence",
            "prompt": PARIS_PROMPT,
            "rendered_prompt": result.rendered_prompt,
            "prompt_token_count": result.prompt_token_count,
            "prompt_token_ids": result.prompt_token_ids,
            "generated_token_ids": result.generated_token_ids,
            "top1_token_id": top1,
            "first_token_top10": result.first_token_top10.iter().map(|(id, logit)| json!({
                "token_id": id,
                "logit": logit,
            })).collect::<Vec<_>>(),
            "continuation_text": result.continuation_text,
            "coherent_paris": ok,
            "wall_clock_secs": wall.as_secs_f64(),
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
            "COHERENCE PASSED: continuation {:?} (peak RSS {:.2} GiB, {:.1}s)",
            result.continuation_text,
            peak as f64 / (1024.0 * 1024.0 * 1024.0),
            wall.as_secs_f64()
        );
    }

    fn run_capture(arguments: &Arguments) {
        // MoE is serial-over-experts; let Accelerate use the cores for sgemm.
        // Cap so we leave headroom for concurrent gate lanes and attention workers.
        // SAFETY: set before any BLAS call in this process.
        unsafe {
            if std::env::var_os("VECLIB_MAXIMUM_THREADS").is_none() {
                std::env::set_var("VECLIB_MAXIMUM_THREADS", "8");
            }
            if std::env::var_os("OMP_NUM_THREADS").is_none() {
                std::env::set_var("OMP_NUM_THREADS", "8");
            }
        }
        let input_json = required(arguments.input_json.clone(), "--input-json")
            .unwrap_or_else(|e| fail(e));
        let output_dir = required(arguments.output_dir.clone(), "--output-dir")
            .unwrap_or_else(|e| fail(e));
        if output_dir.exists() {
            fail(format!(
                "refusing to reuse or overwrite capture output directory {}",
                output_dir.display()
            ));
        }
        if !output_dir
            .parent()
            .is_some_and(|parent| parent.is_dir())
        {
            fail("capture output parent must already exist");
        }

        let (input, probes) =
            parse_input(&input_json, arguments.max_probes).unwrap_or_else(|e| fail(e));
        let input_sha256 = sha256_file(&input_json).unwrap_or_else(|e| fail(e));
        let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
        let naive_hidden_bytes = total_tokens
            .saturating_mul(QWEN30_LAYERS)
            .saturating_mul(QWEN30_HIDDEN)
            .saturating_mul(4);
        // Worst-case unique rows/layer under first-N (no multi-route credit).
        let worst_case_rows_per_layer = arguments
            .max_hidden_tokens_per_expert
            .saturating_mul(QWEN30_EXPERTS)
            .min(total_tokens);
        let retained_hidden_budget_bytes = worst_case_rows_per_layer
            .saturating_mul(QWEN30_LAYERS)
            .saturating_mul(QWEN30_HIDDEN)
            .saturating_mul(4);

        fs::create_dir(&output_dir).unwrap_or_else(|e| {
            fail(format!(
                "cannot create capture output directory {}: {e}",
                output_dir.display()
            ))
        });
        let executable_sha256 = current_executable_sha256().unwrap_or_else(|e| fail(e));
        let index = SourceBf16Index::open(&arguments.source_model_dir)
            .unwrap_or_else(|e| fail(e.to_string()));
        if index.tensor_count() < 18_000 {
            fail(format!(
                "source index has only {} tensors; expected full Q30 catalog (~18867)",
                index.tensor_count()
            ));
        }

        eprintln!(
            "capture: {} probes, {} tokens, per-expert first-N={} (worst-case {} rows/layer); source tensors={}",
            probes.len(),
            total_tokens,
            arguments.max_hidden_tokens_per_expert,
            worst_case_rows_per_layer,
            index.tensor_count()
        );

        let started = Instant::now();
        let mut hiddens = embed_probes(&index, &probes).unwrap_or_else(|e| fail(e.to_string()));
        let mut max_layer_resident = 0u64;
        let mut on_layer = |layer: usize, resident: u64| {
            max_layer_resident = max_layer_resident.max(resident);
            if layer == 0 || layer == 23 || layer == 47 || layer % 8 == 0 {
                eprintln!(
                    "  layer {layer:02}/47 resident_layer_bytes={resident} peak_rss={}",
                    peak_rss_bytes()
                );
            }
            refuse_if_resident_load(peak_rss_bytes());
        };
        let capture_run = capture_all_layers(
            &index,
            &probes,
            &mut hiddens,
            arguments.max_hidden_tokens_per_expert,
            Some(&mut on_layer),
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        let captures = capture_run.captures;
        let capture_timing = capture_run.timing;
        let saturation = capture_run.saturation;
        eprintln!(
            "capture timing: total={:.1}s load={:.1}s attn={:.1}s router={:.1}s expert_gemm={:.1}s retention={:.1}s widen={:.1}s",
            capture_timing.total_secs,
            capture_timing.load_secs,
            capture_timing.attention_secs,
            capture_timing.router_secs,
            capture_timing.expert_gemm_secs,
            capture_timing.retention_write_secs,
            capture_timing.bf16_widen_dense_secs,
        );
        eprintln!(
            "saturation: global_max_token={:?} pct_corpus_after={:.1}% (per-layer first token index when every expert hit N={})",
            saturation.global_max_saturation_token,
            saturation.pct_corpus_after_global_saturation,
            arguments.max_hidden_tokens_per_expert,
        );
        // Free residual streams once captures are materialised into probe rows.
        drop(hiddens);

        let mut probe_rows = Vec::with_capacity(probes.len());
        let mut tokens_executed = 0usize;
        let mut hidden_bytes_written = 0usize;
        let mut route_membership_total = 0usize;
        let mut hidden_rows_retained_total = 0usize;
        let mut hidden_rows_per_layer = vec![0usize; QWEN30_LAYERS];

        for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
            let mut steps = Vec::with_capacity(token_ids.len());
            for (pos, &token_id) in token_ids.iter().enumerate() {
                let layer_caps = &captures[pi][pos];
                if layer_caps.len() != QWEN30_LAYERS {
                    fail(format!(
                        "{probe_id}@{pos}: captured {} layers, expected {QWEN30_LAYERS}",
                        layer_caps.len()
                    ));
                }
                let mut layer_rows = Vec::with_capacity(QWEN30_LAYERS);
                let mut any_layer_retained = false;
                for layer_cap in layer_caps {
                    if layer_cap.selected_expert_ids.len() != 8
                        || layer_cap.normalized_route_weights.len() != 8
                    {
                        fail(format!(
                            "{probe_id}@{pos} L{}: route membership is not top-8",
                            layer_cap.layer
                        ));
                    }
                    route_membership_total = route_membership_total
                        .saturating_add(layer_cap.selected_expert_ids.len());
                    // Per-expert first-N is decided inside capture_all_layers after
                    // routing; the writer trusts non-empty router_input_hidden as the
                    // sole retained-hidden authority (must not re-derive a position set).
                    let store_hidden = !layer_cap.router_input_hidden.is_empty();
                    if store_hidden {
                        if layer_cap.router_input_hidden.len() != QWEN30_HIDDEN {
                            fail(format!(
                                "{probe_id}@{pos} L{}: retained hidden width {} != {QWEN30_HIDDEN}",
                                layer_cap.layer,
                                layer_cap.router_input_hidden.len()
                            ));
                        }
                        any_layer_retained = true;
                        hidden_rows_retained_total =
                            hidden_rows_retained_total.saturating_add(1);
                        if layer_cap.layer < QWEN30_LAYERS {
                            hidden_rows_per_layer[layer_cap.layer] =
                                hidden_rows_per_layer[layer_cap.layer].saturating_add(1);
                        }
                    }
                    let hidden_meta = if store_hidden {
                        let hidden_relative = format!(
                            "hidden/L{:02}/{}/{:06}.f32le",
                            layer_cap.layer, probe_id, pos
                        );
                        let hidden_path = output_dir.join(&hidden_relative);
                        let (hidden_sha256, hidden_bytes) =
                            write_hidden(&hidden_path, &layer_cap.router_input_hidden)
                                .unwrap_or_else(|e| fail(e));
                        hidden_bytes_written = hidden_bytes_written.saturating_add(hidden_bytes);
                        Some(json!({
                            "relative_path": hidden_relative,
                            "sha256": hidden_sha256,
                            "bytes": hidden_bytes,
                            "elements": layer_cap.router_input_hidden.len(),
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
                    "all_48_layers_executed": true,
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
        let peak = peak_rss_bytes();
        refuse_if_resident_load(peak);

        // Self-consistency: every token × every layer has route membership;
        // written hidden bytes match counted retained rows.
        let expected_route_slots = tokens_executed.saturating_mul(QWEN30_LAYERS).saturating_mul(8);
        if route_membership_total != expected_route_slots {
            fail(format!(
                "route membership total {route_membership_total} != expected {expected_route_slots}"
            ));
        }
        let expected_hidden_bytes = hidden_rows_retained_total
            .saturating_mul(QWEN30_HIDDEN)
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

        let corpus_scale = 87_439f64;
        let rate_tokens_per_sec = if wall.as_secs_f64() > 0.0 {
            tokens_executed as f64 / wall.as_secs_f64()
        } else {
            0.0
        };
        let implied_full_corpus_secs = if rate_tokens_per_sec > 0.0 {
            corpus_scale / rate_tokens_per_sec
        } else {
            f64::INFINITY
        };

        let claim_boundary = json!({
            "new_diagnostic_not_historical": true,
            "all_48_layers_embedding_attention_postnorm_router_and_expert_wave_executed": true,
            "final_norm_lm_head_sampler_executed_but_not_retained_as_activation_fit_input": true,
            "autoregressive_feedback_generation_hcli_and_tps_not_executed": true,
            "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": true,
            "qwen30_server_watcher_and_hcli_adapter_are_not_used": true,
            "diagnostic_activation_pricing_only": true,
            "broad_activation_diversity_capture": true,
            "bounded_hidden_storage_not_unbounded_raw_dump": true,
            "source_tokenizer_one_user_native_prompts": true,
            "source_bf16_layer_major_streamed_not_co_resident": true,
            "activations_are_from_bf16_source_not_baseline_binary": true,
        });

        let result = json!({
            "schema": RESULT_SCHEMA,
            "status": "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_SOURCE_BF16_LAYER_MAJOR_ALL_LAYER_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED",
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
                "source_tensor_count": index.tensor_count(),
                "runtime_executable_sha256": executable_sha256,
                "architecture": "Qwen3MoeForCausalLM",
                "metal_only": false,
                "raw_bf16_loader_not_opened": false,
                "source_bf16_layer_major_range_reader": true,
                "co_resident_full_source_load": false,
                "complete_binary_runtime_not_used": true,
                "layers": QWEN30_LAYERS,
                "hidden": QWEN30_HIDDEN,
            },
            "streamed_resource_contract": {
                "kind": "layer_major_range_read_not_co_resident_bypass",
                "full_source_weight_bytes_not_resident": true,
                "per_layer_moe_weights_range_read_then_freed": true,
                "max_layer_resident_bytes_observed": max_layer_resident,
                "all_token_hidden_states_bytes_estimate": total_tokens
                    .saturating_mul(QWEN30_HIDDEN)
                    .saturating_mul(4),
                "declared_streamed_working_set_bound_bytes": max_layer_resident
                    .saturating_add(
                        (total_tokens.saturating_mul(QWEN30_HIDDEN).saturating_mul(4)) as u64
                    )
                    .saturating_add(512 * 1024 * 1024),
                "peak_rss_bytes": peak,
                "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
                "peak_rss_hard_cap_bytes": STREAMED_PEAK_RSS_HARD_CAP_BYTES,
                "no_co_resident_load_occurred": true,
                "weight_bytes_read_once_total_not_per_token": true,
                "does_not_weaken_or_flag_flip_co_resident_memory_gate": true,
            },
            "bounded_storage": {
                "strategy": "per_expert_first_n_router_input_hiddens_plus_full_route_membership",
                "why": "Per-layer stratified subsample shared across 128 experts left ~8 rows/expert on large corpora (organs fail null test below ~16). First-N per expert after routing is known guarantees up to N retained rows per (layer, expert) while keeping full route membership and staying inside the streamed RSS cap.",
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "retention_policy": "first_N_tokens_that_route_to_expert_in_global_token_order",
                "deterministic": true,
                "experts": QWEN30_EXPERTS,
                "worst_case_unique_rows_per_layer": worst_case_rows_per_layer,
                "hidden_rows_retained_total": hidden_rows_retained_total,
                "hidden_rows_retained_per_layer": hidden_rows_per_layer,
                "layers": QWEN30_LAYERS,
                "hidden_width": QWEN30_HIDDEN,
                "total_tokens_executed": tokens_executed,
                "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
                "retained_hidden_budget_bytes": retained_hidden_budget_bytes,
                "retained_hidden_bytes_written": hidden_bytes_written,
                "full_route_membership_for_every_token_every_layer": true,
                "n_fit_distribution": n_fit_dist.clone(),
                "rejected_alternatives": {
                    "full_raw_all_tokens": "unbounded; not acceptable",
                    "per_layer_stratified_subsample": "shared budget across experts; larger corpus spreads routing and starves each expert",
                    "random_reservoir": "not deterministic unless seeded and documented; first-N is byte-identical across runs",
                    "per_expert_gram_only": "2048x2048 f32 = 16MB per (layer,expert); multi-GB at scale and cannot score holdout null/surplus"
                },
            },
            "capture_summary": {
                "probe_count": probe_rows.len(),
                "total_tokens": tokens_executed,
                "layers_executed": QWEN30_LAYERS,
                "broad_activation_diversity": true,
                "all_layer_activation_capture": true,
                "hidden_rows_retained_total": hidden_rows_retained_total,
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "n_fit_distribution": n_fit_dist.clone(),
                "source_bf16_layer_major": true,
                "route_membership_total": route_membership_total,
            },
            "timing": {
                "wall_clock_secs": wall.as_secs_f64(),
                "tokens_executed": tokens_executed,
                "tokens_per_sec": rate_tokens_per_sec,
                "implied_full_corpus_87439_tokens_secs": implied_full_corpus_secs,
                "implied_full_corpus_87439_tokens_hours": implied_full_corpus_secs / 3600.0,
                "breakdown": timing_to_json(&capture_timing),
            },
            "saturation": {
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "layer_all_experts_full_at_token": saturation.layer_all_experts_full_at_token,
                "global_max_saturation_token": saturation.global_max_saturation_token,
                "pct_corpus_after_global_saturation": saturation.pct_corpus_after_global_saturation,
                "note": "After every (layer,expert) first-N reservoir is full, later tokens cannot change the retained set. Expert GEMMs still run for residual (required for later-layer route bit-identity).",
            },
            "probes": probe_rows,
            "logit_provenance": {
                "status": "NOT_EXECUTED_DURING_CAPTURE",
                "reason": "layer-major activation capture records router inputs and routes only; final norm/lm_head is reserved for the separate coherence mode"
            },
            "claim_boundary": claim_boundary,
        });
        let result_path = output_dir.join("capture-result.json");
        write_json_new(&result_path, &result).unwrap_or_else(|e| fail(e));
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": result.get("status"),
                "schema": RESULT_SCHEMA,
                "output_dir": output_dir,
                "capture_summary": result.get("capture_summary"),
                "streamed_resource_contract": result.get("streamed_resource_contract"),
                "timing": result.get("timing"),
                "bounded_storage": {
                    "strategy": "per_expert_first_n_router_input_hiddens_plus_full_route_membership",
                    "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                    "hidden_rows_retained_total": hidden_rows_retained_total,
                    "retained_hidden_bytes_written": hidden_bytes_written,
                    "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
                    "n_fit_distribution": n_fit_dist,
                },
                "claim_boundary": {
                    "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": true,
                    "diagnostic_activation_pricing_only": true,
                    "source_bf16_layer_major_streamed_not_co_resident": true,
                    "activations_are_from_bf16_source_not_baseline_binary": true,
                },
            }))
            .expect("summary must serialize")
        );
    }

    pub fn run() {
        let arguments = parse_arguments().unwrap_or_else(|e| fail(e));
        match arguments.mode {
            Mode::Coherence => run_coherence(&arguments),
            Mode::Capture => run_capture(&arguments),
        }
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}
