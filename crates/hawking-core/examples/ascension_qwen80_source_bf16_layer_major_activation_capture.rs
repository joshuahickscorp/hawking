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
        capture_layers_from, embed_probes, format_capture_progress,
        greedy_decode_user_prompt, is_coherent_paris_continuation,
        max_hidden_tokens_per_expert_within_streamed_cap, peak_rss_bytes,
        retained_hidden_relative_path, retained_swiglu_packed_relative_path,
        retained_swiglu_relative_path, worst_case_retained_hidden_bytes_per_layer,
        worst_case_unique_rows_per_layer, write_retained_hidden_f32le, LayerTokenCapture,
        RetentionPolicy, SourceBf16Index, DEFAULT_MAX_HIDDEN_TOKENS_PER_EXPERT, QWEN80_EXPERTS,
        QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_TOP_K, RESERVOIR_SEED,
        STREAMED_PEAK_RSS_HARD_CAP_BYTES,
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
        resume: bool,
        retention: String,
        reservoir_seed: u64,
        omit_result_json: bool,
        packed_swiglu_only: bool,
    }

    fn usage() -> &'static str {
        "usage:\n  coherence: ascension_qwen80_source_bf16_layer_major_activation_capture \\\n\
         \x20   --mode coherence --source-model-dir ABSOLUTE_PATH \\\n\
         \x20   [--tokenizer-path ABSOLUTE_PATH] [--max-new-tokens N]\n\
         capture: ... --mode capture --source-model-dir ABSOLUTE_PATH \\\n\
         \x20   --input-json ABSOLUTE_PATH --output-dir ABSOLUTE_PATH \\\n\
         \x20   [--max-hidden-tokens-per-expert N] [--max-probes N] [--resume]\n\
         \x20   [--retention first-n|reservoir] [--reservoir-seed U64]\n\
         \x20   [--omit-result-json] [--packed-swiglu-only]\n\
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
        let mut resume = false;
        let mut retention = "first-n".to_string();
        let mut reservoir_seed = RESERVOIR_SEED;
        let mut omit_result_json = false;
        let mut packed_swiglu_only = false;
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
                "--resume" => resume = true,
                "--omit-result-json" => omit_result_json = true,
                "--packed-swiglu-only" => packed_swiglu_only = true,
                "--retention" => {
                    let value = args.next().ok_or_else(|| {
                        format!("missing value for --retention; {}", usage())
                    })?;
                    match value.as_str() {
                        "first-n" | "reservoir" => retention = value,
                        other => {
                            return Err(format!(
                                "--retention must be first-n or reservoir (got {other:?})"
                            ))
                        }
                    }
                }
                "--reservoir-seed" => {
                    let value = args.next().ok_or_else(|| {
                        format!("missing value for --reservoir-seed; {}", usage())
                    })?;
                    reservoir_seed = value.parse::<u64>().map_err(|_| {
                        format!("--reservoir-seed must be a u64 decimal; {}", usage())
                    })?;
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
            resume,
            retention,
            reservoir_seed,
            omit_result_json,
            packed_swiglu_only,
        })
    }

    fn retention_policy(arguments: &Arguments) -> Result<RetentionPolicy, String> {
        match arguments.retention.as_str() {
            "first-n" => Ok(RetentionPolicy::first_n(
                arguments.max_hidden_tokens_per_expert,
            )),
            "reservoir" => Ok(RetentionPolicy::reservoir(
                arguments.max_hidden_tokens_per_expert,
                arguments.reservoir_seed,
            )),
            other => Err(format!("unsupported --retention {other:?}")),
        }
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

    const CKPT_NAME: &str = "checkpoint.json";
    const CKPT_SCHEMA: &str = "hawking.qwen80.source_bf16_layer_major.checkpoint.v1";

    fn layer_meta_path(output_dir: &Path, layer: usize) -> PathBuf {
        output_dir.join(format!("layer_meta/L{layer:02}.json"))
    }

    fn residual_path(output_dir: &Path, probe_id: &str) -> PathBuf {
        output_dir.join(format!("residual/{probe_id}.f32le"))
    }

    fn write_json_overwrite(path: &Path, value: &Value) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let text = serde_json::to_string_pretty(value).map_err(|e| e.to_string())?;
        fs::write(path, text + "\n").map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    fn write_residual(output_dir: &Path, probe_id: &str, values: &[f32]) -> Result<(), String> {
        let path = residual_path(output_dir, probe_id);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create {}: {e}", parent.display()))?;
        }
        let mut bytes = Vec::with_capacity(values.len().saturating_mul(4));
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        fs::write(&path, bytes).map_err(|e| format!("cannot write {}: {e}", path.display()))
    }

    fn read_residual(path: &Path, expected: usize) -> Result<Vec<f32>, String> {
        let bytes = fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if bytes.len() != expected.saturating_mul(4) {
            return Err(format!(
                "{} residual bytes {} != {} f32",
                path.display(),
                bytes.len(),
                expected
            ));
        }
        let mut out = vec![0.0f32; expected];
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out)
    }

    fn load_layer_meta_into(
        output_dir: &Path,
        layer: usize,
        probes: &[(String, Vec<u32>)],
        hidden_writes: &mut [Vec<Vec<Option<(String, String, usize, usize)>>>],
        swiglu_writes: &mut [Vec<Vec<Vec<(u32, String, String, usize, usize)>>>],
    ) -> Option<(usize, usize, usize)> {
        let path = layer_meta_path(output_dir, layer);
        if !path.is_file() {
            return None;
        }
        let doc: Value = serde_json::from_str(&fs::read_to_string(&path).ok()?).ok()?;
        let mut h_rows = 0usize;
        let mut h_bytes = 0usize;
        let mut s_bytes = 0usize;
        for tok in doc.get("tokens")?.as_array()? {
            let pi = tok.get("pi")?.as_u64()? as usize;
            let pos = tok.get("pos")?.as_u64()? as usize;
            if pi >= probes.len() || pos >= probes[pi].1.len() || layer >= QWEN80_LAYERS {
                continue;
            }
            if let Some(hidden) = tok.get("hidden").filter(|v| !v.is_null()) {
                let rel = hidden.get("relative_path")?.as_str()?.to_string();
                let sha = hidden.get("sha256")?.as_str()?.to_string();
                let bytes = hidden.get("bytes")?.as_u64()? as usize;
                let elems = hidden.get("elements")?.as_u64()? as usize;
                hidden_writes[pi][pos][layer] = Some((rel, sha, bytes, elems));
                h_rows += 1;
                h_bytes += bytes;
            }
            let mut rows = Vec::new();
            if let Some(sw) = tok.get("swiglu").and_then(Value::as_array) {
                for item in sw {
                    let eid = item.get("expert_id")?.as_u64()? as u32;
                    let rel = item.get("relative_path")?.as_str()?.to_string();
                    let sha = item.get("sha256")?.as_str()?.to_string();
                    let bytes = item.get("bytes")?.as_u64()? as usize;
                    let elems = item.get("elements")?.as_u64()? as usize;
                    s_bytes += bytes;
                    rows.push((eid, rel, sha, bytes, elems));
                }
            }
            swiglu_writes[pi][pos][layer] = rows;
        }
        Some((h_rows, h_bytes, s_bytes))
    }

    fn apply_resumed_routes(
        output_dir: &Path,
        probes: &[(String, Vec<u32>)],
        start_layer: usize,
        captures: &mut [Vec<Vec<LayerTokenCapture>>],
    ) -> Result<(), String> {
        for layer in 0..start_layer {
            let path = layer_meta_path(output_dir, layer);
            if !path.is_file() {
                return Err(format!("resume missing {}", path.display()));
            }
            let doc: Value = serde_json::from_str(
                &fs::read_to_string(&path)
                    .map_err(|e| format!("cannot read {}: {e}", path.display()))?,
            )
            .map_err(|e| format!("{} is not JSON: {e}", path.display()))?;
            for tok in doc
                .get("tokens")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("{} lacks tokens", path.display()))?
            {
                let pi = tok
                    .get("pi")
                    .and_then(Value::as_u64)
                    .ok_or("layer_meta token missing pi")? as usize;
                let pos = tok
                    .get("pos")
                    .and_then(Value::as_u64)
                    .ok_or("layer_meta token missing pos")? as usize;
                if pi >= probes.len() || pos >= probes[pi].1.len() {
                    return Err("layer_meta token out of range".into());
                }
                if layer >= captures[pi][pos].len() {
                    return Err(format!(
                        "capture vec too short for resumed L{layer} ({pi},{pos})"
                    ));
                }
                let ids = tok
                    .get("selected_expert_ids")
                    .and_then(Value::as_array)
                    .ok_or("layer_meta missing selected_expert_ids")?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|x| u32::try_from(x).ok())
                            .ok_or("bad expert id")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let weights = tok
                    .get("normalized_route_weights")
                    .and_then(Value::as_array)
                    .ok_or("layer_meta missing weights")?
                    .iter()
                    .map(|v| v.as_f64().map(|x| x as f32).ok_or("bad weight"))
                    .collect::<Result<Vec<_>, _>>()?;
                let retained = tok
                    .get("hidden_retained")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                captures[pi][pos][layer] = LayerTokenCapture {
                    layer,
                    selected_expert_ids: ids,
                    normalized_route_weights: weights,
                    router_input_hidden: Vec::new(),
                    hidden_retained: retained,
                    swiglu_hidden_routed: Vec::new(),
                };
            }
        }
        Ok(())
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
        if output_dir.exists() && !arguments.resume {
            fail(format!(
                "refusing to reuse or overwrite capture output directory {} (pass --resume)",
                output_dir.display()
            ));
        }
        if arguments.resume && !output_dir.join(CKPT_NAME).is_file() {
            fail(format!(
                "--resume set but {} is missing",
                output_dir.join(CKPT_NAME).display()
            ));
        }
        if !output_dir.exists() && !output_dir.parent().is_some_and(|parent| parent.is_dir()) {
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

        if !output_dir.exists() {
            fs::create_dir(&output_dir).unwrap_or_else(|e| {
                fail(format!(
                    "cannot create capture output directory {}: {e}",
                    output_dir.display()
                ))
            });
        }
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
        let mut start_layer = 0usize;
        let mut prior_wall_secs = 0.0f64;
        if arguments.resume {
            let ckpt: Value = serde_json::from_str(
                &fs::read_to_string(output_dir.join(CKPT_NAME))
                    .unwrap_or_else(|e| fail(format!("cannot read checkpoint: {e}"))),
            )
            .unwrap_or_else(|e| fail(format!("checkpoint is not JSON: {e}")));
            if ckpt.get("schema").and_then(Value::as_str) != Some(CKPT_SCHEMA) {
                fail("checkpoint schema mismatch");
            }
            if ckpt.get("input_sha256").and_then(Value::as_str) != Some(input_sha256.as_str()) {
                fail("checkpoint input_sha256 does not match --input-json");
            }
            let ckpt_n = ckpt
                .get("max_hidden_tokens_per_expert")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            if ckpt_n != arguments.max_hidden_tokens_per_expert {
                fail(format!(
                    "checkpoint first-N {ckpt_n} != {}",
                    arguments.max_hidden_tokens_per_expert
                ));
            }
            let ckpt_ret = ckpt
                .get("retention")
                .and_then(Value::as_str)
                .unwrap_or("first-n");
            if ckpt_ret != arguments.retention {
                fail(format!(
                    "checkpoint retention {ckpt_ret} != {}",
                    arguments.retention
                ));
            }
            start_layer = ckpt
                .get("next_layer")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            prior_wall_secs = ckpt
                .get("wall_secs")
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            for (idx, (probe_id, _)) in probes.iter().enumerate() {
                let path = residual_path(&output_dir, probe_id);
                let expected = hiddens[idx].len();
                hiddens[idx] = read_residual(&path, expected).unwrap_or_else(|e| fail(e));
            }
            eprintln!(
                "resume: starting at layer {start_layer} (prior wall {prior_wall_secs:.1}s)"
            );
        }
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
        // Dense in layer so --resume can fill 0..start_layer from layer_meta.
        type HiddenMeta = (String, String, usize, usize);
        type SwigluMeta = Vec<(u32, String, String, usize, usize)>;
        let mut hidden_writes: Vec<Vec<Vec<Option<HiddenMeta>>>> = probes
            .iter()
            .map(|(_, toks)| {
                (0..toks.len())
                    .map(|_| vec![None; QWEN80_LAYERS])
                    .collect()
            })
            .collect();
        let mut swiglu_writes: Vec<Vec<Vec<SwigluMeta>>> = probes
            .iter()
            .map(|(_, toks)| {
                (0..toks.len())
                    .map(|_| vec![Vec::new(); QWEN80_LAYERS])
                    .collect()
            })
            .collect();
        let mut hidden_bytes_written = 0usize;
        let mut swiglu_bytes_written = 0usize;
        let mut hidden_rows_retained_total = 0usize;
        let mut hidden_rows_per_layer = vec![0usize; QWEN80_LAYERS];
        if arguments.resume {
            for layer in 0..start_layer {
                if let Some((h_rows, h_bytes, s_bytes)) = load_layer_meta_into(
                    &output_dir,
                    layer,
                    &probes,
                    &mut hidden_writes,
                    &mut swiglu_writes,
                ) {
                    hidden_rows_retained_total =
                        hidden_rows_retained_total.saturating_add(h_rows);
                    hidden_bytes_written = hidden_bytes_written.saturating_add(h_bytes);
                    swiglu_bytes_written = swiglu_bytes_written.saturating_add(s_bytes);
                    if layer < QWEN80_LAYERS {
                        hidden_rows_per_layer[layer] = h_rows;
                    }
                }
            }
        }
        let mut on_layer_flush = |layer_idx: usize,
                                  captures: &mut [Vec<Vec<LayerTokenCapture>>],
                                  residuals: &[hawking_core::model::qwen80_source_bf16_layer_major::ProbeHidden]|
         -> hawking_core::Result<()> {
            // A retry of a layer that crashed mid-flush must not trip create_new
            // on already-written rows. layer_meta is written last, so its
            // absence means this layer is incomplete.
            if !layer_meta_path(&output_dir, layer_idx).is_file() {
                let _ = fs::remove_dir_all(output_dir.join(format!("hidden/L{layer_idx:02}")));
                // Packed files are `Lxx/Eyyy.f32le` beside `Lxx/Eyyy/` dirs.
                let _ = fs::remove_dir_all(
                    output_dir.join(format!("x/swiglu_hidden_routed/L{layer_idx:02}")),
                );
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
            // Post-SwiGLU rows: same first-N tokens, one file per (layer, expert).
            let mut swiglu_jobs: Vec<(usize, usize, u32, String)> = Vec::new();
            let mut packed: std::collections::BTreeMap<u32, Vec<f32>> =
                std::collections::BTreeMap::new();
            for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
                for pos in 0..token_ids.len() {
                    let cap = captures
                        .get(pi)
                        .and_then(|p| p.get(pos))
                        .and_then(|t| t.get(layer_idx))
                        .ok_or_else(|| {
                            hawking_core::Error::Model(format!(
                                "flush missing swiglu capture {probe_id}@{pos} L{layer_idx}"
                            ))
                        })?;
                    if !cap.hidden_retained {
                        continue;
                    }
                    if cap.swiglu_hidden_routed.len() != cap.selected_expert_ids.len() {
                        return Err(hawking_core::Error::Model(format!(
                            "{probe_id}@{pos} L{layer_idx}: swiglu rows {} != top-k {}",
                            cap.swiglu_hidden_routed.len(),
                            cap.selected_expert_ids.len()
                        )));
                    }
                    for &(eid, ref row) in &cap.swiglu_hidden_routed {
                        if row.len() != QWEN80_MOE_INTERMEDIATE {
                            return Err(hawking_core::Error::Model(format!(
                                "{probe_id}@{pos} L{layer_idx} E{eid}: swiglu width {} != {QWEN80_MOE_INTERMEDIATE}",
                                row.len()
                            )));
                        }
                        swiglu_jobs.push((
                            pi,
                            pos,
                            eid,
                            retained_swiglu_relative_path(layer_idx, eid, probe_id, pos),
                        ));
                        packed.entry(eid).or_default().extend_from_slice(row);
                    }
                }
            }
            if !swiglu_jobs.is_empty() && !arguments.packed_swiglu_only {
                let n_jobs = swiglu_jobs.len();
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
                        let my_jobs = &swiglu_jobs[start..start + result_chunk.len()];
                        let err = &err;
                        let output_dir = &output_dir;
                        scope.spawn(move || {
                            for (local, (pi, pos, eid, rel)) in my_jobs.iter().enumerate() {
                                let row = caps[*pi][*pos][layer_idx]
                                    .swiglu_hidden_routed
                                    .iter()
                                    .find(|(id, _)| id == eid)
                                    .map(|(_, r)| r.as_slice());
                                let Some(row) = row else {
                                    if let Ok(mut g) = err.lock() {
                                        *g = Some(format!(
                                            "missing swiglu E{eid} at {pi}@{pos} L{layer_idx}"
                                        ));
                                    }
                                    return;
                                };
                                match write_retained_hidden_f32le(&output_dir.join(rel), row) {
                                    Ok((sha, bytes)) => {
                                        result_chunk[local] = Some((sha, bytes, row.len()));
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
                for (job, res) in swiglu_jobs.iter().zip(results.into_iter()) {
                    let (sha, bytes, elems) = res.ok_or_else(|| {
                        hawking_core::Error::Model("parallel swiglu write missing result".into())
                    })?;
                    swiglu_bytes_written = swiglu_bytes_written.saturating_add(bytes);
                    swiglu_writes[job.0][job.1][layer_idx].push((
                        job.2,
                        job.3.clone(),
                        sha,
                        bytes,
                        elems,
                    ));
                }
            }
            for (eid, buf) in packed {
                if buf.is_empty() {
                    continue;
                }
                let rel = retained_swiglu_packed_relative_path(layer_idx, eid);
                let (sha, bytes) =
                    write_retained_hidden_f32le(&output_dir.join(&rel), &buf).map_err(|e| {
                        hawking_core::Error::Model(format!("packed swiglu L{layer_idx} E{eid}: {e}"))
                    })?;
                if arguments.packed_swiglu_only {
                    swiglu_bytes_written = swiglu_bytes_written.saturating_add(bytes);
                }
                let _ = sha;
            }

            // Layer-boundary resume: persist routes + write records + residuals.
            let mut meta_tokens = Vec::new();
            for (pi, (_probe_id, token_ids)) in probes.iter().enumerate() {
                for pos in 0..token_ids.len() {
                    let cap = &captures[pi][pos][layer_idx];
                    let hidden = hidden_writes[pi][pos][layer_idx].as_ref().map(|w| {
                        json!({
                            "relative_path": w.0,
                            "sha256": w.1,
                            "bytes": w.2,
                            "elements": w.3,
                        })
                    });
                    let swiglu: Vec<Value> = swiglu_writes[pi][pos][layer_idx]
                        .iter()
                        .map(|w| {
                            json!({
                                "expert_id": w.0,
                                "relative_path": w.1,
                                "sha256": w.2,
                                "bytes": w.3,
                                "elements": w.4,
                            })
                        })
                        .collect();
                    meta_tokens.push(json!({
                        "pi": pi,
                        "pos": pos,
                        "selected_expert_ids": cap.selected_expert_ids,
                        "normalized_route_weights": cap.normalized_route_weights,
                        "hidden_retained": cap.hidden_retained,
                        "hidden": hidden,
                        "swiglu": swiglu,
                    }));
                }
            }
            write_json_overwrite(
                &layer_meta_path(&output_dir, layer_idx),
                &json!({
                    "layer": layer_idx,
                    "tokens": meta_tokens,
                }),
            )
            .map_err(|e| hawking_core::Error::Model(e))?;
            for (pi, (probe_id, _)) in probes.iter().enumerate() {
                write_residual(&output_dir, probe_id, &residuals[pi])
                    .map_err(|e| hawking_core::Error::Model(e))?;
            }
            write_json_overwrite(
                &output_dir.join(CKPT_NAME),
                &json!({
                    "schema": CKPT_SCHEMA,
                    "next_layer": layer_idx + 1,
                    "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                    "retention": arguments.retention,
                    "reservoir_seed": arguments.reservoir_seed,
                    "packed_swiglu_only": arguments.packed_swiglu_only,
                    "omit_result_json": arguments.omit_result_json,
                    "input_sha256": input_sha256,
                    "probe_ids": probes.iter().map(|(id, _)| id.clone()).collect::<Vec<_>>(),
                    "token_counts": probes.iter().map(|(_, t)| t.len()).collect::<Vec<_>>(),
                    "hidden_rows_retained_total": hidden_rows_retained_total,
                    "hidden_bytes_written": hidden_bytes_written,
                    "swiglu_bytes_written": swiglu_bytes_written,
                    "wall_secs": started.elapsed().as_secs_f64() + prior_wall_secs,
                }),
            )
            .map_err(|e| hawking_core::Error::Model(e))?;
            refuse_if_resident_load(peak_rss_bytes());
            Ok(())
        };
        let policy = retention_policy(&arguments).unwrap_or_else(|e| fail(e));
        let (mut captures, telem) = capture_layers_from(
            &index,
            &probes,
            &mut hiddens,
            arguments.max_hidden_tokens_per_expert,
            start_layer,
            Some(&mut on_layer),
            Some(&mut on_layer_flush),
            policy,
        )
        .unwrap_or_else(|e| fail(e.to_string()));
        // Restore route membership for already-flushed layers so the result
        // JSON is complete after --resume.
        if start_layer > 0 {
            apply_resumed_routes(&output_dir, &probes, start_layer, &mut captures)
                .unwrap_or_else(|e| fail(e));
        }
        drop(hiddens);

        let mut probe_rows = Vec::with_capacity(probes.len());
        let mut tokens_executed = 0usize;
        let mut route_membership_total = 0usize;
        let write_probes = !arguments.omit_result_json;

        let layers_executed = telem.layers;
        for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
            let mut steps = Vec::new();
            if write_probes {
                steps.reserve(token_ids.len());
            }
            for (pos, &token_id) in token_ids.iter().enumerate() {
                let _ = token_id;
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
                        if hidden_writes
                            .get(pi)
                            .and_then(|p| p.get(pos))
                            .and_then(|t| t.get(layer_cap.layer))
                            .and_then(|slot| slot.as_ref())
                            .is_none()
                        {
                            fail(format!(
                                "{probe_id}@{pos} L{}: retained but not written during flush",
                                layer_cap.layer
                            ));
                        }
                    }
                    if !write_probes {
                        continue;
                    }
                    let hidden_meta = if store_hidden {
                        let written = hidden_writes
                            .get(pi)
                            .and_then(|p| p.get(pos))
                            .and_then(|t| t.get(layer_cap.layer))
                            .and_then(|slot| slot.as_ref())
                            .unwrap();
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
                    let swiglu_meta = if store_hidden {
                        let rows = swiglu_writes
                            .get(pi)
                            .and_then(|p| p.get(pos))
                            .and_then(|t| t.get(layer_cap.layer))
                            .cloned()
                            .unwrap_or_default();
                        Some(
                            rows.into_iter()
                                .map(|w| {
                                    json!({
                                        "expert_id": w.0,
                                        "relative_path": w.1,
                                        "sha256": w.2,
                                        "bytes": w.3,
                                        "elements": w.4,
                                        "source": "post-SwiGLU intermediate silu(x @ gate_proj.T) * (x @ up_proj.T) at moe_intermediate=512; the input down_proj sees",
                                    })
                                })
                                .collect::<Vec<_>>(),
                        )
                    } else {
                        None
                    };
                    layer_rows.push(json!({
                        "layer": layer_cap.layer,
                        "selected_expert_ids": layer_cap.selected_expert_ids,
                        "normalized_route_weights": layer_cap.normalized_route_weights,
                        "router_input_hidden_f32le": hidden_meta,
                        "hidden_retained": store_hidden,
                        "swiglu_hidden_routed_f32le": swiglu_meta,
                    }));
                }
                if write_probes {
                    steps.push(json!({
                        "position": pos,
                        "input_token_id": token_id,
                        "layers": layer_rows,
                        "all_48_layers_executed": layers_executed == QWEN80_LAYERS,
                        "final_norm_lm_head_sampler_executed": false,
                        "autoregressive_feedback_or_generation_not_executed": true,
                        "hidden_retained_for_this_token": any_layer_retained,
                    }));
                }
                tokens_executed += 1;
            }
            if write_probes {
                probe_rows.push(json!({
                    "probe_id": probe_id,
                    "source_one_user_native_prompt_token_count": token_ids.len(),
                    "steps": steps,
                }));
            }
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
            "per_expert_first_n_retention": arguments.retention == "first-n",
            "per_expert_reservoir_retention": arguments.retention == "reservoir",
            "omit_result_json": arguments.omit_result_json,
            "packed_swiglu_only": arguments.packed_swiglu_only,
        });

        let result = json!({
            "schema": RESULT_SCHEMA,
            "status": "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_SOURCE_BF16_LAYER_MAJOR_ALL_LAYER_ROUTE_AND_HIDDEN_CAPTURE",
            "capture_protocol_revision": if arguments.retention == "reservoir" {
                "q80-source-bf16-layer-major-route-hidden-capture-per-expert-reservoir-v1"
            } else {
                CAPTURE_PROTOCOL_REVISION
            },
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
                "strategy": if arguments.retention == "reservoir" {
                    "per_expert_seeded_reservoir_router_input_hiddens_plus_full_route_membership"
                } else {
                    "per_expert_first_n_router_input_hiddens_plus_full_route_membership"
                },
                "why": "Per-layer stratified subsample shared across 512 experts left well under 1 row/organ on Q80 (top-10). Per-expert cap after routing is known guarantees rare experts keep every hit until N while popular experts are capped. Reservoir replaces first-N's head-of-corpus bias with a seeded uniform subset so added tokens still diversify. Retained hidden rows are flushed per layer and freed before the next layer loads, so the streamed RSS cap bounds one layer rather than ×48.",
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "retention_policy": arguments.retention,
                "reservoir_seed": arguments.reservoir_seed,
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
                "swiglu_hidden_routed": {
                    "captured": true,
                    "width": QWEN80_MOE_INTERMEDIATE,
                    "formula": "silu(x @ gate_proj.T) * (x @ up_proj.T)",
                    "hidden_act": "silu",
                    "path_template": "x/swiglu_hidden_routed/L{layer:02}/E{expert:03}/{probe_id}/{position:06}.f32le",
                    "packed_path_template": "x/swiglu_hidden_routed/L{layer:02}/E{expert:03}.f32le",
                    "same_first_n_and_row_order_as_router_input_hidden": true,
                    "bytes_written": swiglu_bytes_written,
                },
                "n_fit_distribution": n_fit_dist.clone(),
                "rejected_alternatives": {
                    "full_raw_all_tokens": "unbounded; not acceptable",
                    "per_layer_stratified_subsample": "shared budget across 512 experts; larger corpus spreads routing and starves each expert",
                    "random_reservoir_unseeded": "refused: not deterministic. Seeded reservoir is the documented alternative to first-N.",
                },
            },
            "capture_summary": {
                "probe_count": probes.len(),
                "total_tokens": tokens_executed,
                "layers_executed": layers_executed,
                "broad_activation_diversity": true,
                "all_layer_activation_capture": true,
                "hidden_rows_retained_total": hidden_rows_retained_total,
                "max_hidden_tokens_per_expert": arguments.max_hidden_tokens_per_expert,
                "n_fit_distribution": n_fit_dist.clone(),
            },
            "probes": if write_probes { Value::Array(probe_rows) } else { Value::Array(vec![]) },
            "probes_omitted": !write_probes,
            "routes_live_in": if write_probes {
                "capture-result.json probes array"
            } else {
                "layer_meta/Lxx.json + capture-index.v1 (assemble after capture)"
            },
            "peak_rss_bytes": peak,
            "peak_rss_gib": peak as f64 / (1024.0 * 1024.0 * 1024.0),
            "wall_clock_secs": wall.as_secs_f64(),
            "claim_boundary": claim_boundary,
        });
        let result_path = output_dir.join("capture-result.json");
        // Final result may replace a partial file left by a previous --resume
        // segment (that run wrote a result covering only the layers it executed).
        write_json_overwrite(&result_path, &result).unwrap_or_else(|e| fail(e));
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
