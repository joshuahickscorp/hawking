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
//!     write retained hidden rows + full route membership
//!     free the layer weights
//! ```
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
        capture_all_layers, embed_probes, greedy_decode_user_prompt, is_coherent_paris_continuation,
        peak_rss_bytes, SourceBf16Index, STREAMED_PEAK_RSS_HARD_CAP_BYTES, QWEN80_HIDDEN,
        QWEN80_LAYERS, QWEN80_TOP_K,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::{BTreeSet, HashSet};
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
        "q80-source-bf16-layer-major-route-hidden-capture-stratified-subsample-v1";
    const TRACE_STATUS: &str = "NEW_DIAGNOSTIC_NOT_HISTORICAL";
    const DEFAULT_MAX_HIDDEN_TOKENS_PER_LAYER: usize = 1024;
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
        max_hidden_tokens_per_layer: usize,
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
         \x20   [--max-hidden-tokens-per-layer N] [--max-probes N]"
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
        let mut max_hidden_tokens_per_layer = DEFAULT_MAX_HIDDEN_TOKENS_PER_LAYER;
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
                    let value = args.next().ok_or_else(|| {
                        format!(
                            "missing value for --max-hidden-tokens-per-layer; {}",
                            usage()
                        )
                    })?;
                    max_hidden_tokens_per_layer =
                        parse_usize(&value, "--max-hidden-tokens-per-layer")?;
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
        if max_hidden_tokens_per_layer == 0 {
            return Err("--max-hidden-tokens-per-layer must be positive".into());
        }
        if max_new_tokens == 0 {
            return Err("--max-new-tokens must be positive".into());
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
            max_hidden_tokens_per_layer,
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

    fn select_hidden_positions(
        probes: &[(String, Vec<u32>)],
        max_hidden_tokens: usize,
    ) -> BTreeSet<(usize, usize)> {
        let total: usize = probes.iter().map(|(_, t)| t.len()).sum();
        if total == 0 || max_hidden_tokens == 0 {
            return BTreeSet::new();
        }
        if total <= max_hidden_tokens {
            let mut all = BTreeSet::new();
            for (pi, (_, tokens)) in probes.iter().enumerate() {
                for pos in 0..tokens.len() {
                    all.insert((pi, pos));
                }
            }
            return all;
        }
        let mut quotas: Vec<(usize, usize, f64)> = probes
            .iter()
            .enumerate()
            .map(|(pi, (_, tokens))| {
                let exact = max_hidden_tokens as f64 * (tokens.len() as f64) / (total as f64);
                let base = exact.floor() as usize;
                let frac = exact - base as f64;
                (pi, base.min(tokens.len()), frac)
            })
            .collect();
        let mut allocated: usize = quotas.iter().map(|(_, b, _)| *b).sum();
        let mut order: Vec<usize> = (0..quotas.len()).collect();
        order.sort_by(|a, b| {
            quotas[*b]
                .2
                .partial_cmp(&quotas[*a].2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        let mut oi = 0usize;
        while allocated < max_hidden_tokens && !order.is_empty() {
            let idx = order[oi % order.len()];
            let pi = quotas[idx].0;
            let len = probes[pi].1.len();
            if quotas[idx].1 < len {
                quotas[idx].1 += 1;
                allocated += 1;
            }
            oi += 1;
            if oi > max_hidden_tokens.saturating_mul(4) {
                break;
            }
        }
        let mut selected = BTreeSet::new();
        for (pi, quota, _) in quotas {
            let len = probes[pi].1.len();
            if quota == 0 || len == 0 {
                continue;
            }
            if quota >= len {
                for pos in 0..len {
                    selected.insert((pi, pos));
                }
                continue;
            }
            for i in 0..quota {
                let pos = (i * len) / quota;
                selected.insert((pi, pos.min(len - 1)));
            }
        }
        selected
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
            .create_new(true)
            .write(true)
            .open(path)
            .map_err(|e| format!("cannot create hidden {}: {e}", path.display()))?;
        let mut digest = Sha256::new();
        for value in values {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes)
                .map_err(|e| format!("cannot write hidden {}: {e}", path.display()))?;
            digest.update(bytes);
        }
        file.flush()
            .map_err(|e| format!("cannot flush hidden {}: {e}", path.display()))?;
        Ok((format!("{:x}", digest.finalize()), values.len() * 4))
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
        let tokenizer = arguments.tokenizer_path.clone().unwrap_or_else(|| {
            arguments.source_model_dir.join("tokenizer.json")
        });
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
        let result = greedy_decode_user_prompt(
            &index,
            &tokenizer,
            PARIS_PROMPT,
            arguments.max_new_tokens,
        )
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
        let hidden_positions =
            select_hidden_positions(&probes, arguments.max_hidden_tokens_per_layer);
        let hidden_tokens_retained = hidden_positions.len();
        let naive_hidden_bytes = total_tokens
            .saturating_mul(QWEN80_LAYERS)
            .saturating_mul(QWEN80_HIDDEN)
            .saturating_mul(4);
        let retained_hidden_budget_bytes = hidden_tokens_retained
            .saturating_mul(QWEN80_LAYERS)
            .saturating_mul(QWEN80_HIDDEN)
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
        if index.tensor_count() < 70_000 {
            fail(format!(
                "source index has only {} tensors; expected full Q80 catalog (~74391)",
                index.tensor_count()
            ));
        }

        eprintln!(
            "capture: {} probes, {} tokens, retain {} hidden positions/layer; source tensors={}",
            probes.len(),
            total_tokens,
            hidden_tokens_retained,
            index.tensor_count()
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
        let (captures, telem) =
            capture_all_layers(&index, &probes, &mut hiddens, Some(&mut on_layer))
                .unwrap_or_else(|e| fail(e.to_string()));
        drop(hiddens);

        let mut probe_rows = Vec::with_capacity(probes.len());
        let mut tokens_executed = 0usize;
        let mut hidden_bytes_written = 0usize;
        let mut route_membership_total = 0usize;

        for (pi, (probe_id, token_ids)) in probes.iter().enumerate() {
            let mut steps = Vec::with_capacity(token_ids.len());
            for (pos, &token_id) in token_ids.iter().enumerate() {
                let store_hidden = hidden_positions.contains(&(pi, pos));
                let layer_caps = &captures[pi][pos];
                if layer_caps.len() != QWEN80_LAYERS {
                    fail(format!(
                        "{probe_id}@{pos}: captured {} layers, expected {QWEN80_LAYERS}",
                        layer_caps.len()
                    ));
                }
                let mut layer_rows = Vec::with_capacity(QWEN80_LAYERS);
                for layer_cap in layer_caps {
                    if layer_cap.selected_expert_ids.len() != QWEN80_TOP_K
                        || layer_cap.normalized_route_weights.len() != QWEN80_TOP_K
                    {
                        fail(format!(
                            "{probe_id}@{pos} L{}: route membership is not top-{QWEN80_TOP_K}",
                            layer_cap.layer
                        ));
                    }
                    route_membership_total = route_membership_total
                        .saturating_add(layer_cap.selected_expert_ids.len());
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
                    "hidden_retained_for_this_token": store_hidden,
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
            .saturating_mul(QWEN80_LAYERS)
            .saturating_mul(QWEN80_TOP_K);
        if route_membership_total != expected_route_slots {
            fail(format!(
                "route membership total {route_membership_total} != expected {expected_route_slots}"
            ));
        }
        let expected_hidden_bytes = hidden_tokens_retained
            .saturating_mul(QWEN80_LAYERS)
            .saturating_mul(QWEN80_HIDDEN)
            .saturating_mul(4);
        if hidden_bytes_written != expected_hidden_bytes {
            fail(format!(
                "hidden bytes written {hidden_bytes_written} != expected {expected_hidden_bytes}"
            ));
        }

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
            },
            "bounded_storage": {
                "strategy": "stratified_token_subsample_raw_hiddens_plus_full_route_membership",
                "why": "SVD fit needs Gram = X'X/n and surplus-over-null needs holdout rows; full raw dump is multi-GB",
                "max_hidden_tokens_per_layer": arguments.max_hidden_tokens_per_layer,
                "hidden_tokens_retained_per_layer": hidden_tokens_retained,
                "layers": QWEN80_LAYERS,
                "hidden_width": QWEN80_HIDDEN,
                "total_tokens_executed": tokens_executed,
                "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
                "retained_hidden_budget_bytes": retained_hidden_budget_bytes,
                "retained_hidden_bytes_written": hidden_bytes_written,
                "full_route_membership_for_every_token_every_layer": true,
            },
            "capture_summary": {
                "probe_count": probe_rows.len(),
                "total_tokens": tokens_executed,
                "layers_executed": QWEN80_LAYERS,
                "broad_activation_diversity": true,
                "all_layer_activation_capture": true,
                "hidden_tokens_retained": hidden_tokens_retained,
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
