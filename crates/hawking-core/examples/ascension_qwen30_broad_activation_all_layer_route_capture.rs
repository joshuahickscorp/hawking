//! All-layer broad activation route+hidden capture for Q30.
//!
//! Extends the L0 broad capture (`ascension_qwen30_current_hcli_layer0_route_capture`)
//! so every layer 0..47 records router membership and (bounded) router-input
//! hiddens on the same broad prompt set.
//!
//! ## Bounded storage (hard requirement)
//!
//! Naive raw storage is ~1.5 GB:
//!   48 layers × 3,929 tokens × 2048 f32 ≈ 1.55e9 bytes.
//!
//! **Chosen strategy: stratified token subsample of raw hiddens + full route
//! membership for every token.**
//!
//! - Full route IDs/weights for all tokens at all layers (tiny; needed for hit
//!   counts and cold-expert accounting).
//! - Raw f32 router-input hiddens only for a deterministic stratified subsample
//!   of tokens (default 1024 tokens shared across all layers).
//! - Size budget at default: 48 × 1024 × 2048 × 4 = **384 MB** of hiddens,
//!   independent of prompt length.
//!
//! Why not per-(layer, expert) Gram matrices?  A 2048×2048 f32 Gram is 16 MB;
//! times ~100 hit experts × 48 layers is multi-GB and loses holdout rows that
//! the surplus-over-null metric needs. Subsampled raw rows let the fit build
//! Gram at packing time and still score null/surplus on a true holdout.
//!
//! This is a diagnostic capture only. No coherence, HCLI, TPS, or capability
//! claim. Does not use the production server / watcher / HCLI adapter.

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("qwen30 all-layer broad activation capture requires macOS Metal");
    std::process::exit(2);
}

#[cfg(target_os = "macos")]
mod macos {
    use hawking_core::model::qwen30_complete_runtime::{
        Qwen30AllLayerRouterCaptureStep, Qwen30CompleteNativeRuntime, Qwen30CompleteRuntimeOptions,
        Qwen30GateUpSwiGluKernel, Qwen30PackedMatvecKernel,
    };
    use hawking_core::model::qwen_complete_binary::{
        CompleteBinaryAdmission, QwenCompleteBinaryModel,
    };
    use serde_json::{json, Value};
    use sha2::{Digest, Sha256};
    use std::collections::{HashSet, BTreeSet};
    use std::env;
    use std::fs::{self, File, OpenOptions};
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::process;

    /// Accept the sealed broad L0 prompt set (same 32 probes / ~3929 tokens).
    const BROAD_INPUT_SCHEMA: &str =
        "hawking.ascension.qwen30_broad_activation_layer0_route_capture_input.v1";
    /// Optional dedicated all-layer input schema (same shape; explicit intent).
    const ALL_LAYER_INPUT_SCHEMA: &str =
        "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_input.v1";
    const RESULT_SCHEMA: &str =
        "hawking.ascension.qwen30_broad_activation_all_layer_route_capture_result.v1";
    const CAPTURE_PROTOCOL_REVISION: &str =
        "all-layer-route-hidden-capture-stratified-subsample-v1";
    const TRACE_STATUS: &str = "NEW_DIAGNOSTIC_NOT_HISTORICAL";
    const BROAD_MIN_PROBES: usize = 12;
    const BROAD_MAX_PROBES: usize = 64;
    const DEFAULT_MAX_HIDDEN_TOKENS_PER_LAYER: usize = 1024;
    const QWEN30_LAYERS: usize = 48;
    const QWEN30_HIDDEN: usize = 2048;

    struct Arguments {
        manifest: PathBuf,
        expected_manifest_seal_sha256: String,
        expected_source_audit_seal_sha256: String,
        expected_source_revision: String,
        input_json: PathBuf,
        output_dir: PathBuf,
        max_seq_len: usize,
        max_hidden_tokens_per_layer: usize,
    }

    fn usage() -> &'static str {
        "usage: ascension_qwen30_broad_activation_all_layer_route_capture \\
            --manifest ABSOLUTE_PATH \\
            --expected-manifest-seal-sha256 SHA256 \\
            --expected-source-audit-seal-sha256 SHA256 \\
            --expected-source-revision REVISION \\
            --input-json ABSOLUTE_PATH --output-dir ABSOLUTE_PATH \\
            [--max-seq-len N] [--max-hidden-tokens-per-layer N]"
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
        let mut manifest = None;
        let mut expected_manifest_seal_sha256 = None;
        let mut expected_source_audit_seal_sha256 = None;
        let mut expected_source_revision = None;
        let mut input_json = None;
        let mut output_dir = None;
        let mut max_seq_len = 4096usize;
        let mut max_hidden_tokens_per_layer = DEFAULT_MAX_HIDDEN_TOKENS_PER_LAYER;
        let mut args = env::args().skip(1);
        while let Some(flag) = args.next() {
            let value = args
                .next()
                .ok_or_else(|| format!("missing value for {flag:?}; {}", usage()))?;
            match flag.as_str() {
                "--manifest" => {
                    if manifest.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--manifest was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-manifest-seal-sha256" => {
                    if expected_manifest_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-manifest-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-audit-seal-sha256" => {
                    if expected_source_audit_seal_sha256.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-audit-seal-sha256 was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--expected-source-revision" => {
                    if expected_source_revision.replace(value).is_some() {
                        return Err(format!(
                            "--expected-source-revision was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--input-json" => {
                    if input_json.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--input-json was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--output-dir" => {
                    if output_dir.replace(PathBuf::from(value)).is_some() {
                        return Err(format!(
                            "--output-dir was supplied more than once; {}",
                            usage()
                        ));
                    }
                }
                "--max-seq-len" => max_seq_len = parse_usize(&value, "--max-seq-len")?,
                "--max-hidden-tokens-per-layer" => {
                    max_hidden_tokens_per_layer =
                        parse_usize(&value, "--max-hidden-tokens-per-layer")?;
                }
                _ => return Err(format!("unsupported option {flag:?}; {}", usage())),
            }
        }
        if max_seq_len == 0 {
            return Err("--max-seq-len must be positive".into());
        }
        if max_hidden_tokens_per_layer == 0 {
            return Err("--max-hidden-tokens-per-layer must be positive".into());
        }
        // Hard size gate: refuse an unbounded capture request.
        let budget_bytes = max_hidden_tokens_per_layer
            .saturating_mul(QWEN30_LAYERS)
            .saturating_mul(QWEN30_HIDDEN)
            .saturating_mul(4);
        const MAX_HIDDEN_BUDGET_BYTES: usize = 768 * 1024 * 1024; // 768 MiB hard cap
        if budget_bytes > MAX_HIDDEN_BUDGET_BYTES {
            return Err(format!(
                "--max-hidden-tokens-per-layer {} implies ~{budget_bytes} hidden bytes (> {MAX_HIDDEN_BUDGET_BYTES} hard cap); reduce the bound",
                max_hidden_tokens_per_layer
            ));
        }
        Ok(Arguments {
            manifest: absolute(required(manifest, "--manifest")?, "--manifest")?,
            expected_manifest_seal_sha256: required(
                expected_manifest_seal_sha256,
                "--expected-manifest-seal-sha256",
            )?,
            expected_source_audit_seal_sha256: required(
                expected_source_audit_seal_sha256,
                "--expected-source-audit-seal-sha256",
            )?,
            expected_source_revision: required(
                expected_source_revision,
                "--expected-source-revision",
            )?,
            input_json: absolute(required(input_json, "--input-json")?, "--input-json")?,
            output_dir: absolute(required(output_dir, "--output-dir")?, "--output-dir")?,
            max_seq_len,
            max_hidden_tokens_per_layer,
        })
    }

    fn admission(arguments: &Arguments) -> CompleteBinaryAdmission {
        CompleteBinaryAdmission {
            model: QwenCompleteBinaryModel::Qwen30Coder,
            expected_manifest_seal_sha256: arguments.expected_manifest_seal_sha256.clone(),
            expected_source_audit_seal_sha256: arguments.expected_source_audit_seal_sha256.clone(),
            expected_source_revision: arguments.expected_source_revision.clone(),
        }
    }

    fn sha256_file(path: &Path) -> Result<String, String> {
        let mut file =
            File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        let mut digest = Sha256::new();
        let mut chunk = [0u8; 1024 * 1024];
        loop {
            let read = file
                .read(&mut chunk)
                .map_err(|error| format!("cannot hash {}: {error}", path.display()))?;
            if read == 0 {
                break;
            }
            digest.update(&chunk[..read]);
        }
        Ok(format!("{:x}", digest.finalize()))
    }

    fn current_executable_sha256() -> Result<String, String> {
        let path = env::current_exe()
            .map_err(|error| format!("cannot resolve capture executable: {error}"))?;
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
                    .and_then(|value| u32::try_from(value).ok())
                    .ok_or_else(|| format!("{probe_id} contains an invalid token ID"))
            })
            .collect()
    }

    fn parse_input(path: &Path) -> Result<(Value, Vec<(String, Vec<u32>)>), String> {
        let bytes = fs::read(path)
            .map_err(|error| format!("cannot read capture input {}: {error}", path.display()))?;
        let document: Value = serde_json::from_slice(&bytes)
            .map_err(|error| format!("capture input is not JSON: {error}"))?;
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
        if probes.len() < BROAD_MIN_PROBES {
            return Err(format!(
                "broad capture input must contain at least {BROAD_MIN_PROBES} probes (got {})",
                probes.len()
            ));
        }
        if probes.len() > BROAD_MAX_PROBES {
            return Err(format!(
                "broad capture input must contain at most {BROAD_MAX_PROBES} probes (got {})",
                probes.len()
            ));
        }
        let mut seen = HashSet::new();
        let mut result = Vec::with_capacity(probes.len());
        for probe in probes {
            let probe_id = probe
                .get("probe_id")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
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
            if token_ids.len() < 8 {
                return Err(format!(
                    "{probe_id} is too short for broad activation diversity ({} tokens)",
                    token_ids.len()
                ));
            }
            result.push((probe_id, token_ids));
        }
        Ok((document, result))
    }

    /// Deterministic stratified subsample of (probe_index, position) pairs.
    ///
    /// Allocates slots proportional to each probe's length so short probes are
    /// not erased, then takes an evenly spaced stride within each probe. The
    /// same token set is retained for every layer (residual path is shared).
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
        // Proportional allocation with largest-remainder so sum == max_hidden_tokens.
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
            // Evenly spaced positions across the probe.
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
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "cannot create hidden capture directory {}: {error}",
                parent.display()
            )
        })?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("cannot create hidden capture {}: {error}", path.display()))?;
        let mut digest = Sha256::new();
        for value in values {
            let bytes = value.to_le_bytes();
            file.write_all(&bytes).map_err(|error| {
                format!("cannot write hidden capture {}: {error}", path.display())
            })?;
            digest.update(bytes);
        }
        file.flush()
            .map_err(|error| format!("cannot flush hidden capture {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync hidden capture {}: {error}", path.display()))?;
        Ok((
            format!("{:x}", digest.finalize()),
            values.len() * std::mem::size_of::<f32>(),
        ))
    }

    fn capture_step_row(
        output_dir: &Path,
        probe_id: &str,
        capture: Qwen30AllLayerRouterCaptureStep,
        store_hidden: bool,
        hidden_bytes_written: &mut usize,
    ) -> Result<Value, String> {
        let mut layer_rows = Vec::with_capacity(capture.layers.len());
        for layer_cap in &capture.layers {
            let hidden_meta = if store_hidden {
                let hidden_relative = format!(
                    "hidden/L{:02}/{}/{:06}.f32le",
                    layer_cap.layer, probe_id, capture.position
                );
                let hidden_path = output_dir.join(&hidden_relative);
                let (hidden_sha256, hidden_bytes) =
                    write_hidden(&hidden_path, &layer_cap.router_input_hidden)?;
                *hidden_bytes_written = hidden_bytes_written.saturating_add(hidden_bytes);
                Some(json!({
                    "relative_path": hidden_relative,
                    "sha256": hidden_sha256,
                    "bytes": hidden_bytes,
                    "elements": layer_cap.router_input_hidden.len(),
                    "source": "device-produced post-attention RMSNorm buffer at this layer, copied after router top-k and before expert wave",
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
        Ok(json!({
            "position": capture.position,
            "input_token_id": capture.input_token_id,
            "layers": layer_rows,
            "all_48_layers_executed": true,
            "final_norm_lm_head_sampler_executed": true,
            "autoregressive_feedback_or_generation_not_executed": true,
            "hidden_retained_for_this_token": store_hidden,
        }))
    }

    fn write_json_new(path: &Path, value: &Value) -> Result<(), String> {
        let text = serde_json::to_string_pretty(value)
            .map_err(|error| format!("cannot serialize capture result: {error}"))?;
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("cannot create result {}: {error}", path.display()))?;
        file.write_all(text.as_bytes())
            .map_err(|error| format!("cannot write result {}: {error}", path.display()))?;
        file.write_all(b"\n")
            .map_err(|error| format!("cannot finish result {}: {error}", path.display()))?;
        file.flush()
            .map_err(|error| format!("cannot flush result {}: {error}", path.display()))?;
        file.sync_all()
            .map_err(|error| format!("cannot sync result {}: {error}", path.display()))?;
        Ok(())
    }

    fn fail(detail: impl AsRef<str>) -> ! {
        eprintln!(
            "qwen30 broad all-layer route capture refused: {}",
            detail.as_ref()
        );
        process::exit(2);
    }

    pub fn run() {
        let arguments = parse_arguments().unwrap_or_else(|error| fail(error));
        if arguments.output_dir.exists() {
            fail(format!(
                "refusing to reuse or overwrite route capture output directory {}",
                arguments.output_dir.display()
            ));
        }
        if !arguments
            .output_dir
            .parent()
            .is_some_and(|parent| parent.is_dir())
        {
            fail("route capture output parent must already exist");
        }
        let (input, probes) =
            parse_input(&arguments.input_json).unwrap_or_else(|error| fail(error));
        let input_sha256 = sha256_file(&arguments.input_json).unwrap_or_else(|error| fail(error));
        let total_tokens: usize = probes.iter().map(|(_, t)| t.len()).sum();
        let hidden_positions =
            select_hidden_positions(&probes, arguments.max_hidden_tokens_per_layer);
        let hidden_tokens_retained = hidden_positions.len();
        let naive_hidden_bytes = total_tokens
            .saturating_mul(QWEN30_LAYERS)
            .saturating_mul(QWEN30_HIDDEN)
            .saturating_mul(4);
        let retained_hidden_budget_bytes = hidden_tokens_retained
            .saturating_mul(QWEN30_LAYERS)
            .saturating_mul(QWEN30_HIDDEN)
            .saturating_mul(4);

        fs::create_dir(&arguments.output_dir).unwrap_or_else(|error| {
            fail(format!(
                "cannot create route capture output directory {}: {error}",
                arguments.output_dir.display()
            ))
        });
        let executable_sha256 = current_executable_sha256().unwrap_or_else(|error| fail(error));
        let mut runtime = Qwen30CompleteNativeRuntime::load(
            &arguments.manifest,
            &admission(&arguments),
            Qwen30CompleteRuntimeOptions {
                max_seq_len: arguments.max_seq_len,
                trace_dispatch: false,
                packed_matvec_kernel: Qwen30PackedMatvecKernel::ScalarControl,
                gate_up_swiglu_kernel: Qwen30GateUpSwiGluKernel::ThreeDispatchControl,
            },
        )
        .unwrap_or_else(|error| fail(error.to_string()));

        if runtime.config.layers != QWEN30_LAYERS {
            fail(format!(
                "runtime reports {} layers, expected {QWEN30_LAYERS}",
                runtime.config.layers
            ));
        }
        if runtime.config.hidden != QWEN30_HIDDEN {
            fail(format!(
                "runtime reports hidden {}, expected {QWEN30_HIDDEN}",
                runtime.config.hidden
            ));
        }

        let mut probe_rows = Vec::with_capacity(probes.len());
        let mut tokens_executed = 0usize;
        let mut hidden_bytes_written = 0usize;
        for (probe_index, (probe_id, token_ids)) in probes.iter().enumerate() {
            if token_ids.len() > arguments.max_seq_len {
                fail(format!(
                    "{probe_id} token length {} exceeds capture max sequence {}",
                    token_ids.len(),
                    arguments.max_seq_len
                ));
            }
            runtime.reset();
            let mut steps = Vec::with_capacity(token_ids.len());
            for (position, &token_id) in token_ids.iter().enumerate() {
                let store_hidden = hidden_positions.contains(&(probe_index, position));
                let capture = runtime
                    .capture_all_layers_router_for_token(token_id)
                    .unwrap_or_else(|error| fail(error.to_string()));
                if capture.position != position {
                    fail(format!(
                        "{probe_id} position mismatch: runtime {}, expected {position}",
                        capture.position
                    ));
                }
                if capture.layers.len() != QWEN30_LAYERS {
                    fail(format!(
                        "{probe_id}@{position}: captured {} layers, expected {QWEN30_LAYERS}",
                        capture.layers.len()
                    ));
                }
                steps.push(
                    capture_step_row(
                        &arguments.output_dir,
                        probe_id,
                        capture,
                        store_hidden,
                        &mut hidden_bytes_written,
                    )
                    .unwrap_or_else(|error| fail(error)),
                );
                tokens_executed += 1;
            }
            probe_rows.push(json!({
                "probe_id": probe_id,
                "source_one_user_native_prompt_token_count": steps.len(),
                "steps": steps,
            }));
        }

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
        });

        let result = json!({
            "schema": RESULT_SCHEMA,
            "status": "EARNED_NEW_DIAGNOSTIC_NOT_HISTORICAL_BROAD_ACTIVATION_ALL_LAYER_ROUTE_AND_HIDDEN_CAPTURE_UNQUALIFIED",
            "capture_protocol_revision": CAPTURE_PROTOCOL_REVISION,
            "input": {
                "path": arguments.input_json,
                "sha256": input_sha256,
                "schema": input.get("schema"),
                "status": input.get("status"),
            },
            "runtime_binding": {
                "manifest_path": arguments.manifest,
                "manifest_seal_sha256": runtime.artifact_manifest_seal(),
                "source_revision": runtime.config.source_revision,
                "runtime_executable_sha256": executable_sha256,
                "architecture": "Qwen3MoeForCausalLM",
                "metal_only": true,
                "raw_bf16_loader_not_opened": true,
                "immutable_complete_payload_catalog": {
                    "validated_during_process_admission": true,
                    "verified_payload_count": runtime.verified_payload_count(),
                    "expected_complete_tensor_count": 18867,
                    "complete_verified_payload_cache": runtime.has_complete_verified_payload_cache(),
                },
                "packed_matvec_kernel": runtime.packed_matvec_kernel().receipt_name(),
                "gate_up_swiglu_kernel": runtime.gate_up_swiglu_kernel().receipt_name(),
                "layers": QWEN30_LAYERS,
                "hidden": QWEN30_HIDDEN,
            },
            "bounded_storage": {
                "strategy": "stratified_token_subsample_raw_hiddens_plus_full_route_membership",
                "why": "SVD fit needs Gram = X'X/n (buildable from raw rows at pack time) and surplus-over-null needs true holdout rows; full raw dump is ~1.5GB; per-expert Gram dumps are multi-GB and lose holdout rows",
                "max_hidden_tokens_per_layer": arguments.max_hidden_tokens_per_layer,
                "hidden_tokens_retained_per_layer": hidden_tokens_retained,
                "layers": QWEN30_LAYERS,
                "hidden_width": QWEN30_HIDDEN,
                "total_tokens_executed": tokens_executed,
                "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
                "retained_hidden_budget_bytes": retained_hidden_budget_bytes,
                "retained_hidden_bytes_written": hidden_bytes_written,
                "full_route_membership_for_every_token_every_layer": true,
                "rejected_alternatives": {
                    "full_raw_all_tokens": "unbounded (~1.5 GB for this prompt set); not acceptable",
                    "per_expert_gram_only": "2048x2048 f32 = 16MB per (layer,expert); multi-GB at scale and cannot score holdout null/surplus"
                },
            },
            "capture_summary": {
                "probe_count": probe_rows.len(),
                "total_tokens": tokens_executed,
                "layers_executed": QWEN30_LAYERS,
                "broad_activation_diversity": true,
                "all_layer_activation_capture": true,
                "hidden_tokens_retained": hidden_tokens_retained,
            },
            "probes": probe_rows,
            "logit_provenance": {
                "status": "EXECUTED_BUT_NOT_RETAINED",
                "reason": "all-layer residual path runs final norm/lm_head/sampler via the shared greedy primitive so deeper-layer hiddens are causally real; logits are not written and are not a capability claim"
            },
            "claim_boundary": claim_boundary,
        });
        let result_path = arguments.output_dir.join("capture-result.json");
        write_json_new(&result_path, &result).unwrap_or_else(|error| fail(error));
        println!(
            "{}",
            serde_json::to_string(&json!({
                "status": result.get("status"),
                "schema": RESULT_SCHEMA,
                "output_dir": arguments.output_dir,
                "capture_summary": result.get("capture_summary"),
                "bounded_storage": {
                    "strategy": "stratified_token_subsample_raw_hiddens_plus_full_route_membership",
                    "hidden_tokens_retained_per_layer": hidden_tokens_retained,
                    "retained_hidden_bytes_written": hidden_bytes_written,
                    "naive_all_token_hidden_bytes_estimate": naive_hidden_bytes,
                },
                "claim_boundary": {
                    "does_not_claim_coherence_hcli_tps_tg_capability_or_tournament": true,
                    "diagnostic_activation_pricing_only": true,
                },
            }))
            .expect("summary must serialize")
        );
    }
}

#[cfg(target_os = "macos")]
fn main() {
    macos::run();
}
