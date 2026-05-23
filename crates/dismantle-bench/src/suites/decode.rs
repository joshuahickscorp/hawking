use crate::competitors::{Competitor, LlamaCppBackend, MlxBackend};
use crate::BenchOptions;
use anyhow::{anyhow, Result};
use dismantle_core::{
    profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
    StreamEvent,
};

const PROMPT: &str = "Once upon a time";
const TEMPERATURE: f32 = 0.0;

/// Returns the prompt for this decode bench. Defaults to `PROMPT` but can
/// be overridden by `DISMANTLE_BENCH_PROMPT_FILE=<path>` for long-context
/// gate decisions (e.g. the MLA flash-attn lever which only wins past
/// ~1K seq_len). The file is read once and the contents used verbatim.
fn resolve_prompt() -> String {
    if let Ok(path) = std::env::var("DISMANTLE_BENCH_PROMPT_FILE") {
        match std::fs::read_to_string(&path) {
            Ok(s) => return s,
            Err(e) => eprintln!(
                "warning: DISMANTLE_BENCH_PROMPT_FILE={path} unreadable ({e}); falling back to default prompt"
            ),
        }
    }
    PROMPT.to_string()
}

pub fn run(opts: &BenchOptions) -> Result<serde_json::Value> {
    let backend = opts.backend.as_str();
    match backend {
        "dismantle" | "" => run_dismantle(opts),
        "llamacpp" | "llama.cpp" => run_competitor(opts, &mut LlamaCppBackend::new()),
        "mlx" => run_competitor(opts, &mut MlxBackend::new()),
        other => Err(anyhow!(
            "unknown backend `{other}` (expected dismantle/llamacpp/mlx)"
        )),
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    mut tps: Vec<f64>,
    backend: &str,
    version: Option<String>,
    max_new_tokens: usize,
    trial_stats: Vec<serde_json::Value>,
    kernel_profile: Option<&str>,
    speculate_mode: &str,
    verify_window: usize,
) -> serde_json::Value {
    tps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if tps.is_empty() {
        0.0
    } else {
        tps[tps.len() / 2]
    };
    let min = tps.first().copied().unwrap_or(0.0);
    let max = tps.last().copied().unwrap_or(0.0);
    serde_json::json!({
        // Top-level alias used by run-gates.sh's `bench-decode` and
        // `perf-ratio-assert` validators.
        "decode_tps":            median,
        "tokens_per_sec_min":    min,
        "tokens_per_sec_median": median,
        "tokens_per_sec_max":    max,
        "trials":                tps,
        "trial_stats":           trial_stats,
        "backend":               backend,
        "version":               version,
        "prompt":                PROMPT,
        "max_new_tokens":        max_new_tokens,
        "temperature":           TEMPERATURE,
        "metric":                "decode_only_tps",
        "kernel_profile":        kernel_profile,
        "speculate_mode":        speculate_mode,
        "verify_window":         verify_window,
    })
}

fn run_dismantle(opts: &BenchOptions) -> Result<serde_json::Value> {
    let kernel_profile = match opts.kernel_profile.as_ref() {
        Some(path) => Some(KernelProfile::load(path)?),
        None => None,
    };
    let speculate_mode = SpeculateMode::from_cli(
        if opts.speculate_mode.is_empty() {
            None
        } else {
            Some(opts.speculate_mode.as_str())
        },
        false,
    )
    .map_err(|e| anyhow!("{e}"))?;
    let cfg = EngineConfig {
        kernel_profile,
        speculate_mode,
        speculate: speculate_mode != SpeculateMode::Off,
        verify_window: opts.verify_window,
        trace_dispatch: opts.trace_dispatch,
        ..EngineConfig::default()
    };
    let mut engine = dismantle_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow!("load engine: {e}"))?;

    let mut tps = Vec::with_capacity(opts.trials);
    let mut trial_stats = Vec::with_capacity(opts.trials);
    // Accumulate dispatch samples across all trials; per-token timing is
    // meaningful in aggregate. We serialize them in the returned value so
    // lib.rs can compute summaries for --trace-json.
    let mut all_dispatch_samples: Vec<dismantle_core::metal::DispatchSample> = Vec::new();
    let mut total_decode_ms: f64 = 0.0;
    let prompt = resolve_prompt();
    for _ in 0..opts.trials {
        let req = GenerateRequest {
            prompt: prompt.clone(),
            max_new_tokens: opts.max_new_tokens,
            sampling: SamplingParams {
                temperature: TEMPERATURE,
                seed: Some(42),
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
        };
        let mut produced = 0usize;
        let mut sink = |ev: StreamEvent| {
            if let StreamEvent::Token { .. } = ev {
                produced += 1;
            }
        };
        let stats = engine
            .generate(req, &mut sink)
            .map_err(|e| anyhow!("{e}"))?;
        let secs = (stats.decode_ms / 1000.0).max(1e-6);
        let decode_tps = stats.completion_tokens as f64 / secs;
        tps.push(decode_tps);
        total_decode_ms += stats.decode_ms;
        let mut ts = serde_json::json!({
            "decode_tps": decode_tps,
            "produced_tokens": produced,
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "prefill_ms": stats.prefill_ms,
            "decode_ms": stats.decode_ms,
            "draft_accepted": stats.draft_accepted,
            "draft_rejected": stats.draft_rejected,
            "profile_id": stats.profile_id,
            "device_id": stats.device_id,
            "trace_hash": stats.trace_hash,
            "dispatch_count": stats.dispatch_samples.len(),
        });
        // Add structural counters when tracing was active (via --trace-dispatch
        // flag or DISMANTLE_TRACE_DISPATCH env var -- both set trace_dispatch on
        // the MetalContext, so non-zero counts indicate tracing was on).
        if stats.metal_buffers_created > 0 && stats.completion_tokens > 0 {
            let ct = stats.completion_tokens as f64;
            if let Some(obj) = ts.as_object_mut() {
                obj.insert("metal_buffers_created_per_token".into(),
                    serde_json::json!(stats.metal_buffers_created as f64 / ct));
                obj.insert("cpu_alloc_bytes_per_token".into(),
                    serde_json::json!(stats.metal_bytes_allocated as f64 / ct));
                obj.insert("dispatch_commits_per_token".into(),
                    serde_json::json!(stats.metal_commits as f64 / ct));
            }
        }
        trial_stats.push(ts);
        all_dispatch_samples.extend(stats.dispatch_samples);
    }
    let profile_path = opts
        .kernel_profile
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    let mut result = finalize(
        tps,
        "dismantle",
        None,
        opts.max_new_tokens,
        trial_stats,
        profile_path.as_deref(),
        speculate_mode.as_str(),
        opts.verify_window,
    );
    // Attach dispatch samples to the result so lib.rs can build summaries.
    // Raw samples are only present when DISMANTLE_TRACE_DISPATCH=1 (otherwise
    // all_dispatch_samples is empty and we omit both fields).
    if !all_dispatch_samples.is_empty() {
        let total_dispatch_us: u64 = all_dispatch_samples.iter().map(|s| s.wall_us).sum();
        let total_decode_us = (total_decode_ms * 1000.0) as u64;
        if let Some(obj) = result.as_object_mut() {
            obj.insert(
                "dispatch_samples".to_string(),
                serde_json::to_value(&all_dispatch_samples).unwrap_or_default(),
            );
            obj.insert(
                "dispatch_total_us".to_string(),
                serde_json::Value::Number(total_dispatch_us.into()),
            );
            obj.insert(
                "dispatch_total_decode_us".to_string(),
                serde_json::Value::Number(total_decode_us.into()),
            );
        }
    }
    Ok(result)
}

fn run_competitor(opts: &BenchOptions, backend: &mut dyn Competitor) -> Result<serde_json::Value> {
    let mut tps = Vec::with_capacity(opts.trials);
    let mut trial_stats = Vec::with_capacity(opts.trials);
    let name = backend.name();
    let version = backend.version();

    // Drop the first trial as warm-up -- model load and shader compile
    // are first-trial costs that distort decode-tps. Trial budget +1
    // when called this way.
    let trials_with_warmup = opts.trials + 1;
    for trial in 0..trials_with_warmup {
        let m = backend.run(&opts.weights, PROMPT, opts.max_new_tokens, TEMPERATURE)?;
        if trial == 0 {
            // warm-up, discard
            continue;
        }
        if let Some(t) = m.decode_tps {
            tps.push(t);
            trial_stats.push(serde_json::json!({
                "decode_tps": t,
                "prefill_tps": m.prefill_tps,
                "ttft_ms": m.ttft_ms,
                "peak_rss_mb": m.peak_rss_mb,
                "output_chars": m.output.len(),
            }));
        }
    }
    if tps.is_empty() {
        return Err(anyhow!(
            "backend `{name}` produced no decode_tps measurements"
        ));
    }
    Ok(finalize(
        tps,
        name,
        Some(version),
        opts.max_new_tokens,
        trial_stats,
        None,
        "off",
        0,
    ))
}
