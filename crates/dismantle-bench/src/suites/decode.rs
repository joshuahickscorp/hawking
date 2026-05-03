//! decode-tps: tokens/sec at batch=1, configurable completion length, fixed
//! seed. Reported as min / median / max over N trials.
//!
//! Backend-aware as of haul 3 / B4: when `opts.backend != "dismantle"`,
//! the suite drives a `Competitor` (llama.cpp or MLX) instead of the
//! in-process Engine. The shape of the JSON result is the same
//! regardless of backend so run-gates.sh's `bench-decode` validator
//! can parse uniformly.

use crate::competitors::{Competitor, LlamaCppBackend, MlxBackend};
use crate::BenchOptions;
use anyhow::{anyhow, Result};
use dismantle_core::{
    profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
    StreamEvent,
};

const PROMPT: &str = "Once upon a time";
const TEMPERATURE: f32 = 0.0;

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
        ..EngineConfig::default()
    };
    let mut engine = dismantle_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow!("load engine: {e}"))?;

    let mut tps = Vec::with_capacity(opts.trials);
    let mut trial_stats = Vec::with_capacity(opts.trials);
    for _ in 0..opts.trials {
        let req = GenerateRequest {
            prompt: PROMPT.into(),
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
        trial_stats.push(serde_json::json!({
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
        }));
    }
    let profile_path = opts
        .kernel_profile
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());
    Ok(finalize(
        tps,
        "dismantle",
        None,
        opts.max_new_tokens,
        trial_stats,
        profile_path.as_deref(),
        speculate_mode.as_str(),
        opts.verify_window,
    ))
}

fn run_competitor(opts: &BenchOptions, backend: &mut dyn Competitor) -> Result<serde_json::Value> {
    let mut tps = Vec::with_capacity(opts.trials);
    let mut trial_stats = Vec::with_capacity(opts.trials);
    let name = backend.name();
    let version = backend.version();

    // Drop the first trial as warm-up — model load and shader compile
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
