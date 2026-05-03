//! prefill-tps@1024: tokens/sec on a 1024-token prompt before the
//! first generation step. The headline benchmark for Phase 2 (single-
//! launch MoE). Also reports time-to-first-token (TTFT).

use crate::BenchOptions;
use anyhow::Result;
use dismantle_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};
use std::time::Instant;

pub fn run(opts: &BenchOptions) -> Result<serde_json::Value> {
    let cfg = EngineConfig::default();
    let mut engine = dismantle_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;

    // 1024-token prompt produced by repeating a known phrase; for
    // honesty we'd pull from a fixed corpus, but that lands with the
    // wax suite where reproducibility matters most.
    let prompt = "the quick brown fox jumps over the lazy dog. ".repeat(80);
    let mut prefill_ms = Vec::new();
    let mut ttft_ms = Vec::new();
    for _ in 0..opts.trials {
        let req = GenerateRequest {
            prompt: prompt.clone(),
            max_new_tokens: 1,
            sampling: SamplingParams {
                temperature: 0.0,
                seed: Some(0),
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
        };
        let start = Instant::now();
        let mut first_token: Option<f64> = None;
        let mut sink = |ev: StreamEvent| {
            if matches!(ev, StreamEvent::Token { .. }) && first_token.is_none() {
                first_token = Some(start.elapsed().as_secs_f64() * 1000.0);
            }
        };
        let stats = engine
            .generate(req, &mut sink)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        prefill_ms.push(stats.prefill_ms);
        ttft_ms.push(first_token.unwrap_or(stats.prefill_ms));
    }
    let median = |xs: &mut Vec<f64>| {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };
    let p_med = median(&mut prefill_ms.clone());
    let t_med = median(&mut ttft_ms.clone());
    let prompt_tokens_estimate = 1024usize;
    Ok(serde_json::json!({
        "prefill_ms_median": p_med,
        "ttft_ms_median": t_med,
        "prefill_tps_estimate": (prompt_tokens_estimate as f64) / (p_med / 1000.0).max(1e-9),
        "trials": prefill_ms,
    }))
}
