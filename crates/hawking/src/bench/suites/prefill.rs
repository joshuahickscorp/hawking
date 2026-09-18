use super::super::BenchOptions;
use anyhow::Result;
use hawking_core::{EngineConfig, GenerateRequest, SamplingParams, StreamEvent};
use std::time::Instant;

pub fn run(opts: &BenchOptions) -> Result<serde_json::Value> {
    let cfg = EngineConfig::default();
    let mut engine = hawking_core::model::load_engine(&opts.weights, cfg)
        .map_err(|e| anyhow::anyhow!("load engine: {e}"))?;

    // 1024-token prompt produced by repeating a known phrase; for
    // honesty we'd pull from a fixed corpus, but that lands with the
    // wax suite where reproducibility matters most.
    let prompt = "the quick brown fox jumps over the lazy dog. ".repeat(80);
    let mut prefill_ns = Vec::new();
    let mut ttft_ns = Vec::new();
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
            json_mode: false,
        };
        let start = Instant::now();
        let mut first_token: Option<u64> = None;
        let mut sink = |ev: StreamEvent| {
            if matches!(ev, StreamEvent::Token { .. }) && first_token.is_none() {
                first_token = Some(start.elapsed().as_nanos().min(u64::MAX as u128) as u64);
            }
        };
        let stats = engine
            .generate(req, &mut sink)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        prefill_ns.push(stats.prefill_elapsed_ns());
        ttft_ns.push(first_token.unwrap_or(stats.prefill_elapsed_ns()));
    }
    let median = |xs: &mut Vec<u64>| {
        xs.sort_unstable();
        xs[xs.len() / 2]
    };
    let p_med_ns = median(&mut prefill_ns.clone());
    let t_med_ns = median(&mut ttft_ns.clone());
    let p_med = p_med_ns as f64 / 1_000_000.0;
    let t_med = t_med_ns as f64 / 1_000_000.0;
    let prompt_tokens_estimate = 1024usize;
    Ok(serde_json::json!({
        "timing_unit": "ns",
        "prefill_ns_median": p_med_ns,
        "ttft_ns_median": t_med_ns,
        "prefill_ms_median": p_med,
        "ttft_ms_median": t_med,
        "prefill_tps_estimate": (prompt_tokens_estimate as f64) / (p_med / 1000.0).max(1e-9),
        "trials_ns": prefill_ns,
        "trials_ms": prefill_ns.iter().map(|nanoseconds| *nanoseconds as f64 / 1_000_000.0).collect::<Vec<_>>(),
        "trials": prefill_ns.iter().map(|nanoseconds| *nanoseconds as f64 / 1_000_000.0).collect::<Vec<_>>(),
    }))
}
