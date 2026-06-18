use anyhow::Result;
use dismantle_core::{
    profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode,
    StreamEvent,
};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, Write};
use std::path::PathBuf;

pub struct BenchServerOptions {
    pub weights: PathBuf,
    pub kernel_profile: Option<PathBuf>,
    pub speculate: Option<String>,
    pub verify_window: usize,
    pub trace_dispatch: bool,
}

#[derive(Debug, Deserialize)]
struct BenchRequest {
    id: String,
    prompt: String,
    max_tokens: usize,
    #[serde(default)]
    seed: Option<u64>,
}

#[derive(Debug, Serialize)]
struct BenchResponse {
    id: String,
    prompt_tokens: usize,
    completion_tokens: usize,
    completion_text: String,
    dec_tps: f64,
    prefill_ms: f64,
    decode_ms: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    structural: Option<StructuralMetrics>,
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct StructuralMetrics {
    commits_per_token: f64,
    buffers_per_token: f64,
    alloc_bytes_per_token: f64,
}

pub fn run(opts: BenchServerOptions) -> Result<()> {
    let speculate_mode = SpeculateMode::from_cli(opts.speculate.as_deref(), false)?;
    let profile = match opts.kernel_profile.as_ref() {
        Some(p) => Some(KernelProfile::load(p)?),
        None => None,
    };
    let cfg = EngineConfig {
        speculate: speculate_mode != SpeculateMode::Off,
        speculate_mode,
        verify_window: opts.verify_window,
        kernel_profile: profile,
        trace_dispatch: opts.trace_dispatch,
        ..Default::default()
    };

    eprintln!(
        "[bench-server] loading model from {}",
        opts.weights.display()
    );
    let load_start = std::time::Instant::now();
    let mut engine = dismantle_core::model::load_engine(&opts.weights, cfg)?;
    eprintln!(
        "[bench-server] model loaded in {:.1}s — ready for requests on stdin (JSON-line)",
        load_start.elapsed().as_secs_f64()
    );
    eprintln!(
        "[bench-server] request format: {{\"id\":\"req_1\",\"prompt\":\"...\",\"max_tokens\":12}}"
    );

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = line?;
        let line = line.trim().to_string();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let req: BenchRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                let resp = BenchResponse {
                    id: "unknown".into(),
                    prompt_tokens: 0,
                    completion_tokens: 0,
                    completion_text: String::new(),
                    dec_tps: 0.0,
                    prefill_ms: 0.0,
                    decode_ms: 0.0,
                    structural: None,
                    error: Some(format!("parse error: {e}")),
                };
                writeln!(out, "{}", serde_json::to_string(&resp)?)?;
                out.flush()?;
                continue;
            }
        };

        let id = req.id.clone();

        // Reset KV cache before each request — stateless across requests (v1).
        engine.reset_kv_for_test();

        let gen_req = GenerateRequest {
            prompt: req.prompt.clone(),
            max_new_tokens: req.max_tokens,
            sampling: SamplingParams {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repetition_penalty: 1.0,
                seed: req.seed.or(Some(42)),
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 60_000,
            json_mode: false,
        };

        let mut completion_text = String::new();
        let mut final_stats = None;

        let result = engine.generate(gen_req, &mut |ev| match ev {
            StreamEvent::Token { text, .. } => completion_text.push_str(&text),
            StreamEvent::Done { stats, .. } => final_stats = Some(stats),
        });

        let resp = match result {
            Ok(_gen_stats) => {
                let stats = final_stats.unwrap_or_default();
                let dec_tps = if stats.decode_ms > 0.0 {
                    stats.completion_tokens as f64 / (stats.decode_ms / 1000.0)
                } else {
                    0.0
                };
                let structural = if stats.metal_commits > 0 && stats.completion_tokens > 0 {
                    let ct = stats.completion_tokens as f64;
                    Some(StructuralMetrics {
                        commits_per_token: stats.metal_commits as f64 / ct,
                        buffers_per_token: stats.metal_buffers_created as f64 / ct,
                        alloc_bytes_per_token: stats.metal_bytes_allocated as f64 / ct,
                    })
                } else {
                    None
                };
                BenchResponse {
                    id,
                    prompt_tokens: stats.prompt_tokens,
                    completion_tokens: stats.completion_tokens,
                    completion_text,
                    dec_tps,
                    prefill_ms: stats.prefill_ms,
                    decode_ms: stats.decode_ms,
                    structural,
                    error: None,
                }
            }
            Err(e) => BenchResponse {
                id,
                prompt_tokens: 0,
                completion_tokens: 0,
                completion_text: String::new(),
                dec_tps: 0.0,
                prefill_ms: 0.0,
                decode_ms: 0.0,
                structural: None,
                error: Some(format!("{e}")),
            },
        };

        writeln!(out, "{}", serde_json::to_string(&resp)?)?;
        out.flush()?;
        eprintln!(
            "[bench-server] req={} tokens={} dec_tps={:.2} prefill={:.1}ms decode={:.1}ms",
            resp.id, resp.completion_tokens, resp.dec_tps, resp.prefill_ms, resp.decode_ms
        );
    }

    eprintln!("[bench-server] EOF — exiting");
    Ok(())
}
