use super::{Competitor, Measurement};
use anyhow::{anyhow, Result};
use dismantle_core::{
    Engine, EngineConfig, GenStats, GenerateRequest, SamplingParams, StreamEvent,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct DismantleBackend {
    weights: PathBuf,
    engine: Option<Box<dyn Engine>>,
    version: String,
}

impl DismantleBackend {
    pub fn new(weights: &Path) -> Self {
        Self {
            weights: weights.to_owned(),
            engine: None,
            version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }

    fn ensure_loaded(&mut self) -> Result<&mut Box<dyn Engine>> {
        if self.engine.is_none() {
            let cfg = EngineConfig::default();
            self.engine = Some(
                dismantle_core::model::load_engine(&self.weights, cfg)
                    .map_err(|e| anyhow!("dismantle load: {e}"))?,
            );
        }
        Ok(self.engine.as_mut().unwrap())
    }
}

impl Competitor for DismantleBackend {
    fn name(&self) -> &'static str {
        "dismantle"
    }
    fn version(&self) -> String {
        self.version.clone()
    }
    fn phase_tag(&self) -> Option<u32> {
        Some(0)
    } // Phase 0 today.

    fn run(
        &mut self,
        _weights: &Path,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> Result<Measurement> {
        let engine = self.ensure_loaded()?;
        let req = GenerateRequest {
            prompt: prompt.to_string(),
            max_new_tokens: max_tokens,
            sampling: SamplingParams {
                temperature,
                seed: Some(42),
                ..SamplingParams::default()
            },
            stop: Vec::new(),
            abort: None,
            max_stall_ms: 0,
            json_mode: false,
        };

        let start = Instant::now();
        let mut first_token_at: Option<f64> = None;
        let mut text = String::new();
        let mut done_stats: Option<GenStats> = None;
        let mut sink = |ev: StreamEvent| match ev {
            StreamEvent::Token { text: t, .. } => {
                if first_token_at.is_none() {
                    first_token_at = Some(start.elapsed().as_secs_f64() * 1000.0);
                }
                text.push_str(&t);
            }
            StreamEvent::Done { stats, reason: _ } => {
                done_stats = Some(stats);
            }
        };
        engine
            .generate(req, &mut sink)
            .map_err(|e| anyhow!("dismantle generate: {e}"))?;
        let stats = done_stats.unwrap_or_default();

        let decode_secs = stats.decode_ms / 1000.0;
        let decode_tps = if decode_secs > 0.0 {
            Some(stats.completion_tokens as f64 / decode_secs)
        } else {
            None
        };
        let prefill_secs = stats.prefill_ms / 1000.0;
        let prefill_tps = if prefill_secs > 0.0 {
            Some(stats.prompt_tokens as f64 / prefill_secs)
        } else {
            None
        };

        Ok(Measurement {
            decode_tps,
            prefill_tps,
            ttft_ms: first_token_at.or(Some(stats.prefill_ms)),
            peak_rss_mb: None, // not measured in-process; the shell harness samples ps
            output: text,
        })
    }
}
