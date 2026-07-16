#[rustfmt::skip]
pub mod competitors {
    pub mod hawking {
        use super::{Competitor, Measurement};
        use anyhow::{anyhow, Result};
        use hawking_core::{Engine, EngineConfig, GenStats, GenerateRequest, SamplingParams, StreamEvent};
        use std::path::{Path, PathBuf};
        use std::time::Instant;

        pub struct HawkingBackend {
            weights: PathBuf,
            engine: Option<Box<dyn Engine>>,
            version: String,
        }

        impl HawkingBackend {
            pub fn new(weights: &Path) -> Self {
                Self { weights: weights.to_owned(), engine: None, version: env!("CARGO_PKG_VERSION").to_string() }
            }

            fn ensure_loaded(&mut self) -> Result<&mut Box<dyn Engine>> {
                if self.engine.is_none() {
                    let cfg = EngineConfig::default();
                    self.engine = Some(
                        hawking_core::model::load_engine(&self.weights, cfg)
                            .map_err(|e| anyhow!("hawking load: {e}"))?,
                    );
                }
                Ok(self.engine.as_mut().unwrap())
            }
        }

        impl Competitor for HawkingBackend {
            fn name(&self) -> &'static str {
                "hawking"
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
                    sampling: SamplingParams { temperature, seed: Some(42), ..SamplingParams::default() },
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
                engine.generate(req, &mut sink).map_err(|e| anyhow!("hawking generate: {e}"))?;
                let stats = done_stats.unwrap_or_default();

                let decode_secs = stats.decode_ms / 1000.0;
                let decode_tps =
                    if decode_secs > 0.0 { Some(stats.completion_tokens as f64 / decode_secs) } else { None };
                let prefill_secs = stats.prefill_ms / 1000.0;
                let prefill_tps =
                    if prefill_secs > 0.0 { Some(stats.prompt_tokens as f64 / prefill_secs) } else { None };

                Ok(Measurement {
                    decode_tps,
                    prefill_tps,
                    ttft_ms: first_token_at.or(Some(stats.prefill_ms)),
                    peak_rss_mb: None, // not measured in-process; the shell harness samples ps
                    output: text,
                })
            }
        }
    }
    pub mod llamacpp {
        use super::{extract_after, Competitor, Measurement};
        use anyhow::{bail, Result};
        use std::path::Path;
        use std::process::Command;

        pub struct LlamaCppBackend {
            binary: String,
            version: String,
        }

        impl Default for LlamaCppBackend {
            fn default() -> Self {
                Self::new()
            }
        }

        impl LlamaCppBackend {
            pub fn new() -> Self {
                let binary = std::env::var("HAWKING_LLAMA_CLI").unwrap_or_else(|_| "llama-cli".into());
                let version = Command::new(&binary)
                    .arg("--version")
                    .output()
                    .ok()
                    .and_then(|o| {
                        let mut s = String::from_utf8_lossy(&o.stdout).into_owned();
                        if s.is_empty() {
                            s = String::from_utf8_lossy(&o.stderr).into_owned();
                        }
                        s.lines().next().map(|l| l.to_string())
                    })
                    .unwrap_or_else(|| "unknown".into());
                Self { binary, version }
            }
        }

        impl Competitor for LlamaCppBackend {
            fn name(&self) -> &'static str {
                "llamacpp"
            }
            fn version(&self) -> String {
                self.version.clone()
            }

            fn run(
                &mut self,
                weights: &Path,
                prompt: &str,
                max_tokens: usize,
                temperature: f32,
            ) -> Result<Measurement> {
                let output = Command::new(&self.binary)
                    .arg("--model")
                    .arg(weights)
                    .arg("--prompt")
                    .arg(prompt)
                    .arg("--predict")
                    .arg(max_tokens.to_string())
                    .arg("--temp")
                    .arg(temperature.to_string())
                    .arg("-ngl")
                    .arg("99")
                    .arg("--no-display-prompt")
                    .arg("--no-warmup")
                    .output()?;
                let stderr = String::from_utf8_lossy(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout);
                if !output.status.success() && stderr.is_empty() {
                    bail!("llama-cli exited {} with no stderr", output.status);
                }

                // Look for the two summary lines:
                //   prompt eval time = X ms /  N tokens (Y ms per token, Z tokens per second)
                //         eval time = X ms /  N tokens (Y ms per token, Z tokens per second)
                let prefill_tps = stderr
                    .lines()
                    .find(|l| l.contains("prompt eval time"))
                    .and_then(|l| extract_after(l, "tokens per second"));
                let decode_tps = stderr
                    .lines()
                    .find(|l| {
                        let lt = l.trim_start();
                        lt.starts_with("eval time") || lt.starts_with("       eval time")
                    })
                    .and_then(|l| extract_after(l, "tokens per second"));
                let ttft_ms =
                    stderr.lines().find(|l| l.contains("prompt eval time")).and_then(|l| extract_after(l, "="));

                Ok(Measurement {
                    decode_tps,
                    prefill_tps,
                    ttft_ms,
                    peak_rss_mb: None, // llama-cli doesn't report; could sample externally
                    output: stdout.into_owned(),
                })
            }
        }
    }
    pub mod mlx {
        use super::{extract_after, Competitor, Measurement};
        use anyhow::{bail, Result};
        use std::path::Path;
        use std::process::Command;

        pub struct MlxBackend {
            binary: String,
            version: String,
            model_id: String,
        }

        impl MlxBackend {
            pub fn new() -> Self {
                let binary = std::env::var("HAWKING_MLX_GENERATE").unwrap_or_else(|_| "mlx_lm.generate".into());
                let model_id = std::env::var("HAWKING_MLX_MODEL_ID")
                    .unwrap_or_else(|_| "mlx-community/DeepSeek-V2-Lite-Chat-4bit-mlx".into());
                // mlx_lm doesn't expose --version; cite the python package
                // version instead.
                let version = Command::new("python3")
                    .args(["-c", "import mlx_lm; print(mlx_lm.__version__)"])
                    .output()
                    .ok()
                    .and_then(|o| {
                        let s = String::from_utf8_lossy(&o.stdout).into_owned();
                        s.lines().next().map(|l| format!("mlx_lm {}", l.trim()))
                    })
                    .unwrap_or_else(|| "unknown".into());
                Self { binary, version, model_id }
            }
        }

        impl Default for MlxBackend {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Competitor for MlxBackend {
            fn name(&self) -> &'static str {
                "mlx"
            }
            fn version(&self) -> String {
                self.version.clone()
            }

            fn run(
                &mut self,
                _weights: &Path,
                prompt: &str,
                max_tokens: usize,
                temperature: f32,
            ) -> Result<Measurement> {
                let output = Command::new(&self.binary)
                    .arg("--model")
                    .arg(&self.model_id)
                    .arg("--prompt")
                    .arg(prompt)
                    .arg("--max-tokens")
                    .arg(max_tokens.to_string())
                    .arg("--temp")
                    .arg(temperature.to_string())
                    .output()?;
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !output.status.success() {
                    bail!(
                        "mlx_lm.generate exited {}; stderr: {}",
                        output.status,
                        stderr.lines().take(5).collect::<Vec<_>>().join(" | ")
                    );
                }

                // Parse `Prompt: 39 tokens, 58.373 tokens-per-sec` and
                // `Generation: 8 tokens, 172.062 tokens-per-sec`. mlx_lm
                // writes these to stdout (not stderr), but check both for
                // robustness across versions.
                let blob: String = format!("{stdout}\n{stderr}");
                let prefill_tps = blob
                    .lines()
                    .find(|l| l.trim_start().starts_with("Prompt:"))
                    .and_then(|l| extract_after(l, "tokens,"));
                let decode_tps = blob
                    .lines()
                    .find(|l| l.trim_start().starts_with("Generation:"))
                    .and_then(|l| extract_after(l, "tokens,"));

                // Trim the heredoc fence and capture only the model's own
                // output between the `==========` markers, matching what we'd
                // hand back from llama-cli.
                let body = stdout
                    .split("==========")
                    .nth(1)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| stdout.into_owned());

                Ok(Measurement { decode_tps, prefill_tps, ttft_ms: None, peak_rss_mb: None, output: body })
            }
        }

        #[cfg(test)]
        mod tests {
            use crate::competitors::extract_after;

            #[test]
            fn parses_mlx_summary_lines() {
                let blob = "==========\n\
                            Some output\n\
                            ==========\n\
                            Prompt: 39 tokens, 58.373 tokens-per-sec\n\
                            Generation: 8 tokens, 172.062 tokens-per-sec\n\
                            Peak memory: 0.790 GB\n";
                let prefill = blob
                    .lines()
                    .find(|l| l.trim_start().starts_with("Prompt:"))
                    .and_then(|l| extract_after(l, "tokens,"));
                let decode = blob
                    .lines()
                    .find(|l| l.trim_start().starts_with("Generation:"))
                    .and_then(|l| extract_after(l, "tokens,"));
                assert_eq!(prefill, Some(58.373));
                assert_eq!(decode, Some(172.062));
            }
        }
    }

    pub use hawking::HawkingBackend;
    pub use llamacpp::LlamaCppBackend;
    pub use mlx::MlxBackend;

    use anyhow::Result;
    use std::path::Path;

    /// A single (backend, prompt) measurement.
    #[derive(Debug, Clone, Default)]
    pub struct Measurement {
        pub decode_tps: Option<f64>,
        pub prefill_tps: Option<f64>,
        pub ttft_ms: Option<f64>,
        pub peak_rss_mb: Option<f64>,
        pub output: String,
    }

    /// One backend in the head-to-head matrix.
    pub trait Competitor: Send {
        /// Short identifier used in JSON output ("llamacpp", "hawking").
        fn name(&self) -> &'static str;

        /// Pinned version string. Captured at construction time so the
        /// JSON output is reproducible against the same backend version.
        fn version(&self) -> String;

        /// Phase tag -- only hawking uses this (returns `Some(0)` until
        /// Phase 1 lands real Metal kernels). All competitors return `None`.
        fn phase_tag(&self) -> Option<u32> {
            None
        }

        /// Run one measurement on this prompt.
        fn run(&mut self, weights: &Path, prompt: &str, max_tokens: usize, temperature: f32) -> Result<Measurement>;
    }

    /// Parse a number out of an arbitrary output blob using a regex-ish
    /// substring match. Helper used by the `llamacpp` stdout/stderr
    /// parser -- keeps the backend files small and tested.
    pub(crate) fn extract_after(s: &str, marker: &str) -> Option<f64> {
        let pos = s.find(marker)?;
        let tail = &s[pos + marker.len()..];
        // Skip whitespace, '=', '(', etc.
        let tail = tail.trim_start_matches([' ', '=', '(', ':']);
        let mut end = 0usize;
        for (i, c) in tail.char_indices() {
            if c.is_ascii_digit() || c == '.' || c == '-' {
                end = i + c.len_utf8();
            } else {
                break;
            }
        }
        if end == 0 {
            return None;
        }
        tail[..end].parse::<f64>().ok()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn extract_after_handles_assignment() {
            assert_eq!(extract_after("foo = 42.5 bar", "foo"), Some(42.5));
        }

        #[test]
        fn extract_after_handles_paren() {
            let s = "eval time = 1234.5 ms / 256 runs (123.45 tokens per second)";
            assert_eq!(extract_after(s, "(").map(|x| (x * 100.0).round() / 100.0), Some(123.45));
        }

        #[test]
        fn extract_after_returns_none_for_missing() {
            assert_eq!(extract_after("nothing here", "missing"), None);
        }
    }
}
#[rustfmt::skip]
pub mod suites {
    pub mod bandwidth {
        use crate::BenchOptions;
        use anyhow::Result;

        pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "phase1_pending": true,
                "note": "real numbers land with the wedge-2 fused-dequant kernel (Phase 1).",
            }))
        }
    }
    pub mod competitive {
        use crate::competitors::{Competitor, HawkingBackend, LlamaCppBackend};
        use crate::BenchOptions;
        use anyhow::Result;
        use serde_json::json;

        pub fn run(opts: &BenchOptions) -> Result<serde_json::Value> {
            let prompts = load_prompts();
            if prompts.is_empty() {
                return Ok(json!({
                    "error": "no prompts found in tools/competitors/prompts.txt; \
                              run from the workspace root or copy prompts.txt next \
                              to the binary",
                }));
            }

            let mut backends: Vec<Box<dyn Competitor>> =
                vec![Box::new(LlamaCppBackend::new()), Box::new(HawkingBackend::new(&opts.weights))];

            let backend_meta: Vec<serde_json::Value> = backends
                .iter()
                .map(|b| {
                    json!({
                        "name": b.name(),
                        "version": b.version(),
                        "phase": b.phase_tag(),
                    })
                })
                .collect();

            let mut rows = Vec::new();
            for (prompt_idx, (tier, prompt)) in prompts.iter().enumerate() {
                for backend in backends.iter_mut() {
                    let mut trials = Vec::new();
                    for _ in 0..opts.trials {
                        match backend.run(&opts.weights, prompt, opts.max_new_tokens, 0.0) {
                            Ok(m) => trials.push(json!({
                                "decode_tps":  m.decode_tps,
                                "prefill_tps": m.prefill_tps,
                                "ttft_ms":     m.ttft_ms,
                                "peak_rss_mb": m.peak_rss_mb,
                                "output_chars": m.output.len(),
                            })),
                            Err(e) => trials.push(json!({"error": e.to_string()})),
                        }
                    }
                    rows.push(json!({
                        "prompt_idx": prompt_idx + 1,
                        "tier": tier,
                        "backend": backend.name(),
                        "trials": trials,
                    }));
                }
            }

            Ok(json!({
                "model": opts.model_id,
                "hw": detect_hw(),
                "backends": backend_meta,
                "rows": rows,
                "audit_doc": "docs/competitive_audit.md",
            }))
        }

        /// Read `tools/competitors/prompts.txt` from the workspace root.
        /// Returns `Vec<(tier, prompt)>`. Falls back to a tiny built-in list
        /// if the file is missing (so the suite still produces *something*
        /// for smoke-testing without the workspace layout in place).
        fn load_prompts() -> Vec<(String, String)> {
            let candidates = [
                std::path::PathBuf::from("tools/competitors/prompts.txt"),
                std::path::PathBuf::from("../tools/competitors/prompts.txt"),
                std::path::PathBuf::from("../../tools/competitors/prompts.txt"),
            ];
            for p in &candidates {
                if let Ok(s) = std::fs::read_to_string(p) {
                    return parse_prompts(&s);
                }
            }
            vec![
                ("SHORT".into(), "Once upon a time".into()),
                ("MED".into(), "Explain how Mixture of Experts routing works in three sentences.".into()),
            ]
        }

        fn parse_prompts(s: &str) -> Vec<(String, String)> {
            s.lines()
                .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
                .filter_map(|l| {
                    let mut it = l.splitn(2, '|');
                    let tier = it.next()?.trim().to_string();
                    let prompt = it.next()?.trim().to_string();
                    Some((tier, prompt))
                })
                .collect()
        }

        fn detect_hw() -> String {
            #[cfg(target_os = "macos")]
            {
                std::process::Command::new("sysctl")
                    .args(["-n", "machdep.cpu.brand_string"])
                    .output()
                    .ok()
                    .and_then(|o| String::from_utf8(o.stdout).ok())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| "unknown".into())
            }
            #[cfg(not(target_os = "macos"))]
            {
                "non-macos".into()
            }
        }

        #[cfg(test)]
        mod tests {
            use super::*;

            #[test]
            fn parse_prompts_strips_comments_and_blanks() {
                let s = "# header\n\nSHORT|hello\nMED|world\n\n# trailer\n";
                let p = parse_prompts(s);
                assert_eq!(p.len(), 2);
                assert_eq!(p[0], ("SHORT".to_string(), "hello".to_string()));
                assert_eq!(p[1], ("MED".to_string(), "world".to_string()));
            }
        }
    }
    pub mod decode {
        use crate::competitors::{Competitor, LlamaCppBackend, MlxBackend};
        use crate::BenchOptions;
        use anyhow::{anyhow, Result};
        use hawking_core::{
            profile::KernelProfile, EngineConfig, GenerateRequest, SamplingParams, SpeculateMode, StreamEvent,
        };

        const PROMPT: &str = "Once upon a time";
        const TEMPERATURE: f32 = 0.0;

        /// Returns the prompt for this decode bench. Defaults to `PROMPT` but can
        /// be overridden by `HAWKING_BENCH_PROMPT_FILE=<path>` for long-context
        /// gate decisions (e.g. the MLA flash-attn lever which only wins past
        /// ~1K seq_len). The file is read once and the contents used verbatim.
        fn resolve_prompt() -> String {
            if let Ok(path) = std::env::var("HAWKING_BENCH_PROMPT_FILE") {
                match std::fs::read_to_string(&path) {
                    Ok(s) => return s,
                    Err(e) => eprintln!(
                        "warning: HAWKING_BENCH_PROMPT_FILE={path} unreadable ({e}); falling back to default prompt"
                    ),
                }
            }
            PROMPT.to_string()
        }

        pub fn run(opts: &BenchOptions) -> Result<serde_json::Value> {
            let backend = opts.backend.as_str();
            match backend {
                "hawking" | "" => run_hawking(opts),
                "llamacpp" | "llama.cpp" => run_competitor(opts, &mut LlamaCppBackend::new()),
                "mlx" => run_competitor(opts, &mut MlxBackend::new()),
                other => Err(anyhow!("unknown backend `{other}` (expected hawking/llamacpp/mlx)")),
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
            let median = if tps.is_empty() { 0.0 } else { tps[tps.len() / 2] };
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

        fn run_hawking(opts: &BenchOptions) -> Result<serde_json::Value> {
            let kernel_profile = match opts.kernel_profile.as_ref() {
                Some(path) => Some(KernelProfile::load(path)?),
                None => None,
            };
            let speculate_mode = SpeculateMode::from_cli(
                if opts.speculate_mode.is_empty() { None } else { Some(opts.speculate_mode.as_str()) },
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
            let mut engine =
                hawking_core::model::load_engine(&opts.weights, cfg).map_err(|e| anyhow!("load engine: {e}"))?;

            let mut tps = Vec::with_capacity(opts.trials);
            let mut trial_stats = Vec::with_capacity(opts.trials);
            // Accumulate dispatch samples across all trials; per-token timing is
            // meaningful in aggregate. We serialize them in the returned value so
            // lib.rs can compute summaries for --trace-json.
            let mut all_dispatch_samples: Vec<hawking_core::metal::DispatchSample> = Vec::new();
            let mut total_decode_ms: f64 = 0.0;
            let prompt = resolve_prompt();
            for _ in 0..opts.trials {
                let req = GenerateRequest {
                    prompt: prompt.clone(),
                    max_new_tokens: opts.max_new_tokens,
                    sampling: SamplingParams { temperature: TEMPERATURE, seed: Some(42), ..SamplingParams::default() },
                    stop: Vec::new(),
                    abort: None,
                    max_stall_ms: 0,
                    json_mode: false,
                };
                let mut produced = 0usize;
                let mut sink = |ev: StreamEvent| {
                    if let StreamEvent::Token { .. } = ev {
                        produced += 1;
                    }
                };
                let stats = engine.generate(req, &mut sink).map_err(|e| anyhow!("{e}"))?;
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
                // flag or HAWKING_TRACE_DISPATCH env var -- both set trace_dispatch on
                // the MetalContext, so non-zero counts indicate tracing was on).
                if stats.metal_buffers_created > 0 && stats.completion_tokens > 0 {
                    let ct = stats.completion_tokens as f64;
                    if let Some(obj) = ts.as_object_mut() {
                        obj.insert(
                            "metal_buffers_created_per_token".into(),
                            serde_json::json!(stats.metal_buffers_created as f64 / ct),
                        );
                        obj.insert(
                            "cpu_alloc_bytes_per_token".into(),
                            serde_json::json!(stats.metal_bytes_allocated as f64 / ct),
                        );
                        obj.insert(
                            "dispatch_commits_per_token".into(),
                            serde_json::json!(stats.metal_commits as f64 / ct),
                        );
                    }
                }
                trial_stats.push(ts);
                all_dispatch_samples.extend(stats.dispatch_samples);
            }
            let profile_path = opts.kernel_profile.as_ref().map(|p| p.to_string_lossy().to_string());
            let mut result = finalize(
                tps,
                "hawking",
                None,
                opts.max_new_tokens,
                trial_stats,
                profile_path.as_deref(),
                speculate_mode.as_str(),
                opts.verify_window,
            );
            // Attach dispatch samples to the result so lib.rs can build summaries.
            // Raw samples are only present when HAWKING_TRACE_DISPATCH=1 (otherwise
            // all_dispatch_samples is empty and we omit both fields).
            if !all_dispatch_samples.is_empty() {
                let total_dispatch_us: u64 = all_dispatch_samples.iter().map(|s| s.wall_us).sum();
                let total_decode_us = (total_decode_ms * 1000.0) as u64;
                if let Some(obj) = result.as_object_mut() {
                    obj.insert(
                        "dispatch_samples".to_string(),
                        serde_json::to_value(&all_dispatch_samples).unwrap_or_default(),
                    );
                    obj.insert("dispatch_total_us".to_string(), serde_json::Value::Number(total_dispatch_us.into()));
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
                return Err(anyhow!("backend `{name}` produced no decode_tps measurements"));
            }
            Ok(finalize(tps, name, Some(version), opts.max_new_tokens, trial_stats, None, "off", 0))
        }
    }
    pub mod prefill {
        use crate::BenchOptions;
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
            let mut prefill_ms = Vec::new();
            let mut ttft_ms = Vec::new();
            for _ in 0..opts.trials {
                let req = GenerateRequest {
                    prompt: prompt.clone(),
                    max_new_tokens: 1,
                    sampling: SamplingParams { temperature: 0.0, seed: Some(0), ..SamplingParams::default() },
                    stop: Vec::new(),
                    abort: None,
                    max_stall_ms: 0,
                    json_mode: false,
                };
                let start = Instant::now();
                let mut first_token: Option<f64> = None;
                let mut sink = |ev: StreamEvent| {
                    if matches!(ev, StreamEvent::Token { .. }) && first_token.is_none() {
                        first_token = Some(start.elapsed().as_secs_f64() * 1000.0);
                    }
                };
                let stats = engine.generate(req, &mut sink).map_err(|e| anyhow::anyhow!("{e}"))?;
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
    }
    pub mod throughput {
        use crate::BenchOptions;
        use anyhow::Result;

        pub fn run(_opts: &BenchOptions) -> Result<serde_json::Value> {
            Ok(serde_json::json!({
                "phase4_pending": true,
                "note": "batch>1 lands with the continuous-batching slot manager (Phase 4).",
            }))
        }
    }
}

use anyhow::Result;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub weights: PathBuf,
    pub model_id: String,
    pub suite: String,
    pub json_out: Option<PathBuf>,
    pub trials: usize,
    pub max_new_tokens: usize,
    pub trace_json: Option<PathBuf>,
    pub kernel_profile: Option<PathBuf>,
    pub speculate_mode: String,
    pub verify_window: usize,
    /// Which backend to run a single-suite bench against. `"hawking"`
    /// (default) drives the in-process Engine; `"llamacpp"` and
    /// `"mlx"` shell out to the corresponding competitor binaries.
    /// Only the decode and prefill suites honor this -- the
    /// `competitive` suite explicitly runs all three.
    pub backend: String,
    /// When true, enables Metal dispatch tracing and structural counters
    /// (buffers/commits per token). Mirrors `--trace-dispatch` CLI flag.
    pub trace_dispatch: bool,
}

#[derive(Debug, Serialize)]
pub struct BenchReport {
    pub model_id: String,
    pub suite: String,
    pub trials: usize,
    pub max_new_tokens: usize,
    pub kernel_profile: Option<String>,
    pub speculate_mode: String,
    pub verify_window: usize,
    pub results: serde_json::Value,
}

pub fn run(opts: BenchOptions) -> Result<()> {
    use suites::*;

    let res: serde_json::Value = match opts.suite.as_str() {
        "decode-tps" | "decode" => decode::run(&opts)?,
        "prefill-tps" | "prefill" => prefill::run(&opts)?,
        "throughput-vs-batch" | "throughput" => throughput::run(&opts)?,
        "bandwidth-utilization" | "bandwidth" => bandwidth::run(&opts)?,
        "competitive" => competitive::run(&opts)?,
        // Deprecated alias from the pre-2026-04-audit naming. Kept for
        // one release; will be removed in v0.2.
        "wax-vs-llama-cpp" | "wax" => {
            eprintln!(
                "[hawking-bench] suite name `wax` is deprecated; use `competitive`. \
                       (Renamed after the 2026-04 competitive audit; see ROADMAP.md.)"
            );
            competitive::run(&opts)?
        }
        "all" => serde_json::json!({
            "decode":      decode::run(&opts).unwrap_or_default(),
            "prefill":     prefill::run(&opts).unwrap_or_default(),
            "throughput":  throughput::run(&opts).unwrap_or_default(),
            "bandwidth":   bandwidth::run(&opts).unwrap_or_default(),
            "competitive": competitive::run(&opts).unwrap_or_default(),
        }),
        other => anyhow::bail!("unknown suite `{other}`"),
    };

    let report = BenchReport {
        model_id: opts.model_id.clone(),
        suite: opts.suite.clone(),
        trials: opts.trials,
        max_new_tokens: opts.max_new_tokens,
        kernel_profile: opts
            .kernel_profile
            .as_ref()
            .map(|p| p.display().to_string()),
        speculate_mode: opts.speculate_mode.clone(),
        verify_window: opts.verify_window,
        results: res,
    };
    let s = serde_json::to_string_pretty(&report)?;
    if let Some(trace_path) = &opts.trace_json {
        let report_hash = short_hash(s.as_bytes());
        let decode_tps = report
            .results
            .get("decode_tps")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Extract dispatch samples from the decode suite result (only present
        // when HAWKING_TRACE_DISPATCH=1).
        let raw_samples = report.results.get("dispatch_samples");
        let dispatch_total_us = report
            .results
            .get("dispatch_total_us")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let dispatch_total_decode_us = report
            .results
            .get("dispatch_total_decode_us")
            .and_then(|v| v.as_u64())
            .unwrap_or(1);
        let dispatch_pct = if dispatch_total_decode_us > 0 {
            (dispatch_total_us as f64 / dispatch_total_decode_us as f64) * 100.0
        } else {
            0.0
        };

        let kernel_summary = if let Some(serde_json::Value::Array(samples)) = raw_samples {
            build_kernel_summary(samples)
        } else {
            serde_json::Value::Array(vec![])
        };
        let layer_summary = if let Some(serde_json::Value::Array(samples)) = raw_samples {
            build_layer_summary(samples)
        } else {
            serde_json::Value::Array(vec![])
        };
        // Only embed raw samples when present (gated by env var on the engine side).
        let dispatch_samples_value = raw_samples.cloned().unwrap_or(serde_json::Value::Null);

        let trace = serde_json::json!({
            "schema_version": 2,
            "kind": "hawking-bench-trace",
            "metric": "decode_only_tps",
            "report_hash": report_hash,
            "target_tps": 60.0,
            "target_gap_x": if decode_tps > 0.0 { 60.0 / decode_tps } else { 0.0 },
            "rss_mb_after_run": current_rss_mb(),
            "profile_path": opts.kernel_profile.as_ref().map(|p| p.display().to_string()),
            "speculate_mode": opts.speculate_mode,
            "verify_window": opts.verify_window,
            "dispatch_wall_pct_of_decode": dispatch_pct,
            "dispatch_total_us": dispatch_total_us,
            "kernel_summary": kernel_summary,
            "layer_summary": layer_summary,
            "dispatch_samples": dispatch_samples_value,
            "report": report,
        });
        std::fs::write(trace_path, serde_json::to_string_pretty(&trace)?)?;
    }
    match opts.json_out {
        Some(p) => std::fs::write(&p, s)?,
        None => println!("{s}"),
    }
    Ok(())
}

/// Aggregate per-dispatch samples into per-kernel statistics.
fn build_kernel_summary(samples: &[serde_json::Value]) -> serde_json::Value {
    use std::collections::HashMap;
    // kernel_name → Vec<wall_us>
    let mut by_kernel: HashMap<&str, Vec<u64>> = HashMap::new();
    for s in samples {
        let name = s
            .get("kernel_name")
            .and_then(|v| v.as_str())
            .unwrap_or("other");
        let us = s.get("wall_us").and_then(|v| v.as_u64()).unwrap_or(0);
        by_kernel.entry(name).or_default().push(us);
    }
    let mut rows: Vec<serde_json::Value> = by_kernel
        .iter()
        .map(|(name, times)| {
            let count = times.len() as u64;
            let total: u64 = times.iter().sum();
            let mean = total.checked_div(count).unwrap_or(0);
            let mut sorted = times.clone();
            sorted.sort_unstable();
            let p50 = sorted[sorted.len() / 2];
            let p99 = sorted[(sorted.len() * 99 / 100).min(sorted.len() - 1)];
            serde_json::json!({
                "kernel": name,
                "count": count,
                "total_us": total,
                "mean_us": mean,
                "p50_us": p50,
                "p99_us": p99,
            })
        })
        .collect();
    // Sort descending by total_us.
    rows.sort_by(|a, b| {
        let ta = a.get("total_us").and_then(|v| v.as_u64()).unwrap_or(0);
        let tb = b.get("total_us").and_then(|v| v.as_u64()).unwrap_or(0);
        tb.cmp(&ta)
    });
    serde_json::Value::Array(rows)
}

/// Aggregate per-dispatch samples into per-layer statistics.
fn build_layer_summary(samples: &[serde_json::Value]) -> serde_json::Value {
    use std::collections::HashMap;
    // layer → (total_us, HashMap<kernel, total_us>)
    let mut by_layer: HashMap<u32, (u64, HashMap<String, u64>)> = HashMap::new();
    for s in samples {
        let layer = match s.get("layer_hint").and_then(|v| v.as_u64()) {
            Some(l) => l as u32,
            None => continue, // skip non-layer dispatches (final norm, LM head)
        };
        let name = s
            .get("kernel_name")
            .and_then(|v| v.as_str())
            .unwrap_or("other");
        let us = s.get("wall_us").and_then(|v| v.as_u64()).unwrap_or(0);
        let entry = by_layer.entry(layer).or_default();
        entry.0 += us;
        *entry.1.entry(name.to_string()).or_default() += us;
    }
    let mut rows: Vec<serde_json::Value> = by_layer
        .iter()
        .map(|(layer, (total, kernels))| {
            serde_json::json!({
                "layer": layer,
                "total_us": total,
                "kernels": kernels,
            })
        })
        .collect();
    rows.sort_by_key(|r| r.get("layer").and_then(|v| v.as_u64()).unwrap_or(u64::MAX));
    serde_json::Value::Array(rows)
}

fn short_hash(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    h.finalize()
        .iter()
        .take(12)
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn current_rss_mb() -> Option<f64> {
    let pid = std::process::id().to_string();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
        .output()
        .ok()?;
    let kb = String::from_utf8(out.stdout)
        .ok()?
        .trim()
        .parse::<f64>()
        .ok()?;
    Some(kb / 1024.0)
}
