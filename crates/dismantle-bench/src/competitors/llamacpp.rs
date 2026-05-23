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
        let binary = std::env::var("DISMANTLE_LLAMA_CLI").unwrap_or_else(|_| "llama-cli".into());
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
        let ttft_ms = stderr
            .lines()
            .find(|l| l.contains("prompt eval time"))
            .and_then(|l| extract_after(l, "="));

        Ok(Measurement {
            decode_tps,
            prefill_tps,
            ttft_ms,
            peak_rss_mb: None, // llama-cli doesn't report; could sample externally
            output: stdout.into_owned(),
        })
    }
}
