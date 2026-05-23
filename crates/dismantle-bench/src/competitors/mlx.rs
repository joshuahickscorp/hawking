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
        let binary =
            std::env::var("DISMANTLE_MLX_GENERATE").unwrap_or_else(|_| "mlx_lm.generate".into());
        let model_id = std::env::var("DISMANTLE_MLX_MODEL_ID")
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
        Self {
            binary,
            version,
            model_id,
        }
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

        Ok(Measurement {
            decode_tps,
            prefill_tps,
            ttft_ms: None,
            peak_rss_mb: None,
            output: body,
        })
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
