//! `hawking-eval` — a local, $0 coding-capability harness for HIDE.
//!
//! It drives a model through the existing OpenAI-compatible serve path
//! (`POST /v1/chat/completions`) and scores deterministic tasks, reporting
//! pass@1 with an honest Wilson confidence interval. At N=100 the Wilson
//! half-width is ~10pp, so only models whose intervals do NOT overlap are
//! distinguishable — the report carries the interval so we never over-claim a
//! sub-10pp difference (the H3 "build hawking-eval first" gate).
//!
//! The scoring + statistics are pure and unit-tested; the network client sits
//! behind [`CompletionClient`], so the runner is exercised with a deterministic
//! mock in CI and against a real server when one is running. This crate makes
//! NO engine change — it only consumes the serve path.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Standard-normal quantile for a 95% interval.
pub const Z_95: f64 = 1.96;

#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    #[error("http: {0}")]
    Http(String),
    #[error("decode: {0}")]
    Decode(String),
}

/// A single deterministic task: the model passes iff every `expect` substring
/// appears in its (greedy) completion. Crude but reproducible — the point is a
/// stable, contamination-auditable signal, not semantic grading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub prompt: String,
    pub expect: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskOutcome {
    pub id: String,
    pub passed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    pub outcomes: Vec<TaskOutcome>,
    pub passes: usize,
    pub total: usize,
    pub pass_at_1: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// Score one completion against a task (deterministic): all `expect` substrings
/// must be present.
pub fn score(task: &Task, completion: &str) -> bool {
    task.expect.iter().all(|e| completion.contains(e.as_str()))
}

/// Wilson score interval for `passes`/`n` at quantile `z`. Returns `(lo, hi)`
/// clamped to `[0,1]`; `n == 0` yields the uninformative `(0, 1)`.
pub fn wilson_interval(passes: usize, n: usize, z: f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 1.0);
    }
    let n = n as f64;
    let p = passes as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (p + z2 / (2.0 * n)) / denom;
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denom;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

/// Negative log-likelihood of the `target` token under a logit distribution --
/// the M4 continuous co-metric (W-F1-nll), ~5x lower variance than discrete
/// pass@1 at small N. Numerically stable via log-sum-exp. Pure; the engine
/// logit-emit seam that feeds it (`StreamEvent` carries no logits today) is the
/// model-gated part. Returns `+inf` for an empty distribution or out-of-range
/// target.
pub fn nll_from_logits(logits: &[f32], target: usize) -> f32 {
    if logits.is_empty() || target >= logits.len() {
        return f32::INFINITY;
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sumexp: f32 = logits.iter().map(|&l| (l - max).exp()).sum();
    let log_z = max + sumexp.ln();
    log_z - logits[target]
}

/// A source of greedy completions (the serve path, or a mock in tests).
#[async_trait]
pub trait CompletionClient {
    async fn complete(&self, prompt: &str) -> Result<String, EvalError>;
}

/// Run every task once (greedy) and produce a report with pass@1 + Wilson CI.
pub async fn run_suite<C: CompletionClient + Sync>(
    client: &C,
    tasks: &[Task],
) -> Result<EvalReport, EvalError> {
    let mut outcomes = Vec::with_capacity(tasks.len());
    for t in tasks {
        let out = client.complete(&t.prompt).await?;
        outcomes.push(TaskOutcome {
            id: t.id.clone(),
            passed: score(t, &out),
        });
    }
    let passes = outcomes.iter().filter(|o| o.passed).count();
    let total = outcomes.len();
    let (ci_low, ci_high) = wilson_interval(passes, total, Z_95);
    let pass_at_1 = if total == 0 {
        0.0
    } else {
        passes as f64 / total as f64
    };
    Ok(EvalReport {
        outcomes,
        passes,
        total,
        pass_at_1,
        ci_low,
        ci_high,
    })
}

/// Load eval tasks from JSONL text (one `Task` JSON object per line) -- the
/// non-gated loader for benchmark corpora (HumanEval+/BCB-Hard/Aider). Blank
/// lines are skipped; a malformed line errors with its 1-based number. The
/// corpus FILES themselves (and the model run over them) are the gated part;
/// this parser feeds [`run_suite`] once a file is present.
pub fn load_tasks_jsonl(text: &str) -> Result<Vec<Task>, EvalError> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let task: Task = serde_json::from_str(line)
            .map_err(|e| EvalError::Decode(format!("line {}: {e}", i + 1)))?;
        out.push(task);
    }
    Ok(out)
}

/// OpenAI-compatible client for the local serve path. Greedy decode
/// (temperature 0) so verdicts are reproducible. Built on the raw body/string
/// path so it does not depend on any optional reqwest feature.
pub struct OpenAiClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
}

impl OpenAiClient {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            base_url: base_url.into(),
            model: model.into(),
        }
    }
}

#[async_trait]
impl CompletionClient for OpenAiClient {
    async fn complete(&self, prompt: &str) -> Result<String, EvalError> {
        let url = format!(
            "{}/v1/chat/completions",
            self.base_url.trim_end_matches('/')
        );
        let body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": prompt }],
            "temperature": 0.0,
            "stream": false,
        });
        let body_str =
            serde_json::to_string(&body).map_err(|e| EvalError::Decode(e.to_string()))?;
        let resp = self
            .http
            .post(&url)
            .header("content-type", "application/json")
            .body(body_str)
            .send()
            .await
            .map_err(|e| EvalError::Http(e.to_string()))?;
        let text = resp
            .text()
            .await
            .map_err(|e| EvalError::Http(e.to_string()))?;
        let v: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| EvalError::Decode(e.to_string()))?;
        Ok(v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    struct Mock(HashMap<String, String>);

    #[async_trait]
    impl CompletionClient for Mock {
        async fn complete(&self, prompt: &str) -> Result<String, EvalError> {
            Ok(self.0.get(prompt).cloned().unwrap_or_default())
        }
    }

    fn tasks() -> Vec<Task> {
        vec![
            Task {
                id: "a".into(),
                prompt: "pa".into(),
                expect: vec!["foo".into()],
            },
            Task {
                id: "b".into(),
                prompt: "pb".into(),
                expect: vec!["bar".into()],
            },
        ]
    }

    #[test]
    fn score_requires_all_substrings() {
        let t = Task {
            id: "x".into(),
            prompt: String::new(),
            expect: vec!["a".into(), "b".into()],
        };
        assert!(score(&t, "x a y b z"));
        assert!(!score(&t, "only a"));
    }

    #[test]
    fn wilson_half_width_is_about_ten_points_at_n100() {
        let (lo, hi) = wilson_interval(50, 100, Z_95);
        let half = (hi - lo) / 2.0;
        assert!((0.09..0.10).contains(&half), "half-width {half}");
        assert_eq!(wilson_interval(0, 0, Z_95), (0.0, 1.0));
    }

    #[test]
    fn nll_rewards_confident_correct_and_punishes_wrong() {
        let logits = vec![100.0, 0.0, 0.0];
        assert!(
            nll_from_logits(&logits, 0) < 1e-3,
            "confident correct -> ~0 NLL"
        );
        assert!(
            nll_from_logits(&logits, 1) > 50.0,
            "confident wrong -> high NLL"
        );
    }

    #[test]
    fn nll_uniform_is_log_n_and_guards_bounds() {
        let logits = vec![0.0; 4];
        assert!((nll_from_logits(&logits, 2) - (4f32).ln()).abs() < 1e-5);
        assert_eq!(nll_from_logits(&[], 0), f32::INFINITY);
        assert_eq!(nll_from_logits(&[1.0], 5), f32::INFINITY);
    }

    #[test]
    fn load_tasks_jsonl_parses_and_skips_blanks() {
        let jsonl = "\n{\"id\":\"a\",\"prompt\":\"p1\",\"expect\":[\"foo\"]}\n\n{\"id\":\"b\",\"prompt\":\"p2\",\"expect\":[]}\n";
        let tasks = load_tasks_jsonl(jsonl).expect("parse");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0].id, "a");
        assert_eq!(tasks[0].expect, vec!["foo".to_string()]);
        assert!(tasks[1].expect.is_empty());
    }

    #[test]
    fn load_tasks_jsonl_reports_bad_line_number() {
        let bad = "{\"id\":\"a\",\"prompt\":\"p\",\"expect\":[]}\noops";
        let err = load_tasks_jsonl(bad).unwrap_err();
        assert!(format!("{err}").contains("line 2"), "got: {err}");
    }

    #[tokio::test]
    async fn run_suite_is_deterministic() {
        let mut m = HashMap::new();
        m.insert("pa".to_string(), "has foo".to_string());
        m.insert("pb".to_string(), "no match".to_string());
        let client = Mock(m);
        let r1 = run_suite(&client, &tasks()).await.unwrap();
        let r2 = run_suite(&client, &tasks()).await.unwrap();
        assert_eq!(r1, r2, "same inputs => identical report");
        assert_eq!(r1.passes, 1);
        assert_eq!(r1.total, 2);
        assert!((r1.pass_at_1 - 0.5).abs() < 1e-9);
        assert_eq!(
            serde_json::to_string(&r1).unwrap(),
            serde_json::to_string(&r2).unwrap(),
            "verdict JSON is byte-identical across runs"
        );
    }
}
