//! Probabilistic oracles — fallback & tie-break only (bible ch.02 §4.6.3).
//!
//! These run ONLY when no deterministic oracle applies (e.g. a `synthesize`
//! step with no buildable artifact). They never override `build`/`test`.
//!
//! * [`ConsistencyOracle`] — self-consistency vote over K samples (§3.1). Cheap,
//!   local, surprisingly strong; the majority/centroid is the verdict.
//! * [`LlmJudgeOracle`] — the model critiques the candidate against the step's
//!   predicate. Strictly gated: used only as the last resort.

use crate::runtime_client::KernelRuntimeClient;
use crate::verify::oracle::{Cost, Oracle, OracleClass, Verdict, VerdictStatus, VerificationInput};
use futures::future::BoxFuture;
use hide_core::runtime::{InferenceRequest, StreamChunk};
use hide_core::Result;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Self-consistency vote (§4.6.3). Samples the model `k` times for a short
/// yes/no judgement against the step predicate and takes the majority. The
/// score is the agreement fraction.
pub struct ConsistencyOracle {
    runtime: Arc<KernelRuntimeClient>,
    k: u8,
    predicate: String,
}

impl ConsistencyOracle {
    pub fn new(runtime: Arc<KernelRuntimeClient>, k: u8, predicate: impl Into<String>) -> Self {
        Self {
            runtime,
            k: k.max(1),
            predicate: predicate.into(),
        }
    }

    async fn sample_once(&self, candidate: &str) -> Result<bool> {
        let prompt = format!(
            "You are a strict verifier. Does the following candidate satisfy this requirement?\n\
             Requirement: {}\n\nCandidate:\n{}\n\nAnswer YES or NO only.",
            self.predicate, candidate
        );
        let request = InferenceRequest {
            task_kind: "verify".to_string(),
            prompt,
            messages: Vec::new(),
            max_output_tokens: 4,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: BTreeMap::new(),
        };
        let mut buf = String::new();
        let mut sink = |chunk: StreamChunk| {
            if let StreamChunk::Token { text, .. } = chunk {
                buf.push_str(&text);
            }
            Ok(())
        };
        self.runtime.generate(request, &mut sink).await?;
        let answer = buf.trim().to_ascii_lowercase();
        Ok(answer.starts_with("yes") || answer.starts_with('y') || answer.contains("yes"))
    }
}

impl Oracle for ConsistencyOracle {
    fn name(&self) -> &str {
        "consistency"
    }
    fn class(&self) -> OracleClass {
        OracleClass::Probabilistic
    }
    fn cost_hint(&self) -> Cost {
        Cost::Medium
    }
    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            let mut yes = 0u32;
            for _ in 0..self.k {
                if self.sample_once(&input.candidate_output).await? {
                    yes += 1;
                }
            }
            let score = yes as f32 / self.k as f32;
            let status = if score > 0.5 {
                VerdictStatus::Pass
            } else if yes == 0 {
                VerdictStatus::Fail
            } else {
                VerdictStatus::Inconclusive
            };
            let mut v = Verdict::pass(
                "consistency",
                OracleClass::Probabilistic,
                format!("{yes}/{} votes yes", self.k),
            );
            v.status = status;
            v.score = score;
            Ok(v)
        })
    }
}

/// LLM-as-judge (§4.6.3) — the strictly-fallback critic. One critique; the score
/// is parsed from a leading `0.0..=1.0`. Never overrides a deterministic verdict
/// (the gate enforces that by class).
pub struct LlmJudgeOracle {
    runtime: Arc<KernelRuntimeClient>,
    predicate: String,
}

impl LlmJudgeOracle {
    pub fn new(runtime: Arc<KernelRuntimeClient>, predicate: impl Into<String>) -> Self {
        Self {
            runtime,
            predicate: predicate.into(),
        }
    }
}

impl Oracle for LlmJudgeOracle {
    fn name(&self) -> &str {
        "llm_judge"
    }
    fn class(&self) -> OracleClass {
        OracleClass::Probabilistic
    }
    fn cost_hint(&self) -> Cost {
        Cost::Medium
    }
    fn verify<'a>(&'a self, input: &'a VerificationInput) -> BoxFuture<'a, Result<Verdict>> {
        Box::pin(async move {
            let prompt = format!(
                "Rate from 0.0 to 1.0 how well the candidate meets the requirement. \
                 Output the number first.\nRequirement: {}\n\nCandidate:\n{}",
                self.predicate, input.candidate_output
            );
            let request = InferenceRequest {
                task_kind: "verify".to_string(),
                prompt,
                messages: Vec::new(),
                max_output_tokens: 8,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: BTreeMap::new(),
            };
            let mut buf = String::new();
            let mut sink = |chunk: StreamChunk| {
                if let StreamChunk::Token { text, .. } = chunk {
                    buf.push_str(&text);
                }
                Ok(())
            };
            self.runtime.generate(request, &mut sink).await?;
            let score = parse_leading_float(&buf).unwrap_or(0.0);
            let mut v = Verdict::pass(
                "llm_judge",
                OracleClass::Probabilistic,
                format!("judge score {score:.2}"),
            );
            v.score = score;
            v.status = if score >= 0.5 {
                VerdictStatus::Pass
            } else {
                VerdictStatus::Fail
            };
            Ok(v)
        })
    }
}

fn parse_leading_float(s: &str) -> Option<f32> {
    let t = s.trim();
    let mut end = 0;
    for (i, c) in t.char_indices() {
        if c.is_ascii_digit() || c == '.' {
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    t.get(..end).and_then(|p| p.parse::<f32>().ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hawking_orch::inference::StubInferenceClient;
    use hawking_orch::registry::RoleRegistry;
    use hawking_orch::router::SimpleRouter;

    fn runtime(response: &str) -> Arc<KernelRuntimeClient> {
        let registry = Arc::new(RoleRegistry::with_default_local_roles());
        let router = Arc::new(SimpleRouter::new(registry));
        let inference = Arc::new(StubInferenceClient::new(response));
        Arc::new(KernelRuntimeClient::new(router, inference))
    }

    #[test]
    fn parses_leading_float() {
        assert_eq!(parse_leading_float("0.83 because ..."), Some(0.83));
        assert_eq!(parse_leading_float("1.0"), Some(1.0));
    }

    #[tokio::test]
    async fn consistency_unanimous_yes_passes() {
        let oracle = ConsistencyOracle::new(runtime("YES"), 3, "does the thing");
        let mut input = VerificationInput::new("/tmp");
        input.candidate_output = "the thing".to_string();
        let v = oracle.verify(&input).await.unwrap();
        assert_eq!(v.status, VerdictStatus::Pass);
        assert_eq!(v.score, 1.0);
        assert_eq!(v.class, OracleClass::Probabilistic);
    }

    #[tokio::test]
    async fn judge_low_score_fails() {
        let oracle = LlmJudgeOracle::new(runtime("0.2 not great"), "be great");
        let v = oracle
            .verify(&VerificationInput::new("/tmp"))
            .await
            .unwrap();
        assert_eq!(v.status, VerdictStatus::Fail);
    }
}
