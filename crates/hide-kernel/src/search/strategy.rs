//! Search & sampling-scale strategies (bible ch.02 §4.8) — where free local
//! compute becomes reliability (K4).
//!
//! The centerpiece is [`best_of_n`]: fork N candidate attempts, verify each with
//! the step's deterministic oracles, and keep the oracle-passing candidate with
//! the best tie-break score. [`pick_tier`] escalates React → BestOfN → ToT by
//! difficulty.

use crate::runtime_client::KernelRuntimeClient;
use crate::verify::oracle::{Verdict, VerdictStatus};
use crate::verify::{OracleSuite, VerificationInput};
use futures::future::BoxFuture;
use hide_core::runtime::{InferenceRequest, StreamChunk};
use hide_core::Result;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    pub summary: String,
    /// The candidate's raw produced output (diff / text).
    #[serde(default)]
    pub output: String,
    pub score: f32,
    pub verdicts: Vec<Verdict>,
}

impl Candidate {
    /// Oracle-first score: a candidate that passes all deterministic oracles
    /// outranks any that fails one, regardless of probabilistic score (§4.8.2).
    pub fn rank_key(&self) -> (u8, f32) {
        let det_ok = self
            .verdicts
            .iter()
            .filter(|v| v.is_deterministic())
            .all(|v| v.status != VerdictStatus::Fail)
            && self.verdicts.iter().any(|v| v.is_deterministic());
        let any_fail = self.verdicts.iter().any(|v| v.status == VerdictStatus::Fail);
        let tier = if det_ok {
            2
        } else if !any_fail {
            1
        } else {
            0
        };
        (tier, self.score)
    }
}

pub trait SearchStrategy: Send + Sync {
    fn name(&self) -> &str;
    fn generate<'a>(&'a self, prompt: &'a str) -> BoxFuture<'a, Result<Vec<Candidate>>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EscalationLadder {
    pub tiers: Vec<SearchTier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchTier {
    React,
    BestOfN,
    TreeOfThoughts,
    Lats,
    Debate,
}

/// Pick the search tier for a step from its difficulty + whether it has a
/// deterministic oracle (§4.8 escalation ladder). A hard step *with* an oracle
/// to rank candidates escalates to best-of-N; a very hard branchy step goes to
/// ToT; otherwise ReAct (the cheap default).
pub fn pick_tier(difficulty: f32, has_deterministic_oracle: bool, breadth: u8) -> SearchTier {
    if breadth <= 1 {
        return SearchTier::React;
    }
    if difficulty > 0.85 {
        SearchTier::TreeOfThoughts
    } else if difficulty > 0.5 && has_deterministic_oracle {
        SearchTier::BestOfN
    } else {
        SearchTier::React
    }
}

/// Best-of-N (Tier 2, the workhorse). Generate `n` candidate outputs for the
/// prompt, verify each with the step's oracles, and return them sorted best-first
/// by oracle-first score. The selected candidate is `result[0]` (if any).
///
/// Isolation: candidates are generated and scored in-memory here. When a worktree
/// is available the caller can route each candidate's effects through an isolated
/// `git.worktree.*` (the dispatcher seam); the scoring contract is identical.
pub async fn best_of_n(
    runtime: &KernelRuntimeClient,
    suite: &OracleSuite,
    oracle_ids: &[String],
    prompt: &str,
    base_input: &VerificationInput,
    n: u8,
) -> Result<Vec<Candidate>> {
    let n = n.max(1);
    let mut candidates = Vec::with_capacity(n as usize);
    for i in 0..n {
        let output = generate_once(runtime, prompt).await?;
        let mut input = base_input.clone();
        input.candidate_output = output.clone();
        let verdicts = suite.run(oracle_ids, &input).await?;
        // tie-break score = max probabilistic score, else 1.0 if all det pass.
        let prob_score = verdicts
            .iter()
            .filter(|v| !v.is_deterministic())
            .map(|v| v.score)
            .fold(0.0_f32, f32::max);
        let det_pass = verdicts
            .iter()
            .filter(|v| v.is_deterministic())
            .all(|v| v.status != VerdictStatus::Fail);
        let score = if prob_score > 0.0 {
            prob_score
        } else if det_pass {
            1.0
        } else {
            0.0
        };
        candidates.push(Candidate {
            id: format!("cand-{i}"),
            summary: output.chars().take(80).collect(),
            output,
            score,
            verdicts,
        });
    }
    candidates.sort_by(|a, b| b.rank_key().partial_cmp(&a.rank_key()).unwrap());
    Ok(candidates)
}

async fn generate_once(runtime: &KernelRuntimeClient, prompt: &str) -> Result<String> {
    let request = InferenceRequest {
        task_kind: "code".to_string(),
        prompt: prompt.to_string(),
        messages: Vec::new(),
        max_output_tokens: 512,
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
    runtime.generate(request, &mut sink).await?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_tier_defaults_to_react_at_breadth_one() {
        assert_eq!(pick_tier(0.99, true, 1), SearchTier::React);
    }

    #[test]
    fn pick_tier_escalates_with_oracle_and_breadth() {
        assert_eq!(pick_tier(0.6, true, 4), SearchTier::BestOfN);
        assert_eq!(pick_tier(0.9, true, 4), SearchTier::TreeOfThoughts);
        assert_eq!(pick_tier(0.6, false, 4), SearchTier::React);
    }

    #[test]
    fn candidate_rank_prefers_oracle_pass() {
        use crate::verify::oracle::OracleClass;
        let pass = Candidate {
            id: "a".into(),
            summary: String::new(),
            output: String::new(),
            score: 0.1,
            verdicts: vec![Verdict::pass("build", OracleClass::Deterministic, "ok")],
        };
        let fail = Candidate {
            id: "b".into(),
            summary: String::new(),
            output: String::new(),
            score: 0.99,
            verdicts: vec![Verdict::fail("build", OracleClass::Deterministic, "no", vec![])],
        };
        assert!(pass.rank_key() > fail.rank_key());
    }
}
