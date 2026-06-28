//! Confidence signals and self-consistency voting (ch.06 §4.7).
//!
//! The exact per-token logit signals (token confidence, entropy, self-certainty)
//! are a gated runtime readback ([RUNTIME-SIDE — LATER]). What works **today**,
//! with no runtime hook, is **self-consistency voting**: sample `k` completions
//! (cheap profile), normalize, cluster identical answers, and score agreement.
//! That is the [SHELL-TODAY] escalation gate the cascade (`escalation.rs`)
//! consumes.
//!
//! This module is pure: it operates on already-collected sample strings, so it
//! is fully testable without a runtime.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The result of voting over k samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VoteResult {
    /// The plurality answer (normalized form), or `None` if there were no samples.
    pub winner: Option<String>,
    /// One *original* (un-normalized) sample text that maps to the winner.
    pub winner_original: Option<String>,
    /// Fraction of samples agreeing with the winner, in [0,1].
    pub agreement: f32,
    /// Number of distinct normalized answers.
    pub distinct: usize,
    /// Total samples considered.
    pub total: usize,
    /// Per-answer tallies (normalized answer → count), sorted desc by count.
    pub tallies: Vec<(String, usize)>,
}

impl VoteResult {
    /// A scalar confidence in [0,1] derived from agreement and answer spread.
    /// Agreement dominates; a long tail of distinct alternatives discounts it.
    pub fn confidence(&self) -> f32 {
        if self.total == 0 {
            return 0.0;
        }
        // Penalize fragmentation: with d distinct answers over n samples, the
        // "spread factor" shrinks confidence when many singletons appear.
        let spread = if self.distinct <= 1 {
            1.0
        } else {
            1.0 - ((self.distinct - 1) as f32 / self.total as f32) * 0.5
        };
        (self.agreement * spread).clamp(0.0, 1.0)
    }
}

/// How to normalize a sample before clustering. Choosing the right normalizer
/// is what makes voting work for a given task (a tool-call JSON should be
/// canonicalized; a classifier label just trimmed/lowercased).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnswerNormalizer {
    /// Trim only.
    Trimmed,
    /// Trim + lowercase (classifier labels, yes/no).
    CaseFold,
    /// Parse as JSON and re-serialize with sorted keys (tool-calls, structured
    /// single-value answers). Falls back to trimmed text if not JSON.
    CanonicalJson,
}

impl AnswerNormalizer {
    pub fn normalize(&self, sample: &str) -> String {
        match self {
            AnswerNormalizer::Trimmed => sample.trim().to_string(),
            AnswerNormalizer::CaseFold => sample.trim().to_lowercase(),
            AnswerNormalizer::CanonicalJson => canonical_json(sample.trim())
                .unwrap_or_else(|| sample.trim().to_string()),
        }
    }
}

/// Re-serialize a JSON document with object keys sorted (recursively), so
/// `{"a":1,"b":2}` and `{"b":2,"a":1}` cluster together. `None` if not JSON.
fn canonical_json(s: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(s).ok()?;
    Some(canonicalize_value(&value).to_string())
}

fn canonicalize_value(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut sorted = serde_json::Map::new();
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            for k in keys {
                sorted.insert(k.clone(), canonicalize_value(&map[k]));
            }
            serde_json::Value::Object(sorted)
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize_value).collect())
        }
        other => other.clone(),
    }
}

/// Vote over a set of samples using the given normalizer.
pub fn self_consistency_vote(samples: &[String], normalizer: AnswerNormalizer) -> VoteResult {
    let total = samples.len();
    if total == 0 {
        return VoteResult {
            winner: None,
            winner_original: None,
            agreement: 0.0,
            distinct: 0,
            total: 0,
            tallies: Vec::new(),
        };
    }
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut first_original: HashMap<String, String> = HashMap::new();
    for s in samples {
        let key = normalizer.normalize(s);
        *counts.entry(key.clone()).or_insert(0) += 1;
        first_original.entry(key).or_insert_with(|| s.clone());
    }
    let mut tallies: Vec<(String, usize)> = counts.into_iter().collect();
    // Sort by count desc, then by answer asc for determinism on ties.
    tallies.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let distinct = tallies.len();
    let (winner, winner_count) = tallies
        .first()
        .map(|(k, c)| (Some(k.clone()), *c))
        .unwrap_or((None, 0));
    let winner_original = winner
        .as_ref()
        .and_then(|w| first_original.get(w).cloned());
    let agreement = winner_count as f32 / total as f32;
    VoteResult {
        winner,
        winner_original,
        agreement,
        distinct,
        total,
        tallies,
    }
}

/// Shannon entropy (nats) of a probability distribution, ignoring zeros.
/// Exposed so a future runtime logprob path can reuse the same metric for
/// first-token entropy (§4.7).
pub fn entropy(probs: &[f32]) -> f32 {
    probs
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// Self-certainty: divergence of a distribution from uniform, normalized to
/// [0,1] (0 = uniform / maximally uncertain, 1 = a point mass). This is the
/// §4.7 "separates correct from incorrect better than perplexity" signal,
/// computed from a distribution once the runtime exposes one.
pub fn self_certainty(probs: &[f32]) -> f32 {
    let n = probs.len();
    if n <= 1 {
        return 1.0;
    }
    let max_entropy = (n as f32).ln();
    let h = entropy(probs);
    (1.0 - h / max_entropy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unanimous_vote_is_high_confidence() {
        let samples = vec!["yes".into(), "yes".into(), "yes".into()];
        let r = self_consistency_vote(&samples, AnswerNormalizer::CaseFold);
        assert_eq!(r.winner.as_deref(), Some("yes"));
        assert_eq!(r.agreement, 1.0);
        assert_eq!(r.distinct, 1);
        assert!(r.confidence() > 0.95);
    }

    #[test]
    fn split_vote_is_low_confidence() {
        let samples = vec!["a".into(), "b".into(), "c".into()];
        let r = self_consistency_vote(&samples, AnswerNormalizer::Trimmed);
        assert_eq!(r.distinct, 3);
        assert!((r.agreement - 1.0 / 3.0).abs() < 1e-6);
        assert!(r.confidence() < 0.4);
    }

    #[test]
    fn json_clusters_regardless_of_key_order() {
        let samples = vec![
            "{\"name\":\"edit\",\"path\":\"a\"}".into(),
            "{\"path\":\"a\",\"name\":\"edit\"}".into(),
            "{\"name\":\"read\"}".into(),
        ];
        let r = self_consistency_vote(&samples, AnswerNormalizer::CanonicalJson);
        // The two reorderings cluster → winner has 2 of 3.
        assert_eq!(r.distinct, 2);
        assert!((r.agreement - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn entropy_and_self_certainty() {
        // Uniform over 2 → entropy ln(2), self-certainty 0.
        let uniform = [0.5, 0.5];
        assert!((entropy(&uniform) - 2.0_f32.ln()).abs() < 1e-5);
        assert!(self_certainty(&uniform) < 1e-5);
        // Point mass → self-certainty 1.
        let point = [1.0, 0.0];
        assert!((self_certainty(&point) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn empty_samples_are_zero_confidence() {
        let r = self_consistency_vote(&[], AnswerNormalizer::Trimmed);
        assert_eq!(r.total, 0);
        assert_eq!(r.confidence(), 0.0);
        assert!(r.winner.is_none());
    }
}
