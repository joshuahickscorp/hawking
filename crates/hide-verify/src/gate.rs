//! The verification gate: the authority rule, encoded (Bible Book IX, sec 28-29).
//!
//! The gate reconciles the verdicts from every tier into one decision. Its
//! defining rule, from which everything else follows: a probabilistic review
//! (Tier4) may NEVER overrule a failing deterministic gate. This is not advice;
//! it is enforced by [`apply_gate`], which returns [`GateDecision::Reject`] on
//! any deterministic failure regardless of what the review says.

use serde::{Deserialize, Serialize};

use crate::oracle::Verdict;
use crate::tier::VerificationTier;

/// A verdict tagged with the tier and oracle that produced it. The gate consumes
/// a slice of these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TieredVerdict {
    pub tier: VerificationTier,
    pub oracle: String,
    pub verdict: Verdict,
}

impl TieredVerdict {
    pub fn new(tier: VerificationTier, oracle: impl Into<String>, verdict: Verdict) -> Self {
        Self {
            tier,
            oracle: oracle.into(),
            verdict,
        }
    }
}

/// The gate's decision.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum GateDecision {
    Accept,
    Reject { reasons: Vec<String> },
    /// Nothing decisive ran (no deterministic pass and no blocking failure), so
    /// the gate cannot accept on faith.
    Inconclusive,
}

/// Reconcile tier verdicts into one decision, honoring the authority rule.
///
/// Order of authority:
/// 1. Any DETERMINISTIC failure forces [`GateDecision::Reject`], unconditionally.
///    A probabilistic review Pass cannot rescue it.
/// 2. If no deterministic verdict passed, the gate is [`GateDecision::Inconclusive`]:
///    a review alone can never carry a change to Accept.
/// 3. With the deterministic gate passed, a review failure still blocks (Reject),
///    and only then does a clean review yield [`GateDecision::Accept`].
pub fn apply_gate(verdicts: &[TieredVerdict]) -> GateDecision {
    let det_fail: Vec<String> = verdicts
        .iter()
        .filter(|v| v.tier.is_deterministic())
        .filter_map(reason_if_fail)
        .collect();
    if !det_fail.is_empty() {
        return GateDecision::Reject { reasons: det_fail };
    }

    let any_det_pass = verdicts
        .iter()
        .any(|v| v.tier.is_deterministic() && v.verdict.is_pass());
    if !any_det_pass {
        return GateDecision::Inconclusive;
    }

    let review_fail: Vec<String> = verdicts
        .iter()
        .filter(|v| v.tier.is_probabilistic())
        .filter_map(reason_if_fail)
        .collect();
    if !review_fail.is_empty() {
        return GateDecision::Reject {
            reasons: review_fail,
        };
    }

    GateDecision::Accept
}

fn reason_if_fail(v: &TieredVerdict) -> Option<String> {
    match &v.verdict {
        Verdict::Fail { reasons } => Some(format!("{}: {}", v.oracle, reasons.join("; "))),
        _ => None,
    }
}

/// The authority invariant, encoded as a value so it can be asserted directly: a
/// probabilistic review can never override a deterministic verdict. Always
/// `false`. See [`apply_gate`], which enforces it.
pub const fn probabilistic_can_override_deterministic() -> bool {
    false
}
