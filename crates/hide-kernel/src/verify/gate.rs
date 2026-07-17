//! The Verification Gate (bible ch.02 §4.6.4).
//!
//! Decides a step's fate from its oracle verdicts. The authority rule (A.2 /
//! §3.2): **Deterministic verdicts are authoritative**; a Probabilistic score
//! only ranks *within* the deterministic-pass set and never overrides a
//! `build`/`test` failure.

use crate::verify::oracle::{OracleClass, Verdict, VerdictStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationGate {
    /// Probabilistic-fallback acceptance threshold (only consulted when no
    /// deterministic oracle applied).
    pub min_score: f32,
}

impl Default for VerificationGate {
    fn default() -> Self {
        Self { min_score: 0.7 }
    }
}

impl VerificationGate {
    pub fn with_threshold(min_score: f32) -> Self {
        Self { min_score }
    }

    /// Decide from the verdicts (§4.6.4). Deterministic first:
    /// * any deterministic Fail  → Repair
    /// * all deterministic Pass (≥1)  → Accept
    /// * no deterministic verdict → fall back to probabilistic vs `min_score`.
    pub fn decide(&self, verdicts: &[Verdict]) -> GateDecision {
        let det: Vec<&Verdict> = verdicts
            .iter()
            .filter(|v| v.class == OracleClass::Deterministic)
            .collect();

        if !det.is_empty() {
            // A deterministic oracle is authoritative.
            if det.iter().any(|v| v.status == VerdictStatus::Fail) {
                return GateDecision::Repair;
            }
            if det
                .iter()
                .all(|v| matches!(v.status, VerdictStatus::Pass | VerdictStatus::Skipped))
                && det.iter().any(|v| v.status == VerdictStatus::Pass)
            {
                return GateDecision::Accept;
            }
            // Deterministic ran but was Inconclusive across the board → consistency.
            return GateDecision::Inconclusive;
        }

        // No deterministic oracle applied — probabilistic fallback.
        let prob: Vec<&Verdict> = verdicts
            .iter()
            .filter(|v| v.class == OracleClass::Probabilistic)
            .collect();
        if prob.is_empty() {
            // Nothing ran at all → can't accept on faith (K1).
            return GateDecision::Inconclusive;
        }
        if prob.iter().any(|v| v.status == VerdictStatus::Fail) {
            return GateDecision::Repair;
        }
        let best = prob
            .iter()
            .filter(|v| v.status == VerdictStatus::Pass)
            .map(|v| v.score)
            .fold(0.0_f32, f32::max);
        if best >= self.min_score {
            GateDecision::Accept
        } else {
            GateDecision::Repair
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDecision {
    Accept,
    Repair,
    Replan,
    /// No oracle could decide — route to consistency/judge (probabilistic).
    Inconclusive,
    Abort,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify::oracle::Failure;

    fn det_pass() -> Verdict {
        Verdict::pass("build", OracleClass::Deterministic, "ok")
    }
    fn det_fail() -> Verdict {
        Verdict::fail(
            "build",
            OracleClass::Deterministic,
            "E0308",
            vec![Failure::new("type", "mismatched types")],
        )
    }
    fn prob_pass(score: f32) -> Verdict {
        let mut v = Verdict::pass("judge", OracleClass::Probabilistic, "looks good");
        v.score = score;
        v
    }

    #[test]
    fn deterministic_pass_accepts() {
        assert_eq!(
            VerificationGate::default().decide(&[det_pass()]),
            GateDecision::Accept
        );
    }

    #[test]
    fn deterministic_fail_repairs() {
        assert_eq!(
            VerificationGate::default().decide(&[det_fail()]),
            GateDecision::Repair
        );
    }

    #[test]
    fn deterministic_outranks_probabilistic() {
        // A high-scoring probabilistic PASS must NOT rescue a deterministic FAIL.
        let verdicts = vec![det_fail(), prob_pass(1.0)];
        assert_eq!(
            VerificationGate::default().decide(&verdicts),
            GateDecision::Repair
        );
    }

    #[test]
    fn probabilistic_only_uses_threshold() {
        let gate = VerificationGate::with_threshold(0.7);
        assert_eq!(gate.decide(&[prob_pass(0.9)]), GateDecision::Accept);
        assert_eq!(gate.decide(&[prob_pass(0.5)]), GateDecision::Repair);
    }

    #[test]
    fn no_oracle_is_inconclusive() {
        assert_eq!(
            VerificationGate::default().decide(&[]),
            GateDecision::Inconclusive
        );
    }
}
