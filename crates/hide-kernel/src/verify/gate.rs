use crate::verify::oracle::{Verdict, VerdictStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationGate {
    pub require_deterministic_pass: bool,
    pub min_score: f32,
}

impl Default for VerificationGate {
    fn default() -> Self {
        Self {
            require_deterministic_pass: true,
            min_score: 1.0,
        }
    }
}

impl VerificationGate {
    pub fn decide(&self, verdicts: &[Verdict]) -> GateDecision {
        if verdicts.iter().any(|v| v.status == VerdictStatus::Fail) {
            return GateDecision::Repair;
        }
        if verdicts
            .iter()
            .any(|v| v.status == VerdictStatus::Pass && v.score >= self.min_score)
        {
            GateDecision::Accept
        } else {
            GateDecision::Replan
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateDecision {
    Accept,
    Repair,
    Replan,
    Abort,
}
