//! Independent verification for high-risk conclusion promotion.
//!
//! Law: **no agent promotes its own high-risk conclusion.** Promotion needs
//! independent verification. Consensus is weak evidence; a reproduced defect
//! outranks votes.

use serde::{Deserialize, Serialize};

use crate::lenses::agent::AgentId;
use crate::lenses::error::{Result, YouError};
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::roles::AgentRole;

/// Risk class of a conclusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConclusionRisk {
    Low,
    High,
}

/// A swarm conclusion awaiting (or denied) promotion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Conclusion {
    pub id: String,
    pub text: String,
    /// Authoring agent — cannot self-promote when risk is High.
    pub author_agent_id: AgentId,
    pub author_role: AgentRole,
    pub risk: ConclusionRisk,
    pub evidence_tier: EvidenceTier,
}

/// Evidence presented to the promotion board.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum PromotionEvidence {
    /// Votes from other agents. Weak; never sufficient alone for high-risk.
    Consensus { tally: VoteTally },
    /// Distinct verifier agent confirms.
    IndependentVerification {
        verifier_agent_id: AgentId,
        verifier_role: AgentRole,
        note: String,
    },
    /// A defect/oracle was reproduced. Strongest common path.
    Reproduction {
        reproducer_agent_id: AgentId,
        detail: String,
    },
}

/// Vote counts (weak evidence).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct VoteTally {
    pub for_promotion: u32,
    pub against: u32,
    pub abstain: u32,
}

impl VoteTally {
    pub fn majority_for(&self) -> bool {
        self.for_promotion > self.against && self.for_promotion > 0
    }
}

/// Outcome of a promotion attempt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum PromotionDecision {
    Promoted {
        conclusion_id: String,
        basis: String,
        evidence_tier: EvidenceTier,
    },
    Refused {
        conclusion_id: String,
        reason: String,
    },
}

/// Board that enforces independent verification for high-risk promotions.
#[derive(Debug, Clone, Default)]
pub struct PromotionBoard {
    pub decisions: Vec<PromotionDecision>,
}

impl PromotionBoard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Attempt to promote `conclusion` with the given evidence.
    ///
    /// Rules:
    /// 1. Low risk may promote on cited+ evidence without independent verifier.
    /// 2. High risk requires independent verification or reproduction.
    /// 3. Author cannot be the sole verifier (self-promotion banned).
    /// 4. Consensus alone never promotes high-risk, even with unanimous votes.
    /// 5. Reproduction outranks consensus when both are present.
    pub fn try_promote(
        &mut self,
        conclusion: &Conclusion,
        evidence: &[PromotionEvidence],
    ) -> Result<PromotionDecision> {
        let decision = match conclusion.risk {
            ConclusionRisk::Low => self.promote_low(conclusion, evidence),
            ConclusionRisk::High => self.promote_high(conclusion, evidence),
        };
        self.decisions.push(decision.clone());
        match &decision {
            PromotionDecision::Promoted { .. } => Ok(decision),
            PromotionDecision::Refused { reason, .. } => {
                Err(YouError::PromotionRefused(reason.clone()))
            }
        }
    }

    fn promote_low(
        &self,
        conclusion: &Conclusion,
        evidence: &[PromotionEvidence],
    ) -> PromotionDecision {
        if evidence.is_empty() && conclusion.evidence_tier < EvidenceTier::Cited {
            return PromotionDecision::Refused {
                conclusion_id: conclusion.id.clone(),
                reason: "low-risk promotion still needs cited+ tier or supporting evidence".into(),
            };
        }
        PromotionDecision::Promoted {
            conclusion_id: conclusion.id.clone(),
            basis: "low_risk".into(),
            evidence_tier: conclusion.evidence_tier,
        }
    }

    fn promote_high(
        &self,
        conclusion: &Conclusion,
        evidence: &[PromotionEvidence],
    ) -> PromotionDecision {
        // Scan for forbidden self-promotion first.
        for e in evidence {
            if let PromotionEvidence::IndependentVerification {
                verifier_agent_id, ..
            } = e
            {
                if verifier_agent_id == &conclusion.author_agent_id {
                    return PromotionDecision::Refused {
                        conclusion_id: conclusion.id.clone(),
                        reason: format!(
                            "agent {} cannot promote its own high-risk conclusion",
                            conclusion.author_agent_id
                        ),
                    };
                }
            }
            if let PromotionEvidence::Reproduction {
                reproducer_agent_id,
                ..
            } = e
            {
                if reproducer_agent_id == &conclusion.author_agent_id {
                    return PromotionDecision::Refused {
                        conclusion_id: conclusion.id.clone(),
                        reason: format!(
                            "agent {} cannot self-reproduce to promote its own high-risk conclusion",
                            conclusion.author_agent_id
                        ),
                    };
                }
            }
        }

        // Reproduction outranks everything else when present and independent.
        if let Some(PromotionEvidence::Reproduction { detail, .. }) = evidence.iter().find(|e| {
            matches!(e, PromotionEvidence::Reproduction { reproducer_agent_id, .. }
                if reproducer_agent_id != &conclusion.author_agent_id)
        }) {
            return PromotionDecision::Promoted {
                conclusion_id: conclusion.id.clone(),
                basis: format!("reproduction:{detail}"),
                evidence_tier: EvidenceTier::Reproduced,
            };
        }

        // Independent verification by a Verifier (or any non-author agent).
        if let Some(PromotionEvidence::IndependentVerification {
            verifier_agent_id,
            verifier_role,
            note,
        }) = evidence.iter().find(|e| {
            matches!(e, PromotionEvidence::IndependentVerification { verifier_agent_id, .. }
                if verifier_agent_id != &conclusion.author_agent_id)
        }) {
            // Prefer Verifier role but any distinct agent counts as independent.
            let _ = verifier_role;
            return PromotionDecision::Promoted {
                conclusion_id: conclusion.id.clone(),
                basis: format!("independent_verification:{verifier_agent_id}:{note}"),
                evidence_tier: EvidenceTier::IndependentlyVerified,
            };
        }

        // Consensus alone is weak — refuse high-risk even with majority.
        if evidence
            .iter()
            .any(|e| matches!(e, PromotionEvidence::Consensus { tally } if tally.majority_for()))
        {
            return PromotionDecision::Refused {
                conclusion_id: conclusion.id.clone(),
                reason: "consensus is weak evidence; high-risk promotion requires independent \
                         verification or reproduction"
                    .into(),
            };
        }

        PromotionDecision::Refused {
            conclusion_id: conclusion.id.clone(),
            reason: "high-risk conclusion lacks independent verification or reproduction".into(),
        }
    }
}
