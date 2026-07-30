//! Evidence tiers for claims carried in capsules and swarm conclusions.
//!
//! Higher tiers outrank lower ones. Consensus (votes) is not a tier; a
//! reproduced defect is stronger evidence than agreement among agents.

use serde::{Deserialize, Serialize};

/// How strongly a claim is supported. Ordered by authority (weakest first).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceTier {
    /// Bare assertion with no backing.
    Asserted,
    /// Multiple agents agreed; weak — never sufficient alone for high-risk promotion.
    Consensus,
    /// Grounded in cited sources or prior research with quality grades.
    Cited,
    /// Independently checked by a distinct verifier agent.
    IndependentlyVerified,
    /// A defect or acceptance condition was reproduced (oracles, tests, fixtures).
    Reproduced,
}

impl EvidenceTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Asserted => "asserted",
            Self::Consensus => "consensus",
            Self::Cited => "cited",
            Self::IndependentlyVerified => "independently_verified",
            Self::Reproduced => "reproduced",
        }
    }

    /// True if this tier is strong enough to support promoting a high-risk
    /// conclusion (independent verification or reproduction).
    pub fn supports_high_risk_promotion(self) -> bool {
        matches!(self, Self::IndependentlyVerified | Self::Reproduced)
    }

    /// Consensus is deliberately weaker than reproduction.
    pub fn outranked_by_reproduction(self) -> bool {
        self < Self::Reproduced
    }
}

impl std::fmt::Display for EvidenceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
