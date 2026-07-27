//! Swarm operating modes.

use serde::{Deserialize, Serialize};

/// How a swarm coordinates its agents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmMode {
    ParallelResearch,
    Debate,
    IndependentReplication,
    PlanTournament,
    CreativeDivergence,
    FactVerification,
    DocumentAssembly,
    PersonalAdministration,
}

impl SwarmMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ParallelResearch => "parallel_research",
            Self::Debate => "debate",
            Self::IndependentReplication => "independent_replication",
            Self::PlanTournament => "plan_tournament",
            Self::CreativeDivergence => "creative_divergence",
            Self::FactVerification => "fact_verification",
            Self::DocumentAssembly => "document_assembly",
            Self::PersonalAdministration => "personal_administration",
        }
    }

    pub fn all() -> &'static [SwarmMode] {
        &[
            Self::ParallelResearch,
            Self::Debate,
            Self::IndependentReplication,
            Self::PlanTournament,
            Self::CreativeDivergence,
            Self::FactVerification,
            Self::DocumentAssembly,
            Self::PersonalAdministration,
        ]
    }
}

impl std::fmt::Display for SwarmMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
