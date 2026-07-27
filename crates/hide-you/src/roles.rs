//! Swarm agent roles — governed team positions, not prompt labels alone.

use serde::{Deserialize, Serialize};

/// Role each swarm agent holds. A YOU swarm is a governed team.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    Researcher,
    Planner,
    Critic,
    FactChecker,
    Writer,
    DataAnalyst,
    ImageAnalyst,
    ConnectorOperator,
    Scheduler,
    Archivist,
    Specialist,
    /// Independent check path; never the author of a conclusion it promotes.
    Verifier,
}

impl AgentRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Researcher => "researcher",
            Self::Planner => "planner",
            Self::Critic => "critic",
            Self::FactChecker => "fact_checker",
            Self::Writer => "writer",
            Self::DataAnalyst => "data_analyst",
            Self::ImageAnalyst => "image_analyst",
            Self::ConnectorOperator => "connector_operator",
            Self::Scheduler => "scheduler",
            Self::Archivist => "archivist",
            Self::Specialist => "specialist",
            Self::Verifier => "verifier",
        }
    }

    pub fn all() -> &'static [AgentRole] {
        &[
            Self::Researcher,
            Self::Planner,
            Self::Critic,
            Self::FactChecker,
            Self::Writer,
            Self::DataAnalyst,
            Self::ImageAnalyst,
            Self::ConnectorOperator,
            Self::Scheduler,
            Self::Archivist,
            Self::Specialist,
            Self::Verifier,
        ]
    }

    /// Verifier may confirm others' conclusions; it does not author promotions
    /// of its own high-risk claims without a second independent path.
    pub fn is_verifier(self) -> bool {
        matches!(self, Self::Verifier)
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
