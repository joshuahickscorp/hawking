//! Resource economics for swarms and agents.
//!
//! More agents is not automatically better. Exhaustion of any budget axis is a
//! hard stop; the swarm records which axis fired.

use serde::{Deserialize, Serialize};

/// Which budget axis was exhausted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetAxis {
    CpuMs,
    RamMb,
    Tokens,
    Steps,
    WallMs,
    AgentCount,
}

impl BudgetAxis {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CpuMs => "cpu_ms",
            Self::RamMb => "ram_mb",
            Self::Tokens => "tokens",
            Self::Steps => "steps",
            Self::WallMs => "wall_ms",
            Self::AgentCount => "agent_count",
        }
    }
}

/// Resource bounds. Exhaustion is a hard stop.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ResourceBudget {
    pub max_cpu_ms: Option<u64>,
    pub max_ram_mb: Option<u64>,
    pub max_tokens: Option<u64>,
    pub max_steps: Option<u32>,
    pub max_wall_ms: Option<u64>,
    /// Soft ceiling on concurrent/total agents for the swarm.
    pub max_agents: Option<u32>,
}

/// Cumulative spend against a budget.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BudgetUsage {
    pub cpu_ms: u64,
    pub ram_mb_peak: u64,
    pub tokens: u64,
    pub steps: u32,
    pub wall_ms: u64,
    pub agents_launched: u32,
}

impl BudgetUsage {
    /// Which axis, if any, is exhausted under `budget`.
    pub fn exhausted_axis(&self, budget: &ResourceBudget) -> Option<BudgetAxis> {
        if budget.max_cpu_ms.is_some_and(|m| self.cpu_ms >= m) {
            return Some(BudgetAxis::CpuMs);
        }
        if budget.max_ram_mb.is_some_and(|m| self.ram_mb_peak >= m) {
            return Some(BudgetAxis::RamMb);
        }
        if budget.max_tokens.is_some_and(|m| self.tokens >= m) {
            return Some(BudgetAxis::Tokens);
        }
        if budget.max_steps.is_some_and(|m| self.steps >= m) {
            return Some(BudgetAxis::Steps);
        }
        if budget.max_wall_ms.is_some_and(|m| self.wall_ms >= m) {
            return Some(BudgetAxis::WallMs);
        }
        if budget.max_agents.is_some_and(|m| self.agents_launched > m) {
            return Some(BudgetAxis::AgentCount);
        }
        None
    }

    pub fn is_exhausted(&self, budget: &ResourceBudget) -> bool {
        self.exhausted_axis(budget).is_some()
    }
}

/// When a swarm or agent must halt. Enforced, not advisory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StopCondition {
    Never,
    AfterSteps { count: u32 },
    AfterWallMs { ms: u64 },
    ConditionMet { name: String },
    BudgetOnly,
}

/// Why a swarm or agent halted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum StopReason {
    BudgetExhausted { axis: String },
    AfterSteps { count: u32 },
    AfterWallMs { ms: u64 },
    ConditionMet { name: String },
    Cancelled,
    AuthorityDenied { what: String },
    Completed,
}

/// Swarm-level budget + stop condition package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SwarmBudget {
    pub resources: ResourceBudget,
    pub stop: StopCondition,
}

impl Default for SwarmBudget {
    fn default() -> Self {
        Self {
            resources: ResourceBudget {
                max_cpu_ms: Some(60_000),
                max_ram_mb: Some(512),
                max_tokens: Some(50_000),
                max_steps: Some(100),
                max_wall_ms: Some(300_000),
                max_agents: Some(8),
            },
            stop: StopCondition::BudgetOnly,
        }
    }
}

impl SwarmBudget {
    pub fn tight_fixture() -> Self {
        Self {
            resources: ResourceBudget {
                max_cpu_ms: Some(100),
                max_ram_mb: Some(64),
                max_tokens: Some(50),
                max_steps: Some(3),
                max_wall_ms: Some(1_000),
                max_agents: Some(4),
            },
            stop: StopCondition::BudgetOnly,
        }
    }
}
