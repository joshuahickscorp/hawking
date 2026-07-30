//! YOU swarms — governed teams on the fleet substrate concept.
//!
//! A swarm is not prompt multiplication: each agent receives goal, role,
//! context capsule, model/profile, tools/connectors, permissions, budget,
//! deadline, output schema, and verification contract. Resource economics
//! are enforced; exceeding budget halts the swarm and records why.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::agent::{AgentReceipt, AgentSpec};
use crate::lenses::budget::{BudgetUsage, ResourceBudget, StopCondition, StopReason, SwarmBudget};
use crate::lenses::capability::{SurfaceCapability, SurfacePermissionSet};
use crate::lenses::error::{Result, YouError};
use crate::lenses::fixture::FixtureProvider;
use crate::lenses::modes::SwarmMode;
use crate::lenses::roles::AgentRole;

/// Stable swarm id (`swm_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SwarmId(pub String);

impl SwarmId {
    pub fn new() -> Self {
        Self(format!(
            "swm_{}",
            ulid::Ulid::new().to_string().to_ascii_lowercase()
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SwarmId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SwarmId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Lifecycle of a swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmStatus {
    Pending,
    Running,
    /// Budget or stop condition fired; further steps refuse.
    Halted,
    Completed,
    Cancelled,
}

/// A governed swarm: mode, agents, shared goal, budget, permission root.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Swarm {
    pub id: SwarmId,
    pub goal: String,
    pub mode: SwarmMode,
    pub agents: Vec<AgentSpec>,
    /// Root permission set; every agent capability must be within this.
    pub permissions: SurfacePermissionSet,
    pub budget: SwarmBudget,
    pub usage: BudgetUsage,
    pub status: SwarmStatus,
    pub stop_reason: Option<StopReason>,
    pub receipts: Vec<AgentReceipt>,
    pub created_ms: u64,
    pub updated_ms: u64,
}

impl Swarm {
    /// Declare a swarm. Agents receive subset capabilities of `permissions`.
    pub fn declare(
        goal: impl Into<String>,
        mode: SwarmMode,
        permissions: SurfacePermissionSet,
        budget: SwarmBudget,
        now_ms: u64,
    ) -> Self {
        Self {
            id: SwarmId::new(),
            goal: goal.into(),
            mode,
            agents: Vec::new(),
            permissions,
            budget,
            usage: BudgetUsage::default(),
            status: SwarmStatus::Pending,
            stop_reason: None,
            receipts: Vec::new(),
            created_ms: now_ms,
            updated_ms: now_ms,
        }
    }

    /// Add an agent whose capability is derived as a subset of swarm permissions.
    pub fn add_agent(&mut self, mut spec: AgentSpec) -> Result<()> {
        if !matches!(self.status, SwarmStatus::Pending | SwarmStatus::Running) {
            return Err(YouError::InvalidState(format!(
                "swarm {} cannot add agents in status {:?}",
                self.id, self.status
            )));
        }
        // Derive capability: requested tools/connectors must be in swarm set.
        let cap = self.permissions.derive_capability_subset(
            spec.tools.iter().map(String::as_str),
            spec.connectors.iter().map(String::as_str),
        )?;
        // If tools/connectors empty, grant empty capability (still within set).
        if spec.tools.is_empty() && spec.connectors.is_empty() {
            spec.permissions = SurfaceCapability::default();
        } else {
            spec.permissions = cap;
        }
        if !spec.permissions.is_within(&self.permissions) {
            return Err(YouError::CapabilityMissing(
                "agent capability not within swarm permission set".into(),
            ));
        }

        // Agent-count ceiling: refuse before launch if already at max.
        if let Some(max) = self.budget.resources.max_agents {
            if (self.agents.len() as u32) >= max {
                return Err(YouError::BudgetExhausted(format!(
                    "max_agents={max} already reached"
                )));
            }
        }

        self.agents.push(spec);
        self.usage.agents_launched = self.agents.len() as u32;
        Ok(())
    }

    /// Convenience: build and add an agent with role + goal under swarm perms.
    pub fn spawn_role(
        &mut self,
        role: AgentRole,
        agent_goal: impl Into<String>,
        tools: impl IntoIterator<Item = impl Into<String>>,
        connectors: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<usize> {
        let tools: Vec<String> = tools.into_iter().map(Into::into).collect();
        let connectors: Vec<String> = connectors.into_iter().map(Into::into).collect();
        let cap = self.permissions.derive_capability_subset(
            tools.iter().map(String::as_str),
            connectors.iter().map(String::as_str),
        )?;
        let spec = AgentSpec::builder(role, agent_goal)
            .tools(tools)
            .connectors(connectors)
            .permissions(cap)
            .budget(self.budget.resources.clone())
            .stop(self.budget.stop.clone())
            .context(serde_json::json!({
                "swarm_id": self.id.as_str(),
                "swarm_goal": self.goal,
                "mode": self.mode.as_str(),
            }))
            .build();
        self.add_agent(spec)?;
        Ok(self.agents.len() - 1)
    }

    pub fn may_run(&self) -> bool {
        matches!(self.status, SwarmStatus::Pending | SwarmStatus::Running)
    }

    /// Run all pending agents once via the fixture provider. Enforces budget
    /// after each agent; on exhaustion, halts and records why.
    pub fn run_round(
        &mut self,
        provider: &FixtureProvider,
        now_ms: u64,
    ) -> Result<Vec<AgentReceipt>> {
        if !self.may_run() {
            return Err(YouError::InvalidState(format!(
                "swarm {} is {:?}: {:?}",
                self.id, self.status, self.stop_reason
            )));
        }
        self.status = SwarmStatus::Running;
        self.updated_ms = now_ms;

        let mut round = Vec::new();
        // Snapshot agents to avoid borrow issues while mutating self.
        let agents: Vec<AgentSpec> = self.agents.clone();
        for spec in &agents {
            if let Some(reason) = self.check_stop() {
                self.halt(reason, now_ms);
                break;
            }
            let receipt = provider.run(spec);
            self.apply_receipt(&receipt);
            round.push(receipt.clone());
            self.receipts.push(receipt);

            if let Some(reason) = self.check_stop() {
                self.halt(reason, now_ms);
                break;
            }
        }

        if self.status == SwarmStatus::Running {
            // Round finished without budget halt.
            if self.agents.len() == self.receipts.len()
                || matches!(self.budget.stop, StopCondition::AfterSteps { .. })
            {
                // Leave Running if more rounds possible; mark Completed only
                // when stop says so or caller finishes explicitly.
            }
        }
        Ok(round)
    }

    /// Mark swarm completed when work is done under budget.
    pub fn complete(&mut self, now_ms: u64) -> Result<()> {
        if self.status == SwarmStatus::Halted {
            return Err(YouError::InvalidState(
                "cannot complete a halted swarm".into(),
            ));
        }
        self.status = SwarmStatus::Completed;
        self.stop_reason = Some(StopReason::Completed);
        self.updated_ms = now_ms;
        Ok(())
    }

    fn apply_receipt(&mut self, receipt: &AgentReceipt) {
        self.usage.tokens = self.usage.tokens.saturating_add(receipt.tokens_used);
        self.usage.steps = self.usage.steps.saturating_add(receipt.steps_used);
        self.usage.cpu_ms = self.usage.cpu_ms.saturating_add(receipt.cpu_ms);
        self.usage.wall_ms = self.usage.wall_ms.saturating_add(receipt.cpu_ms); // fixture: wall≈cpu
        if receipt.ram_mb > self.usage.ram_mb_peak {
            self.usage.ram_mb_peak = receipt.ram_mb;
        }
    }

    fn check_stop(&self) -> Option<StopReason> {
        if let Some(axis) = self.usage.exhausted_axis(&self.budget.resources) {
            return Some(StopReason::BudgetExhausted {
                axis: axis.as_str().to_string(),
            });
        }
        match &self.budget.stop {
            StopCondition::Never | StopCondition::BudgetOnly => None,
            StopCondition::AfterSteps { count } => {
                if self.usage.steps >= *count {
                    Some(StopReason::AfterSteps { count: *count })
                } else {
                    None
                }
            }
            StopCondition::AfterWallMs { ms } => {
                if self.usage.wall_ms >= *ms {
                    Some(StopReason::AfterWallMs { ms: *ms })
                } else {
                    None
                }
            }
            StopCondition::ConditionMet { .. } => None,
        }
    }

    fn halt(&mut self, reason: StopReason, now_ms: u64) {
        self.status = SwarmStatus::Halted;
        self.stop_reason = Some(reason);
        self.updated_ms = now_ms;
    }

    /// Signal an external stop condition.
    pub fn signal_condition(&mut self, name: &str, now_ms: u64) -> Result<()> {
        match &self.budget.stop {
            StopCondition::ConditionMet { name: expected } if expected == name => {
                self.halt(
                    StopReason::ConditionMet {
                        name: name.to_string(),
                    },
                    now_ms,
                );
                Ok(())
            }
            _ => Err(YouError::InvalidState(format!(
                "condition '{name}' is not this swarm's stop condition"
            ))),
        }
    }

    /// Inspectable declaration for contracts and UI.
    pub fn declaration(&self) -> Value {
        serde_json::json!({
            "id": self.id.as_str(),
            "goal": self.goal,
            "mode": self.mode.as_str(),
            "status": self.status,
            "stop_reason": self.stop_reason,
            "budget": self.budget,
            "usage": self.usage,
            "permissions": {
                "tools": self.permissions.tools().iter().cloned().collect::<Vec<_>>(),
                "connectors": self.permissions.connectors().iter().cloned().collect::<Vec<_>>(),
            },
            "agents": self.agents.iter().map(|a| serde_json::json!({
                "id": a.id.as_str(),
                "role": a.role.as_str(),
                "goal": a.goal,
                "tools": a.tools,
                "connectors": a.connectors,
                "model_profile": a.model_profile,
                "output_schema": a.output_schema.schema_id,
            })).collect::<Vec<_>>(),
            "receipt_count": self.receipts.len(),
        })
    }

    /// Shared goal + mode context capsule payload for agents.
    pub fn context_capsule_template(&self) -> Value {
        serde_json::json!({
            "swarm_id": self.id.as_str(),
            "goal": self.goal,
            "mode": self.mode.as_str(),
            "kind": "swarm_context",
        })
    }
}

/// Helper: default YOU swarm permission root (personal connectors read, no shell).
pub fn you_swarm_permissions() -> SurfacePermissionSet {
    SurfacePermissionSet::new(
        [
            "connector.read",
            "memory.read",
            "research.read",
            "object.read",
            "write.draft",
        ],
        ["gmail", "calendar", "personal_vault", "rss"],
    )
}

/// Helper: tight resource budget for tests that expect early halt.
pub fn test_budget(max_tokens: u64, max_steps: u32) -> SwarmBudget {
    SwarmBudget {
        resources: ResourceBudget {
            max_cpu_ms: Some(10_000),
            max_ram_mb: Some(256),
            max_tokens: Some(max_tokens),
            max_steps: Some(max_steps),
            max_wall_ms: Some(60_000),
            max_agents: Some(8),
        },
        stop: StopCondition::BudgetOnly,
    }
}
