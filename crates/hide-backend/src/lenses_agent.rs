//! Swarm agent specification and fixture receipts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::lenses::budget::{ResourceBudget, StopCondition};
use crate::lenses::capability::SurfaceCapability;
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::roles::AgentRole;

/// Stable agent id (`agt_…`).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new() -> Self {
        Self(format!(
            "agt_{}",
            ulid::Ulid::new().to_string().to_ascii_lowercase()
        ))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Declared output shape an agent must produce (schema id + optional JSON Schema).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputSchema {
    pub schema_id: String,
    #[serde(default)]
    pub json_schema: Value,
}

impl OutputSchema {
    pub fn named(schema_id: impl Into<String>) -> Self {
        Self {
            schema_id: schema_id.into(),
            json_schema: Value::Null,
        }
    }
}

/// What independent verification is required for this agent's outputs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationContract {
    /// Minimum evidence tier before a high-risk claim may leave the agent.
    pub min_high_risk_tier: EvidenceTier,
    /// Whether a distinct Verifier role is required for promotion.
    pub require_independent_verifier: bool,
    /// Whether a reproduced defect/oracle outranks consensus votes.
    pub reproduction_outranks_consensus: bool,
}

impl Default for VerificationContract {
    fn default() -> Self {
        Self {
            min_high_risk_tier: EvidenceTier::IndependentlyVerified,
            require_independent_verifier: true,
            reproduction_outranks_consensus: true,
        }
    }
}

/// Full agent brief: everything a swarm member receives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentSpec {
    pub id: AgentId,
    pub goal: String,
    pub role: AgentRole,
    /// Context capsule id or free-form context payload (claims, not capabilities).
    pub context_capsule: Value,
    /// Model/profile label (fixture only; no real routing).
    pub model_profile: String,
    pub tools: Vec<String>,
    pub connectors: Vec<String>,
    /// Capability derived from the swarm's permission set (non-widening).
    pub permissions: SurfaceCapability,
    pub budget: ResourceBudget,
    pub deadline_ms: Option<u64>,
    pub output_schema: OutputSchema,
    pub verification: VerificationContract,
    pub stop: StopCondition,
}

impl AgentSpec {
    pub fn builder(role: AgentRole, goal: impl Into<String>) -> AgentSpecBuilder {
        AgentSpecBuilder {
            role,
            goal: goal.into(),
            context_capsule: Value::Null,
            model_profile: "fixture/general".into(),
            tools: Vec::new(),
            connectors: Vec::new(),
            permissions: SurfaceCapability::default(),
            budget: ResourceBudget::default(),
            deadline_ms: None,
            output_schema: OutputSchema::named("default"),
            verification: VerificationContract::default(),
            stop: StopCondition::Never,
        }
    }
}

/// Fluent builder for [`AgentSpec`].
pub struct AgentSpecBuilder {
    role: AgentRole,
    goal: String,
    context_capsule: Value,
    model_profile: String,
    tools: Vec<String>,
    connectors: Vec<String>,
    permissions: SurfaceCapability,
    budget: ResourceBudget,
    deadline_ms: Option<u64>,
    output_schema: OutputSchema,
    verification: VerificationContract,
    stop: StopCondition,
}

impl AgentSpecBuilder {
    pub fn context(mut self, ctx: Value) -> Self {
        self.context_capsule = ctx;
        self
    }

    pub fn model_profile(mut self, profile: impl Into<String>) -> Self {
        self.model_profile = profile.into();
        self
    }

    pub fn tools(mut self, tools: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tools = tools.into_iter().map(Into::into).collect();
        self
    }

    pub fn connectors(mut self, c: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.connectors = c.into_iter().map(Into::into).collect();
        self
    }

    pub fn permissions(mut self, cap: SurfaceCapability) -> Self {
        self.permissions = cap;
        self
    }

    pub fn budget(mut self, budget: ResourceBudget) -> Self {
        self.budget = budget;
        self
    }

    pub fn deadline_ms(mut self, ms: u64) -> Self {
        self.deadline_ms = Some(ms);
        self
    }

    pub fn output_schema(mut self, schema: OutputSchema) -> Self {
        self.output_schema = schema;
        self
    }

    pub fn verification(mut self, v: VerificationContract) -> Self {
        self.verification = v;
        self
    }

    pub fn stop(mut self, stop: StopCondition) -> Self {
        self.stop = stop;
        self
    }

    pub fn build(self) -> AgentSpec {
        AgentSpec {
            id: AgentId::new(),
            goal: self.goal,
            role: self.role,
            context_capsule: self.context_capsule,
            model_profile: self.model_profile,
            tools: self.tools,
            connectors: self.connectors,
            permissions: self.permissions,
            budget: self.budget,
            deadline_ms: self.deadline_ms,
            output_schema: self.output_schema,
            verification: self.verification,
            stop: self.stop,
        }
    }
}

/// Deterministic fixture receipt from one agent step (no real inference).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentReceipt {
    pub agent_id: AgentId,
    pub role: AgentRole,
    pub ok: bool,
    pub summary: String,
    pub tokens_used: u64,
    pub steps_used: u32,
    pub cpu_ms: u64,
    pub ram_mb: u64,
    pub claims: Vec<crate::lenses::capsule::Claim>,
    pub evidence_tier: EvidenceTier,
}
