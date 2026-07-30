//! Fixture model provider — no real inference.
//!
//! Deterministic canned replies keyed by role + goal fragment. Used so swarm
//! orchestration and handoff tests never load a model or touch Metal.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::lenses::agent::{AgentReceipt, AgentSpec};
use crate::lenses::capsule::Claim;
use crate::lenses::evidence::EvidenceTier;
use crate::lenses::roles::AgentRole;

/// One canned reply the fixture provider may return.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FixtureReply {
    pub summary: String,
    pub tokens_used: u64,
    pub steps_used: u32,
    pub cpu_ms: u64,
    pub ram_mb: u64,
    pub evidence_tier: EvidenceTier,
    pub claim_texts: Vec<String>,
}

impl Default for FixtureReply {
    fn default() -> Self {
        Self {
            summary: "fixture ok".into(),
            tokens_used: 10,
            steps_used: 1,
            cpu_ms: 5,
            ram_mb: 16,
            evidence_tier: EvidenceTier::Asserted,
            claim_texts: vec!["fixture claim".into()],
        }
    }
}

/// Model-free provider. Role-keyed defaults; optional overrides by agent id.
#[derive(Debug, Clone, Default)]
pub struct FixtureProvider {
    role_defaults: std::collections::BTreeMap<String, FixtureReply>,
    agent_overrides: std::collections::BTreeMap<String, FixtureReply>,
}

impl FixtureProvider {
    pub fn new() -> Self {
        let mut p = Self::default();
        for role in AgentRole::all() {
            p.role_defaults.insert(
                role.as_str().to_string(),
                FixtureReply {
                    summary: format!("fixture:{role}"),
                    tokens_used: 12,
                    steps_used: 1,
                    cpu_ms: 8,
                    ram_mb: 20,
                    evidence_tier: match role {
                        AgentRole::Verifier | AgentRole::FactChecker => {
                            EvidenceTier::IndependentlyVerified
                        }
                        AgentRole::Researcher => EvidenceTier::Cited,
                        _ => EvidenceTier::Asserted,
                    },
                    claim_texts: vec![format!("{role} output")],
                },
            );
        }
        p
    }

    pub fn override_agent(mut self, agent_id: &str, reply: FixtureReply) -> Self {
        self.agent_overrides.insert(agent_id.to_string(), reply);
        self
    }

    pub fn override_role(mut self, role: AgentRole, reply: FixtureReply) -> Self {
        self.role_defaults.insert(role.as_str().to_string(), reply);
        self
    }

    /// Produce a deterministic receipt for an agent. No model call.
    pub fn run(&self, spec: &AgentSpec) -> AgentReceipt {
        let reply = self
            .agent_overrides
            .get(spec.id.as_str())
            .or_else(|| self.role_defaults.get(spec.role.as_str()))
            .cloned()
            .unwrap_or_default();

        let claims: Vec<Claim> = reply
            .claim_texts
            .iter()
            .enumerate()
            .map(|(i, text)| Claim {
                id: format!("clm_{}_{}", spec.id.as_str(), i),
                text: text.clone(),
                evidence_tier: reply.evidence_tier,
                payload: json!({
                    "goal": spec.goal,
                    "role": spec.role.as_str(),
                    "model_profile": spec.model_profile,
                }),
            })
            .collect();

        AgentReceipt {
            agent_id: spec.id.clone(),
            role: spec.role,
            ok: true,
            summary: reply.summary,
            tokens_used: reply.tokens_used,
            steps_used: reply.steps_used,
            cpu_ms: reply.cpu_ms,
            ram_mb: reply.ram_mb,
            claims,
            evidence_tier: reply.evidence_tier,
        }
    }

    /// Inspectable catalog for contracts/tests.
    pub fn catalog(&self) -> Value {
        json!({
            "kind": "fixture_provider",
            "roles": self.role_defaults.keys().cloned().collect::<Vec<_>>(),
            "agent_overrides": self.agent_overrides.keys().cloned().collect::<Vec<_>>(),
            "real_inference": false,
        })
    }
}
