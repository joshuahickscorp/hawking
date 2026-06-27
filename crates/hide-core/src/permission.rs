use crate::error::{HideError, Result};
use crate::ids::{GrantId, PluginId, RunId, TimestampMs};
use crate::types::{Decision, Effect, EffectKind, ResourceScope, RiskLevel};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capability {
    pub kind: String,
    pub scope: ResourceScope,
}

impl Capability {
    pub fn new(kind: impl Into<String>, pattern: impl Into<String>) -> Self {
        let kind = kind.into();
        Self {
            scope: ResourceScope {
                kind: kind.clone(),
                pattern: pattern.into(),
            },
            kind,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityGrant {
    pub id: GrantId,
    pub capabilities: Vec<Capability>,
    pub decision: Decision,
    pub granted_by: GrantActor,
    pub run_id: Option<RunId>,
    pub plugin_id: Option<PluginId>,
    pub expires_at_ms: Option<TimestampMs>,
    pub exact_effect_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantActor {
    User,
    Policy,
    System,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    pub id: String,
    pub capability_kind: String,
    pub scope_pattern: String,
    pub decision: Decision,
    pub max_risk: RiskLevel,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionPolicy {
    pub default_decision: Decision,
    pub rules: Vec<PermissionRule>,
    pub risk_gates: Vec<RiskGate>,
}

impl Default for PermissionPolicy {
    fn default() -> Self {
        Self {
            default_decision: Decision::Ask,
            rules: Vec::new(),
            risk_gates: vec![RiskGate::lethal_trifecta()],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiskGate {
    pub id: String,
    pub description: String,
    pub forced_decision: Decision,
}

impl RiskGate {
    pub fn lethal_trifecta() -> Self {
        Self {
            id: "lethal_trifecta".to_string(),
            description:
                "private data + untrusted content + exfiltration ability must be explicitly gated"
                    .to_string(),
            forced_decision: Decision::Ask,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRequest {
    pub capability_kind: String,
    pub target: String,
    pub risk: RiskLevel,
    pub effects: Vec<Effect>,
    pub grant: Option<CapabilityGrant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionVerdict {
    pub decision: Decision,
    pub reason: String,
    pub grant_id: Option<GrantId>,
}

pub trait PermissionEngine: Send + Sync {
    fn evaluate(&self, request: &PermissionRequest) -> PermissionVerdict;

    fn require_allowed(&self, request: &PermissionRequest) -> Result<()> {
        let verdict = self.evaluate(request);
        match verdict.decision {
            Decision::Allow => Ok(()),
            Decision::Ask => Err(HideError::PolicyDenied(format!(
                "approval required for {}: {}",
                request.capability_kind, verdict.reason
            ))),
            Decision::Deny => Err(HideError::PolicyDenied(verdict.reason)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct StaticPermissionEngine {
    pub policy: PermissionPolicy,
}

impl StaticPermissionEngine {
    pub fn new(policy: PermissionPolicy) -> Self {
        Self { policy }
    }

    fn rule_matches(rule: &PermissionRule, request: &PermissionRequest) -> bool {
        rule.capability_kind == request.capability_kind
            && pattern_matches(&rule.scope_pattern, &request.target)
            && request.risk <= rule.max_risk
    }
}

impl PermissionEngine for StaticPermissionEngine {
    fn evaluate(&self, request: &PermissionRequest) -> PermissionVerdict {
        if request
            .effects
            .iter()
            .any(|e| e.kind == EffectKind::Network && e.risk >= RiskLevel::High)
        {
            return PermissionVerdict {
                decision: Decision::Ask,
                reason: "high-risk network effect requires explicit approval".to_string(),
                grant_id: request.grant.as_ref().map(|g| g.id.clone()),
            };
        }

        let matching: Vec<_> = self
            .policy
            .rules
            .iter()
            .filter(|rule| Self::rule_matches(rule, request))
            .collect();

        if let Some(rule) = matching.iter().find(|rule| rule.decision == Decision::Deny) {
            return PermissionVerdict {
                decision: Decision::Deny,
                reason: rule.reason.clone(),
                grant_id: request.grant.as_ref().map(|g| g.id.clone()),
            };
        }

        if let Some(rule) = matching
            .iter()
            .find(|rule| rule.decision == Decision::Allow)
        {
            return PermissionVerdict {
                decision: Decision::Allow,
                reason: rule.reason.clone(),
                grant_id: request.grant.as_ref().map(|g| g.id.clone()),
            };
        }

        PermissionVerdict {
            decision: self.policy.default_decision,
            reason: "no matching policy rule".to_string(),
            grant_id: request.grant.as_ref().map(|g| g.id.clone()),
        }
    }
}

fn pattern_matches(pattern: &str, target: &str) -> bool {
    if pattern == "*" || pattern == "**" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix("/**") {
        return target.starts_with(prefix);
    }
    pattern == target
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deny_beats_allow() {
        let engine = StaticPermissionEngine::new(PermissionPolicy {
            default_decision: Decision::Ask,
            rules: vec![
                PermissionRule {
                    id: "allow".to_string(),
                    capability_kind: "fs.write".to_string(),
                    scope_pattern: "/tmp/**".to_string(),
                    decision: Decision::Allow,
                    max_risk: RiskLevel::High,
                    reason: "tmp allowed".to_string(),
                },
                PermissionRule {
                    id: "deny".to_string(),
                    capability_kind: "fs.write".to_string(),
                    scope_pattern: "/tmp/secrets/**".to_string(),
                    decision: Decision::Deny,
                    max_risk: RiskLevel::Critical,
                    reason: "secrets denied".to_string(),
                },
            ],
            risk_gates: Vec::new(),
        });
        let verdict = engine.evaluate(&PermissionRequest {
            capability_kind: "fs.write".to_string(),
            target: "/tmp/secrets/key".to_string(),
            risk: RiskLevel::Low,
            effects: Vec::new(),
            grant: None,
        });
        assert_eq!(verdict.decision, Decision::Deny);
    }
}
