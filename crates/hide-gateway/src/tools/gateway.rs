//! Tool gateway facade: catalog + bundle retrieve under enforcement.

use super::bundle::{ToolBundle, ToolRef};
use super::enforce::{
    EffectBoundary, SessionAffinity, ToolEnforcement, ToolHealth, ToolHealthStatus, ToolPolicy,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ToolGatewayError {
    #[error("unknown bundle: {0}")]
    UnknownBundle(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("required tool unhealthy: {tool_id} ({status:?})")]
    UnhealthyTool {
        tool_id: String,
        status: ToolHealthStatus,
    },
    #[error("policy denied tool {tool_id}: {reason}")]
    PolicyDenied { tool_id: String, reason: String },
    #[error("missing credential {credential} for session {session_id} (tool {tool_id})")]
    MissingCredential {
        session_id: String,
        tool_id: String,
        credential: String,
    },
    #[error("schema missing for tool {0}")]
    SchemaMissing(String),
}

/// A bundle granted to a session after enforcement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GrantedBundle {
    pub session_id: String,
    pub bundle: ToolBundle,
    pub tools: Vec<ToolRef>,
    pub sticky_key: Option<String>,
}

/// Tool gateway: retrieves sets, not only isolated tools.
pub struct ToolGateway {
    tools: BTreeMap<String, ToolRef>,
    bundles: BTreeMap<String, ToolBundle>,
    health: BTreeMap<String, ToolHealth>,
    enforcement: ToolEnforcement,
    /// session_id → granted bundle ids (session affinity ledger).
    session_bundles: BTreeMap<String, BTreeSet<String>>,
}

impl ToolGateway {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
            bundles: BTreeMap::new(),
            health: BTreeMap::new(),
            enforcement: ToolEnforcement::new(),
            session_bundles: BTreeMap::new(),
        }
    }

    pub fn register_tool(&mut self, tool: ToolRef) {
        self.health
            .entry(tool.id.clone())
            .or_insert_with(ToolHealth::default);
        self.tools.insert(tool.id.clone(), tool);
    }

    pub fn register_bundle(&mut self, bundle: ToolBundle) {
        self.bundles.insert(bundle.id.clone(), bundle);
    }

    pub fn set_health(&mut self, tool_id: impl Into<String>, health: ToolHealth) {
        self.health.insert(tool_id.into(), health);
    }

    pub fn grant_credential(&mut self, session_id: &str, key: &str) {
        self.enforcement.grant_credential(session_id, key);
    }

    pub fn session_has_bundle(&self, session_id: &str, bundle_id: &str) -> bool {
        self.session_bundles
            .get(session_id)
            .map(|s| s.contains(bundle_id))
            .unwrap_or(false)
    }

    /// Retrieve a mutually-useful tool set under policy + health + credentials.
    pub fn retrieve_bundle(
        &mut self,
        bundle_id: &str,
        session: &SessionAffinity,
        policy: &ToolPolicy,
    ) -> Result<GrantedBundle, ToolGatewayError> {
        let bundle = self
            .bundles
            .get(bundle_id)
            .cloned()
            .ok_or_else(|| ToolGatewayError::UnknownBundle(bundle_id.to_string()))?;

        let mut granted_tools = Vec::with_capacity(bundle.members.len());

        for member in &bundle.members {
            let tool = self
                .tools
                .get(&member.tool_id)
                .cloned()
                .ok_or_else(|| ToolGatewayError::UnknownTool(member.tool_id.clone()))?;

            // Version present (scaffold: non-empty raw).
            if tool.version.as_str().is_empty() {
                return Err(ToolGatewayError::UnknownTool(format!(
                    "{} (empty version)",
                    tool.id
                )));
            }

            // Schemas required for grant (progressive disclosure: model may not
            // see them yet, but the gateway must hold them).
            if tool.input_schema.is_null() {
                return Err(ToolGatewayError::SchemaMissing(tool.id.clone()));
            }

            // Health
            let health = self.health.get(&tool.id).cloned().unwrap_or_default();
            if policy.require_healthy
                && member.required
                && !matches!(
                    health.status,
                    ToolHealthStatus::Healthy | ToolHealthStatus::Degraded
                )
            {
                return Err(ToolGatewayError::UnhealthyTool {
                    tool_id: tool.id.clone(),
                    status: health.status,
                });
            }

            // Effect boundaries
            for effect in &tool.effects {
                if !ToolEnforcement::effect_allowed(policy, *effect) {
                    return Err(ToolGatewayError::PolicyDenied {
                        tool_id: tool.id.clone(),
                        reason: format!(
                            "effect {} exceeds policy max {:?} (network allowed={})",
                            effect.as_str(),
                            policy.max_effect,
                            policy.allow_network
                        ),
                    });
                }
            }
            if tool.effects.contains(&EffectBoundary::Network) && !policy.allow_network {
                return Err(ToolGatewayError::PolicyDenied {
                    tool_id: tool.id.clone(),
                    reason: "network disabled by policy".into(),
                });
            }

            // Credentials
            if let Some(cred) = &tool.requires_credential {
                if !self
                    .enforcement
                    .session_has_credential(&session.session_id, cred)
                {
                    return Err(ToolGatewayError::MissingCredential {
                        session_id: session.session_id.clone(),
                        tool_id: tool.id.clone(),
                        credential: cred.clone(),
                    });
                }
            }

            granted_tools.push(tool);
        }

        self.session_bundles
            .entry(session.session_id.clone())
            .or_default()
            .insert(bundle.id.clone());

        Ok(GrantedBundle {
            session_id: session.session_id.clone(),
            bundle,
            tools: granted_tools,
            sticky_key: session.sticky_key.clone(),
        })
    }
}

impl Default for ToolGateway {
    fn default() -> Self {
        Self::new()
    }
}
