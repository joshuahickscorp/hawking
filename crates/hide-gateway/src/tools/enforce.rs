//! Enforcement surface: credentials, policy, session affinity, effects,
//! health, version, input/output schemas (bible §16).

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Coarse effect boundary — aligns with `hide-core::EffectKind` /
/// `extension_registry::Effect` without taking a hard dependency yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectBoundary {
    Read = 0,
    Write = 1,
    Execute = 2,
    Network = 3,
    Model = 4,
}

impl EffectBoundary {
    pub fn rank(self) -> u8 {
        self as u8
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Execute => "execute",
            Self::Network => "network",
            Self::Model => "model",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolVersion {
    pub raw: String,
}

impl ToolVersion {
    pub fn parse(s: impl Into<String>) -> Self {
        Self { raw: s.into() }
    }

    pub fn as_str(&self) -> &str {
        &self.raw
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolHealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolHealth {
    pub status: ToolHealthStatus,
    pub last_error: Option<String>,
}

impl Default for ToolHealth {
    fn default() -> Self {
        Self {
            status: ToolHealthStatus::Unknown,
            last_error: None,
        }
    }
}

/// Session pin so MCP/stateful tools stay on the same connection (affinity).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionAffinity {
    pub session_id: String,
    /// Optional MCP/server sticky key (stdio pid or HTTP session).
    pub sticky_key: Option<String>,
}

impl SessionAffinity {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            sticky_key: None,
        }
    }

    pub fn with_sticky(mut self, key: impl Into<String>) -> Self {
        self.sticky_key = Some(key.into());
        self
    }
}

/// Policy gate applied at bundle retrieval (mirrors capability profiles).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolPolicy {
    /// Highest effect a tool in the granted set may declare.
    pub max_effect: EffectBoundary,
    pub allow_network: bool,
    pub require_healthy: bool,
    /// e.g. `fast` / `maximum` / `gate` / `sandbox` — informational + rule hook.
    pub profile: String,
}

/// Bundle-time enforcement checks (pure; gateway calls these).
#[derive(Debug, Clone, Default)]
pub struct ToolEnforcement {
    /// credential_key → set of session ids that hold it.
    credentials: std::collections::BTreeMap<String, BTreeSet<String>>,
}

impl ToolEnforcement {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn grant_credential(&mut self, session_id: &str, key: &str) {
        self.credentials
            .entry(key.to_string())
            .or_default()
            .insert(session_id.to_string());
    }

    pub fn session_has_credential(&self, session_id: &str, key: &str) -> bool {
        self.credentials
            .get(key)
            .map(|s| s.contains(session_id))
            .unwrap_or(false)
    }

    pub fn effect_allowed(policy: &ToolPolicy, effect: EffectBoundary) -> bool {
        if effect == EffectBoundary::Network && !policy.allow_network {
            return false;
        }
        effect.rank() <= policy.max_effect.rank()
    }
}
