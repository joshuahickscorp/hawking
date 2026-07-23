use crate::ids::ValueId;
use crate::types::{Decision, Provenance, TrustLevel};
use serde::{Deserialize, Serialize};

// NOTE: not `Eq` — `Provenance.confidence` is an `f32`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaintedValue {
    pub id: ValueId,
    pub provenance: Provenance,
    pub labels: Vec<String>,
}

impl TaintedValue {
    pub fn trusted(source: impl Into<String>) -> Self {
        Self {
            id: ValueId::new(),
            provenance: Provenance::trusted(source),
            labels: Vec::new(),
        }
    }

    pub fn is_untrusted(&self) -> bool {
        matches!(
            self.provenance.trust,
            TrustLevel::ToolOutput | TrustLevel::Network | TrustLevel::Untrusted
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LethalTrifectaAssessment {
    pub has_private_data: bool,
    pub has_untrusted_content: bool,
    pub has_exfiltration: bool,
    pub decision: Decision,
    pub reason: String,
}

impl LethalTrifectaAssessment {
    pub fn assess(
        has_private_data: bool,
        has_untrusted_content: bool,
        has_exfiltration: bool,
    ) -> Self {
        let triggered = has_private_data && has_untrusted_content && has_exfiltration;
        Self {
            has_private_data,
            has_untrusted_content,
            has_exfiltration,
            decision: if triggered {
                Decision::Ask
            } else {
                Decision::Allow
            },
            reason: if triggered {
                "lethal-trifecta risk requires explicit approval".to_string()
            } else {
                "no lethal-trifecta risk".to_string()
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxProfile {
    pub tier: SandboxTier,
    pub read_roots: Vec<String>,
    pub write_roots: Vec<String>,
    pub allowed_commands: Vec<String>,
    pub network: NetworkPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxTier {
    None,
    ReadOnly,
    WorkspaceWrite,
    Seatbelt,
    MicroVm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkPolicy {
    pub default: Decision,
    pub allowed_hosts: Vec<String>,
    pub denied_hosts: Vec<String>,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self {
            default: Decision::Deny,
            allowed_hosts: Vec::new(),
            denied_hosts: Vec::new(),
        }
    }
}
