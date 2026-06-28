use hide_core::config::HideConfig;
use hide_core::permission::{PermissionPolicy, PermissionRule, RiskGate, StaticPermissionEngine};
use hide_core::types::RiskLevel;
use hide_security::redaction::Redactor;
use hide_security::sandbox::{
    default_workspace_profile, render_macos_seatbelt, RenderedSandboxProfile,
};
use hide_security::storage::AtRestPolicy;

#[derive(Debug, Clone)]
pub struct SecurityServices {
    pub redactor: Redactor,
    pub at_rest: AtRestPolicy,
}

impl Default for SecurityServices {
    fn default() -> Self {
        Self {
            redactor: Redactor::default(),
            at_rest: AtRestPolicy::default(),
        }
    }
}

impl SecurityServices {
    pub fn render_workspace_sandbox(&self, root: impl Into<String>) -> RenderedSandboxProfile {
        let profile = default_workspace_profile(root);
        render_macos_seatbelt(&profile)
    }

    pub fn policy_for_config(config: &HideConfig) -> PermissionPolicy {
        let workspace = config.workspace_root.display().to_string();
        PermissionPolicy {
            default_decision: config.security.default_decision,
            rules: vec![
                PermissionRule {
                    id: "workspace-read".to_string(),
                    capability_kind: "fs.read".to_string(),
                    scope_pattern: format!("{workspace}/**"),
                    decision: hide_core::types::Decision::Allow,
                    max_risk: RiskLevel::Low,
                    reason: "workspace reads are allowed".to_string(),
                },
                PermissionRule {
                    id: "git-status".to_string(),
                    capability_kind: "fs.read".to_string(),
                    scope_pattern: "git.status".to_string(),
                    decision: hide_core::types::Decision::Allow,
                    max_risk: RiskLevel::Low,
                    reason: "git status is a read-only workspace snapshot".to_string(),
                },
                PermissionRule {
                    id: "workspace-write".to_string(),
                    capability_kind: "fs.write".to_string(),
                    scope_pattern: format!("{workspace}/**"),
                    decision: config.security.workspace_write_default,
                    max_risk: RiskLevel::High,
                    reason: "workspace write follows configured policy".to_string(),
                },
                PermissionRule {
                    id: "shell-exec".to_string(),
                    capability_kind: "process.exec".to_string(),
                    scope_pattern: "*".to_string(),
                    decision: config.security.shell_default,
                    max_risk: RiskLevel::High,
                    reason: "shell execution follows configured policy".to_string(),
                },
            ],
            risk_gates: vec![RiskGate::lethal_trifecta()],
        }
    }

    pub fn permission_engine(config: &HideConfig) -> StaticPermissionEngine {
        StaticPermissionEngine::new(Self::policy_for_config(config))
    }
}
