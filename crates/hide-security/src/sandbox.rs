use hide_core::security::{NetworkPolicy, SandboxProfile, SandboxTier};
use hide_core::types::Decision;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RenderedSandboxProfile {
    pub tier: SandboxTier,
    pub profile_text: String,
    pub warnings: Vec<String>,
}

pub fn render_macos_seatbelt(profile: &SandboxProfile) -> RenderedSandboxProfile {
    let mut text = String::from("(version 1)\n(deny default)\n");
    let mut warnings = Vec::new();
    for root in &profile.read_roots {
        text.push_str(&format!(
            "(allow file-read* (subpath \"{}\"))\n",
            escape(root)
        ));
    }
    for root in &profile.write_roots {
        text.push_str(&format!(
            "(allow file-write* (subpath \"{}\"))\n",
            escape(root)
        ));
    }
    match profile.network.default {
        Decision::Deny => {
            warnings.push("network default deny; host proxy required for egress".to_string())
        }
        Decision::Ask => {
            warnings.push("network ask cannot be enforced by seatbelt alone".to_string())
        }
        Decision::Allow => text.push_str("(allow network*)\n"),
    }
    if profile.allowed_commands.is_empty() {
        warnings.push("no process-exec allowlist rendered yet".to_string());
    }
    RenderedSandboxProfile {
        tier: profile.tier,
        profile_text: text,
        warnings,
    }
}

pub fn default_workspace_profile(root: impl Into<String>) -> SandboxProfile {
    SandboxProfile {
        tier: SandboxTier::Seatbelt,
        read_roots: vec![root.into()],
        write_roots: Vec::new(),
        allowed_commands: Vec::new(),
        network: NetworkPolicy::default(),
    }
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}
