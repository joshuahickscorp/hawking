//! Typed effect ledger + policy decisions over tool calls (bible sec 40, sec
//! 78.1 #7).
//!
//! Every tool the host can invoke declares a set of side EFFECTS up front (bible
//! sec 24, enforced in `hide-extension-registry`). This module turns those
//! honest declarations into a durable, typed POLICY DECISION: for a given tool
//! call it looks the tool's effects up in the builtin capability registry,
//! consults the existing `hide-security` permission engine, and derives one
//! [`PolicyDecision`]. Bible sec 40.1 ("record every policy decision") is honored
//! by the host recording each derived decision as a durable `policy.decision`
//! event; [`BackendHost::policy_decisions`](crate::BackendHost::policy_decisions)
//! reads them back.
//!
//! This layer is ADDITIVE and does NOT weaken the dispatcher's own gating (the
//! `ToolDispatcher` still evaluates the permission engine and refuses a non-Allow
//! verdict before any tool runs). It is a typed ledger + a decision surface a
//! planner / UI can reason over, not a second enforcement point.
//!
//! Effects are READ FROM THE REGISTRY, never hardcoded here: the mapping is keyed
//! on `hide_extension_registry::build_builtin_tool_registry`, so if a tool's
//! declared effects change, this classification follows without an edit.
//!
//! ## Deferred model leg
//!
//! The derivation is entirely deterministic and MODEL-FREE. A model-assisted
//! refinement (natural-language policy rules, learned risk scoring of a specific
//! `args` payload, or an LLM reviewer that upgrades/downgrades a decision) is
//! `DEFERRED_MODEL_REQUIRED`: this module never loads a model and never will on
//! this path.

use hide_core::permission::PermissionVerdict;
use hide_core::types::Decision;
use hide_extension_registry::{build_builtin_tool_registry, Effect};
use serde::{Deserialize, Serialize};

/// The typed decision the policy layer derives for a tool call. Richer than the
/// permission engine's ternary [`Decision`] (Allow/Ask/Deny): it carries the
/// scoped-grant variants a UI can offer and the two elevated gates
/// (`RequireSandbox` for process execution, `RequireReviewer` for irreversible /
/// privileged / git-history mutation) that the effect classification derives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyDecision {
    /// Allowed unconditionally (read-only effects).
    Allow,
    /// Allowed for exactly this one call.
    AllowOnce,
    /// Allowed for the remainder of the session.
    AllowForSession,
    /// Allowed for the whole repository.
    AllowForRepo,
    /// Requires an explicit human approval before running.
    Ask,
    /// Denied by policy.
    Deny,
    /// Must run under sandbox isolation (process execution).
    RequireSandbox,
    /// Requires a separate reviewer (irreversible / privileged / git-history
    /// mutation) before running.
    RequireReviewer,
}

impl PolicyDecision {
    pub fn as_str(&self) -> &'static str {
        match self {
            PolicyDecision::Allow => "allow",
            PolicyDecision::AllowOnce => "allow_once",
            PolicyDecision::AllowForSession => "allow_for_session",
            PolicyDecision::AllowForRepo => "allow_for_repo",
            PolicyDecision::Ask => "ask",
            PolicyDecision::Deny => "deny",
            PolicyDecision::RequireSandbox => "require_sandbox",
            PolicyDecision::RequireReviewer => "require_reviewer",
        }
    }

    /// Whether this decision mandates sandbox isolation before the call runs.
    pub fn requires_sandbox(&self) -> bool {
        matches!(self, PolicyDecision::RequireSandbox)
    }

    /// Whether this decision is a form of allow (any scope). `Ask`, `Deny`, and
    /// the two `Require*` gates are NOT allows.
    pub fn is_allow(&self) -> bool {
        matches!(
            self,
            PolicyDecision::Allow
                | PolicyDecision::AllowOnce
                | PolicyDecision::AllowForSession
                | PolicyDecision::AllowForRepo
        )
    }
}

/// The durable payload of a `policy.decision` event (bible sec 40.1). Carries the
/// tool id, the effects that were classified, the derived decision, and the
/// human-readable reason. Written once per [`evaluate_tool_policy`] call and read
/// back by `policy_decisions`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PolicyDecisionRecord {
    pub tool: String,
    /// The declared effects (as their stable string names), read from the
    /// capability registry, never hardcoded.
    pub effects: Vec<String>,
    pub decision: PolicyDecision,
    pub reason: String,
}

/// Look up a tool id's DECLARED effects from the builtin capability registry.
///
/// This is the [`Effect`] class mapping: it reuses
/// `hide_extension_registry::build_builtin_tool_registry` so the effects come
/// from each tool's honest manifest declaration, not a table maintained here. An
/// unknown / unregistered tool id yields an empty set (which the derivation
/// treats conservatively as "ask").
pub fn tool_declared_effects(tool_id: &str) -> Vec<Effect> {
    build_builtin_tool_registry()
        .declared_effects(tool_id)
        .unwrap_or_default()
}

/// Derive a [`PolicyDecision`] from a tool's declared effects and the permission
/// engine's verdict.
///
/// Precedence (most restrictive first), per bible sec 40:
/// 1. read-only (only [`Effect::Read`]) -> [`PolicyDecision::Allow`], no sandbox.
/// 2. an explicit engine [`Decision::Deny`] -> [`PolicyDecision::Deny`] (fail
///    closed for any elevated effect).
/// 3. [`Effect::Irreversible`] / [`Effect::Privileged`] / [`Effect::GitMutation`]
///    -> [`PolicyDecision::RequireReviewer`].
/// 4. [`Effect::Execute`] / [`Effect::Process`] -> [`PolicyDecision::RequireSandbox`].
/// 5. [`Effect::Network`] / [`Effect::ExternalMutation`] / [`Effect::SecretAccess`]
///    -> [`PolicyDecision::Ask`].
/// 6. [`Effect::Write`] -> the engine's decision (Allow/Ask/Deny).
/// 7. empty / unclassified -> [`PolicyDecision::Ask`] (conservative default).
pub fn derive_policy_decision(
    effects: &[Effect],
    verdict: &PermissionVerdict,
) -> (PolicyDecision, String) {
    let has = |e: Effect| effects.contains(&e);

    // 1. Read-only is always allowed and never needs the sandbox.
    if !effects.is_empty() && effects.iter().all(|e| *e == Effect::Read) {
        return (
            PolicyDecision::Allow,
            "read-only effects; allowed without sandbox".to_string(),
        );
    }

    // 2. An explicit engine denial is absolute (fail closed).
    if verdict.decision == Decision::Deny {
        return (
            PolicyDecision::Deny,
            format!("permission engine denied: {}", verdict.reason),
        );
    }

    // 3. Irreversible / privileged / git-history mutation needs a reviewer.
    if has(Effect::Irreversible) || has(Effect::Privileged) || has(Effect::GitMutation) {
        let which = if has(Effect::Irreversible) {
            "irreversible"
        } else if has(Effect::Privileged) {
            "privileged"
        } else {
            "git-history mutation"
        };
        return (
            PolicyDecision::RequireReviewer,
            format!("{which} effect requires a reviewer before running"),
        );
    }

    // 4. Process execution must run isolated.
    if has(Effect::Execute) || has(Effect::Process) {
        return (
            PolicyDecision::RequireSandbox,
            "process execution requires sandbox isolation".to_string(),
        );
    }

    // 5. Reaching the network / an external system / secret material needs
    //    explicit approval.
    if has(Effect::Network) || has(Effect::ExternalMutation) || has(Effect::SecretAccess) {
        return (
            PolicyDecision::Ask,
            "network / external / secret effect requires explicit approval".to_string(),
        );
    }

    // 6. A plain write follows the permission engine's decision.
    if has(Effect::Write) {
        let decision = match verdict.decision {
            Decision::Allow => PolicyDecision::Allow,
            Decision::Ask => PolicyDecision::Ask,
            Decision::Deny => PolicyDecision::Deny,
        };
        return (
            decision,
            format!("write effect; permission engine: {}", verdict.reason),
        );
    }

    // 7. No classifiable effect (unknown tool, empty declaration): ask.
    (
        PolicyDecision::Ask,
        "no classifiable declared effects; approval required".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::GrantId;

    fn verdict(decision: Decision) -> PermissionVerdict {
        PermissionVerdict {
            decision,
            reason: "test".to_string(),
            grant_id: None::<GrantId>,
        }
    }

    #[test]
    fn declared_effects_come_from_the_registry() {
        // The classification source is the registry, not a local table.
        assert_eq!(tool_declared_effects("fs.read"), vec![Effect::Read]);
        assert!(tool_declared_effects("shell.run").contains(&Effect::Process));
        assert_eq!(tool_declared_effects("git.commit"), vec![Effect::GitMutation]);
        // An unknown id is empty (conservative).
        assert!(tool_declared_effects("does.not.exist").is_empty());
    }

    #[test]
    fn read_only_is_allowed_without_sandbox() {
        let (d, _) = derive_policy_decision(&[Effect::Read], &verdict(Decision::Ask));
        assert_eq!(d, PolicyDecision::Allow);
        assert!(!d.requires_sandbox());
    }

    #[test]
    fn execute_requires_sandbox() {
        let (d, _) =
            derive_policy_decision(&[Effect::Execute, Effect::Process], &verdict(Decision::Allow));
        assert_eq!(d, PolicyDecision::RequireSandbox);
        assert!(d.requires_sandbox());
    }

    #[test]
    fn git_mutation_requires_reviewer() {
        let (d, _) = derive_policy_decision(&[Effect::GitMutation], &verdict(Decision::Allow));
        assert!(matches!(
            d,
            PolicyDecision::Ask | PolicyDecision::RequireReviewer
        ));
        assert_eq!(d, PolicyDecision::RequireReviewer);
    }

    #[test]
    fn write_follows_engine_decision() {
        let (allow, _) = derive_policy_decision(&[Effect::Write], &verdict(Decision::Allow));
        assert_eq!(allow, PolicyDecision::Allow);
        let (ask, _) = derive_policy_decision(&[Effect::Write], &verdict(Decision::Ask));
        assert_eq!(ask, PolicyDecision::Ask);
        let (deny, _) = derive_policy_decision(&[Effect::Write], &verdict(Decision::Deny));
        assert_eq!(deny, PolicyDecision::Deny);
    }

    #[test]
    fn network_and_external_ask() {
        let (d, _) = derive_policy_decision(
            &[Effect::Read, Effect::Network, Effect::ExternalMutation],
            &verdict(Decision::Allow),
        );
        assert_eq!(d, PolicyDecision::Ask);
    }

    #[test]
    fn empty_effects_ask() {
        let (d, _) = derive_policy_decision(&[], &verdict(Decision::Allow));
        assert_eq!(d, PolicyDecision::Ask);
    }

    #[test]
    fn decision_serializes_snake_case() {
        assert_eq!(PolicyDecision::RequireSandbox.as_str(), "require_sandbox");
        let j = serde_json::to_string(&PolicyDecision::RequireReviewer).unwrap();
        assert_eq!(j, "\"require_reviewer\"");
    }
}
