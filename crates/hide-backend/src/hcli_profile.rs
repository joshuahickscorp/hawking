//! Named, bounded compute profiles for HCLI agent runs.
//!
//! A profile changes **how much exploration** an HCLI run is allowed to use;
//! it does not widen effect permissions.  All profiles default to
//! [`Autonomy::SuggestOnly`], so writes, shell commands, and other effects still
//! require the host's ordinary approval path unless the caller explicitly opts
//! into a different autonomy mode through a separate, auditable control.
//!
//! The `search_breadth` and `self_consistency_k` fields are exposed honestly.
//! They are budget/configuration values today; the current `AgentKernel` does
//! not yet turn them into a parallel swarm by itself.  A caller should record
//! [`HcliProfileSpec`] in its receipt and report realized subagent/model-call
//! counts separately.

use hide_kernel::govern::{Autonomy, Budget};
use hide_kernel::machine::state::AgentState;
use serde::{Deserialize, Serialize};

/// HCLI's explicit compute profiles, from everyday local use to a bounded
/// long-running research run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HcliProfile {
    /// A capable default for ordinary local agent work.
    Balanced,
    /// Broad and long-running, intended for substantial investigation.
    Power,
    /// The largest supported bounded local run.  It is still not unlimited.
    Maximum,
}

impl Default for HcliProfile {
    fn default() -> Self {
        Self::Balanced
    }
}

/// Serializable, receipt-friendly description of a profile application.
///
/// `configured_*` values are deliberately distinct from realized work.  For
/// example, requesting `search_breadth = 12` does not imply that twelve agents
/// actually ran unless the executor reports them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HcliProfileSpec {
    pub profile: HcliProfile,
    pub budget: Budget,
    pub default_autonomy: Autonomy,
    pub configured_search_breadth: u8,
    pub configured_self_consistency_k: u8,
    pub effect_policy: &'static str,
    pub bounded_safety_note: &'static str,
    pub realization_note: &'static str,
}

impl HcliProfile {
    /// Stable CLI spellings accepted by HCLI.
    pub const ALL: [Self; 3] = [Self::Balanced, Self::Power, Self::Maximum];

    /// Parse a profile name without silently falling back to a more permissive
    /// profile.  `None` should be surfaced as a CLI validation error.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "balanced" | "default" => Some(Self::Balanced),
            "power" => Some(Self::Power),
            "maximum" | "max" => Some(Self::Maximum),
            _ => None,
        }
    }

    /// Stable lowercase name for CLI output and evidence receipts.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Power => "power",
            Self::Maximum => "maximum",
        }
    }

    /// The finite, kernel-enforced caps for this profile.
    ///
    /// These intentionally avoid `0`, because the governor interprets zero
    /// wall-clock as unbounded.  `token_budget_hint` remains informational in
    /// the kernel (rather than an abort condition), but is finite here so the
    /// chosen intent is visible to context and receipt consumers.
    pub fn budget(self) -> Budget {
        match self {
            Self::Balanced => Budget {
                max_steps: 160,
                max_repairs: 6,
                max_replans: 8,
                max_wallclock_ms: 60 * 60 * 1_000,
                max_subagents: 16,
                max_stack_depth: 6,
                max_tool_calls: 400,
                max_edits_per_file: 12,
                max_search_depth: 2,
                search_breadth: 2,
                self_consistency_k: 2,
                token_budget_hint: 1_000_000,
            },
            Self::Power => Budget {
                max_steps: 480,
                max_repairs: 16,
                max_replans: 24,
                max_wallclock_ms: 4 * 60 * 60 * 1_000,
                max_subagents: 48,
                max_stack_depth: 8,
                max_tool_calls: 2_000,
                max_edits_per_file: 40,
                max_search_depth: 4,
                search_breadth: 6,
                self_consistency_k: 4,
                token_budget_hint: 8_000_000,
            },
            Self::Maximum => Budget {
                max_steps: 1_200,
                max_repairs: 32,
                max_replans: 48,
                max_wallclock_ms: 8 * 60 * 60 * 1_000,
                max_subagents: 96,
                max_stack_depth: 10,
                max_tool_calls: 8_000,
                max_edits_per_file: 100,
                max_search_depth: 6,
                search_breadth: 12,
                self_consistency_k: 8,
                token_budget_hint: 32_000_000,
            },
        }
    }

    /// The default autonomy is deliberately independent of compute power.
    /// A high-budget run still stops for approval before effectful actions.
    pub const fn default_autonomy(self) -> Autonomy {
        // Keep the match so adding a profile cannot accidentally inherit a
        // future enum default that is less restrictive.
        match self {
            Self::Balanced | Self::Power | Self::Maximum => Autonomy::SuggestOnly,
        }
    }

    /// A transparent snapshot suitable for `hcli run --json` and receipts.
    pub fn spec(self) -> HcliProfileSpec {
        let budget = self.budget();
        HcliProfileSpec {
            profile: self,
            configured_search_breadth: budget.search_breadth,
            configured_self_consistency_k: budget.self_consistency_k,
            budget,
            default_autonomy: self.default_autonomy(),
            effect_policy: "suggest_only; high compute does not grant raw effects",
            bounded_safety_note: "All governor caps are finite. Effect permissions remain separately gated by the host.",
            realization_note: "Breadth and self-consistency are requested budget values; report realized subagents and model calls independently.",
        }
    }

    /// Apply this profile to a freshly-started kernel state before its first
    /// transition.  The profile never changes `AgentKernel` autonomy or bypasses
    /// the tool permission/sandbox layers.
    pub fn apply_to_state(self, state: &mut AgentState) {
        state.budget = self.budget();
    }
}

impl std::fmt::Display for HcliProfile {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::{RunId, SessionId};

    #[test]
    fn profile_parser_accepts_only_named_safe_presets() {
        assert_eq!(HcliProfile::parse("balanced"), Some(HcliProfile::Balanced));
        assert_eq!(HcliProfile::parse(" DEFAULT "), Some(HcliProfile::Balanced));
        assert_eq!(HcliProfile::parse("POWER"), Some(HcliProfile::Power));
        assert_eq!(HcliProfile::parse("max"), Some(HcliProfile::Maximum));
        assert_eq!(HcliProfile::parse("unlimited"), None);
        assert_eq!(HcliProfile::parse("full_auto"), None);
    }

    #[test]
    fn profiles_scale_compute_but_keep_effects_suggest_only() {
        let balanced = HcliProfile::Balanced.spec();
        let power = HcliProfile::Power.spec();
        let maximum = HcliProfile::Maximum.spec();

        assert_eq!(balanced.default_autonomy, Autonomy::SuggestOnly);
        assert_eq!(power.default_autonomy, Autonomy::SuggestOnly);
        assert_eq!(maximum.default_autonomy, Autonomy::SuggestOnly);
        assert!(balanced
            .effect_policy
            .contains("does not grant raw effects"));

        assert!(balanced.budget.max_steps < power.budget.max_steps);
        assert!(power.budget.max_steps < maximum.budget.max_steps);
        assert!(balanced.budget.max_tool_calls < power.budget.max_tool_calls);
        assert!(power.budget.max_tool_calls < maximum.budget.max_tool_calls);
        assert!(balanced.configured_search_breadth < power.configured_search_breadth);
        assert!(power.configured_search_breadth < maximum.configured_search_breadth);
        assert!(
            balanced.configured_self_consistency_k < power.configured_self_consistency_k
                && power.configured_self_consistency_k < maximum.configured_self_consistency_k
        );
    }

    #[test]
    fn every_profile_has_finite_governor_caps() {
        for profile in HcliProfile::ALL {
            let budget = profile.budget();
            assert!(budget.max_steps > 0, "{profile}");
            assert!(budget.max_wallclock_ms > 0, "{profile}");
            assert!(budget.max_tool_calls > 0, "{profile}");
            assert!(budget.max_subagents > 0, "{profile}");
            assert!(budget.max_stack_depth > 0, "{profile}");
            assert!(budget.search_breadth > 0, "{profile}");
            assert!(budget.self_consistency_k > 0, "{profile}");
            assert!(budget.token_budget_hint > 0, "{profile}");
        }
    }

    #[test]
    fn applying_a_profile_changes_only_the_budget() {
        let mut state = AgentState::new(SessionId::new(), RunId::new(), "audit".to_string());
        state.steer.push("do not drop this".to_string());
        let run_id = state.run_id.clone();

        HcliProfile::Maximum.apply_to_state(&mut state);

        assert_eq!(state.budget, HcliProfile::Maximum.budget());
        assert_eq!(state.run_id, run_id);
        assert_eq!(state.steer, vec!["do not drop this"]);
    }
}
