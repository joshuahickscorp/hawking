//! The plan-as-data contract (bible ch.02 Appendix A.1).
//!
//! A plan is a DAG of steps. **Each step declares its `acceptance` up front** —
//! the oracle contract that must pass before the step advances. This is the
//! chapter's most important field (K1: no state advances on faith): the plan
//! commits, *before acting*, to how each step will be machine-verified.

use crate::govern::Budget;
use crate::search::strategy::SearchTier;
use hide_core::ids::{PlanId, StepId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub id: PlanId,
    pub title: String,
    pub objective: String,
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
    /// The governor contract for this plan (A.5). Carried on the plan so a
    /// replan can revise caps and so subagents inherit a derived child budget.
    #[serde(default)]
    pub budget: Budget,
}

impl Plan {
    /// A minimal one-step plan whose single step verifies via the given oracles.
    /// Used by the stub planner and tests; the real planner emits richer DAGs.
    pub fn single_step(title: impl Into<String>, objective: impl Into<String>) -> Self {
        Self {
            id: PlanId::new(),
            title: title.into(),
            objective: objective.into(),
            steps: vec![PlanStep::new(
                "Architecture scaffold pass",
                StepKind::Edit,
                Acceptance::predicate("folder/module structure exists and core contracts compile"),
            )],
            status: PlanStatus::Active,
            budget: Budget::default(),
        }
    }

    pub fn step(&self, id: &StepId) -> Option<&PlanStep> {
        self.steps.iter().find(|s| &s.id == id)
    }

    pub fn step_mut(&mut self, id: &StepId) -> Option<&mut PlanStep> {
        self.steps.iter_mut().find(|s| &s.id == id)
    }
}

/// A single plan step (A.1). `acceptance` is required (the verifier contract).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: StepId,
    /// The step this one elaborates (for decomposition); `None` at the top level.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<StepId>,
    pub title: String,
    pub kind: StepKind,
    /// Why this step exists — carried forward into repair/replan lessons.
    #[serde(default)]
    pub rationale: String,
    pub dependencies: Vec<StepId>,
    pub status: StepStatus,
    /// THE VERIFIER CONTRACT — the oracles that must pass for this step (K1).
    pub acceptance: Acceptance,
    /// Optional concrete tool the act stage should dispatch (e.g. `"build.run"`,
    /// `"edit.write_file"`). When set, `Act` runs it through the tool dispatcher.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_hint: Option<String>,
    /// Args for `tool_hint` (a JSON object).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_args: Option<serde_json::Value>,
    /// Artifacts this step produces that downstream steps consume.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub produced: Vec<String>,
    /// Per-step search-tier override (escalate this hard step to best-of-N/ToT).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_hint: Option<SearchHint>,
    /// How many times this step has been attempted (act stage).
    #[serde(default)]
    pub attempts: u32,
    /// How many repair cycles this step has consumed.
    #[serde(default)]
    pub repairs: u32,
}

impl PlanStep {
    pub fn new(title: impl Into<String>, kind: StepKind, acceptance: Acceptance) -> Self {
        Self {
            id: StepId::new(),
            parent: None,
            title: title.into(),
            kind,
            rationale: String::new(),
            dependencies: Vec::new(),
            status: StepStatus::Pending,
            acceptance,
            tool_hint: None,
            tool_args: None,
            produced: Vec::new(),
            search_hint: None,
            attempts: 0,
            repairs: 0,
        }
    }

    /// Does the step mutate the world (needs an autonomy gate / approval)?
    pub fn is_effectful(&self) -> bool {
        matches!(
            self.kind,
            StepKind::Edit | StepKind::Command | StepKind::Delegate
        )
    }
}

/// The verifier contract (A.1 `acceptance`). Lists the oracle ids that must pass,
/// the human predicate, optional test selectors, and a probabilistic threshold
/// used only when no deterministic oracle applies.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Acceptance {
    /// Oracle ids resolved against the oracle registry (deterministic preferred).
    #[serde(default)]
    pub oracles: Vec<String>,
    /// Human-readable success condition.
    pub predicate: String,
    /// Optional test selectors for the `test` oracle.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tests: Vec<String>,
    /// Probabilistic fallback threshold (only consulted when no deterministic
    /// oracle applies).
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

fn default_threshold() -> f32 {
    0.7
}

impl Acceptance {
    /// An acceptance with only a human predicate (no oracle ids) — verified by the
    /// probabilistic fallback. Useful for `synthesize`/`investigate` steps.
    pub fn predicate(predicate: impl Into<String>) -> Self {
        Self {
            oracles: Vec::new(),
            predicate: predicate.into(),
            tests: Vec::new(),
            threshold: default_threshold(),
        }
    }

    /// An acceptance backed by a list of (deterministic) oracle ids.
    pub fn with_oracles(predicate: impl Into<String>, oracles: Vec<String>) -> Self {
        Self {
            oracles,
            predicate: predicate.into(),
            tests: Vec::new(),
            threshold: default_threshold(),
        }
    }
}

/// Per-step search override (A.1 `search_hint`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHint {
    pub tier: SearchTier,
    #[serde(default = "default_n")]
    pub n: u32,
}

fn default_n() -> u32 {
    4
}

/// Aligned to A.1: investigate / edit / command / verify / synthesize /
/// decompose / delegate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    Investigate,
    Edit,
    Command,
    Verify,
    Synthesize,
    Decompose,
    Delegate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Ready,
    Running,
    Blocked,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    Active,
    Completed,
    Failed,
    Superseded,
}
