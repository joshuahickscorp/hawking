//! The Governor — the single chokepoint (K8) that makes runaway agents
//! *structurally impossible* (bible ch.02 §4.3).
//!
//! Every FSM transition calls [`Governor::check`] first. The defining choice
//! (K4): hard caps are **wallclock / steps / effect-counts**, not token spend —
//! because compute is free locally, the limiting reagent is time and the number
//! of irreversible effects, not tokens. `token_budget_hint` only informs the
//! context compiler; it never aborts by default.

use crate::machine::state::AgentState;
use serde::{Deserialize, Serialize};

/// The budget object (A.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub max_steps: u32,
    pub max_repairs: u8,
    pub max_replans: u8,
    pub max_wallclock_ms: u64,
    pub max_subagents: u32,
    pub max_stack_depth: u8,
    pub max_tool_calls: u32,
    pub max_edits_per_file: u32,
    pub max_search_depth: u8,
    /// Best-of-N breadth (default 1 = ReAct).
    pub search_breadth: u8,
    pub self_consistency_k: u8,
    /// Informational only (K4): never aborts the run by default. `0` = unbounded.
    pub token_budget_hint: u64,
}

impl Default for Budget {
    fn default() -> Self {
        // Defaults from §4.3.1 / Appendix A.5.
        Self {
            max_steps: 80,
            max_repairs: 3,
            max_replans: 4,
            max_wallclock_ms: 30 * 60 * 1000,
            max_subagents: 8,
            max_stack_depth: 5,
            max_tool_calls: 200,
            max_edits_per_file: 5,
            max_search_depth: 0,
            search_breadth: 1,
            self_consistency_k: 1,
            token_budget_hint: 0,
        }
    }
}

impl Budget {
    /// Derive a child budget for a subagent: same caps, fewer steps/effects,
    /// one less stack level (so recursion stays bounded — K8).
    pub fn child(&self) -> Self {
        Self {
            max_steps: (self.max_steps / 2).max(4),
            max_tool_calls: (self.max_tool_calls / 2).max(8),
            max_stack_depth: self.max_stack_depth.saturating_sub(1),
            max_subagents: self.max_subagents.saturating_sub(1),
            ..self.clone()
        }
    }
}

/// Consumed-so-far, checked against [`Budget`] (A.5 `BudgetLedger`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct BudgetLedger {
    pub steps: u32,
    pub replans: u8,
    pub tool_calls: u32,
    pub subagents_live: u32,
    pub subagents_total: u32,
    pub stack_depth: u8,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Wall-clock millis at run start (set on first `tick`).
    #[serde(default)]
    pub started_ms: Option<u64>,
    /// Last observed elapsed wall-clock (telemetry).
    #[serde(default)]
    pub elapsed_ms: u64,
}

impl BudgetLedger {
    pub fn consume_step(&mut self) {
        self.steps += 1;
    }

    pub fn consume_tool_call(&mut self) {
        self.tool_calls += 1;
    }

    pub fn consume_replan(&mut self) {
        self.replans = self.replans.saturating_add(1);
    }

    pub fn add_tokens(&mut self, input: u64, output: u64) {
        self.input_tokens += input;
        self.output_tokens += output;
    }

    /// Record elapsed wall-clock against the run start.
    pub fn tick(&mut self, now_ms: u64) {
        let started = *self.started_ms.get_or_insert(now_ms);
        self.elapsed_ms = now_ms.saturating_sub(started);
    }
}

/// Why the governor aborted a run (A.5 / §4.3.2). Structured so the kernel can
/// finalize honestly with the reason rather than panicking.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "cap", content = "detail")]
pub enum AbortReason {
    Steps(String),
    Wallclock(String),
    ToolCalls(String),
    Replans(String),
    StackDepth(String),
    Subagents(String),
    Interrupted(String),
}

impl AbortReason {
    pub fn message(&self) -> &str {
        match self {
            AbortReason::Steps(m)
            | AbortReason::Wallclock(m)
            | AbortReason::ToolCalls(m)
            | AbortReason::Replans(m)
            | AbortReason::StackDepth(m)
            | AbortReason::Subagents(m)
            | AbortReason::Interrupted(m) => m,
        }
    }
}

/// The governor's verdict for a transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GovernDecision {
    Proceed,
    Abort(AbortReason),
    /// An interrupt requested a pause (suggest-only/effectful or explicit Pause).
    Pause(String),
}

/// Autonomy levels (§4.3). Effectful steps under `SuggestOnly` pause for
/// approval; `ReadOnly` forbids effects entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Autonomy {
    /// Full auto — effectful steps run without approval.
    #[default]
    FullAuto,
    /// Effectful steps pause awaiting approval.
    SuggestOnly,
    /// No effectful steps may run at all.
    ReadOnly,
}

/// Interrupts the host can inject between transitions (§4.3.2). Polled by the
/// driver before each transition.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Interrupt {
    Abort,
    Pause,
    Steer { instruction: String },
}

/// The single chokepoint. Holds the autonomy level and an optional pending
/// interrupt; `check` enforces every A.5 cap before a transition.
#[derive(Debug, Clone, Default)]
pub struct Governor {
    pub autonomy: Autonomy,
    pub pending_interrupt: Option<Interrupt>,
}

impl Governor {
    pub fn new(autonomy: Autonomy) -> Self {
        Self {
            autonomy,
            pending_interrupt: None,
        }
    }

    /// Inject an interrupt to be consumed on the next `check`.
    pub fn interrupt(&mut self, interrupt: Interrupt) {
        self.pending_interrupt = Some(interrupt);
    }

    /// Enforce all caps + poll interrupts. Called before every transition (K8).
    /// `now_ms` lets the caller (and tests) control the wall-clock.
    pub fn check(&mut self, state: &mut AgentState, now_ms: u64) -> GovernDecision {
        state.ledger.tick(now_ms);

        // 1. Poll interrupts first — an abort/pause beats any budget check.
        if let Some(interrupt) = self.pending_interrupt.take() {
            match interrupt {
                Interrupt::Abort => {
                    return GovernDecision::Abort(AbortReason::Interrupted(
                        "host requested abort".to_string(),
                    ));
                }
                Interrupt::Pause => {
                    return GovernDecision::Pause("host requested pause".to_string());
                }
                Interrupt::Steer { instruction } => {
                    // Steering rewrites the objective; record and proceed.
                    state.steer.push(instruction);
                }
            }
        }

        // 2. Hard caps (K4: wallclock/steps/effect-counts).
        let b = &state.budget;
        let l = &state.ledger;
        if l.steps >= b.max_steps {
            return GovernDecision::Abort(AbortReason::Steps(format!(
                "step cap reached ({}/{})",
                l.steps, b.max_steps
            )));
        }
        if b.max_wallclock_ms > 0 && l.elapsed_ms >= b.max_wallclock_ms {
            return GovernDecision::Abort(AbortReason::Wallclock(format!(
                "wallclock cap reached ({}ms/{}ms)",
                l.elapsed_ms, b.max_wallclock_ms
            )));
        }
        if l.tool_calls >= b.max_tool_calls {
            return GovernDecision::Abort(AbortReason::ToolCalls(format!(
                "tool-call cap reached ({}/{})",
                l.tool_calls, b.max_tool_calls
            )));
        }
        if l.replans > b.max_replans {
            return GovernDecision::Abort(AbortReason::Replans(format!(
                "replan cap reached ({}/{})",
                l.replans, b.max_replans
            )));
        }
        if l.stack_depth > b.max_stack_depth {
            return GovernDecision::Abort(AbortReason::StackDepth(format!(
                "stack-depth cap reached ({}/{})",
                l.stack_depth, b.max_stack_depth
            )));
        }
        if l.subagents_total > b.max_subagents {
            return GovernDecision::Abort(AbortReason::Subagents(format!(
                "subagent cap reached ({}/{})",
                l.subagents_total, b.max_subagents
            )));
        }
        GovernDecision::Proceed
    }

    /// Whether an effectful step may run under the current autonomy. `false`
    /// means the driver must pause for approval (SuggestOnly) or skip (ReadOnly).
    pub fn may_run_effect(&self) -> EffectAuthorization {
        match self.autonomy {
            Autonomy::FullAuto => EffectAuthorization::Allow,
            Autonomy::SuggestOnly => EffectAuthorization::NeedsApproval,
            Autonomy::ReadOnly => EffectAuthorization::Forbidden,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectAuthorization {
    Allow,
    NeedsApproval,
    Forbidden,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::{RunId, SessionId};

    fn state_with_budget(budget: Budget) -> AgentState {
        let mut s = AgentState::new(SessionId::new(), RunId::new(), "obj".to_string());
        s.budget = budget;
        s
    }

    #[test]
    fn governor_aborts_on_step_cap_with_structured_reason() {
        let mut gov = Governor::default();
        let mut state = state_with_budget(Budget {
            max_steps: 2,
            ..Budget::default()
        });
        state.ledger.steps = 2;
        match gov.check(&mut state, 1000) {
            GovernDecision::Abort(AbortReason::Steps(_)) => {}
            other => panic!("expected step-cap abort, got {other:?}"),
        }
    }

    #[test]
    fn governor_aborts_on_wallclock() {
        let mut gov = Governor::default();
        let mut state = state_with_budget(Budget {
            max_wallclock_ms: 100,
            ..Budget::default()
        });
        // started at t=0, now at t=500 → elapsed 500 > 100.
        state.ledger.tick(0);
        match gov.check(&mut state, 500) {
            GovernDecision::Abort(AbortReason::Wallclock(_)) => {}
            other => panic!("expected wallclock abort, got {other:?}"),
        }
    }

    #[test]
    fn governor_abort_interrupt_beats_budget() {
        let mut gov = Governor::default();
        gov.interrupt(Interrupt::Abort);
        let mut state = state_with_budget(Budget::default());
        assert!(matches!(
            gov.check(&mut state, 0),
            GovernDecision::Abort(AbortReason::Interrupted(_))
        ));
    }

    #[test]
    fn suggest_only_needs_approval_for_effects() {
        let gov = Governor::new(Autonomy::SuggestOnly);
        assert_eq!(gov.may_run_effect(), EffectAuthorization::NeedsApproval);
        let gov = Governor::new(Autonomy::ReadOnly);
        assert_eq!(gov.may_run_effect(), EffectAuthorization::Forbidden);
        let gov = Governor::new(Autonomy::FullAuto);
        assert_eq!(gov.may_run_effect(), EffectAuthorization::Allow);
    }
}
