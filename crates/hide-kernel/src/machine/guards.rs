//! Transition guards (bible ch.02 §4.4 / §4.5.2).

use crate::machine::state::AgentState;
use crate::plan::dag::PlanDag;
use crate::plan::schema::PlanStep;

/// The plan exists and is a DAG (acyclic). A cyclic plan must be replanned
/// (§4.5.2), never executed.
pub fn plan_is_acyclic(state: &AgentState) -> bool {
    state.plan.as_ref().map(PlanDag::acyclic).unwrap_or(false)
}

/// There is at least one ready step (deps satisfied ∧ pending).
pub fn plan_has_ready_step(state: &AgentState) -> bool {
    state
        .plan
        .as_ref()
        .map(|plan| !PlanDag::ready_steps(plan).is_empty())
        .unwrap_or(false)
}

/// The current cursor step still has repair budget left.
pub fn repairs_remaining(state: &AgentState) -> bool {
    state.cursor_repair_count() < state.budget.max_repairs
}

/// The current cursor step (if any).
pub fn cursor_step(state: &AgentState) -> Option<&PlanStep> {
    let (plan, cursor) = (state.plan.as_ref()?, state.cursor.as_ref()?);
    plan.step(cursor)
}

/// The current cursor step mutates the world (needs an autonomy/approval gate).
pub fn cursor_is_effectful(state: &AgentState) -> bool {
    cursor_step(state)
        .map(PlanStep::is_effectful)
        .unwrap_or(false)
}
