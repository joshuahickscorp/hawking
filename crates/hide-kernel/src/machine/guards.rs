use crate::machine::state::AgentState;
use crate::plan::dag::PlanDag;

pub fn budget_allows_step(state: &AgentState) -> bool {
    state.ledger.within(&state.budget)
}

pub fn plan_has_ready_step(state: &AgentState) -> bool {
    state
        .plan
        .as_ref()
        .map(|plan| !PlanDag::ready_steps(plan).is_empty())
        .unwrap_or(false)
}
