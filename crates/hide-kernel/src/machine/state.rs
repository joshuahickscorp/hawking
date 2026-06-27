use crate::govern::{Budget, BudgetLedger};
use crate::plan::schema::{Plan, StepStatus};
use crate::verify::oracle::Verdict;
use hide_core::ids::{RunId, SessionId, StepId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Intake,
    Plan,
    SelectStep,
    Act,
    Observe,
    Verify,
    Repair,
    Replan,
    Finalize,
    Done,
    Aborted,
    Paused,
}

impl Phase {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Phase::Done | Phase::Aborted)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentState {
    pub session_id: SessionId,
    pub run_id: RunId,
    pub objective: String,
    pub phase: Phase,
    pub plan: Option<Plan>,
    pub cursor: Option<StepId>,
    pub budget: Budget,
    pub ledger: BudgetLedger,
    pub last_verdict: Option<Verdict>,
    pub repair_count: BTreeMap<StepId, u8>,
    pub pending_approval: Option<String>,
}

impl AgentState {
    pub fn new(session_id: SessionId, run_id: RunId, objective: String) -> Self {
        Self {
            session_id,
            run_id,
            objective,
            phase: Phase::Intake,
            plan: None,
            cursor: None,
            budget: Budget::default(),
            ledger: BudgetLedger::default(),
            last_verdict: None,
            repair_count: BTreeMap::new(),
            pending_approval: None,
        }
    }

    pub fn mark_cursor(&mut self, status: StepStatus) {
        if let (Some(plan), Some(cursor)) = (&mut self.plan, &self.cursor) {
            if let Some(step) = plan.steps.iter_mut().find(|step| &step.id == cursor) {
                step.status = status;
            }
        }
    }
}
