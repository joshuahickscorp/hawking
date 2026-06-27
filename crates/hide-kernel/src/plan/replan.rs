use crate::plan::schema::{Plan, PlanStatus};
use hide_core::ids::StepId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplanRequest {
    pub failed_step: Option<StepId>,
    pub reason: String,
    pub local_only: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplanResult {
    pub plan: Plan,
    pub changed_steps: Vec<StepId>,
}

pub fn supersede(mut plan: Plan) -> Plan {
    plan.status = PlanStatus::Superseded;
    plan
}
