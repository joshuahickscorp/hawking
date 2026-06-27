use crate::machine::effects::{custom_agent_event, state_event};
use crate::machine::guards::{budget_allows_step, plan_has_ready_step};
use crate::machine::state::{AgentState, Phase};
use crate::plan::dag::PlanDag;
use crate::plan::schema::{Plan, StepStatus};
use crate::verify::oracle::{Verdict, VerdictStatus};
use hide_core::event::{EventPayload, EventSource, NewEvent, PlanEvent};
use hide_core::persistence::DynEventLog;
use hide_core::{HideError, Result};
use serde_json::json;

pub struct AgentDriver {
    events: DynEventLog,
}

impl AgentDriver {
    pub fn new(events: DynEventLog) -> Self {
        Self { events }
    }

    pub async fn step(&self, state: &mut AgentState) -> Result<()> {
        if !budget_allows_step(state) {
            state.phase = Phase::Aborted;
            self.emit_phase(state, "budget exhausted").await?;
            return Ok(());
        }
        state.ledger.consume_step();
        match state.phase {
            Phase::Intake => {
                state.phase = Phase::Plan;
                self.emit_phase(state, "intake complete").await?;
            }
            Phase::Plan => {
                let plan = Plan::single_step("Scaffold architecture", &state.objective);
                self.events
                    .append(NewEvent {
                        session_id: state.session_id.clone(),
                        run_id: Some(state.run_id.clone()),
                        parent: None,
                        source: EventSource::Agent,
                        kind: "plan.created".into(),
                        payload: EventPayload::Plan(PlanEvent {
                            action: "created".to_string(),
                            step_id: None,
                            plan: Some(serde_json::to_value(&plan)?),
                        }),
                        redactions: Vec::new(),
                    })
                    .await?;
                state.plan = Some(plan);
                state.phase = Phase::SelectStep;
            }
            Phase::SelectStep => {
                if !plan_has_ready_step(state) {
                    state.phase = Phase::Finalize;
                    self.emit_phase(state, "no ready steps remain").await?;
                } else {
                    let plan = state.plan.as_ref().ok_or_else(|| {
                        HideError::InvalidState("select without plan".to_string())
                    })?;
                    state.cursor = PlanDag::ready_steps(plan).first().cloned();
                    state.mark_cursor(StepStatus::Running);
                    state.phase = Phase::Act;
                    self.emit_phase(state, "selected ready step").await?;
                }
            }
            Phase::Act => {
                self.events
                    .append(custom_agent_event(
                        state.session_id.clone(),
                        state.run_id.clone(),
                        "agent.action.stubbed",
                        json!({ "step_id": state.cursor, "note": "tool/model action boundary scaffolded" }),
                    ))
                    .await?;
                state.phase = Phase::Observe;
            }
            Phase::Observe => {
                self.emit_phase(state, "observation recorded as data")
                    .await?;
                state.phase = Phase::Verify;
            }
            Phase::Verify => {
                state.last_verdict = Some(Verdict {
                    status: VerdictStatus::Pass,
                    score: 1.0,
                    oracle: "stub".to_string(),
                    detail: "placeholder verification passed".to_string(),
                });
                state.mark_cursor(StepStatus::Completed);
                state.cursor = None;
                state.phase = Phase::SelectStep;
                self.emit_phase(state, "verification passed").await?;
            }
            Phase::Finalize => {
                state.phase = Phase::Done;
                self.emit_phase(state, "run finalized").await?;
            }
            Phase::Repair | Phase::Replan | Phase::Done | Phase::Aborted | Phase::Paused => {
                self.emit_phase(state, "phase has no scaffold transition")
                    .await?;
            }
        }
        Ok(())
    }

    async fn emit_phase(&self, state: &AgentState, detail: impl Into<String>) -> Result<()> {
        self.events
            .append(state_event(
                state.session_id.clone(),
                state.run_id.clone(),
                format!("{:?}", state.phase),
                detail,
            ))
            .await?;
        Ok(())
    }
}
