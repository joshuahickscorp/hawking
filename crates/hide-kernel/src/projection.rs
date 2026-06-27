use crate::session::SessionProjection;
use hide_core::event::{Event, EventPayload};
use hide_core::ids::SessionId;
use hide_core::Result;

pub trait ProjectionEngine: Send + Sync {
    fn fold(&self, initial: SessionProjection, events: &[Event]) -> Result<SessionProjection>;
}

#[derive(Default)]
pub struct BasicProjectionEngine;

impl ProjectionEngine for BasicProjectionEngine {
    fn fold(
        &self,
        mut projection: SessionProjection,
        events: &[Event],
    ) -> Result<SessionProjection> {
        for event in events {
            projection.session_id = event.session_id.clone();
            if let Some(run_id) = &event.run_id {
                projection.active_run = Some(run_id.clone());
            }
            match &event.payload {
                EventPayload::UserIntent(intent) => {
                    projection
                        .transcript
                        .push(format!("intent:{} {}", intent.intent, intent.args));
                }
                EventPayload::AgentState(state) => {
                    projection
                        .transcript
                        .push(format!("agent:{} {}", state.phase, state.detail));
                    projection.phase = match state.phase.as_str() {
                        "intake" => Some(crate::machine::state::Phase::Intake),
                        "plan" => Some(crate::machine::state::Phase::Plan),
                        "select_step" => Some(crate::machine::state::Phase::SelectStep),
                        "act" => Some(crate::machine::state::Phase::Act),
                        "observe" => Some(crate::machine::state::Phase::Observe),
                        "verify" => Some(crate::machine::state::Phase::Verify),
                        "repair" => Some(crate::machine::state::Phase::Repair),
                        "replan" => Some(crate::machine::state::Phase::Replan),
                        "finalize" => Some(crate::machine::state::Phase::Finalize),
                        "done" => Some(crate::machine::state::Phase::Done),
                        "aborted" => Some(crate::machine::state::Phase::Aborted),
                        "paused" => Some(crate::machine::state::Phase::Paused),
                        _ => projection.phase,
                    };
                }
                EventPayload::Plan(plan_event) => {
                    if let Some(plan) = &plan_event.plan {
                        projection
                            .transcript
                            .push(format!("plan:{}", plan_event.action));
                        projection.plan = serde_json::from_value(plan.clone()).ok();
                    }
                }
                EventPayload::Error(error) => projection.errors.push(error.message.clone()),
                _ => {}
            }
        }
        Ok(projection)
    }
}

pub fn empty_projection(session_id: SessionId) -> SessionProjection {
    SessionProjection::empty(session_id)
}
