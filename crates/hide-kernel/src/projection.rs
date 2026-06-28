use crate::session::SessionProjection;
use hide_core::event::{AgentStateEvent, ErrorEvent, Event, PlanEvent, UserIntentEvent};
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
            // Open payload: dispatch on the dotted kind, then read the typed
            // view off the `Value` payload (`payload_as`).
            match event.kind.as_str() {
                "user.intent" => {
                    if let Some(intent) = event.payload_as::<UserIntentEvent>() {
                        projection
                            .transcript
                            .push(format!("intent:{} {}", intent.intent, intent.args));
                    }
                }
                "agent.phase" => {
                    if let Some(state) = event.payload_as::<AgentStateEvent>() {
                        projection
                            .transcript
                            .push(format!("agent:{} {}", state.phase, state.detail));
                        // The driver emits the snake_case wire name (Phase::wire_name),
                        // so this snake_case match round-trips correctly (the prior
                        // PascalCase `{:?}` mismatch is fixed). Parse via serde so the
                        // mapping stays in sync with the enum's rename_all.
                        if let Ok(phase) = serde_json::from_value::<crate::machine::state::Phase>(
                            serde_json::Value::String(state.phase.clone()),
                        ) {
                            projection.phase = Some(phase);
                        }
                    }
                }
                "verify.result" => {
                    if let Some(v) = event.payload_as::<crate::verify::oracle::Verdict>() {
                        projection
                            .transcript
                            .push(format!("verify:{} {:?}", v.oracle, v.status));
                    }
                }
                "run.aborted" => {
                    projection
                        .errors
                        .push(format!("run aborted: {}", event.payload));
                }
                "plan.created" => {
                    if let Some(plan_event) = event.payload_as::<PlanEvent>() {
                        if let Some(plan) = &plan_event.plan {
                            projection
                                .transcript
                                .push(format!("plan:{}", plan_event.action));
                            projection.plan = serde_json::from_value(plan.clone()).ok();
                        }
                    }
                }
                "error" => {
                    if let Some(error) = event.payload_as::<ErrorEvent>() {
                        projection.errors.push(error.message);
                    }
                }
                _ => {}
            }
        }
        Ok(projection)
    }
}

pub fn empty_projection(session_id: SessionId) -> SessionProjection {
    SessionProjection::empty(session_id)
}
