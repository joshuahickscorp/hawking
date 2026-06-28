use hide_core::event::{AgentStateEvent, EventSource, NewEvent};
use hide_core::ids::{RunId, SessionId};
use serde_json::Value;

pub fn state_event(
    session_id: SessionId,
    run_id: RunId,
    phase: impl Into<String>,
    detail: impl Into<String>,
) -> NewEvent {
    NewEvent::agent_state(
        session_id,
        run_id,
        AgentStateEvent {
            phase: phase.into(),
            detail: detail.into(),
        },
    )
}

pub fn custom_agent_event(
    session_id: SessionId,
    run_id: RunId,
    kind: &'static str,
    value: Value,
) -> NewEvent {
    NewEvent::of(session_id, EventSource::Agent, kind, value).with_run(run_id)
}
