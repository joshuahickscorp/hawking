use hide_core::event::{AgentStateEvent, EventPayload, EventSource, NewEvent};
use hide_core::ids::{RunId, SessionId};
use serde_json::Value;

pub fn state_event(
    session_id: SessionId,
    run_id: RunId,
    phase: impl Into<String>,
    detail: impl Into<String>,
) -> NewEvent {
    NewEvent {
        session_id,
        run_id: Some(run_id),
        parent: None,
        source: EventSource::Agent,
        kind: "agent.phase".into(),
        payload: EventPayload::AgentState(AgentStateEvent {
            phase: phase.into(),
            detail: detail.into(),
        }),
        redactions: Vec::new(),
    }
}

pub fn custom_agent_event(
    session_id: SessionId,
    run_id: RunId,
    kind: &'static str,
    value: Value,
) -> NewEvent {
    NewEvent {
        session_id,
        run_id: Some(run_id),
        parent: None,
        source: EventSource::Agent,
        kind: kind.into(),
        payload: EventPayload::Custom(value),
        redactions: Vec::new(),
    }
}
