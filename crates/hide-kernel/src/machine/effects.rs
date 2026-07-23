//! Effect helpers + the Live/Replay mode switch (bible tenet K5).
//!
//! In `Replay` mode effects DO NOT run: the driver folds the recorded
//! `Observation` outcomes from the event log instead of re-firing the `Action`.

use hide_core::event::{AgentStateEvent, EventClass, EventSource, NewEvent};
use hide_core::ids::{EventId, RunId, SessionId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Execution mode (K5). `Live` runs effects; `Replay` folds recorded outcomes
/// and never re-fires actions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    #[default]
    Live,
    Replay,
}

impl Mode {
    pub fn is_replay(&self) -> bool {
        matches!(self, Mode::Replay)
    }
}

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

/// An `Action`-class agent event (the effect boundary). Its outcome is recorded
/// as a paired `Observation` carrying `cause` = this action's id.
pub fn action_event(
    session_id: SessionId,
    run_id: RunId,
    kind: impl Into<String>,
    value: Value,
) -> NewEvent {
    NewEvent::of(session_id, EventSource::Agent, kind, value)
        .with_run(run_id)
        .with_class(EventClass::Action)
}

/// An `Observation`-class event (the recorded outcome of an action), carrying the
/// causing action's event id (OpenHands-style pairing; replay folds these — T3).
pub fn observation_event(
    session_id: SessionId,
    run_id: RunId,
    kind: impl Into<String>,
    cause: EventId,
    value: Value,
) -> NewEvent {
    NewEvent::of(session_id, EventSource::Agent, kind, value)
        .with_run(run_id)
        .with_cause(cause)
        .with_class(EventClass::Observation)
}
