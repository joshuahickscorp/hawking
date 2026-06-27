//! Headless HIDE agent kernel.
//!
//! The kernel is the deterministic brain above the model: sessions, plan-as-data,
//! budget governance, verification boundaries, and replay-safe event emission.

pub mod checkpoint;
pub mod cooperate;
pub mod govern;
pub mod machine;
pub mod plan;
pub mod projection;
pub mod runtime_client;
pub mod search;
pub mod session;
pub mod skills;
pub mod subagent;
pub mod tools;
pub mod verify;

use crate::machine::driver::AgentDriver;
use crate::machine::state::AgentState;
use hide_core::event::{EventPayload, EventSource, NewEvent, UserIntentEvent};
use hide_core::ids::{RunId, SessionId};
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use serde_json::json;

pub struct AgentKernel {
    events: DynEventLog,
}

impl AgentKernel {
    pub fn new(events: DynEventLog) -> Self {
        Self { events }
    }

    pub async fn start_run(
        &self,
        session_id: SessionId,
        objective: impl Into<String>,
    ) -> Result<AgentState> {
        let objective = objective.into();
        let run_id = RunId::new();
        self.events
            .append(NewEvent {
                session_id: session_id.clone(),
                run_id: Some(run_id.clone()),
                parent: None,
                source: EventSource::User,
                kind: "user.intent.submit_turn".into(),
                payload: EventPayload::UserIntent(UserIntentEvent {
                    intent: "submit_turn".to_string(),
                    args: json!({ "objective": objective }),
                }),
                redactions: Vec::new(),
            })
            .await?;
        Ok(AgentState::new(session_id, run_id, objective))
    }

    pub async fn step(&self, state: &mut AgentState) -> Result<()> {
        AgentDriver::new(self.events.clone()).step(state).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::InMemoryEventLog;
    use std::sync::Arc;

    #[tokio::test]
    async fn kernel_can_drive_minimal_run_to_done() {
        let log = Arc::new(InMemoryEventLog::new());
        let kernel = AgentKernel::new(log.clone());
        let mut state = kernel
            .start_run(SessionId::new(), "scaffold the thing")
            .await
            .unwrap();
        for _ in 0..12 {
            if state.phase.is_terminal() {
                break;
            }
            kernel.step(&mut state).await.unwrap();
        }
        assert!(state.phase.is_terminal());
        assert!(log.len() >= 5);
    }
}
