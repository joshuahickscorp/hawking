use crate::machine::state::AgentState;
use hide_core::event::Event;
use hide_core::ids::SessionId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentCheckpoint {
    pub session_id: SessionId,
    pub seq: u64,
    pub state: AgentState,
    pub source_event: Option<Event>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReplayRequest {
    pub session_id: SessionId,
    pub target_seq: u64,
    pub live_resume: bool,
}
