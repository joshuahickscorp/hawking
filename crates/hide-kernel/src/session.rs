use crate::machine::state::Phase;
use crate::plan::schema::Plan;
use hide_core::ids::{RunId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionProjection {
    pub session_id: SessionId,
    pub active_run: Option<RunId>,
    pub phase: Option<Phase>,
    pub plan: Option<Plan>,
    pub transcript: Vec<String>,
    pub open_files: Vec<String>,
    pub errors: Vec<String>,
}

impl SessionProjection {
    pub fn empty(session_id: SessionId) -> Self {
        Self {
            session_id,
            active_run: None,
            phase: None,
            plan: None,
            transcript: Vec::new(),
            open_files: Vec::new(),
            errors: Vec::new(),
        }
    }
}
