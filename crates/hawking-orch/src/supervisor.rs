use hide_core::runtime::RuntimeSupervisorState;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeLock {
    pub pid: Option<u32>,
    pub port: u16,
    pub model_id: String,
    pub started_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeSupervisorStatus {
    pub state: RuntimeSupervisorState,
    pub lock: Option<RuntimeLock>,
    pub consecutive_failures: u32,
    pub last_error: Option<String>,
}

impl RuntimeSupervisorStatus {
    pub fn down() -> Self {
        Self {
            state: RuntimeSupervisorState::Down,
            lock: None,
            consecutive_failures: 0,
            last_error: None,
        }
    }
}
