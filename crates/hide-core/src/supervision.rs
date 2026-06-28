use crate::ids::TimestampMs;
use crate::runtime::RuntimeSupervisorState;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessSpec {
    pub name: String,
    pub argv: Vec<String>,
    pub cwd: Option<String>,
    pub env: BTreeMap<String, String>,
    pub health_url: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessStatus {
    pub name: String,
    pub pid: Option<u32>,
    pub state: RuntimeSupervisorState,
    pub started_at_ms: Option<TimestampMs>,
    pub restarts: u32,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackoffPolicy {
    pub delays_ms: Vec<u64>,
    pub max_restarts_per_window: u32,
    pub window_ms: u64,
}

impl Default for BackoffPolicy {
    fn default() -> Self {
        Self {
            delays_ms: vec![1_000, 2_000, 4_000, 8_000, 30_000],
            max_restarts_per_window: 5,
            window_ms: 5 * 60 * 1000,
        }
    }
}
