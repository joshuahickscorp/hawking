use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchJob {
    pub id: String,
    pub job_ids: Vec<String>,
    pub schedule: BatchSchedule,
    pub report_on_wake: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchSchedule {
    pub earliest_start_ms: Option<u64>,
    pub require_idle: bool,
    pub require_power: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WakeReport {
    pub batch_id: String,
    pub completed: Vec<String>,
    pub failed: Vec<String>,
    pub still_running: Vec<String>,
    pub summary: String,
}
