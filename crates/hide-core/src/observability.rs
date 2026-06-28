use crate::ids::{now_ms, RunId, SessionId, TimestampMs};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LogRecord {
    pub ts_ms: TimestampMs,
    pub level: LogLevel,
    pub target: String,
    pub message: String,
    pub session_id: Option<SessionId>,
    pub run_id: Option<RunId>,
    pub fields: BTreeMap<String, String>,
}

impl LogRecord {
    pub fn info(target: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            ts_ms: now_ms(),
            level: LogLevel::Info,
            target: target.into(),
            message: message.into(),
            session_id: None,
            run_id: None,
            fields: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricSample {
    pub ts_ms: TimestampMs,
    pub name: String,
    pub value: f64,
    pub labels: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthReport {
    pub component: String,
    pub status: HealthStatus,
    pub checks: Vec<HealthCheck>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Ok,
    Degraded,
    Failed,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthStatus,
    pub detail: String,
}
