use hide_core::ids::{now_ms, RunId};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hypothesis {
    pub id: String,
    pub statement: String,
    pub source_claim_ids: Vec<String>,
    pub status: HypothesisStatus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisStatus {
    Proposed,
    Testing,
    Supported,
    Refuted,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperimentRun {
    pub id: RunId,
    pub hypothesis_id: String,
    pub command: Vec<String>,
    pub params: BTreeMap<String, String>,
    pub metrics: BTreeMap<String, f64>,
    pub artifacts: Vec<String>,
    pub started_at_ms: u64,
}

impl ExperimentRun {
    pub fn new(hypothesis_id: impl Into<String>, command: Vec<String>) -> Self {
        Self {
            id: RunId::new(),
            hypothesis_id: hypothesis_id.into(),
            command,
            params: BTreeMap::new(),
            metrics: BTreeMap::new(),
            artifacts: Vec::new(),
            started_at_ms: now_ms(),
        }
    }
}
