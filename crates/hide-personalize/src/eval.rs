use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalCase {
    pub id: String,
    pub task: String,
    pub oracle: EvalOracle,
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvalOracle {
    Command { argv: Vec<String> },
    GoldenDiff { diff_hash: String },
    Regex { pattern: String },
    Human,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvalResult {
    pub case_id: String,
    pub passed: bool,
    pub score: f32,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdapterGateReport {
    pub base_accept_rate: f32,
    pub candidate_accept_rate: f32,
    pub min_delta: f32,
}

impl AdapterGateReport {
    pub fn passes(&self) -> bool {
        self.candidate_accept_rate >= self.base_accept_rate + self.min_delta
    }
}
