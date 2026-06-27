use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationRequest {
    pub objective: String,
    pub changed_files: Vec<String>,
    pub assumptions: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimulationResult {
    pub predicted_failures: Vec<String>,
    pub suggested_oracles: Vec<String>,
    pub confidence: f32,
}
