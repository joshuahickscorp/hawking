use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfidenceSignal {
    pub entropy: Option<f32>,
    pub max_probability: Option<f32>,
    pub margin: Option<f32>,
    pub self_certainty: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstraintRequest {
    pub name: String,
    pub schema: String,
    pub fallback_json_mode: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DraftControl {
    pub enabled: bool,
    pub proposer: String,
    pub verify_greedy_lossless: bool,
}
