use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptModule {
    pub name: String,
    pub schema_version: u32,
    pub input_schema: Value,
    pub output_schema: Value,
    pub template: String,
    pub history: Vec<PromptVersion>,
    pub metric: OptimizationMetric,
    pub min_eval_n: u16,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PromptVersion {
    pub version: u32,
    pub template: String,
    pub score: f64,
    pub eval_n: u16,
    pub promoted_at_ms: u64,
    pub promoted_by: PromotedBy,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationMetric {
    Accuracy,
    AcceptRate,
    LatencyMs,
    OraclePassRate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotedBy {
    Human,
    AutoDspy { optimizer_run_id: String },
    Adas { run_id: String },
}

impl PromptModule {
    pub fn promote(&mut self, version: PromptVersion) -> bool {
        if version.eval_n < self.min_eval_n {
            return false;
        }
        if self
            .history
            .last()
            .map_or(true, |current| version.score > current.score)
        {
            self.template = version.template.clone();
            self.history.push(version);
            true
        } else {
            false
        }
    }
}
