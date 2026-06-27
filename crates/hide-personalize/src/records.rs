use hide_core::ids::{now_ms, RunId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonalizationRecord {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub observed_at_ms: u64,
    pub task_type: TaskClass,
    pub prompt_hash: String,
    pub context_fingerprint: String,
    pub outcome: Outcome,
    pub diff_proposed: String,
    pub diff_accepted: String,
    pub latency_ms: u32,
    pub tok_s: Option<f32>,
    pub reject_reason: Option<String>,
    pub model_role: String,
    pub active_adapters: Vec<String>,
}

impl PersonalizationRecord {
    pub fn accepted(
        task_type: TaskClass,
        prompt_hash: impl Into<String>,
        diff: impl Into<String>,
    ) -> Self {
        let diff = diff.into();
        Self {
            session_id: SessionId::new(),
            run_id: None,
            observed_at_ms: now_ms(),
            task_type,
            prompt_hash: prompt_hash.into(),
            context_fingerprint: "unknown".to_string(),
            outcome: Outcome::Accepted,
            diff_proposed: diff.clone(),
            diff_accepted: diff,
            latency_ms: 0,
            tok_s: None,
            reject_reason: None,
            model_role: "hero".to_string(),
            active_adapters: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    EditCode,
    WriteTest,
    Refactor,
    ExplainCode,
    CommitMsg,
    Diagnose,
    Research,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Accepted,
    Modified { edit_distance_chars: u32 },
    Rejected,
    Abandoned,
}
