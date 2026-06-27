use hide_core::ids::RunId;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionFeedback {
    pub run_id: RunId,
    pub step_id: Option<String>,
    pub reward: f32,
    pub signal: FeedbackSignal,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackSignal {
    BuildPassed,
    TestPassed,
    TestFailed,
    UserAccepted,
    UserRejected,
    LatencyImproved,
    RegressionDetected,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RlefTrajectory {
    pub id: String,
    pub feedback: Vec<ExecutionFeedback>,
    pub total_reward: f32,
}

impl RlefTrajectory {
    pub fn recompute_reward(&mut self) {
        self.total_reward = self.feedback.iter().map(|f| f.reward).sum();
    }
}
