use hide_core::ids::{PlanId, StepId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Plan {
    pub id: PlanId,
    pub title: String,
    pub objective: String,
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
}

impl Plan {
    pub fn single_step(title: impl Into<String>, objective: impl Into<String>) -> Self {
        Self {
            id: PlanId::new(),
            title: title.into(),
            objective: objective.into(),
            steps: vec![PlanStep {
                id: StepId::new(),
                title: "Architecture scaffold pass".to_string(),
                kind: StepKind::Implementation,
                dependencies: Vec::new(),
                status: StepStatus::Pending,
                tool_hint: None,
                acceptance_criteria: vec![
                    "folder/module structure exists".to_string(),
                    "core contracts compile".to_string(),
                ],
            }],
            status: PlanStatus::Active,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: StepId,
    pub title: String,
    pub kind: StepKind,
    pub dependencies: Vec<StepId>,
    pub status: StepStatus,
    pub tool_hint: Option<String>,
    pub acceptance_criteria: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    Research,
    Implementation,
    ToolCall,
    Verification,
    Repair,
    Subagent,
    Summary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepStatus {
    Pending,
    Ready,
    Running,
    Blocked,
    Completed,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStatus {
    Draft,
    Active,
    Completed,
    Failed,
    Superseded,
}
