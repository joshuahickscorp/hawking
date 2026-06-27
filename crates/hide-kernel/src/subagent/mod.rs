use hide_core::ids::{RunId, SessionId};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentSpec {
    pub name: String,
    pub objective: String,
    pub isolation: IsolationMode,
    pub max_steps: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IsolationMode {
    SharedReadOnly,
    Worktree,
    FreshContext,
    MicroVm,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentHandle {
    pub session_id: SessionId,
    pub run_id: RunId,
    pub spec: SubagentSpec,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubagentReturn {
    pub handle: SubagentHandle,
    pub summary: String,
    pub changed_files: Vec<String>,
    pub confidence: u8,
}
