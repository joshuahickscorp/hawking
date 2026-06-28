use crate::ids::{EventId, RunId, SessionId};
use crate::types::BlobRef;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum Intent {
    SubmitTurn {
        session_id: SessionId,
        text: String,
        attachments: Vec<BlobRef>,
    },
    CancelRun {
        run_id: RunId,
    },
    PauseRun {
        run_id: RunId,
    },
    ResumeRun {
        run_id: RunId,
    },
    AcceptDiff {
        run_id: RunId,
        diff_id: String,
    },
    RejectDiff {
        run_id: RunId,
        diff_id: String,
    },
    ScrubToEvent {
        session_id: SessionId,
        event_id: EventId,
    },
    ForkSession {
        session_id: SessionId,
        at_event: EventId,
    },
    OpenFile {
        path: String,
        line: Option<u32>,
    },
    RunCommand {
        argv: Vec<String>,
        cwd: Option<String>,
    },
    Custom {
        name: String,
        payload: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentAck {
    pub accepted: bool,
    pub event_seq: Option<u64>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UiEvent {
    pub seq: u64,
    pub session_id: Option<SessionId>,
    pub kind: UiEventKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum UiEventKind {
    ProjectionPatch {
        projection: String,
        patch: Value,
    },
    TokenBatch {
        stream_id: String,
        text: String,
    },
    RuntimeStatus {
        status: String,
        detail: Option<String>,
    },
    ToolProgress {
        call_id: String,
        message: String,
    },
    SecurityGate {
        gate: String,
        message: String,
    },
    Error {
        code: String,
        message: String,
    },
    Custom(Value),
}
