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
        /// Optional per-hunk target. `None` (the default, backward compatible)
        /// means the whole diff; `Some(hunk_id)` accepts exactly one hunk.
        #[serde(default)]
        hunk_id: Option<String>,
    },
    RejectDiff {
        run_id: RunId,
        diff_id: String,
        /// Optional per-hunk target. `None` (the default, backward compatible)
        /// reverts the whole diff; `Some(hunk_id)` reverts exactly one hunk.
        #[serde(default)]
        hunk_id: Option<String>,
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

/// The answer to one intent. THREE outcomes, not two: `accepted && !held` is done,
/// `accepted && held` means the intent was recorded but its effect is parked at an
/// approval gate and has NOT run, and `!accepted` is a refusal. A caller that reads
/// only `accepted` would show a held destructive command as finished, so `held` is a
/// field of its own rather than prose inside `message`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentAck {
    pub accepted: bool,
    /// Recorded, but the effect awaits a human approve/deny. Additive: an older client
    /// that never sends it deserializes as `false` (the previous two-state meaning).
    #[serde(default)]
    pub held: bool,
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
        /// The id of the RECORDED event this step is, when there is one. A tool
        /// call id is not an `EventId`, so a client that addressed a boundary
        /// (`fork_session`, `checkpoint_create`) with `call_id` was always
        /// resolved as `NotFound`. Streamed process output is not a recorded
        /// event, so it carries `None` and no boundary verb can address it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        event_id: Option<String>,
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
