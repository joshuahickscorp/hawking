//! ACP permission requests (spec-derived).
//!
//! Spec-derived (ACP): before an effectful action the agent sends
//! `session/request_permission` with the `sessionId`, the `toolCall` it wants to
//! run, and a list of `options` the user may pick. Each option has an
//! `optionId`, a display `name`, and a `kind` (allow/reject, once/always). The
//! client replies with an outcome: a selected option, or a cancellation. Only
//! the public wire shape is mirrored; no proprietary source is copied.

use serde::{Deserialize, Serialize};

use crate::ids::AcpSessionId;
use crate::tool_call::ToolCallUpdate;

/// The category of a permission option. Spec-derived (ACP) `PermissionOptionKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionOptionKind {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
}

/// One choice the user may pick when granting or denying permission.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PermissionOption {
    pub option_id: String,
    pub name: String,
    pub kind: PermissionOptionKind,
}

impl PermissionOption {
    /// Build an option with a stable id derived from its kind.
    pub fn new(kind: PermissionOptionKind, name: impl Into<String>) -> Self {
        let option_id = match kind {
            PermissionOptionKind::AllowOnce => "allow_once",
            PermissionOptionKind::AllowAlways => "allow_always",
            PermissionOptionKind::RejectOnce => "reject_once",
            PermissionOptionKind::RejectAlways => "reject_always",
        }
        .to_string();
        Self {
            option_id,
            name: name.into(),
            kind,
        }
    }
}

/// The agent's `session/request_permission` request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RequestPermissionRequest {
    pub session_id: AcpSessionId,
    pub tool_call: ToolCallUpdate,
    pub options: Vec<PermissionOption>,
}

/// How the client resolved a permission request. Spec-derived (ACP): tagged on
/// `outcome`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PermissionOutcome {
    /// The user chose one of the offered options.
    Selected {
        #[serde(rename = "optionId")]
        option_id: String,
    },
    /// The request was cancelled (for example the turn was interrupted).
    Cancelled,
}

/// The client's reply to a permission request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RequestPermissionResponse {
    pub outcome: PermissionOutcome,
}

/// The standard four-option grant/deny menu HIDE offers for an approvable
/// action. Ordered allow-once, allow-always, reject-once, reject-always.
pub fn standard_options() -> Vec<PermissionOption> {
    vec![
        PermissionOption::new(PermissionOptionKind::AllowOnce, "Allow once"),
        PermissionOption::new(PermissionOptionKind::AllowAlways, "Allow always"),
        PermissionOption::new(PermissionOptionKind::RejectOnce, "Reject"),
        PermissionOption::new(PermissionOptionKind::RejectAlways, "Reject always"),
    ]
}
