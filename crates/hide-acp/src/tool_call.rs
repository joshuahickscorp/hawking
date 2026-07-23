//! ACP tool-call surfaces (spec-derived): tool calls, their updates, and the
//! content a tool call carries -- plain content, an edit/diff projection, and a
//! terminal projection.
//!
//! Spec-derived (ACP): a `tool_call` session update reports a call the agent
//! started (`toolCallId`, `title`, `kind`, `status`, `content`, `locations`,
//! `rawInput`); a `tool_call_update` reports progress or completion. Tool-call
//! content is a union on `type`: a `content` block, a `diff` (path + optional
//! old text + new text), or a `terminal` reference. Only the public wire shape
//! is mirrored; no proprietary source is copied.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::content::ContentBlock;
use crate::ids::{AcpTerminalId, AcpToolCallId};

/// What category of work a tool call performs. Spec-derived (ACP) `ToolKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolKind {
    Read,
    Edit,
    Delete,
    Move,
    Search,
    Execute,
    Think,
    Fetch,
    Other,
}

/// Lifecycle of a tool call. Spec-derived (ACP) `ToolCallStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

/// A file location a tool call touches (for the editor to reveal).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallLocation {
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
}

/// The content a tool call reports. Spec-derived (ACP): a union on `type`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolCallContent {
    /// A regular content block (text, resource, etc.).
    Content { content: ContentBlock },
    /// An edit projection: the change to one file. `oldText` is omitted for a
    /// new file. Spec-derived (ACP) diff content.
    Diff {
        path: String,
        #[serde(rename = "oldText", default, skip_serializing_if = "Option::is_none")]
        old_text: Option<String>,
        #[serde(rename = "newText")]
        new_text: String,
    },
    /// A terminal projection: a reference to a live terminal whose bytes stream
    /// over the ACP terminal channel.
    ///
    /// DEFERRED_MODEL_REQUIRED: the live byte stream (the actual running
    /// process behind `terminalId`) is produced by a model-bearing runtime and
    /// is not implemented or claimed here. The model-free boundary emits the
    /// reference and, alongside it, a text snapshot of any recorded chunk so a
    /// replay fixture stays lossless.
    Terminal {
        #[serde(rename = "terminalId")]
        terminal_id: AcpTerminalId,
    },
}

impl ToolCallContent {
    /// A plain-text content item.
    pub fn text(text: impl Into<String>) -> Self {
        ToolCallContent::Content {
            content: ContentBlock::text(text),
        }
    }

    /// The wire `type` tag.
    pub fn type_tag(&self) -> &'static str {
        match self {
            ToolCallContent::Content { .. } => "content",
            ToolCallContent::Diff { .. } => "diff",
            ToolCallContent::Terminal { .. } => "terminal",
        }
    }
}

/// A `tool_call` session update: a call the agent has started.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub tool_call_id: AcpToolCallId,
    pub title: String,
    pub kind: ToolKind,
    pub status: ToolCallStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub locations: Vec<ToolCallLocation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_input: Option<Value>,
}

/// A `tool_call_update`: progress or completion for an existing tool call. Every
/// field except the id is optional so an update carries only what changed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCallUpdate {
    pub tool_call_id: AcpToolCallId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<ToolKind>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<ToolCallContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<Value>,
}
