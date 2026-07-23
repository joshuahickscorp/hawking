//! ACP session surfaces (spec-derived): create/load a session, prompt a turn,
//! and stream agent output as session updates.
//!
//! Spec-derived (ACP): `session/new` creates a session (returning a `sessionId`);
//! `session/load` resumes one; `session/prompt` runs a turn over a list of
//! content blocks and finishes with a `stopReason`. While a turn runs, the agent
//! pushes `session/update` notifications, each a `SessionNotification` carrying a
//! `sessionId` and an `update` tagged on `sessionUpdate`. Only the public wire
//! shape is mirrored; no proprietary source is copied.

use serde::{Deserialize, Serialize};

use crate::content::ContentBlock;
use crate::ids::AcpSessionId;
use crate::tool_call::{ToolCall, ToolCallUpdate};

/// An MCP server the client asks the agent to connect for a session. Carried
/// opaquely by the boundary (HIDE wires it through its own tool registry).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct McpServer {
    pub name: String,
    pub command: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
}

/// `session/new` params.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpNewSessionRequest {
    /// The working directory the session runs against.
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServer>,
}

/// `session/new` result: the freshly minted session id.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpNewSessionResponse {
    pub session_id: AcpSessionId,
}

/// `session/load` params: resume an existing session.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpLoadSessionRequest {
    pub session_id: AcpSessionId,
    pub cwd: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mcp_servers: Vec<McpServer>,
}

/// `session/prompt` params: run a turn over these content blocks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpPromptRequest {
    pub session_id: AcpSessionId,
    pub prompt: Vec<ContentBlock>,
}

/// Why a turn stopped. Spec-derived (ACP) `StopReason`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    MaxTurnRequests,
    Refusal,
    Cancelled,
}

/// `session/prompt` result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcpPromptResponse {
    pub stop_reason: StopReason,
}

/// Priority of a plan entry. Spec-derived (ACP) `PlanEntryPriority`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryPriority {
    High,
    Medium,
    Low,
}

/// Status of a plan entry. Spec-derived (ACP) `PlanEntryStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanEntryStatus {
    Pending,
    InProgress,
    Completed,
}

/// One item in the agent's plan.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanEntry {
    pub content: String,
    pub priority: PlanEntryPriority,
    pub status: PlanEntryStatus,
}

/// The agent's whole plan, as an ACP `plan` update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AcpPlan {
    pub entries: Vec<PlanEntry>,
}

/// One streamed change to a session. Spec-derived (ACP): internally tagged on
/// `sessionUpdate`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "sessionUpdate", rename_all = "snake_case")]
pub enum SessionUpdate {
    /// A chunk of the agent's visible reply.
    AgentMessageChunk { content: ContentBlock },
    /// A chunk of the agent's reasoning (shown separately by the client).
    AgentThoughtChunk { content: ContentBlock },
    /// An echo of user input inside the session record.
    UserMessageChunk { content: ContentBlock },
    /// A tool call the agent started.
    ToolCall(ToolCall),
    /// Progress or completion for a tool call.
    ToolCallUpdate(ToolCallUpdate),
    /// The agent's plan.
    Plan(AcpPlan),
}

impl SessionUpdate {
    /// The wire `sessionUpdate` tag.
    pub fn tag(&self) -> &'static str {
        match self {
            SessionUpdate::AgentMessageChunk { .. } => "agent_message_chunk",
            SessionUpdate::AgentThoughtChunk { .. } => "agent_thought_chunk",
            SessionUpdate::UserMessageChunk { .. } => "user_message_chunk",
            SessionUpdate::ToolCall(_) => "tool_call",
            SessionUpdate::ToolCallUpdate(_) => "tool_call_update",
            SessionUpdate::Plan(_) => "plan",
        }
    }
}

/// A `session/update` notification: which session, and what changed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SessionNotification {
    pub session_id: AcpSessionId,
    pub update: SessionUpdate,
}

impl SessionNotification {
    pub fn new(session_id: AcpSessionId, update: SessionUpdate) -> Self {
        Self { session_id, update }
    }
}
