//! Items: the atomic events inside a turn (Bible sec 14).
//!
//! A [`Turn`](crate::model::Turn) is a list of [`Item`]s. An item's payload is
//! its [`ItemKind`], an adjacently tagged enum (`{ "kind": ..., "payload": ...
//! }`) whose variants cover every listed item kind: messages, reasoning,
//! plans, context receipts, tool calls and results, shell streams, patches and
//! diffs, approvals, verifications, artifacts, checkpoints, state capsules,
//! agent lifecycle, steering, interrupts, errors, completion, and blockers.
//!
//! Each payload is its own struct so the JSON Schema names it and a reader can
//! validate one item kind at a time.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::ids::{
    ApprovalId, ItemId, PlanId, StepId, ToolCallId, ToolId, TurnId, VerificationId,
};
use crate::model::{Artifact, Checkpoint, CompletionStatus, Risk, StateCapsuleRef};
use crate::plan::{Effect, Plan};

/// An attachment carried on a user message. Mirrors `hide_core::types::BlobRef`
/// so the compat bridge maps between them losslessly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Attachment {
    pub id: String,
    pub hash: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub media_type: Option<String>,
}

// -- item payloads ---------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct UserMessage {
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentMessage {
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ReasoningSummary {
    pub text: String,
}

/// One source folded into the model's context, with the tokens it cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ContextSource {
    pub name: String,
    #[serde(default)]
    pub uri: Option<String>,
    #[serde(default)]
    pub trust: Option<String>,
    #[serde(default)]
    pub token_estimate: Option<u64>,
}

/// A context receipt: the honest ledger of what was pulled into context for a
/// turn (Bible: bounded recall is the moat, so the receipt is first-class).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ContextReceipt {
    #[serde(default)]
    pub sources: Vec<ContextSource>,
    #[serde(default)]
    pub total_token_estimate: Option<u64>,
    #[serde(default)]
    pub note: Option<String>,
}

/// What changed about a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanMutationKind {
    AddStep,
    RemoveStep,
    UpdateStep,
    Reorder,
    Replace,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PlanMutation {
    pub plan: PlanId,
    pub kind: PlanMutationKind,
    #[serde(default)]
    pub step: Option<StepId>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolCall {
    pub call_id: ToolCallId,
    pub tool: ToolId,
    pub arguments: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolResult {
    pub call_id: ToolCallId,
    pub ok: bool,
    pub output: Value,
    #[serde(default)]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ShellChannel {
    Stdout,
    Stderr,
}

/// A chunk of a running command's output stream.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ShellStream {
    #[serde(default)]
    pub call_id: Option<ToolCallId>,
    pub channel: ShellChannel,
    pub chunk: String,
}

/// A proposed multi-file patch, carried as a unified diff.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Patch {
    pub patch_id: String,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub files: Vec<String>,
    pub unified_diff: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum DiffStatus {
    Proposed,
    Applied,
    Rejected,
    Reverted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DiffHunk {
    pub old_start: u32,
    pub old_lines: u32,
    pub new_start: u32,
    pub new_lines: u32,
    pub text: String,
}

/// A per-file diff with reviewable status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Diff {
    pub diff_id: String,
    pub path: String,
    #[serde(default)]
    pub hunks: Vec<DiffHunk>,
    pub status: DiffStatus,
}

/// A request for the user to approve an effectful action before it runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ApprovalRequest {
    pub request_id: ApprovalId,
    pub action: String,
    pub risk: Risk,
    #[serde(default)]
    pub effects: Vec<Effect>,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ApprovalDecision {
    Approved,
    Denied,
    Deferred,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ApprovalResult {
    pub request_id: ApprovalId,
    pub decision: ApprovalDecision,
    #[serde(default)]
    pub reason: Option<String>,
}

/// A request to grade a target against an oracle.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VerificationRequest {
    pub request_id: VerificationId,
    pub oracle: crate::ids::OracleId,
    #[serde(default)]
    pub target: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcome {
    Pass,
    Fail,
    Inconclusive,
}

/// The receipt returned when an oracle finishes grading.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VerificationReceipt {
    pub request_id: VerificationId,
    pub oracle: crate::ids::OracleId,
    pub outcome: VerificationOutcome,
    #[serde(default)]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentSpawn {
    pub agent: crate::ids::AgentId,
    pub role: String,
    pub objective: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AgentResult {
    pub agent: crate::ids::AgentId,
    pub outcome: CompletionStatus,
    #[serde(default)]
    pub summary: Option<String>,
}

/// A mid-turn steering directive from the user (sec 15 `turn/steer`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Steer {
    pub directive: String,
}

/// An interrupt of the running turn (sec 15 `turn/interrupt`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Interrupt {
    #[serde(default)]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ErrorItem {
    pub code: String,
    pub message: String,
}

/// The turn finished. Carries the terminal status and an optional summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Completion {
    pub status: CompletionStatus,
    #[serde(default)]
    pub summary: Option<String>,
}

/// The turn cannot proceed and needs the user. Distinct from an error: a
/// blocker is a solvable stall, not a failure.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Blocker {
    pub code: String,
    pub message: String,
    #[serde(default)]
    pub needs: Option<String>,
}

// -- the kind enum ---------------------------------------------------------

/// Every kind of item a turn can carry (Bible sec 14). Adjacently tagged so
/// each item is `{ "kind": "<name>", "payload": { ... } }`.
///
/// Note: the model lists `agent_message` in two places (the assistant's reply
/// and inter-agent messaging). Both map to the single [`ItemKind::AgentMessage`]
/// variant; the surrounding `agent_spawn`/`agent_result` items disambiguate the
/// coordination context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", content = "payload", rename_all = "snake_case")]
pub enum ItemKind {
    UserMessage(UserMessage),
    AgentMessage(AgentMessage),
    ReasoningSummary(ReasoningSummary),
    Plan(Plan),
    PlanMutation(PlanMutation),
    ContextReceipt(ContextReceipt),
    ToolCall(ToolCall),
    ToolResult(ToolResult),
    ShellStream(ShellStream),
    Patch(Patch),
    Diff(Diff),
    ApprovalRequest(ApprovalRequest),
    ApprovalResult(ApprovalResult),
    VerificationRequest(VerificationRequest),
    VerificationReceipt(VerificationReceipt),
    Artifact(Artifact),
    Checkpoint(Checkpoint),
    StateCapsule(StateCapsuleRef),
    AgentSpawn(AgentSpawn),
    AgentResult(AgentResult),
    Steer(Steer),
    Interrupt(Interrupt),
    Error(ErrorItem),
    Completion(Completion),
    Blocker(Blocker),
}

impl ItemKind {
    /// The wire tag for this kind (the value of the `"kind"` field). Useful for
    /// coverage tests and routing.
    pub fn tag(&self) -> &'static str {
        match self {
            ItemKind::UserMessage(_) => "user_message",
            ItemKind::AgentMessage(_) => "agent_message",
            ItemKind::ReasoningSummary(_) => "reasoning_summary",
            ItemKind::Plan(_) => "plan",
            ItemKind::PlanMutation(_) => "plan_mutation",
            ItemKind::ContextReceipt(_) => "context_receipt",
            ItemKind::ToolCall(_) => "tool_call",
            ItemKind::ToolResult(_) => "tool_result",
            ItemKind::ShellStream(_) => "shell_stream",
            ItemKind::Patch(_) => "patch",
            ItemKind::Diff(_) => "diff",
            ItemKind::ApprovalRequest(_) => "approval_request",
            ItemKind::ApprovalResult(_) => "approval_result",
            ItemKind::VerificationRequest(_) => "verification_request",
            ItemKind::VerificationReceipt(_) => "verification_receipt",
            ItemKind::Artifact(_) => "artifact",
            ItemKind::Checkpoint(_) => "checkpoint",
            ItemKind::StateCapsule(_) => "state_capsule",
            ItemKind::AgentSpawn(_) => "agent_spawn",
            ItemKind::AgentResult(_) => "agent_result",
            ItemKind::Steer(_) => "steer",
            ItemKind::Interrupt(_) => "interrupt",
            ItemKind::Error(_) => "error",
            ItemKind::Completion(_) => "completion",
            ItemKind::Blocker(_) => "blocker",
        }
    }
}

/// An item: one event in a turn, ordered by `seq`. The `kind` payload is
/// flattened so an item is a single flat object on the wire.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Item {
    pub id: ItemId,
    #[serde(default)]
    pub turn: Option<TurnId>,
    pub seq: u64,
    #[serde(flatten)]
    pub kind: ItemKind,
    pub created_ms: u64,
}

impl Item {
    /// Convenience constructor for building fixtures and compat mappings.
    pub fn new(id: impl Into<ItemId>, seq: u64, kind: ItemKind) -> Self {
        Self {
            id: id.into(),
            turn: None,
            seq,
            kind,
            created_ms: 0,
        }
    }
}
