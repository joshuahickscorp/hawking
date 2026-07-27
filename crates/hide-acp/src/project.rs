//! `ProjectHideToAcp`: project the HIDE schema authority onto ACP.
//!
//! HIDE's turn is a stream of `hide_protocol::Item`s (and the server pushes
//! `hide_protocol::Notification`s). An ACP editor consumes `session/update`
//! notifications and `session/request_permission` requests. This module maps one
//! onto the other, honoring the effective capability set: an item that HIDE
//! would render richly degrades to a plainer ACP shape when the client cannot
//! receive the rich one. Projection is total over known item kinds -- a kind
//! with no honest ACP surface projects to nothing rather than to a fake.
//! Nothing here runs a model.

use serde_json::Value;

use hide_protocol::item::{
    ApprovalRequest, Diff as HideDiff, DiffStatus, ItemKind, Patch, ShellStream, ToolCall as HideToolCall,
    ToolResult,
};
use hide_protocol::model::Risk;
use hide_protocol::plan::{Effect, Plan};
use hide_protocol::{Item, Notification};

use crate::capability::EffectiveCapabilities;
use crate::content::ContentBlock;
use crate::ids::{AcpSessionId, AcpTerminalId, AcpToolCallId};
use crate::permission::{standard_options, RequestPermissionRequest};
use crate::session::{
    AcpPlan, PlanEntry, PlanEntryPriority, PlanEntryStatus, SessionNotification, SessionUpdate,
};
use crate::tool_call::{
    ToolCall, ToolCallContent, ToolCallLocation, ToolCallStatus, ToolCallUpdate, ToolKind,
};

/// One thing the boundary sends toward the ACP client as a result of a HIDE
/// event: a session update, or a permission request.
#[derive(Debug, Clone, PartialEq)]
pub enum AcpOutbound {
    /// A `session/update` notification.
    Update(SessionNotification),
    /// A `session/request_permission` request.
    Permission(RequestPermissionRequest),
}

impl AcpOutbound {
    /// The ACP method this outbound corresponds to (for routing and tests).
    pub fn method(&self) -> &'static str {
        match self {
            AcpOutbound::Update(_) => "session/update",
            AcpOutbound::Permission(_) => "session/request_permission",
        }
    }
}

/// Projects HIDE items and notifications for one ACP session under a fixed
/// effective capability set.
#[derive(Debug, Clone)]
pub struct ProjectHideToAcp {
    session: AcpSessionId,
    caps: EffectiveCapabilities,
}

impl ProjectHideToAcp {
    pub fn new(session: AcpSessionId, caps: EffectiveCapabilities) -> Self {
        Self { session, caps }
    }

    /// The ACP session these projections target.
    pub fn session(&self) -> &AcpSessionId {
        &self.session
    }

    /// The effective capabilities in force.
    pub fn capabilities(&self) -> &EffectiveCapabilities {
        &self.caps
    }

    fn update(&self, u: SessionUpdate) -> AcpOutbound {
        AcpOutbound::Update(SessionNotification::new(self.session.clone(), u))
    }

    /// Project one item to zero or more ACP outbounds, in order.
    pub fn project_item(&self, item: &Item) -> Vec<AcpOutbound> {
        self.project_kind(&item.kind)
    }

    /// Project a whole ordered item stream, preserving order.
    pub fn project_items<'a, I>(&self, items: I) -> Vec<AcpOutbound>
    where
        I: IntoIterator<Item = &'a Item>,
    {
        items.into_iter().flat_map(|i| self.project_item(i)).collect()
    }

    /// Project a server notification (the streaming form). `item/added` and
    /// `item/updated` re-enter [`Self::project_kind`]; other notifications map
    /// to their nearest ACP surface or to nothing.
    pub fn project_notification(&self, n: &Notification) -> Vec<AcpOutbound> {
        match n {
            Notification::ItemAdded { item } | Notification::ItemUpdated { item } => {
                self.project_kind(&item.kind)
            }
            Notification::PlanUpdated { plan } => vec![self.update(self.plan_update(plan))],
            Notification::ApprovalRequested { request } => self.approval(request),
            Notification::Error { message, .. } => {
                vec![self.update(SessionUpdate::AgentMessageChunk {
                    content: ContentBlock::text(format!("error: {message}")),
                })]
            }
            // Lifecycle pushes (session/turn/agent status, checkpoints, state)
            // are carried by ACP's request/response envelope, not by session
            // updates; they project to nothing here.
            _ => Vec::new(),
        }
    }

    fn project_kind(&self, kind: &ItemKind) -> Vec<AcpOutbound> {
        match kind {
            ItemKind::UserMessage(m) => vec![self.update(SessionUpdate::UserMessageChunk {
                content: ContentBlock::text(&m.text),
            })],
            ItemKind::AgentMessage(m) => vec![self.update(SessionUpdate::AgentMessageChunk {
                content: ContentBlock::text(&m.text),
            })],
            ItemKind::ReasoningSummary(r) => {
                // Honest degrade: without a thought surface, reasoning still
                // reaches the user as a plain message rather than vanishing.
                let update = if self.caps.thoughts {
                    SessionUpdate::AgentThoughtChunk {
                        content: ContentBlock::text(&r.text),
                    }
                } else {
                    SessionUpdate::AgentMessageChunk {
                        content: ContentBlock::text(&r.text),
                    }
                };
                vec![self.update(update)]
            }
            ItemKind::Plan(p) => vec![self.update(self.plan_update(p))],
            ItemKind::ToolCall(tc) => vec![self.update(SessionUpdate::ToolCall(self.tool_call(tc)))],
            ItemKind::ToolResult(tr) => {
                vec![self.update(SessionUpdate::ToolCallUpdate(self.tool_result(tr)))]
            }
            ItemKind::ShellStream(s) => {
                vec![self.update(SessionUpdate::ToolCallUpdate(self.shell(s)))]
            }
            ItemKind::Patch(p) => vec![self.update(SessionUpdate::ToolCall(self.patch(p)))],
            ItemKind::Diff(d) => vec![self.update(SessionUpdate::ToolCall(self.diff(d)))],
            ItemKind::ApprovalRequest(a) => self.approval(a),
            ItemKind::VerificationReceipt(v) => {
                let outcome = format!("{:?}", v.outcome).to_lowercase();
                vec![self.update(SessionUpdate::AgentMessageChunk {
                    content: ContentBlock::text(format!(
                        "verification {}: {outcome}",
                        v.oracle.as_str()
                    )),
                })]
            }
            ItemKind::Artifact(a) => {
                let uri = a.path.clone().unwrap_or_else(|| a.name.clone());
                vec![self.update(SessionUpdate::AgentMessageChunk {
                    content: ContentBlock::ResourceLink {
                        uri,
                        name: Some(a.name.clone()),
                        mime_type: None,
                    },
                })]
            }
            ItemKind::Error(e) => vec![self.update(SessionUpdate::AgentMessageChunk {
                content: ContentBlock::text(format!("error [{}]: {}", e.code, e.message)),
            })],
            ItemKind::Blocker(b) => vec![self.update(SessionUpdate::AgentMessageChunk {
                content: ContentBlock::text(format!("blocked [{}]: {}", b.code, b.message)),
            })],
            // Kinds with no honest ACP surface (coordination, receipts,
            // checkpoints, steering, the client-originated results). They
            // project to nothing rather than to a fabricated update.
            _ => Vec::new(),
        }
    }

    fn plan_update(&self, plan: &Plan) -> SessionUpdate {
        let entries = plan
            .steps
            .iter()
            .map(|s| PlanEntry {
                content: s.objective.clone(),
                priority: PlanEntryPriority::Medium,
                // HIDE plan steps carry no per-step status; they enter the plan
                // pending until a later plan snapshot supersedes them.
                status: PlanEntryStatus::Pending,
            })
            .collect();
        SessionUpdate::Plan(AcpPlan { entries })
    }

    fn tool_call(&self, tc: &HideToolCall) -> ToolCall {
        ToolCall {
            tool_call_id: AcpToolCallId::new(tc.call_id.as_str()),
            title: tc.tool.as_str().to_string(),
            kind: tool_kind_for(tc.tool.as_str()),
            status: ToolCallStatus::InProgress,
            content: Vec::new(),
            locations: Vec::new(),
            raw_input: Some(tc.arguments.clone()),
        }
    }

    fn tool_result(&self, tr: &ToolResult) -> ToolCallUpdate {
        let mut content = Vec::new();
        if let Some(text) = value_to_text(&tr.output) {
            if !text.is_empty() {
                content.push(ToolCallContent::text(text));
            }
        }
        if let Some(err) = &tr.error {
            content.push(ToolCallContent::text(format!("error: {err}")));
        }
        ToolCallUpdate {
            tool_call_id: AcpToolCallId::new(tr.call_id.as_str()),
            status: Some(if tr.ok {
                ToolCallStatus::Completed
            } else {
                ToolCallStatus::Failed
            }),
            title: None,
            kind: None,
            content,
            raw_output: Some(tr.output.clone()),
        }
    }

    fn shell(&self, s: &ShellStream) -> ToolCallUpdate {
        let call_id = s
            .call_id
            .as_ref()
            .map(|c| c.as_str().to_string())
            .unwrap_or_else(|| "shell".to_string());
        let mut content = Vec::new();
        // Terminal projection: reference the live terminal when the client can
        // render one. The bytes over that terminal are DEFERRED_MODEL_REQUIRED;
        // the recorded chunk is attached as a text snapshot so replay is
        // lossless. Without terminal support we degrade to text only.
        if self.caps.terminal {
            content.push(ToolCallContent::Terminal {
                terminal_id: AcpTerminalId::new(format!("term_{call_id}")),
            });
        }
        content.push(ToolCallContent::text(&s.chunk));
        ToolCallUpdate {
            tool_call_id: AcpToolCallId::new(call_id),
            status: Some(ToolCallStatus::InProgress),
            title: None,
            kind: Some(ToolKind::Execute),
            content,
            raw_output: None,
        }
    }

    fn patch(&self, p: &Patch) -> ToolCall {
        let parsed = crate::unified_diff::parse_unified_diff(&p.unified_diff);
        let content: Vec<ToolCallContent> = if parsed.is_empty() {
            // Fallback: the diff did not parse into file sections; surface it
            // verbatim as text so nothing is dropped.
            vec![ToolCallContent::text(&p.unified_diff)]
        } else {
            parsed
                .into_iter()
                .map(|f| ToolCallContent::Diff {
                    path: f.path,
                    old_text: f.old_text,
                    new_text: f.new_text,
                })
                .collect()
        };
        let locations = p
            .files
            .iter()
            .map(|path| ToolCallLocation {
                path: path.clone(),
                line: None,
            })
            .collect();
        ToolCall {
            tool_call_id: AcpToolCallId::new(&p.patch_id),
            title: p.summary.clone().unwrap_or_else(|| "Proposed edit".to_string()),
            kind: ToolKind::Edit,
            status: ToolCallStatus::Pending,
            content,
            locations,
            raw_input: None,
        }
    }

    fn diff(&self, d: &HideDiff) -> ToolCall {
        let mut old_text = String::new();
        let mut new_text = String::new();
        for hunk in &d.hunks {
            let (o, n) = crate::unified_diff::reconstruct_hunk(&hunk.text);
            old_text.push_str(&o);
            new_text.push_str(&n);
        }
        let status = match d.status {
            DiffStatus::Applied => ToolCallStatus::Completed,
            DiffStatus::Rejected => ToolCallStatus::Failed,
            DiffStatus::Reverted => ToolCallStatus::Failed,
            DiffStatus::Proposed => ToolCallStatus::Pending,
        };
        ToolCall {
            tool_call_id: AcpToolCallId::new(&d.diff_id),
            title: d.path.clone(),
            kind: ToolKind::Edit,
            status,
            content: vec![ToolCallContent::Diff {
                path: d.path.clone(),
                old_text: Some(old_text),
                new_text,
            }],
            locations: vec![ToolCallLocation {
                path: d.path.clone(),
                line: None,
            }],
            raw_input: None,
        }
    }

    fn approval(&self, a: &ApprovalRequest) -> Vec<AcpOutbound> {
        let mut content = Vec::new();
        if let Some(detail) = &a.detail {
            content.push(ToolCallContent::text(detail));
        }
        let tool_call = ToolCallUpdate {
            tool_call_id: AcpToolCallId::new(a.request_id.as_str()),
            status: Some(ToolCallStatus::Pending),
            title: Some(a.action.clone()),
            kind: Some(kind_for_effects(&a.effects, a.risk)),
            content,
            raw_output: None,
        };
        vec![AcpOutbound::Permission(RequestPermissionRequest {
            session_id: self.session.clone(),
            tool_call,
            options: standard_options(),
        })]
    }
}

/// Map a HIDE tool id to the nearest ACP `ToolKind` by its name. Unknown tools
/// are `Other`; the mapping is a display hint, not a security boundary.
fn tool_kind_for(tool: &str) -> ToolKind {
    let t = tool.to_lowercase();
    if t.contains("read") || t.contains("cat") || t.contains("open") {
        ToolKind::Read
    } else if t.contains("edit") || t.contains("write") || t.contains("patch") {
        ToolKind::Edit
    } else if t.contains("delete") || t.contains("rm") {
        ToolKind::Delete
    } else if t.contains("move") || t.contains("rename") || t.contains("mv") {
        ToolKind::Move
    } else if t.contains("search") || t.contains("grep") || t.contains("find") {
        ToolKind::Search
    } else if t.contains("shell") || t.contains("exec") || t.contains("run") || t.contains("bash") {
        ToolKind::Execute
    } else if t.contains("fetch") || t.contains("http") || t.contains("web") {
        ToolKind::Fetch
    } else {
        ToolKind::Other
    }
}

/// Choose an ACP tool kind for an approval from its declared effects.
fn kind_for_effects(effects: &[Effect], _risk: Risk) -> ToolKind {
    if effects.iter().any(|e| matches!(e, Effect::WriteFs | Effect::Vcs)) {
        ToolKind::Edit
    } else if effects.iter().any(|e| matches!(e, Effect::Shell | Effect::Process)) {
        ToolKind::Execute
    } else if effects.iter().any(|e| matches!(e, Effect::Network)) {
        ToolKind::Fetch
    } else {
        ToolKind::Other
    }
}

/// Render a tool-result JSON value as display text. Strings pass through; other
/// shapes serialize compactly.
fn value_to_text(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::String(s) => Some(s.clone()),
        other => serde_json::to_string(other).ok(),
    }
}
