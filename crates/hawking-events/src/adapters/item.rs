//! Adapter: hide-protocol `Item` → canonical Event.
//!
//! # Authority note
//!
//! `Item` / `ItemKind` remain the **turn wire schema** authority (hide-protocol).
//! They are not the durable product-event log. When a turn item must be
//! recorded durably, project it through this adapter.

use hide_core::event::EventClass;
use hide_core::ids::SessionId;
use hide_protocol::item::{Item, ItemKind};
use serde_json::{json, Value};

use crate::categories::Category;
use crate::envelope::{CanonicalEvent, ContentVerification, NewCanonical, Subsystem};

/// Map a protocol Item into a provisional canonical event under `session`.
pub fn item_to_canonical(session_id: SessionId, item: &Item) -> CanonicalEvent {
    let (category, kind, payload, class) = map_kind(&item.kind);
    CanonicalEvent::sequence(
        item.seq,
        NewCanonical::new(
            session_id,
            Subsystem::HideBackend,
            ContentVerification::Provisional,
            category,
            payload,
        )
        .with_class(class)
        .with_kind(kind),
    )
}

fn map_kind(kind: &ItemKind) -> (Category, &'static str, Value, EventClass) {
    match kind {
        ItemKind::UserMessage(m) => (
            Category::Text,
            "model.token",
            json!({ "role": "user", "text": m.text, "attachments": m.attachments.len() }),
            EventClass::Action,
        ),
        ItemKind::AgentMessage(m) => (
            Category::Text,
            "model.token",
            json!({ "role": "agent", "text": m.text }),
            EventClass::Neither,
        ),
        ItemKind::ReasoningSummary(m) => (
            Category::Reasoning,
            "model.reasoning",
            json!({ "text": m.text }),
            EventClass::Neither,
        ),
        ItemKind::Plan(p) => (
            Category::Plans,
            "plan.created",
            json!({ "plan": serde_json::to_value(p).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::PlanMutation(m) => (
            Category::Plans,
            "plan.mutation",
            json!({ "mutation": serde_json::to_value(m).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::ToolCall(t) => (
            Category::Tools,
            "tool.call",
            json!({
                "call_id": t.call_id.to_string(),
                "tool": t.tool.to_string(),
                "arguments": t.arguments,
            }),
            EventClass::Action,
        ),
        ItemKind::ToolResult(t) => (
            Category::Tools,
            "tool.result",
            json!({
                "call_id": t.call_id.to_string(),
                "ok": t.ok,
                "output": t.output,
                "error": t.error,
            }),
            EventClass::Observation,
        ),
        ItemKind::ShellStream(s) => (
            Category::Tools,
            "tool.call",
            json!({ "shell": true, "chunk": s.chunk }),
            EventClass::Neither,
        ),
        ItemKind::Patch(p) => (
            Category::Edits,
            "edit.patch",
            json!({ "patch_id": p.patch_id, "files": p.files, "unified_diff": p.unified_diff }),
            EventClass::Action,
        ),
        ItemKind::Diff(d) => (
            Category::Edits,
            "edit.diff",
            json!({ "diff": serde_json::to_value(d).unwrap_or(Value::Null) }),
            EventClass::Observation,
        ),
        ItemKind::ApprovalRequest(a) => (
            Category::Permissions,
            "security.gate",
            json!({ "approval": serde_json::to_value(a).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::ApprovalResult(a) => (
            Category::Permissions,
            "security.decision",
            json!({ "result": serde_json::to_value(a).unwrap_or(Value::Null) }),
            EventClass::Observation,
        ),
        ItemKind::VerificationRequest(v) => (
            Category::Verification,
            "verify.request",
            json!({ "request": serde_json::to_value(v).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::VerificationReceipt(v) => (
            Category::Verification,
            "verify.receipt",
            json!({ "receipt": serde_json::to_value(v).unwrap_or(Value::Null) }),
            EventClass::Observation,
        ),
        ItemKind::AgentSpawn(a) => (
            Category::Agents,
            "agent.spawn",
            json!({ "spawn": serde_json::to_value(a).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::AgentResult(a) => (
            Category::Agents,
            "agent.result",
            json!({ "result": serde_json::to_value(a).unwrap_or(Value::Null) }),
            EventClass::Observation,
        ),
        ItemKind::Error(e) => (
            Category::Errors,
            "error",
            json!({ "code": e.code, "message": e.message, "recoverable": false }),
            EventClass::Neither,
        ),
        ItemKind::Completion(c) => (
            Category::Usage,
            "model.usage",
            json!({ "completion": serde_json::to_value(c).unwrap_or(Value::Null) }),
            EventClass::Neither,
        ),
        ItemKind::ContextReceipt(r) => (
            Category::ModelLifecycle,
            "runtime.status",
            json!({ "context_receipt": serde_json::to_value(r).unwrap_or(Value::Null) }),
            EventClass::Neither,
        ),
        ItemKind::Artifact(a) => (
            Category::Edits,
            "edit.diff",
            json!({ "artifact": serde_json::to_value(a).unwrap_or(Value::Null) }),
            EventClass::Neither,
        ),
        ItemKind::Checkpoint(c) => (
            Category::ModelLifecycle,
            "runtime.status",
            json!({ "checkpoint": serde_json::to_value(c).unwrap_or(Value::Null), "status": "checkpoint" }),
            EventClass::Neither,
        ),
        ItemKind::StateCapsule(s) => (
            Category::ModelLifecycle,
            "runtime.status",
            json!({ "state_capsule": serde_json::to_value(s).unwrap_or(Value::Null) }),
            EventClass::Neither,
        ),
        ItemKind::Steer(s) => (
            Category::Agents,
            "agent.phase",
            json!({ "steer": serde_json::to_value(s).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::Interrupt(i) => (
            Category::Agents,
            "agent.phase",
            json!({ "interrupt": serde_json::to_value(i).unwrap_or(Value::Null) }),
            EventClass::Action,
        ),
        ItemKind::Blocker(b) => (
            Category::Warnings,
            "system.warning",
            json!({ "blocker": true, "code": b.code, "message": b.message }),
            EventClass::Neither,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::with_deterministic_ids;
    use hide_protocol::ids::ItemId;
    use hide_protocol::item::{AgentMessage, Item, ItemKind, ReasoningSummary};
    #[test]
    fn agent_message_and_reasoning_map_correctly() {
        with_deterministic_ids(30, || {
            let ses = SessionId::from("ses_item");
            let msg = Item::new(
                ItemId::from("itm_1"),
                3,
                ItemKind::AgentMessage(AgentMessage {
                    text: "answer".into(),
                }),
            );
            let c = item_to_canonical(ses.clone(), &msg);
            assert_eq!(c.category, Category::Text);
            assert_eq!(c.seq(), 3);
            let r = Item::new(
                ItemId::from("itm_2"),
                4,
                ItemKind::ReasoningSummary(ReasoningSummary {
                    text: "think".into(),
                }),
            );
            let c2 = item_to_canonical(ses, &r);
            assert_eq!(c2.category, Category::Reasoning);
            assert_eq!(c2.kind(), "model.reasoning");
        });
    }
}
