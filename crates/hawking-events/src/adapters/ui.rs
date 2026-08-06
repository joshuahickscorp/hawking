//! Adapter: `hide_core::api::UiEvent` → canonical Event.
//!
//! # Authority note
//!
//! `UiEvent` is a **live projection** (Wire-B UI bus), not a durable log.
//! hide-backend already maps durable `Event` → `UiEvent` in
//! `replay::event_to_ui_event`. This module is the reverse elevation for
//! callers that only hold a UiEvent and need a canonical record.

use hide_core::api::{UiEvent, UiEventKind};
use hide_core::event::EventClass;
use hide_core::ids::SessionId;
use serde_json::json;

use crate::categories::Category;
use crate::envelope::{CanonicalEvent, ContentVerification, NewCanonical, Subsystem};

/// Elevate a UiEvent into a provisional canonical event.
///
/// When `session_id` is missing on the UiEvent, the provided fallback is used.
pub fn ui_event_to_canonical(event: &UiEvent, fallback_session: SessionId) -> CanonicalEvent {
    let session = event.session_id.clone().unwrap_or(fallback_session);
    let (category, kind, payload, class) = match &event.kind {
        UiEventKind::TokenBatch { stream_id, text } => (
            Category::Text,
            "model.token",
            json!({ "stream_id": stream_id, "text": text }),
            EventClass::Neither,
        ),
        UiEventKind::RuntimeStatus { status, detail } => (
            Category::ModelLifecycle,
            "runtime.status",
            json!({ "status": status, "detail": detail }),
            EventClass::Neither,
        ),
        UiEventKind::ToolProgress {
            call_id,
            message,
            event_id,
        } => (
            Category::Tools,
            "tool.call",
            json!({
                "call_id": call_id,
                "message": message,
                "source_event_id": event_id,
            }),
            EventClass::Action,
        ),
        UiEventKind::SecurityGate { gate, message } => (
            Category::Permissions,
            "security.gate",
            json!({ "gate": gate, "message": message }),
            EventClass::Neither,
        ),
        UiEventKind::Error { code, message } => (
            Category::Errors,
            "error",
            json!({ "code": code, "message": message, "recoverable": true }),
            EventClass::Neither,
        ),
        UiEventKind::ProjectionPatch { projection, patch } => (
            Category::ModelLifecycle,
            "runtime.status",
            json!({ "projection": projection, "patch": patch, "status": "projection_patch" }),
            EventClass::Neither,
        ),
        UiEventKind::Custom(v) => (
            Category::ModelLifecycle,
            "runtime.status",
            json!({ "custom": v, "status": "custom" }),
            EventClass::Neither,
        ),
    };

    let input = NewCanonical::new(
        session,
        Subsystem::HideBackend,
        ContentVerification::Provisional,
        category,
        payload,
    )
    .with_class(class)
    .with_kind(kind);
    // Preserve the UiEvent's sequence as the durable seq so projections that
    // already carried a seq remain ordered identically after elevation.
    CanonicalEvent::sequence(event.seq, input)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::with_deterministic_ids;
    #[test]
    fn token_batch_elevates_to_text() {
        with_deterministic_ids(20, || {
            let ses = SessionId::from("ses_ui");
            let ui = UiEvent {
                seq: 9,
                session_id: Some(ses.clone()),
                kind: UiEventKind::TokenBatch {
                    stream_id: "s1".into(),
                    text: "hello".into(),
                },
            };
            let c = ui_event_to_canonical(&ui, ses);
            assert_eq!(c.seq(), 9);
            assert_eq!(c.category, Category::Text);
            assert_eq!(c.kind(), "model.token");
            assert_eq!(c.event.payload["text"], "hello");
        });
    }
    #[test]
    fn error_elevates_to_errors_category() {
        with_deterministic_ids(21, || {
            let ses = SessionId::from("ses_ui2");
            let ui = UiEvent {
                seq: 1,
                session_id: None,
                kind: UiEventKind::Error {
                    code: "E".into(),
                    message: "boom".into(),
                },
            };
            let c = ui_event_to_canonical(&ui, ses);
            assert_eq!(c.category, Category::Errors);
            assert_eq!(c.kind(), "error");
        });
    }
}
