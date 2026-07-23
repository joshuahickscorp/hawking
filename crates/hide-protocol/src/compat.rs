//! Compat bridge: `hide_core::api` (Intent + UiEvent) <-> this authority.
//!
//! `hide-core`'s `Intent`/`UiEvent` pair is HIDE's CURRENT transport ("Wire-B"
//! in the Bible: intent-in, events-out over HTTP/WS to `hide-serve`). This
//! crate is where that transport CONVERGES: the semantic object model
//! (sec 14) and protocol (sec 15) are the authority Wire-B elevates onto. This
//! module makes that elevation reconcilable in BOTH directions WITHOUT
//! refactoring `hide-core`: it maps the existing shapes onto Turns, Items, and
//! Notifications, and back, so the two can coexist while callers migrate.
//!
//! The mapping is deliberately narrow and lossless where it needs to round-trip
//! (an `Intent::SubmitTurn` survives a round trip through a user-message
//! [`Item`]). Approximate elevations (an intent to its nearest [`Method`]) are
//! labeled as such.

use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::ids::SessionId as CoreSessionId;
use hide_core::types::BlobRef;

use crate::ids::{ApprovalId, ItemId, SessionId, ThreadId, ToolCallId, TurnId};
use crate::item::{Attachment, Item, ItemKind, UserMessage};
use crate::model::{Risk, TurnRole, TurnStatus};
use crate::protocol::{ItemDelta, Method, Notification};
use crate::{item::ApprovalRequest, model::Turn};

/// A turn together with the session it belongs to. `hide_core::Intent`
/// addresses a session directly, while a protocol [`Turn`] hangs off a thread;
/// this pair preserves the session id across the mapping so the reverse is
/// lossless.
#[derive(Debug, Clone, PartialEq)]
pub struct MappedTurn {
    pub session: SessionId,
    pub turn: Turn,
}

fn blobref_to_attachment(b: &BlobRef) -> Attachment {
    Attachment {
        id: b.id.to_string(),
        hash: b.hash.clone(),
        size_bytes: b.size_bytes,
        media_type: b.media_type.clone(),
    }
}

fn attachment_to_blobref(a: &Attachment) -> BlobRef {
    BlobRef {
        id: a.id.clone().into(),
        hash: a.hash.clone(),
        size_bytes: a.size_bytes,
        media_type: a.media_type.clone(),
    }
}

/// Map a `hide_core` intent to a protocol turn. Currently only
/// `Intent::SubmitTurn` elevates cleanly to a Turn (a user message); the other
/// intents are control signals better expressed as [`Method`] calls
/// (see [`intent_method`]) and return `None` here.
pub fn intent_to_turn(intent: &Intent) -> Option<MappedTurn> {
    match intent {
        Intent::SubmitTurn {
            session_id,
            text,
            attachments,
        } => {
            let attach = attachments.iter().map(blobref_to_attachment).collect();
            let item = Item::new(
                ItemId::from("itm_compat_0"),
                0,
                ItemKind::UserMessage(UserMessage {
                    text: text.clone(),
                    attachments: attach,
                }),
            );
            // The Turn hangs off a thread; carry the session id (as the thread
            // key) so the shape is well-formed. The reverse reads the session
            // from `MappedTurn.session`, so no information is lost.
            let turn = Turn {
                id: TurnId::from("trn_compat_0"),
                thread: ThreadId::from(session_id.as_str()),
                role: TurnRole::User,
                status: TurnStatus::Pending,
                items: vec![item],
                parent_turn: None,
                created_ms: 0,
            };
            Some(MappedTurn {
                session: SessionId::from(session_id.as_str()),
                turn,
            })
        }
        _ => None,
    }
}

/// Reverse of [`intent_to_turn`]: recover the `hide_core` `SubmitTurn` from a
/// user-role turn carrying a user-message item. Returns `None` if the turn is
/// not a user turn with a user message.
pub fn turn_to_intent(mapped: &MappedTurn) -> Option<Intent> {
    if mapped.turn.role != TurnRole::User {
        return None;
    }
    let msg = mapped.turn.items.iter().find_map(|it| match &it.kind {
        ItemKind::UserMessage(m) => Some(m),
        _ => None,
    })?;
    let attachments = msg.attachments.iter().map(attachment_to_blobref).collect();
    Some(Intent::SubmitTurn {
        session_id: CoreSessionId::from(mapped.session.as_str()),
        text: msg.text.clone(),
        attachments,
    })
}

/// The nearest protocol [`Method`] an intent elevates to. This is an
/// approximate bridge for routing during migration -- several `hide-core`
/// intents collapse onto the same method (both diff verbs become
/// `approval/respond`) -- not a lossless mapping.
pub fn intent_method(intent: &Intent) -> Method {
    match intent {
        Intent::SubmitTurn { .. } => Method::TurnCreate,
        Intent::CancelRun { .. } => Method::TurnInterrupt,
        Intent::PauseRun { .. } => Method::TurnPause,
        Intent::ResumeRun { .. } => Method::TurnResume,
        Intent::AcceptDiff { .. } => Method::ApprovalRespond,
        Intent::RejectDiff { .. } => Method::ApprovalRespond,
        Intent::ScrubToEvent { .. } => Method::TurnGet,
        Intent::ForkSession { .. } => Method::ThreadFork,
        Intent::OpenFile { .. } => Method::ArtifactGet,
        Intent::RunCommand { .. } => Method::TurnCreate,
        Intent::Custom { .. } => Method::TurnCreate,
    }
}

/// Map a `hide_core` UiEvent onto a protocol [`Notification`]. This is the
/// events-out half of the bridge: the streaming UiEventKind variants elevate
/// onto the sec 15.5 notification set.
pub fn uievent_to_notification(event: &UiEvent) -> Notification {
    match &event.kind {
        UiEventKind::TokenBatch { stream_id, text } => Notification::ItemDeltaNotification {
            item: ItemId::from(stream_id.as_str()),
            delta: ItemDelta {
                append_text: Some(text.clone()),
                shell_chunk: None,
            },
        },
        UiEventKind::RuntimeStatus { status, detail } => Notification::RuntimeStatus {
            status: status.clone(),
            detail: detail.clone(),
        },
        UiEventKind::ToolProgress { call_id, message, .. } => Notification::ToolProgress {
            call_id: ToolCallId::from(call_id.as_str()),
            message: message.clone(),
        },
        UiEventKind::SecurityGate { gate, message } => Notification::ApprovalRequested {
            request: ApprovalRequest {
                request_id: ApprovalId::from(gate.as_str()),
                action: gate.clone(),
                risk: Risk::Medium,
                effects: Vec::new(),
                detail: Some(message.clone()),
            },
        },
        UiEventKind::Error { code, message } => Notification::Error {
            code: code.clone(),
            message: message.clone(),
        },
        UiEventKind::ProjectionPatch { projection, patch } => Notification::Custom {
            name: format!("projection/{projection}"),
            payload: patch.clone(),
        },
        UiEventKind::Custom(value) => Notification::Custom {
            name: "custom".to_string(),
            payload: value.clone(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::ids::BlobId;

    fn sample_submit_turn() -> Intent {
        Intent::SubmitTurn {
            session_id: CoreSessionId::from("ses_demo"),
            text: "try five fixes for the flaky auth retry".to_string(),
            attachments: vec![BlobRef {
                id: BlobId::from("blb_1"),
                hash: "deadbeef".to_string(),
                size_bytes: 42,
                media_type: Some("text/plain".to_string()),
            }],
        }
    }

    #[test]
    fn submit_turn_maps_to_user_message_item() {
        let intent = sample_submit_turn();
        let mapped = intent_to_turn(&intent).expect("submit_turn maps to a turn");
        assert_eq!(mapped.turn.role, TurnRole::User);
        assert_eq!(mapped.turn.items.len(), 1);
        match &mapped.turn.items[0].kind {
            ItemKind::UserMessage(m) => {
                assert_eq!(m.text, "try five fixes for the flaky auth retry");
                assert_eq!(m.attachments.len(), 1);
                assert_eq!(m.attachments[0].id, "blb_1");
            }
            other => panic!("expected user_message item, got {other:?}"),
        }
    }

    #[test]
    fn submit_turn_round_trips_through_the_bridge() {
        let intent = sample_submit_turn();
        let mapped = intent_to_turn(&intent).unwrap();
        let back = turn_to_intent(&mapped).expect("user turn maps back to an intent");
        assert_eq!(back, intent, "intent survives the round trip losslessly");
    }

    #[test]
    fn non_submit_intents_do_not_map_to_a_turn() {
        let intent = Intent::PauseRun {
            run_id: hide_core::ids::RunId::from("run_1"),
        };
        assert!(intent_to_turn(&intent).is_none());
        assert_eq!(intent_method(&intent), Method::TurnPause);
    }

    #[test]
    fn uievents_elevate_to_notifications() {
        let ev = UiEvent {
            seq: 7,
            session_id: None,
            kind: UiEventKind::ToolProgress {
                call_id: "tcl_9".to_string(),
                message: "running 12 tests".to_string(),
                event_id: None,
            },
        };
        match uievent_to_notification(&ev) {
            Notification::ToolProgress { call_id, message } => {
                assert_eq!(call_id.as_str(), "tcl_9");
                assert_eq!(message, "running 12 tests");
            }
            other => panic!("expected tool/progress notification, got {other:?}"),
        }
    }
}
