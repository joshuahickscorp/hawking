use hawking_events::adapters::{
    item_to_canonical, seed_event_to_canonical, stream_event_to_canonical, ui_event_to_canonical,
    SeedFsmEvent, StreamEventView,
};
use hawking_events::{
    all_categories, kind_for_category, CanonicalEvent, Category, ContentVerification, NewCanonical,
    Subsystem, COMPETING_MODELS,
};
use hide_core::api::{UiEvent, UiEventKind};
use hide_core::ids::{with_deterministic_ids, SessionId};
use hide_protocol::ids::{ItemId, ToolCallId, ToolId};
use hide_protocol::item::{Item, ItemKind, ToolCall, ToolResult};
use serde_json::json;
#[test]
fn round_trip_every_category() {
    with_deterministic_ids(100, || {
        let ses = SessionId::from("ses_round");
        for (i, cat) in all_categories().iter().enumerate() {
            let c = CanonicalEvent::sequence(
                (i as u64) + 1,
                NewCanonical::new(
                    ses.clone(),
                    Subsystem::Bridge,
                    ContentVerification::TargetVerified,
                    *cat,
                    json!({ "probe": cat.as_str() }),
                ),
            );
            assert_eq!(c.category, *cat);
            assert_eq!(c.kind(), kind_for_category(*cat));
            assert_eq!(c.seq(), (i as u64) + 1);
            assert_eq!(c.session_id(), &ses);
            assert!(!c.id().as_str().is_empty());
            assert_eq!(c.subsystem, Subsystem::Bridge);
            assert_eq!(c.verification, ContentVerification::TargetVerified);
            let again = c.round_trip_json().expect("serde round-trip");
            assert_eq!(again.category, *cat);
            assert_eq!(again.kind(), c.kind());
            assert_eq!(again.seq(), c.seq());
            assert_eq!(again.verification, ContentVerification::TargetVerified);
            assert_eq!(again.event.payload["probe"], cat.as_str());
        }
        assert_eq!(all_categories().len(), 24);
    });
}
#[test]
fn seventeen_you_events_round_trip_on_same_bus() {
    use hawking_events::{ProducingSurface, YOU_EVENTS};
    with_deterministic_ids(400, || {
        let ses = SessionId::from("ses_you");
        assert_eq!(YOU_EVENTS.len(), 17);
        for (i, spec) in YOU_EVENTS.iter().enumerate() {
            let c = spec.sequence(
                (i as u64) + 1,
                ses.clone(),
                json!({ "probe": spec.event.as_pascal() }),
                None,
            );
            assert_eq!(c.kind(), spec.kind);
            assert_eq!(c.category, spec.category);
            assert_eq!(c.surface, ProducingSurface::You);
            assert_eq!(c.subsystem, Subsystem::HideYou);
            assert_eq!(c.verification, spec.default_verification);
            assert!(!c.id().as_str().is_empty());
            assert_eq!(c.seq(), (i as u64) + 1);
            assert_eq!(c.session_id(), &ses);
            let again = c.round_trip_json().expect("serde round-trip");
            assert_eq!(again.kind(), spec.kind);
            assert_eq!(again.surface, ProducingSurface::You);
            assert_eq!(again.verification, spec.default_verification);
            if spec.default_verification == ContentVerification::Provisional {
                assert_eq!(again.verification, ContentVerification::Provisional);
            }
        }
    });
}
#[test]
fn deprecated_adapters_produce_identical_canonical_events() {
    with_deterministic_ids(200, || {
        let ses = SessionId::from("ses_adapt");
        let ui = UiEvent {
            seq: 11,
            session_id: Some(ses.clone()),
            kind: UiEventKind::ToolProgress {
                call_id: "call_1".into(),
                message: "running".into(),
                event_id: None,
            },
        };
        let from_ui = ui_event_to_canonical(&ui, ses.clone());
        let item = Item::new(
            ItemId::from("itm_tc"),
            11,
            ItemKind::ToolCall(ToolCall {
                call_id: ToolCallId::from("call_1"),
                tool: ToolId::from("fs.read"),
                arguments: json!({ "path": "x" }),
            }),
        );
        let from_item = item_to_canonical(ses.clone(), &item);
        assert_eq!(from_ui.category, Category::Tools);
        assert_eq!(from_item.category, Category::Tools);
        assert_eq!(from_ui.kind(), "tool.call");
        assert_eq!(from_item.kind(), "tool.call");
        assert_eq!(from_ui.seq(), from_item.seq());
        assert_eq!(from_ui.session_id(), from_item.session_id());
        let stream = stream_event_to_canonical(
            ses.clone(),
            12,
            "s0",
            &StreamEventView::Token {
                id: 1,
                text: "hi".into(),
            },
        );
        let ui_tok = ui_event_to_canonical(
            &UiEvent {
                seq: 12,
                session_id: Some(ses.clone()),
                kind: UiEventKind::TokenBatch {
                    stream_id: "s0".into(),
                    text: "hi".into(),
                },
            },
            ses.clone(),
        );
        assert_eq!(stream.category, ui_tok.category);
        assert_eq!(stream.kind(), ui_tok.kind());
        assert_eq!(stream.event.payload["text"], ui_tok.event.payload["text"]);
        let a = seed_event_to_canonical(ses.clone(), 1, "idle", SeedFsmEvent::Prepare, "prepared");
        let b = seed_event_to_canonical(ses, 1, "idle", SeedFsmEvent::Prepare, "prepared");
        assert_eq!(a.kind(), b.kind());
        assert_eq!(a.event.payload, b.event.payload);
        assert_eq!(a.category, b.category);
    });
}
#[test]
fn tool_result_item_is_observation_class() {
    with_deterministic_ids(300, || {
        let item = Item::new(
            ItemId::from("itm_tr"),
            1,
            ItemKind::ToolResult(ToolResult {
                call_id: ToolCallId::from("c"),
                ok: true,
                output: json!("ok"),
                error: None,
            }),
        );
        let c = item_to_canonical(SessionId::from("ses"), &item);
        assert_eq!(c.kind(), "tool.result");
        assert!(c.event.is_observation());
    });
}
#[test]
fn six_competing_models_enumerated() {
    assert_eq!(COMPETING_MODELS.len(), 6);
    assert_eq!(
        COMPETING_MODELS
            .iter()
            .filter(|m| m.name.contains("Event")
                || m.name.contains("Item")
                || m.name.contains("ledger"))
            .count(),
        6
    );
    let canonical = COMPETING_MODELS
        .iter()
        .filter(|m| matches!(m.status, hawking_events::MigrationStatus::Canonical))
        .count();
    assert_eq!(canonical, 1);
}
