//! The async client over MockTransport (test group c).
//!
//! Proves the client builds a typed `hide-protocol` Method request, sends it
//! through the transport, and parses a typed result deterministically. The
//! transport is an in-memory mock: no socket, no model.

use hide_sdk::{Client, MockTransport, SdkError, Transport};

use hide_protocol::ids::RequestId;
use hide_protocol::model::{Session, SessionStatus, Turn, TurnRole, TurnStatus};
use hide_protocol::protocol::{Method, Notification, RpcError};

fn sample_session() -> Session {
    Session {
        id: hide_protocol::ids::SessionId::from("ses_1"),
        workspace: hide_protocol::ids::WorkspaceId::from("wsp_1"),
        repository: None,
        environment: None,
        title: Some("auth retry".into()),
        threads: vec![],
        status: SessionStatus::Active,
        created_ms: 7,
    }
}

fn sample_turn() -> Turn {
    Turn {
        id: hide_protocol::ids::TurnId::from("trn_1"),
        thread: hide_protocol::ids::ThreadId::from("thr_1"),
        role: TurnRole::Agent,
        status: TurnStatus::Running,
        items: vec![],
        parent_turn: None,
        created_ms: 5,
    }
}

#[tokio::test]
async fn session_start_sends_typed_method_and_parses_typed_result() {
    let session = sample_session();
    let mock =
        MockTransport::new().on(Method::SessionNew, serde_json::to_value(&session).unwrap());
    let client = Client::new(mock);

    let got: Session = client
        .session_start("wsp_1", Some("auth retry"))
        .await
        .expect("session_start should succeed");
    assert_eq!(got, session, "the typed result parses back to the fixture");

    // The transport saw exactly one request, carrying the right method, a
    // deterministically minted id, and the helper's params.
    let received = client.transport().received();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].method, Method::SessionNew);
    assert_eq!(received[0].id, RequestId::from("req_1"));
    assert_eq!(
        received[0].params,
        serde_json::json!({ "workspace": "wsp_1", "title": "auth retry" })
    );
}

#[tokio::test]
async fn turn_start_parses_a_typed_turn_and_mints_sequential_ids() {
    let turn = sample_turn();
    let mock = MockTransport::new().on(Method::TurnCreate, serde_json::to_value(&turn).unwrap());
    let client = Client::new(mock);

    // Two calls: ids must advance req_1 -> req_2 deterministically.
    let first: Turn = client.turn_start("thr_1", "go").await.unwrap();
    let second: Turn = client.turn_start("thr_1", "again").await.unwrap();
    assert_eq!(first, turn);
    assert_eq!(second, turn);

    let received = client.transport().received();
    assert_eq!(received.len(), 2);
    assert_eq!(received[0].id, RequestId::from("req_1"));
    assert_eq!(received[1].id, RequestId::from("req_2"));
    assert_eq!(received[1].method, Method::TurnCreate);
    assert_eq!(
        received[1].params,
        serde_json::json!({ "thread": "thr_1", "text": "again" })
    );
}

#[tokio::test]
async fn rpc_error_envelope_surfaces_as_sdk_error() {
    let mock = MockTransport::new().on_error(
        Method::TurnCreate,
        RpcError {
            code: -32000,
            message: "turn rejected".into(),
            data: None,
        },
    );
    let client = Client::new(mock);

    let err = client
        .turn_start::<Turn>("thr_1", "go")
        .await
        .expect_err("an error envelope must surface");
    match err {
        SdkError::Rpc { method, code, message } => {
            assert_eq!(method, "turn/create");
            assert_eq!(code, -32000);
            assert_eq!(message, "turn rejected");
        }
        other => panic!("expected SdkError::Rpc, got {other:?}"),
    }
}

#[tokio::test]
async fn unhandled_method_reports_a_missing_mock_handler() {
    let client = Client::new(MockTransport::new());
    let err = client
        .turn_start::<Turn>("thr_1", "go")
        .await
        .expect_err("no handler registered");
    assert!(matches!(err, SdkError::Unhandled(m) if m == "turn/create"));
}

#[tokio::test]
async fn item_subscribe_acks_then_drains_notifications() {
    let mock = MockTransport::new()
        .on(Method::ItemSubscribe, serde_json::json!({ "subscribed": true }))
        .push_notification(Notification::TurnStarted {
            turn: hide_protocol::ids::TurnId::from("trn_1"),
        })
        .push_notification(Notification::RuntimeStatus {
            status: "ready".into(),
            detail: None,
        });
    let client = Client::new(mock);

    let notifications = client.item_subscribe("thr_1").await.unwrap();
    assert_eq!(notifications.len(), 2, "both preloaded notifications drain");
    assert_eq!(notifications[0].method(), "turn/started");
    assert_eq!(notifications[1].method(), "runtime/status");

    // The subscribe request itself was sent as item/subscribe.
    let received = client.transport().received();
    assert_eq!(received.len(), 1);
    assert_eq!(received[0].method, Method::ItemSubscribe);

    // A second drain is empty: notifications are consumed.
    let again = client.transport().notifications().await;
    assert!(again.is_empty(), "notifications are drained once");
}
