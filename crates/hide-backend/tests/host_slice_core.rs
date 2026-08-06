//! Compact host tests for S3 slice behaviours (intent acceptance path).
use hide_backend::{BackendHost, BackendServices};
use hide_core::api::Intent;
use hide_core::config::HideConfig;
use hide_core::ids::SessionId;
use tempfile::tempdir;

fn test_host() -> (tempfile::TempDir, BackendHost) {
    let dir = tempdir().expect("tempdir");
    let config = HideConfig::for_workspace(dir.path());
    let host =
        BackendHost::from_services(BackendServices::open(config).expect("services")).expect("host");
    (dir, host)
}

#[tokio::test]
async fn submit_turn_intent_is_accepted() {
    let (_dir, host) = test_host();
    let session = host.services.session();
    let ack = host
        .handle_intent(Intent::SubmitTurn {
            session_id: session,
            text: "hello".into(),
            attachments: vec![],
        })
        .await
        .expect("intent");
    assert!(ack.accepted, "{ack:?}");
}

#[tokio::test]
async fn empty_submit_turn_returns_ack() {
    let (_dir, host) = test_host();
    let session = SessionId::new();
    let ack = host
        .handle_intent(Intent::SubmitTurn {
            session_id: session,
            text: "   ".into(),
            attachments: vec![],
        })
        .await
        .expect("intent");
    let _ = ack;
}

#[tokio::test]
async fn host_opens_with_session_services() {
    let (_dir, host) = test_host();
    let session = host.services.session();
    assert!(!session.as_str().is_empty());
}
