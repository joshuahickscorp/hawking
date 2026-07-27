//! The WIRE-REACHABLE paths, driven the way a client drives them: `handle_intent` only.
//!
//! Every test here exists because a static reading of the code, and a suite of in-process tests
//! that called `host.dispatch_tool` directly with a run id, both said the diff review surface
//! worked while a real client driving a real hide-serve got nothing. The one write a client can
//! reach (`save_file`) went Intent -> fs connector -> ToolDispatcher, around `dispatch_tool`, so:
//!
//!   * no `DiffProposal` and no diff / diff_chip projection (the HunkReview surface had no
//!     producer at all), hence nothing to accept or reject;
//!   * no `tool.call` / `tool.result` pair, so the timeline never saw an app write;
//!   * no `diff.proposed` event, so a checkpoint's `repo_state` coverage was the empty digest and a
//!     CODE rewind reverted nothing on disk while reporting success.
//!
//! So NOTHING in this file may call `dispatch_tool`: a test that reaches past the wire is exactly
//! the test that passed while production was broken. Model-free, headless.

use hide_backend::host::HunkStatus;
use hide_backend::{BackendHost, BackendServices};
use hide_core::api::{Intent, UiEventKind};
use hide_core::config::HideConfig;
use hide_core::ids::{now_ms, RunId, SessionId};
use hide_core::types::Decision;
use serde_json::json;

/// A write-allowed headless host over a fresh temp workspace (the policy hold is not what is under
/// test here; `write_lease_trace_a_task_edits_and_the_diff_store_fills` covers that).
fn write_host(tag: &str) -> (BackendHost, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("hide_wire_{tag}_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = HideConfig::for_workspace(&dir);
    config.security.workspace_write_default = Decision::Allow;
    let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
    (host, dir)
}

/// The editor save AS A CLIENT SENDS IT: a custom intent carrying the session (the app's
/// `runCommand` fills `session_id` into every custom payload).
async fn save(
    host: &BackendHost,
    session: &SessionId,
    path: &str,
    content: &str,
) -> hide_core::api::IntentAck {
    host.handle_intent(Intent::Custom {
        name: "save_file".to_string(),
        payload: json!({ "path": path, "content": content, "session_id": session.as_str() }),
    })
    .await
    .unwrap()
}

/// Any custom intent, as a client sends it.
async fn custom(
    host: &BackendHost,
    name: &str,
    payload: serde_json::Value,
) -> hide_core::api::IntentAck {
    host.handle_intent(Intent::Custom {
        name: name.to_string(),
        payload,
    })
    .await
    .unwrap()
}

fn diff_id_for(session: &SessionId) -> String {
    format!("diff-{}", BackendHost::editor_run(session).as_str())
}

async fn kinds(host: &BackendHost, session: &SessionId) -> Vec<String> {
    host.services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap()
        .into_iter()
        .map(|e| e.kind)
        .collect()
}

/// D1: the write a client can actually reach registers an addressable diff, publishes the
/// projection the review surface binds, and records the tool pair the timeline reads.
#[tokio::test]
async fn save_through_the_wire_path_records_a_diff_and_publishes_it() {
    let (host, dir) = write_host("save");
    let session = host.services.session();
    let diff_id = diff_id_for(&session);
    std::fs::write(dir.join("pool.rs"), "fn pool() {}\n").unwrap();
    std::fs::write(dir.join("retry.rs"), "fn retry() {}\n").unwrap();
    let mut rx = host.subscribe_ui();

    let ack = save(&host, &session, "pool.rs", "fn pool() { /* capped */ }\n").await;
    assert!(ack.accepted && !ack.held, "the save is allowed: {ack:?}");
    save(&host, &session, "retry.rs", "fn retry() { /* jitter */ }\n").await;
    assert_eq!(
        std::fs::read_to_string(dir.join("pool.rs")).unwrap(),
        "fn pool() { /* capped */ }\n"
    );

    // The diff registry, which had no wire producer at all.
    let proposal = host
        .diff_get(&diff_id)
        .expect("a client save registers a DiffProposal");
    assert_eq!(proposal.hunks.len(), 2, "two saves = two addressable hunks");
    assert_eq!(proposal.hunks[0].before, "fn pool() {}\n");
    assert_eq!(proposal.hunks[0].after, "fn pool() { /* capped */ }\n");
    assert!(proposal.hunks.iter().all(|h| h.status == HunkStatus::Pending));

    // The projections the HunkReview + diff chip bind.
    let (mut diff, mut chip) = (false, false);
    while let Ok(ev) = rx.try_recv() {
        if let UiEventKind::ProjectionPatch { projection, .. } = &ev.kind {
            diff |= projection == "diff";
            chip |= projection == "diff_chip";
        }
    }
    assert!(diff && chip, "both diff projections publish for a client save");

    // The durable pair the timeline and the transcript search read.
    let kinds = kinds(&host, &session).await;
    assert!(kinds.iter().any(|k| k == "tool.call"), "{kinds:?}");
    assert!(kinds.iter().any(|k| k == "tool.result"), "{kinds:?}");
    assert_eq!(
        kinds.iter().filter(|k| *k == "diff.proposed").count(),
        2,
        "one diff.proposed per save: {kinds:?}"
    );

    // And a hunk is addressable: rejecting one reverts THAT file and keeps the other.
    let hunk = proposal
        .hunks
        .iter()
        .find(|h| h.file.ends_with("retry.rs"))
        .unwrap()
        .hunk_id
        .clone();
    let ack = host
        .handle_intent(Intent::RejectDiff {
            run_id: BackendHost::editor_run(&session),
            diff_id: diff_id.clone(),
            hunk_id: Some(hunk),
        })
        .await
        .unwrap();
    assert!(ack.accepted && !ack.held, "a single-hunk reject is not gated: {ack:?}");
    assert_eq!(
        std::fs::read_to_string(dir.join("retry.rs")).unwrap(),
        "fn retry() {}\n",
        "the rejected hunk is reverted on disk"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("pool.rs")).unwrap(),
        "fn pool() { /* capped */ }\n",
        "the kept hunk is untouched"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The verifying applier still refuses a stale `base_hash` on the wire path, and a refusal is NOT
/// a hold (no approval fixes a conflict) and leaves the bytes alone.
#[tokio::test]
async fn a_stale_base_hash_is_refused_on_the_wire_path() {
    let (host, dir) = write_host("conflict");
    let session = host.services.session();
    std::fs::write(dir.join("a.rs").as_path(), "one\n").unwrap();
    save(&host, &session, "a.rs", "two\n").await;

    let ack = host
        .handle_intent(Intent::Custom {
            name: "save_file".to_string(),
            payload: json!({
                "path": "a.rs",
                "content": "clobbered",
                "base_hash": "0".repeat(64),
                "session_id": session.as_str(),
            }),
        })
        .await
        .unwrap();
    assert!(!ack.accepted && !ack.held, "a conflict is refused, not held: {ack:?}");
    assert!(ack.message.unwrap_or_default().contains("refused"));
    assert_eq!(std::fs::read_to_string(dir.join("a.rs")).unwrap(), "two\n");
    let _ = std::fs::remove_dir_all(dir);
}

/// D2: a CODE rewind reverts what the app itself wrote. The checkpoint's `repo_state` coverage was
/// the empty digest and the rewind was a no-op on disk that still reported clean.
#[tokio::test]
async fn a_code_rewind_reverts_a_wire_save_on_disk() {
    let (host, dir) = write_host("rewind");
    let session = host.services.session();
    let file = dir.join("pool.rs");
    std::fs::write(&file, "fn pool() {}\n").unwrap();
    save(&host, &session, "pool.rs", "at the checkpoint\n").await;

    let checkpoint = host
        .checkpoint_create(session.clone(), None, "before the change")
        .await
        .unwrap();
    assert_eq!(
        checkpoint.coverage.repo_state.count, 1,
        "the checkpoint covers the file the app wrote"
    );

    save(&host, &session, "pool.rs", "after the checkpoint\n").await;
    assert_eq!(std::fs::read_to_string(&file).unwrap(), "after the checkpoint\n");

    // The wire shape: Ask -> gate -> approve. Nothing here reaches into the approved-write scope.
    let ack = host
        .handle_intent(Intent::Custom {
            name: "checkpoint_rewind".to_string(),
            payload: json!({ "checkpoint_id": checkpoint.checkpoint_id, "target": "code" }),
        })
        .await
        .unwrap();
    assert!(ack.held, "checkpoint_rewind is Ask: {ack:?}");
    let gate = ack
        .message
        .as_deref()
        .and_then(|m| m.split("gate=").nth(1))
        .map(|g| g.split_whitespace().next().unwrap_or(g).trim_end_matches(')').to_string())
        .expect("the hold names its gate");
    host.handle_intent(Intent::Custom {
        name: "approve_gate".to_string(),
        payload: json!({ "gate": gate }),
    })
    .await
    .unwrap();

    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "at the checkpoint\n",
        "the code rewind really reverts the working tree"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// D4: the sealed review receipt is exportable by a client, over a diff the client's own save made.
#[tokio::test]
async fn a_review_receipt_is_exportable_over_the_wire() {
    let (host, dir) = write_host("receipt");
    let session = host.services.session();
    std::fs::write(dir.join("a.rs"), "one\n").unwrap();
    save(&host, &session, "a.rs", "two\n").await;
    let diff_id = diff_id_for(&session);

    let ack = host
        .handle_intent(Intent::Custom {
            name: "export_review_receipt".to_string(),
            payload: json!({ "diff_id": diff_id, "session_id": session.as_str() }),
        })
        .await
        .unwrap();
    assert!(ack.accepted, "the receipt has a wire verb now: {ack:?}");
    let sealed = host.diff_review_receipts(&session).await.unwrap();
    assert_eq!(sealed.len(), 1, "one durable diff.receipt");
    assert_eq!(sealed[0].diff_id, diff_id);
    assert!(!sealed[0].seal.is_empty(), "the receipt is sealed");
    assert_eq!(sealed[0].hunks.len(), 1);

    // An unknown diff is refused honestly rather than sealing an empty body.
    let ack = host
        .handle_intent(Intent::Custom {
            name: "export_review_receipt".to_string(),
            payload: json!({ "diff_id": "diff-nope" }),
        })
        .await
        .unwrap();
    assert!(!ack.accepted, "{ack:?}");
    let _ = std::fs::remove_dir_all(dir);
}

/// D3: a client can attach to, stop, and capture the process it started. These host methods had no
/// wire trigger at all, so a started process could not be stopped from any client.
#[tokio::test]
async fn the_process_controls_are_reachable_over_the_wire() {
    // Fail-closed sandbox, as the other process traces do: no OS sandbox, no confined start.
    if !std::path::Path::new("/usr/bin/sandbox-exec").exists() {
        return;
    }
    let (host, dir) = write_host("process");
    let session = host.services.session();
    let mut rx = host.subscribe_ui();
    let id = host.start_process(
        vec![
            "sh".to_string(),
            "-c".to_string(),
            "i=0; while true; do echo tick $i; i=$((i+1)); sleep 0.1; done".to_string(),
        ],
        None,
        std::collections::BTreeMap::new(),
        true,
        Some(session.to_string()),
    );
    for _ in 0..100 {
        if host.process_state(&id).map(|s| s.line_count >= 2).unwrap_or(false) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
    }

    let ack = custom(
        &host,
        "attach_process",
        json!({ "process": id, "session_id": session.as_str() }),
    )
    .await;
    assert!(ack.accepted, "attach has a wire verb: {ack:?}");
    let ack = custom(&host, "capture_process_artifact", json!({ "process": id })).await;
    assert!(ack.accepted, "capture has a wire verb: {ack:?}");
    let ack = custom(&host, "stop_process", json!({ "process": id })).await;
    assert!(ack.accepted, "stop has a wire verb: {ack:?}");
    assert!(!host.process_alive(&id), "stop really stops the process");

    // The honest negative is about the PROCESS now, not about a missing handler.
    let ack = custom(&host, "stop_process", json!({ "process": "proc:nope" })).await;
    assert!(!ack.accepted);
    let msg = ack.message.unwrap_or_default();
    assert!(msg.contains("unknown process"), "{msg}");
    assert!(!msg.contains("no host handler"), "{msg}");

    let mut attached = false;
    let mut artifact = false;
    while let Ok(ev) = rx.try_recv() {
        if let UiEventKind::Custom(data) = &ev.kind {
            attached |= data.get("kind").and_then(|k| k.as_str()) == Some("process_attached");
            artifact |= data.get("kind").and_then(|k| k.as_str()) == Some("process_artifact");
        }
    }
    assert!(attached, "attach replays the buffered output to the client");
    assert!(artifact, "capture reports the durable artifact");
    let _ = std::fs::remove_dir_all(dir);
}

/// The contract itself: every name a client may send has a host arm, and the three process verbs
/// plus the receipt verb are on it. Guards the "recorded but has no host handler" ack the process
/// controls used to return.
#[test]
fn the_new_wire_names_are_on_the_contract() {
    for name in [
        "attach_process",
        "stop_process",
        "capture_process_artifact",
        "export_review_receipt",
    ] {
        assert!(
            hide_protocol::command::WIRE_CUSTOM_NAMES.contains(&name),
            "{name} must be on the wire contract"
        );
    }
    // and the run a save groups under is derived, not stored, so a restart addresses the same diff
    let session = SessionId::from("ses_test");
    assert_eq!(
        BackendHost::editor_run(&session),
        RunId::from("editor-ses_test")
    );
}
