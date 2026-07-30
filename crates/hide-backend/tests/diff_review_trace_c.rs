use hide_backend::host::HunkStatus;
use hide_backend::{BackendHost, BackendServices};
use hide_core::api::Intent;
use hide_core::config::HideConfig;
use hide_core::ids::{now_ms, RunId};
use hide_core::tool::ToolCall;
use hide_core::types::Decision;
use hide_kernel::verify_plane::SourceFile;
use serde_json::json;
fn write_host(tag: &str) -> (BackendHost, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("hide_trace_c_{tag}_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = HideConfig::for_workspace(&dir);
    config.security.workspace_write_default = Decision::Allow;
    let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
    (host, dir)
}
async fn scripted_edit(
    host: &BackendHost,
    session: &hide_core::ids::SessionId,
    run: &RunId,
    path: &str,
    content: &str,
) {
    let result = host
        .dispatch_tool(
            session.clone(),
            Some(run.clone()),
            ToolCall::new(
                "edit.write_file",
                json!({ "path": path, "content": content }),
            ),
        )
        .await
        .unwrap();
    assert!(result.status == hide_core::tool::ToolStatus::Ok);
}
fn count_kind(events: &[hide_core::event::Event], kind: &str) -> usize {
    events.iter().filter(|e| e.kind == kind).count()
}
#[tokio::test]
async fn trace_c_hunk_addressable_diff_review() {
    let (host, dir) = write_host("full");
    let session = host.services.session();
    let run = RunId::new();
    let diff_id = format!("diff-{}", run.as_str());
    let path_a = dir.join("a.rs").to_string_lossy().to_string();
    let path_b = dir.join("b.rs").to_string_lossy().to_string();
    let (rel_a, rel_b) = ("a.rs", "b.rs");
    let orig_a = "pub fn a() -> u32 { 1 }\n";
    let orig_b = "pub fn b() -> u32 { 3 }\n";
    let new_a = "pub fn a() -> u32 { 2 }\n";
    let new_b = "pub fn b() -> u32 { 4 }\n";
    std::fs::write(&path_a, orig_a).unwrap();
    std::fs::write(&path_b, orig_b).unwrap();
    scripted_edit(&host, &session, &run, &path_a, new_a).await;
    scripted_edit(&host, &session, &run, &path_b, new_b).await;
    assert_eq!(std::fs::read_to_string(&path_a).unwrap(), new_a);
    assert_eq!(std::fs::read_to_string(&path_b).unwrap(), new_b);
    let proposal = host.diff_get(&diff_id).expect("diff registered");
    assert_eq!(proposal.hunks.len(), 2, "two edits = two addressable hunks");
    for h in &proposal.hunks {
        assert!(!h.base_hash.is_empty(), "hunk carries a base hash");
        assert_eq!(
            h.provenance.agent, "edit.write_file",
            "provenance names the agent"
        );
        assert_eq!(h.status, HunkStatus::Pending);
    }
    let hunk_a = proposal
        .hunks
        .iter()
        .find(|h| h.file == rel_a)
        .expect("hunk for file a")
        .clone();
    assert_eq!(hunk_a.before, orig_a);
    assert_eq!(hunk_a.after, new_a);
    assert_eq!(
        hunk_a.base_hash,
        blake3::hash(orig_a.as_bytes()).to_hex().to_string()
    );
    let before = host
        .run_static_analysis(
            session.clone(),
            vec![SourceFile::new(rel_a, new_a), SourceFile::new(rel_b, new_b)],
        )
        .await
        .unwrap();
    let before_id = before.receipt.verification_id.clone();
    assert_eq!(host.verification_receipts(&session).await.unwrap().len(), 1);
    let ack = host
        .handle_intent(Intent::RejectDiff {
            run_id: run.clone(),
            diff_id: diff_id.clone(),
            hunk_id: Some(hunk_a.hunk_id.clone()),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    assert_eq!(
        std::fs::read_to_string(&path_a).unwrap(),
        orig_a,
        "file a reverted"
    );
    assert_eq!(
        std::fs::read_to_string(&path_b).unwrap(),
        new_b,
        "file b kept"
    );
    let after_reject = host.diff_get(&diff_id).unwrap();
    let ha = after_reject.hunks.iter().find(|h| h.file == rel_a).unwrap();
    let hb = after_reject.hunks.iter().find(|h| h.file == rel_b).unwrap();
    assert_eq!(ha.status, HunkStatus::Rejected);
    assert_eq!(hb.status, HunkStatus::Pending);
    let invalidated = host.invalidated_verification_ids(&session).await.unwrap();
    assert!(
        invalidated.contains(&before_id),
        "rejected file invalidates its receipt"
    );
    let cur_a = std::fs::read_to_string(&path_a).unwrap();
    let cur_b = std::fs::read_to_string(&path_b).unwrap();
    let after = host
        .run_static_analysis(
            session.clone(),
            vec![SourceFile::new(rel_a, cur_a), SourceFile::new(rel_b, cur_b)],
        )
        .await
        .unwrap();
    let after_id = after.receipt.verification_id.clone();
    assert_ne!(after_id, before_id, "reverify mints a fresh receipt");
    let invalidated2 = host.invalidated_verification_ids(&session).await.unwrap();
    assert!(
        !invalidated2.contains(&after_id),
        "the fresh receipt is not invalidated"
    );
    let exported = host
        .export_diff_review_receipt(
            &diff_id,
            vec![before.receipt.clone()],
            vec![after.receipt.clone()],
        )
        .await
        .unwrap();
    assert!(!exported.seal.is_empty(), "receipt is sealed");
    let read_back = host.diff_review_receipts(&session).await.unwrap();
    assert_eq!(read_back.len(), 1);
    let rb = &read_back[0];
    assert_eq!(rb.seal, exported.seal);
    assert_eq!(rb.hunks.len(), 2);
    assert_eq!(rb.verification_before.len(), 1);
    assert_eq!(rb.verification_after.len(), 1);
    assert_eq!(rb.verification_before[0].verification_id, before_id);
    assert_eq!(rb.verification_after[0].verification_id, after_id);
    assert_eq!(
        rb.hunks
            .iter()
            .filter(|h| h.status == HunkStatus::Rejected)
            .count(),
        1
    );
    let events = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    assert_eq!(
        count_kind(&events, "diff.proposed"),
        2,
        "one per captured edit"
    );
    assert_eq!(count_kind(&events, "diff.hunk.rejected"), 1);
    assert_eq!(count_kind(&events, "verify.result"), 2, "before + after");
    assert_eq!(count_kind(&events, "verify.invalidated"), 1);
    assert_eq!(count_kind(&events, "diff.receipt"), 1);
    let rejected_evt = events
        .iter()
        .find(|e| e.kind == "diff.hunk.rejected")
        .unwrap();
    assert_eq!(
        rejected_evt.payload.get("hunk_id").and_then(|v| v.as_str()),
        Some(hunk_a.hunk_id.as_str())
    );
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn apply_hunk_and_apply_diff_keep_without_writing() {
    let (host, dir) = write_host("apply");
    let session = host.services.session();
    let run = RunId::new();
    let diff_id = format!("diff-{}", run.as_str());
    let path = dir.join("k.rs").to_string_lossy().to_string();
    std::fs::write(&path, "pub fn k() {}\n").unwrap();
    scripted_edit(&host, &session, &run, &path, "pub fn k() -> u8 { 0 }\n").await;
    let hunk = host.diff_get(&diff_id).unwrap().hunks[0].hunk_id.clone();
    let p = host.apply_hunk(&diff_id, &hunk).await.unwrap();
    assert_eq!(p.hunks[0].status, HunkStatus::Accepted);
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "pub fn k() -> u8 { 0 }\n"
    );
    let p2 = host.apply_diff(&diff_id).await.unwrap();
    assert_eq!(p2.hunks[0].status, HunkStatus::Accepted);
    assert!(host.apply_hunk(&diff_id, "nope").await.is_err());
    assert!(host.apply_diff("no-such-diff").await.is_err());
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn reject_hunk_reverts_and_revert_diff_undoes_all() {
    let (host, dir) = write_host("revert");
    let session = host.services.session();
    let run = RunId::new();
    let diff_id = format!("diff-{}", run.as_str());
    let path_a = dir.join("ra.rs").to_string_lossy().to_string();
    let path_b = dir.join("rb.rs").to_string_lossy().to_string();
    std::fs::write(&path_a, "A0\n").unwrap();
    std::fs::write(&path_b, "B0\n").unwrap();
    scripted_edit(&host, &session, &run, &path_a, "A1\n").await;
    scripted_edit(&host, &session, &run, &path_b, "B1\n").await;
    let ha = host
        .diff_get(&diff_id)
        .unwrap()
        .hunks
        .iter()
        .find(|h| h.file == "ra.rs")
        .unwrap()
        .hunk_id
        .clone();
    host.reject_hunk(&diff_id, &ha).await.unwrap();
    assert_eq!(std::fs::read_to_string(&path_a).unwrap(), "A0\n");
    assert_eq!(std::fs::read_to_string(&path_b).unwrap(), "B1\n");
    let p = hide_backend::tools::with_approved_writes(host.revert_diff(&diff_id))
        .await
        .unwrap();
    assert!(p.hunks.iter().all(|h| h.status == HunkStatus::Rejected));
    assert_eq!(std::fs::read_to_string(&path_b).unwrap(), "B0\n");
    let _ = std::fs::remove_dir_all(dir);
}
#[test]
fn optional_hunk_id_parses_backward_compatible() {
    let legacy: Intent = serde_json::from_value(json!({
        "type": "accept_diff",
        "data": { "run_id": "run_1", "diff_id": "d1" }
    }))
    .unwrap();
    match legacy {
        Intent::AcceptDiff {
            hunk_id, diff_id, ..
        } => {
            assert_eq!(hunk_id, None);
            assert_eq!(diff_id, "d1");
        }
        _ => panic!("expected AcceptDiff"),
    }
    let targeted: Intent = serde_json::from_value(json!({
        "type": "reject_diff",
        "data": { "run_id": "run_1", "diff_id": "d1", "hunk_id": "d1-h0" }
    }))
    .unwrap();
    match targeted {
        Intent::RejectDiff { hunk_id, .. } => assert_eq!(hunk_id.as_deref(), Some("d1-h0")),
        _ => panic!("expected RejectDiff"),
    }
}
#[tokio::test]
async fn whole_diff_revert_is_gated_whichever_payload_shape_asks_for_it() {
    let (host, dir) = write_host("gate_shape");
    let session = host.services.session();
    let run = RunId::new();
    let diff_id = format!("diff-{}", run.as_str());
    let path_a = dir.join("ga.rs").to_string_lossy().to_string();
    std::fs::write(&path_a, "A0\n").unwrap();
    scripted_edit(&host, &session, &run, &path_a, "A1\n").await;
    let ack = host
        .handle_intent(Intent::RejectDiff {
            run_id: run.clone(),
            diff_id: diff_id.clone(),
            hunk_id: None,
        })
        .await
        .unwrap();
    assert!(ack.accepted, "the request is recorded");
    assert!(
        ack.held,
        "a whole-diff revert asked for as reject_diff must be held, not run"
    );
    assert_eq!(std::fs::read_to_string(&path_a).unwrap(), "A1\n");
    assert!(host
        .diff_get(&diff_id)
        .unwrap()
        .hunks
        .iter()
        .all(|h| h.status != HunkStatus::Rejected));
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn a_refused_save_is_held_with_its_reason_and_approving_runs_it() {
    let dir = std::env::temp_dir().join(format!("hide_trace_c_save_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let host = BackendHost::open_workspace(&dir).unwrap();
    std::fs::write(dir.join("s.txt"), "old\n").unwrap();
    let mut rx = host.subscribe_ui();
    let ack = host
        .handle_intent(Intent::Custom {
            name: "save_file".to_string(),
            payload: json!({
                "path": "s.txt",
                "content": "new\n",
                "base_hash": blake3::hash(b"old\n").to_hex().to_string(),
            }),
        })
        .await
        .unwrap();
    assert!(ack.held, "a refused save is held at the gate");
    let message = ack.message.clone().unwrap_or_default();
    assert!(message.contains("policy"));
    assert_eq!(std::fs::read_to_string(dir.join("s.txt")).unwrap(), "old\n");
    let gate = loop {
        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should arrive")
            .expect("broadcast delivers");
        if let hide_core::api::UiEventKind::SecurityGate { gate, message } = ev.kind {
            assert!(message.contains("policy"), "the gate states why: {message}");
            break gate;
        }
    };
    host.handle_intent(Intent::Custom {
        name: "approve_gate".to_string(),
        payload: json!({ "gate": gate }),
    })
    .await
    .unwrap();
    assert_eq!(std::fs::read_to_string(dir.join("s.txt")).unwrap(), "new\n");
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn a_save_with_a_stale_base_hash_conflicts() {
    let (host, dir) = write_host("save_stale");
    std::fs::write(dir.join("c.txt"), "current\n").unwrap();
    let ack = host
        .handle_intent(Intent::Custom {
            name: "save_file".to_string(),
            payload: json!({
                "path": "c.txt",
                "content": "mine\n",
                "base_hash": blake3::hash(b"what the editor read\n").to_hex().to_string(),
            }),
        })
        .await
        .unwrap();
    assert!(!ack.held, "writes are allowed here, so nothing is gated");
    assert!(
        !ack.accepted,
        "a refused write must not ack as accepted: {ack:?}"
    );
    assert!(
        ack.message.is_some(),
        "the refusal carries the applier's reason"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("c.txt")).unwrap(),
        "current\n"
    );
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn an_agent_edit_publishes_the_diff_projection_and_a_status_change_republishes_it() {
    let (host, dir) = write_host("projection");
    let session = host.services.session();
    let run = RunId::new();
    let diff_id = format!("diff-{}", run.as_str());
    let path = dir.join("p.rs").to_string_lossy().to_string();
    std::fs::write(&path, "fn p() -> u32 { 1 }\n").unwrap();
    let mut rx = host.subscribe_ui();
    scripted_edit(&host, &session, &run, &path, "fn p() -> u32 { 2 }\n").await;
    let patch = next_diff_patch(&mut rx).await;
    assert_eq!(patch["diff_id"], json!(diff_id));
    assert_eq!(patch["run_id"], json!(run.as_str()));
    assert_eq!(
        patch["path"],
        json!("p.rs"),
        "the Monaco model names the file, workspace-relative"
    );
    assert_eq!(patch["lang"], json!("rust"));
    assert_eq!(patch["before"], json!("fn p() -> u32 { 1 }\n"));
    let hunks = patch["hunks"].as_array().expect("hunks is an array");
    assert_eq!(hunks.len(), 1);
    let h = &hunks[0];
    assert_eq!(h["id"], h["hunk_id"]);
    assert_eq!(h["status"], json!("pending"));
    assert_eq!(h["file"], json!("p.rs"));
    assert_eq!(
        h["base_hash"],
        json!(blake3::hash(b"fn p() -> u32 { 1 }\n").to_hex().to_string())
    );
    assert_eq!(h["provenance"]["agent"], json!("edit.write_file"));
    assert!(h["header"].as_str().unwrap().starts_with("@@ "));
    let kinds: Vec<&str> = h["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["kind"].as_str().unwrap())
        .collect();
    assert!(
        kinds.contains(&"del") && kinds.contains(&"add"),
        "{kinds:?}"
    );
    host.apply_hunk(&diff_id, h["hunk_id"].as_str().unwrap())
        .await
        .unwrap();
    let after = next_diff_patch(&mut rx).await;
    assert_eq!(after["hunks"][0]["status"], json!("accepted"));
    let _ = std::fs::remove_dir_all(dir);
}
async fn next_diff_patch(
    rx: &mut tokio::sync::broadcast::Receiver<hide_core::api::UiEvent>,
) -> serde_json::Value {
    loop {
        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should arrive")
            .expect("broadcast delivers");
        if let hide_core::api::UiEventKind::ProjectionPatch { projection, patch } = ev.kind {
            if projection == "diff" {
                return patch;
            }
        }
    }
}
#[tokio::test]
async fn a_kernel_dispatch_records_tool_events_and_a_revertible_hunk() {
    let (host, dir) = write_host("kernel_dispatch");
    let session = host.services.session();
    let run = RunId::new();
    let path = dir.join("k.rs").to_string_lossy().to_string();
    std::fs::write(&path, "fn k() -> u32 { 1 }\n").unwrap();
    let dispatcher = host.build_turn_dispatcher(session.clone(), Some(run.clone()));
    let result = dispatcher
        .dispatch(ToolCall::new(
            "edit.write_file",
            json!({ "path": path, "content": "fn k() -> u32 { 2 }\n" }),
        ))
        .await
        .unwrap();
    assert_eq!(result.status, hide_core::tool::ToolStatus::Ok);
    let events = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    assert_eq!(
        count_kind(&events, "tool.call"),
        1,
        "the agent's call is recorded"
    );
    assert_eq!(count_kind(&events, "tool.result"), 1);
    assert_eq!(
        count_kind(&events, "diff.proposed"),
        1,
        "and it produced a diff"
    );
    assert!(events
        .iter()
        .filter(|e| e.kind == "tool.call" || e.kind == "tool.result")
        .all(|e| e.run_id.as_ref() == Some(&run)));
    let proposal = host
        .diff_get(&format!("diff-{}", run.as_str()))
        .expect("diff registered");
    assert_eq!(proposal.hunks.len(), 1);
    let hunk = proposal.hunks[0].clone();
    assert_eq!(
        hunk.file, "k.rs",
        "workspace-relative, the spelling receipts use"
    );
    assert_eq!(hunk.before, "fn k() -> u32 { 1 }\n");
    assert_eq!(hunk.status, HunkStatus::Pending);
    host.reject_hunk(&proposal.diff_id, &hunk.hunk_id)
        .await
        .unwrap();
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "fn k() -> u32 { 1 }\n"
    );
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn a_wire_driven_rewind_invalidates_the_receipts_it_reverted() {
    let (host, dir) = write_host("wire_rewind");
    let session = host.services.session();
    std::fs::write(
        dir.join("w.rs").to_string_lossy().to_string(),
        "fn w() { }\n",
    )
    .unwrap();
    let intent = |name: &str, payload: serde_json::Value| Intent::Custom {
        name: name.to_string(),
        payload,
    };
    let sid = session.to_string();
    let ack = host
        .handle_intent(intent(
            "save_file",
            json!({ "session_id": sid, "path": "w.rs", "content": "fn w() { let _ = x.unwrap(); }\n" }),
        ))
        .await
        .unwrap();
    assert!(ack.accepted && !ack.held, "{ack:?}");
    let ack = host
        .handle_intent(intent(
            "run_static_analysis",
            json!({ "session_id": sid, "paths": ["w.rs"] }),
        ))
        .await
        .unwrap();
    assert!(ack.accepted, "{ack:?}");
    let receipt_id = host.verification_receipts(&session).await.unwrap()[0]
        .receipt
        .verification_id
        .clone();
    let checkpoint = host
        .checkpoint_create(session.clone(), None, "before")
        .await
        .unwrap();
    let ack = host
        .handle_intent(intent(
            "save_file",
            json!({ "session_id": sid, "path": "w.rs", "content": "fn w() { let _ = y.unwrap(); }\n" }),
        ))
        .await
        .unwrap();
    assert!(ack.accepted && !ack.held, "{ack:?}");
    let mut rx = host.subscribe_ui();
    let ack = host
        .handle_intent(intent(
            "checkpoint_rewind",
            json!({ "checkpoint_id": checkpoint.checkpoint_id, "target": "code" }),
        ))
        .await
        .unwrap();
    assert!(
        ack.held,
        "an Ask command is held, never run on arrival: {ack:?}"
    );
    let gate = loop {
        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should arrive")
            .expect("broadcast delivers");
        if let hide_core::api::UiEventKind::SecurityGate { gate, .. } = ev.kind {
            break gate;
        }
    };
    let ack = host
        .handle_intent(intent("approve_gate", json!({ "gate": gate })))
        .await
        .unwrap();
    assert!(ack.accepted, "the released rewind ran: {ack:?}");
    assert_eq!(
        std::fs::read_to_string(dir.join("w.rs")).unwrap(),
        "fn w() { let _ = x.unwrap(); }\n"
    );
    let invalidated = host.invalidated_verification_ids(&session).await.unwrap();
    assert!(invalidated.contains(&receipt_id));
    let _ = std::fs::remove_dir_all(dir);
}
#[tokio::test]
async fn a_per_hunk_reject_is_held_for_approval_on_the_shipped_default() {
    let dir = std::env::temp_dir().join(format!("hide_trace_c_reject_ask_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let host = BackendHost::open_workspace(&dir).unwrap();
    let session = host.services.session();
    std::fs::write(dir.join("u.txt"), "old\n").unwrap();
    let mut rx = host.subscribe_ui();
    async fn next_gate(
        rx: &mut tokio::sync::broadcast::Receiver<hide_core::api::UiEvent>,
    ) -> String {
        loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let hide_core::api::UiEventKind::SecurityGate { gate, .. } = ev.kind {
                return gate;
            }
        }
    }
    let approve = |gate: String| {
        host.handle_intent(Intent::Custom {
            name: "approve_gate".to_string(),
            payload: json!({ "gate": gate }),
        })
    };
    host.handle_intent(Intent::Custom {
        name: "save_file".to_string(),
        payload: json!({ "session_id": session.to_string(), "path": "u.txt", "content": "new\n" }),
    })
    .await
    .unwrap();
    let gate = next_gate(&mut rx).await;
    assert!(approve(gate).await.unwrap().accepted);
    assert_eq!(std::fs::read_to_string(dir.join("u.txt")).unwrap(), "new\n");
    let proposal = host
        .diff_get(&format!("diff-editor-{}", session.as_str()))
        .expect("the approved save recorded a hunk");
    let hunk_id = proposal.hunks[0].hunk_id.clone();
    let ack = host
        .handle_intent(Intent::RejectDiff {
            run_id: RunId::from(format!("editor-{}", session.as_str())),
            diff_id: proposal.diff_id.clone(),
            hunk_id: Some(hunk_id.clone()),
        })
        .await
        .unwrap();
    assert!(
        ack.held,
        "a policy-refused undo is offered for approval, not thrown away: {ack:?}"
    );
    assert_eq!(std::fs::read_to_string(dir.join("u.txt")).unwrap(), "new\n");
    let gate = next_gate(&mut rx).await;
    assert!(approve(gate).await.unwrap().accepted);
    assert_eq!(std::fs::read_to_string(dir.join("u.txt")).unwrap(), "old\n");
    assert_eq!(
        host.diff_get(&proposal.diff_id).unwrap().hunks[0].status,
        HunkStatus::Rejected
    );
    let _ = std::fs::remove_dir_all(dir);
}
