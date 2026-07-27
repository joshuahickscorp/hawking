//! TRACE C (HIDE census sec 23): hunk-addressable diff review with provenance,
//! model-free, headless, through the REAL host path with scripted drivers.
//!
//! The edit flow is IMMEDIATE: the `edit.*` catalog tools apply and re-verify to
//! disk during the turn. So a diff is the set of changes already on disk; keeping
//! a hunk marks it accepted, rejecting a hunk reverts it on disk via an inverse
//! write through the same verifying applier. This trace drives that end to end:
//!
//! 1. A scripted driver applies a two-file patch via the REAL edit tools (two
//!    `host.dispatch_tool` calls under one run_id).
//! 2. `diff_get` shows each hunk with provenance + base_hash.
//! 3. Affected verification runs (`run_static_analysis`) and yields a receipt.
//! 4. ONE hunk is rejected (the real `RejectDiff` intent with a hunk_id); that
//!    file is reverted on disk while the other file keeps its change.
//! 5. The invalidated scope is reverified and yields a FRESH receipt.
//! 6. The sealed review receipt is exported and read back.
//!
//! Every step appends a durable event; the event log is asserted to agree with
//! the projection (`diff_get`). No model, no subprocess, no staged model download.

use hide_backend::host::HunkStatus;
use hide_backend::{BackendHost, BackendServices};
use hide_core::api::Intent;
use hide_core::config::HideConfig;
use hide_core::ids::{now_ms, RunId};
use hide_core::tool::ToolCall;
use hide_core::types::Decision;
use hide_verify::SourceFile;
use serde_json::json;

/// A write-allowed headless host over a fresh temp workspace.
fn write_host(tag: &str) -> (BackendHost, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("hide_trace_c_{tag}_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = HideConfig::for_workspace(&dir);
    config.security.workspace_write_default = Decision::Allow;
    let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
    (host, dir)
}

/// Apply a whole-file edit through the REAL `edit.write_file` tool via the host
/// dispatch path under `run`, returning nothing (the change lands on disk + is
/// captured as a diff hunk).
async fn scripted_edit(host: &BackendHost, session: &hide_core::ids::SessionId, run: &RunId, path: &str, content: &str) {
    let result = host
        .dispatch_tool(
            session.clone(),
            Some(run.clone()),
            ToolCall::new("edit.write_file", json!({ "path": path, "content": content })),
        )
        .await
        .unwrap();
    assert!(
        result.status == hide_core::tool::ToolStatus::Ok,
        "scripted edit must apply: {result:?}"
    );
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
    // What a hunk and a receipt BOTH name the file: workspace-relative. The edits below are
    // dispatched with the absolute path a client sends, and the receipts are sealed over the
    // relative one `run_static_analysis` gets from the editor, which is the production pairing.
    let (rel_a, rel_b) = ("a.rs", "b.rs");
    let orig_a = "pub fn a() -> u32 { 1 }\n";
    let orig_b = "pub fn b() -> u32 { 3 }\n";
    let new_a = "pub fn a() -> u32 { 2 }\n";
    let new_b = "pub fn b() -> u32 { 4 }\n";
    std::fs::write(&path_a, orig_a).unwrap();
    std::fs::write(&path_b, orig_b).unwrap();

    // 1. Scripted driver applies a two-file patch via the REAL edit tools.
    scripted_edit(&host, &session, &run, &path_a, new_a).await;
    scripted_edit(&host, &session, &run, &path_b, new_b).await;
    assert_eq!(std::fs::read_to_string(&path_a).unwrap(), new_a);
    assert_eq!(std::fs::read_to_string(&path_b).unwrap(), new_b);

    // 2. Open the diff: each hunk carries provenance + base_hash.
    let proposal = host.diff_get(&diff_id).expect("diff registered");
    assert_eq!(proposal.hunks.len(), 2, "two edits = two addressable hunks");
    for h in &proposal.hunks {
        assert!(!h.base_hash.is_empty(), "hunk carries a base hash");
        assert_eq!(h.provenance.agent, "edit.write_file", "provenance names the agent");
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
    // base_hash is the blake3 of the pre-image.
    assert_eq!(
        hunk_a.base_hash,
        blake3::hash(orig_a.as_bytes()).to_hex().to_string()
    );

    // 3. Run affected verification over the changed files -> a receipt (before).
    let before = host
        .run_static_analysis(
            session.clone(),
            vec![
                SourceFile::new(rel_a, new_a),
                SourceFile::new(rel_b, new_b),
            ],
        )
        .await
        .unwrap();
    let before_id = before.receipt.verification_id.clone();
    assert_eq!(host.verification_receipts(&session).await.unwrap().len(), 1);

    // 4. Reject ONE hunk via the REAL RejectDiff intent + hunk_id path. File a is
    //    reverted on disk; file b keeps its change.
    let ack = host
        .handle_intent(Intent::RejectDiff {
            run_id: run.clone(),
            diff_id: diff_id.clone(),
            hunk_id: Some(hunk_a.hunk_id.clone()),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    assert_eq!(std::fs::read_to_string(&path_a).unwrap(), orig_a, "file a reverted");
    assert_eq!(std::fs::read_to_string(&path_b).unwrap(), new_b, "file b kept");

    // The projection reflects the reject: hunk a Rejected, hunk b still Pending.
    let after_reject = host.diff_get(&diff_id).unwrap();
    let ha = after_reject.hunks.iter().find(|h| h.file == rel_a).unwrap();
    let hb = after_reject.hunks.iter().find(|h| h.file == rel_b).unwrap();
    assert_eq!(ha.status, HunkStatus::Rejected);
    assert_eq!(hb.status, HunkStatus::Pending);

    // The before-receipt is now invalidated (its scope intersects file a).
    let invalidated = host.invalidated_verification_ids(&session).await.unwrap();
    assert!(invalidated.contains(&before_id), "rejected file invalidates its receipt");

    // 5. Reverify the invalidated scope over the CURRENT disk state -> fresh receipt.
    let cur_a = std::fs::read_to_string(&path_a).unwrap();
    let cur_b = std::fs::read_to_string(&path_b).unwrap();
    let after = host
        .run_static_analysis(
            session.clone(),
            vec![
                SourceFile::new(rel_a, cur_a),
                SourceFile::new(rel_b, cur_b),
            ],
        )
        .await
        .unwrap();
    let after_id = after.receipt.verification_id.clone();
    assert_ne!(after_id, before_id, "reverify mints a fresh receipt");
    let invalidated2 = host.invalidated_verification_ids(&session).await.unwrap();
    assert!(!invalidated2.contains(&after_id), "the fresh receipt is not invalidated");

    // 6. Export the sealed review receipt and read it back.
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
    // The sealed hunks agree with the projection (one Rejected, one Pending).
    assert_eq!(
        rb.hunks.iter().filter(|h| h.status == HunkStatus::Rejected).count(),
        1
    );

    // Every step appended a durable event; the log agrees with the projection.
    let events = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    assert_eq!(count_kind(&events, "diff.proposed"), 2, "one per captured edit");
    assert_eq!(count_kind(&events, "diff.hunk.rejected"), 1);
    assert_eq!(count_kind(&events, "verify.result"), 2, "before + after");
    assert_eq!(count_kind(&events, "verify.invalidated"), 1);
    assert_eq!(count_kind(&events, "diff.receipt"), 1);
    // The rejected event names the rejected hunk (provenance carried).
    let rejected_evt = events.iter().find(|e| e.kind == "diff.hunk.rejected").unwrap();
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

    // apply_hunk marks Accepted; the on-disk content is untouched (already applied).
    let p = host.apply_hunk(&diff_id, &hunk).await.unwrap();
    assert_eq!(p.hunks[0].status, HunkStatus::Accepted);
    assert_eq!(std::fs::read_to_string(&path).unwrap(), "pub fn k() -> u8 { 0 }\n");

    // apply_diff over an already-accepted diff is a no-op on disk and idempotent.
    let p2 = host.apply_diff(&diff_id).await.unwrap();
    assert_eq!(p2.hunks[0].status, HunkStatus::Accepted);

    // Unknown ids error cleanly.
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

    // reject_hunk on a directly reverts a, leaves b.
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

    // revert_diff undoes the rest (b) and skips the already-rejected a. It is `ApprovalPolicy::Ask`
    // and the policy is enforced at the EFFECT, so no channel routes around it: a direct call has to
    // stand in the released-gate scope, exactly as `run_approved_intent` does.
    let p = hide_backend::tools::with_approved_writes(host.revert_diff(&diff_id))
        .await
        .unwrap();
    assert!(p.hunks.iter().all(|h| h.status == HunkStatus::Rejected));
    assert_eq!(std::fs::read_to_string(&path_b).unwrap(), "B0\n");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn optional_hunk_id_parses_backward_compatible() {
    // Legacy payload (no hunk_id) deserializes with hunk_id = None.
    let legacy: Intent = serde_json::from_value(json!({
        "type": "accept_diff",
        "data": { "run_id": "run_1", "diff_id": "d1" }
    }))
    .unwrap();
    match legacy {
        Intent::AcceptDiff { hunk_id, diff_id, .. } => {
            assert_eq!(hunk_id, None);
            assert_eq!(diff_id, "d1");
        }
        _ => panic!("expected AcceptDiff"),
    }

    // New payload carries a hunk_id.
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

/// The approval policy attaches to the EFFECT, not to the name that carried the request.
/// `reject_diff` with no `hunk_id` asks for the SAME whole-diff on-disk revert that `revert_diff`
/// declares `Ask` for, so it is held at the same gate. Before this it ran ungated, which meant a
/// shipped button reached a gated effect by sending the other payload shape.
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
    assert_eq!(
        std::fs::read_to_string(&path_a).unwrap(),
        "A1\n",
        "nothing on disk was reverted while the effect is parked at the gate"
    );
    assert!(
        host.diff_get(&diff_id)
            .unwrap()
            .hunks
            .iter()
            .all(|h| h.status != HunkStatus::Rejected),
        "and no hunk was marked decided"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A save the permission policy refuses is HELD with the policy's own reason and an approval path,
/// not thrown away as a bare failure. The shipped default (workspace_write_default = Ask) refuses
/// every save, so this is the ordinary path, and `base_hash` rides through it.
#[tokio::test]
async fn a_refused_save_is_held_with_its_reason_and_approving_runs_it() {
    let dir = std::env::temp_dir().join(format!("hide_trace_c_save_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    // The SHIPPED default config: no write is allowed outright.
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
    assert!(
        message.contains("policy"),
        "the ack carries the policy's real reason, not a generic failure: {message}"
    );
    assert_eq!(std::fs::read_to_string(dir.join("s.txt")).unwrap(), "old\n");

    // The gate the user sees carries that same reason and an id to approve.
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
    assert_eq!(
        std::fs::read_to_string(dir.join("s.txt")).unwrap(),
        "new\n",
        "approving the gate performs the save that was held"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The save's concurrency guard is real: a `base_hash` that no longer matches the file on disk
/// conflicts instead of clobbering the change made since the buffer was read.
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
    // The refusal has to reach the ack. It used to be an Error UiEvent beside an accepted ack,
    // so the editor printed "saved c.txt" for the write the applier had just rejected.
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
        "current\n",
        "a stale base_hash must not clobber the file"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The diff-review surface has a LIVE producer: a real agent edit publishes the `diff`
/// projection the frontend reads (app/src/surfaces/ide/types.ts parseDiff), carrying the
/// per-hunk provenance and base_hash, and a hunk status change republishes it. Before this,
/// the only diff surfacing was an untyped Custom event the frontend routes nowhere, so on a
/// live host the review rendered nothing at all.
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
    // The id the view model reads AND the id the wire addresses are both present.
    assert_eq!(h["id"], h["hunk_id"]);
    assert_eq!(h["status"], json!("pending"));
    assert_eq!(h["file"], json!("p.rs"));
    assert_eq!(
        h["base_hash"],
        json!(blake3::hash(b"fn p() -> u32 { 1 }\n").to_hex().to_string())
    );
    assert_eq!(
        h["provenance"]["agent"],
        json!("edit.write_file"),
        "the hunk carries the provenance the host recorded"
    );
    assert!(
        h["header"].as_str().unwrap().starts_with("@@ "),
        "header is the @@ context label the review renders: {}",
        h["header"]
    );
    let kinds: Vec<&str> = h["lines"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["kind"].as_str().unwrap())
        .collect();
    assert!(kinds.contains(&"del") && kinds.contains(&"add"), "{kinds:?}");

    // A hunk status change republishes the projection with the new status.
    host.apply_hunk(&diff_id, h["hunk_id"].as_str().unwrap())
        .await
        .unwrap();
    let after = next_diff_patch(&mut rx).await;
    assert_eq!(after["hunks"][0]["status"], json!("accepted"));
    let _ = std::fs::remove_dir_all(dir);
}

/// The next `projection_patch{projection:"diff"}` off the UI bus.
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

/// THE AGENT'S OWN WRITE PATH, exercised as the agent exercises it.
///
/// The kernel never calls `host.dispatch_tool`: `build_turn_kernel` hands it a `ToolDispatcher`
/// and `hide_kernel::machine::driver` calls `dispatcher.dispatch(ToolCall::new(tool, args))`
/// straight off that object. This test builds the SAME object the same way (the production
/// `build_turn_dispatcher` the kernel builder uses) and dispatches through it exactly as the
/// driver does, so what it proves is that a dispatch owned by the kernel - not a host wrapper -
/// records the tool events and the reviewable, revertible hunk.
///
/// What it does NOT prove: the turn itself. Driving a real turn needs a served model, and this
/// build serves none (DEFERRED_MODEL_REQUIRED), so the step that CHOOSES the call is unexercised.
/// Everything from the dispatcher down is the production path, byte for byte.
#[tokio::test]
async fn a_kernel_dispatch_records_tool_events_and_a_revertible_hunk() {
    let (host, dir) = write_host("kernel_dispatch");
    let session = host.services.session();
    let run = RunId::new();
    let path = dir.join("k.rs").to_string_lossy().to_string();
    std::fs::write(&path, "fn k() -> u32 { 1 }\n").unwrap();

    // The object the kernel holds (host.rs `build_turn_kernel` -> `.dispatcher(..)`).
    let dispatcher = host.build_turn_dispatcher(session.clone(), Some(run.clone()));
    // The call the driver makes: no session, no run, no host in sight.
    let result = dispatcher
        .dispatch(ToolCall::new(
            "edit.write_file",
            json!({ "path": path, "content": "fn k() -> u32 { 2 }\n" }),
        ))
        .await
        .unwrap();
    assert_eq!(result.status, hide_core::tool::ToolStatus::Ok);

    // 1. The tool events the timeline and transcript search read.
    let events = host.services.event_log.scan(Some(session.clone()), None, None).await.unwrap();
    assert_eq!(count_kind(&events, "tool.call"), 1, "the agent's call is recorded");
    assert_eq!(count_kind(&events, "tool.result"), 1);
    assert_eq!(count_kind(&events, "diff.proposed"), 1, "and it produced a diff");
    assert!(
        events
            .iter()
            .filter(|e| e.kind == "tool.call" || e.kind == "tool.result")
            .all(|e| e.run_id.as_ref() == Some(&run)),
        "grouped under the turn's run, not orphaned"
    );

    // 2. The hunk: addressable, per-hunk revertible, named the way a receipt names a file.
    let proposal = host.diff_get(&format!("diff-{}", run.as_str())).expect("diff registered");
    assert_eq!(proposal.hunks.len(), 1);
    let hunk = proposal.hunks[0].clone();
    assert_eq!(hunk.file, "k.rs", "workspace-relative, the spelling receipts use");
    assert_eq!(hunk.before, "fn k() -> u32 { 1 }\n");
    assert_eq!(hunk.status, HunkStatus::Pending);

    // 3. The consequence that matters: the agent's edit can be undone.
    host.reject_hunk(&proposal.diff_id, &hunk.hunk_id).await.unwrap();
    assert_eq!(
        std::fs::read_to_string(&path).unwrap(),
        "fn k() -> u32 { 1 }\n",
        "an agent edit is revertible per hunk"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// A rewind driven the way a client drives it invalidates the receipts covering what it reverted.
///
/// Every step here is a wire-reachable intent (`/v1/hide/intent` routes straight into
/// `handle_intent`): save, verify, checkpoint, save again, rewind-with-approval. The receipt is
/// sealed by `run_static_analysis` (workspace-relative scope) and the hunk is recorded by the
/// dispatch recorder, and the point of the test is that those two spellings MEET: they used to be
/// relative and absolute respectively, so `paths_intersect` could never match and no wire-driven
/// rewind could invalidate anything.
#[tokio::test]
async fn a_wire_driven_rewind_invalidates_the_receipts_it_reverted() {
    let (host, dir) = write_host("wire_rewind");
    let session = host.services.session();
    std::fs::write(dir.join("w.rs").to_string_lossy().to_string(), "fn w() { }\n").unwrap();

    let intent = |name: &str, payload: serde_json::Value| Intent::Custom {
        name: name.to_string(),
        payload,
    };
    let sid = session.to_string();

    // 1. The client's save (the one wire-reachable workspace write).
    let ack = host
        .handle_intent(intent(
            "save_file",
            json!({ "session_id": sid, "path": "w.rs", "content": "fn w() { let _ = x.unwrap(); }\n" }),
        ))
        .await
        .unwrap();
    assert!(ack.accepted && !ack.held, "{ack:?}");

    // 2. The client's verification over that file -> a durable receipt scoping "w.rs".
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

    // 3. Checkpoint, then a further save AFTER the boundary (what the rewind must undo).
    let checkpoint = host.checkpoint_create(session.clone(), None, "before").await.unwrap();
    let ack = host
        .handle_intent(intent(
            "save_file",
            json!({ "session_id": sid, "path": "w.rs", "content": "fn w() { let _ = y.unwrap(); }\n" }),
        ))
        .await
        .unwrap();
    assert!(ack.accepted && !ack.held, "{ack:?}");

    // 4. Rewind the code, as a client must: the intent is HELD (checkpoint_rewind is
    //    ApprovalPolicy::Ask) and released by an approve_gate intent.
    let mut rx = host.subscribe_ui();
    let ack = host
        .handle_intent(intent(
            "checkpoint_rewind",
            json!({ "checkpoint_id": checkpoint.checkpoint_id, "target": "code" }),
        ))
        .await
        .unwrap();
    assert!(ack.held, "an Ask command is held, never run on arrival: {ack:?}");
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

    // The bytes went back...
    assert_eq!(
        std::fs::read_to_string(dir.join("w.rs")).unwrap(),
        "fn w() { let _ = x.unwrap(); }\n",
        "the post-boundary save is reverted on disk"
    );
    // ...and the receipt covering that file is no longer claimed as valid.
    let invalidated = host.invalidated_verification_ids(&session).await.unwrap();
    assert!(
        invalidated.contains(&receipt_id),
        "a wire-driven rewind invalidates the receipts whose scope it touched: {invalidated:?}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

/// The review surface's undo works on the SHIPPED default.
///
/// `workspace_write_default = Ask` refuses every write, and a per-hunk reject writes (the inverse
/// write that puts the pre-image back). It used to be REFUSED outright, so on the default config
/// the undo button was dead with no approval offered - while its sibling, the save, was held and
/// approvable. Both now take the one hold-and-approve rule.
#[tokio::test]
async fn a_per_hunk_reject_is_held_for_approval_on_the_shipped_default() {
    let dir = std::env::temp_dir().join(format!("hide_trace_c_reject_ask_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let host = BackendHost::open_workspace(&dir).unwrap();
    let session = host.services.session();
    std::fs::write(dir.join("u.txt"), "old\n").unwrap();
    let mut rx = host.subscribe_ui();

    async fn next_gate(rx: &mut tokio::sync::broadcast::Receiver<hide_core::api::UiEvent>) -> String {
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

    // A save, held and approved, so there is a hunk to undo.
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

    // The undo: held, not refused.
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

    // And approving it actually undoes the hunk.
    let gate = next_gate(&mut rx).await;
    assert!(approve(gate).await.unwrap().accepted);
    assert_eq!(
        std::fs::read_to_string(dir.join("u.txt")).unwrap(),
        "old\n",
        "approving the held undo reverts the hunk on disk"
    );
    assert_eq!(
        host.diff_get(&proposal.diff_id).unwrap().hunks[0].status,
        HunkStatus::Rejected
    );
    let _ = std::fs::remove_dir_all(dir);
}
