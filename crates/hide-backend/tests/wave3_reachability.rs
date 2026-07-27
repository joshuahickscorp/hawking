//! Backend wave 3 (model-free) FE-reachability + campaign Trace B.
//!
//! Three headless, deterministic, model-free integration proofs that built host
//! capabilities are now reachable from the surfaces the FE actually consumes:
//!
//!   1. Trace B (campaign sec 23): a selected-code READ-ONLY side chat folds a
//!      CONCISE TYPED result back onto its parent -- never the full child
//!      transcript -- with cited evidence and recorded ancestry.
//!   2. The `diagnostics` PROJECTION feed (the StatusBar Problems counter binds to
//!      real error/warning counts instead of a hardcoded 0/0).
//!   3. Transcript SEARCH over the `/intent` custom (literal + structured filters;
//!      semantic search stays DEFERRED_MODEL_REQUIRED).
//!
//! Everything here is the real host: no model, no subprocess, no staged artifact.

use hide_backend::{BackendHost, EvidenceLink, SessionRelationship, SideChatResult, TranscriptQuery};
use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::event::NewEvent;
use hide_core::ids::now_ms;
use hide_verify::SourceFile;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

fn test_host() -> BackendHost {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hide_wave3_{}_{}", now_ms(), uniq));
    BackendHost::open_workspace(&dir).unwrap()
}

/// The first `ProjectionPatch` on the bus whose projection NAME matches, or a
/// panic on timeout. Non-matching events (other projections / Custom) are skipped.
async fn next_projection(
    rx: &mut tokio::sync::broadcast::Receiver<UiEvent>,
    name: &str,
) -> serde_json::Value {
    loop {
        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should arrive")
            .expect("broadcast delivers");
        if let UiEventKind::ProjectionPatch { projection, patch } = ev.kind {
            if projection == name {
                return patch;
            }
        }
    }
}

/// The first `Custom` UiEvent on the bus whose `kind` field matches, or a panic on
/// timeout.
async fn next_custom(
    rx: &mut tokio::sync::broadcast::Receiver<UiEvent>,
    kind: &str,
) -> serde_json::Value {
    loop {
        let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should arrive")
            .expect("broadcast delivers");
        if let UiEventKind::Custom(value) = ev.kind {
            if value.get("kind").and_then(|k| k.as_str()) == Some(kind) {
                return value;
            }
        }
    }
}

// --- (1) Trace B: selected-code side chat, concise typed foldback -------------

#[tokio::test]
async fn trace_b_side_chat_folds_a_bounded_typed_result_not_the_full_transcript() {
    let host = test_host();
    let parent = host.services.session();
    let log = host.services.event_log.clone();
    let mut rx = host.subscribe_ui();

    // (0) A stable source ref + hash over a fixture file (the "selected code").
    let fixture_path = "src/net.rs";
    let fixture = "pub fn parse_port(raw: &str) -> u16 {\n    raw.parse::<u16>().unwrap_or(0)\n}\n";
    let selection_hash = hide_verify::source_hash_of([(fixture_path, fixture)]);
    let selection_ref = format!("{fixture_path}#{selection_hash}");

    // (1) Seed the PARENT with references to the selected symbol -- what the child
    // will cite as evidence.
    for text in [
        "parse_port is called from the listener bootstrap",
        "parse_port is also referenced in the config loader",
    ] {
        log.append(NewEvent::system(
            parent.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": text }),
        ))
        .await
        .unwrap();
    }
    let parent_before = log.scan(Some(parent.clone()), None, None).await.unwrap().len();

    // (2) Create a READ-ONLY investigation side chat about the selection.
    let (child, child_record, _proj) = host
        .create_side_chat(parent.clone(), None, true)
        .await
        .unwrap();

    // Ancestry is recorded on the child record: it points back at the parent as a
    // read-only SideChat.
    assert_eq!(child_record.parent_session_id.as_ref(), Some(&parent));
    assert_eq!(child_record.relationship, SessionRelationship::SideChat);
    assert!(child_record.read_only, "an investigation side chat is read-only");

    // (3) The child records its OWN private investigation (a unique token that must
    // NEVER leak onto the parent on merge).
    const CHILD_SECRET: &str = "CHILDONLYSECRET_do_not_leak";
    log.append(NewEvent::system(
        child.clone(),
        "agent.message",
        json!({ "role": "assistant", "text": format!("investigating {selection_ref}: {CHILD_SECRET}") }),
    ))
    .await
    .unwrap();

    // (4) The child searches references (the model-free search path) and cites the
    // hits as evidence.
    let hits = host
        .search_transcript(&TranscriptQuery::literal("parse_port").in_session(parent.clone()))
        .await
        .unwrap();
    assert!(hits.len() >= 2, "the child finds the seeded references: {hits:?}");
    let evidence: Vec<EvidenceLink> = hits.iter().map(EvidenceLink::from_hit).collect();

    // (5) Merge a CONCISE TYPED result back onto the parent.
    let result = SideChatResult::new(
        format!(
            "parse_port is referenced in {} places; selection {selection_ref}",
            hits.len()
        ),
        evidence.clone(),
        "investigation",
    );
    let merged = host
        .merge_side_chat_result(child.clone(), parent.clone(), result)
        .await
        .unwrap();

    // (A) The parent transcript grew by EXACTLY ONE typed merge event.
    let parent_after = log.scan(Some(parent.clone()), None, None).await.unwrap();
    assert_eq!(
        parent_after.len(),
        parent_before + 1,
        "the parent gains exactly one bounded merge event"
    );
    let merge_events: Vec<_> = parent_after
        .iter()
        .filter(|e| e.kind == "session.merge_summary")
        .collect();
    assert_eq!(merge_events.len(), 1);
    assert_eq!(merge_events[0].id, merged.id);

    // (B) It is a typed foldback: summary + evidence + kind, NOT the child
    // transcript. The child's private token appears in NO parent event.
    let payload = &merge_events[0].payload;
    assert_eq!(payload.get("kind").and_then(|v| v.as_str()), Some("investigation"));
    assert!(payload
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap()
        .contains("parse_port is referenced"));
    assert_eq!(
        payload.get("evidence").and_then(|v| v.as_array()).unwrap().len(),
        evidence.len()
    );
    for e in &parent_after {
        assert!(
            !serde_json::to_string(&e.payload).unwrap().contains(CHILD_SECRET),
            "the full child transcript must NEVER be folded into the parent"
        );
    }

    // (C) The evidence cites a REAL parent transcript item (session + event id).
    let cited_event = payload["evidence"][0]["event_id"].as_str().unwrap();
    assert!(
        parent_after.iter().any(|e| e.id.as_str() == cited_event),
        "evidence cites a real parent event id"
    );

    // (D) The child ancestry shows in the deterministic conversation graph: the
    // parent lists the child as a direct child, and the child names the parent in
    // its ancestry.
    let parent_graph = host.conversation_graph(&parent);
    assert!(
        parent_graph.children.iter().any(|c| c.session_id == child),
        "the parent graph lists the side chat as a direct child"
    );
    assert!(
        parent_graph.edges.iter().any(|e| e.parent == parent && e.child == child),
        "a parent->child edge is recorded"
    );
    let child_graph = host.conversation_graph(&child);
    assert!(
        child_graph.ancestry.iter().any(|a| a.session_id == parent),
        "the child graph names the parent in its ancestry"
    );

    // (E) Durable events + the projection AGREE: the published `side_chat_merged`
    // UiEvent carries the same summary + evidence count as the durable event.
    let ui = next_custom(&mut rx, "side_chat_merged").await;
    assert_eq!(ui.get("summary"), payload.get("summary"));
    assert_eq!(ui.get("result_kind").and_then(|v| v.as_str()), Some("investigation"));
    assert_eq!(
        ui.get("evidence").and_then(|v| v.as_array()).unwrap().len(),
        evidence.len()
    );
    assert_eq!(ui.get("parent").and_then(|v| v.as_str()), Some(parent.as_str()));
    assert_eq!(ui.get("side_chat").and_then(|v| v.as_str()), Some(child.as_str()));

    // (F) The side chat itself stays intact -- its private transcript is untouched
    // and the merge landed on the parent, not the child.
    let child_events = log.scan(Some(child.clone()), None, None).await.unwrap();
    assert!(
        child_events
            .iter()
            .any(|e| serde_json::to_string(&e.payload).unwrap().contains(CHILD_SECRET)),
        "the child keeps its own investigation transcript"
    );
    assert!(
        !child_events.iter().any(|e| e.kind == "session.merge_summary"),
        "the merge lands on the parent, not the child"
    );

    // (G) A parent-scoped transcript search SURFACES the cited summary (role
    // `side_chat`), proving the foldback is a searchable transcript item.
    let summary_hits = host
        .search_transcript(&TranscriptQuery::literal("selection").in_session(parent.clone()))
        .await
        .unwrap();
    assert!(
        summary_hits.iter().any(|h| h.role.as_deref() == Some("side_chat")),
        "the merged typed result is searchable on the parent as a side_chat item"
    );
}

// --- (2) Diagnostics projection feed -----------------------------------------

/// A fixture with planted deterministic issues: one marker macro (`todo!()`, an
/// Error) and one `.unwrap()` outside test code (a Warning).
fn dirty_fixture() -> &'static str {
    "pub fn parse_port(raw: &str) -> u16 {\n    raw.parse::<u16>().unwrap()\n}\n\npub fn not_done() {\n    todo!()\n}\n"
}

#[tokio::test]
async fn diagnostics_feed_publishes_real_nonzero_counts_for_planted_issues() {
    let host = test_host();
    let session = host.services.session();
    let mut rx = host.subscribe_ui();

    let receipt = host
        .run_static_analysis(session.clone(), vec![SourceFile::new("src/net.rs", dirty_fixture())])
        .await
        .unwrap();
    assert!(receipt.verdict().is_fail(), "planted issues fail the gate");

    // The diagnostics PROJECTION was published (the surface the StatusBar reads).
    let patch = next_projection(&mut rx, "diagnostics").await;
    let errors = patch["errors"].as_u64().unwrap();
    let warnings = patch["warnings"].as_u64().unwrap();
    assert!(errors >= 1, "the marker macro is an Error: {patch}");
    assert!(warnings >= 1, "the unwrap-outside-test is a Warning: {patch}");

    // by_file cites exactly the analyzed file, and its per-file counts sum to the
    // totals.
    let by_file = patch["by_file"].as_array().unwrap();
    assert_eq!(by_file.len(), 1, "one analyzed file: {patch}");
    assert_eq!(by_file[0]["file"].as_str(), Some("src/net.rs"));
    assert_eq!(
        by_file[0]["errors"].as_u64().unwrap() + by_file[0]["warnings"].as_u64().unwrap(),
        errors + warnings
    );

    // The projection ties back to the sealed receipt.
    assert_eq!(patch["last_verification_id"], json!(receipt.receipt.verification_id));
}

#[tokio::test]
async fn diagnostics_feed_is_zero_for_clean_source() {
    let host = test_host();
    let session = host.services.session();
    let mut rx = host.subscribe_ui();

    let clean = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
    let receipt = host
        .run_static_analysis(session.clone(), vec![SourceFile::new("src/math.rs", clean)])
        .await
        .unwrap();
    assert!(receipt.is_pass(), "clean source passes the gate");

    let patch = next_projection(&mut rx, "diagnostics").await;
    assert_eq!(patch["errors"], json!(0), "clean source has zero errors: {patch}");
    assert_eq!(patch["warnings"], json!(0), "clean source has zero warnings: {patch}");
    assert!(
        patch["by_file"].as_array().unwrap().is_empty(),
        "a clean source lists no problem files: {patch}"
    );
}

// --- (3) Transcript search over the /intent custom ---------------------------

#[tokio::test]
async fn transcript_search_over_intent_returns_literal_and_structured_hits() {
    let host = test_host();
    let session = host.services.session();
    let log = host.services.event_log.clone();
    let mut rx = host.subscribe_ui();

    // Seed a small transcript: an assistant item and a user item both carrying the
    // literal token, so a literal search matches both and a role filter narrows it.
    log.append(NewEvent::system(
        session.clone(),
        "agent.message",
        json!({ "role": "assistant", "text": "parse_port returns ZZLITERAL on success" }),
    ))
    .await
    .unwrap();
    log.append(NewEvent::system(
        session.clone(),
        "agent.message",
        json!({ "role": "user", "text": "where is ZZLITERAL used" }),
    ))
    .await
    .unwrap();

    // (a) LITERAL search over /intent, dialed with the FE's registered custom name
    // (`run_search`, wire.ts CUSTOM_NAMES): both items match.
    let ack = host
        .handle_intent(Intent::Custom {
            name: "run_search".to_string(),
            payload: json!({ "query": "ZZLITERAL" }),
        })
        .await
        .unwrap();
    assert!(ack.accepted, "the search intent is accepted");
    let results = next_custom(&mut rx, "search_results").await;
    assert_eq!(results["query"].as_str(), Some("ZZLITERAL"));
    assert_eq!(results["count"].as_u64(), Some(2), "literal search hits both items: {results}");

    // (b) STRUCTURED filter (role) over /intent: only the assistant item matches.
    let ack = host
        .handle_intent(Intent::Custom {
            name: "search_transcript".to_string(),
            payload: json!({ "query": "ZZLITERAL", "role": "assistant" }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let results = next_custom(&mut rx, "search_results").await;
    assert_eq!(
        results["count"].as_u64(),
        Some(1),
        "the role filter narrows to the assistant item: {results}"
    );
    assert_eq!(results["hits"][0]["role"].as_str(), Some("assistant"));

    // (c) A structured filter under a `scopes` object is honored too (kind filter).
    let ack = host
        .handle_intent(Intent::Custom {
            name: "search".to_string(),
            payload: json!({ "query": "ZZLITERAL", "scopes": { "kind": "agent.message" } }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let results = next_custom(&mut rx, "search_results").await;
    assert_eq!(
        results["count"].as_u64(),
        Some(2),
        "the kind filter under scopes still matches both agent.message items: {results}"
    );
}
