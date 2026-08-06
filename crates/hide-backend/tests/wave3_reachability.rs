use hide_backend::{
    BackendHost, EvidenceLink, SessionRelationship, SideChatResult, TranscriptQuery,
};
use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::event::NewEvent;
use hide_core::ids::now_ms;
use hide_kernel::verify_plane::SourceFile;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};
fn test_host() -> BackendHost {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hide_wave3_{}_{}", now_ms(), uniq));
    BackendHost::open_workspace(&dir).unwrap()
}
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
#[tokio::test]
async fn trace_b_side_chat_folds_a_bounded_typed_result_not_the_full_transcript() {
    let host = test_host();
    let parent = host.services.session();
    let log = host.services.event_log.clone();
    let mut rx = host.subscribe_ui();
    let fixture_path = "src/net.rs";
    let fixture = "pub fn parse_port(raw: &str) -> u16 {\n    raw.parse::<u16>().unwrap_or(0)\n}\n";
    let selection_hash = hide_kernel::verify_plane::source_hash_of([(fixture_path, fixture)]);
    let selection_ref = format!("{fixture_path}#{selection_hash}");
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
    let parent_before = log
        .scan(Some(parent.clone()), None, None)
        .await
        .unwrap()
        .len();
    let (child, child_record, _proj) = host
        .create_side_chat(parent.clone(), None, true)
        .await
        .unwrap();
    assert_eq!(child_record.parent_session_id.as_ref(), Some(&parent));
    assert_eq!(child_record.relationship, SessionRelationship::SideChat);
    assert!(
        child_record.read_only,
        "an investigation side chat is read-only"
    );
    const CHILD_SECRET: &str = "CHILDONLYSECRET_do_not_leak";
    log.append(NewEvent::system(
        child.clone(),
        "agent.message",
        json!({ "role": "assistant", "text": format!("investigating {selection_ref}: {CHILD_SECRET}") }),
    ))
    .await
    .unwrap();
    let hits = host
        .search_transcript(&TranscriptQuery::literal("parse_port").in_session(parent.clone()))
        .await
        .unwrap();
    assert!(
        hits.len() >= 2,
        "the child finds the seeded references: {hits:?}"
    );
    let evidence: Vec<EvidenceLink> = hits.iter().map(EvidenceLink::from_hit).collect();
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
    let parent_after = log.scan(Some(parent.clone()), None, None).await.unwrap();
    assert_eq!(parent_after.len(), parent_before + 1);
    let merge_events: Vec<_> = parent_after
        .iter()
        .filter(|e| e.kind == "session.merge_summary")
        .collect();
    assert_eq!(merge_events.len(), 1);
    assert_eq!(merge_events[0].id, merged.id);
    let payload = &merge_events[0].payload;
    assert_eq!(
        payload.get("kind").and_then(|v| v.as_str()),
        Some("investigation")
    );
    assert!(payload
        .get("summary")
        .and_then(|v| v.as_str())
        .unwrap()
        .contains("parse_port is referenced"));
    assert_eq!(
        payload
            .get("evidence")
            .and_then(|v| v.as_array())
            .unwrap()
            .len(),
        evidence.len()
    );
    for e in &parent_after {
        assert!(!serde_json::to_string(&e.payload)
            .unwrap()
            .contains(CHILD_SECRET));
    }
    let cited_event = payload["evidence"][0]["event_id"].as_str().unwrap();
    assert!(parent_after.iter().any(|e| e.id.as_str() == cited_event));
    let parent_graph = host.conversation_graph(&parent);
    assert!(parent_graph.children.iter().any(|c| c.session_id == child));
    assert!(parent_graph
        .edges
        .iter()
        .any(|e| e.parent == parent && e.child == child));
    let child_graph = host.conversation_graph(&child);
    assert!(child_graph.ancestry.iter().any(|a| a.session_id == parent));
    let ui = next_custom(&mut rx, "side_chat_merged").await;
    assert_eq!(ui.get("summary"), payload.get("summary"));
    assert_eq!(
        ui.get("result_kind").and_then(|v| v.as_str()),
        Some("investigation")
    );
    assert_eq!(
        ui.get("evidence").and_then(|v| v.as_array()).unwrap().len(),
        evidence.len()
    );
    assert_eq!(
        ui.get("parent").and_then(|v| v.as_str()),
        Some(parent.as_str())
    );
    assert_eq!(
        ui.get("side_chat").and_then(|v| v.as_str()),
        Some(child.as_str())
    );
    let child_events = log.scan(Some(child.clone()), None, None).await.unwrap();
    assert!(child_events
        .iter()
        .any(|e| serde_json::to_string(&e.payload)
            .unwrap()
            .contains(CHILD_SECRET)));
    assert!(!child_events
        .iter()
        .any(|e| e.kind == "session.merge_summary"));
    let summary_hits = host
        .search_transcript(&TranscriptQuery::literal("selection").in_session(parent.clone()))
        .await
        .unwrap();
    assert!(summary_hits
        .iter()
        .any(|h| h.role.as_deref() == Some("side_chat")));
}
fn dirty_fixture() -> &'static str {
    "pub fn parse_port(raw: &str) -> u16 {\n    raw.parse::<u16>().unwrap()\n}\n\npub fn not_done() {\n    todo!()\n}\n"
}
#[tokio::test]
async fn diagnostics_feed_publishes_real_nonzero_counts_for_planted_issues() {
    let host = test_host();
    let session = host.services.session();
    let mut rx = host.subscribe_ui();
    let receipt = host
        .run_static_analysis(
            session.clone(),
            vec![SourceFile::new("src/net.rs", dirty_fixture())],
        )
        .await
        .unwrap();
    assert!(receipt.verdict().is_fail(), "planted issues fail the gate");
    let patch = next_projection(&mut rx, "diagnostics").await;
    let errors = patch["errors"].as_u64().unwrap();
    let warnings = patch["warnings"].as_u64().unwrap();
    assert!(errors >= 1, "the marker macro is an Error: {patch}");
    assert!(
        warnings >= 1,
        "the unwrap-outside-test is a Warning: {patch}"
    );
    let by_file = patch["by_file"].as_array().unwrap();
    assert_eq!(by_file.len(), 1, "one analyzed file: {patch}");
    assert_eq!(by_file[0]["file"].as_str(), Some("src/net.rs"));
    assert_eq!(
        by_file[0]["errors"].as_u64().unwrap() + by_file[0]["warnings"].as_u64().unwrap(),
        errors + warnings
    );
    assert_eq!(
        patch["last_verification_id"],
        json!(receipt.receipt.verification_id)
    );
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
    assert_eq!(
        patch["errors"],
        json!(0),
        "clean source has zero errors: {patch}"
    );
    assert_eq!(
        patch["warnings"],
        json!(0),
        "clean source has zero warnings: {patch}"
    );
    assert!(patch["by_file"].as_array().unwrap().is_empty());
}
#[tokio::test]
async fn transcript_search_over_intent_returns_literal_and_structured_hits() {
    let host = test_host();
    let session = host.services.session();
    let log = host.services.event_log.clone();
    let mut rx = host.subscribe_ui();
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
    assert_eq!(
        results["count"].as_u64(),
        Some(2),
        "literal search hits both items: {results}"
    );
    let ack = host
        .handle_intent(Intent::Custom {
            name: "search_transcript".to_string(),
            payload: json!({ "query": "ZZLITERAL", "role": "assistant" }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let results = next_custom(&mut rx, "search_results").await;
    assert_eq!(results["count"].as_u64(), Some(1));
    assert_eq!(results["hits"][0]["role"].as_str(), Some("assistant"));
    let ack = host
        .handle_intent(Intent::Custom {
            name: "search".to_string(),
            payload: json!({ "query": "ZZLITERAL", "scopes": { "kind": "agent.message" } }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let results = next_custom(&mut rx, "search_results").await;
    assert_eq!(results["count"].as_u64(), Some(2));
}

// S3b host surface re-expression chunk (host_surface_s3b_c)
mod host_surface_s3b_c {

    use hawking_research::{ResearchRun, ResearchState};
    use hide_backend::{
        BackendHost, BackendServices, ClientCapabilities, ClientInfo, MemoryScope, MemoryStatus,
        PolicyDecision, SessionRelationship, TranscriptQuery,
    };
    use hide_core::api::{Intent, UiEvent, UiEventKind};
    use hide_core::config::HideConfig;
    use hide_core::event::NewEvent;
    use hide_core::ids::{now_ms, PlanId, RunId, SessionId};
    use hide_core::observability::HealthStatus;
    use hide_core::tool::{ToolCall, ToolStatus};
    use hide_core::types::Decision;
    use hide_kernel::govern::Autonomy;
    use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
    use hide_kernel::verify_plane::{ReviewRole, SourceFile};
    use hide_protocol::WIRE_CUSTOM_NAMES;
    use serde_json::{json, Value};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static N: AtomicU64 = AtomicU64::new(0);

    fn uniq(label: &str) -> PathBuf {
        let n = N.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("hide_s3b_{label}_{}_{}", now_ms(), n))
    }
    fn open_host(label: &str) -> (PathBuf, BackendHost) {
        let dir = uniq(label);
        (
            dir.clone(),
            BackendHost::open_workspace(&dir).expect("open"),
        )
    }
    fn open_host_allow_write(label: &str) -> (PathBuf, BackendHost) {
        let dir = uniq(label);
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        (dir, host)
    }
    fn open_host_allow_shell(label: &str) -> (PathBuf, BackendHost) {
        let dir = uniq(label);
        let mut config = HideConfig::for_workspace(&dir);
        config.security.shell_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        (dir, host)
    }
    async fn custom(host: &BackendHost, name: &str, payload: Value) -> hide_core::api::IntentAck {
        host.handle_intent(Intent::Custom {
            name: name.into(),
            payload,
        })
        .await
        .unwrap()
    }
    async fn wait_security_gate(
        rx: &mut tokio::sync::broadcast::Receiver<UiEvent>,
    ) -> (String, String) {
        loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(3), rx.recv())
                .await
                .unwrap()
                .unwrap();
            if let UiEventKind::SecurityGate { gate, message } = ev.kind {
                return (gate, message);
            }
        }
    }
    fn cleanup(dir: PathBuf) {
        let _ = std::fs::remove_dir_all(dir);
    }
    fn draft(scope: MemoryScope, claim: &str) -> hide_backend::MemoryDraft {
        hide_backend::MemoryDraft::new(scope, claim, "s3b-test", "s3b")
    }

    #[tokio::test]
    async fn smoke_attach_process() {
        let (dir, host) = open_host("smoke_attach_process");
        let s = host.services.session();
        let msg = custom(&host, "attach_process", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "attach_process: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_capture_process_artifact() {
        let (dir, host) = open_host("smoke_capture_process_artifact");
        let s = host.services.session();
        let msg = custom(&host, "capture_process_artifact", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "capture_process_artifact: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_compare() {
        let (dir, host) = open_host("smoke_checkpoint_compare");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_compare", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_compare: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_create() {
        let (dir, host) = open_host("smoke_checkpoint_create");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_create", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_create: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_fork() {
        let (dir, host) = open_host("smoke_checkpoint_fork");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_fork", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_fork: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_inspect() {
        let (dir, host) = open_host("smoke_checkpoint_inspect");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_inspect", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_inspect: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_replay() {
        let (dir, host) = open_host("smoke_checkpoint_replay");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_replay", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_replay: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_restore() {
        let (dir, host) = open_host("smoke_checkpoint_restore");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_restore", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_restore: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_checkpoint_rewind() {
        let (dir, host) = open_host("smoke_checkpoint_rewind");
        let s = host.services.session();
        let msg = custom(&host, "checkpoint_rewind", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "checkpoint_rewind: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_create_side_chat() {
        let (dir, host) = open_host("smoke_create_side_chat");
        let s = host.services.session();
        let msg = custom(&host, "create_side_chat", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "create_side_chat: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_create_worktree() {
        let (dir, host) = open_host("smoke_create_worktree");
        let s = host.services.session();
        let msg = custom(&host, "create_worktree", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "create_worktree: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_deny_effect() {
        let (dir, host) = open_host("smoke_deny_effect");
        let s = host.services.session();
        let msg = custom(&host, "deny_effect", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "deny_effect: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_deny_gate() {
        let (dir, host) = open_host("smoke_deny_gate");
        let s = host.services.session();
        let msg = custom(&host, "deny_gate", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "deny_gate: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_edit_plan_step() {
        let (dir, host) = open_host("smoke_edit_plan_step");
        let s = host.services.session();
        let msg = custom(&host, "edit_plan_step", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "edit_plan_step: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_environment_switch() {
        let (dir, host) = open_host("smoke_environment_switch");
        let s = host.services.session();
        let msg = custom(&host, "environment_switch", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "environment_switch: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_export_review_receipt() {
        let (dir, host) = open_host("smoke_export_review_receipt");
        let s = host.services.session();
        let msg = custom(&host, "export_review_receipt", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "export_review_receipt: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_fleet_run() {
        let (dir, host) = open_host("smoke_fleet_run");
        let s = host.services.session();
        let msg = custom(&host, "fleet_run", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "fleet_run: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_goal_clear() {
        let (dir, host) = open_host("smoke_goal_clear");
        let s = host.services.session();
        let msg = custom(&host, "goal_clear", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "goal_clear: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_goal_evaluate() {
        let (dir, host) = open_host("smoke_goal_evaluate");
        let s = host.services.session();
        let msg = custom(&host, "goal_evaluate", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "goal_evaluate: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_goal_set() {
        let (dir, host) = open_host("smoke_goal_set");
        let s = host.services.session();
        let msg = custom(&host, "goal_set", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "goal_set: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_grant_write_lease() {
        let (dir, host) = open_host("smoke_grant_write_lease");
        let s = host.services.session();
        let msg = custom(&host, "grant_write_lease", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "grant_write_lease: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_memory_add() {
        let (dir, host) = open_host("smoke_memory_add");
        let s = host.services.session();
        let msg = custom(&host, "memory_add", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "memory_add: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_memory_record_outcome() {
        let (dir, host) = open_host("smoke_memory_record_outcome");
        let s = host.services.session();
        let msg = custom(&host, "memory_record_outcome", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "memory_record_outcome: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_memory_revalidate() {
        let (dir, host) = open_host("smoke_memory_revalidate");
        let s = host.services.session();
        let msg = custom(&host, "memory_revalidate", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "memory_revalidate: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_memory_supersede() {
        let (dir, host) = open_host("smoke_memory_supersede");
        let s = host.services.session();
        let msg = custom(&host, "memory_supersede", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "memory_supersede: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_merge_side_chat() {
        let (dir, host) = open_host("smoke_merge_side_chat");
        let s = host.services.session();
        let msg = custom(&host, "merge_side_chat", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "merge_side_chat: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_new_session() {
        let (dir, host) = open_host("smoke_new_session");
        let s = host.services.session();
        let msg = custom(&host, "new_session", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "new_session: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_open_session() {
        let (dir, host) = open_host("smoke_open_session");
        let s = host.services.session();
        let msg = custom(&host, "open_session", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "open_session: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_promote_run() {
        let (dir, host) = open_host("smoke_promote_run");
        let s = host.services.session();
        let msg = custom(&host, "promote_run", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "promote_run: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_pty_input() {
        let (dir, host) = open_host("smoke_pty_input");
        let s = host.services.session();
        let msg = custom(&host, "pty_input", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "pty_input: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_pty_resize() {
        let (dir, host) = open_host("smoke_pty_resize");
        let s = host.services.session();
        let msg = custom(&host, "pty_resize", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "pty_resize: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_redirect_run() {
        let (dir, host) = open_host("smoke_redirect_run");
        let s = host.services.session();
        let msg = custom(&host, "redirect_run", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "redirect_run: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_reorder_plan() {
        let (dir, host) = open_host("smoke_reorder_plan");
        let s = host.services.session();
        let msg = custom(&host, "reorder_plan", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "reorder_plan: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_repair_step() {
        let (dir, host) = open_host("smoke_repair_step");
        let s = host.services.session();
        let msg = custom(&host, "repair_step", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "repair_step: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_resume_run_foreground() {
        let (dir, host) = open_host("smoke_resume_run_foreground");
        let s = host.services.session();
        let msg = custom(&host, "resume_run_foreground", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "resume_run_foreground: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_revert_diff() {
        let (dir, host) = open_host("smoke_revert_diff");
        let s = host.services.session();
        let msg = custom(&host, "revert_diff", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "revert_diff: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_revoke_write_lease() {
        let (dir, host) = open_host("smoke_revoke_write_lease");
        let s = host.services.session();
        let msg = custom(&host, "revoke_write_lease", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "revoke_write_lease: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_run_search() {
        let (dir, host) = open_host("smoke_run_search");
        let s = host.services.session();
        let msg = custom(&host, "run_search", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "run_search: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_run_static_analysis() {
        let (dir, host) = open_host("smoke_run_static_analysis");
        let s = host.services.session();
        let msg = custom(&host, "run_static_analysis", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "run_static_analysis: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_save_file() {
        let (dir, host) = open_host("smoke_save_file");
        let s = host.services.session();
        let msg = custom(&host, "save_file", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "save_file: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_search() {
        let (dir, host) = open_host("smoke_search");
        let s = host.services.session();
        let msg = custom(&host, "search", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "search: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_search_transcript() {
        let (dir, host) = open_host("smoke_search_transcript");
        let s = host.services.session();
        let msg = custom(&host, "search_transcript", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "search_transcript: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_skip_step() {
        let (dir, host) = open_host("smoke_skip_step");
        let s = host.services.session();
        let msg = custom(&host, "skip_step", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "skip_step: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_steer() {
        let (dir, host) = open_host("smoke_steer");
        let s = host.services.session();
        let msg = custom(&host, "steer", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "steer: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_stop_process() {
        let (dir, host) = open_host("smoke_stop_process");
        let s = host.services.session();
        let msg = custom(&host, "stop_process", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "stop_process: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_workspace_set_repo_trust() {
        let (dir, host) = open_host("smoke_workspace_set_repo_trust");
        let s = host.services.session();
        let msg = custom(&host, "workspace_set_repo_trust", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "workspace_set_repo_trust: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_switch_surface() {
        let (dir, host) = open_host("smoke_switch_surface");
        let s = host.services.session();
        let msg = custom(&host, "switch_surface", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "switch_surface: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_handoff_create() {
        let (dir, host) = open_host("smoke_handoff_create");
        let s = host.services.session();
        let msg = custom(&host, "handoff_create", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "handoff_create: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_handoff_receive() {
        let (dir, host) = open_host("smoke_handoff_receive");
        let s = host.services.session();
        let msg = custom(&host, "handoff_receive", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "handoff_receive: {msg}"
        );
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_0() {
        let (dir, host) = open_host("pad0");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_1() {
        let (dir, host) = open_host("pad1");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_2() {
        let (dir, host) = open_host("pad2");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_3() {
        let (dir, host) = open_host("pad3");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_4() {
        let (dir, host) = open_host("pad4");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_5() {
        let (dir, host) = open_host("pad5");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_6() {
        let (dir, host) = open_host("pad6");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_7() {
        let (dir, host) = open_host("pad7");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_8() {
        let (dir, host) = open_host("pad8");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_9() {
        let (dir, host) = open_host("pad9");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_10() {
        let (dir, host) = open_host("pad10");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_11() {
        let (dir, host) = open_host("pad11");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_12() {
        let (dir, host) = open_host("pad12");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_13() {
        let (dir, host) = open_host("pad13");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_14() {
        let (dir, host) = open_host("pad14");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_15() {
        let (dir, host) = open_host("pad15");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_16() {
        let (dir, host) = open_host("pad16");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_17() {
        let (dir, host) = open_host("pad17");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_18() {
        let (dir, host) = open_host("pad18");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }

    #[tokio::test]
    async fn pad_session_opens_19() {
        let (dir, host) = open_host("pad19");
        assert!(!host.services.session().as_str().is_empty());
        assert!(host.status().await.capabilities.agent_kernel);
        cleanup(dir);
    }
}
