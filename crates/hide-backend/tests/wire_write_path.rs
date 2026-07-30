use hide_backend::host::HunkStatus;
use hide_backend::{BackendHost, BackendServices};
use hide_core::api::{Intent, UiEventKind};
use hide_core::config::HideConfig;
use hide_core::ids::{now_ms, RunId, SessionId};
use hide_core::types::Decision;
use serde_json::json;
fn write_host(tag: &str) -> (BackendHost, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!("hide_wire_{tag}_{}", now_ms()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut config = HideConfig::for_workspace(&dir);
    config.security.workspace_write_default = Decision::Allow;
    let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
    (host, dir)
}
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
    let proposal = host
        .diff_get(&diff_id)
        .expect("a client save registers a DiffProposal");
    assert_eq!(proposal.hunks.len(), 2, "two saves = two addressable hunks");
    assert_eq!(proposal.hunks[0].before, "fn pool() {}\n");
    assert_eq!(proposal.hunks[0].after, "fn pool() { /* capped */ }\n");
    assert!(proposal
        .hunks
        .iter()
        .all(|h| h.status == HunkStatus::Pending));
    let (mut diff, mut chip) = (false, false);
    while let Ok(ev) = rx.try_recv() {
        if let UiEventKind::ProjectionPatch { projection, .. } = &ev.kind {
            diff |= projection == "diff";
            chip |= projection == "diff_chip";
        }
    }
    assert!(
        diff && chip,
        "both diff projections publish for a client save"
    );
    let kinds = kinds(&host, &session).await;
    assert!(kinds.iter().any(|k| k == "tool.call"), "{kinds:?}");
    assert!(kinds.iter().any(|k| k == "tool.result"), "{kinds:?}");
    assert_eq!(kinds.iter().filter(|k| *k == "diff.proposed").count(), 2);
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
    assert!(
        ack.accepted && !ack.held,
        "a single-hunk reject is not gated: {ack:?}"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("retry.rs")).unwrap(),
        "fn retry() {}\n"
    );
    assert_eq!(
        std::fs::read_to_string(dir.join("pool.rs")).unwrap(),
        "fn pool() { /* capped */ }\n"
    );
    let _ = std::fs::remove_dir_all(dir);
}
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
    assert!(
        !ack.accepted && !ack.held,
        "a conflict is refused, not held: {ack:?}"
    );
    assert!(ack.message.unwrap_or_default().contains("refused"));
    assert_eq!(std::fs::read_to_string(dir.join("a.rs")).unwrap(), "two\n");
    let _ = std::fs::remove_dir_all(dir);
}
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
    assert_eq!(checkpoint.coverage.repo_state.count, 1);
    save(&host, &session, "pool.rs", "after the checkpoint\n").await;
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "after the checkpoint\n"
    );
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
        .map(|g| {
            g.split_whitespace()
                .next()
                .unwrap_or(g)
                .trim_end_matches(')')
                .to_string()
        })
        .expect("the hold names its gate");
    host.handle_intent(Intent::Custom {
        name: "approve_gate".to_string(),
        payload: json!({ "gate": gate }),
    })
    .await
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(&file).unwrap(),
        "at the checkpoint\n"
    );
    let _ = std::fs::remove_dir_all(dir);
}
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
#[tokio::test]
async fn the_process_controls_are_reachable_over_the_wire() {
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
        if host
            .process_state(&id)
            .map(|s| s.line_count >= 2)
            .unwrap_or(false)
        {
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
#[test]
fn the_new_wire_names_are_on_the_contract() {
    for name in [
        "attach_process",
        "stop_process",
        "capture_process_artifact",
        "export_review_receipt",
    ] {
        assert!(hide_protocol::command::WIRE_CUSTOM_NAMES.contains(&name));
    }
    let session = SessionId::from("ses_test");
    assert_eq!(
        BackendHost::editor_run(&session),
        RunId::from("editor-ses_test")
    );
}

// S3b host surface re-expression chunk (host_surface_s3b_a)
mod host_surface_s3b_a {

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
    async fn safe_shell_not_held() {
        let (dir, host) = open_host_allow_shell("s");
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["printf".into(), "ok".into()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted && !ack.held);
        cleanup(dir);
    }
    #[tokio::test]
    async fn dangerous_sudo_held() {
        let (dir, host) = open_host("dsudo");
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["sudo".into(), "rm".into(), "f".into()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted && ack.held);
        cleanup(dir);
    }
    #[tokio::test]
    async fn dangerous_rmrf_held() {
        let (dir, host) = open_host("drmrf");
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["rm".into(), "-rf".into(), "/".into()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted && ack.held);
        cleanup(dir);
    }
    #[tokio::test]
    async fn dangerous_home_held() {
        let (dir, host) = open_host("dhome");
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["rm".into(), "-rf".into(), "~".into()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted && ack.held);
        cleanup(dir);
    }
    #[tokio::test]
    async fn dangerous_dd_held() {
        let (dir, host) = open_host("ddd");
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["dd".into(), "if=x".into(), "of=/dev/disk0".into()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted && ack.held);
        cleanup(dir);
    }
    #[tokio::test]
    async fn dangerous_mkfs_held() {
        let (dir, host) = open_host("dmkfs");
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["mkfs.hidetest".into()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted && ack.held);
        cleanup(dir);
    }
    #[tokio::test]
    async fn approve_releases_gate() {
        let (dir, host) = open_host("ap");
        let mut rx = host.subscribe_ui();
        host.handle_intent(Intent::RunCommand {
            argv: vec!["mkfs.hidetest".into()],
            cwd: None,
        })
        .await
        .unwrap();
        let (gate, msg) = wait_security_gate(&mut rx).await;
        assert!(msg.contains("mkfs.hidetest"));
        assert!(
            custom(&host, "approve_gate", json!({"gate":gate}))
                .await
                .accepted
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn deny_drops_gate() {
        let (dir, host) = open_host("dn");
        let mut rx = host.subscribe_ui();
        host.handle_intent(Intent::RunCommand {
            argv: vec!["mkfs.hidetest".into()],
            cwd: None,
        })
        .await
        .unwrap();
        let (gate, _) = wait_security_gate(&mut rx).await;
        assert!(
            custom(&host, "deny_gate", json!({"gate":gate}))
                .await
                .accepted
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn unknown_gate_refused() {
        let (dir, host) = open_host("ug");
        for n in ["approve_gate", "deny_gate"] {
            let ack = custom(&host, n, json!({"gate":"command:999"})).await;
            assert!(!ack.accepted);
        }
        cleanup(dir);
    }
    #[tokio::test]
    async fn wire_names_handled() {
        let (dir, host) = open_host("w");
        for name in WIRE_CUSTOM_NAMES {
            let msg = custom(&host, name, json!({}))
                .await
                .message
                .unwrap_or_default();
            assert!(!msg.contains("has no host handler"), "{name}: {msg}");
        }
        cleanup(dir);
    }
    #[tokio::test]
    async fn unhandled_custom_refused() {
        let (dir, host) = open_host("u");
        let ack = custom(&host, "create_pr", json!({})).await;
        assert!(!ack.accepted && ack.event_seq.is_some());
        cleanup(dir);
    }
    #[tokio::test]
    async fn unhandled_unknown_recorded() {
        let (dir, host) = open_host("u2");
        let ack = custom(&host, "totally_unknown_xyz", json!({})).await;
        assert!(!ack.accepted && ack.event_seq.is_some());
        cleanup(dir);
    }
    #[tokio::test]
    async fn session_opens() {
        let (dir, host) = open_host("s");
        assert!(!host.services.session().as_str().is_empty());
        cleanup(dir);
    }
    #[tokio::test]
    async fn runtime_down() {
        let (dir, host) = open_host("r");
        let st = host
            .call_connector("runtime", "state", json!({}))
            .await
            .unwrap();
        assert_eq!(st["state"], "down");
        assert!(host.runtime_state().is_none());
        cleanup(dir);
    }
    #[tokio::test]
    async fn status_surface() {
        let (dir, host) = open_host("st");
        let s = host.status().await;
        assert!(s.capabilities.agent_kernel);
        assert!(s.tools.iter().any(|t| t.name == "fs.write"));
        assert!(s.connectors.iter().any(|c| c.id == "research"));
        assert!(s.model_roles.iter().any(|r| r.name == "hawking-hero-coder"));
        cleanup(dir);
    }
    #[tokio::test]
    async fn caps_agent_kernel() {
        let (d, h) = open_host("c1");
        assert!(h.status().await.capabilities.agent_kernel);
        cleanup(d);
    }
    #[tokio::test]
    async fn caps_fleet() {
        let (d, h) = open_host("c2");
        assert!(h.status().await.capabilities.fleet);
        cleanup(d);
    }
    #[tokio::test]
    async fn caps_model_orch() {
        let (d, h) = open_host("c3");
        assert!(h.status().await.capabilities.model_orchestration);
        cleanup(d);
    }
    #[tokio::test]
    async fn caps_remote_false() {
        let (d, h) = open_host("c4");
        assert!(!h.status().await.capabilities.remote_protocol);
        cleanup(d);
    }
    #[tokio::test]
    async fn health_ok() {
        let (d, h) = open_host("h");
        let health = h.health().await;
        assert_eq!(health.status, HealthStatus::Ok);
        assert!(health.checks.iter().any(|c| c.name == "tools"));
        cleanup(d);
    }
    #[tokio::test]
    async fn tool_dispatch_events() {
        let (dir, host) = open_host_allow_write("td");
        let s = host.services.session();
        let f = dir.join("t.txt");
        let r = host
            .dispatch_tool(
                s.clone(),
                None,
                ToolCall::new(
                    "fs.write",
                    json!({"path":f.to_string_lossy(),"content":"host write","create_dirs":true}),
                ),
            )
            .await
            .unwrap();
        assert_eq!(r.status, ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&f).unwrap(), "host write");
        let ev = host
            .services
            .event_log
            .scan(Some(s.clone()), None, None)
            .await
            .unwrap();
        assert!(
            ev.iter().any(|e| e.kind == "tool.call") && ev.iter().any(|e| e.kind == "tool.result")
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn policy_read_allow() {
        let (d, h) = open_host("pr");
        let x = h
            .evaluate_tool_policy(
                &h.services.session(),
                "fs.read",
                &json!({"path":d.join("a").to_string_lossy()}),
            )
            .await
            .unwrap();
        assert_eq!(x, PolicyDecision::Allow);
        cleanup(d);
    }
    #[tokio::test]
    async fn policy_shell_sandbox() {
        let (d, h) = open_host("ps");
        let x = h
            .evaluate_tool_policy(&h.services.session(), "shell.run", &json!({"argv":["ls"]}))
            .await
            .unwrap();
        assert_eq!(x, PolicyDecision::RequireSandbox);
        cleanup(d);
    }
    #[tokio::test]
    async fn policy_git_ask() {
        let (d, h) = open_host("pg");
        let x = h
            .evaluate_tool_policy(&h.services.session(), "git.commit", &json!({"message":"w"}))
            .await
            .unwrap();
        assert!(matches!(
            x,
            PolicyDecision::Ask | PolicyDecision::RequireReviewer
        ));
        cleanup(d);
    }
    #[tokio::test]
    async fn policy_edit_ask() {
        let (d, h) = open_host("pe");
        let x = h
            .evaluate_tool_policy(
                &h.services.session(),
                "edit.write_file",
                &json!({"path":"x","content":"y"}),
            )
            .await
            .unwrap();
        assert_eq!(x, PolicyDecision::Ask);
        cleanup(d);
    }
    #[tokio::test]
    async fn policy_ledger_four() {
        let (dir, host) = open_host("pl");
        let s = host.services.session();
        let _ = host
            .evaluate_tool_policy(
                &s,
                "fs.read",
                &json!({"path":dir.join("a").to_string_lossy()}),
            )
            .await
            .unwrap();
        let _ = host
            .evaluate_tool_policy(&s, "shell.run", &json!({"argv":["ls"]}))
            .await
            .unwrap();
        let _ = host
            .evaluate_tool_policy(&s, "git.commit", &json!({"message":"w"}))
            .await
            .unwrap();
        let _ = host
            .evaluate_tool_policy(&s, "edit.write_file", &json!({"path":"b","content":"x"}))
            .await
            .unwrap();
        assert_eq!(host.policy_decisions(&s).await.unwrap().len(), 4);
        cleanup(dir);
    }
    #[tokio::test]
    async fn write_policy_allow() {
        let (dir, host) = open_host_allow_write("wp");
        let s = host.services.session();
        let d = host
            .evaluate_tool_policy(
                &s,
                "edit.write_file",
                &json!({"path":dir.join("c").to_string_lossy(),"content":"x"}),
            )
            .await
            .unwrap();
        assert_eq!(d, PolicyDecision::Allow);
        cleanup(dir);
    }
    #[tokio::test]
    async fn run_command_api() {
        let (dir, host) = open_host_allow_shell("rc");
        assert!(
            host.handle_intent(Intent::RunCommand {
                argv: vec!["printf".into(), "i".into()],
                cwd: None
            })
            .await
            .unwrap()
            .accepted
        );
        let r = host
            .run_command(
                host.services.session(),
                vec!["printf".into(), "api".into()],
                None,
            )
            .await
            .unwrap();
        assert_eq!(r.status, ToolStatus::Ok);
        cleanup(dir);
    }
    #[tokio::test]
    async fn connector_research() {
        let (dir, host) = open_host("cr");
        let mut run = ResearchRun::new("host connector");
        run.state = ResearchState::Complete;
        host.call_connector("research", "runs.append", json!({"run":run}))
            .await
            .unwrap();
        let listed = host
            .call_connector("research", "runs.list", json!({"limit":1}))
            .await
            .unwrap();
        assert_eq!(listed["runs"].as_array().unwrap().len(), 1);
        cleanup(dir);
    }
    #[tokio::test]
    async fn submit_turn_accepted() {
        let (d, h) = open_host("st");
        assert!(
            h.handle_intent(Intent::SubmitTurn {
                session_id: h.services.session(),
                text: "hello".into(),
                attachments: vec![]
            })
            .await
            .unwrap()
            .accepted
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn empty_submit() {
        let (d, h) = open_host("es");
        let _ = h
            .handle_intent(Intent::SubmitTurn {
                session_id: SessionId::new(),
                text: "   ".into(),
                attachments: vec![],
            })
            .await
            .unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn submit_offline_model_free() {
        let (d, h) = open_host("so");
        assert!(
            h.handle_intent(Intent::SubmitTurn {
                session_id: h.services.session(),
                text: "hi".into(),
                attachments: vec![]
            })
            .await
            .unwrap()
            .accepted
        );
        assert!(h.runtime_state().is_none());
        cleanup(d);
    }
    #[tokio::test]
    async fn new_session_accepted() {
        let (d, h) = open_host("ns");
        assert!(custom(&h, "new_session", json!({})).await.accepted);
        cleanup(d);
    }
    #[tokio::test]
    async fn switch_surface_handled() {
        let (d, h) = open_host("ss");
        let msg = custom(
            &h,
            "switch_surface",
            json!({"session_id":h.services.session().as_str(),"surface":"you"}),
        )
        .await
        .message
        .unwrap_or_default();
        assert!(!msg.contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn cpr_intents() {
        let (d, h) = open_host("cpr");
        let r = RunId::new();
        for i in [
            Intent::CancelRun { run_id: r.clone() },
            Intent::PauseRun { run_id: r.clone() },
            Intent::ResumeRun { run_id: r.clone() },
        ] {
            let _ = h.handle_intent(i).await.unwrap();
        }
        cleanup(d);
    }
    #[tokio::test]
    async fn diff_intents() {
        let (d, h) = open_host("di");
        let r = RunId::new();
        let _ = h
            .handle_intent(Intent::AcceptDiff {
                run_id: r.clone(),
                diff_id: "d".into(),
                hunk_id: None,
            })
            .await
            .unwrap();
        let _ = h
            .handle_intent(Intent::RejectDiff {
                run_id: r,
                diff_id: "d".into(),
                hunk_id: None,
            })
            .await
            .unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn open_file_intent() {
        let (dir, h) = open_host("of");
        let p = dir.join("a.rs");
        std::fs::write(&p, "fn a(){}").unwrap();
        let _ = h
            .handle_intent(Intent::OpenFile {
                path: p.to_string_lossy().into(),
                line: None,
            })
            .await
            .unwrap();
        cleanup(dir);
    }
    #[tokio::test]
    async fn fleet_handled() {
        let (d, h) = open_host("fl");
        let msg = custom(
            &h,
            "fleet_run",
            json!({"session_id":h.services.session().as_str(),"objective":"o"}),
        )
        .await
        .message
        .unwrap_or_default();
        assert!(!msg.contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn static_analysis_dirty() {
        let (dir, h) = open_host("sad");
        let dirty = "fn f(){let x:Option<i32>=None;let _=x.unwrap();}";
        let p = dir.join("d.rs");
        std::fs::write(&p, dirty).unwrap();
        let _ = h
            .run_static_analysis(
                h.services.session(),
                vec![SourceFile::new(p.to_string_lossy(), dirty)],
            )
            .await
            .unwrap();
        cleanup(dir);
    }
    #[tokio::test]
    async fn static_analysis_clean() {
        let (dir, h) = open_host("sac");
        let clean = "pub fn add(a:i32,b:i32)->i32{a+b}";
        let p = dir.join("c.rs");
        std::fs::write(&p, clean).unwrap();
        let _ = h
            .run_static_analysis(
                h.services.session(),
                vec![SourceFile::new(p.to_string_lossy(), clean)],
            )
            .await
            .unwrap();
        cleanup(dir);
    }
    #[tokio::test]
    async fn static_analysis_intent() {
        let (dir, h) = open_host("sai");
        let p = dir.join("x.rs");
        std::fs::write(&p, "fn f(){}").unwrap();
        let msg = custom(
            &h,
            "run_static_analysis",
            json!({"session_id":h.services.session().as_str(),"paths":[p.to_string_lossy()]}),
        )
        .await
        .message
        .unwrap_or_default();
        assert!(!msg.contains("has no host handler"));
        cleanup(dir);
    }
    #[tokio::test]
    async fn review_profiles() {
        let (d, h) = open_host("rp");
        let profiles = h.review_role_profiles();
        assert!(!profiles.is_empty());
        let c = h.review_role_profile(ReviewRole::Correctness);
        assert!(!c.focus.is_empty());
        cleanup(d);
    }
    #[test]
    fn probabilistic_no_override() {
        assert!(!hide_kernel::verify_plane::probabilistic_can_override_deterministic());
    }
    #[tokio::test]
    async fn memory_scopes() {
        let (d, h) = open_host("ms");
        let sess = MemoryScope::Session(h.services.session().as_str().into());
        let repo = MemoryScope::Repo("r".into());
        let user = MemoryScope::User("u".into());
        h.memory_add(draft(sess.clone(), "s")).unwrap();
        h.memory_add(draft(repo.clone(), "r")).unwrap();
        h.memory_add(draft(user.clone(), "u")).unwrap();
        assert_eq!(h.memory_list(&sess).len(), 1);
        assert_eq!(h.memory_list(&repo).len(), 1);
        assert_eq!(h.memory_list(&user).len(), 1);
        cleanup(d);
    }
    #[tokio::test]
    async fn memory_supersede() {
        let (d, h) = open_host("msu");
        let scope = MemoryScope::Repo("s".into());
        let old = h.memory_add(draft(scope.clone(), "old")).unwrap();
        let (oa, new) = h
            .memory_supersede(&old.memory_id, draft(scope.clone(), "new"))
            .unwrap();
        assert_eq!(oa.status, MemoryStatus::Superseded);
        assert_eq!(new.status, MemoryStatus::Active);
        assert_eq!(h.memory_list(&scope).len(), 2);
        assert_eq!(h.memory_context(&scope).len(), 1);
        cleanup(d);
    }
    #[tokio::test]
    async fn memory_outcome() {
        let (d, h) = open_host("mo");
        let r = h
            .memory_add(draft(MemoryScope::Repo("o".into()), "c"))
            .unwrap();
        let a = h.memory_record_outcome(&r.memory_id, true).unwrap();
        assert!(a.outcome_score >= r.outcome_score);
        assert!(h.memory_record_outcome("missing", true).is_err());
        cleanup(d);
    }
    #[tokio::test]
    async fn memory_get_missing() {
        let (d, h) = open_host("mm");
        assert!(h.memory_get("nope").is_none());
        cleanup(d);
    }
    #[tokio::test]
    async fn goal_roundtrip() {
        let (d, h) = open_host("g");
        let s = h.services.session();
        h.goal_set(s.clone(), "tests", vec!["tests".into()])
            .unwrap();
        assert!(h.goal_get(&s).is_some());
        let _ = h.goal_evaluate(&s).await.unwrap();
        h.goal_clear(&s).unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn goal_none() {
        let (d, h) = open_host("g0");
        assert!(h.goal_get(&h.services.session()).is_none());
        cleanup(d);
    }
}
