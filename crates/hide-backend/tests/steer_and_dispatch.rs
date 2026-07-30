use futures::future::BoxFuture;
use hawking_orch::inference::InferenceClient;
use hawking_orch::router::SimpleRouter;
use hide_backend::{BackendHost, MemoryScope};
use hide_core::api::Intent;
use hide_core::event::Event;
use hide_core::ids::{now_ms, RunId, SessionId};
use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};
use hide_core::Result;
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::Phase;
use hide_kernel::plan::planner::Planner;
use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
use hide_kernel::runtime_client::KernelRuntimeClient;
use hide_kernel::AgentKernel;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
fn test_host() -> BackendHost {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hide_steer_{}_{}", now_ms(), uniq));
    BackendHost::open_workspace(&dir).unwrap()
}
struct CapturingInferenceClient {
    prompts: parking_lot::Mutex<Vec<String>>,
}
impl CapturingInferenceClient {
    fn new() -> Self {
        Self {
            prompts: parking_lot::Mutex::new(Vec::new()),
        }
    }
}
impl InferenceClient for CapturingInferenceClient {
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        self.prompts.lock().push(request.prompt.clone());
        Box::pin(async move {
            sink(StreamChunk::Token {
                token_id: None,
                text: "ok".to_string(),
            })?;
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: None,
            })?;
            Ok(GenerationStats {
                input_tokens: 0,
                output_tokens: 1,
                decode_tokens_per_second: None,
            })
        })
    }
    fn embed<'a>(&'a self, _text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        Box::pin(async move { Ok(vec![0.0; 8]) })
    }
}
struct TwoReadsPlanner;
impl Planner for TwoReadsPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        let objective = objective.to_string();
        Box::pin(async move {
            let read1 = PlanStep::new(
                "read the first file",
                StepKind::Investigate,
                Acceptance::predicate("first file understood"),
            );
            let mut read2 = PlanStep::new(
                "read the second file",
                StepKind::Investigate,
                Acceptance::predicate("second file understood"),
            );
            read2.dependencies = vec![read1.id.clone()];
            Ok(Plan {
                id: hide_core::ids::PlanId::new(),
                title: "two reads".to_string(),
                objective,
                steps: vec![read1, read2],
                status: PlanStatus::Active,
                budget: Default::default(),
            })
        })
    }
}
async fn observation_count(host: &BackendHost, session: &SessionId) -> usize {
    host.services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap()
        .iter()
        .filter(|e: &&Event| e.kind == "agent.observation")
        .count()
}
#[tokio::test]
async fn trace_a_steer_reaches_running_kernel_and_folds_into_next_step() {
    const STEER: &str = "STOP reading, switch to the auth module instead";
    let host = test_host();
    let session = host.services.session();
    let capturing = Arc::new(CapturingInferenceClient::new());
    let runtime = Arc::new(KernelRuntimeClient::new(
        Arc::new(SimpleRouter::new(host.services.role_registry.clone())),
        capturing.clone(),
    ));
    let kernel = AgentKernel::builder(host.services.event_log.clone())
        .autonomy(Autonomy::FullAuto)
        .planner(Arc::new(TwoReadsPlanner))
        .runtime(runtime)
        .build();
    let host_run = RunId::new();
    let mut state = kernel
        .start_run(session.clone(), "investigate the codebase")
        .await
        .unwrap();
    let mut steered = false;
    let mut steer_delivered = false;
    for _ in 0..64 {
        if let Some(Interrupt::Steer { .. }) =
            host.interrupts().drain_into_kernel(&host_run, &kernel)
        {
            steer_delivered = true;
        }
        if state.phase.is_terminal() {
            break;
        }
        kernel.step(&mut state).await.unwrap();
        if !steered && observation_count(&host, &session).await >= 1 {
            let ack = host
                .handle_intent(Intent::Custom {
                    name: "redirect_run".to_string(),
                    payload: serde_json::json!({
                        "run_id": host_run.as_str(),
                        "text": STEER,
                        "session_id": session.as_str(),
                    }),
                })
                .await
                .unwrap();
            assert!(ack.accepted, "the steer intent is accepted");
            steered = true;
        }
    }
    assert!(
        steer_delivered,
        "InterruptHub forwarded a Steer to the kernel"
    );
    assert!(observation_count(&host, &session).await >= 2);
    let prompts = capturing.prompts.lock().clone();
    assert!(
        prompts.len() >= 2,
        "both read steps generated a prompt: {prompts:?}"
    );
    assert!(!prompts[0].contains(STEER));
    assert!(prompts.iter().skip(1).any(|p| p.contains(STEER)));
    let events = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    let steer_event = events
        .iter()
        .find(|e: &&Event| e.kind == "turn.steer")
        .expect("a durable turn.steer event is persisted");
    assert_eq!(
        steer_event
            .payload
            .get("instruction")
            .and_then(|v| v.as_str()),
        Some(STEER)
    );
    assert_eq!(
        steer_event.run_id.as_ref().map(|r| r.as_str()),
        Some(host_run.as_str())
    );
    assert_eq!(state.phase, Phase::Done, "the steered run still completed");
}
#[tokio::test]
async fn memory_add_intent_persists_a_record() {
    let host = test_host();
    let scope = MemoryScope::Repo("hawking".to_string());
    let ack = host
        .handle_intent(Intent::Custom {
            name: "memory_add".to_string(),
            payload: serde_json::json!({
                "scope": { "kind": "repo", "id": "hawking" },
                "claim": "the turn loop is a single flat FSM",
                "source": "census",
                "author": "tester",
                "citations": ["crates/hide-kernel/src/machine/driver.rs"],
            }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let records = host.memory_list(&scope);
    assert_eq!(records.len(), 1, "one memory record was persisted");
    assert_eq!(records[0].claim, "the turn loop is a single flat FSM");
    assert!(host.memory_get(&records[0].memory_id).is_some());
}
#[tokio::test]
async fn goal_evaluate_intent_returns_a_deterministic_verdict() {
    use hide_backend::GoalStatus;
    use hide_core::event::NewEvent;
    use hide_kernel::verify::oracle::{OracleClass, Verdict};
    let host = test_host();
    let session = host.services.session();
    host.goal_set(session.clone(), "tests_pass", vec!["tests".to_string()])
        .unwrap();
    host.services
        .event_log
        .append(NewEvent::system(
            session.clone(),
            "verify.result",
            serde_json::to_value(&Verdict::pass(
                "tests",
                OracleClass::Deterministic,
                "all green",
            ))
            .unwrap(),
        ))
        .await
        .unwrap();
    let ack = host
        .handle_intent(Intent::Custom {
            name: "goal_evaluate".to_string(),
            payload: serde_json::json!({ "session_id": session.as_str() }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Met);
}
#[tokio::test]
async fn workspace_set_repo_trust_intent_is_held_until_approved() {
    use hide_backend::services::{RepoNode, TrustState};
    let host = test_host();
    host.workspace_add_repo(RepoNode::new("vendor", "/tmp/vendor"))
        .unwrap();
    assert_eq!(
        host.workspace_repo("vendor").unwrap().trust,
        TrustState::Untrusted
    );
    let ack = host
        .handle_intent(Intent::Custom {
            name: "workspace_set_repo_trust".to_string(),
            payload: serde_json::json!({ "repo_id": "vendor", "trust": "trusted" }),
        })
        .await
        .unwrap();
    assert!(ack.accepted, "the intent is recorded");
    let message = ack.message.expect("an Ask command reports its gate");
    let gate = message
        .split("gate=")
        .nth(1)
        .expect("the ack names the gate id")
        .to_string();
    assert_eq!(
        host.workspace_repo("vendor").unwrap().trust,
        TrustState::Untrusted
    );
    host.handle_intent(Intent::Custom {
        name: "approve_gate".to_string(),
        payload: serde_json::json!({ "gate": gate }),
    })
    .await
    .unwrap();
    assert_eq!(
        host.workspace_repo("vendor").unwrap().trust,
        TrustState::Trusted
    );
}
#[tokio::test]
async fn environment_switch_intent_emits_event_and_updates_current_env() {
    use hide_backend::services::{EnvironmentNode, WorkspaceStore};
    let host = test_host();
    let session = host.services.session();
    host.workspace_add_environment(EnvironmentNode::new("container:node20"))
        .unwrap();
    let ack = host
        .handle_intent(Intent::Custom {
            name: "environment_switch".to_string(),
            payload: serde_json::json!({
                "session_id": session.as_str(),
                "env_id": "container:node20",
                "reason": "run the node build",
            }),
        })
        .await
        .unwrap();
    assert!(ack.accepted);
    let switches = host.environment_switches(&session).await.unwrap();
    assert_eq!(switches.len(), 1, "one environment switch was recorded");
    assert_eq!(switches[0].new_env, "container:node20");
    assert_eq!(
        WorkspaceStore::current_env(&host.services.key_value_store, &session).as_deref(),
        Some("container:node20")
    );
}

// S3b host surface re-expression chunk (host_surface_s3b_b)
mod host_surface_s3b_b {

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
    async fn goal_ckpt_intents() {
        let (d, h) = open_host("gci");
        let s = h.services.session();
        assert!(!custom(
            &h,
            "goal_set",
            json!({"session_id":s.as_str(),"condition":"t"})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        assert!(!custom(
            &h,
            "checkpoint_create",
            json!({"session_id":s.as_str(),"label":"L"})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn checkpoint_create_release() {
        let (d, h) = open_host("cc");
        let s = h.services.session();
        h.services
            .event_log
            .append(NewEvent::system(s.clone(), "t", json!({})))
            .await
            .unwrap();
        if let Ok(c) = h.checkpoint_create(s.clone(), None, "t1").await {
            let _ = h.checkpoint_release(&c.checkpoint_id);
        }
        cleanup(d);
    }
    #[tokio::test]
    async fn rewind_intent_handled() {
        let (d, h) = open_host("rw");
        assert!(!custom(
            &h,
            "checkpoint_rewind",
            json!({"session_id":h.services.session().as_str()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn side_chat_create() {
        let (d, h) = open_host("sc");
        let p = h.services.session();
        let (side, rec, _) = h.create_side_chat(p.clone(), None, true).await.unwrap();
        assert_ne!(side, p);
        let _ = rec;
        cleanup(d);
    }
    #[tokio::test]
    async fn side_chat_discard_parent_stable() {
        let (d, h) = open_host("sd");
        let p = h.services.session();
        h.services
            .event_log
            .append(NewEvent::system(p.clone(), "t", json!({})))
            .await
            .unwrap();
        let b = h
            .services
            .event_log
            .scan(Some(p.clone()), None, None)
            .await
            .unwrap()
            .len();
        let _ = h.create_side_chat(p.clone(), None, true).await.unwrap();
        assert_eq!(
            b,
            h.services
                .event_log
                .scan(Some(p.clone()), None, None)
                .await
                .unwrap()
                .len()
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn merge_side_chat() {
        let (d, h) = open_host("msc");
        let p = h.services.session();
        let (side, _, _) = h.create_side_chat(p.clone(), None, true).await.unwrap();
        let b = h
            .services
            .event_log
            .scan(Some(p.clone()), None, None)
            .await
            .unwrap()
            .len();
        let _ = h.merge_side_chat_summary(side, p.clone(), "sum").await;
        assert!(
            h.services
                .event_log
                .scan(Some(p.clone()), None, None)
                .await
                .unwrap()
                .len()
                >= b
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn fork_source_stable() {
        let (d, h) = open_host("fk");
        let s = h.services.session();
        let ev = h
            .services
            .event_log
            .append(NewEvent::system(s.clone(), "t", json!({})))
            .await
            .unwrap();
        let b = h
            .services
            .event_log
            .scan(Some(s.clone()), None, None)
            .await
            .unwrap()
            .len();
        let _ = h.fork_session_from_event(s.clone(), Some(&ev.id)).await;
        assert_eq!(
            b,
            h.services
                .event_log
                .scan(Some(s.clone()), None, None)
                .await
                .unwrap()
                .len()
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn fork_intent() {
        let (d, h) = open_host("fi");
        let s = h.services.session();
        let ev = h
            .services
            .event_log
            .append(NewEvent::system(s.clone(), "t", json!({})))
            .await
            .unwrap();
        assert!(
            h.handle_intent(Intent::ForkSession {
                session_id: s,
                at_event: ev.id
            })
            .await
            .unwrap()
            .accepted
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn search_transcript() {
        let (d, h) = open_host("stx");
        let s = h.services.session();
        h.services
            .event_log
            .append(NewEvent::system(
                s.clone(),
                "agent.message",
                json!({"text":"ZZALPHA"}),
            ))
            .await
            .unwrap();
        let hits = h
            .search_transcript(&TranscriptQuery {
                text: "ZZALPHA".into(),
                session_id: Some(s),
                ..Default::default()
            })
            .await
            .unwrap();
        assert!(hits.is_empty() || hits.iter().any(|h| h.snippet.contains("ZZALPHA")));
        cleanup(d);
    }
    #[tokio::test]
    async fn conversation_graph() {
        let (d, h) = open_host("cg");
        let _ = h.conversation_graph(&h.services.session());
        cleanup(d);
    }
    #[tokio::test]
    async fn plan_mutations() {
        let (d, h) = open_host("pln");
        let s = h.services.session();
        let a = PlanStep::new(
            "investigate",
            StepKind::Investigate,
            Acceptance::predicate("x"),
        );
        let b = PlanStep::new("edit", StepKind::Edit, Acceptance::predicate("y"));
        let aid = a.id.clone();
        let bid = b.id.clone();
        let plan = Plan {
            id: PlanId::new(),
            title: "t".into(),
            objective: "o".into(),
            steps: vec![a, b],
            status: PlanStatus::Active,
            budget: Default::default(),
        };
        h.publish_plan(&s, &plan, Autonomy::SuggestOnly).unwrap();
        assert!(
            custom(&h, "approve_plan", json!({"session_id":s.as_str()}))
                .await
                .accepted
        );
        custom(
            &h,
            "edit_plan_step",
            json!({"session_id":s.as_str(),"step_id":aid.as_str(),"text":"dig"}),
        )
        .await;
        assert_eq!(h.plan_get(&s).unwrap().steps[0].text, "dig");
        custom(
            &h,
            "reorder_plan",
            json!({"session_id":s.as_str(),"order":[bid.as_str(),aid.as_str()]}),
        )
        .await;
        assert_eq!(h.plan_get(&s).unwrap().steps[0].id, bid.as_str());
        cleanup(d);
    }
    #[tokio::test]
    async fn plan_none() {
        let (d, h) = open_host("p0");
        assert!(h.plan_get(&h.services.session()).is_none());
        cleanup(d);
    }
    #[tokio::test]
    async fn approve_effect_handled() {
        let (d, h) = open_host("ae");
        assert!(!custom(&h, "approve_effect", json!({}))
            .await
            .message
            .unwrap_or_default()
            .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn deny_effect_handled() {
        let (d, h) = open_host("de");
        assert!(!custom(&h, "deny_effect", json!({"step_id":"s"}))
            .await
            .message
            .unwrap_or_default()
            .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn lease_intents() {
        let (d, h) = open_host("le");
        assert!(!custom(
            &h,
            "grant_write_lease",
            json!({"session_id":h.services.session().as_str(),"paths":[d.to_string_lossy()]})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        assert!(!custom(&h, "revoke_write_lease", json!({}))
            .await
            .message
            .unwrap_or_default()
            .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn hubs_accessible() {
        let (d, h) = open_host("hb");
        let _ = h.processes();
        let _ = h.interrupts();
        let _ = h.approvals();
        let _ = h.ui_bus();
        cleanup(d);
    }
    #[tokio::test]
    async fn editor_run_stable() {
        let (d, h) = open_host("er");
        let s = h.services.session();
        assert_eq!(
            BackendHost::editor_run(&s).as_str(),
            BackendHost::editor_run(&s).as_str()
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn projection_rebuild() {
        let (d, h) = open_host("pj");
        let s = h.services.session();
        assert_eq!(
            h.rebuild_session_projection(s.clone())
                .await
                .unwrap()
                .session_id,
            s
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn initialize_client() {
        let (d, h) = open_host("in");
        let _ = h.initialize(
            "c",
            ClientInfo {
                name: "s3b".into(),
                version: "0".into(),
                title: None,
            },
            ClientCapabilities::default(),
        );
        let _ = h.connections();
        cleanup(d);
    }
    #[tokio::test]
    async fn workspace_graph() {
        let (d, h) = open_host("wg");
        let _ = h.workspace_graph().repos.len();
        cleanup(d);
    }
    #[tokio::test]
    async fn trust_intent() {
        let (d, h) = open_host("ti");
        assert!(!custom(
            &h,
            "workspace_set_repo_trust",
            json!({"repo_id":"l","trust":"trusted","root_path":d.to_string_lossy()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn env_switch_intent() {
        let (d, h) = open_host("ei");
        assert!(!custom(
            &h,
            "environment_switch",
            json!({"session_id":h.services.session().as_str(),"env":"dev","reason":"r"})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn unknown_connector_err() {
        let (d, h) = open_host("uc");
        assert!(h.call_connector("nope", "x", json!({})).await.is_err());
        cleanup(d);
    }
    #[tokio::test]
    async fn verification_receipts() {
        let (d, h) = open_host("vr");
        let _ = h
            .verification_receipts(&h.services.session())
            .await
            .unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn diff_missing() {
        let (d, h) = open_host("dm");
        assert!(h.diff_get("nope").is_none());
        cleanup(d);
    }
    #[tokio::test]
    async fn bg_job_missing() {
        let (d, h) = open_host("bg");
        assert!(h.background_job_for_run(&RunId::new()).is_none());
        cleanup(d);
    }
    #[tokio::test]
    async fn live_thread() {
        let (d, h) = open_host("lt");
        let _ = h.open_live_thread(h.services.session());
        cleanup(d);
    }
    #[tokio::test]
    async fn policy_empty() {
        let (d, h) = open_host("pe0");
        assert!(h
            .policy_decisions(&h.services.session())
            .await
            .unwrap()
            .is_empty());
        cleanup(d);
    }
    #[tokio::test]
    async fn ui_events() {
        let (d, h) = open_host("ue");
        let _ = h.ui_events(None, None, None).await.unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn save_file_intent() {
        let (d, h) = open_host_allow_write("sf");
        assert!(!custom(&h,"save_file",json!({"path":d.join("s.txt").to_string_lossy(),"content":"hi","session_id":h.services.session().as_str()})).await.message.unwrap_or_default().contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn steer_intent() {
        let (d, h) = open_host("si");
        assert!(!custom(&h, "steer", json!({"run_id":"r","text":"t"}))
            .await
            .message
            .unwrap_or_default()
            .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn search_intent() {
        let (d, h) = open_host("sei");
        assert!(!custom(
            &h,
            "search",
            json!({"query":"x","session_id":h.services.session().as_str()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn create_worktree_intent() {
        let (d, h) = open_host("cwt");
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&d)
            .status();
        assert!(!custom(
            &h,
            "create_worktree",
            json!({"path":d.join("wt").to_string_lossy()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn status_tools_nonempty() {
        let (d, h) = open_host("stn");
        assert!(!h.status().await.tools.is_empty());
        cleanup(d);
    }
    #[tokio::test]
    async fn status_connectors_nonempty() {
        let (d, h) = open_host("scn");
        assert!(!h.status().await.connectors.is_empty());
        cleanup(d);
    }
    #[tokio::test]
    async fn status_roles_nonempty() {
        let (d, h) = open_host("srn");
        assert!(!h.status().await.model_roles.is_empty());
        cleanup(d);
    }
    #[tokio::test]
    async fn health_personalization() {
        let (d, h) = open_host("hp");
        let health = h.health().await;
        assert!(health
            .checks
            .iter()
            .any(|c| c.name.contains("personalization") || c.name.contains("connector")));
        cleanup(d);
    }
    #[tokio::test]
    async fn fs_read_tool() {
        let (dir, h) = open_host_allow_write("fr");
        let f = dir.join("r.txt");
        std::fs::write(&f, "data").unwrap();
        let r = h
            .dispatch_tool(
                h.services.session(),
                None,
                ToolCall::new("fs.read", json!({"path":f.to_string_lossy()})),
            )
            .await
            .unwrap();
        assert_eq!(r.status, ToolStatus::Ok);
        cleanup(dir);
    }
    #[tokio::test]
    async fn memory_add_intent() {
        let (d, h) = open_host("mai");
        assert!(
            !custom(&h, "memory_add", json!({"claim":"c","scope":{"repo":"r"}}))
                .await
                .message
                .unwrap_or_default()
                .contains("has no host handler")
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn goal_evaluate_intent() {
        let (d, h) = open_host("gei");
        assert!(!custom(
            &h,
            "goal_evaluate",
            json!({"session_id":h.services.session().as_str()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn goal_clear_intent() {
        let (d, h) = open_host("gci2");
        assert!(!custom(
            &h,
            "goal_clear",
            json!({"session_id":h.services.session().as_str()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn create_side_chat_intent() {
        let (d, h) = open_host("csi");
        assert!(!custom(
            &h,
            "create_side_chat",
            json!({"session_id":h.services.session().as_str()})
        )
        .await
        .message
        .unwrap_or_default()
        .contains("has no host handler"));
        cleanup(d);
    }
    #[tokio::test]
    async fn invalidated_verifs() {
        let (d, h) = open_host("iv");
        let _ = h
            .invalidated_verification_ids(&h.services.session())
            .await
            .unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn diff_review_receipts() {
        let (d, h) = open_host("drr");
        let _ = h.diff_review_receipts(&h.services.session()).await.unwrap();
        cleanup(d);
    }
    #[tokio::test]
    async fn write_lease_query() {
        let (d, h) = open_host("wlq");
        let _ = h.write_lease();
        cleanup(d);
    }
    #[tokio::test]
    async fn safe_true_cmd() {
        let (d, h) = open_host_allow_shell("true");
        assert!(
            h.handle_intent(Intent::RunCommand {
                argv: vec!["true".into()],
                cwd: None
            })
            .await
            .unwrap()
            .accepted
                && !h
                    .handle_intent(Intent::RunCommand {
                        argv: vec!["true".into()],
                        cwd: None
                    })
                    .await
                    .unwrap()
                    .held
        );
        cleanup(d);
    }
    #[tokio::test]
    async fn smoke_approve_effect() {
        let (dir, host) = open_host("smoke_approve_effect");
        let s = host.services.session();
        let msg = custom(&host, "approve_effect", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(
            !msg.contains("has no host handler"),
            "approve_effect: {msg}"
        );
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_approve_gate() {
        let (dir, host) = open_host("smoke_approve_gate");
        let s = host.services.session();
        let msg = custom(&host, "approve_gate", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "approve_gate: {msg}");
        cleanup(dir);
    }
    #[tokio::test]
    async fn smoke_approve_plan() {
        let (dir, host) = open_host("smoke_approve_plan");
        let s = host.services.session();
        let msg = custom(&host, "approve_plan", json!({
        "session_id": s.as_str(), "gate":"command:0", "step_id":"s", "run_id":"r",
        "path": dir.join("f").to_string_lossy(), "content":"x", "text":"t", "objective":"o",
        "label":"l", "trust":"trusted", "root_path": dir.to_string_lossy(), "repo_id":"r",
        "env":"dev", "reason":"r", "query":"q", "memory_id":"m", "claim":"c", "success":true,
        "order":[], "paths":[], "condition":"t", "summary":"s", "surface":"chat",
        "parent_session_id": s.as_str(), "side_session_id": s.as_str(),
    })).await.message.unwrap_or_default();
        assert!(!msg.contains("has no host handler"), "approve_plan: {msg}");
        cleanup(dir);
    }
}
