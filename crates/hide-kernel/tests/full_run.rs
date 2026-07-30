use futures::future::BoxFuture;
use hawking_orch::inference::StubInferenceClient;
use hawking_orch::registry::RoleRegistry;
use hawking_orch::router::SimpleRouter;
use hide_core::event::{Event, EventLog, InMemoryEventLog};
use hide_core::ids::SessionId;
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use hide_kernel::govern::Autonomy;
use hide_kernel::machine::effects::Mode;
use hide_kernel::machine::state::{AgentState, Phase};
use hide_kernel::plan::planner::Planner;
use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
use hide_kernel::runtime_client::KernelRuntimeClient;
use hide_kernel::{allow_all_dispatcher, AgentKernel};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
struct FixedPlanner {
    oracles: Vec<String>,
    kind: StepKind,
    tool_hint: Option<String>,
    tool_args: Option<serde_json::Value>,
}
impl FixedPlanner {
    fn new(oracles: Vec<String>, kind: StepKind) -> Self {
        Self {
            oracles,
            kind,
            tool_hint: None,
            tool_args: None,
        }
    }
}
impl Planner for FixedPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        let oracles = self.oracles.clone();
        let kind = self.kind;
        let tool_hint = self.tool_hint.clone();
        let tool_args = self.tool_args.clone();
        let objective = objective.to_string();
        Box::pin(async move {
            let mut step = PlanStep::new(
                "make the change",
                kind,
                Acceptance::with_oracles("workspace type-checks", oracles),
            );
            step.tool_hint = tool_hint;
            step.tool_args = tool_args;
            Ok(Plan {
                id: hide_core::ids::PlanId::new(),
                title: "fixed".into(),
                objective,
                steps: vec![step],
                status: PlanStatus::Active,
                budget: Default::default(),
            })
        })
    }
}
fn unique() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}_{}_{}",
        std::process::id(),
        hide_core::ids::now_ms(),
        N.fetch_add(1, Ordering::SeqCst)
    )
}
fn git(dir: &Path, args: &[&str]) {
    let _ = Command::new("git").args(args).current_dir(dir).output();
}
fn make_repo(valid: bool) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hide_kernel_it_{}", unique()));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n",
    )
    .unwrap();
    let lib = if valid {
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n"
    } else {
        // type error: returns a &str where i32 is declared.
        "pub fn add(a: i32, b: i32) -> i32 { \"not an int\" }\n"
    };
    std::fs::write(dir.join("src/lib.rs"), lib).unwrap();
    git(&dir, &["init", "-q"]);
    git(&dir, &["config", "user.email", "t@t.t"]);
    git(&dir, &["config", "user.name", "t"]);
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-qm", "init"]);
    dir
}
fn runtime() -> Arc<KernelRuntimeClient> {
    runtime_with("// edit applied")
}
fn runtime_with(out: &str) -> Arc<KernelRuntimeClient> {
    let registry = Arc::new(RoleRegistry::with_default_local_roles());
    let router = Arc::new(SimpleRouter::new(registry));
    let inference = Arc::new(StubInferenceClient::new(out));
    Arc::new(KernelRuntimeClient::new(router, inference))
}
fn build_kernel_with_stub(
    log: DynEventLog,
    root: &Path,
    planner: Arc<dyn Planner>,
    mode: Mode,
    stub_out: &str,
) -> AgentKernel {
    let dispatcher = allow_all_dispatcher(root.to_string_lossy().to_string());
    AgentKernel::builder(log)
        .workspace_root(root.to_string_lossy().to_string())
        .autonomy(Autonomy::FullAuto)
        .mode(mode)
        .planner(planner)
        .runtime(runtime_with(stub_out))
        .dispatcher(dispatcher.clone())
        .with_standard_oracles(dispatcher)
        .build()
}
fn build_kernel(
    log: DynEventLog,
    root: &Path,
    planner: Arc<dyn Planner>,
    mode: Mode,
) -> AgentKernel {
    let dispatcher = allow_all_dispatcher(root.to_string_lossy().to_string());
    AgentKernel::builder(log)
        .workspace_root(root.to_string_lossy().to_string())
        .autonomy(Autonomy::FullAuto)
        .mode(mode)
        .planner(planner)
        .runtime(runtime())
        .dispatcher(dispatcher.clone())
        .with_standard_oracles(dispatcher)
        .build()
}
async fn drive(kernel: &AgentKernel, state: &mut AgentState, max: usize) -> Vec<Phase> {
    let mut phases = vec![state.phase];
    for _ in 0..max {
        if state.phase.is_terminal() {
            break;
        }
        kernel.step(state).await.unwrap();
        phases.push(state.phase);
    }
    phases
}
async fn phase_names(log: &Arc<InMemoryEventLog>) -> Vec<String> {
    let events = log.scan(None, None, None).await.unwrap();
    events
        .iter()
        .filter(|e: &&Event| e.kind == "agent.phase")
        .filter_map(|e| {
            e.payload
                .get("phase")
                .and_then(|v| v.as_str())
                .map(String::from)
        })
        .collect()
}
#[tokio::test]
async fn full_run_passes_through_real_oracle_to_done() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(
        vec!["typecheck".to_string()],
        StepKind::Edit,
    ));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Live);
    let mut state = kernel
        .start_run(SessionId::new(), "implement add()")
        .await
        .unwrap();
    let phases = drive(&kernel, &mut state, 60).await;
    assert_eq!(
        state.phase,
        Phase::Done,
        "run must finish (phases: {phases:?})"
    );
    let names = phase_names(&log).await;
    for expected in ["plan", "select_step", "act", "observe", "verify", "done"] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing phase '{expected}' in {names:?}"
        );
    }
    let v = state
        .last_verdict
        .expect("a verdict was produced by the real oracle");
    assert!(v.is_deterministic());
    assert_eq!(v.status, hide_kernel::verify::oracle::VerdictStatus::Pass);
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn failing_real_oracle_triggers_repair() {
    let repo = make_repo(false); // code does NOT type-check
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(
        vec!["typecheck".to_string()],
        StepKind::Edit,
    ));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Live);
    let mut state = kernel
        .start_run(SessionId::new(), "implement add()")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;
    let names = phase_names(&log).await;
    assert!(names.iter().any(|n| n == "repair"));
    let v = state.last_verdict.expect("a verdict was produced");
    assert_eq!(v.status, hide_kernel::verify::oracle::VerdictStatus::Fail);
    assert!(
        !v.failures.is_empty(),
        "real diagnostics parsed into failures"
    );
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn replay_mode_does_not_run_effects() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Investigate));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Replay);
    let mut state = kernel.start_run(SessionId::new(), "noop").await.unwrap();
    drive(&kernel, &mut state, 40).await;
    let events = log.scan(None, None, None).await.unwrap();
    let action_count = events.iter().filter(|e| e.kind == "agent.action").count();
    assert_eq!(action_count, 0, "replay must not fire Action effects");
    assert!(state.phase.is_terminal());
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn low_tool_call_budget_trips_governor_abort() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    let mut planner = FixedPlanner::new(vec![], StepKind::Command);
    planner.tool_hint = Some("shell.run".to_string());
    planner.tool_args = Some(serde_json::json!({ "argv": ["echo", "hi"] }));
    let kernel = build_kernel(log.clone(), &repo, Arc::new(planner), Mode::Live);
    let mut state = kernel
        .start_run(SessionId::new(), "run a command")
        .await
        .unwrap();
    state.budget.max_tool_calls = 1; // cap reached after a single dispatch
    let _ = drive(&kernel, &mut state, 80).await;
    assert_eq!(
        state.phase,
        Phase::Aborted,
        "low tool-call budget must abort the run"
    );
    assert!(
        state.ledger.tool_calls >= 1,
        "a tool dispatch must be counted"
    );
    let events = log.scan(None, None, None).await.unwrap();
    let abort = events
        .iter()
        .find(|e| e.kind == "run.aborted")
        .expect("a run.aborted event must be recorded");
    assert_eq!(
        abort.payload.get("cap").and_then(|v| v.as_str()),
        Some("tool_calls")
    );
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn effectful_step_without_oracle_is_not_soft_accepted() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Edit));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Live);
    let mut state = kernel
        .start_run(SessionId::new(), "edit but verify nothing")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;
    let events = log.scan(None, None, None).await.unwrap();
    let effectful_soft_accept = events.iter().any(|e| {
        e.kind == "verify.soft_accept"
            && matches!(
                e.payload.get("kind").and_then(|v| v.as_str()),
                Some("Edit") | Some("Command") | Some("Delegate")
            )
    });
    assert!(
        !effectful_soft_accept,
        "effectful step with no oracle must NOT be soft-accepted"
    );
    let phase_names = phase_names(&log).await;
    assert!(phase_names.iter().any(|n| n == "repair" || n == "replan"));
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn model_step_dispatches_emitted_tool_call() {
    let repo = make_repo(true);
    let libpath = repo.join("src/lib.rs");
    let stub_out = format!(
        "<tool_call>{{\"name\":\"fs.read\",\"arguments\":{{\"path\":\"{}\"}}}}</tool_call>",
        libpath.to_string_lossy()
    );
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Investigate));
    let kernel = build_kernel_with_stub(log.clone(), &repo, planner, Mode::Live, &stub_out);
    let mut state = kernel
        .start_run(SessionId::new(), "investigate the code")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;
    let events = log.scan(None, None, None).await.unwrap();
    let dispatched = events.iter().any(|e: &Event| {
        e.kind == "agent.observation"
            && e.payload
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().any(|c| {
                        c.get("tool").and_then(|t| t.as_str()) == Some("fs.read")
                            && c.get("dispatched").and_then(|d| d.as_bool()) == Some(true)
                    })
                })
                .unwrap_or(false)
    });
    assert!(dispatched);
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn model_step_does_not_auto_dispatch_a_mutating_tool() {
    let repo = make_repo(true);
    let target = repo.join("hacked.txt");
    let stub_out = format!(
        "<tool_call>{{\"name\":\"edit.write_file\",\"arguments\":{{\"path\":\"{}\",\"content\":\"pwned\"}}}}</tool_call>",
        target.to_string_lossy()
    );
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Investigate));
    let kernel = build_kernel_with_stub(log.clone(), &repo, planner, Mode::Live, &stub_out);
    let mut state = kernel
        .start_run(SessionId::new(), "investigate")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;
    assert!(
        !target.exists(),
        "a model step must not auto-execute a mutating tool"
    );
    let events = log.scan(None, None, None).await.unwrap();
    let proposed = events.iter().any(|e: &Event| {
        e.kind == "agent.observation"
            && e.payload
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().any(|c| {
                        c.get("tool").and_then(|t| t.as_str()) == Some("edit.write_file")
                            && c.get("dispatched").and_then(|d| d.as_bool()) == Some(false)
                    })
                })
                .unwrap_or(false)
    });
    assert!(
        proposed,
        "the mutating call must be recorded as proposed, not dispatched"
    );
    let _ = std::fs::remove_dir_all(repo);
}
#[tokio::test]
async fn model_step_does_not_auto_dispatch_subprocess_readonly_tool() {
    let repo = make_repo(true);
    let stub_out =
        "<tool_call>{\"name\":\"git.diff\",\"arguments\":{\"ref\":\"HEAD\"}}</tool_call>"
            .to_string();
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Investigate));
    let kernel = build_kernel_with_stub(log.clone(), &repo, planner, Mode::Live, &stub_out);
    let mut state = kernel
        .start_run(SessionId::new(), "investigate")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;
    let events = log.scan(None, None, None).await.unwrap();
    let proposed = events.iter().any(|e: &Event| {
        e.kind == "agent.observation"
            && e.payload
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter().any(|c| {
                        c.get("tool").and_then(|t| t.as_str()) == Some("git.diff")
                            && c.get("dispatched").and_then(|d| d.as_bool()) == Some(false)
                    })
                })
                .unwrap_or(false)
    });
    assert!(
        proposed,
        "a subprocess read-only tool must not auto-dispatch from a model step"
    );
    let _ = std::fs::remove_dir_all(repo);
}
