//! Flagship integration test (bible ch.02 K1) — a full agent run on a tiny REAL
//! cargo repo, driven by a [`StubInferenceClient`] (deterministic, no live
//! model) but verified by **REAL** `cargo`/`git` oracles shelling through the
//! `hide-tools` catalog.
//!
//! Proves end-to-end:
//! * the FSM walks INTAKE → PLAN → SELECT_STEP → ACT → OBSERVE → VERIFY(real
//!   `cargo check`) → … → DONE on a valid repo;
//! * a deliberately broken repo makes the real oracle FAIL and the gate routes
//!   the run into REPAIR (K1: no advance on faith);
//! * replay mode does not re-fire effects.

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

// --- a deterministic planner that emits a known single-step plan -------------

struct FixedPlanner {
    oracles: Vec<String>,
    kind: StepKind,
    /// Optional tool the step dispatches through the real dispatcher.
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

// --- tiny real cargo repo in a tempdir ---------------------------------------

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

/// Create a minimal cargo lib crate. `valid=false` writes code that does not
/// type-check, so the real `cargo check` oracle fails.
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

fn build_kernel(log: DynEventLog, root: &Path, planner: Arc<dyn Planner>, mode: Mode) -> AgentKernel {
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
    let planner = Arc::new(FixedPlanner::new(vec!["typecheck".to_string()], StepKind::Edit));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Live);

    let mut state = kernel
        .start_run(SessionId::new(), "implement add()")
        .await
        .unwrap();
    let phases = drive(&kernel, &mut state, 60).await;

    assert_eq!(state.phase, Phase::Done, "run must finish (phases: {phases:?})");
    // The FSM visited the canonical lifecycle phases.
    let names = phase_names(&log).await;
    for expected in ["plan", "select_step", "act", "observe", "verify", "done"] {
        assert!(
            names.iter().any(|n| n == expected),
            "missing phase '{expected}' in {names:?}"
        );
    }
    // A real deterministic verdict was recorded and it PASSED (cargo check exit 0).
    let v = state.last_verdict.expect("a verdict was produced by the real oracle");
    assert!(v.is_deterministic());
    assert_eq!(v.status, hide_kernel::verify::oracle::VerdictStatus::Pass);

    let _ = std::fs::remove_dir_all(repo);
}

#[tokio::test]
async fn failing_real_oracle_triggers_repair() {
    let repo = make_repo(false); // code does NOT type-check
    let log = Arc::new(InMemoryEventLog::new());
    let planner = Arc::new(FixedPlanner::new(vec!["typecheck".to_string()], StepKind::Edit));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Live);

    let mut state = kernel
        .start_run(SessionId::new(), "implement add()")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;

    // The run visited REPAIR because the real `cargo check` oracle FAILED.
    let names = phase_names(&log).await;
    assert!(
        names.iter().any(|n| n == "repair"),
        "broken repo must drive into repair; phases: {names:?}"
    );
    // The recorded verdict is a deterministic FAIL with structured failures.
    let v = state.last_verdict.expect("a verdict was produced");
    assert_eq!(v.status, hide_kernel::verify::oracle::VerdictStatus::Fail);
    assert!(!v.failures.is_empty(), "real diagnostics parsed into failures");

    let _ = std::fs::remove_dir_all(repo);
}

#[tokio::test]
async fn replay_mode_does_not_run_effects() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    // No oracles + non-effectful → in replay the action folds without effect
    // and the soft-step gate accepts it, so the run still reaches terminal.
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Investigate));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Replay);

    let mut state = kernel
        .start_run(SessionId::new(), "noop")
        .await
        .unwrap();
    drive(&kernel, &mut state, 40).await;

    // In replay mode no `agent.action` Action event is emitted (effects skipped).
    let events = log.scan(None, None, None).await.unwrap();
    let action_count = events.iter().filter(|e| e.kind == "agent.action").count();
    assert_eq!(action_count, 0, "replay must not fire Action effects");
    // It still reaches a terminal state by folding.
    assert!(state.phase.is_terminal());

    let _ = std::fs::remove_dir_all(repo);
}

/// FIX 1 (budget): a low `max_tool_calls` budget must trip a structured Governor
/// Abort once a real tool is dispatched. Before the fix the ledger's
/// `tool_calls` stayed 0 (the driver never consumed it) so the cap was inert.
#[tokio::test]
async fn low_tool_call_budget_trips_governor_abort() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    // A Command step that dispatches a real `shell.run` (echo) through the
    // dispatcher — and declares no oracle, so after the dispatch the gate is
    // inconclusive and the loop re-attempts, dispatching again. With a cap of 1,
    // the second Governor check must abort on the tool-call cap.
    let mut planner = FixedPlanner::new(vec![], StepKind::Command);
    planner.tool_hint = Some("shell.run".to_string());
    planner.tool_args = Some(serde_json::json!({ "argv": ["echo", "hi"] }));
    let kernel = build_kernel(log.clone(), &repo, Arc::new(planner), Mode::Live);

    let mut state = kernel.start_run(SessionId::new(), "run a command").await.unwrap();
    state.budget.max_tool_calls = 1; // cap reached after a single dispatch

    let _ = drive(&kernel, &mut state, 80).await;

    assert_eq!(state.phase, Phase::Aborted, "low tool-call budget must abort the run");
    // At least one real tool call was consumed against the ledger.
    assert!(state.ledger.tool_calls >= 1, "a tool dispatch must be counted");
    // The abort was the structured ToolCalls cap (cap = "tool_calls").
    let events = log.scan(None, None, None).await.unwrap();
    let abort = events
        .iter()
        .find(|e| e.kind == "run.aborted")
        .expect("a run.aborted event must be recorded");
    assert_eq!(
        abort.payload.get("cap").and_then(|v| v.as_str()),
        Some("tool_calls"),
        "abort must be the structured tool-call cap, payload: {:?}",
        abort.payload
    );

    let _ = std::fs::remove_dir_all(repo);
}

/// FIX 3 (soft-step escape hatch): an EFFECTFUL step with no declared oracle must
/// NOT be soft-accepted. It produced an effect with nothing verifying it, so K1
/// forbids advancing on faith — the gate is Inconclusive and the loop must route
/// to repair/replan rather than silently completing the step.
#[tokio::test]
async fn effectful_step_without_oracle_is_not_soft_accepted() {
    let repo = make_repo(true);
    let log = Arc::new(InMemoryEventLog::new());
    // Edit (effectful) with NO oracle and NO tool_hint → act_model records the
    // change but no oracle runs. The soft-step branch must NOT fire (it is gated
    // on `!is_effectful()`); the step goes through the gate instead.
    let planner = Arc::new(FixedPlanner::new(vec![], StepKind::Edit));
    let kernel = build_kernel(log.clone(), &repo, planner, Mode::Live);

    let mut state = kernel
        .start_run(SessionId::new(), "edit but verify nothing")
        .await
        .unwrap();
    let _ = drive(&kernel, &mut state, 60).await;

    let events = log.scan(None, None, None).await.unwrap();
    // The EFFECTFUL step must never be soft-accepted. (A localized replan may
    // legitimately soft-accept a NON-effectful `Investigate` probe it inserts —
    // that is the documented, correct behavior — so we assert specifically that
    // no soft-accept names an effectful step kind.)
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
    // The effectful step instead routed through repair (the gate returned
    // Inconclusive on its empty verdict set, not Accept) — it did not silently pass.
    let phase_names = phase_names(&log).await;
    assert!(
        phase_names.iter().any(|n| n == "repair" || n == "replan"),
        "effectful unverified step must route to repair/replan; phases: {phase_names:?}"
    );

    let _ = std::fs::remove_dir_all(repo);
}

#[tokio::test]
async fn model_step_dispatches_emitted_tool_call() {
    // End-to-end: a model step whose generated text contains a <tool_call> must
    // actually dispatch it through the permission-gated loop, and the resulting
    // observation event must record the dispatch. This proves Phase 0 is wired
    // into the live agent driver, not just a standalone library.
    let repo = make_repo(true);
    let libpath = repo.join("src/lib.rs");
    let stub_out = format!(
        "<tool_call>{{\"name\":\"fs.read\",\"arguments\":{{\"path\":\"{}\"}}}}</tool_call>",
        libpath.to_string_lossy()
    );
    let log = Arc::new(InMemoryEventLog::new());
    // A model step (Investigate), no oracles: it generates, we dispatch, done.
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
            && e
                .payload
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
    assert!(
        dispatched,
        "the model-emitted fs.read tool call must be dispatched and recorded in an observation"
    );

    let _ = std::fs::remove_dir_all(repo);
}

#[tokio::test]
async fn model_step_does_not_auto_dispatch_a_mutating_tool() {
    // Doctrine guard: a model step must NOT be able to mutate the workspace by
    // emitting a tool call. A write tool it emits is recorded as "proposed" but
    // never executed; the file must not appear.
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

    // The write must NOT have happened.
    assert!(!target.exists(), "a model step must not auto-execute a mutating tool");
    // And it must be recorded as proposed / not dispatched.
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
    assert!(proposed, "the mutating call must be recorded as proposed, not dispatched");

    let _ = std::fs::remove_dir_all(repo);
}

#[tokio::test]
async fn model_step_does_not_auto_dispatch_subprocess_readonly_tool() {
    // git.diff is annotated read-only but shells out; the deny-by-default allowlist
    // must refuse to auto-dispatch it from a model step (defense against the
    // arg-injection escalation the review found). Recorded as proposed, not run.
    let repo = make_repo(true);
    let stub_out =
        "<tool_call>{\"name\":\"git.diff\",\"arguments\":{\"ref\":\"HEAD\"}}</tool_call>".to_string();
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
