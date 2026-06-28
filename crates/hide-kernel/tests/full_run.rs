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
}

impl Planner for FixedPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        let oracles = self.oracles.clone();
        let kind = self.kind;
        let objective = objective.to_string();
        Box::pin(async move {
            let step = PlanStep::new(
                "make the change",
                kind,
                Acceptance::with_oracles("workspace type-checks", oracles),
            );
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
    let registry = Arc::new(RoleRegistry::with_default_local_roles());
    let router = Arc::new(SimpleRouter::new(registry));
    let inference = Arc::new(StubInferenceClient::new("// edit applied"));
    Arc::new(KernelRuntimeClient::new(router, inference))
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
    let planner = Arc::new(FixedPlanner {
        oracles: vec!["typecheck".to_string()],
        kind: StepKind::Edit,
    });
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
    let planner = Arc::new(FixedPlanner {
        oracles: vec!["typecheck".to_string()],
        kind: StepKind::Edit,
    });
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
    let planner = Arc::new(FixedPlanner {
        // No oracles + non-effectful → in replay the action folds without effect
        // and the soft-step gate accepts it, so the run still reaches terminal.
        oracles: vec![],
        kind: StepKind::Investigate,
    });
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
