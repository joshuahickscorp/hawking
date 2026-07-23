//! FIRST MANDATORY MODEL-FREE IMPLEMENTATION RECEIPT (HIDE Implementation Bible,
//! Book XXI §76 / RIP §1.5 "Prove").
//!
//! This proves the REAL HIDE app path end-to-end with NO model. A deterministic
//! SCRIPTED decision driver substitutes ONLY for the two model-decision seams - 
//! the [`Planner`] (which steps to take) and the [`InferenceClient`] (what a model
//! step generates) - and NOTHING else. Every other subsystem is the real one:
//!
//! * the real [`AgentKernel`] FSM driver (plan → select → act → observe → verify →
//!   repair/replan → done), gate, Governor and budget ledger;
//! * the real permission-gated [`ToolDispatcher`] over the real `hide-tools`
//!   catalog (edit.search_replace / edit.write_file / fs.read / search.text /
//!   test.run / compile.check) - the SecurityServices `StaticPermissionEngine`;
//! * the real deterministic `ProcessOracle` suite shelling out to REAL `cargo`
//!   (`cargo check` / `cargo test`) against a REAL temp git repo;
//! * the real durable `BackendHost` / `BackendServices` event log (JSONL on disk),
//!   session registry, context compiler (`hawking-context` over the code index),
//!   and replay/resume service.
//!
//! It proves ORCHESTRATION / DURABILITY / TOOLS / VERIFICATION - it does NOT prove
//! intelligence. The model is deferred: the receipt is stamped
//! `model = "scripted-driver (DEFERRED_MODEL_REQUIRED)"`.
//!
//! The scripted arc is a genuine failure-then-recovery: the fixture crate ships a
//! real bug (`largest` returns the first element) with a real failing test. Plan A
//! reads/searches the code, edits a second file, then applies a WRONG fix that
//! still fails the REAL `cargo test` oracle; the kernel reacts (localized then full
//! replan); Plan B applies the corrected fix and the REAL `cargo test` passes.
//!
//! Behaviors covered (of the FIRST_RECEIPT_HARNESS 20): 1 (open+trust, via workspace
//! open + a durable `repo.trusted` marker - the dedicated trust-gate subsystem is a
//! documented future increment), 2 (session), 3 (goal + declared acceptance oracles),
//! 4 (compile context from the real index, folded into the objective), 5 (plan with
//! declared oracles), 6 (read + search files), 8 (edit transactionally), 9 (build /
//! typecheck / tests via real ProcessOracle), 10 (react to a failure: repair/replan),
//! 11 (accept a steering message: a real `Interrupt::Steer` at a safe boundary), 12
//! (green verification receipt: `verify.result` events), 14 (persist session), 15
//! (restart server: drop + re-open `BackendHost`), 16 (resume thread: replay from the
//! durable log), 20 (export receipt). Follow-up increment (NOT covered here): 17-19
//! (fork thread for review / search transcript / reviewer compare) and the thin-mode
//! baseline (RIP §1.5 "Prove"-delta) - noted in the receipt.

use futures::future::BoxFuture;
use hawking_context::compiler::CompileInput;
use hawking_context::profiles::ContextProfile;
use hawking_context::sources::CodeIndexContextSource;
use hawking_context::ContextCompiler;
use hawking_index::{CodeIndex, InMemoryCodeIndex};
use hawking_orch::inference::{deterministic_embedding, InferenceClient};
use hawking_orch::registry::RoleRegistry;
use hawking_orch::router::SimpleRouter;
use hide_backend::{BackendHost, BackendServices};
use hide_core::config::HideConfig;
use hide_core::event::{Event, NewEvent};
use hide_core::ids::{now_ms, PlanId, RunId, SessionId};
use hide_core::persistence::DynEventLog;
use hide_core::runtime::{
    GenerationStats, InferenceRequest, RolePurpose, StreamChunk, TokenSink,
};
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry};
use hide_core::types::Decision;
use hide_core::Result;
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::{AgentState, Phase};
use hide_kernel::plan::planner::Planner;
use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
use hide_kernel::runtime_client::KernelRuntimeClient;
use hide_kernel::{AgentKernel, Grounding};
use parking_lot::Mutex;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

// ===========================================================================
// Fixture: a tiny REAL cargo crate with a REAL bug and a REAL failing test.
// ===========================================================================

const LIB_BUGGY: &str = "pub mod math;\n\n/// BUG: returns the first element, not the largest.\npub fn largest(v: &[i32]) -> i32 {\n    v[0]\n}\n";
const LIB_WRONG_FIX: &str = "pub mod math;\n\n/// First attempt (WRONG): returns the last element.\npub fn largest(v: &[i32]) -> i32 {\n    v[v.len() - 1]\n}\n";
const LIB_CORRECT_FIX: &str = "pub mod math;\n\n/// Correct: returns the maximum element.\npub fn largest(v: &[i32]) -> i32 {\n    let mut m = v[0];\n    for &x in v {\n        if x > m {\n            m = x;\n        }\n    }\n    m\n}\n";
const MATH_SRC: &str = "pub fn scale(x: i32) -> i32 { x * 2 }\n";
const TEST_SRC: &str = "#[test]\nfn largest_of_three() {\n    assert_eq!(fixture::largest(&[1, 5, 3]), 5);\n}\n\n#[test]\nfn scale_doubles() {\n    assert_eq!(fixture::math::scale(2), 4);\n}\n";

fn unique() -> String {
    static N: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}_{}_{}",
        std::process::id(),
        now_ms(),
        N.fetch_add(1, Ordering::SeqCst)
    )
}

fn git(dir: &Path, args: &[&str]) {
    let _ = Command::new("git").args(args).current_dir(dir).output();
}

/// Create the multi-file buggy fixture crate + initial git commit. Returns its root.
fn make_buggy_repo() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("hide_first_receipt_{}", unique()));
    std::fs::create_dir_all(dir.join("src")).unwrap();
    std::fs::create_dir_all(dir.join("tests")).unwrap();
    // `[workspace]` pins this crate as its own workspace root so `cargo` never
    // walks up into an ancestor workspace of the temp dir.
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n\n[workspace]\n",
    )
    .unwrap();
    std::fs::write(dir.join("src/lib.rs"), LIB_BUGGY).unwrap();
    std::fs::write(dir.join("src/math.rs"), MATH_SRC).unwrap();
    std::fs::write(dir.join("tests/behavior.rs"), TEST_SRC).unwrap();
    git(&dir, &["init", "-q"]);
    git(&dir, &["config", "user.email", "t@t.t"]);
    git(&dir, &["config", "user.name", "t"]);
    git(&dir, &["add", "-A"]);
    git(&dir, &["commit", "-qm", "init: buggy largest()"]);
    dir
}

/// A permissive-but-REAL security config: the real `StaticPermissionEngine` +
/// `policy_for_config` rules are used; only the decisions are set to Allow so the
/// headless run can write files and shell `cargo` without an interactive approver.
fn permissive_config(root: &Path) -> HideConfig {
    let mut config = HideConfig::for_workspace(root);
    config.security.default_decision = Decision::Allow;
    config.security.workspace_write_default = Decision::Allow;
    config.security.shell_default = Decision::Allow;
    config
}

/// Build the REAL permission-gated dispatcher over the REAL tool catalog, rooted
/// at the fixture with the OS sandbox disabled (the doctrine's already-confined
/// worktree opt-out - an isolated temp repo - so `cargo` can write `target/`).
fn build_ws_dispatcher(config: &HideConfig, root: &str) -> Arc<ToolDispatcher> {
    let registry = Arc::new(ToolRegistry::default());
    hide_tools::register_builtin_tools_with(
        &registry,
        hide_tools::ShellConfig {
            workspace_root: Some(root.to_string()),
            disable_sandbox: true,
            ..Default::default()
        },
    );
    Arc::new(hide_backend::tools::build_default_tool_dispatcher(
        config, registry,
    ))
}

fn tool_names(config: &HideConfig, root: &str) -> Vec<String> {
    let registry = ToolRegistry::default();
    hide_tools::register_builtin_tools_with(
        &registry,
        hide_tools::ShellConfig {
            workspace_root: Some(root.to_string()),
            disable_sandbox: true,
            ..Default::default()
        },
    );
    let _ = config;
    let mut names: Vec<String> = registry.specs().into_iter().map(|s| s.name).collect();
    names.sort();
    names
}

// ===========================================================================
// Scripted model-decision seam #1: the InferenceClient (what a model step emits).
//
// Grammar: each `generate` returns the completion at its call index (last entry is
// the fallback for any further call). The completion drives model steps in the
// EXACT grammar the kernel driver parses - Hermes/Qwen `<tool_call>{json}</tool_call>`
// blocks (crates/hide-kernel/src/tools/parse.rs), which the driver auto-dispatches
// ONLY for the read-only allowlist (fs.read/search.text) - proving read+search.
// ===========================================================================

struct ScriptedInferenceClient {
    responses: Vec<String>,
    calls: Mutex<usize>,
}

impl ScriptedInferenceClient {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            calls: Mutex::new(0),
        }
    }
    fn call_count(&self) -> usize {
        *self.calls.lock()
    }
}

/// A read+search investigation turn, in the driver's tool-call grammar.
fn investigate_turn(lib_path: &str, root: &str) -> String {
    let read = json!({ "name": "fs.read", "arguments": { "path": lib_path } });
    let search = json!({ "name": "search.text", "arguments": { "pattern": "largest", "root": root } });
    format!(
        "Investigating the defect.\n<tool_call>{read}</tool_call>\n<tool_call>{search}</tool_call>"
    )
}

impl InferenceClient for ScriptedInferenceClient {
    fn generate<'a>(
        &'a self,
        _request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        let idx = {
            let mut c = self.calls.lock();
            let i = *c;
            *c += 1;
            i
        };
        let text = self
            .responses
            .get(idx)
            .or_else(|| self.responses.last())
            .cloned()
            .unwrap_or_default();
        Box::pin(async move {
            sink(StreamChunk::Token {
                token_id: None,
                text,
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

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        let owned = text.to_string();
        Box::pin(async move { Ok(deterministic_embedding(&owned, 8)) })
    }
}

// ===========================================================================
// Scripted model-decision seam #2: the Planner (which steps to take).
//
// Grammar: `synthesize` returns a DAG-as-data plan keyed by its call index.
//   * call 0 (initial plan)   -> Plan A: investigate (model reads/searches) ->
//     edit math.rs (typecheck oracle) -> WRONG edit of lib.rs (test oracle, fails).
//   * call >=1 (full replan)   -> Plan B: corrected edit of lib.rs (test oracle, passes).
// Each step DECLARES its acceptance oracles up front (K1). The edit steps carry a
// real `tool_hint`/`tool_args`, so `Act` dispatches the REAL hide-tools edit tool.
// ===========================================================================

struct ScriptedPlanner {
    lib_path: String,
    math_path: String,
    calls: Mutex<usize>,
}

impl ScriptedPlanner {
    fn new(lib_path: String, math_path: String) -> Self {
        Self {
            lib_path,
            math_path,
            calls: Mutex::new(0),
        }
    }
    fn synth_count(&self) -> usize {
        *self.calls.lock()
    }
}

impl Planner for ScriptedPlanner {
    fn synthesize<'a>(&'a self, objective: &'a str) -> BoxFuture<'a, Result<Plan>> {
        let n = {
            let mut c = self.calls.lock();
            let i = *c;
            *c += 1;
            i
        };
        let objective = objective.to_string();
        let lib = self.lib_path.clone();
        let math = self.math_path.clone();
        Box::pin(async move {
            if n == 0 {
                Ok(plan_a(&objective, &lib, &math))
            } else {
                Ok(plan_b(&objective, &lib))
            }
        })
    }
}

fn plan_a(objective: &str, lib: &str, math: &str) -> Plan {
    let mut investigate = PlanStep::new(
        "Read the failing test and locate the defect",
        StepKind::Investigate,
        Acceptance::predicate("relevant files and the defect identified"),
    );
    investigate.rationale = "read src/lib.rs and search for `largest`".to_string();

    let mut edit_math = PlanStep::new(
        "Add a helper to math.rs (second file, multi-file change)",
        StepKind::Edit,
        Acceptance::with_oracles("workspace type-checks", vec!["typecheck".to_string()]),
    );
    edit_math.dependencies = vec![investigate.id.clone()];
    edit_math.tool_hint = Some("edit.search_replace".to_string());
    edit_math.tool_args = Some(json!({
        "path": math,
        "edits": [{
            "search": "{ x * 2 }",
            "replace": "{ x * 2 }\npub fn triple(x: i32) -> i32 { x * 3 }"
        }]
    }));

    let mut edit_lib = PlanStep::new(
        "Fix largest() in lib.rs (first attempt)",
        StepKind::Edit,
        Acceptance::with_oracles("the failing test passes", vec!["test".to_string()]),
    );
    edit_lib.dependencies = vec![edit_math.id.clone()];
    edit_lib.tool_hint = Some("edit.write_file".to_string());
    edit_lib.tool_args = Some(json!({ "path": lib, "content": LIB_WRONG_FIX }));

    Plan {
        id: PlanId::new(),
        title: "scripted plan A (first attempt)".to_string(),
        objective: objective.to_string(),
        steps: vec![investigate, edit_math, edit_lib],
        status: PlanStatus::Active,
        budget: Default::default(),
    }
}

fn plan_b(objective: &str, lib: &str) -> Plan {
    let mut edit = PlanStep::new(
        "Correct largest() to return the maximum element",
        StepKind::Edit,
        Acceptance::with_oracles("the failing test passes", vec!["test".to_string()]),
    );
    edit.rationale = "the last-element fix still failed the test; return the max".to_string();
    edit.tool_hint = Some("edit.write_file".to_string());
    edit.tool_args = Some(json!({ "path": lib, "content": LIB_CORRECT_FIX }));

    Plan {
        id: PlanId::new(),
        title: "scripted plan B (corrected)".to_string(),
        objective: objective.to_string(),
        steps: vec![edit],
        status: PlanStatus::Active,
        budget: Default::default(),
    }
}

// ===========================================================================
// Drive the REAL kernel the host builds, over REAL services + tools + oracles.
// ===========================================================================

struct FlowOutcome {
    state: AgentState,
    run_id: RunId,
    generate_calls: usize,
    synth_calls: usize,
    context_used_tokens: usize,
    context_retained: usize,
    folded_objective: String,
}

#[allow(clippy::too_many_arguments)]
async fn run_scripted_flow(
    event_log: DynEventLog,
    role_registry: Arc<RoleRegistry>,
    code_index: Arc<InMemoryCodeIndex>,
    dispatcher: Arc<ToolDispatcher>,
    fixture: &Path,
    session: SessionId,
    steer: Option<&str>,
) -> FlowOutcome {
    let root = fixture.to_string_lossy().to_string();
    let lib = fixture.join("src/lib.rs").to_string_lossy().to_string();
    let math = fixture.join("src/math.rs").to_string_lossy().to_string();

    // (Behavior 4) Compile a REAL ContextPack from the code index and fold it into
    // the run objective - the same recipe the host's `run_turn_kernel` uses.
    let base_objective =
        "largest should return the maximum element so its failing test passes".to_string();
    let role = role_registry
        .by_purpose(RolePurpose::HeroCoder)
        .into_iter()
        .next()
        .or_else(|| role_registry.all().into_iter().next())
        .expect("at least one model role is registered");
    let max_input = role.model.context_tokens.max(4096);
    let mut compiler = ContextCompiler::new();
    compiler.add_source(CodeIndexContextSource::new(
        code_index.clone() as Arc<dyn CodeIndex>,
        16,
    ));
    let compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(max_input),
            model: role.model.clone(),
            task: base_objective.clone(),
        })
        .await
        .expect("context compile");
    let context_used_tokens = compiled.manifest.used_tokens;
    let context_retained = compiled.manifest.retained.len();
    event_log
        .append(NewEvent::system(
            session.clone(),
            "context.compiled",
            json!({
                "used_tokens": context_used_tokens,
                "retained": context_retained,
                "path": "first_receipt",
            }),
        ))
        .await
        .unwrap();
    let objective = if compiled.prompt.trim().is_empty() {
        base_objective
    } else {
        format!("{}\n\n{}", compiled.prompt, base_objective)
    };

    let inference = Arc::new(ScriptedInferenceClient::new(vec![investigate_turn(&lib, &root)]));
    let planner = Arc::new(ScriptedPlanner::new(lib.clone(), math.clone()));
    let runtime = Arc::new(KernelRuntimeClient::new(
        Arc::new(SimpleRouter::new(role_registry.clone())),
        inference.clone() as Arc<dyn InferenceClient>,
    ));
    let grounding = Arc::new(Grounding::new(code_index.clone() as Arc<dyn CodeIndex>));

    // Build the SAME kernel the host builds (mirrors `BackendHost::build_turn_kernel`):
    // runtime + grounding + real dispatcher + the standard deterministic oracles - 
    // but with the scripted planner + inference at the model boundary. FullAuto so
    // effectful edits run headless (the SuggestOnly approval path is covered by the
    // existing host tests); the steering intervention is exercised below.
    let kernel = AgentKernel::builder(event_log.clone())
        .workspace_root(root.clone())
        .autonomy(Autonomy::FullAuto)
        .grounding(grounding)
        .planner(planner.clone() as Arc<dyn Planner>)
        .runtime(runtime)
        .dispatcher(dispatcher.clone())
        .with_standard_oracles(dispatcher)
        .build();

    let mut state = kernel.start_run(session, objective.clone()).await.unwrap();
    // Move straight to replan on the first oracle failure (the corrected plan is a
    // full-replan resynthesis - the doctrinal failure-driven correction).
    state.budget.max_repairs = 0;

    // Drive the FSM, injecting one real steering interrupt the first time the run
    // enters a Replan boundary (behavior 11: accept a steering message).
    let mut steered = false;
    for _ in 0..200 {
        if state.phase.is_terminal() {
            break;
        }
        if !steered && steer.is_some() && state.phase == Phase::Replan {
            kernel.interrupt(Interrupt::Steer {
                instruction: steer.unwrap().to_string(),
            });
            steered = true;
        }
        kernel.step(&mut state).await.unwrap();
    }

    FlowOutcome {
        run_id: state.run_id.clone(),
        generate_calls: inference.call_count(),
        synth_calls: planner.synth_count(),
        context_used_tokens,
        context_retained,
        folded_objective: objective,
        state,
    }
}

/// Seed the code index with the fixture sources so the context compile has real
/// spans to retain (behavior 4).
fn seed_index(index: &InMemoryCodeIndex, root: &Path) {
    index.add_text_file(
        "src/lib.rs",
        std::fs::read_to_string(root.join("src/lib.rs")).unwrap(),
        None,
    );
    index.add_text_file(
        "src/math.rs",
        std::fs::read_to_string(root.join("src/math.rs")).unwrap(),
        None,
    );
    index.add_text_file(
        "tests/behavior.rs",
        std::fs::read_to_string(root.join("tests/behavior.rs")).unwrap(),
        None,
    );
    // A task-anchor whose line contains the objective plus a token (`ZZFIXTURE`)
    // that exists ONLY in this indexed span - so retrieving it into the run
    // objective proves the compiled ContextPack (not the raw prompt) was folded in
    // (the sanctioned `ZZONLYINFILE` pattern from the host's kernel-turn test).
    index.add_text_file(
        "docs/TASK.md",
        "largest should return the maximum element so its failing test passes \
         (fixture task anchor ZZFIXTURE)",
        None,
    );
}

/// A stable, id/timestamp-free signature token per event (for replay-equivalence).
fn normalize(e: &Event) -> String {
    let p = &e.payload;
    let s = |k: &str| p.get(k).and_then(|v| v.as_str()).unwrap_or("?").to_string();
    match e.kind.as_str() {
        "agent.phase" => format!("phase:{}", s("phase")),
        "verify.result" => format!("verify:{}:{}", s("oracle"), s("status")),
        "plan.created" => format!("plan.created:{}", s("action")),
        "plan.replanned" => format!("plan.replanned:{}", s("mode")),
        k => k.to_string(),
    }
}

async fn event_signature(log: &DynEventLog, session: &SessionId) -> Vec<String> {
    let events = log.scan(Some(session.clone()), None, None).await.unwrap();
    events.iter().map(normalize).collect()
}

/// Run one full scripted flow on a fresh in-memory stack (no disk). Used to prove
/// replay-equivalence without the durable-host machinery.
async fn run_once_in_memory() -> Vec<String> {
    let repo = make_buggy_repo();
    let config = permissive_config(&repo);
    let event_log: DynEventLog = Arc::new(hide_core::event::InMemoryEventLog::new());
    let role_registry = Arc::new(RoleRegistry::with_default_local_roles());
    let code_index = Arc::new(InMemoryCodeIndex::default());
    seed_index(&code_index, &repo);
    let dispatcher = build_ws_dispatcher(&config, &repo.to_string_lossy());
    let session = SessionId::new();

    let outcome = run_scripted_flow(
        event_log.clone(),
        role_registry,
        code_index,
        dispatcher,
        &repo,
        session.clone(),
        None,
    )
    .await;
    assert_eq!(
        outcome.state.phase,
        Phase::Done,
        "the in-memory scripted flow must reach Done"
    );
    let sig = event_signature(&event_log, &session).await;
    let _ = std::fs::remove_dir_all(&repo);
    sig
}

// ===========================================================================
// The receipt (hide.receipt.v1) - emitted from the REAL run's durable event log.
// ===========================================================================

fn read_git(dir: &Path, args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

fn short_hash(s: &str) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// Build the receipt from real run data + the durable event stream.
#[allow(clippy::too_many_arguments)]
fn build_receipt(
    fixture: &Path,
    events: &[Event],
    tools: &[String],
    outcome: &FlowOutcome,
    before_exit: i32,
    after_exit: i32,
    interventions: usize,
    wall_ms: u128,
) -> Value {
    // Actions: the tool dispatches + model tool-calls the driver recorded.
    let mut actions: Vec<Value> = Vec::new();
    for e in events {
        if e.kind != "agent.observation" {
            continue;
        }
        if let Some(tool) = e.payload.get("tool").and_then(|v| v.as_str()) {
            actions.push(json!({
                "kind": "tool",
                "tool": tool,
                "effect": "write_or_exec",
                "ok": e.payload.get("ok").and_then(|v| v.as_bool()).unwrap_or(false),
            }));
        }
        if let Some(calls) = e.payload.get("tool_calls").and_then(|v| v.as_array()) {
            for c in calls {
                actions.push(json!({
                    "kind": "model_tool_call",
                    "tool": c.get("tool").cloned().unwrap_or(Value::Null),
                    "effect": "read_only",
                    "ok": c.get("dispatched").and_then(|v| v.as_bool()).unwrap_or(false),
                }));
            }
        }
        if actions.len() >= 25 {
            break;
        }
    }
    let effects_approved = actions
        .iter()
        .filter(|a| a["kind"] == "tool" && a["ok"] == true)
        .count();
    // Verify timing: sum the recorded oracle durations.
    let verify_ms: u64 = events
        .iter()
        .filter(|e| e.kind == "verify.result")
        .filter_map(|e| e.payload.get("duration_ms").and_then(|v| v.as_u64()))
        .sum();

    let head = read_git(fixture, &["rev-parse", "HEAD"]);
    let dirty = read_git(fixture, &["status", "--porcelain"]);
    let diff = read_git(fixture, &["diff"]);

    json!({
        "schema": "hide.receipt.v1",
        "task": "first-model-free-receipt: fix buggy largest() so its unit test passes",
        "repo_snapshot": format!("{}+dirty:{}", head, short_hash(&dirty)),
        "hardware": format!("{}/{} (test host)", std::env::consts::OS, std::env::consts::ARCH),
        "model": "scripted-driver (DEFERRED_MODEL_REQUIRED)",
        "run_id": outcome.run_id.as_str(),
        "model_free": true,
        "proves": "orchestration/durability/tools/verification - NOT intelligence",
        "effort_policy": "Interactive",
        "tools_enabled": tools,
        "context": {
            "pack_id": format!("ctx_{}", short_hash(&outcome.folded_objective)),
            "used_tokens": outcome.context_used_tokens,
            "retained_spans": outcome.context_retained,
            "sources": ["repo://src/lib.rs", "repo://src/math.rs", "repo://tests/behavior.rs"],
        },
        "actions": actions,
        "effects_approved": effects_approved,
        "autonomy": "full_auto (headless); effects auto-authorized, not human-approved",
        "timings_ms": { "wall": wall_ms, "model": 0, "tool": null, "verify": verify_ms },
        "compute": { "tokens_out": outcome.generate_calls, "prefill_reused": false, "gpu_ms": 0 },
        "plan": { "synth_calls": outcome.synth_calls, "replans": outcome.state.replan_count },
        "patch": format!("diffhash:{}", short_hash(&diff)),
        "tests": {
            "command": "cargo test",
            "before": if before_exit == 0 { "green" } else { "red" },
            "after": if after_exit == 0 { "green" } else { "red" },
        },
        "regression": "no_worse",
        "interventions": interventions,
        "failure_modes": [
            "scripted first-attempt patch (last-element) failed the real cargo test oracle",
            "recovered via localized-then-full replan + a corrected patch"
        ],
        "baseline": "thin-mode one-shot comparison NOT run (follow-up increment)",
        "behaviors_deferred": ["17 fork", "18 search-transcript", "19 reviewer-compare", "thin-mode baseline"],
        "accepted": after_exit == 0,
        "acceptance_note": "acceptance = the real deterministic cargo-test oracle is green (no human in a headless run)",
    })
}

// ===========================================================================
// TEST 1 - the flagship: the REAL app path, model-free, to a green receipt.
// ===========================================================================

#[tokio::test]
async fn first_model_free_implementation_receipt() {
    let started = std::time::Instant::now();
    let repo = make_buggy_repo();
    let config = permissive_config(&repo);
    let root = repo.to_string_lossy().to_string();

    // (Behavior 1) Open + trust the repository: open the workspace host and record
    // a durable trust marker. (The dedicated trust-gate subsystem is a future
    // increment; opening the workspace is the current trust boundary.)
    let services = BackendServices::open(config.clone()).unwrap();
    let host = BackendHost::from_services(services).unwrap();
    let session = host.services.session(); // (Behavior 2) stable session
    host.services
        .event_log
        .append(NewEvent::system(
            session.clone(),
            "repo.trusted",
            json!({ "root": root, "by": "first_receipt_test" }),
        ))
        .await
        .unwrap();

    seed_index(&host.services.code_index, &repo);
    let dispatcher = build_ws_dispatcher(&config, &root);

    // Capture the REAL "before" state: the failing test is RED.
    let before = dispatcher
        .dispatch(ToolCall::new(
            "test.run",
            json!({ "cwd": root, "argv": ["cargo", "test"] }),
        ))
        .await
        .unwrap();
    let before_exit = before.exit_code.unwrap_or(-1);
    assert!(
        before.ok,
        "test.run must be ok:true even when tests fail (EXEC_NONZERO is data)"
    );
    assert_ne!(before_exit, 0, "the fixture test must start RED");

    // Drive the REAL kernel loop, model-free, with a real steering intervention.
    let outcome = run_scripted_flow(
        host.services.event_log.clone(),
        host.services.role_registry.clone(),
        host.services.code_index.clone(),
        dispatcher.clone(),
        &repo,
        session.clone(),
        Some("prefer a clear, idiomatic maximum-scan"),
    )
    .await;

    assert_eq!(
        outcome.state.phase,
        Phase::Done,
        "the run must reach Done (phase: {:?})",
        outcome.state.phase
    );
    assert!(
        outcome.synth_calls >= 2,
        "the failure must drive a full replan (a 2nd plan synthesis), got {}",
        outcome.synth_calls
    );
    assert!(
        outcome.generate_calls >= 1,
        "at least one model step must have generated (investigate)"
    );

    // (Behaviors 3,5) The plan was persisted with declared acceptance oracles.
    let events = host
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    let plan_created = events
        .iter()
        .find(|e| e.kind == "plan.created" && e.payload.get("action").and_then(|a| a.as_str()) == Some("created"))
        .expect("a plan.created event must be persisted");
    let oracles = plan_created
        .payload
        .pointer("/plan/steps")
        .and_then(|v| v.as_array())
        .map(|steps| {
            steps
                .iter()
                .filter_map(|s| s.pointer("/acceptance/oracles").and_then(|o| o.as_array()))
                .flatten()
                .filter_map(|o| o.as_str())
                .map(String::from)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    assert!(
        oracles.iter().any(|o| o == "test"),
        "the plan must declare the `test` acceptance oracle up front, got: {oracles:?}"
    );

    // (Behavior 4) The compiled context rode into the run objective, and the
    // planner's persisted objective carries it (proving the compile fed the run).
    assert!(
        outcome.folded_objective.contains("largest should return the maximum"),
        "the objective must carry the task"
    );
    let planned_objective = plan_created
        .payload
        .pointer("/plan/objective")
        .and_then(|v| v.as_str())
        .unwrap_or_default();
    assert!(
        planned_objective.contains("largest should return the maximum"),
        "the compiled objective must ride into the persisted plan"
    );
    // `ZZFIXTURE` exists ONLY in the seeded index span - its presence proves the
    // compiled ContextPack (a real retrieved span), not just the raw prompt, was
    // folded into the run objective (behavior 4, end-to-end).
    assert!(
        outcome.context_retained >= 1 && planned_objective.contains("ZZFIXTURE"),
        "the compiled ContextPack must retrieve a real index span into the objective \
         (retained={}, objective={planned_objective:?})",
        outcome.context_retained
    );

    // (Behavior 6) The model step read + searched real files (auto-dispatched).
    let dispatched_read = events.iter().any(|e| {
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
    assert!(dispatched_read, "the model step must dispatch a real fs.read");

    // (Behaviors 8,9,10,12) A REAL oracle FAIL then a REAL oracle PASS were recorded.
    let saw_fail = events.iter().any(|e| {
        e.kind == "verify.result"
            && e.payload.get("oracle").and_then(|v| v.as_str()) == Some("test")
            && e.payload.get("status").and_then(|v| v.as_str()) == Some("fail")
    });
    let saw_pass = events.iter().any(|e| {
        e.kind == "verify.result"
            && e.payload.get("oracle").and_then(|v| v.as_str()) == Some("test")
            && e.payload.get("status").and_then(|v| v.as_str()) == Some("pass")
    });
    assert!(saw_fail, "the real cargo-test oracle must have FAILED the wrong patch");
    assert!(saw_pass, "the real cargo-test oracle must have PASSED the corrected patch");
    assert!(
        events.iter().any(|e| e.kind == "plan.replanned"),
        "the kernel must have reacted to the failure with a replan"
    );

    // The corrected edit is on disk: assert the source now returns the max.
    let final_lib = std::fs::read_to_string(repo.join("src/lib.rs")).unwrap();
    assert!(
        final_lib.contains("if x > m"),
        "the corrected max-scan fix must be on disk"
    );

    // Capture the REAL "after" state: the test is GREEN (the acceptance gate).
    let after = dispatcher
        .dispatch(ToolCall::new(
            "test.run",
            json!({ "cwd": root, "argv": ["cargo", "test"] }),
        ))
        .await
        .unwrap();
    let after_exit = after.exit_code.unwrap_or(-1);
    assert_eq!(after_exit, 0, "the fixture test must end GREEN");

    // (Behaviors 14,15,16) Persist + restart + resume: drop the host, re-open the
    // SAME durable workspace, and replay the thread from disk.
    drop(host);
    let host2 = BackendHost::from_services(BackendServices::open(config.clone()).unwrap()).unwrap();
    assert_eq!(
        host2.services.session(),
        session,
        "the durable session id must survive a restart"
    );
    let replayed = host2
        .services
        .event_log
        .scan(Some(session.clone()), None, None)
        .await
        .unwrap();
    assert!(
        replayed.iter().any(|e| e.kind == "plan.created"),
        "the plan must survive the restart"
    );
    assert!(
        replayed.iter().any(|e| {
            e.kind == "verify.result"
                && e.payload.get("oracle").and_then(|v| v.as_str()) == Some("test")
                && e.payload.get("status").and_then(|v| v.as_str()) == Some("pass")
        }),
        "the green verdict must survive the restart"
    );
    // Resume the thread: rebuild its projection from the durable log.
    let projection = host2.rebuild_session_projection(session.clone()).await.unwrap();
    assert_eq!(
        projection.session_id, session,
        "the resumed projection must be for the same thread"
    );

    // (Behavior 20) Export the hide.receipt.v1 to a stable path.
    let tools = tool_names(&config, &root);
    let receipt = build_receipt(
        &repo,
        &replayed,
        &tools,
        &outcome,
        before_exit,
        after_exit,
        1, // one steering intervention
        started.elapsed().as_millis(),
    );
    let receipt_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .join("target/hide_receipts");
    std::fs::create_dir_all(&receipt_dir).unwrap();
    let receipt_path = receipt_dir.join("first_model_free_receipt.json");
    std::fs::write(
        &receipt_path,
        serde_json::to_string_pretty(&receipt).unwrap(),
    )
    .unwrap();

    // Assert the receipt exported and round-trips with the load-bearing fields.
    assert!(receipt_path.exists(), "the receipt must be exported");
    let reloaded: Value =
        serde_json::from_str(&std::fs::read_to_string(&receipt_path).unwrap()).unwrap();
    assert_eq!(reloaded["schema"], "hide.receipt.v1");
    assert_eq!(reloaded["tests"]["before"], "red");
    assert_eq!(reloaded["tests"]["after"], "green");
    assert_eq!(reloaded["accepted"], true);
    assert_eq!(reloaded["model"], "scripted-driver (DEFERRED_MODEL_REQUIRED)");
    assert!(reloaded["actions"].as_array().map(|a| !a.is_empty()).unwrap_or(false));

    eprintln!(
        "hide.receipt.v1 exported to {}",
        receipt_path.to_string_lossy()
    );

    let _ = std::fs::remove_dir_all(&repo);
}

// ===========================================================================
// TEST 2 - determinism: two runs of the scripted flow are replay-equivalent.
// ===========================================================================

#[tokio::test]
async fn scripted_flow_is_deterministic_replay_equivalent() {
    let sig1 = run_once_in_memory().await;
    let sig2 = run_once_in_memory().await;
    assert_eq!(
        sig1, sig2,
        "the scripted flow must produce an identical event-kind sequence across runs"
    );
    // Sanity: the signature reflects the real fail -> replan -> pass arc.
    assert!(
        sig1.iter().any(|s| s == "verify:test:fail"),
        "must record a real oracle failure; sig: {sig1:?}"
    );
    assert!(
        sig1.iter().any(|s| s == "verify:test:pass"),
        "must record a real oracle pass; sig: {sig1:?}"
    );
    assert!(
        sig1.iter().any(|s| s.starts_with("plan.replanned")),
        "must replan after the failure; sig: {sig1:?}"
    );
    assert!(
        sig1.iter().any(|s| s == "phase:done"),
        "must reach a terminal Done; sig: {sig1:?}"
    );
}
