//! Wave-1 integration: (A) mid-turn STEER end to end through the host path
//! (campaign Trace A) and (B) the new custom-name dispatch arms that make built
//! host methods reachable over `/v1/hide/intent`.
//!
//! All model-free: the only scripted piece is the kernel's model client (a
//! prompt-capturing [`CapturingInferenceClient`]); everything else -- the host,
//! the InterruptHub, the event log, the workspace graph -- is the real thing.

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

// --- test scaffolding --------------------------------------------------------

fn test_host() -> BackendHost {
    static N: AtomicU64 = AtomicU64::new(0);
    let uniq = N.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("hide_steer_{}_{}", now_ms(), uniq));
    BackendHost::open_workspace(&dir).unwrap()
}

/// A model client that RECORDS every prompt it is asked to generate, so a test
/// can assert the steer text was folded into the next planning step's prompt.
/// Deterministic + model-free (the scripted "model decision" the task allows).
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

/// A planner that emits two sequential non-effectful READ steps (read2 depends on
/// read1). Both are `Investigate` (a model step that soft-accepts with no oracle),
/// so the run walks read1 -> read2 -> Done and a steer delivered between them is
/// folded into read2's prompt.
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

// --- (A) Trace A: mid-turn steering end to end -------------------------------

#[tokio::test]
async fn trace_a_steer_reaches_running_kernel_and_folds_into_next_step() {
    const STEER: &str = "STOP reading, switch to the auth module instead";

    let host = test_host();
    let session = host.services.session();

    // The kernel shares the host's event log + role registry; its model client is
    // the prompt-capturing scripted client.
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

    // The host-side run id the InterruptHub is keyed on (distinct from the run's
    // own internal id), exactly as `run_turn_kernel` keys it.
    let host_run = RunId::new();
    let mut state = kernel
        .start_run(session.clone(), "investigate the codebase")
        .await
        .unwrap();

    let mut steered = false;
    let mut steer_delivered = false;
    for _ in 0..64 {
        // Forward any host-buffered interrupt into the kernel (the exact seam the
        // live turn loop uses). Capture that a Steer was actually delivered.
        if let Some(Interrupt::Steer { .. }) =
            host.interrupts().drain_into_kernel(&host_run, &kernel)
        {
            steer_delivered = true;
        }
        if state.phase.is_terminal() {
            break;
        }
        kernel.step(&mut state).await.unwrap();

        // Once the FIRST read has produced an observation (a completed read),
        // deliver the steer through the REAL host path (the shipped `redirect_run`
        // gesture). Exactly once.
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

    // (1) The InterruptHub delivered a Steer into the running kernel.
    assert!(steer_delivered, "InterruptHub forwarded a Steer to the kernel");

    // (3) The earlier completed reads are still present (both reads ran).
    assert!(
        observation_count(&host, &session).await >= 2,
        "both read observations remain in the log"
    );

    // (2) The next planning step's prompt reflects the steer text. read1's prompt
    // was captured BEFORE the steer (so it must NOT contain it); read2's prompt was
    // built AFTER the governor folded the steer into `state.steer`, so it MUST.
    let prompts = capturing.prompts.lock().clone();
    assert!(
        prompts.len() >= 2,
        "both read steps generated a prompt: {prompts:?}"
    );
    assert!(
        !prompts[0].contains(STEER),
        "the first read ran before the steer, so its prompt is un-steered"
    );
    assert!(
        prompts.iter().skip(1).any(|p| p.contains(STEER)),
        "a later read's prompt carries the steer directive: {prompts:?}"
    );

    // (4) A durable steer event was persisted, and (5) it AGREES with the projection
    // (the same instruction the kernel folded into the next prompt).
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
        steer_event.payload.get("instruction").and_then(|v| v.as_str()),
        Some(STEER),
        "the durable steer event carries the directive"
    );
    assert_eq!(
        steer_event.run_id.as_ref().map(|r| r.as_str()),
        Some(host_run.as_str()),
        "the steer event is tagged with the run it steered"
    );

    assert_eq!(state.phase, Phase::Done, "the steered run still completed");
}

// --- (B) custom-name dispatch arms ------------------------------------------

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

    // The dispatch ran inline: memory_list shows the durable record.
    let records = host.memory_list(&scope);
    assert_eq!(records.len(), 1, "one memory record was persisted");
    assert_eq!(records[0].claim, "the turn loop is a single flat FSM");
    // memory_get by the minted id round-trips.
    assert!(host.memory_get(&records[0].memory_id).is_some());
}

#[tokio::test]
async fn goal_evaluate_intent_returns_a_deterministic_verdict() {
    use hide_backend::GoalStatus;
    use hide_core::event::NewEvent;
    use hide_kernel::verify::oracle::{OracleClass, Verdict};

    let host = test_host();
    let session = host.services.session();

    // A goal whose acceptance names the "tests" oracle, plus a PASSING verify.result.
    host.goal_set(session.clone(), "tests_pass", vec!["tests".to_string()])
        .unwrap();
    host.services
        .event_log
        .append(NewEvent::system(
            session.clone(),
            "verify.result",
            serde_json::to_value(&Verdict::pass("tests", OracleClass::Deterministic, "all green"))
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

    // The deterministic evaluation ran against verify.result and advanced the goal
    // to Met (the verdict), proving the intent routed to goal_evaluate.
    assert_eq!(
        host.goal_get(&session).unwrap().status,
        GoalStatus::Met,
        "goal_evaluate advanced the durable status to Met against the passing verdict"
    );
}

/// `workspace_set_repo_trust` is declared `ApprovalPolicy::Ask` by the command authority, so the
/// intent alone must NOT flip trust: the effect is parked at the security gate and only an
/// `approve_gate` for the returned gate id releases it.
#[tokio::test]
async fn workspace_set_repo_trust_intent_is_held_until_approved() {
    use hide_backend::services::{RepoNode, TrustState};

    let host = test_host();
    host.workspace_add_repo(RepoNode::new("vendor", "/tmp/vendor"))
        .unwrap();
    assert_eq!(
        host.workspace_repo("vendor").unwrap().trust,
        TrustState::Untrusted,
        "a fresh repo is untrusted"
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
        TrustState::Untrusted,
        "an Ask command must not take effect before it is approved"
    );

    host.handle_intent(Intent::Custom {
        name: "approve_gate".to_string(),
        payload: serde_json::json!({ "gate": gate }),
    })
    .await
    .unwrap();

    assert_eq!(
        host.workspace_repo("vendor").unwrap().trust,
        TrustState::Trusted,
        "approval released the held effect and flipped the repo trust"
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

    // The durable environment.switch event is on the session log.
    let switches = host.environment_switches(&session).await.unwrap();
    assert_eq!(switches.len(), 1, "one environment switch was recorded");
    assert_eq!(switches[0].new_env, "container:node20");

    // The session's current-environment pointer advanced.
    assert_eq!(
        WorkspaceStore::current_env(&host.services.key_value_store, &session).as_deref(),
        Some("container:node20"),
        "current_env advanced to the switched environment"
    );
}
