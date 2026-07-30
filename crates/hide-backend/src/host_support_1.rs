use crate::approval::{ApprovalDecision, ApprovalHub};
use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::interrupt::InterruptHub;
use crate::memory::{
    MemoryDraft, MemoryLedger, MemoryRecord, MemoryRevalidation, MemoryScope, MemoryStatus,
    PrivacyClass, RevalidateTarget,
};
use crate::policy::{
    derive_policy_decision, tool_declared_effects, PolicyDecision, PolicyDecisionRecord,
};
use crate::process::{ProcessState, ProcessSupervisor, StartSpec};
use crate::replay::BackendReplayService;
use crate::rewind::{self, CheckpointCoverage, FileChange, ForkPoint, RewindTarget, StateRef};
use crate::security::SecurityServices;
use crate::initialize::{ClientCapabilities, ClientInfo, ConnectionRegistry, InitializeResponse};
use crate::live_thread::LiveThread;
use crate::services::{
    BackendCapabilities, BackendServices, Budget, CheckpointRecord, CheckpointStore,
    EnvironmentNode, EnvironmentSwitch, GoalOutcome, GoalRecord, GoalStatus, GoalStore, GoalVerdict,
    JobRecord, JobStatus, JobStore, RepoNode, SharedBackend, Trigger, TriggerEvent, TrustState,
    WorkspaceEdge, WorkspaceEdgeKind, WorkspaceGraph, WorkspaceStore,
};
use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
use crate::surfaces::SurfaceGraphService;
use crate::tools::{build_default_tool_dispatcher, build_default_tool_registry};
use crate::ui_bus::UiEventBus;
use hide_core::api::{Intent, IntentAck, UiEvent, UiEventKind};
use hide_core::event::{Event, NewEvent, ToolCallEvent, ToolResultEvent};
use hide_core::ids::{EventId, RunId, SessionId, StepId};
use hide_core::observability::{HealthCheck, HealthReport, HealthStatus};
use hide_core::runtime::{ModelRole, RuntimeSupervisorState};
use hide_core::tool::{ToolCall, ToolDispatcher, ToolRegistry, ToolResult, ToolSpec, ToolStatus};
use hide_core::Result;
use hide_fleet::manager::KernelRunLauncher;
use hide_fleet::{
    AgentJob, ConcurrencyClass, FleetConfig, FleetGovernor, FleetManager, OsResourceProbe,
    PriorityClass,
};
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::{AgentState, ApprovalRequest, Phase};
use hide_kernel::session::SessionProjection;
use hide_kernel::{AgentKernel, Grounding};
// Bible Book IX sec 28-29 / sec 78.1 #6: the deterministic verification plane.
// The colliding names (`Verdict`, `VerificationInput`, `Oracle`) are qualified
// as `hide_kernel::verify_plane::*` at their (few) use sites so the function-local
// `hide_kernel::verify::oracle::*` imports in the goal path and the tests keep
// their meaning; only the non-colliding types are imported here.
use hide_kernel::verify_plane::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use super::*;


/// Increment 2: drive an accepted `SubmitTurn` through the REAL agent kernel
/// loop (fixing defect S1 - the host built [`AgentKernel::new`] = StubPlanner and
/// never used the wired [`AgentKernel::builder`]). This is the spawnable twin of
/// [`generate_submit_turn`]: it takes owned clones (so it is `'static`) plus the
/// pre-built `kernel` (from [`BackendHost::build_turn_kernel`]).
///
/// It (1) compiles a REAL `ContextPack` (bible §4.2) - the same recipe as
/// [`run_turn_core`] - (2) folds that compiled context into the run objective
/// (`objective = "{compiled}\n\n{prompt}"`), (3) calls
/// [`AgentKernel::start_run`] and loops [`AgentKernel::step`] until the phase is
/// terminal (bounded by `max_steps`), forwarding any `Cancel`/`Pause` the host
/// buffered for this run into `kernel.interrupt` each iteration, and (4)
/// publishes the post-turn `context_manifest` + a `turn` completion patch (Spine
/// A parity with the single-shot path). The driver
/// ([`hide_kernel::machine::driver`]) already persists `plan.created` /
/// `agent.action` / `agent.observation` / `verify.result` to the event log.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_turn_kernel(
    kernel: AgentKernel,
    event_log: hide_core::persistence::DynEventLog,
    key_value_store: hide_core::persistence::DynKeyValueStore,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
    classed_memory: hawking_context::DynClassedMemory,
    ui_bus: Arc<UiEventBus>,
    interrupts: Arc<InterruptHub>,
    approvals: Arc<ApprovalHub>,
    run_id: RunId,
    session_id: SessionId,
    base_url: String,
    prompt: String,
    max_steps: usize,
    repo_instructions: Arc<crate::compat_instructions::ResolvedInstructions>,
) -> Result<AgentState> {
    use crate::connectors::choose_context_role;
    use crate::model_provider::HttpModelProvider;
    use hawking_context::compiler::CompileInput;
    use hawking_context::profiles::ContextProfile;
    use hawking_context::sources::{ClassedMemoryContextSource, CodeIndexContextSource};
    use hawking_context::{
        ClassBudgets, ContextCompiler, InMemoryMemoryStore, MemoryKind,
    };
    use hide_core::types::Provenance;

    // (Spine A) Snapshot the live ceiling ONCE (best-effort; `None` when the serve
    // is down or predates the context route) so the post-turn manifest is real.
    let live_snap = HttpModelProvider::new(base_url.clone())
        .get_context_info()
        .await
        .map(|i| {
            (
                i.recurrent_state_bytes,
                i.ctx_len_native,
                i.ctx_len_effective.or(i.ctx_len_native).unwrap_or(0),
            )
        });

    // Working memory (turn-local): RAII guard clears on every exit path
    // (Ok / early Err / panic), not only the success return.
    let turn_id = run_id.as_str().to_string();
    let _working_guard = crate::classed_writers::WorkingTurnGuard::begin(
        classed_memory.clone(),
        turn_id.clone(),
        session_id.as_str(),
        Some(run_id.as_str()),
        &prompt,
    );

    // --- (S3) Compile a REAL ContextPack - same recipe as `run_turn_core`. ---
    // §7.3 honesty: prefer live-measured native; never treat effective as native.
    let role = choose_context_role(&role_registry, None)?;
    let config_native = role.model.context_tokens.max(4096);
    let live_native = live_snap.and_then(|(_, n, _)| n).filter(|n| *n > 0);
    let live_effective = live_snap.map(|(_, _, c)| c).filter(|c| *c > 0);
    let capability = declare_turn_capability(
        config_native,
        live_native,
        live_effective,
        None,
        false,
    );
    let max_input = capability.pack_budget_tokens(false).max(256);
    let mut model = role.model.clone();
    model.context_tokens = max_input;
    // Tokenizer-true packing when HIDE_TOKENIZER / weights-adjacent tokenizer.json
    // is available; otherwise heuristic and seal reports used_estimated.
    let counter = hawking_context::TokenCounter::discover_from_env()
        .unwrap_or_else(hawking_context::TokenCounter::heuristic);
    let mut compiler = ContextCompiler::new().with_counter(counter);
    compiler.add_source(CodeIndexContextSource::new(code_index, 16));
    // Six memory classes: independent per-class budgets (not one kind filter).
    let class_budgets = ClassBudgets::from_total((max_input / 8).max(64));
    compiler.add_source(
        ClassedMemoryContextSource::new(classed_memory.clone(), class_budgets)
            .with_session(session_id.as_str())
            .with_turn(turn_id.clone()),
    );
    // Bible sec 20 / sec 78.1 #11: fold the repo's resolved Claude Code migration
    // instructions into the compiled context as a pinned instruction source
    // (read-last-wins precedence). No-op for an un-migrated repo (resolves empty).
    if !repo_instructions.is_empty() {
        compiler.add_source(repo_instructions.as_source());
    }
    let mut compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(max_input),
            model,
            task: prompt.clone(),
        })
        .await?;
    let pre_live = live_snap.map(|(state_bytes, native, ceiling)| {
        build_live_manifest(state_bytes, native, ceiling, compiled.manifest.used_tokens)
    });
    seal_compiled_manifest(
        &mut compiled.manifest,
        capability,
        pre_live.as_ref(),
        compiled.tokens_estimated,
    );
    // Surface per-class memory budgets on the context meter.
    if let (Some(meter), Some(ret)) = (
        compiled.manifest.meter.as_mut(),
        classed_memory.last_retrieval(),
    ) {
        meter.explanations.extend(ret.budget_explanations());
    }
    // Spine B (best-effort): accrue the Project Brain; a brain write never fails a turn.
    let brain = InMemoryMemoryStore::record(
        MemoryKind::Project,
        format!(
            "task: {prompt}\nkernel turn: retained {} spans, {} tokens used",
            compiled.manifest.retained.len(),
            compiled.manifest.used_tokens
        ),
        Provenance::trusted("submit_turn.kernel"),
    );
    let _ = memory.upsert(brain).await;

    // (F3) Rebuild REAL conversation history from the durable event log (the same
    // recipe as `run_turn_core`) and ensure the current user prompt is the final
    // user message. The live path logs `user.intent.submit_turn` before spawning,
    // so the current prompt is usually already present (we do NOT duplicate it);
    // headless callers pass a fresh prompt we append here. This threads prior
    // turns in so the kernel plans + acts with real multi-turn continuity.
    let mut messages = rebuild_history(&event_log, &session_id).await?;
    if messages
        .last()
        .map(|m| m.role != "user" || m.content != prompt)
        .unwrap_or(true)
    {
        messages.push(hide_core::runtime::InferenceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });
    }
    let history_block = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    // (2) Fold the compiled context + rendered history into the run objective so
    // the planner + every step (and the durable `plan.created` event) are grounded
    // in real context AND prior turns.
    let objective = if compiled.prompt.trim().is_empty() {
        history_block
    } else {
        format!("{}\n\n{}", compiled.prompt, history_block)
    };
    let used_est = objective.len() / 4;

    // Durable marker (parity with the single-shot path): compile stats +
    // honest capability / rot / meter. Its seq keys the published UiEvents.
    let marker = event_log
        .append(NewEvent::system(
            session_id.clone(),
            "context.compiled",
            context_compiled_payload(
                &compiled.manifest,
                None,
                "kernel",
                Some(run_id.as_str()),
            ),
        ))
        .await?;
    let seq = marker.seq;

    // Context receipt: which repo instruction files folded into this turn (parity
    // with the single-shot path). Logged only when the repo carried them.
    if !repo_instructions.is_empty() {
        event_log
            .append(NewEvent::system(
                session_id.clone(),
                "context.instructions",
                repo_instructions.receipt_json(),
            ))
            .await?;
    }

    // (Spine A) Publish a partial live manifest from the pre-run snapshot so the
    // Context Stack reflects the ceiling before the loop advances.
    if let Some((state_bytes, native, ceiling)) = live_snap {
        let live = build_live_manifest(state_bytes, native, ceiling, used_est);
        if let Ok(mut lj) = serde_json::to_value(&live) {
            if let Some(o) = lj.as_object_mut() {
                o.insert("used_tokens_estimate".to_string(), json!(used_est));
                o.insert("estimated".to_string(), json!(true));
                o.insert("partial".to_string(), json!(true));
            }
            ui_bus.publish(UiEvent {
                seq,
                session_id: Some(session_id.clone()),
                kind: UiEventKind::ProjectionPatch {
                    projection: "context_manifest".to_string(),
                    patch: json!({ "live": lj }),
                },
            });
        }
    }

    // (3) Drive the FSM to a terminal phase, forwarding host interrupts each step
    // and completing the effect+approval round-trip (§78.1 #7) when the driver
    // pauses on an effectful step under bounded autonomy.
    let mut state = kernel.start_run(session_id.clone(), objective).await?;
    // The step currently announced as awaiting approval (so the request surfaces
    // exactly once per pause, not on every idempotent Paused spin).
    let mut announced_approval: Option<StepId> = None;
    // Plan-domain emitter (Stage 1): publish the durable `plan` projection as the
    // plan evolves. `last_plan` de-dupes: we emit on the first synthesis and on
    // any change (step-status advance, replan), not on every idempotent spin.
    let plan_autonomy = turn_kernel_autonomy();
    let mut last_plan: Option<hide_kernel::plan::schema::Plan> = None;
    for _ in 0..max_steps {
        // Forward any Cancel/Pause the host buffered for this run into the kernel
        // (consumed by the Governor on the next transition, K8).
        interrupts.drain_into_kernel(&run_id, &kernel);
        if state.phase.is_terminal() {
            break;
        }
        kernel.step(&mut state).await?;

        // Publish + persist the plan projection whenever the live plan changes.
        // `store_and_publish` writes the durable KV record AND pushes the `plan`
        // ProjectionPatch on Wire-B, so the PlanCard and the mutation handlers
        // share one source of truth.
        if state.plan != last_plan {
            if let Some(plan) = &state.plan {
                let record = crate::plan_domain::PlanRecord::from_kernel(plan, plan_autonomy);
                if let Err(e) = crate::plan_domain::store_and_publish(
                    &key_value_store,
                    &ui_bus,
                    &session_id,
                    seq,
                    &record,
                ) {
                    ui_bus.publish(UiEvent {
                        seq,
                        session_id: Some(session_id.clone()),
                        kind: UiEventKind::Error {
                            code: "plan_projection".to_string(),
                            message: e.to_string(),
                        },
                    });
                }
            }
            last_plan = state.plan.clone();
        }

        // Effect+approval round-trip: while the driver holds the run at
        // `Phase::Paused` with a `pending_approval`, surface the request once and
        // deliver any host decision from the `ApprovalHub`. We NEVER auto-approve:
        // absent a decision the run stays paused (and eventually leaves the loop
        // when `max_steps` is exhausted). A tight no-yield spin would burn that
        // budget in <1ms and make a concurrent `approve_effect`/`deny_effect`
        // impossible to land; yield while parking so the live intent path can
        // deposit into the hub.
        if state.phase == Phase::Paused {
            if let Some(request) = state.pending_approval.clone() {
                if announced_approval.as_ref() != Some(&request.step_id) {
                    announce_approval_request(
                        &event_log, &ui_bus, &session_id, &run_id, &request,
                    )
                    .await?;
                    announced_approval = Some(request.step_id.clone());
                }
                // Drain a decision for this run. A decision that names a different
                // step than the one pending is stale and ignored (re-buffered
                // decisions do not resurface, matching InterruptHub semantics).
                //
                // Asymmetry is deliberate (W5): Deny with no step_id is fail-safe
                // and resolves whatever is pending; Approve with no step_id never
                // matches (defence in depth against a buffered blanket approve).
                if let Some((step_id, decision)) = approvals.take(&run_id) {
                    let targets_pending = match (decision, step_id.as_ref()) {
                        (ApprovalDecision::Deny, None) => true,
                        (ApprovalDecision::Approve, None) => false,
                        (_, Some(s)) => s == &request.step_id,
                    };
                    if targets_pending {
                        match decision {
                            ApprovalDecision::Approve => {
                                state.approve_pending_effect();
                            }
                            ApprovalDecision::Deny => {
                                state.deny_pending_effect();
                            }
                        }
                        record_approval_resolved(
                            &event_log, &session_id, &run_id, &request, decision,
                        )
                        .await?;
                        // A later effectful step re-announces (fresh step id).
                        announced_approval = None;
                    } else {
                        // Wrong step: keep parking (yield so a corrected intent
                        // can arrive).
                        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                    }
                } else {
                    // Still waiting on the human/host: yield the runtime so a
                    // concurrent handle_intent can deposit into the hub.
                    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
                }
            }
        }
    }

    // (F2) Surface the turn's VISIBLE assistant answer on Wire-B - the gap that
    // kept this path opt-in (a client saw driver telemetry but no answer). The
    // kernel streams model output into `agent.observation` payloads rather than the
    // ui_bus, so derive the answer post-hoc from what the run produced (or a
    // synthesized completion summary when the turn produced no model text) and
    // publish it as one coalesced `TokenBatch`, mirroring how `run_turn_core`
    // surfaces its final assistant text.
    let answer = derive_kernel_turn_answer(&event_log, &session_id, &state).await?;
    let stream_id = format!("kernel-{}", run_id.as_str());
    ui_bus.publish_token(seq, Some(session_id.clone()), stream_id, &answer);
    ui_bus.flush(Some(session_id.clone()));

    // (F4) Persist the assistant turn so the NEXT turn's `rebuild_history` sees it
    // (multi-turn continuity, parity with `run_turn_core`'s post-turn persist).
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": answer }),
        ))
        .await?;

    // (4) Post-turn publish: the full live `context_manifest` (Spine A) + a `turn`
    // completion patch carrying the terminal phase (mirrors the single-shot post-turn).
    publish_turn_context_manifest(base_url, &ui_bus, &session_id, seq, used_est).await;
    // Task COMPLETION revokes the write lease: the run this lease was granted for has reached a
    // terminal phase, so the authorization it carried is spent. Scoped to THIS run, so a lease held
    // by another task survives.
    if crate::tools::revoke_write_lease_for_run(run_id.as_str(), Some(session_id.as_str())).is_some()
    {
        publish_write_lease(&ui_bus, None, "the task completed");
    }
    ui_bus.publish(UiEvent {
        seq,
        session_id: Some(session_id.clone()),
        kind: UiEventKind::ProjectionPatch {
            projection: "turn".to_string(),
            patch: json!({
                "phase": state.phase.wire_name(),
                "run_id": run_id.as_str(),
            }),
        },
    });

    // Working memory: cleared by `_working_guard` Drop on scope exit.
    Ok(state)
}

/// Derive the turn's VISIBLE assistant answer (F2) from what the kernel run
/// produced. The driver streams model output into `agent.observation` payloads
/// (`{"generated": ...}`); the last non-empty one for THIS run is the natural
/// answer. When the run produced no model text (a pure tool/effect turn), we
/// synthesize a concise completion summary from the terminal phase + last verdict
/// - NEVER a model call - so a client always sees a real answer rather than
/// nothing.
pub(crate) async fn derive_kernel_turn_answer(
    event_log: &hide_core::persistence::DynEventLog,
    session_id: &SessionId,
    state: &AgentState,
) -> Result<String> {
    let events = event_log.scan(Some(session_id.clone()), None, None).await?;
    // The driver tags observations with the run's OWN id (`state.run_id`, minted
    // inside `start_run`) - NOT the host-side `run_id` used for interrupts - so we
    // scope to that to read back only THIS run's model output.
    let generated = events
        .iter()
        .filter(|e| e.run_id.as_ref() == Some(&state.run_id) && e.kind == "agent.observation")
        .filter_map(|e| e.payload.get("generated").and_then(|g| g.as_str()))
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .last()
        .map(|s| s.to_string());
    Ok(generated.unwrap_or_else(|| synthesize_completion_summary(state)))
}

/// Non-model completion summary (the F2 fallback): a one-line verdict of what the
/// run did, from its terminal phase + last acceptance verdict. Used only when the
/// run yielded no natural model answer (e.g. a pure edit turn). Deterministic -
/// no model call - so it can never fabricate content.
pub(crate) fn synthesize_completion_summary(state: &AgentState) -> String {
    let phase = state.phase.wire_name();
    match &state.last_verdict {
        Some(v) => {
            let status = format!("{:?}", v.status).to_lowercase();
            let detail = v.detail.trim();
            if detail.is_empty() {
                format!("Turn {phase}: verification {status}.")
            } else {
                format!("Turn {phase}: verification {status} ({detail}).")
            }
        }
        None => format!("Turn {phase}: no verification ran."),
    }
}

/// Surface a paused effectful step awaiting approval: append a durable
/// `approval.requested` event (so a reconnecting client can replay it) AND push a
/// live `Custom` UiEvent on Wire-B (so a connected client sees it now), mirroring
/// how the security gate surfaces a held command. Carries the `run_id` + the
/// pending `step_id` the `approve_effect`/`deny_effect` intent echoes back.
pub(crate) async fn announce_approval_request(
    event_log: &hide_core::persistence::DynEventLog,
    ui_bus: &Arc<UiEventBus>,
    session_id: &SessionId,
    run_id: &RunId,
    request: &ApprovalRequest,
) -> Result<()> {
    let record = event_log
        .append(NewEvent::system(
            session_id.clone(),
            "approval.requested",
            json!({
                "run_id": run_id.as_str(),
                "step_id": request.step_id.as_str(),
                "summary": request.summary,
                "effects": request.effects,
            }),
        ))
        .await?;
    ui_bus.publish(UiEvent {
        seq: record.seq,
        session_id: Some(session_id.clone()),
        // `kind` is the discriminator EVERY other Custom UiEvent uses and the one the
        // frontend router switches on (app/src/store.ts). This event carried `type`,
        // so the only surface that could act on a paused effectful step never saw it
        // and a SuggestOnly turn deadlocked awaiting an approval nobody could give.
        kind: UiEventKind::Custom(json!({
            "kind": "approval_requested",
            "run_id": run_id.as_str(),
            "step_id": request.step_id.as_str(),
            "summary": request.summary,
            "effects": request.effects,
        })),
    });
    Ok(())
}

/// Record how a pending approval was resolved (durable audit): the decision plus
/// the step it applied to. The effect it authorizes (or the skip a deny causes)
/// is recorded by the driver's own `agent.action`/`agent.phase` events.
pub(crate) async fn record_approval_resolved(
    event_log: &hide_core::persistence::DynEventLog,
    session_id: &SessionId,
    run_id: &RunId,
    request: &ApprovalRequest,
    decision: ApprovalDecision,
) -> Result<()> {
    let decision_str = match decision {
        ApprovalDecision::Approve => "approve",
        ApprovalDecision::Deny => "deny",
    };
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "approval.resolved",
            json!({
                "run_id": run_id.as_str(),
                "step_id": request.step_id.as_str(),
                "decision": decision_str,
            }),
        ))
        .await?;
    Ok(())
}

/// Publish the live `context_manifest` the Context Stack reads (Spine A), read
/// live from the engine's `/v1/hawking/context` - never a constant. Shared by the
/// kernel turn path; a `None` snapshot (serve down / pre-context build) publishes
/// nothing rather than a fake ceiling.
pub(crate) async fn publish_turn_context_manifest(
    base_url: String,
    ui_bus: &Arc<UiEventBus>,
    session_id: &SessionId,
    seq: u64,
    used_est: usize,
) {
    use crate::model_provider::HttpModelProvider;
    let ctx_provider = HttpModelProvider::new(base_url);
    if let Some(info) = ctx_provider.get_context_info().await {
        let ceiling = info.ctx_len_effective.or(info.ctx_len_native).unwrap_or(0);
        let live = build_live_manifest(info.recurrent_state_bytes, info.ctx_len_native, ceiling, used_est);
        let capability = declare_turn_capability(
            info.ctx_len_native.unwrap_or(ceiling).max(1),
            info.ctx_len_native,
            info.ctx_len_effective.or(Some(ceiling)),
            Some(info.tq_multiplier),
            info.tq_estimated,
        );
        let empty = hawking_context::ContextManifest::new(info.ctx_len_native.unwrap_or(ceiling));
        let rot = hawking_context::detect_context_rot(
            &empty,
            Some(live.occupancy),
            Some(live.watermark),
            live.recall_fidelity,
            hawking_context::RotThresholds::default(),
        );
        let meter = hawking_context::ContextMeter::from_parts(
            &capability,
            used_est,
            true,
            Some(&live),
            Some(&rot),
        );
        let mut live_json = serde_json::to_value(&live).unwrap_or_else(|_| json!({}));
        if let Some(obj) = live_json.as_object_mut() {
            obj.insert("used_tokens_estimate".to_string(), json!(used_est));
            obj.insert("estimated".to_string(), json!(true));
        }
        ui_bus.publish(UiEvent {
            seq,
            session_id: Some(session_id.clone()),
            kind: UiEventKind::ProjectionPatch {
                projection: "context_manifest".to_string(),
                patch: json!({
                    "model_id": info.model_id,
                    "arch": info.arch,
                    "ctx_len_native": info.ctx_len_native,
                    "ctx_len_effective": info.ctx_len_effective,
                    "tq_multiplier": info.tq_multiplier,
                    "tq_estimated": info.tq_estimated,
                    "recurrent_state_bytes": info.recurrent_state_bytes,
                    "active_slots": info.active_slots,
                    "free_slots": info.free_slots,
                    "live": live_json,
                    "capability": capability,
                    "rot": rot,
                    "meter": meter,
                    "native_is_not_usable": true,
                }),
            },
        });
    }
}

/// What [`run_turn_core`] returns to its callers: the full completion plus the
/// two bits the live [`generate_submit_turn`] path needs to publish its post-turn
/// `context_manifest` (the stream's seq, and the folded-prompt char length for
/// the used-token estimate).
pub(crate) struct TurnOutcome {
    pub(crate) completion: String,
    pub(crate) stream_seq: u64,
    pub(crate) prompt_chars: usize,
}

/// The SINGLE generation core both entry points funnel through
/// ([`BackendHost::generate_and_publish`] and the spawnable
/// [`generate_submit_turn`]) so the live path and headless tests exercise ONE
/// code path and can never drift.
///
/// It fixes the Phase-1b facade: instead of a raw prompt with an empty history
/// and a hard `max_output_tokens: 256`, it (1) compiles a REAL `ContextPack`
/// from the code index, (2) rebuilds REAL message history from the event log,
/// (3) folds compiled context + history + the user prompt into `prompt` (the
/// native generate route ignores `messages`), (4) derives the output budget from
/// the model window minus what the context consumed, and (5) persists a
/// `context.compiled` marker before streaming and an `agent.message` assistant
/// event after - so the NEXT turn sees this turn in its history.
///
/// `live_ceiling` (the pre-streaming `/v1/hawking/context` snapshot) is `Some`
/// only on the live path; when set, the token sink emits a throttled per-step
/// occupancy patch. `run_id_label` tags the `runtime.generation` event.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_turn_core(
    inference: Arc<dyn hawking_orch::inference::InferenceClient>,
    event_log: hide_core::persistence::DynEventLog,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
    classed_memory: hawking_context::DynClassedMemory,
    ui_bus: Arc<UiEventBus>,
    session_id: SessionId,
    prompt: String,
    live_ceiling: Option<(Option<usize>, Option<usize>, usize)>,
    run_id_label: Option<String>,
    repo_instructions: Arc<crate::compat_instructions::ResolvedInstructions>,
) -> Result<TurnOutcome> {
    use crate::connectors::choose_context_role;
    use hawking_context::compiler::CompileInput;
    use hawking_context::profiles::ContextProfile;
    use hawking_context::sources::{ClassedMemoryContextSource, CodeIndexContextSource};
    use hawking_context::{
        ClassBudgets, ContextCompiler, InMemoryMemoryStore, MemoryKind,
    };
    use hawking_orch::router::SimpleRouter;
    use hide_core::runtime::{InferenceMessage, InferenceRequest, StreamChunk};
    use hide_core::types::Provenance;
    use hide_kernel::runtime_client::KernelRuntimeClient;

    // Working memory (turn-local): sole TurnWriteCap mint is inside
    // WorkingTurnGuard::begin; Drop clears the row on every exit path.
    let turn_id = run_id_label
        .clone()
        .unwrap_or_else(|| format!("turn-{}", session_id.as_str()));
    let _working_guard = crate::classed_writers::WorkingTurnGuard::begin(
        classed_memory.clone(),
        turn_id.clone(),
        session_id.as_str(),
        run_id_label.as_deref(),
        &prompt,
    );

    // --- (S3) Compile a REAL ContextPack (bible §4.2). Mirrors the `context`
    // connector so both share one recipe: pick the coding role, size the window
    // to its model, and let the code-index + classed memory compete for budget. ---
    // §7.3 honesty: prefer live-measured native over the role/config default;
    // never pack against an inflated effective ceiling *as if* it were native.
    let role = choose_context_role(&role_registry, None)?;
    let config_native = role.model.context_tokens.max(4096);
    let live_native = live_ceiling.and_then(|(_, n, _)| n).filter(|n| *n > 0);
    let live_effective = live_ceiling
        .map(|(_, _, c)| c)
        .filter(|c| *c > 0);
    let capability = declare_turn_capability(
        config_native,
        live_native,
        live_effective,
        None,
        false,
    );
    // Pack against measured/config native (conservative). Effective is reported
    // separately and is never treated as a larger native window.
    let max_input = capability.pack_budget_tokens(false).max(256);
    let mut model = role.model.clone();
    model.context_tokens = max_input;
    // Tokenizer-true packing when a real tokenizer is discoverable (bible §4.2).
    let counter = hawking_context::TokenCounter::discover_from_env()
        .unwrap_or_else(hawking_context::TokenCounter::heuristic);
    let mut compiler = ContextCompiler::new().with_counter(counter);
    compiler.add_source(CodeIndexContextSource::new(code_index, 16));
    // Six memory classes: independent per-class budgets (not one kind filter).
    let class_budgets = ClassBudgets::from_total((max_input / 8).max(64));
    compiler.add_source(
        ClassedMemoryContextSource::new(classed_memory.clone(), class_budgets)
            .with_session(session_id.as_str())
            .with_turn(turn_id.clone()),
    );
    // Bible sec 20 / sec 78.1 #11: fold the repo's resolved Claude Code migration
    // instructions (CLAUDE.md tree + un-scoped rules) into the compiled context as
    // a pinned instruction/system source, honoring precedence (read-last-wins).
    // Added only when the repo actually carries them (an un-migrated repo resolves
    // empty and this is a no-op).
    if !repo_instructions.is_empty() {
        compiler.add_source(repo_instructions.as_source());
    }
    let mut compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(max_input),
            model,
            task: prompt.clone(),
        })
        .await?;
    // Pre-stream live reading (when the ceiling was snapshotted) so rot/meter
    // can include occupancy before generation advances.
    let pre_live = live_ceiling.map(|(state_bytes, native, ceiling)| {
        build_live_manifest(state_bytes, native, ceiling, compiled.manifest.used_tokens)
    });
    seal_compiled_manifest(
        &mut compiled.manifest,
        capability,
        pre_live.as_ref(),
        compiled.tokens_estimated,
    );
    // Surface per-class memory budgets on the context meter.
    if let (Some(meter), Some(ret)) = (
        compiled.manifest.meter.as_mut(),
        classed_memory.last_retrieval(),
    ) {
        meter.explanations.extend(ret.budget_explanations());
    }
    // Spine B (best-effort): accrue the Project Brain with this compile. A brain
    // write must never fail a turn.
    let brain = InMemoryMemoryStore::record(
        MemoryKind::Project,
        format!(
            "task: {prompt}\nretained {} spans, {} tokens used",
            compiled.manifest.retained.len(),
            compiled.manifest.used_tokens
        ),
        Provenance::trusted("submit_turn.compile"),
    );
    let _ = memory.upsert(brain).await;

    // --- (S2) Rebuild REAL message history from the durable event log, then
    // ensure the current user prompt is the final user message (the live path's
    // `user.intent.submit_turn` is already logged, so it is usually present
    // already; `generate_and_publish` may pass an explicit prompt that is not). ---
    let mut messages = rebuild_history(&event_log, &session_id).await?;
    if messages
        .last()
        .map(|m| m.role != "user" || m.content != prompt)
        .unwrap_or(true)
    {
        messages.push(InferenceMessage {
            role: "user".to_string(),
            content: prompt.clone(),
        });
    }
    let history_block = messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");
    // The native `/v1/hawking/generate` route sends only `prompt` (it drops
    // `messages`), so FOLD compiled context + rendered history into `prompt`.
    // `messages` is still populated for a future Chat-route switch.
    let folded_prompt = if compiled.prompt.trim().is_empty() {
        history_block
    } else {
        format!("{}\n\n{}", compiled.prompt, history_block)
    };
    let prompt_chars = folded_prompt.len();

    // --- (S2) Derive the output budget from the window minus what context ate,
    // clamped to a sane band - replacing the hard-coded 256 facade. ---
    // `HIDE_MAX_OUTPUT_TOKENS` (positive int) is an optional hard cap for live
    // smoke / small-model turns; it never *raises* the derived budget.
    let derived = max_input
        .saturating_sub(compiled.manifest.used_tokens)
        .clamp(256, 2048);
    let out_budget = std::env::var("HIDE_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|n| *n > 0)
        .map(|cap| derived.min(cap))
        .unwrap_or(derived);

    // Durable marker: compile stats + honest capability / rot / meter.
    // The compile receipt lives on the event log (not a pre-token Wire-B patch)
    // so token-first consumers (flagship boot path) are not starved of TokenBatch.
    // Post-turn publish_turn_context_manifest / generate_submit_turn re-emit
    // capability+rot+meter on the live context_manifest projection.
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "context.compiled",
            context_compiled_payload(
                &compiled.manifest,
                Some(out_budget),
                "single_shot",
                run_id_label.as_deref(),
            ),
        ))
        .await?;

    // Context receipt: which repo instruction files (CLAUDE.md tree + un-scoped
    // rules) folded into this turn's context, in launch order. Logged only when
    // the repo carried migration instructions.
    if !repo_instructions.is_empty() {
        event_log
            .append(NewEvent::system(
                session_id.clone(),
                "context.instructions",
                repo_instructions.receipt_json(),
            ))
            .await?;
    }

    let request = InferenceRequest {
        task_kind: "code".to_string(),
        prompt: folded_prompt,
        messages,
        max_output_tokens: out_budget,
        sampler: None,
        grammar: None,
        want_logprobs: false,
        metadata: Default::default(),
    };

    // Route through the kernel runtime-client seam (router + inference client).
    let router = Arc::new(SimpleRouter::new(role_registry));
    let runtime = KernelRuntimeClient::new(router, inference);

    // A stable seq to key the published UiEvent stream off of.
    let status_event = event_log
        .append(NewEvent::system(
            session_id.clone(),
            "runtime.generation",
            json!({ "task": "code", "run_id": run_id_label }),
        ))
        .await?;
    let stream_id = status_event.seq.to_string();

    let mut buf = String::new();
    {
        let bus = ui_bus.clone();
        let sess = session_id.clone();
        let sid = stream_id.clone();
        let seq = status_event.seq;
        let mut tok_count = 0usize;
        let mut sink = |chunk: StreamChunk| {
            match chunk {
                StreamChunk::Token { text, .. } => {
                    buf.push_str(&text);
                    bus.publish_token(seq, Some(sess.clone()), &sid, &text);
                    // Throttled per-step occupancy (every 32 tokens), partial patch
                    // - only when the live ceiling was snapshotted (live path).
                    tok_count += 1;
                    if tok_count % 32 == 0 {
                        if let Some((state_bytes, native, ceiling)) = live_ceiling {
                            let used_est = (prompt_chars + buf.len()) / 4;
                            let live = build_live_manifest(state_bytes, native, ceiling, used_est);
                            if let Ok(mut lj) = serde_json::to_value(&live) {
                                if let Some(o) = lj.as_object_mut() {
                                    o.insert("used_tokens_estimate".to_string(), json!(used_est));
                                    o.insert("estimated".to_string(), json!(true));
                                    o.insert("partial".to_string(), json!(true));
                                }
                                bus.publish(UiEvent {
                                    seq,
                                    session_id: Some(sess.clone()),
                                    kind: UiEventKind::ProjectionPatch {
                                        projection: "context_manifest".to_string(),
                                        patch: json!({ "live": lj }),
                                    },
                                });
                            }
                        }
                    }
                }
                StreamChunk::Done { .. } => {
                    bus.flush(Some(sess.clone()));
                }
                StreamChunk::Error { message } => {
                    bus.publish(UiEvent {
                        seq,
                        session_id: Some(sess.clone()),
                        kind: UiEventKind::Error {
                            code: "generation".to_string(),
                            message,
                        },
                    });
                }
            }
            Ok(())
        };
        runtime.generate(request, &mut sink).await?;
    }

    // (S2) Persist the assistant turn so the NEXT turn's `rebuild_history` sees it.
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": buf }),
        ))
        .await?;

    // Working memory must not outlive the turn — `_working_guard` Drop clears it.
    Ok(TurnOutcome {
        completion: buf,
        stream_seq: status_event.seq,
        prompt_chars,
    })
}

/// Rebuild the prior conversation as `InferenceMessage`s from the durable event
/// log: a `user.intent.submit_turn` becomes a `user` message (its `args.text`),
/// and an `agent.message` with `role == "assistant"` becomes an `assistant`
/// message (its `text`). Everything else is ignored. Ordered by seq (scan order).
pub(crate) async fn rebuild_history(
    event_log: &hide_core::persistence::DynEventLog,
    session_id: &SessionId,
) -> Result<Vec<hide_core::runtime::InferenceMessage>> {
    use hide_core::runtime::InferenceMessage;
    let events = event_log.scan(Some(session_id.clone()), None, None).await?;
    let mut out = Vec::new();
    for ev in events {
        match ev.kind.as_str() {
            "user.intent.submit_turn" => {
                if let Some(text) = ev
                    .payload
                    .get("args")
                    .and_then(|a| a.get("text"))
                    .and_then(|t| t.as_str())
                {
                    if !text.is_empty() {
                        out.push(InferenceMessage {
                            role: "user".to_string(),
                            content: text.to_string(),
                        });
                    }
                }
            }
            "agent.message" => {
                let role = ev
                    .payload
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("assistant");
                if role == "assistant" {
                    if let Some(text) = ev.payload.get("text").and_then(|t| t.as_str()) {
                        out.push(InferenceMessage {
                            role: "assistant".to_string(),
                            content: text.to_string(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Spine A (W-F2-1): pick the live-context regime. An SSM (a model reporting a
/// constant recurrent-state footprint) surfaces recall FIDELITY from the
/// calibratable probe; a transformer surfaces KV occupancy. The probe is the
/// swap point for a measured boot-needle curve later.
pub(crate) fn build_live_manifest(
    recurrent_state_bytes: Option<usize>,
    ctx_len_native: Option<usize>,
    ceiling: usize,
    state_age_tokens: usize,
) -> hawking_context::manifest::ManifestLive {
    use hawking_context::fidelity::{LinearFidelity, RecallFidelityProbe};
    use hawking_context::manifest::ManifestLive;
    if let Some(state_bytes) = recurrent_state_bytes {
        let probe = LinearFidelity::new(ctx_len_native.unwrap_or(0));
        let fidelity = probe.fidelity(state_age_tokens);
        ManifestLive::ssm(state_bytes, state_age_tokens, fidelity, ceiling)
    } else {
        ManifestLive::transformer(state_age_tokens, ceiling)
    }
}

/// Honest §7.3 capability declaration for one turn.
///
/// Prefer a live-measured native window over the role/config default. Never
/// promote effective (`.tq` / position-scaled) or retrieval-usable context into
/// `native_maximum`. Validated quality/agentic and curves stay unmeasured until
/// a real calibration produces them.
pub(crate) fn declare_turn_capability(
    config_native: usize,
    live_native: Option<usize>,
    live_effective: Option<usize>,
    tq_multiplier: Option<f32>,
    tq_estimated: bool,
) -> hawking_context::ContextCapability {
    use hawking_context::{CompactionMode, ContextCapability, RetrievalMode};
    ContextCapability::declare(
        config_native,
        live_native,
        live_effective,
        tq_multiplier,
        tq_estimated,
        // Code-index + memory retrieval feed the packer; still capped by the window.
        RetrievalMode::RetrieveThenPack,
        // Compiler degrade ladder with recall-gated rollback (Spine B).
        CompactionMode::DegradeWithRecallGate,
    )
}
