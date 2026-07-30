use crate::approval::{ApprovalDecision, ApprovalHub};
use crate::commands::CommandRouter;
use crate::connectors::{register_backend_connectors, ConnectorRegistry, ConnectorStatus};
use crate::initialize::{ClientCapabilities, ClientInfo, ConnectionRegistry, InitializeResponse};
use crate::interrupt::InterruptHub;
use crate::live_thread::LiveThread;
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
use crate::services::{
    BackendCapabilities, BackendServices, Budget, CheckpointRecord, CheckpointStore,
    EnvironmentNode, EnvironmentSwitch, GoalOutcome, GoalRecord, GoalStatus, GoalStore,
    GoalVerdict, JobRecord, JobStatus, JobStore, RepoNode, SharedBackend, Trigger, TriggerEvent,
    TrustState, WorkspaceEdge, WorkspaceEdgeKind, WorkspaceGraph, WorkspaceStore,
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
use super::*;
use hide_kernel::verify_plane::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

impl BackendHost {
    pub async fn handle_intent(&self, intent: Intent) -> Result<IntentAck> {
        // Snapshot the SubmitTurn parameters before the router consumes the
        // intent (it takes `intent` by value and returns only an `IntentAck`).
        let submit = match &intent {
            Intent::SubmitTurn {
                session_id, text, ..
            } => Some((session_id.clone(), text.clone())),
            _ => None,
        };
        // Snapshot a RunCommand too: an accepted one actually executes in the workspace and streams
        // its output back as tool_progress (the integrated terminal renders those rows).
        let run_cmd = match &intent {
            Intent::RunCommand { argv, cwd } => Some((argv.clone(), cwd.clone())),
            _ => None,
        };
        // Terminal / process custom intents. `pty_input` writes bytes to a live process's stdin,
        // `pty_resize` records its terminal geometry (`{ process?, data }` / `{ process?, cols,
        // rows }`; an absent `process` targets the most recently started live process).
        // `attach_process`, `stop_process` and `capture_process_artifact` address ONE named
        // process (`{ process }`): re-attach after a navigation, stop what you started, keep the
        // output as a durable artifact.
        let process_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "pty_input"
                        | "pty_resize"
                        | "attach_process"
                        | "stop_process"
                        | "capture_process_artifact"
                ) =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };
        // A held command's approve/deny round-trip: `approve_gate`/`deny_gate` carry the gate id the
        // `SecurityGate` UiEvent was emitted with. `(approve, gate_id)`.
        let gate_action: Option<(bool, String)> = match &intent {
            Intent::Custom { name, payload } if name == "approve_gate" || name == "deny_gate" => {
                payload
                    .get("gate")
                    .and_then(|v| v.as_str())
                    .map(|g| (name == "approve_gate", g.to_string()))
            }
            _ => None,
        };
        // A paused effectful kernel step's approve/deny round-trip: `approve_effect`/
        // `deny_effect` carry the `run_id` and (for approve, required) `step_id`
        // the `approval.requested` event was emitted with. `(approve, run_id, step_id)`.
        let approval_action: Option<(bool, RunId, Option<StepId>)> = match &intent {
            Intent::Custom { name, payload }
                if name == "approve_effect" || name == "deny_effect" =>
            {
                payload.get("run_id").and_then(|v| v.as_str()).map(|r| {
                    let step = payload
                        .get("step_id")
                        .and_then(|v| v.as_str())
                        .map(StepId::from);
                    (name == "approve_effect", RunId::from(r), step)
                })
            }
            _ => None,
        };
        // A ForkSession intent (bible sec 78.1 #7): snapshot the source + boundary
        // so, once the router has recorded the intent, the host actually forks a
        // new independent session, records ancestry, and surfaces the new thread.
        let fork_action: Option<(SessionId, hide_core::ids::EventId)> = match &intent {
            Intent::ForkSession {
                session_id,
                at_event,
            } => Some((session_id.clone(), at_event.clone())),
            _ => None,
        };
        // Side-chat lifecycle custom intents (bible sec 32-33, sec 78.1 #9):
        // `create_side_chat` forks a read-only side chat; `merge_side_chat` folds
        // its typed summary back onto the parent. Snapshotted so we act once the
        // router has recorded the intent (mirrors the ForkSession path).
        let side_chat_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if name == "create_side_chat" || name == "merge_side_chat" =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };
        // Launcher (courtyard) custom intents: snapshot the ones with a side effect so we can act after
        // the router has recorded them in the event log. `fleet_run` is additive: it reaches
        // `BackendHost::fleet_run` the same way neighbouring intents reach their handlers.
        let launcher_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "create_worktree" | "new_session" | "open_session" | "fleet_run"
                ) =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };
        // Surface graph (YOU / CHAT / IDE): switch lens, seal claim-only handoff, receive.
        // Snapshot so the router records the intent first; effects run only when accepted.
        let surface_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "switch_surface" | "handoff_create" | "handoff_receive"
                ) =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };
        // Durable Goal + Checkpoint custom intents (bible sec 14, sec 15.4, sec
        // 78.1 #3): snapshot so we act once the router has recorded the intent
        // (mirrors the ForkSession / side-chat paths).
        let goal_checkpoint_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "goal_set"
                        | "goal_clear"
                        | "checkpoint_create"
                        | "checkpoint_restore"
                        | "checkpoint_rewind"
                        | "checkpoint_replay"
                        | "checkpoint_fork"
                        | "checkpoint_compare"
                        | "checkpoint_inspect"
                ) =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };
        // Mid-turn STEER (census priority 6, the true end-to-end hole): the FE
        // `redirect_run` gesture carries `{ run_id, text, session_id? }`. Snapshot
        // it so, once the router has recorded the intent, we deliver a real
        // `InterruptHub::Steer` to the running kernel (mirrors how CancelRun/
        // PauseRun route to Abort/Pause) and persist a durable `turn.steer` event.
        let steer_action: Option<(RunId, String, Option<SessionId>)> = match &intent {
            Intent::Custom { name, payload } if name == "redirect_run" || name == "steer" => {
                payload.get("run_id").and_then(|v| v.as_str()).map(|run| {
                    let text = payload
                        .get("text")
                        .or_else(|| payload.get("instruction"))
                        .or_else(|| payload.get("directive"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let session = payload
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .map(SessionId::from);
                    (RunId::from(run), text, session)
                })
            }
            _ => None,
        };
        // Durable Memory + Goal-eval + Workspace-trust + Environment-switch custom
        // intents (bible sec 21-22, 14, 35): snapshot so we route to the existing
        // tested host method once the router has recorded the intent (mirrors the
        // goal/checkpoint path).
        let memory_workspace_env_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "memory_add"
                        | "memory_supersede"
                        | "memory_record_outcome"
                        | "memory_revalidate"
                        | "goal_evaluate"
                        | "workspace_set_repo_trust"
                        | "environment_switch"
                ) =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };

        // Stage 4 background-promotion custom intents: `promote_run` promotes a
        // live interactive run to a durable background job (no restart);
        // `resume_run_foreground` reattaches a reconnecting client to a promoted
        // run and resumes it in the foreground. Snapshotted so we act once the
        // router has recorded the intent (mirrors the memory/goal paths). The
        // steer / pause / stop / fork gestures on a promoted run reuse the existing
        // `redirect_run` / `pause_run` / `cancel_run` / `fork_session` intents,
        // which already route by run id, so no new arm is needed for those.
        let background_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if name == "promote_run" || name == "resume_run_foreground" =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };

        // Plan-domain custom intents (Stage 1, bible sec 14): the PlanCard's
        // approve / edit / reorder / skip / repair gestures. Snapshotted so we
        // mutate the durable plan record + republish the `plan` projection once the
        // router has recorded the intent (mirrors the goal/memory paths). These
        // stop being log-only.
        let plan_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "approve_plan"
                        | "edit_plan_step"
                        | "reorder_plan"
                        | "skip_step"
                        | "repair_step"
                ) =>
            {
                Some((name.clone(), payload.clone()))
            }
            _ => None,
        };

        // Transcript SEARCH over /intent (census sec 32-33): the command palette /
        // Search panel dials `/intent` (never `/rpc`), so the built literal +
        // structured search needs a custom-name arm. `run_search` is the FE's
        // registered name (wire.ts CUSTOM_NAMES); `search` / `search_transcript`
        // are accepted aliases. The payload carries `{ query, scopes?, limit?, ...
        // }`; semantic search stays DEFERRED_MODEL_REQUIRED. Snapshotted so we run
        // it once the router has recorded the intent (mirrors the memory/goal paths).
        let search_action: Option<Value> = match &intent {
            Intent::Custom { name, payload }
                if name == "run_search" || name == "search" || name == "search_transcript" =>
            {
                Some(payload.clone())
            }
            _ => None,
        };

        // Diff review (census sec 23): the accept/reject gestures gain real
        // per-hunk targeting (the optional `hunk_id`), and the custom name
        // `revert_diff` routes to `revert_diff`, so all three stop being log-only.
        // `(op, diff_id, hunk_id)`.
        let diff_action: Option<(&'static str, String, Option<String>)> = match &intent {
            Intent::AcceptDiff {
                diff_id, hunk_id, ..
            } => Some(("accept", diff_id.clone(), hunk_id.clone())),
            Intent::RejectDiff {
                diff_id, hunk_id, ..
            } => Some(("reject", diff_id.clone(), hunk_id.clone())),
            Intent::Custom { name, payload } if name == "revert_diff" => Some((
                "revert_diff",
                payload
                    .get("diff_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string(),
                None,
            )),
            _ => None,
        };

        // The editor save (`{ path, content, base_hash? }`). It rides the intent channel like every
        // other effect, so the permission engine's refusal reaches the approval gate instead of
        // being thrown at a connector caller with nowhere to take it.
        let save_action: Option<Value> = match &intent {
            Intent::Custom { name, payload } if name == "save_file" => Some(payload.clone()),
            _ => None,
        };

        // Static analysis over the intent channel (census priority 1): the Problems counter's only
        // producer. `{ session_id?, sources: [{path,text}] }`, or `{ session_id?, paths: [rel] }`
        // to read them from the workspace. Model-free (the hide-verify Tier1 oracle).
        let static_analysis_action: Option<Value> = match &intent {
            Intent::Custom { name, payload } if name == "run_static_analysis" => {
                Some(payload.clone())
            }
            _ => None,
        };

        // The sealed diff review receipt (`{ diff_id, session_id? }`). The host could seal one all
        // along and no client could ask for it; it is reachable now that a wire-reachable write
        // actually produces a diff to seal.
        let review_receipt_action: Option<Value> = match &intent {
            Intent::Custom { name, payload } if name == "export_review_receipt" => {
                Some(payload.clone())
            }
            _ => None,
        };

        // Effect policy (`CommandSpec::approval_policy`): an intent whose EFFECT the command
        // authority marks `Ask` is RECORDED but its effect is parked at the security gate.
        // `approve_gate` releases it, `deny_gate` drops it. Enforced here because the host is the
        // only place that sees every intent regardless of which surface dispatched it.
        let ask_action: Option<(String, Value)> =
            Self::effect_command(&intent).filter(|(name, _)| Self::requires_approval(name));

        // An HONEST ack: a custom name with no handler here is recorded (the log is the audit
        // trail) but NOT reported as accepted, so a frontend control can never look like it worked.
        let unhandled: Option<String> = match &intent {
            Intent::Custom { name, .. } if !HANDLED_CUSTOM_NAMES.contains(&name.as_str()) => {
                Some(name.clone())
            }
            _ => None,
        };

        // Write-lease revocation, read in ONE place off the intent itself. Every trigger that can
        // arrive as an intent is here, so a new surface firing an existing name cannot miss one:
        // explicit revocation, task cancellation, session closure / fork / switch, a rewind past
        // the grant, repository trust withdrawn, and a scope change. Read off the RECORDED intent
        // rather than off the effect, so an approval-gated rewind revokes the moment it is asked
        // for instead of after it runs. Revoking early only ever narrows what is permitted.
        // (The two triggers that are not intents: task COMPLETION revokes in the turn driver's
        // terminal publish, and RESTART invalidates because the lease is process memory only.)
        let lease_revocation: Option<(&'static str, LeaseRevokeScope)> = match &intent {
            Intent::CancelRun { run_id } => Some((
                "the task was cancelled",
                LeaseRevokeScope::Run(run_id.as_str().to_string()),
            )),
            Intent::ForkSession { .. } => Some(("the session was forked", LeaseRevokeScope::Any)),
            Intent::Custom { name, payload } => match name.as_str() {
                "revoke_write_lease" => Some(("revoked by the user", LeaseRevokeScope::Any)),
                "new_session" | "open_session" => {
                    Some(("the session was closed", LeaseRevokeScope::Any))
                }
                "checkpoint_restore" | "checkpoint_rewind" => Some((
                    "the session was rewound past the grant",
                    LeaseRevokeScope::Any,
                )),
                // Trust withdrawn from the leased repo. A re-trust is not a revocation.
                "workspace_set_repo_trust"
                    if payload.get("trust").and_then(|v| v.as_str()) != Some("trusted") =>
                {
                    payload.get("repo_id").and_then(|v| v.as_str()).map(|repo| {
                        (
                            "repository trust was withdrawn",
                            LeaseRevokeScope::Repo(repo.to_string()),
                        )
                    })
                }
                // The environment carries the fs roots, so switching it changes the scope.
                "environment_switch" => {
                    Some(("the environment scope changed", LeaseRevokeScope::Any))
                }
                _ => None,
            },
            _ => None,
        };

        let mut ack = self.commands.handle(intent).await?;

        if ack.accepted {
            if let Some((reason, scope)) = lease_revocation {
                if let Some(revoked) = scope.revoke() {
                    publish_write_lease(&self.ui_bus, None, reason);
                    self.ui_bus.publish(UiEvent {
                        seq: 0,
                        session_id: None,
                        kind: UiEventKind::Custom(json!({
                            "kind": "write_lease_revoked",
                            "lease_id": revoked.lease_id,
                            "reason": reason,
                        })),
                    });
                }
            }
        }

        // Park an `Ask` command's effect. The intent IS recorded, so the ack stays accepted, but
        // `held` is set so no caller can read this as done: the effect has not run and will not run
        // until `approve_gate` releases it.
        let mut effect_ok = ack.accepted;
        if let (true, Some((name, payload))) = (ack.accepted, ask_action) {
            effect_ok = false;
            match self.hold_at_gate(
                PendingAction::Intent {
                    name: name.clone(),
                    payload,
                },
                format!("approval required before {name} takes effect"),
            ) {
                Ok(gate) => {
                    ack.held = true;
                    ack.message = Some(format!("held for approval: gate={gate}"));
                }
                Err(err) => self.effect_failed(&mut ack, &name, err.to_string()),
            }
        }

        // Only an ACCEPTED SubmitTurn starts generation (a rejected one, e.g.
        // empty text, returned `accepted: false` and logged nothing).
        if let (true, Some((session_id, prompt))) = (effect_ok, submit) {
            self.spawn_submit_turn_generation(session_id, prompt);
        }
        // A destructive argv is parked at the SAME gate an `Ask` command is, so it reports the same
        // third state: the intent is recorded, nothing ran, and the caller is told so.
        if let (true, Some((argv, cwd))) = (effect_ok, run_cmd) {
            match self.spawn_command_run(argv, cwd) {
                Ok(Some(gate)) => {
                    ack.held = true;
                    ack.message = Some(format!("held for approval: gate={gate}"));
                }
                Ok(None) => {}
                Err(err) => self.effect_failed(&mut ack, "run_command", err.to_string()),
            }
        }
        // Terminal / process side effect: deliver a keystroke (`pty_input`), a resize
        // (`pty_resize`), or an attach / stop / capture to the named process, once the intent is
        // recorded. A failure (no such process, or a non-interactive one) refuses the ack and
        // surfaces as an Error UiEvent.
        if let (true, Some((name, payload))) = (effect_ok, process_action) {
            if let Err(err) = self.handle_process_intent(&name, &payload).await {
                self.effect_failed(&mut ack, &name, err);
            }
        }
        // Release or drop a held gated command once its decision intent is recorded.
        if let (true, Some((approve, gate))) = (effect_ok, gate_action) {
            let outcome = if approve {
                self.approve_gate(&gate).await
            } else {
                self.deny_gate(&gate)
            };
            if let Err(err) = outcome {
                self.effect_failed(
                    &mut ack,
                    if approve { "approve_gate" } else { "deny_gate" },
                    err.to_string(),
                );
            }
        }
        // Deliver a paused effectful step's decision to the running turn's mailbox
        // once the decision intent is recorded. The turn drains it while paused to
        // resume (approve) or skip (deny) the step. Buffered if it arrives before
        // the turn reaches its pause.
        //
        // Security (W5): `approve_effect` REQUIRES `step_id`. Depositing Approve
        // with `None` while nothing is paused creates a blanket buffered approval
        // applied to the next effectful step the user was never shown. Deny with
        // no step_id stays fail-safe and is intentionally allowed.
        if let (true, Some((approve, run, step))) = (effect_ok, approval_action) {
            if approve && step.is_none() {
                self.effect_failed(
                    &mut ack,
                    "approve_effect",
                    "approve_effect requires step_id (blanket approve is refused)".to_string(),
                );
            } else {
                let decision = if approve {
                    ApprovalDecision::Approve
                } else {
                    ApprovalDecision::Deny
                };
                self.approvals.decide(run, step, decision);
            }
        }
        // Fork a new independent session once the ForkSession intent is recorded.
        if let (true, Some((from, at_event))) = (effect_ok, fork_action) {
            self.spawn_fork_session(from, at_event);
        }
        // Side-chat lifecycle side effects, once the intent is safely in the log.
        if let (true, Some((name, payload))) = (effect_ok, side_chat_action) {
            match name.as_str() {
                // Fork a read-only side chat from a parent at an (optional) boundary.
                "create_side_chat" => {
                    if let Some(parent) = payload.get("session_id").and_then(|v| v.as_str()) {
                        let at_event = payload
                            .get("at_event")
                            .and_then(|v| v.as_str())
                            .map(EventId::from);
                        let inherit = payload
                            .get("inherit")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(true);
                        self.spawn_create_side_chat(SessionId::from(parent), at_event, inherit);
                    }
                }
                // Merge a side chat's typed summary back onto its parent session.
                "merge_side_chat" => {
                    if let (Some(side), Some(parent), Some(summary)) = (
                        payload.get("side_chat").and_then(|v| v.as_str()),
                        payload.get("parent").and_then(|v| v.as_str()),
                        payload.get("summary").and_then(|v| v.as_str()),
                    ) {
                        self.spawn_merge_side_chat(
                            SessionId::from(side),
                            SessionId::from(parent),
                            summary.to_string(),
                        );
                    }
                }
                _ => {}
            }
        }
        // Launcher side effects, once the intent is safely in the log.
        if let (true, Some((name, payload))) = (effect_ok, launcher_action) {
            match name.as_str() {
                // Create a real, isolated git worktree so a session can run on its own branch.
                "create_worktree" => {
                    self.spawn_worktree_add(payload.get("branch").and_then(|v| v.as_str()));
                }
                // Mint a fresh session and publish it so the courtyard composer hands off to a clean run.
                "new_session" => self.emit_new_session(),
                // Load a past session: republish its recorded transcript so the FE (which adopts the
                // session off any event's session_id) switches to it and re-renders. Real events from
                // the log, never fabricated.
                "open_session" => {
                    if let Some(id) = payload.get("session_id").and_then(|v| v.as_str()) {
                        self.spawn_open_session(SessionId::from(id));
                    }
                }
                // W2: live entry for `BackendHost::fleet_run`. Payload: `{ task|objective,
                // session_id? }`. Runs inline (model-free fleet scheduling is short) so the
                // ack carries the terminal status; a failure refuses the ack.
                "fleet_run" => {
                    let objective = payload
                        .get("task")
                        .or_else(|| payload.get("objective"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("fleet run")
                        .to_string();
                    let session = payload
                        .get("session_id")
                        .and_then(|v| v.as_str())
                        .map(SessionId::from)
                        .unwrap_or_else(|| self.services.session());
                    match self.fleet_run(session.clone(), objective.clone()).await {
                        Ok(status) => {
                            ack.message = Some(format!("fleet_run status={status}"));
                            self.ui_bus.publish(UiEvent {
                                seq: 0,
                                session_id: Some(session),
                                kind: UiEventKind::Custom(json!({
                                    "kind": "fleet_run_completed",
                                    "status": status,
                                    "task": objective,
                                })),
                            });
                        }
                        Err(err) => self.effect_failed(&mut ack, "fleet_run", err.to_string()),
                    }
                }
                _ => {}
            }
        }
        // Durable Goal + Checkpoint side effects, once the intent is safely in the
        // log. Run inline via the tested host methods (they emit their own
        // UiEvents); a failure refuses the ack (see `effect_failed`) so the goal chip
        // cannot read as set when the host stored nothing.
        if let (true, Some((name, payload))) = (effect_ok, goal_checkpoint_action) {
            if let Err(err) = self.handle_goal_checkpoint_intent(&name, &payload).await {
                self.effect_failed(&mut ack, &name, err.to_string());
            }
        }
        // Surface graph side effects (switch lens / claim-only handoff). Same
        // session identity throughout; capability never rides the capsule.
        if let (true, Some((name, payload))) = (effect_ok, surface_action) {
            if let Err(err) = self.handle_surface_intent(&name, &payload).await {
                self.effect_failed(&mut ack, &name, err.to_string());
            }
        }
        // Mid-turn steer side effect: deliver the real InterruptHub signal + the
        // durable steer event once the intent is safely recorded. A failure to
        // persist the steer event surfaces as an Error UiEvent (the signal itself
        // is fire-and-forget onto the hub, so the running turn still observes it).
        if let (true, Some((run_id, text, session))) = (effect_ok, steer_action) {
            if let Err(err) = self.steer_run(run_id, text, session).await {
                self.effect_failed(&mut ack, "redirect_run", err.to_string());
            }
        }
        // Durable Memory / Goal-eval / Workspace-trust / Environment-switch side
        // effects, once the intent is safely in the log. Routes to the tested host
        // method (never duplicates its logic); a failure refuses the ack, exactly like
        // the goal/checkpoint path.
        if let (true, Some((name, payload))) = (effect_ok, memory_workspace_env_action) {
            if let Err(err) = self
                .handle_memory_workspace_env_intent(&name, &payload)
                .await
            {
                self.effect_failed(&mut ack, &name, err.to_string());
            }
        }
        // Stage 4 background-promotion side effect, once the intent is safely in
        // the log. Routes to the promote / resume-in-foreground host methods; a
        // failure (missing run_id, unknown job) refuses the ack, exactly like the
        // memory / plan paths.
        if let (true, Some((name, payload))) = (effect_ok, background_action) {
            if let Err(err) = self.handle_background_intent(&name, &payload).await {
                self.effect_failed(&mut ack, &name, err.to_string());
            }
        }
        // Plan-domain side effect, once the intent is safely in the log. Routes to
        // the durable plan handler (mutate + republish); a failure (unknown plan /
        // step / invalid order) refuses the ack, exactly like the goal / memory paths.
        if let (true, Some((name, payload))) = (effect_ok, plan_action) {
            if let Err(err) = self.handle_plan_intent(&name, &payload).await {
                self.effect_failed(&mut ack, &name, err.to_string());
            }
        }

        // Transcript search side effect, once the intent is safely in the log.
        // Runs the model-free literal + structured search and surfaces the hits as
        // a `search_results` UiEvent (the FE reads UiEvents; no /rpc dial needed).
        // A failure refuses the ack and surfaces as an Error UiEvent.
        if let (true, Some(payload)) = (effect_ok, search_action) {
            match self.handle_search_intent(&payload).await {
                Ok(hits) => self.publish_search_results(&payload, &hits),
                Err(err) => self.effect_failed(&mut ack, "search", err.to_string()),
            }
        }

        // Diff review side effect, once the intent is safely in the log. Routes to
        // the real apply/revert host method (no longer log-only); a failure (e.g.
        // an unknown diff, or a revert that conflicts) REFUSES the ack as well as
        // publishing the error, so the review surface cannot print "done" over a
        // hunk that is still on disk.
        if let (true, Some((op, diff_id, hunk_id))) = (effect_ok, diff_action) {
            let outcome = match (op, &hunk_id) {
                ("accept", Some(h)) => self.apply_hunk(&diff_id, h).await.map(|_| ()),
                ("accept", None) => self.apply_diff(&diff_id).await.map(|_| ()),
                ("reject", Some(h)) => self.reject_hunk(&diff_id, h).await.map(|_| ()),
                // The whole-diff revert. Reached only with `effect_ok`, which `effect_command`
                // clears for both of the shapes that ask for it, so neither one runs ungated.
                ("reject", None) | ("revert_diff", _) => {
                    self.revert_diff(&diff_id).await.map(|_| ())
                }
                _ => Ok(()),
            };
            // The reject/revert arms WRITE (the inverse write that puts the pre-image back), so on
            // the shipped `Ask` default they take the same hold-at-the-gate path the save does:
            // the review surface's undo is offered for approval instead of being refused outright.
            let name = match (op, &hunk_id) {
                ("reject", Some(_)) => "reject_hunk",
                ("reject", None) | ("revert_diff", _) => "revert_diff",
                _ => op,
            };
            let payload = match &hunk_id {
                Some(h) => json!({ "diff_id": diff_id, "hunk_id": h }),
                None => json!({ "diff_id": diff_id }),
            };
            self.write_effect_outcome(&mut ack, name, &payload, outcome);
        }

        // The editor save. Same permission-gated, verifying applier the agent's edits take. A
        // policy that refuses workspace writes (the shipped default is Ask) does NOT end the save
        // here: the write is held at the security gate carrying the policy's own reason, so the
        // user can approve it, exactly like the other held effects. Any OTHER failure (the
        // base_hash conflict the applier raises when the file moved under the buffer is the one
        // that matters) refuses the ack, so the editor says the save was refused instead of
        // printing "saved <path>" over a write that never landed.
        if let (true, Some(payload)) = (effect_ok, save_action) {
            let outcome = self.save_file_effect(&payload).await;
            self.write_effect_outcome(&mut ack, "save_file", &payload, outcome);
        }

        // Static analysis side effect: run the model-free Tier1 oracle and publish the diagnostics
        // projection the Problems counter binds. A failure surfaces as an Error UiEvent.
        if let (true, Some(payload)) = (effect_ok, static_analysis_action) {
            if let Err(err) = self.handle_static_analysis_intent(&payload).await {
                self.effect_failed(&mut ack, "run_static_analysis", err.to_string());
            }
        }

        // Seal and publish a diff review receipt, once the intent is recorded.
        if let (true, Some(payload)) = (effect_ok, review_receipt_action) {
            if let Err(err) = self.handle_export_review_receipt_intent(&payload).await {
                self.effect_failed(&mut ack, "export_review_receipt", err.to_string());
            }
        }

        // Honest negative ack, last: the event IS in the log, but nothing here acts on this name,
        // so the caller is told so instead of being handed a false success.
        if let Some(name) = unhandled {
            ack.accepted = false;
            ack.message = Some(format!(
                "custom intent '{name}' is recorded but has no host handler"
            ));
        }
        Ok(ack)
    }

    /// A side effect that FAILED after the intent was recorded. The event IS in the
    /// log, but nothing landed, so the ack must not read as success: this publishes
    /// the Error UiEvent AND refuses the ack. Every side effect in [`Self::handle_intent`]
    /// reports through here, so no surface can print "saved" / "done" for work the host
    /// did not do (an 8-second error toast beside a success line is not a refusal).
    pub fn publish_plan(
        &self,
        session: &SessionId,
        plan: &hide_kernel::plan::schema::Plan,
        autonomy: Autonomy,
    ) -> Result<()> {
        let record = crate::plan_domain::PlanRecord::from_kernel(plan, autonomy);
        crate::plan_domain::store_and_publish(
            &self.services.key_value_store,
            &self.ui_bus,
            session,
            0,
            &record,
        )
    }

    /// The session's durable plan record, if one has been published.
    pub fn plan_get(&self, session: &SessionId) -> Option<crate::plan_domain::PlanRecord> {
        crate::plan_domain::PlanRecordStore::get(&self.services.key_value_store, session)
    }

    /// Dispatch a PlanCard custom intent (Stage 1, bible sec 14): mutate the
    /// session's durable plan record and republish the `plan` projection. Payload
    /// shapes (all carry `session_id`):
    ///
    /// * `approve_plan`   -> `{ session_id, step_id? }` (absent step_id = whole plan)
    /// * `edit_plan_step` -> `{ session_id, step_id, text }`
    /// * `reorder_plan`   -> `{ session_id, order: [step_id, ..] }`
    /// * `skip_step`      -> `{ session_id, step_id, reason? }`
    /// * `repair_step`    -> `{ session_id, step_id }`
    ///
    /// Errors when no plan is set for the session, a named step is unknown, or a
    /// reorder is not a permutation; the caller surfaces it as an Error UiEvent.
    pub(crate) fn permission_verdict_for(
        &self,
        tool_id: &str,
        args: &Value,
    ) -> hide_core::permission::PermissionVerdict {
        use hide_core::permission::{PermissionEngine, PermissionRequest};
        use hide_core::types::RiskLevel;
        let engine = SecurityServices::permission_engine(&self.services.config);
        let spec = self.tools.get(tool_id).map(|tool| tool.spec().clone());
        let capability_kind = spec
            .as_ref()
            .and_then(|s| s.capabilities_required.first().cloned())
            .unwrap_or_else(|| "tool.call".to_string());
        let risk = match spec.as_ref() {
            Some(s) if s.annotations.destructive => RiskLevel::High,
            Some(_) => RiskLevel::Low,
            None => RiskLevel::High,
        };
        let target = policy_target_from_args(tool_id, args);
        engine.evaluate(&PermissionRequest {
            capability_kind,
            target,
            risk,
            effects: Vec::new(),
            grant: None,
        })
    }
}
