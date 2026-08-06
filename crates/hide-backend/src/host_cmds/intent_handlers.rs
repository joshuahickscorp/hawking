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
    pub(crate) async fn handle_static_analysis_intent(&self, payload: &Value) -> Result<()> {
        let session = payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(SessionId::from)
            .unwrap_or_else(|| self.services.session());
        let mut sources: Vec<SourceFile> = payload
            .get("sources")
            .and_then(|v| v.as_array())
            .map(|rows| {
                rows.iter()
                    .filter_map(|r| {
                        Some(SourceFile::new(
                            r.get("path").and_then(|v| v.as_str())?,
                            r.get("text").and_then(|v| v.as_str())?,
                        ))
                    })
                    .collect()
            })
            .unwrap_or_default();
        if let Some(paths) = payload.get("paths").and_then(|v| v.as_array()) {
            let root = &self.services.config.workspace_root;
            for p in paths.iter().filter_map(|v| v.as_str()) {
                let abs = crate::connectors::workspace_resolve(root, p)?;
                let text = std::fs::read_to_string(&abs)
                    .map_err(|e| hide_core::error::HideError::Storage(format!("{p}: {e}")))?;
                sources.push(SourceFile::new(p, text));
            }
        }
        if sources.is_empty() {
            return Err(hide_core::error::HideError::Message(
                "run_static_analysis: give 'sources' or 'paths'".to_string(),
            ));
        }
        self.run_static_analysis(session, sources).await.map(|_| ())
    }

    /// Seed (or replace) the session's durable plan record from a live kernel
    /// plan and publish the `plan` projection. Used by plan intents and tests;
    /// fleet/agent kernels share the same [`crate::plan_domain::store_and_publish`]
    /// seam (product SubmitTurn is sole [`run_turn_core`]).
    pub(crate) async fn handle_plan_intent(&self, name: &str, payload: &Value) -> Result<()> {
        let missing = |field: &str| {
            hide_core::error::HideError::Message(format!("{name}: missing '{field}'"))
        };
        let session = payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| missing("session_id"))?;
        let session = SessionId::from(session);
        let mut record = self.plan_get(&session).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!("no plan for session {session}"))
        })?;
        let step_id = payload.get("step_id").and_then(|v| v.as_str());
        let ok = match name {
            "approve_plan" => record.approve(step_id),
            "edit_plan_step" => {
                let text = payload
                    .get("text")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("text"))?;
                let sid = step_id.ok_or_else(|| missing("step_id"))?;
                record.edit_step(sid, text)
            }
            "reorder_plan" => {
                let order: Vec<String> = payload
                    .get("order")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect()
                    })
                    .ok_or_else(|| missing("order"))?;
                record.reorder(&order)
            }
            "skip_step" => {
                let sid = step_id.ok_or_else(|| missing("step_id"))?;
                let reason = payload
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("skipped by user");
                record.skip_step(sid, reason)
            }
            "repair_step" => {
                let sid = step_id.ok_or_else(|| missing("step_id"))?;
                record.repair_failed_step(sid).is_some()
            }
            _ => return Ok(()),
        };
        if !ok {
            return Err(hide_core::error::HideError::Message(format!(
                "{name}: no matching step, or invalid order"
            )));
        }
        crate::plan_domain::store_and_publish(
            &self.services.key_value_store,
            &self.ui_bus,
            &session,
            0,
            &record,
        )
    }

    /// Dispatch a durable Goal / Checkpoint custom intent to the corresponding
    /// tested host method (bible sec 14, sec 15.4). Payload shapes:
    ///
    /// * `goal_set`         -> `{ session_id, condition, acceptance: [oracle,..] }`
    /// * `goal_clear`       -> `{ session_id }`
    /// * `checkpoint_create`-> `{ session_id, at_event?, label? }`
    /// * `checkpoint_restore`-> `{ checkpoint_id }`
    /// * `checkpoint_rewind` -> `{ checkpoint_id, target: "code"|"conversation"|"both" }`
    /// * `checkpoint_replay` -> `{ checkpoint_id }`
    /// * `checkpoint_fork`   -> `{ checkpoint_id }`
    /// * `checkpoint_compare`-> `{ checkpoint_id, session_id }` or `{ session_id, other_session_id }`
    /// * `checkpoint_inspect`-> `{ checkpoint_id }`
    ///
    /// A malformed payload (e.g. a missing session_id) errors; the caller surfaces
    /// it as an Error UiEvent. The methods themselves emit the success UiEvents.
    pub(crate) async fn handle_goal_checkpoint_intent(
        &self,
        name: &str,
        payload: &Value,
    ) -> Result<()> {
        let missing = |field: &str| {
            hide_core::error::HideError::Message(format!("{name}: missing '{field}'"))
        };
        match name {
            "goal_set" => {
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                let condition = payload
                    .get("condition")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("condition"))?;
                let acceptance = payload
                    .get("acceptance")
                    .and_then(|v| v.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                self.goal_set(SessionId::from(session), condition, acceptance)?;
            }
            "goal_clear" => {
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                self.goal_clear(&SessionId::from(session))?;
            }
            "checkpoint_create" => {
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                let at_event = payload
                    .get("at_event")
                    .and_then(|v| v.as_str())
                    .map(EventId::from);
                let label = payload
                    .get("label")
                    .and_then(|v| v.as_str())
                    .unwrap_or("checkpoint")
                    .to_string();
                self.checkpoint_create(SessionId::from(session), at_event.as_ref(), label)
                    .await?;
            }
            "checkpoint_restore" => {
                let checkpoint_id = payload
                    .get("checkpoint_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("checkpoint_id"))?;
                self.checkpoint_restore(checkpoint_id).await?;
            }
            "checkpoint_rewind" => {
                let checkpoint_id = payload
                    .get("checkpoint_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("checkpoint_id"))?;
                // No default: "both" is the widest, most destructive domain, so an
                // omitted target is REFUSED rather than guessed.
                let target_str = payload
                    .get("target")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("target"))?;
                let target = RewindTarget::parse(target_str).ok_or_else(|| {
                    hide_core::error::HideError::Message(format!(
                        "{name}: unknown target '{target_str}' (code|conversation|both)"
                    ))
                })?;
                self.checkpoint_rewind(checkpoint_id, target).await?;
            }
            "checkpoint_replay" => {
                let checkpoint_id = payload
                    .get("checkpoint_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("checkpoint_id"))?;
                self.checkpoint_replay(checkpoint_id).await?;
            }
            "checkpoint_fork" => {
                let checkpoint_id = payload
                    .get("checkpoint_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("checkpoint_id"))?;
                self.checkpoint_fork(checkpoint_id).await?;
            }
            "checkpoint_compare" => {
                // Two shapes: checkpoint-vs-session (checkpoint_id + session_id) or
                // session-vs-session (session_id + other_session_id).
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                if let Some(checkpoint_id) = payload.get("checkpoint_id").and_then(|v| v.as_str()) {
                    self.compare_to_checkpoint(checkpoint_id, &SessionId::from(session))
                        .await?;
                } else {
                    let other = payload
                        .get("other_session_id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| missing("checkpoint_id or other_session_id"))?;
                    self.compare_session_code(&SessionId::from(session), &SessionId::from(other))
                        .await?;
                }
            }
            "checkpoint_inspect" => {
                let checkpoint_id = payload
                    .get("checkpoint_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("checkpoint_id"))?;
                self.checkpoint_inspect(checkpoint_id).await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Deliver a mid-turn STEER to a running run (bible ch.02 sec 4.3.2, census
    /// priority 6). This is the true end-to-end wiring the shipped `redirect_run`
    /// gesture was missing: it (1) signals a real [`Interrupt::Steer`] onto the
    /// shared [`InterruptHub`] keyed by `run_id` -- exactly how `CancelRun`/
    /// `PauseRun`/`ResumeRun` route to `Abort`/`Pause`/`Resume` -- so the kernel
    /// loop drains it (`drain_into_kernel`) and the Governor folds the text into
    /// `state.steer`, prepended to the next planning step's prompt; and (2)
    /// persists a durable `turn.steer` event (+ a `turn_steer` UiEvent) so the
    /// redirect is auditable and shows in the projection. Shared by the Wire-A
    /// `redirect_run` intent and the protocol `turn/steer` RPC.
    ///
    /// The session defaults to the control session when the caller does not name
    /// one (the FE gesture carries only `{ run_id, text }`); a caller that knows
    /// the run's session passes it so the steer event lands on that thread.
    pub(crate) async fn handle_memory_workspace_env_intent(
        &self,
        name: &str,
        payload: &Value,
    ) -> Result<()> {
        let missing = |field: &str| {
            hide_core::error::HideError::Message(format!("{name}: missing '{field}'"))
        };
        match name {
            "memory_add" => {
                let draft = parse_memory_draft(payload)?;
                let record = self.memory_add(draft)?;
                self.publish_memory("memory_added", &record);
            }
            "memory_supersede" => {
                let old_id = payload
                    .get("old_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("old_id"))?;
                let replacement = payload
                    .get("replacement")
                    .ok_or_else(|| missing("replacement"))?;
                let (_old, new) =
                    self.memory_supersede(old_id, parse_memory_draft(replacement)?)?;
                self.publish_memory("memory_superseded", &new);
            }
            "memory_record_outcome" => {
                let memory_id = payload
                    .get("memory_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("memory_id"))?;
                let success = payload
                    .get("success")
                    .and_then(|v| v.as_bool())
                    .ok_or_else(|| missing("success"))?;
                let record = self.memory_record_outcome(memory_id, success)?;
                self.publish_memory("memory_outcome_recorded", &record);
            }
            "memory_revalidate" => {
                let target = if let Some(id) = payload.get("memory_id").and_then(|v| v.as_str()) {
                    RevalidateTarget::record(id)
                } else if let Some(scope) = payload.get("scope") {
                    RevalidateTarget::scope(serde_json::from_value(scope.clone()).map_err(|e| {
                        hide_core::error::HideError::Message(format!("{name}: bad scope: {e}"))
                    })?)
                } else {
                    return Err(missing("memory_id or scope"));
                };
                let repo_root = payload
                    .get("repo_root")
                    .and_then(|v| v.as_str())
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(|| self.services.config.workspace_root.clone());
                let verdicts = self.memory_revalidate(target, &repo_root)?;
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::Custom(json!({
                        "kind": "memory_revalidated",
                        "verdicts": serde_json::to_value(&verdicts).unwrap_or_else(|_| json!([])),
                    })),
                });
            }
            "goal_evaluate" => {
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                let session = SessionId::from(session);
                // Routes to the tested evaluator (deterministic, model-free); it
                // advances + surfaces a Met transition itself. Surface the verdict
                // for every outcome so the FE sees the acceptance result.
                let verdict = self.goal_evaluate(&session).await?;
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: Some(session),
                    kind: UiEventKind::Custom(json!({
                        "kind": "goal_evaluated",
                        "verdict": serde_json::to_value(&verdict).unwrap_or_else(|_| json!({})),
                    })),
                });
            }
            "workspace_set_repo_trust" => {
                let repo_id = payload
                    .get("repo_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("repo_id"))?;
                let trust: TrustState = payload
                    .get("trust")
                    .cloned()
                    .map(serde_json::from_value)
                    .transpose()
                    .map_err(|e| {
                        hide_core::error::HideError::Message(format!("{name}: bad trust: {e}"))
                    })?
                    .ok_or_else(|| missing("trust"))?;
                // The add-folder flow is the ONE way a repo enters the graph from the app, and the
                // trust decision is where it arrives: `workspace_add_repo` has no wire name, so a
                // trust call used to hit a repo that was never there and return `Ok(None)` with no
                // event and no error, leaving the control pending forever. The folder's own path
                // comes with the decision, so the node is created here (untrusted, per
                // trust-before-config) and then the decision is applied to it. Without a path there
                // is nothing to create, so this refuses instead of no-opping.
                if self.workspace_repo(repo_id).is_none() {
                    let root_path = payload
                        .get("root_path")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| missing("root_path"))?;
                    self.workspace_add_repo(RepoNode::new(repo_id, root_path))?;
                }
                let repo = self
                    .workspace_set_repo_trust(repo_id, trust)?
                    .ok_or_else(|| unknown_repo(repo_id))?;
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::Custom(json!({
                        "kind": "repo_trust_set",
                        "repo": serde_json::to_value(&repo).unwrap_or_else(|_| json!({})),
                    })),
                });
            }
            "environment_switch" => {
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                let env_id = payload
                    .get("env_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("env_id"))?;
                let reason = payload
                    .get("reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("environment switch");
                // Routes to the tested method: it appends the durable
                // `environment.switch` event, advances current_env, and emits its
                // own `environment_switch` UiEvent.
                self.environment_switch(SessionId::from(session), env_id, reason)
                    .await?;
            }
            _ => {}
        }
        Ok(())
    }

    /// Publish a memory-lifecycle UiEvent carrying the record (bible sec 21-22).
    /// The memory ledger is durable in KV; this surfaces the change on Wire-B so
    /// the Context Stack reflects it (parity with the goal/checkpoint publishers).
    pub(crate) fn spawn_worktree_add(&self, branch: Option<&str>) {
        let raw = branch.unwrap_or("session");
        let slug: String = raw
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                    c.to_ascii_lowercase()
                } else {
                    '-'
                }
            })
            .collect();
        let slug = slug.trim_matches('-');
        let slug = if slug.is_empty() { "session" } else { slug };
        let root = self.services.config.workspace_root.clone();
        let repo = root
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("repo")
            .to_string();
        let dest = root
            .parent()
            .map(|p| p.join(format!("{repo}-{slug}")))
            .unwrap_or_else(|| root.join(format!(".hide-worktree-{slug}")));
        let argv = vec![
            "git".to_string(),
            "worktree".to_string(),
            "add".to_string(),
            "-b".to_string(),
            format!("hide/{slug}"),
            dest.to_string_lossy().to_string(),
        ];
        self.spawn_exec(argv, None);
    }

    /// Mint a fresh session id and publish an idle `turn` projection under it, so the FE adopts the new
    /// session (its event router tracks `session_id` off any event) and the transcript starts clean.
    pub(crate) async fn handle_surface_intent(&self, name: &str, payload: &Value) -> Result<()> {
        match name {
            "switch_surface" => {
                let surface_name =
                    payload
                        .get("surface")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| {
                            hide_core::error::HideError::Message(
                                "switch_surface requires surface".into(),
                            )
                        })?;
                let surface = crate::surfaces::SurfaceGraphService::parse_surface(surface_name)
                    .map_err(hide_core::error::HideError::Message)?;
                let view = self.surfaces.switch_surface(surface)?;
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: Some(SessionId::from(view.session_id.as_str())),
                    kind: UiEventKind::Custom(json!({
                        "kind": "surface_switched",
                        "surface": view.active_surface,
                        "session_id": view.session_id,
                    })),
                });
                Ok(())
            }
            "handoff_create" => {
                let kind_name = payload
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        hide_core::error::HideError::Message("handoff_create requires kind".into())
                    })?;
                let kind = crate::surfaces::SurfaceGraphService::parse_kind(kind_name)
                    .map_err(hide_core::error::HideError::Message)?;
                let claims = crate::surfaces::claims_from_payload(
                    payload.get("claims").unwrap_or(&Value::Array(vec![])),
                )
                .map_err(hide_core::error::HideError::Message)?;
                let exclusions = crate::surfaces::exclusions_from_payload(
                    payload
                        .get("deliberately_excludes")
                        .unwrap_or(&Value::Array(vec![])),
                )
                .map_err(hide_core::error::HideError::Message)?;
                let body = payload.get("body").cloned().unwrap_or_else(|| json!({}));
                let actor = payload
                    .get("actor")
                    .and_then(|v| v.as_str())
                    .unwrap_or("user");
                let now = hide_core::ids::now_ms();
                let capsule = self
                    .surfaces
                    .handoff_create(kind, claims, exclusions, body, actor, now)
                    .await?;
                // Double-check the safety property at the host boundary.
                if capsule.try_extract_capability().is_ok() {
                    return Err(hide_core::error::HideError::PolicyDenied(
                        "handoff_create produced a capability-bearing capsule".into(),
                    ));
                }
                Ok(())
            }
            "handoff_receive" => {
                let capsule_id = payload
                    .get("capsule_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        hide_core::error::HideError::Message(
                            "handoff_receive requires capsule_id".into(),
                        )
                    })?;
                let _view = self.surfaces.handoff_receive(capsule_id).await?;
                Ok(())
            }
            other => Err(hide_core::error::HideError::Message(format!(
                "unknown surface intent {other}"
            ))),
        }
    }

    /// Load a past session: scan its recorded events, map them to UiEvents, and republish them on the
    /// live bus so the FE (which adopts the session off any event's `session_id`) switches to it and
    /// re-renders the transcript. Every event is real, read straight from the log; nothing is fabricated.
    pub(crate) fn spawn_open_session(&self, sid: SessionId) {
        let replay = self.replay.clone();
        let bus = Arc::clone(&self.ui_bus);
        tokio::spawn(async move {
            match replay.ui_events(Some(sid.clone()), None, None).await {
                Ok(events) => {
                    for ev in events {
                        bus.publish(ev);
                    }
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(sid),
                        kind: UiEventKind::RuntimeStatus {
                            status: "error".to_string(),
                            detail: Some(format!("could not load session: {err}")),
                        },
                    });
                }
            }
        });
    }

    /// Execute an accepted `RunCommand` and stream its stdout and stderr back as
    /// `tool_progress` UiEvents (the terminal mirrors those). The command runs
    /// SANDBOX-confined through the process surface, inheriting the same OS
    /// confinement the agent's `shell.run` tool gets (Trace D (a)).
    /// Returns the gate id when the command was PARKED instead of started, so the caller can mark
    /// the ack `held`. Without that return the ack read `accepted` for a command the host refused
    /// to run and the terminal printed "started ... (sandbox confined)" for it.
    pub(crate) fn spawn_command_run(
        &self,
        argv: Vec<String>,
        cwd: Option<String>,
    ) -> Result<Option<String>> {
        if argv.is_empty() {
            return Ok(None);
        }
        // Security gate: a genuinely destructive command is NOT dropped. It is parked under a unique
        // gate id and surfaced as a `SecurityGate` UiEvent; the user's `approve_gate` (with that id)
        // releases and runs it, `deny_gate` drops it. Ordinary dev commands run immediately.
        if let Some(reason) = dangerous_command(&argv) {
            let gate = self.hold_at_gate(
                PendingAction::Command {
                    argv: argv.clone(),
                    cwd: cwd.clone(),
                },
                format!("blocked: {} ({})", argv.join(" "), reason),
            )?;
            return Ok(Some(gate));
        }
        self.spawn_supervised(argv, cwd);
        Ok(None)
    }

    /// Run a gate-cleared terminal command (a safe command, or a user-approved
    /// one) through the sandboxed process surface. Streams stdout/stderr back as
    /// `tool_progress`; OS-confined (fail-closed), so an interactive terminal
    /// command can never write outside the workspace or reach the network.
    pub(crate) fn spawn_supervised(&self, argv: Vec<String>, cwd: Option<String>) {
        let owner = self.commands.control_session().to_string();
        let mut spec = StartSpec::command(argv, cwd);
        spec.owner = Some(owner);
        self.processes.start(spec, &self.shell_config());
    }

    /// Legacy raw command runner (UNSANDBOXED). Retained ONLY for the internal,
    /// trusted `spawn_worktree_add` path, which must `git worktree add` into a
    /// SIBLING directory outside the workspace root (a write the sandbox would
    /// deny). User-facing terminal commands go through `spawn_supervised`.
    pub(crate) fn spawn_exec(&self, argv: Vec<String>, cwd: Option<String>) {
        let ui_bus = self.ui_bus.clone();
        let root = self.services.config.workspace_root.clone();
        tokio::spawn(async move {
            exec_command_streamed(ui_bus, root, argv, cwd).await;
        });
    }

    /// The `ShellConfig` the process surface confines with: writes scoped to the
    /// workspace root, the absolute `.hide/log` write-deny threaded in. Mirrors the
    /// posture `hide_kernel::tooling::shell` renders for `shell.run`.
    pub(crate) fn hold_at_gate(&self, action: PendingAction, message: String) -> Result<String> {
        let gate = self.gate_book.hold(action).ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "not held: {} approvals are already awaiting a decision; answer or deny them first",
                GateBook::CAP
            ))
        })?;
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: None,
            kind: UiEventKind::SecurityGate {
                gate: gate.clone(),
                message,
            },
        });
        Ok(gate)
    }

    /// The outcome rule for an intent whose effect writes the workspace.
    ///
    /// `PolicyDenied` is not a failure, it is "the human has not said yes yet", so the effect is
    /// HELD at the gate under its own name and can be approved; anything else refuses the ack.
    /// Shared, because holding only the arm somebody noticed (the editor save) left every sibling
    /// write - the per-hunk reject the review surface's undo is - permanently refused with no
    /// approval path on the shipped `Ask` default.
    pub(crate) fn write_effect_outcome(
        &self,
        ack: &mut IntentAck,
        name: &str,
        payload: &Value,
        outcome: Result<()>,
    ) {
        match outcome {
            Ok(()) => {}
            Err(hide_core::error::HideError::PolicyDenied(reason)) => {
                match self.hold_at_gate(
                    PendingAction::Intent {
                        name: name.to_string(),
                        payload: payload.clone(),
                    },
                    reason.clone(),
                ) {
                    Ok(gate) => {
                        ack.held = true;
                        ack.message = Some(format!("held for approval: gate={gate} ({reason})"));
                    }
                    Err(err) => self.effect_failed(ack, name, err.to_string()),
                }
            }
            Err(err) => self.effect_failed(ack, name, err.to_string()),
        }
    }

    /// Approve a held gated action: release it from the book and run it (bypassing the gate, since
    /// the user approved). A `Command` stays SANDBOX-confined (approval clears the deny-list gate,
    /// not the OS confinement); an `Intent` runs the effect its `ApprovalPolicy::Ask` spec held
    /// back. A no-op if the gate id is unknown (already taken, denied, or evicted).
    pub(crate) async fn approve_gate(&self, gate: &str) -> Result<()> {
        match self.gate_book.take(gate) {
            Some(PendingAction::Command { argv, cwd }) => {
                self.spawn_supervised(argv, cwd);
                Ok(())
            }
            // The approval is only as good as the effect it released: a released write that the
            // applier then refuses (a `base_hash` conflict, most often) is NOT an approved action
            // that happened, so the error travels back to the ack instead of being an 8-second
            // toast beside a surface that closed as success.
            Some(PendingAction::Intent { name, payload }) => {
                self.run_approved_intent(&name, &payload).await
            }
            // Unknown, already answered, or dropped: nothing ran, so nothing may read as accepted.
            None => Err(hide_core::error::HideError::NotFound(format!(
                "gate {gate} is not awaiting a decision (already answered, denied, or never held)"
            ))),
        }
    }

    /// Run the effect of a custom intent that was held at the gate because its `CommandSpec`
    /// declares [`ApprovalPolicy::Ask`]. Routes to the SAME handler the un-gated path uses; the
    /// intent itself was already recorded in the event log when it arrived.
    ///
    /// EVERY name `requires_approval` returns true for must have an arm here, or approving the gate
    /// is the only thing that ever happens and the command is permanently non-functional. The test
    /// `every_ask_command_has_a_release_handler` walks the catalog and fails if one is missing.
    ///
    /// The WHOLE body runs inside [`crate::tools::with_approved_writes`], the one approved-write
    /// scope, so every arm's effect (and everything it calls: `revert_diff` and the rewind's
    /// per-hunk peel both bottom out in `inverse_write`, `save_file` in the `fs` connector) sees the
    /// approval the user just gave. It used to be relaxed in the `save_file` arm alone, which left
    /// every sibling releasing into a `PolicyDenied` on the shipped default.
    pub(crate) async fn released_effect(&self, name: &str, payload: &Value) -> Result<()> {
        match name {
            "create_worktree" => {
                self.spawn_worktree_add(payload.get("branch").and_then(|v| v.as_str()));
                Ok(())
            }
            "revert_diff" => {
                let diff_id = payload
                    .get("diff_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        hide_core::error::HideError::Message(
                            "revert_diff: missing 'diff_id'".to_string(),
                        )
                    })?;
                self.revert_diff(diff_id).await.map(|_| ())
            }
            "workspace_set_repo_trust" => {
                self.handle_memory_workspace_env_intent(name, payload).await
            }
            // The user approved THIS write, so it runs through the same one save path the ungated
            // save takes (same path confinement, same verifying applier, same `base_hash` conflict
            // guard, same diff capture); the approval is carried by the scope this whole function
            // runs in.
            "save_file" => self.save_file_effect(payload).await,
            // The review surface's undo. It writes (the inverse write that puts the pre-image
            // back), so on the shipped `Ask` default it arrives here through the same hold-and-
            // approve path the save takes.
            "reject_hunk" => {
                let diff_id = payload
                    .get("diff_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        hide_core::error::HideError::Message(
                            "reject_hunk: missing 'diff_id'".to_string(),
                        )
                    })?;
                let hunk_id = payload
                    .get("hunk_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        hide_core::error::HideError::Message(
                            "reject_hunk: missing 'hunk_id'".to_string(),
                        )
                    })?;
                self.reject_hunk(diff_id, hunk_id).await.map(|_| ())
            }
            "checkpoint_restore" | "checkpoint_rewind" => {
                self.handle_goal_checkpoint_intent(name, payload).await
            }
            // The lease grant. It runs here and only here, so the human approval at the gate IS the
            // grant condition; nothing installs a lease without passing through this release.
            "grant_write_lease" => self.handle_grant_write_lease(payload).await,
            other => Err(hide_core::error::HideError::Message(format!(
                "{other}: approval-gated but no release handler"
            ))),
        }
    }

    /// The catalog command an intent will actually EFFECT, with the payload that effect needs.
    ///
    /// The approval policy hangs off THIS, not off the wire name that carried the request, because
    /// one effect can be reached by more than one payload shape: `reject_diff` with no `hunk_id` is
    /// the same whole-diff on-disk revert as `revert_diff`, and reading the policy off the name
    /// alone let the ungated button next to the gated one perform the gated effect.
    pub(crate) fn requires_approval(name: &str) -> bool {
        use hide_protocol::command::ApprovalPolicy;
        static ASK: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
        ASK.get_or_init(|| {
            hide_protocol::command::command_catalog()
                .into_iter()
                .filter(|s| s.approval_policy == ApprovalPolicy::Ask)
                .map(|s| s.id)
                .collect()
        })
        .iter()
        .any(|n| n == name)
    }

    /// Refuse an approval-gated EFFECT that did not arrive through a released gate.
    ///
    /// Enforced at the effect, not at a transport, because the intent boundary is not the only way
    /// in: `POST /v1/hide/rpc` reaches `checkpoint_restore` straight off `BackendHost`, skipping
    /// `handle_intent`, `effect_command`, `requires_approval` and the gate book entirely, and so
    /// does any in-process caller. Guarding the one transport somebody remembered is what let a
    /// declared-`Ask` command run unapproved. `run_approved_intent` runs the release inside
    /// [`crate::tools::with_approved_writes`], which is exactly the released-gate scope, so the
    /// approved path passes and every other path is refused. The policy itself is still read off
    /// the ONE catalog, so there is no second list of gated names.
    pub(crate) fn spawn_submit_turn_generation(&self, session_id: SessionId, prompt: String) {
        let run_id = RunId::new();
        match self.runtime_base_url() {
            Some(base_url) => {
                // Register the run with the interrupt hub so control intents can
                // reach it (the generation task polls it cooperatively).
                // Sole product SubmitTurn path: always `generate_submit_turn` →
                // `run_turn_core`. The incomplete HIDE_KERNEL_TURN dual branch is
                // gone; AgentKernel remains for fleet/tooling via build_turn_kernel.
                let ui_bus = self.ui_bus.clone();
                let role_registry = self.services.role_registry.clone();
                let event_log = self.services.event_log.clone();
                let code_index = self.services.code_index.clone();
                let memory = self.services.memory_store.clone();
                let classed_memory = self.services.classed_memory.clone();
                let interrupts = self.interrupts.clone();
                let run = run_id.clone();
                let repo_instructions = self.services.repo_instructions.clone();
                tokio::spawn(async move {
                    if let Err(e) = generate_submit_turn(
                        event_log,
                        role_registry,
                        code_index,
                        memory,
                        classed_memory,
                        ui_bus.clone(),
                        interrupts,
                        run,
                        session_id.clone(),
                        base_url,
                        prompt,
                        repo_instructions,
                    )
                    .await
                    {
                        // Surface the failure on the same typed Wire-B channel;
                        // never swallow it.
                        ui_bus.publish(UiEvent {
                            seq: 0,
                            session_id: Some(session_id),
                            kind: UiEventKind::Error {
                                code: "generation".to_string(),
                                message: e.to_string(),
                            },
                        });
                    }
                });
            }
            None => {
                // No model online: surface "model offline" as a real UiEvent.
                let status = self
                    .runtime_state()
                    .map(|s| format!("{s:?}").to_lowercase())
                    .unwrap_or_else(|| "down".to_string());
                let detail = match self.runtime.is_some() {
                    true => "runtime not ready; reconnect when it reports ready".to_string(),
                    false => "no model configured (set HIDE_MODEL_WEIGHTS)".to_string(),
                };
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: Some(session_id),
                    kind: UiEventKind::RuntimeStatus {
                        status,
                        detail: Some(detail),
                    },
                });
            }
        }
    }

    pub(crate) async fn handle_process_intent(
        &self,
        name: &str,
        payload: &Value,
    ) -> std::result::Result<(), String> {
        let id = payload.get("process").and_then(|v| v.as_str());
        let named = || id.ok_or_else(|| format!("{name}: missing 'process'"));
        match name {
            "pty_input" => {
                let data = payload
                    .get("data")
                    .or_else(|| payload.get("text"))
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| "pty_input: missing 'data'".to_string())?;
                self.processes.write_stdin(id, data).await
            }
            "pty_resize" => {
                let cols = payload.get("cols").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                let rows = payload.get("rows").and_then(|v| v.as_u64()).unwrap_or(0) as u16;
                self.processes.resize(id, cols, rows)
            }
            // Re-attaching replays the buffered output onto the bus under the attaching session, so
            // a re-navigated terminal comes back with its scrollback instead of an empty pane.
            "attach_process" => {
                let id = named()?;
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .map(SessionId::from)
                    .unwrap_or_else(|| self.services.session());
                let lines = self
                    .attach_process(id, session.clone())
                    .ok_or_else(|| format!("unknown process {id}"))?;
                self.publish_custom(
                    Some(session),
                    json!({ "kind": "process_attached", "process": id, "lines": lines }),
                );
                Ok(())
            }
            "stop_process" => {
                let id = named()?;
                if !self.stop_process(id) {
                    return Err(format!("unknown process {id}"));
                }
                self.publish_custom(None, json!({ "kind": "process_stopped", "process": id }));
                Ok(())
            }
            "capture_process_artifact" => {
                let id = named()?;
                let blob = self
                    .capture_process_artifact(id)
                    .map_err(|e| e.to_string())?;
                self.publish_custom(
                    None,
                    json!({
                        "kind": "process_artifact",
                        "process": id,
                        "artifact": serde_json::to_value(&blob).unwrap_or(Value::Null),
                    }),
                );
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Publish a seq-0 `Custom` UiEvent on the live bus.
    pub(crate) async fn save_file_effect(&self, payload: &Value) -> Result<()> {
        let path = payload
            .get("path")
            .and_then(|v| v.as_str())
            .ok_or_else(|| hide_core::error::HideError::Config("missing path".to_string()))?;
        let content = payload
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| hide_core::error::HideError::Config("missing content".to_string()))?;
        // Same confinement the connector read path uses: `..`, absolute and prefix components are
        // refused before the dispatcher ever sees the call.
        let abs = crate::connectors::workspace_resolve(&self.services.config.workspace_root, path)?;
        let mut args = json!({ "path": abs.to_string_lossy(), "content": content });
        // `base_hash` (the blake3 of the text the caller read) is passed through when supplied, so
        // a concurrently-changed file CONFLICTS instead of being clobbered.
        if let Some(base) = payload.get("base_hash").and_then(|v| v.as_str()) {
            args["base_hash"] = json!(base);
        }
        let session = payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(SessionId::from)
            .unwrap_or_else(|| self.services.session());
        let run = Self::editor_run(&session);
        let result = self
            .dispatch_tool(session, Some(run), ToolCall::new("edit.write_file", args))
            .await?;
        // `Tool`, not `PolicyDenied`: the permission verdict is the `?` above. This is the applier
        // refusing (a `base_hash` conflict, most often), which no approval fixes, so it must not be
        // mistaken for something to hold at a gate.
        if result.status != ToolStatus::Ok {
            return Err(hide_core::error::HideError::Tool(format!(
                "write of {path} refused: {}",
                result
                    .error
                    .as_ref()
                    .map(|e| format!("{}: {}", e.code, e.message))
                    .unwrap_or_else(|| "rejected by the applier".to_string())
            )));
        }
        Ok(())
    }

    /// Dispatch a tool ATTRIBUTED to a session and (optionally) a run.
    ///
    /// A thin caller of the recorded path: the durable `tool.call`/`tool.result` pair, the live
    /// `ToolProgress`, and the reviewable/revertible diff an `edit.*` write produces are recorded
    /// by [`DispatchRecorder`] hanging off the dispatcher itself, so the kernel agent - which holds
    /// the dispatcher directly and never enters this function - records exactly the same things.
    /// All this adds is the attribution: without it a dispatch is recorded against the default
    /// session with no run, which is honest but ungrouped.
    pub(crate) async fn handle_search_intent(
        &self,
        payload: &Value,
    ) -> Result<Vec<crate::replay::TranscriptHit>> {
        let scopes = payload.get("scopes");
        // A field is read from the top level, falling back to `scopes`.
        let field = |name: &str| {
            payload
                .get(name)
                .or_else(|| scopes.and_then(|s| s.get(name)))
        };
        let mut query = crate::replay::TranscriptQuery {
            text: payload
                .get("query")
                .or_else(|| payload.get("text"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            ..Default::default()
        };
        query.session_id = field("session_id")
            .and_then(|v| v.as_str())
            .map(SessionId::from);
        query.kind = field("kind").and_then(|v| v.as_str()).map(str::to_string);
        query.role = field("role").and_then(|v| v.as_str()).map(str::to_string);
        query.since_ts = field("since_ts").and_then(|v| v.as_u64());
        query.until_ts = field("until_ts").and_then(|v| v.as_u64());
        query.limit = payload
            .get("limit")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize);
        self.search_transcript(&query).await
    }

    /// Surface transcript-search hits to the FE as a `search_results` UiEvent
    /// (echoing the query so the palette can correlate the response).
    pub(crate) async fn handle_grant_write_lease(&self, payload: &Value) -> Result<()> {
        let repo_id = payload
            .get("repo_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                hide_core::error::HideError::Message(
                    "grant_write_lease: missing 'repo_id'".to_string(),
                )
            })?;
        let repo = self
            .workspace_repo(repo_id)
            .ok_or_else(|| unknown_repo(repo_id))?;
        if repo.trust != TrustState::Trusted {
            return Err(hide_core::error::HideError::PolicyDenied(format!(
                "{repo_id} is not trusted; trust the repository before granting it a write lease"
            )));
        }
        let scopes = match payload.get("scopes").and_then(|v| v.as_array()) {
            Some(rels) if !rels.is_empty() => rels
                .iter()
                .filter_map(|v| v.as_str())
                .map(|rel| crate::connectors::workspace_resolve(&repo.root_path, rel))
                .collect::<Result<Vec<_>>>()?,
            _ => vec![repo.root_path.clone()],
        };
        let lease = crate::tools::install_write_lease(crate::tools::WriteLease {
            lease_id: hide_core::ids::GrantId::new().as_str().to_string(),
            repo_id: repo_id.to_string(),
            session_id: payload
                .get("session_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            run_id: payload
                .get("run_id")
                .and_then(|v| v.as_str())
                .map(str::to_string),
            scopes,
            granted_ms: hide_core::ids::now_ms(),
        });
        publish_write_lease(&self.ui_bus, Some(&lease), "granted");
        Ok(())
    }

    /// The lease in force, if any (test / inspection / the status read).
    pub(crate) async fn inverse_write(
        &self,
        session: &SessionId,
        file: &str,
        before: &str,
        after: &str,
    ) -> Result<()> {
        let after_hash = blake3::hash(after.as_bytes()).to_hex().to_string();
        // Through the same recorded path as every other write (attributed, with no run: an undo is
        // a tool step in the timeline, not a new reviewable hunk of its own). Recorded hunk paths
        // are workspace-relative, so the applier is handed the absolute spelling.
        let path = if Path::new(file).is_absolute() {
            file.to_string()
        } else {
            self.services
                .config
                .workspace_root
                .join(file)
                .to_string_lossy()
                .into_owned()
        };
        let result = self
            .dispatch_tool(
                session.clone(),
                None,
                ToolCall::new(
                    "edit.write_file",
                    json!({ "path": path, "content": before, "base_hash": after_hash }),
                ),
            )
            .await?;
        if result.status != ToolStatus::Ok {
            return Err(hide_core::error::HideError::Message(format!(
                "revert of {file} failed: {}",
                tool_result_summary(&result)
            )));
        }
        Ok(())
    }

    /// Append a durable `diff.*` event carrying the diff projection + the target
    /// hunk's provenance (when a single hunk is addressed).
    pub(crate) async fn record_diff_event(
        &self,
        proposal: &DiffProposal,
        kind: &str,
        hunk_id: Option<&str>,
    ) -> Result<()> {
        let provenance = hunk_id
            .and_then(|id| proposal.hunk(id))
            .map(|h| serde_json::to_value(&h.provenance).unwrap_or(Value::Null));
        self.services
            .event_log
            .append(NewEvent::system(
                proposal.session_id.clone(),
                kind,
                json!({
                    "diff_id": proposal.diff_id,
                    "run_id": proposal.run_id,
                    "hunk_id": hunk_id,
                    "provenance": provenance,
                    "proposal": serde_json::to_value(proposal).unwrap_or(Value::Null),
                }),
            ))
            .await?;
        Ok(())
    }

    /// Republish the diff projection onto the Wire-B bus so the FE HunkReview
    /// control re-renders with the current per-hunk status. The ONE producer of the
    /// diff surface: every path that creates a proposal or flips a hunk status
    /// (`record_edit_diff`, `apply_diff`, `apply_hunk`, `reject_hunk`, `revert_diff`)
    /// routes through here.
    ///
    /// This used to publish an untyped `Custom{kind:"diff"}` that the frontend routes
    /// nowhere, so it decayed into a truncated JSON toast and the whole review surface
    /// had no live data at all. It now publishes the two projections the surface
    /// actually reads (see [`diff_projections`]).
    pub(crate) async fn handle_export_review_receipt_intent(&self, payload: &Value) -> Result<()> {
        let diff_id = payload
            .get("diff_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                hide_core::error::HideError::Message(
                    "export_review_receipt: missing 'diff_id'".to_string(),
                )
            })?;
        let proposal = self
            .diff_get(diff_id)
            .ok_or_else(|| unknown_diff(diff_id))?;
        let session = payload
            .get("session_id")
            .and_then(|v| v.as_str())
            .map(SessionId::from)
            .unwrap_or_else(|| proposal.session_id.clone());
        let (before, after): (Vec<VerificationReceipt>, Vec<VerificationReceipt>) = self
            .verification_receipts(&session)
            .await?
            .into_iter()
            .map(|r| r.receipt)
            .partition(|r| r.started_ms < proposal.created_ms);
        let receipt = self
            .export_diff_review_receipt(diff_id, before, after)
            .await?;
        self.publish_custom(
            Some(session),
            json!({
                "kind": "diff_review_receipt",
                "record": serde_json::to_value(&receipt).unwrap_or(Value::Null),
            }),
        );
        Ok(())
    }

    /// Every sealed diff review receipt recorded for a session, in log order.
    pub(crate) async fn handle_background_intent(&self, name: &str, payload: &Value) -> Result<()> {
        let missing = |field: &str| {
            hide_core::error::HideError::Message(format!("{name}: missing '{field}'"))
        };
        match name {
            "promote_run" => {
                let run_id = payload
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("run_id"))?;
                let session = payload
                    .get("session_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("session_id"))?;
                let goal_id = payload
                    .get("goal_id")
                    .and_then(|v| v.as_str())
                    .map(str::to_string);
                let budget = payload
                    .get("budget")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                self.promote_run_to_background(
                    RunId::from(run_id),
                    SessionId::from(session),
                    goal_id,
                    budget,
                )
                .await?;
            }
            "resume_run_foreground" => {
                let job_id = payload
                    .get("job_id")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| missing("job_id"))?;
                self.resume_background_job_in_foreground(job_id).await?;
            }
            _ => {}
        }
        Ok(())
    }

    // --- Outcome-governed durable MEMORY + revalidation (bible sec 21-22, sec 78.1 #16) ---

    /// Add a durable, provenance-carrying MEMORY record (bible sec 21-22). The
    /// record is minted `Active` at the neutral outcome score from the `draft`
    /// (scope + claim + source + author + citations + privacy + optional expiry)
    /// and written to the KV `memory` namespace keyed by its minted id, so it
    /// survives a workspace reopen. Returns the stored record (with its id).
    pub(crate) fn spawn_create_side_chat(
        &self,
        parent: SessionId,
        at_event: Option<EventId>,
        inherit: bool,
    ) {
        let replay = self.replay.clone();
        let sessions = self.services.sessions.clone();
        let kv = self.services.key_value_store.clone();
        let bus = Arc::clone(&self.ui_bus);
        tokio::spawn(async move {
            match branch_and_record(
                &replay,
                &sessions,
                &kv,
                parent.clone(),
                at_event,
                crate::services::SessionRelationship::SideChat,
                true,
                inherit,
            )
            .await
            {
                Ok((new_session, record, _)) => {
                    bus.publish(UiEvent {
                        seq: record.forked_at.unwrap_or(0),
                        session_id: Some(new_session),
                        kind: UiEventKind::Custom(json!({
                            "kind": "side_chat_created",
                            "record": serde_json::to_value(&record).unwrap_or_else(|_| json!({})),
                        })),
                    });
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(parent),
                        kind: UiEventKind::Error {
                            code: "create_side_chat".to_string(),
                            message: err.to_string(),
                        },
                    });
                }
            }
        });
    }

    /// Perform a `merge_side_chat` custom intent: append the merge summary onto
    /// the parent (spawned; the ack returns immediately). Surfacing is done by
    /// [`Self::merge_side_chat_summary`]; a failure surfaces as an Error UiEvent.
    pub(crate) fn spawn_merge_side_chat(
        &self,
        side_chat: SessionId,
        parent: SessionId,
        summary: String,
    ) {
        let event_log = self.services.event_log.clone();
        let bus = Arc::clone(&self.ui_bus);
        let result = SideChatResult::summary_only(summary);
        tokio::spawn(async move {
            let appended = event_log
                .append(NewEvent::system(
                    parent.clone(),
                    "session.merge_summary",
                    result.merge_event_payload(&side_chat),
                ))
                .await;
            match appended {
                Ok(event) => {
                    bus.publish(UiEvent {
                        seq: event.seq,
                        session_id: Some(parent.clone()),
                        kind: UiEventKind::Custom(result.merged_ui_payload(&parent, &side_chat)),
                    });
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(parent),
                        kind: UiEventKind::Error {
                            code: "merge_side_chat".to_string(),
                            message: err.to_string(),
                        },
                    });
                }
            }
        });
    }

    /// Publish a `session_forked` UiEvent carrying the new thread's record, under
    /// the new session id (so the FE, which adopts a session off any event's
    /// `session_id`, switches to the fork).
    pub(crate) fn spawn_fork_session(&self, from: SessionId, at_event: hide_core::ids::EventId) {
        let replay = self.replay.clone();
        let sessions = self.services.sessions.clone();
        let kv = self.services.key_value_store.clone();
        let bus = Arc::clone(&self.ui_bus);
        tokio::spawn(async move {
            match fork_and_record(&replay, &sessions, &kv, from.clone(), Some(at_event)).await {
                Ok((new_session, record, _)) => {
                    bus.publish(UiEvent {
                        seq: record.forked_at.unwrap_or(0),
                        session_id: Some(new_session),
                        kind: UiEventKind::Custom(json!({
                            "kind": "session_forked",
                            "record": serde_json::to_value(&record).unwrap_or_else(|_| json!({})),
                        })),
                    });
                }
                Err(err) => {
                    bus.publish(UiEvent {
                        seq: 0,
                        session_id: Some(from),
                        kind: UiEventKind::Error {
                            code: "fork_session".to_string(),
                            message: err.to_string(),
                        },
                    });
                }
            }
        });
    }
}
