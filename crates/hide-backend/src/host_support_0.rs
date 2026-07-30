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


/// Shared BRANCH-by-event core: resolve a boundary, mint a fresh INDEPENDENT
/// lineage, and durably record ANCESTRY with an explicit relationship + read-only
/// flag -- WITHOUT publishing (the caller owns surfacing). Used by the fork path
/// ([`fork_and_record`]) and the side-chat path
/// ([`BackendHost::create_side_chat`]).
///
/// An explicit `at_event` resolves strictly (`NotFound` if it is absent from the
/// source); `None` branches the whole session (its current tail). When `inherit`
/// is true the source prefix up to the boundary is copied forward into the new
/// lineage (the fork/side-chat sees the pre-boundary history); when false the new
/// session starts empty, with only its ANCESTRY (parent + boundary) recorded.
///
/// Ancestry lives in the KV `session_records` namespace, NOT the new session's
/// own event log, so it never pollutes the transcript and survives a reopen.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn branch_and_record(
    replay: &BackendReplayService,
    sessions: &Arc<crate::services::SessionRegistry>,
    kv: &hide_core::persistence::DynKeyValueStore,
    from: SessionId,
    at_event: Option<EventId>,
    relationship: crate::services::SessionRelationship,
    read_only: bool,
    inherit: bool,
) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
    let at_seq = match &at_event {
        Some(id) => replay.seq_of_event(from.clone(), id).await?,
        None => replay.latest_seq(from.clone()).await?,
    };
    let (new_session, projection) = if inherit {
        // Copy the source prefix forward under a fresh session id (independent).
        replay.fork_session(from.clone(), at_seq).await?
    } else {
        // A fresh, empty branch: mint a new id and build its (empty) projection
        // without carrying any prefix; nothing is appended to any log.
        let new_session = SessionId::new();
        let projection = replay.rebuild_session(new_session.clone()).await?;
        (new_session, projection)
    };
    let record = crate::services::SessionRecord::branch(
        new_session.clone(),
        from,
        at_seq,
        at_event,
        relationship,
        read_only,
    );
    sessions.record_session(kv, &record);
    Ok((new_session, record, projection))
}

/// Shared fork-by-event core (used by both the direct
/// [`BackendHost::fork_session_from_event`] method and the spawned `ForkSession`
/// intent path): a read/write [`SessionRelationship::Fork`] that inherits the
/// source prefix. Delegates to [`branch_and_record`].
pub(crate) async fn fork_and_record(
    replay: &BackendReplayService,
    sessions: &Arc<crate::services::SessionRegistry>,
    kv: &hide_core::persistence::DynKeyValueStore,
    from: SessionId,
    at_event: Option<EventId>,
) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
    branch_and_record(
        replay,
        sessions,
        kv,
        from,
        at_event,
        crate::services::SessionRelationship::Fork,
        false,
        true,
    )
    .await
}

/// DETERMINISTIC, model-free evaluation of a [`GoalRecord`] against the durable
/// `verify.result` evidence in a session's event log (bible sec 14). The evidence
/// read is EXACTLY the session's verification receipts: each `verify.result` event
/// carries a [`Verdict`](hide_kernel::verify::oracle::Verdict); we take the LATEST
/// verdict per oracle (log order == seq order) and the overall latest.
///
/// * STRUCTURED acceptance (oracle names present): every named oracle must have a
///   latest verdict of `Pass` -> `Met`; a missing or non-`Pass` oracle -> `NotMet`
///   with the reason. The consulted verdict event ids are the returned evidence.
/// * No acceptance, but a recognized verification `condition` ([`is_verification_condition`]):
///   the session's latest verification verdict must be `Pass`.
/// * Otherwise (a natural-language condition, no structured acceptance): the
///   outcome is `DeferredModelRequired` -- a model would be needed to judge it, and
///   NO model is called here.
/// Parse a [`MemoryDraft`] out of a custom-intent Value (bible sec 21-22). Used by
/// the `memory_add` / `memory_supersede` dispatch arms. `MemoryDraft` intentionally
/// does not derive `Deserialize` (its id/score/status are derived, not supplied),
/// so the required provenance (scope + claim + source + author) is read explicitly
/// and the optional refinements are layered via the builder setters.
pub(crate) fn parse_memory_draft(payload: &Value) -> Result<MemoryDraft> {
    let field = |name: &str| {
        hide_core::error::HideError::Message(format!("memory draft: missing '{name}'"))
    };
    let scope: MemoryScope = serde_json::from_value(
        payload.get("scope").cloned().ok_or_else(|| field("scope"))?,
    )
    .map_err(|e| hide_core::error::HideError::Message(format!("memory draft: bad scope: {e}")))?;
    let claim = payload
        .get("claim")
        .and_then(|v| v.as_str())
        .ok_or_else(|| field("claim"))?;
    let source = payload
        .get("source")
        .and_then(|v| v.as_str())
        .ok_or_else(|| field("source"))?;
    let author = payload
        .get("author")
        .and_then(|v| v.as_str())
        .ok_or_else(|| field("author"))?;
    let mut draft = MemoryDraft::new(scope, claim, source, author);
    if let Some(confidence) = payload.get("confidence").and_then(|v| v.as_f64()) {
        draft = draft.with_confidence(confidence as f32);
    }
    if let Some(citations) = payload.get("citations").and_then(|v| v.as_array()) {
        draft = draft.with_citations(
            citations
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
        );
    }
    if let Some(invalidation) = payload.get("invalidation").and_then(|v| v.as_array()) {
        draft = draft.with_invalidation(
            invalidation
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
        );
    }
    if let Some(privacy) = payload.get("privacy") {
        let privacy: PrivacyClass = serde_json::from_value(privacy.clone()).map_err(|e| {
            hide_core::error::HideError::Message(format!("memory draft: bad privacy: {e}"))
        })?;
        draft = draft.with_privacy(privacy);
    }
    if let Some(expiry) = payload.get("expiry_ms").and_then(|v| v.as_u64()) {
        draft = draft.with_expiry_ms(Some(expiry));
    }
    Ok(draft)
}

pub(crate) fn evaluate_goal(goal: &GoalRecord, events: &[Event]) -> GoalVerdict {
    use hide_kernel::verify::oracle::{Verdict, VerdictStatus};

    // Latest verdict (+ its event id) per oracle, and the overall latest, walking
    // the log in seq order (`scan` already returns events seq-ordered).
    let mut latest_by_oracle: std::collections::HashMap<String, (Verdict, EventId)> =
        std::collections::HashMap::new();
    let mut overall_latest: Option<(Verdict, EventId)> = None;
    for event in events {
        if event.kind != "verify.result" {
            continue;
        }
        if let Some(verdict) = event.payload_as::<Verdict>() {
            overall_latest = Some((verdict.clone(), event.id.clone()));
            latest_by_oracle.insert(verdict.oracle.clone(), (verdict, event.id.clone()));
        }
    }

    let mk = |outcome, reason, evidence| GoalVerdict {
        goal_id: goal.goal_id.clone(),
        session_id: goal.session_id.clone(),
        outcome,
        reason,
        evidence,
    };

    if !goal.acceptance.is_empty() {
        // STRUCTURED path: every named oracle must have a latest verdict == Pass.
        let mut evidence = Vec::new();
        for oracle in &goal.acceptance {
            match latest_by_oracle.get(oracle) {
                None => {
                    return mk(
                        GoalOutcome::NotMet,
                        format!("no verification evidence yet for oracle '{oracle}'"),
                        evidence,
                    );
                }
                Some((verdict, id)) => {
                    evidence.push(id.clone());
                    if verdict.status != VerdictStatus::Pass {
                        return mk(
                            GoalOutcome::NotMet,
                            format!(
                                "oracle '{oracle}' did not pass (latest status: {:?}): {}",
                                verdict.status, verdict.detail
                            ),
                            evidence,
                        );
                    }
                }
            }
        }
        return mk(
            GoalOutcome::Met,
            format!("all {} acceptance oracle(s) passed", goal.acceptance.len()),
            evidence,
        );
    }

    // No structured acceptance: fall back to the session's latest verification
    // verdict, but only when the condition reads as a verification condition.
    if is_verification_condition(&goal.condition) {
        match overall_latest {
            None => mk(
                GoalOutcome::NotMet,
                "no verification evidence yet for this session".to_string(),
                Vec::new(),
            ),
            Some((verdict, id)) => {
                let evidence = vec![id];
                if verdict.status == VerdictStatus::Pass {
                    mk(
                        GoalOutcome::Met,
                        format!("latest verification verdict passed (oracle '{}')", verdict.oracle),
                        evidence,
                    )
                } else {
                    mk(
                        GoalOutcome::NotMet,
                        format!(
                            "latest verification verdict did not pass (oracle '{}', status {:?})",
                            verdict.oracle, verdict.status
                        ),
                        evidence,
                    )
                }
            }
        }
    } else {
        mk(
            GoalOutcome::DeferredModelRequired,
            "natural-language condition requires a model to judge \
             (deferred_model_required); no model was called"
                .to_string(),
            Vec::new(),
        )
    }
}

/// Whether a goal `condition` reads as a recognized deterministic VERIFICATION
/// condition -- one evaluable model-free against `verify.result` evidence (e.g.
/// `"tests_pass"`, `"verify green"`). Everything else is a natural-language
/// condition (`DEFERRED_MODEL_REQUIRED`). Case/separator-insensitive.
pub(crate) fn is_verification_condition(condition: &str) -> bool {
    let norm = condition.trim().to_lowercase().replace([' ', '-'], "_");
    norm.contains("pass")
        || norm.contains("verify")
        || norm.contains("verification")
        || norm.contains("green")
        || norm.contains("test")
}

/// The recorder every dispatch reports to.
///
/// It hangs off the [`ToolDispatcher`] itself, so the kernel agent (which holds the dispatcher
/// directly), the editor save and anything added later all produce the SAME record: the durable
/// `tool.call`/`tool.result` pair the timeline and transcript search read, the live `ToolProgress`,
/// and - for a write - the addressable [`DiffProposal`] the hunk review surface, the checkpoint's
/// `repo_state` coverage and the code rewind all read. This used to live in a host wrapper that
/// exactly one production caller passed a run to, so an agent edit produced none of it.
pub(crate) struct DispatchRecorder {
    services: SharedBackend,
    ui_bus: Arc<UiEventBus>,
    /// Attribution fixed at construction, for a dispatcher built for ONE task (the turn kernel's).
    /// Unset on the host's shared dispatcher, which serves every session and reads the ambient
    /// [`crate::tools::dispatch_context`] instead.
    bound: Option<crate::tools::DispatchContext>,
}

/// The tools that WRITE the workspace, whose pre/post image is captured as a reviewable hunk.
pub(crate) fn writes_workspace(tool: &str) -> bool {
    tool.starts_with("edit.") || tool == "fs.write"
}

impl DispatchRecorder {
    pub(crate) fn new(services: SharedBackend, ui_bus: Arc<UiEventBus>) -> Self {
        Self {
            services,
            ui_bus,
            bound: None,
        }
    }

    /// A recorder for one task's dispatcher: every call through it is that session's and that
    /// run's, whatever task polls it (a task-local would not survive the kernel spawning one).
    pub(crate) fn bound_to(
        services: SharedBackend,
        ui_bus: Arc<UiEventBus>,
        bound: crate::tools::DispatchContext,
    ) -> Self {
        Self {
            services,
            ui_bus,
            bound: Some(bound),
        }
    }

    /// Who this call is for. An unattributed dispatch is still RECORDED (against the default
    /// session, ungrouped) rather than silently vanishing.
    pub(crate) fn context(&self) -> crate::tools::DispatchContext {
        self.bound
            .clone()
            .or_else(crate::tools::dispatch_context)
            .unwrap_or_else(|| crate::tools::DispatchContext {
                session_id: self.services.session(),
                run_id: None,
            })
    }

    /// The ONE spelling rule for a written path: absolute to touch the file, workspace-relative to
    /// RECORD it. Every downstream consumer (the diff store, `rewind::code_state`, the checkpoint
    /// coverage digest, and the verification receipts, whose scope is workspace-relative) then
    /// compares one spelling instead of two that can never match.
    pub(crate) fn locate(&self, path: &str) -> (PathBuf, String) {
        let root = &self.services.config.workspace_root;
        let raw = Path::new(path);
        let abs = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            root.join(raw)
        };
        let rel = workspace_relative(root, &abs);
        (abs, rel)
    }

    /// Record the applied call. Returns an error only for a storage failure; the caller surfaces it
    /// (the dispatch itself already happened, so it is never rolled back).
    pub(crate) async fn record(
        &self,
        call: &ToolCall,
        before: Option<Value>,
        result: &ToolResult,
    ) -> Result<()> {
        let ctx = self.context();
        let mut call_new = NewEvent::tool_call(
            ctx.session_id.clone(),
            ToolCallEvent {
                call_id: call.call_id.clone(),
                tool_name: call.tool.clone(),
                capability_grant_id: call.capability_grant_id.clone(),
                args: call.args.clone(),
                predicted_effects: result.effects.clone(),
            },
        );
        call_new.run_id = ctx.run_id.clone();
        let call_event_record = self.services.event_log.append(call_new).await?;
        // The tool.result Observation pairs back to the tool.call Action via `cause`
        // (T3 Action/Observation replay pairing).
        let mut result_new = NewEvent::tool_result(
            ctx.session_id.clone(),
            ToolResultEvent {
                call_id: result.call_id.clone(),
                ok: result.status == ToolStatus::Ok,
                summary: tool_result_summary(result),
                output: result.structured_content.clone(),
                bytes_ref: result.bytes_ref.clone(),
            },
        );
        result_new.run_id = ctx.run_id.clone();
        result_new.cause = Some(call_event_record.id);
        let result_event = self.services.event_log.append(result_new).await?;
        self.services.projection_store.put_projection(
            &result_event.session_id,
            result_event.seq,
            json!({
                "projection": "last_tool_result",
                "tool_status": result.status,
                "tool_output": result.structured_content.clone(),
            }),
        )?;
        // Push the tool progress onto the live Wire-B bus (in addition to the durable log the pull
        // API replays from).
        self.ui_bus.publish(UiEvent {
            seq: result_event.seq,
            session_id: Some(result_event.session_id.clone()),
            kind: UiEventKind::ToolProgress {
                call_id: result.call_id.as_str().to_string(),
                message: if result.status == ToolStatus::Ok {
                    tool_result_summary(result)
                } else {
                    format!("failed: {}", tool_result_summary(result))
                },
                // The RECORDED event this step is, so a timeline can address it as a boundary.
                // `seq_of_event` resolves exactly this id.
                event_id: Some(result_event.id.as_str().to_string()),
            },
        });
        // Procedural memory: only a *successful* command/build/test receipt becomes
        // a recipe. Mint site is classed_writers::write_procedural_from_receipt.
        let _ = crate::classed_writers::write_procedural_from_receipt(
            &self.services.classed_memory,
            call,
            result,
            ctx.session_id.as_str(),
            ctx.run_id.as_ref().map(|r| r.as_str()),
        );
        // Register the applied write as an addressable diff hunk (census sec 23): the
        // immediate-apply flow already wrote to disk, so we read the post-image and record
        // before/after for later per-hunk keep or revert. Grouped by the run, so an unattributed
        // dispatch records its events and no hunk (there is nothing to group it under).
        if let (Some(pre), Some(run)) = (before, ctx.run_id.as_ref()) {
            if result.status == ToolStatus::Ok {
                let abs = pre.get("abs").and_then(|v| v.as_str()).unwrap_or_default();
                let file = pre.get("file").and_then(|v| v.as_str()).unwrap_or_default();
                let text = pre.get("before").and_then(|v| v.as_str()).unwrap_or_default();
                let after = std::fs::read_to_string(abs).unwrap_or_default();
                if after != text {
                    self.record_edit_diff(
                        &ctx.session_id,
                        run,
                        &call.tool,
                        file.to_string(),
                        text.to_string(),
                        after,
                    )
                    .await?;
                }
            }
        }
        Ok(())
    }

    pub(crate) async fn record_edit_diff(
        &self,
        session_id: &SessionId,
        run_id: &RunId,
        tool_name: &str,
        file: String,
        before: String,
        after: String,
    ) -> Result<()> {
        let kv = &self.services.key_value_store;
        let diff_id = format!("diff-{}", run_id.as_str());
        let mut proposal = DiffStore::get(kv, &diff_id).unwrap_or_else(|| DiffProposal {
            diff_id: diff_id.clone(),
            run_id: run_id.as_str().to_string(),
            session_id: session_id.clone(),
            created_ms: hide_core::ids::now_ms(),
            created_from: DiffProvenance {
                plan_step: None,
                agent: tool_name.to_string(),
                turn: 0,
            },
            hunks: Vec::new(),
        });
        let turn = proposal.hunks.len() as u64;
        let base_hash = blake3::hash(before.as_bytes()).to_hex().to_string();
        proposal.hunks.push(DiffHunk {
            hunk_id: format!("{diff_id}-h{turn}"),
            file,
            base_hash,
            before,
            after,
            status: HunkStatus::Pending,
            provenance: DiffProvenance {
                plan_step: None,
                agent: tool_name.to_string(),
                turn,
            },
        });
        DiffStore::put(kv, &proposal)?;
        self.services
            .event_log
            .append(NewEvent::system(
                session_id.clone(),
                "diff.proposed",
                serde_json::to_value(&proposal).unwrap_or(Value::Null),
            ))
            .await?;
        publish_diff_to(&self.ui_bus, &proposal);
        Ok(())
    }
}

impl hide_core::tool::DispatchObserver for DispatchRecorder {
    fn before(&self, call: &ToolCall) -> Option<Value> {
        if !writes_workspace(&call.tool) {
            return None;
        }
        let path = call.args.get("path").and_then(|v| v.as_str())?;
        let (abs, rel) = self.locate(path);
        Some(json!({
            "abs": abs.to_string_lossy(),
            "file": rel,
            "before": std::fs::read_to_string(&abs).unwrap_or_default(),
        }))
    }

    fn after<'a>(
        &'a self,
        call: &'a ToolCall,
        before: Option<Value>,
        result: &'a ToolResult,
    ) -> futures::future::BoxFuture<'a, ()> {
        Box::pin(async move {
            if let Err(err) = self.record(call, before, result).await {
                // The tool already ran; a recording failure must be visible, not swallowed.
                self.ui_bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::Error {
                        code: "dispatch_record".to_string(),
                        message: err.to_string(),
                    },
                });
            }
        })
    }
}

/// A path spelled relative to the workspace root when it is inside it, unchanged otherwise.
pub(crate) fn workspace_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Publish the two diff projections the review surface reads. Free of the host so the recorder
/// hanging off the dispatcher publishes through exactly the same producer.
pub(crate) fn publish_diff_to(ui_bus: &UiEventBus, proposal: &DiffProposal) {
    let (diff, chips) = diff_projections(proposal);
    for (projection, patch) in [("diff", diff), ("diff_chip", chips)] {
        ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(proposal.session_id.clone()),
            kind: UiEventKind::ProjectionPatch {
                projection: projection.to_string(),
                patch,
            },
        });
    }
}

/// True if two path scopes share any file/directory, using hide-verify's
/// containment-aware [`paths_intersect`](hide_kernel::verify_plane::paths_intersect) semantics
/// (a directory scope intersects a file it contains). Drives the authority
/// reconciliation in [`BackendHost::reconcile_review_for_scope`] so a review is
/// only weighed against deterministic receipts for the SAME scope.
pub(crate) fn scopes_intersect(a: &[String], b: &[String]) -> bool {
    a.iter()
        .any(|x| b.iter().any(|y| hide_kernel::verify_plane::paths_intersect(x, y)))
}

pub(crate) fn unknown_diff(diff_id: &str) -> hide_core::error::HideError {
    hide_core::error::HideError::Message(format!("unknown diff {diff_id}"))
}

pub(crate) fn unknown_hunk(hunk_id: &str) -> hide_core::error::HideError {
    hide_core::error::HideError::Message(format!("unknown hunk {hunk_id}"))
}

pub(crate) fn unknown_repo(repo_id: &str) -> hide_core::error::HideError {
    hide_core::error::HideError::NotFound(format!("unknown repo {repo_id}"))
}

/// Which lease a revocation trigger applies to. A trigger that names a run or a repo revokes only
/// a lease that belongs to it, so another task's cancellation cannot take this task's lease away.
pub(crate) enum LeaseRevokeScope {
    Any,
    Run(String),
    Repo(String),
}

impl LeaseRevokeScope {
    pub(crate) fn revoke(&self) -> Option<crate::tools::WriteLease> {
        match self {
            Self::Any => crate::tools::revoke_write_lease("revoked"),
            Self::Run(run) => crate::tools::revoke_write_lease_for_run(run, None),
            Self::Repo(repo) => crate::tools::revoke_write_lease_for_repo(repo),
        }
    }
}

/// Publish the write lease onto the EXISTING `status` projection the status bar already routes.
///
/// A free function because the run-completion revoke fires from the kernel turn driver, which holds
/// the bus but not the host. `active: false` is published on revoke so the bar clears rather than
/// keeping a lease on screen that no longer exists.
pub(crate) fn publish_write_lease(
    ui_bus: &UiEventBus,
    lease: Option<&crate::tools::WriteLease>,
    note: &str,
) {
    ui_bus.publish(UiEvent {
        seq: 0,
        session_id: None,
        kind: UiEventKind::ProjectionPatch {
            projection: "status".to_string(),
            patch: write_lease_patch(lease, note),
        },
    });
}

/// The `status` projection patch a write lease renders as. ONE shape, because the lease reaches a
/// client two ways: the live publish above, and the fresh-client read the home connector serves
/// (`connectors::HomeConnector`). It is a process-global static held in memory only, never a durable
/// event, so a reloaded tab could not learn about a lease that was still being honoured; a replayed
/// grant/revoke pair would have been worse, since the static does not survive a host restart and the
/// log would claim a lease nothing holds.
pub(crate) fn write_lease_patch(
    lease: Option<&crate::tools::WriteLease>,
    note: &str,
) -> serde_json::Value {
    json!({
        "write_lease": {
            "active": lease.is_some(),
            "note": note,
            "lease_id": lease.map(|l| l.lease_id.clone()),
            "repo_id": lease.map(|l| l.repo_id.clone()),
            "scopes": lease.map(|l| l
                .scopes
                .iter()
                .map(|s| s.display().to_string())
                .collect::<Vec<_>>())
                .unwrap_or_default(),
            "granted_ms": lease.map(|l| l.granted_ms),
        }
    })
}

// --- The diff projection the FE actually reads (census sec 23) ---
//
// app/src/surfaces/ide/types.ts folds `projection_patch{projection:"diff"}` into a
// DiffDoc {diff_id, run_id, path, lang, before, after, hunks[{id, header, lines, status}]}
// and app/src/surfaces/chat/parts.ts folds `{projection:"diff_chip"}` into
// {chips:[{diff_id, path, added, removed, status}]}. The host record is per RUN and
// spans files, the view model names ONE file, so the host adapts:
//   * `hunks` carries EVERY hunk of the run (each also keeping the wire fields the
//     review reads back: hunk_id, file, base_hash, provenance), so review is never
//     silently narrowed to one file;
//   * `path`/`before`/`after` name the most recently edited file, which is what the
//     side-by-side Monaco model is built from.
// `stale` is not published: the record has no disk read, so it would be a guess.
// ponytail: no on-disk drift check. Publish `stale` when the host reads the file's
// current hash back at publish time.

/// How many unchanged lines are kept either side of the changed block.
pub(crate) const DIFF_CONTEXT_LINES: usize = 3;

/// Monaco language id from a file extension. Unknown extensions read as plaintext
/// rather than as a guess that would syntax-colour the wrong grammar.
pub(crate) fn monaco_language(file: &str) -> &'static str {
    match file.rsplit('.').next().unwrap_or("") {
        "rs" => "rust",
        "ts" | "tsx" => "typescript",
        "js" | "jsx" => "javascript",
        "json" => "json",
        "md" => "markdown",
        "py" => "python",
        "toml" => "toml",
        "yaml" | "yml" => "yaml",
        "html" => "html",
        "css" => "css",
        "sh" => "shell",
        _ => "plaintext",
    }
}

/// Fold one hunk's whole-file pre/post images into the FE's line view: common
/// leading and trailing lines are context, the middle is the removed block then the
/// added block. Returns the lines, the `@@` header and the (added, removed) counts.
///
/// ponytail: prefix/suffix trim, not an LCS diff, so an edit that touches two far
/// apart regions of one file reads as a single wide block. hide-backend does not
/// depend on `similar`; wire it in if per-region hunks are wanted.
pub(crate) fn hunk_line_view(file: &str, before: &str, after: &str) -> (Vec<Value>, String, usize, usize) {
    let old: Vec<&str> = before.lines().collect();
    let new: Vec<&str> = after.lines().collect();
    let mut pre = 0;
    while pre < old.len() && pre < new.len() && old[pre] == new[pre] {
        pre += 1;
    }
    let mut suf = 0;
    while suf < old.len() - pre
        && suf < new.len() - pre
        && old[old.len() - 1 - suf] == new[new.len() - 1 - suf]
    {
        suf += 1;
    }
    let old_mid = &old[pre..old.len() - suf];
    let new_mid = &new[pre..new.len() - suf];
    let ctx = |text: &str, o: usize, n: usize| {
        json!({ "kind": "ctx", "text": text, "oldNo": o, "newNo": n })
    };
    let mut lines: Vec<Value> = Vec::new();
    for i in pre.saturating_sub(DIFF_CONTEXT_LINES)..pre {
        lines.push(ctx(old[i], i + 1, i + 1));
    }
    for (i, text) in old_mid.iter().enumerate() {
        lines.push(json!({ "kind": "del", "text": text, "oldNo": pre + i + 1, "newNo": null }));
    }
    for (i, text) in new_mid.iter().enumerate() {
        lines.push(json!({ "kind": "add", "text": text, "oldNo": null, "newNo": pre + i + 1 }));
    }
    for k in 0..suf.min(DIFF_CONTEXT_LINES) {
        lines.push(ctx(
            old[old.len() - suf + k],
            old.len() - suf + k + 1,
            new.len() - suf + k + 1,
        ));
    }
    let header = format!(
        "@@ -{},{} +{},{} @@ {file}",
        pre + 1,
        old_mid.len(),
        pre + 1,
        new_mid.len()
    );
    (lines, header, new_mid.len(), old_mid.len())
}

/// The `(diff, diff_chip)` projection patches for a proposal. Shared by the live
/// publish and by the reconnect replay so both surfaces see the SAME shape.
pub(crate) fn diff_projections(proposal: &DiffProposal) -> (Value, Value) {
    let mut hunks: Vec<Value> = Vec::new();
    // file -> (added, removed, any_pending, any_kept)
    let mut per_file: Vec<(String, usize, usize, bool, bool)> = Vec::new();
    for h in &proposal.hunks {
        let (lines, header, added, removed) = hunk_line_view(&h.file, &h.before, &h.after);
        let status = match h.status {
            HunkStatus::Pending => "pending",
            HunkStatus::Accepted => "accepted",
            HunkStatus::Rejected => "rejected",
        };
        hunks.push(json!({
            "id": h.hunk_id,
            "hunk_id": h.hunk_id,
            "file": h.file,
            "base_hash": h.base_hash,
            "header": header,
            "status": status,
            "lines": lines,
            "provenance": h.provenance,
        }));
        match per_file.iter_mut().find(|(f, ..)| *f == h.file) {
            Some(row) => {
                row.1 += added;
                row.2 += removed;
                row.3 |= h.status == HunkStatus::Pending;
                row.4 |= h.status != HunkStatus::Rejected;
            }
            None => per_file.push((
                h.file.clone(),
                added,
                removed,
                h.status == HunkStatus::Pending,
                h.status != HunkStatus::Rejected,
            )),
        }
    }
    let latest = proposal.hunks.last();
    let file = latest.map(|h| h.file.as_str()).unwrap_or("");
    // The Monaco model for that file: its FIRST pre-image and its LAST post-image, so
    // several edits to one file read as one before/after rather than as the last one alone.
    let before = proposal
        .hunks
        .iter()
        .find(|h| h.file == file)
        .map(|h| h.before.as_str())
        .unwrap_or("");
    let after = latest.map(|h| h.after.as_str()).unwrap_or("");
    let diff = json!({
        "diff_id": proposal.diff_id,
        "run_id": proposal.run_id,
        "path": file,
        "lang": monaco_language(file),
        "before": before,
        "after": after,
        "hunks": hunks,
    });
    let chips: Vec<Value> = per_file
        .iter()
        .map(|(f, added, removed, pending, kept)| {
            json!({
                "diff_id": proposal.diff_id,
                "run_id": proposal.run_id,
                "path": f,
                "added": added,
                "removed": removed,
                "status": if *pending { "proposed" } else if *kept { "applied" } else { "rejected" },
            })
        })
        .collect();
    (diff, json!({ "chips": chips }))
}

/// The spawnable twin of [`BackendHost::generate_and_publish`]: it takes owned
/// clones (so it is `'static` for `tokio::spawn`) and wires the run's `run_id`
/// into the [`InterruptHub`] so `CancelRun`/`PauseRun` reach it. A `CancelRun`
/// that lands before the (single-shot) HTTP generate fires aborts the run with
/// a `RuntimeStatus` notice rather than a fake completion.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn generate_submit_turn(
    event_log: hide_core::persistence::DynEventLog,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
    classed_memory: hawking_context::DynClassedMemory,
    ui_bus: Arc<UiEventBus>,
    interrupts: Arc<InterruptHub>,
    run_id: RunId,
    session_id: SessionId,
    base_url: String,
    prompt: String,
    repo_instructions: Arc<crate::compat_instructions::ResolvedInstructions>,
) -> Result<String> {
    use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
    use hide_kernel::govern::Interrupt;

    // Cooperative cancel: a CancelRun/PauseRun buffered for this run before we
    // start aborts cleanly (surfaced as a RuntimeStatus, not a fake token).
    if matches!(interrupts.take(&run_id), Some(Interrupt::Abort)) {
        ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(session_id),
            kind: UiEventKind::RuntimeStatus {
                status: "cancelled".to_string(),
                detail: Some(format!(
                    "run {} cancelled before generation",
                    run_id.as_str()
                )),
            },
        });
        return Ok(String::new());
    }

    // W-F6-1: snapshot the live ceiling ONCE (before streaming) so the shared
    // core's sync token sink can emit a throttled per-step occupancy patch with
    // no per-token HTTP round-trip. The authoritative full `ManifestLive` patch
    // still fires post-turn (below).
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

    // The live model behind the uniform inference seam. Generation runs through
    // the SHARED `run_turn_core` so this path and `generate_and_publish` build the
    // IDENTICAL real request (compiled context + real history + a derived budget)
    // and can never drift.
    let inference: Arc<dyn hawking_orch::inference::InferenceClient> = Arc::new(
        ModelProviderInferenceClient::new(HttpModelProvider::new(base_url.clone())),
    );
    let outcome = run_turn_core(
        inference,
        event_log,
        role_registry,
        code_index,
        memory,
        classed_memory,
        ui_bus.clone(),
        session_id.clone(),
        prompt,
        live_snap,
        Some(run_id.as_str().to_string()),
        repo_instructions,
    )
    .await?;
    let buf = outcome.completion;
    let prompt_chars = outcome.prompt_chars;

    // Spine A: publish the live context_manifest the Context Stack reads. The
    // effective ceiling is the engine's measured `.tq` multiplier x native (read
    // live, never a constant). `used_tokens` here is a labeled per-turn estimate;
    // precise per-token occupancy arrives once the engine reports sequence position.
    // Native and effective stay distinct; retrieval/usable is never reported as native.
    {
        let ctx_provider = HttpModelProvider::new(base_url);
        if let Some(info) = ctx_provider.get_context_info().await {
            let ceiling = info.ctx_len_effective.or(info.ctx_len_native).unwrap_or(0);
            let used_est = (prompt_chars + buf.len()) / 4;
            // Spine A (W-F2-1): build a real `ManifestLive`. For an SSM (RWKV-7,
            // which reports a constant recurrent state) the regime is recall
            // FIDELITY -- "how sharp", via the calibratable probe -- not KV
            // saturation; the watermark bands then key off `1 - fidelity`.
            let live = build_live_manifest(
                info.recurrent_state_bytes,
                info.ctx_len_native,
                ceiling,
                used_est,
            );
            let capability = declare_turn_capability(
                info.ctx_len_native.unwrap_or(ceiling).max(1),
                info.ctx_len_native,
                info.ctx_len_effective.or(Some(ceiling)),
                Some(info.tq_multiplier),
                info.tq_estimated,
            );
            // Post-turn rot from the live reading so the loop can notice degradation.
            let empty = hawking_context::ContextManifest::new(
                info.ctx_len_native.unwrap_or(ceiling),
            );
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
                seq: outcome.stream_seq,
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
    Ok(buf)
}

/// Bounded autonomy for fleet/agent kernels built by the host (`build_turn_kernel`,
/// fleet launchers). Defaults to [`Autonomy::SuggestOnly`] so effectful steps
/// pause for approval rather than running unsandboxed. `HIDE_KERNEL_AUTONOMY=
/// full_auto` (or `read_only`) overrides it. Product `SubmitTurn` uses
/// [`run_turn_core`] only and does not consult this.
pub(crate) fn turn_kernel_autonomy() -> Autonomy {
    match std::env::var("HIDE_KERNEL_AUTONOMY").ok().as_deref() {
        Some("full_auto") | Some("full") => Autonomy::FullAuto,
        Some("read_only") | Some("readonly") => Autonomy::ReadOnly,
        _ => Autonomy::SuggestOnly,
    }
}
