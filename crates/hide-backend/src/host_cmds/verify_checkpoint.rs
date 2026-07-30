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
    pub async fn run_static_analysis(
        &self,
        session: SessionId,
        sources: Vec<SourceFile>,
    ) -> Result<StaticAnalysisReceipt> {
        use hide_kernel::verify_plane::Oracle;

        let oracle = StaticAnalysisOracle::new();
        let started_ms = hide_core::ids::now_ms();
        let input = hide_kernel::verify_plane::VerificationInput::from_sources(sources.clone());
        let outcome = oracle.evaluate(&input);
        let duration_ms = hide_core::ids::now_ms().saturating_sub(started_ms);

        // Scope = the analyzed file paths (sorted + deduped): drives the
        // re-review dependency model and the authority reconciliation below.
        let mut scope: Vec<String> = sources.iter().map(|s| s.path.clone()).collect();
        scope.sort();
        scope.dedup();

        // Tie the verdict to an exact snapshot of the sources.
        let source_hash = hide_kernel::verify_plane::source_hash_of(
            sources.iter().map(|s| (s.path.as_str(), s.text.as_str())),
        );
        let verification_id = format!(
            "va-{}-{started_ms}",
            &source_hash[..source_hash.len().min(16)]
        );

        let receipt = VerificationReceipt::new(
            verification_id,
            VerificationTier::Tier1Deterministic,
            oracle.name(),
            None, // in-process oracle: no command was run
            scope,
            source_hash,
            outcome.verdict.clone(),
            started_ms,
            duration_ms,
        );
        let record = StaticAnalysisReceipt {
            receipt,
            findings: outcome.evidence.findings.clone(),
        };

        // Durable: append a `verify.result`-shaped event carrying the receipt +
        // findings-summary to the SAME session log (auditable + recoverable).
        self.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "verify.result",
                serde_json::to_value(&record).unwrap_or(Value::Null),
            ))
            .await?;
        // Verification class memory: sole VerifierWriteCap mint lives in
        // classed_writers::write_verification_from_receipt (never model turn).
        crate::classed_writers::write_verification_from_receipt(
            &self.services.classed_memory,
            &record.receipt,
            &record.findings_summary(),
            record.is_pass(),
            session.as_str(),
            None,
        );
        self.publish_verification(&record, &session);
        self.publish_diagnostics(&record, &session);
        Ok(record)
    }

    /// Every durable static-analysis receipt recorded for a session, in log order
    /// (bible sec 29 reader). Filters the session's `verify.result` events to the
    /// hide-verify receipts (a `hide_kernel` `Verdict` payload, which shares the
    /// event kind, is a disjoint shape and is skipped).
    pub async fn verification_receipts(
        &self,
        session: &SessionId,
    ) -> Result<Vec<StaticAnalysisReceipt>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(events
            .into_iter()
            .filter(|event| event.kind == "verify.result")
            .filter_map(|event| event.payload_as::<StaticAnalysisReceipt>())
            .collect())
    }

    /// The Tier4 review-role profiles as DATA (bible Book IX sec 28): correctness,
    /// security, performance, api-compatibility, tests, documentation, simplicity,
    /// scope. Each profile describes what a reviewer of that role focuses on, the
    /// context it needs, its output schema, and its acceptance condition.
    ///
    /// DEFERRED_MODEL_REQUIRED: EXECUTING a review role needs a model and is out
    /// of scope here. This returns profiles (data), NEVER a
    /// [`Verdict`](hide_kernel::verify_plane::Verdict), and performs NO model call.
    pub fn review_role_profiles(&self) -> Vec<ReviewRoleProfile> {
        hide_kernel::verify_plane::all_profiles()
    }

    /// The DATA profile for a single review role (bible Book IX sec 28). Like
    /// [`Self::review_role_profiles`], this is DEFERRED_MODEL_REQUIRED: it returns
    /// the profile, never a verdict, and calls no model.
    pub fn review_role_profile(&self, role: ReviewRole) -> ReviewRoleProfile {
        hide_kernel::verify_plane::profile_for(role)
    }

    /// Reconcile a set of probabilistic review verdicts against the deterministic
    /// static-analysis receipts covering `scope`, honoring THE AUTHORITY RULE
    /// (bible Book IX sec 28-29): a probabilistic review may NEVER override a
    /// failing deterministic (Tier0/Tier1) receipt for the same scope.
    ///
    /// The deterministic receipts whose scope intersects `scope` are folded into
    /// [`TieredVerdict`]s and reconciled with the `reviews` through
    /// [`hide_kernel::verify_plane::apply_gate`], which returns [`GateDecision::Reject`] on ANY
    /// deterministic failure regardless of what the review says. A review Pass can
    /// therefore never flip a Tier1 Fail. Model-free.
    pub fn reconcile_review_for_scope(
        &self,
        scope: &[String],
        deterministic: &[StaticAnalysisReceipt],
        reviews: &[TieredVerdict],
    ) -> GateDecision {
        let mut verdicts: Vec<TieredVerdict> = deterministic
            .iter()
            .filter(|r| scopes_intersect(&r.receipt.scope, scope))
            .map(|r| {
                TieredVerdict::new(
                    r.receipt.tier,
                    r.receipt.oracle.clone(),
                    r.receipt.verdict.clone(),
                )
            })
            .collect();
        verdicts.extend(reviews.iter().cloned());
        hide_kernel::verify_plane::apply_gate(&verdicts)
    }

    /// Publish the `diagnostics` PROJECTION patch on Wire-B (the surface the FE
    /// actually consumes, alongside `turn` / `context_manifest`) so the StatusBar
    /// Problems counter binds to the real error/warning counts from the sealed
    /// receipt instead of a hardcoded 0/0. Additive: a new projection NAME only,
    /// no new UiEventKind. The durable `verify.result` receipt stays untouched and
    /// readable via [`Self::verification_receipts`].
    pub fn diff_get(&self, diff_id: &str) -> Option<DiffProposal> {
        DiffStore::get(&self.services.key_value_store, diff_id)
    }

    /// Keep the whole diff: mark every still-pending hunk Accepted and record a
    /// `diff.hunk.accepted` event per hunk. Nothing is written (already on disk).
    pub async fn apply_diff(&self, diff_id: &str) -> Result<DiffProposal> {
        let kv = &self.services.key_value_store;
        let mut proposal = DiffStore::get(kv, diff_id).ok_or_else(|| unknown_diff(diff_id))?;
        let ids: Vec<String> = proposal
            .hunks
            .iter()
            .filter(|h| h.status == HunkStatus::Pending)
            .map(|h| h.hunk_id.clone())
            .collect();
        for h in proposal.hunks.iter_mut() {
            if h.status == HunkStatus::Pending {
                h.status = HunkStatus::Accepted;
            }
        }
        DiffStore::put(kv, &proposal)?;
        for id in &ids {
            self.record_diff_event(&proposal, "diff.hunk.accepted", Some(id))
                .await?;
        }
        self.publish_diff(&proposal);
        Ok(proposal)
    }

    /// Keep exactly one hunk (mark Accepted). The change is already on disk from
    /// the immediate-apply flow, so this records the decision + a durable
    /// `diff.hunk.accepted` event carrying provenance; nothing is written.
    pub async fn apply_hunk(&self, diff_id: &str, hunk_id: &str) -> Result<DiffProposal> {
        let kv = &self.services.key_value_store;
        let mut proposal = DiffStore::get(kv, diff_id).ok_or_else(|| unknown_diff(diff_id))?;
        {
            let h = proposal
                .hunk_mut(hunk_id)
                .ok_or_else(|| unknown_hunk(hunk_id))?;
            h.status = HunkStatus::Accepted;
        }
        DiffStore::put(kv, &proposal)?;
        self.record_diff_event(&proposal, "diff.hunk.accepted", Some(hunk_id))
            .await?;
        self.publish_diff(&proposal);
        Ok(proposal)
    }

    /// Revert exactly one hunk on disk via an inverse write through the SAME
    /// verifying applier the agent uses (`edit.write_file` guarded by the
    /// post-image hash as `base_hash`: a hunk superseded by a later edit conflicts
    /// instead of clobbering). Marks the hunk Rejected, invalidates the
    /// verification receipts whose scope intersects the file, and records a durable
    /// `diff.hunk.rejected` event carrying provenance.
    pub async fn reject_hunk(&self, diff_id: &str, hunk_id: &str) -> Result<DiffProposal> {
        let kv = &self.services.key_value_store;
        let mut proposal = DiffStore::get(kv, diff_id).ok_or_else(|| unknown_diff(diff_id))?;
        let (file, before, after) = {
            let h = proposal
                .hunk(hunk_id)
                .ok_or_else(|| unknown_hunk(hunk_id))?;
            (h.file.clone(), h.before.clone(), h.after.clone())
        };
        self.inverse_write(&proposal.session_id, &file, &before, &after)
            .await?;
        if let Some(h) = proposal.hunk_mut(hunk_id) {
            h.status = HunkStatus::Rejected;
        }
        DiffStore::put(kv, &proposal)?;
        self.record_diff_event(&proposal, "diff.hunk.rejected", Some(hunk_id))
            .await?;
        self.invalidate_verifications_for_files(&proposal.session_id, &[file])
            .await?;
        self.publish_diff(&proposal);
        Ok(proposal)
    }

    /// Undo the whole diff: revert every still-applied (Pending or Accepted) hunk
    /// on disk in reverse capture order (so later edits to the same file peel off
    /// first), invalidate the intersecting verification receipts, and record a
    /// durable `diff.reverted` event.
    pub async fn revert_diff(&self, diff_id: &str) -> Result<DiffProposal> {
        Self::gated_effect("revert_diff")?;
        let kv = &self.services.key_value_store;
        let mut proposal = DiffStore::get(kv, diff_id).ok_or_else(|| unknown_diff(diff_id))?;
        let mut reverted_files: Vec<String> = Vec::new();
        for i in (0..proposal.hunks.len()).rev() {
            if proposal.hunks[i].status == HunkStatus::Rejected {
                continue;
            }
            let (file, before, after) = {
                let h = &proposal.hunks[i];
                (h.file.clone(), h.before.clone(), h.after.clone())
            };
            self.inverse_write(&proposal.session_id, &file, &before, &after)
                .await?;
            proposal.hunks[i].status = HunkStatus::Rejected;
            reverted_files.push(file);
        }
        DiffStore::put(kv, &proposal)?;
        self.record_diff_event(&proposal, "diff.reverted", None)
            .await?;
        self.invalidate_verifications_for_files(&proposal.session_id, &reverted_files)
            .await?;
        self.publish_diff(&proposal);
        Ok(proposal)
    }

    /// Write `before` back to `file` through the registered `edit.write_file` tool
    /// (the same verifying applier the agent uses), guarded by
    /// `base_hash == blake3(after)` so a file changed since the edit conflicts
    /// instead of being clobbered.
    ///
    /// ponytail: reverting a newly created file writes an empty file rather than
    /// deleting it. Delete-on-revert when a created-file hunk needs true undo.
    pub(crate) async fn invalidate_verifications_for_files(
        &self,
        session: &SessionId,
        files: &[String],
    ) -> Result<()> {
        if files.is_empty() {
            return Ok(());
        }
        let receipts = self.verification_receipts(session).await?;
        let already = self.invalidated_verification_ids(session).await?;
        let stale: Vec<String> = receipts
            .iter()
            .filter(|r| scopes_intersect(&r.receipt.scope, files))
            .map(|r| r.receipt.verification_id.clone())
            .filter(|id| !already.contains(id))
            .collect();
        if stale.is_empty() {
            return Ok(());
        }
        self.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "verify.invalidated",
                json!({ "verification_ids": stale, "scope": files, "reason": "diff hunk rejected" }),
            ))
            .await?;
        Ok(())
    }

    /// The verification ids marked invalidated for a session (folded from
    /// `verify.invalidated` events). A receipt whose id is here should be rerun.
    pub async fn invalidated_verification_ids(&self, session: &SessionId) -> Result<Vec<String>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        let mut out = Vec::new();
        for e in events {
            if e.kind == "verify.invalidated" {
                if let Some(ids) = e.payload.get("verification_ids").and_then(|v| v.as_array()) {
                    out.extend(ids.iter().filter_map(|v| v.as_str().map(str::to_string)));
                }
            }
        }
        Ok(out)
    }

    /// Export a sealed review receipt over a diff (census sec 23): the hunks with
    /// their accept/reject status + provenance, plus the verification receipts
    /// before and after the review. Sealed with a blake3 over the canonical body
    /// and recorded as a durable `diff.receipt` event; read back via
    /// [`Self::diff_review_receipts`].
    pub async fn export_diff_review_receipt(
        &self,
        diff_id: &str,
        verification_before: Vec<VerificationReceipt>,
        verification_after: Vec<VerificationReceipt>,
    ) -> Result<DiffReviewReceipt> {
        let proposal = self
            .diff_get(diff_id)
            .ok_or_else(|| unknown_diff(diff_id))?;
        let sealed_ms = hide_core::ids::now_ms();
        let body = json!({
            "diff_id": proposal.diff_id,
            "run_id": proposal.run_id,
            "hunks": proposal.hunks,
            "verification_before": verification_before,
            "verification_after": verification_after,
            "sealed_ms": sealed_ms,
        });
        let seal = blake3::hash(serde_json::to_string(&body).unwrap_or_default().as_bytes())
            .to_hex()
            .to_string();
        let receipt = DiffReviewReceipt {
            diff_id: proposal.diff_id.clone(),
            run_id: proposal.run_id.clone(),
            hunks: proposal.hunks.clone(),
            verification_before,
            verification_after,
            sealed_ms,
            seal,
        };
        self.services
            .event_log
            .append(NewEvent::system(
                proposal.session_id.clone(),
                "diff.receipt",
                serde_json::to_value(&receipt).unwrap_or(Value::Null),
            ))
            .await?;
        Ok(receipt)
    }

    /// The wire arm for [`Self::export_diff_review_receipt`] (`{ diff_id, session_id? }`): seal the
    /// diff's hunks with the session's verification receipts and publish the sealed record.
    ///
    /// The before/after split is the diff's own `created_ms`: a receipt sealed before the first
    /// hunk of this diff was recorded verified the pre-review tree, and one sealed after verified
    /// the reviewed tree. No client input picks the split, so two exports of the same diff seal the
    /// same body.
    pub async fn diff_review_receipts(
        &self,
        session: &SessionId,
    ) -> Result<Vec<DiffReviewReceipt>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(events
            .into_iter()
            .filter(|e| e.kind == "diff.receipt")
            .filter_map(|e| e.payload_as::<DiffReviewReceipt>())
            .collect())
    }

    // --- Durable GOAL (bible sec 14, sec 78.1 #3) ---

    /// Set (or replace) a session's durable GOAL: a persisted completion
    /// `condition` + a STRUCTURED, model-free `acceptance` (oracle names whose
    /// latest `verify.result` verdict must be `Pass`). The record is written to the
    /// KV `goals` namespace keyed by session, so it survives a workspace reopen.
    /// Surfaces a `goal_set` UiEvent under the session.
    pub fn goal_set(
        &self,
        session: SessionId,
        condition: impl Into<String>,
        acceptance: Vec<String>,
    ) -> Result<GoalRecord> {
        let record =
            GoalRecord::active(GoalStore::new_id(&session), session, condition, acceptance);
        GoalStore::put(&self.services.key_value_store, &record)?;
        self.publish_goal(&record, "goal_set");
        Ok(record)
    }

    /// The session's durable goal, if one is set.
    pub fn goal_get(&self, session: &SessionId) -> Option<GoalRecord> {
        GoalStore::get(&self.services.key_value_store, session)
    }

    /// Retire a session's goal: flip its status to `Cleared` (durably) and return
    /// the cleared record. `None` when no goal was set. Surfaces a `goal_cleared`
    /// UiEvent.
    pub fn goal_clear(&self, session: &SessionId) -> Result<Option<GoalRecord>> {
        let kv = &self.services.key_value_store;
        match GoalStore::get(kv, session) {
            Some(mut record) => {
                record.status = GoalStatus::Cleared;
                record.updated_ms = hide_core::ids::now_ms();
                GoalStore::put(kv, &record)?;
                self.publish_goal(&record, "goal_cleared");
                Ok(Some(record))
            }
            None => Ok(None),
        }
    }

    /// DETERMINISTICALLY evaluate a session's goal against durable evidence in the
    /// event log -- NO model. The acceptance (oracle names) is checked against the
    /// LATEST `verify.result` verdict for each named oracle in the session; an
    /// empty acceptance falls back to the session's latest verification verdict.
    /// The verdict carries the outcome (`Met`/`NotMet`/`DeferredModelRequired`), a
    /// reason, and the ids of the evidence events consulted.
    ///
    /// A natural-language / model-judged condition is `DEFERRED_MODEL_REQUIRED`:
    /// this path never loads a model. When the outcome is `Met`, the goal's durable
    /// status is advanced to `Met` and a `goal_met` UiEvent is surfaced.
    ///
    /// Errors with `NotFound` when no goal is set for the session.
    pub async fn goal_evaluate(&self, session: &SessionId) -> Result<GoalVerdict> {
        let kv = &self.services.key_value_store;
        let mut goal = GoalStore::get(kv, session).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!("no goal set for session {session}"))
        })?;
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        let verdict = evaluate_goal(&goal, &events);
        // Advance + surface a Met transition durably (idempotent): only when the
        // goal is not already Met (and not deliberately Cleared).
        if verdict.outcome == GoalOutcome::Met && goal.status == GoalStatus::Active {
            goal.status = GoalStatus::Met;
            goal.updated_ms = hide_core::ids::now_ms();
            GoalStore::put(kv, &goal)?;
            self.publish_goal_met(&goal, &verdict);
        }
        Ok(verdict)
    }

    /// Publish a goal-lifecycle UiEvent (`goal_set` / `goal_cleared`) carrying the
    /// record, under the goal's session.
    pub async fn checkpoint_create(
        &self,
        session: SessionId,
        at_event: Option<&EventId>,
        label: impl Into<String>,
    ) -> Result<CheckpointRecord> {
        let at_seq = match at_event {
            Some(id) => self.replay.seq_of_event(session.clone(), id).await?,
            None => self.replay.latest_seq(session.clone()).await?,
        };
        let coverage = self.compute_coverage(&session, at_seq).await?;
        let record = CheckpointRecord::seal(
            CheckpointStore::new_id(&session, at_seq),
            session,
            at_event.cloned(),
            at_seq,
            label,
            coverage,
        );
        CheckpointStore::put(&self.services.key_value_store, &record)?;
        // Durable as well as live. The `checkpoint_created` publish below is bus-only, so a browser
        // reload lost the id that seven of the ten history verbs address while the record itself was
        // still on disk. Recorded HERE, the one place a CheckpointRecord is ever minted, so every
        // client that catches up (`replay::event_to_ui_event` maps `checkpoint.created`) gets it
        // back rather than each surface needing its own read.
        self.services
            .event_log
            .append(NewEvent::system(
                record.session_id.clone(),
                "checkpoint.created",
                serde_json::to_value(&record).unwrap_or_else(|_| json!({})),
            ))
            .await?;
        self.publish_checkpoint(&record, "checkpoint_created");
        Ok(record)
    }

    /// Compute the [`CheckpointCoverage`] references at a boundary (bible sec
    /// 15.4; consolidation Trace E): the code (repo) state, the thread and plan
    /// state (folded from the log at the boundary), the goal in force, and the
    /// artifact references. Model-free: a live model-state capsule stays
    /// `DEFERRED_MODEL_REQUIRED` and is recorded as `None`.
    pub(crate) async fn compute_coverage(
        &self,
        session: &SessionId,
        at_seq: u64,
    ) -> Result<CheckpointCoverage> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        // Thread + plan come from the projection folded to the boundary (reusing
        // the time-travel scrub); repo + artifacts fold the log directly.
        let projection = self.replay.scrub_to_event(session.clone(), at_seq).await?;
        let code = rewind::code_state(&events, Some(at_seq));
        let repo_items: Vec<String> = code.iter().map(|(f, h)| format!("{f}:{h}")).collect();
        let plan = match projection.plan.as_ref() {
            Some(p) => {
                StateRef::counted(p.steps.len(), &serde_json::to_string(p).unwrap_or_default())
            }
            None => StateRef::default(),
        };
        let goal =
            GoalStore::get(&self.services.key_value_store, session).map(|g| rewind::GoalRef {
                goal_id: g.goal_id,
                status: format!("{:?}", g.status),
                condition: g.condition,
            });
        let artifacts = rewind::artifact_refs(&events, Some(at_seq));
        Ok(CheckpointCoverage {
            repo_state: StateRef::of(&repo_items),
            thread: StateRef::of(&projection.transcript),
            plan,
            goal,
            artifacts: StateRef::of(&artifacts),
            live_state_capsule: None, // DEFERRED_MODEL_REQUIRED
        })
    }

    /// Every durable checkpoint for a session, ordered deterministically.
    pub fn checkpoint_list(&self, session: &SessionId) -> Vec<CheckpointRecord> {
        CheckpointStore::list_for_session(&self.services.key_value_store, session)
    }

    /// Release (delete) a durable checkpoint by id. Idempotent: an unknown id is
    /// a no-op success. This is the host-side authority for RPC `state/release`
    /// (and does not involve the superseded `hide-state` capsule crate).
    pub fn checkpoint_release(&self, checkpoint_id: &str) -> Result<()> {
        CheckpointStore::delete(&self.services.key_value_store, checkpoint_id)
    }

    /// Restore a CHECKPOINT: produce a NEW session whose durable history is the
    /// checkpoint's source folded up to (and including) the checkpoint boundary.
    /// The integrity digest is VERIFIED first (a tampered boundary errors); an
    /// unknown checkpoint id errors with `NotFound`. Independence + fold reuse
    /// [`BackendReplayService::fork_session`] exactly as the fork path does, so the
    /// source is untouched. Ancestry (parent = the checkpoint's source + the
    /// boundary) is recorded in the KV `session_records` namespace. Surfaces a
    /// `checkpoint_restored` UiEvent under the restored session.
    pub async fn checkpoint_restore(
        &self,
        checkpoint_id: &str,
    ) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
        Self::gated_effect("checkpoint_restore")?;
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        if !record.verify_integrity() {
            return Err(hide_core::error::HideError::InvalidState(format!(
                "checkpoint {checkpoint_id} failed integrity check (boundary tampered)"
            )));
        }
        // Fold the source up to the sealed boundary into a fresh, independent
        // lineage (reuses the fork machinery), then record ancestry pointing back
        // at the checkpoint's source + boundary.
        let (restored, projection) = self
            .replay
            .fork_session(record.session_id.clone(), record.at_seq)
            .await?;
        let ancestry = crate::services::SessionRecord::fork(
            restored.clone(),
            record.session_id.clone(),
            record.at_seq,
            record.at_event.clone(),
        );
        self.services
            .sessions
            .record_session(&self.services.key_value_store, &ancestry);
        self.publish_checkpoint_restored(&restored, &record, &ancestry);
        Ok((restored, ancestry, projection))
    }

    /// Publish a `checkpoint_created` UiEvent carrying the record, under its session.
    pub(crate) fn fork_marker(
        &self,
        parent: &SessionId,
        inherited: usize,
        at_seq: u64,
    ) -> (ForkPoint, NewEvent) {
        let fp = ForkPoint::new(parent.clone(), inherited, at_seq);
        let marker = NewEvent::system(
            parent.clone(),
            rewind::FORK_POINT_KIND,
            serde_json::to_value(&fp).unwrap_or(Value::Null),
        );
        (fp, marker)
    }

    /// REWIND a domain (code / conversation / both) back to a checkpoint boundary
    /// into a fresh, independent child session (consolidation Trace E). The child
    /// re-materializes the checkpoint prefix (inherited context, behind a
    /// [`ForkPoint`] marker) plus every post-boundary event whose domain the target
    /// does NOT revert, so a code-only rewind reverts the code while PRESERVING the
    /// conversation (and vice versa). Reports the reverted files and the
    /// verification receipts the rewind invalidates (post-boundary receipts whose
    /// file scope intersects a reverted file, using the same path-intersection the
    /// verify authority rule uses).
    ///
    /// A `Code` or `Both` rewind ALSO reverts the working tree: every post-boundary
    /// hunk is rejected newest-first through [`Self::reject_hunk`], i.e. the same
    /// verifying inverse write the diff reject path uses, so the files on disk
    /// really do go back to the boundary. The source session's history is not
    /// rewritten; the disk revert is recorded on it as ordinary `diff.hunk.rejected`
    /// events, exactly as if the hunks had been rejected by review. Model-free.
    pub async fn checkpoint_rewind(
        &self,
        checkpoint_id: &str,
        target: RewindTarget,
    ) -> Result<RewindOutcome> {
        Self::gated_effect("checkpoint_rewind")?;
        let record = self.load_verified_checkpoint(checkpoint_id)?;
        let source = record.session_id.clone();
        let at_seq = record.at_seq;
        let events = self
            .services
            .event_log
            .scan(Some(source.clone()), None, None)
            .await?;

        // What a code rewind reverts: files changed between the boundary and the
        // tail (a conversation-only rewind reverts no code).
        let base = rewind::code_state(&events, Some(at_seq));
        let head = rewind::code_state(&events, None);
        let reverted_files = match target {
            RewindTarget::Conversation => Vec::new(),
            RewindTarget::Code | RewindTarget::Both => rewind::changed_files(&base, &head),
        };
        let receipts = rewind::receipt_scopes(&events, at_seq);
        let invalidated_receipts = rewind::invalidated_receipts(&reverted_files, &receipts);

        // Revert those files ON DISK before minting the child, newest hunk first so
        // later edits to the same file peel off first. Same verifying inverse write
        // as the diff reject path, so a file changed since the edit CONFLICTS and
        // fails the rewind closed instead of being clobbered.
        // ponytail: no transaction, so a conflict part way leaves the earlier files
        // reverted (the same exposure `revert_diff` already has). Add a staged
        // write-back if a partially reverted tree ever becomes a real problem.
        if target != RewindTarget::Conversation {
            for (diff_id, hunk_id) in rewind::post_boundary_hunks(&events, at_seq)
                .into_iter()
                .rev()
            {
                let already_reverted = self.diff_get(&diff_id).is_some_and(|p| {
                    p.hunks
                        .iter()
                        .any(|h| h.hunk_id == hunk_id && h.status == HunkStatus::Rejected)
                });
                if !already_reverted {
                    self.reject_hunk(&diff_id, &hunk_id).await?;
                }
            }
        }

        // Seed a fresh lineage from the surviving events behind a fork marker.
        let child_events = rewind::rewind_child_events(&events, at_seq, target);
        let inherited = rewind::inherited_len(&events, at_seq);
        let (fork_point, marker) = self.fork_marker(&source, inherited, at_seq);
        let (child, projection) = self
            .replay
            .seed_child_session(Some(marker), &child_events)
            .await?;

        let ancestry = crate::services::SessionRecord::fork(
            child.clone(),
            source.clone(),
            at_seq,
            record.at_event.clone(),
        );
        self.services
            .sessions
            .record_session(&self.services.key_value_store, &ancestry);
        self.publish_checkpoint_child(
            "checkpoint_rewound",
            &child,
            &record,
            json!({
                "target": target,
                "reverted_files": reverted_files,
                "invalidated_receipts": invalidated_receipts,
                "fork_point": fork_point,
            }),
        );
        Ok(RewindOutcome {
            session_id: child,
            target,
            fork_point,
            reverted_files,
            invalidated_receipts,
            projection,
            ancestry,
        })
    }

    /// REPLAY from a checkpoint: re-apply the whole recorded history from the
    /// checkpoint forward onto a fresh, independent lineage seeded at the
    /// checkpoint (behind a [`ForkPoint`] marker). The post-boundary source events
    /// are the replayed set (the child's own records). Unlike a rewind, replay
    /// drops nothing. Model-free.
    pub async fn checkpoint_replay(&self, checkpoint_id: &str) -> Result<ReplayOutcome> {
        let record = self.load_verified_checkpoint(checkpoint_id)?;
        let source = record.session_id.clone();
        let at_seq = record.at_seq;
        let events = self
            .services
            .event_log
            .scan(Some(source.clone()), None, None)
            .await?;
        let child_events: Vec<&Event> = events.iter().collect();
        let replayed_events: Vec<EventId> = events
            .iter()
            .filter(|e| e.seq > at_seq)
            .map(|e| e.id.clone())
            .collect();
        let inherited = rewind::inherited_len(&events, at_seq);
        let (fork_point, marker) = self.fork_marker(&source, inherited, at_seq);
        let (child, projection) = self
            .replay
            .seed_child_session(Some(marker), &child_events)
            .await?;
        let ancestry = crate::services::SessionRecord::fork(
            child.clone(),
            source.clone(),
            at_seq,
            record.at_event.clone(),
        );
        self.services
            .sessions
            .record_session(&self.services.key_value_store, &ancestry);
        self.publish_checkpoint_child(
            "checkpoint_replayed",
            &child,
            &record,
            json!({ "replayed": replayed_events.len(), "fork_point": fork_point }),
        );
        Ok(ReplayOutcome {
            session_id: child,
            fork_point,
            replayed_events,
            projection,
            ancestry,
        })
    }

    /// FORK from a checkpoint into an ephemeral branch: a new lineage seeded ONLY
    /// with the checkpoint's inherited prefix (behind a [`ForkPoint`] marker), to
    /// explore an alternative from the boundary with no post-boundary carry-over.
    /// Recorded as an [`SessionRelationship::EphemeralFork`](crate::services::SessionRelationship)
    /// so a client can prune it without ceremony. Model-free.
    pub async fn checkpoint_fork(&self, checkpoint_id: &str) -> Result<ForkOutcome> {
        let record = self.load_verified_checkpoint(checkpoint_id)?;
        let source = record.session_id.clone();
        let at_seq = record.at_seq;
        let events = self
            .services
            .event_log
            .scan(Some(source.clone()), None, None)
            .await?;
        let child_events: Vec<&Event> = events.iter().filter(|e| e.seq <= at_seq).collect();
        let inherited = child_events.len();
        let (fork_point, marker) = self.fork_marker(&source, inherited, at_seq);
        let (child, projection) = self
            .replay
            .seed_child_session(Some(marker), &child_events)
            .await?;
        let ancestry = crate::services::SessionRecord::ephemeral_fork(
            child.clone(),
            source.clone(),
            at_seq,
            record.at_event.clone(),
        );
        self.services
            .sessions
            .record_session(&self.services.key_value_store, &ancestry);
        self.publish_checkpoint_child(
            "checkpoint_forked",
            &child,
            &record,
            json!({ "fork_point": fork_point }),
        );
        Ok(ForkOutcome {
            session_id: child,
            fork_point,
            projection,
            ancestry,
        })
    }

    /// COMPARE a session's current code state against a checkpoint's boundary code
    /// state (current-versus-checkpoint): the file-level added/removed/modified
    /// changes. Model-free.
    pub async fn checkpoint_inspect(&self, checkpoint_id: &str) -> Result<CheckpointInspection> {
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        let integrity_ok = record.verify_integrity();
        let current = self
            .compute_coverage(&record.session_id, record.at_seq)
            .await?;
        let drift = coverage_drift(&record.coverage, &current);

        let events = self
            .services
            .event_log
            .scan(Some(record.session_id.clone()), None, None)
            .await?;
        let base = rewind::code_state(&events, Some(record.at_seq));
        let head = rewind::code_state(&events, None);
        let reverted_files = rewind::changed_files(&base, &head);
        let receipts = rewind::receipt_scopes(&events, record.at_seq);
        let invalidated_receipts = rewind::invalidated_receipts(&reverted_files, &receipts);

        Ok(CheckpointInspection {
            checkpoint_id: record.checkpoint_id.clone(),
            integrity_ok,
            coverage_current: drift.is_empty(),
            drift,
            reverted_files,
            invalidated_receipts,
            coverage: record.coverage.clone(),
        })
    }
}
