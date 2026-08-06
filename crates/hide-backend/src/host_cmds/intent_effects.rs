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
    pub(crate) fn runtime_base_url(&self) -> Option<String> {
        let sup = self.runtime.as_ref()?;
        if sup.state() == RuntimeSupervisorState::Ready {
            let url = sup.base_url()?;
            self.install_runtime_embedder(&url);
            Some(url)
        } else {
            None
        }
    }

    /// Wire the live embeddings endpoint into SqliteCodeIndex when present.
    pub(crate) fn install_runtime_embedder(&self, base_url: &str) {
        if let Some(sqlite) = self.services.sqlite_index.as_ref() {
            let client: Arc<dyn hawking_index::EmbeddingClient> = Arc::new(
                hawking_index::HttpEmbeddingClient::new(base_url.to_string()),
            );
            sqlite.set_embedder(Some(client));
        }
    }

    /// Handle a Wire-A intent. The `IntentAck` is returned SYNCHRONOUSLY (the
    /// contract is unchanged); generation, when an accepted `SubmitTurn`
    /// triggers it, is spawned as a background task that streams tokens onto the
    /// Wire-B bus. The ack does not wait for generation.
    pub(crate) fn effect_failed(&self, ack: &mut IntentAck, code: &str, message: String) {
        ack.accepted = false;
        ack.message = Some(message.clone());
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: None,
            kind: UiEventKind::Error {
                code: code.to_string(),
                message,
            },
        });
    }

    /// Dispatch `run_static_analysis` (bible Book IX sec 28-29). Payload:
    /// `{ session_id?, sources: [{path,text}] }` (the editor's live buffers) or
    /// `{ session_id?, paths: [workspace-relative] }` (read from disk, confined to the root).
    pub(crate) fn emit_new_session(&self) {
        let sid = SessionId::new();
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(sid),
            kind: UiEventKind::ProjectionPatch {
                projection: "turn".to_string(),
                patch: json!({ "phase": "idle", "run_id": Value::Null }),
            },
        });
    }

    /// YOU / CHAT / IDE surface graph intents. Switch is a lens change on the
    /// primary session; handoffs seal or open claim capsules only.
    pub(crate) fn shell_config(&self) -> hide_kernel::tooling::ShellConfig {
        hide_kernel::tooling::ShellConfig {
            workspace_root: Some(
                self.services
                    .config
                    .workspace_root
                    .to_string_lossy()
                    .into_owned(),
            ),
            hide_dir: Some(self.services.layout().hide_dir),
            // Nested agent/CI seats break nested sandbox-exec. Unit tests assert
            // process lifecycle and streaming; SBPL profile coverage is in
            // hide-kernel::security. Production builds keep confinement (false).
            disable_sandbox: cfg!(test),
            ..Default::default()
        }
    }

    /// Park an effect at the security gate and announce it: the ONE place an action becomes
    /// "held", so every held effect is announced the same way and a book that cannot take another
    /// decision refuses instead of dropping one silently.
    pub(crate) async fn run_approved_intent(&self, name: &str, payload: &Value) -> Result<()> {
        crate::tools::with_approved_writes(self.released_effect(name, payload)).await
    }

    pub(crate) fn effect_command(intent: &Intent) -> Option<(String, Value)> {
        match intent {
            Intent::RejectDiff {
                diff_id,
                hunk_id: None,
                ..
            } => Some(("revert_diff".to_string(), json!({ "diff_id": diff_id }))),
            Intent::Custom { name, payload } => Some((name.clone(), payload.clone())),
            _ => None,
        }
    }

    /// Whether the command authority marks this command [`ApprovalPolicy::Ask`], i.e. its effect
    /// may not run until a human approves. Read straight off the ONE registry so a policy change in
    /// the catalog is enforced without a second list to keep in sync. Binding-agnostic on purpose:
    /// filtering to `Custom` bindings meant no `Intent`-bound row could ever be enforced whatever
    /// policy it declared.
    pub(crate) fn gated_effect(name: &str) -> Result<()> {
        if Self::requires_approval(name) && !crate::tools::gate_released() {
            return Err(hide_core::error::HideError::PolicyDenied(format!(
                "{name} requires approval: send it as an intent so it is held at the security gate"
            )));
        }
        Ok(())
    }

    /// Deny a held gated command: drop it without running. An unknown gate is refused, for the
    /// same reason approving one is: the caller is answering something that is not there.
    pub(crate) fn deny_gate(&self, gate: &str) -> Result<()> {
        if self.gate_book.remove(gate) {
            return Ok(());
        }
        Err(hide_core::error::HideError::NotFound(format!(
            "gate {gate} is not awaiting a decision (already answered, denied, or never held)"
        )))
    }

    /// The count of commands currently parked awaiting an approve/deny decision (test/inspection).
    #[cfg(test)]
    pub(crate) fn pending_gate_count(&self) -> usize {
        self.gate_book.len()
    }

    /// Increment 2 (defect S1): build the fully-wired agent kernel a live
    /// `SubmitTurn` routes through - the REAL loop, not the minimal
    /// [`AgentKernel::new`] (StubPlanner + no oracles) the host held before.
    /// Mirrors the working recipe in `hide-kernel/tests/full_run.rs`:
    ///
    /// * `runtime` - a [`KernelRuntimeClient`] over a [`SimpleRouter`] and the
    ///   host's HTTP [`ModelProviderInferenceClient`], so `.runtime(..)` also
    ///   auto-installs a `RuntimePlanner` (the model plans, we own acceptance).
    /// * `dispatcher` - a permission-gated [`ToolDispatcher`] built from the
    ///   host's tool registry + the config's **real** permission engine. NOT
    ///   `allow_all_dispatcher`, which bypasses permissions.
    /// * `grounding` - codebase [`Grounding`] over the code index.
    /// * `autonomy` - a BOUNDED level ([`turn_kernel_autonomy`] defaults to
    ///   `SuggestOnly`) so an effectful step pauses for approval rather than
    ///   running an unsandboxed shell unattended; `HIDE_KERNEL_AUTONOMY` widens it.
    /// * `with_standard_oracles` - the deterministic build/typecheck/test/lint
    ///   oracles (no state advances on faith, K1).
    pub(crate) fn publish_custom(&self, session_id: Option<SessionId>, data: Value) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id,
            kind: UiEventKind::Custom(data),
        });
    }

    /// The run a session's editor saves are grouped under, so every save lands on ONE addressable
    /// [`DiffProposal`] (`diff-editor-<session>`) instead of a diff per keystroke-save. Stable per
    /// session and derived, not stored, so a restart addresses the same diff.
    pub(crate) fn publish_search_results(
        &self,
        payload: &Value,
        hits: &[crate::replay::TranscriptHit],
    ) {
        let query = payload
            .get("query")
            .or_else(|| payload.get("text"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: None,
            kind: UiEventKind::Custom(json!({
                "kind": "search_results",
                "query": query,
                "count": hits.len(),
                "hits": serde_json::to_value(hits).unwrap_or_else(|_| json!([])),
            })),
        });
    }

    /// The conversation graph (bible sec 32-33) rooted at `session_id`: the node,
    /// its ancestry chain (to a root), and its direct children (forks / side chats
    /// / ephemeral forks), with parent->child edges. A bounded, DETERMINISTIC
    /// projection over the durable `session_records` KV -- no model, safe headless.
    pub(crate) fn publish_diagnostics(&self, record: &StaticAnalysisReceipt, session: &SessionId) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(session.clone()),
            kind: UiEventKind::ProjectionPatch {
                projection: "diagnostics".to_string(),
                patch: record.diagnostics_projection(),
            },
        });
    }

    /// Publish a `verification_receipt` UiEvent carrying the receipt + a
    /// findings-summary, under the analyzed session.
    pub(crate) fn publish_verification(&self, record: &StaticAnalysisReceipt, session: &SessionId) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(session.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "verification_receipt",
                "summary": record.findings_summary(),
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Hunk-addressable diff review (census sec 23) ---

    /// Capture one applied `edit.*` call as an addressable hunk on the run's
    /// [`DiffProposal`] (creating it on the first edit of the run). The edit has
    /// ALREADY been written to disk by the verifying applier; this records the
    /// whole-file pre-image/post-image so the change can later be kept or reverted
    /// per hunk. Appends a durable `diff.proposed` event and republishes the diff
    /// projection. Called from [`Self::dispatch_tool`] for every successful edit
    /// under a run.
    /// Read a diff's hunks WITH provenance + base hash (census sec 23 reader).
    pub(crate) fn publish_diff(&self, proposal: &DiffProposal) {
        publish_diff_to(&self.ui_bus, proposal);
    }

    /// Mark the verification receipts whose scope intersects any of `files` as
    /// invalidated (census sec 23): append a durable `verify.invalidated` event
    /// naming the affected verification ids + scope so a rerun is warranted. Reuses
    /// the same scope-intersection logic ([`scopes_intersect`] /
    /// `hide_kernel::verify_plane::paths_intersect`) as [`Self::reconcile_review_for_scope`].
    /// Model-free.
    pub(crate) fn publish_goal(&self, record: &GoalRecord, kind: &str) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(record.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Publish a `goal_met` UiEvent carrying the record + the evaluation verdict.
    pub(crate) fn publish_goal_met(&self, record: &GoalRecord, verdict: &GoalVerdict) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(record.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "goal_met",
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
                "verdict": serde_json::to_value(verdict).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Durable CHECKPOINT (bible sec 15.4, sec 78.1 #3) ---

    /// Create a durable CHECKPOINT: a named restore boundary over a session's
    /// event-sourced history. The boundary is `at_event` (resolved strictly;
    /// `NotFound` if absent) or, when `None`, the session's current tail. The
    /// record seals a blake3 `integrity` digest over its boundary identity
    /// (session + seq + boundary event) so a later restore can prove the boundary
    /// was not tampered. Written to the KV `checkpoints` namespace; surfaces a
    /// `checkpoint_created` UiEvent under the session.
    pub(crate) fn load_verified_checkpoint(&self, checkpoint_id: &str) -> Result<CheckpointRecord> {
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        if !record.verify_integrity() {
            return Err(hide_core::error::HideError::InvalidState(format!(
                "checkpoint {checkpoint_id} failed integrity check (boundary or coverage tampered)"
            )));
        }
        Ok(record)
    }

    /// The code (repo) state of a session, folding `diff.proposed` up to `up_to`
    /// (or the tail when `None`): file -> latest content hash.
    pub(crate) async fn code_state_of(
        &self,
        session: &SessionId,
        up_to: Option<u64>,
    ) -> Result<std::collections::BTreeMap<String, String>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(rewind::code_state(&events, up_to))
    }

    /// Build the fork-boundary marker for a child inheriting `inherited` prefix
    /// events up to parent seq `at_seq`.
    pub async fn compare_to_checkpoint(
        &self,
        checkpoint_id: &str,
        session: &SessionId,
    ) -> Result<CodeComparison> {
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        let base = self
            .code_state_of(&record.session_id, Some(record.at_seq))
            .await?;
        let head = self.code_state_of(session, None).await?;
        Ok(CodeComparison {
            base: format!("checkpoint:{}", record.checkpoint_id),
            head: format!("session:{}", session.as_str()),
            files: rewind::diff_code_states(&base, &head),
        })
    }

    /// COMPARE two sessions' current code states (compare branches). Model-free.
    pub async fn compare_session_code(
        &self,
        a: &SessionId,
        b: &SessionId,
    ) -> Result<CodeComparison> {
        let base = self.code_state_of(a, None).await?;
        let head = self.code_state_of(b, None).await?;
        Ok(CodeComparison {
            base: format!("session:{}", a.as_str()),
            head: format!("session:{}", b.as_str()),
            files: rewind::diff_code_states(&base, &head),
        })
    }

    /// INSPECT a checkpoint's integrity + coverage (consolidation Trace E part d):
    /// whether the sealed digest verifies, whether the coverage recomputed from the
    /// current source log still matches (drift detection), and which verification
    /// receipts a code rewind from here would invalidate. Model-free.
    pub(crate) fn publish_job(&self, record: &JobRecord, kind: &str) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(record.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Stage 4: durable-thread lifecycle + Initialize + background promotion ---

    /// Open a durable-thread writer (Stage 4 four-verb lifecycle) over a session's
    /// event log. Appended items are lazy until an explicit `flush` / `persist` /
    /// `shutdown`; `discard` drops them without a durable write. Wrap it in a
    /// [`crate::live_thread::LiveThreadInitGuard`] to make a failed session init
    /// discard its partial event stream.
    pub(crate) fn publish_side_chat_created(
        &self,
        new_session: &SessionId,
        record: &crate::services::SessionRecord,
    ) {
        self.ui_bus.publish(UiEvent {
            seq: record.forked_at.unwrap_or(0),
            session_id: Some(new_session.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "side_chat_created",
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Publish a `side_chat_merged` UiEvent under the PARENT (the merge lands on
    /// the parent, not the side chat), carrying both ids + the merged summary.
    pub(crate) fn publish_side_chat_merged(
        &self,
        parent: &SessionId,
        side_chat: &SessionId,
        result: &SideChatResult,
        seq: u64,
    ) {
        self.ui_bus.publish(UiEvent {
            seq,
            session_id: Some(parent.clone()),
            kind: UiEventKind::Custom(result.merged_ui_payload(parent, side_chat)),
        });
    }

    /// Perform a `create_side_chat` custom intent: spawn the side-chat creation so
    /// the intent ack returns immediately (mirrors [`Self::spawn_fork_session`]).
    /// A failure (e.g. an unknown boundary event) surfaces as an Error UiEvent.
    pub(crate) fn publish_session_forked(
        &self,
        new_session: &SessionId,
        record: &crate::services::SessionRecord,
    ) {
        self.ui_bus.publish(UiEvent {
            seq: record.forked_at.unwrap_or(0),
            session_id: Some(new_session.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "session_forked",
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }
}
