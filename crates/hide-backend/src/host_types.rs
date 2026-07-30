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
    pub fn verdict(&self) -> &hide_kernel::verify_plane::Verdict {
        &self.receipt.verdict
    }

    /// Whether the deterministic verdict passed.
    pub fn is_pass(&self) -> bool {
        self.receipt.verdict.is_pass()
    }

    /// A compact human-readable count of findings by severity (e.g.
    /// `"2 error, 1 warning"`), or `"no findings"` when clean.
    pub fn findings_summary(&self) -> String {
        use hide_kernel::verify_plane::Severity;
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
        use hide_kernel::verify_plane::Severity;
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

pub(crate) fn default_side_chat_kind() -> String {
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
    pub(crate) fn merge_event_payload(&self, side_chat: &SessionId) -> Value {
        json!({
            "side_chat": side_chat.as_str(),
            "summary": self.summary,
            "evidence": self.evidence,
            "kind": self.kind,
        })
    }

    /// The `side_chat_merged` UiEvent payload (under the PARENT).
    pub(crate) fn merged_ui_payload(&self, parent: &SessionId, side_chat: &SessionId) -> Value {
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
// turn (`hide_kernel::tooling::edit::run_plan` -> `std::fs::write`). So a "diff" is the set
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
    pub(crate) fn hunk(&self, hunk_id: &str) -> Option<&DiffHunk> {
        self.hunks.iter().find(|h| h.hunk_id == hunk_id)
    }
    pub(crate) fn hunk_mut(&mut self, hunk_id: &str) -> Option<&mut DiffHunk> {
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
pub(crate) fn coverage_drift(
    sealed: &CheckpointCoverage,
    current: &CheckpointCoverage,
) -> Vec<String> {
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
pub(crate) struct DiffStore;

impl DiffStore {
    const NAMESPACE: &'static str = "diffs";

    pub(crate) fn put(
        kv: &hide_core::persistence::DynKeyValueStore,
        record: &DiffProposal,
    ) -> Result<()> {
        kv.put(
            Self::NAMESPACE,
            &record.diff_id,
            serde_json::to_value(record)?,
        )
    }

    pub(crate) fn get(
        kv: &hide_core::persistence::DynKeyValueStore,
        diff_id: &str,
    ) -> Option<DiffProposal> {
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
pub(crate) const HANDLED_CUSTOM_NAMES: &[&str] = &[
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
    "fleet_run",
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
    // YOU / CHAT / IDE shared session graph (claim-only handoffs).
    "switch_surface",
    "handoff_create",
    "handoff_receive",
];
