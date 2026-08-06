use crate::personalize::{
    DynPersonalizationStore, InMemoryPersonalizationStore, JsonlPersonalizationStore,
};
use hawking_context::{
    ClassedMemorySystem, ContextCompiler, DynClassedMemory, InMemoryMemoryStore, MemoryStore,
    SqliteMemoryStore, TokenCounter,
};
use hawking_index::{CodeIndex, InMemoryCodeIndex, SqliteCodeIndex};
use hawking_orch::RoleRegistry;
use hawking_research::{DynResearchLedger, InMemoryResearchLedger, JsonlResearchLedger};
use hide_core::config::HideConfig;
use hide_core::event::JsonlEventLog;
use hide_core::ids::{now_ms, EventId, SessionId};
use hide_core::persistence::{
    DynBlobStore, DynEventLog, DynEventLogIntegrity, DynKeyValueStore, DynProjectionStore,
    FileBlobStore, FileKeyValueStore, FileProjectionStore, InMemoryBlobStore,
    InMemoryKeyValueStore, InMemoryProjectionStore,
};
use hide_core::project::WorkspaceLayout;
use hide_core::Result;
use hide_kernel::security::audit::EventChainAuditor;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Shared code-index handle consumed by grounding / context compile / connectors.
use super::*;

// --- Durable Goal + Checkpoint records (bible sec 14, sec 15.4, sec 78.1 #3) ---

/// The lifecycle of a durable [`GoalRecord`] (bible sec 14): a persisted
/// completion condition either awaiting evidence (`Active`), satisfied by durable
/// evidence (`Met`), or retired (`Cleared`). Snake_case so it round-trips in the
/// KV store; `Active` is the default so a record written before this field
/// existed still deserializes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalStatus {
    #[default]
    Active,
    Met,
    Cleared,
}

/// A durable GOAL (bible sec 14): a persisted completion condition + acceptance +
/// status, scoped to one session. Stored in the KV `goals` namespace keyed by
/// `session_id` (one active goal per session; a re-`goal_set` replaces it). The
/// `condition` is a human label; `acceptance` is the STRUCTURED, model-free spec:
/// a list of oracle names whose latest `verify.result` verdict must be `Pass` for
/// the goal to be `Met`. An empty `acceptance` falls back to "the latest
/// verification verdict for this session must be Pass". Natural-language / model
/// judgement of the `condition` is `DEFERRED_MODEL_REQUIRED` (see
/// [`GoalOutcome::DeferredModelRequired`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalRecord {
    pub goal_id: String,
    pub session_id: SessionId,
    /// The completion condition (a human label, e.g. `"tests_pass"`).
    pub condition: String,
    /// STRUCTURED acceptance: oracle names whose latest verdict must be `Pass`.
    /// Empty => "the latest verification verdict must be Pass".
    #[serde(default)]
    pub acceptance: Vec<String>,
    pub status: GoalStatus,
    pub created_ms: u64,
    pub updated_ms: u64,
}

impl GoalRecord {
    /// A fresh `Active` goal for a session.
    pub fn active(
        goal_id: impl Into<String>,
        session_id: SessionId,
        condition: impl Into<String>,
        acceptance: Vec<String>,
    ) -> Self {
        let now = now_ms();
        Self {
            goal_id: goal_id.into(),
            session_id,
            condition: condition.into(),
            acceptance,
            status: GoalStatus::Active,
            created_ms: now,
            updated_ms: now,
        }
    }
}

/// The deterministic outcome of a [`GoalRecord`] evaluation against durable
/// evidence. `Met`/`NotMet` are decided model-free from the session's
/// `verify.result` evidence; `DeferredModelRequired` marks a condition that would
/// need a model to judge (no model is ever called for it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GoalOutcome {
    Met,
    NotMet,
    DeferredModelRequired,
}

/// The verdict returned by `goal_evaluate`: the deterministic outcome, a
/// human-readable reason, and the event ids of the verification evidence that was
/// read (for auditability). No model; derived purely from the durable event log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GoalVerdict {
    pub goal_id: String,
    pub session_id: SessionId,
    pub outcome: GoalOutcome,
    pub reason: String,
    /// The `verify.result` event ids consulted to reach this verdict.
    #[serde(default)]
    pub evidence: Vec<EventId>,
}

impl GoalVerdict {
    pub fn is_met(&self) -> bool {
        self.outcome == GoalOutcome::Met
    }
}

/// Durable persistence for [`GoalRecord`]s over the KV store (bible sec 14). A
/// stateless facade over the `goals` namespace keyed by `session_id`, mirroring
/// how [`SessionRegistry`] wraps the `session_records` namespace.
pub struct GoalStore;

impl GoalStore {
    pub const NAMESPACE: &'static str = "goals";

    /// Mint a fresh, unique goal id (blake3 over session + wall-clock micros).
    pub fn new_id(session: &SessionId) -> String {
        subbit_id("goal", session, hide_core::ids::now_micros() as u128)
    }

    /// Durably write (or replace) a session's goal. Keyed by session id so there
    /// is one active goal per session.
    pub fn put(kv: &DynKeyValueStore, record: &GoalRecord) -> Result<()> {
        let value = serde_json::to_value(record)?;
        kv.put(Self::NAMESPACE, record.session_id.as_str(), value)
    }

    /// Look up a session's durable goal, if any.
    pub fn get(kv: &DynKeyValueStore, session: &SessionId) -> Option<GoalRecord> {
        kv.get(Self::NAMESPACE, session.as_str())
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }
}

/// A durable CHECKPOINT (bible sec 15.4; consolidation Trace E): a named restore
/// boundary over the event-sourced history of a session. It pins the boundary
/// (`at_seq` + the optional `at_event` it resolved from) and covers, beyond the
/// event boundary, a [`CheckpointCoverage`] set of references: repo state, thread
/// + plan + goal state, and artifacts (a live model-state capsule stays
/// `DEFERRED_MODEL_REQUIRED`). The `integrity` digest seals the boundary identity
/// AND the coverage, so a restore/rewind can prove neither the boundary nor any
/// covered reference was tampered before folding the source up to it. Stored in
/// the KV `checkpoints` namespace keyed by `checkpoint_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointRecord {
    pub checkpoint_id: String,
    pub session_id: SessionId,
    /// The boundary event this checkpoint was created at (`None` = the session
    /// tail at creation time).
    pub at_event: Option<EventId>,
    /// The resolved boundary `seq` (inclusive): the source is folded up to here.
    pub at_seq: u64,
    pub label: String,
    pub created_ms: u64,
    /// The references this checkpoint covers (repo / thread / plan / goal /
    /// artifacts). Defaulted so records written before coverage existed still
    /// deserialize.
    #[serde(default)]
    pub coverage: crate::rewind::CheckpointCoverage,
    /// blake3 hex over the boundary identity (session + seq + boundary event) AND
    /// the coverage digest.
    pub integrity: String,
}

impl CheckpointRecord {
    /// Build a checkpoint over a resolved boundary + its coverage, sealing the
    /// integrity digest over both.
    pub fn seal(
        checkpoint_id: impl Into<String>,
        session_id: SessionId,
        at_event: Option<EventId>,
        at_seq: u64,
        label: impl Into<String>,
        coverage: crate::rewind::CheckpointCoverage,
    ) -> Self {
        let integrity = sealed_integrity(&session_id, at_seq, at_event.as_ref(), &coverage);
        Self {
            checkpoint_id: checkpoint_id.into(),
            session_id,
            at_event,
            at_seq,
            label: label.into(),
            created_ms: now_ms(),
            coverage,
            integrity,
        }
    }

    /// Recompute the sealed digest (boundary identity + coverage) and compare it
    /// to the stored one: `true` iff BOTH the boundary and every covered reference
    /// are intact (untampered).
    pub fn verify_integrity(&self) -> bool {
        self.integrity
            == sealed_integrity(
                &self.session_id,
                self.at_seq,
                self.at_event.as_ref(),
                &self.coverage,
            )
    }
}

/// Durable persistence for [`CheckpointRecord`]s over the KV store (bible sec
/// 15.4). Keyed by `checkpoint_id`; `list_for_session` walks the namespace and
/// scopes to one session, ordered deterministically (created_ms then id).
pub struct CheckpointStore;

impl CheckpointStore {
    pub const NAMESPACE: &'static str = "checkpoints";

    /// Mint a fresh, unique checkpoint id (blake3 over session + boundary + micros).
    pub fn new_id(session: &SessionId, at_seq: u64) -> String {
        subbit_id(
            "ckpt",
            session,
            (hide_core::ids::now_micros() as u128) ^ (at_seq as u128),
        )
    }

    pub fn put(kv: &DynKeyValueStore, record: &CheckpointRecord) -> Result<()> {
        let value = serde_json::to_value(record)?;
        kv.put(Self::NAMESPACE, &record.checkpoint_id, value)
    }

    pub fn get(kv: &DynKeyValueStore, checkpoint_id: &str) -> Option<CheckpointRecord> {
        kv.get(Self::NAMESPACE, checkpoint_id)
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    /// Every checkpoint for a session, ordered deterministically (created_ms then
    /// checkpoint id) so the list is stable across runs and reopens.
    pub fn list_for_session(kv: &DynKeyValueStore, session: &SessionId) -> Vec<CheckpointRecord> {
        let mut out: Vec<CheckpointRecord> = kv
            .list(Self::NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<CheckpointRecord>(value).ok())
            .filter(|record| &record.session_id == session)
            .collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.checkpoint_id.cmp(&b.checkpoint_id))
        });
        out
    }

    /// Drop a checkpoint record from the durable store (state/release). Missing
    /// ids are a no-op success so release is idempotent.
    pub fn delete(kv: &DynKeyValueStore, checkpoint_id: &str) -> Result<()> {
        kv.delete(Self::NAMESPACE, checkpoint_id)
    }
}

/// The blake3-hex digest over a checkpoint's BOUNDARY IDENTITY: the source
/// session, the inclusive boundary `seq`, and the optional boundary event id.
/// This is what a restore recomputes and compares to prove the stored boundary
/// was not tampered (same blake3 family as the event-log chain).
pub fn checkpoint_integrity(
    session_id: &SessionId,
    at_seq: u64,
    at_event: Option<&EventId>,
) -> String {
    let material = format!(
        "{}|{}|{}",
        session_id.as_str(),
        at_seq,
        at_event.map(|e| e.as_str()).unwrap_or("")
    );
    blake3::hash(material.as_bytes()).to_hex().to_string()
}

/// The FULL sealed digest a checkpoint stores and verifies: the boundary identity
/// ([`checkpoint_integrity`]) folded with the coverage digest, so tampering EITHER
/// the boundary or any covered reference is caught by [`CheckpointRecord::verify_integrity`].
pub(crate) fn sealed_integrity(
    session_id: &SessionId,
    at_seq: u64,
    at_event: Option<&EventId>,
    coverage: &crate::rewind::CheckpointCoverage,
) -> String {
    let material = format!(
        "{}|{}",
        checkpoint_integrity(session_id, at_seq, at_event),
        coverage.digest()
    );
    blake3::hash(material.as_bytes()).to_hex().to_string()
}

/// A short, unique, prefixed id derived from a session + a wall-clock seed
/// (blake3, first 24 hex chars). Used for goal/checkpoint ids so they are stable
/// strings without pulling a separate id crate.
pub(crate) fn subbit_id(prefix: &str, session: &SessionId, seed: u128) -> String {
    let material = format!("{}|{}", session.as_str(), seed);
    let hex = blake3::hash(material.as_bytes()).to_hex();
    format!("{prefix}_{}", &hex.as_str()[..24])
}
