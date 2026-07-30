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

// --- Durable background jobs + triggers (bible sec 73-75, sec 78.1 #17) -------
//
// A durable, goal-bound background JOB that survives a restart. The RECORD, its
// TRIGGER EVALUATION (does an incoming event wake the job), and RECOVERY
// (rebuilding the active set on a fresh host) are all REAL and MODEL-FREE. The
// ACTUAL agent execution of a woken job (dispatching a turn / plan to a model,
// spawning an agent) is DEFERRED_MODEL_REQUIRED: nothing in this module ever runs
// a model or spawns an agent. Likewise, PARSING a cron [`Schedule`] and deciding
// WHEN it should fire against the wall clock is left to the caller's scheduler
// tick; here a `Time` trigger is matched deterministically by string equality
// against the spec of a fired [`TriggerEvent::Time`].

/// The id of a durable checkpoint a job has pinned (a [`CheckpointRecord::checkpoint_id`]).
/// A type alias, not a newtype, so it round-trips as a plain string in the KV store.
pub type CheckpointId = String;

/// A resource BUDGET bounding a durable [`JobRecord`]'s execution (bible sec 73).
/// Every field is optional; an unset field means "unbounded on that axis". These
/// are RECORDED bounds only, model-free; enforcing them against a live agent turn
/// is DEFERRED_MODEL_REQUIRED.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct Budget {
    /// Max wall-clock seconds the job may run.
    pub max_wall_secs: Option<u64>,
    /// Max agent steps the job may take.
    pub max_steps: Option<u32>,
    /// Max model tokens the job may consume.
    pub max_tokens: Option<u64>,
    /// Max spend, in USD millicents (1/100000 of a dollar), to avoid floats.
    pub max_usd_millicents: Option<u64>,
}

/// An optional SCHEDULE for a durable job (bible sec 74): a cron expression (e.g.
/// `"0 9 * * 1-5"`) or a one-shot ISO-8601 `at` timestamp, plus an optional
/// timezone label. The string is stored verbatim; a fired schedule tick is
/// matched deterministically against a [`Trigger::Time`] carrying the same spec.
/// PARSING the cron and computing the next fire time is DEFERRED (the scheduler
/// tick is the caller's job).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    /// A cron expression or a one-shot ISO-8601 timestamp.
    pub cron_or_at: String,
    /// An optional timezone label (e.g. `"UTC"`); display-only, model-free.
    #[serde(default)]
    pub timezone: Option<String>,
}

impl Schedule {
    /// A schedule from a cron/at spec with no timezone.
    pub fn new(cron_or_at: impl Into<String>) -> Self {
        Self {
            cron_or_at: cron_or_at.into(),
            timezone: None,
        }
    }

    pub fn with_timezone(mut self, timezone: impl Into<String>) -> Self {
        self.timezone = Some(timezone.into());
        self
    }
}

/// A durable job TRIGGER (bible sec 74-75): a condition whose matching incoming
/// event should WAKE the job. Matching is DETERMINISTIC (see [`Trigger::matches`]),
/// model-free. Snake_case + externally-tagged so it round-trips in the KV store.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Trigger {
    /// Fire on a schedule tick whose spec (cron/at string) equals this one.
    Time(String),
    /// Fire when a git push lands.
    GitPush,
    /// Fire when a pull request is opened.
    PrOpened,
    /// Fire when an issue is opened.
    IssueOpened,
    /// Fire when CI reports a failure.
    CiFailure,
    /// Fire when a changed path matches this glob (e.g. `"src/**/*.rs"`).
    FileChange(String),
    /// Fire on a dependency security advisory.
    DependencyAdvisory,
    /// Fire on a named monitoring alert (matched by name equality).
    MonitoringAlert(String),
    /// Fire ONLY on an explicit manual event (never on any other event kind).
    Manual,
}

impl Trigger {
    /// DETERMINISTIC match of this trigger against an incoming [`TriggerEvent`]:
    /// the kinds must agree, and for a parameterized trigger the payload must also
    /// match (a `Time` spec by string equality, a `FileChange` glob against the
    /// event's path, a `MonitoringAlert` by name equality). No model. A `Manual`
    /// trigger fires only on a `Manual` event.
    pub fn matches(&self, event: &TriggerEvent) -> bool {
        match (self, event) {
            (Trigger::Time(spec), TriggerEvent::Time(fired)) => spec == fired,
            (Trigger::GitPush, TriggerEvent::GitPush) => true,
            (Trigger::PrOpened, TriggerEvent::PrOpened) => true,
            (Trigger::IssueOpened, TriggerEvent::IssueOpened) => true,
            (Trigger::CiFailure, TriggerEvent::CiFailure) => true,
            (Trigger::FileChange(glob), TriggerEvent::FileChange(path)) => glob_matches(glob, path),
            (Trigger::DependencyAdvisory, TriggerEvent::DependencyAdvisory) => true,
            (Trigger::MonitoringAlert(name), TriggerEvent::MonitoringAlert(fired)) => name == fired,
            (Trigger::Manual, TriggerEvent::Manual) => true,
            _ => false,
        }
    }
}

/// An incoming EVENT evaluated against a job's triggers (bible sec 75). Each
/// variant carries the payload a deterministic match needs; it matches a
/// [`Trigger`] of the same kind (with glob / name / spec matching where the
/// trigger is parameterized). Model-free; the wake decision is
/// [`JobRecord::matches_event`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TriggerEvent {
    /// A schedule tick fired for this cron/at spec.
    Time(String),
    GitPush,
    PrOpened,
    IssueOpened,
    CiFailure,
    /// A file changed at this (repo-relative) path.
    FileChange(String),
    DependencyAdvisory,
    /// A named monitoring alert fired.
    MonitoringAlert(String),
    /// An explicit manual wake request.
    Manual,
}

/// Deterministic glob match of a (repo-relative) `path` against a `glob` pattern
/// (globset semantics: `**` spans separators, `*` does not). A malformed glob
/// never panics; it simply matches nothing.
pub(crate) fn glob_matches(glob: &str, path: &str) -> bool {
    globset::Glob::new(glob)
        .map(|g| g.compile_matcher().is_match(path))
        .unwrap_or(false)
}

/// The lifecycle status of a durable [`JobRecord`] (bible sec 73). Snake_case so
/// it round-trips in the KV store; `Pending` is the default so a record written
/// before this field existed still deserializes. `Done` / `Cancelled` / `Failed`
/// are TERMINAL and excluded from the recovered active set on restart (see
/// [`JobStore::recover`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    #[default]
    Pending,
    Running,
    Blocked,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    /// Whether this status is TERMINAL: the job is finished for good (`Done`,
    /// `Cancelled`, or `Failed`) and is NOT rebuilt into the active set on a
    /// restart.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Cancelled | Self::Failed)
    }

    /// Whether this status is ACTIVE (still has work): `Pending`, `Running`, or
    /// `Blocked`. The recovered set on restart is exactly the active jobs.
    pub fn is_active(&self) -> bool {
        !self.is_terminal()
    }
}

/// A durable BACKGROUND JOB (bible sec 73-75, sec 78.1 #17): a goal-bound unit of
/// work that SURVIVES A RESTART. It binds identity + provenance (session, and
/// optional repo / goal / plan / permissions refs), a resource [`Budget`], an
/// optional [`Schedule`], the [`Trigger`]s that should wake it, the pinned
/// [`CheckpointId`]s, a [`JobStatus`] lifecycle, timestamps, and the last error.
/// Stored in the KV `jobs` namespace keyed by `job_id`; its lifecycle transitions
/// are also appended to the session's durable event log (so the record is bound
/// to that log and auditable), and a fresh host's `jobs_recover()` rebuilds the
/// active set from the durable store.
///
/// The record, its trigger evaluation, and recovery are REAL + MODEL-FREE. The
/// ACTUAL agent execution of a woken job is DEFERRED_MODEL_REQUIRED: nothing here
/// runs a model or spawns an agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobRecord {
    pub job_id: String,
    pub session_id: SessionId,
    /// The repo (in the workspace graph) this job is scoped to, if any.
    pub repo_id: Option<String>,
    /// The durable goal (bible sec 14) this job advances, if any.
    pub goal_id: Option<String>,
    /// A ref (blob hash / path) to the job's plan, if one is pinned.
    pub plan_ref: Option<String>,
    /// A ref to the permission grant set the job runs under, if any.
    pub permissions_ref: Option<String>,
    #[serde(default)]
    pub budget: Budget,
    pub schedule: Option<Schedule>,
    #[serde(default)]
    pub triggers: Vec<Trigger>,
    #[serde(default)]
    pub checkpoints: Vec<CheckpointId>,
    #[serde(default)]
    pub status: JobStatus,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// The last error recorded on a `Failed`/`Blocked` transition, if any.
    pub last_error: Option<String>,
    /// When this job was PROMOTED from a live interactive run (Stage 4 background
    /// promotion), the id of that STILL-RUNNING run. The promoted job reuses the
    /// running run (no restart); control gestures (steer / pause / stop / fork)
    /// route to this run id. `None` for a job that never bound to a live run.
    /// `#[serde(default)]` so a record written before this field existed still
    /// deserializes to `None`.
    #[serde(default)]
    pub run_id: Option<String>,
}

impl JobRecord {
    /// A fresh PENDING job for a session with the given triggers + budget. A unique
    /// `job_id` is minted (blake3 over session + wall-clock micros); the optional
    /// refs / schedule / checkpoints are layered on via the builder methods.
    pub fn pending(session_id: SessionId, triggers: Vec<Trigger>, budget: Budget) -> Self {
        let now = now_ms();
        let job_id = JobStore::new_id(&session_id);
        Self {
            job_id,
            session_id,
            repo_id: None,
            goal_id: None,
            plan_ref: None,
            permissions_ref: None,
            budget,
            schedule: None,
            triggers,
            checkpoints: Vec::new(),
            status: JobStatus::Pending,
            created_ms: now,
            updated_ms: now,
            last_error: None,
            run_id: None,
        }
    }

    /// Bind this job to a live interactive run (Stage 4 background promotion):
    /// the promoted job reuses that still-running run rather than restarting it.
    pub fn with_run(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    pub fn with_repo(mut self, repo_id: impl Into<String>) -> Self {
        self.repo_id = Some(repo_id.into());
        self
    }

    pub fn with_goal(mut self, goal_id: impl Into<String>) -> Self {
        self.goal_id = Some(goal_id.into());
        self
    }

    pub fn with_plan_ref(mut self, plan_ref: impl Into<String>) -> Self {
        self.plan_ref = Some(plan_ref.into());
        self
    }

    pub fn with_permissions_ref(mut self, permissions_ref: impl Into<String>) -> Self {
        self.permissions_ref = Some(permissions_ref.into());
        self
    }

    pub fn with_schedule(mut self, schedule: Schedule) -> Self {
        self.schedule = Some(schedule);
        self
    }

    pub fn with_checkpoint(mut self, checkpoint_id: impl Into<CheckpointId>) -> Self {
        self.checkpoints.push(checkpoint_id.into());
        self
    }

    /// DETERMINISTIC wake predicate: does an incoming `event` match ANY trigger on
    /// this job? No model. The actual dispatch of the woken job is
    /// DEFERRED_MODEL_REQUIRED.
    pub fn matches_event(&self, event: &TriggerEvent) -> bool {
        self.triggers.iter().any(|trigger| trigger.matches(event))
    }
}

/// Durable persistence + recovery for [`JobRecord`]s over the KV store (bible sec
/// 73). A stateless facade over the `jobs` namespace keyed by `job_id`, mirroring
/// [`GoalStore`] / [`CheckpointStore`]. `recover` rebuilds the ACTIVE
/// (non-terminal) job set from the durable store, which is what survives a restart.
pub struct JobStore;

impl JobStore {
    pub const NAMESPACE: &'static str = "jobs";

    /// Mint a fresh, unique job id (blake3 over session + wall-clock micros).
    pub fn new_id(session: &SessionId) -> String {
        subbit_id("job", session, hide_core::ids::now_micros() as u128)
    }

    /// Durably write (or replace) a job, keyed by `job_id`.
    pub fn put(kv: &DynKeyValueStore, record: &JobRecord) -> Result<()> {
        let value = serde_json::to_value(record)?;
        kv.put(Self::NAMESPACE, &record.job_id, value)
    }

    /// Look up a job by id, if any.
    pub fn get(kv: &DynKeyValueStore, job_id: &str) -> Option<JobRecord> {
        kv.get(Self::NAMESPACE, job_id)
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    /// Every durable job, ordered deterministically (created_ms then job_id) so
    /// the list is stable across runs and reopens.
    pub fn list_all(kv: &DynKeyValueStore) -> Vec<JobRecord> {
        let mut out: Vec<JobRecord> = kv
            .list(Self::NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<JobRecord>(value).ok())
            .collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.job_id.cmp(&b.job_id))
        });
        out
    }

    /// Rebuild the ACTIVE job set (Pending / Running / Blocked) from the durable
    /// store: exactly what a fresh host should resume watching after a restart.
    /// Terminal jobs (Done / Cancelled / Failed) are excluded. Deterministic order
    /// (created_ms then job_id).
    pub fn recover(kv: &DynKeyValueStore) -> Vec<JobRecord> {
        Self::list_all(kv)
            .into_iter()
            .filter(|job| job.status.is_active())
            .collect()
    }
}
