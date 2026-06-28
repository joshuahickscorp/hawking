//! The task queue & job schema (bible ch.09 §4.5).
//!
//! The queue is **a projection of the event log** (P8): jobs are created and
//! mutated by appending `job.*` events; [`JobGraph::project_from`] rebuilds the
//! in-memory graph by folding those events. The `JobGraph` itself is a fast
//! in-memory index over that durable truth — never a second authoritative store.
//!
//! The schema is reconciled to Appendix A.1: `kind`, `parent_job`, `base_ref`,
//! `isolation`, `concurrency_class`, `attempts`/`max_attempts`, `result_ref`,
//! `schedule`, `schema_version`. The status set restores `Admitted`,
//! `Preempted`, and `Merging`. The `concurrency_class` Model-vs-CpuOnly split is
//! load-bearing for the scheduler's two-pool admission (§4.5.1).

use hide_core::event::{Event, EventClass, EventSource, NewEvent};
use hide_core::ids::{now_ms, RunId, SessionId};
use hide_core::persistence::DynEventLog;
use hide_core::types::BlobRef;
use hide_core::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

pub const JOB_SCHEMA_VERSION: u16 = 1;

/// What kind of work this job represents (A.1 `kind`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobKind {
    #[default]
    AgentRun,
    Batch,
    TestShard,
    Research,
    Merge,
    Custom,
}

/// Who created this job (A.1 `created_by`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CreatedBy {
    User,
    Agent,
    Schedule,
}

/// Isolation backend for the job's workspace (A.1 `isolation`, §4.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Isolation {
    #[default]
    Worktree,
    Overlay,
    Container,
    None,
}

/// The load-bearing two-pool split (§4.5.1). A `Model`-class job holds one of the
/// runtime's `max_batch_size` generation slots — the scarce resource. A `CpuOnly`
/// job (test shard, build, grep) does not compete for the model and runs far
/// wider, bounded only by cores/RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConcurrencyClass {
    #[default]
    Model,
    CpuOnly,
}

/// Priority classes (strict order, §4.6.3). `Interactive` always wins; it
/// preempts lower classes for a model slot. Fair-share applies *within* a class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorityClass {
    Interactive,
    High,
    Normal,
    Batch,
    Idle,
}

/// Job lifecycle status (A.1 `status`). Restores `Admitted`/`Preempted`/`Merging`
/// over the original scaffold's reduced set.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Admitted,
    Running,
    Paused,
    Preempted,
    Merging,
    Done,
    Failed,
    Cancelled,
}

impl JobStatus {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled
        )
    }

    /// Whether this status holds a live resource grant (counts against ceilings).
    pub fn is_live(self) -> bool {
        matches!(
            self,
            JobStatus::Admitted | JobStatus::Running | JobStatus::Merging
        )
    }
}

/// Resource admission inputs (A.1 `resource_hint`; advisory — the Governor
/// decides). The `concurrency_class` routes the job to the Model or CpuOnly pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub memory_mb: u64,
    pub needs_gpu: bool,
    pub est_wallclock_ms: u64,
    pub generation_slots: u32,
    pub ports: u16,
    pub concurrency_class: ConcurrencyClass,
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self {
            memory_mb: 1024,
            needs_gpu: true,
            est_wallclock_ms: 600_000,
            generation_slots: 1,
            ports: 0,
            concurrency_class: ConcurrencyClass::Model,
        }
    }
}

impl ResourceRequest {
    /// A CPU-only request (test shard / build) — no model slot, no GPU.
    pub fn cpu_only(memory_mb: u64) -> Self {
        Self {
            memory_mb,
            needs_gpu: false,
            est_wallclock_ms: 120_000,
            generation_slots: 0,
            ports: 0,
            concurrency_class: ConcurrencyClass::CpuOnly,
        }
    }
}

/// What ch.02 run to launch (the black box, A.1 `run_spec`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSpec {
    pub objective: String,
    /// §4.2 pattern hint (single|fanout|tournament|…) — see `crate::patterns`.
    pub pattern: Option<String>,
    pub profile: Option<String>,
    pub max_steps: u32,
    pub required_resources: ResourceRequest,
}

impl Default for JobSpec {
    fn default() -> Self {
        Self {
            objective: String::new(),
            pattern: None,
            profile: None,
            max_steps: 128,
            required_resources: ResourceRequest::default(),
        }
    }
}

/// Schedule gate for batch jobs (A.1 `schedule`, §4.7.2).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobSchedule {
    /// Earliest start (wall-clock ms since epoch).
    pub earliest_start_ms: Option<u64>,
    /// Gate conditions that must ALL hold before the batch fires.
    pub gates: Vec<ScheduleGate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduleGate {
    Idle,
    AcPower,
    ThermalOk,
    Cron,
}

/// The queue unit (A.1 `Job`). Reconciled to the binding contract.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentJob {
    pub id: String,
    pub kind: JobKind,
    pub title: String,
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub spec: JobSpec,
    pub status: JobStatus,
    pub priority: PriorityClass,
    pub created_by: CreatedBy,
    /// Fan-out children / batch members point at their parent (the job DAG).
    pub parent_job: Option<String>,
    /// Job-level DAG edges (this runs after these complete).
    pub dependencies: Vec<String>,
    pub isolation: Isolation,
    /// The commit each worktree forks from.
    pub base_ref: String,
    pub schedule: Option<JobSchedule>,
    pub attempts: u32,
    pub max_attempts: u32,
    /// Blob ref to the run's outcome summary (set on completion).
    pub result_ref: Option<BlobRef>,
    pub created_at_ms: u64,
    pub admitted_at_ms: Option<u64>,
    pub finished_at_ms: Option<u64>,
    pub schema_version: u16,
}

impl AgentJob {
    pub fn new(objective: impl Into<String>, priority: PriorityClass) -> Self {
        let objective = objective.into();
        Self {
            id: new_job_id(),
            kind: JobKind::AgentRun,
            title: truncate_title(&objective),
            session_id: SessionId::new(),
            run_id: None,
            spec: JobSpec {
                objective,
                ..JobSpec::default()
            },
            status: JobStatus::Queued,
            priority,
            created_by: CreatedBy::User,
            parent_job: None,
            dependencies: Vec::new(),
            isolation: Isolation::Worktree,
            base_ref: "HEAD".to_string(),
            schedule: None,
            attempts: 0,
            max_attempts: 1,
            result_ref: None,
            created_at_ms: now_ms(),
            admitted_at_ms: None,
            finished_at_ms: None,
            schema_version: JOB_SCHEMA_VERSION,
        }
    }

    pub fn with_kind(mut self, kind: JobKind) -> Self {
        self.kind = kind;
        self
    }

    pub fn with_session(mut self, session: SessionId) -> Self {
        self.session_id = session;
        self
    }

    pub fn with_parent(mut self, parent: impl Into<String>) -> Self {
        self.parent_job = Some(parent.into());
        self
    }

    pub fn depends_on(mut self, deps: impl IntoIterator<Item = String>) -> Self {
        self.dependencies = deps.into_iter().collect();
        self
    }

    pub fn with_resources(mut self, resources: ResourceRequest) -> Self {
        self.spec.required_resources = resources;
        self
    }

    pub fn with_concurrency_class(mut self, class: ConcurrencyClass) -> Self {
        self.spec.required_resources.concurrency_class = class;
        self
    }

    pub fn with_base_ref(mut self, base_ref: impl Into<String>) -> Self {
        self.base_ref = base_ref.into();
        self
    }

    pub fn concurrency_class(&self) -> ConcurrencyClass {
        self.spec.required_resources.concurrency_class
    }
}

fn new_job_id() -> String {
    format!("job_{}", ulid::Ulid::new())
}

fn truncate_title(objective: &str) -> String {
    let first_line = objective.lines().next().unwrap_or(objective);
    if first_line.chars().count() <= 80 {
        first_line.to_string()
    } else {
        let mut t: String = first_line.chars().take(77).collect();
        t.push_str("...");
        t
    }
}

/// The in-memory job graph: a fast index that is also a projection of the event
/// log. Mutations go through `enqueue_logged`/`set_status_logged`, which append
/// the durable `job.*` event AND update the index; `project_from` rebuilds the
/// whole index from the log on startup (crash recovery, P8).
#[derive(Debug, Default)]
pub struct JobGraph {
    jobs: RwLock<BTreeMap<String, AgentJob>>,
}

impl JobGraph {
    pub fn new() -> Self {
        Self::default()
    }

    // --- pure in-memory index ops (used by the projection fold + tests) ---

    pub fn enqueue(&self, job: AgentJob) {
        self.jobs.write().insert(job.id.clone(), job);
    }

    pub fn get(&self, id: &str) -> Option<AgentJob> {
        self.jobs.read().get(id).cloned()
    }

    pub fn all(&self) -> Vec<AgentJob> {
        self.jobs.read().values().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.jobs.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.read().is_empty()
    }

    pub fn set_status(&self, id: &str, status: JobStatus) {
        if let Some(job) = self.jobs.write().get_mut(id) {
            apply_status_side_effects(job, status);
        }
    }

    pub fn set_run_id(&self, id: &str, run_id: RunId) {
        if let Some(job) = self.jobs.write().get_mut(id) {
            job.run_id = Some(run_id);
        }
    }

    pub fn set_result_ref(&self, id: &str, result_ref: BlobRef) {
        if let Some(job) = self.jobs.write().get_mut(id) {
            job.result_ref = Some(result_ref);
        }
    }

    /// Count live jobs (admitted/running/merging) in a concurrency class — the
    /// scheduler's two-pool occupancy read (§4.5.1).
    pub fn live_in_class(&self, class: ConcurrencyClass) -> u32 {
        self.jobs
            .read()
            .values()
            .filter(|j| j.status.is_live() && j.concurrency_class() == class)
            .count() as u32
    }

    /// Ready jobs: queued/preempted with all deps `Done`, ordered by priority
    /// then least-recently-created (fair-share-within-class proxy). Kahn-style
    /// ready-set (§4.5.2).
    pub fn ready_jobs(&self) -> Vec<AgentJob> {
        let jobs = self.jobs.read();
        let completed: BTreeSet<_> = jobs
            .values()
            .filter(|job| job.status == JobStatus::Done)
            .map(|job| job.id.clone())
            .collect();
        let mut ready: Vec<_> = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Queued | JobStatus::Preempted))
            .filter(|job| job.dependencies.iter().all(|dep| completed.contains(dep)))
            .cloned()
            .collect();
        ready.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| a.created_at_ms.cmp(&b.created_at_ms))
                .then_with(|| a.id.cmp(&b.id))
        });
        ready
    }

    /// The lowest-priority live job at or below `floor` priority — the preemption
    /// victim selector (§4.6.2 step 2). Returns the *weakest* candidate.
    pub fn lowest_priority_live_below(&self, floor: PriorityClass) -> Option<AgentJob> {
        self.jobs
            .read()
            .values()
            .filter(|j| j.status.is_live() && j.priority > floor)
            .filter(|j| j.concurrency_class() == ConcurrencyClass::Model)
            .max_by(|a, b| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| b.created_at_ms.cmp(&a.created_at_ms))
            })
            .cloned()
    }

    pub fn has_waiting(&self, priority: PriorityClass) -> bool {
        self.jobs
            .read()
            .values()
            .any(|j| j.priority == priority && matches!(j.status, JobStatus::Queued))
    }

    /// Cycle detection over the dependency DAG (rejects cyclic graphs at enqueue,
    /// §4.5.2 / F1). Real DFS with a recursion stack.
    pub fn has_cycle(&self) -> bool {
        let jobs = self.jobs.read();
        let mut color: BTreeMap<&str, u8> = BTreeMap::new(); // 0=white 1=gray 2=black
        for id in jobs.keys() {
            if dfs_cycle(id.as_str(), &jobs, &mut color) {
                return true;
            }
        }
        false
    }

    // --- durable, event-logged ops (the queue-as-projection path, P8) ---

    /// Enqueue a job AND append `job.enqueued` carrying the FULL job record (so
    /// `project_from` can rehydrate it verbatim). Rejects a cyclic graph (§4.5.2).
    pub async fn enqueue_logged(&self, log: &DynEventLog, job: AgentJob) -> Result<()> {
        {
            let mut guard = self.jobs.write();
            guard.insert(job.id.clone(), job.clone());
        }
        if self.has_cycle() {
            self.jobs.write().remove(&job.id);
            return Err(hide_core::HideError::InvalidState(format!(
                "job {} would create a dependency cycle",
                job.id
            )));
        }
        log.append(job_event(
            &job,
            "job.enqueued",
            EventClass::Neither,
            json!({ "job_id": job.id, "job": job }),
        ))
        .await?;
        Ok(())
    }

    /// Transition status AND append the matching `job.*` event (A.5). The index
    /// and the log move together, so a replay reconstructs the same state.
    pub async fn set_status_logged(
        &self,
        log: &DynEventLog,
        id: &str,
        status: JobStatus,
        extra: Value,
    ) -> Result<()> {
        let job = {
            let mut guard = self.jobs.write();
            match guard.get_mut(id) {
                Some(job) => {
                    apply_status_side_effects(job, status);
                    job.clone()
                }
                None => {
                    return Err(hide_core::HideError::NotFound(format!("job {id}")));
                }
            }
        };
        let (kind, class) = status_event_kind(status);
        let mut payload = json!({ "job_id": id, "status": status });
        merge_payload(&mut payload, extra);
        log.append(job_event(&job, kind, class, payload)).await?;
        Ok(())
    }

    /// Rebuild the entire graph from the event log (crash recovery / replay, P8).
    /// Folds `job.enqueued` → insert, `job.*` status events → transition. Folding
    /// records data; it never re-launches a run (T3).
    pub fn project_from(&self, events: &[Event]) {
        let mut guard = self.jobs.write();
        guard.clear();
        for event in events {
            match event.kind.as_str() {
                "job.enqueued" => {
                    if let Some(job) = event.payload_as::<EnqueuedSnapshot>() {
                        if let Some(full) = job.job {
                            guard.insert(full.id.clone(), full);
                        }
                    }
                }
                kind if kind.starts_with("job.") => {
                    if let Some(p) = event.payload_as::<StatusPatch>() {
                        if let (Some(job_id), Some(status)) = (p.job_id, p.status) {
                            if let Some(job) = guard.get_mut(&job_id) {
                                apply_status_side_effects(job, status);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Apply the bookkeeping side effects of a status transition (timestamps, attempt
/// counter) consistently whether the transition came from the index path or the
/// projection fold.
fn apply_status_side_effects(job: &mut AgentJob, status: JobStatus) {
    match status {
        JobStatus::Admitted if job.admitted_at_ms.is_none() => {
            job.admitted_at_ms = Some(now_ms());
        }
        JobStatus::Running => {
            // First entry into Running counts as an attempt.
            if job.status != JobStatus::Running {
                job.attempts += 1;
            }
        }
        s if s.is_terminal() && job.finished_at_ms.is_none() => {
            job.finished_at_ms = Some(now_ms());
        }
        _ => {}
    }
    job.status = status;
}

/// Map a status to the A.5 event kind + class.
fn status_event_kind(status: JobStatus) -> (&'static str, EventClass) {
    match status {
        JobStatus::Admitted => ("job.admitted", EventClass::Neither),
        JobStatus::Running => ("job.started", EventClass::Neither),
        JobStatus::Preempted => ("job.preempted", EventClass::Action),
        JobStatus::Merging => ("job.merging", EventClass::Neither),
        JobStatus::Done | JobStatus::Failed | JobStatus::Cancelled => {
            ("job.completed", EventClass::Observation)
        }
        JobStatus::Paused => ("job.paused", EventClass::Neither),
        JobStatus::Queued => ("job.requeued", EventClass::Neither),
    }
}

fn job_event(job: &AgentJob, kind: &str, class: EventClass, payload: Value) -> NewEvent {
    let mut ev = NewEvent::of(job.session_id.clone(), EventSource::System, kind, payload)
        .with_class(class);
    if let Some(run) = &job.run_id {
        ev = ev.with_run(run.clone());
    }
    ev
}

fn merge_payload(base: &mut Value, extra: Value) {
    if let (Some(base_obj), Value::Object(extra_obj)) = (base.as_object_mut(), extra) {
        for (k, v) in extra_obj {
            base_obj.insert(k, v);
        }
    }
}

/// The full job is embedded in the `job.enqueued` event so the projection can
/// rehydrate it verbatim.
#[derive(Debug, Serialize, Deserialize)]
struct EnqueuedSnapshot {
    #[serde(default)]
    job: Option<AgentJob>,
}

#[derive(Debug, Deserialize)]
struct StatusPatch {
    job_id: Option<String>,
    status: Option<JobStatus>,
}

fn dfs_cycle<'a>(
    id: &'a str,
    jobs: &'a BTreeMap<String, AgentJob>,
    color: &mut BTreeMap<&'a str, u8>,
) -> bool {
    match color.get(id) {
        Some(2) => return false, // fully explored
        Some(1) => return true,  // back-edge → cycle
        _ => {}
    }
    color.insert(id, 1);
    if let Some(job) = jobs.get(id) {
        for dep in &job.dependencies {
            // Resolve the dep against the keyset so dangling deps don't recurse.
            if let Some((dep_key, _)) = jobs.get_key_value(dep) {
                if dfs_cycle(dep_key.as_str(), jobs, color) {
                    return true;
                }
            }
        }
    }
    color.insert(id, 2);
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::InMemoryEventLog;
    use std::sync::Arc;

    #[test]
    fn ready_jobs_respect_dependencies_and_priority() {
        let g = JobGraph::new();
        let a = AgentJob::new("a", PriorityClass::Normal);
        let a_id = a.id.clone();
        let b = AgentJob::new("b", PriorityClass::Interactive).depends_on([a_id.clone()]);
        g.enqueue(a);
        g.enqueue(b.clone());
        // b depends on a (not done) → only a is ready.
        let ready = g.ready_jobs();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, a_id);
        // Complete a → b becomes ready.
        g.set_status(&a_id, JobStatus::Done);
        let ready = g.ready_jobs();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, b.id);
    }

    #[test]
    fn cycle_detection_rejects_back_edge() {
        let g = JobGraph::new();
        let mut a = AgentJob::new("a", PriorityClass::Normal);
        let mut b = AgentJob::new("b", PriorityClass::Normal);
        a.dependencies = vec![b.id.clone()];
        b.dependencies = vec![a.id.clone()];
        g.enqueue(a);
        g.enqueue(b);
        assert!(g.has_cycle());
    }

    #[test]
    fn two_pool_occupancy_counts_per_class() {
        let g = JobGraph::new();
        let mut m = AgentJob::new("model", PriorityClass::Normal);
        m.status = JobStatus::Running;
        let mut c = AgentJob::new("cpu", PriorityClass::Normal)
            .with_concurrency_class(ConcurrencyClass::CpuOnly);
        c.status = JobStatus::Running;
        g.enqueue(m);
        g.enqueue(c);
        assert_eq!(g.live_in_class(ConcurrencyClass::Model), 1);
        assert_eq!(g.live_in_class(ConcurrencyClass::CpuOnly), 1);
    }

    #[tokio::test]
    async fn queue_is_a_projection_of_the_event_log() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let g = JobGraph::new();
        let job = AgentJob::new("durable goal", PriorityClass::Normal);
        let id = job.id.clone();
        g.enqueue_logged(&log, job).await.unwrap();
        g.set_status_logged(&log, &id, JobStatus::Admitted, json!({}))
            .await
            .unwrap();
        g.set_status_logged(&log, &id, JobStatus::Running, json!({}))
            .await
            .unwrap();
        g.set_status_logged(&log, &id, JobStatus::Done, json!({ "result": "ok" }))
            .await
            .unwrap();

        // Rebuild a fresh graph purely from the log → identical terminal state.
        let events = log.scan(None, None, None).await.unwrap();
        let rebuilt = JobGraph::new();
        rebuilt.project_from(&events);
        let job = rebuilt.get(&id).expect("job rehydrated from log");
        assert_eq!(job.status, JobStatus::Done);
        assert!(job.admitted_at_ms.is_some());
        assert!(job.finished_at_ms.is_some());
        assert_eq!(job.attempts, 1);
    }

    #[tokio::test]
    async fn enqueue_logged_rejects_cycle_and_emits_event() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let g = JobGraph::new();
        let a = AgentJob::new("a", PriorityClass::Normal);
        let a_id = a.id.clone();
        g.enqueue_logged(&log, a).await.unwrap();
        // A self-cycle via dependency on itself is rejected.
        let mut bad = AgentJob::new("bad", PriorityClass::Normal);
        bad.dependencies = vec![bad.id.clone()];
        assert!(g.enqueue_logged(&log, bad).await.is_err());
        // The valid job's enqueue event is in the log.
        let events = log.scan(None, None, None).await.unwrap();
        assert!(events
            .iter()
            .any(|e| e.kind == "job.enqueued" && e.payload["job_id"] == a_id.as_str()));
    }
}
