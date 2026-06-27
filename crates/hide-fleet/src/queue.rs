use hide_core::ids::{now_ms, RunId, SessionId};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AgentJob {
    pub id: String,
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub spec: JobSpec,
    pub status: JobStatus,
    pub priority: PriorityClass,
    pub dependencies: Vec<String>,
    pub created_at_ms: u64,
}

impl AgentJob {
    pub fn new(objective: impl Into<String>, priority: PriorityClass) -> Self {
        Self {
            id: format!("job_{}", now_ms()),
            session_id: SessionId::new(),
            run_id: None,
            spec: JobSpec {
                objective: objective.into(),
                pattern: None,
                max_steps: 128,
                required_resources: ResourceRequest::default(),
            },
            status: JobStatus::Queued,
            priority,
            dependencies: Vec::new(),
            created_at_ms: now_ms(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSpec {
    pub objective: String,
    pub pattern: Option<String>,
    pub max_steps: u32,
    pub required_resources: ResourceRequest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub memory_mb: u64,
    pub generation_slots: u32,
    pub ports: u16,
}

impl Default for ResourceRequest {
    fn default() -> Self {
        Self {
            memory_mb: 1024,
            generation_slots: 1,
            ports: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriorityClass {
    Interactive,
    UserInitiated,
    Batch,
    Background,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Ready,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Default)]
pub struct JobGraph {
    jobs: RwLock<BTreeMap<String, AgentJob>>,
}

impl JobGraph {
    pub fn enqueue(&self, job: AgentJob) {
        self.jobs.write().insert(job.id.clone(), job);
    }

    pub fn get(&self, id: &str) -> Option<AgentJob> {
        self.jobs.read().get(id).cloned()
    }

    pub fn set_status(&self, id: &str, status: JobStatus) {
        if let Some(job) = self.jobs.write().get_mut(id) {
            job.status = status;
        }
    }

    pub fn ready_jobs(&self) -> Vec<AgentJob> {
        let jobs = self.jobs.read();
        let completed: BTreeSet<_> = jobs
            .values()
            .filter(|job| job.status == JobStatus::Completed)
            .map(|job| job.id.clone())
            .collect();
        let mut ready: Vec<_> = jobs
            .values()
            .filter(|job| matches!(job.status, JobStatus::Queued | JobStatus::Ready))
            .filter(|job| job.dependencies.iter().all(|dep| completed.contains(dep)))
            .cloned()
            .collect();
        ready.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .then_with(|| a.created_at_ms.cmp(&b.created_at_ms))
        });
        ready
    }

    pub fn has_cycle(&self) -> bool {
        let jobs = self.jobs.read();
        for id in jobs.keys() {
            let mut visiting = BTreeSet::new();
            if visit(id, &jobs, &mut visiting) {
                return true;
            }
        }
        false
    }
}

fn visit(id: &str, jobs: &BTreeMap<String, AgentJob>, visiting: &mut BTreeSet<String>) -> bool {
    if !visiting.insert(id.to_string()) {
        return true;
    }
    if let Some(job) = jobs.get(id) {
        for dep in &job.dependencies {
            if visit(dep, jobs, visiting) {
                return true;
            }
        }
    }
    visiting.remove(id);
    false
}
