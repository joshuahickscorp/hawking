//! Ingestion queue: priority, retries, visible failure. Never silent drop.

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use ulid::Ulid;

use crate::objects::error::{ObjectError, Result};
use crate::objects::hash::ContentHash;
use crate::objects::permissions::ObjectPermissions;
use crate::objects::retention::RetentionPolicy;
use crate::objects::schema::{ObjectSource, StageName};

/// Higher numbers run first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Priority(pub u8);

impl Priority {
    pub const LOW: Priority = Priority(10);
    pub const NORMAL: Priority = Priority(50);
    pub const HIGH: Priority = Priority(80);
    pub const CRITICAL: Priority = Priority(100);
}

/// Status of a queued ingestion job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    /// Waiting to retry a failed stage.
    RetryWait,
    /// All stages complete.
    Succeeded,
    /// Exhausted retries — failed **visibly**, still on the dead-letter log.
    FailedVisible,
}

/// One ingestion job.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IngestJob {
    pub id: String,
    pub priority: Priority,
    /// Sequence for FIFO within the same priority (lower = older = first).
    pub seq: u64,
    pub status: JobStatus,
    pub mime: String,
    pub source: ObjectSource,
    pub permissions: ObjectPermissions,
    pub retention: RetentionPolicy,
    pub label: Option<String>,
    pub created_by: String,
    /// Filled once Receive/hash completes.
    pub content_hash: Option<ContentHash>,
    /// Next stage to run (or retry).
    pub next_stage: StageName,
    pub attempts: u32,
    pub max_attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_error: Option<String>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl IngestJob {
    pub fn new(
        priority: Priority,
        seq: u64,
        mime: String,
        source: ObjectSource,
        permissions: ObjectPermissions,
        retention: RetentionPolicy,
        label: Option<String>,
        created_by: String,
        now_ms: u64,
    ) -> Self {
        Self {
            id: format!("ijob_{}", Ulid::new()),
            priority,
            seq,
            status: JobStatus::Queued,
            mime,
            source,
            permissions,
            retention,
            label,
            created_by,
            content_hash: None,
            next_stage: StageName::Receive,
            attempts: 0,
            max_attempts: 3,
            last_error: None,
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        }
    }
}

/// Heap entry: higher priority first; for equal priority, lower seq first.
#[derive(Debug, Clone, Eq, PartialEq)]
struct HeapEntry {
    priority: Priority,
    seq: u64,
    job_id: String,
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.seq.cmp(&self.seq),
            o => o,
        }
    }
}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Priority queue + dead-letter for visible failures.
#[derive(Debug, Default)]
pub struct IngestQueue {
    heap: BinaryHeap<HeapEntry>,
    jobs: std::collections::BTreeMap<String, IngestJob>,
    /// Jobs that exhausted retries — never silently dropped.
    dead_letter: Vec<IngestJob>,
    next_seq: u64,
}

impl IngestQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue(&mut self, mut job: IngestJob) -> String {
        if job.seq == 0 {
            job.seq = self.next_seq;
            self.next_seq += 1;
        } else {
            self.next_seq = self.next_seq.max(job.seq + 1);
        }
        let id = job.id.clone();
        self.heap.push(HeapEntry {
            priority: job.priority,
            seq: job.seq,
            job_id: id.clone(),
        });
        self.jobs.insert(id.clone(), job);
        id
    }

    pub fn len(&self) -> usize {
        self.jobs
            .values()
            .filter(|j| matches!(j.status, JobStatus::Queued | JobStatus::RetryWait))
            .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, id: &str) -> Option<&IngestJob> {
        self.jobs.get(id)
    }

    pub fn dead_letter(&self) -> &[IngestJob] {
        &self.dead_letter
    }

    /// Pop the highest-priority ready job. Returns error only if queue empty
    /// of runnable jobs — never drops a failed job.
    pub fn pop_ready(&mut self) -> Result<IngestJob> {
        while let Some(entry) = self.heap.pop() {
            if let Some(job) = self.jobs.get(&entry.job_id) {
                if matches!(job.status, JobStatus::Queued | JobStatus::RetryWait) {
                    let mut job = self.jobs.remove(&entry.job_id).unwrap();
                    job.status = JobStatus::Running;
                    return Ok(job);
                }
            }
        }
        Err(ObjectError::QueueEmpty)
    }

    /// Re-queue after a partial stage (resume) or for the next stage.
    pub fn requeue(&mut self, mut job: IngestJob, now_ms: u64) {
        job.status = JobStatus::Queued;
        job.updated_at_ms = now_ms;
        let id = job.id.clone();
        self.heap.push(HeapEntry {
            priority: job.priority,
            seq: job.seq,
            job_id: id.clone(),
        });
        self.jobs.insert(id, job);
    }

    /// Record a retryable stage failure. If attempts exhausted, move to dead
    /// letter with [`JobStatus::FailedVisible`] — never silent drop.
    pub fn fail_stage(
        &mut self,
        mut job: IngestJob,
        stage: StageName,
        detail: impl Into<String>,
        now_ms: u64,
    ) -> JobStatus {
        let detail = detail.into();
        job.attempts += 1;
        job.last_error = Some(format!("{}: {detail}", stage.as_str()));
        job.updated_at_ms = now_ms;
        job.next_stage = stage;

        if job.attempts >= job.max_attempts {
            job.status = JobStatus::FailedVisible;
            let status = job.status;
            self.dead_letter.push(job);
            status
        } else {
            job.status = JobStatus::RetryWait;
            let status = job.status;
            let id = job.id.clone();
            self.heap.push(HeapEntry {
                priority: job.priority,
                seq: job.seq,
                job_id: id.clone(),
            });
            self.jobs.insert(id, job);
            status
        }
    }

    pub fn complete(&mut self, mut job: IngestJob, now_ms: u64) {
        job.status = JobStatus::Succeeded;
        job.updated_at_ms = now_ms;
        // Keep terminal jobs in the map for inspection.
        self.jobs.insert(job.id.clone(), job);
    }

    pub fn put_running(&mut self, job: IngestJob) {
        self.jobs.insert(job.id.clone(), job);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objects::permissions::{ObjectPermissions, Surface};
    fn job(pri: Priority, seq: u64) -> IngestJob {
        IngestJob::new(
            pri,
            seq,
            "text/plain".into(),
            ObjectSource::Synthetic { label: "t".into() },
            ObjectPermissions::owner_only("u", vec![Surface::You]),
            RetentionPolicy::durable(),
            None,
            "u".into(),
            0,
        )
    }
    #[test]
    fn higher_priority_first() {
        let mut q = IngestQueue::new();
        let low = job(Priority::LOW, 1);
        let high = job(Priority::HIGH, 2);
        let low_id = low.id.clone();
        let high_id = high.id.clone();
        q.enqueue(low);
        q.enqueue(high);
        let first = q.pop_ready().unwrap();
        assert_eq!(first.id, high_id);
        let second = q.pop_ready().unwrap();
        assert_eq!(second.id, low_id);
    }
    #[test]
    fn exhausted_retries_go_to_dead_letter_not_dropped() {
        let mut q = IngestQueue::new();
        let mut j = job(Priority::NORMAL, 1);
        j.max_attempts = 2;
        let id = j.id.clone();
        q.enqueue(j);
        let j = q.pop_ready().unwrap();
        let st = q.fail_stage(j, StageName::Persist, "disk full", 1);
        assert_eq!(st, JobStatus::RetryWait);
        let j = q.pop_ready().unwrap();
        assert_eq!(j.id, id);
        let st = q.fail_stage(j, StageName::Persist, "disk full again", 2);
        assert_eq!(st, JobStatus::FailedVisible);
        assert_eq!(q.dead_letter().len(), 1);
        assert_eq!(q.dead_letter()[0].id, id);
        assert!(q.is_empty());
    }
}
