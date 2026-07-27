//! Per-event checkpoint ledger (bible ch.08 §4.6).
//!
//! The run ledger in [`crate::run_ledger`] records *run summaries*; this records
//! the *per-event journal* that makes an overnight run resumable: each state
//! transition and each fetched/read doc appends a line, so a crash at 3 a.m.
//! resumes from the last completed state without re-fetching (CAS dedup) or
//! re-extracting (content-addressed nodes).

use hide_core::error::Result;
use hide_core::ids::RunId;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// One journal event for a run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointEvent {
    pub run_id: String,
    pub seq: u64,
    pub at_ms: u64,
    pub kind: CheckpointKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum CheckpointKind {
    /// Run opened with this topic/seed.
    Opened { topic: String, seed: u64 },
    /// Entered a pipeline state (its `{:?}` name).
    State { state: String },
    /// A source doc was fetched + content-addressed (recorded so we never
    /// re-fetch the same content_hash on resume).
    Fetched {
        doc_id: String,
        content_hash: Option<String>,
    },
    /// A round of the reflect loop completed.
    Round {
        round: u32,
        coverage: f32,
        novelty: f32,
    },
    /// Run finalized.
    Done { docs_read: usize, claims: usize },
}

/// Append-only, line-per-event journal for one or many runs.
pub trait CheckpointLedger: Send + Sync {
    fn append(&self, event: CheckpointEvent) -> Result<()>;
    fn events_for(&self, run_id: &str) -> Result<Vec<CheckpointEvent>>;

    /// The set of content hashes already fetched for a run — used to skip
    /// re-fetching on resume.
    fn fetched_hashes(&self, run_id: &str) -> Result<std::collections::HashSet<String>> {
        let mut out = std::collections::HashSet::new();
        for e in self.events_for(run_id)? {
            if let CheckpointKind::Fetched {
                content_hash: Some(h),
                ..
            } = e.kind
            {
                out.insert(h);
            }
        }
        Ok(out)
    }

    /// The last state name recorded for a run, if any (the resume point).
    fn last_state(&self, run_id: &str) -> Result<Option<String>> {
        let mut last = None;
        for e in self.events_for(run_id)? {
            if let CheckpointKind::State { state } = e.kind {
                last = Some(state);
            }
        }
        Ok(last)
    }
}

pub type DynCheckpointLedger = Arc<dyn CheckpointLedger>;

#[derive(Default)]
pub struct InMemoryCheckpointLedger {
    events: Mutex<Vec<CheckpointEvent>>,
}

impl InMemoryCheckpointLedger {
    pub fn new() -> Self {
        Self::default()
    }
}

impl CheckpointLedger for InMemoryCheckpointLedger {
    fn append(&self, event: CheckpointEvent) -> Result<()> {
        self.events.lock().push(event);
        Ok(())
    }

    fn events_for(&self, run_id: &str) -> Result<Vec<CheckpointEvent>> {
        Ok(self
            .events
            .lock()
            .iter()
            .filter(|e| e.run_id == run_id)
            .cloned()
            .collect())
    }
}

/// A JSONL checkpoint journal on disk (one file holds all runs; filter by id).
#[derive(Debug, Clone)]
pub struct JsonlCheckpointLedger {
    path: PathBuf,
    seq: Arc<Mutex<u64>>,
}

impl JsonlCheckpointLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            File::create(&path)?;
        }
        Ok(Self {
            path,
            seq: Arc::new(Mutex::new(0)),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl CheckpointLedger for JsonlCheckpointLedger {
    fn append(&self, event: CheckpointEvent) -> Result<()> {
        *self.seq.lock() = event.seq;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut file, &event)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        Ok(())
    }

    fn events_for(&self, run_id: &str) -> Result<Vec<CheckpointEvent>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut out = Vec::new();
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let e: CheckpointEvent = serde_json::from_str(&line)?;
            if e.run_id == run_id {
                out.push(e);
            }
        }
        Ok(out)
    }
}

/// A small monotonic sequencer for a run's checkpoint events.
pub struct RunJournal {
    run_id: RunId,
    ledger: DynCheckpointLedger,
    seq: u64,
}

impl RunJournal {
    pub fn new(run_id: RunId, ledger: DynCheckpointLedger) -> Self {
        // Resume the sequence number from any existing events.
        let seq = ledger
            .events_for(run_id.as_str())
            .ok()
            .and_then(|e| e.last().map(|l| l.seq))
            .unwrap_or(0);
        Self {
            run_id,
            ledger,
            seq,
        }
    }

    pub fn record(&mut self, kind: CheckpointKind) -> Result<()> {
        self.seq += 1;
        self.ledger.append(CheckpointEvent {
            run_id: self.run_id.0.clone(),
            seq: self.seq,
            at_ms: hide_core::ids::now_ms(),
            kind,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_checkpoints_filter_and_resume() {
        let dir = std::env::temp_dir().join(format!("hawking_ckpt_{}", hide_core::ids::now_ms()));
        let path = dir.join("ckpt.jsonl");
        let ledger: DynCheckpointLedger = Arc::new(JsonlCheckpointLedger::open(&path).unwrap());
        let run = RunId::from("run_a");
        let mut j = RunJournal::new(run.clone(), ledger.clone());
        j.record(CheckpointKind::Opened {
            topic: "kv cache".into(),
            seed: 42,
        })
        .unwrap();
        j.record(CheckpointKind::State {
            state: "Fetch".into(),
        })
        .unwrap();
        j.record(CheckpointKind::Fetched {
            doc_id: "doc:1".into(),
            content_hash: Some("h1".into()),
        })
        .unwrap();

        // A different run's event must not bleed in.
        let mut other = RunJournal::new(RunId::from("run_b"), ledger.clone());
        other
            .record(CheckpointKind::State {
                state: "Read".into(),
            })
            .unwrap();

        assert_eq!(ledger.events_for("run_a").unwrap().len(), 3);
        assert_eq!(
            ledger.last_state("run_a").unwrap().as_deref(),
            Some("Fetch")
        );
        assert!(ledger.fetched_hashes("run_a").unwrap().contains("h1"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
