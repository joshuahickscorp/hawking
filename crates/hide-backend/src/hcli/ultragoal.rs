//! Durable HAIDER/HCLI mission state.
//!
//! Disk state is authoritative. A conversation is not.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

pub const MISSION_SCHEMA: &str = "haider.mission.v1";

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MissionState {
    pub schema: String,
    pub project_root: String,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub ultragoal: Option<Ultragoal>,
    pub goal: Option<String>,
    pub steers: Vec<Steer>,
    pub obligations: Vec<Obligation>,
    pub frontier: Vec<String>,
    pub receipts: Vec<ReceiptRef>,
    pub workers: Vec<WorkerRef>,
    pub next_action: Option<String>,
    pub model: Option<String>,
    pub endpoint: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ultragoal {
    pub id: String,
    pub title: String,
    pub source_text: String,
    pub source_hash: String,
    pub clauses: Vec<String>,
    pub invariants: Vec<String>,
    pub acceptance: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Obligation {
    pub id: String,
    pub text: String,
    pub status: ObligationStatus,
    pub depends_on: Vec<String>,
    pub receipt: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObligationStatus {
    Pending,
    InProgress,
    Done,
    Blocked,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Steer {
    pub id: String,
    pub text: String,
    pub created_ms: u64,
    pub applied: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptRef {
    pub id: String,
    pub summary: String,
    pub created_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerRef {
    pub id: String,
    pub command: Vec<String>,
    pub status: String,
    pub output_path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IngestSummary {
    pub kind: String,
    pub tokens: usize,
    pub clauses: usize,
    pub obligations: usize,
}

impl MissionState {
    pub fn load(root: &Path) -> Self {
        let path = root.join(".haider").join("mission.json");
        let mut state = match fs::read_to_string(&path) {
            Ok(s) => serde_json::from_str(&s).unwrap_or_default(),
            Err(_) => Self::default(),
        };
        if state.schema.is_empty() {
            state.schema = MISSION_SCHEMA.to_string();
        }
        if state.project_root.is_empty() {
            state.project_root = root.display().to_string();
        }
        if state.created_ms == 0 {
            state.created_ms = now_ms();
        }
        state.updated_ms = now_ms();
        state
    }

    pub fn save(&self, root: &Path) -> Result<(), Box<dyn std::error::Error>> {
        let dir = root.join(".haider");
        fs::create_dir_all(&dir)?;
        let path = dir.join("mission.json");
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn ingest_text(&mut self, text: &str) -> IngestSummary {
        let now = now_ms();
        let title = text
            .lines()
            .find(|l| !l.trim().is_empty())
            .map(|l| l.trim().to_string())
            .unwrap_or_else(|| "ultragoal".into());
        let clauses: Vec<String> = text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        let invariants: Vec<String> = clauses
            .iter()
            .filter(|l| {
                let n = l.to_lowercase();
                n.contains("must") || n.contains("invariant") || n.contains("never")
            })
            .cloned()
            .collect();
        let acceptance: Vec<String> = clauses
            .iter()
            .filter(|l| {
                let n = l.to_lowercase();
                n.contains("pass") || n.contains("acceptance")
            })
            .cloned()
            .collect();
        let obligations = clauses
            .iter()
            .enumerate()
            .take(200)
            .map(|(i, c)| Obligation {
                id: format!("ob_{now}_{i}"),
                text: c.clone(),
                status: ObligationStatus::Pending,
                depends_on: Vec::new(),
                receipt: None,
            })
            .collect();

        self.ultragoal = Some(Ultragoal {
            id: format!("ug_{now}"),
            title: title.clone(),
            source_text: text.to_string(),
            source_hash: fnv_hash(text),
            clauses: clauses.clone(),
            invariants,
            acceptance,
        });
        self.goal = Some(title);
        self.obligations = obligations;
        self.updated_ms = now;
        self.refresh_frontier();

        IngestSummary {
            kind: "ultragoal".into(),
            tokens: estimate_tokens(text),
            clauses: self.ultragoal.as_ref().map(|u| u.clauses.len()).unwrap_or(0),
            obligations: self.obligations.len(),
        }
    }

    pub fn set_goal(&mut self, goal: &str) {
        self.goal = Some(goal.to_string());
        self.updated_ms = now_ms();
    }

    pub fn add_steer(&mut self, text: &str) -> Steer {
        let steer = Steer {
            id: format!("st_{}", now_ms()),
            text: text.to_string(),
            created_ms: now_ms(),
            applied: false,
        };
        self.steers.push(steer.clone());
        self.updated_ms = now_ms();
        steer
    }

    pub fn add_receipt(&mut self, id: &str, summary: &str) {
        self.receipts.push(ReceiptRef {
            id: id.to_string(),
            summary: summary.to_string(),
            created_ms: now_ms(),
        });
        self.updated_ms = now_ms();
    }

    pub fn add_worker(&mut self, worker: WorkerRef) {
        self.workers.push(worker);
        self.updated_ms = now_ms();
    }

    pub fn update_worker_status(&mut self, id: &str, status: &str) {
        if let Some(w) = self.workers.iter_mut().find(|w| w.id == id) {
            w.status = status.to_string();
            self.updated_ms = now_ms();
        }
    }

    pub fn mark_obligation_done(&mut self, id: &str, receipt: Option<String>) {
        if let Some(o) = self.obligations.iter_mut().find(|o| o.id == id) {
            o.status = ObligationStatus::Done;
            o.receipt = receipt;
            self.updated_ms = now_ms();
            self.refresh_frontier();
        }
    }

    pub fn frontier(&self) -> Vec<&Obligation> {
        self.obligations
            .iter()
            .filter(|o| matches!(o.status, ObligationStatus::Pending | ObligationStatus::InProgress))
            .collect()
    }

    pub fn next_action(&self) -> Option<&str> {
        self.obligations
            .iter()
            .find(|o| o.status == ObligationStatus::Pending)
            .map(|o| o.text.as_str())
    }

    fn refresh_frontier(&mut self) {
        self.frontier = self
            .obligations
            .iter()
            .filter(|o| matches!(o.status, ObligationStatus::Pending | ObligationStatus::InProgress))
            .map(|o| o.id.clone())
            .collect();
        self.next_action = self.next_action().map(|s| s.to_string());
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn estimate_tokens(text: &str) -> usize {
    text.split_whitespace().count()
}

fn fnv_hash(s: &str) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}
