use crate::pipeline::{ResearchRun, ResearchState};
use hide_core::{HideError, Result};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type DynResearchLedger = Arc<dyn ResearchLedger>;

pub trait ResearchLedger: Send + Sync {
    fn append_run(&self, run: &ResearchRun) -> Result<()>;
    fn load_runs(&self) -> Result<Vec<ResearchRun>>;

    fn latest(&self) -> Result<Option<ResearchRun>> {
        Ok(self.load_runs()?.into_iter().last())
    }

    fn load_by_state(&self, state: ResearchState) -> Result<Vec<ResearchRun>> {
        Ok(self
            .load_runs()?
            .into_iter()
            .filter(|run| run.state == state)
            .collect())
    }
}

#[derive(Debug, Default)]
pub struct InMemoryResearchLedger {
    runs: Mutex<Vec<ResearchRun>>,
}

impl InMemoryResearchLedger {
    pub fn new() -> Self {
        Self::default()
    }
}

impl ResearchLedger for InMemoryResearchLedger {
    fn append_run(&self, run: &ResearchRun) -> Result<()> {
        self.runs.lock().push(run.clone());
        Ok(())
    }

    fn load_runs(&self) -> Result<Vec<ResearchRun>> {
        Ok(self.runs.lock().clone())
    }
}

#[derive(Debug, Clone)]
pub struct JsonlResearchLedger {
    path: PathBuf,
}

impl JsonlResearchLedger {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            File::create(&path)?;
        }
        Ok(Self { path })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl ResearchLedger for JsonlResearchLedger {
    fn append_run(&self, run: &ResearchRun) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut file, run)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        Ok(())
    }

    fn load_runs(&self) -> Result<Vec<ResearchRun>> {
        read_runs(&self.path)
    }
}

fn read_runs(path: &Path) -> Result<Vec<ResearchRun>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut runs = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let run = serde_json::from_str(&line).map_err(|err| {
            HideError::Storage(format!(
                "failed to parse research ledger {} line {}: {err}",
                path.display(),
                idx + 1
            ))
        })?;
        runs.push(run);
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_research_ledger_roundtrips_runs() {
        let dir =
            std::env::temp_dir().join(format!("hawking_research_{}", hide_core::ids::now_ms()));
        let path = dir.join("runs.jsonl");
        let ledger = JsonlResearchLedger::open(&path).unwrap();
        let mut run = ResearchRun::new("paged attention");
        run.state = ResearchState::Complete;
        run.docs_read = 2;

        ledger.append_run(&run).unwrap();

        let reopened = JsonlResearchLedger::open(&path).unwrap();
        let loaded = reopened.load_runs().unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].topic, "paged attention");
        assert_eq!(reopened.latest().unwrap().unwrap().docs_read, 2);
        assert_eq!(
            reopened
                .load_by_state(ResearchState::Complete)
                .unwrap()
                .len(),
            1
        );
        let _ = std::fs::remove_dir_all(dir);
    }
}
