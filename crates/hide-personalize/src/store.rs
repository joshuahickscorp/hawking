use crate::records::{PersonalizationRecord, TaskClass};
use hide_core::{HideError, Result};
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type DynPersonalizationStore = Arc<dyn PersonalizationStore>;

pub trait PersonalizationStore: Send + Sync {
    fn append(&self, record: &PersonalizationRecord) -> Result<()>;
    fn load_all(&self) -> Result<Vec<PersonalizationRecord>>;

    fn load_recent(&self, limit: usize) -> Result<Vec<PersonalizationRecord>> {
        let mut records = self.load_all()?;
        if records.len() > limit {
            records.drain(..records.len() - limit);
        }
        Ok(records)
    }

    fn load_by_task(
        &self,
        task_class: TaskClass,
        limit: usize,
    ) -> Result<Vec<PersonalizationRecord>> {
        let mut records: Vec<_> = self
            .load_all()?
            .into_iter()
            .filter(|record| record.task_type == task_class)
            .collect();
        if records.len() > limit {
            records.drain(..records.len() - limit);
        }
        Ok(records)
    }
}

#[derive(Debug, Default)]
pub struct InMemoryPersonalizationStore {
    records: Mutex<Vec<PersonalizationRecord>>,
}

impl InMemoryPersonalizationStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PersonalizationStore for InMemoryPersonalizationStore {
    fn append(&self, record: &PersonalizationRecord) -> Result<()> {
        self.records.lock().push(record.clone());
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<PersonalizationRecord>> {
        Ok(self.records.lock().clone())
    }
}

#[derive(Debug, Clone)]
pub struct JsonlPersonalizationStore {
    path: PathBuf,
}

impl JsonlPersonalizationStore {
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

impl PersonalizationStore for JsonlPersonalizationStore {
    fn append(&self, record: &PersonalizationRecord) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut file, record)?;
        file.write_all(b"\n")?;
        file.sync_data()?;
        Ok(())
    }

    fn load_all(&self) -> Result<Vec<PersonalizationRecord>> {
        read_records(&self.path)
    }
}

fn read_records(path: &Path) -> Result<Vec<PersonalizationRecord>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut records = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str(&line).map_err(|err| {
            HideError::Storage(format!(
                "failed to parse personalization store {} line {}: {err}",
                path.display(),
                idx + 1
            ))
        })?;
        records.push(record);
    }
    Ok(records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jsonl_personalization_store_roundtrips_records() {
        let dir =
            std::env::temp_dir().join(format!("hide_personalize_{}", hide_core::ids::now_ms()));
        let path = dir.join("records.jsonl");
        let store = JsonlPersonalizationStore::open(&path).unwrap();
        let first = PersonalizationRecord::accepted(TaskClass::EditCode, "prompt-a", "diff-a");
        let second = PersonalizationRecord::accepted(TaskClass::WriteTest, "prompt-b", "diff-b");

        store.append(&first).unwrap();
        store.append(&second).unwrap();

        let reopened = JsonlPersonalizationStore::open(&path).unwrap();
        let loaded = reopened.load_all().unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].prompt_hash, "prompt-a");
        let recent = reopened.load_recent(1).unwrap();
        assert_eq!(recent[0].prompt_hash, "prompt-b");
        let edit_records = reopened.load_by_task(TaskClass::EditCode, 10).unwrap();
        assert_eq!(edit_records.len(), 1);
        let _ = std::fs::remove_dir_all(dir);
    }
}
