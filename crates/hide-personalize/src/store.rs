//! Record persistence + the scrub-on-write + the curated `dataset/vNNN` layout
//! (bible §11.1.1 / §11.1.2).
//!
//! Two real things beyond plain JSONL append:
//!   * **Scrub-on-write** — every record's `diff_proposed` / `diff_accepted` is
//!     run through the real [`hide_security::Redactor`] (the same detector suite
//!     ch.10 applies to the event log) *before* it is persisted, so a secret in
//!     a proposed diff never reaches disk (§11.1.1, the privacy invariant).
//!   * **`PersonalLayout`** — the `~/.hawking/personal/{records,dataset/vNNN,
//!     adapters,eval}` directory map (§11.1.2), with a real "next dataset
//!     version" allocator so curate can write `dataset/v001`, `v002`, ….

use crate::records::{PersonalizationRecord, TaskClass};
use hide_core::{HideError, Result};
use hide_security::Redactor;
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

/// Scrub a record's diffs in place using the supplied redactor. Returns the
/// total number of redactions applied across both diffs (so the caller can emit
/// a `security.redaction` event if it wants to, §4.8).
pub fn scrub_record(redactor: &Redactor, record: &mut PersonalizationRecord) -> usize {
    let mut total = 0;
    let proposed = redactor.redact(&record.diff_proposed);
    total += proposed
        .redactions
        .iter()
        .map(|r| r.occurrences)
        .sum::<usize>();
    record.diff_proposed = proposed.text;
    let accepted = redactor.redact(&record.diff_accepted);
    total += accepted
        .redactions
        .iter()
        .map(|r| r.occurrences)
        .sum::<usize>();
    record.diff_accepted = accepted.text;
    total
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

/// JSONL store that **scrubs secrets on every write** (§11.1.1).
pub struct JsonlPersonalizationStore {
    path: PathBuf,
    redactor: Redactor,
}

impl JsonlPersonalizationStore {
    /// Open (creating the file + parents) with the default redaction suite
    /// (pattern detectors + entropy catch-all).
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        Self::open_with_redactor(path, Redactor::default())
    }

    pub fn open_with_redactor(path: impl Into<PathBuf>, redactor: Redactor) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            File::create(&path)?;
        }
        Ok(Self { path, redactor })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl PersonalizationStore for JsonlPersonalizationStore {
    fn append(&self, record: &PersonalizationRecord) -> Result<()> {
        // Scrub-on-write: the secret never reaches disk.
        let mut scrubbed = record.clone();
        scrub_record(&self.redactor, &mut scrubbed);

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        serde_json::to_writer(&mut file, &scrubbed)?;
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

/// The `~/.hawking/personal/` directory map (§11.1.2).
///
/// ```text
/// <root>/
///   records/          # raw scrubbed JSONL, append-only, user-deletable
///   dataset/          # curated, versioned SFT records (v001, v002, …)
///   adapters/         # trained LoRA checkpoints (written by Hawking Condense)
///   eval/             # held-out accept-rate measurement
/// ```
#[derive(Debug, Clone)]
pub struct PersonalLayout {
    root: PathBuf,
}

impl PersonalLayout {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn records_dir(&self) -> PathBuf {
        self.root.join("records")
    }

    pub fn dataset_dir(&self) -> PathBuf {
        self.root.join("dataset")
    }

    pub fn adapters_dir(&self) -> PathBuf {
        self.root.join("adapters")
    }

    pub fn eval_dir(&self) -> PathBuf {
        self.root.join("eval")
    }

    /// Create every directory in the map (idempotent).
    pub fn ensure(&self) -> Result<()> {
        for dir in [
            self.records_dir(),
            self.dataset_dir(),
            self.adapters_dir(),
            self.eval_dir(),
        ] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// The directory for a specific dataset version, e.g. `dataset/v001`.
    pub fn dataset_version_dir(&self, version: u32) -> PathBuf {
        self.dataset_dir().join(format!("v{version:03}"))
    }

    /// Scan `dataset/` for `vNNN` directories and return the next free version
    /// (1 if none exist). Lets curate write a fresh `vNNN` without clobbering.
    pub fn next_dataset_version(&self) -> Result<u32> {
        let dir = self.dataset_dir();
        if !dir.exists() {
            return Ok(1);
        }
        let mut max = 0u32;
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix('v') {
                if let Ok(n) = rest.parse::<u32>() {
                    max = max.max(n);
                }
            }
        }
        Ok(max + 1)
    }

    /// Today's raw records file: `records/<date>.jsonl`. (Date is a coarse
    /// partition key; the bible's `<date>/<ulid>.jsonl` is a finer variant —
    /// either is replay-equivalent since records carry their own ids.)
    pub fn records_file_for_today(&self) -> PathBuf {
        // Avoid a chrono dep: derive a YYYYMMDD-ish key from the unix day.
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let day = secs / 86_400;
        self.records_dir().join(format!("day{day}.jsonl"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::records::TaskClass;

    #[test]
    fn jsonl_personalization_store_roundtrips_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let store = JsonlPersonalizationStore::open(&path).unwrap();
        let first = PersonalizationRecord::accepted(TaskClass::EditCode, "prompt-a", "diff-a");
        let second = PersonalizationRecord::accepted(TaskClass::WriteTest, "prompt-b", "diff-b");

        store.append(&first).unwrap();
        store.append(&second).unwrap();

        let reopened = JsonlPersonalizationStore::open(&path).unwrap();
        let loaded = reopened.load_all().unwrap();
        assert_eq!(loaded.len(), 2);
        let recent = reopened.load_recent(1).unwrap();
        assert_eq!(recent[0].diff_accepted, "diff-b");
        let edit_records = reopened.load_by_task(TaskClass::EditCode, 10).unwrap();
        assert_eq!(edit_records.len(), 1);
    }

    #[test]
    fn scrub_on_write_removes_secret_from_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let store = JsonlPersonalizationStore::open(&path).unwrap();
        // A GitHub PAT embedded in a proposed diff.
        let secret = "ghp_0123456789abcdefABCDEF0123456789abcdef";
        let rec = PersonalizationRecord::accepted(
            TaskClass::EditCode,
            "p",
            format!("+const TOKEN = \"{secret}\";"),
        );
        store.append(&rec).unwrap();

        let raw = std::fs::read_to_string(&path).unwrap();
        assert!(!raw.contains(secret), "secret must not reach disk: {raw}");
        assert!(raw.contains("redacted"), "redaction marker expected");

        // The in-memory record we passed in is untouched (scrub is on the copy).
        assert!(rec.diff_proposed.contains(secret));
    }

    #[test]
    fn dataset_version_allocator() {
        let dir = tempfile::tempdir().unwrap();
        let layout = PersonalLayout::new(dir.path());
        layout.ensure().unwrap();
        assert_eq!(layout.next_dataset_version().unwrap(), 1);
        std::fs::create_dir_all(layout.dataset_version_dir(1)).unwrap();
        std::fs::create_dir_all(layout.dataset_version_dir(2)).unwrap();
        assert_eq!(layout.next_dataset_version().unwrap(), 3);
        assert!(layout.dataset_version_dir(7).ends_with("v007"));
    }
}
