//! Personalization capture store — record + durable log only.
//! Training loops are out of Behaviour Constitution / S3 slice scope.

use hide_core::ids::{now_micros, RunId, SessionId};
use serde::{Deserialize, Serialize};

/// blake3-256 digest. Serialized as a lowercase hex string so the JSONL log
/// stays human-inspectable (§11.1.5 "the user can find and inspect every
/// record") while the in-memory form is the fixed 32-byte array the bible
/// mandates for `prompt_hash` / `context_fingerprint`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Hash32(pub [u8; 32]);

impl Hash32 {
    /// blake3 of arbitrary bytes — the canonical way to mint a `prompt_hash`
    /// (over the final system prompt + user message) or a `context_fingerprint`
    /// (over the set of file paths + their sizes).
    pub fn of(bytes: impl AsRef<[u8]>) -> Self {
        Self(*blake3::hash(bytes.as_ref()).as_bytes())
    }

    /// Fold an iterator of `(path, size)` pairs into a single fingerprint in a
    /// stable, order-independent way (the set of files, not their order).
    pub fn fingerprint_files<'a, I>(files: I) -> Self
    where
        I: IntoIterator<Item = (&'a str, u64)>,
    {
        let mut entries: Vec<(String, u64)> =
            files.into_iter().map(|(p, s)| (p.to_string(), s)).collect();
        entries.sort();
        let mut hasher = blake3::Hasher::new();
        for (path, size) in entries {
            hasher.update(path.as_bytes());
            hasher.update(&size.to_le_bytes());
        }
        Self(*hasher.finalize().as_bytes())
    }

    pub fn to_hex(self) -> String {
        let mut s = String::with_capacity(64);
        for byte in self.0 {
            s.push_str(&format!("{byte:02x}"));
        }
        s
    }

    pub fn from_hex(hex: &str) -> Option<Self> {
        if hex.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).ok()?;
        }
        Some(Self(out))
    }
}

impl Serialize for Hash32 {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> std::result::Result<S::Ok, S::Error> {
        ser.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for Hash32 {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(de)?;
        Hash32::from_hex(&s).ok_or_else(|| serde::de::Error::custom("invalid blake3 hex digest"))
    }
}

/// One captured agent turn (§11.1.1). Written to the personal records log only
/// **after** the outcome is known. Never leaves the device.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PersonalizationRecord {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    /// Microsecond-precision wall-clock of the observation (was `_ms`).
    pub observed_at_us: u64,
    pub task_type: TaskClass,
    /// blake3 of the final system prompt + user message.
    pub prompt_hash: Hash32,
    /// blake3 of the set of file paths + their sizes at call time.
    pub context_fingerprint: Hash32,
    pub outcome: Outcome,
    pub diff_proposed: String,
    pub diff_accepted: String,
    pub latency_ms: u32,
    /// Decode throughput of the generation that produced `diff_proposed`.
    /// Non-optional (bible §11.1.1).
    pub tok_s: f32,
    pub reject_reason: Option<String>,
    pub model_role: String,
    pub active_adapters: Vec<String>,
}

impl PersonalizationRecord {
    /// The shared skeleton; the four public ctors fill `outcome` + the diffs.
    fn base(task_type: TaskClass, prompt: &str, outcome: Outcome) -> Self {
        Self {
            session_id: SessionId::new(),
            run_id: None,
            observed_at_us: now_micros(),
            task_type,
            prompt_hash: Hash32::of(prompt),
            context_fingerprint: Hash32::of(""),
            outcome,
            diff_proposed: String::new(),
            diff_accepted: String::new(),
            latency_ms: 0,
            tok_s: 0.0,
            reject_reason: None,
            model_role: "hero".to_string(),
            active_adapters: Vec::new(),
        }
    }

    /// User accepted the diff verbatim. `diff_accepted == diff_proposed`.
    pub fn accepted(task_type: TaskClass, prompt: &str, diff: impl Into<String>) -> Self {
        let diff = diff.into();
        let mut rec = Self::base(task_type, prompt, Outcome::Accepted);
        rec.diff_proposed = diff.clone();
        rec.diff_accepted = diff;
        rec
    }

    /// User accepted a manually-edited version. `edit_distance_chars` measures
    /// how much they rewrote (drives curate's rewrite-ratio gate, §11.1.2 rule 2).
    pub fn modified(
        task_type: TaskClass,
        prompt: &str,
        proposed: impl Into<String>,
        accepted: impl Into<String>,
        edit_distance_chars: u32,
    ) -> Self {
        let mut rec = Self::base(
            task_type,
            prompt,
            Outcome::Modified {
                edit_distance_chars,
            },
        );
        rec.diff_proposed = proposed.into();
        rec.diff_accepted = accepted.into();
        rec
    }

    /// User rejected the suggestion. `diff_accepted` stays empty; the proposed
    /// diff is retained as the negative half of a future DPO pair (§11.1.2 rule 3).
    pub fn rejected(
        task_type: TaskClass,
        prompt: &str,
        proposed: impl Into<String>,
        reason: Option<String>,
    ) -> Self {
        let mut rec = Self::base(task_type, prompt, Outcome::Rejected);
        rec.diff_proposed = proposed.into();
        rec.reject_reason = reason;
        rec
    }

    /// Session ended before an explicit accept/reject — an implicit partial
    /// signal (§11.1.1). Curation drops these.
    pub fn abandoned(task_type: TaskClass, prompt: &str, proposed: impl Into<String>) -> Self {
        let mut rec = Self::base(task_type, prompt, Outcome::Abandoned);
        rec.diff_proposed = proposed.into();
        rec
    }

    /// Builder helper for the capture layer to attach the real context manifest.
    pub fn with_context_fingerprint(mut self, fp: Hash32) -> Self {
        self.context_fingerprint = fp;
        self
    }

    /// Builder helper to record decode throughput + latency for curation rule 4.
    pub fn with_perf(mut self, tok_s: f32, latency_ms: u32) -> Self {
        self.tok_s = tok_s;
        self.latency_ms = latency_ms;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskClass {
    EditCode,
    WriteTest,
    Refactor,
    ExplainCode,
    CommitMsg,
    Diagnose,
    Research,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// User accepted the diff without modification.
    Accepted,
    /// User accepted a manually-edited version of the diff.
    Modified { edit_distance_chars: u32 },
    /// User rejected (undo, explicit dismiss, or replaced entirely).
    Rejected,
    /// Session ended before explicit accept/reject.
    Abandoned,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hash32_hex_roundtrip() {
        let h = Hash32::of("hello");
        let hex = h.to_hex();
        assert_eq!(hex.len(), 64);
        assert_eq!(Hash32::from_hex(&hex), Some(h));
        assert_eq!(Hash32::from_hex("nothex"), None);
    }
    #[test]
    fn fingerprint_is_order_independent() {
        let a = Hash32::fingerprint_files([("a.rs", 10), ("b.rs", 20)]);
        let b = Hash32::fingerprint_files([("b.rs", 20), ("a.rs", 10)]);
        assert_eq!(a, b);
        let c = Hash32::fingerprint_files([("a.rs", 11), ("b.rs", 20)]);
        assert_ne!(a, c);
    }
    #[test]
    fn all_four_outcome_ctors_exist() {
        let p = "system+user";
        assert_eq!(
            PersonalizationRecord::accepted(TaskClass::EditCode, p, "d").outcome,
            Outcome::Accepted
        );
        assert_eq!(
            PersonalizationRecord::modified(TaskClass::Refactor, p, "a", "b", 3).outcome,
            Outcome::Modified {
                edit_distance_chars: 3
            }
        );
        assert_eq!(
            PersonalizationRecord::rejected(TaskClass::WriteTest, p, "d", Some("nope".into()))
                .outcome,
            Outcome::Rejected
        );
        assert_eq!(
            PersonalizationRecord::abandoned(TaskClass::Diagnose, p, "d").outcome,
            Outcome::Abandoned
        );
        let r1 = PersonalizationRecord::accepted(TaskClass::EditCode, p, "x");
        let r2 = PersonalizationRecord::rejected(TaskClass::EditCode, p, "y", None);
        assert_eq!(r1.prompt_hash, r2.prompt_hash);
    }
    #[test]
    fn record_serde_roundtrips_with_hex_hashes() {
        let rec =
            PersonalizationRecord::accepted(TaskClass::EditCode, "p", "diff").with_perf(42.0, 100);
        let json = serde_json::to_string(&rec).unwrap();
        assert!(json.contains(&rec.prompt_hash.to_hex()));
        let back: PersonalizationRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(back, rec);
    }
}

// PersonalizationRecord and TaskClass are defined above in this module.
use hide_core::{HideError, Result};
use hide_kernel::security::Redactor;
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
mod store_tests {
    use super::TaskClass;
    use super::*;
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
