use crate::error::Result;
use crate::event::{Event, EventLog};
use crate::ids::{BlobId, SessionId};
use crate::types::BlobRef;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub type DynEventLog = Arc<dyn EventLog>;
pub type DynEventLogIntegrity = Arc<dyn EventLogIntegrity>;
pub type DynBlobStore = Arc<dyn BlobStore>;
pub type DynProjectionStore = Arc<dyn ProjectionStore>;
pub type DynKeyValueStore = Arc<dyn KeyValueStore>;

pub trait BlobStore: Send + Sync {
    fn put(&self, bytes: Vec<u8>, media_type: Option<String>) -> Result<BlobRef>;
    fn get(&self, blob: &BlobRef) -> Result<Option<Vec<u8>>>;
}

pub trait ProjectionStore: Send + Sync {
    fn put_projection(&self, session_id: &SessionId, seq: u64, projection: Value) -> Result<()>;
    fn latest_projection(&self, session_id: &SessionId) -> Result<Option<(u64, Value)>>;
}

pub trait KeyValueStore: Send + Sync {
    fn put(&self, table: &str, key: &str, value: Value) -> Result<()>;
    fn get(&self, table: &str, key: &str) -> Result<Option<Value>>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StoreHealth {
    pub name: String,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct InMemoryBlobStore {
    blobs: Mutex<BTreeMap<String, Vec<u8>>>,
}

#[derive(Debug, Clone)]
pub struct FileBlobStore {
    root: PathBuf,
}

impl FileBlobStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn path_for_hash(&self, hash: &str) -> PathBuf {
        let prefix = hash.get(..2).unwrap_or("00");
        self.root.join(prefix).join(hash)
    }
}

impl BlobStore for InMemoryBlobStore {
    fn put(&self, bytes: Vec<u8>, media_type: Option<String>) -> Result<BlobRef> {
        let id = BlobId::new();
        let hash = format!("stub-{}-{}", id.as_str(), bytes.len());
        self.blobs.lock().insert(hash.clone(), bytes.clone());
        Ok(BlobRef {
            id,
            hash,
            size_bytes: bytes.len() as u64,
            media_type,
        })
    }

    fn get(&self, blob: &BlobRef) -> Result<Option<Vec<u8>>> {
        Ok(self.blobs.lock().get(&blob.hash).cloned())
    }
}

impl BlobStore for FileBlobStore {
    fn put(&self, bytes: Vec<u8>, media_type: Option<String>) -> Result<BlobRef> {
        let hash = sha256_hex(&bytes);
        let path = self.path_for_hash(&hash);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, &bytes)?;
            std::fs::rename(tmp, &path)?;
        }
        Ok(BlobRef {
            id: BlobId::new(),
            hash,
            size_bytes: bytes.len() as u64,
            media_type,
        })
    }

    fn get(&self, blob: &BlobRef) -> Result<Option<Vec<u8>>> {
        let path = self.path_for_hash(&blob.hash);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(std::fs::read(path)?))
    }
}

#[derive(Debug, Default)]
pub struct InMemoryProjectionStore {
    projections: Mutex<BTreeMap<SessionId, Vec<(u64, Value)>>>,
}

#[derive(Debug, Clone)]
pub struct FileProjectionStore {
    root: PathBuf,
}

impl FileProjectionStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn session_path(&self, session_id: &SessionId) -> PathBuf {
        self.root.join(format!("{}.jsonl", session_id.as_str()))
    }
}

impl ProjectionStore for InMemoryProjectionStore {
    fn put_projection(&self, session_id: &SessionId, seq: u64, projection: Value) -> Result<()> {
        self.projections
            .lock()
            .entry(session_id.clone())
            .or_default()
            .push((seq, projection));
        Ok(())
    }

    fn latest_projection(&self, session_id: &SessionId) -> Result<Option<(u64, Value)>> {
        Ok(self
            .projections
            .lock()
            .get(session_id)
            .and_then(|items| items.last().cloned()))
    }
}

impl ProjectionStore for FileProjectionStore {
    fn put_projection(&self, session_id: &SessionId, seq: u64, projection: Value) -> Result<()> {
        let path = self.session_path(session_id);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let record = ProjectionRecord { seq, projection };
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        serde_json::to_writer(&mut file, &record)?;
        use std::io::Write;
        file.write_all(b"\n")?;
        file.sync_data()?;
        Ok(())
    }

    fn latest_projection(&self, session_id: &SessionId) -> Result<Option<(u64, Value)>> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(None);
        }
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        use std::io::BufRead;
        let mut latest = None;
        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let record: ProjectionRecord = serde_json::from_str(&line)?;
            latest = Some((record.seq, record.projection));
        }
        Ok(latest)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ProjectionRecord {
    seq: u64,
    projection: Value,
}

#[derive(Debug, Default)]
pub struct InMemoryKeyValueStore {
    tables: Mutex<BTreeMap<String, BTreeMap<String, Value>>>,
}

impl KeyValueStore for InMemoryKeyValueStore {
    fn put(&self, table: &str, key: &str, value: Value) -> Result<()> {
        self.tables
            .lock()
            .entry(table.to_string())
            .or_default()
            .insert(key.to_string(), value);
        Ok(())
    }

    fn get(&self, table: &str, key: &str) -> Result<Option<Value>> {
        Ok(self
            .tables
            .lock()
            .get(table)
            .and_then(|items| items.get(key).cloned()))
    }
}

#[derive(Debug, Clone)]
pub struct FileKeyValueStore {
    root: PathBuf,
}

impl FileKeyValueStore {
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(&root)?;
        Ok(Self { root })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    fn value_path(&self, table: &str, key: &str) -> PathBuf {
        self.root
            .join(sanitize_component(table))
            .join(format!("{}.json", sanitize_component(key)))
    }
}

impl KeyValueStore for FileKeyValueStore {
    fn put(&self, table: &str, key: &str, value: Value) -> Result<()> {
        let path = self.value_path(table, key);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&value)?)?;
        std::fs::rename(tmp, path)?;
        Ok(())
    }

    fn get(&self, table: &str, key: &str) -> Result<Option<Value>> {
        let path = self.value_path(table, key);
        if !path.exists() {
            return Ok(None);
        }
        Ok(Some(serde_json::from_slice(&std::fs::read(path)?)?))
    }
}

pub trait EventLogIntegrity: Send + Sync {
    fn verify_chain(&self, events: &[Event]) -> Result<IntegrityReport>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub ok: bool,
    pub checked_events: usize,
    pub chain_root: Option<String>,
    pub detail: String,
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex_lower(&digest)
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn sanitize_component(input: &str) -> String {
    let mut out = String::new();
    for byte in input.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' => out.push(byte as char),
            other => out.push_str(&format!("_{other:02x}")),
        }
    }
    if out.is_empty() {
        "_".to_string()
    } else {
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_blob_store_roundtrips_content_addressed_bytes() {
        let dir = std::env::temp_dir().join(format!("hide_blob_{}", crate::ids::now_ms()));
        let store = FileBlobStore::open(&dir).unwrap();
        let blob = store
            .put(b"durable bytes".to_vec(), Some("text/plain".to_string()))
            .unwrap();
        let loaded = store.get(&blob).unwrap().unwrap();
        assert_eq!(loaded, b"durable bytes");
        let same = store.put(b"durable bytes".to_vec(), None).unwrap();
        assert_eq!(blob.hash, same.hash);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn file_projection_store_returns_latest_projection() {
        let dir = std::env::temp_dir().join(format!("hide_projection_{}", crate::ids::now_ms()));
        let store = FileProjectionStore::open(&dir).unwrap();
        let session = SessionId::new();
        store
            .put_projection(&session, 1, serde_json::json!({ "phase": "plan" }))
            .unwrap();
        store
            .put_projection(&session, 2, serde_json::json!({ "phase": "done" }))
            .unwrap();
        let latest = store.latest_projection(&session).unwrap().unwrap();
        assert_eq!(latest.0, 2);
        assert_eq!(latest.1["phase"], "done");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn file_key_value_store_roundtrips_json_values() {
        let dir = std::env::temp_dir().join(format!("hide_kv_{}", crate::ids::now_ms()));
        let store = FileKeyValueStore::open(&dir).unwrap();
        store
            .put(
                "sessions",
                "session/with/slashes",
                serde_json::json!({ "state": "running" }),
            )
            .unwrap();
        let loaded = store
            .get("sessions", "session/with/slashes")
            .unwrap()
            .unwrap();
        assert_eq!(loaded["state"], "running");
        assert!(store.get("sessions", "missing").unwrap().is_none());
        let _ = std::fs::remove_dir_all(dir);
    }
}
