use crate::error::{HideError, Result};
use crate::ids::{now_ms, EventId, GrantId, RunId, SessionId, StepId, TimestampMs, ToolCallId};
use crate::types::{BlobRef, Decision, EffectSet, Provenance};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const EVENT_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventKind(pub String);

impl EventKind {
    pub fn new(kind: impl Into<String>) -> Self {
        Self(kind.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for EventKind {
    fn from(value: &str) -> Self {
        Self::new(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventSource {
    User,
    Agent,
    Tool,
    Runtime,
    System,
    Plugin(String),
    Index,
    Memory,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub schema_version: u16,
    pub id: EventId,
    pub seq: u64,
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub parent: Option<EventId>,
    pub source: EventSource,
    pub kind: EventKind,
    pub payload: EventPayload,
    pub timestamp_ms: TimestampMs,
    pub redactions: Vec<String>,
    pub chain_hash: Option<String>,
}

impl Event {
    pub fn new(seq: u64, input: NewEvent) -> Self {
        Self {
            schema_version: EVENT_SCHEMA_VERSION,
            id: EventId::new(),
            seq,
            session_id: input.session_id,
            run_id: input.run_id,
            parent: input.parent,
            source: input.source,
            kind: input.kind,
            payload: input.payload,
            timestamp_ms: now_ms(),
            redactions: input.redactions,
            chain_hash: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewEvent {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub parent: Option<EventId>,
    pub source: EventSource,
    pub kind: EventKind,
    pub payload: EventPayload,
    pub redactions: Vec<String>,
}

impl NewEvent {
    pub fn system(
        session_id: SessionId,
        kind: impl Into<EventKind>,
        payload: EventPayload,
    ) -> Self {
        Self {
            session_id,
            run_id: None,
            parent: None,
            source: EventSource::System,
            kind: kind.into(),
            payload,
            redactions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum EventPayload {
    UserIntent(UserIntentEvent),
    AgentState(AgentStateEvent),
    Plan(PlanEvent),
    ToolCall(ToolCallEvent),
    ToolResult(ToolResultEvent),
    RuntimeStatus(RuntimeStatusEvent),
    RuntimeToken(TokenEvent),
    Security(SecurityEvent),
    Projection(ProjectionEvent),
    Memory(MemoryEvent),
    Index(IndexEvent),
    Error(ErrorEvent),
    Custom(Value),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserIntentEvent {
    pub intent: String,
    pub args: Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentStateEvent {
    pub phase: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanEvent {
    pub action: String,
    pub step_id: Option<StepId>,
    pub plan: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallEvent {
    pub call_id: ToolCallId,
    pub tool_name: String,
    pub capability_grant_id: Option<GrantId>,
    pub args: Value,
    pub predicted_effects: EffectSet,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultEvent {
    pub call_id: ToolCallId,
    pub ok: bool,
    pub summary: String,
    pub output: Option<Value>,
    pub bytes_ref: Option<BlobRef>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeStatusEvent {
    pub provider: String,
    pub status: String,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenEvent {
    pub stream_id: String,
    pub token_id: Option<u32>,
    pub text: String,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub gate: String,
    pub decision: Decision,
    pub grant_id: Option<GrantId>,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectionEvent {
    pub projection: String,
    pub patch: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub action: String,
    pub record_id: String,
    pub provenance: Provenance,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexEvent {
    pub action: String,
    pub generation: u64,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ErrorEvent {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
}

pub trait EventLog: Send + Sync {
    fn append<'a>(&'a self, event: NewEvent) -> BoxFuture<'a, Result<Event>>;

    fn scan<'a>(
        &'a self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<Event>>>;
}

#[derive(Debug, Default)]
pub struct InMemoryEventLog {
    next_seq: AtomicU64,
    events: Mutex<Vec<Event>>,
}

#[derive(Debug)]
pub struct JsonlEventLog {
    path: PathBuf,
    state: Mutex<JsonlEventLogState>,
}

#[derive(Debug, Clone)]
struct JsonlEventLogState {
    next_seq: u64,
    previous_hash: Vec<u8>,
}

impl JsonlEventLog {
    pub fn open(path: impl Into<PathBuf>) -> Result<Self> {
        let path = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if !path.exists() {
            File::create(&path)?;
        }
        let events = read_events(&path)?;
        let next_seq = events.last().map_or(1, |event| event.seq + 1);
        let previous_hash = events
            .last()
            .and_then(|event| event.chain_hash.as_deref())
            .map(hex_decode)
            .transpose()?
            .unwrap_or_else(|| vec![0u8; 32]);
        Ok(Self {
            path,
            state: Mutex::new(JsonlEventLogState {
                next_seq,
                previous_hash,
            }),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl InMemoryEventLog {
    pub fn new() -> Self {
        Self {
            next_seq: AtomicU64::new(1),
            events: Mutex::new(Vec::new()),
        }
    }

    pub fn len(&self) -> usize {
        self.events.lock().len()
    }
}

impl EventLog for InMemoryEventLog {
    fn append<'a>(&'a self, event: NewEvent) -> BoxFuture<'a, Result<Event>> {
        Box::pin(async move {
            let seq = self.next_seq.fetch_add(1, Ordering::SeqCst);
            let event = Event::new(seq, event);
            self.events.lock().push(event.clone());
            Ok(event)
        })
    }

    fn scan<'a>(
        &'a self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<Event>>> {
        Box::pin(async move {
            let mut out = Vec::new();
            for event in self.events.lock().iter() {
                if session_id
                    .as_ref()
                    .map_or(false, |sid| sid != &event.session_id)
                {
                    continue;
                }
                if after_seq.map_or(false, |seq| event.seq <= seq) {
                    continue;
                }
                out.push(event.clone());
                if limit.map_or(false, |n| out.len() >= n) {
                    break;
                }
            }
            Ok(out)
        })
    }
}

impl EventLog for JsonlEventLog {
    fn append<'a>(&'a self, event: NewEvent) -> BoxFuture<'a, Result<Event>> {
        Box::pin(async move {
            let mut state = self.state.lock();
            let mut event = Event::new(state.next_seq, event);
            let chain_hash = compute_chain_hash(&state.previous_hash, &event)?;
            event.chain_hash = Some(hex_lower(&chain_hash));

            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.path)?;
            serde_json::to_writer(&mut file, &event)?;
            file.write_all(b"\n")?;
            file.sync_data()?;

            state.next_seq += 1;
            state.previous_hash = chain_hash;
            Ok(event)
        })
    }

    fn scan<'a>(
        &'a self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<Event>>> {
        Box::pin(async move {
            let mut out = Vec::new();
            for event in read_events(&self.path)? {
                if session_id
                    .as_ref()
                    .map_or(false, |sid| sid != &event.session_id)
                {
                    continue;
                }
                if after_seq.map_or(false, |seq| event.seq <= seq) {
                    continue;
                }
                out.push(event);
                if limit.map_or(false, |n| out.len() >= n) {
                    break;
                }
            }
            Ok(out)
        })
    }
}

fn read_events(path: &Path) -> Result<Vec<Event>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut events = Vec::new();
    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let event: Event = serde_json::from_str(&line).map_err(|err| {
            HideError::Storage(format!(
                "failed to parse event log {} line {}: {err}",
                path.display(),
                idx + 1
            ))
        })?;
        events.push(event);
    }
    Ok(events)
}

fn compute_chain_hash(previous_hash: &[u8], event: &Event) -> Result<Vec<u8>> {
    let mut canonical = event.clone();
    canonical.chain_hash = None;
    let bytes = serde_json::to_vec(&canonical)?;
    let mut hasher = Sha256::new();
    hasher.update(previous_hash);
    hasher.update(bytes);
    Ok(hasher.finalize().to_vec())
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

fn hex_decode(input: &str) -> Result<Vec<u8>> {
    if input.len() % 2 != 0 {
        return Err(HideError::Storage("odd-length hex digest".to_string()));
    }
    let mut out = Vec::with_capacity(input.len() / 2);
    let bytes = input.as_bytes();
    for pair in bytes.chunks_exact(2) {
        let high = hex_val(pair[0])?;
        let low = hex_val(pair[1])?;
        out.push((high << 4) | low);
    }
    Ok(out)
}

fn hex_val(byte: u8) -> Result<u8> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(HideError::Storage("invalid hex digest".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn in_memory_log_assigns_ordered_sequences() {
        let log = InMemoryEventLog::new();
        let session = SessionId::new();
        let first = log
            .append(NewEvent::system(
                session.clone(),
                "system.started",
                EventPayload::Custom(Value::Null),
            ))
            .await
            .unwrap();
        let second = log
            .append(NewEvent::system(
                session.clone(),
                "system.ready",
                EventPayload::Custom(Value::Null),
            ))
            .await
            .unwrap();
        assert_eq!(first.seq, 1);
        assert_eq!(second.seq, 2);
        assert_eq!(log.scan(Some(session), None, None).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn jsonl_log_persists_and_reopens_with_chain_hashes() {
        let dir = std::env::temp_dir().join(format!("hide_event_log_{}", now_ms()));
        let path = dir.join("events.jsonl");
        let log = JsonlEventLog::open(&path).unwrap();
        let session = SessionId::new();
        let first = log
            .append(NewEvent::system(
                session.clone(),
                "system.started",
                EventPayload::Custom(Value::Null),
            ))
            .await
            .unwrap();
        let second = log
            .append(NewEvent::system(
                session.clone(),
                "system.ready",
                EventPayload::Custom(Value::Null),
            ))
            .await
            .unwrap();
        assert!(first.chain_hash.is_some());
        assert!(second.chain_hash.is_some());
        assert_ne!(first.chain_hash, second.chain_hash);

        let reopened = JsonlEventLog::open(&path).unwrap();
        let loaded = reopened
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert_eq!(loaded.len(), 2);
        let third = reopened
            .append(NewEvent::system(
                session,
                "system.done",
                EventPayload::Custom(Value::Null),
            ))
            .await
            .unwrap();
        assert_eq!(third.seq, 3);
        let _ = std::fs::remove_dir_all(dir);
    }
}
