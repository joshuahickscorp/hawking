use crate::error::{HideError, Result};
use crate::ids::{now_micros, EventId, GrantId, RunId, SessionId, StepId, ToolCallId};
use crate::types::{BlobRef, Decision, EffectSet, Provenance};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const EVENT_SCHEMA_VERSION: u16 = 1;

/// Event class (ch.01 §4.6). Following OpenHands, effect-bearing events are
/// `Action`s; their recorded outcomes are `Observation`s (carrying `cause`).
/// Most lifecycle events are `Neither`. Replay applies `Observation` outcomes
/// as data and never re-fires `Action`s (T3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventClass {
    Action,
    Observation,
    #[default]
    Neither,
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

/// The single immutable record (ch.01 §4.6 — the cross-cutting contract every
/// other chapter binds to).
///
/// The payload is an **open** `serde_json::Value`, not a closed Rust enum: the
/// bible explicitly rejected the giant-enum approach because it makes the core a
/// bottleneck for every new event kind and breaks WASM-plugin emission. Kinds
/// are open dotted strings (e.g. `"tool.call"`); typed constructors and
/// [`Event::payload_as`] keep emission/consumption type-safe over the `Value`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub schema_version: u16,
    pub seq: u64,
    pub id: EventId,
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    /// Direct causal parent (builds the causal DAG).
    pub parent: Option<EventId>,
    /// For `Observation`s: the `Action` that produced this (OpenHands-style
    /// Action/Observation pairing for replay — T3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<EventId>,
    /// Wall-clock micros since epoch (informational; `seq` is the order).
    pub ts: u64,
    pub source: EventSource,
    /// Sub-actor id (which agent/plugin/tool instance emitted this).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
    /// Event class (Action / Observation / Neither).
    #[serde(default)]
    pub class: EventClass,
    /// Open, dotted, registry-validated kind (e.g. `"tool.call"`).
    pub kind: String,
    /// Kind-specific body. Open `Value` so new kinds never require a core edit.
    /// Kept as a named field (not `#[serde(flatten)]`) so it never competes with
    /// the flattened `ext` forward-compat capture below.
    pub payload: Value,
    /// JSON-pointer paths scrubbed for privacy/secrets (the redaction is itself
    /// auditable).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub redactions: Vec<String>,
    /// blake3 chain hash over the canonical bytes + the previous hash. `None`
    /// until the durable log assigns it on append.
    // NOTE: hide-security/audit.rs must use the same blake3 chain — see WP-6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chain_hash: Option<String>,
    /// Forward-compat capture of unknown fields (T10) — additive-by-default.
    #[serde(flatten, default)]
    pub ext: BTreeMap<String, Value>,
}

impl Event {
    pub fn new(seq: u64, input: NewEvent) -> Self {
        Self {
            schema_version: EVENT_SCHEMA_VERSION,
            seq,
            id: EventId::new(),
            session_id: input.session_id,
            run_id: input.run_id,
            parent: input.parent,
            cause: input.cause,
            ts: now_micros(),
            source: input.source,
            actor: input.actor,
            class: input.class,
            kind: input.kind,
            payload: input.payload,
            redactions: input.redactions,
            chain_hash: None,
            ext: BTreeMap::new(),
        }
    }

    /// Deserialize the open payload into a typed view (e.g.
    /// `event.payload_as::<ToolCallEvent>()`). Returns `None` if the payload
    /// does not match the requested shape — the type-safe read side of the
    /// open-payload contract.
    pub fn payload_as<T: DeserializeOwned>(&self) -> Option<T> {
        serde_json::from_value(self.payload.clone()).ok()
    }

    pub fn is_action(&self) -> bool {
        self.class == EventClass::Action
    }

    pub fn is_observation(&self) -> bool {
        self.class == EventClass::Observation
    }
}

/// A not-yet-sequenced event. The single-writer log assigns `seq`/`id`/`ts`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NewEvent {
    pub session_id: SessionId,
    pub run_id: Option<RunId>,
    pub parent: Option<EventId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cause: Option<EventId>,
    pub source: EventSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actor: Option<String>,
    #[serde(default)]
    pub class: EventClass,
    pub kind: String,
    pub payload: Value,
    #[serde(default)]
    pub redactions: Vec<String>,
}

impl NewEvent {
    /// Build a `System`-sourced event with an arbitrary JSON payload.
    pub fn system(session_id: SessionId, kind: impl Into<String>, payload: Value) -> Self {
        Self {
            session_id,
            run_id: None,
            parent: None,
            cause: None,
            source: EventSource::System,
            actor: None,
            class: EventClass::Neither,
            kind: kind.into(),
            payload,
            redactions: Vec::new(),
        }
    }

    /// Generic builder: source + kind + open payload, class `Neither`.
    pub fn of(
        session_id: SessionId,
        source: EventSource,
        kind: impl Into<String>,
        payload: Value,
    ) -> Self {
        Self {
            session_id,
            run_id: None,
            parent: None,
            cause: None,
            source,
            actor: None,
            class: EventClass::Neither,
            kind: kind.into(),
            payload,
            redactions: Vec::new(),
        }
    }

    pub fn with_run(mut self, run_id: RunId) -> Self {
        self.run_id = Some(run_id);
        self
    }

    pub fn with_parent(mut self, parent: EventId) -> Self {
        self.parent = Some(parent);
        self
    }

    pub fn with_cause(mut self, cause: EventId) -> Self {
        self.cause = Some(cause);
        self
    }

    pub fn with_class(mut self, class: EventClass) -> Self {
        self.class = class;
        self
    }

    pub fn with_actor(mut self, actor: impl Into<String>) -> Self {
        self.actor = Some(actor.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Typed payload views + ergonomic constructors.
//
// Storage is open (`payload: Value`) but emission stays type-safe: each known
// kind has a serde view struct + a `NewEvent` constructor that serializes it.
// Reading back is `event.payload_as::<TheView>()`.
// ---------------------------------------------------------------------------

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

/// Serialize a typed payload view into the open `Value`. Infallible for the
/// in-tree view structs; falls back to `Null` only on the impossible serde
/// failure path.
fn to_payload<T: Serialize>(view: &T) -> Value {
    serde_json::to_value(view).unwrap_or(Value::Null)
}

impl NewEvent {
    pub fn user_intent(session_id: SessionId, view: UserIntentEvent) -> Self {
        Self {
            kind: "user.intent".to_string(),
            payload: to_payload(&view),
            class: EventClass::Action,
            ..Self::of(session_id, EventSource::User, "user.intent", Value::Null)
        }
    }

    pub fn agent_state(session_id: SessionId, run_id: RunId, view: AgentStateEvent) -> Self {
        Self {
            run_id: Some(run_id),
            payload: to_payload(&view),
            ..Self::of(session_id, EventSource::Agent, "agent.phase", Value::Null)
        }
    }

    pub fn plan(session_id: SessionId, run_id: RunId, view: PlanEvent) -> Self {
        Self {
            run_id: Some(run_id),
            payload: to_payload(&view),
            ..Self::of(session_id, EventSource::Agent, "plan.created", Value::Null)
        }
    }

    pub fn tool_call(session_id: SessionId, view: ToolCallEvent) -> Self {
        Self {
            payload: to_payload(&view),
            class: EventClass::Action,
            ..Self::of(session_id, EventSource::Agent, "tool.call", Value::Null)
        }
    }

    /// `cause` should point at the `tool.call` event id (Action/Observation
    /// pairing, T3).
    pub fn tool_result(session_id: SessionId, view: ToolResultEvent) -> Self {
        Self {
            payload: to_payload(&view),
            class: EventClass::Observation,
            ..Self::of(session_id, EventSource::Tool, "tool.result", Value::Null)
        }
    }

    pub fn observation(
        session_id: SessionId,
        kind: impl Into<String>,
        cause: EventId,
        payload: Value,
    ) -> Self {
        Self {
            cause: Some(cause),
            class: EventClass::Observation,
            ..Self::of(session_id, EventSource::Tool, kind, payload)
        }
    }

    pub fn error(session_id: SessionId, view: ErrorEvent) -> Self {
        Self {
            payload: to_payload(&view),
            ..Self::of(session_id, EventSource::System, "error", Value::Null)
        }
    }
}

pub trait EventLog: Send + Sync {
    fn append<'a>(&'a self, event: NewEvent) -> BoxFuture<'a, Result<Event>>;

    fn scan<'a>(
        &'a self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> BoxFuture<'a, Result<Vec<Event>>>;

    /// Spine B: archive events with `seq < before_seq` to a durable cold store so
    /// a compaction/summary can read them later, WITHOUT mutating the live
    /// (hash-chained) log — chain integrity is preserved by never rewriting in
    /// place. Returns the number of events archived. Default: no-op for logs that
    /// do not support archival.
    fn compact_before<'a>(&'a self, _before_seq: u64) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move { Ok(0) })
    }
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

    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
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

    /// Chain-safe archival: copy events with `seq < before_seq` into a sibling
    /// `<log>.archive` file (append-only) and leave the live, hash-chained log
    /// untouched. The cold store feeds compaction/summary; the live chain still
    /// verifies. (In-place truncation would re-anchor the chain and is deferred.)
    fn compact_before<'a>(&'a self, before_seq: u64) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let cold: Vec<Event> = read_events(&self.path)?
                .into_iter()
                .filter(|event| event.seq < before_seq)
                .collect();
            if cold.is_empty() {
                return Ok(0);
            }
            let archive_path = self.path.with_extension("jsonl.archive");
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&archive_path)?;
            for event in &cold {
                writeln!(file, "{}", serde_json::to_string(event)?)?;
            }
            Ok(cold.len())
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

/// blake3 chain hash over the previous hash + the canonical bytes of the event
/// (with `chain_hash` cleared). Replaces SHA-256 per ch.01 §4.7 / §4.8.
// NOTE: hide-security/audit.rs must use the same blake3 chain — see WP-6.
fn compute_chain_hash(previous_hash: &[u8], event: &Event) -> Result<Vec<u8>> {
    let mut canonical = event.clone();
    canonical.chain_hash = None;
    let bytes = serde_json::to_vec(&canonical)?;
    let mut hasher = blake3::Hasher::new();
    hasher.update(previous_hash);
    hasher.update(&bytes);
    Ok(hasher.finalize().as_bytes().to_vec())
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
    use crate::ids::now_ms;

    #[tokio::test]
    async fn in_memory_log_assigns_ordered_sequences() {
        let log = InMemoryEventLog::new();
        let session = SessionId::new();
        let first = log
            .append(NewEvent::system(
                session.clone(),
                "system.started",
                Value::Null,
            ))
            .await
            .unwrap();
        let second = log
            .append(NewEvent::system(
                session.clone(),
                "system.ready",
                Value::Null,
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
                Value::Null,
            ))
            .await
            .unwrap();
        let second = log
            .append(NewEvent::system(
                session.clone(),
                "system.ready",
                Value::Null,
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
            .append(NewEvent::system(session, "system.done", Value::Null))
            .await
            .unwrap();
        assert_eq!(third.seq, 3);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn blake3_chain_detects_tampering() {
        // Build a two-event chain, then tamper with the first event's payload
        // and assert its recomputed blake3 hash no longer matches the embedded
        // one (integrity / tamper detection, ch.01 §4.8).
        let session = SessionId::new();
        let mut first = Event::new(
            1,
            NewEvent::system(session.clone(), "a", serde_json::json!({ "n": 1 })),
        );
        let h1 = compute_chain_hash(&[0u8; 32], &first).unwrap();
        first.chain_hash = Some(hex_lower(&h1));

        let second = Event::new(
            2,
            NewEvent::system(session, "b", serde_json::json!({ "n": 2 })),
        );
        let h2 = compute_chain_hash(&h1, &second).unwrap();

        // Untampered: recomputation matches the embedded hash.
        let recomputed = compute_chain_hash(&[0u8; 32], &first).unwrap();
        assert_eq!(
            first.chain_hash.as_deref(),
            Some(hex_lower(&recomputed).as_str())
        );

        // Tamper with the payload → recomputed hash diverges, so the embedded
        // hash (and every downstream hash) no longer verifies.
        let mut tampered = first.clone();
        tampered.payload = serde_json::json!({ "n": 999 });
        let after = compute_chain_hash(&[0u8; 32], &tampered).unwrap();
        assert_ne!(h1, after, "tampering changes the chain hash");
        // The follow-on hash, recomputed from the tampered prefix, also breaks.
        let h2_after = compute_chain_hash(&after, &second).unwrap();
        assert_ne!(h2, h2_after, "tamper propagates down the chain");
    }

    #[test]
    fn typed_constructor_round_trips_through_value_payload() {
        let session = SessionId::new();
        let call_id = ToolCallId::new();
        let new = NewEvent::tool_call(
            session,
            ToolCallEvent {
                call_id: call_id.clone(),
                tool_name: "fs.read".to_string(),
                capability_grant_id: None,
                args: serde_json::json!({ "path": "src/main.rs" }),
                predicted_effects: EffectSet::default(),
            },
        );
        let event = Event::new(1, new);
        assert_eq!(event.kind, "tool.call");
        assert_eq!(event.class, EventClass::Action);
        let view: ToolCallEvent = event.payload_as().expect("payload is a ToolCallEvent");
        assert_eq!(view.call_id, call_id);
        assert_eq!(view.tool_name, "fs.read");
    }

    #[test]
    fn unknown_ext_fields_survive_serde_round_trip() {
        // An event serialized by a newer/foreign producer carrying an unknown
        // top-level field must round-trip it (T10 forward-compat).
        let session = SessionId::new();
        let event = Event::new(
            7,
            NewEvent::system(session, "future.kind", serde_json::json!({ "a": 1 })),
        );
        let mut as_json = serde_json::to_value(&event).unwrap();
        as_json["unknown_future_field"] = serde_json::json!({ "nested": true });
        let restored: Event = serde_json::from_value(as_json).unwrap();
        assert_eq!(
            restored.ext.get("unknown_future_field"),
            Some(&serde_json::json!({ "nested": true })),
            "unknown field captured in ext"
        );
        // And it must serialize back out (ext flattened) without loss.
        let reserialized = serde_json::to_value(&restored).unwrap();
        assert_eq!(reserialized["unknown_future_field"]["nested"], true);
        // The known open payload also survives (under the named `payload` field).
        assert_eq!(reserialized["payload"]["a"], 1);
    }
}
