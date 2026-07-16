//! Shared HIDE contracts.
//!
//! This crate intentionally contains data models, traits, and small in-memory
//! scaffolds only. Runtime-heavy pieces live behind traits so the Tauri host,
//! headless kernel tests, and future CLI can all share the same architecture.

#[rustfmt::skip]
pub mod api {
    use crate::ids::{EventId, RunId, SessionId};
    use crate::types::BlobRef;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", content = "data", rename_all = "snake_case")]
    pub enum Intent {
        SubmitTurn { session_id: SessionId, text: String, attachments: Vec<BlobRef> },
        CancelRun { run_id: RunId },
        PauseRun { run_id: RunId },
        ResumeRun { run_id: RunId },
        AcceptDiff { run_id: RunId, diff_id: String },
        RejectDiff { run_id: RunId, diff_id: String },
        ScrubToEvent { session_id: SessionId, event_id: EventId },
        ForkSession { session_id: SessionId, at_event: EventId },
        OpenFile { path: String, line: Option<u32> },
        RunCommand { argv: Vec<String>, cwd: Option<String> },
        Custom { name: String, payload: Value },
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct IntentAck {
        pub accepted: bool,
        pub event_seq: Option<u64>,
        pub message: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct UiEvent {
        pub seq: u64,
        pub session_id: Option<SessionId>,
        pub kind: UiEventKind,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", content = "data", rename_all = "snake_case")]
    pub enum UiEventKind {
        ProjectionPatch { projection: String, patch: Value },
        TokenBatch { stream_id: String, text: String },
        RuntimeStatus { status: String, detail: Option<String> },
        ToolProgress { call_id: String, message: String },
        SecurityGate { gate: String, message: String },
        Error { code: String, message: String },
        Custom(Value),
    }
}
#[rustfmt::skip]
pub mod config {
    use crate::types::Decision;
    use crate::Result;
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct HideConfig {
        pub user_root: PathBuf,
        pub workspace_root: PathBuf,
        pub runtime: RuntimeConfig,
        pub persistence: PersistenceConfig,
        pub security: SecurityConfig,
        pub context: ContextConfig,
        pub index: IndexConfig,
    }

    impl HideConfig {
        pub fn for_workspace(workspace_root: impl Into<PathBuf>) -> Self {
            let workspace_root = workspace_root.into();
            let user_root = std::env::var_os("HAWKING_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".hawking")))
                .unwrap_or_else(|| PathBuf::from(".hawking"));
            Self {
                user_root,
                workspace_root,
                runtime: RuntimeConfig::default(),
                persistence: PersistenceConfig::default(),
                security: SecurityConfig::default(),
                context: ContextConfig::default(),
                index: IndexConfig::default(),
            }
        }

        pub fn load_json(path: impl AsRef<Path>) -> Result<Self> {
            Ok(serde_json::from_slice(&std::fs::read(path)?)?)
        }

        pub fn save_json(&self, path: impl AsRef<Path>) -> Result<()> {
            let path = path.as_ref();
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_vec_pretty(self)?)?;
            Ok(())
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RuntimeConfig {
        pub provider: String,
        pub base_url: String,
        pub spawn_sidecar: bool,
        pub health_timeout_ms: u64,
        pub restart_backoff_ms: Vec<u64>,
    }

    impl Default for RuntimeConfig {
        fn default() -> Self {
            Self {
                provider: "hawking-local".to_string(),
                base_url: "http://127.0.0.1:8080".to_string(),
                spawn_sidecar: true,
                health_timeout_ms: 60_000,
                restart_backoff_ms: vec![1_000, 2_000, 4_000, 8_000, 30_000],
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PersistenceConfig {
        pub fsync_every_event: bool,
        pub segment_bytes: u64,
        pub snapshot_interval_events: u64,
        pub encryption_at_rest: bool,
    }

    impl Default for PersistenceConfig {
        fn default() -> Self {
            Self {
                fsync_every_event: false,
                segment_bytes: 64 * 1024 * 1024,
                snapshot_interval_events: 250,
                encryption_at_rest: false,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SecurityConfig {
        pub default_decision: Decision,
        pub network_default: Decision,
        pub shell_default: Decision,
        pub workspace_write_default: Decision,
        pub require_exact_effect_grants: bool,
    }

    impl Default for SecurityConfig {
        fn default() -> Self {
            Self {
                default_decision: Decision::Ask,
                network_default: Decision::Deny,
                shell_default: Decision::Ask,
                workspace_write_default: Decision::Ask,
                require_exact_effect_grants: true,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ContextConfig {
        pub max_input_tokens: usize,
        pub reserve_output_tokens: usize,
        pub memory_top_k: usize,
        pub code_top_k: usize,
    }

    impl Default for ContextConfig {
        fn default() -> Self {
            Self { max_input_tokens: 16_384, reserve_output_tokens: 2_048, memory_top_k: 12, code_top_k: 40 }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct IndexConfig {
        pub enable_daemon: bool,
        pub lexical_first: bool,
        pub embedding_rerank: bool,
        pub max_file_bytes: u64,
    }

    impl Default for IndexConfig {
        fn default() -> Self {
            Self { enable_daemon: true, lexical_first: true, embedding_rerank: true, max_file_bytes: 2 * 1024 * 1024 }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ConfigLayer {
        pub name: String,
        pub source: PathBuf,
        pub locked: bool,
        pub config: HideConfig,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn config_roundtrips_as_json() {
            let dir = std::env::temp_dir().join(format!("hide_config_{}", crate::ids::now_ms()));
            let path = dir.join(".hide").join("config.json");
            let mut config = HideConfig::for_workspace(&dir);
            config.runtime.base_url = "http://127.0.0.1:9999".to_string();

            config.save_json(&path).unwrap();
            let loaded = HideConfig::load_json(&path).unwrap();

            assert_eq!(loaded.workspace_root, dir);
            assert_eq!(loaded.runtime.base_url, "http://127.0.0.1:9999");
            let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap());
        }
    }
}
#[rustfmt::skip]
pub mod error {
    use thiserror::Error;

    #[derive(Debug, Error)]
    pub enum HideError {
        #[error("{0}")]
        Message(String),

        #[error("configuration error: {0}")]
        Config(String),

        #[error("policy denied: {0}")]
        PolicyDenied(String),

        #[error("capability missing: {0}")]
        CapabilityMissing(String),

        #[error("not found: {0}")]
        NotFound(String),

        #[error("invalid state: {0}")]
        InvalidState(String),

        #[error("tool error: {0}")]
        Tool(String),

        #[error("runtime unavailable: {0}")]
        RuntimeUnavailable(String),

        #[error("storage error: {0}")]
        Storage(String),

        #[error(transparent)]
        Io(#[from] std::io::Error),

        #[error(transparent)]
        Serde(#[from] serde_json::Error),
    }

    impl HideError {
        pub fn msg(message: impl Into<String>) -> Self {
            Self::Message(message.into())
        }
    }

    pub type Result<T> = std::result::Result<T, HideError>;
}
#[rustfmt::skip]
pub mod event {
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
        pub fn of(session_id: SessionId, source: EventSource, kind: impl Into<String>, payload: Value) -> Self {
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

        pub fn observation(session_id: SessionId, kind: impl Into<String>, cause: EventId, payload: Value) -> Self {
            Self {
                cause: Some(cause),
                class: EventClass::Observation,
                ..Self::of(session_id, EventSource::Tool, kind, payload)
            }
        }

        pub fn error(session_id: SessionId, view: ErrorEvent) -> Self {
            Self { payload: to_payload(&view), ..Self::of(session_id, EventSource::System, "error", Value::Null) }
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
            Ok(Self { path, state: Mutex::new(JsonlEventLogState { next_seq, previous_hash }) })
        }

        pub fn path(&self) -> &Path {
            &self.path
        }
    }

    impl InMemoryEventLog {
        pub fn new() -> Self {
            Self { next_seq: AtomicU64::new(1), events: Mutex::new(Vec::new()) }
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
                        .is_some_and(|sid| sid != &event.session_id)
                    {
                        continue;
                    }
                    if after_seq.is_some_and(|seq| event.seq <= seq) {
                        continue;
                    }
                    out.push(event.clone());
                    if limit.is_some_and(|n| out.len() >= n) {
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

                let mut file = OpenOptions::new().create(true).append(true).open(&self.path)?;
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
                        .is_some_and(|sid| sid != &event.session_id)
                    {
                        continue;
                    }
                    if after_seq.is_some_and(|seq| event.seq <= seq) {
                        continue;
                    }
                    out.push(event);
                    if limit.is_some_and(|n| out.len() >= n) {
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
                let cold: Vec<Event> =
                    read_events(&self.path)?.into_iter().filter(|event| event.seq < before_seq).collect();
                if cold.is_empty() {
                    return Ok(0);
                }
                let archive_path = self.path.with_extension("jsonl.archive");
                let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&archive_path)?;
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
                HideError::Storage(format!("failed to parse event log {} line {}: {err}", path.display(), idx + 1))
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
            let first = log.append(NewEvent::system(session.clone(), "system.started", Value::Null)).await.unwrap();
            let second = log.append(NewEvent::system(session.clone(), "system.ready", Value::Null)).await.unwrap();
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
            let first = log.append(NewEvent::system(session.clone(), "system.started", Value::Null)).await.unwrap();
            let second = log.append(NewEvent::system(session.clone(), "system.ready", Value::Null)).await.unwrap();
            assert!(first.chain_hash.is_some());
            assert!(second.chain_hash.is_some());
            assert_ne!(first.chain_hash, second.chain_hash);

            let reopened = JsonlEventLog::open(&path).unwrap();
            let loaded = reopened.scan(Some(session.clone()), None, None).await.unwrap();
            assert_eq!(loaded.len(), 2);
            let third = reopened.append(NewEvent::system(session, "system.done", Value::Null)).await.unwrap();
            assert_eq!(third.seq, 3);
            let _ = std::fs::remove_dir_all(dir);
        }

        #[test]
        fn blake3_chain_detects_tampering() {
            // Build a two-event chain, then tamper with the first event's payload
            // and assert its recomputed blake3 hash no longer matches the embedded
            // one (integrity / tamper detection, ch.01 §4.8).
            let session = SessionId::new();
            let mut first = Event::new(1, NewEvent::system(session.clone(), "a", serde_json::json!({ "n": 1 })));
            let h1 = compute_chain_hash(&[0u8; 32], &first).unwrap();
            first.chain_hash = Some(hex_lower(&h1));

            let second = Event::new(2, NewEvent::system(session, "b", serde_json::json!({ "n": 2 })));
            let h2 = compute_chain_hash(&h1, &second).unwrap();

            // Untampered: recomputation matches the embedded hash.
            let recomputed = compute_chain_hash(&[0u8; 32], &first).unwrap();
            assert_eq!(first.chain_hash.as_deref(), Some(hex_lower(&recomputed).as_str()));

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
            let event = Event::new(7, NewEvent::system(session, "future.kind", serde_json::json!({ "a": 1 })));
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
}
#[rustfmt::skip]
pub mod ids {
    use serde::{Deserialize, Serialize};
    use std::cell::Cell;
    use std::fmt;
    use std::time::{SystemTime, UNIX_EPOCH};
    use ulid::Ulid;

    thread_local! {
        /// Optional deterministic id source for tests. When set, ids are minted as
        /// `{prefix}_{seed:026X}` with a monotonically increasing seed instead of a
        /// random ULID, so replay/integrity tests are reproducible (T6).
        static DETERMINISTIC_SEED: Cell<Option<u128>> = const { Cell::new(None) };
    }

    /// Scope guard installing a deterministic, monotonically increasing id source on
    /// the current thread for the duration of the closure. Restores the previous
    /// source on exit. Intended for tests that assert id monotonicity / replay.
    pub fn with_deterministic_ids<T>(start: u128, f: impl FnOnce() -> T) -> T {
        let previous = DETERMINISTIC_SEED.with(|cell| cell.replace(Some(start)));
        let out = f();
        DETERMINISTIC_SEED.with(|cell| cell.set(previous));
        out
    }

    fn next_ulid_body() -> String {
        if let Some(seed) = DETERMINISTIC_SEED.with(|cell| cell.get()) {
            DETERMINISTIC_SEED.with(|cell| cell.set(Some(seed + 1)));
            // Encode the seed as a ULID so it stays lexicographically sortable and
            // 26 chars wide, matching the random path's format.
            return Ulid::from(seed).to_string();
        }
        Ulid::new().to_string()
    }

    fn next_id(prefix: &str) -> String {
        format!("{prefix}_{}", next_ulid_body())
    }

    macro_rules! id_newtype {
        ($name:ident, $prefix:literal) => {
            #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
            pub struct $name(pub String);

            impl $name {
                pub fn new() -> Self {
                    Self(next_id($prefix))
                }

                pub fn as_str(&self) -> &str {
                    &self.0
                }
            }

            impl Default for $name {
                fn default() -> Self {
                    Self::new()
                }
            }

            impl From<String> for $name {
                fn from(value: String) -> Self {
                    Self(value)
                }
            }

            impl From<&str> for $name {
                fn from(value: &str) -> Self {
                    Self(value.to_string())
                }
            }

            impl fmt::Display for $name {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str(&self.0)
                }
            }
        };
    }

    id_newtype!(EventId, "evt");
    id_newtype!(SessionId, "ses");
    id_newtype!(RunId, "run");
    id_newtype!(PlanId, "pln");
    id_newtype!(StepId, "stp");
    id_newtype!(ToolCallId, "tcl");
    id_newtype!(ToolResultId, "trs");
    id_newtype!(GrantId, "gnt");
    id_newtype!(PluginId, "plg");
    id_newtype!(WorkspaceId, "wsp");
    id_newtype!(BlobId, "blb");
    id_newtype!(ValueId, "val");
    id_newtype!(ModelId, "mdl");
    id_newtype!(RoleId, "rol");

    pub type TimestampMs = u64;

    pub fn now_ms() -> TimestampMs {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64
    }

    /// Wall-clock microseconds since the Unix epoch — the `ts` field on `Event`
    /// (informational; `seq` is the authoritative order, ch.01 §4.6).
    pub fn now_micros() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros() as u64
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn ulid_ids_are_sortable_and_unique() {
            let a = EventId::new();
            let b = EventId::new();
            assert_ne!(a, b);
            assert!(a.as_str().starts_with("evt_"));
            // ULIDs minted later sort lexicographically >= earlier ones.
            assert!(b.as_str() >= a.as_str());
        }

        #[test]
        fn deterministic_ids_are_monotonic_and_reproducible() {
            let first = with_deterministic_ids(0, || (0..4).map(|_| EventId::new().0).collect::<Vec<_>>());
            let second = with_deterministic_ids(0, || (0..4).map(|_| EventId::new().0).collect::<Vec<_>>());
            assert_eq!(first, second, "same seed yields identical id sequence");
            for pair in first.windows(2) {
                assert!(pair[1] > pair[0], "deterministic ids are strictly increasing");
            }
        }
    }
}
#[rustfmt::skip]
pub mod migration {
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    pub struct SchemaVersion {
        pub major: u16,
        pub minor: u16,
        pub patch: u16,
    }

    impl SchemaVersion {
        pub const fn new(major: u16, minor: u16, patch: u16) -> Self {
            Self { major, minor, patch }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MigrationPlan {
        pub artifact: String,
        pub from: SchemaVersion,
        pub to: SchemaVersion,
        pub steps: Vec<MigrationStep>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MigrationStep {
        pub id: String,
        pub description: String,
        pub replayable: bool,
        pub destructive: bool,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct MigrationReport {
        pub applied: Vec<String>,
        pub skipped: Vec<String>,
        pub warnings: Vec<String>,
    }
}
#[rustfmt::skip]
pub mod observability {
    use crate::ids::{now_ms, RunId, SessionId, TimestampMs};
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum LogLevel {
        Trace,
        Debug,
        Info,
        Warn,
        Error,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct LogRecord {
        pub ts_ms: TimestampMs,
        pub level: LogLevel,
        pub target: String,
        pub message: String,
        pub session_id: Option<SessionId>,
        pub run_id: Option<RunId>,
        pub fields: BTreeMap<String, String>,
    }

    impl LogRecord {
        pub fn info(target: impl Into<String>, message: impl Into<String>) -> Self {
            Self {
                ts_ms: now_ms(),
                level: LogLevel::Info,
                target: target.into(),
                message: message.into(),
                session_id: None,
                run_id: None,
                fields: BTreeMap::new(),
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct MetricSample {
        pub ts_ms: TimestampMs,
        pub name: String,
        pub value: f64,
        pub labels: BTreeMap<String, String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct HealthReport {
        pub component: String,
        pub status: HealthStatus,
        pub checks: Vec<HealthCheck>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum HealthStatus {
        Ok,
        Degraded,
        Failed,
        Unknown,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct HealthCheck {
        pub name: String,
        pub status: HealthStatus,
        pub detail: String,
    }
}
#[rustfmt::skip]
pub mod permission {
    use crate::error::{HideError, Result};
    use crate::ids::{GrantId, PluginId, RunId, TimestampMs};
    use crate::types::{Decision, Effect, EffectKind, ResourceScope, RiskLevel};
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Capability {
        pub kind: String,
        pub scope: ResourceScope,
    }

    impl Capability {
        pub fn new(kind: impl Into<String>, pattern: impl Into<String>) -> Self {
            let kind = kind.into();
            Self { scope: ResourceScope { kind: kind.clone(), pattern: pattern.into() }, kind }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct CapabilityGrant {
        pub id: GrantId,
        pub capabilities: Vec<Capability>,
        pub decision: Decision,
        pub granted_by: GrantActor,
        pub run_id: Option<RunId>,
        pub plugin_id: Option<PluginId>,
        pub expires_at_ms: Option<TimestampMs>,
        pub exact_effect_hash: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum GrantActor {
        User,
        Policy,
        System,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PermissionRule {
        pub id: String,
        pub capability_kind: String,
        pub scope_pattern: String,
        pub decision: Decision,
        pub max_risk: RiskLevel,
        pub reason: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PermissionPolicy {
        pub default_decision: Decision,
        pub rules: Vec<PermissionRule>,
        pub risk_gates: Vec<RiskGate>,
    }

    impl Default for PermissionPolicy {
        fn default() -> Self {
            Self { default_decision: Decision::Ask, rules: Vec::new(), risk_gates: vec![RiskGate::lethal_trifecta()] }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct RiskGate {
        pub id: String,
        pub description: String,
        pub forced_decision: Decision,
    }

    impl RiskGate {
        pub fn lethal_trifecta() -> Self {
            Self {
                id: "lethal_trifecta".to_string(),
                description: "private data + untrusted content + exfiltration ability must be explicitly gated"
                    .to_string(),
                forced_decision: Decision::Ask,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PermissionRequest {
        pub capability_kind: String,
        pub target: String,
        pub risk: RiskLevel,
        pub effects: Vec<Effect>,
        pub grant: Option<CapabilityGrant>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct PermissionVerdict {
        pub decision: Decision,
        pub reason: String,
        pub grant_id: Option<GrantId>,
    }

    pub trait PermissionEngine: Send + Sync {
        fn evaluate(&self, request: &PermissionRequest) -> PermissionVerdict;

        fn require_allowed(&self, request: &PermissionRequest) -> Result<()> {
            let verdict = self.evaluate(request);
            match verdict.decision {
                Decision::Allow => Ok(()),
                Decision::Ask => Err(HideError::PolicyDenied(format!(
                    "approval required for {}: {}",
                    request.capability_kind, verdict.reason
                ))),
                Decision::Deny => Err(HideError::PolicyDenied(verdict.reason)),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct StaticPermissionEngine {
        pub policy: PermissionPolicy,
    }

    impl StaticPermissionEngine {
        pub fn new(policy: PermissionPolicy) -> Self {
            Self { policy }
        }

        fn rule_matches(rule: &PermissionRule, request: &PermissionRequest) -> bool {
            rule.capability_kind == request.capability_kind
                && pattern_matches(&rule.scope_pattern, &request.target)
                && request.risk <= rule.max_risk
        }
    }

    impl PermissionEngine for StaticPermissionEngine {
        fn evaluate(&self, request: &PermissionRequest) -> PermissionVerdict {
            if request.effects.iter().any(|e| e.kind == EffectKind::Network && e.risk >= RiskLevel::High) {
                return PermissionVerdict {
                    decision: Decision::Ask,
                    reason: "high-risk network effect requires explicit approval".to_string(),
                    grant_id: request.grant.as_ref().map(|g| g.id.clone()),
                };
            }

            let matching: Vec<_> = self.policy.rules.iter().filter(|rule| Self::rule_matches(rule, request)).collect();

            if let Some(rule) = matching.iter().find(|rule| rule.decision == Decision::Deny) {
                return PermissionVerdict {
                    decision: Decision::Deny,
                    reason: rule.reason.clone(),
                    grant_id: request.grant.as_ref().map(|g| g.id.clone()),
                };
            }

            if let Some(rule) = matching.iter().find(|rule| rule.decision == Decision::Allow) {
                return PermissionVerdict {
                    decision: Decision::Allow,
                    reason: rule.reason.clone(),
                    grant_id: request.grant.as_ref().map(|g| g.id.clone()),
                };
            }

            PermissionVerdict {
                decision: self.policy.default_decision,
                reason: "no matching policy rule".to_string(),
                grant_id: request.grant.as_ref().map(|g| g.id.clone()),
            }
        }
    }

    fn pattern_matches(pattern: &str, target: &str) -> bool {
        if pattern == "*" || pattern == "**" {
            return true;
        }
        if let Some(prefix) = pattern.strip_suffix("/**") {
            return target.starts_with(prefix);
        }
        pattern == target
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn deny_beats_allow() {
            let engine = StaticPermissionEngine::new(PermissionPolicy {
                default_decision: Decision::Ask,
                rules: vec![
                    PermissionRule {
                        id: "allow".to_string(),
                        capability_kind: "fs.write".to_string(),
                        scope_pattern: "/tmp/**".to_string(),
                        decision: Decision::Allow,
                        max_risk: RiskLevel::High,
                        reason: "tmp allowed".to_string(),
                    },
                    PermissionRule {
                        id: "deny".to_string(),
                        capability_kind: "fs.write".to_string(),
                        scope_pattern: "/tmp/secrets/**".to_string(),
                        decision: Decision::Deny,
                        max_risk: RiskLevel::Critical,
                        reason: "secrets denied".to_string(),
                    },
                ],
                risk_gates: Vec::new(),
            });
            let verdict = engine.evaluate(&PermissionRequest {
                capability_kind: "fs.write".to_string(),
                target: "/tmp/secrets/key".to_string(),
                risk: RiskLevel::Low,
                effects: Vec::new(),
                grant: None,
            });
            assert_eq!(verdict.decision, Decision::Deny);
        }
    }
}
#[rustfmt::skip]
pub mod persistence {
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
            Ok(BlobRef { id, hash, size_bytes: bytes.len() as u64, media_type })
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
            Ok(BlobRef { id: BlobId::new(), hash, size_bytes: bytes.len() as u64, media_type })
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
            self.projections.lock().entry(session_id.clone()).or_default().push((seq, projection));
            Ok(())
        }

        fn latest_projection(&self, session_id: &SessionId) -> Result<Option<(u64, Value)>> {
            Ok(self.projections.lock().get(session_id).and_then(|items| items.last().cloned()))
        }
    }

    impl ProjectionStore for FileProjectionStore {
        fn put_projection(&self, session_id: &SessionId, seq: u64, projection: Value) -> Result<()> {
            let path = self.session_path(session_id);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let record = ProjectionRecord { seq, projection };
            let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
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
            self.tables.lock().entry(table.to_string()).or_default().insert(key.to_string(), value);
            Ok(())
        }

        fn get(&self, table: &str, key: &str) -> Result<Option<Value>> {
            Ok(self.tables.lock().get(table).and_then(|items| items.get(key).cloned()))
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
            self.root.join(sanitize_component(table)).join(format!("{}.json", sanitize_component(key)))
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
            let blob = store.put(b"durable bytes".to_vec(), Some("text/plain".to_string())).unwrap();
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
            store.put_projection(&session, 1, serde_json::json!({ "phase": "plan" })).unwrap();
            store.put_projection(&session, 2, serde_json::json!({ "phase": "done" })).unwrap();
            let latest = store.latest_projection(&session).unwrap().unwrap();
            assert_eq!(latest.0, 2);
            assert_eq!(latest.1["phase"], "done");
            let _ = std::fs::remove_dir_all(dir);
        }

        #[test]
        fn file_key_value_store_roundtrips_json_values() {
            let dir = std::env::temp_dir().join(format!("hide_kv_{}", crate::ids::now_ms()));
            let store = FileKeyValueStore::open(&dir).unwrap();
            store.put("sessions", "session/with/slashes", serde_json::json!({ "state": "running" })).unwrap();
            let loaded = store.get("sessions", "session/with/slashes").unwrap().unwrap();
            assert_eq!(loaded["state"], "running");
            assert!(store.get("sessions", "missing").unwrap().is_none());
            let _ = std::fs::remove_dir_all(dir);
        }
    }
}
#[rustfmt::skip]
pub mod plugin {
    use crate::ids::PluginId;
    use crate::permission::Capability;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ExtensionManifest {
        pub id: PluginId,
        pub name: String,
        pub version: String,
        pub runtime: ExtensionRuntime,
        pub required_capabilities: Vec<Capability>,
        pub contributions: Vec<ExtensionContribution>,
        pub metadata: BTreeMap<String, String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ExtensionRuntime {
        TrustedRust,
        WasmComponent,
        McpServer,
        Skill,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    pub enum ExtensionContribution {
        Tool { name: String },
        Panel { id: String },
        ModelProvider { id: String },
        Indexer { language: String },
        MemoryStore { id: String },
        Command { id: String },
        EventKind { event_kind: String },
    }

    #[derive(Debug, Default)]
    pub struct ExtensionRegistry {
        manifests: BTreeMap<PluginId, ExtensionManifest>,
    }

    impl ExtensionRegistry {
        pub fn register(&mut self, manifest: ExtensionManifest) {
            self.manifests.insert(manifest.id.clone(), manifest);
        }

        pub fn get(&self, id: &PluginId) -> Option<&ExtensionManifest> {
            self.manifests.get(id)
        }

        pub fn len(&self) -> usize {
            self.manifests.len()
        }

        pub fn is_empty(&self) -> bool {
            self.manifests.is_empty()
        }
    }
}
#[rustfmt::skip]
pub mod project {
    use crate::ids::WorkspaceId;
    use serde::{Deserialize, Serialize};
    use std::path::{Path, PathBuf};

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Workspace {
        pub id: WorkspaceId,
        pub root: PathBuf,
        pub hide_dir: PathBuf,
    }

    impl Workspace {
        pub fn new(root: impl Into<PathBuf>) -> Self {
            let root = root.into();
            let hide_dir = root.join(".hide");
            Self { id: WorkspaceId::new(), root, hide_dir }
        }

        pub fn layout(&self) -> WorkspaceLayout {
            WorkspaceLayout::new(&self.root)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct WorkspaceLayout {
        pub root: PathBuf,
        pub hide_dir: PathBuf,
        pub event_log: PathBuf,
        pub snapshots: PathBuf,
        pub projections: PathBuf,
        pub metadata_db: PathBuf,
        pub kv: PathBuf,
        pub vectors_db: PathBuf,
        pub blobs: PathBuf,
        pub taint: PathBuf,
        pub cache: PathBuf,
        pub sandbox: PathBuf,
        pub tmp: PathBuf,
    }

    impl WorkspaceLayout {
        pub fn new(root: &Path) -> Self {
            let hide_dir = root.join(".hide");
            Self {
                root: root.to_path_buf(),
                event_log: hide_dir.join("log"),
                snapshots: hide_dir.join("snapshots"),
                projections: hide_dir.join("projections"),
                metadata_db: hide_dir.join("meta.sqlite"),
                kv: hide_dir.join("kv"),
                vectors_db: hide_dir.join("vectors.sqlite"),
                blobs: hide_dir.join("blobs"),
                taint: hide_dir.join("taint"),
                cache: hide_dir.join("cache"),
                sandbox: hide_dir.join("sandbox"),
                tmp: hide_dir.join("tmp"),
                hide_dir,
            }
        }
    }
}
#[rustfmt::skip]
pub mod runtime {
    use crate::error::Result;
    use crate::ids::{ModelId, RoleId};
    use futures::future::BoxFuture;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ProviderCaps {
        pub streaming: bool,
        pub embeddings: bool,
        pub grammar: bool,
        pub raw_logits: bool,
        pub logprobs: bool,
        pub lora: bool,
        pub kv_handles: bool,
        pub native_tokens_endpoint: bool,
    }

    impl ProviderCaps {
        pub fn hawking_local_shell_today() -> Self {
            Self {
                streaming: true,
                embeddings: true,
                grammar: false,
                raw_logits: false,
                logprobs: false,
                lora: false,
                kv_handles: false,
                native_tokens_endpoint: true,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ModelDescriptor {
        pub id: ModelId,
        pub name: String,
        pub architecture: ModelArchitecture,
        pub context_tokens: usize,
        pub tokenizer_signature: String,
        pub footprint_mb: u64,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ModelArchitecture {
        Transformer,
        Ssm,
        Hybrid,
        Unknown,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ModelRole {
        pub id: RoleId,
        pub name: String,
        pub purpose: RolePurpose,
        pub model: ModelDescriptor,
        pub caps: ProviderCaps,
        pub default_sampler: SamplerProfile,
        /// Localhost endpoint this role is served from (ch.06 §4.4). Optional so
        /// roles can be declared before an endpoint is resolved.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub endpoint: Option<String>,
        /// Relative cost hint for the scheduler/admission (ch.06 §4.11).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub cost: Option<f32>,
        /// The role to escalate to when confidence is low — expresses the
        /// confidence-aware cascade graph (ch.06 §4.4, the chapter's thesis).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub escalates_to: Option<RoleId>,
        pub metadata: BTreeMap<String, String>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RolePurpose {
        HeroCoder,
        FastDraft,
        Embedder,
        Reranker,
        Summarizer,
        Classifier,
        ToolPlanner,
        /// Long-context SSM (RWKV-7 / Mamba-2) routing (ch.06).
        SsmLong,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct SamplerProfile {
        pub temperature: f32,
        pub top_k: Option<u32>,
        pub top_p: Option<f32>,
        pub repetition_penalty: Option<f32>,
        pub seed: Option<u64>,
        pub deterministic: bool,
    }

    impl SamplerProfile {
        pub fn deterministic_edit() -> Self {
            Self {
                temperature: 0.0,
                top_k: None,
                top_p: None,
                repetition_penalty: None,
                seed: Some(0),
                deterministic: true,
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct InferenceRequest {
        pub task_kind: String,
        pub prompt: String,
        pub messages: Vec<InferenceMessage>,
        pub max_output_tokens: usize,
        pub sampler: Option<SamplerProfile>,
        pub grammar: Option<String>,
        pub want_logprobs: bool,
        pub metadata: BTreeMap<String, String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct InferenceMessage {
        pub role: String,
        pub content: String,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum StreamChunk {
        Token { token_id: Option<u32>, text: String },
        Done { reason: String, stats: Option<GenerationStats> },
        Error { message: String },
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct GenerationStats {
        pub input_tokens: usize,
        pub output_tokens: usize,
        pub decode_tokens_per_second: Option<f32>,
    }

    pub type TokenSink<'a> = &'a mut (dyn FnMut(StreamChunk) -> Result<()> + Send);

    pub trait ModelProvider: Send + Sync {
        fn id(&self) -> &str;
        fn capabilities(&self) -> ProviderCaps;
        fn generate<'a>(
            &'a self,
            request: InferenceRequest,
            sink: TokenSink<'a>,
        ) -> BoxFuture<'a, Result<GenerationStats>>;
        fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RuntimeSupervisorState {
        Down,
        Booting,
        Ready,
        Degraded,
        Failed,
    }
}
#[rustfmt::skip]
pub mod security {
    use crate::ids::ValueId;
    use crate::types::{Decision, Provenance, TrustLevel};
    use serde::{Deserialize, Serialize};

    // NOTE: not `Eq` — `Provenance.confidence` is an `f32`.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct TaintedValue {
        pub id: ValueId,
        pub provenance: Provenance,
        pub labels: Vec<String>,
    }

    impl TaintedValue {
        pub fn trusted(source: impl Into<String>) -> Self {
            Self { id: ValueId::new(), provenance: Provenance::trusted(source), labels: Vec::new() }
        }

        pub fn is_untrusted(&self) -> bool {
            matches!(self.provenance.trust, TrustLevel::ToolOutput | TrustLevel::Network | TrustLevel::Untrusted)
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct LethalTrifectaAssessment {
        pub has_private_data: bool,
        pub has_untrusted_content: bool,
        pub has_exfiltration: bool,
        pub decision: Decision,
        pub reason: String,
    }

    impl LethalTrifectaAssessment {
        pub fn assess(has_private_data: bool, has_untrusted_content: bool, has_exfiltration: bool) -> Self {
            let triggered = has_private_data && has_untrusted_content && has_exfiltration;
            Self {
                has_private_data,
                has_untrusted_content,
                has_exfiltration,
                decision: if triggered { Decision::Ask } else { Decision::Allow },
                reason: if triggered {
                    "lethal-trifecta risk requires explicit approval".to_string()
                } else {
                    "no lethal-trifecta risk".to_string()
                },
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct SandboxProfile {
        pub tier: SandboxTier,
        pub read_roots: Vec<String>,
        pub write_roots: Vec<String>,
        pub allowed_commands: Vec<String>,
        pub network: NetworkPolicy,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum SandboxTier {
        None,
        ReadOnly,
        WorkspaceWrite,
        Seatbelt,
        MicroVm,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct NetworkPolicy {
        pub default: Decision,
        pub allowed_hosts: Vec<String>,
        pub denied_hosts: Vec<String>,
    }

    impl Default for NetworkPolicy {
        fn default() -> Self {
            Self { default: Decision::Deny, allowed_hosts: Vec::new(), denied_hosts: Vec::new() }
        }
    }
}
#[rustfmt::skip]
pub mod supervision {
    use crate::ids::TimestampMs;
    use crate::runtime::RuntimeSupervisorState;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ProcessSpec {
        pub name: String,
        pub argv: Vec<String>,
        pub cwd: Option<String>,
        pub env: BTreeMap<String, String>,
        pub health_url: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ProcessStatus {
        pub name: String,
        pub pid: Option<u32>,
        pub state: RuntimeSupervisorState,
        pub started_at_ms: Option<TimestampMs>,
        pub restarts: u32,
        pub last_error: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BackoffPolicy {
        pub delays_ms: Vec<u64>,
        pub max_restarts_per_window: u32,
        pub window_ms: u64,
    }

    impl Default for BackoffPolicy {
        fn default() -> Self {
            Self {
                delays_ms: vec![1_000, 2_000, 4_000, 8_000, 30_000],
                max_restarts_per_window: 5,
                window_ms: 5 * 60 * 1000,
            }
        }
    }
}
#[rustfmt::skip]
pub mod tool {
    use crate::error::{HideError, Result};
    use crate::ids::{GrantId, ToolCallId};
    use crate::permission::{PermissionEngine, PermissionRequest};
    use crate::types::{BlobRef, Decision, EffectSet, RiskLevel};
    use futures::future::BoxFuture;
    use parking_lot::RwLock;
    use serde::{Deserialize, Serialize};
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ToolSpec {
        pub name: String,
        pub title: String,
        pub version: String,
        pub wire_version: u16,
        pub description: String,
        pub input_schema: Value,
        pub output_schema: Option<Value>,
        pub annotations: ToolAnnotations,
        pub capabilities_required: Vec<String>,
        pub output_cap_bytes: u64,
        pub timeout_ms: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ToolAnnotations {
        pub read_only: bool,
        pub destructive: bool,
        pub idempotent: bool,
        pub open_world: bool,
    }

    impl Default for ToolAnnotations {
        fn default() -> Self {
            Self { read_only: false, destructive: true, idempotent: false, open_world: true }
        }
    }

    /// The canonical effect request (ch.03 §4.2.2). Maps 1:1 to the `tool.call`
    /// event payload. Field names match the bible wire shape exactly.
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ToolCall {
        /// ULID; unique within the run; correlates result + events.
        pub call_id: ToolCallId,
        /// `ToolSpec.name` (registry-resolved).
        pub tool: String,
        pub args: Value,
        /// References Ch.01's grant ledger (TT3).
        pub capability_grant_id: Option<GrantId>,
        pub wire_version: u16,
        /// Optional execution directives (dry-run, idempotency, timeout override).
        #[serde(default)]
        pub x: ToolCallExt,
    }

    /// Optional per-call execution directives (`ToolCall.x`, ch.03 §4.2.2).
    #[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub struct ToolCallExt {
        #[serde(default)]
        pub dry_run: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub idempotency_key: Option<String>,
        /// `≤` the spec cap; cannot exceed it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub timeout_ms_override: Option<u64>,
    }

    impl ToolCall {
        /// Convenience constructor with a fresh `call_id`, current wire version, and
        /// default execution directives.
        pub fn new(tool: impl Into<String>, args: Value) -> Self {
            Self {
                call_id: ToolCallId::new(),
                tool: tool.into(),
                args,
                capability_grant_id: None,
                wire_version: TOOL_WIRE_VERSION,
                x: ToolCallExt::default(),
            }
        }
    }

    /// The current tool-wire-format version (ch.03 §4.2, TT11).
    pub const TOOL_WIRE_VERSION: u16 = 1;

    /// The canonical recorded outcome (ch.03 §4.2.3). Maps 1:1 to the `tool.result`
    /// event payload (TT4). `output` is the typed body; large bodies spill to
    /// `bytes_ref`. `provenance` marks the body as UNTRUSTED data, not instructions
    /// (TT8).
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct ToolResult {
        pub call_id: ToolCallId,
        /// `false` ⇒ see `error`; maps to MCP `isError` (§4.10). NOTE: a non-zero
        /// process exit is `ok:true` with `exit_code` set — `EXEC_NONZERO` is data,
        /// not a tool failure (§4.2.3).
        pub ok: bool,
        pub status: ToolStatus,
        /// Optional MCP-style multimodal blocks.
        #[serde(default)]
        pub content: Vec<ToolContent>,
        /// Typed body validated against `ToolSpec.output_schema` (`output`).
        #[serde(default)]
        pub structured_content: Option<Value>,
        /// Large bodies spill here as a blake3 CAS ref (TT5).
        pub bytes_ref: Option<BlobRef>,
        /// For process-shaped tools (shell/test/build); `None` otherwise.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub exit_code: Option<i32>,
        pub effects: EffectSet,
        /// TT8: the result body is untrusted data. Defaults to `"tool-output"`.
        #[serde(default = "default_tool_provenance")]
        pub provenance: String,
        /// Execution stats (duration, cache hit, dry-run origin).
        #[serde(default)]
        pub stats: ToolStats,
        pub error: Option<ToolError>,
    }

    fn default_tool_provenance() -> String {
        "tool-output".to_string()
    }

    /// Execution stats carried on every `ToolResult` (ch.03 §4.2.3 `stats`).
    #[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
    pub struct ToolStats {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub duration_ms: Option<u64>,
        #[serde(default)]
        pub cached: bool,
        #[serde(default)]
        pub from_dry_run: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum ToolStatus {
        Ok,
        ToolError,
        ProtocolError,
        Cancelled,
        TimedOut,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum ToolContent {
        Text { text: String },
        Json { value: Value },
        Blob { blob: BlobRef },
    }

    /// The structured failure (when `ok=false`). Designed to be self-correcting:
    /// the agent loop feeds it back so the model can fix the call (ch.03 §4.2.3).
    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ToolError {
        /// Stable taxonomy code (`ARG_INVALID`/`NOT_FOUND`/`EXEC_NONZERO`/… §4.2.3).
        pub code: String,
        pub message: String,
        /// Can the same model fix-and-retry? (Ch.02 uses this.)
        pub retriable: bool,
        /// Actionable hint for the model to repair the call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub fix_hint: Option<String>,
        /// JSON-pointer into `args` for precise UI/model targeting.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pub schema_path: Option<String>,
    }

    impl ToolError {
        /// Minimal constructor; `fix_hint`/`schema_path` default to `None`.
        pub fn new(code: impl Into<String>, message: impl Into<String>, retriable: bool) -> Self {
            Self { code: code.into(), message: message.into(), retriable, fix_hint: None, schema_path: None }
        }
    }

    impl ToolResult {
        pub fn ok(call_id: ToolCallId, structured_content: Option<Value>, effects: EffectSet) -> Self {
            Self {
                call_id,
                ok: true,
                status: ToolStatus::Ok,
                content: Vec::new(),
                structured_content,
                bytes_ref: None,
                exit_code: None,
                effects,
                provenance: default_tool_provenance(),
                stats: ToolStats::default(),
                error: None,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct ToolCtx {
        pub grant_id: Option<GrantId>,
        pub deadline_ms: Option<u64>,
        pub output_cap_bytes: u64,
    }

    pub trait Tool: Send + Sync {
        fn spec(&self) -> &ToolSpec;

        fn call<'a>(&'a self, args: Value, ctx: ToolCtx) -> BoxFuture<'a, ToolResult>;

        fn simulate<'a>(&'a self, _args: &'a Value, _ctx: ToolCtx) -> BoxFuture<'a, Option<EffectSet>> {
            Box::pin(async { None })
        }

        fn purity(&self) -> Purity {
            Purity::Impure
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Purity {
        Pure,
        PureFs,
        Impure,
    }

    #[derive(Default)]
    pub struct ToolRegistry {
        tools: RwLock<BTreeMap<String, Arc<dyn Tool>>>,
    }

    impl ToolRegistry {
        pub fn register<T: Tool + 'static>(&self, tool: T) {
            self.tools.write().insert(tool.spec().name.clone(), Arc::new(tool));
        }

        pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
            self.tools.read().get(name).cloned()
        }

        pub fn specs(&self) -> Vec<ToolSpec> {
            self.tools.read().values().map(|tool| tool.spec().clone()).collect()
        }
    }

    pub struct ToolDispatcher {
        registry: Arc<ToolRegistry>,
        policy: Arc<dyn PermissionEngine>,
    }

    impl ToolDispatcher {
        pub fn new(registry: Arc<ToolRegistry>, policy: Arc<dyn PermissionEngine>) -> Self {
            Self { registry, policy }
        }

        /// Whether the named tool is registered and declares itself read-only. Used to
        /// decide whether a model-emitted call may be auto-dispatched from a model step
        /// (read-only only) versus requiring an authorized plan step (any mutation).
        /// Unknown tools return `false` (conservative: not auto-dispatchable).
        pub fn is_read_only(&self, name: &str) -> bool {
            self.registry.get(name).map(|tool| tool.spec().annotations.read_only).unwrap_or(false)
        }

        pub async fn dispatch(&self, call: ToolCall) -> Result<ToolResult> {
            let call_id = call.call_id.clone();
            let tool =
                self.registry.get(&call.tool).ok_or_else(|| HideError::NotFound(format!("tool {}", call.tool)))?;
            let spec = tool.spec().clone();
            let predicted = tool
                .simulate(
                    &call.args,
                    ToolCtx {
                        grant_id: call.capability_grant_id.clone(),
                        deadline_ms: Some(spec.timeout_ms),
                        output_cap_bytes: spec.output_cap_bytes,
                    },
                )
                .await
                .unwrap_or_default();
            let target =
                predicted.effects.first().map(|effect| effect.target.clone()).unwrap_or_else(|| spec.name.clone());
            let request = PermissionRequest {
                capability_kind: spec.capabilities_required.first().cloned().unwrap_or_else(|| "tool.call".to_string()),
                target,
                risk: if spec.annotations.destructive { RiskLevel::High } else { RiskLevel::Low },
                effects: predicted.effects.clone(),
                grant: None,
            };
            let verdict = self.policy.evaluate(&request);
            if verdict.decision != Decision::Allow {
                return Err(HideError::PolicyDenied(verdict.reason));
            }
            if call.x.dry_run {
                return Ok(ToolResult {
                    call_id,
                    ok: true,
                    status: ToolStatus::Ok,
                    content: vec![ToolContent::Json { value: serde_json::to_value(&predicted)? }],
                    structured_content: Some(serde_json::json!({
                        "dry_run": true,
                        "tool": spec.name,
                        "predicted_effects": predicted,
                    })),
                    bytes_ref: None,
                    exit_code: None,
                    effects: predicted,
                    provenance: default_tool_provenance(),
                    stats: ToolStats { from_dry_run: true, ..ToolStats::default() },
                    error: None,
                });
            }
            let mut result = tool
                .call(
                    call.args,
                    ToolCtx {
                        grant_id: call.capability_grant_id,
                        deadline_ms: Some(spec.timeout_ms),
                        output_cap_bytes: spec.output_cap_bytes,
                    },
                )
                .await;
            result.call_id = call_id;
            Ok(result)
        }
    }
}
#[rustfmt::skip]
pub mod types {
    use crate::ids::BlobId;
    use serde::{Deserialize, Serialize};
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum TrustLevel {
        Trusted,
        UserAuthored,
        Workspace,
        ToolOutput,
        Network,
        Untrusted,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum RiskLevel {
        Trivial,
        Low,
        Medium,
        High,
        Critical,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum Decision {
        Allow,
        Ask,
        Deny,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct TextRange {
        pub start_line: u32,
        pub start_col: u32,
        pub end_line: u32,
        pub end_col: u32,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ByteRange {
        pub start: u64,
        pub end: u64,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct BlobRef {
        pub id: BlobId,
        pub hash: String,
        pub size_bytes: u64,
        pub media_type: Option<String>,
    }

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    pub struct Provenance {
        pub source: String,
        pub trust: TrustLevel,
        /// Confidence in this provenance's trust claim, 0.0..=1.0 (bible A.2/F12).
        /// Defaults to 1.0 for trusted/builtin sources.
        #[serde(default = "default_confidence")]
        pub confidence: f32,
        pub labels: Vec<String>,
        pub derived_from: Vec<String>,
    }

    fn default_confidence() -> f32 {
        1.0
    }

    impl Provenance {
        pub fn trusted(source: impl Into<String>) -> Self {
            Self {
                source: source.into(),
                trust: TrustLevel::Trusted,
                confidence: 1.0,
                labels: Vec::new(),
                derived_from: Vec::new(),
            }
        }

        /// Set the confidence in this provenance (builder).
        pub fn with_confidence(mut self, confidence: f32) -> Self {
            self.confidence = confidence;
            self
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct ResourceScope {
        pub kind: String,
        pub pattern: String,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(rename_all = "snake_case")]
    pub enum EffectKind {
        Read,
        Write,
        Delete,
        Execute,
        Network,
        Model,
        Plugin,
        Unknown,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct Effect {
        pub kind: EffectKind,
        pub target: String,
        pub bytes_hash: Option<String>,
        pub risk: RiskLevel,
        pub metadata: BTreeMap<String, String>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
    pub struct EffectSet {
        pub effects: Vec<Effect>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
    pub struct FileSpan {
        pub path: PathBuf,
        pub range: Option<TextRange>,
        pub content_hash: Option<String>,
    }
}

pub use error::{HideError, Result};
