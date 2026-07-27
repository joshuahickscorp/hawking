use hide_core::api::{UiEvent, UiEventKind};
use hide_core::event::{
    ErrorEvent, Event, ProjectionEvent, RuntimeStatusEvent, SecurityEvent, TokenEvent,
    ToolCallEvent, ToolResultEvent,
};
use hide_core::ids::{EventId, SessionId};
use hide_core::persistence::{DynEventLog, DynProjectionStore};
use hide_core::Result;
use hide_kernel::projection::{empty_projection, BasicProjectionEngine, ProjectionEngine};
use hide_kernel::session::SessionProjection;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Default bound on a transcript search result set (a client asks for more via
/// [`TranscriptQuery::limit`], capped by [`MAX_SEARCH_LIMIT`]).
pub const DEFAULT_SEARCH_LIMIT: usize = 50;
/// Hard ceiling on a transcript search result set: a search can never return an
/// unbounded page even if a caller asks for one.
pub const MAX_SEARCH_LIMIT: usize = 500;

/// A durable transcript/item search over the event log (bible sec 32-33).
///
/// LITERAL substring + STRUCTURED filters only. Semantic / embedding search is
/// `DEFERRED_MODEL_REQUIRED`: this path never loads a model, so it stays usable
/// headless and offline. All filters are AND-combined; an empty `text` matches
/// every item that passes the structured filters.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranscriptQuery {
    /// Literal (case-insensitive) substring over item text. Empty matches every
    /// item that passes the structured filters (still deterministic + bounded).
    #[serde(default)]
    pub text: String,
    /// Restrict to one session/thread. `None` searches every session.
    #[serde(default)]
    pub session_id: Option<SessionId>,
    /// Restrict to one exact event kind (e.g. `"tool.result"`).
    #[serde(default)]
    pub kind: Option<String>,
    /// Restrict to one item role (`"user"`, `"assistant"`, `"tool"`).
    #[serde(default)]
    pub role: Option<String>,
    /// Inclusive lower bound on the event wall-clock (micros since epoch).
    #[serde(default)]
    pub since_ts: Option<u64>,
    /// Inclusive upper bound on the event wall-clock (micros since epoch).
    #[serde(default)]
    pub until_ts: Option<u64>,
    /// Bounded result count. `None` or `0` falls back to [`DEFAULT_SEARCH_LIMIT`];
    /// any value is capped at [`MAX_SEARCH_LIMIT`].
    #[serde(default)]
    pub limit: Option<usize>,
}

impl TranscriptQuery {
    /// A bare literal query over every session (no structured filters).
    pub fn literal(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            ..Self::default()
        }
    }

    /// Scope the query to a single session/thread.
    pub fn in_session(mut self, session_id: SessionId) -> Self {
        self.session_id = Some(session_id);
        self
    }

    /// Restrict to a single exact event kind.
    pub fn with_kind(mut self, kind: impl Into<String>) -> Self {
        self.kind = Some(kind.into());
        self
    }

    /// Restrict to a single item role.
    pub fn with_role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }
}

/// One transcript search hit: enough to locate + render the item without a
/// re-scan (session + event id + kind + role + a bounded snippet).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptHit {
    pub session_id: SessionId,
    pub event_id: EventId,
    pub seq: u64,
    pub kind: String,
    pub role: Option<String>,
    pub snippet: String,
    pub ts: u64,
}

#[derive(Clone)]
pub struct BackendReplayService {
    events: DynEventLog,
    projections: DynProjectionStore,
    engine: Arc<dyn ProjectionEngine>,
}

impl BackendReplayService {
    pub fn new(events: DynEventLog, projections: DynProjectionStore) -> Self {
        Self {
            events,
            projections,
            engine: Arc::new(BasicProjectionEngine),
        }
    }

    pub fn with_engine(
        events: DynEventLog,
        projections: DynProjectionStore,
        engine: Arc<dyn ProjectionEngine>,
    ) -> Self {
        Self {
            events,
            projections,
            engine,
        }
    }

    pub async fn rebuild_session(&self, session_id: SessionId) -> Result<SessionProjection> {
        let events = self
            .events
            .scan(Some(session_id.clone()), None, None)
            .await?;
        let projection = self.engine.fold(empty_projection(session_id), &events)?;
        let seq = events.last().map_or(0, |event| event.seq);
        self.projections.put_projection(
            &projection.session_id,
            seq,
            serde_json::to_value(&projection)?,
        )?;
        Ok(projection)
    }

    /// Spine B: rebuild a session by folding the LIVE TAIL (events with
    /// `seq > after_seq`) on top of a pre-computed `summary` projection, instead
    /// of folding the whole log from empty. This is how a session resumes from a
    /// compacted summary: the cold prefix (archived via
    /// [`EventLog::compact_before`](hide_core::event::EventLog::compact_before))
    /// is represented by `summary`, and only the recent tail is replayed. The
    /// caller supplies the summary (built by the compaction/summary step); replay
    /// stays a pure fold, so this never loses determinism.
    pub async fn rebuild_with_summary(
        &self,
        session_id: SessionId,
        summary: SessionProjection,
        after_seq: u64,
    ) -> Result<SessionProjection> {
        let tail = self
            .events
            .scan(Some(session_id), Some(after_seq), None)
            .await?;
        self.engine.fold(summary, &tail)
    }

    pub async fn ui_events(
        &self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<UiEvent>> {
        let events = self.events.scan(session_id, after_seq, limit).await?;
        Ok(events.iter().flat_map(event_to_ui_events).collect())
    }

    /// Time-travel: rebuild a session's projection folding the log **up to and
    /// including** `seq` (scrub the timeline to that point). Events after `seq`
    /// are ignored — the returned projection is the state the session was in at
    /// that moment. Unlike [`Self::rebuild_session`], this does **not** persist
    /// over the live projection (a scrub is a read-only view into the past).
    pub async fn scrub_to_event(
        &self,
        session_id: SessionId,
        seq: u64,
    ) -> Result<SessionProjection> {
        let all = self
            .events
            .scan(Some(session_id.clone()), None, None)
            .await?;
        let prefix: Vec<_> = all.into_iter().filter(|e| e.seq <= seq).collect();
        self.engine.fold(empty_projection(session_id), &prefix)
    }

    /// Resolve a session's prefix by `EventId` (the Wire-A `ScrubToEvent` carries
    /// an id, not a seq). Returns the seq the id maps to, or `NotFound`.
    pub async fn seq_of_event(
        &self,
        session_id: SessionId,
        event_id: &hide_core::ids::EventId,
    ) -> Result<u64> {
        let all = self.events.scan(Some(session_id), None, None).await?;
        all.iter()
            .find(|e| &e.id == event_id)
            .map(|e| e.seq)
            .ok_or_else(|| {
                hide_core::error::HideError::NotFound(format!("event {event_id} not in session"))
            })
    }

    /// The latest `seq` in a session (0 when the session is empty). Used to fork
    /// the WHOLE session (its current tail) when no explicit boundary is given.
    pub async fn latest_seq(&self, session_id: SessionId) -> Result<u64> {
        let events = self.events.scan(Some(session_id), None, None).await?;
        Ok(events.last().map_or(0, |event| event.seq))
    }

    /// Time-travel: **fork** a new session seeded from `from`'s log prefix up to
    /// and including `at_seq`. Every prefix event is re-appended under a fresh
    /// `SessionId` (preserving order/kind/payload), then the new session's
    /// projection is built + persisted. The original session is untouched — the
    /// fork is a genuine branch (the bible's "explore an alternative from here").
    /// Returns the new session id + its projection.
    pub async fn fork_session(
        &self,
        from: SessionId,
        at_seq: u64,
    ) -> Result<(SessionId, SessionProjection)> {
        let prefix: Vec<_> = self
            .events
            .scan(Some(from.clone()), None, None)
            .await?
            .into_iter()
            .filter(|e| e.seq <= at_seq)
            .collect();
        let new_session = SessionId::new();
        // Re-append each prefix event under the new session. We carry the kind +
        // payload + class; ids/seq/chain are reassigned by the log (the fork is a
        // new lineage, not a copy of the old chain).
        for event in &prefix {
            let mut new_event = hide_core::event::NewEvent::of(
                new_session.clone(),
                event.source.clone(),
                &event.kind,
                event.payload.clone(),
            );
            new_event.class = event.class;
            new_event.run_id = event.run_id.clone();
            new_event.actor = event.actor.clone();
            self.events.append(new_event).await?;
        }
        let forked = self.rebuild_session(new_session.clone()).await?;
        Ok((new_session, forked))
    }

    /// Seed a fresh, independent child session from an EXPLICIT ordered event list
    /// (the rewind/replay/fork substrate). An optional `marker` is appended FIRST
    /// (its `session_id` is rebound to the new session), then each `inherited`
    /// event is re-appended preserving kind/payload/class/run/actor/source (ids and
    /// seq are reassigned by the log, a new lineage). Returns the new session id +
    /// its rebuilt projection. Unlike [`Self::fork_session`], the caller chooses the
    /// exact events (so a domain-scoped rewind can drop a subset) and the leading
    /// [`crate::rewind::ForkPoint`] marker distinguishes inherited context from the
    /// child's own records.
    pub async fn seed_child_session(
        &self,
        marker: Option<hide_core::event::NewEvent>,
        inherited: &[&Event],
    ) -> Result<(SessionId, SessionProjection)> {
        let new_session = SessionId::new();
        if let Some(mut m) = marker {
            m.session_id = new_session.clone();
            self.events.append(m).await?;
        }
        for event in inherited {
            let mut new_event = hide_core::event::NewEvent::of(
                new_session.clone(),
                event.source.clone(),
                &event.kind,
                event.payload.clone(),
            );
            new_event.class = event.class;
            new_event.run_id = event.run_id.clone();
            new_event.actor = event.actor.clone();
            self.events.append(new_event).await?;
        }
        let projection = self.rebuild_session(new_session.clone()).await?;
        Ok((new_session, projection))
    }

    /// Search the durable transcript (bible sec 32-33) for items (user/assistant
    /// messages, tool results) matching a LITERAL substring + STRUCTURED filters
    /// ([`TranscriptQuery`]). Results are ordered deterministically by `seq`
    /// (ascending, the log's total order) and bounded by `query.limit`. Each hit
    /// carries session_id + event id + kind + a bounded snippet so a client can
    /// jump straight to the item. No model, no embeddings: semantic search is
    /// `DEFERRED_MODEL_REQUIRED`.
    pub async fn search_transcript(&self, query: &TranscriptQuery) -> Result<Vec<TranscriptHit>> {
        let limit = match query.limit {
            Some(n) if n > 0 => n.min(MAX_SEARCH_LIMIT),
            _ => DEFAULT_SEARCH_LIMIT,
        };
        // `scan` returns events already ordered by `seq`. Scoping to a session at
        // the log is cheaper than post-filtering; a cross-session search passes
        // `None`. The resulting order (and thus the ranking) is deterministic.
        let events = self
            .events
            .scan(query.session_id.clone(), None, None)
            .await?;
        let needle = query.text.to_lowercase();
        let mut hits = Vec::new();
        for event in &events {
            // Structured filters first (cheap), then the item-text match.
            if let Some(kind) = &query.kind {
                if &event.kind != kind {
                    continue;
                }
            }
            if query.since_ts.is_some_and(|since| event.ts < since) {
                continue;
            }
            if query.until_ts.is_some_and(|until| event.ts > until) {
                continue;
            }
            let Some((role, text)) = extract_item(event) else {
                continue;
            };
            if let Some(want) = &query.role {
                if role.as_deref() != Some(want.as_str()) {
                    continue;
                }
            }
            // Literal, case-insensitive substring. An empty needle matches every
            // item (so an empty query + a structured filter still returns items).
            let match_at = if needle.is_empty() {
                Some(0)
            } else {
                text.to_lowercase().find(&needle)
            };
            let Some(at) = match_at else {
                continue;
            };
            hits.push(TranscriptHit {
                session_id: event.session_id.clone(),
                event_id: event.id.clone(),
                seq: event.seq,
                kind: event.kind.clone(),
                role,
                snippet: snippet_around(&text, at, needle.len()),
                ts: event.ts,
            });
            if hits.len() >= limit {
                break;
            }
        }
        Ok(hits)
    }
}

/// Extract a transcript item's `(role, text)` from a text-bearing event. Returns
/// `None` for events that carry no user/assistant/tool item text (they are
/// skipped by search). Kinds covered: `user.intent.submit_turn` (the user's
/// text), `agent.message` (assistant/other), `tool.result` (summary + structured
/// output), and streamed `token`/`token_batch` text.
fn extract_item(event: &hide_core::event::Event) -> Option<(Option<String>, String)> {
    match event.kind.as_str() {
        "user.intent.submit_turn" => {
            let text = event
                .payload
                .get("args")
                .and_then(|a| a.get("text"))
                .and_then(|t| t.as_str())?;
            non_empty(text).map(|t| (Some("user".to_string()), t))
        }
        "agent.message" => {
            let role = event
                .payload
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("assistant")
                .to_string();
            let text = event.payload.get("text").and_then(|t| t.as_str())?;
            non_empty(text).map(|t| (Some(role), t))
        }
        "tool.result" => {
            let summary = event
                .payload
                .get("summary")
                .and_then(|t| t.as_str())
                .unwrap_or("");
            // Include the structured output text too, so a search can find tool
            // output, not only the one-line summary.
            let output = event
                .payload
                .get("output")
                .filter(|o| !o.is_null())
                .map(|o| o.to_string())
                .unwrap_or_default();
            let text = if output.is_empty() {
                summary.to_string()
            } else {
                format!("{summary} {output}")
            };
            non_empty(&text).map(|t| (Some("tool".to_string()), t))
        }
        "token" | "token_batch" => {
            let text = event.payload.get("text").and_then(|t| t.as_str())?;
            non_empty(text).map(|t| (Some("assistant".to_string()), t))
        }
        // A side chat merged its typed summary back onto THIS (parent) session:
        // the cited summary is a searchable transcript item (role `side_chat`) so
        // a parent-scoped search surfaces it. The side chat id lives in the
        // payload (`side_chat`) for a client to jump back to the source thread.
        "session.merge_summary" => {
            let summary = event.payload.get("summary").and_then(|t| t.as_str())?;
            non_empty(summary).map(|t| (Some("side_chat".to_string()), t))
        }
        _ => None,
    }
}

/// The item text, or `None` when it is blank (blank items are not searchable).
/// Returns the ORIGINAL (untrimmed) text so snippets keep their spacing.
fn non_empty(s: &str) -> Option<String> {
    if s.trim().is_empty() {
        None
    } else {
        Some(s.to_string())
    }
}

/// Characters of surrounding context to keep on each side of a match.
const SNIPPET_CONTEXT: usize = 64;

/// Build a bounded snippet around a match at byte offset `at` (into `text`),
/// with `...` elision markers when the window is clipped. Char-boundary safe:
/// offsets are snapped to boundaries and clamped, so this never panics even if a
/// case-folded offset lands mid-codepoint. An empty match (`at == 0`,
/// `match_len == 0`, the empty-query path) yields the item's leading window.
fn snippet_around(text: &str, at: usize, match_len: usize) -> String {
    let start = floor_boundary(text, at.saturating_sub(SNIPPET_CONTEXT));
    let end = ceil_boundary(text, at.saturating_add(match_len).saturating_add(SNIPPET_CONTEXT));
    let mut out = String::new();
    if start > 0 {
        out.push_str("...");
    }
    out.push_str(text[start..end].trim());
    if end < text.len() {
        out.push_str("...");
    }
    out
}

/// Largest char boundary `<= i` (clamped to `text.len()`).
fn floor_boundary(text: &str, mut i: usize) -> usize {
    if i >= text.len() {
        return text.len();
    }
    while i > 0 && !text.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest char boundary `>= i` (clamped to `text.len()`).
fn ceil_boundary(text: &str, mut i: usize) -> usize {
    if i >= text.len() {
        return text.len();
    }
    while i < text.len() && !text.is_char_boundary(i) {
        i += 1;
    }
    i
}

/// The UiEvents one durable event replays as: exactly one, EXCEPT a durable `diff.*`
/// event, which replays as both projections the diff-review surface reads. The catch-up
/// GET runs on every socket reconnect, so a hunk decided during a disconnect window has
/// to arrive in the shape the live publisher uses ([`crate::host::diff_projections`]) or
/// the review would sit on a status the host no longer holds.
pub fn event_to_ui_events(event: &Event) -> Vec<UiEvent> {
    let Some(proposal) = diff_event_proposal(event) else {
        return vec![event_to_ui_event(event)];
    };
    let (diff, chips) = crate::host::diff_projections(&proposal);
    [("diff", diff), ("diff_chip", chips)]
        .into_iter()
        .map(|(projection, patch)| UiEvent {
            seq: event.seq,
            session_id: Some(event.session_id.clone()),
            kind: UiEventKind::ProjectionPatch {
                projection: projection.to_string(),
                patch,
            },
        })
        .collect()
}

/// The proposal a durable `diff.*` event carries: the whole payload for `diff.proposed`,
/// the nested `proposal` for the per-hunk status events. Anything else (the sealed
/// `diff.receipt`) reads as None and replays through the generic mapper.
fn diff_event_proposal(event: &Event) -> Option<crate::host::DiffProposal> {
    if !event.kind.starts_with("diff.") {
        return None;
    }
    let body = match event.kind.as_str() {
        "diff.proposed" => &event.payload,
        _ => event.payload.get("proposal")?,
    };
    serde_json::from_value(body.clone()).ok()
}

/// One replayed transcript line, in the shape the store's EventRouter appends. `event_id` is the
/// dedupe key: the catch-up GET and `open_session` both replay lines a live socket may already have
/// delivered, and a conversation must not double.
fn transcript_message(event: &Event, role: &str, text: &str) -> UiEventKind {
    UiEventKind::Custom(serde_json::json!({
        "kind": "transcript_message",
        "event_id": event.id,
        "role": role,
        "text": text,
    }))
}

pub fn event_to_ui_event(event: &Event) -> UiEvent {
    // The kernel never ships internal-only events; UI events are a
    // projection-flavored subset keyed off the dotted event kind, reading the
    // typed view off the open `Value` payload.
    let kind = match event.kind.as_str() {
        "projection.patch" => {
            event
                .payload_as::<ProjectionEvent>()
                .map(|projection| UiEventKind::ProjectionPatch {
                    projection: projection.projection,
                    patch: projection.patch,
                })
        }
        "token" | "token_batch" => {
            event
                .payload_as::<TokenEvent>()
                .map(|token| UiEventKind::TokenBatch {
                    stream_id: token.stream_id,
                    text: token.text,
                })
        }
        "runtime.status" => {
            event
                .payload_as::<RuntimeStatusEvent>()
                .map(|status| UiEventKind::RuntimeStatus {
                    status: status.status,
                    detail: status.detail,
                })
        }
        "tool.call" => event
            .payload_as::<ToolCallEvent>()
            .map(|call| UiEventKind::ToolProgress {
                call_id: call.call_id.as_str().to_string(),
                message: format!("started {}", call.tool_name),
                event_id: Some(event.id.as_str().to_string()),
            }),
        "tool.result" => {
            let event_id = event.id.as_str().to_string();
            event
                .payload_as::<ToolResultEvent>()
                .map(|result| UiEventKind::ToolProgress {
                    call_id: result.call_id.as_str().to_string(),
                    message: if result.ok {
                        result.summary
                    } else {
                        format!("failed: {}", result.summary)
                    },
                    event_id: Some(event_id),
                })
        }
        "security.gate" => {
            event
                .payload_as::<SecurityEvent>()
                .map(|security| UiEventKind::SecurityGate {
                    gate: security.gate,
                    message: security.detail,
                })
        }
        "error" => event
            .payload_as::<ErrorEvent>()
            .map(|error| UiEventKind::Error {
                code: error.code,
                message: error.message,
            }),
        // A paused effectful step awaiting approval. The catch-up GET runs on EVERY connect, so the
        // replayed shape has to be the one the frontend routes on: the live publish carries the
        // undotted `kind: "approval_requested"` with run_id/step_id at the top level. Falling
        // through to the generic Custom arm below would hand the client the dotted event kind and a
        // nested payload, which routes nowhere, so a paused turn stayed invisible after a reconnect.
        "approval.requested" => {
            let mut payload = event.payload.clone();
            if let Some(obj) = payload.as_object_mut() {
                obj.insert(
                    "kind".to_string(),
                    serde_json::Value::String("approval_requested".to_string()),
                );
                Some(UiEventKind::Custom(payload))
            } else {
                None
            }
        }
        // THE TRANSCRIPT. The durable log holds a conversation as a `user.intent.submit_turn` (the
        // user's line, under `args.text`) and an assistant `agent.message` (the model's) - exactly
        // the pair `host::rebuild_history` reads. Neither replayed as anything a client could
        // render, so `open_session` republished a session that showed up as truncated JSON in the
        // status bar and a reload showed an empty conversation while the log held the whole thing.
        // Both map to the ONE `transcript_message` shape the store's EventRouter appends, carrying
        // `event_id` so a replay of a line already on screen is dropped rather than doubled.
        "user.intent.submit_turn" => event
            .payload
            .get("args")
            .and_then(|a| a.get("text"))
            .and_then(|t| t.as_str())
            .filter(|t| !t.trim().is_empty())
            .map(|text| transcript_message(event, "user", text)),
        "agent.message" => {
            let role = match event.payload.get("role").and_then(|r| r.as_str()) {
                Some("user") => "user",
                _ => "assistant",
            };
            event
                .payload
                .get("text")
                .and_then(|t| t.as_str())
                .filter(|t| !t.trim().is_empty())
                .map(|text| transcript_message(event, role, text))
        }
        // A sealed checkpoint. The live publish is bus-only, so every checkpoint-addressed history
        // verb died on a reload while the record was still on disk; `host::checkpoint_create` now
        // records this durably and it replays in the shape `store.ts` reads the id from.
        "checkpoint.created" => Some(UiEventKind::Custom(serde_json::json!({
            "kind": "checkpoint_created",
            "record": event.payload,
        }))),
        _ => None,
    };
    UiEvent {
        seq: event.seq,
        session_id: Some(event.session_id.clone()),
        kind: kind.unwrap_or_else(|| {
            UiEventKind::Custom(serde_json::json!({
                "event_id": event.id,
                "kind": event.kind,
                "source": event.source,
                "payload": event.payload,
            }))
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{
        AgentStateEvent, EventLog, InMemoryEventLog, NewEvent, ToolResultEvent,
    };
    use hide_core::ids::{RunId, SessionId, ToolCallId};
    use hide_core::persistence::{InMemoryProjectionStore, ProjectionStore};

    #[tokio::test]
    async fn replay_rebuilds_and_persists_session_projection() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        events
            .append(NewEvent::agent_state(
                session.clone(),
                RunId::new(),
                AgentStateEvent {
                    phase: "plan".to_string(),
                    detail: "building plan".to_string(),
                },
            ))
            .await
            .unwrap();

        let replay = BackendReplayService::new(events, projections.clone());
        let projection = replay.rebuild_session(session.clone()).await.unwrap();

        assert!(projection
            .transcript
            .iter()
            .any(|line| line.contains("building plan")));
        assert!(
            projections.latest_projection(&session).unwrap().unwrap().1["transcript"]
                .as_array()
                .unwrap()
                .len()
                == 1
        );
    }

    #[tokio::test]
    async fn replay_maps_tool_results_to_ui_events() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        let call_id = ToolCallId::new();
        events
            .append(NewEvent::tool_result(
                session.clone(),
                ToolResultEvent {
                    call_id: call_id.clone(),
                    ok: true,
                    summary: "done".to_string(),
                    output: None,
                    bytes_ref: None,
                },
            ))
            .await
            .unwrap();

        let replay = BackendReplayService::new(events, projections);
        let ui_events = replay.ui_events(Some(session), None, None).await.unwrap();

        assert_eq!(ui_events.len(), 1);
        assert!(matches!(
            &ui_events[0].kind,
            UiEventKind::ToolProgress { call_id: id, message, .. }
                if id == call_id.as_str() && message == "done"
        ));
    }

    /// The timeline addresses a boundary with the id a step CARRIES, so that id has to be the kind
    /// `seq_of_event` resolves. It used to be handed a `ToolCallId`, which resolves to `NotFound`,
    /// which is why `fork_session` and `checkpoint_create` always failed on a live host.
    #[tokio::test]
    async fn tool_progress_carries_the_event_id_the_boundary_resolver_accepts() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        let call_id = ToolCallId::new();
        events
            .append(NewEvent::tool_result(
                session.clone(),
                ToolResultEvent {
                    call_id: call_id.clone(),
                    ok: true,
                    summary: "done".to_string(),
                    output: None,
                    bytes_ref: None,
                },
            ))
            .await
            .unwrap();

        let replay = BackendReplayService::new(events, projections);
        let ui = replay
            .ui_events(Some(session.clone()), None, None)
            .await
            .unwrap();
        let UiEventKind::ToolProgress { event_id, .. } = &ui[0].kind else {
            panic!("expected tool progress");
        };
        let event_id = event_id.as_deref().expect("a recorded step carries its event id");
        assert_eq!(
            replay
                .seq_of_event(session.clone(), &hide_core::ids::EventId::from(event_id))
                .await
                .unwrap(),
            ui[0].seq
        );
        // The tool call id is NOT resolvable, which is the defect this guards.
        assert!(replay
            .seq_of_event(session, &hide_core::ids::EventId::from(call_id.as_str()))
            .await
            .is_err());
    }

    async fn seed_three_phases(events: &Arc<InMemoryEventLog>, session: &SessionId) -> RunId {
        let run = RunId::new();
        for phase in ["plan", "act", "verify"] {
            events
                .append(NewEvent::agent_state(
                    session.clone(),
                    run.clone(),
                    AgentStateEvent {
                        phase: phase.to_string(),
                        detail: format!("entered {phase}"),
                    },
                ))
                .await
                .unwrap();
        }
        run
    }

    #[tokio::test]
    async fn scrub_to_event_rebuilds_prefix_only() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        seed_three_phases(&events, &session).await;
        let replay = BackendReplayService::new(events.clone(), projections);

        // Full rebuild sees all three phase lines.
        let full = replay.rebuild_session(session.clone()).await.unwrap();
        assert_eq!(full.transcript.len(), 3);

        // Scrub to seq 2 sees only the first two.
        let scrubbed = replay.scrub_to_event(session.clone(), 2).await.unwrap();
        assert_eq!(scrubbed.transcript.len(), 2);
        assert!(scrubbed.transcript.iter().any(|l| l.contains("act")));
        assert!(!scrubbed.transcript.iter().any(|l| l.contains("verify")));
    }

    #[tokio::test]
    async fn fork_session_branches_a_new_lineage_from_prefix() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        seed_three_phases(&events, &session).await;
        let replay = BackendReplayService::new(events.clone(), projections);

        // Fork at seq 2: the new session carries the first two events only.
        let (forked_id, forked) = replay.fork_session(session.clone(), 2).await.unwrap();
        assert_ne!(forked_id, session);
        assert_eq!(forked.transcript.len(), 2);

        // The original session is untouched (still 3 events).
        let original = events
            .scan(Some(session.clone()), None, None)
            .await
            .unwrap();
        assert_eq!(original.len(), 3);
        // The fork is a separate lineage of 2 events under the new session id.
        let branch = events.scan(Some(forked_id), None, None).await.unwrap();
        assert_eq!(branch.len(), 2);
    }

    #[tokio::test]
    async fn fork_is_independent_appending_to_fork_does_not_touch_source() {
        // Pin the core fork guarantee: after a fork, appending to the fork's
        // lineage never appears in the source (and vice versa). The source stays
        // exactly as it was; the fork carries the prefix + its own new event.
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        seed_three_phases(&events, &session).await;
        let replay = BackendReplayService::new(events.clone(), projections);

        let (forked_id, _) = replay.fork_session(session.clone(), 2).await.unwrap();
        // Append a NEW event to the fork only.
        events
            .append(NewEvent::system(
                forked_id.clone(),
                "agent.message",
                serde_json::json!({ "role": "assistant", "text": "fork-only note" }),
            ))
            .await
            .unwrap();

        // Source is untouched: still exactly its 3 original events, none carrying
        // the fork-only note.
        let source = events.scan(Some(session), None, None).await.unwrap();
        assert_eq!(source.len(), 3, "source is unchanged by a fork append");
        assert!(
            !source.iter().any(|e| e.kind == "agent.message"),
            "the fork-only append must not leak into the source"
        );
        // Fork now has the 2 prefix events + its own append = 3, independent.
        let fork = events.scan(Some(forked_id), None, None).await.unwrap();
        assert_eq!(fork.len(), 3, "the fork carries prefix + its own append");
        assert!(fork.iter().any(|e| e.kind == "agent.message"));
    }

    /// Seed two sessions with distinct tokens (ZZALPHA / ZZBETA) across item
    /// kinds (submit_turn, agent.message, tool.result). Returns `(session_a,
    /// session_b, appended_events)` so tests can assert exact event ids.
    async fn seed_search_corpus(
        events: &Arc<InMemoryEventLog>,
    ) -> (SessionId, SessionId, Vec<Event>) {
        let a = SessionId::new();
        let b = SessionId::new();
        let mut appended = Vec::new();
        // Session A: user + assistant + tool, all carrying ZZALPHA.
        appended.push(
            events
                .append(NewEvent::system(
                    a.clone(),
                    "user.intent.submit_turn",
                    serde_json::json!({
                        "intent": "submit_turn",
                        "args": { "text": "please ZZALPHA the widget" }
                    }),
                ))
                .await
                .unwrap(),
        );
        appended.push(
            events
                .append(NewEvent::system(
                    a.clone(),
                    "agent.message",
                    serde_json::json!({ "role": "assistant", "text": "done with ZZALPHA task" }),
                ))
                .await
                .unwrap(),
        );
        appended.push(
            events
                .append(NewEvent::tool_result(
                    a.clone(),
                    ToolResultEvent {
                        call_id: ToolCallId::new(),
                        ok: true,
                        summary: "ran the ZZALPHA tool".to_string(),
                        output: None,
                        bytes_ref: None,
                    },
                ))
                .await
                .unwrap(),
        );
        // Session B: user + tool, all carrying ZZBETA.
        appended.push(
            events
                .append(NewEvent::system(
                    b.clone(),
                    "user.intent.submit_turn",
                    serde_json::json!({
                        "intent": "submit_turn",
                        "args": { "text": "handle ZZBETA now" }
                    }),
                ))
                .await
                .unwrap(),
        );
        appended.push(
            events
                .append(NewEvent::tool_result(
                    b.clone(),
                    ToolResultEvent {
                        call_id: ToolCallId::new(),
                        ok: true,
                        summary: "ZZBETA output here".to_string(),
                        output: None,
                        bytes_ref: None,
                    },
                ))
                .await
                .unwrap(),
        );
        (a, b, appended)
    }

    #[tokio::test]
    async fn search_literal_finds_only_matching_items_with_session_event_snippet() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let (a, _b, seeded) = seed_search_corpus(&events).await;
        let replay = BackendReplayService::new(events, projections);

        let hits = replay
            .search_transcript(&TranscriptQuery::literal("ZZALPHA"))
            .await
            .unwrap();
        // Exactly the three session-A items carry ZZALPHA; no session-B item does.
        assert_eq!(hits.len(), 3, "only the ZZALPHA items match");
        assert!(
            hits.iter().all(|h| h.session_id == a),
            "every hit is in session A"
        );
        // Deterministic order: ascending by seq (the log's total order).
        assert!(
            hits.windows(2).all(|w| w[0].seq < w[1].seq),
            "hits ranked by ascending seq"
        );
        // Each hit carries the right event id + kind + a snippet with the token.
        let seeded_a: Vec<&Event> = seeded.iter().filter(|e| e.session_id == a).collect();
        for (hit, ev) in hits.iter().zip(seeded_a.iter()) {
            assert_eq!(hit.event_id, ev.id, "hit carries the source event id");
            assert_eq!(hit.kind, ev.kind);
            assert!(
                hit.snippet.contains("ZZALPHA"),
                "snippet quotes the match: {}",
                hit.snippet
            );
        }
        // Roles are surfaced (user / assistant / tool across the three items).
        let roles: Vec<Option<&str>> = hits.iter().map(|h| h.role.as_deref()).collect();
        assert_eq!(
            roles,
            vec![Some("user"), Some("assistant"), Some("tool")],
            "each item's role is surfaced in seq order"
        );
    }

    #[tokio::test]
    async fn search_kind_and_session_filters_narrow_and_scope() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let (a, b, _seeded) = seed_search_corpus(&events).await;
        let replay = BackendReplayService::new(events, projections);

        // Kind filter narrows: ZZALPHA + tool.result -> just the one tool result.
        let tool_hits = replay
            .search_transcript(&TranscriptQuery::literal("ZZALPHA").with_kind("tool.result"))
            .await
            .unwrap();
        assert_eq!(tool_hits.len(), 1, "kind filter narrows to the tool result");
        assert_eq!(tool_hits[0].kind, "tool.result");
        assert_eq!(tool_hits[0].role.as_deref(), Some("tool"));

        // Session filter scopes: ZZBETA lives only in session B.
        let b_hits = replay
            .search_transcript(&TranscriptQuery::literal("ZZBETA").in_session(b.clone()))
            .await
            .unwrap();
        assert_eq!(b_hits.len(), 2, "both ZZBETA items are in session B");
        assert!(b_hits.iter().all(|h| h.session_id == b));
        // ZZBETA scoped to session A finds nothing (scope is a hard boundary).
        let none = replay
            .search_transcript(&TranscriptQuery::literal("ZZBETA").in_session(a))
            .await
            .unwrap();
        assert!(none.is_empty(), "ZZBETA is absent from session A");

        // Role filter: only user items across all sessions (one per session).
        let user_hits = replay
            .search_transcript(&TranscriptQuery {
                role: Some("user".to_string()),
                ..TranscriptQuery::default()
            })
            .await
            .unwrap();
        assert_eq!(user_hits.len(), 2, "one user item per session");
        assert!(user_hits.iter().all(|h| h.role.as_deref() == Some("user")));
    }

    #[tokio::test]
    async fn search_empty_query_with_kind_filter_and_bound_are_deterministic() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let (_a, _b, _seeded) = seed_search_corpus(&events).await;
        let replay = BackendReplayService::new(events, projections);

        // Empty query + kind filter still works: every tool.result item (both
        // sessions), no substring required.
        let all_tools = replay
            .search_transcript(&TranscriptQuery::default().with_kind("tool.result"))
            .await
            .unwrap();
        assert_eq!(all_tools.len(), 2, "empty query returns all tool results");
        assert!(all_tools.iter().all(|h| h.kind == "tool.result"));
        assert!(
            all_tools.iter().all(|h| !h.snippet.is_empty()),
            "an empty-query hit still carries a leading snippet"
        );

        // Bounded: a broad token ('ZZ' matches every seeded item) capped at 1.
        let bounded = replay
            .search_transcript(&TranscriptQuery {
                text: "ZZ".to_string(),
                limit: Some(1),
                ..TranscriptQuery::default()
            })
            .await
            .unwrap();
        assert_eq!(bounded.len(), 1, "limit bounds the result set");

        // Deterministic: the same query twice yields identical results.
        let q = TranscriptQuery::literal("ZZ");
        let first = replay.search_transcript(&q).await.unwrap();
        let second = replay.search_transcript(&q).await.unwrap();
        assert_eq!(first, second, "search is deterministic across runs");
        assert_eq!(first.len(), 5, "'ZZ' matches all five seeded items");
    }

    /// The catch-up GET runs on every connect, so a replayed approval request must arrive in the
    /// SAME shape the live bus publishes: undotted `kind`, run_id/step_id at the top level. The
    /// regression this locks: the dotted `approval.requested` fell through to the generic Custom
    /// arm, the frontend router matched nothing, and a paused turn stayed invisible after a drop.
    #[tokio::test]
    async fn replayed_approval_request_matches_the_live_shape() {
        let events = Arc::new(InMemoryEventLog::new());
        let session = SessionId::new();
        let record = events
            .append(NewEvent::system(
                session.clone(),
                "approval.requested",
                serde_json::json!({
                    "run_id": "run-1",
                    "step_id": "step-1",
                    "summary": "write a file",
                    "effects": ["write_fs"],
                }),
            ))
            .await
            .unwrap();
        let ui = event_to_ui_event(&record);
        let UiEventKind::Custom(v) = ui.kind else {
            panic!("an approval request replays as a Custom UiEvent");
        };
        assert_eq!(v["kind"], "approval_requested", "undotted, the router switches on this");
        assert_eq!(v["run_id"], "run-1", "run_id is top level, not nested in payload");
        assert_eq!(v["step_id"], "step-1");
        assert_eq!(v["summary"], "write a file");
    }
}
