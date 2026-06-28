use hide_core::api::{UiEvent, UiEventKind};
use hide_core::event::{
    Event, ErrorEvent, ProjectionEvent, RuntimeStatusEvent, SecurityEvent, TokenEvent,
    ToolCallEvent, ToolResultEvent,
};
use hide_core::ids::SessionId;
use hide_core::persistence::{DynEventLog, DynProjectionStore};
use hide_core::Result;
use hide_kernel::projection::{empty_projection, BasicProjectionEngine, ProjectionEngine};
use hide_kernel::session::SessionProjection;
use std::sync::Arc;

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

    pub async fn ui_events(
        &self,
        session_id: Option<SessionId>,
        after_seq: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<UiEvent>> {
        let events = self.events.scan(session_id, after_seq, limit).await?;
        Ok(events.iter().map(event_to_ui_event).collect())
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
}

pub fn event_to_ui_event(event: &Event) -> UiEvent {
    // The kernel never ships internal-only events; UI events are a
    // projection-flavored subset keyed off the dotted event kind, reading the
    // typed view off the open `Value` payload.
    let kind = match event.kind.as_str() {
        "projection.patch" => event
            .payload_as::<ProjectionEvent>()
            .map(|projection| UiEventKind::ProjectionPatch {
                projection: projection.projection,
                patch: projection.patch,
            }),
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
            }),
        "tool.result" => {
            event
                .payload_as::<ToolResultEvent>()
                .map(|result| UiEventKind::ToolProgress {
                    call_id: result.call_id.as_str().to_string(),
                    message: if result.ok {
                        result.summary
                    } else {
                        format!("failed: {}", result.summary)
                    },
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
            UiEventKind::ToolProgress { call_id: id, message }
                if id == call_id.as_str() && message == "done"
        ));
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
}
