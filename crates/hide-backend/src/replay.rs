use hide_core::api::{UiEvent, UiEventKind};
use hide_core::event::{Event, EventPayload};
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
}

pub fn event_to_ui_event(event: &Event) -> UiEvent {
    UiEvent {
        seq: event.seq,
        session_id: Some(event.session_id.clone()),
        kind: match &event.payload {
            EventPayload::Projection(projection) => UiEventKind::ProjectionPatch {
                projection: projection.projection.clone(),
                patch: projection.patch.clone(),
            },
            EventPayload::RuntimeToken(token) => UiEventKind::TokenBatch {
                stream_id: token.stream_id.clone(),
                text: token.text.clone(),
            },
            EventPayload::RuntimeStatus(status) => UiEventKind::RuntimeStatus {
                status: status.status.clone(),
                detail: status.detail.clone(),
            },
            EventPayload::ToolCall(call) => UiEventKind::ToolProgress {
                call_id: call.call_id.as_str().to_string(),
                message: format!("started {}", call.tool_name),
            },
            EventPayload::ToolResult(result) => UiEventKind::ToolProgress {
                call_id: result.call_id.as_str().to_string(),
                message: if result.ok {
                    result.summary.clone()
                } else {
                    format!("failed: {}", result.summary)
                },
            },
            EventPayload::Security(security) => UiEventKind::SecurityGate {
                gate: security.gate.clone(),
                message: security.detail.clone(),
            },
            EventPayload::Error(error) => UiEventKind::Error {
                code: error.code.clone(),
                message: error.message.clone(),
            },
            other => UiEventKind::Custom(serde_json::json!({
                "event_id": event.id,
                "kind": event.kind.as_str(),
                "source": event.source,
                "payload": other,
            })),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{
        AgentStateEvent, EventLog, EventPayload, EventSource, InMemoryEventLog, NewEvent,
        ToolResultEvent,
    };
    use hide_core::ids::{SessionId, ToolCallId};
    use hide_core::persistence::{InMemoryProjectionStore, ProjectionStore};

    #[tokio::test]
    async fn replay_rebuilds_and_persists_session_projection() {
        let events = Arc::new(InMemoryEventLog::new());
        let projections = Arc::new(InMemoryProjectionStore::default());
        let session = SessionId::new();
        events
            .append(NewEvent {
                session_id: session.clone(),
                run_id: None,
                parent: None,
                source: EventSource::Agent,
                kind: "agent.state".into(),
                payload: EventPayload::AgentState(AgentStateEvent {
                    phase: "plan".to_string(),
                    detail: "building plan".to_string(),
                }),
                redactions: Vec::new(),
            })
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
            .append(NewEvent {
                session_id: session.clone(),
                run_id: None,
                parent: None,
                source: EventSource::Tool,
                kind: "tool.result".into(),
                payload: EventPayload::ToolResult(ToolResultEvent {
                    call_id: call_id.clone(),
                    ok: true,
                    summary: "done".to_string(),
                    output: None,
                    bytes_ref: None,
                }),
                redactions: Vec::new(),
            })
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
}
