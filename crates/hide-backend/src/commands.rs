use hide_core::api::{Intent, IntentAck};
use hide_core::event::{NewEvent, UserIntentEvent};
use hide_core::ids::SessionId;
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use serde_json::json;

pub struct CommandRouter {
    events: DynEventLog,
    control_session: SessionId,
}

impl CommandRouter {
    pub fn new(events: DynEventLog) -> Self {
        Self {
            events,
            control_session: SessionId::new(),
        }
    }

    pub fn with_control_session(events: DynEventLog, control_session: SessionId) -> Self {
        Self {
            events,
            control_session,
        }
    }

    pub fn control_session(&self) -> &SessionId {
        &self.control_session
    }

    pub async fn handle(&self, intent: Intent) -> Result<IntentAck> {
        let (session_id, intent_name, args) = match intent {
            Intent::SubmitTurn {
                session_id,
                text,
                attachments,
            } => (
                session_id,
                "submit_turn".to_string(),
                json!({ "text": text, "attachments": attachments }),
            ),
            Intent::CancelRun { run_id } => (
                self.control_session.clone(),
                "cancel_run".to_string(),
                json!({ "run_id": run_id }),
            ),
            Intent::PauseRun { run_id } => (
                self.control_session.clone(),
                "pause_run".to_string(),
                json!({ "run_id": run_id }),
            ),
            Intent::ResumeRun { run_id } => (
                self.control_session.clone(),
                "resume_run".to_string(),
                json!({ "run_id": run_id }),
            ),
            Intent::AcceptDiff { run_id, diff_id } => (
                self.control_session.clone(),
                "accept_diff".to_string(),
                json!({ "run_id": run_id, "diff_id": diff_id }),
            ),
            Intent::RejectDiff { run_id, diff_id } => (
                self.control_session.clone(),
                "reject_diff".to_string(),
                json!({ "run_id": run_id, "diff_id": diff_id }),
            ),
            Intent::ScrubToEvent {
                session_id,
                event_id,
            } => (
                session_id,
                "scrub_to_event".to_string(),
                json!({ "event_id": event_id }),
            ),
            Intent::ForkSession {
                session_id,
                at_event,
            } => (
                session_id,
                "fork_session".to_string(),
                json!({ "at_event": at_event }),
            ),
            Intent::OpenFile { path, line } => (
                self.control_session.clone(),
                "open_file".to_string(),
                json!({ "path": path, "line": line }),
            ),
            Intent::RunCommand { argv, cwd } => (
                self.control_session.clone(),
                "run_command".to_string(),
                json!({ "argv": argv, "cwd": cwd }),
            ),
            Intent::Custom { name, payload } => (
                self.control_session.clone(),
                format!("custom.{name}"),
                payload,
            ),
        };
        // Preserve the namespaced kind (`user.intent.<name>`) while carrying the
        // typed UserIntent view in the open payload.
        let mut new_event = NewEvent::user_intent(
            session_id,
            UserIntentEvent {
                intent: intent_name.clone(),
                args,
            },
        );
        new_event.kind = format!("user.intent.{intent_name}");
        let event = self.events.append(new_event).await?;
        Ok(IntentAck {
            accepted: true,
            event_seq: Some(event.seq),
            message: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{EventLog, InMemoryEventLog};
    use hide_core::ids::RunId;
    use std::sync::Arc;

    #[tokio::test]
    async fn command_router_records_control_intents() {
        let log = Arc::new(InMemoryEventLog::new());
        let control_session = SessionId::new();
        let router = CommandRouter::with_control_session(log.clone(), control_session.clone());
        let ack = router
            .handle(Intent::CancelRun {
                run_id: RunId::new(),
            })
            .await
            .unwrap();

        assert!(ack.accepted);
        let events = log.scan(Some(control_session), None, None).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind.as_str(), "user.intent.cancel_run");
    }

    #[tokio::test]
    async fn command_router_records_submit_turn_in_session() {
        let log = Arc::new(InMemoryEventLog::new());
        let router = CommandRouter::new(log.clone());
        let session = SessionId::new();
        router
            .handle(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "hello".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();

        let events = log.scan(Some(session), None, None).await.unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].kind.as_str(), "user.intent.submit_turn");
    }
}
