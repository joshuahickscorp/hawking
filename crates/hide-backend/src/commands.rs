//! The intent command router — the transport-agnostic Wire-A entry (bible ch.07
//! §4.4).
//!
//! [`CommandRouter::handle`] takes a typed [`Intent`], **validates** it, appends
//! a `user.intent.*` event on success, and returns an [`IntentAck`]. Two things
//! the scaffold lacked, now real:
//!
//! 1. **Validation / rejection** — each handler validates its args and can
//!    *reject* (`IntentAck { accepted: false, message: Some(reason) }`). `accepted`
//!    is no longer always `true`: an empty `SubmitTurn`, an empty-argv
//!    `RunCommand`, or a `Custom` with a blank name is refused *before* anything
//!    is logged.
//! 2. **Control intents actually signal** — `CancelRun`/`PauseRun`/`ResumeRun`
//!    don't just append a log line; they push an [`Interrupt`] onto the
//!    [`InterruptHub`] for that `run_id`, which the running kernel polls between
//!    transitions (the `hide_kernel::govern::Interrupt` seam). A run with no
//!    listener still records the intent (the signal is buffered for when a
//!    listener attaches).
//!
//! ## Deferred seam — Tauri
//!
//! `handle` is a plain `async fn` on purpose: it is **transport-agnostic**. A
//! future Tauri layer wraps it in a thin `#[tauri::command]` that the frontend
//! reaches via `invoke('hide_intent', { intent })` — the command does nothing but
//! deserialize the `Intent`, call `router.handle(intent)`, and serialize the
//! `IntentAck`. We deliberately do **not** add the `tauri` dep in the shell (the
//! host stays headless + unit-testable); the wrapper is post-shell host work.

use crate::interrupt::InterruptHub;
use hide_core::api::{Intent, IntentAck};
use hide_core::event::{NewEvent, UserIntentEvent};
use hide_core::ids::SessionId;
use hide_core::persistence::DynEventLog;
use hide_core::Result;
use hide_kernel::govern::Interrupt;
use serde_json::json;
use std::sync::Arc;

pub struct CommandRouter {
    events: DynEventLog,
    control_session: SessionId,
    interrupts: Arc<InterruptHub>,
}

impl CommandRouter {
    pub fn new(events: DynEventLog) -> Self {
        Self::with_interrupts(events, Arc::new(InterruptHub::default()))
    }

    pub fn with_control_session(events: DynEventLog, control_session: SessionId) -> Self {
        Self {
            events,
            control_session,
            interrupts: Arc::new(InterruptHub::default()),
        }
    }

    pub fn with_interrupts(events: DynEventLog, interrupts: Arc<InterruptHub>) -> Self {
        Self {
            events,
            control_session: SessionId::new(),
            interrupts,
        }
    }

    pub fn control_session(&self) -> &SessionId {
        &self.control_session
    }

    /// The interrupt hub control intents signal onto. The host shares this with
    /// the kernel/fleet so `Cancel`/`Pause`/`Resume` actually reach a running run.
    pub fn interrupts(&self) -> &Arc<InterruptHub> {
        &self.interrupts
    }

    pub async fn handle(&self, intent: Intent) -> Result<IntentAck> {
        // 1. Validate. A rejection returns *before* anything is appended.
        if let Err(reason) = validate(&intent) {
            return Ok(IntentAck {
                accepted: false,
                event_seq: None,
                message: Some(reason),
            });
        }

        // 2. Control intents signal the running run via the interrupt hub. The
        //    signal is buffered even if no listener has attached yet.
        match &intent {
            Intent::CancelRun { run_id } => {
                self.interrupts.signal(run_id.clone(), Interrupt::Abort);
            }
            Intent::PauseRun { run_id } => {
                self.interrupts.signal(run_id.clone(), Interrupt::Pause);
            }
            Intent::ResumeRun { run_id } => {
                // Resume clears any buffered pause (the run continues).
                self.interrupts.clear(run_id);
            }
            _ => {}
        }

        // 3. Map to the durable intent event.
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

/// Validate an intent's arguments. `Err(reason)` => the router rejects it with
/// `accepted: false` and the reason in `message`.
fn validate(intent: &Intent) -> std::result::Result<(), String> {
    match intent {
        Intent::SubmitTurn { text, .. } => {
            if text.trim().is_empty() {
                return Err("submit_turn: text must not be empty".to_string());
            }
        }
        Intent::RunCommand { argv, .. } => {
            if argv.is_empty() || argv[0].trim().is_empty() {
                return Err("run_command: argv must name a program".to_string());
            }
        }
        Intent::OpenFile { path, .. } => {
            if path.trim().is_empty() {
                return Err("open_file: path must not be empty".to_string());
            }
        }
        Intent::AcceptDiff { diff_id, .. } | Intent::RejectDiff { diff_id, .. } => {
            if diff_id.trim().is_empty() {
                return Err("diff intent: diff_id must not be empty".to_string());
            }
        }
        Intent::Custom { name, .. } => {
            if name.trim().is_empty() {
                return Err("custom: name must not be empty".to_string());
            }
        }
        // Control + time-travel intents carry typed ids; nothing to reject.
        Intent::CancelRun { .. }
        | Intent::PauseRun { .. }
        | Intent::ResumeRun { .. }
        | Intent::ScrubToEvent { .. }
        | Intent::ForkSession { .. } => {}
    }
    Ok(())
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

    #[tokio::test]
    async fn empty_submit_turn_is_rejected_without_logging() {
        let log = Arc::new(InMemoryEventLog::new());
        let router = CommandRouter::new(log.clone());
        let session = SessionId::new();
        let ack = router
            .handle(Intent::SubmitTurn {
                session_id: session.clone(),
                text: "   ".to_string(),
                attachments: Vec::new(),
            })
            .await
            .unwrap();
        assert!(!ack.accepted);
        assert!(ack.message.unwrap().contains("must not be empty"));
        // Rejection logs nothing.
        let events = log.scan(Some(session), None, None).await.unwrap();
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn empty_argv_run_command_is_rejected() {
        let log = Arc::new(InMemoryEventLog::new());
        let router = CommandRouter::new(log.clone());
        let ack = router
            .handle(Intent::RunCommand {
                argv: Vec::new(),
                cwd: None,
            })
            .await
            .unwrap();
        assert!(!ack.accepted);
    }

    #[tokio::test]
    async fn cancel_run_signals_the_interrupt_hub() {
        let log = Arc::new(InMemoryEventLog::new());
        let router = CommandRouter::new(log.clone());
        let run = RunId::new();
        router
            .handle(Intent::CancelRun {
                run_id: run.clone(),
            })
            .await
            .unwrap();
        // The hub buffered an Abort for this run.
        assert!(matches!(
            router.interrupts().take(&run),
            Some(Interrupt::Abort)
        ));
    }
}
