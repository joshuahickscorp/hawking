//! Live [`TurnHandler`] / [`SessionBinder`] that reach
//! [`hide_backend::BackendHost`] through the existing intent path.
//!
//! No new dispatch mechanism: an ACP `session/prompt` is mapped to
//! [`HideTurnIntent`] by the server, then this handler posts
//! [`hide_core::api::Intent::SubmitTurn`] via [`BackendHost::handle_intent`] —
//! the same Wire-A path `/v1/hide/intent` uses. Sessions are minted with the
//! host's stable open-or-create registry.
//!
//! Model-free turns (no `HIDE_MODEL_WEIGHTS`) still complete: the host records
//! the intent and surfaces "model offline" rather than fabricating tokens. This
//! module only closes the ACP → host wire; generation quality is out of scope.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use hide_backend::BackendHost;
use hide_core::api::Intent;
use hide_core::types::BlobRef;
use hide_protocol::ids::{ItemId, SessionId as ProtocolSessionId, ThreadId};
use hide_protocol::item::{
    AgentMessage, Blocker, Completion, Item, ItemKind, UserMessage as ProtocolUserMessage,
};
use hide_protocol::model::CompletionStatus;

use crate::ids::AcpSessionId;
use crate::ingest::HideTurnIntent;
use crate::server::{MintedSession, SessionBinder, TurnEvent, TurnHandler};

/// [`TurnHandler`] that deposits each mapped prompt onto a shared
/// [`BackendHost`] as `Intent::SubmitTurn`.
pub struct BackendTurnHandler {
    host: Arc<BackendHost>,
}

impl BackendTurnHandler {
    pub fn new(host: Arc<BackendHost>) -> Self {
        Self { host }
    }

    pub fn host(&self) -> &Arc<BackendHost> {
        &self.host
    }
}

impl TurnHandler for BackendTurnHandler {
    fn handle_turn(&mut self, intent: &HideTurnIntent) -> Vec<TurnEvent> {
        let session = hide_core::ids::SessionId::from(intent.session.as_str());
        let text = intent.message.text.clone();
        let attachments: Vec<BlobRef> = intent
            .message
            .attachments
            .iter()
            .map(|a| BlobRef {
                id: hide_core::ids::BlobId::from(a.id.as_str()),
                hash: a.hash.clone(),
                size_bytes: a.size_bytes,
                media_type: a.media_type.clone(),
            })
            .collect();

        let host = self.host.clone();
        // TurnHandler is sync (ACP server loop); bridge with a short block_on
        // on the current runtime if one is running, else a throwaway runtime.
        let ack = block_on(async move {
            host.handle_intent(Intent::SubmitTurn {
                session_id: session,
                text: text.clone(),
                attachments,
            })
            .await
        });

        match ack {
            Ok(ack) if ack.accepted => {
                // The host accepted the turn. Project the user message plus an
                // honest completion: with no model online the host surfaces
                // "model offline" on Wire-B rather than fabricating tokens, so
                // we do not invent agent text here either.
                let offline = self.host.runtime_state().is_none();
                let mut events = vec![TurnEvent::Item(Item::new(
                    ItemId::new("acp_user"),
                    0,
                    ItemKind::UserMessage(ProtocolUserMessage {
                        text: intent.message.text.clone(),
                        attachments: intent.message.attachments.clone(),
                    }),
                ))];
                if offline {
                    events.push(TurnEvent::Item(Item::new(
                        ItemId::new("acp_blocker_offline"),
                        1,
                        ItemKind::Blocker(Blocker {
                            code: "model_offline".to_string(),
                            message: "turn accepted on BackendHost; no model runtime configured (set HIDE_MODEL_WEIGHTS)"
                                .to_string(),
                            needs: Some("a Ready RuntimeSupervisor / HIDE_MODEL_WEIGHTS".to_string()),
                        }),
                    )));
                    events.push(TurnEvent::Item(Item::new(
                        ItemId::new("acp_completion"),
                        2,
                        ItemKind::Completion(Completion {
                            status: CompletionStatus::Partial,
                            summary: Some(
                                "SubmitTurn accepted on host; generation deferred (model offline)"
                                    .to_string(),
                            ),
                        }),
                    )));
                } else {
                    events.push(TurnEvent::Item(Item::new(
                        ItemId::new("acp_agent"),
                        1,
                        ItemKind::AgentMessage(AgentMessage {
                            text: "turn accepted on BackendHost".to_string(),
                        }),
                    )));
                    events.push(TurnEvent::Item(Item::new(
                        ItemId::new("acp_completion"),
                        2,
                        ItemKind::Completion(Completion {
                            status: CompletionStatus::Success,
                            summary: Some("SubmitTurn accepted on host".to_string()),
                        }),
                    )));
                }
                events
            }
            Ok(ack) => {
                let reason = ack
                    .message
                    .unwrap_or_else(|| "SubmitTurn refused by host".to_string());
                vec![
                    TurnEvent::Item(Item::new(
                        ItemId::new("acp_blocker_refused"),
                        0,
                        ItemKind::Blocker(Blocker {
                            code: "submit_refused".to_string(),
                            message: reason.clone(),
                            needs: None,
                        }),
                    )),
                    TurnEvent::Item(Item::new(
                        ItemId::new("acp_completion_refused"),
                        1,
                        ItemKind::Completion(Completion {
                            status: CompletionStatus::Failed,
                            summary: Some(reason),
                        }),
                    )),
                ]
            }
            Err(e) => vec![
                TurnEvent::Item(Item::new(
                    ItemId::new("acp_blocker_error"),
                    0,
                    ItemKind::Blocker(Blocker {
                        code: "host_error".to_string(),
                        message: e.to_string(),
                        needs: None,
                    }),
                )),
                TurnEvent::Item(Item::new(
                    ItemId::new("acp_completion_error"),
                    1,
                    ItemKind::Completion(Completion {
                        status: CompletionStatus::Failed,
                        summary: Some(e.to_string()),
                    }),
                )),
            ],
        }
    }
}

/// [`SessionBinder`] that mints HIDE sessions through the host registry.
pub struct HostSessionBinder {
    host: Arc<BackendHost>,
    n: AtomicU64,
}

impl HostSessionBinder {
    pub fn new(host: Arc<BackendHost>) -> Self {
        Self {
            host,
            n: AtomicU64::new(0),
        }
    }
}

impl SessionBinder for HostSessionBinder {
    fn new_session(&mut self, cwd: &str) -> MintedSession {
        let n = self.n.fetch_add(1, Ordering::Relaxed) + 1;
        // Stable open-or-create per ACP slot so a reloaded editor reattaches.
        let name = format!("acp_{n}_{cwd}");
        let session = self.host.services.session_named(&name);
        MintedSession {
            acp: AcpSessionId::new(format!("sess_{n}")),
            session: ProtocolSessionId::new(session.as_str().to_string()),
            thread: ThreadId::new(format!("thr_{n}")),
        }
    }
}

/// Run an async future from the sync TurnHandler surface.
fn block_on<F, T>(fut: F) -> T
where
    F: std::future::Future<Output = T>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => tokio::task::block_in_place(|| handle.block_on(fut)),
        Err(_) => tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime for BackendTurnHandler")
            .block_on(fut),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_backend::BackendServices;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_protocol::item::UserMessage;
    use hide_protocol::protocol::Method;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    fn host() -> Arc<BackendHost> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let uniq = COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hide_acp_backend_{}_{}", now_ms(), uniq));
        let config = HideConfig::for_workspace(&dir);
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        Arc::new(host)
    }
    #[test]
    fn backend_turn_handler_posts_submit_turn_to_the_host() {
        let host = host();
        let session = host.services.session();
        let mut handler = BackendTurnHandler::new(host.clone());
        let intent = HideTurnIntent {
            method: Method::TurnCreate,
            session: ProtocolSessionId::new(session.as_str().to_string()),
            thread: ThreadId::new("thr_1"),
            message: UserMessage {
                text: "hello from acp".to_string(),
                attachments: vec![],
            },
        };
        let events = handler.handle_turn(&intent);
        assert!(
            events.iter().any(|e| matches!(
                e,
                TurnEvent::Item(Item {
                    kind: ItemKind::UserMessage(UserMessage { text, .. }),
                    ..
                }) if text == "hello from acp"
            )),
            "handler projects the user message: {events:?}"
        );
        assert!(events.iter().any(|e| matches!(
            e,
            TurnEvent::Item(Item {
                kind: ItemKind::Completion(_),
                ..
            })
        )));
        let log = host.services.event_log.clone();
        let events_on_host =
            block_on(async move { log.scan(Some(session), None, None).await.unwrap() });
        assert!(events_on_host.iter().any(|e| {
            e.kind.contains("submit") || e.payload.to_string().contains("hello from acp")
        }));
    }
    #[test]
    fn host_session_binder_mints_stable_host_sessions() {
        let host = host();
        let mut binder = HostSessionBinder::new(host.clone());
        let a = binder.new_session("/repo");
        let b = binder.new_session("/repo");
        assert_ne!(a.session.as_str(), b.session.as_str());
        let again = host.services.session_named("acp_1_/repo");
        assert_eq!(a.session.as_str(), again.as_str());
    }
}
