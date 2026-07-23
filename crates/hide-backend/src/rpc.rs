//! The protocol RPC surface: make the elevated `hide-protocol` Agent Server
//! (Bible sec 15, sec 78.1 #2) REACHABLE, additively, on top of the existing
//! [`BackendHost`] session model.
//!
//! [`BackendHost::rpc`] is a single dispatcher that maps each protocol
//! [`Method`] onto an EXISTING host capability:
//!
//! * `thread/fork`          -> [`BackendHost::fork_session_from_event`]
//! * `session/get`, `thread/get`, `thread/list` -> [`BackendHost::conversation_graph`]
//! * `item/list`            -> [`BackendHost::search_transcript`]
//! * `goal/set` `goal/get` `goal/list` -> [`BackendHost::goal_set`] / [`BackendHost::goal_get`]
//! * `checkpoint/create` `checkpoint/list` `checkpoint/restore`
//!                          -> the durable `checkpoint_*` family
//! * `state/inspect`        -> a light, model-free runtime inspection
//! * `approval/respond`     -> the [`crate::approval::ApprovalHub`]
//!
//! Every other method returns a TYPED [`RpcResult::NotImplemented`] carrying the
//! reason it is deferred. Nothing is faked: a method whose host capability is
//! not built in this model-free harness is honestly reported, never stubbed with
//! a fabricated success. A model-assisted capability is
//! `DEFERRED_MODEL_REQUIRED`; the transport-only paths (turn control) already
//! ride `/v1/hide/intent` and are deferred here on purpose.
//!
//! The server-to-client Notification stream (sec 15.5) is the elevation of the
//! existing Wire-B UiEvent bus. That mapping already exists in the protocol
//! crate's compat bridge; [`ui_event_to_notification`] re-exposes it here so a
//! transport can turn a live [`UiEvent`] into a protocol [`Notification`]
//! without reaching across crates.

use crate::approval::ApprovalDecision;
use crate::host::BackendHost;
use crate::replay::TranscriptQuery;
use hide_core::api::UiEvent;
use hide_core::ids::{EventId, RunId, SessionId, StepId};
use hide_protocol::protocol::{Method, Notification};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// The typed outcome of a protocol RPC dispatch (Bible sec 15).
///
/// A three-way, self-describing result: a concrete `Ok` payload (the host
/// method's own result, serialized), an honest `NotImplemented` (a recognized
/// method whose host capability is not built yet, DEFERRED), or an `Error` (the
/// method dispatched but the host capability rejected it: bad params, not
/// found, ...). Serializes internally tagged on `status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum RpcResult {
    /// The method dispatched to a host capability and produced `result`.
    Ok { method: String, result: Value },
    /// The method is recognized but its host capability is not built in this
    /// model-free harness yet. Typed and honest, never a faked success.
    NotImplemented { method: String, reason: String },
    /// The method dispatched but the underlying host capability errored.
    Error { method: String, message: String },
}

impl RpcResult {
    /// A successful dispatch carrying the host method's serialized result.
    pub fn ok(method: Method, result: Value) -> Self {
        RpcResult::Ok {
            method: method.as_str().to_string(),
            result,
        }
    }

    /// A recognized-but-deferred method (typed, not a panic, not a fake).
    pub fn not_implemented(method: Method, reason: impl Into<String>) -> Self {
        RpcResult::NotImplemented {
            method: method.as_str().to_string(),
            reason: reason.into(),
        }
    }

    /// A dispatch whose host capability rejected the call.
    pub fn error(method: Method, message: impl Into<String>) -> Self {
        RpcResult::Error {
            method: method.as_str().to_string(),
            message: message.into(),
        }
    }

    /// Whether this is a successful dispatch.
    pub fn is_ok(&self) -> bool {
        matches!(self, RpcResult::Ok { .. })
    }

    /// Whether this is a typed `NotImplemented`.
    pub fn is_not_implemented(&self) -> bool {
        matches!(self, RpcResult::NotImplemented { .. })
    }

    /// The `Ok` result payload, if any (for callers/tests reading the value).
    pub fn result(&self) -> Option<&Value> {
        match self {
            RpcResult::Ok { result, .. } => Some(result),
            _ => None,
        }
    }
}

/// Map a live Wire-B [`UiEvent`] onto a protocol [`Notification`] (sec 15.5).
///
/// The Notification stream is the elevation of the existing UiEvent bus: this
/// reuses the protocol crate's compat bridge so the server's push channel and
/// the protocol's notification set stay in sync. Cheap + model-free. A
/// transport that wants to speak the elevated protocol over its event socket
/// forwards `ui_event_to_notification(&event)` in place of the raw UiEvent.
pub fn ui_event_to_notification(event: &UiEvent) -> Notification {
    hide_protocol::compat::uievent_to_notification(event)
}

/// Params carrying exactly one session id, addressed as `session` (or the
/// `session_id` alias). Shared by the read methods.
#[derive(Debug, Deserialize)]
struct SessionParams {
    #[serde(alias = "session_id")]
    session: String,
}

impl BackendHost {
    /// Dispatch a protocol [`Method`] (Bible sec 15) onto the existing host
    /// capability that serves it, returning a typed [`RpcResult`].
    ///
    /// This is the ADDITIVE reachability seam: it never rewrites the internal
    /// session model, it composes the already-built `fork_session_from_event` /
    /// `search_transcript` / `conversation_graph` / `goal_*` / `checkpoint_*` /
    /// approval-hub capabilities. An unmapped method returns a typed
    /// [`RpcResult::NotImplemented`] (DEFERRED) rather than panicking or faking a
    /// result.
    /// Map a live Wire-B [`UiEvent`] onto a protocol [`Notification`] FOR A
    /// SPECIFIC CONNECTION (Stage 4 opt-out suppression). Returns `None` when the
    /// connection opted out of that notification's wire method at Initialize (see
    /// [`crate::initialize::ClientCapabilities::opt_out_notification_methods`]), so
    /// the emit path simply skips it. A connection that never initialized, or did
    /// not opt out of this method, gets the notification as usual. This is where
    /// the negotiated per-connection capabilities are consulted.
    pub fn notification_for_connection(
        &self,
        connection_id: &str,
        event: &UiEvent,
    ) -> Option<Notification> {
        let notification = ui_event_to_notification(event);
        if self
            .connections()
            .is_notification_suppressed(connection_id, notification.method())
        {
            None
        } else {
            Some(notification)
        }
    }

    pub async fn rpc(&self, method: Method, params: Value) -> RpcResult {
        match method {
            // -- thread/fork -> fork_session (event-boundary variant) -----------
            // Forks a NEW independent session from `from`, optionally at an event
            // boundary (`at_event`); `None` forks the whole session's tail. Records
            // ancestry + surfaces a UiEvent, and returns the new session id.
            Method::ThreadFork => {
                #[derive(Deserialize)]
                struct P {
                    #[serde(alias = "session_id")]
                    from: String,
                    #[serde(default)]
                    at_event: Option<String>,
                }
                let p: P = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let at = p.at_event.map(EventId::from);
                match self
                    .fork_session_from_event(SessionId::from(p.from), at.as_ref())
                    .await
                {
                    Ok((new_session, record, _projection)) => RpcResult::ok(
                        method,
                        json!({ "session_id": new_session, "record": record }),
                    ),
                    Err(e) => RpcResult::error(method, e.to_string()),
                }
            }

            // -- session/get, thread/get, thread/list -> conversation_graph -----
            // The bounded, deterministic graph rooted at the session: the node,
            // its ancestry chain, and its direct children (forks / side chats).
            Method::SessionGet | Method::ThreadGet | Method::ThreadList => {
                let p: SessionParams = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let graph = self.conversation_graph(&SessionId::from(p.session));
                RpcResult::ok(method, to_value_or_error(method, &graph))
            }

            // -- item/list -> search_transcript ---------------------------------
            // A literal-substring + structured-filter transcript search (model-free;
            // semantic search is DEFERRED_MODEL_REQUIRED). The params deserialize
            // straight into the host's own TranscriptQuery.
            Method::ItemList => {
                let query: TranscriptQuery = match serde_json::from_value(params) {
                    Ok(q) => q,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                match self.search_transcript(&query).await {
                    Ok(hits) => RpcResult::ok(method, json!({ "hits": hits })),
                    Err(e) => RpcResult::error(method, e.to_string()),
                }
            }

            // -- goal/set -> goal_set -------------------------------------------
            Method::GoalSet => {
                #[derive(Deserialize)]
                struct P {
                    #[serde(alias = "session_id")]
                    session: String,
                    condition: String,
                    #[serde(default)]
                    acceptance: Vec<String>,
                }
                let p: P = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                match self.goal_set(SessionId::from(p.session), p.condition, p.acceptance) {
                    Ok(record) => RpcResult::ok(method, to_value_or_error(method, &record)),
                    Err(e) => RpcResult::error(method, e.to_string()),
                }
            }

            // -- goal/get -> goal_get -------------------------------------------
            Method::GoalGet => {
                let p: SessionParams = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let goal = self.goal_get(&SessionId::from(p.session));
                RpcResult::ok(method, json!({ "goal": goal }))
            }

            // -- goal/list -> goal_get (a session carries at most one goal) ------
            Method::GoalList => {
                let p: SessionParams = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let goals: Vec<_> = self.goal_get(&SessionId::from(p.session)).into_iter().collect();
                RpcResult::ok(method, json!({ "goals": goals }))
            }

            // -- checkpoint/create -> checkpoint_create -------------------------
            Method::CheckpointCreate => {
                #[derive(Deserialize)]
                struct P {
                    #[serde(alias = "session_id")]
                    session: String,
                    #[serde(default)]
                    at_event: Option<String>,
                    #[serde(default)]
                    label: String,
                }
                let p: P = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let at = p.at_event.map(EventId::from);
                match self
                    .checkpoint_create(SessionId::from(p.session), at.as_ref(), p.label)
                    .await
                {
                    Ok(record) => RpcResult::ok(method, to_value_or_error(method, &record)),
                    Err(e) => RpcResult::error(method, e.to_string()),
                }
            }

            // -- checkpoint/list -> checkpoint_list -----------------------------
            Method::CheckpointList => {
                let p: SessionParams = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let list = self.checkpoint_list(&SessionId::from(p.session));
                RpcResult::ok(method, json!({ "checkpoints": list }))
            }

            // -- checkpoint/restore -> checkpoint_restore -----------------------
            Method::CheckpointRestore => {
                #[derive(Deserialize)]
                struct P {
                    #[serde(alias = "id")]
                    checkpoint_id: String,
                }
                let p: P = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                match self.checkpoint_restore(&p.checkpoint_id).await {
                    Ok((restored, record, _projection)) => RpcResult::ok(
                        method,
                        json!({ "session_id": restored, "record": record }),
                    ),
                    Err(e) => RpcResult::error(method, e.to_string()),
                }
            }

            // -- state/inspect -> a light, model-free runtime inspection --------
            // The durable state-capsule model (save/load/fork/release) is not built
            // (see below); inspect surfaces the one piece of live state the host
            // owns today: whether a model runtime is configured and its state.
            Method::StateInspect => RpcResult::ok(
                method,
                json!({
                    "model_configured": self.runtime_state().is_some(),
                    "runtime": self.runtime_state(),
                }),
            ),

            // -- approval/respond -> the ApprovalHub ----------------------------
            // Deposit a human decision on a paused effectful step. `approve: bool`
            // or a `decision` string ("approve" / "deny"); an optional `step_id`
            // scopes it. Model-free.
            Method::ApprovalRespond => {
                #[derive(Deserialize)]
                struct P {
                    run_id: String,
                    #[serde(default)]
                    step_id: Option<String>,
                    #[serde(default)]
                    approve: Option<bool>,
                    #[serde(default)]
                    decision: Option<String>,
                }
                let p: P = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                let approve = match p.approve {
                    Some(b) => b,
                    None => match p.decision.as_deref() {
                        Some("approve") | Some("allow") | Some("accept") => true,
                        Some("deny") | Some("reject") => false,
                        _ => {
                            return RpcResult::error(
                                method,
                                "approval/respond requires `approve: bool` or `decision: \"approve\"|\"deny\"`",
                            )
                        }
                    },
                };
                let decision = if approve {
                    ApprovalDecision::Approve
                } else {
                    ApprovalDecision::Deny
                };
                self.approvals()
                    .decide(RunId::from(p.run_id), p.step_id.map(StepId::from), decision);
                RpcResult::ok(method, json!({ "accepted": true, "approved": approve }))
            }

            // -- workspace/* : lifecycle not built in the harness (DEFERRED) -----
            Method::WorkspaceCreate
            | Method::WorkspaceOpen
            | Method::WorkspaceClose
            | Method::WorkspaceGet
            | Method::WorkspaceList => RpcResult::not_implemented(
                method,
                "workspace lifecycle is not built in the model-free harness (DEFERRED)",
            ),

            // -- environment/* : provisioning not built (DEFERRED) --------------
            Method::EnvironmentCreate
            | Method::EnvironmentGet
            | Method::EnvironmentList
            | Method::EnvironmentDispose => RpcResult::not_implemented(
                method,
                "environment provisioning is not built (DEFERRED)",
            ),

            // -- session/* (beyond get/fork): not exposed yet (DEFERRED) --------
            Method::SessionNew | Method::SessionList | Method::SessionClose => {
                RpcResult::not_implemented(
                    method,
                    "session lifecycle beyond get/fork is not exposed on the RPC surface yet (DEFERRED)",
                )
            }

            // -- thread create / side-chat: not wired to RPC yet (DEFERRED) -----
            Method::ThreadNew | Method::ThreadForkEphemeral | Method::ThreadMergeSummary => {
                RpcResult::not_implemented(
                    method,
                    "thread create / side-chat is not wired to the RPC surface yet (DEFERRED)",
                )
            }

            // -- turn/steer -> steer_run (protocol parity with Wire-A redirect_run)
            // Delivers a real InterruptHub Steer to the running run so the kernel
            // folds the directive into its next planning step, and records the
            // durable `turn.steer` event. Params: `{ run_id, directive|text,
            // session_id? }`.
            Method::TurnSteer => {
                #[derive(Deserialize)]
                struct P {
                    run_id: String,
                    #[serde(alias = "text", alias = "instruction")]
                    directive: String,
                    #[serde(default, alias = "session")]
                    session_id: Option<String>,
                }
                let p: P = match serde_json::from_value(params) {
                    Ok(p) => p,
                    Err(e) => return RpcResult::error(method, format!("bad params: {e}")),
                };
                match self
                    .steer_run(
                        RunId::from(p.run_id),
                        p.directive,
                        p.session_id.map(SessionId::from),
                    )
                    .await
                {
                    Ok(event) => RpcResult::ok(
                        method,
                        json!({ "accepted": true, "event_seq": event.seq }),
                    ),
                    Err(e) => RpcResult::error(method, e.to_string()),
                }
            }

            // -- turn/* (rest) : served by /v1/hide/intent (Wire-A); RPC DEFERRED --
            Method::TurnCreate
            | Method::TurnGet
            | Method::TurnInterrupt
            | Method::TurnPause
            | Method::TurnResume => RpcResult::not_implemented(
                method,
                "turn control is served by /v1/hide/intent (Wire-A); the RPC turn/* surface is DEFERRED",
            ),

            // -- item/get, item/subscribe: DEFERRED -----------------------------
            // item/list maps to transcript search above; get-by-id has no host
            // method, and subscribe rides the UiEvent -> Notification bus
            // (`ui_event_to_notification`) rather than a request/response.
            Method::ItemGet | Method::ItemSubscribe => RpcResult::not_implemented(
                method,
                "item get-by-id / subscribe is DEFERRED (item/list -> transcript search; subscribe rides the notification bus)",
            ),

            // -- agent/* : model-assisted, DEFERRED_MODEL_REQUIRED --------------
            Method::AgentSpawn | Method::AgentGet | Method::AgentList | Method::AgentResult => {
                RpcResult::not_implemented(
                    method,
                    "agent spawn / result is DEFERRED_MODEL_REQUIRED",
                )
            }

            // -- state save/load/fork/release: capsule model not built (DEFERRED)
            Method::StateSave
            | Method::StateLoad
            | Method::StateFork
            | Method::StateRelease => RpcResult::not_implemented(
                method,
                "the durable state-capsule model is not built (DEFERRED)",
            ),

            // -- approval/request: a server push, not a client request (DEFERRED)
            Method::ApprovalRequestMethod => RpcResult::not_implemented(
                method,
                "approval/request is a server push (sec 15.5); a client-initiated request is DEFERRED",
            ),

            // -- artifact/* : artifact store not built (DEFERRED) ---------------
            Method::ArtifactGet | Method::ArtifactList | Method::ArtifactPut => {
                RpcResult::not_implemented(method, "the artifact store is not built (DEFERRED)")
            }
        }
    }
}

/// Serialize a host result into a JSON value, degrading a (near-impossible)
/// serialization failure into a readable string rather than panicking.
fn to_value_or_error<T: Serialize>(method: Method, value: &T) -> Value {
    serde_json::to_value(value)
        .unwrap_or_else(|e| json!({ "serialize_error": format!("{}: {e}", method.as_str()) }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::BackendServices;
    use hide_core::api::Intent;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_core::types::Decision;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn host_for_test() -> BackendHost {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let uniq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hide_rpc_{}_{}", now_ms(), uniq));
        let mut config = HideConfig::for_workspace(&dir);
        config.security.shell_default = Decision::Allow;
        BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap()
    }

    /// Seed a session with one user turn so it has a durable transcript to search
    /// and a tail to fork. Returns the session id.
    async fn seed_session(host: &BackendHost, text: &str) -> SessionId {
        let session = host.services.session();
        host.handle_intent(Intent::SubmitTurn {
            session_id: session.clone(),
            text: text.to_string(),
            attachments: vec![],
        })
        .await
        .unwrap();
        session
    }

    #[tokio::test]
    async fn thread_fork_forks_and_returns_a_new_session_id() {
        let host = host_for_test();
        let session = seed_session(&host, "fork me at the tail").await;
        let out = host
            .rpc(Method::ThreadFork, json!({ "from": session.as_str() }))
            .await;
        let result = out.result().expect("thread/fork returns an Ok result");
        let new_id = result["session_id"].as_str().expect("carries a session_id");
        assert!(!new_id.is_empty(), "the fork's session id is non-empty");
        assert_ne!(new_id, session.as_str(), "the fork is a NEW, independent session");
    }

    #[tokio::test]
    async fn item_list_searches_the_transcript_and_returns_hits() {
        let host = host_for_test();
        let session = seed_session(&host, "the flaky auth retry needs five fixes").await;
        let out = host
            .rpc(
                Method::ItemList,
                json!({ "text": "flaky auth", "session_id": session.as_str() }),
            )
            .await;
        let result = out.result().expect("item/list returns an Ok result");
        let hits = result["hits"].as_array().expect("carries a hits array");
        assert!(!hits.is_empty(), "the literal substring matches the seeded turn");
    }

    #[tokio::test]
    async fn goal_set_records_a_durable_goal() {
        let host = host_for_test();
        let session = host.services.session();
        let out = host
            .rpc(
                Method::GoalSet,
                json!({
                    "session": session.as_str(),
                    "condition": "all oracles pass",
                    "acceptance": ["build", "test"]
                }),
            )
            .await;
        assert!(out.is_ok(), "goal/set dispatches successfully");
        // The record is durable: goal_get reads it straight back.
        let stored = host.goal_get(&session).expect("the goal was recorded durably");
        assert_eq!(stored.condition, "all oracles pass");
        assert_eq!(stored.acceptance, vec!["build".to_string(), "test".to_string()]);
    }

    #[tokio::test]
    async fn checkpoint_create_records_a_checkpoint() {
        let host = host_for_test();
        let session = seed_session(&host, "checkpoint this boundary").await;
        let out = host
            .rpc(
                Method::CheckpointCreate,
                json!({ "session": session.as_str(), "label": "before-refactor" }),
            )
            .await;
        assert!(out.is_ok(), "checkpoint/create dispatches successfully");
        // The record is durable: checkpoint_list reads it straight back.
        let list = host.checkpoint_list(&session);
        assert_eq!(list.len(), 1, "exactly one checkpoint was recorded");
        assert_eq!(list[0].label, "before-refactor");
    }

    #[tokio::test]
    async fn approval_respond_deposits_a_decision() {
        let host = host_for_test();
        let run = hide_core::ids::RunId::new();
        let out = host
            .rpc(
                Method::ApprovalRespond,
                json!({ "run_id": run.as_str(), "approve": true }),
            )
            .await;
        assert!(out.is_ok(), "approval/respond dispatches successfully");
        // The decision landed in the hub's mailbox for that run.
        assert!(host.approvals().is_pending(&run), "the decision was deposited");
    }

    #[tokio::test]
    async fn state_inspect_returns_a_model_free_snapshot() {
        let host = host_for_test();
        let out = host.rpc(Method::StateInspect, json!({})).await;
        let result = out.result().expect("state/inspect returns an Ok result");
        // No model is configured in the harness, so model_configured is false and
        // runtime is null. The point is a TYPED, non-panicking snapshot.
        assert_eq!(result["model_configured"], json!(false));
    }

    #[tokio::test]
    async fn unimplemented_method_returns_typed_not_implemented_not_a_panic() {
        let host = host_for_test();
        for method in [
            Method::WorkspaceCreate,
            Method::AgentSpawn,
            Method::ArtifactGet,
            Method::StateSave,
            Method::TurnCreate,
        ] {
            let out = host.rpc(method, json!({})).await;
            assert!(
                out.is_not_implemented(),
                "{} must return a typed NotImplemented",
                method.as_str()
            );
        }
    }

    #[tokio::test]
    async fn bad_params_return_a_typed_error_not_a_panic() {
        let host = host_for_test();
        // thread/fork requires a `from` session id; an empty object is bad params.
        let out = host.rpc(Method::ThreadFork, json!({})).await;
        assert!(
            matches!(out, RpcResult::Error { .. }),
            "missing required params surface as a typed Error"
        );
    }

    #[test]
    fn ui_event_maps_onto_a_protocol_notification() {
        use hide_core::api::{UiEvent, UiEventKind};
        let ev = UiEvent {
            seq: 3,
            session_id: None,
            kind: UiEventKind::RuntimeStatus {
                status: "ready".to_string(),
                detail: Some("model online".to_string()),
            },
        };
        match ui_event_to_notification(&ev) {
            Notification::RuntimeStatus { status, detail } => {
                assert_eq!(status, "ready");
                assert_eq!(detail.as_deref(), Some("model online"));
            }
            other => panic!("expected runtime/status notification, got {other:?}"),
        }
    }
}
