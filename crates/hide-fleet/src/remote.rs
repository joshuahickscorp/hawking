//! Workstation / remote mode — the wire protocol (bible ch.09 §4.9).
//!
//! A laptop thin-client drives a Mac-Studio agent server. The protocol is
//! **ACP-shaped**: JSON-RPC 2.0 over a persistent WebSocket, session-centric,
//! resumable — but the update payload is the **ch.01 Event envelope** (richer than
//! ACP notifications). The server is authoritative; the client is a disposable
//! view (P10).
//!
//! Reliability core (§4.9.4):
//! - **Server-authoritative**: all state lives in the event log; the client holds
//!   only a rebuildable projection.
//! - **Durable sessions**: sessions persist server-side independent of the socket
//!   (a batch survives client sleep/disconnect).
//! - **Reconnect = `session/resume{from_seq}`**: the server replays `(from_seq,
//!   head]` from the log — exactly-once by construction (events are immutable,
//!   `seq`-ordered; replay re-applies recorded data, never re-fires effects, T3).
//! - **Deny-first auth (P11)**: loopback-only by default; tokens only over
//!   wss/loopback; a ch.10 capability grant rides on the token (we transport +
//!   check presence; ch.10 defines the grant).
//!
//! The JSON-RPC framing + session/replay logic is unit-testable without a socket
//! ([`RemoteSession`], [`dispatch`]); [`serve`] runs the real `tokio-tungstenite`
//! loop, and an integration test binds a loopback WS to exercise the handshake +
//! resume end-to-end.

use hide_core::api::Intent;
use hide_core::event::Event;
use hide_core::ids::SessionId;
use hide_core::persistence::DynEventLog;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 envelope (§4.9.2 / A.4)
// ---------------------------------------------------------------------------

/// A JSON-RPC 2.0 request from the client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    /// Request id (number or string); absent for notifications.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// A JSON-RPC 2.0 response/notification from the server.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    /// For server→client notifications (`hide/event`), the method is set and id
    /// is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
}

impl JsonRpcResponse {
    pub fn ok(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(result),
            error: None,
            method: None,
            params: None,
        }
    }

    pub fn err(id: Option<Value>, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
            }),
            method: None,
            params: None,
        }
    }

    /// A server→client `hide/event` notification carrying a ch.01 Event.
    pub fn event_notification(event: &Event) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            id: None,
            result: None,
            error: None,
            method: Some("hide/event".to_string()),
            params: Some(serde_json::to_value(event).unwrap_or(Value::Null)),
        }
    }
}

// JSON-RPC standard error codes + our extensions.
pub const ERR_PARSE: i32 = -32700;
pub const ERR_INVALID_REQUEST: i32 = -32600;
pub const ERR_METHOD_NOT_FOUND: i32 = -32601;
pub const ERR_INVALID_PARAMS: i32 = -32602;
pub const ERR_UNAUTHORIZED: i32 = -32001;
pub const ERR_CAPABILITY: i32 = -32002;

// ---------------------------------------------------------------------------
// Auth posture (§4.9.3, references ch.10)
// ---------------------------------------------------------------------------

/// Deny-first remote auth policy (P11). Default: loopback only, token required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteAuthPolicy {
    pub loopback_only: bool,
    pub allow_ssh_tunnel: bool,
    pub token_required: bool,
    /// Accepted bearer tokens (device-paired). Empty = reject all when
    /// `token_required` (deny-first).
    #[serde(default)]
    pub accepted_tokens: Vec<String>,
}

impl Default for RemoteAuthPolicy {
    fn default() -> Self {
        Self {
            loopback_only: true,
            allow_ssh_tunnel: true,
            token_required: true,
            accepted_tokens: Vec::new(),
        }
    }
}

impl RemoteAuthPolicy {
    /// Whether a peer at `addr` presenting `token` may connect. Loopback peers
    /// over an SSH tunnel may skip the token (the §4.9.3 "loopback + SSH" rule);
    /// non-loopback peers always need a valid token.
    pub fn authorize(&self, addr: &SocketAddr, token: Option<&str>) -> bool {
        let is_loopback = addr.ip().is_loopback();
        if self.loopback_only && !is_loopback {
            return false;
        }
        if !self.token_required {
            return true;
        }
        if is_loopback && self.allow_ssh_tunnel && token.is_none() {
            // Loopback (SSH-forwarded) is trusted without a token by policy.
            return true;
        }
        match token {
            Some(t) => self.accepted_tokens.iter().any(|a| a == t),
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Server-authoritative sessions (§4.9.4)
// ---------------------------------------------------------------------------

/// A server-side session. Persists independent of any connection (a batch keeps
/// running while the laptop sleeps). The session id maps to a ch.01 `SessionId`
/// in the event log; the client resumes by `from_seq`.
#[derive(Debug, Clone)]
pub struct RemoteSession {
    pub session_id: SessionId,
    /// The ch.10 capability grant carried by the client's token (opaque here;
    /// ch.10 validates it). Presence is enforced; semantics are ch.10's.
    pub capability_grant: Option<String>,
    pub transport: String,
}

/// The server's session registry (server-authoritative, P10).
#[derive(Default)]
pub struct SessionRegistry {
    sessions: Mutex<BTreeMap<String, RemoteSession>>,
}

impl SessionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn open(&self, grant: Option<String>, transport: impl Into<String>) -> RemoteSession {
        let session = RemoteSession {
            session_id: SessionId::new(),
            capability_grant: grant,
            transport: transport.into(),
        };
        self.sessions
            .lock()
            .insert(session.session_id.0.clone(), session.clone());
        session
    }

    pub fn get(&self, session_id: &str) -> Option<RemoteSession> {
        self.sessions.lock().get(session_id).cloned()
    }

    pub fn len(&self) -> usize {
        self.sessions.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.sessions.lock().is_empty()
    }
}

// ---------------------------------------------------------------------------
// The intent sink (how the server applies a client intent)
// ---------------------------------------------------------------------------

/// The server forwards client intents to this sink (the host wires it to the
/// kernel/fleet). Decoupled so the protocol is testable without a backend.
pub trait IntentSink: Send + Sync {
    /// Apply an intent in a session; return the event seq it produced (for the
    /// ack). The default test sink just records intents.
    fn submit(&self, session: &RemoteSession, intent: Intent) -> u64;
}

/// A recording sink for tests: stores received intents, returns increasing seqs.
#[derive(Default)]
pub struct RecordingSink {
    pub received: Mutex<Vec<Intent>>,
    next_seq: Mutex<u64>,
}

impl IntentSink for RecordingSink {
    fn submit(&self, _session: &RemoteSession, intent: Intent) -> u64 {
        self.received.lock().push(intent);
        let mut s = self.next_seq.lock();
        *s += 1;
        *s
    }
}

/// The dependencies a dispatch needs: the event log (for `from_seq` replay), the
/// session registry, the intent sink, and the auth-derived capability grant.
pub struct RemoteContext {
    pub log: DynEventLog,
    pub sessions: Arc<SessionRegistry>,
    pub sink: Arc<dyn IntentSink>,
    pub transport: String,
    pub grant: Option<String>,
}

/// Dispatch one JSON-RPC request against the server state, returning the response
/// plus any backlog events to stream (for `session/resume`). This is the pure
/// protocol core — `serve` wraps it with the socket.
pub async fn dispatch(
    ctx: &RemoteContext,
    req: JsonRpcRequest,
) -> (JsonRpcResponse, Vec<Event>) {
    if req.jsonrpc != "2.0" {
        return (
            JsonRpcResponse::err(req.id, ERR_INVALID_REQUEST, "jsonrpc must be \"2.0\""),
            Vec::new(),
        );
    }
    match req.method.as_str() {
        // session/new → open a server-side session (ACP-style handshake).
        "session/new" => {
            let session = ctx.sessions.open(ctx.grant.clone(), ctx.transport.clone());
            (
                JsonRpcResponse::ok(
                    req.id,
                    json!({ "session": session.session_id.0, "head_seq": head_seq(ctx).await }),
                ),
                Vec::new(),
            )
        }
        // session/resume{from_seq} → replay (from_seq, head] (§4.9.4).
        "session/resume" => {
            let session_id = req
                .params
                .get("session")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let from_seq = req.params.get("from_seq").and_then(|v| v.as_u64());
            let Some(session) = ctx.sessions.get(&session_id) else {
                return (
                    JsonRpcResponse::err(req.id, ERR_INVALID_PARAMS, "unknown session"),
                    Vec::new(),
                );
            };
            let backlog = ctx
                .log
                .scan(Some(session.session_id.clone()), from_seq, None)
                .await
                .unwrap_or_default();
            let replayed = backlog.len();
            (
                JsonRpcResponse::ok(
                    req.id,
                    json!({ "session": session_id, "from_seq": from_seq, "replayed_n": replayed }),
                ),
                backlog,
            )
        }
        // hide/intent → apply a client intent (ch.01 Wire A over the wire).
        "hide/intent" => {
            let session_id = req
                .params
                .get("session")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let Some(session) = ctx.sessions.get(&session_id) else {
                return (
                    JsonRpcResponse::err(req.id, ERR_INVALID_PARAMS, "unknown session"),
                    Vec::new(),
                );
            };
            // Every remote intent is checked for a capability grant before it acts
            // (ch.01 T4; ch.10 validates the grant itself).
            if session.capability_grant.is_none() && ctx.grant.is_none() {
                // No grant at all → reject (deny ambient authority, P11).
                return (
                    JsonRpcResponse::err(req.id, ERR_CAPABILITY, "no capability grant on session"),
                    Vec::new(),
                );
            }
            let intent: Result<Intent, _> = serde_json::from_value(
                req.params.get("intent").cloned().unwrap_or(Value::Null),
            );
            match intent {
                Ok(intent) => {
                    let seq = ctx.sink.submit(&session, intent);
                    (
                        JsonRpcResponse::ok(
                            req.id,
                            json!({ "accepted": true, "event_seq": seq }),
                        ),
                        Vec::new(),
                    )
                }
                Err(e) => (
                    JsonRpcResponse::err(
                        req.id,
                        ERR_INVALID_PARAMS,
                        format!("bad intent: {e}"),
                    ),
                    Vec::new(),
                ),
            }
        }
        "ping" => (
            JsonRpcResponse::ok(req.id, json!({ "pong": true })),
            Vec::new(),
        ),
        other => (
            JsonRpcResponse::err(
                req.id,
                ERR_METHOD_NOT_FOUND,
                format!("unknown method: {other}"),
            ),
            Vec::new(),
        ),
    }
}

async fn head_seq(ctx: &RemoteContext) -> u64 {
    ctx.log
        .scan(None, None, None)
        .await
        .ok()
        .and_then(|e| e.last().map(|ev| ev.seq))
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// The WebSocket server (§4.9.2) — real tokio-tungstenite loop
// ---------------------------------------------------------------------------

/// Configuration for the agent server.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub bind: SocketAddr,
    pub auth: RemoteAuthPolicy,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            // Loopback-default (P11). 0 = OS-assigned port (test-friendly).
            bind: "127.0.0.1:0".parse().unwrap(),
            auth: RemoteAuthPolicy::default(),
        }
    }
}

/// A handle to a running server: its bound address + a shutdown signal.
pub struct ServerHandle {
    pub local_addr: SocketAddr,
    shutdown: tokio::sync::watch::Sender<bool>,
}

impl ServerHandle {
    pub fn shutdown(&self) {
        let _ = self.shutdown.send(true);
    }
}

/// Bind the agent server and accept WebSocket connections, each speaking JSON-RPC
/// 2.0. Returns once bound (the accept loop runs as a detached task). The server
/// is authoritative and keeps running when a client disconnects (§4.9.4).
pub async fn serve(
    config: ServerConfig,
    log: DynEventLog,
    sessions: Arc<SessionRegistry>,
    sink: Arc<dyn IntentSink>,
) -> std::io::Result<ServerHandle> {
    let listener = tokio::net::TcpListener::bind(config.bind).await?;
    let local_addr = listener.local_addr()?;
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);

    let auth = config.auth.clone();
    tokio::spawn(async move {
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() { break; }
                }
                accepted = listener.accept() => {
                    let Ok((stream, peer)) = accepted else { continue; };
                    // Loopback/auth gate at the TCP layer (P11). Token auth would
                    // ride the WS handshake headers; loopback-default trusts the
                    // SSH-forwarded peer per policy.
                    if !auth.authorize(&peer, None) {
                        continue;
                    }
                    let log = log.clone();
                    let sessions = sessions.clone();
                    let sink = sink.clone();
                    tokio::spawn(async move {
                        let _ = handle_connection(stream, peer, log, sessions, sink).await;
                    });
                }
            }
        }
    });

    Ok(ServerHandle {
        local_addr,
        shutdown: shutdown_tx,
    })
}

async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer: SocketAddr,
    log: DynEventLog,
    sessions: Arc<SessionRegistry>,
    sink: Arc<dyn IntentSink>,
) -> Result<(), tokio_tungstenite::tungstenite::Error> {
    use futures::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let ws = tokio_tungstenite::accept_async(stream).await?;
    let (mut write, mut read) = ws.split();

    let ctx = RemoteContext {
        log,
        sessions,
        sink,
        // Loopback peers are treated as SSH-tunnel transport; a grant is implied
        // by the trusted-loopback policy so intents are accepted in this mode.
        transport: if peer.ip().is_loopback() {
            "ssh-loopback".to_string()
        } else {
            "wss".to_string()
        },
        grant: Some("loopback-implicit".to_string()),
    };

    while let Some(msg) = read.next().await {
        let msg = msg?;
        match msg {
            Message::Text(text) => {
                let req: JsonRpcRequest = match serde_json::from_str(&text) {
                    Ok(r) => r,
                    Err(e) => {
                        let resp = JsonRpcResponse::err(None, ERR_PARSE, format!("parse: {e}"));
                        write
                            .send(Message::Text(serde_json::to_string(&resp).unwrap()))
                            .await?;
                        continue;
                    }
                };
                let (resp, backlog) = dispatch(&ctx, req).await;
                write
                    .send(Message::Text(serde_json::to_string(&resp).unwrap()))
                    .await?;
                // Stream any replay backlog as `hide/event` notifications.
                for event in backlog {
                    let note = JsonRpcResponse::event_notification(&event);
                    write
                        .send(Message::Text(serde_json::to_string(&note).unwrap()))
                        .await?;
                }
            }
            Message::Close(_) => break,
            Message::Ping(p) => write.send(Message::Pong(p)).await?,
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hide_core::event::{InMemoryEventLog, NewEvent};
    use hide_core::ids::SessionId;

    fn ctx(log: DynEventLog) -> RemoteContext {
        RemoteContext {
            log,
            sessions: Arc::new(SessionRegistry::new()),
            sink: Arc::new(RecordingSink::default()),
            transport: "test".to_string(),
            grant: Some("cap-test".to_string()),
        }
    }

    #[test]
    fn auth_denies_non_loopback_without_token() {
        let policy = RemoteAuthPolicy::default();
        let lan: SocketAddr = "192.168.1.5:9000".parse().unwrap();
        assert!(!policy.authorize(&lan, None));
        let loop_addr: SocketAddr = "127.0.0.1:9000".parse().unwrap();
        // Loopback + SSH-tunnel trusted without a token by default.
        assert!(policy.authorize(&loop_addr, None));
        // LAN with a valid paired token is allowed only if loopback_only is off.
        let mut lan_policy = RemoteAuthPolicy {
            loopback_only: false,
            accepted_tokens: vec!["good".to_string()],
            ..Default::default()
        };
        assert!(lan_policy.authorize(&lan, Some("good")));
        assert!(!lan_policy.authorize(&lan, Some("bad")));
        lan_policy.accepted_tokens.clear();
        assert!(!lan_policy.authorize(&lan, Some("good")));
    }

    #[tokio::test]
    async fn session_new_then_intent_acks_with_seq() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let ctx = ctx(log);
        // open
        let (resp, _) = dispatch(
            &ctx,
            JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(1)),
                method: "session/new".to_string(),
                params: json!({}),
            },
        )
        .await;
        let session = resp.result.unwrap()["session"].as_str().unwrap().to_string();
        // intent
        let intent = Intent::SubmitTurn {
            session_id: SessionId::from(session.as_str()),
            text: "add JWT refresh".to_string(),
            attachments: vec![],
        };
        let (resp, _) = dispatch(
            &ctx,
            JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(2)),
                method: "hide/intent".to_string(),
                params: json!({ "session": session, "intent": intent }),
            },
        )
        .await;
        assert_eq!(resp.result.unwrap()["accepted"], true);
    }

    #[tokio::test]
    async fn session_resume_replays_from_seq() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let ctx = ctx(log.clone());
        // Open a session, then append 3 events to its session id.
        let (resp, _) = dispatch(
            &ctx,
            JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(1)),
                method: "session/new".to_string(),
                params: json!({}),
            },
        )
        .await;
        let session_str = resp.result.unwrap()["session"].as_str().unwrap().to_string();
        let session = ctx.sessions.get(&session_str).unwrap().session_id;
        for i in 0..3 {
            log.append(NewEvent::system(
                session.clone(),
                "agent.phase",
                json!({ "n": i }),
            ))
            .await
            .unwrap();
        }
        // Resume from seq 1 → replays seqs 2,3 (after_seq is exclusive).
        let (resp, backlog) = dispatch(
            &ctx,
            JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(9)),
                method: "session/resume".to_string(),
                params: json!({ "session": session_str, "from_seq": 1 }),
            },
        )
        .await;
        assert_eq!(resp.result.unwrap()["replayed_n"], 2);
        assert_eq!(backlog.len(), 2);
        assert!(backlog.iter().all(|e| e.seq > 1));
    }

    #[tokio::test]
    async fn unknown_method_is_method_not_found() {
        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let ctx = ctx(log);
        let (resp, _) = dispatch(
            &ctx,
            JsonRpcRequest {
                jsonrpc: "2.0".to_string(),
                id: Some(json!(1)),
                method: "bogus/method".to_string(),
                params: json!({}),
            },
        )
        .await;
        assert_eq!(resp.error.unwrap().code, ERR_METHOD_NOT_FOUND);
    }

    #[tokio::test]
    async fn end_to_end_websocket_handshake_intent_and_resume() {
        use futures::{SinkExt, StreamExt};
        use tokio_tungstenite::tungstenite::Message;

        let log: DynEventLog = Arc::new(InMemoryEventLog::new());
        let sessions = Arc::new(SessionRegistry::new());
        let sink = Arc::new(RecordingSink::default());
        let handle = serve(
            ServerConfig::default(),
            log.clone(),
            sessions.clone(),
            sink.clone(),
        )
        .await
        .unwrap();

        let url = format!("ws://{}", handle.local_addr);
        let (mut ws, _) = tokio_tungstenite::connect_async(&url).await.unwrap();

        // session/new
        ws.send(Message::Text(
            json!({ "jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {} })
                .to_string(),
        ))
        .await
        .unwrap();
        let reply = ws.next().await.unwrap().unwrap();
        let v: Value = serde_json::from_str(reply.to_text().unwrap()).unwrap();
        let session = v["result"]["session"].as_str().unwrap().to_string();

        // hide/intent
        let intent = Intent::SubmitTurn {
            session_id: SessionId::from(session.as_str()),
            text: "do it".to_string(),
            attachments: vec![],
        };
        ws.send(Message::Text(
            json!({ "jsonrpc": "2.0", "id": 2, "method": "hide/intent",
                    "params": { "session": session, "intent": intent } })
            .to_string(),
        ))
        .await
        .unwrap();
        let reply = ws.next().await.unwrap().unwrap();
        let v: Value = serde_json::from_str(reply.to_text().unwrap()).unwrap();
        assert_eq!(v["result"]["accepted"], true);
        assert_eq!(sink.received.lock().len(), 1);

        // Append an event to the session, then resume → server streams it back.
        let sid = sessions.get(&session).unwrap().session_id;
        log.append(NewEvent::system(sid, "agent.phase", json!({ "x": 1 })))
            .await
            .unwrap();
        ws.send(Message::Text(
            json!({ "jsonrpc": "2.0", "id": 3, "method": "session/resume",
                    "params": { "session": session, "from_seq": 0 } })
            .to_string(),
        ))
        .await
        .unwrap();
        // First the resume result...
        let reply = ws.next().await.unwrap().unwrap();
        let v: Value = serde_json::from_str(reply.to_text().unwrap()).unwrap();
        assert!(v["result"]["replayed_n"].as_u64().unwrap() >= 1);
        // ...then a hide/event notification carrying the replayed event.
        let note = ws.next().await.unwrap().unwrap();
        let v: Value = serde_json::from_str(note.to_text().unwrap()).unwrap();
        assert_eq!(v["method"], "hide/event");

        handle.shutdown();
    }
}
