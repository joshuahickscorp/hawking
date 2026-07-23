//! hide-serve: the thin localhost HTTP + WebSocket transport that exposes the
//! already-built [`hide_backend::BackendHost`] to the HIDE web front end.
//!
//! This is the seam the FE `ipc.ts` client targets (HIDE_PLAN Part D2). It
//! mirrors `crates/hawking-serve`'s axum style (router + state + handlers, with
//! a thin bin that boots and serves), (de)serializes JSON, and otherwise does
//! nothing: the contract types and host behavior are unchanged.
//!
//! Routes:
//!   POST /v1/hide/intent     -> host.handle_intent(Intent)         -> IntentAck
//!   GET  /v1/hide/events     (WS upgrade)  -> host.subscribe_ui()  -> UiEvent frames
//!   GET  /v1/hide/events?after_seq=N (plain GET) -> host.ui_events -> [UiEvent]
//!   POST /v1/hide/connector  -> host.call_connector(id, method, p) -> Value
//!   POST /v1/hide/rpc        -> host.rpc(Method, params)           -> RpcResult
//!
//! T5: depends only on hide-backend + hide-core, NEVER hawking-core/serve.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        FromRequestParts, Query, Request, State,
    },
    http::{header, HeaderValue, Method, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use hide_backend::BackendHost;
use hide_core::api::{Intent, UiEvent, UiEventKind};
use hide_core::ids::SessionId;
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::sync::broadcast::error::RecvError;
use tower_http::cors::{AllowOrigin, CorsLayer};

/// Build the axum router for the HIDE transport, with the shared
/// [`BackendHost`] as state. Pure: no I/O, no binding — the bin (or a test)
/// drives it with `axum::serve` / `oneshot`.
pub fn router(host: Arc<BackendHost>) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/hide/intent", post(post_intent))
        // ONE path serves both wires: the WS upgrade (live Wire-B) and the
        // plain GET catch-up. The handler distinguishes them by the presence
        // of the upgrade extractor (Some => WS, None => plain GET).
        .route("/v1/hide/events", get(get_events))
        .route("/v1/hide/connector", post(post_connector))
        // Additive: the elevated Agent Server protocol surface (Bible sec 15).
        // Leaves /v1/hide/intent (Wire-A) untouched.
        .route("/v1/hide/rpc", post(post_rpc))
        // Permissive-localhost CORS so the browser at the Vite dev origin
        // (e.g. http://127.0.0.1:5273 or :5174) can reach this localhost
        // transport (127.0.0.1:8744). The OPTIONS preflight is answered by the
        // layer itself. WS upgrades do not use CORS, but a permissive origin
        // keeps the cross-origin browser path usable for the JSON routes.
        .layer(cors_layer())
        .with_state(host)
}

/// The origins allowed to reach this loopback transport: the Tauri webview and the Vite dev server.
/// A localhost service's real threat is the user's own browser, not a remote machine: with `Any`,
/// any website the user visits could POST intents to 127.0.0.1:8744 (running commands, reading and
/// writing files) and read the response. Locking the origin closes that drive-by surface. Override
/// the dev origin with `HIDE_ALLOW_ORIGIN` if the Vite port differs. WS upgrades do not use CORS.
fn allowed_origins() -> Vec<HeaderValue> {
    let mut raw = vec![
        "tauri://localhost".to_string(),
        "http://localhost:5273".to_string(),
        "http://127.0.0.1:5273".to_string(),
    ];
    if let Ok(extra) = std::env::var("HIDE_ALLOW_ORIGIN") {
        raw.extend(
            extra
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty()),
        );
    }
    raw.iter().filter_map(|o| o.parse().ok()).collect()
}

fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(AllowOrigin::list(allowed_origins()))
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE])
}

async fn healthz() -> &'static str {
    "ok"
}

/// Map a `hide_core` host error onto a JSON HTTP error. Every failure is
/// surfaced (never silently swallowed); the body mirrors the FE's expectation
/// of a readable `{ error: { message } }` shape.
fn host_error(status: StatusCode, message: impl Into<String>) -> Response {
    (
        status,
        Json(json!({ "error": { "message": message.into() } })),
    )
        .into_response()
}

// ── Wire-A: POST /v1/hide/intent ────────────────────────────────────────────

/// Deserialize the `Intent`, hand it to `host.handle_intent`, serialize the
/// `IntentAck`. A rejected ack (`accepted: false`) is a 200 body, not an HTTP
/// error — the FE surfaces it. Only a host-side failure becomes a 500.
async fn post_intent(State(host): State<Arc<BackendHost>>, Json(intent): Json<Intent>) -> Response {
    match host.handle_intent(intent).await {
        Ok(ack) => Json(ack).into_response(),
        Err(e) => host_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ── Wire-B: GET /v1/hide/events (WS upgrade) + ?after_seq catch-up ──────────

#[derive(Debug, Default, Deserialize)]
struct EventsQuery {
    /// Reconnect cursor for the plain-GET catch-up. Absent for the WS upgrade.
    after_seq: Option<u64>,
    /// Optional session filter for the catch-up replay.
    session_id: Option<String>,
    /// Optional cap on the number of buffered events returned.
    limit: Option<usize>,
}

/// Distinguish the WS upgrade from the plain GET by the presence of the
/// `Upgrade: websocket` header (the standard axum pattern for serving both on
/// one path). `WebSocketUpgrade` is not an optional extractor in axum 0.8, so
/// we take the whole request, sniff the header, and run the upgrade extractor
/// by hand only on a real upgrade. On a WS upgrade we forward the live
/// `subscribe_ui()` stream; on a plain GET we serve the `ui_events` pull
/// catch-up as a JSON array.
async fn get_events(State(host): State<Arc<BackendHost>>, req: Request) -> Response {
    if is_ws_upgrade(&req) {
        let (mut parts, _body) = req.into_parts();
        match WebSocketUpgrade::from_request_parts(&mut parts, &()).await {
            Ok(upgrade) => upgrade.on_upgrade(move |socket| forward_ui_events(socket, host)),
            // A malformed upgrade (bad version/key) is surfaced, not swallowed.
            Err(rej) => rej.into_response(),
        }
    } else {
        // Parse the catch-up query off the URI (Query as a standalone extractor
        // would consume the request; do it directly from the parts).
        let q: EventsQuery = Query::try_from_uri(req.uri())
            .map(|Query(q)| q)
            .unwrap_or_default();
        let session = q.session_id.map(SessionId::from);
        match host.ui_events(session, q.after_seq, q.limit).await {
            Ok(events) => Json(events).into_response(),
            Err(e) => host_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        }
    }
}

/// True when the request carries an `Upgrade: websocket` header — the marker
/// that separates the live WS subscription from the plain-GET catch-up.
fn is_ws_upgrade(req: &Request) -> bool {
    req.headers()
        .get(header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false)
}

/// Forward every `UiEvent` from the host's broadcast bus onto the socket as a
/// JSON text frame, until the socket closes. A `Lagged` signal is SURFACED as
/// an Error UiEvent frame (never silently dropped); a `Closed` bus ends the
/// stream cleanly. A serialization failure also surfaces as an Error frame.
async fn forward_ui_events(mut socket: WebSocket, host: Arc<BackendHost>) {
    let mut rx = host.subscribe_ui();
    loop {
        match rx.recv().await {
            Ok(event) => {
                let text = match serde_json::to_string(&event) {
                    Ok(text) => text,
                    Err(e) => error_frame("serialize", &e.to_string()),
                };
                if socket.send(Message::Text(text.into())).await.is_err() {
                    break; // client went away
                }
            }
            Err(RecvError::Lagged(skipped)) => {
                // Surface the drop rather than silently swallowing it: the FE
                // can then re-sync via GET /v1/hide/events?after_seq=N.
                let frame = error_frame(
                    "lagged",
                    &format!(
                        "subscriber lagged; {skipped} events dropped, reconnect with after_seq"
                    ),
                );
                if socket.send(Message::Text(frame.into())).await.is_err() {
                    break;
                }
            }
            Err(RecvError::Closed) => break, // bus closed: end the stream cleanly
        }
    }
}

/// Build an `Error` [`UiEvent`] JSON frame so a transport-level problem reaches
/// the FE through the same typed channel as real events.
fn error_frame(code: &str, message: &str) -> String {
    let event = UiEvent {
        seq: 0,
        session_id: None,
        kind: UiEventKind::Error {
            code: code.to_string(),
            message: message.to_string(),
        },
    };
    // UiEvent serializes infallibly here (no non-string map keys); fall back to
    // a hand-built object on the impossible error so we still emit something.
    serde_json::to_string(&event).unwrap_or_else(|_| {
        json!({"seq":0,"session_id":null,"kind":{"type":"error","data":{"code":code,"message":message}}})
            .to_string()
    })
}

// ── POST /v1/hide/connector ─────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ConnectorReq {
    id: String,
    method: String,
    #[serde(default)]
    params: Value,
}

/// Dispatch `{id, method, params}` to `host.call_connector` and return the JSON
/// result. A connector error (unknown id/method, bad params) is surfaced as a
/// 400 with the host's message.
async fn post_connector(
    State(host): State<Arc<BackendHost>>,
    Json(req): Json<ConnectorReq>,
) -> Response {
    // This route is a READ channel. A durable write has to arrive as an Intent, which is where the
    // approval gate and the command catalog are. The gate is an ALLOWLIST of the read methods, so
    // an arm nobody classified is refused rather than served: the blocklist it replaces missed the
    // code-index mutation, the absolute-path read that escapes the workspace confinement, and the
    // context compile that upserts durable memory.
    if !hide_backend::connectors::connector_method_is_read(&req.method) {
        return host_error(
            StatusCode::FORBIDDEN,
            format!(
                "{} is not a connector read, so it is not reachable over this route: send the matching intent",
                req.method
            ),
        );
    }
    match host.call_connector(&req.id, &req.method, req.params).await {
        Ok(value) => Json(value).into_response(),
        Err(e) => host_error(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

// ── Elevated protocol: POST /v1/hide/rpc ────────────────────────────────────

/// A hide-protocol Method envelope (Bible sec 15): a correlation `id`, a
/// slash-namespaced `method`, and opaque `params`. This is a tolerant superset
/// of `hide_protocol::Request` (the `id` is optional here so a client may omit
/// it), keyed on the protocol's own closed [`hide_protocol::protocol::Method`]
/// enum so an unrecognized method string is rejected at deserialize time.
#[derive(Debug, Deserialize)]
struct RpcEnvelope {
    #[serde(default)]
    #[allow(dead_code)]
    id: Option<String>,
    method: hide_protocol::protocol::Method,
    #[serde(default)]
    params: Value,
}

/// Deserialize the Method envelope, dispatch `(method, params)` through
/// `host.rpc`, and return the TYPED [`hide_backend::RpcResult`] as a 200 body.
///
/// Additive: this reaches the elevated Agent Server protocol without touching
/// `/v1/hide/intent`. A recognized-but-deferred method returns a typed
/// `NotImplemented` in a 200 body (not a transport error) and a dispatch the
/// host rejects returns a typed `Error` body, mirroring how a rejected intent
/// ack is a 200 body. Only a malformed envelope (bad JSON / unknown method
/// string) becomes a 4xx via axum's `Json` rejection.
async fn post_rpc(State(host): State<Arc<BackendHost>>, Json(env): Json<RpcEnvelope>) -> Response {
    let result = host.rpc(env.method, env.params).await;
    Json(result).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use hide_backend::BackendServices;
    use hide_core::api::IntentAck;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_core::types::Decision;
    use http_body_util::BodyExt;
    use tower::ServiceExt; // oneshot

    fn host_for_test() -> Arc<BackendHost> {
        // Unique per call: cargo runs tests in parallel, and a now_ms()-only name collides
        // when two tests start in the same millisecond, sharing a host and leaking events.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let uniq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("hide_serve_{}_{}", now_ms(), uniq));
        let mut config = HideConfig::for_workspace(&dir);
        // Allow shell so the RunCommand intent round-trips end to end.
        config.security.shell_default = Decision::Allow;
        let host = BackendHost::from_services(BackendServices::open(config).unwrap()).unwrap();
        Arc::new(host)
    }

    #[test]
    fn router_builds() {
        // The router constructs over a real host without panicking — the
        // load-bearing "it wires" smoke test.
        let _router = router(host_for_test());
    }

    #[tokio::test]
    async fn intent_round_trips_through_the_handler() {
        let app = router(host_for_test());
        // A valid RunCommand intent flows FE -> POST /v1/hide/intent ->
        // host.handle_intent -> IntentAck, accepted.
        let body = serde_json::to_vec(&Intent::RunCommand {
            argv: vec!["printf".to_string(), "hide".to_string()],
            cwd: None,
        })
        .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/intent")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let ack: IntentAck = serde_json::from_slice(&bytes).unwrap();
        assert!(ack.accepted, "valid RunCommand must be accepted");
    }

    #[tokio::test]
    async fn cors_allows_the_app_origin_and_blocks_foreign_sites() {
        // The Tauri webview origin is granted CORS: the header is echoed back.
        let resp = router(host_for_test())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/healthz")
                    .header("origin", "tauri://localhost")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            resp.headers()
                .get("access-control-allow-origin")
                .and_then(|v| v.to_str().ok()),
            Some("tauri://localhost"),
            "the app origin is allowed to read responses"
        );

        // An arbitrary website gets NO CORS grant, so the browser blocks it from reading the
        // response. This is the drive-by RCE surface (any page POSTing to 127.0.0.1:8744), closed.
        let resp = router(host_for_test())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/healthz")
                    .header("origin", "https://evil.example.com")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert!(
            resp.headers().get("access-control-allow-origin").is_none(),
            "a foreign origin must not receive a CORS grant"
        );
    }

    #[tokio::test]
    async fn empty_intent_is_rejected_as_200_body() {
        let app = router(host_for_test());
        // An empty-argv RunCommand is rejected by the host: accepted:false with
        // an HTTP 200 (a rejected ack is a body, not a transport error).
        let body = serde_json::to_vec(&Intent::RunCommand {
            argv: vec![],
            cwd: None,
        })
        .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/intent")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let ack: IntentAck = serde_json::from_slice(&bytes).unwrap();
        assert!(!ack.accepted, "empty RunCommand must be rejected");
        assert!(ack.message.is_some(), "rejection must carry a reason");
    }

    /// The connector route is a READ channel. Every mutating or workspace-escaping arm is refused
    /// here and has to arrive as an Intent, where the permission engine and the gate are. The last
    /// three rows are the ones the old blocklist missed: the code index mutation, the absolute-path
    /// read that walks straight past the `fs` confinement, and the compile that upserts memory.
    #[tokio::test]
    async fn connector_route_refuses_the_durable_write_methods() {
        for (id, method) in [
            ("personalization", "records.append"),
            ("research", "runs.append"),
            ("fs", "write_file"),
            ("code_index", "file.add_text"),
            ("code_index", "file.index"),
            ("context", "compile"),
        ] {
            let app = router(host_for_test());
            let body = serde_json::to_vec(&json!({ "id": id, "method": method, "params": {} })).unwrap();
            let resp = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/hide/connector")
                        .header("content-type", "application/json")
                        .body(Body::from(body))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::FORBIDDEN,
                "{id}.{method} must not be reachable from the app transport"
            );
        }
    }

    #[tokio::test]
    async fn connector_route_dispatches() {
        let app = router(host_for_test());
        // The runtime connector lists model roles; the route forwards
        // {id, method, params} to host.call_connector and returns the Value.
        let body = serde_json::to_vec(&json!({
            "id": "runtime",
            "method": "roles.list",
            "params": {},
        }))
        .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/connector")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert!(
            value["roles"]
                .as_array()
                .map(|a| !a.is_empty())
                .unwrap_or(false),
            "runtime.roles.list must return a non-empty roles array"
        );
    }

    #[tokio::test]
    async fn unknown_connector_surfaces_an_error() {
        let app = router(host_for_test());
        // An allowlisted READ method, so the 400 comes from the unknown connector id and not from
        // the read gate (which answers 403 and would mask this).
        let body = serde_json::to_vec(&json!({
            "id": "does_not_exist",
            "method": "health",
            "params": {},
        }))
        .unwrap();
        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/connector")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        // A failure is surfaced, not swallowed.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn events_plain_get_returns_a_json_array() {
        let app = router(host_for_test());
        // The plain GET (no upgrade header) is the catch-up: it returns a JSON
        // array (empty on a fresh host), NOT a WS upgrade.
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/hide/events?after_seq=0")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let events: Vec<UiEvent> = serde_json::from_slice(&bytes).unwrap();
        assert!(events.is_empty(), "fresh host has no buffered events");
    }

    /// THE RELOAD. A browser reload has exactly two ways in: the catch-up GET and the connect-time
    /// connector read. Everything the shell needs to come back has to arrive on one of them, so this
    /// drives the setup over `/v1/hide/intent` and asserts ONLY over those two client routes - no
    /// host method is called to read a property the client has to be able to read for itself.
    ///
    /// Covers three defects that shared one shape (live-bus-only state): a replayed session showed
    /// no transcript, a sealed checkpoint's id was lost so seven history verbs died, and a write
    /// lease stayed in force while its indicator vanished.
    #[tokio::test]
    async fn a_reload_recovers_the_transcript_the_checkpoint_and_the_write_lease() {
        let host = host_for_test();
        let session = host.services.session();

        // The user's line. No model is served here (the preview host has none either), so the turn
        // itself cannot run - but the intent is durable, and that is what a reopened session renders.
        let body = serde_json::to_vec(&Intent::SubmitTurn {
            session_id: session.clone(),
            text: "render me after a reload".to_string(),
            attachments: vec![],
        })
        .unwrap();
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/intent")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // A checkpoint, over the same intent route the palette and the timeline use.
        let body = serde_json::to_vec(&json!({
            "type": "custom",
            "data": {
                "name": "checkpoint_create",
                "payload": { "session_id": session.as_str(), "label": "before the reload" },
            },
        }))
        .unwrap();
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/intent")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // A lease in force. Granting it is setup (the grant path has its own coverage); what is
        // under test is whether a client that was not connected when it was granted can SEE it.
        hide_backend::tools::install_write_lease(hide_backend::tools::WriteLease {
            lease_id: "gr_reload".to_string(),
            repo_id: "repo_reload".to_string(),
            session_id: Some(session.as_str().to_string()),
            run_id: None,
            scopes: vec![std::path::PathBuf::from("/tmp/hide-reload-scope")],
            granted_ms: now_ms(),
        });

        // The intent handlers spawn, so give them a tick to land in the durable log.
        for _ in 0..50 {
            if !host
                .services
                .event_log
                .scan(Some(session.clone()), None, None)
                .await
                .unwrap()
                .iter()
                .any(|e| e.kind == "checkpoint.created")
            {
                tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            } else {
                break;
            }
        }

        // ROUTE 1: the catch-up a fresh tab makes, from seq 0, scoped to the session it renders.
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(format!(
                        "/v1/hide/events?after_seq=0&session_id={}",
                        session.as_str()
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let frames: Vec<serde_json::Value> = serde_json::from_slice(&bytes).unwrap();
        let customs: Vec<&serde_json::Value> = frames
            .iter()
            .filter(|f| f["kind"]["type"] == "custom")
            .map(|f| &f["kind"]["data"])
            .collect();
        assert!(
            customs.iter().any(|d| d["kind"] == "transcript_message"
                && d["role"] == "user"
                && d["text"] == "render me after a reload"
                && d["event_id"].is_string()),
            "the catch-up must carry the transcript, deduped on its durable event id: {customs:?}"
        );
        assert!(
            customs.iter().any(|d| d["kind"] == "checkpoint_created"
                && d["record"]["checkpoint_id"].as_str().is_some_and(|id| id.starts_with("ckpt_"))),
            "the catch-up must carry the sealed checkpoint id: {customs:?}"
        );

        // ROUTE 2: the connect-time read (store.ts connectStore). The lease comes back on it.
        let body = serde_json::to_vec(&json!({ "id": "home", "method": "digest", "params": {} })).unwrap();
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/connector")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let digest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(digest["status"]["write_lease"]["active"], json!(true));
        assert_eq!(digest["status"]["write_lease"]["lease_id"], json!("gr_reload"));
        // and the same read never invents a model on a host that has none.
        assert_eq!(digest["home"]["digest"]["favorite_model"], json!("unknown"));

        // Revoked, and the next fresh client sees it cleared rather than a lease nobody holds.
        hide_backend::tools::revoke_write_lease("test");
        let body = serde_json::to_vec(&json!({ "id": "home", "method": "digest", "params": {} })).unwrap();
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/connector")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let digest: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(digest["status"]["write_lease"]["active"], json!(false));
    }

    #[tokio::test]
    async fn rpc_route_dispatches_a_method_envelope() {
        let host = host_for_test();
        let session = host.services.session();

        // A goal/set Method envelope flows FE -> POST /v1/hide/rpc -> host.rpc ->
        // a typed RpcResult (status "ok").
        let body = serde_json::to_vec(&json!({
            "id": "req_1",
            "method": "goal/set",
            "params": {
                "session": session.as_str(),
                "condition": "green ci",
                "acceptance": ["test"],
            },
        }))
        .unwrap();
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/rpc")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["status"], json!("ok"), "goal/set is a typed Ok result");
        assert_eq!(value["method"], json!("goal/set"));

        // A recognized-but-deferred method returns a typed NotImplemented, still
        // a 200 body (not a transport error) and never a panic.
        let body = serde_json::to_vec(&json!({
            "id": "req_2",
            "method": "artifact/get",
            "params": {},
        }))
        .unwrap();
        let resp = router(host.clone())
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/hide/rpc")
                    .header("content-type", "application/json")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            value["status"],
            json!("not_implemented"),
            "a deferred method is typed, not a 500"
        );
    }
}
