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
//!
//! T5: depends only on hide-backend + hide-core, NEVER hawking-core/serve.

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        FromRequestParts, Query, Request, State,
    },
    http::{header, StatusCode},
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
use tower_http::cors::{Any, CorsLayer};

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
        // Permissive-localhost CORS so the browser at the Vite dev origin
        // (e.g. http://127.0.0.1:5273 or :5174) can reach this localhost
        // transport (127.0.0.1:8744). The OPTIONS preflight is answered by the
        // layer itself. WS upgrades do not use CORS, but a permissive origin
        // keeps the cross-origin browser path usable for the JSON routes.
        .layer(cors_layer())
        .with_state(host)
}

/// A permissive CORS layer for the localhost dev surface: any origin, the
/// methods the FE uses (GET/POST + the OPTIONS preflight), and any request
/// header. This is intentionally open because the transport binds to loopback
/// only (no cross-machine exposure to defend here), and the Vite dev server's
/// origin/port is not fixed.
fn cors_layer() -> CorsLayer {
    CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any)
}

async fn healthz() -> &'static str {
    "ok"
}

/// Map a `hide_core` host error onto a JSON HTTP error. Every failure is
/// surfaced (never silently swallowed); the body mirrors the FE's expectation
/// of a readable `{ error: { message } }` shape.
fn host_error(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": { "message": message.into() } }))).into_response()
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
                    &format!("subscriber lagged; {skipped} events dropped, reconnect with after_seq"),
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
    match host.call_connector(&req.id, &req.method, req.params).await {
        Ok(value) => Json(value).into_response(),
        Err(e) => host_error(StatusCode::BAD_REQUEST, e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use hide_core::api::IntentAck;
    use hide_core::config::HideConfig;
    use hide_core::ids::now_ms;
    use hide_core::types::Decision;
    use hide_backend::BackendServices;
    use http_body_util::BodyExt;
    use tower::ServiceExt; // oneshot

    fn host_for_test() -> Arc<BackendHost> {
        let dir = std::env::temp_dir().join(format!("hide_serve_{}", now_ms()));
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
            value["roles"].as_array().map(|a| !a.is_empty()).unwrap_or(false),
            "runtime.roles.list must return a non-empty roles array"
        );
    }

    #[tokio::test]
    async fn unknown_connector_surfaces_an_error() {
        let app = router(host_for_test());
        let body = serde_json::to_vec(&json!({
            "id": "does_not_exist",
            "method": "noop",
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
}
