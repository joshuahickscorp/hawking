//! axum routes for OpenAI-compatible endpoints:
//!   POST /v1/chat/completions   (SSE streaming)
//!   POST /v1/completions        (legacy, also SSE)
//!   GET  /v1/models
//!   GET  /healthz
//!   GET  /metrics               (Prometheus textfile)

use axum::{
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use crate::batch::driver::BatchDriver;
use dismantle_core::{Engine, GenerateRequest, SamplingParams};
use futures::stream::Stream;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::mpsc as async_mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Per-slot token channel item: `Ok(text)` for each generated token,
/// `Err(())` to signal stream end (EOS, max_tokens reached, or error).
type SlotToken = Result<String, ()>;

/// Structured, OpenAI-compatible error.
///
/// Serializes to `{"error": {"message": ..., "type": ..., "code": ...}}` and
/// carries a stable HTTP status. The `code` field is a machine-readable,
/// stable token (see the constants below); the `type` field mirrors OpenAI's
/// coarse error families (`invalid_request_error`, `internal_error`).
#[derive(Debug, Clone)]
pub struct ApiError {
    status: StatusCode,
    message: String,
    error_type: &'static str,
    code: &'static str,
}

impl ApiError {
    /// Body could not be parsed as the expected JSON shape (syntax error,
    /// wrong types, or a missing required field that serde rejects).
    pub fn invalid_json(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            code: "invalid_json",
        }
    }

    /// A required parameter was syntactically present but semantically empty
    /// (e.g. `messages: []` or an empty `prompt`).
    pub fn missing_parameter(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            error_type: "invalid_request_error",
            code: "missing_required_parameter",
        }
    }

    /// Generation failed inside the engine, or the worker task panicked.
    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
            error_type: "internal_error",
            code: "internal_error",
        }
    }

    fn to_body(&self) -> serde_json::Value {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
            }
        })
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.to_body())).into_response()
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): {}", self.code, self.status, self.message)
    }
}

impl std::error::Error for ApiError {}

/// Parse a request body into `T`, mapping any serde failure to a structured
/// [`ApiError`] with the `invalid_json` code. Centralizes the malformed-input
/// path so every route reports errors with the same machine-readable shape.
fn parse_json<T: serde::de::DeserializeOwned>(body: &Bytes) -> Result<T, ApiError> {
    serde_json::from_slice::<T>(body)
        .map_err(|e| ApiError::invalid_json(format!("invalid request body: {e}")))
}

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Box<dyn Engine>>>,
    /// Continuous-batching driver — shared with the background decode loop.
    /// HTTP handlers take this lock briefly for admit only.
    pub driver: Arc<Mutex<BatchDriver>>,
    /// Per-slot SSE token senders. The background loop writes here;
    /// `sse_response` reads. Keyed by stable slot_id.
    pub slot_senders: Arc<Mutex<HashMap<u32, async_mpsc::Sender<SlotToken>>>>,
    /// Requests waiting for a free batch slot. Bounded at `max_batch * 8`.
    /// Tuple: (request, token_sender, is_chat_format).
    pub wait_queue: Arc<Mutex<VecDeque<(GenerateRequest, async_mpsc::Sender<SlotToken>, bool)>>>,
    pub model_arch: String,
    pub max_batch: usize,
    pub requests_admitted: Arc<AtomicU64>,
    pub tokens_generated: Arc<AtomicU64>,
    pub requests_queued: Arc<AtomicU64>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/metrics", get(metrics))
        .with_state(state)
}

async fn healthz() -> &'static str {
    "ok"
}

async fn metrics(State(s): State<AppState>) -> String {
    let admitted = s.requests_admitted.load(Ordering::Relaxed);
    let tokens = s.tokens_generated.load(Ordering::Relaxed);
    let driver = s.driver.lock();
    let active = driver.scheduler.active_count();
    let queued = s.wait_queue.lock().len();
    let lane = driver.lane_stats.clone();
    drop(driver);
    format!(
        "# HELP dismantle_requests_admitted_total Requests successfully admitted to a batch slot\n\
         # TYPE dismantle_requests_admitted_total counter\n\
         dismantle_requests_admitted_total {admitted}\n\
         # HELP dismantle_tokens_generated_total Tokens generated across all slots\n\
         # TYPE dismantle_tokens_generated_total counter\n\
         dismantle_tokens_generated_total {tokens}\n\
         # HELP dismantle_active_slots Current number of active decode slots\n\
         # TYPE dismantle_active_slots gauge\n\
         dismantle_active_slots {active}\n\
         # HELP dismantle_queued_requests Requests waiting for a free slot\n\
         # TYPE dismantle_queued_requests gauge\n\
         dismantle_queued_requests {queued}\n\
         # HELP dismantle_greedy_decode_steps_total Decode steps routed through the token-only greedy lane\n\
         # TYPE dismantle_greedy_decode_steps_total counter\n\
         dismantle_greedy_decode_steps_total {}\n\
         # HELP dismantle_logits_decode_steps_total Decode steps that materialized full logits\n\
         # TYPE dismantle_logits_decode_steps_total counter\n\
         dismantle_logits_decode_steps_total {}\n\
         # HELP dismantle_gpu_readback_bytes_total Cumulative GPU→CPU readback bytes\n\
         # TYPE dismantle_gpu_readback_bytes_total counter\n\
         dismantle_gpu_readback_bytes_total {}\n",
        lane.greedy_steps, lane.logits_steps, lane.readback_bytes,
    )
}

#[derive(Serialize)]
struct ModelInfo {
    id: String,
    object: &'static str,
}

#[derive(Serialize)]
struct ListModels {
    object: &'static str,
    data: Vec<ModelInfo>,
}

async fn list_models(State(s): State<AppState>) -> Json<ListModels> {
    let id = s.engine.lock().model_id().to_string();
    Json(ListModels {
        object: "list",
        data: vec![ModelInfo {
            id,
            object: "model",
        }],
    })
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatReq {
    #[allow(dead_code)]
    model: Option<String>,
    messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

fn default_max_tokens() -> usize {
    256
}

#[derive(Deserialize)]
struct CompletionReq {
    #[allow(dead_code)]
    model: Option<String>,
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stream: bool,
}

async fn chat_completions(State(s): State<AppState>, body: Bytes) -> Response {
    let req: ChatReq = match parse_json(&body) {
        Ok(req) => req,
        Err(e) => return e.into_response(),
    };
    if req.messages.is_empty() {
        return ApiError::missing_parameter("'messages' must contain at least one message")
            .into_response();
    }
    let prompt = render_chat(&req.messages, &s.model_arch);
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.0,
        seed: req.seed,
    };
    let gen = GenerateRequest {
        prompt,
        max_new_tokens: req.max_tokens,
        sampling,
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
    };
    if req.stream {
        sse_response(s, gen, /*chat=*/ true).into_response()
    } else {
        json_full_response(s, gen, /*chat=*/ true)
            .await
            .into_response()
    }
}

async fn completions(State(s): State<AppState>, body: Bytes) -> Response {
    let req: CompletionReq = match parse_json(&body) {
        Ok(req) => req,
        Err(e) => return e.into_response(),
    };
    if req.prompt.is_empty() {
        return ApiError::missing_parameter("'prompt' must not be empty").into_response();
    }
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.0,
        seed: req.seed,
    };
    let gen = GenerateRequest {
        prompt: req.prompt,
        max_new_tokens: req.max_tokens,
        sampling,
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
    };
    if req.stream {
        sse_response(s, gen, /*chat=*/ false).into_response()
    } else {
        json_full_response(s, gen, /*chat=*/ false)
            .await
            .into_response()
    }
}

fn render_chat(msgs: &[ChatMessage], model_arch: &str) -> String {
    match model_arch {
        "deepseek2" => render_chat_deepseek(msgs),
        a if a.starts_with("qwen2") => render_chat_qwen2(msgs),
        _ => render_chat_generic(msgs),
    }
}

fn render_chat_deepseek(msgs: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in msgs {
        match m.role.as_str() {
            "system" => s.push_str(&format!("{}\n\n", m.content)),
            "user" => s.push_str(&format!("User: {}\n\n", m.content)),
            "assistant" => s.push_str(&format!("Assistant: {}\n\n", m.content)),
            _ => {}
        }
    }
    s.push_str("Assistant:");
    s
}

fn render_chat_qwen2(msgs: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            m.role, m.content
        ));
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

fn render_chat_generic(msgs: &[ChatMessage]) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str(&format!("<|{}|>\n{}\n", m.role, m.content));
    }
    s.push_str("<|assistant|>\n");
    s
}

fn sse_response(
    state: AppState,
    req: GenerateRequest,
    chat: bool,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // SSE → client channel (receives formatted SSE events).
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
    // Token channel: the background decode loop sends raw text fragments here.
    let (tok_tx, mut tok_rx) = async_mpsc::channel::<SlotToken>(256);

    // Admit the request under a short lock (tokenize + slot assignment only).
    // The engine lock is held only for the encoding step, not for generation.
    let slot_id_opt = {
        let engine = state.engine.lock();
        let mut driver = state.driver.lock();
        driver.admit(&**engine, req.clone()).ok().flatten()
    };

    if let Some(slot_id) = slot_id_opt {
        // Slot available immediately — register sender and start serving.
        state.requests_admitted.fetch_add(1, Ordering::Relaxed);
        state.slot_senders.lock().insert(slot_id, tok_tx);
    } else {
        // No free slot — queue the request for deferred admission.
        let queue_cap = state.max_batch * 8;
        if state.wait_queue.lock().len() >= queue_cap {
            // Queue is also full — error immediately.
            let sse_tx2 = sse_tx.clone();
            tokio::spawn(async move {
                let body = serde_json::json!({
                    "error": {"message": "server busy — no batch slot available",
                              "type": "server_error", "code": "slot_exhausted"}
                });
                let _ = sse_tx2.send(Ok(Event::default().data(body.to_string()))).await;
            });
            return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
        }
        state.requests_queued.fetch_add(1, Ordering::Relaxed);
        state.wait_queue.lock().push_back((req, tok_tx, chat));
        // The SSE forwarder below is still spawned and will stream tokens once
        // the request is admitted from the queue when a slot frees.
    };

    // Forward raw token strings from the per-slot channel to SSE events.
    tokio::spawn(async move {
        while let Some(item) = tok_rx.recv().await {
            match item {
                Ok(text) => {
                    let chunk = if chat {
                        serde_json::json!({
                            "choices": [{"delta": {"content": text}, "index": 0}],
                            "object": "chat.completion.chunk",
                        })
                    } else {
                        serde_json::json!({
                            "choices": [{"text": text, "index": 0}],
                            "object": "text_completion",
                        })
                    };
                    if sse_tx.send(Ok(Event::default().data(chunk.to_string()))).await.is_err() {
                        break;
                    }
                }
                Err(()) => {
                    let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    break;
                }
            }
        }
    });

    Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
}

async fn json_full_response(
    state: AppState,
    req: GenerateRequest,
    chat: bool,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Admit under a short lock (tokenize + slot assignment only) — does NOT hold
    // the engine mutex for the full generation.
    let slot_id = {
        let engine = state.engine.lock();
        let mut driver = state.driver.lock();
        driver
            .admit(&**engine, req)
            .map_err(|e| ApiError::internal(format!("admit failed: {e}")))?
            .ok_or_else(|| ApiError::internal("server busy — no batch slot available"))?
    };

    // std::sync::mpsc (not tokio) because we block-wait inside spawn_blocking.
    let (tok_tx, tok_rx) = std::sync::mpsc::channel::<SlotToken>();
    // The background loop expects a tokio::sync::mpsc::Sender; wrap via an async
    // bridge: allocate a small tokio channel, spawn a task that forwards into our
    // std channel.
    let (async_tx, mut async_rx) = async_mpsc::channel::<SlotToken>(256);
    state.slot_senders.lock().insert(slot_id, async_tx);

    // Bridge task: forward from the tokio channel into the std channel.
    let tok_tx2 = tok_tx.clone();
    tokio::spawn(async move {
        while let Some(item) = async_rx.recv().await {
            // If the receiver side (spawn_blocking below) is gone, stop forwarding.
            if tok_tx2.send(item).is_err() {
                break;
            }
        }
    });
    drop(tok_tx); // only tok_tx2 (owned by the bridge) keeps the sender alive

    // Block-wait in a dedicated thread so we don't hold any mutex.
    let res = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let mut text = String::new();
        for item in tok_rx {
            match item {
                Ok(t) => text.push_str(&t),
                Err(()) => break, // EOS sentinel
            }
        }
        let body = if chat {
            serde_json::json!({
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]
            })
        } else {
            serde_json::json!({
                "object": "text_completion",
                "choices": [{"index": 0, "text": text}]
            })
        };
        Ok(body)
    })
    .await
    .map_err(|_| ApiError::internal("generation task panicked"))?
    .map_err(ApiError::internal)?;
    Ok(Json(res))
}
