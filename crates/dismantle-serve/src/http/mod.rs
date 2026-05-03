//! axum routes for OpenAI-compatible endpoints:
//!   POST /v1/chat/completions   (SSE streaming)
//!   POST /v1/completions        (legacy, also SSE)
//!   GET  /v1/models
//!   GET  /healthz
//!   GET  /metrics               (Prometheus textfile)

use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse,
    },
    routing::{get, post},
    Json, Router,
};
use dismantle_core::{Engine, GenerateRequest, SamplingParams, StreamEvent};
use futures::stream::Stream;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<Mutex<Box<dyn Engine>>>,
    pub model_arch: String,
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

async fn metrics() -> &'static str {
    // Real metrics arrive in Phase 5.
    "# dismantle_metrics 1\n"
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

async fn chat_completions(
    State(s): State<AppState>,
    Json(req): Json<ChatReq>,
) -> impl IntoResponse {
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

async fn completions(
    State(s): State<AppState>,
    Json(req): Json<CompletionReq>,
) -> impl IntoResponse {
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
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::task::spawn_blocking(move || {
        let mut engine = state.engine.lock();
        let mut sink = |ev: StreamEvent| match ev {
            StreamEvent::Token { text, .. } => {
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
                let _ = tx.blocking_send(Ok(Event::default().data(chunk.to_string())));
            }
            StreamEvent::Done { .. } => {
                let _ = tx.blocking_send(Ok(Event::default().data("[DONE]")));
            }
        };
        let _ = engine.generate(req, &mut sink);
    });
    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

async fn json_full_response(
    state: AppState,
    req: GenerateRequest,
    chat: bool,
) -> Json<serde_json::Value> {
    let res = tokio::task::spawn_blocking(move || -> serde_json::Value {
        let mut engine = state.engine.lock();
        let mut text = String::new();
        let mut sink = |ev: StreamEvent| {
            if let StreamEvent::Token { text: t, .. } = ev {
                text.push_str(&t);
            }
        };
        let _ = engine.generate(req, &mut sink);
        if chat {
            serde_json::json!({
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]
            })
        } else {
            serde_json::json!({
                "object": "text_completion",
                "choices": [{"index": 0, "text": text}]
            })
        }
    })
    .await
    .unwrap_or(serde_json::json!({"error": "task panicked"}));
    Json(res)
}
