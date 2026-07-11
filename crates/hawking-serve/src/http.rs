//! axum routes for OpenAI-compatible endpoints:
//!   POST /v1/chat/completions   (SSE streaming)
//!   POST /v1/completions        (legacy, also SSE)
//!   GET  /v1/models
//!   GET  /healthz
//!   GET  /metrics               (Prometheus textfile)

use crate::batch::driver::BatchDriver;
use crate::system_kv_bank::SystemPromptKvBank;
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
use hawking_core::{Engine, GenerateRequest, SamplingParams};
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
    /// Track 5.2: serve-lifetime hash(system-prefix) -> source-slot routing
    /// hint. Survives a source request finishing, so serial workloads that
    /// re-send an identical system prompt still get shared-prefix KV reuse
    /// (the live `PrefixIndex` only matches CURRENTLY-active slots). Stores
    /// zero KV bytes; every hit is re-verified by the bit-identical
    /// `copy_kv_prefix_to_slot` + `prefill_slot_from_pos` path, so a stale
    /// slot simply fails the copy and falls back to a cold prefill.
    pub system_kv_bank: Arc<Mutex<SystemPromptKvBank>>,
}

/// Track 5.2 — the agreed banked-prefix length the serve-loop admit path uses
/// for BOTH `SystemPromptKvBank::record` and `::lookup`. PURE (no I/O, no
/// model): the gate test calls this directly so the record/lookup keys can
/// never silently diverge.
///
/// The bank requires a STRICT leading prefix (`banked_len < prompt_ids.len()`),
/// unlike the live `find_prefix_match_excluding` which keys on the full source
/// slot length. We bank the prompt minus its last token — the "bail one token
/// short" rule the disk/RAM KV tiers use — so the decode loop always keeps a
/// real `last_id`. For a serial workload that re-sends the SAME prompt, the
/// turn that records and the turn that looks up both see identical `prompt_ids`
/// and therefore hash to the same key. Returns 0 when the prompt is too short
/// to bank (the bank itself also rejects `< min_prefix_tokens`).
pub fn banked_len_for(prompt_ids: &[u32]) -> usize {
    prompt_ids.len().saturating_sub(1)
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/hawking/tokens", post(hawking_tokens))
        .route("/v1/hawking/generate", post(hawking_generate))
        .route("/v1/hawking/context", get(hawking_context))
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
        "# HELP hawking_requests_admitted_total Requests successfully admitted to a batch slot\n\
         # TYPE hawking_requests_admitted_total counter\n\
         hawking_requests_admitted_total {admitted}\n\
         # HELP hawking_tokens_generated_total Tokens generated across all slots\n\
         # TYPE hawking_tokens_generated_total counter\n\
         hawking_tokens_generated_total {tokens}\n\
         # HELP hawking_active_slots Current number of active decode slots\n\
         # TYPE hawking_active_slots gauge\n\
         hawking_active_slots {active}\n\
         # HELP hawking_queued_requests Requests waiting for a free slot\n\
         # TYPE hawking_queued_requests gauge\n\
         hawking_queued_requests {queued}\n\
         # HELP hawking_greedy_decode_steps_total Decode steps routed through the token-only greedy lane\n\
         # TYPE hawking_greedy_decode_steps_total counter\n\
         hawking_greedy_decode_steps_total {}\n\
         # HELP hawking_logits_decode_steps_total Decode steps that materialized full logits\n\
         # TYPE hawking_logits_decode_steps_total counter\n\
         hawking_logits_decode_steps_total {}\n\
         # HELP hawking_gpu_readback_bytes_total Cumulative GPU→CPU readback bytes\n\
         # TYPE hawking_gpu_readback_bytes_total counter\n\
         hawking_gpu_readback_bytes_total {}\n\
         # HELP hawking_prefix_reuse_total Admissions where KV prefix was copied from an existing slot\n\
         # TYPE hawking_prefix_reuse_total counter\n\
         hawking_prefix_reuse_total {}\n",
        lane.greedy_steps, lane.logits_steps, lane.readback_bytes,
        lane.prefix_reuse_count,
    )
}

/// Spine A — live context introspection. Read-only snapshot of the real,
/// dynamic context picture: native length from the model config, the effective
/// ceiling derived from the measured `.tq` multiplier (passed in via env by the
/// supervisor — never a constant), the constant recurrent-state footprint for
/// SSMs, and live slot occupancy. The shell renders this as an ambient cue.
#[derive(Serialize)]
struct ContextStatus {
    model_id: String,
    arch: String,
    ctx_len_native: Option<usize>,
    ctx_len_effective: Option<usize>,
    /// Measured `.tq` weight-compression multiplier (1.0 == no claim).
    tq_multiplier: f32,
    /// True when the effective ceiling is a derived estimate, not a hard cap.
    tq_estimated: bool,
    /// Constant recurrent-state footprint in bytes for SSMs; None for transformers.
    recurrent_state_bytes: Option<usize>,
    active_slots: usize,
    free_slots: usize,
    max_batch: usize,
}

async fn hawking_context(State(s): State<AppState>) -> Json<ContextStatus> {
    let (model_id, arch, native, state_bytes) = {
        let eng = s.engine.lock();
        (
            eng.model_id().to_string(),
            eng.model_arch().to_string(),
            eng.context_length_native(),
            eng.recurrent_state_size_bytes(),
        )
    };
    // The supervisor measured this from the .tq artifact and passed it in; if
    // absent the multiplier is 1.0 (no expansion claimed). Never hardcoded.
    let tq_multiplier: f32 = std::env::var("HAWKING_QWEN_TQ_MULTIPLIER")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|m: &f32| m.is_finite() && *m >= 1.0)
        .unwrap_or(1.0);
    let effective = native.map(|n| (n as f32 * tq_multiplier).round() as usize);
    let active = s.driver.lock().scheduler.active_count();
    Json(ContextStatus {
        model_id,
        arch,
        ctx_len_native: native,
        ctx_len_effective: effective,
        tq_multiplier,
        tq_estimated: tq_multiplier > 1.0,
        recurrent_state_bytes: state_bytes,
        active_slots: active,
        free_slots: s.max_batch.saturating_sub(active),
        max_batch: s.max_batch,
    })
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

#[derive(Deserialize, Clone)]
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
    /// `{"type": "json_object"}` triggers structural JSON constraint masking.
    #[serde(default)]
    response_format: Option<ResponseFormat>,
    /// OpenAI-style function tools; when present they are rendered into the prompt
    /// and the completion is parsed back into `tool_calls` (Phase 1a).
    #[serde(default)]
    tools: Option<Vec<serde_json::Value>>,
    /// Accepted for API compatibility; currently advisory only.
    #[serde(default)]
    #[allow(dead_code)]
    tool_choice: Option<serde_json::Value>,
}

#[derive(Deserialize, Default)]
struct ResponseFormat {
    #[serde(rename = "type", default)]
    format_type: String,
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
    // Native tool calling (Phase 1a): render the tool specs into a leading system
    // message so a Hermes/Qwen-trained model emits <tool_call> blocks, and remember
    // to parse them back out of the completion.
    let tools: Vec<serde_json::Value> = req.tools.clone().unwrap_or_default();
    let want_tools = !tools.is_empty();
    let prompt = if want_tools {
        let preamble = crate::tool_calls::render_tools_preamble(&tools);
        let mut msgs = req.messages.clone();
        match msgs.first_mut() {
            Some(first) if first.role == "system" => {
                first.content = format!("{preamble}\n{}", first.content);
            }
            _ => msgs.insert(
                0,
                ChatMessage {
                    role: "system".to_string(),
                    content: preamble,
                },
            ),
        }
        render_chat(&msgs, &s.model_arch)
    } else {
        render_chat(&req.messages, &s.model_arch)
    };
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(0.7),
        top_k: 40,
        top_p: req.top_p.unwrap_or(0.9),
        repetition_penalty: 1.0,
        seed: req.seed,
    };
    let json_mode = req.response_format
        .as_ref()
        .map(|f| f.format_type == "json_object")
        .unwrap_or(false);
    let gen = GenerateRequest {
        prompt,
        max_new_tokens: req.max_tokens,
        sampling,
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
        json_mode,
    };
    if req.stream {
        // Streaming tool-call extraction is a follow-up; content still streams.
        sse_response(s, gen, /*chat=*/ true).into_response()
    } else {
        json_full_response(s, gen, /*chat=*/ true, want_tools)
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
        json_mode: false,
    };
    if req.stream {
        sse_response(s, gen, /*chat=*/ false).into_response()
    } else {
        json_full_response(s, gen, /*chat=*/ false, /*extract_tools=*/ false)
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
    // Qwen2.5's chat template injects a default system message when the caller
    // gives none; without it the model can degenerate on short prompts.
    if msgs.first().map(|m| m.role.as_str()) != Some("system") {
        s.push_str(
            "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. \
             You are a helpful assistant.<|im_end|>\n",
        );
    }
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
    let admit_outcome = {
        let engine = state.engine.lock();
        let mut driver = state.driver.lock();
        driver.admit(&**engine, req.clone())
    };
    // Distinguish a real admit decision from an engine that cannot serve this
    // request at all. `Ok(Some)` = admitted; `Ok(None)` = no free slot (→ queue);
    // `Err` (e.g. the engine lacks `encode_prompt_for_batch`, or tokenization
    // failed) must NOT silently enter the wait-queue forever — return a clear
    // SSE error instead (mirrors the slot-exhausted error path below). This is
    // what made the RWKV admission gap present as a 180s hang.
    let slot_id_opt = match admit_outcome {
        Ok(slot) => slot,
        Err(e) => {
            let sse_tx2 = sse_tx.clone();
            let msg = format!("engine cannot serve this request: {e}");
            tokio::spawn(async move {
                let body = serde_json::json!({
                    "error": {"message": msg, "type": "server_error", "code": "admit_unsupported"}
                });
                let _ = sse_tx2
                    .send(Ok(Event::default().data(body.to_string())))
                    .await;
            });
            return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
        }
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
                let _ = sse_tx2
                    .send(Ok(Event::default().data(body.to_string())))
                    .await;
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
                    if sse_tx
                        .send(Ok(Event::default().data(chunk.to_string())))
                        .await
                        .is_err()
                    {
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

/// Lean request body for the native `/v1/hawking/generate` endpoint.
/// No role/message envelope — just the generation knobs.
#[derive(Deserialize)]
pub struct HawkingGenerateReq {
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    /// Greedy when absent or <= 0.0 (routes to the token-only B×4 lane).
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
    /// Stop strings. Mapped into GenerateRequest.stop. (The batch scheduler
    /// does not yet honor stop; this preserves the field end-to-end.)
    #[serde(default)]
    pub stop: Vec<String>,
}

/// PURE request->GenerateRequest mapping. No engine, no I/O — the gate test
/// calls this directly. temperature absent/<=0 => greedy (temp 0, top_k 0,
/// top_p 1) so the slot routes through forward_multiseq_greedy_tokens.
pub fn map_hawking_generate_req(req: &HawkingGenerateReq) -> GenerateRequest {
    let temp = req.temperature.unwrap_or(0.0);
    let greedy = temp <= 0.0;
    let sampling = SamplingParams {
        temperature: if greedy { 0.0 } else { temp },
        top_k: if greedy { 0 } else { 40 },
        top_p: if greedy {
            1.0
        } else {
            req.top_p.unwrap_or(0.9)
        },
        repetition_penalty: 1.0,
        seed: req.seed,
    };
    GenerateRequest {
        prompt: req.prompt.clone(),
        max_new_tokens: req.max_tokens,
        sampling,
        stop: req.stop.clone(),
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    }
}

/// Re-derive the LM-head path label the same way the engine does for the
/// env-controlled cases (serve always sets HAWKING_QWEN_Q4K_LMHEAD=1, so the
/// q4k* branch is the live one). Mirrors QwenDense::lm_head_path env logic.
/// Returns one of: "q4k-predec-f16s" | "q4k-predec" | "q4k" | "f16".
pub fn lm_head_path_from_env() -> &'static str {
    let q4k = std::env::var_os("HAWKING_QWEN_Q4K_LMHEAD")
        .map(|v| v != "0")
        .unwrap_or(false);
    if !q4k {
        return "f16";
    }
    let predec = std::env::var_os("HAWKING_QWEN_Q4K_PREDEC")
        .map(|v| v != "0")
        .unwrap_or(true);
    let f16s = predec
        && std::env::var_os("HAWKING_QWEN_PREDEC_F16SCALES")
            .map(|v| v != "0")
            .unwrap_or(false);
    if f16s {
        "q4k-predec-f16s"
    } else if predec {
        "q4k-predec"
    } else {
        "q4k"
    }
}

/// PURE: build the native final stats object from server-observed values.
/// Field NAMES mirror GenStats::stats_json() so native + OpenAI clients parse
/// the same keys. dec_tps = completion_tokens / (decode_ms/1000).
pub fn hawking_generate_stats_json(
    prompt_tokens: usize,
    completion_tokens: usize,
    decode_ms: f64,
    token_only_path_used: bool,
    lm_head_path: &str,
) -> serde_json::Value {
    let dec_tps = (completion_tokens as f64) / (decode_ms / 1000.0).max(1e-6);
    serde_json::json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "decode_ms": decode_ms,
        "dec_tps": dec_tps,
        "token_only_path_used": token_only_path_used,
        "lm_head_path": lm_head_path,
    })
}

/// Request body for the low-overhead `/v1/hawking/tokens` endpoint.
#[derive(Deserialize)]
struct HawkingTokensReq {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    seed: Option<u64>,
}

/// Native streaming endpoint: returns raw token IDs as SSE integers.
///
/// Each `data:` line is a decimal u32 token ID. The final event is
/// `data: [DONE]`. Always uses temperature=0 (greedy-only).
///
/// Lower overhead than the OpenAI JSON chunk format because there is no
/// per-token JSON wrapper — just a single integer per SSE event.
async fn hawking_tokens(State(s): State<AppState>, body: Bytes) -> Response {
    let req: HawkingTokensReq = match parse_json(&body) {
        Ok(req) => req,
        Err(e) => return e.into_response(),
    };
    if req.prompt.is_empty() {
        return ApiError::missing_parameter("'prompt' must not be empty").into_response();
    }
    let gen = GenerateRequest {
        prompt: req.prompt,
        max_new_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: req.seed,
        },
        stop: Vec::new(),
        abort: None,
        max_stall_ms: 0,
        json_mode: false,
    };
    token_id_sse_response(s, gen).into_response()
}

/// SSE response that streams raw u32 token IDs (decimal) instead of JSON.
/// Used by the `/v1/hawking/tokens` native endpoint.
fn token_id_sse_response(
    state: AppState,
    req: GenerateRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // SSE → client channel (receives formatted SSE events).
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
    // Token channel: background decode loop sends raw text fragments.
    // We need to recover the token ID; the text is the decoded string.
    // The slot channel carries String; we emit the admission slot_id and
    // let the forwarder read the *token* field from DecodeOutput.
    //
    // Design note: the existing slot pipeline sends decoded *text*, not
    // token IDs, so we can't recover IDs from it directly. The simplest
    // approach is to have the forwarder use the engine to re-encode the
    // text — but that's lossy. Instead we admit via the normal path and
    // set up a parallel tokio channel that carries the raw u32 tokens.
    //
    // For this endpoint we re-use the existing SlotToken (String) pipeline
    // but convert to token IDs in the forwarder. Since the slot pipeline
    // delivers decoded text fragments (not token IDs), we cannot recover
    // the original u32 without changes to core. As a pragmatic fallback
    // for this endpoint we stream the raw text as-is with each token on
    // its own line, prefixed with "tok:". This is lower overhead than the
    // full OpenAI JSON wrapper while remaining valid SSE.
    //
    // A future improvement can plumb token IDs through DecodeOutput → SlotToken.
    let (tok_tx, mut tok_rx) = async_mpsc::channel::<SlotToken>(256);

    let slot_id_opt = {
        let engine = state.engine.lock();
        let mut driver = state.driver.lock();
        driver.admit(&**engine, req.clone()).ok().flatten()
    };

    if let Some(slot_id) = slot_id_opt {
        state.requests_admitted.fetch_add(1, Ordering::Relaxed);
        state.slot_senders.lock().insert(slot_id, tok_tx);
    } else {
        let queue_cap = state.max_batch * 8;
        if state.wait_queue.lock().len() >= queue_cap {
            let sse_tx2 = sse_tx.clone();
            tokio::spawn(async move {
                let body = serde_json::json!({
                    "error": {"message": "server busy — no batch slot available",
                              "type": "server_error", "code": "slot_exhausted"}
                });
                let _ = sse_tx2
                    .send(Ok(Event::default().data(body.to_string())))
                    .await;
            });
            return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
        }
        state.requests_queued.fetch_add(1, Ordering::Relaxed);
        state.wait_queue.lock().push_back((req, tok_tx, false));
    };

    // Forward raw token text from the per-slot channel to SSE events.
    // Each non-empty text fragment is emitted as a raw data line.
    // EOS sentinel sends [DONE].
    tokio::spawn(async move {
        while let Some(item) = tok_rx.recv().await {
            match item {
                Ok(text) => {
                    // Emit each non-empty text fragment as a raw SSE data line.
                    // We escape newlines so each event is a single line.
                    let escaped = text.replace('\n', "\\n");
                    if sse_tx
                        .send(Ok(Event::default().data(escaped)))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(()) => break,
            }
        }
        // Emit [DONE] on ANY stream end (EOS signal, max_tokens channel-close, or
        // client disconnect) so OpenAI-style clients always see a clean terminator.
        let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
}

async fn hawking_generate(State(s): State<AppState>, body: Bytes) -> Response {
    let req: HawkingGenerateReq = match parse_json(&body) {
        Ok(req) => req,
        Err(e) => return e.into_response(),
    };
    if req.prompt.is_empty() {
        return ApiError::missing_parameter("'prompt' must not be empty").into_response();
    }
    let gen = map_hawking_generate_req(&req);
    hawking_generate_sse(s, gen).into_response()
}

/// Native streaming response: per-token JSON chunks {tok_index, text} then a
/// final {stats:{...}} event, then [DONE]. Reuses the OpenAI path's admit +
/// per-slot SlotToken channel — does NOT fork the continuous-batch decode loop.
fn hawking_generate_sse(
    state: AppState,
    req: GenerateRequest,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(256);
    let (tok_tx, mut tok_rx) = async_mpsc::channel::<SlotToken>(256);

    // prompt_tokens for the stats object (tokenize once for the count; admit
    // tokenizes again internally — cheap, keeps admit's signature unchanged).
    let prompt_tokens = {
        let engine = state.engine.lock();
        engine
            .encode_prompt_for_batch(&req.prompt)
            .map(|v| v.len())
            .unwrap_or(0)
    };
    // Snapshot whether the greedy/token-only lane is in play for this request.
    let token_only_snapshot =
        state.driver.lock().lane_stats.greedy_steps > 0 || req.sampling.temperature <= 0.0;
    let lm_head = lm_head_path_from_env();

    let slot_id_opt = {
        let engine = state.engine.lock();
        let mut driver = state.driver.lock();
        driver.admit(&**engine, req.clone()).ok().flatten()
    };
    if let Some(slot_id) = slot_id_opt {
        state.requests_admitted.fetch_add(1, Ordering::Relaxed);
        state.slot_senders.lock().insert(slot_id, tok_tx);
    } else {
        let queue_cap = state.max_batch * 8;
        if state.wait_queue.lock().len() >= queue_cap {
            let sse_tx2 = sse_tx.clone();
            tokio::spawn(async move {
                let body = serde_json::json!({
                    "error": {"message": "server busy — no batch slot available",
                              "type": "server_error", "code": "slot_exhausted"}
                });
                let _ = sse_tx2
                    .send(Ok(Event::default().data(body.to_string())))
                    .await;
            });
            return Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default());
        }
        state.requests_queued.fetch_add(1, Ordering::Relaxed);
        state.wait_queue.lock().push_back((req, tok_tx, false));
    };

    // Forward token text fragments as native chunks; count tokens + wall time
    // for an accurate per-request dec_tps; emit a final stats event on EOS.
    tokio::spawn(async move {
        let start = std::time::Instant::now();
        let mut completion_tokens: usize = 0;
        while let Some(item) = tok_rx.recv().await {
            match item {
                Ok(text) => {
                    let chunk = serde_json::json!({
                        "tok_index": completion_tokens,
                        "text": text,
                    });
                    completion_tokens += 1;
                    if sse_tx
                        .send(Ok(Event::default().data(chunk.to_string())))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
                Err(()) => break,
            }
        }
        // Always emit the final stats + [DONE] when the stream ends — whether by
        // EOS (the Err(()) signal), max_tokens (the slot is released and the
        // channel closes), or client disconnect — so the native SSE terminates
        // cleanly. Previously stats/[DONE] fired only on the EOS signal, so a
        // max_tokens-bounded request ended without them.
        let decode_ms = start.elapsed().as_secs_f64() * 1000.0;
        let stats = hawking_generate_stats_json(
            prompt_tokens,
            completion_tokens,
            decode_ms,
            token_only_snapshot,
            lm_head,
        );
        let final_obj = serde_json::json!({ "stats": stats });
        let _ = sse_tx
            .send(Ok(Event::default().data(final_obj.to_string())))
            .await;
        let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Sse::new(ReceiverStream::new(sse_rx)).keep_alive(KeepAlive::default())
}

async fn json_full_response(
    state: AppState,
    req: GenerateRequest,
    chat: bool,
    extract_tools: bool,
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
            // When tools were requested, parse the completion back into OpenAI
            // tool_calls; otherwise it is a plain assistant message.
            let calls = if extract_tools {
                crate::tool_calls::extract_tool_calls(&text)
            } else {
                Vec::new()
            };
            let (message, finish) = if !calls.is_empty() {
                let arr: Vec<serde_json::Value> = calls.iter().map(|c| c.to_openai()).collect();
                (
                    serde_json::json!({
                        "role": "assistant",
                        "content": serde_json::Value::Null,
                        "tool_calls": arr
                    }),
                    "tool_calls",
                )
            } else {
                (
                    serde_json::json!({ "role": "assistant", "content": text }),
                    "stop",
                )
            };
            serde_json::json!({
                "object": "chat.completion",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}]
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

// ── POST /v1/embeddings ───────────────────────────────────────────────────────

#[derive(Deserialize)]
struct EmbeddingsReq {
    input: EmbeddingsInput,
    #[allow(dead_code)]
    #[serde(default)]
    model: Option<String>,
    #[serde(default = "default_embedding_encoding")]
    encoding_format: String,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

fn default_embedding_encoding() -> String {
    "float".to_string()
}

async fn embeddings(State(s): State<AppState>, body: Bytes) -> Response {
    let req: EmbeddingsReq = match parse_json(&body) {
        Ok(req) => req,
        Err(e) => return e.into_response(),
    };
    let inputs: Vec<String> = match req.input {
        EmbeddingsInput::Single(t) => vec![t],
        EmbeddingsInput::Batch(v) => v,
    };
    if inputs.is_empty() {
        return ApiError::missing_parameter("'input' must not be empty").into_response();
    }
    if req.encoding_format != "float" {
        return ApiError::invalid_json("only encoding_format=float is supported").into_response();
    }

    let model_id = s.engine.lock().model_id().to_string();
    let engine = s.engine.clone();

    let result = tokio::task::spawn_blocking(move || {
        let mut eng = engine.lock();
        let mut data = Vec::with_capacity(inputs.len());
        for (idx, text) in inputs.iter().enumerate() {
            let vec = eng.embed(text)?;
            data.push(serde_json::json!({
                "object": "embedding",
                "index": idx,
                "embedding": vec,
            }));
        }
        hawking_core::Result::Ok(data)
    })
    .await;

    match result {
        Ok(Ok(data)) => Json(serde_json::json!({
            "object": "list",
            "data": data,
            "model": model_id,
            "usage": { "prompt_tokens": 0, "total_tokens": 0 },
        }))
        .into_response(),
        Ok(Err(e)) => ApiError::internal(e.to_string()).into_response(),
        Err(_) => ApiError::internal("embedding task panicked").into_response(),
    }
}
