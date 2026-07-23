//! HTTP `ModelProvider` over the supervised `hawking serve` (bible ch.01 §4.3 /
//! ch.06 §4.4).
//!
//! [`HttpModelProvider`] implements `hide_core::runtime::ModelProvider` against a
//! live runtime reached over HTTP only (T5 — no engine-crate link). It speaks the
//! three serve endpoints:
//!
//! * `POST /v1/hawking/generate` — native completion (preferred).
//! * `POST /v1/chat/completions` — OpenAI-compatible chat (message-shaped reqs).
//! * `POST /v1/embeddings` — a single embedding vector.
//!
//! With this wired, the kernel's `Act` step can finally generate against a live
//! model *through the host* (the audit's B4/B10 gap: "the runtime is never
//! booted; nothing flows end-to-end"). The base URL comes from the
//! [`crate::supervisor::RuntimeSupervisor`], so provider and supervisor agree on
//! where the child is listening.
//!
//! Tests drive it against the in-process fake from `supervisor::testkit` (a TCP
//! listener answering the same JSON shapes) — no model required.

use futures::future::BoxFuture;
use futures::StreamExt;
use hide_core::error::{HideError, Result};
use hide_core::runtime::{
    GenerationStats, InferenceRequest, ModelProvider, ProviderCaps, SamplerProfile, StreamChunk,
    TokenSink,
};
use serde_json::{json, Value};
use std::time::Duration;

/// Which serve route a generation targets (mirrors the orch client).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerateRoute {
    /// `/v1/hawking/generate` — lean native body.
    Native,
    /// `/v1/chat/completions` — OpenAI-compatible chat body.
    Chat,
}

/// Spine A: the engine's live context snapshot, mirroring `hawking-serve`'s
/// `/v1/hawking/context` response. `#[serde(default)]` throughout so a serve
/// build that predates a field still deserializes (forward-compatible).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ContextInfo {
    #[serde(default)]
    pub model_id: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub ctx_len_native: Option<usize>,
    #[serde(default)]
    pub ctx_len_effective: Option<usize>,
    #[serde(default)]
    pub tq_multiplier: f32,
    #[serde(default)]
    pub tq_estimated: bool,
    #[serde(default)]
    pub recurrent_state_bytes: Option<usize>,
    #[serde(default)]
    pub active_slots: usize,
    #[serde(default)]
    pub free_slots: usize,
    #[serde(default)]
    pub max_batch: usize,
}

/// A reqwest-backed [`ModelProvider`] pointed at a (supervised) serve instance.
pub struct HttpModelProvider {
    base_url: String,
    route: GenerateRoute,
    client: reqwest::Client,
    id: String,
}

impl HttpModelProvider {
    /// Construct against `base_url` (`http://host:port`), preferring the native
    /// generate route.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_route(base_url, GenerateRoute::Native)
    }

    pub fn with_route(base_url: impl Into<String>, route: GenerateRoute) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_default();
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            route,
            client,
            id: "hawking-serve-http".to_string(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    /// Spine A: read the engine's live context picture from `GET /v1/hawking/context`
    /// (native + effective ceiling, the measured `.tq` multiplier, recurrent-state
    /// bytes, slot occupancy). `None` if the serve instance is down or pre-context
    /// (old build) — the caller then shows no live ceiling rather than a fake one.
    pub async fn get_context_info(&self) -> Option<ContextInfo> {
        let resp = self
            .client
            .get(self.url("/v1/hawking/context"))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json::<ContextInfo>().await.ok()
    }

    fn sampler(request: &InferenceRequest) -> SamplerProfile {
        request
            .sampler
            .clone()
            .unwrap_or_else(SamplerProfile::deterministic_edit)
    }

    fn prompt_text(request: &InferenceRequest) -> String {
        if !request.prompt.is_empty() {
            return request.prompt.clone();
        }
        request
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn native_body(request: &InferenceRequest) -> Value {
        let s = Self::sampler(request);
        // The native `/v1/hawking/generate` route always responds with an SSE
        // token stream (it ignores a `stream:false`); we stream it token by
        // token below. The field is kept truthful for any client that honours it.
        json!({
            "prompt": Self::prompt_text(request),
            "max_tokens": request.max_output_tokens,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "seed": s.seed,
            "stop": [],
            "stream": true,
        })
    }

    fn chat_body(request: &InferenceRequest) -> Value {
        let s = Self::sampler(request);
        let messages: Vec<Value> = if request.messages.is_empty() {
            vec![json!({ "role": "user", "content": request.prompt })]
        } else {
            request
                .messages
                .iter()
                .map(|m| json!({ "role": m.role, "content": m.content }))
                .collect()
        };
        json!({
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "seed": s.seed,
            "stream": false,
        })
    }

    /// Consume the native SSE token stream from `/v1/hawking/generate`,
    /// forwarding each token fragment to `sink` as a [`StreamChunk::Token`] (so
    /// the UI renders tokens as they arrive), the final stats as a
    /// [`StreamChunk::Done`], and a server-side error as a
    /// [`StreamChunk::Error`]. Frames are reassembled across network-chunk
    /// boundaries by buffering and splitting on newlines. Returns the terminal
    /// stats (zeroed if the stream ended without a stats event).
    async fn stream_native_sse(
        resp: reqwest::Response,
        sink: TokenSink<'_>,
    ) -> Result<GenerationStats> {
        let mut body = resp.bytes_stream();
        let mut buf = String::new();
        let mut final_stats: Option<GenerationStats> = None;
        let mut done = false;

        while let Some(item) = body.next().await {
            let bytes = item.map_err(|e| {
                HideError::RuntimeUnavailable(format!("generate stream read failed: {e}"))
            })?;
            buf.push_str(&String::from_utf8_lossy(&bytes));

            // Drain whole lines; keep the trailing partial line in `buf`.
            while let Some(nl) = buf.find('\n') {
                let line: String = buf.drain(..=nl).collect();
                match parse_native_sse_line(&line) {
                    SseChunk::Token(text) => {
                        sink(StreamChunk::Token {
                            token_id: None,
                            text,
                        })?;
                    }
                    SseChunk::Done(stats) => {
                        // Both the stats object and the `[DONE]` terminator yield
                        // a Done; keep the FIRST stats-bearing one (the `[DONE]`
                        // line carries zeroed stats and must not clobber it).
                        if final_stats.is_none() || !is_zero_stats(&stats) {
                            final_stats = Some(stats);
                        }
                        done = true;
                    }
                    SseChunk::Error(message) => {
                        sink(StreamChunk::Error {
                            message: message.clone(),
                        })?;
                        return Err(HideError::RuntimeUnavailable(message));
                    }
                    SseChunk::Ignore => {}
                }
            }
        }
        // Parse any final buffered line without a trailing newline.
        if !buf.trim().is_empty() {
            match parse_native_sse_line(&buf) {
                SseChunk::Token(text) => sink(StreamChunk::Token {
                    token_id: None,
                    text,
                })?,
                SseChunk::Done(stats) => {
                    if final_stats.is_none() || !is_zero_stats(&stats) {
                        final_stats = Some(stats);
                    }
                }
                SseChunk::Error(message) => {
                    sink(StreamChunk::Error {
                        message: message.clone(),
                    })?;
                    return Err(HideError::RuntimeUnavailable(message));
                }
                SseChunk::Ignore => {}
            }
        }

        let stats = final_stats.unwrap_or(GenerationStats {
            input_tokens: 0,
            output_tokens: 0,
            decode_tokens_per_second: None,
        });
        // Terminal Done so the bus flushes the coalesced batch (even if the
        // stream ended without an explicit stats/[DONE] line).
        let _ = done;
        sink(StreamChunk::Done {
            reason: "stop".to_string(),
            stats: Some(stats.clone()),
        })?;
        Ok(stats)
    }
}

/// Extract the completion text + stats from a non-streaming response of either
/// route. Pure so it is unit-tested without a network.
pub fn extract_completion(route: GenerateRoute, body: &Value) -> (String, GenerationStats) {
    let text = match route {
        GenerateRoute::Native => body
            .get("text")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        GenerateRoute::Chat => body
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
    };
    let stats_obj = body.get("stats").or_else(|| body.get("usage"));
    let stats = GenerationStats {
        input_tokens: stats_obj
            .and_then(|s| s.get("input_tokens").or_else(|| s.get("prompt_tokens")))
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        output_tokens: stats_obj
            .and_then(|s| {
                s.get("output_tokens")
                    .or_else(|| s.get("completion_tokens"))
            })
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize,
        decode_tokens_per_second: stats_obj
            .and_then(|s| s.get("dec_tps"))
            .and_then(Value::as_f64)
            .map(|v| v as f32),
    };
    (text, stats)
}

/// One parsed SSE `data:` payload from the native `/v1/hawking/generate`
/// stream. The route emits, per line:
///   * `data: {"tok_index": N, "text": "..."}`   (a token fragment)
///   * `data: {"stats": {...}}`                   (the terminal stats object)
///   * `data: {"error": {"message": ...}}`        (a server-side error)
///   * `data: [DONE]`                             (the stream terminator)
#[derive(Debug, Clone, PartialEq)]
pub enum SseChunk {
    Token(String),
    Done(GenerationStats),
    Error(String),
    /// A non-data / comment / keep-alive line, or an unrecognised object.
    Ignore,
}

/// Parse a single SSE line (with or without the leading `data: `) from the
/// native generate stream into an [`SseChunk`]. Pure so it is unit-tested
/// without a network. A `[DONE]` terminator with no preceding stats yields a
/// zero-stat `Done`.
pub fn parse_native_sse_line(line: &str) -> SseChunk {
    let line = line.trim();
    let payload = match line.strip_prefix("data:") {
        Some(rest) => rest.trim(),
        None => return SseChunk::Ignore,
    };
    if payload.is_empty() {
        return SseChunk::Ignore;
    }
    if payload == "[DONE]" {
        return SseChunk::Done(GenerationStats {
            input_tokens: 0,
            output_tokens: 0,
            decode_tokens_per_second: None,
        });
    }
    let value: Value = match serde_json::from_str(payload) {
        Ok(v) => v,
        Err(_) => return SseChunk::Ignore,
    };
    if let Some(err) = value.get("error") {
        let msg = err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("runtime error")
            .to_string();
        return SseChunk::Error(msg);
    }
    if value.get("stats").is_some() {
        let (_text, stats) = extract_completion(GenerateRoute::Native, &value);
        return SseChunk::Done(stats);
    }
    if let Some(text) = value.get("text").and_then(Value::as_str) {
        return SseChunk::Token(text.to_string());
    }
    SseChunk::Ignore
}

/// True for the zeroed stats the `[DONE]` terminator carries, used so the real
/// stats event (which precedes `[DONE]`) is not clobbered by the terminator.
fn is_zero_stats(s: &GenerationStats) -> bool {
    s.input_tokens == 0 && s.output_tokens == 0 && s.decode_tokens_per_second.is_none()
}

/// Extract the first embedding vector from a `/v1/embeddings` response.
pub fn extract_embedding(body: &Value) -> Result<Vec<f32>> {
    body.get("data")
        .and_then(|d| d.get(0))
        .and_then(|e| e.get("embedding"))
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_f64)
                .map(|v| v as f32)
                .collect()
        })
        .ok_or_else(|| {
            HideError::RuntimeUnavailable(
                "embeddings response missing data[0].embedding".to_string(),
            )
        })
}

impl ModelProvider for HttpModelProvider {
    fn id(&self) -> &str {
        &self.id
    }

    fn capabilities(&self) -> ProviderCaps {
        ProviderCaps::hawking_local_shell_today()
    }

    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        Box::pin(async move {
            let (path, body) = match self.route {
                GenerateRoute::Native => ("/v1/hawking/generate", Self::native_body(&request)),
                GenerateRoute::Chat => ("/v1/chat/completions", Self::chat_body(&request)),
            };
            let resp = self
                .client
                .post(self.url(path))
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    HideError::RuntimeUnavailable(format!("generate request failed: {e}"))
                })?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!(
                    "generate returned {}",
                    resp.status()
                )));
            }

            // The real `hawking serve` answers the native route with an SSE token
            // stream (`text/event-stream`); the in-process fake answers with a
            // plain JSON body. Branch on the content type so BOTH work: stream
            // token-by-token to the UI when SSE, or fall back to the one-batch
            // path for a JSON body. (Chat is always treated as one JSON body.)
            let is_sse = matches!(self.route, GenerateRoute::Native)
                && resp
                    .headers()
                    .get(reqwest::header::CONTENT_TYPE)
                    .and_then(|v| v.to_str().ok())
                    .map(|v| v.contains("text/event-stream"))
                    .unwrap_or(false);

            if is_sse {
                return Self::stream_native_sse(resp, sink).await;
            }

            let value: Value = resp.json().await.map_err(|e| {
                HideError::RuntimeUnavailable(format!("generate decode failed: {e}"))
            })?;
            let (text, stats) = extract_completion(self.route, &value);
            // Emit the whole completion as one token batch, then a terminal Done —
            // the same contract the streaming path produces, so callers (the
            // kernel `Act` step / token-bus) don't branch on stream vs non-stream.
            sink(StreamChunk::Token {
                token_id: None,
                text,
            })?;
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: Some(stats.clone()),
            })?;
            Ok(stats)
        })
    }

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        Box::pin(async move {
            let resp = self
                .client
                .post(self.url("/v1/embeddings"))
                .json(&json!({ "input": text, "encoding_format": "float" }))
                .send()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("embed request failed: {e}")))?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!(
                    "embeddings returned {}",
                    resp.status()
                )));
            }
            let value: Value = resp
                .json()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("embed decode failed: {e}")))?;
            extract_embedding(&value)
        })
    }
}

/// Adapter: expose a [`ModelProvider`] as the orchestrator's `InferenceClient`
/// so the kernel's `KernelRuntimeClient` can generate through the *host's* HTTP
/// provider. Both traits share the `generate(request, sink)` + `embed(text)`
/// shape, so this is a thin forwarding wrapper — the seam that lets the kernel's
/// `Act` step reach the supervised runtime via the host.
pub struct ModelProviderInferenceClient<P: ModelProvider> {
    provider: P,
}

impl<P: ModelProvider> ModelProviderInferenceClient<P> {
    pub fn new(provider: P) -> Self {
        Self { provider }
    }
}

impl<P: ModelProvider> hawking_orch::inference::InferenceClient
    for ModelProviderInferenceClient<P>
{
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        self.provider.generate(request, sink)
    }

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        self.provider.embed(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supervisor::testkit::FakeRuntime;
    use std::sync::Arc;

    #[test]
    fn extract_native_completion_reads_text_and_stats() {
        let body = json!({
            "text": "hello world",
            "stats": { "input_tokens": 3, "output_tokens": 2, "dec_tps": 41.5 }
        });
        let (text, stats) = extract_completion(GenerateRoute::Native, &body);
        assert_eq!(text, "hello world");
        assert_eq!(stats.input_tokens, 3);
        assert_eq!(stats.output_tokens, 2);
        assert_eq!(stats.decode_tokens_per_second, Some(41.5));
    }

    #[test]
    fn extract_chat_completion_reads_delta_content() {
        let body = json!({
            "choices": [{ "message": { "content": "chat reply" } }],
            "usage": { "prompt_tokens": 4, "completion_tokens": 5 }
        });
        let (text, stats) = extract_completion(GenerateRoute::Chat, &body);
        assert_eq!(text, "chat reply");
        assert_eq!(stats.input_tokens, 4);
        assert_eq!(stats.output_tokens, 5);
    }

    #[test]
    fn extract_embedding_reads_first_vector() {
        let body = json!({ "data": [{ "embedding": [0.5, 0.25] }] });
        assert_eq!(extract_embedding(&body).unwrap(), vec![0.5f32, 0.25]);
    }

    #[test]
    fn parse_native_sse_lines_cover_token_stats_error_done() {
        // A token fragment.
        assert_eq!(
            parse_native_sse_line("data: {\"tok_index\": 0, \"text\": \" Paris\"}"),
            SseChunk::Token(" Paris".to_string())
        );
        // The terminal stats object → Done carrying the parsed stats.
        match parse_native_sse_line(
            "data: {\"stats\": {\"prompt_tokens\": 3, \"completion_tokens\": 5, \"dec_tps\": 40.0}}",
        ) {
            SseChunk::Done(stats) => {
                assert_eq!(stats.input_tokens, 3);
                assert_eq!(stats.output_tokens, 5);
            }
            other => panic!("expected Done, got {other:?}"),
        }
        // A server-side error.
        assert_eq!(
            parse_native_sse_line("data: {\"error\": {\"message\": \"server busy\"}}"),
            SseChunk::Error("server busy".to_string())
        );
        // The terminator.
        assert!(matches!(
            parse_native_sse_line("data: [DONE]"),
            SseChunk::Done(_)
        ));
        // Keep-alive / comment / blank lines are ignored.
        assert_eq!(parse_native_sse_line(": keep-alive"), SseChunk::Ignore);
        assert_eq!(parse_native_sse_line(""), SseChunk::Ignore);
    }

    /// An in-process SSE server answering the native generate route exactly as
    /// `hawking serve` does (token chunks → stats → [DONE]), so the provider's
    /// real streaming path is exercised without a model. Proves: tokens arrive
    /// individually (not one batch) and the terminal Done carries stats.
    #[tokio::test]
    async fn generate_streams_native_sse_token_by_token() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf).await;
            let sse = concat!(
                "data: {\"tok_index\":0,\"text\":\" Paris\"}\n\n",
                "data: {\"tok_index\":1,\"text\":\".\"}\n\n",
                "data: {\"stats\":{\"prompt_tokens\":3,\"completion_tokens\":2,\"dec_tps\":40.0}}\n\n",
                "data: [DONE]\n\n",
            );
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                sse.len(),
                sse
            );
            let _ = stream.write_all(resp.as_bytes()).await;
            let _ = stream.flush().await;
        });

        let provider = HttpModelProvider::new(format!("http://{addr}"));
        let mut tokens: Vec<String> = Vec::new();
        let mut done = false;
        {
            let mut sink = |chunk: StreamChunk| {
                match chunk {
                    StreamChunk::Token { text, .. } => tokens.push(text),
                    StreamChunk::Done { .. } => done = true,
                    StreamChunk::Error { message } => panic!("error chunk: {message}"),
                }
                Ok(())
            };
            let req = InferenceRequest {
                task_kind: "code".into(),
                prompt: "The capital of France is".into(),
                messages: Vec::new(),
                max_output_tokens: 16,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: Default::default(),
            };
            let stats = provider.generate(req, &mut sink).await.unwrap();
            assert_eq!(stats.output_tokens, 2);
        }
        // Tokens arrived as individual fragments (streaming), not one blob.
        assert_eq!(tokens, vec![" Paris".to_string(), ".".to_string()]);
        assert!(done, "stream must end with a terminal Done");
    }

    #[tokio::test]
    async fn generate_against_fake_runtime_emits_token_and_done() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let provider = HttpModelProvider::new(rt.base_url());
        let mut tokens = Vec::new();
        let mut done = false;
        {
            let mut sink = |chunk: StreamChunk| {
                match chunk {
                    StreamChunk::Token { text, .. } => tokens.push(text),
                    StreamChunk::Done { .. } => done = true,
                    StreamChunk::Error { message } => panic!("error chunk: {message}"),
                }
                Ok(())
            };
            let req = InferenceRequest {
                task_kind: "edit".into(),
                prompt: "write a function".into(),
                messages: Vec::new(),
                max_output_tokens: 16,
                sampler: None,
                grammar: None,
                want_logprobs: false,
                metadata: Default::default(),
            };
            let stats = provider.generate(req, &mut sink).await.unwrap();
            assert_eq!(stats.output_tokens, 2);
        }
        assert_eq!(tokens, vec!["fake generate".to_string()]);
        assert!(done);
        rt.stop();
    }

    #[tokio::test]
    async fn embed_against_fake_runtime_returns_vector() {
        let rt = Arc::new(FakeRuntime::spawn().await);
        let provider = HttpModelProvider::new(rt.base_url());
        let vec = provider.embed("hello").await.unwrap();
        assert_eq!(vec, vec![0.1f32, 0.2, 0.3]);
        rt.stop();
    }
}
