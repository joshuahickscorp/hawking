//! Live HTTP client for a `hawking-serve` instance.
//!
//! Replaces the old hand-rolled blocking `TcpStream` with a `reqwest` client
//! that streams **incrementally** — bytes arrive, are parsed into SSE events,
//! and forwarded to the caller's [`TokenSink`] as they land (no buffering the
//! whole response first). Three endpoints are spoken:
//!
//! * `/v1/hawking/generate` — native SSE, `{ "text", "tok_index" }` token frames
//!   plus a trailing `{ "stats": { … } }` frame and a `[DONE]` terminator.
//! * `/v1/chat/completions` — OpenAI-compatible SSE, `choices[].delta.content`.
//! * `/v1/embeddings` — JSON, `data[0].embedding`.
//!
//! The runtime is **not** running during tests, so the streaming and parsing
//! logic is factored into pure functions ([`parse_native_sse_event`],
//! [`parse_openai_sse_event`], [`extract_embedding`]) that are unit-tested
//! directly; the network path is exercised only behind the live trait.

use crate::inference::InferenceClient;
use eventsource_stream::Eventsource;
use futures::future::BoxFuture;
use futures::StreamExt;
use hide_core::error::{HideError, Result};
use hide_core::runtime::{
    GenerationStats, InferenceRequest, SamplerProfile, StreamChunk, TokenSink,
};
use serde_json::{json, Value};
use std::time::Duration;

/// Which serve route a generation should target. The native route is preferred
/// when the role advertises `native_tokens_endpoint`; chat is the portable
/// fallback (and the only route for message-shaped requests against a model
/// that wants a chat template).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GenerateRoute {
    Native,
    Chat,
}

#[derive(Debug, Clone)]
pub struct HawkingHttpClient {
    pub base_url: String,
    pub route: GenerateRoute,
    pub timeout: Duration,
    client: reqwest::Client,
}

impl HawkingHttpClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_route(base_url, GenerateRoute::Native)
    }

    pub fn with_route(base_url: impl Into<String>, route: GenerateRoute) -> Self {
        let timeout = Duration::from_secs(300);
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .unwrap_or_default();
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            route,
            timeout,
            client,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn sampler_or_default(request: &InferenceRequest) -> SamplerProfile {
        request
            .sampler
            .clone()
            .unwrap_or_else(SamplerProfile::deterministic_edit)
    }

    fn prompt_text(request: &InferenceRequest) -> String {
        if request.prompt.is_empty() {
            request
                .messages
                .iter()
                .map(|m| format!("{}: {}", m.role, m.content))
                .collect::<Vec<_>>()
                .join("\n")
        } else {
            request.prompt.clone()
        }
    }

    /// Body for `/v1/hawking/generate`.
    pub fn build_native_generate_body(request: &InferenceRequest) -> Value {
        let sampler = Self::sampler_or_default(request);
        let mut body = json!({
            "prompt": Self::prompt_text(request),
            "max_tokens": request.max_output_tokens,
            "temperature": sampler.temperature,
            "stream": true,
            "stop": [],
        });
        if let Some(top_p) = sampler.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(top_k) = sampler.top_k {
            body["top_k"] = json!(top_k);
        }
        if let Some(seed) = sampler.seed {
            body["seed"] = json!(seed);
        }
        if let Some(rp) = sampler.repetition_penalty {
            body["repetition_penalty"] = json!(rp);
        }
        if request.grammar.is_some() {
            // Today's surface only honors generic json_mode; richer grammar is a
            // runtime ask (ch.06 §4.5.4). Flag it so the server can opt in.
            body["json_mode"] = json!(true);
        }
        body
    }

    /// Body for `/v1/chat/completions`.
    pub fn build_chat_body(request: &InferenceRequest) -> Value {
        let sampler = Self::sampler_or_default(request);
        let messages: Vec<Value> = if request.messages.is_empty() {
            vec![json!({ "role": "user", "content": request.prompt })]
        } else {
            request
                .messages
                .iter()
                .map(|m| json!({ "role": m.role, "content": m.content }))
                .collect()
        };
        let mut body = json!({
            "messages": messages,
            "max_tokens": request.max_output_tokens,
            "temperature": sampler.temperature,
            "stream": true,
        });
        if let Some(top_p) = sampler.top_p {
            body["top_p"] = json!(top_p);
        }
        if let Some(seed) = sampler.seed {
            body["seed"] = json!(seed);
        }
        if request.grammar.is_some() {
            body["response_format"] = json!({ "type": "json_object" });
        }
        body
    }

    async fn stream_sse(
        &self,
        path: &str,
        body: Value,
        route: GenerateRoute,
        sink: TokenSink<'_>,
    ) -> Result<GenerationStats> {
        let resp = self
            .client
            .post(self.url(path))
            .header("Accept", "text/event-stream")
            .json(&body)
            .send()
            .await
            .map_err(|e| HideError::RuntimeUnavailable(format!("request to {path} failed: {e}")))?;
        if !resp.status().is_success() {
            return Err(HideError::RuntimeUnavailable(format!(
                "{path} returned HTTP {}",
                resp.status()
            )));
        }

        let mut stats = GenerationStats {
            input_tokens: 0,
            output_tokens: 0,
            decode_tokens_per_second: None,
        };
        let mut stream = resp.bytes_stream().eventsource();
        while let Some(event) = stream.next().await {
            let event = event.map_err(|e| {
                HideError::RuntimeUnavailable(format!("SSE stream error on {path}: {e}"))
            })?;
            let parsed = match route {
                GenerateRoute::Native => parse_native_sse_event(&event.data, &mut stats),
                GenerateRoute::Chat => parse_openai_sse_event(&event.data, &mut stats),
            };
            match parsed {
                SseStep::Token(chunk) => sink(chunk)?,
                SseStep::Done(reason) => {
                    sink(StreamChunk::Done {
                        reason,
                        stats: Some(stats.clone()),
                    })?;
                    return Ok(stats);
                }
                SseStep::Ignore => {}
            }
        }
        // Stream ended without an explicit terminator — still close it out.
        sink(StreamChunk::Done {
            reason: "eof".to_string(),
            stats: Some(stats.clone()),
        })?;
        Ok(stats)
    }
}

impl InferenceClient for HawkingHttpClient {
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        Box::pin(async move {
            match self.route {
                GenerateRoute::Native => {
                    let body = Self::build_native_generate_body(&request);
                    self.stream_sse("/v1/hawking/generate", body, GenerateRoute::Native, sink)
                        .await
                }
                GenerateRoute::Chat => {
                    let body = Self::build_chat_body(&request);
                    self.stream_sse("/v1/chat/completions", body, GenerateRoute::Chat, sink)
                        .await
                }
            }
        })
    }

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        Box::pin(async move {
            let body = json!({ "input": text, "model": "hawking-local" });
            let resp = self
                .client
                .post(self.url("/v1/embeddings"))
                .json(&body)
                .send()
                .await
                .map_err(|e| {
                    HideError::RuntimeUnavailable(format!("embeddings request failed: {e}"))
                })?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!(
                    "/v1/embeddings returned HTTP {}",
                    resp.status()
                )));
            }
            let value: Value = resp.json().await.map_err(|e| {
                HideError::RuntimeUnavailable(format!("embeddings decode failed: {e}"))
            })?;
            extract_embedding(&value)
                .ok_or_else(|| HideError::RuntimeUnavailable("no embedding in response".into()))
        })
    }
}

/// Outcome of parsing one SSE `data:` payload.
enum SseStep {
    Token(StreamChunk),
    Done(String),
    Ignore,
}

/// Parse a native `/v1/hawking/generate` SSE frame, mutating running stats.
fn parse_native_sse_event(data: &str, stats: &mut GenerationStats) -> SseStep {
    let data = data.trim();
    if data == "[DONE]" {
        return SseStep::Done("stop".to_string());
    }
    let value: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return SseStep::Ignore,
    };
    if let Some(raw_stats) = value.get("stats") {
        if let Some(pt) = raw_stats.get("prompt_tokens").and_then(|v| v.as_u64()) {
            stats.input_tokens = pt as usize;
        }
        if let Some(ct) = raw_stats.get("completion_tokens").and_then(|v| v.as_u64()) {
            stats.output_tokens = ct as usize;
        }
        if let Some(tps) = raw_stats.get("dec_tps").and_then(|v| v.as_f64()) {
            stats.decode_tokens_per_second = Some(tps as f32);
        }
    }
    if let Some(text) = value.get("text").and_then(|v| v.as_str()) {
        stats.output_tokens += 1;
        return SseStep::Token(StreamChunk::Token {
            token_id: value
                .get("tok_index")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32),
            text: text.to_string(),
        });
    }
    SseStep::Ignore
}

/// Parse an OpenAI `/v1/chat/completions` SSE delta frame.
fn parse_openai_sse_event(data: &str, stats: &mut GenerationStats) -> SseStep {
    let data = data.trim();
    if data == "[DONE]" {
        return SseStep::Done("stop".to_string());
    }
    let value: Value = match serde_json::from_str(data) {
        Ok(v) => v,
        Err(_) => return SseStep::Ignore,
    };
    if let Some(usage) = value.get("usage") {
        if let Some(pt) = usage.get("prompt_tokens").and_then(|v| v.as_u64()) {
            stats.input_tokens = pt as usize;
        }
        if let Some(ct) = usage.get("completion_tokens").and_then(|v| v.as_u64()) {
            stats.output_tokens = ct as usize;
        }
    }
    let choice = value.get("choices").and_then(|c| c.get(0));
    if let Some(choice) = choice {
        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            if !reason.is_empty() {
                // A delta with content AND a finish_reason can co-occur; emit the
                // content first if present, otherwise close.
                if let Some(content) = choice
                    .get("delta")
                    .and_then(|d| d.get("content"))
                    .and_then(|v| v.as_str())
                    .filter(|s| !s.is_empty())
                {
                    stats.output_tokens += 1;
                    return SseStep::Token(StreamChunk::Token {
                        token_id: None,
                        text: content.to_string(),
                    });
                }
                return SseStep::Done(reason.to_string());
            }
        }
        if let Some(content) = choice
            .get("delta")
            .and_then(|d| d.get("content"))
            .and_then(|v| v.as_str())
        {
            if !content.is_empty() {
                stats.output_tokens += 1;
                return SseStep::Token(StreamChunk::Token {
                    token_id: None,
                    text: content.to_string(),
                });
            }
        }
    }
    SseStep::Ignore
}

/// Pull the first embedding vector out of an `/v1/embeddings` response.
fn extract_embedding(value: &Value) -> Option<Vec<f32>> {
    let arr = value
        .get("data")
        .and_then(|d| d.get(0))
        .and_then(|e| e.get("embedding"))
        // Allow a bare {"embedding": [...]} too.
        .or_else(|| value.get("embedding"))?;
    let vec = arr
        .as_array()?
        .iter()
        .map(|v| v.as_f64().map(|f| f as f32))
        .collect::<Option<Vec<f32>>>()?;
    Some(vec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn empty_stats() -> GenerationStats {
        GenerationStats {
            input_tokens: 0,
            output_tokens: 0,
            decode_tokens_per_second: None,
        }
    }

    #[test]
    fn native_event_emits_token() {
        let mut stats = empty_stats();
        match parse_native_sse_event("{\"tok_index\":3,\"text\":\"hi\"}", &mut stats) {
            SseStep::Token(StreamChunk::Token { token_id, text }) => {
                assert_eq!(token_id, Some(3));
                assert_eq!(text, "hi");
            }
            _ => panic!("expected token"),
        }
    }

    #[test]
    fn native_event_folds_stats_and_done() {
        let mut stats = empty_stats();
        let _ = parse_native_sse_event(
            "{\"stats\":{\"prompt_tokens\":5,\"completion_tokens\":9,\"dec_tps\":42.5}}",
            &mut stats,
        );
        assert_eq!(stats.input_tokens, 5);
        assert_eq!(stats.output_tokens, 9);
        assert_eq!(stats.decode_tokens_per_second, Some(42.5));
        assert!(matches!(
            parse_native_sse_event("[DONE]", &mut stats),
            SseStep::Done(_)
        ));
    }

    #[test]
    fn openai_delta_emits_content() {
        let mut stats = empty_stats();
        let frame = "{\"choices\":[{\"delta\":{\"content\":\"foo\"},\"finish_reason\":null}]}";
        match parse_openai_sse_event(frame, &mut stats) {
            SseStep::Token(StreamChunk::Token { text, .. }) => assert_eq!(text, "foo"),
            _ => panic!("expected token"),
        }
        assert_eq!(stats.output_tokens, 1);
    }

    #[test]
    fn openai_finish_reason_closes() {
        let mut stats = empty_stats();
        let frame = "{\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}";
        assert!(matches!(
            parse_openai_sse_event(frame, &mut stats),
            SseStep::Done(_)
        ));
    }

    #[test]
    fn embedding_extracted_from_openai_shape() {
        let v: Value = serde_json::from_str(
            "{\"data\":[{\"embedding\":[0.1,0.2,0.3],\"index\":0}]}",
        )
        .unwrap();
        assert_eq!(extract_embedding(&v), Some(vec![0.1, 0.2, 0.3]));
    }

    #[test]
    fn chat_body_uses_messages_or_prompt() {
        let req = InferenceRequest {
            task_kind: "chat".into(),
            prompt: "hello".into(),
            messages: Vec::new(),
            max_output_tokens: 16,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: BTreeMap::new(),
        };
        let body = HawkingHttpClient::build_chat_body(&req);
        assert_eq!(body["messages"][0]["content"], "hello");
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn native_body_carries_sampler() {
        let mut req = InferenceRequest {
            task_kind: "t".into(),
            prompt: "p".into(),
            messages: Vec::new(),
            max_output_tokens: 7,
            sampler: Some(SamplerProfile {
                temperature: 0.3,
                top_k: Some(40),
                top_p: Some(0.9),
                repetition_penalty: Some(1.1),
                seed: Some(11),
                deterministic: false,
            }),
            grammar: Some("g".into()),
            want_logprobs: false,
            metadata: BTreeMap::new(),
        };
        let body = HawkingHttpClient::build_native_generate_body(&req);
        assert_eq!(body["max_tokens"], 7);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["seed"], 11);
        assert_eq!(body["json_mode"], true);
        req.grammar = None;
        let body = HawkingHttpClient::build_native_generate_body(&req);
        assert!(body.get("json_mode").is_none());
    }
}
