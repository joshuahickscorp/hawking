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
        json!({
            "prompt": Self::prompt_text(request),
            "max_tokens": request.max_output_tokens,
            "temperature": s.temperature,
            "top_p": s.top_p,
            "seed": s.seed,
            "stop": [],
            "stream": false,
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
        .ok_or_else(|| HideError::RuntimeUnavailable("embeddings response missing data[0].embedding".to_string()))
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
                .map_err(|e| HideError::RuntimeUnavailable(format!("generate request failed: {e}")))?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!(
                    "generate returned {}",
                    resp.status()
                )));
            }
            let value: Value = resp
                .json()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("generate decode failed: {e}")))?;
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
