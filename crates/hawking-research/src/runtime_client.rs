//! The Research Lab's only path to a model (bible ch.08 §4.1).
//!
//! The bible is emphatic that the lab never hard-depends on a model: every
//! intelligent step (planner decomposition, cited synthesis, parse-time cleanup)
//! goes through a [`RuntimeClient`] trait so tests run against a deterministic
//! stub and production runs against the local Hawking runtime.
//!
//! Rather than invent a parallel HTTP client, this wraps `hawking-orch`'s
//! [`InferenceClient`] (the workspace's single inference seam). `chat()` drives
//! `generate()` and collects the streamed tokens into one string; `embed()`
//! forwards directly. Production wires the real `HawkingHttpClient`; tests wire
//! `StubInferenceClient`.

use futures::future::BoxFuture;
use hawking_orch::InferenceClient;
use hide_core::error::Result;
use hide_core::runtime::{InferenceMessage, InferenceRequest, StreamChunk};
use std::collections::BTreeMap;
use std::sync::Arc;

/// A model turn: a task kind, an optional system preamble, and the user prompt.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub task_kind: String,
    pub system: Option<String>,
    pub prompt: String,
    pub max_output_tokens: usize,
}

impl ChatRequest {
    pub fn new(task_kind: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            task_kind: task_kind.into(),
            system: None,
            prompt: prompt.into(),
            max_output_tokens: 1024,
        }
    }

    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_output_tokens = n;
        self
    }

    fn into_inference(self) -> InferenceRequest {
        let mut messages = Vec::new();
        if let Some(sys) = &self.system {
            messages.push(InferenceMessage {
                role: "system".to_string(),
                content: sys.clone(),
            });
        }
        messages.push(InferenceMessage {
            role: "user".to_string(),
            content: self.prompt.clone(),
        });
        InferenceRequest {
            task_kind: self.task_kind,
            prompt: self.prompt,
            messages,
            max_output_tokens: self.max_output_tokens,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: BTreeMap::new(),
        }
    }
}

/// The model boundary for the Research Lab. Backed by [`InferenceClient`].
pub trait RuntimeClient: Send + Sync {
    /// Embed one text into a vector (used for dedup, retrieval, clustering).
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;

    /// Run a chat turn and return the full collected completion text.
    fn chat<'a>(&'a self, req: ChatRequest) -> BoxFuture<'a, Result<String>>;

    /// Embed a batch (default: sequential single-embeds; an HTTP client may
    /// override with a real batched request).
    fn embed_batch<'a>(&'a self, texts: &'a [String]) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed(t).await?);
            }
            Ok(out)
        })
    }
}

/// Adapts any [`InferenceClient`] into a [`RuntimeClient`].
pub struct InferenceRuntime {
    client: Arc<dyn InferenceClient>,
}

impl InferenceRuntime {
    pub fn new(client: Arc<dyn InferenceClient>) -> Self {
        Self { client }
    }
}

impl RuntimeClient for InferenceRuntime {
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        self.client.embed(text)
    }

    fn chat<'a>(&'a self, req: ChatRequest) -> BoxFuture<'a, Result<String>> {
        let request = req.into_inference();
        Box::pin(async move {
            let mut collected = String::new();
            let mut error: Option<String> = None;
            {
                let mut sink = |chunk: StreamChunk| -> Result<()> {
                    match chunk {
                        StreamChunk::Token { text, .. } => collected.push_str(&text),
                        StreamChunk::Error { message } => error = Some(message),
                        StreamChunk::Done { .. } => {}
                    }
                    Ok(())
                };
                self.client.generate(request, &mut sink).await?;
            }
            if let Some(message) = error {
                return Err(hide_core::HideError::RuntimeUnavailable(message));
            }
            Ok(collected)
        })
    }
}

/// Convenience: build a [`RuntimeClient`] from `StubInferenceClient` for tests.
pub fn stub_runtime(response: impl Into<String>) -> Arc<dyn RuntimeClient> {
    Arc::new(InferenceRuntime::new(Arc::new(
        hawking_orch::StubInferenceClient::new(response),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn chat_collects_streamed_tokens() {
        let rt = stub_runtime("a cited synthesis");
        let out = rt
            .chat(ChatRequest::new("synthesize", "summarize"))
            .await
            .unwrap();
        assert_eq!(out, "a cited synthesis");
    }

    #[tokio::test]
    async fn embed_is_deterministic() {
        let rt = stub_runtime("");
        let a = rt.embed("paged attention").await.unwrap();
        let b = rt.embed("paged attention").await.unwrap();
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn embed_batch_matches_single() {
        let rt = stub_runtime("");
        let texts = vec!["one".to_string(), "two".to_string()];
        let batch = rt.embed_batch(&texts).await.unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], rt.embed("one").await.unwrap());
    }
}
