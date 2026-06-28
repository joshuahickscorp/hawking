//! The uniform inference seam.
//!
//! [`InferenceClient`] is the single boundary the orchestrator (and `hide-kernel`
//! via `KernelRuntimeClient`) crosses to reach a live model. There are three
//! capabilities behind it, mirroring `hawking-serve`'s HTTP surface:
//!
//! * [`InferenceClient::generate`] — streaming completion / chat
//!   (`/v1/hawking/generate` native SSE, or `/v1/chat/completions` OpenAI SSE).
//! * [`InferenceClient::embed`] — a vector embedding (`/v1/embeddings`), the
//!   embedder role's only capability.
//!
//! Live HTTP is gated behind this trait; tests use [`StubInferenceClient`].

use futures::future::BoxFuture;
use hide_core::error::Result;
use hide_core::runtime::{GenerationStats, InferenceRequest, StreamChunk, TokenSink};

/// The boundary every model call crosses. Implemented by [`crate::http_client`]
/// for live HTTP and by [`StubInferenceClient`] for tests / offline routing.
pub trait InferenceClient: Send + Sync {
    /// Stream a completion. The sink receives `Token` chunks then a terminal
    /// `Done` (or `Error`). Returns aggregate [`GenerationStats`].
    fn generate<'a>(
        &'a self,
        request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>>;

    /// Embed a single text into a vector (`/v1/embeddings`). The embedder role
    /// is driven entirely through this method.
    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>>;
}

/// A deterministic test double. `generate` emits `response` as one token then a
/// `Done`; `embed` returns a stable hashed pseudo-vector so retrieval/voting
/// tests are reproducible without a runtime.
#[derive(Debug, Clone)]
pub struct StubInferenceClient {
    pub response: String,
    /// Dimension of the deterministic embedding vector.
    pub embed_dim: usize,
}

impl StubInferenceClient {
    pub fn new(response: impl Into<String>) -> Self {
        Self {
            response: response.into(),
            embed_dim: 8,
        }
    }
}

impl Default for StubInferenceClient {
    fn default() -> Self {
        Self::new(String::new())
    }
}

/// A stable, content-derived pseudo-embedding: hashes byte-windows into buckets
/// so identical inputs map to identical vectors and similar inputs overlap.
/// Real enough for cosine-based voting/dedup tests; not a semantic embedding.
pub fn deterministic_embedding(text: &str, dim: usize) -> Vec<f32> {
    let dim = dim.max(1);
    let mut v = vec![0.0f32; dim];
    for token in text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()) {
        // FNV-1a over the lowercased token.
        let mut h: u64 = 0xcbf29ce484222325;
        for b in token.to_ascii_lowercase().bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let bucket = (h % dim as u64) as usize;
        v[bucket] += 1.0;
    }
    // L2-normalize so cosine similarity is a dot product.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

impl InferenceClient for StubInferenceClient {
    fn generate<'a>(
        &'a self,
        _request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        Box::pin(async move {
            sink(StreamChunk::Token {
                token_id: None,
                text: self.response.clone(),
            })?;
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: None,
            })?;
            Ok(GenerationStats {
                input_tokens: 0,
                output_tokens: 1,
                decode_tokens_per_second: None,
            })
        })
    }

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        let dim = self.embed_dim;
        let owned = text.to_string();
        Box::pin(async move { Ok(deterministic_embedding(&owned, dim)) })
    }
}

/// A scripted stub that returns a different completion on each successive
/// `generate` call — used to test the escalation cascade (cheap role stumbles,
/// stronger role succeeds).
#[derive(Debug)]
pub struct ScriptedInferenceClient {
    responses: parking_lot::Mutex<std::collections::VecDeque<String>>,
    fallback: String,
}

impl ScriptedInferenceClient {
    pub fn new(responses: impl IntoIterator<Item = String>) -> Self {
        let responses: std::collections::VecDeque<String> = responses.into_iter().collect();
        let fallback = responses.back().cloned().unwrap_or_default();
        Self {
            responses: parking_lot::Mutex::new(responses),
            fallback,
        }
    }
}

impl InferenceClient for ScriptedInferenceClient {
    fn generate<'a>(
        &'a self,
        _request: InferenceRequest,
        sink: TokenSink<'a>,
    ) -> BoxFuture<'a, Result<GenerationStats>> {
        let next = self
            .responses
            .lock()
            .pop_front()
            .unwrap_or_else(|| self.fallback.clone());
        Box::pin(async move {
            sink(StreamChunk::Token {
                token_id: None,
                text: next,
            })?;
            sink(StreamChunk::Done {
                reason: "stop".to_string(),
                stats: None,
            })?;
            Ok(GenerationStats {
                input_tokens: 0,
                output_tokens: 1,
                decode_tokens_per_second: None,
            })
        })
    }

    fn embed<'a>(&'a self, text: &'a str) -> BoxFuture<'a, Result<Vec<f32>>> {
        let owned = text.to_string();
        Box::pin(async move { Ok(deterministic_embedding(&owned, 8)) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn stub_embeddings_are_deterministic_and_normalized() {
        let client = StubInferenceClient::new("hello");
        let a = client.embed("fn main() {}").await.unwrap();
        let b = client.embed("fn main() {}").await.unwrap();
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5 || norm == 0.0);
    }

    #[tokio::test]
    async fn scripted_client_advances_per_call() {
        let client =
            ScriptedInferenceClient::new(vec!["first".to_string(), "second".to_string()]);
        let mut got = Vec::new();
        let mut sink = |chunk: StreamChunk| {
            if let StreamChunk::Token { text, .. } = chunk {
                got.push(text);
            }
            Ok(())
        };
        let req = InferenceRequest {
            task_kind: "t".into(),
            prompt: "p".into(),
            messages: Vec::new(),
            max_output_tokens: 1,
            sampler: None,
            grammar: None,
            want_logprobs: false,
            metadata: Default::default(),
        };
        client.generate(req.clone(), &mut sink).await.unwrap();
        client.generate(req, &mut sink).await.unwrap();
        assert_eq!(got, vec!["first".to_string(), "second".to_string()]);
    }
}
