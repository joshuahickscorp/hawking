//! Embedding client seam (bible §4.2.2 / §4.6.3).
//!
//! Relevance and redundancy scoring need vectors. The live path calls the
//! runtime's `/v1/embeddings` (OpenAI-compatible) so candidates are embedded by
//! the *same model* that will read them. The runtime is not up during tests, so
//! all live calls go behind a trait with a deterministic hashing stub.

use async_trait::async_trait;
use hide_core::error::{HideError, Result};
use serde::Deserialize;

/// A source of text embeddings. Implementors are cheap to clone or `Arc`-wrap.
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Embed a batch of texts. Returns one vector per input, in order.
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// Embed a single text (default: batch of one).
    async fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(&[text.to_string()]).await?;
        v.pop()
            .ok_or_else(|| HideError::RuntimeUnavailable("empty embedding response".into()))
    }

    /// Embedding dimensionality (for callers that pre-allocate). Best-effort.
    fn dim(&self) -> usize {
        0
    }
}

/// Cosine similarity between two equal-length vectors, in `[-1, 1]`. Returns 0
/// for mismatched or empty vectors (caller treats that as "no signal").
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// `reqwest`-backed client hitting the runtime's `/v1/embeddings`.
#[derive(Clone)]
pub struct HttpEmbeddingClient {
    base_url: String,
    model: String,
    client: reqwest::Client,
}

impl HttpEmbeddingClient {
    /// `base_url` is e.g. `http://127.0.0.1:8080`; the endpoint `/v1/embeddings`
    /// is appended.
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

#[async_trait]
impl EmbeddingClient for HttpEmbeddingClient {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let resp = self
            .client
            .post(format!("{}/v1/embeddings", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings request: {e}")))?;
        if !resp.status().is_success() {
            return Err(HideError::RuntimeUnavailable(format!(
                "embeddings HTTP {}",
                resp.status()
            )));
        }
        let parsed: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings decode: {e}")))?;
        Ok(parsed.data.into_iter().map(|d| d.embedding).collect())
    }
}

/// A deterministic, dependency-free embedding stub for tests and offline use.
/// Hashes whitespace tokens into a fixed-width bag-of-words vector — similar
/// texts share dimensions, so cosine is meaningful (not random) without a model.
#[derive(Clone)]
pub struct HashingEmbeddingClient {
    dim: usize,
}

impl Default for HashingEmbeddingClient {
    fn default() -> Self {
        Self { dim: 256 }
    }
}

impl HashingEmbeddingClient {
    pub fn with_dim(dim: usize) -> Self {
        Self { dim: dim.max(1) }
    }

    fn embed_text(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for tok in text.split_whitespace() {
            let lower = tok.to_lowercase();
            // blake3 the token → bucket index; deterministic and well-spread.
            let h = blake3::hash(lower.as_bytes());
            let bytes = h.as_bytes();
            let idx =
                (u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize) % self.dim;
            v[idx] += 1.0;
        }
        v
    }
}

#[async_trait]
impl EmbeddingClient for HashingEmbeddingClient {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.embed_text(t)).collect())
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn hashing_embeddings_make_similar_text_closer() {
        let c = HashingEmbeddingClient::default();
        let q = c.embed_one("database migration sqlx pool").await.unwrap();
        let near = c
            .embed_one("the database pool is built with sqlx")
            .await
            .unwrap();
        let far = c
            .embed_one("rocket telemetry orbital insertion burn")
            .await
            .unwrap();
        let s_near = cosine(&q, &near);
        let s_far = cosine(&q, &far);
        assert!(s_near > s_far, "near={s_near} far={s_far}");
        assert!(s_near > 0.0);
    }

    #[test]
    fn cosine_handles_degenerate() {
        assert_eq!(cosine(&[], &[]), 0.0);
        assert_eq!(cosine(&[1.0], &[0.0]), 0.0);
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    }
}
