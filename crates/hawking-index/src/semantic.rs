//! The semantic index: hybrid retrieval (lexical ⊕ symbol ⊕ vector) → RRF →
//! rerank (bible §4.7).
//!
//! - `EmbeddingClient` is a swappable trait; `HttpEmbeddingClient` talks to
//!   `hawking-serve` `POST /v1/embeddings`; tests use `StubEmbeddingClient`.
//! - Vectors are stored as f32 in SQLite (see `store`); recall is **cosine over
//!   stored vectors** (no heavy ANN dep), which is exact and fits an IDE shard.
//! - `reciprocal_rank_fusion` + a rerank actually run in `HybridRetriever::search`
//!   (RRF is no longer dead).

use crate::store::SqliteStore;
use futures::future::BoxFuture;
use hide_core::{HideError, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingRecord {
    pub chunk_id: String,
    pub model_id: String,
    pub dim: usize,
    pub vector: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HybridRetrievalWeights {
    pub lexical: f32,
    pub symbol: f32,
    pub semantic: f32,
    pub graph: f32,
}

impl Default for HybridRetrievalWeights {
    fn default() -> Self {
        // Per ground truth + SOTA: lexical/symbol carry recall; vector re-ranks
        // (low weight today because embed() is a logits proxy).
        Self {
            lexical: 1.0,
            symbol: 1.0,
            semantic: 0.3,
            graph: 0.75,
        }
    }
}

/// `RRF(d) = Σ 1/(k + rank)`, k=60 (Cormack 2009; the Elasticsearch default).
pub fn reciprocal_rank_fusion(ranks: &[usize], k: f32) -> f32 {
    ranks.iter().map(|rank| 1.0 / (k + *rank as f32)).sum()
}

pub const RRF_K: f32 = 60.0;

/// A swappable embedding client (the live runtime is NOT up during tests).
pub trait EmbeddingClient: Send + Sync {
    /// Embed a batch of texts. Returns one vector per input.
    fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>>;
    /// The model id stamped on stored vectors (for versioning / lazy re-embed).
    fn model_id(&self) -> String;
}

/// HTTP client to `hawking-serve` `POST /v1/embeddings` (OpenAI-shaped).
pub struct HttpEmbeddingClient {
    base_url: String,
    model_id: String,
    client: reqwest::Client,
}

impl HttpEmbeddingClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            model_id: "logits-proxy:default".to_string(),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = id.into();
        self
    }
}

#[derive(Serialize)]
struct EmbeddingsRequest {
    input: Vec<String>,
    encoding_format: &'static str,
}

#[derive(Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingDatum>,
}

#[derive(Deserialize)]
struct EmbeddingDatum {
    embedding: Vec<f32>,
}

impl EmbeddingClient for HttpEmbeddingClient {
    fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
        Box::pin(async move {
            if texts.is_empty() {
                return Ok(Vec::new());
            }
            let url = format!("{}/v1/embeddings", self.base_url.trim_end_matches('/'));
            let resp = self
                .client
                .post(&url)
                .json(&EmbeddingsRequest {
                    input: texts,
                    encoding_format: "float",
                })
                .send()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings request: {e}")))?;
            if !resp.status().is_success() {
                return Err(HideError::RuntimeUnavailable(format!(
                    "embeddings status {}",
                    resp.status()
                )));
            }
            let body: EmbeddingsResponse = resp
                .json()
                .await
                .map_err(|e| HideError::RuntimeUnavailable(format!("embeddings decode: {e}")))?;
            Ok(body.data.into_iter().map(|d| d.embedding).collect())
        })
    }

    fn model_id(&self) -> String {
        self.model_id.clone()
    }
}

/// A deterministic stub embedding client for tests (no live runtime).
///
/// Produces a small bag-of-chars vector so cosine similarity is meaningful for
/// tests without any network.
pub struct StubEmbeddingClient {
    pub model_id: String,
    pub dim: usize,
}

impl Default for StubEmbeddingClient {
    fn default() -> Self {
        Self {
            model_id: "stub-embed:test".to_string(),
            dim: 32,
        }
    }
}

impl EmbeddingClient for StubEmbeddingClient {
    fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
        let dim = self.dim;
        Box::pin(async move { Ok(texts.iter().map(|t| bag_of_chars(t, dim)).collect()) })
    }
    fn model_id(&self) -> String {
        self.model_id.clone()
    }
}

fn bag_of_chars(text: &str, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dim];
    for b in text.bytes() {
        v[(b as usize) % dim] += 1.0;
    }
    l2_normalize(&mut v);
    v
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
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
    let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
    dot / denom
}

fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// One fused, reranked result.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedHit {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
    pub snippet: String,
    /// Combined RRF score (post-rerank ordering preserved in returned order).
    pub score: f32,
    pub legs: Vec<String>,
}

/// A leg's ranked output: ordered keys (a key identifies a candidate location).
#[derive(Debug, Clone, Default)]
pub struct LegRanking {
    pub name: String,
    pub weight: f32,
    /// Ordered candidate keys (rank 0 = best). Key = "file:start_line".
    pub ranked_keys: Vec<String>,
}

/// Fuse multiple leg rankings via weighted RRF.
///
/// Returns candidate keys sorted by fused score descending.
pub fn fuse_legs(legs: &[LegRanking], k: f32) -> Vec<(String, f32)> {
    use std::collections::HashMap;
    let mut scores: HashMap<String, f32> = HashMap::new();
    for leg in legs {
        for (rank, key) in leg.ranked_keys.iter().enumerate() {
            *scores.entry(key.clone()).or_insert(0.0) += leg.weight * (1.0 / (k + rank as f32));
        }
    }
    let mut out: Vec<(String, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out
}

/// A reranker over fused candidates. Today: a lexical-overlap precision pass
/// (a free, deterministic stand-in for the bible's local-LLM listwise rerank;
/// the LLM rerank slots in behind this same boundary).
pub trait Reranker: Send + Sync {
    fn rerank(&self, query: &str, candidates: Vec<FusedHit>) -> Vec<FusedHit>;
}

pub struct LexicalOverlapReranker;

impl Reranker for LexicalOverlapReranker {
    fn rerank(&self, query: &str, mut candidates: Vec<FusedHit>) -> Vec<FusedHit> {
        let q_terms: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|t| !t.is_empty())
            .map(|t| t.to_lowercase())
            .collect();
        for c in candidates.iter_mut() {
            let snip = c.snippet.to_lowercase();
            let overlap = q_terms.iter().filter(|t| snip.contains(t.as_str())).count() as f32;
            // blend RRF score with overlap (rerank boosts precise term matches)
            c.score = c.score * 0.5 + overlap * 0.1;
        }
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates
    }
}

/// The hybrid retriever: runs the vector leg (cosine over stored vectors),
/// fuses with externally-supplied lexical/symbol legs via RRF, then reranks.
pub struct HybridRetriever<'a, E: EmbeddingClient> {
    store: &'a SqliteStore,
    embedder: &'a E,
    weights: HybridRetrievalWeights,
}

impl<'a, E: EmbeddingClient> HybridRetriever<'a, E> {
    pub fn new(store: &'a SqliteStore, embedder: &'a E) -> Self {
        Self {
            store,
            embedder,
            weights: HybridRetrievalWeights::default(),
        }
    }

    pub fn with_weights(mut self, weights: HybridRetrievalWeights) -> Self {
        self.weights = weights;
        self
    }

    /// The vector leg: embed the query, cosine over stored vectors, return the
    /// top candidate keys ordered by similarity.
    pub async fn vector_leg(&self, query: &str, k: usize) -> Result<LegRanking> {
        let qvec = self.embedder.embed(vec![query.to_string()]).await?;
        let qvec = match qvec.into_iter().next() {
            Some(v) => v,
            None => return Ok(LegRanking::default()),
        };
        let mut scored: Vec<(String, f32, String)> = self
            .store
            .all_vectors()?
            .into_iter()
            .map(|sv| {
                let key = format!("{}:{}", sv.file, sv.start_line);
                let sim = cosine(&qvec, &sv.vector);
                (key, sim, sv.file)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(LegRanking {
            name: "vector".to_string(),
            weight: self.weights.semantic,
            ranked_keys: scored.into_iter().map(|(key, _, _)| key).collect(),
        })
    }

    /// Full search: vector leg + caller-supplied lexical & symbol legs → RRF →
    /// rerank. The caller passes lexical/symbol rankings (from `store`) so this
    /// stays decoupled from how those legs are produced.
    ///
    /// The vector (semantic) leg always runs; use [`search_with_legs`] to skip it.
    ///
    /// [`search_with_legs`]: HybridRetriever::search_with_legs
    pub async fn search(
        &self,
        query: &str,
        lexical: LegRanking,
        symbol: LegRanking,
        snippets: &std::collections::HashMap<String, FusedHit>,
        reranker: &dyn Reranker,
        k_final: usize,
    ) -> Result<Vec<FusedHit>> {
        self.search_with_legs(query, lexical, symbol, snippets, reranker, k_final, true)
            .await
    }

    /// As [`search`](HybridRetriever::search), but `include_semantic` toggles the
    /// vector leg. When `false` the embedder is never invoked (no `embed()` call,
    /// no cosine pass) — the result is a pure lexical⊕symbol fusion. This is what
    /// lets `include_semantic` on a query actually turn the vector leg off.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_with_legs(
        &self,
        query: &str,
        lexical: LegRanking,
        symbol: LegRanking,
        snippets: &std::collections::HashMap<String, FusedHit>,
        reranker: &dyn Reranker,
        k_final: usize,
        include_semantic: bool,
    ) -> Result<Vec<FusedHit>> {
        let mut legs = vec![lexical, symbol];
        if include_semantic {
            legs.push(self.vector_leg(query, 50).await?);
        }
        let fused = fuse_legs(&legs, RRF_K);

        let mut hits: Vec<FusedHit> = Vec::new();
        for (key, score) in fused.into_iter().take(50) {
            if let Some(base) = snippets.get(&key) {
                let mut h = base.clone();
                h.score = score;
                hits.push(h);
            }
        }
        let reranked = reranker.rerank(query, hits);
        Ok(reranked.into_iter().take(k_final).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn rrf_kernel_sums_reciprocals() {
        let v = reciprocal_rank_fusion(&[0, 1], 60.0);
        assert!((v - (1.0 / 60.0 + 1.0 / 61.0)).abs() < 1e-6);
    }

    #[test]
    fn fuse_legs_weights_and_orders() {
        let a = LegRanking {
            name: "lex".into(),
            weight: 1.0,
            ranked_keys: vec!["x:1".into(), "y:1".into()],
        };
        let b = LegRanking {
            name: "vec".into(),
            weight: 0.3,
            ranked_keys: vec!["y:1".into(), "x:1".into()],
        };
        let fused = fuse_legs(&[a, b], 60.0);
        // x:1 is rank0 in the heavier leg → should lead
        assert_eq!(fused[0].0, "x:1");
    }

    #[test]
    fn cosine_basic() {
        assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[tokio::test]
    async fn vector_leg_uses_cosine_over_stored_vectors() {
        let store = SqliteStore::open_in_memory().unwrap();
        let out = crate::parse::parse_source("q.rs", "pub fn alpha() { compute(); }");
        let chunks = crate::parse::chunk_file("q.rs", "pub fn alpha() { compute(); }");
        store
            .upsert_file(
                "q.rs",
                "rust",
                "h",
                "ok",
                "pub fn alpha() { compute(); }",
                &out.symbols,
                &out.occurrences,
                &chunks,
                1,
            )
            .unwrap();
        let embedder = StubEmbeddingClient::default();
        // embed and store the chunk vector
        let pending = store.pending_chunks(10).unwrap();
        let txt = "pub fn alpha() { compute(); }";
        let vecs = embedder.embed(vec![txt.to_string()]).await.unwrap();
        store
            .store_vector(&pending[0].chunk_id, &embedder.model_id(), &vecs[0])
            .unwrap();

        let retriever = HybridRetriever::new(&store, &embedder);
        let leg = retriever.vector_leg("compute alpha", 5).await.unwrap();
        assert!(!leg.ranked_keys.is_empty(), "vector leg must return hits");
    }

    /// An embedder that records how many times `embed()` was invoked, so a test
    /// can prove the vector leg was (or wasn't) run.
    struct CountingEmbedder {
        calls: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    }
    impl EmbeddingClient for CountingEmbedder {
        fn embed<'a>(&'a self, texts: Vec<String>) -> BoxFuture<'a, Result<Vec<Vec<f32>>>> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Box::pin(async move { Ok(texts.iter().map(|t| bag_of_chars(t, 16)).collect()) })
        }
        fn model_id(&self) -> String {
            "counting:test".into()
        }
    }

    #[tokio::test]
    async fn include_semantic_toggles_vector_leg() {
        use std::sync::atomic::Ordering;
        use std::sync::Arc;
        let store = SqliteStore::open_in_memory().unwrap();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let embedder = CountingEmbedder {
            calls: calls.clone(),
        };
        let retriever = HybridRetriever::new(&store, &embedder);

        let snippets: HashMap<String, FusedHit> = HashMap::new();
        let lexical = LegRanking {
            name: "lexical".into(),
            weight: 1.0,
            ranked_keys: vec![],
        };
        let symbol = LegRanking {
            name: "symbol".into(),
            weight: 1.0,
            ranked_keys: vec![],
        };

        // include_semantic = false → embedder must NOT be called.
        retriever
            .search_with_legs(
                "q",
                lexical.clone(),
                symbol.clone(),
                &snippets,
                &LexicalOverlapReranker,
                5,
                false,
            )
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "vector leg ran despite include_semantic=false"
        );

        // include_semantic = true → embedder IS called exactly once (the query embed).
        retriever
            .search_with_legs(
                "q",
                lexical,
                symbol,
                &snippets,
                &LexicalOverlapReranker,
                5,
                true,
            )
            .await
            .unwrap();
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "vector leg should have embedded the query"
        );
    }

    #[tokio::test]
    async fn full_search_fuses_and_reranks() {
        let store = SqliteStore::open_in_memory().unwrap();
        let embedder = StubEmbeddingClient::default();
        let retriever = HybridRetriever::new(&store, &embedder);

        let mut snippets = HashMap::new();
        snippets.insert(
            "a.rs:1".to_string(),
            FusedHit {
                file: "a.rs".into(),
                start_line: 1,
                end_line: 2,
                snippet: "fn target_function() {}".into(),
                score: 0.0,
                legs: vec![],
            },
        );
        let lexical = LegRanking {
            name: "lexical".into(),
            weight: 1.0,
            ranked_keys: vec!["a.rs:1".into()],
        };
        let symbol = LegRanking {
            name: "symbol".into(),
            weight: 1.0,
            ranked_keys: vec!["a.rs:1".into()],
        };
        let hits = retriever
            .search(
                "target_function",
                lexical,
                symbol,
                &snippets,
                &LexicalOverlapReranker,
                10,
            )
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].file, "a.rs");
    }
}
