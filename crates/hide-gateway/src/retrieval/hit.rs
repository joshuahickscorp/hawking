//! Provenance envelope every search result must carry (bible §15).

use serde::{Deserialize, Serialize};

/// BLAKE3 content hash (hex), matching durable HIDE/CAS stamping elsewhere.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(String);

impl ContentHash {
    pub fn of(bytes: impl AsRef<[u8]>) -> Self {
        let dig = blake3::hash(bytes.as_ref());
        Self(dig.to_hex().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ContentHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Who produced the evidence (host, paper site, repo, receipt store, …).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SourceDomainId(String);

impl SourceDomainId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Which retrieval leg produced the hit — maps to `hawking-index::SearchResultSource`
/// plus gateway-specific channels (bundle, cross-check).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalChannel {
    Lexical,
    Symbol,
    Semantic,
    Graph,
    /// Tool/skill catalog match (progressive-disclosure compact index).
    Catalog,
    /// Secondary path used only for critical-claim verification.
    CrossCheck,
}

/// Authority prior ∈ [0, 1]. Higher = more trusted provenance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AuthorityRank(f32);

impl AuthorityRank {
    pub fn new(v: f32) -> Self {
        Self(v.clamp(0.0, 1.0))
    }

    pub fn value(self) -> f32 {
        self.0
    }
}

/// Prompt-injection triage for retrieved text (data, never instructions).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum InjectionStatus {
    Clean,
    Suspected { reason: String },
    Blocked { reason: String },
}

/// One edge in the claim→source graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimEdge {
    /// Stable claim key (often the hit id or a extracted assertion id).
    pub claim_id: String,
    /// Source that supports (or refutes) the claim.
    pub source_id: String,
    pub source_domain: SourceDomainId,
    pub relation: ClaimRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimRelation {
    Supports,
    Mentions,
    Refutes,
    SelfSource,
}

/// One retrieval hit with the full §15 envelope.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalHit {
    pub id: String,
    pub domain: super::domain::IndexDomain,
    pub source_domain: SourceDomainId,
    pub retrieval_channel: RetrievalChannel,
    pub content_hash: ContentHash,
    pub authority_rank: AuthorityRank,
    pub claim_edges: Vec<ClaimEdge>,
    /// Fraction of supporting sources that are independent (filled by ranker).
    pub independent_source_diversity: f32,
    pub injection_status: InjectionStatus,
    pub title: String,
    pub snippet: String,
    /// Raw domain score before global ranking.
    pub score: f32,
}

impl RetrievalHit {
    /// Test/fixture constructor with a self-source claim edge.
    pub fn synthetic(
        id: impl Into<String>,
        domain: super::domain::IndexDomain,
        source: impl Into<String>,
        body: impl Into<String>,
        score: f32,
    ) -> Self {
        let id = id.into();
        let source_domain = SourceDomainId::new(source);
        let body = body.into();
        let content_hash = ContentHash::of(body.as_bytes());
        Self {
            claim_edges: vec![ClaimEdge {
                claim_id: id.clone(),
                source_id: id.clone(),
                source_domain: source_domain.clone(),
                relation: ClaimRelation::SelfSource,
            }],
            id,
            domain,
            source_domain,
            retrieval_channel: RetrievalChannel::Lexical,
            content_hash,
            authority_rank: AuthorityRank::new(0.5),
            independent_source_diversity: 0.0,
            injection_status: InjectionStatus::Clean,
            title: String::new(),
            snippet: body,
            score,
        }
    }
}
