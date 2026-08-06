//! Retrieval ranking: smallest relevant evidence set, not everything.

use super::domain::IndexDomain;
use super::hit::{InjectionStatus, RetrievalHit};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Query against one or more domain indexes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalQuery {
    pub text: String,
    /// Max hits to return after ranking (bible: smallest relevant set).
    pub limit: usize,
    /// Empty = all registered domains.
    pub domains: Vec<IndexDomain>,
    /// When true, drop `InjectionStatus::Suspected` from the final set.
    pub exclude_suspected_injection: bool,
}

impl RetrievalQuery {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            limit: 8,
            domains: Vec::new(),
            exclude_suspected_injection: false,
        }
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit.max(1);
        self
    }

    pub fn with_domains(mut self, domains: Vec<IndexDomain>) -> Self {
        self.domains = domains;
        self
    }

    pub fn exclude_suspected(mut self) -> Self {
        self.exclude_suspected_injection = true;
        self
    }

    pub fn allows_domain(&self, d: IndexDomain) -> bool {
        self.domains.is_empty() || self.domains.contains(&d)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedEvidence {
    pub hit: RetrievalHit,
    pub final_score: f32,
}

/// Output of a ranker: a trimmed set plus accounting for honesty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedSet {
    pub hits: Vec<RankedEvidence>,
    pub candidates_considered: usize,
    /// Fraction of distinct source domains among returned hits.
    pub independent_source_diversity: f32,
}

impl RankedSet {
    pub fn compute_diversity(&self) -> f32 {
        if self.hits.is_empty() {
            return 0.0;
        }
        let domains: BTreeSet<_> = self
            .hits
            .iter()
            .map(|h| h.hit.source_domain.as_str())
            .collect();
        // Fraction of hits that come from distinct source domains.
        // 1 hit / 1 domain → 1.0; 2 hits / 2 domains → 1.0; 4 hits / 1 domain → 0.25.
        domains.len() as f32 / self.hits.len() as f32
    }

    pub fn with_diversity(mut self) -> Self {
        self.independent_source_diversity = self.compute_diversity();
        for h in &mut self.hits {
            h.hit.independent_source_diversity = self.independent_source_diversity;
        }
        self
    }
}

/// Weights for the global ranker (domain score already includes authority).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ScoringWeights {
    pub relevance: f32,
    pub authority: f32,
    pub injection_penalty: f32,
}

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            relevance: 1.0,
            authority: 0.35,
            injection_penalty: 0.85,
        }
    }
}

/// Pluggable ranking interface.
pub trait RetrievalRanker: Send + Sync {
    fn rank(&self, query: &RetrievalQuery, candidates: Vec<RetrievalHit>) -> RankedSet;
}

/// Default scaffold ranker: score × authority, penalize injection, hard limit.
#[derive(Debug, Clone, Default)]
pub struct MinimalSetRanker {
    pub weights: ScoringWeights,
}

impl RetrievalRanker for MinimalSetRanker {
    fn rank(&self, query: &RetrievalQuery, candidates: Vec<RetrievalHit>) -> RankedSet {
        let candidates_considered = candidates.len();
        let mut scored: Vec<RankedEvidence> = candidates
            .into_iter()
            .filter(|h| {
                if query.exclude_suspected_injection {
                    !matches!(h.injection_status, InjectionStatus::Suspected { .. })
                } else {
                    true
                }
            })
            .filter(|h| !matches!(h.injection_status, InjectionStatus::Blocked { .. }))
            .map(|h| {
                let mut final_score =
                    h.score * self.weights.relevance + h.authority_rank.value() * self.weights.authority;
                if matches!(h.injection_status, InjectionStatus::Suspected { .. }) {
                    final_score *= 1.0 - self.weights.injection_penalty;
                }
                RankedEvidence {
                    hit: h,
                    final_score,
                }
            })
            .collect();

        scored.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.hit.id.cmp(&b.hit.id))
        });
        scored.truncate(query.limit);

        RankedSet {
            hits: scored,
            candidates_considered,
            independent_source_diversity: 0.0,
        }
        .with_diversity()
    }
}
