//! Facade: multi-domain retrieve + critical-claim cross-check.

use super::domain::{DomainIndex, IndexDomain};
use super::hit::{RetrievalChannel, RetrievalHit};
use super::rank::{RankedSet, RetrievalQuery, RetrievalRanker};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum RetrievalError {
    #[error("no domain index registered for {0:?}")]
    DomainMissing(IndexDomain),
    #[error("claim not found for cross-check: {0}")]
    ClaimNotFound(String),
    #[error("cross-check requires a different retrieval path (same domain given twice)")]
    SamePathCrossCheck,
}

/// Report from verifying a critical claim via an alternate domain/channel.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrossCheckReport {
    pub claim_id: String,
    pub primary_domain: IndexDomain,
    pub primary_channel: RetrievalChannel,
    pub alternate_domain: IndexDomain,
    pub alternate_channel: RetrievalChannel,
    pub primary_hit: Option<RetrievalHit>,
    pub alternate_hits: Vec<RetrievalHit>,
    /// Distinct source domains across primary + alternate evidence.
    pub independent_source_diversity: f32,
    /// True when at least one alternate hit shares lexical support with the claim.
    pub corroborated: bool,
}

/// Multi-index retrieval gateway (bible §15).
pub struct RetrievalGateway {
    indexes: Vec<Box<dyn DomainIndex>>,
    ranker: Box<dyn RetrievalRanker>,
}

impl RetrievalGateway {
    pub fn new(ranker: impl RetrievalRanker + 'static) -> Self {
        Self {
            indexes: Vec::new(),
            ranker: Box::new(ranker),
        }
    }

    pub fn register(&mut self, index: Box<dyn DomainIndex>) {
        // Replace existing registration for the same domain (last write wins).
        self.indexes.retain(|i| i.domain() != index.domain());
        self.indexes.push(index);
    }

    pub fn domains(&self) -> Vec<IndexDomain> {
        self.indexes.iter().map(|i| i.domain()).collect()
    }

    /// Search allowed domains, rank, return the smallest relevant set.
    pub fn retrieve(&self, query: &RetrievalQuery) -> Result<RankedSet, RetrievalError> {
        let mut candidates = Vec::new();
        for idx in &self.indexes {
            if !query.allows_domain(idx.domain()) {
                continue;
            }
            candidates.extend(idx.search(query));
        }
        Ok(self.ranker.rank(query, candidates))
    }

    /// Critical claims must be cross-checked through a different retrieval path.
    ///
    /// Scaffold: alternate **domain** is the different path (WEB vs REPOSITORY,
    /// etc.). A future step can also force a different channel inside one domain
    /// (lexical vs semantic), reusing `hawking-index` hybrid legs.
    pub fn cross_check_critical(
        &self,
        claim_id: &str,
        primary_domain: IndexDomain,
        alternate_domain: IndexDomain,
        query_text: &str,
    ) -> Result<CrossCheckReport, RetrievalError> {
        if primary_domain == alternate_domain {
            return Err(RetrievalError::SamePathCrossCheck);
        }

        let primary_q = RetrievalQuery::new(query_text)
            .with_domains(vec![primary_domain])
            .with_limit(8);
        let primary_set = self.retrieve(&primary_q)?;
        let primary_hit = primary_set
            .hits
            .into_iter()
            .map(|e| e.hit)
            .find(|h| h.id == claim_id)
            .or_else(|| {
                // Fall back to top primary hit when id is the provisional claim key.
                self.retrieve(&primary_q)
                    .ok()
                    .and_then(|s| s.hits.into_iter().next().map(|e| e.hit))
            });

        let Some(primary_hit) = primary_hit else {
            return Err(RetrievalError::ClaimNotFound(claim_id.to_string()));
        };

        let alt_q = RetrievalQuery::new(query_text)
            .with_domains(vec![alternate_domain])
            .with_limit(5);
        let alt_set = self.retrieve(&alt_q)?;
        let mut alternate_hits: Vec<_> = alt_set.hits.into_iter().map(|e| e.hit).collect();
        for h in &mut alternate_hits {
            h.retrieval_channel = RetrievalChannel::CrossCheck;
        }

        let mut domains = std::collections::BTreeSet::new();
        domains.insert(primary_hit.source_domain.as_str().to_string());
        for h in &alternate_hits {
            domains.insert(h.source_domain.as_str().to_string());
        }
        let evidence_n = 1 + alternate_hits.len();
        let independent_source_diversity = if evidence_n == 0 {
            0.0
        } else {
            domains.len() as f32 / evidence_n as f32
        };

        let claim_terms: Vec<_> = query_text
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();
        let corroborated = alternate_hits.iter().any(|h| {
            let snip = h.snippet.to_lowercase();
            claim_terms.iter().any(|t| snip.contains(t))
        });

        Ok(CrossCheckReport {
            claim_id: claim_id.to_string(),
            primary_domain,
            primary_channel: primary_hit.retrieval_channel,
            alternate_domain,
            alternate_channel: RetrievalChannel::CrossCheck,
            primary_hit: Some(primary_hit),
            alternate_hits,
            independent_source_diversity,
            corroborated,
        })
    }
}
