//! Typed records for the five separate indexes (bible §15).

use super::hit::{
    AuthorityRank, ClaimEdge, ClaimRelation, ContentHash, InjectionStatus, RetrievalChannel,
    RetrievalHit, SourceDomainId,
};
use super::rank::RetrievalQuery;
use serde::{Deserialize, Serialize};

/// The five retrieval domains — never mixed in one store.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexDomain {
    Web,
    Repository,
    Tool,
    Experience,
    Skill,
}

impl IndexDomain {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Web => "web",
            Self::Repository => "repository",
            Self::Tool => "tool",
            Self::Experience => "experience",
            Self::Skill => "skill",
        }
    }
}

/// WEB INDEX — papers, documentation, release notes, remote source trees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WebRecord {
    pub id: String,
    pub source_domain: SourceDomainId,
    pub url: String,
    pub title: String,
    pub body: String,
    pub authority_rank: AuthorityRank,
    pub injection_status: InjectionStatus,
}

/// REPOSITORY INDEX — files, symbols, commits, receipts, prior experiments.
///
/// Production backing: `hawking-index` (`SqliteCodeIndex` / hybrid retriever).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepositoryRecord {
    pub id: String,
    pub source_domain: SourceDomainId,
    pub path: String,
    pub symbol: Option<String>,
    pub body: String,
    pub authority_rank: AuthorityRank,
    pub commit: Option<String>,
}

/// TOOL INDEX — MCP/HCLI tools and compatible sets (compact catalog rows).
///
/// Production backing: `hide-core::ToolRegistry` + `extension_registry` manifests
/// + MCP descriptors. Full JSON schemas stay deferred until grant (ToolSearch).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolIndexRecord {
    pub id: String,
    pub source_domain: SourceDomainId,
    pub tool_name: String,
    pub description: String,
    pub schema_digest: ContentHash,
    pub version: String,
    pub effects: Vec<String>,
    pub authority_rank: AuthorityRank,
}

/// EXPERIENCE INDEX — past failures, accepted fixes, benchmark mechanisms.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExperienceRecord {
    pub id: String,
    pub source_domain: SourceDomainId,
    pub summary: String,
    pub failure_tag: Option<String>,
    pub fix_tag: Option<String>,
    pub authority_rank: AuthorityRank,
}

/// SKILL INDEX — verified reusable workflows.
///
/// Production backing: `hide-kernel::skills::SkillStore` (capture-on-success).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillIndexRecord {
    pub id: String,
    pub source_domain: SourceDomainId,
    pub name: String,
    pub trigger: String,
    pub body: String,
    pub success_count: u32,
    pub fail_count: u32,
    pub importance: f32,
    pub authority_rank: AuthorityRank,
}

/// Per-domain search surface. Implementations must never return hits from a
/// different [`IndexDomain`].
pub trait DomainIndex: Send + Sync {
    fn domain(&self) -> IndexDomain;
    fn search(&self, query: &RetrievalQuery) -> Vec<RetrievalHit>;
}

/// Scaffold store: one domain, lexical match, deterministic scoring.
#[derive(Debug, Clone)]
pub struct InMemoryDomainIndex {
    domain: IndexDomain,
    web: Vec<WebRecord>,
    repository: Vec<RepositoryRecord>,
    tools: Vec<ToolIndexRecord>,
    experience: Vec<ExperienceRecord>,
    skills: Vec<SkillIndexRecord>,
}

impl InMemoryDomainIndex {
    pub fn new(domain: IndexDomain) -> Self {
        Self {
            domain,
            web: Vec::new(),
            repository: Vec::new(),
            tools: Vec::new(),
            experience: Vec::new(),
            skills: Vec::new(),
        }
    }

    pub fn insert_web(&mut self, r: WebRecord) {
        debug_assert_eq!(self.domain, IndexDomain::Web);
        self.web.push(r);
    }

    pub fn insert_repository(&mut self, r: RepositoryRecord) {
        debug_assert_eq!(self.domain, IndexDomain::Repository);
        self.repository.push(r);
    }

    pub fn insert_tool(&mut self, r: ToolIndexRecord) {
        debug_assert_eq!(self.domain, IndexDomain::Tool);
        self.tools.push(r);
    }

    pub fn insert_experience(&mut self, r: ExperienceRecord) {
        debug_assert_eq!(self.domain, IndexDomain::Experience);
        self.experience.push(r);
    }

    pub fn insert_skill(&mut self, r: SkillIndexRecord) {
        debug_assert_eq!(self.domain, IndexDomain::Skill);
        self.skills.push(r);
    }
}

fn lexical_score(query: &str, text: &str) -> f32 {
    let q: Vec<&str> = query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .collect();
    if q.is_empty() {
        return 0.0;
    }
    let hay = text.to_lowercase();
    let hits = q
        .iter()
        .filter(|t| hay.contains(&t.to_lowercase()))
        .count() as f32;
    hits / q.len() as f32
}

fn self_edge(id: &str, source: &SourceDomainId) -> ClaimEdge {
    ClaimEdge {
        claim_id: id.to_string(),
        source_id: id.to_string(),
        source_domain: source.clone(),
        relation: ClaimRelation::SelfSource,
    }
}

impl DomainIndex for InMemoryDomainIndex {
    fn domain(&self) -> IndexDomain {
        self.domain
    }

    fn search(&self, query: &RetrievalQuery) -> Vec<RetrievalHit> {
        let mut hits = Vec::new();
        match self.domain {
            IndexDomain::Web => {
                for r in &self.web {
                    let score = lexical_score(&query.text, &format!("{} {}", r.title, r.body));
                    if score <= 0.0 {
                        continue;
                    }
                    hits.push(RetrievalHit {
                        id: r.id.clone(),
                        domain: IndexDomain::Web,
                        source_domain: r.source_domain.clone(),
                        retrieval_channel: RetrievalChannel::Lexical,
                        content_hash: ContentHash::of(r.body.as_bytes()),
                        authority_rank: r.authority_rank,
                        claim_edges: vec![self_edge(&r.id, &r.source_domain)],
                        independent_source_diversity: 0.0,
                        injection_status: r.injection_status.clone(),
                        title: r.title.clone(),
                        snippet: r.body.chars().take(240).collect(),
                        score: score * r.authority_rank.value(),
                    });
                }
            }
            IndexDomain::Repository => {
                for r in &self.repository {
                    let blob = format!(
                        "{} {} {}",
                        r.path,
                        r.symbol.as_deref().unwrap_or(""),
                        r.body
                    );
                    let score = lexical_score(&query.text, &blob);
                    if score <= 0.0 {
                        continue;
                    }
                    let channel = if r.symbol.as_ref().is_some_and(|s| {
                        query
                            .text
                            .to_lowercase()
                            .contains(&s.to_lowercase())
                    }) {
                        RetrievalChannel::Symbol
                    } else {
                        RetrievalChannel::Lexical
                    };
                    hits.push(RetrievalHit {
                        id: r.id.clone(),
                        domain: IndexDomain::Repository,
                        source_domain: r.source_domain.clone(),
                        retrieval_channel: channel,
                        content_hash: ContentHash::of(r.body.as_bytes()),
                        authority_rank: r.authority_rank,
                        claim_edges: vec![self_edge(&r.id, &r.source_domain)],
                        independent_source_diversity: 0.0,
                        injection_status: InjectionStatus::Clean,
                        title: r.symbol.clone().unwrap_or_else(|| r.path.clone()),
                        snippet: r.body.chars().take(240).collect(),
                        score: score * r.authority_rank.value(),
                    });
                }
            }
            IndexDomain::Tool => {
                for r in &self.tools {
                    let score =
                        lexical_score(&query.text, &format!("{} {}", r.tool_name, r.description));
                    if score <= 0.0 {
                        continue;
                    }
                    hits.push(RetrievalHit {
                        id: r.id.clone(),
                        domain: IndexDomain::Tool,
                        source_domain: r.source_domain.clone(),
                        retrieval_channel: RetrievalChannel::Catalog,
                        content_hash: r.schema_digest.clone(),
                        authority_rank: r.authority_rank,
                        claim_edges: vec![self_edge(&r.id, &r.source_domain)],
                        independent_source_diversity: 0.0,
                        injection_status: InjectionStatus::Clean,
                        title: r.tool_name.clone(),
                        snippet: r.description.clone(),
                        score: score * r.authority_rank.value(),
                    });
                }
            }
            IndexDomain::Experience => {
                for r in &self.experience {
                    let score = lexical_score(
                        &query.text,
                        &format!(
                            "{} {} {}",
                            r.summary,
                            r.failure_tag.as_deref().unwrap_or(""),
                            r.fix_tag.as_deref().unwrap_or("")
                        ),
                    );
                    if score <= 0.0 {
                        continue;
                    }
                    hits.push(RetrievalHit {
                        id: r.id.clone(),
                        domain: IndexDomain::Experience,
                        source_domain: r.source_domain.clone(),
                        retrieval_channel: RetrievalChannel::Lexical,
                        content_hash: ContentHash::of(r.summary.as_bytes()),
                        authority_rank: r.authority_rank,
                        claim_edges: vec![self_edge(&r.id, &r.source_domain)],
                        independent_source_diversity: 0.0,
                        injection_status: InjectionStatus::Clean,
                        title: r.failure_tag.clone().unwrap_or_else(|| r.id.clone()),
                        snippet: r.summary.clone(),
                        score: score * r.authority_rank.value(),
                    });
                }
            }
            IndexDomain::Skill => {
                for r in &self.skills {
                    let score =
                        lexical_score(&query.text, &format!("{} {} {}", r.name, r.trigger, r.body));
                    if score <= 0.0 {
                        continue;
                    }
                    let total = (r.success_count + r.fail_count).max(1) as f32;
                    let success_rate = r.success_count as f32 / total;
                    // Mirrors SkillStore: relevance × importance × success.
                    let score = score * r.importance * success_rate * r.authority_rank.value();
                    hits.push(RetrievalHit {
                        id: r.id.clone(),
                        domain: IndexDomain::Skill,
                        source_domain: r.source_domain.clone(),
                        retrieval_channel: RetrievalChannel::Catalog,
                        content_hash: ContentHash::of(r.body.as_bytes()),
                        authority_rank: r.authority_rank,
                        claim_edges: vec![self_edge(&r.id, &r.source_domain)],
                        independent_source_diversity: 0.0,
                        injection_status: InjectionStatus::Clean,
                        title: r.name.clone(),
                        snippet: r.body.chars().take(240).collect(),
                        score,
                    });
                }
            }
        }
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        if hits.len() > query.limit.saturating_mul(4).max(query.limit) {
            // Domain may over-fetch; global ranker trims to query.limit.
            hits.truncate(query.limit.saturating_mul(4).max(8));
        }
        hits
    }
}
