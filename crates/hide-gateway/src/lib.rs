//! HCLI Retrieval Gateway + Tool Gateway (ascension bible §§15–16).
//!
//! **Scaffold status:** typed index records, retrieval ranking, tool-bundle
//! retrieval, enforcement surface, and failure classification are real code
//! with unit tests. Live WEB/REPOSITORY/TOOL/EXPERIENCE/SKILL backends, MCP
//! registration, and model-facing packing are **not** implemented here.
//!
//! ## Existing patterns reused (do not reinvent)
//!
//! | Pattern in this session / repo | Gateway reuse |
//! |---|---|
//! | `hawking-index` hybrid legs + RRF + `SearchResultSource` | multi-channel retrieval + channel identity on hits |
//! | `hide-kernel::extension_registry` progressive disclosure | tool index compact-first; load full schema only on grant |
//! | `hide-core::tool::ToolSpec` / annotations / effects | tool health + schema + effect boundaries |
//! | `hide-kernel::skills::SkillStore` ranking | SKILL index rank = relevance × importance × success |
//! | `hide-kernel::subagent` role isolation | tool bundles as role-scoped mutually-useful sets |
//! | Agent tool `subagent_type` + capability modes | profile-gated tool sets (read-only / execute / gate) |
//! | ToolSearch deferred-tool pattern | never inject every tool schema; retrieve smallest set |
//! | grok-orchestration MCP tiers (sandbox vs `gate`) | tool health + session affinity + profile policy |
//!
//! Integration later: REPOSITORY → `hawking-index`; TOOL → `hide-core::ToolRegistry`
//! + MCP descriptors; SKILL → `SkillStore`; EXPERIENCE → sealed receipts /
//! negative-science inheritance (bible §32).

#![forbid(unsafe_code)]

pub mod retrieval;
pub mod tools;

pub use retrieval::{
    AuthorityRank, ClaimEdge, ContentHash, CrossCheckReport, DomainIndex, ExperienceRecord,
    IndexDomain, InjectionStatus, RankedEvidence, RankedSet, RepositoryRecord, RetrievalChannel,
    RetrievalError, RetrievalGateway, RetrievalHit, RetrievalQuery, RetrievalRanker,
    SkillIndexRecord, SourceDomainId, ToolIndexRecord, WebRecord,
};
pub use tools::{
    EffectBoundary, FailureClass, KernelBundle, ModelFailureKind, SessionAffinity, ToolBundle,
    ToolDefectKind, ToolEnforcement, ToolGateway, ToolGatewayError, ToolHealth, ToolHealthStatus,
    ToolPolicy, ToolRef, ToolVersion,
};
