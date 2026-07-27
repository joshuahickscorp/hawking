//! Outcome-governed durable memory + revalidation (bible sec 21-22, sec 78.1 #16).
//!
//! A [`MemoryRecord`] is a durable, provenance-carrying CLAIM about a workspace
//! (a decision, a file fact, a constraint, a failed approach). Unlike the Spine B
//! Project Brain (the `hawking_context::MemoryStore` sqlite store), these records
//! are GOVERNED by outcome and REVALIDATED against the repo on disk:
//!
//! * Every record carries provenance -- `source`, `author`, `created_ms`, and a
//!   `confidence` -- plus the `citations` (path or `path#symbol` refs) that ground
//!   its claim, and the `invalidation` conditions that would retire it.
//! * OUTCOME GOVERNANCE: [`MemoryRecord::record_outcome`] raises `outcome_score`
//!   and `use_count` on a success and lowers the score on a failure; a score that
//!   falls below [`QUARANTINE_FLOOR`] flips the record to `Quarantined`, so a
//!   claim that keeps steering the agent wrong stops entering context.
//! * REVALIDATION: [`resolve_citation`] checks each citation against the CURRENT
//!   repo on disk (a cited path must exist; a `path#symbol` file must exist AND
//!   contain the symbol via a lexical scan). A record whose citation no longer
//!   resolves is QUARANTINED (see `BackendHost::memory_revalidate`).
//!
//! Only `Active` (and not-expired) records are eligible to enter context.
//!
//! SEMANTIC / meaning revalidation -- deciding whether a claim is still TRUE in
//! spirit even when its citations still resolve -- is `DEFERRED_MODEL_REQUIRED`:
//! nothing in this module ever loads or calls a model. The citation check is a
//! deterministic, lexical, disk-grounded gate.

use hide_core::ids::{now_micros, now_ms};
use hide_core::persistence::DynKeyValueStore;
use hide_core::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The neutral starting outcome score for a fresh record (governance is relative
/// to this midpoint: successes climb toward 1.0, failures fall toward 0.0).
pub const INITIAL_OUTCOME_SCORE: f32 = 0.5;

/// Below this floor an outcome-governed record is `Quarantined` (it stops being
/// eligible to enter context). A single failure from the neutral start does not
/// cross it; repeated failures do.
pub const QUARANTINE_FLOOR: f32 = 0.25;

/// How much a recorded SUCCESS raises `outcome_score` (clamped to 1.0).
pub const SUCCESS_DELTA: f32 = 0.1;

/// How much a recorded FAILURE lowers `outcome_score` (clamped to 0.0).
pub const FAILURE_DELTA: f32 = 0.2;

/// The privacy class of a memory record: how freely its `claim` may be shared or
/// surfaced. Snake_case so it round-trips in the KV store; `Private` is the
/// conservative default so a record written before this field existed, or minted
/// without an explicit class, is treated as the most restricted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyClass {
    /// Freely shareable (public docs, OSS facts).
    Public,
    /// Shareable within the team/workspace.
    Team,
    /// Restricted to this user (the conservative default).
    #[default]
    Private,
    /// Sensitive: never surfaced outside a strictly-scoped, audited path.
    Secret,
}

/// The scope a memory record is bound to, each carrying its own id string (bible
/// sec 21-22). A `Session` claim is local to one thread; a `Repo` claim holds for
/// a codebase; a `User` claim holds across a person's workspaces. Adjacently
/// tagged (`{ "kind": "repo", "id": "..." }`) so it round-trips in the KV store
/// and compares by value for scoped listing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "id")]
pub enum MemoryScope {
    Session(String),
    Repo(String),
    User(String),
}

impl MemoryScope {
    /// The snake_case kind label (`session` / `repo` / `user`).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Session(_) => "session",
            Self::Repo(_) => "repo",
            Self::User(_) => "user",
        }
    }

    /// The scope's id string.
    pub fn id(&self) -> &str {
        match self {
            Self::Session(id) | Self::Repo(id) | Self::User(id) => id,
        }
    }

    /// A stable `kind:id` key for reason strings and derived ids.
    pub fn key(&self) -> String {
        format!("{}:{}", self.kind(), self.id())
    }
}

/// The lifecycle status of a memory record (bible sec 21-22). Snake_case so it
/// round-trips; `Active` is the default so a record written before this field
/// existed still deserializes.
///
/// * `Active`: eligible to enter context (subject to expiry).
/// * `Quarantined`: withheld -- either its outcome score fell below the floor, or
///   a citation no longer resolves against the repo on disk.
/// * `Superseded`: replaced by a newer record (history is preserved, not erased;
///   the two are linked via `superseded_by` / `supersedes`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryStatus {
    #[default]
    Active,
    Quarantined,
    Superseded,
}

/// A durable, provenance-carrying, outcome-governed memory record (bible sec
/// 21-22). Stored in the KV [`MemoryLedger::NAMESPACE`] namespace keyed by
/// `memory_id`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub memory_id: String,
    pub scope: MemoryScope,
    /// The claim itself (a decision, a file fact, a constraint, a failed approach).
    pub claim: String,
    /// Where the claim came from (a tool run, a doc, a conversation turn).
    pub source: String,
    /// Who authored it (an agent role, a user, a subsystem).
    pub author: String,
    pub created_ms: u64,
    /// When the claim was last revalidated against the repo (bumped by revalidate).
    pub last_validated_ms: u64,
    /// Initial trust in the claim (0.0..=1.0), distinct from the governed score.
    pub confidence: f32,
    /// Grounding refs: a repo-relative `path`, or a `path#symbol` ref.
    pub citations: Vec<String>,
    /// Conditions that would retire the claim (human-readable triggers).
    pub invalidation: Vec<String>,
    pub privacy: PrivacyClass,
    /// Optional hard expiry (ms since epoch): past it the record is not eligible.
    pub expiry_ms: Option<u64>,
    /// How many times the claim has been exercised (raised by every outcome).
    pub use_count: u64,
    /// The governed score (0.0..=1.0); below [`QUARANTINE_FLOOR`] -> `Quarantined`.
    pub outcome_score: f32,
    pub status: MemoryStatus,
    /// When this record was superseded, the id of the record that replaced it.
    /// Defaulted `None` so pre-existing records deserialize.
    #[serde(default)]
    pub superseded_by: Option<String>,
    /// When this record replaced another, the id of the one it superseded.
    #[serde(default)]
    pub supersedes: Option<String>,
}

impl MemoryRecord {
    /// Build a fresh `Active` record from a [`MemoryDraft`], minting a stable id
    /// (blake3 over the scope key + claim + a wall-clock micros seed). `created_ms`
    /// and `last_validated_ms` start equal; the record starts at
    /// [`INITIAL_OUTCOME_SCORE`] with `use_count` 0.
    pub fn from_draft(draft: MemoryDraft) -> Self {
        let now = now_ms();
        let memory_id = mint_memory_id(&draft.scope, &draft.claim);
        Self {
            memory_id,
            scope: draft.scope,
            claim: draft.claim,
            source: draft.source,
            author: draft.author,
            created_ms: now,
            last_validated_ms: now,
            confidence: draft.confidence.clamp(0.0, 1.0),
            citations: draft.citations,
            invalidation: draft.invalidation,
            privacy: draft.privacy,
            expiry_ms: draft.expiry_ms,
            use_count: 0,
            outcome_score: INITIAL_OUTCOME_SCORE,
            status: MemoryStatus::Active,
            superseded_by: None,
            supersedes: None,
        }
    }

    /// Record an OUTCOME of exercising this claim (bible sec 21-22 governance).
    /// Every outcome raises `use_count`. A success raises `outcome_score` (clamped
    /// to 1.0); a failure lowers it (clamped to 0.0) and, if it falls below
    /// [`QUARANTINE_FLOOR`] while still `Active`, QUARANTINES the record. A success
    /// never silently reactivates a quarantined record (reactivation is a separate,
    /// deliberate act).
    pub fn record_outcome(&mut self, success: bool) {
        self.use_count = self.use_count.saturating_add(1);
        if success {
            self.outcome_score = (self.outcome_score + SUCCESS_DELTA).min(1.0);
        } else {
            self.outcome_score = (self.outcome_score - FAILURE_DELTA).max(0.0);
            if self.outcome_score < QUARANTINE_FLOOR && self.status == MemoryStatus::Active {
                self.status = MemoryStatus::Quarantined;
            }
        }
    }

    /// True once `expiry_ms` is set and has passed relative to `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expiry_ms.map(|e| now_ms >= e).unwrap_or(false)
    }

    /// Eligible to ENTER CONTEXT: `Active` and not expired. Only such records feed
    /// the context compiler.
    pub fn is_eligible(&self, now_ms: u64) -> bool {
        self.status == MemoryStatus::Active && !self.is_expired(now_ms)
    }
}

/// The inputs to mint a new [`MemoryRecord`] (the id, timestamps, `use_count`,
/// `outcome_score`, and `status` are all derived, not supplied). `new` sets safe
/// defaults (neutral confidence, no citations, `Private`, no expiry) that the
/// builder-style setters refine.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryDraft {
    pub scope: MemoryScope,
    pub claim: String,
    pub source: String,
    pub author: String,
    pub confidence: f32,
    pub citations: Vec<String>,
    pub invalidation: Vec<String>,
    pub privacy: PrivacyClass,
    pub expiry_ms: Option<u64>,
}

impl MemoryDraft {
    /// A draft with the required provenance (scope + claim + source + author) and
    /// safe defaults for the rest.
    pub fn new(
        scope: MemoryScope,
        claim: impl Into<String>,
        source: impl Into<String>,
        author: impl Into<String>,
    ) -> Self {
        Self {
            scope,
            claim: claim.into(),
            source: source.into(),
            author: author.into(),
            confidence: 0.5,
            citations: Vec::new(),
            invalidation: Vec::new(),
            privacy: PrivacyClass::Private,
            expiry_ms: None,
        }
    }

    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }

    pub fn with_citations(mut self, citations: Vec<String>) -> Self {
        self.citations = citations;
        self
    }

    pub fn with_invalidation(mut self, invalidation: Vec<String>) -> Self {
        self.invalidation = invalidation;
        self
    }

    pub fn with_privacy(mut self, privacy: PrivacyClass) -> Self {
        self.privacy = privacy;
        self
    }

    pub fn with_expiry_ms(mut self, expiry_ms: Option<u64>) -> Self {
        self.expiry_ms = expiry_ms;
        self
    }
}

/// The target of a revalidation pass: a single record by id, or every record in a
/// scope (bible sec 21-22). Matches the `record_or_scope` spec of
/// `BackendHost::memory_revalidate`.
#[derive(Debug, Clone, PartialEq)]
pub enum RevalidateTarget {
    Record(String),
    Scope(MemoryScope),
}

impl RevalidateTarget {
    pub fn record(memory_id: impl Into<String>) -> Self {
        Self::Record(memory_id.into())
    }

    pub fn scope(scope: MemoryScope) -> Self {
        Self::Scope(scope)
    }
}

/// The per-record verdict of a revalidation pass (bible sec 21-22): the status
/// after revalidation, whether every citation still resolved on disk, the
/// citations that did NOT, and a human-readable reason.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryRevalidation {
    pub memory_id: String,
    pub status: MemoryStatus,
    /// True iff every citation resolved against the repo on disk.
    pub resolved: bool,
    /// The citations that no longer resolve (empty when `resolved`).
    pub unresolved: Vec<String>,
    pub reason: String,
    /// Set when this pass changed the record's status (e.g. `Active` ->
    /// `Quarantined`), so a caller can react to the transition.
    pub quarantined: bool,
}

/// The result of resolving ONE citation against the repo on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct CitationResolution {
    pub citation: String,
    pub resolved: bool,
    pub detail: String,
}

/// Resolve a single citation against the CURRENT repo rooted at `repo_root`
/// (bible sec 21-22). A citation is either a repo-relative `path` (the file must
/// exist) or a `path#symbol` ref (the file must exist AND contain `symbol` via a
/// lexical scan). Deterministic and model-free: SEMANTIC agreement between the
/// claim and the code is `DEFERRED_MODEL_REQUIRED` and never checked here.
pub fn resolve_citation(repo_root: &Path, citation: &str) -> CitationResolution {
    let (rel_path, symbol) = match citation.split_once('#') {
        Some((path, sym)) => (path, Some(sym)),
        None => (citation, None),
    };
    let full: PathBuf = repo_root.join(rel_path);
    if !full.exists() {
        return CitationResolution {
            citation: citation.to_string(),
            resolved: false,
            detail: format!("path '{rel_path}' does not exist"),
        };
    }
    match symbol {
        None => CitationResolution {
            citation: citation.to_string(),
            resolved: true,
            detail: "path exists".to_string(),
        },
        Some(sym) => {
            // A `path#symbol` ref must point at a readable FILE that contains the
            // symbol (lexical scan). A directory, or an unreadable file, fails.
            let content = std::fs::read_to_string(&full).unwrap_or_default();
            if contains_symbol(&content, sym) {
                CitationResolution {
                    citation: citation.to_string(),
                    resolved: true,
                    detail: format!("symbol '{sym}' found"),
                }
            } else {
                CitationResolution {
                    citation: citation.to_string(),
                    resolved: false,
                    detail: format!("symbol '{sym}' not found in '{rel_path}'"),
                }
            }
        }
    }
}

/// Lexical scan for a symbol in file content. A simple identifier (all
/// alphanumeric or `_`) is matched as a whole TOKEN (so `foo` does not match
/// `foobar`); a symbol carrying other characters (e.g. `Foo::bar`) falls back to a
/// substring search. Model-free by construction.
fn contains_symbol(content: &str, symbol: &str) -> bool {
    if symbol.is_empty() {
        return true;
    }
    let is_identifier = symbol.chars().all(|c| c.is_alphanumeric() || c == '_');
    if is_identifier {
        content
            .split(|c: char| !(c.is_alphanumeric() || c == '_'))
            .any(|token| token == symbol)
    } else {
        content.contains(symbol)
    }
}

/// Mint a stable, unique memory id (blake3 over the scope key + claim + a
/// wall-clock micros seed, first 24 hex chars). Prefixed `mem_`.
fn mint_memory_id(scope: &MemoryScope, claim: &str) -> String {
    let material = format!("{}|{}|{}", scope.key(), claim, now_micros());
    let hex = blake3::hash(material.as_bytes()).to_hex();
    format!("mem_{}", &hex.as_str()[..24])
}

/// Durable persistence for [`MemoryRecord`]s over the KV store (bible sec 21-22).
/// A stateless facade over the `memory` namespace keyed by `memory_id`, mirroring
/// how [`crate::services::GoalStore`] / [`crate::services::CheckpointStore`] wrap
/// their namespaces.
pub struct MemoryLedger;

impl MemoryLedger {
    pub const NAMESPACE: &'static str = "memory";

    /// Durably write (or replace) a record, keyed by `memory_id`.
    pub fn put(kv: &DynKeyValueStore, record: &MemoryRecord) -> Result<()> {
        let value = serde_json::to_value(record)?;
        kv.put(Self::NAMESPACE, &record.memory_id, value)
    }

    /// Look up a record by id, if any.
    pub fn get(kv: &DynKeyValueStore, memory_id: &str) -> Option<MemoryRecord> {
        kv.get(Self::NAMESPACE, memory_id)
            .ok()
            .flatten()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    /// Every record, ordered deterministically (created_ms then memory_id) so the
    /// list is stable across runs and reopens.
    pub fn list_all(kv: &DynKeyValueStore) -> Vec<MemoryRecord> {
        let mut out: Vec<MemoryRecord> = kv
            .list(Self::NAMESPACE)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(_, value)| serde_json::from_value::<MemoryRecord>(value).ok())
            .collect();
        out.sort_by(|a, b| {
            a.created_ms
                .cmp(&b.created_ms)
                .then_with(|| a.memory_id.cmp(&b.memory_id))
        });
        out
    }

    /// Every record BOUND to `scope`, ordered deterministically. Scope equality is
    /// by value (kind + id), so a `Repo("x")` list never returns a `Session("x")`
    /// record.
    pub fn list_scope(kv: &DynKeyValueStore, scope: &MemoryScope) -> Vec<MemoryRecord> {
        Self::list_all(kv)
            .into_iter()
            .filter(|record| &record.scope == scope)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contains_symbol_matches_whole_identifier_tokens_only() {
        assert!(contains_symbol("fn parse_line() {}", "parse_line"));
        // A substring of a larger identifier does NOT match (whole-token scan).
        assert!(!contains_symbol("fn parse_lines() {}", "parse_line"));
        // A non-identifier symbol falls back to substring.
        assert!(contains_symbol("impl Foo { fn bar() {} }", "Foo"));
        assert!(contains_symbol("let x = Foo::bar();", "Foo::bar"));
        assert!(!contains_symbol("let x = 1;", "missing"));
    }

    #[test]
    fn outcome_governance_raises_on_success_and_quarantines_below_floor() {
        let mut record = MemoryRecord::from_draft(MemoryDraft::new(
            MemoryScope::Repo("r".to_string()),
            "claim",
            "src",
            "author",
        ));
        assert_eq!(record.outcome_score, INITIAL_OUTCOME_SCORE);
        assert_eq!(record.use_count, 0);

        // Successes raise the score and the use count, never past 1.0.
        record.record_outcome(true);
        record.record_outcome(true);
        assert!(record.outcome_score > INITIAL_OUTCOME_SCORE);
        assert_eq!(record.use_count, 2);
        assert_eq!(record.status, MemoryStatus::Active);

        // Repeated failures drive the score below the floor -> Quarantined.
        for _ in 0..6 {
            record.record_outcome(false);
        }
        assert!(record.outcome_score < QUARANTINE_FLOOR);
        assert_eq!(record.status, MemoryStatus::Quarantined);
        assert_eq!(record.use_count, 8);
    }

    #[test]
    fn resolve_citation_checks_path_and_symbol_against_disk() {
        let dir = std::env::temp_dir().join(format!("hide_mem_cite_{}", now_ms()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.rs"), "pub fn alpha() {}\n").unwrap();

        // Bare path that exists resolves.
        assert!(resolve_citation(&dir, "a.rs").resolved);
        // Missing path does not.
        assert!(!resolve_citation(&dir, "b.rs").resolved);
        // path#symbol that exists resolves; a missing symbol does not.
        assert!(resolve_citation(&dir, "a.rs#alpha").resolved);
        assert!(!resolve_citation(&dir, "a.rs#omega").resolved);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn scope_listing_is_by_value_and_expiry_gates_eligibility() {
        let now = now_ms();
        let mut record = MemoryRecord::from_draft(MemoryDraft::new(
            MemoryScope::Session("s".to_string()),
            "claim",
            "src",
            "author",
        ));
        assert!(record.is_eligible(now));
        // A past expiry makes it ineligible even while Active.
        record.expiry_ms = Some(now.saturating_sub(1));
        assert!(!record.is_eligible(now));
    }
}
