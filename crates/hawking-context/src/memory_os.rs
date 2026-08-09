//! Memory OS facade (Ascension Bible §17).
//!
//! **Honest scope:** the durable substrate already lives in three places:
//!
//! | Layer | Crate / module | Role today |
//! |-------|----------------|------------|
//! | Six-class store | [`crate::memory_classes`] | Working / episodic / semantic_project / procedural / user / verification with write caps, pin/expire/forget |
//! | Hierarchical store | [`crate::memory`] | FTS5 + cosine retrieval, supersedes, pin, decay |
//! | Outcome ledger | `hide_backend::memory` | source, confidence, expiry, quarantine, revalidation, supersedes |
//! | File memory tool | `hide_kernel::tooling_memory` | Claude-parity path-rooted scratchpad |
//!
//! Bible §17 adds an **L0–L5 tier model** and a **unified tool surface**
//! (`memory.store|retrieve|update|consolidate|invalidate|archive|forget|explain`)
//! with a **full item schema** (source, timestamp, confidence, scope, expiry,
//! supersedes, contradicts, verification state). Stale claims must not silently
//! become permanent truth.
//!
//! This module is the **gap-fill scaffold**: tier types, the full item schema,
//! mapping onto existing classes, and an in-memory OS that enforces the stale-
//! claim rule. It does **not** replace `ClassedMemorySystem` or the host ledger.

use crate::memory_classes::{MemoryClass, PersonalScope};
use hide_core::ids::now_ms;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

// ---------------------------------------------------------------------------
// L0–L5 tiers (bible §17)
// ---------------------------------------------------------------------------

/// Memory OS retention tiers from the Ascension Bible.
///
/// These are **not** a second class store. They are a retention/intent lens over
/// existing substrate (see [`MemoryTier::from_class`] and the plan doc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// L0 ACTIVE — current task state and immediate tool outputs.
    L0Active,
    /// L1 SESSION — hypotheses, plans, checkpoints, unresolved failures.
    L1Session,
    /// L2 PROJECT — repository map, architecture decisions, accepted receipts.
    L2Project,
    /// L3 SKILLS — verified reusable procedures.
    L3Skills,
    /// L4 GRAVEYARD — failed mechanisms, causes, reopen conditions.
    L4Graveyard,
    /// L5 ARCHIVE — compressed historical evidence and source-bound artifacts.
    L5Archive,
}

impl MemoryTier {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::L0Active => "l0_active",
            Self::L1Session => "l1_session",
            Self::L2Project => "l2_project",
            Self::L3Skills => "l3_skills",
            Self::L4Graveyard => "l4_graveyard",
            Self::L5Archive => "l5_archive",
        }
    }

    pub fn all() -> [MemoryTier; 6] {
        [
            Self::L0Active,
            Self::L1Session,
            Self::L2Project,
            Self::L3Skills,
            Self::L4Graveyard,
            Self::L5Archive,
        ]
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "l0_active" | "L0" | "active" => Some(Self::L0Active),
            "l1_session" | "L1" | "session" => Some(Self::L1Session),
            "l2_project" | "L2" | "project" => Some(Self::L2Project),
            "l3_skills" | "L3" | "skills" => Some(Self::L3Skills),
            "l4_graveyard" | "L4" | "graveyard" => Some(Self::L4Graveyard),
            "l5_archive" | "L5" | "archive" => Some(Self::L5Archive),
            _ => None,
        }
    }

    /// Human retention rule for the tier.
    pub fn retention_rule(self) -> &'static str {
        match self {
            Self::L0Active => "turn/task local; dies with task end unless promoted",
            Self::L1Session => "session-bound; evicted with session unless promoted",
            Self::L2Project => "workspace-durable; supersede/retire explicit",
            Self::L3Skills => "verified procedures only; admitted via Skill Foundry",
            Self::L4Graveyard => {
                "failed mechanisms retained with reopen conditions; never auto-promoted to L2/L3"
            }
            Self::L5Archive => "compressed history; retrieval opt-in; not default context",
        }
    }

    /// Whether items in this tier may enter the default compile context.
    pub fn eligible_for_default_context(self) -> bool {
        matches!(
            self,
            Self::L0Active | Self::L1Session | Self::L2Project | Self::L3Skills
        )
    }

    /// Map an existing six-class record onto a default tier.
    ///
    /// L4/L5 are **not** expressible as classes today — they require explicit
    /// placement via [`MemoryOs::invalidate`] / [`MemoryOs::archive`].
    pub fn from_class(class: MemoryClass) -> Self {
        match class {
            MemoryClass::Working => Self::L0Active,
            MemoryClass::Episodic => Self::L1Session,
            MemoryClass::SemanticProject | MemoryClass::User | MemoryClass::Verification => {
                Self::L2Project
            }
            MemoryClass::Procedural => Self::L3Skills,
        }
    }

    /// Best-effort reverse map for bridging into classed writes.
    ///
    /// L4 Graveyard and L5 Archive have no dedicated class; bridge callers
    /// should keep the item in the OS facade or use verification + tags.
    pub fn to_class_hint(self) -> Option<MemoryClass> {
        match self {
            Self::L0Active => Some(MemoryClass::Working),
            Self::L1Session => Some(MemoryClass::Episodic),
            Self::L2Project => Some(MemoryClass::SemanticProject),
            Self::L3Skills => Some(MemoryClass::Procedural),
            Self::L4Graveyard | Self::L5Archive => None,
        }
    }
}

impl std::fmt::Display for MemoryTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Verification state (stale-claim protection)
// ---------------------------------------------------------------------------

/// Lifecycle / trust state of a memory item. Stale or contradicted claims cannot
/// be treated as permanent truth for default retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    /// Fresh write; no verification pass yet.
    Unverified,
    /// Asserted by a writer (model or human) without independent proof.
    Asserted,
    /// Exercised / tested with at least one outcome.
    Tested,
    /// Verifier-proven (maps to evidence_tier "proven" / VerifierWriteCap path).
    Proven,
    /// Marked as contradicting another active claim (both stay inspectable).
    Contradicted,
    /// Explicitly invalidated; not eligible for default context.
    Invalidated,
    /// Past hard expiry; soft-left the working set.
    Expired,
    /// Moved to L5 archive; opt-in retrieval only.
    Archived,
    /// Parked in L4 graveyard with reopen conditions.
    Graveyarded,
}

impl VerificationState {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Unverified => "unverified",
            Self::Asserted => "asserted",
            Self::Tested => "tested",
            Self::Proven => "proven",
            Self::Contradicted => "contradicted",
            Self::Invalidated => "invalidated",
            Self::Expired => "expired",
            Self::Archived => "archived",
            Self::Graveyarded => "graveyarded",
        }
    }

    /// Eligible for default context compile (bible: stale claims cannot silently
    /// become permanent truth).
    pub fn eligible_for_default_context(self) -> bool {
        matches!(
            self,
            Self::Unverified | Self::Asserted | Self::Tested | Self::Proven
        )
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "unverified" => Some(Self::Unverified),
            "asserted" => Some(Self::Asserted),
            "tested" => Some(Self::Tested),
            "proven" => Some(Self::Proven),
            "contradicted" => Some(Self::Contradicted),
            "invalidated" => Some(Self::Invalidated),
            "expired" => Some(Self::Expired),
            "archived" => Some(Self::Archived),
            "graveyarded" => Some(Self::Graveyarded),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Full item schema (bible §17 + Claude auto-memory prior art)
// ---------------------------------------------------------------------------

/// One Memory OS item. Field set is the bible contract; Claude's auto-memory
/// frontmatter (`name`, `description`, `type`, `originSessionId`) is prior art
/// for source/timestamp/scope metadata, not a second store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryItem {
    pub id: String,
    pub tier: MemoryTier,
    /// Claim / body text.
    pub text: String,
    /// Where it came from (tool run, doc, turn, human).
    pub source: String,
    /// Wall-clock create time (ms).
    pub timestamp_ms: u64,
    /// Trust in the claim, 0.0..=1.0.
    pub confidence: f32,
    /// Personal / organisational scope (reuses classed-memory scopes).
    pub scope: PersonalScope,
    /// Hard expiry; `None` means no TTL. Expired items cannot stay "Active truth".
    pub expiry_ms: Option<u64>,
    /// Id of the item this supersedes (version chain).
    pub supersedes: Option<String>,
    /// Ids of claims this item contradicts (surface for resolution; both remain).
    pub contradicts: Vec<String>,
    pub verification_state: VerificationState,
    /// Optional tags (graveyard reopen conditions, skill name, etc.).
    pub tags: Vec<String>,
    /// When true, exempt from soft expiry (never from forget).
    pub pinned: bool,
    /// Last update / touch time.
    pub updated_at_ms: u64,
}

impl MemoryItem {
    /// True when past hard expiry relative to `now_ms`.
    pub fn is_expired(&self, now_ms: u64) -> bool {
        self.expiry_ms.map(|e| now_ms >= e).unwrap_or(false)
    }

    /// Default-context eligibility: tier + verification + expiry + not pinned-out.
    ///
    /// Pinned items still respect invalidation/graveyard/archive — pin only
    /// defers soft expiry, it does not resurrect dead claims as truth.
    pub fn is_eligible_default(&self, now_ms: u64) -> bool {
        if self.is_expired(now_ms) {
            return false;
        }
        if !self.tier.eligible_for_default_context() {
            return false;
        }
        self.verification_state.eligible_for_default_context()
    }

    /// Apply soft expiry transition when due (does not delete).
    pub fn apply_expiry_if_due(&mut self, now_ms: u64) -> bool {
        if self.pinned {
            return false;
        }
        if self.is_expired(now_ms)
            && self.verification_state != VerificationState::Expired
            && self.verification_state != VerificationState::Archived
            && self.verification_state != VerificationState::Graveyarded
            && self.verification_state != VerificationState::Invalidated
        {
            self.verification_state = VerificationState::Expired;
            self.updated_at_ms = now_ms;
            return true;
        }
        false
    }
}

/// Draft for [`MemoryOs::store`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryItemDraft {
    pub text: String,
    pub source: String,
    pub tier: MemoryTier,
    pub confidence: f32,
    pub scope: PersonalScope,
    pub expiry_ms: Option<u64>,
    pub supersedes: Option<String>,
    pub contradicts: Vec<String>,
    pub verification_state: VerificationState,
    pub tags: Vec<String>,
    pub pinned: bool,
}

impl MemoryItemDraft {
    pub fn new(text: impl Into<String>, source: impl Into<String>, tier: MemoryTier) -> Self {
        Self {
            text: text.into(),
            source: source.into(),
            tier,
            confidence: 0.5,
            scope: PersonalScope::Workspace,
            expiry_ms: None,
            supersedes: None,
            contradicts: Vec::new(),
            verification_state: VerificationState::Unverified,
            tags: Vec::new(),
            pinned: false,
        }
    }

    pub fn with_confidence(mut self, c: f32) -> Self {
        self.confidence = c.clamp(0.0, 1.0);
        self
    }

    pub fn with_scope(mut self, scope: PersonalScope) -> Self {
        self.scope = scope;
        self
    }

    pub fn with_expiry_ms(mut self, at: u64) -> Self {
        self.expiry_ms = Some(at);
        self
    }

    pub fn with_supersedes(mut self, id: impl Into<String>) -> Self {
        self.supersedes = Some(id.into());
        self
    }

    pub fn with_contradicts(mut self, ids: Vec<String>) -> Self {
        self.contradicts = ids;
        self
    }

    pub fn with_verification(mut self, state: VerificationState) -> Self {
        self.verification_state = state;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }
}

/// Query for [`MemoryOs::retrieve`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryOsQuery {
    pub text: Option<String>,
    pub tiers: Vec<MemoryTier>,
    pub include_ineligible: bool,
    pub top_k: usize,
}

impl MemoryOsQuery {
    pub fn new(top_k: usize) -> Self {
        Self {
            text: None,
            tiers: Vec::new(),
            include_ineligible: false,
            top_k,
        }
    }

    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    pub fn in_tiers(mut self, tiers: Vec<MemoryTier>) -> Self {
        self.tiers = tiers;
        self
    }

    pub fn include_ineligible(mut self) -> Self {
        self.include_ineligible = true;
        self
    }
}

/// Patch for [`MemoryOs::update`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryItemPatch {
    pub text: Option<String>,
    pub confidence: Option<f32>,
    pub expiry_ms: Option<Option<u64>>,
    pub tags: Option<Vec<String>>,
    pub pinned: Option<bool>,
    pub verification_state: Option<VerificationState>,
    pub contradicts: Option<Vec<String>>,
}

/// Explanation of why an item is or is not eligible (bible `memory.explain`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryExplain {
    pub id: String,
    pub tier: MemoryTier,
    pub verification_state: VerificationState,
    pub eligible_default: bool,
    pub reasons: Vec<String>,
    pub supersedes: Option<String>,
    pub contradicts: Vec<String>,
    pub confidence: f32,
    pub expiry_ms: Option<u64>,
    pub source: String,
    pub timestamp_ms: u64,
}

/// Result of consolidate: merge N items into one, superseding the rest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConsolidateResult {
    pub kept: MemoryItem,
    pub superseded_ids: Vec<String>,
}

/// Memory OS tool surface (bible §17).
pub trait MemoryOs: Send + Sync {
    fn store(&self, draft: MemoryItemDraft) -> Result<MemoryItem, MemoryOsError>;
    fn retrieve(&self, query: MemoryOsQuery) -> Result<Vec<MemoryItem>, MemoryOsError>;
    fn update(&self, id: &str, patch: MemoryItemPatch) -> Result<MemoryItem, MemoryOsError>;
    fn consolidate(
        &self,
        ids: &[String],
        into_text: &str,
    ) -> Result<ConsolidateResult, MemoryOsError>;
    fn invalidate(&self, id: &str, reason: &str) -> Result<MemoryItem, MemoryOsError>;
    fn archive(&self, id: &str) -> Result<MemoryItem, MemoryOsError>;
    fn forget(&self, id: &str) -> Result<bool, MemoryOsError>;
    fn explain(&self, id: &str) -> Result<MemoryExplain, MemoryOsError>;
    fn get(&self, id: &str) -> Result<Option<MemoryItem>, MemoryOsError>;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MemoryOsError {
    #[error("memory item not found: {0}")]
    NotFound(String),
    #[error("invalid memory operation: {0}")]
    Invalid(String),
}

// ---------------------------------------------------------------------------
// In-memory scaffold (tests + pre-bridge)
// ---------------------------------------------------------------------------

/// Scaffold Memory OS with full schema enforcement and stale-claim protection.
///
/// Not the production durable store — bridges to classed/ledger stores are
/// planned (see `HCLI_MEMORY_OS_PLAN.md`). This proves the tool contract and
/// the "stale claims cannot become permanent truth" rule.
#[derive(Debug, Default)]
pub struct InMemoryMemoryOs {
    items: RwLock<BTreeMap<String, MemoryItem>>,
    next_id: AtomicU64,
    /// Injectable clock for expiry tests (ms). When 0, uses [`now_ms`].
    clock_ms: AtomicU64,
}

impl InMemoryMemoryOs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Force the clock used for store/expiry (tests).
    pub fn set_clock_ms(&self, ms: u64) {
        self.clock_ms.store(ms, Ordering::Relaxed);
    }

    fn now(&self) -> u64 {
        let c = self.clock_ms.load(Ordering::Relaxed);
        if c == 0 {
            now_ms()
        } else {
            c
        }
    }

    fn mint_id(&self) -> String {
        let n = self.next_id.fetch_add(1, Ordering::Relaxed);
        format!("mos-{n}")
    }

    fn touch_expiries(&self) {
        let now = self.now();
        let mut map = self.items.write();
        for item in map.values_mut() {
            item.apply_expiry_if_due(now);
        }
    }

    /// Park a failed mechanism in L4 with reopen conditions (tags).
    pub fn graveyard(
        &self,
        id: &str,
        reopen_conditions: Vec<String>,
    ) -> Result<MemoryItem, MemoryOsError> {
        let now = self.now();
        let mut map = self.items.write();
        let item = map
            .get_mut(id)
            .ok_or_else(|| MemoryOsError::NotFound(id.into()))?;
        item.tier = MemoryTier::L4Graveyard;
        item.verification_state = VerificationState::Graveyarded;
        item.tags = reopen_conditions;
        item.updated_at_ms = now;
        Ok(item.clone())
    }
}

impl MemoryOs for InMemoryMemoryOs {
    fn store(&self, draft: MemoryItemDraft) -> Result<MemoryItem, MemoryOsError> {
        // Proven claims cannot be written into L4/L5 as if they were active truth.
        if matches!(draft.tier, MemoryTier::L4Graveyard | MemoryTier::L5Archive)
            && draft.verification_state.eligible_for_default_context()
        {
            return Err(MemoryOsError::Invalid(
                "L4/L5 writes must not claim default-context verification states; use invalidate/archive/graveyard".into(),
            ));
        }
        let now = self.now();
        let id = self.mint_id();
        let mut item = MemoryItem {
            id: id.clone(),
            tier: draft.tier,
            text: draft.text,
            source: draft.source,
            timestamp_ms: now,
            confidence: draft.confidence.clamp(0.0, 1.0),
            scope: draft.scope,
            expiry_ms: draft.expiry_ms,
            supersedes: draft.supersedes,
            contradicts: draft.contradicts.clone(),
            verification_state: draft.verification_state,
            tags: draft.tags,
            pinned: draft.pinned,
            updated_at_ms: now,
        };
        item.apply_expiry_if_due(now);

        // Symmetric contradict edges: if A contradicts B, mark both Contradicted
        // when either was still "truth-eligible" — never silently keep both as Proven.
        let mut map = self.items.write();
        if let Some(old_id) = &item.supersedes {
            if let Some(old) = map.get_mut(old_id) {
                // Supersession does not delete; old leaves default eligibility.
                if old.verification_state.eligible_for_default_context() {
                    old.verification_state = VerificationState::Invalidated;
                    old.updated_at_ms = now;
                }
            }
        }
        for other_id in &draft.contradicts {
            if let Some(other) = map.get_mut(other_id) {
                if other.verification_state.eligible_for_default_context()
                    || other.verification_state == VerificationState::Contradicted
                {
                    other.verification_state = VerificationState::Contradicted;
                    if !other.contradicts.iter().any(|c| c == &id) {
                        other.contradicts.push(id.clone());
                    }
                    other.updated_at_ms = now;
                }
            }
            if item.verification_state.eligible_for_default_context() {
                item.verification_state = VerificationState::Contradicted;
            }
        }
        map.insert(id, item.clone());
        Ok(item)
    }

    fn retrieve(&self, query: MemoryOsQuery) -> Result<Vec<MemoryItem>, MemoryOsError> {
        self.touch_expiries();
        let now = self.now();
        let map = self.items.read();
        let mut hits: Vec<MemoryItem> = map
            .values()
            .filter(|item| {
                if !query.tiers.is_empty() && !query.tiers.contains(&item.tier) {
                    return false;
                }
                if !query.include_ineligible && !item.is_eligible_default(now) {
                    return false;
                }
                if let Some(ref q) = query.text {
                    let q = q.to_lowercase();
                    if !item.text.to_lowercase().contains(&q)
                        && !item.tags.iter().any(|t| t.to_lowercase().contains(&q))
                        && !item.source.to_lowercase().contains(&q)
                    {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect();
        // Prefer higher confidence, then fresher.
        hits.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.updated_at_ms.cmp(&a.updated_at_ms))
        });
        let k = if query.top_k == 0 { 10 } else { query.top_k };
        hits.truncate(k);
        Ok(hits)
    }

    fn update(&self, id: &str, patch: MemoryItemPatch) -> Result<MemoryItem, MemoryOsError> {
        let now = self.now();
        let mut map = self.items.write();
        let item = map
            .get_mut(id)
            .ok_or_else(|| MemoryOsError::NotFound(id.into()))?;
        if let Some(text) = patch.text {
            item.text = text;
        }
        if let Some(c) = patch.confidence {
            item.confidence = c.clamp(0.0, 1.0);
        }
        if let Some(exp) = patch.expiry_ms {
            item.expiry_ms = exp;
        }
        if let Some(tags) = patch.tags {
            item.tags = tags;
        }
        if let Some(p) = patch.pinned {
            item.pinned = p;
        }
        if let Some(vs) = patch.verification_state {
            // Guard: cannot silently promote Expired/Invalidated/Graveyarded/Archived
            // to Proven without going through an explicit re-admit path later.
            if matches!(
                item.verification_state,
                VerificationState::Invalidated
                    | VerificationState::Expired
                    | VerificationState::Archived
                    | VerificationState::Graveyarded
            ) && vs.eligible_for_default_context()
            {
                return Err(MemoryOsError::Invalid(
                    "cannot promote stale/invalidated/archived/graveyarded claims to truth via update; store a new superseding item".into(),
                ));
            }
            item.verification_state = vs;
        }
        if let Some(c) = patch.contradicts {
            item.contradicts = c;
        }
        item.updated_at_ms = now;
        item.apply_expiry_if_due(now);
        Ok(item.clone())
    }

    fn consolidate(
        &self,
        ids: &[String],
        into_text: &str,
    ) -> Result<ConsolidateResult, MemoryOsError> {
        if ids.is_empty() {
            return Err(MemoryOsError::Invalid(
                "consolidate requires at least one id".into(),
            ));
        }
        let now = self.now();
        // Snapshot sources before mutation.
        let snapshots: Vec<MemoryItem> = {
            let map = self.items.read();
            let mut out = Vec::new();
            for id in ids {
                let item = map
                    .get(id)
                    .ok_or_else(|| MemoryOsError::NotFound(id.clone()))?;
                out.push(item.clone());
            }
            out
        };
        let primary = &snapshots[0];
        let max_conf = snapshots
            .iter()
            .map(|s| s.confidence)
            .fold(0.0_f32, f32::max);
        let mut tags: Vec<String> = snapshots.iter().flat_map(|s| s.tags.clone()).collect();
        tags.sort();
        tags.dedup();
        tags.push(format!("consolidated_from:{}", ids.join(",")));

        let draft = MemoryItemDraft {
            text: into_text.into(),
            source: format!("consolidate:{}", primary.source),
            tier: primary.tier,
            confidence: max_conf,
            scope: primary.scope,
            expiry_ms: primary.expiry_ms,
            supersedes: Some(primary.id.clone()),
            contradicts: Vec::new(),
            verification_state: VerificationState::Asserted,
            tags,
            pinned: false,
        };
        // store() will invalidate the supersedes target; invalidate the rest.
        let kept = self.store(draft)?;
        let mut superseded_ids = vec![primary.id.clone()];
        {
            let mut map = self.items.write();
            for id in ids.iter().skip(1) {
                if let Some(old) = map.get_mut(id) {
                    old.verification_state = VerificationState::Invalidated;
                    old.updated_at_ms = now;
                    old.supersedes = None; // they are superseded-by kept, not chains
                    superseded_ids.push(id.clone());
                }
            }
            // Link kept.supersedes chain note via tags already; also mark primary.
            if let Some(old) = map.get_mut(&primary.id) {
                old.verification_state = VerificationState::Invalidated;
                old.updated_at_ms = now;
            }
        }
        Ok(ConsolidateResult {
            kept,
            superseded_ids,
        })
    }

    fn invalidate(&self, id: &str, reason: &str) -> Result<MemoryItem, MemoryOsError> {
        let now = self.now();
        let mut map = self.items.write();
        let item = map
            .get_mut(id)
            .ok_or_else(|| MemoryOsError::NotFound(id.into()))?;
        item.verification_state = VerificationState::Invalidated;
        item.tags.push(format!("invalidate:{reason}"));
        item.updated_at_ms = now;
        Ok(item.clone())
    }

    fn archive(&self, id: &str) -> Result<MemoryItem, MemoryOsError> {
        let now = self.now();
        let mut map = self.items.write();
        let item = map
            .get_mut(id)
            .ok_or_else(|| MemoryOsError::NotFound(id.into()))?;
        item.tier = MemoryTier::L5Archive;
        item.verification_state = VerificationState::Archived;
        item.updated_at_ms = now;
        Ok(item.clone())
    }

    fn forget(&self, id: &str) -> Result<bool, MemoryOsError> {
        let mut map = self.items.write();
        let removed = map.remove(id).is_some();
        // Clear dangling supersedes / contradict edges.
        for item in map.values_mut() {
            if item.supersedes.as_deref() == Some(id) {
                item.supersedes = None;
            }
            item.contradicts.retain(|c| c != id);
        }
        Ok(removed)
    }

    fn explain(&self, id: &str) -> Result<MemoryExplain, MemoryOsError> {
        self.touch_expiries();
        let now = self.now();
        let map = self.items.read();
        let item = map
            .get(id)
            .ok_or_else(|| MemoryOsError::NotFound(id.into()))?;
        let eligible = item.is_eligible_default(now);
        let mut reasons = Vec::new();
        reasons.push(format!(
            "tier={} ({})",
            item.tier.as_str(),
            item.tier.retention_rule()
        ));
        reasons.push(format!(
            "verification_state={}",
            item.verification_state.as_str()
        ));
        if item.is_expired(now) {
            reasons.push("hard expiry passed".into());
        }
        if !item.tier.eligible_for_default_context() {
            reasons.push("tier is not in default context set (L0–L3 only)".into());
        }
        if !item.verification_state.eligible_for_default_context() {
            reasons.push(
                "verification state blocks default context (stale/invalid/contradicted/archived)"
                    .into(),
            );
        }
        if eligible {
            reasons.push("eligible for default context".into());
        } else {
            reasons.push("NOT eligible for default context".into());
        }
        if item.pinned {
            reasons.push("pinned (exempt from soft expiry only)".into());
        }
        Ok(MemoryExplain {
            id: item.id.clone(),
            tier: item.tier,
            verification_state: item.verification_state,
            eligible_default: eligible,
            reasons,
            supersedes: item.supersedes.clone(),
            contradicts: item.contradicts.clone(),
            confidence: item.confidence,
            expiry_ms: item.expiry_ms,
            source: item.source.clone(),
            timestamp_ms: item.timestamp_ms,
        })
    }

    fn get(&self, id: &str) -> Result<Option<MemoryItem>, MemoryOsError> {
        self.touch_expiries();
        Ok(self.items.read().get(id).cloned())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_maps_from_existing_classes() {
        assert_eq!(
            MemoryTier::from_class(MemoryClass::Working),
            MemoryTier::L0Active
        );
        assert_eq!(
            MemoryTier::from_class(MemoryClass::Episodic),
            MemoryTier::L1Session
        );
        assert_eq!(
            MemoryTier::from_class(MemoryClass::SemanticProject),
            MemoryTier::L2Project
        );
        assert_eq!(
            MemoryTier::from_class(MemoryClass::Procedural),
            MemoryTier::L3Skills
        );
        assert_eq!(
            MemoryTier::from_class(MemoryClass::Verification),
            MemoryTier::L2Project
        );
        assert!(MemoryTier::L4Graveyard.to_class_hint().is_none());
        assert!(MemoryTier::L5Archive.to_class_hint().is_none());
    }

    #[test]
    fn store_retrieve_forget_roundtrip() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(1_000);
        let a = os
            .store(
                MemoryItemDraft::new(
                    "repo uses hawking-context for memory",
                    "audit",
                    MemoryTier::L2Project,
                )
                .with_confidence(0.9)
                .with_verification(VerificationState::Asserted),
            )
            .unwrap();
        let hits = os
            .retrieve(MemoryOsQuery::new(5).with_text("hawking-context"))
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, a.id);
        assert!(os.forget(&a.id).unwrap());
        assert!(os.get(&a.id).unwrap().is_none());
        assert!(os
            .retrieve(MemoryOsQuery::new(5).with_text("hawking-context"))
            .unwrap()
            .is_empty());
    }

    #[test]
    fn stale_expiry_cannot_enter_default_retrieve() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(100);
        let item = os
            .store(
                MemoryItemDraft::new("temporary hypothesis", "session", MemoryTier::L1Session)
                    .with_expiry_ms(200)
                    .with_verification(VerificationState::Asserted),
            )
            .unwrap();
        assert!(item.is_eligible_default(100));
        os.set_clock_ms(250);
        // Default retrieve must drop it.
        let hits = os.retrieve(MemoryOsQuery::new(10)).unwrap();
        assert!(hits.iter().all(|h| h.id != item.id));
        // Inspect path still sees it when include_ineligible.
        let all = os
            .retrieve(MemoryOsQuery::new(10).include_ineligible())
            .unwrap();
        let found = all.iter().find(|h| h.id == item.id).unwrap();
        assert_eq!(found.verification_state, VerificationState::Expired);
        assert!(!found.is_eligible_default(250));
    }

    #[test]
    fn cannot_promote_invalidated_to_proven_via_update() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(1);
        let item = os
            .store(
                MemoryItemDraft::new("wrong claim", "model", MemoryTier::L2Project)
                    .with_verification(VerificationState::Asserted),
            )
            .unwrap();
        os.invalidate(&item.id, "citation gone").unwrap();
        let err = os
            .update(
                &item.id,
                MemoryItemPatch {
                    verification_state: Some(VerificationState::Proven),
                    ..Default::default()
                },
            )
            .unwrap_err();
        assert!(matches!(err, MemoryOsError::Invalid(_)));
        // New superseding store is the legal path.
        let fixed = os
            .store(
                MemoryItemDraft::new("corrected claim", "human", MemoryTier::L2Project)
                    .with_supersedes(&item.id)
                    .with_verification(VerificationState::Proven)
                    .with_confidence(0.95),
            )
            .unwrap();
        assert_eq!(fixed.supersedes.as_deref(), Some(item.id.as_str()));
        let old = os.get(&item.id).unwrap().unwrap();
        assert_eq!(old.verification_state, VerificationState::Invalidated);
        let eligible = os.retrieve(MemoryOsQuery::new(10)).unwrap();
        assert!(eligible.iter().any(|h| h.id == fixed.id));
        assert!(!eligible.iter().any(|h| h.id == item.id));
    }

    #[test]
    fn contradict_marks_both_ineligible() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(1);
        let a = os
            .store(
                MemoryItemDraft::new("layout: A owns compiler", "distill", MemoryTier::L2Project)
                    .with_verification(VerificationState::Asserted),
            )
            .unwrap();
        let b = os
            .store(
                MemoryItemDraft::new("layout: B owns compiler", "distill", MemoryTier::L2Project)
                    .with_contradicts(vec![a.id.clone()])
                    .with_verification(VerificationState::Asserted),
            )
            .unwrap();
        let a2 = os.get(&a.id).unwrap().unwrap();
        let b2 = os.get(&b.id).unwrap().unwrap();
        assert_eq!(a2.verification_state, VerificationState::Contradicted);
        assert_eq!(b2.verification_state, VerificationState::Contradicted);
        assert!(a2.contradicts.contains(&b.id));
        let eligible = os.retrieve(MemoryOsQuery::new(10)).unwrap();
        assert!(!eligible.iter().any(|h| h.id == a.id || h.id == b.id));
    }

    #[test]
    fn archive_and_graveyard_leave_default_context() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(1);
        let a = os
            .store(
                MemoryItemDraft::new("old receipt", "run", MemoryTier::L2Project)
                    .with_verification(VerificationState::Tested),
            )
            .unwrap();
        let b = os
            .store(
                MemoryItemDraft::new("failed approach X", "session", MemoryTier::L1Session)
                    .with_verification(VerificationState::Tested),
            )
            .unwrap();
        os.archive(&a.id).unwrap();
        os.graveyard(&b.id, vec!["reopen_if:new_evidence".into()])
            .unwrap();
        let eligible = os.retrieve(MemoryOsQuery::new(10)).unwrap();
        assert!(eligible.is_empty());
        let a2 = os.get(&a.id).unwrap().unwrap();
        assert_eq!(a2.tier, MemoryTier::L5Archive);
        assert_eq!(a2.verification_state, VerificationState::Archived);
        let b2 = os.get(&b.id).unwrap().unwrap();
        assert_eq!(b2.tier, MemoryTier::L4Graveyard);
        assert_eq!(b2.verification_state, VerificationState::Graveyarded);
        assert!(b2.tags.iter().any(|t| t.starts_with("reopen_if:")));
    }

    #[test]
    fn consolidate_supersedes_inputs() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(1);
        let a = os
            .store(
                MemoryItemDraft::new("fact one", "s", MemoryTier::L2Project).with_confidence(0.4),
            )
            .unwrap();
        let b = os
            .store(
                MemoryItemDraft::new("fact two", "s", MemoryTier::L2Project).with_confidence(0.8),
            )
            .unwrap();
        let res = os
            .consolidate(&[a.id.clone(), b.id.clone()], "fact one+two merged")
            .unwrap();
        assert!((res.kept.confidence - 0.8).abs() < 1e-6);
        assert_eq!(res.superseded_ids.len(), 2);
        let eligible = os.retrieve(MemoryOsQuery::new(10)).unwrap();
        assert_eq!(eligible.len(), 1);
        assert_eq!(eligible[0].id, res.kept.id);
        assert!(eligible[0].text.contains("merged"));
    }

    #[test]
    fn explain_surfaces_full_schema() {
        let os = InMemoryMemoryOs::new();
        os.set_clock_ms(42);
        let item = os
            .store(
                MemoryItemDraft::new("claim", "tool:scan", MemoryTier::L2Project)
                    .with_confidence(0.7)
                    .with_verification(VerificationState::Asserted)
                    .with_expiry_ms(999),
            )
            .unwrap();
        let exp = os.explain(&item.id).unwrap();
        assert_eq!(exp.id, item.id);
        assert_eq!(exp.source, "tool:scan");
        assert_eq!(exp.timestamp_ms, 42);
        assert!((exp.confidence - 0.7).abs() < 1e-6);
        assert_eq!(exp.expiry_ms, Some(999));
        assert!(exp.eligible_default);
        assert!(!exp.reasons.is_empty());
    }

    #[test]
    fn all_tools_exist_on_trait_surface() {
        // Compile-time + runtime smoke that the bible tool names are wired.
        let os: &dyn MemoryOs = &InMemoryMemoryOs::new();
        let item = os
            .store(MemoryItemDraft::new("x", "y", MemoryTier::L0Active))
            .unwrap();
        let _ = os.retrieve(MemoryOsQuery::new(1)).unwrap();
        let _ = os
            .update(
                &item.id,
                MemoryItemPatch {
                    text: Some("x2".into()),
                    ..Default::default()
                },
            )
            .unwrap();
        let _ = os.invalidate(&item.id, "done").unwrap();
        // re-store for archive path
        let item2 = os
            .store(MemoryItemDraft::new("z", "y", MemoryTier::L1Session))
            .unwrap();
        let _ = os.archive(&item2.id).unwrap();
        let item3 = os
            .store(MemoryItemDraft::new("c", "y", MemoryTier::L2Project))
            .unwrap();
        let _ = os.consolidate(&[item3.id.clone()], "c'").unwrap();
        let _ = os.explain(&item.id).unwrap();
        let _ = os.forget(&item.id).unwrap();
    }
}
