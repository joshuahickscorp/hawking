//! Six distinct memory systems for the HIDE Context OS.
//!
//! The legacy [`crate::memory::MemoryStore`] keeps a single table with a `kind`
//! column (Working / Episodic / Semantic / Procedural / Project). That is five
//! **labels** on one blob store — not six systems. This module is the real
//! substrate the context compiler and `SubmitTurn` consult:
//!
//! | class              | store          | write authority              | lifetime              |
//! |--------------------|----------------|------------------------------|-----------------------|
//! | working            | RAM            | [`TurnWriteCap`]             | one turn              |
//! | episodic           | SQLite table   | [`EpisodicWriteCap`]         | session, replayable   |
//! | semantic_project   | SQLite table   | [`ProjectWriteCap`]          | durable / workspace   |
//! | procedural         | SQLite table   | [`ProceduralWriteCap`]       | durable / workspace   |
//! | user               | SQLite (user)  | [`UserWriteCap`]             | durable / cross-ws    |
//! | verification       | SQLite table   | [`VerifierWriteCap`]         | durable / verifier    |
//!
//! Write authority is a **type boundary**: the verification write path takes a
//! [`VerifierWriteCap`] that the turn-generation path never holds. Provenance
//! `authority` is stamped by the write method from the capability type, so the
//! turn path cannot forge a verifier or user write.

use crate::budget::estimate_tokens;
use hide_core::error::{HideError, Result};
use hide_core::ids::now_ms;
use parking_lot::Mutex;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Classes + write authority (type boundary)
// ---------------------------------------------------------------------------

/// The six HIDE memory classes. Wire names match the Context OS contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum MemoryClass {
    Working,
    Episodic,
    SemanticProject,
    Procedural,
    User,
    Verification,
}

impl MemoryClass {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::Episodic => "episodic",
            Self::SemanticProject => "semantic_project",
            Self::Procedural => "procedural",
            Self::User => "user",
            Self::Verification => "verification",
        }
    }

    pub fn all() -> [MemoryClass; 6] {
        [
            Self::Working,
            Self::Episodic,
            Self::SemanticProject,
            Self::Procedural,
            Self::User,
            Self::Verification,
        ]
    }

    /// The retrieval question the context compiler asks this class.
    pub fn retrieval_question(self) -> &'static str {
        match self {
            Self::Working => "what is the live scratch for this turn?",
            Self::Episodic => "what did we try this session (turns, tools, edits, verdicts)?",
            Self::SemanticProject => {
                "what durable facts about this repository are relevant (layout, conventions, invariants)?"
            }
            Self::Procedural => {
                "what recipes, build/test commands, skills, or hooks worked here?"
            }
            Self::User => "what standing preferences and instructions apply for this person?",
            Self::Verification => {
                "what claims are asserted vs proven, at which evidence tier, by which run?"
            }
        }
    }

    /// Human description of retention.
    pub fn retention_rule(self) -> &'static str {
        match self {
            Self::Working => "turn_local: cleared by end_turn; never persisted",
            Self::Episodic => "session: retained while session_id is live; evict_session drops it",
            Self::SemanticProject => "durable: survives session restart; workspace-scoped",
            Self::Procedural => "durable: survives session restart; workspace-scoped",
            Self::User => "durable: survives session restart; NOT workspace-scoped",
            Self::Verification => "durable: survives session restart; never model-overwritten",
        }
    }

    /// Human description of eviction.
    pub fn eviction_rule(self) -> &'static str {
        match self {
            Self::Working => "end_turn() drops all working records for that turn_id",
            Self::Episodic => "evict_session(session_id) retires that session's episodes",
            Self::SemanticProject => "supersede/retire explicit; no session TTL",
            Self::Procedural => "supersede on newer successful recipe; no session TTL",
            Self::User => "explicit user delete only; never workspace teardown",
            Self::Verification => "verifier supersede only; model path cannot evict",
        }
    }
}

/// Who is allowed to write a class — stamped onto every record by the write API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WriteAuthority {
    /// Kernel / executor for the live turn's scratch.
    Turn,
    /// Event stream (turns, tool invocations, edits, verdicts).
    EventStream,
    /// Explicit project write or distillation from episodic.
    ProjectDistill,
    /// Successful tool receipt path.
    ToolReceipt,
    /// Explicit user-scoped intent only (never distillation).
    UserExplicit,
    /// Verifier path only (never the model turn).
    Verifier,
}

// ---------------------------------------------------------------------------
// Personal scopes (orthogonal to the six classes)
// ---------------------------------------------------------------------------

/// Eight personal scopes. Orthogonal to [`MemoryClass`]: a record has exactly
/// one class and exactly one scope.
///
/// Connector-scoped content never becomes global without an explicit, recorded
/// promotion (see [`ClassedMemorySystem::set_scope`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PersonalScope {
    Global,
    Workspace,
    Project,
    Conversation,
    Connector,
    Person,
    PrivateVault,
    Ephemeral,
}

impl PersonalScope {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Workspace => "workspace",
            Self::Project => "project",
            Self::Conversation => "conversation",
            Self::Connector => "connector",
            Self::Person => "person",
            Self::PrivateVault => "private_vault",
            Self::Ephemeral => "ephemeral",
        }
    }

    pub fn all() -> [PersonalScope; 8] {
        [
            Self::Global,
            Self::Workspace,
            Self::Project,
            Self::Conversation,
            Self::Connector,
            Self::Person,
            Self::PrivateVault,
            Self::Ephemeral,
        ]
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "global" => Some(Self::Global),
            "workspace" => Some(Self::Workspace),
            "project" => Some(Self::Project),
            "conversation" => Some(Self::Conversation),
            "connector" => Some(Self::Connector),
            "person" => Some(Self::Person),
            "private_vault" => Some(Self::PrivateVault),
            "ephemeral" => Some(Self::Ephemeral),
            _ => None,
        }
    }

    /// Default scope for a class when the draft does not set one.
    pub fn default_for_class(class: MemoryClass) -> Self {
        match class {
            MemoryClass::Working => Self::Conversation,
            MemoryClass::Episodic => Self::Conversation,
            MemoryClass::SemanticProject => Self::Workspace,
            MemoryClass::Procedural => Self::Workspace,
            MemoryClass::User => Self::Global,
            MemoryClass::Verification => Self::Workspace,
        }
    }
}

impl std::fmt::Display for PersonalScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Filter for [`ClassedMemorySystem::inspect`].
#[derive(Debug, Clone, Default)]
pub struct InspectFilter {
    pub class: Option<MemoryClass>,
    pub scope: Option<PersonalScope>,
    /// When true, include expired records (default false for active inspect).
    pub include_expired: bool,
    /// When true, include working (turn-local) records.
    pub include_working: bool,
}

/// Portable export of everything the user owns in the six class stores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryExport {
    pub schema: String,
    pub exported_at_ms: u64,
    pub workspace_id: String,
    pub records: Vec<ClassMemoryRecord>,
    pub promotions: Vec<ScopePromotion>,
    pub disabled_classes: Vec<String>,
}

/// Recorded scope transition (connector → global never silent).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScopePromotion {
    pub record_id: String,
    pub from_scope: PersonalScope,
    pub to_scope: PersonalScope,
    pub at_ms: u64,
    pub approved_by: String,
}

/// Capability: kernel/executor writing working memory for one turn.
/// Construct only at turn start — not held by the verifier or user intent path.
#[derive(Debug, Clone)]
pub struct TurnWriteCap {
    pub turn_id: String,
}

impl TurnWriteCap {
    pub fn new(turn_id: impl Into<String>) -> Self {
        Self {
            turn_id: turn_id.into(),
        }
    }
}

/// Capability: append episodic records from the event stream.
#[derive(Debug, Clone, Copy)]
pub struct EpisodicWriteCap {
    _private: (),
}

impl EpisodicWriteCap {
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: semantic_project writes (explicit + distillation).
#[derive(Debug, Clone, Copy)]
pub struct ProjectWriteCap {
    _private: (),
}

impl ProjectWriteCap {
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: procedural writes from successful tool receipts.
#[derive(Debug, Clone, Copy)]
pub struct ProceduralWriteCap {
    _private: (),
}

impl ProceduralWriteCap {
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: explicit user-scoped preference writes.
/// The model turn / distillation path must not hold this.
#[derive(Debug, Clone, Copy)]
pub struct UserWriteCap {
    _private: (),
}

impl UserWriteCap {
    /// Mint only at the user-intent entry point.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

/// Capability: verification-memory writes.
/// The model turn path must not hold this — requiring this type at the write
/// site makes a turn→verification write obvious in any diff.
#[derive(Debug, Clone, Copy)]
pub struct VerifierWriteCap {
    _private: (),
}

impl VerifierWriteCap {
    /// Mint only at the verifier entry point.
    pub fn mint() -> Self {
        Self { _private: () }
    }
}

// ---------------------------------------------------------------------------
// Records + provenance
// ---------------------------------------------------------------------------

/// Provenance stamped on every classed memory record.
///
/// `authority` is set by the write method from the capability type — callers
/// cannot pass an arbitrary authority for protected classes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassProvenance {
    /// Who wrote it (subsystem / path name).
    pub writer: String,
    pub written_at_ms: u64,
    pub turn_id: Option<String>,
    pub run_id: Option<String>,
    /// Evidence supporting the claim (semantic_project / procedural / verification).
    pub evidence: Vec<String>,
    /// Authority class — stamped by the API, not caller-chosen for protected writes.
    pub authority: WriteAuthority,
}

impl ClassProvenance {
    fn stamped(
        writer: impl Into<String>,
        authority: WriteAuthority,
        turn_id: Option<String>,
        run_id: Option<String>,
        evidence: Vec<String>,
    ) -> Self {
        Self {
            writer: writer.into(),
            written_at_ms: now_ms(),
            turn_id,
            run_id,
            evidence,
            authority,
        }
    }
}

/// A record in one of the six class stores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassMemoryRecord {
    pub id: String,
    pub class: MemoryClass,
    /// Exactly one personal scope, orthogonal to class.
    pub scope: PersonalScope,
    pub text: String,
    pub importance: f32,
    /// Workspace id for workspace-scoped classes; always `None` for `user`.
    pub workspace_id: Option<String>,
    /// Session id for episodic (and optional context for others).
    pub session_id: Option<String>,
    pub provenance: ClassProvenance,
    /// Evidence tier for verification (asserted / tested / proven); unused elsewhere.
    pub evidence_tier: Option<String>,
    /// Pinned records are exempt from expiry, never from forget.
    pub pinned: bool,
    /// Soft-expired (left the working set); still reachable by inspect until forgotten.
    pub expired: bool,
    /// Absolute expiry deadline; `None` means no TTL.
    pub expire_at_ms: Option<u64>,
    /// When this record supersedes another (correct creates a new id).
    pub supersedes: Option<String>,
}

/// Draft content supplied by a writer. Authority is NOT on the draft.
#[derive(Debug, Clone)]
pub struct ClassMemoryDraft {
    pub text: String,
    pub importance: f32,
    pub turn_id: Option<String>,
    pub run_id: Option<String>,
    pub evidence: Vec<String>,
    pub session_id: Option<String>,
    pub evidence_tier: Option<String>,
    pub scope: Option<PersonalScope>,
    pub expire_at_ms: Option<u64>,
    pub supersedes: Option<String>,
}

impl ClassMemoryDraft {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            importance: 0.5,
            turn_id: None,
            run_id: None,
            evidence: Vec::new(),
            session_id: None,
            evidence_tier: None,
            scope: None,
            expire_at_ms: None,
            supersedes: None,
        }
    }

    pub fn with_importance(mut self, importance: f32) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    pub fn with_turn(mut self, turn_id: impl Into<String>) -> Self {
        self.turn_id = Some(turn_id.into());
        self
    }

    pub fn with_run(mut self, run_id: impl Into<String>) -> Self {
        self.run_id = Some(run_id.into());
        self
    }

    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    pub fn with_evidence_tier(mut self, tier: impl Into<String>) -> Self {
        self.evidence_tier = Some(tier.into());
        self
    }

    pub fn with_scope(mut self, scope: PersonalScope) -> Self {
        self.scope = Some(scope);
        self
    }

    pub fn with_expire_at_ms(mut self, at: u64) -> Self {
        self.expire_at_ms = Some(at);
        self
    }

    pub fn with_supersedes(mut self, id: impl Into<String>) -> Self {
        self.supersedes = Some(id.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Per-class budgets + retrieval
// ---------------------------------------------------------------------------

/// Independent token budgets per class for one compile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassBudgets {
    pub working: usize,
    pub episodic: usize,
    pub semantic_project: usize,
    pub procedural: usize,
    pub user: usize,
    pub verification: usize,
}

impl ClassBudgets {
    /// Split a total memory-region allowance into independent per-class caps.
    /// Fractions are intentional and sum to 1.0.
    pub fn from_total(total: usize) -> Self {
        let part = |pct: f32| ((total as f32) * pct).floor() as usize;
        Self {
            working: part(0.12),
            episodic: part(0.18),
            semantic_project: part(0.25),
            procedural: part(0.15),
            user: part(0.15),
            verification: part(0.15),
        }
    }

    /// Default budgets for tests / small windows.
    pub fn default_small() -> Self {
        Self {
            working: 64,
            episodic: 96,
            semantic_project: 128,
            procedural: 96,
            user: 64,
            verification: 96,
        }
    }

    pub fn for_class(&self, class: MemoryClass) -> usize {
        match class {
            MemoryClass::Working => self.working,
            MemoryClass::Episodic => self.episodic,
            MemoryClass::SemanticProject => self.semantic_project,
            MemoryClass::Procedural => self.procedural,
            MemoryClass::User => self.user,
            MemoryClass::Verification => self.verification,
        }
    }
}

/// One class's retrieval result for a compile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClassRetrievalSlice {
    pub class: MemoryClass,
    pub question: String,
    pub budget_tokens: usize,
    pub used_tokens: usize,
    pub hits: Vec<ClassMemoryRecord>,
}

/// Full multi-class retrieval for one compile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ClassCompileRetrieval {
    pub slices: Vec<ClassRetrievalSlice>,
}

impl ClassCompileRetrieval {
    pub fn budget_explanations(&self) -> Vec<String> {
        self.slices
            .iter()
            .map(|s| {
                format!(
                    "memory_class.{}: budget={} used={} hits={} question={:?}",
                    s.class.as_str(),
                    s.budget_tokens,
                    s.used_tokens,
                    s.hits.len(),
                    s.question
                )
            })
            .collect()
    }

    pub fn slice(&self, class: MemoryClass) -> Option<&ClassRetrievalSlice> {
        self.slices.iter().find(|s| s.class == class)
    }
}

// ---------------------------------------------------------------------------
// Classed memory system
// ---------------------------------------------------------------------------

/// The six real memory systems.
///
/// - `working`: RAM only, cleared by [`Self::end_turn`].
/// - `episodic` / `semantic_project` / `procedural` / `verification`: workspace SQLite, separate tables.
/// - `user`: separate user-scoped SQLite (not under the workspace root).
///
/// User controls (inspect / correct / pin / scope / expire / forget / export / disable)
/// apply across all six classes. Write authority types are never weakened: correcting
/// a verification record still requires [`VerifierWriteCap`].
pub struct ClassedMemorySystem {
    workspace_id: String,
    /// Live turn scratch: turn_id → records.
    working: Mutex<BTreeMap<String, Vec<ClassMemoryRecord>>>,
    /// Workspace-durable classes (four tables).
    workspace_db: Mutex<Connection>,
    /// User preferences (cross-workspace).
    user_db: Mutex<Connection>,
    /// Last compile retrieval (for meter explanations).
    last_retrieval: Mutex<Option<ClassCompileRetrieval>>,
    /// Paths kept for restart tests / diagnostics.
    workspace_db_path: Option<PathBuf>,
    user_db_path: Option<PathBuf>,
    /// Classes the user has disabled: no new writes, excluded from compile retrieval.
    /// Existing records remain inspectable and forgettable.
    disabled_classes: Mutex<BTreeSet<MemoryClass>>,
}

impl ClassedMemorySystem {
    /// Open (or create) durable stores. `user_db_path` must NOT live under the
    /// workspace so user memory survives workspace teardown and is shared.
    pub fn open(
        workspace_id: impl Into<String>,
        workspace_db_path: impl AsRef<Path>,
        user_db_path: impl AsRef<Path>,
    ) -> Result<Self> {
        let workspace_db_path = workspace_db_path.as_ref().to_path_buf();
        let user_db_path = user_db_path.as_ref().to_path_buf();
        if let Some(p) = workspace_db_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        if let Some(p) = user_db_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        let wconn = Connection::open(&workspace_db_path).map_err(sql_err)?;
        let uconn = Connection::open(&user_db_path).map_err(sql_err)?;
        init_workspace_schema(&wconn)?;
        init_user_schema(&uconn)?;
        Ok(Self {
            workspace_id: workspace_id.into(),
            working: Mutex::new(BTreeMap::new()),
            workspace_db: Mutex::new(wconn),
            user_db: Mutex::new(uconn),
            last_retrieval: Mutex::new(None),
            workspace_db_path: Some(workspace_db_path),
            user_db_path: Some(user_db_path),
            disabled_classes: Mutex::new(BTreeSet::new()),
        })
    }

    /// In-memory durable stores (tests).
    pub fn open_in_memory(workspace_id: impl Into<String>) -> Result<Self> {
        let wconn = Connection::open_in_memory().map_err(sql_err)?;
        let uconn = Connection::open_in_memory().map_err(sql_err)?;
        init_workspace_schema(&wconn)?;
        init_user_schema(&uconn)?;
        Ok(Self {
            workspace_id: workspace_id.into(),
            working: Mutex::new(BTreeMap::new()),
            workspace_db: Mutex::new(wconn),
            user_db: Mutex::new(uconn),
            last_retrieval: Mutex::new(None),
            workspace_db_path: None,
            user_db_path: None,
            disabled_classes: Mutex::new(BTreeSet::new()),
        })
    }

    /// Re-open durable paths after a simulated session restart (tests / recovery).
    pub fn reopen(&self) -> Result<Self> {
        let wpath = self
            .workspace_db_path
            .as_ref()
            .ok_or_else(|| HideError::Storage("no workspace db path to reopen".into()))?;
        let upath = self
            .user_db_path
            .as_ref()
            .ok_or_else(|| HideError::Storage("no user db path to reopen".into()))?;
        Self::open(self.workspace_id.clone(), wpath, upath)
    }

    pub fn workspace_id(&self) -> &str {
        &self.workspace_id
    }

    // ----- writes (capability-gated) ---------------------------------------

    fn ensure_class_enabled(&self, class: MemoryClass) -> Result<()> {
        if self.disabled_classes.lock().contains(&class) {
            return Err(HideError::PolicyDenied(format!(
                "memory class {} is disabled by user control",
                class.as_str()
            )));
        }
        Ok(())
    }

    fn resolve_scope(class: MemoryClass, draft: &ClassMemoryDraft) -> PersonalScope {
        draft
            .scope
            .unwrap_or_else(|| PersonalScope::default_for_class(class))
    }

    /// Write working (turn-local) memory. Requires [`TurnWriteCap`].
    pub fn write_working(
        &self,
        cap: &TurnWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Working)?;
        let scope = Self::resolve_scope(MemoryClass::Working, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("working"),
            class: MemoryClass::Working,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::Turn,
                Some(cap.turn_id.clone()),
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        self.working
            .lock()
            .entry(cap.turn_id.clone())
            .or_default()
            .push(rec.clone());
        Ok(rec)
    }

    /// Clear working memory for a turn. Working must not outlive its turn.
    pub fn end_turn(&self, turn_id: &str) {
        self.working.lock().remove(turn_id);
    }

    /// Append an episodic record. Requires [`EpisodicWriteCap`].
    pub fn write_episodic(
        &self,
        _cap: &EpisodicWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Episodic)?;
        let scope = Self::resolve_scope(MemoryClass::Episodic, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("episodic"),
            class: MemoryClass::Episodic,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::EventStream,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_episodic", &rec)?;
        Ok(rec)
    }

    /// Evict all episodic records for a session (session end / GC).
    /// Real deletion — not a tombstone.
    pub fn evict_session(&self, session_id: &str) -> Result<usize> {
        let conn = self.workspace_db.lock();
        let n = conn
            .execute(
                "DELETE FROM mem_episodic WHERE session_id = ?1",
                [session_id],
            )
            .map_err(sql_err)?;
        // Also drop any durable records that still reference this session id
        // in non-episodic tables (should be rare; keeps ephemeral cleanup honest).
        for table in [
            "mem_semantic_project",
            "mem_procedural",
            "mem_verification",
        ] {
            let _ = conn
                .execute(
                    &format!("DELETE FROM {table} WHERE session_id = ?1"),
                    [session_id],
                )
                .map_err(sql_err)?;
        }
        Ok(n)
    }

    /// Cap unbounded session growth: drop the oldest episodic rows for
    /// `session_id` until at most `keep` remain. Ids are ULIDs (time-ordered),
    /// so ascending order is oldest-first. Returns the number of rows deleted.
    pub fn prune_episodic_session(&self, session_id: &str, keep: usize) -> Result<usize> {
        let conn = self.workspace_db.lock();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM mem_episodic WHERE session_id = ?1",
                [session_id],
                |r| r.get(0),
            )
            .map_err(sql_err)?;
        let excess = (count as usize).saturating_sub(keep);
        if excess == 0 {
            return Ok(0);
        }
        // Same-table DELETE needs a nested subquery in SQLite.
        let n = conn
            .execute(
                "DELETE FROM mem_episodic WHERE id IN (
                    SELECT id FROM (
                        SELECT id FROM mem_episodic
                        WHERE session_id = ?1
                        ORDER BY id ASC
                        LIMIT ?2
                    )
                )",
                rusqlite::params![session_id, excess as i64],
            )
            .map_err(sql_err)?;
        Ok(n)
    }

    /// Write semantic_project. Requires [`ProjectWriteCap`].
    pub fn write_semantic_project(
        &self,
        _cap: &ProjectWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::SemanticProject)?;
        let scope = Self::resolve_scope(MemoryClass::SemanticProject, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("sem_proj"),
            class: MemoryClass::SemanticProject,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::ProjectDistill,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_semantic_project", &rec)?;
        Ok(rec)
    }

    /// Write procedural (successful tool recipe). Requires [`ProceduralWriteCap`].
    pub fn write_procedural(
        &self,
        _cap: &ProceduralWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Procedural)?;
        let scope = Self::resolve_scope(MemoryClass::Procedural, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("procedural"),
            class: MemoryClass::Procedural,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::ToolReceipt,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_procedural", &rec)?;
        Ok(rec)
    }

    /// Write user preference. Requires [`UserWriteCap`] — never distillation.
    /// Records have `workspace_id = None` (cross-workspace).
    pub fn write_user(
        &self,
        _cap: &UserWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::User)?;
        let scope = Self::resolve_scope(MemoryClass::User, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("user"),
            class: MemoryClass::User,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: None, // not workspace-scoped
            session_id: None,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::UserExplicit,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: None,
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_user(&self.user_db, &rec)?;
        Ok(rec)
    }

    /// Write verification claim/evidence. Requires [`VerifierWriteCap`].
    /// Authority is always `Verifier` — cannot be forged from the turn path.
    pub fn write_verification(
        &self,
        _cap: &VerifierWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.ensure_class_enabled(MemoryClass::Verification)?;
        let scope = Self::resolve_scope(MemoryClass::Verification, &draft);
        let rec = ClassMemoryRecord {
            id: mint_id("verify"),
            class: MemoryClass::Verification,
            scope,
            text: draft.text,
            importance: draft.importance.clamp(0.0, 1.0),
            workspace_id: Some(self.workspace_id.clone()),
            session_id: draft.session_id,
            provenance: ClassProvenance::stamped(
                writer,
                WriteAuthority::Verifier,
                draft.turn_id,
                draft.run_id,
                draft.evidence,
            ),
            evidence_tier: draft.evidence_tier.or_else(|| Some("asserted".into())),
            pinned: false,
            expired: false,
            expire_at_ms: draft.expire_at_ms,
            supersedes: draft.supersedes,
        };
        insert_workspace(&self.workspace_db, "mem_verification", &rec)?;
        Ok(rec)
    }

    // ----- reads -----------------------------------------------------------

    pub fn list_working(&self, turn_id: &str) -> Vec<ClassMemoryRecord> {
        self.working
            .lock()
            .get(turn_id)
            .cloned()
            .unwrap_or_default()
    }

    pub fn list_class(&self, class: MemoryClass) -> Result<Vec<ClassMemoryRecord>> {
        match class {
            MemoryClass::Working => {
                // All turns' working (rare; prefer list_working).
                let map = self.working.lock();
                Ok(map.values().flatten().cloned().collect())
            }
            MemoryClass::User => list_user(&self.user_db),
            MemoryClass::Episodic => list_workspace(&self.workspace_db, "mem_episodic"),
            MemoryClass::SemanticProject => {
                list_workspace(&self.workspace_db, "mem_semantic_project")
            }
            MemoryClass::Procedural => list_workspace(&self.workspace_db, "mem_procedural"),
            MemoryClass::Verification => list_workspace(&self.workspace_db, "mem_verification"),
        }
    }

    pub fn count(&self, class: MemoryClass) -> Result<usize> {
        Ok(self.list_class(class)?.len())
    }

    /// Retrieve for context compile: each class is asked its own question and
    /// filled under its own token budget. Results are independent — filling
    /// one class does not borrow from another.
    ///
    /// Disabled classes and expired records are excluded. Pinned records still
    /// participate if not expired.
    pub fn retrieve_for_compile(
        &self,
        task: &str,
        turn_id: Option<&str>,
        session_id: Option<&str>,
        budgets: &ClassBudgets,
    ) -> Result<ClassCompileRetrieval> {
        let disabled = self.disabled_classes.lock().clone();
        let mut slices = Vec::with_capacity(6);
        for class in MemoryClass::all() {
            let budget = budgets.for_class(class);
            if disabled.contains(&class) {
                slices.push(ClassRetrievalSlice {
                    class,
                    question: class.retrieval_question().to_string(),
                    budget_tokens: budget,
                    used_tokens: 0,
                    hits: Vec::new(),
                });
                continue;
            }
            let mut candidates = match class {
                MemoryClass::Working => match turn_id {
                    Some(t) => self.list_working(t),
                    None => Vec::new(),
                },
                MemoryClass::Episodic => {
                    let mut all = self.list_class(class)?;
                    if let Some(sid) = session_id {
                        all.retain(|r| r.session_id.as_deref() == Some(sid));
                    }
                    all
                }
                other => self.list_class(other)?,
            };
            candidates.retain(|r| !r.expired);
            let ranked = rank_for_query(task, candidates);
            let (hits, used) = pack_to_budget(ranked, budget);
            slices.push(ClassRetrievalSlice {
                class,
                question: class.retrieval_question().to_string(),
                budget_tokens: budget,
                used_tokens: used,
                hits,
            });
        }
        let retrieval = ClassCompileRetrieval { slices };
        *self.last_retrieval.lock() = Some(retrieval.clone());
        Ok(retrieval)
    }

    pub fn last_retrieval(&self) -> Option<ClassCompileRetrieval> {
        self.last_retrieval.lock().clone()
    }

    // ----- eight user controls ---------------------------------------------

    /// **inspect** — every durable record is reachable. No hidden permanent memory.
    pub fn inspect(&self, filter: &InspectFilter) -> Result<Vec<ClassMemoryRecord>> {
        let mut out = Vec::new();
        let classes: Vec<MemoryClass> = match filter.class {
            Some(c) => vec![c],
            None => MemoryClass::all().to_vec(),
        };
        for class in classes {
            if class == MemoryClass::Working && !filter.include_working {
                continue;
            }
            let mut rows = self.list_class(class)?;
            if let Some(scope) = filter.scope {
                rows.retain(|r| r.scope == scope);
            }
            if !filter.include_expired {
                rows.retain(|r| !r.expired);
            }
            out.extend(rows);
        }
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    /// Look up a single record by id across all stores (including working).
    pub fn get(&self, id: &str) -> Result<Option<ClassMemoryRecord>> {
        for class in MemoryClass::all() {
            for r in self.list_class(class)? {
                if r.id == id {
                    return Ok(Some(r));
                }
            }
        }
        Ok(None)
    }

    /// **correct** (user class). Supersession: new record names the old; both
    /// remain until forgotten. Requires [`UserWriteCap`].
    pub fn correct_user(
        &self,
        cap: &UserWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::User {
            return Err(HideError::InvalidState(
                "correct_user requires a user-class record".into(),
            ));
        }
        self.write_user(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_supersedes(id),
        )
    }

    /// **correct** (verification class). Still requires [`VerifierWriteCap`].
    pub fn correct_verification(
        &self,
        cap: &VerifierWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::Verification {
            return Err(HideError::InvalidState(
                "correct_verification requires a verification-class record".into(),
            ));
        }
        self.write_verification(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_evidence(old.provenance.evidence.clone())
                .with_evidence_tier(
                    old.evidence_tier
                        .clone()
                        .unwrap_or_else(|| "asserted".into()),
                )
                .with_supersedes(id),
        )
    }

    /// **correct** (semantic_project). Requires [`ProjectWriteCap`].
    pub fn correct_semantic_project(
        &self,
        cap: &ProjectWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::SemanticProject {
            return Err(HideError::InvalidState(
                "correct_semantic_project requires a semantic_project record".into(),
            ));
        }
        self.write_semantic_project(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_evidence(old.provenance.evidence.clone())
                .with_supersedes(id),
        )
    }

    /// **correct** (procedural). Requires [`ProceduralWriteCap`].
    pub fn correct_procedural(
        &self,
        cap: &ProceduralWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::Procedural {
            return Err(HideError::InvalidState(
                "correct_procedural requires a procedural record".into(),
            ));
        }
        self.write_procedural(
            cap,
            writer,
            ClassMemoryDraft::new(new_text.into())
                .with_scope(old.scope)
                .with_importance(old.importance)
                .with_evidence(old.provenance.evidence.clone())
                .with_supersedes(id),
        )
    }

    /// **correct** (episodic). Requires [`EpisodicWriteCap`].
    pub fn correct_episodic(
        &self,
        cap: &EpisodicWriteCap,
        id: &str,
        writer: impl Into<String>,
        new_text: impl Into<String>,
    ) -> Result<ClassMemoryRecord> {
        let old = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        if old.class != MemoryClass::Episodic {
            return Err(HideError::InvalidState(
                "correct_episodic requires an episodic record".into(),
            ));
        }
        let mut draft = ClassMemoryDraft::new(new_text.into())
            .with_scope(old.scope)
            .with_importance(old.importance)
            .with_supersedes(id);
        if let Some(sid) = old.session_id {
            draft = draft.with_session(sid);
        }
        self.write_episodic(cap, writer, draft)
    }

    /// **pin** — pinned records are exempt from expiry, never from forget.
    pub fn pin(&self, id: &str, pinned: bool) -> Result<()> {
        let rec = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        self.update_record_flags(&rec, Some(pinned), None)
    }

    /// **scope** — explicit, recorded scope transition. Connector content cannot
    /// reach global any other way.
    pub fn set_scope(
        &self,
        id: &str,
        to_scope: PersonalScope,
        approved_by: impl Into<String>,
    ) -> Result<ScopePromotion> {
        let approved_by = approved_by.into();
        if approved_by.is_empty() {
            return Err(HideError::InvalidState(
                "scope promotion requires an explicit approver".into(),
            ));
        }
        let rec = self
            .get(id)?
            .ok_or_else(|| HideError::NotFound(format!("memory record {id}")))?;
        let from = rec.scope;
        if from == to_scope {
            return Ok(ScopePromotion {
                record_id: id.into(),
                from_scope: from,
                to_scope,
                at_ms: now_ms(),
                approved_by,
            });
        }
        self.update_scope_column(&rec, to_scope)?;
        let promo = ScopePromotion {
            record_id: id.into(),
            from_scope: from,
            to_scope,
            at_ms: now_ms(),
            approved_by,
        };
        self.record_promotion(&promo)?;
        Ok(promo)
    }

    /// **expire** — mark due records expired (leave working set). Pinned survive.
    pub fn expire_due(&self, now_ms: u64) -> Result<usize> {
        let mut n = 0usize;
        for class in MemoryClass::all() {
            if class == MemoryClass::Working {
                // Working is turn-local; expire flag is not load-bearing there.
                continue;
            }
            for rec in self.list_class(class)? {
                if rec.expired || rec.pinned {
                    continue;
                }
                if let Some(at) = rec.expire_at_ms {
                    if now_ms >= at {
                        self.update_record_flags(&rec, None, Some(true))?;
                        n += 1;
                    }
                }
            }
        }
        Ok(n)
    }

    /// **forget** — real deletion for user-owned data. Not a tombstone.
    ///
    /// Returns true if a record was removed. Works for every class, including
    /// working (RAM) and durable tables. Also clears dangling edges: scope
    /// promotions for the id, and any `supersedes` pointers that named it.
    /// Export after forget must not reintroduce the forgotten row via residual
    /// audit edges.
    pub fn forget(&self, id: &str) -> Result<bool> {
        let mut removed = false;
        // Working (RAM)
        {
            let mut map = self.working.lock();
            for rows in map.values_mut() {
                let before = rows.len();
                rows.retain(|r| r.id != id);
                if rows.len() < before {
                    removed = true;
                }
                // Clear supersedes edges in RAM that pointed at the forgotten id.
                for r in rows.iter_mut() {
                    if r.supersedes.as_deref() == Some(id) {
                        r.supersedes = None;
                    }
                }
            }
        }
        // Durable workspace tables
        {
            let conn = self.workspace_db.lock();
            for table in [
                "mem_episodic",
                "mem_semantic_project",
                "mem_procedural",
                "mem_verification",
            ] {
                let n = conn
                    .execute(&format!("DELETE FROM {table} WHERE id = ?1"), [id])
                    .map_err(sql_err)?;
                if n > 0 {
                    removed = true;
                }
                // Null supersedes edges that named the forgotten record.
                conn.execute(
                    &format!("UPDATE {table} SET supersedes = NULL WHERE supersedes = ?1"),
                    [id],
                )
                .map_err(sql_err)?;
            }
            // Scope-promotion audit rows for a forgotten record are dangling edges.
            conn.execute(
                "DELETE FROM mem_scope_promotions WHERE record_id = ?1",
                [id],
            )
            .map_err(sql_err)?;
        }
        // User db
        {
            let conn = self.user_db.lock();
            let n = conn
                .execute("DELETE FROM mem_user WHERE id = ?1", [id])
                .map_err(sql_err)?;
            if n > 0 {
                removed = true;
            }
            conn.execute(
                "UPDATE mem_user SET supersedes = NULL WHERE supersedes = ?1",
                [id],
            )
            .map_err(sql_err)?;
        }
        Ok(removed)
    }

    /// Forget every record in a personal scope (real deletion).
    pub fn forget_scope(&self, scope: PersonalScope) -> Result<usize> {
        let ids: Vec<String> = self
            .inspect(&InspectFilter {
                scope: Some(scope),
                include_expired: true,
                include_working: true,
                ..Default::default()
            })?
            .into_iter()
            .map(|r| r.id)
            .collect();
        let mut n = 0usize;
        for id in ids {
            if self.forget(&id)? {
                n += 1;
            }
        }
        Ok(n)
    }

    /// **export** — portable, complete, readable without this tool (JSON).
    pub fn export(&self) -> Result<MemoryExport> {
        let records = self.inspect(&InspectFilter {
            include_expired: true,
            include_working: false,
            ..Default::default()
        })?;
        let promotions = self.list_promotions()?;
        let disabled: Vec<String> = self
            .disabled_classes
            .lock()
            .iter()
            .map(|c| c.as_str().to_string())
            .collect();
        Ok(MemoryExport {
            schema: "hide.you.memory_export.v1".into(),
            exported_at_ms: now_ms(),
            workspace_id: self.workspace_id.clone(),
            records,
            promotions,
            disabled_classes: disabled,
        })
    }

    /// **disable** — user control that blocks new writes and compile retrieval
    /// for a class. Existing records remain inspectable and forgettable.
    pub fn disable_class(&self, class: MemoryClass, disabled: bool) {
        let mut set = self.disabled_classes.lock();
        if disabled {
            set.insert(class);
        } else {
            set.remove(&class);
        }
    }

    pub fn is_class_disabled(&self, class: MemoryClass) -> bool {
        self.disabled_classes.lock().contains(&class)
    }

    pub fn list_promotions(&self) -> Result<Vec<ScopePromotion>> {
        let conn = self.workspace_db.lock();
        let mut stmt = conn
            .prepare(
                "SELECT record_id, from_scope, to_scope, at_ms, approved_by
                 FROM mem_scope_promotions ORDER BY at_ms ASC",
            )
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |row| {
                let from_s: String = row.get(1)?;
                let to_s: String = row.get(2)?;
                Ok(ScopePromotion {
                    record_id: row.get(0)?,
                    from_scope: PersonalScope::parse(&from_s).unwrap_or(PersonalScope::Workspace),
                    to_scope: PersonalScope::parse(&to_s).unwrap_or(PersonalScope::Workspace),
                    at_ms: row.get::<_, i64>(3)? as u64,
                    approved_by: row.get(4)?,
                })
            })
            .map_err(sql_err)?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(sql_err)?;
        Ok(rows)
    }

    /// Count durable records that still reference a session id (any class table).
    pub fn durable_refs_to_session(&self, session_id: &str) -> Result<usize> {
        let mut n = 0usize;
        for class in [
            MemoryClass::Episodic,
            MemoryClass::SemanticProject,
            MemoryClass::Procedural,
            MemoryClass::Verification,
        ] {
            n += self
                .list_class(class)?
                .into_iter()
                .filter(|r| r.session_id.as_deref() == Some(session_id))
                .count();
        }
        Ok(n)
    }

    fn record_promotion(&self, promo: &ScopePromotion) -> Result<()> {
        let conn = self.workspace_db.lock();
        conn.execute(
            "INSERT INTO mem_scope_promotions
             (record_id, from_scope, to_scope, at_ms, approved_by)
             VALUES (?1,?2,?3,?4,?5)",
            rusqlite::params![
                promo.record_id,
                promo.from_scope.as_str(),
                promo.to_scope.as_str(),
                promo.at_ms as i64,
                promo.approved_by,
            ],
        )
        .map_err(sql_err)?;
        Ok(())
    }

    fn update_scope_column(&self, rec: &ClassMemoryRecord, to: PersonalScope) -> Result<()> {
        match rec.class {
            MemoryClass::Working => {
                let mut map = self.working.lock();
                for rows in map.values_mut() {
                    for r in rows.iter_mut() {
                        if r.id == rec.id {
                            r.scope = to;
                            return Ok(());
                        }
                    }
                }
                Err(HideError::NotFound(format!("memory record {}", rec.id)))
            }
            MemoryClass::User => {
                let conn = self.user_db.lock();
                let n = conn
                    .execute(
                        "UPDATE mem_user SET scope = ?1 WHERE id = ?2",
                        rusqlite::params![to.as_str(), rec.id],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
            other => {
                let table = table_for_class(other)?;
                let conn = self.workspace_db.lock();
                let n = conn
                    .execute(
                        &format!("UPDATE {table} SET scope = ?1 WHERE id = ?2"),
                        rusqlite::params![to.as_str(), rec.id],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
        }
    }

    fn update_record_flags(
        &self,
        rec: &ClassMemoryRecord,
        pinned: Option<bool>,
        expired: Option<bool>,
    ) -> Result<()> {
        match rec.class {
            MemoryClass::Working => {
                let mut map = self.working.lock();
                for rows in map.values_mut() {
                    for r in rows.iter_mut() {
                        if r.id == rec.id {
                            if let Some(p) = pinned {
                                r.pinned = p;
                            }
                            if let Some(e) = expired {
                                r.expired = e;
                            }
                            return Ok(());
                        }
                    }
                }
                Err(HideError::NotFound(format!("memory record {}", rec.id)))
            }
            MemoryClass::User => {
                let conn = self.user_db.lock();
                let n = conn
                    .execute(
                        "UPDATE mem_user SET
                           pinned = COALESCE(?1, pinned),
                           expired = COALESCE(?2, expired)
                         WHERE id = ?3",
                        rusqlite::params![
                            pinned.map(|b| b as i64),
                            expired.map(|b| b as i64),
                            rec.id,
                        ],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
            other => {
                let table = table_for_class(other)?;
                let conn = self.workspace_db.lock();
                let n = conn
                    .execute(
                        &format!(
                            "UPDATE {table} SET
                               pinned = COALESCE(?1, pinned),
                               expired = COALESCE(?2, expired)
                             WHERE id = ?3"
                        ),
                        rusqlite::params![
                            pinned.map(|b| b as i64),
                            expired.map(|b| b as i64),
                            rec.id,
                        ],
                    )
                    .map_err(sql_err)?;
                if n == 0 {
                    return Err(HideError::NotFound(format!("memory record {}", rec.id)));
                }
                Ok(())
            }
        }
    }
}

fn table_for_class(class: MemoryClass) -> Result<&'static str> {
    match class {
        MemoryClass::Episodic => Ok("mem_episodic"),
        MemoryClass::SemanticProject => Ok("mem_semantic_project"),
        MemoryClass::Procedural => Ok("mem_procedural"),
        MemoryClass::Verification => Ok("mem_verification"),
        MemoryClass::Working | MemoryClass::User => Err(HideError::Storage(
            "working/user are not workspace tables".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// Ranking / packing helpers
// ---------------------------------------------------------------------------

fn rank_for_query(task: &str, mut records: Vec<ClassMemoryRecord>) -> Vec<ClassMemoryRecord> {
    let task_l = task.to_lowercase();
    let terms: Vec<&str> = task_l.split_whitespace().filter(|t| t.len() > 2).collect();
    records.sort_by(|a, b| {
        let sa = score_record(&terms, a);
        let sb = score_record(&terms, b);
        sb.partial_cmp(&sa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    records
}

fn score_record(terms: &[&str], r: &ClassMemoryRecord) -> f32 {
    let text = r.text.to_lowercase();
    let overlap = if terms.is_empty() {
        0.5
    } else {
        let hits = terms.iter().filter(|t| text.contains(*t)).count();
        hits as f32 / terms.len() as f32
    };
    overlap + r.importance.clamp(0.0, 1.0)
}

fn pack_to_budget(
    ranked: Vec<ClassMemoryRecord>,
    budget: usize,
) -> (Vec<ClassMemoryRecord>, usize) {
    if budget == 0 {
        return (Vec::new(), 0);
    }
    let mut hits = Vec::new();
    let mut used = 0usize;
    for r in ranked {
        let cost = estimate_tokens(&r.text).max(1);
        if used + cost > budget {
            continue;
        }
        used += cost;
        hits.push(r);
    }
    (hits, used)
}

fn mint_id(prefix: &str) -> String {
    format!("{prefix}_{}", ulid::Ulid::new())
}

fn sql_err(e: rusqlite::Error) -> HideError {
    HideError::Storage(format!("classed memory: {e}"))
}

fn init_workspace_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS mem_episodic (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'conversation',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_semantic_project (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'workspace',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_procedural (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'workspace',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_verification (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'workspace',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        CREATE TABLE IF NOT EXISTS mem_scope_promotions (
            record_id TEXT NOT NULL,
            from_scope TEXT NOT NULL,
            to_scope TEXT NOT NULL,
            at_ms INTEGER NOT NULL,
            approved_by TEXT NOT NULL
        );
        "#,
    )
    .map_err(sql_err)?;
    // Same class of bug as user.db: CREATE TABLE IF NOT EXISTS never alters an
    // existing workspace file that pre-dates the control columns.
    migrate_add_missing_columns(
        conn,
        "mem_episodic",
        &[
            ("scope", "TEXT NOT NULL DEFAULT 'conversation'"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
            ("expired", "INTEGER NOT NULL DEFAULT 0"),
            ("expire_at_ms", "INTEGER"),
            ("supersedes", "TEXT"),
        ],
    )?;
    for table in [
        "mem_semantic_project",
        "mem_procedural",
        "mem_verification",
    ] {
        migrate_add_missing_columns(
            conn,
            table,
            &[
                ("scope", "TEXT NOT NULL DEFAULT 'workspace'"),
                ("pinned", "INTEGER NOT NULL DEFAULT 0"),
                ("expired", "INTEGER NOT NULL DEFAULT 0"),
                ("expire_at_ms", "INTEGER"),
                ("supersedes", "TEXT"),
            ],
        )?;
    }
    Ok(())
}

fn init_user_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS mem_user (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            importance REAL NOT NULL,
            workspace_id TEXT,
            session_id TEXT,
            provenance_json TEXT NOT NULL,
            evidence_tier TEXT,
            scope TEXT NOT NULL DEFAULT 'global',
            pinned INTEGER NOT NULL DEFAULT 0,
            expired INTEGER NOT NULL DEFAULT 0,
            expire_at_ms INTEGER,
            supersedes TEXT
        );
        "#,
    )
    .map_err(sql_err)?;
    migrate_add_missing_columns(
        conn,
        "mem_user",
        &[
            ("scope", "TEXT NOT NULL DEFAULT 'global'"),
            ("pinned", "INTEGER NOT NULL DEFAULT 0"),
            ("expired", "INTEGER NOT NULL DEFAULT 0"),
            ("expire_at_ms", "INTEGER"),
            ("supersedes", "TEXT"),
        ],
    )?;
    Ok(())
}

/// Add columns an older database is missing.
///
/// `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, and the user
/// class is deliberately cross-workspace -- its database lives under the user root and
/// survives every workspace, every test run and every schema change. So a `user.db` written
/// before the scope/retention columns existed stays exactly as it was while the SELECT that
/// reads it grows new columns, and every read fails with "no such column: scope".
///
/// That is what happened here: the file was created by the first six-class lane, the second
/// lane added the columns to CREATE TABLE and to the query, and nothing migrated the file in
/// between. Idempotent, and cheap enough to run on every open.
fn migrate_add_missing_columns(
    conn: &Connection,
    table: &str,
    columns: &[(&str, &str)],
) -> Result<()> {
    let mut have: Vec<String> = Vec::new();
    {
        let mut stmt = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .map_err(sql_err)?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(1))
            .map_err(sql_err)?;
        for r in rows {
            have.push(r.map_err(sql_err)?);
        }
    }
    for (name, decl) in columns {
        if !have.iter().any(|h| h == name) {
            conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {name} {decl};"))
                .map_err(sql_err)?;
        }
    }
    Ok(())
}

fn insert_workspace(
    conn: &Mutex<Connection>,
    table: &str,
    rec: &ClassMemoryRecord,
) -> Result<()> {
    // Table names are compile-time constants only.
    debug_assert!(matches!(
        table,
        "mem_episodic" | "mem_semantic_project" | "mem_procedural" | "mem_verification"
    ));
    let prov = serde_json::to_string(&rec.provenance)?;
    let sql = format!(
        "INSERT OR REPLACE INTO {table}
         (id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
          scope, pinned, expired, expire_at_ms, supersedes)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)"
    );
    let conn = conn.lock();
    conn.execute(
        &sql,
        rusqlite::params![
            rec.id,
            rec.text,
            rec.importance as f64,
            rec.workspace_id,
            rec.session_id,
            prov,
            rec.evidence_tier,
            rec.scope.as_str(),
            rec.pinned as i64,
            rec.expired as i64,
            rec.expire_at_ms.map(|v| v as i64),
            rec.supersedes,
        ],
    )
    .map_err(sql_err)?;
    Ok(())
}

fn insert_user(conn: &Mutex<Connection>, rec: &ClassMemoryRecord) -> Result<()> {
    let prov = serde_json::to_string(&rec.provenance)?;
    let conn = conn.lock();
    conn.execute(
        "INSERT OR REPLACE INTO mem_user
         (id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
          scope, pinned, expired, expire_at_ms, supersedes)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
        rusqlite::params![
            rec.id,
            rec.text,
            rec.importance as f64,
            rec.workspace_id, // always None for user
            rec.session_id,
            prov,
            rec.evidence_tier,
            rec.scope.as_str(),
            rec.pinned as i64,
            rec.expired as i64,
            rec.expire_at_ms.map(|v| v as i64),
            rec.supersedes,
        ],
    )
    .map_err(sql_err)?;
    Ok(())
}

fn list_workspace(conn: &Mutex<Connection>, table: &str) -> Result<Vec<ClassMemoryRecord>> {
    debug_assert!(matches!(
        table,
        "mem_episodic" | "mem_semantic_project" | "mem_procedural" | "mem_verification"
    ));
    let class = match table {
        "mem_episodic" => MemoryClass::Episodic,
        "mem_semantic_project" => MemoryClass::SemanticProject,
        "mem_procedural" => MemoryClass::Procedural,
        "mem_verification" => MemoryClass::Verification,
        _ => return Err(HideError::Storage(format!("unknown table {table}"))),
    };
    let sql = format!(
        "SELECT id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
                scope, pinned, expired, expire_at_ms, supersedes
         FROM {table}"
    );
    let conn = conn.lock();
    let mut stmt = conn.prepare(&sql).map_err(sql_err)?;
    let rows = stmt
        .query_map([], |row| row_to_record(row, class))
        .map_err(sql_err)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql_err)?;
    Ok(rows)
}

fn list_user(conn: &Mutex<Connection>) -> Result<Vec<ClassMemoryRecord>> {
    let conn = conn.lock();
    let mut stmt = conn
        .prepare(
            "SELECT id, text, importance, workspace_id, session_id, provenance_json, evidence_tier,
                    scope, pinned, expired, expire_at_ms, supersedes
             FROM mem_user",
        )
        .map_err(sql_err)?;
    let rows = stmt
        .query_map([], |row| row_to_record(row, MemoryClass::User))
        .map_err(sql_err)?
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(sql_err)?;
    Ok(rows)
}

fn row_to_record(
    row: &rusqlite::Row<'_>,
    class: MemoryClass,
) -> rusqlite::Result<ClassMemoryRecord> {
    let provenance_json: String = row.get(5)?;
    let provenance: ClassProvenance = serde_json::from_str(&provenance_json).unwrap_or_else(|_| {
        ClassProvenance {
            writer: "unknown".into(),
            written_at_ms: 0,
            turn_id: None,
            run_id: None,
            evidence: Vec::new(),
            authority: WriteAuthority::Turn,
        }
    });
    let scope_s: String = row.get(7)?;
    let scope = PersonalScope::parse(&scope_s)
        .unwrap_or_else(|| PersonalScope::default_for_class(class));
    Ok(ClassMemoryRecord {
        id: row.get(0)?,
        class,
        scope,
        text: row.get(1)?,
        importance: row.get::<_, f64>(2)? as f32,
        workspace_id: row.get(3)?,
        session_id: row.get(4)?,
        provenance,
        evidence_tier: row.get(6)?,
        pinned: row.get::<_, i64>(8)? != 0,
        expired: row.get::<_, i64>(9)? != 0,
        expire_at_ms: row
            .get::<_, Option<i64>>(10)?
            .map(|v| v as u64),
        supersedes: row.get(11)?,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn system() -> ClassedMemorySystem {
        ClassedMemorySystem::open_in_memory("ws-test").unwrap()
    }

    // --- retention / lifetime (one per class) ------------------------------

    #[test]
    fn retention_working_dies_at_turn_end() {
        let sys = system();
        let cap = TurnWriteCap::new("turn-1");
        sys.write_working(
            &cap,
            "kernel",
            ClassMemoryDraft::new("open tool: edit foo.rs"),
        )
        .unwrap();
        assert_eq!(sys.list_working("turn-1").len(), 1);
        sys.end_turn("turn-1");
        assert!(
            sys.list_working("turn-1").is_empty(),
            "working must die at turn end"
        );
    }

    #[test]
    fn retention_episodic_evicted_with_session() {
        let sys = system();
        let cap = EpisodicWriteCap::mint();
        sys.write_episodic(
            &cap,
            "event_stream",
            ClassMemoryDraft::new("tried cargo test, failed")
                .with_session("sess-a")
                .with_turn("t1"),
        )
        .unwrap();
        sys.write_episodic(
            &cap,
            "event_stream",
            ClassMemoryDraft::new("other session note").with_session("sess-b"),
        )
        .unwrap();
        assert_eq!(sys.evict_session("sess-a").unwrap(), 1);
        let left = sys.list_class(MemoryClass::Episodic).unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].session_id.as_deref(), Some("sess-b"));
    }

    #[test]
    fn retention_semantic_project_survives_session_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user.db");
        let sys = ClassedMemorySystem::open("ws-restart", &wdb, &udb).unwrap();
        let cap = ProjectWriteCap::mint();
        sys.write_semantic_project(
            &cap,
            "distill",
            ClassMemoryDraft::new("layout: crates/hawking-context owns the compiler")
                .with_evidence(vec!["scan:crates/hawking-context".into()]),
        )
        .unwrap();
        drop(sys);
        // Session restart = reopen durable path; working is gone, semantic stays.
        let sys2 = ClassedMemorySystem::open("ws-restart", &wdb, &udb).unwrap();
        let hits = sys2.list_class(MemoryClass::SemanticProject).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].text.contains("hawking-context"));
        assert_eq!(hits[0].workspace_id.as_deref(), Some("ws-restart"));
    }

    #[test]
    fn retention_procedural_survives_session_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user.db");
        let sys = ClassedMemorySystem::open("ws-proc", &wdb, &udb).unwrap();
        let cap = ProceduralWriteCap::mint();
        sys.write_procedural(
            &cap,
            "tool_receipt",
            ClassMemoryDraft::new("cargo test -p hawking-context works")
                .with_evidence(vec!["exit_code:0".into()]),
        )
        .unwrap();
        let sys2 = ClassedMemorySystem::open("ws-proc", &wdb, &udb).unwrap();
        assert_eq!(sys2.count(MemoryClass::Procedural).unwrap(), 1);
    }

    #[test]
    fn retention_user_is_not_workspace_scoped() {
        let dir = tempfile::tempdir().unwrap();
        // Shared user db across two "workspaces".
        let udb = dir.path().join("user_shared.db");
        let w1 = dir.path().join("ws1.db");
        let w2 = dir.path().join("ws2.db");
        let a = ClassedMemorySystem::open("workspace-a", &w1, &udb).unwrap();
        let cap = UserWriteCap::mint();
        a.write_user(
            &cap,
            "user_intent",
            ClassMemoryDraft::new("prefer concise answers"),
        )
        .unwrap();
        let rec = a.list_class(MemoryClass::User).unwrap();
        assert!(rec[0].workspace_id.is_none(), "user must not be workspace-scoped");

        // Different workspace id, same user db → preference still there.
        let b = ClassedMemorySystem::open("workspace-b", &w2, &udb).unwrap();
        let prefs = b.list_class(MemoryClass::User).unwrap();
        assert_eq!(prefs.len(), 1);
        assert!(prefs[0].text.contains("concise"));
        assert!(prefs[0].workspace_id.is_none());
    }

    #[test]
    fn retention_verification_survives_session_restart() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user.db");
        let sys = ClassedMemorySystem::open("ws-v", &wdb, &udb).unwrap();
        let cap = VerifierWriteCap::mint();
        sys.write_verification(
            &cap,
            "verifier",
            ClassMemoryDraft::new("claim: memory classes are separate stores")
                .with_evidence(vec!["test:retention_verification".into()])
                .with_evidence_tier("proven")
                .with_run("run-99"),
        )
        .unwrap();
        let sys2 = ClassedMemorySystem::open("ws-v", &wdb, &udb).unwrap();
        let hits = sys2.list_class(MemoryClass::Verification).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].evidence_tier.as_deref(), Some("proven"));
        assert_eq!(hits[0].provenance.authority, WriteAuthority::Verifier);
    }

    // --- write authority ---------------------------------------------------

    #[test]
    fn write_authority_verification_and_user_require_caps() {
        let sys = system();
        // Type system: only these methods accept VerifierWriteCap / UserWriteCap.
        // Runtime proof: stamped authority cannot be supplied by the draft.
        let vcap = VerifierWriteCap::mint();
        let v = sys
            .write_verification(
                &vcap,
                "verifier",
                ClassMemoryDraft::new("proven fact").with_evidence_tier("proven"),
            )
            .unwrap();
        assert_eq!(v.provenance.authority, WriteAuthority::Verifier);
        assert_eq!(v.class, MemoryClass::Verification);

        let ucap = UserWriteCap::mint();
        let u = sys
            .write_user(&ucap, "user_intent", ClassMemoryDraft::new("be terse"))
            .unwrap();
        assert_eq!(u.provenance.authority, WriteAuthority::UserExplicit);
        assert!(u.workspace_id.is_none());

        // Turn path can write working with TurnWriteCap; authority is Turn.
        let tcap = TurnWriteCap::new("t");
        let w = sys
            .write_working(&tcap, "kernel", ClassMemoryDraft::new("scratch"))
            .unwrap();
        assert_eq!(w.provenance.authority, WriteAuthority::Turn);
        assert_ne!(
            w.provenance.authority,
            WriteAuthority::Verifier,
            "turn path must not stamp verifier authority"
        );
        assert_ne!(w.provenance.authority, WriteAuthority::UserExplicit);

        // There is no write_verification without VerifierWriteCap — compile-time.
        // Runtime: a turn-stamped record can never land in verification table
        // because only write_verification inserts there.
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 1);
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 1);
        assert_eq!(sys.count(MemoryClass::Working).unwrap(), 1);
    }

    // --- multi-class compile budgets ---------------------------------------

    #[test]
    fn retrieve_for_compile_uses_independent_class_budgets() {
        let sys = system();
        // Flood semantic with long text; give it a tiny budget.
        let pcap = ProjectWriteCap::mint();
        for i in 0..20 {
            sys.write_semantic_project(
                &pcap,
                "distill",
                ClassMemoryDraft::new(format!(
                    "semantic project fact number {i} about the repository layout and conventions that are durable"
                ))
                .with_importance(0.9),
            )
            .unwrap();
        }
        let vcap = VerifierWriteCap::mint();
        sys.write_verification(
            &vcap,
            "verifier",
            ClassMemoryDraft::new("verification: the six classes are real")
                .with_evidence_tier("proven")
                .with_importance(1.0),
        )
        .unwrap();
        let ucap = UserWriteCap::mint();
        sys.write_user(
            &ucap,
            "user_intent",
            ClassMemoryDraft::new("user prefers short diffs").with_importance(1.0),
        )
        .unwrap();

        let budgets = ClassBudgets {
            working: 0,
            episodic: 0,
            semantic_project: 40, // tight — cannot take all 20
            procedural: 0,
            user: 200,
            verification: 200,
        };
        let ret = sys
            .retrieve_for_compile("repository layout conventions", None, None, &budgets)
            .unwrap();

        let sem = ret.slice(MemoryClass::SemanticProject).unwrap();
        let ver = ret.slice(MemoryClass::Verification).unwrap();
        let user = ret.slice(MemoryClass::User).unwrap();

        assert!(
            sem.used_tokens <= sem.budget_tokens,
            "semantic used {} > budget {}",
            sem.used_tokens,
            sem.budget_tokens
        );
        assert!(
            sem.hits.len() < 20,
            "semantic budget must cap hits, got {}",
            sem.hits.len()
        );
        // Independent: verification still retrieved despite semantic flood.
        assert_eq!(ver.hits.len(), 1, "verification must not be starved");
        assert_eq!(user.hits.len(), 1, "user must not be starved");
        assert!(ver.used_tokens <= ver.budget_tokens);
        assert!(user.used_tokens <= user.budget_tokens);
        // Six slices, each with its own question.
        assert_eq!(ret.slices.len(), 6);
        assert!(ret
            .slices
            .iter()
            .any(|s| s.question.contains("durable facts")));
        assert!(ret
            .slices
            .iter()
            .any(|s| s.question.contains("asserted vs proven")));
    }

    // --- provenance non-forgeable from turn path ---------------------------

    #[test]
    fn provenance_authority_not_forgeable_from_turn_path() {
        let sys = system();
        let tcap = TurnWriteCap::new("turn-x");
        // Even if the draft tries to look "official", authority is stamped Turn.
        let rec = sys
            .write_working(
                &tcap,
                "kernel.turn",
                ClassMemoryDraft::new("I am totally a verifier claim")
                    .with_evidence(vec!["forged".into()])
                    .with_evidence_tier("proven")
                    .with_run("fake-run"),
            )
            .unwrap();
        assert_eq!(rec.provenance.authority, WriteAuthority::Turn);
        assert_eq!(rec.provenance.writer, "kernel.turn");
        assert_eq!(rec.provenance.turn_id.as_deref(), Some("turn-x"));
        assert!(rec.provenance.written_at_ms > 0);
        // Working write does not create a verification row.
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 0);

        // Only VerifierWriteCap path stamps Verifier + lands in verification table.
        let v = sys
            .write_verification(
                &VerifierWriteCap::mint(),
                "hide-verify",
                ClassMemoryDraft::new("real claim")
                    .with_evidence(vec!["test:x".into()])
                    .with_run("run-1"),
            )
            .unwrap();
        assert_eq!(v.provenance.authority, WriteAuthority::Verifier);
        assert_eq!(v.provenance.run_id.as_deref(), Some("run-1"));
        assert!(!v.provenance.evidence.is_empty());
    }

    // --- eight user controls + properties ----------------------------------

    #[test]
    fn property_no_hidden_permanent_memory_inspect_and_forget() {
        // Every durable record is reachable by inspect and removable by forget;
        // forget is real deletion for user-scoped data.
        let sys = system();
        let u = sys
            .write_user(
                &UserWriteCap::mint(),
                "user_intent",
                ClassMemoryDraft::new("prefer dark mode"),
            )
            .unwrap();
        let v = sys
            .write_verification(
                &VerifierWriteCap::mint(),
                "verifier",
                ClassMemoryDraft::new("claim proven")
                    .with_evidence_tier("proven"),
            )
            .unwrap();
        let p = sys
            .write_semantic_project(
                &ProjectWriteCap::mint(),
                "distill",
                ClassMemoryDraft::new("crate layout fact"),
            )
            .unwrap();

        let all = sys
            .inspect(&InspectFilter {
                include_expired: true,
                ..Default::default()
            })
            .unwrap();
        let ids: Vec<_> = all.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&u.id.as_str()), "user record must be inspectable");
        assert!(ids.contains(&v.id.as_str()), "verification must be inspectable");
        assert!(ids.contains(&p.id.as_str()), "semantic must be inspectable");

        assert!(sys.forget(&u.id).unwrap());
        assert!(sys.get(&u.id).unwrap().is_none(), "forget must really delete");
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);

        // export is portable JSON-serializable without this tool
        let exp = sys.export().unwrap();
        assert_eq!(exp.schema, "hide.you.memory_export.v1");
        let json = serde_json::to_string_pretty(&exp).unwrap();
        assert!(json.contains("claim proven") || json.contains(&v.id));
        let round: MemoryExport = serde_json::from_str(&json).unwrap();
        assert_eq!(round.records.len(), exp.records.len());
    }

    #[test]
    fn property_correct_verification_requires_verifier_cap() {
        let sys = system();
        let rec = sys
            .write_verification(
                &VerifierWriteCap::mint(),
                "verifier",
                ClassMemoryDraft::new("old claim").with_evidence_tier("asserted"),
            )
            .unwrap();
        let corrected = sys
            .correct_verification(
                &VerifierWriteCap::mint(),
                &rec.id,
                "verifier",
                "corrected claim",
            )
            .unwrap();
        assert_eq!(corrected.provenance.authority, WriteAuthority::Verifier);
        assert_eq!(corrected.supersedes.as_deref(), Some(rec.id.as_str()));
        // Both remain until forgotten (supersession, not in-place edit).
        assert_eq!(sys.count(MemoryClass::Verification).unwrap(), 2);
        assert!(sys.get(&rec.id).unwrap().is_some());
    }

    #[test]
    fn property_scope_orthogonal_to_class_and_promotion_recorded() {
        let sys = system();
        let rec = sys
            .write_episodic(
                &EpisodicWriteCap::mint(),
                "event_stream",
                ClassMemoryDraft::new("email snippet")
                    .with_scope(PersonalScope::Connector)
                    .with_session("s1"),
            )
            .unwrap();
        assert_eq!(rec.class, MemoryClass::Episodic);
        assert_eq!(rec.scope, PersonalScope::Connector);

        // Not global until explicit promotion.
        let global = sys
            .inspect(&InspectFilter {
                scope: Some(PersonalScope::Global),
                include_expired: true,
                ..Default::default()
            })
            .unwrap();
        assert!(global.is_empty());

        let promo = sys
            .set_scope(&rec.id, PersonalScope::Global, "user")
            .unwrap();
        assert_eq!(promo.from_scope, PersonalScope::Connector);
        assert_eq!(promo.to_scope, PersonalScope::Global);
        assert_eq!(promo.approved_by, "user");

        let after = sys.get(&rec.id).unwrap().unwrap();
        assert_eq!(after.scope, PersonalScope::Global);
        assert_eq!(after.class, MemoryClass::Episodic); // class unchanged
        assert_eq!(sys.list_promotions().unwrap().len(), 1);

        // Empty approver refused.
        assert!(sys
            .set_scope(&rec.id, PersonalScope::Person, "")
            .is_err());
    }

    #[test]
    fn property_pin_expire_disable_controls() {
        let sys = system();
        let a = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("will expire").with_expire_at_ms(100),
            )
            .unwrap();
        let b = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("pinned forever").with_expire_at_ms(100),
            )
            .unwrap();
        sys.pin(&b.id, true).unwrap();
        assert_eq!(sys.expire_due(200).unwrap(), 1);
        let a2 = sys.get(&a.id).unwrap().unwrap();
        let b2 = sys.get(&b.id).unwrap().unwrap();
        assert!(a2.expired);
        assert!(!b2.expired);
        assert!(b2.pinned);

        // disable blocks writes and compile retrieval; records stay inspectable
        sys.disable_class(MemoryClass::User, true);
        assert!(sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("blocked"),
            )
            .is_err());
        let still = sys
            .inspect(&InspectFilter {
                class: Some(MemoryClass::User),
                include_expired: true,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(still.len(), 2);
        let ret = sys
            .retrieve_for_compile("prefer", None, None, &ClassBudgets::default_small())
            .unwrap();
        assert!(
            ret.slice(MemoryClass::User).unwrap().hits.is_empty(),
            "disabled class must not enter compile"
        );
    }

    #[test]
    fn property_export_and_real_deletion_path() {
        let sys = system();
        let id = sys
            .write_procedural(
                &ProceduralWriteCap::mint(),
                "tool",
                ClassMemoryDraft::new("cargo test works"),
            )
            .unwrap()
            .id;
        let exp = sys.export().unwrap();
        assert!(exp.records.iter().any(|r| r.id == id));
        assert!(sys.forget(&id).unwrap());
        assert!(!sys.export().unwrap().records.iter().any(|r| r.id == id));
        assert!(sys.list_class(MemoryClass::Procedural).unwrap().is_empty());
    }

    #[test]
    fn property_eight_scopes_closed_vocabulary() {
        assert_eq!(PersonalScope::all().len(), 8);
        for s in PersonalScope::all() {
            assert_eq!(PersonalScope::parse(s.as_str()), Some(s));
        }
        assert!(PersonalScope::parse("ambient_global").is_none());
    }

    /// The exact bug class that shipped once: user.db already exists without the
    /// control columns, CREATE TABLE IF NOT EXISTS is a no-op, and reads fail
    /// with "no such column". Migration must be PRAGMA table_info + ALTER.
    #[test]
    fn property_user_db_legacy_schema_migrates_idempotently() {
        let dir = tempfile::tempdir().unwrap();
        let wdb = dir.path().join("ws.db");
        let udb = dir.path().join("user_legacy.db");

        // Simulate a user.db written by the first six-class lane (no scope/pin).
        {
            let conn = rusqlite::Connection::open(&udb).unwrap();
            conn.execute_batch(
                r#"
                CREATE TABLE mem_user (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    importance REAL NOT NULL,
                    workspace_id TEXT,
                    session_id TEXT,
                    provenance_json TEXT NOT NULL,
                    evidence_tier TEXT
                );
                INSERT INTO mem_user
                  (id, text, importance, workspace_id, session_id, provenance_json, evidence_tier)
                VALUES
                  ('user_legacy_1', 'prefer terse', 0.9, NULL, NULL,
                   '{"writer":"user","written_at_ms":1,"turn_id":null,"run_id":null,"evidence":[],"authority":"user_explicit"}',
                   NULL);
                "#,
            )
            .unwrap();
            // Confirm pre-migration shape.
            let mut stmt = conn.prepare("PRAGMA table_info(mem_user)").unwrap();
            let cols: Vec<String> = stmt
                .query_map([], |r| r.get::<_, String>(1))
                .unwrap()
                .map(|r| r.unwrap())
                .collect();
            assert!(!cols.iter().any(|c| c == "scope"));
            assert!(!cols.iter().any(|c| c == "pinned"));
        }

        // Open must migrate, not fail on SELECT scope.
        let sys = ClassedMemorySystem::open("ws-mig", &wdb, &udb).unwrap();
        let listed = sys.list_class(MemoryClass::User).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, "user_legacy_1");
        assert_eq!(listed[0].text, "prefer terse");
        assert_eq!(listed[0].scope, PersonalScope::Global); // default after migrate
        assert!(!listed[0].pinned);

        // Pin / forget work on the migrated row.
        sys.pin("user_legacy_1", true).unwrap();
        assert!(sys.get("user_legacy_1").unwrap().unwrap().pinned);

        // Idempotent: reopen does not fail or duplicate columns.
        let sys2 = ClassedMemorySystem::open("ws-mig", &wdb, &udb).unwrap();
        assert_eq!(sys2.count(MemoryClass::User).unwrap(), 1);
        assert!(sys2.get("user_legacy_1").unwrap().unwrap().pinned);
    }

    /// Forget is real deletion of the record AND of dangling edges (promotions,
    /// supersedes pointers). Export must not retain residual audit of forgotten ids.
    #[test]
    fn property_forget_clears_dangling_edges_and_export() {
        let sys = system();
        let old = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("old preference"),
            )
            .unwrap();
        let corrected = sys
            .correct_user(
                &UserWriteCap::mint(),
                &old.id,
                "user",
                "new preference",
            )
            .unwrap();
        assert_eq!(corrected.supersedes.as_deref(), Some(old.id.as_str()));

        // Promote a workspace episodic so a promotion row exists for the id.
        let epi = sys
            .write_episodic(
                &EpisodicWriteCap::mint(),
                "event",
                ClassMemoryDraft::new("connector blob")
                    .with_scope(PersonalScope::Connector)
                    .with_session("s-edge"),
            )
            .unwrap();
        sys.set_scope(&epi.id, PersonalScope::Global, "user")
            .unwrap();
        assert_eq!(sys.list_promotions().unwrap().len(), 1);

        // Forget the superseded old user record: supersedes edge on corrected clears.
        assert!(sys.forget(&old.id).unwrap());
        let still = sys.get(&corrected.id).unwrap().unwrap();
        assert!(
            still.supersedes.is_none(),
            "supersedes edge must not dangle after forget"
        );

        // Forget the promoted episodic: promotion row must go too.
        assert!(sys.forget(&epi.id).unwrap());
        assert!(
            sys.list_promotions().unwrap().is_empty(),
            "promotion rows for forgotten ids are dangling edges"
        );

        let exp = sys.export().unwrap();
        assert!(!exp.records.iter().any(|r| r.id == old.id));
        assert!(!exp.records.iter().any(|r| r.id == epi.id));
        assert!(!exp.promotions.iter().any(|p| p.record_id == epi.id));
        // Surviving corrected row remains, without a pointer at the forgotten id.
        assert!(exp.records.iter().any(|r| r.id == corrected.id));
        let json = serde_json::to_string(&exp).unwrap();
        assert!(!json.contains(&old.id));
        assert!(!json.contains("connector blob"));
    }

    /// Export is portable record data, not a capability grant. There is no
    /// import path that rehydrates forgotten records or mints live authority.
    #[test]
    fn property_export_carries_no_capability_and_no_reimport_path() {
        let sys = system();
        let id = sys
            .write_user(
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("secret preference xyz"),
            )
            .unwrap()
            .id;
        let exp = sys.export().unwrap();
        let json = serde_json::to_string(&exp).unwrap();
        // No tool/connector grant vocabulary in the export schema.
        assert!(!json.contains("\"tools\""));
        assert!(!json.contains("\"connectors\""));
        assert!(!json.contains("JobCapability"));
        assert!(!json.contains("SurfaceCapability"));
        assert_eq!(exp.schema, "hide.you.memory_export.v1");

        // Forget removes content from subsequent export (no outliving copy).
        assert!(sys.forget(&id).unwrap());
        let after = sys.export().unwrap();
        let after_json = serde_json::to_string(&after).unwrap();
        assert!(!after_json.contains("secret preference xyz"));
        assert!(!after.records.iter().any(|r| r.id == id));

        // ClassedMemorySystem has no import_export / apply_export API that would
        // reintroduce the forgotten row. Round-tripping the pre-forget JSON into
        // MemoryExport is data only — it does not write itself back.
        let stale: MemoryExport = serde_json::from_str(&json).unwrap();
        assert!(stale.records.iter().any(|r| r.id == id));
        assert!(
            sys.get(&id).unwrap().is_none(),
            "deserializing an old export must not rehydrate the store"
        );
        assert_eq!(sys.count(MemoryClass::User).unwrap(), 0);
    }
}

/// Shared handle type used by backend services and context sources.
pub type DynClassedMemory = Arc<ClassedMemorySystem>;
