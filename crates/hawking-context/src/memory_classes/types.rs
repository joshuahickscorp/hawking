//! Memory class types, write-authority caps, drafts, and compile budgets.
//!
//! BC-CONTEXT_OS-001 / write-authority type boundary (BC-SECURITY caps).

use hide_core::ids::now_ms;
use serde::{Deserialize, Serialize};

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
    pub(super) fn stamped(
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
