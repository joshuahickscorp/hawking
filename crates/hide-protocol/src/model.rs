//! The semantic object model (Bible sec 14).
//!
//! These are the durable nouns of the HIDE Agent Server: the containers
//! (Workspace, Repository, Environment), the conversation spine (Session ->
//! Thread -> Turn -> Item), and the standalone objects a turn refers to
//! (Artifact, Checkpoint, StateCapsuleRef, Tool, Oracle). Goals, Plans, and
//! Agents live in [`crate::plan`]; Items and their kinds live in
//! [`crate::item`], so this file holds the object graph they hang from.
//!
//! Every type derives serde AND schemars from ONE definition, so the JSON
//! Schema is generated, never hand-written and never allowed to drift from the
//! Rust shape. This crate is model-free: the objects describe state, they do
//! not run anything.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::ids::{
    CheckpointId, EnvironmentId, OracleId, RepositoryId, SessionId, StateCapsuleId, ThreadId,
    ToolId, TurnId, WorkspaceId,
};
use crate::item::Item;
use crate::plan::Effect;

// -- shared enums ----------------------------------------------------------

/// Version control backing a repository.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum VcsKind {
    Git,
    None,
    Other,
}

/// The kind of execution environment a session runs against.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentKind {
    Local,
    Container,
    Remote,
    Sandbox,
}

/// Lifecycle state of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Idle,
    Closed,
}

/// Who authored a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    User,
    Agent,
    System,
    Tool,
}

/// Lifecycle state of a turn. `steer`/`interrupt`/`pause`/`resume` (sec 15)
/// move a turn between these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TurnStatus {
    Pending,
    Running,
    Paused,
    Completed,
    Interrupted,
    Failed,
}

/// Terminal outcome of a turn, agent, or completion item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    Success,
    Partial,
    Failed,
    Cancelled,
}

/// Risk banding on an action that may need approval. Mirrors the concept in
/// `hide_core::types::RiskLevel` but is defined locally so the schema authority
/// stays self-contained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Risk {
    Trivial,
    Low,
    Medium,
    High,
    Critical,
}

/// The kind of artifact a step or turn produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    File,
    Patch,
    Report,
    Binary,
    Log,
    Other,
}

/// What an oracle checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum OracleKind {
    Test,
    Build,
    Lint,
    TypeCheck,
    Custom,
}

// -- containers ------------------------------------------------------------

/// The outermost container. Binds the repositories a session may touch, the
/// environments it may run in, and the sessions themselves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Workspace {
    pub id: WorkspaceId,
    pub name: String,
    pub repositories: Vec<Repository>,
    pub environments: Vec<Environment>,
    #[serde(default)]
    pub sessions: Vec<SessionId>,
    #[serde(default)]
    pub default_environment: Option<EnvironmentId>,
    pub created_ms: u64,
}

/// A source repository inside a workspace.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Repository {
    pub id: RepositoryId,
    pub workspace: WorkspaceId,
    pub name: String,
    pub root_path: String,
    pub vcs: VcsKind,
    #[serde(default)]
    pub remote_url: Option<String>,
    #[serde(default)]
    pub head_ref: Option<String>,
}

/// An execution environment a session binds to. `capabilities` are free-form
/// tags (for example `"shell"`, `"network"`) the host advertises.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Environment {
    pub id: EnvironmentId,
    pub workspace: WorkspaceId,
    pub name: String,
    pub kind: EnvironmentKind,
    pub working_dir: String,
    #[serde(default)]
    pub platform: Option<String>,
    #[serde(default)]
    pub capabilities: Vec<String>,
}

/// A session: one working context over a workspace. Threads (including forks)
/// hang off it; the session references them by id so a session stays a light
/// header while threads carry the turn history.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Session {
    pub id: SessionId,
    pub workspace: WorkspaceId,
    #[serde(default)]
    pub repository: Option<RepositoryId>,
    #[serde(default)]
    pub environment: Option<EnvironmentId>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub threads: Vec<ThreadId>,
    pub status: SessionStatus,
    pub created_ms: u64,
}

/// A thread: an ordered line of turns within a session. Forking a thread
/// (sec 15 `thread/fork`, `thread/fork_ephemeral`) records the parent and the
/// turn it branched from, so lineage is always recoverable. An ephemeral fork
/// is a scratch branch whose only durable return is a merge summary
/// (`thread/merge_summary`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Thread {
    pub id: ThreadId,
    pub session: SessionId,
    #[serde(default)]
    pub parent: Option<ThreadId>,
    #[serde(default)]
    pub forked_at_turn: Option<TurnId>,
    #[serde(default)]
    pub ephemeral: bool,
    #[serde(default)]
    pub title: Option<String>,
    pub turns: Vec<Turn>,
    pub created_ms: u64,
}

/// A turn: one authored contribution to a thread, carrying an ordered list of
/// items. `parent_turn` records causal provenance (the "why" spine).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Turn {
    pub id: TurnId,
    pub thread: ThreadId,
    pub role: TurnRole,
    pub status: TurnStatus,
    pub items: Vec<Item>,
    #[serde(default)]
    pub parent_turn: Option<TurnId>,
    pub created_ms: u64,
}

// -- standalone objects a turn refers to -----------------------------------

/// A produced artifact: a file, patch, report, or other output. `produced_by`
/// links back to the plan step (Bible sec 14) that generated it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Artifact {
    pub id: crate::ids::ArtifactId,
    pub name: String,
    pub kind: ArtifactKind,
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub digest: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub produced_by: Option<crate::ids::StepId>,
    pub created_ms: u64,
}

/// A checkpoint: a named, restorable boundary in a thread. It may bind a state
/// capsule (the runtime bytes) and/or a VCS ref (the source-tree position).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Checkpoint {
    pub id: CheckpointId,
    #[serde(default)]
    pub session: Option<SessionId>,
    #[serde(default)]
    pub thread: Option<ThreadId>,
    #[serde(default)]
    pub at_turn: Option<TurnId>,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub capsule: Option<StateCapsuleRef>,
    #[serde(default)]
    pub vcs_ref: Option<String>,
    pub created_ms: u64,
}

/// A pointer to a state capsule held by `hide-state`. This is only a reference:
/// it names the capsule and pins its digest and identity binding so a reader
/// can locate and verify the bytes.
///
/// DEFERRED_MODEL_REQUIRED: the live production of capsule bytes from a running
/// engine, and their rebind into a runtime, is not implemented here and cannot
/// be claimed. `hide-state` defines the byte schema; a model-bearing runtime
/// fills it. This struct only carries the address and the integrity pins.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StateCapsuleRef {
    pub id: StateCapsuleId,
    /// The capsule id as known to the `hide-state` store.
    pub capsule_id: String,
    /// Integrity digest of the capsule payload (verified by `hide-state`).
    pub digest: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub size_bytes: Option<u64>,
    pub created_ms: u64,
}

/// A tool the agent may invoke. `input_schema`/`output_schema` are JSON Schema
/// documents (carried as opaque JSON here). `effects` declares up front what
/// the tool may do, so a scope check can run before it is ever called.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Tool {
    pub id: ToolId,
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub effects: Vec<Effect>,
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: Option<serde_json::Value>,
    #[serde(default)]
    pub requires_approval: bool,
}

/// An oracle: a deterministic acceptance check a plan step (or verification
/// request) is graded against. Its `command` is the check to run; `acceptance`
/// states the pass condition in prose.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Oracle {
    pub id: OracleId,
    pub name: String,
    pub kind: OracleKind,
    #[serde(default)]
    pub command: Option<Vec<String>>,
    pub acceptance: String,
    /// Whether the check is reproducible run-to-run. HIDE oracles are expected
    /// to be deterministic (Bible law 17); a non-deterministic oracle is a
    /// smell the model should surface, not hide.
    #[serde(default = "default_true")]
    pub deterministic: bool,
}

fn default_true() -> bool {
    true
}
