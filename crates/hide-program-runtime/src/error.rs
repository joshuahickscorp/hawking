//! Typed errors raised by the runtime.
//!
//! Every failure mode is a distinct, matchable variant. The limiter raises
//! [`RuntimeError::LimitExceeded`] with a [`LimitKind`] so a host can tell which
//! budget tripped without string-matching.

use thiserror::Error;

/// The kind of resource budget that was exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LimitKind {
    /// Too many AST nodes evaluated.
    Instruction,
    /// The virtual wall clock advanced past the budget.
    WallTime,
    /// A produced value exceeded the peak-memory budget.
    Memory,
    /// The serialized program result exceeded the output budget.
    OutputBytes,
    /// Too many host handles were called.
    ToolCall,
    /// A map operator requested more concurrency than allowed.
    Concurrency,
    /// A spilled artifact exceeded the per-artifact byte budget.
    ArtifactByte,
    /// Evaluation nested deeper than allowed.
    Recursion,
}

impl LimitKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            LimitKind::Instruction => "instruction",
            LimitKind::WallTime => "wall_time",
            LimitKind::Memory => "memory",
            LimitKind::OutputBytes => "output_bytes",
            LimitKind::ToolCall => "tool_call",
            LimitKind::Concurrency => "concurrency",
            LimitKind::ArtifactByte => "artifact_byte",
            LimitKind::Recursion => "recursion",
        }
    }
}

/// Failure surfaced by a host handle. Kept separate from [`RuntimeError`] so a
/// host can report a read failure (missing file, denied scope) without pretending
/// it is a runtime bug. A handle can mark a failure retryable so
/// `retry_with_policy` will re-attempt it.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("handle {handle} failed: {message}")]
pub struct HandleError {
    pub handle: String,
    pub message: String,
    pub retryable: bool,
}

impl HandleError {
    pub fn new(handle: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            handle: handle.into(),
            message: message.into(),
            retryable: false,
        }
    }

    pub fn retryable(handle: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            handle: handle.into(),
            message: message.into(),
            retryable: true,
        }
    }
}

/// Anything that can go wrong while running a program.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RuntimeError {
    /// A resource budget was exhausted. This is the sandbox doing its job, not a
    /// program bug.
    #[error("{kind} limit exceeded (limit {limit}, needed {needed})", kind = kind.as_str())]
    LimitExceeded {
        kind: LimitKind,
        limit: u64,
        needed: u64,
    },

    /// The program called a handle it was not granted. There is no way to
    /// escalate: the host decides grants, the runtime only enforces them.
    #[error("handle {0} is not granted to this program")]
    HandleNotGranted(String),

    /// A referenced variable was not bound in scope.
    #[error("unbound variable: {0}")]
    UnboundVariable(String),

    /// An operation received a value of the wrong shape.
    #[error("type error: {0}")]
    Type(String),

    /// A value failed `schema_validate`.
    #[error("schema validation failed: {0}")]
    Schema(String),

    /// A host handle returned an error and it was not (or no longer) retryable.
    #[error(transparent)]
    Handle(#[from] HandleError),
}

impl RuntimeError {
    pub fn limit(kind: LimitKind, limit: u64, needed: u64) -> Self {
        RuntimeError::LimitExceeded { kind, limit, needed }
    }

    /// The [`LimitKind`] this error carries, if it is a limit error.
    pub fn limit_kind(&self) -> Option<LimitKind> {
        match self {
            RuntimeError::LimitExceeded { kind, .. } => Some(*kind),
            _ => None,
        }
    }
}

/// Convenience result alias for runtime operations.
pub type Result<T> = std::result::Result<T, RuntimeError>;
