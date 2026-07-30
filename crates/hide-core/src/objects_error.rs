//! Typed errors for the YOU object store.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, ObjectError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ObjectError {
    #[error("object not found: {0}")]
    NotFound(String),

    #[error("reference not found: {0}")]
    RefNotFound(String),

    #[error("permission denied: {reason}")]
    PermissionDenied { reason: String },

    #[error("retention expired or not readable: {reason}")]
    RetentionDenied { reason: String },

    #[error("storage budget exceeded: need {need} bytes, available {available} under {budget}")]
    BudgetExceeded {
        need: u64,
        available: u64,
        budget: String,
    },

    #[error("object too large: {size} bytes exceeds max_object_bytes {max}")]
    ObjectTooLarge { size: u64, max: u64 },

    #[error("queue job failed visibly (not dropped): job={job_id} stage={stage}: {detail}")]
    StageFailed {
        job_id: String,
        stage: String,
        detail: String,
    },

    #[error("stage not ready: {stage} (status={status})")]
    StageNotReady { stage: String, status: String },

    #[error("raw bytes are not reachable from the context-compile path")]
    RawBytesForbidden,

    #[error("derivative not available: {kind} for {content_hash}")]
    DerivativeMissing { kind: String, content_hash: String },

    #[error("content address mismatch: expected {expected}, actual {actual}")]
    ContentAddressMismatch { expected: String, actual: String },

    #[error("io: {0}")]
    Io(String),

    #[error("invalid argument: {0}")]
    Invalid(String),

    #[error("queue empty")]
    QueueEmpty,
}

impl From<std::io::Error> for ObjectError {
    fn from(e: std::io::Error) -> Self {
        ObjectError::Io(e.to_string())
    }
}
