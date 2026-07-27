//! Typed errors for capsule schemas, serialization, compatibility, and stores.
//!
//! Every rejection is a distinct variant so callers branch on the reason
//! programmatically rather than parsing a message. In particular
//! [`IncompatibleReason`] names exactly which identity field disagreed, so a
//! loader never has to scrape a string to learn why a capsule cannot bind to a
//! live runtime.

use thiserror::Error;

/// The precise reason a sealed capsule cannot bind to a live runtime identity.
///
/// Each variant carries the capsule-side value and the live-side value for the
/// one field that disagreed, so the caller can report or reconcile without
/// re-deriving the comparison.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum IncompatibleReason {
    #[error("model weights id differs: capsule {capsule:?}, live {live:?}")]
    ModelWeights { capsule: String, live: String },

    #[error("architecture id differs: capsule {capsule:?}, live {live:?}")]
    Arch { capsule: String, live: String },

    #[error("tokenizer id differs: capsule {capsule:?}, live {live:?}")]
    Tokenizer { capsule: String, live: String },

    #[error("prompt ABI version differs: capsule {capsule:?}, live {live:?}")]
    PromptAbi { capsule: String, live: String },

    #[error("tool registry id differs: capsule {capsule:?}, live {live:?}")]
    ToolRegistry { capsule: String, live: String },

    #[error("engine build id differs: capsule {capsule:?}, live {live:?}")]
    EngineBuild { capsule: String, live: String },

    #[error("security domain differs: capsule {capsule:?}, live {live:?}")]
    SecurityDomain { capsule: String, live: String },
}

/// Errors surfaced by capsule serialization, integrity, and the store impls.
#[derive(Debug, Error)]
pub enum CapsuleError {
    #[error("not a capsule byte stream: magic header did not match")]
    BadMagic,

    #[error("unsupported capsule format version {found} (this build reads {supported})")]
    UnsupportedVersion { found: u16, supported: u16 },

    #[error("capsule byte stream truncated: {detail}")]
    Truncated { detail: &'static str },

    #[error("declared payload length {declared} does not match actual {actual}")]
    LengthMismatch { declared: u64, actual: u64 },

    #[error("integrity check failed: the header digest does not match the payload")]
    IntegrityMismatch,

    #[error("content address mismatch: expected {expected:?}, computed {actual:?}")]
    ContentAddressMismatch { expected: String, actual: String },

    #[error("stored object is corrupt: {detail}")]
    Corrupt { detail: String },

    #[error("no capsule with id {0:?}")]
    NotFound(String),

    #[error("capsule metadata is not valid json: {0}")]
    Meta(#[from] serde_json::Error),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, CapsuleError>;
