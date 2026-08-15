use thiserror::Error;

pub type Result<T> = std::result::Result<T, CommsError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum CommsError {
    #[error("invalid packet: {0}")]
    Invalid(String),

    #[error("unsealed latent transfer refused: {0}")]
    UnsealedLatent(String),

    #[error("latent experimental gate closed: {0}")]
    LatentGateClosed(String),

    #[error("cross-model latent transfer refused: sender={sender} receiver={receiver}")]
    CrossModelLatent { sender: String, receiver: String },

    #[error("latent packet expired at {expiry_unix_ms} (now {now_unix_ms})")]
    Expired {
        expiry_unix_ms: u64,
        now_unix_ms: u64,
    },

    #[error("capability scope denied: required {required}, have {have}")]
    CapabilityDenied { required: String, have: String },

    #[error("payload hash mismatch: expected {expected}, got {got}")]
    HashMismatch { expected: String, got: String },

    #[error("deferred: {0}")]
    /// Live KV / hidden-state transfer — not implemented in this scaffold.
    Deferred(&'static str),
}
