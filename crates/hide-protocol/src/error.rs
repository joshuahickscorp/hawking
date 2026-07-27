//! Errors surfaced by the protocol layer.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("unknown method: {0}")]
    UnknownMethod(String),

    #[error("no compatible protocol version; client offered {offered:?}, server supports {supported:?}")]
    VersionMismatch {
        offered: Vec<String>,
        supported: Vec<String>,
    },

    #[error("cannot map hide-core intent onto the protocol: {0}")]
    UnmappableIntent(String),

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;
