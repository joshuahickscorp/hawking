//! Errors surfaced by the ACP boundary.

use thiserror::Error;

/// Something went wrong mapping between ACP and the HIDE schema authority, or
/// while negotiating the handshake.
#[derive(Debug, Error)]
pub enum AcpError {
    /// The client offered a protocol version the agent cannot speak at all.
    #[error("unsupported ACP protocol version: client offered {offered}, agent speaks 1..={agent_max}")]
    UnsupportedVersion { offered: u16, agent_max: u16 },

    /// An ACP session id has no HIDE session/thread binding. The client must run
    /// `session/new` (or `session/load`) before prompting.
    #[error("unknown ACP session: {0}")]
    UnknownSession(String),

    /// A HIDE item kind has no honest ACP projection (for example an internal
    /// coordination item that the editor surface does not model).
    #[error("no ACP projection for HIDE item kind: {0}")]
    Unprojectable(String),

    /// An ACP prompt carried nothing a HIDE turn could act on.
    #[error("empty ACP prompt: no text or resource content blocks")]
    EmptyPrompt,

    #[error("serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    /// A transport read or write failed (for the line/stdio transport).
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience alias for the crate.
pub type Result<T> = std::result::Result<T, AcpError>;
