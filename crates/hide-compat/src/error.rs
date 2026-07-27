use thiserror::Error;

/// Errors surfaced by the compatibility readers. Parsing is intentionally
/// lenient (a malformed optional file is skipped, not fatal); these variants
/// cover the cases where the caller genuinely cannot proceed.
#[derive(Debug, Error)]
pub enum CompatError {
    #[error("io error at {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("json parse error in {path}: {source}")]
    Json {
        path: String,
        #[source]
        source: serde_json::Error,
    },

    #[error("invalid glob {glob:?}: {source}")]
    Glob {
        glob: String,
        #[source]
        source: globset::Error,
    },

    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, CompatError>;

impl CompatError {
    pub fn msg(m: impl Into<String>) -> Self {
        CompatError::Message(m.into())
    }
}
