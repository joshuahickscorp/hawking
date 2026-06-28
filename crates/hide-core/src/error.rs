use thiserror::Error;

#[derive(Debug, Error)]
pub enum HideError {
    #[error("{0}")]
    Message(String),

    #[error("configuration error: {0}")]
    Config(String),

    #[error("policy denied: {0}")]
    PolicyDenied(String),

    #[error("capability missing: {0}")]
    CapabilityMissing(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("tool error: {0}")]
    Tool(String),

    #[error("runtime unavailable: {0}")]
    RuntimeUnavailable(String),

    #[error("storage error: {0}")]
    Storage(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Serde(#[from] serde_json::Error),
}

impl HideError {
    pub fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

pub type Result<T> = std::result::Result<T, HideError>;
