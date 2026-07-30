//! Error types for hide-you.

use thiserror::Error;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum YouError {
    #[error("policy denied: {0}")]
    PolicyDenied(String),

    #[error("capability missing: {0}")]
    CapabilityMissing(String),

    #[error("invalid handoff: {0}")]
    InvalidHandoff(String),

    #[error("budget exhausted: {0}")]
    BudgetExhausted(String),

    #[error("invalid state: {0}")]
    InvalidState(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("promotion refused: {0}")]
    PromotionRefused(String),

    #[error("{0}")]
    Message(String),
}

pub type Result<T> = std::result::Result<T, YouError>;
