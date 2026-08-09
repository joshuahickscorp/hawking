use thiserror::Error;

pub type Result<T> = std::result::Result<T, PerceptionError>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum PerceptionError {
    #[error("document not found: {0}")]
    DocumentNotFound(String),

    #[error("page out of range: page {page} (document has {pages} pages)")]
    PageOutOfRange { page: u32, pages: u32 },

    #[error("region not found: {0}")]
    RegionNotFound(String),

    #[error(
        "stage skipped: {stage} requires capability {capability} (not available in this backend)"
    )]
    CapabilityMissing {
        stage: &'static str,
        capability: &'static str,
    },

    #[error("budget exhausted at stage {stage}: remaining cost units {remaining}")]
    BudgetExhausted { stage: &'static str, remaining: u64 },

    #[error("invalid request: {0}")]
    InvalidRequest(String),

    #[error("backend deferred: {0}")]
    /// Real OCR/vision/PDF work — not implemented in this scaffold.
    Deferred(&'static str),
}
