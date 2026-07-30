//! Connector error types. Fail closed on revocation, missing capability, and
//! unauthorized memory promotion.

use thiserror::Error;

use crate::connector_abi::abi::FamilyId;
use crate::connector_abi::account::AccountId;
use crate::connector_abi::memory::MemoryScope;

/// Result alias for connector operations.
pub type Result<T> = std::result::Result<T, ConnectorError>;

/// Errors produced by the connector ABI, registry, and implementations.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum ConnectorError {
    #[error("family `{0}` is declared but not implemented; construction refused")]
    DeclaredNotConstructible(FamilyId),

    #[error("family `{0}` is not registered")]
    UnknownFamily(FamilyId),

    #[error("family `{0}` does not declare write capability; write is a type boundary refusal")]
    WriteNotDeclared(FamilyId),

    #[error("account `{0}` is revoked; operation failed closed")]
    AccountRevoked(AccountId),

    #[error("account handle generation mismatch (revoked or stale); fail closed")]
    StaleHandle,

    #[error("account handle family mismatch: handle is `{handle}`, connector is `{connector}`")]
    AccountFamilyMismatch {
        handle: FamilyId,
        connector: FamilyId,
    },

    #[error("no ambient credential lookup: connector must receive an explicit AccountHandle")]
    AmbientCredentialForbidden,

    #[error("account `{0}` not found in account store")]
    AccountNotFound(AccountId),

    #[error("credential bound to account `{0}` is not readable by family `{1}`")]
    CredentialIsolation(AccountId, FamilyId),

    #[error("write refused: no permission grant for effect `{0}`")]
    WritePermissionDenied(String),

    #[error("write refused: silent execution forbidden; a WriteReceipt is required")]
    WriteReceiptRequired,

    #[error("write receipt `{0}` is invalid or already consumed")]
    InvalidWriteReceipt(String),

    #[error("memory write to scope `{target}` refused from connector scope; promotion required")]
    SilentMemoryPromotion { target: MemoryScope },

    #[error("promotion to user memory requires UserMemoryPromotionCap")]
    UserPromotionCapRequired,

    #[error("io: {0}")]
    Io(String),

    #[error("parse: {0}")]
    Parse(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("rate limit: {0}")]
    RateLimit(String),

    #[error("invalid request: {0}")]
    InvalidRequest(String),
}

impl From<std::io::Error> for ConnectorError {
    fn from(e: std::io::Error) -> Self {
        ConnectorError::Io(e.to_string())
    }
}
