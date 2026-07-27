//! hide-connectors: HIDE YOU connector ABI and registry.
//!
//! YOU is the private personal-AI surface alongside CHAT and IDE. It reaches
//! the user's world through connectors — the highest-risk subsystem, because a
//! personal assistant touches real accounts.
//!
//! # Safety properties (the point)
//!
//! 1. **Default read-only, least privilege.** A connector that does not declare
//!    write has no [`ConnectorWrite`] impl; write is a type boundary.
//! 2. **No ambient credentials.** Every call takes an explicit [`AccountHandle`].
//!    There is no global credential lookup; one account cannot read another's.
//! 3. **Every write is an effect** that goes through [`PermissionGate`] and
//!    leaves a [`WriteReceipt`]. Silent execution is refused.
//! 4. **Connector data never silently enters global memory.** Ingest lands in
//!    connector scope; promotion to user/semantic requires an explicit cap.
//! 5. **Revocation is real.** Revoking an account invalidates handles; in-flight
//!    operations re-check and fail closed.
//!
//! # Implementation status
//!
//! Only `local_folder` and `rss` are implemented (fixture / local filesystem,
//! no network). Every other family is fully declared in the ABI and refuses
//! construction.
//!
//! ```
//! use hide_connectors::{AccountStore, ConnectorRegistry, CredentialMaterial, ListRequest};
//! use hide_connectors::abi::FamilyId;
//!
//! let reg = ConnectorRegistry::builtin();
//! assert!(reg.construct("github").is_err()); // declared, not constructible
//!
//! let folder = reg.construct("local_folder").unwrap();
//! let mut accounts = AccountStore::new();
//! // In real tests the root is a tempdir; doctest only shows the shape.
//! let _ = (folder, accounts, FamilyId::new("local_folder"), ListRequest::default(), CredentialMaterial { material: "/tmp".into() });
//! ```

pub mod abi;
pub mod account;
pub mod connector;
pub mod effects;
pub mod error;
pub mod families;
pub mod impls;
pub mod memory;
pub mod registry;

pub use abi::{
    AuditPolicy, AuthMethod, ChangeTransport, ConnectorAbi, ConnectorScope, EffectClass, FamilyId,
    ImplementationStatus, ObjectType, OfflineCache, RateLimit, ReadCapability, RevocationPolicy,
    SyncMode, WriteCapability,
};
pub use account::{
    AccountHandle, AccountId, AccountStore, CredentialMaterial, InFlightGuard,
};
pub use connector::{
    BTreeMapStr, Connector, ConnectorObject, ConnectorRead, ConnectorWrite, DeclaredConnector,
    ListRequest, ReadRequest,
};
pub use effects::{
    execute_with_receipt, execute_without_receipt, ConnectorWriteProposal, PermissionDecision,
    PermissionGate, PermissionPolicy, WriteKind, WriteReceipt, WriteResult,
};
pub use error::{ConnectorError, Result};
pub use impls::{LocalFolderConnector, RssConnector};
pub use memory::{
    ConnectorIngestCap, ConnectorMemoryStore, MemoryRecord, MemoryScope, SemanticPromotionCap,
    UserMemoryPromotionCap,
};
pub use registry::{ConnectorRegistry, LiveConnector, RegistryDocument};
