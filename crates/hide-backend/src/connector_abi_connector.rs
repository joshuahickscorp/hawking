//! Connector trait surface with a read/write type boundary.
//!
//! - Every live connector implements [`ConnectorRead`].
//! - Only connectors that declare write implement [`ConnectorWrite`].
//! - A function that needs to write takes `T: ConnectorWrite`, so a read-only
//!   connector cannot be asked to write at compile time.
//! - Declared (non-implemented) families have no type that implements either
//!   trait; construction is refused by the registry.

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::{ConnectorAbi, FamilyId, WriteCapability};
use crate::connector_abi::account::{AccountHandle, AccountStore, InFlightGuard};
use crate::connector_abi::error::{ConnectorError, Result};

/// A listed or fetched object from a connector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectorObject {
    pub id: String,
    pub object_type: String,
    pub title: String,
    /// Optional body / summary text.
    pub content: Option<String>,
    pub metadata: BTreeMapStr,
}

/// Simple ordered string map for metadata (serde-friendly).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BTreeMapStr(pub std::collections::BTreeMap<String, String>);

impl BTreeMapStr {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, k: impl Into<String>, v: impl Into<String>) {
        self.0.insert(k.into(), v.into());
    }
}

/// Read request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadRequest {
    /// Object id, path, feed item id, etc.
    pub locator: String,
}

/// List request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ListRequest {
    /// Optional prefix / folder / query.
    pub prefix: Option<String>,
    pub limit: usize,
}

impl Default for ListRequest {
    fn default() -> Self {
        Self {
            prefix: None,
            limit: 100,
        }
    }
}

/// Shared identity every connector exposes.
pub trait Connector: Send + Sync {
    fn family_id(&self) -> &FamilyId;
    fn abi(&self) -> &ConnectorAbi;
}

/// Read surface. All implemented connectors implement this.
pub trait ConnectorRead: Connector {
    fn list(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ListRequest,
    ) -> Result<Vec<ConnectorObject>>;

    fn fetch(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        request: &ReadRequest,
    ) -> Result<ConnectorObject>;
}

/// Write surface. A type boundary: only connectors that declare write implement
/// this trait. Read-only connectors (local_folder, rss) do not, so they cannot
/// be passed to functions that require [`ConnectorWrite`].
///
/// Implementations prepare proposals only; execution goes through the
/// permission gate and a write receipt (see [`crate::connector_abi::effects`]).
pub trait ConnectorWrite: ConnectorRead {
    /// The declared write capability. Must be writable.
    fn write_capability(&self) -> &WriteCapability;

    /// Prepare a write proposal. Does not execute. Requires a live handle.
    fn prepare_write(
        &self,
        store: &AccountStore,
        handle: &AccountHandle,
        kind: crate::connector_abi::effects::WriteKind,
        target: impl Into<String>,
        payload: impl Into<String>,
        summary: impl Into<String>,
    ) -> Result<crate::connector_abi::effects::ConnectorWriteProposal> {
        if !self.write_capability().is_writable() {
            return Err(ConnectorError::WriteNotDeclared(self.family_id().clone()));
        }
        let guard = InFlightGuard::begin(store, handle, self.family_id())?;
        let kind = kind;
        let effect = match kind {
            crate::connector_abi::effects::WriteKind::Delete => {
                crate::connector_abi::abi::EffectClass::Delete
            }
            _ => crate::connector_abi::abi::EffectClass::Write,
        };
        let proposal = crate::connector_abi::effects::ConnectorWriteProposal {
            family_id: self.family_id().clone(),
            account_id: handle.account_id().clone(),
            kind,
            effect,
            summary: summary.into(),
            target: target.into(),
            payload: payload.into(),
        };
        guard.complete(store)?;
        Ok(proposal)
    }
}

/// Marker used by the registry: a family that is only declared has no
/// constructible type. Attempting to construct yields
/// [`ConnectorError::DeclaredNotConstructible`].
pub struct DeclaredConnector;

impl DeclaredConnector {
    pub fn try_construct(family_id: FamilyId) -> Result<Self> {
        Err(ConnectorError::DeclaredNotConstructible(family_id))
    }
}
