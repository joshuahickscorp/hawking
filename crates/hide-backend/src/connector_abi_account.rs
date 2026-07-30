//! Account handles: explicit, non-ambient credentials.
//!
//! There is no global credential lookup. A connector receives an
//! [`AccountHandle`] that the account store minted for one family and one
//! account. Handles carry a generation; revoking an account bumps the
//! generation so every outstanding handle fails closed.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

use crate::connector_abi::abi::FamilyId;
use crate::connector_abi::error::{ConnectorError, Result};

/// Stable account identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AccountId(pub String);

impl AccountId {
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AccountId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Opaque credential material. Never ambient: only reachable through a live
/// [`AccountHandle`] that the store validates.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialMaterial {
    /// Opaque token / path / fixture key. Not a global secret lookup key.
    pub material: String,
}

/// An explicit account handle passed into every connector call.
///
/// Constructed only by [`AccountStore::mint_handle`]. Connectors do not look up
/// credentials themselves.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountHandle {
    pub account_id: AccountId,
    pub family_id: FamilyId,
    /// Generation at mint time; must match the store's current generation.
    pub generation: u64,
    /// Opaque credential token bound to this account only.
    pub(crate) credential: CredentialMaterial,
}

impl AccountHandle {
    pub fn account_id(&self) -> &AccountId {
        &self.account_id
    }
    pub fn family_id(&self) -> &FamilyId {
        &self.family_id
    }
    /// Credential material for *this* account only. Still requires the store
    /// to validate the handle is live before use.
    pub fn credential_material(&self) -> &str {
        &self.credential.material
    }
}

struct AccountRecord {
    family_id: FamilyId,
    credential: CredentialMaterial,
    generation: u64,
    revoked: bool,
    label: String,
}

/// The sole place credentials live. Connectors never reach into this store
/// globally; they only use the handle they were given, re-validated here.
#[derive(Default)]
pub struct AccountStore {
    accounts: BTreeMap<AccountId, AccountRecord>,
    /// Monotonic id for auto-generated account ids.
    next: AtomicU64,
}

impl AccountStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an account for a family with explicit credential material.
    /// Returns the account id. Does not mint a handle — call [`mint_handle`].
    pub fn register(
        &mut self,
        family_id: FamilyId,
        label: impl Into<String>,
        credential: CredentialMaterial,
    ) -> AccountId {
        let n = self.next.fetch_add(1, Ordering::Relaxed);
        let id = AccountId::new(format!("{}-{}", family_id.as_str(), n));
        self.accounts.insert(
            id.clone(),
            AccountRecord {
                family_id,
                credential,
                generation: 1,
                revoked: false,
                label: label.into(),
            },
        );
        id
    }

    /// Mint an explicit handle. This is the only way a connector receives
    /// credentials. There is no ambient / process-global lookup.
    pub fn mint_handle(&self, account_id: &AccountId) -> Result<AccountHandle> {
        let rec = self
            .accounts
            .get(account_id)
            .ok_or_else(|| ConnectorError::AccountNotFound(account_id.clone()))?;
        if rec.revoked {
            return Err(ConnectorError::AccountRevoked(account_id.clone()));
        }
        Ok(AccountHandle {
            account_id: account_id.clone(),
            family_id: rec.family_id.clone(),
            generation: rec.generation,
            credential: rec.credential.clone(),
        })
    }

    /// Validate a handle is still live for the expected family. Call at the
    /// start of every operation and again before completing an in-flight write.
    pub fn validate(&self, handle: &AccountHandle, expected_family: &FamilyId) -> Result<()> {
        if &handle.family_id != expected_family {
            return Err(ConnectorError::AccountFamilyMismatch {
                handle: handle.family_id.clone(),
                connector: expected_family.clone(),
            });
        }
        let rec = self
            .accounts
            .get(&handle.account_id)
            .ok_or_else(|| ConnectorError::AccountNotFound(handle.account_id.clone()))?;
        if rec.revoked {
            return Err(ConnectorError::AccountRevoked(handle.account_id.clone()));
        }
        if rec.generation != handle.generation {
            return Err(ConnectorError::StaleHandle);
        }
        if rec.family_id != handle.family_id {
            return Err(ConnectorError::AccountFamilyMismatch {
                handle: handle.family_id.clone(),
                connector: rec.family_id.clone(),
            });
        }
        // Credential isolation: the handle's material must match this account only.
        if rec.credential != handle.credential {
            return Err(ConnectorError::CredentialIsolation(
                handle.account_id.clone(),
                handle.family_id.clone(),
            ));
        }
        Ok(())
    }

    /// Revoke an account. Invalidates all outstanding handles (generation bump)
    /// and marks the account revoked so minting fails closed.
    pub fn revoke(&mut self, account_id: &AccountId) -> Result<()> {
        let rec = self
            .accounts
            .get_mut(account_id)
            .ok_or_else(|| ConnectorError::AccountNotFound(account_id.clone()))?;
        rec.revoked = true;
        rec.generation = rec.generation.saturating_add(1);
        // Clear credential material so any leaked copy cannot be re-used via a
        // forged handle with matching generation (generation already mismatch).
        rec.credential = CredentialMaterial {
            material: String::new(),
        };
        Ok(())
    }

    pub fn is_revoked(&self, account_id: &AccountId) -> bool {
        self.accounts
            .get(account_id)
            .map(|r| r.revoked)
            .unwrap_or(true)
    }

    pub fn label(&self, account_id: &AccountId) -> Option<&str> {
        self.accounts.get(account_id).map(|r| r.label.as_str())
    }

    /// Deliberately absent: there is no way to look up "the" credential for a
    /// family without an account id. This method documents that law.
    pub fn ambient_lookup_forbidden() -> ConnectorError {
        ConnectorError::AmbientCredentialForbidden
    }
}

/// Guard that re-validates a handle before completing an in-flight operation.
///
/// Does not borrow the store for its lifetime, so the store may be mutated
/// (e.g. revoked) between [`begin`](Self::begin) and
/// [`complete`](Self::complete). Dropping without `complete` is fine; the
/// operation simply did not finish.
pub struct InFlightGuard {
    handle: AccountHandle,
    family: FamilyId,
}

impl InFlightGuard {
    pub fn begin(store: &AccountStore, handle: &AccountHandle, family: &FamilyId) -> Result<Self> {
        store.validate(handle, family)?;
        Ok(Self {
            handle: handle.clone(),
            family: family.clone(),
        })
    }

    /// Re-check the handle against the (possibly updated) store. On revocation
    /// mid-flight this returns [`ConnectorError::AccountRevoked`] /
    /// [`ConnectorError::StaleHandle`] and the caller must not complete.
    pub fn complete(self, store: &AccountStore) -> Result<AccountHandle> {
        store.validate(&self.handle, &self.family)?;
        Ok(self.handle)
    }

    pub fn handle(&self) -> &AccountHandle {
        &self.handle
    }
}
