//! YOU privacy modes, enforced at the **capability boundary**.
//!
//! Modes are not advice to subsystems. A network-disabled session must not be
//! able to **construct** a network-capable handle at all — tests assert
//! construction failure, not that a later call is refused.
//!
//! Modes:
//! - `local_offline` — no network for any subsystem; connectors refuse construct
//! - `network_disabled` — research/network refuse construct; local tools continue
//! - `connector_disabled` — network may exist for research; connectors refuse construct
//! - `ephemeral_no_memory` — working only; nothing durable is written
//! - `encrypted_vault` — private_vault scope is sealed behind a vault handle

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use crate::memory_classes::{
    ClassMemoryDraft, ClassMemoryRecord, ClassedMemorySystem, EpisodicWriteCap, MemoryClass,
    PersonalScope, ProceduralWriteCap, ProjectWriteCap, TurnWriteCap, UserWriteCap,
    VerifierWriteCap,
};
use hide_core::error::{HideError, Result};

/// Privacy modes from the YOU surface authority contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PrivacyMode {
    LocalOffline,
    NetworkDisabled,
    ConnectorDisabled,
    EphemeralNoMemory,
    EncryptedVault,
}

impl PrivacyMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LocalOffline => "local_offline",
            Self::NetworkDisabled => "network_disabled",
            Self::ConnectorDisabled => "connector_disabled",
            Self::EphemeralNoMemory => "ephemeral_no_memory",
            Self::EncryptedVault => "encrypted_vault",
        }
    }

    pub fn all() -> [PrivacyMode; 5] {
        [
            Self::LocalOffline,
            Self::NetworkDisabled,
            Self::ConnectorDisabled,
            Self::EphemeralNoMemory,
            Self::EncryptedVault,
        ]
    }
}

/// Why a capability handle could not be constructed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyBoundaryError {
    pub handle: String,
    pub modes: Vec<String>,
    pub reason: String,
}

impl std::fmt::Display for PrivacyBoundaryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "privacy boundary: cannot construct {} under modes [{}]: {}",
            self.handle,
            self.modes.join(","),
            self.reason
        )
    }
}

impl std::error::Error for PrivacyBoundaryError {}

impl From<PrivacyBoundaryError> for HideError {
    fn from(e: PrivacyBoundaryError) -> Self {
        HideError::CapabilityMissing(e.to_string())
    }
}

/// Active privacy policy for a session. Modes compose (set union).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrivacyPolicy {
    pub modes: BTreeSet<PrivacyMode>,
}

impl PrivacyPolicy {
    pub fn open() -> Self {
        Self::default()
    }

    pub fn with_mode(mut self, mode: PrivacyMode) -> Self {
        self.modes.insert(mode);
        self
    }

    pub fn from_modes(modes: impl IntoIterator<Item = PrivacyMode>) -> Self {
        Self {
            modes: modes.into_iter().collect(),
        }
    }

    pub fn has(&self, mode: PrivacyMode) -> bool {
        self.modes.contains(&mode)
    }

    pub fn allows_network(&self) -> bool {
        !self.has(PrivacyMode::LocalOffline) && !self.has(PrivacyMode::NetworkDisabled)
    }

    pub fn allows_connectors(&self) -> bool {
        !self.has(PrivacyMode::LocalOffline) && !self.has(PrivacyMode::ConnectorDisabled)
    }

    pub fn allows_durable_memory(&self) -> bool {
        !self.has(PrivacyMode::EphemeralNoMemory)
    }

    pub fn requires_encrypted_vault(&self) -> bool {
        self.has(PrivacyMode::EncryptedVault)
    }

    pub fn is_ephemeral(&self) -> bool {
        self.has(PrivacyMode::EphemeralNoMemory)
    }

    fn mode_labels(&self) -> Vec<String> {
        self.modes.iter().map(|m| m.as_str().to_string()).collect()
    }
}

// ---------------------------------------------------------------------------
// Capability handles — construction is the gate
// ---------------------------------------------------------------------------

/// Network-capable handle. Construction fails under local_offline / network_disabled.
///
/// Holding this type is proof the session was allowed network at construct time.
/// There is no "soft refuse on call" path for construction-gated code.
#[derive(Debug)]
pub struct NetworkCapableHandle {
    _private: (),
}

impl NetworkCapableHandle {
    /// Construct a network-capable handle. **Fails at construction**, not on use.
    pub fn try_construct(
        policy: &PrivacyPolicy,
    ) -> std::result::Result<Self, PrivacyBoundaryError> {
        if !policy.allows_network() {
            return Err(PrivacyBoundaryError {
                handle: "NetworkCapableHandle".into(),
                modes: policy.mode_labels(),
                reason: "session policy forbids network; handle must not exist".into(),
            });
        }
        Ok(Self { _private: () })
    }
}

/// Connector-capable handle. Construction fails under local_offline / connector_disabled.
#[derive(Debug)]
pub struct ConnectorCapableHandle {
    _private: (),
}

impl ConnectorCapableHandle {
    pub fn try_construct(
        policy: &PrivacyPolicy,
    ) -> std::result::Result<Self, PrivacyBoundaryError> {
        if !policy.allows_connectors() {
            return Err(PrivacyBoundaryError {
                handle: "ConnectorCapableHandle".into(),
                modes: policy.mode_labels(),
                reason: "session policy forbids connectors; handle must not exist".into(),
            });
        }
        Ok(Self { _private: () })
    }
}

/// Handle for reading/writing `private_vault` scope under encrypted_vault mode.
/// Without encrypted_vault mode, vault writes use ordinary memory paths.
/// With encrypted_vault mode, only this handle may touch private_vault scope.
#[derive(Debug)]
pub struct EncryptedVaultHandle {
    _private: (),
}

impl EncryptedVaultHandle {
    pub fn try_construct(
        policy: &PrivacyPolicy,
    ) -> std::result::Result<Self, PrivacyBoundaryError> {
        if !policy.requires_encrypted_vault() {
            return Err(PrivacyBoundaryError {
                handle: "EncryptedVaultHandle".into(),
                modes: policy.mode_labels(),
                reason: "encrypted_vault mode is not active".into(),
            });
        }
        Ok(Self { _private: () })
    }
}

// ---------------------------------------------------------------------------
// Privacy-bound session over ClassedMemorySystem
// ---------------------------------------------------------------------------

/// A session bound to a privacy policy. Durable memory writes go through this
/// type so ephemeral mode cannot leave an episodic trace.
pub struct PrivacySession {
    pub session_id: String,
    pub policy: PrivacyPolicy,
}

impl PrivacySession {
    pub fn new(session_id: impl Into<String>, policy: PrivacyPolicy) -> Self {
        Self {
            session_id: session_id.into(),
            policy,
        }
    }

    pub fn write_working(
        &self,
        mem: &ClassedMemorySystem,
        cap: &TurnWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        // Working is always allowed (turn-local scratch).
        let mut d = draft;
        if d.session_id.is_none() {
            d.session_id = Some(self.session_id.clone());
        }
        mem.write_working(cap, writer, d)
    }

    fn refuse_durable(&self, class: MemoryClass) -> Result<()> {
        if !self.policy.allows_durable_memory() {
            return Err(HideError::PolicyDenied(format!(
                "ephemeral_no_memory: durable class {} write refused; working only",
                class.as_str()
            )));
        }
        Ok(())
    }

    fn check_vault_scope(&self, draft: &ClassMemoryDraft) -> Result<()> {
        if draft.scope == Some(PersonalScope::PrivateVault)
            && self.policy.requires_encrypted_vault()
        {
            // Callers must use write_*_vault with EncryptedVaultHandle.
            return Err(HideError::CapabilityMissing(
                "private_vault under encrypted_vault mode requires EncryptedVaultHandle".into(),
            ));
        }
        Ok(())
    }

    pub fn write_episodic(
        &self,
        mem: &ClassedMemorySystem,
        cap: &EpisodicWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.refuse_durable(MemoryClass::Episodic)?;
        self.check_vault_scope(&draft)?;
        let mut d = draft;
        if d.session_id.is_none() {
            d.session_id = Some(self.session_id.clone());
        }
        mem.write_episodic(cap, writer, d)
    }

    pub fn write_semantic_project(
        &self,
        mem: &ClassedMemorySystem,
        cap: &ProjectWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.refuse_durable(MemoryClass::SemanticProject)?;
        self.check_vault_scope(&draft)?;
        mem.write_semantic_project(cap, writer, draft)
    }

    pub fn write_procedural(
        &self,
        mem: &ClassedMemorySystem,
        cap: &ProceduralWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.refuse_durable(MemoryClass::Procedural)?;
        self.check_vault_scope(&draft)?;
        mem.write_procedural(cap, writer, draft)
    }

    pub fn write_user(
        &self,
        mem: &ClassedMemorySystem,
        cap: &UserWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.refuse_durable(MemoryClass::User)?;
        self.check_vault_scope(&draft)?;
        mem.write_user(cap, writer, draft)
    }

    pub fn write_verification(
        &self,
        mem: &ClassedMemorySystem,
        cap: &VerifierWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.refuse_durable(MemoryClass::Verification)?;
        self.check_vault_scope(&draft)?;
        mem.write_verification(cap, writer, draft)
    }

    /// Vault-scoped user write under encrypted_vault mode. Requires the vault handle.
    pub fn write_user_vault(
        &self,
        mem: &ClassedMemorySystem,
        _vault: &EncryptedVaultHandle,
        cap: &UserWriteCap,
        writer: impl Into<String>,
        draft: ClassMemoryDraft,
    ) -> Result<ClassMemoryRecord> {
        self.refuse_durable(MemoryClass::User)?;
        let d = draft.with_scope(PersonalScope::PrivateVault);
        mem.write_user(cap, writer, d)
    }

    /// End the session. Under ephemeral mode, purge any durable refs to this
    /// session id so nothing durable references it.
    pub fn end(self, mem: &ClassedMemorySystem) -> Result<EphemeralEndReport> {
        let mut report = EphemeralEndReport {
            session_id: self.session_id.clone(),
            was_ephemeral: self.policy.is_ephemeral(),
            durable_refs_before: mem.durable_refs_to_session(&self.session_id)?,
            durable_refs_after: 0,
            purged: 0,
        };
        if self.policy.is_ephemeral() {
            // Real deletion of any session-linked durable rows.
            report.purged = mem.evict_session(&self.session_id)?;
            // Also drop ephemeral-scoped durable records (any class).
            let n = mem.forget_scope(PersonalScope::Ephemeral)?;
            report.purged += n;
        }
        report.durable_refs_after = mem.durable_refs_to_session(&self.session_id)?;
        Ok(report)
    }
}

/// Proof that an ephemeral session left no durable trace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EphemeralEndReport {
    pub session_id: String,
    pub was_ephemeral: bool,
    pub durable_refs_before: usize,
    pub durable_refs_after: usize,
    pub purged: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn property_network_disabled_cannot_construct_network_handle() {
        let policy = PrivacyPolicy::open().with_mode(PrivacyMode::NetworkDisabled);
        let err = NetworkCapableHandle::try_construct(&policy).unwrap_err();
        assert_eq!(err.handle, "NetworkCapableHandle");
        assert!(err.modes.iter().any(|m| m == "network_disabled"));
        assert!(NetworkCapableHandle::try_construct(&PrivacyPolicy::open()).is_ok());
        let offline = PrivacyPolicy::open().with_mode(PrivacyMode::LocalOffline);
        assert!(NetworkCapableHandle::try_construct(&offline).is_err());
        assert!(ConnectorCapableHandle::try_construct(&offline).is_err());
    }
    #[test]
    fn property_connector_disabled_cannot_construct_connector_handle() {
        let policy = PrivacyPolicy::open().with_mode(PrivacyMode::ConnectorDisabled);
        assert!(ConnectorCapableHandle::try_construct(&policy).is_err());
        assert!(NetworkCapableHandle::try_construct(&policy).is_ok());
    }
    #[test]
    fn property_ephemeral_means_ephemeral_no_durable_trace() {
        let mem = ClassedMemorySystem::open_in_memory("ws-eph").unwrap();
        let policy = PrivacyPolicy::open().with_mode(PrivacyMode::EphemeralNoMemory);
        let session = PrivacySession::new("eph-sess-1", policy);
        let tcap = TurnWriteCap::new("t1");
        session
            .write_working(&mem, &tcap, "kernel", ClassMemoryDraft::new("scratch only"))
            .unwrap();
        assert_eq!(mem.list_working("t1").len(), 1);
        assert!(session
            .write_episodic(
                &mem,
                &EpisodicWriteCap::mint(),
                "event_stream",
                ClassMemoryDraft::new("should not land"),
            )
            .is_err());
        assert!(session
            .write_user(
                &mem,
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("no prefs in ephemeral"),
            )
            .is_err());
        assert!(session
            .write_verification(
                &mem,
                &VerifierWriteCap::mint(),
                "verifier",
                ClassMemoryDraft::new("no verify"),
            )
            .is_err());
        let report = session.end(&mem).unwrap();
        assert!(report.was_ephemeral);
        assert_eq!(report.durable_refs_after, 0);
        assert_eq!(mem.durable_refs_to_session("eph-sess-1").unwrap(), 0);
        assert_eq!(mem.count(MemoryClass::Episodic).unwrap(), 0);
        mem.end_turn("t1");
        assert!(mem.list_working("t1").is_empty());
    }
    #[test]
    fn property_ephemeral_end_purges_accidental_session_refs() {
        let mem = ClassedMemorySystem::open_in_memory("ws-eph2").unwrap();
        mem.write_episodic(
            &EpisodicWriteCap::mint(),
            "leak",
            ClassMemoryDraft::new("leaked").with_session("eph-sess-2"),
        )
        .unwrap();
        assert_eq!(mem.durable_refs_to_session("eph-sess-2").unwrap(), 1);
        let session = PrivacySession::new(
            "eph-sess-2",
            PrivacyPolicy::open().with_mode(PrivacyMode::EphemeralNoMemory),
        );
        let report = session.end(&mem).unwrap();
        assert_eq!(report.durable_refs_after, 0);
        assert!(report.purged >= 1);
        assert_eq!(mem.count(MemoryClass::Episodic).unwrap(), 0);
    }
    #[test]
    fn property_encrypted_vault_requires_handle() {
        let mem = ClassedMemorySystem::open_in_memory("ws-vault").unwrap();
        let policy = PrivacyPolicy::open().with_mode(PrivacyMode::EncryptedVault);
        let session = PrivacySession::new("s-vault", policy.clone());
        let err = session
            .write_user(
                &mem,
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("secret").with_scope(PersonalScope::PrivateVault),
            )
            .unwrap_err();
        assert!(matches!(err, HideError::CapabilityMissing(_)));
        assert!(EncryptedVaultHandle::try_construct(&PrivacyPolicy::open()).is_err());
        let vault = EncryptedVaultHandle::try_construct(&policy).unwrap();
        let rec = session
            .write_user_vault(
                &mem,
                &vault,
                &UserWriteCap::mint(),
                "user",
                ClassMemoryDraft::new("secret"),
            )
            .unwrap();
        assert_eq!(rec.scope, PersonalScope::PrivateVault);
    }
}
