//! Seal fields required on every LEVEL 3 latent packet (bible §20).
//!
//! ```text
//! sender identity
//! model and revision
//! layer ranges
//! shape/dtype
//! visible commitment
//! payload hash
//! session
//! expiry
//! capability scope
//! ```
//!
//! No unsealed latent or KV transfer.

use crate::error::{CommsError, Result};
use serde::{Deserialize, Serialize};

/// Feature / policy gate. Live transfer stays closed even when a packet seals.
pub const LATENT_EXPERIMENTAL_GATE: &str = "hcli.comms.latent.experimental";

/// Whether a latent seal is complete enough to *consider* transfer (still gated).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SealStatus {
    /// All bible-required fields present and hashes well-formed.
    Sealed,
    /// Missing or inconsistent fields — transfer must refuse.
    Unsealed,
}

/// Capability scope bound into the seal — what the receiver may do with the
/// payload. Aligns with HCLI/session capability thinking (`hcli_bridge`
/// capability areas, hide-protocol environment capabilities).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilityScope {
    /// e.g. `same_model_session_transfer`, `embedding_share`, `kv_read_only`.
    pub scopes: Vec<String>,
    /// Explicit deny list (wins over scopes).
    #[serde(default)]
    pub deny: Vec<String>,
    /// If true, only same model+revision may accept (bible default for L3 start).
    #[serde(default = "default_true")]
    pub same_model_only: bool,
}

fn default_true() -> bool {
    true
}

impl Default for CapabilityScope {
    fn default() -> Self {
        Self {
            scopes: vec!["same_model_session_transfer".into()],
            deny: vec![
                "cross_model_latent".into(),
                "unsealed_kv".into(),
                "credential_read".into(),
            ],
            same_model_only: true,
        }
    }
}

impl CapabilityScope {
    pub fn allows(&self, scope: &str) -> bool {
        if self.deny.iter().any(|d| d == scope) {
            return false;
        }
        self.scopes.iter().any(|s| s == scope)
    }

    pub fn validate(&self) -> Result<()> {
        if self.scopes.is_empty() {
            return Err(CommsError::Invalid(
                "capability scope must list at least one granted scope".into(),
            ));
        }
        if self.allows("unsealed_kv") {
            return Err(CommsError::UnsealedLatent(
                "capability scope must not grant unsealed_kv".into(),
            ));
        }
        Ok(())
    }
}

/// Human/auditable commitment that remains inspectable even if the latent
/// payload is opaque (bible: visible commitments remain auditable).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VisibleCommitment {
    /// Short natural-language summary of what the latent is intended to carry.
    pub summary: String,
    /// Optional LEVEL 1 text commitment the sender also published.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_commitment: Option<String>,
    /// Optional LEVEL 2 structured commitment id / hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structured_commitment_hash: Option<String>,
}

impl VisibleCommitment {
    pub fn validate(&self) -> Result<()> {
        if self.summary.trim().is_empty() {
            return Err(CommsError::Invalid(
                "visible commitment summary must be non-empty".into(),
            ));
        }
        if self.summary.len() > 8 * 1024 {
            return Err(CommsError::Invalid(
                "visible commitment summary exceeds 8KiB".into(),
            ));
        }
        Ok(())
    }
}

/// Aggregate seal header — every latent packet carries one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SealHeader {
    pub sender_identity: String,
    pub model: crate::level3::ModelIdentity,
    pub layer_ranges: Vec<crate::level3::LayerRange>,
    pub shape: Vec<u64>,
    pub dtype: crate::level3::LatentDType,
    pub visible_commitment: VisibleCommitment,
    /// blake3 of the opaque payload bytes (or of the empty placeholder).
    pub payload_hash: String,
    pub session_id: String,
    /// Unix epoch ms after which the packet must not be accepted.
    pub expiry_unix_ms: u64,
    pub capability_scope: CapabilityScope,
    /// Explicit experimental marker — must be true for L3.
    pub experimental: bool,
    /// Gate id the controller must have open to attempt transfer.
    pub experimental_gate: String,
}

impl SealHeader {
    pub fn status(&self) -> SealStatus {
        if self.validate_fields().is_ok() {
            SealStatus::Sealed
        } else {
            SealStatus::Unsealed
        }
    }

    pub fn validate_fields(&self) -> Result<()> {
        if self.sender_identity.trim().is_empty() {
            return Err(CommsError::UnsealedLatent("missing sender identity".into()));
        }
        self.model.validate()?;
        if self.layer_ranges.is_empty() {
            return Err(CommsError::UnsealedLatent("missing layer ranges".into()));
        }
        for r in &self.layer_ranges {
            r.validate()?;
        }
        if self.shape.is_empty() {
            return Err(CommsError::UnsealedLatent("missing shape".into()));
        }
        if !self.payload_hash.starts_with("blake3:") || self.payload_hash.len() < 16 {
            return Err(CommsError::UnsealedLatent(
                "payload hash must be blake3:…".into(),
            ));
        }
        if self.session_id.trim().is_empty() {
            return Err(CommsError::UnsealedLatent("missing session".into()));
        }
        if self.expiry_unix_ms == 0 {
            return Err(CommsError::UnsealedLatent("missing expiry".into()));
        }
        self.visible_commitment.validate()?;
        self.capability_scope.validate()?;
        if !self.experimental {
            return Err(CommsError::LatentGateClosed(
                "latent packets must set experimental=true".into(),
            ));
        }
        if self.experimental_gate != LATENT_EXPERIMENTAL_GATE {
            return Err(CommsError::LatentGateClosed(format!(
                "expected gate {LATENT_EXPERIMENTAL_GATE}, got {}",
                self.experimental_gate
            )));
        }
        Ok(())
    }

    pub fn check_not_expired(&self, now_unix_ms: u64) -> Result<()> {
        if now_unix_ms > self.expiry_unix_ms {
            return Err(CommsError::Expired {
                expiry_unix_ms: self.expiry_unix_ms,
                now_unix_ms,
            });
        }
        Ok(())
    }

    pub fn verify_payload_bytes(&self, payload: &[u8]) -> Result<()> {
        let got = format!("blake3:{}", blake3::hash(payload).to_hex());
        if got != self.payload_hash {
            return Err(CommsError::HashMismatch {
                expected: self.payload_hash.clone(),
                got,
            });
        }
        Ok(())
    }
}
