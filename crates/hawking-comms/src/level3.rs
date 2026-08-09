//! LEVEL 3 — LATENT: experimental sealed-packet format only.
//!
//! **Does not implement** actual hidden-state / embedding / KV transfer.
//! Begin same-model-to-same-model only (bible §20). Cross-model requires
//! trained alignment and independent evidence.

use crate::error::{CommsError, Result};
use crate::seal::{
    CapabilityScope, SealHeader, SealStatus, VisibleCommitment, LATENT_EXPERIMENTAL_GATE,
};
use serde::{Deserialize, Serialize};

/// What the opaque payload is claimed to be (declaration only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentKind {
    HiddenState,
    Embedding,
    KvCache,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LatentDType {
    F32,
    F16,
    Bf16,
    F8,
    I8,
    I32,
    Other,
}

impl LatentDType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F8 => "f8",
            Self::I8 => "i8",
            Self::I32 => "i32",
            Self::Other => "other",
        }
    }
}

/// Inclusive layer range `[start, end]` (0-based).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    pub fn validate(&self) -> Result<()> {
        if self.end < self.start {
            return Err(CommsError::Invalid(format!(
                "layer range end {} < start {}",
                self.end, self.start
            )));
        }
        Ok(())
    }
}

/// Model identity pins for seal + same-model policy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelIdentity {
    /// e.g. `Qwen/Qwen3-Coder-30B-A3B-Instruct`
    pub model_id: String,
    /// Exact revision / commit / gravity artifact digest.
    pub revision: String,
    /// Optional family tag (`qwen3`, `deepseek`, …).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
}

impl ModelIdentity {
    pub fn qwen30b(revision: impl Into<String>) -> Self {
        Self {
            model_id: "Qwen/Qwen3-Coder-30B-A3B-Instruct".into(),
            revision: revision.into(),
            family: Some("qwen3".into()),
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.model_id.trim().is_empty() || self.revision.trim().is_empty() {
            return Err(CommsError::UnsealedLatent(
                "model identity requires model_id and revision".into(),
            ));
        }
        Ok(())
    }

    pub fn same_as(&self, other: &Self) -> bool {
        self.model_id == other.model_id && self.revision == other.revision
    }
}

/// Reference to opaque payload bytes — **not** the live tensor itself.
///
/// Actual byte transport / GPU bind is DEFERRED. This only names and seals.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LatentPayloadRef {
    /// Content-addressed blob id or path pin.
    pub blob_ref: String,
    pub size_bytes: u64,
    /// True when payload bytes are present on the local host; false means
    /// header-only (still must be sealed).
    pub bytes_present: bool,
}

/// Sealed LEVEL 3 packet. Experimental; no unsealed transfer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LatentPacket {
    pub schema: String,
    pub kind: LatentKind,
    pub seal: SealHeader,
    /// Opaque payload pointer only. May be empty when header-only.
    pub payload: LatentPayloadRef,
    /// Receiver session this packet is addressed to (may equal sender session
    /// for same-session hops; typically peer Qwen session).
    pub recipient_session_id: String,
    /// Declared receiver model — must match seal.model when same_model_only.
    pub recipient_model: ModelIdentity,
}

impl LatentPacket {
    /// Build a sealed same-model Qwen→Qwen packet (format only; no transfer).
    pub fn sealed_same_model_qwen(
        sender_identity: impl Into<String>,
        sender_session: impl Into<String>,
        recipient_session: impl Into<String>,
        revision: impl Into<String>,
        layer_ranges: Vec<LayerRange>,
        shape: Vec<u64>,
        dtype: LatentDType,
        kind: LatentKind,
        visible_summary: impl Into<String>,
        payload_bytes: &[u8],
        expiry_unix_ms: u64,
    ) -> Result<Self> {
        let model = ModelIdentity::qwen30b(revision);
        let payload_hash = format!("blake3:{}", blake3::hash(payload_bytes).to_hex());
        let sender_session = sender_session.into();
        let seal = SealHeader {
            sender_identity: sender_identity.into(),
            model: model.clone(),
            layer_ranges,
            shape,
            dtype,
            visible_commitment: VisibleCommitment {
                summary: visible_summary.into(),
                text_commitment: None,
                structured_commitment_hash: None,
            },
            payload_hash,
            session_id: sender_session,
            expiry_unix_ms,
            capability_scope: CapabilityScope::default(),
            experimental: true,
            experimental_gate: LATENT_EXPERIMENTAL_GATE.to_string(),
        };
        let packet = Self {
            schema: crate::LATENT_PACKET_SCHEMA.to_string(),
            kind,
            seal,
            payload: LatentPayloadRef {
                blob_ref: format!("latent:{}", blake3::hash(payload_bytes).to_hex()),
                size_bytes: payload_bytes.len() as u64,
                bytes_present: !payload_bytes.is_empty(),
            },
            recipient_session_id: recipient_session.into(),
            recipient_model: model,
        };
        packet.validate_sealed()?;
        Ok(packet)
    }

    pub fn seal_status(&self) -> SealStatus {
        self.seal.status()
    }

    /// Full seal + same-model policy check. **Does not** transfer anything.
    pub fn validate_sealed(&self) -> Result<()> {
        if self.schema != crate::LATENT_PACKET_SCHEMA {
            return Err(CommsError::Invalid(format!(
                "latent schema {}",
                self.schema
            )));
        }
        self.seal.validate_fields()?;
        if self.recipient_session_id.trim().is_empty() {
            return Err(CommsError::UnsealedLatent(
                "missing recipient session".into(),
            ));
        }
        self.recipient_model.validate()?;

        // Same-model-only by default (bible: begin same-model Qwen session pairs).
        if self.seal.capability_scope.same_model_only
            && !self.seal.model.same_as(&self.recipient_model)
        {
            return Err(CommsError::CrossModelLatent {
                sender: format!("{}@{}", self.seal.model.model_id, self.seal.model.revision),
                receiver: format!(
                    "{}@{}",
                    self.recipient_model.model_id, self.recipient_model.revision
                ),
            });
        }

        // Explicit refusal of unsealed_kv capability.
        if self.seal.capability_scope.allows("unsealed_kv") {
            return Err(CommsError::UnsealedLatent(
                "unsealed_kv capability is forbidden".into(),
            ));
        }

        Ok(())
    }

    /// Check expiry + payload hash. Still does **not** bind into a runtime.
    pub fn validate_for_acceptance(
        &self,
        now_unix_ms: u64,
        payload_bytes: Option<&[u8]>,
    ) -> Result<()> {
        self.validate_sealed()?;
        self.seal.check_not_expired(now_unix_ms)?;
        if let Some(bytes) = payload_bytes {
            self.seal.verify_payload_bytes(bytes)?;
        }
        // Live transfer remains deferred even when acceptance checks pass.
        Err(CommsError::Deferred(
            "latent transfer binding is experimental and not implemented; sealed format only",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::BusEnvelope;

    #[test]
    fn sealed_same_model_packet_validates() {
        let bytes = b"not-real-kv-just-format-bytes";
        let pkt = LatentPacket::sealed_same_model_qwen(
            "agent_executor",
            "ses_qwen_a",
            "ses_qwen_b",
            "rev-test-001",
            vec![LayerRange::new(0, 7)],
            vec![1, 8, 128],
            LatentDType::F16,
            LatentKind::KvCache,
            "KV slice layers 0-7 for continued decode (format test)",
            bytes,
            9_999_999_999_999,
        )
        .unwrap();
        assert_eq!(pkt.seal_status(), SealStatus::Sealed);
        pkt.validate_sealed().unwrap();
        // Acceptance still refuses live transfer.
        let err = pkt.validate_for_acceptance(1_000, Some(bytes)).unwrap_err();
        assert!(matches!(err, CommsError::Deferred(_)));

        let env = BusEnvelope::from_latent(pkt).unwrap();
        env.validate().unwrap();
        assert!(env.level().is_experimental());
    }

    #[test]
    fn unsealed_missing_sender_refused() {
        let mut pkt = LatentPacket::sealed_same_model_qwen(
            "agent",
            "ses_a",
            "ses_b",
            "rev",
            vec![LayerRange::new(0, 0)],
            vec![1],
            LatentDType::F32,
            LatentKind::Embedding,
            "embedding share test",
            b"x",
            9_999_999_999_999,
        )
        .unwrap();
        pkt.seal.sender_identity.clear();
        let err = pkt.validate_sealed().unwrap_err();
        assert!(matches!(err, CommsError::UnsealedLatent(_)));
    }

    #[test]
    fn cross_model_refused_when_same_model_only() {
        let mut pkt = LatentPacket::sealed_same_model_qwen(
            "agent",
            "ses_a",
            "ses_b",
            "rev",
            vec![LayerRange::new(0, 0)],
            vec![4],
            LatentDType::F16,
            LatentKind::HiddenState,
            "hidden state",
            b"abc",
            9_999_999_999_999,
        )
        .unwrap();
        pkt.recipient_model = ModelIdentity {
            model_id: "other/model".into(),
            revision: "x".into(),
            family: None,
        };
        let err = pkt.validate_sealed().unwrap_err();
        assert!(matches!(err, CommsError::CrossModelLatent { .. }));
    }

    #[test]
    fn payload_hash_mismatch_detected() {
        let bytes = b"payload-a";
        let pkt = LatentPacket::sealed_same_model_qwen(
            "agent",
            "ses_a",
            "ses_b",
            "rev",
            vec![LayerRange::new(1, 2)],
            vec![2, 2],
            LatentDType::F32,
            LatentKind::Other,
            "other latent",
            bytes,
            9_999_999_999_999,
        )
        .unwrap();
        let err = pkt.seal.verify_payload_bytes(b"payload-b").unwrap_err();
        assert!(matches!(err, CommsError::HashMismatch { .. }));
    }
}
