//! Capsule kinds and the header that describes a sealed capsule.
//!
//! The header is everything a reader needs to identify a capsule and verify its
//! payload without loading the payload itself: what kind of state it holds, the
//! model and runtime it was captured under, where in the sequence it sits, its
//! ancestry, its size, and the integrity digest of its bytes.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::integrity::Integrity;

/// The kind of runtime state a capsule holds. Every kind shares the same header
/// and integrity story; the type only tells a consumer how to interpret the
/// opaque payload once it has been verified and bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CapsuleType {
    /// Recurrent (state-space) hidden state.
    Recurrent,
    /// Attention key/value cache.
    Kv,
    /// A hybrid model carrying both a recurrent state and a key/value cache.
    HybridRecurrentKv,
    /// A reference to a shared prefix cache rather than the cache bytes.
    PrefixCacheRef,
    /// Serialized tool-runtime state.
    ToolRuntime,
    /// Serialized browser-session state.
    Browser,
    /// A repository checkpoint.
    RepoCheckpoint,
    /// A projection of a conversation into a compact carried form.
    ConversationProjection,
}

/// A capsule identifier. Minted as a ULID string so ids sort by creation time
/// and are unique without coordination.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CapsuleId(pub String);

impl CapsuleId {
    /// Mint a fresh, unique id.
    pub fn new() -> Self {
        CapsuleId(Ulid::new().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CapsuleId {
    fn default() -> Self {
        CapsuleId::new()
    }
}

impl std::fmt::Display for CapsuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Wall-clock milliseconds since the Unix epoch, captured once when a header is
/// sealed. Informational only; ordering between capsules uses ancestry, not
/// this timestamp.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// The self-describing header of a capsule.
///
/// `bytes` is the payload length and `integrity` is the digest of the payload,
/// so a reader can check both without trusting the byte stream that carried
/// them. `dtype` and `device` are free-form runtime tags (for example `"f16"`,
/// `"metal"`); the crate never interprets them, it only carries them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsuleHeader {
    pub capsule_id: CapsuleId,
    pub capsule_type: CapsuleType,
    pub model_id: String,
    pub model_hash: String,
    pub runtime_version: String,
    pub dtype: String,
    pub device: String,
    /// Sequence position the capsule was captured at.
    pub position: u64,
    /// Hash of the context pack the capsule was produced against.
    pub context_pack_hash: String,
    /// The capsule this one was forked from, if any.
    pub parent_capsule_id: Option<CapsuleId>,
    /// Wall-clock milliseconds since the Unix epoch at seal time.
    pub created_at: u64,
    /// Length of the payload in bytes.
    pub bytes: u64,
    /// Digest of the payload.
    pub integrity: Integrity,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capsule_ids_are_distinct() {
        let a = CapsuleId::new();
        let b = CapsuleId::new();
        assert_ne!(a, b);
        assert_eq!(a.as_str().len(), 26);
    }

    #[test]
    fn capsule_type_roundtrips_through_json() {
        for ty in [
            CapsuleType::Recurrent,
            CapsuleType::Kv,
            CapsuleType::HybridRecurrentKv,
            CapsuleType::PrefixCacheRef,
            CapsuleType::ToolRuntime,
            CapsuleType::Browser,
            CapsuleType::RepoCheckpoint,
            CapsuleType::ConversationProjection,
        ] {
            let json = serde_json::to_string(&ty).unwrap();
            let back: CapsuleType = serde_json::from_str(&json).unwrap();
            assert_eq!(ty, back);
        }
    }
}
