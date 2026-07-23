//! The capsule itself: header, identity, payload, and its byte format.
//!
//! A capsule serializes to a self-describing byte stream: a magic tag, a format
//! version, a length-prefixed JSON metadata block (header plus identity), then
//! the raw payload. Reading is integrity-checked: [`Capsule::from_bytes`]
//! rejects a stream whose payload length or digest disagrees with the header,
//! so a flipped byte can never be loaded silently.

use serde::{Deserialize, Serialize};

use crate::error::{CapsuleError, IncompatibleReason, Result};
use crate::header::{now_ms, CapsuleHeader, CapsuleId, CapsuleType};
use crate::identity::IdentityBinding;
use crate::integrity::{Integrity, IntegrityAlgo};

/// Magic tag at the head of every serialized capsule.
const MAGIC: &[u8; 8] = b"HIDECAP1";
/// The byte format this build writes and reads.
const FORMAT_VERSION: u16 = 1;
/// Fixed-size prefix: magic (8) plus version (2) plus meta length (4).
const PREFIX_LEN: usize = 8 + 2 + 4;

/// Owned metadata block, used when reading a capsule back from bytes.
#[derive(Deserialize)]
struct MetaOwned {
    header: CapsuleHeader,
    identity: IdentityBinding,
}

/// Borrowing metadata block, used when writing a capsule so serialization does
/// not clone the header and identity.
#[derive(Serialize)]
struct MetaRef<'a> {
    header: &'a CapsuleHeader,
    identity: &'a IdentityBinding,
}

/// Capsule metadata without the payload, returned by [`Capsule::inspect`].
///
/// Everything needed to identify and audit a capsule is here; the payload is
/// deliberately absent so an inspector never has to materialize it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleInspect {
    pub header: CapsuleHeader,
    pub identity: IdentityBinding,
}

/// A sealed capsule: descriptive header, identity binding, and opaque payload.
///
/// The payload is private so a capsule cannot drift out of agreement with the
/// length and digest recorded in its header; read it with [`Capsule::payload`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Capsule {
    header: CapsuleHeader,
    identity: IdentityBinding,
    payload: Vec<u8>,
}

impl Capsule {
    pub fn header(&self) -> &CapsuleHeader {
        &self.header
    }

    pub fn identity(&self) -> &IdentityBinding {
        &self.identity
    }

    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    pub fn capsule_id(&self) -> &CapsuleId {
        &self.header.capsule_id
    }

    pub fn parent_capsule_id(&self) -> Option<&CapsuleId> {
        self.header.parent_capsule_id.as_ref()
    }

    /// Whether this capsule can bind to a live runtime identity. Delegates to
    /// [`IdentityBinding::is_loadable`].
    pub fn is_loadable(&self, live: &IdentityBinding) -> std::result::Result<(), IncompatibleReason> {
        self.identity.is_loadable(live)
    }

    /// Fork this capsule: a byte-for-byte copy of the payload under a fresh id,
    /// with ancestry recorded. The new capsule's `parent_capsule_id` points at
    /// this capsule and its `created_at` is refreshed; everything else,
    /// including the payload bytes and integrity digest, is preserved.
    pub fn fork(&self) -> Capsule {
        let mut header = self.header.clone();
        header.parent_capsule_id = Some(self.header.capsule_id.clone());
        header.capsule_id = CapsuleId::new();
        header.created_at = now_ms();
        Capsule {
            header,
            identity: self.identity.clone(),
            payload: self.payload.clone(),
        }
    }

    /// Release the capsule's payload, zeroing the bytes before dropping them,
    /// and return the number of bytes reclaimed. Consumes the capsule.
    pub fn release(mut self) -> usize {
        let n = self.payload.len();
        for byte in self.payload.iter_mut() {
            *byte = 0;
        }
        self.payload.clear();
        n
    }

    /// Metadata without the payload.
    pub fn inspect(&self) -> CapsuleInspect {
        CapsuleInspect {
            header: self.header.clone(),
            identity: self.identity.clone(),
        }
    }

    /// Serialize to the self-describing, integrity-carrying byte stream.
    pub fn to_bytes(&self) -> Vec<u8> {
        let meta = MetaRef {
            header: &self.header,
            identity: &self.identity,
        };
        // Header and identity are plain data with infallible serialization.
        let meta_json = serde_json::to_vec(&meta).expect("capsule metadata serializes");
        let mut out = Vec::with_capacity(PREFIX_LEN + meta_json.len() + self.payload.len());
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(&(meta_json.len() as u32).to_le_bytes());
        out.extend_from_slice(&meta_json);
        out.extend_from_slice(&self.payload);
        out
    }

    /// Parse a capsule from bytes, verifying the payload length and digest
    /// against the header. Rejects a truncated stream, a bad magic tag, an
    /// unknown version, a length mismatch, or a digest mismatch.
    pub fn from_bytes(bytes: &[u8]) -> Result<Capsule> {
        let (meta, meta_end) = parse_prefix_and_meta(bytes)?;
        let payload = bytes[meta_end..].to_vec();

        if payload.len() as u64 != meta.header.bytes {
            return Err(CapsuleError::LengthMismatch {
                declared: meta.header.bytes,
                actual: payload.len() as u64,
            });
        }
        if !meta.header.integrity.verify(&payload) {
            return Err(CapsuleError::IntegrityMismatch);
        }
        Ok(Capsule {
            header: meta.header,
            identity: meta.identity,
            payload,
        })
    }

    /// Parse only the metadata of a serialized capsule, without copying the
    /// payload. Used to inspect a stored capsule cheaply. The prefix and
    /// metadata are validated; the payload bytes are not read into memory, so
    /// this does not verify the payload digest.
    pub fn inspect_bytes(bytes: &[u8]) -> Result<CapsuleInspect> {
        let (meta, _meta_end) = parse_prefix_and_meta(bytes)?;
        Ok(CapsuleInspect {
            header: meta.header,
            identity: meta.identity,
        })
    }
}

/// Validate the fixed prefix and decode the metadata block. Returns the decoded
/// metadata and the offset at which the payload begins.
fn parse_prefix_and_meta(bytes: &[u8]) -> Result<(MetaOwned, usize)> {
    if bytes.len() < PREFIX_LEN {
        return Err(CapsuleError::Truncated { detail: "prefix" });
    }
    if &bytes[0..8] != MAGIC {
        return Err(CapsuleError::BadMagic);
    }
    let version = u16::from_le_bytes([bytes[8], bytes[9]]);
    if version != FORMAT_VERSION {
        return Err(CapsuleError::UnsupportedVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }
    let meta_len = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]) as usize;
    let meta_end = PREFIX_LEN
        .checked_add(meta_len)
        .ok_or(CapsuleError::Truncated { detail: "meta length overflow" })?;
    if bytes.len() < meta_end {
        return Err(CapsuleError::Truncated { detail: "metadata" });
    }
    let meta: MetaOwned = serde_json::from_slice(&bytes[PREFIX_LEN..meta_end])?;
    Ok((meta, meta_end))
}

/// Fields common to every capsule in a sealing, gathered so [`CapsuleBuilder`]
/// stays readable. All have sensible empty defaults.
#[derive(Debug, Clone)]
pub struct CapsuleBuilder {
    capsule_type: CapsuleType,
    model_id: String,
    identity: IdentityBinding,
    model_hash: String,
    runtime_version: String,
    dtype: String,
    device: String,
    position: u64,
    context_pack_hash: String,
    parent_capsule_id: Option<CapsuleId>,
    algo: IntegrityAlgo,
}

impl CapsuleBuilder {
    /// Start a builder for a capsule of the given kind, model, and identity.
    /// The integrity algorithm defaults to blake3 and all descriptive tags
    /// default to empty; set them with the methods below before sealing.
    pub fn new(capsule_type: CapsuleType, model_id: impl Into<String>, identity: IdentityBinding) -> Self {
        CapsuleBuilder {
            capsule_type,
            model_id: model_id.into(),
            identity,
            model_hash: String::new(),
            runtime_version: String::new(),
            dtype: String::new(),
            device: String::new(),
            position: 0,
            context_pack_hash: String::new(),
            parent_capsule_id: None,
            algo: IntegrityAlgo::Blake3,
        }
    }

    pub fn model_hash(mut self, v: impl Into<String>) -> Self {
        self.model_hash = v.into();
        self
    }

    pub fn runtime_version(mut self, v: impl Into<String>) -> Self {
        self.runtime_version = v.into();
        self
    }

    pub fn dtype(mut self, v: impl Into<String>) -> Self {
        self.dtype = v.into();
        self
    }

    pub fn device(mut self, v: impl Into<String>) -> Self {
        self.device = v.into();
        self
    }

    pub fn position(mut self, v: u64) -> Self {
        self.position = v;
        self
    }

    pub fn context_pack_hash(mut self, v: impl Into<String>) -> Self {
        self.context_pack_hash = v.into();
        self
    }

    pub fn parent(mut self, v: CapsuleId) -> Self {
        self.parent_capsule_id = Some(v);
        self
    }

    pub fn integrity_algo(mut self, algo: IntegrityAlgo) -> Self {
        self.algo = algo;
        self
    }

    /// Seal the builder over `payload`, minting a fresh id and computing the
    /// payload length and integrity digest.
    pub fn seal(self, payload: Vec<u8>) -> Capsule {
        let integrity = Integrity::compute(self.algo, &payload);
        let header = CapsuleHeader {
            capsule_id: CapsuleId::new(),
            capsule_type: self.capsule_type,
            model_id: self.model_id,
            model_hash: self.model_hash,
            runtime_version: self.runtime_version,
            dtype: self.dtype,
            device: self.device,
            position: self.position,
            context_pack_hash: self.context_pack_hash,
            parent_capsule_id: self.parent_capsule_id,
            created_at: now_ms(),
            bytes: payload.len() as u64,
            integrity,
        };
        Capsule {
            header,
            identity: self.identity,
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> IdentityBinding {
        IdentityBinding {
            model_weights_id: "w".into(),
            arch_id: "a".into(),
            tokenizer_id: "t".into(),
            prompt_abi_version: "1".into(),
            tool_registry_id: "r".into(),
            engine_build_id: "b".into(),
            security_domain: "d".into(),
        }
    }

    fn sample(payload: Vec<u8>) -> Capsule {
        CapsuleBuilder::new(CapsuleType::Recurrent, "model-x", identity())
            .runtime_version("rt-1")
            .dtype("f16")
            .device("metal")
            .position(42)
            .context_pack_hash("ctx-hash")
            .seal(payload)
    }

    #[test]
    fn seal_records_length_and_digest() {
        let payload = vec![9u8; 128];
        let c = sample(payload.clone());
        assert_eq!(c.header().bytes, 128);
        assert!(c.header().integrity.verify(&payload));
        assert_eq!(c.header().capsule_type, CapsuleType::Recurrent);
        assert!(c.header().parent_capsule_id.is_none());
    }

    #[test]
    fn to_bytes_from_bytes_is_byte_identical() {
        let c = sample((0u8..200).collect());
        let bytes = c.to_bytes();
        let back = Capsule::from_bytes(&bytes).unwrap();
        assert_eq!(c, back);
        // Re-serializing the parsed capsule yields the same bytes.
        assert_eq!(back.to_bytes(), bytes);
    }

    #[test]
    fn flipped_payload_byte_is_rejected() {
        let c = sample((0u8..64).collect());
        let mut bytes = c.to_bytes();
        // Flip the final byte, which is inside the payload.
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(matches!(
            Capsule::from_bytes(&bytes),
            Err(CapsuleError::IntegrityMismatch)
        ));
    }

    #[test]
    fn bad_magic_and_short_stream_are_rejected() {
        assert!(matches!(
            Capsule::from_bytes(b"too short"),
            Err(CapsuleError::BadMagic) | Err(CapsuleError::Truncated { .. })
        ));
        assert!(matches!(
            Capsule::from_bytes(b"XXXXXXXX\x01\x00\x00\x00\x00\x00"),
            Err(CapsuleError::BadMagic)
        ));
    }

    #[test]
    fn wrong_version_is_rejected() {
        let c = sample(vec![1, 2, 3]);
        let mut bytes = c.to_bytes();
        bytes[8] = 0xFF; // corrupt the version low byte
        assert!(matches!(
            Capsule::from_bytes(&bytes),
            Err(CapsuleError::UnsupportedVersion { .. })
        ));
    }

    #[test]
    fn fork_preserves_payload_sets_ancestry_and_new_id() {
        let parent = sample((0u8..80).collect());
        let child = parent.fork();
        assert_ne!(child.capsule_id(), parent.capsule_id());
        assert_eq!(child.parent_capsule_id(), Some(parent.capsule_id()));
        assert_eq!(child.payload(), parent.payload());
        assert_eq!(child.header().integrity, parent.header().integrity);
        assert_eq!(child.header().bytes, parent.header().bytes);
        // The fork is itself a valid, integrity-consistent capsule.
        let round = Capsule::from_bytes(&child.to_bytes()).unwrap();
        assert_eq!(round, child);
    }

    #[test]
    fn inspect_returns_metadata_without_payload() {
        let c = sample((0u8..50).collect());
        let meta = c.inspect();
        assert_eq!(meta.header, *c.header());
        assert_eq!(meta.identity, *c.identity());
        // The inspect view records the payload length but carries no payload.
        assert_eq!(meta.header.bytes, c.payload().len() as u64);
        // inspect_bytes over the serialized form agrees.
        let from_bytes = Capsule::inspect_bytes(&c.to_bytes()).unwrap();
        assert_eq!(from_bytes, meta);
    }

    #[test]
    fn release_reports_reclaimed_bytes() {
        let c = sample(vec![7u8; 256]);
        assert_eq!(c.release(), 256);
    }

    #[test]
    fn is_loadable_delegates_to_identity() {
        let c = sample(vec![1]);
        assert!(c.is_loadable(&identity()).is_ok());
        let mut live = identity();
        live.security_domain = "other".into();
        assert!(matches!(
            c.is_loadable(&live),
            Err(IncompatibleReason::SecurityDomain { .. })
        ));
    }

    #[test]
    fn sha256_sealed_capsule_roundtrips() {
        let c = CapsuleBuilder::new(CapsuleType::Kv, "m", identity())
            .integrity_algo(IntegrityAlgo::Sha256)
            .seal(vec![3u8; 300]);
        assert_eq!(c.header().integrity.algo, IntegrityAlgo::Sha256);
        let back = Capsule::from_bytes(&c.to_bytes()).unwrap();
        assert_eq!(c, back);
    }
}
