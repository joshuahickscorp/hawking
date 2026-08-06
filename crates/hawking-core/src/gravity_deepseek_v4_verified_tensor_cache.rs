//! Bounded authenticated source-native tensor-slice cache for a future
//! DeepSeek-V4 Metal executor.
//!
//! The full-stream reader deliberately re-hashes every touched chunk on every
//! read.  That is the right default for a diagnostic reader, but a real
//! executor must retain already-authenticated static controls without silently
//! materialising the 149 GiB parent stream.  This module is the narrow storage
//! seam between those two requirements:
//!
//! - every miss is satisfied only by
//!   [`DeepSeekV4FullStreamReader::read_verified_range`], so every touched
//!   content-addressed chunk is checked before bytes enter the cache;
//! - every hit is bound to the admitted artifact identity and carries a
//!   SHA-256 of the retained source-native slice;
//! - capacity and entry limits are explicit and fail closed; and
//! - eviction is deterministic LRU with observable counters.
//!
//! It has no Metal allocation, device upload, routing, execution, Engine,
//! serving, HCLI, or TPS surface.  An executor may use the returned immutable
//! bytes as an upload source only after recording its own device-residency
//! receipt.  Routed FP4 experts keep using the separate two-tier expert cache;
//! this cache is for bounded controls, rows, and head tiles.

use std::collections::{BTreeMap, VecDeque};
use std::ops::Range;
use std::sync::Arc;

use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{DeepSeekV4FullStreamReader, DeepSeekV4SourceIdentity};
use crate::{Error, Result};

const GIB: usize = 1024 * 1024 * 1024;

/// Hard upper bound for one process-local control/tile cache.  It is far below
/// the full source stream and forces a future runtime to spell out any larger
/// residency plan instead of accidentally retaining the parent model.
pub const MAX_VERIFIED_TENSOR_CACHE_BYTES: usize = 12 * GIB;
/// No one cache entry may exceed this limit.  Large vocabulary/head tensors
/// must be requested as bounded tiles rather than as an implicit full read.
pub const MAX_VERIFIED_TENSOR_CACHE_ENTRY_BYTES: usize = 64 * 1024 * 1024;

/// Explicit memory limits for [`DeepSeekV4VerifiedTensorCache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4VerifiedTensorCacheConfig {
    pub capacity_bytes: usize,
    pub max_entry_bytes: usize,
}

impl Default for DeepSeekV4VerifiedTensorCacheConfig {
    fn default() -> Self {
        Self {
            // Enough for several source-native attention controls while
            // remaining deliberately below the full model by orders of
            // magnitude.  A runtime must opt in to a larger (still capped)
            // residency budget.
            capacity_bytes: 512 * 1024 * 1024,
            max_entry_bytes: MAX_VERIFIED_TENSOR_CACHE_ENTRY_BYTES,
        }
    }
}

/// Immutable identity of the admitted artifact allowed to fill this cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4VerifiedTensorCacheArtifactBinding {
    pub artifact_root: String,
    pub manifest_seal_sha256: String,
    pub manifest_file_sha256: String,
    pub restart_receipt_seal_sha256: String,
    pub source_repository: String,
    pub source_revision: String,
}

impl DeepSeekV4VerifiedTensorCacheArtifactBinding {
    fn from_reader(reader: &DeepSeekV4FullStreamReader) -> Self {
        let DeepSeekV4SourceIdentity {
            repository,
            revision,
        } = reader.source_identity();
        Self {
            artifact_root: reader.artifact_root().display().to_string(),
            manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            manifest_file_sha256: reader.manifest_file_sha256().to_owned(),
            restart_receipt_seal_sha256: reader.restart_seal_sha256().to_owned(),
            source_repository: repository.clone(),
            source_revision: revision.clone(),
        }
    }

    fn matches_reader(&self, reader: &DeepSeekV4FullStreamReader) -> bool {
        self == &Self::from_reader(reader)
    }
}

/// Public identity and geometry of one immutable cache entry.  `payload_sha256`
/// is calculated over exactly the returned native byte range, not decoded
/// values or a source-parent file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4VerifiedTensorSliceBinding {
    pub artifact: DeepSeekV4VerifiedTensorCacheArtifactBinding,
    pub tensor_name: String,
    pub tensor_dtype: String,
    pub tensor_shape: Vec<u64>,
    pub tensor_bytes: u64,
    pub range_start: u64,
    pub range_end: u64,
    pub payload_sha256: String,
}

impl DeepSeekV4VerifiedTensorSliceBinding {
    pub fn payload_bytes(&self) -> u64 {
        self.range_end - self.range_start
    }
}

/// An immutable, source-native authenticated slice.  The raw payload is kept
/// private to prevent callers from relabelling it; `bytes()` gives a read-only
/// upload-ready view.
#[derive(Debug, Clone)]
pub struct DeepSeekV4VerifiedTensorSlice {
    binding: DeepSeekV4VerifiedTensorSliceBinding,
    bytes: Arc<[u8]>,
}

impl DeepSeekV4VerifiedTensorSlice {
    pub fn binding(&self) -> &DeepSeekV4VerifiedTensorSliceBinding {
        &self.binding
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SliceKey {
    tensor_name: String,
    range_start: u64,
    range_end: u64,
}

impl SliceKey {
    fn from_request(name: &str, range: &Range<u64>) -> Self {
        Self {
            tensor_name: name.to_owned(),
            range_start: range.start,
            range_end: range.end,
        }
    }
}

/// Monotonic cache accounting.  A `verified_source_reads` increment means a
/// reader miss completed successfully; hits never claim a fresh source hash.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeepSeekV4VerifiedTensorCacheCounters {
    pub requests: u64,
    pub hits: u64,
    pub misses: u64,
    pub verified_source_reads: u64,
    pub source_payload_bytes: u64,
    pub evictions: u64,
    pub evicted_bytes: u64,
}

/// A stable, metadata-only snapshot suitable for a future runtime receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4VerifiedTensorCacheState {
    pub artifact: DeepSeekV4VerifiedTensorCacheArtifactBinding,
    pub capacity_bytes: usize,
    pub max_entry_bytes: usize,
    pub resident_bytes: usize,
    pub entries_lru_to_mru: Vec<DeepSeekV4VerifiedTensorSliceBinding>,
    pub counters: DeepSeekV4VerifiedTensorCacheCounters,
}

/// Whether an acquire reused retained authenticated bytes or performed a fresh
/// verified reader operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4VerifiedTensorCacheResult {
    Hit,
    VerifiedSourceRead,
}

impl DeepSeekV4VerifiedTensorCacheResult {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Hit => "hit",
            Self::VerifiedSourceRead => "verified_source_read",
        }
    }
}

/// Result of one cache acquire.  The caller can retain the returned `Arc`
/// independently of the cache, but cache accounting counts only cache-owned
/// entries.  A real executor should release its upload source once a GPU copy
/// is complete rather than using this as a second unbounded model store.
#[derive(Debug, Clone)]
pub struct DeepSeekV4VerifiedTensorCacheAccess {
    pub result: DeepSeekV4VerifiedTensorCacheResult,
    pub slice: DeepSeekV4VerifiedTensorSlice,
    pub state_after: DeepSeekV4VerifiedTensorCacheState,
}

/// Byte-bounded LRU of verified source-native tensor ranges.
#[derive(Debug)]
pub struct DeepSeekV4VerifiedTensorCache {
    binding: DeepSeekV4VerifiedTensorCacheArtifactBinding,
    config: DeepSeekV4VerifiedTensorCacheConfig,
    resident_bytes: usize,
    entries: BTreeMap<SliceKey, DeepSeekV4VerifiedTensorSlice>,
    lru: VecDeque<SliceKey>,
    counters: DeepSeekV4VerifiedTensorCacheCounters,
}

impl DeepSeekV4VerifiedTensorCache {
    /// Construct an empty cache bound to this exact admitted artifact.
    pub fn new(
        reader: &DeepSeekV4FullStreamReader,
        config: DeepSeekV4VerifiedTensorCacheConfig,
    ) -> Result<Self> {
        validate_config(config)?;
        Ok(Self {
            binding: DeepSeekV4VerifiedTensorCacheArtifactBinding::from_reader(reader),
            config,
            resident_bytes: 0,
            entries: BTreeMap::new(),
            lru: VecDeque::new(),
            counters: DeepSeekV4VerifiedTensorCacheCounters::default(),
        })
    }

    pub fn artifact_binding(&self) -> &DeepSeekV4VerifiedTensorCacheArtifactBinding {
        &self.binding
    }

    pub const fn config(&self) -> DeepSeekV4VerifiedTensorCacheConfig {
        self.config
    }

    pub const fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }

    pub const fn counters(&self) -> DeepSeekV4VerifiedTensorCacheCounters {
        self.counters
    }

    pub fn state(&self) -> DeepSeekV4VerifiedTensorCacheState {
        DeepSeekV4VerifiedTensorCacheState {
            artifact: self.binding.clone(),
            capacity_bytes: self.config.capacity_bytes,
            max_entry_bytes: self.config.max_entry_bytes,
            resident_bytes: self.resident_bytes,
            entries_lru_to_mru: self
                .lru
                .iter()
                .filter_map(|key| self.entries.get(key))
                .map(|entry| entry.binding.clone())
                .collect(),
            counters: self.counters,
        }
    }

    /// Return an immutable verified native range.  A miss reads only the
    /// caller-requested range and requires the reader to hash every touched
    /// content-addressed source chunk before caching it.
    pub fn acquire(
        &mut self,
        reader: &DeepSeekV4FullStreamReader,
        name: &str,
        range: Range<u64>,
    ) -> Result<DeepSeekV4VerifiedTensorCacheAccess> {
        self.require_same_artifact(reader)?;
        let metadata = reader.tensor_metadata(name)?;
        validate_request(metadata.bytes, &range, self.config.max_entry_bytes, name)?;
        self.counters.requests =
            increment(self.counters.requests, "verified tensor cache requests")?;
        let key = SliceKey::from_request(name, &range);
        if let Some(entry) = self.entries.get(&key).cloned() {
            self.touch(&key)?;
            self.counters.hits = increment(self.counters.hits, "verified tensor cache hits")?;
            self.assert_invariants()?;
            return Ok(DeepSeekV4VerifiedTensorCacheAccess {
                result: DeepSeekV4VerifiedTensorCacheResult::Hit,
                slice: entry,
                state_after: self.state(),
            });
        }

        let payload_len = usize::try_from(range.end - range.start)
            .map_err(|_| cache_error("verified tensor cache range exceeds host usize"))?;
        self.ensure_space(payload_len)?;
        // This is intentionally the only load path.  It both bounds the host
        // allocation and verifies all source chunks touched by the slice.
        let raw = reader.read_verified_range(name, range.clone(), self.config.max_entry_bytes)?;
        if raw.len() != payload_len {
            return Err(cache_error(
                "verified reader returned a range with unexpected byte count",
            ));
        }
        let binding = DeepSeekV4VerifiedTensorSliceBinding {
            artifact: self.binding.clone(),
            tensor_name: name.to_owned(),
            tensor_dtype: metadata.dtype.clone(),
            tensor_shape: metadata.shape.clone(),
            tensor_bytes: metadata.bytes,
            range_start: range.start,
            range_end: range.end,
            payload_sha256: sha256_hex(&raw),
        };
        let slice = DeepSeekV4VerifiedTensorSlice {
            binding,
            bytes: Arc::<[u8]>::from(raw),
        };
        if self.entries.insert(key.clone(), slice.clone()).is_some() {
            return Err(cache_error(
                "verified tensor cache key was unexpectedly occupied",
            ));
        }
        self.lru.push_back(key);
        self.resident_bytes = self
            .resident_bytes
            .checked_add(payload_len)
            .ok_or_else(|| cache_error("verified tensor cache resident byte overflow"))?;
        self.counters.misses = increment(self.counters.misses, "verified tensor cache misses")?;
        self.counters.verified_source_reads = increment(
            self.counters.verified_source_reads,
            "verified tensor cache source reads",
        )?;
        self.counters.source_payload_bytes = self
            .counters
            .source_payload_bytes
            .checked_add(payload_len as u64)
            .ok_or_else(|| cache_error("verified tensor cache source byte counter overflow"))?;
        self.assert_invariants()?;
        Ok(DeepSeekV4VerifiedTensorCacheAccess {
            result: DeepSeekV4VerifiedTensorCacheResult::VerifiedSourceRead,
            slice,
            state_after: self.state(),
        })
    }

    /// Explicitly evict every cache-owned entry.  Existing returned `Arc`s
    /// remain valid immutable verified bytes, so this does not revoke an
    /// in-flight upload; it only releases the cache's ownership.
    pub fn clear(&mut self) -> Result<()> {
        let evicted = self.resident_bytes;
        let count = self.entries.len() as u64;
        self.entries.clear();
        self.lru.clear();
        self.resident_bytes = 0;
        self.counters.evictions = self
            .counters
            .evictions
            .checked_add(count)
            .ok_or_else(|| cache_error("verified tensor cache eviction counter overflow"))?;
        self.counters.evicted_bytes = self
            .counters
            .evicted_bytes
            .checked_add(evicted as u64)
            .ok_or_else(|| cache_error("verified tensor cache evicted byte counter overflow"))?;
        self.assert_invariants()
    }

    fn require_same_artifact(&self, reader: &DeepSeekV4FullStreamReader) -> Result<()> {
        if !self.binding.matches_reader(reader) {
            return Err(cache_error(
                "verified tensor cache refuses a reader with a different artifact/source identity",
            ));
        }
        Ok(())
    }

    fn ensure_space(&mut self, incoming_bytes: usize) -> Result<()> {
        if incoming_bytes > self.config.capacity_bytes {
            return Err(cache_error(format!(
                "verified tensor cache entry {incoming_bytes} bytes exceeds its {}-byte capacity",
                self.config.capacity_bytes
            )));
        }
        while self.resident_bytes.saturating_add(incoming_bytes) > self.config.capacity_bytes {
            let oldest = self.lru.pop_front().ok_or_else(|| {
                cache_error("verified tensor cache has resident bytes but no LRU entry")
            })?;
            let removed = self.entries.remove(&oldest).ok_or_else(|| {
                cache_error("verified tensor cache LRU entry disappeared from map")
            })?;
            let bytes = removed.bytes.len();
            self.resident_bytes = self
                .resident_bytes
                .checked_sub(bytes)
                .ok_or_else(|| cache_error("verified tensor cache resident byte underflow"))?;
            self.counters.evictions =
                increment(self.counters.evictions, "verified tensor cache evictions")?;
            self.counters.evicted_bytes = self
                .counters
                .evicted_bytes
                .checked_add(bytes as u64)
                .ok_or_else(|| {
                    cache_error("verified tensor cache evicted byte counter overflow")
                })?;
        }
        Ok(())
    }

    fn touch(&mut self, key: &SliceKey) -> Result<()> {
        let position = self
            .lru
            .iter()
            .position(|existing| existing == key)
            .ok_or_else(|| cache_error("verified tensor cache hit is absent from LRU"))?;
        let removed = self
            .lru
            .remove(position)
            .ok_or_else(|| cache_error("verified tensor cache LRU remove failed"))?;
        self.lru.push_back(removed);
        Ok(())
    }

    fn assert_invariants(&self) -> Result<()> {
        let bytes = self.entries.values().try_fold(0usize, |sum, entry| {
            sum.checked_add(entry.bytes.len())
                .ok_or_else(|| cache_error("verified tensor cache byte sum overflow"))
        })?;
        if bytes != self.resident_bytes || self.resident_bytes > self.config.capacity_bytes {
            return Err(cache_error(
                "verified tensor cache byte accounting is inconsistent",
            ));
        }
        if self.entries.len() != self.lru.len()
            || self.lru.iter().any(|key| !self.entries.contains_key(key))
        {
            return Err(cache_error(
                "verified tensor cache map/LRU entries are inconsistent",
            ));
        }
        if self.counters.hits.saturating_add(self.counters.misses) != self.counters.requests
            || self.counters.verified_source_reads != self.counters.misses
        {
            return Err(cache_error(
                "verified tensor cache counter accounting is inconsistent",
            ));
        }
        Ok(())
    }
}

fn validate_config(config: DeepSeekV4VerifiedTensorCacheConfig) -> Result<()> {
    if config.capacity_bytes == 0 || config.capacity_bytes > MAX_VERIFIED_TENSOR_CACHE_BYTES {
        return Err(cache_error(format!(
            "verified tensor cache capacity must be within 1..={MAX_VERIFIED_TENSOR_CACHE_BYTES} bytes",
        )));
    }
    if config.max_entry_bytes == 0
        || config.max_entry_bytes > MAX_VERIFIED_TENSOR_CACHE_ENTRY_BYTES
        || config.max_entry_bytes > config.capacity_bytes
    {
        return Err(cache_error(format!(
            "verified tensor cache entry limit must be within 1..={} bytes and no larger than capacity",
            MAX_VERIFIED_TENSOR_CACHE_ENTRY_BYTES,
        )));
    }
    Ok(())
}

fn validate_request(
    tensor_bytes: u64,
    range: &Range<u64>,
    max_entry_bytes: usize,
    name: &str,
) -> Result<()> {
    if range.start >= range.end || range.end > tensor_bytes {
        return Err(cache_error(format!(
            "{name}: cache range {}..{} is outside source tensor bytes {tensor_bytes}",
            range.start, range.end
        )));
    }
    let bytes = usize::try_from(range.end - range.start)
        .map_err(|_| cache_error(format!("{name}: cache range exceeds host usize")))?;
    if bytes > max_entry_bytes {
        return Err(cache_error(format!(
            "{name}: cache range {bytes} bytes exceeds explicit entry limit {max_entry_bytes}; tile the source tensor",
        )));
    }
    Ok(())
}

fn increment(value: u64, label: &str) -> Result<u64> {
    value
        .checked_add(1)
        .ok_or_else(|| cache_error(format!("{label} counter overflow")))
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn cache_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 verified tensor cache: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_rejects_unbounded_or_full_stream_sized_limits() {
        assert!(validate_config(DeepSeekV4VerifiedTensorCacheConfig {
            capacity_bytes: 0,
            max_entry_bytes: 1,
        })
        .is_err());
        assert!(validate_config(DeepSeekV4VerifiedTensorCacheConfig {
            capacity_bytes: MAX_VERIFIED_TENSOR_CACHE_BYTES + 1,
            max_entry_bytes: 1,
        })
        .is_err());
        assert!(validate_config(DeepSeekV4VerifiedTensorCacheConfig {
            capacity_bytes: 1024,
            max_entry_bytes: 1025,
        })
        .is_err());
    }

    #[test]
    fn request_requires_nonempty_bounded_tile() {
        assert!(validate_request(128, &(0..0), 64, "x").is_err());
        assert!(validate_request(128, &(32..129), 128, "x").is_err());
        assert!(validate_request(128, &(0..65), 64, "x").is_err());
        assert!(validate_request(128, &(64..128), 64, "x").is_ok());
    }

    #[test]
    fn lru_eviction_releases_oldest_cache_owned_slice() {
        let artifact = DeepSeekV4VerifiedTensorCacheArtifactBinding {
            artifact_root: "/immutable/artifact".to_owned(),
            manifest_seal_sha256: "a".repeat(64),
            manifest_file_sha256: "b".repeat(64),
            restart_receipt_seal_sha256: "c".repeat(64),
            source_repository: "deepseek-ai/DeepSeek-V4-Flash".to_owned(),
            source_revision: "r".repeat(40),
        };
        let config = DeepSeekV4VerifiedTensorCacheConfig {
            capacity_bytes: 8,
            max_entry_bytes: 8,
        };
        let mut cache = DeepSeekV4VerifiedTensorCache {
            binding: artifact.clone(),
            config,
            resident_bytes: 8,
            entries: BTreeMap::new(),
            lru: VecDeque::new(),
            counters: DeepSeekV4VerifiedTensorCacheCounters {
                requests: 1,
                misses: 1,
                verified_source_reads: 1,
                source_payload_bytes: 8,
                ..DeepSeekV4VerifiedTensorCacheCounters::default()
            },
        };
        for (name, bytes) in [("old", [1u8; 4]), ("new", [2u8; 4])] {
            let key = SliceKey {
                tensor_name: name.to_owned(),
                range_start: 0,
                range_end: 4,
            };
            cache.lru.push_back(key.clone());
            cache.entries.insert(
                key,
                DeepSeekV4VerifiedTensorSlice {
                    binding: DeepSeekV4VerifiedTensorSliceBinding {
                        artifact: artifact.clone(),
                        tensor_name: name.to_owned(),
                        tensor_dtype: "BF16".to_owned(),
                        tensor_shape: vec![2],
                        tensor_bytes: 4,
                        range_start: 0,
                        range_end: 4,
                        payload_sha256: sha256_hex(&bytes),
                    },
                    bytes: Arc::<[u8]>::from(bytes.to_vec()),
                },
            );
        }
        cache.ensure_space(4).expect("evict oldest");
        assert_eq!(cache.resident_bytes, 4);
        assert_eq!(cache.lru.len(), 1);
        assert_eq!(cache.lru.front().unwrap().tensor_name, "new");
        assert_eq!(cache.counters.evictions, 1);
        assert_eq!(cache.counters.evicted_bytes, 4);
        cache
            .assert_invariants()
            .expect("consistent after eviction");
    }
}
