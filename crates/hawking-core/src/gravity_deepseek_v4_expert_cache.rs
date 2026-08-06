//! Bounded source-chunk-backed routed-expert cache for DeepSeek-V4-Flash.
//!
//! This module is deliberately a storage primitive below a future V4 runtime.
//! It consumes only [`DeepSeekV4FullStreamReader`] verified reads, and holds
//! source-native FP4 `w1` / `w2` / `w3` plus their E8M0 scale tensors in a
//! byte-bounded hot/cold cache.  It has no router, model forward, GPU resource,
//! command buffer, engine, HCLI, or TPS surface.
//!
//! The cache resolves only the exact 43-layer / 256-routed-expert V4 body and
//! rejects a source pair whose tensor names, representation, dimensions, or
//! scale geometry do not match the pinned source contract.  Every cold read is
//! a `DeepSeekV4FullStreamReader::read_verified_full` call, so every declared
//! source chunk touched by a cache fill is regular/non-symlink-admitted and
//! SHA-256 checked before its bytes are retained.

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4Segment, NativeScalePair, NativeScalePairKind,
    PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::{Error, Result};
use std::collections::{BTreeMap, VecDeque};

/// Exact count of base DeepSeek-V4 transformer layers in the admitted stream.
pub const DSV4F_LAYER_COUNT: u16 = 43;
/// Exact number of routed experts in each base DeepSeek-V4 layer.
pub const DSV4F_ROUTED_EXPERT_COUNT: u16 = 256;

/// A routed expert location in the immutable 43-layer DeepSeek-V4 body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExpertBundleKey {
    pub layer: u16,
    pub expert: u16,
}

impl ExpertBundleKey {
    pub const fn new(layer: u16, expert: u16) -> Self {
        Self { layer, expert }
    }

    /// Validate the fixed base-body coordinate before it is interpolated into
    /// a source tensor name.
    pub fn validate(self) -> Result<()> {
        if self.layer >= DSV4F_LAYER_COUNT || self.expert >= DSV4F_ROUTED_EXPERT_COUNT {
            return Err(cache_error(format!(
                "routed expert key layer={} expert={} escapes the pinned {}-layer / {}-expert V4 body",
                self.layer, self.expert, DSV4F_LAYER_COUNT, DSV4F_ROUTED_EXPERT_COUNT
            )));
        }
        Ok(())
    }
}

/// One of the three source-native tensors forming a routed FFN expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertOperator {
    W1,
    W2,
    W3,
}

impl ExpertOperator {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::W1 => "w1",
            Self::W2 => "w2",
            Self::W3 => "w3",
        }
    }
}

/// A source chunk that a full-tensor cache fill will read and verify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertSourceChunkPath {
    pub tensor_name: String,
    pub tensor_role: &'static str,
    pub chunk_relpath: String,
    pub chunk_sha256: String,
    pub bytes: u64,
}

/// Exact source-native descriptor for one routed expert operator and scale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertOperatorDescriptor {
    pub operator: ExpertOperator,
    pub weight_name: String,
    pub scale_name: String,
    pub source_shard: String,
    pub representation: NativeScalePairKind,
    pub out_rows: u64,
    pub packed_k: u64,
    pub logical_k: u64,
    pub scale_rows: u64,
    pub scale_cols: u64,
    pub weight_bytes: u64,
    pub scale_bytes: u64,
    pub source_chunk_paths: Vec<ExpertSourceChunkPath>,
}

impl ExpertOperatorDescriptor {
    pub fn payload_bytes(&self) -> u64 {
        self.weight_bytes + self.scale_bytes
    }

    pub fn verified_chunk_bytes(&self) -> u64 {
        self.source_chunk_paths.iter().map(|path| path.bytes).sum()
    }
}

/// Exact source-native layout required for a routed DeepSeek-V4 expert.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBundleDescriptor {
    pub key: ExpertBundleKey,
    pub operators: [ExpertOperatorDescriptor; 3],
    pub payload_bytes: u64,
    pub verified_chunk_bytes_per_fill: u64,
    pub source_chunk_read_count_per_fill: usize,
}

impl ExpertBundleDescriptor {
    pub fn operator(&self, operator: ExpertOperator) -> &ExpertOperatorDescriptor {
        self.operators
            .iter()
            .find(|entry| entry.operator == operator)
            .expect("expert bundle is constructed with w1/w2/w3 exactly once")
    }
}

/// Resolve and validate the three native FP4 pairs for one routed expert.
///
/// This is metadata-only.  It does not read tensor payload bytes or construct
/// a cache.  The returned descriptor records every source chunk a later
/// full-tensor cache fill will read through the admitted reader.
pub fn resolve_expert_bundle(
    reader: &DeepSeekV4FullStreamReader,
    key: ExpertBundleKey,
) -> Result<ExpertBundleDescriptor> {
    key.validate()?;
    validate_reader_identity(reader)?;

    let w1 = resolve_operator(reader, key, OperatorExpectation::W1)?;
    let w2 = resolve_operator(reader, key, OperatorExpectation::W2)?;
    let w3 = resolve_operator(reader, key, OperatorExpectation::W3)?;
    let operators = [w1, w2, w3];
    let payload_bytes = operators.iter().try_fold(0u64, |total, operator| {
        total
            .checked_add(operator.payload_bytes())
            .ok_or_else(|| cache_error("expert bundle payload byte count overflow"))
    })?;
    let verified_chunk_bytes_per_fill = operators.iter().try_fold(0u64, |total, operator| {
        total
            .checked_add(operator.verified_chunk_bytes())
            .ok_or_else(|| cache_error("expert bundle verified chunk byte count overflow"))
    })?;
    let source_chunk_read_count_per_fill = operators
        .iter()
        .map(|operator| operator.source_chunk_paths.len())
        .sum();

    // For a complete tensor read, every segment is a source-chunk read.  The
    // admitted descriptor makes its segments exactly cover the tensor, so the
    // byte accounting must close before a cache can be created.
    if payload_bytes != verified_chunk_bytes_per_fill {
        return Err(cache_error(format!(
            "expert layer={} expert={} has {} payload bytes but {} verified chunk bytes per full cache fill",
            key.layer, key.expert, payload_bytes, verified_chunk_bytes_per_fill
        )));
    }

    Ok(ExpertBundleDescriptor {
        key,
        operators,
        payload_bytes,
        verified_chunk_bytes_per_fill,
        source_chunk_read_count_per_fill,
    })
}

/// The tier which supplied a cache operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertCacheResult {
    DemandHotHit,
    DemandColdHitPromoted,
    DemandSourceLoadedHot,
    PrefetchHotHit,
    PrefetchColdHit,
    PrefetchSourceLoadedCold,
}

impl ExpertCacheResult {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DemandHotHit => "demand_hot_hit",
            Self::DemandColdHitPromoted => "demand_cold_hit_promoted",
            Self::DemandSourceLoadedHot => "demand_source_loaded_hot",
            Self::PrefetchHotHit => "prefetch_hot_hit",
            Self::PrefetchColdHit => "prefetch_cold_hit",
            Self::PrefetchSourceLoadedCold => "prefetch_source_loaded_cold",
        }
    }
}

/// Actual source-I/O accounting emitted only for a cache-fill operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertBundleSourceRead {
    pub key: ExpertBundleKey,
    pub payload_bytes_returned: u64,
    pub verified_chunk_bytes: u64,
    pub source_chunk_read_count: usize,
    pub chunk_paths: Vec<ExpertSourceChunkPath>,
}

/// Monotonic counters for a bounded hot/cold cache instance.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExpertCacheCounters {
    pub demand_requests: u64,
    pub prefetch_requests: u64,
    pub demand_hot_hits: u64,
    pub demand_cold_hits: u64,
    pub demand_misses: u64,
    pub prefetch_hot_hits: u64,
    pub prefetch_cold_hits: u64,
    pub prefetch_misses: u64,
    pub promotions: u64,
    pub hot_demotions: u64,
    pub hot_evictions: u64,
    pub cold_evictions: u64,
    pub demand_source_loads: u64,
    pub prefetch_source_loads: u64,
    pub source_bundle_loads: u64,
    pub source_tensor_reads: u64,
    pub source_chunk_reads: u64,
    pub source_payload_bytes_returned: u64,
    pub source_verified_chunk_bytes: u64,
}

/// A snapshot of bounded cache occupancy after one operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertCacheState {
    pub hot_capacity_bytes: u64,
    pub cold_capacity_bytes: u64,
    pub hot_resident_bytes: u64,
    pub cold_resident_bytes: u64,
    pub hot_keys_lru_to_mru: Vec<ExpertBundleKey>,
    pub cold_keys_lru_to_mru: Vec<ExpertBundleKey>,
    pub counters: ExpertCacheCounters,
}

/// Outcome of one demand or prefetch cache operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertCacheAccess {
    pub key: ExpertBundleKey,
    pub result: ExpertCacheResult,
    pub source_read: Option<ExpertBundleSourceRead>,
    pub state_after: ExpertCacheState,
}

/// Process-resident source-native buffers for one admitted expert bundle.
///
/// The bytes are intentionally not decoded, uploaded, or persisted.  A future
/// source-faithful runtime may borrow these exact bytes only after separately
/// implementing the required native FP4 execution path.
#[derive(Debug)]
pub struct CachedExpertBundle {
    descriptor: ExpertBundleDescriptor,
    operators: [CachedExpertOperator; 3],
}

#[derive(Debug)]
struct CachedExpertOperator {
    operator: ExpertOperator,
    weight: Vec<u8>,
    scale: Vec<u8>,
}

impl CachedExpertBundle {
    pub fn descriptor(&self) -> &ExpertBundleDescriptor {
        &self.descriptor
    }

    pub fn payload_bytes(&self) -> u64 {
        self.descriptor.payload_bytes
    }

    /// Borrow the original native packed weight and E8M0 scale bytes.  This
    /// accessor performs no conversion or execution.
    pub fn operator_payload(&self, operator: ExpertOperator) -> Option<(&[u8], &[u8])> {
        self.operators
            .iter()
            .find(|entry| entry.operator == operator)
            .map(|entry| (entry.weight.as_slice(), entry.scale.as_slice()))
    }
}

/// A byte-bounded two-tier routed-expert cache.
///
/// Demand misses fill hot.  Prefetch misses fill cold.  A demand cold hit is
/// promoted to hot; the hot LRU is demoted to cold when that tier can fit a
/// bundle, otherwise dropped as a hot eviction.  Both tiers are bounded before
/// any source read is started.
#[derive(Debug)]
pub struct DeepSeekV4ExpertBundleCache {
    hot_capacity_bytes: u64,
    cold_capacity_bytes: u64,
    hot_bytes: u64,
    cold_bytes: u64,
    hot: BTreeMap<ExpertBundleKey, CachedExpertBundle>,
    cold: BTreeMap<ExpertBundleKey, CachedExpertBundle>,
    hot_lru: VecDeque<ExpertBundleKey>,
    cold_lru: VecDeque<ExpertBundleKey>,
    counters: ExpertCacheCounters,
}

impl DeepSeekV4ExpertBundleCache {
    /// Construct an empty cache with explicit byte capacities.  Capacity is
    /// checked against each exact resolved bundle before a read can begin.
    pub fn new(hot_capacity_bytes: u64, cold_capacity_bytes: u64) -> Result<Self> {
        if hot_capacity_bytes == 0 {
            return Err(cache_error("expert hot cache capacity must be non-zero"));
        }
        Ok(Self {
            hot_capacity_bytes,
            cold_capacity_bytes,
            hot_bytes: 0,
            cold_bytes: 0,
            hot: BTreeMap::new(),
            cold: BTreeMap::new(),
            hot_lru: VecDeque::new(),
            cold_lru: VecDeque::new(),
            counters: ExpertCacheCounters::default(),
        })
    }

    pub fn counters(&self) -> ExpertCacheCounters {
        self.counters
    }

    pub fn state(&self) -> ExpertCacheState {
        ExpertCacheState {
            hot_capacity_bytes: self.hot_capacity_bytes,
            cold_capacity_bytes: self.cold_capacity_bytes,
            hot_resident_bytes: self.hot_bytes,
            cold_resident_bytes: self.cold_bytes,
            hot_keys_lru_to_mru: self.hot_lru.iter().copied().collect(),
            cold_keys_lru_to_mru: self.cold_lru.iter().copied().collect(),
            counters: self.counters,
        }
    }

    /// Acquire a routed expert for a future decode step.  This remains storage
    /// only: no router result, token, computation, GPU upload, or command
    /// submission is performed here.
    pub fn acquire(
        &mut self,
        reader: &DeepSeekV4FullStreamReader,
        key: ExpertBundleKey,
    ) -> Result<ExpertCacheAccess> {
        let descriptor = resolve_expert_bundle(reader, key)?;
        self.counters.demand_requests =
            increment(self.counters.demand_requests, "demand requests")?;

        let (result, source_read) = if self.hot.contains_key(&key) {
            self.counters.demand_hot_hits =
                increment(self.counters.demand_hot_hits, "demand hot hits")?;
            touch_lru(&mut self.hot_lru, key);
            (ExpertCacheResult::DemandHotHit, None)
        } else if self.cold.contains_key(&key) {
            self.ensure_hot_bundle_fits(descriptor.payload_bytes)?;
            self.counters.demand_cold_hits =
                increment(self.counters.demand_cold_hits, "demand cold hits")?;
            self.counters.promotions = increment(self.counters.promotions, "promotions")?;
            let bundle = self.remove_cold(key)?;
            self.insert_hot(bundle)?;
            (ExpertCacheResult::DemandColdHitPromoted, None)
        } else {
            self.ensure_hot_bundle_fits(descriptor.payload_bytes)?;
            self.counters.demand_misses = increment(self.counters.demand_misses, "demand misses")?;
            let (bundle, source_read) = materialize_bundle(reader, descriptor)?;
            self.record_source_load(&source_read, false)?;
            self.insert_hot(bundle)?;
            (ExpertCacheResult::DemandSourceLoadedHot, Some(source_read))
        };
        self.assert_invariants()?;
        Ok(ExpertCacheAccess {
            key,
            result,
            source_read,
            state_after: self.state(),
        })
    }

    /// Begin a bounded source fill before an eventual future route selection.
    /// A prefetch has no routing semantics; its caller is responsible for any
    /// predictor and must inspect these counters for hit accuracy later.
    pub fn prefetch(
        &mut self,
        reader: &DeepSeekV4FullStreamReader,
        key: ExpertBundleKey,
    ) -> Result<ExpertCacheAccess> {
        let descriptor = resolve_expert_bundle(reader, key)?;
        self.counters.prefetch_requests =
            increment(self.counters.prefetch_requests, "prefetch requests")?;

        let (result, source_read) = if self.hot.contains_key(&key) {
            self.counters.prefetch_hot_hits =
                increment(self.counters.prefetch_hot_hits, "prefetch hot hits")?;
            touch_lru(&mut self.hot_lru, key);
            (ExpertCacheResult::PrefetchHotHit, None)
        } else if self.cold.contains_key(&key) {
            self.counters.prefetch_cold_hits =
                increment(self.counters.prefetch_cold_hits, "prefetch cold hits")?;
            touch_lru(&mut self.cold_lru, key);
            (ExpertCacheResult::PrefetchColdHit, None)
        } else {
            self.ensure_cold_bundle_fits(descriptor.payload_bytes)?;
            self.counters.prefetch_misses =
                increment(self.counters.prefetch_misses, "prefetch misses")?;
            let (bundle, source_read) = materialize_bundle(reader, descriptor)?;
            self.record_source_load(&source_read, true)?;
            self.insert_cold(bundle)?;
            (
                ExpertCacheResult::PrefetchSourceLoadedCold,
                Some(source_read),
            )
        };
        self.assert_invariants()?;
        Ok(ExpertCacheAccess {
            key,
            result,
            source_read,
            state_after: self.state(),
        })
    }

    /// Borrow a currently resident buffer without loading, decoding, or
    /// changing LRU state.  This is useful for a later runtime adapter to bind
    /// explicit cache residency to its own command topology.
    pub fn resident(&self, key: ExpertBundleKey) -> Option<&CachedExpertBundle> {
        self.hot.get(&key).or_else(|| self.cold.get(&key))
    }

    /// Check that byte counts, maps, and LRU queues remain one-to-one and
    /// below their explicit capacities.
    pub fn assert_invariants(&self) -> Result<()> {
        if self.hot_bytes > self.hot_capacity_bytes || self.cold_bytes > self.cold_capacity_bytes {
            return Err(cache_error(
                "expert cache resident bytes exceed configured tier capacity",
            ));
        }
        let hot_sum = self.hot.values().try_fold(0u64, |total, bundle| {
            total
                .checked_add(bundle.payload_bytes())
                .ok_or_else(|| cache_error("expert hot cache byte total overflow"))
        })?;
        let cold_sum = self.cold.values().try_fold(0u64, |total, bundle| {
            total
                .checked_add(bundle.payload_bytes())
                .ok_or_else(|| cache_error("expert cold cache byte total overflow"))
        })?;
        if hot_sum != self.hot_bytes || cold_sum != self.cold_bytes {
            return Err(cache_error(
                "expert cache map byte total differs from tracked bytes",
            ));
        }
        validate_lru("hot", &self.hot, &self.hot_lru)?;
        validate_lru("cold", &self.cold, &self.cold_lru)?;
        if self.hot.keys().any(|key| self.cold.contains_key(key)) {
            return Err(cache_error(
                "expert cache key is resident in both hot and cold tiers",
            ));
        }
        Ok(())
    }

    fn ensure_hot_bundle_fits(&self, bytes: u64) -> Result<()> {
        if bytes > self.hot_capacity_bytes {
            return Err(cache_error(format!(
                "expert bundle has {bytes} bytes but hot capacity is {} bytes",
                self.hot_capacity_bytes
            )));
        }
        Ok(())
    }

    fn ensure_cold_bundle_fits(&self, bytes: u64) -> Result<()> {
        if bytes > self.cold_capacity_bytes {
            return Err(cache_error(format!(
                "expert bundle has {bytes} bytes but cold capacity is {} bytes",
                self.cold_capacity_bytes
            )));
        }
        Ok(())
    }

    fn record_source_load(
        &mut self,
        source_read: &ExpertBundleSourceRead,
        prefetch: bool,
    ) -> Result<()> {
        self.counters.source_bundle_loads =
            increment(self.counters.source_bundle_loads, "source bundle loads")?;
        if prefetch {
            self.counters.prefetch_source_loads =
                increment(self.counters.prefetch_source_loads, "prefetch source loads")?;
        } else {
            self.counters.demand_source_loads =
                increment(self.counters.demand_source_loads, "demand source loads")?;
        }
        self.counters.source_tensor_reads =
            checked_add(self.counters.source_tensor_reads, 6, "source tensor reads")?;
        self.counters.source_chunk_reads = checked_add(
            self.counters.source_chunk_reads,
            u64::try_from(source_read.source_chunk_read_count)
                .map_err(|_| cache_error("source chunk read count exceeds u64"))?,
            "source chunk reads",
        )?;
        self.counters.source_payload_bytes_returned = checked_add(
            self.counters.source_payload_bytes_returned,
            source_read.payload_bytes_returned,
            "source payload bytes returned",
        )?;
        self.counters.source_verified_chunk_bytes = checked_add(
            self.counters.source_verified_chunk_bytes,
            source_read.verified_chunk_bytes,
            "source verified chunk bytes",
        )?;
        Ok(())
    }

    fn remove_cold(&mut self, key: ExpertBundleKey) -> Result<CachedExpertBundle> {
        let bundle = self
            .cold
            .remove(&key)
            .ok_or_else(|| cache_error("cold cache lookup disappeared during promotion"))?;
        self.cold_bytes = self
            .cold_bytes
            .checked_sub(bundle.payload_bytes())
            .ok_or_else(|| cache_error("cold cache byte underflow"))?;
        remove_lru(&mut self.cold_lru, key)?;
        Ok(bundle)
    }

    fn insert_hot(&mut self, bundle: CachedExpertBundle) -> Result<()> {
        let key = bundle.descriptor.key;
        let bytes = bundle.payload_bytes();
        self.ensure_hot_bundle_fits(bytes)?;
        if self.hot.contains_key(&key) || self.cold.contains_key(&key) {
            return Err(cache_error(
                "attempted to insert duplicate routed expert cache key",
            ));
        }
        while self.hot_bytes.saturating_add(bytes) > self.hot_capacity_bytes {
            let lru = self
                .hot_lru
                .pop_front()
                .ok_or_else(|| cache_error("hot cache had bytes but no LRU entry"))?;
            let displaced = self
                .hot
                .remove(&lru)
                .ok_or_else(|| cache_error("hot LRU entry had no cached bundle"))?;
            self.hot_bytes = self
                .hot_bytes
                .checked_sub(displaced.payload_bytes())
                .ok_or_else(|| cache_error("hot cache byte underflow"))?;
            if displaced.payload_bytes() <= self.cold_capacity_bytes {
                self.counters.hot_demotions =
                    increment(self.counters.hot_demotions, "hot demotions")?;
                self.insert_cold(displaced)?;
            } else {
                self.counters.hot_evictions =
                    increment(self.counters.hot_evictions, "hot evictions")?;
            }
        }
        self.hot_bytes = checked_add(self.hot_bytes, bytes, "hot cache resident bytes")?;
        if self.hot.insert(key, bundle).is_some() {
            return Err(cache_error("hot cache duplicate insertion after precheck"));
        }
        self.hot_lru.push_back(key);
        Ok(())
    }

    fn insert_cold(&mut self, bundle: CachedExpertBundle) -> Result<()> {
        let key = bundle.descriptor.key;
        let bytes = bundle.payload_bytes();
        self.ensure_cold_bundle_fits(bytes)?;
        if self.hot.contains_key(&key) || self.cold.contains_key(&key) {
            return Err(cache_error(
                "attempted to insert duplicate routed expert cache key",
            ));
        }
        while self.cold_bytes.saturating_add(bytes) > self.cold_capacity_bytes {
            let lru = self
                .cold_lru
                .pop_front()
                .ok_or_else(|| cache_error("cold cache had bytes but no LRU entry"))?;
            let evicted = self
                .cold
                .remove(&lru)
                .ok_or_else(|| cache_error("cold LRU entry had no cached bundle"))?;
            self.cold_bytes = self
                .cold_bytes
                .checked_sub(evicted.payload_bytes())
                .ok_or_else(|| cache_error("cold cache byte underflow"))?;
            self.counters.cold_evictions =
                increment(self.counters.cold_evictions, "cold evictions")?;
        }
        self.cold_bytes = checked_add(self.cold_bytes, bytes, "cold cache resident bytes")?;
        if self.cold.insert(key, bundle).is_some() {
            return Err(cache_error("cold cache duplicate insertion after precheck"));
        }
        self.cold_lru.push_back(key);
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
enum OperatorExpectation {
    W1,
    W2,
    W3,
}

impl OperatorExpectation {
    const fn operator(self) -> ExpertOperator {
        match self {
            Self::W1 => ExpertOperator::W1,
            Self::W2 => ExpertOperator::W2,
            Self::W3 => ExpertOperator::W3,
        }
    }

    const fn expected_geometry(self) -> (u64, u64, u64, u64, u64) {
        match self {
            // Gate/up project 4096 hidden channels to 2048 expert channels.
            Self::W1 | Self::W3 => (2048, 2048, 4096, 2048, 128),
            // Down projects 2048 expert channels back to 4096 hidden channels.
            Self::W2 => (4096, 1024, 2048, 4096, 64),
        }
    }
}

fn resolve_operator(
    reader: &DeepSeekV4FullStreamReader,
    key: ExpertBundleKey,
    expected: OperatorExpectation,
) -> Result<ExpertOperatorDescriptor> {
    let operator = expected.operator();
    let stem = format!(
        "layers.{}.ffn.experts.{}.{}",
        key.layer,
        key.expert,
        operator.as_str()
    );
    let expected_weight_name = format!("{stem}.weight");
    let expected_scale_name = format!("{stem}.scale");
    let pair = reader.native_scale_pair(&expected_weight_name)?;
    validate_fp4_pair(&pair, &expected_weight_name, &expected_scale_name, expected)?;
    let source_chunk_paths = source_paths(&pair, &expected_weight_name, &expected_scale_name)?;
    let (out_rows, packed_k, logical_k, scale_rows, scale_cols) = expected.expected_geometry();
    Ok(ExpertOperatorDescriptor {
        operator,
        weight_name: expected_weight_name,
        scale_name: expected_scale_name,
        source_shard: pair.weight.source_shard.clone(),
        representation: pair.kind,
        out_rows,
        packed_k,
        logical_k,
        scale_rows,
        scale_cols,
        weight_bytes: pair.weight.bytes,
        scale_bytes: pair.scale.bytes,
        source_chunk_paths,
    })
}

fn validate_reader_identity(reader: &DeepSeekV4FullStreamReader) -> Result<()> {
    if reader.source_identity().repository != PINNED_REPOSITORY
        || reader.source_identity().revision != PINNED_REVISION
    {
        return Err(cache_error(
            "expert cache requires the pinned DeepSeek-V4-Flash full-stream reader identity",
        ));
    }
    Ok(())
}

fn validate_fp4_pair(
    pair: &NativeScalePair<'_>,
    expected_weight_name: &str,
    expected_scale_name: &str,
    expected: OperatorExpectation,
) -> Result<()> {
    let (out_rows, packed_k, logical_k, scale_rows, scale_cols) = expected.expected_geometry();
    let expected_weight_bytes = out_rows
        .checked_mul(packed_k)
        .ok_or_else(|| cache_error("expected FP4 weight bytes overflow"))?;
    let expected_scale_bytes = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| cache_error("expected E8M0 scale bytes overflow"))?;
    if pair.kind != NativeScalePairKind::Fp4E2M1fnX2
        || pair.weight.name != expected_weight_name
        || pair.scale.name != expected_scale_name
        || pair.weight.dtype != "I8"
        || pair.scale.dtype != "F8_E8M0"
        || pair.weight.shape.as_slice() != [out_rows, packed_k]
        || pair.scale.shape.as_slice() != [scale_rows, scale_cols]
        || pair.out_rows != out_rows
        || pair.packed_k != packed_k
        || pair.logical_k != logical_k
        || pair.scale_rows != scale_rows
        || pair.scale_cols != scale_cols
        || pair.weight.bytes != expected_weight_bytes
        || pair.scale.bytes != expected_scale_bytes
        || pair.weight.source_shard != pair.scale.source_shard
        || pair.weight.segments.is_empty()
        || pair.scale.segments.is_empty()
    {
        return Err(cache_error(format!(
            "{} does not match the exact native DeepSeek-V4 routed-expert FP4 pair geometry",
            expected_weight_name
        )));
    }
    Ok(())
}

fn source_paths(
    pair: &NativeScalePair<'_>,
    weight_name: &str,
    scale_name: &str,
) -> Result<Vec<ExpertSourceChunkPath>> {
    let mut paths = Vec::with_capacity(pair.weight.segments.len() + pair.scale.segments.len());
    extend_source_paths(&mut paths, &pair.weight.segments, weight_name, "weight")?;
    extend_source_paths(&mut paths, &pair.scale.segments, scale_name, "scale")?;
    Ok(paths)
}

fn extend_source_paths(
    output: &mut Vec<ExpertSourceChunkPath>,
    segments: &[DeepSeekV4Segment],
    tensor_name: &str,
    tensor_role: &'static str,
) -> Result<()> {
    for segment in segments {
        if segment.bytes == 0 || segment.chunk_relpath.is_empty() || segment.sha256.len() != 64 {
            return Err(cache_error(format!(
                "{tensor_name}: admitted source chunk path is unexpectedly incomplete"
            )));
        }
        output.push(ExpertSourceChunkPath {
            tensor_name: tensor_name.to_owned(),
            tensor_role,
            chunk_relpath: segment.chunk_relpath.clone(),
            chunk_sha256: segment.sha256.clone(),
            bytes: segment.bytes,
        });
    }
    Ok(())
}

fn materialize_bundle(
    reader: &DeepSeekV4FullStreamReader,
    descriptor: ExpertBundleDescriptor,
) -> Result<(CachedExpertBundle, ExpertBundleSourceRead)> {
    let mut operator_payloads = Vec::with_capacity(3);
    let mut returned_bytes = 0u64;
    for operator in &descriptor.operators {
        let weight_bound = usize::try_from(operator.weight_bytes)
            .map_err(|_| cache_error("expert weight byte bound exceeds host usize"))?;
        let scale_bound = usize::try_from(operator.scale_bytes)
            .map_err(|_| cache_error("expert scale byte bound exceeds host usize"))?;
        let weight = reader.read_verified_full(&operator.weight_name, weight_bound)?;
        let scale = reader.read_verified_full(&operator.scale_name, scale_bound)?;
        if weight.len() as u64 != operator.weight_bytes
            || scale.len() as u64 != operator.scale_bytes
        {
            return Err(cache_error(format!(
                "{} returned an unexpected native payload length",
                operator.weight_name
            )));
        }
        returned_bytes = checked_add(
            returned_bytes,
            operator.payload_bytes(),
            "materialized expert payload bytes",
        )?;
        operator_payloads.push(CachedExpertOperator {
            operator: operator.operator,
            weight,
            scale,
        });
    }
    let operators: [CachedExpertOperator; 3] = operator_payloads
        .try_into()
        .map_err(|_| cache_error("expert cache materialization did not produce w1/w2/w3"))?;
    if returned_bytes != descriptor.payload_bytes {
        return Err(cache_error(
            "expert cache returned-byte accounting does not close",
        ));
    }
    let chunk_paths = descriptor
        .operators
        .iter()
        .flat_map(|operator| operator.source_chunk_paths.iter().cloned())
        .collect::<Vec<_>>();
    let verified_chunk_bytes = chunk_paths.iter().try_fold(0u64, |total, path| {
        total
            .checked_add(path.bytes)
            .ok_or_else(|| cache_error("expert cache verified chunk-byte accounting overflow"))
    })?;
    if verified_chunk_bytes != descriptor.verified_chunk_bytes_per_fill
        || chunk_paths.len() != descriptor.source_chunk_read_count_per_fill
    {
        return Err(cache_error(
            "expert cache chunk path accounting does not close",
        ));
    }
    Ok((
        CachedExpertBundle {
            descriptor: descriptor.clone(),
            operators,
        },
        ExpertBundleSourceRead {
            key: descriptor.key,
            payload_bytes_returned: returned_bytes,
            verified_chunk_bytes,
            source_chunk_read_count: chunk_paths.len(),
            chunk_paths,
        },
    ))
}

fn touch_lru(lru: &mut VecDeque<ExpertBundleKey>, key: ExpertBundleKey) {
    if let Some(position) = lru.iter().position(|item| *item == key) {
        lru.remove(position);
    }
    lru.push_back(key);
}

fn remove_lru(lru: &mut VecDeque<ExpertBundleKey>, key: ExpertBundleKey) -> Result<()> {
    let position = lru
        .iter()
        .position(|item| *item == key)
        .ok_or_else(|| cache_error("cache map entry had no matching LRU entry"))?;
    lru.remove(position);
    Ok(())
}

fn validate_lru(
    tier: &str,
    map: &BTreeMap<ExpertBundleKey, CachedExpertBundle>,
    lru: &VecDeque<ExpertBundleKey>,
) -> Result<()> {
    if map.len() != lru.len() {
        return Err(cache_error(format!("{tier} cache map/LRU length mismatch")));
    }
    for key in lru {
        if !map.contains_key(key) || lru.iter().filter(|candidate| *candidate == key).count() != 1 {
            return Err(cache_error(format!(
                "{tier} cache LRU contains an unknown or duplicate key"
            )));
        }
    }
    Ok(())
}

fn increment(value: u64, label: &str) -> Result<u64> {
    checked_add(value, 1, label)
}

fn checked_add(value: u64, additional: u64, label: &str) -> Result<u64> {
    value
        .checked_add(additional)
        .ok_or_else(|| cache_error(format!("{label} overflow")))
}

fn cache_error(message: impl Into<String>) -> Error {
    Error::Gravity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gravity_deepseek_v4::DeepSeekV4TensorMetadata;

    fn metadata(name: &str, dtype: &str, shape: &[u64], bytes: u64) -> DeepSeekV4TensorMetadata {
        DeepSeekV4TensorMetadata {
            name: name.to_owned(),
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            data_offsets: [0, bytes],
            bytes,
            source_file_start: 0,
            source_file_end: bytes,
            source_shard: "model-00001-of-00046.safetensors".to_owned(),
            segments: vec![DeepSeekV4Segment {
                bytes,
                chunk_relpath: format!("chunks/aa/{}", "a".repeat(64)),
                sha256: "a".repeat(64),
                source_file_start: 0,
                source_file_end: bytes,
                tensor_start: 0,
                tensor_end: bytes,
                row_start: 0,
                row_count: shape[0],
            }],
        }
    }

    #[test]
    fn routed_key_rejects_out_of_body_coordinates() {
        assert!(ExpertBundleKey::new(42, 255).validate().is_ok());
        assert!(ExpertBundleKey::new(43, 0).validate().is_err());
        assert!(ExpertBundleKey::new(0, 256).validate().is_err());
    }

    #[test]
    fn fp4_pair_validation_rejects_wrong_native_geometry() {
        let weight = metadata(
            "layers.0.ffn.experts.0.w1.weight",
            "I8",
            &[2048, 2048],
            2048 * 2048,
        );
        let scale = metadata(
            "layers.0.ffn.experts.0.w1.scale",
            "F8_E8M0",
            &[2048, 128],
            2048 * 128,
        );
        let good = NativeScalePair {
            kind: NativeScalePairKind::Fp4E2M1fnX2,
            weight: &weight,
            scale: &scale,
            out_rows: 2048,
            packed_k: 2048,
            logical_k: 4096,
            scale_rows: 2048,
            scale_cols: 128,
        };
        assert!(validate_fp4_pair(
            &good,
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
            OperatorExpectation::W1,
        )
        .is_ok());

        let wrong = NativeScalePair {
            logical_k: 2048,
            ..good
        };
        assert!(validate_fp4_pair(
            &wrong,
            "layers.0.ffn.experts.0.w1.weight",
            "layers.0.ffn.experts.0.w1.scale",
            OperatorExpectation::W1,
        )
        .is_err());
    }
}
