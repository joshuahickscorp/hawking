//! Bounded execution-context scaffold for the future DeepSeek-V4 causal loop.
//!
//! It owns source-backed control staging, an explicit mHC state slot, 43
//! explicit KV/state-slot layouts, a bounded routed-expert cache, and a
//! deterministic command-graph ledger. It intentionally has no Engine,
//! Metal encoder, forward, sampling, HCLI, or TPS surface.

use std::collections::{BTreeMap, VecDeque};

use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4_expert_cache::{
    resolve_expert_bundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey, ExpertCacheAccess,
    ExpertCacheState, ExpertOperator,
};
use crate::gravity_deepseek_v4_layer_source_anchors::{
    verify_deepseek_v4_layer_source_anchors, DeepSeekV4LayerSourceAnchors,
};
use crate::gravity_deepseek_v4_runtime_spine::{
    DeepSeekV4CompressionMode, DeepSeekV4ControlProjection, DeepSeekV4ExpertProjection,
    DeepSeekV4RuntimeSpine, DeepSeekV4StagedNativePair, DeepSeekV4StagedTensor,
    DSV4F_BASE_LAYER_COUNT, DSV4F_HC_MULT, DSV4F_HIDDEN_SIZE, DSV4F_TOP_K_EXPERTS,
    MAX_STAGED_OPERATOR_BYTES, PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES,
    PROVISIONAL_ROUTED_EXPERT_COLD_CEILING_BYTES, PROVISIONAL_ROUTED_EXPERT_HOT_CEILING_BYTES,
};
use crate::gravity_deepseek_v4_verified_tensor_cache::{
    DeepSeekV4VerifiedTensorCache, DeepSeekV4VerifiedTensorCacheConfig,
    DeepSeekV4VerifiedTensorCacheCounters,
};
use crate::{Error, Result};

pub const DSV4F_KV_WINDOW_TOKENS: usize = 128;
pub const DSV4F_KV_HEAD_DIM: usize = 512;
pub const DEFAULT_COMMAND_LEDGER_CAPACITY: usize = 512;

/// Capacities are contracts only: opening this context does not allocate a
/// model-sized buffer or upload anything to Metal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ExecutionContextConfig {
    pub max_context_tokens: usize,
    pub control_staging_capacity_bytes: usize,
    pub control_resident_ceiling_bytes: u64,
    pub routed_expert_hot_capacity_bytes: u64,
    pub routed_expert_cold_capacity_bytes: u64,
    pub command_ledger_capacity: usize,
    /// When `Some`, control tensor staging uses the authenticated verified
    /// tensor cache so multi-layer loops do not re-stream/re-verify static
    /// controls already held. `None` preserves the historical direct-read path.
    pub verified_tensor_cache: Option<DeepSeekV4VerifiedTensorCacheConfig>,
}

impl Default for DeepSeekV4ExecutionContextConfig {
    fn default() -> Self {
        Self {
            max_context_tokens: 8 * 1024,
            control_staging_capacity_bytes: MAX_STAGED_OPERATOR_BYTES,
            control_resident_ceiling_bytes: PROVISIONAL_CONTROL_RESIDENT_CEILING_BYTES,
            routed_expert_hot_capacity_bytes: PROVISIONAL_ROUTED_EXPERT_HOT_CEILING_BYTES,
            routed_expert_cold_capacity_bytes: PROVISIONAL_ROUTED_EXPERT_COLD_CEILING_BYTES,
            command_ledger_capacity: DEFAULT_COMMAND_LEDGER_CAPACITY,
            // Default on: multi-layer staging must not re-verify every control.
            verified_tensor_cache: Some(DeepSeekV4VerifiedTensorCacheConfig::default()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4MhcBranch {
    Attention,
    Ffn,
}

impl DeepSeekV4MhcBranch {
    const fn stem(self) -> &'static str {
        match self {
            Self::Attention => "hc_attn",
            Self::Ffn => "hc_ffn",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Attention => "attention",
            Self::Ffn => "ffn",
        }
    }
}

/// A source-shaped top-6 set supplied by a future router. Constructing it is
/// not routing; the context uses it only for bounded FP4 bundle acquisition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4SelectedRouteSet {
    pub experts: [u16; DSV4F_TOP_K_EXPERTS],
}

impl DeepSeekV4SelectedRouteSet {
    pub fn new(experts: [u16; DSV4F_TOP_K_EXPERTS]) -> Result<Self> {
        let mut seen = [false; 256];
        for expert in experts {
            let index = expert as usize;
            if index >= seen.len() || seen[index] {
                return Err(context_error(format!(
                    "route set must contain six unique expert ids in 0..256; found {expert}"
                )));
            }
            seen[index] = true;
        }
        Ok(Self { experts })
    }
}

/// Raw source bytes held by the bounded control arena.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepSeekV4ControlPayload {
    EmbeddingRow {
        token_id: u32,
        bf16_bits: Vec<u16>,
    },
    Tensor(DeepSeekV4StagedTensor),
    NativePair(DeepSeekV4StagedNativePair),
    MhcControl {
        layer: usize,
        branch: DeepSeekV4MhcBranch,
        tensors: [DeepSeekV4StagedTensor; 3],
    },
}

impl DeepSeekV4ControlPayload {
    fn bytes(&self) -> usize {
        match self {
            Self::EmbeddingRow { bf16_bits, .. } => bf16_bits.len() * std::mem::size_of::<u16>(),
            Self::Tensor(value) => value.bytes.len(),
            Self::NativePair(value) => value.weight.bytes.len() + value.scale.bytes.len(),
            Self::MhcControl { tensors, .. } => tensors.iter().map(|value| value.bytes.len()).sum(),
        }
    }

    fn label(&self) -> String {
        match self {
            Self::EmbeddingRow { token_id, .. } => format!("embed.weight[row={token_id}]"),
            Self::Tensor(value) => value.name.clone(),
            Self::NativePair(value) => format!("{}+{}", value.weight.name, value.scale.name),
            Self::MhcControl { layer, branch, .. } => {
                format!("layers.{layer}.{}", branch.stem())
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct DeepSeekV4ControlLease {
    pub id: u64,
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ControlArenaEntry {
    lease: DeepSeekV4ControlLease,
    payload: DeepSeekV4ControlPayload,
    bytes: usize,
}

/// Deterministic FIFO control staging. This never silently reloads an evicted
/// entry; callers receive a stale-lease error and must explicitly restage.
#[derive(Debug)]
pub struct DeepSeekV4ControlStagingArena {
    capacity_bytes: usize,
    resident_bytes: usize,
    next_id: u64,
    generation: u64,
    entries: BTreeMap<u64, ControlArenaEntry>,
    fifo: VecDeque<u64>,
    eviction_count: u64,
}

impl DeepSeekV4ControlStagingArena {
    pub fn new(capacity_bytes: usize) -> Result<Self> {
        if capacity_bytes == 0 || capacity_bytes > MAX_STAGED_OPERATOR_BYTES {
            return Err(context_error(format!(
                "control staging capacity must be within 1..={MAX_STAGED_OPERATOR_BYTES} bytes"
            )));
        }
        Ok(Self {
            capacity_bytes,
            resident_bytes: 0,
            next_id: 0,
            generation: 0,
            entries: BTreeMap::new(),
            fifo: VecDeque::new(),
            eviction_count: 0,
        })
    }

    pub fn capacity_bytes(&self) -> usize {
        self.capacity_bytes
    }
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
    pub fn eviction_count(&self) -> u64 {
        self.eviction_count
    }

    pub fn insert(&mut self, payload: DeepSeekV4ControlPayload) -> Result<DeepSeekV4ControlLease> {
        let bytes = payload.bytes();
        if bytes == 0 || bytes > self.capacity_bytes {
            return Err(context_error(format!(
                "control payload {} has {bytes} bytes, outside the {}-byte arena",
                payload.label(),
                self.capacity_bytes
            )));
        }
        while self.resident_bytes.saturating_add(bytes) > self.capacity_bytes {
            let oldest = self.fifo.pop_front().ok_or_else(|| {
                context_error("control arena has resident bytes but no FIFO entry")
            })?;
            let removed = self
                .entries
                .remove(&oldest)
                .ok_or_else(|| context_error("control FIFO entry disappeared from arena"))?;
            self.resident_bytes = self
                .resident_bytes
                .checked_sub(removed.bytes)
                .ok_or_else(|| context_error("control arena byte underflow"))?;
            self.eviction_count = self
                .eviction_count
                .checked_add(1)
                .ok_or_else(|| context_error("control arena eviction counter overflow"))?;
        }
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| context_error("control arena lease id overflow"))?;
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| context_error("control arena generation overflow"))?;
        let lease = DeepSeekV4ControlLease {
            id: self.next_id,
            generation: self.generation,
        };
        self.resident_bytes = self
            .resident_bytes
            .checked_add(bytes)
            .ok_or_else(|| context_error("control arena byte overflow"))?;
        if self
            .entries
            .insert(
                lease.id,
                ControlArenaEntry {
                    lease,
                    payload,
                    bytes,
                },
            )
            .is_some()
        {
            return Err(context_error("control arena lease collision"));
        }
        self.fifo.push_back(lease.id);
        self.assert_invariants()?;
        Ok(lease)
    }

    pub fn get(&self, lease: DeepSeekV4ControlLease) -> Result<&DeepSeekV4ControlPayload> {
        let entry = self
            .entries
            .get(&lease.id)
            .ok_or_else(|| context_error(format!("control lease {} has been evicted", lease.id)))?;
        if entry.lease != lease {
            return Err(context_error(format!(
                "control lease {} generation is stale",
                lease.id
            )));
        }
        Ok(&entry.payload)
    }

    pub fn assert_invariants(&self) -> Result<()> {
        let sum = self.entries.values().try_fold(0usize, |total, entry| {
            total
                .checked_add(entry.bytes)
                .ok_or_else(|| context_error("control arena byte sum overflow"))
        })?;
        if sum != self.resident_bytes || self.resident_bytes > self.capacity_bytes {
            return Err(context_error(
                "control arena byte accounting is inconsistent",
            ));
        }
        if self.entries.len() != self.fifo.len()
            || self.fifo.iter().any(|id| !self.entries.contains_key(id))
        {
            return Err(context_error("control arena entries/FIFO are inconsistent"));
        }
        Ok(())
    }
}

/// Host-side, exact BF16 mHC seed state. It is initialized as four copies of
/// an authenticated embedding row, exactly matching source HC expansion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4MhcStateSlot {
    pub copies: usize,
    pub hidden_size: usize,
    pub bf16_bits: Vec<u16>,
    pub initialized: bool,
}

impl DeepSeekV4MhcStateSlot {
    fn new() -> Self {
        Self {
            copies: DSV4F_HC_MULT,
            hidden_size: DSV4F_HIDDEN_SIZE,
            bf16_bits: vec![0; DSV4F_HC_MULT * DSV4F_HIDDEN_SIZE],
            initialized: false,
        }
    }

    fn seed_embedding(&mut self, embedding: &[u16]) -> Result<()> {
        if embedding.len() != DSV4F_HIDDEN_SIZE {
            return Err(context_error(
                "embedding row width is not the source hidden size",
            ));
        }
        for copy in self.bf16_bits.chunks_exact_mut(DSV4F_HIDDEN_SIZE) {
            copy.copy_from_slice(embedding);
        }
        self.initialized = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.bf16_bits.fill(0);
        self.initialized = false;
    }
}

/// Explicit source-shaped KV slot. Its physical device representation is
/// deliberately unallocated until the attention/KV parity rung establishes it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4KvStateSlot {
    pub layer: usize,
    pub compression: DeepSeekV4CompressionMode,
    pub sliding_window_tokens: usize,
    pub compressed_tokens_capacity: usize,
    pub logical_value_width: usize,
    pub logical_bf16_bytes_ceiling: usize,
    pub storage_allocated: bool,
    pub writes_completed: usize,
}

impl DeepSeekV4KvStateSlot {
    fn new(
        layer: usize,
        compression: DeepSeekV4CompressionMode,
        max_context: usize,
    ) -> Result<Self> {
        let compressed_tokens_capacity = match compression.ratio() {
            0 => 0,
            ratio => max_context / ratio,
        };
        let token_capacity = DSV4F_KV_WINDOW_TOKENS
            .checked_add(compressed_tokens_capacity)
            .ok_or_else(|| context_error("KV token capacity overflow"))?;
        let logical_bf16_bytes_ceiling = token_capacity
            .checked_mul(DSV4F_KV_HEAD_DIM)
            .and_then(|value| value.checked_mul(std::mem::size_of::<u16>()))
            .ok_or_else(|| context_error("KV byte capacity overflow"))?;
        Ok(Self {
            layer,
            compression,
            sliding_window_tokens: DSV4F_KV_WINDOW_TOKENS,
            compressed_tokens_capacity,
            logical_value_width: DSV4F_KV_HEAD_DIM,
            logical_bf16_bytes_ceiling,
            storage_allocated: false,
            writes_completed: 0,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4DecodeState {
    pub position: usize,
    pub max_context_tokens: usize,
    pub m_hc: DeepSeekV4MhcStateSlot,
    pub kv_slots: Vec<DeepSeekV4KvStateSlot>,
}

impl DeepSeekV4DecodeState {
    fn new(max_context_tokens: usize, spine: &DeepSeekV4RuntimeSpine) -> Result<Self> {
        let kv_slots = spine
            .topology()
            .layers
            .iter()
            .map(|binding| {
                DeepSeekV4KvStateSlot::new(binding.layer, binding.compression, max_context_tokens)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            position: 0,
            max_context_tokens,
            m_hc: DeepSeekV4MhcStateSlot::new(),
            kv_slots,
        })
    }

    fn reset(&mut self) {
        self.position = 0;
        self.m_hc.reset();
        for slot in &mut self.kv_slots {
            slot.storage_allocated = false;
            slot.writes_completed = 0;
        }
    }
}

/// Logical future operations, explicitly distinct from actual GPU dispatches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4CommandNodeKind {
    EmbedAndMhcSeed,
    MhcAttentionPre,
    AttentionQkvAndNorm,
    AttentionIndexAndKv,
    AttentionOutputAndResidual,
    MhcFfnPre,
    RouterTop6,
    RoutedExpertWave,
    SharedExpertAndCombine,
    MhcFfnResidual,
    FinalMhcHeadAndNorm,
    LmHeadTopKSampling,
    ReadbackAndHcliStream,
}

impl DeepSeekV4CommandNodeKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EmbedAndMhcSeed => "embed_and_mhc_seed",
            Self::MhcAttentionPre => "mhc_attention_pre",
            Self::AttentionQkvAndNorm => "attention_qkv_and_norm",
            Self::AttentionIndexAndKv => "attention_index_and_kv",
            Self::AttentionOutputAndResidual => "attention_output_and_residual",
            Self::MhcFfnPre => "mhc_ffn_pre",
            Self::RouterTop6 => "router_top6",
            Self::RoutedExpertWave => "routed_expert_wave",
            Self::SharedExpertAndCombine => "shared_expert_and_combine",
            Self::MhcFfnResidual => "mhc_ffn_residual",
            Self::FinalMhcHeadAndNorm => "final_mhc_head_and_norm",
            Self::LmHeadTopKSampling => "lm_head_top_k_sampling",
            Self::ReadbackAndHcliStream => "readback_and_hcli_stream",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4CommandGraphNode {
    pub ordinal: usize,
    pub layer: Option<usize>,
    pub kind: DeepSeekV4CommandNodeKind,
    pub depends_on: Option<usize>,
    pub logical_submission_group: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4TokenCommandGraph {
    pub position: usize,
    pub nodes: Vec<DeepSeekV4CommandGraphNode>,
    pub graph_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepSeekV4CommandLedgerEventKind {
    DecodeInputPrepared,
    ControlPayloadStaged,
    RoutedExpertAccess,
    FullForwardDenied,
}

impl DeepSeekV4CommandLedgerEventKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::DecodeInputPrepared => "decode_input_prepared",
            Self::ControlPayloadStaged => "control_payload_staged",
            Self::RoutedExpertAccess => "routed_expert_access",
            Self::FullForwardDenied => "full_forward_denied",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4CommandLedgerEvent {
    pub sequence: u64,
    pub token_position: usize,
    pub kind: DeepSeekV4CommandLedgerEventKind,
    pub label: String,
    pub graph_sha256: Option<String>,
    pub planned_nodes: usize,
    pub actual_command_buffers: usize,
    pub actual_gpu_dispatches: usize,
    pub actual_cpu_visible_waits: usize,
}

#[derive(Debug)]
pub struct DeepSeekV4CommandGraphLedger {
    capacity: usize,
    next_sequence: u64,
    events: VecDeque<DeepSeekV4CommandLedgerEvent>,
    evicted_events: u64,
}

impl DeepSeekV4CommandGraphLedger {
    pub fn new(capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(context_error("command ledger capacity must be non-zero"));
        }
        Ok(Self {
            capacity,
            next_sequence: 0,
            events: VecDeque::new(),
            evicted_events: 0,
        })
    }
    pub fn events(&self) -> &VecDeque<DeepSeekV4CommandLedgerEvent> {
        &self.events
    }
    pub fn evicted_events(&self) -> u64 {
        self.evicted_events
    }

    fn record(
        &mut self,
        position: usize,
        kind: DeepSeekV4CommandLedgerEventKind,
        label: impl Into<String>,
        graph: Option<&DeepSeekV4TokenCommandGraph>,
    ) -> Result<()> {
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| context_error("ledger sequence overflow"))?;
        if self.events.len() == self.capacity {
            self.events.pop_front();
            self.evicted_events = self
                .evicted_events
                .checked_add(1)
                .ok_or_else(|| context_error("ledger eviction overflow"))?;
        }
        self.events.push_back(DeepSeekV4CommandLedgerEvent {
            sequence: self.next_sequence,
            token_position: position,
            kind,
            label: label.into(),
            graph_sha256: graph.map(|value| value.graph_sha256.clone()),
            planned_nodes: graph.map(|value| value.nodes.len()).unwrap_or(0),
            actual_command_buffers: 0,
            actual_gpu_dispatches: 0,
            actual_cpu_visible_waits: 0,
        });
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4PreparedDecodeInput {
    pub token_id: u32,
    pub position: usize,
    pub embedding_lease: DeepSeekV4ControlLease,
    pub command_graph: DeepSeekV4TokenCommandGraph,
}

/// This type is intentionally not an Engine. It is the direct host for future
/// Metal resources and an actual source-parity-gated causal loop.
pub struct DeepSeekV4ExecutionContext {
    spine: DeepSeekV4RuntimeSpine,
    config: DeepSeekV4ExecutionContextConfig,
    control_arena: DeepSeekV4ControlStagingArena,
    expert_cache: DeepSeekV4ExpertBundleCache,
    /// Compact per-layer source anchors (tensor names/modes for layers 0..42).
    layer_source_anchors: DeepSeekV4LayerSourceAnchors,
    /// Optional authenticated control/tile cache shared across layer stages.
    verified_tensor_cache: Option<DeepSeekV4VerifiedTensorCache>,
    decode_state: DeepSeekV4DecodeState,
    command_ledger: DeepSeekV4CommandGraphLedger,
}

impl DeepSeekV4ExecutionContext {
    pub fn open(
        root: impl AsRef<std::path::Path>,
        config: DeepSeekV4ExecutionContextConfig,
    ) -> Result<Self> {
        Self::from_spine(DeepSeekV4RuntimeSpine::admit(root)?, config)
    }

    pub fn from_spine(
        spine: DeepSeekV4RuntimeSpine,
        config: DeepSeekV4ExecutionContextConfig,
    ) -> Result<Self> {
        validate_context_config(&spine, &config)?;
        let control_arena =
            DeepSeekV4ControlStagingArena::new(config.control_staging_capacity_bytes)?;
        let expert_cache = DeepSeekV4ExpertBundleCache::new(
            config.routed_expert_hot_capacity_bytes,
            config.routed_expert_cold_capacity_bytes,
        )?;
        let layer_source_anchors = verify_deepseek_v4_layer_source_anchors(spine.reader())?;
        let verified_tensor_cache = match config.verified_tensor_cache {
            Some(cache_config) => {
                Some(DeepSeekV4VerifiedTensorCache::new(spine.reader(), cache_config)?)
            }
            None => None,
        };
        let decode_state = DeepSeekV4DecodeState::new(config.max_context_tokens, &spine)?;
        let command_ledger = DeepSeekV4CommandGraphLedger::new(config.command_ledger_capacity)?;
        Ok(Self {
            spine,
            config,
            control_arena,
            expert_cache,
            layer_source_anchors,
            verified_tensor_cache,
            decode_state,
            command_ledger,
        })
    }

    pub fn spine(&self) -> &DeepSeekV4RuntimeSpine {
        &self.spine
    }
    pub fn config(&self) -> &DeepSeekV4ExecutionContextConfig {
        &self.config
    }
    pub fn control_arena(&self) -> &DeepSeekV4ControlStagingArena {
        &self.control_arena
    }
    pub fn expert_cache_state(&self) -> ExpertCacheState {
        self.expert_cache.state()
    }
    /// Compact, verified per-layer tensor anchors for layers 0..42.
    pub fn layer_source_anchors(&self) -> &DeepSeekV4LayerSourceAnchors {
        &self.layer_source_anchors
    }
    /// Counters for the optional verified control/tile cache, if enabled.
    pub fn verified_tensor_cache_counters(&self) -> Option<DeepSeekV4VerifiedTensorCacheCounters> {
        self.verified_tensor_cache
            .as_ref()
            .map(DeepSeekV4VerifiedTensorCache::counters)
    }
    pub fn decode_state(&self) -> &DeepSeekV4DecodeState {
        &self.decode_state
    }
    pub fn command_ledger(&self) -> &DeepSeekV4CommandGraphLedger {
        &self.command_ledger
    }

    /// Performs a real verified embedding-row read and source HC replication.
    /// It does not execute any decoder layer and cannot advance position.
    pub fn prepare_decode_input(&mut self, token_id: u32) -> Result<DeepSeekV4PreparedDecodeInput> {
        if self.decode_state.position >= self.decode_state.max_context_tokens {
            return Err(context_error(
                "decode position reaches configured context capacity",
            ));
        }
        let embedding = self.spine.load_embedding_row_bf16(token_id)?;
        self.decode_state.m_hc.seed_embedding(&embedding)?;
        let lease = self
            .control_arena
            .insert(DeepSeekV4ControlPayload::EmbeddingRow {
                token_id,
                bf16_bits: embedding,
            })?;
        let graph = canonical_token_graph(self.decode_state.position, &self.spine);
        self.command_ledger.record(
            self.decode_state.position,
            DeepSeekV4CommandLedgerEventKind::DecodeInputPrepared,
            format!("token={token_id} lease={}", lease.id),
            Some(&graph),
        )?;
        Ok(DeepSeekV4PreparedDecodeInput {
            token_id,
            position: self.decode_state.position,
            embedding_lease: lease,
            command_graph: graph,
        })
    }

    pub fn stage_mhc_control(
        &mut self,
        layer: usize,
        branch: DeepSeekV4MhcBranch,
    ) -> Result<DeepSeekV4ControlLease> {
        self.spine.topology().layer(layer)?;
        let stem = branch.stem();
        let tensors = [
            self.stage_full_control_tensor(&format!("layers.{layer}.{stem}_fn"))?,
            self.stage_full_control_tensor(&format!("layers.{layer}.{stem}_base"))?,
            self.stage_full_control_tensor(&format!("layers.{layer}.{stem}_scale"))?,
        ];
        let lease = self
            .control_arena
            .insert(DeepSeekV4ControlPayload::MhcControl {
                layer,
                branch,
                tensors,
            })?;
        self.record_control_stage(lease)
    }

    pub fn stage_attention_control(
        &mut self,
        layer: usize,
        projection: DeepSeekV4ControlProjection,
    ) -> Result<DeepSeekV4ControlLease> {
        let pair = if self.verified_tensor_cache.is_some() {
            let binding = self.spine.topology().layer(layer)?;
            let weight_name = binding.control_weight_name(projection);
            self.stage_native_pair_cached(&weight_name, crate::gravity_deepseek_v4::NativeScalePairKind::Fp8E4M3fn)?
        } else {
            self.spine.stage_control_pair(layer, projection)?
        };
        let lease = self
            .control_arena
            .insert(DeepSeekV4ControlPayload::NativePair(pair))?;
        self.record_control_stage(lease)
    }

    pub fn stage_shared_expert_control(
        &mut self,
        layer: usize,
        projection: DeepSeekV4ExpertProjection,
    ) -> Result<DeepSeekV4ControlLease> {
        let pair = if self.verified_tensor_cache.is_some() {
            let binding = self.spine.topology().layer(layer)?;
            let weight_name = binding.shared_expert_weight_name(projection);
            self.stage_native_pair_cached(&weight_name, crate::gravity_deepseek_v4::NativeScalePairKind::Fp8E4M3fn)?
        } else {
            self.spine.stage_shared_expert_pair(layer, projection)?
        };
        let lease = self
            .control_arena
            .insert(DeepSeekV4ControlPayload::NativePair(pair))?;
        self.record_control_stage(lease)
    }

    /// Acquires source-native FP4 buffers for an already-selected top-6 set.
    /// It does not calculate router logits or execute the experts.
    pub fn acquire_selected_route_set(
        &mut self,
        layer: usize,
        route_set: DeepSeekV4SelectedRouteSet,
    ) -> Result<Vec<ExpertCacheAccess>> {
        self.spine.topology().layer(layer)?;
        let layer =
            u16::try_from(layer).map_err(|_| context_error("layer does not fit cache key"))?;
        let mut accesses = Vec::with_capacity(DSV4F_TOP_K_EXPERTS);
        for expert in route_set.experts {
            let access = self
                .expert_cache
                .acquire(self.spine.reader(), ExpertBundleKey::new(layer, expert))?;
            self.command_ledger.record(
                self.decode_state.position,
                DeepSeekV4CommandLedgerEventKind::RoutedExpertAccess,
                format!(
                    "layer={layer} expert={expert} result={}",
                    access.result.as_str()
                ),
                None,
            )?;
            accesses.push(access);
        }
        Ok(accesses)
    }

    /// Exact native byte hand-off point for a future FP4 Metal encoder.
    pub fn cached_routed_operator(
        &self,
        layer: usize,
        expert: u16,
        projection: DeepSeekV4ExpertProjection,
    ) -> Result<(&[u8], &[u8])> {
        self.spine.topology().layer(layer)?;
        let key = ExpertBundleKey::new(
            u16::try_from(layer).map_err(|_| context_error("layer does not fit cache key"))?,
            expert,
        );
        key.validate()?;
        let bundle = self
            .expert_cache
            .resident(key)
            .ok_or_else(|| context_error("requested routed expert is not resident"))?;
        let operator = match projection {
            DeepSeekV4ExpertProjection::W1 => ExpertOperator::W1,
            DeepSeekV4ExpertProjection::W2 => ExpertOperator::W2,
            DeepSeekV4ExpertProjection::W3 => ExpertOperator::W3,
        };
        bundle
            .operator_payload(operator)
            .ok_or_else(|| context_error("resident routed expert misses native operator"))
    }

    /// Explicit no-forward gate. It records the denial with zero actual GPU
    /// counters so a planned graph can never be promoted to a runtime claim.
    pub fn require_full_causal_execution(&mut self) -> Result<()> {
        let result = self.spine.capabilities().require_full_causal_runtime();
        if result.is_err() {
            self.command_ledger.record(
                self.decode_state.position,
                DeepSeekV4CommandLedgerEventKind::FullForwardDenied,
                "no registered 43-layer Engine/forward; preparation only",
                None,
            )?;
        }
        result
    }

    pub fn reset_decode_state(&mut self) {
        self.decode_state.reset();
    }

    fn stage_full_control_tensor(&mut self, name: &str) -> Result<DeepSeekV4StagedTensor> {
        let metadata = self.spine.reader().tensor_metadata(name)?;
        let bytes = metadata.bytes;
        let bytes_usize =
            usize::try_from(bytes).map_err(|_| context_error("tensor bytes exceed host usize"))?;
        if bytes_usize > self.control_arena.capacity_bytes() {
            return Err(context_error(format!(
                "{name}: exceeds configured control arena"
            )));
        }
        if let Some(cache) = self.verified_tensor_cache.as_mut() {
            let access = cache.acquire(self.spine.reader(), name, 0..bytes)?;
            let _ = access.result; // Hit or VerifiedSourceRead; both yield verified bytes.
            return Ok(DeepSeekV4StagedTensor {
                name: metadata.name.clone(),
                dtype: metadata.dtype.clone(),
                shape: metadata.shape.clone(),
                source_shard: metadata.source_shard.clone(),
                range: 0..bytes,
                bytes: access.slice.bytes().to_vec(),
            });
        }
        self.spine
            .stage_base_tensor_range(name, 0..bytes, self.control_arena.capacity_bytes())
    }

    fn stage_native_pair_cached(
        &mut self,
        weight_name: &str,
        expected_kind: crate::gravity_deepseek_v4::NativeScalePairKind,
    ) -> Result<DeepSeekV4StagedNativePair> {
        let pair = self.spine.reader().native_scale_pair(weight_name)?;
        if pair.kind != expected_kind {
            return Err(context_error(format!(
                "{weight_name}: native pair kind {} differs from required {}",
                pair.kind.as_str(),
                expected_kind.as_str()
            )));
        }
        let scale_name = pair.scale.name.clone();
        let kind = pair.kind;
        let logical_k = pair.logical_k;
        let out_rows = pair.out_rows;
        let weight = self.stage_full_control_tensor(weight_name)?;
        let scale = self.stage_full_control_tensor(&scale_name)?;
        Ok(DeepSeekV4StagedNativePair {
            kind,
            weight,
            scale,
            logical_k,
            out_rows,
        })
    }

    fn record_control_stage(
        &mut self,
        lease: DeepSeekV4ControlLease,
    ) -> Result<DeepSeekV4ControlLease> {
        let label = self.control_arena.get(lease)?.label();
        self.command_ledger.record(
            self.decode_state.position,
            DeepSeekV4CommandLedgerEventKind::ControlPayloadStaged,
            format!("lease={} {label}", lease.id),
            None,
        )?;
        Ok(lease)
    }
}

fn validate_context_config(
    spine: &DeepSeekV4RuntimeSpine,
    config: &DeepSeekV4ExecutionContextConfig,
) -> Result<()> {
    if config.max_context_tokens == 0 || config.command_ledger_capacity == 0 {
        return Err(context_error(
            "context and command-ledger capacities must be non-zero",
        ));
    }
    if config.control_staging_capacity_bytes == 0
        || config.control_staging_capacity_bytes > MAX_STAGED_OPERATOR_BYTES
    {
        return Err(context_error(
            "control staging capacity escapes the source-bound limit",
        ));
    }
    if config.control_resident_ceiling_bytes == 0
        || config.control_resident_ceiling_bytes
            > spine.residency_plan().control_resident_ceiling_bytes
    {
        return Err(context_error(
            "control resident ceiling escapes sealed plan",
        ));
    }
    if config.routed_expert_hot_capacity_bytes == 0
        || config.routed_expert_hot_capacity_bytes
            > spine.residency_plan().routed_expert_hot_ceiling_bytes
        || config.routed_expert_cold_capacity_bytes
            > spine.residency_plan().routed_expert_cold_ceiling_bytes
    {
        return Err(context_error("routed cache capacity escapes sealed plan"));
    }
    let minimum_bundle =
        resolve_expert_bundle(spine.reader(), ExpertBundleKey::new(0, 0))?.payload_bytes;
    if config.routed_expert_hot_capacity_bytes < minimum_bundle
        || (config.routed_expert_cold_capacity_bytes != 0
            && config.routed_expert_cold_capacity_bytes < minimum_bundle)
    {
        return Err(context_error(format!(
            "routed cache tier is below one exact {minimum_bundle}-byte expert bundle"
        )));
    }
    Ok(())
}

fn canonical_token_graph(
    position: usize,
    spine: &DeepSeekV4RuntimeSpine,
) -> DeepSeekV4TokenCommandGraph {
    let mut nodes = Vec::with_capacity(1 + DSV4F_BASE_LAYER_COUNT * 9 + 3);
    push_node(
        &mut nodes,
        None,
        DeepSeekV4CommandNodeKind::EmbedAndMhcSeed,
        0,
    );
    for binding in &spine.topology().layers {
        let attention_group = 1 + binding.layer * 2;
        let ffn_group = attention_group + 1;
        for kind in [
            DeepSeekV4CommandNodeKind::MhcAttentionPre,
            DeepSeekV4CommandNodeKind::AttentionQkvAndNorm,
            DeepSeekV4CommandNodeKind::AttentionIndexAndKv,
            DeepSeekV4CommandNodeKind::AttentionOutputAndResidual,
        ] {
            push_node(&mut nodes, Some(binding.layer), kind, attention_group);
        }
        for kind in [
            DeepSeekV4CommandNodeKind::MhcFfnPre,
            DeepSeekV4CommandNodeKind::RouterTop6,
            DeepSeekV4CommandNodeKind::RoutedExpertWave,
            DeepSeekV4CommandNodeKind::SharedExpertAndCombine,
            DeepSeekV4CommandNodeKind::MhcFfnResidual,
        ] {
            push_node(&mut nodes, Some(binding.layer), kind, ffn_group);
        }
    }
    let final_group = 1 + DSV4F_BASE_LAYER_COUNT * 2;
    for kind in [
        DeepSeekV4CommandNodeKind::FinalMhcHeadAndNorm,
        DeepSeekV4CommandNodeKind::LmHeadTopKSampling,
        DeepSeekV4CommandNodeKind::ReadbackAndHcliStream,
    ] {
        push_node(&mut nodes, None, kind, final_group);
    }
    let graph_sha256 = graph_hash(position, &nodes);
    DeepSeekV4TokenCommandGraph {
        position,
        nodes,
        graph_sha256,
    }
}

fn push_node(
    nodes: &mut Vec<DeepSeekV4CommandGraphNode>,
    layer: Option<usize>,
    kind: DeepSeekV4CommandNodeKind,
    logical_submission_group: usize,
) {
    let ordinal = nodes.len();
    nodes.push(DeepSeekV4CommandGraphNode {
        ordinal,
        layer,
        kind,
        depends_on: ordinal.checked_sub(1),
        logical_submission_group,
    });
}

fn graph_hash(position: usize, nodes: &[DeepSeekV4CommandGraphNode]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(position.to_le_bytes());
    for node in nodes {
        hasher.update(node.ordinal.to_le_bytes());
        hasher.update(node.layer.unwrap_or(usize::MAX).to_le_bytes());
        hasher.update(node.kind.as_str().as_bytes());
        hasher.update(node.depends_on.unwrap_or(usize::MAX).to_le_bytes());
        hasher.update(node.logical_submission_group.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn context_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!("DeepSeek-V4 execution context: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_tensor(label: &str, bytes: usize) -> DeepSeekV4ControlPayload {
        DeepSeekV4ControlPayload::Tensor(DeepSeekV4StagedTensor {
            name: label.to_owned(),
            dtype: "U8".to_owned(),
            shape: vec![bytes as u64],
            source_shard: "test".to_owned(),
            range: 0..bytes as u64,
            bytes: vec![0; bytes],
        })
    }

    #[test]
    fn route_set_requires_unique_valid_top_six() {
        assert!(DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 5]).is_ok());
        assert!(DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 4]).is_err());
        assert!(DeepSeekV4SelectedRouteSet::new([0, 1, 2, 3, 4, 256]).is_err());
    }

    #[test]
    fn control_arena_evicts_deterministically_and_stale_leases_fail() {
        let mut arena = DeepSeekV4ControlStagingArena::new(8).unwrap();
        let first = arena.insert(tiny_tensor("first", 4)).unwrap();
        let second = arena.insert(tiny_tensor("second", 6)).unwrap();
        assert!(arena.get(first).is_err());
        assert!(arena.get(second).is_ok());
        assert_eq!(arena.resident_bytes(), 6);
        assert_eq!(arena.eviction_count(), 1);
        arena.assert_invariants().unwrap();
    }

    #[test]
    fn mhc_seed_replicates_embedding_across_four_copies() {
        let mut slot = DeepSeekV4MhcStateSlot::new();
        let row: Vec<u16> = (0..DSV4F_HIDDEN_SIZE).map(|value| value as u16).collect();
        slot.seed_embedding(&row).unwrap();
        assert!(slot.initialized);
        for copy in slot.bf16_bits.chunks_exact(DSV4F_HIDDEN_SIZE) {
            assert_eq!(copy, row.as_slice());
        }
    }

    #[test]
    fn kv_layout_tracks_source_compression_ratios_without_allocating() {
        let ratio4 =
            DeepSeekV4KvStateSlot::new(2, DeepSeekV4CompressionMode::Ratio4WithIndexer, 8 * 1024)
                .unwrap();
        let ratio128 =
            DeepSeekV4KvStateSlot::new(3, DeepSeekV4CompressionMode::Ratio128, 8 * 1024).unwrap();
        assert_eq!(ratio4.compressed_tokens_capacity, 2048);
        assert_eq!(ratio128.compressed_tokens_capacity, 64);
        assert!(!ratio4.storage_allocated);
    }
}
