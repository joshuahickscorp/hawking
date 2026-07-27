//! Per-token cost ledger for Temporal Gravity / BASE_RUNTIME_MAXIMIZED.
//!
//! Default-off, additive instrumentation. When enabled, exclusive CPU wall
//! time is attributed across a fixed set of buckets that are required to sum
//! (plus an **explicit unattributed remainder**) to the measured token wall
//! time. A separate **device timeline** records GPU execution and queue wait
//! from Metal command-buffer timestamps so CPU encode, GPU work, and GPU
//! idle-while-waiting stay separable.
//!
//! Enable with `HAWKING_COST_LEDGER=1`, or programmatically via
//! [`set_enabled`] / [`begin_token`]. Disabled paths are a single atomic load
//! and do not allocate.
//!
//! Nesting uses an exclusive stack: entering a child bucket pauses the parent
//! so nested regions never double-count. That is what makes
//! `sum(buckets) + unattributed ≈ wall` a meaningful identity rather than
//! an accounting fiction.
//!
//! ## Hard rule — no catch-all orchestration
//!
//! An unattributed remainder is reported as its own line
//! ([`TokenCostReport::unattributed_us`]) with its own magnitude. It is
//! **never** folded into a neighbour. [`Bucket::CpuOrchestration`] exists only
//! for callers that *explicitly* scope residual glue between named stages
//! (legacy name kept for existing hooks); it is not the wall remainder.
//!
//! ## Hook points (do not invent silent proxies)
//!
//! Call these from the decode path when wiring is available. This module owns
//! the ledger; concurrent GPU-resident-state work should call in from
//! `gravity_glm` without this module owning that file:
//!
//! | Call site | Bucket / API |
//! |---|---|
//! | artifact container lookup | `Scope::new(Bucket::ContainerLookup)` |
//! | SHA / integrity verify | `Scope::new(Bucket::ArtifactVerificationAndSha)` + `record_sha_verification` |
//! | packed index / PQ host decode | `Scope::new(Bucket::PackedIndexDecode)` |
//! | host↔device copy | `Scope::new(Bucket::HostDeviceTransfer)` + `record_transfer` |
//! | Metal encode / submit / wait | `add_duration(Metal*)` + `record_gpu_command_buffer` after wait |
//! | attention + IndexShare (host loops today) | `Scope::new(Bucket::AttentionAndIndexShare)` |
//! | router top-k | `Scope::new(Bucket::Routing)` |
//! | shared / routed experts | `Scope::new(Bucket::SharedExperts)` / `RoutedExperts` |
//! | KV append / state | `Scope::new(Bucket::KvUpdate)` |
//! | RMSNorm / LayerNorm | `Scope::new(Bucket::Norm)` |
//! | final head + sampling | `Scope::new(Bucket::FinalHeadAndSampling)` |
//! | residency snapshot | `record_residency` |
//! | page-fault delta | sampled automatically at begin/end when OS supports it |
//! | active weight bytes / ops | `record_active_bytes` / `record_operations` |
//!
//! ## Device sources
//!
//! | Quantity | Source |
//! |---|---|
//! | exclusive CPU buckets | host `Instant` exclusive stack |
//! | `metal_encode` / `metal_submit` / `metal_synchronize` | host clock around encode / `commit` / `wait_until_completed` |
//! | `gpu_execution_us` | `MTLCommandBuffer.GPUEndTime − GPUStartTime` (CFTimeInterval) |
//! | `gpu_queue_wait_us` | derived: `max(0, host_wait_us − gpu_execution_us)` per CB, summed |
//! | counter-sample timestamps | optional; only when a device exposes `timestamp` counter set **and** a caller encodes sample markers — not substituted with a CPU proxy |
//! | page faults | `getrusage(RUSAGE_SELF)` minflt/majflt delta when available |
//! | unattributed | derived: `wall − sum(buckets)` |
//! | profiler overhead | host clock of ledger enter/exit/add bookkeeping |

use serde::Serialize;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Env var that turns the ledger on for the process (`=1`).
pub const COST_LEDGER_ENV: &str = "HAWKING_COST_LEDGER";

/// How a reported quantity was obtained. Never claim a CPU proxy is a GPU
/// counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MetricSource {
    /// Host `Instant` exclusive-stack or scoped duration.
    CpuClock,
    /// `MTLCommandBuffer` `GPUStartTime` / `GPUEndTime`.
    GpuTimestamp,
    /// Metal counter sample buffer (timestamp common counter set).
    CounterSample,
    /// Arithmetic from other measured quantities.
    Derived,
    /// Explicitly not measured on this path / hardware.
    Unavailable,
}

/// Fixed exclusive time buckets. Order is stable for reports.
///
/// These partition **CPU wall** only. GPU execution lives on
/// [`DeviceTimeline`] and must not be double-counted into exclusive time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Bucket {
    ArtifactVerificationAndSha = 0,
    ContainerLookup = 1,
    PackedIndexDecode = 2,
    /// Explicitly scoped residual CPU glue between named stages.
    /// **Not** the unattributed wall remainder — that is
    /// [`TokenCostReport::unattributed_us`].
    CpuOrchestration = 3,
    HostDeviceTransfer = 4,
    MetalEncode = 5,
    MetalSubmit = 6,
    /// CPU wall spent inside `wait_until_completed` (includes GPU work +
    /// queue delay from the host's perspective). Pair with
    /// [`DeviceTimeline::gpu_execution_us`].
    MetalSynchronize = 7,
    AttentionAndIndexShare = 8,
    Routing = 9,
    SharedExperts = 10,
    RoutedExperts = 11,
    KvUpdate = 12,
    FinalHeadAndSampling = 13,
    /// RMSNorm / LayerNorm exclusive CPU (or device-side when hooked).
    Norm = 14,
}

impl Bucket {
    pub const ALL: [Bucket; 15] = [
        Bucket::ArtifactVerificationAndSha,
        Bucket::ContainerLookup,
        Bucket::PackedIndexDecode,
        Bucket::CpuOrchestration,
        Bucket::HostDeviceTransfer,
        Bucket::MetalEncode,
        Bucket::MetalSubmit,
        Bucket::MetalSynchronize,
        Bucket::AttentionAndIndexShare,
        Bucket::Routing,
        Bucket::SharedExperts,
        Bucket::RoutedExperts,
        Bucket::KvUpdate,
        Bucket::FinalHeadAndSampling,
        Bucket::Norm,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Bucket::ArtifactVerificationAndSha => "artifact_verification_and_sha",
            Bucket::ContainerLookup => "container_lookup",
            Bucket::PackedIndexDecode => "packed_index_decode",
            // Honest name: this is only what callers explicitly scoped.
            Bucket::CpuOrchestration => "cpu_residual_scoped",
            Bucket::HostDeviceTransfer => "host_device_transfer",
            Bucket::MetalEncode => "metal_encode",
            Bucket::MetalSubmit => "metal_submit",
            Bucket::MetalSynchronize => "metal_synchronize_cpu_wait",
            Bucket::AttentionAndIndexShare => "attention_and_indexshare",
            Bucket::Routing => "routing",
            Bucket::SharedExperts => "shared_experts",
            Bucket::RoutedExperts => "routed_experts",
            Bucket::KvUpdate => "kv_update",
            Bucket::FinalHeadAndSampling => "final_head_and_sampling",
            Bucket::Norm => "norm",
        }
    }

    /// Provenance of the exclusive-time series for this bucket.
    pub fn source(self) -> MetricSource {
        MetricSource::CpuClock
    }

    /// Human-readable note for reports / gate docs.
    pub fn source_note(self) -> &'static str {
        match self {
            Bucket::MetalSynchronize => {
                "host Instant around wait_until_completed; not GPU occupancy"
            }
            Bucket::CpuOrchestration => {
                "explicit Scope only; never auto-absorbs unattributed remainder"
            }
            Bucket::MetalEncode => "host Instant around Metal encode path",
            Bucket::MetalSubmit => "host Instant around command buffer commit",
            _ => "host Instant exclusive stack",
        }
    }

    fn index(self) -> usize {
        self as u8 as usize
    }
}

/// One host↔device transfer observed while the ledger is recording a token.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TransferRecord {
    pub bytes: u64,
    /// `true` = host → device, `false` = device → host.
    pub host_to_device: bool,
    pub kind: &'static str,
}

/// One completed Metal command buffer with host and (when available) GPU times.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct GpuCommandBufferSample {
    /// Host wall for `commit` (µs).
    pub host_commit_us: u64,
    /// Host wall for `wait_until_completed` (µs).
    pub host_wait_us: u64,
    /// `GPUEndTime − GPUStartTime` in µs when timestamps were readable.
    pub gpu_execution_us: Option<u64>,
    /// Derived queue / schedule wait: `max(0, host_wait_us − gpu_execution_us)`.
    /// `None` when GPU timestamps were unavailable — **not** filled with a
    /// CPU-only proxy.
    pub gpu_queue_wait_us: Option<u64>,
    /// Raw `GPUStartTime` (CFTimeInterval seconds) when available.
    pub gpu_start_s: Option<f64>,
    /// Raw `GPUEndTime` (CFTimeInterval seconds) when available.
    pub gpu_end_s: Option<f64>,
    pub dispatches_in_buffer: u64,
}

/// Device-side timeline for one token. Independent of the exclusive CPU stack:
/// GPU execution overlaps host `metal_synchronize_cpu_wait`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct DeviceTimeline {
    /// Sum of per-CB GPU execution times (µs). Source: GPU timestamps.
    pub gpu_execution_us: u64,
    /// Sum of per-CB derived queue waits (µs). `None` if **no** CB yielded
    /// GPU timestamps this token.
    pub gpu_queue_wait_us: Option<u64>,
    pub gpu_timestamps_observed: u64,
    pub gpu_timestamps_missing: u64,
    /// Whether the process has probed a Metal timestamp counter set.
    pub counter_sample_probed: bool,
    /// Whether the device exposes the `timestamp` common counter set.
    pub counter_sample_supported: Option<bool>,
    /// Whether any counter-sample markers were actually encoded this token.
    /// Encoding markers is opt-in; absence is reported, not proxied.
    pub counter_samples_recorded: u64,
    pub command_buffers: Vec<GpuCommandBufferSample>,
    pub notes: Vec<&'static str>,
}

impl Default for DeviceTimeline {
    fn default() -> Self {
        Self {
            gpu_execution_us: 0,
            gpu_queue_wait_us: None,
            gpu_timestamps_observed: 0,
            gpu_timestamps_missing: 0,
            counter_sample_probed: false,
            counter_sample_supported: None,
            counter_samples_recorded: 0,
            command_buffers: Vec::new(),
            notes: Vec::new(),
        }
    }
}

/// Where a matvec's active weight bytes came from. Partitions
/// [`TokenCounters::active_bytes_read`] so a 4× "overread vs geometry"
/// figure can be attributed instead of argued.
///
/// Geometry ([`SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES`]) only covers
/// [`ActiveByteCategory::RoutedExperts`] under an 8×3×78 idealisation.
/// Everything else is **required non-geometry traffic**, not necessarily
/// waste — until a category is shown to re-read or widen beyond need.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ActiveByteCategory {
    RoutedExperts,
    SharedExperts,
    DenseMlp,
    Attention,
    Indexer,
    Router,
    LmHead,
    Other,
}

impl ActiveByteCategory {
    pub const ALL: [ActiveByteCategory; 8] = [
        ActiveByteCategory::RoutedExperts,
        ActiveByteCategory::SharedExperts,
        ActiveByteCategory::DenseMlp,
        ActiveByteCategory::Attention,
        ActiveByteCategory::Indexer,
        ActiveByteCategory::Router,
        ActiveByteCategory::LmHead,
        ActiveByteCategory::Other,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            ActiveByteCategory::RoutedExperts => "routed_experts",
            ActiveByteCategory::SharedExperts => "shared_experts",
            ActiveByteCategory::DenseMlp => "dense_mlp",
            ActiveByteCategory::Attention => "attention",
            ActiveByteCategory::Indexer => "indexer",
            ActiveByteCategory::Router => "router",
            ActiveByteCategory::LmHead => "lm_head",
            ActiveByteCategory::Other => "other",
        }
    }

    fn index(self) -> usize {
        match self {
            ActiveByteCategory::RoutedExperts => 0,
            ActiveByteCategory::SharedExperts => 1,
            ActiveByteCategory::DenseMlp => 2,
            ActiveByteCategory::Attention => 3,
            ActiveByteCategory::Indexer => 4,
            ActiveByteCategory::Router => 5,
            ActiveByteCategory::LmHead => 6,
            ActiveByteCategory::Other => 7,
        }
    }
}

/// Classify a weight tensor name into an active-byte category.
///
/// Name grammar follows the sealed GLM MoE artifact (`model.layers.*.…`,
/// `lm_head.weight`). Unknown shapes land in [`ActiveByteCategory::Other`]
/// rather than being forced into a neighbour.
pub fn classify_weight_name(name: &str) -> ActiveByteCategory {
    if name == "lm_head.weight" || name.ends_with("lm_head.weight") {
        return ActiveByteCategory::LmHead;
    }
    if name.contains("shared_experts") {
        return ActiveByteCategory::SharedExperts;
    }
    if name.contains(".experts.") {
        return ActiveByteCategory::RoutedExperts;
    }
    if name.contains(".indexer.") {
        return ActiveByteCategory::Indexer;
    }
    if name.contains("self_attn") {
        return ActiveByteCategory::Attention;
    }
    // Router gate is `…mlp.gate.weight` (no `_proj`); dense/sparse MLP
    // projections are `gate_proj` / `up_proj` / `down_proj`.
    if name.ends_with("mlp.gate.weight") || name.contains("mlp.gate.weight") {
        return ActiveByteCategory::Router;
    }
    if name.contains("gate_proj") || name.contains("up_proj") || name.contains("down_proj") {
        // Dense-layer MLP (no `.experts.` / `shared_experts` above).
        return ActiveByteCategory::DenseMlp;
    }
    ActiveByteCategory::Other
}

/// Counters that usually explain a bandwidth-starved MoE, independent of time.
#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
pub struct TokenCounters {
    pub command_buffers_submitted: u64,
    pub dispatches_encoded: u64,
    /// Every place the CPU waited on the GPU (`wait_until_completed`).
    pub synchronization_points: u64,
    pub host_to_device_bytes: u64,
    pub device_to_host_bytes: u64,
    pub host_to_device_transfers: u64,
    pub device_to_host_transfers: u64,
    /// Heap / Metal buffer allocations observed on the hot path.
    pub allocations: u64,
    pub allocation_bytes: u64,
    /// Weight bytes actually touched for matvec this token (resident or not).
    /// Equal to the sum of [`TokenCounters::active_bytes_by_category`] when
    /// every matvec used [`record_active_bytes_in`].
    pub active_bytes_read: u64,
    /// Partition of [`active_bytes_read`] by tensor class. Keys are stable
    /// snake_case names from [`ActiveByteCategory::as_str`].
    #[serde(default)]
    pub active_bytes_by_category: serde_json::Map<String, serde_json::Value>,
    /// First-touch loads (decode + upload) this token — zero on a warm cache hit.
    pub first_touch_load_bytes: u64,
    pub matvec_calls: u64,
    pub matvec_batch_calls: u64,
    pub matvec_batch_items: u64,
    /// Times `dense()` / `row()` re-entered the artifact (SHA path when verify on).
    pub dense_calls: u64,
    pub row_calls: u64,
    pub sha_verifications: u64,
    /// Abstract operation count (caller-defined units, e.g. FMA or matvec rows).
    pub operations: u64,
    /// Minor page faults this token (delta of `ru_minflt`), when OS supports.
    pub page_faults_minor: Option<u64>,
    /// Major page faults / page-ins this token (delta of `ru_majflt`).
    pub page_faults_major: Option<u64>,
    /// Whether page-fault sampling was available on this platform.
    pub page_faults_available: bool,
    /// GPU weight-cache resident bytes at end of token (if recorded).
    pub residency_bytes: Option<u64>,
    pub residency_entries: Option<u64>,
    pub residency_evictions: Option<u64>,
}

/// Full report for one instrumented decode token.
#[derive(Debug, Clone, Serialize)]
pub struct TokenCostReport {
    pub schema: &'static str,
    pub wall_us: u64,
    /// Exclusive microseconds per bucket. Keys are stable snake_case names.
    pub buckets_us: serde_json::Map<String, serde_json::Value>,
    /// Provenance map: bucket name → source note.
    pub bucket_sources: serde_json::Map<String, serde_json::Value>,
    /// `wall_us - sum(buckets_us)`. An unattributed remainder is a finding —
    /// never absorbed into `cpu_residual_scoped` or any other bucket.
    pub unattributed_us: u64,
    /// Stable name for the unattributed line (hard rule: own name + magnitude).
    pub unattributed_name: &'static str,
    /// Signed residual so over-attribution (instrument bug) is visible.
    pub unattributed_signed_us: i64,
    pub attributed_us: u64,
    pub attributed_fraction: f64,
    pub counters: TokenCounters,
    /// Device-side GPU execution / queue wait (not exclusive-stack).
    pub device: DeviceTimeline,
    /// Geometry the gate quotes: 8 × 3 × 1_378_368 × 78 when known.
    pub geometry_active_bytes: Option<u64>,
    pub active_bytes_vs_geometry_fraction: Option<f64>,
    /// Host+device bytes moved this token vs geometry (informational).
    pub bytes_moved_total: u64,
    pub bytes_moved_vs_geometry_fraction: Option<f64>,
    pub transfers: Vec<TransferRecord>,
    /// Host time spent inside ledger bookkeeping this token (µs).
    pub profiler_overhead_us: u64,
    pub profiler_overhead_fraction: f64,
}

impl TokenCostReport {
    pub fn to_json_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Percentile summary over a multi-token run. Tail latency is what a token
/// graph fixes — means alone are insufficient.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Percentiles {
    pub n: usize,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub sum: f64,
}

impl Percentiles {
    /// Nearest-rank percentiles over `samples` (copied and sorted).
    /// Empty input yields zeros with `n = 0`.
    pub fn from_slice(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                n: 0,
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                sum: 0.0,
            };
        }
        let mut v = samples.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        let sum: f64 = v.iter().sum();
        let mean = sum / n as f64;
        let rank = |p: f64| -> f64 {
            // nearest-rank: ceil(p * n) - 1, clamped
            let idx = ((p * n as f64).ceil() as usize).saturating_sub(1).min(n - 1);
            v[idx]
        };
        Self {
            n,
            mean,
            min: v[0],
            max: v[n - 1],
            p50: rank(0.50),
            p95: rank(0.95),
            p99: rank(0.99),
            sum,
        }
    }

    pub fn from_u64_slice(samples: &[u64]) -> Self {
        let f: Vec<f64> = samples.iter().map(|&x| x as f64).collect();
        Self::from_slice(&f)
    }
}

/// Multi-token aggregation for Temporal Gravity reports.
#[derive(Debug, Clone, Serialize)]
pub struct AggregateLedger {
    pub schema: &'static str,
    pub token_count: usize,
    pub wall_us: Percentiles,
    pub unattributed_us: Percentiles,
    pub attributed_fraction: Percentiles,
    pub profiler_overhead_us: Percentiles,
    /// Per exclusive-time bucket: p50/p95/p99 across tokens.
    pub buckets_us: serde_json::Map<String, serde_json::Value>,
    pub device_gpu_execution_us: Percentiles,
    pub device_gpu_queue_wait_us: Percentiles,
    /// Tokens that had zero GPU timestamps (all CBs missing).
    pub tokens_missing_gpu_timestamps: usize,
    pub counters_mean: serde_json::Map<String, serde_json::Value>,
    pub geometry_active_bytes: Option<u64>,
    pub active_bytes_read: Percentiles,
    pub bytes_moved_total: Percentiles,
    pub notes: Vec<&'static str>,
}

/// Aggregate one or more per-token reports into p50/p95/p99 distributions.
pub fn aggregate_reports(reports: &[TokenCostReport]) -> AggregateLedger {
    let token_count = reports.len();
    let wall: Vec<u64> = reports.iter().map(|r| r.wall_us).collect();
    let unattr: Vec<u64> = reports.iter().map(|r| r.unattributed_us).collect();
    let attr_frac: Vec<f64> = reports.iter().map(|r| r.attributed_fraction).collect();
    let overhead: Vec<u64> = reports.iter().map(|r| r.profiler_overhead_us).collect();
    let gpu_exec: Vec<u64> = reports.iter().map(|r| r.device.gpu_execution_us).collect();
    let gpu_q: Vec<f64> = reports
        .iter()
        .filter_map(|r| r.device.gpu_queue_wait_us.map(|u| u as f64))
        .collect();
    let active: Vec<u64> = reports
        .iter()
        .map(|r| r.counters.active_bytes_read)
        .collect();
    let moved: Vec<u64> = reports.iter().map(|r| r.bytes_moved_total).collect();
    let missing_gpu = reports
        .iter()
        .filter(|r| r.device.gpu_timestamps_observed == 0 && r.counters.command_buffers_submitted > 0)
        .count();

    let mut buckets_us = serde_json::Map::new();
    for b in Bucket::ALL {
        let samples: Vec<u64> = reports
            .iter()
            .map(|r| {
                r.buckets_us
                    .get(b.as_str())
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0)
            })
            .collect();
        let p = Percentiles::from_u64_slice(&samples);
        buckets_us.insert(
            b.as_str().to_string(),
            serde_json::to_value(&p).unwrap_or(serde_json::Value::Null),
        );
    }
    // Unattributed is also a first-class distribution line.
    buckets_us.insert(
        "unattributed".to_string(),
        serde_json::to_value(&Percentiles::from_u64_slice(&unattr))
            .unwrap_or(serde_json::Value::Null),
    );

    let mut counters_mean = serde_json::Map::new();
    if token_count > 0 {
        let n = token_count as f64;
        let mean_u64 = |f: fn(&TokenCounters) -> u64| -> f64 {
            reports.iter().map(|r| f(&r.counters) as f64).sum::<f64>() / n
        };
        counters_mean.insert(
            "command_buffers_submitted".into(),
            serde_json::json!(mean_u64(|c| c.command_buffers_submitted)),
        );
        counters_mean.insert(
            "dispatches_encoded".into(),
            serde_json::json!(mean_u64(|c| c.dispatches_encoded)),
        );
        counters_mean.insert(
            "synchronization_points".into(),
            serde_json::json!(mean_u64(|c| c.synchronization_points)),
        );
        counters_mean.insert(
            "active_bytes_read".into(),
            serde_json::json!(mean_u64(|c| c.active_bytes_read)),
        );
        counters_mean.insert(
            "operations".into(),
            serde_json::json!(mean_u64(|c| c.operations)),
        );
        counters_mean.insert(
            "matvec_calls".into(),
            serde_json::json!(mean_u64(|c| c.matvec_calls)),
        );
        counters_mean.insert(
            "host_to_device_bytes".into(),
            serde_json::json!(mean_u64(|c| c.host_to_device_bytes)),
        );
        counters_mean.insert(
            "device_to_host_bytes".into(),
            serde_json::json!(mean_u64(|c| c.device_to_host_bytes)),
        );
    }

    let geometry = reports.iter().find_map(|r| r.geometry_active_bytes);

    AggregateLedger {
        schema: "hawking.gravity.per_token_cost_ledger_aggregate.v1",
        token_count,
        wall_us: Percentiles::from_u64_slice(&wall),
        unattributed_us: Percentiles::from_u64_slice(&unattr),
        attributed_fraction: Percentiles::from_slice(&attr_frac),
        profiler_overhead_us: Percentiles::from_u64_slice(&overhead),
        buckets_us,
        device_gpu_execution_us: Percentiles::from_u64_slice(&gpu_exec),
        device_gpu_queue_wait_us: Percentiles::from_slice(&gpu_q),
        tokens_missing_gpu_timestamps: missing_gpu,
        counters_mean,
        geometry_active_bytes: geometry,
        active_bytes_read: Percentiles::from_u64_slice(&active),
        bytes_moved_total: Percentiles::from_u64_slice(&moved),
        notes: vec![
            "p50/p95/p99 are nearest-rank over complete decode tokens.",
            "unattributed is never folded into cpu_residual_scoped.",
            "device_gpu_* are independent of exclusive CPU buckets (overlap metal_synchronize_cpu_wait).",
            "gpu_queue_wait is derived only when GPU timestamps exist; otherwise unavailable, not proxied.",
            "profiler_overhead_us is ledger bookkeeping cost disclosed for every token.",
        ],
    }
}

/// Static catalogue of every report line and how it is sourced. Used by
/// examples and gate docs so unavailability is explicit.
pub fn bucket_source_catalogue() -> Vec<serde_json::Value> {
    let mut rows = Vec::new();
    for b in Bucket::ALL {
        rows.push(serde_json::json!({
            "name": b.as_str(),
            "source": b.source(),
            "note": b.source_note(),
            "timeline": "cpu_exclusive",
        }));
    }
    rows.push(serde_json::json!({
        "name": "unattributed",
        "source": MetricSource::Derived,
        "note": "wall_us - sum(exclusive buckets); own line, never absorbed",
        "timeline": "cpu_exclusive",
    }));
    rows.push(serde_json::json!({
        "name": "gpu_execution_us",
        "source": MetricSource::GpuTimestamp,
        "note": "sum of MTLCommandBuffer GPUEndTime-GPUStartTime",
        "timeline": "device",
    }));
    rows.push(serde_json::json!({
        "name": "gpu_queue_wait_us",
        "source": MetricSource::Derived,
        "note": "per CB max(0, host_wait_us - gpu_execution_us); None if timestamps missing",
        "timeline": "device",
    }));
    rows.push(serde_json::json!({
        "name": "counter_sample_gpu_ns",
        "source": MetricSource::CounterSample,
        "note": "only when timestamp counter set exists AND markers are encoded; otherwise unavailable",
        "timeline": "device",
    }));
    rows.push(serde_json::json!({
        "name": "page_faults_minor/major",
        "source": MetricSource::CounterSample,
        "note": "getrusage(RUSAGE_SELF) ru_minflt/ru_majflt delta on Unix; unavailable elsewhere",
        "timeline": "host_os",
    }));
    rows.push(serde_json::json!({
        "name": "profiler_overhead_us",
        "source": MetricSource::CpuClock,
        "note": "host time inside ledger enter/exit/add/finish",
        "timeline": "profiler",
    }));
    rows
}

// ── internal state ─────────────────────────────────────────────────────────

struct Frame {
    bucket: Bucket,
    /// Nanos accumulated exclusively into this frame while it was active.
    exclusive_ns: u128,
    /// When this frame last became the active (top-of-stack) frame.
    resumed_at: Option<Instant>,
}

struct TokenState {
    wall_start: Instant,
    nanos: [u128; 15],
    stack: Vec<Frame>,
    counters: TokenCounters,
    /// Parallel to [`ActiveByteCategory::ALL`]; folded into
    /// `counters.active_bytes_by_category` at finish.
    active_by_cat: [u64; 8],
    transfers: Vec<TransferRecord>,
    geometry_active_bytes: Option<u64>,
    device: DeviceTimeline,
    /// Nanos spent inside ledger bookkeeping (profiler self-cost).
    profiler_overhead_ns: u128,
    /// Page-fault baseline at begin_token, if available.
    fault_baseline: Option<(u64, u64)>,
}

impl TokenState {
    fn new() -> Self {
        Self {
            wall_start: Instant::now(),
            nanos: [0; 15],
            stack: Vec::new(),
            counters: TokenCounters::default(),
            active_by_cat: [0; 8],
            transfers: Vec::new(),
            geometry_active_bytes: None,
            device: DeviceTimeline::default(),
            profiler_overhead_ns: 0,
            fault_baseline: sample_page_faults(),
        }
    }

    fn charge_overhead(&mut self, started: Instant) {
        self.profiler_overhead_ns = self
            .profiler_overhead_ns
            .saturating_add(started.elapsed().as_nanos());
    }

    fn pause_top(&mut self, now: Instant) {
        if let Some(frame) = self.stack.last_mut() {
            if let Some(t0) = frame.resumed_at.take() {
                frame.exclusive_ns = frame
                    .exclusive_ns
                    .saturating_add(now.duration_since(t0).as_nanos());
            }
        }
    }

    fn resume_top(&mut self, now: Instant) {
        if let Some(frame) = self.stack.last_mut() {
            frame.resumed_at = Some(now);
        }
    }

    fn enter(&mut self, bucket: Bucket) {
        let oh = Instant::now();
        let now = Instant::now();
        self.pause_top(now);
        self.stack.push(Frame {
            bucket,
            exclusive_ns: 0,
            resumed_at: Some(now),
        });
        self.charge_overhead(oh);
    }

    fn exit(&mut self, bucket: Bucket) {
        let oh = Instant::now();
        let now = Instant::now();
        let Some(mut frame) = self.stack.pop() else {
            self.charge_overhead(oh);
            return;
        };
        // Mismatched exit is a programming error; still fold time into the
        // frame's own bucket so we never silently drop measured work.
        if frame.bucket != bucket {
            eprintln!(
                "[cost_ledger] mismatched exit: expected {:?}, got {:?}",
                frame.bucket, bucket
            );
        }
        if let Some(t0) = frame.resumed_at.take() {
            frame.exclusive_ns = frame
                .exclusive_ns
                .saturating_add(now.duration_since(t0).as_nanos());
        }
        self.nanos[frame.bucket.index()] =
            self.nanos[frame.bucket.index()].saturating_add(frame.exclusive_ns);
        self.resume_top(now);
        self.charge_overhead(oh);
    }

    /// Add exclusive time to a bucket without stack nesting (for split
    /// encode/submit/wait where a parent scope already owns the region).
    ///
    /// Pauses the open parent, folds its live time into `exclusive_ns`, then
    /// deducts `ns` from that parent so wall time is not double-counted.
    fn add_ns(&mut self, bucket: Bucket, ns: u128) {
        if ns == 0 {
            return;
        }
        let oh = Instant::now();
        let now = Instant::now();
        self.pause_top(now);
        if let Some(frame) = self.stack.last_mut() {
            // Parent exclusive now includes the just-measured sub-interval.
            frame.exclusive_ns = frame.exclusive_ns.saturating_sub(ns);
        }
        self.nanos[bucket.index()] = self.nanos[bucket.index()].saturating_add(ns);
        self.resume_top(now);
        self.charge_overhead(oh);
    }

    fn push_gpu_cb(&mut self, sample: GpuCommandBufferSample) {
        let oh = Instant::now();
        match sample.gpu_execution_us {
            Some(exec) => {
                self.device.gpu_execution_us =
                    self.device.gpu_execution_us.saturating_add(exec);
                self.device.gpu_timestamps_observed =
                    self.device.gpu_timestamps_observed.saturating_add(1);
                if let Some(q) = sample.gpu_queue_wait_us {
                    let acc = self.device.gpu_queue_wait_us.get_or_insert(0);
                    *acc = acc.saturating_add(q);
                }
            }
            None => {
                self.device.gpu_timestamps_missing =
                    self.device.gpu_timestamps_missing.saturating_add(1);
            }
        }
        if self.device.command_buffers.len() < 8192 {
            self.device.command_buffers.push(sample);
        }
        self.charge_overhead(oh);
    }

    fn finish(mut self) -> TokenCostReport {
        let oh = Instant::now();
        let now = Instant::now();
        // Drain any open scopes (should be empty if callers balanced).
        while let Some(mut frame) = self.stack.pop() {
            if let Some(t0) = frame.resumed_at.take() {
                frame.exclusive_ns = frame
                    .exclusive_ns
                    .saturating_add(now.duration_since(t0).as_nanos());
            }
            self.nanos[frame.bucket.index()] =
                self.nanos[frame.bucket.index()].saturating_add(frame.exclusive_ns);
        }
        let wall_ns = now.duration_since(self.wall_start).as_nanos();
        let wall_us = (wall_ns / 1_000) as u64;
        let mut buckets_us = serde_json::Map::new();
        let mut bucket_sources = serde_json::Map::new();
        // Floor each bucket to whole microseconds first, then sum — so
        // `attributed_us == sum(buckets_us.values())` exactly (no 1 µs
        // residual from summing nanos then dividing once).
        let mut attributed_us: u64 = 0;
        for b in Bucket::ALL {
            let us = (self.nanos[b.index()] / 1_000) as u64;
            attributed_us = attributed_us.saturating_add(us);
            buckets_us.insert(b.as_str().to_string(), serde_json::json!(us));
            bucket_sources.insert(
                b.as_str().to_string(),
                serde_json::json!({
                    "source": b.source(),
                    "note": b.source_note(),
                }),
            );
        }
        let unattributed_signed_us = wall_us as i64 - attributed_us as i64;
        let unattributed_us = unattributed_signed_us.max(0) as u64;
        let attributed_fraction = if wall_us == 0 {
            0.0
        } else {
            attributed_us as f64 / wall_us as f64
        };
        let active_bytes_vs_geometry_fraction = self
            .geometry_active_bytes
            .filter(|&g| g > 0)
            .map(|g| self.counters.active_bytes_read as f64 / g as f64);

        // Page-fault delta.
        if let (Some((bmin, bmaj)), Some((emin, emaj))) =
            (self.fault_baseline, sample_page_faults())
        {
            self.counters.page_faults_available = true;
            self.counters.page_faults_minor = Some(emin.saturating_sub(bmin));
            self.counters.page_faults_major = Some(emaj.saturating_sub(bmaj));
        } else {
            self.counters.page_faults_available = sample_page_faults().is_some();
            if !self.counters.page_faults_available {
                self.counters.page_faults_minor = None;
                self.counters.page_faults_major = None;
            }
        }

        // Materialize the category partition so JSON consumers see a stable
        // map that sums to active_bytes_read (when every record went through
        // record_active_bytes_in / the categorized path).
        let mut by_cat = serde_json::Map::new();
        let mut cat_sum = 0u64;
        for c in ActiveByteCategory::ALL {
            let v = self.active_by_cat[c.index()];
            cat_sum = cat_sum.saturating_add(v);
            by_cat.insert(c.as_str().to_string(), serde_json::json!(v));
        }
        self.counters.active_bytes_by_category = by_cat;
        // If a caller used the uncategorized record_active_bytes, the sum of
        // categories can be lower than active_bytes_read; surface the gap as
        // `other` only when other is zero and the gap is positive so we never
        // invent bytes that were not observed.
        if cat_sum < self.counters.active_bytes_read {
            let gap = self.counters.active_bytes_read - cat_sum;
            let other_idx = ActiveByteCategory::Other.index();
            self.active_by_cat[other_idx] =
                self.active_by_cat[other_idx].saturating_add(gap);
            self.counters
                .active_bytes_by_category
                .insert(
                    ActiveByteCategory::Other.as_str().to_string(),
                    serde_json::json!(self.active_by_cat[other_idx]),
                );
        }

        let bytes_moved_total = self
            .counters
            .host_to_device_bytes
            .saturating_add(self.counters.device_to_host_bytes)
            .saturating_add(self.counters.active_bytes_read);
        let bytes_moved_vs_geometry_fraction = self
            .geometry_active_bytes
            .filter(|&g| g > 0)
            .map(|g| bytes_moved_total as f64 / g as f64);

        // Device notes when timestamps were sparse.
        if self.device.gpu_timestamps_missing > 0 && self.device.notes.is_empty() {
            self.device.notes.push(
                "one or more command buffers lacked readable GPUStartTime/GPUEndTime",
            );
        }
        if self.device.gpu_timestamps_observed == 0
            && self.counters.command_buffers_submitted > 0
        {
            self.device.notes.push(
                "GPU timestamps unavailable this token; gpu_queue_wait_us left unset (no CPU proxy)",
            );
        }
        if !self.device.counter_sample_probed {
            self.device.notes.push(
                "Metal timestamp counter set not probed this token; counter_samples_recorded=0",
            );
        } else if self.device.counter_sample_supported == Some(false) {
            self.device.notes.push(
                "device has no 'timestamp' common counter set; counter samples unavailable",
            );
        } else if self.device.counter_samples_recorded == 0 {
            self.device.notes.push(
                "timestamp counter set present but no sample markers encoded this token",
            );
        }

        self.profiler_overhead_ns = self
            .profiler_overhead_ns
            .saturating_add(oh.elapsed().as_nanos());
        let profiler_overhead_us = (self.profiler_overhead_ns / 1_000) as u64;
        let profiler_overhead_fraction = if wall_us == 0 {
            0.0
        } else {
            profiler_overhead_us as f64 / wall_us as f64
        };

        TokenCostReport {
            schema: "hawking.gravity.per_token_cost_ledger.v2",
            wall_us,
            buckets_us,
            bucket_sources,
            unattributed_us,
            unattributed_name: "unattributed",
            unattributed_signed_us,
            attributed_us,
            attributed_fraction,
            counters: self.counters,
            device: self.device,
            geometry_active_bytes: self.geometry_active_bytes,
            active_bytes_vs_geometry_fraction,
            bytes_moved_total,
            bytes_moved_vs_geometry_fraction,
            transfers: self.transfers,
            profiler_overhead_us,
            profiler_overhead_fraction,
        }
    }
}

// ── page faults via getrusage (no extra crate) ─────────────────────────────

#[cfg(unix)]
fn sample_page_faults() -> Option<(u64, u64)> {
    // Platform-correct rusage so minflt/majflt land at the right offsets.
    // Darwin: timeval is { i64 tv_sec; i32 tv_usec; /* +4 pad */ } (16 B).
    // Linux LP64: timeval is { i64 tv_sec; i64 tv_usec } (16 B).
    #[cfg(target_os = "macos")]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TimeVal {
        tv_sec: i64,
        tv_usec: i32,
        _pad: i32,
    }
    #[cfg(not(target_os = "macos"))]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct TimeVal {
        tv_sec: i64,
        tv_usec: i64,
    }
    #[repr(C)]
    struct Rusage {
        ru_utime: TimeVal,
        ru_stime: TimeVal,
        ru_maxrss: i64,
        ru_ixrss: i64,
        ru_idrss: i64,
        ru_isrss: i64,
        ru_minflt: i64,
        ru_majflt: i64,
        // remaining kernel fields; size so the write never overflows the stack
        _pad: [i64; 8],
    }
    extern "C" {
        fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }
    // RUSAGE_SELF = 0 on Darwin and Linux.
    const RUSAGE_SELF: i32 = 0;
    #[cfg(target_os = "macos")]
    let zero_tv = TimeVal {
        tv_sec: 0,
        tv_usec: 0,
        _pad: 0,
    };
    #[cfg(not(target_os = "macos"))]
    let zero_tv = TimeVal {
        tv_sec: 0,
        tv_usec: 0,
    };
    let mut u = Rusage {
        ru_utime: zero_tv,
        ru_stime: zero_tv,
        ru_maxrss: 0,
        ru_ixrss: 0,
        ru_idrss: 0,
        ru_isrss: 0,
        ru_minflt: 0,
        ru_majflt: 0,
        _pad: [0; 8],
    };
    let rc = unsafe { getrusage(RUSAGE_SELF, &mut u) };
    if rc != 0 {
        return None;
    }
    Some((u.ru_minflt.max(0) as u64, u.ru_majflt.max(0) as u64))
}

#[cfg(not(unix))]
fn sample_page_faults() -> Option<(u64, u64)> {
    None
}

// ── process / thread switches ──────────────────────────────────────────────

static PROCESS_ENABLED: AtomicBool = AtomicBool::new(false);
static ENV_RESOLVED: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TOKEN: RefCell<Option<TokenState>> = const { RefCell::new(None) };
    /// Per-thread override of the process switch. `None` defers to
    /// `PROCESS_ENABLED` (env / process-wide `set_enabled`). Lets unit
    /// tests enable on one thread without racing siblings.
    static THREAD_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Resolve `HAWKING_COST_LEDGER` once. Safe to call repeatedly.
pub fn resolve_env() {
    if ENV_RESOLVED.swap(true, Ordering::Relaxed) {
        return;
    }
    if crate::env_on(COST_LEDGER_ENV) {
        PROCESS_ENABLED.store(true, Ordering::Relaxed);
    }
}

/// Programmatic enable/disable for **this thread**. Does not start a token;
/// see [`begin_token`]. Prefer this over the env var in tests and examples.
pub fn set_enabled(on: bool) {
    ENV_RESOLVED.store(true, Ordering::Relaxed);
    THREAD_ENABLED.with(|c| c.set(Some(on)));
}

/// Process-wide enable (also used by env resolution). Rarely needed outside
/// of multi-thread servers that want one switch for every worker.
pub fn set_enabled_process(on: bool) {
    ENV_RESOLVED.store(true, Ordering::Relaxed);
    PROCESS_ENABLED.store(on, Ordering::Relaxed);
}

/// True when the ledger switch is on for this thread. Does not require an
/// active token — used by hot-path hooks as a cheap early-out.
pub fn is_enabled() -> bool {
    resolve_env();
    if let Some(on) = THREAD_ENABLED.with(|c| c.get()) {
        return on;
    }
    PROCESS_ENABLED.load(Ordering::Relaxed)
}

/// True when a token is currently being recorded on this thread.
pub fn is_recording() -> bool {
    if !is_enabled() {
        return false;
    }
    TOKEN.with(|t| t.borrow().is_some())
}

/// Start exclusive attribution for one decode token on this thread.
/// No-op (and returns false) when the ledger is disabled.
pub fn begin_token() -> bool {
    if !is_enabled() {
        return false;
    }
    TOKEN.with(|t| {
        *t.borrow_mut() = Some(TokenState::new());
    });
    true
}

/// Finish the current token and return its report. Returns `None` when no
/// token was active (or the ledger is off).
pub fn end_token() -> Option<TokenCostReport> {
    if !is_enabled() {
        return None;
    }
    TOKEN.with(|t| t.borrow_mut().take().map(TokenState::finish))
}

/// RAII scope that charges exclusive time to `bucket` while it is alive.
pub struct Scope {
    bucket: Bucket,
    active: bool,
}

impl Scope {
    pub fn new(bucket: Bucket) -> Self {
        let active = is_recording();
        if active {
            TOKEN.with(|t| {
                if let Some(state) = t.borrow_mut().as_mut() {
                    state.enter(bucket);
                }
            });
        }
        Self { bucket, active }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if self.active {
            TOKEN.with(|t| {
                if let Some(state) = t.borrow_mut().as_mut() {
                    state.exit(self.bucket);
                }
            });
        }
    }
}

/// Enter a bucket for the duration of `f`.
#[inline]
pub fn with_bucket<R>(bucket: Bucket, f: impl FnOnce() -> R) -> R {
    let _scope = Scope::new(bucket);
    f()
}

/// Charge `duration` to `bucket` (and deduct from any open parent). Prefer
/// [`Scope`] for nested regions; use this for split encode/submit/wait
/// slices measured with their own `Instant`s.
pub fn add_duration(bucket: Bucket, duration: std::time::Duration) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.add_ns(bucket, duration.as_nanos());
        }
    });
}

pub fn record_command_buffer() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.command_buffers_submitted =
                state.counters.command_buffers_submitted.saturating_add(1);
        }
    });
}

pub fn record_dispatches(n: u64) {
    if n == 0 || !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.dispatches_encoded =
                state.counters.dispatches_encoded.saturating_add(n);
        }
    });
}

pub fn record_sync_point() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.synchronization_points =
                state.counters.synchronization_points.saturating_add(1);
        }
    });
}

/// Record one completed command buffer's host + GPU times.
///
/// Call **after** `wait_until_completed`. Pass `gpu_start_s` / `gpu_end_s`
/// from `MTLCommandBuffer.GPUStartTime` / `GPUEndTime` when readable; pass
/// `None` when the driver returns zeros — do **not** invent values.
pub fn record_gpu_command_buffer(
    host_commit_us: u64,
    host_wait_us: u64,
    gpu_start_s: Option<f64>,
    gpu_end_s: Option<f64>,
    dispatches_in_buffer: u64,
) {
    if !is_recording() {
        return;
    }
    let (gpu_execution_us, gpu_queue_wait_us) = match (gpu_start_s, gpu_end_s) {
        (Some(s), Some(e)) if e > s && (e - s) > 0.0 => {
            let exec = ((e - s) * 1_000_000.0) as u64;
            let q = host_wait_us.saturating_sub(exec);
            (Some(exec), Some(q))
        }
        _ => (None, None),
    };
    let sample = GpuCommandBufferSample {
        host_commit_us,
        host_wait_us,
        gpu_execution_us,
        gpu_queue_wait_us,
        gpu_start_s,
        gpu_end_s,
        dispatches_in_buffer,
    };
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.push_gpu_cb(sample);
        }
    });
}

/// Record the result of probing the Metal `timestamp` common counter set.
/// Does not claim samples were encoded.
pub fn record_counter_sample_capability(probed: bool, supported: Option<bool>) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.device.counter_sample_probed = probed;
            state.device.counter_sample_supported = supported;
        }
    });
}

/// Increment the count of counter-sample markers actually resolved this token.
pub fn record_counter_samples(n: u64) {
    if n == 0 || !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.device.counter_samples_recorded =
                state.device.counter_samples_recorded.saturating_add(n);
        }
    });
}

pub fn record_transfer(bytes: u64, host_to_device: bool, kind: &'static str) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            if host_to_device {
                state.counters.host_to_device_bytes =
                    state.counters.host_to_device_bytes.saturating_add(bytes);
                state.counters.host_to_device_transfers =
                    state.counters.host_to_device_transfers.saturating_add(1);
            } else {
                state.counters.device_to_host_bytes =
                    state.counters.device_to_host_bytes.saturating_add(bytes);
                state.counters.device_to_host_transfers =
                    state.counters.device_to_host_transfers.saturating_add(1);
            }
            // Cap transfer log so a long warm run does not grow without bound
            // when someone leaves the ledger on for many tokens.
            if state.transfers.len() < 4096 {
                state.transfers.push(TransferRecord {
                    bytes,
                    host_to_device,
                    kind,
                });
            }
        }
    });
}

pub fn record_allocation(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.allocations = state.counters.allocations.saturating_add(1);
            state.counters.allocation_bytes =
                state.counters.allocation_bytes.saturating_add(bytes);
        }
    });
}

pub fn record_active_bytes(bytes: u64) {
    // Uncategorized path — still counted in the total; finish() folds any
    // uncategorized remainder into `other` so the category map stays a
    // complete partition.
    record_active_bytes_in(ActiveByteCategory::Other, bytes);
}

/// Record weight bytes touched by a matvec, attributed to a tensor class.
pub fn record_active_bytes_in(category: ActiveByteCategory, bytes: u64) {
    if !is_recording() || bytes == 0 {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.active_bytes_read =
                state.counters.active_bytes_read.saturating_add(bytes);
            let i = category.index();
            state.active_by_cat[i] = state.active_by_cat[i].saturating_add(bytes);
        }
    });
}

/// Convenience: classify `name` then [`record_active_bytes_in`].
pub fn record_active_bytes_for(name: &str, bytes: u64) {
    record_active_bytes_in(classify_weight_name(name), bytes);
}

pub fn record_first_touch_load_bytes(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.first_touch_load_bytes =
                state.counters.first_touch_load_bytes.saturating_add(bytes);
        }
    });
}

pub fn record_matvec_call() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.matvec_calls = state.counters.matvec_calls.saturating_add(1);
        }
    });
}

pub fn record_matvec_batch(items: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.matvec_batch_calls =
                state.counters.matvec_batch_calls.saturating_add(1);
            state.counters.matvec_batch_items =
                state.counters.matvec_batch_items.saturating_add(items);
        }
    });
}

pub fn record_dense_call() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.dense_calls = state.counters.dense_calls.saturating_add(1);
        }
    });
}

pub fn record_row_call() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.row_calls = state.counters.row_calls.saturating_add(1);
        }
    });
}

pub fn record_sha_verification() {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.sha_verifications =
                state.counters.sha_verifications.saturating_add(1);
        }
    });
}

/// Record abstract operation count (e.g. estimated FMAs or scored attention
/// cells). Units are caller-defined; the report surfaces the sum only.
pub fn record_operations(n: u64) {
    if n == 0 || !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.operations = state.counters.operations.saturating_add(n);
        }
    });
}

/// Snapshot GPU weight-cache residency at end of (or during) a token.
pub fn record_residency(bytes: u64, entries: u64, evictions: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.counters.residency_bytes = Some(bytes);
            state.counters.residency_entries = Some(entries);
            state.counters.residency_evictions = Some(evictions);
        }
    });
}

/// Gate geometry: active routed-expert bytes per token.
/// `experts_per_tok * 3 projections * bytes_per_projection * n_layers`.
pub fn set_geometry_active_bytes(bytes: u64) {
    if !is_recording() {
        return;
    }
    TOKEN.with(|t| {
        if let Some(state) = t.borrow_mut().as_mut() {
            state.geometry_active_bytes = Some(bytes);
        }
    });
}

/// Default geometry quoted by BASE_RUNTIME_MAXIMIZED_GATE for the sealed
/// Math-Preserve artifact (8 × 3 × 1_378_368 × 78).
///
/// **This is not the full per-token weight traffic.** It only counts routed
/// expert projections under an idealised 78-sparse-layer schedule. The sealed
/// General-R0 artifact has 3 dense MLP layers + 75 sparse, plus attention,
/// indexer, router, shared expert, and a native `lm_head` every token. See
/// [`sealed_glm_active_byte_schedule`].
pub const SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES: u64 = 8 * 3 * 1_378_368 * 78;

/// Sealed General-R0 / Math-Preserve sizes used by the static active-byte
/// schedule. **Derived from shard headers on disk** (descriptor `bytes` for
/// `gravity-pq`; f32-widened element count for `native.bf16`). Not live GPU
/// measurements. GPU path active bytes for PQ are `codebooks.length() +
/// codes.length()` ≈ `descriptor.bytes - 60` (drop 64 B header, add 4 B codes
/// pad); the difference is <0.01% and is ignored in the static schedule so the
/// geometry constant stays comparable.
///
/// Source artifact:
/// `…/GLM-5.2/b4734de4facf877f85769a911abafc5283eab3d9/General-R0`.
pub mod sealed_glm_sizes {
    /// Packed expert (or shared-expert) projection payload bytes.
    pub const EXPERT_PROJ_BYTES: u64 = 1_378_368;
    /// Dense-MLP projection (intermediate 12288) payload bytes.
    pub const DENSE_MLP_PROJ_BYTES: u64 = 8_259_648;
    /// Attention projections per layer (q_a + q_b + kv_a + kv_b + o_proj).
    pub const ATTENTION_PER_LAYER_BYTES: u64 =
        1_378_368 + 3_672_128 + 389_184 + 1_607_744 + 11_012_160;
    /// Full-indexer natives, **f32-widened** (stored bf16 × 2).
    /// wq_b 16_777_216 + wk 1_572_864 + weights_proj 393_216, each ×2.
    pub const INDEXER_PER_FULL_LAYER_F32_BYTES: u64 =
        (16_777_216 + 1_572_864 + 393_216) * 2;
    /// Router `mlp.gate.weight` f32-widened (stored bf16 3_145_728).
    pub const ROUTER_F32_BYTES: u64 = 3_145_728 * 2;
    /// `lm_head.weight` f32-widened: 154_880 × 6_144 × 4.
    pub const LM_HEAD_F32_BYTES: u64 = 154_880 * 6_144 * 4;
    /// Stored bf16 size of lm_head (no widen).
    pub const LM_HEAD_BF16_BYTES: u64 = 154_880 * 6_144 * 2;
}

/// Static per-token active-byte schedule for the sealed GLM MoE forward.
///
/// Built from architecture counts × sealed tensor sizes. This is what
/// `active_bytes_read` **should** be on the GPU path if every matvec records
/// exact tensor extents once. Compare a live category breakdown against it;
/// a gap is a finding, not something to absorb into a neighbour.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SealedGlmActiveByteSchedule {
    pub method: &'static str,
    pub n_layers: u64,
    pub n_sparse_layers: u64,
    pub n_dense_mlp_layers: u64,
    pub n_full_indexer_layers: u64,
    pub experts_per_tok: u64,
    pub geometry_routed_bytes: u64,
    pub routed_experts_bytes: u64,
    pub shared_experts_bytes: u64,
    pub dense_mlp_bytes: u64,
    pub attention_bytes: u64,
    pub indexer_bytes: u64,
    pub router_bytes: u64,
    pub lm_head_bytes: u64,
    pub total_active_bytes: u64,
    /// `total - geometry`. Mostly required non-routed work + f32 widen tax.
    pub surplus_vs_geometry_bytes: u64,
    /// f32-vs-bf16 inflation on natives the GPU path currently widens.
    pub native_f32_widen_tax_bytes: u64,
}

/// Static schedule for the sealed Math-Preserve forward (no device needed).
pub fn sealed_glm_active_byte_schedule() -> SealedGlmActiveByteSchedule {
    use sealed_glm_sizes::*;
    let n_layers = 78u64;
    let n_dense = 3u64;
    let n_sparse = n_layers - n_dense; // 75
    let n_full_idx = 21u64;
    let ept = 8u64;
    let routed = ept * 3 * EXPERT_PROJ_BYTES * n_sparse;
    let shared = 1 * 3 * EXPERT_PROJ_BYTES * n_sparse;
    let dense_mlp = 3 * DENSE_MLP_PROJ_BYTES * n_dense;
    let attention = ATTENTION_PER_LAYER_BYTES * n_layers;
    let indexer = INDEXER_PER_FULL_LAYER_F32_BYTES * n_full_idx;
    let router = ROUTER_F32_BYTES * n_sparse;
    let lm_head = LM_HEAD_F32_BYTES;
    let total = routed + shared + dense_mlp + attention + indexer + router + lm_head;
    let geometry = SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES;
    // Widen tax: natives billed at f32 while stored as bf16.
    let widen = (LM_HEAD_F32_BYTES - LM_HEAD_BF16_BYTES)
        + (INDEXER_PER_FULL_LAYER_F32_BYTES / 2) * n_full_idx // half of f32 is the tax
        + (ROUTER_F32_BYTES / 2) * n_sparse;
    SealedGlmActiveByteSchedule {
        method: "static_from_sealed_artifact_headers_and_forward_schedule",
        n_layers,
        n_sparse_layers: n_sparse,
        n_dense_mlp_layers: n_dense,
        n_full_indexer_layers: n_full_idx,
        experts_per_tok: ept,
        geometry_routed_bytes: geometry,
        routed_experts_bytes: routed,
        shared_experts_bytes: shared,
        dense_mlp_bytes: dense_mlp,
        attention_bytes: attention,
        indexer_bytes: indexer,
        router_bytes: router,
        lm_head_bytes: lm_head,
        total_active_bytes: total,
        surplus_vs_geometry_bytes: total.saturating_sub(geometry),
        native_f32_widen_tax_bytes: widen,
    }
}

/// Compute geometry from arch fields when the per-projection byte size is
/// known (from a live PQ header). Falls back to the sealed constant when
/// `bytes_per_projection` is `None`.
pub fn geometry_active_bytes(
    n_layers: usize,
    experts_per_tok: usize,
    bytes_per_projection: Option<u64>,
) -> u64 {
    let bpp = bytes_per_projection.unwrap_or(1_378_368);
    (n_layers as u64)
        .saturating_mul(experts_per_tok as u64)
        .saturating_mul(3)
        .saturating_mul(bpp)
}

// ── synthetic report builder (unit tests / offline aggregation) ────────────

/// Build a [`TokenCostReport`] from synthetic numbers without running decode.
/// Used to unit-test aggregation and the unattributed identity without a
/// Metal device.
pub fn synthetic_report(
    wall_us: u64,
    bucket_us: &[(Bucket, u64)],
    counters: TokenCounters,
    device: DeviceTimeline,
    geometry_active_bytes: Option<u64>,
    profiler_overhead_us: u64,
) -> TokenCostReport {
    let mut buckets_us = serde_json::Map::new();
    let mut bucket_sources = serde_json::Map::new();
    let mut attributed_us = 0u64;
    for b in Bucket::ALL {
        buckets_us.insert(b.as_str().to_string(), serde_json::json!(0u64));
        bucket_sources.insert(
            b.as_str().to_string(),
            serde_json::json!({
                "source": b.source(),
                "note": b.source_note(),
            }),
        );
    }
    for &(b, us) in bucket_us {
        attributed_us = attributed_us.saturating_add(us);
        buckets_us.insert(b.as_str().to_string(), serde_json::json!(us));
    }
    let unattributed_signed_us = wall_us as i64 - attributed_us as i64;
    let unattributed_us = unattributed_signed_us.max(0) as u64;
    let attributed_fraction = if wall_us == 0 {
        0.0
    } else {
        attributed_us as f64 / wall_us as f64
    };
    let bytes_moved_total = counters
        .host_to_device_bytes
        .saturating_add(counters.device_to_host_bytes)
        .saturating_add(counters.active_bytes_read);
    let active_bytes_vs_geometry_fraction = geometry_active_bytes
        .filter(|&g| g > 0)
        .map(|g| counters.active_bytes_read as f64 / g as f64);
    let bytes_moved_vs_geometry_fraction = geometry_active_bytes
        .filter(|&g| g > 0)
        .map(|g| bytes_moved_total as f64 / g as f64);
    let profiler_overhead_fraction = if wall_us == 0 {
        0.0
    } else {
        profiler_overhead_us as f64 / wall_us as f64
    };
    TokenCostReport {
        schema: "hawking.gravity.per_token_cost_ledger.v2",
        wall_us,
        buckets_us,
        bucket_sources,
        unattributed_us,
        unattributed_name: "unattributed",
        unattributed_signed_us,
        attributed_us,
        attributed_fraction,
        counters,
        device,
        geometry_active_bytes,
        active_bytes_vs_geometry_fraction,
        bytes_moved_total,
        bytes_moved_vs_geometry_fraction,
        transfers: Vec::new(),
        profiler_overhead_us,
        profiler_overhead_fraction,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn with_clean_ledger<R>(f: impl FnOnce() -> R) -> R {
        set_enabled(true);
        // Ensure no leftover token.
        let _ = end_token();
        let out = f();
        let _ = end_token();
        set_enabled(false);
        out
    }

    #[test]
    fn disabled_is_noop() {
        set_enabled(false);
        assert!(!begin_token());
        assert!(end_token().is_none());
        let _s = Scope::new(Bucket::Routing);
        record_dispatches(99);
        assert!(!is_recording());
    }

    #[test]
    fn exclusive_stack_partitions_time() {
        with_clean_ledger(|| {
            assert!(begin_token());
            {
                let _a = Scope::new(Bucket::AttentionAndIndexShare);
                std::thread::sleep(Duration::from_millis(5));
                {
                    let _m = Scope::new(Bucket::MetalEncode);
                    std::thread::sleep(Duration::from_millis(5));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            {
                let _r = Scope::new(Bucket::Routing);
                std::thread::sleep(Duration::from_millis(5));
            }
            let report = end_token().expect("report");
            let attn = report.buckets_us["attention_and_indexshare"]
                .as_u64()
                .unwrap();
            let enc = report.buckets_us["metal_encode"].as_u64().unwrap();
            let route = report.buckets_us["routing"].as_u64().unwrap();
            // Each sleep is ~5ms; allow wide slack for CI scheduling.
            assert!(enc >= 3_000, "encode us={enc}");
            assert!(attn >= 6_000, "attn exclusive us={attn}");
            assert!(route >= 3_000, "route us={route}");
            let sum: u64 = report
                .buckets_us
                .values()
                .filter_map(|v| v.as_u64())
                .sum();
            assert_eq!(sum, report.attributed_us);
            // Exclusive identity: attributed + unattributed ≈ wall (within 1ms slack).
            let covered = report.attributed_us + report.unattributed_us;
            let delta = (covered as i64 - report.wall_us as i64).unsigned_abs();
            assert!(
                delta < 1_000,
                "covered={covered} wall={} delta={delta}",
                report.wall_us
            );
            // Nested encode must not be inside attn's exclusive total in a
            // way that makes attn ≈ encode+2*5ms; attn should be ~10ms not ~15ms.
            assert!(
                attn < enc + 12_000,
                "attn should exclude nested encode: attn={attn} enc={enc}"
            );
            // Profiler overhead must be disclosed and non-zero after work.
            assert!(
                report.profiler_overhead_us > 0 || report.profiler_overhead_fraction >= 0.0,
                "overhead fields present"
            );
            assert_eq!(report.unattributed_name, "unattributed");
        });
    }

    #[test]
    fn unattributed_is_explicit_when_no_scopes() {
        with_clean_ledger(|| {
            assert!(begin_token());
            std::thread::sleep(Duration::from_millis(3));
            let report = end_token().expect("report");
            assert_eq!(report.attributed_us, 0);
            assert!(report.unattributed_us >= 2_000);
            assert!(report.unattributed_signed_us > 0);
            assert_eq!(report.unattributed_name, "unattributed");
            // Hard rule: remainder is NOT in cpu_residual_scoped.
            let residual = report.buckets_us["cpu_residual_scoped"].as_u64().unwrap();
            assert_eq!(residual, 0);
        });
    }

    #[test]
    fn counters_accumulate() {
        with_clean_ledger(|| {
            assert!(begin_token());
            set_geometry_active_bytes(SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES);
            record_command_buffer();
            record_dispatches(8);
            record_sync_point();
            record_transfer(1024, true, "x_upload");
            record_transfer(2048, false, "y_download");
            record_allocation(4096);
            record_active_bytes(1_378_368);
            record_matvec_batch(8);
            record_sha_verification();
            record_operations(1_000_000);
            record_residency(32 << 30, 64, 2);
            record_gpu_command_buffer(
                10,   // commit
                5000, // wait
                Some(100.0),
                Some(100.003), // 3000 µs GPU
                8,
            );
            record_counter_sample_capability(true, Some(true));
            // markers not encoded
            let report = end_token().expect("report");
            assert_eq!(report.counters.command_buffers_submitted, 1);
            assert_eq!(report.counters.dispatches_encoded, 8);
            assert_eq!(report.counters.synchronization_points, 1);
            assert_eq!(report.counters.host_to_device_bytes, 1024);
            assert_eq!(report.counters.device_to_host_bytes, 2048);
            assert_eq!(report.counters.allocations, 1);
            assert_eq!(report.counters.active_bytes_read, 1_378_368);
            assert_eq!(report.counters.operations, 1_000_000);
            assert_eq!(report.counters.residency_bytes, Some(32 << 30));
            assert_eq!(
                report.geometry_active_bytes,
                Some(SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES)
            );
            assert!(report.active_bytes_vs_geometry_fraction.unwrap() < 0.01);
            assert_eq!(report.transfers.len(), 2);
            assert_eq!(report.device.gpu_execution_us, 3000);
            assert_eq!(report.device.gpu_queue_wait_us, Some(2000)); // 5000-3000
            assert_eq!(report.device.gpu_timestamps_observed, 1);
            assert_eq!(report.device.counter_sample_supported, Some(true));
            assert_eq!(report.device.counter_samples_recorded, 0);
            assert!(report
                .device
                .notes
                .iter()
                .any(|n| n.contains("no sample markers")));
        });
    }

    #[test]
    fn gpu_timestamps_missing_leaves_queue_wait_unset() {
        with_clean_ledger(|| {
            assert!(begin_token());
            record_command_buffer();
            record_gpu_command_buffer(5, 1000, None, None, 1);
            let report = end_token().expect("report");
            assert_eq!(report.device.gpu_execution_us, 0);
            assert_eq!(report.device.gpu_queue_wait_us, None);
            assert_eq!(report.device.gpu_timestamps_missing, 1);
            assert!(report
                .device
                .notes
                .iter()
                .any(|n| n.contains("no CPU proxy") || n.contains("lacked readable")));
        });
    }

    #[test]
    fn geometry_helper_matches_gate_number() {
        // 8 experts × 3 projections × 1_378_368 × 78 layers ≈ 2.58 GB.
        let g = geometry_active_bytes(78, 8, Some(1_378_368));
        assert_eq!(g, SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES);
        assert_eq!(g, 8u64 * 3 * 1_378_368 * 78);
        // Sanity: ~2.58e9 bytes as the gate states.
        assert!((g as f64 - 2.58e9).abs() < 5e6, "g={g}");
    }

    #[test]
    fn classify_weight_name_partitions_glm_schedule() {
        assert_eq!(
            classify_weight_name("lm_head.weight"),
            ActiveByteCategory::LmHead
        );
        assert_eq!(
            classify_weight_name("model.layers.3.mlp.experts.7.gate_proj.weight"),
            ActiveByteCategory::RoutedExperts
        );
        assert_eq!(
            classify_weight_name("model.layers.3.mlp.shared_experts.down_proj.weight"),
            ActiveByteCategory::SharedExperts
        );
        assert_eq!(
            classify_weight_name("model.layers.0.mlp.gate_proj.weight"),
            ActiveByteCategory::DenseMlp
        );
        assert_eq!(
            classify_weight_name("model.layers.3.mlp.gate.weight"),
            ActiveByteCategory::Router
        );
        assert_eq!(
            classify_weight_name("model.layers.0.self_attn.o_proj.weight"),
            ActiveByteCategory::Attention
        );
        assert_eq!(
            classify_weight_name("model.layers.0.self_attn.indexer.wq_b.weight"),
            ActiveByteCategory::Indexer
        );
        assert_eq!(
            classify_weight_name("model.norm.weight"),
            ActiveByteCategory::Other
        );
    }

    #[test]
    fn active_bytes_categories_sum_to_total() {
        with_clean_ledger(|| {
            assert!(begin_token());
            record_active_bytes_for("lm_head.weight", 100);
            record_active_bytes_for(
                "model.layers.3.mlp.experts.0.gate_proj.weight",
                200,
            );
            record_active_bytes_for(
                "model.layers.3.mlp.shared_experts.up_proj.weight",
                50,
            );
            record_active_bytes_for("model.layers.0.self_attn.q_a_proj.weight", 30);
            record_active_bytes(7); // uncategorized → other
            let report = end_token().expect("report");
            assert_eq!(report.counters.active_bytes_read, 387);
            let cat = &report.counters.active_bytes_by_category;
            assert_eq!(cat["lm_head"].as_u64(), Some(100));
            assert_eq!(cat["routed_experts"].as_u64(), Some(200));
            assert_eq!(cat["shared_experts"].as_u64(), Some(50));
            assert_eq!(cat["attention"].as_u64(), Some(30));
            assert_eq!(cat["other"].as_u64(), Some(7));
            let sum: u64 = ActiveByteCategory::ALL
                .iter()
                .map(|c| cat[c.as_str()].as_u64().unwrap_or(0))
                .sum();
            assert_eq!(sum, report.counters.active_bytes_read);
        });
    }

    #[test]
    fn sealed_static_schedule_matches_header_derived_constants() {
        // STATIC (not live-measured): sizes taken from General-R0 shard
        // headers on 2026-07-26. See OVERREAD_BYTE_LEDGER.json.
        let s = sealed_glm_active_byte_schedule();
        assert_eq!(s.geometry_routed_bytes, SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES);
        // Categories must sum to the static total.
        let sum = s.routed_experts_bytes
            + s.shared_experts_bytes
            + s.dense_mlp_bytes
            + s.attention_bytes
            + s.indexer_bytes
            + s.router_bytes
            + s.lm_head_bytes;
        assert_eq!(sum, s.total_active_bytes);
        // lm_head alone is the dominant non-geometry term (~3.8 GB f32).
        assert_eq!(s.lm_head_bytes, 154_880u64 * 6_144 * 4);
        // Sparse layers are 75, not 78 — geometry over-counts 3 dense layers.
        assert_eq!(s.n_sparse_layers, 75);
        assert_eq!(s.n_dense_mlp_layers, 3);
        assert_eq!(s.n_full_indexer_layers, 21);
        // Static total is ~9.34 GB; the live mean was 10.46 GB — gap is
        // disclosed, not papered over.
        assert!((s.total_active_bytes as f64 - 9.34e9).abs() < 2e7);
    }

    #[test]
    fn bucket_names_are_gate_stable() {
        let names: Vec<_> = Bucket::ALL.iter().map(|b| b.as_str()).collect();
        assert!(names.contains(&"artifact_verification_and_sha"));
        assert!(names.contains(&"metal_encode"));
        assert!(names.contains(&"metal_submit"));
        assert!(names.contains(&"metal_synchronize_cpu_wait"));
        assert!(names.contains(&"attention_and_indexshare"));
        assert!(names.contains(&"routed_experts"));
        assert!(names.contains(&"norm"));
        assert!(names.contains(&"cpu_residual_scoped"));
        // Hard rule: no bucket literally named "orchestration" that absorbs remainder.
        assert!(!names.iter().any(|n| *n == "cpu_orchestration"));
        assert!(!names.iter().any(|n| *n == "orchestration"));
        assert_eq!(names.len(), 15);
    }

    #[test]
    fn percentiles_nearest_rank() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let p = Percentiles::from_slice(&samples);
        assert_eq!(p.n, 100);
        assert_eq!(p.min, 1.0);
        assert_eq!(p.max, 100.0);
        assert_eq!(p.p50, 50.0);
        assert_eq!(p.p95, 95.0);
        assert_eq!(p.p99, 99.0);
        assert!((p.mean - 50.5).abs() < 1e-9);
    }

    #[test]
    fn aggregate_synthetic_tokens_reports_tails_and_unattributed() {
        // Five synthetic tokens with increasing wall and a large unattributed
        // hole on the last one — must stay visible in p99, not absorbed.
        let mut reports = Vec::new();
        for i in 0usize..5 {
            let wall = 1_000_000 + (i as u64) * 200_000; // 1.0s .. 1.8s
            let metal_sync = 400_000u64;
            let encode = 50_000u64;
            let attn = 100_000u64;
            let attributed = metal_sync + encode + attn;
            let mut counters = TokenCounters::default();
            counters.command_buffers_submitted = 100 + i as u64;
            counters.synchronization_points = 100 + i as u64;
            counters.dispatches_encoded = 200;
            counters.active_bytes_read = SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES;
            counters.operations = 10_000 + i as u64;
            let mut device = DeviceTimeline::default();
            device.gpu_execution_us = 20_000 + i as u64 * 1_000;
            device.gpu_queue_wait_us = Some(380_000);
            device.gpu_timestamps_observed = 100;
            device.counter_sample_probed = true;
            device.counter_sample_supported = Some(true);
            device.counter_samples_recorded = 0;
            reports.push(synthetic_report(
                wall,
                &[
                    (Bucket::MetalSynchronize, metal_sync),
                    (Bucket::MetalEncode, encode),
                    (Bucket::AttentionAndIndexShare, attn),
                ],
                counters,
                device,
                Some(SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES),
                500 + i as u64 * 10, // disclosed profiler overhead
            ));
            assert_eq!(reports[i].attributed_us, attributed);
            assert_eq!(reports[i].unattributed_us, wall - attributed);
            // Never absorb into residual scoped bucket.
            assert_eq!(
                reports[i].buckets_us["cpu_residual_scoped"].as_u64().unwrap(),
                0
            );
        }

        let agg = aggregate_reports(&reports);
        assert_eq!(agg.token_count, 5);
        assert_eq!(agg.wall_us.p50, 1_400_000.0); // token index 2
        assert_eq!(agg.wall_us.p95, 1_800_000.0);
        assert_eq!(agg.wall_us.p99, 1_800_000.0);
        assert!(agg.unattributed_us.p50 > 400_000.0);
        assert!(agg.unattributed_us.max > agg.unattributed_us.p50);
        // Unattributed present as its own distribution line.
        assert!(agg.buckets_us.contains_key("unattributed"));
        let un_line = &agg.buckets_us["unattributed"];
        assert!(un_line.get("p99").and_then(|v| v.as_f64()).unwrap() > 1_000_000.0);
        assert_eq!(agg.device_gpu_execution_us.n, 5);
        assert_eq!(agg.tokens_missing_gpu_timestamps, 0);
        assert!(agg.profiler_overhead_us.mean >= 500.0);
        assert_eq!(
            agg.geometry_active_bytes,
            Some(SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES)
        );
        // Active bytes ≈ geometry.
        assert!((agg.active_bytes_read.mean - SEALED_ARTIFACT_ACTIVE_ROUTED_BYTES as f64).abs() < 1.0);
    }

    #[test]
    fn aggregate_marks_missing_gpu_timestamps() {
        let mut counters = TokenCounters::default();
        counters.command_buffers_submitted = 10;
        let device = DeviceTimeline {
            gpu_execution_us: 0,
            gpu_queue_wait_us: None,
            gpu_timestamps_observed: 0,
            gpu_timestamps_missing: 10,
            counter_sample_probed: true,
            counter_sample_supported: Some(false),
            counter_samples_recorded: 0,
            command_buffers: Vec::new(),
            notes: vec!["device has no timestamp counter set"],
        };
        let r = synthetic_report(
            100_000,
            &[(Bucket::MetalSynchronize, 80_000)],
            counters,
            device,
            None,
            100,
        );
        let agg = aggregate_reports(&[r]);
        assert_eq!(agg.tokens_missing_gpu_timestamps, 1);
        assert_eq!(agg.device_gpu_queue_wait_us.n, 0);
    }

    #[test]
    fn catalogue_lists_unattributed_and_device_lines() {
        let cat = bucket_source_catalogue();
        let names: Vec<String> = cat
            .iter()
            .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
            .collect();
        assert!(names.iter().any(|n| n == "unattributed"));
        assert!(names.iter().any(|n| n == "gpu_execution_us"));
        assert!(names.iter().any(|n| n == "gpu_queue_wait_us"));
        assert!(names.iter().any(|n| n == "profiler_overhead_us"));
        assert!(!names.iter().any(|n| n == "orchestration"));
    }

    #[test]
    fn record_gpu_command_buffer_zero_delta_is_missing() {
        with_clean_ledger(|| {
            assert!(begin_token());
            // end == start → treat as unavailable
            record_gpu_command_buffer(1, 100, Some(1.0), Some(1.0), 1);
            let report = end_token().expect("report");
            assert_eq!(report.device.gpu_timestamps_missing, 1);
            assert_eq!(report.device.gpu_execution_us, 0);
            assert!(report.device.gpu_queue_wait_us.is_none());
        });
    }
}
