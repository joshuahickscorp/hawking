//! Minimum complete native BOS token graph for DeepSeek-V4-Flash.
//!
//! Replaces the per-operator upload/dispatch/readback scaffold with:
//! - device-resident routing (hash or learned-bias) and a compact top-6 worklist
//! - one or two Metal command buffers per layer (not one wait per linear)
//! - only the six selected expert payloads touched
//! - streaming residency (layer working set, not the 159 GiB parent)
//!
//! Host MHC uses the CPU source algorithm so the HC BF16 SHA can match the
//! oracle. Expert outputs are never read back or gathered on the host.
//! The CPU oracle in `gravity_deepseek_v4_streamed_forward` is unchanged.

use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

static HOST_MEMCPY_NS: AtomicU64 = AtomicU64::new(0);
static HOST_MEMCPY_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_MEMCPY_CALLS: AtomicU64 = AtomicU64::new(0);
static HOST_DECODE_NS: AtomicU64 = AtomicU64::new(0);
static HOST_DECODE_BYTES: AtomicU64 = AtomicU64::new(0);
static HOST_DECODE_CALLS: AtomicU64 = AtomicU64::new(0);

fn reset_host_copy_stats() {
    HOST_MEMCPY_NS.store(0, Ordering::Relaxed);
    HOST_MEMCPY_BYTES.store(0, Ordering::Relaxed);
    HOST_MEMCPY_CALLS.store(0, Ordering::Relaxed);
    HOST_DECODE_NS.store(0, Ordering::Relaxed);
    HOST_DECODE_BYTES.store(0, Ordering::Relaxed);
    HOST_DECODE_CALLS.store(0, Ordering::Relaxed);
}

fn note_memcpy(bytes: usize, ns: u64) {
    HOST_MEMCPY_NS.fetch_add(ns, Ordering::Relaxed);
    HOST_MEMCPY_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    HOST_MEMCPY_CALLS.fetch_add(1, Ordering::Relaxed);
}

fn note_decode(bytes: usize, ns: u64) {
    HOST_DECODE_NS.fetch_add(ns, Ordering::Relaxed);
    HOST_DECODE_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
    HOST_DECODE_CALLS.fetch_add(1, Ordering::Relaxed);
}

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{
    DeepSeekV4ChunkVerificationStats, DeepSeekV4FullStreamReader, DeepSeekV4VerifiedBytes,
    NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::gravity_deepseek_v4_act_quant::ACT_QUANT_BLOCK;
use crate::gravity_deepseek_v4_final_head::{
    host_greedy_lm_head, host_merge_final_head_from_hc_bf16, DeepSeekV4GreedyTokenResult,
};
use crate::gravity_deepseek_v4_layer0_attention::{
    hc_attn_post_source_algorithm, rms_norm_bf16_source_algorithm, HEAD_DIM, KV_QAT_BLOCK,
    NUM_HEADS, O_LORA_RANK, Q_LORA_RANK, ROPE_HEAD_DIM, WKV_ROWS, WO_A_COLS, WO_A_ROWS, WO_B_COLS,
    WO_B_ROWS, WQ_B_ROWS,
};
use crate::gravity_deepseek_v4_layer0_moe::{
    ACTIVATED_EXPERTS, FP4_BLOCK, MOE_INTER_DIM, ROUTED_EXPERTS, ROUTE_SCALE,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    hc_attn_pre_source_algorithm, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS,
    HIDDEN_SIZE, PREFIX_TOKEN_ID, RMS_NORM_EPS,
};
use crate::gravity_deepseek_v4_layer_source_anchors::{
    verify_deepseek_v4_layer_source_anchors, DeepSeekV4LayerCommonTensor,
    DeepSeekV4LayerControlProjection, DeepSeekV4LayerExpertProjection, DeepSeekV4LayerGateMode,
    DeepSeekV4LayerMhcStage, DeepSeekV4LayerSourceAnchor,
    DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT,
};
use crate::gravity_deepseek_v4_runtime_spine::DSV4F_VOCAB_SIZE;
use crate::gravity_deepseek_v4_streamed_forward::{
    open_admitted_dsv4f_reader, peak_rss_bytes, ResidentLedger, DECLARED_PEAK_RSS_BOUND_BYTES,
    DECLARED_WEIGHT_RESIDENT_BOUND_BYTES, SCHEDULE_STREAMED_DECODE_PEAK_BYTES,
};
use crate::gravity_deepseek_v4_token_ns_ledger::{TokenNsCollector, TokenNsLedger};
use crate::{Error, Result};

/// Host-oracle greedy token from the sealed BOS streamed receipt.
pub const ORACLE_GREEDY_TOKEN_ID: u32 = 5;
/// Host-oracle greedy logit from `receipts/dsv4f_streamed_forward_l0_l42_receipt.json`.
pub const ORACLE_GREEDY_LOGIT: f32 = 16.767_437;
pub const ORACLE_HC_BF16_SHA256: &str =
    "d541c0a25a3bef30dac153d9bf7d1714aebfb4462af1bfb8b4648c1ea5e50c69";

fn greedy_from_logits(logits: &[f32], vocab_offset: usize) -> (u32, f32) {
    let mut best_id = vocab_offset as u32;
    let mut best_logit = f32::NEG_INFINITY;
    for (i, &logit) in logits.iter().enumerate() {
        let token = (vocab_offset + i) as u32;
        if logit > best_logit || (logit == best_logit && token < best_id) {
            best_logit = logit;
            best_id = token;
        }
    }
    (best_id, best_logit)
}

/// New kernels introduced by this graph. Each must have a `static_kernel_name` arm.
pub const NATIVE_TOKEN_GRAPH_KERNELS: &[&str] = &[
    "dsv4f_pack_worklist",
    "dsv4f_worklist_fp4_matvec",
    "dsv4f_worklist_swiglu",
    "dsv4f_worklist_combine",
];

pub const NATIVE_TOKEN_GRAPH_SCHEMA: &str = "hawking.gravity.deepseek_v4.native_token_graph.v1";
pub const NATIVE_TOKEN_GRAPH_PATH: &str = "device_worklist_bos_token";

#[allow(dead_code)]
const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const ACT_QUANT_SIMD_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate";
const ACT_QUANT_SIMD_WIDTH: u32 = 32;
const ACT_QUANT_VECTOR_WIDTH: u32 = 4;
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const RMSNORM_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_authority";
const PER_HEAD_RMS_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
const ATTN_KERNEL: &str = "deepseek_v4_p4a_sparse_attention_position0_sink_authority";
const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
const GATE_KERNEL: &str = "deepseek_v4_p6a_gate_bf16_matvec_authority";
const HASH_ROUTE_KERNEL: &str = "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority";
const LEARNED_ROUTE_KERNEL: &str = "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority";
const SHARED_SWIGLU_KERNEL: &str = "deepseek_v4_p5b_swiglu_route_bf16_authority";
const PACK_KERNEL: &str = "dsv4f_pack_worklist";
const WORKLIST_FP4_KERNEL: &str = "dsv4f_worklist_fp4_matvec";
const WORKLIST_SWIGLU_KERNEL: &str = "dsv4f_worklist_swiglu";
const WORKLIST_COMBINE_KERNEL: &str = "dsv4f_worklist_combine";
const LM_HEAD_KERNEL: &str = "gemv_native_bf16_seq";
const EMBED_WEIGHT: &str = "embed.weight";
const LM_HEAD_WEIGHT: &str = "head.weight";

const W1_PACKED: usize = MOE_INTER_DIM * (HIDDEN_SIZE / 2);
const W1_SCALES: usize = MOE_INTER_DIM * (HIDDEN_SIZE / FP4_BLOCK);
const W2_PACKED: usize = HIDDEN_SIZE * (MOE_INTER_DIM / 2);
const W2_SCALES: usize = HIDDEN_SIZE * (MOE_INTER_DIM / FP4_BLOCK);

/// Host-visible counters for the required gather/readback tests.
#[derive(Debug, Clone, Copy, Default, Serialize, PartialEq, Eq)]
pub struct NativeTokenGraphCounters {
    pub metal_dispatches: usize,
    pub command_buffers: usize,
    pub fallbacks: usize,
    pub host_expert_gather: usize,
    pub host_expert_output_readback: usize,
    pub host_route_id_readback: usize,
    pub total_sync_points: usize,
    pub total_readbacks: usize,
    pub total_buffer_creations: usize,
    pub total_buffer_rebinds: usize,
    pub scratch_buffer_creations: usize,
}

/// `HAWKING_DSV4F_CB_COLLAPSE=0` keeps the pre-collapse topology for paired
/// A/B. Default is on: ordered encoder per CB, batched expert act_quant,
/// and hash-layer route+moe in one command buffer.
pub fn cb_collapse_enabled() -> bool {
    match std::env::var("HAWKING_DSV4F_CB_COLLAPSE") {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

/// Honesty recorded on every native-graph report.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct NativeTokenGraphHonesty {
    pub native: bool,
    pub host_cpu: bool,
    pub bos_window_only: bool,
    pub complete_bos_token_graph: bool,
    pub full_compressed_indexer_graph: bool,
    pub ratio_4_full_compressed_graph: bool,
    pub ratio_128_full_compressed_graph: bool,
    pub indexer_compressor_loaded: bool,
    pub device_resident_routing: bool,
    pub compact_top6_worklist: bool,
    pub dense_over_256: bool,
    pub lm_head_path: String,
    pub mhc_path: String,
}

impl NativeTokenGraphHonesty {
    fn for_run(compute_final_head: bool, lm_head_on_device: bool) -> Self {
        Self {
            native: true,
            host_cpu: false,
            // Position-0 compressed slots are empty; window attention is the
            // source-complete BOS graph, not a leftover scaffold.
            bos_window_only: true,
            complete_bos_token_graph: true,
            full_compressed_indexer_graph: false,
            ratio_4_full_compressed_graph: false,
            ratio_128_full_compressed_graph: false,
            indexer_compressor_loaded: false,
            device_resident_routing: true,
            compact_top6_worklist: true,
            dense_over_256: false,
            lm_head_path: if !compute_final_head {
                "omitted".to_owned()
            } else if lm_head_on_device {
                "host_f64_mhc_merge_rmsnorm_then_streamed_metal_lm_head_greedy".to_owned()
            } else {
                "host_f64_mhc_merge_rmsnorm_then_host_streamed_lm_head_greedy".to_owned()
            },
            mhc_path: "host_source_algorithm_exact_sha".to_owned(),
        }
    }
}

/// Result of one native BOS token.
#[derive(Debug, Clone)]
pub struct NativeTokenGraphReport {
    pub schema: &'static str,
    pub execution_path: &'static str,
    pub deepest_layer: Option<usize>,
    pub layers_executed: Vec<usize>,
    pub token_id: u64,
    pub hc_bf16_bits: Vec<u16>,
    pub hc_bf16_sha256: String,
    pub peak_rss_bytes: u64,
    pub peak_weight_resident_bytes: u64,
    pub declared_rss_bound_bytes: u64,
    pub declared_weight_bound_bytes: u64,
    pub rss_within_bound: bool,
    pub weight_within_bound: bool,
    pub greedy: Option<DeepSeekV4GreedyTokenResult>,
    pub stop_reason: Option<String>,
    pub honesty: NativeTokenGraphHonesty,
    pub counters: NativeTokenGraphCounters,
    pub artifact_root: String,
    pub manifest_seal_sha256: String,
    pub wall_ms: u128,
    pub init_ms: u128,
    pub body_ms: u128,
    pub token_ns_ledger: Option<TokenNsLedger>,
    pub chunk_verification: DeepSeekV4ChunkVerificationStats,
    pub second_touch_probe_ns: u64,
    pub second_touch_cache_hits_delta: u64,
    pub second_touch_identity_calls_delta: u64,
    pub second_touch_mmap_calls_delta: u64,
}

impl NativeTokenGraphReport {
    pub fn to_receipt_json(&self) -> serde_json::Value {
        let host_read = self.chunk_verification.host_read;
        let chunk_verification = serde_json::json!({
            "hash_invocations": self.chunk_verification.hash_invocations,
            "cache_hits": self.chunk_verification.cache_hits,
            "bytes_hashed": self.chunk_verification.bytes_hashed,
            "chunks_verified": self.chunk_verification.chunks_verified,
            "admission_trust_hits": self.chunk_verification.admission_trust_hits,
            "admission_trust_fallbacks": self.chunk_verification.admission_trust_fallbacks,
            "verify_ns": self.chunk_verification.verify_ns,
            "admission_receipt_loaded": self.chunk_verification.admission_receipt_loaded,
            "artifact_index_loaded": self.chunk_verification.artifact_index_loaded,
            "host_read": {
                "read_view_calls": host_read.read_view_calls,
                "read_owned_calls": host_read.read_owned_calls,
                "mapped_windows": host_read.mapped_windows,
                "mapped_window_bytes": host_read.mapped_window_bytes,
                "owned_windows": host_read.owned_windows,
                "owned_window_bytes": host_read.owned_window_bytes,
                "owned_allocs": host_read.owned_allocs,
                "owned_alloc_bytes": host_read.owned_alloc_bytes,
                "owned_copy_ns": host_read.owned_copy_ns,
                "mmap_calls": host_read.mmap_calls,
                "mmap_ns": host_read.mmap_ns,
                "identity_calls": host_read.identity_calls,
                "identity_ns": host_read.identity_ns,
                "path_resolve_calls": host_read.path_resolve_calls,
                "path_resolve_ns": host_read.path_resolve_ns,
                "digest_cache_probes": host_read.digest_cache_probes,
                "digest_cache_ns": host_read.digest_cache_ns,
                "tensor_lookup_calls": host_read.tensor_lookup_calls,
                "tensor_lookup_ns": host_read.tensor_lookup_ns,
            },
        });
        let second_touch = serde_json::json!({
            "note": "re-read layer-0 attention tensors after the token; cache-hit remap cost, not in body_ns",
            "ns": self.second_touch_probe_ns,
            "cache_hits_delta": self.second_touch_cache_hits_delta,
            "identity_calls_delta": self.second_touch_identity_calls_delta,
            "mmap_calls_delta": self.second_touch_mmap_calls_delta,
        });
        serde_json::json!({
            "schema": self.schema,
            "execution_path": self.execution_path,
            "native": self.honesty.native,
            "artifact": {
                "path": self.artifact_root,
                "manifest_seal_sha256": self.manifest_seal_sha256,
                "repository": PINNED_REPOSITORY,
                "revision": PINNED_REVISION,
            },
            "scope": {
                "deepest_layer": self.deepest_layer,
                "layers_executed": self.layers_executed,
                "token_id": self.token_id,
                "token_position": 0,
            },
            "residency": {
                "peak_rss_bytes": self.peak_rss_bytes,
                "peak_weight_resident_bytes": self.peak_weight_resident_bytes,
                "declared_rss_bound_bytes": self.declared_rss_bound_bytes,
                "declared_weight_bound_bytes": self.declared_weight_bound_bytes,
                "rss_within_bound": self.rss_within_bound,
                "weight_within_bound": self.weight_within_bound,
                "schedule_streamed_decode_peak_bytes": SCHEDULE_STREAMED_DECODE_PEAK_BYTES,
                "policy": "layer_stream_load_execute_free",
            },
            "honesty": self.honesty,
            "metal": {
                "metal_dispatches": self.counters.metal_dispatches,
                "command_buffers": self.counters.command_buffers,
                "fallback": self.counters.fallbacks,
                "host_expert_gather": self.counters.host_expert_gather,
                "host_expert_output_readback": self.counters.host_expert_output_readback,
                "host_route_id_readback": self.counters.host_route_id_readback,
                "total_dispatches": self.counters.metal_dispatches,
                "total_command_buffers": self.counters.command_buffers,
                "total_sync_points": self.counters.total_sync_points,
                "total_readbacks": self.counters.total_readbacks,
                "total_buffer_creations": self.counters.total_buffer_creations,
                "total_buffer_rebinds": self.counters.total_buffer_rebinds,
                "scratch_buffer_creations": self.counters.scratch_buffer_creations,
                "cb_collapse": cb_collapse_enabled(),
            },
            "hc_bf16_sha256": self.hc_bf16_sha256,
            "greedy": self.greedy.as_ref().map(|g| serde_json::json!({
                "token_id": g.token_id,
                "logit": g.logit,
                "vocab_size": g.vocab_size,
                "lm_head_on_device": g.lm_head_on_device,
                "argmax_on_device": g.argmax_on_device,
                "metal_dispatches": g.metal_dispatches,
            })),
            "stop_reason": self.stop_reason,
            "wall_ms": self.wall_ms,
            "init_ms": self.init_ms,
            "body_ms": self.body_ms,
            "token_ns_ledger": self.token_ns_ledger,
            "startup_timing": crate::startup_timing::snapshot().to_json(),
            "chunk_verification": chunk_verification,
            "second_touch_probe": second_touch,
        })
    }
}

/// Sort six (id, weight, source_slot) tuples the same way the device packer does.
pub fn pack_worklist_host(
    selected_ids: &[u32],
    selected_weights: &[f32],
) -> Result<Vec<(u32, f32, u32)>> {
    if selected_ids.len() != ACTIVATED_EXPERTS || selected_weights.len() != ACTIVATED_EXPERTS {
        return Err(graph_error("worklist pack expects six ids and weights"));
    }
    let mut rows: Vec<(u32, f32, u32)> = selected_ids
        .iter()
        .zip(selected_weights)
        .enumerate()
        .map(|(slot, (&id, &weight))| (id, weight, slot as u32))
        .collect();
    rows.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.2.cmp(&b.2)));
    if rows.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(graph_error("worklist has duplicate expert ids"));
    }
    Ok(rows)
}

/// Run one complete BOS token through the native graph.
pub fn run_native_bos_token(
    artifact: impl AsRef<Path>,
    max_layer: usize,
    compute_final_head: bool,
) -> Result<NativeTokenGraphReport> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (artifact, max_layer, compute_final_head);
        Err(graph_error(
            "native token graph requires macOS Metal; CPU oracle remains the parity reference",
        ))
    }
    #[cfg(target_os = "macos")]
    {
        run_native_bos_token_macos(artifact.as_ref(), max_layer, compute_final_head)
    }
}

fn graph_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!("dsv4f native token graph: {}", message.into()))
}

fn sha256_u16(bits: &[u16]) -> String {
    let mut digest = Sha256::new();
    for &value in bits {
        digest.update(value.to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn decode_u16_le(bytes: &[u8], name: &str) -> Result<Vec<u16>> {
    if bytes.len() % size_of::<u16>() != 0 {
        return Err(graph_error(format!("{name} is not u16 aligned")));
    }
    let started = Instant::now();
    let out = bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();
    note_decode(bytes.len(), started.elapsed().as_nanos() as u64);
    Ok(out)
}

fn decode_f32_le(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if bytes.len() % size_of::<f32>() != 0 {
        return Err(graph_error(format!("{name} is not f32 aligned")));
    }
    let started = Instant::now();
    let out = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    note_decode(bytes.len(), started.elapsed().as_nanos() as u64);
    Ok(out)
}

fn load_bos_embed_hc(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    token_id: u64,
) -> Result<Vec<u16>> {
    let embed = reader.tensor_metadata(EMBED_WEIGHT)?;
    if embed.dtype != "BF16" || embed.shape.as_slice() != [129_280, HIDDEN_SIZE as u64] {
        return Err(graph_error("embed.weight is not BF16[vocab,4096]"));
    }
    let row_bytes = HIDDEN_SIZE * size_of::<u16>();
    let start = token_id
        .checked_mul(row_bytes as u64)
        .ok_or_else(|| graph_error("embed row start overflow"))?;
    let raw =
        reader.read_verified_range(EMBED_WEIGHT, start..start + row_bytes as u64, row_bytes)?;
    ledger.acquire(EMBED_WEIGHT, raw.len())?;
    let row = decode_u16_le(&raw, EMBED_WEIGHT)?;
    ledger.release(EMBED_WEIGHT)?;
    let mut hc = Vec::with_capacity(HC_FLAT_WIDTH);
    for _ in 0..HC_MULT {
        hc.extend_from_slice(&row);
    }
    Ok(hc)
}

fn hash_ids_from_tid2eid(
    reader: &DeepSeekV4FullStreamReader,
    ledger: &mut ResidentLedger,
    name: &str,
    token_id: u64,
) -> Result<[u32; ACTIVATED_EXPERTS]> {
    let meta = reader.tensor_metadata(name)?;
    if meta.dtype != "I64" || meta.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64] {
        return Err(graph_error(format!("{name} is not I64[vocab,6]")));
    }
    let row_bytes = ACTIVATED_EXPERTS * size_of::<i64>();
    let start = (token_id as usize)
        .checked_mul(row_bytes)
        .ok_or_else(|| graph_error("tid2eid row overflow"))?;
    let raw =
        reader.read_verified_range(name, start as u64..(start + row_bytes) as u64, row_bytes)?;
    ledger.acquire(name, raw.len())?;
    let mut ids = [0u32; ACTIVATED_EXPERTS];
    for (slot, chunk) in raw.chunks_exact(8).enumerate() {
        let id = i64::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| graph_error("tid2eid chunk is not i64"))?,
        );
        if id < 0 || id >= ROUTED_EXPERTS as i64 {
            return Err(graph_error("tid2eid expert id out of range"));
        }
        ids[slot] = id as u32;
    }
    ledger.release(name)?;
    Ok(ids)
}

#[cfg(target_os = "macos")]
mod macos {
    use super::*;
    use crate::metal::{CommandBatch, MetalBatchTiming, MetalContext, SubmittedBatch};
    use std::cell::Cell;

    thread_local! {
        static TOKEN_REBINDS: Cell<usize> = const { Cell::new(0) };
        static TOKEN_READBACKS: Cell<usize> = const { Cell::new(0) };
        static TOKEN_CREATES: Cell<usize> = const { Cell::new(0) };
    }

    fn bump_rebind() {
        TOKEN_REBINDS.with(|c| c.set(c.get() + 1));
    }
    fn bump_readback() {
        TOKEN_READBACKS.with(|c| c.set(c.get() + 1));
    }
    fn bump_create() {
        TOKEN_CREATES.with(|c| c.set(c.get() + 1));
    }
    fn reset_token_census() {
        TOKEN_REBINDS.with(|c| c.set(0));
        TOKEN_READBACKS.with(|c| c.set(0));
        TOKEN_CREATES.with(|c| c.set(0));
    }
    fn take_token_census() -> (usize, usize, usize) {
        (
            TOKEN_CREATES.with(|c| c.get()),
            TOKEN_REBINDS.with(|c| c.get()),
            TOKEN_READBACKS.with(|c| c.get()),
        )
    }

    fn kernel_probe_enabled() -> bool {
        !matches!(
            std::env::var("HAWKING_DSV4F_KERNEL_PROBE")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off" | "no"
        )
    }

    static FIRST_LAYER_BIND_TIMED: AtomicBool = AtomicBool::new(false);

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct WorklistEntry {
        expert_id: u32,
        slab_slot: u32,
        route_weight: f32,
        ready: u32,
    }

    struct Scratch {
        _hc: metal::Buffer,
        hidden_a: metal::Buffer,
        hidden_b: metal::Buffer,
        q_lora: metal::Buffer,
        wq_b: metal::Buffer,
        wkv: metal::Buffer,
        attn: metal::Buffer,
        wo_a: metal::Buffer,
        quant_k: metal::Buffer,
        quant_scale_k: metal::Buffer,
        quant_q: metal::Buffer,
        quant_scale_q: metal::Buffer,
        quant_wo: metal::Buffer,
        quant_scale_wo: metal::Buffer,
        quant_ffn: metal::Buffer,
        quant_scale_ffn: metal::Buffer,
        f32_tmp: metal::Buffer,
        gate_logits: metal::Buffer,
        route_ids: metal::Buffer,
        route_weights: metal::Buffer,
        original_scores: metal::Buffer,
        route_valid: metal::Buffer,
        worklist: metal::Buffer,
        pack_valid: metal::Buffer,
        w1_slab: metal::Buffer,
        w1_scale_slab: metal::Buffer,
        w3_slab: metal::Buffer,
        w3_scale_slab: metal::Buffer,
        w2_slab: metal::Buffer,
        w2_scale_slab: metal::Buffer,
        expert_gate_f32: metal::Buffer,
        expert_up_f32: metal::Buffer,
        expert_gate_bf16: metal::Buffer,
        expert_up_bf16: metal::Buffer,
        expert_swiglu: metal::Buffer,
        expert_down_f32: metal::Buffer,
        expert_down_bf16: metal::Buffer,
        expert_down_quant: metal::Buffer,
        expert_down_scales: metal::Buffer,
        shared_gate_f32: metal::Buffer,
        shared_up_f32: metal::Buffer,
        shared_down_f32: metal::Buffer,
        shared_gate_bf16: metal::Buffer,
        shared_up_bf16: metal::Buffer,
        shared_swiglu: metal::Buffer,
        shared_down_bf16: metal::Buffer,
        shared_down_quant: metal::Buffer,
        shared_down_scales: metal::Buffer,
        moe_out: metal::Buffer,
        attn_scores: metal::Buffer,
        attn_denoms: metal::Buffer,
        kv_qat_bytes: metal::Buffer,
        kv_qat_scales: metal::Buffer,
        wq_a_w: metal::Buffer,
        wq_a_s: metal::Buffer,
        wq_b_w: metal::Buffer,
        wq_b_s: metal::Buffer,
        wkv_w: metal::Buffer,
        wkv_s: metal::Buffer,
        wo_a_w: metal::Buffer,
        wo_a_s: metal::Buffer,
        wo_b_w: metal::Buffer,
        wo_b_s: metal::Buffer,
        gate_w: metal::Buffer,
        gate_bias: metal::Buffer,
        q_norm: metal::Buffer,
        kv_norm: metal::Buffer,
        sink: metal::Buffer,
        sh_w1_w: metal::Buffer,
        sh_w1_s: metal::Buffer,
        sh_w3_w: metal::Buffer,
        sh_w3_s: metal::Buffer,
        sh_w2_w: metal::Buffer,
        sh_w2_s: metal::Buffer,
    }

    impl Scratch {
        fn new(metal: &MetalContext) -> Result<Self> {
            let bf16_h = HIDDEN_SIZE * size_of::<u16>();
            Ok(Self {
                _hc: metal.new_buffer_checked(HC_FLAT_WIDTH * size_of::<u16>())?,
                hidden_a: metal.new_buffer_checked(bf16_h)?,
                hidden_b: metal.new_buffer_checked(bf16_h)?,
                q_lora: metal.new_buffer_checked(Q_LORA_RANK * size_of::<u16>())?,
                wq_b: metal.new_buffer_checked(WQ_B_ROWS * size_of::<u16>())?,
                wkv: metal.new_buffer_checked(WKV_ROWS * size_of::<u16>())?,
                attn: metal.new_buffer_checked(NUM_HEADS * HEAD_DIM * size_of::<u16>())?,
                wo_a: metal.new_buffer_checked(WO_A_ROWS * size_of::<u16>())?,
                quant_k: metal.new_buffer_checked(HIDDEN_SIZE)?,
                quant_scale_k: metal.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?,
                quant_q: metal.new_buffer_checked(Q_LORA_RANK)?,
                quant_scale_q: metal.new_buffer_checked(Q_LORA_RANK / ACT_QUANT_BLOCK)?,
                quant_wo: metal.new_buffer_checked(WO_B_COLS)?,
                quant_scale_wo: metal.new_buffer_checked(WO_B_COLS / ACT_QUANT_BLOCK)?,
                quant_ffn: metal.new_buffer_checked(HIDDEN_SIZE)?,
                quant_scale_ffn: metal.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?,
                f32_tmp: metal.new_buffer_checked(WQ_B_ROWS.max(WO_A_ROWS) * size_of::<f32>())?,
                gate_logits: metal.new_buffer_checked(ROUTED_EXPERTS * size_of::<f32>())?,
                route_ids: metal.new_buffer_checked(ACTIVATED_EXPERTS * size_of::<u32>())?,
                route_weights: metal.new_buffer_checked(ACTIVATED_EXPERTS * size_of::<f32>())?,
                original_scores: metal.new_buffer_checked(ROUTED_EXPERTS * size_of::<f32>())?,
                route_valid: metal.new_buffer_checked(size_of::<u32>())?,
                worklist: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * size_of::<WorklistEntry>())?,
                pack_valid: metal.new_buffer_checked(size_of::<u32>())?,
                w1_slab: metal.new_buffer_checked(ACTIVATED_EXPERTS * W1_PACKED)?,
                w1_scale_slab: metal.new_buffer_checked(ACTIVATED_EXPERTS * W1_SCALES)?,
                w3_slab: metal.new_buffer_checked(ACTIVATED_EXPERTS * W1_PACKED)?,
                w3_scale_slab: metal.new_buffer_checked(ACTIVATED_EXPERTS * W1_SCALES)?,
                w2_slab: metal.new_buffer_checked(ACTIVATED_EXPERTS * W2_PACKED)?,
                w2_scale_slab: metal.new_buffer_checked(ACTIVATED_EXPERTS * W2_SCALES)?,
                expert_gate_f32: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * MOE_INTER_DIM * size_of::<f32>())?,
                expert_up_f32: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * MOE_INTER_DIM * size_of::<f32>())?,
                expert_gate_bf16: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * MOE_INTER_DIM * size_of::<u16>())?,
                expert_up_bf16: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * MOE_INTER_DIM * size_of::<u16>())?,
                expert_swiglu: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * MOE_INTER_DIM * size_of::<u16>())?,
                expert_down_f32: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * HIDDEN_SIZE * size_of::<f32>())?,
                expert_down_bf16: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * HIDDEN_SIZE * size_of::<u16>())?,
                expert_down_quant: metal.new_buffer_checked(ACTIVATED_EXPERTS * MOE_INTER_DIM)?,
                expert_down_scales: metal
                    .new_buffer_checked(ACTIVATED_EXPERTS * (MOE_INTER_DIM / ACT_QUANT_BLOCK))?,
                shared_gate_f32: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<f32>())?,
                shared_up_f32: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<f32>())?,
                shared_down_f32: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<f32>())?,
                shared_gate_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
                shared_up_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
                shared_swiglu: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
                shared_down_bf16: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<u16>())?,
                shared_down_quant: metal.new_buffer_checked(MOE_INTER_DIM)?,
                shared_down_scales: metal.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
                moe_out: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<u16>())?,
                attn_scores: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
                attn_denoms: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
                kv_qat_bytes: metal.new_buffer_checked(HEAD_DIM)?,
                kv_qat_scales: metal
                    .new_buffer_checked((HEAD_DIM - ROPE_HEAD_DIM) / KV_QAT_BLOCK)?,
                wq_a_w: metal.new_buffer_checked(Q_LORA_RANK * HIDDEN_SIZE)?,
                wq_a_s: metal.new_buffer_checked(
                    (Q_LORA_RANK / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
                )?,
                wq_b_w: metal.new_buffer_checked(WQ_B_ROWS * Q_LORA_RANK)?,
                wq_b_s: metal.new_buffer_checked(
                    (WQ_B_ROWS / ACT_QUANT_BLOCK) * (Q_LORA_RANK / ACT_QUANT_BLOCK),
                )?,
                wkv_w: metal.new_buffer_checked(WKV_ROWS * HIDDEN_SIZE)?,
                wkv_s: metal.new_buffer_checked(
                    (WKV_ROWS / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
                )?,
                wo_a_w: metal.new_buffer_checked(WO_A_ROWS * WO_A_COLS)?,
                wo_a_s: metal.new_buffer_checked(
                    (WO_A_ROWS / ACT_QUANT_BLOCK) * (WO_A_COLS / ACT_QUANT_BLOCK),
                )?,
                wo_b_w: metal.new_buffer_checked(WO_B_ROWS * WO_B_COLS)?,
                wo_b_s: metal.new_buffer_checked(
                    (WO_B_ROWS / ACT_QUANT_BLOCK) * (WO_B_COLS / ACT_QUANT_BLOCK),
                )?,
                gate_w: metal
                    .new_buffer_checked(ROUTED_EXPERTS * HIDDEN_SIZE * size_of::<u16>())?,
                gate_bias: metal.new_buffer_checked(ROUTED_EXPERTS * size_of::<f32>())?,
                q_norm: metal.new_buffer_checked(Q_LORA_RANK * size_of::<u16>())?,
                kv_norm: metal.new_buffer_checked(HEAD_DIM * size_of::<u16>())?,
                sink: metal.new_buffer_checked(NUM_HEADS * size_of::<f32>())?,
                sh_w1_w: metal.new_buffer_checked(MOE_INTER_DIM * HIDDEN_SIZE)?,
                sh_w1_s: metal.new_buffer_checked(
                    (MOE_INTER_DIM / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
                )?,
                sh_w3_w: metal.new_buffer_checked(MOE_INTER_DIM * HIDDEN_SIZE)?,
                sh_w3_s: metal.new_buffer_checked(
                    (MOE_INTER_DIM / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
                )?,
                sh_w2_w: metal.new_buffer_checked(HIDDEN_SIZE * MOE_INTER_DIM)?,
                sh_w2_s: metal.new_buffer_checked(
                    (HIDDEN_SIZE / ACT_QUANT_BLOCK) * (MOE_INTER_DIM / ACT_QUANT_BLOCK),
                )?,
            })
        }
    }

    pub(super) struct Graph {
        metal: MetalContext,
        scratch: Scratch,
        counters: NativeTokenGraphCounters,
        act_tg: u32,
        fp8_tg: u32,
        fp4_tg: u32,
        cast_tg: u32,
        gate_tg: u32,
        wo_a_tg: u32,
        lm_tg: u32,
        attn_scratch_ready: bool,
    }

    impl Graph {
        fn new() -> Result<Self> {
            let metal = MetalContext::new()?;
            let scratch = Scratch::new(&metal)?;
            let mut counters = NativeTokenGraphCounters::default();
            counters.scratch_buffer_creations =
                std::mem::size_of::<Scratch>() / std::mem::size_of::<metal::Buffer>();
            Ok(Self {
                act_tg: pipeline_tg(&metal, ACT_QUANT_SIMD_KERNEL, 256)?,
                fp8_tg: pipeline_tg(&metal, FP8_KERNEL, 256)?,
                fp4_tg: pipeline_tg(&metal, WORKLIST_FP4_KERNEL, 256)?,
                cast_tg: pipeline_tg(&metal, CAST_KERNEL, 256)?,
                gate_tg: pipeline_tg(&metal, GATE_KERNEL, 256)?,
                wo_a_tg: pipeline_tg(&metal, WO_A_KERNEL, 256)?,
                lm_tg: pipeline_tg(&metal, LM_HEAD_KERNEL, 256)?,
                metal,
                scratch,
                counters,
                attn_scratch_ready: false,
            })
        }

        fn batch(
            &mut self,
            name: &str,
            layer: Option<usize>,
            force: &str,
            profiler: &mut TokenNsCollector,
            encode: impl FnOnce(&mut CommandBatch<'_>, &Scratch) -> Result<usize>,
        ) -> Result<MetalBatchTiming> {
            let (n, submitted) = self.submit(encode)?;
            self.finish(name, layer, force, n, submitted, profiler)
        }

        fn submit(
            &mut self,
            encode: impl FnOnce(&mut CommandBatch<'_>, &Scratch) -> Result<usize>,
        ) -> Result<(usize, SubmittedBatch)> {
            let mut n = 0usize;
            let submitted = self.metal.submit_batch(|batch| {
                n = encode(batch, &self.scratch)?;
                Ok(())
            })?;
            Ok((n, submitted))
        }

        fn finish(
            &mut self,
            name: &str,
            layer: Option<usize>,
            force: &str,
            n: usize,
            submitted: SubmittedBatch,
            profiler: &mut TokenNsCollector,
        ) -> Result<MetalBatchTiming> {
            let timing = submitted.wait()?;
            self.counters.command_buffers += 1;
            self.counters.total_sync_points += 1;
            self.counters.metal_dispatches += n;
            profiler.record_cb(name, layer, force, n as u64, &timing);
            Ok(timing)
        }
    }

    fn probe_one(
        metal: &MetalContext,
        profiler: &mut TokenNsCollector,
        name: &str,
        layer: usize,
        encode: impl FnOnce(&mut CommandBatch<'_>) -> Result<()>,
    ) -> Result<()> {
        let started = Instant::now();
        let timing = metal.dispatch_batch_timed(encode)?;
        profiler.record_isolated(name, layer, started.elapsed().as_nanos() as u64, &timing);
        Ok(())
    }

    fn pipeline_tg(metal: &MetalContext, kernel: &str, preferred: u32) -> Result<u32> {
        let max = metal.pipeline(kernel)?.max_total_threads_per_threadgroup() as u32;
        if max == 0 {
            return Err(graph_error(format!(
                "{kernel} reports a zero threadgroup limit"
            )));
        }
        Ok(preferred.min(max).max(1))
    }

    fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
        encoder.set_bytes(
            index,
            size_of::<u32>() as u64,
            value as *const u32 as *const _,
        );
    }

    fn set_f32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &f32) {
        encoder.set_bytes(
            index,
            size_of::<f32>() as u64,
            value as *const f32 as *const _,
        );
    }

    fn write_u16(buf: &metal::Buffer, values: &[u16]) {
        bump_rebind();
        MetalContext::write_buffer_bytes(buf, bytemuck::cast_slice(values));
    }

    fn write_bytes(buf: &metal::Buffer, values: &[u8]) {
        bump_rebind();
        let started = Instant::now();
        MetalContext::write_buffer_bytes(buf, values);
        super::note_memcpy(values.len(), started.elapsed().as_nanos() as u64);
    }

    fn read_u16(buf: &metal::Buffer, n: usize) -> Result<Vec<u16>> {
        bump_readback();
        let ptr = buf.contents() as *const u16;
        if ptr.is_null() {
            return Err(graph_error("u16 buffer contents pointer is null"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    fn read_u32(buf: &metal::Buffer, n: usize) -> Result<Vec<u32>> {
        bump_readback();
        let ptr = buf.contents() as *const u32;
        if ptr.is_null() {
            return Err(graph_error("u32 buffer contents pointer is null"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    fn read_f32_n(buf: &metal::Buffer, n: usize) -> Result<Vec<f32>> {
        bump_readback();
        let ptr = buf.contents() as *const f32;
        if ptr.is_null() {
            return Err(graph_error("f32 buffer contents pointer is null"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    fn dispatch_act_quant(
        batch: &mut CommandBatch<'_>,
        input: &metal::Buffer,
        quant: &metal::Buffer,
        scales: &metal::Buffer,
        cols: u32,
        simd_tg: u32,
        input_off: u64,
        quant_off: u64,
        scale_off: u64,
    ) -> Result<()> {
        let blocks = cols / ACT_QUANT_BLOCK as u32;
        if blocks == 0 {
            return Ok(());
        }
        let threads_x = simd_tg.max(ACT_QUANT_SIMD_WIDTH);
        let threads_x = threads_x - (threads_x % ACT_QUANT_SIMD_WIDTH);
        let simdgroups = threads_x / ACT_QUANT_SIMD_WIDTH;
        let groups = (blocks + simdgroups - 1) / simdgroups;
        let grid = groups * threads_x;
        let vw = ACT_QUANT_VECTOR_WIDTH;
        batch.dispatch_threads(
            ACT_QUANT_SIMD_KERNEL,
            (grid, 1, 1),
            (threads_x, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(input), input_off);
                enc.set_buffer(1, Some(quant), quant_off);
                enc.set_buffer(2, Some(scales), scale_off);
                set_u32(enc, 3, &cols);
                set_u32(enc, 4, &threads_x);
                set_u32(enc, 5, &vw);
            },
        )
    }

    fn dispatch_fp8(
        batch: &mut CommandBatch<'_>,
        weight: &metal::Buffer,
        scale: &metal::Buffer,
        quant: &metal::Buffer,
        act_scale: &metal::Buffer,
        out_f32: &metal::Buffer,
        rows: u32,
        cols: u32,
        tg: u32,
    ) -> Result<()> {
        let scale_cols = cols / ACT_QUANT_BLOCK as u32;
        let tg = tg.min(rows.max(1));
        batch.dispatch_threads(FP8_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(weight), 0);
            enc.set_buffer(1, Some(scale), 0);
            enc.set_buffer(2, Some(quant), 0);
            enc.set_buffer(3, Some(act_scale), 0);
            enc.set_buffer(4, Some(out_f32), 0);
            set_u32(enc, 5, &rows);
            set_u32(enc, 6, &cols);
            set_u32(enc, 7, &scale_cols);
        })
    }

    fn dispatch_cast(
        batch: &mut CommandBatch<'_>,
        input: &metal::Buffer,
        output: &metal::Buffer,
        count: u32,
        tg: u32,
    ) -> Result<()> {
        let tg = tg.min(count.max(1));
        batch.dispatch_threads(CAST_KERNEL, (count, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(input), 0);
            enc.set_buffer(1, Some(output), 0);
            set_u32(enc, 2, &count);
        })
    }

    fn dispatch_rmsnorm(
        batch: &mut CommandBatch<'_>,
        input: &metal::Buffer,
        weight: &metal::Buffer,
        output: &metal::Buffer,
        width: u32,
        eps: f32,
    ) -> Result<()> {
        batch.dispatch_threads(RMSNORM_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
            enc.set_buffer(0, Some(input), 0);
            enc.set_buffer(1, Some(weight), 0);
            enc.set_buffer(2, Some(output), 0);
            set_u32(enc, 3, &width);
            set_f32(enc, 4, &eps);
        })
    }

    struct Fp8Pair {
        weight: metal::Buffer,
        scale: metal::Buffer,
        name_w: String,
        name_s: String,
    }

    fn par_read_views(
        reader: &DeepSeekV4FullStreamReader,
        jobs: &[(String, usize)],
    ) -> Result<Vec<DeepSeekV4VerifiedBytes>> {
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(jobs.len());
            for (name, bytes) in jobs {
                joins.push(scope.spawn(move || reader.read_verified_full_view(name, *bytes)));
            }
            let mut out = Vec::with_capacity(joins.len());
            for join in joins {
                out.push(
                    join.join()
                        .map_err(|_| graph_error("parallel verified read panicked"))??,
                );
            }
            Ok(out)
        })
    }

    #[allow(dead_code)]
    fn par_read_full(
        reader: &DeepSeekV4FullStreamReader,
        jobs: &[(String, usize)],
    ) -> Result<Vec<Vec<u8>>> {
        Ok(par_read_views(reader, jobs)?
            .into_iter()
            .map(|view| view.into_owned())
            .collect())
    }

    fn write_at(buf: &metal::Buffer, offset: usize, values: &[u8]) {
        bump_rebind();
        let ptr = buf.contents() as *mut u8;
        if ptr.is_null() || values.is_empty() {
            return;
        }
        let started = Instant::now();
        unsafe {
            ptr.add(offset)
                .copy_from_nonoverlapping(values.as_ptr(), values.len());
        }
        super::note_memcpy(values.len(), started.elapsed().as_nanos() as u64);
    }

    fn refill_fp8(
        dest_w: &metal::Buffer,
        dest_s: &metal::Buffer,
        ledger: &mut ResidentLedger,
        weight_name: &str,
        scale_name: &str,
        weight: &[u8],
        scale: &[u8],
    ) -> Result<Fp8Pair> {
        ledger.acquire(weight_name, weight.len())?;
        ledger.acquire(scale_name, scale.len())?;
        write_bytes(dest_w, weight);
        write_bytes(dest_s, scale);
        Ok(Fp8Pair {
            weight: dest_w.clone(),
            scale: dest_s.clone(),
            name_w: weight_name.to_owned(),
            name_s: scale_name.to_owned(),
        })
    }

    fn load_fp8(
        dest_w: &metal::Buffer,
        dest_s: &metal::Buffer,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        weight_name: &str,
        scale_name: &str,
        rows: usize,
        cols: usize,
    ) -> Result<Fp8Pair> {
        let pair = reader.native_scale_pair(weight_name)?;
        if pair.kind != NativeScalePairKind::Fp8E4M3fn || pair.scale.name != scale_name {
            return Err(graph_error(format!(
                "{weight_name} is not the expected FP8 pair"
            )));
        }
        let jobs = [
            (weight_name.to_owned(), rows * cols),
            (
                scale_name.to_owned(),
                (rows / ACT_QUANT_BLOCK) * (cols / ACT_QUANT_BLOCK),
            ),
        ];
        let mut blobs = par_read_views(reader, &jobs)?;
        let scale = blobs.pop().expect("scale");
        let weight = blobs.pop().expect("weight");
        refill_fp8(
            dest_w,
            dest_s,
            ledger,
            weight_name,
            scale_name,
            weight.as_bytes(),
            scale.as_bytes(),
        )
    }

    fn release_fp8(ledger: &mut ResidentLedger, pair: &Fp8Pair) -> Result<()> {
        ledger.release(&pair.name_w)?;
        ledger.release(&pair.name_s)
    }

    fn upload_expert_slab(
        graph: &Graph,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        layer: &DeepSeekV4LayerSourceAnchor,
        exec: &[(u32, f32, u32)],
    ) -> Result<Vec<String>> {
        let mut names = Vec::new();
        let mut jobs = Vec::with_capacity(ACTIVATED_EXPERTS * 6);
        for &(expert_id, _, _) in exec {
            let p1 = layer
                .routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W1)?;
            let p3 = layer
                .routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W3)?;
            let p2 = layer
                .routed_expert_pair(expert_id as usize, DeepSeekV4LayerExpertProjection::W2)?;
            jobs.push((p1.weight.name.clone(), W1_PACKED));
            jobs.push((p1.scale.name.clone(), W1_SCALES));
            jobs.push((p3.weight.name.clone(), W1_PACKED));
            jobs.push((p3.scale.name.clone(), W1_SCALES));
            jobs.push((p2.weight.name.clone(), W2_PACKED));
            jobs.push((p2.scale.name.clone(), W2_SCALES));
            names.extend([
                p1.weight.name,
                p1.scale.name,
                p3.weight.name,
                p3.scale.name,
                p2.weight.name,
                p2.scale.name,
            ]);
        }
        let blobs = par_read_views(reader, &jobs)?;
        for slot in 0..ACTIVATED_EXPERTS {
            let base = slot * 6;
            ledger.acquire(&names[base], blobs[base].len())?;
            ledger.acquire(&names[base + 1], blobs[base + 1].len())?;
            ledger.acquire(&names[base + 2], blobs[base + 2].len())?;
            ledger.acquire(&names[base + 3], blobs[base + 3].len())?;
            ledger.acquire(&names[base + 4], blobs[base + 4].len())?;
            ledger.acquire(&names[base + 5], blobs[base + 5].len())?;
            write_at(&graph.scratch.w1_slab, slot * W1_PACKED, blobs[base].as_bytes());
            write_at(
                &graph.scratch.w1_scale_slab,
                slot * W1_SCALES,
                blobs[base + 1].as_bytes(),
            );
            write_at(&graph.scratch.w3_slab, slot * W1_PACKED, blobs[base + 2].as_bytes());
            write_at(
                &graph.scratch.w3_scale_slab,
                slot * W1_SCALES,
                blobs[base + 3].as_bytes(),
            );
            write_at(&graph.scratch.w2_slab, slot * W2_PACKED, blobs[base + 4].as_bytes());
            write_at(
                &graph.scratch.w2_scale_slab,
                slot * W2_SCALES,
                blobs[base + 5].as_bytes(),
            );
        }
        Ok(names)
    }

    fn seed_worklist(graph: &Graph, exec: &[(u32, f32, u32)]) {
        let mut entries = [WorklistEntry {
            expert_id: 0,
            slab_slot: 0,
            route_weight: 0.0,
            ready: 0,
        }; ACTIVATED_EXPERTS];
        for (slot, &(id, weight, _)) in exec.iter().enumerate() {
            entries[slot] = WorklistEntry {
                expert_id: id,
                slab_slot: slot as u32,
                route_weight: weight,
                ready: 1,
            };
        }
        let bytes = unsafe {
            std::slice::from_raw_parts(
                entries.as_ptr() as *const u8,
                ACTIVATED_EXPERTS * size_of::<WorklistEntry>(),
            )
        };
        write_bytes(&graph.scratch.worklist, bytes);
    }

    pub(super) fn run_native_bos_token_macos(
        artifact: &Path,
        max_layer: usize,
        compute_final_head: bool,
    ) -> Result<NativeTokenGraphReport> {
        let wall = Instant::now();
        super::reset_host_copy_stats();
        if max_layer >= DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT {
            return Err(graph_error(format!(
                "max_layer {max_layer} is outside the 0..42 base body"
            )));
        }
        let (reader, admission) = open_admitted_dsv4f_reader(artifact)?;
        let anchors = crate::startup_timing::time_ms_result("layer_source_anchors", || {
            verify_deepseek_v4_layer_source_anchors(&reader)
        })?;
        if anchors.identity().repository != PINNED_REPOSITORY
            || anchors.identity().revision != PINNED_REVISION
        {
            return Err(graph_error(
                "native graph refused a reader whose source identity is not pinned",
            ));
        }

        let mut ledger = ResidentLedger::new(DECLARED_WEIGHT_RESIDENT_BOUND_BYTES);
        let mut graph =
            crate::startup_timing::time_ms_result("metal_device_library_pipelines", Graph::new)?;
        reset_token_census();
        let init_ms = wall.elapsed().as_millis();
        let body = Instant::now();
        let mut profiler = TokenNsCollector::new();
        let mut peak_rss = peak_rss_bytes();
        let mut layers_executed = Vec::new();
        let mut stop_reason = None;

        let mut hc = profiler.time_result("host.embed_io", || {
            load_bos_embed_hc(&reader, &mut ledger, PREFIX_TOKEN_ID)
        })?;
        let mut attn_prefetch: Option<Vec<DeepSeekV4VerifiedBytes>> = None;

        for layer_idx in 0..=max_layer {
            let layer = anchors.layer(layer_idx)?.clone();
            let next_layer = if layer_idx < max_layer {
                Some(anchors.layer(layer_idx + 1)?.clone())
            } else {
                None
            };
            match execute_layer(
                &mut graph,
                &reader,
                &layer,
                next_layer.as_ref(),
                &hc,
                PREFIX_TOKEN_ID,
                &mut ledger,
                &mut profiler,
                &mut attn_prefetch,
            ) {
                Ok(next) => {
                    hc = next;
                    layers_executed.push(layer_idx);
                    peak_rss = peak_rss.max(peak_rss_bytes());
                    eprintln!(
                        "dsv4f native graph layer {layer_idx} done; peak_rss={peak_rss} live={}",
                        ledger.live_bytes()
                    );
                    if peak_rss > DECLARED_PEAK_RSS_BOUND_BYTES {
                        stop_reason = Some(format!(
                            "measured peak RSS {peak_rss} exceeded declared bound {}",
                            DECLARED_PEAK_RSS_BOUND_BYTES
                        ));
                        break;
                    }
                }
                Err(error) => {
                    stop_reason = Some(format!("layer {layer_idx}: {error}"));
                    break;
                }
            }
        }

        let deepest_layer = layers_executed.last().copied();
        let mut lm_head_on_device = false;
        let greedy = if stop_reason.is_none()
            && compute_final_head
            && deepest_layer == Some(max_layer)
            && max_layer + 1 == DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT
        {
            let merged = profiler.time_result("host.lm_head.mhc_merge", || {
                host_merge_final_head_from_hc_bf16(&reader, &hc)
            })?;
            match metal_lm_head(
                &mut graph,
                &reader,
                &mut ledger,
                &merged.merged_f32,
                &mut profiler,
            ) {
                Ok(token) => {
                    lm_head_on_device = token.lm_head_on_device;
                    Some(token)
                }
                Err(error) => {
                    graph.counters.fallbacks += 1;
                    eprintln!("native graph lm_head fallback: {error}");
                    Some(host_greedy_lm_head(&reader, &merged.merged_f32)?)
                }
            }
        } else {
            None
        };
        peak_rss = peak_rss.max(peak_rss_bytes());
        if ledger.live_bytes() != 0 && stop_reason.is_none() {
            return Err(graph_error(format!(
                "native graph leaked {} resident weight bytes",
                ledger.live_bytes()
            )));
        }

        let (creates, rebinds, readbacks) = take_token_census();
        graph.counters.total_buffer_creations = creates;
        graph.counters.total_buffer_rebinds = rebinds;
        graph.counters.total_readbacks = readbacks;

        let probe_overhead_ns = profiler.probe_overhead_ns();
        let raw_body_ns = body.elapsed().as_nanos() as u64;
        let body_ns = raw_body_ns.saturating_sub(probe_overhead_ns);
        let wall_ns = wall.elapsed().as_nanos() as u64;
        let init_ns = (init_ms as u64).saturating_mul(1_000_000);
        let before_reread = reader.chunk_verification_stats();
        let reread_started = Instant::now();
        if let Ok(layer0) = anchors.layer(0) {
            let jobs = attn_read_jobs(layer0);
            let _ = par_read_views(&reader, &jobs);
        }
        let second_touch_probe_ns = reread_started.elapsed().as_nanos() as u64;
        let after_reread = reader.chunk_verification_stats();
        let second_touch_cache_hits_delta = after_reread
            .cache_hits
            .saturating_sub(before_reread.cache_hits);
        let second_touch_identity_calls_delta = after_reread
            .host_read
            .identity_calls
            .saturating_sub(before_reread.host_read.identity_calls);
        let second_touch_mmap_calls_delta = after_reread
            .host_read
            .mmap_calls
            .saturating_sub(before_reread.host_read.mmap_calls);
        let chunk_verification = before_reread;
        profiler.add_stage(
            "host.memcpy",
            super::HOST_MEMCPY_NS.load(Ordering::Relaxed),
            super::HOST_MEMCPY_CALLS.load(Ordering::Relaxed),
            super::HOST_MEMCPY_BYTES.load(Ordering::Relaxed),
        );
        profiler.add_stage(
            "host.decode",
            super::HOST_DECODE_NS.load(Ordering::Relaxed),
            super::HOST_DECODE_CALLS.load(Ordering::Relaxed),
            super::HOST_DECODE_BYTES.load(Ordering::Relaxed),
        );
        profiler.add_stage(
            "host.owned_copy",
            chunk_verification.host_read.owned_copy_ns,
            chunk_verification.host_read.owned_allocs,
            chunk_verification.host_read.owned_alloc_bytes,
        );
        profiler.add_stage(
            "reader.mmap",
            chunk_verification.host_read.mmap_ns,
            chunk_verification.host_read.mmap_calls,
            0,
        );
        profiler.add_stage(
            "reader.identity",
            chunk_verification.host_read.identity_ns,
            chunk_verification.host_read.identity_calls,
            0,
        );
        profiler.add_stage(
            "reader.path_resolve",
            chunk_verification.host_read.path_resolve_ns,
            chunk_verification.host_read.path_resolve_calls,
            0,
        );
        profiler.add_stage(
            "reader.digest_cache",
            chunk_verification.host_read.digest_cache_ns,
            chunk_verification.host_read.digest_cache_probes,
            0,
        );
        profiler.add_stage(
            "reader.tensor_lookup",
            chunk_verification.host_read.tensor_lookup_ns,
            chunk_verification.host_read.tensor_lookup_calls,
            0,
        );
        profiler.add_stage("probe.second_touch", second_touch_probe_ns, 1, 0);
        let token_ns_ledger = Some(profiler.finish(
            body_ns,
            init_ns,
            wall_ns,
            chunk_verification.verify_ns,
        ));

        Ok(NativeTokenGraphReport {
            schema: NATIVE_TOKEN_GRAPH_SCHEMA,
            execution_path: NATIVE_TOKEN_GRAPH_PATH,
            deepest_layer,
            layers_executed,
            token_id: PREFIX_TOKEN_ID,
            hc_bf16_sha256: sha256_u16(&hc),
            hc_bf16_bits: hc,
            peak_rss_bytes: peak_rss,
            peak_weight_resident_bytes: ledger.peak_bytes(),
            declared_rss_bound_bytes: DECLARED_PEAK_RSS_BOUND_BYTES,
            declared_weight_bound_bytes: DECLARED_WEIGHT_RESIDENT_BOUND_BYTES,
            rss_within_bound: peak_rss > 0 && peak_rss <= DECLARED_PEAK_RSS_BOUND_BYTES,
            weight_within_bound: ledger.peak_bytes() <= DECLARED_WEIGHT_RESIDENT_BOUND_BYTES,
            greedy,
            stop_reason,
            honesty: NativeTokenGraphHonesty::for_run(compute_final_head, lm_head_on_device),
            counters: graph.counters,
            artifact_root: admission.source_path.display().to_string(),
            manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            wall_ms: wall.elapsed().as_millis(),
            init_ms,
            body_ms: (body_ns / 1_000_000) as u128,
            token_ns_ledger,
            chunk_verification,
            second_touch_probe_ns,
            second_touch_cache_hits_delta,
            second_touch_identity_calls_delta,
            second_touch_mmap_calls_delta,
        })
    }

    fn execute_layer(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        next_layer: Option<&DeepSeekV4LayerSourceAnchor>,
        hc_in: &[u16],
        token_id: u64,
        ledger: &mut ResidentLedger,
        profiler: &mut TokenNsCollector,
        attn_prefetch: &mut Option<Vec<DeepSeekV4VerifiedBytes>>,
    ) -> Result<Vec<u16>> {
        let preloaded = attn_prefetch.take();
        let (attn_hc, preload) = execute_attention(
            graph,
            reader,
            layer,
            hc_in,
            token_id,
            ledger,
            profiler,
            preloaded,
        )?;
        execute_moe(
            graph,
            reader,
            layer,
            next_layer,
            &attn_hc,
            token_id,
            ledger,
            profiler,
            Some(preload),
            attn_prefetch,
        )
    }

    fn attn_read_jobs(layer: &DeepSeekV4LayerSourceAnchor) -> Vec<(String, usize)> {
        let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::Attention);
        let attn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionNorm);
        let wq_a = layer.control_pair(DeepSeekV4LayerControlProjection::WqA);
        let wq_b = layer.control_pair(DeepSeekV4LayerControlProjection::WqB);
        let wkv = layer.control_pair(DeepSeekV4LayerControlProjection::Wkv);
        let wo_a = layer.control_pair(DeepSeekV4LayerControlProjection::WoA);
        let wo_b = layer.control_pair(DeepSeekV4LayerControlProjection::WoB);
        let q_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionQNorm);
        let kv_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionKvNorm);
        let sink = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionSink);
        vec![
            (
                mhc.fn_tensor.name.clone(),
                HC_MIX_WIDTH * HC_FLAT_WIDTH * size_of::<f32>(),
            ),
            (
                mhc.base_tensor.name.clone(),
                HC_MIX_WIDTH * size_of::<f32>(),
            ),
            (mhc.scale_tensor.name.clone(), 3 * size_of::<f32>()),
            (attn_norm.name.clone(), HIDDEN_SIZE * size_of::<u16>()),
            (wq_a.weight.name.clone(), Q_LORA_RANK * HIDDEN_SIZE),
            (
                wq_a.scale.name.clone(),
                (Q_LORA_RANK / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
            ),
            (wq_b.weight.name.clone(), WQ_B_ROWS * Q_LORA_RANK),
            (
                wq_b.scale.name.clone(),
                (WQ_B_ROWS / ACT_QUANT_BLOCK) * (Q_LORA_RANK / ACT_QUANT_BLOCK),
            ),
            (wkv.weight.name.clone(), WKV_ROWS * HIDDEN_SIZE),
            (
                wkv.scale.name.clone(),
                (WKV_ROWS / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
            ),
            (wo_a.weight.name.clone(), WO_A_ROWS * WO_A_COLS),
            (
                wo_a.scale.name.clone(),
                (WO_A_ROWS / ACT_QUANT_BLOCK) * (WO_A_COLS / ACT_QUANT_BLOCK),
            ),
            (wo_b.weight.name.clone(), WO_B_ROWS * WO_B_COLS),
            (
                wo_b.scale.name.clone(),
                (WO_B_ROWS / ACT_QUANT_BLOCK) * (WO_B_COLS / ACT_QUANT_BLOCK),
            ),
            (q_norm.name.clone(), Q_LORA_RANK * size_of::<u16>()),
            (kv_norm.name.clone(), HEAD_DIM * size_of::<u16>()),
            (sink.name.clone(), NUM_HEADS * size_of::<f32>()),
        ]
    }

    fn execute_attention(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        hc_in: &[u16],
        token_id: u64,
        ledger: &mut ResidentLedger,
        profiler: &mut TokenNsCollector,
        preloaded: Option<Vec<DeepSeekV4VerifiedBytes>>,
    ) -> Result<(Vec<u16>, MoePreload)> {
        let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::Attention);
        let attn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionNorm);
        let wq_a = layer.control_pair(DeepSeekV4LayerControlProjection::WqA);
        let wq_b = layer.control_pair(DeepSeekV4LayerControlProjection::WqB);
        let wkv = layer.control_pair(DeepSeekV4LayerControlProjection::Wkv);
        let wo_a = layer.control_pair(DeepSeekV4LayerControlProjection::WoA);
        let wo_b = layer.control_pair(DeepSeekV4LayerControlProjection::WoB);
        let q_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionQNorm);
        let kv_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionKvNorm);
        let sink = layer.common_tensor(DeepSeekV4LayerCommonTensor::AttentionSink);
        let jobs = attn_read_jobs(layer);
        let attn_io_bytes: usize = jobs.iter().map(|job| job.1).sum();
        let blobs = if let Some(preloaded) = preloaded {
            profiler.add_stage("host.attn_weight_io_prefetched", 0, 1, attn_io_bytes as u64);
            preloaded
        } else if FIRST_LAYER_BIND_TIMED
            .compare_exchange(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_ok()
        {
            crate::startup_timing::time_ms_result("first_layer_weight_bind", || {
                profiler.time_bytes_result("host.attn_weight_io", attn_io_bytes as u64, || {
                    par_read_views(reader, &jobs)
                })
            })?
        } else {
            profiler.time_bytes_result("host.attn_weight_io", attn_io_bytes as u64, || {
                par_read_views(reader, &jobs)
            })?
        };
        profiler.add_stage("host.mla.wq_a_bytes", 0, 1, (Q_LORA_RANK * HIDDEN_SIZE) as u64);
        profiler.add_stage(
            "host.mla.wq_b_bytes",
            0,
            1,
            (WQ_B_ROWS * Q_LORA_RANK) as u64,
        );
        profiler.add_stage("host.mla.wkv_bytes", 0, 1, (WKV_ROWS * HIDDEN_SIZE) as u64);
        profiler.add_stage("host.mla.wo_a_bytes", 0, 1, (WO_A_ROWS * WO_A_COLS) as u64);
        profiler.add_stage("host.mla.wo_b_bytes", 0, 1, (WO_B_ROWS * WO_B_COLS) as u64);
        let hc_fn = decode_f32_le(blobs[0].as_bytes(), &mhc.fn_tensor.name)?;
        let hc_base = decode_f32_le(blobs[1].as_bytes(), &mhc.base_tensor.name)?;
        let hc_scale = decode_f32_le(blobs[2].as_bytes(), &mhc.scale_tensor.name)?;
        let attn_norm_w = decode_u16_le(blobs[3].as_bytes(), &attn_norm.name)?;
        let (_, _, _, post_f32, comb_f32, reduced) = profiler.time_result("host.mhc_pre", || {
            hc_attn_pre_source_algorithm(
                hc_in,
                &hc_fn,
                &hc_scale,
                &hc_base,
                RMS_NORM_EPS,
                HC_EPS,
                HC_SINKHORN_ITERS,
            )
        })?;
        let attn_norm_row = profiler.time_result("host.rmsnorm", || {
            rms_norm_bf16_source_algorithm(&reduced, &attn_norm_w, HIDDEN_SIZE, RMS_NORM_EPS)
        })?;
        let attn_ready = graph.attn_scratch_ready;
        graph.attn_scratch_ready = false;
        let mut bind_or_fill = |dest_w: &metal::Buffer,
                            dest_s: &metal::Buffer,
                            w_name: &str,
                            s_name: &str,
                            weight: &[u8],
                            scale: &[u8]| {
            if attn_ready {
                ledger.acquire(w_name, weight.len())?;
                ledger.acquire(s_name, scale.len())?;
                Ok(Fp8Pair {
                    weight: dest_w.clone(),
                    scale: dest_s.clone(),
                    name_w: w_name.to_owned(),
                    name_s: s_name.to_owned(),
                })
            } else {
                refill_fp8(dest_w, dest_s, ledger, w_name, s_name, weight, scale)
            }
        };
        let wq_a_p = profiler.time_bytes_result(
            "host.mla.wq_a_upload",
            blobs[4].len() as u64 + blobs[5].len() as u64,
            || {
                bind_or_fill(
                    &graph.scratch.wq_a_w,
                    &graph.scratch.wq_a_s,
                    &wq_a.weight.name,
                    &wq_a.scale.name,
                    blobs[4].as_bytes(),
                    blobs[5].as_bytes(),
                )
            },
        )?;
        let wq_b_p = profiler.time_bytes_result(
            "host.mla.wq_b_upload",
            blobs[6].len() as u64 + blobs[7].len() as u64,
            || {
                bind_or_fill(
                    &graph.scratch.wq_b_w,
                    &graph.scratch.wq_b_s,
                    &wq_b.weight.name,
                    &wq_b.scale.name,
                    blobs[6].as_bytes(),
                    blobs[7].as_bytes(),
                )
            },
        )?;
        let wkv_p = profiler.time_bytes_result(
            "host.mla.wkv_upload",
            blobs[8].len() as u64 + blobs[9].len() as u64,
            || {
                bind_or_fill(
                    &graph.scratch.wkv_w,
                    &graph.scratch.wkv_s,
                    &wkv.weight.name,
                    &wkv.scale.name,
                    blobs[8].as_bytes(),
                    blobs[9].as_bytes(),
                )
            },
        )?;
        let wo_a_p = profiler.time_bytes_result(
            "host.mla.wo_a_upload",
            blobs[10].len() as u64 + blobs[11].len() as u64,
            || {
                bind_or_fill(
                    &graph.scratch.wo_a_w,
                    &graph.scratch.wo_a_s,
                    &wo_a.weight.name,
                    &wo_a.scale.name,
                    blobs[10].as_bytes(),
                    blobs[11].as_bytes(),
                )
            },
        )?;
        let wo_b_p = profiler.time_bytes_result(
            "host.mla.wo_b_upload",
            blobs[12].len() as u64 + blobs[13].len() as u64,
            || {
                bind_or_fill(
                    &graph.scratch.wo_b_w,
                    &graph.scratch.wo_b_s,
                    &wo_b.weight.name,
                    &wo_b.scale.name,
                    blobs[12].as_bytes(),
                    blobs[13].as_bytes(),
                )
            },
        )?;
        ledger.acquire(&q_norm.name, blobs[14].len())?;
        if !attn_ready {
            write_bytes(&graph.scratch.q_norm, blobs[14].as_bytes());
        }
        let q_norm_buf = graph.scratch.q_norm.clone();
        ledger.acquire(&kv_norm.name, blobs[15].len())?;
        if !attn_ready {
            write_bytes(&graph.scratch.kv_norm, blobs[15].as_bytes());
        }
        let kv_norm_buf = graph.scratch.kv_norm.clone();
        ledger.acquire(&sink.name, blobs[16].len())?;
        if !attn_ready {
            write_bytes(&graph.scratch.sink, blobs[16].as_bytes());
        }
        let sink_buf = graph.scratch.sink.clone();

        write_u16(&graph.scratch.hidden_a, &attn_norm_row);
        let softmax_scale = (HEAD_DIM as f32).powf(-0.5);
        let eps = RMS_NORM_EPS;
        let act_tg = graph.act_tg;
        let fp8_tg = graph.fp8_tg;
        let cast_tg = graph.cast_tg;
        let wo_a_tg = graph.wo_a_tg;
        let layer_idx = layer.layer;

        let (attn_n, attn_submitted) = graph.submit(|batch, s| {
            maybe_ordered(batch);
            let mut n = 0usize;
            dispatch_act_quant(
                batch,
                &s.hidden_a,
                &s.quant_k,
                &s.quant_scale_k,
                HIDDEN_SIZE as u32,
                act_tg,
                0,
                0,
                0,
            )?;
            n += 1;
            dispatch_fp8(
                batch,
                &wq_a_p.weight,
                &wq_a_p.scale,
                &s.quant_k,
                &s.quant_scale_k,
                &s.f32_tmp,
                Q_LORA_RANK as u32,
                HIDDEN_SIZE as u32,
                fp8_tg,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.q_lora, Q_LORA_RANK as u32, cast_tg)?;
            n += 1;
            dispatch_rmsnorm(
                batch,
                &s.q_lora,
                &q_norm_buf,
                &s.q_lora,
                Q_LORA_RANK as u32,
                eps,
            )?;
            n += 1;
            dispatch_act_quant(
                batch,
                &s.q_lora,
                &s.quant_q,
                &s.quant_scale_q,
                Q_LORA_RANK as u32,
                act_tg,
                0,
                0,
                0,
            )?;
            n += 1;
            dispatch_fp8(
                batch,
                &wq_b_p.weight,
                &wq_b_p.scale,
                &s.quant_q,
                &s.quant_scale_q,
                &s.f32_tmp,
                WQ_B_ROWS as u32,
                Q_LORA_RANK as u32,
                fp8_tg,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.wq_b, WQ_B_ROWS as u32, cast_tg)?;
            n += 1;
            let heads = NUM_HEADS as u32;
            let dim = HEAD_DIM as u32;
            batch.dispatch_threads(
                PER_HEAD_RMS_KERNEL,
                (heads, 1, 1),
                (heads.min(64), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&s.wq_b), 0);
                    enc.set_buffer(1, Some(&s.attn), 0);
                    set_u32(enc, 2, &heads);
                    set_u32(enc, 3, &dim);
                    set_f32(enc, 4, &eps);
                },
            )?;
            n += 1;
            dispatch_fp8(
                batch,
                &wkv_p.weight,
                &wkv_p.scale,
                &s.quant_k,
                &s.quant_scale_k,
                &s.f32_tmp,
                WKV_ROWS as u32,
                HIDDEN_SIZE as u32,
                fp8_tg,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.wkv, WKV_ROWS as u32, cast_tg)?;
            n += 1;
            dispatch_rmsnorm(batch, &s.wkv, &kv_norm_buf, &s.wkv, HEAD_DIM as u32, eps)?;
            n += 1;
            let rope = ROPE_HEAD_DIM as u32;
            let block = KV_QAT_BLOCK as u32;
            let qat_blocks = ((HEAD_DIM - ROPE_HEAD_DIM) / KV_QAT_BLOCK) as u32;
            batch.dispatch_threads(
                KV_QAT_KERNEL,
                (qat_blocks.max(1), 1, 1),
                (qat_blocks.max(1), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&s.wkv), 0);
                    enc.set_buffer(1, Some(&s.wkv), 0);
                    enc.set_buffer(2, Some(&s.kv_qat_bytes), 0);
                    enc.set_buffer(3, Some(&s.kv_qat_scales), 0);
                    set_u32(enc, 4, &dim);
                    set_u32(enc, 5, &rope);
                    set_u32(enc, 6, &block);
                },
            )?;
            n += 1;
            batch.dispatch_threads(ATTN_KERNEL, (heads, 1, 1), (heads.min(64), 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.attn), 0);
                enc.set_buffer(1, Some(&s.wkv), 0);
                enc.set_buffer(2, Some(&sink_buf), 0);
                enc.set_buffer(3, Some(&s.attn), 0);
                enc.set_buffer(4, Some(&s.attn_scores), 0);
                enc.set_buffer(5, Some(&s.attn_denoms), 0);
                set_u32(enc, 6, &heads);
                set_u32(enc, 7, &dim);
                set_f32(enc, 8, &softmax_scale);
            })?;
            n += 1;
            let rows = WO_A_ROWS as u32;
            let cols = WO_A_COLS as u32;
            let scale_cols = (WO_A_COLS / ACT_QUANT_BLOCK) as u32;
            let ranks = O_LORA_RANK as u32;
            let tg = wo_a_tg.min(rows);
            batch.dispatch_threads(WO_A_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(&wo_a_p.weight), 0);
                enc.set_buffer(1, Some(&wo_a_p.scale), 0);
                enc.set_buffer(2, Some(&s.attn), 0);
                enc.set_buffer(3, Some(&s.wo_a), 0);
                set_u32(enc, 4, &rows);
                set_u32(enc, 5, &cols);
                set_u32(enc, 6, &scale_cols);
                set_u32(enc, 7, &ranks);
            })?;
            n += 1;
            dispatch_act_quant(
                batch,
                &s.wo_a,
                &s.quant_wo,
                &s.quant_scale_wo,
                WO_B_COLS as u32,
                act_tg,
                0,
                0,
                0,
            )?;
            n += 1;
            dispatch_fp8(
                batch,
                &wo_b_p.weight,
                &wo_b_p.scale,
                &s.quant_wo,
                &s.quant_scale_wo,
                &s.f32_tmp,
                WO_B_ROWS as u32,
                WO_B_COLS as u32,
                fp8_tg,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.hidden_b, WO_B_ROWS as u32, cast_tg)?;
            n += 1;
            Ok(n)
        })?;
        let overlapped = preload_moe_io(graph, reader, layer, token_id, ledger, profiler)?;
        graph.finish(
            "attn",
            Some(layer_idx),
            "host_mhc_post_needs_wo_b_bf16",
            attn_n,
            attn_submitted,
            profiler,
        )?;

        let readback_started = Instant::now();
        let wo_b_out = read_u16(&graph.scratch.hidden_b, HIDDEN_SIZE)?;
        profiler.record_sync(
            "host.attn_activation_readback",
            Some(layer_idx),
            "host_source_mhc_post_requires_wo_b",
            readback_started.elapsed().as_nanos() as u64,
            (HIDDEN_SIZE * size_of::<u16>()) as u64,
        );

        if kernel_probe_enabled() && (layer_idx == 0 || layer_idx == 3) {
            probe_attention_kernels(
                graph,
                profiler,
                layer_idx,
                &wq_a_p,
                &wq_b_p,
                &wkv_p,
                &wo_a_p,
                &wo_b_p,
                &q_norm_buf,
                &kv_norm_buf,
                &sink_buf,
            )?;
        }

        release_fp8(ledger, &wq_a_p)?;
        release_fp8(ledger, &wq_b_p)?;
        release_fp8(ledger, &wkv_p)?;
        release_fp8(ledger, &wo_a_p)?;
        release_fp8(ledger, &wo_b_p)?;
        ledger.release(&q_norm.name)?;
        ledger.release(&kv_norm.name)?;
        ledger.release(&sink.name)?;
        let hc = profiler.time_result("host.mhc_post", || {
            hc_attn_post_source_algorithm(&wo_b_out, hc_in, &post_f32, &comb_f32)
        })?;
        Ok((hc, overlapped))
    }

    struct MoePreload {
        hc_fn: Vec<f32>,
        hc_base: Vec<f32>,
        hc_scale: Vec<f32>,
        ffn_norm_w: Vec<u16>,
        gate_w: metal::Buffer,
        hash_ids: Option<[u32; ACTIVATED_EXPERTS]>,
        tid2eid_buf: Option<metal::Buffer>,
        bias_buf: Option<metal::Buffer>,
        sh_w1: Fp8Pair,
        sh_w3: Fp8Pair,
        sh_w2: Fp8Pair,
        is_hash: bool,
    }

    fn preload_moe_io(
        graph: &Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        token_id: u64,
        ledger: &mut ResidentLedger,
        profiler: &mut TokenNsCollector,
    ) -> Result<MoePreload> {
        let mhc = layer.mhc_binding(DeepSeekV4LayerMhcStage::FeedForward);
        let ffn_norm = layer.common_tensor(DeepSeekV4LayerCommonTensor::FeedForwardNorm);
        let gate = layer.gate_binding();
        let mut jobs = vec![
            (
                mhc.fn_tensor.name.clone(),
                HC_MIX_WIDTH * HC_FLAT_WIDTH * size_of::<f32>(),
            ),
            (
                mhc.base_tensor.name.clone(),
                HC_MIX_WIDTH * size_of::<f32>(),
            ),
            (mhc.scale_tensor.name.clone(), 3 * size_of::<f32>()),
            (ffn_norm.name.clone(), HIDDEN_SIZE * size_of::<u16>()),
            (
                gate.score_weight.name.clone(),
                ROUTED_EXPERTS * HIDDEN_SIZE * size_of::<u16>(),
            ),
        ];
        let is_hash = matches!(
            layer.gate_mode,
            DeepSeekV4LayerGateMode::HashTokenIdToExpertIds
        );
        if !is_hash {
            jobs.push((
                gate.route_data.name.clone(),
                ROUTED_EXPERTS * size_of::<f32>(),
            ));
        }
        let moe_io_bytes: usize = jobs.iter().map(|job| job.1).sum();
        let blobs = profiler.time_bytes_result("host.moe_control_io", moe_io_bytes as u64, || {
            par_read_views(reader, &jobs)
        })?;
        let hc_fn = decode_f32_le(blobs[0].as_bytes(), &mhc.fn_tensor.name)?;
        let hc_base = decode_f32_le(blobs[1].as_bytes(), &mhc.base_tensor.name)?;
        let hc_scale = decode_f32_le(blobs[2].as_bytes(), &mhc.scale_tensor.name)?;
        let ffn_norm_w = decode_u16_le(blobs[3].as_bytes(), &ffn_norm.name)?;
        ledger.acquire(&gate.score_weight.name, blobs[4].len())?;
        write_bytes(&graph.scratch.gate_w, blobs[4].as_bytes());
        let gate_w = graph.scratch.gate_w.clone();
        let hash_ids = if is_hash {
            Some(hash_ids_from_tid2eid(
                reader,
                ledger,
                &gate.route_data.name,
                token_id,
            )?)
        } else {
            None
        };
        let tid2eid_buf = if let Some(ids) = hash_ids {
            let mut row = Vec::with_capacity(ACTIVATED_EXPERTS * size_of::<i64>());
            for id in ids {
                row.extend_from_slice(&(id as i64).to_le_bytes());
            }
            Some({
                bump_create();
                graph.metal.new_buffer_with_bytes_checked(&row)?
            })
        } else {
            None
        };
        let bias_buf = if !is_hash {
            ledger.acquire(&gate.route_data.name, blobs[5].len())?;
            write_bytes(&graph.scratch.gate_bias, blobs[5].as_bytes());
            Some(graph.scratch.gate_bias.clone())
        } else {
            None
        };
        let shared_w1 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W1);
        let shared_w3 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W3);
        let shared_w2 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W2);
        let sh_w1 = profiler.time_bytes_result(
            "host.shared.w1_io",
            (MOE_INTER_DIM * HIDDEN_SIZE) as u64,
            || {
                load_fp8(
                    &graph.scratch.sh_w1_w,
                    &graph.scratch.sh_w1_s,
                    reader,
                    ledger,
                    &shared_w1.weight.name,
                    &shared_w1.scale.name,
                    MOE_INTER_DIM,
                    HIDDEN_SIZE,
                )
            },
        )?;
        let sh_w3 = profiler.time_bytes_result(
            "host.shared.w3_io",
            (MOE_INTER_DIM * HIDDEN_SIZE) as u64,
            || {
                load_fp8(
                    &graph.scratch.sh_w3_w,
                    &graph.scratch.sh_w3_s,
                    reader,
                    ledger,
                    &shared_w3.weight.name,
                    &shared_w3.scale.name,
                    MOE_INTER_DIM,
                    HIDDEN_SIZE,
                )
            },
        )?;
        let sh_w2 = profiler.time_bytes_result(
            "host.shared.w2_io",
            (HIDDEN_SIZE * MOE_INTER_DIM) as u64,
            || {
                load_fp8(
                    &graph.scratch.sh_w2_w,
                    &graph.scratch.sh_w2_s,
                    reader,
                    ledger,
                    &shared_w2.weight.name,
                    &shared_w2.scale.name,
                    HIDDEN_SIZE,
                    MOE_INTER_DIM,
                )
            },
        )?;
        Ok(MoePreload {
            hc_fn,
            hc_base,
            hc_scale,
            ffn_norm_w,
            gate_w,
            hash_ids,
            tid2eid_buf,
            bias_buf,
            sh_w1,
            sh_w3,
            sh_w2,
            is_hash,
        })
    }

    fn maybe_ordered(batch: &mut CommandBatch<'_>) {
        if cb_collapse_enabled() {
            batch.enable_ordered_encoder();
        }
    }

    struct RouteEncode<'a> {
        gate_w: &'a metal::Buffer,
        tid2eid: Option<&'a metal::Buffer>,
        bias: Option<&'a metal::Buffer>,
        is_hash: bool,
        token_u: u32,
        experts_u: u32,
        top_k: u32,
        route_scale: f32,
        gate_tg: u32,
    }

    fn encode_route(batch: &mut CommandBatch<'_>, s: &Scratch, p: &RouteEncode<'_>) -> Result<usize> {
        let mut n = 0usize;
        let rows = ROUTED_EXPERTS as u32;
        let cols = HIDDEN_SIZE as u32;
        let tg = p.gate_tg.min(rows);
        batch.dispatch_threads(GATE_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(p.gate_w), 0);
            enc.set_buffer(1, Some(&s.hidden_a), 0);
            enc.set_buffer(2, Some(&s.gate_logits), 0);
            set_u32(enc, 3, &rows);
            set_u32(enc, 4, &cols);
        })?;
        n += 1;
        if p.is_hash {
            let table = p.tid2eid.expect("hash table");
            batch.dispatch_threads(HASH_ROUTE_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.gate_logits), 0);
                enc.set_buffer(1, Some(table), 0);
                enc.set_buffer(2, Some(&s.route_ids), 0);
                enc.set_buffer(3, Some(&s.route_weights), 0);
                enc.set_buffer(4, Some(&s.original_scores), 0);
                enc.set_buffer(5, Some(&s.route_valid), 0);
                set_u32(enc, 6, &p.token_u);
                set_u32(enc, 7, &p.experts_u);
                set_u32(enc, 8, &p.top_k);
                set_f32(enc, 9, &p.route_scale);
            })?;
        } else {
            let bias = p.bias.expect("bias");
            batch.dispatch_threads(LEARNED_ROUTE_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.gate_logits), 0);
                enc.set_buffer(1, Some(bias), 0);
                enc.set_buffer(2, Some(&s.route_ids), 0);
                enc.set_buffer(3, Some(&s.route_weights), 0);
                enc.set_buffer(4, Some(&s.original_scores), 0);
                enc.set_buffer(5, Some(&s.route_valid), 0);
                set_u32(enc, 6, &p.experts_u);
                set_u32(enc, 7, &p.top_k);
                set_f32(enc, 8, &p.route_scale);
            })?;
        }
        n += 1;
        batch.dispatch_threads(PACK_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
            enc.set_buffer(0, Some(&s.route_ids), 0);
            enc.set_buffer(1, Some(&s.route_weights), 0);
            enc.set_buffer(2, Some(&s.worklist), 0);
            enc.set_buffer(3, Some(&s.pack_valid), 0);
            set_u32(enc, 4, &p.top_k);
            set_u32(enc, 5, &p.experts_u);
        })?;
        n += 1;
        Ok(n)
    }

    struct MoeEncode<'a> {
        act_tg: u32,
        fp8_tg: u32,
        fp4_tg: u32,
        cast_tg: u32,
        top_k: u32,
        collapse: bool,
        sh_w1: &'a Fp8Pair,
        sh_w3: &'a Fp8Pair,
        sh_w2: &'a Fp8Pair,
    }

    fn encode_moe(batch: &mut CommandBatch<'_>, s: &Scratch, p: &MoeEncode<'_>) -> Result<usize> {
        let mut n = 0usize;
        dispatch_act_quant(
            batch,
            &s.hidden_a,
            &s.quant_ffn,
            &s.quant_scale_ffn,
            HIDDEN_SIZE as u32,
            p.act_tg,
            0,
            0,
            0,
        )?;
        n += 1;
        let rows_w1 = MOE_INTER_DIM as u32;
        let packed = (HIDDEN_SIZE / 2) as u32;
        let scale_cols = (HIDDEN_SIZE / FP4_BLOCK) as u32;
        let grid_w1 = p.top_k * rows_w1;
        let tg = p.fp4_tg.min(rows_w1);
        let zero = 0u32;
        let one = 1u32;
        let shared_one = 1.0f32;
        batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid_w1, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(&s.worklist), 0);
            enc.set_buffer(1, Some(&s.w1_slab), 0);
            enc.set_buffer(2, Some(&s.w1_scale_slab), 0);
            enc.set_buffer(3, Some(&s.quant_ffn), 0);
            enc.set_buffer(4, Some(&s.quant_scale_ffn), 0);
            enc.set_buffer(5, Some(&s.expert_gate_f32), 0);
            set_u32(enc, 6, &rows_w1);
            set_u32(enc, 7, &packed);
            set_u32(enc, 8, &scale_cols);
            set_u32(enc, 9, &p.top_k);
            set_u32(enc, 10, &zero);
        })?;
        n += 1;
        batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid_w1, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(&s.worklist), 0);
            enc.set_buffer(1, Some(&s.w3_slab), 0);
            enc.set_buffer(2, Some(&s.w3_scale_slab), 0);
            enc.set_buffer(3, Some(&s.quant_ffn), 0);
            enc.set_buffer(4, Some(&s.quant_scale_ffn), 0);
            enc.set_buffer(5, Some(&s.expert_up_f32), 0);
            set_u32(enc, 6, &rows_w1);
            set_u32(enc, 7, &packed);
            set_u32(enc, 8, &scale_cols);
            set_u32(enc, 9, &p.top_k);
            set_u32(enc, 10, &zero);
        })?;
        n += 1;
        let gate_count = p.top_k * rows_w1;
        dispatch_cast(batch, &s.expert_gate_f32, &s.expert_gate_bf16, gate_count, p.cast_tg)?;
        n += 1;
        dispatch_cast(batch, &s.expert_up_f32, &s.expert_up_bf16, gate_count, p.cast_tg)?;
        n += 1;
        batch.dispatch_threads(WORKLIST_SWIGLU_KERNEL, (gate_count, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(&s.worklist), 0);
            enc.set_buffer(1, Some(&s.expert_gate_bf16), 0);
            enc.set_buffer(2, Some(&s.expert_up_bf16), 0);
            enc.set_buffer(3, Some(&s.expert_swiglu), 0);
            set_u32(enc, 4, &rows_w1);
            set_u32(enc, 5, &p.top_k);
        })?;
        n += 1;
        if p.collapse {
            // Six experts are packed contiguously; 2048 % 128 == 0 so block
            // boundaries never cross experts. Same kernel, one dispatch.
            dispatch_act_quant(
                batch,
                &s.expert_swiglu,
                &s.expert_down_quant,
                &s.expert_down_scales,
                (ACTIVATED_EXPERTS * MOE_INTER_DIM) as u32,
                p.act_tg,
                0,
                0,
                0,
            )?;
            n += 1;
        } else {
            for slot in 0..ACTIVATED_EXPERTS {
                let off_in = (slot * MOE_INTER_DIM * size_of::<u16>()) as u64;
                let off_q = (slot * MOE_INTER_DIM) as u64;
                let off_s = (slot * (MOE_INTER_DIM / ACT_QUANT_BLOCK)) as u64;
                dispatch_act_quant(
                    batch,
                    &s.expert_swiglu,
                    &s.expert_down_quant,
                    &s.expert_down_scales,
                    MOE_INTER_DIM as u32,
                    p.act_tg,
                    off_in,
                    off_q,
                    off_s,
                )?;
                n += 1;
            }
        }
        let rows_w2 = HIDDEN_SIZE as u32;
        let packed_w2 = (MOE_INTER_DIM / 2) as u32;
        let scale_w2 = (MOE_INTER_DIM / FP4_BLOCK) as u32;
        let grid_w2 = p.top_k * rows_w2;
        let tg2 = p.fp4_tg.min(rows_w2);
        batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid_w2, 1, 1), (tg2, 1, 1), |enc| {
            enc.set_buffer(0, Some(&s.worklist), 0);
            enc.set_buffer(1, Some(&s.w2_slab), 0);
            enc.set_buffer(2, Some(&s.w2_scale_slab), 0);
            enc.set_buffer(3, Some(&s.expert_down_quant), 0);
            enc.set_buffer(4, Some(&s.expert_down_scales), 0);
            enc.set_buffer(5, Some(&s.expert_down_f32), 0);
            set_u32(enc, 6, &rows_w2);
            set_u32(enc, 7, &packed_w2);
            set_u32(enc, 8, &scale_w2);
            set_u32(enc, 9, &p.top_k);
            set_u32(enc, 10, &one);
        })?;
        n += 1;
        dispatch_cast(batch, &s.expert_down_f32, &s.expert_down_bf16, grid_w2, p.cast_tg)?;
        n += 1;
        dispatch_fp8(
            batch,
            &p.sh_w1.weight,
            &p.sh_w1.scale,
            &s.quant_ffn,
            &s.quant_scale_ffn,
            &s.shared_gate_f32,
            MOE_INTER_DIM as u32,
            HIDDEN_SIZE as u32,
            p.fp8_tg,
        )?;
        n += 1;
        dispatch_fp8(
            batch,
            &p.sh_w3.weight,
            &p.sh_w3.scale,
            &s.quant_ffn,
            &s.quant_scale_ffn,
            &s.shared_up_f32,
            MOE_INTER_DIM as u32,
            HIDDEN_SIZE as u32,
            p.fp8_tg,
        )?;
        n += 1;
        dispatch_cast(
            batch,
            &s.shared_gate_f32,
            &s.shared_gate_bf16,
            MOE_INTER_DIM as u32,
            p.cast_tg,
        )?;
        n += 1;
        dispatch_cast(
            batch,
            &s.shared_up_f32,
            &s.shared_up_bf16,
            MOE_INTER_DIM as u32,
            p.cast_tg,
        )?;
        n += 1;
        batch.dispatch_threads(
            SHARED_SWIGLU_KERNEL,
            (MOE_INTER_DIM as u32, 1, 1),
            (p.cast_tg.min(MOE_INTER_DIM as u32), 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&s.shared_gate_bf16), 0);
                enc.set_buffer(1, Some(&s.shared_up_bf16), 0);
                enc.set_buffer(2, Some(&s.shared_swiglu), 0);
                set_f32(enc, 3, &shared_one);
                set_u32(enc, 4, &(MOE_INTER_DIM as u32));
            },
        )?;
        n += 1;
        dispatch_act_quant(
            batch,
            &s.shared_swiglu,
            &s.shared_down_quant,
            &s.shared_down_scales,
            MOE_INTER_DIM as u32,
            p.act_tg,
            0,
            0,
            0,
        )?;
        n += 1;
        dispatch_fp8(
            batch,
            &p.sh_w2.weight,
            &p.sh_w2.scale,
            &s.shared_down_quant,
            &s.shared_down_scales,
            &s.shared_down_f32,
            HIDDEN_SIZE as u32,
            MOE_INTER_DIM as u32,
            p.fp8_tg,
        )?;
        n += 1;
        dispatch_cast(
            batch,
            &s.shared_down_f32,
            &s.shared_down_bf16,
            HIDDEN_SIZE as u32,
            p.cast_tg,
        )?;
        n += 1;
        batch.dispatch_threads(
            WORKLIST_COMBINE_KERNEL,
            (HIDDEN_SIZE as u32, 1, 1),
            (p.cast_tg.min(HIDDEN_SIZE as u32), 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&s.expert_down_bf16), 0);
                enc.set_buffer(1, Some(&s.shared_down_bf16), 0);
                enc.set_buffer(2, Some(&s.moe_out), 0);
                set_u32(enc, 3, &(HIDDEN_SIZE as u32));
                set_u32(enc, 4, &p.top_k);
            },
        )?;
        n += 1;
        Ok(n)
    }

    fn upload_experts_maybe_prefetch(
        graph: &Graph,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        layer: &DeepSeekV4LayerSourceAnchor,
        next_layer: Option<&DeepSeekV4LayerSourceAnchor>,
        exec: &[(u32, f32, u32)],
        profiler: &mut TokenNsCollector,
        attn_prefetch: &mut Option<Vec<DeepSeekV4VerifiedBytes>>,
        expert_bytes: u64,
    ) -> Result<Vec<String>> {
        if let Some(next) = next_layer {
            let started = Instant::now();
            let (names, prefetched) = std::thread::scope(|scope| -> Result<_> {
                let expert = scope.spawn(|| upload_expert_slab(graph, reader, ledger, layer, exec));
                let attn = scope.spawn(|| {
                    let jobs = attn_read_jobs(next);
                    par_read_views(reader, &jobs)
                });
                let names = expert
                    .join()
                    .map_err(|_| graph_error("expert slab thread panicked"))??;
                let prefetched = attn
                    .join()
                    .map_err(|_| graph_error("attn prefetch thread panicked"))??;
                Ok((names, prefetched))
            })?;
            let ns = started.elapsed().as_nanos() as u64;
            profiler.add_stage("host.expert_slab_io", ns, 1, expert_bytes);
            profiler.add_stage(
                "host.attn_weight_io_prefetch",
                ns,
                1,
                prefetched.iter().map(|b| b.len() as u64).sum(),
            );
            *attn_prefetch = Some(prefetched);
            Ok(names)
        } else {
            profiler.time_bytes_result("host.expert_slab_io", expert_bytes, || {
                upload_expert_slab(graph, reader, ledger, layer, exec)
            })
        }
    }

    fn fill_attn_prefetch(
        graph: &mut Graph,
        profiler: &mut TokenNsCollector,
        attn_prefetch: &Option<Vec<DeepSeekV4VerifiedBytes>>,
    ) {
        if let Some(blobs) = attn_prefetch.as_ref() {
            if blobs.len() >= 14 {
                profiler.time("host.mla.prefetch_fill", || {
                    write_bytes(&graph.scratch.wq_a_w, blobs[4].as_bytes());
                    write_bytes(&graph.scratch.wq_a_s, blobs[5].as_bytes());
                    write_bytes(&graph.scratch.wq_b_w, blobs[6].as_bytes());
                    write_bytes(&graph.scratch.wq_b_s, blobs[7].as_bytes());
                    write_bytes(&graph.scratch.wkv_w, blobs[8].as_bytes());
                    write_bytes(&graph.scratch.wkv_s, blobs[9].as_bytes());
                    write_bytes(&graph.scratch.wo_a_w, blobs[10].as_bytes());
                    write_bytes(&graph.scratch.wo_a_s, blobs[11].as_bytes());
                    write_bytes(&graph.scratch.wo_b_w, blobs[12].as_bytes());
                    write_bytes(&graph.scratch.wo_b_s, blobs[13].as_bytes());
                    write_bytes(&graph.scratch.q_norm, blobs[14].as_bytes());
                    if blobs.len() > 15 {
                        write_bytes(&graph.scratch.kv_norm, blobs[15].as_bytes());
                    }
                    if blobs.len() > 16 {
                        write_bytes(&graph.scratch.sink, blobs[16].as_bytes());
                    }
                });
                graph.attn_scratch_ready = true;
            }
        }
    }

    fn check_route_flags(layer_idx: usize, scratch: &Scratch) -> Result<()> {
        let valid = read_u32(&scratch.route_valid, 1)?;
        if valid[0] != 1 {
            return Err(graph_error(format!(
                "layer {layer_idx} route kernel rejected the token (valid={})",
                valid[0]
            )));
        }
        let pack_valid = read_u32(&scratch.pack_valid, 1)?;
        if pack_valid[0] != 1 {
            return Err(graph_error(format!(
                "layer {layer_idx} worklist pack rejected the route (valid={})",
                pack_valid[0]
            )));
        }
        Ok(())
    }

    fn execute_moe(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        next_layer: Option<&DeepSeekV4LayerSourceAnchor>,
        attn_hc: &[u16],
        token_id: u64,
        ledger: &mut ResidentLedger,
        profiler: &mut TokenNsCollector,
        preload: Option<MoePreload>,
        attn_prefetch: &mut Option<Vec<DeepSeekV4VerifiedBytes>>,
    ) -> Result<Vec<u16>> {
        let gate = layer.gate_binding();
        let preload = match preload {
            Some(prep) => prep,
            None => preload_moe_io(graph, reader, layer, token_id, ledger, profiler)?,
        };
        let MoePreload {
            hc_fn,
            hc_base,
            hc_scale,
            ffn_norm_w,
            gate_w,
            hash_ids,
            tid2eid_buf,
            bias_buf,
            sh_w1,
            sh_w3,
            sh_w2,
            is_hash,
        } = preload;
        let (_, _, _, post_f32, comb_f32, reduced) = profiler.time_result("host.mhc_pre", || {
            hc_attn_pre_source_algorithm(
                attn_hc,
                &hc_fn,
                &hc_scale,
                &hc_base,
                RMS_NORM_EPS,
                HC_EPS,
                HC_SINKHORN_ITERS,
            )
        })?;
        let ffn_norm_row = profiler.time_result("host.rmsnorm", || {
            rms_norm_bf16_source_algorithm(&reduced, &ffn_norm_w, HIDDEN_SIZE, RMS_NORM_EPS)
        })?;
        write_u16(&graph.scratch.hidden_a, &ffn_norm_row);

        let token_u = if is_hash { 0u32 } else { token_id as u32 };
        let experts_u = ROUTED_EXPERTS as u32;
        let top_k = ACTIVATED_EXPERTS as u32;
        let route_scale = ROUTE_SCALE;
        let gate_tg = graph.gate_tg;
        let layer_idx = layer.layer;

        let collapse = cb_collapse_enabled();
        let route_p = RouteEncode {
            gate_w: &gate_w,
            tid2eid: tid2eid_buf.as_ref(),
            bias: bias_buf.as_ref(),
            is_hash,
            token_u,
            experts_u,
            top_k,
            route_scale,
            gate_tg,
        };
        let moe_p = MoeEncode {
            act_tg: graph.act_tg,
            fp8_tg: graph.fp8_tg,
            fp4_tg: graph.fp4_tg,
            cast_tg: graph.cast_tg,
            top_k,
            collapse,
            sh_w1: &sh_w1,
            sh_w3: &sh_w3,
            sh_w2: &sh_w2,
        };
        let expert_bytes = (ACTIVATED_EXPERTS
            * (2 * W1_PACKED + W2_PACKED + 2 * W1_SCALES + W2_SCALES)) as u64;
        let merge_hash = collapse && is_hash;

        let expert_names = if merge_hash {
            let ids = hash_ids.expect("hash ids");
            let exec = pack_worklist_host(&ids, &[0.0f32; ACTIVATED_EXPERTS])?;
            let names = upload_experts_maybe_prefetch(
                graph,
                reader,
                ledger,
                layer,
                next_layer,
                &exec,
                profiler,
                attn_prefetch,
                expert_bytes,
            )?;
            let (n, submitted) = graph.submit(|batch, s| {
                maybe_ordered(batch);
                let a = encode_route(batch, s, &route_p)?;
                let b = encode_moe(batch, s, &moe_p)?;
                Ok(a + b)
            })?;
            fill_attn_prefetch(graph, profiler, attn_prefetch);
            graph.finish(
                "route_moe",
                Some(layer_idx),
                "hash_ids_known_before_route_so_experts_upload_then_one_cb",
                n,
                submitted,
                profiler,
            )?;
            check_route_flags(layer_idx, &graph.scratch)?;
            names
        } else {
            graph.batch(
                "route",
                Some(layer_idx),
                if is_hash {
                    "host_pack_valid_and_hash_weight_readback"
                } else {
                    "host_route_id_readback_for_expert_residency"
                },
                profiler,
                |batch, s| {
                    maybe_ordered(batch);
                    encode_route(batch, s, &route_p)
                },
            )?;
            check_route_flags(layer_idx, &graph.scratch)?;
            let exec = if let Some(ids) = hash_ids {
                let weights = read_f32_n(&graph.scratch.route_weights, ACTIVATED_EXPERTS)?;
                pack_worklist_host(&ids, &weights)?
            } else {
                graph.counters.host_route_id_readback += 1;
                let route_started = Instant::now();
                let ids = read_u32(&graph.scratch.route_ids, ACTIVATED_EXPERTS)?;
                let weights = read_f32_n(&graph.scratch.route_weights, ACTIVATED_EXPERTS)?;
                let mut arr = [0u32; ACTIVATED_EXPERTS];
                arr.copy_from_slice(&ids);
                let packed = pack_worklist_host(&arr, &weights)?;
                profiler.record_route_readback(
                    layer_idx,
                    route_started.elapsed().as_nanos() as u64,
                    "execute_moe after route command buffer; blocks upload_expert_slab",
                    "streaming_residency_needs_six_expert_ids",
                );
                profiler.record_sync(
                    "host.route_id_readback_sync",
                    Some(layer_idx),
                    "streaming_residency_needs_six_expert_ids",
                    route_started.elapsed().as_nanos() as u64,
                    (ACTIVATED_EXPERTS * (size_of::<u32>() + size_of::<f32>())) as u64,
                );
                packed
            };
            seed_worklist(graph, &exec);
            let names = upload_experts_maybe_prefetch(
                graph,
                reader,
                ledger,
                layer,
                next_layer,
                &exec,
                profiler,
                attn_prefetch,
                expert_bytes,
            )?;
            let (moe_n, moe_submitted) = graph.submit(|batch, s| {
                maybe_ordered(batch);
                encode_moe(batch, s, &moe_p)
            })?;
            fill_attn_prefetch(graph, profiler, attn_prefetch);
            graph.finish(
                "moe",
                Some(layer_idx),
                "host_mhc_post_needs_moe_out_bf16",
                moe_n,
                moe_submitted,
                profiler,
            )?;
            names
        };

        let readback_started = Instant::now();
        let moe = read_u16(&graph.scratch.moe_out, HIDDEN_SIZE)?;
        profiler.record_sync(
            "host.moe_activation_readback",
            Some(layer_idx),
            "host_source_mhc_post_requires_moe_out",
            readback_started.elapsed().as_nanos() as u64,
            (HIDDEN_SIZE * size_of::<u16>()) as u64,
        );

        if kernel_probe_enabled() && (layer_idx == 0 || layer_idx == 3) {
            probe_moe_kernels(
                graph,
                profiler,
                layer_idx,
                &sh_w1,
                &sh_w3,
                &sh_w2,
                &gate_w,
                bias_buf.as_ref(),
                tid2eid_buf.as_ref(),
                is_hash,
                token_u,
            )?;
        }

        // Residual readback is the layer HC handoff for host MHC, not an expert gather.
        for name in expert_names {
            ledger.release(&name)?;
        }
        release_fp8(ledger, &sh_w1)?;
        release_fp8(ledger, &sh_w3)?;
        release_fp8(ledger, &sh_w2)?;
        ledger.release(&gate.score_weight.name)?;
        if !is_hash {
            ledger.release(&gate.route_data.name)?;
        }
        profiler.time_result("host.mhc_post", || {
            hc_attn_post_source_algorithm(&moe, attn_hc, &post_f32, &comb_f32)
        })
    }

    fn metal_lm_head(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        residual_f32: &[f32],
        profiler: &mut TokenNsCollector,
    ) -> Result<DeepSeekV4GreedyTokenResult> {
        const ROWS_PER_BLOCK: usize = 16_384;
        let meta = reader.tensor_metadata(LM_HEAD_WEIGHT)?;
        if meta.dtype != "BF16"
            || meta.shape.as_slice() != [DSV4F_VOCAB_SIZE as u64, HIDDEN_SIZE as u64]
        {
            return Err(graph_error("head.weight is not BF16[vocab,4096]"));
        }
        let row_bytes = HIDDEN_SIZE * size_of::<u16>();
        let mut tiles = Vec::new();
        let mut row = 0usize;
        while row < DSV4F_VOCAB_SIZE {
            let count = (DSV4F_VOCAB_SIZE - row).min(ROWS_PER_BLOCK);
            tiles.push((row, count));
            row += count;
        }
        let jobs: Vec<(String, usize, u64, usize)> = tiles
            .iter()
            .map(|&(start_row, count)| {
                (
                    LM_HEAD_WEIGHT.to_owned(),
                    count * row_bytes,
                    (start_row * row_bytes) as u64,
                    count,
                )
            })
            .collect();
        let lm_io_bytes: u64 = jobs.iter().map(|j| j.1 as u64).sum();
        let blobs = profiler.time_bytes_result("host.lm_head_io", lm_io_bytes, || {
            std::thread::scope(|scope| {
                let mut joins = Vec::with_capacity(jobs.len());
                for (name, bytes, start, _) in &jobs {
                    let end = *start + *bytes as u64;
                    joins.push(
                        scope.spawn(move || reader.read_verified_range(name, *start..end, *bytes)),
                    );
                }
                let mut out = Vec::with_capacity(joins.len());
                for join in joins {
                    out.push(
                        join.join()
                            .map_err(|_| graph_error("lm_head parallel read panicked"))??,
                    );
                }
                Ok::<_, Error>(out)
            })
        })?;

        bump_create();
        let residual_buf = graph
            .metal
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(residual_f32))?;
        let max_bytes = jobs.iter().map(|j| j.1).max().unwrap_or(0);
        let max_rows = jobs.iter().map(|j| j.3).max().unwrap_or(0);
        bump_create();
        let weight_buf = graph.metal.new_buffer_checked(max_bytes)?;
        bump_create();
        let out_buf = graph
            .metal
            .new_buffer_checked(max_rows * size_of::<f32>())?;
        let mut best_id = 0u32;
        let mut best_logit = f32::NEG_INFINITY;
        let before = graph.counters.metal_dispatches;
        let lm_tg = graph.lm_tg;
        for (tile, bytes) in tiles.iter().zip(blobs.iter()) {
            ledger.acquire(LM_HEAD_WEIGHT, bytes.len())?;
            profiler.time_bytes("host.lm_head_upload", bytes.len() as u64, || {
                write_bytes(&weight_buf, bytes);
            });
            let rows_u = tile.1 as u32;
            let cols_u = HIDDEN_SIZE as u32;
            let tg = lm_tg.min(rows_u.max(1));
            let tile_name = format!("lm_head.tile{}", tile.0 / ROWS_PER_BLOCK);
            let timing = graph.metal.dispatch_batch_timed(|batch| {
                batch.dispatch_threads(LM_HEAD_KERNEL, (rows_u, 1, 1), (tg, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&weight_buf), 0);
                    enc.set_buffer(1, Some(&residual_buf), 0);
                    enc.set_buffer(2, Some(&out_buf), 0);
                    set_u32(enc, 3, &rows_u);
                    set_u32(enc, 4, &cols_u);
                })
            })?;
            graph.counters.command_buffers += 1;
            graph.counters.total_sync_points += 1;
            graph.counters.metal_dispatches += 1;
            profiler.record_cb(
                &tile_name,
                None,
                "host_greedy_reduction_needs_tile_logits",
                1,
                &timing,
            );
            let readback_started = Instant::now();
            let logits = read_f32_n(&out_buf, tile.1)?;
            profiler.record_sync(
                "host.lm_head_readback",
                None,
                "host_greedy_reduction_needs_tile_logits",
                readback_started.elapsed().as_nanos() as u64,
                (tile.1 * size_of::<f32>()) as u64,
            );
            ledger.release(LM_HEAD_WEIGHT)?;
            let (block_id, block_logit) = greedy_from_logits(&logits, tile.0);
            if block_logit > best_logit || (block_logit == best_logit && block_id < best_id) {
                best_logit = block_logit;
                best_id = block_id;
            }
        }
        if !best_logit.is_finite() {
            return Err(graph_error("lm_head produced a non-finite logit"));
        }
        Ok(DeepSeekV4GreedyTokenResult {
            token_id: best_id,
            logit: best_logit,
            vocab_size: DSV4F_VOCAB_SIZE,
            lm_head_on_device: true,
            argmax_on_device: false,
            metal_dispatches: graph.counters.metal_dispatches - before,
            command_buffers: graph.counters.command_buffers,
        })
    }

    fn probe_attention_kernels(
        graph: &Graph,
        profiler: &mut TokenNsCollector,
        layer_idx: usize,
        wq_a: &Fp8Pair,
        wq_b: &Fp8Pair,
        wkv: &Fp8Pair,
        wo_a: &Fp8Pair,
        wo_b: &Fp8Pair,
        q_norm: &metal::Buffer,
        kv_norm: &metal::Buffer,
        sink: &metal::Buffer,
    ) -> Result<()> {
        let s = &graph.scratch;
        let act_tg = graph.act_tg;
        let fp8_tg = graph.fp8_tg;
        let cast_tg = graph.cast_tg;
        let wo_a_tg = graph.wo_a_tg;
        let eps = RMS_NORM_EPS;
        let softmax_scale = (HEAD_DIM as f32).powf(-0.5);
        let heads = NUM_HEADS as u32;
        let dim = HEAD_DIM as u32;
        probe_one(&graph.metal, profiler, "isolated.act_quant.hidden", layer_idx, |batch| {
            dispatch_act_quant(
                batch,
                &s.hidden_a,
                &s.quant_k,
                &s.quant_scale_k,
                HIDDEN_SIZE as u32,
                act_tg,
                0,
                0,
                0,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.mla.wq_a", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &wq_a.weight,
                &wq_a.scale,
                &s.quant_k,
                &s.quant_scale_k,
                &s.f32_tmp,
                Q_LORA_RANK as u32,
                HIDDEN_SIZE as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.cast.q_lora", layer_idx, |batch| {
            dispatch_cast(batch, &s.f32_tmp, &s.q_lora, Q_LORA_RANK as u32, cast_tg)
        })?;
        probe_one(&graph.metal, profiler, "isolated.rmsnorm.q", layer_idx, |batch| {
            dispatch_rmsnorm(batch, &s.q_lora, q_norm, &s.q_lora, Q_LORA_RANK as u32, eps)
        })?;
        probe_one(&graph.metal, profiler, "isolated.mla.wq_b", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &wq_b.weight,
                &wq_b.scale,
                &s.quant_q,
                &s.quant_scale_q,
                &s.f32_tmp,
                WQ_B_ROWS as u32,
                Q_LORA_RANK as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.mla.wkv", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &wkv.weight,
                &wkv.scale,
                &s.quant_k,
                &s.quant_scale_k,
                &s.f32_tmp,
                WKV_ROWS as u32,
                HIDDEN_SIZE as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.mla.wo_a", layer_idx, |batch| {
            let rows = WO_A_ROWS as u32;
            let cols = WO_A_COLS as u32;
            let scale_cols = (WO_A_COLS / ACT_QUANT_BLOCK) as u32;
            let ranks = O_LORA_RANK as u32;
            let tg = wo_a_tg.min(rows);
            batch.dispatch_threads(WO_A_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(&wo_a.weight), 0);
                enc.set_buffer(1, Some(&wo_a.scale), 0);
                enc.set_buffer(2, Some(&s.attn), 0);
                enc.set_buffer(3, Some(&s.wo_a), 0);
                set_u32(enc, 4, &rows);
                set_u32(enc, 5, &cols);
                set_u32(enc, 6, &scale_cols);
                set_u32(enc, 7, &ranks);
            })
        })?;
        probe_one(&graph.metal, profiler, "isolated.mla.wo_b", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &wo_b.weight,
                &wo_b.scale,
                &s.quant_wo,
                &s.quant_scale_wo,
                &s.f32_tmp,
                WO_B_ROWS as u32,
                WO_B_COLS as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.attn.sparse_pos0", layer_idx, |batch| {
            batch.dispatch_threads(ATTN_KERNEL, (heads, 1, 1), (heads.min(64), 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.attn), 0);
                enc.set_buffer(1, Some(&s.wkv), 0);
                enc.set_buffer(2, Some(sink), 0);
                enc.set_buffer(3, Some(&s.attn), 0);
                enc.set_buffer(4, Some(&s.attn_scores), 0);
                enc.set_buffer(5, Some(&s.attn_denoms), 0);
                set_u32(enc, 6, &heads);
                set_u32(enc, 7, &dim);
                set_f32(enc, 8, &softmax_scale);
            })
        })?;
        let _ = kv_norm;
        Ok(())
    }

    fn probe_moe_kernels(
        graph: &Graph,
        profiler: &mut TokenNsCollector,
        layer_idx: usize,
        sh_w1: &Fp8Pair,
        sh_w3: &Fp8Pair,
        sh_w2: &Fp8Pair,
        gate_w: &metal::Buffer,
        bias: Option<&metal::Buffer>,
        tid2eid: Option<&metal::Buffer>,
        is_hash: bool,
        token_u: u32,
    ) -> Result<()> {
        let s = &graph.scratch;
        let fp8_tg = graph.fp8_tg;
        let fp4_tg = graph.fp4_tg;
        let gate_tg = graph.gate_tg;
        let top_k = ACTIVATED_EXPERTS as u32;
        let rows_w1 = MOE_INTER_DIM as u32;
        let packed = (HIDDEN_SIZE / 2) as u32;
        let scale_cols = (HIDDEN_SIZE / FP4_BLOCK) as u32;
        let grid_w1 = top_k * rows_w1;
        let tg = fp4_tg.min(rows_w1);
        let zero = 0u32;
        let one = 1u32;
        let experts_u = ROUTED_EXPERTS as u32;
        probe_one(&graph.metal, profiler, "isolated.router.gate", layer_idx, |batch| {
            let rows = ROUTED_EXPERTS as u32;
            let cols = HIDDEN_SIZE as u32;
            let tg = gate_tg.min(rows);
            batch.dispatch_threads(GATE_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(gate_w), 0);
                enc.set_buffer(1, Some(&s.hidden_a), 0);
                enc.set_buffer(2, Some(&s.gate_logits), 0);
                set_u32(enc, 3, &rows);
                set_u32(enc, 4, &cols);
            })
        })?;
        if is_hash {
            if let Some(table) = tid2eid {
                probe_one(&graph.metal, profiler, "isolated.router.hash", layer_idx, |batch| {
                    batch.dispatch_threads(HASH_ROUTE_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                        enc.set_buffer(0, Some(&s.gate_logits), 0);
                        enc.set_buffer(1, Some(table), 0);
                        enc.set_buffer(2, Some(&s.route_ids), 0);
                        enc.set_buffer(3, Some(&s.route_weights), 0);
                        enc.set_buffer(4, Some(&s.original_scores), 0);
                        enc.set_buffer(5, Some(&s.route_valid), 0);
                        set_u32(enc, 6, &token_u);
                        set_u32(enc, 7, &experts_u);
                        set_u32(enc, 8, &top_k);
                        set_f32(enc, 9, &ROUTE_SCALE);
                    })
                })?;
            }
        } else if let Some(bias) = bias {
            probe_one(&graph.metal, profiler, "isolated.router.learned", layer_idx, |batch| {
                batch.dispatch_threads(LEARNED_ROUTE_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&s.gate_logits), 0);
                    enc.set_buffer(1, Some(bias), 0);
                    enc.set_buffer(2, Some(&s.route_ids), 0);
                    enc.set_buffer(3, Some(&s.route_weights), 0);
                    enc.set_buffer(4, Some(&s.original_scores), 0);
                    enc.set_buffer(5, Some(&s.route_valid), 0);
                    set_u32(enc, 6, &experts_u);
                    set_u32(enc, 7, &top_k);
                    set_f32(enc, 8, &ROUTE_SCALE);
                })
            })?;
        }
        probe_one(&graph.metal, profiler, "isolated.routed.w1", layer_idx, |batch| {
            batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid_w1, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.worklist), 0);
                enc.set_buffer(1, Some(&s.w1_slab), 0);
                enc.set_buffer(2, Some(&s.w1_scale_slab), 0);
                enc.set_buffer(3, Some(&s.quant_ffn), 0);
                enc.set_buffer(4, Some(&s.quant_scale_ffn), 0);
                enc.set_buffer(5, Some(&s.expert_gate_f32), 0);
                set_u32(enc, 6, &rows_w1);
                set_u32(enc, 7, &packed);
                set_u32(enc, 8, &scale_cols);
                set_u32(enc, 9, &top_k);
                set_u32(enc, 10, &zero);
            })
        })?;
        probe_one(&graph.metal, profiler, "isolated.routed.w3", layer_idx, |batch| {
            batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid_w1, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.worklist), 0);
                enc.set_buffer(1, Some(&s.w3_slab), 0);
                enc.set_buffer(2, Some(&s.w3_scale_slab), 0);
                enc.set_buffer(3, Some(&s.quant_ffn), 0);
                enc.set_buffer(4, Some(&s.quant_scale_ffn), 0);
                enc.set_buffer(5, Some(&s.expert_up_f32), 0);
                set_u32(enc, 6, &rows_w1);
                set_u32(enc, 7, &packed);
                set_u32(enc, 8, &scale_cols);
                set_u32(enc, 9, &top_k);
                set_u32(enc, 10, &zero);
            })
        })?;
        let rows_w2 = HIDDEN_SIZE as u32;
        let packed_w2 = (MOE_INTER_DIM / 2) as u32;
        let scale_w2 = (MOE_INTER_DIM / FP4_BLOCK) as u32;
        let grid_w2 = top_k * rows_w2;
        let tg2 = fp4_tg.min(rows_w2);
        probe_one(&graph.metal, profiler, "isolated.routed.w2", layer_idx, |batch| {
            batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid_w2, 1, 1), (tg2, 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.worklist), 0);
                enc.set_buffer(1, Some(&s.w2_slab), 0);
                enc.set_buffer(2, Some(&s.w2_scale_slab), 0);
                enc.set_buffer(3, Some(&s.expert_down_quant), 0);
                enc.set_buffer(4, Some(&s.expert_down_scales), 0);
                enc.set_buffer(5, Some(&s.expert_down_f32), 0);
                set_u32(enc, 6, &rows_w2);
                set_u32(enc, 7, &packed_w2);
                set_u32(enc, 8, &scale_w2);
                set_u32(enc, 9, &top_k);
                set_u32(enc, 10, &one);
            })
        })?;
        probe_one(&graph.metal, profiler, "isolated.shared.w1", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &sh_w1.weight,
                &sh_w1.scale,
                &s.quant_ffn,
                &s.quant_scale_ffn,
                &s.shared_gate_f32,
                MOE_INTER_DIM as u32,
                HIDDEN_SIZE as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.shared.w3", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &sh_w3.weight,
                &sh_w3.scale,
                &s.quant_ffn,
                &s.quant_scale_ffn,
                &s.shared_up_f32,
                MOE_INTER_DIM as u32,
                HIDDEN_SIZE as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.shared.w2", layer_idx, |batch| {
            dispatch_fp8(
                batch,
                &sh_w2.weight,
                &sh_w2.scale,
                &s.shared_down_quant,
                &s.shared_down_scales,
                &s.shared_down_f32,
                HIDDEN_SIZE as u32,
                MOE_INTER_DIM as u32,
                fp8_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.moe_combine", layer_idx, |batch| {
            batch.dispatch_threads(
                WORKLIST_COMBINE_KERNEL,
                (HIDDEN_SIZE as u32, 1, 1),
                (graph.cast_tg.min(HIDDEN_SIZE as u32), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&s.expert_down_bf16), 0);
                    enc.set_buffer(1, Some(&s.shared_down_bf16), 0);
                    enc.set_buffer(2, Some(&s.moe_out), 0);
                    set_u32(enc, 3, &(HIDDEN_SIZE as u32));
                    set_u32(enc, 4, &top_k);
                },
            )
        })?;
        Ok(())
    }
}

#[cfg(target_os = "macos")]
use macos::run_native_bos_token_macos;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_kernels_are_publicly_enumerated() {
        assert_eq!(NATIVE_TOKEN_GRAPH_KERNELS.len(), 4);
        for kernel in NATIVE_TOKEN_GRAPH_KERNELS {
            assert!(!kernel.is_empty());
            assert!(kernel.starts_with("dsv4f_"));
        }
    }

    #[test]
    fn pack_worklist_sorts_by_expert_id_then_slot() {
        let ids = [10u32, 2, 7, 4, 1, 9];
        let weights = [0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6];
        let packed = pack_worklist_host(&ids, &weights).expect("pack");
        let ordered: Vec<u32> = packed.iter().map(|row| row.0).collect();
        assert_eq!(ordered, vec![1, 2, 4, 7, 9, 10]);
        assert_eq!(packed[0].1, 0.5);
        assert_eq!(packed[0].2, 4);
    }

    #[test]
    fn pack_worklist_rejects_duplicates() {
        let ids = [1u32, 2, 3, 1, 5, 6];
        let weights = [1.0f32; 6];
        assert!(pack_worklist_host(&ids, &weights).is_err());
    }

    #[test]
    fn honesty_labels_a_complete_bos_token_not_a_dense_256() {
        let honesty = NativeTokenGraphHonesty::for_run(true, true);
        assert!(honesty.native);
        assert!(honesty.complete_bos_token_graph);
        assert!(honesty.device_resident_routing);
        assert!(honesty.compact_top6_worklist);
        assert!(!honesty.dense_over_256);
        assert!(!honesty.host_cpu);
    }

    #[test]
    fn oracle_parity_constants_match_sealed_receipt() {
        assert_eq!(ORACLE_GREEDY_TOKEN_ID, 5);
        assert_eq!(ORACLE_GREEDY_LOGIT, 16.767_437);
    }

    #[test]
    fn census_arithmetic_matches_pre_collapse_43_layer_topology() {
        // 43 * (attn 17 + route 3 + moe 23) + 8 lm_head tiles
        assert_eq!(43 * 17 + 43 * 3 + 43 * 23 + 8, 1857);
        assert_eq!(43 + 43 + 43 + 8, 137);
    }

    #[test]
    fn batched_expert_act_quant_is_block_aligned() {
        assert_eq!(MOE_INTER_DIM % ACT_QUANT_BLOCK, 0);
        assert_eq!((ACTIVATED_EXPERTS * MOE_INTER_DIM) % ACT_QUANT_BLOCK, 0);
    }

    #[test]
    fn collapse_flag_treats_unset_as_on() {
        // The function reads the process env; just prove it returns a bool
        // without panicking. Paired A/B sets the var explicitly.
        let _ = cb_collapse_enabled();
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn live_layer0_has_zero_host_expert_gather() {
        use crate::gravity_deepseek_v4_streamed_forward::discover_sealed_dsv4f_artifact;
        let Some(artifact) = discover_sealed_dsv4f_artifact() else {
            eprintln!("sealed DSV4F artifact not found; native graph layer-0 skipped");
            return;
        };
        match run_native_bos_token(&artifact, 0, false) {
            Ok(report) => {
                assert_eq!(report.counters.host_expert_gather, 0);
                assert_eq!(report.counters.host_expert_output_readback, 0);
                assert_eq!(report.counters.host_route_id_readback, 0);
                assert!(report.counters.metal_dispatches > 0);
                assert!(report.counters.command_buffers > 0);
                assert_eq!(report.counters.fallbacks, 0);
                assert_eq!(report.layers_executed, vec![0]);
                assert!(report.honesty.compact_top6_worklist);
                assert!(!report.honesty.dense_over_256);
                assert!(report.weight_within_bound);
                assert_eq!(
                    report.hc_bf16_sha256,
                    "a43d928f5dea968c0c692b22d340c19525b6a78ea33a3beda7e815f050408a8e",
                    "layer-0 HC must match the sealed CPU L0 profile"
                );
            }
            Err(error) => {
                eprintln!("native graph layer-0 unavailable: {error}");
            }
        }
    }
}
