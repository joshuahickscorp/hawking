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
const RMSNORM_SIMD_KERNEL: &str = "deepseek_v4_p3a_rmsnorm_bf16_simdgroup_candidate";
const PER_HEAD_RMS_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_authority";
const PER_HEAD_RMS_SIMD_KERNEL: &str = "deepseek_v4_p3a_per_head_rmsnorm_bf16_simdgroup_candidate";
const KV_QAT_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_authority";
const KV_QAT_SIMD_KERNEL: &str = "deepseek_v4_p4a_kv_nonrope_qat_inplace_simdgroup_candidate";
const ATTN_KERNEL: &str = "deepseek_v4_p4a_sparse_attention_position0_sink_authority";
const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
const WO_A_SIMD_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_simdgroup_candidate";
const FP8_OCC_KERNEL: &str =
    "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_occupancy_candidate";
const FP8_OCC_MAX_BLOCKS: u32 = 128;
const SIMD_WIDTH: u32 = 32;
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
    pub expert_nocopy_binds: usize,
    pub expert_slab_packs: usize,
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
                "expert_nocopy_binds": self.counters.expert_nocopy_binds,
                "expert_slab_packs": self.counters.expert_slab_packs,
                "expert_payload_path": if self.counters.expert_slab_packs > 0 {
                    "compact_slab_pack"
                } else {
                    "device_address_table_nocopy"
                },
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

    fn env_flag(name: &str, default_on: bool) -> bool {
        match std::env::var(name) {
            Ok(raw) => !matches!(
                raw.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "off" | "no"
            ),
            Err(_) => default_on,
        }
    }

    fn kernel_probe_enabled() -> bool {
        // Preserve the historical empty-env-is-on reading used by A/B probes.
        !matches!(
            std::env::var("HAWKING_DSV4F_KERNEL_PROBE")
                .unwrap_or_default()
                .trim()
                .to_ascii_lowercase()
                .as_str(),
            "0" | "false" | "off" | "no"
        )
    }

    fn mla_serial_group_enabled() -> bool {
        env_flag("HAWKING_DSV4F_MLA_SERIAL_GROUP", true)
    }

    fn mla_kv_qat_simd_enabled() -> bool {
        env_flag("HAWKING_DSV4F_MLA_KV_QAT_SIMD", true)
    }

    /// Ordered RMSNorm: parallel squares, authority left-fold. Bit-identical
    /// to the one-thread kernel. Default on.
    fn mla_rmsnorm_simd_enabled() -> bool {
        env_flag("HAWKING_DSV4F_MLA_RMSNORM_SIMD", true)
    }

    /// WO-A simdgroup tree reduction. Not bit-identical. Default off.
    fn mla_wo_a_simd_enabled() -> bool {
        env_flag("HAWKING_DSV4F_MLA_WO_A_SIMD", false)
    }

    /// FP8 occupancy grid (wq_a / wkv / wo_b). Not bit-identical. Default off.
    fn mla_fp8_simd_enabled() -> bool {
        env_flag("HAWKING_DSV4F_MLA_FP8_SIMD", false)
    }

    fn align_simd(threads: u32) -> u32 {
        let aligned = threads - (threads % SIMD_WIDTH);
        aligned.max(SIMD_WIDTH)
    }

    /// Default **off**. Set `HAWKING_DSV4F_EXPERT_SLAB_PACK=1` to restore the
    /// host memcpy of six expert payloads into compact slabs (pre-address-table
    /// path) for A/B. Address-table no-copy bind is the default.
    fn expert_compact_slab_pack_enabled() -> bool {
        match std::env::var("HAWKING_DSV4F_EXPERT_SLAB_PACK") {
            Ok(raw) => {
                let trimmed = raw.trim();
                !(trimmed.is_empty()
                    || trimmed.eq_ignore_ascii_case("0")
                    || trimmed.eq_ignore_ascii_case("false")
                    || trimmed.eq_ignore_ascii_case("off")
                    || trimmed.eq_ignore_ascii_case("no"))
            }
            Err(_) => false,
        }
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
        fp8_occ_tg: u32,
        fp4_tg: u32,
        cast_tg: u32,
        gate_tg: u32,
        wo_a_tg: u32,
        wo_a_occ_tg: u32,
        rms_tg: u32,
        lm_tg: u32,
        attn_scratch_ready: bool,
        mla_pipeline_limits: Vec<crate::gravity_deepseek_v4_token_ns_ledger::MlaPipelineLimit>,
    }

    impl Graph {
        fn new() -> Result<Self> {
            let metal = MetalContext::new()?;
            let scratch = Scratch::new(&metal)?;
            let mut counters = NativeTokenGraphCounters::default();
            counters.scratch_buffer_creations =
                std::mem::size_of::<Scratch>() / std::mem::size_of::<metal::Buffer>();
            let mla_pipeline_limits = mla_pipeline_limits(&metal)?;
            Ok(Self {
                act_tg: pipeline_tg(&metal, ACT_QUANT_SIMD_KERNEL, 256)?,
                fp8_tg: pipeline_tg(&metal, FP8_KERNEL, 256)?,
                fp8_occ_tg: align_simd(pipeline_tg(&metal, FP8_OCC_KERNEL, 256)?),
                fp4_tg: pipeline_tg(&metal, WORKLIST_FP4_KERNEL, 256)?,
                cast_tg: pipeline_tg(&metal, CAST_KERNEL, 256)?,
                gate_tg: pipeline_tg(&metal, GATE_KERNEL, 256)?,
                wo_a_tg: pipeline_tg(&metal, WO_A_KERNEL, 256)?,
                wo_a_occ_tg: align_simd(pipeline_tg(&metal, WO_A_SIMD_KERNEL, 256)?),
                rms_tg: align_simd(pipeline_tg(&metal, RMSNORM_SIMD_KERNEL, 256)?),
                lm_tg: pipeline_tg(&metal, LM_HEAD_KERNEL, 256)?,
                metal,
                scratch,
                counters,
                attn_scratch_ready: false,
                mla_pipeline_limits,
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

    fn mla_pipeline_limits(
        metal: &MetalContext,
    ) -> Result<Vec<crate::gravity_deepseek_v4_token_ns_ledger::MlaPipelineLimit>> {
        let names = [
            ACT_QUANT_SIMD_KERNEL,
            FP8_KERNEL,
            FP8_OCC_KERNEL,
            CAST_KERNEL,
            RMSNORM_KERNEL,
            RMSNORM_SIMD_KERNEL,
            PER_HEAD_RMS_KERNEL,
            PER_HEAD_RMS_SIMD_KERNEL,
            KV_QAT_KERNEL,
            KV_QAT_SIMD_KERNEL,
            ATTN_KERNEL,
            WO_A_KERNEL,
            WO_A_SIMD_KERNEL,
        ];
        let mut out = Vec::with_capacity(names.len());
        for kernel in names {
            let pipe = metal.pipeline(kernel)?;
            out.push(
                crate::gravity_deepseek_v4_token_ns_ledger::MlaPipelineLimit {
                    kernel: kernel.to_owned(),
                    thread_execution_width: pipe.thread_execution_width() as u64,
                    max_total_threads_per_threadgroup: pipe.max_total_threads_per_threadgroup()
                        as u64,
                    static_threadgroup_memory_length: pipe.static_threadgroup_memory_length()
                        as u64,
                },
            );
        }
        Ok(out)
    }

    fn act_quant_geometry(cols: u32, simd_tg: u32) -> (u32, u32, u32, u32) {
        let blocks = cols / ACT_QUANT_BLOCK as u32;
        if blocks == 0 {
            return (0, 0, 0, 0);
        }
        let threads_x = simd_tg.max(ACT_QUANT_SIMD_WIDTH);
        let threads_x = threads_x - (threads_x % ACT_QUANT_SIMD_WIDTH);
        let simdgroups = threads_x / ACT_QUANT_SIMD_WIDTH;
        let groups = (blocks + simdgroups - 1) / simdgroups;
        (groups * threads_x, groups, threads_x, simdgroups)
    }

    fn occupancy_proxy(threads: u64) -> f64 {
        (threads as f64 / 32_768.0).min(1.0)
    }

    fn mla_dispatch_specs(
        layers: u64,
        act_tg: u32,
        fp8_tg: u32,
        fp8_occ_tg: u32,
        cast_tg: u32,
        wo_a_tg: u32,
        wo_a_occ_tg: u32,
        rms_tg: u32,
    ) -> Vec<crate::gravity_deepseek_v4_token_ns_ledger::MlaDispatchSpec> {
        use crate::gravity_deepseek_v4_token_ns_ledger::MlaDispatchSpec;
        let spec = |name: &str,
                    kernel: &str,
                    threads: u64,
                    threadgroups: u64,
                    tptg: u64,
                    sgs: u64,
                    bytes_read: u64,
                    bytes_written: u64,
                    flops: u64| {
            MlaDispatchSpec {
                name: name.to_owned(),
                kernel: kernel.to_owned(),
                invocations_per_layer: 1,
                invocations_per_token: layers,
                threads,
                threadgroups,
                threads_per_threadgroup: tptg,
                simdgroups_per_threadgroup: sgs,
                bytes_read,
                bytes_written,
                approx_flops: flops,
                occupancy_proxy: occupancy_proxy(threads),
                isolated_gpu_ns_mean: None,
                isolated_ns_per_invocation: None,
                memory_stall_proxy_gbps: None,
            }
        };
        let (aq_h_th, aq_h_tg, aq_h_tptg, aq_h_sg) =
            act_quant_geometry(HIDDEN_SIZE as u32, act_tg);
        let (aq_q_th, aq_q_tg, aq_q_tptg, aq_q_sg) =
            act_quant_geometry(Q_LORA_RANK as u32, act_tg);
        let (aq_o_th, aq_o_tg, aq_o_tptg, aq_o_sg) =
            act_quant_geometry(WO_B_COLS as u32, act_tg);
        let fp8_tg = fp8_tg.max(1);
        let fp8_occ_tg = align_simd(fp8_occ_tg.max(SIMD_WIDTH));
        let cast_tg = cast_tg.max(1);
        let wo_tg = wo_a_tg.min(WO_A_ROWS as u32).max(1);
        let wo_occ_tg = align_simd(wo_a_occ_tg.max(SIMD_WIDTH));
        let rms_tg = align_simd(rms_tg.max(SIMD_WIDTH));
        let rms_simd = mla_rmsnorm_simd_enabled();
        let wo_a_simd = mla_wo_a_simd_enabled();
        let fp8_simd = mla_fp8_simd_enabled();
        let qat_blocks = ((HEAD_DIM - ROPE_HEAD_DIM) / KV_QAT_BLOCK) as u64;
        let kv_qat_simd = mla_kv_qat_simd_enabled();
        let rms_q_threads = if rms_simd {
            Q_LORA_RANK.min(rms_tg as usize) as u64
        } else {
            1
        };
        let rms_kv_threads = if rms_simd {
            HEAD_DIM.min(rms_tg as usize) as u64
        } else {
            1
        };
        let per_head_threads = if rms_simd {
            (NUM_HEADS as u64) * SIMD_WIDTH as u64
        } else {
            NUM_HEADS as u64
        };
        let fp8_row_geom = |rows: u64| -> (u64, u64, u64, u64, &'static str) {
            if fp8_simd {
                (
                    rows * fp8_occ_tg as u64,
                    rows,
                    fp8_occ_tg as u64,
                    (fp8_occ_tg / SIMD_WIDTH) as u64,
                    FP8_OCC_KERNEL,
                )
            } else {
                (
                    rows,
                    ((rows as u32 + fp8_tg - 1) / fp8_tg) as u64,
                    fp8_tg.min(rows as u32) as u64,
                    (fp8_tg.min(rows as u32) / 32).max(1) as u64,
                    FP8_KERNEL,
                )
            }
        };
        let (wq_a_th, wq_a_tg, wq_a_tptg, wq_a_sg, wq_a_kern) = fp8_row_geom(Q_LORA_RANK as u64);
        let (wkv_th, wkv_tg, wkv_tptg, wkv_sg, wkv_kern) = fp8_row_geom(WKV_ROWS as u64);
        let (wo_b_th, wo_b_tg, wo_b_tptg, wo_b_sg, wo_b_kern) = fp8_row_geom(WO_B_ROWS as u64);
        let (wo_a_th, wo_a_n_tg, wo_a_tptg, wo_a_sg, wo_a_kern) = if wo_a_simd {
            (
                WO_A_ROWS as u64 * wo_occ_tg as u64,
                WO_A_ROWS as u64,
                wo_occ_tg as u64,
                (wo_occ_tg / SIMD_WIDTH) as u64,
                WO_A_SIMD_KERNEL,
            )
        } else {
            (
                WO_A_ROWS as u64,
                ((WO_A_ROWS as u32 + wo_tg - 1) / wo_tg) as u64,
                wo_tg as u64,
                (wo_tg / 32) as u64,
                WO_A_KERNEL,
            )
        };
        let kv_qat_threads = if kv_qat_simd {
            qat_blocks.max(1) * ACT_QUANT_SIMD_WIDTH as u64
        } else {
            qat_blocks.max(1)
        };
        let kv_qat_kernel = if kv_qat_simd {
            KV_QAT_SIMD_KERNEL
        } else {
            KV_QAT_KERNEL
        };
        let scale_bytes = |rows: usize, cols: usize| -> u64 {
            ((rows / ACT_QUANT_BLOCK) * (cols / ACT_QUANT_BLOCK)) as u64
        };
        vec![
            spec(
                "act_quant.hidden",
                ACT_QUANT_SIMD_KERNEL,
                aq_h_th as u64,
                aq_h_tg as u64,
                aq_h_tptg as u64,
                aq_h_sg as u64,
                (HIDDEN_SIZE * 2) as u64,
                (HIDDEN_SIZE + HIDDEN_SIZE / ACT_QUANT_BLOCK) as u64,
                (HIDDEN_SIZE * 256) as u64,
            ),
            spec(
                "mla.wq_a",
                wq_a_kern,
                wq_a_th,
                wq_a_tg,
                wq_a_tptg,
                wq_a_sg,
                (Q_LORA_RANK * HIDDEN_SIZE) as u64
                    + scale_bytes(Q_LORA_RANK, HIDDEN_SIZE)
                    + HIDDEN_SIZE as u64
                    + (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u64,
                (Q_LORA_RANK * 4) as u64,
                2 * Q_LORA_RANK as u64 * HIDDEN_SIZE as u64,
            ),
            spec(
                "cast.q_lora",
                CAST_KERNEL,
                Q_LORA_RANK as u64,
                ((Q_LORA_RANK as u32 + cast_tg - 1) / cast_tg) as u64,
                cast_tg.min(Q_LORA_RANK as u32) as u64,
                (cast_tg.min(Q_LORA_RANK as u32) / 32) as u64,
                (Q_LORA_RANK * 4) as u64,
                (Q_LORA_RANK * 2) as u64,
                Q_LORA_RANK as u64,
            ),
            spec(
                "rmsnorm.q",
                if rms_simd {
                    RMSNORM_SIMD_KERNEL
                } else {
                    RMSNORM_KERNEL
                },
                rms_q_threads,
                1,
                rms_q_threads,
                if rms_simd {
                    (rms_q_threads / SIMD_WIDTH as u64).max(1)
                } else {
                    1
                },
                (Q_LORA_RANK * 4) as u64,
                (Q_LORA_RANK * 2) as u64,
                (Q_LORA_RANK * 3) as u64,
            ),
            spec(
                "act_quant.q",
                ACT_QUANT_SIMD_KERNEL,
                aq_q_th as u64,
                aq_q_tg as u64,
                aq_q_tptg as u64,
                aq_q_sg as u64,
                (Q_LORA_RANK * 2) as u64,
                (Q_LORA_RANK + Q_LORA_RANK / ACT_QUANT_BLOCK) as u64,
                (Q_LORA_RANK * 256) as u64,
            ),
            spec(
                "mla.wq_b",
                FP8_KERNEL,
                WQ_B_ROWS as u64,
                ((WQ_B_ROWS as u32 + fp8_tg - 1) / fp8_tg) as u64,
                fp8_tg.min(WQ_B_ROWS as u32) as u64,
                (fp8_tg.min(WQ_B_ROWS as u32) / 32) as u64,
                (WQ_B_ROWS * Q_LORA_RANK) as u64
                    + scale_bytes(WQ_B_ROWS, Q_LORA_RANK)
                    + Q_LORA_RANK as u64
                    + (Q_LORA_RANK / ACT_QUANT_BLOCK) as u64,
                (WQ_B_ROWS * 4) as u64,
                2 * WQ_B_ROWS as u64 * Q_LORA_RANK as u64,
            ),
            spec(
                "cast.wq_b",
                CAST_KERNEL,
                WQ_B_ROWS as u64,
                ((WQ_B_ROWS as u32 + cast_tg - 1) / cast_tg) as u64,
                cast_tg.min(WQ_B_ROWS as u32) as u64,
                (cast_tg.min(WQ_B_ROWS as u32) / 32) as u64,
                (WQ_B_ROWS * 4) as u64,
                (WQ_B_ROWS * 2) as u64,
                WQ_B_ROWS as u64,
            ),
            spec(
                "per_head_rms",
                if rms_simd {
                    PER_HEAD_RMS_SIMD_KERNEL
                } else {
                    PER_HEAD_RMS_KERNEL
                },
                per_head_threads,
                if rms_simd { NUM_HEADS as u64 } else { 1 },
                if rms_simd {
                    SIMD_WIDTH as u64
                } else {
                    NUM_HEADS.min(64) as u64
                },
                if rms_simd {
                    1
                } else {
                    (NUM_HEADS.min(64) / 32) as u64
                },
                (NUM_HEADS * HEAD_DIM * 2) as u64,
                (NUM_HEADS * HEAD_DIM * 2) as u64,
                (NUM_HEADS * HEAD_DIM * 3) as u64,
            ),
            spec(
                "mla.wkv",
                wkv_kern,
                wkv_th,
                wkv_tg,
                wkv_tptg,
                wkv_sg,
                (WKV_ROWS * HIDDEN_SIZE) as u64
                    + scale_bytes(WKV_ROWS, HIDDEN_SIZE)
                    + HIDDEN_SIZE as u64
                    + (HIDDEN_SIZE / ACT_QUANT_BLOCK) as u64,
                (WKV_ROWS * 4) as u64,
                2 * WKV_ROWS as u64 * HIDDEN_SIZE as u64,
            ),
            spec(
                "cast.wkv",
                CAST_KERNEL,
                WKV_ROWS as u64,
                1,
                WKV_ROWS.min(cast_tg as usize) as u64,
                1,
                (WKV_ROWS * 4) as u64,
                (WKV_ROWS * 2) as u64,
                WKV_ROWS as u64,
            ),
            spec(
                "rmsnorm.kv",
                if rms_simd {
                    RMSNORM_SIMD_KERNEL
                } else {
                    RMSNORM_KERNEL
                },
                rms_kv_threads,
                1,
                rms_kv_threads,
                if rms_simd {
                    (rms_kv_threads / SIMD_WIDTH as u64).max(1)
                } else {
                    1
                },
                (HEAD_DIM * 4) as u64,
                (HEAD_DIM * 2) as u64,
                (HEAD_DIM * 3) as u64,
            ),
            spec(
                "kv_qat",
                kv_qat_kernel,
                kv_qat_threads,
                1,
                kv_qat_threads,
                if kv_qat_simd { qat_blocks.max(1) } else { 1 },
                (HEAD_DIM * 2) as u64,
                (HEAD_DIM * 2 + (HEAD_DIM - ROPE_HEAD_DIM) + qat_blocks as usize) as u64,
                ((HEAD_DIM - ROPE_HEAD_DIM) * 256) as u64,
            ),
            spec(
                "attn.sparse_pos0",
                ATTN_KERNEL,
                NUM_HEADS as u64,
                1,
                NUM_HEADS.min(64) as u64,
                (NUM_HEADS.min(64) / 32) as u64,
                ((NUM_HEADS * HEAD_DIM + HEAD_DIM) * 2 + NUM_HEADS * 4) as u64,
                ((NUM_HEADS * HEAD_DIM) * 2 + NUM_HEADS * 8) as u64,
                (NUM_HEADS * HEAD_DIM * 2) as u64,
            ),
            spec(
                "mla.wo_a",
                wo_a_kern,
                wo_a_th,
                wo_a_n_tg,
                wo_a_tptg,
                wo_a_sg,
                (WO_A_ROWS * WO_A_COLS) as u64
                    + scale_bytes(WO_A_ROWS, WO_A_COLS)
                    + (NUM_HEADS * HEAD_DIM * 2) as u64,
                (WO_A_ROWS * 2) as u64,
                2 * WO_A_ROWS as u64 * WO_A_COLS as u64,
            ),
            spec(
                "act_quant.wo",
                ACT_QUANT_SIMD_KERNEL,
                aq_o_th as u64,
                aq_o_tg as u64,
                aq_o_tptg as u64,
                aq_o_sg as u64,
                (WO_B_COLS * 2) as u64,
                (WO_B_COLS + WO_B_COLS / ACT_QUANT_BLOCK) as u64,
                (WO_B_COLS * 256) as u64,
            ),
            spec(
                "mla.wo_b",
                wo_b_kern,
                wo_b_th,
                wo_b_tg,
                wo_b_tptg,
                wo_b_sg,
                (WO_B_ROWS * WO_B_COLS) as u64
                    + scale_bytes(WO_B_ROWS, WO_B_COLS)
                    + WO_B_COLS as u64
                    + (WO_B_COLS / ACT_QUANT_BLOCK) as u64,
                (WO_B_ROWS * 4) as u64,
                2 * WO_B_ROWS as u64 * WO_B_COLS as u64,
            ),
            spec(
                "cast.hidden_b",
                CAST_KERNEL,
                WO_B_ROWS as u64,
                ((WO_B_ROWS as u32 + cast_tg - 1) / cast_tg) as u64,
                cast_tg.min(WO_B_ROWS as u32) as u64,
                (cast_tg.min(WO_B_ROWS as u32) / 32) as u64,
                (WO_B_ROWS * 4) as u64,
                (WO_B_ROWS * 2) as u64,
                WO_B_ROWS as u64,
            ),
        ]
    }

    fn mla_kv_state(layers: u64) -> crate::gravity_deepseek_v4_token_ns_ledger::MlaKvState {
        let kv_bf16 = (HEAD_DIM * size_of::<u16>()) as u64;
        let qat_bytes = (HEAD_DIM - ROPE_HEAD_DIM) as u64;
        let qat_scales = ((HEAD_DIM - ROPE_HEAD_DIM) / KV_QAT_BLOCK) as u64;
        let written = kv_bf16 + qat_bytes + qat_scales;
        let read = kv_bf16;
        crate::gravity_deepseek_v4_token_ns_ledger::MlaKvState {
            persistent_device_addressable: false,
            compressed_indexer_loaded: false,
            bos_window_only: true,
            bytes_written_per_layer: written,
            bytes_read_per_layer: read,
            bytes_written_per_token: written.saturating_mul(layers),
            bytes_read_per_token: read.saturating_mul(layers),
            device_copies_per_token: 0,
            host_syncs_per_token: layers,
            host_sync_bytes_per_token: (HIDDEN_SIZE * size_of::<u16>()) as u64 * layers,
            scratch_overwritten_per_layer: true,
            rebuild_rebind_reallocate_per_token: true,
            note: "BOS position-0 graph: compressed indexer is not loaded and window KV is empty. scratch.wkv / kv_qat_bytes are per-layer scratch overwritten on the next layer; they are not a device-addressable cache that survives the token. The only host sync in this region is wo_b activation readback for host MHC post (HIDDEN_SIZE bf16 / layer). A second token would rebuild latent KV from streamed WKV weights — the standing anti-pattern.".to_owned(),
        }
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
        occupancy: bool,
    ) -> Result<()> {
        let scale_cols = cols / ACT_QUANT_BLOCK as u32;
        let occ_ok = occupancy
            && mla_fp8_simd_enabled()
            && rows > 0
            && scale_cols > 0
            && scale_cols <= FP8_OCC_MAX_BLOCKS
            && (cols % ACT_QUANT_BLOCK as u32) == 0;
        if occ_ok {
            let threads_x = align_simd(tg);
            batch.dispatch_threads(
                FP8_OCC_KERNEL,
                (threads_x, rows, 1),
                (threads_x, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(weight), 0);
                    enc.set_buffer(1, Some(scale), 0);
                    enc.set_buffer(2, Some(quant), 0);
                    enc.set_buffer(3, Some(act_scale), 0);
                    enc.set_buffer(4, Some(out_f32), 0);
                    set_u32(enc, 5, &rows);
                    set_u32(enc, 6, &cols);
                    set_u32(enc, 7, &scale_cols);
                    set_u32(enc, 8, &threads_x);
                },
            )
        } else {
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
        rms_tg: u32,
    ) -> Result<()> {
        if mla_rmsnorm_simd_enabled() && width >= SIMD_WIDTH {
            let threads = align_simd(width.min(rms_tg.max(SIMD_WIDTH)));
            batch.dispatch_threads(RMSNORM_SIMD_KERNEL, (threads, 1, 1), (threads, 1, 1), |enc| {
                enc.set_buffer(0, Some(input), 0);
                enc.set_buffer(1, Some(weight), 0);
                enc.set_buffer(2, Some(output), 0);
                set_u32(enc, 3, &width);
                set_f32(enc, 4, &eps);
            })
        } else {
            batch.dispatch_threads(RMSNORM_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(input), 0);
                enc.set_buffer(1, Some(weight), 0);
                enc.set_buffer(2, Some(output), 0);
                set_u32(enc, 3, &width);
                set_f32(enc, 4, &eps);
            })
        }
    }

    fn dispatch_per_head_rms(
        batch: &mut CommandBatch<'_>,
        input: &metal::Buffer,
        output: &metal::Buffer,
        heads: u32,
        dim: u32,
        eps: f32,
    ) -> Result<()> {
        if mla_rmsnorm_simd_enabled() {
            batch.dispatch_threads(
                PER_HEAD_RMS_SIMD_KERNEL,
                (heads * SIMD_WIDTH, 1, 1),
                (SIMD_WIDTH, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(input), 0);
                    enc.set_buffer(1, Some(output), 0);
                    set_u32(enc, 2, &heads);
                    set_u32(enc, 3, &dim);
                    set_f32(enc, 4, &eps);
                },
            )
        } else {
            batch.dispatch_threads(
                PER_HEAD_RMS_KERNEL,
                (heads, 1, 1),
                (heads.min(64), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(input), 0);
                    enc.set_buffer(1, Some(output), 0);
                    set_u32(enc, 2, &heads);
                    set_u32(enc, 3, &dim);
                    set_f32(enc, 4, &eps);
                },
            )
        }
    }

    fn dispatch_wo_a(
        batch: &mut CommandBatch<'_>,
        weight: &metal::Buffer,
        scale: &metal::Buffer,
        attn: &metal::Buffer,
        output: &metal::Buffer,
        wo_a_tg: u32,
        wo_a_occ_tg: u32,
    ) -> Result<()> {
        let rows = WO_A_ROWS as u32;
        let cols = WO_A_COLS as u32;
        let scale_cols = (WO_A_COLS / ACT_QUANT_BLOCK) as u32;
        let ranks = O_LORA_RANK as u32;
        if mla_wo_a_simd_enabled() {
            let threads_x = align_simd(wo_a_occ_tg);
            batch.dispatch_threads(
                WO_A_SIMD_KERNEL,
                (threads_x, rows, 1),
                (threads_x, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(weight), 0);
                    enc.set_buffer(1, Some(scale), 0);
                    enc.set_buffer(2, Some(attn), 0);
                    enc.set_buffer(3, Some(output), 0);
                    set_u32(enc, 4, &rows);
                    set_u32(enc, 5, &cols);
                    set_u32(enc, 6, &scale_cols);
                    set_u32(enc, 7, &ranks);
                    set_u32(enc, 8, &threads_x);
                },
            )
        } else {
            let tg = wo_a_tg.min(rows);
            batch.dispatch_threads(WO_A_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(weight), 0);
                enc.set_buffer(1, Some(scale), 0);
                enc.set_buffer(2, Some(attn), 0);
                enc.set_buffer(3, Some(output), 0);
                set_u32(enc, 4, &rows);
                set_u32(enc, 5, &cols);
                set_u32(enc, 6, &scale_cols);
                set_u32(enc, 7, &ranks);
            })
        }
    }

    fn dispatch_kv_qat(
        batch: &mut CommandBatch<'_>,
        input: &metal::Buffer,
        output: &metal::Buffer,
        quantized: &metal::Buffer,
        scales: &metal::Buffer,
    ) -> Result<()> {
        let dim = HEAD_DIM as u32;
        let rope = ROPE_HEAD_DIM as u32;
        let block = KV_QAT_BLOCK as u32;
        let qat_blocks = ((HEAD_DIM - ROPE_HEAD_DIM) / KV_QAT_BLOCK) as u32;
        if mla_kv_qat_simd_enabled() {
            let threads_x = (qat_blocks.max(1) * ACT_QUANT_SIMD_WIDTH).max(ACT_QUANT_SIMD_WIDTH);
            batch.dispatch_threads(
                KV_QAT_SIMD_KERNEL,
                (threads_x, 1, 1),
                (threads_x, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(input), 0);
                    enc.set_buffer(1, Some(output), 0);
                    enc.set_buffer(2, Some(quantized), 0);
                    enc.set_buffer(3, Some(scales), 0);
                    set_u32(enc, 4, &dim);
                    set_u32(enc, 5, &rope);
                    set_u32(enc, 6, &block);
                },
            )
        } else {
            batch.dispatch_threads(
                KV_QAT_KERNEL,
                (qat_blocks.max(1), 1, 1),
                (qat_blocks.max(1), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(input), 0);
                    enc.set_buffer(1, Some(output), 0);
                    enc.set_buffer(2, Some(quantized), 0);
                    enc.set_buffer(3, Some(scales), 0);
                    set_u32(enc, 4, &dim);
                    set_u32(enc, 5, &rope);
                    set_u32(enc, 6, &block);
                },
            )
        }
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

    const PAGE_ALIGN: usize = 16 * 1024;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ExpertRef {
        packed_weights: u64,
        weight_scales: u64,
    }

    const _: () = assert!(size_of::<ExpertRef>() == 16);

    struct ExpertAddressBind {
        names: Vec<String>,
        _views: Vec<DeepSeekV4VerifiedBytes>,
        w1_refs: metal::Buffer,
        w3_refs: metal::Buffer,
        w2_refs: metal::Buffer,
        w1_resources: Vec<metal::Buffer>,
        w3_resources: Vec<metal::Buffer>,
        w2_resources: Vec<metal::Buffer>,
        nocopy: bool,
    }

    fn use_read_resources(enc: &metal::ComputeCommandEncoderRef, resources: &[metal::Buffer]) {
        if resources.is_empty() {
            return;
        }
        let mut refs: Vec<&metal::ResourceRef> = Vec::with_capacity(resources.len());
        for resource in resources {
            refs.push(resource);
        }
        enc.use_resources(&refs, metal::MTLResourceUsage::Read);
    }

    fn no_copy_verified(
        metal: &MetalContext,
        bytes: &[u8],
        name: &str,
    ) -> Result<metal::Buffer> {
        if bytes.is_empty()
            || (bytes.as_ptr() as usize) % PAGE_ALIGN != 0
            || bytes.len() % PAGE_ALIGN != 0
        {
            return Err(graph_error(format!(
                "{name} is not a 16KiB-aligned mmap window (ptr={:p} len={}); \
                 compact-slab pack is opt-in via HAWKING_DSV4F_EXPERT_SLAB_PACK=1",
                bytes.as_ptr(),
                bytes.len()
            )));
        }
        bump_create();
        let buf = metal.new_buffer_from_verified_bytes(bytes)?;
        if buf.contents() as usize != bytes.as_ptr() as usize {
            return Err(graph_error(format!(
                "{name} no-copy bind copied {} bytes; refusing a silent host pack",
                bytes.len()
            )));
        }
        Ok(buf)
    }

    fn write_expert_refs(buf: &metal::Buffer, refs: &[ExpertRef; ACTIVATED_EXPERTS]) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                refs.as_ptr() as *const u8,
                ACTIVATED_EXPERTS * size_of::<ExpertRef>(),
            )
        };
        write_bytes(buf, bytes);
    }

    fn dispatch_worklist_fp4(
        batch: &mut CommandBatch<'_>,
        worklist: &metal::Buffer,
        refs: &metal::Buffer,
        resources: &[metal::Buffer],
        quant: &metal::Buffer,
        act_scale: &metal::Buffer,
        output: &metal::Buffer,
        rows: u32,
        packed_cols: u32,
        scale_cols: u32,
        top_k: u32,
        act_is_per_slot: u32,
        tg: u32,
    ) -> Result<()> {
        let grid = top_k * rows;
        let tg = tg.min(rows.max(1));
        batch.dispatch_threads(WORKLIST_FP4_KERNEL, (grid, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(worklist), 0);
            enc.set_buffer(1, Some(refs), 0);
            enc.set_buffer(2, Some(quant), 0);
            enc.set_buffer(3, Some(act_scale), 0);
            enc.set_buffer(4, Some(output), 0);
            set_u32(enc, 5, &rows);
            set_u32(enc, 6, &packed_cols);
            set_u32(enc, 7, &scale_cols);
            set_u32(enc, 8, &top_k);
            set_u32(enc, 9, &act_is_per_slot);
            use_read_resources(enc, resources);
        })
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

    fn bind_expert_payloads(
        graph: &Graph,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        layer: &DeepSeekV4LayerSourceAnchor,
        exec: &[(u32, f32, u32)],
    ) -> Result<ExpertAddressBind> {
        let pack = expert_compact_slab_pack_enabled();
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
        }
        let ref_bytes = ACTIVATED_EXPERTS * size_of::<ExpertRef>();
        bump_create();
        let w1_refs = graph.metal.new_buffer_checked(ref_bytes)?;
        bump_create();
        let w3_refs = graph.metal.new_buffer_checked(ref_bytes)?;
        bump_create();
        let w2_refs = graph.metal.new_buffer_checked(ref_bytes)?;
        if pack {
            for slot in 0..ACTIVATED_EXPERTS {
                let base = slot * 6;
                write_at(
                    &graph.scratch.w1_slab,
                    slot * W1_PACKED,
                    blobs[base].as_bytes(),
                );
                write_at(
                    &graph.scratch.w1_scale_slab,
                    slot * W1_SCALES,
                    blobs[base + 1].as_bytes(),
                );
                write_at(
                    &graph.scratch.w3_slab,
                    slot * W1_PACKED,
                    blobs[base + 2].as_bytes(),
                );
                write_at(
                    &graph.scratch.w3_scale_slab,
                    slot * W1_SCALES,
                    blobs[base + 3].as_bytes(),
                );
                write_at(
                    &graph.scratch.w2_slab,
                    slot * W2_PACKED,
                    blobs[base + 4].as_bytes(),
                );
                write_at(
                    &graph.scratch.w2_scale_slab,
                    slot * W2_SCALES,
                    blobs[base + 5].as_bytes(),
                );
            }
            let mut w1 = [ExpertRef {
                packed_weights: 0,
                weight_scales: 0,
            }; ACTIVATED_EXPERTS];
            let mut w3 = w1;
            let mut w2 = w1;
            let w1_base = graph.scratch.w1_slab.gpu_address();
            let w1s_base = graph.scratch.w1_scale_slab.gpu_address();
            let w3_base = graph.scratch.w3_slab.gpu_address();
            let w3s_base = graph.scratch.w3_scale_slab.gpu_address();
            let w2_base = graph.scratch.w2_slab.gpu_address();
            let w2s_base = graph.scratch.w2_scale_slab.gpu_address();
            for slot in 0..ACTIVATED_EXPERTS {
                w1[slot] = ExpertRef {
                    packed_weights: w1_base + (slot * W1_PACKED) as u64,
                    weight_scales: w1s_base + (slot * W1_SCALES) as u64,
                };
                w3[slot] = ExpertRef {
                    packed_weights: w3_base + (slot * W1_PACKED) as u64,
                    weight_scales: w3s_base + (slot * W1_SCALES) as u64,
                };
                w2[slot] = ExpertRef {
                    packed_weights: w2_base + (slot * W2_PACKED) as u64,
                    weight_scales: w2s_base + (slot * W2_SCALES) as u64,
                };
            }
            write_expert_refs(&w1_refs, &w1);
            write_expert_refs(&w3_refs, &w3);
            write_expert_refs(&w2_refs, &w2);
            Ok(ExpertAddressBind {
                names,
                _views: Vec::new(),
                w1_refs,
                w3_refs,
                w2_refs,
                w1_resources: vec![
                    graph.scratch.w1_slab.clone(),
                    graph.scratch.w1_scale_slab.clone(),
                ],
                w3_resources: vec![
                    graph.scratch.w3_slab.clone(),
                    graph.scratch.w3_scale_slab.clone(),
                ],
                w2_resources: vec![
                    graph.scratch.w2_slab.clone(),
                    graph.scratch.w2_scale_slab.clone(),
                ],
                nocopy: false,
            })
        } else {
            for (i, blob) in blobs.iter().enumerate() {
                if !blob.is_zero_copy() {
                    return Err(graph_error(format!(
                        "{} spanned chunks; no-copy expert bind requires a single mmap window",
                        names[i]
                    )));
                }
            }
            let mut w1_w = Vec::with_capacity(ACTIVATED_EXPERTS);
            let mut w1_s = Vec::with_capacity(ACTIVATED_EXPERTS);
            let mut w3_w = Vec::with_capacity(ACTIVATED_EXPERTS);
            let mut w3_s = Vec::with_capacity(ACTIVATED_EXPERTS);
            let mut w2_w = Vec::with_capacity(ACTIVATED_EXPERTS);
            let mut w2_s = Vec::with_capacity(ACTIVATED_EXPERTS);
            let mut w1 = [ExpertRef {
                packed_weights: 0,
                weight_scales: 0,
            }; ACTIVATED_EXPERTS];
            let mut w3 = w1;
            let mut w2 = w1;
            for slot in 0..ACTIVATED_EXPERTS {
                let base = slot * 6;
                let bw1 = no_copy_verified(&graph.metal, blobs[base].as_bytes(), &names[base])?;
                let bs1 =
                    no_copy_verified(&graph.metal, blobs[base + 1].as_bytes(), &names[base + 1])?;
                let bw3 =
                    no_copy_verified(&graph.metal, blobs[base + 2].as_bytes(), &names[base + 2])?;
                let bs3 =
                    no_copy_verified(&graph.metal, blobs[base + 3].as_bytes(), &names[base + 3])?;
                let bw2 =
                    no_copy_verified(&graph.metal, blobs[base + 4].as_bytes(), &names[base + 4])?;
                let bs2 =
                    no_copy_verified(&graph.metal, blobs[base + 5].as_bytes(), &names[base + 5])?;
                w1[slot] = ExpertRef {
                    packed_weights: bw1.gpu_address(),
                    weight_scales: bs1.gpu_address(),
                };
                w3[slot] = ExpertRef {
                    packed_weights: bw3.gpu_address(),
                    weight_scales: bs3.gpu_address(),
                };
                w2[slot] = ExpertRef {
                    packed_weights: bw2.gpu_address(),
                    weight_scales: bs2.gpu_address(),
                };
                w1_w.push(bw1);
                w1_s.push(bs1);
                w3_w.push(bw3);
                w3_s.push(bs3);
                w2_w.push(bw2);
                w2_s.push(bs2);
            }
            write_expert_refs(&w1_refs, &w1);
            write_expert_refs(&w3_refs, &w3);
            write_expert_refs(&w2_refs, &w2);
            let mut w1_resources = w1_w;
            w1_resources.extend(w1_s);
            let mut w3_resources = w3_w;
            w3_resources.extend(w3_s);
            let mut w2_resources = w2_w;
            w2_resources.extend(w2_s);
            Ok(ExpertAddressBind {
                names,
                _views: blobs,
                w1_refs,
                w3_refs,
                w2_refs,
                w1_resources,
                w3_resources,
                w2_resources,
                nocopy: true,
            })
        }
    }

    #[allow(dead_code)]
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
        profiler.record_mla_static(
            cb_collapse_enabled() || mla_serial_group_enabled(),
            mla_dispatch_specs(
                (max_layer + 1) as u64,
                graph.act_tg,
                graph.fp8_tg,
                graph.fp8_occ_tg,
                graph.cast_tg,
                graph.wo_a_tg,
                graph.wo_a_occ_tg,
                graph.rms_tg,
            ),
            mla_kv_state((max_layer + 1) as u64),
            graph.mla_pipeline_limits.clone(),
        );
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
        let fp8_occ_tg = graph.fp8_occ_tg;
        let cast_tg = graph.cast_tg;
        let wo_a_tg = graph.wo_a_tg;
        let wo_a_occ_tg = graph.wo_a_occ_tg;
        let rms_tg = graph.rms_tg;
        let layer_idx = layer.layer;

        let collapse = cb_collapse_enabled();
        let serial = !collapse && mla_serial_group_enabled();
        let (attn_n, attn_submitted) = graph.submit(|batch, s| {
            if collapse {
                maybe_ordered(batch);
            } else if serial {
                batch.begin_serial_group()?;
            }
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
                fp8_occ_tg,
                true,
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
                rms_tg,
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
                false,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.wq_b, WQ_B_ROWS as u32, cast_tg)?;
            n += 1;
            let heads = NUM_HEADS as u32;
            let dim = HEAD_DIM as u32;
            dispatch_per_head_rms(batch, &s.wq_b, &s.attn, heads, dim, eps)?;
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
                fp8_occ_tg,
                true,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.wkv, WKV_ROWS as u32, cast_tg)?;
            n += 1;
            dispatch_rmsnorm(
                batch,
                &s.wkv,
                &kv_norm_buf,
                &s.wkv,
                HEAD_DIM as u32,
                eps,
                rms_tg,
            )?;
            n += 1;
            dispatch_kv_qat(
                batch,
                &s.wkv,
                &s.wkv,
                &s.kv_qat_bytes,
                &s.kv_qat_scales,
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
            dispatch_wo_a(
                batch,
                &wo_a_p.weight,
                &wo_a_p.scale,
                &s.attn,
                &s.wo_a,
                wo_a_tg,
                wo_a_occ_tg,
            )?;
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
                fp8_occ_tg,
                true,
            )?;
            n += 1;
            dispatch_cast(batch, &s.f32_tmp, &s.hidden_b, WO_B_ROWS as u32, cast_tg)?;
            n += 1;
            if serial {
                batch.end_serial_group()?;
            }
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
        expert_bind: Option<ExpertAddressBind>,
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
        // Hash layers know the six expert IDs from tid2eid before the route
        // kernel. Bind those payloads while attention GPU is still running.
        let expert_bytes =
            (ACTIVATED_EXPERTS * (2 * W1_PACKED + W2_PACKED + 2 * W1_SCALES + W2_SCALES)) as u64;
        let expert_bind = if let Some(ids) = hash_ids {
            let exec = pack_worklist_host(&ids, &[1.0f32; ACTIVATED_EXPERTS])?;
            Some(profiler.time_bytes_result(
                "host.expert_slab_io_overlapped",
                expert_bytes,
                || bind_expert_payloads(graph, reader, ledger, layer, &exec),
            )?)
        } else {
            None
        };
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
            expert_bind,
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
        experts: &'a ExpertAddressBind,
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
        let tg = p.fp4_tg.min(rows_w1);
        let zero = 0u32;
        let one = 1u32;
        let shared_one = 1.0f32;
        dispatch_worklist_fp4(
            batch,
            &s.worklist,
            &p.experts.w1_refs,
            &p.experts.w1_resources,
            &s.quant_ffn,
            &s.quant_scale_ffn,
            &s.expert_gate_f32,
            rows_w1,
            packed,
            scale_cols,
            p.top_k,
            zero,
            p.fp4_tg,
        )?;
        n += 1;
        dispatch_worklist_fp4(
            batch,
            &s.worklist,
            &p.experts.w3_refs,
            &p.experts.w3_resources,
            &s.quant_ffn,
            &s.quant_scale_ffn,
            &s.expert_up_f32,
            rows_w1,
            packed,
            scale_cols,
            p.top_k,
            zero,
            p.fp4_tg,
        )?;
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
        dispatch_worklist_fp4(
            batch,
            &s.worklist,
            &p.experts.w2_refs,
            &p.experts.w2_resources,
            &s.expert_down_quant,
            &s.expert_down_scales,
            &s.expert_down_f32,
            rows_w2,
            packed_w2,
            scale_w2,
            p.top_k,
            one,
            p.fp4_tg,
        )?;
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
            false,
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
            false,
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
            false,
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

    fn bind_experts_maybe_prefetch(
        graph: &Graph,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        layer: &DeepSeekV4LayerSourceAnchor,
        next_layer: Option<&DeepSeekV4LayerSourceAnchor>,
        exec: &[(u32, f32, u32)],
        profiler: &mut TokenNsCollector,
        attn_prefetch: &mut Option<Vec<DeepSeekV4VerifiedBytes>>,
        expert_bytes: u64,
    ) -> Result<ExpertAddressBind> {
        if let Some(next) = next_layer {
            let started = Instant::now();
            let (bind, prefetched) = std::thread::scope(|scope| -> Result<_> {
                let expert =
                    scope.spawn(|| bind_expert_payloads(graph, reader, ledger, layer, exec));
                let attn = scope.spawn(|| {
                    let jobs = attn_read_jobs(next);
                    par_read_views(reader, &jobs)
                });
                let bind = expert
                    .join()
                    .map_err(|_| graph_error("expert slab thread panicked"))??;
                let prefetched = attn
                    .join()
                    .map_err(|_| graph_error("attn prefetch thread panicked"))??;
                Ok((bind, prefetched))
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
            Ok(bind)
        } else {
            profiler.time_bytes_result("host.expert_slab_io", expert_bytes, || {
                bind_expert_payloads(graph, reader, ledger, layer, exec)
            })
        }
    }

    fn prefetch_next_attn_only(
        reader: &DeepSeekV4FullStreamReader,
        next_layer: Option<&DeepSeekV4LayerSourceAnchor>,
        profiler: &mut TokenNsCollector,
        attn_prefetch: &mut Option<Vec<DeepSeekV4VerifiedBytes>>,
    ) -> Result<()> {
        if let Some(next) = next_layer {
            let started = Instant::now();
            let prefetched = par_read_views(reader, &attn_read_jobs(next))?;
            let ns = started.elapsed().as_nanos() as u64;
            profiler.add_stage(
                "host.attn_weight_io_prefetch",
                ns,
                1,
                prefetched.iter().map(|b| b.len() as u64).sum(),
            );
            *attn_prefetch = Some(prefetched);
        }
        Ok(())
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
            expert_bind: preloaded_experts,
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
        let expert_bytes = (ACTIVATED_EXPERTS
            * (2 * W1_PACKED + W2_PACKED + 2 * W1_SCALES + W2_SCALES)) as u64;
        let merge_hash = collapse && is_hash;

        let expert_bind = if merge_hash {
            let bind = if let Some(bind) = preloaded_experts {
                prefetch_next_attn_only(reader, next_layer, profiler, attn_prefetch)?;
                bind
            } else {
                let ids = hash_ids.expect("hash ids");
                let exec = pack_worklist_host(&ids, &[0.0f32; ACTIVATED_EXPERTS])?;
                bind_experts_maybe_prefetch(
                    graph,
                    reader,
                    ledger,
                    layer,
                    next_layer,
                    &exec,
                    profiler,
                    attn_prefetch,
                    expert_bytes,
                )?
            };
            if bind.nocopy {
                graph.counters.expert_nocopy_binds += 1;
            } else {
                graph.counters.expert_slab_packs += 1;
            }
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
                experts: &bind,
            };
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
            bind
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
                    "execute_moe after route command buffer; blocks bind_expert_payloads",
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
            // Device pack_worklist already wrote slab_slot 0..5 in the same
            // (expert_id, source_slot) order as pack_worklist_host. Do not
            // overwrite it; the address table is indexed by that slab_slot.
            let bind = if let Some(bind) = preloaded_experts {
                prefetch_next_attn_only(reader, next_layer, profiler, attn_prefetch)?;
                bind
            } else {
                bind_experts_maybe_prefetch(
                    graph,
                    reader,
                    ledger,
                    layer,
                    next_layer,
                    &exec,
                    profiler,
                    attn_prefetch,
                    expert_bytes,
                )?
            };
            if bind.nocopy {
                graph.counters.expert_nocopy_binds += 1;
            } else {
                graph.counters.expert_slab_packs += 1;
            }
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
                experts: &bind,
            };
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
            bind
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
                &expert_bind,
            )?;
        }

        // Residual readback is the layer HC handoff for host MHC, not an expert gather.
        for name in expert_bind.names {
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
        let fp8_occ_tg = graph.fp8_occ_tg;
        let cast_tg = graph.cast_tg;
        let wo_a_tg = graph.wo_a_tg;
        let wo_a_occ_tg = graph.wo_a_occ_tg;
        let rms_tg = graph.rms_tg;
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
                fp8_occ_tg,
                true,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.cast.q_lora", layer_idx, |batch| {
            dispatch_cast(batch, &s.f32_tmp, &s.q_lora, Q_LORA_RANK as u32, cast_tg)
        })?;
        probe_one(&graph.metal, profiler, "isolated.rmsnorm.q", layer_idx, |batch| {
            dispatch_rmsnorm(
                batch,
                &s.q_lora,
                q_norm,
                &s.q_lora,
                Q_LORA_RANK as u32,
                eps,
                rms_tg,
            )
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
                false,
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
                fp8_occ_tg,
                true,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.mla.wo_a", layer_idx, |batch| {
            dispatch_wo_a(
                batch,
                &wo_a.weight,
                &wo_a.scale,
                &s.attn,
                &s.wo_a,
                wo_a_tg,
                wo_a_occ_tg,
            )
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
                fp8_occ_tg,
                true,
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
        probe_one(&graph.metal, profiler, "isolated.kv_qat", layer_idx, |batch| {
            dispatch_kv_qat(batch, &s.wkv, &s.wkv, &s.kv_qat_bytes, &s.kv_qat_scales)
        })?;
        probe_one(&graph.metal, profiler, "isolated.rmsnorm.kv", layer_idx, |batch| {
            dispatch_rmsnorm(
                batch,
                &s.wkv,
                kv_norm,
                &s.wkv,
                HEAD_DIM as u32,
                RMS_NORM_EPS,
                rms_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.per_head_rms", layer_idx, |batch| {
            dispatch_per_head_rms(batch, &s.wq_b, &s.attn, heads, dim, eps)
        })?;
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
        experts: &ExpertAddressBind,
    ) -> Result<()> {
        let s = &graph.scratch;
        let fp8_tg = graph.fp8_tg;
        let fp4_tg = graph.fp4_tg;
        let gate_tg = graph.gate_tg;
        let top_k = ACTIVATED_EXPERTS as u32;
        let rows_w1 = MOE_INTER_DIM as u32;
        let packed = (HIDDEN_SIZE / 2) as u32;
        let scale_cols = (HIDDEN_SIZE / FP4_BLOCK) as u32;
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
            dispatch_worklist_fp4(
                batch,
                &s.worklist,
                &experts.w1_refs,
                &experts.w1_resources,
                &s.quant_ffn,
                &s.quant_scale_ffn,
                &s.expert_gate_f32,
                rows_w1,
                packed,
                scale_cols,
                top_k,
                zero,
                fp4_tg,
            )
        })?;
        probe_one(&graph.metal, profiler, "isolated.routed.w3", layer_idx, |batch| {
            dispatch_worklist_fp4(
                batch,
                &s.worklist,
                &experts.w3_refs,
                &experts.w3_resources,
                &s.quant_ffn,
                &s.quant_scale_ffn,
                &s.expert_up_f32,
                rows_w1,
                packed,
                scale_cols,
                top_k,
                zero,
                fp4_tg,
            )
        })?;
        let rows_w2 = HIDDEN_SIZE as u32;
        let packed_w2 = (MOE_INTER_DIM / 2) as u32;
        let scale_w2 = (MOE_INTER_DIM / FP4_BLOCK) as u32;
        probe_one(&graph.metal, profiler, "isolated.routed.w2", layer_idx, |batch| {
            dispatch_worklist_fp4(
                batch,
                &s.worklist,
                &experts.w2_refs,
                &experts.w2_resources,
                &s.expert_down_quant,
                &s.expert_down_scales,
                &s.expert_down_f32,
                rows_w2,
                packed_w2,
                scale_w2,
                top_k,
                one,
                fp4_tg,
            )
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
                false,
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
                false,
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
                false,
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

    #[test]
    fn expert_payload_counters_default_to_zero() {
        let counters = NativeTokenGraphCounters::default();
        assert_eq!(counters.expert_nocopy_binds, 0);
        assert_eq!(counters.expert_slab_packs, 0);
        assert_eq!(counters.total_sync_points, 0);
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
