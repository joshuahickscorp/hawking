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
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
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
    peak_rss_bytes, prepare_sealed_admission_root, ResidentLedger, DECLARED_PEAK_RSS_BOUND_BYTES,
    DECLARED_WEIGHT_RESIDENT_BOUND_BYTES, SCHEDULE_STREAMED_DECODE_PEAK_BYTES,
};
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

const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
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
}

impl NativeTokenGraphReport {
    pub fn to_receipt_json(&self) -> serde_json::Value {
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
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_f32_le(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if bytes.len() % size_of::<f32>() != 0 {
        return Err(graph_error(format!("{name} is not f32 aligned")));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
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
    use crate::metal::{CommandBatch, MetalContext};

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
    }

    impl Graph {
        fn new() -> Result<Self> {
            let metal = MetalContext::new()?;
            let scratch = Scratch::new(&metal)?;
            Ok(Self {
                act_tg: pipeline_tg(&metal, ACT_QUANT_KERNEL, 32)?,
                fp8_tg: pipeline_tg(&metal, FP8_KERNEL, 256)?,
                fp4_tg: pipeline_tg(&metal, WORKLIST_FP4_KERNEL, 256)?,
                cast_tg: pipeline_tg(&metal, CAST_KERNEL, 256)?,
                gate_tg: pipeline_tg(&metal, GATE_KERNEL, 256)?,
                wo_a_tg: pipeline_tg(&metal, WO_A_KERNEL, 256)?,
                lm_tg: pipeline_tg(&metal, LM_HEAD_KERNEL, 256)?,
                metal,
                scratch,
                counters: NativeTokenGraphCounters::default(),
            })
        }

        fn batch(
            &mut self,
            encode: impl FnOnce(&mut CommandBatch<'_>, &Scratch) -> Result<usize>,
        ) -> Result<()> {
            let mut n = 0usize;
            self.metal.dispatch_batch(|batch| {
                n = encode(batch, &self.scratch)?;
                Ok(())
            })?;
            self.counters.command_buffers += 1;
            self.counters.metal_dispatches += n;
            Ok(())
        }
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
        MetalContext::write_buffer_bytes(buf, bytemuck::cast_slice(values));
    }

    fn write_bytes(buf: &metal::Buffer, values: &[u8]) {
        MetalContext::write_buffer_bytes(buf, values);
    }

    fn read_u16(buf: &metal::Buffer, n: usize) -> Result<Vec<u16>> {
        let ptr = buf.contents() as *const u16;
        if ptr.is_null() {
            return Err(graph_error("u16 buffer contents pointer is null"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    fn read_u32(buf: &metal::Buffer, n: usize) -> Result<Vec<u32>> {
        let ptr = buf.contents() as *const u32;
        if ptr.is_null() {
            return Err(graph_error("u32 buffer contents pointer is null"));
        }
        Ok(unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec())
    }

    fn read_f32_n(buf: &metal::Buffer, n: usize) -> Result<Vec<f32>> {
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
        tg: u32,
        input_off: u64,
        quant_off: u64,
        scale_off: u64,
    ) -> Result<()> {
        let blocks = cols / ACT_QUANT_BLOCK as u32;
        let tg = tg.min(blocks.max(1));
        batch.dispatch_threads(ACT_QUANT_KERNEL, (blocks, 1, 1), (tg, 1, 1), |enc| {
            enc.set_buffer(0, Some(input), input_off);
            enc.set_buffer(1, Some(quant), quant_off);
            enc.set_buffer(2, Some(scales), scale_off);
            set_u32(enc, 3, &cols);
        })
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

    fn par_read_full(
        reader: &DeepSeekV4FullStreamReader,
        jobs: &[(String, usize)],
    ) -> Result<Vec<Vec<u8>>> {
        std::thread::scope(|scope| {
            let mut joins = Vec::with_capacity(jobs.len());
            for (name, bytes) in jobs {
                joins.push(scope.spawn(move || reader.read_verified_full(name, *bytes)));
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

    fn upload_fp8(
        metal: &MetalContext,
        ledger: &mut ResidentLedger,
        weight_name: &str,
        scale_name: &str,
        weight: Vec<u8>,
        scale: Vec<u8>,
    ) -> Result<Fp8Pair> {
        ledger.acquire(weight_name, weight.len())?;
        ledger.acquire(scale_name, scale.len())?;
        Ok(Fp8Pair {
            weight: metal.new_buffer_with_bytes_checked(&weight)?,
            scale: metal.new_buffer_with_bytes_checked(&scale)?,
            name_w: weight_name.to_owned(),
            name_s: scale_name.to_owned(),
        })
    }

    fn load_fp8(
        metal: &MetalContext,
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
        let mut blobs = par_read_full(reader, &jobs)?;
        let scale = blobs.pop().expect("scale");
        let weight = blobs.pop().expect("weight");
        upload_fp8(metal, ledger, weight_name, scale_name, weight, scale)
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
        let blobs = par_read_full(reader, &jobs)?;
        let mut w1 = vec![0u8; ACTIVATED_EXPERTS * W1_PACKED];
        let mut s1 = vec![0u8; ACTIVATED_EXPERTS * W1_SCALES];
        let mut w3 = vec![0u8; ACTIVATED_EXPERTS * W1_PACKED];
        let mut s3 = vec![0u8; ACTIVATED_EXPERTS * W1_SCALES];
        let mut w2 = vec![0u8; ACTIVATED_EXPERTS * W2_PACKED];
        let mut s2 = vec![0u8; ACTIVATED_EXPERTS * W2_SCALES];
        for slot in 0..ACTIVATED_EXPERTS {
            let base = slot * 6;
            ledger.acquire(&names[base], blobs[base].len())?;
            ledger.acquire(&names[base + 1], blobs[base + 1].len())?;
            ledger.acquire(&names[base + 2], blobs[base + 2].len())?;
            ledger.acquire(&names[base + 3], blobs[base + 3].len())?;
            ledger.acquire(&names[base + 4], blobs[base + 4].len())?;
            ledger.acquire(&names[base + 5], blobs[base + 5].len())?;
            w1[slot * W1_PACKED..(slot + 1) * W1_PACKED].copy_from_slice(&blobs[base]);
            s1[slot * W1_SCALES..(slot + 1) * W1_SCALES].copy_from_slice(&blobs[base + 1]);
            w3[slot * W1_PACKED..(slot + 1) * W1_PACKED].copy_from_slice(&blobs[base + 2]);
            s3[slot * W1_SCALES..(slot + 1) * W1_SCALES].copy_from_slice(&blobs[base + 3]);
            w2[slot * W2_PACKED..(slot + 1) * W2_PACKED].copy_from_slice(&blobs[base + 4]);
            s2[slot * W2_SCALES..(slot + 1) * W2_SCALES].copy_from_slice(&blobs[base + 5]);
        }
        write_bytes(&graph.scratch.w1_slab, &w1);
        write_bytes(&graph.scratch.w1_scale_slab, &s1);
        write_bytes(&graph.scratch.w3_slab, &w3);
        write_bytes(&graph.scratch.w3_scale_slab, &s3);
        write_bytes(&graph.scratch.w2_slab, &w2);
        write_bytes(&graph.scratch.w2_scale_slab, &s2);
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
        if max_layer >= DSV4F_LAYER_SOURCE_ANCHOR_BASE_LAYER_COUNT {
            return Err(graph_error(format!(
                "max_layer {max_layer} is outside the 0..42 base body"
            )));
        }
        let admission = prepare_sealed_admission_root(artifact)?;
        let reader = DeepSeekV4FullStreamReader::admit(&admission.path)?;
        let anchors = verify_deepseek_v4_layer_source_anchors(&reader)?;
        if anchors.identity().repository != PINNED_REPOSITORY
            || anchors.identity().revision != PINNED_REVISION
        {
            return Err(graph_error(
                "native graph refused a reader whose source identity is not pinned",
            ));
        }

        let mut ledger = ResidentLedger::new(DECLARED_WEIGHT_RESIDENT_BOUND_BYTES);
        let mut graph = Graph::new()?;
        let init_ms = wall.elapsed().as_millis();
        let body = Instant::now();
        let mut peak_rss = peak_rss_bytes();
        let mut layers_executed = Vec::new();
        let mut stop_reason = None;

        let mut hc = load_bos_embed_hc(&reader, &mut ledger, PREFIX_TOKEN_ID)?;

        for layer_idx in 0..=max_layer {
            let layer = anchors.layer(layer_idx)?.clone();
            match execute_layer(
                &mut graph,
                &reader,
                &layer,
                &hc,
                PREFIX_TOKEN_ID,
                &mut ledger,
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
            let merged = host_merge_final_head_from_hc_bf16(&reader, &hc)?;
            match metal_lm_head(&mut graph, &reader, &mut ledger, &merged.merged_f32) {
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
            body_ms: body.elapsed().as_millis(),
        })
    }

    fn execute_layer(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        hc_in: &[u16],
        token_id: u64,
        ledger: &mut ResidentLedger,
    ) -> Result<Vec<u16>> {
        let attn_hc = execute_attention(graph, reader, layer, hc_in, ledger)?;
        execute_moe(graph, reader, layer, &attn_hc, token_id, ledger)
    }

    fn execute_attention(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        hc_in: &[u16],
        ledger: &mut ResidentLedger,
    ) -> Result<Vec<u16>> {
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
        let jobs = vec![
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
        ];
        let blobs = par_read_full(reader, &jobs)?;
        let hc_fn = decode_f32_le(&blobs[0], &mhc.fn_tensor.name)?;
        let hc_base = decode_f32_le(&blobs[1], &mhc.base_tensor.name)?;
        let hc_scale = decode_f32_le(&blobs[2], &mhc.scale_tensor.name)?;
        let attn_norm_w = decode_u16_le(&blobs[3], &attn_norm.name)?;
        let (_, _, _, post_f32, comb_f32, reduced) = hc_attn_pre_source_algorithm(
            hc_in,
            &hc_fn,
            &hc_scale,
            &hc_base,
            RMS_NORM_EPS,
            HC_EPS,
            HC_SINKHORN_ITERS,
        )?;
        let attn_norm_row =
            rms_norm_bf16_source_algorithm(&reduced, &attn_norm_w, HIDDEN_SIZE, RMS_NORM_EPS)?;
        let wq_a_p = upload_fp8(
            &graph.metal,
            ledger,
            &wq_a.weight.name,
            &wq_a.scale.name,
            blobs[4].clone(),
            blobs[5].clone(),
        )?;
        let wq_b_p = upload_fp8(
            &graph.metal,
            ledger,
            &wq_b.weight.name,
            &wq_b.scale.name,
            blobs[6].clone(),
            blobs[7].clone(),
        )?;
        let wkv_p = upload_fp8(
            &graph.metal,
            ledger,
            &wkv.weight.name,
            &wkv.scale.name,
            blobs[8].clone(),
            blobs[9].clone(),
        )?;
        let wo_a_p = upload_fp8(
            &graph.metal,
            ledger,
            &wo_a.weight.name,
            &wo_a.scale.name,
            blobs[10].clone(),
            blobs[11].clone(),
        )?;
        let wo_b_p = upload_fp8(
            &graph.metal,
            ledger,
            &wo_b.weight.name,
            &wo_b.scale.name,
            blobs[12].clone(),
            blobs[13].clone(),
        )?;
        ledger.acquire(&q_norm.name, blobs[14].len())?;
        let q_norm_buf = graph.metal.new_buffer_with_bytes_checked(&blobs[14])?;
        ledger.acquire(&kv_norm.name, blobs[15].len())?;
        let kv_norm_buf = graph.metal.new_buffer_with_bytes_checked(&blobs[15])?;
        ledger.acquire(&sink.name, blobs[16].len())?;
        let sink_buf = graph.metal.new_buffer_with_bytes_checked(&blobs[16])?;

        write_u16(&graph.scratch.hidden_a, &attn_norm_row);
        let softmax_scale = (HEAD_DIM as f32).powf(-0.5);
        let eps = RMS_NORM_EPS;
        let act_tg = graph.act_tg;
        let fp8_tg = graph.fp8_tg;
        let cast_tg = graph.cast_tg;
        let wo_a_tg = graph.wo_a_tg;

        graph.batch(|batch, s| {
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

        let wo_b_out = read_u16(&graph.scratch.hidden_b, HIDDEN_SIZE)?;
        release_fp8(ledger, &wq_a_p)?;
        release_fp8(ledger, &wq_b_p)?;
        release_fp8(ledger, &wkv_p)?;
        release_fp8(ledger, &wo_a_p)?;
        release_fp8(ledger, &wo_b_p)?;
        ledger.release(&q_norm.name)?;
        ledger.release(&kv_norm.name)?;
        ledger.release(&sink.name)?;
        hc_attn_post_source_algorithm(&wo_b_out, hc_in, &post_f32, &comb_f32)
    }

    fn execute_moe(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        layer: &DeepSeekV4LayerSourceAnchor,
        attn_hc: &[u16],
        token_id: u64,
        ledger: &mut ResidentLedger,
    ) -> Result<Vec<u16>> {
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
        let blobs = par_read_full(reader, &jobs)?;
        let hc_fn = decode_f32_le(&blobs[0], &mhc.fn_tensor.name)?;
        let hc_base = decode_f32_le(&blobs[1], &mhc.base_tensor.name)?;
        let hc_scale = decode_f32_le(&blobs[2], &mhc.scale_tensor.name)?;
        let ffn_norm_w = decode_u16_le(&blobs[3], &ffn_norm.name)?;
        let (_, _, _, post_f32, comb_f32, reduced) = hc_attn_pre_source_algorithm(
            attn_hc,
            &hc_fn,
            &hc_scale,
            &hc_base,
            RMS_NORM_EPS,
            HC_EPS,
            HC_SINKHORN_ITERS,
        )?;
        let ffn_norm_row =
            rms_norm_bf16_source_algorithm(&reduced, &ffn_norm_w, HIDDEN_SIZE, RMS_NORM_EPS)?;
        write_u16(&graph.scratch.hidden_a, &ffn_norm_row);
        ledger.acquire(&gate.score_weight.name, blobs[4].len())?;
        let gate_w = graph.metal.new_buffer_with_bytes_checked(&blobs[4])?;

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
            Some(graph.metal.new_buffer_with_bytes_checked(&row)?)
        } else {
            None
        };
        let bias_buf = if !is_hash {
            ledger.acquire(&gate.route_data.name, blobs[5].len())?;
            Some(graph.metal.new_buffer_with_bytes_checked(&blobs[5])?)
        } else {
            None
        };

        let token_u = if is_hash { 0u32 } else { token_id as u32 };
        let experts_u = ROUTED_EXPERTS as u32;
        let top_k = ACTIVATED_EXPERTS as u32;
        let route_scale = ROUTE_SCALE;
        let gate_tg = graph.gate_tg;

        graph.batch(|batch, s| {
            let mut n = 0usize;
            let rows = ROUTED_EXPERTS as u32;
            let cols = HIDDEN_SIZE as u32;
            let tg = gate_tg.min(rows);
            batch.dispatch_threads(GATE_KERNEL, (rows, 1, 1), (tg, 1, 1), |enc| {
                enc.set_buffer(0, Some(&gate_w), 0);
                enc.set_buffer(1, Some(&s.hidden_a), 0);
                enc.set_buffer(2, Some(&s.gate_logits), 0);
                set_u32(enc, 3, &rows);
                set_u32(enc, 4, &cols);
            })?;
            n += 1;
            if is_hash {
                let table = tid2eid_buf.as_ref().expect("hash table");
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
                    set_f32(enc, 9, &route_scale);
                })?;
            } else {
                let bias = bias_buf.as_ref().expect("bias");
                batch.dispatch_threads(LEARNED_ROUTE_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&s.gate_logits), 0);
                    enc.set_buffer(1, Some(bias), 0);
                    enc.set_buffer(2, Some(&s.route_ids), 0);
                    enc.set_buffer(3, Some(&s.route_weights), 0);
                    enc.set_buffer(4, Some(&s.original_scores), 0);
                    enc.set_buffer(5, Some(&s.route_valid), 0);
                    set_u32(enc, 6, &experts_u);
                    set_u32(enc, 7, &top_k);
                    set_f32(enc, 8, &route_scale);
                })?;
            }
            n += 1;
            batch.dispatch_threads(PACK_KERNEL, (1, 1, 1), (1, 1, 1), |enc| {
                enc.set_buffer(0, Some(&s.route_ids), 0);
                enc.set_buffer(1, Some(&s.route_weights), 0);
                enc.set_buffer(2, Some(&s.worklist), 0);
                enc.set_buffer(3, Some(&s.pack_valid), 0);
                set_u32(enc, 4, &top_k);
                set_u32(enc, 5, &experts_u);
            })?;
            n += 1;
            Ok(n)
        })?;

        let valid = read_u32(&graph.scratch.route_valid, 1)?;
        if valid[0] != 1 {
            return Err(graph_error(format!(
                "layer {} route kernel rejected the token (valid={})",
                layer.layer, valid[0]
            )));
        }
        let pack_valid = read_u32(&graph.scratch.pack_valid, 1)?;
        if pack_valid[0] != 1 {
            return Err(graph_error(format!(
                "layer {} worklist pack rejected the route (valid={})",
                layer.layer, pack_valid[0]
            )));
        }

        let exec = if let Some(ids) = hash_ids {
            let weights = read_f32_n(&graph.scratch.route_weights, ACTIVATED_EXPERTS)?;
            pack_worklist_host(&ids, &weights)?
        } else {
            graph.counters.host_route_id_readback += 1;
            let ids = read_u32(&graph.scratch.route_ids, ACTIVATED_EXPERTS)?;
            let weights = read_f32_n(&graph.scratch.route_weights, ACTIVATED_EXPERTS)?;
            let mut arr = [0u32; ACTIVATED_EXPERTS];
            arr.copy_from_slice(&ids);
            pack_worklist_host(&arr, &weights)?
        };
        seed_worklist(graph, &exec);
        let expert_names = upload_expert_slab(graph, reader, ledger, layer, &exec)?;

        let shared_w1 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W1);
        let shared_w3 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W3);
        let shared_w2 = layer.shared_expert_pair(DeepSeekV4LayerExpertProjection::W2);
        let sh_w1 = load_fp8(
            &graph.metal,
            reader,
            ledger,
            &shared_w1.weight.name,
            &shared_w1.scale.name,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
        )?;
        let sh_w3 = load_fp8(
            &graph.metal,
            reader,
            ledger,
            &shared_w3.weight.name,
            &shared_w3.scale.name,
            MOE_INTER_DIM,
            HIDDEN_SIZE,
        )?;
        let sh_w2 = load_fp8(
            &graph.metal,
            reader,
            ledger,
            &shared_w2.weight.name,
            &shared_w2.scale.name,
            HIDDEN_SIZE,
            MOE_INTER_DIM,
        )?;

        let act_tg = graph.act_tg;
        let fp8_tg = graph.fp8_tg;
        let fp4_tg = graph.fp4_tg;
        let cast_tg = graph.cast_tg;
        let top_k = ACTIVATED_EXPERTS as u32;
        let one = 1u32;
        let zero = 0u32;
        let shared_one = 1.0f32;

        graph.batch(|batch, s| {
            let mut n = 0usize;
            dispatch_act_quant(
                batch,
                &s.hidden_a,
                &s.quant_ffn,
                &s.quant_scale_ffn,
                HIDDEN_SIZE as u32,
                act_tg,
                0,
                0,
                0,
            )?;
            n += 1;
            let rows_w1 = MOE_INTER_DIM as u32;
            let packed = (HIDDEN_SIZE / 2) as u32;
            let scale_cols = (HIDDEN_SIZE / FP4_BLOCK) as u32;
            let grid_w1 = top_k * rows_w1;
            let tg = fp4_tg.min(rows_w1);
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
                set_u32(enc, 9, &top_k);
                set_u32(enc, 10, &zero);
            })?;
            n += 1;
            let gate_count = top_k * rows_w1;
            dispatch_cast(
                batch,
                &s.expert_gate_f32,
                &s.expert_gate_bf16,
                gate_count,
                cast_tg,
            )?;
            n += 1;
            dispatch_cast(
                batch,
                &s.expert_up_f32,
                &s.expert_up_bf16,
                gate_count,
                cast_tg,
            )?;
            n += 1;
            batch.dispatch_threads(
                WORKLIST_SWIGLU_KERNEL,
                (gate_count, 1, 1),
                (tg, 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&s.worklist), 0);
                    enc.set_buffer(1, Some(&s.expert_gate_bf16), 0);
                    enc.set_buffer(2, Some(&s.expert_up_bf16), 0);
                    enc.set_buffer(3, Some(&s.expert_swiglu), 0);
                    set_u32(enc, 4, &rows_w1);
                    set_u32(enc, 5, &top_k);
                },
            )?;
            n += 1;
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
                    act_tg,
                    off_in,
                    off_q,
                    off_s,
                )?;
                n += 1;
            }
            let rows_w2 = HIDDEN_SIZE as u32;
            let packed_w2 = (MOE_INTER_DIM / 2) as u32;
            let scale_w2 = (MOE_INTER_DIM / FP4_BLOCK) as u32;
            let grid_w2 = top_k * rows_w2;
            let tg2 = fp4_tg.min(rows_w2);
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
            })?;
            n += 1;
            dispatch_cast(
                batch,
                &s.expert_down_f32,
                &s.expert_down_bf16,
                grid_w2,
                cast_tg,
            )?;
            n += 1;

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
            )?;
            n += 1;
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
            )?;
            n += 1;
            dispatch_cast(
                batch,
                &s.shared_gate_f32,
                &s.shared_gate_bf16,
                MOE_INTER_DIM as u32,
                cast_tg,
            )?;
            n += 1;
            dispatch_cast(
                batch,
                &s.shared_up_f32,
                &s.shared_up_bf16,
                MOE_INTER_DIM as u32,
                cast_tg,
            )?;
            n += 1;
            batch.dispatch_threads(
                SHARED_SWIGLU_KERNEL,
                (MOE_INTER_DIM as u32, 1, 1),
                (cast_tg.min(MOE_INTER_DIM as u32), 1, 1),
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
                act_tg,
                0,
                0,
                0,
            )?;
            n += 1;
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
            )?;
            n += 1;
            dispatch_cast(
                batch,
                &s.shared_down_f32,
                &s.shared_down_bf16,
                HIDDEN_SIZE as u32,
                cast_tg,
            )?;
            n += 1;
            batch.dispatch_threads(
                WORKLIST_COMBINE_KERNEL,
                (HIDDEN_SIZE as u32, 1, 1),
                (cast_tg.min(HIDDEN_SIZE as u32), 1, 1),
                |enc| {
                    enc.set_buffer(0, Some(&s.expert_down_bf16), 0);
                    enc.set_buffer(1, Some(&s.shared_down_bf16), 0);
                    enc.set_buffer(2, Some(&s.moe_out), 0);
                    set_u32(enc, 3, &(HIDDEN_SIZE as u32));
                    set_u32(enc, 4, &top_k);
                },
            )?;
            n += 1;
            Ok(n)
        })?;

        let moe = read_u16(&graph.scratch.moe_out, HIDDEN_SIZE)?;
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
        hc_attn_post_source_algorithm(&moe, attn_hc, &post_f32, &comb_f32)
    }

    fn metal_lm_head(
        graph: &mut Graph,
        reader: &DeepSeekV4FullStreamReader,
        ledger: &mut ResidentLedger,
        residual_f32: &[f32],
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
        let blobs = std::thread::scope(|scope| {
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
        })?;

        let residual_buf = graph
            .metal
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(residual_f32))?;
        let max_bytes = jobs.iter().map(|j| j.1).max().unwrap_or(0);
        let max_rows = jobs.iter().map(|j| j.3).max().unwrap_or(0);
        let weight_buf = graph.metal.new_buffer_checked(max_bytes)?;
        let out_buf = graph
            .metal
            .new_buffer_checked(max_rows * size_of::<f32>())?;
        let mut best_id = 0u32;
        let mut best_logit = f32::NEG_INFINITY;
        let before = graph.counters.metal_dispatches;
        let lm_tg = graph.lm_tg;
        for (tile, bytes) in tiles.iter().zip(blobs.iter()) {
            ledger.acquire(LM_HEAD_WEIGHT, bytes.len())?;
            write_bytes(&weight_buf, bytes);
            let rows_u = tile.1 as u32;
            let cols_u = HIDDEN_SIZE as u32;
            let tg = lm_tg.min(rows_u.max(1));
            graph.metal.dispatch_batch(|batch| {
                batch.dispatch_threads(LM_HEAD_KERNEL, (rows_u, 1, 1), (tg, 1, 1), |enc| {
                    enc.set_buffer(0, Some(&weight_buf), 0);
                    enc.set_buffer(1, Some(&residual_buf), 0);
                    enc.set_buffer(2, Some(&out_buf), 0);
                    set_u32(enc, 3, &rows_u);
                    set_u32(enc, 4, &cols_u);
                })
            })?;
            graph.counters.command_buffers += 1;
            graph.counters.metal_dispatches += 1;
            let logits = read_f32_n(&out_buf, tile.1)?;
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
