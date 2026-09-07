//! Reusable macOS-only P6 device graph for the bounded DeepSeek-V4 MoE body.
//!
//! This module extracts the source-bound graph exercised by the sealed P6A
//! component receipt into a P7-compatible library surface. Its Gate dispatch
//! uses the separately admitted P0 C4 SIMDgroup reduction candidate; all
//! subsequent route/MoE operations retain their existing P6A authority
//! kernels and order. It is deliberately not an Engine, causal loop, endpoint,
//! parity receipt, or TPS path. It owns only static source controls and device
//! intermediates; its caller owns the `MetalContext`, BF16 predecessor buffer,
//! and returned MoE/route buffers.
//!
//! Gate modes:
//! - **Hash** (`tid2eid`, layers 0..2): experts are known from the token id
//!   before activation; all six bundles are resident at prepare time and the
//!   graph runs in two command buffers with no mid-graph host route readback.
//! - **Learned-bias** (layers 3..42): experts are activation-dependent. The
//!   graph is **two-phase**: (1) Gate + learned route on device, CPU-visible
//!   wait to read the six selected IDs, load those expert bundles from the
//!   sealed stream; (2) expert body + combine. Route weights stay on device;
//!   only the selected expert *IDs* cross the host boundary for residency.
//!   When the existing single-CB candidate is enabled, the post-residency
//!   body is encoded as one command buffer; the route/readback boundary stays
//!   explicit.
//!
//! This is not an Engine, causal loop, endpoint, parity receipt, or TPS path.

use std::collections::BTreeMap;
use std::mem::size_of;
use std::path::PathBuf;

use sha2::{Digest, Sha256};

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePair, NativeScalePairKind,
};
use crate::gravity_deepseek_v4_act_quant::ACT_QUANT_BLOCK;
use crate::gravity_deepseek_v4_expert_cache::{
    CachedExpertBundle, DeepSeekV4ExpertBundleCache, ExpertBundleKey, ExpertOperator,
};
use crate::gravity_deepseek_v4_layer0_moe::{
    ACTIVATED_EXPERTS, MOE_INTER_DIM, ROUTED_EXPERTS, ROUTE_SCALE,
};
use crate::gravity_deepseek_v4_layer0_prefix::HIDDEN_SIZE;
use crate::gravity_deepseek_v4_layer_plan::DeepSeekV4LayerDeviceCatalog;
use crate::gravity_deepseek_v4_layer_source_anchors::DeepSeekV4LayerGateMode;
use crate::gravity_deepseek_v4_p7_composition::{
    DeepSeekV4P7FfnSourceContract, DeepSeekV4P7P6DeviceExecutor, DeepSeekV4P7P6DeviceInput,
    DeepSeekV4P7P6DeviceOutput, DSV4F_P7_FFN_NORM_BF16_BYTES,
};
#[cfg(test)]
use crate::gravity_deepseek_v4_p7_composition::{
    DSV4F_P7_GATE_LOGITS_F32_BYTES, DSV4F_P7_ROUTE_VALID_U32_BYTES,
};
use crate::metal::{CommandBatch, MetalContext, TokenPipelineCache};
use crate::{Error, Result};

const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const ACT_QUANT_SIMD_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_simdgroup_block_candidate";
const ACT_QUANT_SIMD_WIDTH: u32 = 32;
const ACT_QUANT_SIMD_VECTOR_WIDTH: u32 = 4;
/// Fixed P6 down-wave quantizer: six routed tensors plus the shared tensor.
/// It preserves the scalar authority block grammar while packing the seven
/// independent resource bindings behind one dispatch.
pub const P6_BATCHED_DOWN_QAT_ENV: &str = "HAWKING_DSV4F_P6_BATCHED_DOWN_QAT";
const P6_BATCHED_DOWN_QAT_KERNEL: &str = "deepseek_v4_p6a_act_quant_bf16_ue8m0_fixed7_authority";
const P6_BATCHED_DOWN_QAT_TENSORS: u32 = 7;
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const FP8_SIMD_KERNEL: &str =
    "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_simdgroup_v4_splitk_candidate";
// The candidate owns one output row per threadgroup. Eight SIMDgroups split
// the source's 32 activation blocks, matching the candidate shader's bounded
// partial array while retaining the P6 authority's 256-thread geometry.
const FP8_SIMD_THREADS_X: u32 = 256;
const FP8_SIMD_ROWS_PER_TG: u32 = 1;
const BF16_CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const P5B_FP4_KERNEL: &str = "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority";
const FP4_SIMD_KERNEL: &str = "deepseek_v4_fp4_e2m1fn_x2_e8m0_matvec_simdgroup_v4_splitk_candidate";
const FP4_SIMD_THREADS_X: u32 = 64;
const FP4_SIMD_ROWS_PER_TG: u32 = 4;
const P6_FP4_GATE_UP_SWIGLU_KERNEL: &str =
    "deepseek_v4_p6a_fp4_gate_up_swiglu_route_weight_buffer_bf16_fused_authority";
const P6_FP4_GATE_UP_SWIGLU_SIMD_KERNEL: &str =
    "deepseek_v4_p6a_fp4_gate_up_swiglu_route_weight_buffer_bf16_fused_simd_candidate";
const P6_FP4_GATE_UP_SWIGLU_SIMD_THREADS_X: u32 = 256;
const P6_FP4_GATE_UP_SWIGLU_SIMD_ROWS_PER_TG: u32 = 8;
const P6_FP4_DOWN_BF16_KERNEL: &str = "deepseek_v4_p6a_fp4_down_bf16_fused_authority";
const P6_FP4_DOWN_BF16_SIMD_KERNEL: &str = "deepseek_v4_p6a_fp4_down_bf16_fused_simd_candidate";
const P6_FP4_DOWN_BF16_SIMD_THREADS_X: u32 = 256;
const P6_FP4_DOWN_BF16_SIMD_ROWS_PER_TG: u32 = 8;
/// Full downstream candidate: routed FP4 W2, shared FP8 W2, and final
/// combine share one dependent launch. It is meaningful only together with
/// the routed down-fusion records, and supersedes the separate shared-down
/// fusion while enabled.
pub const P6_FP4_DOWN_SHARED_COMBINE_FUSED_ENV: &str =
    "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED";
const P6_FP4_DOWN_SHARED_COMBINE_FUSED_KERNEL: &str =
    "deepseek_v4_p6a_fp4_down_fp8_shared_combine_bf16_fused_authority";
/// Shared with the native graph: the source-order FP8 W1/W3 reductions,
/// explicit BF16 round-trips, and clamped SwiGLU are one guarded primitive.
pub const P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_ENV: &str =
    "HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED";
const P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_KERNEL: &str = "deepseek_v4_fp8_gate_up_swiglu_bf16_fused";
/// Opt-in shared FP8 W2/BF16/combine fusion. The routed outputs have already
/// crossed their BF16 boundary, so this one launch preserves the source
/// combine order while removing shared W2 staging and the standalone combine.
pub const P6_SHARED_FP8_DOWN_COMBINE_FUSED_ENV: &str =
    "HAWKING_DSV4F_FP8_SHARED_DOWN_COMBINE_FUSED";
const P6_SHARED_FP8_DOWN_COMBINE_FUSED_KERNEL: &str = "deepseek_v4_fp8_down_bf16_combine_fused";
const P5B_SWIGLU_KERNEL: &str = "deepseek_v4_p5b_swiglu_route_bf16_authority";
// Admitted by the isolated P0 Gate-reduction sweep only. The C4 kernel maps
// one 32-thread SIMDgroup to each Gate row and must remain a single P6 Gate
// dispatch/encoder within the existing first command buffer.
/// Metadata-only identity of the admitted P0 C4 Gate candidate.
pub const P6_C4_GATE_KERNEL: &str = "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate";
/// Exact C4 threadgroup geometry; not a runtime throughput claim.
pub const P6_C4_GATE_SIMDGROUP_THREADS: u32 = 32;
/// Exact one-token C4 grid geometry; not a runtime throughput claim.
pub const P6_C4_GATE_GRID_THREADS: u32 = ROUTED_EXPERTS as u32 * P6_C4_GATE_SIMDGROUP_THREADS;
const P6A_ROUTE_KERNEL: &str = "deepseek_v4_p6a_hash_route_sqrtsoftplus_authority";
/// Admitted learned-bias route kernel (exact top-k IDs vs F64 oracle sealed).
pub const P6A_LEARNED_ROUTE_KERNEL: &str =
    "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority";
const P6A_SWIGLU_KERNEL: &str = "deepseek_v4_p6a_swiglu_route_weight_buffer_bf16_authority";
const P6A_COMBINE_KERNEL: &str = "deepseek_v4_p6a_route6_shared_combine_bf16_authority";
/// Opt-in topology candidate: append the dependent down/combine wave to the
/// same command buffer as gate/up/SwiGLU. `=0` preserves the historical A/B.
pub const P6_SINGLE_CB_ENV: &str = "HAWKING_DSV4F_P6_SINGLE_CB";
/// Opt-in prefix-wave candidate: Gate and activation quantization only read
/// the same input and write disjoint outputs, so they can share one concurrent
/// encoder before the dependent route kernel. `=0` preserves two encoders.
pub const P6_PREFIX_CONCURRENT_ENV: &str = "HAWKING_DSV4F_P6_PREFIX_CONCURRENT";
/// Opt-in source-quantizer candidate; `=0` keeps the one-block authority.
pub const P6_ACT_QUANT_SIMD_ENV: &str = "HAWKING_DSV4F_P6_ACT_QUANT_SIMD";
/// Opt-in routed-expert FP4 split-K candidate; `=0` keeps serial authority.
pub const P6_FP4_SIMD_ENV: &str = "HAWKING_DSV4F_P6_FP4_SIMD";
/// Opt-in shared-expert FP8 split-K candidate; `=0` keeps serial authority.
pub const P6_FP8_SIMD_ENV: &str = "HAWKING_DSV4F_P6_FP8_SIMD";
/// Opt-in routed FP4 gate/up/SwiGLU fusion; `=0` preserves the 30-dispatch
/// routed authority sequence for a matched control.
pub const P6_FP4_GATE_UP_SWIGLU_FUSED_ENV: &str = "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED";
/// Opt-in occupancy sibling for the P6 fused routed gate/up/SwiGLU path.
/// It is sampled only when the fusion switch is enabled.
pub const P6_FP4_GATE_UP_SWIGLU_SIMD_ENV: &str = "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD";
/// Shared resident pipeline-cache switch used by native/fullseq/P6 batches.
pub const P6_PIPELINE_CACHE_ENV: &str = "HAWKING_FLASH_PIPELINE_CACHE_REUSE";
/// Opt-in host-ceremony candidate for learned-bias P6 route changes. `=0`
/// preserves re-admission of the sealed reader on every route change.
pub const P6_LEARNED_READER_REUSE_ENV: &str = "HAWKING_DSV4F_P6_LEARNED_READER_REUSE";
/// Opt-in bounded source-bundle cache candidate for learned-bias route changes.
/// `=0` drops the six-bundle cache after each route load.
pub const P6_LEARNED_EXPERT_CACHE_REUSE_ENV: &str = "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE";
/// Opt-in routed FP4 W2-to-BF16 fusion; `=0` keeps the six W2 and six cast
/// launches as the authority sequence.
pub const P6_FP4_DOWN_BF16_FUSED_ENV: &str = "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED";
/// Opt-in occupancy sibling for the fixed-six routed W2-to-BF16 fusion.
/// It is sampled only when the base down-fusion switch is enabled.
pub const P6_FP4_DOWN_BF16_SIMD_ENV: &str = "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD";

const HIDDEN_BF16_BYTES: usize = HIDDEN_SIZE * size_of::<u16>();
const ROUTE_IDS_BYTES: usize = ACTIVATED_EXPERTS * size_of::<u32>();
const ROUTE_WEIGHTS_BYTES: usize = ACTIVATED_EXPERTS * size_of::<f32>();
const GATE_LOGITS_BYTES: usize = ROUTED_EXPERTS * size_of::<f32>();
const ORIGINAL_SCORES_BYTES: usize = ROUTED_EXPERTS * size_of::<f32>();
const ROUTE_VALID_BYTES: usize = size_of::<u32>();
const GATE_BIAS_BYTES: usize = ROUTED_EXPERTS * size_of::<f32>();
/// A live-path-independent placeholder for a buffer whose authority writer
/// is disabled by a fused candidate.  The pointer remains non-null for the
/// dormant ABI/resource record, but no candidate dispatch may dereference it.
const P6_DORMANT_BUFFER_BYTES: usize = 1;

#[inline]
fn allocate_p6_scratch(metal: &MetalContext, bytes: usize, live: bool) -> Result<metal::Buffer> {
    metal.new_buffer_checked(if live { bytes } else { P6_DORMANT_BUFFER_BYTES })
}

/// Exact fixed topology of one hash-gate `DeepSeekV4Layer0P6MetalExecutor::execute`
/// call with the historical A/B control. These are structural counts for the
/// bounded reusable P6 graph, not a runtime or TPS measurement. Each
/// `dispatch_batch` commits and waits once.
pub const DSV4F_P6_DEVICE_COMMAND_BUFFERS: usize = 2;
pub const DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS: usize = 2;
/// Candidate topology when `HAWKING_DSV4F_P6_PREFIX_CONCURRENT=1` is
/// enabled. Dispatch count and command-buffer count remain unchanged.
pub const DSV4F_P6_PREFIX_CONCURRENT_COMPUTE_ENCODERS: usize = 9;
/// Candidate topology when `HAWKING_DSV4F_P6_SINGLE_CB=1` is enabled.
pub const DSV4F_P6_SINGLE_CB_COMMAND_BUFFERS: usize = 1;
pub const DSV4F_P6_SINGLE_CB_CPU_VISIBLE_WAITS: usize = 1;
pub const DSV4F_P6_DEVICE_DISPATCHES: usize = 60;
pub const DSV4F_P6_DEVICE_COMPUTE_ENCODERS: usize = 10;
/// Candidate topology when routed FP4 gate/up/SwiGLU fusion is enabled.
/// The down/combine wave remains unchanged; only the 29 routed epilogue
/// dispatches collapse into one fixed six-expert launch.
pub const DSV4F_P6_FUSED_GATE_UP_SWIGLU_DISPATCHES: usize = 31;
/// Candidate topology when routed FP4 W2 plus BF16 cast fusion is enabled.
/// The six routed W2 launches and six routed casts become one fixed-six
/// indirect launch; shared W2/cast and the combine remain unchanged.
pub const DSV4F_P6_FUSED_DOWN_BF16_DISPATCHES: usize = 49;
/// Candidate topology when both fixed-six routed epilogues are enabled.
pub const DSV4F_P6_FUSED_GATE_UP_AND_DOWN_DISPATCHES: usize = 20;
/// Candidate topology when the shared FP8 gate/up/SwiGLU epilogue is fused.
/// The shared W1/W3, two casts, and shared SwiGLU become one dispatch.
pub const DSV4F_P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_DISPATCHES: usize = 56;
/// Candidate topology when routed gate/up/SwiGLU and shared FP8 fusion are
/// both enabled; the down/combine wave remains on its authority topology.
pub const DSV4F_P6_FUSED_GATE_UP_AND_SHARED_DISPATCHES: usize = 27;
/// Candidate topology when shared FP8 W2, its BF16 boundary, and the final
/// fixed-six combine become one dispatch.
pub const DSV4F_P6_SHARED_FP8_DOWN_COMBINE_FUSED_DISPATCHES: usize = 58;
/// Candidate topology when routed FP4 W2 and shared FP8 W2/combine are
/// collapsed into one source-order dependent launch.
pub const DSV4F_P6_FUSED_DOWN_SHARED_COMBINE_DISPATCHES: usize = 46;
/// Candidate topology when the six routed and one shared down-QAT launches
/// become one fixed-seven indirect dispatch. The dependent down projection
/// and combine waves are otherwise unchanged.
pub const DSV4F_P6_BATCHED_DOWN_QAT_DISPATCHES: usize = 54;
/// Candidate topology when routed gate/up, shared gate/up, routed down, and
/// shared down/combine fusions and fixed-seven down-QAT batching are enabled
/// together.
pub const DSV4F_P6_FUSED_EPILOGUE_STACK_DISPATCHES: usize = 8;
/// Same composed stack as above, but with the full downstream fusion replacing
/// the two remaining routed/shared down launches.
pub const DSV4F_P6_FUSED_EPILOGUE_STACK_FULL_DOWN_DISPATCHES: usize = 7;
/// Full-down stack encoder count after the fixed-seven QAT producer and its
/// indirect down/combine consumer share one serial dependency encoder.
pub const DSV4F_P6_FUSED_EPILOGUE_STACK_FULL_DOWN_COMPUTE_ENCODERS: usize = 4;

/// Learned-bias two-phase topology: phase-1 (gate+QAT+route), host residency
/// load of the six selected experts, phase-2a (W1/W3/casts/SwiGLU), phase-2b
/// (down-QAT/W2/casts/combine). Same kernel count as hash (60); one extra
/// command buffer and CPU-visible wait for the dynamic expert load boundary.
pub const DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS: usize = 3;
pub const DSV4F_P6_LEARNED_DEVICE_CPU_VISIBLE_WAITS: usize = 3;
pub const DSV4F_P6_LEARNED_DEVICE_DISPATCHES: usize = 60;
/// Learned-bias topology when the existing single-CB candidate is enabled:
/// the route/readback/load boundary remains, while the two post-load device
/// waves share one command buffer and fence.
pub const DSV4F_P6_LEARNED_SINGLE_CB_COMMAND_BUFFERS: usize = 2;
pub const DSV4F_P6_LEARNED_SINGLE_CB_CPU_VISIBLE_WAITS: usize = 2;
/// Host-visible boundary used only to read selected expert IDs for residency.
pub const DSV4F_P6_LEARNED_HOST_ROUTE_ID_READBACK: bool = true;

/// Source-bound P6 selection.  `token_position` is not consumed by the
/// hash-table kernel, but is retained and checked so the P7 handoff cannot
/// apply a token's source control plan to a different decode position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeepSeekV4P6SourceControls {
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
}

impl DeepSeekV4P6SourceControls {
    pub const fn new(layer: usize, token_id: u32, token_position: usize) -> Self {
        Self {
            layer,
            token_id,
            token_position,
        }
    }
}

/// Metadata-only source binding for one resident expert.  The raw source
/// bytes are uploaded during preparation and are never exposed by this API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P6ResidentExpertBinding {
    pub source_top_slot: u32,
    pub expert_id: u32,
    pub w1_weight_name: String,
    pub w3_weight_name: String,
    pub w2_weight_name: String,
}

/// Gate route data bound for one P6 prepare. Hash layers upload the full
/// `tid2eid` table; learned-bias layers upload `gate.bias` and resolve experts
/// only after the on-device route.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeepSeekV4P6GateRouteBinding {
    HashTid2Eid {
        tid2eid_name: String,
        tid2eid_sha256: String,
        /// Known before activation from the token-id row.
        selected_expert_ids_top_slot_order: [u32; ACTIVATED_EXPERTS],
    },
    LearnedBias {
        bias_name: String,
        bias_sha256: String,
        /// True: execute will read selected IDs on the host for residency only.
        host_route_id_readback_for_residency: bool,
    },
}

/// Immutable source binding held by the reusable executor.  It confirms that
/// static Gate/route/expert controls were selected from the admitted Gravity
/// stream before their direct device upload; it contains no hidden-state data
/// and no host-computed route weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4P6SourceBindings {
    pub artifact_manifest_seal_sha256: String,
    pub layer: usize,
    pub token_id: u32,
    pub token_position: usize,
    pub gate_mode: DeepSeekV4LayerGateMode,
    pub gate_weight_name: String,
    pub gate_weight_sha256: String,
    pub route: DeepSeekV4P6GateRouteBinding,
    /// For hash: filled at prepare. For learned: placeholders until execute
    /// resolves dynamic top-k (names may be empty pre-execute).
    pub selected_expert_ids_top_slot_order: [u32; ACTIVATED_EXPERTS],
    pub resident_experts_numeric_source_order:
        [DeepSeekV4P6ResidentExpertBinding; ACTIVATED_EXPERTS],
    pub shared_w1_weight_name: String,
    pub shared_w3_weight_name: String,
    pub shared_w2_weight_name: String,
    pub source_parent_retained: bool,
    pub host_activation_handoff_permitted: bool,
    pub host_route_weight_handoff_permitted: bool,
    /// Learned two-phase only: host reads selected expert IDs for cache fill.
    pub host_route_id_readback_for_residency: bool,
}

#[derive(Clone, Copy)]
struct PairGeometry {
    kind: NativeScalePairKind,
    rows: usize,
    logical_k: usize,
    packed_k: usize,
    scale_rows: usize,
    scale_cols: usize,
}

const FP4_W1_W3: PairGeometry = PairGeometry {
    kind: NativeScalePairKind::Fp4E2M1fnX2,
    rows: MOE_INTER_DIM,
    logical_k: HIDDEN_SIZE,
    packed_k: HIDDEN_SIZE / 2,
    scale_rows: MOE_INTER_DIM,
    scale_cols: HIDDEN_SIZE / 32,
};
const FP4_W2: PairGeometry = PairGeometry {
    kind: NativeScalePairKind::Fp4E2M1fnX2,
    rows: HIDDEN_SIZE,
    logical_k: MOE_INTER_DIM,
    packed_k: MOE_INTER_DIM / 2,
    scale_rows: HIDDEN_SIZE,
    scale_cols: MOE_INTER_DIM / 32,
};
const FP8_W1_W3: PairGeometry = PairGeometry {
    kind: NativeScalePairKind::Fp8E4M3fn,
    rows: MOE_INTER_DIM,
    logical_k: HIDDEN_SIZE,
    packed_k: HIDDEN_SIZE,
    scale_rows: MOE_INTER_DIM / ACT_QUANT_BLOCK,
    scale_cols: HIDDEN_SIZE / ACT_QUANT_BLOCK,
};
const FP8_W2: PairGeometry = PairGeometry {
    kind: NativeScalePairKind::Fp8E4M3fn,
    rows: HIDDEN_SIZE,
    logical_k: MOE_INTER_DIM,
    packed_k: MOE_INTER_DIM,
    scale_rows: HIDDEN_SIZE / ACT_QUANT_BLOCK,
    scale_cols: MOE_INTER_DIM / ACT_QUANT_BLOCK,
};

#[derive(Clone)]
struct NativeFp4Gpu {
    weight: metal::Buffer,
    scale: metal::Buffer,
    rows: u32,
    packed_k: u32,
    scale_cols: u32,
}

/// One source-sealed routed expert's uploaded FP4 weight triplet.  This is a
/// bounded GPU residency cache, not a second decoded model representation:
/// the executor keeps at most the six expert IDs in the current route and
/// reuses their existing Metal buffers when a later learned route overlaps.
#[derive(Clone)]
struct CachedP6ExpertGpu {
    w1: NativeFp4Gpu,
    w3: NativeFp4Gpu,
    w2: NativeFp4Gpu,
    w1_name: String,
    w3_name: String,
    w2_name: String,
}

#[inline]
fn retain_selected_expert_ids<T>(
    cache: &mut BTreeMap<u32, T>,
    selected_top_slot: &[u32; ACTIVATED_EXPERTS],
) {
    cache.retain(|expert_id, _| selected_top_slot.contains(expert_id));
    debug_assert!(cache.len() <= ACTIVATED_EXPERTS);
}

struct NativeFp8Gpu {
    weight: metal::Buffer,
    scale: metal::Buffer,
    rows: u32,
    logical_k: u32,
    scale_cols: u32,
}

/// Indirect fixed-slot ABI for the opt-in P6 routed gate/up/SwiGLU fusion.
/// The output pointer keeps the existing per-expert scratch buffers and the
/// route slot preserves source top-k weighting after numeric expert ordering.
#[repr(C)]
#[derive(Clone, Copy)]
struct P6Fp4GateUpRef {
    gate_weights: u64,
    gate_scales: u64,
    up_weights: u64,
    up_scales: u64,
    output_bf16: u64,
    route_slot: u32,
    ready: u32,
}

const _: () = assert!(size_of::<P6Fp4GateUpRef>() == 48);

/// Indirect fixed-slot ABI for the opt-in P6 routed FP4 W2-to-BF16 fusion.
/// Each slot carries its own QAT output/scales because routed SwiGLU inputs
/// differ by expert; the output remains the existing per-expert BF16 buffer.
#[repr(C)]
#[derive(Clone, Copy)]
struct P6Fp4DownRef {
    weights: u64,
    weight_scales: u64,
    activations: u64,
    activation_scales: u64,
    output_bf16: u64,
    ready: u32,
    reserved: u32,
}

const _: () = assert!(size_of::<P6Fp4DownRef>() == 48);

/// Fixed-seven indirect ABI for the P6 down activation-quantization wave.
/// The pointer records are source scratch buffers only; no weight payload is
/// retained through this candidate.
#[repr(C)]
#[derive(Clone, Copy)]
struct P6ActQuantRef {
    input_bf16: u64,
    quantized: u64,
    act_scales: u64,
    ready: u32,
    reserved: u32,
}

const _: () = assert!(size_of::<P6ActQuantRef>() == 32);

struct RoutedExpertGpu {
    source_top_slot: u32,
    w1: NativeFp4Gpu,
    w3: NativeFp4Gpu,
    w2: NativeFp4Gpu,
    gate_f32: metal::Buffer,
    up_f32: metal::Buffer,
    gate_bf16: metal::Buffer,
    up_bf16: metal::Buffer,
    swiglu_bf16: metal::Buffer,
    down_quant: metal::Buffer,
    down_scales: metal::Buffer,
    down_f32: metal::Buffer,
    down_bf16: metal::Buffer,
}

struct SharedExpertGpu {
    w1: NativeFp8Gpu,
    w3: NativeFp8Gpu,
    w2: NativeFp8Gpu,
    gate_f32: metal::Buffer,
    up_f32: metal::Buffer,
    gate_bf16: metal::Buffer,
    up_bf16: metal::Buffer,
    swiglu_bf16: metal::Buffer,
    down_quant: metal::Buffer,
    down_scales: metal::Buffer,
    down_f32: metal::Buffer,
    down_bf16: metal::Buffer,
}

#[derive(Clone, Copy)]
struct ThreadGeometry {
    qat: u32,
    qat_kernel: &'static str,
    qat_vector_width: u32,
    batched_down_qat: bool,
    batched_down_qat_threads: u32,
    fp4: u32,
    fp4_kernel: &'static str,
    fp4_threads_x: u32,
    fp4_rows_per_tg: u32,
    fp8: u32,
    fp8_kernel: &'static str,
    fp8_threads_x: u32,
    fp8_rows_per_tg: u32,
    cast: u32,
    gate: u32,
    route: u32,
    routed_swiglu: u32,
    shared_swiglu: u32,
    shared_fp8_gate_up_swiglu: bool,
    shared_fp8_gate_up_swiglu_threads: u32,
    shared_fp8_down_combine: bool,
    shared_fp8_down_combine_threads: u32,
    combine: u32,
    fused_gate_up_swiglu: bool,
    fused_gate_up_swiglu_kernel: &'static str,
    fused_gate_up_swiglu_threads: u32,
    fused_gate_up_swiglu_threads_x: u32,
    fused_gate_up_swiglu_rows_per_tg: u32,
    fused_down_bf16: bool,
    fused_down_bf16_kernel: &'static str,
    fused_down_bf16_threads: u32,
    fused_down_bf16_threads_x: u32,
    fused_down_bf16_rows_per_tg: u32,
    fused_down_shared_combine: bool,
    fused_down_shared_combine_threads: u32,
}

/// Which on-device route kernel + host residency policy this executor uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum P6RouteMode {
    Hash,
    Learned,
}

/// Reusable P6 resource graph.  Static weights and all execution
/// intermediates are device-resident.  It intentionally does not retain a
/// `MetalContext`: both preparation and execution use a context owned by the
/// caller, and a command-queue identity check prevents cross-context use.
///
/// For learned-bias mode, `artifact_root` is retained so execute can admit
/// the sealed reader and load the six dynamically selected expert bundles.
pub struct DeepSeekV4Layer0P6MetalExecutor {
    controls: DeepSeekV4P6SourceControls,
    bindings: DeepSeekV4P6SourceBindings,
    route_mode: P6RouteMode,
    /// Present only for learned-bias two-phase residency loads.
    artifact_root: Option<PathBuf>,
    /// Optional metadata-only reader retained across learned route changes.
    /// Reuse removes repeated manifest/index admission after the immutable
    /// seal is checked; the separate bounded GPU cache below owns any weight
    /// buffer reuse.
    learned_reader: Option<DeepSeekV4FullStreamReader>,
    learned_reader_reuse: bool,
    /// Optional bounded CPU source cache retained across learned route
    /// changes. It is independent from metadata-reader reuse and remains
    /// source-native; GPU buffer reuse is tracked separately below.
    learned_expert_cache: Option<DeepSeekV4ExpertBundleCache>,
    /// Optional bounded GPU weight cache retained across learned route
    /// changes. It is keyed by source expert ID and pruned to the current
    /// six-ID route before any new upload, so source-cache reuse does not
    /// silently turn into an all-experts GPU residency commitment.
    learned_gpu_expert_cache: Option<BTreeMap<u32, CachedP6ExpertGpu>>,
    learned_expert_cache_reuse: bool,
    context_queue_identity: usize,
    /// Snapshot the command-buffer A/B at preparation so the hot path does
    /// not re-read process environment state for every token.
    single_command_buffer: bool,
    /// Snapshot whether the independent Gate/QAT prefix shares one concurrent
    /// encoder before the route dependency boundary.
    prefix_concurrent: bool,
    threads: ThreadGeometry,
    gate_weight: metal::Buffer,
    /// Hash: full tid2eid table. Learned: gate.bias F32[256].
    route_table: metal::Buffer,
    input_quant: metal::Buffer,
    input_scales: metal::Buffer,
    gate_logits: metal::Buffer,
    route_scores: metal::Buffer,
    route_valid: metal::Buffer,
    routed: [RoutedExpertGpu; ACTIVATED_EXPERTS],
    /// Learned: false until the first execute fills weight buffers.
    experts_loaded: bool,
    shared: SharedExpertGpu,
    /// Resident pipeline handles across this P6 executor's adjacent batches.
    /// The `=0` control restores a fresh per-batch cache for an independent A/B.
    pipeline_cache: TokenPipelineCache,
    pipeline_cache_reuse: bool,
    /// Fixed six-slot indirect records for the opt-in fused routed epilogue.
    fused_gate_up_refs: Option<metal::Buffer>,
    /// The weight buffers reached through `fused_gate_up_refs`; they must be
    /// declared read-resident because the shader consumes GPU addresses.
    fused_gate_up_weight_resources: Vec<metal::Buffer>,
    /// The per-expert output buffers reached through the same indirect refs.
    fused_gate_up_output_resources: Vec<metal::Buffer>,
    /// Fixed six-slot indirect records for the opt-in routed W2-to-BF16
    /// fusion.
    fused_down_refs: Option<metal::Buffer>,
    fused_down_resources: Vec<metal::Buffer>,
    fused_down_output_resources: Vec<metal::Buffer>,
    /// Fixed seven-slot indirect records for the opt-in down-QAT batching.
    batched_down_qat_refs: Option<metal::Buffer>,
    batched_down_qat_resources: Vec<metal::Buffer>,
}

impl DeepSeekV4Layer0P6MetalExecutor {
    /// Prepare the source-bound graph from an admitted stream and expert cache.
    ///
    /// - **Hash gate**: acquires the six tid2eid-selected expert bundles into
    ///   the hot cache and uploads them before any dispatch.
    /// - **Learned-bias gate**: uploads Gate weight + bias + shared expert
    ///   only; the six routed expert weight slots are empty placeholders and
    ///   are filled during execute after the on-device route (two-phase).
    ///   `cache` may be empty at prepare; execute admits the reader and fills
    ///   a fresh hot cache for the selected IDs. Reader reuse is controlled by
    ///   an explicit A/B switch sampled during preparation.
    pub fn prepare(
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        cache: &mut DeepSeekV4ExpertBundleCache,
        controls: DeepSeekV4P6SourceControls,
    ) -> Result<Self> {
        let catalog = DeepSeekV4LayerDeviceCatalog::admit(reader)?;
        let plan = catalog.plan(controls.layer)?;
        plan.require_moe_device()?;
        match plan.gate_mode {
            DeepSeekV4LayerGateMode::HashTokenIdToExpertIds => {
                Self::prepare_hash(metal, reader, cache, controls, &catalog)
            }
            DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias => {
                Self::prepare_learned(metal, reader, controls, &catalog)
            }
        }
    }

    fn prepare_hash(
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        cache: &mut DeepSeekV4ExpertBundleCache,
        controls: DeepSeekV4P6SourceControls,
        catalog: &DeepSeekV4LayerDeviceCatalog,
    ) -> Result<Self> {
        let (tid2eid_bytes, selected_ids_top_slot_order, execution) =
            source_route_plan(reader, controls)?;
        for &(_, expert_id) in &execution {
            let key = ExpertBundleKey::new(
                u16::try_from(controls.layer)
                    .map_err(|_| p6_error("P6 layer does not fit source cache key"))?,
                u16::try_from(expert_id)
                    .map_err(|_| p6_error("P6 expert ID does not fit source cache key"))?,
            );
            cache.acquire(reader, key)?;
        }
        let cache_state = cache.state();
        for &(_, expert_id) in &execution {
            let key = ExpertBundleKey::new(controls.layer as u16, expert_id as u16);
            if !cache_state.hot_keys_lru_to_mru.contains(&key) {
                return Err(p6_error(format!(
                    "P6 requires all six selected expert bundles resident in the hot cache simultaneously; expert {expert_id} is not hot"
                )));
            }
        }

        let (gate_weight_name, gate_bytes) = load_gate_weight(reader, controls.layer)?;
        let tid2eid_name = catalog
            .plan(controls.layer)?
            .gate_route_data_name(catalog.anchors())?;
        let expected_gate_weight = catalog
            .plan(controls.layer)?
            .gate_score_weight_name(catalog.anchors())?;
        if gate_weight_name != expected_gate_weight
            || tid2eid_name != format!("layers.{}.ffn.gate.tid2eid", controls.layer)
        {
            return Err(p6_error(
                "P6 gate tensor names disagree with the verified layer-source anchors",
            ));
        }

        let mut resident_bindings = Vec::with_capacity(ACTIVATED_EXPERTS);
        let mut routed = Vec::with_capacity(ACTIVATED_EXPERTS);
        for &(source_top_slot, expert_id) in &execution {
            let key = ExpertBundleKey::new(controls.layer as u16, expert_id as u16);
            let bundle = cache.resident(key).ok_or_else(|| {
                p6_error(format!(
                    "P6 expert {expert_id} disappeared from source cache during preparation"
                ))
            })?;
            let (w1, w1_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W1, FP4_W1_W3)?;
            let (w3, w3_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W3, FP4_W1_W3)?;
            let (w2, w2_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W2, FP4_W2)?;
            resident_bindings.push(DeepSeekV4P6ResidentExpertBinding {
                source_top_slot,
                expert_id,
                w1_weight_name: w1_name,
                w3_weight_name: w3_name,
                w2_weight_name: w2_name,
            });
            routed.push(allocate_routed_expert(metal, source_top_slot, w1, w3, w2)?);
        }
        let routed: [RoutedExpertGpu; ACTIVATED_EXPERTS] = routed
            .try_into()
            .map_err(|_| p6_error("P6 did not prepare exactly six routed GPU experts"))?;
        let resident_experts_numeric_source_order: [DeepSeekV4P6ResidentExpertBinding;
            ACTIVATED_EXPERTS] = resident_bindings
            .try_into()
            .map_err(|_| p6_error("P6 resident binding count is not six"))?;

        let (shared, shared_names) = prepare_shared_expert(metal, reader, controls.layer)?;
        let fused_gate_up_swiglu = crate::env_on(P6_FP4_GATE_UP_SWIGLU_FUSED_ENV);
        let fused_gate_up_swiglu_simd =
            fused_gate_up_swiglu && crate::env_on(P6_FP4_GATE_UP_SWIGLU_SIMD_ENV);
        let shared_fp8_gate_up_swiglu = crate::env_on(P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_ENV);
        let fused_down_bf16 = crate::env_on(P6_FP4_DOWN_BF16_FUSED_ENV);
        let fused_down_shared_combine =
            fused_down_bf16 && crate::env_on(P6_FP4_DOWN_SHARED_COMBINE_FUSED_ENV);
        let shared_fp8_down_combine =
            !fused_down_shared_combine && crate::env_on(P6_SHARED_FP8_DOWN_COMBINE_FUSED_ENV);
        let batched_down_qat = crate::env_on(P6_BATCHED_DOWN_QAT_ENV);
        let fused_down_bf16_simd = fused_down_bf16
            && !fused_down_shared_combine
            && crate::env_on(P6_FP4_DOWN_BF16_SIMD_ENV);
        let threads = precompile_threads(
            metal,
            fused_gate_up_swiglu,
            fused_gate_up_swiglu_simd,
            shared_fp8_gate_up_swiglu,
            shared_fp8_down_combine,
            batched_down_qat,
            fused_down_bf16,
            fused_down_bf16_simd,
            fused_down_shared_combine,
        )?;
        let (fused_gate_up_refs, fused_gate_up_weight_resources, fused_gate_up_output_resources) =
            if fused_gate_up_swiglu {
                let (refs, weight_resources, output_resources) =
                    prepare_fused_gate_up_refs(metal, &routed, true)?;
                (Some(refs), weight_resources, output_resources)
            } else {
                (None, Vec::new(), Vec::new())
            };
        let (fused_down_refs, fused_down_resources, fused_down_output_resources) =
            if fused_down_bf16 {
                let (refs, resources, output_resources) =
                    prepare_fused_down_refs(metal, &routed, true)?;
                (Some(refs), resources, output_resources)
            } else {
                (None, Vec::new(), Vec::new())
            };
        let (batched_down_qat_refs, batched_down_qat_resources) = if batched_down_qat {
            let (refs, resources) = prepare_batched_down_qat_refs(metal, &routed, &shared)?;
            (Some(refs), resources)
        } else {
            (None, Vec::new())
        };
        let bindings = DeepSeekV4P6SourceBindings {
            artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            layer: controls.layer,
            token_id: controls.token_id,
            token_position: controls.token_position,
            gate_mode: DeepSeekV4LayerGateMode::HashTokenIdToExpertIds,
            gate_weight_name,
            gate_weight_sha256: sha256(&gate_bytes),
            route: DeepSeekV4P6GateRouteBinding::HashTid2Eid {
                tid2eid_name,
                tid2eid_sha256: sha256(&tid2eid_bytes),
                selected_expert_ids_top_slot_order: selected_ids_top_slot_order,
            },
            selected_expert_ids_top_slot_order: selected_ids_top_slot_order,
            resident_experts_numeric_source_order,
            shared_w1_weight_name: shared_names.0,
            shared_w3_weight_name: shared_names.1,
            shared_w2_weight_name: shared_names.2,
            source_parent_retained: false,
            host_activation_handoff_permitted: false,
            host_route_weight_handoff_permitted: false,
            host_route_id_readback_for_residency: false,
        };
        Ok(Self {
            controls,
            bindings,
            route_mode: P6RouteMode::Hash,
            artifact_root: None,
            learned_reader: None,
            learned_reader_reuse: false,
            learned_expert_cache: None,
            learned_gpu_expert_cache: None,
            learned_expert_cache_reuse: false,
            context_queue_identity: context_queue_identity(metal),
            single_command_buffer: crate::env_on(P6_SINGLE_CB_ENV),
            prefix_concurrent: crate::env_on(P6_PREFIX_CONCURRENT_ENV),
            threads,
            gate_weight: metal.new_buffer_with_bytes_checked(&gate_bytes)?,
            route_table: metal.new_buffer_with_bytes_checked(&tid2eid_bytes)?,
            input_quant: metal.new_buffer_checked(HIDDEN_SIZE)?,
            input_scales: metal.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?,
            gate_logits: metal.new_buffer_checked(GATE_LOGITS_BYTES)?,
            route_scores: metal.new_buffer_checked(ORIGINAL_SCORES_BYTES)?,
            route_valid: metal.new_buffer_checked(ROUTE_VALID_BYTES)?,
            routed,
            experts_loaded: true,
            shared,
            pipeline_cache: Default::default(),
            pipeline_cache_reuse: crate::env_opt_out(P6_PIPELINE_CACHE_ENV),
            fused_gate_up_refs,
            fused_gate_up_weight_resources,
            fused_gate_up_output_resources,
            fused_down_refs,
            fused_down_resources,
            fused_down_output_resources,
            batched_down_qat_refs,
            batched_down_qat_resources,
        })
    }

    fn prepare_learned(
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        controls: DeepSeekV4P6SourceControls,
        catalog: &DeepSeekV4LayerDeviceCatalog,
    ) -> Result<Self> {
        let (gate_weight_name, gate_bytes) = load_gate_weight(reader, controls.layer)?;
        let expected_gate_weight = catalog
            .plan(controls.layer)?
            .gate_score_weight_name(catalog.anchors())?;
        let bias_name = catalog
            .plan(controls.layer)?
            .gate_route_data_name(catalog.anchors())?;
        if gate_weight_name != expected_gate_weight
            || bias_name != format!("layers.{}.ffn.gate.bias", controls.layer)
        {
            return Err(p6_error(
                "P6 learned-bias gate tensor names disagree with verified layer-source anchors",
            ));
        }
        let bias_meta = reader.tensor_metadata(&bias_name)?;
        if bias_meta.dtype != "F32"
            || bias_meta.shape.as_slice() != [ROUTED_EXPERTS as u64]
            || bias_meta.bytes != GATE_BIAS_BYTES as u64
        {
            return Err(p6_error(
                "P6 learned-bias gate.bias geometry is not F32[256]",
            ));
        }
        let bias_bytes = reader.read_verified_full(&bias_name, bias_meta.bytes as usize)?;
        if bias_bytes.len() != GATE_BIAS_BYTES {
            return Err(p6_error(
                "P6 learned-bias gate.bias read returned unexpected length",
            ));
        }

        // Placeholder expert slots (weights filled after on-device route).
        let mut resident_bindings = Vec::with_capacity(ACTIVATED_EXPERTS);
        let mut routed = Vec::with_capacity(ACTIVATED_EXPERTS);
        for source_top_slot in 0..ACTIVATED_EXPERTS {
            let w1 = empty_fp4(metal, FP4_W1_W3)?;
            let w3 = empty_fp4(metal, FP4_W1_W3)?;
            let w2 = empty_fp4(metal, FP4_W2)?;
            resident_bindings.push(DeepSeekV4P6ResidentExpertBinding {
                source_top_slot: source_top_slot as u32,
                expert_id: u32::MAX, // unresolved until execute
                w1_weight_name: String::new(),
                w3_weight_name: String::new(),
                w2_weight_name: String::new(),
            });
            routed.push(allocate_routed_expert(
                metal,
                source_top_slot as u32,
                w1,
                w3,
                w2,
            )?);
        }
        let routed: [RoutedExpertGpu; ACTIVATED_EXPERTS] = routed
            .try_into()
            .map_err(|_| p6_error("P6 learned did not allocate six expert slots"))?;
        let resident_experts_numeric_source_order: [DeepSeekV4P6ResidentExpertBinding;
            ACTIVATED_EXPERTS] = resident_bindings
            .try_into()
            .map_err(|_| p6_error("P6 learned resident binding count is not six"))?;

        let (shared, shared_names) = prepare_shared_expert(metal, reader, controls.layer)?;
        let fused_gate_up_swiglu = crate::env_on(P6_FP4_GATE_UP_SWIGLU_FUSED_ENV);
        let fused_gate_up_swiglu_simd =
            fused_gate_up_swiglu && crate::env_on(P6_FP4_GATE_UP_SWIGLU_SIMD_ENV);
        let shared_fp8_gate_up_swiglu = crate::env_on(P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_ENV);
        let fused_down_bf16 = crate::env_on(P6_FP4_DOWN_BF16_FUSED_ENV);
        let fused_down_shared_combine =
            fused_down_bf16 && crate::env_on(P6_FP4_DOWN_SHARED_COMBINE_FUSED_ENV);
        let shared_fp8_down_combine =
            !fused_down_shared_combine && crate::env_on(P6_SHARED_FP8_DOWN_COMBINE_FUSED_ENV);
        let batched_down_qat = crate::env_on(P6_BATCHED_DOWN_QAT_ENV);
        let fused_down_bf16_simd = fused_down_bf16
            && !fused_down_shared_combine
            && crate::env_on(P6_FP4_DOWN_BF16_SIMD_ENV);
        let threads = precompile_threads(
            metal,
            fused_gate_up_swiglu,
            fused_gate_up_swiglu_simd,
            shared_fp8_gate_up_swiglu,
            shared_fp8_down_combine,
            batched_down_qat,
            fused_down_bf16,
            fused_down_bf16_simd,
            fused_down_shared_combine,
        )?;
        let (fused_gate_up_refs, fused_gate_up_weight_resources, fused_gate_up_output_resources) =
            if fused_gate_up_swiglu {
                let (refs, weight_resources, output_resources) =
                    prepare_fused_gate_up_refs(metal, &routed, false)?;
                (Some(refs), weight_resources, output_resources)
            } else {
                (None, Vec::new(), Vec::new())
            };
        let (fused_down_refs, fused_down_resources, fused_down_output_resources) =
            if fused_down_bf16 {
                let (refs, resources, output_resources) =
                    prepare_fused_down_refs(metal, &routed, false)?;
                (Some(refs), resources, output_resources)
            } else {
                (None, Vec::new(), Vec::new())
            };
        let (batched_down_qat_refs, batched_down_qat_resources) = if batched_down_qat {
            let (refs, resources) = prepare_batched_down_qat_refs(metal, &routed, &shared)?;
            (Some(refs), resources)
        } else {
            (None, Vec::new())
        };
        let artifact_root = reader.artifact_root().to_path_buf();
        let bindings = DeepSeekV4P6SourceBindings {
            artifact_manifest_seal_sha256: reader.manifest_seal_sha256().to_owned(),
            layer: controls.layer,
            token_id: controls.token_id,
            token_position: controls.token_position,
            gate_mode: DeepSeekV4LayerGateMode::LearnedScoresWithSelectionBias,
            gate_weight_name,
            gate_weight_sha256: sha256(&gate_bytes),
            route: DeepSeekV4P6GateRouteBinding::LearnedBias {
                bias_name,
                bias_sha256: sha256(&bias_bytes),
                host_route_id_readback_for_residency: true,
            },
            selected_expert_ids_top_slot_order: [u32::MAX; ACTIVATED_EXPERTS],
            resident_experts_numeric_source_order,
            shared_w1_weight_name: shared_names.0,
            shared_w3_weight_name: shared_names.1,
            shared_w2_weight_name: shared_names.2,
            source_parent_retained: false,
            host_activation_handoff_permitted: false,
            host_route_weight_handoff_permitted: false,
            host_route_id_readback_for_residency: true,
        };
        let learned_reader_reuse = crate::env_on(P6_LEARNED_READER_REUSE_ENV);
        let learned_expert_cache_reuse = crate::env_on(P6_LEARNED_EXPERT_CACHE_REUSE_ENV);
        Ok(Self {
            controls,
            bindings,
            route_mode: P6RouteMode::Learned,
            artifact_root: Some(artifact_root),
            learned_reader: None,
            learned_reader_reuse,
            learned_expert_cache: None,
            learned_gpu_expert_cache: None,
            learned_expert_cache_reuse,
            context_queue_identity: context_queue_identity(metal),
            single_command_buffer: crate::env_on(P6_SINGLE_CB_ENV),
            prefix_concurrent: crate::env_on(P6_PREFIX_CONCURRENT_ENV),
            threads,
            gate_weight: metal.new_buffer_with_bytes_checked(&gate_bytes)?,
            route_table: metal.new_buffer_with_bytes_checked(&bias_bytes)?,
            input_quant: metal.new_buffer_checked(HIDDEN_SIZE)?,
            input_scales: metal.new_buffer_checked(HIDDEN_SIZE / ACT_QUANT_BLOCK)?,
            gate_logits: metal.new_buffer_checked(GATE_LOGITS_BYTES)?,
            route_scores: metal.new_buffer_checked(ORIGINAL_SCORES_BYTES)?,
            route_valid: metal.new_buffer_checked(ROUTE_VALID_BYTES)?,
            routed,
            experts_loaded: false,
            shared,
            pipeline_cache: Default::default(),
            pipeline_cache_reuse: crate::env_opt_out(P6_PIPELINE_CACHE_ENV),
            fused_gate_up_refs,
            fused_gate_up_weight_resources,
            fused_gate_up_output_resources,
            fused_down_refs,
            fused_down_resources,
            fused_down_output_resources,
            batched_down_qat_refs,
            batched_down_qat_resources,
        })
    }

    /// Convenience constructor that binds the executor to the same source
    /// controls already admitted by P7.  It validates only the static
    /// contract; the BF16 activation remains exclusively in P7's caller-owned
    /// device buffer until `execute_p6_on_device` is called.
    pub fn prepare_for_p7(
        metal: &MetalContext,
        reader: &DeepSeekV4FullStreamReader,
        cache: &mut DeepSeekV4ExpertBundleCache,
        source: &DeepSeekV4P7FfnSourceContract,
    ) -> Result<Self> {
        if source.ffn_norm.name != format!("layers.{}.ffn_norm.weight", source.layer)
            || source.ffn_norm.dtype != "BF16"
            || source.ffn_norm.shape.as_slice() != [HIDDEN_SIZE as u64]
            || source.ffn_norm.bytes != DSV4F_P7_FFN_NORM_BF16_BYTES
            || source.host_activation_handoff_permitted
            || !source.source_upload_required_before_execution
        {
            return Err(p6_error(
                "P7 source contract is not a valid BF16[4096] no-host P6 predecessor binding",
            ));
        }
        Self::prepare(
            metal,
            reader,
            cache,
            DeepSeekV4P6SourceControls::new(source.layer, source.token_id, source.token_position),
        )
    }

    pub const fn source_controls(&self) -> DeepSeekV4P6SourceControls {
        self.controls
    }

    pub const fn act_quant_kernel(&self) -> &'static str {
        self.threads.qat_kernel
    }

    pub const fn act_quant_threads(&self) -> u32 {
        self.threads.qat
    }

    pub const fn fp8_kernel(&self) -> &'static str {
        self.threads.fp8_kernel
    }

    pub const fn fp8_threads(&self) -> u32 {
        self.threads.fp8
    }

    pub fn source_bindings(&self) -> &DeepSeekV4P6SourceBindings {
        &self.bindings
    }

    fn validate_p7_input(&self, input: &DeepSeekV4P7P6DeviceInput<'_>) -> Result<()> {
        if context_queue_identity(input.metal) != self.context_queue_identity {
            return Err(p6_error(
                "P6 executor was prepared for a different caller-owned MetalContext/queue",
            ));
        }
        if input.layer != self.controls.layer
            || input.token_id != self.controls.token_id
            || input.token_position != self.controls.token_position
        {
            return Err(p6_error(
                "P6 device input does not match the source-bound layer/token/position controls",
            ));
        }
        if input.ffn_norm_bf16.length() < HIDDEN_BF16_BYTES as u64 {
            return Err(p6_error("P6 input buffer is smaller than BF16[4096]"));
        }
        Ok(())
    }

    fn execute(
        &mut self,
        input: DeepSeekV4P7P6DeviceInput<'_>,
    ) -> Result<DeepSeekV4P7P6DeviceOutput> {
        self.validate_p7_input(&input)?;
        match self.route_mode {
            P6RouteMode::Hash => self.execute_hash(input),
            P6RouteMode::Learned => self.execute_learned(input),
        }
    }

    fn execute_hash(
        &mut self,
        input: DeepSeekV4P7P6DeviceInput<'_>,
    ) -> Result<DeepSeekV4P7P6DeviceOutput> {
        let moe_output_bf16 = input.metal.new_buffer_checked(HIDDEN_BF16_BYTES)?;
        let route_ids_u32 = input.metal.new_buffer_checked(ROUTE_IDS_BYTES)?;
        let route_weights_f32 = input.metal.new_buffer_checked(ROUTE_WEIGHTS_BYTES)?;

        // The historical A/B uses two command buffers. The explicit candidate
        // keeps the same ordered concurrent groups but appends the dependent
        // down/combine wave to the first buffer, removing one CPU-visible wait
        // without introducing a host activation or route-weight handoff.
        if self.single_command_buffer {
            self.dispatch_cached_batch(input.metal, |this, batch| {
                this.dispatch_hash_prefix(batch, &input, &route_ids_u32, &route_weights_f32)?;
                this.dispatch_down_projections_and_combine(batch, &moe_output_bf16)
            })?;
        } else {
            self.dispatch_cached_batch(input.metal, |this, batch| {
                this.dispatch_hash_prefix(batch, &input, &route_ids_u32, &route_weights_f32)
            })?;
            self.dispatch_cached_batch(input.metal, |this, batch| {
                this.dispatch_down_projections_and_combine(batch, &moe_output_bf16)
            })?;
        }

        let output = DeepSeekV4P7P6DeviceOutput {
            moe_output_bf16,
            route_ids_u32,
            route_weights_f32,
            gate_logits_f32: self.gate_logits.to_owned(),
            original_scores_f32: self.route_scores.to_owned(),
            route_valid_u32: self.route_valid.to_owned(),
        };
        output.validate()?;
        Ok(output)
    }

    fn dispatch_hash_prefix(
        &self,
        batch: &mut CommandBatch<'_>,
        input: &DeepSeekV4P7P6DeviceInput<'_>,
        route_ids_u32: &metal::Buffer,
        route_weights_f32: &metal::Buffer,
    ) -> Result<()> {
        if self.prefix_concurrent {
            batch.begin_concurrent_group()?;
            dispatch_gate_concurrent(
                batch,
                &self.gate_weight,
                input.ffn_norm_bf16,
                &self.gate_logits,
                self.threads.gate,
            )?;
            dispatch_act_quant_concurrent(
                batch,
                input.ffn_norm_bf16,
                &self.input_quant,
                &self.input_scales,
                HIDDEN_SIZE as u32,
                self.threads.qat,
                self.threads.qat_kernel,
                self.threads.qat_vector_width,
            )?;
            batch.end_concurrent_group()?;
        } else {
            dispatch_gate(
                batch,
                &self.gate_weight,
                input.ffn_norm_bf16,
                &self.gate_logits,
                self.threads.gate,
            )?;
            dispatch_act_quant_ordered(
                batch,
                input.ffn_norm_bf16,
                &self.input_quant,
                &self.input_scales,
                HIDDEN_SIZE as u32,
                self.threads.qat,
                self.threads.qat_kernel,
                self.threads.qat_vector_width,
            )?;
        }
        dispatch_hash_route(
            batch,
            &self.gate_logits,
            &self.route_table,
            route_ids_u32,
            route_weights_f32,
            &self.route_scores,
            &self.route_valid,
            self.controls.token_id,
            self.threads.route,
        )?;
        self.dispatch_up_projections_and_swiglu(batch, route_weights_f32)
    }

    /// Two-phase learned-bias MoE:
    /// 1. Gate + QAT + learned route (device)
    /// 2. Host reads selected expert IDs only; loads six FP4 bundles
    /// 3. Expert body + combine (device; route weights stay on device)
    fn execute_learned(
        &mut self,
        input: DeepSeekV4P7P6DeviceInput<'_>,
    ) -> Result<DeepSeekV4P7P6DeviceOutput> {
        let moe_output_bf16 = input.metal.new_buffer_checked(HIDDEN_BF16_BYTES)?;
        let route_ids_u32 = input.metal.new_buffer_checked(ROUTE_IDS_BYTES)?;
        let route_weights_f32 = input.metal.new_buffer_checked(ROUTE_WEIGHTS_BYTES)?;

        // Phase 1: gate logits, activation quant, learned top-k route.
        self.dispatch_cached_batch(input.metal, |this, batch| {
            if this.prefix_concurrent {
                batch.begin_concurrent_group()?;
                dispatch_gate_concurrent(
                    batch,
                    &this.gate_weight,
                    input.ffn_norm_bf16,
                    &this.gate_logits,
                    this.threads.gate,
                )?;
                dispatch_act_quant_concurrent(
                    batch,
                    input.ffn_norm_bf16,
                    &this.input_quant,
                    &this.input_scales,
                    HIDDEN_SIZE as u32,
                    this.threads.qat,
                    this.threads.qat_kernel,
                    this.threads.qat_vector_width,
                )?;
                batch.end_concurrent_group()?;
            } else {
                dispatch_gate(
                    batch,
                    &this.gate_weight,
                    input.ffn_norm_bf16,
                    &this.gate_logits,
                    this.threads.gate,
                )?;
                dispatch_act_quant_ordered(
                    batch,
                    input.ffn_norm_bf16,
                    &this.input_quant,
                    &this.input_scales,
                    HIDDEN_SIZE as u32,
                    this.threads.qat,
                    this.threads.qat_kernel,
                    this.threads.qat_vector_width,
                )?;
            }
            dispatch_learned_route(
                batch,
                &this.gate_logits,
                &this.route_table,
                &route_ids_u32,
                &route_weights_f32,
                &this.route_scores,
                &this.route_valid,
                this.threads.route,
            )?;
            Ok(())
        })?;

        // Host residency boundary: selected expert IDs only (not weights/acts).
        let selected = read_u32_buffer(&route_ids_u32, ACTIVATED_EXPERTS)?;
        let valid = read_u32_buffer(&self.route_valid, 1)?[0];
        if valid != 1 {
            return Err(p6_error(format!(
                "P6 learned-bias route kernel valid code {valid}"
            )));
        }
        for (slot, &id) in selected.iter().enumerate() {
            if id >= ROUTED_EXPERTS as u32 {
                return Err(p6_error(format!(
                    "P6 learned route selected out-of-range expert {id} at slot {slot}"
                )));
            }
        }
        // Reject duplicates (source top-k must be unique for independent waves).
        let mut sorted = selected;
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(p6_error("P6 learned route produced duplicate expert IDs"));
        }

        // A reusable learned executor may be replayed for diagnostics or a
        // stable route window. Keep the already-uploaded source bundles when
        // the device route is identical; reloading them would re-admit the
        // source and replace six GPU buffers without changing any arithmetic.
        if !self.experts_loaded || self.bindings.selected_expert_ids_top_slot_order != selected {
            self.load_learned_experts(input.metal, &selected)?;
            self.bindings.selected_expert_ids_top_slot_order = selected;
        }

        // Phase 2a/2b: experts are now resident.  The single-CB candidate is
        // also valid after this host residency boundary: all expert buffers
        // and indirect records have been updated before encoding starts, so
        // the two dependency-ordered device waves can share one commit/wait.
        // The default keeps separate command buffers for an independent A/B.
        if self.single_command_buffer {
            self.dispatch_cached_batch(input.metal, |this, batch| {
                this.dispatch_up_projections_and_swiglu(batch, &route_weights_f32)?;
                this.dispatch_down_projections_and_combine(batch, &moe_output_bf16)
            })?;
        } else {
            // Phase 2a: W1/W3 + casts + SwiGLU (experts now resident).
            self.dispatch_cached_batch(input.metal, |this, batch| {
                this.dispatch_up_projections_and_swiglu(batch, &route_weights_f32)
            })?;

            // Phase 2b: down projections + combine.
            self.dispatch_cached_batch(input.metal, |this, batch| {
                this.dispatch_down_projections_and_combine(batch, &moe_output_bf16)
            })?;
        }

        let output = DeepSeekV4P7P6DeviceOutput {
            moe_output_bf16,
            route_ids_u32,
            route_weights_f32,
            gate_logits_f32: self.gate_logits.to_owned(),
            original_scores_f32: self.route_scores.to_owned(),
            route_valid_u32: self.route_valid.to_owned(),
        };
        output.validate()?;
        Ok(output)
    }

    /// Carry P6's warmed pipeline handles across its adjacent command
    /// buffers. The closure receives an immutable executor view so the cache
    /// itself can be moved out and restored even if encoding fails.
    fn dispatch_cached_batch(
        &mut self,
        metal: &MetalContext,
        encode: impl FnOnce(&Self, &mut CommandBatch<'_>) -> Result<()>,
    ) -> Result<()> {
        let reuse = self.pipeline_cache_reuse;
        let mut pipeline_cache = std::mem::take(&mut self.pipeline_cache);
        let result = if reuse {
            metal.dispatch_batch_with_pipeline_cache(&mut pipeline_cache, |batch| {
                encode(self, batch)
            })
        } else {
            metal.dispatch_batch(|batch| encode(self, batch))
        };
        self.pipeline_cache = pipeline_cache;
        result
    }

    fn load_learned_experts(
        &mut self,
        metal: &MetalContext,
        selected_top_slot: &[u32; ACTIVATED_EXPERTS],
    ) -> Result<()> {
        // Move the metadata-only reader out while this method updates GPU
        // buffers. This keeps the reuse candidate borrow-safe: the reader is
        // not a borrow of the mutable executor during expert replacement.
        let reader = self.take_or_admit_learned_reader()?;
        let reusable_cache = if self.learned_expert_cache_reuse {
            self.learned_expert_cache.take()
        } else {
            None
        };
        let result = self.load_learned_experts_from_reader(
            metal,
            selected_top_slot,
            &reader,
            reusable_cache,
        );
        if self.learned_reader_reuse {
            self.learned_reader = Some(reader);
        }
        match result {
            Ok(cache) => {
                if self.learned_expert_cache_reuse {
                    self.learned_expert_cache = cache;
                }
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn load_learned_experts_from_reader(
        &mut self,
        metal: &MetalContext,
        selected_top_slot: &[u32; ACTIVATED_EXPERTS],
        reader: &DeepSeekV4FullStreamReader,
        reusable_cache: Option<DeepSeekV4ExpertBundleCache>,
    ) -> Result<Option<DeepSeekV4ExpertBundleCache>> {
        // Capacity for six full expert bundles (~80 MB).
        let mut hot_bytes = 0u64;
        for &expert_id in selected_top_slot {
            let key = ExpertBundleKey::new(self.controls.layer as u16, expert_id as u16);
            let desc =
                crate::gravity_deepseek_v4_expert_cache::resolve_expert_bundle(&reader, key)?;
            hot_bytes = hot_bytes
                .checked_add(desc.payload_bytes)
                .ok_or_else(|| p6_error("P6 learned hot capacity overflow"))?;
        }
        let mut cache = match reusable_cache {
            Some(cache) if cache.state().hot_capacity_bytes >= hot_bytes => cache,
            _ => DeepSeekV4ExpertBundleCache::new(hot_bytes, 0)?,
        };
        for &expert_id in selected_top_slot {
            let key = ExpertBundleKey::new(self.controls.layer as u16, expert_id as u16);
            cache.acquire(&reader, key)?;
        }

        // Keep the GPU-side cache bounded by the current route before adding
        // any new buffers.  This is deliberately coupled to the existing
        // learned expert-cache A/B: control=0 restores fresh uploads and
        // cannot accidentally retain a second GPU model copy.
        let mut gpu_cache = if self.learned_expert_cache_reuse {
            self.learned_gpu_expert_cache.take().unwrap_or_default()
        } else {
            BTreeMap::new()
        };
        if self.learned_expert_cache_reuse {
            retain_selected_expert_ids(&mut gpu_cache, selected_top_slot);
        } else {
            gpu_cache.clear();
        }

        for (slot, &expert_id) in selected_top_slot.iter().enumerate() {
            let key = ExpertBundleKey::new(self.controls.layer as u16, expert_id as u16);
            let bundle = cache.resident(key).ok_or_else(|| {
                p6_error(format!(
                    "P6 learned expert {expert_id} missing from hot cache after acquire"
                ))
            })?;
            let uploaded = if self.learned_expert_cache_reuse {
                if let Some(cached) = gpu_cache.get(&expert_id) {
                    cached.clone()
                } else {
                    let (w1, w1_name) =
                        upload_cached_fp4(metal, bundle, ExpertOperator::W1, FP4_W1_W3)?;
                    let (w3, w3_name) =
                        upload_cached_fp4(metal, bundle, ExpertOperator::W3, FP4_W1_W3)?;
                    let (w2, w2_name) =
                        upload_cached_fp4(metal, bundle, ExpertOperator::W2, FP4_W2)?;
                    let uploaded = CachedP6ExpertGpu {
                        w1,
                        w3,
                        w2,
                        w1_name,
                        w3_name,
                        w2_name,
                    };
                    gpu_cache.insert(expert_id, uploaded.clone());
                    uploaded
                }
            } else {
                let (w1, w1_name) =
                    upload_cached_fp4(metal, bundle, ExpertOperator::W1, FP4_W1_W3)?;
                let (w3, w3_name) =
                    upload_cached_fp4(metal, bundle, ExpertOperator::W3, FP4_W1_W3)?;
                let (w2, w2_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W2, FP4_W2)?;
                CachedP6ExpertGpu {
                    w1,
                    w3,
                    w2,
                    w1_name,
                    w3_name,
                    w2_name,
                }
            };
            // Replace placeholder weight buffers (scratch buffers retained).
            self.routed[slot].w1 = uploaded.w1;
            self.routed[slot].w3 = uploaded.w3;
            self.routed[slot].w2 = uploaded.w2;
            self.routed[slot].source_top_slot = slot as u32;
            self.bindings.resident_experts_numeric_source_order[slot] =
                DeepSeekV4P6ResidentExpertBinding {
                    source_top_slot: slot as u32,
                    expert_id,
                    w1_weight_name: uploaded.w1_name,
                    w3_weight_name: uploaded.w3_name,
                    w2_weight_name: uploaded.w2_name,
                };
        }
        if let Some(refs) = self.fused_gate_up_refs.as_ref() {
            self.fused_gate_up_weight_resources = fused_gate_up_resources(&self.routed).0;
            write_fused_gate_up_refs(refs, &self.routed, true)?;
        }
        if let Some(refs) = self.fused_down_refs.as_ref() {
            let (resources, output_resources) = fused_down_resources(&self.routed);
            self.fused_down_resources = resources;
            self.fused_down_output_resources = output_resources;
            write_fused_down_refs(refs, &self.routed, true)?;
        }
        self.experts_loaded = true;
        if self.learned_expert_cache_reuse {
            debug_assert_eq!(gpu_cache.len(), ACTIVATED_EXPERTS);
            self.learned_gpu_expert_cache = Some(gpu_cache);
        } else {
            self.learned_gpu_expert_cache = None;
        }
        // Silence unused on hash-only builds / future diagnostics.
        let _ = metal;
        Ok(if self.learned_expert_cache_reuse {
            Some(cache)
        } else {
            None
        })
    }

    /// Admit the learned source reader at the route-residency boundary.
    ///
    /// The authority path intentionally re-admits on every route change so
    /// the candidate has a matched control. The opt-in candidate retains only
    /// the verified metadata reader and reuses its digest/index caches; it
    /// still creates a fresh bounded source cache; when expert-cache reuse is
    /// enabled, a separate six-entry GPU weight cache also avoids re-uploading
    /// overlapping expert bundles.
    fn take_or_admit_learned_reader(&mut self) -> Result<DeepSeekV4FullStreamReader> {
        if self.learned_reader_reuse {
            if let Some(reader) = self.learned_reader.take() {
                return Ok(reader);
            }
        }

        let root = self
            .artifact_root
            .as_ref()
            .ok_or_else(|| {
                p6_error("P6 learned execute missing artifact_root for expert residency")
            })?
            .clone();
        let reader = DeepSeekV4FullStreamReader::admit(root)?;
        if reader.manifest_seal_sha256() != self.bindings.artifact_manifest_seal_sha256 {
            return Err(p6_error(
                "P6 learned reader manifest seal changed across the residency boundary",
            ));
        }
        Ok(reader)
    }

    fn dispatch_up_projections_and_swiglu(
        &self,
        batch: &mut CommandBatch<'_>,
        route_weights_f32: &metal::Buffer,
    ) -> Result<()> {
        let shared_fp8_fused = self.threads.shared_fp8_gate_up_swiglu;
        batch.begin_concurrent_group()?;
        if self.threads.fused_gate_up_swiglu {
            let fused_gate_up_refs = self.fused_gate_up_refs.as_ref().ok_or_else(|| {
                p6_error("P6 fused gate/up/SwiGLU was enabled without indirect refs")
            })?;
            dispatch_fused_gate_up_swiglu(
                batch,
                fused_gate_up_refs,
                &self.fused_gate_up_weight_resources,
                &self.fused_gate_up_output_resources,
                &self.input_quant,
                &self.input_scales,
                route_weights_f32,
                MOE_INTER_DIM as u32,
                HIDDEN_SIZE as u32 / 2,
                HIDDEN_SIZE as u32 / 32,
                ACTIVATED_EXPERTS as u32,
                self.threads.fused_gate_up_swiglu_kernel,
                self.threads.fused_gate_up_swiglu_threads,
                self.threads.fused_gate_up_swiglu_threads_x,
                self.threads.fused_gate_up_swiglu_rows_per_tg,
            )?;
        } else {
            for expert in &self.routed {
                dispatch_fp4(
                    batch,
                    &expert.w1,
                    &self.input_quant,
                    &self.input_scales,
                    &expert.gate_f32,
                    self.threads.fp4,
                    self.threads.fp4_kernel,
                    self.threads.fp4_threads_x,
                    self.threads.fp4_rows_per_tg,
                )?;
                dispatch_fp4(
                    batch,
                    &expert.w3,
                    &self.input_quant,
                    &self.input_scales,
                    &expert.up_f32,
                    self.threads.fp4,
                    self.threads.fp4_kernel,
                    self.threads.fp4_threads_x,
                    self.threads.fp4_rows_per_tg,
                )?;
            }
        }
        if shared_fp8_fused {
            dispatch_shared_fp8_gate_up_swiglu(
                batch,
                &self.shared.w1,
                &self.shared.w3,
                &self.input_quant,
                &self.input_scales,
                &self.shared.swiglu_bf16,
                self.threads.shared_fp8_gate_up_swiglu_threads,
            )?;
        } else {
            dispatch_fp8(
                batch,
                &self.shared.w1,
                &self.input_quant,
                &self.input_scales,
                &self.shared.gate_f32,
                self.threads.fp8,
                self.threads.fp8_kernel,
                self.threads.fp8_threads_x,
                self.threads.fp8_rows_per_tg,
            )?;
            dispatch_fp8(
                batch,
                &self.shared.w3,
                &self.input_quant,
                &self.input_scales,
                &self.shared.up_f32,
                self.threads.fp8,
                self.threads.fp8_kernel,
                self.threads.fp8_threads_x,
                self.threads.fp8_rows_per_tg,
            )?;
        }
        batch.end_concurrent_group()?;

        if !self.threads.fused_gate_up_swiglu || !shared_fp8_fused {
            batch.begin_concurrent_group()?;
            if !self.threads.fused_gate_up_swiglu {
                for expert in &self.routed {
                    dispatch_bf16_cast(
                        batch,
                        &expert.gate_f32,
                        &expert.gate_bf16,
                        MOE_INTER_DIM as u32,
                        self.threads.cast,
                    )?;
                    dispatch_bf16_cast(
                        batch,
                        &expert.up_f32,
                        &expert.up_bf16,
                        MOE_INTER_DIM as u32,
                        self.threads.cast,
                    )?;
                }
            }
            if !shared_fp8_fused {
                dispatch_bf16_cast(
                    batch,
                    &self.shared.gate_f32,
                    &self.shared.gate_bf16,
                    MOE_INTER_DIM as u32,
                    self.threads.cast,
                )?;
                dispatch_bf16_cast(
                    batch,
                    &self.shared.up_f32,
                    &self.shared.up_bf16,
                    MOE_INTER_DIM as u32,
                    self.threads.cast,
                )?;
            }
            batch.end_concurrent_group()?;
        }

        if !self.threads.fused_gate_up_swiglu || !shared_fp8_fused {
            batch.begin_concurrent_group()?;
            if !self.threads.fused_gate_up_swiglu {
                for expert in &self.routed {
                    dispatch_routed_swiglu(
                        batch,
                        &expert.gate_bf16,
                        &expert.up_bf16,
                        &expert.swiglu_bf16,
                        route_weights_f32,
                        expert.source_top_slot,
                        self.threads.routed_swiglu,
                    )?;
                }
            }
            if !shared_fp8_fused {
                dispatch_shared_swiglu(
                    batch,
                    &self.shared.gate_bf16,
                    &self.shared.up_bf16,
                    &self.shared.swiglu_bf16,
                    self.threads.shared_swiglu,
                )?;
            }
            batch.end_concurrent_group()?;
        }
        Ok(())
    }

    fn dispatch_down_projections_and_combine(
        &self,
        batch: &mut CommandBatch<'_>,
        moe_output_bf16: &metal::Buffer,
    ) -> Result<()> {
        // The fixed-seven QAT kernel is already one dispatch whose seven
        // independent lanes are represented inside its grid. When its only
        // consumer is the full fused down/combine kernel, keep both in one
        // explicit serial encoder and insert a resource barrier. This removes
        // one encoder without serialising any of the other P6 waves.
        let collapse_down_tail = self.single_command_buffer
            && self.threads.batched_down_qat
            && self.threads.fused_down_shared_combine;
        if collapse_down_tail {
            batch.begin_serial_group()?;
        } else {
            batch.begin_concurrent_group()?;
        }
        if self.threads.batched_down_qat {
            let refs = self
                .batched_down_qat_refs
                .as_ref()
                .ok_or_else(|| p6_error("P6 batched down-QAT was enabled without indirect refs"))?;
            dispatch_batched_down_qat(
                batch,
                refs,
                &self.batched_down_qat_resources,
                MOE_INTER_DIM as u32,
                self.threads.batched_down_qat_threads,
            )?;
        } else {
            for expert in &self.routed {
                dispatch_act_quant_concurrent(
                    batch,
                    &expert.swiglu_bf16,
                    &expert.down_quant,
                    &expert.down_scales,
                    MOE_INTER_DIM as u32,
                    self.threads.qat,
                    self.threads.qat_kernel,
                    self.threads.qat_vector_width,
                )?;
            }
            dispatch_act_quant_concurrent(
                batch,
                &self.shared.swiglu_bf16,
                &self.shared.down_quant,
                &self.shared.down_scales,
                MOE_INTER_DIM as u32,
                self.threads.qat,
                self.threads.qat_kernel,
                self.threads.qat_vector_width,
            )?;
        }
        if collapse_down_tail {
            // The full fusion follows the GPU addresses stored in the fixed
            // routed records, so declare exactly the QAT write ranges at the
            // dependency boundary. The shared and routed weights are read
            // only by the following dispatch and need no write-side barrier.
            let qat_output_resources: Vec<&metal::ResourceRef> = self.batched_down_qat_resources
                [P6_BATCHED_DOWN_QAT_TENSORS as usize..P6_BATCHED_DOWN_QAT_TENSORS as usize * 3]
                .iter()
                .map(|buffer| -> &metal::ResourceRef { &**buffer })
                .collect();
            batch.memory_barrier_in_serial_group(&qat_output_resources)?;
        } else {
            batch.end_concurrent_group()?;
        }

        if self.threads.fused_down_shared_combine {
            // The full downstream fusion consumes the seven completed QAT
            // outputs only after the QAT wave has closed. It computes routed
            // FP4 W2 rows, shared FP8 W2 rows, and the source-order BF16
            // combine directly into the caller-owned output, so no routed or
            // shared down intermediate is materialized for this branch.
            let refs = self.fused_down_refs.as_ref().ok_or_else(|| {
                p6_error("P6 full down fusion was enabled without routed indirect refs")
            })?;
            if !collapse_down_tail {
                batch.begin_concurrent_group()?;
            }
            dispatch_fused_down_shared_combine(
                batch,
                refs,
                &self.fused_down_resources,
                &self.shared.w2,
                &self.shared.down_quant,
                &self.shared.down_scales,
                moe_output_bf16,
                self.threads.fused_down_shared_combine_threads,
            )?;
            if collapse_down_tail {
                batch.end_serial_group()?;
            } else {
                batch.end_concurrent_group()?;
            }
            return Ok(());
        }

        if self.threads.shared_fp8_down_combine {
            // The shared fused consumer must start after routed W2 and any
            // routed casts have completed because it reads their BF16 rows.
            // Keep those two dependency boundaries explicit; the authority
            // path below retains the original four-wave down topology.
            batch.begin_concurrent_group()?;
            if self.threads.fused_down_bf16 {
                let refs = self.fused_down_refs.as_ref().ok_or_else(|| {
                    p6_error("P6 fused down FP4/BF16 was enabled without indirect refs")
                })?;
                dispatch_fused_down_bf16(
                    batch,
                    refs,
                    &self.fused_down_resources,
                    &self.fused_down_output_resources,
                    self.threads.fused_down_bf16_kernel,
                    self.threads.fused_down_bf16_threads,
                    self.threads.fused_down_bf16_threads_x,
                    self.threads.fused_down_bf16_rows_per_tg,
                )?;
            } else {
                for expert in &self.routed {
                    dispatch_fp4(
                        batch,
                        &expert.w2,
                        &expert.down_quant,
                        &expert.down_scales,
                        &expert.down_f32,
                        self.threads.fp4,
                        self.threads.fp4_kernel,
                        self.threads.fp4_threads_x,
                        self.threads.fp4_rows_per_tg,
                    )?;
                }
            }
            batch.end_concurrent_group()?;

            if !self.threads.fused_down_bf16 {
                batch.begin_concurrent_group()?;
                for expert in &self.routed {
                    dispatch_bf16_cast(
                        batch,
                        &expert.down_f32,
                        &expert.down_bf16,
                        HIDDEN_SIZE as u32,
                        self.threads.cast,
                    )?;
                }
                batch.end_concurrent_group()?;
            }

            let routed_bf16 = [
                &self.routed[0].down_bf16,
                &self.routed[1].down_bf16,
                &self.routed[2].down_bf16,
                &self.routed[3].down_bf16,
                &self.routed[4].down_bf16,
                &self.routed[5].down_bf16,
            ];
            batch.begin_concurrent_group()?;
            dispatch_shared_fp8_down_bf16_combine(
                batch,
                &self.shared.w2,
                &self.shared.down_quant,
                &self.shared.down_scales,
                &routed_bf16,
                moe_output_bf16,
                self.threads.shared_fp8_down_combine_threads,
            )?;
            batch.end_concurrent_group()?;
        } else {
            batch.begin_concurrent_group()?;
            if self.threads.fused_down_bf16 {
                let refs = self.fused_down_refs.as_ref().ok_or_else(|| {
                    p6_error("P6 fused down FP4/BF16 was enabled without indirect refs")
                })?;
                dispatch_fused_down_bf16(
                    batch,
                    refs,
                    &self.fused_down_resources,
                    &self.fused_down_output_resources,
                    self.threads.fused_down_bf16_kernel,
                    self.threads.fused_down_bf16_threads,
                    self.threads.fused_down_bf16_threads_x,
                    self.threads.fused_down_bf16_rows_per_tg,
                )?;
            } else {
                for expert in &self.routed {
                    dispatch_fp4(
                        batch,
                        &expert.w2,
                        &expert.down_quant,
                        &expert.down_scales,
                        &expert.down_f32,
                        self.threads.fp4,
                        self.threads.fp4_kernel,
                        self.threads.fp4_threads_x,
                        self.threads.fp4_rows_per_tg,
                    )?;
                }
            }
            dispatch_fp8(
                batch,
                &self.shared.w2,
                &self.shared.down_quant,
                &self.shared.down_scales,
                &self.shared.down_f32,
                self.threads.fp8,
                self.threads.fp8_kernel,
                self.threads.fp8_threads_x,
                self.threads.fp8_rows_per_tg,
            )?;
            batch.end_concurrent_group()?;

            batch.begin_concurrent_group()?;
            if !self.threads.fused_down_bf16 {
                for expert in &self.routed {
                    dispatch_bf16_cast(
                        batch,
                        &expert.down_f32,
                        &expert.down_bf16,
                        HIDDEN_SIZE as u32,
                        self.threads.cast,
                    )?;
                }
            }
            dispatch_bf16_cast(
                batch,
                &self.shared.down_f32,
                &self.shared.down_bf16,
                HIDDEN_SIZE as u32,
                self.threads.cast,
            )?;
            batch.end_concurrent_group()?;

            batch.begin_concurrent_group()?;
            dispatch_combine(
                batch,
                [
                    &self.routed[0].down_bf16,
                    &self.routed[1].down_bf16,
                    &self.routed[2].down_bf16,
                    &self.routed[3].down_bf16,
                    &self.routed[4].down_bf16,
                    &self.routed[5].down_bf16,
                ],
                &self.shared.down_bf16,
                moe_output_bf16,
                self.threads.combine,
            )?;
            batch.end_concurrent_group()?;
        }
        Ok(())
    }
}

impl DeepSeekV4P7P6DeviceExecutor for DeepSeekV4Layer0P6MetalExecutor {
    fn execute_p6_on_device(
        &mut self,
        input: DeepSeekV4P7P6DeviceInput<'_>,
    ) -> Result<DeepSeekV4P7P6DeviceOutput> {
        self.execute(input)
    }
}

fn source_route_plan(
    reader: &DeepSeekV4FullStreamReader,
    controls: DeepSeekV4P6SourceControls,
) -> Result<(Vec<u8>, [u32; ACTIVATED_EXPERTS], Vec<(u32, u32)>)> {
    // This bounded source-table read is a residency plan only: it decides
    // which six immutable weight bundles the cache must make hot before an
    // execution. The full table is still uploaded and the P6A device kernel
    // independently gathers the same row, computes scores/weights, and emits
    // the caller-owned route metadata. No host score, weight, or activation
    // crosses this boundary.
    let tid2eid_name = format!("layers.{}.ffn.gate.tid2eid", controls.layer);
    let metadata = reader.tensor_metadata(&tid2eid_name)?;
    let row_bytes = ACTIVATED_EXPERTS
        .checked_mul(size_of::<i64>())
        .ok_or_else(|| p6_error("P6 tid2eid row byte count overflow"))?;
    if metadata.dtype != "I64" || metadata.bytes as usize % row_bytes != 0 {
        return Err(p6_error(
            "P6 tid2eid source tensor is not a complete I64[*,6] table",
        ));
    }
    let row_count = metadata.bytes as usize / row_bytes;
    let token_row = usize::try_from(controls.token_id)
        .map_err(|_| p6_error("P6 token ID does not fit source route-table index"))?;
    if token_row >= row_count {
        return Err(p6_error(format!(
            "P6 token {} exceeds tid2eid table row count {row_count}",
            controls.token_id
        )));
    }
    let tid2eid_bytes = reader.read_verified_full(&tid2eid_name, metadata.bytes as usize)?;
    if tid2eid_bytes.len() != metadata.bytes as usize {
        return Err(p6_error(
            "P6 tid2eid source read returned an unexpected length",
        ));
    }
    let start = token_row
        .checked_mul(row_bytes)
        .ok_or_else(|| p6_error("P6 tid2eid row offset overflow"))?;
    let row = &tid2eid_bytes[start..start + row_bytes];
    let mut selected = Vec::with_capacity(ACTIVATED_EXPERTS);
    for (slot, bytes) in row.chunks_exact(size_of::<i64>()).enumerate() {
        let expert = i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| p6_error("P6 tid2eid row has incomplete I64 entry"))?,
        );
        if !(0..ROUTED_EXPERTS as i64).contains(&expert) {
            return Err(p6_error(format!(
                "P6 tid2eid row selected out-of-range expert {expert}"
            )));
        }
        selected.push((
            u32::try_from(slot).map_err(|_| p6_error("P6 route slot exceeds u32"))?,
            u32::try_from(expert).map_err(|_| p6_error("P6 expert ID exceeds u32"))?,
        ));
    }
    let top_slot_ids: [u32; ACTIVATED_EXPERTS] = selected
        .iter()
        .map(|(_, expert)| *expert)
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| p6_error("P6 tid2eid row did not yield six IDs"))?;
    selected.sort_unstable_by_key(|(slot, expert)| (*expert, *slot));
    if selected.windows(2).any(|pair| pair[0].1 == pair[1].1) {
        return Err(p6_error(
            "P6 tid2eid row has duplicate expert IDs and cannot form six independent waves",
        ));
    }
    Ok((tid2eid_bytes, top_slot_ids, selected))
}

fn upload_cached_fp4(
    metal: &MetalContext,
    bundle: &CachedExpertBundle,
    operator: ExpertOperator,
    expected: PairGeometry,
) -> Result<(NativeFp4Gpu, String)> {
    let descriptor = bundle.descriptor().operator(operator);
    let (weight, scale) = bundle
        .operator_payload(operator)
        .ok_or_else(|| p6_error("P6 source cache bundle is missing a required FP4 operator"))?;
    validate_pair_geometry(
        descriptor.representation,
        descriptor.out_rows,
        descriptor.logical_k,
        descriptor.packed_k,
        descriptor.scale_rows,
        descriptor.scale_cols,
        descriptor.weight_bytes,
        descriptor.scale_bytes,
        expected,
        &descriptor.weight_name,
    )?;
    if weight.len() as u64 != descriptor.weight_bytes
        || scale.len() as u64 != descriptor.scale_bytes
    {
        return Err(p6_error(
            "P6 cached FP4 payload length differs from its source descriptor",
        ));
    }
    Ok((
        NativeFp4Gpu {
            weight: metal.new_buffer_with_bytes_checked(weight)?,
            scale: metal.new_buffer_with_bytes_checked(scale)?,
            rows: u32::try_from(expected.rows).map_err(|_| p6_error("P6 FP4 rows exceed u32"))?,
            packed_k: u32::try_from(expected.packed_k)
                .map_err(|_| p6_error("P6 FP4 packed K exceeds u32"))?,
            scale_cols: u32::try_from(expected.scale_cols)
                .map_err(|_| p6_error("P6 FP4 scale columns exceed u32"))?,
        },
        descriptor.weight_name.clone(),
    ))
}

fn upload_verified_fp8(
    metal: &MetalContext,
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    expected: PairGeometry,
) -> Result<(NativeFp8Gpu, String)> {
    let pair = reader.native_scale_pair(weight_name)?;
    validate_native_pair(&pair, expected, weight_name)?;
    let weight = reader.read_verified_full(weight_name, pair.weight.bytes as usize)?;
    let scale = reader.read_verified_full(&pair.scale.name, pair.scale.bytes as usize)?;
    if weight.len() != pair.weight.bytes as usize || scale.len() != pair.scale.bytes as usize {
        return Err(p6_error(
            "P6 shared FP8 source read returned an unexpected length",
        ));
    }
    Ok((
        NativeFp8Gpu {
            weight: metal.new_buffer_with_bytes_checked(&weight)?,
            scale: metal.new_buffer_with_bytes_checked(&scale)?,
            rows: u32::try_from(expected.rows).map_err(|_| p6_error("P6 FP8 rows exceed u32"))?,
            logical_k: u32::try_from(expected.logical_k)
                .map_err(|_| p6_error("P6 FP8 logical K exceeds u32"))?,
            scale_cols: u32::try_from(expected.scale_cols)
                .map_err(|_| p6_error("P6 FP8 scale columns exceed u32"))?,
        },
        weight_name.to_owned(),
    ))
}

fn validate_native_pair(
    pair: &NativeScalePair<'_>,
    expected: PairGeometry,
    weight_name: &str,
) -> Result<()> {
    let expected_scale_name = weight_name
        .strip_suffix(".weight")
        .ok_or_else(|| p6_error("P6 native pair weight name lacks .weight suffix"))?
        .to_owned()
        + ".scale";
    validate_pair_geometry(
        pair.kind,
        pair.out_rows,
        pair.logical_k,
        pair.packed_k,
        pair.scale_rows,
        pair.scale_cols,
        pair.weight.bytes,
        pair.scale.bytes,
        expected,
        weight_name,
    )?;
    if pair.weight.name != weight_name
        || pair.scale.name != expected_scale_name
        || pair.weight.source_shard != pair.scale.source_shard
    {
        return Err(p6_error(format!(
            "P6 native source pair naming/shard contract failed for {weight_name}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_pair_geometry(
    kind: NativeScalePairKind,
    rows: u64,
    logical_k: u64,
    packed_k: u64,
    scale_rows: u64,
    scale_cols: u64,
    weight_bytes: u64,
    scale_bytes: u64,
    expected: PairGeometry,
    label: &str,
) -> Result<()> {
    let expected_weight_bytes = expected
        .rows
        .checked_mul(expected.packed_k)
        .ok_or_else(|| p6_error("P6 native pair expected weight byte overflow"))?;
    let expected_scale_bytes = expected
        .scale_rows
        .checked_mul(expected.scale_cols)
        .ok_or_else(|| p6_error("P6 native pair expected scale byte overflow"))?;
    if kind != expected.kind
        || rows != expected.rows as u64
        || logical_k != expected.logical_k as u64
        || packed_k != expected.packed_k as u64
        || scale_rows != expected.scale_rows as u64
        || scale_cols != expected.scale_cols as u64
        || weight_bytes != expected_weight_bytes as u64
        || scale_bytes != expected_scale_bytes as u64
    {
        return Err(p6_error(format!(
            "P6 native pair geometry does not match its source contract: {label}"
        )));
    }
    Ok(())
}

fn allocate_routed_expert(
    metal: &MetalContext,
    source_top_slot: u32,
    w1: NativeFp4Gpu,
    w3: NativeFp4Gpu,
    w2: NativeFp4Gpu,
) -> Result<RoutedExpertGpu> {
    let fused_gate_up = crate::env_on(P6_FP4_GATE_UP_SWIGLU_FUSED_ENV);
    let fused_down = crate::env_on(P6_FP4_DOWN_BF16_FUSED_ENV);
    let fused_full_down = fused_down && crate::env_on(P6_FP4_DOWN_SHARED_COMBINE_FUSED_ENV);
    Ok(RoutedExpertGpu {
        source_top_slot,
        w1,
        w3,
        w2,
        gate_f32: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<f32>(), !fused_gate_up)?,
        up_f32: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<f32>(), !fused_gate_up)?,
        gate_bf16: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<u16>(), !fused_gate_up)?,
        up_bf16: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<u16>(), !fused_gate_up)?,
        swiglu_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        down_quant: metal.new_buffer_checked(MOE_INTER_DIM)?,
        down_scales: metal.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
        down_f32: allocate_p6_scratch(metal, HIDDEN_SIZE * size_of::<f32>(), !fused_down)?,
        down_bf16: allocate_p6_scratch(metal, HIDDEN_BF16_BYTES, !fused_full_down)?,
    })
}

fn prepare_fused_gate_up_refs(
    metal: &MetalContext,
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    ready: bool,
) -> Result<(metal::Buffer, Vec<metal::Buffer>, Vec<metal::Buffer>)> {
    let refs = fused_gate_up_ref_records(routed, ready);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            refs.as_ptr() as *const u8,
            ACTIVATED_EXPERTS * size_of::<P6Fp4GateUpRef>(),
        )
    };
    let ref_buffer = metal.new_buffer_with_bytes_checked(bytes)?;
    let (weight_resources, output_resources) = fused_gate_up_resources(routed);
    Ok((ref_buffer, weight_resources, output_resources))
}

fn fused_gate_up_ref_records(
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    ready: bool,
) -> [P6Fp4GateUpRef; ACTIVATED_EXPERTS] {
    std::array::from_fn(|slot| {
        let expert = &routed[slot];
        P6Fp4GateUpRef {
            gate_weights: expert.w1.weight.gpu_address(),
            gate_scales: expert.w1.scale.gpu_address(),
            up_weights: expert.w3.weight.gpu_address(),
            up_scales: expert.w3.scale.gpu_address(),
            output_bf16: expert.swiglu_bf16.gpu_address(),
            route_slot: expert.source_top_slot,
            ready: u32::from(ready),
        }
    })
}

fn fused_gate_up_resources(
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
) -> (Vec<metal::Buffer>, Vec<metal::Buffer>) {
    let mut weight_resources = Vec::with_capacity(ACTIVATED_EXPERTS * 4);
    let mut output_resources = Vec::with_capacity(ACTIVATED_EXPERTS);
    for expert in routed {
        weight_resources.extend([
            expert.w1.weight.clone(),
            expert.w1.scale.clone(),
            expert.w3.weight.clone(),
            expert.w3.scale.clone(),
        ]);
        output_resources.push(expert.swiglu_bf16.clone());
    }
    (weight_resources, output_resources)
}

fn write_fused_gate_up_refs(
    buffer: &metal::Buffer,
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    ready: bool,
) -> Result<()> {
    let refs = fused_gate_up_ref_records(routed, ready);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            refs.as_ptr() as *const u8,
            ACTIVATED_EXPERTS * size_of::<P6Fp4GateUpRef>(),
        )
    };
    let ptr = buffer.contents() as *mut u8;
    if ptr.is_null() {
        return Err(p6_error("P6 fused gate/up ref buffer contents() is null"));
    }
    unsafe {
        ptr.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
    }
    Ok(())
}

fn prepare_fused_down_refs(
    metal: &MetalContext,
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    ready: bool,
) -> Result<(metal::Buffer, Vec<metal::Buffer>, Vec<metal::Buffer>)> {
    let refs = fused_down_ref_records(routed, ready);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            refs.as_ptr() as *const u8,
            ACTIVATED_EXPERTS * size_of::<P6Fp4DownRef>(),
        )
    };
    let ref_buffer = metal.new_buffer_with_bytes_checked(bytes)?;
    let (resources, output_resources) = fused_down_resources(routed);
    Ok((ref_buffer, resources, output_resources))
}

fn fused_down_ref_records(
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    ready: bool,
) -> [P6Fp4DownRef; ACTIVATED_EXPERTS] {
    std::array::from_fn(|slot| {
        let expert = &routed[slot];
        P6Fp4DownRef {
            weights: expert.w2.weight.gpu_address(),
            weight_scales: expert.w2.scale.gpu_address(),
            activations: expert.down_quant.gpu_address(),
            activation_scales: expert.down_scales.gpu_address(),
            output_bf16: expert.down_bf16.gpu_address(),
            ready: u32::from(ready),
            reserved: 0,
        }
    })
}

fn fused_down_resources(
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
) -> (Vec<metal::Buffer>, Vec<metal::Buffer>) {
    let mut resources = Vec::with_capacity(ACTIVATED_EXPERTS * 4);
    let mut output_resources = Vec::with_capacity(ACTIVATED_EXPERTS);
    for expert in routed {
        resources.extend([
            expert.w2.weight.clone(),
            expert.w2.scale.clone(),
            expert.down_quant.clone(),
            expert.down_scales.clone(),
        ]);
        output_resources.push(expert.down_bf16.clone());
    }
    (resources, output_resources)
}

fn write_fused_down_refs(
    buffer: &metal::Buffer,
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    ready: bool,
) -> Result<()> {
    let refs = fused_down_ref_records(routed, ready);
    let bytes = unsafe {
        std::slice::from_raw_parts(
            refs.as_ptr() as *const u8,
            ACTIVATED_EXPERTS * size_of::<P6Fp4DownRef>(),
        )
    };
    let ptr = buffer.contents() as *mut u8;
    if ptr.is_null() {
        return Err(p6_error("P6 fused down ref buffer contents() is null"));
    }
    unsafe {
        ptr.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
    }
    Ok(())
}

fn prepare_batched_down_qat_refs(
    metal: &MetalContext,
    routed: &[RoutedExpertGpu; ACTIVATED_EXPERTS],
    shared: &SharedExpertGpu,
) -> Result<(metal::Buffer, Vec<metal::Buffer>)> {
    let mut refs = Vec::with_capacity(P6_BATCHED_DOWN_QAT_TENSORS as usize);
    let mut inputs = Vec::with_capacity(P6_BATCHED_DOWN_QAT_TENSORS as usize);
    let mut quantized = Vec::with_capacity(P6_BATCHED_DOWN_QAT_TENSORS as usize);
    let mut scales = Vec::with_capacity(P6_BATCHED_DOWN_QAT_TENSORS as usize);
    for expert in routed {
        refs.push(P6ActQuantRef {
            input_bf16: expert.swiglu_bf16.gpu_address(),
            quantized: expert.down_quant.gpu_address(),
            act_scales: expert.down_scales.gpu_address(),
            ready: 1,
            reserved: 0,
        });
        inputs.push(expert.swiglu_bf16.clone());
        quantized.push(expert.down_quant.clone());
        scales.push(expert.down_scales.clone());
    }
    refs.push(P6ActQuantRef {
        input_bf16: shared.swiglu_bf16.gpu_address(),
        quantized: shared.down_quant.gpu_address(),
        act_scales: shared.down_scales.gpu_address(),
        ready: 1,
        reserved: 0,
    });
    inputs.push(shared.swiglu_bf16.clone());
    quantized.push(shared.down_quant.clone());
    scales.push(shared.down_scales.clone());
    debug_assert_eq!(refs.len(), P6_BATCHED_DOWN_QAT_TENSORS as usize);

    let bytes = unsafe {
        std::slice::from_raw_parts(
            refs.as_ptr() as *const u8,
            refs.len() * size_of::<P6ActQuantRef>(),
        )
    };
    let ref_buffer = metal.new_buffer_with_bytes_checked(bytes)?;
    let mut resources = Vec::with_capacity(refs.len() * 3);
    resources.extend(inputs);
    resources.extend(quantized);
    resources.extend(scales);
    Ok((ref_buffer, resources))
}

fn allocate_shared_expert(
    metal: &MetalContext,
    w1: NativeFp8Gpu,
    w3: NativeFp8Gpu,
    w2: NativeFp8Gpu,
) -> Result<SharedExpertGpu> {
    let fused_gate_up = crate::env_on(P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_ENV);
    let fused_full_down = crate::env_on(P6_FP4_DOWN_BF16_FUSED_ENV)
        && crate::env_on(P6_FP4_DOWN_SHARED_COMBINE_FUSED_ENV);
    let fused_down = fused_full_down
        || (!fused_full_down && crate::env_on(P6_SHARED_FP8_DOWN_COMBINE_FUSED_ENV));
    Ok(SharedExpertGpu {
        w1,
        w3,
        w2,
        gate_f32: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<f32>(), !fused_gate_up)?,
        up_f32: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<f32>(), !fused_gate_up)?,
        gate_bf16: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<u16>(), !fused_gate_up)?,
        up_bf16: allocate_p6_scratch(metal, MOE_INTER_DIM * size_of::<u16>(), !fused_gate_up)?,
        swiglu_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        down_quant: metal.new_buffer_checked(MOE_INTER_DIM)?,
        down_scales: metal.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
        down_f32: allocate_p6_scratch(metal, HIDDEN_SIZE * size_of::<f32>(), !fused_down)?,
        down_bf16: allocate_p6_scratch(metal, HIDDEN_BF16_BYTES, !fused_down)?,
    })
}

fn precompile_threads(
    metal: &MetalContext,
    fused_gate_up_swiglu: bool,
    fused_gate_up_swiglu_simd: bool,
    shared_fp8_gate_up_swiglu: bool,
    shared_fp8_down_combine: bool,
    batched_down_qat: bool,
    fused_down_bf16: bool,
    fused_down_bf16_simd: bool,
    fused_down_shared_combine: bool,
) -> Result<ThreadGeometry> {
    let pipeline_max = |kernel: &str| -> Result<u32> {
        Ok(metal.pipeline(kernel)?.max_total_threads_per_threadgroup() as u32)
    };
    // Both route kernels require a single-thread dispatch; admit either.
    let route = require_threads(pipeline_max(P6A_ROUTE_KERNEL)?, 1, P6A_ROUTE_KERNEL)?;
    let _ = require_threads(
        pipeline_max(P6A_LEARNED_ROUTE_KERNEL)?,
        1,
        P6A_LEARNED_ROUTE_KERNEL,
    )?;
    let qat_simd = crate::env_on(P6_ACT_QUANT_SIMD_ENV);
    let qat_kernel = if qat_simd {
        ACT_QUANT_SIMD_KERNEL
    } else {
        ACT_QUANT_KERNEL
    };
    let qat = require_threads(
        pipeline_max(qat_kernel)?,
        if qat_simd { 256 } else { 32 },
        qat_kernel,
    )?;
    let fp4_simd = crate::env_on(P6_FP4_SIMD_ENV);
    let fp4_kernel = if fp4_simd {
        FP4_SIMD_KERNEL
    } else {
        P5B_FP4_KERNEL
    };
    let fp4 = require_threads(pipeline_max(fp4_kernel)?, 256, fp4_kernel)?;
    let fp8_simd = crate::env_on(P6_FP8_SIMD_ENV);
    let fp8_kernel = if fp8_simd {
        FP8_SIMD_KERNEL
    } else {
        FP8_KERNEL
    };
    let fp8_threads = if fp8_simd {
        FP8_SIMD_THREADS_X * FP8_SIMD_ROWS_PER_TG
    } else {
        256
    };
    let fp8 = require_threads(pipeline_max(fp8_kernel)?, fp8_threads, fp8_kernel)?;
    let (
        fused_gate_up_swiglu_kernel,
        fused_gate_up_swiglu_threads,
        fused_gate_up_swiglu_threads_x,
        fused_gate_up_swiglu_rows_per_tg,
    ) = if fused_gate_up_swiglu {
        let kernel = if fused_gate_up_swiglu_simd {
            P6_FP4_GATE_UP_SWIGLU_SIMD_KERNEL
        } else {
            P6_FP4_GATE_UP_SWIGLU_KERNEL
        };
        let threads = require_threads(pipeline_max(kernel)?, 256, kernel)?;
        (
            kernel,
            threads,
            if fused_gate_up_swiglu_simd {
                P6_FP4_GATE_UP_SWIGLU_SIMD_THREADS_X
            } else {
                0
            },
            if fused_gate_up_swiglu_simd {
                P6_FP4_GATE_UP_SWIGLU_SIMD_ROWS_PER_TG
            } else {
                1
            },
        )
    } else {
        (P6_FP4_GATE_UP_SWIGLU_KERNEL, 0, 0, 0)
    };
    let (
        fused_down_bf16_kernel,
        fused_down_bf16_threads,
        fused_down_bf16_threads_x,
        fused_down_bf16_rows_per_tg,
    ) = if fused_down_shared_combine {
        let kernel = P6_FP4_DOWN_SHARED_COMBINE_FUSED_KERNEL;
        let threads = require_threads(pipeline_max(kernel)?, 256, kernel)?;
        (kernel, threads, 0, 1)
    } else if fused_down_bf16 {
        let kernel = if fused_down_bf16_simd {
            P6_FP4_DOWN_BF16_SIMD_KERNEL
        } else {
            P6_FP4_DOWN_BF16_KERNEL
        };
        let threads = require_threads(pipeline_max(kernel)?, 256, kernel)?;
        (
            kernel,
            threads,
            if fused_down_bf16_simd {
                P6_FP4_DOWN_BF16_SIMD_THREADS_X
            } else {
                0
            },
            if fused_down_bf16_simd {
                P6_FP4_DOWN_BF16_SIMD_ROWS_PER_TG
            } else {
                1
            },
        )
    } else {
        (P6_FP4_DOWN_BF16_KERNEL, 0, 0, 0)
    };
    let shared_fp8_gate_up_swiglu_threads = if shared_fp8_gate_up_swiglu {
        require_threads(
            pipeline_max(P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_KERNEL)?,
            256,
            P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_KERNEL,
        )?
    } else {
        0
    };
    let shared_fp8_down_combine_threads = if shared_fp8_down_combine {
        require_threads(
            pipeline_max(P6_SHARED_FP8_DOWN_COMBINE_FUSED_KERNEL)?,
            256,
            P6_SHARED_FP8_DOWN_COMBINE_FUSED_KERNEL,
        )?
    } else {
        0
    };
    let batched_down_qat_threads = if batched_down_qat {
        require_threads(
            pipeline_max(P6_BATCHED_DOWN_QAT_KERNEL)?,
            32,
            P6_BATCHED_DOWN_QAT_KERNEL,
        )?
    } else {
        0
    };
    Ok(ThreadGeometry {
        qat,
        qat_kernel,
        qat_vector_width: if qat_simd {
            ACT_QUANT_SIMD_VECTOR_WIDTH
        } else {
            0
        },
        batched_down_qat,
        batched_down_qat_threads,
        fp4,
        fp4_kernel,
        fp4_threads_x: if fp4_simd { FP4_SIMD_THREADS_X } else { 0 },
        fp4_rows_per_tg: if fp4_simd { FP4_SIMD_ROWS_PER_TG } else { 1 },
        fp8,
        fp8_kernel,
        fp8_threads_x: if fp8_simd { FP8_SIMD_THREADS_X } else { 0 },
        fp8_rows_per_tg: if fp8_simd { FP8_SIMD_ROWS_PER_TG } else { 1 },
        cast: require_threads(pipeline_max(BF16_CAST_KERNEL)?, 256, BF16_CAST_KERNEL)?,
        gate: require_exact_simdgroup_threads(metal)?,
        route,
        routed_swiglu: require_threads(pipeline_max(P6A_SWIGLU_KERNEL)?, 256, P6A_SWIGLU_KERNEL)?,
        shared_swiglu: require_threads(pipeline_max(P5B_SWIGLU_KERNEL)?, 256, P5B_SWIGLU_KERNEL)?,
        shared_fp8_gate_up_swiglu,
        shared_fp8_gate_up_swiglu_threads,
        shared_fp8_down_combine,
        shared_fp8_down_combine_threads,
        combine: require_threads(pipeline_max(P6A_COMBINE_KERNEL)?, 256, P6A_COMBINE_KERNEL)?,
        fused_gate_up_swiglu,
        fused_gate_up_swiglu_kernel,
        fused_gate_up_swiglu_threads,
        fused_gate_up_swiglu_threads_x,
        fused_gate_up_swiglu_rows_per_tg,
        fused_down_bf16,
        fused_down_bf16_kernel,
        fused_down_bf16_threads,
        fused_down_bf16_threads_x,
        fused_down_bf16_rows_per_tg,
        fused_down_shared_combine,
        fused_down_shared_combine_threads: if fused_down_shared_combine {
            fused_down_bf16_threads
        } else {
            0
        },
    })
}

fn load_gate_weight(
    reader: &DeepSeekV4FullStreamReader,
    layer: usize,
) -> Result<(String, Vec<u8>)> {
    let gate_weight_name = format!("layers.{layer}.ffn.gate.weight");
    let gate_metadata = reader.tensor_metadata(&gate_weight_name)?;
    if gate_metadata.dtype != "BF16"
        || gate_metadata.shape.as_slice() != [ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64]
        || gate_metadata.bytes != (ROUTED_EXPERTS * HIDDEN_BF16_BYTES) as u64
    {
        return Err(p6_error(
            "P6 Gate source tensor geometry is not BF16[256,4096]",
        ));
    }
    let gate_bytes = reader.read_verified_full(&gate_weight_name, gate_metadata.bytes as usize)?;
    if gate_bytes.len() != gate_metadata.bytes as usize {
        return Err(p6_error(
            "P6 Gate source read returned an unexpected length",
        ));
    }
    Ok((gate_weight_name, gate_bytes))
}

fn prepare_shared_expert(
    metal: &MetalContext,
    reader: &DeepSeekV4FullStreamReader,
    layer: usize,
) -> Result<(SharedExpertGpu, (String, String, String))> {
    let shared_stem = format!("layers.{layer}.ffn.shared_experts");
    let (shared_w1, shared_w1_weight_name) = upload_verified_fp8(
        metal,
        reader,
        &format!("{shared_stem}.w1.weight"),
        FP8_W1_W3,
    )?;
    let (shared_w3, shared_w3_weight_name) = upload_verified_fp8(
        metal,
        reader,
        &format!("{shared_stem}.w3.weight"),
        FP8_W1_W3,
    )?;
    let (shared_w2, shared_w2_weight_name) =
        upload_verified_fp8(metal, reader, &format!("{shared_stem}.w2.weight"), FP8_W2)?;
    Ok((
        allocate_shared_expert(metal, shared_w1, shared_w3, shared_w2)?,
        (
            shared_w1_weight_name,
            shared_w3_weight_name,
            shared_w2_weight_name,
        ),
    ))
}

fn empty_fp4(metal: &MetalContext, geom: PairGeometry) -> Result<NativeFp4Gpu> {
    let _weight_bytes = geom
        .rows
        .checked_mul(geom.packed_k)
        .ok_or_else(|| p6_error("P6 empty FP4 weight size overflow"))?;
    let _scale_bytes = geom
        .scale_rows
        .checked_mul(geom.scale_cols)
        .ok_or_else(|| p6_error("P6 empty FP4 scale size overflow"))?;
    // Learned-bias preparation cannot know the six expert IDs until after the
    // device route and the host ID-only residency boundary.  Do not reserve
    // six full FP4 triplets merely to populate pointers that are guaranteed to
    // be replaced before any expert dispatch.  Geometry stays authoritative;
    // only the dormant storage is reduced to a guarded non-null placeholder.
    Ok(NativeFp4Gpu {
        weight: metal.new_buffer_checked(P6_DORMANT_BUFFER_BYTES)?,
        scale: metal.new_buffer_checked(P6_DORMANT_BUFFER_BYTES)?,
        rows: u32::try_from(geom.rows).map_err(|_| p6_error("P6 FP4 rows exceed u32"))?,
        packed_k: u32::try_from(geom.packed_k)
            .map_err(|_| p6_error("P6 FP4 packed K exceeds u32"))?,
        scale_cols: u32::try_from(geom.scale_cols)
            .map_err(|_| p6_error("P6 FP4 scale columns exceed u32"))?,
    })
}

fn read_u32_buffer(buf: &metal::Buffer, n: usize) -> Result<[u32; ACTIVATED_EXPERTS]> {
    // Specialized for ACTIVATED_EXPERTS-sized reads and single-element valid.
    if n == ACTIVATED_EXPERTS {
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return Err(p6_error("P6 route id buffer contents() is null"));
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, n * 4) };
        let mut out = [0u32; ACTIVATED_EXPERTS];
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = u32::from_le_bytes(
                bytes[i * 4..i * 4 + 4]
                    .try_into()
                    .map_err(|_| p6_error("P6 route id byte slice"))?,
            );
        }
        return Ok(out);
    }
    if n == 1 {
        let ptr = buf.contents() as *const u8;
        if ptr.is_null() {
            return Err(p6_error("P6 route valid buffer contents() is null"));
        }
        let bytes = unsafe { std::slice::from_raw_parts(ptr, 4) };
        let mut out = [0u32; ACTIVATED_EXPERTS];
        out[0] = u32::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| p6_error("P6 route valid byte slice"))?,
        );
        return Ok(out);
    }
    Err(p6_error("P6 read_u32_buffer supports n=1 or n=6 only"))
}

fn require_exact_simdgroup_threads(metal: &MetalContext) -> Result<u32> {
    let pipeline = metal.pipeline(P6_C4_GATE_KERNEL)?;
    let execution_width = pipeline.thread_execution_width();
    if execution_width != u64::from(P6_C4_GATE_SIMDGROUP_THREADS) {
        return Err(p6_error(format!(
            "P6 C4 Gate kernel {P6_C4_GATE_KERNEL} requires execution width {}, got {execution_width}",
            P6_C4_GATE_SIMDGROUP_THREADS,
        )));
    }
    let max_total_threads =
        u32::try_from(pipeline.max_total_threads_per_threadgroup()).map_err(|_| {
            p6_error(format!(
                "P6 C4 Gate kernel {P6_C4_GATE_KERNEL} reports a non-u32 threadgroup limit"
            ))
        })?;
    require_threads(
        max_total_threads,
        P6_C4_GATE_SIMDGROUP_THREADS,
        P6_C4_GATE_KERNEL,
    )
}

fn require_threads(max: u32, preferred: u32, kernel: &str) -> Result<u32> {
    if max < preferred {
        return Err(p6_error(format!(
            "P6 kernel {kernel} supports only {max} threads, below required {preferred}"
        )));
    }
    Ok(preferred)
}

fn dispatch_gate(
    batch: &mut CommandBatch<'_>,
    gate_weight: &metal::Buffer,
    input: &metal::Buffer,
    logits: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    if threads != P6_C4_GATE_SIMDGROUP_THREADS {
        return Err(p6_error(format!(
            "P6 C4 Gate dispatch requires exactly {} threads per threadgroup, got {threads}",
            P6_C4_GATE_SIMDGROUP_THREADS,
        )));
    }
    batch.dispatch_threads(
        P6_C4_GATE_KERNEL,
        (P6_C4_GATE_GRID_THREADS, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(gate_weight), 0);
            encoder.set_buffer(1, Some(input), 0);
            encoder.set_buffer(2, Some(logits), 0);
            set_u32(encoder, 3, &(ROUTED_EXPERTS as u32));
            set_u32(encoder, 4, &(HIDDEN_SIZE as u32));
        },
    )
}

fn dispatch_gate_concurrent(
    batch: &mut CommandBatch<'_>,
    gate_weight: &metal::Buffer,
    input: &metal::Buffer,
    logits: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    if threads != P6_C4_GATE_SIMDGROUP_THREADS {
        return Err(p6_error(format!(
            "P6 C4 Gate dispatch requires exactly {} threads per threadgroup, got {threads}",
            P6_C4_GATE_SIMDGROUP_THREADS,
        )));
    }
    batch.dispatch_threads_in_concurrent_group(
        P6_C4_GATE_KERNEL,
        (P6_C4_GATE_GRID_THREADS, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(gate_weight), 0);
            encoder.set_buffer(1, Some(input), 0);
            encoder.set_buffer(2, Some(logits), 0);
            set_u32(encoder, 3, &(ROUTED_EXPERTS as u32));
            set_u32(encoder, 4, &(HIDDEN_SIZE as u32));
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_hash_route(
    batch: &mut CommandBatch<'_>,
    logits: &metal::Buffer,
    tid2eid: &metal::Buffer,
    ids: &metal::Buffer,
    weights: &metal::Buffer,
    scores: &metal::Buffer,
    valid: &metal::Buffer,
    token_id: u32,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads(P6A_ROUTE_KERNEL, (1, 1, 1), (threads, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(logits), 0);
        encoder.set_buffer(1, Some(tid2eid), 0);
        encoder.set_buffer(2, Some(ids), 0);
        encoder.set_buffer(3, Some(weights), 0);
        encoder.set_buffer(4, Some(scores), 0);
        encoder.set_buffer(5, Some(valid), 0);
        set_u32(encoder, 6, &token_id);
        set_u32(encoder, 7, &(ROUTED_EXPERTS as u32));
        set_u32(encoder, 8, &(ACTIVATED_EXPERTS as u32));
        encoder.set_bytes(
            9,
            size_of::<f32>() as u64,
            &ROUTE_SCALE as *const f32 as *const _,
        );
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_learned_route(
    batch: &mut CommandBatch<'_>,
    logits: &metal::Buffer,
    bias: &metal::Buffer,
    ids: &metal::Buffer,
    weights: &metal::Buffer,
    scores: &metal::Buffer,
    valid: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads(
        P6A_LEARNED_ROUTE_KERNEL,
        (1, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(logits), 0);
            encoder.set_buffer(1, Some(bias), 0);
            encoder.set_buffer(2, Some(ids), 0);
            encoder.set_buffer(3, Some(weights), 0);
            encoder.set_buffer(4, Some(scores), 0);
            encoder.set_buffer(5, Some(valid), 0);
            set_u32(encoder, 6, &(ROUTED_EXPERTS as u32));
            set_u32(encoder, 7, &(ACTIVATED_EXPERTS as u32));
            encoder.set_bytes(
                8,
                size_of::<f32>() as u64,
                &ROUTE_SCALE as *const f32 as *const _,
            );
        },
    )
}

fn dispatch_act_quant_ordered(
    batch: &mut CommandBatch<'_>,
    input_bf16: &metal::Buffer,
    quantized: &metal::Buffer,
    scales: &metal::Buffer,
    cols: u32,
    threads: u32,
    kernel: &'static str,
    vector_width: u32,
) -> Result<()> {
    if kernel == ACT_QUANT_SIMD_KERNEL {
        let blocks = cols / ACT_QUANT_BLOCK as u32;
        let simdgroups = (threads / ACT_QUANT_SIMD_WIDTH).max(1);
        let groups = blocks.div_ceil(simdgroups);
        batch.dispatch_threads(
            kernel,
            (groups * threads, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_bf16), 0);
                encoder.set_buffer(1, Some(quantized), 0);
                encoder.set_buffer(2, Some(scales), 0);
                set_u32(encoder, 3, &cols);
                set_u32(encoder, 4, &threads);
                set_u32(encoder, 5, &vector_width);
            },
        )
    } else {
        batch.dispatch_threads(
            kernel,
            (cols / ACT_QUANT_BLOCK as u32, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input_bf16), 0);
                encoder.set_buffer(1, Some(quantized), 0);
                encoder.set_buffer(2, Some(scales), 0);
                set_u32(encoder, 3, &cols);
            },
        )
    }
}

fn dispatch_act_quant_concurrent(
    batch: &mut CommandBatch<'_>,
    input_bf16: &metal::Buffer,
    quantized: &metal::Buffer,
    scales: &metal::Buffer,
    cols: u32,
    threads: u32,
    kernel: &'static str,
    vector_width: u32,
) -> Result<()> {
    let (grid, tg) = if kernel == ACT_QUANT_SIMD_KERNEL {
        let blocks = cols / ACT_QUANT_BLOCK as u32;
        let simdgroups = (threads / ACT_QUANT_SIMD_WIDTH).max(1);
        (blocks.div_ceil(simdgroups) * threads, threads)
    } else {
        (cols / ACT_QUANT_BLOCK as u32, threads)
    };
    batch.dispatch_threads_in_concurrent_group(kernel, (grid, 1, 1), (tg, 1, 1), |encoder| {
        encoder.set_buffer(0, Some(input_bf16), 0);
        encoder.set_buffer(1, Some(quantized), 0);
        encoder.set_buffer(2, Some(scales), 0);
        set_u32(encoder, 3, &cols);
        if kernel == ACT_QUANT_SIMD_KERNEL {
            set_u32(encoder, 4, &threads);
            set_u32(encoder, 5, &vector_width);
        }
    })
}

fn dispatch_batched_down_qat(
    batch: &mut CommandBatch<'_>,
    refs: &metal::Buffer,
    resources: &[metal::Buffer],
    cols: u32,
    threads: u32,
) -> Result<()> {
    if cols == 0
        || cols % ACT_QUANT_BLOCK as u32 != 0
        || threads == 0
        || resources.len() != P6_BATCHED_DOWN_QAT_TENSORS as usize * 3
    {
        return Err(p6_error(
            "P6 fixed-seven down-QAT geometry or resources are invalid",
        ));
    }
    let blocks = cols / ACT_QUANT_BLOCK as u32;
    let grid = P6_BATCHED_DOWN_QAT_TENSORS * blocks;
    batch.dispatch_threads_in_active_group(
        P6_BATCHED_DOWN_QAT_KERNEL,
        (grid, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(refs), 0);
            set_u32(encoder, 1, &cols);
            set_u32(encoder, 2, &P6_BATCHED_DOWN_QAT_TENSORS);
            let input_refs: [&metal::ResourceRef; P6_BATCHED_DOWN_QAT_TENSORS as usize] =
                std::array::from_fn(|index| &**resources[index]);
            encoder.use_resources(&input_refs, metal::MTLResourceUsage::Read);
            let quantized_refs: [&metal::ResourceRef; P6_BATCHED_DOWN_QAT_TENSORS as usize] =
                std::array::from_fn(|index| {
                    &**resources[P6_BATCHED_DOWN_QAT_TENSORS as usize + index]
                });
            encoder.use_resources(&quantized_refs, metal::MTLResourceUsage::Write);
            let scale_refs: [&metal::ResourceRef; P6_BATCHED_DOWN_QAT_TENSORS as usize] =
                std::array::from_fn(|index| {
                    &**resources[P6_BATCHED_DOWN_QAT_TENSORS as usize * 2 + index]
                });
            encoder.use_resources(&scale_refs, metal::MTLResourceUsage::Write);
        },
    )
}

fn dispatch_fp4(
    batch: &mut CommandBatch<'_>,
    pair: &NativeFp4Gpu,
    activation: &metal::Buffer,
    activation_scales: &metal::Buffer,
    output: &metal::Buffer,
    threads: u32,
    kernel: &'static str,
    threads_x: u32,
    rows_per_tg: u32,
) -> Result<()> {
    if kernel == FP4_SIMD_KERNEL {
        if threads_x == 0
            || rows_per_tg == 0
            || threads_x * rows_per_tg != threads
            || pair.rows % rows_per_tg != 0
            || pair.packed_k % 4 != 0
        {
            return Err(p6_error("P6 FP4 SIMD candidate geometry is not divisible"));
        }
        batch.dispatch_threads_in_concurrent_group(
            kernel,
            (threads_x, pair.rows, 1),
            (threads_x, rows_per_tg, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&pair.weight), 0);
                encoder.set_buffer(1, Some(&pair.scale), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output), 0);
                set_u32(encoder, 5, &pair.rows);
                set_u32(encoder, 6, &pair.packed_k);
                set_u32(encoder, 7, &pair.scale_cols);
                set_u32(encoder, 8, &threads_x);
            },
        )
    } else {
        batch.dispatch_threads_in_concurrent_group(
            kernel,
            (pair.rows, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&pair.weight), 0);
                encoder.set_buffer(1, Some(&pair.scale), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output), 0);
                set_u32(encoder, 5, &pair.rows);
                set_u32(encoder, 6, &pair.packed_k);
                set_u32(encoder, 7, &pair.scale_cols);
            },
        )
    }
}

fn dispatch_fused_gate_up_swiglu(
    batch: &mut CommandBatch<'_>,
    refs: &metal::Buffer,
    weight_resources: &[metal::Buffer],
    output_resources: &[metal::Buffer],
    quantized: &metal::Buffer,
    act_scales: &metal::Buffer,
    route_weights: &metal::Buffer,
    rows: u32,
    packed_cols: u32,
    scale_cols: u32,
    top_k: u32,
    kernel: &'static str,
    threads: u32,
    threads_x: u32,
    rows_per_tg: u32,
) -> Result<()> {
    let (grid, tg) = if kernel == P6_FP4_GATE_UP_SWIGLU_SIMD_KERNEL {
        if threads_x == 0
            || rows_per_tg == 0
            || threads_x != threads
            || rows % rows_per_tg != 0
            || (threads_x & 31) != 0
        {
            return Err(p6_error(
                "P6 fused gate/up/SwiGLU SIMD geometry is not divisible",
            ));
        }
        let groups_per_slot = rows.div_ceil(rows_per_tg);
        (top_k * groups_per_slot * threads_x, (threads_x, 1, 1))
    } else {
        if threads == 0 || top_k == 0 || rows == 0 {
            return Err(p6_error(
                "P6 fused gate/up/SwiGLU authority geometry is empty",
            ));
        }
        (top_k * rows, (threads, 1, 1))
    };
    if weight_resources.len() != ACTIVATED_EXPERTS * 4
        || output_resources.len() != ACTIVATED_EXPERTS
    {
        return Err(p6_error(
            "P6 fused gate/up/SwiGLU indirect resource count is not fixed-six",
        ));
    }
    batch.dispatch_threads_in_concurrent_group(kernel, (grid, 1, 1), tg, |encoder| {
        encoder.set_buffer(0, Some(refs), 0);
        encoder.set_buffer(1, Some(quantized), 0);
        encoder.set_buffer(2, Some(act_scales), 0);
        encoder.set_buffer(3, Some(route_weights), 0);
        set_u32(encoder, 4, &rows);
        set_u32(encoder, 5, &packed_cols);
        set_u32(encoder, 6, &scale_cols);
        set_u32(encoder, 7, &top_k);
        let weight_refs: [&metal::ResourceRef; ACTIVATED_EXPERTS * 4] =
            std::array::from_fn(|index| &**weight_resources[index]);
        encoder.use_resources(&weight_refs, metal::MTLResourceUsage::Read);
        let output_refs: [&metal::ResourceRef; ACTIVATED_EXPERTS] =
            std::array::from_fn(|index| &**output_resources[index]);
        encoder.use_resources(&output_refs, metal::MTLResourceUsage::Write);
    })
}

fn dispatch_fused_down_bf16(
    batch: &mut CommandBatch<'_>,
    refs: &metal::Buffer,
    resources: &[metal::Buffer],
    output_resources: &[metal::Buffer],
    kernel: &'static str,
    threads: u32,
    threads_x: u32,
    rows_per_tg: u32,
) -> Result<()> {
    let (grid, tg) = if kernel == P6_FP4_DOWN_BF16_SIMD_KERNEL {
        if threads_x == 0
            || rows_per_tg != P6_FP4_DOWN_BF16_SIMD_ROWS_PER_TG
            || threads_x != threads
            || rows_per_tg == 0
            || HIDDEN_SIZE as u32 % rows_per_tg != 0
            || (threads_x & 31) != 0
        {
            return Err(p6_error(
                "P6 fused down FP4/BF16 SIMD geometry is not divisible",
            ));
        }
        let groups_per_slot = (HIDDEN_SIZE as u32).div_ceil(rows_per_tg);
        (
            ACTIVATED_EXPERTS as u32 * groups_per_slot * threads_x,
            (threads_x, 1, 1),
        )
    } else {
        if kernel != P6_FP4_DOWN_BF16_KERNEL || threads == 0 {
            return Err(p6_error(
                "P6 fused down FP4/BF16 kernel geometry is invalid",
            ));
        }
        (
            ACTIVATED_EXPERTS as u32 * HIDDEN_SIZE as u32,
            (threads, 1, 1),
        )
    };
    if resources.len() != ACTIVATED_EXPERTS * 4 || output_resources.len() != ACTIVATED_EXPERTS {
        return Err(p6_error(
            "P6 fused down indirect resource count is not fixed-six",
        ));
    }
    batch.dispatch_threads_in_concurrent_group(kernel, (grid, 1, 1), tg, |encoder| {
        encoder.set_buffer(0, Some(refs), 0);
        set_u32(encoder, 1, &(HIDDEN_SIZE as u32));
        set_u32(encoder, 2, &(MOE_INTER_DIM as u32 / 2));
        set_u32(encoder, 3, &(MOE_INTER_DIM as u32 / 32));
        let resource_refs: [&metal::ResourceRef; ACTIVATED_EXPERTS * 4] =
            std::array::from_fn(|index| &**resources[index]);
        encoder.use_resources(&resource_refs, metal::MTLResourceUsage::Read);
        let output_refs: [&metal::ResourceRef; ACTIVATED_EXPERTS] =
            std::array::from_fn(|index| &**output_resources[index]);
        encoder.use_resources(&output_refs, metal::MTLResourceUsage::Write);
    })
}

fn dispatch_fp8(
    batch: &mut CommandBatch<'_>,
    pair: &NativeFp8Gpu,
    activation: &metal::Buffer,
    activation_scales: &metal::Buffer,
    output: &metal::Buffer,
    threads: u32,
    kernel: &'static str,
    threads_x: u32,
    rows_per_tg: u32,
) -> Result<()> {
    if kernel == FP8_SIMD_KERNEL {
        if threads_x == 0
            || rows_per_tg != FP8_SIMD_ROWS_PER_TG
            || threads_x * rows_per_tg != threads
            || (threads_x & 31) != 0
            || pair.rows % rows_per_tg != 0
            || pair.logical_k % ACT_QUANT_BLOCK as u32 != 0
            || pair.scale_cols > 32
        {
            return Err(p6_error("P6 FP8 SIMD candidate geometry is not divisible"));
        }
        batch.dispatch_threads_in_concurrent_group(
            kernel,
            (threads_x, pair.rows, 1),
            (threads_x, rows_per_tg, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&pair.weight), 0);
                encoder.set_buffer(1, Some(&pair.scale), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output), 0);
                set_u32(encoder, 5, &pair.rows);
                set_u32(encoder, 6, &pair.logical_k);
                set_u32(encoder, 7, &pair.scale_cols);
                set_u32(encoder, 8, &threads_x);
            },
        )
    } else {
        batch.dispatch_threads_in_concurrent_group(
            kernel,
            (pair.rows, 1, 1),
            (threads, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&pair.weight), 0);
                encoder.set_buffer(1, Some(&pair.scale), 0);
                encoder.set_buffer(2, Some(activation), 0);
                encoder.set_buffer(3, Some(activation_scales), 0);
                encoder.set_buffer(4, Some(output), 0);
                set_u32(encoder, 5, &pair.rows);
                set_u32(encoder, 6, &pair.logical_k);
                set_u32(encoder, 7, &pair.scale_cols);
            },
        )
    }
}

fn dispatch_bf16_cast(
    batch: &mut CommandBatch<'_>,
    input: &metal::Buffer,
    output: &metal::Buffer,
    count: u32,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads_in_concurrent_group(
        BF16_CAST_KERNEL,
        (count, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(input), 0);
            encoder.set_buffer(1, Some(output), 0);
            set_u32(encoder, 2, &count);
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch_routed_swiglu(
    batch: &mut CommandBatch<'_>,
    gate: &metal::Buffer,
    up: &metal::Buffer,
    output: &metal::Buffer,
    route_weights: &metal::Buffer,
    route_slot: u32,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads_in_concurrent_group(
        P6A_SWIGLU_KERNEL,
        (MOE_INTER_DIM as u32, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(gate), 0);
            encoder.set_buffer(1, Some(up), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_buffer(3, Some(route_weights), 0);
            set_u32(encoder, 4, &route_slot);
            set_u32(encoder, 5, &(MOE_INTER_DIM as u32));
        },
    )
}

fn dispatch_shared_fp8_gate_up_swiglu(
    batch: &mut CommandBatch<'_>,
    gate: &NativeFp8Gpu,
    up: &NativeFp8Gpu,
    quantized: &metal::Buffer,
    act_scales: &metal::Buffer,
    output_bf16: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    let rows = gate.rows;
    let cols = gate.logical_k;
    if rows == 0
        || cols == 0
        || cols % ACT_QUANT_BLOCK as u32 != 0
        || up.rows != rows
        || up.logical_k != cols
        || gate.scale_cols != cols / ACT_QUANT_BLOCK as u32
        || up.scale_cols != gate.scale_cols
        || threads == 0
    {
        return Err(p6_error(
            "P6 shared FP8 gate/up/SwiGLU geometry is not source-compatible",
        ));
    }
    let scale_cols = gate.scale_cols;
    let route_weight = 1.0_f32;
    batch.dispatch_threads_in_concurrent_group(
        P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_KERNEL,
        (rows, 1, 1),
        (threads.min(rows), 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(&gate.weight), 0);
            encoder.set_buffer(1, Some(&gate.scale), 0);
            encoder.set_buffer(2, Some(&up.weight), 0);
            encoder.set_buffer(3, Some(&up.scale), 0);
            encoder.set_buffer(4, Some(quantized), 0);
            encoder.set_buffer(5, Some(act_scales), 0);
            encoder.set_buffer(6, Some(output_bf16), 0);
            set_u32(encoder, 7, &rows);
            set_u32(encoder, 8, &cols);
            set_u32(encoder, 9, &scale_cols);
            encoder.set_bytes(
                10,
                size_of::<f32>() as u64,
                &route_weight as *const f32 as *const _,
            );
        },
    )
}

fn dispatch_fused_down_shared_combine(
    batch: &mut CommandBatch<'_>,
    routed_refs: &metal::Buffer,
    routed_resources: &[metal::Buffer],
    shared: &NativeFp8Gpu,
    shared_quantized: &metal::Buffer,
    shared_act_scales: &metal::Buffer,
    output_bf16: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    let rows = shared.rows;
    let shared_cols = shared.logical_k;
    let routed_packed_cols = MOE_INTER_DIM as u32 / 2;
    let routed_scale_cols = MOE_INTER_DIM as u32 / 32;
    let shared_scale_cols = shared.scale_cols;
    if rows == 0
        || rows != HIDDEN_SIZE as u32
        || shared_cols != MOE_INTER_DIM as u32
        || shared_cols % ACT_QUANT_BLOCK as u32 != 0
        || shared_scale_cols != shared_cols / ACT_QUANT_BLOCK as u32
        || routed_packed_cols == 0
        || routed_scale_cols == 0
        || threads == 0
        || routed_resources.len() != ACTIVATED_EXPERTS * 4
    {
        return Err(p6_error(
            "P6 full down FP4/FP8/combine geometry is not source-compatible",
        ));
    }
    batch.dispatch_threads_in_active_group(
        P6_FP4_DOWN_SHARED_COMBINE_FUSED_KERNEL,
        (rows, 1, 1),
        (threads.min(rows), 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(routed_refs), 0);
            encoder.set_buffer(1, Some(&shared.weight), 0);
            encoder.set_buffer(2, Some(&shared.scale), 0);
            encoder.set_buffer(3, Some(shared_quantized), 0);
            encoder.set_buffer(4, Some(shared_act_scales), 0);
            encoder.set_buffer(5, Some(output_bf16), 0);
            set_u32(encoder, 6, &rows);
            set_u32(encoder, 7, &routed_packed_cols);
            set_u32(encoder, 8, &routed_scale_cols);
            set_u32(encoder, 9, &shared_cols);
            set_u32(encoder, 10, &shared_scale_cols);
            let resource_refs: [&metal::ResourceRef; ACTIVATED_EXPERTS * 4] =
                std::array::from_fn(|index| &**routed_resources[index]);
            encoder.use_resources(&resource_refs, metal::MTLResourceUsage::Read);
        },
    )
}

fn dispatch_shared_fp8_down_bf16_combine(
    batch: &mut CommandBatch<'_>,
    shared: &NativeFp8Gpu,
    quantized: &metal::Buffer,
    act_scales: &metal::Buffer,
    routed_bf16: &[&metal::Buffer; ACTIVATED_EXPERTS],
    output_bf16: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    let rows = shared.rows;
    let cols = shared.logical_k;
    if rows == 0
        || cols == 0
        || cols % ACT_QUANT_BLOCK as u32 != 0
        || shared.scale_cols != cols / ACT_QUANT_BLOCK as u32
        || threads == 0
    {
        return Err(p6_error(
            "P6 shared FP8 down/combine geometry is not source-compatible",
        ));
    }
    batch.dispatch_threads_in_concurrent_group(
        P6_SHARED_FP8_DOWN_COMBINE_FUSED_KERNEL,
        (rows, 1, 1),
        (threads.min(rows), 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(&shared.weight), 0);
            encoder.set_buffer(1, Some(&shared.scale), 0);
            encoder.set_buffer(2, Some(quantized), 0);
            encoder.set_buffer(3, Some(act_scales), 0);
            for (slot, buffer) in routed_bf16.iter().enumerate() {
                encoder.set_buffer(4u64 + slot as u64, Some(buffer), 0);
            }
            encoder.set_buffer(10, Some(output_bf16), 0);
            let scale_cols = shared.scale_cols;
            set_u32(encoder, 11, &rows);
            set_u32(encoder, 12, &cols);
            set_u32(encoder, 13, &scale_cols);
        },
    )
}

fn dispatch_shared_swiglu(
    batch: &mut CommandBatch<'_>,
    gate: &metal::Buffer,
    up: &metal::Buffer,
    output: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    let route_weight = 1.0_f32;
    batch.dispatch_threads_in_concurrent_group(
        P5B_SWIGLU_KERNEL,
        (MOE_INTER_DIM as u32, 1, 1),
        (threads, 1, 1),
        |encoder| {
            encoder.set_buffer(0, Some(gate), 0);
            encoder.set_buffer(1, Some(up), 0);
            encoder.set_buffer(2, Some(output), 0);
            encoder.set_bytes(
                3,
                size_of::<f32>() as u64,
                &route_weight as *const f32 as *const _,
            );
            set_u32(encoder, 4, &(MOE_INTER_DIM as u32));
        },
    )
}

fn dispatch_combine(
    batch: &mut CommandBatch<'_>,
    routed: [&metal::Buffer; ACTIVATED_EXPERTS],
    shared: &metal::Buffer,
    output: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads_in_concurrent_group(
        P6A_COMBINE_KERNEL,
        (HIDDEN_SIZE as u32, 1, 1),
        (threads, 1, 1),
        |encoder| {
            for (index, buffer) in routed.iter().enumerate() {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            encoder.set_buffer(6, Some(shared), 0);
            encoder.set_buffer(7, Some(output), 0);
            set_u32(encoder, 8, &(HIDDEN_SIZE as u32));
        },
    )
}

fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
    encoder.set_bytes(
        index,
        size_of::<u32>() as u64,
        value as *const u32 as *const _,
    );
}

fn context_queue_identity(context: &MetalContext) -> usize {
    context.queue() as *const _ as usize
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn p6_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 reusable P6 device graph: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::{SHADER_MATMUL, SHADER_MOE};

    #[test]
    fn reusable_p6_graph_uses_admitted_c4_gate_and_authority_kernels() {
        for kernel in [
            P5B_FP4_KERNEL,
            P5B_SWIGLU_KERNEL,
            P6_C4_GATE_KERNEL,
            P6A_ROUTE_KERNEL,
            P6A_LEARNED_ROUTE_KERNEL,
            P6A_SWIGLU_KERNEL,
            P6A_COMBINE_KERNEL,
        ] {
            assert!(
                SHADER_MOE.contains(&format!("kernel void {kernel}(")),
                "P6 graph must use the admitted C4 Gate or established moe.metal authority kernel {kernel}"
            );
        }
        for kernel in [
            ACT_QUANT_KERNEL,
            FP8_KERNEL,
            BF16_CAST_KERNEL,
            P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_KERNEL,
            P6_SHARED_FP8_DOWN_COMBINE_FUSED_KERNEL,
        ] {
            assert!(
                SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                "P6 graph must reuse the established matmul authority kernel {kernel}"
            );
        }
        assert!(
            SHADER_MATMUL.contains(&format!("kernel void {FP4_SIMD_KERNEL}(")),
            "P6 FP4 SIMD candidate must remain in the runtime Metal library"
        );
        assert_eq!(P6_FP4_SIMD_ENV, "HAWKING_DSV4F_P6_FP4_SIMD");
        assert_eq!(FP4_SIMD_THREADS_X * FP4_SIMD_ROWS_PER_TG, 256);
        assert!(
            SHADER_MATMUL.contains(&format!("kernel void {FP8_SIMD_KERNEL}(")),
            "P6 FP8 SIMD candidate must remain in the runtime Metal library"
        );
        assert_eq!(P6_FP8_SIMD_ENV, "HAWKING_DSV4F_P6_FP8_SIMD");
        assert_eq!(FP8_SIMD_THREADS_X * FP8_SIMD_ROWS_PER_TG, 256);
        for kernel in [
            P6_FP4_GATE_UP_SWIGLU_KERNEL,
            P6_FP4_GATE_UP_SWIGLU_SIMD_KERNEL,
            P6_FP4_DOWN_BF16_KERNEL,
            P6_FP4_DOWN_BF16_SIMD_KERNEL,
            P6_FP4_DOWN_SHARED_COMBINE_FUSED_KERNEL,
        ] {
            assert!(
                SHADER_MOE.contains(&format!("kernel void {kernel}(")),
                "P6 fused routed gate/up/SwiGLU kernel must remain in the runtime Metal library"
            );
        }
        assert_eq!(
            P6_FP4_GATE_UP_SWIGLU_FUSED_ENV,
            "HAWKING_DSV4F_FP4_GATE_UP_SWIGLU_FUSED"
        );
        assert_eq!(
            P6_FP4_GATE_UP_SWIGLU_SIMD_ENV,
            "HAWKING_DSV4F_P6_FP4_GATE_UP_SWIGLU_SIMD"
        );
        assert_eq!(P6_FP4_GATE_UP_SWIGLU_SIMD_THREADS_X, 256);
        assert_eq!(P6_FP4_GATE_UP_SWIGLU_SIMD_ROWS_PER_TG, 8);
        assert_eq!(size_of::<P6Fp4GateUpRef>(), 48);
        assert_eq!(size_of::<P6Fp4DownRef>(), 48);
        assert_eq!(P6_PIPELINE_CACHE_ENV, "HAWKING_FLASH_PIPELINE_CACHE_REUSE");
        assert_eq!(
            P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_ENV,
            "HAWKING_DSV4F_FP8_SHARED_GATE_UP_SWIGLU_FUSED"
        );
        assert_eq!(
            P6_SHARED_FP8_DOWN_COMBINE_FUSED_ENV,
            "HAWKING_DSV4F_FP8_SHARED_DOWN_COMBINE_FUSED"
        );
        assert_eq!(
            P6_PREFIX_CONCURRENT_ENV,
            "HAWKING_DSV4F_P6_PREFIX_CONCURRENT"
        );
        assert_eq!(
            P6_LEARNED_READER_REUSE_ENV,
            "HAWKING_DSV4F_P6_LEARNED_READER_REUSE"
        );
        assert_eq!(
            P6_LEARNED_EXPERT_CACHE_REUSE_ENV,
            "HAWKING_DSV4F_P6_LEARNED_EXPERT_CACHE_REUSE"
        );
        assert_eq!(
            P6_FP4_DOWN_BF16_FUSED_ENV,
            "HAWKING_DSV4F_P6_FP4_DOWN_BF16_FUSED"
        );
        assert_eq!(
            P6_FP4_DOWN_BF16_SIMD_ENV,
            "HAWKING_DSV4F_P6_FP4_DOWN_BF16_SIMD"
        );
        assert_eq!(P6_FP4_DOWN_BF16_SIMD_THREADS_X, 256);
        assert_eq!(P6_FP4_DOWN_BF16_SIMD_ROWS_PER_TG, 8);
        assert_eq!(
            P6_FP4_DOWN_SHARED_COMBINE_FUSED_ENV,
            "HAWKING_DSV4F_P6_FP4_DOWN_SHARED_COMBINE_FUSED"
        );
    }

    #[test]
    fn reusable_p6_fused_gate_up_swiglu_dispatch_budget_is_closed() {
        // Authority batch 1 has 38 dispatches: gate/QAT/route, fourteen
        // routed/shared projections, fourteen casts, and seven SwiGLUs. The
        // candidate replaces only the routed 30-dispatch epilogue with one
        // fixed-six indirect launch; the shared path and down wave remain.
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_FUSED_GATE_UP_SWIGLU_DISPATCHES, 31);
        assert_eq!(
            DSV4F_P6_DEVICE_DISPATCHES - DSV4F_P6_FUSED_GATE_UP_SWIGLU_DISPATCHES,
            29
        );
    }

    #[test]
    fn reusable_p6_fused_down_bf16_dispatch_budget_is_closed() {
        // The six routed W2 launches plus six routed casts become one
        // fixed-six indirect launch. QAT, shared W2/cast, and final combine
        // remain separate and source-ordered.
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_FUSED_DOWN_BF16_DISPATCHES, 49);
        assert_eq!(DSV4F_P6_FUSED_GATE_UP_AND_DOWN_DISPATCHES, 20);
        assert_eq!(
            DSV4F_P6_DEVICE_DISPATCHES - DSV4F_P6_FUSED_DOWN_BF16_DISPATCHES,
            11
        );
        assert_eq!(
            DSV4F_P6_FUSED_GATE_UP_SWIGLU_DISPATCHES - DSV4F_P6_FUSED_GATE_UP_AND_DOWN_DISPATCHES,
            11
        );
    }

    #[test]
    fn reusable_p6_shared_fp8_fusion_budget_is_closed() {
        // Shared W1/W3, their two BF16 casts, and shared SwiGLU become one
        // source-order FP8 gate/up/SwiGLU launch. Routed and down authorities
        // remain unchanged in this isolated candidate.
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_DISPATCHES, 56);
        assert_eq!(
            DSV4F_P6_DEVICE_DISPATCHES - DSV4F_P6_SHARED_FP8_GATE_UP_SWIGLU_FUSED_DISPATCHES,
            4
        );
        assert_eq!(
            DSV4F_P6_FUSED_GATE_UP_AND_SHARED_DISPATCHES,
            DSV4F_P6_FUSED_GATE_UP_SWIGLU_DISPATCHES - 4
        );
    }

    #[test]
    fn reusable_p6_shared_down_combine_fusion_budget_is_closed() {
        // Shared FP8 W2, its BF16 cast, and the six-way BF16 combine become
        // one source-order launch. Routed W2/casts and shared QAT remain
        // unchanged in this isolated candidate.
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_SHARED_FP8_DOWN_COMBINE_FUSED_DISPATCHES, 58);
        assert_eq!(
            DSV4F_P6_DEVICE_DISPATCHES - DSV4F_P6_SHARED_FP8_DOWN_COMBINE_FUSED_DISPATCHES,
            2
        );
        assert_eq!(DSV4F_P6_FUSED_EPILOGUE_STACK_DISPATCHES, 8);
        assert_eq!(DSV4F_P6_FUSED_EPILOGUE_STACK_FULL_DOWN_COMPUTE_ENCODERS, 4);
    }

    #[test]
    fn reusable_p6_full_downstream_fusion_budget_is_closed() {
        // The six routed W2 launches, seven BF16 staging launches, the shared
        // W2 launch, and the final combine become one source-order launch.
        // The fixed-seven QAT wave remains its explicit predecessor.
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_FUSED_DOWN_SHARED_COMBINE_DISPATCHES, 46);
        assert_eq!(
            DSV4F_P6_DEVICE_DISPATCHES - DSV4F_P6_FUSED_DOWN_SHARED_COMBINE_DISPATCHES,
            14
        );
        assert_eq!(DSV4F_P6_FUSED_EPILOGUE_STACK_FULL_DOWN_DISPATCHES, 7);
    }

    #[test]
    fn reusable_p6_batched_down_qat_budget_is_closed() {
        // Six routed and one shared source-order QAT blocks become one
        // fixed-seven indirect launch; the enclosing concurrent wave stays
        // one compute encoder in either form.
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_BATCHED_DOWN_QAT_DISPATCHES, 54);
        assert_eq!(
            DSV4F_P6_DEVICE_DISPATCHES - DSV4F_P6_BATCHED_DOWN_QAT_DISPATCHES,
            6
        );
        assert_eq!(P6_BATCHED_DOWN_QAT_TENSORS, 7);
    }

    #[test]
    fn learned_two_phase_topology_is_explicit() {
        assert_eq!(DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS, 3);
        assert_eq!(DSV4F_P6_LEARNED_DEVICE_CPU_VISIBLE_WAITS, 3);
        assert_eq!(DSV4F_P6_LEARNED_DEVICE_DISPATCHES, 60);
        assert_eq!(DSV4F_P6_LEARNED_SINGLE_CB_COMMAND_BUFFERS, 2);
        assert_eq!(DSV4F_P6_LEARNED_SINGLE_CB_CPU_VISIBLE_WAITS, 2);
        assert!(DSV4F_P6_LEARNED_HOST_ROUTE_ID_READBACK);
        assert_eq!(
            P6A_LEARNED_ROUTE_KERNEL,
            "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority"
        );
    }

    #[test]
    fn learned_gpu_weight_cache_is_bounded_to_the_active_route() {
        let selected = [3, 9, 17, 41, 88, 255];
        let mut ids = BTreeMap::<u32, ()>::from([
            (3, ()),
            (9, ()),
            (17, ()),
            (41, ()),
            (88, ()),
            (255, ()),
            (127, ()),
        ]);
        retain_selected_expert_ids(&mut ids, &selected);
        assert_eq!(ids.len(), ACTIVATED_EXPERTS);
        assert!(ids.keys().all(|expert_id| selected.contains(expert_id)));
    }

    #[test]
    fn reusable_executor_implements_the_existing_p7_trait() {
        fn assert_p7_executor<T: DeepSeekV4P7P6DeviceExecutor>() {}
        assert_p7_executor::<DeepSeekV4Layer0P6MetalExecutor>();
    }

    #[test]
    fn source_controls_retain_the_p7_coordinate_triple() {
        let controls = DeepSeekV4P6SourceControls::new(0, 17, 3);
        assert_eq!(controls.layer, 0);
        assert_eq!(controls.token_id, 17);
        assert_eq!(controls.token_position, 3);
    }

    #[test]
    fn reusable_p6_topology_is_explicit_and_closed() {
        // Batch 1: gate + QAT + route + 14 W1/W3 + 14 casts + 7 SwiGLU.
        // Batch 2: 7 down-QAT + 7 W2 + 7 casts + one source-order combine.
        assert_eq!(DSV4F_P6_DEVICE_COMMAND_BUFFERS, 2);
        assert_eq!(DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS, 2);
        assert_eq!(DSV4F_P6_DEVICE_DISPATCHES, 38 + 22);
        assert_eq!(DSV4F_P6_DEVICE_COMPUTE_ENCODERS, 6 + 4);
        assert_eq!(DSV4F_P6_PREFIX_CONCURRENT_COMPUTE_ENCODERS, 9);
        assert_eq!(
            DSV4F_P6_DEVICE_COMPUTE_ENCODERS - DSV4F_P6_PREFIX_CONCURRENT_COMPUTE_ENCODERS,
            1
        );
        assert_eq!(DSV4F_P6_SINGLE_CB_COMMAND_BUFFERS, 1);
        assert_eq!(DSV4F_P6_SINGLE_CB_CPU_VISIBLE_WAITS, 1);
        assert_eq!(P6_SINGLE_CB_ENV, "HAWKING_DSV4F_P6_SINGLE_CB");
    }

    #[test]
    fn reusable_p6_gate_is_exactly_one_c4_simdgroup_per_row() {
        assert_eq!(
            P6_C4_GATE_KERNEL,
            "deepseek_v4_p0_gate_reduction_c4_simd32_fma_candidate"
        );
        assert_eq!(P6_C4_GATE_SIMDGROUP_THREADS, 32);
        assert_eq!(P6_C4_GATE_GRID_THREADS, 8_192);
        assert_eq!(
            P6_C4_GATE_GRID_THREADS,
            ROUTED_EXPERTS as u32 * P6_C4_GATE_SIMDGROUP_THREADS
        );
    }

    #[test]
    fn reusable_p6_output_contract_exposes_device_only_route_diagnostics() {
        // Field access is deliberately compile-time only: this verifies that
        // P6 returns device buffers for later completed-graph diagnostics
        // without mapping any buffer or introducing a host handoff.
        fn require_observability_fields(output: &DeepSeekV4P7P6DeviceOutput) {
            let _ = &output.route_valid_u32;
            let _ = &output.gate_logits_f32;
            let _ = &output.original_scores_f32;
        }
        let _ = require_observability_fields as fn(&DeepSeekV4P7P6DeviceOutput);
        let _ =
            DeepSeekV4P7P6DeviceOutput::validate as fn(&DeepSeekV4P7P6DeviceOutput) -> Result<()>;

        assert_eq!(ROUTE_VALID_BYTES, DSV4F_P7_ROUTE_VALID_U32_BYTES);
        assert_eq!(GATE_LOGITS_BYTES, DSV4F_P7_GATE_LOGITS_F32_BYTES);
        assert_eq!(ORIGINAL_SCORES_BYTES, DSV4F_P7_GATE_LOGITS_F32_BYTES);
    }
}
