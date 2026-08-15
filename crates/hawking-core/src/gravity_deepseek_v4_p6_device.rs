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
//!
//! This is not an Engine, causal loop, endpoint, parity receipt, or TPS path.

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
use crate::metal::{CommandBatch, MetalContext};
use crate::{Error, Result};

const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const BF16_CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const P5B_FP4_KERNEL: &str = "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority";
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

const HIDDEN_BF16_BYTES: usize = HIDDEN_SIZE * size_of::<u16>();
const ROUTE_IDS_BYTES: usize = ACTIVATED_EXPERTS * size_of::<u32>();
const ROUTE_WEIGHTS_BYTES: usize = ACTIVATED_EXPERTS * size_of::<f32>();
const GATE_LOGITS_BYTES: usize = ROUTED_EXPERTS * size_of::<f32>();
const ORIGINAL_SCORES_BYTES: usize = ROUTED_EXPERTS * size_of::<f32>();
const ROUTE_VALID_BYTES: usize = size_of::<u32>();
const GATE_BIAS_BYTES: usize = ROUTED_EXPERTS * size_of::<f32>();

/// Exact fixed topology of one hash-gate `DeepSeekV4Layer0P6MetalExecutor::execute`
/// call. These are structural counts for the bounded reusable P6 graph, not
/// a runtime or TPS measurement. Each `dispatch_batch` commits and waits once.
pub const DSV4F_P6_DEVICE_COMMAND_BUFFERS: usize = 2;
pub const DSV4F_P6_DEVICE_CPU_VISIBLE_WAITS: usize = 2;
pub const DSV4F_P6_DEVICE_DISPATCHES: usize = 60;
pub const DSV4F_P6_DEVICE_COMPUTE_ENCODERS: usize = 10;

/// Learned-bias two-phase topology: phase-1 (gate+QAT+route), host residency
/// load of the six selected experts, phase-2a (W1/W3/casts/SwiGLU), phase-2b
/// (down-QAT/W2/casts/combine). Same kernel count as hash (60); one extra
/// command buffer and CPU-visible wait for the dynamic expert load boundary.
pub const DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS: usize = 3;
pub const DSV4F_P6_LEARNED_DEVICE_CPU_VISIBLE_WAITS: usize = 3;
pub const DSV4F_P6_LEARNED_DEVICE_DISPATCHES: usize = 60;
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

struct NativeFp4Gpu {
    weight: metal::Buffer,
    scale: metal::Buffer,
    rows: u32,
    packed_k: u32,
    scale_cols: u32,
}

struct NativeFp8Gpu {
    weight: metal::Buffer,
    scale: metal::Buffer,
    rows: u32,
    logical_k: u32,
    scale_cols: u32,
}

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
    fp4: u32,
    fp8: u32,
    cast: u32,
    gate: u32,
    route: u32,
    routed_swiglu: u32,
    shared_swiglu: u32,
    combine: u32,
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
/// For learned-bias mode, `artifact_root` is retained so execute can re-admit
/// the sealed reader and load the six dynamically selected expert bundles.
pub struct DeepSeekV4Layer0P6MetalExecutor {
    controls: DeepSeekV4P6SourceControls,
    bindings: DeepSeekV4P6SourceBindings,
    route_mode: P6RouteMode,
    /// Present only for learned-bias two-phase residency loads.
    artifact_root: Option<PathBuf>,
    context_queue_identity: usize,
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
}

impl DeepSeekV4Layer0P6MetalExecutor {
    /// Prepare the source-bound graph from an admitted stream and expert cache.
    ///
    /// - **Hash gate**: acquires the six tid2eid-selected expert bundles into
    ///   the hot cache and uploads them before any dispatch.
    /// - **Learned-bias gate**: uploads Gate weight + bias + shared expert
    ///   only; the six routed expert weight slots are empty placeholders and
    ///   are filled during execute after the on-device route (two-phase).
    ///   `cache` may be empty at prepare; execute re-admits the reader and
    ///   fills a fresh hot cache for the selected IDs.
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
        let threads = precompile_threads(metal)?;
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
            context_queue_identity: context_queue_identity(metal),
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
        let threads = precompile_threads(metal)?;
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
        Ok(Self {
            controls,
            bindings,
            route_mode: P6RouteMode::Learned,
            artifact_root: Some(artifact_root),
            context_queue_identity: context_queue_identity(metal),
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

        // The first batch preserves the existing P6 operation order: admitted
        // C4 device Gate, predecessor QAT, device tid2eid/score normalization,
        // then concurrent W1/W3, casts, and routed/shared SwiGLU. There is no
        // CPU-visible activation or route-weight handoff between these encoders.
        input.metal.dispatch_batch(|batch| {
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
            )?;
            dispatch_hash_route(
                batch,
                &self.gate_logits,
                &self.route_table,
                &route_ids_u32,
                &route_weights_f32,
                &self.route_scores,
                &self.route_valid,
                self.controls.token_id,
                self.threads.route,
            )?;
            self.dispatch_up_projections_and_swiglu(batch, &route_weights_f32)?;
            Ok(())
        })?;

        input.metal.dispatch_batch(|batch| {
            self.dispatch_down_projections_and_combine(batch, &moe_output_bf16)
        })?;

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
        input.metal.dispatch_batch(|batch| {
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
            )?;
            dispatch_learned_route(
                batch,
                &self.gate_logits,
                &self.route_table,
                &route_ids_u32,
                &route_weights_f32,
                &self.route_scores,
                &self.route_valid,
                self.threads.route,
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

        self.load_learned_experts(input.metal, &selected)?;
        self.bindings.selected_expert_ids_top_slot_order = selected;

        // Phase 2a: W1/W3 + casts + SwiGLU (experts now resident).
        input.metal.dispatch_batch(|batch| {
            self.dispatch_up_projections_and_swiglu(batch, &route_weights_f32)
        })?;

        // Phase 2b: down projections + combine.
        input.metal.dispatch_batch(|batch| {
            self.dispatch_down_projections_and_combine(batch, &moe_output_bf16)
        })?;

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

    fn load_learned_experts(
        &mut self,
        metal: &MetalContext,
        selected_top_slot: &[u32; ACTIVATED_EXPERTS],
    ) -> Result<()> {
        let root = self.artifact_root.as_ref().ok_or_else(|| {
            p6_error("P6 learned execute missing artifact_root for expert residency")
        })?;
        let reader = DeepSeekV4FullStreamReader::admit(root)?;
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
        let mut cache = DeepSeekV4ExpertBundleCache::new(hot_bytes, 0)?;
        for &expert_id in selected_top_slot {
            let key = ExpertBundleKey::new(self.controls.layer as u16, expert_id as u16);
            cache.acquire(&reader, key)?;
        }

        for (slot, &expert_id) in selected_top_slot.iter().enumerate() {
            let key = ExpertBundleKey::new(self.controls.layer as u16, expert_id as u16);
            let bundle = cache.resident(key).ok_or_else(|| {
                p6_error(format!(
                    "P6 learned expert {expert_id} missing from hot cache after acquire"
                ))
            })?;
            let (w1, w1_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W1, FP4_W1_W3)?;
            let (w3, w3_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W3, FP4_W1_W3)?;
            let (w2, w2_name) = upload_cached_fp4(metal, bundle, ExpertOperator::W2, FP4_W2)?;
            // Replace placeholder weight buffers (scratch buffers retained).
            self.routed[slot].w1 = w1;
            self.routed[slot].w3 = w3;
            self.routed[slot].w2 = w2;
            self.routed[slot].source_top_slot = slot as u32;
            self.bindings.resident_experts_numeric_source_order[slot] =
                DeepSeekV4P6ResidentExpertBinding {
                    source_top_slot: slot as u32,
                    expert_id,
                    w1_weight_name: w1_name,
                    w3_weight_name: w3_name,
                    w2_weight_name: w2_name,
                };
        }
        self.experts_loaded = true;
        // Silence unused on hash-only builds / future diagnostics.
        let _ = metal;
        Ok(())
    }

    fn dispatch_up_projections_and_swiglu(
        &self,
        batch: &mut CommandBatch<'_>,
        route_weights_f32: &metal::Buffer,
    ) -> Result<()> {
        batch.begin_concurrent_group()?;
        for expert in &self.routed {
            dispatch_fp4(
                batch,
                &expert.w1,
                &self.input_quant,
                &self.input_scales,
                &expert.gate_f32,
                self.threads.fp4,
            )?;
            dispatch_fp4(
                batch,
                &expert.w3,
                &self.input_quant,
                &self.input_scales,
                &expert.up_f32,
                self.threads.fp4,
            )?;
        }
        dispatch_fp8(
            batch,
            &self.shared.w1,
            &self.input_quant,
            &self.input_scales,
            &self.shared.gate_f32,
            self.threads.fp8,
        )?;
        dispatch_fp8(
            batch,
            &self.shared.w3,
            &self.input_quant,
            &self.input_scales,
            &self.shared.up_f32,
            self.threads.fp8,
        )?;
        batch.end_concurrent_group()?;

        batch.begin_concurrent_group()?;
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
        batch.end_concurrent_group()?;

        batch.begin_concurrent_group()?;
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
        dispatch_shared_swiglu(
            batch,
            &self.shared.gate_bf16,
            &self.shared.up_bf16,
            &self.shared.swiglu_bf16,
            self.threads.shared_swiglu,
        )?;
        batch.end_concurrent_group()?;
        Ok(())
    }

    fn dispatch_down_projections_and_combine(
        &self,
        batch: &mut CommandBatch<'_>,
        moe_output_bf16: &metal::Buffer,
    ) -> Result<()> {
        batch.begin_concurrent_group()?;
        for expert in &self.routed {
            dispatch_act_quant_concurrent(
                batch,
                &expert.swiglu_bf16,
                &expert.down_quant,
                &expert.down_scales,
                MOE_INTER_DIM as u32,
                self.threads.qat,
            )?;
        }
        dispatch_act_quant_concurrent(
            batch,
            &self.shared.swiglu_bf16,
            &self.shared.down_quant,
            &self.shared.down_scales,
            MOE_INTER_DIM as u32,
            self.threads.qat,
        )?;
        batch.end_concurrent_group()?;

        batch.begin_concurrent_group()?;
        for expert in &self.routed {
            dispatch_fp4(
                batch,
                &expert.w2,
                &expert.down_quant,
                &expert.down_scales,
                &expert.down_f32,
                self.threads.fp4,
            )?;
        }
        dispatch_fp8(
            batch,
            &self.shared.w2,
            &self.shared.down_quant,
            &self.shared.down_scales,
            &self.shared.down_f32,
            self.threads.fp8,
        )?;
        batch.end_concurrent_group()?;

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
    Ok(RoutedExpertGpu {
        source_top_slot,
        w1,
        w3,
        w2,
        gate_f32: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<f32>())?,
        up_f32: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<f32>())?,
        gate_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        up_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        swiglu_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        down_quant: metal.new_buffer_checked(MOE_INTER_DIM)?,
        down_scales: metal.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
        down_f32: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<f32>())?,
        down_bf16: metal.new_buffer_checked(HIDDEN_BF16_BYTES)?,
    })
}

fn allocate_shared_expert(
    metal: &MetalContext,
    w1: NativeFp8Gpu,
    w3: NativeFp8Gpu,
    w2: NativeFp8Gpu,
) -> Result<SharedExpertGpu> {
    Ok(SharedExpertGpu {
        w1,
        w3,
        w2,
        gate_f32: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<f32>())?,
        up_f32: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<f32>())?,
        gate_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        up_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        swiglu_bf16: metal.new_buffer_checked(MOE_INTER_DIM * size_of::<u16>())?,
        down_quant: metal.new_buffer_checked(MOE_INTER_DIM)?,
        down_scales: metal.new_buffer_checked(MOE_INTER_DIM / ACT_QUANT_BLOCK)?,
        down_f32: metal.new_buffer_checked(HIDDEN_SIZE * size_of::<f32>())?,
        down_bf16: metal.new_buffer_checked(HIDDEN_BF16_BYTES)?,
    })
}

fn precompile_threads(metal: &MetalContext) -> Result<ThreadGeometry> {
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
    Ok(ThreadGeometry {
        qat: require_threads(pipeline_max(ACT_QUANT_KERNEL)?, 32, ACT_QUANT_KERNEL)?,
        fp4: require_threads(pipeline_max(P5B_FP4_KERNEL)?, 256, P5B_FP4_KERNEL)?,
        fp8: require_threads(pipeline_max(FP8_KERNEL)?, 256, FP8_KERNEL)?,
        cast: require_threads(pipeline_max(BF16_CAST_KERNEL)?, 256, BF16_CAST_KERNEL)?,
        gate: require_exact_simdgroup_threads(metal)?,
        route,
        routed_swiglu: require_threads(pipeline_max(P6A_SWIGLU_KERNEL)?, 256, P6A_SWIGLU_KERNEL)?,
        shared_swiglu: require_threads(pipeline_max(P5B_SWIGLU_KERNEL)?, 256, P5B_SWIGLU_KERNEL)?,
        combine: require_threads(pipeline_max(P6A_COMBINE_KERNEL)?, 256, P6A_COMBINE_KERNEL)?,
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
    let weight_bytes = geom
        .rows
        .checked_mul(geom.packed_k)
        .ok_or_else(|| p6_error("P6 empty FP4 weight size overflow"))?;
    let scale_bytes = geom
        .scale_rows
        .checked_mul(geom.scale_cols)
        .ok_or_else(|| p6_error("P6 empty FP4 scale size overflow"))?;
    Ok(NativeFp4Gpu {
        weight: metal.new_buffer_checked(weight_bytes)?,
        scale: metal.new_buffer_checked(scale_bytes)?,
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
) -> Result<()> {
    batch.dispatch_threads(
        ACT_QUANT_KERNEL,
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

fn dispatch_act_quant_concurrent(
    batch: &mut CommandBatch<'_>,
    input_bf16: &metal::Buffer,
    quantized: &metal::Buffer,
    scales: &metal::Buffer,
    cols: u32,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads_in_concurrent_group(
        ACT_QUANT_KERNEL,
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

fn dispatch_fp4(
    batch: &mut CommandBatch<'_>,
    pair: &NativeFp4Gpu,
    activation: &metal::Buffer,
    activation_scales: &metal::Buffer,
    output: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads_in_concurrent_group(
        P5B_FP4_KERNEL,
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

fn dispatch_fp8(
    batch: &mut CommandBatch<'_>,
    pair: &NativeFp8Gpu,
    activation: &metal::Buffer,
    activation_scales: &metal::Buffer,
    output: &metal::Buffer,
    threads: u32,
) -> Result<()> {
    batch.dispatch_threads_in_concurrent_group(
        FP8_KERNEL,
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
        for kernel in [ACT_QUANT_KERNEL, FP8_KERNEL, BF16_CAST_KERNEL] {
            assert!(
                SHADER_MATMUL.contains(&format!("kernel void {kernel}(")),
                "P6 graph must reuse the established matmul authority kernel {kernel}"
            );
        }
    }

    #[test]
    fn learned_two_phase_topology_is_explicit() {
        assert_eq!(DSV4F_P6_LEARNED_DEVICE_COMMAND_BUFFERS, 3);
        assert_eq!(DSV4F_P6_LEARNED_DEVICE_CPU_VISIBLE_WAITS, 3);
        assert_eq!(DSV4F_P6_LEARNED_DEVICE_DISPATCHES, 60);
        assert!(DSV4F_P6_LEARNED_HOST_ROUTE_ID_READBACK);
        assert_eq!(
            P6A_LEARNED_ROUTE_KERNEL,
            "deepseek_v4_p6a_learned_bias_route_sqrtsoftplus_authority"
        );
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
