//! CPU-only source-algorithm checkpoint for the DeepSeek-V4 layer-0 MoE
//! branch at one tokenizer-bound BOS / position-zero input.
//!
//! The bounded predecessor is the public layer-0 attention oracle.  This
//! module continues its result through exactly the source grammar below:
//!
//! ```text
//! attention HC state
//!   -> hc_ffn_pre / Sinkhorn / ffn RMSNorm
//!   -> Gate BF16 scores + layer-0 hash `tid2eid` selection
//!   -> six routed FP4 SwiGLU experts and one FP8 shared SwiGLU expert
//!   -> source-order route-weighted f32 combine
//!   -> hc_ffn_post
//! ```
//!
//! It intentionally remains a scalar, source-derived CPU *algorithm oracle*.
//! It does not execute the upstream Python/Torch/TileLang runtime, construct a
//! registered Hawking engine, allocate Metal resources, generate a token, or
//! measure an endpoint or TPS.  Its reduction order is documented and checked
//! as a useful parity-ladder precursor, never as independently-executed
//! upstream runtime parity.

use crate::gravity_deepseek_v4::{DeepSeekV4FullStreamReader, NativeScalePairKind};
use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, decode_e4m3fn, decode_e8m0fnu, fp8_e4m3fn_ue8m0_matvec,
    ActQuantizedBf16Row, Fp8MatvecCpuResult, ACT_QUANT_BLOCK,
};
use crate::gravity_deepseek_v4_layer0_attention::{
    hc_attn_post_source_algorithm, layer0_attention_cpu_oracle, rms_norm_bf16_source_algorithm,
    verify_layer0_attention_source_anchors, DeepSeekV4Layer0AttentionSourceAnchors,
    Layer0AttentionCpuOracleResult,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    hc_attn_pre_source_algorithm, HC_EPS, HC_FLAT_WIDTH, HC_MIX_WIDTH, HC_MULT, HC_SINKHORN_ITERS,
    HIDDEN_SIZE, PREFIX_TOKEN_ID, RMS_NORM_EPS,
};
use crate::{Error, Result};
use half::bf16;
use serde_json::Value;

/// The full source's layer-0 MoE uses hash routing because `0 <
/// n_hash_layers` in the pinned inference config.
pub const LAYER0_FFN_GATE_TID2EID: &str = "layers.0.ffn.gate.tid2eid";
pub const LAYER0_FFN_GATE_WEIGHT: &str = "layers.0.ffn.gate.weight";
pub const LAYER0_FFN_NORM_WEIGHT: &str = "layers.0.ffn_norm.weight";
pub const LAYER0_HC_FFN_FN: &str = "layers.0.hc_ffn_fn";
pub const LAYER0_HC_FFN_BASE: &str = "layers.0.hc_ffn_base";
pub const LAYER0_HC_FFN_SCALE: &str = "layers.0.hc_ffn_scale";

pub const ROUTED_EXPERTS: usize = 256;
pub const ACTIVATED_EXPERTS: usize = 6;
pub const SHARED_EXPERTS: usize = 1;
pub const MOE_INTER_DIM: usize = 2048;
pub const HASH_LAYERS: usize = 3;
pub const ROUTE_SCALE: f32 = 1.5;
pub const SWIGLU_LIMIT: f32 = 10.0;
pub const FP4_BLOCK: usize = 32;

/// Binds the MoE conversion to the already source-hash-bound attention
/// predecessor.  The attention binding itself includes the exact official
/// `model.py`, `kernel.py`, config, tokenizer, and conversion-script hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Layer0MoeSourceAnchors {
    pub attention: DeepSeekV4Layer0AttentionSourceAnchors,
    pub layer0_hash_routing: bool,
    pub source_tid2eid_storage_dtype: String,
}

/// One reusable source `linear` stage.  The output is BF16 because the
/// upstream kernel creates an output in the current default BF16 dtype.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedLinearCpuStage {
    pub quantized_input: ActQuantizedBf16Row,
    pub output: Fp8MatvecCpuResult,
}

/// Bounded checkpoint of the layer-0 `hc_ffn_pre` transition.  It is kept
/// separate from the attention result so a receipt can attest to the exact
/// source successor boundary without persisting raw activations.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0FfnHcPreCpuResult {
    pub flat_rsqrt: f32,
    pub mixes_f32: Vec<f32>,
    pub pre_f32: Vec<f32>,
    pub post_f32: Vec<f32>,
    pub comb_f32: Vec<f32>,
    pub reduced_bf16_bits: Vec<u16>,
}

/// The source Gate produces scores even for a hash-routed layer.  The exact
/// `tid2eid` row determines IDs, while the gathered *unbiased* scores remain
/// the weights used to scale routed expert activations.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0HashRouteCpuResult {
    pub token_id: u64,
    pub logits_f32: Vec<f32>,
    pub original_scores_f32: Vec<f32>,
    pub selected_expert_ids: Vec<u64>,
    pub selected_weights_f32: Vec<f32>,
}

/// Independent FP64 authority for the bounded layer-0 Gate/hash-route
/// control surface.  It consumes an already materialized BF16[4096] FFn-norm
/// row and rereads the raw BF16 Gate matrix plus the native I64 `tid2eid` row
/// through verified reader APIs.
///
/// This is deliberately a post-completion diagnostic helper.  It allocates no
/// Metal resources and has no path for its results to feed a device graph.  A
/// caller that captures its BF16 input from a device execution must do so only
/// after that execution has completed; this helper then supplies the FP64
/// reference vectors and exact hash-route IDs needed by Numeric Parity V2.1.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0HashRouteF64Authority {
    pub token_id: u64,
    pub logits_f64: Vec<f64>,
    pub original_scores_f64: Vec<f64>,
    pub selected_expert_ids: Vec<u64>,
    pub selected_weights_f64: Vec<f64>,
}

/// Independent FP64 authority for the source `hc_ffn_pre`/Sinkhorn control
/// surface.  It accepts an arbitrary completed BF16[4,4096] attention HC
/// state, rereads verified F32 `hc_ffn_*` controls, and projects the reduced
/// FP64 result through the declared BF16 source-store boundary.
///
/// This is CPU-only post-completion diagnostic code.  It does not create a
/// Metal buffer, submit work, or offer a route for its values to affect P7 or
/// any runtime graph.  It exists so a later observer can construct same-input
/// host/FP64 parity data after it has read a completed device state.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MhcFfnControlF64Authority {
    pub flat_rsqrt_f64: f64,
    pub mixes_f64: Vec<f64>,
    pub pre_f64: Vec<f64>,
    pub post_f64: Vec<f64>,
    pub comb_f64: Vec<f64>,
    pub reduced_bf16_bits: Vec<u16>,
}

/// Independent FP64 authority for source `Block.hc_post` after the layer-0
/// MoE branch.  `child_state_f64` retains the source operation order:
/// `post[output_lane] * moe[feature]`, then ascending residual source lanes
/// weighted by the corresponding *comb column*.  `child_state_bf16_bits`
/// models the declared source F32-to-BF16 output store.
///
/// This is a post-completion diagnostic result only.  It consumes captured
/// BF16 input rows and verified mHC control tensors, creates no Metal
/// resources, and does not feed a runtime graph.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MhcFfnPostF64Authority {
    pub controls: Layer0MhcFfnControlF64Authority,
    pub child_state_f64: Vec<f64>,
    pub child_state_bf16_bits: Vec<u16>,
}

/// One FP64 view of a source-native quantized linear stage.  `output_f64`
/// is accumulated from the exact E4M3/E2M1 and E8M0 storage bytes; the
/// accompanying BF16 vector is the mandatory source output-store boundary
/// used by the next stage.
///
/// This deliberately does not retain raw weight bytes.  The reader verifies
/// and streams those bytes for each stage, then they are dropped after the
/// bounded FP64 result has been produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeF64LinearStage {
    pub output_f64: Vec<f64>,
    pub output_bf16_bits: Vec<u16>,
}

/// One routed-expert body observed by the CPU-only FP64 authority.  The
/// `w1`/`w3` results are rounded to their declared BF16 stores before
/// `swiglu_f64` is evaluated; `w2_input_quantized` records the resulting
/// source-native activation bytes and scales for Numeric Parity V2.1.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeF64RoutedExpertAuthority {
    pub source_top_slot: usize,
    pub expert_id: u64,
    pub route_weight_f64: f64,
    pub w1: Layer0MoeF64LinearStage,
    pub w3: Layer0MoeF64LinearStage,
    pub swiglu_f64: Vec<f64>,
    pub swiglu_bf16_bits: Vec<u16>,
    pub w2_input_quantized: ActQuantizedBf16Row,
    pub w2: Layer0MoeF64LinearStage,
}

/// The always-on shared FP8 expert observed by the CPU-only FP64 body
/// authority.  It is intentionally separate from a routed expert because
/// source semantics apply no router weight to its SwiGLU result.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeF64SharedExpertAuthority {
    pub w1: Layer0MoeF64LinearStage,
    pub w3: Layer0MoeF64LinearStage,
    pub swiglu_f64: Vec<f64>,
    pub swiglu_bf16_bits: Vec<u16>,
    pub w2_input_quantized: ActQuantizedBf16Row,
    pub w2: Layer0MoeF64LinearStage,
}

/// One source-order routed contribution to the MoE sum.  The F32 and FP64
/// direct-row body authorities both follow exactly this order; keeping it as
/// data makes the otherwise easy-to-miss hash-route ordering available to a
/// later V2.1 checker without retaining model weights.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layer0MoeCombineOrder {
    pub source_top_slot: usize,
    pub expert_id: u64,
}

/// CPU-only FP64 authority for the layer-0 MoE body beginning at a completed
/// P7 FFn-norm BF16[4096] row and ending at the MoE BF16 store.  It includes
/// the gate/hash-route controls, all six selected native-FP4 experts, the
/// native-FP8 shared expert, and the source numeric-order combine, but
/// intentionally stops *before* `hc_ffn_post`.
///
/// The bounded result holds parity-relevant vectors and discrete activation
/// storage only.  Source weight windows are verified, consumed, and dropped
/// stage by stage.  This helper is post-completion CPU diagnostic code: it
/// allocates no Metal resources and has no route into P7 or the runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeBodyF64Authority {
    pub token_id: u64,
    /// Native E4M3/E8M0 input storage shared by every W1/W3 projection.
    pub ffn_norm_quantized: ActQuantizedBf16Row,
    pub route: Layer0HashRouteF64Authority,
    /// Numeric expert-ID order used for the serial source combine.
    pub routed_combine_order: Vec<Layer0MoeCombineOrder>,
    pub routed_experts: Vec<Layer0MoeF64RoutedExpertAuthority>,
    pub shared_expert: Layer0MoeF64SharedExpertAuthority,
    /// FP64 serial sum of BF16-rounded routed W2 results, then shared W2.
    pub combined_f64: Vec<f64>,
    /// Source F32-to-BF16 rounded MoE output, ready for a separate mHC-post
    /// diagnostic if one is later required.
    pub moe_output_bf16_bits: Vec<u16>,
}

/// Direct source-F32 layer-0 MoE body from an externally completed FFn-norm
/// BF16[4096] row.  This is the host candidate for a same-input Numeric
/// Parity V2.1 score against a P7 capture and [`Layer0MoeBodyF64Authority`].
/// It intentionally ends at the MoE BF16 store rather than recomputing the
/// attention predecessor or entering `hc_ffn_post`.
///
/// Like the other CPU oracles, it is a verified-reader diagnostic helper: no
/// Metal allocations, command submission, runtime registration, or endpoint
/// behavior is reachable through this type.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeBodyF32OracleResult {
    pub token_id: u64,
    /// The exact source native input bytes/scales observed for W1/W3.
    pub ffn_norm_quantized: ActQuantizedBf16Row,
    pub route: Layer0HashRouteCpuResult,
    /// Numeric expert-ID order used for the serial source F32 combine.
    pub routed_combine_order: Vec<Layer0MoeCombineOrder>,
    pub routed_experts: Vec<RoutedExpertCpuResult>,
    pub shared_expert: SharedExpertCpuResult,
    /// Source F32 sum of BF16-rounded W2 contributions, before `y.type_as`.
    pub moe_output_f32: Vec<f32>,
    pub moe_output_bf16_bits: Vec<u16>,
}

/// A source-order routed expert invocation.  Entries are ordered by
/// `(expert_id, source top-slot)` to mirror the upstream loop over expert IDs
/// followed by the `torch.where(indices == i)` result order.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedExpertCpuResult {
    pub source_top_slot: usize,
    pub expert_id: u64,
    pub route_weight: f32,
    pub gate: QuantizedLinearCpuStage,
    pub up: QuantizedLinearCpuStage,
    pub weighted_swiglu_bf16_bits: Vec<u16>,
    pub down: QuantizedLinearCpuStage,
}

/// The always-on shared FP8 expert execution.  It is intentionally modeled
/// separately from routed experts: it receives no router weight.
#[derive(Debug, Clone, PartialEq)]
pub struct SharedExpertCpuResult {
    pub gate: QuantizedLinearCpuStage,
    pub up: QuantizedLinearCpuStage,
    pub swiglu_bf16_bits: Vec<u16>,
    pub down: QuantizedLinearCpuStage,
}

/// Complete source-derived MoE successor after an already-proven layer-0
/// attention mHC state.  The predecessor is passed in rather than recreated
/// so a bounded causal continuation can carry its exact four BF16 lanes into
/// the FFN without substituting a position-zero state.
///
/// This remains a CPU source-algorithm result.  In particular, accepting an
/// HC state here does not make the caller a decoder runtime or establish any
/// upstream-runtime parity claim.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeSuccessorCpuOracleResult {
    pub ffn_hc_pre: Layer0FfnHcPreCpuResult,
    pub ffn_norm_bf16_bits: Vec<u16>,
    pub route: Layer0HashRouteCpuResult,
    pub routed_experts: Vec<RoutedExpertCpuResult>,
    pub shared_expert: SharedExpertCpuResult,
    pub moe_output_bf16_bits: Vec<u16>,
    pub hc_ffn_post_bf16_bits: Vec<u16>,
}

/// Complete bounded layer-0 MoE result.  All vectors are caller-memory only;
/// the companion receipt records hashes and finite sufficient statistics, not
/// the raw hidden states or complete tensor payloads.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0MoeCpuOracleResult {
    pub attention: Layer0AttentionCpuOracleResult,
    pub ffn_hc_pre: Layer0FfnHcPreCpuResult,
    pub ffn_norm_bf16_bits: Vec<u16>,
    pub route: Layer0HashRouteCpuResult,
    pub routed_experts: Vec<RoutedExpertCpuResult>,
    pub shared_expert: SharedExpertCpuResult,
    pub moe_output_bf16_bits: Vec<u16>,
    pub hc_ffn_post_bf16_bits: Vec<u16>,
}

/// Verify the exact static source/config/tensor geometry used by the bounded
/// layer-0 MoE branch.  The source code grammar anchors make an accidental
/// extension of this interpreter to a non-hash or non-FP4 layer fail closed.
pub fn verify_layer0_moe_source_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4Layer0MoeSourceAnchors> {
    let attention = verify_layer0_attention_source_anchors(reader)?;
    let inference_config = parse_json(
        &reader.read_verified_metadata_asset("inference/config.json", 64 * 1024)?,
        "inference/config.json",
    )?;
    let model_config = parse_json(
        &reader.read_verified_metadata_asset("config.json", 64 * 1024)?,
        "config.json",
    )?;
    let model_py = reader.read_verified_metadata_asset("inference/model.py", 128 * 1024)?;
    let kernel_py = reader.read_verified_metadata_asset("inference/kernel.py", 128 * 1024)?;

    if json_u64(&inference_config, &["dim"], "inference dim")? != HIDDEN_SIZE as u64
        || json_u64(
            &inference_config,
            &["moe_inter_dim"],
            "inference moe_inter_dim",
        )? != MOE_INTER_DIM as u64
        || json_u64(
            &inference_config,
            &["n_routed_experts"],
            "inference n_routed_experts",
        )? != ROUTED_EXPERTS as u64
        || json_u64(
            &inference_config,
            &["n_shared_experts"],
            "inference n_shared_experts",
        )? != SHARED_EXPERTS as u64
        || json_u64(
            &inference_config,
            &["n_activated_experts"],
            "inference n_activated_experts",
        )? != ACTIVATED_EXPERTS as u64
        || json_u64(
            &inference_config,
            &["n_hash_layers"],
            "inference n_hash_layers",
        )? != HASH_LAYERS as u64
        || json_string(&inference_config, &["score_func"], "inference score_func")?
            != "sqrtsoftplus"
        || !json_f64_eq(&inference_config, &["route_scale"], ROUTE_SCALE)
        || !json_f64_eq(&inference_config, &["swiglu_limit"], SWIGLU_LIMIT)
        || json_string(
            &inference_config,
            &["expert_dtype"],
            "inference expert_dtype",
        )? != "fp4"
        || json_u64(&model_config, &["hidden_size"], "model hidden size")? != HIDDEN_SIZE as u64
        || json_u64(
            &model_config,
            &["moe_intermediate_size"],
            "model moe intermediate size",
        )? != MOE_INTER_DIM as u64
        || json_u64(&model_config, &["n_routed_experts"], "model routed experts")?
            != ROUTED_EXPERTS as u64
        || json_u64(&model_config, &["n_shared_experts"], "model shared experts")?
            != SHARED_EXPERTS as u64
        || json_u64(
            &model_config,
            &["num_experts_per_tok"],
            "model activated experts",
        )? != ACTIVATED_EXPERTS as u64
        || json_string(&model_config, &["scoring_func"], "model scoring func")? != "sqrtsoftplus"
        || !json_f64_eq(&model_config, &["routed_scaling_factor"], ROUTE_SCALE)
        || !json_f64_eq(&model_config, &["swiglu_limit"], SWIGLU_LIMIT)
        || !json_f64_eq(&model_config, &["rms_norm_eps"], RMS_NORM_EPS)
        || json_u64(&model_config, &["hc_mult"], "model hc_mult")? != HC_MULT as u64
        || json_u64(
            &model_config,
            &["hc_sinkhorn_iters"],
            "model hc sinkhorn iters",
        )? != HC_SINKHORN_ITERS as u64
        || !json_f64_eq(&model_config, &["hc_eps"], 1.0e-6)
    {
        return Err(gravity(
            "pinned source configs differ from the layer-0 MoE source contract",
        ));
    }

    // Source-level grammar pins the otherwise easy-to-miss property that the
    // hash path still computes scores/weights, but chooses IDs from tid2eid.
    for (asset, needle) in [
        (
            &model_py,
            b"self.hash = layer_id < args.n_hash_layers".as_slice(),
        ),
        (&model_py, b"indices = self.tid2eid[input_ids]".as_slice()),
        (
            &model_py,
            b"weights = original_scores.gather(1, indices)".as_slice(),
        ),
        (&model_py, b"weights *= self.route_scale".as_slice()),
        (&model_py, b"x = F.silu(gate) * up".as_slice()),
        (&model_py, b"return self.w2(x.to(dtype))".as_slice()),
        (&model_py, b"y += self.shared_experts(x)".as_slice()),
        (
            &model_py,
            b"x, post, comb = self.hc_pre(x, self.hc_ffn_fn".as_slice(),
        ),
        (&kernel_py, b"def fp4_gemm_kernel(N, K".as_slice()),
        (
            &kernel_py,
            b"Weight: 1x32 quant on K (reduce dim), FP4 with E8M0 scale".as_slice(),
        ),
    ] {
        if !asset.windows(needle.len()).any(|window| window == needle) {
            return Err(gravity("pinned MoE source grammar anchor is absent"));
        }
    }

    expect_tensor(
        reader,
        LAYER0_HC_FFN_FN,
        "F32",
        &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
    )?;
    expect_tensor(reader, LAYER0_HC_FFN_BASE, "F32", &[HC_MIX_WIDTH as u64])?;
    expect_tensor(reader, LAYER0_HC_FFN_SCALE, "F32", &[3])?;
    expect_tensor(
        reader,
        LAYER0_FFN_NORM_WEIGHT,
        "BF16",
        &[HIDDEN_SIZE as u64],
    )?;
    expect_tensor(
        reader,
        LAYER0_FFN_GATE_WEIGHT,
        "BF16",
        &[ROUTED_EXPERTS as u64, HIDDEN_SIZE as u64],
    )?;
    let tid2eid = expect_tensor(
        reader,
        LAYER0_FFN_GATE_TID2EID,
        "I64",
        &[129_280, ACTIVATED_EXPERTS as u64],
    )?;
    if reader.tensor_metadata("layers.0.ffn.gate.bias").is_ok() {
        return Err(gravity(
            "layer-0 hash Gate unexpectedly contains a bias tensor",
        ));
    }

    Ok(DeepSeekV4Layer0MoeSourceAnchors {
        attention,
        layer0_hash_routing: true,
        source_tid2eid_storage_dtype: tid2eid.dtype.clone(),
    })
}

/// Execute the bounded complete layer-0 MoE successor path for the exact BOS
/// / position-zero attention predecessor.  The result must be sealed by a
/// caller that explicitly preserves its CPU-oracle boundary.
pub fn layer0_moe_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<Layer0MoeCpuOracleResult> {
    let attention = layer0_attention_cpu_oracle(reader)?;
    let successor = layer0_moe_successor_cpu_oracle(
        reader,
        PREFIX_TOKEN_ID,
        &attention.hc_attn_post_bf16_bits,
    )?;

    Ok(Layer0MoeCpuOracleResult {
        attention,
        ffn_hc_pre: successor.ffn_hc_pre,
        ffn_norm_bf16_bits: successor.ffn_norm_bf16_bits,
        route: successor.route,
        routed_experts: successor.routed_experts,
        shared_expert: successor.shared_expert,
        moe_output_bf16_bits: successor.moe_output_bf16_bits,
        hc_ffn_post_bf16_bits: successor.hc_ffn_post_bf16_bits,
    })
}

/// Execute the exact layer-0 MoE successor from an explicit source-derived
/// attention HC residual state and the source token ID that indexes the hash
/// routing table.  This is the reusable parity target for position-one: it
/// keeps the position-one four-lane attention state and uses
/// `tid2eid[token_id]`, instead of silently reusing the BOS routing row.
pub fn layer0_moe_successor_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    attention_hc_post_bf16_bits: &[u16],
) -> Result<Layer0MoeSuccessorCpuOracleResult> {
    verify_layer0_moe_source_anchors(reader)?;
    if attention_hc_post_bf16_bits.len() != HC_FLAT_WIDTH {
        return Err(gravity(
            "layer-0 attention predecessor does not expose the expected HC state",
        ));
    }

    let hc_fn = read_f32_tensor(reader, LAYER0_HC_FFN_FN, HC_MIX_WIDTH * HC_FLAT_WIDTH)?;
    let hc_base = read_f32_tensor(reader, LAYER0_HC_FFN_BASE, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tensor(reader, LAYER0_HC_FFN_SCALE, 3)?;
    let (flat_rsqrt, mixes_f32, pre_f32, post_f32, comb_f32, reduced_bf16_bits) =
        hc_attn_pre_source_algorithm(
            attention_hc_post_bf16_bits,
            &hc_fn,
            &hc_scale,
            &hc_base,
            RMS_NORM_EPS,
            1.0e-6,
            HC_SINKHORN_ITERS,
        )?;
    let ffn_hc_pre = Layer0FfnHcPreCpuResult {
        flat_rsqrt,
        mixes_f32,
        pre_f32,
        post_f32,
        comb_f32,
        reduced_bf16_bits,
    };

    let ffn_norm_weight = read_bf16_tensor(reader, LAYER0_FFN_NORM_WEIGHT, HIDDEN_SIZE)?;
    let ffn_norm_bf16_bits = rms_norm_bf16_source_algorithm(
        &ffn_hc_pre.reduced_bf16_bits,
        &ffn_norm_weight,
        HIDDEN_SIZE,
        RMS_NORM_EPS,
    )?;
    let route = layer0_hash_route_cpu_oracle_for_token(reader, token_id, &ffn_norm_bf16_bits)?;

    let mut execution_slots: Vec<usize> = (0..ACTIVATED_EXPERTS).collect();
    execution_slots.sort_unstable_by_key(|&slot| (route.selected_expert_ids[slot], slot));
    let mut routed_experts = Vec::with_capacity(ACTIVATED_EXPERTS);
    let mut routed_sum_f32 = vec![0.0_f32; HIDDEN_SIZE];
    for source_top_slot in execution_slots {
        let expert_id = route.selected_expert_ids[source_top_slot];
        let route_weight = route.selected_weights_f32[source_top_slot];
        let expert = routed_expert_cpu_oracle(
            reader,
            source_top_slot,
            expert_id,
            route_weight,
            &ffn_norm_bf16_bits,
        )?;
        // `y` is source float32.  The upstream loops in numeric expert ID
        // order, which is precisely the ordering used above.
        for (accumulator, &bits) in routed_sum_f32.iter_mut().zip(&expert.down.output.bf16_bits) {
            *accumulator += bf16::from_bits(bits).to_f32();
        }
        routed_experts.push(expert);
    }

    let shared_expert = shared_expert_cpu_oracle(reader, &ffn_norm_bf16_bits)?;
    for (accumulator, &bits) in routed_sum_f32
        .iter_mut()
        .zip(&shared_expert.down.output.bf16_bits)
    {
        *accumulator += bf16::from_bits(bits).to_f32();
    }
    if routed_sum_f32.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "layer-0 MoE float32 combine produced a non-finite value",
        ));
    }
    // Source: `y.type_as(x)` before Block.hc_post receives the FNN result.
    let moe_output_bf16_bits: Vec<u16> = routed_sum_f32
        .iter()
        .copied()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    let hc_ffn_post_bf16_bits = hc_attn_post_source_algorithm(
        &moe_output_bf16_bits,
        attention_hc_post_bf16_bits,
        &ffn_hc_pre.post_f32,
        &ffn_hc_pre.comb_f32,
    )?;

    Ok(Layer0MoeSuccessorCpuOracleResult {
        ffn_hc_pre,
        ffn_norm_bf16_bits,
        route,
        routed_experts,
        shared_expert,
        moe_output_bf16_bits,
        hc_ffn_post_bf16_bits,
    })
}

/// Independently evaluate layer-0 `hc_ffn_pre` and its Sinkhorn controls in
/// FP64 from an arbitrary BF16[4,4096] attention-HC state.
///
/// The F32 source controls are reread through verified artifact APIs and
/// promoted directly to FP64.  The returned reduced row is rounded through
/// the declared source F32-to-BF16 store boundary, making it suitable as the
/// exact BF16 input to a subsequent same-input Gate authority.  This helper
/// is diagnostic-only: it has no Metal dependency and must be called only
/// after the caller's device execution has completed.
pub fn layer0_mhc_ffn_control_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    attention_hc_post_bf16_bits: &[u16],
) -> Result<Layer0MhcFfnControlF64Authority> {
    if attention_hc_post_bf16_bits.len() != HC_FLAT_WIDTH {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN authority requires BF16[4,4096] attention state",
        ));
    }
    let hc_fn = read_f32_tensor(reader, LAYER0_HC_FFN_FN, HC_MIX_WIDTH * HC_FLAT_WIDTH)?;
    let hc_base = read_f32_tensor(reader, LAYER0_HC_FFN_BASE, HC_MIX_WIDTH)?;
    let hc_scale = read_f32_tensor(reader, LAYER0_HC_FFN_SCALE, 3)?;
    layer0_mhc_ffn_control_f64_from_verified_controls(
        attention_hc_post_bf16_bits,
        &hc_fn,
        &hc_scale,
        &hc_base,
    )
}

/// Independently execute the layer-0 mHC-FFN post transition in FP64 from a
/// completed attention HC BF16[4,4096] state and a completed MoE BF16[4096]
/// row.
///
/// The function rereads the admitted `hc_ffn_*` controls through
/// [`layer0_mhc_ffn_control_f64_authority`], preserving its source-order
/// Sinkhorn calculation.  It then applies source `Block.hc_post` ordering in
/// FP64 and explicitly rounds the resulting child state through the source
/// F32-to-BF16 boundary.  It is CPU-only, post-completion diagnostic code and
/// intentionally has no P7/Metal/runtime surface.
pub fn layer0_mhc_ffn_post_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    attention_hc_post_bf16_bits: &[u16],
    moe_output_bf16_bits: &[u16],
) -> Result<Layer0MhcFfnPostF64Authority> {
    if attention_hc_post_bf16_bits.len() != HC_FLAT_WIDTH
        || moe_output_bf16_bits.len() != HIDDEN_SIZE
    {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN-post authority requires BF16[4,4096] attention state and BF16[4096] MoE row",
        ));
    }
    // Keep the post authority anchored to the same source/config grammar as
    // the control and expert-side authorities before processing a P7 capture.
    verify_layer0_moe_source_anchors(reader)?;
    let controls = layer0_mhc_ffn_control_f64_authority(reader, attention_hc_post_bf16_bits)?;
    layer0_mhc_ffn_post_f64_from_control_authority(
        attention_hc_post_bf16_bits,
        moe_output_bf16_bits,
        controls,
    )
}

fn layer0_mhc_ffn_post_f64_from_control_authority(
    attention_hc_post_bf16_bits: &[u16],
    moe_output_bf16_bits: &[u16],
    controls: Layer0MhcFfnControlF64Authority,
) -> Result<Layer0MhcFfnPostF64Authority> {
    if attention_hc_post_bf16_bits.len() != HC_FLAT_WIDTH
        || moe_output_bf16_bits.len() != HIDDEN_SIZE
        || controls.mixes_f64.len() != HC_MIX_WIDTH
        || controls.pre_f64.len() != HC_MULT
        || controls.post_f64.len() != HC_MULT
        || controls.comb_f64.len() != HC_MULT * HC_MULT
        || controls.reduced_bf16_bits.len() != HIDDEN_SIZE
        || !controls.flat_rsqrt_f64.is_finite()
        || controls
            .mixes_f64
            .iter()
            .chain(&controls.pre_f64)
            .chain(&controls.post_f64)
            .chain(&controls.comb_f64)
            .any(|value| !value.is_finite())
        || controls
            .reduced_bf16_bits
            .iter()
            .any(|&bits| !bf16::from_bits(bits).to_f32().is_finite())
    {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN-post authority controls differ from the pinned source geometry",
        ));
    }

    let residual_hc_f64 = attention_hc_post_bf16_bits
        .iter()
        .map(|&bits| f64::from(bf16::from_bits(bits).to_f32()))
        .collect::<Vec<_>>();
    let moe_f64 = moe_output_bf16_bits
        .iter()
        .map(|&bits| f64::from(bf16::from_bits(bits).to_f32()))
        .collect::<Vec<_>>();
    if residual_hc_f64
        .iter()
        .chain(&moe_f64)
        .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN-post authority BF16 input contains a non-finite value",
        ));
    }

    let mut child_state_f64 = Vec::with_capacity(HC_FLAT_WIDTH);
    for output_lane in 0..HC_MULT {
        for feature in 0..HIDDEN_SIZE {
            // Source `Block.hc_post`: post scales the branch result first,
            // then source lanes 0..3 are added in ascending order.  Source
            // comb columns select output lanes, hence `[source, output]`.
            let mut value = controls.post_f64[output_lane] * moe_f64[feature];
            for source_lane in 0..HC_MULT {
                value += controls.comb_f64[source_lane * HC_MULT + output_lane]
                    * residual_hc_f64[source_lane * HIDDEN_SIZE + feature];
            }
            if !value.is_finite() {
                return Err(gravity(
                    "layer-0 FP64 mHC-FFN-post authority produced a non-finite child-state value",
                ));
            }
            child_state_f64.push(value);
        }
    }
    let child_state_bf16_bits = f64_values_to_source_bf16(
        &child_state_f64,
        "layer-0 FP64 mHC-FFN-post authority child-state store",
    )?;
    Ok(Layer0MhcFfnPostF64Authority {
        controls,
        child_state_f64,
        child_state_bf16_bits,
    })
}

fn layer0_mhc_ffn_control_f64_from_verified_controls(
    attention_hc_post_bf16_bits: &[u16],
    hc_fn_f32: &[f32],
    hc_scale_f32: &[f32],
    hc_base_f32: &[f32],
) -> Result<Layer0MhcFfnControlF64Authority> {
    if attention_hc_post_bf16_bits.len() != HC_FLAT_WIDTH
        || hc_fn_f32.len() != HC_MIX_WIDTH * HC_FLAT_WIDTH
        || hc_scale_f32.len() != 3
        || hc_base_f32.len() != HC_MIX_WIDTH
        || hc_fn_f32
            .iter()
            .chain(hc_scale_f32)
            .chain(hc_base_f32)
            .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN authority controls differ from the pinned source geometry",
        ));
    }

    let flat: Vec<f64> = attention_hc_post_bf16_bits
        .iter()
        .map(|&bits| f64::from(bf16::from_bits(bits).to_f32()))
        .collect();
    if flat.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN authority attention state contains a non-finite BF16 value",
        ));
    }
    let hc_scale: Vec<f64> = hc_scale_f32.iter().copied().map(f64::from).collect();
    let hc_base: Vec<f64> = hc_base_f32.iter().copied().map(f64::from).collect();
    let norm_eps = f64::from(RMS_NORM_EPS);
    let hc_eps = f64::from(HC_EPS);

    // Source: `torch.rsqrt(x.square().mean(-1) + norm_eps)`, retained as a
    // serial FP64 sum for the independent authority.
    let mut mean_square_sum = 0.0_f64;
    for &value in &flat {
        mean_square_sum += value * value;
    }
    let flat_rsqrt_f64 = 1.0_f64 / (mean_square_sum / HC_FLAT_WIDTH as f64 + norm_eps).sqrt();
    if !flat_rsqrt_f64.is_finite() {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN authority rsqrt is non-finite",
        ));
    }

    // Source `F.linear(x, hc_fn) * rsqrt`, with the raw source F32 weights
    // decoded directly to FP64 rather than transcribing the CPU F32 result.
    let mut mixes_f64 = vec![0.0_f64; HC_MIX_WIDTH];
    for row in 0..HC_MIX_WIDTH {
        let mut accumulator = 0.0_f64;
        let weights = &hc_fn_f32[row * HC_FLAT_WIDTH..(row + 1) * HC_FLAT_WIDTH];
        for (&weight, &value) in weights.iter().zip(&flat) {
            accumulator += f64::from(weight) * value;
        }
        let mix = accumulator * flat_rsqrt_f64;
        if !mix.is_finite() {
            return Err(gravity(
                "layer-0 FP64 mHC-FFN authority linear mix is non-finite",
            ));
        }
        mixes_f64[row] = mix;
    }

    let (pre_f64, post_f64, comb_f64) =
        hc_split_sinkhorn_f64_authority(&mixes_f64, &hc_scale, &hc_base, hc_eps)?;
    let mut reduced_bf16_bits = Vec::with_capacity(HIDDEN_SIZE);
    for feature in 0..HIDDEN_SIZE {
        // Source reduction retains ascending lane order before its BF16
        // `type_as` boundary.  The authority computes the arithmetic in FP64
        // then explicitly emulates the declared source F32->BF16 store.
        let mut reduced = 0.0_f64;
        for lane in 0..HC_MULT {
            reduced += pre_f64[lane] * flat[lane * HIDDEN_SIZE + feature];
        }
        if !reduced.is_finite() {
            return Err(gravity(
                "layer-0 FP64 mHC-FFN authority reduction is non-finite",
            ));
        }
        let source_store_f32 = reduced as f32;
        if !source_store_f32.is_finite() {
            return Err(gravity(
                "layer-0 FP64 mHC-FFN authority reduction overflows source F32 storage",
            ));
        }
        reduced_bf16_bits.push(bf16::from_f32(source_store_f32).to_bits());
    }

    Ok(Layer0MhcFfnControlF64Authority {
        flat_rsqrt_f64,
        mixes_f64,
        pre_f64,
        post_f64,
        comb_f64,
        reduced_bf16_bits,
    })
}

fn hc_split_sinkhorn_f64_authority(
    mixes: &[f64],
    hc_scale: &[f64],
    hc_base: &[f64],
    eps: f64,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    if mixes.len() != HC_MIX_WIDTH
        || hc_scale.len() != 3
        || hc_base.len() != HC_MIX_WIDTH
        || !(eps.is_finite() && eps > 0.0)
        || mixes
            .iter()
            .chain(hc_scale)
            .chain(hc_base)
            .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN authority Sinkhorn inputs differ from the pinned contract",
        ));
    }

    let mut pre_f64 = Vec::with_capacity(HC_MULT);
    let mut post_f64 = Vec::with_capacity(HC_MULT);
    for lane in 0..HC_MULT {
        pre_f64.push(sigmoid_f64_authority(mixes[lane] * hc_scale[0] + hc_base[lane]) + eps);
        post_f64.push(
            2.0_f64
                * sigmoid_f64_authority(
                    mixes[lane + HC_MULT] * hc_scale[1] + hc_base[lane + HC_MULT],
                ),
        );
    }

    let mut comb_f64 = vec![0.0_f64; HC_MULT * HC_MULT];
    for row in 0..HC_MULT {
        for column in 0..HC_MULT {
            let index = row * HC_MULT + column;
            let source_index = index + HC_MULT * 2;
            comb_f64[index] = mixes[source_index] * hc_scale[2] + hc_base[source_index];
        }
    }

    // First source pass is softmax(-1) + eps followed by a column pass. The
    // remaining 19 row/column passes retain the source ordering exactly.
    for row in 0..HC_MULT {
        let start = row * HC_MULT;
        let mut row_max = f64::NEG_INFINITY;
        for &value in &comb_f64[start..start + HC_MULT] {
            row_max = row_max.max(value);
        }
        let mut row_sum = 0.0_f64;
        for column in 0..HC_MULT {
            let index = start + column;
            comb_f64[index] = (comb_f64[index] - row_max).exp();
            row_sum += comb_f64[index];
        }
        if !(row_sum.is_finite() && row_sum > 0.0) {
            return Err(gravity(
                "layer-0 FP64 mHC-FFN authority initial softmax row is invalid",
            ));
        }
        for column in 0..HC_MULT {
            let index = start + column;
            comb_f64[index] = comb_f64[index] / row_sum + eps;
        }
    }
    normalize_comb_columns_f64_authority(&mut comb_f64, eps)?;
    for _ in 0..HC_SINKHORN_ITERS - 1 {
        normalize_comb_rows_f64_authority(&mut comb_f64, eps)?;
        normalize_comb_columns_f64_authority(&mut comb_f64, eps)?;
    }
    if pre_f64
        .iter()
        .chain(&post_f64)
        .chain(&comb_f64)
        .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "layer-0 FP64 mHC-FFN authority Sinkhorn produced a non-finite control",
        ));
    }
    Ok((pre_f64, post_f64, comb_f64))
}

fn normalize_comb_rows_f64_authority(comb: &mut [f64], eps: f64) -> Result<()> {
    for row in 0..HC_MULT {
        let start = row * HC_MULT;
        let mut sum = 0.0_f64;
        for &value in &comb[start..start + HC_MULT] {
            sum += value;
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(gravity(
                "layer-0 FP64 mHC-FFN authority row normalization sum is invalid",
            ));
        }
        for value in &mut comb[start..start + HC_MULT] {
            *value /= sum + eps;
        }
    }
    Ok(())
}

fn normalize_comb_columns_f64_authority(comb: &mut [f64], eps: f64) -> Result<()> {
    for column in 0..HC_MULT {
        let mut sum = 0.0_f64;
        for row in 0..HC_MULT {
            sum += comb[row * HC_MULT + column];
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(gravity(
                "layer-0 FP64 mHC-FFN authority column normalization sum is invalid",
            ));
        }
        for row in 0..HC_MULT {
            let index = row * HC_MULT + column;
            comb[index] /= sum + eps;
        }
    }
    Ok(())
}

fn sigmoid_f64_authority(value: f64) -> f64 {
    1.0_f64 / (1.0_f64 + (-value).exp())
}

/// Run the serial source-derived Gate diagnostic/transcription for layer 0.
///
/// It decodes the admitted BF16 rows and accumulates them in Rust in source
/// column order. Upstream `model.py` uses framework `F.linear`, so this is not
/// a claim about its exact framework instruction or reduction behavior. A
/// separately qualified external-logit path is available for a bound source
/// reference when a caller has verified its provenance.
pub fn layer0_hash_route_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
    ffn_norm_bf16_bits: &[u16],
) -> Result<Layer0HashRouteCpuResult> {
    layer0_hash_route_cpu_oracle_for_token(reader, PREFIX_TOKEN_ID, ffn_norm_bf16_bits)
}

/// Run the serial source-derived Gate diagnostic/transcription and the exact
/// hash-routing table row for a specific tokenizer ID. Layer 0 takes its six
/// IDs from `tid2eid[input_ids]`; the score calculation remains token-state
/// dependent and is deliberately not replaced by the hash row. This default
/// path preserves historical diagnostic behavior, not a framework-`F.linear`
/// arithmetic claim.
pub fn layer0_hash_route_cpu_oracle_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
) -> Result<Layer0HashRouteCpuResult> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity("layer-0 Gate input must be one BF16 hidden row"));
    }
    let gate_weight =
        read_bf16_tensor(reader, LAYER0_FFN_GATE_WEIGHT, ROUTED_EXPERTS * HIDDEN_SIZE)?;
    let input: Vec<f32> = ffn_norm_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "layer-0 Gate BF16 input contains a non-finite value",
        ));
    }
    let mut logits_f32 = Vec::with_capacity(ROUTED_EXPERTS);
    for row in 0..ROUTED_EXPERTS {
        let mut accumulator = 0.0_f32;
        let weights = &gate_weight[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
        for (&activation, &weight_bits) in input.iter().zip(weights) {
            let weight = bf16::from_bits(weight_bits).to_f32();
            if !weight.is_finite() {
                return Err(gravity(
                    "layer-0 Gate BF16 weight contains a non-finite value",
                ));
            }
            accumulator += activation * weight;
        }
        if !accumulator.is_finite() {
            return Err(gravity("layer-0 Gate logit is non-finite"));
        }
        logits_f32.push(accumulator);
    }

    layer0_hash_route_cpu_oracle_from_verified_logits_for_token(reader, token_id, &logits_f32)
}

/// Construct the source hash-route controls from a previously verified
/// external F32[256] Gate-logit vector. This deliberately accepts logits
/// only, not route IDs or weights: the admitted `tid2eid` row is reread and
/// the source sqrt-softplus and normalization arithmetic stay local.
///
/// The caller is responsible for proving the external-logit provenance and
/// that it belongs to the exact BF16 Gate input being diagnosed. In
/// particular, this helper does not replace the default serial source-Gate
/// path used by [`layer0_hash_route_cpu_oracle_for_token`].
pub fn layer0_hash_route_cpu_oracle_from_verified_logits_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    verified_logits_f32: &[f32],
) -> Result<Layer0HashRouteCpuResult> {
    if verified_logits_f32.len() != ROUTED_EXPERTS
        || verified_logits_f32.iter().any(|value| !value.is_finite())
    {
        return Err(gravity(
            "layer-0 verified external Gate logits must be finite F32[256]",
        ));
    }
    let logits_f32 = verified_logits_f32.to_vec();
    let original_scores_f32 = logits_f32
        .iter()
        .copied()
        .map(sqrt_softplus_source_algorithm)
        .collect::<Result<Vec<_>>>()?;

    let tid2eid = reader.tensor_metadata(LAYER0_FFN_GATE_TID2EID)?;
    if tid2eid.dtype != "I64"
        || tid2eid.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64]
        || tid2eid.bytes != (129_280 * ACTIVATED_EXPERTS * std::mem::size_of::<i64>()) as u64
    {
        return Err(gravity(
            "layer-0 tid2eid metadata differs from the source routing contract",
        ));
    }
    if token_id >= tid2eid.shape[0] {
        return Err(gravity(
            "layer-0 tid2eid token ID is outside the admitted source table",
        ));
    }
    let row_bytes = ACTIVATED_EXPERTS
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| gravity("layer-0 tid2eid row byte count overflow"))?;
    let start = usize::try_from(token_id)
        .map_err(|_| gravity("layer-0 tid2eid token ID does not fit host usize"))?
        .checked_mul(row_bytes)
        .ok_or_else(|| gravity("layer-0 tid2eid byte offset overflow"))?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| gravity("layer-0 tid2eid row end overflow"))?;
    let raw_ids =
        reader.read_verified_range(LAYER0_FFN_GATE_TID2EID, start as u64..end as u64, row_bytes)?;
    let mut selected_expert_ids = Vec::with_capacity(ACTIVATED_EXPERTS);
    for bytes in raw_ids.chunks_exact(std::mem::size_of::<i64>()) {
        let raw = i64::from_le_bytes(
            bytes
                .try_into()
                .map_err(|_| gravity("layer-0 tid2eid chunk is not a complete i64 index"))?,
        );
        if raw < 0 || raw >= ROUTED_EXPERTS as i64 {
            return Err(gravity(
                "layer-0 tid2eid contains an out-of-range expert ID",
            ));
        }
        selected_expert_ids.push(raw as u64);
    }
    if selected_expert_ids.len() != ACTIVATED_EXPERTS {
        return Err(gravity("layer-0 tid2eid read did not yield six expert IDs"));
    }
    let mut selected_weights_f32: Vec<f32> = selected_expert_ids
        .iter()
        .map(|&expert| original_scores_f32[expert as usize])
        .collect();
    let sum = selected_weights_f32.iter().copied().sum::<f32>();
    if !(sum.is_finite() && sum > 0.0) {
        return Err(gravity("layer-0 hash Gate selected-score sum is invalid"));
    }
    for weight in &mut selected_weights_f32 {
        *weight = (*weight / sum) * ROUTE_SCALE;
        if !weight.is_finite() {
            return Err(gravity("layer-0 hash Gate route weight is non-finite"));
        }
    }
    Ok(Layer0HashRouteCpuResult {
        token_id,
        logits_f32,
        original_scores_f32,
        selected_expert_ids,
        selected_weights_f32,
    })
}

/// Construct a CPU diagnostic route from a bounded, already-qualified source
/// target that includes every post-linear route value. This is crate-private:
/// only the calibration verifier may admit the externally supplied scores,
/// IDs, and weights. The verified artifact `tid2eid` row is still reread
/// through the existing logit route helper before the target is accepted.
pub(crate) fn layer0_hash_route_cpu_oracle_from_verified_gate_route_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    verified_logits_f32: &[f32],
    verified_original_scores_f32: &[f32],
    verified_selected_expert_ids: &[u16],
    verified_selected_weights_f32: &[f32],
) -> Result<Layer0HashRouteCpuResult> {
    if verified_original_scores_f32.len() != ROUTED_EXPERTS
        || verified_original_scores_f32
            .iter()
            .any(|value| !(value.is_finite() && *value > 0.0))
        || verified_selected_expert_ids.len() != ACTIVATED_EXPERTS
        || verified_selected_expert_ids
            .iter()
            .any(|&value| usize::from(value) >= ROUTED_EXPERTS)
        || verified_selected_weights_f32.len() != ACTIVATED_EXPERTS
        || verified_selected_weights_f32
            .iter()
            .any(|value| !(value.is_finite() && *value > 0.0))
    {
        return Err(gravity(
            "layer-0 verified external Gate route has invalid finite geometry",
        ));
    }
    // Reuse the strict verified-table read and finite-logit validation, but
    // retain the qualified source's post-linear score/weight values below.
    let table_route = layer0_hash_route_cpu_oracle_from_verified_logits_for_token(
        reader,
        token_id,
        verified_logits_f32,
    )?;
    let selected_expert_ids = verified_selected_expert_ids
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    if table_route.selected_expert_ids != selected_expert_ids {
        return Err(gravity(
            "layer-0 qualified external Gate-route IDs differ from verified tid2eid row",
        ));
    }
    let selected_weight_sum = verified_selected_weights_f32.iter().copied().sum::<f32>();
    if !selected_weight_sum.is_finite() || (selected_weight_sum - ROUTE_SCALE).abs() > 1.0e-5 {
        return Err(gravity(
            "layer-0 qualified external Gate-route weights do not sum to route scale",
        ));
    }
    Ok(Layer0HashRouteCpuResult {
        token_id,
        logits_f32: verified_logits_f32.to_vec(),
        original_scores_f32: verified_original_scores_f32.to_vec(),
        selected_expert_ids,
        selected_weights_f32: verified_selected_weights_f32.to_vec(),
    })
}

/// Independently accumulate the layer-0 Gate/hash-route controls in FP64 for
/// one actual BF16[4096] FFn-norm row and one source `tid2eid` token row.
///
/// Unlike [`layer0_hash_route_cpu_oracle_for_token`], this does not make the
/// source CPU F32 path an authority: every BF16 activation and weight is
/// decoded directly to FP64, every dot product accumulates in FP64, and the
/// sqrt-softplus/normalization steps also remain FP64.  The returned vectors
/// are therefore suitable as the reference argument to
/// [`crate::numeric_parity::score_pair`], with the source CPU F32 and device
/// F32 paths supplied separately as candidates.
///
/// This remains CPU-only diagnostic code.  It has no Metal dependency and
/// must only be invoked after a caller has completed any device graph whose
/// BF16 input it is diagnosing.
pub fn layer0_hash_route_f64_authority_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
) -> Result<Layer0HashRouteF64Authority> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 FP64 Gate authority input must be one BF16 hidden row",
        ));
    }
    let gate_weight =
        read_bf16_tensor(reader, LAYER0_FFN_GATE_WEIGHT, ROUTED_EXPERTS * HIDDEN_SIZE)?;
    let tid2eid = reader.tensor_metadata(LAYER0_FFN_GATE_TID2EID)?;
    if tid2eid.dtype != "I64"
        || tid2eid.shape.as_slice() != [129_280, ACTIVATED_EXPERTS as u64]
        || tid2eid.bytes != (129_280 * ACTIVATED_EXPERTS * std::mem::size_of::<i64>()) as u64
    {
        return Err(gravity(
            "layer-0 FP64 Gate authority tid2eid metadata differs from the source routing contract",
        ));
    }
    if token_id >= tid2eid.shape[0] {
        return Err(gravity(
            "layer-0 FP64 Gate authority token ID is outside the admitted source table",
        ));
    }
    let row_bytes = ACTIVATED_EXPERTS
        .checked_mul(std::mem::size_of::<i64>())
        .ok_or_else(|| gravity("layer-0 FP64 Gate authority tid2eid row byte count overflow"))?;
    let start = usize::try_from(token_id)
        .map_err(|_| gravity("layer-0 FP64 Gate authority token ID does not fit host usize"))?
        .checked_mul(row_bytes)
        .ok_or_else(|| gravity("layer-0 FP64 Gate authority tid2eid byte offset overflow"))?;
    let end = start
        .checked_add(row_bytes)
        .ok_or_else(|| gravity("layer-0 FP64 Gate authority tid2eid row end overflow"))?;
    let raw_ids =
        reader.read_verified_range(LAYER0_FFN_GATE_TID2EID, start as u64..end as u64, row_bytes)?;
    if raw_ids.len() != row_bytes {
        return Err(gravity(
            "layer-0 FP64 Gate authority tid2eid read has an unexpected byte count",
        ));
    }
    let selected_expert_ids = raw_ids
        .chunks_exact(std::mem::size_of::<i64>())
        .map(|bytes| {
            let raw =
                i64::from_le_bytes(bytes.try_into().map_err(|_| {
                    gravity("layer-0 FP64 Gate authority tid2eid chunk is malformed")
                })?);
            if !(0..ROUTED_EXPERTS as i64).contains(&raw) {
                return Err(gravity(
                    "layer-0 FP64 Gate authority tid2eid contains an out-of-range expert ID",
                ));
            }
            Ok(raw as u64)
        })
        .collect::<Result<Vec<_>>>()?;
    if selected_expert_ids.len() != ACTIVATED_EXPERTS {
        return Err(gravity(
            "layer-0 FP64 Gate authority tid2eid read did not yield six expert IDs",
        ));
    }
    layer0_hash_route_f64_from_verified_values(
        token_id,
        ffn_norm_bf16_bits,
        &gate_weight,
        selected_expert_ids,
    )
}

fn layer0_hash_route_f64_from_verified_values(
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
    gate_weight_bf16_bits: &[u16],
    selected_expert_ids: Vec<u64>,
) -> Result<Layer0HashRouteF64Authority> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 FP64 Gate authority input must be one BF16 hidden row",
        ));
    }
    if gate_weight_bf16_bits.len() != ROUTED_EXPERTS * HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 FP64 Gate authority weight matrix has an invalid BF16 geometry",
        ));
    }
    if selected_expert_ids.len() != ACTIVATED_EXPERTS
        || selected_expert_ids
            .iter()
            .any(|&expert| expert >= ROUTED_EXPERTS as u64)
    {
        return Err(gravity(
            "layer-0 FP64 Gate authority selected expert IDs have an invalid geometry or range",
        ));
    }

    let input: Vec<f64> = ffn_norm_bf16_bits
        .iter()
        .map(|&bits| f64::from(bf16::from_bits(bits).to_f32()))
        .collect();
    if input.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "layer-0 FP64 Gate authority BF16 input contains a non-finite value",
        ));
    }

    let mut logits_f64 = Vec::with_capacity(ROUTED_EXPERTS);
    let mut original_scores_f64 = Vec::with_capacity(ROUTED_EXPERTS);
    for row in 0..ROUTED_EXPERTS {
        let mut accumulator = 0.0_f64;
        let weights = &gate_weight_bf16_bits[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
        for (&activation, &weight_bits) in input.iter().zip(weights) {
            let weight = f64::from(bf16::from_bits(weight_bits).to_f32());
            if !weight.is_finite() {
                return Err(gravity(
                    "layer-0 FP64 Gate authority BF16 weight contains a non-finite value",
                ));
            }
            accumulator += activation * weight;
        }
        if !accumulator.is_finite() {
            return Err(gravity(
                "layer-0 FP64 Gate authority produced a non-finite logit",
            ));
        }
        logits_f64.push(accumulator);
        original_scores_f64.push(sqrt_softplus_f64_authority(accumulator)?);
    }

    let mut selected_weights_f64: Vec<f64> = selected_expert_ids
        .iter()
        .map(|&expert| original_scores_f64[expert as usize])
        .collect();
    let sum = selected_weights_f64.iter().sum::<f64>();
    if !(sum.is_finite() && sum > 0.0) {
        return Err(gravity(
            "layer-0 FP64 Gate authority selected-score sum is invalid",
        ));
    }
    for weight in &mut selected_weights_f64 {
        *weight = (*weight / sum) * f64::from(ROUTE_SCALE);
        if !weight.is_finite() {
            return Err(gravity(
                "layer-0 FP64 Gate authority route weight is non-finite",
            ));
        }
    }

    Ok(Layer0HashRouteF64Authority {
        token_id,
        logits_f64,
        original_scores_f64,
        selected_expert_ids,
        selected_weights_f64,
    })
}

/// Execute the source-F32 layer-0 MoE body from an already completed
/// FFn-norm BF16[4096] row and its exact hash-routing token ID.
///
/// This is deliberately the same-input companion to
/// [`layer0_moe_body_f64_authority_for_token`], rather than a wrapper around
/// [`layer0_moe_successor_cpu_oracle`] that silently recreates FFn-norm from
/// attention.  It supplies a valid source-F32 candidate for a P7 captured
/// row: Gate/route, six routed experts, shared expert, source numeric-order
/// combine, and the MoE BF16 store.  It intentionally stops before mHC-post
/// and is CPU-only verified-reader diagnostic code.
pub fn layer0_moe_body_f32_oracle_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
) -> Result<Layer0MoeBodyF32OracleResult> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 source-F32 MoE body oracle input must be one BF16 hidden row",
        ));
    }
    verify_layer0_moe_source_anchors(reader)?;
    let route = layer0_hash_route_cpu_oracle_for_token(reader, token_id, ffn_norm_bf16_bits)?;
    layer0_moe_body_f32_oracle_from_validated_route(reader, token_id, ffn_norm_bf16_bits, route)
}

/// Execute the source-F32 layer-0 MoE body from a completed BF16[4096] row
/// while deriving its route from externally verified F32[256] Gate logits.
///
/// This is an opt-in CPU diagnostic seam for a qualified, source-bound Gate
/// calibration. It does not alter the default serial-Gate oracle or any
/// device/runtime route; callers must first prove the external logits bind to
/// this exact completed BF16 input.
pub fn layer0_moe_body_f32_oracle_from_verified_gate_logits_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
    verified_gate_logits_f32: &[f32],
) -> Result<Layer0MoeBodyF32OracleResult> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 calibrated source-F32 MoE body input must be one BF16 hidden row",
        ));
    }
    verify_layer0_moe_source_anchors(reader)?;
    let route = layer0_hash_route_cpu_oracle_from_verified_logits_for_token(
        reader,
        token_id,
        verified_gate_logits_f32,
    )?;
    layer0_moe_body_f32_oracle_from_validated_route(reader, token_id, ffn_norm_bf16_bits, route)
}

/// Crate-private continuation for a fully qualified external source route.
/// It remains post-completion CPU diagnostic code and cannot inject values
/// into the Metal/runtime graph.
pub(crate) fn layer0_moe_body_f32_oracle_from_verified_gate_route_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
    verified_logits_f32: &[f32],
    verified_original_scores_f32: &[f32],
    verified_selected_expert_ids: &[u16],
    verified_selected_weights_f32: &[f32],
) -> Result<Layer0MoeBodyF32OracleResult> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 qualified route source-F32 MoE body input must be one BF16 hidden row",
        ));
    }
    verify_layer0_moe_source_anchors(reader)?;
    let route = layer0_hash_route_cpu_oracle_from_verified_gate_route_for_token(
        reader,
        token_id,
        verified_logits_f32,
        verified_original_scores_f32,
        verified_selected_expert_ids,
        verified_selected_weights_f32,
    )?;
    layer0_moe_body_f32_oracle_from_validated_route(reader, token_id, ffn_norm_bf16_bits, route)
}

fn layer0_moe_body_f32_oracle_from_validated_route(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
    route: Layer0HashRouteCpuResult,
) -> Result<Layer0MoeBodyF32OracleResult> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE
        || route.token_id != token_id
        || route.logits_f32.len() != ROUTED_EXPERTS
        || route.original_scores_f32.len() != ROUTED_EXPERTS
        || route.selected_expert_ids.len() != ACTIVATED_EXPERTS
        || route.selected_weights_f32.len() != ACTIVATED_EXPERTS
        || route
            .logits_f32
            .iter()
            .chain(&route.original_scores_f32)
            .chain(&route.selected_weights_f32)
            .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "layer-0 source-F32 MoE body received an invalid validated Gate route",
        ));
    }
    let ffn_norm_quantized = act_quant_bf16_ue8m0(ffn_norm_bf16_bits)?;
    let execution_slots = source_hash_expert_execution_slots(&route.selected_expert_ids)?;
    let routed_combine_order = execution_slots
        .iter()
        .map(|&source_top_slot| Layer0MoeCombineOrder {
            source_top_slot,
            expert_id: route.selected_expert_ids[source_top_slot],
        })
        .collect::<Vec<_>>();

    let mut moe_output_f32 = vec![0.0_f32; HIDDEN_SIZE];
    let mut routed_experts = Vec::with_capacity(ACTIVATED_EXPERTS);
    for &source_top_slot in &execution_slots {
        let expert_id = route.selected_expert_ids[source_top_slot];
        let route_weight = route.selected_weights_f32[source_top_slot];
        let expert = routed_expert_cpu_oracle(
            reader,
            source_top_slot,
            expert_id,
            route_weight,
            ffn_norm_bf16_bits,
        )?;
        // `y` is source F32.  Source iterates routed expert IDs in numeric
        // order, then adds the permanent shared contribution afterward.
        for (accumulator, &bits) in moe_output_f32.iter_mut().zip(&expert.down.output.bf16_bits) {
            *accumulator += bf16::from_bits(bits).to_f32();
        }
        routed_experts.push(expert);
    }

    let shared_expert = shared_expert_cpu_oracle(reader, ffn_norm_bf16_bits)?;
    for (accumulator, &bits) in moe_output_f32
        .iter_mut()
        .zip(&shared_expert.down.output.bf16_bits)
    {
        *accumulator += bf16::from_bits(bits).to_f32();
    }
    if moe_output_f32.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "layer-0 source-F32 MoE body source-order combine produced a non-finite value",
        ));
    }
    let moe_output_bf16_bits = moe_output_f32
        .iter()
        .copied()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    Ok(Layer0MoeBodyF32OracleResult {
        token_id,
        ffn_norm_quantized,
        route,
        routed_combine_order,
        routed_experts,
        shared_expert,
        moe_output_f32,
        moe_output_bf16_bits,
    })
}

fn source_hash_expert_execution_slots(selected_expert_ids: &[u64]) -> Result<Vec<usize>> {
    if selected_expert_ids.len() != ACTIVATED_EXPERTS
        || selected_expert_ids
            .iter()
            .any(|&expert_id| expert_id >= ROUTED_EXPERTS as u64)
    {
        return Err(gravity(
            "layer-0 source hash-route expert IDs differ from the pinned six-expert contract",
        ));
    }
    let mut execution_slots: Vec<usize> = (0..ACTIVATED_EXPERTS).collect();
    // The source loops numeric expert IDs.  Preserve input top-slot order if
    // the source hash row contains a duplicate ID.
    execution_slots.sort_unstable_by_key(|&slot| (selected_expert_ids[slot], slot));
    Ok(execution_slots)
}

/// Independently execute the complete layer-0 MoE *body* in FP64 from one
/// completed P7 FFn-norm BF16[4096] row and the exact token ID that indexes
/// the source hash-route table.
///
/// The authority deliberately ends after the source MoE BF16 store.  It does
/// not recreate or feed `hc_ffn_post`, create Metal resources, submit a
/// command buffer, or alter the P7/runtime path.  The public output keeps the
/// FP64 continuous vectors and the source-native activation/BF16 boundaries
/// needed to score completed device captures with Numeric Parity V2.1, while
/// verified raw weight windows are dropped after each projection.
pub fn layer0_moe_body_f64_authority_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
    ffn_norm_bf16_bits: &[u16],
) -> Result<Layer0MoeBodyF64Authority> {
    if ffn_norm_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "layer-0 FP64 MoE body authority input must be one BF16 hidden row",
        ));
    }
    // Bind this independent implementation to the same pinned source grammar
    // as the CPU oracle before accepting a runtime-captured row.
    verify_layer0_moe_source_anchors(reader)?;

    let route = layer0_hash_route_f64_authority_for_token(reader, token_id, ffn_norm_bf16_bits)?;
    let ffn_norm_quantized = act_quant_bf16_ue8m0(ffn_norm_bf16_bits)?;

    // Source `MoE.forward` iterates expert IDs in numeric order.  The helper
    // also preserves top-slot order if an admitted hash row has a duplicate.
    let execution_slots = source_hash_expert_execution_slots(&route.selected_expert_ids)?;
    let routed_combine_order = execution_slots
        .iter()
        .map(|&source_top_slot| Layer0MoeCombineOrder {
            source_top_slot,
            expert_id: route.selected_expert_ids[source_top_slot],
        })
        .collect::<Vec<_>>();

    let mut combined_f64 = vec![0.0_f64; HIDDEN_SIZE];
    let mut routed_experts = Vec::with_capacity(ACTIVATED_EXPERTS);
    for &source_top_slot in &execution_slots {
        let expert_id = route.selected_expert_ids[source_top_slot];
        let route_weight_f64 = route.selected_weights_f64[source_top_slot];
        let expert = routed_expert_f64_authority(
            reader,
            source_top_slot,
            expert_id,
            route_weight_f64,
            &ffn_norm_quantized,
        )?;
        add_bf16_row_to_f64_source_order(
            &mut combined_f64,
            &expert.w2.output_bf16_bits,
            "routed FP64 MoE W2",
        )?;
        routed_experts.push(expert);
    }

    let shared_expert = shared_expert_f64_authority(reader, &ffn_norm_quantized)?;
    // Source adds the permanent shared expert after the numeric-ID routed
    // loop.  Keep this a serial scalar loop rather than a reduction helper so
    // the arithmetic order is explicit and inspectable.
    add_bf16_row_to_f64_source_order(
        &mut combined_f64,
        &shared_expert.w2.output_bf16_bits,
        "shared FP64 MoE W2",
    )?;
    if combined_f64.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "layer-0 FP64 MoE body source-order combine produced a non-finite value",
        ));
    }
    let moe_output_bf16_bits =
        f64_values_to_source_bf16(&combined_f64, "layer-0 FP64 MoE body source-order combine")?;

    Ok(Layer0MoeBodyF64Authority {
        token_id,
        ffn_norm_quantized,
        route,
        routed_combine_order,
        routed_experts,
        shared_expert,
        combined_f64,
        moe_output_bf16_bits,
    })
}

fn routed_expert_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    source_top_slot: usize,
    expert_id: u64,
    route_weight_f64: f64,
    input_quantized: &ActQuantizedBf16Row,
) -> Result<Layer0MoeF64RoutedExpertAuthority> {
    if source_top_slot >= ACTIVATED_EXPERTS
        || expert_id >= ROUTED_EXPERTS as u64
        || !route_weight_f64.is_finite()
    {
        return Err(gravity(
            "layer-0 FP64 routed-expert authority invocation metadata is invalid",
        ));
    }
    let stem = format!("layers.0.ffn.experts.{expert_id}");
    let w1 = fp4_linear_f64_authority(
        reader,
        &format!("{stem}.w1.weight"),
        &format!("{stem}.w1.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_quantized,
    )?;
    let w3 = fp4_linear_f64_authority(
        reader,
        &format!("{stem}.w3.weight"),
        &format!("{stem}.w3.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_quantized,
    )?;
    let (swiglu_f64, swiglu_bf16_bits) = swiglu_f64_authority(
        &w1.output_bf16_bits,
        &w3.output_bf16_bits,
        Some(route_weight_f64),
    )?;
    let w2_input_quantized = act_quant_bf16_ue8m0(&swiglu_bf16_bits)?;
    let w2 = fp4_linear_f64_authority(
        reader,
        &format!("{stem}.w2.weight"),
        &format!("{stem}.w2.scale"),
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &w2_input_quantized,
    )?;
    Ok(Layer0MoeF64RoutedExpertAuthority {
        source_top_slot,
        expert_id,
        route_weight_f64,
        w1,
        w3,
        swiglu_f64,
        swiglu_bf16_bits,
        w2_input_quantized,
        w2,
    })
}

fn shared_expert_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    input_quantized: &ActQuantizedBf16Row,
) -> Result<Layer0MoeF64SharedExpertAuthority> {
    let stem = "layers.0.ffn.shared_experts";
    let w1 = fp8_linear_f64_authority(
        reader,
        &format!("{stem}.w1.weight"),
        &format!("{stem}.w1.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_quantized,
    )?;
    let w3 = fp8_linear_f64_authority(
        reader,
        &format!("{stem}.w3.weight"),
        &format!("{stem}.w3.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_quantized,
    )?;
    let (swiglu_f64, swiglu_bf16_bits) =
        swiglu_f64_authority(&w1.output_bf16_bits, &w3.output_bf16_bits, None)?;
    let w2_input_quantized = act_quant_bf16_ue8m0(&swiglu_bf16_bits)?;
    let w2 = fp8_linear_f64_authority(
        reader,
        &format!("{stem}.w2.weight"),
        &format!("{stem}.w2.scale"),
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &w2_input_quantized,
    )?;
    Ok(Layer0MoeF64SharedExpertAuthority {
        w1,
        w3,
        swiglu_f64,
        swiglu_bf16_bits,
        w2_input_quantized,
        w2,
    })
}

#[derive(Debug)]
struct Layer0MoeF64NativePair {
    kind: NativeScalePairKind,
    output_rows: usize,
    logical_k: usize,
    packed_k: usize,
    scale_rows: usize,
    scale_cols: usize,
    raw_weight: Vec<u8>,
    raw_scale: Vec<u8>,
}

fn fp4_linear_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input_quantized: &ActQuantizedBf16Row,
) -> Result<Layer0MoeF64LinearStage> {
    let pair = read_native_pair_f64_authority(
        reader,
        weight_name,
        scale_name,
        NativeScalePairKind::Fp4E2M1fnX2,
        output_rows,
        logical_k,
    )?;
    let output_f64 = fp4_matvec_f64_authority(input_quantized, &pair)?;
    let output_bf16_bits =
        f64_values_to_source_bf16(&output_f64, "layer-0 FP64 native-FP4 linear")?;
    Ok(Layer0MoeF64LinearStage {
        output_f64,
        output_bf16_bits,
    })
}

fn fp8_linear_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input_quantized: &ActQuantizedBf16Row,
) -> Result<Layer0MoeF64LinearStage> {
    let pair = read_native_pair_f64_authority(
        reader,
        weight_name,
        scale_name,
        NativeScalePairKind::Fp8E4M3fn,
        output_rows,
        logical_k,
    )?;
    let output_f64 = fp8_matvec_f64_authority(input_quantized, &pair)?;
    let output_bf16_bits =
        f64_values_to_source_bf16(&output_f64, "layer-0 FP64 native-FP8 linear")?;
    Ok(Layer0MoeF64LinearStage {
        output_f64,
        output_bf16_bits,
    })
}

fn read_native_pair_f64_authority(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    expected_kind: NativeScalePairKind,
    output_rows: usize,
    logical_k: usize,
) -> Result<Layer0MoeF64NativePair> {
    if output_rows == 0 || logical_k == 0 || logical_k % ACT_QUANT_BLOCK != 0 {
        return Err(gravity(
            "layer-0 FP64 native-linear authority has an invalid requested geometry",
        ));
    }
    let (packed_k, scale_rows, scale_cols, weight_dtype) = match expected_kind {
        NativeScalePairKind::Fp4E2M1fnX2 => {
            if logical_k % FP4_BLOCK != 0 {
                return Err(gravity(
                    "layer-0 FP64 native-FP4 authority logical K is not 32-aligned",
                ));
            }
            (logical_k / 2, output_rows, logical_k / FP4_BLOCK, "I8")
        }
        NativeScalePairKind::Fp8E4M3fn => {
            if output_rows % ACT_QUANT_BLOCK != 0 {
                return Err(gravity(
                    "layer-0 FP64 native-FP8 authority output rows are not 128-aligned",
                ));
            }
            (
                logical_k,
                output_rows / ACT_QUANT_BLOCK,
                logical_k / ACT_QUANT_BLOCK,
                "F8_E4M3",
            )
        }
    };
    let pair = reader.native_scale_pair(weight_name)?;
    if pair.kind != expected_kind
        || pair.weight.name != weight_name
        || pair.scale.name != scale_name
        || pair.weight.dtype != weight_dtype
        || pair.scale.dtype != "F8_E8M0"
        || pair.weight.shape.as_slice() != [output_rows as u64, packed_k as u64]
        || pair.scale.shape.as_slice() != [scale_rows as u64, scale_cols as u64]
        || pair.out_rows != output_rows as u64
        || pair.packed_k != packed_k as u64
        || pair.logical_k != logical_k as u64
        || pair.scale_rows != scale_rows as u64
        || pair.scale_cols != scale_cols as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native pair for the layer-0 FP64 MoE body authority"
        )));
    }
    let weight_bytes = output_rows
        .checked_mul(packed_k)
        .ok_or_else(|| gravity("layer-0 FP64 native-linear weight byte count overflow"))?;
    let scale_bytes = scale_rows
        .checked_mul(scale_cols)
        .ok_or_else(|| gravity("layer-0 FP64 native-linear scale byte count overflow"))?;
    let raw_weight = reader.read_verified_full(weight_name, weight_bytes)?;
    let raw_scale = reader.read_verified_full(scale_name, scale_bytes)?;
    if raw_weight.len() != weight_bytes || raw_scale.len() != scale_bytes {
        return Err(gravity(
            "layer-0 FP64 native-linear verified read returned an unexpected byte count",
        ));
    }
    Ok(Layer0MoeF64NativePair {
        kind: expected_kind,
        output_rows,
        logical_k,
        packed_k,
        scale_rows,
        scale_cols,
        raw_weight,
        raw_scale,
    })
}

fn fp4_matvec_f64_authority(
    activation: &ActQuantizedBf16Row,
    pair: &Layer0MoeF64NativePair,
) -> Result<Vec<f64>> {
    if pair.kind != NativeScalePairKind::Fp4E2M1fnX2
        || pair.logical_k == 0
        || pair.logical_k % FP4_BLOCK != 0
        || pair.logical_k % ACT_QUANT_BLOCK != 0
        || pair.packed_k != pair.logical_k / 2
        || pair.scale_rows != pair.output_rows
        || pair.scale_cols != pair.logical_k / FP4_BLOCK
        || activation.activation_e4m3fn.len() != pair.logical_k
        || activation.scales_e8m0fnu.len() != pair.logical_k / ACT_QUANT_BLOCK
        || activation.decoded_scales_f32.len() != pair.logical_k / ACT_QUANT_BLOCK
        || pair.raw_weight.len() != pair.output_rows * pair.packed_k
        || pair.raw_scale.len() != pair.scale_rows * pair.scale_cols
    {
        return Err(gravity(
            "layer-0 FP64 native-FP4 authority activation or weight geometry is invalid",
        ));
    }
    let mut output_f64 = Vec::with_capacity(pair.output_rows);
    for row in 0..pair.output_rows {
        let mut row_accumulator = 0.0_f64;
        for block in 0..pair.scale_cols {
            let mut block_accumulator = 0.0_f64;
            let start = block * FP4_BLOCK;
            for column in start..start + FP4_BLOCK {
                let packed = pair.raw_weight[row * pair.packed_k + column / 2];
                let nibble = if column & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                block_accumulator +=
                    f64::from(decode_e4m3fn(activation.activation_e4m3fn[column])?)
                        * decode_e2m1fn_f64_authority(nibble)?;
            }
            let activation_scale = f64::from(decode_e8m0fnu(
                activation.scales_e8m0fnu[block / (ACT_QUANT_BLOCK / FP4_BLOCK)],
            )?);
            let weight_scale = f64::from(decode_e8m0fnu(
                pair.raw_scale[row * pair.scale_cols + block],
            )?);
            row_accumulator += block_accumulator * activation_scale * weight_scale;
        }
        if !row_accumulator.is_finite() {
            return Err(gravity(
                "layer-0 FP64 native-FP4 authority produced a non-finite output",
            ));
        }
        output_f64.push(row_accumulator);
    }
    Ok(output_f64)
}

fn fp8_matvec_f64_authority(
    activation: &ActQuantizedBf16Row,
    pair: &Layer0MoeF64NativePair,
) -> Result<Vec<f64>> {
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.output_rows == 0
        || pair.output_rows % ACT_QUANT_BLOCK != 0
        || pair.logical_k == 0
        || pair.logical_k % ACT_QUANT_BLOCK != 0
        || pair.packed_k != pair.logical_k
        || pair.scale_rows != pair.output_rows / ACT_QUANT_BLOCK
        || pair.scale_cols != pair.logical_k / ACT_QUANT_BLOCK
        || activation.activation_e4m3fn.len() != pair.logical_k
        || activation.scales_e8m0fnu.len() != pair.logical_k / ACT_QUANT_BLOCK
        || activation.decoded_scales_f32.len() != pair.logical_k / ACT_QUANT_BLOCK
        || pair.raw_weight.len() != pair.output_rows * pair.logical_k
        || pair.raw_scale.len() != pair.scale_rows * pair.scale_cols
    {
        return Err(gravity(
            "layer-0 FP64 native-FP8 authority activation or weight geometry is invalid",
        ));
    }
    let mut output_f64 = Vec::with_capacity(pair.output_rows);
    for row in 0..pair.output_rows {
        let mut row_accumulator = 0.0_f64;
        for block in 0..pair.scale_cols {
            let mut block_accumulator = 0.0_f64;
            let start = block * ACT_QUANT_BLOCK;
            for column in start..start + ACT_QUANT_BLOCK {
                block_accumulator +=
                    f64::from(decode_e4m3fn(activation.activation_e4m3fn[column])?)
                        * f64::from(decode_e4m3fn(
                            pair.raw_weight[row * pair.logical_k + column],
                        )?);
            }
            let activation_scale = f64::from(decode_e8m0fnu(activation.scales_e8m0fnu[block])?);
            let weight_scale = f64::from(decode_e8m0fnu(
                pair.raw_scale[(row / ACT_QUANT_BLOCK) * pair.scale_cols + block],
            )?);
            row_accumulator += block_accumulator * activation_scale * weight_scale;
        }
        if !row_accumulator.is_finite() {
            return Err(gravity(
                "layer-0 FP64 native-FP8 authority produced a non-finite output",
            ));
        }
        output_f64.push(row_accumulator);
    }
    Ok(output_f64)
}

fn swiglu_f64_authority(
    gate_bf16_bits: &[u16],
    up_bf16_bits: &[u16],
    route_weight_f64: Option<f64>,
) -> Result<(Vec<f64>, Vec<u16>)> {
    if gate_bf16_bits.len() != MOE_INTER_DIM || up_bf16_bits.len() != MOE_INTER_DIM {
        return Err(gravity(
            "layer-0 FP64 SwiGLU authority geometry is not one 2048-wide expert row",
        ));
    }
    if route_weight_f64.is_some_and(|weight| !weight.is_finite()) {
        return Err(gravity(
            "layer-0 FP64 SwiGLU authority route weight is non-finite",
        ));
    }
    let mut swiglu_f64 = Vec::with_capacity(MOE_INTER_DIM);
    for (&gate_bits, &up_bits) in gate_bf16_bits.iter().zip(up_bf16_bits) {
        // Preserve the source boundary: W1/W3 have already stored BF16.  The
        // source clamps `gate` only above and `up` on both sides.
        let gate = f64::from(bf16::from_bits(gate_bits).to_f32()).min(f64::from(SWIGLU_LIMIT));
        let up = f64::from(bf16::from_bits(up_bits).to_f32())
            .clamp(-f64::from(SWIGLU_LIMIT), f64::from(SWIGLU_LIMIT));
        if !gate.is_finite() || !up.is_finite() {
            return Err(gravity(
                "layer-0 FP64 SwiGLU authority BF16 stage output is non-finite",
            ));
        }
        let mut value = silu_f64_source_order(gate) * up;
        if let Some(weight) = route_weight_f64 {
            value *= weight;
        }
        if !value.is_finite() {
            return Err(gravity(
                "layer-0 FP64 SwiGLU authority produced a non-finite activation",
            ));
        }
        swiglu_f64.push(value);
    }
    let swiglu_bf16_bits = f64_values_to_source_bf16(&swiglu_f64, "layer-0 FP64 SwiGLU authority")?;
    Ok((swiglu_f64, swiglu_bf16_bits))
}

fn silu_f64_source_order(value: f64) -> f64 {
    // Same algebraic branches as the source F32 authority, but retain the
    // continuous reference calculation in FP64.
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        value * exp / (1.0 + exp)
    }
}

fn decode_e2m1fn_f64_authority(nibble: u8) -> Result<f64> {
    if nibble > 0x0f {
        return Err(gravity(
            "layer-0 FP64 native-FP4 authority packed value exceeds a nibble",
        ));
    }
    const TABLE: [f64; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    Ok(TABLE[nibble as usize])
}

fn f64_values_to_source_bf16(values: &[f64], stage: &str) -> Result<Vec<u16>> {
    let mut bf16_bits = Vec::with_capacity(values.len());
    for &value in values {
        // The source tensor store is F32 -> BF16.  An explicit intermediate
        // makes that contract visible rather than relying on a hypothetical
        // direct F64 -> BF16 conversion with different halfway behavior.
        if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
            return Err(gravity(format!(
                "{stage} cannot be represented by the source F32-to-BF16 store"
            )));
        }
        let source_f32 = value as f32;
        if !source_f32.is_finite() {
            return Err(gravity(format!(
                "{stage} F64-to-F32 source-store conversion became non-finite"
            )));
        }
        bf16_bits.push(bf16::from_f32(source_f32).to_bits());
    }
    Ok(bf16_bits)
}

fn add_bf16_row_to_f64_source_order(
    accumulator: &mut [f64],
    contribution_bf16_bits: &[u16],
    stage: &str,
) -> Result<()> {
    if accumulator.len() != HIDDEN_SIZE || contribution_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(format!(
            "{stage} does not have one hidden-width BF16 contribution"
        )));
    }
    for (sum, &bits) in accumulator.iter_mut().zip(contribution_bf16_bits) {
        let contribution = f64::from(bf16::from_bits(bits).to_f32());
        if !contribution.is_finite() {
            return Err(gravity(format!("{stage} BF16 contribution is non-finite")));
        }
        *sum += contribution;
        if !sum.is_finite() {
            return Err(gravity(format!("{stage} source-order sum is non-finite")));
        }
    }
    Ok(())
}

fn routed_expert_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
    source_top_slot: usize,
    expert_id: u64,
    route_weight: f32,
    input_bf16_bits: &[u16],
) -> Result<RoutedExpertCpuResult> {
    if source_top_slot >= ACTIVATED_EXPERTS
        || expert_id >= ROUTED_EXPERTS as u64
        || !route_weight.is_finite()
    {
        return Err(gravity("routed-expert invocation metadata is invalid"));
    }
    let stem = format!("layers.0.ffn.experts.{expert_id}");
    let gate = fp4_linear_bf16_source_algorithm(
        reader,
        &format!("{stem}.w1.weight"),
        &format!("{stem}.w1.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_bf16_bits,
    )?;
    let up = fp4_linear_bf16_source_algorithm(
        reader,
        &format!("{stem}.w3.weight"),
        &format!("{stem}.w3.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_bf16_bits,
    )?;
    let weighted_swiglu_bf16_bits = swiglu_bf16_source_algorithm(
        &gate.output.bf16_bits,
        &up.output.bf16_bits,
        Some(route_weight),
    )?;
    let down = fp4_linear_bf16_source_algorithm(
        reader,
        &format!("{stem}.w2.weight"),
        &format!("{stem}.w2.scale"),
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &weighted_swiglu_bf16_bits,
    )?;
    Ok(RoutedExpertCpuResult {
        source_top_slot,
        expert_id,
        route_weight,
        gate,
        up,
        weighted_swiglu_bf16_bits,
        down,
    })
}

fn shared_expert_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
    input_bf16_bits: &[u16],
) -> Result<SharedExpertCpuResult> {
    let stem = "layers.0.ffn.shared_experts";
    let gate = fp8_linear_bf16_source_algorithm(
        reader,
        &format!("{stem}.w1.weight"),
        &format!("{stem}.w1.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_bf16_bits,
    )?;
    let up = fp8_linear_bf16_source_algorithm(
        reader,
        &format!("{stem}.w3.weight"),
        &format!("{stem}.w3.scale"),
        MOE_INTER_DIM,
        HIDDEN_SIZE,
        input_bf16_bits,
    )?;
    let swiglu_bf16_bits =
        swiglu_bf16_source_algorithm(&gate.output.bf16_bits, &up.output.bf16_bits, None)?;
    let down = fp8_linear_bf16_source_algorithm(
        reader,
        &format!("{stem}.w2.weight"),
        &format!("{stem}.w2.scale"),
        HIDDEN_SIZE,
        MOE_INTER_DIM,
        &swiglu_bf16_bits,
    )?;
    Ok(SharedExpertCpuResult {
        gate,
        up,
        swiglu_bf16_bits,
        down,
    })
}

/// Source `Expert.forward`: W1/W3 output storage is BF16, activation is
/// float32, the per-source configured SwiGLU clamps apply, the optional
/// router weight multiplies in float32, then W2 receives the BF16 cast.
pub fn swiglu_bf16_source_algorithm(
    gate_bf16_bits: &[u16],
    up_bf16_bits: &[u16],
    route_weight: Option<f32>,
) -> Result<Vec<u16>> {
    if gate_bf16_bits.len() != MOE_INTER_DIM || up_bf16_bits.len() != MOE_INTER_DIM {
        return Err(gravity(
            "SwiGLU source geometry is not one 2048-wide expert row",
        ));
    }
    if route_weight.is_some_and(|weight| !weight.is_finite()) {
        return Err(gravity("SwiGLU route weight is non-finite"));
    }
    let mut result = Vec::with_capacity(MOE_INTER_DIM);
    for (&gate_bits, &up_bits) in gate_bf16_bits.iter().zip(up_bf16_bits) {
        // The source clamps `up` on both sides and `gate` only above.
        let gate = bf16::from_bits(gate_bits).to_f32().min(SWIGLU_LIMIT);
        let up = bf16::from_bits(up_bits)
            .to_f32()
            .clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT);
        if !gate.is_finite() || !up.is_finite() {
            return Err(gravity("SwiGLU BF16 stage output is non-finite"));
        }
        let mut value = silu_source_algorithm(gate) * up;
        if let Some(weight) = route_weight {
            value *= weight;
        }
        if !value.is_finite() {
            return Err(gravity("SwiGLU activation is non-finite"));
        }
        result.push(bf16::from_f32(value).to_bits());
    }
    Ok(result)
}

/// Source-native FP4 matrix vector multiplication.  The scalar grouping
/// mirrors `kernel.py::fp4_gemm_kernel`: 32-K block dot product, then its
/// corresponding 128-K activation and 32-K weight scales.  E2M1 values are
/// exactly representable in the temporary FP8 cast used by that kernel.
pub fn fp4_e2m1fn_x2_ue8m0_matvec(
    activation: &ActQuantizedBf16Row,
    weights_e2m1fn_x2: &[u8],
    weight_scales_e8m0fnu: &[u8],
    output_rows: usize,
    logical_k: usize,
) -> Result<Fp8MatvecCpuResult> {
    if output_rows == 0
        || logical_k == 0
        || logical_k % ACT_QUANT_BLOCK != 0
        || logical_k % FP4_BLOCK != 0
        || activation.activation_e4m3fn.len() != logical_k
        || activation.scales_e8m0fnu.len() != logical_k / ACT_QUANT_BLOCK
        || activation.decoded_scales_f32.len() != logical_k / ACT_QUANT_BLOCK
    {
        return Err(gravity("FP4 source GEMV activation geometry is invalid"));
    }
    let packed_k = logical_k / 2;
    let scale_cols = logical_k / FP4_BLOCK;
    if weights_e2m1fn_x2.len()
        != output_rows
            .checked_mul(packed_k)
            .ok_or_else(|| gravity("FP4 source GEMV packed weight length overflow"))?
        || weight_scales_e8m0fnu.len()
            != output_rows
                .checked_mul(scale_cols)
                .ok_or_else(|| gravity("FP4 source GEMV scale length overflow"))?
    {
        return Err(gravity("FP4 source GEMV weight/scale geometry is invalid"));
    }
    let mut output = Vec::with_capacity(output_rows);
    for row in 0..output_rows {
        let packed_row = row * packed_k;
        let scale_row = row * scale_cols;
        let mut row_accumulator = 0.0_f32;
        for block in 0..scale_cols {
            let start = block * FP4_BLOCK;
            let mut block_accumulator = 0.0_f32;
            for column in start..start + FP4_BLOCK {
                let activation_value = decode_e4m3fn(activation.activation_e4m3fn[column])?;
                let packed = weights_e2m1fn_x2[packed_row + column / 2];
                let nibble = if column & 1 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                block_accumulator += activation_value * decode_e2m1fn(nibble)?;
            }
            let activation_scale =
                activation.decoded_scales_f32[block / (ACT_QUANT_BLOCK / FP4_BLOCK)];
            let weight_scale = decode_e8m0fnu(weight_scales_e8m0fnu[scale_row + block])?;
            row_accumulator += block_accumulator * activation_scale * weight_scale;
        }
        if !row_accumulator.is_finite() {
            return Err(gravity("FP4 source GEMV produced a non-finite output"));
        }
        output.push(row_accumulator);
    }
    let bf16_bits = output
        .iter()
        .copied()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    Ok(Fp8MatvecCpuResult {
        fp32: output,
        bf16_bits,
    })
}

fn fp4_linear_bf16_source_algorithm(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input_bf16_bits: &[u16],
) -> Result<QuantizedLinearCpuStage> {
    let pair = reader.native_scale_pair(weight_name)?;
    let packed_k = logical_k / 2;
    let scale_cols = logical_k / FP4_BLOCK;
    if pair.kind != NativeScalePairKind::Fp4E2M1fnX2
        || pair.weight.name != weight_name
        || pair.scale.name != scale_name
        || pair.weight.dtype != "I8"
        || pair.scale.dtype != "F8_E8M0"
        || pair.weight.shape.as_slice() != [output_rows as u64, packed_k as u64]
        || pair.scale.shape.as_slice() != [output_rows as u64, scale_cols as u64]
        || pair.out_rows != output_rows as u64
        || pair.packed_k != packed_k as u64
        || pair.logical_k != logical_k as u64
        || pair.scale_rows != output_rows as u64
        || pair.scale_cols != scale_cols as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native FP4/E8M0 source pair"
        )));
    }
    let quantized_input = act_quant_bf16_ue8m0(input_bf16_bits)?;
    let weights = reader.read_verified_full(
        weight_name,
        output_rows
            .checked_mul(packed_k)
            .ok_or_else(|| gravity("FP4 source weight bytes overflow"))?,
    )?;
    let scales = reader.read_verified_full(
        scale_name,
        output_rows
            .checked_mul(scale_cols)
            .ok_or_else(|| gravity("FP4 source scale bytes overflow"))?,
    )?;
    let output =
        fp4_e2m1fn_x2_ue8m0_matvec(&quantized_input, &weights, &scales, output_rows, logical_k)?;
    Ok(QuantizedLinearCpuStage {
        quantized_input,
        output,
    })
}

fn fp8_linear_bf16_source_algorithm(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input_bf16_bits: &[u16],
) -> Result<QuantizedLinearCpuStage> {
    let pair = reader.native_scale_pair(weight_name)?;
    let scale_rows = output_rows / ACT_QUANT_BLOCK;
    let scale_cols = logical_k / ACT_QUANT_BLOCK;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.weight.name != weight_name
        || pair.scale.name != scale_name
        || pair.weight.dtype != "F8_E4M3"
        || pair.scale.dtype != "F8_E8M0"
        || pair.weight.shape.as_slice() != [output_rows as u64, logical_k as u64]
        || pair.scale.shape.as_slice() != [scale_rows as u64, scale_cols as u64]
        || pair.out_rows != output_rows as u64
        || pair.logical_k != logical_k as u64
        || pair.scale_rows != scale_rows as u64
        || pair.scale_cols != scale_cols as u64
    {
        return Err(gravity(format!(
            "{weight_name} is not the expected native FP8/E8M0 source pair"
        )));
    }
    let quantized_input = act_quant_bf16_ue8m0(input_bf16_bits)?;
    let weights = reader.read_verified_full(
        weight_name,
        output_rows
            .checked_mul(logical_k)
            .ok_or_else(|| gravity("FP8 source weight bytes overflow"))?,
    )?;
    let scales = reader.read_verified_full(
        scale_name,
        scale_rows
            .checked_mul(scale_cols)
            .ok_or_else(|| gravity("FP8 source scale bytes overflow"))?,
    )?;
    let output =
        fp8_e4m3fn_ue8m0_matvec(&quantized_input, &weights, &scales, output_rows, logical_k)?;
    Ok(QuantizedLinearCpuStage {
        quantized_input,
        output,
    })
}

fn sqrt_softplus_source_algorithm(logit: f32) -> Result<f32> {
    if !logit.is_finite() {
        return Err(gravity("sqrt-softplus received a non-finite Gate logit"));
    }
    // `torch.nn.functional.softplus` defaults to beta=1, threshold=20.  The
    // threshold branch prevents an overflow and retains the source's linear
    // approximation for large positive logits.
    let softplus = if logit > 20.0 {
        logit
    } else if logit >= 0.0 {
        logit + (-logit).exp().ln_1p()
    } else {
        logit.exp().ln_1p()
    };
    let score = softplus.sqrt();
    if !(score.is_finite() && score > 0.0) {
        return Err(gravity("sqrt-softplus produced an invalid Gate score"));
    }
    Ok(score)
}

/// FP64 form of the source thresholded `sqrt(softplus(logit))` Gate score.
/// Kept separate from the CPU F32 implementation so an F32 accumulation or
/// libm result cannot become the continuous-value authority by accident.
fn sqrt_softplus_f64_authority(logit: f64) -> Result<f64> {
    if !logit.is_finite() {
        return Err(gravity(
            "FP64 sqrt-softplus authority received a non-finite Gate logit",
        ));
    }
    let softplus = if logit > 20.0 {
        logit
    } else if logit >= 0.0 {
        logit + (-logit).exp().ln_1p()
    } else {
        logit.exp().ln_1p()
    };
    let score = softplus.sqrt();
    if !(score.is_finite() && score > 0.0) {
        return Err(gravity(
            "FP64 sqrt-softplus authority produced an invalid Gate score",
        ));
    }
    Ok(score)
}

fn silu_source_algorithm(value: f32) -> f32 {
    // Algebraically `value * sigmoid(value)`, using a branch that avoids
    // unnecessary overflow in the source-visible `F.silu` range.
    if value >= 0.0 {
        value / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        value * exp / (1.0 + exp)
    }
}

fn decode_e2m1fn(nibble: u8) -> Result<f32> {
    if nibble > 0x0f {
        return Err(gravity("E2M1FN packed value exceeds a nibble"));
    }
    const TABLE: [f32; 16] = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    Ok(TABLE[nibble as usize])
}

fn expect_tensor<'a>(
    reader: &'a DeepSeekV4FullStreamReader,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<&'a crate::gravity_deepseek_v4::DeepSeekV4TensorMetadata> {
    let tensor = reader.tensor_metadata(name)?;
    let expected_bytes = shape.iter().try_fold(1u64, |accumulator, &dimension| {
        accumulator
            .checked_mul(dimension)
            .ok_or_else(|| gravity("tensor shape byte product overflow"))
    })? * match dtype {
        "BF16" => 2,
        "F32" => 4,
        "I64" => 8,
        _ => return Err(gravity("unsupported expected non-native tensor dtype")),
    };
    if tensor.dtype != dtype
        || tensor.shape.as_slice() != shape
        || tensor.bytes != expected_bytes
        || tensor.segments.is_empty()
    {
        return Err(gravity(format!(
            "{name} does not match the pinned layer-0 MoE tensor contract"
        )));
    }
    Ok(tensor)
}

fn read_bf16_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    elements: usize,
) -> Result<Vec<u16>> {
    let tensor = reader.tensor_metadata(name)?;
    let expected_bytes = elements
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| gravity("BF16 tensor byte count overflow"))?;
    if tensor.dtype != "BF16" || tensor.bytes != expected_bytes as u64 {
        return Err(gravity(format!("{name} is not the expected BF16 tensor")));
    }
    let bytes = reader.read_verified_full(name, expected_bytes)?;
    decode_u16_le(&bytes, name)
}

fn read_f32_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>> {
    let tensor = reader.tensor_metadata(name)?;
    let expected_bytes = elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| gravity("F32 tensor byte count overflow"))?;
    if tensor.dtype != "F32" || tensor.bytes != expected_bytes as u64 {
        return Err(gravity(format!("{name} is not the expected F32 tensor")));
    }
    let bytes = reader.read_verified_full(name, expected_bytes)?;
    let values = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four-byte f32 chunk")))
        .collect::<Vec<_>>();
    if values.len() != elements || values.iter().any(|value| !value.is_finite()) {
        return Err(gravity(format!(
            "{name} F32 values are malformed or non-finite"
        )));
    }
    Ok(values)
}

fn decode_u16_le(bytes: &[u8], name: &str) -> Result<Vec<u16>> {
    if bytes.len() % std::mem::size_of::<u16>() != 0 {
        return Err(gravity(format!("{name} is not aligned as BF16 bytes")));
    }
    let values = bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes(chunk.try_into().expect("two-byte BF16 chunk")))
        .collect::<Vec<_>>();
    if values
        .iter()
        .any(|bits| !bf16::from_bits(*bits).to_f32().is_finite())
    {
        return Err(gravity(format!("{name} contains a non-finite BF16 value")));
    }
    Ok(values)
}

fn parse_json(bytes: &[u8], name: &str) -> Result<Value> {
    serde_json::from_slice(bytes)
        .map_err(|error| gravity(format!("{name} is not valid JSON: {error}")))
}

fn json_path<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a Value> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| gravity(format!("pinned source config lacks {label}")))?;
    }
    Ok(current)
}

fn json_u64(value: &Value, path: &[&str], label: &str) -> Result<u64> {
    json_path(value, path, label)?.as_u64().ok_or_else(|| {
        gravity(format!(
            "pinned source config {label} is not an unsigned integer"
        ))
    })
}

fn json_string<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a str> {
    json_path(value, path, label)?
        .as_str()
        .ok_or_else(|| gravity(format!("pinned source config {label} is not a string")))
}

fn json_f64_eq(value: &Value, path: &[&str], expected: f32) -> bool {
    json_path(value, path, "floating value")
        .ok()
        .and_then(Value::as_f64)
        .map(|actual| (actual as f32).to_bits() == expected.to_bits())
        .unwrap_or(false)
}

fn gravity(message: impl Into<String>) -> Error {
    Error::Gravity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1fn_table_matches_native_fp4_contract() {
        let expected = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for (bits, value) in expected.into_iter().enumerate() {
            assert_eq!(decode_e2m1fn(bits as u8).unwrap(), value);
        }
        assert!(decode_e2m1fn(16).is_err());
    }

    #[test]
    fn source_sqrt_softplus_keeps_positive_finite_scores() {
        for value in [-100.0, -3.0, 0.0, 3.0, 20.0, 100.0] {
            assert!(sqrt_softplus_source_algorithm(value).unwrap().is_finite());
            assert!(sqrt_softplus_source_algorithm(value).unwrap() > 0.0);
        }
    }

    #[test]
    fn f64_gate_authority_is_score_pair_ready_for_a_bf16_row() {
        let input = vec![bf16::from_f32(0.0).to_bits(); HIDDEN_SIZE];
        let weights = vec![bf16::from_f32(0.0).to_bits(); ROUTED_EXPERTS * HIDDEN_SIZE];
        let selected_ids = vec![72, 168, 184, 142, 174, 177];
        let authority = layer0_hash_route_f64_from_verified_values(
            19_923,
            &input,
            &weights,
            selected_ids.clone(),
        )
        .unwrap();

        assert_eq!(authority.token_id, 19_923);
        assert_eq!(authority.logits_f64.len(), ROUTED_EXPERTS);
        assert_eq!(authority.original_scores_f64.len(), ROUTED_EXPERTS);
        assert_eq!(authority.selected_expert_ids, selected_ids);
        assert_eq!(authority.selected_weights_f64.len(), ACTIVATED_EXPERTS);
        assert!(authority.logits_f64.iter().all(|&value| value == 0.0));
        assert!(authority
            .original_scores_f64
            .iter()
            .all(|value| value.is_finite() && *value > 0.0));
        assert_eq!(authority.selected_weights_f64.iter().sum::<f64>(), 1.5);

        let host_logits: Vec<f32> = authority
            .logits_f64
            .iter()
            .map(|&value| value as f32)
            .collect();
        let score = crate::numeric_parity::score_pair(
            &host_logits,
            &host_logits,
            &authority.logits_f64,
            &crate::numeric_parity::Bounds::continuous_only(),
        );
        assert!(
            score.pass,
            "authority must be usable by score_pair: {score:?}"
        );
    }

    #[test]
    fn f64_gate_authority_rejects_non_bf16_hidden_geometry() {
        let error = layer0_hash_route_f64_from_verified_values(
            19_923,
            &[],
            &[],
            vec![72, 168, 184, 142, 174, 177],
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("FP64 Gate authority input must be one BF16 hidden row"));
    }

    #[test]
    fn direct_row_body_uses_source_numeric_expert_order() {
        let ids = [72, 168, 184, 142, 174, 177];
        assert_eq!(
            source_hash_expert_execution_slots(&ids).unwrap(),
            vec![0, 3, 1, 4, 5, 2]
        );
        assert!(source_hash_expert_execution_slots(&ids[..ACTIVATED_EXPERTS - 1]).is_err());
    }

    #[test]
    fn f64_mhc_ffn_authority_preserves_the_reduced_bf16_store_boundary() {
        let attention = vec![bf16::from_f32(1.0).to_bits(); HC_FLAT_WIDTH];
        let hc_fn = vec![0.0_f32; HC_MIX_WIDTH * HC_FLAT_WIDTH];
        let hc_scale = vec![1.0_f32; 3];
        let hc_base = vec![0.0_f32; HC_MIX_WIDTH];
        let authority = layer0_mhc_ffn_control_f64_from_verified_controls(
            &attention, &hc_fn, &hc_scale, &hc_base,
        )
        .unwrap();

        assert!(authority.flat_rsqrt_f64.is_finite());
        assert_eq!(authority.mixes_f64.len(), HC_MIX_WIDTH);
        assert_eq!(authority.pre_f64.len(), HC_MULT);
        assert_eq!(authority.post_f64.len(), HC_MULT);
        assert_eq!(authority.comb_f64.len(), HC_MULT * HC_MULT);
        assert_eq!(authority.reduced_bf16_bits.len(), HIDDEN_SIZE);
        assert!(authority.mixes_f64.iter().all(|&value| value == 0.0));
        assert!(authority
            .pre_f64
            .iter()
            .chain(&authority.post_f64)
            .chain(&authority.comb_f64)
            .all(|value| value.is_finite()));
        assert!(authority
            .reduced_bf16_bits
            .iter()
            .all(|&bits| bits == bf16::from_f32(2.0).to_bits()));

        let host_mixes: Vec<f32> = authority
            .mixes_f64
            .iter()
            .map(|&value| value as f32)
            .collect();
        let score = crate::numeric_parity::score_pair(
            &host_mixes,
            &host_mixes,
            &authority.mixes_f64,
            &crate::numeric_parity::Bounds::continuous_only(),
        );
        assert!(
            score.pass,
            "authority controls must be score_pair-ready: {score:?}"
        );
    }

    #[test]
    fn f64_mhc_ffn_authority_rejects_non_hc_attention_geometry() {
        let error =
            layer0_mhc_ffn_control_f64_from_verified_controls(&[], &[], &[], &[]).unwrap_err();
        assert!(error
            .to_string()
            .contains("FP64 mHC-FFN authority controls differ from the pinned source geometry"));
    }

    #[test]
    fn f64_mhc_ffn_post_authority_uses_comb_columns_and_bf16_store() {
        let mut attention = Vec::with_capacity(HC_FLAT_WIDTH);
        for lane in 0..HC_MULT {
            attention.extend(std::iter::repeat_n(
                bf16::from_f32(lane as f32).to_bits(),
                HIDDEN_SIZE,
            ));
        }
        let moe = vec![bf16::from_f32(1.0).to_bits(); HIDDEN_SIZE];
        let mut comb = vec![0.0_f64; HC_MULT * HC_MULT];
        for lane in 0..HC_MULT {
            comb[lane * HC_MULT + lane] = 1.0;
        }
        let controls = Layer0MhcFfnControlF64Authority {
            flat_rsqrt_f64: 1.0,
            mixes_f64: vec![0.0; HC_MIX_WIDTH],
            pre_f64: vec![0.0; HC_MULT],
            post_f64: vec![1.0; HC_MULT],
            comb_f64: comb,
            reduced_bf16_bits: vec![bf16::from_f32(0.0).to_bits(); HIDDEN_SIZE],
        };
        let authority =
            layer0_mhc_ffn_post_f64_from_control_authority(&attention, &moe, controls).unwrap();

        assert_eq!(authority.child_state_f64.len(), HC_FLAT_WIDTH);
        assert_eq!(authority.child_state_bf16_bits.len(), HC_FLAT_WIDTH);
        for lane in 0..HC_MULT {
            let expected = 1.0 + lane as f64;
            assert_eq!(authority.child_state_f64[lane * HIDDEN_SIZE], expected);
            assert_eq!(
                authority.child_state_bf16_bits[lane * HIDDEN_SIZE],
                bf16::from_f32(expected as f32).to_bits()
            );
        }
    }

    #[test]
    fn f64_mhc_ffn_post_authority_rejects_invalid_capture_geometry() {
        let controls = Layer0MhcFfnControlF64Authority {
            flat_rsqrt_f64: 1.0,
            mixes_f64: vec![0.0; HC_MIX_WIDTH],
            pre_f64: vec![0.0; HC_MULT],
            post_f64: vec![0.0; HC_MULT],
            comb_f64: vec![0.0; HC_MULT * HC_MULT],
            reduced_bf16_bits: vec![bf16::from_f32(0.0).to_bits(); HIDDEN_SIZE],
        };
        let error = layer0_mhc_ffn_post_f64_from_control_authority(&[], &[], controls).unwrap_err();
        assert!(error.to_string().contains(
            "FP64 mHC-FFN-post authority controls differ from the pinned source geometry"
        ));
    }

    #[test]
    fn f64_fp4_authority_decodes_native_bytes_before_the_bf16_store() {
        let input = vec![bf16::from_f32(1.0).to_bits(); ACT_QUANT_BLOCK];
        let quantized = act_quant_bf16_ue8m0(&input).unwrap();
        let pair = Layer0MoeF64NativePair {
            kind: NativeScalePairKind::Fp4E2M1fnX2,
            output_rows: 1,
            logical_k: ACT_QUANT_BLOCK,
            packed_k: ACT_QUANT_BLOCK / 2,
            scale_rows: 1,
            scale_cols: ACT_QUANT_BLOCK / FP4_BLOCK,
            // Both low/high nibbles are E2M1FN `1.0`.
            raw_weight: vec![0x22; ACT_QUANT_BLOCK / 2],
            // E8M0 exponent 127 is exactly one.
            raw_scale: vec![127; ACT_QUANT_BLOCK / FP4_BLOCK],
        };
        let output = fp4_matvec_f64_authority(&quantized, &pair).unwrap();
        assert_eq!(output, vec![128.0]);
        assert_eq!(
            f64_values_to_source_bf16(&output, "test").unwrap(),
            vec![bf16::from_f32(128.0).to_bits()]
        );
    }

    #[test]
    fn f64_fp8_authority_decodes_native_bytes_before_the_bf16_store() {
        let input = vec![bf16::from_f32(1.0).to_bits(); ACT_QUANT_BLOCK];
        let quantized = act_quant_bf16_ue8m0(&input).unwrap();
        let pair = Layer0MoeF64NativePair {
            kind: NativeScalePairKind::Fp8E4M3fn,
            output_rows: ACT_QUANT_BLOCK,
            logical_k: ACT_QUANT_BLOCK,
            packed_k: ACT_QUANT_BLOCK,
            scale_rows: 1,
            scale_cols: 1,
            // E4M3FN exponent 7 / mantissa 0 is exactly one.
            raw_weight: vec![0x38; ACT_QUANT_BLOCK * ACT_QUANT_BLOCK],
            raw_scale: vec![127],
        };
        let output = fp8_matvec_f64_authority(&quantized, &pair).unwrap();
        assert_eq!(output.len(), ACT_QUANT_BLOCK);
        assert!(output.iter().all(|&value| value == 128.0));
        assert!(f64_values_to_source_bf16(&output, "test")
            .unwrap()
            .iter()
            .all(|&bits| bits == bf16::from_f32(128.0).to_bits()));
    }

    #[test]
    fn f64_swiglu_authority_uses_bf16_inputs_and_f64_route_weight() {
        let gate = vec![bf16::from_f32(1.0).to_bits(); MOE_INTER_DIM];
        let up = vec![bf16::from_f32(2.0).to_bits(); MOE_INTER_DIM];
        let route_weight = 0.5_f64;
        let (continuous, stored) = swiglu_f64_authority(&gate, &up, Some(route_weight)).unwrap();
        let expected = silu_f64_source_order(1.0) * 2.0 * route_weight;
        assert!(continuous
            .iter()
            .all(|&value| (value - expected).abs() < f64::EPSILON));
        assert!(stored
            .iter()
            .all(|&bits| bits == bf16::from_f32(expected as f32).to_bits()));
    }

    #[test]
    fn f64_combine_is_serial_and_rejects_non_hidden_width_rows() {
        let mut sum = vec![0.0_f64; HIDDEN_SIZE];
        let one = vec![bf16::from_f32(1.0).to_bits(); HIDDEN_SIZE];
        let two = vec![bf16::from_f32(2.0).to_bits(); HIDDEN_SIZE];
        add_bf16_row_to_f64_source_order(&mut sum, &one, "first").unwrap();
        add_bf16_row_to_f64_source_order(&mut sum, &two, "second").unwrap();
        assert!(sum.iter().all(|&value| value == 3.0));
        assert!(add_bf16_row_to_f64_source_order(&mut sum, &[], "bad").is_err());
    }

    #[test]
    fn swiglu_clamps_then_applies_route_weight_before_bf16() {
        let gate = vec![bf16::from_f32(100.0).to_bits(); MOE_INTER_DIM];
        let up = vec![bf16::from_f32(-100.0).to_bits(); MOE_INTER_DIM];
        let output = swiglu_bf16_source_algorithm(&gate, &up, Some(0.5)).unwrap();
        let value = bf16::from_bits(output[0]).to_f32();
        assert!(value.is_finite());
        assert!(value < 0.0);
        assert_eq!(output.len(), MOE_INTER_DIM);
    }
}
