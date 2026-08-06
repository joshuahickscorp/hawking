//! CPU-only source-algorithm checkpoint for one complete DeepSeek-V4 layer-0
//! attention path at the bounded `(batch=1, sequence=1, position=0)` point.
//!
//! This is intentionally a *parity ladder* rung, not an engine.  It consumes
//! the sealed content-addressed full stream through the admitted reader and
//! transcribes the pinned Python/TileLang grammar below:
//!
//! ```text
//! BOS embedding / mHC-attn-pre / attn RMSNorm
//!   -> WQ-A / Q RMSNorm / WQ-B / per-head Q RMSNorm
//!   -> WKV / KV RMSNorm / non-RoPE FP8 QAT simulation
//!   -> position-0, ratio-0 sparse attention
//!   -> converted WO-A BF16 einsum / WO-B
//!   -> mHC-attn-post
//! ```
//!
//! It does not execute PyTorch, TileLang, a full model, an Engine, Metal, an
//! endpoint, or a token loop.  In particular, the scalar reduction order is a
//! source-derived CPU algorithm reference, **not** a claim of bit-identical
//! upstream CUDA/TileLang execution.

use crate::gravity_deepseek_v4::{DeepSeekV4FullStreamReader, NativeScalePairKind};
use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, decode_e4m3fn, decode_e8m0fnu, encode_e4m3fn_rne, layer0_wq_a_cpu_oracle,
    rounded_ue8m0_scale_byte, ActQuantizedBf16Row, Fp8MatvecCpuResult, Layer0WqACpuOracleResult,
    ACT_QUANT_BLOCK,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    layer0_prefix_cpu_oracle, verify_layer0_prefix_source_anchors,
    DeepSeekV4Layer0PrefixSourceAnchors, Layer0PrefixCpuOracleResult, HC_MULT, HIDDEN_SIZE,
    RMS_NORM_EPS,
};
use crate::{Error, Result};
use half::bf16;
use serde_json::Value;

/// Hash of the pinned source conversion script.  Its WO-A conversion is part
/// of this checkpoint's source contract: the parent tensor is native FP8 but
/// `convert.py` materializes a BF16 WO-A parameter before `model.py` calls
/// `torch.einsum`.
pub const OFFICIAL_INFERENCE_CONVERT_PY_SHA256: &str =
    "912acfc20bdd9ae4dbd5bde9dc7c8e61f6d27b6826d3ac2d052b2534c0881454";

pub const LAYER0_Q_NORM_WEIGHT: &str = "layers.0.attn.q_norm.weight";
pub const LAYER0_WQ_B_WEIGHT: &str = "layers.0.attn.wq_b.weight";
pub const LAYER0_WQ_B_SCALE: &str = "layers.0.attn.wq_b.scale";
pub const LAYER0_WKV_WEIGHT: &str = "layers.0.attn.wkv.weight";
pub const LAYER0_WKV_SCALE: &str = "layers.0.attn.wkv.scale";
pub const LAYER0_KV_NORM_WEIGHT: &str = "layers.0.attn.kv_norm.weight";
pub const LAYER0_ATTN_SINK: &str = "layers.0.attn.attn_sink";
pub const LAYER0_WO_A_WEIGHT: &str = "layers.0.attn.wo_a.weight";
pub const LAYER0_WO_A_SCALE: &str = "layers.0.attn.wo_a.scale";
pub const LAYER0_WO_B_WEIGHT: &str = "layers.0.attn.wo_b.weight";
pub const LAYER0_WO_B_SCALE: &str = "layers.0.attn.wo_b.scale";

pub const Q_LORA_RANK: usize = 1024;
pub const NUM_HEADS: usize = 64;
pub const HEAD_DIM: usize = 512;
pub const ROPE_HEAD_DIM: usize = 64;
pub const NON_ROPE_HEAD_DIM: usize = HEAD_DIM - ROPE_HEAD_DIM;
pub const O_GROUPS: usize = 8;
pub const O_LORA_RANK: usize = 1024;
pub const WINDOW_SIZE: usize = 128;
pub const LAYER0_COMPRESS_RATIO: usize = 0;
pub const KV_QAT_BLOCK: usize = 64;
pub const WQ_B_ROWS: usize = NUM_HEADS * HEAD_DIM;
pub const WKV_ROWS: usize = HEAD_DIM;
pub const WO_A_ROWS: usize = O_GROUPS * O_LORA_RANK;
pub const WO_A_COLS: usize = NUM_HEADS * HEAD_DIM / O_GROUPS;
pub const WO_B_ROWS: usize = HIDDEN_SIZE;
pub const WO_B_COLS: usize = O_GROUPS * O_LORA_RANK;

/// The source bindings beyond the prefix checkpoint required to interpret the
/// attention and converted-WO-A paths.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Layer0AttentionSourceAnchors {
    pub prefix: DeepSeekV4Layer0PrefixSourceAnchors,
    pub inference_convert_py_sha256: String,
}

/// A source FP8 `linear` call, including its real source-native activation
/// quantization bytes and BF16 output checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp8LinearCpuStage {
    pub quantized_input: ActQuantizedBf16Row,
    pub output: Fp8MatvecCpuResult,
}

/// The in-place QAT call used for the non-RoPE part of layer-0 KV.  The source
/// retains the last 64 positional dimensions as BF16 and writes BF16
/// quantize/dequantize results into the first 448 dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct KvInplaceQatCpuResult {
    pub output_bf16_bits: Vec<u16>,
    pub non_rope_activation_e4m3fn: Vec<u8>,
    pub non_rope_scales_e8m0fnu: Vec<u8>,
}

/// Bounded complete layer-0 attention checkpoint.  Values are retained only
/// long enough for the receipt producer to make hashes and finite statistics;
/// this is one fixed tokenizer-bound BOS trace, not a prompt trace store.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0AttentionCpuOracleResult {
    pub prefix: Layer0PrefixCpuOracleResult,
    pub wq_a: Layer0WqACpuOracleResult,
    pub q_norm_bf16_bits: Vec<u16>,
    pub wq_b: Fp8LinearCpuStage,
    pub q_head_norm_bf16_bits: Vec<u16>,
    pub q_position0_rope_bf16_bits: Vec<u16>,
    pub wkv: Fp8LinearCpuStage,
    pub kv_norm_bf16_bits: Vec<u16>,
    pub kv_inplace_qat: KvInplaceQatCpuResult,
    pub kv_position0_rope_bf16_bits: Vec<u16>,
    pub sparse_attention_scores_f32: Vec<f32>,
    pub sparse_attention_sink_denominators_f32: Vec<f32>,
    pub sparse_attention_bf16_bits: Vec<u16>,
    pub sparse_attention_derotated_bf16_bits: Vec<u16>,
    pub wo_a_bf16_bits: Vec<u16>,
    pub wo_b: Fp8LinearCpuStage,
    pub hc_attn_post_bf16_bits: Vec<u16>,
}

/// Validate the pinned source grammar and all constants that make the
/// position-zero, ratio-zero attention path unambiguous.
pub fn verify_layer0_attention_source_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4Layer0AttentionSourceAnchors> {
    let prefix = verify_layer0_prefix_source_anchors(reader)?;
    let inference_convert_py_sha256 = reader
        .source_metadata_asset_sha256("inference/convert.py")?
        .to_owned();
    if inference_convert_py_sha256 != OFFICIAL_INFERENCE_CONVERT_PY_SHA256 {
        return Err(gravity(
            "layer-0 attention convert.py differs from the pinned official source anchor",
        ));
    }

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
    let convert_py = reader.read_verified_metadata_asset("inference/convert.py", 128 * 1024)?;

    if json_u64(&inference_config, &["dim"], "inference dim")? != HIDDEN_SIZE as u64
        || json_u64(&inference_config, &["n_layers"], "inference layer count")? != 43
        || json_u64(&inference_config, &["n_heads"], "inference heads")? != NUM_HEADS as u64
        || json_u64(&inference_config, &["q_lora_rank"], "inference q_lora_rank")?
            != Q_LORA_RANK as u64
        || json_u64(&inference_config, &["head_dim"], "inference head_dim")? != HEAD_DIM as u64
        || json_u64(
            &inference_config,
            &["rope_head_dim"],
            "inference rope_head_dim",
        )? != ROPE_HEAD_DIM as u64
        || json_u64(&inference_config, &["o_groups"], "inference o_groups")? != O_GROUPS as u64
        || json_u64(&inference_config, &["o_lora_rank"], "inference o_lora_rank")?
            != O_LORA_RANK as u64
        || json_u64(&inference_config, &["window_size"], "inference window_size")?
            != WINDOW_SIZE as u64
        || json_string(&inference_config, &["dtype"], "inference dtype")? != "fp8"
        || json_string(&inference_config, &["scale_fmt"], "inference scale_fmt")? != "ue8m0"
        || json_array_u64(
            &inference_config,
            &["compress_ratios"],
            "inference compress_ratios",
        )?
        .first()
        .copied()
            != Some(LAYER0_COMPRESS_RATIO as u64)
        || json_u64(&model_config, &["hidden_size"], "model hidden_size")? != HIDDEN_SIZE as u64
        || json_u64(&model_config, &["num_hidden_layers"], "model layer count")? != 43
        || json_u64(&model_config, &["num_attention_heads"], "model heads")? != NUM_HEADS as u64
        || json_u64(&model_config, &["q_lora_rank"], "model q_lora_rank")? != Q_LORA_RANK as u64
        || json_u64(&model_config, &["head_dim"], "model head_dim")? != HEAD_DIM as u64
        || json_u64(&model_config, &["qk_rope_head_dim"], "model rope_head_dim")?
            != ROPE_HEAD_DIM as u64
        || json_u64(&model_config, &["o_groups"], "model o_groups")? != O_GROUPS as u64
        || json_u64(&model_config, &["o_lora_rank"], "model o_lora_rank")? != O_LORA_RANK as u64
        || json_u64(&model_config, &["sliding_window"], "model sliding_window")?
            != WINDOW_SIZE as u64
        || !json_f64_eq(&model_config, &["rms_norm_eps"], RMS_NORM_EPS)
        || json_string(
            &model_config,
            &["quantization_config", "scale_fmt"],
            "model FP8 scale format",
        )? != "ue8m0"
        || json_string(
            &model_config,
            &["quantization_config", "fmt"],
            "model FP8 format",
        )? != "e4m3"
        || json_array_u64(
            &model_config,
            &["quantization_config", "weight_block_size"],
            "model FP8 weight block size",
        )? != vec![ACT_QUANT_BLOCK as u64, ACT_QUANT_BLOCK as u64]
    {
        return Err(gravity(
            "pinned source configs differ from the layer-0 attention contract",
        ));
    }

    // The exact hashes above are the authority.  These short grammar anchors
    // protect the translation against accidentally carrying it to an unlike
    // source layout when the code is refactored locally.
    for (asset, needle) in [
        (
            &model_py,
            b"q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))".as_slice(),
        ),
        (
            &model_py,
            b"act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)".as_slice(),
        ),
        (
            &model_py,
            b"o = torch.einsum(\"bsgd,grd->bsgr\", o, wo_a)".as_slice(),
        ),
        (
            &kernel_py,
            b"def sparse_attn_kernel(h: int, d: int, scale=None):".as_slice(),
        ),
        (
            &convert_py,
            b"if name.endswith(\"wo_a.weight\"):".as_slice(),
        ),
    ] {
        if !asset.windows(needle.len()).any(|window| window == needle) {
            return Err(gravity("pinned source grammar anchor is absent"));
        }
    }

    Ok(DeepSeekV4Layer0AttentionSourceAnchors {
        prefix,
        inference_convert_py_sha256,
    })
}

/// Execute the complete bounded attention ladder for source-tokenizer BOS id
/// zero at layer 0 / position 0.  The caller must emit a receipt boundary that
/// keeps this separate from upstream-runtime parity and runtime/TPS evidence.
pub fn layer0_attention_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<Layer0AttentionCpuOracleResult> {
    verify_layer0_attention_source_anchors(reader)?;
    let prefix = layer0_prefix_cpu_oracle(reader)?;

    let wq_a = layer0_wq_a_cpu_oracle(reader, &prefix.attn_norm_bf16_bits)?;
    if wq_a.quantized_input != prefix.wq_a_input_act_quant {
        return Err(gravity(
            "layer-0 WQ-A quantization diverged from the prefix source-linear handoff",
        ));
    }
    let q_norm_weight = read_bf16_tensor(reader, LAYER0_Q_NORM_WEIGHT, Q_LORA_RANK)?;
    let q_norm_bf16_bits = rms_norm_bf16_source_algorithm(
        &wq_a.output.bf16_bits,
        &q_norm_weight,
        Q_LORA_RANK,
        RMS_NORM_EPS,
    )?;
    let wq_b = fp8_linear_bf16_cached(
        reader,
        LAYER0_WQ_B_WEIGHT,
        LAYER0_WQ_B_SCALE,
        WQ_B_ROWS,
        Q_LORA_RANK,
        &q_norm_bf16_bits,
    )?;
    let q_head_norm_bf16_bits = per_head_rms_norm_bf16_source_algorithm(
        &wq_b.output.bf16_bits,
        NUM_HEADS,
        HEAD_DIM,
        RMS_NORM_EPS,
    )?;
    let q_position0_rope_bf16_bits =
        position_zero_rope_identity(&q_head_norm_bf16_bits, NUM_HEADS, HEAD_DIM, ROPE_HEAD_DIM)?;

    let wkv = fp8_linear_bf16_cached(
        reader,
        LAYER0_WKV_WEIGHT,
        LAYER0_WKV_SCALE,
        WKV_ROWS,
        HIDDEN_SIZE,
        &prefix.attn_norm_bf16_bits,
    )?;
    let kv_norm_weight = read_bf16_tensor(reader, LAYER0_KV_NORM_WEIGHT, HEAD_DIM)?;
    let kv_norm_bf16_bits = rms_norm_bf16_source_algorithm(
        &wkv.output.bf16_bits,
        &kv_norm_weight,
        HEAD_DIM,
        RMS_NORM_EPS,
    )?;
    let kv_inplace_qat = kv_non_rope_inplace_qat_source_algorithm(
        &kv_norm_bf16_bits,
        HEAD_DIM,
        ROPE_HEAD_DIM,
        KV_QAT_BLOCK,
    )?;
    let kv_position0_rope_bf16_bits =
        position_zero_rope_identity(&kv_inplace_qat.output_bf16_bits, 1, HEAD_DIM, ROPE_HEAD_DIM)?;

    let attn_sink = read_f32_tensor(reader, LAYER0_ATTN_SINK, NUM_HEADS)?;
    let (
        sparse_attention_scores_f32,
        sparse_attention_sink_denominators_f32,
        sparse_attention_bf16_bits,
    ) = sparse_attention_position_zero_source_algorithm(
        &q_position0_rope_bf16_bits,
        &kv_position0_rope_bf16_bits,
        &attn_sink,
        NUM_HEADS,
        HEAD_DIM,
    )?;
    let sparse_attention_derotated_bf16_bits = position_zero_rope_identity(
        &sparse_attention_bf16_bits,
        NUM_HEADS,
        HEAD_DIM,
        ROPE_HEAD_DIM,
    )?;

    let wo_a_bf16_bits =
        wo_a_bf16_einsum_source_algorithm(reader, &sparse_attention_derotated_bf16_bits)?;
    let wo_b = fp8_linear_bf16_cached(
        reader,
        LAYER0_WO_B_WEIGHT,
        LAYER0_WO_B_SCALE,
        WO_B_ROWS,
        WO_B_COLS,
        &wo_a_bf16_bits,
    )?;
    let hc_attn_post_bf16_bits = hc_attn_post_source_algorithm(
        &wo_b.output.bf16_bits,
        &prefix.hc_replicated_bf16_bits,
        &prefix.hc_post_f32,
        &prefix.hc_comb_f32,
    )?;

    Ok(Layer0AttentionCpuOracleResult {
        prefix,
        wq_a,
        q_norm_bf16_bits,
        wq_b,
        q_head_norm_bf16_bits,
        q_position0_rope_bf16_bits,
        wkv,
        kv_norm_bf16_bits,
        kv_inplace_qat,
        kv_position0_rope_bf16_bits,
        sparse_attention_scores_f32,
        sparse_attention_sink_denominators_f32,
        sparse_attention_bf16_bits,
        sparse_attention_derotated_bf16_bits,
        wo_a_bf16_bits,
        wo_b,
        hc_attn_post_bf16_bits,
    })
}

/// Source `RMSNorm.forward` for an arbitrary bounded width: BF16 activation
/// and checkpoint weight, FP32 variance/product, then a BF16 output cast.
pub fn rms_norm_bf16_source_algorithm(
    input_bf16_bits: &[u16],
    weight_bf16_bits: &[u16],
    width: usize,
    eps: f32,
) -> Result<Vec<u16>> {
    if width == 0
        || input_bf16_bits.len() != width
        || weight_bf16_bits.len() != width
        || !(eps.is_finite() && eps > 0.0)
    {
        return Err(gravity("RMSNorm source geometry/constants are invalid"));
    }
    let mut sum_square = 0.0_f32;
    for &bits in input_bf16_bits {
        let value = bf16::from_bits(bits).to_f32();
        if !value.is_finite() {
            return Err(gravity("RMSNorm input contains a non-finite BF16 value"));
        }
        sum_square += value * value;
    }
    let reciprocal_rms = 1.0_f32 / (sum_square / width as f32 + eps).sqrt();
    if !reciprocal_rms.is_finite() {
        return Err(gravity("RMSNorm reciprocal RMS is non-finite"));
    }
    let mut output = Vec::with_capacity(width);
    for (&input, &weight) in input_bf16_bits.iter().zip(weight_bf16_bits) {
        let value = bf16::from_bits(input).to_f32();
        // The shipped `RMSNorm` parameter is FP32 after the BF16 checkpoint
        // value is loaded, so decode then multiply in f32 before the output
        // type cast.
        let scale = bf16::from_bits(weight).to_f32();
        if !scale.is_finite() {
            return Err(gravity("RMSNorm weight contains a non-finite BF16 value"));
        }
        let normalized = value * reciprocal_rms * scale;
        if !normalized.is_finite() {
            return Err(gravity("RMSNorm output is non-finite"));
        }
        output.push(bf16::from_f32(normalized).to_bits());
    }
    Ok(output)
}

/// Source `q *= rsqrt(q.square().mean(-1) + eps)` with one scalar reduction
/// per local attention head and BF16 write-back.
pub fn per_head_rms_norm_bf16_source_algorithm(
    input_bf16_bits: &[u16],
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<Vec<u16>> {
    if heads == 0
        || head_dim == 0
        || input_bf16_bits.len()
            != heads
                .checked_mul(head_dim)
                .ok_or_else(|| gravity("per-head RMSNorm geometry overflow"))?
        || !(eps.is_finite() && eps > 0.0)
    {
        return Err(gravity(
            "per-head RMSNorm source geometry/constants are invalid",
        ));
    }
    let mut output = Vec::with_capacity(input_bf16_bits.len());
    for head in input_bf16_bits.chunks_exact(head_dim) {
        let mut sum_square = 0.0_f32;
        for &bits in head {
            let value = bf16::from_bits(bits).to_f32();
            if !value.is_finite() {
                return Err(gravity(
                    "per-head RMSNorm input contains a non-finite BF16 value",
                ));
            }
            sum_square += value * value;
        }
        let reciprocal_rms = 1.0_f32 / (sum_square / head_dim as f32 + eps).sqrt();
        if !reciprocal_rms.is_finite() {
            return Err(gravity("per-head RMSNorm reciprocal RMS is non-finite"));
        }
        for &bits in head {
            let value = bf16::from_bits(bits).to_f32() * reciprocal_rms;
            if !value.is_finite() {
                return Err(gravity("per-head RMSNorm output is non-finite"));
            }
            output.push(bf16::from_f32(value).to_bits());
        }
    }
    Ok(output)
}

/// Position zero gives a zero rotary phase for every source frequency, so
/// `apply_rotary_emb` is an identity after its BF16 copy-back.  Keeping this
/// as an executable guard avoids silently using the shortcut at another
/// position or with an invalid RoPE partition.
pub fn position_zero_rope_identity(
    input_bf16_bits: &[u16],
    rows: usize,
    head_dim: usize,
    rope_head_dim: usize,
) -> Result<Vec<u16>> {
    if rows == 0
        || head_dim == 0
        || rope_head_dim == 0
        || rope_head_dim > head_dim
        || rope_head_dim % 2 != 0
        || input_bf16_bits.len()
            != rows
                .checked_mul(head_dim)
                .ok_or_else(|| gravity("position-zero RoPE geometry overflow"))?
        || input_bf16_bits
            .iter()
            .any(|bits| !bf16::from_bits(*bits).to_f32().is_finite())
    {
        return Err(gravity(
            "position-zero RoPE source geometry/value is invalid",
        ));
    }
    // `precompute_freqs_cis(...)[0] = polar(1, 0)`, and source
    // `apply_rotary_emb` copies the result back into the BF16 input tensor.
    Ok(input_bf16_bits.to_vec())
}

/// Source `act_quant(..., block_size=64, scale_fmt="ue8m0",
/// scale_dtype=float8_e8m0fnu, inplace=True)` used on `kv[..., :-rd]`.
pub fn kv_non_rope_inplace_qat_source_algorithm(
    kv_bf16_bits: &[u16],
    head_dim: usize,
    rope_head_dim: usize,
    block_size: usize,
) -> Result<KvInplaceQatCpuResult> {
    if head_dim == 0
        || rope_head_dim == 0
        || rope_head_dim >= head_dim
        || block_size == 0
        || kv_bf16_bits.len() != head_dim
        || (head_dim - rope_head_dim) % block_size != 0
    {
        return Err(gravity("KV in-place QAT source geometry is invalid"));
    }
    let non_rope = head_dim - rope_head_dim;
    let mut output_bf16_bits = kv_bf16_bits.to_vec();
    let mut activation = Vec::with_capacity(non_rope);
    let mut scales = Vec::with_capacity(non_rope / block_size);
    for (block_index, block) in kv_bf16_bits[..non_rope]
        .chunks_exact(block_size)
        .enumerate()
    {
        let mut amax = 0.0_f32;
        for &bits in block {
            let value = bf16::from_bits(bits).to_f32();
            if !value.is_finite() {
                return Err(gravity(
                    "KV in-place QAT input contains a non-finite BF16 value",
                ));
            }
            amax = amax.max(value.abs());
        }
        let scale_bits = rounded_ue8m0_scale_byte(amax)?;
        let scale = decode_e8m0fnu(scale_bits)?;
        scales.push(scale_bits);
        for (offset, &bits) in block.iter().enumerate() {
            let input = bf16::from_bits(bits).to_f32();
            let encoded = encode_e4m3fn_rne((input / scale).clamp(-448.0, 448.0))?;
            activation.push(encoded);
            // Matches kernel.py's FP8 cast -> f32 scale multiply -> BF16
            // `out_dtype` store for the `inplace=True` branch.
            let dequantized = decode_e4m3fn(encoded)? * scale;
            let target = block_index * block_size + offset;
            output_bf16_bits[target] = bf16::from_f32(dequantized).to_bits();
        }
    }
    if output_bf16_bits[non_rope..] != kv_bf16_bits[non_rope..] {
        return Err(gravity("KV in-place QAT mutated the protected RoPE suffix"));
    }
    Ok(KvInplaceQatCpuResult {
        output_bf16_bits,
        non_rope_activation_e4m3fn: activation,
        non_rope_scales_e8m0fnu: scales,
    })
}

/// The exact layer-0 sparse-attention specialization selected by the source:
/// one causal window index (`0`) and no compressed/index branch.  It retains
/// source kernel behavior that the attention sink competes with the sole KV
/// score rather than being added to that score.
pub fn sparse_attention_position_zero_source_algorithm(
    q_bf16_bits: &[u16],
    kv_bf16_bits: &[u16],
    attn_sink: &[f32],
    heads: usize,
    head_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<u16>)> {
    if heads == 0
        || head_dim == 0
        || q_bf16_bits.len()
            != heads
                .checked_mul(head_dim)
                .ok_or_else(|| gravity("sparse attention Q geometry overflow"))?
        || kv_bf16_bits.len() != head_dim
        || attn_sink.len() != heads
        || attn_sink.iter().any(|value| !value.is_finite())
    {
        return Err(gravity(
            "position-zero sparse attention source geometry is invalid",
        ));
    }
    let kv: Vec<f32> = kv_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if kv.iter().any(|value| !value.is_finite()) {
        return Err(gravity("position-zero sparse attention KV is non-finite"));
    }
    let softmax_scale = (head_dim as f32).powf(-0.5);
    let mut scores = Vec::with_capacity(heads);
    let mut denominators = Vec::with_capacity(heads);
    let mut output = Vec::with_capacity(q_bf16_bits.len());
    for (head_index, q_head) in q_bf16_bits.chunks_exact(head_dim).enumerate() {
        let mut dot = 0.0_f32;
        for (&q_bits, &kv_value) in q_head.iter().zip(&kv) {
            let q_value = bf16::from_bits(q_bits).to_f32();
            if !q_value.is_finite() {
                return Err(gravity("position-zero sparse attention Q is non-finite"));
            }
            dot += q_value * kv_value;
        }
        let score = dot * softmax_scale;
        let sink_exp = (attn_sink[head_index] - score).exp();
        let denominator = 1.0_f32 + sink_exp;
        if !score.is_finite() || !denominator.is_finite() || denominator <= 0.0 {
            return Err(gravity(
                "position-zero sparse attention score/sink denominator is not finite",
            ));
        }
        scores.push(score);
        denominators.push(denominator);
        // With exactly one selected KV index, kernel.py's online softmax has
        // exp(score-score)=1, then adds exp(attn_sink-score) to `sum_exp`.
        // `acc_s` is cast to BF16 before the value GEMM, but exactly one is
        // representable, so the numerator is the source BF16 KV row.
        for &kv_value in &kv {
            output.push(bf16::from_f32(kv_value / denominator).to_bits());
        }
    }
    Ok((scores, denominators, output))
}

/// Source `convert.py` WO-A FP8/E8M0 materialization followed by
/// `model.py`'s grouped BF16 `torch.einsum("bsgd,grd->bsgr", o, wo_a)`.
/// The source does not use `model.py::linear` for this operator.
pub fn wo_a_bf16_einsum_source_algorithm(
    reader: &DeepSeekV4FullStreamReader,
    attention_bf16_bits: &[u16],
) -> Result<Vec<u16>> {
    if attention_bf16_bits.len() != NUM_HEADS * HEAD_DIM {
        return Err(gravity("WO-A input is not [64, 512] BF16"));
    }
    let scale_name = expect_native_fp8_pair(
        reader,
        LAYER0_WO_A_WEIGHT,
        LAYER0_WO_A_SCALE,
        WO_A_ROWS,
        WO_A_COLS,
    )?;
    let weights = reader.read_verified_full(LAYER0_WO_A_WEIGHT, WO_A_ROWS * WO_A_COLS)?;
    let scales = reader.read_verified_full(
        &scale_name,
        (WO_A_ROWS / ACT_QUANT_BLOCK) * (WO_A_COLS / ACT_QUANT_BLOCK),
    )?;
    let (fp8_values, fp8_finite) = e4m3fn_lookup()?;
    let (e8m0_values, e8m0_finite) = e8m0_lookup()?;
    validate_fp8_bytes(&weights, &fp8_finite, "WO-A source FP8 weights")?;
    validate_fp8_bytes(&scales, &e8m0_finite, "WO-A source E8M0 scales")?;
    let input: Vec<f32> = attention_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().any(|value| !value.is_finite()) {
        return Err(gravity(
            "WO-A attention input contains a non-finite BF16 value",
        ));
    }
    let input_max_abs = input
        .iter()
        .fold(0.0_f32, |maximum, value| maximum.max(value.abs()));

    let mut output = Vec::with_capacity(WO_A_ROWS);
    let mut converted_weight_max_abs = 0.0_f32;
    let scale_cols = WO_A_COLS / ACT_QUANT_BLOCK;
    for group in 0..O_GROUPS {
        let input_group = &input[group * WO_A_COLS..(group + 1) * WO_A_COLS];
        for rank in 0..O_LORA_RANK {
            let row = group * O_LORA_RANK + rank;
            let mut accumulator = 0.0_f32;
            for column in 0..WO_A_COLS {
                let raw = weights[row * WO_A_COLS + column];
                let scale_index = (row / ACT_QUANT_BLOCK) * scale_cols + column / ACT_QUANT_BLOCK;
                let raw_scale = scales[scale_index];
                let scale = e8m0_values[raw_scale as usize];
                // `convert.py`: raw FP8 -> FP32 * raw E8M0 scale -> BF16
                // materialized parameter.  The einsum sees this BF16 value.
                let converted_weight = bf16::from_f32(fp8_values[raw as usize] * scale).to_f32();
                if !converted_weight.is_finite() {
                    return Err(gravity(format!(
                        "WO-A conversion produced non-finite BF16 at group={group} rank={rank} row={row} column={column} raw_weight={raw} raw_scale={raw_scale} scale_index={scale_index} decoded_weight={} decoded_scale={scale}",
                        fp8_values[raw as usize],
                    )));
                }
                converted_weight_max_abs = converted_weight_max_abs.max(converted_weight.abs());
                accumulator += input_group[column] * converted_weight;
            }
            if !accumulator.is_finite() {
                return Err(gravity(format!(
                    "WO-A grouped BF16 einsum produced a non-finite value at group={group} rank={rank} row={row} accumulator={accumulator} input_max_abs={input_max_abs} converted_weight_max_abs={converted_weight_max_abs}",
                )));
            }
            output.push(bf16::from_f32(accumulator).to_bits());
        }
    }
    Ok(output)
}

/// Source `Block.hc_post` after the attention residual branch.  In
/// `comb.unsqueeze(-1) * residual.unsqueeze(-2)`, source comb row `j` weights
/// residual HC lane `j` and comb column `k` yields output HC lane `k`.
pub fn hc_attn_post_source_algorithm(
    attention_bf16_bits: &[u16],
    residual_hc_bf16_bits: &[u16],
    post_f32: &[f32],
    comb_f32: &[f32],
) -> Result<Vec<u16>> {
    if attention_bf16_bits.len() != HIDDEN_SIZE
        || residual_hc_bf16_bits.len() != HC_MULT * HIDDEN_SIZE
        || post_f32.len() != HC_MULT
        || comb_f32.len() != HC_MULT * HC_MULT
        || post_f32
            .iter()
            .chain(comb_f32)
            .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "mHC attention post source geometry/value is invalid",
        ));
    }
    let attention: Vec<f32> = attention_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    let residual: Vec<f32> = residual_hc_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if attention
        .iter()
        .chain(&residual)
        .any(|value| !value.is_finite())
    {
        return Err(gravity("mHC attention post BF16 input is non-finite"));
    }
    let mut output = Vec::with_capacity(HC_MULT * HIDDEN_SIZE);
    for output_lane in 0..HC_MULT {
        for feature in 0..HIDDEN_SIZE {
            let mut value = post_f32[output_lane] * attention[feature];
            for source_lane in 0..HC_MULT {
                value += comb_f32[source_lane * HC_MULT + output_lane]
                    * residual[source_lane * HIDDEN_SIZE + feature];
            }
            if !value.is_finite() {
                return Err(gravity("mHC attention post output is non-finite"));
            }
            // `y.type_as(x)` where attention x is BF16.
            output.push(bf16::from_f32(value).to_bits());
        }
    }
    Ok(output)
}

fn fp8_linear_bf16_cached(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    expected_scale_name: &str,
    output_rows: usize,
    logical_k: usize,
    input_bf16_bits: &[u16],
) -> Result<Fp8LinearCpuStage> {
    let scale_name = expect_native_fp8_pair(
        reader,
        weight_name,
        expected_scale_name,
        output_rows,
        logical_k,
    )?;
    let quantized_input = act_quant_bf16_ue8m0(input_bf16_bits)?;
    let weights = reader.read_verified_full(weight_name, output_rows * logical_k)?;
    let scales = reader.read_verified_full(
        &scale_name,
        (output_rows / ACT_QUANT_BLOCK) * (logical_k / ACT_QUANT_BLOCK),
    )?;
    let output = fp8_e4m3fn_ue8m0_matvec_cached(
        &quantized_input,
        &weights,
        &scales,
        output_rows,
        logical_k,
    )?;
    Ok(Fp8LinearCpuStage {
        quantized_input,
        output,
    })
}

/// Identical scalar block accumulator grammar to the established act-quant
/// oracle's FP8 matvec, with source-native finite bytes predecoded once.  The
/// lookup avoids adding a decoder call to every one of the 100M bounded CPU
/// products while preserving the same f32 product/add/block-scale order.
fn fp8_e4m3fn_ue8m0_matvec_cached(
    activation: &ActQuantizedBf16Row,
    weights_e4m3fn: &[u8],
    weight_scales_e8m0fnu: &[u8],
    output_rows: usize,
    logical_k: usize,
) -> Result<Fp8MatvecCpuResult> {
    if output_rows == 0
        || logical_k == 0
        || output_rows % ACT_QUANT_BLOCK != 0
        || logical_k % ACT_QUANT_BLOCK != 0
        || activation.activation_e4m3fn.len() != logical_k
        || activation.scales_e8m0fnu.len() != logical_k / ACT_QUANT_BLOCK
        || activation.decoded_scales_f32.len() != logical_k / ACT_QUANT_BLOCK
        || weights_e4m3fn.len() != output_rows * logical_k
        || weight_scales_e8m0fnu.len()
            != (output_rows / ACT_QUANT_BLOCK) * (logical_k / ACT_QUANT_BLOCK)
    {
        return Err(gravity("cached FP8 source matvec geometry is invalid"));
    }
    let (e4_values, e4_finite) = e4m3fn_lookup()?;
    let (e8_values, e8_finite) = e8m0_lookup()?;
    validate_fp8_bytes(
        &activation.activation_e4m3fn,
        &e4_finite,
        "FP8 source matvec activation",
    )?;
    validate_fp8_bytes(weights_e4m3fn, &e4_finite, "FP8 source matvec weights")?;
    validate_fp8_bytes(
        weight_scales_e8m0fnu,
        &e8_finite,
        "FP8 source matvec scales",
    )?;
    if activation
        .decoded_scales_f32
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(gravity("FP8 source matvec activation scale is invalid"));
    }

    let activations: Vec<f32> = activation
        .activation_e4m3fn
        .iter()
        .map(|raw| e4_values[*raw as usize])
        .collect();
    let scale_cols = logical_k / ACT_QUANT_BLOCK;
    let mut output = Vec::with_capacity(output_rows);
    for row in 0..output_rows {
        let row_base = row * logical_k;
        let scale_row = row / ACT_QUANT_BLOCK;
        let mut row_accumulator = 0.0_f32;
        for block in 0..scale_cols {
            let combined_scale = activation.decoded_scales_f32[block]
                * e8_values[weight_scales_e8m0fnu[scale_row * scale_cols + block] as usize];
            let start = block * ACT_QUANT_BLOCK;
            let mut block_accumulator = 0.0_f32;
            for column in start..start + ACT_QUANT_BLOCK {
                block_accumulator +=
                    activations[column] * e4_values[weights_e4m3fn[row_base + column] as usize];
            }
            row_accumulator += block_accumulator * combined_scale;
        }
        if !row_accumulator.is_finite() {
            return Err(gravity(
                "cached FP8 source matvec produced a non-finite value",
            ));
        }
        output.push(row_accumulator);
    }
    let bf16_bits = output
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect();
    Ok(Fp8MatvecCpuResult {
        fp32: output,
        bf16_bits,
    })
}

fn expect_native_fp8_pair(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    expected_scale_name: &str,
    output_rows: usize,
    logical_k: usize,
) -> Result<String> {
    let pair = reader.native_scale_pair(weight_name)?;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.weight.name != weight_name
        || pair.scale.name != expected_scale_name
        || pair.weight.shape.as_slice() != [output_rows as u64, logical_k as u64]
        || pair.scale.shape.as_slice()
            != [
                (output_rows / ACT_QUANT_BLOCK) as u64,
                (logical_k / ACT_QUANT_BLOCK) as u64,
            ]
        || pair.out_rows != output_rows as u64
        || pair.logical_k != logical_k as u64
        || pair.packed_k != logical_k as u64
        || pair.scale_rows != (output_rows / ACT_QUANT_BLOCK) as u64
        || pair.scale_cols != (logical_k / ACT_QUANT_BLOCK) as u64
    {
        return Err(gravity(format!(
            "{weight_name} does not match the pinned native FP8/E8M0 source contract",
        )));
    }
    Ok(pair.scale.name.clone())
}

fn read_bf16_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    width: usize,
) -> Result<Vec<u16>> {
    let tensor = reader.tensor_metadata(name)?;
    if tensor.dtype != "BF16"
        || tensor.shape.as_slice() != [width as u64]
        || tensor.bytes != (width * 2) as u64
    {
        return Err(gravity(format!(
            "{name} is not the pinned BF16 [{width}] tensor"
        )));
    }
    decode_u16_le(&reader.read_verified_full(name, width * 2)?, name)
}

fn read_f32_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    width: usize,
) -> Result<Vec<f32>> {
    let tensor = reader.tensor_metadata(name)?;
    if tensor.dtype != "F32"
        || tensor.shape.as_slice() != [width as u64]
        || tensor.bytes != (width * 4) as u64
    {
        return Err(gravity(format!(
            "{name} is not the pinned F32 [{width}] tensor"
        )));
    }
    decode_f32_le(&reader.read_verified_full(name, width * 4)?, name)
}

fn e4m3fn_lookup() -> Result<([f32; 256], [bool; 256])> {
    let mut values = [0.0_f32; 256];
    let mut finite = [false; 256];
    for raw in 0u16..=u8::MAX as u16 {
        if let Ok(value) = decode_e4m3fn(raw as u8) {
            values[raw as usize] = value;
            finite[raw as usize] = true;
        }
    }
    if finite.iter().filter(|present| **present).count() != 254 {
        return Err(gravity("E4M3FN finite decode table is malformed"));
    }
    Ok((values, finite))
}

fn e8m0_lookup() -> Result<([f32; 256], [bool; 256])> {
    let mut values = [0.0_f32; 256];
    let mut finite = [false; 256];
    for raw in 0u16..=u8::MAX as u16 {
        if let Ok(value) = decode_e8m0fnu(raw as u8) {
            values[raw as usize] = value;
            finite[raw as usize] = true;
        }
    }
    if finite.iter().filter(|present| **present).count() != 255 {
        return Err(gravity("E8M0FNU finite decode table is malformed"));
    }
    Ok((values, finite))
}

fn validate_fp8_bytes(bytes: &[u8], finite: &[bool; 256], label: &str) -> Result<()> {
    if bytes.iter().any(|raw| !finite[*raw as usize]) {
        return Err(gravity(format!(
            "{label} contains the source dtype NaN encoding"
        )));
    }
    Ok(())
}

fn decode_u16_le(bytes: &[u8], name: &str) -> Result<Vec<u16>> {
    if bytes.len() % 2 != 0 {
        return Err(gravity(format!("{name} BF16 bytes are not u16 aligned")));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_f32_le(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(gravity(format!("{name} F32 bytes are not f32 aligned")));
    }
    let output: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect();
    if output.iter().any(|value| !value.is_finite()) {
        return Err(gravity(format!(
            "{name} contains a non-finite F32 source value"
        )));
    }
    Ok(output)
}

fn parse_json(bytes: &[u8], name: &str) -> Result<Value> {
    serde_json::from_slice(bytes)
        .map_err(|error| gravity(format!("pinned {name} JSON parsing failed: {error}")))
}

fn json_path<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a Value> {
    let mut current = value;
    for key in path {
        current = current
            .get(*key)
            .ok_or_else(|| gravity(format!("pinned source lacks {label}")))?;
    }
    Ok(current)
}

fn json_u64(value: &Value, path: &[&str], label: &str) -> Result<u64> {
    json_path(value, path, label)?
        .as_u64()
        .ok_or_else(|| gravity(format!("{label} is not an unsigned JSON integer")))
}

fn json_string<'a>(value: &'a Value, path: &[&str], label: &str) -> Result<&'a str> {
    json_path(value, path, label)?
        .as_str()
        .ok_or_else(|| gravity(format!("{label} is not a JSON string")))
}

fn json_array_u64(value: &Value, path: &[&str], label: &str) -> Result<Vec<u64>> {
    json_path(value, path, label)?
        .as_array()
        .ok_or_else(|| gravity(format!("{label} is not a JSON array")))?
        .iter()
        .map(|entry| {
            entry
                .as_u64()
                .ok_or_else(|| gravity(format!("{label} has a non-u64 item")))
        })
        .collect()
}

fn json_f64_eq(value: &Value, path: &[&str], expected: f32) -> bool {
    json_path(value, path, "numeric config value")
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
    fn cached_fp8_matvec_matches_established_scalar_oracle() {
        let input = vec![bf16::from_f32(1.0).to_bits(); ACT_QUANT_BLOCK];
        let activation = act_quant_bf16_ue8m0(&input).unwrap();
        let mut weights = vec![0x38_u8; ACT_QUANT_BLOCK * ACT_QUANT_BLOCK];
        weights[17] = 0x40;
        let scales = vec![0x7f_u8];
        let established = crate::gravity_deepseek_v4_act_quant::fp8_e4m3fn_ue8m0_matvec(
            &activation,
            &weights,
            &scales,
            ACT_QUANT_BLOCK,
            ACT_QUANT_BLOCK,
        )
        .unwrap();
        let cached = fp8_e4m3fn_ue8m0_matvec_cached(
            &activation,
            &weights,
            &scales,
            ACT_QUANT_BLOCK,
            ACT_QUANT_BLOCK,
        )
        .unwrap();
        assert_eq!(cached, established);
    }

    #[test]
    fn kv_qat_preserves_rope_suffix_and_has_64wide_scales() {
        let input: Vec<u16> = (0..HEAD_DIM)
            .map(|index| bf16::from_f32((index as f32 - 200.0) / 64.0).to_bits())
            .collect();
        let result =
            kv_non_rope_inplace_qat_source_algorithm(&input, HEAD_DIM, ROPE_HEAD_DIM, KV_QAT_BLOCK)
                .unwrap();
        assert_eq!(result.output_bf16_bits.len(), HEAD_DIM);
        assert_eq!(result.non_rope_activation_e4m3fn.len(), NON_ROPE_HEAD_DIM);
        assert_eq!(
            result.non_rope_scales_e8m0fnu.len(),
            NON_ROPE_HEAD_DIM / KV_QAT_BLOCK
        );
        assert_eq!(
            &result.output_bf16_bits[NON_ROPE_HEAD_DIM..],
            &input[NON_ROPE_HEAD_DIM..]
        );
    }

    #[test]
    fn position_zero_sparse_attention_keeps_one_kv_and_sink_competition() {
        let q = vec![bf16::from_f32(1.0).to_bits(); 4];
        let kv = vec![bf16::from_f32(1.0).to_bits(); 2];
        let (scores, denominators, output) =
            sparse_attention_position_zero_source_algorithm(&q, &kv, &[0.0, 0.0], 2, 2).unwrap();
        let expected_score = 2.0_f32 * 2.0_f32.powf(-0.5);
        assert_eq!(scores, vec![expected_score, expected_score]);
        let expected_denominator = 1.0 + (-expected_score).exp();
        assert_eq!(
            denominators,
            vec![expected_denominator, expected_denominator]
        );
        let expected = bf16::from_f32(1.0 / expected_denominator).to_bits();
        assert_eq!(output, vec![expected; 4]);
    }

    #[test]
    fn hc_post_uses_source_comb_columns_as_output_lanes() {
        let attention = vec![bf16::from_f32(1.0).to_bits(); HIDDEN_SIZE];
        let mut residual = Vec::with_capacity(HC_MULT * HIDDEN_SIZE);
        for lane in 0..HC_MULT {
            residual.extend(std::iter::repeat_n(
                bf16::from_f32(lane as f32).to_bits(),
                HIDDEN_SIZE,
            ));
        }
        let post = vec![1.0_f32; HC_MULT];
        let mut comb = vec![0.0_f32; HC_MULT * HC_MULT];
        for index in 0..HC_MULT {
            comb[index * HC_MULT + index] = 1.0;
        }
        let output = hc_attn_post_source_algorithm(&attention, &residual, &post, &comb).unwrap();
        for lane in 0..HC_MULT {
            assert_eq!(
                output[lane * HIDDEN_SIZE],
                bf16::from_f32(1.0 + lane as f32).to_bits()
            );
        }
    }
}
