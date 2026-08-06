//! CPU-only source-algorithm checkpoint for the first DeepSeek-V4 token path.
//!
//! This module deliberately implements only the bounded prefix below the
//! first layer's `wq_a` projection:
//!
//! ```text
//! pinned tokenizer BOS id 0
//!   -> one real BF16 embed row
//!   -> four BF16 Hyper-Connection copies
//!   -> layer-0 hc_attn_pre / Sinkhorn (20 iterations)
//!   -> layer-0 attention RMSNorm
//!   -> BF16 [4096] input accepted by the source `wq_a` FP8 Linear grammar
//! ```
//!
//! The implementation is a scalar CPU transcription of the pinned source
//! formula and ordering.  It is not an independently executed upstream
//! runtime, a full forward, GPU work, an endpoint, or TPS evidence.  In
//! particular, PyTorch/TileLang reduction and transcendental implementation
//! details are intentionally not claimed bit-identical here.

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, DeepSeekV4TensorMetadata, NativeScalePairKind,
};
use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, verify_source_algorithm_anchors, ActQuantizedBf16Row,
    DeepSeekV4ActQuantSourceAnchors, ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS,
    LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
};
use crate::{Error, Result};
use half::bf16;
use serde_json::Value;

/// Fixed source-tokenizer BOS id used by this bounded no-prompt checkpoint.
pub const PREFIX_TOKEN_ID: u64 = 0;
/// Exact BPE vocabulary string bound to [`PREFIX_TOKEN_ID`] in the pinned
/// `tokenizer.json`, and also the `tokenizer_config.json` BOS content.
pub const PREFIX_TOKEN_STRING: &str = "<｜begin▁of▁sentence｜>";
pub const HIDDEN_SIZE: usize = 4096;
pub const HC_MULT: usize = 4;
pub const HC_MIX_WIDTH: usize = (2 + HC_MULT) * HC_MULT;
pub const HC_FLAT_WIDTH: usize = HC_MULT * HIDDEN_SIZE;
pub const HC_SINKHORN_ITERS: usize = 20;
pub const HC_EPS: f32 = 1.0e-6;
pub const RMS_NORM_EPS: f32 = 1.0e-6;
pub const VOCAB_SIZE: u64 = 129_280;

pub const EMBED_WEIGHT: &str = "embed.weight";
pub const LAYER0_HC_ATTN_FN: &str = "layers.0.hc_attn_fn";
pub const LAYER0_HC_ATTN_BASE: &str = "layers.0.hc_attn_base";
pub const LAYER0_HC_ATTN_SCALE: &str = "layers.0.hc_attn_scale";
pub const LAYER0_ATTN_NORM_WEIGHT: &str = "layers.0.attn_norm.weight";

/// Exact hashes of the two tokenizer assets admitted by the full-stream
/// reader.  They are checked in addition to the source model/kernel/config
/// anchors from the preceding FP8 act-quant checkpoint.
pub const OFFICIAL_TOKENIZER_JSON_SHA256: &str =
    "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf";
pub const OFFICIAL_TOKENIZER_CONFIG_JSON_SHA256: &str =
    "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547";

/// The complete static-source binding required by the prefix checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4Layer0PrefixSourceAnchors {
    pub act_quant: DeepSeekV4ActQuantSourceAnchors,
    pub tokenizer_json_sha256: String,
    pub tokenizer_config_json_sha256: String,
}

/// CPU-only state made observable so the receipt producer can hash bounded
/// intermediate values without exposing the fixed token's raw activations.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0PrefixCpuOracleResult {
    pub token_id: u64,
    pub embed_bf16_bits: Vec<u16>,
    pub hc_replicated_bf16_bits: Vec<u16>,
    pub hc_flat_rsqrt: f32,
    pub hc_mixes_f32: Vec<f32>,
    pub hc_pre_f32: Vec<f32>,
    pub hc_post_f32: Vec<f32>,
    pub hc_comb_f32: Vec<f32>,
    pub hc_attn_pre_bf16_bits: Vec<u16>,
    pub attn_norm_bf16_bits: Vec<u16>,
    pub wq_a_input_act_quant: ActQuantizedBf16Row,
}

/// Verify all static assets and source configuration constants needed by the
/// source-algorithm transcription.  This makes the fixed id a real tokenizer
/// binding rather than an unqualified magic number.
pub fn verify_layer0_prefix_source_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4Layer0PrefixSourceAnchors> {
    let act_quant = verify_source_algorithm_anchors(reader)?;
    let tokenizer_json_sha256 = reader
        .source_metadata_asset_sha256("tokenizer.json")?
        .to_owned();
    let tokenizer_config_json_sha256 = reader
        .source_metadata_asset_sha256("tokenizer_config.json")?
        .to_owned();
    if tokenizer_json_sha256 != OFFICIAL_TOKENIZER_JSON_SHA256
        || tokenizer_config_json_sha256 != OFFICIAL_TOKENIZER_CONFIG_JSON_SHA256
    {
        return Err(gravity(
            "layer-0 prefix tokenizer assets differ from pinned official anchors",
        ));
    }

    // Re-read these bounded assets through the admitted reader.  The reader
    // re-checks their file identities/hashes, then the exact source values
    // below bind the token id, geometry, and numerical constants.
    let tokenizer = parse_json(
        &reader.read_verified_metadata_asset("tokenizer.json", 8 * 1024 * 1024)?,
        "tokenizer.json",
    )?;
    let tokenizer_config = parse_json(
        &reader.read_verified_metadata_asset("tokenizer_config.json", 64 * 1024)?,
        "tokenizer_config.json",
    )?;
    let model_config = parse_json(
        &reader.read_verified_metadata_asset("config.json", 64 * 1024)?,
        "config.json",
    )?;
    let inference_config = parse_json(
        &reader.read_verified_metadata_asset("inference/config.json", 64 * 1024)?,
        "inference/config.json",
    )?;

    let vocab_id = tokenizer
        .get("model")
        .and_then(Value::as_object)
        .and_then(|model| model.get("vocab"))
        .and_then(Value::as_object)
        .and_then(|vocab| vocab.get(PREFIX_TOKEN_STRING))
        .and_then(Value::as_u64)
        .ok_or_else(|| gravity("pinned tokenizer.json lacks the exact BOS BPE mapping"))?;
    if vocab_id != PREFIX_TOKEN_ID
        || json_string(
            &tokenizer_config,
            &["bos_token", "content"],
            "tokenizer_config BOS content",
        )? != PREFIX_TOKEN_STRING
        || json_u64(&model_config, &["bos_token_id"], "config BOS id")? != PREFIX_TOKEN_ID
        || json_u64(&model_config, &["vocab_size"], "config vocab size")? != VOCAB_SIZE
        || json_u64(&model_config, &["hidden_size"], "config hidden size")? != HIDDEN_SIZE as u64
        || json_u64(&model_config, &["hc_mult"], "config hc_mult")? != HC_MULT as u64
        || json_u64(
            &model_config,
            &["hc_sinkhorn_iters"],
            "config hc_sinkhorn_iters",
        )? != HC_SINKHORN_ITERS as u64
        || !json_f64_eq(&model_config, &["hc_eps"], HC_EPS)
        || !json_f64_eq(&model_config, &["rms_norm_eps"], RMS_NORM_EPS)
        || json_u64(&inference_config, &["hc_mult"], "inference config hc_mult")? != HC_MULT as u64
        || json_u64(
            &inference_config,
            &["hc_sinkhorn_iters"],
            "inference config hc_sinkhorn_iters",
        )? != HC_SINKHORN_ITERS as u64
    {
        return Err(gravity(
            "pinned source tokenizer/config values differ from the layer-0 prefix contract",
        ));
    }

    Ok(DeepSeekV4Layer0PrefixSourceAnchors {
        act_quant,
        tokenizer_json_sha256,
        tokenizer_config_json_sha256,
    })
}

/// Execute the bounded prefix from the fixed source-tokenizer BOS id through
/// the BF16 input accepted by layer-0 `attn.wq_a`.
pub fn layer0_prefix_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<Layer0PrefixCpuOracleResult> {
    layer0_prefix_cpu_oracle_for_token(reader, PREFIX_TOKEN_ID)
}

/// Execute the same bounded tokenizer-to-WQ-A prefix for one admitted BPE
/// token id.  This is the narrow continuation seam for position-specific
/// attention proofs: it does not retain or imply a causal model state.
pub fn layer0_prefix_cpu_oracle_for_token(
    reader: &DeepSeekV4FullStreamReader,
    token_id: u64,
) -> Result<Layer0PrefixCpuOracleResult> {
    verify_layer0_prefix_source_anchors(reader)?;
    if token_id >= VOCAB_SIZE {
        return Err(gravity(
            "tokenizer-bound prefix token id is outside the vocabulary",
        ));
    }

    let embed = expect_tensor(
        reader,
        EMBED_WEIGHT,
        "BF16",
        &[VOCAB_SIZE, HIDDEN_SIZE as u64],
    )?;
    let hc_fn = expect_tensor(
        reader,
        LAYER0_HC_ATTN_FN,
        "F32",
        &[HC_MIX_WIDTH as u64, HC_FLAT_WIDTH as u64],
    )?;
    let hc_base = expect_tensor(reader, LAYER0_HC_ATTN_BASE, "F32", &[HC_MIX_WIDTH as u64])?;
    let hc_scale = expect_tensor(reader, LAYER0_HC_ATTN_SCALE, "F32", &[3])?;
    let attn_norm = expect_tensor(
        reader,
        LAYER0_ATTN_NORM_WEIGHT,
        "BF16",
        &[HIDDEN_SIZE as u64],
    )?;
    let wq_a = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
    if wq_a.kind != NativeScalePairKind::Fp8E4M3fn
        || wq_a.weight.name != LAYER0_WQ_A_WEIGHT
        || wq_a.scale.name != LAYER0_WQ_A_SCALE
        || wq_a.weight.shape.as_slice() != [LAYER0_WQ_A_ROWS as u64, LAYER0_WQ_A_COLS as u64]
        || wq_a.scale.shape.as_slice()
            != [
                (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) as u64,
                (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u64,
            ]
    {
        return Err(gravity(
            "layer-0 WQ-A no longer has its pinned FP8/E8M0 input contract",
        ));
    }

    let row_bytes = (HIDDEN_SIZE * std::mem::size_of::<u16>()) as u64;
    let row_start = token_id
        .checked_mul(row_bytes)
        .ok_or_else(|| gravity("tokenizer-bound embed row start overflow"))?;
    if row_start
        .checked_add(row_bytes)
        .ok_or_else(|| gravity("tokenizer-bound embed row end overflow"))?
        > embed.bytes
    {
        return Err(gravity(
            "tokenizer-bound embed row escapes the admitted embedding tensor",
        ));
    }
    let embed_raw = reader.read_verified_range(
        EMBED_WEIGHT,
        row_start..row_start + row_bytes,
        row_bytes as usize,
    )?;
    let embed_bf16_bits = decode_u16_le(&embed_raw, "embed.weight row")?;
    if embed_bf16_bits.len() != HIDDEN_SIZE {
        return Err(gravity(
            "tokenizer-bound embed row has the wrong BF16 width",
        ));
    }

    // Matches `h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)`: four separate
    // BF16 lanes in lane-major order, each exactly equal to the real embed row.
    let mut hc_replicated_bf16_bits = Vec::with_capacity(HC_FLAT_WIDTH);
    for _ in 0..HC_MULT {
        hc_replicated_bf16_bits.extend_from_slice(&embed_bf16_bits);
    }

    let hc_fn_f32 = decode_f32_le(
        &reader.read_verified_full(LAYER0_HC_ATTN_FN, hc_fn.bytes as usize)?,
        LAYER0_HC_ATTN_FN,
    )?;
    let hc_base_f32 = decode_f32_le(
        &reader.read_verified_full(LAYER0_HC_ATTN_BASE, hc_base.bytes as usize)?,
        LAYER0_HC_ATTN_BASE,
    )?;
    let hc_scale_f32 = decode_f32_le(
        &reader.read_verified_full(LAYER0_HC_ATTN_SCALE, hc_scale.bytes as usize)?,
        LAYER0_HC_ATTN_SCALE,
    )?;
    let attn_norm_bf16 = decode_u16_le(
        &reader.read_verified_full(LAYER0_ATTN_NORM_WEIGHT, attn_norm.bytes as usize)?,
        LAYER0_ATTN_NORM_WEIGHT,
    )?;

    let (hc_flat_rsqrt, hc_mixes_f32, hc_pre_f32, hc_post_f32, hc_comb_f32, hc_attn_pre_bf16_bits) =
        hc_attn_pre_source_algorithm(
            &hc_replicated_bf16_bits,
            &hc_fn_f32,
            &hc_scale_f32,
            &hc_base_f32,
            RMS_NORM_EPS,
            HC_EPS,
            HC_SINKHORN_ITERS,
        )?;
    let attn_norm_bf16_bits =
        rms_norm_source_algorithm(&hc_attn_pre_bf16_bits, &attn_norm_bf16, RMS_NORM_EPS)?;
    if attn_norm_bf16_bits.len() != LAYER0_WQ_A_COLS {
        return Err(gravity(
            "attention RMSNorm did not produce the BF16 [4096] WQ-A input",
        ));
    }
    // This is the source's immediate `linear` handoff for an FP8 WQ-A weight.
    // The bounded checkpoint intentionally ends before `fp8_gemm` itself.
    let wq_a_input_act_quant = act_quant_bf16_ue8m0(&attn_norm_bf16_bits)?;

    Ok(Layer0PrefixCpuOracleResult {
        token_id,
        embed_bf16_bits,
        hc_replicated_bf16_bits,
        hc_flat_rsqrt,
        hc_mixes_f32,
        hc_pre_f32,
        hc_post_f32,
        hc_comb_f32,
        hc_attn_pre_bf16_bits,
        attn_norm_bf16_bits,
        wq_a_input_act_quant,
    })
}

/// Scalar transcription of `Block.hc_pre` and
/// `kernel.py::hc_split_sinkhorn_kernel`.  The Sinkhorn call is deliberately
/// expressed in the same first-row/first-column then 19 row/column-pass
/// ordering as the source kernel, including its placement of `+ eps`.
pub fn hc_attn_pre_source_algorithm(
    replicated_bf16_bits: &[u16],
    hc_fn: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    norm_eps: f32,
    hc_eps: f32,
    sinkhorn_iters: usize,
) -> Result<(f32, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u16>)> {
    if replicated_bf16_bits.len() != HC_FLAT_WIDTH
        || hc_fn.len() != HC_MIX_WIDTH * HC_FLAT_WIDTH
        || hc_scale.len() != 3
        || hc_base.len() != HC_MIX_WIDTH
        || !(norm_eps.is_finite() && norm_eps > 0.0)
        || !(hc_eps.is_finite() && hc_eps > 0.0)
        || sinkhorn_iters != HC_SINKHORN_ITERS
    {
        return Err(gravity(
            "layer-0 hc_attn_pre source-algorithm geometry/constants differ from the pinned contract",
        ));
    }
    let flat: Vec<f32> = replicated_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if flat.iter().any(|value| !value.is_finite()) {
        return Err(gravity("embed BF16 row contains a non-finite value"));
    }

    // `x.flatten(2).float(); torch.rsqrt(x.square().mean(-1) + norm_eps)`.
    let mut mean_square_sum = 0.0_f32;
    for &value in &flat {
        mean_square_sum += value * value;
    }
    let hc_flat_rsqrt = 1.0_f32 / (mean_square_sum / HC_FLAT_WIDTH as f32 + norm_eps).sqrt();
    if !hc_flat_rsqrt.is_finite() {
        return Err(gravity("hc_attn_pre rsqrt is non-finite"));
    }

    // `F.linear(x, hc_fn) * rsqrt`, no bias.  This scalar f32 inner-loop
    // order is documented as a source-algorithm transcription, not a claim
    // of the upstream BLAS/TileLang reduction implementation.
    let mut mixes = vec![0.0_f32; HC_MIX_WIDTH];
    for row in 0..HC_MIX_WIDTH {
        let mut accumulator = 0.0_f32;
        let weight_row = &hc_fn[row * HC_FLAT_WIDTH..(row + 1) * HC_FLAT_WIDTH];
        for (&weight, &value) in weight_row.iter().zip(&flat) {
            accumulator += weight * value;
        }
        mixes[row] = accumulator * hc_flat_rsqrt;
    }
    if mixes.iter().any(|value| !value.is_finite()) {
        return Err(gravity("hc_attn_pre linear mixes are non-finite"));
    }

    let (pre, post, comb) =
        hc_split_sinkhorn_source_algorithm(&mixes, hc_scale, hc_base, hc_eps, sinkhorn_iters)?;
    let mut reduced_bf16 = Vec::with_capacity(HIDDEN_SIZE);
    for feature in 0..HIDDEN_SIZE {
        // `torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)`: lane order
        // remains 0,1,2,3 before the source casts the reduced vector to BF16.
        let mut value = 0.0_f32;
        for lane in 0..HC_MULT {
            value += pre[lane] * flat[lane * HIDDEN_SIZE + feature];
        }
        if !value.is_finite() {
            return Err(gravity("hc_attn_pre reduction is non-finite"));
        }
        reduced_bf16.push(bf16::from_f32(value).to_bits());
    }
    Ok((hc_flat_rsqrt, mixes, pre, post, comb, reduced_bf16))
}

/// Exact source ordering of the kernel's sigmoid / softmax / Sinkhorn path.
pub fn hc_split_sinkhorn_source_algorithm(
    mixes: &[f32],
    hc_scale: &[f32],
    hc_base: &[f32],
    eps: f32,
    sinkhorn_iters: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
    if mixes.len() != HC_MIX_WIDTH
        || hc_scale.len() != 3
        || hc_base.len() != HC_MIX_WIDTH
        || !(eps.is_finite() && eps > 0.0)
        || sinkhorn_iters != HC_SINKHORN_ITERS
        || mixes
            .iter()
            .chain(hc_scale)
            .chain(hc_base)
            .any(|value| !value.is_finite())
    {
        return Err(gravity(
            "hc_split_sinkhorn source-algorithm input differs from the pinned contract",
        ));
    }

    let mut pre = Vec::with_capacity(HC_MULT);
    let mut post = Vec::with_capacity(HC_MULT);
    for lane in 0..HC_MULT {
        pre.push(sigmoid(mixes[lane] * hc_scale[0] + hc_base[lane]) + eps);
        post.push(2.0 * sigmoid(mixes[lane + HC_MULT] * hc_scale[1] + hc_base[lane + HC_MULT]));
    }
    let mut comb = vec![0.0_f32; HC_MULT * HC_MULT];
    for row in 0..HC_MULT {
        for column in 0..HC_MULT {
            let index = row * HC_MULT + column;
            let source_index = index + HC_MULT * 2;
            comb[index] = mixes[source_index] * hc_scale[2] + hc_base[source_index];
        }
    }

    // First source pass: `softmax(-1) + eps`, then one column normalization.
    // The `+ eps` belongs to each normalized softmax element (not its
    // denominator), matching the TileLang statement's operator order.
    for row in 0..HC_MULT {
        let start = row * HC_MULT;
        let row_max = comb[start..start + HC_MULT]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut row_sum = 0.0_f32;
        for column in 0..HC_MULT {
            let index = start + column;
            comb[index] = (comb[index] - row_max).exp();
            row_sum += comb[index];
        }
        if !(row_sum.is_finite() && row_sum > 0.0) {
            return Err(gravity("hc_split_sinkhorn initial softmax row is invalid"));
        }
        for column in 0..HC_MULT {
            let index = start + column;
            comb[index] = comb[index] / row_sum + eps;
        }
    }
    normalize_comb_columns(&mut comb, eps)?;

    // The source has already completed one row/column pass above; its serial
    // loop therefore executes exactly 19 additional row/column passes.
    for _ in 0..sinkhorn_iters - 1 {
        normalize_comb_rows(&mut comb, eps)?;
        normalize_comb_columns(&mut comb, eps)?;
    }
    if pre
        .iter()
        .chain(&post)
        .chain(&comb)
        .any(|value| !value.is_finite())
    {
        return Err(gravity("hc_split_sinkhorn produced a non-finite value"));
    }
    Ok((pre, post, comb))
}

/// Source `RMSNorm.forward`: BF16 input -> f32 variance/multiply -> BF16
/// output.  The stored BF16 RMS weight has already been converted to f32 at
/// model-load time in the pinned Python class.
pub fn rms_norm_source_algorithm(
    input_bf16_bits: &[u16],
    weight_bf16_bits: &[u16],
    eps: f32,
) -> Result<Vec<u16>> {
    if input_bf16_bits.len() != HIDDEN_SIZE
        || weight_bf16_bits.len() != HIDDEN_SIZE
        || !(eps.is_finite() && eps > 0.0)
    {
        return Err(gravity(
            "layer-0 attention RMSNorm geometry/constants differ from the pinned contract",
        ));
    }
    let input: Vec<f32> = input_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    let weight: Vec<f32> = weight_bf16_bits
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    if input.iter().chain(&weight).any(|value| !value.is_finite()) {
        return Err(gravity("attention RMSNorm BF16 input/weight is non-finite"));
    }
    let mut sum_square = 0.0_f32;
    for &value in &input {
        sum_square += value * value;
    }
    let reciprocal_rms = 1.0_f32 / (sum_square / HIDDEN_SIZE as f32 + eps).sqrt();
    if !reciprocal_rms.is_finite() {
        return Err(gravity("attention RMSNorm reciprocal RMS is non-finite"));
    }
    let mut output = Vec::with_capacity(HIDDEN_SIZE);
    for (&value, &scale) in input.iter().zip(&weight) {
        let normalized = value * reciprocal_rms * scale;
        if !normalized.is_finite() {
            return Err(gravity("attention RMSNorm output is non-finite"));
        }
        output.push(bf16::from_f32(normalized).to_bits());
    }
    Ok(output)
}

fn normalize_comb_rows(comb: &mut [f32], eps: f32) -> Result<()> {
    for row in 0..HC_MULT {
        let start = row * HC_MULT;
        let mut sum = 0.0_f32;
        for value in &comb[start..start + HC_MULT] {
            sum += *value;
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(gravity(
                "hc_split_sinkhorn row normalization sum is invalid",
            ));
        }
        for value in &mut comb[start..start + HC_MULT] {
            *value /= sum + eps;
        }
    }
    Ok(())
}

fn normalize_comb_columns(comb: &mut [f32], eps: f32) -> Result<()> {
    for column in 0..HC_MULT {
        let mut sum = 0.0_f32;
        for row in 0..HC_MULT {
            sum += comb[row * HC_MULT + column];
        }
        if !(sum.is_finite() && sum > 0.0) {
            return Err(gravity(
                "hc_split_sinkhorn column normalization sum is invalid",
            ));
        }
        for row in 0..HC_MULT {
            let index = row * HC_MULT + column;
            comb[index] /= sum + eps;
        }
    }
    Ok(())
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn expect_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    dtype: &str,
    shape: &[u64],
) -> Result<DeepSeekV4TensorMetadata> {
    let tensor = reader.tensor_metadata(name)?;
    if tensor.dtype != dtype || tensor.shape.as_slice() != shape {
        return Err(gravity(format!(
            "{name} has source dtype/shape {:?}/{:?}; expected {dtype:?}/{shape:?}",
            tensor.dtype, tensor.shape
        )));
    }
    Ok(tensor.clone())
}

fn decode_u16_le(bytes: &[u8], name: &str) -> Result<Vec<u16>> {
    if bytes.len() % std::mem::size_of::<u16>() != 0 {
        return Err(gravity(format!("{name} BF16 bytes are not u16 aligned")));
    }
    Ok(bytes
        .chunks_exact(std::mem::size_of::<u16>())
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn decode_f32_le(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(gravity(format!("{name} F32 bytes are not f32 aligned")));
    }
    let values: Vec<f32> = bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(gravity(format!("{name} has a non-finite F32 source value")));
    }
    Ok(values)
}

fn parse_json(bytes: &[u8], name: &str) -> Result<Value> {
    serde_json::from_slice(bytes)
        .map_err(|error| gravity(format!("pinned {name} JSON parsing failed: {error}")))
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

fn json_f64_eq(value: &Value, path: &[&str], expected: f32) -> bool {
    json_path(value, path, "numeric config value")
        .ok()
        .and_then(Value::as_f64)
        .map(|actual| (actual as f32).to_bits() == expected.to_bits())
        .unwrap_or(false)
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

fn gravity(message: impl Into<String>) -> Error {
    Error::Gravity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_uses_pinned_four_by_four_twenty_iteration_contract() {
        let mixes: Vec<f32> = (0..HC_MIX_WIDTH)
            .map(|index| (index as f32 - 9.0) * 0.125)
            .collect();
        let scale = [0.5, 0.75, 1.25];
        let base: Vec<f32> = (0..HC_MIX_WIDTH)
            .map(|index| (index as f32 - 11.0) * 0.03125)
            .collect();
        let (pre, post, comb) =
            hc_split_sinkhorn_source_algorithm(&mixes, &scale, &base, HC_EPS, HC_SINKHORN_ITERS)
                .unwrap();
        assert_eq!(pre.len(), HC_MULT);
        assert_eq!(post.len(), HC_MULT);
        assert_eq!(comb.len(), HC_MULT * HC_MULT);
        assert!(pre.iter().all(|value| value.is_finite() && *value > HC_EPS));
        assert!(post.iter().all(|value| value.is_finite() && *value > 0.0));
        assert!(comb.iter().all(|value| value.is_finite() && *value > 0.0));
        for row in 0..HC_MULT {
            let sum: f32 = comb[row * HC_MULT..(row + 1) * HC_MULT].iter().sum();
            assert!((sum - 1.0).abs() < 0.002, "row={row} sum={sum}");
        }
        for column in 0..HC_MULT {
            let sum: f32 = (0..HC_MULT).map(|row| comb[row * HC_MULT + column]).sum();
            assert!((sum - 1.0).abs() < 0.002, "column={column} sum={sum}");
        }
    }

    #[test]
    fn layer0_hc_pre_preserves_replicated_bf16_geometry() {
        let row: Vec<u16> = (0..HIDDEN_SIZE)
            .map(|index| bf16::from_f32((index as f32 - 1024.0) / 512.0).to_bits())
            .collect();
        let mut replicated = Vec::with_capacity(HC_FLAT_WIDTH);
        for _ in 0..HC_MULT {
            replicated.extend_from_slice(&row);
        }
        let hc_fn = vec![0.0_f32; HC_MIX_WIDTH * HC_FLAT_WIDTH];
        let hc_scale = [1.0_f32, 1.0, 1.0];
        let hc_base = vec![0.0_f32; HC_MIX_WIDTH];
        let (_, mixes, pre, post, comb, reduced) = hc_attn_pre_source_algorithm(
            &replicated,
            &hc_fn,
            &hc_scale,
            &hc_base,
            RMS_NORM_EPS,
            HC_EPS,
            HC_SINKHORN_ITERS,
        )
        .unwrap();
        assert_eq!(mixes, vec![0.0; HC_MIX_WIDTH]);
        assert_eq!(pre.len(), HC_MULT);
        assert_eq!(post.len(), HC_MULT);
        assert_eq!(comb.len(), HC_MULT * HC_MULT);
        assert_eq!(reduced.len(), HIDDEN_SIZE);
        // All four source lanes are equal, so the pre-weighted reduction is
        // one shared scalar factor times the original row.
        let factor: f32 = pre.iter().sum();
        for (expected, observed) in row.iter().zip(&reduced) {
            assert_eq!(
                bf16::from_f32(bf16::from_bits(*expected).to_f32() * factor).to_bits(),
                *observed
            );
        }
    }

    #[test]
    fn rms_norm_returns_bf16_width_and_rejects_bad_geometry() {
        let input = vec![bf16::from_f32(1.0).to_bits(); HIDDEN_SIZE];
        let weight = vec![bf16::from_f32(1.0).to_bits(); HIDDEN_SIZE];
        let output = rms_norm_source_algorithm(&input, &weight, RMS_NORM_EPS).unwrap();
        assert_eq!(output.len(), HIDDEN_SIZE);
        assert!(
            rms_norm_source_algorithm(&input[..HIDDEN_SIZE - 1], &weight, RMS_NORM_EPS).is_err()
        );
    }
}
