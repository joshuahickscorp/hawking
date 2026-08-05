//! CPU-only source-algorithm oracle for the first DeepSeek-V4 FP8 `Linear` checkpoint.
//!
//! This module intentionally stops before a model runtime boundary.  It turns
//! an explicit BF16 activation row into the exact *algorithm shape* selected
//! by the pinned `inference/model.py::linear` / `inference/kernel.py` pair:
//!
//! ```text
//! BF16 [K]
//!   -> act_quant(block=128, scale_fmt="ue8m0", scale_dtype=E8M0FNU)
//!   -> E4M3FN [K] + E8M0FNU [K / 128]
//!   -> fp8_gemm against E4M3FN/E8M0FNU `layers.0.attn.wq_a`
//! ```
//!
//! It is deliberately a scalar CPU reference, not an alternative runtime.
//! In particular, it has no Metal allocations, dispatches, engine, forward
//! loop, generation, endpoint, or TPS surface.  The source files are bound by
//! exact hashes, but this remains a **source-derived algorithm oracle**, not
//! independently executed upstream runtime parity.

use crate::gravity_deepseek_v4::{
    DeepSeekV4FullStreamReader, NativeScalePairKind, PINNED_REPOSITORY, PINNED_REVISION,
};
use crate::{Error, Result};
use half::bf16;

/// The native activation / weight scale block selected by the pinned V4 FP8
/// `Linear` source path.
pub const ACT_QUANT_BLOCK: usize = 128;
/// The finite maximum of `torch.float8_e4m3fn` used by `act_quant_kernel`.
pub const E4M3FN_MAX: f32 = 448.0;
/// The lower bound applied to each block's absolute maximum by the pinned
/// `act_quant_kernel` before its rounded scale is computed.
pub const ACT_QUANT_AMAX_FLOOR: f32 = 1.0e-4;
/// The exact source tensor used for the bounded first `Linear` checkpoint.
pub const LAYER0_WQ_A_WEIGHT: &str = "layers.0.attn.wq_a.weight";
/// The exact source scale tensor paired with [`LAYER0_WQ_A_WEIGHT`].
pub const LAYER0_WQ_A_SCALE: &str = "layers.0.attn.wq_a.scale";
pub const LAYER0_WQ_A_ROWS: usize = 1024;
pub const LAYER0_WQ_A_COLS: usize = 4096;

/// Pinned official source anchors.  These values are checked in addition to
/// full-stream admission, so a different artifact cannot silently substitute
/// a different algorithm while retaining the same repository/revision label.
pub const OFFICIAL_INFERENCE_MODEL_PY_SHA256: &str =
    "ce962f1face79d4f633d36436576214057a7e11443c9789935e1deb5c6cd1d71";
pub const OFFICIAL_INFERENCE_KERNEL_PY_SHA256: &str =
    "59b325083d7103975cba025bd0d60ea343bb82d8fff53088afb7c04bd380c0c2";
pub const OFFICIAL_INFERENCE_CONFIG_JSON_SHA256: &str =
    "6cc6f816ca73a8d38750194e330398e4f6955b4b45f674f7d29c96da14ccb733";
pub const OFFICIAL_MODEL_CONFIG_JSON_SHA256: &str =
    "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818";

/// Exact source-code anchors that have been admitted and checked before this
/// CPU oracle can run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeepSeekV4ActQuantSourceAnchors {
    pub inference_model_py_sha256: String,
    pub inference_kernel_py_sha256: String,
    pub inference_config_json_sha256: String,
    pub model_config_json_sha256: String,
}

/// Source-derived output of `act_quant` for one BF16 activation row.
///
/// `activation_e4m3fn` and `scales_e8m0fnu` are native storage bytes.  The
/// f32 scales are included only to make the CPU GEMV reference unambiguous;
/// they are losslessly decoded from the E8M0 bytes and are not a second scale
/// representation to persist.
#[derive(Debug, Clone, PartialEq)]
pub struct ActQuantizedBf16Row {
    pub activation_e4m3fn: Vec<u8>,
    pub scales_e8m0fnu: Vec<u8>,
    pub decoded_scales_f32: Vec<f32>,
}

/// Scalar CPU realization of the source `fp8_gemm` block structure.
///
/// The f32 values are the source-kernel-shaped block accumulators.  The BF16
/// bits model the usual `torch.set_default_dtype(torch.bfloat16)` output
/// storage used by the shipped source's example, but are not an independent
/// upstream execution result.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp8MatvecCpuResult {
    pub fp32: Vec<f32>,
    pub bf16_bits: Vec<u16>,
}

/// Bounded result of applying the layer-0 WQ-A source tensor to one explicit
/// deterministic or caller-provided BF16 activation row.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0WqACpuOracleResult {
    pub quantized_input: ActQuantizedBf16Row,
    pub output: Fp8MatvecCpuResult,
}

/// Verify the source-code and config hashes this algorithm transcription
/// depends on.  Full-stream admission has already checked that every listed
/// asset is a regular file whose bytes match its sealed manifest binding; this
/// function additionally pins those bindings to the expected official values.
pub fn verify_source_algorithm_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<DeepSeekV4ActQuantSourceAnchors> {
    let identity = reader.source_identity();
    if identity.repository != PINNED_REPOSITORY || identity.revision != PINNED_REVISION {
        return Err(gravity(
            "act-quant oracle reader source identity is not the pinned DeepSeek-V4-Flash revision",
        ));
    }

    let anchors = DeepSeekV4ActQuantSourceAnchors {
        inference_model_py_sha256: reader
            .source_metadata_asset_sha256("inference/model.py")?
            .to_owned(),
        inference_kernel_py_sha256: reader
            .source_metadata_asset_sha256("inference/kernel.py")?
            .to_owned(),
        inference_config_json_sha256: reader
            .source_metadata_asset_sha256("inference/config.json")?
            .to_owned(),
        model_config_json_sha256: reader
            .source_metadata_asset_sha256("config.json")?
            .to_owned(),
    };
    if anchors.inference_model_py_sha256 != OFFICIAL_INFERENCE_MODEL_PY_SHA256
        || anchors.inference_kernel_py_sha256 != OFFICIAL_INFERENCE_KERNEL_PY_SHA256
        || anchors.inference_config_json_sha256 != OFFICIAL_INFERENCE_CONFIG_JSON_SHA256
        || anchors.model_config_json_sha256 != OFFICIAL_MODEL_CONFIG_JSON_SHA256
    {
        return Err(gravity(
            "act-quant oracle source-code/config hashes differ from pinned official anchors",
        ));
    }
    Ok(anchors)
}

/// Decode one finite `float8_e4m3fn` byte according to the pinned source
/// grammar.  `0x7f` / `0xff` are the only NaN encodings and are rejected.
pub fn decode_e4m3fn(bits: u8) -> Result<f32> {
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return Err(gravity("E4M3FN contains its NaN encoding"));
    }
    let magnitude = if exponent == 0 {
        (mantissa as f32) * 0.001_953_125_f32 // 2^-9, exact in f32.
    } else {
        // The E4 bias is 7; `exponent + 120` is the matching f32 exponent
        // field and E4's 3 fraction bits occupy f32 bits 22..20.
        f32::from_bits(((exponent as u32 + 120) << 23) | ((mantissa as u32) << 20))
    };
    Ok(if bits & 0x80 == 0 {
        magnitude
    } else {
        -magnitude
    })
}

/// Decode one finite `float8_e8m0fnu` scale byte.  Byte zero represents
/// `2^-127`, an f32 subnormal; `0xff` is the sole NaN encoding.
pub fn decode_e8m0fnu(bits: u8) -> Result<f32> {
    if bits == 0xff {
        return Err(gravity("E8M0FNU contains its NaN encoding"));
    }
    Ok(if bits == 0 {
        f32::from_bits(0x0040_0000) // 2^-127
    } else {
        f32::from_bits((bits as u32) << 23)
    })
}

/// Encode an already-clamped finite f32 as `float8_e4m3fn` using
/// round-to-nearest, ties-to-even.  `act_quant` calls this only after clamping
/// the scaled value to `[-448, 448]`.
pub fn encode_e4m3fn_rne(value: f32) -> Result<u8> {
    if !value.is_finite() {
        return Err(gravity("cannot encode non-finite E4M3FN activation"));
    }
    if value.abs() > E4M3FN_MAX {
        return Err(gravity(
            "E4M3FN activation escaped the required finite clamp",
        ));
    }
    if value == 0.0 {
        return Ok(if value.is_sign_negative() { 0x80 } else { 0x00 });
    }

    // There are only 254 finite E4M3FN encodings.  An exhaustive finite table
    // makes the unusual E4M3FN top bin (`448`, while `0x7f` is NaN) explicit
    // and avoids accidentally treating this as IEEE E4M3 with infinity.
    let mut best_bits = None::<u8>;
    let mut best_distance = f32::INFINITY;
    for raw in 0u16..=u8::MAX as u16 {
        let bits = raw as u8;
        let Ok(candidate) = decode_e4m3fn(bits) else {
            continue;
        };
        let distance = (candidate - value).abs();
        match best_bits {
            None => {
                best_bits = Some(bits);
                best_distance = distance;
            }
            Some(_) if distance < best_distance => {
                best_bits = Some(bits);
                best_distance = distance;
            }
            Some(previous)
                if distance == best_distance && (bits & 1) == 0 && (previous & 1) != 0 =>
            {
                // IEEE conversion is nearest-even.  For equal-sign adjacent
                // E4M3FN values, the encoded LSB is the significand parity.
                best_bits = Some(bits);
                best_distance = distance;
            }
            _ => {}
        }
    }
    best_bits.ok_or_else(|| gravity("E4M3FN finite encoding table was empty"))
}

/// Produce the rounded power-of-two E8M0 scale selected by the pinned
/// `fast_round_scale(amax, 1 / 448)` path.  The returned byte is exactly the
/// native `float8_e8m0fnu` storage representation.
pub fn rounded_ue8m0_scale_byte(amax: f32) -> Result<u8> {
    if !amax.is_finite() || amax < 0.0 {
        return Err(gravity(
            "act-quant absolute maximum must be finite and non-negative",
        ));
    }
    let clamped = amax.max(ACT_QUANT_AMAX_FLOOR);
    // This is the source's `fast_log2_ceil`: read f32's exponent and add one
    // only when a nonzero mantissa makes the value strictly above a power of
    // two.  The floor above ensures this product is normal and positive.
    let scaled = clamped * (1.0_f32 / E4M3FN_MAX);
    let raw = scaled.to_bits();
    let exponent_field = ((raw >> 23) & 0xff) as i32;
    let mantissa = raw & 0x007f_ffff;
    if exponent_field == 0 || exponent_field == 0xff {
        return Err(gravity(
            "act-quant rounded-scale input is outside normal f32 range",
        ));
    }
    let exponent = exponent_field - 127 + i32::from(mantissa != 0);
    let e8m0 = exponent + 127;
    if !(0..=254).contains(&e8m0) {
        return Err(gravity(
            "act-quant rounded scale cannot be represented as finite E8M0FNU",
        ));
    }
    let bits = e8m0 as u8;
    // Check the bit-level `fast_pow2` construction rather than relying on an
    // approximate exponentiation implementation.
    let round_trip = decode_e8m0fnu(bits)?;
    if round_trip.to_bits() != ((e8m0 as u32) << 23) && bits != 0 {
        return Err(gravity(
            "act-quant E8M0 round-trip lost its exact power-of-two scale",
        ));
    }
    Ok(bits)
}

/// Transcribe the pinned source `act_quant` configuration used by FP8
/// `Linear`: exact BF16 input bits, block size 128, rounded UE8M0 scales, and
/// E4M3FN output bytes.
pub fn act_quant_bf16_ue8m0(input_bf16_bits: &[u16]) -> Result<ActQuantizedBf16Row> {
    if input_bf16_bits.is_empty() || input_bf16_bits.len() % ACT_QUANT_BLOCK != 0 {
        return Err(gravity(format!(
            "act-quant requires a nonempty BF16 row whose length is divisible by {ACT_QUANT_BLOCK}",
        )));
    }
    let mut input = Vec::with_capacity(input_bf16_bits.len());
    for &bits in input_bf16_bits {
        let value = bf16::from_bits(bits).to_f32();
        if !value.is_finite() {
            return Err(gravity("act-quant input BF16 row contains NaN or infinity"));
        }
        input.push(value);
    }

    let mut activation_e4m3fn = Vec::with_capacity(input.len());
    let mut scales_e8m0fnu = Vec::with_capacity(input.len() / ACT_QUANT_BLOCK);
    let mut decoded_scales_f32 = Vec::with_capacity(input.len() / ACT_QUANT_BLOCK);
    for block in input.chunks_exact(ACT_QUANT_BLOCK) {
        let amax = block
            .iter()
            .fold(0.0_f32, |current, value| current.max(value.abs()));
        let scale_bits = rounded_ue8m0_scale_byte(amax)?;
        let scale = decode_e8m0fnu(scale_bits)?;
        scales_e8m0fnu.push(scale_bits);
        decoded_scales_f32.push(scale);
        for &value in block {
            // Mirrors `T.clamp(x / s, -448, 448)` before the E4M3FN cast.
            activation_e4m3fn.push(encode_e4m3fn_rne(
                (value / scale).clamp(-E4M3FN_MAX, E4M3FN_MAX),
            )?);
        }
    }
    Ok(ActQuantizedBf16Row {
        activation_e4m3fn,
        scales_e8m0fnu,
        decoded_scales_f32,
    })
}

/// CPU transcription of the source `fp8_gemm` block structure for one row of
/// activations and an `[out, K]` E4M3FN weight matrix.  Both activation and
/// weight scales use 128-wide K blocks; weight scales additionally cover 128
/// output rows per scale row.
pub fn fp8_e4m3fn_ue8m0_matvec(
    activation: &ActQuantizedBf16Row,
    weights_e4m3fn: &[u8],
    weight_scales_e8m0fnu: &[u8],
    output_rows: usize,
    logical_k: usize,
) -> Result<Fp8MatvecCpuResult> {
    if logical_k == 0
        || output_rows == 0
        || logical_k % ACT_QUANT_BLOCK != 0
        || output_rows % ACT_QUANT_BLOCK != 0
    {
        return Err(gravity(
            "FP8 source GEMV requires nonzero [out, K] dimensions divisible by 128",
        ));
    }
    if activation.activation_e4m3fn.len() != logical_k
        || activation.scales_e8m0fnu.len() != logical_k / ACT_QUANT_BLOCK
        || activation.decoded_scales_f32.len() != logical_k / ACT_QUANT_BLOCK
    {
        return Err(gravity(
            "FP8 source GEMV activation geometry is not [K] plus [K/128] scales",
        ));
    }
    let expected_weight_bytes = output_rows
        .checked_mul(logical_k)
        .ok_or_else(|| gravity("FP8 source GEMV weight byte count overflow"))?;
    if weights_e4m3fn.len() != expected_weight_bytes {
        return Err(gravity(
            "FP8 source GEMV weight byte count differs from [out, K]",
        ));
    }
    let scale_cols = logical_k / ACT_QUANT_BLOCK;
    let expected_scale_bytes = (output_rows / ACT_QUANT_BLOCK)
        .checked_mul(scale_cols)
        .ok_or_else(|| gravity("FP8 source GEMV scale byte count overflow"))?;
    if weight_scales_e8m0fnu.len() != expected_scale_bytes {
        return Err(gravity(
            "FP8 source GEMV weight-scale byte count differs from [out/128, K/128]",
        ));
    }

    let mut output = Vec::with_capacity(output_rows);
    for row in 0..output_rows {
        let row_base = row * logical_k;
        let scale_row = row / ACT_QUANT_BLOCK;
        let mut row_accumulator = 0.0_f32;
        for block in 0..scale_cols {
            let activation_scale = activation.decoded_scales_f32[block];
            let weight_scale =
                decode_e8m0fnu(weight_scales_e8m0fnu[scale_row * scale_cols + block])?;
            let combined_scale = activation_scale * weight_scale;
            let start = block * ACT_QUANT_BLOCK;
            let mut block_accumulator = 0.0_f32;
            for col in start..start + ACT_QUANT_BLOCK {
                let a = decode_e4m3fn(activation.activation_e4m3fn[col])?;
                let b = decode_e4m3fn(weights_e4m3fn[row_base + col])?;
                // Deliberately explicit scalar product-then-add order.  It
                // reflects the source kernel's FP32 block accumulator shape;
                // it is not a claim of bitwise TileLang/CUDA execution parity.
                block_accumulator += a * b;
            }
            row_accumulator += block_accumulator * combined_scale;
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

/// Read exactly the real layer-0 WQ-A weight/scale pair through the admitted
/// content-addressed reader, quantize the supplied BF16 input, and run the
/// CPU source-algorithm GEMV.  At most one 4 MiB tensor and one 256-byte scale
/// tensor are held temporarily; no parent source file is materialized.
pub fn layer0_wq_a_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
    input_bf16_bits: &[u16],
) -> Result<Layer0WqACpuOracleResult> {
    verify_source_algorithm_anchors(reader)?;
    let scale_name = {
        let pair = reader.native_scale_pair(LAYER0_WQ_A_WEIGHT)?;
        if pair.kind != NativeScalePairKind::Fp8E4M3fn
            || pair.weight.name != LAYER0_WQ_A_WEIGHT
            || pair.scale.name != LAYER0_WQ_A_SCALE
            || pair.weight.shape.as_slice() != [LAYER0_WQ_A_ROWS as u64, LAYER0_WQ_A_COLS as u64]
            || pair.scale.shape.as_slice()
                != [
                    (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) as u64,
                    (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK) as u64,
                ]
            || pair.logical_k != LAYER0_WQ_A_COLS as u64
            || pair.out_rows != LAYER0_WQ_A_ROWS as u64
        {
            return Err(gravity(
                "layer-0 WQ-A does not match the pinned FP8/E8M0 source geometry",
            ));
        }
        pair.scale.name.clone()
    };

    let quantized_input = act_quant_bf16_ue8m0(input_bf16_bits)?;
    let weights =
        reader.read_verified_full(LAYER0_WQ_A_WEIGHT, LAYER0_WQ_A_ROWS * LAYER0_WQ_A_COLS)?;
    let scales = reader.read_verified_full(
        &scale_name,
        (LAYER0_WQ_A_ROWS / ACT_QUANT_BLOCK) * (LAYER0_WQ_A_COLS / ACT_QUANT_BLOCK),
    )?;
    let output = fp8_e4m3fn_ue8m0_matvec(
        &quantized_input,
        &weights,
        &scales,
        LAYER0_WQ_A_ROWS,
        LAYER0_WQ_A_COLS,
    )?;
    Ok(Layer0WqACpuOracleResult {
        quantized_input,
        output,
    })
}

/// Deterministic, exact-BF16 input for the bounded receipt.  It is an
/// algorithm probe input, not a hidden state captured from a prompt or a
/// model forward.  Every element is assembled directly from a finite BF16 bit
/// pattern, avoiding host RNG and f32-to-BF16 rounding variability.
pub fn deterministic_wq_a_input_bf16() -> Vec<u16> {
    (0..LAYER0_WQ_A_COLS)
        .map(|index| {
            let block = index / ACT_QUANT_BLOCK;
            let sign = if (index.wrapping_mul(17).wrapping_add(block * 3)) & 1 == 0 {
                0u16
            } else {
                0x8000u16
            };
            // Exponents 121..131 keep all inputs finite and provide a stable
            // range of scale rungs across 32 independent blocks.
            let exponent = 121u16 + ((index.wrapping_mul(13).wrapping_add(block * 7)) % 11) as u16;
            let mantissa = ((index.wrapping_mul(29).wrapping_add(block * 11)) % 128) as u16;
            sign | (exponent << 7) | mantissa
        })
        .collect()
}

fn gravity(message: impl Into<String>) -> Error {
    Error::Gravity(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3fn_codec_keeps_finite_top_bin_and_rejects_nan() {
        assert_eq!(decode_e4m3fn(0x00).unwrap().to_bits(), 0.0_f32.to_bits());
        assert!(decode_e4m3fn(0x80).unwrap().is_sign_negative());
        assert_eq!(decode_e4m3fn(0x38).unwrap(), 1.0);
        assert_eq!(decode_e4m3fn(0x7e).unwrap(), 448.0);
        assert!(decode_e4m3fn(0x7f).is_err());
        assert!(decode_e4m3fn(0xff).is_err());
        for raw in 0u16..=u8::MAX as u16 {
            let bits = raw as u8;
            if let Ok(value) = decode_e4m3fn(bits) {
                assert_eq!(encode_e4m3fn_rne(value).unwrap(), bits, "bits={bits:#04x}");
            }
        }
    }

    #[test]
    fn e4m3fn_encoder_uses_nearest_even_at_halfway_values() {
        // Midway between 1.0 (0x38, even) and 1.125 (0x39, odd).
        assert_eq!(encode_e4m3fn_rne(1.0625).unwrap(), 0x38);
        // Midway between 1.125 (0x39, odd) and 1.25 (0x3a, even).
        assert_eq!(encode_e4m3fn_rne(1.1875).unwrap(), 0x3a);
        assert_eq!(encode_e4m3fn_rne(-0.0).unwrap(), 0x80);
    }

    #[test]
    fn e8m0_codec_includes_byte_zero_subnormal_and_rejects_nan() {
        assert_eq!(decode_e8m0fnu(0).unwrap(), 2.0_f32.powi(-127));
        assert_eq!(decode_e8m0fnu(0x7f).unwrap(), 1.0);
        assert_eq!(decode_e8m0fnu(0x80).unwrap(), 2.0);
        assert!(decode_e8m0fnu(0xff).is_err());
    }

    #[test]
    fn act_quant_uses_ue8m0_rounding_and_minimum_scale() {
        let one = vec![0x3f80u16; ACT_QUANT_BLOCK];
        let quantized = act_quant_bf16_ue8m0(&one).unwrap();
        // ceil(log2(1 / 448)) = -8 -> E8M0 exponent byte 119.
        assert_eq!(quantized.scales_e8m0fnu, vec![119]);
        assert_eq!(quantized.decoded_scales_f32, vec![2.0_f32.powi(-8)]);
        assert_eq!(quantized.activation_e4m3fn, vec![0x78; ACT_QUANT_BLOCK]);

        let zeros = vec![0u16; ACT_QUANT_BLOCK];
        let zero_quantized = act_quant_bf16_ue8m0(&zeros).unwrap();
        // ceil(log2(1e-4 / 448)) = -22 -> E8M0 byte 105.
        assert_eq!(zero_quantized.scales_e8m0fnu, vec![105]);
        assert_eq!(zero_quantized.activation_e4m3fn, vec![0; ACT_QUANT_BLOCK]);
    }

    #[test]
    fn fp8_matvec_preserves_source_block_scale_layout() {
        let input = act_quant_bf16_ue8m0(&vec![0x3f80u16; ACT_QUANT_BLOCK]).unwrap();
        // 128 rows x 128 columns are required by the native [out/128, K/128]
        // scale geometry.  Both units and weight scale decode to one.
        let weights = vec![0x38u8; ACT_QUANT_BLOCK * ACT_QUANT_BLOCK];
        let scales = vec![0x7fu8];
        let output =
            fp8_e4m3fn_ue8m0_matvec(&input, &weights, &scales, ACT_QUANT_BLOCK, ACT_QUANT_BLOCK)
                .unwrap();
        assert_eq!(output.fp32, vec![128.0; ACT_QUANT_BLOCK]);
        assert_eq!(
            output.bf16_bits,
            vec![bf16::from_f32(128.0).to_bits(); ACT_QUANT_BLOCK]
        );
    }

    #[test]
    fn act_quant_rejects_bad_geometry_and_nonfinite_bf16() {
        assert!(act_quant_bf16_ue8m0(&vec![0u16; ACT_QUANT_BLOCK - 1]).is_err());
        assert!(act_quant_bf16_ue8m0(&vec![0x7f80u16; ACT_QUANT_BLOCK]).is_err());
    }

    #[test]
    fn deterministic_input_is_exact_bf16_and_repeatable() {
        let first = deterministic_wq_a_input_bf16();
        let second = deterministic_wq_a_input_bf16();
        assert_eq!(first, second);
        assert_eq!(first.len(), LAYER0_WQ_A_COLS);
        assert!(first
            .iter()
            .all(|bits| bf16::from_bits(*bits).to_f32().is_finite()));
    }
}
