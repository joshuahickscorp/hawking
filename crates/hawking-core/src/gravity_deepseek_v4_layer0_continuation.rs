//! Bounded source-algorithm continuation proof for DeepSeek-V4 layer 0.
//!
//! This module covers the first real causal cache transition only:
//! tokenizer-bound `[BOS, Hello]`, layer 0, ratio zero, position one.  It
//! proves the Q/K RoPE phase, device-cache-compatible two-row KV layout, and
//! the source sparse-attention/sink arithmetic.  It intentionally stops
//! before WO-A/WO-B/mHC-post: P4A already establishes that tail at position
//! zero, while this module isolates the new causal state/read/write boundary.

use crate::gravity_deepseek_v4::{DeepSeekV4FullStreamReader, NativeScalePairKind};
use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, fp8_e4m3fn_ue8m0_matvec, ActQuantizedBf16Row, Fp8MatvecCpuResult,
    ACT_QUANT_BLOCK, LAYER0_WQ_A_COLS, LAYER0_WQ_A_ROWS, LAYER0_WQ_A_SCALE, LAYER0_WQ_A_WEIGHT,
};
use crate::gravity_deepseek_v4_layer0_attention::{
    hc_attn_post_source_algorithm, kv_non_rope_inplace_qat_source_algorithm,
    per_head_rms_norm_bf16_source_algorithm, position_zero_rope_identity,
    rms_norm_bf16_source_algorithm, verify_layer0_attention_source_anchors,
    wo_a_bf16_einsum_source_algorithm, KvInplaceQatCpuResult, HEAD_DIM, KV_QAT_BLOCK,
    LAYER0_ATTN_SINK, LAYER0_KV_NORM_WEIGHT, LAYER0_Q_NORM_WEIGHT, LAYER0_WKV_SCALE,
    LAYER0_WKV_WEIGHT, LAYER0_WO_B_SCALE, LAYER0_WO_B_WEIGHT, LAYER0_WQ_B_SCALE,
    LAYER0_WQ_B_WEIGHT, NUM_HEADS, Q_LORA_RANK, ROPE_HEAD_DIM, WKV_ROWS, WO_B_COLS, WO_B_ROWS,
    WQ_B_ROWS,
};
use crate::gravity_deepseek_v4_layer0_prefix::{
    layer0_prefix_cpu_oracle, layer0_prefix_cpu_oracle_for_token, Layer0PrefixCpuOracleResult,
    PREFIX_TOKEN_ID, RMS_NORM_EPS,
};
use crate::{Error, Result};
use half::bf16;
use serde_json::Value;

/// The second token of the deliberately tiny, tokenizer-bound continuation
/// trace.  The fixed BPE mapping is verified from the admitted tokenizer
/// before the embedding row is read.
pub const POSITION1_TOKEN_ID: u64 = 19_923;
pub const POSITION1_TOKEN_STRING: &str = "Hello";
pub const POSITION1: usize = 1;
pub const POSITION1_KV_ROWS: usize = 2;
pub const WINDOW_SIZE: usize = 128;

pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_FACTOR: f32 = 16.0;
pub const ROPE_ORIGINAL_SEQ_LEN: usize = 65_536;
pub const ROPE_BETA_FAST: f32 = 32.0;
pub const ROPE_BETA_SLOW: f32 = 1.0;

/// One explicit FP8 source linear operation at the position-one input.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuationFp8LinearCpuStage {
    pub quantized_input: ActQuantizedBf16Row,
    pub output: Fp8MatvecCpuResult,
}

/// Source-derived position-one RoPE table.  The device may upload these 32
/// static F32 cos/sin pairs; they are source configuration, not host-produced
/// hidden state.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionOneRopeTable {
    pub cos_f32: Vec<f32>,
    pub sin_f32: Vec<f32>,
}

/// Bounded CPU authority for the two-row causal KV transition.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0Position1ContinuationCpuOracleResult {
    pub token0_id: u64,
    pub token1_id: u64,
    pub token1_prefix: Layer0PrefixCpuOracleResult,
    pub rope_table: PositionOneRopeTable,
    pub position0_wkv: ContinuationFp8LinearCpuStage,
    pub position0_kv_norm_bf16_bits: Vec<u16>,
    pub position0_kv_inplace_qat: KvInplaceQatCpuResult,
    pub position0_kv_rope_bf16_bits: Vec<u16>,
    pub wq_a: ContinuationFp8LinearCpuStage,
    pub q_norm_bf16_bits: Vec<u16>,
    pub wq_b: ContinuationFp8LinearCpuStage,
    pub q_head_norm_bf16_bits: Vec<u16>,
    pub q_position1_rope_bf16_bits: Vec<u16>,
    pub wkv: ContinuationFp8LinearCpuStage,
    pub kv_norm_bf16_bits: Vec<u16>,
    pub kv_inplace_qat: KvInplaceQatCpuResult,
    pub kv_position1_rope_bf16_bits: Vec<u16>,
    /// Row-major `[cache_position, head_dim]`, containing exactly rows 0 and
    /// 1 after the second source cache write.
    pub kv_cache_two_rows_bf16_bits: Vec<u16>,
    /// `[head, causal_kv_position]`, in the exact `topk_idxs=[0,1]` order.
    pub sparse_attention_scores_f32: Vec<f32>,
    pub sparse_attention_sink_denominators_f32: Vec<f32>,
    pub sparse_attention_bf16_bits: Vec<u16>,
    pub sparse_attention_derotated_bf16_bits: Vec<u16>,
}

/// A bounded extension of the causal-KV receipt through the exact source
/// attention tail at position one.  It remains one tiny tokenizer-bound
/// trace, but exposes the child-body handoff P7 needs: the four BF16 mHC
/// residual lanes after `WO-A -> WO-B -> hc_attn_post`.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer0Position1CompleteAttentionCpuOracleResult {
    pub causal: Layer0Position1ContinuationCpuOracleResult,
    pub wo_a_bf16_bits: Vec<u16>,
    pub wo_b: ContinuationFp8LinearCpuStage,
    pub hc_attention_post_bf16_bits: Vec<u16>,
}

/// Verify the exact second-token mapping and RoPE configuration used by the
/// pinned inference source.  These checks are intentionally independent from
/// a tokenizer implementation so no unqualified prompt encoding is smuggled
/// into the bounded proof.
pub fn verify_layer0_position1_continuation_anchors(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<()> {
    verify_layer0_attention_source_anchors(reader)?;
    let tokenizer: Value = serde_json::from_slice(
        &reader.read_verified_metadata_asset("tokenizer.json", 8 * 1024 * 1024)?,
    )
    .map_err(|error| continuation(format!("tokenizer JSON: {error}")))?;
    let token_id = tokenizer
        .pointer("/model/vocab")
        .and_then(Value::as_object)
        .and_then(|vocab| vocab.get(POSITION1_TOKEN_STRING))
        .and_then(Value::as_u64)
        .ok_or_else(|| continuation("tokenizer lacks bounded position-one BPE token"))?;
    if token_id != POSITION1_TOKEN_ID {
        return Err(continuation(
            "position-one BPE token mapping differs from pinned source",
        ));
    }
    let config: Value = serde_json::from_slice(
        &reader.read_verified_metadata_asset("inference/config.json", 64 * 1024)?,
    )
    .map_err(|error| continuation(format!("inference config JSON: {error}")))?;
    if json_u64(&config, "rope_head_dim")? != ROPE_HEAD_DIM as u64
        || json_u64(&config, "rope_theta")? != ROPE_THETA as u64
        || json_u64(&config, "original_seq_len")? != ROPE_ORIGINAL_SEQ_LEN as u64
        || !json_f32_eq(&config, "rope_factor", ROPE_FACTOR)
        || json_u64(&config, "beta_fast")? != ROPE_BETA_FAST as u64
        || json_u64(&config, "beta_slow")? != ROPE_BETA_SLOW as u64
        || json_u64(&config, "window_size")? != WINDOW_SIZE as u64
        || json_array_first_u64(&config, "compress_ratios")? != 0
    {
        return Err(continuation(
            "position-one RoPE/window/ratio-zero source configuration differs",
        ));
    }
    Ok(())
}

/// Execute the source-derived layer-0 position-one continuation path through
/// causal two-row sparse attention.  This is a CPU oracle only; it neither
/// runs an upstream runtime nor establishes a causal Engine.
pub fn layer0_position1_continuation_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<Layer0Position1ContinuationCpuOracleResult> {
    verify_layer0_position1_continuation_anchors(reader)?;
    let position0_prefix = layer0_prefix_cpu_oracle(reader)?;
    let token1_prefix = layer0_prefix_cpu_oracle_for_token(reader, POSITION1_TOKEN_ID)?;
    let rope_table = layer0_position1_rope_table(reader)?;

    let kv_norm_weight = read_bf16_tensor(reader, LAYER0_KV_NORM_WEIGHT, HEAD_DIM)?;
    let position0_wkv = fp8_linear(
        reader,
        LAYER0_WKV_WEIGHT,
        LAYER0_WKV_SCALE,
        WKV_ROWS,
        4096,
        &position0_prefix.attn_norm_bf16_bits,
    )?;
    let position0_kv_norm_bf16_bits = rms_norm_bf16_source_algorithm(
        &position0_wkv.output.bf16_bits,
        &kv_norm_weight,
        HEAD_DIM,
        RMS_NORM_EPS,
    )?;
    let position0_kv_inplace_qat = kv_non_rope_inplace_qat_source_algorithm(
        &position0_kv_norm_bf16_bits,
        HEAD_DIM,
        ROPE_HEAD_DIM,
        KV_QAT_BLOCK,
    )?;
    let position0_kv_rope_bf16_bits = position_zero_rope_identity(
        &position0_kv_inplace_qat.output_bf16_bits,
        1,
        HEAD_DIM,
        ROPE_HEAD_DIM,
    )?;

    let wq_a = fp8_linear(
        reader,
        LAYER0_WQ_A_WEIGHT,
        LAYER0_WQ_A_SCALE,
        LAYER0_WQ_A_ROWS,
        LAYER0_WQ_A_COLS,
        &token1_prefix.attn_norm_bf16_bits,
    )?;
    let q_norm_weight = read_bf16_tensor(reader, LAYER0_Q_NORM_WEIGHT, Q_LORA_RANK)?;
    let q_norm_bf16_bits = rms_norm_bf16_source_algorithm(
        &wq_a.output.bf16_bits,
        &q_norm_weight,
        Q_LORA_RANK,
        RMS_NORM_EPS,
    )?;
    let wq_b = fp8_linear(
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
    let q_position1_rope_bf16_bits = rope_bf16_source_algorithm(
        &q_head_norm_bf16_bits,
        NUM_HEADS,
        HEAD_DIM,
        ROPE_HEAD_DIM,
        &rope_table,
        false,
    )?;

    let wkv = fp8_linear(
        reader,
        LAYER0_WKV_WEIGHT,
        LAYER0_WKV_SCALE,
        WKV_ROWS,
        4096,
        &token1_prefix.attn_norm_bf16_bits,
    )?;
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
    let kv_position1_rope_bf16_bits = rope_bf16_source_algorithm(
        &kv_inplace_qat.output_bf16_bits,
        1,
        HEAD_DIM,
        ROPE_HEAD_DIM,
        &rope_table,
        false,
    )?;
    let mut kv_cache_two_rows_bf16_bits = Vec::with_capacity(POSITION1_KV_ROWS * HEAD_DIM);
    kv_cache_two_rows_bf16_bits.extend_from_slice(&position0_kv_rope_bf16_bits);
    kv_cache_two_rows_bf16_bits.extend_from_slice(&kv_position1_rope_bf16_bits);
    let attn_sink = read_f32_tensor(reader, LAYER0_ATTN_SINK, NUM_HEADS)?;
    let (scores, denominators, sparse_attention_bf16_bits) =
        sparse_attention_position1_two_kv_source_algorithm(
            &q_position1_rope_bf16_bits,
            &kv_cache_two_rows_bf16_bits,
            &attn_sink,
        )?;
    let sparse_attention_derotated_bf16_bits = rope_bf16_source_algorithm(
        &sparse_attention_bf16_bits,
        NUM_HEADS,
        HEAD_DIM,
        ROPE_HEAD_DIM,
        &rope_table,
        true,
    )?;

    Ok(Layer0Position1ContinuationCpuOracleResult {
        token0_id: PREFIX_TOKEN_ID,
        token1_id: POSITION1_TOKEN_ID,
        token1_prefix,
        rope_table,
        position0_wkv,
        position0_kv_norm_bf16_bits,
        position0_kv_inplace_qat,
        position0_kv_rope_bf16_bits,
        wq_a,
        q_norm_bf16_bits,
        wq_b,
        q_head_norm_bf16_bits,
        q_position1_rope_bf16_bits,
        wkv,
        kv_norm_bf16_bits,
        kv_inplace_qat,
        kv_position1_rope_bf16_bits,
        kv_cache_two_rows_bf16_bits,
        sparse_attention_scores_f32: scores,
        sparse_attention_sink_denominators_f32: denominators,
        sparse_attention_bf16_bits,
        sparse_attention_derotated_bf16_bits,
    })
}

/// Continue the bounded position-one causal-KV proof through the source
/// attention tail.  This is intentionally a CPU source-algorithm authority,
/// not a model runtime or a claim of upstream PyTorch/TileLang execution.
pub fn layer0_position1_complete_attention_cpu_oracle(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<Layer0Position1CompleteAttentionCpuOracleResult> {
    let causal = layer0_position1_continuation_cpu_oracle(reader)?;
    let wo_a_bf16_bits =
        wo_a_bf16_einsum_source_algorithm(reader, &causal.sparse_attention_derotated_bf16_bits)?;
    let wo_b = fp8_linear(
        reader,
        LAYER0_WO_B_WEIGHT,
        LAYER0_WO_B_SCALE,
        WO_B_ROWS,
        WO_B_COLS,
        &wo_a_bf16_bits,
    )?;
    let hc_attention_post_bf16_bits = hc_attn_post_source_algorithm(
        &wo_b.output.bf16_bits,
        &causal.token1_prefix.hc_replicated_bf16_bits,
        &causal.token1_prefix.hc_post_f32,
        &causal.token1_prefix.hc_comb_f32,
    )?;
    Ok(Layer0Position1CompleteAttentionCpuOracleResult {
        causal,
        wo_a_bf16_bits,
        wo_b,
        hc_attention_post_bf16_bits,
    })
}

/// Source position-one rotary application, using the exact 32 complex pairs
/// selected by the pinned YaRN configuration.  All non-RoPE dimensions are
/// copied unchanged and the final `copy_` is represented by a BF16 store.
pub fn rope_bf16_source_algorithm(
    input_bf16_bits: &[u16],
    rows: usize,
    head_dim: usize,
    rope_head_dim: usize,
    table: &PositionOneRopeTable,
    inverse: bool,
) -> Result<Vec<u16>> {
    if rows == 0
        || head_dim == 0
        || rope_head_dim == 0
        || rope_head_dim > head_dim
        || rope_head_dim % 2 != 0
        || input_bf16_bits.len() != rows * head_dim
        || table.cos_f32.len() != rope_head_dim / 2
        || table.sin_f32.len() != rope_head_dim / 2
        || table
            .cos_f32
            .iter()
            .chain(&table.sin_f32)
            .any(|value| !value.is_finite())
    {
        return Err(continuation("position-one RoPE geometry/table is invalid"));
    }
    let mut output = input_bf16_bits.to_vec();
    let rope_start = head_dim - rope_head_dim;
    for row in 0..rows {
        let row_start = row * head_dim + rope_start;
        for pair in 0..rope_head_dim / 2 {
            let left = bf16::from_bits(input_bf16_bits[row_start + pair * 2]).to_f32();
            let right = bf16::from_bits(input_bf16_bits[row_start + pair * 2 + 1]).to_f32();
            let cos = table.cos_f32[pair];
            let sin = if inverse {
                -table.sin_f32[pair]
            } else {
                table.sin_f32[pair]
            };
            if !left.is_finite() || !right.is_finite() {
                return Err(continuation("position-one RoPE input is non-finite"));
            }
            let out_left = left * cos - right * sin;
            let out_right = left * sin + right * cos;
            if !out_left.is_finite() || !out_right.is_finite() {
                return Err(continuation("position-one RoPE result is non-finite"));
            }
            output[row_start + pair * 2] = bf16::from_f32(out_left).to_bits();
            output[row_start + pair * 2 + 1] = bf16::from_f32(out_right).to_bits();
        }
    }
    Ok(output)
}

/// Exact ratio-zero position-one causal sparse-attention specialization for
/// `topk_idxs=[0, 1, -1, ...]`.  The source TileLang kernel casts the two
/// online-softmax numerator weights to BF16 before its value GEMM; that store
/// boundary is intentionally represented below.
pub fn sparse_attention_position1_two_kv_source_algorithm(
    q_bf16_bits: &[u16],
    kv_cache_two_rows_bf16_bits: &[u16],
    attn_sink: &[f32],
) -> Result<(Vec<f32>, Vec<f32>, Vec<u16>)> {
    if q_bf16_bits.len() != WQ_B_ROWS
        || kv_cache_two_rows_bf16_bits.len() != POSITION1_KV_ROWS * HEAD_DIM
        || attn_sink.len() != NUM_HEADS
        || attn_sink.iter().any(|value| !value.is_finite())
    {
        return Err(continuation(
            "position-one sparse attention geometry is invalid",
        ));
    }
    let scale = (HEAD_DIM as f32).powf(-0.5);
    let mut scores = Vec::with_capacity(NUM_HEADS * POSITION1_KV_ROWS);
    let mut denominators = Vec::with_capacity(NUM_HEADS);
    let mut output = Vec::with_capacity(WQ_B_ROWS);
    for head in 0..NUM_HEADS {
        let mut score = [0.0f32; POSITION1_KV_ROWS];
        for kv_row in 0..POSITION1_KV_ROWS {
            let mut dot = 0.0f32;
            for dim in 0..HEAD_DIM {
                let q = bf16::from_bits(q_bf16_bits[head * HEAD_DIM + dim]).to_f32();
                let kv =
                    bf16::from_bits(kv_cache_two_rows_bf16_bits[kv_row * HEAD_DIM + dim]).to_f32();
                if !q.is_finite() || !kv.is_finite() {
                    return Err(continuation("position-one sparse Q/KV is non-finite"));
                }
                dot += q * kv;
            }
            score[kv_row] = dot * scale;
            if !score[kv_row].is_finite() {
                return Err(continuation("position-one sparse score is non-finite"));
            }
        }
        let score_max = score[0].max(score[1]);
        let numerator0 = (score[0] - score_max).exp();
        let numerator1 = (score[1] - score_max).exp();
        let denominator = numerator0 + numerator1 + (attn_sink[head] - score_max).exp();
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(continuation("position-one sparse denominator is invalid"));
        }
        scores.extend_from_slice(&score);
        denominators.push(denominator);
        // `T.copy(acc_s, acc_s_cast)` in the source makes these BF16 weights
        // for the value GEMM, while the denominator stays FP32.
        let numerator0_bf16 = bf16::from_f32(numerator0).to_f32();
        let numerator1_bf16 = bf16::from_f32(numerator1).to_f32();
        for dim in 0..HEAD_DIM {
            let value0 = bf16::from_bits(kv_cache_two_rows_bf16_bits[dim]).to_f32();
            let value1 = bf16::from_bits(kv_cache_two_rows_bf16_bits[HEAD_DIM + dim]).to_f32();
            let value = (numerator0_bf16 * value0 + numerator1_bf16 * value1) / denominator;
            if !value.is_finite() {
                return Err(continuation("position-one sparse output is non-finite"));
            }
            output.push(bf16::from_f32(value).to_bits());
        }
    }
    Ok((scores, denominators, output))
}

/// Derive the bounded position-one YaRN table directly from the admitted
/// source configuration.  The table is static source control, not a hidden
/// activation, so a caller-owned device graph may upload it without creating
/// a host activation bridge.
pub fn layer0_position1_rope_table(
    reader: &DeepSeekV4FullStreamReader,
) -> Result<PositionOneRopeTable> {
    verify_layer0_position1_continuation_anchors(reader)?;
    let low = find_correction_dim(ROPE_BETA_FAST).floor().max(0.0) as usize;
    let high = find_correction_dim(ROPE_BETA_SLOW)
        .ceil()
        .min((ROPE_HEAD_DIM - 1) as f64) as usize;
    let mut cos_f32 = Vec::with_capacity(ROPE_HEAD_DIM / 2);
    let mut sin_f32 = Vec::with_capacity(ROPE_HEAD_DIM / 2);
    for index in 0..ROPE_HEAD_DIM / 2 {
        let exponent = (index * 2) as f32 / ROPE_HEAD_DIM as f32;
        let base_frequency = 1.0 / ROPE_THETA.powf(exponent);
        let ramp = ((index as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0);
        let smooth = 1.0 - ramp;
        let frequency = base_frequency / ROPE_FACTOR * (1.0 - smooth) + base_frequency * smooth;
        let (sin, cos) = (POSITION1 as f32 * frequency).sin_cos();
        cos_f32.push(cos);
        sin_f32.push(sin);
    }
    Ok(PositionOneRopeTable { cos_f32, sin_f32 })
}

fn find_correction_dim(num_rotations: f32) -> f64 {
    ROPE_HEAD_DIM as f64
        * ((ROPE_ORIGINAL_SEQ_LEN as f64 / (num_rotations as f64 * 2.0 * std::f64::consts::PI))
            .ln())
        / (2.0 * (ROPE_THETA as f64).ln())
}

fn fp8_linear(
    reader: &DeepSeekV4FullStreamReader,
    weight_name: &str,
    scale_name: &str,
    rows: usize,
    cols: usize,
    input_bf16_bits: &[u16],
) -> Result<ContinuationFp8LinearCpuStage> {
    let pair = reader.native_scale_pair(weight_name)?;
    if pair.kind != NativeScalePairKind::Fp8E4M3fn
        || pair.scale.name != scale_name
        || pair.weight.shape.as_slice() != [rows as u64, cols as u64]
        || pair.scale.shape.as_slice()
            != [
                (rows / ACT_QUANT_BLOCK) as u64,
                (cols / ACT_QUANT_BLOCK) as u64,
            ]
        || input_bf16_bits.len() != cols
    {
        return Err(continuation(format!(
            "{weight_name} FP8 source geometry differs"
        )));
    }
    let quantized_input = act_quant_bf16_ue8m0(input_bf16_bits)?;
    let weight = reader.read_verified_full(weight_name, rows * cols)?;
    let scale = reader.read_verified_full(
        scale_name,
        (rows / ACT_QUANT_BLOCK) * (cols / ACT_QUANT_BLOCK),
    )?;
    let output = fp8_e4m3fn_ue8m0_matvec(&quantized_input, &weight, &scale, rows, cols)?;
    Ok(ContinuationFp8LinearCpuStage {
        quantized_input,
        output,
    })
}

fn read_bf16_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    width: usize,
) -> Result<Vec<u16>> {
    let metadata = reader.tensor_metadata(name)?;
    if metadata.dtype != "BF16"
        || metadata.shape.as_slice() != [width as u64]
        || metadata.bytes != (width * 2) as u64
    {
        return Err(continuation(format!("{name} is not BF16[{width}]")));
    }
    let bytes = reader.read_verified_full(name, width * 2)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn read_f32_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    width: usize,
) -> Result<Vec<f32>> {
    let metadata = reader.tensor_metadata(name)?;
    if metadata.dtype != "F32"
        || metadata.shape.as_slice() != [width as u64]
        || metadata.bytes != (width * 4) as u64
    {
        return Err(continuation(format!("{name} is not F32[{width}]")));
    }
    let bytes = reader.read_verified_full(name, width * 4)?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_bits(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(continuation(format!("{name} contains non-finite F32")));
    }
    Ok(values)
}

fn json_u64(config: &Value, key: &str) -> Result<u64> {
    config
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| continuation(format!("inference config {key} is not a u64")))
}

fn json_f32_eq(config: &Value, key: &str, expected: f32) -> bool {
    config
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| (value as f32 - expected).abs() <= 1.0e-7)
        .unwrap_or(false)
}

fn json_array_first_u64(config: &Value, key: &str) -> Result<u64> {
    config
        .get(key)
        .and_then(Value::as_array)
        .and_then(|values| values.first())
        .and_then(Value::as_u64)
        .ok_or_else(|| continuation(format!("inference config {key}[0] is not a u64")))
}

fn continuation(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 layer-0 continuation: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_one_rope_rotates_and_inverse_restores_bounded_pairs() {
        let table = PositionOneRopeTable {
            cos_f32: vec![0.0],
            sin_f32: vec![1.0],
        };
        let input = vec![bf16::from_f32(1.0).to_bits(), bf16::from_f32(2.0).to_bits()];
        let rotated = rope_bf16_source_algorithm(&input, 1, 2, 2, &table, false).unwrap();
        assert_eq!(
            rotated,
            vec![
                bf16::from_f32(-2.0).to_bits(),
                bf16::from_f32(1.0).to_bits()
            ]
        );
        assert_eq!(
            rope_bf16_source_algorithm(&rotated, 1, 2, 2, &table, true).unwrap(),
            input
        );
    }
}
