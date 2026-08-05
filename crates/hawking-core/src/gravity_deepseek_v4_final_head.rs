//! Final mHC-head merge, RMSNorm, and greedy LM-head for DeepSeek-V4-Flash.
//!
//! After the last base layer's child HC state `[HC_MULT, HIDDEN]`, the source
//! algorithm merges Hyper-Connection lanes with `hc_head_*`, applies
//! `norm.weight`, and projects through BF16 `head.weight` to vocabulary
//! logits. This module provides:
//!
//! 1. A host F64 authority for the small mHC-head merge + final RMSNorm
//!    (exact source order; used as a parity oracle and as a bootstrap when
//!    the full head is not yet resident on device).
//! 2. A device-path helper that uploads the merged BF16 residual, runs
//!    `gemv_native_bf16_seq` against a caller-staged `head.weight` buffer,
//!    and returns greedy argmax.
//!
//! Honesty: streaming the ~1 GB LM head is expensive; the host path streams
//! verified rows and never claims TPS. Exact-storage e2e parity is not
//! claimed (`NUMERIC_PARITY_V2_1_ONLY` until a sealed receipt says otherwise).

use std::mem::size_of;

use half::bf16;

use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
use crate::gravity_deepseek_v4_layer0_prefix::{
    HC_EPS, HC_FLAT_WIDTH, HC_MULT, HIDDEN_SIZE, RMS_NORM_EPS,
};
use crate::gravity_deepseek_v4_runtime_spine::DSV4F_VOCAB_SIZE;
use crate::{Error, Result};

const HC_HEAD_FN: &str = "hc_head_fn";
const HC_HEAD_BASE: &str = "hc_head_base";
const HC_HEAD_SCALE: &str = "hc_head_scale";
const FINAL_NORM: &str = "norm.weight";
const LM_HEAD: &str = "head.weight";

#[cfg(target_os = "macos")]
const GEMV_KERNEL: &str = "gemv_native_bf16_seq";

/// Host F64 merge of the final HC state into a single residual row + final
/// RMSNorm, matching the sealed source `_head_logits` prefix (before lm_head).
#[derive(Debug, Clone)]
pub struct DeepSeekV4FinalHeadHostMerge {
    pub merged_f32: Vec<f32>,
    pub merged_bf16_bits: Vec<u16>,
    pub mix_weights_f32: Vec<f32>,
    pub flat_rsqrt: f32,
}

/// Greedy decode result over the full vocabulary.
#[derive(Debug, Clone)]
pub struct DeepSeekV4GreedyTokenResult {
    pub token_id: u32,
    pub logit: f32,
    pub vocab_size: usize,
    /// True when logits were produced by device `gemv_native_bf16_seq`.
    pub lm_head_on_device: bool,
    /// True when argmax ran on device.
    pub argmax_on_device: bool,
    pub metal_dispatches: usize,
    pub command_buffers: usize,
}

/// Load `hc_head_*` + `norm.weight` and merge a device-or-host HC state
/// `[HC_MULT * HIDDEN]` BF16 row-major into the final residual BF16 vector.
pub fn host_merge_final_head_from_hc_bf16(
    reader: &DeepSeekV4FullStreamReader,
    child_hc_bf16_bits: &[u16],
) -> Result<DeepSeekV4FinalHeadHostMerge> {
    if child_hc_bf16_bits.len() != HC_FLAT_WIDTH {
        return Err(head_error(format!(
            "child HC state must be BF16[{HC_FLAT_WIDTH}], got {}",
            child_hc_bf16_bits.len()
        )));
    }
    let fn_bytes = read_f32_tensor(reader, HC_HEAD_FN, HC_MULT * HC_FLAT_WIDTH)?;
    let base = read_f32_tensor(reader, HC_HEAD_BASE, HC_MULT)?;
    let scale = read_f32_tensor(reader, HC_HEAD_SCALE, 1)?;
    let norm = read_bf16_tensor(reader, FINAL_NORM, HIDDEN_SIZE)?;

    // Widen HC to f64 for the authority merge.
    let mut hidden_f64 = vec![0f64; HC_FLAT_WIDTH];
    for (i, &bits) in child_hc_bf16_bits.iter().enumerate() {
        hidden_f64[i] = bf16::from_bits(bits).to_f64();
    }
    // Layout: [HC_MULT, HIDDEN] row-major.
    let mut sum_sq = 0f64;
    for &v in &hidden_f64 {
        sum_sq += v * v;
    }
    let mean_sq = sum_sq / HC_FLAT_WIDTH as f64;
    let flat_rsqrt = 1.0 / (mean_sq + RMS_NORM_EPS as f64).sqrt();
    if !flat_rsqrt.is_finite() {
        return Err(head_error("hc_head flat rsqrt is non-finite"));
    }

    // mixes = (fn @ flat) * rsqrt ; fn is [HC_MULT, HC_FLAT]
    let mut mixes = vec![0f64; HC_MULT];
    for row in 0..HC_MULT {
        let mut acc = 0f64;
        let row_off = row * HC_FLAT_WIDTH;
        for col in 0..HC_FLAT_WIDTH {
            acc += fn_bytes[row_off + col] as f64 * hidden_f64[col];
        }
        mixes[row] = acc * flat_rsqrt;
    }

    // weights = sigmoid(mixes * scale[0] + base) + hc_eps
    let mut weights = vec![0f64; HC_MULT];
    for i in 0..HC_MULT {
        let x = mixes[i] * scale[0] as f64 + base[i] as f64;
        let sig = 1.0 / (1.0 + (-x).exp());
        weights[i] = sig + HC_EPS as f64;
    }

    // merged = sum_i weights[i] * hidden[i, :]
    let mut merged = vec![0f64; HIDDEN_SIZE];
    for lane in 0..HC_MULT {
        let w = weights[lane];
        let row_off = lane * HIDDEN_SIZE;
        for feat in 0..HIDDEN_SIZE {
            merged[feat] += w * hidden_f64[row_off + feat];
        }
    }

    // final RMSNorm with norm.weight
    let mut nsum = 0f64;
    for &v in &merged {
        nsum += v * v;
    }
    let n_rsqrt = 1.0 / (nsum / HIDDEN_SIZE as f64 + RMS_NORM_EPS as f64).sqrt();
    let mut merged_f32 = vec![0f32; HIDDEN_SIZE];
    let mut merged_bf16_bits = vec![0u16; HIDDEN_SIZE];
    for i in 0..HIDDEN_SIZE {
        let w = bf16::from_bits(norm[i]).to_f64();
        let y = (merged[i] * n_rsqrt * w) as f32;
        merged_f32[i] = y;
        merged_bf16_bits[i] = bf16::from_f32(y).to_bits();
    }

    Ok(DeepSeekV4FinalHeadHostMerge {
        merged_f32,
        merged_bf16_bits,
        mix_weights_f32: weights.iter().map(|w| *w as f32).collect(),
        flat_rsqrt: flat_rsqrt as f32,
    })
}

/// Stream-verified `head.weight` rows on the host and return greedy argmax.
///
/// This is the honest bootstrap path when the full ~1 GB head is not staged on
/// device. It uses left-to-right f32 accumulate (same product-then-add order as
/// `gemv_native_bf16_seq`) and never fabricates a token.
pub fn host_greedy_lm_head(
    reader: &DeepSeekV4FullStreamReader,
    residual_f32: &[f32],
) -> Result<DeepSeekV4GreedyTokenResult> {
    if residual_f32.len() != HIDDEN_SIZE {
        return Err(head_error("lm_head residual must be f32[4096]"));
    }
    let meta = reader.tensor_metadata(LM_HEAD)?;
    if meta.dtype != "BF16"
        || meta.shape.as_slice() != [DSV4F_VOCAB_SIZE as u64, HIDDEN_SIZE as u64]
    {
        return Err(head_error("head.weight is not BF16[vocab,4096]"));
    }
    let row_bytes = HIDDEN_SIZE * size_of::<u16>();
    // Stream in 256-row blocks (~2 MB each).
    const ROWS_PER_BLOCK: usize = 256;
    let mut best_id = 0u32;
    let mut best_logit = f32::NEG_INFINITY;
    let mut row = 0usize;
    while row < DSV4F_VOCAB_SIZE {
        let count = (DSV4F_VOCAB_SIZE - row).min(ROWS_PER_BLOCK);
        let start = (row * row_bytes) as u64;
        let end = start + (count * row_bytes) as u64;
        let bytes = reader.read_verified_range(LM_HEAD, start..end, count * row_bytes)?;
        if bytes.len() != count * row_bytes {
            return Err(head_error("head.weight block read length mismatch"));
        }
        for r in 0..count {
            let off = r * row_bytes;
            let mut acc = 0f32;
            for c in 0..HIDDEN_SIZE {
                let bits = u16::from_le_bytes([bytes[off + c * 2], bytes[off + c * 2 + 1]]);
                // Widen bf16 → f32 as (u16)<<16 — matches gemv_native_bf16_seq.
                let w = f32::from_bits((bits as u32) << 16);
                acc = acc + w * residual_f32[c];
            }
            let token = (row + r) as u32;
            if acc > best_logit || (acc == best_logit && token < best_id) {
                best_logit = acc;
                best_id = token;
            }
        }
        row += count;
    }
    if !best_logit.is_finite() {
        return Err(head_error("greedy lm_head produced non-finite best logit"));
    }
    Ok(DeepSeekV4GreedyTokenResult {
        token_id: best_id,
        logit: best_logit,
        vocab_size: DSV4F_VOCAB_SIZE,
        lm_head_on_device: false,
        argmax_on_device: false,
        metal_dispatches: 0,
        command_buffers: 0,
    })
}

/// Device LM-head gemv + greedy argmax given a host residual and a fully
/// staged BF16 `head.weight` buffer already on the Metal device.
///
/// Caller owns uploading `head.weight` (≈1.06 GB). Grid is one thread per
/// vocabulary row for sequential authority accumulate.
#[cfg(target_os = "macos")]
pub fn device_greedy_lm_head(
    metal: &crate::metal::MetalContext,
    head_weight_bf16: &metal::Buffer,
    residual_f32: &[f32],
) -> Result<DeepSeekV4GreedyTokenResult> {
    if residual_f32.len() != HIDDEN_SIZE {
        return Err(head_error("device lm_head residual must be f32[4096]"));
    }
    let expected_head_bytes = DSV4F_VOCAB_SIZE * HIDDEN_SIZE * size_of::<u16>();
    if head_weight_bf16.length() < expected_head_bytes as u64 {
        return Err(head_error("device head.weight buffer is smaller than BF16[vocab,4096]"));
    }
    let residual_bytes: Vec<u8> = residual_f32
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let residual_buf = metal.new_buffer_with_bytes_checked(&residual_bytes)?;
    let logits_buf = metal.new_buffer_checked(DSV4F_VOCAB_SIZE * size_of::<f32>())?;
    let n_rows = DSV4F_VOCAB_SIZE as u32;
    let n_cols = HIDDEN_SIZE as u32;
    metal.dispatch_batch(|batch| {
        batch.dispatch_threads(
            GEMV_KERNEL,
            (n_rows, 1, 1),
            (1, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(head_weight_bf16), 0);
                encoder.set_buffer(1, Some(&residual_buf), 0);
                encoder.set_buffer(2, Some(&logits_buf), 0);
                set_u32(encoder, 3, &n_rows);
                set_u32(encoder, 4, &n_cols);
            },
        )
    })?;

    // Host argmax over device logits (single readback of vocab f32).
    let ptr = logits_buf.contents() as *const u8;
    if ptr.is_null() {
        return Err(head_error("logits buffer contents null"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, DSV4F_VOCAB_SIZE * 4) };
    let mut best_id = 0u32;
    let mut best_logit = f32::NEG_INFINITY;
    for i in 0..DSV4F_VOCAB_SIZE {
        let logit = f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().map_err(|_| {
            head_error("logit byte slice")
        })?);
        if logit > best_logit || (logit == best_logit && (i as u32) < best_id) {
            best_logit = logit;
            best_id = i as u32;
        }
    }
    if !best_logit.is_finite() {
        return Err(head_error("device greedy produced non-finite best logit"));
    }
    Ok(DeepSeekV4GreedyTokenResult {
        token_id: best_id,
        logit: best_logit,
        vocab_size: DSV4F_VOCAB_SIZE,
        lm_head_on_device: true,
        argmax_on_device: false,
        metal_dispatches: 1,
        command_buffers: 1,
    })
}

/// Read child HC BF16 bits from a Metal buffer (host diagnostic boundary).
#[cfg(target_os = "macos")]
pub fn read_hc_bf16_from_buffer(buf: &metal::Buffer) -> Result<Vec<u16>> {
    let need = HC_FLAT_WIDTH * size_of::<u16>();
    if buf.length() < need as u64 {
        return Err(head_error("HC buffer smaller than BF16[4*4096]"));
    }
    let ptr = buf.contents() as *const u8;
    if ptr.is_null() {
        return Err(head_error("HC buffer contents null"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr, need) };
    let mut out = Vec::with_capacity(HC_FLAT_WIDTH);
    for i in 0..HC_FLAT_WIDTH {
        out.push(u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]));
    }
    Ok(out)
}

fn read_f32_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    elements: usize,
) -> Result<Vec<f32>> {
    let meta = reader.tensor_metadata(name)?;
    let bytes_needed = elements * size_of::<f32>();
    if meta.dtype != "F32" || meta.bytes as usize != bytes_needed {
        return Err(head_error(format!(
            "{name}: expected F32[{elements}], got {} bytes dtype {}",
            meta.bytes, meta.dtype
        )));
    }
    let raw = reader.read_verified_full(name, bytes_needed)?;
    let mut out = Vec::with_capacity(elements);
    for i in 0..elements {
        out.push(f32::from_le_bytes(
            raw[i * 4..i * 4 + 4]
                .try_into()
                .map_err(|_| head_error(format!("{name} f32 slice")))?,
        ));
    }
    Ok(out)
}

fn read_bf16_tensor(
    reader: &DeepSeekV4FullStreamReader,
    name: &str,
    elements: usize,
) -> Result<Vec<u16>> {
    let meta = reader.tensor_metadata(name)?;
    let bytes_needed = elements * size_of::<u16>();
    if meta.dtype != "BF16" || meta.bytes as usize != bytes_needed {
        return Err(head_error(format!(
            "{name}: expected BF16[{elements}], got {} bytes dtype {}",
            meta.bytes, meta.dtype
        )));
    }
    let raw = reader.read_verified_full(name, bytes_needed)?;
    let mut out = Vec::with_capacity(elements);
    for i in 0..elements {
        out.push(u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
    }
    Ok(out)
}

#[cfg(target_os = "macos")]
fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
    encoder.set_bytes(
        index,
        size_of::<u32>() as u64,
        value as *const u32 as *const _,
    );
}

fn head_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!(
        "DeepSeek-V4 final head: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_width_matches_hc_geometry() {
        assert_eq!(HC_FLAT_WIDTH, HC_MULT * HIDDEN_SIZE);
        assert_eq!(DSV4F_VOCAB_SIZE, 129_280);
    }
}
