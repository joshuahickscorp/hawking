//! Opt-in Metal operators for the streamed DeepSeek-V4-Flash BOS decode.
//!
//! Default-off. Each operator uploads one working set, dispatches, reads
//! back, and drops the buffers so the streaming residency policy is
//! unchanged. `metal_dispatches` and `fallbacks` are real counters.

use std::mem::size_of;

use half::bf16;

use crate::gravity_deepseek_v4_act_quant::{
    act_quant_bf16_ue8m0, decode_e4m3fn, decode_e8m0fnu, fp8_e4m3fn_ue8m0_matvec, ACT_QUANT_BLOCK,
};
use crate::gravity_deepseek_v4_final_head::DeepSeekV4GreedyTokenResult;
use crate::gravity_deepseek_v4_layer0_attention::{O_GROUPS, O_LORA_RANK, WO_A_COLS, WO_A_ROWS};
use crate::gravity_deepseek_v4_layer0_moe::{fp4_e2m1fn_x2_ue8m0_matvec, FP4_BLOCK};
use crate::gravity_deepseek_v4_layer0_prefix::HIDDEN_SIZE;
use crate::gravity_deepseek_v4_runtime_spine::DSV4F_VOCAB_SIZE;
use crate::{Error, Result};

const ACT_QUANT_KERNEL: &str = "deepseek_v4_act_quant_bf16_ue8m0_authority";
const FP8_KERNEL: &str = "deepseek_v4_fp8_act_quant_e4m3fn_e8m0_matvec_authority";
const FP4_KERNEL: &str = "deepseek_v4_p5b_fp4_act_quant_e2m1fn_x2_e8m0_matvec_authority";
const WO_A_KERNEL: &str = "deepseek_v4_p4a_wo_a_convert_bf16_einsum_authority";
const CAST_KERNEL: &str = "deepseek_v4_p3a_fp32_to_bf16_authority";
const GATE_KERNEL: &str = "deepseek_v4_p6a_gate_bf16_matvec_authority";
const LM_HEAD_KERNEL: &str = "gemv_native_bf16_seq";

/// Stated per-element tolerance for Metal vs CPU linear outputs (decoded BF16).
pub const LINEAR_ABS_TOL: f32 = 1.0e-3;
pub const LINEAR_REL_TOL: f32 = 1.0e-3;
/// Stated end-to-end greedy logit tolerance against the host oracle.
pub const GREEDY_LOGIT_ABS_TOL: f32 = 5.0e-2;
/// Host-oracle greedy token from the sealed BOS streamed receipt.
pub const ORACLE_GREEDY_TOKEN_ID: u32 = 5;
/// Host-oracle greedy logit from `receipts/dsv4f_streamed_forward_l0_l42_receipt.json`.
pub const ORACLE_GREEDY_LOGIT: f32 = 16.767_437;

/// Streamed Metal session. One per decode. Holds the compiled library and
/// live dispatch/fallback counters.
pub struct StreamedNativeSession {
    #[cfg(target_os = "macos")]
    metal: crate::metal::MetalContext,
    #[cfg(target_os = "macos")]
    act_quant_tg: u32,
    #[cfg(target_os = "macos")]
    fp8_tg: u32,
    #[cfg(target_os = "macos")]
    fp4_tg: u32,
    #[cfg(target_os = "macos")]
    wo_a_tg: u32,
    #[cfg(target_os = "macos")]
    cast_tg: u32,
    #[cfg(target_os = "macos")]
    gate_tg: u32,
    #[cfg(target_os = "macos")]
    lm_head_tg: u32,
    dispatches: usize,
    fallbacks: usize,
    fallback_reasons: Vec<String>,
}

impl StreamedNativeSession {
    pub fn new() -> Result<Self> {
        #[cfg(not(target_os = "macos"))]
        {
            Err(native_error(
                "streamed Metal operators require macOS Metal; CPU oracle remains the default",
            ))
        }
        #[cfg(target_os = "macos")]
        {
            let metal = crate::metal::MetalContext::new()?;
            let act_quant_tg = pipeline_tg(&metal, ACT_QUANT_KERNEL, 32)?;
            let fp8_tg = pipeline_tg(&metal, FP8_KERNEL, 256)?;
            let fp4_tg = pipeline_tg(&metal, FP4_KERNEL, 256)?;
            let wo_a_tg = pipeline_tg(&metal, WO_A_KERNEL, 256)?;
            let cast_tg = pipeline_tg(&metal, CAST_KERNEL, 256)?;
            let gate_tg = pipeline_tg(&metal, GATE_KERNEL, 256)?;
            let lm_head_tg = pipeline_tg(&metal, LM_HEAD_KERNEL, 256)?;
            Ok(Self {
                metal,
                act_quant_tg,
                fp8_tg,
                fp4_tg,
                wo_a_tg,
                cast_tg,
                gate_tg,
                lm_head_tg,
                dispatches: 0,
                fallbacks: 0,
                fallback_reasons: Vec::new(),
            })
        }
    }

    pub fn metal_dispatches(&self) -> usize {
        self.dispatches
    }

    pub fn fallbacks(&self) -> usize {
        self.fallbacks
    }

    pub fn fallback_reasons(&self) -> &[String] {
        &self.fallback_reasons
    }

    pub fn record_fallback(&mut self, reason: impl Into<String>) {
        self.fallbacks += 1;
        self.fallback_reasons.push(reason.into());
    }

    pub fn fp8_linear(
        &mut self,
        input_bf16: &[u16],
        weights: &[u8],
        scales: &[u8],
        rows: usize,
        logical_k: usize,
    ) -> Result<Vec<u16>> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (input_bf16, weights, scales, rows, logical_k);
            Err(native_error("fp8_linear requires macOS Metal"))
        }
        #[cfg(target_os = "macos")]
        {
            self.fp8_linear_macos(input_bf16, weights, scales, rows, logical_k)
        }
    }

    pub fn fp4_linear(
        &mut self,
        input_bf16: &[u16],
        weights: &[u8],
        scales: &[u8],
        rows: usize,
        logical_k: usize,
    ) -> Result<Vec<u16>> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (input_bf16, weights, scales, rows, logical_k);
            Err(native_error("fp4_linear requires macOS Metal"))
        }
        #[cfg(target_os = "macos")]
        {
            self.fp4_linear_macos(input_bf16, weights, scales, rows, logical_k)
        }
    }

    pub fn wo_a_einsum(
        &mut self,
        attention_bf16: &[u16],
        weights: &[u8],
        scales: &[u8],
    ) -> Result<Vec<u16>> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (attention_bf16, weights, scales);
            Err(native_error("wo_a_einsum requires macOS Metal"))
        }
        #[cfg(target_os = "macos")]
        {
            self.wo_a_einsum_macos(attention_bf16, weights, scales)
        }
    }

    pub fn gate_logits(&mut self, input_bf16: &[u16], weight_bf16: &[u16]) -> Result<Vec<f32>> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (input_bf16, weight_bf16);
            Err(native_error("gate_logits requires macOS Metal"))
        }
        #[cfg(target_os = "macos")]
        {
            self.gate_logits_macos(input_bf16, weight_bf16)
        }
    }

    pub fn lm_head_block(
        &mut self,
        residual_f32: &[f32],
        weight_bf16_bytes: &[u8],
        rows: usize,
    ) -> Result<Vec<f32>> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (residual_f32, weight_bf16_bytes, rows);
            Err(native_error("lm_head_block requires macOS Metal"))
        }
        #[cfg(target_os = "macos")]
        {
            self.lm_head_block_macos(residual_f32, weight_bf16_bytes, rows)
        }
    }

    #[cfg(target_os = "macos")]
    fn fp8_linear_macos(
        &mut self,
        input_bf16: &[u16],
        weights: &[u8],
        scales: &[u8],
        rows: usize,
        logical_k: usize,
    ) -> Result<Vec<u16>> {
        if input_bf16.len() != logical_k
            || logical_k == 0
            || rows == 0
            || logical_k % ACT_QUANT_BLOCK != 0
            || rows % ACT_QUANT_BLOCK != 0
        {
            return Err(native_error("fp8_linear geometry is invalid"));
        }
        if weights.len() != rows * logical_k {
            return Err(native_error("fp8_linear weight byte count mismatch"));
        }
        let scale_cols = logical_k / ACT_QUANT_BLOCK;
        if scales.len() != (rows / ACT_QUANT_BLOCK) * scale_cols {
            return Err(native_error("fp8_linear scale byte count mismatch"));
        }

        let input_bytes = u16_as_bytes(input_bf16);
        let input_buf = self.metal.new_buffer_with_bytes_checked(input_bytes)?;
        let quant_buf = self.metal.new_buffer_checked(logical_k)?;
        let act_scale_buf = self.metal.new_buffer_checked(scale_cols)?;
        let weight_buf = self.metal.new_buffer_from_verified_bytes(weights)?;
        let scale_buf = self.metal.new_buffer_from_verified_bytes(scales)?;
        let out_f32_buf = self.metal.new_buffer_checked(rows * size_of::<f32>())?;
        let out_bf16_buf = self.metal.new_buffer_checked(rows * size_of::<u16>())?;

        let cols_u = logical_k as u32;
        let rows_u = rows as u32;
        let scale_cols_u = scale_cols as u32;
        let blocks = (logical_k / ACT_QUANT_BLOCK) as u32;
        let aq_tg = self.act_quant_tg.min(blocks.max(1));
        let fp8_tg = self.fp8_tg.min(rows_u.max(1));
        let cast_tg = self.cast_tg.min(rows_u.max(1));

        self.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(ACT_QUANT_KERNEL, (blocks, 1, 1), (aq_tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&input_buf), 0);
                encoder.set_buffer(1, Some(&quant_buf), 0);
                encoder.set_buffer(2, Some(&act_scale_buf), 0);
                set_u32(encoder, 3, &cols_u);
            })?;
            batch.dispatch_threads(FP8_KERNEL, (rows_u, 1, 1), (fp8_tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&weight_buf), 0);
                encoder.set_buffer(1, Some(&scale_buf), 0);
                encoder.set_buffer(2, Some(&quant_buf), 0);
                encoder.set_buffer(3, Some(&act_scale_buf), 0);
                encoder.set_buffer(4, Some(&out_f32_buf), 0);
                set_u32(encoder, 5, &rows_u);
                set_u32(encoder, 6, &cols_u);
                set_u32(encoder, 7, &scale_cols_u);
            })?;
            batch.dispatch_threads(CAST_KERNEL, (rows_u, 1, 1), (cast_tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&out_f32_buf), 0);
                encoder.set_buffer(1, Some(&out_bf16_buf), 0);
                set_u32(encoder, 2, &rows_u);
            })
        })?;
        self.dispatches += 3;
        read_u16_buffer(&out_bf16_buf, rows)
    }

    #[cfg(target_os = "macos")]
    fn fp4_linear_macos(
        &mut self,
        input_bf16: &[u16],
        weights: &[u8],
        scales: &[u8],
        rows: usize,
        logical_k: usize,
    ) -> Result<Vec<u16>> {
        if input_bf16.len() != logical_k
            || logical_k == 0
            || rows == 0
            || logical_k % ACT_QUANT_BLOCK != 0
            || logical_k % FP4_BLOCK != 0
        {
            return Err(native_error("fp4_linear geometry is invalid"));
        }
        let packed_k = logical_k / 2;
        let scale_cols = logical_k / FP4_BLOCK;
        if weights.len() != rows * packed_k {
            return Err(native_error("fp4_linear packed weight byte count mismatch"));
        }
        if scales.len() != rows * scale_cols {
            return Err(native_error("fp4_linear scale byte count mismatch"));
        }

        let input_bytes = u16_as_bytes(input_bf16);
        let input_buf = self.metal.new_buffer_with_bytes_checked(input_bytes)?;
        let quant_buf = self.metal.new_buffer_checked(logical_k)?;
        let act_scale_buf = self.metal.new_buffer_checked(logical_k / ACT_QUANT_BLOCK)?;
        let weight_buf = self.metal.new_buffer_from_verified_bytes(weights)?;
        let scale_buf = self.metal.new_buffer_from_verified_bytes(scales)?;
        let out_f32_buf = self.metal.new_buffer_checked(rows * size_of::<f32>())?;
        let out_bf16_buf = self.metal.new_buffer_checked(rows * size_of::<u16>())?;

        let cols_u = logical_k as u32;
        let rows_u = rows as u32;
        let packed_u = packed_k as u32;
        let scale_cols_u = scale_cols as u32;
        let blocks = (logical_k / ACT_QUANT_BLOCK) as u32;
        let aq_tg = self.act_quant_tg.min(blocks.max(1));
        let fp4_tg = self.fp4_tg.min(rows_u.max(1));
        let cast_tg = self.cast_tg.min(rows_u.max(1));

        self.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(ACT_QUANT_KERNEL, (blocks, 1, 1), (aq_tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&input_buf), 0);
                encoder.set_buffer(1, Some(&quant_buf), 0);
                encoder.set_buffer(2, Some(&act_scale_buf), 0);
                set_u32(encoder, 3, &cols_u);
            })?;
            batch.dispatch_threads(FP4_KERNEL, (rows_u, 1, 1), (fp4_tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&weight_buf), 0);
                encoder.set_buffer(1, Some(&scale_buf), 0);
                encoder.set_buffer(2, Some(&quant_buf), 0);
                encoder.set_buffer(3, Some(&act_scale_buf), 0);
                encoder.set_buffer(4, Some(&out_f32_buf), 0);
                set_u32(encoder, 5, &rows_u);
                set_u32(encoder, 6, &packed_u);
                set_u32(encoder, 7, &scale_cols_u);
            })?;
            batch.dispatch_threads(CAST_KERNEL, (rows_u, 1, 1), (cast_tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&out_f32_buf), 0);
                encoder.set_buffer(1, Some(&out_bf16_buf), 0);
                set_u32(encoder, 2, &rows_u);
            })
        })?;
        self.dispatches += 3;
        read_u16_buffer(&out_bf16_buf, rows)
    }

    #[cfg(target_os = "macos")]
    fn wo_a_einsum_macos(
        &mut self,
        attention_bf16: &[u16],
        weights: &[u8],
        scales: &[u8],
    ) -> Result<Vec<u16>> {
        if attention_bf16.len() != O_GROUPS * WO_A_COLS {
            return Err(native_error("wo_a attention is not [8, 4096] BF16"));
        }
        if weights.len() != WO_A_ROWS * WO_A_COLS {
            return Err(native_error("wo_a weight byte count mismatch"));
        }
        let scale_cols = WO_A_COLS / ACT_QUANT_BLOCK;
        if scales.len() != (WO_A_ROWS / ACT_QUANT_BLOCK) * scale_cols {
            return Err(native_error("wo_a scale byte count mismatch"));
        }

        let attn_bytes = u16_as_bytes(attention_bf16);
        let attn_buf = self.metal.new_buffer_with_bytes_checked(attn_bytes)?;
        let weight_buf = self.metal.new_buffer_from_verified_bytes(weights)?;
        let scale_buf = self.metal.new_buffer_from_verified_bytes(scales)?;
        let out_buf = self
            .metal
            .new_buffer_checked(WO_A_ROWS * size_of::<u16>())?;

        let rows_u = WO_A_ROWS as u32;
        let cols_u = WO_A_COLS as u32;
        let scale_cols_u = scale_cols as u32;
        let ranks_u = O_LORA_RANK as u32;
        let tg = self.wo_a_tg.min(rows_u);

        self.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(WO_A_KERNEL, (rows_u, 1, 1), (tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&weight_buf), 0);
                encoder.set_buffer(1, Some(&scale_buf), 0);
                encoder.set_buffer(2, Some(&attn_buf), 0);
                encoder.set_buffer(3, Some(&out_buf), 0);
                set_u32(encoder, 4, &rows_u);
                set_u32(encoder, 5, &cols_u);
                set_u32(encoder, 6, &scale_cols_u);
                set_u32(encoder, 7, &ranks_u);
            })
        })?;
        self.dispatches += 1;
        read_u16_buffer(&out_buf, WO_A_ROWS)
    }

    #[cfg(target_os = "macos")]
    fn gate_logits_macos(&mut self, input_bf16: &[u16], weight_bf16: &[u16]) -> Result<Vec<f32>> {
        use crate::gravity_deepseek_v4_layer0_moe::ROUTED_EXPERTS;
        if input_bf16.len() != HIDDEN_SIZE {
            return Err(native_error("gate input is not BF16[4096]"));
        }
        if weight_bf16.len() != ROUTED_EXPERTS * HIDDEN_SIZE {
            return Err(native_error("gate weight is not BF16[256, 4096]"));
        }
        let input_buf = self
            .metal
            .new_buffer_with_bytes_checked(u16_as_bytes(input_bf16))?;
        let weight_buf = self
            .metal
            .new_buffer_with_bytes_checked(u16_as_bytes(weight_bf16))?;
        let out_buf = self
            .metal
            .new_buffer_checked(ROUTED_EXPERTS * size_of::<f32>())?;
        let rows_u = ROUTED_EXPERTS as u32;
        let cols_u = HIDDEN_SIZE as u32;
        let tg = self.gate_tg.min(rows_u);
        self.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(GATE_KERNEL, (rows_u, 1, 1), (tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&weight_buf), 0);
                encoder.set_buffer(1, Some(&input_buf), 0);
                encoder.set_buffer(2, Some(&out_buf), 0);
                set_u32(encoder, 3, &rows_u);
                set_u32(encoder, 4, &cols_u);
            })
        })?;
        self.dispatches += 1;
        read_f32_buffer(&out_buf, ROUTED_EXPERTS)
    }

    #[cfg(target_os = "macos")]
    fn lm_head_block_macos(
        &mut self,
        residual_f32: &[f32],
        weight_bf16_bytes: &[u8],
        rows: usize,
    ) -> Result<Vec<f32>> {
        if residual_f32.len() != HIDDEN_SIZE {
            return Err(native_error("lm_head residual is not f32[4096]"));
        }
        if rows == 0 || weight_bf16_bytes.len() != rows * HIDDEN_SIZE * size_of::<u16>() {
            return Err(native_error("lm_head block geometry is invalid"));
        }
        let residual_bytes = f32_as_bytes(residual_f32);
        let residual_buf = self.metal.new_buffer_with_bytes_checked(residual_bytes)?;
        let weight_buf = self
            .metal
            .new_buffer_from_verified_bytes(weight_bf16_bytes)?;
        let out_buf = self.metal.new_buffer_checked(rows * size_of::<f32>())?;
        let rows_u = rows as u32;
        let cols_u = HIDDEN_SIZE as u32;
        let tg = self.lm_head_tg.min(rows_u.max(1));
        self.metal.dispatch_batch(|batch| {
            batch.dispatch_threads(LM_HEAD_KERNEL, (rows_u, 1, 1), (tg, 1, 1), |encoder| {
                encoder.set_buffer(0, Some(&weight_buf), 0);
                encoder.set_buffer(1, Some(&residual_buf), 0);
                encoder.set_buffer(2, Some(&out_buf), 0);
                set_u32(encoder, 3, &rows_u);
                set_u32(encoder, 4, &cols_u);
            })
        })?;
        self.dispatches += 1;
        read_f32_buffer(&out_buf, rows)
    }
}

/// CPU reference used by per-operator parity tests (same algorithm as the
/// streamed host oracle).
pub fn cpu_fp8_linear(
    input_bf16: &[u16],
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    logical_k: usize,
) -> Result<Vec<u16>> {
    let quantized = act_quant_bf16_ue8m0(input_bf16)?;
    let output = fp8_e4m3fn_ue8m0_matvec(&quantized, weights, scales, rows, logical_k)?;
    Ok(output.bf16_bits)
}

pub fn cpu_fp4_linear(
    input_bf16: &[u16],
    weights: &[u8],
    scales: &[u8],
    rows: usize,
    logical_k: usize,
) -> Result<Vec<u16>> {
    let quantized = act_quant_bf16_ue8m0(input_bf16)?;
    let output = fp4_e2m1fn_x2_ue8m0_matvec(&quantized, weights, scales, rows, logical_k)?;
    Ok(output.bf16_bits)
}

pub fn cpu_wo_a_einsum(attention_bf16: &[u16], weights: &[u8], scales: &[u8]) -> Result<Vec<u16>> {
    if attention_bf16.len() != O_GROUPS * WO_A_COLS {
        return Err(native_error("WO-A CPU input is not [8, 4096] BF16"));
    }
    let input: Vec<f32> = attention_bf16
        .iter()
        .map(|bits| bf16::from_bits(*bits).to_f32())
        .collect();
    let scale_cols = WO_A_COLS / ACT_QUANT_BLOCK;
    let mut output = Vec::with_capacity(WO_A_ROWS);
    for group in 0..O_GROUPS {
        let input_group = &input[group * WO_A_COLS..(group + 1) * WO_A_COLS];
        for rank in 0..O_LORA_RANK {
            let row = group * O_LORA_RANK + rank;
            let mut acc = 0.0_f32;
            for column in 0..WO_A_COLS {
                let raw = weights[row * WO_A_COLS + column];
                let scale_index = (row / ACT_QUANT_BLOCK) * scale_cols + column / ACT_QUANT_BLOCK;
                let converted =
                    bf16::from_f32(decode_e4m3fn(raw)? * decode_e8m0fnu(scales[scale_index])?)
                        .to_f32();
                acc += input_group[column] * converted;
            }
            if !acc.is_finite() {
                return Err(native_error("WO-A CPU einsum produced a non-finite value"));
            }
            output.push(bf16::from_f32(acc).to_bits());
        }
    }
    Ok(output)
}

/// Compare two BF16 vectors after decoding to f32. Returns (max_abs, max_rel, pass).
pub fn bf16_parity(
    cpu: &[u16],
    gpu: &[u16],
    abs_tol: f32,
    rel_tol: f32,
) -> Result<(f32, f32, bool)> {
    if cpu.len() != gpu.len() || cpu.is_empty() {
        return Err(native_error("parity vectors have incompatible length"));
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut passing = true;
    for (&c, &g) in cpu.iter().zip(gpu) {
        let cr = bf16::from_bits(c).to_f32();
        let gr = bf16::from_bits(g).to_f32();
        if !cr.is_finite() || !gr.is_finite() {
            return Err(native_error("parity encountered a non-finite value"));
        }
        let abs = (cr - gr).abs();
        let rel = abs / cr.abs().max(1.0e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > abs_tol + rel_tol * cr.abs() {
            passing = false;
        }
    }
    Ok((max_abs, max_rel, passing))
}

pub fn f32_parity(
    cpu: &[f32],
    gpu: &[f32],
    abs_tol: f32,
    rel_tol: f32,
) -> Result<(f32, f32, bool)> {
    if cpu.len() != gpu.len() || cpu.is_empty() {
        return Err(native_error("f32 parity vectors have incompatible length"));
    }
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut passing = true;
    for (&cr, &gr) in cpu.iter().zip(gpu) {
        if !cr.is_finite() || !gr.is_finite() {
            return Err(native_error("f32 parity encountered a non-finite value"));
        }
        let abs = (cr - gr).abs();
        let rel = abs / cr.abs().max(1.0e-6);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        if abs > abs_tol + rel_tol * cr.abs() {
            passing = false;
        }
    }
    Ok((max_abs, max_rel, passing))
}

pub fn greedy_logit_within_tolerance(observed: f32, oracle: f32) -> bool {
    observed.is_finite() && (observed - oracle).abs() <= GREEDY_LOGIT_ABS_TOL
}

/// Host greedy over a streamed Metal lm-head: the caller supplies residual
/// and already-read weight bytes; this only runs the GEMV + argmax.
pub fn greedy_from_logits(logits: &[f32], vocab_offset: usize) -> (u32, f32) {
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

pub fn finish_greedy(
    token_id: u32,
    logit: f32,
    metal_dispatches: usize,
) -> Result<DeepSeekV4GreedyTokenResult> {
    if !logit.is_finite() {
        return Err(native_error("greedy lm_head produced a non-finite logit"));
    }
    Ok(DeepSeekV4GreedyTokenResult {
        token_id,
        logit,
        vocab_size: DSV4F_VOCAB_SIZE,
        lm_head_on_device: true,
        argmax_on_device: false,
        metal_dispatches,
        command_buffers: metal_dispatches,
    })
}

#[cfg(target_os = "macos")]
fn pipeline_tg(metal: &crate::metal::MetalContext, kernel: &str, preferred: u32) -> Result<u32> {
    let max = metal.pipeline(kernel)?.max_total_threads_per_threadgroup() as u32;
    if max == 0 {
        return Err(native_error(format!(
            "{kernel} reports a zero threadgroup limit"
        )));
    }
    Ok(preferred.min(max).max(1))
}

#[cfg(target_os = "macos")]
fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: &u32) {
    encoder.set_bytes(
        index,
        size_of::<u32>() as u64,
        value as *const u32 as *const _,
    );
}

#[cfg(target_os = "macos")]
fn read_u16_buffer(buf: &metal::Buffer, n: usize) -> Result<Vec<u16>> {
    let ptr = buf.contents() as *const u16;
    if ptr.is_null() {
        return Err(native_error("Metal u16 buffer contents pointer is null"));
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    Ok(slice.to_vec())
}

#[cfg(target_os = "macos")]
fn read_f32_buffer(buf: &metal::Buffer, n: usize) -> Result<Vec<f32>> {
    let ptr = buf.contents() as *const f32;
    if ptr.is_null() {
        return Err(native_error("Metal f32 buffer contents pointer is null"));
    }
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    Ok(slice.to_vec())
}

fn u16_as_bytes(values: &[u16]) -> &[u8] {
    bytemuck::cast_slice(values)
}

#[cfg(target_os = "macos")]
fn f32_as_bytes(values: &[f32]) -> &[u8] {
    bytemuck::cast_slice(values)
}

fn native_error(message: impl Into<String>) -> Error {
    Error::Gravity(format!("dsv4f streamed native: {}", message.into()))
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    use crate::gravity_deepseek_v4_layer0_moe::ROUTED_EXPERTS;

    fn try_session() -> Option<StreamedNativeSession> {
        StreamedNativeSession::new().ok()
    }

    fn finite_e4m3(index: usize) -> u8 {
        let mut byte = (index.wrapping_mul(37).wrapping_add(11) % 127) as u8;
        if byte == 0x7f {
            byte = 0x3c;
        }
        byte
    }

    fn finite_e8m0(index: usize) -> u8 {
        let mut byte = 0x70 + (index % 16) as u8;
        if byte == 0xff {
            byte = 0x7f;
        }
        byte
    }

    fn finite_bf16_row(len: usize, seed: u32) -> Vec<u16> {
        (0..len)
            .map(|i| {
                let signed = ((i as u32).wrapping_mul(73).wrapping_add(seed) % 257) as i32 - 128;
                bf16::from_f32(signed as f32 / 128.0).to_bits()
            })
            .collect()
    }

    #[test]
    fn synthetic_fp8_linear_matches_cpu_oracle() {
        let Some(mut session) = try_session() else {
            eprintln!("Metal unavailable; synthetic fp8 parity skipped");
            return;
        };
        let rows = 128;
        let k = 256;
        let input = finite_bf16_row(k, 19);
        let weights: Vec<u8> = (0..rows * k).map(finite_e4m3).collect();
        let scales: Vec<u8> = (0..(rows / ACT_QUANT_BLOCK) * (k / ACT_QUANT_BLOCK))
            .map(finite_e8m0)
            .collect();
        let before = session.metal_dispatches();
        let gpu = session
            .fp8_linear(&input, &weights, &scales, rows, k)
            .expect("metal fp8");
        let cpu = cpu_fp8_linear(&input, &weights, &scales, rows, k).expect("cpu fp8");
        let (max_abs, max_rel, pass) =
            bf16_parity(&cpu, &gpu, LINEAR_ABS_TOL, LINEAR_REL_TOL).expect("parity");
        assert!(
            pass,
            "fp8 metal vs cpu failed max_abs={max_abs} max_rel={max_rel}"
        );
        assert!(session.metal_dispatches() > before);
        assert_eq!(session.fallbacks(), 0);
    }

    #[test]
    fn synthetic_fp4_linear_matches_cpu_oracle() {
        let Some(mut session) = try_session() else {
            eprintln!("Metal unavailable; synthetic fp4 parity skipped");
            return;
        };
        let rows = 128;
        let k = 256;
        let packed_k = k / 2;
        let input = finite_bf16_row(k, 23);
        let weights: Vec<u8> = (0..rows * packed_k)
            .map(|i| finite_e4m3(i).wrapping_add(3))
            .collect();
        let scales: Vec<u8> = (0..rows * (k / FP4_BLOCK)).map(finite_e8m0).collect();
        let gpu = session
            .fp4_linear(&input, &weights, &scales, rows, k)
            .expect("metal fp4");
        let cpu = cpu_fp4_linear(&input, &weights, &scales, rows, k).expect("cpu fp4");
        let (max_abs, max_rel, pass) =
            bf16_parity(&cpu, &gpu, LINEAR_ABS_TOL, LINEAR_REL_TOL).expect("parity");
        assert!(
            pass,
            "fp4 metal vs cpu failed max_abs={max_abs} max_rel={max_rel}"
        );
        assert!(session.metal_dispatches() > 0);
        assert_eq!(session.fallbacks(), 0);
    }

    #[test]
    fn synthetic_gate_matches_cpu_dot() {
        let Some(mut session) = try_session() else {
            eprintln!("Metal unavailable; synthetic gate parity skipped");
            return;
        };
        let input = finite_bf16_row(HIDDEN_SIZE, 7);
        let mut weights = Vec::with_capacity(ROUTED_EXPERTS * HIDDEN_SIZE);
        for row in 0..ROUTED_EXPERTS {
            weights.extend(finite_bf16_row(HIDDEN_SIZE, 11 + row as u32));
        }
        let gpu = session.gate_logits(&input, &weights).expect("metal gate");
        let input_f: Vec<f32> = input.iter().map(|b| bf16::from_bits(*b).to_f32()).collect();
        let mut cpu = Vec::with_capacity(ROUTED_EXPERTS);
        for row in 0..ROUTED_EXPERTS {
            let mut acc = 0.0_f32;
            let wrow = &weights[row * HIDDEN_SIZE..(row + 1) * HIDDEN_SIZE];
            for (&a, &w) in input_f.iter().zip(wrow) {
                acc += a * bf16::from_bits(w).to_f32();
            }
            cpu.push(acc);
        }
        let (max_abs, max_rel, pass) =
            f32_parity(&cpu, &gpu, LINEAR_ABS_TOL, LINEAR_REL_TOL).expect("gate parity");
        assert!(
            pass,
            "gate metal vs cpu failed max_abs={max_abs} max_rel={max_rel}"
        );
        assert!(session.metal_dispatches() > 0);
        assert_eq!(session.fallbacks(), 0);
    }

    #[test]
    fn dispatch_and_fallback_counters_are_real() {
        let Some(mut session) = try_session() else {
            eprintln!("Metal unavailable; counter test skipped");
            return;
        };
        assert_eq!(session.metal_dispatches(), 0);
        assert_eq!(session.fallbacks(), 0);
        session.record_fallback("unit-test injected fallback");
        assert_eq!(session.fallbacks(), 1);
        assert_eq!(session.fallback_reasons(), ["unit-test injected fallback"]);
        let input = finite_bf16_row(128, 1);
        let weights = vec![0x38u8; 128 * 128];
        let scales = vec![0x7fu8; 1];
        let _ = session.fp8_linear(&input, &weights, &scales, 128, 128);
        assert!(session.metal_dispatches() >= 3);
    }

    #[test]
    fn greedy_token_tolerance_helper_matches_oracle() {
        assert!(greedy_logit_within_tolerance(
            ORACLE_GREEDY_LOGIT,
            ORACLE_GREEDY_LOGIT
        ));
        assert!(greedy_logit_within_tolerance(
            ORACLE_GREEDY_LOGIT + 0.01,
            ORACLE_GREEDY_LOGIT
        ));
        assert!(!greedy_logit_within_tolerance(
            ORACLE_GREEDY_LOGIT + 1.0,
            ORACLE_GREEDY_LOGIT
        ));
        assert_eq!(ORACLE_GREEDY_TOKEN_ID, 5);
    }

    #[test]
    fn live_layer0_operators_match_cpu_oracle() {
        use crate::gravity_deepseek_v4::DeepSeekV4FullStreamReader;
        use crate::gravity_deepseek_v4_layer0_attention::Q_LORA_RANK;
        use crate::gravity_deepseek_v4_streamed_forward::{
            discover_sealed_dsv4f_artifact, prepare_sealed_admission_root,
        };

        let Some(artifact) = discover_sealed_dsv4f_artifact() else {
            eprintln!("sealed DSV4F artifact not found; live operator parity skipped");
            return;
        };
        let Some(mut session) = try_session() else {
            eprintln!("Metal unavailable; live operator parity skipped");
            return;
        };
        let admission = prepare_sealed_admission_root(&artifact).expect("admit");
        let reader = DeepSeekV4FullStreamReader::admit(&admission.path).expect("reader");
        let input = finite_bf16_row(HIDDEN_SIZE, 41);

        let wq_a_w = reader
            .read_verified_full("layers.0.attn.wq_a.weight", Q_LORA_RANK * HIDDEN_SIZE)
            .expect("wq_a weight");
        let wq_a_s = reader
            .read_verified_full(
                "layers.0.attn.wq_a.scale",
                (Q_LORA_RANK / ACT_QUANT_BLOCK) * (HIDDEN_SIZE / ACT_QUANT_BLOCK),
            )
            .expect("wq_a scale");
        let cpu =
            cpu_fp8_linear(&input, &wq_a_w, &wq_a_s, Q_LORA_RANK, HIDDEN_SIZE).expect("cpu wq_a");
        let gpu = session
            .fp8_linear(&input, &wq_a_w, &wq_a_s, Q_LORA_RANK, HIDDEN_SIZE)
            .expect("metal wq_a");
        let (max_abs, max_rel, pass) =
            bf16_parity(&cpu, &gpu, LINEAR_ABS_TOL, LINEAR_REL_TOL).expect("wq_a parity");
        assert!(
            pass,
            "live wq_a metal vs cpu failed max_abs={max_abs} max_rel={max_rel}"
        );

        let packed_k = HIDDEN_SIZE / 2;
        let scale_cols = HIDDEN_SIZE / FP4_BLOCK;
        let w1 = reader
            .read_verified_full(
                "layers.0.ffn.experts.0.w1.weight",
                crate::gravity_deepseek_v4_layer0_moe::MOE_INTER_DIM * packed_k,
            )
            .expect("expert w1");
        let s1 = reader
            .read_verified_full(
                "layers.0.ffn.experts.0.w1.scale",
                crate::gravity_deepseek_v4_layer0_moe::MOE_INTER_DIM * scale_cols,
            )
            .expect("expert w1 scale");
        let cpu = cpu_fp4_linear(
            &input,
            &w1,
            &s1,
            crate::gravity_deepseek_v4_layer0_moe::MOE_INTER_DIM,
            HIDDEN_SIZE,
        )
        .expect("cpu fp4");
        let gpu = session
            .fp4_linear(
                &input,
                &w1,
                &s1,
                crate::gravity_deepseek_v4_layer0_moe::MOE_INTER_DIM,
                HIDDEN_SIZE,
            )
            .expect("metal fp4");
        let (max_abs, max_rel, pass) =
            bf16_parity(&cpu, &gpu, LINEAR_ABS_TOL, LINEAR_REL_TOL).expect("fp4 parity");
        assert!(
            pass,
            "live routed w1 metal vs cpu failed max_abs={max_abs} max_rel={max_rel}"
        );
        assert!(session.metal_dispatches() > 0);
        assert_eq!(session.fallbacks(), 0);
    }
}
