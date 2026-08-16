//! Bind the packed Q80 mixed ≤1.5 catalog to the existing hybrid token graph.
//!
//! The graph is the Q4 hybrid schedule (embed → 48 mixer/MoE layers →
//! terminal greedy). Only the weight service changes:
//!   routed gate  HGRAVB01 binary_group
//!   routed up    HGRAVR02 binary + rice_q1_rms
//!   routed down  HGRAVS01 y = L @ (R @ x)
//!   non-expert   HGRAVU01 uniform-q8 group-64
//!
//! Packed bytes go to registers/simdgroup and are consumed in the same
//! kernel. A dense `W` is never allocated on this path.

use super::qwen80_complete_runtime::{
    qwen80_gqa_apply_sigmoid_gate, qwen80_gqa_causal_attention,
    qwen80_gqa_query_from_interleaved_q_projection, qwen80_gqa_source_norm_rope, qwen80_layer_kind,
    source_qwen80_ba_to_decay_beta, source_qwen80_causal_conv_step_dense,
    source_qwen80_gated_rms_norm, source_qwen80_l2_normalize, source_qwen80_recurrent_deltanet,
    source_qwen80_residual_rms_norm, source_qwen80_split_linear_qkvz, source_qwen80_topk_router,
    Qwen80CanonicalGqaLayout, Qwen80CanonicalLinearDeltaNetLayout, Qwen80LayerKind, QWEN80_EXPERTS,
    QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_TOKENIZER_VOCAB, QWEN80_VOCAB,
};
use super::qwen80_mixed_catalog::{
    Qwen80MixedStreamingCatalog, QWEN80_MIXED_EXPECTED_TENSOR_COUNT, QWEN80_MIXED_MANIFEST_NAME,
};
use super::qwen80_source_bf16_layer_major::{peak_rss_bytes, STREAMED_PEAK_RSS_HARD_CAP_BYTES};
use super::qwen80_uniform_q4_hybrid_decode::{
    load_qwen80_tokenizer, Qwen80ActivationClassCounts, Qwen80ActivationClassTimes,
    Qwen80HybridDecodeState,
};
use super::qwen_complete_binary::{
    max_abs_error, rice_q1_row_ptr, MixedPackedTensor, Q80_DOWN_COLS,
    Q80_DOWN_ROWS, Q80_GATE_COLS, Q80_GATE_ROWS, Q80_HGRAVS_BITS, Q80_HGRAVS_GROUP_SIZE,
    Q80_HGRAVS_RANK,
};
use crate::kernels::{add_inplace, silu_mul};
use crate::tokenizer::Tokenizer;
use crate::{Error, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const QWEN80_MIXED_CLAIM: &str = "MIXED_1P5_GENERATION_GATE_NOT_BASE_TRUE_TPS";
pub const QWEN80_MIXED_EXPECTED_MANIFEST_SEAL: &str =
    "6a09fa747af1431b67e53691bc24dfa421c0a7643c5befb297b2eed0f4a95af6";
pub const QWEN80_MIXED_NUMERIC_TOL: f32 = 2.0e-5;
const MIXED_DEFAULT_ROOT_ABS: &str = "/Users/scammermike/Downloads/hawking/workspace/campaign/records/ascension-sandbox/physical/qwen80/quality-candidates/mixed-1p5-v1";

fn mixed_error(message: impl Into<String>) -> Error {
    Error::Model(format!("qwen80 mixed hybrid decode: {}", message.into()))
}

fn add_secs(slot: &mut f64, started: Instant) {
    *slot += started.elapsed().as_secs_f64();
}

fn require_rss_cap(label: &str) -> Result<()> {
    let peak = peak_rss_bytes();
    if peak > STREAMED_PEAK_RSS_HARD_CAP_BYTES {
        return Err(mixed_error(format!(
            "{label}: peak RSS {peak} exceeds streamed cap {STREAMED_PEAK_RSS_HARD_CAP_BYTES}"
        )));
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn set_u32(encoder: &metal::ComputeCommandEncoderRef, index: u64, value: u32) {
    encoder.set_bytes(
        index,
        std::mem::size_of::<u32>() as u64,
        &value as *const u32 as *const _,
    );
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Qwen80MixedFallbackCounts {
    pub host_mixed_matvec: u64,
    pub host_expert_payload_bind: u64,
    pub dense_w_materialized: u64,
    pub host_q8_vector_decode: u64,
    pub host_q8_embed_gather: u64,
    pub host_activation: u64,
    pub host_sample: u64,
}

impl Qwen80MixedFallbackCounts {
    pub fn silent_or_invalid(&self) -> u64 {
        self.host_mixed_matvec
            .saturating_add(self.host_expert_payload_bind)
            .saturating_add(self.dense_w_materialized)
    }

    pub fn designed_host_ops(&self) -> u64 {
        self.host_q8_vector_decode
            .saturating_add(self.host_q8_embed_gather)
            .saturating_add(self.host_activation)
            .saturating_add(self.host_sample)
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Qwen80MixedNativeCounts {
    pub binary_dispatches: u64,
    pub residual_dispatches: u64,
    pub hgravs_factor_dispatches: u64,
    pub uniform8_dispatches: u64,
    pub routed_expert_waves: u64,
    pub command_buffers: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Qwen80MixedStageTimes {
    pub embed_secs: f64,
    pub deltanet_secs: f64,
    pub gqa_secs: f64,
    pub moe_norm_router_secs: f64,
    pub moe_shared_secs: f64,
    pub moe_routed_secs: f64,
    pub moe_combine_secs: f64,
    pub terminal_secs: f64,
    pub mixed_matvec_secs: f64,
    #[serde(skip)]
    pub activation: Qwen80ActivationClassTimes,
    pub gpu_matvec_ns: u64,
    pub gpu_matvec_timestamps_missing: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Qwen80MixedParityReport {
    pub passed: bool,
    pub samples: Vec<Value>,
    pub dense_w_materialized: bool,
}

struct VectorCache {
    vectors: HashMap<String, Vec<f32>>,
}

impl VectorCache {
    fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }
}

#[cfg(target_os = "macos")]
struct GpuBinary {
    signs: crate::metal::PinnedBuffer,
    scales: crate::metal::PinnedBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    groups_per_row: u32,
}

#[cfg(target_os = "macos")]
struct GpuResidual {
    binary: GpuBinary,
    indices: crate::metal::PinnedBuffer,
    row_ptr: crate::metal::PinnedBuffer,
    residual_signs: crate::metal::PinnedBuffer,
    residual_scale_f16: u32,
}

#[cfg(target_os = "macos")]
struct GpuHgravs {
    left_codes: crate::metal::PinnedBuffer,
    left_scales: crate::metal::PinnedBuffer,
    right_codes: crate::metal::PinnedBuffer,
    right_scales: crate::metal::PinnedBuffer,
    left_rows: u32,
    left_cols: u32,
    right_rows: u32,
    right_cols: u32,
    group_size: u32,
    bits: u32,
    bound: u32,
}

#[cfg(target_os = "macos")]
struct GpuUniform {
    codes: crate::metal::PinnedBuffer,
    scales: crate::metal::PinnedBuffer,
    rows: u32,
    cols: u32,
    group_size: u32,
    bits: u32,
    bound: u32,
}

#[cfg(target_os = "macos")]
enum GpuWeight {
    Binary(GpuBinary),
    Residual(GpuResidual),
    Hgravs(GpuHgravs),
    Uniform(GpuUniform),
}

#[cfg(target_os = "macos")]
struct MixedExpertGpu {
    gate: GpuBinary,
    up: GpuResidual,
    down: GpuHgravs,
}

#[cfg(target_os = "macos")]
struct MixedWave {
    input: crate::metal::PinnedBuffer,
    gate: crate::metal::PinnedBuffer,
    up: crate::metal::PinnedBuffer,
    act: crate::metal::PinnedBuffer,
    mid: crate::metal::PinnedBuffer,
    down: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
struct MetalMixedAccel {
    context: crate::metal::MetalContext,
    weights: HashMap<String, GpuWeight>,
    experts: HashMap<(usize, u16), MixedExpertGpu>,
    wave: MixedWave,
}

#[cfg(target_os = "macos")]
fn as_u8_u16(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(target_os = "macos")]
fn write_f32(buf: &crate::metal::PinnedBuffer, values: &[f32]) {
    crate::metal::MetalContext::write_buffer_bytes(buf, bytemuck::cast_slice(values));
}

#[cfg(target_os = "macos")]
fn read_f32(buf: &crate::metal::PinnedBuffer, n: usize) -> Vec<f32> {
    unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n).to_vec() }
}

#[cfg(target_os = "macos")]
fn encode_binary(
    enc: &metal::ComputeCommandEncoderRef,
    packed: &GpuBinary,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
) {
    enc.set_buffer(0, Some(&packed.signs), 0);
    enc.set_buffer(1, Some(&packed.scales), 0);
    enc.set_buffer(2, Some(input), 0);
    enc.set_buffer(3, Some(output), output_offset);
    set_u32(enc, 4, packed.rows);
    set_u32(enc, 5, packed.cols);
    set_u32(enc, 6, packed.group_size);
    set_u32(enc, 7, packed.groups_per_row);
}

#[cfg(target_os = "macos")]
fn encode_csr(
    enc: &metal::ComputeCommandEncoderRef,
    packed: &GpuResidual,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
) {
    enc.set_buffer(0, Some(&packed.indices), 0);
    enc.set_buffer(1, Some(&packed.row_ptr), 0);
    enc.set_buffer(2, Some(&packed.residual_signs), 0);
    enc.set_buffer(3, Some(input), 0);
    enc.set_buffer(4, Some(output), output_offset);
    set_u32(enc, 5, packed.binary.rows);
    set_u32(enc, 6, packed.binary.cols);
    set_u32(enc, 7, packed.residual_scale_f16);
}

#[cfg(target_os = "macos")]
fn encode_factor(
    enc: &metal::ComputeCommandEncoderRef,
    codes: &crate::metal::PinnedBuffer,
    scales: &crate::metal::PinnedBuffer,
    input: &crate::metal::PinnedBuffer,
    input_offset: u64,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
    rows: u32,
    cols: u32,
    group_size: u32,
    bits: u32,
    bound: u32,
) {
    enc.set_buffer(0, Some(codes), 0);
    enc.set_buffer(1, Some(scales), 0);
    enc.set_buffer(2, Some(input), input_offset);
    enc.set_buffer(3, Some(output), output_offset);
    set_u32(enc, 4, rows);
    set_u32(enc, 5, cols);
    set_u32(enc, 6, group_size);
    set_u32(enc, 7, bits);
    set_u32(enc, 8, bound);
}

#[cfg(target_os = "macos")]
impl MetalMixedAccel {
    fn new() -> Result<Self> {
        let context = crate::metal::MetalContext::new()?;
        let hidden = QWEN80_HIDDEN * 4;
        let mid = 10 * QWEN80_MOE_INTERMEDIATE * 4;
        let rank = 10 * Q80_HGRAVS_RANK * 4;
        let down = 10 * QWEN80_HIDDEN * 4;
        Ok(Self {
            wave: MixedWave {
                input: context.new_buffer_checked(hidden)?,
                gate: context.new_buffer_checked(mid)?,
                up: context.new_buffer_checked(mid)?,
                act: context.new_buffer_checked(mid)?,
                mid: context.new_buffer_checked(rank)?,
                down: context.new_buffer_checked(down)?,
            },
            context,
            weights: HashMap::new(),
            experts: HashMap::new(),
        })
    }

    fn upload_binary(
        &self,
        packed: &super::qwen_complete_binary::BinaryGroupPacked,
    ) -> Result<GpuBinary> {
        Ok(GpuBinary {
            signs: self.context.new_buffer_with_bytes_checked(&packed.signs)?,
            scales: self
                .context
                .new_buffer_with_bytes_checked(&as_u8_u16(&packed.scales_f16))?,
            rows: packed.rows as u32,
            cols: packed.cols as u32,
            group_size: packed.group_size as u32,
            groups_per_row: packed.groups_per_row as u32,
        })
    }

    fn upload_residual(
        &self,
        packed: &super::qwen_complete_binary::RiceQ1Packed,
    ) -> Result<GpuResidual> {
        let indices = if packed.indices.is_empty() {
            super::qwen_complete_binary::expand_rice_indices(packed)?
        } else {
            packed.indices.clone()
        };
        let row_ptr = rice_q1_row_ptr(&indices, packed.binary.rows, packed.binary.cols)?;
        let idx_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
        let ptr_bytes: Vec<u8> = row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect();
        Ok(GpuResidual {
            binary: self.upload_binary(&packed.binary)?,
            indices: self.context.new_buffer_with_bytes_checked(&idx_bytes)?,
            row_ptr: self.context.new_buffer_with_bytes_checked(&ptr_bytes)?,
            residual_signs: self
                .context
                .new_buffer_with_bytes_checked(&packed.residual_signs)?,
            residual_scale_f16: u32::from(packed.residual_scale_f16),
        })
    }

    fn upload_hgravs(
        &self,
        left: &super::qwen_complete_binary::UniformFactorPacked,
        right: &super::qwen_complete_binary::UniformFactorPacked,
    ) -> Result<GpuHgravs> {
        if left.bits != Q80_HGRAVS_BITS
            || right.bits != Q80_HGRAVS_BITS
            || left.group_size != Q80_HGRAVS_GROUP_SIZE
            || right.group_size != Q80_HGRAVS_GROUP_SIZE
            || left.cols != Q80_HGRAVS_RANK
            || right.rows != Q80_HGRAVS_RANK
        {
            return Err(mixed_error(format!(
                "hgravs geometry {}x{} / {}x{} bits={}/{} group={}/{} is not r160_b3",
                left.rows,
                left.cols,
                right.rows,
                right.cols,
                left.bits,
                right.bits,
                left.group_size,
                right.group_size
            )));
        }
        Ok(GpuHgravs {
            left_codes: self.context.new_buffer_with_bytes_checked(&left.codes)?,
            left_scales: self
                .context
                .new_buffer_with_bytes_checked(&as_u8_u16(&left.scales_f16))?,
            right_codes: self.context.new_buffer_with_bytes_checked(&right.codes)?,
            right_scales: self
                .context
                .new_buffer_with_bytes_checked(&as_u8_u16(&right.scales_f16))?,
            left_rows: left.rows as u32,
            left_cols: left.cols as u32,
            right_rows: right.rows as u32,
            right_cols: right.cols as u32,
            group_size: left.group_size as u32,
            bits: u32::from(left.bits),
            bound: u32::from(left.bound),
        })
    }

    fn upload_uniform(
        &self,
        packed: &super::qwen_complete_binary::UniformFactorPacked,
    ) -> Result<GpuUniform> {
        Ok(GpuUniform {
            codes: self.context.new_buffer_with_bytes_checked(&packed.codes)?,
            scales: self
                .context
                .new_buffer_with_bytes_checked(&as_u8_u16(&packed.scales_f16))?,
            rows: packed.rows as u32,
            cols: packed.cols as u32,
            group_size: packed.group_size as u32,
            bits: u32::from(packed.bits),
            bound: u32::from(packed.bound),
        })
    }

    fn upload_tensor(&self, packed: &MixedPackedTensor) -> Result<GpuWeight> {
        match packed {
            MixedPackedTensor::Binary(body) => Ok(GpuWeight::Binary(self.upload_binary(body)?)),
            MixedPackedTensor::Residual(body) => {
                Ok(GpuWeight::Residual(self.upload_residual(body)?))
            }
            MixedPackedTensor::Hgravs { left, right } => {
                Ok(GpuWeight::Hgravs(self.upload_hgravs(left, right)?))
            }
            MixedPackedTensor::Uniform8(body) => Ok(GpuWeight::Uniform(self.upload_uniform(body)?)),
        }
    }

    fn note_timing(
        stages: &mut Qwen80MixedStageTimes,
        native: &mut Qwen80MixedNativeCounts,
        timing: &crate::metal::CommandBufferTiming,
    ) {
        native.command_buffers = native.command_buffers.saturating_add(1);
        match timing.gpu_ns {
            Some(ns) => stages.gpu_matvec_ns = stages.gpu_matvec_ns.saturating_add(ns),
            None => {
                stages.gpu_matvec_timestamps_missing =
                    stages.gpu_matvec_timestamps_missing.saturating_add(1)
            }
        }
    }

    fn matvec(
        &mut self,
        name: &str,
        packed: &MixedPackedTensor,
        input: &[f32],
        output: &mut [f32],
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
    ) -> Result<()> {
        let (rows, cols) = packed.rows_cols()?;
        if input.len() != cols || output.len() != rows {
            return Err(mixed_error(format!(
                "{name} metal matvec geometry {}x{} vs in={} out={}",
                rows,
                cols,
                input.len(),
                output.len()
            )));
        }
        if !self.weights.contains_key(name) {
            let uploaded = self.upload_tensor(packed)?;
            self.weights.insert(name.to_owned(), uploaded);
        }
        let input_buf = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(input))?;
        let output_buf = self.context.new_buffer_checked(rows * 4)?;
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        match self.weights.get(name).expect("uploaded") {
            GpuWeight::Binary(body) => {
                tcb.dispatch_threads(
                    "q80_binary_group_matvec",
                    (body.rows, 1, 1),
                    (256, 1, 1),
                    |enc| encode_binary(enc, body, &input_buf, &output_buf, 0),
                )?;
                native.binary_dispatches = native.binary_dispatches.saturating_add(1);
            }
            GpuWeight::Residual(body) => {
                tcb.dispatch_threads(
                    "q80_binary_group_matvec",
                    (body.binary.rows, 1, 1),
                    (256, 1, 1),
                    |enc| encode_binary(enc, &body.binary, &input_buf, &output_buf, 0),
                )?;
                tcb.dispatch_threads(
                    "q80_sparse_q1_apply_csr",
                    (body.binary.rows, 1, 1),
                    (256, 1, 1),
                    |enc| encode_csr(enc, body, &input_buf, &output_buf, 0),
                )?;
                native.binary_dispatches = native.binary_dispatches.saturating_add(1);
                native.residual_dispatches = native.residual_dispatches.saturating_add(1);
            }
            GpuWeight::Hgravs(body) => {
                let mid = self
                    .context
                    .new_buffer_checked(body.right_rows as usize * 4)?;
                tcb.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (body.right_rows, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_factor(
                            enc,
                            &body.right_codes,
                            &body.right_scales,
                            &input_buf,
                            0,
                            &mid,
                            0,
                            body.right_rows,
                            body.right_cols,
                            body.group_size,
                            body.bits,
                            body.bound,
                        )
                    },
                )?;
                tcb.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (body.left_rows, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_factor(
                            enc,
                            &body.left_codes,
                            &body.left_scales,
                            &mid,
                            0,
                            &output_buf,
                            0,
                            body.left_rows,
                            body.left_cols,
                            body.group_size,
                            body.bits,
                            body.bound,
                        )
                    },
                )?;
                native.hgravs_factor_dispatches =
                    native.hgravs_factor_dispatches.saturating_add(2);
            }
            GpuWeight::Uniform(body) => {
                tcb.dispatch_threads(
                    "q80_hgravs01_factor_matvec",
                    (body.rows, 1, 1),
                    (256, 1, 1),
                    |enc| {
                        encode_factor(
                            enc,
                            &body.codes,
                            &body.scales,
                            &input_buf,
                            0,
                            &output_buf,
                            0,
                            body.rows,
                            body.cols,
                            body.group_size,
                            body.bits,
                            body.bound,
                        )
                    },
                )?;
                native.uniform8_dispatches = native.uniform8_dispatches.saturating_add(1);
            }
        }
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing);
        output.copy_from_slice(&read_f32(&output_buf, rows));
        Ok(())
    }

    fn ensure_expert(
        &mut self,
        catalog: &Qwen80MixedStreamingCatalog,
        layer: usize,
        expert: u16,
    ) -> Result<()> {
        if self.experts.contains_key(&(layer, expert)) {
            return Ok(());
        }
        let gate_name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight");
        let down_name = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight");
        let gate_row = catalog.require_row(&gate_name)?;
        let up_row = catalog.require_row(&up_name)?;
        let down_row = catalog.require_row(&down_name)?;
        if gate_row.codec != 0 || gate_row.organ != 0 {
            return Err(mixed_error(format!(
                "{gate_name} is codec/organ {}/{} not binary/gate",
                gate_row.codec, gate_row.organ
            )));
        }
        if up_row.codec != 1 || up_row.organ != 1 {
            return Err(mixed_error(format!(
                "{up_name} is codec/organ {}/{} not residual/up",
                up_row.codec, up_row.organ
            )));
        }
        if down_row.codec != 2 || down_row.organ != 2 {
            return Err(mixed_error(format!(
                "{down_name} is codec/organ {}/{} not hgravs/down",
                down_row.codec, down_row.organ
            )));
        }
        let gate = match catalog.load_packed(&gate_name)? {
            MixedPackedTensor::Binary(body) => {
                if body.rows != Q80_GATE_ROWS || body.cols != Q80_GATE_COLS {
                    return Err(mixed_error(format!(
                        "{gate_name} geometry {}x{} != 512x2048",
                        body.rows, body.cols
                    )));
                }
                self.upload_binary(&body)?
            }
            _ => return Err(mixed_error(format!("{gate_name} did not parse as binary"))),
        };
        let up = match catalog.load_packed(&up_name)? {
            MixedPackedTensor::Residual(body) => {
                if body.binary.rows != Q80_GATE_ROWS || body.binary.cols != Q80_GATE_COLS {
                    return Err(mixed_error(format!(
                        "{up_name} geometry {}x{} != 512x2048",
                        body.binary.rows, body.binary.cols
                    )));
                }
                self.upload_residual(&body)?
            }
            _ => return Err(mixed_error(format!("{up_name} did not parse as rice residual"))),
        };
        let down = match catalog.load_packed(&down_name)? {
            MixedPackedTensor::Hgravs { left, right } => {
                if left.rows != Q80_DOWN_ROWS || right.cols != Q80_DOWN_COLS {
                    return Err(mixed_error(format!(
                        "{down_name} geometry {}x{} != 2048x512",
                        left.rows, right.cols
                    )));
                }
                self.upload_hgravs(&left, &right)?
            }
            _ => return Err(mixed_error(format!("{down_name} did not parse as hgravs01"))),
        };
        self.experts
            .insert((layer, expert), MixedExpertGpu { gate, up, down });
        Ok(())
    }

    fn routed_wave(
        &mut self,
        catalog: &Qwen80MixedStreamingCatalog,
        layer: usize,
        ids: &[u16],
        weights: &[f32],
        input: &[f32],
        combined: &mut [f32],
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
    ) -> Result<()> {
        if ids.len() != 10 || weights.len() != 10 || input.len() != QWEN80_HIDDEN {
            return Err(mixed_error("routed wave expects top-10 and hidden=2048"));
        }
        for &expert in ids {
            self.ensure_expert(catalog, layer, expert)?;
        }
        write_f32(&self.wave.input, input);
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let mid_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            tcb.dispatch_threads(
                "q80_binary_group_matvec",
                (trip.gate.rows, 1, 1),
                (256, 1, 1),
                |enc| encode_binary(enc, &trip.gate, &self.wave.input, &self.wave.gate, mid_off),
            )?;
            tcb.dispatch_threads(
                "q80_binary_group_matvec",
                (trip.up.binary.rows, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_binary(
                        enc,
                        &trip.up.binary,
                        &self.wave.input,
                        &self.wave.up,
                        mid_off,
                    )
                },
            )?;
            tcb.dispatch_threads(
                "q80_sparse_q1_apply_csr",
                (trip.up.binary.rows, 1, 1),
                (256, 1, 1),
                |enc| encode_csr(enc, &trip.up, &self.wave.input, &self.wave.up, mid_off),
            )?;
            native.binary_dispatches = native.binary_dispatches.saturating_add(2);
            native.residual_dispatches = native.residual_dispatches.saturating_add(1);
        }
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing);

        let gate = read_f32(&self.wave.gate, 10 * QWEN80_MOE_INTERMEDIATE);
        let up = read_f32(&self.wave.up, 10 * QWEN80_MOE_INTERMEDIATE);
        let mut act = vec![0.0f32; 10 * QWEN80_MOE_INTERMEDIATE];
        for slot in 0..10 {
            let a = slot * QWEN80_MOE_INTERMEDIATE;
            let b = a + QWEN80_MOE_INTERMEDIATE;
            silu_mul(&gate[a..b], &up[a..b], &mut act[a..b]);
        }
        write_f32(&self.wave.act, &act);

        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let act_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            let mid_off = (slot * Q80_HGRAVS_RANK * 4) as u64;
            let down_off = (slot * QWEN80_HIDDEN * 4) as u64;
            tcb.dispatch_threads(
                "q80_hgravs01_factor_matvec",
                (trip.down.right_rows, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_factor(
                        enc,
                        &trip.down.right_codes,
                        &trip.down.right_scales,
                        &self.wave.act,
                        act_off,
                        &self.wave.mid,
                        mid_off,
                        trip.down.right_rows,
                        trip.down.right_cols,
                        trip.down.group_size,
                        trip.down.bits,
                        trip.down.bound,
                    )
                },
            )?;
            tcb.dispatch_threads(
                "q80_hgravs01_factor_matvec",
                (trip.down.left_rows, 1, 1),
                (256, 1, 1),
                |enc| {
                    encode_factor(
                        enc,
                        &trip.down.left_codes,
                        &trip.down.left_scales,
                        &self.wave.mid,
                        mid_off,
                        &self.wave.down,
                        down_off,
                        trip.down.left_rows,
                        trip.down.left_cols,
                        trip.down.group_size,
                        trip.down.bits,
                        trip.down.bound,
                    )
                },
            )?;
            native.hgravs_factor_dispatches = native.hgravs_factor_dispatches.saturating_add(2);
        }
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing);
        let down = read_f32(&self.wave.down, 10 * QWEN80_HIDDEN);
        combined.fill(0.0);
        for slot in 0..10 {
            let weight = weights[slot];
            let base = slot * QWEN80_HIDDEN;
            for dim in 0..QWEN80_HIDDEN {
                combined[dim] += down[base + dim] * weight;
            }
        }
        native.routed_expert_waves = native.routed_expert_waves.saturating_add(1);
        Ok(())
    }
}

pub struct Qwen80MixedHybridDecodeSession {
    catalog: Qwen80MixedStreamingCatalog,
    cache: VectorCache,
    pub state: Qwen80HybridDecodeState,
    pub fallbacks: Qwen80MixedFallbackCounts,
    pub native: Qwen80MixedNativeCounts,
    pub stages: Qwen80MixedStageTimes,
    pub activation_counts: Qwen80ActivationClassCounts,
    pub parity: Qwen80MixedParityReport,
    #[cfg(target_os = "macos")]
    metal: Option<MetalMixedAccel>,
    pub metal_error: Option<String>,
}

impl Qwen80MixedHybridDecodeSession {
    pub fn new(catalog: Qwen80MixedStreamingCatalog, max_seq_len: usize) -> Result<Self> {
        if catalog.tensor_count() != QWEN80_MIXED_EXPECTED_TENSOR_COUNT {
            return Err(mixed_error(format!(
                "catalog tensor count {} != {QWEN80_MIXED_EXPECTED_TENSOR_COUNT}",
                catalog.tensor_count()
            )));
        }
        #[cfg(target_os = "macos")]
        let (metal, metal_error) = match MetalMixedAccel::new() {
            Ok(accel) => (Some(accel), None),
            Err(error) => (None, Some(error.to_string())),
        };
        #[cfg(not(target_os = "macos"))]
        let metal_error = Some("mixed hybrid decode requires macOS Metal".to_owned());
        let mut session = Self {
            catalog,
            cache: VectorCache::new(),
            state: Qwen80HybridDecodeState::new(max_seq_len)?,
            fallbacks: Qwen80MixedFallbackCounts::default(),
            native: Qwen80MixedNativeCounts::default(),
            stages: Qwen80MixedStageTimes::default(),
            activation_counts: Qwen80ActivationClassCounts::default(),
            parity: Qwen80MixedParityReport::default(),
            #[cfg(target_os = "macos")]
            metal,
            metal_error,
        };
        session.run_sample_parity()?;
        Ok(session)
    }

    pub fn catalog(&self) -> &Qwen80MixedStreamingCatalog {
        &self.catalog
    }

    pub fn reset_state(&mut self) {
        self.state.reset();
    }

    fn run_sample_parity(&mut self) -> Result<()> {
        #[cfg(not(target_os = "macos"))]
        {
            return Err(mixed_error("sample parity requires Metal"));
        }
        #[cfg(target_os = "macos")]
        {
            if self.metal.is_none() {
                return Err(mixed_error(format!(
                    "Metal is required for mixed generate: {:?}",
                    self.metal_error
                )));
            }
            let samples = [
                (
                    "model.layers.10.mlp.experts.453.gate_proj.weight",
                    0u8,
                    Q80_GATE_COLS,
                    Q80_GATE_ROWS,
                ),
                (
                    "model.layers.10.mlp.experts.453.up_proj.weight",
                    1u8,
                    Q80_GATE_COLS,
                    Q80_GATE_ROWS,
                ),
                (
                    "model.layers.10.mlp.experts.453.down_proj.weight",
                    2u8,
                    Q80_DOWN_COLS,
                    Q80_DOWN_ROWS,
                ),
                (
                    "model.layers.3.self_attn.q_proj.weight",
                    3u8,
                    QWEN80_HIDDEN,
                    0,
                ),
            ];
            let mut report_samples = Vec::new();
            let mut passed = true;
            for (name, codec, cols, expected_rows) in samples {
                let row = self.catalog.require_row(name)?;
                if row.codec != codec {
                    return Err(mixed_error(format!(
                        "parity sample {name} codec {} != {codec}",
                        row.codec
                    )));
                }
                let packed = self.catalog.load_packed(name)?;
                let (rows, packed_cols) = packed.rows_cols()?;
                if packed_cols != cols {
                    return Err(mixed_error(format!(
                        "{name} cols {packed_cols} != {cols}"
                    )));
                }
                if expected_rows != 0 && rows != expected_rows {
                    return Err(mixed_error(format!(
                        "{name} rows {rows} != {expected_rows}"
                    )));
                }
                let input: Vec<f32> = (0..cols)
                    .map(|i| ((i % 17) as f32) * 0.07 - 0.5)
                    .collect();
                let oracle = packed.cpu_matvec(&input)?;
                if oracle.len() != rows {
                    return Err(mixed_error(format!(
                        "{name} oracle rows {} != {rows}",
                        oracle.len()
                    )));
                }
                let mut got = vec![0.0f32; rows];
                self.matvec_named(name, &input, &mut got)?;
                let err = max_abs_error(&oracle, &got);
                let ok = err <= QWEN80_MIXED_NUMERIC_TOL;
                if !ok {
                    passed = false;
                }
                report_samples.push(json!({
                    "tensor": name,
                    "codec": codec,
                    "max_abs_error": err,
                    "tolerance": QWEN80_MIXED_NUMERIC_TOL,
                    "passed": ok,
                    "dense_w_materialized": false,
                }));
                if !ok {
                    return Err(mixed_error(format!(
                        "artifact-oracle parity failed on {name}: max_abs_error={err} > {QWEN80_MIXED_NUMERIC_TOL}"
                    )));
                }
            }
            let norm = self.catalog.load_packed("model.layers.0.input_layernorm.weight")?;
            let decoded = norm.decode_vector_f32()?;
            if decoded.len() != QWEN80_HIDDEN {
                return Err(mixed_error("layernorm vector width drifted"));
            }
            report_samples.push(json!({
                "tensor": "model.layers.0.input_layernorm.weight",
                "codec": 3,
                "vector_elements": decoded.len(),
                "passed": true,
                "note": "1d HGRAVU01 host decode, not a weight GEMV",
            }));
            self.parity = Qwen80MixedParityReport {
                passed,
                samples: report_samples,
                dense_w_materialized: false,
            };
            Ok(())
        }
    }

    fn packed(&self, name: &str) -> Result<MixedPackedTensor> {
        self.catalog.load_packed(name)
    }

    fn matvec_named(&mut self, name: &str, input: &[f32], output: &mut [f32]) -> Result<()> {
        let started = Instant::now();
        let packed = self.packed(name)?;
        #[cfg(target_os = "macos")]
        {
            let Some(metal) = self.metal.as_mut() else {
                return Err(mixed_error(format!(
                    "Metal required for {name}; refusing host mixed matvec"
                )));
            };
            metal.matvec(
                name,
                &packed,
                input,
                output,
                &mut self.native,
                &mut self.stages,
            )?;
            add_secs(&mut self.stages.mixed_matvec_secs, started);
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (packed, started);
            self.fallbacks.host_mixed_matvec = self.fallbacks.host_mixed_matvec.saturating_add(1);
            Err(mixed_error(format!(
                "refusing host mixed matvec for {name}"
            )))
        }
    }

    fn embed(&mut self, token: u32) -> Result<Vec<f32>> {
        if token as usize >= QWEN80_VOCAB {
            return Err(mixed_error(format!(
                "token {token} is outside the embedding vocab"
            )));
        }
        let packed = self.packed("model.embed_tokens.weight")?;
        let hidden = packed.gather_row(token as usize)?;
        if hidden.len() != QWEN80_HIDDEN {
            return Err(mixed_error("embedding row width drifted"));
        }
        self.fallbacks.host_q8_embed_gather =
            self.fallbacks.host_q8_embed_gather.saturating_add(1);
        Ok(hidden)
    }

    fn vector(&mut self, name: &str) -> Result<Vec<f32>> {
        if let Some(existing) = self.cache.vectors.get(name) {
            return Ok(existing.clone());
        }
        let packed = self.packed(name)?;
        let values = packed.decode_vector_f32()?;
        self.fallbacks.host_q8_vector_decode =
            self.fallbacks.host_q8_vector_decode.saturating_add(1);
        self.cache.vectors.insert(name.to_owned(), values.clone());
        Ok(values)
    }

    fn layer_name(layer: usize, suffix: &str) -> String {
        format!("model.layers.{layer}.{suffix}")
    }

    fn mlp(
        &mut self,
        gate_name: &str,
        up_name: &str,
        down_name: &str,
        input: &[f32],
        intermediate: usize,
    ) -> Result<Vec<f32>> {
        let mut gate = vec![0.0f32; intermediate];
        let mut up = vec![0.0f32; intermediate];
        let mut act = vec![0.0f32; intermediate];
        let mut down = vec![0.0f32; QWEN80_HIDDEN];
        let sandwich = Instant::now();
        self.matvec_named(gate_name, input, &mut gate)?;
        self.matvec_named(up_name, input, &mut up)?;
        let silu_started = Instant::now();
        silu_mul(&gate, &up, &mut act);
        add_secs(&mut self.stages.activation.shared_swiglu_secs, silu_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.shared_swiglu =
            self.activation_counts.shared_swiglu.saturating_add(1);
        self.matvec_named(down_name, &act, &mut down)?;
        add_secs(
            &mut self.stages.activation.shared_mlp_sandwich_secs,
            sandwich,
        );
        Ok(down)
    }

    fn deltanet_mixer(&mut self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let slot = self.state.linear_slot_for_layer(layer)?;
        let input_w = self.vector(&Self::layer_name(layer, "input_layernorm.weight"))?;
        let rms_started = Instant::now();
        let rms = source_qwen80_residual_rms_norm(hidden, &input_w)?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            rms_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        let qkvz_rows = layout.qkvz_projection_elements()?;
        let ba_rows = layout.ba_projection_elements()?;
        let mut projected_qkvz = vec![0.0f32; qkvz_rows];
        let mut projected_ba = vec![0.0f32; ba_rows];
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
            &rms,
            &mut projected_qkvz,
        )?;
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.in_proj_ba.weight"),
            &rms,
            &mut projected_ba,
        )?;
        let (raw_query, raw_key, raw_value, z) =
            source_qwen80_split_linear_qkvz(&projected_qkvz, &layout)?;
        let mut mixed_qkv = Vec::with_capacity(layout.conv_channels);
        mixed_qkv.extend_from_slice(&raw_query);
        mixed_qkv.extend_from_slice(&raw_key);
        mixed_qkv.extend_from_slice(&raw_value);
        let conv_w = self.vector(&Self::layer_name(layer, "linear_attn.conv1d.weight"))?;
        let conv_started = Instant::now();
        let (convolved_qkv, next_conv) = source_qwen80_causal_conv_step_dense(
            &mixed_qkv,
            &self.state.linear_conv[slot],
            &conv_w,
            &layout,
        )?;
        add_secs(&mut self.stages.activation.deltanet_conv_secs, conv_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.deltanet_conv =
            self.activation_counts.deltanet_conv.saturating_add(1);
        let raw_query_len = layout.key_elements()?;
        let raw_value_len = layout.value_elements()?;
        let convolved_query = &convolved_qkv[..raw_query_len];
        let convolved_key = &convolved_qkv[raw_query_len..raw_query_len + raw_query_len];
        let convolved_value = convolved_qkv[raw_query_len + raw_query_len..].to_vec();
        if convolved_value.len() != raw_value_len {
            return Err(mixed_error("DeltaNet convolution value geometry drifted"));
        }
        let mut repeated_query = vec![0.0f32; raw_value_len];
        let mut repeated_key = vec![0.0f32; raw_value_len];
        for value_head in 0..layout.value_heads {
            let key_head = value_head / layout.value_heads_per_key_head;
            let mut query_head = convolved_query
                [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
                .to_vec();
            let mut key_head_values = convolved_key
                [key_head * layout.key_head_dim..(key_head + 1) * layout.key_head_dim]
                .to_vec();
            source_qwen80_l2_normalize(
                &mut query_head,
                (layout.key_head_dim as f32).sqrt().recip(),
            )?;
            source_qwen80_l2_normalize(&mut key_head_values, 1.0)?;
            let destination = value_head * layout.key_head_dim;
            repeated_query[destination..destination + layout.key_head_dim]
                .copy_from_slice(&query_head);
            repeated_key[destination..destination + layout.key_head_dim]
                .copy_from_slice(&key_head_values);
        }
        let a_log = self.vector(&Self::layer_name(layer, "linear_attn.A_log"))?;
        let dt_bias = self.vector(&Self::layer_name(layer, "linear_attn.dt_bias"))?;
        let recurrent_started = Instant::now();
        let (decay, beta) =
            source_qwen80_ba_to_decay_beta(&projected_ba, &a_log, &dt_bias, &layout)?;
        let recurrent_output = source_qwen80_recurrent_deltanet(
            &mut self.state.linear_recurrent[slot],
            &repeated_query,
            &repeated_key,
            &convolved_value,
            &decay,
            &beta,
            &layout,
        )?;
        add_secs(
            &mut self.stages.activation.deltanet_recurrent_secs,
            recurrent_started,
        );
        self.state.linear_conv[slot] = next_conv;
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.deltanet_recurrent =
            self.activation_counts.deltanet_recurrent.saturating_add(1);
        let gated_norm = self.vector(&Self::layer_name(layer, "linear_attn.norm.weight"))?;
        let repeated_gated_norm = (0..layout.value_heads)
            .flat_map(|_| gated_norm.iter().copied())
            .collect::<Vec<_>>();
        let gated_started = Instant::now();
        let gated_output = source_qwen80_gated_rms_norm(
            &recurrent_output,
            &z,
            &repeated_gated_norm,
            layout.value_heads,
            layout.value_head_dim,
        )?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            gated_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(
            &Self::layer_name(layer, "linear_attn.out_proj.weight"),
            &gated_output,
            &mut mixer_output,
        )?;
        let mut residual = hidden.to_vec();
        let add_started = Instant::now();
        add_inplace(&mut residual, &mixer_output);
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            add_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(mixed_error(format!(
                "layer {layer} DeltaNet residual is non-finite"
            )));
        }
        Ok(residual)
    }

    fn gqa_mixer(&mut self, layer: usize, hidden: &[f32]) -> Result<Vec<f32>> {
        let layout = Qwen80CanonicalGqaLayout::source_exact();
        let slot = self.state.gqa_slot_for_layer(layer)?;
        let position = self.state.position;
        if position >= self.state.max_seq_len {
            return Err(mixed_error(format!(
                "GQA position {position} exceeds max_seq_len {}",
                self.state.max_seq_len
            )));
        }
        let input_w = self.vector(&Self::layer_name(layer, "input_layernorm.weight"))?;
        let rms_started = Instant::now();
        let rms = source_qwen80_residual_rms_norm(hidden, &input_w)?;
        add_secs(
            &mut self.stages.activation.gqa_input_layernorm_secs,
            rms_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.gqa_input_layernorm =
            self.activation_counts.gqa_input_layernorm.saturating_add(1);
        let mut q_projection = vec![0.0f32; layout.q_proj_rows];
        let mut k_projection = vec![0.0f32; layout.kv_dim];
        let mut v_projection = vec![0.0f32; layout.kv_dim];
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.q_proj.weight"),
            &rms,
            &mut q_projection,
        )?;
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.k_proj.weight"),
            &rms,
            &mut k_projection,
        )?;
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.v_proj.weight"),
            &rms,
            &mut v_projection,
        )?;
        let q_norm = self.vector(&Self::layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = self.vector(&Self::layer_name(layer, "self_attn.k_norm.weight"))?;
        let query_raw = qwen80_gqa_query_from_interleaved_q_projection(&q_projection, &layout)?;
        let rope_started = Instant::now();
        let query = qwen80_gqa_source_norm_rope(
            &query_raw,
            &q_norm,
            layout.query_heads,
            layout.head_dim,
            layout.rotary_dim,
            position,
            "GQA q_norm + partial RoPE",
        )?;
        let key_row = qwen80_gqa_source_norm_rope(
            &k_projection,
            &k_norm,
            layout.key_value_heads,
            layout.head_dim,
            layout.rotary_dim,
            position,
            "GQA k_norm + partial RoPE",
        )?;
        add_secs(&mut self.stages.activation.gqa_norm_rope_secs, rope_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        self.activation_counts.gqa_norm_rope =
            self.activation_counts.gqa_norm_rope.saturating_add(2);
        let start = position * layout.kv_dim;
        let end = start + layout.kv_dim;
        self.state.gqa_key[slot][start..end].copy_from_slice(&key_row);
        self.state.gqa_value[slot][start..end].copy_from_slice(&v_projection);
        let attn_started = Instant::now();
        let attention = qwen80_gqa_causal_attention(
            &query,
            &self.state.gqa_key[slot],
            &self.state.gqa_value[slot],
            position + 1,
            &layout,
        )?;
        let gated = qwen80_gqa_apply_sigmoid_gate(&attention, &q_projection, &layout)?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            attn_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(2);
        let mut mixer_output = vec![0.0f32; QWEN80_HIDDEN];
        self.matvec_named(
            &Self::layer_name(layer, "self_attn.o_proj.weight"),
            &gated,
            &mut mixer_output,
        )?;
        let mut residual = hidden.to_vec();
        let add_started = Instant::now();
        add_inplace(&mut residual, &mixer_output);
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            add_started,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        if residual.iter().any(|value| !value.is_finite()) {
            return Err(mixed_error(format!(
                "layer {layer} GQA residual is non-finite"
            )));
        }
        Ok(residual)
    }

    fn moe_suffix(&mut self, layer: usize, first_residual: &[f32]) -> Result<Vec<f32>> {
        let norm_started = Instant::now();
        let post_w = self.vector(&Self::layer_name(layer, "post_attention_layernorm.weight"))?;
        let norm_op = Instant::now();
        let router_input = source_qwen80_residual_rms_norm(first_residual, &post_w)?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            norm_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        add_secs(&mut self.stages.moe_norm_router_secs, norm_started);

        let shared_started = Instant::now();
        let shared = self.mlp(
            &Self::layer_name(layer, "mlp.shared_expert.gate_proj.weight"),
            &Self::layer_name(layer, "mlp.shared_expert.up_proj.weight"),
            &Self::layer_name(layer, "mlp.shared_expert.down_proj.weight"),
            &router_input,
            QWEN80_MOE_INTERMEDIATE,
        )?;
        add_secs(&mut self.stages.moe_shared_secs, shared_started);

        let router_started = Instant::now();
        let mut router_logits = vec![0.0f32; QWEN80_EXPERTS];
        self.matvec_named(
            &Self::layer_name(layer, "mlp.gate.weight"),
            &router_input,
            &mut router_logits,
        )?;
        let route_op = Instant::now();
        let route = source_qwen80_topk_router(&router_logits)?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            route_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        add_secs(&mut self.stages.moe_norm_router_secs, router_started);

        let mut combined = vec![0.0f32; QWEN80_HIDDEN];
        let routed_started = Instant::now();
        #[cfg(target_os = "macos")]
        {
            let Some(metal) = self.metal.as_mut() else {
                return Err(mixed_error("Metal required for routed mixed experts"));
            };
            metal.routed_wave(
                &self.catalog,
                layer,
                &route.ids,
                &route.weights,
                &router_input,
                &mut combined,
                &mut self.native,
                &mut self.stages,
            )?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            self.fallbacks.host_expert_payload_bind = self
                .fallbacks
                .host_expert_payload_bind
                .saturating_add(30);
            return Err(mixed_error("refusing host mixed expert path"));
        }
        add_secs(&mut self.stages.moe_routed_secs, routed_started);

        let combine_started = Instant::now();
        let mut gate_logit = [0.0f32; 1];
        self.matvec_named(
            &Self::layer_name(layer, "mlp.shared_expert_gate.weight"),
            &router_input,
            &mut gate_logit,
        )?;
        let gate_val = 1.0 / (1.0 + (-gate_logit[0]).exp());
        if !gate_val.is_finite() || !(0.0..=1.0).contains(&gate_val) {
            return Err(mixed_error(format!(
                "layer {layer} shared-expert gate sigmoid is invalid"
            )));
        }
        let combine_op = Instant::now();
        for (dst, value) in combined.iter_mut().zip(shared) {
            *dst += value * gate_val;
        }
        let mut out = first_residual.to_vec();
        add_inplace(&mut out, &combined);
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            combine_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        add_secs(&mut self.stages.moe_combine_secs, combine_started);
        if out.iter().any(|value| !value.is_finite()) {
            return Err(mixed_error(format!(
                "layer {layer} second residual is non-finite"
            )));
        }
        Ok(out)
    }

    fn terminal_greedy(&mut self, hidden: &[f32]) -> Result<u32> {
        let norm_w = self.vector("model.norm.weight")?;
        let norm_op = Instant::now();
        let normed = source_qwen80_residual_rms_norm(hidden, &norm_w)?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            norm_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        let mut logits = vec![0.0f32; QWEN80_VOCAB];
        self.matvec_named("lm_head.weight", &normed, &mut logits)?;
        for logit in logits.iter_mut().skip(QWEN80_TOKENIZER_VOCAB) {
            *logit = f32::NEG_INFINITY;
        }
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().take(QWEN80_TOKENIZER_VOCAB).enumerate() {
            if value > best_v || (value == best_v && index < best_i) {
                best_v = value;
                best_i = index;
            }
        }
        self.fallbacks.host_sample = self.fallbacks.host_sample.saturating_add(1);
        if !best_v.is_finite() {
            return Err(mixed_error("greedy sample saw no finite logit"));
        }
        Ok(best_i as u32)
    }

    pub fn forward_token(&mut self, token: u32) -> Result<u32> {
        if self.state.position >= self.state.max_seq_len {
            return Err(mixed_error(format!(
                "decode position {} exceeds max_seq_len {}",
                self.state.position, self.state.max_seq_len
            )));
        }
        let embed_started = Instant::now();
        let mut hidden = self.embed(token)?;
        add_secs(&mut self.stages.embed_secs, embed_started);
        for layer in 0..QWEN80_LAYERS {
            let first = match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => {
                    let started = Instant::now();
                    let value = self.deltanet_mixer(layer, &hidden)?;
                    add_secs(&mut self.stages.deltanet_secs, started);
                    value
                }
                Qwen80LayerKind::FullAttention => {
                    let started = Instant::now();
                    let value = self.gqa_mixer(layer, &hidden)?;
                    add_secs(&mut self.stages.gqa_secs, started);
                    value
                }
            };
            hidden = self.moe_suffix(layer, &first)?;
        }
        let terminal_started = Instant::now();
        let sampled = self.terminal_greedy(&hidden)?;
        add_secs(&mut self.stages.terminal_secs, terminal_started);
        self.state.position = self.state.position.saturating_add(1);
        require_rss_cap("after mixed hybrid token")?;
        if self.fallbacks.silent_or_invalid() != 0 {
            return Err(mixed_error(format!(
                "silent mixed fallbacks are invalid: {:?}",
                self.fallbacks
            )));
        }
        Ok(sampled)
    }
}

#[derive(Clone, Debug)]
pub struct Qwen80MixedGreedyResult {
    pub prompt: String,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub prefill_secs: f64,
    pub first_token_latency_secs: f64,
    pub decode_secs: f64,
    pub steady_state_decode_secs: f64,
    pub steady_state_tokens: usize,
    pub steady_state_tok_s: f64,
    pub wall_ns_per_token: f64,
    pub gpu_matvec_ns_per_token: f64,
    pub peak_rss_bytes: u64,
    pub fallbacks: Qwen80MixedFallbackCounts,
    pub native: Qwen80MixedNativeCounts,
    pub stages: Qwen80MixedStageTimes,
    pub activation_counts: Qwen80ActivationClassCounts,
    pub complete_physical_bpw: f64,
    pub claim: &'static str,
    pub metal_error: Option<String>,
    pub parity: Qwen80MixedParityReport,
    pub dense_w_materialized: bool,
}

pub fn generate_mixed_greedy(
    session: &mut Qwen80MixedHybridDecodeSession,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
) -> Result<Qwen80MixedGreedyResult> {
    if max_new_tokens == 0 {
        return Err(mixed_error("max_new_tokens must be positive"));
    }
    let prompt_token_ids = tokenizer.encode(prompt, false)?;
    if prompt_token_ids.is_empty() {
        return Err(mixed_error("prompt tokenization produced no tokens"));
    }
    if prompt_token_ids.len() + max_new_tokens > session.state.max_seq_len {
        return Err(mixed_error(
            "prompt + max_new_tokens exceeds session max_seq_len",
        ));
    }
    session.reset_state();
    session.fallbacks = Qwen80MixedFallbackCounts::default();
    session.native = Qwen80MixedNativeCounts::default();
    session.stages = Qwen80MixedStageTimes::default();
    let prefill_started = Instant::now();
    let mut next = 0u32;
    for &token in prompt_token_ids.iter() {
        next = session.forward_token(token)?;
    }
    let prefill_secs = prefill_started.elapsed().as_secs_f64();
    let mut generated = Vec::with_capacity(max_new_tokens);
    generated.push(next);
    let decode_started = Instant::now();
    let mut steady_started = None;
    for _ in 1..max_new_tokens {
        if tokenizer.is_eog(next) {
            break;
        }
        if steady_started.is_none() {
            steady_started = Some(Instant::now());
        }
        next = session.forward_token(next)?;
        generated.push(next);
    }
    let decode_secs = decode_started.elapsed().as_secs_f64();
    let steady_state_tokens = generated.len().saturating_sub(1);
    let steady_state_decode_secs = steady_started
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let steady_state_tok_s = if steady_state_tokens == 0 || steady_state_decode_secs <= 0.0 {
        0.0
    } else {
        steady_state_tokens as f64 / steady_state_decode_secs
    };
    let generated_text = tokenizer.decode(&generated, true)?;
    let wall_tokens = if steady_state_tokens > 0 {
        steady_state_tokens as f64
    } else {
        generated.len().max(1) as f64
    };
    let wall_denom = if steady_state_tokens > 0 {
        steady_state_decode_secs
    } else {
        decode_secs.max(prefill_secs)
    };
    let wall_ns_per_token = if wall_denom > 0.0 {
        (wall_denom / wall_tokens) * 1.0e9
    } else {
        0.0
    };
    let gpu_matvec_ns_per_token = session.stages.gpu_matvec_ns as f64 / wall_tokens;
    Ok(Qwen80MixedGreedyResult {
        prompt: prompt.to_owned(),
        prompt_token_ids,
        generated_token_ids: generated,
        generated_text,
        prefill_secs,
        first_token_latency_secs: prefill_secs,
        decode_secs,
        steady_state_decode_secs,
        steady_state_tokens,
        steady_state_tok_s,
        wall_ns_per_token,
        gpu_matvec_ns_per_token,
        peak_rss_bytes: peak_rss_bytes(),
        fallbacks: session.fallbacks.clone(),
        native: session.native.clone(),
        stages: session.stages.clone(),
        activation_counts: session.activation_counts.clone(),
        complete_physical_bpw: session.catalog.complete_physical_bpw,
        claim: QWEN80_MIXED_CLAIM,
        metal_error: session.metal_error.clone(),
        parity: session.parity.clone(),
        dense_w_materialized: false,
    })
}

pub fn discover_qwen80_mixed_root() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(Qwen80MixedStreamingCatalog::default_root_hint()),
        PathBuf::from(MIXED_DEFAULT_ROOT_ABS),
    ];
    candidates.into_iter().find(|path| {
        path.join(QWEN80_MIXED_MANIFEST_NAME).is_file() && path.join("catalog.hq80m15").is_file()
    })
}

pub fn load_mixed_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    load_qwen80_tokenizer(path)
}
