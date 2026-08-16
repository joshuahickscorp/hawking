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
//! kernel. A dense `W` is never allocated on this path. Occupancy tiles
//! (`HAWKING_Q80_RECON_FUSE`, default on) delete the serial bit-loop
//! reconstruct: binary tg256, fused binary+CSR, 3-bit 8-unpack, Q8 byte
//! load. Serial 1-thread/row remains behind `HAWKING_Q80_RECON_FUSE=0`.

use super::qwen80_complete_runtime::{
    qwen80_gqa_apply_sigmoid_gate, qwen80_gqa_causal_attention,
    qwen80_gqa_query_from_interleaved_q_projection, qwen80_gqa_source_norm_rope, qwen80_layer_kind,
    source_qwen80_ba_to_decay_beta, source_qwen80_causal_conv_step_dense,
    source_qwen80_gated_rms_norm, source_qwen80_l2_normalize, source_qwen80_recurrent_deltanet,
    source_qwen80_residual_rms_norm, source_qwen80_split_linear_qkvz, source_qwen80_topk_router,
    Qwen80CanonicalGqaLayout, Qwen80CanonicalLinearDeltaNetLayout, Qwen80LayerKind, QWEN80_EXPERTS,
    QWEN80_HIDDEN, QWEN80_LAYERS, QWEN80_MOE_INTERMEDIATE, QWEN80_RMS_EPS, QWEN80_TOKENIZER_VOCAB,
    QWEN80_VOCAB,
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
    expand_rice_indices, max_abs_error, mixed_gpu_layout, rice_q1_row_ptr, MixedGpuKind,
    MixedPackedTensor, RiceQ1Packed, Q80_DOWN_COLS, Q80_DOWN_ROWS, Q80_GATE_COLS,
    Q80_GATE_ROWS, Q80_HGRAVS_BITS, Q80_HGRAVS_GROUP_SIZE, Q80_HGRAVS_RANK,
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

/// Default **on**. Skip per-tensor SHA after session admit and pin each
/// layer segment as one no-copy MTLBuffer. Set `HAWKING_Q80_HOST_FACET1=0`
/// to restore the copy+rehash bind path for A/B.
pub fn qwen80_host_facet1_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_HOST_FACET1")
}

/// Default **on**. Merge independent matvecs into one command buffer and
/// keep the routed wave on one fence. Set `HAWKING_Q80_HOST_FACET2=0` to
/// restore one-CB-per-matvec for A/B.
pub fn qwen80_host_facet2_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_HOST_FACET2")
}

const QWEN80_DEVICE_ROPE_THETA: f32 = 5_000_000.0;

/// Default **on**. Keep residual / mixer / SwiGLU / DeltaNet / GQA /
/// RMSNorm on device so a layer is prefix CB + suffix CB (or fused).
/// Set `HAWKING_Q80_DEVICE_ACTIVATIONS=0` to restore the host-activation
/// 7-CB/layer path for A/B.
pub fn qwen80_mixed_device_activations_enabled() -> bool {
    for key in [
        "HAWKING_Q80_DEVICE_ACTIVATIONS",
        "HAWKING_QWEN80_DEVICE_ACTIVATIONS",
        "HAWKING_QWEN80_DEVICE_ACT",
    ] {
        if let Ok(raw) = std::env::var(key) {
            let trimmed = raw.trim();
            if trimmed.eq_ignore_ascii_case("0")
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off")
                || trimmed.eq_ignore_ascii_case("no")
            {
                return false;
            }
            if !trimmed.is_empty() {
                return true;
            }
        }
    }
    true
}

/// Default **on**. Fuse suffix of layer i with mixer+prefix of layer i+1
/// so the token is ~49 CBs (one wait per router readback + terminal).
/// Set `HAWKING_Q80_COLLAPSE_FUSE=0` for the 2-wait topology
/// (prefix CB, suffix CB, ~97 CBs).
pub fn qwen80_mixed_collapse_fuse_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_COLLAPSE_FUSE")
}

/// Readbacks that still force a command-buffer split after device
/// activations. A named leftover is acceptable; an unexplained floor is not.
pub fn qwen80_mixed_remaining_cb_split_readbacks() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "router_logits_readback",
            "512 f32 router logits must return to the host so source_qwen80_topk_router can pick 10 experts and bind their mixed binary/CSR/HGRAVS payloads. A device top-10 does not remove the split: mixed expert buffers are still bound from host-known IDs. A bindless mixed expert table would live in qwen80_device_expert_table.rs, which this lane cannot edit.",
        ),
        (
            "lm_head_logits_readback",
            "Host greedy argmax needs the vocab logit vector after the terminal CB. This is one wait at the end of the token, not a per-layer split. The next token's embed is still a host Q8 row gather, so the sampled id must cross back to the CPU.",
        ),
    ]
}

/// Default **on**. Consume packed codes in-register on occupancy tiles:
/// binary tg256, fused binary+CSR, 3-bit simd3, Q8 byte extract. Set
/// `HAWKING_Q80_RECON_FUSE=0` to restore the serial 1-thread/row path
/// (the 863 ms gpu_matvec baseline).
pub fn qwen80_recon_fuse_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_RECON_FUSE")
}

/// Default **off**. Dispatch G023 `gk_*_simd` (1 simdgroup / row, wide
/// extract) instead of the recon-fuse occupancy tiles. The 8-45x in the
/// discriminator was vs serial bit-walk; this is the A/B against the
/// tiles that already deleted that walk. Residual CSR stays on the
/// fused q80 kernel — there is no family equivalent.
pub fn qwen80_gk_simd_enabled() -> bool {
    crate::env_on("HAWKING_Q80_GK_SIMD")
}

/// Default **on**. After `ensure_named_weight` the session already has
/// rows/cols (and a resident MTLBuffer). Skip `catalog.load_packed` on
/// the GEMV path and reuse that geometry. Set `HAWKING_Q80_CACHE_GEOM=0`
/// to restore per-GEMV mmap-header reparse for A/B.
pub fn qwen80_cache_geom_enabled() -> bool {
    crate::env_opt_out("HAWKING_Q80_CACHE_GEOM")
}

fn tg256_grid(rows: u32) -> (u32, u32, u32) {
    (rows.saturating_mul(256), 1, 1)
}

fn simd8_grid(rows: u32) -> (u32, u32, u32) {
    (rows.div_ceil(8).saturating_mul(256).max(256), 1, 1)
}

/// Requested, logged capability-bar probe. Not a silent fallback.
/// Identity (rank 160, mix 1) keeps the fused incumbent generate path.
#[derive(Clone, Debug, Serialize)]
pub struct MixedDegradeConfig {
    pub hgravs_rank_cap: u32,
    pub gate_mix: f32,
    pub up_mix: f32,
    pub down_mix: f32,
    pub mix_seed: u64,
}

impl Default for MixedDegradeConfig {
    fn default() -> Self {
        Self {
            hgravs_rank_cap: Q80_HGRAVS_RANK as u32,
            gate_mix: 1.0,
            up_mix: 1.0,
            down_mix: 1.0,
            mix_seed: 0xC0B1_7C11,
        }
    }
}

impl MixedDegradeConfig {
    pub fn is_identity(&self) -> bool {
        self.hgravs_rank_cap >= Q80_HGRAVS_RANK as u32
            && (self.gate_mix - 1.0).abs() < 1.0e-6
            && (self.up_mix - 1.0).abs() < 1.0e-6
            && (self.down_mix - 1.0).abs() < 1.0e-6
    }

    pub fn rank_cap(&self) -> usize {
        (self.hgravs_rank_cap as usize).clamp(1, Q80_HGRAVS_RANK)
    }
}

fn mix_seed(base: u64, layer: usize, expert: u16, organ: u8) -> u64 {
    base.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add((layer as u64) << 32)
        .wrapping_add((expert as u64) << 8)
        .wrapping_add(u64::from(organ))
}

/// y <- α y + sqrt(1-α²) n, with n same-energy and orthogonalized against y.
/// Gives cos(y_orig, y_new) = α when ||y|| > 0.
pub fn mix_matched_cosine(y: &mut [f32], alpha: f32, seed: u64) {
    if y.is_empty() || (alpha - 1.0).abs() < 1.0e-6 {
        return;
    }
    let alpha = alpha.clamp(-1.0, 1.0) as f64;
    let mut rng = if seed == 0 { 1 } else { seed };
    let mut noise = vec![0.0f64; y.len()];
    let mut energy_y = 0.0f64;
    let mut energy_n = 0.0f64;
    let mut dot = 0.0f64;
    for (i, &yi) in y.iter().enumerate() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        let u = ((rng >> 11) as f64) * (1.0 / ((1u64 << 53) as f64)) * 2.0 - 1.0;
        let yv = yi as f64;
        noise[i] = u;
        energy_y += yv * yv;
        energy_n += u * u;
        dot += yv * u;
    }
    if energy_y > 1.0e-20 {
        let scale = dot / energy_y;
        energy_n = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            let v = noise[i] - scale * (yi as f64);
            noise[i] = v;
            energy_n += v * v;
        }
    }
    let ny = energy_y.sqrt();
    let nn = energy_n.sqrt();
    if ny < 1.0e-20 || nn < 1.0e-20 {
        return;
    }
    let beta = (1.0 - alpha * alpha).max(0.0).sqrt();
    let n_scale = beta * (ny / nn);
    for (slot, dest) in y.iter_mut().enumerate() {
        *dest = (alpha * (*dest as f64) + n_scale * noise[slot]) as f32;
    }
}

fn mixed_error(message: impl Into<String>) -> Error {
    Error::Model(format!("qwen80 mixed hybrid decode: {}", message.into()))
}

fn add_secs(slot: &mut f64, started: Instant) {
    *slot += started.elapsed().as_secs_f64();
}

fn add_ns(slot: &mut u64, started: Instant) {
    *slot = slot.saturating_add(started.elapsed().as_nanos() as u64);
}

fn add_elapsed_bind(stages: &mut Qwen80MixedStageTimes, started: Instant) {
    let ns = started.elapsed().as_nanos() as u64;
    stages.host_expert_bind_ns = stages.host_expert_bind_ns.saturating_add(ns);
    stages.host_expert_bind_secs += ns as f64 / 1.0e9;
    stages.host_excl.expert_bind = stages.host_excl.expert_bind.saturating_add(ns);
}

/// Production GPU organ for exclusive TOKEN_NS attribution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedGpuOrgan {
    DeltaNet,
    Gqa,
    MoeShared,
    MoeRouted,
    MoeRouter,
    MoeCombineGate,
    Terminal,
    Other,
}

/// Isolated probe / alternate kernel for one uploaded weight.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MixedProbeMode {
    Full,
    Addr,
    Decode,
    BinarySimd,
    GkBinarySimd,
    Q8SimdBytes,
    GkHgravsSimd,
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
    pub compute_dispatches: u64,
    pub expert_nocopy_binds: u64,
    pub expert_copy_binds: u64,
    pub segment_nocopy_binds: u64,
    pub packed_calls: u64,
    pub packed_skipped: u64,
    pub device_activation_dispatches: u64,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedGpuOrganNs {
    pub deltanet: u64,
    pub gqa: u64,
    pub moe_shared: u64,
    pub moe_routed: u64,
    pub moe_router: u64,
    pub moe_combine_gate: u64,
    pub terminal: u64,
    pub other: u64,
}

impl MixedGpuOrganNs {
    pub fn sum(self) -> u64 {
        self.deltanet
            .saturating_add(self.gqa)
            .saturating_add(self.moe_shared)
            .saturating_add(self.moe_routed)
            .saturating_add(self.moe_router)
            .saturating_add(self.moe_combine_gate)
            .saturating_add(self.terminal)
            .saturating_add(self.other)
    }

    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            deltanet: self.deltanet.saturating_sub(rhs.deltanet),
            gqa: self.gqa.saturating_sub(rhs.gqa),
            moe_shared: self.moe_shared.saturating_sub(rhs.moe_shared),
            moe_routed: self.moe_routed.saturating_sub(rhs.moe_routed),
            moe_router: self.moe_router.saturating_sub(rhs.moe_router),
            moe_combine_gate: self.moe_combine_gate.saturating_sub(rhs.moe_combine_gate),
            terminal: self.terminal.saturating_sub(rhs.terminal),
            other: self.other.saturating_sub(rhs.other),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedHostExclusiveNs {
    pub embed: u64,
    pub dn_rms: u64,
    pub dn_rearrange_l2: u64,
    pub dn_conv: u64,
    pub dn_recurrent: u64,
    pub dn_gated: u64,
    pub dn_residual: u64,
    pub gqa_rms: u64,
    pub gqa_interleave: u64,
    pub gqa_rope: u64,
    pub gqa_kv_copy: u64,
    pub gqa_attn: u64,
    pub gqa_residual: u64,
    pub post_rms: u64,
    pub final_rms: u64,
    pub silu: u64,
    pub topk: u64,
    pub combine: u64,
    pub argmax: u64,
    pub buffer_prep: u64,
    pub expert_bind: u64,
    pub catalog_reparse: u64,
    pub vector_clone: u64,
}

impl MixedHostExclusiveNs {
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            embed: self.embed.saturating_sub(rhs.embed),
            dn_rms: self.dn_rms.saturating_sub(rhs.dn_rms),
            dn_rearrange_l2: self.dn_rearrange_l2.saturating_sub(rhs.dn_rearrange_l2),
            dn_conv: self.dn_conv.saturating_sub(rhs.dn_conv),
            dn_recurrent: self.dn_recurrent.saturating_sub(rhs.dn_recurrent),
            dn_gated: self.dn_gated.saturating_sub(rhs.dn_gated),
            dn_residual: self.dn_residual.saturating_sub(rhs.dn_residual),
            gqa_rms: self.gqa_rms.saturating_sub(rhs.gqa_rms),
            gqa_interleave: self.gqa_interleave.saturating_sub(rhs.gqa_interleave),
            gqa_rope: self.gqa_rope.saturating_sub(rhs.gqa_rope),
            gqa_kv_copy: self.gqa_kv_copy.saturating_sub(rhs.gqa_kv_copy),
            gqa_attn: self.gqa_attn.saturating_sub(rhs.gqa_attn),
            gqa_residual: self.gqa_residual.saturating_sub(rhs.gqa_residual),
            post_rms: self.post_rms.saturating_sub(rhs.post_rms),
            final_rms: self.final_rms.saturating_sub(rhs.final_rms),
            silu: self.silu.saturating_sub(rhs.silu),
            topk: self.topk.saturating_sub(rhs.topk),
            combine: self.combine.saturating_sub(rhs.combine),
            argmax: self.argmax.saturating_sub(rhs.argmax),
            buffer_prep: self.buffer_prep.saturating_sub(rhs.buffer_prep),
            expert_bind: self.expert_bind.saturating_sub(rhs.expert_bind),
            catalog_reparse: self.catalog_reparse.saturating_sub(rhs.catalog_reparse),
            vector_clone: self.vector_clone.saturating_sub(rhs.vector_clone),
        }
    }

    pub fn deltanet_host(self) -> u64 {
        self.dn_rearrange_l2
            .saturating_add(self.dn_conv)
            .saturating_add(self.dn_recurrent)
            .saturating_add(self.dn_gated)
            .saturating_add(self.dn_residual)
    }

    pub fn gqa_host(self) -> u64 {
        self.gqa_interleave
            .saturating_add(self.gqa_rope)
            .saturating_add(self.gqa_attn)
            .saturating_add(self.gqa_residual)
    }

    pub fn normalization_host(self) -> u64 {
        self.dn_rms
            .saturating_add(self.gqa_rms)
            .saturating_add(self.post_rms)
            .saturating_add(self.final_rms)
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MixedExclusiveSnap {
    pub encode_ns: u64,
    pub submit_ns: u64,
    pub wait_ns: u64,
    pub gpu_ns: u64,
    pub wait_minus_gpu_ns: u64,
    pub cbs: u64,
    pub dispatches: u64,
    pub timestamps_missing: u64,
    pub gpu_organ: MixedGpuOrganNs,
    pub host_excl: MixedHostExclusiveNs,
}

impl MixedExclusiveSnap {
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self {
            encode_ns: self.encode_ns.saturating_sub(rhs.encode_ns),
            submit_ns: self.submit_ns.saturating_sub(rhs.submit_ns),
            wait_ns: self.wait_ns.saturating_sub(rhs.wait_ns),
            gpu_ns: self.gpu_ns.saturating_sub(rhs.gpu_ns),
            wait_minus_gpu_ns: self.wait_minus_gpu_ns.saturating_sub(rhs.wait_minus_gpu_ns),
            cbs: self.cbs.saturating_sub(rhs.cbs),
            dispatches: self.dispatches.saturating_sub(rhs.dispatches),
            timestamps_missing: self
                .timestamps_missing
                .saturating_sub(rhs.timestamps_missing),
            gpu_organ: self.gpu_organ.saturating_sub(rhs.gpu_organ),
            host_excl: self.host_excl.saturating_sub(rhs.host_excl),
        }
    }

    /// encode + buffer write/read + expert first-touch bind + embed gather
    /// + catalog.load_packed + vector cache clone. Matches the TOKEN_NS
    /// host_preparation composition.
    pub fn host_preparation_ns(&self) -> u64 {
        self.encode_ns
            .saturating_add(self.host_excl.buffer_prep)
            .saturating_add(self.host_excl.expert_bind)
            .saturating_add(self.host_excl.embed)
            .saturating_add(self.host_excl.catalog_reparse)
            .saturating_add(self.host_excl.vector_clone)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MixedTokenSample {
    pub position: u32,
    pub kind: &'static str,
    pub wall_ns: u64,
    pub snap: MixedExclusiveSnap,
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
    pub host_expert_bind_secs: f64,
    pub host_expert_bind_ns: u64,
    #[serde(skip)]
    pub activation: Qwen80ActivationClassTimes,
    pub gpu_matvec_ns: u64,
    pub gpu_matvec_timestamps_missing: u64,
    pub cb_wait_ns: u64,
    pub cb_submit_ns: u64,
    pub cb_encode_ns: u64,
    pub cb_wait_minus_gpu_ns: u64,
    pub gpu_organ: MixedGpuOrganNs,
    pub host_excl: MixedHostExclusiveNs,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Qwen80MixedParityReport {
    pub passed: bool,
    pub samples: Vec<Value>,
    pub dense_w_materialized: bool,
}

struct VectorCache {
    vectors: HashMap<String, Vec<f32>>,
    geometry: HashMap<String, (usize, usize)>,
}

impl VectorCache {
    fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            geometry: HashMap::new(),
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct GpuBinary {
    signs: crate::metal::PinnedBuffer,
    scales: crate::metal::PinnedBuffer,
    sign_off: u64,
    scale_off: u64,
    rows: u32,
    cols: u32,
    group_size: u32,
    groups_per_row: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct GpuResidual {
    binary: GpuBinary,
    indices: crate::metal::PinnedBuffer,
    row_ptr: crate::metal::PinnedBuffer,
    residual_signs: crate::metal::PinnedBuffer,
    residual_sign_off: u64,
    residual_scale_f16: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct GpuHgravs {
    left_codes: crate::metal::PinnedBuffer,
    left_scales: crate::metal::PinnedBuffer,
    right_codes: crate::metal::PinnedBuffer,
    right_scales: crate::metal::PinnedBuffer,
    left_code_off: u64,
    left_scale_off: u64,
    right_code_off: u64,
    right_scale_off: u64,
    left_rows: u32,
    left_cols: u32,
    right_rows: u32,
    right_cols: u32,
    group_size: u32,
    bits: u32,
    bound: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
struct GpuUniform {
    codes: crate::metal::PinnedBuffer,
    scales: crate::metal::PinnedBuffer,
    code_off: u64,
    scale_off: u64,
    rows: u32,
    cols: u32,
    group_size: u32,
    bits: u32,
    bound: u32,
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
enum GpuWeight {
    Binary(GpuBinary),
    Residual(GpuResidual),
    Hgravs(GpuHgravs),
    Uniform(GpuUniform),
}

#[cfg(target_os = "macos")]
#[derive(Clone, Debug)]
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
struct MixedScratch {
    input: crate::metal::PinnedBuffer,
    out_a: crate::metal::PinnedBuffer,
    out_b: crate::metal::PinnedBuffer,
    out_c: crate::metal::PinnedBuffer,
    out_d: crate::metal::PinnedBuffer,
    weights: crate::metal::PinnedBuffer,
    combined: crate::metal::PinnedBuffer,
}

#[cfg(target_os = "macos")]
struct DeviceActivationWorkspace {
    hidden: crate::metal::PinnedBuffer,
    normalized: crate::metal::PinnedBuffer,
    postnorm: crate::metal::PinnedBuffer,
    mixer: crate::metal::PinnedBuffer,
    first_residual: crate::metal::PinnedBuffer,
    gate: crate::metal::PinnedBuffer,
    up: crate::metal::PinnedBuffer,
    act: crate::metal::PinnedBuffer,
    shared: crate::metal::PinnedBuffer,
    shared_logit: crate::metal::PinnedBuffer,
    qkvz: crate::metal::PinnedBuffer,
    ba: crate::metal::PinnedBuffer,
    repeated_q: crate::metal::PinnedBuffer,
    repeated_k: crate::metal::PinnedBuffer,
    conv_v: crate::metal::PinnedBuffer,
    z: crate::metal::PinnedBuffer,
    decay: crate::metal::PinnedBuffer,
    beta: crate::metal::PinnedBuffer,
    rec_out: crate::metal::PinnedBuffer,
    gated: crate::metal::PinnedBuffer,
    q_proj: crate::metal::PinnedBuffer,
    k_proj: crate::metal::PinnedBuffer,
    v_proj: crate::metal::PinnedBuffer,
    query: crate::metal::PinnedBuffer,
    attn: crate::metal::PinnedBuffer,
    gated_attn: crate::metal::PinnedBuffer,
    router_logits: crate::metal::PinnedBuffer,
    logits: crate::metal::PinnedBuffer,
    linear_conv: crate::metal::PinnedBuffer,
    linear_recurrent: crate::metal::PinnedBuffer,
    gqa_key: crate::metal::PinnedBuffer,
    gqa_value: crate::metal::PinnedBuffer,
    vectors: HashMap<String, crate::metal::PinnedBuffer>,
}

#[cfg(target_os = "macos")]
struct MetalMixedAccel {
    context: crate::metal::MetalContext,
    weights: HashMap<String, GpuWeight>,
    experts: HashMap<(usize, u16), MixedExpertGpu>,
    scratch: MixedScratch,
    wave: MixedWave,
    activations: Option<DeviceActivationWorkspace>,
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
    enc.set_buffer(0, Some(&packed.signs), packed.sign_off);
    enc.set_buffer(1, Some(&packed.scales), packed.scale_off);
    enc.set_buffer(2, Some(input), 0);
    enc.set_buffer(3, Some(output), output_offset);
    set_u32(enc, 4, packed.rows);
    set_u32(enc, 5, packed.cols);
    set_u32(enc, 6, packed.group_size);
    set_u32(enc, 7, packed.groups_per_row);
}

#[cfg(target_os = "macos")]
fn encode_binary_csr(
    enc: &metal::ComputeCommandEncoderRef,
    packed: &GpuResidual,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
) {
    enc.set_buffer(0, Some(&packed.binary.signs), packed.binary.sign_off);
    enc.set_buffer(1, Some(&packed.binary.scales), packed.binary.scale_off);
    enc.set_buffer(2, Some(input), 0);
    enc.set_buffer(3, Some(output), output_offset);
    enc.set_buffer(4, Some(&packed.indices), 0);
    enc.set_buffer(5, Some(&packed.row_ptr), 0);
    enc.set_buffer(6, Some(&packed.residual_signs), packed.residual_sign_off);
    set_u32(enc, 7, packed.binary.rows);
    set_u32(enc, 8, packed.binary.cols);
    set_u32(enc, 9, packed.binary.group_size);
    set_u32(enc, 10, packed.binary.groups_per_row);
    set_u32(enc, 11, packed.residual_scale_f16);
}

fn encode_csr(
    enc: &metal::ComputeCommandEncoderRef,
    packed: &GpuResidual,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
) {
    enc.set_buffer(0, Some(&packed.indices), 0);
    enc.set_buffer(1, Some(&packed.row_ptr), 0);
    enc.set_buffer(2, Some(&packed.residual_signs), packed.residual_sign_off);
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
    code_off: u64,
    scale_off: u64,
) {
    enc.set_buffer(0, Some(codes), code_off);
    enc.set_buffer(1, Some(scales), scale_off);
    enc.set_buffer(2, Some(input), input_offset);
    enc.set_buffer(3, Some(output), output_offset);
    set_u32(enc, 4, rows);
    set_u32(enc, 5, cols);
    set_u32(enc, 6, group_size);
    set_u32(enc, 7, bits);
    set_u32(enc, 8, bound);
}

#[cfg(target_os = "macos")]
fn dispatch_binary(
    tcb: &mut crate::metal::TokenCommandBuffer<'_>,
    body: &GpuBinary,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
    serial_name: &str,
) -> Result<()> {
    if qwen80_gk_simd_enabled() {
        tcb.dispatch_threads(
            crate::decode_family::MATVEC_BINARY_SIMD,
            simd8_grid(body.rows),
            (256, 1, 1),
            |enc| encode_binary(enc, body, input, output, output_offset),
        )
    } else if qwen80_recon_fuse_enabled() {
        tcb.dispatch_threads(
            "q80_binary_group_matvec_tg256",
            tg256_grid(body.rows),
            (256, 1, 1),
            |enc| encode_binary(enc, body, input, output, output_offset),
        )
    } else {
        tcb.dispatch_threads(
            serial_name,
            (body.rows, 1, 1),
            (256, 1, 1),
            |enc| encode_binary(enc, body, input, output, output_offset),
        )
    }
}

#[cfg(target_os = "macos")]
fn dispatch_residual(
    tcb: &mut crate::metal::TokenCommandBuffer<'_>,
    body: &GpuResidual,
    input: &crate::metal::PinnedBuffer,
    output: &crate::metal::PinnedBuffer,
    output_offset: u64,
    serial_binary: &str,
) -> Result<()> {
    if qwen80_recon_fuse_enabled() {
        tcb.dispatch_threads(
            "q80_binary_group_csr_matvec_tg256",
            tg256_grid(body.binary.rows),
            (256, 1, 1),
            |enc| encode_binary_csr(enc, body, input, output, output_offset),
        )
    } else {
        tcb.dispatch_threads(
            serial_binary,
            (body.binary.rows, 1, 1),
            (256, 1, 1),
            |enc| encode_binary(enc, &body.binary, input, output, output_offset),
        )?;
        tcb.dispatch_threads(
            "q80_sparse_q1_apply_csr",
            (body.binary.rows, 1, 1),
            (256, 1, 1),
            |enc| encode_csr(enc, body, input, output, output_offset),
        )
    }
}

#[cfg(target_os = "macos")]
fn dispatch_factor(
    tcb: &mut crate::metal::TokenCommandBuffer<'_>,
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
    code_off: u64,
    scale_off: u64,
    serial_name: &str,
) -> Result<()> {
    let encode = |enc: &metal::ComputeCommandEncoderRef| {
        encode_factor(
            enc,
            codes,
            scales,
            input,
            input_offset,
            output,
            output_offset,
            rows,
            cols,
            group_size,
            bits,
            bound,
            code_off,
            scale_off,
        )
    };
    if qwen80_gk_simd_enabled() {
        tcb.dispatch_threads(
            crate::decode_family::MATVEC_HGRAVS_SIMD,
            simd8_grid(rows),
            (256, 1, 1),
            encode,
        )
    } else if qwen80_recon_fuse_enabled() {
        let (name, grid) = if bits == 8 {
            if cols >= 2048 {
                ("q80_uniform8_matvec_tg256", tg256_grid(rows))
            } else {
                ("q80_uniform8_matvec_simd_bytes", simd8_grid(rows))
            }
        } else if bits == 3 {
            ("q80_hgravs01_factor_matvec_simd3", simd8_grid(rows))
        } else {
            ("q80_hgravs01_factor_matvec_simd", simd8_grid(rows))
        };
        tcb.dispatch_threads(name, grid, (256, 1, 1), encode)
    } else {
        tcb.dispatch_threads(serial_name, (rows, 1, 1), (256, 1, 1), encode)
    }
}

#[cfg(target_os = "macos")]
fn gpu_weight_rows_cols(weight: &GpuWeight) -> (usize, usize) {
    match weight {
        GpuWeight::Binary(body) => (body.rows as usize, body.cols as usize),
        GpuWeight::Residual(body) => (body.binary.rows as usize, body.binary.cols as usize),
        GpuWeight::Hgravs(body) => (body.left_rows as usize, body.right_cols as usize),
        GpuWeight::Uniform(body) => (body.rows as usize, body.cols as usize),
    }
}

fn bytes_f32(n: usize, label: &str) -> Result<usize> {
    n.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| mixed_error(format!("{label} byte count overflowed")))
}

#[cfg(target_os = "macos")]
impl DeviceActivationWorkspace {
    fn allocate(context: &crate::metal::MetalContext, max_seq_len: usize) -> Result<Self> {
        let linear = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        linear.validate()?;
        let gqa = Qwen80CanonicalGqaLayout::source_exact();
        gqa.validate()?;
        let mut n_linear = 0usize;
        let mut n_gqa = 0usize;
        for layer in 0..QWEN80_LAYERS {
            match qwen80_layer_kind(layer)? {
                Qwen80LayerKind::LinearAttention => n_linear += 1,
                Qwen80LayerKind::FullAttention => n_gqa += 1,
            }
        }
        let hidden = bytes_f32(QWEN80_HIDDEN, "act hidden")?;
        let mid = bytes_f32(QWEN80_MOE_INTERMEDIATE, "act moe mid")?;
        let qkvz = bytes_f32(linear.qkvz_projection_elements()?, "act qkvz")?;
        let ba = bytes_f32(linear.ba_projection_elements()?, "act ba")?;
        let value = bytes_f32(linear.value_elements()?, "act value")?;
        let q_proj = bytes_f32(gqa.q_proj_rows, "act q_proj")?;
        let kv = bytes_f32(gqa.kv_dim, "act kv")?;
        let query = bytes_f32(gqa.query_dim, "act query")?;
        let conv_bytes = bytes_f32(
            n_linear
                .checked_mul(linear.conv_state_elements()?)
                .ok_or_else(|| mixed_error("act conv state overflowed"))?,
            "act conv state",
        )?;
        let rec_bytes = bytes_f32(
            n_linear
                .checked_mul(linear.recurrent_state_elements()?)
                .ok_or_else(|| mixed_error("act recurrent state overflowed"))?,
            "act recurrent state",
        )?;
        let gqa_bytes = bytes_f32(
            n_gqa
                .checked_mul(max_seq_len)
                .and_then(|v| v.checked_mul(gqa.kv_dim))
                .ok_or_else(|| mixed_error("act gqa cache overflowed"))?,
            "act gqa cache",
        )?;
        Ok(Self {
            hidden: context.new_buffer_checked(hidden)?,
            normalized: context.new_buffer_checked(hidden)?,
            postnorm: context.new_buffer_checked(hidden)?,
            mixer: context.new_buffer_checked(hidden)?,
            first_residual: context.new_buffer_checked(hidden)?,
            gate: context.new_buffer_checked(mid)?,
            up: context.new_buffer_checked(mid)?,
            act: context.new_buffer_checked(mid)?,
            shared: context.new_buffer_checked(hidden)?,
            shared_logit: context.new_buffer_checked(bytes_f32(1, "act shared logit")?)?,
            qkvz: context.new_buffer_checked(qkvz)?,
            ba: context.new_buffer_checked(ba)?,
            repeated_q: context.new_buffer_checked(value)?,
            repeated_k: context.new_buffer_checked(value)?,
            conv_v: context.new_buffer_checked(value)?,
            z: context.new_buffer_checked(value)?,
            decay: context.new_buffer_checked(bytes_f32(linear.value_heads, "act decay")?)?,
            beta: context.new_buffer_checked(bytes_f32(linear.value_heads, "act beta")?)?,
            rec_out: context.new_buffer_checked(value)?,
            gated: context.new_buffer_checked(value)?,
            q_proj: context.new_buffer_checked(q_proj)?,
            k_proj: context.new_buffer_checked(kv)?,
            v_proj: context.new_buffer_checked(kv)?,
            query: context.new_buffer_checked(query)?,
            attn: context.new_buffer_checked(query)?,
            gated_attn: context.new_buffer_checked(query)?,
            router_logits: context.new_buffer_checked(bytes_f32(QWEN80_EXPERTS, "act router")?)?,
            logits: context.new_buffer_checked(bytes_f32(QWEN80_VOCAB, "act logits")?)?,
            linear_conv: context.new_buffer_checked(conv_bytes)?,
            linear_recurrent: context.new_buffer_checked(rec_bytes)?,
            gqa_key: context.new_buffer_checked(gqa_bytes)?,
            gqa_value: context.new_buffer_checked(gqa_bytes)?,
            vectors: HashMap::new(),
        })
    }

    fn zero_state(&mut self) {
        fn zero(buf: &crate::metal::PinnedBuffer) {
            let len = buf.length() as usize;
            if len == 0 {
                return;
            }
            unsafe {
                std::ptr::write_bytes(buf.contents() as *mut u8, 0, len);
            }
        }
        zero(&self.linear_conv);
        zero(&self.linear_recurrent);
        zero(&self.gqa_key);
        zero(&self.gqa_value);
    }
}

#[cfg(target_os = "macos")]
impl MetalMixedAccel {
    fn new(max_seq_len: usize) -> Result<Self> {
        let context = crate::metal::MetalContext::new()?;
        let hidden = QWEN80_HIDDEN * 4;
        let mid = 10 * QWEN80_MOE_INTERMEDIATE * 4;
        let rank = 10 * Q80_HGRAVS_RANK * 4;
        let down = 10 * QWEN80_HIDDEN * 4;
        let lm_head = QWEN80_VOCAB * 4;
        let qkvz = 12_288 * 4;
        let activations = if qwen80_mixed_device_activations_enabled() {
            Some(DeviceActivationWorkspace::allocate(&context, max_seq_len)?)
        } else {
            None
        };
        Ok(Self {
            wave: MixedWave {
                input: context.new_buffer_checked(hidden)?,
                gate: context.new_buffer_checked(mid)?,
                up: context.new_buffer_checked(mid)?,
                act: context.new_buffer_checked(mid)?,
                mid: context.new_buffer_checked(rank)?,
                down: context.new_buffer_checked(down)?,
            },
            scratch: MixedScratch {
                input: context.new_buffer_checked(qkvz.max(hidden))?,
                out_a: context.new_buffer_checked(lm_head)?,
                out_b: context.new_buffer_checked(qkvz)?,
                out_c: context.new_buffer_checked(qkvz)?,
                out_d: context.new_buffer_checked(hidden)?,
                weights: context.new_buffer_checked(10 * 4)?,
                combined: context.new_buffer_checked(hidden)?,
            },
            context,
            weights: HashMap::new(),
            experts: HashMap::new(),
            activations,
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
            sign_off: 0,
            scale_off: 0,
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
            residual_sign_off: 0,
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
            left_code_off: 0,
            left_scale_off: 0,
            right_code_off: 0,
            right_scale_off: 0,
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
            code_off: 0,
            scale_off: 0,
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

    fn copy_slice(&self, bytes: &[u8]) -> Result<crate::metal::PinnedBuffer> {
        self.context.new_buffer_with_bytes_checked(bytes)
    }

    fn upload_from_catalog(
        &mut self,
        catalog: &Qwen80MixedStreamingCatalog,
        name: &str,
        native: &mut Qwen80MixedNativeCounts,
    ) -> Result<GpuWeight> {
        let row = catalog.require_row(name)?;
        let view = catalog.payload_view(name)?;
        let payload = view.as_slice();
        let layout = mixed_gpu_layout(row.codec, payload)?;
        native.expert_nocopy_binds = native.expert_nocopy_binds.saturating_add(1);
        match layout.kind {
            MixedGpuKind::Binary {
                scale_off,
                scale_bytes,
                sign_off,
                sign_bytes,
                group_size,
                groups_per_row,
            } => Ok(GpuWeight::Binary(GpuBinary {
                signs: self.copy_slice(&payload[sign_off..sign_off + sign_bytes])?,
                scales: self.copy_slice(&payload[scale_off..scale_off + scale_bytes])?,
                sign_off: 0,
                scale_off: 0,
                rows: layout.rows,
                cols: layout.cols,
                group_size,
                groups_per_row,
            })),
            MixedGpuKind::Residual {
                ref binary,
                residual_sign_off,
                residual_sign_bytes,
                rice_k,
                first_index,
                rice_off,
                rice_bytes,
                outlier_count,
                residual_scale_f16,
            } => {
                let MixedGpuKind::Binary {
                    scale_off,
                    scale_bytes,
                    sign_off,
                    sign_bytes,
                    group_size,
                    groups_per_row,
                } = binary.as_ref()
                else {
                    return Err(mixed_error("residual binary layout drifted"));
                };
                let rice = RiceQ1Packed {
                    binary: super::qwen_complete_binary::BinaryGroupPacked {
                        rows: layout.rows as usize,
                        cols: layout.cols as usize,
                        group_size: *group_size as usize,
                        groups_per_row: *groups_per_row as usize,
                        scales_f16: Vec::new(),
                        signs: Vec::new(),
                    },
                    first_index,
                    rice_k,
                    rice_bytes: payload[rice_off..rice_off + rice_bytes].to_vec(),
                    outlier_count,
                    residual_scale_f16: residual_scale_f16 as u16,
                    residual_signs: Vec::new(),
                    indices: Vec::new(),
                };
                let indices = expand_rice_indices(&rice)?;
                let row_ptr = rice_q1_row_ptr(&indices, layout.rows as usize, layout.cols as usize)?;
                let idx_bytes: Vec<u8> = indices.iter().flat_map(|v| v.to_le_bytes()).collect();
                let ptr_bytes: Vec<u8> = row_ptr.iter().flat_map(|v| v.to_le_bytes()).collect();
                Ok(GpuWeight::Residual(GpuResidual {
                    binary: GpuBinary {
                        signs: self.copy_slice(&payload[*sign_off..*sign_off + *sign_bytes])?,
                        scales: self.copy_slice(&payload[*scale_off..*scale_off + *scale_bytes])?,
                        sign_off: 0,
                        scale_off: 0,
                        rows: layout.rows,
                        cols: layout.cols,
                        group_size: *group_size,
                        groups_per_row: *groups_per_row,
                    },
                    indices: self.context.new_buffer_with_bytes_checked(&idx_bytes)?,
                    row_ptr: self.context.new_buffer_with_bytes_checked(&ptr_bytes)?,
                    residual_signs: self
                        .copy_slice(&payload[residual_sign_off..residual_sign_off + residual_sign_bytes])?,
                    residual_sign_off: 0,
                    residual_scale_f16,
                }))
            }
            MixedGpuKind::Hgravs { left, right } => Ok(GpuWeight::Hgravs(GpuHgravs {
                left_codes: self.copy_slice(&payload[left.code_off..left.code_off + left.code_bytes])?,
                left_scales: self
                    .copy_slice(&payload[left.scale_off..left.scale_off + left.scale_bytes])?,
                right_codes: self
                    .copy_slice(&payload[right.code_off..right.code_off + right.code_bytes])?,
                right_scales: self
                    .copy_slice(&payload[right.scale_off..right.scale_off + right.scale_bytes])?,
                left_code_off: 0,
                left_scale_off: 0,
                right_code_off: 0,
                right_scale_off: 0,
                left_rows: left.rows,
                left_cols: left.cols,
                right_rows: right.rows,
                right_cols: right.cols,
                group_size: left.group_size,
                bits: left.bits,
                bound: left.bound,
            })),
            MixedGpuKind::Uniform(factor) => Ok(GpuWeight::Uniform(GpuUniform {
                codes: self
                    .copy_slice(&payload[factor.code_off..factor.code_off + factor.code_bytes])?,
                scales: self
                    .copy_slice(&payload[factor.scale_off..factor.scale_off + factor.scale_bytes])?,
                code_off: 0,
                scale_off: 0,
                rows: factor.rows,
                cols: factor.cols,
                group_size: factor.group_size,
                bits: factor.bits,
                bound: factor.bound,
            })),
        }
    }

    fn encode_weight(
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        weight: &GpuWeight,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
        native: &mut Qwen80MixedNativeCounts,
    ) -> Result<()> {
        match weight {
            GpuWeight::Binary(body) => {
                dispatch_binary(tcb, body, input, output, 0, crate::decode_family::matvec_binary())?;
                native.binary_dispatches = native.binary_dispatches.saturating_add(1);
            }
            GpuWeight::Residual(body) => {
                dispatch_residual(tcb, body, input, output, 0, crate::decode_family::matvec_binary())?;
                native.binary_dispatches = native.binary_dispatches.saturating_add(1);
                native.residual_dispatches = native.residual_dispatches.saturating_add(1);
            }
            GpuWeight::Hgravs(_body) => {
                return Err(mixed_error(
                    "hgravs encode_weight needs a mid buffer; use matvec",
                ));
            }
            GpuWeight::Uniform(body) => {
                dispatch_factor(
                    tcb,
                    &body.codes,
                    &body.scales,
                    input,
                    0,
                    output,
                    0,
                    body.rows,
                    body.cols,
                    body.group_size,
                    body.bits,
                    body.bound,
                    body.code_off,
                    body.scale_off,
                    crate::decode_family::matvec_hgravs(),
                )?;
                native.uniform8_dispatches = native.uniform8_dispatches.saturating_add(1);
            }
        }
        let _ = native;
        Ok(())
    }

    fn note_timing(
        stages: &mut Qwen80MixedStageTimes,
        native: &mut Qwen80MixedNativeCounts,
        timing: &crate::metal::CommandBufferTiming,
        organ: MixedGpuOrgan,
    ) {
        native.command_buffers = native.command_buffers.saturating_add(1);
        native.compute_dispatches = native
            .compute_dispatches
            .saturating_add(timing.dispatches);
        stages.cb_wait_ns = stages.cb_wait_ns.saturating_add(timing.wait_ns);
        stages.cb_submit_ns = stages.cb_submit_ns.saturating_add(timing.submit_ns);
        stages.cb_encode_ns = stages.cb_encode_ns.saturating_add(timing.encode_ns);
        match timing.gpu_ns {
            Some(ns) => {
                stages.gpu_matvec_ns = stages.gpu_matvec_ns.saturating_add(ns);
                stages.cb_wait_minus_gpu_ns = stages
                    .cb_wait_minus_gpu_ns
                    .saturating_add(timing.wait_ns.saturating_sub(ns));
                let slot = match organ {
                    MixedGpuOrgan::DeltaNet => &mut stages.gpu_organ.deltanet,
                    MixedGpuOrgan::Gqa => &mut stages.gpu_organ.gqa,
                    MixedGpuOrgan::MoeShared => &mut stages.gpu_organ.moe_shared,
                    MixedGpuOrgan::MoeRouted => &mut stages.gpu_organ.moe_routed,
                    MixedGpuOrgan::MoeRouter => &mut stages.gpu_organ.moe_router,
                    MixedGpuOrgan::MoeCombineGate => &mut stages.gpu_organ.moe_combine_gate,
                    MixedGpuOrgan::Terminal => &mut stages.gpu_organ.terminal,
                    MixedGpuOrgan::Other => &mut stages.gpu_organ.other,
                };
                *slot = slot.saturating_add(ns);
            }
            None => {
                stages.gpu_matvec_timestamps_missing =
                    stages.gpu_matvec_timestamps_missing.saturating_add(1)
            }
        }
    }

    fn weight_geometry(&self, name: &str) -> Result<(usize, usize)> {
        self.weights
            .get(name)
            .map(gpu_weight_rows_cols)
            .ok_or_else(|| mixed_error(format!("{name} is not bound")))
    }

    fn matvec(
        &mut self,
        name: &str,
        packed: Option<&MixedPackedTensor>,
        input: &[f32],
        output: &mut [f32],
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
        organ: MixedGpuOrgan,
    ) -> Result<()> {
        if !self.weights.contains_key(name) {
            let packed = packed.ok_or_else(|| {
                mixed_error(format!("{name} is not bound and no packed tensor was supplied"))
            })?;
            native.expert_copy_binds = native.expert_copy_binds.saturating_add(1);
            let uploaded = self.upload_tensor(packed)?;
            self.weights.insert(name.to_owned(), uploaded);
        }
        let (rows, cols) = self.weight_geometry(name)?;
        if input.len() != cols || output.len() != rows {
            return Err(mixed_error(format!(
                "{name} metal matvec geometry {}x{} vs in={} out={}",
                rows,
                cols,
                input.len(),
                output.len()
            )));
        }
        let prep = Instant::now();
        let input_buf = self
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(input))?;
        let output_buf = self.context.new_buffer_checked(rows * 4)?;
        add_ns(&mut stages.host_excl.buffer_prep, prep);
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        match self.weights.get(name).expect("uploaded") {
            GpuWeight::Binary(body) => {
                dispatch_binary(
                    &mut tcb,
                    body,
                    &input_buf,
                    &output_buf,
                    0,
                    crate::decode_family::matvec_binary(),
                )?;
                native.binary_dispatches = native.binary_dispatches.saturating_add(1);
            }
            GpuWeight::Residual(body) => {
                dispatch_residual(
                    &mut tcb,
                    body,
                    &input_buf,
                    &output_buf,
                    0,
                    crate::decode_family::matvec_binary(),
                )?;
                native.binary_dispatches = native.binary_dispatches.saturating_add(1);
                native.residual_dispatches = native.residual_dispatches.saturating_add(1);
            }
            GpuWeight::Hgravs(body) => {
                let mid = self
                    .context
                    .new_buffer_checked(body.right_rows as usize * 4)?;
                dispatch_factor(
                    &mut tcb,
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
                    body.right_code_off,
                    body.right_scale_off,
                    crate::decode_family::matvec_hgravs(),
                )?;
                dispatch_factor(
                    &mut tcb,
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
                    body.left_code_off,
                    body.left_scale_off,
                    crate::decode_family::matvec_hgravs(),
                )?;
                native.hgravs_factor_dispatches =
                    native.hgravs_factor_dispatches.saturating_add(2);
            }
            GpuWeight::Uniform(body) => {
                dispatch_factor(
                    &mut tcb,
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
                    body.code_off,
                    body.scale_off,
                    crate::decode_family::matvec_hgravs(),
                )?;
                native.uniform8_dispatches = native.uniform8_dispatches.saturating_add(1);
            }
        }
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing, organ);
        let readback = Instant::now();
        output.copy_from_slice(&read_f32(&output_buf, rows));
        add_ns(&mut stages.host_excl.buffer_prep, readback);
        Ok(())
    }

    fn ensure_named_weight(
        &mut self,
        catalog: &Qwen80MixedStreamingCatalog,
        name: &str,
        packed: Option<&MixedPackedTensor>,
        native: &mut Qwen80MixedNativeCounts,
    ) -> Result<()> {
        if self.weights.contains_key(name) {
            return Ok(());
        }
        let uploaded = if qwen80_host_facet1_enabled() {
            self.upload_from_catalog(catalog, name, native)?
        } else {
            let packed = packed.ok_or_else(|| {
                mixed_error(format!(
                    "{name}: facet1 off requires packed() on first bind"
                ))
            })?;
            native.expert_copy_binds = native.expert_copy_binds.saturating_add(1);
            self.upload_tensor(packed)?
        };
        self.weights.insert(name.to_owned(), uploaded);
        Ok(())
    }

    fn matvec_group_same_input(
        &mut self,
        catalog: &Qwen80MixedStreamingCatalog,
        names: &[&str],
        packed: &[MixedPackedTensor],
        input: &[f32],
        outputs: &mut [&mut [f32]],
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
        organ: MixedGpuOrgan,
    ) -> Result<()> {
        if names.len() != outputs.len() || names.len() != packed.len() || names.is_empty() {
            return Err(mixed_error("matvec group arity drifted"));
        }
        if !qwen80_host_facet2_enabled() || names.len() == 1 {
            for i in 0..names.len() {
                self.ensure_named_weight(catalog, names[i], Some(&packed[i]), native)?;
                self.matvec(
                    names[i],
                    Some(&packed[i]),
                    input,
                    outputs[i],
                    native,
                    stages,
                    organ,
                )?;
            }
            return Ok(());
        }
        for i in 0..names.len() {
            self.ensure_named_weight(catalog, names[i], Some(&packed[i]), native)?;
        }
        if names.iter().any(|name| {
            self.weights
                .get(*name)
                .map(|w| matches!(w, GpuWeight::Hgravs(_)))
                .unwrap_or(false)
        }) {
            for i in 0..names.len() {
                self.matvec(
                    names[i],
                    Some(&packed[i]),
                    input,
                    outputs[i],
                    native,
                    stages,
                    organ,
                )?;
            }
            return Ok(());
        }
        let prep = Instant::now();
        write_f32(&self.scratch.input, input);
        add_ns(&mut stages.host_excl.buffer_prep, prep);
        let outs = [
            self.scratch.out_a.clone(),
            self.scratch.out_b.clone(),
            self.scratch.out_c.clone(),
            self.scratch.out_d.clone(),
        ];
        if names.len() > outs.len() {
            return Err(mixed_error("matvec group exceeds scratch slots"));
        }
        let input_buf = self.scratch.input.clone();
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        tcb.begin_serial_group()?;
        for (i, name) in names.iter().enumerate() {
            let weight = self
                .weights
                .get(*name)
                .ok_or_else(|| mixed_error("weight missing after ensure"))?;
            Self::encode_weight(&mut tcb, weight, &input_buf, &outs[i], native)?;
        }
        tcb.end_concurrent_group()?;
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing, organ);
        let readback = Instant::now();
        for (i, out) in outputs.iter_mut().enumerate() {
            out.copy_from_slice(&read_f32(&outs[i], out.len()));
        }
        add_ns(&mut stages.host_excl.buffer_prep, readback);
        Ok(())
    }

    fn matvec_group_bound(
        &mut self,
        names: &[&str],
        input: &[f32],
        outputs: &mut [&mut [f32]],
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
        organ: MixedGpuOrgan,
    ) -> Result<()> {
        if names.len() != outputs.len() || names.is_empty() {
            return Err(mixed_error("matvec group arity drifted"));
        }
        if !qwen80_host_facet2_enabled()
            || names.len() == 1
            || names.iter().any(|name| {
                self.weights
                    .get(*name)
                    .map(|w| matches!(w, GpuWeight::Hgravs(_)))
                    .unwrap_or(false)
            })
        {
            for (name, out) in names.iter().zip(outputs.iter_mut()) {
                self.matvec(name, None, input, out, native, stages, organ)?;
            }
            return Ok(());
        }
        let prep = Instant::now();
        write_f32(&self.scratch.input, input);
        add_ns(&mut stages.host_excl.buffer_prep, prep);
        let outs = [
            self.scratch.out_a.clone(),
            self.scratch.out_b.clone(),
            self.scratch.out_c.clone(),
            self.scratch.out_d.clone(),
        ];
        if names.len() > outs.len() {
            return Err(mixed_error("matvec group exceeds scratch slots"));
        }
        let input_buf = self.scratch.input.clone();
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        tcb.begin_serial_group()?;
        for (i, name) in names.iter().enumerate() {
            let weight = self
                .weights
                .get(*name)
                .ok_or_else(|| mixed_error("weight missing after ensure"))?;
            Self::encode_weight(&mut tcb, weight, &input_buf, &outs[i], native)?;
        }
        tcb.end_concurrent_group()?;
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing, organ);
        let readback = Instant::now();
        for (i, out) in outputs.iter_mut().enumerate() {
            out.copy_from_slice(&read_f32(&outs[i], out.len()));
        }
        add_ns(&mut stages.host_excl.buffer_prep, readback);
        Ok(())
    }

    fn ensure_expert(
        &mut self,
        catalog: &Qwen80MixedStreamingCatalog,
        layer: usize,
        expert: u16,
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
    ) -> Result<()> {
        if self.experts.contains_key(&(layer, expert)) {
            return Ok(());
        }
        let bind_started = Instant::now();
        let gate_name = format!("model.layers.{layer}.mlp.experts.{expert}.gate_proj.weight");
        let up_name = format!("model.layers.{layer}.mlp.experts.{expert}.up_proj.weight");
        let down_name = format!("model.layers.{layer}.mlp.experts.{expert}.down_proj.weight");
        if qwen80_host_facet1_enabled() {
            let gate = match self.upload_from_catalog(catalog, &gate_name, native)? {
                GpuWeight::Binary(body) => body,
                _ => return Err(mixed_error(format!("{gate_name} did not parse as binary"))),
            };
            let up = match self.upload_from_catalog(catalog, &up_name, native)? {
                GpuWeight::Residual(body) => body,
                _ => return Err(mixed_error(format!("{up_name} did not parse as rice residual"))),
            };
            let down = match self.upload_from_catalog(catalog, &down_name, native)? {
                GpuWeight::Hgravs(body) => body,
                _ => return Err(mixed_error(format!("{down_name} did not parse as hgravs01"))),
            };
            self.experts
                .insert((layer, expert), MixedExpertGpu { gate, up, down });
            add_elapsed_bind(stages, bind_started);
            return Ok(());
        }
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
        native.expert_copy_binds = native.expert_copy_binds.saturating_add(3);
        add_elapsed_bind(stages, bind_started);
        Ok(())
    }

    fn routed_wave_fused(
        &mut self,
        _layer: usize,
        ids: &[u16],
        weights: &[f32],
        input: &[f32],
        combined: &mut [f32],
        native: &mut Qwen80MixedNativeCounts,
        stages: &mut Qwen80MixedStageTimes,
    ) -> Result<()> {
        write_f32(&self.wave.input, input);
        write_f32(&self.scratch.weights, weights);
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        tcb.begin_serial_group()?;
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(_layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let mid_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            dispatch_binary(
                &mut tcb,
                &trip.gate,
                &self.wave.input,
                &self.wave.gate,
                mid_off,
                crate::decode_family::matvec_binary(),
            )?;
            dispatch_residual(
                &mut tcb,
                &trip.up,
                &self.wave.input,
                &self.wave.up,
                mid_off,
                crate::decode_family::matvec_binary(),
            )?;
            native.binary_dispatches = native.binary_dispatches.saturating_add(2);
            native.residual_dispatches = native.residual_dispatches.saturating_add(1);
        }
        let silu_n = (10 * QWEN80_MOE_INTERMEDIATE) as u32;
        tcb.dispatch_threads(
            "qwen80_expert_table_silu_mul",
            (silu_n.div_ceil(256) * 256, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&self.wave.gate), 0);
                enc.set_buffer(1, Some(&self.wave.up), 0);
                enc.set_buffer(2, Some(&self.wave.act), 0);
                set_u32(enc, 3, silu_n);
            },
        )?;
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(_layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let act_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            let mid_off = (slot * Q80_HGRAVS_RANK * 4) as u64;
            let down_off = (slot * QWEN80_HIDDEN * 4) as u64;
            dispatch_factor(
                &mut tcb,
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
                trip.down.right_code_off,
                trip.down.right_scale_off,
                crate::decode_family::matvec_hgravs(),
            )?;
            dispatch_factor(
                &mut tcb,
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
                trip.down.left_code_off,
                trip.down.left_scale_off,
                crate::decode_family::matvec_hgravs(),
            )?;
            native.hgravs_factor_dispatches = native.hgravs_factor_dispatches.saturating_add(2);
        }
        let hidden = QWEN80_HIDDEN as u32;
        tcb.dispatch_threads(
            "qwen80_expert_table_weighted_sum",
            (hidden.div_ceil(256) * 256, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&self.wave.down), 0);
                enc.set_buffer(1, Some(&self.scratch.weights), 0);
                enc.set_buffer(2, Some(&self.scratch.combined), 0);
                set_u32(enc, 3, hidden);
                set_u32(enc, 4, 10);
            },
        )?;
        tcb.end_concurrent_group()?;
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing, MixedGpuOrgan::MoeRouted);
        let readback = Instant::now();
        combined.copy_from_slice(&read_f32(&self.scratch.combined, QWEN80_HIDDEN));
        add_ns(&mut stages.host_excl.buffer_prep, readback);
        native.routed_expert_waves = native.routed_expert_waves.saturating_add(1);
        Ok(())
    }

    fn encode_routed_wave_into(
        &self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        ids: &[u16],
        weights: &[f32],
        input: &crate::metal::PinnedBuffer,
        native: &mut Qwen80MixedNativeCounts,
    ) -> Result<()> {
        write_f32(&self.scratch.weights, weights);
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let mid_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            dispatch_binary(
                tcb,
                &trip.gate,
                input,
                &self.wave.gate,
                mid_off,
                crate::decode_family::matvec_binary(),
            )?;
            dispatch_residual(
                tcb,
                &trip.up,
                input,
                &self.wave.up,
                mid_off,
                crate::decode_family::matvec_binary(),
            )?;
            native.binary_dispatches = native.binary_dispatches.saturating_add(2);
            native.residual_dispatches = native.residual_dispatches.saturating_add(1);
        }
        let silu_n = (10 * QWEN80_MOE_INTERMEDIATE) as u32;
        tcb.dispatch_threads(
            "qwen80_expert_table_silu_mul",
            (silu_n.div_ceil(256) * 256, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&self.wave.gate), 0);
                enc.set_buffer(1, Some(&self.wave.up), 0);
                enc.set_buffer(2, Some(&self.wave.act), 0);
                set_u32(enc, 3, silu_n);
            },
        )?;
        native.device_activation_dispatches =
            native.device_activation_dispatches.saturating_add(1);
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let act_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            let mid_off = (slot * Q80_HGRAVS_RANK * 4) as u64;
            let down_off = (slot * QWEN80_HIDDEN * 4) as u64;
            dispatch_factor(
                tcb,
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
                trip.down.right_code_off,
                trip.down.right_scale_off,
                crate::decode_family::matvec_hgravs(),
            )?;
            dispatch_factor(
                tcb,
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
                trip.down.left_code_off,
                trip.down.left_scale_off,
                crate::decode_family::matvec_hgravs(),
            )?;
            native.hgravs_factor_dispatches = native.hgravs_factor_dispatches.saturating_add(2);
        }
        let hidden = QWEN80_HIDDEN as u32;
        tcb.dispatch_threads(
            "qwen80_expert_table_weighted_sum",
            (hidden.div_ceil(256) * 256, 1, 1),
            (256, 1, 1),
            |enc| {
                enc.set_buffer(0, Some(&self.wave.down), 0);
                enc.set_buffer(1, Some(&self.scratch.weights), 0);
                enc.set_buffer(2, Some(&self.scratch.combined), 0);
                set_u32(enc, 3, hidden);
                set_u32(enc, 4, 10);
            },
        )?;
        native.device_activation_dispatches =
            native.device_activation_dispatches.saturating_add(1);
        native.routed_expert_waves = native.routed_expert_waves.saturating_add(1);
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
        degrade: &MixedDegradeConfig,
    ) -> Result<()> {
        if ids.len() != 10 || weights.len() != 10 || input.len() != QWEN80_HIDDEN {
            return Err(mixed_error("routed wave expects top-10 and hidden=2048"));
        }
        for &expert in ids {
            self.ensure_expert(catalog, layer, expert, native, stages)?;
        }
        if qwen80_host_facet2_enabled() && degrade.is_identity() {
            return self.routed_wave_fused(
                layer, ids, weights, input, combined, native, stages,
            );
        }
        write_f32(&self.wave.input, input);
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let mid_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            dispatch_binary(
                &mut tcb,
                &trip.gate,
                &self.wave.input,
                &self.wave.gate,
                mid_off,
                crate::decode_family::matvec_binary(),
            )?;
            dispatch_residual(
                &mut tcb,
                &trip.up,
                &self.wave.input,
                &self.wave.up,
                mid_off,
                crate::decode_family::matvec_binary(),
            )?;
            native.binary_dispatches = native.binary_dispatches.saturating_add(2);
            native.residual_dispatches = native.residual_dispatches.saturating_add(1);
        }
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing, MixedGpuOrgan::MoeRouted);

        let mut gate = read_f32(&self.wave.gate, 10 * QWEN80_MOE_INTERMEDIATE);
        let mut up = read_f32(&self.wave.up, 10 * QWEN80_MOE_INTERMEDIATE);
        if !degrade.is_identity() {
            for (slot, &expert) in ids.iter().enumerate() {
                let a = slot * QWEN80_MOE_INTERMEDIATE;
                let b = a + QWEN80_MOE_INTERMEDIATE;
                mix_matched_cosine(
                    &mut gate[a..b],
                    degrade.gate_mix,
                    mix_seed(degrade.mix_seed, layer, expert, 0),
                );
                mix_matched_cosine(
                    &mut up[a..b],
                    degrade.up_mix,
                    mix_seed(degrade.mix_seed, layer, expert, 1),
                );
            }
        }
        let mut act = vec![0.0f32; 10 * QWEN80_MOE_INTERMEDIATE];
        for slot in 0..10 {
            let a = slot * QWEN80_MOE_INTERMEDIATE;
            let b = a + QWEN80_MOE_INTERMEDIATE;
            silu_mul(&gate[a..b], &up[a..b], &mut act[a..b]);
        }
        write_f32(&self.wave.act, &act);

        let rank_cap = degrade.rank_cap();
        let fused = degrade.is_identity();
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        for (slot, &expert) in ids.iter().enumerate() {
            let trip = self
                .experts
                .get(&(layer, expert))
                .ok_or_else(|| mixed_error("expert missing after ensure"))?;
            let act_off = (slot * QWEN80_MOE_INTERMEDIATE * 4) as u64;
            let mid_off = (slot * Q80_HGRAVS_RANK * 4) as u64;
            dispatch_factor(
                &mut tcb,
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
                trip.down.right_code_off,
                trip.down.right_scale_off,
                crate::decode_family::matvec_hgravs(),
            )?;
            if fused {
                let down_off = (slot * QWEN80_HIDDEN * 4) as u64;
                dispatch_factor(
                    &mut tcb,
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
                    trip.down.left_code_off,
                    trip.down.left_scale_off,
                    crate::decode_family::matvec_hgravs(),
                )?;
            }
            native.hgravs_factor_dispatches = native.hgravs_factor_dispatches.saturating_add(1);
        }
        let timing = tcb.commit_and_wait_timed()?;
        Self::note_timing(stages, native, &timing, MixedGpuOrgan::MoeRouted);

        if !fused {
            if rank_cap < Q80_HGRAVS_RANK {
                let mut mid = read_f32(&self.wave.mid, 10 * Q80_HGRAVS_RANK);
                for slot in 0..10 {
                    let base = slot * Q80_HGRAVS_RANK;
                    mid[base + rank_cap..base + Q80_HGRAVS_RANK].fill(0.0);
                }
                write_f32(&self.wave.mid, &mid);
            }
            let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
            for (slot, &expert) in ids.iter().enumerate() {
                let trip = self
                    .experts
                    .get(&(layer, expert))
                    .ok_or_else(|| mixed_error("expert missing after ensure"))?;
                let mid_off = (slot * Q80_HGRAVS_RANK * 4) as u64;
                let down_off = (slot * QWEN80_HIDDEN * 4) as u64;
                dispatch_factor(
                    &mut tcb,
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
                    trip.down.left_code_off,
                    trip.down.left_scale_off,
                    crate::decode_family::matvec_hgravs(),
                )?;
                native.hgravs_factor_dispatches =
                    native.hgravs_factor_dispatches.saturating_add(1);
            }
            let timing = tcb.commit_and_wait_timed()?;
            Self::note_timing(stages, native, &timing, MixedGpuOrgan::MoeRouted);
        } else {
            native.hgravs_factor_dispatches = native.hgravs_factor_dispatches.saturating_add(10);
        }

        let mut down = read_f32(&self.wave.down, 10 * QWEN80_HIDDEN);
        if !degrade.is_identity() {
            for (slot, &expert) in ids.iter().enumerate() {
                let base = slot * QWEN80_HIDDEN;
                mix_matched_cosine(
                    &mut down[base..base + QWEN80_HIDDEN],
                    degrade.down_mix,
                    mix_seed(degrade.mix_seed, layer, expert, 2),
                );
            }
        }
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

    fn production_kernel_for(weight: &GpuWeight) -> (&'static str, (u32, u32, u32)) {
        match weight {
            GpuWeight::Binary(body) => {
                if qwen80_recon_fuse_enabled() {
                    ("q80_binary_group_matvec_tg256", tg256_grid(body.rows))
                } else {
                    (crate::decode_family::MATVEC_BINARY, (body.rows, 1, 1))
                }
            }
            GpuWeight::Residual(body) => {
                if qwen80_recon_fuse_enabled() {
                    (
                        "q80_binary_group_csr_matvec_tg256",
                        tg256_grid(body.binary.rows),
                    )
                } else {
                    (
                        crate::decode_family::MATVEC_BINARY,
                        (body.binary.rows, 1, 1),
                    )
                }
            }
            GpuWeight::Uniform(body) => {
                if qwen80_recon_fuse_enabled() {
                    if body.bits == 8 && body.cols >= 2048 {
                        ("q80_uniform8_matvec_tg256", tg256_grid(body.rows))
                    } else if body.bits == 8 {
                        ("q80_uniform8_matvec_simd_bytes", simd8_grid(body.rows))
                    } else if body.bits == 3 {
                        ("q80_hgravs01_factor_matvec_simd3", simd8_grid(body.rows))
                    } else {
                        ("q80_hgravs01_factor_matvec_simd", simd8_grid(body.rows))
                    }
                } else {
                    (crate::decode_family::MATVEC_HGRAVS, (body.rows, 1, 1))
                }
            }
            GpuWeight::Hgravs(body) => {
                if qwen80_recon_fuse_enabled() {
                    (
                        "q80_hgravs01_factor_matvec_simd3",
                        simd8_grid(body.right_rows),
                    )
                } else {
                    (
                        crate::decode_family::MATVEC_HGRAVS,
                        (body.right_rows, 1, 1),
                    )
                }
            }
        }
    }

    fn probe_kernel_for(
        weight: &GpuWeight,
        mode: MixedProbeMode,
    ) -> Result<(&'static str, (u32, u32, u32))> {
        let (full, grid) = Self::production_kernel_for(weight);
        let name = match (mode, full) {
            (MixedProbeMode::Full, _) => full,
            (MixedProbeMode::Addr, "q80_binary_group_matvec_tg256") => {
                "q80_binary_group_matvec_tg256_addr_probe"
            }
            (MixedProbeMode::Decode, "q80_binary_group_matvec_tg256") => {
                "q80_binary_group_matvec_tg256_decode_probe"
            }
            (MixedProbeMode::Addr, "q80_binary_group_csr_matvec_tg256") => {
                "q80_binary_group_csr_matvec_tg256_addr_probe"
            }
            (MixedProbeMode::Decode, "q80_binary_group_csr_matvec_tg256") => {
                "q80_binary_group_csr_matvec_tg256_decode_probe"
            }
            (MixedProbeMode::Addr, "q80_uniform8_matvec_tg256") => {
                "q80_uniform8_matvec_tg256_addr_probe"
            }
            (MixedProbeMode::Decode, "q80_uniform8_matvec_tg256") => {
                "q80_uniform8_matvec_tg256_decode_probe"
            }
            (MixedProbeMode::Addr, "q80_uniform8_matvec_simd_bytes") => {
                "q80_uniform8_matvec_simd_bytes_addr_probe"
            }
            (MixedProbeMode::Decode, "q80_uniform8_matvec_simd_bytes") => {
                "q80_uniform8_matvec_simd_bytes_decode_probe"
            }
            (MixedProbeMode::Addr, "q80_hgravs01_factor_matvec_simd3") => {
                "q80_hgravs01_factor_matvec_simd3_addr_probe"
            }
            (MixedProbeMode::Decode, "q80_hgravs01_factor_matvec_simd3") => {
                "q80_hgravs01_factor_matvec_simd3_decode_probe"
            }
            (MixedProbeMode::BinarySimd, _) => "q80_binary_group_matvec_simd",
            (MixedProbeMode::GkBinarySimd, _) => crate::decode_family::MATVEC_BINARY_SIMD,
            (MixedProbeMode::Q8SimdBytes, _) => "q80_uniform8_matvec_simd_bytes",
            (MixedProbeMode::GkHgravsSimd, _) => crate::decode_family::MATVEC_HGRAVS_SIMD,
            (mode, kernel) => {
                return Err(mixed_error(format!(
                    "no probe {mode:?} for production kernel {kernel}"
                )))
            }
        };
        let grid = match mode {
            MixedProbeMode::BinarySimd | MixedProbeMode::GkBinarySimd => {
                let rows = match weight {
                    GpuWeight::Binary(b) => b.rows,
                    GpuWeight::Residual(b) => b.binary.rows,
                    GpuWeight::Uniform(b) => b.rows,
                    GpuWeight::Hgravs(b) => b.right_rows,
                };
                simd8_grid(rows)
            }
            MixedProbeMode::Q8SimdBytes => {
                let rows = match weight {
                    GpuWeight::Uniform(b) => b.rows,
                    GpuWeight::Hgravs(b) => b.right_rows,
                    GpuWeight::Binary(b) => b.rows,
                    GpuWeight::Residual(b) => b.binary.rows,
                };
                simd8_grid(rows)
            }
            MixedProbeMode::GkHgravsSimd => {
                let rows = match weight {
                    GpuWeight::Hgravs(b) => b.right_rows,
                    GpuWeight::Uniform(b) => b.rows,
                    other => {
                        return Err(mixed_error(format!(
                            "gk hgravs simd is not defined for {other:?}"
                        )))
                    }
                };
                simd8_grid(rows)
            }
            _ => grid,
        };
        Ok((name, grid))
    }

    fn encode_weight_mode(
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        weight: &GpuWeight,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
        mode: MixedProbeMode,
    ) -> Result<()> {
        match weight {
            GpuWeight::Hgravs(body) if mode == MixedProbeMode::Full => {
                let mid_rows = body.right_rows;
                // Reuse output as dest of L after writing R into a mid slice of output
                // is unsafe if sizes differ. Isolated Full for hgravs only times R
                // (same launch as one factor). L is measured separately by name.
                dispatch_factor(
                    tcb,
                    &body.right_codes,
                    &body.right_scales,
                    input,
                    0,
                    output,
                    0,
                    body.right_rows,
                    body.right_cols,
                    body.group_size,
                    body.bits,
                    body.bound,
                    body.right_code_off,
                    body.right_scale_off,
                    crate::decode_family::MATVEC_HGRAVS,
                )?;
                let _ = mid_rows;
                return Ok(());
            }
            GpuWeight::Residual(body)
                if matches!(mode, MixedProbeMode::Addr | MixedProbeMode::Decode) =>
            {
                let (kernel, grid) = Self::probe_kernel_for(weight, mode)?;
                tcb.dispatch_threads(kernel, grid, (256, 1, 1), |enc| {
                    encode_binary_csr(enc, body, input, output, 0);
                })?;
                return Ok(());
            }
            _ => {}
        }
        let (kernel, grid) = Self::probe_kernel_for(weight, mode)?;
        match weight {
            GpuWeight::Binary(body) => tcb.dispatch_threads(kernel, grid, (256, 1, 1), |enc| {
                encode_binary(enc, body, input, output, 0);
            }),
            GpuWeight::Residual(body) => tcb.dispatch_threads(kernel, grid, (256, 1, 1), |enc| {
                encode_binary(enc, &body.binary, input, output, 0);
            }),
            GpuWeight::Uniform(body) => tcb.dispatch_threads(kernel, grid, (256, 1, 1), |enc| {
                encode_factor(
                    enc,
                    &body.codes,
                    &body.scales,
                    input,
                    0,
                    output,
                    0,
                    body.rows,
                    body.cols,
                    body.group_size,
                    body.bits,
                    body.bound,
                    body.code_off,
                    body.scale_off,
                );
            }),
            GpuWeight::Hgravs(body) => tcb.dispatch_threads(kernel, grid, (256, 1, 1), |enc| {
                encode_factor(
                    enc,
                    &body.right_codes,
                    &body.right_scales,
                    input,
                    0,
                    output,
                    0,
                    body.right_rows,
                    body.right_cols,
                    body.group_size,
                    body.bits,
                    body.bound,
                    body.right_code_off,
                    body.right_scale_off,
                );
            }),
        }
    }

    fn measure_named(
        &mut self,
        name: &str,
        mode: MixedProbeMode,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let weight = self
            .weights
            .get(name)
            .ok_or_else(|| mixed_error(format!("isolated probe missing uploaded {name}")))?
            .clone();
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        Self::encode_weight_mode(
            &mut tcb,
            &weight,
            &self.scratch.input,
            &self.scratch.out_a,
            mode,
        )?;
        tcb.commit_and_wait_timed()
    }

    fn measure_class(
        &mut self,
        class: &str,
        mode: MixedProbeMode,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let names: Vec<String> = self
            .weights
            .keys()
            .filter(|name| mixed_weight_class(name) == class)
            .cloned()
            .collect();
        if names.is_empty() {
            return Err(mixed_error(format!(
                "no uploaded weights for isolated class {class}"
            )));
        }
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        for name in &names {
            let weight = self
                .weights
                .get(name)
                .ok_or_else(|| mixed_error("weight vanished"))?
                .clone();
            Self::encode_weight_mode(
                &mut tcb,
                &weight,
                &self.scratch.input,
                &self.scratch.out_a,
                mode,
            )?;
        }
        tcb.commit_and_wait_timed()
    }

    fn measure_one_routed_wave(
        &mut self,
        mode: MixedProbeMode,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let key = *self
            .experts
            .keys()
            .next()
            .ok_or_else(|| mixed_error("no bound expert for isolated routed wave"))?;
        let trip = self
            .experts
            .get(&key)
            .ok_or_else(|| mixed_error("expert vanished"))?;
        let mut tcb = crate::metal::TokenCommandBuffer::new(&self.context);
        match mode {
            MixedProbeMode::Full => {
                dispatch_binary(
                    &mut tcb,
                    &trip.gate,
                    &self.wave.input,
                    &self.wave.gate,
                    0,
                    "q80_binary_group_matvec",
                )?;
                dispatch_residual(
                    &mut tcb,
                    &trip.up,
                    &self.wave.input,
                    &self.wave.up,
                    0,
                    "q80_binary_group_matvec",
                )?;
                dispatch_factor(
                    &mut tcb,
                    &trip.down.right_codes,
                    &trip.down.right_scales,
                    &self.wave.act,
                    0,
                    &self.wave.mid,
                    0,
                    trip.down.right_rows,
                    trip.down.right_cols,
                    trip.down.group_size,
                    trip.down.bits,
                    trip.down.bound,
                    trip.down.right_code_off,
                    trip.down.right_scale_off,
                    crate::decode_family::MATVEC_HGRAVS,
                )?;
                dispatch_factor(
                    &mut tcb,
                    &trip.down.left_codes,
                    &trip.down.left_scales,
                    &self.wave.mid,
                    0,
                    &self.wave.down,
                    0,
                    trip.down.left_rows,
                    trip.down.left_cols,
                    trip.down.group_size,
                    trip.down.bits,
                    trip.down.bound,
                    trip.down.left_code_off,
                    trip.down.left_scale_off,
                    crate::decode_family::MATVEC_HGRAVS,
                )?;
            }
            MixedProbeMode::Addr | MixedProbeMode::Decode => {
                Self::encode_weight_mode(
                    &mut tcb,
                    &GpuWeight::Binary(trip.gate.clone()),
                    &self.wave.input,
                    &self.wave.gate,
                    mode,
                )?;
                Self::encode_weight_mode(
                    &mut tcb,
                    &GpuWeight::Residual(trip.up.clone()),
                    &self.wave.input,
                    &self.wave.up,
                    mode,
                )?;
                Self::encode_weight_mode(
                    &mut tcb,
                    &GpuWeight::Hgravs(trip.down.clone()),
                    &self.wave.act,
                    &self.wave.mid,
                    mode,
                )?;
            }
            other => {
                return Err(mixed_error(format!(
                    "routed wave probe does not support {other:?}"
                )))
            }
        }
        tcb.commit_and_wait_timed()
    }
}

fn mixed_weight_class(name: &str) -> &'static str {
    if name.contains("linear_attn") {
        "dn"
    } else if name.contains("self_attn") {
        "gqa"
    } else if name.contains("shared_expert_gate") {
        "combine_gate"
    } else if name.contains("shared_expert") {
        "shared"
    } else if name.contains("mlp.gate.weight") {
        "router"
    } else if name.contains("lm_head") {
        "lm_head"
    } else if name.contains("experts.") {
        "routed"
    } else {
        "other"
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
    pub degrade: MixedDegradeConfig,
    #[cfg(target_os = "macos")]
    metal: Option<MetalMixedAccel>,
    pub metal_error: Option<String>,
}

impl Qwen80MixedHybridDecodeSession {
    pub fn new(mut catalog: Qwen80MixedStreamingCatalog, max_seq_len: usize) -> Result<Self> {
        if catalog.tensor_count() != QWEN80_MIXED_EXPECTED_TENSOR_COUNT {
            return Err(mixed_error(format!(
                "catalog tensor count {} != {QWEN80_MIXED_EXPECTED_TENSOR_COUNT}",
                catalog.tensor_count()
            )));
        }
        if qwen80_host_facet1_enabled() {
            catalog.admit_session()?;
        }
        #[cfg(target_os = "macos")]
        let (metal, metal_error) = match MetalMixedAccel::new(max_seq_len) {
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
            degrade: MixedDegradeConfig::default(),
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

    pub fn set_degrade(&mut self, degrade: MixedDegradeConfig) {
        self.degrade = degrade;
    }

    pub fn reset_state(&mut self) {
        self.state.reset();
        #[cfg(target_os = "macos")]
        if let Some(metal) = self.metal.as_mut() {
            if let Some(act) = metal.activations.as_mut() {
                act.zero_state();
            }
        }
    }

    pub fn cached_geometry_count(&self) -> usize {
        self.cache.geometry.len()
    }

    pub fn exclusive_snap(&self) -> MixedExclusiveSnap {
        MixedExclusiveSnap {
            encode_ns: self.stages.cb_encode_ns,
            submit_ns: self.stages.cb_submit_ns,
            wait_ns: self.stages.cb_wait_ns,
            gpu_ns: self.stages.gpu_matvec_ns,
            wait_minus_gpu_ns: self.stages.cb_wait_minus_gpu_ns,
            cbs: self.native.command_buffers,
            dispatches: self.native.compute_dispatches,
            timestamps_missing: self.stages.gpu_matvec_timestamps_missing,
            gpu_organ: self.stages.gpu_organ,
            host_excl: self.stages.host_excl,
        }
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
                self.matvec_named(name, &input, &mut got, MixedGpuOrgan::Other)?;
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

    fn packed(&mut self, name: &str) -> Result<MixedPackedTensor> {
        self.native.packed_calls = self.native.packed_calls.saturating_add(1);
        let started = Instant::now();
        let packed = self.catalog.load_packed(name)?;
        add_ns(&mut self.stages.host_excl.catalog_reparse, started);
        if let Ok(rc) = packed.rows_cols() {
            self.cache.geometry.entry(name.to_owned()).or_insert(rc);
        }
        Ok(packed)
    }

    #[cfg(target_os = "macos")]
    fn weight_is_bound(&self, name: &str) -> bool {
        self.metal
            .as_ref()
            .map(|metal| metal.weights.contains_key(name))
            .unwrap_or(false)
    }

    #[cfg(target_os = "macos")]
    fn remember_bound_geometry(&mut self, name: &str) {
        if let Some(metal) = self.metal.as_ref() {
            if let Ok(rc) = metal.weight_geometry(name) {
                self.cache.geometry.entry(name.to_owned()).or_insert(rc);
            }
        }
    }

    fn matvec_named(
        &mut self,
        name: &str,
        input: &[f32],
        output: &mut [f32],
        organ: MixedGpuOrgan,
    ) -> Result<()> {
        let started = Instant::now();
        #[cfg(target_os = "macos")]
        {
            let skip_packed = qwen80_cache_geom_enabled()
                && (self.weight_is_bound(name) || qwen80_host_facet1_enabled());
            if skip_packed {
                {
                    let Some(metal) = self.metal.as_mut() else {
                        return Err(mixed_error(format!(
                            "Metal required for {name}; refusing host mixed matvec"
                        )));
                    };
                    metal.ensure_named_weight(
                        &self.catalog,
                        name,
                        None,
                        &mut self.native,
                    )?;
                    metal.matvec(
                        name,
                        None,
                        input,
                        output,
                        &mut self.native,
                        &mut self.stages,
                        organ,
                    )?;
                }
                self.remember_bound_geometry(name);
                self.native.packed_skipped = self.native.packed_skipped.saturating_add(1);
                add_secs(&mut self.stages.mixed_matvec_secs, started);
                return Ok(());
            }
            let packed = self.packed(name)?;
            let Some(metal) = self.metal.as_mut() else {
                return Err(mixed_error(format!(
                    "Metal required for {name}; refusing host mixed matvec"
                )));
            };
            metal.ensure_named_weight(
                &self.catalog,
                name,
                Some(&packed),
                &mut self.native,
            )?;
            metal.matvec(
                name,
                Some(&packed),
                input,
                output,
                &mut self.native,
                &mut self.stages,
                organ,
            )?;
            add_secs(&mut self.stages.mixed_matvec_secs, started);
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (started, organ, input, output);
            let _ = self.packed(name)?;
            self.fallbacks.host_mixed_matvec = self.fallbacks.host_mixed_matvec.saturating_add(1);
            Err(mixed_error(format!(
                "refusing host mixed matvec for {name}"
            )))
        }
    }

    fn matvec_named_group(
        &mut self,
        names: &[&str],
        input: &[f32],
        outputs: &mut [&mut [f32]],
        organ: MixedGpuOrgan,
    ) -> Result<()> {
        if names.len() != outputs.len() {
            return Err(mixed_error("matvec group arity drifted"));
        }
        if !qwen80_host_facet2_enabled() || names.len() < 2 {
            for (name, out) in names.iter().zip(outputs.iter_mut()) {
                self.matvec_named(name, input, out, organ)?;
            }
            return Ok(());
        }
        let started = Instant::now();
        #[cfg(target_os = "macos")]
        {
            let skip_packed = qwen80_cache_geom_enabled()
                && (qwen80_host_facet1_enabled()
                    || names.iter().all(|name| self.weight_is_bound(name)));
            if skip_packed {
                {
                    let Some(metal) = self.metal.as_mut() else {
                        return Err(mixed_error("Metal required for grouped mixed matvec"));
                    };
                    for name in names {
                        metal.ensure_named_weight(
                            &self.catalog,
                            name,
                            None,
                            &mut self.native,
                        )?;
                    }
                    metal.matvec_group_bound(
                        names,
                        input,
                        outputs,
                        &mut self.native,
                        &mut self.stages,
                        organ,
                    )?;
                }
                for name in names {
                    self.remember_bound_geometry(name);
                }
                self.native.packed_skipped = self
                    .native
                    .packed_skipped
                    .saturating_add(names.len() as u64);
                add_secs(&mut self.stages.mixed_matvec_secs, started);
                return Ok(());
            }
            let mut packed = Vec::with_capacity(names.len());
            for name in names {
                packed.push(self.packed(name)?);
            }
            let Some(metal) = self.metal.as_mut() else {
                return Err(mixed_error("Metal required for grouped mixed matvec"));
            };
            metal.matvec_group_same_input(
                &self.catalog,
                names,
                &packed,
                input,
                outputs,
                &mut self.native,
                &mut self.stages,
                organ,
            )?;
            add_secs(&mut self.stages.mixed_matvec_secs, started);
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = started;
            for (name, out) in names.iter().zip(outputs.iter_mut()) {
                self.matvec_named(name, input, out, organ)?;
            }
            Ok(())
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
            let started = Instant::now();
            let cloned = existing.clone();
            add_ns(&mut self.stages.host_excl.vector_clone, started);
            return Ok(cloned);
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
        self.matvec_named_group(
            &[gate_name, up_name],
            input,
            &mut [&mut gate, &mut up],
            MixedGpuOrgan::MoeShared,
        )?;
        let silu_started = Instant::now();
        silu_mul(&gate, &up, &mut act);
        add_secs(&mut self.stages.activation.shared_swiglu_secs, silu_started);
        add_ns(&mut self.stages.host_excl.silu, silu_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.shared_swiglu =
            self.activation_counts.shared_swiglu.saturating_add(1);
        self.matvec_named(down_name, &act, &mut down, MixedGpuOrgan::MoeShared)?;
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
        add_ns(&mut self.stages.host_excl.dn_rms, rms_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        let qkvz_rows = layout.qkvz_projection_elements()?;
        let ba_rows = layout.ba_projection_elements()?;
        let mut projected_qkvz = vec![0.0f32; qkvz_rows];
        let mut projected_ba = vec![0.0f32; ba_rows];
        let qkvz_name = Self::layer_name(layer, "linear_attn.in_proj_qkvz.weight");
        let ba_name = Self::layer_name(layer, "linear_attn.in_proj_ba.weight");
        self.matvec_named_group(
            &[&qkvz_name, &ba_name],
            &rms,
            &mut [&mut projected_qkvz, &mut projected_ba],
            MixedGpuOrgan::DeltaNet,
        )?;
        let rearrange_started = Instant::now();
        let (raw_query, raw_key, raw_value, z) =
            source_qwen80_split_linear_qkvz(&projected_qkvz, &layout)?;
        let mut mixed_qkv = Vec::with_capacity(layout.conv_channels);
        mixed_qkv.extend_from_slice(&raw_query);
        mixed_qkv.extend_from_slice(&raw_key);
        mixed_qkv.extend_from_slice(&raw_value);
        add_ns(&mut self.stages.host_excl.dn_rearrange_l2, rearrange_started);
        let conv_w = self.vector(&Self::layer_name(layer, "linear_attn.conv1d.weight"))?;
        let conv_started = Instant::now();
        let (convolved_qkv, next_conv) = source_qwen80_causal_conv_step_dense(
            &mixed_qkv,
            &self.state.linear_conv[slot],
            &conv_w,
            &layout,
        )?;
        add_secs(&mut self.stages.activation.deltanet_conv_secs, conv_started);
        add_ns(&mut self.stages.host_excl.dn_conv, conv_started);
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
        let l2_started = Instant::now();
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
        add_ns(&mut self.stages.host_excl.dn_rearrange_l2, l2_started);
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
        add_ns(&mut self.stages.host_excl.dn_recurrent, recurrent_started);
        let kv_started = Instant::now();
        self.state.linear_conv[slot] = next_conv;
        add_ns(&mut self.stages.host_excl.gqa_kv_copy, kv_started);
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
        add_ns(&mut self.stages.host_excl.dn_gated, gated_started);
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
            MixedGpuOrgan::DeltaNet,
        )?;
        let mut residual = hidden.to_vec();
        let add_started = Instant::now();
        add_inplace(&mut residual, &mixer_output);
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            add_started,
        );
        add_ns(&mut self.stages.host_excl.dn_residual, add_started);
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
        add_ns(&mut self.stages.host_excl.gqa_rms, rms_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.gqa_input_layernorm =
            self.activation_counts.gqa_input_layernorm.saturating_add(1);
        let mut q_projection = vec![0.0f32; layout.q_proj_rows];
        let mut k_projection = vec![0.0f32; layout.kv_dim];
        let mut v_projection = vec![0.0f32; layout.kv_dim];
        let q_name = Self::layer_name(layer, "self_attn.q_proj.weight");
        let k_name = Self::layer_name(layer, "self_attn.k_proj.weight");
        let v_name = Self::layer_name(layer, "self_attn.v_proj.weight");
        self.matvec_named_group(
            &[&q_name, &k_name, &v_name],
            &rms,
            &mut [&mut q_projection, &mut k_projection, &mut v_projection],
            MixedGpuOrgan::Gqa,
        )?;
        let q_norm = self.vector(&Self::layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = self.vector(&Self::layer_name(layer, "self_attn.k_norm.weight"))?;
        let interleave_started = Instant::now();
        let query_raw = qwen80_gqa_query_from_interleaved_q_projection(&q_projection, &layout)?;
        add_ns(&mut self.stages.host_excl.gqa_interleave, interleave_started);
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
        add_ns(&mut self.stages.host_excl.gqa_rope, rope_started);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(2);
        self.activation_counts.gqa_norm_rope =
            self.activation_counts.gqa_norm_rope.saturating_add(2);
        let start = position * layout.kv_dim;
        let end = start + layout.kv_dim;
        let kv_started = Instant::now();
        self.state.gqa_key[slot][start..end].copy_from_slice(&key_row);
        self.state.gqa_value[slot][start..end].copy_from_slice(&v_projection);
        add_ns(&mut self.stages.host_excl.gqa_kv_copy, kv_started);
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
        add_ns(&mut self.stages.host_excl.gqa_attn, attn_started);
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
            MixedGpuOrgan::Gqa,
        )?;
        let mut residual = hidden.to_vec();
        let add_started = Instant::now();
        add_inplace(&mut residual, &mixer_output);
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            add_started,
        );
        add_ns(&mut self.stages.host_excl.gqa_residual, add_started);
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
        add_ns(&mut self.stages.host_excl.post_rms, norm_op);
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
            MixedGpuOrgan::MoeRouter,
        )?;
        let route_op = Instant::now();
        let route = source_qwen80_topk_router(&router_logits)?;
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            route_op,
        );
        add_ns(&mut self.stages.host_excl.topk, route_op);
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
            let degrade = self.degrade.clone();
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
                &degrade,
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
            MixedGpuOrgan::MoeCombineGate,
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
        add_ns(&mut self.stages.host_excl.combine, combine_op);
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
        add_ns(&mut self.stages.host_excl.final_rms, norm_op);
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        let mut logits = vec![0.0f32; QWEN80_VOCAB];
        self.matvec_named(
            "lm_head.weight",
            &normed,
            &mut logits,
            MixedGpuOrgan::Terminal,
        )?;
        let argmax_started = Instant::now();
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
        add_ns(&mut self.stages.host_excl.argmax, argmax_started);
        self.fallbacks.host_sample = self.fallbacks.host_sample.saturating_add(1);
        if !best_v.is_finite() {
            return Err(mixed_error("greedy sample saw no finite logit"));
        }
        Ok(best_i as u32)
    }

    #[cfg(target_os = "macos")]
    fn device_activations_live(&self) -> bool {
        self.metal
            .as_ref()
            .and_then(|metal| metal.activations.as_ref())
            .is_some()
            && qwen80_mixed_device_activations_enabled()
            && self.degrade.is_identity()
    }

    #[cfg(target_os = "macos")]
    fn new_token_cb(&self) -> Result<crate::metal::TokenCommandBuffer<'static>> {
        let metal = self
            .metal
            .as_ref()
            .ok_or_else(|| mixed_error("device token requires Metal"))?;
        // SAFETY: the Metal context is owned by the session and outlives every
        // command buffer created here. Callers drop the TCB before dropping
        // `metal`.
        let ctx: &'static crate::metal::MetalContext =
            unsafe { &*(&metal.context as *const crate::metal::MetalContext) };
        Ok(crate::metal::TokenCommandBuffer::new(ctx))
    }

    #[cfg(target_os = "macos")]
    fn snapshot_f32(buf: &crate::metal::PinnedBuffer, n: usize) -> Result<Vec<f32>> {
        if buf.length() < (n * std::mem::size_of::<f32>()) as u64 {
            return Err(mixed_error("device snapshot buffer is short"));
        }
        let slice = unsafe { std::slice::from_raw_parts(buf.contents() as *const f32, n) };
        Ok(slice.to_vec())
    }

    #[cfg(target_os = "macos")]
    fn device_vector_buf(&mut self, name: &str) -> Result<crate::metal::PinnedBuffer> {
        if let Some(existing) = self
            .metal
            .as_ref()
            .and_then(|metal| metal.activations.as_ref())
            .and_then(|act| act.vectors.get(name))
            .cloned()
        {
            return Ok(existing);
        }
        let host = self.vector(name)?;
        let metal = self
            .metal
            .as_mut()
            .ok_or_else(|| mixed_error("device vector upload requires Metal"))?;
        let buf = metal
            .context
            .new_buffer_with_bytes_checked(bytemuck::cast_slice(&host))?;
        if let Some(act) = metal.activations.as_mut() {
            act.vectors.insert(name.to_owned(), buf.clone());
        }
        Ok(buf)
    }

    #[cfg(target_os = "macos")]
    fn encode_mixed_matvec(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        name: &str,
        input: &crate::metal::PinnedBuffer,
        output: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let already = self
            .metal
            .as_ref()
            .map(|metal| metal.weights.contains_key(name))
            .unwrap_or(false);
        if !already {
            let packed = self.packed(name)?;
            let metal = self
                .metal
                .as_mut()
                .ok_or_else(|| mixed_error("device matvec requires Metal"))?;
            metal.ensure_named_weight(&self.catalog, name, Some(&packed), &mut self.native)?;
        }
        let weight = self
            .metal
            .as_ref()
            .and_then(|metal| metal.weights.get(name).cloned())
            .ok_or_else(|| mixed_error(format!("{name} missing after ensure")))?;
        MetalMixedAccel::encode_weight(tcb, &weight, input, output, &mut self.native)
    }

    #[cfg(target_os = "macos")]
    fn encode_residual_rmsnorm(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        input: &crate::metal::PinnedBuffer,
        weight_name: &str,
        output: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let weight = self.device_vector_buf(weight_name)?;
        tcb.dispatch_threads(
            "qwen80_residual_rmsnorm_f32",
            (256, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(input), 0);
                encoder.set_buffer(1, Some(&weight), 0);
                encoder.set_buffer(2, Some(output), 0);
                set_u32(encoder, 3, QWEN80_HIDDEN as u32);
                encoder.set_bytes(
                    4,
                    std::mem::size_of::<f32>() as u64,
                    &QWEN80_RMS_EPS as *const f32 as *const _,
                );
                encoder.set_threadgroup_memory_length(0, 256 * 4);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_shared_mlp(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        input: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let (gate, up, act, shared) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            (
                actw.gate.clone(),
                actw.up.clone(),
                actw.act.clone(),
                actw.shared.clone(),
            )
        };
        let sandwich = Instant::now();
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert.gate_proj.weight"),
            input,
            &gate,
        )?;
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert.up_proj.weight"),
            input,
            &up,
        )?;
        tcb.dispatch_threads(
            "qwen80_silu_mul_f32",
            (QWEN80_MOE_INTERMEDIATE as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&gate), 0);
                encoder.set_buffer(1, Some(&up), 0);
                encoder.set_buffer(2, Some(&act), 0);
                set_u32(encoder, 3, QWEN80_MOE_INTERMEDIATE as u32);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert.down_proj.weight"),
            &act,
            &shared,
        )?;
        add_secs(
            &mut self.stages.activation.shared_mlp_sandwich_secs,
            sandwich,
        );
        self.activation_counts.shared_swiglu =
            self.activation_counts.shared_swiglu.saturating_add(1);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_deltanet_mixer(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let layout = Qwen80CanonicalLinearDeltaNetLayout::source_exact();
        let slot = self.state.linear_slot_for_layer(layer)?;
        let conv_off = (slot * layout.conv_state_elements()? * std::mem::size_of::<f32>()) as u64;
        let rec_off = (slot * layout.recurrent_state_elements()? * std::mem::size_of::<f32>()) as u64;
        let (
            normalized,
            qkvz,
            ba,
            repeated_q,
            repeated_k,
            conv_v,
            z,
            decay,
            beta,
            rec_out,
            gated,
            mixer,
            first_residual,
            conv_state,
            rec_state,
        ) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            (
                actw.normalized.clone(),
                actw.qkvz.clone(),
                actw.ba.clone(),
                actw.repeated_q.clone(),
                actw.repeated_k.clone(),
                actw.conv_v.clone(),
                actw.z.clone(),
                actw.decay.clone(),
                actw.beta.clone(),
                actw.rec_out.clone(),
                actw.gated.clone(),
                actw.mixer.clone(),
                actw.first_residual.clone(),
                actw.linear_conv.clone(),
                actw.linear_recurrent.clone(),
            )
        };
        self.encode_residual_rmsnorm(
            tcb,
            hidden,
            &Self::layer_name(layer, "input_layernorm.weight"),
            &normalized,
        )?;
        add_ns(&mut self.stages.host_excl.dn_rms, Instant::now());
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "linear_attn.in_proj_qkvz.weight"),
            &normalized,
            &qkvz,
        )?;
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "linear_attn.in_proj_ba.weight"),
            &normalized,
            &ba,
        )?;
        let conv_w = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.conv1d.weight"))?;
        let conv_started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_qkvz_rearrange_conv_l2_f32",
            (256, layout.key_heads as u32, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&qkvz), 0);
                encoder.set_buffer(1, Some(&conv_w), 0);
                encoder.set_buffer(2, Some(&conv_state), conv_off);
                encoder.set_buffer(3, Some(&repeated_q), 0);
                encoder.set_buffer(4, Some(&repeated_k), 0);
                encoder.set_buffer(5, Some(&conv_v), 0);
                encoder.set_buffer(6, Some(&z), 0);
                set_u32(encoder, 7, layout.key_heads as u32);
                set_u32(encoder, 8, layout.value_heads_per_key_head as u32);
                set_u32(encoder, 9, layout.key_head_dim as u32);
                set_u32(encoder, 10, layout.value_head_dim as u32);
                set_u32(encoder, 11, layout.conv_kernel as u32);
                encoder.set_bytes(
                    12,
                    std::mem::size_of::<f32>() as u64,
                    &QWEN80_RMS_EPS as *const f32 as *const _,
                );
                encoder.set_threadgroup_memory_length(0, 4 * 256 * 4);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_secs(&mut self.stages.activation.deltanet_conv_secs, conv_started);
        add_ns(&mut self.stages.host_excl.dn_conv, conv_started);
        self.activation_counts.deltanet_conv =
            self.activation_counts.deltanet_conv.saturating_add(1);
        let a_log = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.A_log"))?;
        let dt_bias = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.dt_bias"))?;
        let recurrent_started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_ba_to_decay_beta_f32",
            (layout.value_heads as u32, 1, 1),
            (layout.value_heads.min(32).max(1) as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&ba), 0);
                encoder.set_buffer(1, Some(&a_log), 0);
                encoder.set_buffer(2, Some(&dt_bias), 0);
                encoder.set_buffer(3, Some(&decay), 0);
                encoder.set_buffer(4, Some(&beta), 0);
                set_u32(encoder, 5, layout.key_heads as u32);
                set_u32(encoder, 6, layout.value_heads_per_key_head as u32);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        tcb.dispatch_threads(
            "qwen80_gated_delta_decode_tg",
            (layout.key_head_dim as u32, layout.value_heads as u32, 1),
            (layout.key_head_dim as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&rec_state), rec_off);
                encoder.set_buffer(1, Some(&repeated_q), 0);
                encoder.set_buffer(2, Some(&repeated_k), 0);
                encoder.set_buffer(3, Some(&conv_v), 0);
                encoder.set_buffer(4, Some(&decay), 0);
                encoder.set_buffer(5, Some(&beta), 0);
                encoder.set_buffer(6, Some(&rec_out), 0);
                set_u32(encoder, 7, layout.value_heads as u32);
                set_u32(encoder, 8, layout.key_head_dim as u32);
                set_u32(encoder, 9, layout.value_head_dim as u32);
                encoder.set_threadgroup_memory_length(0, 128 * 4);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_secs(
            &mut self.stages.activation.deltanet_recurrent_secs,
            recurrent_started,
        );
        add_ns(&mut self.stages.host_excl.dn_recurrent, recurrent_started);
        self.activation_counts.deltanet_recurrent = self
            .activation_counts
            .deltanet_recurrent
            .saturating_add(1);
        let norm_w = self.device_vector_buf(&Self::layer_name(layer, "linear_attn.norm.weight"))?;
        tcb.dispatch_threads(
            "qwen80_deltanet_gated_rmsnorm_f32",
            (layout.value_heads as u32, 1, 1),
            (layout.value_heads.min(32).max(1) as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&rec_out), 0);
                encoder.set_buffer(1, Some(&z), 0);
                encoder.set_buffer(2, Some(&norm_w), 0);
                encoder.set_buffer(3, Some(&gated), 0);
                set_u32(encoder, 4, layout.value_heads as u32);
                set_u32(encoder, 5, layout.value_head_dim as u32);
                encoder.set_bytes(
                    6,
                    std::mem::size_of::<f32>() as u64,
                    &QWEN80_RMS_EPS as *const f32 as *const _,
                );
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "linear_attn.out_proj.weight"),
            &gated,
            &mixer,
        )?;
        tcb.dispatch_threads(
            "qwen80_add_residual_f32",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(hidden), 0);
                encoder.set_buffer(1, Some(&mixer), 0);
                encoder.set_buffer(2, Some(&first_residual), 0);
                set_u32(encoder, 3, QWEN80_HIDDEN as u32);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_gqa_mixer(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let layout = Qwen80CanonicalGqaLayout::source_exact();
        let slot = self.state.gqa_slot_for_layer(layer)?;
        let position = self.state.position;
        if position >= self.state.max_seq_len {
            return Err(mixed_error(format!(
                "GQA position {position} exceeds max_seq_len {}",
                self.state.max_seq_len
            )));
        }
        let slot_elems = self.state.max_seq_len * layout.kv_dim;
        let cache_off = (slot * slot_elems * std::mem::size_of::<f32>()) as u64;
        let (
            normalized,
            q_proj,
            k_proj,
            v_proj,
            query,
            attn,
            gated_attn,
            mixer,
            first_residual,
            gqa_key,
            gqa_value,
        ) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            (
                actw.normalized.clone(),
                actw.q_proj.clone(),
                actw.k_proj.clone(),
                actw.v_proj.clone(),
                actw.query.clone(),
                actw.attn.clone(),
                actw.gated_attn.clone(),
                actw.mixer.clone(),
                actw.first_residual.clone(),
                actw.gqa_key.clone(),
                actw.gqa_value.clone(),
            )
        };
        let ln_started = Instant::now();
        self.encode_residual_rmsnorm(
            tcb,
            hidden,
            &Self::layer_name(layer, "input_layernorm.weight"),
            &normalized,
        )?;
        add_secs(
            &mut self.stages.activation.gqa_input_layernorm_secs,
            ln_started,
        );
        add_ns(&mut self.stages.host_excl.gqa_rms, ln_started);
        self.activation_counts.gqa_input_layernorm = self
            .activation_counts
            .gqa_input_layernorm
            .saturating_add(1);
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.q_proj.weight"),
            &normalized,
            &q_proj,
        )?;
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.k_proj.weight"),
            &normalized,
            &k_proj,
        )?;
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.v_proj.weight"),
            &normalized,
            &v_proj,
        )?;
        let q_norm = self.device_vector_buf(&Self::layer_name(layer, "self_attn.q_norm.weight"))?;
        let k_norm = self.device_vector_buf(&Self::layer_name(layer, "self_attn.k_norm.weight"))?;
        let rope_started = Instant::now();
        tcb.dispatch_threads(
            "qwen80_gqa_qk_norm_rope_cache_f32",
            (layout.query_heads as u32, 1, 1),
            (layout.query_heads.min(16).max(1) as u32, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&q_proj), 0);
                encoder.set_buffer(1, Some(&k_proj), 0);
                encoder.set_buffer(2, Some(&v_proj), 0);
                encoder.set_buffer(3, Some(&q_norm), 0);
                encoder.set_buffer(4, Some(&k_norm), 0);
                encoder.set_buffer(5, Some(&query), 0);
                encoder.set_buffer(6, Some(&gqa_key), cache_off);
                encoder.set_buffer(7, Some(&gqa_value), cache_off);
                set_u32(encoder, 8, position as u32);
                set_u32(encoder, 9, layout.query_heads as u32);
                set_u32(encoder, 10, layout.key_value_heads as u32);
                set_u32(encoder, 11, layout.head_dim as u32);
                set_u32(encoder, 12, layout.rotary_dim as u32);
                encoder.set_bytes(
                    13,
                    std::mem::size_of::<f32>() as u64,
                    &QWEN80_DEVICE_ROPE_THETA as *const f32 as *const _,
                );
                encoder.set_bytes(
                    14,
                    std::mem::size_of::<f32>() as u64,
                    &QWEN80_RMS_EPS as *const f32 as *const _,
                );
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        add_secs(&mut self.stages.activation.gqa_norm_rope_secs, rope_started);
        add_ns(&mut self.stages.host_excl.gqa_rope, rope_started);
        self.activation_counts.gqa_norm_rope =
            self.activation_counts.gqa_norm_rope.saturating_add(1);
        crate::kernels::mha_decode_f32_tcb(
            tcb,
            &query,
            &gqa_key,
            cache_off as usize,
            &gqa_value,
            cache_off as usize,
            &attn,
            position + 1,
            layout.head_dim,
            layout.query_heads,
            layout.key_value_heads,
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        tcb.dispatch_threads(
            "qwen80_attention_apply_sigmoid_gate",
            (layout.query_dim as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&attn), 0);
                encoder.set_buffer(1, Some(&q_proj), 0);
                encoder.set_buffer(2, Some(&gated_attn), 0);
                set_u32(encoder, 3, layout.query_dim as u32);
                set_u32(encoder, 4, layout.head_dim as u32);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "self_attn.o_proj.weight"),
            &gated_attn,
            &mixer,
        )?;
        tcb.dispatch_threads(
            "qwen80_add_residual_f32",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(hidden), 0);
                encoder.set_buffer(1, Some(&mixer), 0);
                encoder.set_buffer(2, Some(&first_residual), 0);
                set_u32(encoder, 3, QWEN80_HIDDEN as u32);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_mixer_into(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<MixedGpuOrgan> {
        match qwen80_layer_kind(layer)? {
            Qwen80LayerKind::LinearAttention => {
                self.encode_deltanet_mixer(tcb, layer, hidden)?;
                Ok(MixedGpuOrgan::DeltaNet)
            }
            Qwen80LayerKind::FullAttention => {
                self.encode_gqa_mixer(tcb, layer, hidden)?;
                Ok(MixedGpuOrgan::Gqa)
            }
        }
    }

    #[cfg(target_os = "macos")]
    fn encode_moe_prefix(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        postnorm: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let (first_residual, router_logits) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            (actw.first_residual.clone(), actw.router_logits.clone())
        };
        self.encode_residual_rmsnorm(
            tcb,
            &first_residual,
            &Self::layer_name(layer, "post_attention_layernorm.weight"),
            postnorm,
        )?;
        self.encode_shared_mlp(tcb, layer, postnorm)?;
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.gate.weight"),
            postnorm,
            &router_logits,
        )?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_suffix_into(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        layer: usize,
        hidden: &crate::metal::PinnedBuffer,
        postnorm: &crate::metal::PinnedBuffer,
        ids: &[u16],
        weights: &[f32],
    ) -> Result<()> {
        let (shared, shared_logit, first_residual, routed) = {
            let metal = self
                .metal
                .as_mut()
                .ok_or_else(|| mixed_error("device token requires Metal"))?;
            metal.encode_routed_wave_into(
                tcb,
                layer,
                ids,
                weights,
                postnorm,
                &mut self.native,
            )?;
            let actw = metal
                .activations
                .as_ref()
                .ok_or_else(|| mixed_error("device activations missing"))?;
            (
                actw.shared.clone(),
                actw.shared_logit.clone(),
                actw.first_residual.clone(),
                metal.scratch.combined.clone(),
            )
        };
        self.encode_mixed_matvec(
            tcb,
            &Self::layer_name(layer, "mlp.shared_expert_gate.weight"),
            postnorm,
            &shared_logit,
        )?;
        tcb.dispatch_threads(
            "qwen80_moe_combine_second_residual_f32",
            (QWEN80_HIDDEN as u32, 1, 1),
            (256, 1, 1),
            |encoder| {
                encoder.set_buffer(0, Some(&first_residual), 0);
                encoder.set_buffer(1, Some(&routed), 0);
                encoder.set_buffer(2, Some(&shared), 0);
                encoder.set_buffer(3, Some(&shared_logit), 0);
                encoder.set_buffer(4, Some(hidden), 0);
                set_u32(encoder, 5, QWEN80_HIDDEN as u32);
            },
        )?;
        self.native.device_activation_dispatches =
            self.native.device_activation_dispatches.saturating_add(1);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn host_route_and_bind(&mut self, layer: usize) -> Result<([u16; 10], [f32; 10])> {
        let router_logits = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            Self::snapshot_f32(&actw.router_logits, QWEN80_EXPERTS)?
        };
        let route_op = Instant::now();
        let route = source_qwen80_topk_router(&router_logits)?;
        add_ns(&mut self.stages.host_excl.topk, route_op);
        add_secs(
            &mut self.stages.activation.other_host_activation_secs,
            route_op,
        );
        self.fallbacks.host_activation = self.fallbacks.host_activation.saturating_add(1);
        self.activation_counts.other_host_activation = self
            .activation_counts
            .other_host_activation
            .saturating_add(1);
        {
            let metal = self
                .metal
                .as_mut()
                .ok_or_else(|| mixed_error("device token requires Metal"))?;
            for &expert in &route.ids {
                metal.ensure_expert(
                    &self.catalog,
                    layer,
                    expert,
                    &mut self.native,
                    &mut self.stages,
                )?;
            }
        }
        Ok((route.ids, route.weights))
    }

    #[cfg(target_os = "macos")]
    fn commit_device_cb(
        &mut self,
        tcb: crate::metal::TokenCommandBuffer<'static>,
        organ: MixedGpuOrgan,
    ) -> Result<()> {
        let timing = tcb.commit_and_wait_timed()?;
        MetalMixedAccel::note_timing(&mut self.stages, &mut self.native, &timing, organ);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn encode_terminal_into(
        &mut self,
        tcb: &mut crate::metal::TokenCommandBuffer<'_>,
        hidden: &crate::metal::PinnedBuffer,
    ) -> Result<()> {
        let (normalized, logits) = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            (actw.normalized.clone(), actw.logits.clone())
        };
        self.encode_residual_rmsnorm(tcb, hidden, "model.norm.weight", &normalized)?;
        self.encode_mixed_matvec(tcb, "lm_head.weight", &normalized, &logits)?;
        Ok(())
    }

    #[cfg(target_os = "macos")]
    fn sample_logits(&mut self) -> Result<u32> {
        let logits_buf = {
            let actw = self
                .metal
                .as_ref()
                .and_then(|m| m.activations.as_ref())
                .ok_or_else(|| mixed_error("device activations missing"))?;
            actw.logits.clone()
        };
        let logits = Self::snapshot_f32(&logits_buf, QWEN80_VOCAB)?;
        let argmax_started = Instant::now();
        let mut best_i = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (index, &value) in logits.iter().take(QWEN80_TOKENIZER_VOCAB).enumerate() {
            if value > best_v || (value == best_v && index < best_i) {
                best_v = value;
                best_i = index;
            }
        }
        add_ns(&mut self.stages.host_excl.argmax, argmax_started);
        self.fallbacks.host_sample = self.fallbacks.host_sample.saturating_add(1);
        if !best_v.is_finite() {
            return Err(mixed_error("greedy sample saw no finite logit"));
        }
        Ok(best_i as u32)
    }

    #[cfg(target_os = "macos")]
    fn forward_token_device(&mut self, token: u32) -> Result<u32> {
        let embed_started = Instant::now();
        let host_hidden = self.embed(token)?;
        let (hidden, postnorm) = {
            let metal = self
                .metal
                .as_mut()
                .ok_or_else(|| mixed_error("device token requires Metal"))?;
            let actw = metal
                .activations
                .as_ref()
                .ok_or_else(|| mixed_error("device activations missing"))?;
            let hidden = actw.hidden.clone();
            let postnorm = actw.postnorm.clone();
            write_f32(&hidden, &host_hidden);
            (hidden, postnorm)
        };
        add_secs(&mut self.stages.embed_secs, embed_started);
        add_ns(&mut self.stages.host_excl.embed, embed_started);

        let fuse = qwen80_mixed_collapse_fuse_enabled();
        let (mut pending_ids, mut pending_weights) = {
            let mut prefix = self.new_token_cb()?;
            prefix.begin_serial_group()?;
            let started = Instant::now();
            let organ = self.encode_mixer_into(&mut prefix, 0, &hidden)?;
            match organ {
                MixedGpuOrgan::DeltaNet => add_secs(&mut self.stages.deltanet_secs, started),
                _ => add_secs(&mut self.stages.gqa_secs, started),
            }
            let prefix_started = Instant::now();
            self.encode_moe_prefix(&mut prefix, 0, &postnorm)?;
            add_secs(&mut self.stages.moe_shared_secs, prefix_started);
            prefix.end_concurrent_group()?;
            self.commit_device_cb(prefix, organ)?;
            self.host_route_and_bind(0)?
        };

        if fuse {
            for layer in 0..(QWEN80_LAYERS - 1) {
                let next = layer + 1;
                let mut fused = self.new_token_cb()?;
                fused.begin_serial_group()?;
                let suffix_started = Instant::now();
                self.encode_suffix_into(
                    &mut fused,
                    layer,
                    &hidden,
                    &postnorm,
                    &pending_ids,
                    &pending_weights,
                )?;
                add_secs(&mut self.stages.moe_combine_secs, suffix_started);
                add_secs(&mut self.stages.moe_routed_secs, suffix_started);
                let mix_started = Instant::now();
                let organ = self.encode_mixer_into(&mut fused, next, &hidden)?;
                match organ {
                    MixedGpuOrgan::DeltaNet => add_secs(&mut self.stages.deltanet_secs, mix_started),
                    _ => add_secs(&mut self.stages.gqa_secs, mix_started),
                }
                let prefix_started = Instant::now();
                self.encode_moe_prefix(&mut fused, next, &postnorm)?;
                add_secs(&mut self.stages.moe_shared_secs, prefix_started);
                fused.end_concurrent_group()?;
                self.commit_device_cb(fused, organ)?;
                let (ids, weights) = self.host_route_and_bind(next)?;
                pending_ids = ids;
                pending_weights = weights;
            }
            let last = QWEN80_LAYERS - 1;
            let mut last_cb = self.new_token_cb()?;
            last_cb.begin_serial_group()?;
            let suffix_started = Instant::now();
            self.encode_suffix_into(
                &mut last_cb,
                last,
                &hidden,
                &postnorm,
                &pending_ids,
                &pending_weights,
            )?;
            add_secs(&mut self.stages.moe_combine_secs, suffix_started);
            add_secs(&mut self.stages.moe_routed_secs, suffix_started);
            let terminal_started = Instant::now();
            self.encode_terminal_into(&mut last_cb, &hidden)?;
            add_secs(&mut self.stages.terminal_secs, terminal_started);
            last_cb.end_concurrent_group()?;
            self.commit_device_cb(last_cb, MixedGpuOrgan::Terminal)?;
        } else {
            for layer in 0..QWEN80_LAYERS {
                let mut suffix = self.new_token_cb()?;
                suffix.begin_serial_group()?;
                let suffix_started = Instant::now();
                self.encode_suffix_into(
                    &mut suffix,
                    layer,
                    &hidden,
                    &postnorm,
                    &pending_ids,
                    &pending_weights,
                )?;
                add_secs(&mut self.stages.moe_combine_secs, suffix_started);
                add_secs(&mut self.stages.moe_routed_secs, suffix_started);
                suffix.end_concurrent_group()?;
                self.commit_device_cb(suffix, MixedGpuOrgan::MoeRouted)?;
                if layer + 1 < QWEN80_LAYERS {
                    let next = layer + 1;
                    let mut prefix = self.new_token_cb()?;
                    prefix.begin_serial_group()?;
                    let started = Instant::now();
                    let organ = self.encode_mixer_into(&mut prefix, next, &hidden)?;
                    match organ {
                        MixedGpuOrgan::DeltaNet => add_secs(&mut self.stages.deltanet_secs, started),
                        _ => add_secs(&mut self.stages.gqa_secs, started),
                    }
                    let prefix_started = Instant::now();
                    self.encode_moe_prefix(&mut prefix, next, &postnorm)?;
                    add_secs(&mut self.stages.moe_shared_secs, prefix_started);
                    prefix.end_concurrent_group()?;
                    self.commit_device_cb(prefix, organ)?;
                    let (ids, weights) = self.host_route_and_bind(next)?;
                    pending_ids = ids;
                    pending_weights = weights;
                }
            }
            let mut terminal = self.new_token_cb()?;
            terminal.begin_serial_group()?;
            let terminal_started = Instant::now();
            self.encode_terminal_into(&mut terminal, &hidden)?;
            add_secs(&mut self.stages.terminal_secs, terminal_started);
            terminal.end_concurrent_group()?;
            self.commit_device_cb(terminal, MixedGpuOrgan::Terminal)?;
        }

        self.sample_logits()
    }

    pub fn forward_token(&mut self, token: u32) -> Result<u32> {
        if self.state.position >= self.state.max_seq_len {
            return Err(mixed_error(format!(
                "decode position {} exceeds max_seq_len {}",
                self.state.position, self.state.max_seq_len
            )));
        }
        #[cfg(target_os = "macos")]
        if self.device_activations_live() {
            let sampled = self.forward_token_device(token)?;
            self.state.position = self.state.position.saturating_add(1);
            require_rss_cap("after mixed hybrid token")?;
            if self.fallbacks.silent_or_invalid() != 0 {
                return Err(mixed_error(format!(
                    "silent mixed fallbacks are invalid: {:?}",
                    self.fallbacks
                )));
            }
            return Ok(sampled);
        }
        let embed_started = Instant::now();
        let mut hidden = self.embed(token)?;
        add_secs(&mut self.stages.embed_secs, embed_started);
        add_ns(&mut self.stages.host_excl.embed, embed_started);
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

    #[cfg(target_os = "macos")]
    pub fn measure_isolated_named(
        &mut self,
        name: &str,
        mode: MixedProbeMode,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let Some(metal) = self.metal.as_mut() else {
            return Err(mixed_error("Metal required for isolated probe"));
        };
        metal.measure_named(name, mode)
    }

    #[cfg(target_os = "macos")]
    pub fn measure_isolated_class(
        &mut self,
        class: &str,
        mode: MixedProbeMode,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let Some(metal) = self.metal.as_mut() else {
            return Err(mixed_error("Metal required for isolated class"));
        };
        metal.measure_class(class, mode)
    }

    #[cfg(target_os = "macos")]
    pub fn measure_isolated_routed_wave(
        &mut self,
        mode: MixedProbeMode,
    ) -> Result<crate::metal::CommandBufferTiming> {
        let Some(metal) = self.metal.as_mut() else {
            return Err(mixed_error("Metal required for isolated routed wave"));
        };
        metal.measure_one_routed_wave(mode)
    }

    #[cfg(target_os = "macos")]
    pub fn uploaded_weight_names(&self) -> Vec<String> {
        self.metal
            .as_ref()
            .map(|m| m.weights.keys().cloned().collect())
            .unwrap_or_default()
    }

    #[cfg(target_os = "macos")]
    pub fn bound_expert_count(&self) -> usize {
        self.metal.as_ref().map(|m| m.experts.len()).unwrap_or(0)
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
    pub host_expert_bind_ns_per_token: f64,
    pub wait_minus_gpu_ns_per_token: f64,
    pub command_buffers_per_token: f64,
    pub dispatches_per_token: f64,
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
    pub token_samples: Vec<MixedTokenSample>,
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
    let mut token_samples = Vec::new();
    let prefill_started = Instant::now();
    let mut next = 0u32;
    for &token in prompt_token_ids.iter() {
        let before = session.exclusive_snap();
        let wall_t0 = Instant::now();
        next = session.forward_token(token)?;
        let wall_ns = wall_t0.elapsed().as_nanos() as u64;
        let snap = session.exclusive_snap().saturating_sub(before);
        token_samples.push(MixedTokenSample {
            position: session.state.position.saturating_sub(1) as u32,
            kind: "prefill",
            wall_ns,
            snap,
        });
    }
    let prefill_secs = prefill_started.elapsed().as_secs_f64();
    let prefill_bind_ns = session.stages.host_expert_bind_ns;
    let prefill_wait_minus_ns = session.stages.cb_wait_minus_gpu_ns;
    let prefill_cbs = session.native.command_buffers;
    let prefill_dispatches = session.native.compute_dispatches;
    let prefill_gpu_ns = session.stages.gpu_matvec_ns;
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
        let before = session.exclusive_snap();
        let wall_t0 = Instant::now();
        next = session.forward_token(next)?;
        let wall_ns = wall_t0.elapsed().as_nanos() as u64;
        let snap = session.exclusive_snap().saturating_sub(before);
        token_samples.push(MixedTokenSample {
            position: session.state.position.saturating_sub(1) as u32,
            kind: "decode",
            wall_ns,
            snap,
        });
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
    let decode_forwards = wall_tokens;
    let gpu_matvec_ns_per_token =
        session.stages.gpu_matvec_ns.saturating_sub(prefill_gpu_ns) as f64 / decode_forwards;
    let host_expert_bind_ns_per_token = session
        .stages
        .host_expert_bind_ns
        .saturating_sub(prefill_bind_ns) as f64
        / decode_forwards;
    let wait_minus_gpu_ns_per_token = session
        .stages
        .cb_wait_minus_gpu_ns
        .saturating_sub(prefill_wait_minus_ns) as f64
        / decode_forwards;
    let command_buffers_per_token = session.native.command_buffers.saturating_sub(prefill_cbs)
        as f64
        / decode_forwards;
    let dispatches_per_token = session
        .native
        .compute_dispatches
        .saturating_sub(prefill_dispatches) as f64
        / decode_forwards;
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
        host_expert_bind_ns_per_token,
        wait_minus_gpu_ns_per_token,
        command_buffers_per_token,
        dispatches_per_token,
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
        token_samples,
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
